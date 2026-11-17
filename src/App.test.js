import React from "react";
import { render, screen, fireEvent, waitFor, act } from "@testing-library/react";
import App from "./App";
import TrainingPage from "./pages/TrainingPage/TrainingPage";
import ReportPage from "./pages/ReportPage/ReportPage";
import {
  generateSessionId,
  analyzeFinalizedText,
  createSpeechTracker,
  evaluatePoseLandmarks,
  createVisualAccumulator,
} from "./utils/speechMetrics";

// Mock heavy browser CV/chart libraries so tests run fast and deterministically in JSDOM
jest.mock("face-api.js", () => ({
  nets: {
    tinyFaceDetector: { loadFromUri: jest.fn(() => Promise.resolve()) },
    faceExpressionNet: { loadFromUri: jest.fn(() => Promise.resolve()) },
  },
  TinyFaceDetectorOptions: jest.fn(),
  detectSingleFace: jest.fn(() => ({
    withFaceExpressions: () => Promise.resolve(null),
  })),
}));

jest.mock("@mediapipe/pose", () => ({
  Pose: jest.fn().mockImplementation(() => ({
    setOptions: jest.fn(),
    onResults: jest.fn(),
    send: jest.fn(() => Promise.resolve()),
    close: jest.fn(() => Promise.resolve()),
  })),
}));

const mockChartDestroy = jest.fn();
const mockChartConstructor = jest.fn().mockImplementation(() => ({
  destroy: mockChartDestroy,
}));
jest.mock("chart.js/auto", () => ({
  __esModule: true,
  default: function MockChart(...args) {
    return mockChartConstructor(...args);
  },
}));

// ---------------------------------------------------------------------------
// P0-C: Speech recognition deduplication, fillers, WPM, and restart behavior
// ---------------------------------------------------------------------------
describe("speechMetrics pure helpers (P0-C, P0-D, P1-B)", () => {
  test("generateSessionId produces canonical lowercase UUID v4", () => {
    const id1 = generateSessionId();
    const id2 = generateSessionId();
    expect(id1).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
    expect(id1).not.toBe(id2);
  });

  test("three interim variants then one final counts once; repeated final at same index counts once", () => {
    let nowMs = 1000;
    const tracker = createSpeechTracker(() => nowMs);
    tracker.startSession();

    // Three interim updates for result index 0
    tracker.processResultEvent({
      results: [{ 0: { transcript: "hello" }, isFinal: false }],
    });
    tracker.processResultEvent({
      results: [{ 0: { transcript: "hello world" }, isFinal: false }],
    });
    tracker.processResultEvent({
      results: [{ 0: { transcript: "hello world um" }, isFinal: false }],
    });
    nowMs = 16000; // 15s elapsed
    expect(tracker.getSnapshot().wordCount).toBe(0);
    expect(tracker.getSnapshot().fillers).toBe(0);

    // Finalized result at index 0 (4 words, 1 filler "um")
    tracker.processResultEvent({
      results: [{ 0: { transcript: "Hello world, um today." }, isFinal: true }],
    });
    expect(tracker.getSnapshot().wordCount).toBe(4);
    expect(tracker.getSnapshot().fillers).toBe(1);
    // 4 words in 15s (0.25 min) = 16 WPM
    expect(tracker.getSnapshot().wpm).toBe(16);

    // Subsequent event re-delivering finalized index 0 + new finalized index 1
    nowMs = 31000; // 30s elapsed (0.5 min)
    tracker.processResultEvent({
      results: [
        { 0: { transcript: "Hello world, um today." }, isFinal: true },
        { 0: { transcript: "We are ready now" }, isFinal: true },
      ],
    });
    expect(tracker.getSnapshot().wordCount).toBe(8);
    expect(tracker.getSnapshot().wpm).toBe(16);
  });

  test("recognition restart preserves cumulative session totals while resetting instance indices", () => {
    let nowMs = 0;
    const tracker = createSpeechTracker(() => nowMs);
    tracker.startSession();

    nowMs = 30000;
    tracker.processResultEvent({
      results: [{ 0: { transcript: "first sentence here" }, isFinal: true }],
    });
    expect(tracker.getSnapshot().wordCount).toBe(3);

    // Simulate SpeechRecognition ending after pause and starting a fresh instance
    tracker.onInstanceRestart();
    nowMs = 60000;
    // New instance reuses index 0 for new utterance
    tracker.processResultEvent({
      results: [{ 0: { transcript: "second sentence um here" }, isFinal: true }],
    });
    expect(tracker.getSnapshot().wordCount).toBe(7);
    expect(tracker.getSnapshot().fillers).toBe(1);
    expect(tracker.getSnapshot().wpm).toBe(7);
  });

  test("filler detection handles punctuation, multiword phrases, and avoids substring false positives", () => {
    const res = analyzeFinalizedText(
      "Um, I umbrella human, you, know! I mean... kind of sort of basically liked like that."
    );
    // Fillers expected:
    // "um" (1), "you know" (1), "i mean" (1), "kind of" (1), "sort of" (1), "basically" (1), "like" (1)
    // "umbrella", "human", "liked" must NOT match!
    expect(res.fillerCount).toBe(7);
  });

  test("zero-time division is safe and permission denied reports null metrics", () => {
    const nowMs = 5000;
    const tracker = createSpeechTracker(() => nowMs);
    tracker.startSession();
    expect(tracker.getSnapshot().wpm).toBe(0);

    tracker.setStatus("denied");
    const deniedSnap = tracker.getSnapshot();
    expect(deniedSnap.wpm).toBeNull();
    expect(deniedSnap.fillers).toBeNull();
    expect(deniedSnap.wordCount).toBeNull();
  });

  test("evaluatePoseLandmarks checks visibility and visualAccumulator averages across frames", () => {
    // Low-visibility landmarks -> null
    const lowVis = Array.from({ length: 13 }, () => ({ x: 0.5, y: 0.5, visibility: 0.1 }));
    expect(evaluatePoseLandmarks(lowVis, 1280, 720)).toBeNull();

    // Level shoulders + centered nose
    const lm = Array.from({ length: 13 }, () => ({ x: 0.5, y: 0.5, visibility: 0.9 }));
    lm[0] = { x: 0.5, y: 0.3, visibility: 0.9 };
    lm[11] = { x: 0.4, y: 0.6, visibility: 0.9 };
    lm[12] = { x: 0.6, y: 0.6, visibility: 0.9 };
    const obs1 = evaluatePoseLandmarks(lm, 1280, 720);
    expect(obs1).toEqual({ posture: 100, shoulderTilt: 0, eyeContact: "Good" });

    const acc = createVisualAccumulator();
    acc.addPose({ posture: 90, shoulderTilt: -4, eyeContact: "Good" }, 1.0);
    acc.addPose({ posture: 70, shoulderTilt: 4, eyeContact: "Looking Away" }, 6.0);
    acc.addExpression("Happy");
    acc.addExpression("Happy");
    acc.addExpression("Neutral");
    const sum = acc.getSummary();
    expect(sum.posture).toBe(80);
    expect(sum.shoulder_tilt).toBe(0);
    expect(sum.eyeContact).toBe("Good");
    expect(sum.emotion).toBe("Happy");
  });
});

// ---------------------------------------------------------------------------
// P0-B, P0-D, P1-A: TrainingPage WebSocket handling & single-POST state machine
// ---------------------------------------------------------------------------
describe("TrainingPage WebSocket, speech & session finalization (P0-B, P0-D)", () => {
  let mockWsInstances = [];
  let mockSpeechInstances = [];

  class MockWebSocket {
    static OPEN = 1;
    constructor(url) {
      this.url = url;
      this.readyState = MockWebSocket.OPEN;
      this.bufferedAmount = 0;
      this.sent = [];
      mockWsInstances.push(this);
      setTimeout(() => {
        if (this.onopen) this.onopen();
      }, 0);
    }
    send(data) {
      this.sent.push(data);
      if (typeof data === "string") {
        try {
          const parsed = JSON.parse(data);
          if (parsed.type === "start") {
            this.sessionId = parsed.session_id;
          } else if (parsed.type === "finish" && this.onmessage) {
            setTimeout(() => {
              if (this.onmessage) {
                this.onmessage({
                  data: JSON.stringify({
                    type: "finished",
                    session_id: this.sessionId,
                    aggregates: { avg_volume: 3.4, avg_pitch: 182.5, dominant_tone: "Energetic" },
                  }),
                });
              }
            }, 0);
          }
        } catch (_e) {
          // ignore
        }
      }
    }
    close() {
      this.readyState = 3;
      if (this.onclose) this.onclose();
    }
  }

  class MockSpeechRecognition {
    constructor() {
      this.continuous = false;
      this.interimResults = false;
      mockSpeechInstances.push(this);
    }
    start() {}
    stop() {
      if (this.onend) this.onend();
    }
  }

  beforeEach(() => {
    mockWsInstances = [];
    mockSpeechInstances = [];
    global.WebSocket = MockWebSocket;
    window.SpeechRecognition = MockSpeechRecognition;
    global.AudioContext = jest.fn().mockImplementation(() => ({
      sampleRate: 48000,
      state: "running",
      createMediaStreamSource: () => ({ connect: jest.fn(), disconnect: jest.fn() }),
      createScriptProcessor: () => ({ connect: jest.fn(), disconnect: jest.fn() }),
      destination: {},
      close: jest.fn(() => Promise.resolve()),
    }));
    Object.defineProperty(navigator, "mediaDevices", {
      writable: true,
      value: {
        getUserMedia: jest.fn(() =>
          Promise.resolve({
            getTracks: () => [{ stop: jest.fn() }],
          })
        ),
      },
    });
    global.fetch = jest.fn(() =>
      Promise.resolve({
        ok: true,
        status: 200,
        json: () => Promise.resolve({ status: "ok", session_id: "returned-id" }),
      })
    );
  });

  test("valid WS audio_metrics updates volume/pitch/tone; ignores stale session_id; double Stop posts once", async () => {
    const onEndSession = jest.fn();
    render(<TrainingPage onEndSession={onEndSession} onBack={jest.fn()} backendUrl="http://localhost:8010" />);

    const startBtn = screen.getByRole("button", { name: /Start Session/i });
    await act(async () => {
      fireEvent.click(startBtn);
    });

    await waitFor(() => expect(mockWsInstances.length).toBe(1));
    const ws = mockWsInstances[0];

    // Wait for onopen handshake
    await waitFor(() => expect(ws.sent.length).toBeGreaterThanOrEqual(1));
    const startHandshake = JSON.parse(ws.sent[0]);
    expect(startHandshake.type).toBe("start");
    expect(startHandshake.sample_rate).toBe(48000);
    const sid = startHandshake.session_id;

    // Before first valid window, Tone and Pitch show Analyzing…
    expect(screen.getAllByText(/Analyzing…/i).length).toBeGreaterThan(0);

    // Send valid audio_metrics for current session
    act(() => {
      ws.onmessage({
        data: JSON.stringify({
          type: "audio_metrics",
          session_id: sid,
          volume: 3.4,
          pitch: 182.5,
          tone: "Energetic",
        }),
      });
    });

    expect(screen.getByText("3.4")).toBeInTheDocument();
    expect(screen.getByText("182.5 Hz")).toBeInTheDocument();
    expect(screen.getByText("Energetic")).toBeInTheDocument();

    // Send late message with wrong session_id -> must be ignored
    act(() => {
      ws.onmessage({
        data: JSON.stringify({
          type: "audio_metrics",
          session_id: "00000000-0000-4000-8000-000000000000",
          volume: 99.9,
          pitch: 400.0,
          tone: "Calm",
        }),
      });
    });
    expect(screen.getByText("3.4")).toBeInTheDocument();
    expect(screen.getByText("Energetic")).toBeInTheDocument();

    // Malformed JSON -> ignored without crashing
    act(() => {
      ws.onmessage({ data: "{not valid json" });
    });
    expect(screen.getByText("Energetic")).toBeInTheDocument();

    // Double-click End Session -> only ONE POST to /api/session/log
    const endBtn = screen.getByRole("button", { name: /End Session/i });
    await act(async () => {
      fireEvent.click(endBtn);
      fireEvent.click(endBtn);
    });

    await waitFor(() => expect(onEndSession).toHaveBeenCalledTimes(1));
    expect(global.fetch).toHaveBeenCalledTimes(1);
    const postBody = JSON.parse(global.fetch.mock.calls[0][1].body);
    expect(postBody.session_id).toBe(sid);
  });
});

// ---------------------------------------------------------------------------
// P1: ReportPage truthful metrics, no fabricated timeline, 404 & stale fetch guard
// ---------------------------------------------------------------------------
describe("ReportPage truthful statistics & timeline behavior (P1)", () => {
  beforeEach(() => {
    mockChartConstructor.mockClear();
    mockChartDestroy.mockClear();
  });

  test("renders backend report.metrics (not stale App props) and hides trend chart when no timeline exists", async () => {
    global.fetch = jest.fn(() =>
      Promise.resolve({
        ok: true,
        status: 200,
        json: () =>
          Promise.resolve({
            session_id: "session-a",
            report: {
              summary: "You spoke at 142 WPM with 1 filler words.",
              metrics: {
                wpm: 142,
                posture: 91,
                eyeContact: "Good",
                volume: 2.3,
                emotion: "Happy",
              },
              timeline: [],
              insights: {
                posture: "Strong posture",
                eye_contact: "Good alignment",
                emotion: "Happy estimate",
                tone: "Balanced tone",
              },
              recommendations: ["Keep it up."],
              ml_performance_score: 84.5,
            },
          }),
      })
    );

    // Pass stale App `data` prop with completely different numbers (wpm: 10, posture: 20)
    render(
      <ReportPage
        data={{ wpm: 10, posture: 20, eyeContact: "Looking Away", volume: 0.1, emotion: "Sad" }}
        sessionId="session-a"
        onBack={jest.fn()}
        backendUrl="http://localhost:8010"
      />
    );

    await waitFor(() => expect(screen.getByText("142")).toBeInTheDocument());
    expect(screen.getByText("91%")).toBeInTheDocument();
    expect(screen.getByText("No session timeline recorded")).toBeInTheDocument();
    // Chart must NOT be instantiated when timeline is empty
    expect(mockChartConstructor).not.toHaveBeenCalled();
  });

  test("displays 404 error state cleanly when session is missing", async () => {
    global.fetch = jest.fn(() =>
      Promise.resolve({
        ok: false,
        status: 404,
        json: () => Promise.resolve({ error: "Session not found" }),
      })
    );

    render(<ReportPage data={null} sessionId="missing-id" onBack={jest.fn()} backendUrl="http://localhost:8010" />);

    await waitFor(() => expect(screen.getByText(/Session report not found \(404\)/i)).toBeInTheDocument());
  });

  test("App renders HomePage without opening webcam on mount", () => {
    const getUserMediaSpy = jest.fn();
    Object.defineProperty(navigator, "mediaDevices", {
      writable: true,
      value: { getUserMedia: getUserMediaSpy },
    });
    render(<App />);
    expect(screen.getByRole("button", { name: /Start Training/i })).toBeInTheDocument();
    expect(getUserMediaSpy).not.toHaveBeenCalled();
  });
});
