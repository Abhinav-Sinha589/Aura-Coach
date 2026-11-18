import React, { useRef, useEffect, useState } from "react";
import * as faceapi from "face-api.js";
import { Pose, VERSION as POSE_VERSION } from "@mediapipe/pose";
import "./TrainingPage.css";
import bg from "../../assets/background.png";
import { resolveBackendUrl, toWsUrl, newSessionId } from "../../lib/backend";
import { createSpeechTracker, computeWpm, FATAL_SPEECH_ERRORS } from "../../lib/speech";
import {
  headRollDeg, shoulderRollDeg, neckRatio, postureScore, facingMeasure,
  EyeContactEstimator, Ema, RunningMean, median,
} from "../../lib/vision";
import { EmotionSmoother } from "../../lib/emotion";
import { createTipsEngine } from "../../lib/tips";
import { startCapture, floatToPcm16 } from "../../lib/audioCapture";

const POSE_INTERVAL_MS = 66;      // ~15 fps is plenty for posture/head metrics
const POSE_MAX_DUTY = 0.4;        // on slow machines, keep Pose under ~40% of the main thread
const FACE_INTERVAL_MS = 600;     // face-api expression sampling
const UI_TICK_MS = 250;
const TREND_EVERY_MS = 5000;
const MAX_TREND_POINTS = 720;     // 1 h at 5 s
const NECK_CALIB_MS = 3000;
const WS_MAX_BUFFERED = 1_000_000;
const PRE_OPEN_MAX_SEC = 2;

// face-api.js weights are loaded once per page load and reused across sessions.
let faceModelsPromise = null;
function loadFaceModels() {
  if (!faceModelsPromise) {
    faceModelsPromise = Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri("/models"),
      faceapi.nets.faceExpressionNet.loadFromUri("/models"),
    ]).catch((e) => {
      faceModelsPromise = null;
      throw e;
    });
  }
  return faceModelsPromise;
}
const FACE_OPTS = new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.5 });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const round1 = (x) => (x === null || x === undefined ? null : Math.round(x * 10) / 10);

function freshLive() {
  return {
    eye: new EyeContactEstimator(),
    emotion: new EmotionSmoother(),
    headEma: new Ema(0.35, 800),
    postureEma: new Ema(0.2, 800),
    head: new RunningMean(),
    posture: new RunningMean(),
    neckCalib: [],
    neckBaseline: null,
    poseFrames: 0,
    faceDetections: 0,
    eyeState: "Unknown",
    emotionState: "Detecting…",
    audio: {
      status: "connecting", volume: null, pitch: null, tone: null, toneStatus: null,
      voiceActive: false, voiceT: 0, clipHist: [], lastSeq: 0, everReceived: false,
    },
    trend: { t: [], posture: [], wpm: [] },
    trendBucket: new RunningMean(),
    lastTrendAt: 0,
    tips: createTipsEngine(),
    perf: { poseMs: 0, poseFrames: 0, faceMs: 0, faceFrames: 0 },
  };
}

const IDLE_VIEW = {
  posture: null, headTilt: null, eyeContact: null, emotion: null, wpm: null, fillers: null,
  volume: null, pitch: null, tone: null, toneStatus: null, audioStatus: "idle", speechStatus: "idle",
};

export default function TrainingPage({ onEndSession, onBack, backendUrl }) {
  const BACKEND_URL = backendUrl || resolveBackendUrl();
  const WS_URL = toWsUrl(BACKEND_URL) + "/api/audio-stream";

  const videoRef = useRef(null);
  const phaseRef = useRef("idle"); // idle | active | stopping | saving | save_error
  const [phase, setPhaseState] = useState("idle");
  const setPhase = (p) => { phaseRef.current = p; setPhaseState(p); };

  const [cameraError, setCameraError] = useState(null);
  const [visionNote, setVisionNote] = useState(null);
  const [notice, setNotice] = useState(null);
  const [view, setView] = useState(IDLE_VIEW);
  const [tips, setTips] = useState([]);

  const live = useRef(freshLive());
  const sessionRef = useRef(null);
  const audioRef = useRef(null);
  const speechRef = useRef(null);
  const payloadRef = useRef(null);
  const savingRef = useRef(false);
  const faceStatusRef = useRef("loading");

  // ------------------------------------------------------------ vision ----
  function handlePoseResults(res) {
    if (phaseRef.current !== "active" || !sessionRef.current) return;
    const now = performance.now();
    const L = live.current;
    const lm = res.poseLandmarks;
    const video = videoRef.current;
    const w = (res.image && res.image.width) || (video && video.videoWidth) || 1280;
    const h = (res.image && res.image.height) || (video && video.videoHeight) || 720;
    if (!lm) {
      L.eyeState = L.eye.push(null, now);
      return;
    }
    const head = headRollDeg(lm, w, h);
    if (head) {
      L.headEma.push(head.deg, now);
      L.head.push(head.deg);
    }
    const elapsed = Date.now() - sessionRef.current.start;
    const ratio = neckRatio(lm, w, h);
    if (L.neckBaseline === null && ratio !== null) {
      if (elapsed < NECK_CALIB_MS) L.neckCalib.push(ratio);
      else if (L.neckCalib.length >= 10) L.neckBaseline = median(L.neckCalib);
    }
    const p = postureScore(shoulderRollDeg(lm, w, h), ratio, L.neckBaseline);
    if (p !== null) {
      L.postureEma.push(p, now);
      L.posture.push(p);
      L.trendBucket.push(p);
      L.poseFrames += 1;
    }
    L.eyeState = L.eye.push(facingMeasure(lm, w, h), now);
  }

  async function sampleFace(video) {
    const t0 = performance.now();
    const det = await faceapi.detectSingleFace(video, FACE_OPTS).withFaceExpressions();
    live.current.perf.faceMs += performance.now() - t0;
    live.current.perf.faceFrames += 1;
    if (phaseRef.current !== "active") return;
    const L = live.current;
    L.emotionState = L.emotion.push(det ? det.expressions : null, performance.now());
    if (det) L.faceDetections += 1;
  }

  // ------------------------------------------------------------- audio ----
  async function startAudio(sessionId) {
    const a = {
      sessionId, ws: null, capture: null, micStream: null, stopping: false,
      attempts: 0, reconnectTimer: null, finalResolve: null, preOpen: [], preOpenSamples: 0,
      sampleRate: null, dropped: 0,
    };
    audioRef.current = a;
    const L = live.current;

    const sendChunk = (f32) => {
      const ws = a.ws;
      if (ws && ws.readyState === WebSocket.OPEN && a.ready) {
        if (ws.bufferedAmount > WS_MAX_BUFFERED) { a.dropped += f32.length; return; }
        ws.send(floatToPcm16(f32).buffer);
      } else if (!a.stopping && a.sampleRate && a.preOpenSamples < PRE_OPEN_MAX_SEC * a.sampleRate) {
        a.preOpen.push(f32); // keep the first moments of speech while the socket opens
        a.preOpenSamples += f32.length;
      } else {
        a.dropped += f32.length;
      }
    };

    const connect = () => {
      if (a.stopping || audioRef.current !== a) return;
      const ws = new WebSocket(WS_URL);
      ws.binaryType = "arraybuffer";
      a.ws = ws;
      a.ready = false;
      ws.onopen = () => {
        a.attempts = 0;
        ws.send(JSON.stringify({
          type: "start", session_id: sessionId, sample_rate: a.sampleRate,
          channels: 1, encoding: "pcm_s16le",
        }));
      };
      ws.onmessage = (ev) => {
        let msg;
        try { msg = JSON.parse(ev.data); } catch (_) { return; }
        if (msg.session_id && msg.session_id !== sessionId) return; // stale socket
        if (msg.type === "ready") {
          a.ready = true;
          L.audio.status = "listening";
          const queued = a.preOpen;
          a.preOpen = [];
          a.preOpenSamples = 0;
          queued.forEach(sendChunk);
        } else if (msg.type === "metrics") {
          if (phaseRef.current !== "active" || msg.seq <= L.audio.lastSeq) return;
          const au = L.audio;
          au.lastSeq = msg.seq;
          au.everReceived = true;
          au.volume = msg.volume;
          au.pitch = msg.pitch;
          au.tone = msg.tone;
          au.toneStatus = msg.status;
          au.voiceActive = !!msg.voice_active;
          if (msg.voice_active) au.voiceT = performance.now();
          au.clipHist.push(!!msg.clipping);
          if (au.clipHist.length > 4) au.clipHist.shift();
        } else if (msg.type === "final") {
          if (a.finalResolve) a.finalResolve(msg.summary || null);
        } else if (msg.type === "error") {
          console.warn("Audio stream:", msg.error);
        }
      };
      ws.onerror = () => {};
      ws.onclose = () => {
        if (a.finalResolve) a.finalResolve(null);
        if (a.stopping || audioRef.current !== a || phaseRef.current !== "active") return;
        if (a.attempts < 3) {
          L.audio.status = "reconnecting";
          const delay = 1000 * 2 ** a.attempts;
          a.attempts += 1;
          a.reconnectTimer = setTimeout(connect, delay);
        } else {
          L.audio.status = "offline";
        }
      };
    };

    try {
      a.micStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          // Raw signal: browser AGC would flatten the very loudness changes we
          // measure, and the tone model was trained on unprocessed audio.
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });
      if (a.stopping || audioRef.current !== a) {
        a.micStream.getTracks().forEach((t) => t.stop());
        return;
      }
      const capture = await startCapture(a.micStream, sendChunk);
      if (a.stopping || audioRef.current !== a) {
        // session ended while the AudioContext was starting
        await capture.stop();
        a.micStream.getTracks().forEach((t) => t.stop());
        return;
      }
      a.capture = capture;
      a.sampleRate = capture.sampleRate;
      connect();
    } catch (err) {
      console.error("Audio error:", err);
      L.audio.status = "unavailable";
    }
  }

  async function stopAudio(graceful) {
    const a = audioRef.current;
    if (!a) return null;
    a.stopping = true;
    clearTimeout(a.reconnectTimer);
    if (a.capture) await a.capture.stop(); // flushes the last partial chunk
    if (a.micStream) a.micStream.getTracks().forEach((t) => t.stop());
    let summary = null;
    const ws = a.ws;
    if (ws && ws.readyState === WebSocket.OPEN) {
      if (graceful) {
        summary = await Promise.race([
          new Promise((res) => {
            a.finalResolve = res;
            ws.send(JSON.stringify({ type: "stop" }));
          }),
          sleep(4000).then(() => null),
        ]);
      } else {
        ws.send(JSON.stringify({ type: "abort" }));
      }
    }
    try { if (ws) ws.close(); } catch (_) { /* ignore */ }
    audioRef.current = null;
    return summary;
  }

  // ------------------------------------------------------------ speech ----
  function startSpeech() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    const sp = {
      tracker: createSpeechTracker(), status: SR ? "starting" : "unsupported", fatal: !SR,
      stopping: false, recognition: null, restarts: [], endResolve: null, ended: true,
      everStarted: false, serviceErrors: 0,
    };
    speechRef.current = sp;
    if (!SR) return;

    const finishEnd = () => {
      sp.ended = true;
      if (sp.endResolve) sp.endResolve();
    };

    const startInstance = () => {
      const rec = new SR();
      rec.continuous = true;
      rec.interimResults = true;
      rec.lang = "en-US";
      rec.onstart = () => {
        sp.everStarted = true;
        if (sp.status === "starting") sp.status = "listening";
      };
      rec.onresult = (e) => {
        if (speechRef.current !== sp) return;
        // Only isFinal results are counted, once each (see lib/speech.js).
        sp.tracker.ingest(e.results, e.resultIndex, Date.now());
      };
      rec.onerror = (e) => {
        // "no-speech"/"aborted" are normal; anything else means the
        // recogniser itself failed (e.g. "network": service unreachable).
        if (e.error !== "no-speech" && e.error !== "aborted") sp.serviceErrors += 1;
        if (FATAL_SPEECH_ERRORS.has(e.error)) {
          sp.fatal = true;
          sp.status = e.error === "not-allowed" || e.error === "service-not-allowed" ? "denied" : "unavailable";
        }
      };
      rec.onend = () => {
        if (sp.stopping || sp.fatal || speechRef.current !== sp || phaseRef.current !== "active") {
          finishEnd();
          return;
        }
        // Chrome ends recognition after pauses: restart, but never loop on errors.
        const now = Date.now();
        sp.restarts = sp.restarts.filter((t) => now - t < 10000);
        if (sp.restarts.length >= 6) {
          sp.status = "unavailable";
          sp.fatal = true;
          finishEnd();
          return;
        }
        sp.restarts.push(now);
        sp.tracker.newGeneration();
        sp.ended = true;
        setTimeout(() => {
          if (sp.stopping || speechRef.current !== sp || phaseRef.current !== "active") return;
          try { startInstance(); } catch (_) { sp.status = "unavailable"; }
        }, 250);
      };
      sp.recognition = rec;
      sp.ended = false;
      rec.start();
    };

    try {
      startInstance();
    } catch (err) {
      console.error("Speech recognition failed to start:", err);
      sp.status = "unavailable";
      sp.fatal = true;
    }
  }

  async function stopSpeech(graceful) {
    const sp = speechRef.current;
    if (!sp) return;
    sp.stopping = true;
    if (sp.recognition && !sp.ended) {
      // stop() (not abort) lets the engine deliver its last final result.
      await Promise.race([
        new Promise((res) => {
          sp.endResolve = res;
          try { if (graceful) sp.recognition.stop(); else sp.recognition.abort(); } catch (_) { res(); }
        }),
        sleep(1500),
      ]);
    }
  }

  // A recogniser that never produced a result *and* reported service errors
  // didn't measure anything: report "not measured", never a fabricated 0 WPM.
  const speechMeasured = (sp) =>
    !!sp && (sp.tracker.segments > 0 ||
      (sp.everStarted && !sp.fatal && sp.serviceErrors === 0 && sp.status !== "denied"));

  // ------------------------------------------------------- UI / tips tick --
  function tick() {
    const session = sessionRef.current;
    if (phaseRef.current !== "active" || !session) return;
    const now = performance.now();
    const L = live.current;
    const sp = speechRef.current;
    const elapsed = Date.now() - session.start;
    const speechOk = speechMeasured(sp);
    const wpm = speechOk ? computeWpm(sp.tracker.words, elapsed) : null;
    const posture = L.postureEma.get(now);
    const headTilt = L.headEma.get(now);
    const au = L.audio;
    const voiceRecent = au.voiceActive && now - au.voiceT < 1500;

    const m = {
      posture: posture === null ? null : Math.round(posture),
      headTilt: headTilt === null ? null : Math.round(headTilt),
      eyeContact: L.eyeState,
      emotion: faceStatusRef.current === "unavailable" ? "Unavailable" : L.emotionState,
      wpm,
      fillers: speechOk ? sp.tracker.fillers : null,
      recentFillers: speechOk ? sp.tracker.recentFillers(Date.now()) : null,
      volume: au.everReceived ? au.volume : null,
      pitch: au.pitch,
      tone: au.tone,
      toneStatus: au.toneStatus,
      audioStatus: au.status,
      speechStatus: sp ? sp.status : "idle",
      voiceActive: voiceRecent,
      clipping: au.clipHist.filter(Boolean).length >= 2,
    };
    m.hasMeasurement = m.posture !== null || m.wpm !== null || m.volume !== null;

    if (now - L.lastTrendAt >= TREND_EVERY_MS && L.trend.t.length < MAX_TREND_POINTS) {
      if (L.lastTrendAt > 0) {
        L.trend.t.push(Math.round(elapsed / 1000));
        L.trend.posture.push(L.trendBucket.n ? Math.round(L.trendBucket.mean()) : null);
        L.trend.wpm.push(wpm);
      }
      L.trendBucket = new RunningMean();
      L.lastTrendAt = now;
    }

    setTips(L.tips(m, now));
    setView(m);
  }

  // ------------------------------------------------------------ session --
  function buildPayload(session, endedAt, summary) {
    const L = live.current;
    const sp = speechRef.current;
    const elapsed = endedAt - session.start;
    const speech = speechMeasured(sp);
    const eye = L.eye.sessionSummary();
    const postureMean = L.posture.mean();
    const hasTrend = L.trend.t.length >= 2 &&
      [...L.trend.posture, ...L.trend.wpm].some((x) => x !== null && x !== undefined);
    const trend = hasTrend ? L.trend : undefined;
    return {
      session_id: session.id,
      schema_version: 2,
      duration_sec: round1(elapsed / 1000),
      wpm: speech ? computeWpm(sp.tracker.words, elapsed, 3000) : null,
      word_count: speech ? sp.tracker.words : null,
      fillers: speech ? sp.tracker.fillers : null,
      posture: postureMean === null ? null : Math.round(postureMean),
      headTilt: round1(L.head.mean()),
      head_tilt_abs_mean: round1(L.head.absMean()),
      eyeContact: eye.label,
      eye_contact_ratio: eye.ratio,
      emotion: L.emotion.dominant(),
      tone: summary ? summary.dominant_tone : null,
      volume: summary ? summary.avg_volume : null,
      pitch: summary ? summary.avg_pitch : null,
      measured: {
        speech,
        audio: !!(summary && summary.audio_summary && summary.audio_summary.analysis_hops > 0) ||
          L.audio.everReceived,
        pose: L.poseFrames > 0,
        face: L.faceDetections > 0,
      },
      trend,
    };
  }

  async function save(payload) {
    if (savingRef.current) return;
    savingRef.current = true;
    setPhase("saving");
    try {
      const response = await fetch(`${BACKEND_URL}/api/session/log`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const result = await response.json();
      payloadRef.current = null;
      onEndSession(result.session_id || payload.session_id);
    } catch (err) {
      console.error("Error saving session:", err);
      setNotice(`Couldn't save this session (${err.message}). Your results are kept here — retry when the backend is reachable.`);
      setPhase("save_error");
    } finally {
      savingRef.current = false;
    }
  }

  async function startSession() {
    if (phaseRef.current !== "idle") return;
    const id = newSessionId();
    sessionRef.current = { id, start: Date.now() };
    live.current = freshLive();
    setNotice(null);
    setTips([]);
    setPhase("active");
    startSpeech();
    startAudio(id);
  }

  async function endSession() {
    if (phaseRef.current !== "active") return;
    setPhase("stopping");
    const session = sessionRef.current;
    const endedAt = Date.now();
    const dropped = audioRef.current ? audioRef.current.dropped : 0;
    const [, summary] = await Promise.all([stopSpeech(true), stopAudio(true)]);
    const payload = buildPayload(session, endedAt, summary);
    const pf = live.current.perf;
    const secs = Math.max(1, (endedAt - session.start) / 1000);
    console.info(
      `[Aura] session ${payload.duration_sec}s · pose ${(pf.poseFrames / secs).toFixed(1)} fps ` +
      `(${pf.poseFrames ? (pf.poseMs / pf.poseFrames).toFixed(0) : "-"} ms/frame) · face-api ` +
      `${pf.faceFrames ? (pf.faceMs / pf.faceFrames).toFixed(0) : "-"} ms/run · audio sent ` +
      `${summary && summary.audio_summary ? summary.audio_summary.audio_seconds : "?"}s · client-dropped ` +
      `${dropped} samples`
    );
    payloadRef.current = payload;
    await save(payload);
  }

  function abortSession() {
    if (phaseRef.current === "active" || phaseRef.current === "stopping") {
      phaseRef.current = "idle";
      stopSpeech(false);
      stopAudio(false);
    }
  }

  // Latest handlers for long-lived callbacks (avoids stale closures).
  const handlers = useRef({});
  handlers.current = { handlePoseResults, sampleFace, tick, abortSession };

  // Camera + MediaPipe Pose + face-api.js: created once per page mount.
  useEffect(() => {
    let cancelled = false;
    let stream = null;
    let pose = null;
    let poseTimer = null;
    let faceTimer = null;
    let poseBusy = false;
    let faceBusy = false;
    let faceReady = false;
    const video = videoRef.current;

    (async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
          audio: false,
        });
      } catch (e) {
        console.error(e);
        if (!cancelled) setCameraError("Unable to access the camera. Body-language metrics are unavailable.");
        return;
      }
      if (cancelled || !video) {
        stream.getTracks().forEach((t) => t.stop());
        return;
      }
      video.srcObject = stream;
      try { await video.play(); } catch (_) { /* autoplay attribute covers this */ }

      try {
        pose = new Pose({
          locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose@${POSE_VERSION}/${f}`,
        });
        pose.setOptions({
          modelComplexity: 1,
          smoothLandmarks: true,
          enableSegmentation: false,
          minDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });
        pose.onResults((res) => handlers.current.handlePoseResults(res));
        await pose.initialize();
      } catch (e) {
        console.error("Pose failed to load", e);
        if (!cancelled) setVisionNote("Pose model failed to load — posture, head tilt and eye contact are unavailable.");
        pose = null;
      }
      if (cancelled) {
        if (pose) pose.close().catch(() => {});
        return;
      }

      loadFaceModels()
        .then(() => { faceReady = true; faceStatusRef.current = "ready"; })
        .catch((e) => {
          console.error("face-api models failed to load", e);
          faceStatusRef.current = "unavailable";
        });

      // Adaptive rate: if a Pose frame costs more than expected (slow GPU /
      // software WebGL) wait proportionally longer, so audio capture, speech
      // and the UI never starve.
      const poseLoop = async () => {
        if (cancelled) return;
        let wait = POSE_INTERVAL_MS;
        if (pose && phaseRef.current === "active" && video.readyState >= 2 && !poseBusy) {
          poseBusy = true;
          const t0 = performance.now();
          try { await pose.send({ image: video }); } catch (e) { console.warn("pose.send", e); }
          const cost = performance.now() - t0;
          live.current.perf.poseMs += cost;
          live.current.perf.poseFrames += 1;
          wait = Math.max(POSE_INTERVAL_MS, (cost * (1 - POSE_MAX_DUTY)) / POSE_MAX_DUTY);
          poseBusy = false;
        }
        if (!cancelled) poseTimer = setTimeout(poseLoop, wait);
      };
      poseLoop();

      faceTimer = setInterval(async () => {
        if (cancelled || !faceReady || faceBusy || phaseRef.current !== "active" || video.readyState < 2) return;
        faceBusy = true;
        try { await handlers.current.sampleFace(video); } catch (e) { console.warn("face-api", e); }
        faceBusy = false;
      }, FACE_INTERVAL_MS);
    })();

    return () => {
      cancelled = true;
      clearTimeout(poseTimer);
      clearInterval(faceTimer);
      if (pose) pose.close().catch(() => {});
      if (stream) stream.getTracks().forEach((t) => t.stop());
      if (video) video.srcObject = null;
      handlers.current.abortSession();
    };
  }, []);

  useEffect(() => {
    if (phase !== "active") return undefined;
    const id = setInterval(() => handlers.current.tick(), UI_TICK_MS);
    return () => clearInterval(id);
  }, [phase]);

  // -------------------------------------------------------------- render --
  const isActive = phase === "active";
  const dash = "—";
  const v = view;
  const toneText = (() => {
    if (!isActive && phase === "idle") return dash;
    if (v.audioStatus === "unavailable") return "Unavailable";
    if (v.audioStatus === "offline") return "Offline";
    if (v.tone) return v.tone;
    if (v.toneStatus === "warming_up" || v.audioStatus === "connecting") return "Analyzing…";
    if (v.toneStatus === "model_unavailable") return "Unavailable";
    return "Listening…";
  })();
  const speechNote =
    v.speechStatus === "unsupported" ? "Speech recognition isn't supported in this browser"
      : v.speechStatus === "denied" ? "Microphone permission denied for speech recognition"
        : v.speechStatus === "unavailable" ? "Speech recognition unavailable (service unreachable)" : null;
  const audioNote =
    v.audioStatus === "unavailable" ? "Microphone unavailable"
      : v.audioStatus === "reconnecting" ? "Reconnecting audio…"
        : v.audioStatus === "offline" ? "Audio server unreachable" : null;
  const noSpeech = v.speechStatus === "unsupported" || v.speechStatus === "denied";

  const buttonLabel = {
    idle: "▶ Start Session",
    active: "⏹ End Session & View Report",
    stopping: "Finishing analysis…",
    saving: "Saving…",
    save_error: "↻ Retry Save",
  }[phase];
  const onButton = () => {
    if (phase === "idle") startSession();
    else if (phase === "active") endSession();
    else if (phase === "save_error" && payloadRef.current) save(payloadRef.current);
  };

  return (
    <div className="training-root" style={{ backgroundImage: `url(${bg})` }}>
      <header className="topbar glass">
        <div className="brand">Aura Coach</div>
        <button className="home-link" onClick={onBack}>← Home</button>
      </header>

      <div className="grid">
        {/* Left */}
        <section className="card glass card-left">
          <h2 className="card-title">Body Language Analysis</h2>
          <div className="stat"><div className="stat-label">Posture Score</div><div className="stat-value accent">{v.posture === null ? dash : `${v.posture}%`}</div></div>
          <div className="stat" title="Roll of your head from the eye line (MediaPipe Pose). + = tilted toward your right shoulder. Independent of shoulder slope.">
            <div className="stat-label">Head Tilt ⓘ</div>
            <div className="stat-value">{v.headTilt === null ? dash : `${v.headTilt}°`}</div>
            {isActive && v.headTilt === null && <div className="stat-note">Head not detected</div>}
          </div>
          <div className="stat" title="Estimated camera-facing gaze from head orientation; not a medical eye-tracker. Eyes-only glances aren't detected.">
            <div className="stat-label">Eye Contact ⓘ</div>
            <div className="stat-value">{v.eyeContact || dash}</div>
            {v.eyeContact === "Calibrating…" && <div className="stat-note">Look at the camera</div>}
          </div>
          {visionNote && <div className="stat-note">{visionNote}</div>}
        </section>

        {/* Center */}
        <section className="card glass card-center">
          <div className="video-frame">
            {cameraError ? <div className="error">{cameraError}</div> : <video ref={videoRef} autoPlay playsInline muted className="video" />}
          </div>

          {/* Live tips */}
          <div className="live-tips-card">
            <h3>🔮 Live Tips</h3>
            <div className="tips-list">
              {isActive ? tips.map((t) => <p key={t}>{t}</p>)
                : <p>{phase === "idle" ? "Start a session to get live tips." : "Session ended."}</p>}
            </div>
          </div>

          {notice && <div className="stat-note" role="alert">{notice}</div>}

          <div className="actions">
            <button
              className={`btn ${isActive ? "danger" : "start"}`}
              onClick={onButton}
              disabled={phase === "stopping" || phase === "saving"}
            >
              {buttonLabel}
            </button>
          </div>
        </section>

        {/* Right */}
        <section className="card glass card-right">
          <h2 className="card-title">Speaking Style Feedback</h2>
          <div className="stat" title="Finalised recognised words ÷ elapsed session time (pauses included).">
            <div className="stat-label">Speaking Pace</div>
            <div className="stat-value accent">{noSpeech ? "N/A" : v.wpm === null ? dash : `${v.wpm} WPM`}</div>
            {speechNote && <div className="stat-note">{speechNote}</div>}
          </div>
          <div className="stat"><div className="stat-label">Filler Words</div><div className="stat-value">{noSpeech ? "N/A" : v.fillers === null ? dash : v.fillers}</div></div>
          <div className="stat" title="Facial expression from face-api.js, smoothed over ~3 s. An expression, not a measure of how you feel.">
            <div className="stat-label">Emotion</div>
            <div className="stat-value">{v.emotion || dash}</div>
          </div>
          <div className="stat"><div className="stat-label">Tone</div><div className="stat-value">{toneText}</div>{audioNote && <div className="stat-note">{audioNote}</div>}</div>
          <div className="stat" title="Mic signal RMS × 100 (uncalibrated; depends on mic and distance).">
            <div className="stat-label">Volume</div>
            <div className="stat-value">{v.volume === null || v.volume === undefined ? dash : Number(v.volume).toFixed(1)}</div>
          </div>
          <div className="stat" title="Median pitch of voiced speech over the last ~0.75 s.">
            <div className="stat-label">Pitch</div>
            <div className="stat-value">{v.pitch === null || v.pitch === undefined ? dash : `${Math.round(v.pitch)} Hz`}</div>
          </div>
        </section>
      </div>
    </div>
  );
}
