import React, { useRef, useEffect, useState, useCallback } from "react";
import * as faceapi from "face-api.js";
import { Pose } from "@mediapipe/pose";
import "./TrainingPage.css";
import bg from "../../assets/background.png";
import {
  generateSessionId,
  createSpeechTracker,
  evaluatePoseLandmarks,
  createVisualAccumulator,
} from "../../utils/speechMetrics";

const INITIAL_METRICS = {
  posture: null,
  headTilt: null,
  eyeContact: null,
  emotion: null,
  wpm: null,
  fillers: 0,
  volume: null,
  pitch: null,
  tone: null,
  audioStatus: "idle", // idle | connecting | analyzing | active | unavailable
  speechStatus: "idle", // idle | active | unsupported | denied | error
};

const WS_MAX_BUFFERED_BYTES = 65536;
const FATAL_SPEECH_ERRORS = new Set(["not-allowed", "service-not-allowed", "language-not-supported"]);

export default function TrainingPage({ onEndSession, onBack, streamRef, backendUrl }) {
  const videoRef = useRef(null);
  const recognitionRef = useRef(null);
  const wsRef = useRef(null);
  const audioCtxRef = useRef(null);
  const sourceNodeRef = useRef(null);
  const processorRef = useRef(null);
  const audioStreamRef = useRef(null);
  const cameraStreamRef = useRef(null);
  const poseRef = useRef(null);
  const mountedRef = useRef(true);

  // Lifecycle state machine: idle -> starting -> active -> stopping -> saving -> saved | failed
  const phaseRef = useRef("idle");
  const [phase, setPhase] = useState("idle");

  const activeSessionIdRef = useRef(null);
  const sessionStartPerfRef = useRef(null);
  const speechTrackerRef = useRef(null);
  const visualAccRef = useRef(createVisualAccumulator());
  const latestMetricsRef = useRef(INITIAL_METRICS);
  const wsFinishResolverRef = useRef(null);
  const speechEndResolverRef = useRef(null);
  const speechRestartTimerRef = useRef(null);

  const BACKEND_URL =
    backendUrl ||
    process.env.REACT_APP_BACKEND_URL ||
    (typeof window !== "undefined" &&
    window.location.hostname !== "localhost" &&
    window.location.hostname !== "127.0.0.1"
      ? window.location.origin
      : "http://127.0.0.1:8010");

  const WS_URL =
    BACKEND_URL.replace(/^http(s)?:\/\//i, (_match, isHttps) => (isHttps ? "wss://" : "ws://")) +
    "/api/audio-stream";

  const [metrics, setMetrics] = useState(INITIAL_METRICS);
  const [error, setError] = useState(null);

  const updateMetrics = useCallback((updater) => {
    setMetrics((prev) => {
      const next = typeof updater === "function" ? updater(prev) : { ...prev, ...updater };
      latestMetricsRef.current = next;
      return next;
    });
  }, []);

  const setLifecyclePhase = useCallback((nextPhase) => {
    phaseRef.current = nextPhase;
    setPhase(nextPhase);
  }, []);

  // Load camera + Pose + face-api models ONCE on mount; run inference only while active
  useEffect(() => {
    mountedRef.current = true;
    const videoEl = videoRef.current;
    let poseTimer = null;
    let emotionTimer = null;
    let poseBusy = false;
    let faceBusy = false;
    let faceModelsLoaded = false;
    let poseInstance = null;

    const setupVisualPipeline = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        });
        if (!mountedRef.current) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        cameraStreamRef.current = stream;
        if (streamRef) streamRef.current = stream;
        if (videoEl) {
          videoEl.srcObject = stream;
        }
      } catch (camErr) {
        if (mountedRef.current) {
          setError("Camera access unavailable — visual metrics will be marked Unavailable.");
        }
      }

      try {
        await Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri("/models"),
          faceapi.nets.faceExpressionNet.loadFromUri("/models"),
        ]);
        faceModelsLoaded = true;
      } catch (modelErr) {
        // Face expression models failed to load; degrade gracefully
        faceModelsLoaded = false;
      }

      try {
        poseInstance = new Pose({
          locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${f}`,
        });
        poseInstance.setOptions({
          modelComplexity: 1,
          smoothLandmarks: true,
          enableSegmentation: false,
          minDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });
        poseInstance.onResults((res) => {
          if (!mountedRef.current || phaseRef.current !== "active") return;
          if (!res || !res.poseLandmarks) return;
          const vw = videoEl?.videoWidth || 1280;
          const vh = videoEl?.videoHeight || 720;
          const obs = evaluatePoseLandmarks(res.poseLandmarks, vw, vh);
          if (!obs) return;
          const elapsedSec =
            sessionStartPerfRef.current !== null
              ? (performance.now() - sessionStartPerfRef.current) / 1000
              : 0;
          visualAccRef.current.addPose(obs, elapsedSec);
          const snap = speechTrackerRef.current ? speechTrackerRef.current.getSnapshot() : null;
          visualAccRef.current.recordCheckpoint(
            elapsedSec,
            obs.posture,
            snap ? snap.wpm : latestMetricsRef.current.wpm
          );
          updateMetrics((prev) => ({
            ...prev,
            posture: obs.posture !== null ? obs.posture : prev.posture,
            headTilt: obs.shoulderTilt !== null ? obs.shoulderTilt : prev.headTilt,
            eyeContact: obs.eyeContact !== null ? obs.eyeContact : prev.eyeContact,
          }));
        });
        poseRef.current = poseInstance;
      } catch (poseInitErr) {
        poseInstance = null;
      }

      poseTimer = setInterval(async () => {
        if (
          !mountedRef.current ||
          phaseRef.current !== "active" ||
          poseBusy ||
          !poseInstance ||
          !videoEl ||
          videoEl.readyState < 2
        ) {
          return;
        }
        poseBusy = true;
        try {
          await poseInstance.send({ image: videoEl });
        } catch (_e) {
          // Ignore transient frame errors
        } finally {
          poseBusy = false;
        }
      }, 250);

      emotionTimer = setInterval(async () => {
        if (
          !mountedRef.current ||
          phaseRef.current !== "active" ||
          faceBusy ||
          !faceModelsLoaded ||
          !videoEl ||
          videoEl.readyState < 2
        ) {
          return;
        }
        faceBusy = true;
        try {
          const detections = await faceapi
            .detectSingleFace(videoEl, new faceapi.TinyFaceDetectorOptions())
            .withFaceExpressions();
          if (detections && detections.expressions && mountedRef.current && phaseRef.current === "active") {
            const sorted = Object.entries(detections.expressions).sort((a, b) => b[1] - a[1]);
            if (sorted.length > 0 && sorted[0][1] >= 0.25) {
              const raw = sorted[0][0];
              const formatted = raw.charAt(0).toUpperCase() + raw.slice(1);
              visualAccRef.current.addExpression(formatted);
              updateMetrics((prev) => ({ ...prev, emotion: formatted }));
            }
          }
        } catch (_e) {
          // Ignore transient frame errors
        } finally {
          faceBusy = false;
        }
      }, 1200);
    };

    setupVisualPipeline();

    return () => {
      mountedRef.current = false;
      if (poseTimer) clearInterval(poseTimer);
      if (emotionTimer) clearInterval(emotionTimer);
      if (speechRestartTimerRef.current) clearTimeout(speechRestartTimerRef.current);
      if (poseRef.current && typeof poseRef.current.close === "function") {
        Promise.resolve(poseRef.current.close()).catch(() => {});
        poseRef.current = null;
      }
      if (cameraStreamRef.current) {
        cameraStreamRef.current.getTracks().forEach((t) => t.stop());
        cameraStreamRef.current = null;
      }
      if (streamRef?.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      }
      if (videoEl) {
        videoEl.srcObject = null;
      }
    };
  }, [streamRef, updateMetrics]);

  // Audio WebSocket stream
  const startAudioStreaming = async (sid) => {
    try {
      updateMetrics((prev) => ({ ...prev, audioStatus: "connecting" }));
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      if (!mountedRef.current || activeSessionIdRef.current !== sid) {
        stream.getTracks().forEach((t) => t.stop());
        return;
      }
      audioStreamRef.current = stream;

      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      const audioCtx = new AudioCtx();
      audioCtxRef.current = audioCtx;
      // Read the ACTUAL hardware/browser sample rate (typically 44100 or 48000 Hz)
      const actualSampleRate = Math.round(audioCtx.sampleRate || 48000);

      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;
      let streamReady = false;

      ws.onopen = () => {
        if (!mountedRef.current || activeSessionIdRef.current !== sid) {
          ws.close();
          return;
        }
        ws.send(
          JSON.stringify({
            type: "start",
            session_id: sid,
            sample_rate: actualSampleRate,
            channels: 1,
            encoding: "pcm_s16le",
          })
        );
        updateMetrics((prev) => ({ ...prev, audioStatus: "analyzing" }));
      };

      ws.onmessage = (event) => {
        if (!mountedRef.current || typeof event.data !== "string") return;
        let msg;
        try {
          msg = JSON.parse(event.data);
        } catch (_e) {
          return;
        }
        if (!msg || typeof msg !== "object") return;
        if (msg.session_id && msg.session_id !== activeSessionIdRef.current) {
          return; // Ignore late messages from any previous session
        }

        if (msg.type === "ready") {
          streamReady = true;
          updateMetrics((prev) => ({ ...prev, audioStatus: "analyzing" }));
          return;
        }

        if (msg.type === "audio_metrics" && phaseRef.current === "active") {
          updateMetrics((prev) => ({
            ...prev,
            volume: typeof msg.volume === "number" && Number.isFinite(msg.volume) ? msg.volume : prev.volume,
            pitch: typeof msg.pitch === "number" && Number.isFinite(msg.pitch) ? msg.pitch : null,
            tone: typeof msg.tone === "string" && msg.tone ? msg.tone : null,
            audioStatus: "active",
          }));
          return;
        }

        if (msg.type === "finished") {
          const agg = msg.aggregates || {};
          updateMetrics((prev) => ({
            ...prev,
            volume: typeof agg.avg_volume === "number" ? agg.avg_volume : prev.volume,
            pitch: typeof agg.avg_pitch === "number" ? agg.avg_pitch : prev.pitch,
            tone: typeof agg.dominant_tone === "string" ? agg.dominant_tone : prev.tone,
          }));
          if (wsFinishResolverRef.current) {
            const resolve = wsFinishResolverRef.current;
            wsFinishResolverRef.current = null;
            resolve(agg);
          }
          return;
        }

        if (msg.type === "error") {
          updateMetrics((prev) => ({ ...prev, audioStatus: "unavailable" }));
        }
      };

      ws.onerror = () => {
        if (!mountedRef.current || activeSessionIdRef.current !== sid) return;
        updateMetrics((prev) => ({ ...prev, audioStatus: "unavailable" }));
      };

      ws.onclose = () => {
        if (wsFinishResolverRef.current) {
          const resolve = wsFinishResolverRef.current;
          wsFinishResolverRef.current = null;
          resolve(null);
        }
        if (mountedRef.current && phaseRef.current === "active" && activeSessionIdRef.current === sid) {
          updateMetrics((prev) => ({
            ...prev,
            audioStatus: prev.audioStatus === "active" ? prev.audioStatus : "unavailable",
          }));
        }
      };

      const source = audioCtx.createMediaStreamSource(stream);
      sourceNodeRef.current = source;
      const processor = audioCtx.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (event) => {
        if (
          phaseRef.current !== "active" ||
          activeSessionIdRef.current !== sid ||
          !streamReady ||
          ws.readyState !== WebSocket.OPEN
        ) {
          return;
        }
        if (typeof ws.bufferedAmount === "number" && ws.bufferedAmount > WS_MAX_BUFFERED_BYTES) {
          return; // Backpressure guard: drop frame instead of growing an unbounded queue
        }
        const input = event.inputBuffer.getChannelData(0);
        const pcm16 = new Int16Array(input.length);
        for (let i = 0; i < input.length; i += 1) {
          const s = Math.max(-1, Math.min(1, input[i]));
          pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }
        ws.send(pcm16.buffer);
      };

      source.connect(processor);
      processor.connect(audioCtx.destination);
    } catch (_err) {
      if (mountedRef.current) {
        updateMetrics((prev) => ({ ...prev, audioStatus: "unavailable" }));
        setError((prevErr) => prevErr || "Microphone access unavailable — audio metrics marked Unavailable.");
      }
    }
  };

  const finalizeAndStopAudioStreaming = async () => {
    // Stop mic capture and audio nodes first so no new frames enter the socket
    try {
      processorRef.current?.disconnect();
      sourceNodeRef.current?.disconnect();
      audioStreamRef.current?.getTracks().forEach((t) => t.stop());
      if (audioCtxRef.current && audioCtxRef.current.state !== "closed") {
        await Promise.resolve(audioCtxRef.current.close()).catch(() => {});
      }
    } catch (_e) {
      // Ignore cleanup errors
    } finally {
      processorRef.current = null;
      sourceNodeRef.current = null;
      audioStreamRef.current = null;
      audioCtxRef.current = null;
    }

    // Ordered WebSocket flush: send {type:'finish'} and await {type:'finished'} ACK
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      await new Promise((resolve) => {
        const timer = setTimeout(() => {
          wsFinishResolverRef.current = null;
          resolve(null);
        }, 1500);
        wsFinishResolverRef.current = (agg) => {
          clearTimeout(timer);
          resolve(agg);
        };
        try {
          ws.send(JSON.stringify({ type: "finish" }));
        } catch (_e) {
          clearTimeout(timer);
          wsFinishResolverRef.current = null;
          resolve(null);
        }
      });
    }
    try {
      wsRef.current?.close();
    } catch (_e) {
      // Ignore
    }
    wsRef.current = null;
  };

  // Speech recognition (browser) — counts finalized words only
  const startSpeechRecognition = (sid) => {
    const tracker = createSpeechTracker();
    speechTrackerRef.current = tracker;

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      tracker.setStatus("unsupported");
      updateMetrics((prev) => ({ ...prev, speechStatus: "unsupported", wpm: null, fillers: null }));
      return;
    }

    tracker.startSession();
    updateMetrics((prev) => ({ ...prev, speechStatus: "active", wpm: 0, fillers: 0 }));

    let consecutiveErrors = 0;

    const launchInstance = () => {
      if (!mountedRef.current || phaseRef.current !== "active" || activeSessionIdRef.current !== sid) {
        return;
      }
      tracker.onInstanceRestart();
      const recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = "en-US";

      recognition.onresult = (event) => {
        if (!mountedRef.current || activeSessionIdRef.current !== sid) return;
        consecutiveErrors = 0;
        const snap = tracker.processResultEvent(event);
        updateMetrics((prev) => ({
          ...prev,
          wpm: snap.wpm,
          fillers: snap.fillers,
          speechStatus: "active",
        }));
      };

      recognition.onerror = (event) => {
        const errCode = event?.error || "unknown";
        if (FATAL_SPEECH_ERRORS.has(errCode)) {
          tracker.setStatus(errCode === "not-allowed" || errCode === "service-not-allowed" ? "denied" : "unsupported");
          const snap = tracker.getSnapshot();
          updateMetrics((prev) => ({
            ...prev,
            speechStatus: snap.status,
            wpm: snap.wpm,
            fillers: snap.fillers,
          }));
          return;
        }
        consecutiveErrors += 1;
        if (consecutiveErrors >= 5) {
          tracker.setStatus("error");
          const snap = tracker.getSnapshot();
          updateMetrics((prev) => ({
            ...prev,
            speechStatus: "error",
            wpm: snap.wpm,
            fillers: snap.fillers,
          }));
        }
      };

      recognition.onend = () => {
        if (speechEndResolverRef.current) {
          const resolve = speechEndResolverRef.current;
          speechEndResolverRef.current = null;
          resolve();
        }
        const currentStatus = tracker.getSnapshot().status;
        if (
          mountedRef.current &&
          phaseRef.current === "active" &&
          activeSessionIdRef.current === sid &&
          currentStatus === "active" &&
          consecutiveErrors < 5
        ) {
          const delayMs = Math.min(1000, 200 * Math.pow(2, consecutiveErrors));
          speechRestartTimerRef.current = setTimeout(() => {
            launchInstance();
          }, delayMs);
        }
      };

      recognitionRef.current = recognition;
      try {
        recognition.start();
      } catch (_e) {
        // Ignore duplicate start errors
      }
    };

    launchInstance();
  };

  const stopSpeechRecognitionAndFlush = async () => {
    if (speechRestartTimerRef.current) {
      clearTimeout(speechRestartTimerRef.current);
      speechRestartTimerRef.current = null;
    }
    const rec = recognitionRef.current;
    recognitionRef.current = null;
    if (!rec) return;

    await new Promise((resolve) => {
      const timer = setTimeout(() => {
        speechEndResolverRef.current = null;
        resolve();
      }, 600);
      speechEndResolverRef.current = () => {
        clearTimeout(timer);
        resolve();
      };
      try {
        rec.stop();
      } catch (_e) {
        clearTimeout(timer);
        speechEndResolverRef.current = null;
        resolve();
      }
    });
  };

  // Start / End session with explicit state-machine guard against double-clicks
  const handleSessionToggle = async () => {
    if (phaseRef.current === "starting" || phaseRef.current === "stopping" || phaseRef.current === "saving") {
      return;
    }

    if (phaseRef.current === "active") {
      setLifecyclePhase("stopping");
      const sid = activeSessionIdRef.current;

      await Promise.all([stopSpeechRecognitionAndFlush(), finalizeAndStopAudioStreaming()]);

      setLifecyclePhase("saving");
      const speechSnap = speechTrackerRef.current
        ? speechTrackerRef.current.getSnapshot()
        : { status: "unsupported", wordCount: null, fillers: null, wpm: null, captureSeconds: 0 };
      const visualSnap = visualAccRef.current.getSummary();
      const latest = latestMetricsRef.current;

      // Record one final timeline point if we have valid data
      if (speechSnap.captureSeconds >= 1) {
        visualAccRef.current.recordCheckpoint(
          speechSnap.captureSeconds,
          visualSnap.posture,
          speechSnap.wpm
        );
      }
      const finalVisual = visualAccRef.current.getSummary();

      const payload = {
        session_id: sid,
        wpm: speechSnap.wpm,
        wpm_source: speechSnap.wpm !== null ? "speech_recognition" : "unavailable",
        word_count: speechSnap.wordCount,
        capture_seconds: speechSnap.captureSeconds,
        speech_status: speechSnap.status,
        fillers: speechSnap.fillers,
        posture: finalVisual.posture,
        posture_samples: finalVisual.posture_samples,
        shoulder_tilt: finalVisual.shoulder_tilt,
        eyeContact: finalVisual.eyeContact,
        eye_contact_ratio: finalVisual.eye_contact_ratio,
        camera_samples: finalVisual.camera_samples,
        emotion: finalVisual.emotion,
        emotion_counts: finalVisual.emotion_counts,
        audio_status: latest.audioStatus,
        timeline: finalVisual.timeline,
      };

      try {
        const response = await fetch(`${BACKEND_URL}/api/session/log`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const result = await response.json();
        const savedId = result?.session_id || sid;
        setLifecyclePhase("saved");
        onEndSession({
          ...payload,
          volume: latest.volume,
          pitch: latest.pitch,
          tone: latest.tone,
          session_id: savedId,
        });
      } catch (err) {
        setLifecyclePhase("failed");
        setError("Failed to save session to backend — showing local summary.");
        onEndSession({
          ...payload,
          volume: latest.volume,
          pitch: latest.pitch,
          tone: latest.tone,
          session_id: sid,
        });
      }
    } else {
      setLifecyclePhase("starting");
      setError(null);
      const newSessionId = generateSessionId();
      activeSessionIdRef.current = newSessionId;
      sessionStartPerfRef.current = performance.now();
      visualAccRef.current = createVisualAccumulator();
      updateMetrics(INITIAL_METRICS);

      setLifecyclePhase("active");
      startSpeechRecognition(newSessionId);
      await startAudioStreaming(newSessionId);
    }
  };

  const handleBackClick = async () => {
    activeSessionIdRef.current = null;
    setLifecyclePhase("idle");
    await Promise.all([stopSpeechRecognitionAndFlush(), finalizeAndStopAudioStreaming()]);
    onBack();
  };

  const isSessionActive = phase === "active";
  const isBusy = phase === "starting" || phase === "stopping" || phase === "saving";

  const formatWpm = () => {
    if (!isSessionActive && metrics.speechStatus === "idle") return "—";
    if (metrics.speechStatus === "unsupported" || metrics.speechStatus === "denied") return "Unavailable";
    if (metrics.wpm === null || metrics.wpm === undefined) return "Analyzing…";
    return `${metrics.wpm} WPM`;
  };

  const formatAudioTone = () => {
    if (!isSessionActive && metrics.audioStatus === "idle") return "—";
    if (metrics.audioStatus === "unavailable") return "Unavailable";
    if (metrics.audioStatus === "connecting" || metrics.audioStatus === "analyzing") return "Analyzing…";
    return metrics.tone || "Unavailable (silence)";
  };

  const formatAudioPitch = () => {
    if (!isSessionActive && metrics.audioStatus === "idle") return "—";
    if (metrics.audioStatus === "unavailable") return "Unavailable";
    if (metrics.audioStatus === "connecting" || metrics.audioStatus === "analyzing") return "Analyzing…";
    return typeof metrics.pitch === "number" ? `${metrics.pitch.toFixed(1)} Hz` : "Unavailable";
  };

  const formatAudioVolume = () => {
    if (!isSessionActive && metrics.audioStatus === "idle") return "—";
    if (metrics.audioStatus === "unavailable") return "Unavailable";
    if (typeof metrics.volume === "number" && Number.isFinite(metrics.volume)) {
      return metrics.volume.toFixed(1);
    }
    return "Analyzing…";
  };

  return (
    <div className="training-root" style={{ backgroundImage: `url(${bg})` }}>
      <header className="topbar glass">
        <div className="brand">Aura Coach</div>
        <button className="home-link" onClick={handleBackClick} disabled={isBusy}>
          ← Home
        </button>
      </header>

      <div className="grid">
        {/* Left */}
        <section className="card glass card-left">
          <h2 className="card-title">Body Language Analysis</h2>
          <div className="stat">
            <div className="stat-label">Posture Score</div>
            <div className="stat-value accent">
              {typeof metrics.posture === "number" ? `${metrics.posture}%` : isSessionActive ? "Analyzing…" : "—"}
            </div>
          </div>
          <div className="stat">
            <div className="stat-label">Shoulder Tilt</div>
            <div className="stat-value">
              {typeof metrics.headTilt === "number" ? `${metrics.headTilt}°` : isSessionActive ? "Analyzing…" : "—"}
            </div>
          </div>
          <div className="stat">
            <div className="stat-label">Camera Alignment (approx.)</div>
            <div className="stat-value">
              {metrics.eyeContact || (isSessionActive ? "Analyzing…" : "—")}
            </div>
          </div>
        </section>

        {/* Center */}
        <section className="card glass card-center">
          <div className="video-frame">
            <video ref={videoRef} autoPlay playsInline muted className="video" />
            {error && <div className="error">{error}</div>}
          </div>

          {/* Live tips */}
          <div className="live-tips-card">
            <h3>🔮 Live Tips</h3>
            <div className="tips-list">
              {!isSessionActive && <p>Press Start Session to begin live analysis.</p>}
              {isSessionActive && typeof metrics.posture === "number" && metrics.posture < 60 && (
                <p>Straighten your shoulder alignment</p>
              )}
              {isSessionActive && metrics.eyeContact && metrics.eyeContact !== "Good" && (
                <p>Center yourself toward the camera</p>
              )}
              {isSessionActive && typeof metrics.volume === "number" && metrics.volume < 1.5 && (
                <p>Low mic input level</p>
              )}
              {isSessionActive && typeof metrics.fillers === "number" && metrics.fillers > 5 && (
                <p>Reduce filler words</p>
              )}
              {isSessionActive && metrics.tone === "Energetic" && <p>Great vocal energy!</p>}

              {isSessionActive &&
                typeof metrics.posture === "number" &&
                metrics.posture >= 60 &&
                metrics.eyeContact === "Good" &&
                typeof metrics.volume === "number" &&
                metrics.volume >= 1.5 &&
                (metrics.fillers || 0) <= 5 &&
                metrics.tone !== "Energetic" && <p>✅ You’re doing great!</p>}
            </div>
          </div>

          <div className="actions">
            <button
              className={`btn ${isSessionActive ? "danger" : "start"}`}
              onClick={handleSessionToggle}
              disabled={isBusy}
            >
              {phase === "stopping" || phase === "saving"
                ? "⏳ Finalizing Session…"
                : isSessionActive
                ? "⏹ End Session & View Report"
                : "▶ Start Session"}
            </button>
          </div>
        </section>

        {/* Right */}
        <section className="card glass card-right">
          <h2 className="card-title">Speaking Style Feedback</h2>
          <div className="stat">
            <div className="stat-label">Speaking Pace (incl. pauses)</div>
            <div className="stat-value accent">{formatWpm()}</div>
          </div>
          <div className="stat">
            <div className="stat-label">Filler Words (heuristic)</div>
            <div className="stat-value">
              {metrics.fillers === null ? "Unavailable" : metrics.fillers}
            </div>
          </div>
          <div className="stat">
            <div className="stat-label">Expression Estimate</div>
            <div className="stat-value">
              {metrics.emotion || (isSessionActive ? "Analyzing…" : "—")}
            </div>
          </div>
          <div className="stat">
            <div className="stat-label">Tone</div>
            <div className="stat-value">{formatAudioTone()}</div>
          </div>
          <div className="stat">
            <div className="stat-label">Volume (RMS×100)</div>
            <div className="stat-value">{formatAudioVolume()}</div>
          </div>
          <div className="stat">
            <div className="stat-label">Pitch</div>
            <div className="stat-value">{formatAudioPitch()}</div>
          </div>
        </section>
      </div>
    </div>
  );
}

