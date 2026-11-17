/**
 * speechMetrics.js
 * ----------------
 * Pure helpers for:
 *  - RFC4122 v4 UUID generation (with crypto.randomUUID + fallback)
 *  - Finalized-only SpeechRecognition transcript deduplication across restarts
 *  - Word-boundary + multiword-phrase filler counting (no substring overmatch)
 *  - Monotonic-clock session WPM calculation
 *  - Aspect-ratio-aware shoulder tilt & camera-alignment proxy
 *  - Session-long running accumulators for visual metrics & real timeline points
 */

const MULTIWORD_FILLERS = [
  ["you", "know"],
  ["i", "mean"],
  ["kind", "of"],
  ["sort", "of"],
];

// Note: "like" is context-dependent; counted heuristically as a single-token filler
// (disclosed in UI/docs).
const SINGLE_WORD_FILLERS = new Set([
  "um",
  "uh",
  "erm",
  "er",
  "ah",
  "hmm",
  "basically",
  "actually",
  "literally",
  "like",
]);

export function generateSessionId() {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID().toLowerCase();
  }
  const bytes = new Uint8Array(16);
  if (typeof crypto !== "undefined" && typeof crypto.getRandomValues === "function") {
    crypto.getRandomValues(bytes);
  } else {
    for (let i = 0; i < 16; i += 1) {
      bytes[i] = Math.floor(Math.random() * 256);
    }
  }
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

/**
 * Normalize a finalized transcript into lowercase word tokens, stripping
 * punctuation around words so "um,", "you, know!", etc. tokenize cleanly.
 */
export function tokenizeTranscript(text) {
  if (typeof text !== "string" || !text.trim()) return [];
  return text
    .toLowerCase()
    .replace(/['’]/g, "'")
    .replace(/[^a-z0-9'\s]+/g, " ")
    .split(/\s+/)
    .map((w) => w.replace(/^'+|'+$/g, ""))
    .filter(Boolean);
}

/**
 * Count words and filler occurrences in a single finalized utterance.
 * Multiword phrases ("you know", "i mean", "kind of", "sort of") are matched
 * first and consumed so their tokens cannot be double-counted.
 * Substrings inside longer words (e.g. "umbrella", "liked", "human") never match.
 */
export function analyzeFinalizedText(text) {
  const tokens = tokenizeTranscript(text);
  const wordCount = tokens.length;
  if (wordCount === 0) {
    return { wordCount: 0, fillerCount: 0 };
  }

  const consumed = new Array(tokens.length).fill(false);
  let fillerCount = 0;

  for (let i = 0; i < tokens.length - 1; i += 1) {
    if (consumed[i]) continue;
    for (const [first, second] of MULTIWORD_FILLERS) {
      if (tokens[i] === first && tokens[i + 1] === second && !consumed[i + 1]) {
        fillerCount += 1;
        consumed[i] = true;
        consumed[i + 1] = true;
        break;
      }
    }
  }

  for (let i = 0; i < tokens.length; i += 1) {
    if (!consumed[i] && SINGLE_WORD_FILLERS.has(tokens[i])) {
      fillerCount += 1;
    }
  }

  return { wordCount, fillerCount };
}

/**
 * Stateful tracker for a session's SpeechRecognition lifecycle.
 * - Dedupes finalized results by result index within the current recognition instance
 * - Ignores interim results completely
 * - Preserves cumulative totals when a recognition instance restarts after a pause
 */
export function createSpeechTracker(clock = () => performance.now()) {
  let processedFinalIndices = new Set();
  let totalWords = 0;
  let totalFillers = 0;
  let startedAtMs = null;
  let status = "initializing"; // initializing | active | unsupported | denied | error

  return {
    startSession() {
      processedFinalIndices = new Set();
      totalWords = 0;
      totalFillers = 0;
      startedAtMs = clock();
      status = "active";
    },
    onInstanceRestart() {
      // Only reset per-instance result indices; keep cumulative session totals!
      processedFinalIndices = new Set();
    },
    setStatus(nextStatus) {
      status = nextStatus;
    },
    processResultEvent(event) {
      if (!event || !event.results) return this.getSnapshot();
      for (let i = 0; i < event.results.length; i += 1) {
        const res = event.results[i];
        if (!res || !res.isFinal) continue;
        if (processedFinalIndices.has(i)) continue;
        processedFinalIndices.add(i);
        const alt = res[0];
        const transcript = alt && typeof alt.transcript === "string" ? alt.transcript : "";
        const { wordCount, fillerCount } = analyzeFinalizedText(transcript);
        totalWords += wordCount;
        totalFillers += fillerCount;
      }
      return this.getSnapshot();
    },
    getSnapshot() {
      const nowMs = clock();
      const elapsedSeconds =
        startedAtMs !== null ? Math.max(0, (nowMs - startedAtMs) / 1000) : 0;

      if (status === "unsupported" || status === "denied") {
        return {
          status,
          wordCount: null,
          fillers: null,
          wpm: null,
          captureSeconds: round1(elapsedSeconds),
        };
      }
      if (status === "error" && totalWords === 0) {
        return {
          status,
          wordCount: null,
          fillers: null,
          wpm: null,
          captureSeconds: round1(elapsedSeconds),
        };
      }

      let wpm = null;
      if (elapsedSeconds >= 1.0) {
        wpm = Math.round(totalWords / (elapsedSeconds / 60.0));
      } else if (totalWords === 0) {
        wpm = 0;
      }

      return {
        status,
        wordCount: totalWords,
        fillers: totalFillers,
        wpm,
        captureSeconds: round1(elapsedSeconds),
      };
    },
  };
}

function round1(n) {
  return Math.round(n * 10) / 10;
}

/**
 * Compute posture score, aspect-corrected shoulder tilt (deg), and camera-facing
 * alignment proxy from MediaPipe Pose landmarks. Returns null fields when
 * required landmarks are missing or low-visibility (< 0.5).
 */
export function evaluatePoseLandmarks(landmarks, videoWidth = 1280, videoHeight = 720) {
  if (!Array.isArray(landmarks)) return null;
  const L = landmarks[11];
  const R = landmarks[12];
  const nose = landmarks[0];
  const minVis = 0.5;

  const hasShoulders =
    L &&
    R &&
    typeof L.x === "number" &&
    typeof L.y === "number" &&
    typeof R.x === "number" &&
    typeof R.y === "number" &&
    (L.visibility === undefined || L.visibility >= minVis) &&
    (R.visibility === undefined || R.visibility >= minVis);

  const hasNose =
    nose &&
    typeof nose.x === "number" &&
    typeof nose.y === "number" &&
    (nose.visibility === undefined || nose.visibility >= minVis);

  if (!hasShoulders && !hasNose) return null;

  let posture = null;
  let shoulderTilt = null;
  if (hasShoulders) {
    const aspect = videoWidth > 0 && videoHeight > 0 ? videoWidth / videoHeight : 16 / 9;
    const dx = (R.x - L.x) * aspect;
    const dy = R.y - L.y;
    let deg = Math.atan2(dy, dx) * (180 / Math.PI);
    if (deg > 90) deg -= 180;
    if (deg < -90) deg += 180;
    shoulderTilt = Math.round(deg);
    const verticalDiffPct = Math.abs(dy) * 100;
    posture = Math.round(Math.max(0, Math.min(100, 100 - verticalDiffPct * 4)));
  }

  let eyeContact = null;
  if (hasNose) {
    eyeContact = Math.abs(nose.x - 0.5) < 0.08 ? "Good" : "Looking Away";
  }

  return { posture, shoulderTilt, eyeContact };
}

/**
 * Running session-wide visual accumulator so the saved session reflects all
 * valid observations across the session rather than the last frame.
 */
export function createVisualAccumulator() {
  let postureSum = 0;
  let postureCount = 0;
  let tiltSum = 0;
  let tiltCount = 0;
  let goodAlign = 0;
  let awayAlign = 0;
  const emotionCounts = {};
  const timeline = [];
  let lastCheckpointSec = -Infinity;

  return {
    addPose(obs, elapsedSec) {
      if (!obs) return;
      if (typeof obs.posture === "number" && Number.isFinite(obs.posture)) {
        postureSum += obs.posture;
        postureCount += 1;
      }
      if (typeof obs.shoulderTilt === "number" && Number.isFinite(obs.shoulderTilt)) {
        tiltSum += obs.shoulderTilt;
        tiltCount += 1;
      }
      if (obs.eyeContact === "Good") goodAlign += 1;
      else if (obs.eyeContact === "Looking Away") awayAlign += 1;

      if (
        typeof elapsedSec === "number" &&
        elapsedSec >= 0 &&
        elapsedSec - lastCheckpointSec >= 5.0 &&
        timeline.length < 720
      ) {
        lastCheckpointSec = elapsedSec;
        timeline.push({
          t: round1(elapsedSec),
          posture: postureCount > 0 ? round1(postureSum / postureCount) : null,
          wpm: null, // filled by caller with current session WPM
        });
      }
    },
    recordCheckpoint(elapsedSec, currentPosture, currentWpm) {
      if (typeof elapsedSec !== "number" || elapsedSec < 0 || timeline.length >= 720) return;
      if (elapsedSec - lastCheckpointSec < 4.5) return;
      lastCheckpointSec = elapsedSec;
      timeline.push({
        t: round1(elapsedSec),
        posture: typeof currentPosture === "number" && Number.isFinite(currentPosture) ? round1(currentPosture) : null,
        wpm: typeof currentWpm === "number" && Number.isFinite(currentWpm) ? round1(currentWpm) : null,
      });
    },
    addExpression(expr) {
      if (typeof expr !== "string" || !expr) return;
      emotionCounts[expr] = (emotionCounts[expr] || 0) + 1;
    },
    getSummary() {
      const totalAlign = goodAlign + awayAlign;
      let dominantEmotion = null;
      const entries = Object.entries(emotionCounts);
      if (entries.length > 0) {
        entries.sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
        dominantEmotion = entries[0][0];
      }
      return {
        posture: postureCount > 0 ? round1(postureSum / postureCount) : null,
        posture_samples: postureCount,
        shoulder_tilt: tiltCount > 0 ? round1(tiltSum / tiltCount) : null,
        eyeContact:
          totalAlign === 0
            ? null
            : goodAlign / totalAlign >= 0.5
            ? "Good"
            : "Looking Away",
        eye_contact_ratio: totalAlign > 0 ? Math.round((goodAlign / totalAlign) * 1000) / 1000 : null,
        camera_samples: totalAlign,
        emotion: dominantEmotion,
        emotion_counts: { ...emotionCounts },
        timeline: timeline.slice(),
      };
    },
  };
}
