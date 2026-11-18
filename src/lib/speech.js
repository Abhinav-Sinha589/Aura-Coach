// Speech-recognition bookkeeping: count each *finalised* segment exactly once,
// detect filler words/phrases, and compute WPM from real elapsed time.
// Pure functions (no browser APIs) so they can be unit-tested.

export const PHRASE_FILLERS = ["you know", "i mean", "kind of", "sort of"];
export const WORD_FILLERS = ["um", "umm", "uh", "uhh", "uhm", "er", "erm", "ah", "hmm", "basically"];
// "like" is only a filler in some contexts ("and like", "so like", "like, um").
const LIKE_FILLER_PREV = new Set(["and", "so", "but", "or", "yeah", "oh", "well", "um", "uh", "like"]);

export function normalizeTranscript(text) {
  return String(text || "")
    .toLowerCase()
    .replace(/[’']/g, "'")
    .replace(/[^a-z0-9'\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

export function tokenize(text) {
  const t = normalizeTranscript(text);
  return t ? t.split(" ") : [];
}

/** Count fillers in one finalised segment without double counting phrase tokens. */
export function countFillers(text) {
  const tokens = tokenize(text);
  const used = new Array(tokens.length).fill(false);
  let count = 0;
  for (const phrase of PHRASE_FILLERS) {
    const p = phrase.split(" ");
    for (let i = 0; i + p.length <= tokens.length; i++) {
      if (used[i]) continue;
      let ok = true;
      for (let k = 0; k < p.length; k++) {
        if (tokens[i + k] !== p[k] || used[i + k]) { ok = false; break; }
      }
      if (ok) {
        // "kind of"/"sort of" followed by a noun-ish word is often literal
        // ("a kind of fruit"); only count it when not preceded by "a"/"the"/"what".
        if ((phrase === "kind of" || phrase === "sort of") && ["a", "the", "what", "this", "that", "any", "some"].includes(tokens[i - 1])) continue;
        for (let k = 0; k < p.length; k++) used[i + k] = true;
        count++;
      }
    }
  }
  tokens.forEach((tok, i) => {
    if (used[i]) return;
    if (WORD_FILLERS.includes(tok)) {
      count++;
      used[i] = true;
    } else if (tok === "like") {
      const prev = i === 0 ? null : tokens[i - 1];
      if (i === 0 || LIKE_FILLER_PREV.has(prev)) {
        count++;
        used[i] = true;
      }
    }
  });
  return count;
}

export function computeWpm(words, elapsedMs, minElapsedMs = 5000) {
  if (!(elapsedMs >= minElapsedMs)) return null;
  return Math.round(words / (elapsedMs / 60000));
}

/**
 * Tracks finalised recognition results across restarts. Each restart of the
 * SpeechRecognition object starts a new `generation`; result indices restart
 * at 0 there, so seen-keys are scoped per generation.
 */
export function createSpeechTracker() {
  const state = {
    words: 0,
    fillers: 0,
    segments: 0,
    fillerTimes: [],
    seen: new Set(),
    generation: 0,
    lastFinal: { text: "", t: 0, generation: -1 },
  };

  function newGeneration() {
    state.generation += 1;
    state.seen = new Set();
  }

  /** results: SpeechRecognitionResultList-like; returns number of new words. */
  function ingest(results, resultIndex, now = Date.now()) {
    let added = 0;
    for (let i = resultIndex || 0; i < results.length; i++) {
      const r = results[i];
      if (!r || !r.isFinal) continue;
      const key = `${state.generation}:${i}`;
      if (state.seen.has(key)) continue;
      state.seen.add(key);
      const text = normalizeTranscript(r[0] && r[0].transcript);
      if (!text) continue;

      // Some Chrome builds (notably Android) re-emit a growing final
      // ("hello" then "hello world") as separate results: count the delta only.
      let segmentText = text;
      const prev = state.lastFinal;
      if (prev.generation === state.generation && now - prev.t < 2500 && prev.text) {
        if (text === prev.text) segmentText = "";
        else if (text.startsWith(prev.text + " ")) segmentText = text.slice(prev.text.length + 1);
      }
      state.lastFinal = { text, t: now, generation: state.generation };
      if (!segmentText) continue;

      const n = tokenize(segmentText).length;
      const f = countFillers(segmentText);
      state.words += n;
      state.fillers += f;
      state.segments += 1;
      for (let k = 0; k < f; k++) state.fillerTimes.push(now);
      added += n;
    }
    return added;
  }

  function recentFillers(now = Date.now(), windowMs = 60000) {
    while (state.fillerTimes.length && now - state.fillerTimes[0] > windowMs) state.fillerTimes.shift();
    return state.fillerTimes.length;
  }

  return {
    ingest,
    newGeneration,
    recentFillers,
    get words() { return state.words; },
    get fillers() { return state.fillers; },
    get segments() { return state.segments; },
  };
}

export const FATAL_SPEECH_ERRORS = new Set([
  "not-allowed",
  "service-not-allowed",
  "audio-capture",
  "language-not-supported",
  "bad-grammar",
]);
