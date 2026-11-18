// Temporal smoothing for face-api.js expression probabilities.
// face-api.js classifies a facial *expression*; it cannot know how someone
// actually feels. The smoother only reduces frame-to-frame flicker.

export const EXPRESSIONS = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"];

export const capitalize = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : s);

export class EmotionSmoother {
  constructor({ window = 5, minConfidence = 0.45, margin = 0.1, noFaceMs = 2000 } = {}) {
    Object.assign(this, { windowSize: window, minConfidence, margin, noFaceMs });
    this.reset();
  }

  reset() {
    this.samples = [];
    this.current = null;
    this.lastFace = null;
    this.started = false;
    this.counts = {};
    this.rawSwitches = 0;
    this.smoothSwitches = 0;
    this.lastRaw = null;
  }

  /** expressions: {neutral: p, happy: p, ...} from face-api, or null if no face. */
  push(expressions, now) {
    this.started = true;
    if (!expressions) {
      if (this.lastFace !== null && now - this.lastFace > this.noFaceMs) {
        this.samples = [];
        this.current = null;
      }
      return this.state(now);
    }
    this.lastFace = now;
    const probs = EXPRESSIONS.map((k) => Number(expressions[k]) || 0);
    const rawIdx = probs.indexOf(Math.max(...probs));
    if (this.lastRaw !== null && rawIdx !== this.lastRaw) this.rawSwitches += 1;
    this.lastRaw = rawIdx;
    if (probs[rawIdx] < this.minConfidence) return this.state(now); // ambiguous frame

    this.samples.push(probs);
    if (this.samples.length > this.windowSize) this.samples.shift();
    const mean = EXPRESSIONS.map((_, i) => this.samples.reduce((s, p) => s + p[i], 0) / this.samples.length);
    const best = mean.indexOf(Math.max(...mean));
    if (this.current === null) this.current = best;
    else if (best !== this.current && mean[best] - mean[this.current] > this.margin) {
      this.current = best;
      this.smoothSwitches += 1;
    }
    const label = capitalize(EXPRESSIONS[this.current]);
    this.counts[label] = (this.counts[label] || 0) + 1;
    return this.state(now);
  }

  state(now) {
    if (this.lastFace === null) return this.started ? "No face" : "Detecting…";
    if (now - this.lastFace > this.noFaceMs) return "No face";
    return this.current === null ? "Detecting…" : capitalize(EXPRESSIONS[this.current]);
  }

  dominant() {
    const entries = Object.entries(this.counts);
    if (!entries.length) return null;
    return entries.sort((a, b) => b[1] - a[1])[0][0];
  }
}
