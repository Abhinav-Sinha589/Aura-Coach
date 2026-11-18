// Geometry for the live body-language metrics, computed from MediaPipe Pose
// landmarks (normalised x/y in [0,1], `visibility` in [0,1]).
//
// All angles are measured in *pixel* space (x * width, y * height); using
// normalised coordinates directly distorts angles on 16:9 video.
//
// Landmark indices (MediaPipe Pose): 0 nose, 2 left eye, 3 left eye outer,
// 5 right eye, 6 right eye outer, 7 left ear, 8 right ear,
// 11 left shoulder, 12 right shoulder.

export const MIN_VIS = 0.5;

const vis = (p) => (p && typeof p.visibility === "number" ? p.visibility : p ? 1 : 0);
const ok = (...pts) => pts.every((p) => p && vis(p) >= MIN_VIS);
const px = (p, w, h) => ({ x: p.x * w, y: p.y * h });

/**
 * Roll of the line through two bilateral points, in degrees.
 * Points are ordered by image x so labels/mirroring can't flip the line.
 * Positive = the image-left point is lower. In an un-mirrored webcam frame the
 * image-left side is the subject's right, so positive = tilted toward the
 * subject's right shoulder. Pass mirrored=true for mirrored sources.
 */
export function lineRollDeg(a, b, w, h, mirrored = false) {
  let p = px(a, w, h);
  let q = px(b, w, h);
  if (q.x < p.x) [p, q] = [q, p];
  const dx = q.x - p.x;
  if (dx < 1e-6) return null;
  const deg = (Math.atan2(p.y - q.y, dx) * 180) / Math.PI;
  return mirrored ? -deg : deg;
}

/**
 * Head roll (left/right tilt) from the eye line, falling back to the ear line.
 * This measures the head itself: shrugging or tilting the shoulders does not
 * change it. Returns {deg, source} or null when the face isn't visible.
 */
export function headRollDeg(lm, w, h, mirrored = false) {
  if (!lm) return null;
  const pairs = [
    [lm[3], lm[6], "eyes_outer"],
    [lm[2], lm[5], "eyes"],
    [lm[7], lm[8], "ears"],
  ];
  const angles = [];
  let source = null;
  for (const [a, b, name] of pairs.slice(0, 2)) {
    if (ok(a, b)) {
      const d = lineRollDeg(a, b, w, h, mirrored);
      if (d !== null) { angles.push(d); source = source || name; }
    }
  }
  if (!angles.length) {
    const [a, b, name] = pairs[2];
    if (ok(a, b)) {
      const d = lineRollDeg(a, b, w, h, mirrored);
      if (d !== null) { angles.push(d); source = name; }
    }
  }
  if (!angles.length) return null;
  return { deg: angles.reduce((s, x) => s + x, 0) / angles.length, source };
}

export function shoulderRollDeg(lm, w, h, mirrored = false) {
  if (!lm || !ok(lm[11], lm[12])) return null;
  return lineRollDeg(lm[11], lm[12], w, h, mirrored);
}

/**
 * Vertical eye-to-shoulder distance divided by shoulder width. Scale-invariant
 * (camera distance cancels); it shrinks when the head sinks toward the
 * shoulders (slouching). Used only relative to the user's own start-of-session
 * baseline.
 */
export function neckRatio(lm, w, h) {
  if (!lm || !ok(lm[11], lm[12], lm[2], lm[5])) return null;
  const l = px(lm[11], w, h), r = px(lm[12], w, h);
  const el = px(lm[2], w, h), er = px(lm[5], w, h);
  const sw = Math.hypot(l.x - r.x, l.y - r.y);
  if (sw < 1) return null;
  const shoulderY = (l.y + r.y) / 2;
  const eyeY = (el.y + er.y) / 2;
  return (shoulderY - eyeY) / sw;
}

/**
 * Posture score 0–100. The shoulder term keeps the original metric's scale
 * (old: 100 − 4·|Δy·100| on normalised coords ≈ 100 − 4·|shoulder roll°| at a
 * typical webcam distance) but no longer depends on how close you sit.
 * A capped slouch penalty applies only after a per-session baseline exists.
 * Heuristic, not a clinical posture measure.
 */
export function postureScore(shoulderDeg, ratio = null, baselineRatio = null) {
  if (shoulderDeg === null || shoulderDeg === undefined) return null;
  let score = 100 - 4 * Math.abs(shoulderDeg);
  if (ratio !== null && baselineRatio && baselineRatio > 0) {
    const drop = (baselineRatio - ratio) / baselineRatio; // 0.25 = head 25% closer to shoulders
    const penalty = Math.min(25, Math.max(0, (drop - 0.2) * 100));
    score -= penalty;
  }
  return Math.max(0, Math.min(100, score));
}

/**
 * Head-relative facing measurements (independent of where the face is in the
 * frame):
 *  yaw   – nose position between the outer eye corners, centred at 0
 *          (≈ 0.26·tan(yaw) for a typical face; ±0.09 ≈ ±19°)
 *  pitch – nose drop below the eye line in eye-widths (person-specific,
 *          therefore compared with a calibrated baseline)
 */
export function facingMeasure(lm, w, h) {
  if (!lm || !ok(lm[0], lm[3], lm[6])) return null;
  let a = px(lm[3], w, h), b = px(lm[6], w, h);
  if (b.x < a.x) [a, b] = [b, a];
  const eyeDist = Math.hypot(b.x - a.x, b.y - a.y);
  if (eyeDist < 1) return null;
  const n = px(lm[0], w, h);
  // project nose onto the eye line so head roll doesn't masquerade as yaw
  const ux = (b.x - a.x) / eyeDist, uy = (b.y - a.y) / eyeDist;
  const along = ((n.x - a.x) * ux + (n.y - a.y) * uy) / eyeDist; // 0..1
  const across = (-(n.x - a.x) * uy + (n.y - a.y) * ux) / eyeDist; // below line > 0
  return { yaw: along - 0.5, pitch: across };
}

const median = (arr) => {
  const s = [...arr].sort((x, y) => x - y);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

/**
 * Eye Contact estimator: "is the face oriented toward the camera?" with a
 * short per-session calibration, EMA smoothing and hysteresis.
 * It estimates camera-facing head orientation; it does not track the irises,
 * so eyes-only glances are not detected.
 */
export class EyeContactEstimator {
  constructor({ yawTol = 0.09, pitchTol = 0.1, calibMs = 2000, alpha = 0.25, lostMs = 700 } = {}) {
    Object.assign(this, { yawTol, pitchTol, calibMs, alpha, lostMs });
    this.reset();
  }

  reset() {
    this.start = null;
    this.calib = [];
    this.yaw0 = 0;
    this.pitch0 = null;
    this.calibrated = false;
    this.ema = null;
    this.label = "Unknown";
    this.lastSeen = null;
    this.goodFrames = 0;
    this.measuredFrames = 0;
  }

  push(measure, now) {
    if (this.start === null) this.start = now;
    if (!measure) {
      if (this.lastSeen === null || now - this.lastSeen > this.lostMs) {
        this.label = "Unknown";
        this.ema = null;
      }
      return this.state(now);
    }
    this.lastSeen = now;

    if (!this.calibrated) {
      if (Math.abs(measure.yaw) < 0.15) this.calib.push(measure);
      if (now - this.start >= this.calibMs) {
        if (this.calib.length >= 5) {
          this.yaw0 = Math.max(-0.05, Math.min(0.05, median(this.calib.map((m) => m.yaw))));
          this.pitch0 = median(this.calib.map((m) => m.pitch));
        }
        this.calibrated = true; // fall back to yaw-only if calibration was unusable
      } else {
        return this.state(now);
      }
    }

    const yawOk = Math.abs(measure.yaw - this.yaw0) <= this.yawTol;
    const pitchOk = this.pitch0 === null || Math.abs(measure.pitch - this.pitch0) <= this.pitchTol;
    const facing = yawOk && pitchOk ? 1 : 0;
    this.ema = this.ema === null ? facing : this.ema + this.alpha * (facing - this.ema);
    if (this.label === "Unknown") this.label = this.ema >= 0.5 ? "Good" : "Looking Away";
    else if (this.ema >= 0.6) this.label = "Good";
    else if (this.ema <= 0.4) this.label = "Looking Away";

    this.measuredFrames += 1;
    if (this.label === "Good") this.goodFrames += 1;
    return this.state(now);
  }

  state(now) {
    if (!this.calibrated && this.start !== null && now - this.start < this.calibMs) return "Calibrating…";
    return this.label;
  }

  sessionSummary(minFrames = 30) {
    if (this.measuredFrames < minFrames) return { label: "Unknown", ratio: null };
    const ratio = this.goodFrames / this.measuredFrames;
    return { label: ratio >= 0.5 ? "Good" : "Looking Away", ratio: Math.round(ratio * 1000) / 1000 };
  }
}

/** Exponential smoother that resets after a gap. */
export class Ema {
  constructor(alpha, maxGapMs = 1000) {
    this.alpha = alpha;
    this.maxGapMs = maxGapMs;
    this.value = null;
    this.t = null;
  }

  push(x, now) {
    if (x === null || x === undefined || !Number.isFinite(x)) return this.get(now);
    if (this.value === null || (this.t !== null && now - this.t > this.maxGapMs)) this.value = x;
    else this.value += this.alpha * (x - this.value);
    this.t = now;
    return this.value;
  }

  get(now) {
    if (this.value === null || (this.t !== null && now - this.t > this.maxGapMs)) return null;
    return this.value;
  }
}

/** Running mean without storing samples. */
export class RunningMean {
  constructor() { this.n = 0; this.sum = 0; this.absSum = 0; }
  push(x) {
    if (x === null || x === undefined || !Number.isFinite(x)) return;
    this.n += 1; this.sum += x; this.absSum += Math.abs(x);
  }
  mean() { return this.n ? this.sum / this.n : null; }
  absMean() { return this.n ? this.absSum / this.n : null; }
}

export { median };
