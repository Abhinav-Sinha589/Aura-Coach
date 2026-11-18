import {
  headRollDeg, shoulderRollDeg, postureScore, facingMeasure, EyeContactEstimator, neckRatio,
} from "./vision";

const W = 1280;
const H = 720;
const rad = (d) => (d * Math.PI) / 180;

/**
 * Synthetic 2-D face/shoulder landmarks (normalised, like MediaPipe Pose).
 * cx/cy: face centre in pixels; roll: head roll (deg, + = image-left side lower);
 * yaw: head yaw (deg); shoulderRoll: shoulder line roll (deg).
 */
function makePose({ cx = 640, cy = 300, roll = 0, yaw = 0, pitch = 0, shoulderRoll = 0, eyeW = 90,
  hideFace = false, hideRightEye = false } = {}) {
  const lm = Array.from({ length: 33 }, () => ({ x: 0, y: 0, visibility: 0 }));
  const put = (i, xPix, yPix, v = 0.99) => { lm[i] = { x: xPix / W, y: yPix / H, visibility: v }; };
  const r = rad(roll);
  // rotate a face-local point (dx, dy) by the roll; y down, + roll lowers image-left
  const rot = (dx, dy) => [cx + dx * Math.cos(r) + dy * Math.sin(r), cy - dx * Math.sin(r) + dy * Math.cos(r)];
  const shrink = Math.cos(rad(yaw));
  const noseShift = 0.26 * Math.tan(rad(yaw)) * eyeW; // nose protrudes ahead of the eye plane
  const noseDrop = 0.5 * eyeW + 0.25 * Math.sin(rad(pitch)) * eyeW;
  const fv = hideFace ? 0.1 : 0.99;
  put(3, ...rot((eyeW / 2) * shrink, 0), fv); // subject-left outer eye corner = image right
  put(6, ...rot((-eyeW / 2) * shrink, 0), hideRightEye ? 0.1 : fv);
  put(2, ...rot((eyeW / 3) * shrink, 0), fv);
  put(5, ...rot((-eyeW / 3) * shrink, 0), hideRightEye ? 0.1 : fv);
  put(7, ...rot((eyeW * 0.8) * shrink, 10), fv);
  put(8, ...rot((-eyeW * 0.8) * shrink, 10), fv);
  put(0, ...rot(noseShift, noseDrop), fv);
  const s = rad(shoulderRoll);
  const sw = 360;
  put(11, cx + (sw / 2) * Math.cos(s), cy + 260 - (sw / 2) * Math.sin(s));
  put(12, cx - (sw / 2) * Math.cos(s), cy + 260 + (sw / 2) * Math.sin(s));
  return lm;
}

describe("Head Tilt measures the head, not the shoulders", () => {
  test("head tilt with level shoulders moves Head Tilt", () => {
    const h = headRollDeg(makePose({ roll: 15 }), W, H);
    expect(h.deg).toBeCloseTo(15, 0);
    expect(shoulderRollDeg(makePose({ roll: 15 }), W, H)).toBeCloseTo(0, 1);
  });

  test("tilted shoulders with a level head do not move Head Tilt", () => {
    const lm = makePose({ shoulderRoll: 12 });
    expect(headRollDeg(lm, W, H).deg).toBeCloseTo(0, 1);
    expect(shoulderRollDeg(lm, W, H)).toBeCloseTo(12, 0);
  });

  test("the old shoulder-line formula would have reported the shoulders", () => {
    const lm = makePose({ shoulderRoll: 12 });
    const L = lm[11], R = lm[12];
    let old = Math.atan2(R.y - L.y, R.x - L.x) * (180 / Math.PI);
    if (old > 90) old -= 180;
    if (old < -90) old += 180;
    expect(Math.abs(old)).toBeGreaterThan(5);
  });

  test("sign convention is stable and mirroring flips it", () => {
    expect(headRollDeg(makePose({ roll: -10 }), W, H).deg).toBeCloseTo(-10, 0);
    expect(headRollDeg(makePose({ roll: 10 }), W, H, true).deg).toBeCloseTo(-10, 0);
  });

  test("angles are computed in pixel space (16:9 safe)", () => {
    expect(headRollDeg(makePose({ roll: 30 }), W, H).deg).toBeCloseTo(30, 0);
  });

  test("face missing -> null, never a fake 0", () => {
    expect(headRollDeg(makePose({ hideFace: true }), W, H)).toBeNull();
    expect(headRollDeg(null, W, H)).toBeNull();
  });

  test("one eye occluded falls back to ears", () => {
    const h = headRollDeg(makePose({ roll: 8, hideRightEye: true }), W, H);
    expect(h.source).toBe("ears");
    expect(h.deg).toBeCloseTo(8, 0);
  });
});

describe("posture score", () => {
  test("keeps the original scale but ignores camera distance", () => {
    expect(postureScore(0)).toBe(100);
    expect(postureScore(5)).toBe(80);
    expect(postureScore(null)).toBeNull();
    const near = makePose({ shoulderRoll: 5 });
    expect(postureScore(shoulderRollDeg(near, W, H))).toBeCloseTo(80, 0);
  });
  test("slouch penalty only beyond 20% drop from own baseline, capped", () => {
    expect(postureScore(0, 0.9, 1.0)).toBe(100);
    expect(postureScore(0, 0.7, 1.0)).toBeCloseTo(90, 5);
    expect(postureScore(0, 0.1, 1.0)).toBe(75);
  });
  test("neck ratio is scale invariant", () => {
    const a = neckRatio(makePose({}), W, H);
    expect(a).toBeGreaterThan(0);
  });
});

describe("Eye Contact proxy is head-relative", () => {
  function run(frames, est = new EyeContactEstimator()) {
    let state;
    frames.forEach(([lm, t]) => { state = est.push(lm ? facingMeasure(lm, W, H) : null, t); });
    return { state, est };
  }
  const seq = (opts, from, to, step = 66) => {
    const out = [];
    for (let t = from; t <= to; t += step) out.push([makePose(opts), t]);
    return out;
  };

  test("calibrates first, then reports", () => {
    const est = new EyeContactEstimator();
    expect(est.push(facingMeasure(makePose({}), W, H), 0)).toBe("Calibrating…");
  });

  test("centred and facing the camera -> Good", () => {
    expect(run(seq({}, 0, 4000)).state).toBe("Good");
  });

  test("off-centre but facing the camera -> Good (old nose.x test said Looking Away)", () => {
    const frames = [...seq({}, 0, 2100), ...seq({ cx: 260 }, 2166, 5000)];
    expect(run(frames).state).toBe("Good");
    const nose = makePose({ cx: 260 })[0];
    expect(Math.abs(nose.x - 0.5) < 0.05).toBe(false); // old rule
  });

  test("centred but head turned sideways -> Looking Away", () => {
    const frames = [...seq({}, 0, 2100), ...seq({ yaw: 35 }, 2166, 5000)];
    expect(run(frames).state).toBe("Looking Away");
  });

  test("looking down at notes -> Looking Away after calibration", () => {
    const frames = [...seq({}, 0, 2100), ...seq({ pitch: 40 }, 2166, 5000)];
    expect(run(frames).state).toBe("Looking Away");
  });

  test("brief glance does not flip the label (hysteresis)", () => {
    const frames = [...seq({}, 0, 3000), ...seq({ yaw: 35 }, 3066, 3150), ...seq({}, 3216, 4000)];
    expect(run(frames).state).toBe("Good");
  });

  test("no face -> Unknown, and session summary needs enough frames", () => {
    const frames = [...seq({}, 0, 5000), [null, 5100], [null, 6000]];
    const { state, est } = run(frames);
    expect(state).toBe("Unknown");
    expect(est.sessionSummary().label).toBe("Good");
    expect(new EyeContactEstimator().sessionSummary()).toEqual({ label: "Unknown", ratio: null });
  });
});
