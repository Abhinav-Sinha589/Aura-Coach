import { EmotionSmoother } from "./emotion";
import { createTipsEngine } from "./tips";
import { floatToPcm16 } from "./audioCapture";
import { newSessionId, formatClock } from "./backend";

const expr = (top, p = 0.8) => {
  const base = { neutral: 0.02, happy: 0.02, sad: 0.02, angry: 0.02, fearful: 0.02, disgusted: 0.02, surprised: 0.02 };
  return { ...base, [top]: p };
};

describe("EmotionSmoother", () => {
  test("states before and without a face", () => {
    const s = new EmotionSmoother();
    expect(s.state(0)).toBe("Detecting…");
    expect(s.push(null, 0)).toBe("No face");
  });

  test("reduces flicker on alternating noisy frames", () => {
    const s = new EmotionSmoother();
    const pattern = ["neutral", "happy", "neutral", "neutral", "happy", "neutral", "surprised", "neutral",
      "neutral", "happy", "neutral", "neutral"];
    pattern.forEach((e, i) => s.push(expr(e, 0.6), i * 600));
    expect(s.rawSwitches).toBeGreaterThanOrEqual(7);
    expect(s.smoothSwitches).toBeLessThanOrEqual(1);
    expect(s.state(pattern.length * 600)).toBe("Neutral");
  });

  test("still follows a sustained change within a few samples", () => {
    const s = new EmotionSmoother();
    for (let i = 0; i < 5; i++) s.push(expr("neutral"), i * 600);
    let t = 3000;
    let switchedAfter = null;
    for (let k = 1; k <= 5; k++) {
      t += 600;
      if (s.push(expr("happy"), t) === "Happy" && switchedAfter === null) switchedAfter = k;
    }
    expect(switchedAfter).not.toBeNull();
    expect(switchedAfter).toBeLessThanOrEqual(3);
  });

  test("face lost -> 'No face' instead of a stale label", () => {
    const s = new EmotionSmoother();
    s.push(expr("happy"), 0);
    expect(s.push(null, 2500)).toBe("No face");
  });

  test("low-confidence frames are ignored; dominant label over session", () => {
    const s = new EmotionSmoother();
    s.push(expr("angry", 0.3), 0);
    expect(s.dominant()).toBeNull();
    for (let i = 1; i < 6; i++) s.push(expr("happy"), i * 600);
    expect(s.dominant()).toBe("Happy");
  });
});

describe("live tips", () => {
  const base = { posture: 90, eyeContact: "Good", headTilt: 0, voiceActive: false, volume: null,
    clipping: false, recentFillers: 0, tone: null, hasMeasurement: true };

  test("a tip needs its condition to persist, then clears", () => {
    const evaluate = createTipsEngine();
    expect(evaluate({ ...base, posture: 40 }, 0)).not.toContain("Straighten your posture");
    expect(evaluate({ ...base, posture: 40 }, 1600)).toContain("Straighten your posture");
    expect(evaluate({ ...base }, 1800)).toContain("Straighten your posture");
    expect(evaluate({ ...base }, 2900)).not.toContain("Straighten your posture");
  });

  test("'Speak louder' only while actually speaking quietly (no permanent tip)", () => {
    const evaluate = createTipsEngine({ onMs: 0 });
    expect(evaluate({ ...base, volume: 0.2, voiceActive: false }, 0)).not.toContain("Speak louder");
    expect(evaluate({ ...base, volume: 0.4, voiceActive: true }, 10)).toContain("Speak louder");
    evaluate({ ...base, volume: 4, voiceActive: true }, 2000);
    expect(evaluate({ ...base, volume: 4, voiceActive: true }, 3100)).not.toContain("Speak louder");
  });

  test("unknown measurements never trigger warnings; 'doing great' when all good", () => {
    const evaluate = createTipsEngine({ onMs: 0 });
    const tips = evaluate({ ...base, posture: null, eyeContact: "Unknown", headTilt: null }, 0);
    expect(tips).toEqual(["✅ You’re doing great!"]);
    expect(evaluate({ ...base, hasMeasurement: false }, 5000)).toEqual([]);
  });

  test("filler tip uses the recent window, energy tip from model tone", () => {
    const evaluate = createTipsEngine({ onMs: 0 });
    const tips = evaluate({ ...base, recentFillers: 3, tone: "Energetic" }, 0);
    expect(tips).toEqual(expect.arrayContaining(["Reduce filler words", "Great energy!"]));
  });
});

describe("helpers", () => {
  test("PCM16 conversion clamps instead of wrapping on clipping", () => {
    const out = floatToPcm16(new Float32Array([0, 1, -1, 1.5, -2, 0.5]));
    expect(Array.from(out)).toEqual([0, 32767, -32768, 32767, -32768, 16384]);
  });
  test("session ids are strong and server-valid", () => {
    const id = newSessionId();
    expect(id).toMatch(/^session_[0-9a-f]{32}$/);
    expect(newSessionId()).not.toBe(id);
  });
  test("trend labels", () => {
    expect(formatClock(65)).toBe("1:05");
  });
});
