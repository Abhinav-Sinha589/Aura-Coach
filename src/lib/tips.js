// Live-tip conditions with debounce so tips neither flicker nor stick forever.
// A tip appears once its condition has held for `onMs` and disappears after it
// has been false for `offMs`. Conditions only fire on *valid* measurements.

export const LOW_VOLUME = 1.0; // RMS x 100 (~ -40 dBFS) while voice is active
export const HEAD_TILT_TIP_DEG = 12;

export const TIP_DEFS = [
  { id: "posture", text: "Straighten your posture", warn: true,
    when: (m) => m.posture !== null && m.posture < 60 },
  { id: "eye", text: "Maintain eye contact", warn: true,
    when: (m) => m.eyeContact === "Looking Away" },
  { id: "head", text: "Keep your head level", warn: true,
    when: (m) => m.headTilt !== null && Math.abs(m.headTilt) > HEAD_TILT_TIP_DEG },
  { id: "volume", text: "Speak louder", warn: true,
    when: (m) => m.voiceActive && m.volume !== null && m.volume < LOW_VOLUME },
  { id: "clip", text: "Mic is clipping — move back a little", warn: true,
    when: (m) => m.clipping === true },
  { id: "fillers", text: "Reduce filler words", warn: true,
    when: (m) => m.recentFillers !== null && m.recentFillers >= 3 },
  { id: "energy", text: "Great energy!", warn: false,
    when: (m) => m.tone === "Energetic" },
];

export function createTipsEngine({ onMs = 1500, offMs = 1000 } = {}) {
  const since = {};
  const shown = new Set();
  return function evaluate(m, now) {
    for (const def of TIP_DEFS) {
      const active = !!def.when(m);
      const st = since[def.id] || (since[def.id] = { on: null, off: null });
      if (active) {
        st.off = null;
        if (st.on === null) st.on = now;
        if (now - st.on >= onMs) shown.add(def.id);
      } else {
        st.on = null;
        if (st.off === null) st.off = now;
        if (now - st.off >= offMs) shown.delete(def.id);
      }
    }
    const tips = TIP_DEFS.filter((d) => shown.has(d.id)).map((d) => d.text);
    const anyWarning = TIP_DEFS.some((d) => d.warn && shown.has(d.id));
    if (!anyWarning && m.hasMeasurement) tips.push("✅ You’re doing great!");
    return tips;
  };
}
