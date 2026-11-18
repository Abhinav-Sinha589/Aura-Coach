// Microphone capture -> mono Float32 chunks at the AudioContext's real rate.
// Uses an AudioWorklet (audio thread, no drops while MediaPipe/face-api keep
// the main thread busy) and falls back to ScriptProcessor where unavailable.

const CHUNK = 4096;

const WORKLET_SRC = `
class AuraPcmCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buf = new Float32Array(${CHUNK});
    this.n = 0;
    this.port.onmessage = (e) => {
      if (e.data === "flush") {
        if (this.n) this.port.postMessage(this.buf.slice(0, this.n));
        this.n = 0;
        this.port.postMessage("flushed");
      }
    };
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) {
      for (let i = 0; i < ch.length; i++) {
        this.buf[this.n++] = ch[i];
        if (this.n === this.buf.length) {
          this.port.postMessage(this.buf.slice(0));
          this.n = 0;
        }
      }
    }
    return true;
  }
}
registerProcessor("aura-pcm-capture", AuraPcmCapture);
`;

/** Float32 [-1,1] -> little-endian Int16 (clamped, no wrap-around on clipping). */
export function floatToPcm16(input) {
  const out = new Int16Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const s = Math.max(-1, Math.min(1, input[i]));
    out[i] = s < 0 ? Math.round(s * 0x8000) : Math.round(s * 0x7fff);
  }
  return out;
}

/**
 * Start capturing from `stream`. onChunk(Float32Array) is called ~every 85 ms.
 * Returns {sampleRate, stop(): Promise<void>} — stop() flushes the tail.
 */
export async function startCapture(stream, onChunk) {
  const Ctx = window.AudioContext || window.webkitAudioContext;
  const ctx = new Ctx();
  if (ctx.state === "suspended") {
    try { await ctx.resume(); } catch (_) { /* resumes on next gesture */ }
  }
  const source = ctx.createMediaStreamSource(stream);
  const sink = ctx.createGain();
  sink.gain.value = 0; // keep the graph pulling without playing audio back
  sink.connect(ctx.destination);

  let node = null;
  let url = null;
  let flushResolve = null;

  if (ctx.audioWorklet && typeof AudioWorkletNode !== "undefined") {
    try {
      url = URL.createObjectURL(new Blob([WORKLET_SRC], { type: "application/javascript" }));
      await ctx.audioWorklet.addModule(url);
      node = new AudioWorkletNode(ctx, "aura-pcm-capture", {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        channelCount: 1,
        channelCountMode: "explicit",
        channelInterpretation: "speakers", // stereo mics are down-mixed to mono
      });
      node.port.onmessage = (e) => {
        if (e.data === "flushed") { if (flushResolve) flushResolve(); return; }
        onChunk(e.data);
      };
    } catch (err) {
      console.warn("AudioWorklet unavailable, using ScriptProcessor", err);
      node = null;
    }
  }
  if (!node) {
    node = ctx.createScriptProcessor(CHUNK, 1, 1);
    node.onaudioprocess = (e) => onChunk(new Float32Array(e.inputBuffer.getChannelData(0)));
  }
  source.connect(node);
  node.connect(sink);

  return {
    sampleRate: ctx.sampleRate,
    async stop() {
      try {
        if (node.port) {
          await Promise.race([
            new Promise((res) => { flushResolve = res; node.port.postMessage("flush"); }),
            new Promise((res) => setTimeout(res, 300)),
          ]);
        }
      } catch (_) { /* ignore */ }
      try { source.disconnect(); node.disconnect(); sink.disconnect(); } catch (_) { /* ignore */ }
      if (node.port) node.port.onmessage = null;
      else node.onaudioprocess = null;
      try { await ctx.close(); } catch (_) { /* ignore */ }
      if (url) URL.revokeObjectURL(url);
    },
  };
}
