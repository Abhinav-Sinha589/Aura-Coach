"""
audio_pipeline.py
-----------------
Streaming audio analysis for the /api/audio-stream WebSocket.

Why this module exists
~~~~~~~~~~~~~~~~~~~~~~
The tone classifier (backend/models/tone_model.joblib) was trained by
backend/ml/train_tone_model.py on **1.5-second clips sampled at 16 kHz**,
with features computed by backend/ml/audio_features.py:

    [rms, pitch_mean, pitch_std, spectral_centroid, zero_crossing_rate]

The original live path broke that contract in two ways:
  1. Browsers capture at the AudioContext rate (usually 44.1 or 48 kHz) but
     the server decoded every chunk as if it were 16 kHz. Pitch and spectral
     centroid were therefore reported ~3x too low (a 180 Hz tone read as
     ~60 Hz) and the model saw out-of-distribution inputs.
  2. Every ~43 ms ScriptProcessor chunk was classified on its own, instead of
     the 1.5 s windows the model was trained on.

This module fixes both *without touching the trained model or the feature
definitions*: audio is buffered at the client's real sample rate, a 1.5 s
window is resampled to 16 kHz, and the unchanged
`extract_features_from_waveform()` is applied to it. A new analysis runs every
`hop_sec` (default 0.5 s), so the UI updates twice a second while the model
always sees a training-length window.

Things deliberately kept *out* of the model input:
  * Silence gating, voicing detection (librosa.pyin) and the displayed pitch
    are computed separately. The displayed pitch is the median of *voiced*
    frames only; the model still receives YIN mean/std over the whole window
    exactly as in training.
  * Per-window DC offset is removed before analysis. The synthetic training
    clips were zero-mean, so this only moves live input closer to the training
    distribution; it does not change how any feature is defined.
"""
from __future__ import annotations

import threading
import time
from collections import Counter, deque
from typing import Callable, Optional

import numpy as np
import librosa

try:
    from backend.ml.audio_features import extract_features_from_waveform, FEATURE_NAMES
except ImportError:  # running from inside backend/
    from ml.audio_features import extract_features_from_waveform, FEATURE_NAMES

# ---- Contract with backend/ml/train_tone_model.py (SR, DURATION) ----------
TRAINING_SAMPLE_RATE = 16000
TRAINING_WINDOW_SEC = 1.5

DEFAULT_HOP_SEC = 0.5
MIN_INPUT_SAMPLE_RATE = 8000
MAX_INPUT_SAMPLE_RATE = 192000

# Silence gate on RMS of float samples in [-1, 1]. 0.002 ~= -54 dBFS.
SILENCE_GATE_MIN = 0.002
SILENCE_GATE_MAX = 0.015
CLIP_LEVEL = 0.999

# pyin settings for voicing + displayed pitch (not a model input)
PYIN_FMIN = 60.0
PYIN_FMAX = 400.0
PYIN_FRAME = 2048
PYIN_HOP = 512
PYIN_RESOLUTION = 0.25      # semitones; 0.1 (default) is ~3x slower for no visible gain
PYIN_CONTEXT_SEC = 0.75     # only the most recent audio is pitch-tracked
MIN_RECENT_VOICED = 4       # frames (~128 ms) before a pitch is reported

TONE_SMOOTHING_WINDOWS = 3
TONE_HOLD_SEC = 2.0
PITCH_HOLD_SEC = 1.0

# Probability vectors from the RF are averaged over the last few windows.
ToneProbaFn = Callable[[np.ndarray], tuple]


def decode_pcm16le(data: bytes) -> np.ndarray:
    """Little-endian signed 16-bit PCM -> float32 in [-1, 1)."""
    if len(data) % 2:
        data = data[:-1]
    if not data:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def resample_to_training_rate(y: np.ndarray, sr: int) -> np.ndarray:
    if sr == TRAINING_SAMPLE_RATE:
        return y.astype(np.float32, copy=False)
    return librosa.resample(
        y.astype(np.float32, copy=False), orig_sr=sr, target_sr=TRAINING_SAMPLE_RATE
    ).astype(np.float32, copy=False)


def voicing_and_pitch(y16: np.ndarray):
    """
    pyin on the most recent PYIN_CONTEXT_SEC of a 16 kHz window.
    Returns (voiced_fraction, voiced_frames, display_pitch_hz or None).
    Frames whose f0 sits on the search-range edges are treated as unvoiced:
    at speech onsets pyin otherwise latches onto fmax (octave errors).
    """
    y16 = y16[-int(PYIN_CONTEXT_SEC * TRAINING_SAMPLE_RATE):]
    if len(y16) < PYIN_FRAME:
        return 0.0, 0, None
    f0, voiced_flag, _ = librosa.pyin(
        y16,
        fmin=PYIN_FMIN,
        fmax=PYIN_FMAX,
        sr=TRAINING_SAMPLE_RATE,
        frame_length=PYIN_FRAME,
        hop_length=PYIN_HOP,
        resolution=PYIN_RESOLUTION,
    )
    f0 = np.asarray(f0, dtype=float)
    voiced = (np.asarray(voiced_flag, dtype=bool) & np.isfinite(f0)
              & (f0 > PYIN_FMIN * 1.03) & (f0 < PYIN_FMAX * 0.97))
    if voiced.size == 0:
        return 0.0, 0, None
    n_voiced = int(voiced.sum())
    pitch = float(np.median(f0[voiced])) if n_voiced >= MIN_RECENT_VOICED else None
    return float(voiced.mean()), n_voiced, pitch


class SessionAudioAggregate:
    """Bounded, online per-session audio statistics (no growing arrays)."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created = time.time()
        self.last_seen = self.created
        self.hops = 0
        self.voice_hops = 0
        self.volume_sum = 0.0          # sum of rms*100 over voice-active hops
        self.signal_volume_sum = 0.0   # sum of rms*100 over any non-silent hop
        self.signal_hops = 0
        # 1 Hz histogram of per-window voiced pitch -> robust session median
        # with bounded memory (no growing arrays).
        self.pitch_hist = np.zeros(int(PYIN_FMAX) + 1, dtype=np.int64)
        self.pitch_n = 0
        self.tone_counts: Counter = Counter()
        self.tone_windows = 0
        self.clipped_hops = 0
        self.skipped_hops = 0
        self.audio_seconds = 0.0
        self.sample_rate: Optional[int] = None
        self.finalized = False

    def touch(self):
        self.last_seen = time.time()

    def add_pitch(self, hz: float):
        self.pitch_hist[int(np.clip(round(hz), 0, self.pitch_hist.size - 1))] += 1
        self.pitch_n += 1

    def pitch_median(self):
        if not self.pitch_n:
            return None
        cdf = np.cumsum(self.pitch_hist)
        return float(np.searchsorted(cdf, (self.pitch_n + 1) / 2.0))

    def summary(self) -> dict:
        if self.voice_hops:
            avg_volume = self.volume_sum / self.voice_hops
        elif self.signal_hops:
            avg_volume = self.signal_volume_sum / self.signal_hops
        else:
            avg_volume = None
        dominant = self.tone_counts.most_common(1)[0][0] if self.tone_counts else None
        return {
            "avg_volume": round(avg_volume, 3) if avg_volume is not None else None,
            # median of voiced windows (name kept for schema compatibility)
            "avg_pitch": self.pitch_median(),
            "dominant_tone": str(dominant) if dominant is not None else None,
            "tone_counts": {str(k): int(v) for k, v in self.tone_counts.items()},
            "audio_summary": {
                "analysis_hops": self.hops,
                "voice_active_hops": self.voice_hops,
                "tone_windows": self.tone_windows,
                "clipped_hops": self.clipped_hops,
                "skipped_hops": self.skipped_hops,
                "audio_seconds": round(self.audio_seconds, 2),
                "input_sample_rate": self.sample_rate,
                "analysis_sample_rate": TRAINING_SAMPLE_RATE,
                "window_sec": TRAINING_WINDOW_SEC,
                "volume_units": "RMS x 100 of the browser mic signal (uncalibrated; same scale as training)",
            },
        }


class StreamAnalyzer:
    """
    Buffers mono float audio at the client sample rate and analyses the most
    recent 1.5 s every hop. Thread-safe: `push` runs on the event loop while
    `analyze` runs in a worker thread.
    """

    def __init__(
        self,
        input_sr: int,
        aggregate: SessionAudioAggregate,
        tone_proba_fn: Optional[ToneProbaFn] = None,
        hop_sec: float = DEFAULT_HOP_SEC,
        window_sec: float = TRAINING_WINDOW_SEC,
    ):
        if not (MIN_INPUT_SAMPLE_RATE <= int(input_sr) <= MAX_INPUT_SAMPLE_RATE):
            raise ValueError(f"unsupported sample rate {input_sr}")
        self.sr = int(input_sr)
        self.window_n = int(round(window_sec * self.sr))
        self.hop_n = max(1, int(round(hop_sec * self.sr)))
        self.hop_sec = hop_sec
        self.aggregate = aggregate
        aggregate.sample_rate = self.sr
        self.tone_proba_fn = tone_proba_fn
        self._lock = threading.Lock()
        self._buf = np.zeros(0, dtype=np.float32)
        self._pending = 0          # samples received since last analysis
        self._total = 0            # samples received in this connection
        self._floor_hist: deque = deque(maxlen=40)  # ~20 s of hop RMS
        self._proba_hist: deque = deque(maxlen=TONE_SMOOTHING_WINDOWS)
        self._classes = None
        self._last_tone = None
        self._last_tone_conf = None
        self._last_tone_t = -1e9
        self._last_pitch = None
        self._last_pitch_t = -1e9
        self._vol_ema = None
        self.seq = 0

    # -- producer side ------------------------------------------------------
    def push(self, samples: np.ndarray):
        if samples.size == 0:
            return
        with self._lock:
            self._buf = np.concatenate([self._buf, samples.astype(np.float32, copy=False)])
            keep = self.window_n + self.hop_n
            if self._buf.size > keep:
                self._buf = self._buf[-keep:]
            self._pending += samples.size
            self._total += samples.size

    def has_hop(self) -> bool:
        with self._lock:
            return self._pending >= self.hop_n

    def pending_samples(self) -> int:
        with self._lock:
            return self._pending

    # -- analysis -------------------------------------------------------------
    def _gate(self) -> float:
        if not self._floor_hist:
            return SILENCE_GATE_MIN
        floor = float(np.percentile(np.asarray(self._floor_hist), 10))
        return float(np.clip(3.0 * floor, SILENCE_GATE_MIN, SILENCE_GATE_MAX))

    def analyze(self, final: bool = False) -> Optional[dict]:
        with self._lock:
            pending = self._pending
            if pending == 0:
                return None
            if not final and pending < self.hop_n:
                return None
            window = self._buf[-self.window_n:].copy()
            hop = self._buf[-min(pending, self.hop_n):].copy()
            skipped = max(0, pending // self.hop_n - 1)
            self._pending = 0
            stream_t = self._total / self.sr

        t0 = time.perf_counter()
        agg = self.aggregate
        agg.touch()
        agg.hops += 1
        agg.skipped_hops += skipped
        agg.audio_seconds += pending / self.sr

        window = window - float(np.mean(window)) if window.size else window
        hop = hop - float(np.mean(hop)) if hop.size else hop
        hop_rms = float(np.sqrt(np.mean(hop ** 2))) if hop.size else 0.0
        clip_ratio = float(np.mean(np.abs(hop) >= CLIP_LEVEL)) if hop.size else 0.0
        if clip_ratio > 0.001:
            agg.clipped_hops += 1

        gate = self._gate()
        self._floor_hist.append(hop_rms)
        full_window = window.size >= self.window_n
        y16 = resample_to_training_rate(window, self.sr)
        window_rms = float(np.sqrt(np.mean(y16 ** 2))) if y16.size else 0.0
        signal_present = window_rms >= gate

        voiced_frac, recent_voiced, pitch = (0.0, 0, None)
        if hop_rms >= gate:
            try:
                voiced_frac, recent_voiced, pitch = voicing_and_pitch(y16)
            except Exception as e:  # never let pitch tracking kill the stream
                print(f"⚠️ pyin failed: {e}")
        voice_active = hop_rms >= gate and recent_voiced >= MIN_RECENT_VOICED

        volume_raw = hop_rms * 100.0
        if hop_rms >= gate:
            agg.signal_hops += 1
            agg.signal_volume_sum += volume_raw
        if voice_active:
            agg.voice_hops += 1
            agg.volume_sum += volume_raw
        self._vol_ema = volume_raw if self._vol_ema is None else 0.5 * self._vol_ema + 0.5 * volume_raw

        if pitch is not None:
            agg.add_pitch(pitch)
            self._last_pitch, self._last_pitch_t = pitch, stream_t

        # --- RF tone on a training-length window -----------------------------
        tone, tone_conf, tone_raw, status = None, None, None, "ok"
        if not full_window:
            status = "warming_up"
        elif not signal_present:
            status = "silence"
        elif self.tone_proba_fn is None:
            status = "model_unavailable"
        else:
            feats = extract_features_from_waveform(y16, TRAINING_SAMPLE_RATE)
            classes, proba = self.tone_proba_fn(feats)
            if classes is None:
                status = "model_unavailable"
            else:
                self._classes = list(classes)
                proba = np.asarray(proba, dtype=float)
                tone_raw = str(self._classes[int(np.argmax(proba))])
                agg.tone_counts[tone_raw] += 1
                agg.tone_windows += 1
                self._proba_hist.append(proba)
                mean_p = np.mean(np.stack(self._proba_hist), axis=0)
                idx = int(np.argmax(mean_p))
                tone, tone_conf = str(self._classes[idx]), float(mean_p[idx])
                self._last_tone, self._last_tone_conf, self._last_tone_t = tone, tone_conf, stream_t

        if tone is None and status == "silence":
            if stream_t - self._last_tone_t <= TONE_HOLD_SEC and self._last_tone:
                tone, tone_conf, status = self._last_tone, self._last_tone_conf, "holding"
            else:
                self._proba_hist.clear()
        display_pitch = pitch
        if display_pitch is None and stream_t - self._last_pitch_t <= PITCH_HOLD_SEC:
            display_pitch = self._last_pitch

        self.seq += 1
        return {
            "type": "metrics",
            "session_id": agg.session_id,
            "seq": self.seq,
            "t": round(stream_t, 2),
            "status": status,
            "volume": round(self._vol_ema, 2),
            "volume_raw": round(volume_raw, 2),
            "pitch": round(display_pitch, 1) if display_pitch is not None else None,
            "tone": tone,
            "tone_raw": tone_raw,
            "tone_confidence": round(tone_conf, 3) if tone_conf is not None else None,
            "voice_active": bool(voice_active),
            "voiced_fraction": round(voiced_frac, 3),
            "clipping": clip_ratio > 0.001,
            "silence_gate": round(gate * 100, 3),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }


def warm_up():
    """pyin/numba compile on first call (~1 s); do it at startup, not mid-session."""
    y = (0.05 * np.sin(2 * np.pi * 150 * np.arange(int(TRAINING_SAMPLE_RATE * TRAINING_WINDOW_SEC))
                       / TRAINING_SAMPLE_RATE)).astype(np.float32)
    voicing_and_pitch(y)
    extract_features_from_waveform(y, TRAINING_SAMPLE_RATE)
    resample_to_training_rate(np.zeros(4800, dtype=np.float32), 48000)


__all__ = [
    "FEATURE_NAMES",
    "TRAINING_SAMPLE_RATE",
    "TRAINING_WINDOW_SEC",
    "DEFAULT_HOP_SEC",
    "decode_pcm16le",
    "resample_to_training_rate",
    "voicing_and_pitch",
    "SessionAudioAggregate",
    "StreamAnalyzer",
    "warm_up",
]
