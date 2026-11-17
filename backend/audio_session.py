"""
audio_session.py
----------------
Pure (framework-independent) helpers for the live audio WebSocket:

* validate the client's ``start`` handshake (session id, sample rate, ...)
* decode little-endian PCM16 mono transport packets
* buffer packets into the SAME analysis window the tone model was trained on
  (``train_tone_model.py``: SR = 16000, DURATION = 1.5 s) with a 0.5 s hop
* resample each *complete* window to 16 kHz (``scipy.signal.resample_poly``)
* turn a window into honest metrics (silence -> no pitch / no tone)
* keep bounded, per-session running aggregates (no unbounded lists)

Kept separate from ``main.py`` so it can be unit tested without a server.
Nothing here trains, modifies or re-fits a model.
"""
from __future__ import annotations

import json
import math
import re
import threading
import time
from dataclasses import dataclass, field
from math import gcd
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import resample_poly

try:
    from backend.ml.audio_features import extract_features_from_waveform
except ImportError:  # running from inside backend/
    from ml.audio_features import extract_features_from_waveform

# --- Contract with backend/ml/train_tone_model.py (verified: SR=16000, DURATION=1.5) ---
MODEL_SR = 16000
WINDOW_SECONDS = 1.5
HOP_SECONDS = 0.5
MODEL_WINDOW_SAMPLES = int(MODEL_SR * WINDOW_SECONDS)  # 24000

# --- Transport / safety limits ---
MIN_SAMPLE_RATE = 8000
MAX_SAMPLE_RATE = 96000
MAX_CONTROL_MESSAGE_BYTES = 2048
MAX_AUDIO_PACKET_BYTES = 256 * 1024  # 1.36 s of 96 kHz PCM16 mono
MAX_WINDOWS_PER_PACKET = 4  # bounded work per packet (latest windows win)

# Below this RMS (on a [-1, 1] scale) a window is treated as silence:
# volume is still reported, but pitch / tone are not (they'd be guesses).
# Same value as the pre-existing noise gate in main.py.
SILENCE_RMS = 0.001

UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
# Older sessions on disk were saved as ``session_<7 base36 chars>`` or
# ``session_<32 hex>``. They may be READ, never created by new clients.
LEGACY_ID_RE = re.compile(r"^session_[a-z0-9]{1,64}$")


def is_new_session_id(value) -> bool:
    return isinstance(value, str) and bool(UUID_RE.match(value))


def is_readable_session_id(value) -> bool:
    return isinstance(value, str) and bool(UUID_RE.match(value) or LEGACY_ID_RE.match(value))


class HandshakeError(ValueError):
    pass


@dataclass(frozen=True)
class StartConfig:
    session_id: str
    sample_rate: int
    channels: int = 1
    encoding: str = "pcm_s16le"


def parse_start_message(text) -> StartConfig:
    """Validate ``{type:'start', session_id, sample_rate, channels, encoding}``."""
    if not isinstance(text, str) or not text:
        raise HandshakeError("expected a JSON text 'start' message")
    if len(text.encode("utf-8")) > MAX_CONTROL_MESSAGE_BYTES:
        raise HandshakeError("control message too large")
    try:
        msg = json.loads(text)
    except (ValueError, TypeError):
        raise HandshakeError("control message is not valid JSON")
    if not isinstance(msg, dict) or msg.get("type") != "start":
        raise HandshakeError("first message must have type 'start'")

    sid = msg.get("session_id")
    if not is_new_session_id(sid):
        raise HandshakeError("session_id must be a lowercase canonical UUID")

    sr = msg.get("sample_rate")
    if isinstance(sr, bool) or not isinstance(sr, (int, float)) or not math.isfinite(sr) or int(sr) != sr:
        raise HandshakeError("sample_rate must be an integer")
    sr = int(sr)
    if not (MIN_SAMPLE_RATE <= sr <= MAX_SAMPLE_RATE):
        raise HandshakeError(f"sample_rate must be within {MIN_SAMPLE_RATE}-{MAX_SAMPLE_RATE} Hz")

    channels = msg.get("channels", 1)
    if channels != 1 or isinstance(channels, bool):
        raise HandshakeError("only mono (channels=1) audio is supported")

    encoding = msg.get("encoding", "pcm_s16le")
    if encoding != "pcm_s16le":
        raise HandshakeError("only encoding 'pcm_s16le' is supported")

    return StartConfig(session_id=sid, sample_rate=sr, channels=1, encoding=encoding)


def decode_pcm16le(data: bytes) -> np.ndarray:
    """Little-endian signed 16-bit mono -> float32 in [-1, 1)."""
    if len(data) % 2:
        raise ValueError("PCM16 payload must have an even number of bytes")
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def resample_to_model_rate(y: np.ndarray, input_sr: int) -> np.ndarray:
    """Resample a complete analysis window to MODEL_SR with a reduced integer ratio."""
    y = np.asarray(y, dtype=np.float32)
    if input_sr == MODEL_SR:
        out = y
    else:
        g = gcd(int(input_sr), MODEL_SR)
        out = resample_poly(y, MODEL_SR // g, int(input_sr) // g).astype(np.float32)
    # Guarantee the exact trained window length (rounding can be off by 1).
    if len(out) > MODEL_WINDOW_SAMPLES:
        out = out[-MODEL_WINDOW_SAMPLES:]
    elif len(out) < MODEL_WINDOW_SAMPLES:
        out = np.pad(out, (MODEL_WINDOW_SAMPLES - len(out), 0))
    return out


class WindowBuffer:
    """
    Bounded rolling buffer at the client's native sample rate.

    Transport packet size is independent of the analysis window: a window is
    emitted only once 1.5 s of audio exists, then every 0.5 s (hop) after that.
    Memory never exceeds one window + one packet.
    """

    def __init__(self, input_sr: int):
        self.input_sr = int(input_sr)
        self.window_len = int(round(self.input_sr * WINDOW_SECONDS))
        self.hop_len = int(round(self.input_sr * HOP_SECONDS))
        self._buf = np.zeros(0, dtype=np.float32)
        self._since_last = 0  # samples received since the last emitted window
        self._emitted_any = False
        self.total_samples = 0

    def push(self, samples: np.ndarray) -> List[np.ndarray]:
        samples = np.asarray(samples, dtype=np.float32)
        self.total_samples += len(samples)
        windows: List[np.ndarray] = []
        # Process in hop-sized pieces so a large packet yields correctly spaced windows.
        pos = 0
        while pos < len(samples):
            if not self._emitted_any:
                need = self.window_len - len(self._buf)
            else:
                need = self.hop_len - self._since_last
            take = min(max(need, 1), len(samples) - pos)
            chunk = samples[pos:pos + take]
            pos += take
            self._buf = np.concatenate([self._buf, chunk])[-self.window_len:]
            self._since_last += take
            ready = (len(self._buf) >= self.window_len) and (
                not self._emitted_any or self._since_last >= self.hop_len
            )
            if ready:
                windows.append(self._buf.copy())
                self._emitted_any = True
                self._since_last = 0
        if len(windows) > MAX_WINDOWS_PER_PACKET:
            windows = windows[-MAX_WINDOWS_PER_PACKET:]
        return windows

    def flush(self) -> Optional[np.ndarray]:
        """Final window on 'finish': only if there is unanalysed audio AND a full window exists."""
        if self._since_last > 0 and len(self._buf) >= self.window_len:
            self._since_last = 0
            self._emitted_any = True
            return self._buf.copy()
        return None


ToneFn = Callable[[np.ndarray], Tuple[Optional[str], Optional[float]]]


def analyze_window(window_native: np.ndarray, input_sr: int, predict_tone: ToneFn) -> Dict:
    """
    One complete window -> metrics dict. Heavy (librosa) – call via asyncio.to_thread.

    ``volume`` keeps the historical scale (RMS * 100) so saved sessions remain
    comparable. ``pitch``/``tone`` are ``None`` for silence or when the tone
    model is unavailable – never a rule-based guess presented as a prediction.
    """
    y = resample_to_model_rate(window_native, input_sr)
    features = extract_features_from_waveform(y, MODEL_SR)
    rms = float(features[0])
    pitch = float(features[1])
    voiced = rms >= SILENCE_RMS and math.isfinite(pitch) and pitch > 0
    result = {
        "volume": round(rms * 100.0, 3) if math.isfinite(rms) else None,
        "pitch": None,
        "tone": None,
        "tone_confidence": None,
        "voiced": bool(voiced),
        "tone_source": None,
    }
    if not voiced:
        return result
    result["pitch"] = round(pitch, 1)
    tone, conf = predict_tone(features)
    if tone is not None:
        result["tone"] = str(tone)
        result["tone_confidence"] = round(float(conf), 3) if conf is not None else None
        result["tone_source"] = "random_forest"
    return result


@dataclass
class AudioAggregate:
    """Online, O(1)-memory session statistics."""

    windows: int = 0
    volume_sum: float = 0.0
    volume_count: int = 0
    pitch_sum: float = 0.0
    pitch_count: int = 0
    tone_counts: Dict[str, int] = field(default_factory=dict)

    def add(self, m: Dict) -> None:
        self.windows += 1
        v = m.get("volume")
        if isinstance(v, (int, float)) and math.isfinite(v):
            self.volume_sum += float(v)
            self.volume_count += 1
        p = m.get("pitch")
        if isinstance(p, (int, float)) and math.isfinite(p) and p > 0:
            self.pitch_sum += float(p)
            self.pitch_count += 1
        t = m.get("tone")
        if isinstance(t, str) and len(self.tone_counts) < 16:
            self.tone_counts[t] = self.tone_counts.get(t, 0) + 1
        elif isinstance(t, str) and t in self.tone_counts:
            self.tone_counts[t] += 1

    def to_dict(self) -> Dict:
        dominant = None
        if self.tone_counts:
            dominant = sorted(self.tone_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        return {
            "audio_windows": self.windows,
            "avg_volume": round(self.volume_sum / self.volume_count, 3) if self.volume_count else None,
            "avg_pitch": round(self.pitch_sum / self.pitch_count, 1) if self.pitch_count else None,
            "voiced_windows": self.pitch_count,
            "dominant_tone": dominant,
            "tone_counts": dict(self.tone_counts),
        }


class AudioSummaryStore:
    """
    Bounded, TTL'd hand-off of finished audio aggregates from the WebSocket to
    ``POST /api/session/log``. In-process only: NOT shared across multiple
    Uvicorn workers and lost on restart (documented limitation).
    """

    def __init__(self, ttl_seconds: float = 1800.0, max_entries: int = 500, clock=time.monotonic):
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self._clock = clock
        self._data: Dict[str, Tuple[float, Dict]] = {}
        self._lock = threading.Lock()

    def _prune(self) -> None:
        now = self._clock()
        for k in [k for k, (ts, _) in self._data.items() if now - ts > self.ttl]:
            del self._data[k]
        while len(self._data) > self.max_entries:
            oldest = min(self._data.items(), key=lambda kv: kv[1][0])[0]
            del self._data[oldest]

    def put(self, session_id: str, summary: Dict) -> None:
        with self._lock:
            self._data[session_id] = (self._clock(), dict(summary))
            self._prune()

    def get(self, session_id: str) -> Optional[Dict]:
        with self._lock:
            self._prune()
            item = self._data.get(session_id)
            return dict(item[1]) if item else None

    def pop(self, session_id: str) -> None:
        with self._lock:
            self._data.pop(session_id, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)
