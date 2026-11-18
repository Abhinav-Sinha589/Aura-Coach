import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def voice_like(f0: float, sr: int, dur: float, amp: float = 0.08, seed: int = 0) -> np.ndarray:
    """Deterministic harmonic test signal (NOT training data): f0 + 3 harmonics,
    slow amplitude envelope, light noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * dur)) / sr
    y = sum((1.0 / h) * np.sin(2 * np.pi * h * f0 * t) for h in range(1, 4))
    env = 0.7 + 0.3 * np.sin(2 * np.pi * 2.0 * t)
    y = amp * y * env / 1.8 + rng.normal(0, amp * 0.01, t.size)
    return y.astype(np.float32)


def to_pcm16(y: np.ndarray) -> bytes:
    return (np.clip(y, -1, 1) * 32767).astype("<i2").tobytes()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import backend.main as main

    raw = tmp_path / "raw"
    raw.mkdir()
    monkeypatch.setattr(main, "RAW_DIR", str(raw))
    main.SESSION_AUDIO_DATA.clear()
    with TestClient(main.app) as c:
        c.raw_dir = str(raw)
        yield c
