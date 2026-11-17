import hashlib
import json
import math
import pathlib
import sys
import uuid

import numpy as np
import pytest
from fastapi.testclient import TestClient

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "backend"))

from backend import audio_session as aud
from backend import main as app_main
from backend import model_loader
from backend.ml import train_performance_model, train_tone_model
PERF_MODEL_PATH = REPO_ROOT / "backend" / "models" / "performance_model.joblib"
TONE_MODEL_PATH = REPO_ROOT / "backend" / "models" / "tone_model.joblib"

EXPECTED_PERF_SHA256 = "527192f37e2ddd673d1000794cae563d8cbf0a2c2fc4372d68afd88fefc77c2c"
EXPECTED_TONE_SHA256 = "0c6f37019f55e11e286aca4cc664b7c15528d9f4638f8f1d7a75265361b02f14"


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_pcm16_sine(freq_hz: float, sr: int, duration_s: float, amp: float = 0.2) -> bytes:
    n = int(round(sr * duration_s))
    t = np.arange(n, dtype=np.float64) / sr
    y = amp * np.sin(2.0 * math.pi * freq_hz * t)
    pcm = np.clip(np.round(y * 32767.0), -32768, 32767).astype("<i2")
    return pcm.tobytes()


def _make_pcm16_silence(sr: int, duration_s: float) -> bytes:
    n = int(round(sr * duration_s))
    return np.zeros(n, dtype="<i2").tobytes()


@pytest.fixture(autouse=True)
def isolated_raw_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(app_main, "RAW_DIR", str(tmp_path))
    app_main.AUDIO_SUMMARIES = aud.AudioSummaryStore()
    model_loader.load_models()
    yield tmp_path


# ---------------------------------------------------------------------------
# P1: Model binary integrity & training-contract parity
# ---------------------------------------------------------------------------
def test_model_binaries_unchanged_and_load_cleanly():
    assert _sha256(PERF_MODEL_PATH) == EXPECTED_PERF_SHA256
    assert _sha256(TONE_MODEL_PATH) == EXPECTED_TONE_SHA256
    status = model_loader.models_status()
    assert status == {"performance_model_loaded": True, "tone_model_loaded": True}


def test_training_tone_window_contract_matches_audio_session():
    assert aud.MODEL_SR == train_tone_model.SR == 16000
    assert aud.WINDOW_SECONDS == train_tone_model.DURATION == 1.5
    assert aud.MODEL_WINDOW_SAMPLES == 24000


def test_performance_preprocessing_matches_training_clean_sessions():
    assert model_loader.NUMERIC_FEATURES == train_performance_model.NUMERIC_FEATURES
    assert model_loader.CATEGORICAL_FEATURES == train_performance_model.CATEGORICAL_FEATURES
    assert model_loader.CLIP_RANGES == {
        "wpm": (0, 260),
        "posture": (0, 100),
        "fillers": (0, 50),
        "volume": (0, 5),
        "pitch": (0, 500),
    }

    # Outliers + unknown categories should clip & encode without error
    score = model_loader.predict_performance_score({
        "wpm": 999,
        "posture": 150,
        "fillers": -5,
        "volume": 99.0,
        "pitch": 900.0,
        "eyeContact": "UnseenCategory",
        "tone": "UnseenTone",
        "emotion": "Detecting...",
    })
    assert score is not None
    assert 0.0 <= score <= 100.0

    # Essential inputs (wpm, posture) missing -> score withheld (None), never fabricated
    assert model_loader.predict_performance_score({"wpm": None, "posture": 80}) is None
    assert model_loader.predict_performance_score({"wpm": 130, "posture": None}) is None


# ---------------------------------------------------------------------------
# P0-A: Sample rate mismatch, 1.5s windowing, resampling, silence handling
# ---------------------------------------------------------------------------
def test_180hz_sine_at_16khz_and_48khz_yields_comparable_pitch_only_after_full_window():
    for sr in (16000, 48000):
        buf = aud.WindowBuffer(sr)
        # Send 2.0 seconds in small 2048-sample transport frames
        pcm = _make_pcm16_sine(180.0, sr, 2.0, amp=0.2)
        samples = aud.decode_pcm16le(pcm)
        frame_size = 2048
        emitted_at_samples = []
        metrics_list = []

        for start in range(0, len(samples), frame_size):
            chunk = samples[start:start + frame_size]
            windows = buf.push(chunk)
            for w in windows:
                emitted_at_samples.append(buf.total_samples)
                assert len(w) == int(round(sr * 1.5))
                m = aud.analyze_window(w, sr, model_loader.predict_tone)
                metrics_list.append(m)

        # First window must not be emitted before 1.5s (1.5 * sr samples)
        assert emitted_at_samples[0] >= int(round(sr * 1.5))
        # Across 2.0s (1.5s window + 0.5s hop) we get 2 windows
        assert len(metrics_list) == 2
        for m in metrics_list:
            assert m["voiced"] is True
            assert m["pitch"] is not None
            assert abs(m["pitch"] - 180.0) <= 3.0
            assert m["tone"] in ("Calm", "Balanced", "Energetic", "Noise")
            assert m["tone_source"] == "random_forest"


def test_silence_returns_volume_but_null_pitch_and_tone():
    buf = aud.WindowBuffer(48000)
    samples = aud.decode_pcm16le(_make_pcm16_silence(48000, 1.6))
    windows = buf.push(samples)
    assert len(windows) == 1
    m = aud.analyze_window(windows[0], 48000, model_loader.predict_tone)
    assert m["volume"] == 0.0
    assert m["voiced"] is False
    assert m["pitch"] is None
    assert m["tone"] is None
    assert m["tone_confidence"] is None


def test_unavailable_tone_model_returns_null_not_rms_fallback():
    buf = aud.WindowBuffer(16000)
    samples = aud.decode_pcm16le(_make_pcm16_sine(180.0, 16000, 1.5, amp=0.2))
    windows = buf.push(samples)
    m = aud.analyze_window(windows[0], 16000, lambda _feats: (None, None))
    assert m["voiced"] is True
    assert abs(m["pitch"] - 180.0) <= 3.0
    assert m["tone"] is None
    assert m["tone_source"] is None


# ---------------------------------------------------------------------------
# P0-A & P0-D: WebSocket handshake, streaming, finish ACK, isolation
# ---------------------------------------------------------------------------
def test_websocket_rejects_invalid_handshakes_and_binary_before_start():
    client = TestClient(app_main.app)

    # 1. Binary sent before start handshake -> error + close
    with client.websocket_connect("/api/audio-stream") as ws:
        ws.send_bytes(b"\x00\x00" * 100)
        msg = ws.receive_json()
        assert msg["type"] == "error"

    # 2. Invalid sample rate -> error
    with client.websocket_connect("/api/audio-stream") as ws:
        ws.send_text(json.dumps({
            "type": "start",
            "session_id": str(uuid.uuid4()),
            "sample_rate": 4000,
            "channels": 1,
            "encoding": "pcm_s16le",
        }))
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert "sample_rate" in msg["error"]

    # 3. Non-UUID session_id -> error
    with client.websocket_connect("/api/audio-stream") as ws:
        ws.send_text(json.dumps({
            "type": "start",
            "session_id": "session_abc1234",
            "sample_rate": 48000,
            "channels": 1,
            "encoding": "pcm_s16le",
        }))
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert "session_id" in msg["error"]


def test_websocket_full_flow_48khz_and_finish_merges_into_session_log():
    client = TestClient(app_main.app)
    sid = str(uuid.uuid4())

    with client.websocket_connect("/api/audio-stream") as ws:
        ws.send_text(json.dumps({
            "type": "start",
            "session_id": sid,
            "sample_rate": 48000,
            "channels": 1,
            "encoding": "pcm_s16le",
        }))
        ready = ws.receive_json()
        assert ready == {
            "type": "ready",
            "session_id": sid,
            "analysis_window_ms": 1500,
            "hop_ms": 500,
        }

        # Stream 1.6s of 180 Hz at 48 kHz in 0.4s packets
        packet = _make_pcm16_sine(180.0, 48000, 0.4, amp=0.2)
        received_metrics = []
        for i in range(4):
            ws.send_bytes(packet)
            if i < 3:
                # 0.4, 0.8, 1.2s -> < 1.5s, no metrics yet
                pass
            else:
                # 1.6s -> first 1.5s window emitted
                m = ws.receive_json()
                received_metrics.append(m)

        assert len(received_metrics) == 1
        assert received_metrics[0]["type"] == "audio_metrics"
        assert received_metrics[0]["session_id"] == sid
        assert abs(received_metrics[0]["pitch"] - 180.0) <= 3.0

        # Send finish -> flushes tail and returns 'finished' with aggregates
        ws.send_text(json.dumps({"type": "finish"}))
        tail_or_fin = ws.receive_json()
        if tail_or_fin["type"] == "audio_metrics":
            fin = ws.receive_json()
        else:
            fin = tail_or_fin
        assert fin["type"] == "finished"
        assert fin["session_id"] == sid
        assert fin["aggregates"]["audio_finalized"] is True
        assert fin["aggregates"]["voiced_windows"] >= 1

    # Now POST /api/session/log with the same UUID
    payload = {
        "session_id": sid,
        "wpm": 138,
        "word_count": 46,
        "capture_seconds": 20.0,
        "fillers": 2,
        "posture": 84,
        "shoulder_tilt": -2.0,
        "eyeContact": "Good",
        "emotion": "Happy",
        "timeline": [
            {"t": 5.0, "posture": 82.0, "wpm": 130.0},
            {"t": 15.0, "posture": 86.0, "wpm": 138.0},
        ],
    }
    res = client.post("/api/session/log", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["session_id"] == sid
    assert body["duplicate"] is False

    # Verify report uses true WPM (138), merged audio pitch (~180), and recorded timeline
    rep_res = client.get(f"/api/report/{sid}")
    assert rep_res.status_code == 200
    rep = rep_res.json()["report"]
    assert rep["metrics"]["wpm"] == 138.0
    assert rep["metrics"]["wpm_source"] == "speech_recognition"
    assert abs(rep["metrics"]["pitch"] - 180.0) <= 3.0
    assert rep["metrics"]["tone"] is not None
    assert len(rep["timeline"]) == 2
    assert rep["ml_performance_score"] is not None


# ---------------------------------------------------------------------------
# P0-D: Idempotency, 409 conflict, path traversal rejection
# ---------------------------------------------------------------------------
def test_session_log_idempotency_and_conflict():
    client = TestClient(app_main.app)
    sid = str(uuid.uuid4())
    payload = {"session_id": sid, "wpm": 125, "posture": 78, "fillers": 1}

    r1 = client.post("/api/session/log", json=payload)
    assert r1.status_code == 200
    assert r1.json()["duplicate"] is False

    saved_path = pathlib.Path(app_main.RAW_DIR) / f"{sid}.json"
    first_ts = json.loads(saved_path.read_text(encoding="utf-8"))["timestamp"]

    # Identical retry -> 200 duplicate=True, timestamp unchanged
    r2 = client.post("/api/session/log", json=payload)
    assert r2.status_code == 200
    assert r2.json()["duplicate"] is True
    second_ts = json.loads(saved_path.read_text(encoding="utf-8"))["timestamp"]
    assert second_ts == first_ts

    # Conflicting payload for same session_id -> 409, does not overwrite
    r3 = client.post("/api/session/log", json={**payload, "wpm": 160})
    assert r3.status_code == 409
    assert json.loads(saved_path.read_text(encoding="utf-8"))["wpm"] == 125.0


@pytest.mark.parametrize("bad_id", [
    "../evil",
    "..\\evil",
    "session_abc/../../etc/passwd",
    "session_1234567",  # legacy non-UUID not allowed for new POSTs
    "not-a-uuid",
    "",
])
def test_malicious_or_non_uuid_session_ids_rejected_on_post(bad_id):
    client = TestClient(app_main.app)
    res = client.post("/api/session/log", json={"session_id": bad_id, "wpm": 120})
    assert res.status_code == 422


def test_report_404_and_legacy_v1_compatibility(isolated_raw_dir):
    client = TestClient(app_main.app)

    # Missing UUID -> 404
    missing_id = str(uuid.uuid4())
    assert client.get(f"/api/report/{missing_id}").status_code == 404

    # Path traversal in report URL -> 422/404 (never reads outside RAW_DIR)
    assert client.get("/api/report/..%5Cevil").status_code in (404, 422)

    # Write a legacy v1 file (session_abc1234.json) with no schema_version and no timeline
    legacy_id = "session_abc1234"
    legacy_file = isolated_raw_dir / f"{legacy_id}.json"
    legacy_file.write_text(json.dumps({
        "session_id": legacy_id,
        "timestamp": "2026-01-01T10:00:00",
        "wpm": 142,
        "avg_wpm": 124,  # obsolete loudness heuristic in old files
        "posture": 88,
        "fillers": 3,
        "eyeContact": "Good",
        "emotion": "Neutral",
        "avg_volume": 1.2,
        "avg_pitch": 165.0,
        "dominant_tone": "Balanced",
    }), encoding="utf-8")

    res = client.get(f"/api/report/{legacy_id}")
    assert res.status_code == 200
    rep = res.json()["report"]
    # Prefers true speech-recognition wpm (142) over obsolete avg_wpm (124)
    assert rep["metrics"]["wpm"] == 142.0
    assert rep["timeline"] == []
    assert rep["ml_performance_score"] is not None

    # Also appears in /api/sessions while skipping invalid/corrupt files
    (isolated_raw_dir / "corrupt.json").write_text("{bad json", encoding="utf-8")
    (isolated_raw_dir / "evil_name!.json").write_text("{}", encoding="utf-8")
    sessions_res = client.get("/api/sessions")
    assert sessions_res.status_code == 200
    ids = [s["session_id"] for s in sessions_res.json()["sessions"]]
    assert ids == [legacy_id]


def test_audio_summary_store_ttl_and_bounds():
    now = [100.0]
    store = aud.AudioSummaryStore(ttl_seconds=10.0, max_entries=3, clock=lambda: now[0])
    store.put("s1", {"v": 1})
    store.put("s2", {"v": 2})
    store.put("s3", {"v": 3})
    store.put("s4", {"v": 4})
    assert len(store) == 3
    assert store.get("s1") is None
    now[0] = 115.0
    assert store.get("s4") is None


def test_two_concurrent_sessions_have_isolated_aggregates():
    client = TestClient(app_main.app)
    sid1 = str(uuid.uuid4())
    sid2 = str(uuid.uuid4())

    with client.websocket_connect("/api/audio-stream") as ws1, client.websocket_connect("/api/audio-stream") as ws2:
        ws1.send_text(json.dumps({"type": "start", "session_id": sid1, "sample_rate": 16000}))
        ws2.send_text(json.dumps({"type": "start", "session_id": sid2, "sample_rate": 16000}))
        assert ws1.receive_json()["session_id"] == sid1
        assert ws2.receive_json()["session_id"] == sid2

        # Session 1 gets 1.5s of 180 Hz sine; Session 2 gets 1.5s of silence
        ws1.send_bytes(_make_pcm16_sine(180.0, 16000, 1.5, amp=0.2))
        ws2.send_bytes(_make_pcm16_silence(16000, 1.5))

        m1 = ws1.receive_json()
        m2 = ws2.receive_json()
        assert m1["session_id"] == sid1 and m1["voiced"] is True and m1["pitch"] is not None
        assert m2["session_id"] == sid2 and m2["voiced"] is False and m2["pitch"] is None

        ws1.send_text(json.dumps({"type": "finish"}))
        ws2.send_text(json.dumps({"type": "finish"}))
        fin1 = ws1.receive_json()
        fin2 = ws2.receive_json()
        assert fin1["type"] == "finished" and fin1["aggregates"]["voiced_windows"] == 1
        assert fin2["type"] == "finished" and fin2["aggregates"]["voiced_windows"] == 0

