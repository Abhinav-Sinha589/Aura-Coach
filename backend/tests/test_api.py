import hashlib
import json
import os
import warnings

import numpy as np
import pandas as pd
import pytest

from backend.tests.conftest import voice_like, to_pcm16

MODELS = os.path.join(os.path.dirname(__file__), "..", "models")
EXPECTED_SHA256 = {
    "performance_model.joblib": "527192f37e2ddd673d1000794cae563d8cbf0a2c2fc4372d68afd88fefc77c2c",
    "tone_model.joblib": "0c6f37019f55e11e286aca4cc664b7c15528d9f4638f8f1d7a75265361b02f14",
}


# ---------------------------------------------------------------- models ----
def test_model_artifacts_unchanged():
    for name, digest in EXPECTED_SHA256.items():
        with open(os.path.join(MODELS, name), "rb") as f:
            assert hashlib.sha256(f.read()).hexdigest() == digest, name


def test_models_load_without_version_warnings_and_keep_contract():
    import joblib
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        perf = joblib.load(os.path.join(MODELS, "performance_model.joblib"))
        tone = joblib.load(os.path.join(MODELS, "tone_model.joblib"))
    assert list(tone.classes_) == ["Balanced", "Calm", "Energetic", "Noise"]
    assert tone.n_features_in_ == 5
    assert list(perf.named_steps["preprocess"].feature_names_in_) == [
        "wpm", "posture", "fillers", "volume", "pitch", "eyeContact", "tone", "emotion"]


def _original_clean_sessions(df):
    """Verbatim copy of the function that trained the model (pre-refactor)."""
    df = df.copy()
    for col in ["posture", "wpm", "fillers", "volume", "pitch"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    if "avg_wpm" in df.columns:
        df["wpm"] = pd.to_numeric(df["avg_wpm"], errors="coerce").fillna(df["wpm"])
    if "avg_volume" in df.columns:
        df["volume"] = pd.to_numeric(df["avg_volume"], errors="coerce").fillna(df["volume"])
    if "avg_pitch" in df.columns:
        df["pitch"] = pd.to_numeric(df["avg_pitch"], errors="coerce").fillna(df["pitch"])
    if "dominant_tone" in df.columns:
        df["tone"] = df["dominant_tone"].fillna(df.get("tone", "Neutral"))
    df["wpm"] = df["wpm"].clip(0, 260)
    df["posture"] = df["posture"].clip(0, 100)
    df["fillers"] = df["fillers"].clip(0, 50)
    df["volume"] = df["volume"].clip(0, 5)
    df["pitch"] = df["pitch"].clip(0, 500)
    if "eyeContact" not in df.columns:
        df["eyeContact"] = "Unknown"
    df["eyeContact"] = df["eyeContact"].fillna("Unknown").replace({"": "Unknown"})
    if "emotion" not in df.columns:
        df["emotion"] = "Neutral"
    df["emotion"] = df["emotion"].fillna("Neutral").replace({"Detecting...": "Neutral", "": "Neutral"})
    if "tone" not in df.columns:
        df["tone"] = "Neutral"
    df["tone"] = df["tone"].fillna("Neutral").replace({"": "Neutral"})
    return df


def test_shared_preprocessing_is_identical_to_training():
    from backend.ml.session_features import clean_sessions, ALL_FEATURES
    rows = [
        {"wpm": "1601", "posture": "88", "fillers": 3, "eyeContact": "Good", "emotion": "Detecting..."},
        {"wpm": 140, "avg_wpm": 131.2, "avg_volume": 9.3, "avg_pitch": 0.0, "dominant_tone": "Calm",
         "posture": 40, "eyeContact": "", "emotion": "Happy", "tone": "Neutral"},
        {"posture": None, "dominant_tone": None, "tone": "Energetic"},
        {"wpm": None, "fillers": "x", "volume": 0.4, "pitch": 900},
    ]
    df = pd.DataFrame(rows)
    pd.testing.assert_frame_equal(clean_sessions(df)[ALL_FEATURES], _original_clean_sessions(df)[ALL_FEATURES])


# ------------------------------------------------------------- websocket ----
def stream_session(client, sid, sr=48000, seconds=3.0, f0=180, chunk=4096):
    y = voice_like(f0, sr, seconds)
    metrics = []
    with client.websocket_connect("/api/audio-stream") as ws:
        ws.send_text(json.dumps({"type": "start", "session_id": sid, "sample_rate": sr,
                                 "channels": 1, "encoding": "pcm_s16le"}))
        ready = ws.receive_json()
        assert ready["type"] == "ready" and ready["input_sample_rate"] == sr
        for i in range(0, len(y), chunk):
            ws.send_bytes(to_pcm16(y[i:i + chunk]))
        ws.send_text(json.dumps({"type": "stop"}))
        while True:
            msg = ws.receive_json()
            if msg["type"] == "final":
                return metrics, msg
            metrics.append(msg)


def test_websocket_full_session_flow(client):
    sid = "session_0123456789abcdef0123456789abcdef"
    metrics, final = stream_session(client, sid)
    assert final["session_id"] == sid
    summary = final["summary"]
    assert abs(summary["avg_pitch"] - 180) < 4
    assert summary["audio_summary"]["input_sample_rate"] == 48000
    # trailing audio is flushed: ~3 s streamed, ~3 s accounted for
    assert summary["audio_summary"]["audio_seconds"] == pytest.approx(3.0, abs=0.1)
    assert any(m["tone"] for m in metrics)

    payload = {"session_id": sid, "wpm": 132, "fillers": 2, "posture": 81, "headTilt": -3.2,
               "eyeContact": "Good", "eye_contact_ratio": 0.74, "emotion": "Happy",
               "measured": {"speech": True, "audio": True, "pose": True, "face": True},
               "trend": {"t": [5, 10, 15], "posture": [80, 82, 81], "wpm": [120, 130, 132]},
               "avg_pitch": 999}  # server-owned key from client must be ignored
    r = client.post("/api/session/log", json=payload)
    assert r.status_code == 200 and r.json()["duplicate"] is False
    files = os.listdir(client.raw_dir)
    assert files == [f"{sid}.json"]
    saved = json.load(open(os.path.join(client.raw_dir, files[0])))
    assert abs(saved["avg_pitch"] - 180) < 4 and saved["avg_wpm"] == 132

    rep = client.get(f"/api/report/{sid}").json()["report"]
    assert rep["metrics"]["wpm"] == 132 and abs(rep["metrics"]["pitch"] - 180) < 4
    assert rep["metrics"]["volume"] is not None and rep["metrics"]["tone"] is not None
    assert rep["trend"]["t"] == [5, 10, 15]
    assert 0 <= rep["ml_performance_score"] <= 100

    # duplicate POST (retry / double click) -> same single file, flagged
    r2 = client.post("/api/session/log", json=payload)
    assert r2.json()["duplicate"] is True
    assert os.listdir(client.raw_dir) == [f"{sid}.json"]
    again = json.load(open(os.path.join(client.raw_dir, f"{sid}.json")))
    assert again["avg_pitch"] == saved["avg_pitch"] and again["timestamp"] == saved["timestamp"]


def test_websocket_legacy_handshake_still_works(client):
    with client.websocket_connect("/api/audio-stream") as ws:
        ws.send_text(json.dumps({"session_id": "session_legacy1"}))
        assert ws.receive_json()["input_sample_rate"] == 16000
        y = voice_like(150, 16000, 2.0)
        for i in range(0, len(y), 2048):
            ws.send_bytes(to_pcm16(y[i:i + 2048]))
        ws.send_text(json.dumps({"type": "stop"}))
        while ws.receive_json()["type"] != "final":
            pass


def test_websocket_rejects_bad_format(client):
    with client.websocket_connect("/api/audio-stream") as ws:
        ws.send_text(json.dumps({"type": "start", "session_id": "session_x1", "sample_rate": 500}))
        assert ws.receive_json()["type"] == "error"
    with client.websocket_connect("/api/audio-stream") as ws:
        ws.send_text(json.dumps({"type": "start", "session_id": "../etc", "sample_rate": 48000}))
        assert ws.receive_json()["type"] == "error"


def test_websocket_disconnect_is_clean(client, capsys):
    with client.websocket_connect("/api/audio-stream") as ws:
        ws.send_text(json.dumps({"type": "start", "session_id": "session_drop1", "sample_rate": 48000}))
        ws.receive_json()
        ws.send_bytes(to_pcm16(voice_like(150, 48000, 0.6)))
    out = capsys.readouterr().out
    assert "Error in audio stream" not in out


def test_websocket_oversized_frame_dropped(client):
    with client.websocket_connect("/api/audio-stream") as ws:
        ws.send_text(json.dumps({"type": "start", "session_id": "session_big1", "sample_rate": 48000}))
        ws.receive_json()
        ws.send_bytes(b"\x00" * (300 * 1024))
        assert ws.receive_json()["type"] == "error"


# ---------------------------------------------------------------- REST ------
def test_report_404_and_invalid_ids(client):
    assert client.get("/api/report/session_doesnotexist").status_code == 404
    assert client.get("/api/report/..%2F..%2Fetc%2Fpasswd").status_code == 404
    assert client.post("/api/session/log", json={"session_id": "../../x"}).status_code == 400
    assert client.post("/api/session/log", content=b"[1,2]",
                       headers={"content-type": "application/json"}).status_code == 400
    big = {"session_id": "session_big", "x": "a" * 600_000}
    assert client.post("/api/session/log", json=big).status_code == 413


def test_history_reports_are_session_specific(client):
    a = {"session_id": "session_aaaaaaaa", "wpm": 100, "posture": 90, "eyeContact": "Good",
         "emotion": "Happy", "fillers": 0, "measured": {"speech": True, "pose": True, "face": True}}
    b = {"session_id": "session_bbbbbbbb", "wpm": 190, "posture": 35, "eyeContact": "Looking Away",
         "emotion": "Sad", "fillers": 9, "measured": {"speech": True, "pose": True, "face": True}}
    for p in (a, b):
        assert client.post("/api/session/log", json=p).status_code == 200
    hist = client.get("/api/sessions").json()["sessions"]
    assert {s["session_id"] for s in hist} == {"session_aaaaaaaa", "session_bbbbbbbb"}
    ra = client.get("/api/report/session_aaaaaaaa").json()["report"]
    rb = client.get("/api/report/session_bbbbbbbb").json()["report"]
    assert ra["metrics"]["wpm"] == 100 and rb["metrics"]["wpm"] == 190
    assert ra["ml_performance_score"] != rb["ml_performance_score"]
    assert ra["trend"] is None  # no trend recorded -> honest empty state, nothing synthesised
    # no audio was streamed -> audio metrics are missing, not 0/"Neutral"
    assert ra["metrics"]["volume"] is None and ra["metrics"]["tone"] is None
    assert "tone" in (ra["ml_score_note"] or "")


def test_legacy_schema_still_reports(client):
    legacy = {"session_id": "session_k2j3h4g", "wpm": "1601", "posture": "77", "eyeContact": "Unknown",
              "fillers": 4, "emotion": "Detecting...", "timestamp": "2025-07-01T10:00:00",
              "avg_volume": 3.2, "avg_pitch": 0.0, "avg_wpm": 128.4, "dominant_tone": "Neutral"}
    with open(os.path.join(client.raw_dir, "session_k2j3h4g.json"), "w") as f:
        json.dump(legacy, f)
    rep = client.get("/api/report/session_k2j3h4g").json()["report"]
    m = rep["metrics"]
    assert m["wpm"] == 128 and m["posture"] == 77 and m["volume"] == 3.2
    assert m["pitch"] is None and m["tone"] is None and m["emotion"] is None and m["eye_contact"] is None
    assert rep["ml_performance_score"] is not None
    assert client.get("/api/sessions").json()["sessions"][0]["wpm"] == 128


def test_nothing_measured_gives_no_score(client):
    p = {"session_id": "session_empty01", "measured": {"speech": False, "pose": False,
                                                        "face": False, "audio": False}}
    client.post("/api/session/log", json=p)
    rep = client.get("/api/report/session_empty01").json()["report"]
    assert rep["ml_performance_score"] is None and rep["ml_score_note"]
    assert rep["metrics"]["wpm"] is None and rep["metrics"]["posture"] is None


def test_predict_performance_endpoint(client):
    r = client.post("/api/predict/performance", json={"wpm": 145, "posture": 82, "fillers": 2,
                    "volume": 1.1, "pitch": 180, "eyeContact": "Good", "tone": "Balanced",
                    "emotion": "Happy"})
    assert r.status_code == 200 and 0 <= r.json()["ml_performance_score"] <= 100
    assert client.post("/api/predict/performance", json=[1]).status_code == 400


def test_models_status(client):
    s = client.get("/api/models/status").json()
    assert s["performance_model_loaded"] and s["tone_model_loaded"]
    assert s["load_warnings"] == []


def test_resolve_build_file_blocks_traversal(tmp_path):
    from backend.main import resolve_build_file
    build = tmp_path / "build"
    build.mkdir()
    (build / "index.html").write_text("x")
    (tmp_path / "secret.txt").write_text("s")
    assert resolve_build_file(str(build), "index.html")
    assert resolve_build_file(str(build), "../secret.txt") is None
    assert resolve_build_file(str(build), "/etc/passwd") is None


def test_all_empty_trend_is_not_a_trend():
    from backend.sessions import clean_trend
    assert clean_trend({"t": [5, 10], "posture": [None, None], "wpm": [None, None]}) is None
    assert clean_trend({"t": [5, 10], "posture": [None, 80], "wpm": [None, None]}) == {
        "t": [5.0, 10.0], "posture": [None, 80.0]}
    assert clean_trend({"t": [5]}) is None
