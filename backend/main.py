from contextlib import asynccontextmanager
import asyncio
import io
import json
import os
import time
import uuid
from datetime import datetime

import numpy as np
import soundfile as sf
from fastapi import FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

try:
    # Works when run as `uvicorn backend.main:app` from the project root
    from backend.model_loader import (
        load_models, predict_performance_score, predict_tone_proba, models_status,
    )
    from backend.audio_pipeline import (
        StreamAnalyzer, SessionAudioAggregate, decode_pcm16le, warm_up,
        TRAINING_SAMPLE_RATE, TRAINING_WINDOW_SEC, DEFAULT_HOP_SEC,
    )
    from backend.sessions import (
        is_valid_session_id, session_metrics, build_report, sanitize_client_payload,
        missing_score_inputs,
    )
except ImportError:
    # Works when run from inside backend/ (e.g. `uvicorn main:app`)
    from model_loader import (
        load_models, predict_performance_score, predict_tone_proba, models_status,
    )
    from audio_pipeline import (
        StreamAnalyzer, SessionAudioAggregate, decode_pcm16le, warm_up,
        TRAINING_SAMPLE_RATE, TRAINING_WINDOW_SEC, DEFAULT_HOP_SEC,
    )
    from sessions import (
        is_valid_session_id, session_metrics, build_report, sanitize_client_payload,
        missing_score_inputs,
    )


# -------------------------------------------
# 🚀 FASTAPI INITIAL SETUP
# -------------------------------------------
@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Loads backend/models/performance_model.joblib and tone_model.joblib
    load_models()
    try:
        # pyin is JIT-compiled on first use (~1 s); do it now, not mid-session
        await asyncio.to_thread(warm_up)
    except Exception as e:  # pragma: no cover
        print(f"⚠️ audio warm-up failed: {e}")
    yield


app = FastAPI(title="Aura Coach API", lifespan=lifespan)


# CORS: the deployed app is same-origin (FastAPI serves the React build), so
# CORS only matters for local dev / tunnels. No cookies or auth headers are
# used, so credentials are off, which also keeps a "*" origin valid.
raw_custom_origins = os.environ.get("ALLOWED_ORIGINS", "")
custom_origins = [o.strip() for o in raw_custom_origins.split(",") if o.strip()]
default_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://d32w6h95-3000.inc1.devtunnels.ms",
]
allowed_origins = ["*"] if "*" in custom_origins else sorted(set(default_origins + custom_origins))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=os.environ.get("ALLOWED_ORIGIN_REGEX") or None,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


# Directory for saving logs (NOTE: Render's free-tier disk is ephemeral)
RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
os.makedirs(RAW_DIR, exist_ok=True)

MAX_LOG_BYTES = 512 * 1024
MAX_WS_FRAME_BYTES = 256 * 1024
SESSION_TTL_SEC = 30 * 60
HOP_SEC = float(os.environ.get("AUDIO_HOP_SEC", DEFAULT_HOP_SEC))

# Live audio aggregates, keyed by session_id. Bounded online statistics only;
# purged after SESSION_TTL_SEC of inactivity. Single-process state: run one
# uvicorn worker (as the Dockerfile does) or move this to a shared store.
SESSION_AUDIO_DATA: dict = {}
_session_locks: dict = {}


def _purge_stale_sessions():
    now = time.time()
    for sid in [s for s, a in SESSION_AUDIO_DATA.items() if now - a.last_seen > SESSION_TTL_SEC]:
        SESSION_AUDIO_DATA.pop(sid, None)
        _session_locks.pop(sid, None)


def _lock_for(session_id: str) -> asyncio.Lock:
    lock = _session_locks.get(session_id)
    if lock is None:
        lock = _session_locks[session_id] = asyncio.Lock()
    return lock


def _session_path(session_id: str) -> str:
    """Only ever called with validated IDs; double-checks containment anyway."""
    path = os.path.abspath(os.path.join(RAW_DIR, f"{session_id}.json"))
    if os.path.dirname(path) != RAW_DIR:
        raise ValueError("invalid session path")
    return path


def _read_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def _write_json_atomic(path: str, data: dict):
    tmp = f"{path}.{uuid.uuid4().hex}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=4)
    os.replace(tmp, path)


def _error(status: int, message: str):
    return JSONResponse(status_code=status, content={"error": message})


# -------------------------------------------
# 🟢 Log a training session (idempotent per session_id)
# -------------------------------------------
@app.post("/api/session/log")
async def log_session(req: Request):
    """
    Store session metrics + the live-audio aggregate for the same session_id.
    Posting the same session_id again merges into the existing record and is
    reported as a duplicate (safe for retries / double clicks).
    """
    body = await req.body()
    if len(body) > MAX_LOG_BYTES:
        return _error(413, "Session payload too large")
    try:
        data = json.loads(body or b"{}")
    except Exception:
        return _error(400, "Body must be JSON")
    if not isinstance(data, dict):
        return _error(400, "Body must be a JSON object")

    session_id = data.get("session_id") or f"session_{uuid.uuid4().hex}"
    if not is_valid_session_id(session_id):
        return _error(400, "Invalid session_id")
    data = sanitize_client_payload(data)
    data["session_id"] = session_id

    _purge_stale_sessions()
    async with _lock_for(session_id):
        file_path = _session_path(session_id)
        existing = None
        if os.path.exists(file_path):
            try:
                existing = _read_json(file_path)
            except Exception:
                existing = None

        record = dict(existing) if isinstance(existing, dict) else {}
        record.update(data)
        now = datetime.now().isoformat()
        record.setdefault("timestamp", now)
        if existing:
            record["updated_at"] = now

        # Real speech-recognition WPM (finalised words / elapsed time). The
        # old server-side `120 + rms*200` estimate is gone: WPM is never
        # inferred from loudness.
        wpm = data.get("wpm")
        if "wpm" in data:
            record["avg_wpm"] = wpm if isinstance(wpm, (int, float)) else None

        aggregate = SESSION_AUDIO_DATA.pop(session_id, None)
        if aggregate is not None:
            aggregate.finalized = True
            record.update(aggregate.summary())
        elif not existing:
            measured = record.get("measured")
            if isinstance(measured, dict) and measured.get("audio") is False:
                record.update({"avg_volume": None, "avg_pitch": None, "dominant_tone": None})

        await asyncio.to_thread(_write_json_atomic, file_path, record)

    print(f"✅ Session saved → {os.path.basename(file_path)} (audio aggregate: {aggregate is not None})")
    return {"status": "ok", "session_id": session_id, "duplicate": existing is not None}


# -------------------------------------------
# 🔊 Real-Time Audio Streaming
# -------------------------------------------
@app.websocket("/api/audio-stream")
async def audio_stream(ws: WebSocket):
    """
    Protocol (v2):
      client → {"type":"start","session_id":..., "sample_rate":48000,
                "channels":1, "encoding":"pcm_s16le"}
      server → {"type":"ready", ...}
      client → binary little-endian int16 mono PCM at sample_rate
      server → {"type":"metrics", volume, pitch, tone, ...} every hop (0.5 s)
      client → {"type":"stop"}  → server flushes the last window and replies
      server → {"type":"final", "summary": {...}}  (then the client POSTs the log)
    Legacy clients ({"session_id": ...} then raw PCM) are treated as 16 kHz.
    """
    await ws.accept()
    send_lock = asyncio.Lock()
    data_ready = asyncio.Event()
    state = {"analyzer": None, "session_id": None, "closing": False, "stopping": False}
    _purge_stale_sessions()

    async def send(payload: dict):
        async with send_lock:
            await ws.send_json(payload)

    def bind(session_id, sample_rate):
        aggregate = None
        if session_id:
            aggregate = SESSION_AUDIO_DATA.get(session_id)
            if aggregate is None:
                aggregate = SESSION_AUDIO_DATA[session_id] = SessionAudioAggregate(session_id)
        else:
            aggregate = SessionAudioAggregate("unbound")
        state["session_id"] = session_id
        state["analyzer"] = StreamAnalyzer(
            sample_rate, aggregate, tone_proba_fn=predict_tone_proba, hop_sec=HOP_SEC
        )

    async def run_analysis(final=False):
        analyzer = state["analyzer"]
        if analyzer is None:
            return
        try:
            result = await asyncio.to_thread(analyzer.analyze, final)
        except Exception as e:
            print(f"⚠️ audio analysis failed: {e}")
            return
        if result is not None and not state["closing"]:
            await send(result)

    async def analysis_loop():
        while True:
            await data_ready.wait()
            data_ready.clear()
            analyzer = state["analyzer"]
            while analyzer is not None and analyzer.has_hop():
                await run_analysis()
            if state["stopping"]:
                return

    analysis_task = asyncio.create_task(analysis_loop())
    print("🎙️ WebSocket audio stream connected")

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            text = msg.get("text")
            if text:
                try:
                    control = json.loads(text)
                except Exception:
                    continue
                if not isinstance(control, dict):
                    continue
                ctype = control.get("type", "start" if control.get("session_id") else None)

                if ctype == "start":
                    sid = control.get("session_id")
                    if sid is not None and not is_valid_session_id(sid):
                        await send({"type": "error", "error": "invalid session_id"})
                        await ws.close(code=1008)
                        break
                    legacy = "sample_rate" not in control
                    try:
                        sr = int(control.get("sample_rate", TRAINING_SAMPLE_RATE))
                        if control.get("encoding", "pcm_s16le") != "pcm_s16le":
                            raise ValueError("encoding")
                        if int(control.get("channels", 1)) != 1:
                            raise ValueError("channels")
                        bind(sid, sr)
                    except (TypeError, ValueError) as e:
                        await send({"type": "error", "error": f"unsupported audio format ({e})"})
                        await ws.close(code=1003)
                        break
                    print(f"📝 Audio bound: {sid} @ {sr} Hz{' (legacy client)' if legacy else ''}")
                    await send({
                        "type": "ready", "session_id": sid,
                        "input_sample_rate": sr,
                        "analysis_sample_rate": TRAINING_SAMPLE_RATE,
                        "window_sec": TRAINING_WINDOW_SEC, "hop_sec": HOP_SEC,
                    })
                elif ctype == "stop":
                    # Drain: let in-flight analysis finish (never run two
                    # analyses at once), then the trailing partial hop.
                    state["stopping"] = True
                    data_ready.set()
                    try:
                        await analysis_task
                    except asyncio.CancelledError:
                        pass
                    analyzer = state["analyzer"]
                    if analyzer is not None and analyzer.pending_samples() >= int(0.1 * analyzer.sr):
                        await run_analysis(final=True)
                    summary = analyzer.aggregate.summary() if analyzer else None
                    await send({"type": "final", "session_id": state["session_id"], "summary": summary})
                    state["closing"] = True
                elif ctype == "abort":
                    if state["session_id"]:
                        SESSION_AUDIO_DATA.pop(state["session_id"], None)
                    break
                elif ctype == "ping":
                    await send({"type": "pong"})
                continue

            data = msg.get("bytes")
            if not data or state["closing"]:
                continue
            if len(data) > MAX_WS_FRAME_BYTES:
                await send({"type": "error", "error": "audio frame too large; dropped"})
                continue

            if state["analyzer"] is None:
                # Legacy client that never sent a handshake: assume 16 kHz PCM16
                bind(None, TRAINING_SAMPLE_RATE)

            if data[:4] == b"RIFF":
                try:
                    y, wav_sr = sf.read(io.BytesIO(data), dtype="float32")
                    if y.ndim > 1:
                        y = y.mean(axis=1)
                    if wav_sr != state["analyzer"].sr:
                        bind(state["session_id"], wav_sr)
                except Exception:
                    continue
            else:
                y = decode_pcm16le(data)

            state["analyzer"].push(np.asarray(y, dtype=np.float32))
            if state["analyzer"].has_hop():
                data_ready.set()

    except Exception as e:
        print(f"⚠️ Error in audio stream: {type(e).__name__}: {e}")
    finally:
        analysis_task.cancel()
        try:
            await analysis_task
        except (asyncio.CancelledError, Exception):
            pass
        print("🔴 Audio WebSocket closed.")


# -------------------------------------------
# 🧠 Generate Performance Report
# -------------------------------------------
def _score_record(record: dict):
    """ML score + an honest note about which inputs were not measured."""
    measured = record.get("measured") if isinstance(record.get("measured"), dict) else {}
    if measured and not (measured.get("pose") or measured.get("speech")):
        return None, "Not enough was measured in this session to score it."
    score = predict_performance_score(record)
    if score is None:
        return None, "Performance model unavailable."
    missing = missing_score_inputs(session_metrics(record))
    note = None
    if missing:
        note = ("Not measured: " + ", ".join(missing)
                + ". The model scored these with its training defaults.")
    return score, note


@app.get("/api/report/{session_id}")
async def get_report(session_id: str):
    if not is_valid_session_id(session_id):
        return _error(404, "Session not found")
    file_path = _session_path(session_id)
    if not os.path.exists(file_path):
        return _error(404, "Session not found")
    try:
        record = _read_json(file_path)
    except Exception:
        return _error(500, "Session file is unreadable")
    if not isinstance(record, dict):
        return _error(500, "Session file is malformed")

    score, note = _score_record(record)
    return {"session_id": session_id, "report": build_report(record, score, note)}


# -------------------------------------------
# 🧠 Standalone ML performance score prediction
# -------------------------------------------
@app.post("/api/predict/performance")
async def predict_performance(req: Request):
    """
    Body example:
      {"wpm": 145, "posture": 82, "fillers": 2, "volume": 1.1,
       "pitch": 180, "eyeContact": "Good", "tone": "Balanced", "emotion": "Happy"}
    """
    try:
        features = await req.json()
    except Exception:
        return _error(400, "Body must be JSON")
    if not isinstance(features, dict):
        return _error(400, "Body must be a JSON object")
    score = predict_performance_score(features)
    if score is None:
        return _error(503, "Performance model not loaded.")
    return {"ml_performance_score": round(score, 1)}


# -------------------------------------------
# 🧠 ML model health check
# -------------------------------------------
@app.get("/api/models/status")
async def model_status():
    """Quick way to confirm both ML models loaded successfully at startup."""
    return models_status()


# -------------------------------------------
# 📋 List all saved sessions (for Session History page)
# -------------------------------------------
def _list_sessions():
    sessions = []
    if os.path.isdir(RAW_DIR):
        for filename in os.listdir(RAW_DIR):
            if not filename.endswith(".json"):
                continue
            try:
                data = _read_json(os.path.join(RAW_DIR, filename))
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            sid = data.get("session_id", filename[:-5])
            if not is_valid_session_id(sid):
                continue
            m = session_metrics(data)
            sessions.append({
                "session_id": sid,
                "timestamp": data.get("timestamp"),
                "wpm": m["wpm"],
                "posture": m["posture"],
                "emotion": m["emotion"],
                "eyeContact": m["eye_contact"],
                "tone": m["tone"],
            })
    sessions.sort(key=lambda s: s.get("timestamp") or "", reverse=True)
    return sessions


@app.get("/api/sessions")
async def list_sessions():
    """Lightweight summary of every saved session (newest first)."""
    return {"sessions": await asyncio.to_thread(_list_sessions)}


# -------------------------------------------
# 🏠 Frontend Static Assets & SPA Routing
# -------------------------------------------
def resolve_build_file(build_dir: str, full_path: str):
    """Path of a real file inside build_dir, or None (blocks ../ traversal)."""
    if not full_path:
        return None
    root = os.path.realpath(build_dir)
    requested = os.path.realpath(os.path.join(root, full_path))
    if os.path.commonpath([requested, root]) != root or not os.path.isfile(requested):
        return None
    return requested


BUILD_DIR = None
candidate_paths = [
    os.path.join(os.path.dirname(__file__), "build"),
    os.path.join(os.path.dirname(__file__), "..", "build"),
    os.path.join(os.path.dirname(__file__), "..", "dist"),
    "/app/build",
    "/app/backend/build",
]

for p in candidate_paths:
    if os.path.exists(p) and os.path.isdir(p) and os.path.exists(os.path.join(p, "index.html")):
        BUILD_DIR = os.path.realpath(p)
        break

if BUILD_DIR:
    print(f"📦 Serving static frontend from: {BUILD_DIR}")
    static_subdir = os.path.join(BUILD_DIR, "static")
    if os.path.exists(static_subdir):
        app.mount("/static", StaticFiles(directory=static_subdir), name="static")

    models_subdir = os.path.join(BUILD_DIR, "models")
    if os.path.exists(models_subdir):
        app.mount("/models", StaticFiles(directory=models_subdir), name="face_models")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        if full_path == "api" or full_path.startswith("api/"):
            return _error(404, "API route not found")

        # Only serve files that really live inside BUILD_DIR (no ../ traversal)
        requested = resolve_build_file(BUILD_DIR, full_path)
        if requested:
            return FileResponse(requested)

        # Otherwise fallback to index.html for client-side routing
        return FileResponse(os.path.join(BUILD_DIR, "index.html"))
else:
    @app.get("/")
    async def root():
        return {"message": "Aura Coach backend running with real-time persistent speech analysis ✅"}
