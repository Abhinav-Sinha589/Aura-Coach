from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from contextlib import asynccontextmanager
from pathlib import Path
import asyncio, hashlib, json, logging, math, os, tempfile, uuid
from datetime import datetime

try:
    # Works when run as `uvicorn backend.main:app` from the project root
    from backend.model_loader import load_models, predict_performance_score, predict_tone, models_status, imputed_features
    from backend import audio_session as aud
except ImportError:
    # Works when run from inside backend/ (e.g. `uvicorn main:app`)
    from model_loader import load_models, predict_performance_score, predict_tone, models_status, imputed_features
    import audio_session as aud

log = logging.getLogger("aura_coach")


# -------------------------------------------
# 🚀 FASTAPI INITIAL SETUP
# -------------------------------------------
@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Loads backend/models/performance_model.joblib and tone_model.joblib (read-only)
    load_models()
    yield


app = FastAPI(lifespan=lifespan)


# ✅ CORS: explicit allowlist only. In the Render/Docker deployment the React
# build is served by this same FastAPI app (same origin), so CORS is only
# needed for local dev or a separately hosted frontend; add those origins via
# ALLOWED_ORIGINS="https://a.example,https://b.example". The app uses no
# cookies/credentials, so credentials are disabled (also makes "*" safe).
raw_custom_origins = os.environ.get("ALLOWED_ORIGINS", "")
custom_origins = [o.strip() for o in raw_custom_origins.split(",") if o.strip()]

default_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://d32w6h95-3000.inc1.devtunnels.ms",
]
allowed_origins = sorted(set(default_origins + custom_origins)) if "*" not in custom_origins else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


# Directory for saving session JSON (override with AURA_DATA_DIR; on Render's
# free plan this disk is ephemeral and is wiped on redeploy/restart).
RAW_DIR = os.environ.get("AURA_DATA_DIR") or os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

MAX_SESSION_BODY_BYTES = 256 * 1024
MAX_TIMELINE_POINTS = 720
SCHEMA_VERSION = 2

# Finished per-session audio aggregates handed from the WebSocket to the POST.
# Bounded + TTL'd; in-process only (not shared across workers / restarts).
AUDIO_SUMMARIES = aud.AudioSummaryStore()
_SAVE_LOCK = asyncio.Lock()


def _json_error(status: int, message: str):
    return JSONResponse(status_code=status, content={"error": message})


def _session_path(session_id: str):
    """Validated, directory-confined path for a session id (None if unsafe)."""
    if not aud.is_readable_session_id(session_id):
        return None
    base = Path(RAW_DIR).resolve()
    candidate = (base / f"{session_id}.json").resolve()
    if candidate.parent != base:
        return None
    return candidate


def _atomic_write_json(path: Path, data: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _finite(v):
    """float(v) if it is a finite number (bools excluded), else None."""
    if isinstance(v, bool) or v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _opt_str(v, max_len=40):
    if isinstance(v, str) and v.strip() and v != "Detecting...":
        return v.strip()[:max_len]
    return None


# Fields the client may submit. Everything else is dropped; audio aggregates
# and timestamps are server-controlled.
_CLIENT_NUMERIC = {
    "wpm": (0, 1000), "word_count": (0, 10**6), "capture_seconds": (0, 86400),
    "fillers": (0, 10**5), "posture": (0, 100), "posture_samples": (0, 10**7),
    "shoulder_tilt": (-90, 90), "eye_contact_ratio": (0, 1), "camera_samples": (0, 10**7),
}
_CLIENT_STRINGS = ("eyeContact", "emotion", "speech_status", "wpm_source", "audio_status")


def _sanitize_client_payload(data: dict) -> dict:
    out = {}
    for key, (lo, hi) in _CLIENT_NUMERIC.items():
        if key in data:
            f = _finite(data.get(key))
            out[key] = None if f is None else min(max(f, lo), hi)
    for key in _CLIENT_STRINGS:
        if key in data:
            out[key] = _opt_str(data.get(key))
    emo = data.get("emotion_counts")
    if isinstance(emo, dict):
        out["emotion_counts"] = {
            str(k)[:20]: int(v) for k, v in list(emo.items())[:12]
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0
        }
    tl = data.get("timeline")
    if isinstance(tl, list):
        points = []
        for p in tl[:MAX_TIMELINE_POINTS]:
            if not isinstance(p, dict):
                continue
            t = _finite(p.get("t"))
            if t is None or t < 0:
                continue
            posture = _finite(p.get("posture"))
            wpm = _finite(p.get("wpm"))
            points.append({"t": round(t, 1),
                           "posture": None if posture is None else round(posture, 1),
                           "wpm": None if wpm is None else round(wpm, 1)})
        out["timeline"] = points
    return out


# -------------------------------------------
# 🟢 Log a finished training session (exactly-once, idempotent)
# -------------------------------------------
@app.post("/api/session/log")
async def log_session(req: Request):
    """
    Persist one finished session. The client generates a UUID session id and
    uses it for the WebSocket handshake, this POST and the report URL.

    * identical retry of an already-saved session -> 200, record unchanged
    * different payload for an already-saved id    -> 409 (no silent overwrite)
    * invalid JSON -> 400, wrong shape / unsafe id -> 422
    """
    body = await req.body()
    if len(body) > MAX_SESSION_BODY_BYTES:
        return _json_error(413, "Session payload too large")
    try:
        data = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return _json_error(400, "Body must be valid JSON")
    if not isinstance(data, dict):
        return _json_error(422, "Body must be a JSON object")

    session_id = data.get("session_id")
    if session_id is None:
        session_id = str(uuid.uuid4())
    if not aud.is_new_session_id(session_id):
        return _json_error(422, "session_id must be a lowercase canonical UUID")
    path = _session_path(session_id)
    if path is None:
        return _json_error(422, "Invalid session_id")

    client = _sanitize_client_payload(data)
    payload_hash = hashlib.sha256(
        json.dumps({"session_id": session_id, **client}, sort_keys=True).encode("utf-8")
    ).hexdigest()

    async with _SAVE_LOCK:
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return _json_error(409, "A different record already exists for this session_id")
            if isinstance(existing, dict) and existing.get("client_payload_sha256") == payload_hash:
                return {"status": "ok", "session_id": session_id, "duplicate": True}
            return _json_error(409, "Session already saved with different data")

        record = {"session_id": session_id, "schema_version": SCHEMA_VERSION, **client}
        audio = AUDIO_SUMMARIES.get(session_id)
        if audio:
            record.update({k: audio.get(k) for k in (
                "avg_volume", "avg_pitch", "dominant_tone", "tone_counts",
                "audio_windows", "voiced_windows", "audio_finalized", "audio_sample_rate")})
        else:
            record.update({"avg_volume": None, "avg_pitch": None, "dominant_tone": None,
                           "audio_windows": 0, "audio_finalized": False})
        record["timestamp"] = datetime.now().isoformat()
        record["client_payload_sha256"] = payload_hash
        await asyncio.to_thread(_atomic_write_json, path, record)
        AUDIO_SUMMARIES.pop(session_id)

    log.info("Session saved: %s", session_id)
    return {"status": "ok", "session_id": session_id, "duplicate": False}


# -------------------------------------------
# 🔊 Real-Time Audio Streaming (handshake -> windows -> finish/ack)
# -------------------------------------------
HANDSHAKE_TIMEOUT_S = 10.0


async def _send_error_and_close(ws: WebSocket, message: str, code: int = 1008):
    try:
        await ws.send_json({"type": "error", "error": message})
        await ws.close(code=code)
    except Exception:
        pass


@app.websocket("/api/audio-stream")
async def audio_stream(ws: WebSocket):
    """
    Protocol (all JSON control messages carry ``type``):
      client -> {type:'start', session_id:<uuid>, sample_rate:<8000-96000>, channels:1, encoding:'pcm_s16le'}
      server -> {type:'ready', session_id, analysis_window_ms:1500, hop_ms:500}
      client -> binary PCM16LE mono packets (any size up to 256 KB)
      server -> {type:'audio_metrics', session_id, volume, pitch|null, tone|null, tone_confidence|null, ...}
      client -> {type:'finish'}
      server -> {type:'finished', session_id, aggregates:{...}}   then closes
    Binary audio before a valid 'start' is rejected.
    """
    await ws.accept()
    cfg = None
    window_buf = None
    agg = aud.AudioAggregate()
    finished = False

    async def process(windows):
        for w in windows:
            m = await asyncio.to_thread(aud.analyze_window, w, cfg.sample_rate, predict_tone)
            agg.add(m)
            await ws.send_json({"type": "audio_metrics", "session_id": cfg.session_id,
                                "analysis_window_ms": int(aud.WINDOW_SECONDS * 1000), **m})

    def summary(final: bool):
        return {**agg.to_dict(), "audio_finalized": final, "audio_sample_rate": cfg.sample_rate}

    try:
        # ---- handshake ----
        try:
            first = await asyncio.wait_for(ws.receive(), timeout=HANDSHAKE_TIMEOUT_S)
        except asyncio.TimeoutError:
            await _send_error_and_close(ws, "handshake timeout")
            return
        if first.get("type") == "websocket.disconnect":
            return
        try:
            cfg = aud.parse_start_message(first.get("text"))
        except aud.HandshakeError as e:
            await _send_error_and_close(ws, f"invalid start message: {e}")
            return
        window_buf = aud.WindowBuffer(cfg.sample_rate)
        await ws.send_json({"type": "ready", "session_id": cfg.session_id,
                            "analysis_window_ms": int(aud.WINDOW_SECONDS * 1000),
                            "hop_ms": int(aud.HOP_SECONDS * 1000)})

        # ---- stream ----
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            data = msg.get("bytes")
            if data is not None:
                if len(data) > aud.MAX_AUDIO_PACKET_BYTES:
                    await _send_error_and_close(ws, "audio packet too large", code=1009)
                    break
                if not data:
                    continue
                try:
                    samples = aud.decode_pcm16le(data)
                except ValueError:
                    await ws.send_json({"type": "error", "error": "malformed PCM16 packet"})
                    continue
                await process(window_buf.push(samples))
                continue

            text = msg.get("text")
            if text is None:
                continue
            if len(text) > aud.MAX_CONTROL_MESSAGE_BYTES:
                await ws.send_json({"type": "error", "error": "control message too large"})
                continue
            try:
                control = json.loads(text)
            except ValueError:
                await ws.send_json({"type": "error", "error": "control message is not JSON"})
                continue
            if isinstance(control, dict) and control.get("type") == "finish":
                tail = window_buf.flush()
                if tail is not None:
                    await process([tail])
                finished = True
                final = summary(True)
                AUDIO_SUMMARIES.put(cfg.session_id, final)
                await ws.send_json({"type": "finished", "session_id": cfg.session_id, "aggregates": final})
                await ws.close(code=1000)
                break
            await ws.send_json({"type": "error", "error": "unknown control message"})

    except WebSocketDisconnect:
        pass
    except Exception:
        # Don't log raw audio; just the exception class/trace.
        log.exception("Unexpected error in audio stream")
        await _send_error_and_close(ws, "internal error", code=1011)
    finally:
        # Abandoned connection: keep a partial (non-finalised) summary, bounded by TTL.
        if cfg is not None and not finished and agg.windows:
            AUDIO_SUMMARIES.put(cfg.session_id, summary(False))


# -------------------------------------------
# 🧠 Generate Performance Report
# -------------------------------------------
def _extract_session_metrics(data: dict) -> dict:
    """
    Reconcile schema v2 (explicit nulls, true WPM) and legacy v1 session JSONs
    without inventing values when a measurement was absent.
    """
    schema_version = data.get("schema_version", 1)
    raw_wpm = _finite(data.get("wpm"))
    legacy_avg_wpm = _finite(data.get("avg_wpm"))
    if raw_wpm is not None:
        wpm = round(raw_wpm, 1)
        wpm_source = data.get("wpm_source") or "speech_recognition"
    elif schema_version < 2 and legacy_avg_wpm is not None:
        wpm = round(legacy_avg_wpm, 1)
        wpm_source = "legacy_avg_wpm"
    else:
        wpm = None
        wpm_source = data.get("wpm_source") or "unavailable"

    posture = _finite(data.get("posture"))
    posture = round(posture, 1) if posture is not None else None

    shoulder_tilt = _finite(data.get("shoulder_tilt"))
    if shoulder_tilt is None and schema_version < 2:
        shoulder_tilt = _finite(data.get("headTilt"))
    shoulder_tilt = round(shoulder_tilt, 1) if shoulder_tilt is not None else None

    fillers_val = _finite(data.get("fillers"))
    fillers = int(round(fillers_val)) if fillers_val is not None else None

    vol = _finite(data.get("avg_volume"))
    if vol is None:
        vol = _finite(data.get("volume"))
    volume = round(vol, 2) if vol is not None else None

    p = _finite(data.get("avg_pitch"))
    if p is None:
        p = _finite(data.get("pitch"))
    pitch = round(p, 1) if (p is not None and p > 0) else None

    eye_contact = _opt_str(data.get("eyeContact"))
    if eye_contact == "Unknown":
        eye_contact = None
    emotion = _opt_str(data.get("emotion"))
    tone = _opt_str(data.get("dominant_tone")) or _opt_str(data.get("tone"))

    timeline = []
    if isinstance(data.get("timeline"), list):
        for pt in data["timeline"][:MAX_TIMELINE_POINTS]:
            if isinstance(pt, dict) and _finite(pt.get("t")) is not None:
                timeline.append({
                    "t": round(_finite(pt.get("t")), 1),
                    "posture": round(_finite(pt.get("posture")), 1) if _finite(pt.get("posture")) is not None else None,
                    "wpm": round(_finite(pt.get("wpm")), 1) if _finite(pt.get("wpm")) is not None else None,
                })

    return {
        "schema_version": schema_version,
        "timestamp": data.get("timestamp"),
        "wpm": wpm,
        "wpm_source": wpm_source,
        "word_count": int(_finite(data.get("word_count"))) if _finite(data.get("word_count")) is not None else None,
        "capture_seconds": round(_finite(data.get("capture_seconds")), 1) if _finite(data.get("capture_seconds")) is not None else None,
        "speech_status": _opt_str(data.get("speech_status")),
        "posture": posture,
        "shoulder_tilt": shoulder_tilt,
        "eyeContact": eye_contact,
        "fillers": fillers,
        "emotion": emotion,
        "tone": tone,
        "volume": volume,
        "pitch": pitch,
        "timeline": timeline,
    }


@app.get("/api/report/{session_id}")
async def get_report(session_id: str):
    path = _session_path(session_id)
    if path is None:
        return _json_error(422, "Invalid session_id")
    if not path.exists():
        return _json_error(404, "Session not found")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return _json_error(422, "Corrupted session file")
    if not isinstance(data, dict):
        return _json_error(422, "Corrupted session file")

    m = _extract_session_metrics(data)
    wpm = m["wpm"]
    posture = m["posture"]
    eye_contact = m["eyeContact"]
    fillers = m["fillers"]
    emotion = m["emotion"]
    tone = m["tone"]
    pitch = m["pitch"]
    volume = m["volume"]

    pace_part = f"You spoke at {int(round(wpm))} WPM" if wpm is not None else "Speaking pace was not measured"
    filler_part = f" with {fillers} filler words" if fillers is not None else ""
    vol_part = f"{volume}" if volume is not None else "Unavailable"
    pitch_part = f"{pitch} Hz" if pitch is not None else "Unavailable"
    summary = f"{pace_part}{filler_part}. Avg volume: {vol_part}, pitch: {pitch_part}."

    insights = {
        "posture": (
            f"Your posture score was {posture}%, showing {'strong alignment' if posture > 80 else 'room for improvement'}."
            if posture is not None
            else "Posture was not measured during this session."
        ),
        "eye_contact": (
            f"Camera-facing alignment (approx.) was {eye_contact.lower()}, indicating "
            f"{'consistent framing toward the camera' if eye_contact == 'Good' else 'inconsistent camera alignment'}."
            if eye_contact is not None
            else "Camera-facing alignment was not measured."
        ),
        "emotion": (
            f"Dominant facial expression estimate: {emotion}."
            if emotion is not None
            else "Facial expression estimate: Unavailable."
        ),
        "tone": (
            f"Vocal tone estimate was mostly {tone.lower()}."
            if tone is not None
            else "Vocal tone: Unavailable (insufficient voiced audio or model not loaded)."
        ),
    }

    recs = []
    if wpm is not None:
        if wpm == 0:
            recs.append("No finalized speech words were captured — check microphone permissions or speak clearly.")
        elif wpm < 120:
            recs.append("Increase your speaking pace slightly for energy.")
        elif wpm > 160:
            recs.append("Slow down a bit for clarity and emphasis.")
        else:
            recs.append("Pace was balanced and natural.")

    if posture is not None and posture < 70:
        recs.append("Maintain upright shoulders and balanced shoulder alignment.")
    if fillers is not None and fillers > 5:
        recs.append("Reduce filler words such as 'um' and 'like' for smoother delivery.")
    if tone and tone.lower() == "calm":
        recs.append("Consider adding vocal energy to sound more engaging.")
    if tone and tone.lower() == "energetic":
        recs.append("Good vocal projection — keep it controlled for clarity.")
    if emotion and emotion.lower() in ["sad", "angry", "fearful"]:
        recs.append("A warmer facial expression estimate can improve visual connection.")

    if not recs:
        recs.append("Complete a full session with camera and microphone active for detailed recommendations.")

    model_features = {
        "wpm": wpm,
        "posture": posture,
        "fillers": fillers,
        "volume": volume,
        "pitch": pitch,
        "eyeContact": eye_contact,
        "tone": tone,
        "emotion": emotion,
    }
    ml_score = predict_performance_score(model_features)
    imputed = imputed_features(model_features) if ml_score is not None else []

    report = {
        "summary": summary,
        "metrics": m,
        "timeline": m["timeline"],
        "insights": insights,
        "recommendations": recs,
        "ml_performance_score": round(ml_score, 1) if ml_score is not None else None,
        "ml_imputed_features": imputed,
        "ml_note": (
            "Heuristic-bootstrapped Random Forest estimate (trained on 78 historical sessions); "
            "withheld when essential metrics (WPM or posture) are unmeasured."
        ),
    }
    return {"session_id": session_id, "report": report}


# -------------------------------------------
# 🧠 Standalone ML performance score prediction
# -------------------------------------------
@app.post("/api/predict/performance")
async def predict_performance(req: Request):
    """
    Lets the frontend (or a quick curl/Postman test) get an ML performance
    score for an arbitrary feature set without needing a saved session file.
    """
    try:
        features = await req.json()
    except ValueError:
        return _json_error(400, "Body must be valid JSON")
    if not isinstance(features, dict):
        return _json_error(422, "Body must be a JSON object")
    status = models_status()
    if not status.get("performance_model_loaded"):
        return _json_error(503, "Performance model not loaded.")
    score = predict_performance_score(features)
    if score is None:
        return _json_error(422, "Essential features (wpm, posture) are required to compute a score.")
    return {
        "ml_performance_score": round(score, 1),
        "imputed_features": imputed_features(features),
    }


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
@app.get("/api/sessions")
async def list_sessions():
    """
    Returns a lightweight summary of every saved session so the frontend
    can render a history table without fetching each full report.
    """
    sessions = []
    if os.path.isdir(RAW_DIR):
        for filename in os.listdir(RAW_DIR):
            if not filename.endswith(".json") or filename.startswith("."):
                continue
            sid_from_name = filename[:-5]
            if not aud.is_readable_session_id(sid_from_name):
                continue
            file_path = _session_path(sid_from_name)
            if file_path is None:
                continue
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            if not isinstance(data, dict):
                continue

            m = _extract_session_metrics(data)
            sid = data.get("session_id")
            if not aud.is_readable_session_id(sid):
                sid = sid_from_name

            sessions.append({
                "session_id": sid,
                "timestamp": data.get("timestamp"),
                "wpm": m["wpm"],
                "posture": m["posture"],
                "emotion": m["emotion"] or "Unavailable",
            })

    # newest sessions first
    sessions.sort(key=lambda s: s.get("timestamp") or "", reverse=True)
    return {"sessions": sessions}


# -------------------------------------------
# 🏠 Frontend Static Assets & SPA Routing
# -------------------------------------------
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
        BUILD_DIR = os.path.abspath(p)
        break

if BUILD_DIR:
    BUILD_ROOT = Path(BUILD_DIR).resolve()
    static_subdir = BUILD_ROOT / "static"
    if static_subdir.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_subdir)), name="static")

    models_subdir = BUILD_ROOT / "models"
    if models_subdir.is_dir():
        app.mount("/models", StaticFiles(directory=str(models_subdir)), name="face_models")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        # Don't intercept API routes (safety check)
        if full_path.startswith("api/") or full_path == "api":
            return _json_error(404, "API route not found")

        if full_path:
            try:
                candidate = (BUILD_ROOT / full_path).resolve()
                if BUILD_ROOT in candidate.parents and candidate.is_file():
                    return FileResponse(str(candidate))
            except (OSError, ValueError):
                pass

        return FileResponse(str(BUILD_ROOT / "index.html"))
else:
    @app.get("/")
    async def root():
        return {"message": "Aura Coach backend running with real-time persistent speech analysis ✅"}


