"""
sessions.py
-----------
Session-record helpers shared by the REST endpoints:

  * session-ID validation (IDs become file names, so they are whitelisted)
  * normalising old and new saved-session JSON into one metrics view where a
    metric that was never observed is `None`, not a fabricated 0 / "Neutral"
  * building the report (summary, insights, recommendations, real trend)

Older session files (schema v1) are still readable: they may hold strings
for numbers, "Detecting..." placeholders, a fake RMS-derived `avg_wpm`, and
`dominant_tone: "Neutral"` written when no audio was ever analysed.
"""
from __future__ import annotations

import math
import re
from typing import Any, Optional

SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")

# Tone classes the RF tone model can actually output.
TONE_CLASSES = {"Calm", "Balanced", "Energetic", "Noise"}
EYE_CONTACT_LABELS = {"Good", "Looking Away"}
EMOTION_PLACEHOLDERS = {"", "Detecting...", "Detecting…", "Unknown", "No face", "Unavailable"}
SERVER_OWNED_KEYS = {"avg_volume", "avg_pitch", "dominant_tone", "tone_counts", "audio_summary"}
MAX_TREND_POINTS = 2000


def is_valid_session_id(session_id: Any) -> bool:
    return isinstance(session_id, str) and bool(SESSION_ID_RE.match(session_id))


def _num(v) -> Optional[float]:
    if isinstance(v, bool) or v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _measured(record: dict, key: str) -> Optional[bool]:
    m = record.get("measured")
    if isinstance(m, dict) and key in m:
        return bool(m[key])
    return None  # legacy record: unknown


def session_metrics(record: dict) -> dict:
    """Normalised, display-ready metrics. Missing => None."""
    speech = _measured(record, "speech")
    pose = _measured(record, "pose")
    face = _measured(record, "face")

    wpm = _num(record.get("avg_wpm"))
    if wpm is None:
        wpm = _num(record.get("wpm"))
    if speech is False:
        wpm = None
    fillers = _num(record.get("fillers"))
    if speech is False:
        fillers = None

    posture = _num(record.get("posture"))
    head_tilt = _num(record.get("headTilt"))
    head_tilt_abs = _num(record.get("head_tilt_abs_mean"))
    if head_tilt_abs is None and head_tilt is not None:
        head_tilt_abs = abs(head_tilt)
    if pose is False:
        posture = head_tilt = head_tilt_abs = None

    eye = record.get("eyeContact")
    eye = eye if eye in EYE_CONTACT_LABELS else None

    emotion = record.get("emotion")
    emotion = emotion if isinstance(emotion, str) and emotion not in EMOTION_PLACEHOLDERS else None
    if face is False:
        emotion = None

    volume = _num(record.get("avg_volume"))
    if volume is None:
        volume = _num(record.get("volume"))
    pitch = _num(record.get("avg_pitch"))
    if pitch is None:
        pitch = _num(record.get("pitch"))
    if pitch is not None and pitch <= 0:  # 0 Hz was the old "no pitch" placeholder
        pitch = None

    tone = record.get("dominant_tone") or record.get("tone")
    tone = tone if tone in TONE_CLASSES else None

    return {
        "wpm": round(wpm) if wpm is not None else None,
        "fillers": int(fillers) if fillers is not None else None,
        "posture": round(posture) if posture is not None else None,
        "head_tilt": round(head_tilt, 1) if head_tilt is not None else None,
        "head_tilt_abs": round(head_tilt_abs, 1) if head_tilt_abs is not None else None,
        "eye_contact": eye,
        "eye_contact_ratio": _num(record.get("eye_contact_ratio")),
        "emotion": emotion,
        "tone": tone,
        "volume": round(volume, 2) if volume is not None else None,
        "pitch": round(pitch, 1) if pitch is not None else None,
        "duration_sec": _num(record.get("duration_sec")),
        "word_count": _num(record.get("word_count")),
    }


def missing_score_inputs(m: dict) -> list:
    names = {
        "wpm": "speaking pace",
        "posture": "posture",
        "eye_contact": "eye contact",
        "emotion": "emotion",
        "tone": "tone",
        "volume": "volume",
    }
    return [label for key, label in names.items() if m.get(key) is None]


def clean_trend(trend: Any) -> Optional[dict]:
    """Validate a recorded trend {t:[s], posture:[..], wpm:[..]}; never synthesise."""
    if not isinstance(trend, dict):
        return None
    t = trend.get("t")
    if not isinstance(t, list) or len(t) < 2:
        return None
    n = min(len(t), MAX_TREND_POINTS)
    out = {"t": [_num(x) for x in t[:n]]}
    if any(x is None for x in out["t"]):
        return None
    for key in ("posture", "wpm", "volume"):
        series = trend.get(key)
        if isinstance(series, list) and len(series) >= n:
            values = [_num(x) for x in series[:n]]
            if any(v is not None for v in values):  # an all-empty series is no trend
                out[key] = values
    return out if len(out) > 1 else None


def sanitize_client_payload(data: dict) -> dict:
    clean = {k: v for k, v in data.items() if k not in SERVER_OWNED_KEYS}
    if "trend" in clean:
        tr = clean_trend(clean["trend"])
        if tr is None:
            clean.pop("trend")
        else:
            clean["trend"] = tr
    return clean


def build_report(record: dict, ml_score: Optional[float], score_note: Optional[str]) -> dict:
    m = session_metrics(record)
    wpm, fillers, posture = m["wpm"], m["fillers"], m["posture"]
    eye, emotion, tone = m["eye_contact"], m["emotion"], m["tone"]
    volume, pitch = m["volume"], m["pitch"]

    parts = []
    if wpm is not None:
        parts.append(f"You spoke at {wpm} WPM with {fillers if fillers is not None else 0} filler words.")
    else:
        parts.append("Speaking pace wasn't measured (speech recognition unavailable).")
    parts.append(f"Avg volume: {volume if volume is not None else 'not measured'}, "
                 f"pitch: {str(pitch) + ' Hz' if pitch is not None else 'not measured'}.")
    summary = " ".join(parts)

    def eye_text():
        if eye is None:
            return "Eye contact could not be estimated (face not detected)."
        ratio = m.get("eye_contact_ratio")
        extra = f" ({round(ratio * 100)}% of measured time facing the camera)" if ratio is not None else ""
        return (f"Eye contact was {eye.lower()}{extra}, indicating "
                f"{'engagement' if eye == 'Good' else 'inconsistent focus'}.")

    insights = {
        "posture": (f"Your posture score was {posture}%, showing "
                    f"{'strong alignment' if posture > 80 else 'room for improvement'}."
                    if posture is not None else "Posture was not measured (body not detected)."),
        "eye_contact": eye_text(),
        "emotion": (f"Dominant facial expression: {emotion}." if emotion
                    else "Facial expression was not detected."),
        "tone": (f"Vocal tone was mostly {tone.lower()}." if tone
                 else "Vocal tone was not measured (no voiced audio analysed)."),
    }
    if m["head_tilt_abs"] is not None:
        insights["head_tilt"] = f"Average head tilt was {m['head_tilt_abs']:.0f}° from level."

    recs = []
    if wpm is not None:
        if wpm < 120:
            recs.append("Increase your speaking pace slightly for energy.")
        elif wpm > 160:
            recs.append("Slow down a bit for clarity and emphasis.")
        else:
            recs.append("Pace was balanced and natural.")
    if posture is not None and posture < 70:
        recs.append("Maintain upright shoulders and balanced head alignment.")
    if m["head_tilt_abs"] is not None and m["head_tilt_abs"] > 10:
        recs.append("Keep your head level — you tended to tilt it to one side.")
    if eye == "Looking Away":
        recs.append("Look toward the camera more often to build connection.")
    if fillers is not None and fillers > 5:
        recs.append("Reduce filler words such as 'um' and 'like' for smoother delivery.")
    if tone == "Calm":
        recs.append("Consider adding energy to sound more engaging.")
    if tone == "Energetic":
        recs.append("Good vocal projection — keep it controlled for clarity.")
    if emotion and emotion.lower() in ["sad", "angry", "fearful"]:
        recs.append("A warmer tone and expression can improve connection.")
    if not recs:
        recs.append("Excellent performance overall! Keep refining consistency.")

    return {
        "summary": summary,
        "insights": insights,
        "recommendations": recs,
        "ml_performance_score": round(ml_score, 1) if ml_score is not None else None,
        "ml_score_note": score_note,
        "metrics": m,
        "trend": clean_trend(record.get("trend")),
        "measured": record.get("measured") if isinstance(record.get("measured"), dict) else None,
        "timestamp": record.get("timestamp"),
        "schema_version": record.get("schema_version", 1),
    }
