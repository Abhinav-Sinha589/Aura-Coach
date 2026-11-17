"""
model_loader.py
-----------------
Loads the two trained ML models once at FastAPI startup and exposes
small prediction helper functions used by backend/main.py.

If a model file is missing (e.g. you haven't run the training scripts
yet), predictions fall back gracefully to None so the rest of the app
keeps working — main.py falls back to the original rule-based logic
in that case.
"""
import os
import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")

PERFORMANCE_MODEL_PATH = os.path.join(MODEL_DIR, "performance_model.joblib")
TONE_MODEL_PATH = os.path.join(MODEL_DIR, "tone_model.joblib")

_performance_model = None
_tone_model = None


def _safe_load(path):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            print(f"⚠️ Failed to load model at {path}: {e}")
    else:
        print(f"⚠️ Model file not found: {path} (run the matching train_*.py script)")
    return None


def load_models():
    """Call once at FastAPI startup."""
    global _performance_model, _tone_model
    _performance_model = _safe_load(PERFORMANCE_MODEL_PATH)
    _tone_model = _safe_load(TONE_MODEL_PATH)
    print(f"✅ Performance model loaded: {_performance_model is not None}")
    print(f"✅ Tone model loaded: {_tone_model is not None}")


# ---------------------------------------------------------------------------
# Inference-time preprocessing. Mirrors clean_sessions() in
# backend/ml/train_performance_model.py EXACTLY (verified against that file;
# parity is asserted in backend/tests/test_models.py):
#   * numeric: pd.to_numeric(errors="coerce").fillna(0)
#   * clip: wpm 0..260, posture 0..100, fillers 0..50, volume 0..5, pitch 0..500
#   * eyeContact missing/"" -> "Unknown"; emotion missing/""/"Detecting..." ->
#     "Neutral"; tone missing/"" -> "Neutral"
# The fitted estimator is untouched; this only feeds it the same shape of
# input it was trained on.
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = ["wpm", "posture", "fillers", "volume", "pitch"]
CATEGORICAL_FEATURES = ["eyeContact", "tone", "emotion"]
CLIP_RANGES = {"wpm": (0, 260), "posture": (0, 100), "fillers": (0, 50), "volume": (0, 5), "pitch": (0, 500)}
# Without these the score is dominated by imputed values (posture ~54% and wpm
# ~19% of feature importance per performance_model_meta.json) -> withhold it.
ESSENTIAL_FEATURES = ("wpm", "posture")


def _is_missing(v):
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip() == ""
    try:
        return not np.isfinite(float(v))
    except (TypeError, ValueError):
        return True


def prepare_performance_row(features: dict) -> pd.DataFrame:
    row = {}
    for col in NUMERIC_FEATURES:
        val = pd.to_numeric(pd.Series([features.get(col)]), errors="coerce").fillna(0).iloc[0]
        lo, hi = CLIP_RANGES[col]
        row[col] = float(np.clip(val, lo, hi))
    eye = features.get("eyeContact")
    row["eyeContact"] = "Unknown" if eye is None or eye == "" else str(eye)
    emo = features.get("emotion")
    row["emotion"] = "Neutral" if emo is None or emo in ("", "Detecting...") else str(emo)
    tone = features.get("tone")
    row["tone"] = "Neutral" if tone is None or tone == "" else str(tone)
    return pd.DataFrame([row], columns=NUMERIC_FEATURES + CATEGORICAL_FEATURES)


def imputed_features(features: dict):
    """Names of model inputs that are absent and will receive the training-time default."""
    return [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if _is_missing(features.get(c))]


def predict_performance_score(features: dict):
    """
    features: dict with keys wpm, posture, fillers, volume, pitch,
              eyeContact, tone, emotion
    Returns a float 0-100, or None if the model isn't loaded OR an essential
    input (wpm / posture) was not measured — we never invent those.
    """
    if _performance_model is None or not isinstance(features, dict):
        return None
    if any(_is_missing(features.get(k)) for k in ESSENTIAL_FEATURES):
        return None
    row = prepare_performance_row(features)
    try:
        pred = _performance_model.predict(row)[0]
        return float(np.clip(pred, 0, 100))
    except Exception as e:
        print(f"⚠️ performance model prediction failed: {e}")
        return None


def predict_tone(feature_vector: np.ndarray):
    """
    feature_vector: np.ndarray matching audio_features.FEATURE_NAMES order
                     [rms, pitch_mean, pitch_std, spectral_centroid, zcr]
    Returns (label: str, confidence: float) or (None, None) if unavailable.
    """
    if _tone_model is None:
        return None, None
    try:
        vec = np.asarray(feature_vector, dtype=np.float64).reshape(1, -1)
        if vec.shape[1] != 5 or not np.all(np.isfinite(vec)):
            return None, None
        proba = _tone_model.predict_proba(vec)[0]
        idx = int(np.argmax(proba))
        label = _tone_model.classes_[idx]
        confidence = float(proba[idx])
        return label, confidence
    except Exception as e:
        print(f"⚠️ tone model prediction failed: {e}")
        return None, None


def models_status():
    return {
        "performance_model_loaded": _performance_model is not None,
        "tone_model_loaded": _tone_model is not None,
    }
