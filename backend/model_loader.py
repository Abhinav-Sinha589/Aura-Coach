"""
model_loader.py
-----------------
Loads the two trained ML models once at FastAPI startup and exposes
small prediction helper functions used by backend/main.py.

The .joblib artifacts are never modified here. They were serialized with
scikit-learn 1.8.0 (see backend/requirements.txt); loading them with another
version emits InconsistentVersionWarning, which is surfaced in
`models_status()` instead of being silently ignored.

If a model file is missing, predictions return None and the API reports the
metric as unavailable (no rule-based value is passed off as a model output).
"""
import os
import warnings

import joblib
import numpy as np

try:
    from backend.ml.session_features import record_to_feature_row, ALL_FEATURES
except ImportError:
    from ml.session_features import record_to_feature_row, ALL_FEATURES

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")

PERFORMANCE_MODEL_PATH = os.path.join(MODEL_DIR, "performance_model.joblib")
TONE_MODEL_PATH = os.path.join(MODEL_DIR, "tone_model.joblib")

TRAINED_SKLEARN_VERSION = "1.8.0"

_performance_model = None
_tone_model = None
_load_warnings = []


def _safe_load(path):
    if not os.path.exists(path):
        print(f"⚠️ Model file not found: {path} (run the matching train_*.py script)")
        return None
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = joblib.load(path)
        msgs = sorted({str(w.message).split(".")[0] for w in caught})
        for m in msgs:
            print(f"⚠️ {os.path.basename(path)}: {m}")
            _load_warnings.append(f"{os.path.basename(path)}: {m}")
        return model
    except Exception as e:
        print(f"⚠️ Failed to load model at {path}: {e}")
        _load_warnings.append(f"{os.path.basename(path)}: load failed: {e}")
    return None


def load_models():
    """Call once at FastAPI startup."""
    global _performance_model, _tone_model
    _load_warnings.clear()
    _performance_model = _safe_load(PERFORMANCE_MODEL_PATH)
    _tone_model = _safe_load(TONE_MODEL_PATH)
    print(f"✅ Performance model loaded: {_performance_model is not None}")
    print(f"✅ Tone model loaded: {_tone_model is not None}")


def predict_performance_score(features: dict):
    """
    features: a saved-session record or a dict with keys wpm, posture,
              fillers, volume, pitch, eyeContact, tone, emotion.
    The row goes through the *training* preprocessing (clean_sessions) so
    clipping ranges and placeholder categories match what the model saw.
    Returns a float 0-100, or None if the model isn't loaded.
    """
    if _performance_model is None:
        return None
    try:
        row = record_to_feature_row(features)
        pred = _performance_model.predict(row)[0]
        return float(np.clip(pred, 0, 100))
    except Exception as e:
        print(f"⚠️ performance model prediction failed: {e}")
        return None


def predict_tone_proba(feature_vector: np.ndarray):
    """Returns (classes, probabilities) or (None, None) if unavailable."""
    if _tone_model is None:
        return None, None
    try:
        x = np.asarray(feature_vector, dtype=np.float32).reshape(1, -1)
        proba = _tone_model.predict_proba(x)[0]
        return list(_tone_model.classes_), proba
    except Exception as e:
        print(f"⚠️ tone model prediction failed: {e}")
        return None, None


def predict_tone(feature_vector: np.ndarray):
    """
    feature_vector: np.ndarray matching audio_features.FEATURE_NAMES order
                     [rms, pitch_mean, pitch_std, spectral_centroid, zcr]
    Returns (label: str, confidence: float) or (None, None) if unavailable.
    """
    classes, proba = predict_tone_proba(feature_vector)
    if classes is None:
        return None, None
    idx = int(np.argmax(proba))
    return classes[idx], float(proba[idx])


def models_status():
    try:
        import sklearn
        sk_version = sklearn.__version__
    except Exception:  # pragma: no cover
        sk_version = None
    return {
        "performance_model_loaded": _performance_model is not None,
        "tone_model_loaded": _tone_model is not None,
        "tone_classes": list(_tone_model.classes_) if _tone_model is not None else None,
        "performance_features": ALL_FEATURES,
        "sklearn_version": sk_version,
        "trained_sklearn_version": TRAINED_SKLEARN_VERSION,
        "load_warnings": list(_load_warnings),
    }
