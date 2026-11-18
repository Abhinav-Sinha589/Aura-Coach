"""
session_features.py
-------------------
Single source of truth for turning a saved session record into the feature
row of the performance regressor (backend/models/performance_model.joblib).

`clean_sessions` is the exact preprocessing used when the model was trained
(it was moved here verbatim from train_performance_model.py so the training
script and the FastAPI report endpoint share one implementation). Inference
must go through it, otherwise categories such as "Detecting..." or numeric
outliers reach the model in a form it never saw during training.
"""
import pandas as pd

NUMERIC_FEATURES = ["wpm", "posture", "fillers", "volume", "pitch"]
CATEGORICAL_FEATURES = ["eyeContact", "tone", "emotion"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def clean_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Real session logs are messy (mixed types, outliers, unset placeholders
    like "Detecting..."). This applies the same cleaning rules the FastAPI
    endpoints should use at inference time, so train/serve stay consistent.
    """
    df = df.copy()

    # Coerce numeric-ish fields that sometimes arrive as strings
    for col in ["posture", "wpm", "fillers", "volume", "pitch"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Use avg_wpm/avg_volume/avg_pitch if present (these come from the
    # real-time audio websocket aggregation) and fall back to the raw field
    if "avg_wpm" in df.columns:
        df["wpm"] = pd.to_numeric(df["avg_wpm"], errors="coerce").fillna(df["wpm"])
    if "avg_volume" in df.columns:
        df["volume"] = pd.to_numeric(df["avg_volume"], errors="coerce").fillna(df["volume"])
    if "avg_pitch" in df.columns:
        df["pitch"] = pd.to_numeric(df["avg_pitch"], errors="coerce").fillna(df["pitch"])
    if "dominant_tone" in df.columns:
        df["tone"] = df["dominant_tone"].fillna(df.get("tone", "Neutral"))

    # Clip absurd outliers (e.g. wpm=1601 from a speech-recognition glitch)
    df["wpm"] = df["wpm"].clip(0, 260)
    df["posture"] = df["posture"].clip(0, 100)
    df["fillers"] = df["fillers"].clip(0, 50)
    df["volume"] = df["volume"].clip(0, 5)
    df["pitch"] = df["pitch"].clip(0, 500)

    # Normalize categorical placeholders into a clean "Unknown"/"Neutral" bucket
    if "eyeContact" not in df.columns:
        df["eyeContact"] = "Unknown"
    df["eyeContact"] = df["eyeContact"].fillna("Unknown").replace({"": "Unknown"})

    if "emotion" not in df.columns:
        df["emotion"] = "Neutral"
    df["emotion"] = df["emotion"].fillna("Neutral").replace(
        {"Detecting...": "Neutral", "": "Neutral"}
    )

    if "tone" not in df.columns:
        df["tone"] = "Neutral"
    df["tone"] = df["tone"].fillna("Neutral").replace({"": "Neutral"})

    return df


def record_to_feature_row(record: dict) -> pd.DataFrame:
    """One saved-session dict -> 1-row DataFrame in the trained column order."""
    return clean_sessions(pd.DataFrame([record]))[ALL_FEATURES]
