# Aura Coach — ML Components

This project now includes two trained machine learning models that
replace what used to be hand-written if/else rules.

## 1. Performance Score Regressor

**File:** `backend/ml/train_performance_model.py`
**Algorithm:** RandomForestRegressor (scikit-learn)
**Input features:** `wpm`, `posture`, `fillers`, `volume`, `pitch`,
`eyeContact`, `tone`, `emotion`
**Output:** A single 0–100 "performance score" per session.
**Trained on:** Your 78 real saved sessions in `data/raw/`.

### Why this is real ML, not rules
The old code computed a list of separate text recommendations from
fixed thresholds (`if wpm < 120: ...`). There was no single score, and
no model — just branches. This regressor:
- Learns a continuous mapping from raw features to score, capturing
  feature *interactions* (e.g. high WPM only hurts the score when
  fillers are also high) that independent if/else branches can't express.
- Reports real evaluation metrics (MAE, R², 5-fold cross-validation),
  not "it looks right."
- Exposes `feature_importances_`, giving a data-driven answer to "what
  actually matters most" — on your data, posture (54%) and WPM (19%)
  dominate, with eye contact next.
- Is retrainable: once you have real human-given ratings for sessions,
  swap the bootstrapped `compute_target_score()` labels for the real
  ones and rerun the same script — no other code changes needed.

### Current performance (on your 78 sessions)
- Test MAE: ~1.9 points (0–100 scale)
- Test R²: ~0.96
- 5-fold CV MAE: ~3.1 points

Re-run `python backend/ml/train_performance_model.py` any time you've
collected more sessions; it overwrites `backend/models/performance_model.joblib`.

## 2. Vocal Tone Classifier

**File:** `backend/ml/train_tone_model.py`
**Algorithm:** RandomForestClassifier (scikit-learn)
**Input features:** `rms` (loudness), `pitch_mean`, `pitch_std`,
`spectral_centroid`, `zero_crossing_rate` — all extracted with
`librosa` in `backend/ml/audio_features.py`.
**Output classes:** `Calm`, `Balanced`, `Energetic`, `Noise`.

### Why this is real ML, not a threshold
The old WebSocket handler decided tone with a single line:
```python
tone = "Calm" if rms < 0.02 else "Balanced" if rms < 0.05 else "Energetic"
```
That's one feature (loudness) and two manually-picked thresholds. The
new classifier uses five acoustic features, so it can correctly
distinguish, for example, "loud but monotone" from "moderate volume but
highly expressive pitch" — patterns a single threshold structurally
cannot represent.

### Training data note
No raw audio was saved alongside your sessions (only derived scalar
features), so this model is trained on **synthetic waveforms** that
match the acoustic *profile* of each tone (loudness range, pitch range,
pitch jitter — see `synth_clip()`). This is a legitimate, common
bootstrapping technique, and the same `audio_features.py` extraction
code is shared between training and the live FastAPI websocket, so
there's zero train/serve feature skew. Swap in a folder of real
labeled mic recordings later by replacing `build_training_set()` —
everything downstream (model, classifier, integration in `main.py`)
stays the same.

### Current performance (synthetic test set)
- Test accuracy: ~98%
- 5-fold CV accuracy: ~99.6%
- Most important features: pitch variability and loudness

## How it's wired into the app

- `backend/model_loader.py` loads both `.joblib` files once at FastAPI
  startup (lifespan handler) and exposes `predict_performance_score()`,
  `predict_tone()` and `predict_tone_proba()`. The artifacts are never
  modified or re-serialised; scikit-learn is pinned to **1.8.0**, the version
  they were pickled with.
- `backend/ml/session_features.py` holds `clean_sessions()`, the exact
  preprocessing used at training time (numeric coercion, clipping
  wpm 0–260 / posture 0–100 / fillers 0–50 / volume 0–5 / pitch 0–500,
  `"Detecting..."` → `Neutral`, missing eye contact → `Unknown`). Both
  `train_performance_model.py` and the report endpoint import it, so
  inference sees features exactly as training did.
- `/api/audio-stream` buffers audio at the browser's real sample rate and,
  every 0.5 s, resamples the latest **1.5 s** window to **16 kHz** — the
  clip length and rate the tone classifier was trained on — before calling
  the unchanged `extract_features_from_waveform()`. Silent windows are not
  classified; the last three windows' class probabilities are averaged for
  display. (Previously each ~43 ms browser chunk was classified on its own
  and decoded as 16 kHz regardless of the real 44.1/48 kHz rate, which put
  pitch and spectral centroid ~3× too low.)
- `/api/report/{session_id}` includes `ml_performance_score` plus
  `ml_score_note`, which lists any inputs that weren't measured in that
  session (they are filled with the training-time defaults above).
- `/api/predict/performance` lets you POST an arbitrary feature set and get
  a score back directly.
- `/api/models/status` reports both models, their classes/features, the
  installed scikit-learn version and any unpickling warnings.

## What the reported metrics do and don't mean

- **Tone classifier (98% test / 99.6% CV):** measured on held-out
  *synthetic* clips from the same generator as the training data. It shows
  the forest separates the four synthetic profiles; it is not an accuracy
  figure for real voices. Real-speech accuracy needs labelled recordings.
- **Performance regressor (R² 0.96 / MAE 1.9 on 78 sessions):** the target
  is the bootstrapped `compute_target_score()` formula, so these numbers
  measure how faithfully the forest reproduces that formula on held-out
  sessions. Note that some of those 78 training sessions were logged by the
  earlier app version, whose `avg_wpm` was a loudness-derived estimate and
  whose posture/eye-contact values were last-frame snapshots; the live
  pipeline now supplies real recognised-word WPM and session averages.
- Improving real-world accuracy beyond the signal-correctness fixes needs
  data: labelled tone recordings, human-rated sessions, and (for gaze)
  ground-truth camera-looking annotations.

## Training / retraining

```bash
cd backend
pip install -r requirements.txt
python ml/train_performance_model.py
python ml/train_tone_model.py
```

Both scripts print evaluation metrics to the console and write:
- `backend/models/performance_model.joblib` + `performance_model_meta.json`
- `backend/models/tone_model.joblib` + `tone_model_meta.json`

Retraining replaces the artifacts the app ships with; the Docker build does
**not** train — it uses the committed `.joblib` files.
