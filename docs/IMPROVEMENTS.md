# Aura Coach — accuracy & reliability pass (branch `improve/accuracy-reliability-v2`)

Base: `ca1d0f7` (current `main`). No model was retrained, re-serialised or replaced:
`backend/models/*.joblib` SHA-256 are asserted unchanged by `backend/tests/test_api.py`.
All résumé-listed features, fields and technologies are kept (Posture, Head Tilt, Eye Contact,
Emotion, Tone, Pitch, Volume, WPM, Fillers, 0–100 score, tips, reports, history; React, FastAPI,
MediaPipe Pose, face-api.js, both Random Forests, Docker, Render).

## 1. Issue inventory (verified against `ca1d0f7`)

| ID | Reported issue | Status on `main` | Evidence (old code) | Resolution |
|---|---|---|---|---|
| 01 | No WebSocket `onmessage` | **Confirmed** | `TrainingPage.js` `startAudioStreaming` set only onopen/onerror/onclose | Handler updates volume/pitch/tone; stale-session and out-of-order messages ignored |
| 02 | WS and log IDs differ | **Confirmed** | TrainingPage POSTed without `session_id` → server minted a new ID; the WS temp ID's audio aggregate was never merged and leaked | One 128-bit ID per session for WS, POST, file and report |
| 03 | TrainingPage and App both save | **Confirmed** | `handleSessionToggle` POST + `App.handleEndSession` POST | TrainingPage saves once (guarded, idempotent server); App only navigates |
| 04 | App opens a 2nd camera/mic stream | **Confirmed** | `App.js` `getUserMedia({video, audio})` into an unused ref; camera_utils opened a 3rd | TrainingPage is the single owner; all tracks stopped on exit (E2E-verified) |
| 05 | Interim ASR counted repeatedly | **Confirmed** | `onresult` added words from every interim result | Only `isFinal`, once per (generation, index), growing-final dedupe |
| 06 | Recognition stops on pause | **Confirmed** | no `onend` restart | Restart on recoverable end; loop guard; fatal errors → N/A |
| 07 | 44.1/48 k audio treated as 16 k | **Confirmed** | `sr = 16000` hard-coded; 180 Hz tone → **60.5 Hz** measured | Sample rate in handshake; 1.5 s windows resampled to 16 kHz → **180.2 Hz** |
| 08 | WPM from RMS preferred in report | **Confirmed** | `wpm_est = 120 + rms*200` stored as `avg_wpm`, preferred by report/history | Removed; `avg_wpm` = finalised words / elapsed time |
| 09 | Camera/Pose/face models recreated per Start/Stop | **Confirmed** | effect deps `[isSessionActive]`; Pose never `.close()`d | Created once per mount; face-api weights cached per page load; Pose closed |
| 10 | WS disconnect mishandled | **Confirmed** | `receive()` disconnect dict fell through → RuntimeError logged as "Error in audio processing" | Explicit `websocket.disconnect` handling (test asserts no error log) |
| 11 | History report uses live props | **Confirmed** | ReportPage stats/chart from `data` prop (last live session) | Report endpoint returns the session's own metrics; page uses only those |
| 12 | Report stuck loading on null/failed ID | **Confirmed** | `if (!sessionId) return` left `loading=true` | Explicit missing / error (+ Retry) states |
| 13 | Trend labels/count mismatch, fabricated | **Confirmed** | 5 labels, 4 points, synthesised ±offsets | Real 5 s samples with m:ss labels, or "Trend not recorded" |
| 14 | Train/serve preprocessing differs | **Confirmed** | serving skipped `clean_sessions` (e.g. `"Detecting..."` unseen category) | Shared `ml/session_features.py` (verbatim training code; equivalence test) |
| 15 | Volume tip threshold vs RMS scale | **Confirmed** | `volume < 10` (RMS×100) ⇒ "Speak louder" almost permanently | Only while voice-active and < 1.0 (≈ −40 dBFS), debounced |
| 16 | sklearn not pinned for joblib | **Confirmed** | pickles are sklearn **1.8.0**; `scikit-learn>=1.5` installs 1.9 → InconsistentVersionWarning | `scikit-learn==1.8.0`, `numpy>=2` (pickles use `numpy._core`); status endpoint reports warnings |
| 17 | Missing report returns 200 | **Confirmed** | `return {"error": ...}` | 404 (also 400/413/503 where appropriate) |
| 18 | Wildcard CORS with credentials | **Partly** — `"*"` only via env, but `allow_credentials=True` + any `*.onrender.com` regex | `main.py` CORS block | Credentials off (none used), regex only via `ALLOWED_ORIGIN_REGEX` |
| 19 | Generic title / deprecated startup | **Confirmed** | `<title>React App</title>`, `@app.on_event` | "Aura Coach"; lifespan handler |

Additional bugs found and fixed:

- **Path traversal** in the SPA catch-all: `/..%2F..%2Fetc%2Fpasswd` resolved outside `build/`. Now realpath-contained (tested).
- **Saved Posture/Eye Contact/Emotion/Head Tilt were last-frame snapshots**, not session values. Now session means / measured ratios / dominant expression.
- **Tone model was fed 43 ms fragments** (one per ScriptProcessor buffer), not the 1.5 s clips it was trained on.
- **PCM16 conversion wrapped on clipping** (`input*0x7fff` into Int16Array) → clamped.
- **Browser AGC/noise suppression on** flattened volume changes → disabled for the analysis stream.
- **Unbounded global audio arrays**, no cleanup → bounded online aggregates + 30 min TTL.
- **Feature extraction blocked the event loop** → worker thread, single-flight, bounded catch-up.
- **pYIN octave errors at speech onsets** pinned to 400 Hz skewed the session pitch → edge-of-range frames rejected; session pitch is a histogram median.
- **`CI=true npm run build` failed** on face-api.js's Node-only `require("fs")` → `src/fs.js` stub via `jsconfig.json` (no new dependency, no `CI=false`).
- Node 18 base image is EOL → Node 20 LTS.
- `README` claimed the Docker build trains models and exposes port 8080 — neither true; corrected.

Endpoint count: 5 REST (`POST /api/session/log`, `GET /api/report/{id}`, `POST /api/predict/performance`, `GET /api/models/status`, `GET /api/sessions`) + 1 WebSocket (`/api/audio-stream`) = **6 endpoints**, all preserved. (`GET /` / the SPA catch-all serve the app.)

## 2. Decision matrix (uncertain design choices)

| Area | Option | Accuracy potential | Cost / deps | Model compatibility | Risk | Chosen |
|---|---|---|---|---|---|---|
| Head Tilt | Shoulder line (old) | Wrong quantity | none | n/a | Mislabelled metric | ✗ |
| | **Pose eye line (ear fallback), pixel space** | Good for roll | none (Pose already running) | no model input | Low | ✅ |
| | face-api 68-landmark net | Similar | +model weights, +inference | — | Medium | ✗ |
| | Face Mesh / Face Landmarker | Best (also yaw/pitch) | second heavy vision model | — | CPU regression, untestable here | ✗ (future) |
| Eye Contact | `|nose.x−0.5|<0.05` (old) | Measures frame position | none | `eyeContact` labels | Fails off-centre | ✗ |
| | **Head-relative yaw+pitch, 2 s calibration, hysteresis** | Camera-facing head pose | none | same labels Good/Looking Away/Unknown | Misses eyes-only glances | ✅ |
| | Iris tracking (Face Mesh refineLandmarks) | Real gaze direction | extra model per frame | same labels | CPU, can't validate without camera | ✗ (future) |
| Tone window | 43 ms chunks (old) | Out of distribution | — | violates training | — | ✗ |
| | **1.5 s window, 0.5 s hop, 16 kHz** | Matches training | ~22 ms CPU / hop | exact | Low | ✅ |
| Volume | dBFS display | Clearer units | — | would need conversion at API boundary | Silent skew if mixed | ✗ |
| | **Keep RMS×100 (training units), AGC off, voice-gated tip** | Faithful to model | none | exact | Uncalibrated (documented) | ✅ |
| Missing inputs → score | Return no score | — | — | — | Hides partially valid sessions | only when nothing core measured |
| | **Training-time defaults + visible note** | Same as training | — | exact | Disclosed | ✅ |

## 3. Tests run (actual results)

| Suite | Result |
|---|---|
| `python -m pytest backend/tests` (sklearn 1.8.0) | **32 passed** |
| `CI=true npm test -- --watchAll=false` | **43 passed** (4 suites) |
| `CI=true npm run build` | **exit 0**, no ESLint/module warnings (174 third-party face-api.js source-map warnings, which CRA does not count) |
| Docker frontend stage replayed with only the copied files (`npm ci && npm run build`) | exit 0 |
| `docker build` | **Untested** — Docker Hub is unreachable from the build sandbox |
| Headless Chrome E2E (built app + FastAPI, fake camera, 160 Hz WAV as mic) | handshake sent **44100 Hz**; live pitch **160 Hz**; RF tone every 0.5 s; final ack before save; report avg pitch **160.0 Hz**; speech service unreachable ⇒ "not measured" (no fake 0 WPM) |
| Headless Chrome lifecycle E2E | 3× enter/leave → 0 live tracks, 0 open sockets; leaving mid-session → nothing saved, tracks released; 2 sessions (double-clicked End) → exactly 2 files; each History row opens its own report; unknown ID → 404 |

Key measurements:

- Pitch of a 180 Hz tone: old live path **60.5 Hz** (48 kHz decoded as 16 kHz) → new **180.2 Hz** at 16 k, 44.1 k and 48 k (±3 Hz test tolerance).
- Model input window: always **24 000 samples @ 16 kHz** (asserted).
- Server analysis latency: **~22 ms per 0.5 s hop** (pYIN tuned from ~70 ms); old path ~1.5 ms × 23 chunks/s.
- Emotion smoother on a noisy alternating sequence: raw label switches ≥ 7 → smoothed ≤ 1; a sustained change is followed within ≤ 3 samples (~1.8 s).
- WPM: re-delivered final results (20 sentences, each event repeated) → exactly 100 words / 60 s = **100 WPM** (old logic would multiply).

## 4. Manual browser checklist (needs a real camera/mic — not possible in the sandbox)

1. Chrome, laptop: Start Session, speak normally — Tone leaves *Analyzing…* within ~1.5 s, Volume rises when you speak louder and Pitch reads in a plausible range (≈85–180 Hz male, ≈165–255 Hz female).
2. Stay silent for 3 s — Tone shows *Listening…*, Pitch "—", no "Speak louder".
3. Tilt your head ~15° with level shoulders — Head Tilt ≈ ±15°; shrug one shoulder with head level — Head Tilt stays near 0.
4. Sit at the edge of the frame looking at the camera — Eye Contact stays **Good**; turn your head to the side — **Looking Away** within ~1 s; cover your face — **Unknown**.
5. Smile, then neutral — Emotion follows within ~2 s without flickering.
6. Say "um, you know, I mean" — Fillers +3; pause 10 s, keep talking — WPM keeps counting (recognition restarts).
7. End session — one report with your session's numbers and a real trend; open it again from History.
8. Check the browser console's `[Aura] session …` line for Pose fps / ms per frame on your machine.
9. Firefox: WPM/Fillers show *N/A*; everything else works.

## 5. Known limitations / what needs data

- Eye Contact is camera-facing **head orientation**; eyes-only glances aren't detected. True gaze needs iris landmarks and labelled data to tune.
- Posture is a heuristic (shoulder levelness + personal-baseline slouch); no ground truth exists.
- Tone 98%/99.6% were measured on synthetic clips; performance R²/MAE measure agreement with a bootstrapped formula. Real-world accuracy can't be claimed or improved further without labelled recordings / human-rated sessions — the historical numbers are left as historical claims and their provenance is documented in `README_ML.md`.
- Some of the 78 training sessions carried the old RMS-derived `avg_wpm` and last-frame posture; the live pipeline now supplies real values, so the regressor sees cleaner inputs than parts of its training set.
- Session audio aggregates live in process memory (single uvicorn worker); Render's free disk is ephemeral.
- Chrome's speech recognition sends audio to Google; Firefox has none.
