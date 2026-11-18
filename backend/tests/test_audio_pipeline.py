"""Signal-correctness tests for the streaming audio path.

Synthetic waveforms are used only to test code behaviour (sample-rate
handling, windowing, gating). They say nothing about classifier accuracy.
"""
import numpy as np
import pytest

from backend import audio_pipeline as ap
from backend.ml.audio_features import extract_features_from_waveform
from backend.model_loader import load_models, predict_tone_proba
from backend.tests.conftest import voice_like


@pytest.fixture(scope="module", autouse=True)
def _models():
    load_models()


def run_stream(y, sr, chunk=4096, hop_sec=0.5):
    agg = ap.SessionAudioAggregate("session_test")
    an = ap.StreamAnalyzer(sr, agg, tone_proba_fn=predict_tone_proba, hop_sec=hop_sec)
    out = []
    for i in range(0, len(y), chunk):
        an.push(y[i:i + chunk])
        while an.has_hop():
            out.append(an.analyze())
    r = an.analyze(final=True)
    if r:
        out.append(r)
    return out, agg


def sine(f, sr, dur, amp=0.1):
    return (amp * np.sin(2 * np.pi * f * np.arange(int(sr * dur)) / sr)).astype(np.float32)


def test_old_path_misreported_48k_pitch_by_3x():
    """Documents the original bug: 48 kHz chunks decoded as 16 kHz."""
    chunk_48k = sine(180, 48000, 2048 / 48000)
    feats = extract_features_from_waveform(chunk_48k, 16000)
    assert 55 < feats[1] < 65  # ~60 Hz reported for a 180 Hz tone


@pytest.mark.parametrize("sr", [16000, 44100, 48000])
def test_pitch_consistent_across_sample_rates(sr):
    msgs, agg = run_stream(sine(180, sr, 3.0), sr)
    pitches = [m["pitch"] for m in msgs if m["pitch"] is not None]
    assert pitches, "no pitch reported"
    assert abs(np.median(pitches) - 180) < 3
    assert abs(agg.summary()["avg_pitch"] - 180) < 3


def test_model_features_match_between_16k_and_48k():
    y16 = sine(180, 16000, 1.5)
    y48 = sine(180, 48000, 1.5)
    f16 = extract_features_from_waveform(y16, 16000)
    f48 = extract_features_from_waveform(ap.resample_to_training_rate(y48, 48000), 16000)
    assert np.allclose(f16[[0, 1, 4]], f48[[0, 1, 4]], rtol=0.02)
    assert abs(f16[3] - f48[3]) / f16[3] < 0.05  # spectral centroid


def test_model_receives_training_length_window(monkeypatch):
    seen = []
    real = ap.extract_features_from_waveform

    def spy(y, sr):
        seen.append((len(y), sr))
        return real(y, sr)

    monkeypatch.setattr(ap, "extract_features_from_waveform", spy)
    run_stream(voice_like(150, 48000, 3.0), 48000)
    assert seen and all(n == 24000 and sr == 16000 for n, sr in seen)


def test_silence_gives_no_tone_or_pitch():
    msgs, agg = run_stream(np.zeros(48000 * 3, dtype=np.float32), 48000)
    assert all(m["tone"] is None and m["pitch"] is None for m in msgs)
    assert all(m["status"] in ("silence", "warming_up") for m in msgs)
    s = agg.summary()
    assert s["dominant_tone"] is None and s["avg_pitch"] is None and s["avg_volume"] is None


def test_warm_up_period_has_no_tone():
    msgs, _ = run_stream(voice_like(150, 48000, 1.0), 48000)
    assert msgs and all(m["tone"] is None for m in msgs)
    assert msgs[0]["status"] == "warming_up"


def test_voiced_audio_gets_model_tone_and_volume():
    msgs, agg = run_stream(voice_like(160, 48000, 4.0), 48000)
    toned = [m for m in msgs if m["status"] == "ok"]
    assert toned, "voiced audio never reached the tone model"
    assert all(m["tone"] in {"Calm", "Balanced", "Energetic", "Noise"} for m in toned)
    assert all(m["volume"] > 0 for m in toned)
    assert sum(agg.tone_counts.values()) == agg.tone_windows > 0


def test_short_packets_and_odd_bytes():
    y = voice_like(150, 44100, 2.0)
    msgs, _ = run_stream(y, 44100, chunk=137)
    assert len(msgs) >= 3
    assert ap.decode_pcm16le(b"\x01\x00\x02").shape == (1,)


def test_volume_scale_matches_training_units():
    # 0.1-amplitude sine -> RMS 0.0707 -> "volume" 7.07 (RMS x 100, as in training data)
    msgs, _ = run_stream(sine(200, 16000, 2.0), 16000)
    assert abs(msgs[-1]["volume_raw"] - 7.07) < 0.1


def test_clipping_flagged():
    y = np.clip(sine(200, 16000, 2.0, amp=2.0), -1, 1)
    msgs, agg = run_stream(y, 16000)
    assert any(m["clipping"] for m in msgs) and agg.clipped_hops > 0


def test_rejects_bad_sample_rate():
    with pytest.raises(ValueError):
        ap.StreamAnalyzer(1000, ap.SessionAudioAggregate("x"))


def test_analysis_latency_is_bounded():
    msgs, _ = run_stream(voice_like(150, 48000, 4.0), 48000)
    lat = [m["latency_ms"] for m in msgs[3:]]
    assert max(lat) < 400, lat  # hop is 500 ms


def test_burst_is_analysed_hop_by_hop_not_skipped():
    """Audio queued while the socket opened arrives in one burst (~2 s)."""
    agg = ap.SessionAudioAggregate("session_burst")
    an = ap.StreamAnalyzer(48000, agg, tone_proba_fn=predict_tone_proba)
    an.push(voice_like(150, 48000, 2.0))          # one burst
    msgs = []
    while an.has_hop():
        msgs.append(an.analyze())
    assert len(msgs) == 4 and agg.skipped_hops == 0
    assert [m["t"] for m in msgs] == [0.5, 1.0, 1.5, 2.0]


def test_long_backlog_skips_to_realtime():
    agg = ap.SessionAudioAggregate("session_lag")
    an = ap.StreamAnalyzer(16000, agg, tone_proba_fn=predict_tone_proba)
    an.push(voice_like(150, 16000, 10.0))
    n = 0
    while an.has_hop():
        an.analyze()
        n += 1
    assert agg.skipped_hops > 0 and n < 20
    assert agg.audio_seconds == pytest.approx(10.0)
