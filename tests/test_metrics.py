import numpy as np
import pytest

from audiobench.metrics import Thresholds, compare_arrays, evaluate_thresholds, estimate_delay_samples


def test_identical_render_passes():
    sr = 48_000
    t = np.arange(sr) / sr
    signal = (0.4 * np.sin(2 * np.pi * 440 * t))[:, None]
    metrics = compare_arrays(signal, signal.copy(), sr)
    evaluation = evaluate_thresholds(metrics)
    assert metrics["identical_samples"] is True
    assert evaluation["status"] == "PASS"
    assert metrics["duration_delta_ms"] == 0.0


def test_gain_error_fails_fidelity_gate():
    sr = 48_000
    rng = np.random.default_rng(4)
    expected = (rng.normal(size=sr) * 0.1)[:, None]
    actual = expected * 0.8
    metrics = compare_arrays(expected, actual, sr)
    evaluation = evaluate_thresholds(metrics, Thresholds(min_snr_db=60.0))
    assert metrics["snr_db"] is not None
    assert metrics["snr_db"] < 60.0
    assert evaluation["checks"]["fidelity"]["status"] == "FAIL"


def test_delay_sign_is_positive_when_actual_is_late():
    sr = 48_000
    rng = np.random.default_rng(3)
    expected = rng.normal(size=sr)[:, None] * 0.1
    shift = 120
    actual = np.vstack([np.zeros((shift, 1)), expected[:-shift]])
    delay, confidence = estimate_delay_samples(expected, actual, sr, max_lag_ms=10)
    assert delay == shift
    assert confidence is not None and confidence > 0.8


def test_channel_mismatch_raises():
    with pytest.raises(ValueError, match="Channel count mismatch"):
        compare_arrays(np.zeros((100, 1)), np.zeros((100, 2)), 48_000)


def test_different_length_is_detected():
    expected = np.zeros((4800, 1))
    actual = np.zeros((4700, 1))
    metrics = compare_arrays(expected, actual, 48_000)
    evaluation = evaluate_thresholds(metrics)
    assert metrics["duration_delta_ms"] == pytest.approx(-100 / 48)
    assert evaluation["checks"]["duration"]["status"] == "FAIL"


def test_spectral_metrics_separate_gain_from_shape():
    sr = 48_000
    rng = np.random.default_rng(12)
    expected = (rng.normal(size=8192) * 0.1)[:, None]
    actual = expected * 0.5
    metrics = compare_arrays(expected, actual, sr)
    assert metrics["mean_spectral_delta_db"] == pytest.approx(6.0206, abs=0.08)
    assert metrics["spectral_shape_error_db"] == pytest.approx(0.0, abs=1e-6)


def test_frequency_band_gate():
    sr = 48_000
    t = np.arange(sr, dtype=np.float64) / sr
    sig = 0.4 * np.sin(2 * np.pi * 1000 * t)[:, None]
    # Small gain change (0.5 dB boost)
    boosted = sig * (10.0 ** (0.5 / 20.0))
    metrics = compare_arrays(sig, boosted, sr)

    # Gate with 1.0 dB tolerance -> PASS
    eval_pass = evaluate_thresholds(metrics, Thresholds(max_band_delta_db=1.0))
    assert eval_pass["checks"]["frequency_bands"]["status"] == "PASS"

    # Gate with 0.2 dB tolerance -> FAIL
    eval_fail = evaluate_thresholds(metrics, Thresholds(max_band_delta_db=0.2))
    assert eval_fail["checks"]["frequency_bands"]["status"] == "FAIL"


def test_thd_gate():
    sr = 48_000
    t = np.arange(sr, dtype=np.float64) / sr
    # Add 2nd harmonic at -30 dB (3.16%)
    clean = np.sin(2 * np.pi * 1000 * t)[:, None]
    distorted = (clean + 0.0316 * np.sin(2 * np.pi * 2000 * t)[:, None])
    metrics = compare_arrays(clean, distorted, sr)

    # Allow max -20 dB THD -> PASS
    eval_pass = evaluate_thresholds(metrics, Thresholds(max_thd_db=-20.0))
    assert eval_pass["checks"]["distortion_thd"]["status"] == "PASS"

    # Allow max -40 dB THD -> FAIL (-30 dB > -40 dB)
    eval_fail = evaluate_thresholds(metrics, Thresholds(max_thd_db=-40.0))
    assert eval_fail["checks"]["distortion_thd"]["status"] == "FAIL"


def test_true_peak_gate():
    sr = 48_000
    t = np.arange(sr, dtype=np.float64) / sr
    # Inter-sample peak signal at 0 dBTP
    sig = np.sin(2 * np.pi * 12_000 * t + np.pi / 4.0)[:, None]
    metrics = compare_arrays(sig, sig, sr)

    # Ceiling +1.0 dBTP -> PASS
    eval_pass = evaluate_thresholds(metrics, Thresholds(max_true_peak_dbfs=1.0))
    assert eval_pass["checks"]["true_peak"]["status"] == "PASS"

    # Ceiling -1.0 dBTP -> FAIL
    eval_fail = evaluate_thresholds(metrics, Thresholds(max_true_peak_dbfs=-1.0))
    assert eval_fail["checks"]["true_peak"]["status"] == "FAIL"
