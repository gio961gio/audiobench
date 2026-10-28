import numpy as np
import pytest
from scipy.signal import butter, sosfilt

from audiobench.analysis import (
    analyze_band_deviations,
    analyze_thd,
    check_dsp_health,
    compute_true_peak,
)


def test_band_deviations_flat():
    sr = 48_000
    t = np.arange(sr, dtype=np.float64) / sr
    sig = 0.3 * np.sin(2 * np.pi * 1000 * t)[:, None]
    res = analyze_band_deviations(sig, sig.copy(), sr)
    assert res["max_abs_band_delta_db"] is not None
    assert res["max_abs_band_delta_db"] == pytest.approx(0.0, abs=1e-5)
    assert res["bands"]["mid"]["active"] is True
    assert res["bands"]["mid"]["delta_db"] == pytest.approx(0.0, abs=1e-5)


def test_band_deviations_filtered():
    sr = 48_000
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(sr * 2)[:, None] * 0.2

    # Boost low frequencies with 2nd order low-shelf / lowpass
    sos = butter(2, 200, btype="lowpass", fs=sr, output="sos")
    boosted_bass = noise + 2.0 * sosfilt(sos, noise, axis=0)

    res = analyze_band_deviations(noise, boosted_bass, sr)
    # Bass / Sub-bass should exhibit significant positive delta
    bass_delta = res["bands"]["bass"]["delta_db"]
    brilliance_delta = res["bands"]["brilliance"]["delta_db"]

    assert bass_delta is not None and bass_delta > 3.0
    assert brilliance_delta is not None and abs(brilliance_delta) < 1.0


def test_band_deviations_silent_bands_inactive():
    sr = 48_000
    t = np.arange(sr, dtype=np.float64) / sr
    # Pure 1000 Hz tone: sub_bass and brilliance should be inactive
    sig = 0.5 * np.sin(2 * np.pi * 1000 * t)[:, None]
    res = analyze_band_deviations(sig, sig, sr)

    assert res["bands"]["mid"]["active"] is True
    assert res["bands"]["sub_bass"]["active"] is False
    assert res["bands"]["brilliance"]["active"] is False


def test_thd_pure_sine():
    sr = 48_000
    t = np.arange(sr, dtype=np.float64) / sr
    pure_sine = 0.8 * np.sin(2 * np.pi * 1000 * t)

    res = analyze_thd(pure_sine, sr)
    assert res["is_sinusoidal"] is True
    assert res["fundamental_hz"] == pytest.approx(1000.0, abs=2.0)
    assert res["thd_db"] is not None
    assert res["thd_db"] < -90.0  # Ultra-clean FFT window floor
    assert res["thd_percent"] is not None
    assert res["thd_percent"] < 0.01


def test_thd_known_distortion():
    sr = 48_000
    t = np.arange(sr, dtype=np.float64) / sr
    f0 = 1000.0
    # Fundamental: amp 1.0
    # 2nd harmonic (2000 Hz): amp 0.01 (-40 dB, 1.0%)
    # 3rd harmonic (3000 Hz): amp 0.005 (-46 dB, 0.5%)
    # Expected THD = sqrt(0.01^2 + 0.005^2) / 1.0 = sqrt(0.000125) = ~1.118% = -39.03 dB
    signal = (
        1.0 * np.sin(2 * np.pi * f0 * t)
        + 0.01 * np.sin(2 * np.pi * 2 * f0 * t)
        + 0.005 * np.sin(2 * np.pi * 3 * f0 * t)
    )

    res = analyze_thd(signal, sr)
    assert res["is_sinusoidal"] is True
    assert res["fundamental_hz"] == pytest.approx(1000.0, abs=2.0)
    assert res["thd_percent"] == pytest.approx(1.118, abs=0.1)
    assert res["thd_db"] == pytest.approx(-39.03, abs=0.6)


def test_thd_non_sinusoidal_signal():
    sr = 48_000
    rng = np.random.default_rng(99)
    noise = rng.standard_normal(sr) * 0.2

    res = analyze_thd(noise, sr)
    assert res["is_sinusoidal"] is False
    assert res["thd_db"] is None
    assert res["thd_percent"] is None
    assert res["thdn_db"] is None


def test_true_peak_intersample_clipping():
    sr = 48_000
    # Sine at fs/4 (12000 Hz) with 45 degree phase offset:
    # Samples fall at sin(pi/4), sin(3pi/4), sin(5pi/4), sin(7pi/4) -> +/- 0.70711 (-3.01 dBFS).
    # Reconstructed analog peak is 1.0 (0.0 dBTP).
    t = np.arange(sr, dtype=np.float64) / sr
    signal = np.sin(2 * np.pi * 12_000 * t + np.pi / 4.0)

    sample_peak = float(np.max(np.abs(signal)))
    sample_peak_dbfs = 20.0 * np.log10(sample_peak)
    assert sample_peak_dbfs == pytest.approx(-3.01, abs=0.05)

    dbtp, linear = compute_true_peak(signal, sr)
    assert dbtp is not None
    assert dbtp == pytest.approx(0.0, abs=0.15)
    # True-Peak exceeds Sample-Peak by ~3 dB
    assert (dbtp - sample_peak_dbfs) > 2.8


def test_dsp_health_clean():
    t = np.linspace(0, 1, 4800, dtype=np.float64)
    clean = 0.5 * np.sin(2 * np.pi * 440 * t)
    health = check_dsp_health(clean)
    assert health["healthy"] is True
    assert health["has_dc_offset"] is False
    assert health["subnormals_count"] == 0
    assert health["clipped_samples_count"] == 0
    assert health["all_finite"] is True


def test_dsp_health_detects_dc_offset():
    data = np.zeros(1000, dtype=np.float64) + 0.05  # +0.05 DC bias
    health = check_dsp_health(data)
    assert health["has_dc_offset"] is True
    assert health["healthy"] is False
    assert health["dc_offset_amplitude"] == pytest.approx(0.05)


def test_dsp_health_detects_subnormals():
    data = np.zeros(1000, dtype=np.float64)
    data[10] = 1e-40  # Subnormal single precision (< 1.175e-38)
    health = check_dsp_health(data)
    assert health["subnormals_count"] == 1
    assert health["healthy"] is False


def test_dsp_health_detects_anti_phase_stereo_dc_offset():
    # +0.05 on left, -0.05 on right (global average is 0.0, but each channel has DC)
    left = np.zeros((1000, 1), dtype=np.float64) + 0.05
    right = np.zeros((1000, 1), dtype=np.float64) - 0.05
    stereo = np.hstack([left, right])
    health = check_dsp_health(stereo)
    assert health["has_dc_offset"] is True
    assert health["healthy"] is False
    assert health["dc_offset_amplitude"] == pytest.approx(0.05)


def test_dsp_health_detects_clipping():
    data = np.zeros(1000, dtype=np.float64)
    data[5] = 1.0
    data[6] = 1.05
    health = check_dsp_health(data)
    assert health["clipped_samples_count"] == 2
    assert health["healthy"] is False
