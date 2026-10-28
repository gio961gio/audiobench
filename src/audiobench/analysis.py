"""Advanced DSP analysis: band-wise frequency response, THD/THD+N, True-Peak, and health checks."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.signal import resample_poly, welch
from scipy.signal.windows import blackmanharris

EPSILON = 1e-15
FLOAT32_TINY = float(np.finfo(np.float32).tiny)  # 1.17549435e-38

# Standard 7 acoustic frequency bands (Hz)
AUDIO_BANDS: tuple[tuple[str, float, float], ...] = (
    ("sub_bass", 20.0, 60.0),
    ("bass", 60.0, 250.0),
    ("low_mid", 250.0, 500.0),
    ("mid", 500.0, 2000.0),
    ("high_mid", 2000.0, 4000.0),
    ("presence", 4000.0, 6000.0),
    ("brilliance", 6000.0, 20000.0),
)


def _band_power(freqs: np.ndarray, psd: np.ndarray, f_low: float, f_high: float) -> float:
    mask = (freqs >= f_low) & (freqs < f_high)
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(psd[mask], freqs[mask])) if hasattr(np, "trapezoid") else float(np.trapz(psd[mask], freqs[mask]))


def analyze_band_deviations(
    expected: np.ndarray,
    actual: np.ndarray,
    sample_rate: int,
    *,
    min_active_dbfs: float = -90.0,
    noise_floor_dbfs: float = -120.0,
) -> dict[str, Any]:
    """Compute per-band energy deltas (actual vs expected) across 7 acoustic bands.

    Bands below min_active_dbfs in both signals are marked as inactive to avoid
    spurious numerical noise in silent/band-limited regions.
    """
    ref = np.asarray(expected, dtype=np.float64)
    act = np.asarray(actual, dtype=np.float64)
    if ref.ndim == 1:
        ref = ref[:, None]
    if act.ndim == 1:
        act = act[:, None]

    common = min(len(ref), len(act))
    if common < 64:
        return {"bands": {}, "max_abs_band_delta_db": None}

    ref = ref[:common]
    act = act[:common]

    nperseg = min(common, 4096)
    if nperseg < 32:
        return {"bands": {}, "max_abs_band_delta_db": None}

    # Compute Welch PSD averaged over channels
    freqs, pxx_ref = welch(ref, fs=sample_rate, window="hann", nperseg=nperseg, axis=0)
    _, pxx_act = welch(act, fs=sample_rate, window="hann", nperseg=nperseg, axis=0)
    psd_ref = np.mean(pxx_ref, axis=1)
    psd_act = np.mean(pxx_act, axis=1)

    min_active_pwr = 10.0 ** (min_active_dbfs / 10.0)
    noise_floor_pwr = 10.0 ** (noise_floor_dbfs / 10.0)
    nyquist = sample_rate / 2.0

    bands_result: dict[str, dict[str, Any]] = {}
    active_deltas: list[float] = []

    for name, f_low, f_high in AUDIO_BANDS:
        if f_low >= nyquist:
            continue
        effective_high = min(f_high, nyquist)
        p_ref = _band_power(freqs, psd_ref, f_low, effective_high)
        p_act = _band_power(freqs, psd_act, f_low, effective_high)

        ref_dbfs = 10.0 * np.log10(max(p_ref, EPSILON))
        act_dbfs = 10.0 * np.log10(max(p_act, EPSILON))

        is_active = (p_ref >= min_active_pwr) or (p_act >= min_active_pwr)
        if is_active:
            clamped_ref = max(p_ref, noise_floor_pwr)
            clamped_act = max(p_act, noise_floor_pwr)
            delta_db = float(10.0 * np.log10(clamped_act / clamped_ref))
            active_deltas.append(abs(delta_db))
        else:
            delta_db = None

        bands_result[name] = {
            "f_low_hz": f_low,
            "f_high_hz": effective_high,
            "active": is_active,
            "expected_dbfs": float(ref_dbfs) if p_ref > EPSILON else None,
            "actual_dbfs": float(act_dbfs) if p_act > EPSILON else None,
            "delta_db": delta_db,
        }

    max_delta = max(active_deltas) if active_deltas else 0.0
    return {
        "bands": bands_result,
        "max_abs_band_delta_db": float(max_delta) if active_deltas else None,
    }


def analyze_thd(
    signal: np.ndarray,
    sample_rate: int,
    *,
    max_harmonics: int = 10,
    dominant_ratio_threshold: float = 0.1,  # -10 dB minimum fundamental/total
) -> dict[str, Any]:
    """Estimate THD and THD+N on sinusoidal signals using 4-term Blackman-Harris window.

    If the signal is not predominantly sinusoidal, returns is_sinusoidal=False
    and numeric metric fields as None (serializable to JSON as null).
    """
    arr = np.asarray(signal, dtype=np.float64)
    if arr.ndim == 2:
        arr = np.mean(arr, axis=1)
    frames = len(arr)
    if frames < 256:
        return {
            "is_sinusoidal": False,
            "reason": "Signal too short for reliable harmonic analysis",
            "fundamental_hz": None,
            "thd_db": None,
            "thd_percent": None,
            "thdn_db": None,
        }

    # Bounded FFT size for consistent frequency binning
    n = min(frames, 65536)
    # Ensure even length
    if n % 2 != 0:
        n -= 1
    chunk = arr[:n]

    win = blackmanharris(n)
    win_pwr = float(np.sum(win**2))
    if win_pwr <= EPSILON:
        return {"is_sinusoidal": False, "reason": "Silent window", "fundamental_hz": None, "thd_db": None, "thd_percent": None, "thdn_db": None}

    spectrum = np.fft.rfft(chunk * win)
    mag_sq = (np.abs(spectrum) ** 2) / win_pwr
    n_bins = len(mag_sq)

    # Exclude DC region (bins 0..2)
    dc_guard = 3
    if dc_guard >= n_bins - 4:
        return {"is_sinusoidal": False, "reason": "Insufficient spectral bins", "fundamental_hz": None, "thd_db": None, "thd_percent": None, "thdn_db": None}

    k0 = int(np.argmax(mag_sq[dc_guard:])) + dc_guard
    if k0 <= dc_guard or k0 >= n_bins - 2:
        return {"is_sinusoidal": False, "reason": "No clear fundamental found outside DC", "fundamental_hz": None, "thd_db": None, "thd_percent": None, "thdn_db": None}

    # Parabolic interpolation on log-magnitude for accurate peak frequency
    eps = 1e-12
    y1 = float(np.log(mag_sq[k0 - 1] + eps))
    y2 = float(np.log(mag_sq[k0] + eps))
    y3 = float(np.log(mag_sq[k0 + 1] + eps))
    denom = y1 - 2.0 * y2 + y3
    delta_k = -0.5 * (y3 - y1) / denom if abs(denom) > 1e-6 else 0.0
    delta_k = float(np.clip(delta_k, -0.5, 0.5))
    refined_k0 = float(k0) + delta_k
    f0 = refined_k0 * (sample_rate / n)

    # Dynamic bin radius to prevent overlap on low-frequency fundamentals
    bin_radius = min(4, max(1, int(refined_k0 // 3)))
    low_f0 = max(0, k0 - bin_radius)
    high_f0 = min(n_bins, k0 + bin_radius + 1)
    e_fund = float(np.sum(mag_sq[low_f0:high_f0]))

    # Total non-DC energy
    e_total = float(np.sum(mag_sq[dc_guard:]))
    if e_total <= EPSILON or (e_fund / e_total) < dominant_ratio_threshold:
        return {
            "is_sinusoidal": False,
            "reason": "Signal is non-sinusoidal (fundamental not dominant)",
            "fundamental_hz": float(f0),
            "thd_db": None,
            "thd_percent": None,
            "thdn_db": None,
        }

    # Harmonic energies
    harmonic_energies: list[float] = []
    used_bins: set[int] = set(range(low_f0, high_f0))

    for m in range(2, max_harmonics + 1):
        target_k = int(round(m * refined_k0))
        if target_k - bin_radius >= n_bins:
            break
        h_low = max(0, target_k - bin_radius)
        h_high = min(n_bins, target_k + bin_radius + 1)
        if h_low >= n_bins or h_high <= h_low:
            break

        # Find local maximum near expected harmonic
        local_k = h_low + int(np.argmax(mag_sq[h_low:h_high]))
        m_low = max(0, local_k - bin_radius)
        m_high = min(n_bins, local_k + bin_radius + 1)
        new_bins = [b for b in range(m_low, m_high) if b not in used_bins]
        e_harm = float(np.sum(mag_sq[new_bins])) if new_bins else 0.0
        if e_harm > 0:
            harmonic_energies.append(e_harm)
            used_bins.update(new_bins)

    total_harmonics_pwr = sum(harmonic_energies)

    # Residual noise energy: all non-DC bins outside fundamental and harmonic regions
    noise_bins = [b for b in range(dc_guard, n_bins) if b not in used_bins]
    e_noise = float(np.sum(mag_sq[noise_bins])) if noise_bins else 0.0

    if total_harmonics_pwr <= EPSILON:
        thd_db = -120.0
        thd_percent = 0.0
    else:
        ratio = total_harmonics_pwr / e_fund
        thd_db = float(10.0 * np.log10(ratio))
        thd_percent = float(100.0 * np.sqrt(ratio))

    thdn_ratio = (total_harmonics_pwr + e_noise) / e_fund
    thdn_db = float(10.0 * np.log10(max(thdn_ratio, EPSILON)))

    return {
        "is_sinusoidal": True,
        "reason": None,
        "fundamental_hz": float(f0),
        "thd_db": thd_db,
        "thd_percent": thd_percent,
        "thdn_db": thdn_db,
    }


def compute_true_peak(
    signal: np.ndarray,
    sample_rate: int,
) -> tuple[float | None, float | None]:
    """Compute True-Peak in dBTP and linear according to ITU-R BS.1770-4 4x/2x oversampling."""
    arr = np.asarray(signal, dtype=np.float64)
    if arr.size == 0:
        return None, None
    if arr.ndim == 1:
        arr = arr[:, None]

    # ITU-R BS.1770: 4x oversampling for <= 48kHz, 2x for > 48kHz
    up_factor = 4 if sample_rate <= 48_000 else 2
    oversampled = resample_poly(arr, up=up_factor, down=1, axis=0)
    max_linear = float(np.max(np.abs(oversampled)))

    if max_linear <= 0 or not math.isfinite(max_linear):
        return None, None

    dbtp = float(20.0 * np.log10(max_linear))
    return dbtp, max_linear


def check_dsp_health(signal: np.ndarray) -> dict[str, Any]:
    """Analyze DSP sanity: DC offset, subnormal floats, clipping, and finite checks."""
    arr = np.asarray(signal, dtype=np.float64)
    if arr.size == 0:
        return {
            "has_dc_offset": False,
            "dc_offset_amplitude": 0.0,
            "dc_offset_dbfs": None,
            "subnormals_count": 0,
            "clipped_samples_count": 0,
            "clipped_samples_ratio": 0.0,
            "all_finite": True,
            "healthy": True,
        }

    all_finite = bool(np.isfinite(arr).all())
    if not all_finite:
        return {
            "has_dc_offset": False,
            "dc_offset_amplitude": 0.0,
            "dc_offset_dbfs": None,
            "subnormals_count": 0,
            "clipped_samples_count": 0,
            "clipped_samples_ratio": 0.0,
            "all_finite": False,
            "healthy": False,
        }

    if arr.ndim == 1:
        arr = arr[:, None]

    # Check DC offset per channel to prevent inter-channel cancellation (e.g. +DC on L, -DC on R)
    channel_means = np.mean(arr, axis=0)
    abs_max_dc = float(np.max(np.abs(channel_means)))
    dc_dbfs = float(20.0 * np.log10(abs_max_dc)) if abs_max_dc > EPSILON else None
    has_dc = abs_max_dc > 1e-4  # Greater than -80 dBFS

    # Subnormal check: non-zero numbers smaller than float32 tiny limit
    abs_arr = np.abs(arr)
    subnormals_mask = (abs_arr > 0) & (abs_arr < FLOAT32_TINY)
    subnormals_count = int(np.count_nonzero(subnormals_mask))

    # Clipping check: samples at or exceeding 1.0 (-1e-5 headroom margin)
    clipped_mask = abs_arr >= (1.0 - 1e-5)
    clipped_count = int(np.count_nonzero(clipped_mask))
    clipped_ratio = float(clipped_count / arr.size)

    healthy = not has_dc and subnormals_count == 0 and not (clipped_count > 0)

    return {
        "has_dc_offset": has_dc,
        "dc_offset_amplitude": abs_max_dc,
        "dc_offset_dbfs": dc_dbfs,
        "subnormals_count": subnormals_count,
        "clipped_samples_count": clipped_count,
        "clipped_samples_ratio": clipped_ratio,
        "all_finite": all_finite,
        "healthy": healthy,
    }
