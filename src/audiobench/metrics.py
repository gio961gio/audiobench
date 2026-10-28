"""Signal metrics and threshold evaluation for offline render comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import correlate, correlation_lags

EPSILON = 1e-15


@dataclass(frozen=True)
class Thresholds:
    """Quality-gate thresholds used to mark a render comparison PASS/FAIL."""

    min_snr_db: float = 60.0
    max_delay_ms: float = 0.5
    max_duration_delta_ms: float = 1.0
    max_error_dbfs: float = -80.0


def _rms(data: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(data, dtype=np.float64)))) if data.size else 0.0


def _dbfs(value: float) -> float | None:
    if value <= 0 or not np.isfinite(value):
        return None
    return float(20.0 * np.log10(value))


def _corrcoef(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 2 or b.size < 2:
        return None
    x = a.reshape(-1).astype(np.float64, copy=False)
    y = b.reshape(-1).astype(np.float64, copy=False)
    x = x - np.mean(x)
    y = y - np.mean(y)
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator <= EPSILON:
        return None
    return float(np.clip(np.dot(x, y) / denominator, -1.0, 1.0))


def estimate_delay_samples(
    expected: np.ndarray,
    actual: np.ndarray,
    sample_rate: int,
    max_lag_ms: float = 100.0,
) -> tuple[int | None, float | None]:
    """Estimate signed sample delay using channel-wise normalized cross-correlation.

    Positive delay means that *actual* occurs later than *expected*. The value
    is an estimate; periodic signals such as pure tones can make delay ambiguous.
    """
    if expected.ndim != 2 or actual.ndim != 2 or expected.shape[1] != actual.shape[1]:
        return None, None
    common = min(expected.shape[0], actual.shape[0], sample_rate * 5)
    if common < 16:
        return None, None

    # Keep FFT correlation bounded for longer renders. Delay resolution is the
    # original sample rate unless this decimation limit is reached.
    stride = max(1, int(np.ceil(common / 120_000)))
    x = expected[:common:stride]
    y = actual[:common:stride]
    lag_limit = max(0, int(round(max_lag_ms * sample_rate / 1000.0 / stride)))
    if lag_limit == 0:
        return None, None

    combined: np.ndarray | None = None
    lags: np.ndarray | None = None
    informative_channels = 0
    for channel in range(expected.shape[1]):
        ref = x[:, channel].astype(np.float64, copy=False)
        out = y[:, channel].astype(np.float64, copy=False)
        ref = ref - np.mean(ref)
        out = out - np.mean(out)
        norm = float(np.linalg.norm(ref) * np.linalg.norm(out))
        if norm <= EPSILON:
            continue
        cross = correlate(out, ref, mode="full", method="fft") / norm
        channel_lags = correlation_lags(len(out), len(ref), mode="full")
        if combined is None:
            combined = np.zeros_like(cross, dtype=np.float64)
            lags = channel_lags
        combined += np.abs(cross)
        informative_channels += 1

    if combined is None or lags is None or informative_channels == 0:
        return None, None

    allowed = np.abs(lags) <= lag_limit
    if not np.any(allowed):
        return None, None
    allowed_scores = combined[allowed] / informative_channels
    allowed_lags = lags[allowed]
    best_score = float(np.max(allowed_scores))
    if best_score < 0.1:
        return None, best_score

    # Prefer the smallest absolute lag among near-equal peaks. This avoids
    # reporting a full period for a sine wave when zero lag is equally likely.
    near_peak = np.flatnonzero(allowed_scores >= best_score - max(1e-3, best_score * 0.01))
    chosen = int(near_peak[np.argmin(np.abs(allowed_lags[near_peak]))])
    estimated = int(allowed_lags[chosen] * stride)
    return estimated, best_score


def _aligned_pair(
    expected: np.ndarray, actual: np.ndarray, delay_samples: int | None
) -> tuple[np.ndarray, np.ndarray]:
    if delay_samples is None:
        common = min(len(expected), len(actual))
        return expected[:common], actual[:common]
    if delay_samples > 0:
        available = max(0, len(actual) - delay_samples)
        common = min(len(expected), available)
        return expected[:common], actual[delay_samples : delay_samples + common]
    if delay_samples < 0:
        lead = -delay_samples
        common = min(max(0, len(expected) - lead), len(actual))
        return expected[lead : lead + common], actual[:common]
    common = min(len(expected), len(actual))
    return expected[:common], actual[:common]


def _spectral_differences(expected: np.ndarray, actual: np.ndarray) -> tuple[float | None, float | None]:
    """Estimate absolute spectral difference and gain-normalized spectral-shape difference.

    This is a diagnostic magnitude-spectrum comparison, not a calibrated transfer-
    function measurement. A known stimulus and a documented measurement protocol
    are required for a defensible frequency-response claim.
    """
    common = min(len(expected), len(actual))
    if common < 32:
        return None, None
    ref = expected[:common]
    out = actual[:common]
    window = np.hanning(common).astype(np.float64)[:, None]
    window_sum = float(np.sum(window))
    if window_sum <= EPSILON:
        return None, None

    ref_spec = np.mean(np.abs(np.fft.rfft(ref * window, axis=0)), axis=1) / window_sum
    out_spec = np.mean(np.abs(np.fft.rfft(out * window, axis=0)), axis=1) / window_sum
    floor = 1e-12
    ref_db = 20.0 * np.log10(np.maximum(ref_spec, floor))
    out_db = 20.0 * np.log10(np.maximum(out_spec, floor))
    active = np.maximum(ref_db, out_db) > -100.0
    if not np.any(active):
        return 0.0, 0.0
    mean_spectral_delta = float(np.mean(np.abs(ref_db[active] - out_db[active])))

    ref_rms = _rms(ref)
    out_rms = _rms(out)
    if ref_rms <= EPSILON or out_rms <= EPSILON:
        shape_delta = None
    else:
        ref_shape = ref_db - 20.0 * np.log10(ref_rms)
        out_shape = out_db - 20.0 * np.log10(out_rms)
        shape_active = np.maximum(ref_shape, out_shape) > -80.0
        shape_delta = (
            float(np.mean(np.abs(ref_shape[shape_active] - out_shape[shape_active])))
            if np.any(shape_active)
            else 0.0
        )
    return mean_spectral_delta, shape_delta


def compare_arrays(
    expected: np.ndarray,
    actual: np.ndarray,
    sample_rate: int,
    *,
    max_lag_ms: float = 100.0,
) -> dict[str, Any]:
    """Compute useful render-to-render metrics without resampling the inputs."""
    expected = np.asarray(expected, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    if expected.ndim == 1:
        expected = expected[:, None]
    if actual.ndim == 1:
        actual = actual[:, None]
    if expected.ndim != 2 or actual.ndim != 2:
        raise ValueError("Expected and actual arrays must have shape (frames, channels)")
    if expected.size == 0 or actual.size == 0:
        raise ValueError("Expected and actual arrays must not be empty")
    if expected.shape[1] != actual.shape[1]:
        raise ValueError(
            f"Channel count mismatch: expected {expected.shape[1]}, actual {actual.shape[1]}"
        )
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive")
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError("Audio arrays must contain only finite values")

    expected_frames, channels = expected.shape
    actual_frames = actual.shape[0]
    common = min(expected_frames, actual_frames)
    raw_expected = expected[:common]
    raw_actual = actual[:common]
    error = raw_actual - raw_expected
    reference_rms = _rms(raw_expected)
    actual_rms = _rms(raw_actual)
    error_rms = _rms(error)
    delay_samples, delay_correlation = estimate_delay_samples(
        expected, actual, sample_rate, max_lag_ms=max_lag_ms
    )
    aligned_expected, aligned_actual = _aligned_pair(expected, actual, delay_samples)
    aligned_error_rms = _rms(aligned_actual - aligned_expected)
    aligned_reference_rms = _rms(aligned_expected)
    aligned_identical = (
        aligned_expected.shape == aligned_actual.shape
        and np.array_equal(aligned_expected, aligned_actual)
    )
    aligned_snr = (
        float(20 * np.log10(aligned_reference_rms / aligned_error_rms))
        if aligned_error_rms > EPSILON and aligned_reference_rms > EPSILON
        else (None if aligned_error_rms <= EPSILON else None)
    )

    expected_peak = float(np.max(np.abs(expected)))
    actual_peak = float(np.max(np.abs(actual)))
    identical = expected.shape == actual.shape and np.array_equal(expected, actual)
    if identical:
        snr_db: float | None = None  # Interpreted as +infinity / exact match.
        relative_error_db: float | None = None
    elif error_rms <= EPSILON:
        # Prefix matches exactly, but the duration may differ.
        snr_db = None
        relative_error_db = None
    else:
        snr_db = (
            float(20.0 * np.log10(reference_rms / error_rms))
            if reference_rms > EPSILON
            else None
        )
        relative_error_db = (
            float(20.0 * np.log10(error_rms / reference_rms))
            if reference_rms > EPSILON
            else None
        )

    corr = _corrcoef(raw_expected, raw_actual)
    spectral_delta, spectral_shape_error = _spectral_differences(raw_expected, raw_actual)
    return {
        "sample_rate_hz": int(sample_rate),
        "channels": int(channels),
        "expected_frames": int(expected_frames),
        "actual_frames": int(actual_frames),
        "expected_duration_s": float(expected_frames / sample_rate),
        "actual_duration_s": float(actual_frames / sample_rate),
        "duration_delta_ms": float((actual_frames - expected_frames) / sample_rate * 1000.0),
        "expected_peak_dbfs": _dbfs(expected_peak),
        "actual_peak_dbfs": _dbfs(actual_peak),
        "expected_rms_dbfs": _dbfs(_rms(expected)),
        "actual_rms_dbfs": _dbfs(_rms(actual)),
        "peak_delta_db": (
            float(20.0 * np.log10(actual_peak / expected_peak))
            if expected_peak > EPSILON and actual_peak > EPSILON
            else None
        ),
        "rms_delta_db": (
            float(20.0 * np.log10(actual_rms / reference_rms))
            if reference_rms > EPSILON and actual_rms > EPSILON
            else None
        ),
        "error_rms_dbfs": _dbfs(error_rms),
        "snr_db": snr_db,
        "relative_error_db": relative_error_db,
        "correlation": corr,
        "mean_spectral_delta_db": spectral_delta,
        "spectral_shape_error_db": spectral_shape_error,
        "max_abs_error": float(np.max(np.abs(error))) if error.size else 0.0,
        "estimated_delay_samples": delay_samples,
        "estimated_delay_ms": (
            float(delay_samples / sample_rate * 1000.0) if delay_samples is not None else None
        ),
        "delay_correlation": delay_correlation,
        "aligned_snr_db": aligned_snr,
        "aligned_samples_identical": bool(aligned_identical),
        "identical_samples": bool(identical),
        "duration_match": bool(expected_frames == actual_frames),
    }


def evaluate_thresholds(
    metrics: dict[str, Any], thresholds: Thresholds | None = None
) -> dict[str, Any]:
    """Return individual quality-gate checks and an overall PASS/WARN/FAIL."""
    limits = thresholds or Thresholds()
    checks: dict[str, dict[str, Any]] = {}

    duration_delta = abs(float(metrics["duration_delta_ms"]))
    duration_pass = duration_delta <= limits.max_duration_delta_ms
    checks["duration"] = {
        "status": "PASS" if duration_pass else "FAIL",
        "observed": float(metrics["duration_delta_ms"]),
        "limit": limits.max_duration_delta_ms,
        "message": "Duration difference is within the configured tolerance."
        if duration_pass
        else "Duration difference exceeds the configured tolerance.",
    }

    delay_ms = metrics.get("estimated_delay_ms")
    if delay_ms is None:
        no_signal = metrics.get("expected_rms_dbfs") is None and metrics.get("actual_rms_dbfs") is None
        delay_status = "N/A" if no_signal else "WARN"
        delay_message = (
            "Latency is not measurable for silent renders."
            if no_signal
            else "Latency could not be estimated reliably; use a broadband test signal."
        )
        checks["latency"] = {
            "status": delay_status,
            "observed": None,
            "limit": limits.max_delay_ms,
            "message": delay_message,
        }
    else:
        delay_pass = abs(float(delay_ms)) <= limits.max_delay_ms
        checks["latency"] = {
            "status": "PASS" if delay_pass else "FAIL",
            "observed": float(delay_ms),
            "limit": limits.max_delay_ms,
            "message": "Estimated delay is within the configured tolerance."
            if delay_pass
            else "Estimated delay exceeds the configured tolerance.",
        }

    if metrics.get("identical_samples"):
        fidelity_status = "PASS"
        fidelity_observed = "exact match"
        fidelity_message = "The two renders are sample-identical."
    elif metrics.get("snr_db") is not None:
        snr = float(metrics["snr_db"])
        fidelity_status = "PASS" if snr >= limits.min_snr_db else "FAIL"
        fidelity_observed = snr
        fidelity_message = (
            "Signal-to-error ratio meets the configured minimum."
            if fidelity_status == "PASS"
            else "Signal-to-error ratio is below the configured minimum."
        )
    else:
        error_dbfs = metrics.get("error_rms_dbfs")
        error_dbfs = float(error_dbfs) if error_dbfs is not None else None
        # For a silent reference, SNR is undefined. Gate on absolute error level.
        if error_dbfs is None:
            fidelity_status = "PASS" if float(metrics["max_abs_error"]) <= EPSILON else "FAIL"
            fidelity_observed = "no measurable signal" if fidelity_status == "PASS" else None
        else:
            fidelity_status = "PASS" if error_dbfs <= limits.max_error_dbfs else "FAIL"
            fidelity_observed = error_dbfs
        fidelity_message = (
            "Absolute error is within the configured limit."
            if fidelity_status == "PASS"
            else "Absolute error exceeds the configured limit for a silent/near-silent reference."
        )
    checks["fidelity"] = {
        "status": fidelity_status,
        "observed": fidelity_observed,
        "limit": limits.min_snr_db if metrics.get("snr_db") is not None else limits.max_error_dbfs,
        "message": fidelity_message,
    }

    statuses = [item["status"] for item in checks.values()]
    overall = "FAIL" if "FAIL" in statuses else ("WARN" if "WARN" in statuses else "PASS")
    return {"status": overall, "checks": checks}
