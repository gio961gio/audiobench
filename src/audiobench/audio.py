"""Audio file I/O and deterministic test-signal generation."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import soundfile as sf
from scipy.signal import chirp

AudioArray = np.ndarray


def load_audio(path: str | Path) -> tuple[AudioArray, int]:
    """Load audio as float64 with shape (frames, channels)."""
    audio_path = Path(path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not audio_path.is_file():
        raise ValueError(f"Expected an audio file, got: {audio_path}")

    data, sample_rate = sf.read(audio_path, dtype="float64", always_2d=True)
    if data.size == 0:
        raise ValueError(f"Audio file is empty: {audio_path}")
    if not np.isfinite(data).all():
        raise ValueError(f"Audio contains NaN or infinite samples: {audio_path}")
    return data, int(sample_rate)


def write_audio(path: str | Path, data: AudioArray, sample_rate: int) -> None:
    """Write mono/stereo float data as a 24-bit PCM WAV file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(data, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("Audio data must have shape (frames,) or (frames, channels)")
    if not np.isfinite(values).all():
        raise ValueError("Refusing to write NaN or infinite samples")
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive")
    sf.write(out, values, sample_rate, subtype="PCM_24", format="WAV")


def make_test_signals(
    sample_rate: int = 48_000,
    duration_s: float = 1.0,
    seed: int = 7,
) -> Mapping[str, AudioArray]:
    """Create deterministic mono test signals normalized to sensible headroom.

    Signals include an impulse, a 1 kHz sine, a logarithmic sweep, seeded white
    noise, and silence. Arrays are returned as (frames, 1) float64 data.
    """
    if sample_rate < 100:
        raise ValueError("Sample rate must be at least 100 Hz")
    if duration_s <= 0:
        raise ValueError("Duration must be greater than zero")

    frame_count = max(1, int(round(sample_rate * duration_s)))
    t = np.arange(frame_count, dtype=np.float64) / sample_rate
    signals: dict[str, AudioArray] = {}

    impulse = np.zeros(frame_count, dtype=np.float64)
    impulse[0] = 0.8
    signals["impulse"] = impulse[:, None]

    sine_frequency = min(1_000.0, sample_rate * 0.1)
    sine = 0.5 * np.sin(2.0 * np.pi * sine_frequency * t)
    signals["sine_1khz"] = sine[:, None]

    sweep_end = min(20_000.0, sample_rate * 0.45)
    sweep_start = min(20.0, sweep_end / 10.0)
    sweep = 0.4 * chirp(
        t,
        f0=sweep_start,
        f1=sweep_end,
        t1=max(duration_s, 1.0 / sample_rate),
        method="logarithmic",
    )
    signals["sweep"] = sweep[:, None]

    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(frame_count)
    noise -= np.mean(noise)
    noise_peak = float(np.max(np.abs(noise)))
    if noise_peak > 0:
        noise = noise * (0.35 / noise_peak)
    signals["white_noise"] = noise[:, None]
    signals["silence"] = np.zeros((frame_count, 1), dtype=np.float64)
    return signals


def write_test_signals(
    output_dir: str | Path,
    sample_rate: int = 48_000,
    duration_s: float = 1.0,
    seed: int = 7,
) -> list[Path]:
    """Generate all built-in test signals into *output_dir*."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, data in make_test_signals(sample_rate, duration_s, seed).items():
        target = directory / f"{name}.wav"
        write_audio(target, data, sample_rate)
        paths.append(target)
    return paths
