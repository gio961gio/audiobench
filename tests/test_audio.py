import numpy as np
import soundfile as sf
import pytest

from audiobench.audio import load_audio, make_test_signals, write_audio, write_test_signals


def test_signal_generator_is_deterministic():
    first = make_test_signals(sample_rate=8000, duration_s=0.25, seed=9)
    second = make_test_signals(sample_rate=8000, duration_s=0.25, seed=9)
    assert set(first) == {"impulse", "sine_1khz", "sweep", "white_noise", "silence"}
    np.testing.assert_array_equal(first["white_noise"], second["white_noise"])
    assert first["impulse"].shape == (2000, 1)


def test_write_and_load_audio_roundtrip(tmp_path):
    path = tmp_path / "roundtrip.wav"
    data = np.zeros((1024, 1), dtype=np.float64)
    data[0, 0] = 0.5
    write_audio(path, data, 48_000)
    loaded, sample_rate = load_audio(path)
    assert sample_rate == 48_000
    assert loaded.shape == data.shape
    assert loaded[0, 0] == pytest.approx(0.5, abs=1e-6)


def test_generate_writes_five_files(tmp_path):
    paths = write_test_signals(tmp_path / "signals", sample_rate=8000, duration_s=0.1)
    assert len(paths) == 5
    assert all(path.exists() for path in paths)
