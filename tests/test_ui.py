import io
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from audiobench.app import _array_to_wav_bytes, _compute_difference_signal
from audiobench.cli import build_parser, main


def test_array_to_wav_bytes_roundtrip():
    sr = 48000
    t = np.linspace(0, 0.05, int(sr * 0.05), endpoint=False)
    data = (np.sin(2 * np.pi * 440 * t) * 0.5)[:, None]

    wav_bytes = _array_to_wav_bytes(data, sr)
    assert isinstance(wav_bytes, bytes)
    assert len(wav_bytes) > 44  # Valid WAV header + data

    # Read back and verify samples are intact within 16-bit PCM quantization
    read_data, read_sr = sf.read(io.BytesIO(wav_bytes), dtype="float64", always_2d=True)
    assert read_sr == sr
    np.testing.assert_allclose(read_data, data, atol=1e-4)


def test_compute_difference_signal_exact_zero():
    sr = 48000
    data = np.ones((100, 1), dtype=np.float64) * 0.3
    diff, eff_boost, limited = _compute_difference_signal(data, data.copy(), sr, boost_db=24.0)

    assert limited is False
    assert eff_boost == 24.0
    np.testing.assert_allclose(diff, 0.0)


def test_compute_difference_signal_safety_limiter():
    sr = 48000
    # Two signals with high discrepancy (e.g. +0.8 vs -0.8 -> diff = -1.6)
    exp = np.ones((100, 1), dtype=np.float64) * 0.8
    act = np.ones((100, 1), dtype=np.float64) * -0.8

    # Desired boost of 24 dB would cause massive digital clipping (over 30 dBFS)
    diff, eff_boost, limited = _compute_difference_signal(exp, act, sr, boost_db=24.0)

    assert limited is True
    assert eff_boost < 0.0  # Had to be attenuated to fit within -0.5 dBFS
    assert np.max(np.abs(diff)) <= 0.99  # Guaranteed safety ceiling


def test_compute_difference_signal_latency_compensation():
    sr = 48000
    exp = np.zeros((100, 1), dtype=np.float64)
    exp[10, 0] = 1.0  # Impulse at sample 10

    act = np.zeros((100, 1), dtype=np.float64)
    act[15, 0] = 1.0  # Impulse delayed by 5 samples

    # Without alignment: diff is non-zero (clamped to safety ceiling 0.94406)
    diff_unaligned, _, was_limited = _compute_difference_signal(exp, act, sr, align_delay_samples=0, boost_db=0.0)
    assert was_limited is True
    assert np.max(np.abs(diff_unaligned)) == pytest.approx(0.94406, abs=1e-4)

    # With alignment by 5 samples: impulses align and null out perfectly
    diff_aligned, _, _ = _compute_difference_signal(exp, act, sr, align_delay_samples=5, boost_db=0.0)
    np.testing.assert_allclose(diff_aligned, 0.0)


def test_cli_ui_parser():
    parser = build_parser()
    args = parser.parse_args(["ui", "--port", "8888", "--host", "0.0.0.0", "--no-browser"])
    assert args.command == "ui"
    assert args.port == 8888
    assert args.host == "0.0.0.0"
    assert args.no_browser is True


def test_cli_ui_missing_streamlit(capsys):
    with patch("importlib.util.find_spec", return_value=None):
        exit_code = main(["ui"])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "requires 'streamlit'" in captured.err
