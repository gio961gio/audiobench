import json
from pathlib import Path

import numpy as np
import pytest

from audiobench.audio import write_audio
from audiobench.cli import main


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "AudioBench" in captured.out


def test_cli_generate_valid(tmp_path, capsys):
    out_dir = tmp_path / "test_gen"
    exit_code = main(["generate", "--output-dir", str(out_dir), "--sample-rate", "16000", "--duration", "0.1"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Generated 5 deterministic test signals" in captured.out
    assert len(list(out_dir.glob("*.wav"))) == 5


def test_cli_generate_invalid_args(capsys):
    exit_code = main(["generate", "--sample-rate", "-100"])
    assert exit_code == 2
    captured = capsys.readouterr()
    assert "error" in captured.err.lower()


def test_cli_compare_identical_pass(tmp_path, capsys):
    audio_path = tmp_path / "sample.wav"
    data = np.sin(np.linspace(0, 10, 1000, dtype=np.float64))[:, None] * 0.5
    write_audio(audio_path, data, 44_100)

    report_path = tmp_path / "report.html"
    json_path = tmp_path / "report.json"

    exit_code = main(
        [
            "compare",
            "--expected",
            str(audio_path),
            "--actual",
            str(audio_path),
            "--report",
            str(report_path),
            "--json",
            str(json_path),
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "AudioBench result: PASS" in captured.out
    assert report_path.exists()
    assert json_path.exists()


def test_cli_compare_fail_exit_code(tmp_path, capsys):
    ref_path = tmp_path / "ref.wav"
    render_path = tmp_path / "render.wav"
    data = np.sin(np.linspace(0, 10, 1000, dtype=np.float64))[:, None] * 0.5
    write_audio(ref_path, data, 44_100)
    write_audio(render_path, data * 0.1, 44_100)  # Heavy attenuation

    exit_code = main(["compare", "--expected", str(ref_path), "--actual", str(render_path)])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "AudioBench result: FAIL" in captured.out


def test_cli_compare_missing_file_exit_code(tmp_path, capsys):
    exit_code = main(["compare", "--expected", str(tmp_path / "missing.wav"), "--actual", str(tmp_path / "other.wav")])
    assert exit_code == 2
    captured = capsys.readouterr()
    assert "error:" in captured.err


def test_cli_compare_invalid_thresholds(tmp_path, capsys):
    exit_code = main(["compare", "--expected", "a.wav", "--actual", "b.wav", "--min-snr-db", "-5"])
    assert exit_code == 2
    captured = capsys.readouterr()
    assert "error:" in captured.err


def test_cli_batch_valid_pass(tmp_path, capsys):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    wav_path = audio_dir / "sine.wav"
    data = np.sin(np.linspace(0, 50, 4800, dtype=np.float64))[:, None] * 0.4
    write_audio(wav_path, data, 48_000)

    manifest_file = tmp_path / "manifest.json"
    manifest_data = {
        "name": "cli-batch-suite",
        "tests": [
            {
                "id": "t1",
                "description": "Valid sine pass",
                "expected": "audio/sine.wav",
                "actual": "audio/sine.wav",
            }
        ],
    }
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    out_dir = tmp_path / "batch_out"
    exit_code = main(["batch", "--manifest", str(manifest_file), "--output-dir", str(out_dir)])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "AudioBench suite result: PASS" in captured.out
    assert (out_dir / "summary.html").exists()
    assert (out_dir / "summary.json").exists()


def test_cli_batch_fail_exit_code(tmp_path, capsys):
    manifest_file = tmp_path / "failing_manifest.json"
    manifest_data = {
        "name": "failing-suite",
        "tests": [
            {
                "id": "missing-file",
                "expected": "missing.wav",
                "actual": "missing.wav",
            }
        ],
    }
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    exit_code = main(["batch", "--manifest", str(manifest_file)])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "AudioBench suite result: FAIL" in captured.out
    assert "[ERROR] missing-file" in captured.out


def test_cli_batch_invalid_manifest_exit_code(tmp_path, capsys):
    bad_manifest = tmp_path / "bad.json"
    bad_manifest.write_text("invalid json", encoding="utf-8")

    exit_code = main(["batch", "--manifest", str(bad_manifest)])
    assert exit_code == 2
    captured = capsys.readouterr()
    assert "error in manifest:" in captured.err


def test_cli_compare_with_milestone2_options(tmp_path, capsys):
    audio_path = tmp_path / "sample.wav"
    data = np.sin(np.linspace(0, 10, 4800, dtype=np.float64))[:, None] * 0.5
    write_audio(audio_path, data, 48_000)

    exit_code = main(
        [
            "compare",
            "--expected",
            str(audio_path),
            "--actual",
            str(audio_path),
            "--max-band-delta-db",
            "1.5",
            "--max-thd-db",
            "-20.0",
            "--max-true-peak-dbfs",
            "0.0",
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "AudioBench result: PASS" in captured.out
    assert "True-Peak:" in captured.out
