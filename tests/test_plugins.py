import json
import sys
from pathlib import Path

import numpy as np
import pytest

from audiobench.audio import load_audio, write_audio
from audiobench.batch import ManifestError, load_manifest, run_batch
from audiobench.cli import main
from audiobench.plugins import (
    BuiltinDspAdapter,
    CliPluginAdapter,
    PluginError,
    VST3PluginAdapter,
    get_plugin_adapter,
)


def _generate_test_sine(duration_s: float = 0.1, sr: int = 48000, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False, dtype=np.float64)
    return (np.sin(2 * np.pi * freq * t) * 0.5)[:, None]


# -----------------------------------------------------------------------------
# 1. BuiltinDspAdapter tests
# -----------------------------------------------------------------------------


def test_builtin_gain_linear_and_db(tmp_path):
    arr = np.ones((100, 1), dtype=np.float64)

    adapter_linear = BuiltinDspAdapter(effect_type="gain", default_parameters={"multiplier": 0.5})
    out_linear = adapter_linear.render_array(arr, 48000)
    np.testing.assert_allclose(out_linear, 0.5)

    adapter_db = BuiltinDspAdapter(effect_type="gain")
    out_db = adapter_db.render_array(arr, 48000, parameters={"gain_db": -6.0205999})
    np.testing.assert_allclose(out_db, 0.5, rtol=1e-4)


def test_builtin_filters(tmp_path):
    sr = 48000
    # 100 Hz low frequency + 10 kHz high frequency
    t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False, dtype=np.float64)
    sig_low = np.sin(2 * np.pi * 100 * t)[:, None] * 0.4
    sig_high = np.sin(2 * np.pi * 10000 * t)[:, None] * 0.4
    composite = sig_low + sig_high

    # Lowpass at 1 kHz should attenuate 10 kHz component
    lp = BuiltinDspAdapter(effect_type="lowpass", default_parameters={"cutoff_hz": 1000.0, "order": 4})
    out_lp = lp.render_array(composite, sr)
    rms_lp = np.sqrt(np.mean(out_lp**2))
    rms_orig = np.sqrt(np.mean(composite**2))
    assert rms_lp < rms_orig

    # Highpass at 2 kHz should attenuate 100 Hz component
    hp = BuiltinDspAdapter(effect_type="highpass", default_parameters={"cutoff_hz": 2000.0, "order": 4})
    out_hp = hp.render_array(composite, sr)
    rms_hp = np.sqrt(np.mean(out_hp**2))
    assert rms_hp < rms_orig


def test_builtin_delay():
    sr = 48000
    arr = np.zeros((100, 1), dtype=np.float64)
    arr[0, 0] = 1.0  # Impulse at frame 0

    adapter_samples = BuiltinDspAdapter(effect_type="delay", default_parameters={"delay_samples": 10})
    out_samples = adapter_samples.render_array(arr, sr)
    assert out_samples[10, 0] == 1.0
    assert out_samples[0, 0] == 0.0

    # Delay in ms: 1 ms at 48 kHz = 48 samples
    adapter_ms = BuiltinDspAdapter(effect_type="delay")
    out_ms = adapter_ms.render_array(arr, sr, parameters={"delay_ms": 1.0})
    assert out_ms[48, 0] == 1.0
    assert out_ms[0, 0] == 0.0


def test_builtin_soft_clip():
    sr = 48000
    # High amplitude sine
    sine = _generate_test_sine(0.05, sr, freq=500.0) * 4.0  # Peak is 2.0
    adapter = BuiltinDspAdapter(effect_type="soft_clip", default_parameters={"drive_db": 6.0})
    clipped = adapter.render_array(sine, sr)
    # Peak must be strictly compressed within (-1.0, 1.0) by tanh
    assert np.max(np.abs(clipped)) < 1.0


def test_builtin_invalid_effect():
    adapter = BuiltinDspAdapter(effect_type="nonexistent_dsp")
    with pytest.raises(PluginError, match="Unknown built-in DSP effect"):
        adapter.render_array(np.ones((10, 1)), 48000)


def test_builtin_render_file(tmp_path):
    in_wav = tmp_path / "in.wav"
    out_wav = tmp_path / "out.wav"
    data = _generate_test_sine(0.05, 48000)
    write_audio(in_wav, data, 48000)

    adapter = BuiltinDspAdapter("gain", {"gain_db": -6.0})
    result_path = adapter.render_file(in_wav, out_wav, 48000)
    assert result_path == out_wav
    assert out_wav.exists()

    rendered, sr = load_audio(out_wav)
    assert sr == 48000
    np.testing.assert_allclose(rendered, data * (10.0 ** (-6.0 / 20.0)), rtol=1e-4, atol=1e-5)


# -----------------------------------------------------------------------------
# 2. CliPluginAdapter tests
# -----------------------------------------------------------------------------


def test_cli_adapter_execution(tmp_path):
    in_wav = tmp_path / "in.wav"
    out_wav = tmp_path / "out.wav"
    data = _generate_test_sine(0.02, 48000)
    write_audio(in_wav, data, 48000)

    # Use python executable to mock an external DSP CLI tool copying input to output
    cmd = [
        sys.executable,
        "-c",
        "import sys, shutil; shutil.copy(sys.argv[1], sys.argv[2])",
        "{input}",
        "{output}",
    ]

    adapter = CliPluginAdapter(command_template=cmd, base_dir=tmp_path)
    res = adapter.render_file(in_wav, out_wav, 48000)
    assert res == out_wav
    assert out_wav.exists()

    rendered, sr = load_audio(out_wav)
    np.testing.assert_allclose(rendered, data, atol=1e-5)


def test_cli_adapter_failure_exit_code(tmp_path):
    in_wav = tmp_path / "in.wav"
    out_wav = tmp_path / "out.wav"
    write_audio(in_wav, np.zeros((10, 1)), 48000)

    cmd = [sys.executable, "-c", "import sys; sys.exit(42)"]
    adapter = CliPluginAdapter(command_template=cmd)
    with pytest.raises(PluginError, match="CLI plugin exited with code 42"):
        adapter.render_file(in_wav, out_wav, 48000)


def test_cli_adapter_missing_output(tmp_path):
    in_wav = tmp_path / "in.wav"
    out_wav = tmp_path / "out.wav"
    write_audio(in_wav, np.zeros((10, 1)), 48000)

    # Does not create output file
    cmd = [sys.executable, "-c", "pass"]
    adapter = CliPluginAdapter(command_template=cmd)
    with pytest.raises(PluginError, match="did not produce expected output file"):
        adapter.render_file(in_wav, out_wav, 48000)


def test_cli_adapter_timeout(tmp_path):
    in_wav = tmp_path / "in.wav"
    out_wav = tmp_path / "out.wav"
    write_audio(in_wav, np.zeros((10, 1)), 48000)

    cmd = [sys.executable, "-c", "import time; time.sleep(1.0)"]
    adapter = CliPluginAdapter(command_template=cmd, timeout_seconds=0.1)
    with pytest.raises(PluginError, match="timed out"):
        adapter.render_file(in_wav, out_wav, 48000)


# -----------------------------------------------------------------------------
# 3. get_plugin_adapter factory & VST3 adapter tests
# -----------------------------------------------------------------------------


def test_get_plugin_adapter_factory():
    builtin_adapter = get_plugin_adapter({"type": "builtin", "effect": "gain"})
    assert isinstance(builtin_adapter, BuiltinDspAdapter)

    cli_adapter = get_plugin_adapter({"type": "cli", "command": ["dummy", "{input}", "{output}"]})
    assert isinstance(cli_adapter, CliPluginAdapter)

    with pytest.raises(PluginError, match="Plugin configuration must be a dictionary"):
        get_plugin_adapter("not-a-dict")  # type: ignore

    with pytest.raises(PluginError, match="Unsupported plugin adapter type"):
        get_plugin_adapter({"type": "unknown_type"})

    with pytest.raises(PluginError, match="requires a 'path' attribute"):
        get_plugin_adapter({"type": "vst3"})

    with pytest.raises(PluginError, match="requires a 'command' list"):
        get_plugin_adapter({"type": "cli"})


def test_vst3_adapter_missing_plugin(tmp_path):
    with pytest.raises(PluginError, match="Plugin file not found"):
        VST3PluginAdapter(tmp_path / "nonexistent.vst3")


# -----------------------------------------------------------------------------
# 4. Manifest parsing & run_batch dynamic rendering tests
# -----------------------------------------------------------------------------


def test_manifest_validation_dynamic_render(tmp_path):
    manifest_file = tmp_path / "manifest.json"

    # Input without plugin
    bad_manifest = {
        "name": "no-plugin-suite",
        "tests": [{"id": "t1", "expected": "ref.wav", "input": "stim.wav"}],
    }
    manifest_file.write_text(json.dumps(bad_manifest), encoding="utf-8")
    with pytest.raises(ManifestError, match="no plugin is defined"):
        load_manifest(manifest_file)

    # Neither actual nor input
    bad_manifest2 = {
        "name": "neither-suite",
        "tests": [{"id": "t1", "expected": "ref.wav"}],
    }
    manifest_file.write_text(json.dumps(bad_manifest2), encoding="utf-8")
    with pytest.raises(ManifestError, match="must specify either 'actual' or 'input'"):
        load_manifest(manifest_file)


def test_batch_run_with_dynamic_rendering_on_disk(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    stim_path = audio_dir / "stimulus.wav"
    ref_path = audio_dir / "reference.wav"

    data = _generate_test_sine(0.05, 48000)
    write_audio(stim_path, data, 48000)
    # Expected is attenuated by -6 dB
    attenuated = data * (10.0 ** (-6.0 / 20.0))
    write_audio(ref_path, attenuated, 48000)

    manifest_file = tmp_path / "batch_manifest.json"
    manifest_data = {
        "name": "dynamic-rendering-suite",
        "plugin": {
            "type": "builtin",
            "effect": "gain",
            "parameters": {"gain_db": -6.0},
        },
        "tests": [
            {
                "id": "render-case-1",
                "description": "Gain -6 dB rendered on the fly",
                "expected": "audio/reference.wav",
                "input": "audio/stimulus.wav",
            }
        ],
    }
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    out_reports = tmp_path / "reports"
    result = run_batch(manifest_file, output_dir=out_reports)
    assert result.status == "PASS"
    assert result.passed == 1
    assert result.failed == 0
    assert result.errored == 0

    # Ensure rendered file was saved to out_reports / renders / render-case-1_actual.wav
    renders_dir = out_reports / "renders"
    rendered_file = renders_dir / "render-case-1_actual.wav"
    assert rendered_file.exists()

    # Ensure report HTML and JSON were created
    assert (out_reports / "render-case-1.html").exists()
    assert (out_reports / "render-case-1.json").exists()
    assert (out_reports / "summary.html").exists()
    assert (out_reports / "summary.json").exists()


def test_batch_run_with_dynamic_rendering_in_memory(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    stim_path = audio_dir / "stimulus.wav"
    ref_path = audio_dir / "reference.wav"

    data = _generate_test_sine(0.05, 48000)
    write_audio(stim_path, data, 48000)
    attenuated = data * (10.0 ** (-12.0 / 20.0))
    write_audio(ref_path, attenuated, 48000)

    manifest_file = tmp_path / "manifest_in_memory.json"
    manifest_data = {
        "name": "in-memory-suite",
        "tests": [
            {
                "id": "case-override",
                "expected": "audio/reference.wav",
                "input": "audio/stimulus.wav",
                "plugin": {
                    "type": "builtin",
                    "effect": "gain",
                    "parameters": {"gain_db": -12.0},
                },
            }
        ],
    }
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    # output_dir is None
    result = run_batch(manifest_file, output_dir=None)
    assert result.status == "PASS"
    assert result.passed == 1
    assert result.results[0].actual_path == "<rendered-in-memory:case-override>"


# -----------------------------------------------------------------------------
# 5. CLI render command tests
# -----------------------------------------------------------------------------


def test_cli_render_builtin_success(tmp_path, capsys):
    in_wav = tmp_path / "stim.wav"
    out_wav = tmp_path / "rendered.wav"
    data = _generate_test_sine(0.05, 48000)
    write_audio(in_wav, data, 48000)

    exit_code = main(
        [
            "render",
            "--plugin",
            "gain",
            "--input",
            str(in_wav),
            "--output",
            str(out_wav),
            "--params",
            '{"gain_db": -3.0}',
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Rendered:" in captured.out
    assert out_wav.exists()

    rendered, sr = load_audio(out_wav)
    assert sr == 48000
    np.testing.assert_allclose(rendered, data * (10.0 ** (-3.0 / 20.0)), rtol=1e-4, atol=1e-5)


def test_cli_render_params_file(tmp_path, capsys):
    in_wav = tmp_path / "stim.wav"
    out_wav = tmp_path / "rendered.wav"
    params_file = tmp_path / "params.json"
    params_file.write_text(json.dumps({"gain_db": -6.0}), encoding="utf-8")

    data = _generate_test_sine(0.05, 48000)
    write_audio(in_wav, data, 48000)

    exit_code = main(
        [
            "render",
            "--plugin",
            "gain",
            "--input",
            str(in_wav),
            "--output",
            str(out_wav),
            "--params",
            str(params_file),
        ]
    )
    assert exit_code == 0
    assert out_wav.exists()


def test_cli_render_errors(tmp_path, capsys):
    in_wav = tmp_path / "stim.wav"
    out_wav = tmp_path / "rendered.wav"
    write_audio(in_wav, np.zeros((10, 1)), 48000)

    # Missing input file
    exit_code = main(["render", "--plugin", "gain", "--input", str(tmp_path / "nonexistent.wav"), "--output", str(out_wav)])
    assert exit_code == 2

    # Invalid JSON in params
    exit_code = main(["render", "--plugin", "gain", "--input", str(in_wav), "--output", str(out_wav), "--params", "{bad-json}"])
    assert exit_code == 2
