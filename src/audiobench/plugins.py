"""Headless audio plugin hosting and offline DSP rendering adapters."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from scipy.signal import butter, sosfilt

from .audio import load_audio, write_audio

try:
    import pedalboard
    from pedalboard import load_plugin as _pedalboard_load_plugin

    HAS_PEDALBOARD = True
except ImportError:
    HAS_PEDALBOARD = False
    _pedalboard_load_plugin = None


class PluginError(RuntimeError):
    """Raised when plugin loading, configuration, or offline rendering fails."""


class PluginAdapter(Protocol):
    """Interface for offline audio plugin hosts and DSP rendering chains."""

    def render_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        sample_rate: int,
        parameters: dict[str, Any] | None = None,
    ) -> Path:
        """Render an audio file through the plugin and save the result."""
        ...

    def render_array(
        self,
        audio: np.ndarray,
        sample_rate: int,
        parameters: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Render a (frames, channels) float64 array through the plugin."""
        ...

    def reset(self) -> None:
        """Reset internal plugin state (delay buffers, tails, IIR states)."""
        ...


class VST3PluginAdapter:
    """Headless VST3/AU plugin host adapter powered by Spotify's Pedalboard."""

    def __init__(self, plugin_path: str | Path) -> None:
        if not HAS_PEDALBOARD or _pedalboard_load_plugin is None:
            raise PluginError(
                "VST3/AU plugin hosting requires 'pedalboard'. "
                "Install it with: pip install 'audiobench[plugins]'"
            )

        resolved_path = Path(plugin_path).resolve()
        if not resolved_path.exists():
            raise PluginError(f"Plugin file not found: {resolved_path}")

        try:
            self._plugin = _pedalboard_load_plugin(str(resolved_path))
        except Exception as exc:
            raise PluginError(f"Failed to load plugin '{resolved_path}': {exc}") from exc

        self.plugin_path = resolved_path

    def reset(self) -> None:
        if hasattr(self._plugin, "reset"):
            self._plugin.reset()

    def _apply_parameters(self, parameters: dict[str, Any] | None) -> None:
        if not parameters:
            return
        for param_name, val in parameters.items():
            if hasattr(self._plugin, param_name):
                try:
                    setattr(self._plugin, param_name, val)
                except Exception as exc:
                    raise PluginError(
                        f"Failed to set parameter '{param_name}' to {val!r}: {exc}"
                    ) from exc
            else:
                raise PluginError(
                    f"Plugin '{self.plugin_path.name}' has no parameter '{param_name}'"
                )

    def render_array(
        self,
        audio: np.ndarray,
        sample_rate: int,
        parameters: dict[str, Any] | None = None,
    ) -> np.ndarray:
        arr = np.asarray(audio, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim != 2:
            raise ValueError("Audio array must have shape (frames, channels)")

        self.reset()
        self._apply_parameters(parameters)

        # Pedalboard expects (channels, frames) in float32
        pb_input = arr.T.astype(np.float32, copy=False)
        try:
            pb_output = self._plugin(pb_input, sample_rate, reset=False)
        except Exception as exc:
            raise PluginError(f"Plugin processing failed: {exc}") from exc

        # Convert back to (frames, channels) float64
        output = np.asarray(pb_output.T, dtype=np.float64)
        return output

    def render_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        sample_rate: int,
        parameters: dict[str, Any] | None = None,
    ) -> Path:
        data, sr = load_audio(input_path)
        if sr != sample_rate:
            raise ValueError(
                f"File sample rate ({sr} Hz) does not match requested rendering sample rate ({sample_rate} Hz)"
            )
        rendered = self.render_array(data, sample_rate, parameters)
        target = Path(output_path).resolve()
        write_audio(target, rendered, sample_rate)
        return target


class BuiltinDspAdapter:
    """Pure Python/SciPy DSP chain adapter for deterministic offline processing."""

    def __init__(
        self,
        effect_type: str = "gain",
        default_parameters: dict[str, Any] | None = None,
    ) -> None:
        self.effect_type = effect_type.lower()
        self.default_parameters = default_parameters or {}

    def reset(self) -> None:
        pass  # Stateless or state reset on each call

    def render_array(
        self,
        audio: np.ndarray,
        sample_rate: int,
        parameters: dict[str, Any] | None = None,
    ) -> np.ndarray:
        arr = np.asarray(audio, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[:, None]
        params = {**self.default_parameters, **(parameters or {})}

        if self.effect_type == "gain":
            gain_db = params.get("gain_db")
            if gain_db is not None:
                multiplier = 10.0 ** (float(gain_db) / 20.0)
            else:
                multiplier = float(params.get("multiplier", 1.0))
            return arr * multiplier

        if self.effect_type in ("lowpass", "highpass"):
            cutoff = float(params.get("cutoff_hz", 1000.0))
            order = int(params.get("order", 2))
            nyquist = sample_rate / 2.0
            if cutoff >= nyquist:
                cutoff = nyquist - 1.0
            sos = butter(order, cutoff, btype=self.effect_type, fs=sample_rate, output="sos")
            return sosfilt(sos, arr, axis=0)

        if self.effect_type == "delay":
            delay_ms = params.get("delay_ms")
            if delay_ms is not None:
                delay_samples = int(round(float(delay_ms) * sample_rate / 1000.0))
            else:
                delay_samples = int(params.get("delay_samples", 0))

            if delay_samples <= 0:
                return arr.copy()
            out = np.zeros_like(arr)
            if delay_samples < len(arr):
                out[delay_samples:] = arr[:-delay_samples]
            return out

        if self.effect_type == "soft_clip":
            drive_db = float(params.get("drive_db", 0.0))
            drive = 10.0 ** (drive_db / 20.0)
            # Hyperbolic tangent soft saturation
            return np.tanh(arr * drive)

        raise PluginError(f"Unknown built-in DSP effect type: '{self.effect_type}'")

    def render_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        sample_rate: int,
        parameters: dict[str, Any] | None = None,
    ) -> Path:
        data, sr = load_audio(input_path)
        if sr != sample_rate:
            raise ValueError(
                f"File sample rate ({sr} Hz) does not match requested rendering sample rate ({sample_rate} Hz)"
            )
        rendered = self.render_array(data, sample_rate, parameters)
        target = Path(output_path).resolve()
        write_audio(target, rendered, sample_rate)
        return target


class CliPluginAdapter:
    """External CLI executable host adapter with safe parameter interpolation."""

    def __init__(
        self,
        command_template: list[str],
        base_dir: Path | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        if not isinstance(command_template, list) or not command_template:
            raise PluginError("CliPluginAdapter command_template must be a non-empty list of strings")
        self.command_template = command_template
        self.base_dir = base_dir or Path.cwd()
        self.timeout_seconds = timeout_seconds

    def reset(self) -> None:
        pass

    def render_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        sample_rate: int,
        parameters: dict[str, Any] | None = None,
    ) -> Path:
        resolved_input = Path(input_path).resolve()
        _, sr = load_audio(resolved_input)
        if sr != sample_rate:
            raise ValueError(
                f"File sample rate ({sr} Hz) does not match requested rendering sample rate ({sample_rate} Hz)"
            )
        resolved_output = Path(output_path).resolve()
        resolved_output.parent.mkdir(parents=True, exist_ok=True)

        params = parameters or {}
        cmd: list[str] = []
        for token in self.command_template:
            formatted = token.replace("{input}", str(resolved_input)).replace("{output}", str(resolved_output))
            formatted = formatted.replace("{sample_rate}", str(sample_rate))
            for p_k, p_v in params.items():
                formatted = formatted.replace(f"{{{p_k}}}", str(p_v))
            cmd.append(formatted)

        try:
            completed = subprocess.run(
                cmd,
                cwd=self.base_dir,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise PluginError(f"CLI plugin timed out after {self.timeout_seconds}s: {' '.join(cmd)}") from exc
        except OSError as exc:
            raise PluginError(f"Cannot execute CLI plugin command {' '.join(cmd)}: {exc}") from exc

        if completed.returncode != 0:
            err = completed.stderr.strip() or completed.stdout.strip()
            raise PluginError(f"CLI plugin exited with code {completed.returncode}: {err}")

        if not resolved_output.exists():
            raise PluginError(f"CLI plugin succeeded but did not produce expected output file: {resolved_output}")

        return resolved_output

    def render_array(
        self,
        audio: np.ndarray,
        sample_rate: int,
        parameters: dict[str, Any] | None = None,
    ) -> np.ndarray:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_in = Path(tmp_dir) / "temp_input.wav"
            tmp_out = Path(tmp_dir) / "temp_output.wav"
            write_audio(tmp_in, audio, sample_rate)
            self.render_file(tmp_in, tmp_out, sample_rate, parameters)
            out_data, _ = load_audio(tmp_out)
            return out_data


def get_plugin_adapter(
    config: dict[str, Any],
    base_dir: Path | None = None,
) -> PluginAdapter:
    """Factory creating an appropriate PluginAdapter from configuration."""
    if not isinstance(config, dict):
        raise PluginError("Plugin configuration must be a dictionary")

    adapter_type = config.get("type", "").lower()
    base = base_dir or Path.cwd()

    if adapter_type in ("vst3", "au", "vst"):
        raw_path = config.get("path")
        if not raw_path:
            raise PluginError(f"Plugin type '{adapter_type}' requires a 'path' attribute")
        plugin_path = Path(raw_path)
        if not plugin_path.is_absolute():
            plugin_path = (base / plugin_path).resolve()
        return VST3PluginAdapter(plugin_path)

    if adapter_type == "builtin":
        effect = config.get("effect", "gain")
        params = config.get("parameters", {})
        return BuiltinDspAdapter(effect_type=effect, default_parameters=params)

    if adapter_type == "cli":
        command = config.get("command")
        if not command or not isinstance(command, list):
            raise PluginError("Plugin type 'cli' requires a 'command' list of strings")
        timeout = float(config.get("timeout_seconds", 60.0))
        return CliPluginAdapter(command_template=command, base_dir=base, timeout_seconds=timeout)

    raise PluginError(
        f"Unsupported plugin adapter type '{adapter_type}'. Allowed types: 'vst3', 'au', 'builtin', 'cli'"
    )
