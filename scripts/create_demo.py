"""Regenerate portable AudioBench PASS/FAIL demonstrations.

Run from the repository root after installing the project with:
    python -m pip install -e ".[dev]"
    python scripts/create_demo.py

This script deliberately uses the public CLI to create its reports, so it also
acts as a small end-to-end smoke test of the installed command implementation.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from audiobench.audio import write_audio

ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "examples" / "demo"
SAMPLE_RATE = 48_000
DURATION_S = 0.5
DELAY_SAMPLES = 120  # 120 / 48,000 = 2.5 ms


def run_compare(expected: Path, actual: Path, stem: str, expected_exit_code: int) -> None:
    report = DEMO_DIR / f"{stem}.html"
    result_json = DEMO_DIR / f"{stem}.json"
    command = [
        sys.executable,
        "-m",
        "audiobench",
        "compare",
        "--expected",
        str(expected.relative_to(ROOT)),
        "--actual",
        str(actual.relative_to(ROOT)),
        "--report",
        str(report),
        "--json",
        str(result_json),
    ]
    completed = subprocess.run(command, cwd=ROOT, check=False, text=True)
    if completed.returncode != expected_exit_code:
        raise RuntimeError(
            f"Unexpected exit code for {stem}: got {completed.returncode}, "
            f"expected {expected_exit_code}. Command: {' '.join(command)}"
        )


def run_batch(manifest: Path, output_dir: Path, expected_exit_code: int) -> None:
    command = [
        sys.executable,
        "-m",
        "audiobench",
        "batch",
        "--manifest",
        str(manifest.relative_to(ROOT)),
        "--output-dir",
        str(output_dir),
    ]
    completed = subprocess.run(command, cwd=ROOT, check=False, text=True)
    if completed.returncode != expected_exit_code:
        raise RuntimeError(
            f"Unexpected exit code for batch: got {completed.returncode}, "
            f"expected {expected_exit_code}. Command: {' '.join(command)}"
        )


def main() -> int:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    frames = int(round(SAMPLE_RATE * DURATION_S))
    rng = np.random.default_rng(20261027)
    reference = rng.standard_normal(frames).astype(np.float64)
    reference -= reference.mean()
    peak = float(np.max(np.abs(reference)))
    if peak > 0:
        reference *= 0.35 / peak

    reference_path = DEMO_DIR / "reference_white_noise.wav"
    identical_path = DEMO_DIR / "render_identical.wav"
    delayed_path = DEMO_DIR / "render_delayed_2_5ms.wav"
    write_audio(reference_path, reference[:, None], SAMPLE_RATE)
    # Copying the reference WAV ensures a sample-identical control condition.
    shutil.copyfile(reference_path, identical_path)

    delayed = np.zeros_like(reference)
    delayed[DELAY_SAMPLES:] = reference[:-DELAY_SAMPLES]
    write_audio(delayed_path, delayed[:, None], SAMPLE_RATE)

    run_compare(reference_path, identical_path, "pass_report", expected_exit_code=0)
    run_compare(reference_path, delayed_path, "delay_fail_report", expected_exit_code=1)

    # Generate and execute demo batch manifest
    manifest_path = DEMO_DIR / "demo_manifest.json"
    manifest_data = {
        "name": "demo-regression-suite",
        "thresholds": {
            "min_snr_db": 60.0,
            "max_delay_ms": 0.5,
            "max_duration_delta_ms": 1.0,
            "max_error_dbfs": -80.0,
        },
        "tests": [
            {
                "id": "identical-render",
                "description": "Deterministic white noise identical render (PASS)",
                "expected": "reference_white_noise.wav",
                "actual": "render_identical.wav",
            },
            {
                "id": "delayed-render",
                "description": "Artificial 2.5 ms delay exceeding 0.5 ms tolerance (FAIL)",
                "expected": "reference_white_noise.wav",
                "actual": "render_delayed_2_5ms.wav",
            },
        ],
    }
    manifest_path.write_text(json.dumps(manifest_data, indent=2) + "\n", encoding="utf-8")
    batch_output_dir = DEMO_DIR / "batch"
    run_batch(manifest_path, batch_output_dir, expected_exit_code=1)

    print("Demo files, batch suite, and reports regenerated in:")
    print(f"  {DEMO_DIR}")
    print("Expected outcomes: pass_report=PASS, delay_fail_report=FAIL, batch=FAIL (1 pass, 1 fail).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
