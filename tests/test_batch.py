import json
from pathlib import Path

import numpy as np
import pytest

from audiobench.audio import write_audio
from audiobench.batch import (
    BatchManifest,
    BatchRunResult,
    ManifestError,
    TestCaseConfig,
    load_manifest,
    merge_thresholds,
    resolve_path,
    run_batch,
)
from audiobench.metrics import Thresholds


def test_merge_thresholds():
    base = Thresholds(min_snr_db=60.0, max_delay_ms=0.5, max_duration_delta_ms=1.0, max_error_dbfs=-80.0)
    overrides = {"min_snr_db": 80.0, "max_delay_ms": 0.1}
    merged = merge_thresholds(base, overrides)
    assert merged.min_snr_db == 80.0
    assert merged.max_delay_ms == 0.1
    assert merged.max_duration_delta_ms == 1.0
    assert merged.max_error_dbfs == -80.0

    assert merge_thresholds(base, None) == base


def test_resolve_path(tmp_path):
    base_dir = tmp_path / "suite"
    base_dir.mkdir()

    rel = resolve_path(base_dir, "audio/test.wav")
    assert rel == (base_dir / "audio/test.wav").resolve()

    abs_target = (tmp_path / "other/test.wav").resolve()
    assert resolve_path(base_dir, str(abs_target)) == abs_target


def test_load_manifest_valid(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_data = {
        "name": "test-suite",
        "thresholds": {"min_snr_db": 70.0},
        "tests": [
            {
                "id": "case-1",
                "description": "First test",
                "expected": "ref.wav",
                "actual": "out.wav",
                "thresholds": {"max_delay_ms": 0.2},
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

    manifest = load_manifest(manifest_path)
    assert isinstance(manifest, BatchManifest)
    assert manifest.name == "test-suite"
    assert manifest.thresholds.min_snr_db == 70.0
    assert len(manifest.tests) == 1
    assert manifest.tests[0].id == "case-1"
    assert manifest.tests[0].expected == "ref.wav"
    assert manifest.tests[0].actual == "out.wav"
    assert manifest.tests[0].thresholds == {"max_delay_ms": 0.2}


def test_load_manifest_missing_and_invalid_json(tmp_path):
    with pytest.raises(ManifestError, match="not found"):
        load_manifest(tmp_path / "nonexistent.json")

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not-valid-json}", encoding="utf-8")
    with pytest.raises(ManifestError, match="Invalid JSON"):
        load_manifest(bad_json)


@pytest.mark.parametrize(
    "bad_content,match",
    [
        ([], "Manifest root must be a JSON object"),
        ({"tests": []}, "missing valid non-empty 'name'"),
        ({"name": "suite", "tests": []}, "non-empty 'tests' list"),
        ({"name": "suite", "tests": "not a list"}, "non-empty 'tests' list"),
        ({"name": "suite", "thresholds": "bad"}, "must be an object/dict"),
        ({"name": "suite", "thresholds": {"invalid_key": 10}}, "unknown threshold key"),
        ({"name": "suite", "thresholds": {"min_snr_db": "not-a-number"}}, "must be a number"),
        ({"name": "suite", "thresholds": {"min_snr_db": float("nan")}}, "must be finite"),
        ({"name": "suite", "thresholds": {"min_snr_db": float("inf")}}, "must be finite"),
        ({"name": "suite", "tests": [{}]}, "missing valid non-empty 'id'"),
        ({"name": "suite", "tests": [{"id": "bad/slash", "expected": "a.wav", "actual": "b.wav"}]}, "invalid test id"),
        ({"name": "suite", "tests": [{"id": "bad space", "expected": "a.wav", "actual": "b.wav"}]}, "invalid test id"),
        ({"name": "suite", "tests": [{"id": "t1", "expected": ""}]}, "missing valid 'expected'"),
        ({"name": "suite", "tests": [{"id": "t1", "expected": "a.wav", "actual": ""}]}, "missing valid 'actual'"),
        (
            {
                "name": "suite",
                "tests": [
                    {"id": "duplicate", "expected": "a.wav", "actual": "b.wav"},
                    {"id": "duplicate", "expected": "c.wav", "actual": "d.wav"},
                ],
            },
            "Duplicate test id detected",
        ),
    ],
)
def test_load_manifest_schema_validation(tmp_path, bad_content, match):
    manifest_path = tmp_path / "bad_schema.json"
    manifest_path.write_text(json.dumps(bad_content), encoding="utf-8")
    with pytest.raises(ManifestError, match=match):
        load_manifest(manifest_path)


def test_run_batch_passes_and_generates_reports(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    ref_path = audio_dir / "ref.wav"
    render_path = audio_dir / "render.wav"

    data = np.sin(np.linspace(0, 100, 4800, dtype=np.float64))[:, None] * 0.5
    write_audio(ref_path, data, 48_000)
    write_audio(render_path, data.copy(), 48_000)

    manifest_path = tmp_path / "suite.json"
    manifest_data = {
        "name": "pass-suite",
        "tests": [
            {
                "id": "identical-sine",
                "description": "Exact match sine",
                "expected": "audio/ref.wav",
                "actual": "audio/render.wav",
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

    out_dir = tmp_path / "reports"
    result = run_batch(manifest_path, output_dir=out_dir)

    assert isinstance(result, BatchRunResult)
    assert result.status == "PASS"
    assert result.total_tests == 1
    assert result.passed == 1
    assert result.failed == 0
    assert result.errored == 0

    assert (out_dir / "summary.json").exists()
    assert (out_dir / "summary.html").exists()
    assert (out_dir / "identical-sine.html").exists()
    assert (out_dir / "identical-sine.json").exists()

    summary_json = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary_json["status"] == "PASS"
    assert summary_json["summary"]["passed"] == 1
    assert summary_json["summary_html"] is not None
    assert Path(summary_json["summary_html"]).exists()
    assert summary_json["summary_json"] is not None
    assert Path(summary_json["summary_json"]).exists()


def test_run_batch_threshold_override(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    ref_path = audio_dir / "ref.wav"
    render_path = audio_dir / "render.wav"

    data = np.sin(np.linspace(0, 100, 4800, dtype=np.float64))[:, None] * 0.5
    write_audio(ref_path, data, 48_000)
    write_audio(render_path, data * 0.9, 48_000)  # gain error

    manifest_path = tmp_path / "suite_override.json"
    manifest_data = {
        "name": "override-suite",
        "thresholds": {"min_snr_db": 60.0},
        "tests": [
            {
                "id": "strict-fails",
                "expected": "audio/ref.wav",
                "actual": "audio/render.wav",
                # inherits min_snr_db = 60.0 -> will FAIL
            },
            {
                "id": "lenient-passes",
                "expected": "audio/ref.wav",
                "actual": "audio/render.wav",
                "thresholds": {"min_snr_db": 10.0},  # overrides to 10.0 dB -> will PASS
            },
        ],
    }
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

    result = run_batch(manifest_path)
    assert result.total_tests == 2
    assert result.passed == 1
    assert result.failed == 1
    assert result.status == "FAIL"


def test_run_batch_fault_isolation(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    ref_path = audio_dir / "ref.wav"
    render_path = audio_dir / "render.wav"

    data = np.zeros((4800, 1), dtype=np.float64)
    write_audio(ref_path, data, 48_000)
    write_audio(render_path, data, 48_000)

    manifest_path = tmp_path / "suite_isolation.json"
    manifest_data = {
        "name": "fault-isolation-suite",
        "tests": [
            {
                "id": "missing-file-test",
                "expected": "audio/nonexistent.wav",
                "actual": "audio/render.wav",
            },
            {
                "id": "valid-pass-test",
                "expected": "audio/ref.wav",
                "actual": "audio/render.wav",
            },
        ],
    }
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

    out_dir = tmp_path / "reports_iso"
    result = run_batch(manifest_path, output_dir=out_dir)

    assert result.total_tests == 2
    assert result.errored == 1
    assert result.passed == 1
    assert result.status == "FAIL"

    # Verify first test has error recorded and did not prevent second test
    assert result.results[0].status == "ERROR"
    assert "not found" in result.results[0].error_message.lower()
    assert result.results[1].status == "PASS"
    assert (out_dir / "valid-pass-test.html").exists()
