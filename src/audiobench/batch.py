"""Batch execution of audio render regression test suites from JSON manifests."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import __version__
from .audio import load_audio
from .metrics import Thresholds, compare_arrays, evaluate_thresholds
from .report import write_batch_html_summary, write_html_report

VALID_THRESHOLD_KEYS = frozenset(
    {"min_snr_db", "max_delay_ms", "max_duration_delta_ms", "max_error_dbfs"}
)
VALID_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]+$")


class ManifestError(ValueError):
    """Raised when a manifest file is missing, unreadable, or violates schema."""


@dataclass(frozen=True)
class TestCaseConfig:
    """Configuration for a single regression comparison test case."""

    __test__ = False

    id: str
    expected: str
    actual: str
    description: str | None = None
    thresholds: dict[str, float] | None = None


@dataclass(frozen=True)
class BatchManifest:
    """Parsed and validated manifest suite configuration."""

    name: str
    thresholds: Thresholds
    tests: list[TestCaseConfig]


@dataclass
class TestCaseResult:
    """Outcome and artifacts of executing a single test case."""

    __test__ = False

    id: str
    expected_path: str
    actual_path: str
    status: str  # "PASS", "WARN", "FAIL", "ERROR"
    description: str | None = None
    error_message: str | None = None
    metrics: dict[str, Any] | None = None
    evaluation: dict[str, Any] | None = None
    thresholds: dict[str, float] | None = None
    html_report: str | None = None
    json_report: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BatchRunResult:
    """Aggregated outcome of executing an entire test suite."""

    name: str
    manifest_path: str
    total_tests: int
    passed: int
    warned: int
    failed: int
    errored: int
    status: str  # "PASS", "WARN", "FAIL"
    results: list[TestCaseResult]
    output_dir: str | None = None
    summary_html: str | None = None
    summary_json: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": "AudioBench",
            "version": __version__,
            "name": self.name,
            "manifest_path": self.manifest_path,
            "status": self.status,
            "summary": {
                "total": self.total_tests,
                "passed": self.passed,
                "warned": self.warned,
                "failed": self.failed,
                "errored": self.errored,
            },
            "output_dir": self.output_dir,
            "summary_html": self.summary_html,
            "summary_json": self.summary_json,
            "tests": [r.to_dict() for r in self.results],
        }


def _validate_threshold_dict(raw: Any, context: str) -> dict[str, float]:
    if not isinstance(raw, dict):
        raise ManifestError(f"{context}: 'thresholds' must be an object/dict, got {type(raw).__name__}")
    parsed: dict[str, float] = {}
    for key, val in raw.items():
        if key not in VALID_THRESHOLD_KEYS:
            raise ManifestError(
                f"{context}: unknown threshold key '{key}'. Allowed keys: {sorted(VALID_THRESHOLD_KEYS)}"
            )
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise ManifestError(f"{context}: threshold '{key}' must be a number, got {val!r}")
        val_float = float(val)
        if not math.isfinite(val_float):
            raise ManifestError(f"{context}: threshold '{key}' must be finite, got {val!r}")
        parsed[key] = val_float
    return parsed


def merge_thresholds(base: Thresholds, overrides: dict[str, float] | None) -> Thresholds:
    """Return a new Thresholds instance applying any per-test overrides."""
    if not overrides:
        return base
    return Thresholds(
        min_snr_db=overrides.get("min_snr_db", base.min_snr_db),
        max_delay_ms=overrides.get("max_delay_ms", base.max_delay_ms),
        max_duration_delta_ms=overrides.get("max_duration_delta_ms", base.max_duration_delta_ms),
        max_error_dbfs=overrides.get("max_error_dbfs", base.max_error_dbfs),
    )


def load_manifest(path: str | Path) -> BatchManifest:
    """Read and validate a batch manifest JSON file."""
    manifest_file = Path(path)
    if not manifest_file.exists():
        raise ManifestError(f"Manifest file not found: {manifest_file}")
    if not manifest_file.is_file():
        raise ManifestError(f"Manifest path is not a file: {manifest_file}")

    try:
        content = manifest_file.read_text(encoding="utf-8")
        raw = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"Invalid JSON in manifest '{manifest_file}': {exc}") from exc
    except OSError as exc:
        raise ManifestError(f"Cannot read manifest file '{manifest_file}': {exc}") from exc

    if not isinstance(raw, dict):
        raise ManifestError(f"Manifest root must be a JSON object, got {type(raw).__name__}")

    suite_name = raw.get("name")
    if not isinstance(suite_name, str) or not suite_name.strip():
        raise ManifestError("Manifest missing valid non-empty 'name' string")

    suite_thresholds = Thresholds()
    if "thresholds" in raw:
        parsed_suite_thresh = _validate_threshold_dict(raw["thresholds"], "Suite level")
        suite_thresholds = merge_thresholds(suite_thresholds, parsed_suite_thresh)

    raw_tests = raw.get("tests")
    if not isinstance(raw_tests, list) or len(raw_tests) == 0:
        raise ManifestError("Manifest must contain a non-empty 'tests' list")

    seen_ids: set[str] = set()
    test_configs: list[TestCaseConfig] = []

    for index, item in enumerate(raw_tests):
        ctx = f"Test at index {index}"
        if not isinstance(item, dict):
            raise ManifestError(f"{ctx} must be an object, got {type(item).__name__}")

        test_id = item.get("id")
        if not isinstance(test_id, str) or not test_id.strip():
            raise ManifestError(f"{ctx} missing valid non-empty 'id' string")
        if not VALID_ID_PATTERN.match(test_id):
            raise ManifestError(
                f"{ctx}: invalid test id '{test_id}'. IDs must only contain letters, numbers, hyphens, and underscores."
            )
        if test_id in seen_ids:
            raise ManifestError(f"Duplicate test id detected: '{test_id}'")
        seen_ids.add(test_id)

        expected = item.get("expected")
        if not isinstance(expected, str) or not expected.strip():
            raise ManifestError(f"{ctx} ('{test_id}') missing valid 'expected' audio path")

        actual = item.get("actual")
        if not isinstance(actual, str) or not actual.strip():
            raise ManifestError(f"{ctx} ('{test_id}') missing valid 'actual' audio path")

        desc = item.get("description")
        if desc is not None and not isinstance(desc, str):
            raise ManifestError(f"{ctx} ('{test_id}') 'description' must be a string if provided")

        item_thresh: dict[str, float] | None = None
        if "thresholds" in item:
            item_thresh = _validate_threshold_dict(item["thresholds"], f"Test '{test_id}'")

        test_configs.append(
            TestCaseConfig(
                id=test_id,
                expected=expected,
                actual=actual,
                description=desc,
                thresholds=item_thresh,
            )
        )

    return BatchManifest(name=suite_name, thresholds=suite_thresholds, tests=test_configs)


def resolve_path(base_dir: Path, target_path: str | Path) -> Path:
    """Resolve a target path relative to base_dir if not absolute."""
    path = Path(target_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def run_batch(
    manifest_path: str | Path,
    output_dir: str | Path | None = None,
) -> BatchRunResult:
    """Execute a batch manifest suite with fault isolation between test cases."""
    manifest_file = Path(manifest_path).resolve()
    manifest = load_manifest(manifest_file)
    base_dir = manifest_file.parent

    out_dir_path: Path | None = None
    if output_dir is not None:
        out_dir_path = Path(output_dir).resolve()
        out_dir_path.mkdir(parents=True, exist_ok=True)

    test_results: list[TestCaseResult] = []

    for test in manifest.tests:
        resolved_expected = resolve_path(base_dir, test.expected)
        resolved_actual = resolve_path(base_dir, test.actual)
        limits = merge_thresholds(manifest.thresholds, test.thresholds)

        # Fault isolation: an error in one test case must not prevent running the others.
        try:
            expected_data, expected_sr = load_audio(resolved_expected)
            actual_data, actual_sr = load_audio(resolved_actual)
            if expected_sr != actual_sr:
                raise ValueError(
                    f"Sample rate mismatch: expected file is {expected_sr} Hz, actual file is {actual_sr} Hz. "
                    "AudioBench does not silently resample inputs."
                )

            metrics = compare_arrays(expected_data, actual_data, expected_sr)
            evaluation = evaluate_thresholds(metrics, limits)
            case_status = evaluation["status"]

            html_report_path: Path | None = None
            json_report_path: Path | None = None

            if out_dir_path is not None:
                json_report_path = out_dir_path / f"{test.id}.json"
                json_report_path.write_text(
                    json.dumps(
                        {
                            "tool": "AudioBench",
                            "version": __version__,
                            "test_id": test.id,
                            "description": test.description,
                            "expected_path": str(resolved_expected),
                            "actual_path": str(resolved_actual),
                            "thresholds": limits.__dict__,
                            "metrics": metrics,
                            "evaluation": evaluation,
                        },
                        indent=2,
                        allow_nan=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                html_report_path = write_html_report(
                    out_dir_path / f"{test.id}.html",
                    expected_path=resolved_expected,
                    actual_path=resolved_actual,
                    metrics=metrics,
                    evaluation=evaluation,
                    expected=expected_data,
                    actual=actual_data,
                )

            test_results.append(
                TestCaseResult(
                    id=test.id,
                    expected_path=str(resolved_expected),
                    actual_path=str(resolved_actual),
                    status=case_status,
                    description=test.description,
                    error_message=None,
                    metrics=metrics,
                    evaluation=evaluation,
                    thresholds=asdict(limits),
                    html_report=str(html_report_path) if html_report_path else None,
                    json_report=str(json_report_path) if json_report_path else None,
                )
            )
        except Exception as exc:  # Catch FileNotFoundError, ValueError, OSError, RuntimeError
            test_results.append(
                TestCaseResult(
                    id=test.id,
                    expected_path=str(resolved_expected),
                    actual_path=str(resolved_actual),
                    status="ERROR",
                    description=test.description,
                    error_message=str(exc),
                    metrics=None,
                    evaluation=None,
                    thresholds=asdict(limits),
                    html_report=None,
                    json_report=None,
                )
            )

    passed_count = sum(1 for r in test_results if r.status == "PASS")
    warned_count = sum(1 for r in test_results if r.status == "WARN")
    failed_count = sum(1 for r in test_results if r.status == "FAIL")
    errored_count = sum(1 for r in test_results if r.status == "ERROR")

    if failed_count > 0 or errored_count > 0:
        overall_status = "FAIL"
    elif warned_count > 0:
        overall_status = "WARN"
    else:
        overall_status = "PASS"

    summary_html_path: str | None = None
    summary_json_path: str | None = None

    batch_run_result = BatchRunResult(
        name=manifest.name,
        manifest_path=str(manifest_file),
        total_tests=len(test_results),
        passed=passed_count,
        warned=warned_count,
        failed=failed_count,
        errored=errored_count,
        status=overall_status,
        results=test_results,
        output_dir=str(out_dir_path) if out_dir_path else None,
    )

    if out_dir_path is not None:
        json_file = out_dir_path / "summary.json"
        html_file = out_dir_path / "summary.html"
        summary_json_path = str(json_file)
        summary_html_path = str(html_file)
        batch_run_result.summary_json = summary_json_path
        batch_run_result.summary_html = summary_html_path

        json_file.write_text(
            json.dumps(batch_run_result.to_dict(), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        write_batch_html_summary(html_file, batch_run_result)

    return batch_run_result
