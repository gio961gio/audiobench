"""Command-line interface for AudioBench."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .audio import load_audio, write_test_signals
from .batch import ManifestError, run_batch
from .metrics import Thresholds, compare_arrays, evaluate_thresholds
from .report import write_html_report


def _add_generate_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("generate", help="Generate deterministic audio test signals")
    parser.add_argument("--output-dir", default="test_signals", help="Directory for generated WAV files")
    parser.add_argument("--sample-rate", type=int, default=48_000, help="Sample rate in Hz (default: 48000)")
    parser.add_argument("--duration", type=float, default=1.0, help="Signal duration in seconds (default: 1.0)")
    parser.add_argument("--seed", type=int, default=7, help="Seed for reproducible white noise")


def _add_compare_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("compare", help="Compare expected and actual offline audio renders")
    parser.add_argument("--expected", required=True, help="Path to reference/expected WAV file")
    parser.add_argument("--actual", required=True, help="Path to render under test")
    parser.add_argument("--report", default="reports/latest.html", help="HTML report path (default: reports/latest.html)")
    parser.add_argument("--json", dest="json_path", default="reports/latest.json", help="JSON output path (default: reports/latest.json)")
    parser.add_argument("--min-snr-db", type=float, default=60.0, help="Minimum reference-to-error SNR for PASS (default: 60 dB)")
    parser.add_argument("--max-delay-ms", type=float, default=0.5, help="Maximum absolute estimated delay for PASS (default: 0.5 ms)")
    parser.add_argument("--max-duration-delta-ms", type=float, default=1.0, help="Maximum absolute duration delta for PASS (default: 1 ms)")
    parser.add_argument("--max-error-dbfs", type=float, default=-80.0, help="Maximum RMS error for a silent reference (default: -80 dBFS)")
    parser.add_argument("--max-lag-ms", type=float, default=100.0, help="Search range for delay estimation (default: 100 ms)")
    parser.add_argument("--fail-on-warn", action="store_true", help="Return exit code 1 for WARN as well as FAIL")


def _add_batch_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("batch", help="Run a suite of regression comparisons from a JSON manifest")
    parser.add_argument("--manifest", required=True, help="Path to JSON test manifest file")
    parser.add_argument("--output-dir", default="reports/batch", help="Directory for generated reports (default: reports/batch)")
    parser.add_argument("--fail-on-warn", action="store_true", help="Return exit code 1 for WARN as well as FAIL")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audiobench",
        description="Offline audio render regression checks for DSP and plugin developers.",
    )
    parser.add_argument("--version", action="version", version=f"AudioBench {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_generate_parser(subparsers)
    _add_compare_parser(subparsers)
    _add_batch_parser(subparsers)
    return parser


def _run_generate(args: argparse.Namespace) -> int:
    if args.sample_rate <= 0 or args.duration <= 0:
        print("error: --sample-rate and --duration must be positive", file=sys.stderr)
        return 2
    try:
        paths = write_test_signals(args.output_dir, args.sample_rate, args.duration, args.seed)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Generated {len(paths)} deterministic test signals:")
    for path in paths:
        print(f"  {path}")
    return 0


def _run_compare(args: argparse.Namespace) -> int:
    if args.max_lag_ms <= 0 or args.max_delay_ms < 0 or args.max_duration_delta_ms < 0:
        print("error: --max-lag-ms must be positive; delay and duration tolerances must be non-negative", file=sys.stderr)
        return 2
    if args.min_snr_db < 0:
        print("error: --min-snr-db must be non-negative", file=sys.stderr)
        return 2
    try:
        expected, expected_sr = load_audio(args.expected)
        actual, actual_sr = load_audio(args.actual)
        if expected_sr != actual_sr:
            raise ValueError(
                f"Sample rate mismatch: expected file is {expected_sr} Hz, actual file is {actual_sr} Hz. "
                "AudioBench does not silently resample inputs."
            )
        metrics = compare_arrays(expected, actual, expected_sr, max_lag_ms=args.max_lag_ms)
        limits = Thresholds(
            min_snr_db=args.min_snr_db,
            max_delay_ms=args.max_delay_ms,
            max_duration_delta_ms=args.max_duration_delta_ms,
            max_error_dbfs=args.max_error_dbfs,
        )
        evaluation = evaluate_thresholds(metrics, limits)
        json_path = Path(args.json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(
                {
                    "tool": "AudioBench",
                    "version": __version__,
                    "expected_path": str(Path(args.expected)),
                    "actual_path": str(Path(args.actual)),
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
        html_path = write_html_report(
            args.report,
            expected_path=args.expected,
            actual_path=args.actual,
            metrics=metrics,
            evaluation=evaluation,
            expected=expected,
            actual=actual,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"AudioBench result: {evaluation['status']}")
    print(f"  Duration delta: {metrics['duration_delta_ms']:+.3f} ms")
    print(f"  Estimated delay: {_human(metrics['estimated_delay_ms'], ' ms')}")
    print(f"  SNR vs. reference: {_human(metrics['snr_db'], ' dB', exact=metrics['identical_samples'])}")
    print(f"  RMS error: {_human(metrics['error_rms_dbfs'], ' dBFS')}")
    print(f"  HTML report: {html_path}")
    print(f"  JSON result: {json_path}")
    for name, check in evaluation["checks"].items():
        print(f"  [{check['status']}] {name}: {check['message']}")

    if evaluation["status"] == "FAIL" or (args.fail_on_warn and evaluation["status"] == "WARN"):
        return 1
    return 0


def _run_batch(args: argparse.Namespace) -> int:
    try:
        batch_result = run_batch(args.manifest, output_dir=args.output_dir)
    except ManifestError as exc:
        print(f"error in manifest: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"AudioBench suite result: {batch_result.status}")
    print(f"  Suite: {batch_result.name}")
    print(f"  Total tests: {batch_result.total_tests}")
    print(f"  Passed: {batch_result.passed} | Warned: {batch_result.warned} | Failed: {batch_result.failed} | Errors: {batch_result.errored}")
    if batch_result.summary_html:
        print(f"  Summary HTML: {batch_result.summary_html}")
    if batch_result.summary_json:
        print(f"  Summary JSON: {batch_result.summary_json}")

    for test in batch_result.results:
        if test.status == "ERROR":
            print(f"  [{test.status}] {test.id}: {test.error_message}")
        else:
            print(f"  [{test.status}] {test.id}")

    if batch_result.status == "FAIL" or (args.fail_on_warn and batch_result.status == "WARN"):
        return 1
    return 0


def _human(value: float | None, unit: str, *, exact: bool = False) -> str:
    if exact:
        return "inf (sample-identical)"
    if value is None:
        return "not measurable"
    return f"{value:.4f}{unit}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "generate":
        return _run_generate(args)
    if args.command == "compare":
        return _run_compare(args)
    if args.command == "batch":
        return _run_batch(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover - exercised through console script
    raise SystemExit(main())
