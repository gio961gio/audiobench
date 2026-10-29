"""Command-line interface for AudioBench."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from . import __version__
from .audio import load_audio, write_test_signals
from .batch import ManifestError, run_batch
from .history import (
    BenchmarkRecord,
    append_history,
    compare_benchmarks,
    generate_history_html,
    get_default_history_file,
    load_history,
    resolve_baseline,
)
from .metrics import Thresholds, compare_arrays, evaluate_thresholds
from .plugins import PluginError, get_plugin_adapter
from .report import write_html_report


def _add_generate_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("generate", help="Generate deterministic audio test signals")
    parser.add_argument("-o", "--output-dir", default="test_signals", help="Directory for generated WAV files")
    parser.add_argument("-r", "--sample-rate", type=int, default=48_000, help="Sample rate in Hz (default: 48000)")
    parser.add_argument("-d", "--duration", type=float, default=1.0, help="Signal duration in seconds (default: 1.0)")
    parser.add_argument("--seed", type=int, default=7, help="Seed for reproducible white noise")


def _add_compare_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("compare", help="Compare expected and actual offline audio renders")
    parser.add_argument("-e", "--expected", required=True, help="Path to reference/expected WAV file")
    parser.add_argument("-a", "--actual", required=True, help="Path to render under test")
    parser.add_argument("-r", "--report", default="reports/latest.html", help="HTML report path (default: reports/latest.html)")
    parser.add_argument("-j", "--json", dest="json_path", default="reports/latest.json", help="JSON output path (default: reports/latest.json)")
    parser.add_argument("--min-snr-db", type=float, default=60.0, help="Minimum reference-to-error SNR for PASS (default: 60 dB)")
    parser.add_argument("--max-delay-ms", type=float, default=0.5, help="Maximum absolute estimated delay for PASS (default: 0.5 ms)")
    parser.add_argument("--max-duration-delta-ms", type=float, default=1.0, help="Maximum absolute duration delta for PASS (default: 1 ms)")
    parser.add_argument("--max-error-dbfs", type=float, default=-80.0, help="Maximum RMS error for a silent reference (default: -80 dBFS)")
    parser.add_argument("--max-lag-ms", type=float, default=100.0, help="Search range for delay estimation (default: 100 ms)")
    parser.add_argument("--max-band-delta-db", type=float, default=None, help="Maximum allowed dB deviation in any frequency band")
    parser.add_argument("--max-thd-db", type=float, default=None, help="Maximum allowed THD in dB for sinusoidal renders")
    parser.add_argument("--max-true-peak-dbfs", type=float, default=None, help="Maximum allowed True-Peak in dBTP")
    parser.add_argument("--fail-on-warn", action="store_true", help="Return exit code 1 for WARN as well as FAIL")


def _add_batch_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("batch", help="Run a suite of regression comparisons from a JSON manifest")
    parser.add_argument("-m", "--manifest", required=True, help="Path to JSON test manifest file")
    parser.add_argument("-o", "--output-dir", default="reports/batch", help="Directory for generated reports (default: reports/batch)")
    parser.add_argument("--fail-on-warn", action="store_true", help="Return exit code 1 for WARN as well as FAIL")
    parser.add_argument(
        "--baseline",
        help="Compare run against historical baseline (commit hash prefix, 'latest', or summary.json path)",
    )
    parser.add_argument(
        "--record-history",
        action="store_true",
        help="Automatically append suite run to benchmark history (.benchmarks/history.jsonl)",
    )
    parser.add_argument("--history-file", default=None, help="Custom path to history.jsonl")


def _add_render_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("render", help="Render an audio file through a plugin or DSP chain offline")
    parser.add_argument(
        "-p",
        "--plugin",
        required=True,
        help="Plugin path (VST3/AU) or built-in effect name ('gain', 'lowpass', 'highpass', 'delay', 'soft_clip')",
    )
    parser.add_argument("-i", "--input", required=True, help="Input WAV file path")
    parser.add_argument("-o", "--output", required=True, help="Output WAV file path")
    parser.add_argument(
        "--type",
        choices=["vst3", "au", "builtin", "cli"],
        default=None,
        help="Plugin host type (default: inferred from --plugin)",
    )
    parser.add_argument(
        "--params",
        help="JSON string or path to JSON file containing plugin parameters",
    )


def _add_history_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("history", help="Record and inspect longitudinal benchmark trends")
    h_sub = parser.add_subparsers(dest="history_action", required=True)

    # record
    rec_p = h_sub.add_parser("record", help="Append a batch summary to benchmark history")
    rec_p.add_argument("-s", "--summary", required=True, help="Path to batch summary.json")
    rec_p.add_argument("--history-file", default=None, help="Custom path to history.jsonl")

    # diff
    diff_p = h_sub.add_parser("diff", help="Compare current run against a historical baseline")
    diff_p.add_argument(
        "-b",
        "--baseline",
        required=True,
        help="Baseline spec (commit hash prefix, 'latest', 'HEAD~1', or summary.json path)",
    )
    diff_p.add_argument("-c", "--current", required=True, help="Current summary.json path")
    diff_p.add_argument("--history-file", default=None, help="Custom path to history.jsonl")
    diff_p.add_argument(
        "--max-snr-drop",
        type=float,
        default=1.0,
        help="Allowed SNR drop in dB before flagging regression (default: 1.0 dB)",
    )
    diff_p.add_argument(
        "--max-delay-growth-ms",
        type=float,
        default=0.05,
        help="Allowed delay growth in ms before flagging regression (default: 0.05 ms)",
    )

    # report
    rep_p = h_sub.add_parser("report", help="Generate an HTML dashboard showing metric sparkline trends over time")
    rep_p.add_argument("-o", "--output", default="reports/history.html", help="Output HTML report path (default: reports/history.html)")
    rep_p.add_argument("--history-file", default=None, help="Custom path to history.jsonl")


def _add_ui_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("ui", help="Launch the interactive Streamlit Web Studio & A/B Differential Player")
    parser.add_argument("--port", type=int, default=8501, help="Port to run the web server on (default: 8501)")
    parser.add_argument("--host", default="localhost", help="Host address to bind to (default: localhost)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open the browser on launch")


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
    _add_render_parser(subparsers)
    _add_history_parser(subparsers)
    _add_ui_parser(subparsers)
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
    if args.max_band_delta_db is not None and args.max_band_delta_db < 0:
        print("error: --max-band-delta-db must be non-negative", file=sys.stderr)
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
            max_band_delta_db=args.max_band_delta_db,
            max_thd_db=args.max_thd_db,
            max_true_peak_dbfs=args.max_true_peak_dbfs,
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
                    "thresholds": asdict(limits),
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
    if metrics.get("actual_true_peak_dbtp") is not None:
        print(f"  True-Peak: {metrics['actual_true_peak_dbtp']:.2f} dBTP")
    thd_info = metrics.get("actual_thd") or {}
    if thd_info.get("is_sinusoidal") and thd_info.get("thd_db") is not None:
        print(f"  THD: {thd_info['thd_db']:.2f} dB ({thd_info['thd_percent']:.3f}%)")
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

    h_file = Path(args.history_file) if getattr(args, "history_file", None) else get_default_history_file()
    if getattr(args, "record_history", False):
        try:
            rec = BenchmarkRecord.from_summary_data(batch_result.to_dict())
            append_history(rec, h_file)
            print(f"  Benchmark history recorded: {h_file}")
        except Exception as exc:
            print(f"warning: failed to record benchmark history: {exc}", file=sys.stderr)

    if getattr(args, "baseline", None):
        try:
            base_rec = resolve_baseline(args.baseline, history_file=h_file)
            curr_rec = BenchmarkRecord.from_summary_data(batch_result.to_dict())
            diff_res = compare_benchmarks(base_rec, curr_rec)
            if diff_res.is_regression:
                print(f"\n  [REGRESSION DETECTED against baseline '{diff_res.baseline_id}']")
                for r in diff_res.regressions:
                    print(f"    - [{r.test_id}] {r.message}")
                return 1
            else:
                print(f"\n  [Baseline Check PASS: no regression against '{diff_res.baseline_id}']")
        except Exception as exc:
            print(f"error comparing against baseline: {exc}", file=sys.stderr)
            return 2

    if batch_result.status == "FAIL" or (args.fail_on_warn and batch_result.status == "WARN"):
        return 1
    return 0


def _parse_cli_params(raw: str) -> dict[str, Any]:
    raw_str = raw.strip()
    # 1. Direct JSON attempt
    try:
        parsed = json.loads(raw_str)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # 2. PowerShell quote-stripped JSON attempt, e.g. {gain_db: -3.0}
    if raw_str.startswith("{") and raw_str.endswith("}"):
        fixed = re.sub(r'([{\s,])([a-zA-Z_]\w*)\s*:', r'\1"\2":', raw_str)
        try:
            parsed = json.loads(fixed)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    # 3. Simple key=value CLI format, e.g. gain_db=-3.0 or gain_db=-3.0,cutoff_hz=1000
    if "=" in raw_str and not (raw_str.startswith("{") or raw_str.startswith("[")):
        parsed = {}
        for item in raw_str.split(","):
            if "=" in item:
                k, v = item.split("=", 1)
                k = k.strip()
                v = v.strip()
                if v.lower() == "true":
                    parsed[k] = True
                elif v.lower() == "false":
                    parsed[k] = False
                else:
                    try:
                        parsed[k] = int(v)
                    except ValueError:
                        try:
                            parsed[k] = float(v)
                        except ValueError:
                            parsed[k] = v
        if parsed:
            return parsed

    # 4. Fallback to standard json.loads to raise informative ValueError
    return json.loads(raw_str)


def _run_render(args: argparse.Namespace) -> int:
    params: dict[str, Any] = {}
    if args.params:
        params_path = Path(args.params)
        if params_path.exists() and params_path.is_file():
            try:
                params = json.loads(params_path.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"error: invalid JSON in parameters file '{params_path}': {exc}", file=sys.stderr)
                return 2
        else:
            try:
                params = _parse_cli_params(args.params)
            except Exception as exc:
                print(f"error: invalid JSON in --params: {exc}", file=sys.stderr)
                return 2

    plugin_type = args.type
    if plugin_type is None:
        p_str = args.plugin.lower()
        if p_str.endswith(".vst3") or ".vst3" in p_str:
            plugin_type = "vst3"
        elif p_str.endswith(".component"):
            plugin_type = "au"
        elif p_str in ("gain", "lowpass", "highpass", "delay", "soft_clip"):
            plugin_type = "builtin"
        elif Path(args.plugin).exists():
            plugin_type = "vst3"
        else:
            plugin_type = "builtin"

    config: dict[str, Any] = {"type": plugin_type}
    if plugin_type == "builtin":
        config["effect"] = args.plugin
        config["parameters"] = params
    elif plugin_type in ("vst3", "au"):
        config["path"] = args.plugin
        config["parameters"] = params
    elif plugin_type == "cli":
        config["command"] = [args.plugin]
        config["parameters"] = params

    try:
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        _, sr = load_audio(input_path)
        adapter = get_plugin_adapter(config)
        output_file = adapter.render_file(input_path, args.output, sr, params)
    except (PluginError, OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Rendered: {output_file} ({sr} Hz)")
    return 0


def _run_history(args: argparse.Namespace) -> int:
    h_file = Path(args.history_file) if args.history_file else get_default_history_file()

    if args.history_action == "record":
        summary_path = Path(args.summary)
        if not summary_path.exists():
            print(f"error: summary file not found: {summary_path}", file=sys.stderr)
            return 2
        try:
            raw = json.loads(summary_path.read_text(encoding="utf-8"))
            rec = BenchmarkRecord.from_summary_data(raw)
            out_file = append_history(rec, h_file)
            c_label = rec.commit_hash or "unversioned"
            dirty_str = " (dirty)" if rec.is_dirty else ""
            print(f"Recorded benchmark for suite '{rec.suite_name}' [{c_label}{dirty_str}] in {out_file}")
            return 0
        except Exception as exc:
            print(f"error recording benchmark: {exc}", file=sys.stderr)
            return 2

    if args.history_action == "diff":
        try:
            base_rec = resolve_baseline(args.baseline, history_file=h_file)
            curr_path = Path(args.current)
            if not curr_path.exists():
                print(f"error: current summary file not found: {curr_path}", file=sys.stderr)
                return 2
            curr_raw = json.loads(curr_path.read_text(encoding="utf-8"))
            curr_rec = BenchmarkRecord.from_summary_data(curr_raw)

            tolerances = {
                "snr_db": args.max_snr_drop,
                "estimated_delay_ms": args.max_delay_growth_ms,
            }
            diff_res = compare_benchmarks(base_rec, curr_rec, tolerances=tolerances)
        except Exception as exc:
            print(f"error during history diff: {exc}", file=sys.stderr)
            return 2

        print(f"AudioBench History Diff: Baseline [{diff_res.baseline_id}] vs Current [{diff_res.current_id}]")
        if diff_res.is_regression:
            print("  Status: REGRESSION DETECTED")
        else:
            print("  Status: OK (no regression)")

        if diff_res.regressions:
            print("\n  Regressions:")
            for r in diff_res.regressions:
                print(f"    - [{r.test_id}] {r.message}")

        if diff_res.improvements:
            print("\n  Improvements:")
            for i in diff_res.improvements:
                print(f"    + [{i.test_id}] {i.message}")

        if diff_res.status_changes:
            print("\n  Status Changes:")
            for t_id, sc in diff_res.status_changes.items():
                print(f"    * [{t_id}] {sc['baseline']} -> {sc['current']}")

        if diff_res.added_tests:
            print(f"\n  Added tests ({len(diff_res.added_tests)}): {', '.join(diff_res.added_tests)}")
        if diff_res.removed_tests:
            print(f"\n  Removed tests ({len(diff_res.removed_tests)}): {', '.join(diff_res.removed_tests)}")

        return 1 if diff_res.is_regression else 0

    if args.history_action == "report":
        try:
            records = load_history(h_file)
            if not records:
                print(f"warning: no records found in '{h_file}'. Empty report will be created.", file=sys.stderr)
            out_path = generate_history_html(records, args.output)
            print(f"Generated benchmark history report: {out_path} ({len(records)} runs)")
            return 0
        except Exception as exc:
            print(f"error generating history report: {exc}", file=sys.stderr)
            return 2

    return 2


def _run_ui(args: argparse.Namespace) -> int:
    import importlib.util
    import subprocess

    if importlib.util.find_spec("streamlit") is None:
        print(
            "error: AudioBench UI requires 'streamlit'.\n"
            "Install it with: pip install 'audiobench[ui]'",
            file=sys.stderr,
        )
        return 1

    app_path = Path(__file__).resolve().parent / "app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(args.port),
        "--server.address",
        args.host,
    ]
    if args.no_browser:
        cmd.extend(["--server.headless", "true"])
    else:
        cmd.extend(["--server.headless", "false"])

    print(f"Launching AudioBench Web Studio on http://{args.host}:{args.port}...")
    try:
        completed = subprocess.run(cmd, check=False)
        return completed.returncode
    except KeyboardInterrupt:
        print("\nAudioBench Web Studio stopped.")
        return 0
    except OSError as exc:
        print(f"error launching UI: {exc}", file=sys.stderr)
        return 2


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
    if args.command == "render":
        return _run_render(args)
    if args.command == "history":
        return _run_history(args)
    if args.command == "ui":
        return _run_ui(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover - exercised through console script
    raise SystemExit(main())
