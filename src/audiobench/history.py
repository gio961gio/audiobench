"""Historical benchmark tracking and regression trend analysis."""

from __future__ import annotations

import datetime
import html
import json
import math
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import __version__

METRIC_DIRECTIONS: dict[str, str] = {
    "snr_db": "HIGHER",
    "correlation": "HIGHER",
    "estimated_delay_ms": "LOWER",
    "error_rms_dbfs": "LOWER",
    "max_error_dbfs": "LOWER",
    "duration_delta_ms": "LOWER",
    "mean_spectral_magnitude_delta_db": "LOWER",
    "gain_normalized_spectral_shape_error_db": "LOWER",
    "actual_true_peak_dbtp": "LOWER",
    "thd_db": "LOWER",
    "max_band_delta_db": "LOWER",
}

DEFAULT_REGRESSION_TOLERANCES: dict[str, float] = {
    "snr_db": 1.0,
    "estimated_delay_ms": 0.05,
    "actual_true_peak_dbtp": 0.1,
    "thd_db": 1.0,
    "error_rms_dbfs": 1.0,
    "duration_delta_ms": 0.1,
}


class HistoryError(RuntimeError):
    """Raised when benchmark history recording or comparison fails."""


@dataclass
class MetricDelta:
    test_id: str
    metric: str
    baseline_val: float | None
    current_val: float | None
    delta: float | None
    tolerance: float
    is_regression: bool
    direction: str  # "HIGHER" or "LOWER"
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HistoryDiffResult:
    baseline_id: str
    current_id: str
    is_regression: bool
    regressions: list[MetricDelta]
    improvements: list[MetricDelta]
    neutral: list[MetricDelta]
    status_changes: dict[str, dict[str, str]]
    added_tests: list[str]
    removed_tests: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_id": self.baseline_id,
            "current_id": self.current_id,
            "is_regression": self.is_regression,
            "regressions": [r.to_dict() for r in self.regressions],
            "improvements": [i.to_dict() for i in self.improvements],
            "neutral": [n.to_dict() for n in self.neutral],
            "status_changes": self.status_changes,
            "added_tests": self.added_tests,
            "removed_tests": self.removed_tests,
        }


@dataclass
class BenchmarkRecord:
    timestamp: str
    suite_name: str
    status: str
    commit_hash: str | None
    branch: str | None
    is_dirty: bool
    summary: dict[str, int]
    tests: dict[str, dict[str, Any]]
    tool_version: str = __version__

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkRecord:
        return cls(
            timestamp=data["timestamp"],
            suite_name=data.get("suite_name", "Unknown"),
            status=data.get("status", "UNKNOWN"),
            commit_hash=data.get("commit_hash"),
            branch=data.get("branch"),
            is_dirty=data.get("is_dirty", False),
            summary=data.get("summary", {}),
            tests=data.get("tests", {}),
            tool_version=data.get("tool_version", __version__),
        )

    @classmethod
    def from_summary_data(
        cls,
        summary_data: dict[str, Any],
        repo_dir: Path | None = None,
    ) -> BenchmarkRecord:
        git_meta = get_git_metadata(repo_dir)
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        test_map: dict[str, dict[str, Any]] = {}
        for t in summary_data.get("tests", []):
            t_id = t.get("id") or t.get("test_id")
            if not t_id:
                continue
            raw_metrics = t.get("metrics") or {}
            sanitized_metrics: dict[str, Any] = {}
            for m_k, m_v in raw_metrics.items():
                if isinstance(m_v, (int, float)):
                    if math.isnan(m_v):
                        sanitized_metrics[m_k] = None
                    elif math.isinf(m_v):
                        sanitized_metrics[m_k] = 150.0 if m_v > 0 else -150.0
                    else:
                        sanitized_metrics[m_k] = m_v
                elif isinstance(m_v, dict):
                    nested: dict[str, Any] = {}
                    for nk, nv in m_v.items():
                        if isinstance(nv, (int, float)):
                            if math.isnan(nv):
                                nested[nk] = None
                            elif math.isinf(nv):
                                nested[nk] = 150.0 if nv > 0 else -150.0
                            else:
                                nested[nk] = nv
                        else:
                            nested[nk] = nv
                    sanitized_metrics[m_k] = nested
                else:
                    sanitized_metrics[m_k] = m_v

            test_map[t_id] = {
                "status": t.get("status", "UNKNOWN"),
                "metrics": sanitized_metrics,
                "description": t.get("description"),
            }

        return cls(
            timestamp=now_iso,
            suite_name=summary_data.get("name", "Suite"),
            status=summary_data.get("status", "UNKNOWN"),
            commit_hash=git_meta.get("commit_hash"),
            branch=git_meta.get("branch"),
            is_dirty=git_meta.get("is_dirty", False),
            summary=summary_data.get("summary", {}),
            tests=test_map,
            tool_version=summary_data.get("version", __version__),
        )


def get_git_metadata(repo_dir: Path | None = None) -> dict[str, Any]:
    """Retrieve Git commit, branch, and working tree dirtiness safely."""
    base = repo_dir or Path.cwd()
    commit_hash: str | None = None
    branch: str | None = None
    is_dirty = False

    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=base,
            capture_output=True,
            text=True,
            check=False,
            timeout=3,
        )
        if res.returncode == 0 and res.stdout.strip():
            commit_hash = res.stdout.strip()[:10]
    except (OSError, subprocess.SubprocessError):
        pass

    try:
        res_b = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=base,
            capture_output=True,
            text=True,
            check=False,
            timeout=3,
        )
        if res_b.returncode == 0:
            b_str = res_b.stdout.strip()
            if b_str and b_str != "HEAD":
                branch = b_str
            else:
                branch = os.environ.get("GITHUB_REF_NAME") or os.environ.get("CI_COMMIT_BRANCH") or "detached"
    except (OSError, subprocess.SubprocessError):
        pass

    try:
        res_s = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=base,
            capture_output=True,
            text=True,
            check=False,
            timeout=3,
        )
        if res_s.returncode == 0:
            is_dirty = bool(res_s.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass

    return {
        "commit_hash": commit_hash,
        "branch": branch,
        "is_dirty": is_dirty,
    }


def get_default_history_file(base_dir: Path | None = None) -> Path:
    """Return default path to .benchmarks/history.jsonl."""
    base = base_dir or Path.cwd()
    return base / ".benchmarks" / "history.jsonl"


def append_history(record: BenchmarkRecord, history_file: str | Path) -> Path:
    """Append a BenchmarkRecord as a single line JSON to history.jsonl."""
    target = Path(history_file).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record.to_dict(), allow_nan=False) + "\n"
    with open(target, "a", encoding="utf-8") as f:
        f.write(payload)
    return target


def load_history(history_file: str | Path) -> list[BenchmarkRecord]:
    """Read all benchmark entries from history.jsonl."""
    target = Path(history_file).resolve()
    if not target.exists():
        return []
    records: list[BenchmarkRecord] = []
    with open(target, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                data = json.loads(line_str)
                records.append(BenchmarkRecord.from_dict(data))
            except Exception as exc:
                raise HistoryError(f"Corrupt JSON at line {line_no} in '{target}': {exc}") from exc
    return records


def resolve_baseline(
    spec: str,
    history_file: str | Path | None = None,
    base_dir: Path | None = None,
) -> BenchmarkRecord:
    """Resolve a baseline specification to a BenchmarkRecord.

    Spec can be:
    - Path to a JSON summary file.
    - 'latest' (the most recent record in history).
    - 'HEAD~1' or 'previous' (the second most recent record in history).
    - A Git commit hash prefix matching an entry in history.
    """
    path_candidate = Path(spec)
    if path_candidate.exists() and path_candidate.is_file():
        try:
            raw = json.loads(path_candidate.read_text(encoding="utf-8"))
            if "timestamp" in raw and "tests" in raw and isinstance(raw["tests"], dict):
                return BenchmarkRecord.from_dict(raw)
            return BenchmarkRecord.from_summary_data(raw, repo_dir=base_dir)
        except Exception as exc:
            raise HistoryError(f"Cannot load baseline file '{path_candidate}': {exc}") from exc

    h_path = Path(history_file) if history_file else get_default_history_file(base_dir)
    history = load_history(h_path)
    if not history:
        raise HistoryError(f"No history found in '{h_path}' to resolve baseline '{spec}'")

    if spec.lower() == "latest":
        return history[-1]

    if spec.lower() in ("head~1", "previous", "prev"):
        if len(history) < 2:
            raise HistoryError(f"Cannot resolve '{spec}': history has only {len(history)} record(s)")
        return history[-2]

    # Search by commit hash prefix
    for rec in reversed(history):
        if rec.commit_hash and rec.commit_hash.startswith(spec):
            return rec

    raise HistoryError(
        f"Baseline '{spec}' could not be resolved as a file, keyword (latest, HEAD~1), or commit prefix in '{h_path}'"
    )


def _clean_metric_val(val: Any) -> float | None:
    if val is None:
        return None
    try:
        f_val = float(val)
    except (TypeError, ValueError):
        return None
    if math.isnan(f_val):
        return None
    return f_val


def _clamp_metric_val(metric: str, val: float) -> float:
    if metric == "snr_db":
        if math.isinf(val):
            return 150.0 if val > 0 else -120.0
        return max(-120.0, min(150.0, val))
    if metric == "thd_db":
        if math.isinf(val):
            return -150.0 if val < 0 else 0.0
        return max(-150.0, min(0.0, val))
    if math.isinf(val):
        return 9999.0 if val > 0 else -9999.0
    return val


def compare_benchmarks(
    baseline: BenchmarkRecord,
    current: BenchmarkRecord,
    tolerances: dict[str, float] | None = None,
) -> HistoryDiffResult:
    """Compare two benchmark records and detect metric regressions and status changes."""
    active_tolerances = {**DEFAULT_REGRESSION_TOLERANCES, **(tolerances or {})}
    b_tests = baseline.tests
    c_tests = current.tests

    all_test_ids = set(b_tests.keys()) | set(c_tests.keys())
    common_ids = sorted(set(b_tests.keys()) & set(c_tests.keys()))
    added_ids = sorted(set(c_tests.keys()) - set(b_tests.keys()))
    removed_ids = sorted(set(b_tests.keys()) - set(c_tests.keys()))

    regressions: list[MetricDelta] = []
    improvements: list[MetricDelta] = []
    neutral: list[MetricDelta] = []
    status_changes: dict[str, dict[str, str]] = {}

    has_overall_regression = False

    for t_id in common_ids:
        b_case = b_tests[t_id]
        c_case = c_tests[t_id]
        b_status = b_case.get("status", "UNKNOWN")
        c_status = c_case.get("status", "UNKNOWN")

        if b_status != c_status:
            status_changes[t_id] = {"baseline": b_status, "current": c_status}
            if b_status in ("PASS", "WARN") and c_status in ("FAIL", "ERROR"):
                has_overall_regression = True

        b_metrics = b_case.get("metrics") or {}
        c_metrics = c_case.get("metrics") or {}

        # Evaluate all tracked metrics
        for metric, direction in METRIC_DIRECTIONS.items():
            if metric not in b_metrics or metric not in c_metrics:
                continue

            b_raw = _clean_metric_val(b_metrics.get(metric))
            c_raw = _clean_metric_val(c_metrics.get(metric))

            if b_raw is None or c_raw is None:
                continue

            # Special case: both sample-identical inf
            if math.isinf(b_raw) and math.isinf(c_raw) and (b_raw > 0) == (c_raw > 0):
                delta = 0.0
            else:
                b_clamped = _clamp_metric_val(metric, b_raw)
                c_clamped = _clamp_metric_val(metric, c_raw)
                delta = c_clamped - b_clamped

            tol = active_tolerances.get(metric, 1.0)
            is_reg = False
            is_imp = False

            unit = " dB" if metric.endswith("_db") else ""
            label = "SNR" if metric == "snr_db" else metric

            if direction == "HIGHER":
                if delta < -tol:
                    is_reg = True
                    msg = f"{label} dropped by {abs(delta):.2f}{unit} (tolerance: {tol}{unit})"
                elif delta > tol:
                    is_imp = True
                    msg = f"{label} improved by {delta:.2f}{unit}"
                else:
                    msg = f"{label} changed by {delta:+.2f}{unit} (within {tol}{unit} tolerance)"
            else:
                # E.g. Delay, THD, Peak: current is worse if delta > +tol
                if delta > tol:
                    is_reg = True
                    msg = f"{metric} increased by {delta:.4f} (tolerance: {tol})"
                elif delta < -tol:
                    is_imp = True
                    msg = f"{metric} decreased by {abs(delta):.4f}"
                else:
                    msg = f"{metric} changed by {delta:+.4f} (within {tol} tolerance)"

            m_delta = MetricDelta(
                test_id=t_id,
                metric=metric,
                baseline_val=b_raw,
                current_val=c_raw,
                delta=delta,
                tolerance=tol,
                is_regression=is_reg,
                direction=direction,
                message=msg,
            )

            if is_reg:
                regressions.append(m_delta)
                has_overall_regression = True
            elif is_imp:
                improvements.append(m_delta)
            else:
                neutral.append(m_delta)

    b_id_label = baseline.commit_hash or baseline.timestamp[:19]
    c_id_label = current.commit_hash or current.timestamp[:19]

    return HistoryDiffResult(
        baseline_id=b_id_label,
        current_id=c_id_label,
        is_regression=has_overall_regression,
        regressions=regressions,
        improvements=improvements,
        neutral=neutral,
        status_changes=status_changes,
        added_tests=added_ids,
        removed_tests=removed_ids,
    )


def _fmt(val: float | None, unit: str = "", digits: int = 2) -> str:
    if val is None:
        return "—"
    if math.isinf(val):
        return "inf (identical)" if val > 0 else "-inf"
    return f"{val:.{digits}f}{unit}"


def _sparkline_svg(values: list[tuple[str, float]], metric_name: str, width: int = 220, height: int = 44) -> str:
    """Generate a clean vector SVG trendline without client-side JS."""
    if len(values) < 2:
        return '<span class="muted small">Insufficient history</span>'

    nums = [v[1] for v in values]
    min_v, max_v = min(nums), max(nums)
    pad = 6
    plot_w = width - 2 * pad
    plot_h = height - 2 * pad

    if max_v == min_v:
        y_coords = [height / 2.0] * len(nums)
    else:
        y_coords = [pad + (1.0 - (v - min_v) / (max_v - min_v)) * plot_h for v in nums]

    x_step = plot_w / (len(nums) - 1)
    points_str = " ".join(f"{pad + i * x_step:.1f},{y_coords[i]:.1f}" for i in range(len(nums)))

    circles_svg = []
    stroke_color = "#2563eb" if metric_name == "snr_db" else "#ea580c"
    dot_color = "#1d4ed8" if metric_name == "snr_db" else "#c2410c"

    for i, (label, val) in enumerate(values):
        cx = pad + i * x_step
        cy = y_coords[i]
        val_str = "inf" if math.isinf(val) else f"{val:.2f}"
        circles_svg.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3" fill="{dot_color}">'
            f"<title>{html.escape(label)}: {val_str}</title></circle>"
        )

    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'style="overflow:visible; vertical-align:middle;">'
        f'<polyline fill="none" stroke="{stroke_color}" stroke-width="2" points="{points_str}" stroke-linecap="round" stroke-linejoin="round"/>'
        f'{"".join(circles_svg)}'
        f"</svg>"
    )


def generate_history_html(
    records: list[BenchmarkRecord],
    output_path: str | Path,
    max_runs: int = 50,
) -> Path:
    """Render a standalone HTML dashboard showing benchmark metrics over time."""
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        out.write_text("<!doctype html><html><body><p>No records in history.</p></body></html>", encoding="utf-8")
        return out

    # Limit to the most recent max_runs for responsive sparklines
    active_records = records[-max_runs:] if len(records) > max_runs else records

    # Collect all unique test IDs across active records
    all_test_ids: set[str] = set()
    for rec in active_records:
        all_test_ids.update(rec.tests.keys())
    sorted_test_ids = sorted(all_test_ids)

    rows: list[str] = []
    for t_id in sorted_test_ids:
        snr_series: list[tuple[str, float]] = []
        delay_series: list[tuple[str, float]] = []

        latest_status = "UNKNOWN"
        latest_snr_str = "—"
        latest_delay_str = "—"

        for r in active_records:
            lbl = f"{r.commit_hash or r.timestamp[:10]} ({r.branch or 'local'})"
            if t_id in r.tests:
                t_data = r.tests[t_id]
                latest_status = t_data.get("status", "UNKNOWN")
                metrics = t_data.get("metrics") or {}
                snr = _clean_metric_val(metrics.get("snr_db"))
                delay = _clean_metric_val(metrics.get("estimated_delay_ms"))
                if snr is not None:
                    snr_series.append((lbl, _clamp_metric_val("snr_db", snr)))
                    latest_snr_str = _fmt(snr, " dB")
                if delay is not None:
                    delay_series.append((lbl, delay))
                    latest_delay_str = _fmt(delay, " ms", 3)

        snr_chart = _sparkline_svg(snr_series, "snr_db")
        delay_chart = _sparkline_svg(delay_series, "estimated_delay_ms")
        st_class = latest_status.lower()

        rows.append(
            f"<tr>"
            f"<td><strong>{html.escape(t_id)}</strong></td>"
            f"<td><span class='pill {st_class}'>{html.escape(latest_status)}</span></td>"
            f"<td>{latest_snr_str}</td>"
            f"<td>{snr_chart}</td>"
            f"<td>{latest_delay_str}</td>"
            f"<td>{delay_chart}</td>"
            f"</tr>"
        )

    # Run history table
    run_rows: list[str] = []
    for r in reversed(records[-15:]):  # show last 15 runs
        c_short = r.commit_hash or "—"
        dirty_badge = " <span class='pill warn small'>dirty</span>" if r.is_dirty else ""
        run_st = r.status.lower()
        run_rows.append(
            f"<tr>"
            f"<td>{html.escape(r.timestamp[:19].replace('T', ' '))}</td>"
            f"<td><code>{html.escape(c_short)}</code>{dirty_badge}</td>"
            f"<td>{html.escape(r.branch or '—')}</td>"
            f"<td><span class='pill {run_st}'>{html.escape(r.status)}</span></td>"
            f"<td>{r.summary.get('passed', 0)} / {r.summary.get('total', 0)}</td>"
            f"</tr>"
        )

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AudioBench · Historical Benchmark Trends</title>
<style>
:root {{ color-scheme: light dark; font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif; }}
body {{ max-width: 1100px; margin: 0 auto; padding: 28px 20px 56px; line-height: 1.5; }}
header {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 20px; flex-wrap: wrap; margin-bottom: 24px; }}
h1 {{ margin: 0 0 4px; font-size: 2rem; }}
.sub {{ color: #64748b; margin: 0; }}
section {{ margin-top: 32px; }}
h2 {{ font-size: 1.2rem; margin-bottom: 12px; }}
.card {{ border: 1px solid #94a3b855; border-radius: 12px; padding: 16px; overflow: hidden; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ text-align: left; padding: 10px 12px; border-bottom: 1px solid #94a3b833; vertical-align: middle; font-size: .9rem; }}
th {{ font-weight: 650; color: #475569; }}
tbody tr:last-child td {{ border-bottom: 0; }}
.pill {{ display: inline-block; border-radius: 999px; padding: 3px 10px; font-size: .78rem; font-weight: 750; }}
.pass {{ background: #dcfce7; color: #166534; }}
.fail {{ background: #fee2e2; color: #991b1b; }}
.warn {{ background: #fef3c7; color: #92400e; }}
.small {{ font-size: .75rem; padding: 2px 6px; }}
.muted {{ color: #64748b; }}
code {{ font-family: ui-monospace, SFMono-Regular, Consolas, monospace; background: #94a3b822; padding: 2px 5px; border-radius: 4px; }}
</style>
</head>
<body>
<header>
  <div>
    <h1>AudioBench Benchmark History</h1>
    <p class="sub">Longitudinal regression trend tracking · {len(records)} runs recorded</p>
  </div>
</header>

<section>
  <h2>Test Metric Sparklines</h2>
  <div class="card">
    <table>
      <thead>
        <tr>
          <th>Test ID</th>
          <th>Latest Status</th>
          <th>Latest SNR</th>
          <th>SNR Trend (dB)</th>
          <th>Latest Delay</th>
          <th>Delay Trend (ms)</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
  </div>
</section>

<section>
  <h2>Recent Execution History</h2>
  <div class="card">
    <table>
      <thead>
        <tr>
          <th>Date (UTC)</th>
          <th>Git Commit</th>
          <th>Branch</th>
          <th>Status</th>
          <th>Passed / Total</th>
        </tr>
      </thead>
      <tbody>
        {''.join(run_rows)}
      </tbody>
    </table>
  </div>
</section>

<section>
  <p class="muted small">Generated by AudioBench {__version__}. Native standalone SVG charts.</p>
</section>
</body>
</html>"""

    out.write_text(doc, encoding="utf-8")
    return out
