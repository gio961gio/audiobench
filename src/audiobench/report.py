"""Self-contained HTML report generation (no CDN or external assets)."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np


def _fmt(value: Any, unit: str = "", digits: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, str):
        return html.escape(value)
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (int, float)):
        if not np.isfinite(value):
            return "—"
        return f"{value:.{digits}f}{unit}"
    return html.escape(str(value))


def _waveform_svg(expected: np.ndarray, actual: np.ndarray, sample_rate: int) -> str:
    width, height = 900, 220
    left, right, top, bottom = 48, 12, 24, 28
    plot_w, plot_h = width - left - right, height - top - bottom
    frames = min(len(expected), len(actual), sample_rate * 3)
    if frames < 2:
        return '<p class="muted">Not enough samples to draw a waveform.</p>'
    exp = np.mean(expected[:frames], axis=1)
    act = np.mean(actual[:frames], axis=1)
    # Peak-preserving binning keeps transients visible in the compact plot.
    count = min(900, frames)
    edges = np.linspace(0, frames, count + 1, dtype=int)
    x_values = np.arange(count)

    def envelope(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mins = np.empty(count)
        maxs = np.empty(count)
        for i in range(count):
            chunk = values[edges[i] : max(edges[i] + 1, edges[i + 1])]
            mins[i] = float(np.min(chunk)) if chunk.size else 0.0
            maxs[i] = float(np.max(chunk)) if chunk.size else 0.0
        return mins, maxs

    exp_min, exp_max = envelope(exp)
    act_min, act_max = envelope(act)

    def x_px(i: int) -> float:
        return left + (i / max(1, count - 1)) * plot_w

    def y_px(v: float) -> float:
        return top + (1.0 - (np.clip(v, -1.0, 1.0) + 1.0) / 2.0) * plot_h

    def bars(low: np.ndarray, high: np.ndarray) -> str:
        segs = []
        for i in x_values:
            segs.append(
                f"M{x_px(int(i)):.2f},{y_px(float(low[i])):.2f} "
                f"L{x_px(int(i)):.2f},{y_px(float(high[i])):.2f}"
            )
        return " ".join(segs)

    zero_y = y_px(0.0)
    return (
        f'<svg role="img" aria-label="Expected and actual audio waveforms" '
        f'viewBox="0 0 {width} {height}" width="100%" height="auto">'
        f'<line x1="{left}" y1="{zero_y:.2f}" x2="{width-right}" y2="{zero_y:.2f}" stroke="#cbd5e1" stroke-dasharray="4 4"/>'
        f'<path d="{bars(exp_min, exp_max)}" fill="none" stroke="#2563eb" stroke-width="1.25" opacity="0.9"/>'
        f'<path d="{bars(act_min, act_max)}" fill="none" stroke="#ea580c" stroke-width="1.0" opacity="0.85"/>'
        f'<text x="{left}" y="16" font-size="11" fill="currentColor">First {frames / sample_rate:.2f} s · mono fold-down</text>'
        f'<text x="{left}" y="{height-7}" font-size="11" fill="currentColor">0 s</text>'
        f'<text x="{width-right}" y="{height-7}" font-size="11" text-anchor="end" fill="currentColor">{frames / sample_rate:.2f} s</text>'
        '</svg>'
    )


def _row(label: str, observed: Any, unit: str = "", digits: int = 2) -> str:
    return (
        "<tr><th>" + html.escape(label) + "</th><td>"
        + _fmt(observed, unit, digits) + "</td></tr>"
    )


def write_html_report(
    output_path: str | Path,
    *,
    expected_path: str | Path,
    actual_path: str | Path,
    metrics: dict[str, Any],
    evaluation: dict[str, Any],
    expected: np.ndarray,
    actual: np.ndarray,
) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    status = evaluation["status"]
    status_class = "na" if status == "N/A" else status.lower()
    checks = []
    for name, check in evaluation["checks"].items():
        checks.append(
            "<tr>"
            f"<th>{html.escape(name.title())}</th>"
            f"<td><span class='pill {html.escape(check['status'].lower())}'>{html.escape(check['status'])}</span></td>"
            f"<td>{html.escape(check['message'])}</td>"
            "</tr>"
        )

    metric_rows = "\n".join(
        [
            _row("Sample rate", metrics["sample_rate_hz"], " Hz", 0),
            _row("Channels", metrics["channels"], "", 0),
            _row("Expected duration", metrics["expected_duration_s"], " s", 4),
            _row("Actual duration", metrics["actual_duration_s"], " s", 4),
            _row("Duration delta (actual − expected)", metrics["duration_delta_ms"], " ms", 3),
            _row("Expected RMS", metrics["expected_rms_dbfs"], " dBFS"),
            _row("Actual RMS", metrics["actual_rms_dbfs"], " dBFS"),
            _row("Expected sample peak", metrics["expected_peak_dbfs"], " dBFS"),
            _row("Actual sample peak", metrics["actual_peak_dbfs"], " dBFS"),
            _row("RMS error", metrics["error_rms_dbfs"], " dBFS"),
            _row("SNR vs. reference", metrics["snr_db"], " dB"),
            _row("Relative error", metrics["relative_error_db"], " dB"),
            _row("Sample correlation", metrics["correlation"], "", 5),
            _row("Mean spectral magnitude delta", metrics["mean_spectral_delta_db"], " dB"),
            _row("Gain-normalized spectral shape error", metrics["spectral_shape_error_db"], " dB"),
            _row("Estimated latency", metrics["estimated_delay_ms"], " ms", 4),
            _row("Latency correlation score", metrics["delay_correlation"], "", 4),
            _row("SNR after estimated alignment", metrics["aligned_snr_db"], " dB"),
            _row("Samples identical after alignment", metrics["aligned_samples_identical"]),
            _row("Maximum absolute sample error", metrics["max_abs_error"], "", 8),
        ]
    )
    raw_json = html.escape(json.dumps({"metrics": metrics, "evaluation": evaluation}, indent=2, allow_nan=False))
    waveform = _waveform_svg(expected, actual, int(metrics["sample_rate_hz"]))
    title = "AudioBench Comparison Report"
    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ color-scheme: light dark; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
body {{ max-width: 1040px; margin: 0 auto; padding: 28px 20px 56px; line-height: 1.5; }}
header {{ display:flex; align-items:flex-start; justify-content:space-between; gap:20px; flex-wrap:wrap; }}
h1 {{ margin:0 0 4px; font-size: clamp(1.8rem, 4vw, 2.5rem); }}
.sub {{ color: #64748b; margin:0; overflow-wrap:anywhere; }}
section {{ margin-top:28px; }}
h2 {{ font-size:1.15rem; margin-bottom:10px; }}
.pill {{ display:inline-block; border-radius:999px; padding:4px 12px; font-size:.82rem; font-weight:800; letter-spacing:.04em; }}
.pass {{ background:#dcfce7; color:#166534; }} .fail {{ background:#fee2e2; color:#991b1b; }} .warn {{ background:#fef3c7; color:#92400e; }} .na {{ background:#e2e8f0; color:#334155; }}
.card {{ border:1px solid #94a3b855; border-radius:14px; padding:16px; overflow:hidden; }}
table {{ width:100%; border-collapse:collapse; }}
th,td {{ text-align:left; padding:9px 10px; border-bottom:1px solid #94a3b844; vertical-align:top; }}
th {{ font-weight:650; }}
tbody tr:last-child th, tbody tr:last-child td {{ border-bottom:0; }}
.metrics th {{ width:55%; }}
legend {{ color:#64748b; font-size:.9rem; }}
.swatch {{ display:inline-block; width:11px; height:11px; border-radius:2px; margin:0 5px 0 14px; }}
.swatch:first-child {{ margin-left:0; }}
pre {{ white-space:pre-wrap; overflow-wrap:anywhere; font-size:.8rem; max-height:440px; overflow:auto; }}
.muted {{ color:#64748b; }}
@media (max-width:600px) {{ body {{ padding:18px 12px 36px; }} th,td {{ padding:7px 5px; }} }}
</style>
</head>
<body>
<header>
  <div><h1>AudioBench</h1><p class="sub">Offline render comparison · {html.escape(Path(expected_path).name)} vs. {html.escape(Path(actual_path).name)}</p></div>
  <span class="pill {status_class}">{html.escape(status)}</span>
</header>
<section><h2>Quality gates</h2><div class="card"><table><thead><tr><th>Check</th><th>Status</th><th>Details</th></tr></thead><tbody>{''.join(checks)}</tbody></table></div></section>
<section><h2>Waveform overview</h2><div class="card"><p><span class="swatch" style="background:#2563eb"></span>Expected <span class="swatch" style="background:#ea580c"></span>Actual</p>{waveform}<p class="muted">Waveforms are folded down to mono for display only. The metrics below use all channels. Latency estimation can be ambiguous for periodic signals.</p></div></section>
<section><h2>Metrics</h2><div class="card"><table class="metrics"><tbody>{metric_rows}</tbody></table></div></section>
<section><h2>Machine-readable result</h2><div class="card"><pre>{raw_json}</pre></div></section>
<section><p class="muted">Generated by AudioBench. This report compares offline audio renders; it does not load or execute plugin binaries.</p></section>
</body></html>"""
    out.write_text(doc, encoding="utf-8")
    return out


def write_batch_html_summary(
    output_path: str | Path,
    batch_result: Any,
) -> Path:
    """Generate a self-contained HTML summary dashboard for a batch test suite."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    status = getattr(batch_result, "status", "UNKNOWN")
    status_class = status.lower()

    rows: list[str] = []
    for test in getattr(batch_result, "results", []):
        test_status = getattr(test, "status", "UNKNOWN")
        case_class = test_status.lower()

        metrics = getattr(test, "metrics", None) or {}
        snr = metrics.get("snr_db")
        snr_str = "∞ (exact)" if metrics.get("identical_samples") else _fmt(snr, " dB")
        delay = _fmt(metrics.get("estimated_delay_ms"), " ms", 3)
        err_rms = _fmt(metrics.get("error_rms_dbfs"), " dBFS")
        dur_delta = _fmt(metrics.get("duration_delta_ms"), " ms", 2)

        links: list[str] = []
        if getattr(test, "html_report", None):
            report_name = Path(test.html_report).name
            links.append(f'<a href="{html.escape(report_name)}">Report</a>')
        if getattr(test, "json_report", None):
            json_name = Path(test.json_report).name
            links.append(f'<a href="{html.escape(json_name)}">JSON</a>')

        link_html = " · ".join(links) if links else "—"

        err_html = ""
        if getattr(test, "error_message", None):
            err_html = f'<div class="error-msg">{html.escape(test.error_message)}</div>'

        desc = getattr(test, "description", None) or ""
        desc_html = f'<div class="muted small">{html.escape(desc)}</div>' if desc else ""

        rows.append(
            f"<tr>"
            f"<td><strong>{html.escape(test.id)}</strong>{desc_html}</td>"
            f"<td><span class='pill {case_class}'>{html.escape(test_status)}</span></td>"
            f"<td>{snr_str}</td>"
            f"<td>{delay}</td>"
            f"<td>{err_rms}</td>"
            f"<td>{dur_delta}</td>"
            f"<td>{link_html}{err_html}</td>"
            f"</tr>"
        )

    summary_dict = batch_result.to_dict() if hasattr(batch_result, "to_dict") else {}
    raw_json = html.escape(json.dumps(summary_dict, indent=2, allow_nan=False))

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AudioBench Suite Summary · {html.escape(getattr(batch_result, 'name', 'Suite'))}</title>
<style>
:root {{ color-scheme: light dark; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
body {{ max-width: 1100px; margin: 0 auto; padding: 28px 20px 56px; line-height: 1.5; }}
header {{ display:flex; align-items:flex-start; justify-content:space-between; gap:20px; flex-wrap:wrap; }}
h1 {{ margin:0 0 4px; font-size: clamp(1.8rem, 4vw, 2.5rem); }}
.sub {{ color: #64748b; margin:0; }}
section {{ margin-top:28px; }}
h2 {{ font-size:1.15rem; margin-bottom:12px; }}
.pill {{ display:inline-block; border-radius:999px; padding:4px 12px; font-size:.82rem; font-weight:800; letter-spacing:.04em; }}
.pass {{ background:#dcfce7; color:#166534; }}
.fail {{ background:#fee2e2; color:#991b1b; }}
.warn {{ background:#fef3c7; color:#92400e; }}
.error {{ background:#fecaca; color:#7f1d1d; }}
.card {{ border:1px solid #94a3b855; border-radius:14px; padding:16px; overflow:hidden; }}
.stats-grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(130px, 1fr)); gap:12px; margin-top:16px; }}
.stat-box {{ border:1px solid #94a3b844; border-radius:10px; padding:12px; text-align:center; }}
.stat-val {{ font-size:1.6rem; font-weight:700; margin-top:4px; }}
.stat-val.pass {{ color:#16a34a; background:transparent; }}
.stat-val.warn {{ color:#d97706; background:transparent; }}
.stat-val.fail {{ color:#dc2626; background:transparent; }}
.stat-val.error {{ color:#991b1b; background:transparent; }}
table {{ width:100%; border-collapse:collapse; }}
th,td {{ text-align:left; padding:10px 10px; border-bottom:1px solid #94a3b844; vertical-align:top; font-size:.9rem; }}
th {{ font-weight:650; color:#475569; }}
tbody tr:last-child td {{ border-bottom:0; }}
.muted {{ color:#64748b; }}
.small {{ font-size:.8rem; margin-top:2px; }}
.error-msg {{ color:#dc2626; font-size:.8rem; margin-top:4px; }}
a {{ color:#2563eb; text-decoration:none; font-weight:550; }}
a:hover {{ text-decoration:underline; }}
pre {{ white-space:pre-wrap; overflow-wrap:anywhere; font-size:.8rem; max-height:440px; overflow:auto; }}
@media (max-width:700px) {{ body {{ padding:18px 12px 36px; }} table {{ display:block; overflow-x:auto; }} }}
</style>
</head>
<body>
<header>
  <div>
    <h1>AudioBench Suite</h1>
    <p class="sub">Regression test suite · {html.escape(getattr(batch_result, 'name', 'Suite'))}</p>
  </div>
  <span class="pill {status_class}">{html.escape(status)}</span>
</header>

<div class="stats-grid">
  <div class="stat-box"><div class="muted small">Total Tests</div><div class="stat-val">{getattr(batch_result, 'total_tests', 0)}</div></div>
  <div class="stat-box"><div class="muted small">Passed</div><div class="stat-val pass">{getattr(batch_result, 'passed', 0)}</div></div>
  <div class="stat-box"><div class="muted small">Warnings</div><div class="stat-val warn">{getattr(batch_result, 'warned', 0)}</div></div>
  <div class="stat-box"><div class="muted small">Failed</div><div class="stat-val fail">{getattr(batch_result, 'failed', 0)}</div></div>
  <div class="stat-box"><div class="muted small">Errors</div><div class="stat-val error">{getattr(batch_result, 'errored', 0)}</div></div>
</div>

<section>
  <h2>Test Cases</h2>
  <div class="card">
    <table>
      <thead>
        <tr>
          <th>Test Case</th>
          <th>Status</th>
          <th>SNR</th>
          <th>Est. Delay</th>
          <th>RMS Error</th>
          <th>Duration Delta</th>
          <th>Artifacts</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
  </div>
</section>

<section>
  <h2>Suite Machine-Readable Summary</h2>
  <div class="card">
    <pre>{raw_json}</pre>
  </div>
</section>

<section>
  <p class="muted small">Generated by AudioBench. Offline DSP regression test suite summary.</p>
</section>
</body>
</html>"""
    out.write_text(doc, encoding="utf-8")
    return out
