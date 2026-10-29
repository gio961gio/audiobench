"""AudioBench Web Studio Dashboard & A/B Differential Listening Player."""

from __future__ import annotations

import html
import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import streamlit as st

from audiobench import __version__
from audiobench.audio import load_audio
from audiobench.history import get_default_history_file, load_history
from audiobench.metrics import Thresholds, compare_arrays, evaluate_thresholds


# -----------------------------------------------------------------------------
# 1. Styling & Audio Helpers
# -----------------------------------------------------------------------------

def _inject_studio_css() -> None:
    st.markdown(
        """
        <style>
        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }
        div[data-testid="stMetricValue"] {
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
            font-feature-settings: "tnum" !important;
        }
        .studio-card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 16px;
        }
        .pill {
            display: inline-block;
            border-radius: 999px;
            padding: 3px 12px;
            font-size: 0.8rem;
            font-weight: 750;
            letter-spacing: 0.04em;
        }
        .pill-pass { background: #dcfce7; color: #166534; }
        .pill-fail { background: #fee2e2; color: #991b1b; }
        .pill-warn { background: #fef3c7; color: #92400e; }
        .pill-error { background: #fecaca; color: #7f1d1d; }
        .safety-badge {
            background: #451a03;
            border: 1px solid #f59e0b;
            color: #fef3c7;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 0.85rem;
            margin-top: 8px;
            display: inline-block;
        }
        .player-label {
            font-weight: 650;
            font-size: 0.95rem;
            margin-bottom: 4px;
        }
        .muted { color: #8b949e; font-size: 0.85rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _array_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode float64 audio into 16-bit PCM WAV bytes without Streamlit auto-normalization."""
    arr = np.asarray(audio, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    # Clamp to [-1.0, 1.0] before 16-bit PCM integer quantization
    clamped = np.clip(arr, -1.0, 1.0)
    buf = io.BytesIO()
    sf.write(buf, clamped, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _compute_difference_signal(
    expected: np.ndarray,
    actual: np.ndarray,
    sample_rate: int,
    align_delay_samples: int = 0,
    boost_db: float = 24.0,
) -> tuple[np.ndarray, float, bool]:
    """Compute difference residual with optional latency compensation and audio safety limiter.

    Ceiling is placed at -0.5 dBFS (~0.944) to prevent ear/monitor damage on anti-phase signals.
    """
    # Channel alignment
    exp = expected if expected.ndim == 2 else expected[:, None]
    act = actual if actual.ndim == 2 else actual[:, None]

    if exp.shape[1] != act.shape[1]:
        # Fold down to mono if channel counts disagree
        exp = np.mean(exp, axis=1, keepdims=True)
        act = np.mean(act, axis=1, keepdims=True)

    # Optional latency compensation on actual
    if align_delay_samples != 0:
        shifted = np.zeros_like(act)
        if align_delay_samples > 0 and align_delay_samples < len(act):
            shifted[:-align_delay_samples] = act[align_delay_samples:]
            act = shifted
        elif align_delay_samples < 0 and abs(align_delay_samples) < len(act):
            offset = abs(align_delay_samples)
            shifted[offset:] = act[:-offset]
            act = shifted

    common_len = min(len(exp), len(act))
    diff = act[:common_len] - exp[:common_len]

    peak = float(np.max(np.abs(diff))) if diff.size > 0 else 0.0
    desired_gain = 10.0 ** (boost_db / 20.0)
    safety_ceiling = 0.94406  # -0.5 dBFS

    safety_limited = False
    effective_gain = desired_gain

    if peak * desired_gain > safety_ceiling and peak > 0.0:
        effective_gain = safety_ceiling / peak
        effective_boost_db = 20.0 * math.log10(max(effective_gain, 1e-9))
        safety_limited = True
    else:
        effective_boost_db = boost_db

    boosted = np.clip(diff * effective_gain, -0.99, 0.99)
    return boosted, effective_boost_db, safety_limited


# -----------------------------------------------------------------------------
# 2. Cached Loaders
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False, max_entries=32)
def _cached_load_audio(path_str: str) -> tuple[np.ndarray, int]:
    return load_audio(path_str)


@st.cache_data(show_spinner=False, max_entries=32)
def _cached_compare(exp_path: str, act_path: str) -> dict[str, Any]:
    exp_data, exp_sr = load_audio(exp_path)
    act_data, act_sr = load_audio(act_path)
    if exp_sr != act_sr:
        raise ValueError(f"Sample rate mismatch: {exp_sr} Hz vs {act_sr} Hz")
    metrics = compare_arrays(exp_data, act_data, exp_sr)
    limits = Thresholds()
    evaluation = evaluate_thresholds(metrics, limits)
    return {"metrics": metrics, "evaluation": evaluation, "sr": exp_sr}


# -----------------------------------------------------------------------------
# 3. Streamlit Application
# -----------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="AudioBench Studio Dashboard",
        page_icon="🎛️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_studio_css()

    st.title("🎛️ AudioBench Studio")
    st.caption(f"Offline Audio Render Regression Benchmark & A/B Differential Suite · v{__version__}")

    tab_suite, tab_history, tab_sandbox = st.tabs(
        [
            "📊 Suite & A/B Differential Player",
            "📈 Benchmark Trend History",
            "🧪 Live A/B Sandbox",
        ]
    )

    # -------------------------------------------------------------------------
    # TAB 1: Suite & A/B Differential Player
    # -------------------------------------------------------------------------
    with tab_suite:
        demo_summary = Path("examples/demo/batch/summary.json").resolve()
        default_path = str(demo_summary) if demo_summary.exists() else "reports/batch/summary.json"

        with st.sidebar:
            st.header("⚙️ Suite Source")
            summary_path_input = st.text_input("Summary JSON Path", value=default_path)
            uploaded_summary = st.file_uploader("Or Upload summary.json", type=["json"])

        summary_data: dict[str, Any] | None = None
        if uploaded_summary is not None:
            try:
                summary_data = json.loads(uploaded_summary.getvalue().decode("utf-8"))
            except Exception as e:
                st.error(f"Cannot parse uploaded JSON: {e}")
        elif summary_path_input and Path(summary_path_input).exists():
            try:
                summary_data = json.loads(Path(summary_path_input).read_text(encoding="utf-8"))
            except Exception as e:
                st.error(f"Cannot read summary file '{summary_path_input}': {e}")

        if summary_data is None:
            st.info("No summary loaded. Run `audiobench batch --manifest ...` or run `python scripts/create_demo.py` to generate one.")
        else:
            # Top KPI metrics
            st.markdown(f"### Suite: **{summary_data.get('name', 'Regression Suite')}**")
            kpi_cols = st.columns(5)
            summary_counts = summary_data.get("summary", {})
            st_overall = summary_data.get("status", "UNKNOWN")

            kpi_cols[0].metric("Overall Status", st_overall)
            kpi_cols[1].metric("Total Tests", summary_counts.get("total", 0))
            kpi_cols[2].metric("Passed", summary_counts.get("passed", 0))
            kpi_cols[3].metric("Warnings", summary_counts.get("warned", 0))
            kpi_cols[4].metric("Failed / Errors", summary_counts.get("failed", 0) + summary_counts.get("errored", 0))

            tests_list = summary_data.get("tests", [])
            if not tests_list:
                st.warning("No tests found in summary.")
            else:
                test_options = {t["id"]: t for t in tests_list}
                selected_test_id = st.selectbox(
                    "Select Test Case to Inspect & Audition:",
                    options=list(test_options.keys()),
                    format_func=lambda tid: f"{tid} — [{test_options[tid].get('status', 'UNKNOWN')}]",
                )
                selected_test = test_options[selected_test_id]
                case_status = selected_test.get("status", "UNKNOWN")
                badge_class = f"pill-{case_status.lower()}"

                st.markdown(
                    f"#### Case: <code>{html.escape(selected_test_id)}</code> "
                    f"<span class='pill {badge_class}'>{case_status}</span>",
                    unsafe_allow_html=True,
                )
                if selected_test.get("description"):
                    st.caption(selected_test["description"])

                exp_file = selected_test.get("expected_path")
                act_file = selected_test.get("actual_path")

                # Metrics overview card
                metrics = selected_test.get("metrics") or {}
                if metrics:
                    m_cols = st.columns(4)
                    snr_str = "∞ (identical)" if metrics.get("identical_samples") else f"{metrics.get('snr_db', 0.0):.2f} dB"
                    m_cols[0].metric("SNR vs Reference", snr_str)
                    m_cols[1].metric("Estimated Delay", f"{metrics.get('estimated_delay_ms', 0.0):.3f} ms")
                    m_cols[2].metric("RMS Error", f"{metrics.get('error_rms_dbfs', 0.0):.2f} dBFS")
                    tp_val = metrics.get("actual_true_peak_dbtp")
                    m_cols[3].metric("True-Peak", f"{tp_val:.2f} dBTP" if tp_val is not None else "—")

                # -------------------------------------------------------------
                # A/B Differential Listening Player
                # -------------------------------------------------------------
                st.markdown("### 🎧 A/B Differential Listening Player")
                st.markdown(
                    "<p class='muted'>Audition the Reference, Actual Render, and the amplified "
                    "<strong>Difference Signal (e(t) = y(t) - x(t))</strong> to detect artifacts, non-linear distortion, and aliasing.</p>",
                    unsafe_allow_html=True,
                )

                if not exp_file or not Path(exp_file).exists() or not act_file or not Path(act_file).exists():
                    st.warning(f"Audio files for this test are not directly available on disk:\n- Expected: `{exp_file}`\n- Actual: `{act_file}`")
                else:
                    try:
                        exp_data, sr_exp = _cached_load_audio(exp_file)
                        act_data, sr_act = _cached_load_audio(act_file)

                        # Player controls
                        ctrl_cols = st.columns([2, 2, 2])
                        with ctrl_cols[0]:
                            boost_db = st.slider("Difference Gain Boost (dB)", min_value=0.0, max_value=48.0, value=24.0, step=3.0)
                        with ctrl_cols[1]:
                            est_delay = metrics.get("estimated_delay_samples", 0)
                            compensate_delay = st.checkbox("Compensate Latency before Diff", value=False, help="Aligns signals by estimated delay to isolate non-linear distortion from phase comb-filtering")
                        with ctrl_cols[2]:
                            delay_samples_to_use = est_delay if compensate_delay else 0
                            st.write(f"Align offset: `{delay_samples_to_use}` samples")

                        diff_data, eff_boost, was_limited = _compute_difference_signal(
                            exp_data,
                            act_data,
                            sr_exp,
                            align_delay_samples=delay_samples_to_use,
                            boost_db=boost_db,
                        )

                        # Audio players in 3 columns
                        play_cols = st.columns(3)
                        with play_cols[0]:
                            st.markdown("<div class='player-label'>1. Expected (Reference)</div>", unsafe_allow_html=True)
                            st.audio(_array_to_wav_bytes(exp_data, sr_exp), format="audio/wav")

                        with play_cols[1]:
                            st.markdown("<div class='player-label'>2. Actual (Under Test)</div>", unsafe_allow_html=True)
                            st.audio(_array_to_wav_bytes(act_data, sr_act), format="audio/wav")

                        with play_cols[2]:
                            st.markdown(f"<div class='player-label'>3. Difference Residual ({eff_boost:+.1f} dB)</div>", unsafe_allow_html=True)
                            st.audio(_array_to_wav_bytes(diff_data, sr_exp), format="audio/wav")
                            if was_limited:
                                st.markdown(
                                    f"<div class='safety-badge'>⚠️ Safety Limiter Active: Clamped to {eff_boost:+.1f} dB (-0.5 dBFS ceiling)</div>",
                                    unsafe_allow_html=True,
                                )

                    except Exception as e:
                        st.error(f"Error loading audio players: {e}")

    # -------------------------------------------------------------------------
    # TAB 2: Historical Benchmark Trends
    # -------------------------------------------------------------------------
    with tab_history:
        st.markdown("### 📈 Longitudinal Benchmark History")
        h_file = Path("examples/demo/demo_history.jsonl")
        if not h_file.exists():
            h_file = get_default_history_file()

        custom_h_path = st.text_input("History File Path", value=str(h_file))
        records = load_history(custom_h_path)

        if not records:
            st.info(f"No benchmark records found in `{custom_h_path}`. Run `audiobench batch ... --record-history` to start tracking.")
        else:
            st.write(f"Total benchmark snapshots recorded: **{len(records)}**")
            all_tids: set[str] = set()
            for r in records:
                all_tids.update(r.tests.keys())

            hist_cols = st.columns(2)
            with hist_cols[0]:
                selected_hist_tid = st.selectbox("Select Test to View Trend:", options=sorted(all_tids))

            # Build trend line data
            snr_points = []
            labels = []
            for r in records:
                if selected_hist_tid in r.tests:
                    lbl = f"{r.commit_hash or r.timestamp[:10]}"
                    snr_val = r.tests[selected_hist_tid].get("metrics", {}).get("snr_db")
                    if snr_val is not None:
                        snr_points.append(150.0 if math.isinf(snr_val) else snr_val)
                        labels.append(lbl)

            if len(snr_points) > 1:
                st.line_chart(snr_points)
                st.caption(f"SNR (dB) progression across commits for `{selected_hist_tid}`")
            else:
                st.info("Need at least 2 runs in history to render a trend chart.")

            # Table of recent runs
            st.markdown("#### Recent Execution Runs")
            run_table = []
            for r in reversed(records[-10:]):
                run_table.append(
                    {
                        "Date (UTC)": r.timestamp[:19].replace("T", " "),
                        "Commit": r.commit_hash or "—",
                        "Dirty Tree": r.is_dirty,
                        "Status": r.status,
                        "Passed": f"{r.summary.get('passed', 0)} / {r.summary.get('total', 0)}",
                    }
                )
            st.dataframe(run_table, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 3: Live A/B Sandbox
    # -------------------------------------------------------------------------
    with tab_sandbox:
        st.markdown("### 🧪 Live A/B Comparison Sandbox")
        st.markdown("<p class='muted'>Upload two WAV files on the fly for immediate sample-accurate regression analysis and A/B difference listening.</p>", unsafe_allow_html=True)

        up_cols = st.columns(2)
        with up_cols[0]:
            up_exp = st.file_uploader("Expected (Reference) WAV", type=["wav"])
        with up_cols[1]:
            up_act = st.file_uploader("Actual (Render under test) WAV", type=["wav"])

        if up_exp and up_act:
            try:
                exp_data, sr_exp = sf.read(io.BytesIO(up_exp.getvalue()), dtype="float64", always_2d=True)
                act_data, sr_act = sf.read(io.BytesIO(up_act.getvalue()), dtype="float64", always_2d=True)

                if sr_exp != sr_act:
                    st.error(f"Sample rate mismatch: {sr_exp} Hz vs {sr_act} Hz. AudioBench avoids silent resampling.")
                else:
                    metrics = compare_arrays(exp_data, act_data, sr_exp)
                    eval_res = evaluate_thresholds(metrics, Thresholds())

                    res_cols = st.columns(4)
                    res_cols[0].metric("Status", eval_res["status"])
                    res_cols[1].metric("SNR vs Ref", "∞ (identical)" if metrics["identical_samples"] else f"{metrics['snr_db']:.2f} dB")
                    res_cols[2].metric("Estimated Delay", f"{metrics['estimated_delay_ms']:.3f} ms")
                    res_cols[3].metric("RMS Error", f"{metrics['error_rms_dbfs']:.2f} dBFS")

                    st.markdown("#### Audition Live Null Difference")
                    sb_boost = st.slider("Live Difference Gain Boost (dB)", 0.0, 48.0, 24.0, step=3.0, key="sb_boost")
                    sb_diff, sb_eff, sb_limited = _compute_difference_signal(exp_data, act_data, sr_exp, boost_db=sb_boost)

                    sb_play_cols = st.columns(3)
                    with sb_play_cols[0]:
                        st.caption("Reference")
                        st.audio(_array_to_wav_bytes(exp_data, sr_exp), format="audio/wav")
                    with sb_play_cols[1]:
                        st.caption("Actual")
                        st.audio(_array_to_wav_bytes(act_data, sr_act), format="audio/wav")
                    with sb_play_cols[2]:
                        st.caption(f"Difference ({sb_eff:+.1f} dB)")
                        st.audio(_array_to_wav_bytes(sb_diff, sr_exp), format="audio/wav")
                        if sb_limited:
                            st.warning(f"Safety ceiling clamped boost to {sb_eff:+.1f} dB")

            except Exception as e:
                st.error(f"Failed to analyze live audio: {e}")


if __name__ == "__main__":
    main()
