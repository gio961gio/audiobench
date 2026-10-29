import json
from pathlib import Path

import numpy as np
import pytest

from audiobench.cli import main
from audiobench.history import (
    BenchmarkRecord,
    HistoryError,
    append_history,
    compare_benchmarks,
    generate_history_html,
    get_default_history_file,
    get_git_metadata,
    load_history,
    resolve_baseline,
)


def _make_dummy_summary(
    suite_name: str = "demo-suite",
    status: str = "PASS",
    snr_db: float = 80.0,
    delay_ms: float = 0.0,
    commit: str | None = "a1b2c3d4e5",
) -> dict:
    return {
        "tool": "AudioBench",
        "version": "0.1.0",
        "name": suite_name,
        "status": status,
        "summary": {"total": 1, "passed": 1 if status == "PASS" else 0, "failed": 0, "warned": 0, "errored": 0},
        "tests": [
            {
                "id": "sine-test",
                "status": status,
                "metrics": {
                    "snr_db": snr_db,
                    "estimated_delay_ms": delay_ms,
                    "actual_true_peak_dbtp": -6.0,
                    "thd_db": -60.0,
                },
            }
        ],
    }


# -----------------------------------------------------------------------------
# 1. BenchmarkRecord & Storage (append_history / load_history)
# -----------------------------------------------------------------------------


def test_benchmark_record_creation_and_roundtrip(tmp_path):
    summary_data = _make_dummy_summary()
    rec = BenchmarkRecord.from_summary_data(summary_data, repo_dir=tmp_path)
    assert rec.suite_name == "demo-suite"
    assert rec.status == "PASS"
    assert "sine-test" in rec.tests

    as_dict = rec.to_dict()
    restored = BenchmarkRecord.from_dict(as_dict)
    assert restored.suite_name == rec.suite_name
    assert restored.tests["sine-test"]["metrics"]["snr_db"] == 80.0


def test_history_file_append_and_load(tmp_path):
    h_file = tmp_path / "history.jsonl"
    assert load_history(h_file) == []

    rec1 = BenchmarkRecord.from_summary_data(_make_dummy_summary(snr_db=80.0), repo_dir=tmp_path)
    rec2 = BenchmarkRecord.from_summary_data(_make_dummy_summary(snr_db=85.0), repo_dir=tmp_path)

    append_history(rec1, h_file)
    append_history(rec2, h_file)

    loaded = load_history(h_file)
    assert len(loaded) == 2
    assert loaded[0].tests["sine-test"]["metrics"]["snr_db"] == 80.0
    assert loaded[1].tests["sine-test"]["metrics"]["snr_db"] == 85.0


def test_history_corrupt_file(tmp_path):
    h_file = tmp_path / "bad_history.jsonl"
    h_file.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(HistoryError, match="Corrupt JSON at line 1"):
        load_history(h_file)


# -----------------------------------------------------------------------------
# 2. Baseline Resolution
# -----------------------------------------------------------------------------


def test_resolve_baseline_file(tmp_path):
    sum_file = tmp_path / "summary.json"
    sum_file.write_text(json.dumps(_make_dummy_summary(snr_db=90.0)), encoding="utf-8")

    rec = resolve_baseline(str(sum_file), base_dir=tmp_path)
    assert rec.tests["sine-test"]["metrics"]["snr_db"] == 90.0


def test_resolve_baseline_keywords_and_commits(tmp_path):
    h_file = tmp_path / "history.jsonl"
    rec1 = BenchmarkRecord.from_summary_data(_make_dummy_summary(snr_db=70.0), repo_dir=tmp_path)
    rec1.commit_hash = "1111111111"
    rec2 = BenchmarkRecord.from_summary_data(_make_dummy_summary(snr_db=75.0), repo_dir=tmp_path)
    rec2.commit_hash = "2222222222"

    append_history(rec1, h_file)
    append_history(rec2, h_file)

    latest = resolve_baseline("latest", history_file=h_file)
    assert latest.commit_hash == "2222222222"

    prev = resolve_baseline("head~1", history_file=h_file)
    assert prev.commit_hash == "1111111111"

    by_commit = resolve_baseline("1111", history_file=h_file)
    assert by_commit.commit_hash == "1111111111"

    with pytest.raises(HistoryError, match="could not be resolved"):
        resolve_baseline("nonexistent_commit", history_file=h_file)


# -----------------------------------------------------------------------------
# 3. compare_benchmarks (Regression Detection)
# -----------------------------------------------------------------------------


def test_compare_benchmarks_regression_snr_drop(tmp_path):
    base_data = _make_dummy_summary(snr_db=80.0)
    curr_data = _make_dummy_summary(snr_db=75.0)  # Dropped by 5 dB (> 1.0 dB default tolerance)

    base_rec = BenchmarkRecord.from_summary_data(base_data)
    curr_rec = BenchmarkRecord.from_summary_data(curr_data)

    res = compare_benchmarks(base_rec, curr_rec)
    assert res.is_regression is True
    assert len(res.regressions) == 1
    assert res.regressions[0].metric == "snr_db"
    assert "dropped by 5.00 dB" in res.regressions[0].message


def test_compare_benchmarks_regression_delay_growth(tmp_path):
    base_data = _make_dummy_summary(delay_ms=0.0)
    curr_data = _make_dummy_summary(delay_ms=0.2)  # Increased by 0.2 ms (> 0.05 ms tolerance)

    base_rec = BenchmarkRecord.from_summary_data(base_data)
    curr_rec = BenchmarkRecord.from_summary_data(curr_data)

    res = compare_benchmarks(base_rec, curr_rec)
    assert res.is_regression is True
    assert any(r.metric == "estimated_delay_ms" for r in res.regressions)


def test_compare_benchmarks_status_regression(tmp_path):
    base_data = _make_dummy_summary(status="PASS")
    curr_data = _make_dummy_summary(status="FAIL")
    curr_data["tests"][0]["status"] = "FAIL"

    base_rec = BenchmarkRecord.from_summary_data(base_data)
    curr_rec = BenchmarkRecord.from_summary_data(curr_data)

    res = compare_benchmarks(base_rec, curr_rec)
    assert res.is_regression is True
    assert "sine-test" in res.status_changes
    assert res.status_changes["sine-test"]["baseline"] == "PASS"
    assert res.status_changes["sine-test"]["current"] == "FAIL"


def test_compare_benchmarks_added_and_removed_tests(tmp_path):
    base_data = _make_dummy_summary()
    base_data["tests"] = [{"id": "t1", "status": "PASS", "metrics": {"snr_db": 80.0}}]

    curr_data = _make_dummy_summary()
    curr_data["tests"] = [{"id": "t2", "status": "PASS", "metrics": {"snr_db": 80.0}}]

    base_rec = BenchmarkRecord.from_summary_data(base_data)
    curr_rec = BenchmarkRecord.from_summary_data(curr_data)

    res = compare_benchmarks(base_rec, curr_rec)
    assert res.is_regression is False
    assert res.added_tests == ["t2"]
    assert res.removed_tests == ["t1"]


def test_compare_benchmarks_infinite_snr(tmp_path):
    # Both identical infinity
    base_data = _make_dummy_summary(snr_db=float("inf"))
    curr_data = _make_dummy_summary(snr_db=float("inf"))

    base_rec = BenchmarkRecord.from_summary_data(base_data)
    curr_rec = BenchmarkRecord.from_summary_data(curr_data)

    res = compare_benchmarks(base_rec, curr_rec)
    assert res.is_regression is False
    assert len(res.regressions) == 0


# -----------------------------------------------------------------------------
# 4. generate_history_html Dashboard
# -----------------------------------------------------------------------------


def test_generate_history_html(tmp_path):
    out_html = tmp_path / "dashboard.html"

    rec1 = BenchmarkRecord.from_summary_data(_make_dummy_summary(snr_db=80.0, delay_ms=0.01))
    rec1.commit_hash = "abc1111"
    rec2 = BenchmarkRecord.from_summary_data(_make_dummy_summary(snr_db=82.0, delay_ms=0.01))
    rec2.commit_hash = "abc2222"

    res_path = generate_history_html([rec1, rec2], out_html)
    assert res_path == out_html
    assert out_html.exists()

    content = out_html.read_text(encoding="utf-8")
    assert "AudioBench Benchmark History" in content
    assert "sine-test" in content
    assert "<polyline" in content
    assert "<circle" in content
    assert "abc1111" in content


# -----------------------------------------------------------------------------
# 5. CLI history Commands
# -----------------------------------------------------------------------------


def test_cli_history_record_and_diff(tmp_path, capsys):
    h_file = tmp_path / "bench.jsonl"
    sum_v1 = tmp_path / "v1.json"
    sum_v2 = tmp_path / "v2.json"

    sum_v1.write_text(json.dumps(_make_dummy_summary(snr_db=80.0)), encoding="utf-8")
    sum_v2.write_text(json.dumps(_make_dummy_summary(snr_db=65.0)), encoding="utf-8")  # Large drop

    # Record v1
    code_rec = main(["history", "record", "--summary", str(sum_v1), "--history-file", str(h_file)])
    assert code_rec == 0
    captured = capsys.readouterr()
    assert "Recorded benchmark" in captured.out

    # Diff v1 vs v2 -> must detect regression (exit code 1)
    code_diff = main(
        [
            "history",
            "diff",
            "--baseline",
            str(sum_v1),
            "--current",
            str(sum_v2),
            "--history-file",
            str(h_file),
        ]
    )
    assert code_diff == 1
    captured_diff = capsys.readouterr()
    assert "REGRESSION DETECTED" in captured_diff.out
    assert "SNR dropped" in captured_diff.out

    # Report generation
    rep_html = tmp_path / "history_report.html"
    code_rep = main(["history", "report", "--history-file", str(h_file), "--output", str(rep_html)])
    assert code_rep == 0
    assert rep_html.exists()


def test_cli_batch_integration_with_history(tmp_path, capsys):
    # Test batch runner recording history automatically
    from audiobench.audio import write_audio

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    ref = audio_dir / "ref.wav"
    data = (np.sin(np.linspace(0, 10, 4800)) * 0.4)[:, None]
    write_audio(ref, data, 48000)

    manifest_file = tmp_path / "suite.json"
    manifest_data = {
        "name": "history-batch-suite",
        "tests": [{"id": "t1", "expected": "audio/ref.wav", "actual": "audio/ref.wav"}],
    }
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    h_file = tmp_path / "auto_history.jsonl"
    out_dir = tmp_path / "reports"

    # 1. Run batch with --record-history
    code = main(
        [
            "batch",
            "--manifest",
            str(manifest_file),
            "--output-dir",
            str(out_dir),
            "--record-history",
            "--history-file",
            str(h_file),
        ]
    )
    assert code == 0
    assert h_file.exists()
    records = load_history(h_file)
    assert len(records) == 1

    # 2. Run batch with --baseline latest
    code_base = main(
        [
            "batch",
            "--manifest",
            str(manifest_file),
            "--output-dir",
            str(out_dir),
            "--baseline",
            "latest",
            "--history-file",
            str(h_file),
        ]
    )
    assert code_base == 0
    captured = capsys.readouterr()
    assert "Baseline Check PASS" in captured.out
