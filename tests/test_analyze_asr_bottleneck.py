import json
from pathlib import Path

from scripts.analyze_asr_bottleneck import analyze_run, write_reports


def write_jsonl(path: Path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_analyzer_correlates_buffer_rejection_with_engine_tail(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "metadata.json").write_text(json.dumps({"run_id": "run-1"}))
    write_jsonl(
        run / "events.jsonl",
        [
            {"timestamp": "2026-07-16T15:30:00.000Z", "event": "asr_session_opened", "session_id": "s"},
            {"timestamp": "2026-07-16T15:30:05.000Z", "event": "asr_batch_dispatched", "batch_id": "b", "selected_jobs": 5},
            {"timestamp": "2026-07-16T15:30:10.000Z", "event": "asr_engine_group_completed", "batch_id": "b", "engine_call_id": "e", "elapsed_seconds": 4.8, "group_size": 3},
            {"timestamp": "2026-07-16T15:30:12.000Z", "event": "asr_buffer_rejected", "session_id": "s", "reason": "session_pcm_limit", "current": 95_872, "incoming": 3_200, "limit": 96_000},
            {"timestamp": "2026-07-16T15:30:13.000Z", "event": "asr_session_terminal", "session_id": "s", "terminal_state": "failed"},
            {"timestamp": "2026-07-16T15:30:14.000Z", "event": "asr_session_released", "session_id": "s", "buffered_samples": 0, "reserved_samples": 0},
        ],
    )
    (run / "gpu.csv").write_text(
        "sampled_at,index,name,gpu_util_percent\n2026-07-16T15:30:10.000Z,0,A10,97\n"
    )

    report = analyze_run(run)

    assert report["quality"]["warnings"] == []
    assert report["failures"]["buffer_rejections"][0]["reason"] == "session_pcm_limit"
    assert report["failures"]["buffer_rejections"][0]["preceding_engine_call"]["engine_call_id"] == "e"
    assert report["engine"]["slowest_calls"][0]["elapsed_seconds"] == 4.8
    assert report["resources"]["gpu"]["max_utilization_percent"] == 97
    assert report["failures"]["buffer_rejections"][0]["nearest_gpu_sample"]["gpu_util_percent"] == "97"
    assert report["lifecycle"]["incomplete_sessions"] == []


def test_analyzer_reports_missing_evidence_and_writes_both_reports(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "metadata.json").write_text(json.dumps({"run_id": "run-2"}))

    report = analyze_run(run)
    write_reports(run, report)

    assert "events.jsonl is missing or empty" in report["quality"]["warnings"]
    assert (run / "report.json").is_file()
    assert "ASR Bottleneck Report" in (run / "report.md").read_text()


def test_analyzer_reports_safe_engine_group_failure_evidence(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    write_jsonl(
        run / "events.jsonl",
        [{
            "timestamp": "2026-07-18T12:00:00.000Z",
            "event": "asr_engine_group_failed",
            "batch_id": "batch-safe",
            "group_ordinal": 1,
            "group_count": 1,
            "group_size": 8,
            "final_items": 0,
            "accumulated_audio_seconds": 12.5,
            "min_input_audio_seconds": 1.5,
            "max_input_audio_seconds": 1.625,
            "failure_stage": "engine_generate",
            "exception_type": "RuntimeError",
            "exception_message": "private-engine-detail",
        }],
    )

    report = analyze_run(run)
    write_reports(run, report)

    assert report["engine"]["failures"] == [{
        "timestamp": "2026-07-18T12:00:00.000Z",
        "batch_id": "batch-safe",
        "group_ordinal": 1,
        "group_count": 1,
        "group_size": 8,
        "final_items": 0,
        "accumulated_audio_seconds": 12.5,
        "min_input_audio_seconds": 1.5,
        "max_input_audio_seconds": 1.625,
        "failure_stage": "engine_generate",
        "exception_type": "RuntimeError",
    }]
    markdown = (run / "report.md").read_text()
    assert "engine_generate" in markdown
    assert "RuntimeError" in markdown
    assert "batch-safe" in markdown
    assert "private-engine-detail" not in json.dumps(report)
