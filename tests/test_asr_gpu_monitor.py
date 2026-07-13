import json

import pytest

from scripts import asr_gpu_monitor


def record_sequence(tmp_path, outcomes, timestamps):
    remaining = iter(outcomes)
    observed = []
    clock = iter(timestamps)

    def sample_memory(_gpu_index):
        outcome = next(remaining)
        observed.append(outcome)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    asr_gpu_monitor.record_samples(
        tmp_path,
        gpu_index=0,
        interval_seconds=0.25,
        sample_memory=sample_memory,
        monotonic=lambda: next(clock),
        sleep=lambda _seconds: None,
        should_stop=lambda: len(observed) == len(outcomes),
    )
    return json.loads((tmp_path / "summary.json").read_text())


def test_gpu_monitor_rejects_one_success_followed_by_failures(tmp_path):
    summary = record_sequence(
        tmp_path,
        [100, RuntimeError("query failed"), RuntimeError("query failed")],
        [0.0, 0.0, 0.25, 0.5, 0.75],
    )

    assert summary["attempts"] == 3
    assert summary["successes"] == 1
    assert summary["failures"] == 2
    assert summary["failure_times"] == [0.25, 0.5]
    with pytest.raises(asr_gpu_monitor.MonitorValidationError, match="2 sampling failures"):
        asr_gpu_monitor.validate_summary(
            summary,
            maximum_memory_mib=1000,
            minimum_samples=4,
            minimum_span_seconds=0.75,
        )


def test_gpu_monitor_rejects_insufficient_sampling_coverage(tmp_path):
    summary = record_sequence(tmp_path, [100], [0.0, 0.0, 0.25])

    assert summary["successes"] == 1
    assert summary["sample_span_seconds"] == 0.0
    with pytest.raises(asr_gpu_monitor.MonitorValidationError, match="valid samples"):
        asr_gpu_monitor.validate_summary(
            summary,
            maximum_memory_mib=1000,
            minimum_samples=4,
            minimum_span_seconds=0.75,
        )


def test_gpu_monitor_accepts_four_clean_samples_spanning_point_75_seconds(tmp_path):
    summary = record_sequence(
        tmp_path,
        [100, 200, 300, 250],
        [0.0, 0.0, 0.25, 0.5, 0.75, 1.0],
    )

    asr_gpu_monitor.validate_summary(
        summary,
        maximum_memory_mib=1000,
        minimum_samples=4,
        minimum_span_seconds=0.75,
    )
    assert summary["failures"] == 0
    assert summary["sample_span_seconds"] == 0.75
    assert summary["maximum_memory_mib"] == 300
