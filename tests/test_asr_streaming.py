from array import array

import pytest

from app.asr_streaming import (
    ConfirmedPrefixConflict,
    SilenceEndpointDetector,
    StreamingTranscriptState,
    first_punctuation_candidate,
)


def event_pairs(events):
    return [(event.type, event.text) for event in events]


def new_state(**overrides):
    values = {
        "sample_rate": 16000,
        "stable_commit_enabled": True,
        "stable_commit_seconds": 1.0,
        "stable_commit_min_chars": 8,
        "stable_commit_min_updates": 2,
    }
    values.update(overrides)
    return StreamingTranscriptState(**values)


def pcm(value: int, samples: int) -> bytes:
    return array("h", [value] * samples).tobytes()


def test_partial_replaces_previous_partial():
    state = new_state(stable_commit_enabled=False)

    assert event_pairs(state.apply_model_update("hello", processed_samples=1600)) == [
        ("partial", "hello")
    ]
    assert event_pairs(state.apply_model_update("hello world", processed_samples=1600)) == [
        ("partial", "hello world")
    ]


def test_stable_commit_uses_processed_audio_time_and_emits_empty_partial():
    state = new_state()
    text = "这是一个足够长的稳定句子。"

    state.apply_model_update(text, processed_samples=0)
    assert state.apply_model_update(text, processed_samples=8000) == []
    events = state.apply_model_update(text, processed_samples=8000)

    assert event_pairs(events) == [
        ("sentence_final", text),
        ("partial", ""),
    ]
    assert state.confirmed_text == text


def test_wall_clock_delay_without_processed_audio_does_not_commit():
    state = new_state()
    text = "这是一个足够长的稳定句子。"

    state.apply_model_update(text, processed_samples=0)
    assert state.apply_model_update(text, processed_samples=0) == []


def test_vad_commit_emits_sentence_and_empty_partial():
    state = new_state(stable_commit_enabled=False)
    state.apply_model_update("hello world", processed_samples=1600)

    assert event_pairs(state.commit_pending()) == [
        ("sentence_final", "hello world"),
        ("partial", ""),
    ]


def test_confirmed_prefix_conflict_never_reemits_full_model_text():
    state = new_state(stable_commit_enabled=False)
    state.apply_model_update("confirmed", processed_samples=1600)
    state.commit_pending()

    with pytest.raises(ConfirmedPrefixConflict):
        state.apply_model_update("unsafe revision", processed_samples=1600)


def test_confirmed_suffix_overlap_derives_only_safe_continuation():
    state = new_state(stable_commit_enabled=False)
    state.apply_model_update("hello", processed_samples=1600)
    state.commit_pending()

    assert event_pairs(state.apply_model_update("lo world", processed_samples=1600)) == [
        ("partial", " world")
    ]


def test_finish_emits_only_remaining_tail():
    state = new_state(stable_commit_enabled=False)
    state.apply_model_update("hello", processed_samples=1600)

    assert event_pairs(state.finish("hello world")) == [
        ("partial", "hello world"),
        ("final", "hello world"),
    ]


def test_sequences_increase_across_all_events():
    state = new_state(stable_commit_enabled=False)
    first = state.apply_model_update("hello", processed_samples=1600)
    second = state.commit_pending()

    assert [event.sequence for event in first + second] == [1, 2, 3]


def test_protocol_events_and_transcript_events_share_one_sequence():
    state = new_state(stable_commit_enabled=False)

    ready = state.new_event("ready")
    partial = state.apply_model_update("hello", processed_samples=1600)

    assert ready.sequence == 1
    assert partial[0].sequence == 2


def test_transient_or_removed_punctuation_resets_candidate():
    state = new_state(stable_commit_seconds=0.5)
    punctuated = "这是一个足够长的临时句子。"

    state.apply_model_update(punctuated, processed_samples=0)
    state.apply_model_update("这是一个足够长的临时句子", processed_samples=8000)
    events = state.apply_model_update(punctuated, processed_samples=8000)

    assert event_pairs(events) == [("partial", punctuated)]


@pytest.mark.parametrize(
    "text",
    [
        "Dr. Smith is here",
        "value is 3.14 today",
        "visit example.com today",
        "U.S.A. policy changed",
    ],
)
def test_nonterminal_dots_do_not_form_candidates(text):
    assert first_punctuation_candidate(text, min_chars=1) == ""


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('他说：“可以。”然后离开', '他说：“可以。”'),
        ("hello?! next", "hello?!"),
        ("完成。 next", "完成。"),
        ("تم؟ next", "تم؟"),
    ],
)
def test_candidate_includes_closers_and_mixed_terminators(text, expected):
    assert first_punctuation_candidate(text, min_chars=1) == expected


def test_empty_model_revision_clears_partial():
    state = new_state(stable_commit_enabled=False)
    state.apply_model_update("hello", processed_samples=1600)

    assert event_pairs(state.apply_model_update("", processed_samples=1600)) == [
        ("partial", "")
    ]


def test_silence_detector_triggers_once_until_speech_resets_it():
    detector = SilenceEndpointDetector(silence_seconds=0.1, rms_threshold=200)

    assert detector.add_audio(pcm(0, 800), 16000) is False
    assert detector.add_audio(pcm(0, 800), 16000) is True
    assert detector.add_audio(pcm(0, 1600), 16000) is False
    assert detector.add_audio(pcm(1000, 160), 16000) is False
    assert detector.add_audio(pcm(0, 1600), 16000) is True
