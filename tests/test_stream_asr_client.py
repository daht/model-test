from scripts import stream_asr_client


def test_display_state_appends_sentence_final_and_replaces_partial():
    state = stream_asr_client.DisplayState()

    assert state.apply({"type": "partial", "text": "可以到店"}) == "可以到店"
    assert state.apply({"type": "partial", "text": "可以到店使用"}) == "可以到店使用"
    assert state.apply({"type": "sentence_final", "text": "可以到店"}) == "可以到店"
    assert state.apply({"type": "partial", "text": "使用也可以打包"}) == "可以到店使用也可以打包"


def test_display_state_final_uses_remaining_tail():
    state = stream_asr_client.DisplayState()
    state.apply({"type": "sentence_final", "text": "hello"})

    assert state.apply({"type": "final", "text": " world"}) == "hello world"


def test_empty_partial_after_commit_does_not_duplicate_display():
    state = stream_asr_client.DisplayState()

    assert state.apply({"type": "sentence_final", "text": "hello"}) == "hello"
    assert state.apply({"type": "partial", "text": ""}) == "hello"


def test_sequence_tracker_reports_gap_and_non_increasing_sequence():
    tracker = stream_asr_client.SequenceTracker()

    assert tracker.observe({"sequence": 1}) is None
    assert tracker.observe({"sequence": 3}) == "server event sequence gap: expected 2, got 3"
    assert tracker.observe({"sequence": 3}) == "server event sequence is not increasing: 3 after 3"


def test_error_payload_is_not_treated_as_transcript():
    state = stream_asr_client.DisplayState()

    assert state.apply({"type": "error", "code": "server_busy", "text": "secret"}) is None
