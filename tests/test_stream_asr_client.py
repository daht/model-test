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
