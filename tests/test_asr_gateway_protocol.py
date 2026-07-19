import pytest

from app.asr_gateway_backends import ResultMode
from app.asr_gateway_protocol import ProtocolSession, parse_client_command
from app.asr_streaming import ConfirmedPrefixConflict


def test_start_and_control_validation():
    start = parse_client_command({"type": "start", "format": "pcm_s16le", "sample_rate": 16000, "channels": 1, "language": "ZH", "options": {"timestamps": False}})
    assert start.language == "zh"
    assert parse_client_command({"type": "segment"}).type == "segment"
    assert parse_client_command({"type": "finish"}).type == "finish"
    assert parse_client_command({"type": "abort"}).type == "abort"
    for bad in (
        {"type": "start", "format": "wav", "sample_rate": 16000, "channels": 1},
        {"type": "start", "format": "pcm_s16le", "sample_rate": 8000, "channels": 1},
        {"type": "start", "format": "pcm_s16le", "sample_rate": 16000, "channels": 2},
        {"type": "unknown"},
    ):
        with pytest.raises(ValueError): parse_client_command(bad)


def test_partial_replaces_tail_sentence_confirms_and_final_is_unique():
    protocol = ProtocolSession(sample_rate=16000)
    assert protocol.ready()["sequence"] == 1
    first = protocol.apply_result(ResultMode.CUMULATIVE_SNAPSHOT, text="hello", decoded_samples=10)
    second = protocol.apply_result(ResultMode.CUMULATIVE_SNAPSHOT, text="hello world", decoded_samples=10)
    confirmed = protocol.segment()
    final = protocol.final()
    assert first[-1]["text"] == "hello"
    assert second[-1]["text"] == "hello world"
    assert [e["type"] for e in confirmed] == ["sentence_final", "partial"]
    assert final["type"] == "final"
    sequences = [1] + [e["sequence"] for e in first + second + confirmed] + [final["sequence"]]
    assert sequences == list(range(1, len(sequences) + 1))
    with pytest.raises(RuntimeError, match="terminal"): protocol.final()
    with pytest.raises(RuntimeError, match="terminal"): protocol.ready()


def test_confirmed_prefix_is_immutable_and_error_is_sanitized():
    protocol = ProtocolSession(sample_rate=16000)
    protocol.apply_result(ResultMode.CUMULATIVE_SNAPSHOT, text="confirmed", decoded_samples=1)
    protocol.segment()
    with pytest.raises(ConfirmedPrefixConflict):
        protocol.apply_result(ResultMode.CUMULATIVE_SNAPSHOT, text="changed", decoded_samples=1)
    error = ProtocolSession(sample_rate=16000).error(RuntimeError("secret credential value"))
    assert error["code"] == "backend_error"
    assert "secret" not in error["message"]


def test_replaceable_and_confirmed_plus_tail_modes_reuse_transcript_state():
    segment = ProtocolSession(sample_rate=16000, segment_local=True)
    assert segment.apply_result(ResultMode.REPLACEABLE_SEGMENT, text="one", decoded_samples=2, segment_id=7)[0]["text"] == "one"
    combined = ProtocolSession(sample_rate=16000)
    events = combined.apply_result(ResultMode.CONFIRMED_PLUS_TAIL, confirmed_text="fixed", tail_text="tail", decoded_samples=3)
    assert [event["type"] for event in events] == ["sentence_final", "partial", "partial"]
    assert combined.state.confirmed_text == "fixed"


def test_optional_result_metadata_is_attached_without_changing_plain_events():
    protocol = ProtocolSession(sample_rate=16000, segment_local=True)
    plain = protocol.apply_result(
        ResultMode.REPLACEABLE_SEGMENT,
        text="plain",
        decoded_samples=2,
        segment_id=1,
    )
    rich = protocol.apply_result(
        ResultMode.REPLACEABLE_SEGMENT,
        text="rich",
        decoded_samples=2,
        segment_id=1,
        metadata={"language": "zh", "emotion": "neutral", "audio_event": "speech"},
    )
    committed = protocol.segment(metadata={"language": "zh"})
    final = protocol.final(metadata={"language": "zh"})

    assert all("metadata" not in event for event in plain)
    assert rich[-1]["metadata"]["emotion"] == "neutral"
    assert committed[0]["metadata"] == {"language": "zh"}
    assert final["metadata"] == {"language": "zh"}
