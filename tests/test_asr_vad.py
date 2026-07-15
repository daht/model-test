from __future__ import annotations

from collections import deque

import pytest

from app.asr_vad import (
    EnergyVADBackend,
    StreamingVADEndpointDetector,
    VADEndpointState,
    VADRuntimeError,
    create_vad_endpoint_detector,
)
from app.config import Settings


FRAME_SAMPLES = 512
SAMPLE_RATE = 16000


def pcm(value: int, samples: int = FRAME_SAMPLES) -> bytes:
    return int(value).to_bytes(2, "little", signed=True) * samples


class ScriptedVAD:
    def __init__(self, probabilities):
        self.probabilities = deque(probabilities)
        self.frames = []

    def speech_probability(self, frame):
        self.frames.append(frame)
        return self.probabilities.popleft()

    def reset(self):
        return None


def detector(probabilities, **overrides):
    values = {
        "backend": ScriptedVAD(probabilities),
        "sample_rate": SAMPLE_RATE,
        "frame_samples": FRAME_SAMPLES,
        "onset_threshold": 0.65,
        "offset_threshold": 0.35,
        "min_speech_ms": 96,
        "min_silence_ms": 96,
        "hangover_ms": 32,
        "pre_roll_ms": 200,
    }
    values.update(overrides)
    return StreamingVADEndpointDetector(**values)


def test_pure_silence_and_room_noise_never_reach_model_or_endpoint():
    vad = detector([0.01] * 12)

    decisions = [vad.add_audio(pcm(4000)) for _ in range(12)]

    assert all(decision.audio_to_model == b"" for decision in decisions)
    assert all(decision.endpoint is False for decision in decisions)
    assert sum(decision.discarded_samples for decision in decisions) > 0
    assert vad.state is VADEndpointState.WAITING_FOR_SPEECH


def test_onset_candidate_releases_full_candidate_and_exact_200ms_preroll():
    vad = detector([0.01] * 8 + [0.9, 0.8, 0.75])
    silence = [pcm(index + 1) for index in range(8)]
    speech = [pcm(100 + index) for index in range(3)]

    for frame in silence:
        assert vad.add_audio(frame).audio_to_model == b""
    assert vad.add_audio(speech[0]).audio_to_model == b""
    assert vad.add_audio(speech[1]).audio_to_model == b""
    confirmed = vad.add_audio(speech[2])

    expected_preroll = b"".join(silence)[-(SAMPLE_RATE * 200 // 1000) * 2 :]
    assert confirmed.audio_to_model == expected_preroll + b"".join(speech)
    assert confirmed.endpoint is False
    assert vad.state is VADEndpointState.IN_SPEECH


def test_short_speech_burst_is_retained_without_model_decode():
    vad = detector([0.9, 0.8, 0.1, 0.01, 0.01])

    decisions = [vad.add_audio(pcm(2000)) for _ in range(5)]

    assert all(decision.audio_to_model == b"" for decision in decisions)
    assert sum(decision.discarded_samples for decision in decisions) == 0
    assert vad.state is VADEndpointState.WAITING_FOR_SPEECH


def test_rejected_onset_transfers_preroll_without_duplicate_samples():
    vad = detector(
        [0.01, 0.9, 0.01, 0.9, 0.9, 0.9, 0.9],
        sample_rate=1000,
        frame_samples=1,
        min_speech_ms=4,
        pre_roll_ms=10,
    )

    decisions = [vad.add_audio(pcm(value, samples=1)) for value in range(1, 8)]
    released = b"".join(decision.audio_to_model for decision in decisions)
    released_values = [
        int.from_bytes(released[index : index + 2], "little", signed=True)
        for index in range(0, len(released), 2)
    ]

    assert released_values == [1, 2, 3, 4, 5, 6, 7]
    assert len(released_values) + sum(
        decision.discarded_samples for decision in decisions
    ) == 7


def test_endpoint_retains_only_silence_not_released_as_hangover():
    vad = detector(
        [0.9, 0.9, 0.1, 0.1, 0.1, 0.9, 0.9],
        sample_rate=1000,
        frame_samples=1,
        min_speech_ms=2,
        min_silence_ms=3,
        hangover_ms=1,
        pre_roll_ms=10,
    )
    decisions = []

    for value in range(1, 8):
        decision = vad.add_audio(pcm(value, samples=1))
        decisions.append(decision)
        if decision.endpoint:
            decisions.append(vad.endpoint_finalized())

    released = b"".join(decision.audio_to_model for decision in decisions)
    released_values = [
        int.from_bytes(released[index : index + 2], "little", signed=True)
        for index in range(0, len(released), 2)
    ]

    assert released_values == [1, 2, 3, 4, 5, 6, 7]
    assert len(released_values) + sum(
        decision.discarded_samples for decision in decisions
    ) == 7


def test_finish_input_discards_and_resets_entire_unconfirmed_candidate():
    vad = detector(
        [0.1, 0.9],
        sample_rate=1000,
        frame_samples=1,
        min_speech_ms=2,
        pre_roll_ms=10,
    )

    decisions = [
        vad.add_audio(pcm(1, samples=1)),
        vad.add_audio(pcm(2, samples=1)),
        vad.finish_input(),
    ]
    released_samples = sum(len(decision.audio_to_model) // 2 for decision in decisions)
    discarded_samples = sum(decision.discarded_samples for decision in decisions)
    retained_samples = sum(
        len(buffer) // 2
        for buffer in (
            vad._input_buffer,
            vad._pre_roll,
            vad._candidate,
            vad._trailing,
        )
    )

    assert released_samples == 0
    assert discarded_samples == 2
    assert retained_samples == 0
    assert released_samples + discarded_samples + retained_samples == 2
    assert vad._candidate_speech_samples == 0
    assert vad.state is VADEndpointState.WAITING_FOR_SPEECH


def test_trailing_silence_is_released_when_speech_resumes():
    vad = detector([0.9, 0.9, 0.9, 0.1, 0.1, 0.8])
    speech = [pcm(1000 + index) for index in range(3)]
    trailing = [pcm(0), pcm(0)]
    resumed = pcm(2000)

    for frame in speech:
        confirmed = vad.add_audio(frame)
    assert confirmed.audio_to_model == b"".join(speech)
    assert vad.add_audio(trailing[0]).audio_to_model == b""
    assert vad.add_audio(trailing[1]).audio_to_model == b""
    decision = vad.add_audio(resumed)

    assert decision.audio_to_model == b"".join(trailing) + resumed
    assert decision.endpoint is False
    assert vad.state is VADEndpointState.IN_SPEECH


def test_endpoint_sends_only_hangover_and_rearms_after_new_confirmed_speech():
    probabilities = (
        [0.9] * 3
        + [0.01] * 8
        + [0.9] * 3
        + [0.01] * 3
    )
    vad = detector(probabilities)
    endpoints = []
    endpoint_audio = []

    for probability_index in range(len(probabilities)):
        value = 1000 if probabilities[probability_index] > 0.5 else 0
        decision = vad.add_audio(pcm(value))
        if decision.endpoint:
            endpoints.append(probability_index)
            endpoint_audio.append(decision.audio_to_model)
            assert vad.state is VADEndpointState.FINALIZING
            vad.endpoint_finalized()

    assert endpoints == [5, 16]
    assert endpoint_audio == [pcm(0), pcm(0)]
    assert vad.state is VADEndpointState.WAITING_FOR_SPEECH


def test_offset_hysteresis_keeps_mid_probability_audio_inside_speech():
    vad = detector([0.9, 0.8, 0.7, 0.5, 0.4, 0.2, 0.2, 0.2])

    decisions = [vad.add_audio(pcm(1200)) for _ in range(8)]

    assert decisions[3].audio_to_model == pcm(1200)
    assert decisions[4].audio_to_model == pcm(1200)
    assert decisions[-1].endpoint is True


def test_explicit_mock_backend_uses_test_vad_without_onnx_asset():
    vad = create_vad_endpoint_detector(
        Settings(_env_file=None, asr_backend="mock", asr_stream_mode="stateful")
    )

    assert isinstance(vad.backend, EnergyVADBackend)


def test_production_stateful_backend_fails_fast_when_vad_asset_is_missing(tmp_path):
    settings = Settings(
        _env_file=None,
        asr_backend="qwen_vllm",
        asr_stream_mode="stateful",
        asr_model_id="Qwen/Qwen3-ASR-1.7B",
        api_key="unit-test-only-not-a-production-secret-000000",
        asr_vad_model_path=str(tmp_path / "missing.onnx"),
    )

    with pytest.raises(VADRuntimeError, match="asset is missing"):
        create_vad_endpoint_detector(settings)
