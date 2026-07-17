#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np


SAMPLE_RATE = 16_000
SESSION_SECONDS = 15
UPDATE_SECONDS = 2


def emit(**values: object) -> None:
    print(json.dumps(values, ensure_ascii=False, sort_keys=True), flush=True)


def make_options(tokenizer: object, beam_size: int) -> object:
    from faster_whisper.transcribe import (
        TranscriptionOptions,
        get_suppressed_tokens,
    )

    return TranscriptionOptions(
        beam_size=beam_size,
        best_of=1,
        patience=1,
        length_penalty=1,
        repetition_penalty=1,
        no_repeat_ngram_size=3,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4,
        condition_on_previous_text=False,
        prompt_reset_on_temperature=0.5,
        temperatures=[0.0],
        initial_prompt=None,
        prefix=None,
        suppress_blank=True,
        suppress_tokens=get_suppressed_tokens(tokenizer, [-1]),
        without_timestamps=True,
        max_initial_timestamp=0.0,
        word_timestamps=False,
        prepend_punctuations='"\'“¿([{-',
        append_punctuations='"\'.。,，!！?？:：”)]}、',
        multilingual=False,
        max_new_tokens=None,
        clip_timestamps=[],
        hallucination_silence_threshold=None,
        hotwords=None,
    )


def decode_batch(
    model: object,
    pipeline: object,
    waveforms: list[np.ndarray],
    *,
    language: str,
    beam_size: int,
) -> tuple[int, int]:
    from faster_whisper.audio import pad_or_trim
    from faster_whisper.tokenizer import Tokenizer

    extracted = [model.feature_extractor(item)[..., :-1] for item in waveforms]
    input_frames = max(item.shape[-1] for item in extracted)
    features = np.stack(
        [pad_or_trim(item, length=input_frames) for item in extracted]
    )
    tokenizer = Tokenizer(
        model.hf_tokenizer,
        model.model.is_multilingual,
        task="transcribe",
        language=language,
    )
    _, outputs = pipeline.generate_segment_batched(
        features, tokenizer, make_options(tokenizer, beam_size)
    )
    if len(outputs) != len(waveforms):
        raise RuntimeError(
            f"result count mismatch: {len(outputs)} != {len(waveforms)}"
        )
    return input_frames, sum(len(output["tokens"]) for output in outputs)


def timed_calls(
    model: object,
    pipeline: object,
    calls: list[list[np.ndarray]],
    *,
    language: str,
    beam_size: int,
) -> tuple[float, list[int], int]:
    frames = []
    output_tokens = 0
    started = time.perf_counter()
    for waveforms in calls:
        call_frames, call_tokens = decode_batch(
            model,
            pipeline,
            waveforms,
            language=language,
            beam_size=beam_size,
        )
        frames.append(call_frames)
        output_tokens += call_tokens
    return time.perf_counter() - started, frames, output_tokens


def repeated(waveform: np.ndarray, batch_size: int) -> list[np.ndarray]:
    return [waveform] * batch_size


def rolling_calls(audio: np.ndarray, batch_size: int) -> list[list[np.ndarray]]:
    ends = list(range(UPDATE_SECONDS, SESSION_SECONDS, UPDATE_SECONDS))
    ends.append(SESSION_SECONDS)
    return [
        repeated(audio[: end * SAMPLE_RATE], batch_size)
        for end in ends
    ]


def incremental_calls(audio: np.ndarray, batch_size: int) -> list[list[np.ndarray]]:
    calls = []
    for start in range(0, SESSION_SECONDS, UPDATE_SECONDS):
        end = min(start + UPDATE_SECONDS, SESSION_SECONDS)
        calls.append(
            repeated(audio[start * SAMPLE_RATE : end * SAMPLE_RATE], batch_size)
        )
    return calls


def run_pure_engine(
    model: object,
    pipeline: object,
    audio: np.ndarray,
    *,
    batch_size: int,
    language: str,
) -> None:
    beam_size = 5
    unique_audio_seconds = SESSION_SECONDS * batch_size
    engine_elapsed, frames, output_tokens = timed_calls(
        model,
        pipeline,
        [repeated(audio, batch_size)],
        language=language,
        beam_size=beam_size,
    )
    emit(
        experiment="pure_engine",
        status="PASS",
        batch_size=batch_size,
        beam_size=beam_size,
        input_seconds=SESSION_SECONDS,
        input_feature_frames=frames[0],
        elapsed_seconds=round(engine_elapsed, 6),
        audio_seconds_per_gpu_second=round(
            unique_audio_seconds / engine_elapsed, 6
        ),
        output_tokens=output_tokens,
    )


def run_stream_compute(
    model: object,
    pipeline: object,
    audio: np.ndarray,
    *,
    batch_size: int,
    language: str,
) -> None:
    beam_size = 5
    unique_audio_seconds = SESSION_SECONDS * batch_size
    rolling = rolling_calls(audio, batch_size)
    rolling_elapsed, rolling_frames, rolling_tokens = timed_calls(
        model,
        pipeline,
        rolling,
        language=language,
        beam_size=beam_size,
    )
    rolling_input_seconds = sum(
        len(call[0]) / SAMPLE_RATE for call in rolling
    ) * batch_size
    emit(
        experiment="stream_compute",
        mode="rolling_dynamic",
        status="PASS",
        batch_size=batch_size,
        beam_size=beam_size,
        engine_calls=len(rolling),
        unique_audio_seconds=unique_audio_seconds,
        engine_input_audio_seconds=rolling_input_seconds,
        input_feature_frames=rolling_frames,
        elapsed_seconds=round(rolling_elapsed, 6),
        unique_audio_seconds_per_gpu_second=round(
            unique_audio_seconds / rolling_elapsed, 6
        ),
        output_tokens=rolling_tokens,
    )

    incremental = incremental_calls(audio, batch_size)
    incremental_elapsed, incremental_frames, incremental_tokens = timed_calls(
        model,
        pipeline,
        incremental,
        language=language,
        beam_size=beam_size,
    )
    incremental_input_seconds = sum(
        len(call[0]) / SAMPLE_RATE for call in incremental
    ) * batch_size
    emit(
        experiment="stream_compute",
        mode="incremental_dynamic",
        status="PASS",
        batch_size=batch_size,
        beam_size=beam_size,
        engine_calls=len(incremental),
        unique_audio_seconds=unique_audio_seconds,
        engine_input_audio_seconds=incremental_input_seconds,
        input_feature_frames=incremental_frames,
        elapsed_seconds=round(incremental_elapsed, 6),
        unique_audio_seconds_per_gpu_second=round(
            unique_audio_seconds / incremental_elapsed, 6
        ),
        output_tokens=incremental_tokens,
    )
    emit(
        experiment="rolling_removal",
        status="PASS",
        batch_size=batch_size,
        beam_size=beam_size,
        elapsed_speedup=round(rolling_elapsed / incremental_elapsed, 6),
        engine_input_reduction=round(
            rolling_input_seconds / incremental_input_seconds, 6
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--model-id", default=os.environ.get("ASR_MODEL_ID"))
    parser.add_argument("--language", default="zh")
    parser.add_argument("--device", default=os.environ.get("ASR_DEVICE", "cuda:0"))
    parser.add_argument(
        "--compute-type",
        default=os.environ.get("ASR_FASTER_WHISPER_COMPUTE_TYPE", "float16"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.model_id:
        raise SystemExit("ASR_MODEL_ID or --model-id is required")
    if not args.audio.is_file():
        raise SystemExit(f"audio not found: {args.audio}")

    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio
    from faster_whisper.transcribe import BatchedInferencePipeline

    audio = decode_audio(str(args.audio), sampling_rate=SAMPLE_RATE)
    required = SESSION_SECONDS * SAMPLE_RATE
    if len(audio) < required:
        raise SystemExit(f"audio needs {required} samples, got {len(audio)}")
    audio = np.asarray(audio[:required], dtype=np.float32).copy()

    device, separator, index = args.device.partition(":")
    device_index = int(index) if separator else 0
    model = WhisperModel(
        args.model_id,
        device=device,
        device_index=device_index,
        compute_type=args.compute_type,
        num_workers=1,
    )
    pipeline = BatchedInferencePipeline(model)

    emit(
        experiment="start",
        model_id=args.model_id,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        beam_size=5,
        input_seconds=SESSION_SECONDS,
        batch_sizes=[8, 16, 32],
    )
    decode_batch(
        model,
        pipeline,
        [audio[: UPDATE_SECONDS * SAMPLE_RATE]],
        language=args.language,
        beam_size=5,
    )
    emit(experiment="dynamic_smoke", status="PASS")

    failed = False
    experiments = (
        ("pure_engine", run_pure_engine),
        ("stream_compute", run_stream_compute),
    )
    for experiment, runner in experiments:
        for batch_size in (8, 16, 32):
            try:
                runner(
                    model,
                    pipeline,
                    audio,
                    batch_size=batch_size,
                    language=args.language,
                )
            except Exception as exc:
                failed = True
                emit(
                    experiment=experiment,
                    status="ERROR",
                    batch_size=batch_size,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                traceback.print_exc()
    emit(experiment="complete", status="ERROR" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
