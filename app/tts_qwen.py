from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from app.config import Settings


_STOP = object()


@dataclass
class _QwenRequest:
    text: str
    done: threading.Event
    pcm: bytes | None = None
    error: Exception | None = None
    cancelled: bool = False


class Qwen3TTSSynthesizer:
    """Qwen3-TTS Python backend with bounded explicit micro-batching.

    The Qwen Python API returns complete waveforms. This worker therefore
    coalesces requests into the API's list batch, but it is not continuous
    batching and does not produce incremental audio.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None
        self._load_lock = threading.RLock()
        self._load_error: Exception | None = None
        self._queue: queue.Queue[object] = queue.Queue(
            maxsize=settings.tts_qwen_queue_size
        )
        self._worker: threading.Thread | None = None

    def start(self) -> None:
        """Load the model and start the single inference owner thread."""
        with self._load_lock:
            self._load_model()
            if self._worker is not None and self._worker.is_alive():
                return
            self._worker = threading.Thread(
                target=self._worker_loop,
                name="qwen-tts-inference",
                daemon=True,
            )
            self._worker.start()

    def close(self) -> None:
        worker = self._worker
        if worker is None or not worker.is_alive():
            self._worker = None
            return
        while True:
            try:
                self._queue.put(_STOP, timeout=0.1)
                break
            except queue.Full:
                continue
        worker.join(timeout=10)
        self._worker = None

    def warmup(self) -> None:
        """Run one real decode so startup readiness covers inference too."""
        self.start()
        request = _QwenRequest(text="你好", done=threading.Event())
        self._run_batch([request])
        if request.error is not None:
            raise RuntimeError(str(request.error)) from request.error
        if not request.pcm:
            raise RuntimeError("Qwen3-TTS warmup returned no audio")

    @property
    def ready(self) -> bool:
        return (
            self._model is not None
            and self._worker is not None
            and self._worker.is_alive()
        )

    def synthesize(self, text: str, voice: str | None = None) -> bytes:
        from app.tts import _wav_bytes

        return _wav_bytes(
            b"".join(self.stream_pcm(text, voice)),
            sample_rate=self.settings.tts_sample_rate,
        )

    def stream_pcm(self, text: str, voice: str | None = None) -> Iterator[bytes]:
        selected_voice = voice or self.settings.tts_default_voice
        if selected_voice != self.settings.tts_default_voice:
            raise RuntimeError(f"unknown TTS voice: {selected_voice}")

        request = _QwenRequest(text=text, done=threading.Event())
        self.start()
        deadline = time.monotonic() + self.settings.tts_qwen_request_timeout_seconds
        try:
            self._queue.put(request, timeout=max(0.0, deadline - time.monotonic()))
        except queue.Full as exc:
            raise RuntimeError("Qwen3-TTS request queue is full") from exc

        remaining = deadline - time.monotonic()
        if remaining <= 0 or not request.done.wait(remaining):
            request.cancelled = True
            raise RuntimeError("Qwen3-TTS request timed out")
        if request.error is not None:
            raise RuntimeError(str(request.error)) from request.error
        if not request.pcm:
            raise RuntimeError("Qwen3-TTS returned no audio")
        yield request.pcm

    def _load_model(self):
        with self._load_lock:
            if self._model is not None:
                return self._model
            if self._load_error is not None:
                raise RuntimeError(
                    f"Qwen3-TTS backend is unavailable: {self._load_error}"
                ) from self._load_error

            try:
                import torch
                from qwen_tts import Qwen3TTSModel

                dtype = {
                    "auto": torch.float16,
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16,
                    "float32": torch.float32,
                }[self.settings.torch_dtype]
                device_map = self.settings.tts_device
                if device_map == "auto":
                    device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
                self._model = _load_qwen_model(
                    Qwen3TTSModel,
                    self.settings.tts_model_id,
                    device_map=device_map,
                    dtype=dtype,
                )
            except Exception as exc:
                self._load_error = exc
                raise RuntimeError(
                    f"failed to load Qwen3-TTS model {self.settings.tts_model_id}: {exc}"
                ) from exc
            return self._model

    def _worker_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is _STOP:
                return
            assert isinstance(item, _QwenRequest)
            batch = [item]
            deadline = time.monotonic() + self.settings.tts_qwen_batch_wait_ms / 1000
            while len(batch) < self.settings.tts_qwen_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if item is _STOP:
                    self._queue.put(_STOP)
                    break
                assert isinstance(item, _QwenRequest)
                batch.append(item)
            self._run_batch(batch)

    def _run_batch(self, requests: list[_QwenRequest]) -> None:
        try:
            wavs, sample_rate = self._model.generate_custom_voice(
                text=[request.text for request in requests],
                language=self.settings.tts_qwen_language,
                speaker=self.settings.tts_qwen_speaker,
                instruct=self.settings.tts_qwen_instruct or None,
            )
            if int(sample_rate) != self.settings.tts_sample_rate:
                raise RuntimeError(
                    f"Qwen3-TTS returned {sample_rate} Hz; configured output is "
                    f"{self.settings.tts_sample_rate} Hz"
                )
            if len(wavs) != len(requests):
                raise RuntimeError(
                    f"Qwen3-TTS returned {len(wavs)} waveforms for {len(requests)} requests"
                )
            for request, wav in zip(requests, wavs):
                if not request.cancelled:
                    request.pcm = _float_audio_to_pcm(wav)
        except Exception as exc:
            for request in requests:
                request.error = RuntimeError(f"Qwen3-TTS inference failed: {exc}")
        finally:
            for request in requests:
                request.done.set()


def _load_qwen_model(model_class, model_id: str, *, device_map: str, dtype):
    """Use the official Qwen loading contract without masking load failures."""
    return model_class.from_pretrained(
        model_id,
        device_map=device_map,
        dtype=dtype,
    )


def _float_audio_to_pcm(audio) -> bytes:
    samples = np.asarray(audio).reshape(-1)
    if samples.dtype.kind == "f":
        samples = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    else:
        samples = samples.astype("<i2")
    return samples.tobytes()
