from __future__ import annotations

import queue
import uuid
from pathlib import Path
from typing import Iterator

import numpy as np

from app.config import Settings


class TritonTTSSynthesizer:
    """CosyVoice3 Triton decoupled-streaming adapter."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._prompt_audio: np.ndarray | None = None

    def synthesize(self, text: str, voice: str | None = None) -> bytes:
        from app.tts import _wav_bytes

        return _wav_bytes(
            b"".join(self.stream_pcm(text, voice)),
            sample_rate=self.settings.tts_sample_rate,
        )

    def stream_pcm(self, text: str, voice: str | None = None) -> Iterator[bytes]:
        if voice not in (None, self.settings.tts_default_voice):
            raise RuntimeError(f"unknown TTS voice: {voice}")

        grpcclient = self._grpcclient()
        prompt = self._load_prompt()
        inputs = self._inputs(grpcclient, prompt, text)
        outputs = [grpcclient.InferRequestedOutput("waveform")]
        responses: queue.Queue[tuple[object | None, object | None]] = queue.Queue(maxsize=2)

        def callback(result, error) -> None:
            responses.put((result, error))

        client = grpcclient.InferenceServerClient(url=self.settings.tts_triton_url)
        request_id = str(uuid.uuid4())
        client.start_stream(callback=callback)
        yielded = False
        try:
            client.async_stream_infer(
                model_name=self.settings.tts_triton_model_name,
                inputs=inputs,
                outputs=outputs,
                request_id=request_id,
                enable_empty_final_response=True,
            )
            while True:
                result, error = responses.get()
                if error is not None:
                    raise RuntimeError(f"Triton TTS inference failed: {error}")
                assert result is not None
                response = result.get_response()
                final = response.parameters.get("triton_final_response")
                if final is not None and getattr(final, "bool_param", False):
                    break
                audio = result.as_numpy("waveform")
                if audio is None:
                    continue
                pcm = _float_audio_to_pcm(audio)
                if pcm:
                    yielded = True
                    yield pcm
        finally:
            client.stop_stream(cancel_requests=True)

        if not yielded:
            raise RuntimeError("Triton CosyVoice did not return audio")

    def capacity_snapshot(self) -> dict[str, object]:
        ready = False
        try:
            grpcclient = self._grpcclient()
            client = grpcclient.InferenceServerClient(url=self.settings.tts_triton_url)
            ready = bool(
                client.is_server_ready()
                and client.is_model_ready(self.settings.tts_triton_model_name)
            )
        except Exception:
            ready = False
        return {
            "ready": ready,
            "supports_realtime_streaming": True,
            "supports_microbatch": False,
            "queue_depth": None,
            "queue_size": None,
            "batch_size": None,
            "batch_wait_ms": None,
            "dispatched_batches": None,
            "dispatched_requests": None,
            "last_batch_size": None,
        }

    def _grpcclient(self):
        try:
            import tritonclient.grpc as grpcclient
        except ImportError as exc:
            raise RuntimeError(
                "Triton TTS requires tritonclient[grpc] in the API environment"
            ) from exc
        return grpcclient

    def _load_prompt(self) -> np.ndarray:
        if self._prompt_audio is not None:
            return self._prompt_audio
        try:
            import soundfile as sf

            samples, sample_rate = sf.read(
                str(Path(self.settings.tts_prompt_wav).expanduser()),
                dtype="float32",
                always_2d=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"failed to load Triton TTS prompt audio: {exc}"
            ) from exc
        if sample_rate != 16000:
            raise RuntimeError(
                f"Triton CosyVoice prompt audio must be 16000 Hz, got {sample_rate}"
            )
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=-1)
        samples = samples.reshape(-1)
        if samples.size == 0:
            raise RuntimeError("Triton TTS prompt audio is empty")
        self._prompt_audio = samples
        return samples

    def _inputs(self, grpcclient, prompt: np.ndarray, text: str):
        prompt = prompt.reshape(1, -1).astype(np.float32, copy=False)
        prompt_length = np.array([[prompt.shape[1]]], dtype=np.int32)
        inputs = [
            grpcclient.InferInput("reference_wav", prompt.shape, "FP32"),
            grpcclient.InferInput("reference_wav_len", prompt_length.shape, "INT32"),
            grpcclient.InferInput("reference_text", [1, 1], "BYTES"),
            grpcclient.InferInput("target_text", [1, 1], "BYTES"),
        ]
        inputs[0].set_data_from_numpy(prompt)
        inputs[1].set_data_from_numpy(prompt_length)
        inputs[2].set_data_from_numpy(np.array([[self.settings.tts_prompt_text]], dtype=object))
        inputs[3].set_data_from_numpy(np.array([[text]], dtype=object))
        return inputs


def _float_audio_to_pcm(audio) -> bytes:
    samples = np.asarray(audio).reshape(-1)
    if samples.dtype.kind == "f":
        samples = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    else:
        samples = samples.astype("<i2")
    return samples.tobytes()
