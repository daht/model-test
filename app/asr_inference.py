from __future__ import annotations

import asyncio
import itertools
import logging
import queue
import threading
import time
import uuid
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Callable

from app.asr import ASRTranscriber, StreamingTranscriptionResult, TranscriptionResult
from app.config import Settings

logger = logging.getLogger(__name__)


class ASRCoordinatorError(RuntimeError):
    pass


class ASRNotReady(ASRCoordinatorError):
    pass


class ASRQueueFull(ASRCoordinatorError):
    pass


class ASRQueueTimeout(ASRCoordinatorError):
    pass


class ASRInferenceTimeout(ASRCoordinatorError):
    pass


class ASRSessionLimit(ASRCoordinatorError):
    pass


class ASRSessionPoisoned(ASRCoordinatorError):
    pass


class ASRFileTranscriptionDisabled(ASRCoordinatorError):
    pass


class ASRBatchConflict(ASRCoordinatorError):
    pass


@dataclass(frozen=True)
class CoordinatorSnapshot:
    ready: bool
    accepting: bool
    active_streams: int
    queue_depth: int
    queued_audio_seconds: float
    load_error: str | None


class _ChunkedSession:
    def abort(self) -> None:
        return None


@dataclass(order=True)
class _Job:
    priority: int
    sequence: int
    action: str = field(compare=False)
    args: tuple[Any, ...] = field(compare=False)
    session_id: str | None = field(compare=False)
    enqueue_time: float = field(compare=False)
    queue_deadline: float = field(compare=False)
    audio_seconds: float = field(compare=False, default=0.0)
    started: Future[None] = field(compare=False, default_factory=Future)
    result: Future[Any] = field(compare=False, default_factory=Future)
    cancelled: threading.Event = field(compare=False, default_factory=threading.Event)


class ASRInferenceCoordinator:
    def __init__(
        self,
        settings: Settings,
        transcriber_factory: Callable[[], ASRTranscriber],
    ) -> None:
        self.settings = settings
        self._transcriber_factory = transcriber_factory
        self._jobs: queue.PriorityQueue[_Job] = queue.PriorityQueue(
            maxsize=settings.asr_inference_queue_size
        )
        self._sequence = itertools.count()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._startup: Future[None] | None = None
        self._accepting = False
        self._ready = False
        self._load_error: str | None = None
        self._queued_audio_seconds = 0.0
        self._active_sessions: set[str] = set()
        self._pending_streams = 0
        self._poisoned_sessions: set[str] = set()
        self._batch_pending = False
        self._batch_running = False
        self._last_timings: dict[str, tuple[float, float]] = {}

    @property
    def worker_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    async def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._accepting = True
            self._ready = False
            self._load_error = None
            self._startup = Future()
            self._thread = threading.Thread(
                target=self._worker_main,
                name="asr-model-owner",
                daemon=True,
            )
            self._thread.start()
            startup = self._startup
        await asyncio.wrap_future(startup)

    async def stop(self) -> None:
        with self._lock:
            self._accepting = False
            thread = self._thread
            load_failed = self._load_error is not None
        if not thread or not thread.is_alive():
            return
        if load_failed:
            await asyncio.to_thread(thread.join, 10)
            if thread.is_alive():
                raise RuntimeError("ASR inference worker did not stop after load failure")
            return
        shutdown = self._new_job("shutdown", (), None, priority=100, queue_timeout=3600)
        await asyncio.to_thread(self._jobs.put, shutdown)
        await asyncio.to_thread(thread.join, 10)
        if thread.is_alive():
            raise RuntimeError("ASR inference worker did not stop")

    async def create_stream(self, language: str | None) -> str:
        with self._lock:
            self._require_accepting_locked()
            if self._batch_pending or self._batch_running:
                raise ASRBatchConflict("batch transcription owns the model")
            if len(self._active_sessions) + self._pending_streams >= self.settings.asr_max_active_streams:
                raise ASRSessionLimit("maximum active ASR streams reached")
            self._pending_streams += 1
        session_id = uuid.uuid4().hex[:12]
        try:
            return await self._submit(
                "create_stream",
                (session_id, language),
                session_id,
                priority=0,
                queue_timeout=self.settings.asr_stream_queue_timeout_seconds,
                inference_timeout=self.settings.asr_stream_inference_timeout_seconds,
            )
        finally:
            with self._lock:
                self._pending_streams -= 1

    async def create_chunked_stream(self, language: str | None) -> str:
        with self._lock:
            self._require_accepting_locked()
            if self._batch_pending or self._batch_running:
                raise ASRBatchConflict("batch transcription owns the model")
            if len(self._active_sessions) + self._pending_streams >= self.settings.asr_max_active_streams:
                raise ASRSessionLimit("maximum active ASR streams reached")
            self._pending_streams += 1
        session_id = uuid.uuid4().hex[:12]
        try:
            return await self._submit(
                "create_chunked_stream",
                (session_id, language),
                session_id,
                priority=0,
                queue_timeout=self.settings.asr_stream_queue_timeout_seconds,
                inference_timeout=self.settings.asr_stream_inference_timeout_seconds,
            )
        finally:
            with self._lock:
                self._pending_streams -= 1

    async def add_audio(
        self,
        session_id: str,
        pcm_bytes: bytes,
        sample_rate: int,
    ) -> StreamingTranscriptionResult:
        self._require_session_admissible(session_id)
        audio_seconds = (len(pcm_bytes) // 2) / sample_rate if sample_rate > 0 else 0.0
        return await self._submit(
            "add_audio",
            (session_id, pcm_bytes, sample_rate),
            session_id,
            priority=0,
            queue_timeout=self.settings.asr_stream_queue_timeout_seconds,
            inference_timeout=self.settings.asr_stream_inference_timeout_seconds,
            audio_seconds=audio_seconds,
        )

    async def finish_stream(self, session_id: str) -> StreamingTranscriptionResult:
        self._require_session_admissible(session_id)
        return await self._submit(
            "finish_stream",
            (session_id,),
            session_id,
            priority=0,
            queue_timeout=self.settings.asr_stream_queue_timeout_seconds,
            inference_timeout=self.settings.asr_stream_inference_timeout_seconds,
        )

    async def reset_segment(self, session_id: str) -> None:
        self._require_session_admissible(session_id)
        await self._submit(
            "reset_segment",
            (session_id,),
            session_id,
            priority=0,
            queue_timeout=self.settings.asr_stream_queue_timeout_seconds,
            inference_timeout=self.settings.asr_stream_inference_timeout_seconds,
        )

    async def abort_stream(self, session_id: str) -> None:
        with self._lock:
            if session_id not in self._active_sessions:
                return
            if not self._accepting:
                return
        await self._submit(
            "abort_stream",
            (session_id,),
            session_id,
            priority=0,
            queue_timeout=self.settings.asr_stream_queue_timeout_seconds,
            inference_timeout=self.settings.asr_stream_inference_timeout_seconds,
        )

    async def transcribe_file(
        self,
        audio_path: str,
        language: str | None,
    ) -> TranscriptionResult:
        with self._lock:
            self._require_accepting_locked()
            if not self.settings.asr_file_transcribe_enabled:
                raise ASRFileTranscriptionDisabled("file transcription is disabled")
            if self._active_sessions or self._pending_streams or self._batch_pending or self._batch_running:
                raise ASRBatchConflict("live streaming owns the model")
            self._batch_pending = True
        try:
            return await self._submit(
                "transcribe_file",
                (audio_path, language),
                None,
                priority=10,
                queue_timeout=self.settings.asr_stream_queue_timeout_seconds,
                inference_timeout=self.settings.asr_file_inference_timeout_seconds,
            )
        finally:
            with self._lock:
                self._batch_pending = False

    async def transcribe_stream_chunk(
        self,
        session_id: str,
        audio_path: str,
        language: str | None,
        audio_seconds: float,
    ) -> TranscriptionResult:
        self._require_session_admissible(session_id)
        with self._lock:
            if self._batch_pending or self._batch_running:
                raise ASRBatchConflict("batch transcription owns the model")
        return await self._submit(
            "transcribe_stream_chunk",
            (session_id, audio_path, language),
            session_id,
            priority=0,
            queue_timeout=self.settings.asr_stream_queue_timeout_seconds,
            inference_timeout=self.settings.asr_stream_inference_timeout_seconds,
            audio_seconds=audio_seconds,
        )

    def snapshot(self) -> CoordinatorSnapshot:
        with self._lock:
            return CoordinatorSnapshot(
                ready=self._ready,
                accepting=self._accepting,
                active_streams=len(self._active_sessions),
                queue_depth=self._jobs.qsize(),
                queued_audio_seconds=self._queued_audio_seconds,
                load_error=self._load_error,
            )

    def session_timing(self, session_id: str) -> tuple[float, float]:
        with self._lock:
            return self._last_timings.get(session_id, (0.0, 0.0))

    async def _submit(
        self,
        action: str,
        args: tuple[Any, ...],
        session_id: str | None,
        *,
        priority: int,
        queue_timeout: float,
        inference_timeout: float,
        audio_seconds: float = 0.0,
    ) -> Any:
        with self._lock:
            self._require_accepting_locked()
            if self._queued_audio_seconds + audio_seconds > self.settings.asr_max_queued_audio_seconds:
                logger.warning(
                    "asr_queue_rejected reason=audio_limit queue_depth=%d queued_audio_seconds=%.3f",
                    self._jobs.qsize(),
                    self._queued_audio_seconds,
                )
                raise ASRQueueFull("maximum queued audio duration reached")
            self._queued_audio_seconds += audio_seconds
        job = self._new_job(
            action,
            args,
            session_id,
            priority=priority,
            queue_timeout=queue_timeout,
            audio_seconds=audio_seconds,
        )
        try:
            self._jobs.put_nowait(job)
        except queue.Full as exc:
            self._release_audio_reservation(audio_seconds)
            logger.warning(
                "asr_queue_rejected reason=queue_full queue_depth=%d",
                self._jobs.qsize(),
            )
            raise ASRQueueFull("ASR inference queue is full") from exc

        started = asyncio.wrap_future(job.started)
        result = asyncio.wrap_future(job.result)
        try:
            try:
                await asyncio.wait_for(asyncio.shield(started), timeout=queue_timeout)
            except TimeoutError as exc:
                if not job.started.done():
                    job.cancelled.set()
                    logger.warning(
                        "asr_timeout category=queue session_id=%s",
                        session_id or "batch",
                    )
                    raise ASRQueueTimeout("ASR job expired before execution") from exc
                await started
            try:
                return await asyncio.wait_for(asyncio.shield(result), timeout=inference_timeout)
            except TimeoutError as exc:
                if session_id:
                    with self._lock:
                        self._poisoned_sessions.add(session_id)
                logger.warning(
                    "asr_timeout category=inference session_id=%s",
                    session_id or "batch",
                )
                raise ASRInferenceTimeout("ASR inference exceeded its execution timeout") from exc
        except asyncio.CancelledError:
            if job.started.done() and session_id:
                with self._lock:
                    self._poisoned_sessions.add(session_id)
            else:
                job.cancelled.set()
            raise

    def _new_job(
        self,
        action: str,
        args: tuple[Any, ...],
        session_id: str | None,
        *,
        priority: int,
        queue_timeout: float,
        audio_seconds: float = 0.0,
    ) -> _Job:
        enqueue_time = time.monotonic()
        return _Job(
            priority=priority,
            sequence=next(self._sequence),
            action=action,
            args=args,
            session_id=session_id,
            enqueue_time=enqueue_time,
            queue_deadline=enqueue_time + queue_timeout,
            audio_seconds=audio_seconds,
        )

    def _worker_main(self) -> None:
        transcriber: ASRTranscriber | None = None
        sessions: dict[str, Any] = {}
        warmed = False
        try:
            transcriber = self._transcriber_factory()
            if self.settings.asr_eager_load:
                transcriber.warmup()
                warmed = True
                with self._lock:
                    self._ready = True
        except Exception as exc:
            with self._lock:
                self._ready = False
                self._accepting = False
                self._load_error = _sanitize_error(exc)
        finally:
            assert self._startup is not None
            self._startup.set_result(None)

        if transcriber is None:
            return

        while True:
            job = self._jobs.get()
            try:
                if job.action == "shutdown":
                    job.started.set_result(None)
                    for session in list(sessions.values()):
                        try:
                            session.abort()
                        except Exception:
                            pass
                    sessions.clear()
                    with self._lock:
                        self._active_sessions.clear()
                        self._ready = False
                    job.result.set_result(None)
                    return

                if job.cancelled.is_set() or time.monotonic() > job.queue_deadline:
                    error = ASRQueueTimeout("ASR job expired before execution")
                    job.started.set_exception(error)
                    job.result.set_exception(error)
                    continue

                if not warmed:
                    try:
                        transcriber.warmup()
                        warmed = True
                        with self._lock:
                            self._ready = True
                            self._load_error = None
                    except Exception as exc:
                        with self._lock:
                            self._accepting = False
                            self._load_error = _sanitize_error(exc)
                        raise

                queue_wait = time.monotonic() - job.enqueue_time
                job.started.set_result(None)
                inference_started = time.monotonic()
                result = self._execute_job(transcriber, sessions, job)
                inference_elapsed = time.monotonic() - inference_started
                if job.session_id:
                    with self._lock:
                        self._last_timings[job.session_id] = (max(queue_wait, 0.0), inference_elapsed)
                logger.info(
                    "asr_inference_completed job_type=%s session_id=%s queue_wait_ms=%.3f inference_ms=%.3f rtf=%.3f",
                    job.action,
                    job.session_id or "batch",
                    max(queue_wait, 0.0) * 1000,
                    inference_elapsed * 1000,
                    inference_elapsed / job.audio_seconds if job.audio_seconds else 0.0,
                )
                if not job.result.done():
                    job.result.set_result(result)
            except Exception as exc:
                if not job.started.done():
                    job.started.set_result(None)
                if not job.result.done():
                    job.result.set_exception(exc)
            finally:
                self._cleanup_poisoned_session(sessions, job.session_id)
                self._release_audio_reservation(job.audio_seconds)
                self._jobs.task_done()

    def _execute_job(self, transcriber: ASRTranscriber, sessions: dict[str, Any], job: _Job) -> Any:
        if job.action == "create_stream":
            session_id, language = job.args
            sessions[session_id] = transcriber.create_streaming_session(language=language)
            with self._lock:
                self._active_sessions.add(session_id)
            return session_id
        if job.action == "create_chunked_stream":
            session_id, _language = job.args
            sessions[session_id] = _ChunkedSession()
            with self._lock:
                self._active_sessions.add(session_id)
            return session_id
        if job.action == "add_audio":
            session_id, pcm_bytes, sample_rate = job.args
            return sessions[session_id].add_pcm_s16le(pcm_bytes, sample_rate)
        if job.action == "finish_stream":
            (session_id,) = job.args
            session = sessions[session_id]
            try:
                return session.finish()
            except Exception:
                try:
                    session.abort()
                except Exception:
                    pass
                raise
            finally:
                sessions.pop(session_id, None)
                with self._lock:
                    self._active_sessions.discard(session_id)
                    self._last_timings.pop(session_id, None)
        if job.action == "reset_segment":
            (session_id,) = job.args
            return sessions[session_id].reset_segment()
        if job.action == "abort_stream":
            (session_id,) = job.args
            session = sessions.pop(session_id, None)
            try:
                if session is not None:
                    return session.abort()
                return None
            finally:
                with self._lock:
                    self._active_sessions.discard(session_id)
                    self._last_timings.pop(session_id, None)
        if job.action == "transcribe_file":
            audio_path, language = job.args
            with self._lock:
                self._batch_running = True
            try:
                return transcriber.transcribe(audio_path, language=language)
            finally:
                with self._lock:
                    self._batch_running = False
        if job.action == "transcribe_stream_chunk":
            _session_id, audio_path, language = job.args
            return transcriber.transcribe(audio_path, language=language)
        raise RuntimeError(f"unknown ASR coordinator action: {job.action}")

    def _cleanup_poisoned_session(self, sessions: dict[str, Any], session_id: str | None) -> None:
        if not session_id:
            return
        with self._lock:
            poisoned = session_id in self._poisoned_sessions
        if not poisoned:
            return
        session = sessions.pop(session_id, None)
        if session is not None:
            try:
                session.abort()
            finally:
                with self._lock:
                    self._active_sessions.discard(session_id)
                    self._last_timings.pop(session_id, None)

    def _require_session_admissible(self, session_id: str) -> None:
        with self._lock:
            self._require_accepting_locked()
            if session_id in self._poisoned_sessions:
                raise ASRSessionPoisoned("ASR streaming session is poisoned")
            if session_id not in self._active_sessions:
                raise ASRSessionPoisoned("ASR streaming session is closed")

    def _require_accepting_locked(self) -> None:
        if not self._accepting:
            raise ASRNotReady("ASR coordinator is not accepting work")
        if self._load_error:
            raise ASRNotReady("ASR model failed to load")

    def _release_audio_reservation(self, audio_seconds: float) -> None:
        if not audio_seconds:
            return
        with self._lock:
            self._queued_audio_seconds = max(0.0, self._queued_audio_seconds - audio_seconds)


def _sanitize_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: model warmup failed"
