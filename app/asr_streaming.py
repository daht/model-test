from __future__ import annotations

from dataclasses import dataclass

SENTENCE_TERMINATORS = set("。？！｡?!؟۔।॥။.")
SENTENCE_CLOSERS = set("\"'”’」』）)】]》〉")
DOT_ABBREVIATIONS = {
    "dr.",
    "mr.",
    "mrs.",
    "ms.",
    "prof.",
    "e.g.",
    "i.e.",
    "etc.",
    "vs.",
    "st.",
    "jr.",
    "sr.",
    "ph.d.",
}


class ConfirmedPrefixConflict(RuntimeError):
    pass


@dataclass(frozen=True)
class TranscriptEvent:
    type: str
    text: str
    sequence: int


@dataclass
class StablePunctuationTracker:
    enabled: bool
    stable_seconds: float
    min_chars: int
    min_updates: int
    candidate: str = ""
    candidate_audio_time: float = 0.0
    candidate_updates: int = 0

    def observe(self, text: str, audio_time: float) -> str | None:
        next_candidate = first_punctuation_candidate(text, self.min_chars) if self.enabled else ""
        if not next_candidate:
            self.reset()
            return None
        if next_candidate != self.candidate:
            self.candidate = next_candidate
            self.candidate_audio_time = audio_time
            self.candidate_updates = 1
            return None
        self.candidate_updates += 1
        if self.candidate_updates < self.min_updates:
            return None
        if audio_time - self.candidate_audio_time < self.stable_seconds:
            return None
        result = self.candidate
        self.reset()
        return result

    def reset(self) -> None:
        self.candidate = ""
        self.candidate_audio_time = 0.0
        self.candidate_updates = 0


class StreamingTranscriptState:
    def __init__(
        self,
        *,
        sample_rate: int,
        stable_commit_enabled: bool,
        stable_commit_seconds: float,
        stable_commit_min_chars: int,
        stable_commit_min_updates: int,
    ) -> None:
        self.sample_rate = sample_rate
        self.confirmed_text = ""
        self.partial_text = ""
        self.processed_samples = 0
        self._sequence = 0
        self.stable = StablePunctuationTracker(
            stable_commit_enabled,
            stable_commit_seconds,
            stable_commit_min_chars,
            stable_commit_min_updates,
        )

    @property
    def audio_time(self) -> float:
        return self.processed_samples / self.sample_rate

    @property
    def next_sequence(self) -> int:
        return self._sequence + 1

    def apply_model_update(self, text: str, *, processed_samples: int) -> list[TranscriptEvent]:
        self.processed_samples += processed_samples
        previous = self.partial_text
        self.partial_text = self._unconfirmed_tail(text)
        stable_prefix = self.stable.observe(self.partial_text, self.audio_time)
        if stable_prefix:
            return self._commit_prefix(stable_prefix)
        if self.partial_text != previous:
            return [self._event("partial", self.partial_text)]
        return []

    def commit_pending(self) -> list[TranscriptEvent]:
        if not self.partial_text:
            self.stable.reset()
            return []
        sentence = self.partial_text.rstrip()
        self.confirmed_text += sentence
        self.partial_text = ""
        self.stable.reset()
        return [
            self._event("sentence_final", sentence),
            self._event("partial", ""),
        ]

    def finish(self, text: str) -> list[TranscriptEvent]:
        events = self.apply_model_update(text, processed_samples=0)
        events.append(self._event("final", self.partial_text))
        return events

    def reset_segment(self) -> None:
        self.stable.reset()

    def new_event(self, event_type: str, text: str = "") -> TranscriptEvent:
        return self._event(event_type, text)

    def _unconfirmed_tail(self, text: str) -> str:
        if not self.confirmed_text:
            return text
        if text.startswith(self.confirmed_text):
            return text[len(self.confirmed_text) :]
        overlap = min(len(self.confirmed_text), len(text))
        while overlap:
            if self.confirmed_text.endswith(text[:overlap]):
                return text[overlap:]
            overlap -= 1
        raise ConfirmedPrefixConflict("model text conflicts with confirmed transcript prefix")

    def _commit_prefix(self, prefix: str) -> list[TranscriptEvent]:
        if not prefix or not self.partial_text.startswith(prefix):
            return []
        self.confirmed_text += prefix
        self.partial_text = self.partial_text[len(prefix) :]
        self.stable.reset()
        return [
            self._event("sentence_final", prefix),
            self._event("partial", self.partial_text),
        ]

    def _event(self, event_type: str, text: str) -> TranscriptEvent:
        self._sequence += 1
        return TranscriptEvent(event_type, text, self._sequence)


@dataclass
class SilenceEndpointDetector:
    silence_seconds: float
    rms_threshold: int
    current_silence_seconds: float = 0.0
    committed_for_current_silence: bool = False

    def add_audio(self, pcm_bytes: bytes, sample_rate: int) -> bool:
        if not pcm_bytes:
            return False
        if pcm_s16le_rms(pcm_bytes) > self.rms_threshold:
            self.current_silence_seconds = 0.0
            self.committed_for_current_silence = False
            return False
        self.current_silence_seconds += pcm_s16le_duration_seconds(pcm_bytes, sample_rate)
        if self.current_silence_seconds < self.silence_seconds or self.committed_for_current_silence:
            return False
        self.committed_for_current_silence = True
        return True

    def reset(self) -> None:
        self.current_silence_seconds = 0.0
        self.committed_for_current_silence = False


def pcm_s16le_rms(pcm_bytes: bytes) -> int:
    sample_count = len(pcm_bytes) // 2
    if sample_count == 0:
        return 0
    total_square = 0
    for index in range(0, sample_count * 2, 2):
        sample = int.from_bytes(pcm_bytes[index : index + 2], byteorder="little", signed=True)
        total_square += sample * sample
    return int((total_square / sample_count) ** 0.5)


def pcm_s16le_duration_seconds(pcm_bytes: bytes, sample_rate: int) -> float:
    if sample_rate <= 0:
        return 0.0
    return (len(pcm_bytes) // 2) / sample_rate


def split_committed_sentences(text: str) -> tuple[list[str], str]:
    committed: list[str] = []
    sentence_start = 0
    index = 0
    while index < len(text):
        if text[index] in SENTENCE_TERMINATORS and _is_sentence_end(text, index):
            sentence_end = _sentence_boundary_end(text, index)
            sentence = text[sentence_start:sentence_end].strip()
            if sentence:
                committed.append(sentence)
            sentence_start = sentence_end
            while sentence_start < len(text) and text[sentence_start].isspace():
                sentence_start += 1
            index = sentence_start
            continue
        index += 1
    return committed, text[sentence_start:]


def first_punctuation_candidate(text: str, min_chars: int) -> str:
    index = 0
    while index < len(text):
        if text[index] in SENTENCE_TERMINATORS and _is_sentence_end(text, index):
            sentence_end = _sentence_boundary_end(text, index)
            candidate = text[:sentence_end]
            if len("".join(candidate.split())) >= min_chars:
                return candidate
            index = sentence_end
            continue
        index += 1
    return ""


def _sentence_boundary_end(text: str, index: int) -> int:
    sentence_end = index + 1
    while sentence_end < len(text):
        char = text[sentence_end]
        if char not in SENTENCE_TERMINATORS and char not in SENTENCE_CLOSERS:
            break
        sentence_end += 1
    return sentence_end


def _is_sentence_end(text: str, index: int) -> bool:
    if text[index] != ".":
        return True
    previous_char = text[index - 1] if index > 0 else ""
    next_char = text[index + 1] if index + 1 < len(text) else ""
    if previous_char.isdigit() and next_char.isdigit():
        return False
    if previous_char.isalnum() and next_char.isalnum():
        return False
    dot_token = _dot_token(text, index)
    if dot_token.lower() in DOT_ABBREVIATIONS:
        return False
    if len(dot_token) == 2 and dot_token[0].isupper():
        return False
    if _is_dotted_initialism(dot_token):
        return False
    return True


def _is_dotted_initialism(token: str) -> bool:
    parts = token.split(".")
    initials = [part for part in parts if part]
    return len(initials) >= 2 and all(len(part) == 1 and part.isupper() for part in initials)


def _dot_token(text: str, index: int) -> str:
    start = index
    while start > 0 and (text[start - 1].isalpha() or text[start - 1] == "."):
        start -= 1
    return text[start : index + 1]
