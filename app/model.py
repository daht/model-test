from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

from app.config import Settings
from app.schemas import TranslateRequest


class Translator:
    def translate(self, request: TranslateRequest) -> str:
        raise NotImplementedError


@dataclass
class MockTranslator(Translator):
    def translate(self, request: TranslateRequest) -> str:
        return f"[mock {request.source_lang}->{request.target_lang}] {request.text}"


class TransformersTranslator(Translator):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = Lock()
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        model_kwargs = {
            "trust_remote_code": self.settings.trust_remote_code,
            "device_map": self.settings.device,
        }
        if self.settings.torch_dtype != "auto":
            model_kwargs["torch_dtype"] = dtype_map[self.settings.torch_dtype]

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.settings.model_id,
            trust_remote_code=self.settings.trust_remote_code,
        )
        model_class = (
            AutoModelForSeq2SeqLM
            if self.settings.model_task == "seq2seq-lm"
            else AutoModelForCausalLM
        )
        self._model = model_class.from_pretrained(
            self.settings.model_id,
            **model_kwargs,
        )
        self._model.eval()

    def translate(self, request: TranslateRequest) -> str:
        with self._lock:
            self._load()
            assert self._tokenizer is not None
            assert self._model is not None

            prompt = self._build_prompt(request)
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.settings.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
            decoded = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            return self._strip_prompt(decoded, prompt)

    @staticmethod
    def _build_prompt(request: TranslateRequest) -> str:
        lines = [
            "You are a professional machine translation engine.",
            f"Translate from {request.source_lang} to {request.target_lang}.",
        ]
        if request.preserve_format:
            lines.append("Preserve line breaks, numbering, markdown, and placeholders.")
        if request.glossary:
            terms = ", ".join(f"{source} => {target}" for source, target in request.glossary.items())
            lines.append(f"Use this glossary exactly where applicable: {terms}.")
        lines.extend(["Text:", request.text, "Translation:"])
        return "\n".join(lines)

    @staticmethod
    def _strip_prompt(decoded: str, prompt: str) -> str:
        if decoded.startswith(prompt):
            return decoded[len(prompt) :].strip()
        marker = "Translation:"
        if marker in decoded:
            return decoded.rsplit(marker, maxsplit=1)[-1].strip()
        return decoded.strip()


def create_translator(settings: Settings) -> Translator:
    if settings.model_backend == "mock":
        return MockTranslator()
    return TransformersTranslator(settings)
