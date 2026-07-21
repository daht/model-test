from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from threading import Lock

import httpx

from app.config import Settings
from app.schemas import TranslateRequest


LANGUAGE_NAMES = {
    "zh": ("Chinese", "中文"),
    "en": ("English", "英语"),
    "fr": ("French", "法语"),
    "pt": ("Portuguese", "葡萄牙语"),
    "es": ("Spanish", "西班牙语"),
    "ja": ("Japanese", "日语"),
    "tr": ("Turkish", "土耳其语"),
    "ru": ("Russian", "俄语"),
    "ar": ("Arabic", "阿拉伯语"),
    "ko": ("Korean", "韩语"),
    "th": ("Thai", "泰语"),
    "it": ("Italian", "意大利语"),
    "de": ("German", "德语"),
    "vi": ("Vietnamese", "越南语"),
    "ms": ("Malay", "马来语"),
    "id": ("Indonesian", "印尼语"),
    "tl": ("Filipino", "菲律宾语"),
    "hi": ("Hindi", "印地语"),
    "zh-Hant": ("Traditional Chinese", "繁体中文"),
    "pl": ("Polish", "波兰语"),
    "cs": ("Czech", "捷克语"),
    "nl": ("Dutch", "荷兰语"),
    "km": ("Khmer", "高棉语"),
    "my": ("Burmese", "缅甸语"),
    "fa": ("Persian", "波斯语"),
    "gu": ("Gujarati", "古吉拉特语"),
    "ur": ("Urdu", "乌尔都语"),
    "te": ("Telugu", "泰卢固语"),
    "mr": ("Marathi", "马拉地语"),
    "he": ("Hebrew", "希伯来语"),
    "bn": ("Bengali", "孟加拉语"),
    "ta": ("Tamil", "泰米尔语"),
    "uk": ("Ukrainian", "乌克兰语"),
    "bo": ("Tibetan", "藏语"),
    "kk": ("Kazakh", "哈萨克语"),
    "mn": ("Mongolian", "蒙古语"),
    "ug": ("Uyghur", "维吾尔语"),
    "yue": ("Cantonese", "粤语"),
}


class Translator:
    def translate(self, request: TranslateRequest) -> str:
        raise NotImplementedError

    def check_health(self) -> None:
        return None


class TranslationBackendTimeout(RuntimeError):
    def __init__(self) -> None:
        super().__init__("Translation backend timed out")


class TranslationBackendUnavailable(RuntimeError):
    def __init__(self) -> None:
        super().__init__("Translation backend is unavailable")


class VllmTranslator(Translator):
    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        self.settings = settings
        self._base_url = settings.vllm_base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=settings.vllm_timeout_seconds)

    def translate(self, request: TranslateRequest) -> str:
        payload = {
            "model": self.settings.vllm_model,
            "messages": [{"role": "user", "content": _build_prompt(request)}],
            "max_tokens": self.settings.max_new_tokens,
            "temperature": 0.7,
            "top_p": 0.6,
        }
        try:
            response = self._client.post(
                f"{self._base_url}/v1/chat/completions",
                json=payload,
                timeout=self.settings.vllm_timeout_seconds,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError
            return content.strip()
        except httpx.TimeoutException:
            raise TranslationBackendTimeout() from None
        except (
            httpx.RequestError,
            httpx.HTTPStatusError,
            ValueError,
            KeyError,
            IndexError,
            TypeError,
        ):
            raise TranslationBackendUnavailable() from None

    def check_health(self) -> None:
        try:
            response = self._client.get(
                f"{self._base_url}/health",
                timeout=self.settings.vllm_timeout_seconds,
            )
            response.raise_for_status()
        except Exception:
            raise TranslationBackendUnavailable() from None


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

            prompt = _build_prompt(request)
            messages = [{"role": "user", "content": prompt}]
            model_inputs = self._tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt",
            ).to(self._model.device)
            generation_kwargs = {
                "max_new_tokens": self.settings.max_new_tokens,
                "do_sample": True,
                "top_k": 20,
                "top_p": 0.6,
                "temperature": 0.7,
                "repetition_penalty": 1.05,
                "pad_token_id": self._tokenizer.eos_token_id,
            }
            if isinstance(model_inputs, Mapping):
                prompt_length = model_inputs["input_ids"].shape[-1]
                outputs = self._model.generate(**model_inputs, **generation_kwargs)
            else:
                prompt_length = model_inputs.shape[-1]
                outputs = self._model.generate(model_inputs, **generation_kwargs)

            generated_ids = outputs[0][prompt_length:]
            return self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

def create_translator(settings: Settings) -> Translator:
    if settings.model_backend == "mock":
        return MockTranslator()
    if settings.model_backend == "vllm":
        return VllmTranslator(settings)
    return TransformersTranslator(settings)


def _build_prompt(request: TranslateRequest) -> str:
    target_english_name, target_chinese_name = _language_names(request.target_lang)

    if request.source_lang.startswith("zh"):
        lines = [
            f"将以下文本翻译为{target_chinese_name}，注意只需要输出翻译后的结果，不要额外解释：",
        ]
    else:
        lines = [
            f"Translate the following segment into {target_english_name}, without additional explanation.",
        ]
    if request.glossary:
        for source, target in request.glossary.items():
            lines.append(f"{source} => {target}")
    lines.extend(["", request.text])
    return "\n".join(lines)


def _language_names(language_code: str) -> tuple[str, str]:
    return LANGUAGE_NAMES.get(language_code, (language_code, language_code))
