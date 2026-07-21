from collections import UserDict

import httpx
import pytest
from pydantic import ValidationError

from app.config import Settings
from app.model import (
    TranslationBackendTimeout,
    TranslationBackendUnavailable,
    TransformersTranslator,
    VllmTranslator,
    create_translator,
)
from app.schemas import TranslateRequest


class FakeInputIds:
    shape = (1, 3)


class FakeGeneratedRow:
    def __getitem__(self, item):
        assert isinstance(item, slice)
        return [101, 102]


class FakeModelInputs(UserDict):
    def to(self, device):
        self.device = device
        return self


class FakeTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, *args, **kwargs):
        return FakeModelInputs({"input_ids": FakeInputIds(), "attention_mask": object()})

    def decode(self, token_ids, skip_special_tokens):
        assert token_ids == [101, 102]
        assert skip_special_tokens is True
        return "translated text"


class FakeModel:
    device = "cuda:0"

    def generate(self, *args, **kwargs):
        assert args == ()
        assert "input_ids" in kwargs
        assert "attention_mask" in kwargs
        assert kwargs["max_new_tokens"] == 1024
        return [FakeGeneratedRow()]


class TestTransformersTranslator(TransformersTranslator):
    __test__ = False

    def _load(self):
        self._tokenizer = FakeTokenizer()
        self._model = FakeModel()


def test_transformers_translator_handles_batch_encoding_inputs():
    translator = TestTransformersTranslator(Settings())

    result = translator.translate(
        TranslateRequest(
            source_lang="en",
            target_lang="zh",
            text="Hello.",
        )
    )

    assert result == "translated text"


def _vllm_translator(handler, **settings_overrides):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    settings = Settings(
        model_backend="vllm",
        vllm_base_url="http://vllm.test:8000",
        vllm_model="/models/test-model",
        max_new_tokens=77,
        **settings_overrides,
    )
    return VllmTranslator(settings, client=client)


def test_vllm_settings_defaults_and_validation():
    settings = Settings(model_backend="vllm")

    assert settings.vllm_base_url == "http://hy-mt-vllm:8000"
    assert settings.vllm_timeout_seconds == 120.0
    assert settings.vllm_model == "/models/Hy-MT2-1.8B"

    for field, value in (
        ("vllm_base_url", "   "),
        ("vllm_model", "   "),
        ("vllm_timeout_seconds", 0),
        ("vllm_timeout_seconds", 601),
    ):
        with pytest.raises(ValidationError):
            Settings(model_backend="vllm", **{field: value})


def test_create_translator_selects_vllm_backend():
    translator = create_translator(Settings(model_backend="vllm"))

    assert isinstance(translator, VllmTranslator)


def test_vllm_translate_posts_chat_completion_payload_and_strips_content():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url == httpx.URL("http://vllm.test:8000/v1/chat/completions")
        assert request.read()
        assert request.headers["content-type"] == "application/json"
        assert __import__("json").loads(request.content) == {
            "model": "/models/test-model",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Translate the following segment into Chinese, without additional explanation.\n"
                        "OpenAI => OpenAI 公司\n\nHello."
                    ),
                }
            ],
            "max_tokens": 77,
            "temperature": 0.7,
            "top_p": 0.6,
        }
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "  你好。 \n"}}]},
        )

    translator = _vllm_translator(handler)

    result = translator.translate(
        TranslateRequest(
            source_lang="en",
            target_lang="zh",
            text="Hello.",
            glossary={"OpenAI": "OpenAI 公司"},
        )
    )

    assert result == "你好。"


def test_vllm_translate_classifies_timeout_without_leaking_details():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout(
            "secret timeout response from http://private-vllm/request-body",
            request=request,
        )

    translator = _vllm_translator(handler)

    with pytest.raises(TranslationBackendTimeout) as exc_info:
        translator.translate(
            TranslateRequest(source_lang="en", target_lang="zh", text="secret body")
        )

    assert str(exc_info.value) == "Translation backend timed out"


def test_vllm_translate_classifies_connection_failure_without_leaking_details():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(
            "secret connection error at http://private-vllm", request=request
        )

    translator = _vllm_translator(handler)

    with pytest.raises(TranslationBackendUnavailable) as exc_info:
        translator.translate(
            TranslateRequest(source_lang="en", target_lang="zh", text="secret body")
        )

    assert str(exc_info.value) == "Translation backend is unavailable"


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(503, text="secret upstream response"),
        httpx.Response(200, content=b"not-json"),
        httpx.Response(200, json={"choices": []}),
        httpx.Response(200, json={"choices": [{"message": {"content": "   "}}]}),
    ],
    ids=["non-2xx", "invalid-json", "invalid-structure", "empty-content"],
)
def test_vllm_translate_classifies_invalid_responses_as_unavailable(response):
    translator = _vllm_translator(lambda request: response)

    with pytest.raises(TranslationBackendUnavailable) as exc_info:
        translator.translate(
            TranslateRequest(source_lang="en", target_lang="zh", text="secret body")
        )

    assert str(exc_info.value) == "Translation backend is unavailable"


def test_vllm_health_uses_health_endpoint():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url == httpx.URL("http://vllm.test:8000/health")
        return httpx.Response(200)

    _vllm_translator(handler).check_health()


@pytest.mark.parametrize("status_code", [500, 503])
def test_vllm_health_classifies_non_success_as_unavailable(status_code):
    translator = _vllm_translator(
        lambda request: httpx.Response(status_code, text="secret upstream response")
    )

    with pytest.raises(TranslationBackendUnavailable) as exc_info:
        translator.check_health()

    assert str(exc_info.value) == "Translation backend is unavailable"


def test_vllm_health_classifies_network_errors_as_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("secret health timeout", request=request)

    translator = _vllm_translator(handler)

    with pytest.raises(TranslationBackendUnavailable) as exc_info:
        translator.check_health()

    assert str(exc_info.value) == "Translation backend is unavailable"
