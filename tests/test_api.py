import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app, get_translator
from app.model import TranslationBackendTimeout, TranslationBackendUnavailable


client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_dependency_overrides():
    app.dependency_overrides[get_settings] = lambda: Settings(
        model_backend="mock",
        asr_backend="mock",
        api_key="test-key",
    )
    yield
    app.dependency_overrides.clear()


class FakeTranslator:
    def __init__(self, result="fake translation", error=None):
        self.result = result
        self.error = error

    def translate(self, request):
        if self.error:
            raise self.error
        return self.result

    def check_health(self):
        if self.error:
            raise self.error


def test_health_reports_model_name():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model"] == "HY-MT1.5-1.8B"


def test_translate_rejects_missing_api_key():
    response = client.post(
        "/v1/translate",
        json={
            "source_lang": "zh",
            "target_lang": "en",
            "text": "你好",
        },
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key"


def test_translate_returns_translation_with_api_key():
    app.dependency_overrides[get_translator] = lambda: FakeTranslator(
        "[mock zh->en] 你好，欢迎使用我们的产品。"
    )
    response = client.post(
        "/v1/translate",
        headers={"X-API-Key": "test-key"},
        json={
            "source_lang": "zh",
            "target_lang": "en",
            "text": "你好，欢迎使用我们的产品。",
            "glossary": {"产品": "product"},
            "preserve_format": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["translation"] == "[mock zh->en] 你好，欢迎使用我们的产品。"
    assert body["source_lang"] == "zh"
    assert body["target_lang"] == "en"
    assert body["model"] == "HY-MT1.5-1.8B"


def test_translate_rejects_empty_text():
    response = client.post(
        "/v1/translate",
        headers={"X-API-Key": "test-key"},
        json={
            "source_lang": "zh",
            "target_lang": "en",
            "text": "   ",
        },
    )

    assert response.status_code == 422


def test_translate_maps_backend_unavailable_to_502():
    app.dependency_overrides[get_translator] = lambda: FakeTranslator(
        error=TranslationBackendUnavailable()
    )

    response = client.post(
        "/v1/translate",
        headers={"X-API-Key": "test-key"},
        json={"source_lang": "en", "target_lang": "zh", "text": "secret"},
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "Translation backend is unavailable"}


def test_translate_maps_backend_timeout_to_504():
    app.dependency_overrides[get_translator] = lambda: FakeTranslator(
        error=TranslationBackendTimeout()
    )

    response = client.post(
        "/v1/translate",
        headers={"X-API-Key": "test-key"},
        json={"source_lang": "en", "target_lang": "zh", "text": "secret"},
    )

    assert response.status_code == 504
    assert response.json() == {"detail": "Translation backend timed out"}


def test_health_checks_translator_and_maps_unavailable_to_503():
    app.dependency_overrides[get_translator] = lambda: FakeTranslator(
        error=TranslationBackendUnavailable()
    )

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json() == {"detail": "Translation backend is not ready"}
