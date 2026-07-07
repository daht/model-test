from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


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
