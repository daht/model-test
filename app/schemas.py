from pydantic import BaseModel, Field, field_validator


class TranslateRequest(BaseModel):
    source_lang: str = Field(..., min_length=2, max_length=32, examples=["zh"])
    target_lang: str = Field(..., min_length=2, max_length=32, examples=["en"])
    text: str = Field(..., min_length=1, examples=["你好，欢迎使用我们的产品。"])
    glossary: dict[str, str] | None = Field(default=None)
    preserve_format: bool = True

    @field_validator("source_lang", "target_lang", "text")
    @classmethod
    def reject_blank_values(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value cannot be blank")
        return stripped


class TranslateResponse(BaseModel):
    translation: str
    source_lang: str
    target_lang: str
    model: str


class HealthResponse(BaseModel):
    status: str
    model: str
    backend: str


class ASRHealthResponse(BaseModel):
    status: str
    model: str
    backend: str


class TranscribeResponse(BaseModel):
    text: str
    language: str | None
    model: str


class TranscribeStreamInfoResponse(BaseModel):
    websocket_url: str
    audio_format: dict[str, int | str]
    start_message: dict[str, int | str]
    end_message: dict[str, str]
    server_messages: list[dict[str, str]]
