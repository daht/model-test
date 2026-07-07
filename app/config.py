from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = "hy-mt-rest-api"
    model_name: str = "HY-MT1.5-1.8B"
    model_id: str = Field(default="HY-MT1.5-1.8B", description="Hugging Face id or local path")
    model_backend: Literal["transformers", "mock"] = "transformers"
    model_task: Literal["causal-lm", "seq2seq-lm"] = "causal-lm"
    api_key: str = Field(default="change-me", description="Required X-API-Key value")
    device: str = "auto"
    torch_dtype: Literal["auto", "float16", "bfloat16", "float32"] = "float16"
    max_new_tokens: int = 1024
    trust_remote_code: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
