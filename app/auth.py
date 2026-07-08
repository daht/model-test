from fastapi import Depends, Header, HTTPException, status

from app.config import Settings, get_settings


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    current_settings: Settings = Depends(get_settings),
) -> None:
    if not x_api_key or x_api_key != current_settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )


def is_valid_api_key(api_key: str | None, settings: Settings) -> bool:
    return bool(api_key and api_key == settings.api_key)
