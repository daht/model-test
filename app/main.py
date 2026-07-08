from fastapi import Depends, FastAPI

from app.auth import require_api_key
from app.config import Settings, get_settings
from app.model import Translator, create_translator
from app.schemas import HealthResponse, TranslateRequest, TranslateResponse

settings = get_settings()
translator = create_translator(settings)

app = FastAPI(
    title="HY-MT REST API",
    version="0.1.0",
    description="REST API template for HY-MT1.5-1.8B translation deployment.",
)


def get_translator() -> Translator:
    return translator


@app.get("/health", response_model=HealthResponse)
def health(current_settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model=current_settings.model_name,
        backend=current_settings.model_backend,
    )


@app.post(
    "/v1/translate",
    response_model=TranslateResponse,
    dependencies=[Depends(require_api_key)],
)
def translate(
    request: TranslateRequest,
    current_settings: Settings = Depends(get_settings),
    current_translator: Translator = Depends(get_translator),
) -> TranslateResponse:
    translated_text = current_translator.translate(request)
    return TranslateResponse(
        translation=translated_text,
        source_lang=request.source_lang,
        target_lang=request.target_lang,
        model=current_settings.model_name,
    )
