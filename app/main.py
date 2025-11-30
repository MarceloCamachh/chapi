from fastapi import FastAPI

from app.routers.text import router as text_router
from app.routers.voice import router as voice_router


def create_app() -> FastAPI:
    """Crea la instancia principal de FastAPI y monta los routers."""
    app = FastAPI(
        title="Robot Voice Backend",
        version="0.1.0",
        description="Orquestador FastAPI para STT → Gemini → TTS",
    )

    app.include_router(text_router)
    app.include_router(voice_router)

    @app.get("/health", tags=["health"])
    def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()


