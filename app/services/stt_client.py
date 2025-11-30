"""Transcripción de audio usando Gemini multimodal."""

from __future__ import annotations

import mimetypes
import os
from typing import Final, Iterable

from google.genai import types  # type: ignore[import]

from app.services.gemini_client import get_gemini_client


class STTClientError(RuntimeError):
    """Errores específicos del cliente STT."""


GEMINI_STT_MODEL: Final[str] = os.getenv("GEMINI_STT_MODEL", "gemini-2.5-flash")
DEFAULT_STT_PROMPT: Final[str] = os.getenv(
    "GEMINI_STT_PROMPT",
    "Transcribe el audio exactamente en español, sin comentarios adicionales.",
)


def _resolve_mime_type(
    *,
    explicit_mime: str | None,
    filename: str | None,
) -> str:
    if explicit_mime:
        return explicit_mime
    if filename:
        guessed, _ = mimetypes.guess_type(filename)
        if guessed:
            return guessed
    return "audio/wav"


def _build_audio_part(audio_bytes: bytes, mime_type: str) -> types.Part:
    try:
        return types.Part.from_bytes(audio_bytes, mime_type=mime_type)
    except Exception as exc:  # pragma: no cover - validación
        raise STTClientError(
            f"No se pudo preparar el audio para Gemini ({mime_type}): {exc}"
        ) from exc


def _extract_text(response: types.GenerateContentResponse) -> str:
    primary = getattr(response, "text", None) or getattr(
        response, "output_text", None
    )
    if primary:
        return primary.strip()

    candidates: Iterable[types.Candidate] = response.candidates or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", []) or []:
            text = getattr(part, "text", None)
            if text and text.strip():
                return text.strip()

    raise STTClientError("Gemini no devolvió texto en la transcripción.")


def audio_to_text(
    audio_bytes: bytes,
    *,
    filename: str | None = None,
    mime_type: str | None = None,
    prompt: str | None = None,
) -> str:
    """
    Transcribe audio arbitrario usando el modelo multimodal de Gemini.

    Acepta formatos como WAV, MP3 o M4A. prompt permite personalizar la instrucción.
    """

    if not audio_bytes:
        raise STTClientError("Se recibió un archivo de audio vacío.")

    final_mime = _resolve_mime_type(explicit_mime=mime_type, filename=filename)
    audio_part = _build_audio_part(audio_bytes, final_mime)
    instruction = prompt or DEFAULT_STT_PROMPT

    client = get_gemini_client()
    try:
        response = client.models.generate_content(
            model=GEMINI_STT_MODEL,
            contents=[
                types.Part.from_text(instruction),
                audio_part,
            ],
            config=types.GenerateContentConfig(
                response_modalities=["TEXT"],
                temperature=0,
            ),
        )
    except Exception as exc:  # pragma: no cover - depende del servicio
        raise STTClientError(f"Error al invocar Gemini STT: {exc}") from exc

    return _extract_text(response)


