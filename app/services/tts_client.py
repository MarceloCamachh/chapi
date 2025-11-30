"""Cliente Text-to-Speech soportado por Gemini Audio."""

from __future__ import annotations

import io
import os
import wave
from typing import Final, Iterable

from google.genai import types

from app.services.gemini_client import get_gemini_client


class TTSClientError(RuntimeError):
    """Errores específicos del cliente TTS."""


GEMINI_TTS_MODEL: Final[str] = os.getenv(
    "GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts"
)
DEFAULT_GEMINI_VOICE: Final[str | None] = os.getenv("GEMINI_TTS_VOICE")


def _linear16_to_wav(raw_audio: bytes, *, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(raw_audio)
    return buffer.getvalue()


def _extract_audio_part(
    response: types.GenerateContentResponse,
) -> tuple[bytes, str | None]:
    candidates: Iterable[types.Candidate] = response.candidates or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if not content or not getattr(content, "parts", None):
            continue
        for part in content.parts:
            inline = getattr(part, "inline_data", None)
            if inline and inline.data:
                return bytes(inline.data), getattr(inline, "mime_type", None)
    raise TTSClientError("Gemini no devolvió datos de audio en la respuesta.")


def _sample_rate_from_mime(mime_type: str | None, fallback: int = 24_000) -> int:
    if not mime_type:
        return fallback
    for token in mime_type.split(";"):
        token = token.strip()
        if token.startswith("rate="):
            try:
                return int(token.split("=", 1)[1])
            except ValueError:
                return fallback
    return fallback


def _pcm_to_wav(raw_audio: bytes, mime_type: str | None) -> bytes:
    sample_rate = _sample_rate_from_mime(mime_type)
    # Los modelos preview de Gemini devuelven PCM little-endian, así que basta
    # con encapsularlo en un WAV sin hacer byteswap adicional.
    return _linear16_to_wav(raw_audio, sample_rate=sample_rate)


def text_to_audio(
    text: str,
    *,
    voice_name: str | None = None,
) -> bytes:
    """
    Genera audio usando los modelos preview de Gemini TTS.

    voice_name puede tomar cualquiera de las voces soportadas por Gemini
    (p. ej. "Puck", "Sprout", "Charon"). Si no se define, se usa el valor de
    GEMINI_TTS_VOICE o la voz por defecto del modelo.
    """

    normalized_text = text.strip()
    if not normalized_text:
        raise TTSClientError("El texto para TTS está vacío.")

    speech_voice = voice_name or DEFAULT_GEMINI_VOICE
    config_kwargs: dict[str, object] = {"response_modalities": ["AUDIO"]}
    if speech_voice:
        # El transformer acepta strings directas como speech_config=voice_name.
        config_kwargs["speech_config"] = speech_voice

    client = get_gemini_client()
    try:
        response = client.models.generate_content(
            model=GEMINI_TTS_MODEL,
            contents=normalized_text,
            config=types.GenerateContentConfig(**config_kwargs),
        )
    except Exception as exc:  # pragma: no cover - depende del servicio
        raise TTSClientError(f"Error al invocar Gemini TTS: {exc}") from exc

    audio_bytes, mime_type = _extract_audio_part(response)
    return _pcm_to_wav(audio_bytes, mime_type)




