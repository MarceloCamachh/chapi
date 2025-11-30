from __future__ import annotations

import os
from functools import lru_cache
from typing import Iterable, Sequence

from google import genai

DEFAULT_MODEL = "gemini-2.5-flash"


class GeminiClientError(RuntimeError):
    """Excepción específica para errores del cliente de Gemini."""


@lru_cache(maxsize=1)
def _load_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise GeminiClientError(
            "GEMINI_API_KEY no está definido. Configura la variable de entorno antes de usar el cliente."
        )
    return genai.Client(api_key=api_key)


def get_gemini_client() -> genai.Client:
    """Devuelve el cliente cacheado para reutilizar conexión/API key."""
    return _load_client()


def chat_with_gemini(
    prompt: str,
    history: Sequence[str] | None = None,
    *,
    model: str = DEFAULT_MODEL,
    session_id: str | None = None,
) -> str:
    """
    Envía el texto recibido al modelo Gemini y devuelve la respuesta.

    session_id es opcional por si luego se desea persistir contexto por robot.
    """
    contents: list[str] = []
    if history:
        contents.extend(history)
    contents.append(prompt)

    client = get_gemini_client()
    response = client.models.generate_content(
        model=model,
        contents=contents,
    )

    reply = getattr(response, "text", None) or getattr(response, "output_text", None)
    if not reply:
        raise GeminiClientError("La respuesta de Gemini no contiene texto interpretable.")
    return reply.strip()


