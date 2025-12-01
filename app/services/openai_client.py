from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Sequence

import httpx
from openai import OpenAI

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SYSTEM_PROMPT_FILE = os.getenv("OPENAI_SYSTEM_PROMPT_FILE", "prompts/system_prompt.txt")
INTRO_TEXT = "Hola, soy Chapi, tu compañero de apoyo emocional."
_SESSION_GREETING_CACHE: set[str] = set()
_SESSION_GREETING_LOCK = Lock()
_DEFAULT_INTRO_SENT = False
_DEFAULT_INTRO_LOCK = Lock()


class OpenAIClientError(RuntimeError):
    """Errores específicos del cliente de OpenAI."""


@lru_cache(maxsize=1)
def _load_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise OpenAIClientError(
            "OPENAI_API_KEY no está definido. Configúralo en el entorno antes de usar el cliente."
        )
    return OpenAI(api_key=api_key, http_client=_build_httpx_client())


def _build_httpx_client() -> httpx.Client:
    return httpx.Client(follow_redirects=True, timeout=httpx.Timeout(30.0))


def get_openai_client() -> OpenAI:
    """Devuelve el cliente cacheado para reutilizar conexión/API key."""
    return _load_client()


@lru_cache(maxsize=1)
def _load_system_prompt() -> str | None:
    path = Path(SYSTEM_PROMPT_FILE)
    if not path.is_file():
        return None
    content = path.read_text(encoding="utf-8").strip()
    return content or None


def chat_with_openai(
    prompt: str,
    history: Sequence[str] | None = None,
    *,
    model: str = DEFAULT_MODEL,
    session_id: str | None = None,
) -> str:
    """Envía el texto a OpenAI y devuelve la respuesta en limpio."""
    messages: list[dict[str, str]] = []

    system_prompt = _load_system_prompt()
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if history:
        for entry in history:
            messages.append({"role": "user", "content": entry})
    messages.append({"role": "user", "content": prompt})

    client = get_openai_client()
    is_first_turn = _is_first_turn(history, session_id)

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        user=session_id,
    )

    choice = response.choices[0].message.content
    if not choice:
        raise OpenAIClientError("La respuesta de OpenAI no contiene texto interpretable.")
    cleaned = _clean_reply(choice)
    if is_first_turn:
        cleaned = _prefix_intro(cleaned)
    return cleaned


def _clean_reply(raw_text: str) -> str:
    """Normaliza saltos de línea y elimina marcado básico."""
    text = raw_text.strip()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{1,}", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = _remove_emojis(text)
    return text.strip()


def _remove_emojis(text: str) -> str:
    # Rango general de emojis y pictogramas
    emoji_pattern = re.compile(
        "["
        "\U0001F300-\U0001F5FF"
        "\U0001F600-\U0001F64F"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text)


def _is_first_turn(history: Sequence[str] | None, session_id: str | None) -> bool:
    if history:
        return False
    if not session_id:
        return _mark_default_intro()
    with _SESSION_GREETING_LOCK:
        if session_id in _SESSION_GREETING_CACHE:
            return False
        _SESSION_GREETING_CACHE.add(session_id)
        return True


def _mark_default_intro() -> bool:
    global _DEFAULT_INTRO_SENT
    with _DEFAULT_INTRO_LOCK:
        if _DEFAULT_INTRO_SENT:
            return False
        _DEFAULT_INTRO_SENT = True
        return True


def _prefix_intro(reply: str) -> str:
    normalized = reply.lower()
    if normalized.startswith("hola, soy chapi"):
        return reply
    return f"{INTRO_TEXT} {reply}".strip()



