"""Inicializa el paquete principal de la aplicación FastAPI."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

from dotenv import load_dotenv


# Carga variables sensibles desde .env para que os.getenv las encuentre.
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)


def _ensure_google_credentials_file() -> None:
    """
    Permite desplegar usando GOOGLE_APPLICATION_CREDENTIALS_JSON.

    En entornos como Render/Vercel solo se puede almacenar el JSON como texto o
    base64. Este helper crea un archivo temporal y apunta
    GOOGLE_APPLICATION_CREDENTIALS hacia él si aún no existe.
    """
    current_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    raw_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if current_path or not raw_json:
        return

    data = raw_json.strip()
    decoded_text = data
    if not data.startswith("{"):
        try:
            decoded_bytes = base64.b64decode(data, validate=True)
            decoded_text = decoded_bytes.decode("utf-8")
        except Exception:
            decoded_text = data

    try:
        json.loads(decoded_text)
    except json.JSONDecodeError as exc:  # pragma: no cover - validación simple
        raise RuntimeError(
            "GOOGLE_APPLICATION_CREDENTIALS_JSON no contiene JSON válido."
        ) from exc

    runtime_dir = BASE_DIR / ".runtime"
    runtime_dir.mkdir(exist_ok=True)
    target_file = runtime_dir / "google-credentials.json"
    target_file.write_text(decoded_text, encoding="utf-8")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(target_file)


_ensure_google_credentials_file()
