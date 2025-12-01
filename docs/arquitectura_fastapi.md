# Arquitectura de FastAPI para interfaz Robot ↔ Nube

## 1. Vista general centrada en FastAPI

```mermaid
flowchart LR
    A[Robot (Arduino)] -- audio --> B[/FastAPI<br>endpoint /voice/]
    B --> C[Servicio STT<br>(opcional, backend)]
    C --> D[OpenAI API<br>(texto -> texto)]
    D --> E[Servicio TTS<br>(opcional, backend)]
    E -- audio --> A
```

- FastAPI actúa como el “cerebro en la nube”: recibe audio crudo del robot, orquesta los servicios de IA y devuelve audio listo para reproducirse.
- Las integraciones STT/TTS pueden ser servicios reales (Google, Whisper, Azure, etc.) o stubs/mocks mientras se valida el flujo.

## 2. Responsabilidades del backend (FastAPI)

- **Orquestación**: coordinar STT → OpenAI → TTS (STT/TTS siguen usando Gemini por ahora).
- **Gestión de claves**: mantener `OPENAI_API_KEY`, `GEMINI_API_KEY` y cualquier secreto asociado a STT/TTS solo en el backend.
- **Conversación**: opcionalmente almacenar contexto por `session_id` para diálogos continuos.
- **Contrato estable**: exponer endpoints claros para el equipo de hardware.

### 2.1 Endpoints mínimos

| Método | Ruta        | Entrada                              | Salida                           | Uso principal |
|--------|-------------|--------------------------------------|----------------------------------|---------------|
| POST   | `/api/voice`| `audio` en `multipart/form-data`     | `audio/*` (bytes o archivo)      | Robot envía audio, recibe respuesta hablada. |
| POST   | `/api/text` | JSON `{"message": "...", "session_id": "..."}` | JSON `{"reply": "..."}` | Debug rápido o robots que ya realizan STT. |

### 2.2 Configuración segura de `OPENAI_API_KEY`

1. Copia/crea un archivo `.env` en la raíz del repositorio con:
   ```
   OPENAI_API_KEY=tu_clave_de_openai
   OPENAI_MODEL=gpt-4o-mini   # opcional
   ```
2. FastAPI ejecuta `load_dotenv()` al iniciar (ver `app/__init__.py`), por lo que `os.getenv("OPENAI_API_KEY")` queda disponible en `services/openai_client.py`.
3. `.env` está en `.gitignore` para que la clave no se suba al repositorio público.
4. Si además usarás el endpoint de voz, mantén en el mismo archivo la sección de Gemini (`GEMINI_API_KEY`, modelos STT/TTS, etc.) descrita más abajo.

### 2.3 Plantilla de contexto (system prompt)

- El archivo `prompts/system_prompt.txt` define el comportamiento del chatbot (por defecto: apoyo emocional a personas solitarias).
- Puedes sobrescribir la ruta con `OPENAI_SYSTEM_PROMPT_FILE=/ruta/a/tu_prompt.txt`.
- Cualquier actualización en ese archivo se aprovecha automáticamente, gracias a que el cliente lo carga una sola vez al iniciar el proceso.
- La primera respuesta de cada sesión incluye automáticamente la presentación “Hola, soy Chapi, tu compañero de apoyo emocional.”; se controla por `session_id` (en memoria) para no repetirla (si no envías `session_id`, solo se mostrará una vez por reinicio del servidor).

### 2.4 Integración de Gemini para STT/TTS

1. La única credencial obligatoria es `GEMINI_API_KEY`.
2. STT usa `gemini-2.5-flash` (configurable con `GEMINI_STT_MODEL` y `GEMINI_STT_PROMPT`); TTS usa `gemini-2.5-flash-preview-tts` (puedes sobrescribirlo con `GEMINI_TTS_MODEL` y ajustar la voz con `GEMINI_TTS_VOICE`, ej. `Puck`, `Sprout`, `Charon`).
3. Instala dependencias: `pip install -r requirements.txt` (solo `google-genai`, `audioop-lts`, etc.).
4. Si por otros motivos necesitas credenciales de servicio (p. ej. acceso a otras APIs de Google), define `GOOGLE_APPLICATION_CREDENTIALS` o `GOOGLE_APPLICATION_CREDENTIALS_JSON`; `app/__init__.py` generará un archivo temporal en `.runtime/`.

## 3. Diseño interno propuesto

```
app/
├─ main.py                 # Crea FastAPI y monta routers
├─ routers/
│  ├─ voice.py             # /api/voice
│  └─ text.py              # /api/text
└─ services/
   ├─ openai_client.py     # Cliente oficial OpenAI (endpoint texto)
   ├─ gemini_client.py     # Cliente google-genai (STT/TTS y voz)
   ├─ stt_client.py        # Speech-to-Text (real o dummy)
   └─ tts_client.py        # Text-to-Speech (real o dummy)
```

Separar routers y servicios facilita reemplazar mocks por implementaciones reales sin tocar las rutas.

### 3.1 Cliente OpenAI (SDK oficial)

```python
# services/openai_client.py
import os
from openai import OpenAI

API_KEY = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=API_KEY)

def chat_with_openai(prompt: str, history: list[str] | None = None) -> str:
    messages = [{"role": "user", "content": msg} for msg in (history or [])]
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=messages,
    )
    return response.choices[0].message.content.strip()
```

- Instalar con `pip install -U openai`.
- `history` sigue opcional para compartir contexto si se guarda el historial por `session_id`.

### 3.2 Cliente Gemini (google-genai)

```python
# services/gemini_client.py
import os
from google import genai

API_KEY = os.environ["GEMINI_API_KEY"]
client = genai.Client(api_key=API_KEY)

def chat_with_gemini(prompt: str, history: list[str] | None = None) -> str:
    contents = history[:] if history else []
    contents.append(prompt)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
    )
    return response.text
```

- Instalar con `pip install -U google-genai`.
- `history` permite conservar contexto por `session_id` si se almacena en Redis/BD.

### 3.3 Endpoint de texto (pruebas rápidas)

```python
# routers/text.py
from fastapi import APIRouter
from pydantic import BaseModel
from services.openai_client import chat_with_openai

router = APIRouter(prefix="/api", tags=["text"])

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None

class ChatResponse(BaseModel):
    reply: str

@router.post("/text", response_model=ChatResponse)
def chat(req: ChatRequest):
    reply = chat_with_openai(req.message, session_id=req.session_id)
    return ChatResponse(reply=reply)
```

- Ideal para Postman/cURL.
- Simplifica validar Gemini antes de integrar audio.

### 3.4 Endpoint de voz

```python
# routers/voice.py
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse
from services.gemini_client import chat_with_gemini
from services.stt_client import audio_to_text
from services.tts_client import text_to_audio

router = APIRouter(prefix="/api", tags=["voice"])

@router.post("/voice")
async def voice_interaction(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    user_text = audio_to_text(
        audio_bytes,
        filename=audio.filename,
        mime_type=audio.content_type,
    )                                               # STT (Gemini)
    reply_text = chat_with_gemini(user_text)        # Gemini
    reply_audio_bytes = text_to_audio(reply_text)   # TTS (Gemini)

    return StreamingResponse(
        iter([reply_audio_bytes]),
        media_type="audio/wav",
    )
```

- En `app/services/` ya se usa Gemini para ambos pasos: STT (`gemini-2.5-flash`)
  y TTS (`gemini-2.5-flash-preview-tts`). Configura `GEMINI_API_KEY` y, si
  deseas personalizar modelos/voces/prompts, ajusta las variables descritas en
  `.env.example`.

## 4. Contrato con el equipo de Arduino

### /api/voice
- **Método**: `POST`
- **Content-Type**: `multipart/form-data`
- **Campo requerido**: `audio` (archivo `.wav/.mp3/.pcm`, idealmente mono, 16kHz).
- **Respuesta**: `audio/wav` (stream listo para reproducir).

### /api/text
- **Método**: `POST`
- **Body**:
  ```json
  {
    "message": "Hola, ¿quién eres?",
    "session_id": "robot_1"
  }
  ```
- **Respuesta**:
  ```json
  {
    "reply": "Hola, soy tu robot asistente."
  }
  ```

Con este contrato, el equipo de hardware sabe cómo empaquetar audio y qué esperar de regreso, mientras que FastAPI concentra toda la lógica de IA y seguridad.

## 5. Próximos pasos sugeridos

1. Inicializar proyecto (`poetry` o `pip`) e instalar `fastapi`, `uvicorn[standard]`, `google-genai`, `python-multipart`.
2. Crear estructura `app/main.py`, `routers`, `services`.
3. Implementar `stt_client`/`tts_client` como mocks y validar `/api/voice` con archivos de prueba.
4. Conectar servicios reales de STT/TTS y añadir almacenamiento de contexto si se requiere historial por robot.
5. Automatizar despliegue (Docker + Render/Fly.io/Azure) para que el robot apunte a un endpoint fijo en la nube.

Con esta arquitectura, FastAPI se mantiene modular, extensible y lista para evolucionar desde prototipos con mocks hasta producción con servicios de voz reales.


