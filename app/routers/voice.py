from fastapi import APIRouter, File, UploadFile
from fastapi.responses import StreamingResponse

from app.services.gemini_client import chat_with_gemini
from app.services.stt_client import audio_to_text
from app.services.tts_client import text_to_audio

router = APIRouter(prefix="/api", tags=["voice"])


@router.post("/voice")
async def voice_interaction(audio: UploadFile = File(...)) -> StreamingResponse:
    audio_bytes = await audio.read()
    user_text = audio_to_text(
        audio_bytes,
        filename=audio.filename,
        mime_type=audio.content_type,
    )
    reply_text = chat_with_gemini(user_text)
    reply_audio = text_to_audio(reply_text)

    return StreamingResponse(
        iter([reply_audio]),
        media_type="audio/wav",
        headers={
            "Content-Disposition": 'attachment; filename="reply.wav"',
        },
    )


