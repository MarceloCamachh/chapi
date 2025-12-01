from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services.openai_client import chat_with_openai

router = APIRouter(prefix="/api", tags=["text"])


class ChatRequest(BaseModel):
    message: str = Field(..., description="Texto capturado por el robot o un cliente de prueba.")
    session_id: str | None = Field(
        default=None,
        description="Identificador opcional para mantener contexto por robot.",
    )


class ChatResponse(BaseModel):
    reply: str


@router.post("/text", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    reply = chat_with_openai(req.message, session_id=req.session_id)
    return ChatResponse(reply=reply)


