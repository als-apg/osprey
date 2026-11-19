"""Speech-to-text routes for the Simple-view chat's hold-to-talk microphone.

``GET /api/speech/status`` tells the page whether voice input can work here
(the mic button is shown only when it can) and how the operator talks to it
(``web.voice.inputs`` and ``web.voice.key``). ``POST /api/speech/transcribe``
takes one recorded clip as the raw request body and answers ``{"text": ...}``
for the operator to review in the input box; nothing is sent to the agent from
here. See :mod:`osprey.interfaces.web_terminal.speech`.

Both routes sit behind the web terminal's own authentication like every other
route; neither stores audio.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from osprey.interfaces.web_terminal import speech

logger = logging.getLogger(__name__)

router = APIRouter()

#: Recording formats browsers produce, by the file suffix ffmpeg reads them by.
_SUFFIXES = {
    "audio/webm": ".webm",
    "audio/ogg": ".ogg",
    "audio/mp4": ".m4a",
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
}


def _error(status_code: int, message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status_code)


@router.get("/api/speech/status")
async def speech_status() -> dict:
    """Whether voice input can work here, which model it uses, and if not,
    why; and the voice inputs and key the deployment configures."""
    current = speech.status()
    voice = speech.voice_settings()
    return {
        "available": current.available,
        "model": current.model,
        "reason": current.reason,
        "inputs": list(voice.inputs),
        "key": voice.key,
    }


@router.post("/api/speech/transcribe")
async def speech_transcribe(request: Request) -> JSONResponse:
    """Transcribe one recorded clip (the raw request body) to text."""
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > speech.MAX_CLIP_BYTES:
        return _error(413, "That recording is too long. Keep voice prompts under two minutes.")
    audio = await request.body()
    if len(audio) > speech.MAX_CLIP_BYTES:
        return _error(413, "That recording is too long. Keep voice prompts under two minutes.")
    if not audio:
        return _error(400, "No audio was received.")

    media_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
    suffix = _SUFFIXES.get(media_type, ".webm")
    try:
        text = await speech.get_transcriber().transcribe(audio, suffix=suffix)
    except speech.SpeechUnavailable as exc:
        return _error(503, str(exc))
    except Exception:
        logger.exception("Speech transcription failed")
        return _error(500, "The recording could not be transcribed. Try again.")
    return JSONResponse({"text": text})
