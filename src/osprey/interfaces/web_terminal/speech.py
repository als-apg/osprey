"""Speech-to-text for the Simple-view chat's hold-to-talk microphone.

The browser records a short clip while the operator holds the mic button and
posts it here; this module turns it into text with a locally run Whisper model
(OpenAI's open-source speech recognition, MIT licence). The text goes back to
the chat's input box for the operator to review and send: nothing transcribed
here is ever acted on directly.

Optional by design. Whisper (and the PyTorch it runs on) is not one of
Osprey's dependencies, and it reads audio through the ``ffmpeg`` command-line
tool. When either is missing, :func:`status` says why and the chat simply
hides its mic button.

Privacy: a clip exists only as a temporary file for the length of one
transcription and is deleted straight after, whatever happens. No audio is
kept.

Accuracy: Whisper can produce text that was never spoken, mostly from silence
or noise. Segments it judges probably silent are dropped, decoding is greedy
(temperature 0) and does not carry earlier text forward, which keeps that
rare; the operator's review before sending is the real safeguard.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import shutil
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: The Whisper model used when ``OSPREY_SPEECH_MODEL`` is unset: English-only.
#: Tested against ``base.en`` at a console: about a second slower per clip on
#: a CPU, but clearly better with background noise and with the operator
#: leaning back or facing away. Set ``OSPREY_SPEECH_MODEL=base.en`` for speed.
DEFAULT_MODEL = "small.en"

#: How the operator talks to the chat (``web.voice.mode``): ``mic`` toggles
#: listening for a push-to-talk microphone whose button gates the sound,
#: ``key`` holds a keyboard key, ``button`` holds the mic icon, ``off`` hides
#: voice input.
VOICE_MODES = ("mic", "key", "button", "off")
DEFAULT_MODE = "button"

#: The key held to talk in ``key`` mode (``web.voice.key``): modifiers and a
#: key joined by ``+``, parsed by the browser.
DEFAULT_KEY = "Ctrl+Space"

#: Segments Whisper rates at least this likely to be silence are dropped.
NO_SPEECH_DROP = 0.6

#: The longest clip accepted, in bytes (compressed browser audio; roughly two
#: minutes of speech).
MAX_CLIP_BYTES = 2 * 1024 * 1024


def model_name() -> str:
    """The Whisper model to load: ``OSPREY_SPEECH_MODEL``, or the default."""
    return os.environ.get("OSPREY_SPEECH_MODEL", "").strip() or DEFAULT_MODEL


@dataclass(frozen=True)
class VoiceSettings:
    """How the chat offers voice input (``web.voice``)."""

    mode: str = DEFAULT_MODE
    key: str = DEFAULT_KEY


def voice_settings() -> VoiceSettings:
    """``web.voice.mode`` and ``web.voice.key`` from the deployment's config.

    An unknown mode warns and falls back to the default, and an unreadable
    config gives the defaults: a settings mistake never blocks the chat.
    """
    try:
        from osprey_connectors.workspace import load_osprey_config

        voice = load_osprey_config().get("web", {}).get("voice", {})
    except Exception:
        logger.warning("Could not read web.voice; using the defaults", exc_info=True)
        voice = {}
    if not isinstance(voice, dict):
        voice = {}
    mode = str(voice.get("mode", DEFAULT_MODE)).strip().lower()
    if mode not in VOICE_MODES:
        logger.warning(
            "web.voice.mode %r is not one of %s; using %r",
            mode,
            ", ".join(VOICE_MODES),
            DEFAULT_MODE,
        )
        mode = DEFAULT_MODE
    key = str(voice.get("key", DEFAULT_KEY)).strip() or DEFAULT_KEY
    return VoiceSettings(mode, key)


@dataclass(frozen=True)
class SpeechStatus:
    """Whether voice input can work here, and if not, why."""

    available: bool
    model: str
    reason: str = ""


def status() -> SpeechStatus:
    """Report whether Whisper and ffmpeg are installed, without loading a model."""
    model = model_name()
    if importlib.util.find_spec("whisper") is None:
        return SpeechStatus(False, model, "Whisper is not installed on the server.")
    if shutil.which("ffmpeg") is None:
        return SpeechStatus(False, model, "ffmpeg is not installed on the server.")
    return SpeechStatus(True, model)


class SpeechUnavailable(RuntimeError):
    """Voice input cannot work here (Whisper or ffmpeg missing)."""


class Transcriber:
    """Loads one Whisper model on first use and transcribes one clip at a time.

    Loading takes a few seconds and a model holds about a gigabyte, so it is
    loaded once and kept. Transcription runs on a worker thread (it is CPU
    bound) under a lock: on a CPU, two clips at once would each be slower than
    the two in turn.
    """

    def __init__(self, name: str | None = None) -> None:
        self._name = name or model_name()
        self._model: Any = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        if self._model is None:
            import whisper  # optional dependency; checked by status()

            logger.info("Loading Whisper model %s", self._name)
            self._model = whisper.load_model(self._name)
        return self._model

    def transcribe_file(self, path: str) -> str:
        """Transcribe the audio file at *path* to text (blocking)."""
        with self._lock:
            model = self._load()
            result = model.transcribe(
                path,
                language="en",
                fp16=False,
                temperature=0.0,
                condition_on_previous_text=False,
            )
        return spoken_text(result)

    async def transcribe(self, audio: bytes, suffix: str = ".webm") -> str:
        """Transcribe a recorded clip. The audio exists only for this call."""
        current = status()
        if not current.available:
            raise SpeechUnavailable(current.reason)
        fd, path = tempfile.mkstemp(prefix="osprey-speech-", suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(audio)
            return await asyncio.to_thread(self.transcribe_file, path)
        finally:
            Path(path).unlink(missing_ok=True)


def spoken_text(result: dict[str, Any]) -> str:
    """The text of a Whisper result, without segments it judged probably silent."""
    segments = result.get("segments")
    if not segments:
        return str(result.get("text", "")).strip()
    kept = [
        str(segment.get("text", "")).strip()
        for segment in segments
        if float(segment.get("no_speech_prob", 0.0)) < NO_SPEECH_DROP
    ]
    return " ".join(part for part in kept if part).strip()


_transcriber: Transcriber | None = None


def get_transcriber() -> Transcriber:
    """The process-wide transcriber (one model in memory, shared by every user)."""
    global _transcriber
    if _transcriber is None:
        _transcriber = Transcriber()
    return _transcriber
