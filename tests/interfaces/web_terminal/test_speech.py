"""Tests for the Simple-view chat's speech-to-text (speech.py, routes/speech.py).

Whisper itself is never loaded here: a stand-in model returns a fixed result,
so these run without Whisper, PyTorch or ffmpeg installed. What is pinned:
availability is reported honestly; probably-silent segments are dropped;
the clip exists only during transcription, whatever happens; the routes
answer with the transcript or a clear error and never store audio.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal import speech
from osprey.interfaces.web_terminal.routes import speech as speech_routes


class FakeModel:
    """Stands in for a Whisper model; records the files it was given."""

    def __init__(self, result=None, error: Exception | None = None):
        self.result = result or {"text": " read the beam current", "segments": []}
        self.error = error
        self.paths: list[str] = []
        self.existed: list[bool] = []
        self.options: dict = {}

    def transcribe(self, path, **options):
        self.paths.append(path)
        self.existed.append(Path(path).exists())
        self.options = options
        if self.error:
            raise self.error
        return self.result


@pytest.fixture
def available(monkeypatch):
    """Pretend Whisper and ffmpeg are installed."""
    monkeypatch.setattr(speech, "status", lambda: speech.SpeechStatus(True, speech.model_name()))


def transcriber_with(model: FakeModel) -> speech.Transcriber:
    t = speech.Transcriber("base.en")
    t._model = model
    return t


# ---- availability ----------------------------------------------------------


def test_status_reports_missing_whisper(monkeypatch):
    monkeypatch.setattr(speech.importlib.util, "find_spec", lambda name: None)
    current = speech.status()
    assert current.available is False
    assert "Whisper" in current.reason


def test_status_reports_missing_ffmpeg(monkeypatch):
    monkeypatch.setattr(speech.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(speech.shutil, "which", lambda name: None)
    current = speech.status()
    assert current.available is False
    assert "ffmpeg" in current.reason


def test_status_available_when_both_present(monkeypatch):
    monkeypatch.setattr(speech.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(speech.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    assert speech.status().available is True


def test_model_defaults_to_small_en_and_can_be_overridden(monkeypatch):
    monkeypatch.delenv("OSPREY_SPEECH_MODEL", raising=False)
    assert speech.model_name() == "small.en"
    monkeypatch.setenv("OSPREY_SPEECH_MODEL", "base.en")
    assert speech.model_name() == "base.en"


# ---- spoken_text -----------------------------------------------------------


def test_spoken_text_drops_probably_silent_segments():
    result = {
        "text": "read the beam current Thank you.",
        "segments": [
            {"text": " read the beam current", "no_speech_prob": 0.05},
            {"text": " Thank you.", "no_speech_prob": 0.92},
        ],
    }
    assert speech.spoken_text(result) == "read the beam current"


def test_spoken_text_without_segments_uses_the_text():
    assert speech.spoken_text({"text": "  hello  "}) == "hello"


def test_spoken_text_of_silence_is_empty():
    result = {"text": " you", "segments": [{"text": " you", "no_speech_prob": 0.95}]}
    assert speech.spoken_text(result) == ""


# ---- transcription ---------------------------------------------------------


def test_transcribe_uses_the_conservative_settings(available):
    model = FakeModel()
    text = asyncio.run(transcriber_with(model).transcribe(b"audio", ".webm"))
    assert text == "read the beam current"
    assert model.options == {
        "language": "en",
        "fp16": False,
        "temperature": 0.0,
        "condition_on_previous_text": False,
    }


def test_the_clip_exists_only_during_transcription(available):
    model = FakeModel()
    asyncio.run(transcriber_with(model).transcribe(b"audio", ".webm"))
    assert model.existed == [True]
    assert model.paths[0].endswith(".webm")
    assert not Path(model.paths[0]).exists()


def test_the_clip_is_deleted_even_when_transcription_fails(available):
    model = FakeModel(error=RuntimeError("ffmpeg failed"))
    with pytest.raises(RuntimeError):
        asyncio.run(transcriber_with(model).transcribe(b"audio", ".webm"))
    assert not Path(model.paths[0]).exists()


def test_transcribe_refuses_when_unavailable(monkeypatch):
    monkeypatch.setattr(speech, "status", lambda: speech.SpeechStatus(False, "base.en", "no"))
    with pytest.raises(speech.SpeechUnavailable):
        asyncio.run(transcriber_with(FakeModel()).transcribe(b"audio"))


# ---- web.voice -------------------------------------------------------------


def config_says(monkeypatch, config):
    """The deployment's config is *config* (or raises it, if an exception)."""

    def load():
        if isinstance(config, Exception):
            raise config
        return config

    monkeypatch.setattr("osprey_connectors.workspace.load_osprey_config", load)


def test_voice_defaults_to_the_button_and_ctrl_space(monkeypatch):
    config_says(monkeypatch, {})
    assert speech.voice_settings() == speech.VoiceSettings("button", "Ctrl+Space")


@pytest.mark.parametrize("mode", ["mic", "key", "button", "off"])
def test_voice_mode_is_read_from_web_voice(monkeypatch, mode):
    config_says(monkeypatch, {"web": {"voice": {"mode": mode}}})
    assert speech.voice_settings().mode == mode


def test_voice_mode_is_read_without_regard_to_case(monkeypatch):
    config_says(monkeypatch, {"web": {"voice": {"mode": " Mic "}}})
    assert speech.voice_settings().mode == "mic"


def test_an_unknown_voice_mode_warns_and_falls_back_to_the_button(monkeypatch, caplog):
    config_says(monkeypatch, {"web": {"voice": {"mode": "always-on"}}})
    assert speech.voice_settings().mode == "button"
    assert "always-on" in caplog.text


def test_the_voice_key_is_read_from_web_voice(monkeypatch):
    config_says(monkeypatch, {"web": {"voice": {"mode": "key", "key": "Alt+M"}}})
    assert speech.voice_settings() == speech.VoiceSettings("key", "Alt+M")


@pytest.mark.parametrize(
    "config", [RuntimeError("no config.yml"), {"web": {"voice": "mic"}}, {"web": {"voice": None}}]
)
def test_an_unreadable_or_malformed_web_voice_gives_the_defaults(monkeypatch, config):
    config_says(monkeypatch, config)
    assert speech.voice_settings() == speech.VoiceSettings()


# ---- routes ----------------------------------------------------------------


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(speech_routes.router)
    return TestClient(app)


def test_status_route(client, monkeypatch):
    monkeypatch.setattr(
        speech, "status", lambda: speech.SpeechStatus(False, "base.en", "ffmpeg is missing.")
    )
    monkeypatch.setattr(speech, "voice_settings", lambda: speech.VoiceSettings("mic", "F8"))
    assert client.get("/api/speech/status").json() == {
        "available": False,
        "model": "base.en",
        "reason": "ffmpeg is missing.",
        "mode": "mic",
        "key": "F8",
    }


def test_transcribe_route_returns_the_text(client, available, monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(speech, "get_transcriber", lambda: transcriber_with(model))
    res = client.post(
        "/api/speech/transcribe",
        content=b"audio",
        headers={"Content-Type": "audio/ogg;codecs=opus"},
    )
    assert res.status_code == 200
    assert res.json() == {"text": "read the beam current"}
    assert model.paths[0].endswith(".ogg")


def test_transcribe_route_rejects_an_empty_clip(client, available):
    res = client.post("/api/speech/transcribe", content=b"", headers={"Content-Type": "audio/webm"})
    assert res.status_code == 400


def test_transcribe_route_rejects_an_oversized_clip(client, available):
    too_big = b"x" * (speech.MAX_CLIP_BYTES + 1)
    res = client.post(
        "/api/speech/transcribe", content=too_big, headers={"Content-Type": "audio/webm"}
    )
    assert res.status_code == 413


def test_transcribe_route_says_when_unavailable(client, monkeypatch):
    monkeypatch.setattr(
        speech, "status", lambda: speech.SpeechStatus(False, "base.en", "Whisper is not installed.")
    )
    res = client.post(
        "/api/speech/transcribe", content=b"a", headers={"Content-Type": "audio/webm"}
    )
    assert res.status_code == 503
    assert res.json() == {"error": "Whisper is not installed."}


def test_transcribe_route_hides_internal_errors(client, available, monkeypatch):
    model = FakeModel(error=RuntimeError("secret internal detail"))
    monkeypatch.setattr(speech, "get_transcriber", lambda: transcriber_with(model))
    res = client.post(
        "/api/speech/transcribe", content=b"a", headers={"Content-Type": "audio/webm"}
    )
    assert res.status_code == 500
    assert "secret" not in res.text
