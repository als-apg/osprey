"""Automatic session names: the naming call and the hub's scheduler.

``name_session`` is driven with a stubbed ``aget_chat_completion`` (success,
timeout, exception, empty reply, provider and model resolution). The scheduler
is driven with a stubbed namer against the real session meta: de-duplication and one naming in flight; pre-cutoff,
truncated and bookkeeping-only rows never named; a failure retried once per hub
start; ``pending`` held in memory only; one ``session_named`` frame per result;
and ``GET /api/sessions`` answering while the model has not.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.agent_record.client import ClientAnswer
from osprey.agent_runner import primitives
from osprey.interfaces.web_terminal import session_meta
from osprey.interfaces.web_terminal.routes import session as session_routes
from osprey.models import completion
from osprey.models import config as model_config
from osprey.services.session_naming import (
    MAX_TITLE_CHARS,
    NAMING_MAX_TOKENS,
    PROMPT_CHARS,
    SESSION_NAMED_FRAME,
    SessionNamer,
    name_session,
    trim_title,
)
from osprey_connectors import posture_store
from tests.agent_record.fakes import FakeRecordClient

ME = "alice"
T1 = "aaaaaaaa-1111-4222-8333-444444444444"
T2 = "bbbbbbbb-1111-4222-8333-444444444444"
T3 = "cccccccc-1111-4222-8333-444444444444"
SINCE = "2026-11-01T00:00:00+00:00"
AFTER = "2026-11-02T09:00:00+00:00"
BEFORE = "2026-10-31T09:00:00+00:00"


class RecordingBroadcaster:
    def __init__(self) -> None:
        self.frames: list[dict] = []

    def broadcast(self, data: dict, *, reliable: bool = False) -> None:  # noqa: ARG002
        self.frames.append(data)


@pytest.fixture
def state_root(tmp_path, monkeypatch):
    root = tmp_path / "shared_agent_data"
    root.mkdir()
    monkeypatch.delenv(posture_store.CONTROL_CONTEXT_DIR_ENV_VAR, raising=False)
    monkeypatch.setenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, str(root))
    posture_store.invalidate_cache()
    yield root
    posture_store.invalidate_cache()


def _hub(since: str = SINCE) -> FastAPI:
    """A hub app with its session meta loaded and ``naming_since`` at *since*."""
    app = FastAPI()
    app.include_router(session_routes.router)
    app.state.terminal_user = ME
    app.state.broadcaster = RecordingBroadcaster()
    session_meta.load(app)
    app.state.session_meta["naming_since"] = since
    return app


@pytest.fixture
def app(state_root):  # noqa: ARG001
    return _hub()


def _row(tid: str, *, prompt: str = "tune the ring", first_ts: str = AFTER, **extra) -> dict:
    return {
        "transcript_id": tid,
        "title": prompt or "Untitled session",
        "prompt": prompt,
        "first_ts": first_ts,
        "truncated": False,
        **extra,
    }


class StubNamer:
    """Answers each prompt from *answers* (a value, an exception, or a callable)."""

    def __init__(self, answer: Any = "Ring tuning") -> None:
        self.answer = answer
        self.calls: list[tuple[str, Path]] = []
        self.active = 0
        self.most_active = 0

    async def __call__(self, prompt: str, *, project_dir: Path) -> str | None:
        self.calls.append((prompt, project_dir))
        self.active += 1
        self.most_active = max(self.most_active, self.active)
        try:
            await asyncio.sleep(0)
            answer = self.answer(prompt) if callable(self.answer) else self.answer
            if isinstance(answer, BaseException):
                raise answer
            return answer
        finally:
            self.active -= 1


def _namer(app, stub: StubNamer, tmp_path: Path) -> SessionNamer:
    return SessionNamer(app, project_dir=tmp_path / "render", namer=stub)


def _named_frames(app) -> list[dict]:
    return [f for f in app.state.broadcaster.frames if f.get("type") == SESSION_NAMED_FRAME]


# --- trimming --------------------------------------------------------------------------


class TestTrimTitle:
    @pytest.mark.parametrize(
        ("reply", "title"),
        [
            ("Ring orbit tuning", "Ring orbit tuning"),
            ('"Ring orbit tuning."', "Ring orbit tuning"),
            ("Title: **Ring orbit tuning**", "Ring orbit tuning"),
            ("\n\n  Ring orbit tuning  \nbecause the user asked", "Ring orbit tuning"),
            ("one two three four five six seven eight", "one two three four five six"),
            ("", None),
            ("   \n  ", None),
            ('"..."', None),
            (None, None),
        ],
    )
    def test_a_reply_becomes_a_short_title(self, reply, title):
        assert trim_title(reply) == title

    def test_a_long_reply_is_cut_at_a_word_within_the_character_cap(self):
        title = trim_title(" ".join(["superconducting"] * 6))
        assert title is not None and len(title) <= MAX_TITLE_CHARS
        assert title.split() == ["superconducting"] * len(title.split())

    def test_one_overlong_word_is_cut_to_the_cap(self):
        assert trim_title("x" * 200) == "x" * MAX_TITLE_CHARS


# --- name_session with a stubbed aget_chat_completion ---------------------------------


def _spec(**fields: Any) -> SimpleNamespace:
    base = {
        "provider": "gateway",
        "default_model_id": "main-model",
        "alias_models": {"haiku": "fast-model", "sonnet": "main-model", "opus": "big-model"},
    }
    return SimpleNamespace(**(base | fields))


@pytest.fixture
def spec(monkeypatch):
    """The provider spec the route is read from: provider ``gateway``, ``haiku`` → ``fast-model``."""
    specs: list[Path] = []

    def resolve(project_dir, **_k):
        specs.append(project_dir)
        return _spec()

    monkeypatch.setattr(primitives, "_resolve_project_spec", resolve)
    monkeypatch.setattr(
        model_config,
        "get_provider_config",
        lambda name, config_path=None: {"name": name, "config_path": config_path},
    )
    return specs


def _completion(monkeypatch, answer: Any) -> list[dict[str, Any]]:
    """Stub ``aget_chat_completion``: *answer* is returned, raised, or awaited."""
    calls: list[dict[str, Any]] = []

    async def fake(**kwargs):
        calls.append(kwargs)
        if isinstance(answer, BaseException):
            raise answer
        if callable(answer):
            return await answer()
        return answer

    monkeypatch.setattr(completion, "aget_chat_completion", fake)
    return calls


@pytest.mark.usefixtures("spec")
class TestNameSession:
    async def test_success_asks_the_providers_fast_model_once(self, monkeypatch, tmp_path, spec):
        calls = _completion(monkeypatch, '"Orbit correction in sector 3."')
        project = tmp_path / "render"

        title = await name_session("please   correct the orbit\nin sector 3", project_dir=project)

        assert title == "Orbit correction in sector 3"
        assert spec == [project]
        (kwargs,) = calls
        assert kwargs["provider"] == "gateway"
        assert kwargs["model_id"] == "fast-model"
        assert kwargs["provider_config"] == {
            "name": "gateway",
            "config_path": str(project / "config.yml"),
        }
        assert kwargs["max_tokens"] == NAMING_MAX_TOKENS
        system, user = kwargs["chat_request"].messages
        assert system.role == "system" and "name conversations" in system.content
        assert user.role == "user"
        assert user.content == "please correct the orbit in sector 3"

    async def test_the_prompt_is_cut_to_its_cap(self, monkeypatch, tmp_path):
        calls = _completion(monkeypatch, "Long one")
        await name_session("word " * 2000, project_dir=tmp_path)
        assert len(calls[0]["chat_request"].messages[1].content) == PROMPT_CHARS

    async def test_without_a_haiku_alias_the_main_model_names(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            primitives, "_resolve_project_spec", lambda *_a, **_k: _spec(alias_models={})
        )
        calls = _completion(monkeypatch, "Ring tuning")
        assert await name_session("tune", project_dir=tmp_path) == "Ring tuning"
        assert calls[0]["model_id"] == "main-model"

    async def test_without_a_provider_nothing_is_asked(self, monkeypatch, tmp_path, caplog):
        monkeypatch.setattr(primitives, "_resolve_project_spec", lambda *_a, **_k: None)
        calls = _completion(monkeypatch, "Ring tuning")
        assert await name_session("tune", project_dir=tmp_path) is None
        assert calls == []
        assert "no provider" in caplog.text

    async def test_a_resolution_error_returns_none_and_is_logged(
        self, monkeypatch, tmp_path, caplog
    ):
        def broken(*_a, **_k):
            raise FileNotFoundError("no config.yml")

        monkeypatch.setattr(primitives, "_resolve_project_spec", broken)
        assert await name_session("tune", project_dir=tmp_path) is None
        assert "no config.yml" in caplog.text

    async def test_a_timeout_returns_none_and_is_logged(self, monkeypatch, tmp_path, caplog):
        async def slow():
            await asyncio.sleep(10)

        _completion(monkeypatch, slow)
        started = time.monotonic()
        assert await name_session("tune", project_dir=tmp_path, timeout=0.05) is None
        assert time.monotonic() - started < 5
        assert "timed out" in caplog.text

    async def test_an_exception_returns_none_and_is_logged(self, monkeypatch, tmp_path, caplog):
        _completion(monkeypatch, ValueError("API key required for gateway"))
        assert await name_session("tune", project_dir=tmp_path) is None
        assert "API key required" in caplog.text

    @pytest.mark.parametrize("reply", [None, "", "  \n", '""', ["block"]])
    async def test_an_empty_or_unusable_reply_returns_none(
        self, monkeypatch, tmp_path, caplog, reply
    ):
        _completion(monkeypatch, reply)
        assert await name_session("tune", project_dir=tmp_path) is None
        assert "no usable title" in caplog.text

    async def test_an_empty_prompt_asks_nothing(self, monkeypatch, tmp_path):
        calls = _completion(monkeypatch, AssertionError("asked the model"))
        assert await name_session("  \n ", project_dir=tmp_path) is None
        assert calls == []


# --- the scheduler ---------------------------------------------------------------------


class TestScheduler:
    async def test_a_result_is_recorded_done_with_one_frame(self, app, tmp_path):
        stub = StubNamer("  Ring tuning in sector three of the ring today  ")
        namer = _namer(app, stub, tmp_path)

        assert namer.schedule([_row(T1)]) == [T1]
        assert T1 in app.state.session_naming_pending
        await namer.wait_idle()

        meta = session_meta.get(app, T1)
        assert meta["naming"] == "done"
        assert meta["auto_title"] == meta["title"] == "Ring tuning in sector three of"
        assert meta["title_source"] == "auto"
        assert app.state.session_naming_pending == set()
        assert _named_frames(app) == [
            {
                "type": SESSION_NAMED_FRAME,
                "transcript_id": T1,
                "title": "Ring tuning in sector three of",
                "naming": "done",
            }
        ]
        assert stub.calls == [("tune the ring", tmp_path / "render")]

    async def test_one_frame_per_result_and_one_naming_in_flight(self, app, tmp_path):
        stub = StubNamer(lambda prompt: f"About {prompt}")
        namer = _namer(app, stub, tmp_path)

        namer.schedule([_row(T1, prompt="a"), _row(T2, prompt="b"), _row(T3, prompt="c")])
        await namer.wait_idle()

        frames = _named_frames(app)
        assert sorted(f["transcript_id"] for f in frames) == [T1, T2, T3]
        assert {f["transcript_id"]: f["title"] for f in frames}[T2] == "About b"
        assert stub.most_active == 1

    async def test_a_row_is_scheduled_once_while_pending(self, app, tmp_path):
        release = asyncio.Event()

        async def held(prompt, *, project_dir):  # noqa: ARG001
            await release.wait()
            return "Ring tuning"

        namer = SessionNamer(app, project_dir=tmp_path, namer=held)
        assert namer.schedule([_row(T1), _row(T1)]) == [T1]
        await asyncio.sleep(0)
        assert namer.schedule([_row(T1)]) == []
        release.set()
        await namer.wait_idle()
        assert len(_named_frames(app)) == 1
        assert namer.schedule([_row(T1)]) == []  # done: never named again

    @pytest.mark.parametrize(
        "row",
        [
            _row(T1, first_ts=BEFORE),
            _row(T1, first_ts=None),
            _row(T1, truncated=True),
            _row(T1, prompt=""),
            _row(T1, prompt="<command-name>/clear</command-name>"),
            _row(T1, prompt="<local-command-stdout></local-command-stdout>"),
        ],
        ids=["pre-cutoff", "no-first-ts", "truncated", "promptless", "bookkeeping", "stdout"],
    )
    async def test_ineligible_rows_are_never_named(self, app, tmp_path, row):
        stub = StubNamer()
        namer = _namer(app, stub, tmp_path)
        assert namer.schedule([row]) == []
        await namer.wait_idle()
        assert stub.calls == [] and _named_frames(app) == []
        assert app.state.session_naming_pending == set()

    async def test_a_row_with_an_auto_title_is_not_named_again(self, app, tmp_path):
        session_meta.set_auto_title(app, T1, "Earlier name", "done")
        stub = StubNamer()
        assert _namer(app, stub, tmp_path).schedule([_row(T1)]) == []
        assert stub.calls == []

    async def test_a_deleted_session_is_never_named(self, app, tmp_path):
        session_meta.mark_deleted(app, T1)
        stub = StubNamer()
        assert _namer(app, stub, tmp_path).schedule([_row(T1)]) == []
        assert stub.calls == []

    @pytest.mark.parametrize(
        "answer", [None, "", RuntimeError("boom")], ids=["none", "blank", "raises"]
    )
    async def test_a_failure_is_recorded_and_retried_once_per_hub_start(
        self, app, tmp_path, answer
    ):
        stub = StubNamer(answer)
        namer = _namer(app, stub, tmp_path)
        namer.schedule([_row(T1)])
        await namer.wait_idle()

        assert session_meta.get(app, T1)["naming"] == "failed"
        assert _named_frames(app) == [
            {
                "type": SESSION_NAMED_FRAME,
                "transcript_id": T1,
                "title": "tune the ring",
                "naming": "failed",
            }
        ]
        # Not again during this hub start.
        assert namer.schedule([_row(T1)]) == []

        # The next hub start retries it once.
        await namer.shutdown()
        restarted = _hub()
        second = _namer(restarted, stub, tmp_path)
        assert second.schedule([_row(T1)]) == [T1]
        await second.wait_idle()
        assert second.schedule([_row(T1)]) == []
        assert len(stub.calls) == 2

    async def test_a_retry_that_succeeds_is_done(self, app, tmp_path):
        session_meta.set_auto_title(app, T1, None, "failed")
        stub = StubNamer("Ring tuning")
        namer = _namer(app, stub, tmp_path)
        assert namer.schedule([_row(T1)]) == [T1]
        await namer.wait_idle()
        assert session_meta.get(app, T1)["naming"] == "done"

    async def test_pending_does_not_survive_a_restart(self, app, state_root, tmp_path):  # noqa: ARG002
        async def never_answers(prompt, *, project_dir):  # noqa: ARG001
            await asyncio.sleep(3600)

        namer = SessionNamer(app, project_dir=tmp_path, namer=never_answers)
        namer.schedule([_row(T1), _row(T2)])
        await asyncio.sleep(0)
        assert app.state.session_naming_pending == {T1, T2}

        await namer.shutdown()
        assert app.state.session_naming_pending == set()
        assert _named_frames(app) == []
        on_disk = json.loads(session_meta.store_path().read_text("utf-8"))
        assert all(s.get("naming") is None for s in on_disk["sessions"].values())

        restarted = _hub()
        assert restarted.state.session_naming_pending == set()
        assert session_meta.get(restarted, T1)["naming"] is None
        stub = StubNamer()
        again = _namer(restarted, stub, tmp_path)
        assert sorted(again.schedule([_row(T1), _row(T2)])) == [T1, T2]
        await again.wait_idle()

    async def test_nothing_is_scheduled_once_shut_down(self, app, tmp_path):
        stub = StubNamer()
        namer = _namer(app, stub, tmp_path)
        await namer.shutdown()
        assert namer.schedule([_row(T1)]) == []


# --- the listing hook ------------------------------------------------------------------


@pytest.fixture
def record(monkeypatch):
    fake = FakeRecordClient()
    monkeypatch.setattr(session_routes, "record_client", lambda: fake)
    return fake


def _record_row(tid: str, prompt: str, first_ts: str = AFTER) -> dict:
    return {
        "session_id": tid,
        "persons": [ME],
        "identities": [ME],
        "first_ts": first_ts,
        "last_ts": "2026-11-02T10:00:00+00:00",
        "message_count": 2,
        "prompt": prompt,
        "truncated": False,
        "run_ids": [],
        "terminal_session": "",
    }


class TestListingHook:
    def test_the_listing_answers_before_the_model_and_marks_rows_pending(
        self, app, record, tmp_path
    ):
        record.answers["list_sessions"] = ClientAnswer(
            data=[
                _record_row(T1, "tune the ring"),
                _record_row(T2, "an old one", first_ts=BEFORE),
            ],
            complete=True,
        )
        release = threading.Event()
        calls: list[str] = []

        async def blocked(prompt, *, project_dir):  # noqa: ARG001
            calls.append(prompt)
            await asyncio.to_thread(release.wait, 10)
            return "Ring tuning"

        app.state.session_namer = SessionNamer(app, project_dir=tmp_path, namer=blocked)
        with TestClient(app) as client:
            started = time.monotonic()
            rows = {r["transcript_id"]: r for r in client.get("/api/sessions").json()["sessions"]}
            assert time.monotonic() - started < 5
            assert rows[T1]["naming"] == "pending"
            assert rows[T1]["title"] == "tune the ring"
            assert rows[T2]["naming"] is None

            again = {r["transcript_id"]: r for r in client.get("/api/sessions").json()["sessions"]}
            assert again[T1]["naming"] == "pending"

            release.set()
            client.portal.call(app.state.session_namer.wait_idle)
            assert calls == ["tune the ring"]

            named = {r["transcript_id"]: r for r in client.get("/api/sessions").json()["sessions"]}
            assert named[T1]["naming"] == "done"
            assert named[T1]["title"] == "Ring tuning"
            assert named[T1]["title_source"] == "auto"
            assert [f["transcript_id"] for f in _named_frames(app)] == [T1]

    def test_a_hub_without_a_namer_lists_as_before(self, app, record):
        record.answers["list_sessions"] = ClientAnswer(
            data=[_record_row(T1, "tune the ring")], complete=True
        )
        with TestClient(app) as client:
            row = client.get("/api/sessions").json()["sessions"][0]
        assert row["naming"] is None and row["title"] == "tune the ring"
