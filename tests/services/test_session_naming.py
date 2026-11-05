"""Automatic session names: the naming call, its isolation, and the hub's scheduler.

``name_session`` is driven with a stubbed ``run_query`` (success, timeout,
exception, trimming). The scheduler is driven with a stubbed namer against the
real session meta: de-duplication and one naming in flight; pre-cutoff,
truncated and bookkeeping-only rows never named; a failure retried once per hub
start; ``pending`` held in memory only; one ``session_named`` frame per result;
and ``GET /api/sessions`` answering while the model has not. The isolation
tests run the real ``run_query`` and option builder against a stand-in CLI at
the SDK client seam that files a transcript and a record row the way the CLI
does, so what the options say is checked by what a run leaves behind.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.agent_record.client import ClientAnswer
from osprey.agent_record.contract import RECORD_OFF_ENV, RECORD_OFF_VALUE, SPOOL_DIR_ENV
from osprey.agent_runner import primitives, runner
from osprey.agent_runner.project_paths import claude_project_dir
from osprey.agent_runner.tool_names import READ_ONLY_DENIED_BUILTINS
from osprey.interfaces.web_terminal import session_meta
from osprey.interfaces.web_terminal.routes import session as session_routes
from osprey.services import session_naming
from osprey.services.session_naming import (
    MAX_TITLE_CHARS,
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


# --- name_session with a stubbed run_query ---------------------------------------------


@pytest.fixture
def spec(monkeypatch):
    """The provider spec the naming model is read from: ``haiku`` → ``fast-model``."""
    monkeypatch.setattr(
        primitives,
        "_resolve_project_spec",
        lambda *_a, **_k: SimpleNamespace(alias_models={"haiku": "fast-model"}, needs_proxy=False),
    )


def _run(text: str | None, *, is_error: bool = False) -> SimpleNamespace:
    result = SimpleNamespace(is_error=is_error, result=text, subtype="success")
    return SimpleNamespace(result=result, text_blocks=[text] if text else [])


@pytest.mark.usefixtures("spec")
class TestNameSession:
    async def test_success_asks_the_haiku_alias_once_without_tools_settings_or_servers(
        self,
        monkeypatch,
        tmp_path,
    ):
        seen: list[tuple[Path, str, dict]] = []

        async def fake_run_query(project_dir, prompt, **kwargs):
            seen.append((project_dir, prompt, kwargs))
            assert Path(kwargs["cwd"]).is_dir()
            return _run('"Orbit correction in sector 3."')

        monkeypatch.setattr(runner, "run_query", fake_run_query)
        project = tmp_path / "render"

        title = await name_session("please   correct the orbit\nin sector 3", project_dir=project)

        assert title == "Orbit correction in sector 3"
        ((project_dir, prompt, kwargs),) = seen
        assert project_dir == project
        assert prompt == "please correct the orbit in sector 3"
        assert kwargs["model"] == "fast-model"
        assert kwargs["tools"] == []
        assert kwargs["setting_sources"] == []
        assert kwargs["mcp_servers"] == {}
        assert kwargs["await_mcp_servers"] == ()
        assert kwargs["max_turns"] == 1
        assert kwargs["disallowed_tools"] == READ_ONLY_DENIED_BUILTINS
        assert 0 < kwargs["max_budget_usd"] <= 0.1
        assert "env" not in kwargs
        assert kwargs["env_overrides"][RECORD_OFF_ENV] == RECORD_OFF_VALUE
        work_dir = Path(kwargs["cwd"])
        assert work_dir != project and not work_dir.exists()

    async def test_without_a_haiku_alias_the_cli_maps_the_alias_itself(self, monkeypatch, tmp_path):
        monkeypatch.setattr(primitives, "_resolve_project_spec", lambda *_a, **_k: None)
        models: list[str] = []

        async def fake_run_query(project_dir, prompt, **kwargs):  # noqa: ARG001
            models.append(kwargs["model"])
            return _run("Ring tuning")

        monkeypatch.setattr(runner, "run_query", fake_run_query)
        assert await name_session("tune", project_dir=tmp_path) == "Ring tuning"
        assert models == ["haiku"]

    async def test_a_timeout_returns_none_and_is_logged(self, monkeypatch, tmp_path, caplog):
        cwds: list[Path] = []

        async def slow(project_dir, prompt, **kwargs):  # noqa: ARG001
            cwds.append(Path(kwargs["cwd"]))
            await asyncio.sleep(10)

        monkeypatch.setattr(runner, "run_query", slow)
        started = time.monotonic()
        assert await name_session("tune", project_dir=tmp_path, timeout=0.05) is None
        assert time.monotonic() - started < 5
        assert "timed out" in caplog.text
        assert cwds and not cwds[0].exists()

    async def test_an_exception_returns_none_and_is_logged(self, monkeypatch, tmp_path, caplog):
        async def broken(*_a, **_k):
            raise RuntimeError("SDK query failed: no route")

        monkeypatch.setattr(runner, "run_query", broken)
        assert await name_session("tune", project_dir=tmp_path) is None
        assert "no route" in caplog.text

    @pytest.mark.parametrize("run", [_run(None), _run("Name", is_error=True), _run("  \n")])
    async def test_an_empty_or_failed_reply_returns_none(self, monkeypatch, tmp_path, run):
        async def answer(*_a, **_k):
            return run

        monkeypatch.setattr(runner, "run_query", answer)
        assert await name_session("tune", project_dir=tmp_path) is None

    async def test_an_empty_prompt_asks_nothing(self, monkeypatch, tmp_path):
        async def never(*_a, **_k):
            raise AssertionError("asked the model")

        monkeypatch.setattr(runner, "run_query", never)
        assert await name_session("  \n ", project_dir=tmp_path) is None


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


# --- isolation: the real run_query and option builder against a stand-in CLI ----------


ROUTED_ENV = {
    "ANTHROPIC_BASE_URL": "https://gateway.example",
    "ANTHROPIC_AUTH_TOKEN": "routed-token",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "fast-model",
}


class StandInCli:
    """The agent CLI at the ``ClaudeSDKClient`` seam, filing what the real one files.

    Under the env the SDK gives it (the parent's overlaid with the options'), it
    writes its transcript to ``<config dir>/projects/<encoded cwd>/<id>.jsonl``
    and, unless the record is switched off, a record line into the spool.
    """

    def __init__(self) -> None:
        self.options: list[Any] = []

    def client(self, *, options: Any) -> StandInClient:
        self.options.append(options)
        return StandInClient(options)


class StandInClient:
    def __init__(self, options: Any) -> None:
        self._options = options
        self._env = {**os.environ, **options.env}

    async def __aenter__(self) -> StandInClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    async def get_mcp_status(self) -> dict[str, Any]:
        return {"mcpServers": []}

    async def query(self, prompt: Any) -> None:
        sid = str(uuid.uuid4())
        config = Path(self._env["CLAUDE_CONFIG_DIR"])
        encoded = claude_project_dir(self._options.cwd).name
        transcripts = config / "projects" / encoded
        transcripts.mkdir(parents=True, exist_ok=True)
        (transcripts / f"{sid}.jsonl").write_text(json.dumps({"prompt": str(prompt)}) + "\n")
        spool = self._env.get(SPOOL_DIR_ENV, "")
        if self._env.get(RECORD_OFF_ENV) != RECORD_OFF_VALUE and spool:
            Path(spool).mkdir(parents=True, exist_ok=True)
            (Path(spool) / f"{sid}.ndjson").write_text(json.dumps({"kind": "prompt"}) + "\n")

    async def receive_response(self):
        yield AssistantMessage(content=[TextBlock(text="Orbit correction")], model="fast-model")
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="namer",
            result="Orbit correction",
        )


@pytest.fixture
def stand_in(monkeypatch, tmp_path):
    """Route seams stubbed as the option-builder tests do, the CLI stood in for."""
    config_dir = tmp_path / "claude-config"
    spool = tmp_path / "spool"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv(SPOOL_DIR_ENV, str(spool))
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://record.example:4318")
    monkeypatch.delenv(RECORD_OFF_ENV, raising=False)
    monkeypatch.setattr(
        primitives.launcher, "project_agent_cli_path", lambda *_a, **_k: "/bin/claude"
    )
    monkeypatch.setattr(
        primitives.launcher,
        "project_thinking_display",
        lambda *_a, **_k: SimpleNamespace(flag_allowed=False, value=None),
    )
    monkeypatch.setattr(primitives, "sdk_env", lambda *_a, **_k: dict(ROUTED_ENV))
    monkeypatch.setattr(primitives, "resolve_default_model", lambda *_a, **_k: "main-model")
    monkeypatch.setattr(
        primitives,
        "_resolve_project_spec",
        lambda *_a, **_k: SimpleNamespace(alias_models={"haiku": "fast-model"}, needs_proxy=False),
    )
    cli = StandInCli()
    monkeypatch.setattr(runner, "ClaudeSDKClient", cli.client)
    return SimpleNamespace(cli=cli, config_dir=config_dir, spool=spool)


class TestIsolation:
    async def test_the_options_switch_the_record_off_and_keep_login_and_routing(
        self, stand_in, tmp_path
    ):
        project = tmp_path / "render"
        project.mkdir()

        assert await name_session("correct the orbit", project_dir=project) == "Orbit correction"

        (options,) = stand_in.cli.options
        assert options.env[RECORD_OFF_ENV] == RECORD_OFF_VALUE
        assert options.env[SPOOL_DIR_ENV] == ""
        assert options.env["OTEL_EXPORTER_OTLP_ENDPOINT"] == ""
        # The parent's CLAUDE_CONFIG_DIR (a personal /login) is kept, not overridden.
        assert options.env.get("CLAUDE_CONFIG_DIR", str(stand_in.config_dir)) == str(
            stand_in.config_dir
        )
        # Provider routing as the project resolves it.
        for key, value in ROUTED_ENV.items():
            assert options.env[key] == value
        assert options.model == "fast-model"
        # A temporary directory of its own, gone after the run.
        cwd = Path(options.cwd)
        assert cwd != project.resolve() and project.resolve() not in cwd.parents
        assert not cwd.exists()
        assert options.setting_sources == []
        assert options.mcp_servers == {}
        assert options.tools == []
        assert options.max_turns == 1

    async def test_a_run_leaves_no_record_row_and_no_project_transcript(self, stand_in, tmp_path):
        project = tmp_path / "render"
        project.mkdir()
        project_transcripts = claude_project_dir(project)
        project_transcripts.mkdir(parents=True)
        (project_transcripts / f"{T1}.jsonl").write_text("{}\n")
        before = sorted(project_transcripts.glob("*.jsonl"))

        await name_session("correct the orbit", project_dir=project)

        assert sorted(project_transcripts.glob("*.jsonl")) == before
        assert not stand_in.spool.exists() or list(stand_in.spool.iterdir()) == []
        # The namer's own transcript directory is not left behind either.
        (options,) = stand_in.cli.options
        assert not claude_project_dir(options.cwd).exists()

    async def test_the_stand_in_would_record_a_run_that_is_not_switched_off(
        self, stand_in, tmp_path, monkeypatch
    ):
        """Positive control: without the overrides the same stand-in files a record row."""
        monkeypatch.setattr(session_naming, "naming_env", dict)
        project = tmp_path / "render"
        project.mkdir()
        await name_session("correct the orbit", project_dir=project)
        assert len(list(stand_in.spool.iterdir())) == 1
