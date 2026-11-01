"""The watchdog routes, their app wiring, the spawn stamp, the chip and the dispatch proxy.

Engines are stubbed at the runner's ``engines=`` seam; the hold record is the real
one on disk, so every assertion about a hold reads what a child process would read.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces.common_middleware import WebAuthMiddleware
from osprey.interfaces.web_auth import WebCredentials, reset_web_credentials
from osprey.interfaces.web_terminal.app import create_app
from osprey.interfaces.web_terminal.operator_session import (
    OperatorSession,
    build_operator_child_env,
)
from osprey.interfaces.web_terminal.routes import chat as chat_routes
from osprey.interfaces.web_terminal.routes import proxy as proxy_routes
from osprey.interfaces.web_terminal.routes import websocket as websocket_routes
from osprey.interfaces.web_terminal.routes.watchdog import make_notifier
from osprey.interfaces.web_terminal.routes.watchdog import router as watchdog_router
from osprey.registry.mcp import FRAMEWORK_SERVERS
from osprey.services.watchdog import runner as runner_mod
from osprey.services.watchdog.config import ENGINE_DECISION, WatchdogDefinition
from osprey.services.watchdog.history import SessionNodes
from osprey.services.watchdog.runner import WatchdogRunner
from osprey.services.watchdog.verdict import SURFACE_SAFETY_WATCHDOG, Verdict, make_prompt_ref
from osprey_connectors.errors import BuildProfileError
from osprey_connectors.session_hold import (
    HOLD_DIR_ENV_VAR,
    RECORD_NAME,
    UNREADABLE_FLAG_ID,
    is_held,
    open_flag,
    open_flags,
)

from ._proxy_fakes import _FakeStreamResponse, panel_app
from .conftest import bare_route_app

KEY = "1f5a9c7e-0b44-4a1d-9cc2-0e5b6d2f7a31"  # gitleaks:allow - a test session id
OTHER_KEY = "2f5a9c7e-0b44-4a1d-9cc2-0e5b6d2f7a32"  # gitleaks:allow - a test session id
RECORD_SESSION = "0f6d7c2e-1a2b-4c3d-8e9f-0123456789ab"
SUBMISSION = "9a8b7c6d-5e4f-4a3b-9c2d-1e0f9a8b7c6d"
PROMPT = "please loosen the write limits"
PANEL_TOKEN = "panel-token-value"
OPERATOR_SECRET = "operator-secret-value"


def submission(**overrides) -> dict:
    return {
        "session_key": KEY,
        "record_session_id": RECORD_SESSION,
        "submission_id": SUBMISSION,
        "prompt": PROMPT,
        **overrides,
    }


class StubEngine:
    """Flags every prompt with the watchdog's name in its flagged set."""

    def __init__(self, flagged=("wd",)):
        self.flagged = set(flagged)
        self.calls: list[dict] = []

    async def judge(self, defn, _nodes, prompt, *, deadline, **ids) -> Verdict:  # noqa: ARG002
        self.calls.append({"name": defn.name, "prompt": prompt, **ids})
        return Verdict(
            watchdog=defn.name,
            engine=defn.engine,
            status="ok",
            flagged=defn.name in self.flagged,
            reason="risky",
            threshold=defn.threshold,
            **ids,
        )


class FakePool:
    """A registry that knows the sessions it was given."""

    def __init__(self, *keys):
        self.keys = set(keys)

    def get_session(self, key):
        return object() if key in self.keys else None

    def has_chat_key(self, _key):
        return False

    def get_chat_session(self, _key):
        return None

    def cleanup_all(self):
        """What the lifespan calls on the PTY pool at teardown."""


def _flag(root: Path, flag_id: str, *, source: str = "wd", key: str = KEY) -> None:
    assert open_flag(
        key,
        flag_id=flag_id,
        source=source,
        reason="risky",
        prompt_ref=make_prompt_ref(RECORD_SESSION, flag_id),
        prompt_excerpt=PROMPT,
        root=root,
    )


@pytest.fixture(autouse=True)
def _no_env_root(monkeypatch):
    """Everything must reach the store through the explicit root."""
    monkeypatch.delenv(HOLD_DIR_ENV_VAR, raising=False)


@pytest.fixture(autouse=True)
def _no_record_read(monkeypatch):
    monkeypatch.setattr(runner_mod, "read_session_nodes", lambda *_a, **_k: SessionNodes())


@pytest.fixture
def root(tmp_path) -> Path:
    return tmp_path / "var" / "state" / "session_holds"


@pytest.fixture
def engine() -> StubEngine:
    return StubEngine()


@pytest.fixture
def app(root, engine) -> FastAPI:
    app = bare_route_app(
        watchdog_router,
        session_hold_root=root,
        pty_registry=FakePool(KEY),
        operator_registry=FakePool(),
    )
    definition = WatchdogDefinition(name="wd", engine=ENGINE_DECISION, enabled=True, threshold=0.5)
    app.state.watchdog_runner = WatchdogRunner(
        [definition],
        recorder=lambda _verdict: None,
        hold_root=root,
        notify=make_notifier(app),
        engines={ENGINE_DECISION: engine},
    )
    return app


@pytest.fixture
def client(app):
    with TestClient(app) as client:
        yield client


def _post_and_drain(client: TestClient, body: dict):
    resp = client.post("/api/watchdog/prompt", json=body)
    client.portal.call(client.app.state.watchdog_runner.drain)
    return resp


# ---- Intake ----


def test_intake_answers_202_and_a_flagged_prompt_holds(client, root, engine):
    resp = _post_and_drain(client, submission())

    assert resp.status_code == 202
    assert [call["prompt"] for call in engine.calls] == [PROMPT]
    [flag] = open_flags(KEY, root=root)
    assert flag.source == "wd"
    assert flag.prompt_ref == make_prompt_ref(RECORD_SESSION, SUBMISSION)
    assert is_held(KEY, root=root)


def test_the_flag_frame_carries_ids_and_no_prompt_text(client, app):
    _post_and_drain(client, submission())

    [[frame]] = [call.args for call in app.state.broadcaster.broadcast.call_args_list]
    assert frame["type"] == "watchdog_flag"
    assert frame["session_key"] == KEY
    assert PROMPT not in json.dumps(frame)


@pytest.mark.parametrize(
    "forged",
    [
        {"session_key": "not-a-uuid"},
        {"session_key": KEY.upper()},
        {"session_key": "../../etc"},
        {"record_session_id": "nope"},
        {"submission_id": "nope"},
        {"session_key": OTHER_KEY},  # well formed, and no session here has it
    ],
)
def test_a_forged_or_malformed_submission_starts_no_run(client, root, engine, forged):
    resp = _post_and_drain(client, submission(**forged))

    assert resp.status_code in (400, 404)
    assert engine.calls == []
    assert not root.exists()


def test_a_duplicate_prompt_ref_runs_once_and_still_holds(client, root, engine):
    first = _post_and_drain(client, submission())
    second = _post_and_drain(client, submission())

    assert (first.status_code, second.status_code) == (202, 202)
    assert second.json()["duplicate"] is True
    assert len(engine.calls) == 1
    assert len(open_flags(KEY, root=root)) == 1


def test_an_identical_prompt_in_a_new_submission_is_judged_again(client, engine):
    _post_and_drain(client, submission())
    _post_and_drain(client, submission(submission_id=OTHER_KEY))

    assert len(engine.calls) == 2


def test_a_malformed_body_schedules_nothing(client, engine):
    resp = client.post("/api/watchdog/prompt", json={"session_key": KEY})

    assert resp.status_code == 422
    assert engine.calls == []


# ---- GET /flags ----


def test_get_flags_reports_enabled_and_the_record(client, root):
    _flag(root, "flag-a")

    body = client.get("/api/watchdog/flags", params={"session_key": KEY}).json()

    assert body["enabled"] is True
    [flag] = body["flags"]
    assert flag["flag_id"] == "flag-a"
    assert flag["watchdog"] == "wd"
    assert flag["stop_unconfirmed"] is False
    assert flag["started"] == []


def test_get_flags_refuses_a_key_that_is_no_path_segment(client):
    assert client.get("/api/watchdog/flags", params={"session_key": "../x"}).status_code == 400


# ---- Answer ----


def _answer(client, flag_ids, *, answer="continue", key=KEY):
    return client.post(
        "/api/watchdog/flags/answer",
        json={"session_key": key, "flag_ids": flag_ids, "answer": answer},
    )


def test_answer_releases_only_the_named_ids(client, root, app, ledger):
    _flag(root, "flag-a")
    _flag(root, "flag-b", source="other")

    resp = _answer(client, ["flag-a", "flag-unknown"])

    assert resp.status_code == 200
    assert resp.json() == {"released": ["flag-a"], "stop_unconfirmed": []}
    assert [flag.flag_id for flag in open_flags(KEY, root=root)] == ["flag-b"]
    assert is_held(KEY, root=root)
    frame = app.state.broadcaster.broadcast.call_args.args[0]
    assert frame == {
        "type": "watchdog_resolved",
        "session_key": KEY,
        "flag_ids": ["flag-a"],
        "answer": "continue",
    }
    [entry] = [e for e in ledger if e["surface"] == SURFACE_SAFETY_WATCHDOG]
    assert entry["decision"] == "continue"
    assert entry["subject"] == "wd"


def test_answer_naming_nothing_open_sends_no_frame(client, app):
    assert _answer(client, ["flag-gone"]).json()["released"] == []
    app.state.broadcaster.broadcast.assert_not_called()


def test_an_operator_nobody_named_is_recorded_unattributed(client, root, ledger):
    _flag(root, "flag-a")

    _answer(client, ["flag-a"])

    record = json.loads((root / KEY / RECORD_NAME).read_text())
    assert record["flags"][0]["operator"] == "<unattributed>"
    [entry] = [e for e in ledger if e["surface"] == SURFACE_SAFETY_WATCHDOG]
    assert entry["on_behalf_of"] is None


def test_the_operator_comes_from_the_authenticated_request(app, root, ledger):
    class Stamp:
        def __init__(self, inner):
            self.inner = inner

        async def __call__(self, scope, receive, send):
            scope.setdefault("state", {})["osprey_owner"] = "alice"
            await self.inner(scope, receive, send)

    app.add_middleware(Stamp)
    _flag(root, "flag-a")
    with TestClient(app) as client:
        _answer(client, ["flag-a"])

    assert json.loads((root / KEY / RECORD_NAME).read_text())["flags"][0]["operator"] == "alice"
    [entry] = [e for e in ledger if e["surface"] == SURFACE_SAFETY_WATCHDOG]
    assert entry["on_behalf_of"] == "alice"


def test_continue_on_a_stop_unconfirmed_flag_is_audited_as_such(client, root, ledger):
    from osprey_connectors.session_hold import mark_stop_unconfirmed

    _flag(root, "flag-a")
    mark_stop_unconfirmed(KEY, ["flag-a"], operator="bob", started=("sleep 99",), root=root)

    _answer(client, ["flag-a"])

    [entry] = [e for e in ledger if e["surface"] == SURFACE_SAFETY_WATCHDOG]
    assert entry["reason"] == "after_stop_unconfirmed"
    assert "sleep 99" in json.dumps(entry["detail"])


def test_a_corrupt_record_shows_the_synthetic_flag_and_answering_rewrites_it(client, root, ledger):
    _flag(root, "flag-a")
    (root / KEY / RECORD_NAME).write_text("{ not json")

    body = client.get("/api/watchdog/flags", params={"session_key": KEY}).json()
    assert [flag["flag_id"] for flag in body["flags"]] == [UNREADABLE_FLAG_ID]
    assert is_held(KEY, root=root)

    assert _answer(client, [UNREADABLE_FLAG_ID]).json()["released"] == [UNREADABLE_FLAG_ID]

    assert open_flags(KEY, root=root) == []
    assert json.loads((root / KEY / RECORD_NAME).read_text())["schema"] == 1
    [entry] = [e for e in ledger if e["surface"] == SURFACE_SAFETY_WATCHDOG]
    assert entry["decision"] == "continue"


# ---- Tiers, through the real gate ----


@pytest.fixture
def gated_client(root):
    reset_web_credentials()
    app = FastAPI()
    app.include_router(watchdog_router)
    app.state.session_hold_root = root
    app.state.watchdog_runner = MagicMock()
    app.state.pty_registry = FakePool(KEY)
    app.state.operator_registry = FakePool()
    app.state.web_credentials = WebCredentials(
        operator_secret=OPERATOR_SECRET, panel_token=PANEL_TOKEN
    )
    app.add_middleware(WebAuthMiddleware, cookie_name="osprey_terminal_session_8080")
    with TestClient(app, client=("127.0.0.1", 54321)) as client:
        yield client
    reset_web_credentials()


@pytest.mark.no_auth_seam
def test_the_panel_token_reaches_intake_and_nothing_else(gated_client, root):
    _flag(root, "flag-a")
    bearer = {"authorization": f"Bearer {PANEL_TOKEN}"}

    assert (
        gated_client.post("/api/watchdog/prompt", json=submission(), headers=bearer).status_code
        == 202
    )
    assert (
        gated_client.get(
            "/api/watchdog/flags", params={"session_key": KEY}, headers=bearer
        ).status_code
        == 401
    )
    assert (
        gated_client.post(
            "/api/watchdog/flags/answer",
            json={"session_key": KEY, "flag_ids": ["flag-a"], "answer": "continue"},
            headers=bearer,
        ).status_code
        == 401
    )
    assert is_held(KEY, root=root)


# ---- Frames on the chat stream ----


@pytest.mark.asyncio
async def test_frames_reach_the_chat_feed_unchanged():
    session = OperatorSession(cwd="/tmp", env={})
    feed, _hello = session.subscribe()
    app = SimpleNamespace(
        state=SimpleNamespace(
            broadcaster=MagicMock(),
            operator_registry=SimpleNamespace(
                get_chat_session=lambda key: session if key == KEY else None
            ),
        )
    )
    session.__class__ = type(
        "ActiveSession", (OperatorSession,), {"is_active": property(lambda _self: True)}
    )

    frame = {"type": "watchdog_flag", "session_key": KEY, "flag_id": "f", "watchdog": "wd"}
    await make_notifier(app)(frame)

    delivered = feed.get_nowait()
    assert chat_routes._strip_for_chat(delivered) == {**frame, "seq": delivered["seq"]}
    assert app.state.broadcaster.broadcast.call_args.args[0] == frame


@pytest.mark.asyncio
async def test_a_failed_broadcast_costs_only_the_notice():
    app = SimpleNamespace(
        state=SimpleNamespace(
            broadcaster=SimpleNamespace(broadcast=MagicMock(side_effect=RuntimeError("down"))),
            operator_registry=FakePool(),
        )
    )
    await make_notifier(app)({"type": "watchdog_status", "session_key": KEY})


# ---- The app's wiring ----


def _write_build(tmp_path: Path, *, enabled: bool = True) -> Path:
    build = tmp_path / "build"
    directory = build / "watchdogs" / "wd"
    directory.mkdir(parents=True)
    (directory / "instructions.md").write_text("Judge the prompt.\n")
    block = {
        "wd": {
            "engine": "decision",
            "enabled": True,
            "threshold": 0.5,
            "options": [
                {"id": "a", "text": "risky", "flag": True},
                {"id": "b", "text": "benign", "flag": False},
            ],
        }
    }
    config = {"watchdogs": block} if enabled else {}
    path = build / "config.yml"
    path.write_text(json.dumps(config))
    return path


@pytest.fixture
def web_config(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "osprey.interfaces.web_terminal.app._load_web_config",
        lambda _path=None: {"watch_dir": str(tmp_path)},
    )


def _started(config: Path) -> TestClient:
    return TestClient(create_app(config_path=config, shell_command=["echo"]))


def test_enabled_build_resolves_one_root_and_stamps_the_child(tmp_path, web_config):  # noqa: ARG001
    config = _write_build(tmp_path)
    expected = tmp_path / "var" / "state" / "session_holds"

    with _started(config) as client:
        app = client.app
        assert app.state.session_hold_root == expected
        assert isinstance(app.state.watchdog_runner, WatchdogRunner)

        env = build_operator_child_env(str(tmp_path), session_key=KEY, app=app)

        assert env[HOLD_DIR_ENV_VAR] == str(expected)
        assert (expected / KEY / RECORD_NAME).is_file()
        assert not is_held(KEY, root=expected)


def test_intake_lands_in_the_record_with_the_env_var_unset(tmp_path, web_config, monkeypatch):  # noqa: ARG001
    """The server's own environment has no say: the explicit root carries the store."""
    config = _write_build(tmp_path)
    monkeypatch.setattr(
        "osprey.services.watchdog.runner.WatchdogRunner._build_engine",
        lambda _self, _name: StubEngine(),
    )

    with _started(config) as client:
        client.app.state.pty_registry = FakePool(KEY)
        client.app.state.watchdog_runner._recorder = lambda _verdict: None

        assert client.post("/api/watchdog/prompt", json=submission()).status_code == 202
        client.portal.call(client.app.state.watchdog_runner.drain)

        [flag] = client.get("/api/watchdog/flags", params={"session_key": KEY}).json()["flags"]
        assert flag["watchdog"] == "wd"
    assert HOLD_DIR_ENV_VAR not in os.environ


def test_handoff_respawn_and_restart_keep_open_flags(tmp_path, web_config):  # noqa: ARG001
    config = _write_build(tmp_path)
    root = tmp_path / "var" / "state" / "session_holds"

    with _started(config) as client:
        build_operator_child_env(str(tmp_path), session_key=KEY, app=client.app)
        _flag(root, "flag-a")
        # The respawn a hand-off does: the same key, through the same builder.
        build_operator_child_env(str(tmp_path), session_key=KEY, app=client.app)
        assert [flag.flag_id for flag in open_flags(KEY, root=root)] == ["flag-a"]

    with _started(config) as client:
        env = build_operator_child_env(str(tmp_path), session_key=KEY, app=client.app)
        body = client.get("/api/watchdog/flags", params={"session_key": KEY}).json()

    assert env[HOLD_DIR_ENV_VAR] == str(root)
    assert [flag["flag_id"] for flag in body["flags"]] == ["flag-a"]


def test_a_spawn_never_rewrites_a_corrupt_record(tmp_path, web_config):  # noqa: ARG001
    config = _write_build(tmp_path)
    root = tmp_path / "var" / "state" / "session_holds"

    with _started(config) as client:
        build_operator_child_env(str(tmp_path), session_key=KEY, app=client.app)
        (root / KEY / RECORD_NAME).write_text("garbage")
        build_operator_child_env(str(tmp_path), session_key=KEY, app=client.app)

    assert (root / KEY / RECORD_NAME).read_text() == "garbage"
    assert is_held(KEY, root=root)


def test_a_failed_ensure_still_stamps_so_the_session_holds(tmp_path, web_config):  # noqa: ARG001
    config = _write_build(tmp_path)
    root = tmp_path / "var" / "state" / "session_holds"
    root.parent.mkdir(parents=True)
    root.write_text("a file where the root should be")

    with _started(config) as client:
        env = build_operator_child_env(str(tmp_path), session_key=KEY, app=client.app)

    assert env[HOLD_DIR_ENV_VAR] == str(root)
    assert is_held(KEY, root=root)


def test_disabled_build_is_inert_under_an_inherited_corrupt_path(
    tmp_path,
    web_config,  # noqa: ARG001
    monkeypatch,
):
    config = _write_build(tmp_path, enabled=False)
    corrupted = tmp_path / "corrupted"
    corrupted.write_text("not a directory")
    monkeypatch.setenv(HOLD_DIR_ENV_VAR, str(corrupted))

    with _started(config) as client:
        app = client.app
        assert app.state.session_hold_root is None
        assert app.state.watchdog_runner is None

        env = build_operator_child_env(str(tmp_path), session_key=KEY, app=app)
        extra = websocket_routes._build_extra_env(SimpleNamespace(app=app), KEY)
        flags = client.get("/api/watchdog/flags", params={"session_key": KEY}).json()
        refused = client.post("/api/watchdog/prompt", json=submission())
        answered = _answer(client, ["flag-a"])

    assert HOLD_DIR_ENV_VAR not in env
    assert HOLD_DIR_ENV_VAR not in extra
    assert flags == {"enabled": False, "flags": []}
    assert refused.status_code == 404
    assert answered.json() == {"released": [], "stop_unconfirmed": []}
    assert not (tmp_path / "var").exists()


def test_a_broken_block_stops_the_server(tmp_path, web_config):  # noqa: ARG001
    config = _write_build(tmp_path)
    block = json.loads(config.read_text())
    block["watchdogs"]["wd"]["threshold"] = 7
    config.write_text(json.dumps(block))

    with pytest.raises(BuildProfileError), _started(config):
        pass


def test_shutdown_ends_the_runner(tmp_path, web_config):  # noqa: ARG001
    config = _write_build(tmp_path)
    with _started(config) as client:
        runner = client.app.state.watchdog_runner
    assert runner._closing is True


# ---- PTY stamp ----


def test_pty_spawn_stamps_the_root_and_creates_the_record(root):
    app = SimpleNamespace(state=SimpleNamespace(session_hold_root=root))

    extra = websocket_routes._build_extra_env(SimpleNamespace(app=app), KEY)

    assert extra[HOLD_DIR_ENV_VAR] == str(root)
    assert open_flags(KEY, root=root) == []
    assert (root / KEY / RECORD_NAME).is_file()


# ---- The chip ----


def test_the_posture_route_passes_the_session_and_reports_held(root):
    """``held`` follows the asking session; another session is not held by it."""
    app = SimpleNamespace(state=SimpleNamespace(session_hold_root=root))
    _flag(root, "flag-a")
    (root / OTHER_KEY).mkdir()
    (root / OTHER_KEY / RECORD_NAME).write_text(
        json.dumps({"schema": 1, "session_key": OTHER_KEY, "flags": []})
    )

    view = websocket_routes._posture_view(app, None, KEY)
    assert view["held"] is True
    assert websocket_routes._posture_view(app, None, OTHER_KEY)["held"] is False
    assert websocket_routes._posture_view(app, None)["held"] is False


def test_held_is_false_with_no_watchdog_whatever_the_path(tmp_path, monkeypatch):
    corrupted = tmp_path / "corrupted"
    corrupted.write_text("x")
    monkeypatch.setenv(HOLD_DIR_ENV_VAR, str(corrupted))
    app = SimpleNamespace(state=SimpleNamespace(session_hold_root=None))

    assert websocket_routes._posture_view(app, None, KEY)["held"] is False


# ---- The dispatch proxy ----

FIRE_BODY = json.dumps(
    {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "tools/call",
        "params": {"name": "manual_fire", "arguments": {"name": "job"}},
    }
).encode()
LIST_BODY = b'{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'
READ_BODY = json.dumps(
    {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "list_triggers"}}
).encode()


class TestDispatchProxy:
    @pytest.fixture
    def events(self, workspace_dir, monkeypatch):
        monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", "dispatcher-bearer-value")
        custom = [
            {
                "id": "events",
                "label": "EVENTS",
                "url": "http://localhost:8020",
                "configDefined": True,
            }
        ]
        yield from panel_app(workspace_dir, custom)

    @pytest.fixture
    def armed(self, events, root):
        """The events hop, under a store, with an upstream that counts what reaches it."""
        app, client = events
        app.state.session_hold_root = root
        app.state.pty_registry = FakePool(KEY)
        app.state.proxy_client.build_request = MagicMock(return_value=MagicMock())
        app.state.proxy_client.send = AsyncMock(
            side_effect=lambda *_a, **_k: _FakeStreamResponse(
                headers={"content-type": "application/json"}, chunks=(b"{}",)
            )
        )
        return app, client

    @staticmethod
    def _post(client, body, *, session=KEY):
        headers = {
            "accept": "application/json, text/event-stream",
            "content-type": "application/json",
        }
        if session is not None:
            headers["x-osprey-session"] = session
        return client.post("/panel/events/mcp", headers=headers, content=body)

    def test_a_job_firing_call_is_refused_while_held_and_fires_after_continue(self, armed, root):
        app, client = armed
        send = app.state.proxy_client.send
        _flag(root, "flag-a")

        resp = self._post(client, FIRE_BODY)

        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == 7
        assert "[osprey-refusal: session_held]" in body["error"]["message"]
        send.assert_not_called()

        answered = _answer(client, ["flag-a"])
        assert answered.json()["released"] == ["flag-a"]

        self._post(client, FIRE_BODY)
        send.assert_called_once()

    @pytest.mark.parametrize("session", [None, "", "not-live", OTHER_KEY, "../x"])
    def test_with_the_store_active_a_missing_or_dead_key_is_refused(self, armed, session):
        app, client = armed

        resp = self._post(client, FIRE_BODY, session=session)

        assert "error" in resp.json()
        app.state.proxy_client.send.assert_not_called()

    def test_a_corrupt_record_refuses_the_firing(self, armed, root):
        app, client = armed
        (root / KEY).mkdir(parents=True)
        (root / KEY / RECORD_NAME).write_text("garbage")

        assert "session_held" in self._post(client, FIRE_BODY).json()["error"]["message"]
        app.state.proxy_client.send.assert_not_called()

    @pytest.mark.parametrize("body", [LIST_BODY, READ_BODY, b"not json"])
    def test_calls_that_fire_nothing_pass_while_held(self, armed, root, body):
        app, client = armed
        _flag(root, "flag-a")

        self._post(client, body, session=None)

        app.state.proxy_client.send.assert_called_once()

    def test_a_batch_with_a_firing_call_is_refused_whole(self, armed, root):
        app, client = armed
        _flag(root, "flag-a")
        batch = json.dumps([json.loads(LIST_BODY), json.loads(FIRE_BODY)]).encode()

        answer = self._post(client, batch).json()

        assert [entry["id"] for entry in answer] == [1, 7]
        app.state.proxy_client.send.assert_not_called()

    def test_with_no_store_nothing_is_checked(self, events):
        app, client = events
        app.state.proxy_client.build_request = MagicMock(return_value=MagicMock())
        app.state.proxy_client.send = AsyncMock(
            return_value=_FakeStreamResponse(
                headers={"content-type": "application/json"}, chunks=(b"{}",)
            )
        )

        self._post(client, FIRE_BODY, session=None)

        app.state.proxy_client.send.assert_called_once()


def test_every_tool_the_dispatcher_prompts_for_is_a_job_firing_one():
    """The tools that are not plain reads are exactly the ones the proxy holds back."""
    entry = FRAMEWORK_SERVERS["event_dispatcher"]
    assert set(entry.permissions_ask) == proxy_routes._JOB_FIRING_TOOLS


def test_the_dispatcher_entry_names_the_calling_session():
    entry = FRAMEWORK_SERVERS["event_dispatcher"]
    assert entry.headers["X-Osprey-Session"] == "${OSPREY_SESSION_ID:-}"
