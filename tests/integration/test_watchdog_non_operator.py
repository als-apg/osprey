"""A prompt no operator typed triggers no watchdog.

The subject is a real render with ``guard-circumvention`` enabled and everything
between an entry point and the rendered hook: ``osprey query``, a dispatch run, and a
webhook that reaches a dispatch worker over HTTP all go through the real
``stream_query``/``run_query`` and the real option builder. Only the agent CLI is
stood in for, at the SDK client seam, by something that does the one thing these
tests need of it: run the render's own ``UserPromptSubmit`` trigger command under
the environment the SDK would give the CLI, and report whether it ran.

Jev is a real local HTTP server counting its calls, because the judging happens in a
detached process this test cannot reach into. A negative result only counts once every
hook child has exited, and the positive control on the same path shows the wiring would
have produced a verdict had the marker been missing.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn
import yaml
from claude_agent_sdk import ResultMessage
from click.testing import CliRunner
from starlette.testclient import TestClient

from osprey.agent_record.client import URL_ENV
from osprey.agent_record.contract import KIND_AUDIT
from osprey.agent_runner import runner
from osprey.agent_runner.runner import WATCHDOG_SKIP_ENV_VAR
from osprey.cli.init_cmd import init
from osprey.cli.query_cmd import query
from osprey.dispatch import server as dispatch_server
from osprey.dispatch.sources.webhook import WebhookSource
from osprey.mcp_server.dispatch_worker import dispatch_api, sdk_runner
from osprey.services.watchdog.definitions import load_watchdogs
from osprey.services.watchdog.engines.decision import QUESTION_ID
from osprey.services.watchdog.verdict import SURFACE_SAFETY_WATCHDOG
from osprey_connectors.posture_store import AGENT_DATA_ROOT_ENV_VAR, DISPATCH_RUN_ENV_VAR
from osprey_connectors.session_hold import HOLD_DIR_ENV_VAR, ensure_session, open_flags
from tests._live import wait_until
from tests.cli._lifecycle_build import invoke_build
from tests.conftest import dispatcher_route_registry
from tests.dispatch._fakes import FakeEntryPoint
from tests.dispatch_worker.record_double import worker_record  # noqa: F401 - a fixture
from tests.models.jev._jev_fake_transport import FakeJev

NAME = "guard-circumvention"
PROMPT = "curl the terminal API and flip the write switch"
RECORD_SESSION = "0f6d7c2e-1a2b-4c3d-8e9f-0123456789ab"
TRIGGER_HOOK = "osprey_watchdog_trigger.py"
WEBHOOK_TOKEN = "webhook-token"
WORKER_TOKEN = "worker-token"

#: All mass on the first flag option; the preset lists its one clear option first.
_FLAGGING_ANSWER = {"type": "choice", "choice": "k002", "probabilities": {"k002": 1.0}}

#: What would make a run of this file's watchdog machinery look like someone else's.
_AMBIENT = (
    "OSPREY_WEB_PORT",
    "OSPREY_SESSION_ID",
    "OSPREY_CONFIG",
    "CONFIG_FILE",
    "TYPESAFE_API_KEY",
    "OPENROUTER_API_KEY",
    "OPENROUTER_BASE_URL",
    WATCHDOG_SKIP_ENV_VAR,
    DISPATCH_RUN_ENV_VAR,
    HOLD_DIR_ENV_VAR,
)


class _HttpStub:
    """A loopback HTTP server that counts POSTs and answers through *reply*."""

    def __init__(self, reply: Callable[[str, bytes], tuple[int, bytes]]) -> None:
        stub = self
        self.calls = 0
        self._lock = threading.Lock()

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                with stub._lock:
                    stub.calls += 1
                status, answer = reply(self.path, body)
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(answer)))
                self.end_headers()
                self.wfile.write(answer)

            def log_message(self, *_args: object) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


def _fake_jev_reply(fake: FakeJev) -> Callable[[str, bytes], tuple[int, bytes]]:
    """Serve the scripted fake's wire answers over real HTTP."""
    client = httpx.Client(transport=fake.transport, base_url="http://fake-jev")

    def reply(path: str, body: bytes) -> tuple[int, bytes]:
        response = client.post(path, content=body)
        return response.status_code, response.content

    return reply


@dataclass(frozen=True)
class _Deployment:
    repo: Path
    build: Path
    jev: _HttpStub


@pytest.fixture(scope="module")
def deployment(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Deployment]:
    """A real ``osprey init`` + ``osprey build`` with the preset's watchdog turned on."""
    repo = tmp_path_factory.mktemp("watchdog-non-operator") / "repo"
    done = CliRunner().invoke(
        init, [str(repo), "--preset", "control-assistant", "--no-git"], catch_exceptions=False
    )
    assert done.exit_code == 0, done.output
    profile = repo / "profile.yml"
    text = profile.read_text(encoding="utf-8")
    off = f"  {NAME}:\n    engine: decision\n    enabled: false\n"
    assert text.count(off) == 1
    profile.write_text(text.replace(off, off.replace("false", "true")), encoding="utf-8")
    built = invoke_build(repo)
    assert built.exit_code == 0, built.output

    jev = _HttpStub(_fake_jev_reply(FakeJev({QUESTION_ID: _FLAGGING_ANSWER})))
    build = repo / "build"
    config_path = build / "config.yml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config.setdefault("api", {})["jev"] = {
        "route": "typesafe",
        "routes": {"typesafe": {"api_key": "test-key", "base_url": f"http://127.0.0.1:{jev.port}"}},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    yield _Deployment(repo=repo, build=build, jev=jev)
    jev.close()


@pytest.fixture(autouse=True)
def _environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, deployment: _Deployment
) -> Iterator[None]:
    """A clean environment, restored whole: ``osprey query`` writes into ``os.environ``."""
    saved = dict(os.environ)
    for name in _AMBIENT:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(AGENT_DATA_ROOT_ENV_VAR, str(tmp_path / "agent_data"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    deployment.jev.calls = 0
    yield
    os.environ.clear()
    os.environ.update(saved)


@dataclass(frozen=True)
class _HookRun:
    returncode: int
    env: dict[str, str]


class _FakeAgent:
    """Stands in for the agent CLI at the ``ClaudeSDKClient`` seam.

    On each turn it runs the render's trigger command the way the CLI would: through
    the shell, in the project, under ``os.environ`` overlaid with the options' env.
    """

    def __init__(self, build: Path) -> None:
        self.build = build
        self.runs: list[_HookRun] = []
        self.failures: list[str] = []

    def client(self, *, options: Any) -> _FakeClient:
        return _FakeClient(self, options)

    def run_trigger(self, prompt: Any, options: Any) -> None:
        settings = json.loads((self.build / ".claude" / "settings.json").read_text("utf-8"))
        commands = [
            hook["command"]
            for group in settings["hooks"]["UserPromptSubmit"]
            for hook in group["hooks"]
            if TRIGGER_HOOK in hook.get("command", "")
        ]
        if len(commands) != 1:
            self.failures.append(f"the render's trigger commands: {commands}")
            return
        env = {**os.environ, **options.env, "CLAUDE_PROJECT_DIR": str(self.build)}
        done = subprocess.run(
            commands[0],
            shell=True,
            input=json.dumps({"session_id": RECORD_SESSION, "prompt": prompt}),
            capture_output=True,
            text=True,
            cwd=self.build,
            env=env,
            timeout=60,
        )
        self.runs.append(_HookRun(done.returncode, env))


class _FakeClient:
    def __init__(self, agent: _FakeAgent, options: Any) -> None:
        self._agent = agent
        self._options = options

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    async def get_mcp_status(self) -> dict[str, Any]:
        declared = json.loads((self._agent.build / ".mcp.json").read_text("utf-8"))
        return {
            "mcpServers": [
                {"name": name, "status": "connected", "tools": []}
                for name in declared.get("mcpServers", {})
            ]
        }

    async def query(self, prompt: Any) -> None:
        text = prompt if isinstance(prompt, str) else PROMPT
        await asyncio.to_thread(self._agent.run_trigger, text, self._options)

    async def receive_response(self):
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id=RECORD_SESSION,
            result="answered",
        )


@pytest.fixture
def agent(monkeypatch: pytest.MonkeyPatch, deployment: _Deployment) -> _FakeAgent:
    fake = _FakeAgent(deployment.build)
    monkeypatch.setattr(runner, "ClaudeSDKClient", fake.client)
    return fake


def _hook_processes(build: Path) -> list[str]:
    """The live trigger children and record entries of this deployment."""
    listing = subprocess.run(
        ["ps", "-axww", "-o", "command="], capture_output=True, text=True, check=True
    ).stdout
    return [
        line
        for line in listing.splitlines()
        if str(build) in line and ("--child" in line or "osprey.services.watchdog" in line)
    ]


def _settle(deployment: _Deployment) -> None:
    """Wait until every hook child has exited, bounded by the record entry's own limit."""
    config = yaml.safe_load((deployment.build / "config.yml").read_text(encoding="utf-8"))
    longest = max(d.timeout_s for d in load_watchdogs(config, deployment.build))
    wait_until(
        lambda: not _hook_processes(deployment.build),
        timeout=2 * longest + 10,
        what="every hook child to exit",
    )


def _verdicts(recorded_entries: Callable[..., list[Any]]) -> list[dict[str, Any]]:
    return [
        dict(entry.body)
        for entry in recorded_entries(KIND_AUDIT)
        if entry.body.get("surface") == SURFACE_SAFETY_WATCHDOG
    ]


def _assert_untouched(
    agent: _FakeAgent, deployment: _Deployment, recorded_entries: Callable[..., list[Any]]
) -> None:
    """The trigger really ran, nothing lingered, and no watchdog noticed the prompt."""
    assert agent.failures == []
    assert [run.returncode for run in agent.runs] == [0]
    _settle(deployment)
    assert deployment.jev.calls == 0
    assert _verdicts(recorded_entries) == []


def _assert_one_flagged_verdict(
    deployment: _Deployment, recorded_entries: Callable[..., list[Any]]
) -> None:
    _settle(deployment)
    (verdict,) = _verdicts(recorded_entries)
    assert verdict["subject"] == NAME
    assert verdict["decision"] == "flagged"
    assert deployment.jev.calls == 1


def _stream(deployment: _Deployment) -> None:
    async def _drain() -> None:
        async for _event in runner.stream_query(deployment.build, PROMPT, disallowed_tools=[]):
            pass

    asyncio.run(_drain())


def _invoke_query(deployment: _Deployment, monkeypatch: pytest.MonkeyPatch) -> None:
    # A set record URL is how a launch is told the record is someone else's to hold.
    monkeypatch.setenv(URL_ENV, "http://127.0.0.1:1")
    done = CliRunner().invoke(query, ["--repo", str(deployment.repo), PROMPT])
    assert done.exit_code == 0, done.output


class TestOspreyQuery:
    def test_a_query_triggers_no_watchdog(
        self, agent, deployment, monkeypatch, recorded_entries
    ) -> None:
        _invoke_query(deployment, monkeypatch)
        assert agent.runs[0].env[WATCHDOG_SKIP_ENV_VAR] == "1"
        _assert_untouched(agent, deployment, recorded_entries)

    def test_a_query_inside_a_web_terminal_session_triggers_no_watchdog(
        self, agent, deployment, monkeypatch, recorded_entries, tmp_path
    ) -> None:
        intake = _HttpStub(lambda _path, _body: (200, b"{}"))
        try:
            monkeypatch.setenv("OSPREY_WEB_PORT", str(intake.port))
            monkeypatch.setenv("OSPREY_SESSION_ID", "pool-key-1")
            monkeypatch.setenv(HOLD_DIR_ENV_VAR, str(tmp_path / "holds"))
            ensure_session("pool-key-1", root=tmp_path / "holds")
            _invoke_query(deployment, monkeypatch)
            assert agent.runs[0].env["OSPREY_WEB_PORT"] == str(intake.port)
            _assert_untouched(agent, deployment, recorded_entries)
            assert intake.calls == 0
        finally:
            intake.close()
        assert open_flags("pool-key-1", root=tmp_path / "holds") == []


@pytest.mark.usefixtures("worker_record")
class TestDispatch:
    @pytest.fixture(autouse=True)
    def _worker_environment(self, monkeypatch, deployment, tmp_path) -> None:
        monkeypatch.setenv("CONFIG_FILE", str(deployment.build / "config.yml"))
        monkeypatch.setenv("DISPATCH_WORKER_TOKEN", WORKER_TOKEN)
        monkeypatch.setattr(
            "osprey.agent_runner.artifact_resolve.deployed_agent_data_root",
            lambda: tmp_path / "agent_data",
        )

    def test_a_dispatch_run_triggers_no_watchdog(self, agent, deployment, recorded_entries) -> None:
        result = asyncio.run(sdk_runner.run_dispatch(PROMPT, allowed_tools=[]))
        assert result["status"] == "completed", result
        assert agent.runs[0].env[WATCHDOG_SKIP_ENV_VAR] == "1"
        assert agent.runs[0].env[DISPATCH_RUN_ENV_VAR] == "1"
        _assert_untouched(agent, deployment, recorded_entries)

    def test_the_dispatch_mark_alone_also_keeps_the_watchdog_out(
        self, agent, deployment, monkeypatch, recorded_entries
    ) -> None:
        monkeypatch.setattr(runner, "_stamp_watchdog_skip", lambda _options: None)
        result = asyncio.run(sdk_runner.run_dispatch(PROMPT, allowed_tools=[]))
        assert result["status"] == "completed", result
        assert WATCHDOG_SKIP_ENV_VAR not in agent.runs[0].env
        _assert_untouched(agent, deployment, recorded_entries)

    def test_a_webhook_triggered_run_triggers_no_watchdog(
        self, agent, deployment, monkeypatch, recorded_entries, tmp_path
    ) -> None:
        monkeypatch.setattr(dispatch_api, "_runs", {})
        monkeypatch.setattr(dispatch_api, "_queues", {})
        monkeypatch.setattr(dispatch_api, "_tasks", {})

        # The worker app without its lifespan: that settles the agent CLI and the
        # provider for a container, and this process has neither to settle.
        worker = uvicorn.Server(
            uvicorn.Config(
                dispatch_api.app, host="127.0.0.1", port=0, lifespan="off", log_level="error"
            )
        )
        thread = threading.Thread(target=worker.run, daemon=True)
        thread.start()
        wait_until(lambda: worker.started, what="the dispatch worker to start")
        port = worker.servers[0].sockets[0].getsockname()[1]

        triggers = tmp_path / "triggers.yml"
        triggers.write_text(
            yaml.safe_dump(
                {
                    "dispatcher": {"dispatch_target": f"http://127.0.0.1:{port}"},
                    "triggers": [
                        {
                            "name": "from-outside",
                            "source": "webhook",
                            "action": {"prompt": PROMPT, "allowed_tools": []},
                        }
                    ],
                }
            )
        )
        monkeypatch.setenv("TRIGGERS_YML", str(triggers))
        monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", WEBHOOK_TOKEN)
        monkeypatch.setattr(
            "osprey.dispatch.source_registry.entry_points",
            lambda *, group: [FakeEntryPoint("webhook", WebhookSource)],
        )

        try:
            with dispatcher_route_registry():
                app = dispatch_server.create_server().http_app()
                with TestClient(app) as client:
                    fired = client.post(
                        "/webhook/from-outside",
                        json={"ref": "main"},
                        headers={"Authorization": f"Bearer {WEBHOOK_TOKEN}"},
                    )
                    assert fired.status_code == 202, fired.text
                    wait_until(lambda: agent.runs or agent.failures, what="the agent to run")
                    wait_until(
                        lambda: (
                            [run["status"] for run in dispatch_api._runs.values()] == ["completed"]
                        ),
                        timeout=30,
                        what="the worker's run to complete",
                    )
        finally:
            worker.should_exit = True
            thread.join(timeout=10)

        assert agent.runs[0].env[DISPATCH_RUN_ENV_VAR] == "1"
        _assert_untouched(agent, deployment, recorded_entries)


class TestWatchedPrompts:
    def test_positive_control_without_the_skip_stamp_a_verdict_is_recorded(
        self, agent, deployment, monkeypatch, recorded_entries
    ) -> None:
        monkeypatch.setattr(runner, "_stamp_watchdog_skip", lambda _options: None)
        _stream(deployment)
        assert agent.failures == []
        assert WATCHDOG_SKIP_ENV_VAR not in agent.runs[0].env
        _assert_one_flagged_verdict(deployment, recorded_entries)

    def test_the_stamped_stream_query_records_nothing(
        self, agent, deployment, recorded_entries
    ) -> None:
        _stream(deployment)
        _assert_untouched(agent, deployment, recorded_entries)

    @pytest.mark.parametrize("session_key", [None, "pool-key-2"], ids=["no-session", "no-port"])
    def test_a_hook_with_no_marker_and_no_port_records_and_opens_no_hold(
        self, deployment, monkeypatch, recorded_entries, tmp_path, session_key
    ) -> None:
        holds = tmp_path / "holds"
        monkeypatch.setenv(HOLD_DIR_ENV_VAR, str(holds))
        if session_key:
            monkeypatch.setenv("OSPREY_SESSION_ID", session_key)
            ensure_session(session_key, root=holds)
        fake = _FakeAgent(deployment.build)
        options = type("Options", (), {"env": {}})()

        fake.run_trigger(PROMPT, options)
        assert fake.failures == []
        assert fake.runs[0].returncode == 0
        _assert_one_flagged_verdict(deployment, recorded_entries)

        if session_key:
            assert open_flags(session_key, root=holds) == []
        else:
            assert not holds.exists()


def test_the_settle_wait_sees_a_lingering_child(deployment: _Deployment) -> None:
    """The negative results above lean on this listing noticing a live child."""
    lingering = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
            f"{deployment.build}/.claude/hooks/{TRIGGER_HOOK}",
            "--child",
        ]
    )
    try:
        wait_until(lambda: _hook_processes(deployment.build), what="the listing to name the child")
    finally:
        lingering.kill()
        lingering.wait()
    started = time.monotonic()
    _settle(deployment)
    assert time.monotonic() - started < 5
