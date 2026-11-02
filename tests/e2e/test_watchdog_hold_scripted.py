"""A flagged session is held on the real path, with the model and Jev replaced.

Real here: the web terminal app and its chat route, the pinned Claude Code it
launches, the rendered hooks (the trigger hook posts each prompt to the app's
intake), the MCP servers, the decision engine and its HTTP client, the hold record,
the in-process simulator. Replaced: the model (:mod:`tests.e2e.scripted_model`), and
the Jev service, by a local server behind ``api.jev`` that answers with the flag
mass a test asks for, and the event dispatcher behind the EVENTS panel.

A flag opens after its prompt. So a turn that must meet the flag is held back at
the scripted model until the record shows one, and the write cannot beat the verdict.
Each refusal is the hold's own: its reason trailer, in the result the agent received.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from osprey.interfaces._serving import run_app_server
from osprey.interfaces.common_middleware import OPERATOR_SECRET_HEADER, WEB_PORT_ENV
from osprey.interfaces.web_auth import DISPATCHER_TOKEN_ENV, get_web_credentials
from osprey.interfaces.web_terminal.operator_session import (
    build_operator_child_env,
    format_tool_name,
    session_hold_root_of,
)
from osprey.interfaces.web_terminal.routes import chat as chat_routes
from osprey.models.jev.config import TYPESAFE_KEY_ENV
from osprey_connectors.refusals import refusal_phrase
from osprey_connectors.session_hold import RECORD_NAME, open_flags
from tests.e2e import sdk_helpers
from tests.e2e.claude_code.conftest import (
    BOUNDED_SETPOINT,
    _point_at_safety_limits_db,
    _serve_safety_channels,
)
from tests.e2e.claude_code.test_safety_gates_scripted import assert_untouched, served_view
from tests.e2e.scripted_model import ScriptedCall, ScriptedModel
from tests.e2e.sdk_helpers import (
    cli_mcp_ready_env,
    enable_writes_in_project,
    init_project,
    render_dir,
    trust_render,
)
from tests.facility.served_tree import assert_unmoved, served_active, unmoved_band
from tests.interfaces._hub import LONG_LIVED_SHELL, hub_app

pytestmark = [pytest.mark.model_free, pytest.mark.harness_benchmark]

WATCHDOG = "guard-circumvention"
#: The preset's first option, which does not flag, and its second, which does.
ROUTINE_KEY = "k001"
FLAGGING_KEY = "k002"

HELD = "[osprey-refusal: session_held]"

WRITE = "mcp__controls__channel_write"
READ = "mcp__controls__channel_read"
EXECUTE = "mcp__python__execute"
QUEUE_ADD = "mcp__bluesky__queue_add"
QUEUE_START = "mcp__bluesky__queue_start"
DRIVE = "mcp__phoebus__phoebus_drive"

#: Every tool a session can start a write with.
WRITES = (WRITE, EXECUTE, QUEUE_ADD, QUEUE_START, DRIVE)

SETPOINT = BOUNDED_SETPOINT
EXECUTE_DESCRIPTION = "Scripted proof: a write from an executed script"

#: How long a turn or a verdict may take before the proof fails.
TURN_TIMEOUT_S = 180.0
FLAG_TIMEOUT_S = 60.0

CHIP_ROUTE = sdk_helpers.CHIP_POSTURE_PATH
EVENTS_MCP_ROUTE = "/panel/events/mcp"
SESSION_HEADER = "x-osprey-session"
FIRE_JOB = {
    "jsonrpc": "2.0",
    "id": 7,
    "method": "tools/call",
    "params": {"name": "manual_fire", "arguments": {"name": "nightly"}},
}


def write(value: float) -> ScriptedCall:
    return ScriptedCall(WRITE, {"operations": [{"channel": SETPOINT, "value": value}]})


def read() -> ScriptedCall:
    return ScriptedCall(READ, {"channels": [SETPOINT], "include_metadata": False})


def execute(code: str) -> ScriptedCall:
    return ScriptedCall(
        EXECUTE,
        {"code": code, "description": EXECUTE_DESCRIPTION, "execution_mode": "readwrite"},
    )


def write_channel_script(value: float) -> str:
    return (
        "from osprey.runtime import write_channel\n"
        f"write_channel({SETPOINT!r}, {value!r})\n"
        "print('WRITTEN')\n"
    )


def every_write() -> list[list[ScriptedCall]]:
    """One call of each thing a session can start a write with."""
    return [
        [write(61.0)],
        [execute(write_channel_script(62.0))],
        [ScriptedCall(QUEUE_ADD, {"draft_revision": 1})],
        [ScriptedCall(QUEUE_START, {})],
        [ScriptedCall(DRIVE, {"widget": "SetButton", "verb": "click"})],
    ]


def _serve(handler: type[BaseHTTPRequestHandler]) -> tuple[ThreadingHTTPServer, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


class _JsonHandler(BaseHTTPRequestHandler):
    """Answers each POST with the JSON its subclass computes from the body."""

    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        """Silence the per-request stderr line."""

    def reply(self, body: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def do_POST(self) -> None:
        raw = self.rfile.read(int(self.headers.get("content-length", 0)))
        data = json.dumps(self.reply(json.loads(raw))).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class FakeJev:
    """The decision endpoint on a real socket: the flag mass is the test's to set."""

    def __init__(self) -> None:
        self.flagging = False
        self.requests = 0
        fake = self

        class Handler(_JsonHandler):
            def reply(self, body: dict[str, Any]) -> dict[str, Any]:
                fake.requests += 1
                flagging = fake.flagging
                answers = {
                    question: {
                        "type": "choice",
                        "choice": FLAGGING_KEY if flagging else ROUTINE_KEY,
                        "probabilities": {
                            FLAGGING_KEY: 0.9 if flagging else 0.1,
                            ROUTINE_KEY: 0.1 if flagging else 0.9,
                        },
                        "confidence": 0.9,
                    }
                    for question in body["questions"]
                }
                return {
                    "model": body["model"],
                    "answers": answers,
                    "usage": {"input_tokens": 10, "output_tokens": 1},
                }

        self._server, self.url = _serve(Handler)

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()


class FakeDispatcher:
    """What is behind the EVENTS panel: it counts the calls that reach it."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        fake = self

        class Handler(_JsonHandler):
            def reply(self, body: dict[str, Any]) -> dict[str, Any]:
                fake.calls.append(body)
                return {"jsonrpc": "2.0", "id": body.get("id"), "result": {"fired": True}}

        self._server, self.url = _serve(Handler)

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@dataclass
class Conductor:
    """The script of each prompt in one chat, on one scripted model.

    The child is launched once, with one endpoint, and a conversation grows across
    prompts. So a turn is read off the messages after the current prompt, and a
    wait can be set in front of a turn to hold it back.

    ``results`` is what the agent itself received for each tool call, by tool name.
    """

    model: ScriptedModel
    prompt: str = ""
    script: list[list[ScriptedCall]] = field(default_factory=list)
    before_turn: dict[int, Callable[[], None]] = field(default_factory=dict)
    results: dict[str, list[str]] = field(default_factory=dict)
    offered: list[str] = field(default_factory=list)
    delivered: set[int] = field(default_factory=set)
    returned: bool = False

    def install(self) -> None:
        recording = self.model.answer

        def answer(body: dict[str, Any], credential: str | None):
            blocks, stop = recording(body, credential)
            if not body.get("tools"):
                return blocks, stop
            messages = [m for m in body.get("messages") or [] if isinstance(m, dict)]
            start = _last_index_naming(messages, self.prompt)
            if start is None:
                return blocks, stop
            after = messages[start + 1 :]
            turn = sum(1 for m in after if m.get("role") == "assistant")
            self._note_results(messages, after)
            self.offered = [tool.get("name", "") for tool in body["tools"]]
            if turn >= len(self.script):
                self.returned = True
                return blocks, stop
            if turn not in self.delivered and turn in self.before_turn:
                self.before_turn[turn]()
            self.delivered.add(turn)
            return [
                {
                    "type": "tool_use",
                    "id": f"toolu_hold_{uuid.uuid4().hex[:12]}",
                    "name": call.name,
                    "input": dict(call.input),
                }
                for call in self.script[turn]
            ], "tool_use"

        self.model.answer = answer  # type: ignore[method-assign]

    def _note_results(self, messages: list[dict[str, Any]], after: list[dict[str, Any]]) -> None:
        names = {
            block["id"]: block["name"]
            for message in messages
            if message.get("role") == "assistant" and isinstance(message.get("content"), list)
            for block in message["content"]
            if isinstance(block, dict) and block.get("type") == "tool_use"
        }
        seen: dict[str, list[str]] = {}
        for message in after:
            if message.get("role") != "user" or not isinstance(message.get("content"), list):
                continue
            for block in message["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    name = names.get(block.get("tool_use_id"), "?")
                    seen.setdefault(name, []).append(_result_text(block))
        self.results = seen

    def next_prompt(
        self,
        prompt: str,
        script: list[list[ScriptedCall]],
        before_turn: dict[int, Callable[[], None]] | None = None,
    ) -> None:
        self.prompt = prompt
        self.script = script
        self.before_turn = before_turn or {}
        self.results = {}
        self.delivered = set()
        self.returned = False

    def assert_consumed(self) -> None:
        missing = set(range(len(self.script))) - self.delivered
        assert not missing, f"scripted turns never delivered: {sorted(missing)}"
        assert self.returned, "the agent never returned the results of the last scripted turn"


def _last_index_naming(messages: list[dict[str, Any]], prompt: str) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") == "user" and prompt in json.dumps(message.get("content")):
            return index
    return None


def _result_text(block: dict[str, Any]) -> str:
    content = block.get("content")
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content or [] if isinstance(part, dict))


def assert_held_refusal(text: str) -> None:
    """*text* is the hold's own refusal, naming the watchdog that raised the flag."""
    assert HELD in text, f"not the hold's refusal: {text[:400]}"
    assert refusal_phrase("session_held") in text, text[:400]


def setpoint_reading(text: str) -> float:
    """The setpoint's value in the result of a channel read."""
    found = re.search(r'\\?"value\\?":\s*(-?[\d.]+(?:[eE][-+]?\d+)?)', text)
    assert found is not None, f"no reading in: {text[:400]}"
    return float(found.group(1))


@dataclass
class Stack:
    """One served app, its chat session, and every frame the watchdog notified."""

    app: Any
    client: httpx.AsyncClient
    conductor: Conductor
    jev: FakeJev | None
    dispatcher: FakeDispatcher
    repo: Path
    frames: list[dict[str, Any]]
    key: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def root(self) -> Path:
        root = session_hold_root_of(self.app)
        assert root is not None, "the render runs no watchdog"
        return root

    def flags(self) -> list[Any]:
        return open_flags(self.key, root=self.root)

    def wait_for_flag(self) -> None:
        """Block the scripted turn until the session's record shows the flag."""
        deadline = time.monotonic() + FLAG_TIMEOUT_S
        while time.monotonic() < deadline:
            if self.flags():
                return
            time.sleep(0.05)
        raise AssertionError(f"no flag opened within {FLAG_TIMEOUT_S} s: {self.frames}")

    def wait_for_frame(self, wanted: Callable[[dict[str, Any]], bool]) -> None:
        deadline = time.monotonic() + FLAG_TIMEOUT_S
        while time.monotonic() < deadline:
            if any(wanted(frame) for frame in self.frames):
                return
            time.sleep(0.05)
        raise AssertionError(f"the frame never came: {self.frames}")

    async def turn(
        self,
        script: list[list[ScriptedCall]],
        *,
        flagging: bool = False,
        before_turn: dict[int, Callable[[], None]] | None = None,
        on_card: Callable[[dict[str, Any]], Awaitable[bool]] | None = None,
    ) -> list[dict[str, Any]]:
        """Run one prompt through ``POST /api/chat``; return its frames, failing if unfinished."""
        prompt = f"Carry out the scripted steps ({uuid.uuid4().hex})."
        if self.jev is not None:
            self.jev.flagging = flagging
        self.conductor.next_prompt(prompt, script, before_turn)
        frames: list[dict[str, Any]] = []
        async with self.client.stream(
            "POST", "/api/chat", json={"chat_id": self.key, "prompt": prompt}
        ) as response:
            assert response.status_code == 200, await response.aread()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                frame = json.loads(line[len("data: ") :])
                frames.append(frame)
                if frame.get("type") == "permission_request":
                    allow = await on_card(frame) if on_card else True
                    answered = await self.client.post(
                        f"/api/chat/{self.key}/permission",
                        json={"request_id": frame["request_id"], "allow": allow},
                    )
                    assert answered.status_code == 204, answered.text
        try:
            self.conductor.assert_consumed()
        except AssertionError as exc:
            requests = [(r.turn_index, len(r.offered_tools)) for r in self.conductor.model.requests]
            raise AssertionError(f"{exc}; frames {frames}; requests {requests}") from None
        return frames

    async def open_flag_through_intake(self) -> None:
        """Submit a prompt to the intake the trigger hook posts to, with Jev flagging."""
        assert self.jev is not None
        self.jev.flagging = True
        posted = await self.client.post(
            "/api/watchdog/prompt",
            json={
                "session_key": self.key,
                "record_session_id": self.key,
                "submission_id": str(uuid.uuid4()),
                "prompt": "A further prompt from this session.",
            },
        )
        assert posted.status_code == 202, posted.text
        await asyncio.to_thread(self.wait_for_flag)

    async def held(self) -> bool:
        posture = await self.client.get(CHIP_ROUTE, params={"session_id": self.key})
        assert posture.status_code == 200, posture.text
        return posture.json()["held"]

    async def press_continue(self) -> None:
        listed = await self.client.get("/api/watchdog/flags", params={"session_key": self.key})
        flag_ids = [flag["flag_id"] for flag in listed.json()["flags"]]
        assert flag_ids, "nothing to answer"
        answered = await self.client.post(
            "/api/watchdog/flags/answer",
            json={"session_key": self.key, "flag_ids": flag_ids, "answer": "continue"},
        )
        assert answered.status_code == 200, answered.text
        assert answered.json()["released"] == flag_ids


@pytest.fixture(scope="module")
def deployment(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A control-assistant repo: watchdog on, every write surface offered, writes armed."""
    root = tmp_path_factory.mktemp("watchdog-hold")
    repo = init_project(root, "watchdog-hold", provider="als-apg")
    profile_path = repo / "profile.yml"
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    profile["watchdogs"][WATCHDOG]["enabled"] = True
    profile["config"] = {
        **(profile.get("config") or {}),
        "claude_code.servers.bluesky.enabled": True,
        "claude_code.servers.phoebus.enabled": True,
        "phoebus.agent_access": "read_write",
    }
    profile_path.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
    enable_writes_in_project(repo)
    _serve_safety_channels(repo, root / "served")
    _point_at_safety_limits_db(repo)
    return repo


def _point_jev_at(repo: Path, url: str) -> Path:
    """Name *url* as the decision endpoint in the config the app and the servers read."""
    config_path = render_dir(repo) / "config.yml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    api = config.setdefault("api", {})
    api["jev"] = {"route": "typesafe", "routes": {"typesafe": {"base_url": url}}}
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


@pytest.fixture
def serve_stack(
    deployment: Path,
    scripted_model: Callable[[Any], ScriptedModel],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Callable[..., Any]:
    """The factory of one stack: the app served over the deployment, its URLs set."""

    @asynccontextmanager
    async def _stack(*, jev_url: str | None = None) -> AsyncIterator[Stack]:
        jev = None if jev_url else FakeJev()
        dispatcher = FakeDispatcher()
        model = scripted_model([])
        conductor = Conductor(model)
        conductor.install()

        config_path = _point_jev_at(deployment, jev_url or jev.url)  # type: ignore[union-attr]
        render = render_dir(deployment)
        trust_render(deployment)
        monkeypatch.setenv("OSPREY_CONFIG", str(config_path))
        monkeypatch.setenv("CONFIG_FILE", str(config_path))
        monkeypatch.setenv(TYPESAFE_KEY_ENV, "fake-jev-key")
        monkeypatch.setenv(DISPATCHER_TOKEN_ENV, "fake-dispatcher-token")
        monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", "fake-launch-token")
        monkeypatch.setenv("PHOEBUS_REQUIRE_HANDLE", "0")
        monkeypatch.setenv("OSPREY_HOOK_DEBUG", "1")
        for name, value in cli_mcp_ready_env().items():
            monkeypatch.setenv(name, value)

        def chat_child_env(*args: Any, **kwargs: Any) -> dict[str, str]:
            # The chat's own environment, hold stamps included, on the scripted credential.
            return {**build_operator_child_env(*args, **kwargs), **sdk_helpers.sdk_env(render)}

        monkeypatch.setattr(chat_routes, "build_operator_child_env", chat_child_env)

        events = {
            "id": "events",
            "label": "EVENTS",
            "url": dispatcher.url,
            "configDefined": True,
        }
        try:
            with hub_app(
                tmp_path / "watched",
                enabled_panels=(),
                custom_panels=[events],
                shell_command=LONG_LIVED_SHELL,
                config_path=config_path,
                project_dir=render,
            ) as app:
                with run_app_server(app) as base_url:
                    # Read when a child starts, so the trigger hook posts to this app.
                    monkeypatch.setenv(WEB_PORT_ENV, base_url.rsplit(":", 1)[1])
                    frames = _spy_on_notifications(app)
                    async with httpx.AsyncClient(
                        base_url=base_url,
                        timeout=TURN_TIMEOUT_S,
                        headers={OPERATOR_SECRET_HEADER: get_web_credentials(app).operator_secret},
                    ) as client:
                        stack = Stack(app, client, conductor, jev, dispatcher, deployment, frames)
                        try:
                            yield stack
                        finally:
                            await client.delete(f"/api/chat/{stack.key}")
        finally:
            dispatcher.stop()
            if jev is not None:
                jev.stop()

    return _stack


def _spy_on_notifications(app: Any) -> list[dict[str, Any]]:
    """Keep every frame the app broadcasts, on its way to the real broadcaster."""
    frames: list[dict[str, Any]] = []
    broadcaster = app.state.broadcaster
    broadcast = broadcaster.broadcast

    def keep_and_broadcast(frame: dict[str, Any]) -> Any:
        frames.append(dict(frame))
        return broadcast(frame)

    broadcaster.broadcast = keep_and_broadcast
    return frames


async def test_a_flag_holds_every_write_until_continue(serve_stack: Callable[..., Any]) -> None:
    """Flag opens from the prompt; every write is refused; Continue returns them to the gate."""
    async with serve_stack() as stack:
        await stack.turn([[read()]])
        assert await stack.held() is False

        before = setpoint_reading(stack.conductor.results[READ][0])
        cards: list[str] = []

        async def allow(frame: dict[str, Any]) -> bool:
            cards.append(frame.get("tool_name", ""))
            return True

        await stack.turn(
            [*every_write(), [read()]],
            flagging=True,
            before_turn={0: stack.wait_for_flag},
            on_card=allow,
        )

        results = stack.conductor.results
        for tool in WRITES:
            assert tool in results, f"{tool} was not delivered: {stack.conductor.offered}"
            assert_held_refusal(results[tool][0])
        assert cards == [], f"an approval card was raised while held: {cards}"
        assert [flag.source for flag in stack.flags()] == [WATCHDOG]
        assert await stack.held() is True
        assert_untouched(
            served_view(stack.repo),
            SETPOINT,
            before,
            setpoint_reading(results[READ][0]),
            61.0,
        )

        await stack.press_continue()
        assert stack.flags() == []
        assert await stack.held() is False

        asked: list[str] = []

        async def allow_the_channel_write(frame: dict[str, Any]) -> bool:
            asked.append(frame.get("tool_name", ""))
            return frame.get("tool_name") == format_tool_name(WRITE)

        await stack.turn(
            [*every_write(), [read()]],
            on_card=allow_the_channel_write,
        )

        results = stack.conductor.results
        assert set(asked) == {format_tool_name(tool) for tool in WRITES}, asked
        for tool in WRITES:
            assert HELD not in results[tool][0], results[tool][0][:400]
        view = served_view(stack.repo)
        band = unmoved_band(view, SETPOINT, served_active(view))
        assert_unmoved(SETPOINT, 61.0, setpoint_reading(results[READ][0]), band)


async def test_a_write_approved_before_the_flag_is_refused_after_it(
    serve_stack: Callable[..., Any],
) -> None:
    """The operator approved this very write; a flag since then takes it back."""
    async with serve_stack() as stack:
        await stack.turn([[write(41.0)]])
        assert HELD not in stack.conductor.results[WRITE][0]
        assert not stack.flags()

        await stack.turn(
            [[write(41.0)]],
            flagging=True,
            before_turn={0: stack.wait_for_flag},
        )

        assert_held_refusal(stack.conductor.results[WRITE][0])


async def test_a_script_already_running_when_the_flag_opens_is_refused_at_its_write(
    serve_stack: Callable[..., Any],
) -> None:
    """Approved and launched before the flag: ``write_channel`` inside it reads the hold."""
    async with serve_stack() as stack:
        await stack.turn([[read()]])
        before = setpoint_reading(stack.conductor.results[READ][0])
        record = stack.root / stack.key / RECORD_NAME
        code = (
            "import json, time\n"
            "from pathlib import Path\n"
            "from osprey.runtime import write_channel\n"
            f"record = Path({str(record)!r})\n"
            "deadline = time.monotonic() + 90\n"
            "while time.monotonic() < deadline:\n"
            "    if json.loads(record.read_text())['flags']:\n"
            "        break\n"
            "    time.sleep(0.05)\n"
            "else:\n"
            "    raise SystemExit('no flag was seen')\n"
            "try:\n"
            f"    write_channel({SETPOINT!r}, 43.0)\n"
            "except Exception as exc:\n"
            "    print('REFUSED', exc)\n"
            "else:\n"
            "    print('WRITTEN')\n"
        )

        flagging: list[asyncio.Task[None]] = []

        async def flag_after_the_launch() -> None:
            await asyncio.sleep(0.5)
            await stack.open_flag_through_intake()

        async def approve_then_flag(_frame: dict[str, Any]) -> bool:
            # The answer goes out first: the script runs while the verdict is judged.
            flagging.append(asyncio.create_task(flag_after_the_launch()))
            return True

        await stack.turn([[execute(code)], [read()]], on_card=approve_then_flag)
        await asyncio.gather(*flagging)

        results = stack.conductor.results
        assert "REFUSED" in results[EXECUTE][0], results[EXECUTE][0][:600]
        assert "session_held" in results[EXECUTE][0], results[EXECUTE][0][:600]
        assert "WRITTEN" not in results[EXECUTE][0]
        assert_untouched(
            served_view(stack.repo),
            SETPOINT,
            before,
            setpoint_reading(results[READ][0]),
            43.0,
        )


@pytest.mark.parametrize(
    "call",
    [write(44.0), execute(write_channel_script(44.0))],
    ids=["channel_write", "execute"],
)
async def test_a_pending_approval_answered_after_the_flag_is_refused_at_execution(
    serve_stack: Callable[..., Any], call: ScriptedCall
) -> None:
    """Card shown, flag opens, operator says yes: the approval buys nothing."""
    async with serve_stack() as stack:
        cards: list[str] = []

        async def flag_then_approve(frame: dict[str, Any]) -> bool:
            cards.append(frame.get("tool_name", ""))
            await stack.open_flag_through_intake()
            return True

        await stack.turn([[call]], on_card=flag_then_approve)

        assert cards == [format_tool_name(call.name)]
        assert_held_refusal(stack.conductor.results[call.name][0])


async def test_a_dispatch_job_fired_while_held_is_refused_and_fires_after_continue(
    serve_stack: Callable[..., Any],
) -> None:
    """The proxy answers for the dispatcher: nothing reaches it until the release."""
    async with serve_stack() as stack:
        await stack.turn([[read()]], flagging=True, before_turn={0: stack.wait_for_flag})
        headers = {
            "accept": "application/json, text/event-stream",
            SESSION_HEADER: stack.key,
        }

        refused = await stack.client.post(EVENTS_MCP_ROUTE, headers=headers, json=FIRE_JOB)

        assert refused.status_code == 200, refused.text
        assert HELD in refused.json()["error"]["message"]
        assert stack.dispatcher.calls == []

        await stack.press_continue()
        fired = await stack.client.post(EVENTS_MCP_ROUTE, headers=headers, json=FIRE_JOB)

        assert fired.status_code == 200, fired.text
        assert "error" not in fired.json()
        assert [call["params"]["name"] for call in stack.dispatcher.calls] == ["manual_fire"]


async def test_an_unreachable_decision_service_costs_nothing_and_says_unavailable(
    serve_stack: Callable[..., Any],
) -> None:
    """Criterion 11: the prompt completes, no flag and no dialog, the status says unavailable."""
    # Port 9 on loopback is the discard port: nothing listens and the connection is refused.
    async with serve_stack(jev_url="http://127.0.0.1:9") as stack:

        def unavailable_frame(frame: dict[str, Any]) -> bool:
            return frame.get("type") == "watchdog_status" and frame.get("status") == "unavailable"

        def wait_for_status() -> None:
            stack.wait_for_frame(unavailable_frame)

        frames = await stack.turn([[read()]], before_turn={0: wait_for_status})

        assert [frame["is_error"] for frame in frames if frame.get("type") == "result"] == [False]
        assert stack.flags() == []
        assert await stack.held() is False
        assert not [f for f in stack.frames if f.get("type") == "watchdog_flag"]
        assert not [f for f in frames if f.get("type") == "watchdog_flag"]
        assert HELD not in stack.conductor.results[READ][0]
