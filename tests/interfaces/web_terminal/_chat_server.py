"""A live web terminal with the Claude Agent SDK faked at the operator-session seam.

:func:`live_chat_server` serves the real hub, chat routes and agent runner
with only the SDK subprocess replaced by a fake client that replays the plan
registered in :data:`PLANS` for each prompt. The chat browser suites share it;
the module-global fake state is reset at each launch.
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from claude_agent_sdk import TextBlock, ThinkingBlock, ToolUseBlock
from fastapi import Request

from osprey.agent_runner.project_paths import claude_project_dir
from osprey.interfaces._serving import run_app_server
from tests.interfaces._hub import LONG_LIVED_SHELL, hub_app
from tests.interfaces._panel_launch import publish_panel_urls

# The shared SDK-message builders, so the fake yields the real messages the
# agent runner translates.
from tests.interfaces.web_terminal._fakes import (
    assistant_message,
    result_message,
    system_message,
)

_SEAM = "osprey.interfaces.web_terminal.operator_session"
#: The name the agent runner constructs its client under.
_CLIENT = "osprey.agent_runner.session.ClaudeSDKClient"


# ---------------------------------------------------------------------------
# Faked SDK seam: per-prompt conversation plans + a controllable fake client
# ---------------------------------------------------------------------------
#
# All shared state below is module-global, which is safe because the browser
# fixture is function-scoped (tests never run concurrently) and every test
# registers its plans BEFORE the prompt that triggers them — reads happen on the
# server loop, writes on the test thread, always ordered by the round-trip.

# Conversation plans keyed by exact prompt text. Each value is a list of steps;
# see _FakeSDKClient.receive_response for the step vocabulary.
PLANS: dict[str, list[tuple]] = {}

# Every prompt the fake was queried with, across all sessions of the current
# server — lets a test assert the agent seam actually received each turn.
OBSERVED_PROMPTS: list[str] = []

# A gate the fake awaits on a ("gate",) step; POST /__test__/release opens it.
# Reassigned fresh per server (unbound to any loop until the server loop awaits
# it) so it never leaks a previous test's event loop.
_RELEASE_GATE = asyncio.Event()

# Default reply for an unregistered prompt: one line of text, then terminate.
_DEFAULT_PLAN: list[tuple] = [("text", "ok"), ("result",)]

# Every PTY command the hand-off door asked the registry to spawn, in order.
# Recorded at the registry seam (see _record_pty_spawns) because the identity
# half of a spawn — ``--resume <transcript>`` or ``--session-id <key>`` — is
# decided by the door and spelled on the command line, nowhere else.
_PTY_SPAWNS: list[list[str]] = []

# Every fake client constructed on the current server, in order; each records
# itself here so the options a chat session was launched with stay readable.
_CLIENTS: list[_FakeSDKClient] = []


def _reset_fake_state() -> None:
    """Clear per-server fake state; called at each server launch (test thread)."""
    global _RELEASE_GATE
    PLANS.clear()
    OBSERVED_PROMPTS.clear()
    _PTY_SPAWNS.clear()
    _CLIENTS.clear()
    _RELEASE_GATE = asyncio.Event()


class _FakeSDKClient:
    """Stand-in for ``ClaudeSDKClient``, patched where the agent runner builds it.

    ``receive_response`` answers only a prompt that was sent, as the SDK's does:
    it waits for the next ``query`` and replays that prompt's plan, yielding the
    real SDK messages the agent runner translates into the records
    ``_event_to_wire`` turns into chat events. Step vocabulary:

      ("text", md)      one text block (markdown) → a ``text`` event
      ("thinking",)     one thinking block        → drives the activity line
      ("tool_use", nm)  one tool_use block        → activity line phrase for <nm>
      ("system", sub)   a system message
      ("gate",)         block until POST /__test__/release
      ("hang",)         block until the reader is cancelled (Stop tests)
      ("await_interrupt") block until interrupt() (Stop → clean terminal)
      ("result",)       a terminal ResultMessage

    The real ``ClaudeAgentOptions`` the client was constructed with are kept as
    ``options``: they are what ``OperatorSession.start`` chose between
    ``resume=<transcript>`` and ``session_id=<key>``, and the
    ``/__test__/chat-state`` route serves them back for every client still open.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.options = kwargs.get("options")
        self.open = False
        # Bound to the server loop (constructed inside session.start()).
        self._interrupted = asyncio.Event()
        # The session-long reader calls receive_response before any prompt is sent.
        self._pending: asyncio.Queue[str] = asyncio.Queue()
        _CLIENTS.append(self)

    async def __aenter__(self):
        self.open = True
        return self

    async def __aexit__(self, *exc):
        self.open = False
        return False

    async def query(self, prompt: str) -> None:
        OBSERVED_PROMPTS.append(prompt)
        self._interrupted.clear()
        self._pending.put_nowait(prompt)

    async def interrupt(self) -> None:
        self._interrupted.set()

    async def receive_response(self):
        prompt = await self._pending.get()
        plan = PLANS.get(prompt, _DEFAULT_PLAN)
        for step in plan:
            kind = step[0]
            if kind == "text":
                yield assistant_message([TextBlock(step[1])])
            elif kind == "thinking":
                thinking = step[1] if len(step) > 1 else "…"
                yield assistant_message([ThinkingBlock(thinking, "sig")])
            elif kind == "tool_use":
                yield assistant_message([ToolUseBlock("tu_1", step[1], {})])
            elif kind == "system":
                yield system_message(step[1])
            elif kind == "gate":
                await _RELEASE_GATE.wait()
            elif kind == "hang":
                await asyncio.Event().wait()  # blocks until the task is cancelled
            elif kind == "await_interrupt":
                await self._interrupted.wait()
            elif kind == "result":
                yield result_message(is_error=(step[1] if len(step) > 1 else False))


# ---------------------------------------------------------------------------
# Test-only control routes (run on the server loop, driven over HTTP)
# ---------------------------------------------------------------------------


def _install_test_routes(app) -> None:
    """Mount /__test__/* helpers so scenarios drive server state from their loop.

    ``Request`` is imported at module scope (not here): ``from __future__ import
    annotations`` stringizes every annotation, and FastAPI resolves the string
    against the *module* globals — a function-local import leaves it unresolved,
    so the parameter is misread as a query field (422).
    """

    async def _release(request: Request):  # noqa: ARG001 - Request required by FastAPI
        _RELEASE_GATE.set()
        return {"released": True}

    async def _evict_all(request: Request):
        registry = request.app.state.operator_registry
        chat_ids = list(registry.chats._sessions.keys())
        for chat_id in chat_ids:
            await registry.terminate_chat_session(chat_id)
        return {"evicted": len(chat_ids)}

    async def _chat_state(request: Request):
        registry = request.app.state.operator_registry
        chats = list(registry.chats._sessions.values())
        return {
            "n": len(chats),
            "in_flight": any(s.in_flight for s in chats),
            # The identity each open chat client was launched with — the
            # question a hand-off answers.
            "options": [
                {"resume": c.options.resume, "session_id": c.options.session_id}
                for c in _CLIENTS
                if c.open and c.options is not None
            ],
        }

    async def _pty_spawns(request: Request):  # noqa: ARG001 - Request required by FastAPI
        return {"commands": [list(command) for command in _PTY_SPAWNS]}

    async def _pty_state(request: Request):
        registry = request.app.state.pty_registry
        key = request.query_params.get("session_id", "")
        session = registry.get_session(key) if key else None
        return {"live": bool(session is not None and session.is_alive)}

    app.add_api_route("/__test__/release", _release, methods=["POST"])
    app.add_api_route("/__test__/evict-all", _evict_all, methods=["POST"])
    app.add_api_route("/__test__/chat-state", _chat_state, methods=["GET"])
    app.add_api_route("/__test__/pty-spawns", _pty_spawns, methods=["GET"])
    app.add_api_route("/__test__/pty-state", _pty_state, methods=["GET"])


# ---------------------------------------------------------------------------
# The project the server serves: PTY child, turn hook, transcripts
# ---------------------------------------------------------------------------


def _write_turn_hook_settings(project_dir: Path) -> None:
    """Render the turn-state hook registrations a deployment's settings carry.

    ``app.state.turn_hook_present`` is read from these settings at startup and
    is a permission: without it the door refuses a plain hand-off off a PTY
    (``handoff_needs_interrupt``), because nothing can ever report the idle
    edge it would wait for. Writing the real registrations — rather than
    stamping the flag onto ``app.state`` — keeps the suite honest about which
    deployment it is describing, and runs the detection itself.

    All four events a rendered ``settings.json`` wires are written, the busy
    edge included, so the stand-in describes the deployment it stands in for
    rather than only the two registrations the detection happens to read.
    """
    from osprey.interfaces.web_terminal.app import (
        TURN_HOOK_SESSION_START_MATCHER,
        TURN_STATE_HOOK,
    )

    command = f"python3 .claude/hooks/{TURN_STATE_HOOK}"
    entry = [{"hooks": [{"type": "command", "command": command}]}]
    settings = {
        "hooks": {
            "UserPromptSubmit": entry,
            "Stop": entry,
            "StopFailure": entry,
            "SessionStart": [
                {
                    "matcher": TURN_HOOK_SESSION_START_MATCHER,
                    "hooks": [{"type": "command", "command": command}],
                }
            ],
        }
    }
    claude_dir = project_dir / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    (claude_dir / "settings.json").write_text(json.dumps(settings), encoding="utf-8")


def seed_transcript(project_dir: Path, transcript_id: str, turns) -> Path:
    """Write the JSONL transcript a Claude child would have left for *transcript_id*.

    A real transcript is written by a real ``claude`` process, which needs the
    binary and a provider; the conversation is therefore seeded in the shape
    the readers expect (``<config-dir>/projects/<encoded>/<id>.jsonl``, one
    entry per line). This single file is what the resume decision, the Simple
    view's replay and the PTY idle judgement all read.

    Args:
        project_dir: The project the server serves.
        transcript_id: The Claude session id the file is named for.
        turns: ``(role, text)`` pairs, oldest first. An assistant turn's text
            may instead be its list of content blocks (thinking, text, ...).

    Returns:
        The path written.
    """
    directory = claude_project_dir(project_dir)
    directory.mkdir(parents=True, exist_ok=True)
    lines = []
    for index, (role, text) in enumerate(turns):
        if role == "user" or isinstance(text, list):
            content = text
        else:
            content = [{"type": "text", "text": text}]
        lines.append(
            json.dumps(
                {
                    "type": role,
                    "timestamp": f"2026-01-01T00:00:{index:02d}.000Z",
                    "sessionId": transcript_id,
                    "message": {"role": role, "content": content},
                }
            )
        )
    path: Path = directory / f"{transcript_id}.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def seed_record_from(monkeypatch, transcript: Path) -> None:
    """Make the session routes' agent record serve *transcript*'s conversation.

    The record is a fake answering the session the transcript is named for with
    the nodes the record's own parser and timeline builder make of that file,
    and nothing for any other session.
    """
    from osprey.agent_record.client import ClientAnswer
    from osprey.interfaces.web_terminal.routes import session as session_routes
    from tests.agent_record.fakes import FakeRecordClient

    def timeline(**kwargs) -> ClientAnswer:
        named = transcript if kwargs.get("session_id") == transcript.stem else None
        nodes = session_routes._transcript_nodes(named) if named and named.is_file() else []
        return ClientAnswer(data=nodes, complete=True)

    record = FakeRecordClient({"timeline": timeline})
    monkeypatch.setattr(session_routes, "record_client", lambda: record)


def _record_pty_spawns(app) -> None:
    """Record every command the door hands the PTY registry, then spawn it.

    Wraps the live registry rather than the route, so what is recorded is the
    argv a spawn was actually made with.
    """
    registry = app.state.pty_registry
    inner = registry.get_or_create_session

    def get_or_create_session(session_id, command, **kwargs):
        _PTY_SPAWNS.append(list(command))
        return inner(session_id, command, **kwargs)

    registry.get_or_create_session = get_or_create_session


# ---------------------------------------------------------------------------
# Live-server context manager
# ---------------------------------------------------------------------------


@contextmanager
def live_chat_server(
    tmp_path,
    ui_mode: str = "simple",
    *,
    enabled_panels: frozenset[str] = frozenset({"artifacts"}),
    panel_urls: dict[str, str] | None = None,
):
    """Launch a real web terminal with the SDK faked at the operator_session seam.

    Mirrors the panels-browser ``_live_server`` patch set (web/panel config +
    artifact-server bypass) and adds the SDK seam: ``HAS_SDK`` on both the
    session and route modules, and the fake client where the agent runner
    constructs it. ``ui_mode``
    is applied post-startup (root() re-reads it per request), the same
    app.state seam the ui-mode browser suite uses.

    The project served is a fresh directory under ``tmp_path`` carrying the
    turn-state hook registrations, and both roots the session machinery writes
    or reads through — Claude's config directory (transcripts) and the
    agent-data root (posture and transcript-map stores) — are pinned beside it,
    so a run neither reads nor writes developer state.

    Args:
        enabled_panels: The builtin panel ids the hub enables.
        panel_urls: The address each companion server is published at, by
            registry key, for a panel a test serves itself. A panel not named
            is published unlaunched, so by default no companion tab is
            advertised.

    Yields:
        (base_url, app) — live server address and the FastAPI app. The project
        directory is ``tmp_path / "project"``.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir(exist_ok=True)
    project_dir = tmp_path / "project"
    project_dir.mkdir(exist_ok=True)
    _write_turn_hook_settings(project_dir)
    _reset_fake_state()

    patches = [
        patch.dict(
            os.environ,
            {
                "CLAUDE_CONFIG_DIR": str(tmp_path / "claude"),
                "OSPREY_AGENT_DATA_ROOT": str(tmp_path / "agent_data"),
            },
        ),
        # ---- Claude Agent SDK seam ----
        patch(f"{_SEAM}.HAS_SDK", True),
        patch("osprey.interfaces.web_terminal.routes.chat.HAS_SDK", True),
        patch(_CLIENT, _FakeSDKClient),
        patch(
            f"{_SEAM}.build_system_prompt",
            return_value={"type": "preset", "preset": "claude_code"},
        ),
        patch(f"{_SEAM}.get_facility_timezone", return_value=None),
    ]
    with hub_app(
        workspace,
        *patches,
        enabled_panels=enabled_panels,
        launch=publish_panel_urls(panel_urls or {}),
        shell_command=LONG_LIVED_SHELL,
        project_dir=project_dir,
    ) as app:
        _install_test_routes(app)
        with run_app_server(app) as base_url:
            app.state.web_ui_mode = ui_mode
            _record_pty_spawns(app)
            yield base_url, app
