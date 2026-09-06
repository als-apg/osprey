"""Browser tests: Simple-mode operator chat, SSE → live DOM.

Proves the operator chat console behaviors that the FastAPI TestClient cannot
reach because none of them exist until a real browser runs the vendored
markdown/sanitiser globals and the SSE transport against a live event stream:

  1. a streamed prompt renders as sanitised markdown in the chat card;
  2. a second prompt in the same page-load reaches the same session (continuity);
  3. the activity line (the tool's operator phrase) shows during a tool_use and
     clears on text;
  4. Stop mid-stream re-enables the input and the next prompt runs cleanly;
  5. the Expert/Simple toggle swaps the chat card ↔ xterm live, no reload;
  6. hostile model markdown renders inert in the live DOM (no script executes);
  7. a re-created chat with nothing to resume shows the "session reset" divider
     on the next turn — no divider when it resumes, none on a first turn;
  8. flipping the view hands ONE session between the two surfaces: the Simple
     view replays the Expert transcript and the Expert view resumes it again,
     the pointer unchanged and the page never reloaded;
  9. a flip with no transcript to resume starts the chat under the session key;
 10. the view taking the session over shows the transitional state while the
     outgoing agent finishes: ended by "Stop and switch now" on either side,
     and by the turn's own idle edge on the Simple side;
 11. a second tab holding the session gets the refusal in words.

Harness: the panels-browser ``_live_server`` machinery (a real uvicorn server on
a background thread, with ``_load_web_config``/``_load_panel_config``/
``_launch_artifact_server`` patched so no companion backends are needed), plus
the Claude Agent SDK faked at the ``operator_session`` seam. The fake
``ClaudeSDKClient`` replays a per-prompt *plan* of the same ``Fake*`` SDK message
objects the ``operator_session`` unit tests use, so the real
``_message_to_events`` converter and the real ``routes/chat.py`` SSE branch run
end to end — only the SDK subprocess is replaced. A handful of ``/__test__/*``
routes give each scenario a server-loop control channel (release a held turn,
force an eviction, read turn-guard state and what each SDK client was launched
with) without cross-thread event juggling.

The Expert half is real: a genuine PTY runs a long-lived child, and the command
the hand-off door built for it is recorded at the registry seam, so ``--resume
<transcript>`` versus ``--session-id <key>`` is asserted on the argv the spawn
actually used. What no fake can produce is the transcript a real Claude child
writes, so the conversation the Expert view "had" is seeded as a JSONL file in
the discovery directory — the same file the resume, the replay and the idle
judgement all read. The project directory, the Claude config root and the
agent-data root are all under ``tmp_path``, so a run reads no developer state.

Run:
    uv run pytest tests/interfaces/web_terminal/test_chat_browser.py -q

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
import requests
from fastapi import Request

from osprey.agent_runner.project_paths import claude_project_dir
from tests.interfaces._panel_launch import publish_artifact_url
from tests.interfaces.conftest import _apply_all, _run_app_server

# The Fake* SDK-message doubles live with the operator_session unit tests; reuse
# them so isinstance() inside _message_to_events matches what the fake yields.
from tests.interfaces.web_terminal.test_operator_session import (
    FakeAssistantMessage,
    FakeResultMessage,
    FakeSystemMessage,
    FakeTextBlock,
    FakeThinkingBlock,
    FakeToolResultBlock,
    FakeToolUseBlock,
)

# ---------------------------------------------------------------------------
# Playwright availability guard
# ---------------------------------------------------------------------------

try:
    from playwright.sync_api import Page, expect

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False

pytestmark = [pytest.mark.browser, pytest.mark.slow]

_SEAM = "osprey.interfaces.web_terminal.operator_session"
_OP = "#operator-container"


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
_PLANS: dict[str, list[tuple]] = {}

# Every prompt the fake was queried with, across all sessions of the current
# server — lets a test assert the agent seam actually received each turn.
_OBSERVED_PROMPTS: list[str] = []

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


def _reset_fake_state() -> None:
    """Clear per-server fake state; called at each server launch (test thread)."""
    global _RELEASE_GATE
    _PLANS.clear()
    _OBSERVED_PROMPTS.clear()
    _PTY_SPAWNS.clear()
    _RELEASE_GATE = asyncio.Event()


class _FakeSDKClient:
    """Stand-in for ``ClaudeSDKClient`` wired at the operator_session seam.

    ``receive_response`` replays the plan registered for the most recent prompt,
    yielding the ``Fake*`` SDK message objects the real ``_message_to_events``
    converts into chat events. Step vocabulary:

      ("text", md)      one text block (markdown) → a ``text`` event
      ("thinking",)     one thinking block        → drives the activity line
      ("tool_use", nm)  one tool_use block        → activity line phrase for <nm>
      ("system", sub)   a system message
      ("gate",)         block until POST /__test__/release
      ("hang",)         block until the reader is cancelled (Stop tests)
      ("await_interrupt") block until interrupt() (Stop → clean terminal)
      ("result",)       a terminal ResultMessage

    The options the client was constructed with are kept as ``options``: with
    ``ClaudeAgentOptions`` faked to a plain mapping, they are what
    ``OperatorSession.start`` chose between ``resume=<transcript>`` and
    ``session_id=<key>``, and the ``/__test__/chat-state`` route serves them
    back per live chat session.
    """

    def __init__(self, *args, **kwargs) -> None:
        self._prompts: list[str] = []
        self.options = kwargs.get("options")
        # Bound to the server loop (constructed inside session.start()).
        self._interrupted = asyncio.Event()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def query(self, prompt: str) -> None:
        self._prompts.append(prompt)
        _OBSERVED_PROMPTS.append(prompt)
        self._interrupted.clear()

    async def interrupt(self) -> None:
        self._interrupted.set()

    async def receive_response(self):
        plan = _PLANS.get(self._prompts[-1], _DEFAULT_PLAN)
        for step in plan:
            kind = step[0]
            if kind == "text":
                yield FakeAssistantMessage([FakeTextBlock(step[1])])
            elif kind == "thinking":
                yield FakeAssistantMessage([FakeThinkingBlock(step[1] if len(step) > 1 else "…")])
            elif kind == "tool_use":
                yield FakeAssistantMessage([FakeToolUseBlock(step[1], "tu_1", {})])
            elif kind == "system":
                yield FakeSystemMessage(step[1])
            elif kind == "gate":
                await _RELEASE_GATE.wait()
            elif kind == "hang":
                await asyncio.Event().wait()  # blocks until the task is cancelled
            elif kind == "await_interrupt":
                await self._interrupted.wait()
            elif kind == "result":
                yield FakeResultMessage(is_error=(step[1] if len(step) > 1 else False))


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
            # What each live chat's SDK client was launched with — the identity
            # half is the whole question a hand-off answers.
            "options": [getattr(getattr(s, "_client", None), "options", None) for s in chats],
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

# A PTY command that outlives the test AND tolerates the ``--resume <id>`` /
# ``--session-id <id>`` (and any ``--effort <level>``) arguments the websocket
# route appends. ``sleep``/``cat``/``echo`` each exit or error on the extra
# args, which would trip terminal.js's auto-resume failover and clear the
# stored pointer mid-test.
_LONG_LIVED_SHELL = [sys.executable, "-c", "import time; time.sleep(3600)"]


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


def _seed_transcript(project_dir: Path, transcript_id: str, turns) -> Path:
    """Write the JSONL transcript a Claude child would have left for *transcript_id*.

    A real transcript is written by a real ``claude`` process, which needs the
    binary and a provider; the conversation is therefore seeded in the shape
    the readers expect (``<config-dir>/projects/<encoded>/<id>.jsonl``, one
    entry per line). This single file is what the resume decision, the Simple
    view's replay and the PTY idle judgement all read.

    Args:
        project_dir: The project the server serves.
        transcript_id: The Claude session id the file is named for.
        turns: ``(role, text)`` pairs, oldest first.

    Returns:
        The path written.
    """
    directory = claude_project_dir(project_dir)
    directory.mkdir(parents=True, exist_ok=True)
    lines = []
    for index, (role, text) in enumerate(turns):
        content = text if role == "user" else [{"type": "text", "text": text}]
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
def _live_chat_server(tmp_path, ui_mode: str = "simple"):
    """Launch a real web terminal with the SDK faked at the operator_session seam.

    Mirrors the panels-browser ``_live_server`` patch set (web/panel config +
    artifact-server bypass) and adds the SDK seam: ``CLAUDE_SDK_AVAILABLE`` on
    both the session and route modules, the fake client, and the ``Fake*`` type
    globals so ``_message_to_events``'s isinstance checks match. ``ui_mode``
    is applied post-startup (root() re-reads it per request), the same
    app.state seam the ui-mode browser suite uses.

    The project served is a fresh directory under ``tmp_path`` carrying the
    turn-state hook registrations, and both roots the session machinery writes
    or reads through — Claude's config directory (transcripts) and the
    agent-data root (posture and transcript-map stores) — are pinned beside it,
    so a run neither reads nor writes developer state.

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
        patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace)},
        ),
        patch(
            "osprey.interfaces.web_terminal.app._load_panel_config",
            return_value=({"artifacts"}, [], None),
        ),
        patch(
            "osprey.interfaces.web_terminal.app._launch_panel_server",
            side_effect=publish_artifact_url(None),
        ),
        # ---- Claude Agent SDK seam ----
        patch(f"{_SEAM}.CLAUDE_SDK_AVAILABLE", True),
        patch("osprey.interfaces.web_terminal.routes.chat.CLAUDE_SDK_AVAILABLE", True),
        patch(f"{_SEAM}.ClaudeSDKClient", _FakeSDKClient),
        patch(f"{_SEAM}.ClaudeAgentOptions", lambda **kw: kw),
        patch(f"{_SEAM}.AssistantMessage", FakeAssistantMessage),
        patch(f"{_SEAM}.ResultMessage", FakeResultMessage),
        patch(f"{_SEAM}.SystemMessage", FakeSystemMessage),
        patch(f"{_SEAM}.TextBlock", FakeTextBlock),
        patch(f"{_SEAM}.ThinkingBlock", FakeThinkingBlock),
        patch(f"{_SEAM}.ToolUseBlock", FakeToolUseBlock),
        patch(f"{_SEAM}.ToolResultBlock", FakeToolResultBlock),
        patch(
            f"{_SEAM}.build_system_prompt",
            return_value={"type": "preset", "preset": "claude_code"},
        ),
        patch(f"{_SEAM}.get_facility_timezone", return_value=None),
    ]
    with _apply_all(patches):
        from osprey.interfaces.web_terminal.app import create_app

        app = create_app(shell_command=_LONG_LIVED_SHELL, project_dir=project_dir)
        _install_test_routes(app)
        with _run_app_server(app) as base_url:
            app.state.web_ui_mode = ui_mode
            _record_pty_spawns(app)
            yield base_url, app


# ---------------------------------------------------------------------------
# Page + interaction helpers
# ---------------------------------------------------------------------------


# Two probes, both installed before any page script runs.
#
# The hand-off log answers "was the transitional state on screen?" without
# racing it. Both overlays are shown by clearing `hidden` (the console builds
# its card up front) or by being inserted (the terminal builds its own on
# demand), so a mutation observer sees every appearance no matter how briefly
# it stands — a poll would not. Its callback runs at the microtask checkpoint,
# after the synchronous block that wrote the copy.
#
# The load counter answers "did the page reload?". A flip is a live swap of two
# surfaces that are both already in the document; a reload would restart every
# script, drop the xterm and re-run the boot ladder, and is exactly the
# implementation this feature must not have. sessionStorage survives a reload,
# which a window flag would not, so the count is the honest witness — and it is
# counted for the TOP document only, since the hub's same-origin panel iframes
# share that storage and would each inflate the count on their own load.
_PROBE_INIT_SCRIPT = """
(function () {
  if (window.top === window) {
    try {
      var loads = Number(sessionStorage.getItem('__osprey_loads') || '0') + 1;
      sessionStorage.setItem('__osprey_loads', String(loads));
    } catch (e) {}
  }
  window.__handoffLog = [];
  function note() {
    ['#operator-container .op-handoff', '#terminal-handoff'].forEach(function (selector) {
      var el = document.querySelector(selector);
      // `hidden` is how the console toggles its card; offsetParent catches the
      // CSS side (the terminal overlay is display:none in the Simple view), so
      // the log records only copy that was actually on screen.
      if (!el || el.hidden || el.offsetParent === null) return;
      var text = (el.textContent || '').trim();
      var log = window.__handoffLog;
      if (text && text !== log[log.length - 1]) log.push(text);
    });
  }
  // Observed on the document, not the document element: at document-start
  // `document` always exists but `document.documentElement` may not yet, and
  // observing null throws out of this whole probe.
  new MutationObserver(note).observe(document, {
    subtree: true,
    childList: true,
    attributes: true,
    attributeFilter: ['hidden', 'class'],
  });
})();
"""


def _open_chat_page(opener, base_url: str, query: str = "") -> Page:
    """Open a fresh page (on a browser or a context) and wait for the console."""
    page: Page = opener.new_page()
    page.add_init_script(_PROBE_INIT_SCRIPT)
    page.goto(f"{base_url}{query}", wait_until="domcontentloaded")
    # initChat builds the console on DOMContentLoaded; the input row is the
    # last thing appended, so its presence means the console is mounted.
    expect(page.locator(f"{_OP} .op-input-area textarea")).to_be_visible(timeout=10_000)
    return page


def _open_expert_page(opener, base_url: str, query: str = "") -> Page:
    """Open a fresh page in the Expert view and wait for its terminal to connect."""
    page: Page = opener.new_page()
    page.add_init_script(_PROBE_INIT_SCRIPT)
    page.goto(f"{base_url}{query}", wait_until="domcontentloaded")
    expect(page.locator("#terminal-container .xterm")).to_be_visible(timeout=10_000)
    return page


def _click_segment(page: Page, mode: str) -> None:
    """Pick *mode* in the header's display-menu popover, then close the card."""
    trigger = page.locator("#display-menu .display-menu-trigger")
    card = page.locator("#display-menu .display-menu-card")
    trigger.click()
    expect(card).to_have_class(re.compile(r"\bopen\b"))
    page.locator(
        f'#display-menu .display-menu-view .display-seg-option[data-mode="{mode}"]'
    ).click()
    # A mode pick deliberately leaves the card open; close it so the surface
    # under it is the one the next step interacts with.
    trigger.click()
    expect(card).not_to_have_class(re.compile(r"\bopen\b"))


def _pointer(page: Page) -> str | None:
    """The session key this tab is on, read through the pointer module itself."""
    key: str | None = page.evaluate(
        "() => import('/static/js/session-pointer.js').then((m) => m.getPointer())"
    )
    return key


def _wait_for_pointer(page: Page, timeout: float = 15.0) -> str:
    """Block until the page is on a session key and return it."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        key = _pointer(page)
        if key:
            return key
        page.wait_for_timeout(50)
    raise AssertionError("the page never stored a session pointer")


def _handoff_log(page: Page) -> list[str]:
    """Every distinct line the hand-off overlays have shown, in order."""
    lines: list[str] = page.evaluate("() => window.__handoffLog || []")
    return lines


def _page_loads(page: Page) -> int:
    """How many times this tab has run the boot scripts."""
    return int(page.evaluate("() => Number(sessionStorage.getItem('__osprey_loads') || '0')"))


def _pty_commands(base_url: str) -> list[list[str]]:
    """Every PTY command the hand-off door has spawned, in order."""
    resp = requests.get(f"{base_url}/__test__/pty-spawns")
    resp.raise_for_status()
    commands: list[list[str]] = resp.json()["commands"]
    return commands


def _chat_options(base_url: str) -> list[dict]:
    """The SDK options every live chat session was launched with."""
    resp = requests.get(f"{base_url}/__test__/chat-state")
    resp.raise_for_status()
    return [options for options in resp.json()["options"] if options]


def _identity_of(command: list[str]) -> tuple[str, str]:
    """The identity argument pair of a PTY command.

    Asserted rather than a whole-argv comparison because the route appends
    other arguments (an effort level, say) that this suite is not about.
    """
    for flag in ("--resume", "--session-id"):
        if flag in command:
            return flag, command[command.index(flag) + 1]
    raise AssertionError(f"no identity argument in {command}")


def _wait_for_pty_spawns(base_url: str, count: int, timeout: float = 20.0) -> list[list[str]]:
    """Block until the door has spawned *count* PTY children and return them all."""
    deadline = time.monotonic() + timeout
    commands: list[list[str]] = []
    while time.monotonic() < deadline:
        commands = _pty_commands(base_url)
        if len(commands) >= count:
            return commands
        time.sleep(0.05)
    raise AssertionError(f"expected {count} PTY spawns, saw {commands}")


def _report_turn(base_url: str, key: str, state: str) -> None:
    """Report a turn edge for *key* the way the terminal's own child process does.

    The real route, not a poke at ``app.state``: it is the interface contract
    the ``osprey_turn_state`` hook posts on, and going through it runs the
    identifier grammar and the surface rule that decide whether an edge is
    recorded at all.
    """
    resp = requests.post(
        f"{base_url}/api/agent-turn",
        json={
            "session_id": key,
            "pool_key": key,
            "state": state,
            "surface": "expert",
            "ts": time.time(),
            "source": "UserPromptSubmit" if state == "busy" else "Stop",
        },
    )
    resp.raise_for_status()
    assert resp.json()["recorded"] is True, f"the {state} edge was not recorded: {resp.text}"


def _pty_is_live(base_url: str, key: str) -> bool:
    """Whether a PTY for *key* is still pooled and running."""
    resp = requests.get(f"{base_url}/__test__/pty-state", params={"session_id": key})
    resp.raise_for_status()
    return bool(resp.json()["live"])


def _wait_for_chat_options(base_url: str, timeout: float = 20.0) -> list[dict]:
    """Block until a chat session is live and return what each was launched with."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        options = _chat_options(base_url)
        if options:
            return options
        time.sleep(0.05)
    raise AssertionError("no chat session was launched")


def _send(page: Page, text: str) -> None:
    """Type a prompt and submit it (Enter, no Shift → submit)."""
    textarea = page.locator(f"{_OP} .op-input-area textarea")
    textarea.fill(text)
    textarea.press("Enter")


def _wait_chat_idle(base_url: str, timeout: float = 10.0) -> None:
    """Block until no chat turn holds the server-side guard.

    A server-state barrier (not a UI wait): after a client-side Stop the browser
    goes idle the instant the fetch aborts, but the turn guard is released a beat
    later when the server observes the disconnect/terminal. Polling that state
    keeps a follow-up prompt from racing a 409.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = requests.get(f"{base_url}/__test__/chat-state")
        if resp.ok and not resp.json().get("in_flight"):
            return
        time.sleep(0.05)
    raise AssertionError("chat session did not release its turn guard in time")


# ---------------------------------------------------------------------------
# 1. Streamed markdown renders as sanitised HTML in the chat card
# ---------------------------------------------------------------------------


def test_streamed_markdown_renders_in_chat_card(tmp_path, chromium_browser):
    """A prompt's reply renders as real markdown HTML, not inert text.

    Asserts the sanitised-markdown path produced actual elements — ``<strong>``
    from ``**bold**``, ``<code>`` from an inline span, and a fenced ``` ``` ```
    code block with its content intact — rather than the ``textContent`` fallback
    the renderer degrades to when the vendored libraries are missing. The fenced
    block also guards the chat-isolated Marked instance: it must render code
    bodies even though the scaffold gallery reconfigures the shared ``marked``
    singleton on the same page.
    """
    with _live_chat_server(tmp_path) as (base_url, _app):
        _PLANS["show me markup"] = [
            ("text", "Here is **bold** and `inline_code`:\n\n```python\nprint('hi')\n```\n"),
            ("result",),
        ]
        page = _open_chat_page(chromium_browser, base_url)
        _send(page, "show me markup")

        body = page.locator(f"{_OP} .op-entry.assistant .osprey-md-rendered")
        expect(body.locator("strong")).to_have_text("bold", timeout=10_000)
        expect(body.locator("code").first).to_have_text("inline_code")
        expect(body.locator("pre code")).to_contain_text("print('hi')")

        page.close()


# ---------------------------------------------------------------------------
# 2. Multi-turn continuity: second prompt reaches the same session
# ---------------------------------------------------------------------------


@pytest.mark.flaky(
    reruns=2, only_rerun=["AssertionError"]
)  # browser timing under load; passes in isolation
def test_multi_turn_reaches_same_session(tmp_path, chromium_browser):
    """A second prompt in the same page-load reuses the session; both show."""
    with _live_chat_server(tmp_path) as (base_url, _app):
        _PLANS["first question"] = [("text", "first answer"), ("result",)]
        _PLANS["second question"] = [("text", "second answer"), ("result",)]
        page = _open_chat_page(chromium_browser, base_url)

        _send(page, "first question")
        expect(page.locator(f"{_OP} .op-entry.assistant")).to_contain_text(
            "first answer", timeout=10_000
        )
        # Turn must end (input re-enabled) before the second turn is submitted.
        expect(page.locator(f"{_OP} .op-input-area textarea")).to_be_enabled()

        _send(page, "second question")
        expect(page.locator(f"{_OP} .op-entry.assistant").last).to_contain_text(
            "second answer", timeout=10_000
        )

        # Both exchanges are on screen...
        expect(page.locator(f"{_OP} .op-entry.operator")).to_have_count(2)
        expect(page.locator(f"{_OP} .op-entry.assistant")).to_have_count(2)
        # ...and the SDK seam saw both prompts, in order (one reused session).
        assert _OBSERVED_PROMPTS == ["first question", "second question"]

        page.close()


# ---------------------------------------------------------------------------
# 3. Activity line during a tool_use, cleared on text
# ---------------------------------------------------------------------------


def test_activity_line_shows_tool_then_clears_on_text(tmp_path, chromium_browser):
    """Bash's phrase is visible while the turn is held, and clears once text lands."""
    with _live_chat_server(tmp_path) as (base_url, _app):
        _PLANS["run a tool"] = [
            ("tool_use", "Bash"),
            ("gate",),  # hold the turn so the activity line is observable
            ("text", "done running"),
            ("result",),
        ]
        page = _open_chat_page(chromium_browser, base_url)
        _send(page, "run a tool")

        activity = page.locator(f"{_OP} .op-processing")
        expect(activity).to_be_visible(timeout=10_000)
        # chat-render maps tool names to operator phrases; Bash is a mapped name,
        # so the line reads as a sentence rather than echoing the raw tool.
        expect(activity).to_contain_text("Running a shell command")

        # Release the held turn: text arrives, the activity line clears.
        requests.post(f"{base_url}/__test__/release")
        expect(page.locator(f"{_OP} .op-entry.assistant .osprey-md-rendered")).to_contain_text(
            "done running", timeout=10_000
        )
        expect(activity).to_be_hidden()

        page.close()


# ---------------------------------------------------------------------------
# 4. Stop mid-stream: input re-enabled, streaming cleared, next prompt clean
# ---------------------------------------------------------------------------


def test_stop_mid_stream_reenables_and_next_prompt_works(tmp_path, chromium_browser):
    """Stop clears the streaming affordance and re-enables input; a follow-up runs."""
    with _live_chat_server(tmp_path) as (base_url, _app):
        _PLANS["long task"] = [("text", "starting…"), ("await_interrupt",), ("result",)]
        _PLANS["quick follow up"] = [("text", "all clear"), ("result",)]
        page = _open_chat_page(chromium_browser, base_url)
        _send(page, "long task")

        container = page.locator(_OP)
        stop_btn = page.locator(f"{_OP} .op-stop-btn")
        textarea = page.locator(f"{_OP} .op-input-area textarea")

        # Turn is live: streaming edge on, Stop shown, input disabled, partial text.
        expect(container).to_have_class(re.compile(r"\bstreaming\b"), timeout=10_000)
        expect(stop_btn).to_be_visible()
        expect(textarea).to_be_disabled()
        expect(page.locator(f"{_OP} .op-entry.assistant")).to_contain_text("starting…")

        stop_btn.click()

        # Streaming affordance cleared, Stop hidden, input re-enabled.
        expect(container).not_to_have_class(re.compile(r"\bstreaming\b"), timeout=10_000)
        expect(stop_btn).to_be_hidden()
        expect(textarea).to_be_enabled()

        # Next prompt runs cleanly (guard freed server-side first).
        _wait_chat_idle(base_url)
        _send(page, "quick follow up")
        expect(page.locator(f"{_OP} .op-entry.assistant").last).to_contain_text(
            "all clear", timeout=10_000
        )

        page.close()


# ---------------------------------------------------------------------------
# 4b. Send holds its position while Stop comes and goes
# ---------------------------------------------------------------------------


def test_send_button_does_not_move_when_stop_appears_and_leaves(tmp_path, chromium_browser):
    """Send is pinned to the composer's right edge; only Stop moves, inboard of it.

    The textarea takes all the slack (``flex: 1``), so the button cluster is
    anchored to the row's right edge and a member's arrival shifts everything
    BEFORE it and nothing after it. Stop is the conditional member, so it has to
    come FIRST in the cluster or Send slides sideways on every turn.

    The end of a turn is what makes this worth a browser test rather than a
    child-order assertion: ``setStreaming(false)`` hides Stop and re-enables Send
    in the same statement block, so the wrong order drops a live Send onto the
    exact pixels a hand was already moving toward for Stop. Asserted on real
    geometry, because the invariant is "the operator's target does not move" —
    a future layout change could honour the child order and still break it.
    """
    with _live_chat_server(tmp_path) as (base_url, _app):
        _PLANS["long task"] = [("text", "starting…"), ("await_interrupt",), ("result",)]
        page = _open_chat_page(chromium_browser, base_url)

        send_btn = page.locator(f"{_OP} .op-send-btn")
        stop_btn = page.locator(f"{_OP} .op-stop-btn")
        expect(send_btn).to_be_visible()
        idle_x = send_btn.bounding_box()["x"]

        _send(page, "long task")
        expect(stop_btn).to_be_visible(timeout=10_000)
        streaming_x = send_btn.bounding_box()["x"]

        stop_btn.click()
        expect(stop_btn).to_be_hidden(timeout=10_000)
        _wait_chat_idle(base_url)
        settled_x = send_btn.bounding_box()["x"]

        assert streaming_x == pytest.approx(idle_x, abs=1.0), (
            "Send moved when Stop appeared — Stop must be the first child of "
            ".op-input-controls so it opens inboard of Send"
        )
        assert settled_x == pytest.approx(idle_x, abs=1.0), (
            "Send moved back when Stop went away, landing on the pixels Stop had "
            "occupied at the moment Send became clickable again"
        )

        page.close()


# ---------------------------------------------------------------------------
# 5. Mode flip swaps chat card ↔ xterm live, both directions, no reload
# ---------------------------------------------------------------------------


def test_mode_flip_swaps_chat_and_terminal_live(tmp_path, chromium_browser):
    """The header toggle swaps console ↔ xterm via CSS; both stay in the DOM."""
    with _live_chat_server(tmp_path, ui_mode="simple") as (base_url, _app):
        page = _open_chat_page(chromium_browser, base_url)
        html = page.locator("html")
        console = page.locator(_OP)
        term = page.locator("#terminal-container")

        # Simple: console visible, xterm hidden.
        expect(html).to_have_attribute("data-ui-mode", "simple")
        expect(console).to_be_visible()
        expect(term).to_be_hidden()

        # → Expert: xterm visible, console hidden — no reload, both attached.
        _click_segment(page, "expert")
        expect(html).to_have_attribute("data-ui-mode", "expert")
        expect(term).to_be_visible()
        expect(console).to_be_hidden()
        expect(console).to_be_attached()
        expect(term).to_be_attached()

        # → Simple again: the swap is reversible with no teardown.
        _click_segment(page, "simple")
        expect(html).to_have_attribute("data-ui-mode", "simple")
        expect(console).to_be_visible()
        expect(term).to_be_hidden()
        expect(console).to_be_attached()
        expect(term).to_be_attached()

        page.close()


# ---------------------------------------------------------------------------
# 6. Hostile model markdown renders inert in the live DOM
# ---------------------------------------------------------------------------


def test_hostile_markdown_renders_inert(tmp_path, chromium_browser):
    """A <script>/onerror payload from the model is sanitised; nothing executes."""
    with _live_chat_server(tmp_path) as (base_url, _app):
        _PLANS["be evil"] = [
            (
                "text",
                "Safe **bold** "
                '<img src=x onerror="window.__osprey_xss = true"> '
                "<script>window.__osprey_xss = true</script>",
            ),
            ("result",),
        ]
        page = _open_chat_page(chromium_browser, base_url)
        # Any dialog the payload might raise is auto-dismissed (and would still
        # fail the window-flag assertion below), so the test can never hang.
        page.on("dialog", lambda d: d.dismiss())

        _send(page, "be evil")

        body = page.locator(f"{_OP} .op-entry.assistant .osprey-md-rendered")
        # The markdown path ran (not the inert-text fallback): **bold** → <strong>.
        expect(body.locator("strong")).to_have_text("bold", timeout=10_000)
        # DOMPurify stripped the script node and the inline handler.
        expect(body.locator("script")).to_have_count(0)
        onerror_count = page.evaluate(f"() => document.querySelectorAll('{_OP} [onerror]').length")
        assert onerror_count == 0, "an onerror handler survived sanitisation"
        # The injected globals never ran (undefined → not True).
        assert page.evaluate("() => window.__osprey_xss === true") is False

        page.close()


# ---------------------------------------------------------------------------
# 7. Session-expiry divider: a conversation that could not be continued
# ---------------------------------------------------------------------------


def test_session_expiry_divider_after_eviction(tmp_path, chromium_browser):
    """An eviction with nothing to resume makes the next turn show the divider.

    What paints the divider is the conversation ending, not the process dying.
    The evicted key has no transcript on disk here — the faked SDK writes none
    — so the re-created session has nothing to continue and starts a
    conversation of its own, which is what re-emits ``session_reset`` while
    prior turns are on screen. Its counterpart is the resume below, where the
    same eviction draws nothing.
    """
    with _live_chat_server(tmp_path) as (base_url, _app):
        _PLANS["turn one"] = [("text", "answer one"), ("result",)]
        _PLANS["turn two"] = [("text", "answer two"), ("result",)]
        page = _open_chat_page(chromium_browser, base_url)

        _send(page, "turn one")
        expect(page.locator(f"{_OP} .op-entry.assistant")).to_contain_text(
            "answer one", timeout=10_000
        )
        expect(page.locator(f"{_OP} .op-input-area textarea")).to_be_enabled()

        # Count dividers BEFORE the eviction and assert the eviction adds exactly
        # one more. Measuring the delta keeps this test correct whether or not the
        # separate first-turn-divider bug is present.
        divider = page.locator(f"{_OP} .op-system").filter(has_text="session reset")
        before = divider.count()

        _wait_chat_idle(base_url)
        resp = requests.post(f"{base_url}/__test__/evict-all")
        assert resp.json()["evicted"] >= 1

        _send(page, "turn two")
        # The eviction's session_reset paints a fresh divider (prior turns present).
        expect(divider).to_have_count(before + 1, timeout=10_000)
        expect(page.locator(f"{_OP} .op-entry.assistant").last).to_contain_text("answer two")

        page.close()


def test_no_divider_when_the_recreated_session_resumes(tmp_path, chromium_browser):
    """An eviction the key can be resumed through draws no divider at all.

    The same forced eviction as above, on a key whose transcript is on disk:
    the re-created session continues that transcript, so the conversation the
    operator is reading never ended and marking a boundary in it would be a
    claim the server did not make. The divider is reserved for the case that
    genuinely lost the thread.
    """
    with _live_chat_server(tmp_path) as (base_url, _app):
        _PLANS["turn one"] = [("text", "answer one"), ("result",)]
        _PLANS["turn two"] = [("text", "answer two"), ("result",)]
        page = _open_chat_page(chromium_browser, base_url)
        key = _wait_for_pointer(page)

        _send(page, "turn one")
        expect(page.locator(f"{_OP} .op-entry.assistant")).to_contain_text(
            "answer one", timeout=10_000
        )
        expect(page.locator(f"{_OP} .op-input-area textarea")).to_be_enabled()

        # The transcript that turn would have written, so the key has something
        # to be resumed through.
        _seed_transcript(
            tmp_path / "project", key, [("user", "turn one"), ("assistant", "answer one")]
        )

        _wait_chat_idle(base_url)
        resp = requests.post(f"{base_url}/__test__/evict-all")
        assert resp.json()["evicted"] >= 1

        _send(page, "turn two")
        expect(page.locator(f"{_OP} .op-entry.assistant").last).to_contain_text(
            "answer two", timeout=10_000
        )
        # The re-created session resumed rather than started.
        options = _wait_for_chat_options(base_url)
        assert [entry.get("resume") for entry in options] == [key]
        expect(page.locator(f"{_OP} .op-system").filter(has_text="session reset")).to_have_count(0)

        page.close()


def test_no_session_reset_divider_on_fresh_first_turn(tmp_path, chromium_browser):
    """A fresh page's very first turn must NOT show a "session reset" divider.

    The renderer's ``hasPriorExchange`` gate suppresses the first turn's
    ``session_reset`` even though the controller renders the user message before
    the stream starts — so no spurious divider paints under the operator's very
    first prompt. (Regression guard for the first-turn-divider fix.)
    """
    with _live_chat_server(tmp_path) as (base_url, _app):
        _PLANS["hello there"] = [("text", "hi back"), ("result",)]
        page = _open_chat_page(chromium_browser, base_url)

        _send(page, "hello there")
        expect(page.locator(f"{_OP} .op-entry.assistant")).to_contain_text(
            "hi back", timeout=10_000
        )
        # No session-reset divider should exist after a fresh first turn.
        expect(page.locator(f"{_OP} .op-system").filter(has_text="session reset")).to_have_count(0)

        page.close()


# ---------------------------------------------------------------------------
# 8. A flip hands ONE session over: Simple replays it, Expert resumes it back
# ---------------------------------------------------------------------------


def test_handoff_expert_to_simple_and_back_keeps_one_session(tmp_path, chromium_browser):
    """Both views are windows onto one session key and one conversation.

    The Expert view starts a child under the key; the flip to Simple shows the
    transitional state, replays that key's transcript into the console and
    hands back a usable input, with the chat resuming the transcript rather
    than opening a second conversation; the flip back spawns a terminal child
    that resumes the same transcript. Through both flips the browser's pointer
    is the same string and the page never reloads — the surfaces are swapped
    live, which is the whole point of the mechanic.
    """
    with _live_chat_server(tmp_path, ui_mode="expert") as (base_url, _app):
        _PLANS["and then"] = [("text", "the newer answer"), ("result",)]
        page = _open_expert_page(chromium_browser, base_url)
        key = _wait_for_pointer(page)

        # Nothing to resume yet, so the terminal child was started under the key.
        first = _wait_for_pty_spawns(base_url, 1)[0]
        assert _identity_of(first) == ("--session-id", key)

        # The conversation that view then had. Seeded, not generated: a real
        # transcript is written by a real Claude child (see _seed_transcript).
        _seed_transcript(
            tmp_path / "project",
            key,
            [("user", "what is the beam current"), ("assistant", "the earlier answer")],
        )

        _click_segment(page, "simple")

        # The transcript is replayed into the console...
        expect(page.locator(f"{_OP} .op-entry.operator")).to_contain_text(
            "what is the beam current", timeout=20_000
        )
        expect(page.locator(f"{_OP} .op-entry.assistant")).to_contain_text("the earlier answer")
        # ...the input is usable again...
        expect(page.locator(f"{_OP} .op-input-area textarea")).to_be_enabled()
        # ...and the wait was on screen while it happened.
        assert any("Finishing in the other view" in line for line in _handoff_log(page)), (
            f"the transitional state was never shown: {_handoff_log(page)}"
        )

        # The chat continued the transcript rather than starting a conversation
        # of its own under the key.
        options = _wait_for_chat_options(base_url)
        assert [entry.get("resume") for entry in options] == [key]
        assert all(entry.get("session_id") is None for entry in options)

        # The console is live on that same conversation.
        _send(page, "and then")
        expect(page.locator(f"{_OP} .op-entry.assistant").last).to_contain_text(
            "the newer answer", timeout=10_000
        )

        _click_segment(page, "expert")

        expect(page.locator("#terminal-container")).to_be_visible()
        second = _wait_for_pty_spawns(base_url, 2)[1]
        assert _identity_of(second) == ("--resume", key)

        assert _pointer(page) == key, "the flip moved the tab to another session"
        assert _page_loads(page) == 1, "the flip reloaded the page"

        page.close()


# ---------------------------------------------------------------------------
# 9. A flip with nothing to resume starts the chat under the session key
# ---------------------------------------------------------------------------


def test_handoff_before_any_prompt_starts_the_chat_under_the_key(tmp_path, chromium_browser):
    """With no transcript on disk the Simple view opens the key's first conversation.

    The counterpart of the resume above: the same flip, made before anything
    has been said, must not ask the SDK to continue a transcript that does not
    exist. It starts one under the session key instead — so the id the child
    writes its transcript under is the key both views are already on — and the
    pointer is untouched by the flip.
    """
    with _live_chat_server(tmp_path, ui_mode="expert") as (base_url, _app):
        page = _open_expert_page(chromium_browser, base_url)
        key = _wait_for_pointer(page)
        first = _wait_for_pty_spawns(base_url, 1)[0]
        assert _identity_of(first) == ("--session-id", key)

        _click_segment(page, "simple")

        expect(page.locator(f"{_OP} .op-input-area textarea")).to_be_enabled(timeout=20_000)
        options = _wait_for_chat_options(base_url)
        assert [entry.get("session_id") for entry in options] == [key]
        assert all(entry.get("resume") is None for entry in options)

        assert _pointer(page) == key
        assert _page_loads(page) == 1

        page.close()


# ---------------------------------------------------------------------------
# 10. The Expert side of a flip: the wait, and the way out of it
# ---------------------------------------------------------------------------


def test_handoff_pending_shows_the_wait_and_stop_switches_now(tmp_path, chromium_browser):
    """A flip to Expert waits out the chat's turn, and one button ends that wait.

    Only one surface may run the session's agent, and the outgoing one is never
    cut off on a clock — so a flip made mid-turn parks on the terminal card
    behind a live elapsed counter. "Stop and switch now" is the operator's way
    out: it asks for the same session again with the interrupt flag, which ends
    the running turn instead of waiting for it, and the terminal comes up on
    the key.
    """
    with _live_chat_server(tmp_path, ui_mode="simple") as (base_url, _app):
        # The turn never finishes on its own: the gate is deliberately never
        # released, so the wait the operator sees is real.
        _PLANS["hold the line"] = [("text", "working"), ("gate",), ("result",)]
        page = _open_chat_page(chromium_browser, base_url)
        key = _wait_for_pointer(page)

        _send(page, "hold the line")
        expect(page.locator(f"{_OP} .op-entry.assistant")).to_contain_text(
            "working", timeout=10_000
        )

        _click_segment(page, "expert")

        overlay = page.locator("#terminal-handoff")
        expect(overlay).to_be_visible(timeout=10_000)
        expect(overlay.locator(".terminal-handoff-message")).to_contain_text(
            "Finishing in the other view", timeout=10_000
        )
        # The counter is the honest thing to show for a wait with no bound.
        expect(overlay.locator(".terminal-handoff-elapsed")).to_have_text(
            re.compile(r"^\d+:\d{2}$")
        )

        action = overlay.locator(".terminal-handoff-action")
        expect(action).to_have_text("Stop and switch now")
        action.click()

        # The interrupt ends the chat turn, and the terminal takes the key. The
        # faked SDK writes no transcript, so there is nothing to resume and the
        # terminal child starts under the key itself.
        spawned = _wait_for_pty_spawns(base_url, 1)[0]
        assert _identity_of(spawned) == ("--session-id", key)
        expect(overlay).to_be_hidden(timeout=15_000)
        assert _pointer(page) == key
        assert _page_loads(page) == 1

        page.close()


def _flip_into_a_busy_terminal(base_url: str, chromium_browser) -> tuple[Page, str]:
    """Open Expert, mark the key mid-turn, flip to Simple, and return at the wait.

    Shared by the two cases below because the setup is the whole scenario up to
    the fork: what the operator does about the wait is what differs.
    """
    page = _open_expert_page(chromium_browser, base_url)
    key = _wait_for_pointer(page)
    _wait_for_pty_spawns(base_url, 1)
    assert _pty_is_live(base_url, key)

    # The terminal's own child reports a prompt going in — the hook's busy edge.
    _report_turn(base_url, key, "busy")

    _click_segment(page, "simple")

    overlay = page.locator(f"{_OP} .op-handoff")
    expect(overlay).to_be_visible(timeout=10_000)
    expect(overlay.locator(".op-handoff-message")).to_contain_text("Finishing in the other view")
    expect(overlay.locator(".op-handoff-elapsed")).to_have_text(re.compile(r"^\d+:\d{2}$"))
    # The input stays out of reach: the session is not this view's yet.
    expect(page.locator(f"{_OP} .op-input-area textarea")).to_be_disabled()
    expect(overlay.locator(".op-handoff-action")).to_have_text("Stop and switch now")
    return page, key


def test_handoff_busy_expert_to_simple_waits_for_the_turn(tmp_path, chromium_browser):
    """A flip made while the terminal is mid-turn waits, then completes on the idle edge.

    The mirror of the case above, on the surface whose turns are reported by a
    hook rather than read off an SDK client. The outgoing agent is never ended
    on a clock, so the console parks behind the transitional state rather than
    taking the session away mid-answer — and the edge that ends the turn is
    what releases it. The terminal's process is torn down and the console comes
    up on the same key.
    """
    with _live_chat_server(tmp_path, ui_mode="expert") as (base_url, _app):
        page, key = _flip_into_a_busy_terminal(base_url, chromium_browser)

        # The turn ends: the hook's idle edge is the only thing that says so.
        assert _pty_is_live(base_url, key), "the wait ended before the turn did"
        _report_turn(base_url, key, "idle")

        expect(page.locator(f"{_OP} .op-input-area textarea")).to_be_enabled(timeout=20_000)
        expect(page.locator(f"{_OP} .op-handoff")).to_be_hidden()
        assert not _pty_is_live(base_url, key), "the terminal kept the session it handed over"
        assert [entry.get("session_id") for entry in _wait_for_chat_options(base_url)] == [key]
        assert _pointer(page) == key
        assert _page_loads(page) == 1

        page.close()


def test_handoff_busy_expert_to_simple_waits_then_stops(tmp_path, chromium_browser):
    """ "Stop and switch now" must end a wait the operator no longer wants.

    The wait on a hooked terminal is unbounded by design, so this button is the
    only way out of it: it asks for the same session again, this time cutting
    the running turn short. It must leave the console usable on the key.
    """
    with _live_chat_server(tmp_path, ui_mode="expert") as (base_url, _app):
        page, key = _flip_into_a_busy_terminal(base_url, chromium_browser)

        page.locator(f"{_OP} .op-handoff-action").click()

        # The wait ends with the terminal's process gone and the console usable.
        expect(page.locator(f"{_OP} .op-input-area textarea")).to_be_enabled(timeout=25_000)
        expect(page.locator(f"{_OP} .op-handoff")).to_be_hidden()
        assert not _pty_is_live(base_url, key), "the terminal kept the session it handed over"
        assert [entry.get("session_id") for entry in _wait_for_chat_options(base_url)] == [key]

        page.close()


# ---------------------------------------------------------------------------
# 11. A second tab on the same session is refused, in words
# ---------------------------------------------------------------------------


def test_handoff_refused_while_another_tab_holds_the_session(tmp_path, chromium_browser):
    """A session already in use elsewhere is refused, and the console says so.

    Two tabs of one browser share the pointer, so both address the same key —
    and the key can only be in one place. The tab that does not hold it gets
    the refusal rather than a silent second conversation, and the copy names
    the situation the operator is actually in.
    """
    with _live_chat_server(tmp_path, ui_mode="expert") as (base_url, _app):
        context = chromium_browser.new_context()

        holder = _open_expert_page(context, base_url)
        key = _wait_for_pointer(holder)
        _wait_for_pty_spawns(base_url, 1)

        # A second tab, opened in the Simple view on the same pointer.
        second = _open_chat_page(context, base_url, "/?mode=simple")
        assert _pointer(second) == key

        _send(second, "are you there")

        overlay = second.locator(f"{_OP} .op-handoff")
        expect(overlay).to_be_visible(timeout=20_000)
        expect(overlay.locator(".op-handoff-message")).to_have_text(
            "This session is in use in another tab or view."
        )
        # Nothing this view can do about it, so it is offered nothing.
        expect(overlay.locator(".op-handoff-action")).to_be_hidden()
        # The prompt never ran, so the text is back where it was typed.
        expect(second.locator(f"{_OP} .op-input-area textarea")).to_have_value("are you there")

        second.close()
        holder.close()
        context.close()
