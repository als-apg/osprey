"""Browser tests: Simple-mode operator chat, SSE → live DOM.

Proves the operator chat console behaviors that the FastAPI TestClient cannot
reach because none of them exist until a real browser runs the vendored
markdown/sanitiser globals and the SSE transport against a live event stream:

  1. a streamed prompt renders as sanitised markdown in the chat card;
  2. a second prompt in the same page-load reaches the same session (continuity);
  3. the activity line ("Using <tool>…") shows during a tool_use and clears on text;
  4. Stop mid-stream re-enables the input and the next prompt runs cleanly;
  5. the Expert/Simple toggle swaps the chat card ↔ xterm live, no reload;
  6. hostile model markdown renders inert in the live DOM (no script executes);
  7. a forced server-side eviction shows the "session reset" divider on the next
     turn — and NO divider on a fresh page's first turn.

Harness: the panels-browser ``_live_server`` machinery (a real uvicorn server on
a background thread, with ``_load_web_config``/``_load_panel_config``/
``_launch_artifact_server`` patched so no companion backends are needed), plus
the Claude Agent SDK faked at the ``operator_session`` seam. The fake
``ClaudeSDKClient`` replays a per-prompt *plan* of the same ``Fake*`` SDK message
objects the ``operator_session`` unit tests use, so the real
``_message_to_events`` converter and the real ``routes/chat.py`` SSE branch run
end to end — only the SDK subprocess is replaced. A handful of ``/__test__/*``
routes give each scenario a server-loop control channel (release a held turn,
force an eviction, read turn-guard state) without cross-thread event juggling.

Run:
    uv run pytest tests/interfaces/web_terminal/test_chat_browser.py -q

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import asyncio
import re
import time
from contextlib import contextmanager
from unittest.mock import patch

import pytest
import requests
from fastapi import Request

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


def _reset_fake_state() -> None:
    """Clear per-server fake state; called at each server launch (test thread)."""
    global _RELEASE_GATE
    _PLANS.clear()
    _OBSERVED_PROMPTS.clear()
    _RELEASE_GATE = asyncio.Event()


class _FakeSDKClient:
    """Stand-in for ``ClaudeSDKClient`` wired at the operator_session seam.

    ``receive_response`` replays the plan registered for the most recent prompt,
    yielding the ``Fake*`` SDK message objects the real ``_message_to_events``
    converts into chat events. Step vocabulary:

      ("text", md)      one text block (markdown) → a ``text`` event
      ("thinking",)     one thinking block        → drives the activity line
      ("tool_use", nm)  one tool_use block        → activity line "Using <nm>…"
      ("system", sub)   a system message
      ("gate",)         block until POST /__test__/release
      ("hang",)         block until the reader is cancelled (Stop tests)
      ("await_interrupt") block until interrupt() (Stop → clean terminal)
      ("result",)       a terminal ResultMessage
    """

    def __init__(self, *args, **kwargs) -> None:
        self._prompts: list[str] = []
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
        return {"n": len(chats), "in_flight": any(s.in_flight for s in chats)}

    app.add_api_route("/__test__/release", _release, methods=["POST"])
    app.add_api_route("/__test__/evict-all", _evict_all, methods=["POST"])
    app.add_api_route("/__test__/chat-state", _chat_state, methods=["GET"])


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

    Yields:
        (base_url, app) — live server address and the FastAPI app.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir(exist_ok=True)
    _reset_fake_state()

    patches = [
        patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace)},
        ),
        patch(
            "osprey.interfaces.web_terminal.app._load_panel_config",
            return_value=({"artifacts"}, [], None),
        ),
        patch(
            "osprey.interfaces.web_terminal.app._launch_artifact_server",
            side_effect=lambda a: setattr(a.state, "artifact_server_url", None),
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

        app = create_app(shell_command=["echo", "hello"])
        _install_test_routes(app)
        with _run_app_server(app) as base_url:
            app.state.web_ui_mode = ui_mode
            yield base_url, app


# ---------------------------------------------------------------------------
# Page + interaction helpers
# ---------------------------------------------------------------------------


def _open_chat_page(browser, base_url: str, query: str = "") -> Page:
    """Open a fresh page, wait for the console, and drop the welcome overlay."""
    page = browser.new_page()
    page.goto(f"{base_url}{query}", wait_until="domcontentloaded")
    # initChat builds the console on DOMContentLoaded; the input row is the
    # last thing appended, so its presence means the console is mounted.
    expect(page.locator(f"{_OP} .op-input-area textarea")).to_be_visible(timeout=10_000)
    # The first-visit overlay intercepts pointer events; remove it so the
    # textarea and header controls are genuinely interactable.
    page.evaluate("document.getElementById('welcome-overlay')?.remove()")
    return page


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
    """ "Using Bash…" is visible while the turn is held, and clears once text lands."""
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
        expect(activity).to_contain_text("Using Bash")

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
        page.locator('#mode-toggle .mode-segment[data-mode="expert"]').click()
        expect(html).to_have_attribute("data-ui-mode", "expert")
        expect(term).to_be_visible()
        expect(console).to_be_hidden()
        expect(console).to_be_attached()
        expect(term).to_be_attached()

        # → Simple again: the swap is reversible with no teardown.
        page.locator('#mode-toggle .mode-segment[data-mode="simple"]').click()
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
# 7. Session-expiry divider: shown after a forced eviction, absent on turn 1
# ---------------------------------------------------------------------------


def test_session_expiry_divider_after_eviction(tmp_path, chromium_browser):
    """A server-side eviction makes the next turn show the "session reset" divider.

    The recreated session re-emits ``session_reset`` while prior turns are on
    screen, so the renderer marks the boundary with a centred divider.
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

        # Force a server-side eviction: the next turn creates a brand-new session
        # (was_reused False), which re-emits session_reset.
        _wait_chat_idle(base_url)
        resp = requests.post(f"{base_url}/__test__/evict-all")
        assert resp.json()["evicted"] >= 1

        _send(page, "turn two")
        # The eviction's session_reset paints a fresh divider (prior turns present).
        expect(divider).to_have_count(before + 1, timeout=10_000)
        expect(page.locator(f"{_OP} .op-entry.assistant").last).to_contain_text("answer two")

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
