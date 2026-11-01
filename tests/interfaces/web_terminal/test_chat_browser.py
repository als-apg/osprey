"""Browser tests: chat operator chat, SSE → live DOM.

Proves the operator chat console behaviors that the FastAPI TestClient cannot
reach because none of them exist until a real browser runs the vendored
markdown/sanitiser globals and the SSE transport against a live event stream:

  1. a streamed prompt renders as sanitised markdown in the chat card;
  2. a second prompt in the same page-load reaches the same session (continuity);
  3. the activity line (the tool's operator phrase) shows during a tool_use and
     clears on text;
  4. Stop mid-stream re-enables the input and the next prompt runs cleanly;
  5. the Agent tile's ⋯ entry swaps the chat card ↔ xterm live, no reload;
  6. hostile model markdown renders inert in the live DOM (no script executes);
  7. a re-created chat with nothing to resume shows the "session reset" divider
     on the next turn — no divider when it resumes, none on a first turn;
  8. changing the surface hands ONE session between the two: the chat
     replays the terminal transcript and the terminal resumes it again,
     the pointer unchanged and the page never reloaded;
  9. a change with no transcript to resume starts the chat under the session key;
 10. the surface taking the session over shows the transitional state while the
     outgoing agent finishes: ended by "Stop and switch now" on either side,
     and by the turn's own idle edge on the chat side;
 11. a second tab holding the session gets the refusal in words;
 12. a panel citation in a streamed answer opens the KNOWLEDGE tile on the
     cited concept — the one companion panel this module serves for real, the
     okf panel over its own fixture bundle.

Harness: the panels-browser ``_live_server`` machinery (a real uvicorn server on
a background thread, with ``_load_web_config``/``_load_panel_config``/
``_launch_panel_server`` patched so no companion backends are needed), plus
the Claude Agent SDK client faked where the agent runner constructs it. The
fake ``ClaudeSDKClient`` replays a per-prompt *plan* of real SDK messages built
by the helpers the ``operator_session`` unit tests use, so the agent runner's
translation, the real ``_event_to_wire`` converter and the real
``routes/chat.py`` SSE branch run end to end — only the SDK subprocess is replaced. A handful of ``/__test__/*``
routes give each scenario a server-loop control channel (release a held turn,
force an eviction, read turn-guard state and what each SDK client was launched
with) without cross-thread event juggling.

The terminal half is real: a genuine PTY runs a long-lived child, and the command
the hand-off door built for it is recorded at the registry seam, so ``--resume
<transcript>`` versus ``--session-id <key>`` is asserted on the argv the spawn
actually used. What no fake can produce is the transcript a real Claude child
writes, so the conversation the terminal "had" is seeded as a JSONL file in
the discovery directory — the same file the resume, the replay and the idle
judgement all read. The project directory, the Claude config root and the
agent-data root are all under ``tmp_path``, so a run reads no developer state.

Run:
    uv run pytest tests/interfaces/web_terminal/test_chat_browser.py -q

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from urllib.parse import urlsplit

import pytest
import requests

from osprey.interfaces._serving import run_app_server
from tests.interfaces._browser import wait_for_dock_settled
from tests.interfaces.web_terminal._chat_server import (
    OBSERVED_PROMPTS,
    PLANS,
    live_chat_server,
    seed_record_from,
    seed_transcript,
)
from tests.interfaces.web_terminal._surface import arm_terminal_tab, back_to_chat, open_terminal

# ---------------------------------------------------------------------------
# Playwright availability guard
# ---------------------------------------------------------------------------

try:
    from playwright.sync_api import Page, expect
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False

pytestmark = [pytest.mark.browser, pytest.mark.slow]

_OP = "#operator-container"


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
# The load counter answers "did the page reload?". A surface change is a live swap of two
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
      // CSS side (the terminal overlay is display:none in the chat), so
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
    wait_for_dock_settled(page)
    return page


def _open_terminal_page(opener, base_url: str, query: str = "") -> Page:
    """Open a fresh tab already on the terminal and wait for it to connect."""
    page: Page = opener.new_page()
    arm_terminal_tab(page)
    page.add_init_script(_PROBE_INIT_SCRIPT)
    page.goto(f"{base_url}{query}", wait_until="domcontentloaded")
    expect(page.locator("#terminal-container .xterm")).to_be_visible(timeout=10_000)
    wait_for_dock_settled(page)
    return page


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
            "surface": "terminal",
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


def _wait_for_attached_session(page: Page, timeout: float = 20.0) -> str:
    """The key the terminal is attached to, once the server has confirmed it.

    A terminal page writes the pointer twice on boot. The console mints a key
    for a tab that has none, and the terminal then stores the id the server
    confirms on the socket -- a different one, and a third when a page-load
    resume's child exits early and the terminal falls back to a fresh session.
    Only the last of those is a key a PTY is running under, and the gap between
    the console's write and the terminal's is tens of milliseconds, well inside
    the window between the xterm becoming visible and the first read: a page
    that latched the first write names a session that never existed, and every
    server-side question asked about it -- is its PTY live, what was it spawned
    with -- answers no.

    The label carries the id the terminal attached to in `data-session-id`,
    set from the frame that stores it, so a pointer that agrees with it is a
    pointer the terminal is on.
    """
    deadline = time.monotonic() + timeout
    label = page.locator("#terminal-label")
    key: str | None = None
    shown = ""
    while time.monotonic() < deadline:
        # Bounded well inside the deadline above. The default read waits half a
        # minute for a label to attach, which on a page carrying no terminal
        # outlives this helper and answers with a locator timeout instead of
        # the message below, which names both halves of the condition.
        try:
            shown = (label.get_attribute("data-session-id", timeout=1_000) or "").strip()
        except PlaywrightTimeoutError:
            shown = ""
        key = _pointer(page)
        if key and shown == key:
            return key
        page.wait_for_timeout(50)
    raise AssertionError(
        f"the terminal never settled on a session: label {shown!r}, pointer {key!r}"
    )


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
    with live_chat_server(tmp_path) as (base_url, _app):
        PLANS["show me markup"] = [
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


def test_multi_turn_reaches_same_session(tmp_path, chromium_browser):
    """A second prompt sent the moment the input comes back reaches the same session.

    The input is re-enabled when the turn's terminal frame arrives, so that is
    the moment the operator may type again — and the server must already
    accept the next turn then, not a beat later. The second prompt is sent
    with no server-side barrier in between: a guard still held when the
    browser says the turn is over answers 409, and the operator reads "a turn
    is already running" for a turn that has finished.
    """
    with live_chat_server(tmp_path) as (base_url, _app):
        PLANS["first question"] = [("text", "first answer"), ("result",)]
        PLANS["second question"] = [("text", "second answer"), ("result",)]
        page = _open_chat_page(chromium_browser, base_url)

        _send(page, "first question")
        expect(page.locator(f"{_OP} .op-entry.assistant")).to_contain_text(
            "first answer", timeout=10_000
        )
        expect(page.locator(f"{_OP} .op-input-area textarea")).to_be_enabled()

        _send(page, "second question")
        expect(page.locator(f"{_OP} .op-entry.assistant").last).to_contain_text(
            "second answer", timeout=10_000
        )

        # Both exchanges are on screen, and nothing was refused on the way.
        expect(page.locator(f"{_OP} .op-entry.operator")).to_have_count(2)
        expect(page.locator(f"{_OP} .op-entry.assistant")).to_have_count(2)
        expect(page.locator(f"{_OP} .op-system")).to_have_count(0)
        # One chat session held both turns, and it saw both prompts in order.
        state = requests.get(f"{base_url}/__test__/chat-state").json()
        assert state["n"] == 1, state
        assert OBSERVED_PROMPTS == ["first question", "second question"]

        page.close()


# ---------------------------------------------------------------------------
# 3. Activity line during a tool_use, cleared on text
# ---------------------------------------------------------------------------


def test_activity_line_shows_tool_then_clears_on_text(tmp_path, chromium_browser):
    """Bash's phrase is visible while the turn is held, and clears once text lands."""
    with live_chat_server(tmp_path) as (base_url, _app):
        PLANS["run a tool"] = [
            ("tool_use", "Bash"),
            ("gate",),  # hold the turn so the activity line is observable
            ("text", "done running"),
            ("result",),
        ]
        page = _open_chat_page(chromium_browser, base_url)
        _send(page, "run a tool")

        activity = page.locator(f"{_OP} .op-loader")
        expect(activity).to_be_visible(timeout=10_000)
        # chat-activity.js maps tool names to operator phrases; Bash is a mapped name,
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
    with live_chat_server(tmp_path) as (base_url, _app):
        PLANS["long task"] = [("text", "starting…"), ("await_interrupt",), ("result",)]
        PLANS["quick follow up"] = [("text", "all clear"), ("result",)]
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
    """Send and Stop share one slot at the composer's right edge; a turn swaps one for the other.

    The button cluster is anchored to the row's right edge (``margin-left: auto``).
    ``applyControls`` hides Send and shows Stop in one statement block, and the
    reverse when the turn ends, so the operator's target must sit on the same
    pixels both ways. Asserted on real geometry, because the invariant is "the
    operator's target does not move" — a future layout change could keep the
    markup and still shift the slot.
    """
    with live_chat_server(tmp_path) as (base_url, _app):
        PLANS["long task"] = [("text", "starting…"), ("await_interrupt",), ("result",)]
        page = _open_chat_page(chromium_browser, base_url)

        send_btn = page.locator(f"{_OP} .op-send-btn")
        stop_btn = page.locator(f"{_OP} .op-stop-btn")
        expect(send_btn).to_be_visible()
        idle_x = send_btn.bounding_box()["x"]

        _send(page, "long task")
        expect(stop_btn).to_be_visible(timeout=10_000)
        expect(send_btn).to_be_hidden()
        streaming_x = stop_btn.bounding_box()["x"]

        stop_btn.click()
        expect(stop_btn).to_be_hidden(timeout=10_000)
        _wait_chat_idle(base_url)
        settled_x = send_btn.bounding_box()["x"]

        assert streaming_x == pytest.approx(idle_x, abs=1.0), (
            "Stop opened away from Send's slot — the two share the composer's right edge"
        )
        assert settled_x == pytest.approx(idle_x, abs=1.0), (
            "Send moved back when the turn ended, landing away from the slot Stop had "
            "occupied at the moment Send became clickable again"
        )

        page.close()


# ---------------------------------------------------------------------------
# 5. The surface change swaps chat card ↔ xterm live, both directions, no reload
# ---------------------------------------------------------------------------


def test_surface_change_swaps_chat_and_terminal_live(tmp_path, chromium_browser):
    """The ⋯ entry swaps console ↔ xterm via CSS; both stay in the DOM."""
    with live_chat_server(tmp_path) as (base_url, _app):
        page = _open_chat_page(chromium_browser, base_url)
        html = page.locator("html")
        console = page.locator(_OP)
        term = page.locator("#terminal-container")

        # Chat: console visible, xterm hidden.
        expect(html).not_to_have_attribute("data-agent-surface", "terminal")
        expect(console).to_be_visible()
        expect(term).to_be_hidden()

        # → Terminal: xterm visible, console hidden — no reload, both attached.
        open_terminal(page)
        expect(term).to_be_visible()
        expect(console).to_be_hidden()
        expect(console).to_be_attached()
        expect(term).to_be_attached()

        # → Chat again: the swap is reversible with no teardown.
        back_to_chat(page)
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
    with live_chat_server(tmp_path) as (base_url, _app):
        PLANS["be evil"] = [
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
    """No divider after a fresh first turn; exactly one after an unresumable eviction.

    What paints the divider is the conversation ending, not the process dying.
    A fresh page's first turn also opens with ``session_reset`` (the chat has
    no transcript to continue), and the operator's own prompt is not prior
    history, so nothing is drawn there. The evicted key has no transcript on
    disk either — the faked SDK writes none — so the re-created session starts
    a conversation of its own, and that ``session_reset``, arriving with prior
    turns on screen, is the one that paints. Its counterpart is the resume
    below, where the same eviction draws nothing.
    """
    with live_chat_server(tmp_path) as (base_url, _app):
        PLANS["turn one"] = [("text", "answer one"), ("result",)]
        PLANS["turn two"] = [("text", "answer two"), ("result",)]
        page = _open_chat_page(chromium_browser, base_url)

        _send(page, "turn one")
        expect(page.locator(f"{_OP} .op-entry.assistant")).to_contain_text(
            "answer one", timeout=10_000
        )
        expect(page.locator(f"{_OP} .op-input-area textarea")).to_be_enabled()

        # The first turn opened a conversation; it did not lose one.
        divider = page.locator(f"{_OP} .op-system").filter(has_text="session reset")
        expect(divider).to_have_count(0)

        _wait_chat_idle(base_url)
        resp = requests.post(f"{base_url}/__test__/evict-all")
        assert resp.json()["evicted"] >= 1

        _send(page, "turn two")
        # The eviction's session_reset paints a fresh divider (prior turns present).
        expect(divider).to_have_count(1, timeout=10_000)
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
    with live_chat_server(tmp_path) as (base_url, _app):
        PLANS["turn one"] = [("text", "answer one"), ("result",)]
        PLANS["turn two"] = [("text", "answer two"), ("result",)]
        page = _open_chat_page(chromium_browser, base_url)
        key = _wait_for_pointer(page)

        _send(page, "turn one")
        expect(page.locator(f"{_OP} .op-entry.assistant")).to_contain_text(
            "answer one", timeout=10_000
        )
        expect(page.locator(f"{_OP} .op-input-area textarea")).to_be_enabled()

        # The transcript that turn would have written, so the key has something
        # to be resumed through.
        seed_transcript(
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


# ---------------------------------------------------------------------------
# 8. A surface change hands ONE session over: chat replays it, terminal resumes it
# ---------------------------------------------------------------------------


def test_handoff_terminal_to_chat_and_back_keeps_one_session(
    tmp_path, chromium_browser, monkeypatch
):
    """Both surfaces are windows onto one session key and one conversation.

    The terminal starts a child under the key; going back to chat shows the
    transitional state, replays that key's transcript into the console and
    hands back a usable input, with the chat resuming the transcript rather
    than opening a second conversation; opening it again spawns a terminal child
    that resumes the same transcript. Through both changes the browser's pointer
    is the same string and the page never reloads — the surfaces are swapped
    live, which is the whole point of the mechanic.
    """
    with live_chat_server(tmp_path) as (base_url, _app):
        PLANS["and then"] = [("text", "the newer answer"), ("result",)]
        page = _open_terminal_page(chromium_browser, base_url)
        key = _wait_for_attached_session(page)

        # Nothing to resume yet, so the terminal child was started under the key.
        first = _wait_for_pty_spawns(base_url, 1)[0]
        assert _identity_of(first) == ("--session-id", key)

        # The conversation that view then had. Seeded, not generated: a real
        # transcript is written by a real Claude child (see seed_transcript).
        transcript = seed_transcript(
            tmp_path / "project",
            key,
            [("user", "what is the beam current"), ("assistant", "the earlier answer")],
        )
        # The chat replays the conversation from the agent record, so the
        # record is seeded too: it serves the nodes its own transcript parser and
        # timeline builder make of that file, as the ingested record would.
        seed_record_from(monkeypatch, transcript)

        back_to_chat(page)

        # The transcript is replayed into the console...
        expect(page.locator(f"{_OP} .op-entry.operator")).to_contain_text(
            "what is the beam current", timeout=20_000
        )
        expect(page.locator(f"{_OP} .op-entry.assistant")).to_contain_text("the earlier answer")
        # ...the input is usable again...
        expect(page.locator(f"{_OP} .op-input-area textarea")).to_be_enabled()
        # ...and the restart was on screen while it happened. The terminal was
        # idle, so what the operator watched was the agent moving, not a turn
        # finishing.
        assert any("Restarting the agent in this surface" in line for line in _handoff_log(page)), (
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

        open_terminal(page)

        expect(page.locator("#terminal-container")).to_be_visible()
        second = _wait_for_pty_spawns(base_url, 2)[1]
        assert _identity_of(second) == ("--resume", key)

        assert _pointer(page) == key, "the change moved the tab to another session"
        assert _page_loads(page) == 1, "the change reloaded the page"

        page.close()


# ---------------------------------------------------------------------------
# 9. A change with nothing to resume starts the chat under the session key
# ---------------------------------------------------------------------------


def test_handoff_before_any_prompt_starts_the_chat_under_the_key(tmp_path, chromium_browser):
    """With no transcript on disk the chat opens the key's first conversation.

    The counterpart of the resume above: the same change, made before anything
    has been said, must not ask the SDK to continue a transcript that does not
    exist. It starts one under the session key instead — so the id the child
    writes its transcript under is the key both surfaces are already on — and the
    pointer is untouched by the change.
    """
    with live_chat_server(tmp_path) as (base_url, _app):
        page = _open_terminal_page(chromium_browser, base_url)
        key = _wait_for_attached_session(page)
        first = _wait_for_pty_spawns(base_url, 1)[0]
        assert _identity_of(first) == ("--session-id", key)

        back_to_chat(page)

        expect(page.locator(f"{_OP} .op-input-area textarea")).to_be_enabled(timeout=20_000)
        options = _wait_for_chat_options(base_url)
        assert [entry.get("session_id") for entry in options] == [key]
        assert all(entry.get("resume") is None for entry in options)

        assert _pointer(page) == key
        assert _page_loads(page) == 1

        page.close()


# ---------------------------------------------------------------------------
# 10. The terminal side of a change: the wait, and the way out of it
# ---------------------------------------------------------------------------


def test_handoff_pending_shows_the_wait_and_stop_switches_now(tmp_path, chromium_browser):
    """Opening the terminal waits out the chat's turn, and one button ends that wait.

    Only one surface may run the session's agent, and the outgoing one is never
    cut off on a clock — so a change made mid-turn parks on the terminal card
    behind a live elapsed counter. "Stop and switch now" is the operator's way
    out: it asks for the same session again with the interrupt flag, which ends
    the running turn instead of waiting for it, and the terminal comes up on
    the key.
    """
    with live_chat_server(tmp_path) as (base_url, _app):
        # The turn never finishes on its own: the gate is deliberately never
        # released, so the wait the operator sees is real.
        PLANS["hold the line"] = [("text", "working"), ("gate",), ("result",)]
        page = _open_chat_page(chromium_browser, base_url)
        key = _wait_for_pointer(page)

        _send(page, "hold the line")
        expect(page.locator(f"{_OP} .op-entry.assistant")).to_contain_text(
            "working", timeout=10_000
        )

        open_terminal(page)

        overlay = page.locator("#terminal-handoff")
        expect(overlay).to_be_visible(timeout=10_000)
        expect(overlay.locator(".terminal-handoff-message")).to_contain_text(
            "Finishing in the other surface", timeout=10_000
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
    """Open the terminal, mark the key mid-turn, go back to chat, and return at the wait.

    Shared by the two cases below because the setup is the whole scenario up to
    the fork: what the operator does about the wait is what differs.
    """
    page = _open_terminal_page(chromium_browser, base_url)
    key = _wait_for_attached_session(page)
    _wait_for_pty_spawns(base_url, 1)
    assert _pty_is_live(base_url, key)

    # The terminal's own child reports a prompt going in — the hook's busy edge.
    _report_turn(base_url, key, "busy")

    back_to_chat(page)

    overlay = page.locator(f"{_OP} .op-handoff")
    expect(overlay).to_be_visible(timeout=10_000)
    # The console has no word from the server on whether the terminal is
    # mid-turn, so it shows the restart first and the wait once the restart
    # budget is spent — the copy below arrives a few seconds in.
    expect(overlay.locator(".op-handoff-message")).to_contain_text(
        "Finishing in the other surface", timeout=15_000
    )
    expect(overlay.locator(".op-handoff-elapsed")).to_have_text(re.compile(r"^\d+:\d{2}$"))
    # The input stays out of reach: the session is not this surface's yet.
    expect(page.locator(f"{_OP} .op-input-area textarea")).to_be_disabled()
    expect(overlay.locator(".op-handoff-action")).to_have_text("Stop and switch now")
    return page, key


def test_handoff_busy_terminal_to_chat_waits_for_the_turn(tmp_path, chromium_browser):
    """A change made while the terminal is mid-turn waits, then completes on the idle edge.

    The mirror of the case above, on the surface whose turns are reported by a
    hook rather than read off an SDK client. The outgoing agent is never ended
    on a clock, so the console parks behind the transitional state rather than
    taking the session away mid-answer — and the edge that ends the turn is
    what releases it. The terminal's process is torn down and the console comes
    up on the same key.
    """
    with live_chat_server(tmp_path) as (base_url, _app):
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


def test_handoff_busy_terminal_to_chat_waits_then_stops(tmp_path, chromium_browser):
    """ "Stop and switch now" must end a wait the operator no longer wants.

    The wait on a hooked terminal is unbounded by design, so this button is the
    only way out of it: it asks for the same session again, this time cutting
    the running turn short. It must leave the console usable on the key.
    """
    with live_chat_server(tmp_path) as (base_url, _app):
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
    with live_chat_server(tmp_path) as (base_url, _app):
        context = chromium_browser.new_context()

        holder = _open_terminal_page(context, base_url)
        _wait_for_pty_spawns(base_url, 1)
        # The key the second tab will adopt, which is the attached one rather
        # than whatever the console minted on its way to it.
        key = _wait_for_attached_session(holder)

        # A second tab, opened in the chat on the same pointer.
        second = _open_chat_page(context, base_url)
        assert _pointer(second) == key

        _send(second, "are you there")

        overlay = second.locator(f"{_OP} .op-handoff")
        expect(overlay).to_be_visible(timeout=20_000)
        expect(overlay.locator(".op-handoff-message")).to_have_text(
            "This session is in use in another tab."
        )
        # Nothing this surface can do about it, so it is offered nothing.
        expect(overlay.locator(".op-handoff-action")).to_be_hidden()
        # The prompt never ran, so the text is back where it was typed.
        expect(second.locator(f"{_OP} .op-input-area textarea")).to_have_value("are you there")

        second.close()
        holder.close()
        context.close()


# ---------------------------------------------------------------------------
# 12. A panel citation in an answer opens the KNOWLEDGE tile on that concept
# ---------------------------------------------------------------------------

#: The knowledge bundle the okf panel's own suite serves; ``devices/bpm`` is in it.
_OKF_BUNDLE = Path(__file__).resolve().parents[1] / "okf_panel" / "fixtures" / "bundle"


def test_panel_citation_opens_the_knowledge_tile_on_the_concept(tmp_path, chromium_browser):
    """A citation in a streamed answer is a click that opens KNOWLEDGE on that concept.

    The whole path runs for real: the fake SDK streams the markdown, the browser
    renders it through the vendored marked and DOMPurify, the hub classifies the
    click and navigates the okf panel's iframe to the linked URL, fragment
    included, and the panel — served here from its fixture bundle — follows the
    fragment to the concept. The chat boots chat-only on an empty
    workspace, so the click is also what reveals the dock.
    """
    from osprey.interfaces.okf_panel.app import create_app as create_okf_app

    with (
        run_app_server(create_okf_app(str(_OKF_BUNDLE))) as okf_url,
        live_chat_server(
            tmp_path, enabled_panels=frozenset({"artifacts", "okf"}), panel_urls={"okf": okf_url}
        ) as (base_url, _app),
    ):
        PLANS["where is the bpm documented"] = [
            ("text", "The monitor is described under [BPM](panel/okf#devices/bpm)."),
            ("result",),
        ]
        page = _open_chat_page(chromium_browser, base_url)
        _send(page, "where is the bpm documented")

        citation = page.locator(f"{_OP} .op-entry.assistant .osprey-md-rendered a")
        expect(citation).to_have_attribute("href", "panel/okf#devices/bpm", timeout=10_000)
        citation.click()

        tile = page.locator('.dock-iframe-overlay iframe[data-panel-id="okf"]')
        expect(tile).to_be_visible(timeout=10_000)
        # The tile was navigated to the linked URL, fragment included (the hub
        # adds its own embed query on the way).
        src = urlsplit(tile.get_attribute("src") or "")
        assert (src.path, src.fragment) == ("/panel/okf", "devices/bpm"), src
        reader = page.frame_locator('.dock-iframe-overlay iframe[data-panel-id="okf"]')
        expect(reader.locator("#reader-content h1.concept-title")).to_have_text(
            "Beam Position Monitor", timeout=10_000
        )
        # The hub itself never navigated away.
        assert page.url.startswith(base_url), page.url
        page.close()


# ---------------------------------------------------------------------------
# Thinking: each turn that reasoned shows one collapsed summary
# ---------------------------------------------------------------------------


def _expect_collapsed_thinking(entry, reasoning: str) -> None:
    """*entry* carries one finished, collapsed thinking row that opens on *reasoning*."""
    row = entry.locator(".op-thinking")
    expect(row).to_have_count(1, timeout=10_000)
    expect(row).to_have_attribute("data-state", "done")
    head = row.locator(".op-row-head")
    expect(head).to_have_attribute("aria-expanded", "false")
    expect(row.locator(".op-row-body")).to_be_hidden()
    head.click()
    expect(row.locator(".op-row-body")).to_be_visible()
    expect(row.locator(".op-row-body")).to_have_text(reasoning)


def test_live_turn_shows_a_collapsed_thinking_summary(tmp_path, chromium_browser):
    """A turn that reasoned ends with its reasoning folded into one row above the answer."""
    with live_chat_server(tmp_path) as (base_url, _app):
        PLANS["think first"] = [
            ("thinking", "the orbit moved because BPM 3 drifted"),
            ("text", "BPM 3 drifted"),
            ("result",),
        ]
        page = _open_chat_page(chromium_browser, base_url)
        _send(page, "think first")

        entry = page.locator(f"{_OP} .op-entry.assistant")
        expect(entry.locator(".osprey-md-rendered")).to_contain_text(
            "BPM 3 drifted", timeout=10_000
        )
        _expect_collapsed_thinking(entry, "the orbit moved because BPM 3 drifted")

        page.close()


def test_replayed_turns_show_their_thinking_collapsed(tmp_path, chromium_browser, monkeypatch):
    """The chat opened on a session replays each turn's thinking from the record, folded.

    The record serves the turns the session's transcript holds. The first agent
    turn reasoned and shows its summary; the second's block was ``omitted``
    (written with no text) and shows a row that does not open. No tool input or
    result is drawn for a turn that made no call.
    """
    with live_chat_server(tmp_path) as (base_url, _app):
        page = _open_chat_page(chromium_browser, base_url)
        key = _wait_for_pointer(page)

        transcript = seed_transcript(
            tmp_path / "project",
            key,
            [
                ("user", "why did the orbit move"),
                (
                    "assistant",
                    [
                        {"type": "thinking", "thinking": "compare BPM 3 with its neighbours"},
                        {"type": "text", "text": "BPM 3 alone drifted"},
                    ],
                ),
                ("user", "and now"),
                (
                    "assistant",
                    [
                        {"type": "thinking", "thinking": ""},
                        {"type": "text", "text": "it is back in place"},
                    ],
                ),
            ],
        )
        seed_record_from(monkeypatch, transcript)

        # A reload binds the console to the same key and replays its conversation.
        page.reload()
        entries = page.locator(f"{_OP} .op-entry.assistant")
        expect(entries).to_have_count(2, timeout=20_000)
        expect(entries.nth(0)).to_contain_text("BPM 3 alone drifted")
        expect(entries.nth(1)).to_contain_text("it is back in place")

        _expect_collapsed_thinking(entries.nth(0), "compare BPM 3 with its neighbours")
        hidden = entries.nth(1).locator(".op-thinking")
        expect(hidden).to_have_count(1)
        expect(hidden).to_have_class(re.compile(r"\bempty\b"))
        expect(hidden.locator(".op-row-head")).to_be_disabled()
        expect(page.locator(f"{_OP} .op-tools")).to_have_count(0)

        page.close()
