"""Browser e2e: the session tree on the hermetic hub, seeded through the real writers.

Each test boots :func:`~docs.screenshots.contact_sheet.hermetic_hub` with its
e2e seeds: the real agent record over the seeded transcripts, the scripted chat
child and the stubbed naming call. Runs are written by the executor's own
response builder and refusal head and by the approval hook's own preview and
settle functions; uploads go through the composer; a terminal ``/clear`` is the
hook's ``POST /api/agent-turn``. Pins: one Runs row per run with its parts and
its status arriving live; the ``/clear`` split, entering an older transcript and
a live terminal ``/clear``; the rows the tree leaves out; an upload
after ``/clear``; resume across reload, a fresh context and sign-out; the
streaming guard and browsing; naming, rename and revert, the switcher and More
sessions; the ``beam_profile`` stub viewer and its fallback; the palette and
the delete dialog.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest import mock

import httpx
import pytest
from docs.screenshots.contact_sheet import (
    BEAM_PROFILE_KIND,
    BEAM_PROFILE_MARKER_CLASS,
    SEEDED_SESSIONS,
    _append_transcript,
    _clear_lines,
    _exchange,
    hermetic_hub,
)

pytestmark = [pytest.mark.browser, pytest.mark.slow]

try:
    from playwright.sync_api import Page, expect

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False

if not _PLAYWRIGHT_AVAILABLE:  # pragma: no cover
    pytest.skip("playwright not installed", allow_module_level=True)

_RF_FIRST, _RF_AFTER_CLEAR, _PROFILE = SEEDED_SESSIONS
_TIMEOUT_MS = 20_000
_TREE = "#sidebar-session-tree"
_ARTIFACTS_FRAME = 'iframe.panel-iframe[data-panel-id="artifacts"]'
_DIALOG = ".posture-modal-overlay:not([data-closing])"
#: The docked Artifacts tile's close button: the hub's one tile besides the Agent's.
_ARTIFACTS_TILE_CLOSE = '.tile-tab-close[aria-label="Close tile"]'

_HOOK = (
    Path(__file__).resolve().parents[3]
    / "src/osprey/templates/claude_code/claude/hooks/osprey_approval.py"
)

#: Opens every closed node of the tree (sessions, form groups, runs),
#: one per poll, and answers true once ten polls in a row found none closed:
#: an opened session lists its groups only once its listing has loaded.
_OPEN_ALL_JS = """() => {
  const tree = document.querySelector('#sidebar-session-tree');
  const closed = tree.querySelector('[data-group-head][aria-expanded="false"]')
    ?? tree.querySelector('.osprey-tree-run[aria-expanded="false"] [data-tree-disc]')
    ?? tree.querySelector('.osprey-tree-session[aria-expanded="false"] [data-tree-disc]');
  if (closed) {
    window.__e2eQuietPolls = 0;
    closed.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    return false;
  }
  window.__e2eQuietPolls = (window.__e2eQuietPolls ?? 0) + 1;
  return window.__e2eQuietPolls >= 10;
}"""


# --- fixtures -----------------------------------------------------------------------


@pytest.fixture
def hub():
    with hermetic_hub(e2e=True) as running:
        yield running


@pytest.fixture
def client(hub):
    from osprey.interfaces.common_middleware import OPERATOR_SECRET_HEADER
    from osprey.interfaces.web_auth import get_web_credentials

    with httpx.Client(
        base_url=hub.base_url,
        timeout=30.0,
        headers={OPERATOR_SECRET_HEADER: get_web_credentials().operator_secret},
    ) as c:
        yield c


#: These tests read the tree in Kind mode, which an operator chooses (Activity is
#: the default); a mode already stored, e.g. by a test's own switch, is kept.
_KIND_MODE_JS = """(() => {
  const key = 'osprey-sidebar-artifacts-view';
  if (localStorage.getItem(key) === null) localStorage.setItem(key, 'kind');
})()"""


@pytest.fixture
def context(chromium_browser):
    ctx = chromium_browser.new_context(viewport={"width": 1400, "height": 900})
    ctx.add_init_script(_KIND_MODE_JS)
    yield ctx
    ctx.close()


@pytest.fixture
def page(context):
    return context.new_page()


# --- page helpers -------------------------------------------------------------------


def _boot(page: Page, hub) -> None:
    page.goto(hub.base_url, wait_until="domcontentloaded")
    expect(page.locator(f"{_TREE} .osprey-tree-session").first).to_be_visible(timeout=_TIMEOUT_MS)


def _session(page: Page, sid: str):
    return page.locator(f'{_TREE} .osprey-tree-session[data-session-id="{sid}"]')


def _active(page: Page):
    return page.locator(f"{_TREE} .osprey-tree-session.active")


def _row(page: Page, artifact_id: str):
    return page.locator(f'{_TREE} .osprey-list-row[data-id="{artifact_id}"]')


def _open_all(page: Page) -> None:
    page.evaluate("() => { window.__e2eQuietPolls = 0; }")
    page.wait_for_function(_OPEN_ALL_JS, timeout=_TIMEOUT_MS, polling=100)


def _open_session(page: Page, sid: str) -> None:
    """Show a session's artifacts: only the session in use opens, so enter it first."""
    row = _session(page, sid)
    if "active" not in (row.get_attribute("class") or "").split():
        row.locator(".osprey-tree-title").click()
        expect(row).to_have_class(_re(r"\bactive\b"), timeout=_TIMEOUT_MS)
    _click_until_open(row, row.locator("[data-tree-disc]"))


def _open_group(page: Page, sid: str, form: str) -> None:
    head = page.locator(f'{_TREE} [data-group-head="{sid}:{form}"]')
    expect(head).to_be_visible(timeout=_TIMEOUT_MS)
    _click_until_open(head, head)


def _click_until_open(node, toggle) -> None:
    """Open *node* by clicking *toggle* until it stays open. The page opens it too,
    unawaited (the reveal of the session's last artifact on load and on entering), and a
    click toggles: a click that lands just after that reveal closes it again."""
    for _ in range(10):
        if node.get_attribute("aria-expanded") != "true":
            toggle.click()
        with suppress(AssertionError):
            expect(node).to_have_attribute("aria-expanded", "true", timeout=1000)
            return
    expect(node).to_have_attribute("aria-expanded", "true")


def _group_count(page: Page, sid: str, form: str):
    return page.locator(f'{_TREE} [data-group-head="{sid}:{form}"] .osprey-tree-n')


def _menu(page: Page, row, item: str) -> None:
    row.click(button="right")
    page.get_by_role("menuitem", name=item).click()


def _pointer(page: Page) -> str:
    return page.evaluate("() => localStorage.getItem('osprey-pty-session')")


def _settle_pointer(page: Page) -> None:
    """Outwait the page's debounced last-active write (300 ms)."""
    page.wait_for_timeout(700)


def _send(page: Page, prompt: str) -> None:
    """Send *prompt* from the composer; a composer still binding keeps the text, so
    Enter is pressed until the composer takes it."""
    box = page.locator('#operator-container textarea[aria-label="Message the agent"]')
    box.fill(prompt)
    for _ in range(20):
        box.press("Enter")
        with suppress(AssertionError):
            expect(box).to_have_value("", timeout=500)
            return
    raise AssertionError(f"the composer never sent {prompt!r}")


# --- writers ------------------------------------------------------------------------


def _store(hub):
    from osprey.stores.artifact_store import ArtifactStore

    return ArtifactStore(workspace_root=hub.project_dir, auto_launch=False)


@contextmanager
def _writers(hub, key: str) -> Iterator[object]:
    """The real writers aimed at the hub's store, tagging *key*'s session.

    Yields the approval hook loaded as a module, the way the harness runs it.
    """
    from osprey.stores import artifact_store

    spec = importlib.util.spec_from_file_location("osprey_approval_e2e", _HOOK)
    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)
    artifact_store.reset_artifact_store()
    with (
        mock.patch(
            "osprey_connectors.workspace.resolve_shared_data_root", return_value=hub.project_dir
        ),
        mock.patch.dict("os.environ", {"OSPREY_SESSION_ID": key}),
    ):
        try:
            yield hook
        finally:
            artifact_store.reset_artifact_store()


def _ask(hook, call_id: str) -> str:
    """The approval preview an ``execute`` ask saves; its artifact id."""
    _url, artifact_id = hook._create_pre_execution_notebook(
        "x = 1\nprint(x)\n", "readonly", {}, tool_use_id=call_id
    )
    assert artifact_id, "the hook saved no preview"
    return artifact_id


def _deny(hook, preview_id: str) -> None:
    hook._settle_review({hook.REVIEW_ARTIFACT_KEY: preview_id}, "not_run")


def _execute(call_id: str, *, success: bool = True, tool_source: str = "execute") -> None:
    """The executor's response builder for one call that ran."""
    from fastmcp.exceptions import ToolError

    from osprey.audit.call import call_scope
    from osprey.mcp_server.python_executor.executor import ExecutionResult
    from osprey.mcp_server.python_executor.tools._response_builder import (
        build_execution_response,
    )

    result = ExecutionResult(
        success=success,
        stdout="ok\n" if success else "",
        stderr="" if success else "Traceback: boom\n",
        execution_method_used="subprocess",
        execution_time_seconds=0.1,
        error_message=None if success else "boom",
    )
    patterns = {"has_writes": False, "has_reads": False, "detected_patterns": {}}

    async def _respond() -> None:
        with call_scope(call_id, None), suppress(ToolError):
            await build_execution_response(
                "x = 1\nprint(x)\n", "probe", "readonly", result, patterns, True, tool_source
            )

    # Playwright's sync API holds this thread's event loop; the builder gets its own.
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(asyncio.run, _respond()).result()


def _refuse(call_id: str) -> None:
    """The executor's head for a call refused before launch."""
    from osprey.audit.call import call_scope
    from osprey.mcp_server.python_executor.tools._response_builder import save_refusal_head

    with call_scope(call_id, None):
        save_refusal_head(
            tool_source="execute",
            code="caput('SR:X', 1)\n",
            description="refused",
            execution_mode="readwrite",
            error="writes are off",
            launched=False,
        )


def _run_row(page: Page, call_id: str):
    return page.locator(f'{_TREE} .osprey-tree-run[data-run="{call_id}"]')


# --- runs ---------------------------------------------------------------------------


def test_every_run_is_one_runs_row_and_an_awaited_one_settles_live(hub, page):
    sid = key = _PROFILE.session_key
    calls = {
        name: f"toolu_e2e_{name}"
        for name in ("approved", "failed", "denied", "refused", "plain", "file", "waiting")
    }
    with _writers(hub, key) as hook:
        _ask(hook, calls["approved"])
        _execute(calls["approved"])
        _execute(calls["failed"], success=False)
        _deny(hook, _ask(hook, calls["denied"]))
        _refuse(calls["refused"])
        _execute(calls["plain"])
        _execute(calls["file"], tool_source="execute_file")
        _ask(hook, calls["waiting"])

    _boot(page, hub)
    _open_session(page, sid)
    _open_group(page, sid, "runs")
    expect(_group_count(page, sid, "runs")).to_have_text("7")
    expected = {
        "approved": "succeeded",
        "failed": "failed",
        "denied": "not_run",
        "refused": "not_run",
        "plain": "succeeded",
        "file": "succeeded",
        "waiting": "running",
    }
    for name, status in expected.items():
        row = _run_row(page, calls[name])
        expect(row).to_have_count(1)
        expect(row).to_have_attribute("data-run-status", status)
    expect(_run_row(page, calls["waiting"]).locator(".osprey-dot")).to_have_count(0)
    expect(_run_row(page, calls["denied"]).locator(".osprey-tree-meta")).to_have_text("not run")
    expect(_run_row(page, calls["approved"]).locator(".osprey-dot.ok")).to_have_count(1)
    expect(_run_row(page, calls["failed"]).locator(".osprey-dot.err")).to_have_count(1)

    # The approved run's parts sit under its one row.
    _run_row(page, calls["approved"]).locator("[data-tree-disc]").click()
    parts = page.locator(f"{_TREE} .osprey-tree-part")
    expect(parts.filter(has_text="Approval preview")).to_have_count(1)
    expect(parts.filter(has_text="Code")).to_have_count(1)

    # The operator approves the waiting call: its status arrives without a reload.
    with _writers(hub, key):
        _execute(calls["waiting"])
    expect(_run_row(page, calls["waiting"])).to_have_attribute(
        "data-run-status", "succeeded", timeout=_TIMEOUT_MS
    )
    expect(_group_count(page, sid, "runs")).to_have_text("7")


# --- /clear -------------------------------------------------------------------------


def test_entering_an_older_transcript_keeps_its_save_and_a_terminal_clear_shows_live(
    hub, client, page
):
    _boot(page, hub)
    # The RF key was /clear-ed: its older transcript is a row of its own.
    expect(_session(page, _RF_FIRST.transcript_id)).to_have_count(1)
    expect(_session(page, _RF_AFTER_CLEAR.transcript_id)).to_have_count(1)
    expect(_session(page, _PROFILE.transcript_id)).to_have_count(1)

    _session(page, _RF_FIRST.transcript_id).locator(".osprey-tree-title").click()
    expect(_session(page, _RF_FIRST.transcript_id)).to_have_class(
        _re(r"\bactive\b"), timeout=_TIMEOUT_MS
    )
    entered = _pointer(page)
    assert entered and entered != _RF_FIRST.session_key, "entering must mint a new key"

    _send(page, "Plot the cavity temperatures.")
    expect(_session(page, _RF_FIRST.transcript_id).locator(".osprey-tree-n")).to_have_text(
        "1", timeout=_TIMEOUT_MS
    )
    (turn,) = [t for t in hub.chat.turns if t.prompt == "Plot the cavity temperatures."]
    assert turn.session_key == entered
    assert turn.transcript_id == _RF_FIRST.transcript_id

    # A /clear in the terminal and a first prompt there: the turn hook's report
    # names the new transcript, and its row opens live as the Agent's session.
    new_tid = str(uuid.uuid4())
    now = datetime.now(UTC)
    _append_transcript(
        hub.session_dir,
        new_tid,
        _clear_lines(new_tid, now) + _exchange(new_tid, "Read the RF power.", "Done", now),
    )
    response = client.post(
        "/api/agent-turn",
        json={
            "session_id": new_tid,
            "pool_key": entered,
            "state": "idle",
            "surface": "terminal",
        },
    )
    assert response.status_code == 200, response.text
    expect(_session(page, new_tid)).to_have_count(1, timeout=_TIMEOUT_MS)
    expect(_active(page)).to_have_attribute("data-session-id", new_tid, timeout=_TIMEOUT_MS)

    _settle_pointer(page)
    page.reload(wait_until="domcontentloaded")
    expect(_active(page)).to_have_attribute("data-session-id", new_tid, timeout=_TIMEOUT_MS)


# --- what the tree holds ------------------------------------------------------------


def test_unowned_dispatch_and_watchdog_artifacts_are_absent(hub, page):
    store = _store(hub)
    loose = store.save_file(b"# loose\n", "loose.md", "markdown", title="Loose note", session_id="")
    dispatched = store.save_file(
        b"# dispatched\n",
        "dispatched.md",
        "markdown",
        title="Dispatched report",
        run_id="dispatch-run-1",
        session_id=_PROFILE.session_key,
    )
    watched = store.save_file(
        b"# watchdog\n",
        "watchdog.md",
        "markdown",
        title="Watchdog report",
        origin="watchdog",
        session_id=_PROFILE.session_key,
    )

    _boot(page, hub)
    _open_all(page)
    # Every row the tree shows is an agent session's: no bucket for the rest.
    expect(page.locator(f"{_TREE} [data-tree-earlier]")).to_have_count(0)
    expect(_row(page, loose.id)).to_have_count(0)
    expect(_row(page, dispatched.id)).to_have_count(0)
    expect(_row(page, watched.id)).to_have_count(0)
    expect(page.locator(_TREE)).not_to_contain_text("Dispatched report")
    expect(page.locator(_TREE)).not_to_contain_text("Watchdog report")


def test_an_image_uploaded_after_clear_lands_under_the_new_session(hub, page, tmp_path):
    _boot(page, hub)
    # A prompt first, so the pre-/clear transcript is a session row of its own.
    _send(page, "Check the beam current.")
    _wait_for(lambda: any(t.prompt == "Check the beam current." for t in hub.chat.turns))
    (first,) = [t for t in hub.chat.turns if t.prompt == "Check the beam current."]
    before = first.transcript_id
    _send(page, "/clear")
    _wait_for(lambda: any(t.prompt.startswith("/clear") for t in hub.chat.turns))
    (cleared,) = [t for t in hub.chat.turns if t.prompt.startswith("/clear")]

    png = tmp_path / "screen.png"
    png.write_bytes(_PNG)
    # A file picked while the /clear turn still runs is refused (the composer is
    # disabled), so attach only once the composer takes input again, and see the chip.
    box = page.locator('#operator-container textarea[aria-label="Message the agent"]')
    expect(box).to_be_enabled(timeout=_TIMEOUT_MS)
    page.locator("#operator-container input[type=file]").set_input_files(str(png))
    expect(page.locator("#operator-container .op-attach-chip")).to_have_count(
        1, timeout=_TIMEOUT_MS
    )
    _send(page, "What does this screen show?")
    _wait_for(lambda: any(t.uploads for t in hub.chat.turns))

    sid = cleared.transcript_id
    expect(_session(page, sid)).to_have_count(1, timeout=_TIMEOUT_MS)
    _open_session(page, sid)
    _open_group(page, sid, "images")
    image = page.locator(f'{_TREE} [data-session-box="{sid}"] .osprey-list-row', has_text="screen")
    expect(image).to_have_count(1, timeout=_TIMEOUT_MS)
    image.click()
    frame = page.frame_locator(_ARTIFACTS_FRAME)
    expect(frame.locator("#preview-content img").first).to_be_visible(timeout=_TIMEOUT_MS)

    # The image is tagged to the new session only: the pre-/clear one lists none.
    assert before and before != sid
    expect(_session(page, before)).to_have_count(1, timeout=_TIMEOUT_MS)
    _open_session(page, before)
    _open_all(page)
    old = page.locator(f'{_TREE} [data-session-box="{before}"] .osprey-list-row', has_text="screen")
    expect(old).to_have_count(0)


def _wait_for(predicate, timeout_s: float = 20.0) -> None:
    import time

    deadline = time.monotonic() + timeout_s
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.1)


#: A 1x1 PNG.
_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082"
)


# --- resume -------------------------------------------------------------------------


def test_a_session_name_click_reveals_its_last_artifact_and_every_return_resumes_it(
    hub, page, chromium_browser
):
    profile = next(e for e in _store(hub).list_entries() if e.kind == BEAM_PROFILE_KIND)
    sid = _PROFILE.transcript_id
    _boot(page, hub)
    _open_session(page, sid)
    _open_all(page)
    _row(page, profile.id).click()
    expect(_row(page, profile.id)).to_have_attribute("aria-selected", "true")

    # Away to another session, then back by its name: the last-open artifact shows.
    _session(page, _RF_AFTER_CLEAR.transcript_id).locator(".osprey-tree-title").click()
    expect(_active(page)).to_have_attribute(
        "data-session-id", _RF_AFTER_CLEAR.transcript_id, timeout=_TIMEOUT_MS
    )
    _session(page, sid).locator(".osprey-tree-title").click()
    expect(_active(page)).to_have_attribute("data-session-id", sid, timeout=_TIMEOUT_MS)
    expect(_row(page, profile.id)).to_have_attribute("aria-selected", "true", timeout=_TIMEOUT_MS)

    # Reload: the same session, the artifact shown in the docked panel.
    _settle_pointer(page)
    page.reload(wait_until="domcontentloaded")
    expect(_active(page)).to_have_attribute("data-session-id", sid, timeout=_TIMEOUT_MS)
    expect(_row(page, profile.id)).to_have_attribute("aria-selected", "true", timeout=_TIMEOUT_MS)

    # A fresh context: the server's last-active.
    fresh = chromium_browser.new_context(viewport={"width": 1400, "height": 900})
    try:
        other = fresh.new_page()
        _boot(other, hub)
        expect(_active(other)).to_have_attribute("data-session-id", sid, timeout=_TIMEOUT_MS)
        # Its last-open artifact comes back: shown (and selected) when docked,
        # else revealed in the tree.
        expect(other.locator(f'{_TREE} .osprey-list-row[data-id="{profile.id}"]')).to_have_count(
            1, timeout=_TIMEOUT_MS
        )
    finally:
        fresh.close()

    # Signing out clears the pointer (asserted through #logout-btn in
    # test_logout_resume_browser.py); here the pointer is cleared as the logout
    # leaves it, and the next login resumes the server's last-active.
    from osprey.interfaces._serving import authorize_browser_context

    page.evaluate("() => fetch('api/terminal/logout', {method: 'POST'})")
    page.evaluate("() => localStorage.removeItem('osprey-pty-session')")
    authorize_browser_context(page.context)
    page.reload(wait_until="domcontentloaded")
    expect(_active(page)).to_have_attribute("data-session-id", sid, timeout=_TIMEOUT_MS)

    # The pen button opens a new session, and a reload stays on it.
    page.locator("button.op-session-new").click()
    expect(_active(page)).to_have_attribute(
        "data-session-id", f"key:{_pointer(page)}", timeout=_TIMEOUT_MS
    )
    minted = _pointer(page)
    _settle_pointer(page)
    page.reload(wait_until="domcontentloaded")
    expect(_active(page)).to_have_attribute("data-session-id", f"key:{minted}", timeout=_TIMEOUT_MS)


def test_without_a_docked_artifacts_panel_page_load_only_reveals(hub, page):
    profile = next(e for e in _store(hub).list_entries() if e.kind == BEAM_PROFILE_KIND)
    sid = _PROFILE.transcript_id
    _boot(page, hub)
    _session(page, sid).locator(".osprey-tree-title").click()
    expect(_active(page)).to_have_attribute("data-session-id", sid, timeout=_TIMEOUT_MS)
    _open_all(page)
    _row(page, profile.id).click()
    expect(_row(page, profile.id)).to_have_attribute("aria-selected", "true")

    # The hermetic hub serves Artifacts alone, so a merely closed tile leaves an
    # empty service slot that the boot landing fills again (panel-manager's
    # ensureActivePanel, by design). Removing the panel from the rail is the
    # operator's way to keep it undocked across a reload.
    _remove_artifacts_from_rail(page)
    page.wait_for_timeout(1500)  # the layout and the pointer are written debounced
    page.reload(wait_until="domcontentloaded")
    expect(_active(page)).to_have_attribute("data-session-id", sid, timeout=_TIMEOUT_MS)
    # The layout is left as it was: the Artifacts tile is not docked again.
    page.wait_for_timeout(1000)
    expect(page.locator(_ARTIFACTS_TILE_CLOSE)).to_have_count(0)
    expect(_row(page, profile.id)).to_have_count(1, timeout=_TIMEOUT_MS)
    # Nothing shows an artifact, so no row is highlighted.
    expect(page.locator(f'{_TREE} [aria-selected="true"]')).to_have_count(0)


def _remove_artifacts_from_rail(page: Page) -> None:
    """Take Artifacts off the rail (the POST the panel menu's remove sends)."""
    ok = page.evaluate(
        """() => fetch('api/panel-visibility', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({panel: 'artifacts', visible: false}),
        }).then((r) => r.ok)"""
    )
    assert ok
    expect(page.locator(_ARTIFACTS_TILE_CLOSE)).to_have_count(0, timeout=_TIMEOUT_MS)


# --- the streaming guard and browsing -----------------------------------------------


def test_the_agent_row_folds_the_tree_and_option_click_expands_every_session(hub, page):
    """With the Agent pane open the Agent row folds and unfolds the whole tree (a
    display rule once outranked ``hidden``, so the tree stayed on screen); ⌥-click
    expands every session, a second ⌥-click collapses them all. The row has no
    chevron, and its menu's Expand all does what ⌥-click does."""
    _boot(page, hub)
    agent = page.locator("#sidebar-agent-row")
    tree = page.locator(_TREE)
    expect(page.locator("html")).not_to_have_attribute("data-agent", "closed")
    expect(tree).to_be_visible()
    expect(agent.locator("svg.osprey-nav-disc, [data-disc]")).to_have_count(0)

    agent.click()
    expect(agent).to_have_attribute("aria-expanded", "false")
    expect(tree).to_be_hidden()
    agent.click()
    expect(agent).to_have_attribute("aria-expanded", "true")
    expect(tree).to_be_visible()

    sessions = page.locator(f"{_TREE} .osprey-tree-session")
    folded = page.locator(f'{_TREE} .osprey-tree-session[aria-expanded="false"]')
    expect(sessions).not_to_have_count(0)
    expect(sessions).not_to_have_count(1)
    agent.click(modifiers=["Alt"])
    expect(folded).to_have_count(0)
    expect(page.locator(f"{_TREE} [data-session-box]")).to_have_count(sessions.count())
    expect(page.locator("html")).not_to_have_attribute("data-agent", "closed")

    sessions.last.locator(".osprey-tree-title").click(modifiers=["Alt"])
    expect(page.locator(f'{_TREE} .osprey-tree-session[aria-expanded="true"]')).to_have_count(0)
    expect(page.locator(f"{_TREE} [data-session-box]")).to_have_count(0)

    # The menu's Expand all is the same gesture: every session opens.
    _menu(page, agent, "Expand all")
    expect(folded).to_have_count(0)
    expect(page.locator(f"{_TREE} [data-session-box]")).to_have_count(sessions.count())
    expect(agent).to_have_attribute("aria-expanded", "true")

    # Keyboard still folds the tree from the row.
    agent.focus()
    page.keyboard.press("Enter")
    expect(agent).to_have_attribute("aria-expanded", "false")
    page.keyboard.press("Space")
    expect(agent).to_have_attribute("aria-expanded", "true")


def test_a_session_click_while_streaming_asks_and_another_session_never_opens_in_place(hub, page):
    _boot(page, hub)
    before = _active(page).get_attribute("data-session-id")

    # Only the session in use shows its artifacts: another has no disclosure.
    other = _session(page, _PROFILE.transcript_id)
    expect(other).to_have_attribute("aria-expanded", "false")
    expect(other.locator("[data-tree-disc]")).to_have_count(0)
    expect(_active(page)).to_have_attribute("data-session-id", before)

    hub.chat.delay_s = 30.0
    _send(page, "Scan the corrector currents.")
    expect(page.locator("#operator-container")).to_have_class(
        _re(r"\bstreaming\b"), timeout=_TIMEOUT_MS
    )

    target = _session(page, _PROFILE.transcript_id).locator(".osprey-tree-title")
    target.click()
    dialog = page.locator(_DIALOG)
    expect(dialog.locator(".posture-modal-title")).to_have_text("Stop the turn?")
    dialog.locator(".posture-modal-cancel").click()
    expect(page.locator(_DIALOG)).to_have_count(0)
    expect(_active(page)).not_to_have_attribute("data-session-id", _PROFILE.transcript_id)

    target.click()
    page.locator(_DIALOG).locator(".posture-modal-confirm", has_text="Stop & switch").click()
    expect(_active(page)).to_have_attribute(
        "data-session-id", _PROFILE.transcript_id, timeout=_TIMEOUT_MS
    )
    # The turn was stopped, not left running behind the switch.
    expect(page.locator("#operator-container")).not_to_have_class(
        _re(r"\bstreaming\b"), timeout=_TIMEOUT_MS
    )


def _re(pattern: str):
    import re

    return re.compile(pattern)


# --- each session keeps its own layout ----------------------------------------------


def test_a_panel_closed_in_one_session_is_back_on_returning_to_another(hub, page):
    a, b = _PROFILE.transcript_id, _RF_AFTER_CLEAR.transcript_id
    tile = page.locator(_ARTIFACTS_TILE_CLOSE)
    _boot(page, hub)
    _session(page, a).locator(".osprey-tree-title").click()
    expect(_active(page)).to_have_attribute("data-session-id", a, timeout=_TIMEOUT_MS)
    expect(tile).to_have_count(1, timeout=_TIMEOUT_MS)
    page.wait_for_timeout(1000)  # the session's layout is written debounced

    # B starts from the layout on screen; closing the panel there is B's alone.
    _session(page, b).locator(".osprey-tree-title").click()
    expect(_active(page)).to_have_attribute("data-session-id", b, timeout=_TIMEOUT_MS)
    tile.click()
    expect(tile).to_have_count(0, timeout=_TIMEOUT_MS)
    page.wait_for_timeout(1000)

    _session(page, a).locator(".osprey-tree-title").click()
    expect(_active(page)).to_have_attribute("data-session-id", a, timeout=_TIMEOUT_MS)
    expect(tile).to_have_count(1, timeout=_TIMEOUT_MS)

    _session(page, b).locator(".osprey-tree-title").click()
    expect(_active(page)).to_have_attribute("data-session-id", b, timeout=_TIMEOUT_MS)
    expect(tile).to_have_count(0, timeout=_TIMEOUT_MS)


# --- names, the switcher and More sessions ------------------------------------------


def test_a_name_arrives_live_and_rename_reverts_and_the_switcher_switches(hub, page):
    _boot(page, hub)
    hub.naming.delay_s = 3.0
    _send(page, "Measure the beam lifetime now.")
    row = _active(page)
    expect(row).to_contain_text("naming…", timeout=_TIMEOUT_MS)
    title = row.locator(".osprey-tree-title")
    expect(title).to_have_text("Named: Measure the beam lifetime", timeout=_TIMEOUT_MS)
    expect(row).not_to_contain_text("naming…")
    hub.naming.delay_s = 0.0

    # An automatic name has nothing to revert, and a click already opens the session.
    row.click(button="right")
    expect(page.get_by_role("menuitem", name="Rename…")).to_be_visible()
    expect(page.get_by_role("menuitem", name="Use automatic name")).to_have_count(0)
    expect(page.get_by_role("menuitem", name="Open in Agent")).to_have_count(0)
    page.keyboard.press("Escape")

    _menu(page, row, "Rename…")
    rename = page.locator("input.osprey-tree-rename")
    rename.fill("Beam check")
    rename.press("Enter")
    expect(title).to_have_text("Beam check", timeout=_TIMEOUT_MS)
    _menu(page, row, "Use automatic name")
    expect(title).to_have_text("Named: Measure the beam lifetime", timeout=_TIMEOUT_MS)

    # The switcher lists the sessions and switches the Agent.
    page.locator("span.op-session-label").click()
    menu = page.locator("#session-switcher-menu")
    expect(menu).to_be_visible()
    menu.locator(f'.session-switcher-row[data-transcript-id="{_PROFILE.transcript_id}"]').click()
    expect(_active(page)).to_have_attribute(
        "data-session-id", _PROFILE.transcript_id, timeout=_TIMEOUT_MS
    )
    page.locator("span.op-session-label").click()
    menu.locator(".session-switcher-new").click()
    expect(_active(page)).to_have_attribute("data-session-id", _re(r"^key:"), timeout=_TIMEOUT_MS)

    # Only the active session's title is bold; past ones sit at the regular weight.
    # The tree re-renders after the switch; to_have_css re-resolves and retries,
    # where a one-shot evaluate can read a detached node's empty style.
    expect(_active(page).locator(".osprey-tree-title")).to_have_css(
        "font-weight", "600", timeout=_TIMEOUT_MS
    )
    past = page.locator(f"{_TREE} .osprey-tree-session:not(.active) .osprey-tree-title")
    expect(past.first).to_be_visible(timeout=_TIMEOUT_MS)
    for i in range(past.count()):
        expect(past.nth(i)).to_have_css("font-weight", "400", timeout=_TIMEOUT_MS)


def _many_sessions(project_dir: Path, session_dir: Path) -> None:  # noqa: ARG001
    from osprey.interfaces.web_terminal import session_meta
    from osprey.stores import session_ledger

    state_dir = session_meta.ledger_state_dir()
    start = datetime.now(UTC) - timedelta(days=2)
    for n in range(205):
        tid = str(uuid.UUID(int=n + 1))
        when = start + timedelta(minutes=n)
        _append_transcript(session_dir, tid, _exchange(tid, f"Question {n}", "Done", when))
        session_ledger.append(state_dir, tid, tid, when)


@pytest.fixture
def crowded_hub():
    with hermetic_hub(e2e=True, seed=_many_sessions) as running:
        yield running


def test_more_sessions_loads_the_page_after_the_first(crowded_hub, page):
    _boot(page, crowded_hub)
    more = page.locator(f"{_TREE} [data-tree-more]")
    expect(more).to_be_visible(timeout=_TIMEOUT_MS)
    first_page = page.locator(f"{_TREE} .osprey-tree-session").count()
    more.click()
    expect(page.locator(f"{_TREE} .osprey-tree-session")).not_to_have_count(
        first_page, timeout=_TIMEOUT_MS
    )
    assert page.locator(f"{_TREE} .osprey-tree-session").count() > first_page


# --- the extension viewer -----------------------------------------------------------


def test_beam_profile_renders_through_its_stub_viewer_and_falls_back(hub, page):
    store = _store(hub)
    profile = next(e for e in store.list_entries() if e.kind == BEAM_PROFILE_KIND)
    broken = store.save_file(
        json.dumps({"sigma_x_um": 1.0}).encode("utf-8"),
        "broken_profile.json",
        "json",
        title="Broken beam profile",
        mime_type="application/json",
        metadata={"viewer_throws": True},
        session_id=_PROFILE.session_key,
        kind=BEAM_PROFILE_KIND,
    )
    warnings: list[str] = []
    page.on("console", lambda m: warnings.append(m.text) if m.type == "warning" else None)
    _boot(page, hub)
    _open_session(page, _PROFILE.transcript_id)
    _open_all(page)
    frame = page.frame_locator(_ARTIFACTS_FRAME)

    _row(page, profile.id).click()
    marker = frame.locator(f".ext-viewer-container .{BEAM_PROFILE_MARKER_CLASS}")
    expect(marker).to_be_visible(timeout=_TIMEOUT_MS)
    expect(marker).to_have_attribute("data-viewer", BEAM_PROFILE_KIND)

    _row(page, broken.id).click()
    expect(frame.locator(f".{BEAM_PROFILE_MARKER_CLASS}")).to_have_count(0, timeout=_TIMEOUT_MS)
    expect(frame.locator("#preview-content .preview-viewport")).to_be_visible()
    _wait_for(lambda: any("failed; showing the base view" in w for w in warnings))


def test_open_in_new_window_keeps_the_artifact_the_tile_showed(hub, page):
    """The frame loads again in its window and asks for its selection there: the
    hub finds the frame wherever it lives and answers, so the window shows the
    artifact the tile showed, and so does the tile after Reattach."""
    store = _store(hub)
    profile = next(e for e in store.list_entries() if e.kind == BEAM_PROFILE_KIND)
    _boot(page, hub)
    _open_session(page, _PROFILE.transcript_id)
    _open_all(page)
    _row(page, profile.id).click()
    marker = f".ext-viewer-container .{BEAM_PROFILE_MARKER_CLASS}"
    expect(page.frame_locator(_ARTIFACTS_FRAME).locator(marker)).to_be_visible(timeout=_TIMEOUT_MS)

    with page.expect_popup() as info:
        page.locator('.tile-tab[data-panel-id="iframe:artifacts"] .tile-tab-window').click()
    popup = info.value
    frame = popup.frame_locator(_ARTIFACTS_FRAME)
    expect(frame.locator(marker)).to_be_visible(timeout=_TIMEOUT_MS)
    expect(frame.locator("#preview-empty")).to_be_hidden()

    # A pick in the hub's tree reaches the frame in the window.
    note = store.save_file(
        b"# window note\n",
        "window_note.md",
        "markdown",
        title="Window note",
        session_id=_PROFILE.session_key,
    )
    _open_all(page)
    _row(page, note.id).click()
    expect(frame.locator(marker)).to_have_count(0, timeout=_TIMEOUT_MS)
    expect(frame.locator("#preview-content .preview-viewport")).to_be_visible()

    popup.locator(".attached-window-root > .tile-tab button.tile-tab-reattach").click()
    docked = page.frame_locator(_ARTIFACTS_FRAME)
    expect(docked.locator("#preview-content .preview-viewport")).to_be_visible(timeout=_TIMEOUT_MS)
    expect(docked.locator("#preview-empty")).to_be_hidden()


# --- the palette and delete ---------------------------------------------------------


def test_palette_find_enters_a_session_and_delete_asks_first(hub, page):
    _boot(page, hub)
    # Named or still under its first prompt, the session's title carries this text;
    # the name arriving live is test_a_name_arrives_live_..., not this test's subject.
    title = "Check the RF cavity"
    expect(_session(page, _RF_FIRST.transcript_id).locator(".osprey-tree-title")).to_contain_text(
        title, timeout=_TIMEOUT_MS
    )
    page.locator("#command-palette-btn").click()
    page.locator("input.command-palette-input").fill("Check the RF")
    page.locator(".command-palette-item--sessions", has_text=title).first.click()
    expect(_active(page)).to_have_attribute(
        "data-session-id", _RF_FIRST.transcript_id, timeout=_TIMEOUT_MS
    )

    store = _store(hub)
    doomed = store.save_file(
        b"# doomed\n",
        "doomed.md",
        "markdown",
        title="Doomed note",
        session_id=_PROFILE.session_key,
    )
    _open_session(page, _PROFILE.transcript_id)
    _open_all(page)
    row = _row(page, doomed.id)
    expect(row).to_have_count(1, timeout=_TIMEOUT_MS)
    _menu(page, row, "Delete…")
    dialog = page.locator(_DIALOG)
    expect(dialog.locator(".posture-modal-title")).to_have_text("Delete artifact?")
    dialog.locator(".posture-modal-cancel").click()
    expect(row).to_have_count(1)
    assert store.get_entry(doomed.id) is not None

    _menu(page, row, "Delete…")
    page.locator(_DIALOG).locator(".posture-modal-confirm").click()
    expect(row).to_have_count(0, timeout=_TIMEOUT_MS)


def test_deleting_a_past_session_drops_its_row_and_its_artifacts(hub, page):
    profile = next(e for e in _store(hub).list_entries() if e.kind == BEAM_PROFILE_KIND)
    sid = _PROFILE.transcript_id
    _boot(page, hub)
    current = _session(page, _RF_AFTER_CLEAR.transcript_id)
    current.locator(".osprey-tree-title").click()
    expect(_active(page)).to_have_attribute(
        "data-session-id", _RF_AFTER_CLEAR.transcript_id, timeout=_TIMEOUT_MS
    )
    _settle_pointer(page)

    # The session the Agent is on offers Delete… disabled.
    current.click(button="right")
    expect(page.get_by_role("menuitem", name="Delete…")).to_have_attribute("aria-disabled", "true")
    page.keyboard.press("Escape")

    row = _session(page, sid)
    expect(row).to_have_attribute("data-count", _re(r"^[1-9]"), timeout=_TIMEOUT_MS)
    _menu(page, row, "Delete…")
    dialog = page.locator(_DIALOG)
    expect(dialog.locator(".posture-modal-title")).to_have_text("Delete session?")
    expect(dialog).to_contain_text("This deletes the session")
    dialog.locator(".posture-modal-confirm").click()
    expect(row).to_have_count(0, timeout=_TIMEOUT_MS)
    assert _store(hub).get_entry(profile.id) is None

    page.reload(wait_until="domcontentloaded")
    expect(_active(page)).to_have_attribute(
        "data-session-id", _RF_AFTER_CLEAR.transcript_id, timeout=_TIMEOUT_MS
    )
    expect(_session(page, sid)).to_have_count(0)
