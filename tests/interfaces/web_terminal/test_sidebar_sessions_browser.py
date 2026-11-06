"""Browser e2e: the Sessions row and its session rows on the hermetic hub.

The sidebar row that lists the agent's sessions is named "Sessions" (the pane it
opens stays "Agent"); its "+" starts a new session; a clicked session takes the
accent; the session the agent works in breathes (still under reduced motion); and a
folded session shows its artifacts in a peek on hover, where a click opens one.
"""

from __future__ import annotations

import re
from contextlib import suppress

import pytest
from docs.screenshots.contact_sheet import (
    BEAM_PROFILE_KIND,
    BEAM_PROFILE_MARKER_CLASS,
    SEEDED_SESSIONS,
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
_ROW = "#sidebar-agent-row"
_PEEK = "#sidebar-session-peek"
_ARTIFACTS_FRAME = 'iframe.panel-iframe[data-panel-id="artifacts"]'


@pytest.fixture
def hub():
    with hermetic_hub(e2e=True) as running:
        yield running


@pytest.fixture
def context(chromium_browser):
    ctx = chromium_browser.new_context(viewport={"width": 1400, "height": 900})
    yield ctx
    ctx.close()


@pytest.fixture
def page(context):
    return context.new_page()


def _boot(page: Page, hub) -> None:
    page.goto(hub.base_url, wait_until="domcontentloaded")
    expect(page.locator(f"{_TREE} .osprey-tree-session").first).to_be_visible(timeout=_TIMEOUT_MS)


def _session(page: Page, sid: str):
    return page.locator(f'{_TREE} .osprey-tree-session[data-session-id="{sid}"]')


def _active(page: Page):
    return page.locator(f"{_TREE} .osprey-tree-session.active")


def _send(page: Page, prompt: str) -> None:
    box = page.locator('#operator-container textarea[aria-label="Message the agent"]')
    box.fill(prompt)
    for _ in range(20):
        box.press("Enter")
        with suppress(AssertionError):
            expect(box).to_have_value("", timeout=500)
            return
    expect(box).to_have_value("")


def _bar(page: Page, sid: str, prop: str) -> str:
    """A computed property of a session row's accent bar (its ``::before``)."""
    return page.evaluate(
        """([sid, prop]) => {
          const row = document.querySelector(
            `#sidebar-session-tree .osprey-tree-session[data-session-id="${sid}"]`);
          return getComputedStyle(row, '::before').getPropertyValue(prop);
        }""",
        [sid, prop],
    )


def test_the_row_is_named_sessions_and_its_plus_starts_a_new_session(hub, page):
    _boot(page, hub)
    row = page.locator(_ROW)
    expect(row.locator(".osprey-nav-label")).to_have_text("Sessions")
    expect(row).to_have_attribute("aria-label", "Sessions")
    expect(row).to_have_attribute("data-tip", "Sessions")
    # Expanded, the row is a heading: its icon is in the DOM but not shown.
    expect(row.locator('svg[data-icon="agent"]')).to_have_count(1)
    expect(row.locator('svg[data-icon="agent"]')).to_be_hidden()
    # Collapsed to icons only, the row's icon is shown.
    page.locator("#sidebar-toggle").click()
    expect(page.locator("html")).to_have_attribute("data-sidebar", "collapsed")
    expect(row.locator('svg[data-icon="agent"]')).to_be_visible()
    page.locator("#sidebar-toggle").click()
    expect(page.locator("html")).to_have_attribute("data-sidebar", "expanded")
    expect(row.locator('svg[data-icon="agent"]')).to_be_hidden()
    # One menu, the right-click one: no "more" button beside the row.
    expect(page.locator("#sidebar-agent-wrap button:not(#sidebar-session-tree *)")).to_have_count(2)
    row.click(button="right")
    expect(page.get_by_role("menu", name="Sessions actions")).to_be_visible()
    page.keyboard.press("Escape")

    # Away to a recorded session, then "+": a new session becomes the active one.
    _session(page, _PROFILE.transcript_id).locator(".osprey-tree-title").click()
    expect(_active(page)).to_have_attribute(
        "data-session-id", _PROFILE.transcript_id, timeout=_TIMEOUT_MS
    )
    plus = page.locator("#sidebar-new-session")
    expect(plus).to_have_attribute("aria-label", "New session")
    plus.click()
    expect(_active(page)).to_have_attribute(
        "data-session-id", re.compile(r"^key:"), timeout=_TIMEOUT_MS
    )
    expect(_active(page).locator(".osprey-tree-title")).to_have_text("New session")


def test_a_clicked_session_takes_the_accent_and_only_the_active_one_is_bold(hub, page):
    _boot(page, hub)
    before = _active(page).get_attribute("data-session-id")
    assert before != _PROFILE.transcript_id
    expect(page.locator(f"{_TREE} .osprey-tree-session.accent")).to_have_count(1)

    _session(page, _PROFILE.transcript_id).locator(".osprey-tree-title").click()
    target = _session(page, _PROFILE.transcript_id)
    expect(target).to_have_class(re.compile(r"\baccent\b"), timeout=_TIMEOUT_MS)
    expect(target).to_have_class(re.compile(r"\bactive\b"), timeout=_TIMEOUT_MS)
    expect(page.locator(f"{_TREE} .osprey-tree-session.accent")).to_have_count(1)
    expect(page.locator(f"{_TREE} .osprey-tree-session.active")).to_have_count(1)
    assert _bar(page, _PROFILE.transcript_id, "content") != "none"
    weights = page.evaluate(
        """() => [...document.querySelectorAll('#sidebar-session-tree .osprey-tree-session')]
          .map((r) => [r.classList.contains('active'), Number(getComputedStyle(r).fontWeight)])"""
    )
    assert all((w >= 600) == active for active, w in weights), weights


def test_the_session_the_agent_works_in_breathes(hub, page):
    _boot(page, hub)
    sid = _active(page).get_attribute("data-session-id")
    expect(page.locator(f"{_TREE} .osprey-tree-session.is-busy")).to_have_count(0)

    hub.chat.delay_s = 30.0
    _send(page, "Scan the corrector currents.")
    busy = page.locator(f"{_TREE} .osprey-tree-session.is-busy")
    expect(busy).to_have_count(1, timeout=_TIMEOUT_MS)
    sid = busy.get_attribute("data-session-id")
    expect(busy).to_have_class(re.compile(r"\bactive\b"))
    assert _bar(page, sid, "animation-name") == "osprey-tree-pulse"

    # Reduced motion: the accent stays, still.
    page.emulate_media(reduced_motion="reduce")
    assert _bar(page, sid, "animation-name") == "none"
    assert _bar(page, sid, "content") != "none"


def test_a_folded_session_peeks_at_its_artifacts_and_a_click_opens_one(hub, page):
    from osprey.stores.artifact_store import ArtifactStore

    store = ArtifactStore(workspace_root=hub.project_dir, auto_launch=False)
    profile = next(e for e in store.list_entries() if e.kind == BEAM_PROFILE_KIND)
    _boot(page, hub)
    before = _active(page).get_attribute("data-session-id")
    folded = _session(page, _PROFILE.transcript_id)
    expect(folded).to_have_attribute("aria-expanded", "false")

    peek = page.locator(_PEEK)
    folded.locator(".osprey-tree-title").hover()
    expect(peek).to_be_visible(timeout=_TIMEOUT_MS)
    expect(peek.locator(".osprey-popover-title")).to_have_text(
        folded.locator(".osprey-tree-title").inner_text()
    )
    row = peek.locator(f'.osprey-list-row[data-id="{profile.id}"]')
    expect(row).to_be_visible()
    # The peek is the only popup: the row title has no tooltip of its own.
    page.wait_for_timeout(600)
    expect(page.locator("#osprey-tooltip")).to_be_hidden()
    title = folded.locator(".osprey-tree-title")
    expect(title).not_to_have_attribute("title", re.compile(".*"))
    expect(title).not_to_have_attribute("data-tip-detail", re.compile(".*"))

    # The pointer moves into the peek: it stays open.
    row.hover()
    page.wait_for_timeout(400)
    expect(peek).to_be_visible()

    # Escape closes it; a fresh hover opens it again.
    page.keyboard.press("Escape")
    expect(peek).to_be_hidden()
    page.mouse.move(900, 600)
    folded.locator(".osprey-tree-title").hover()
    expect(peek).to_be_visible(timeout=_TIMEOUT_MS)

    # A click opens the artifact in the viewer, as the tree does; no switch.
    peek.locator(f'.osprey-list-row[data-id="{profile.id}"]').click()
    expect(peek).to_be_hidden()
    marker = page.frame_locator(_ARTIFACTS_FRAME).locator(
        f".ext-viewer-container .{BEAM_PROFILE_MARKER_CLASS}"
    )
    expect(marker).to_be_visible(timeout=_TIMEOUT_MS)
    expect(_active(page)).to_have_attribute("data-session-id", before)

    # Leaving the row and the peek closes it.
    folded.locator(".osprey-tree-title").hover()
    expect(peek).to_be_visible(timeout=_TIMEOUT_MS)
    page.mouse.move(900, 600)
    expect(peek).to_be_hidden()


def test_the_keyboard_reaches_the_pin_button_and_keeps_it_through_the_redraw(hub, page):
    _boot(page, hub)
    page.mouse.move(700, 450)
    row = _session(page, _PROFILE.transcript_id)
    button = row.locator("[data-tree-pin]")
    count = row.locator(".osprey-tree-n")

    # At rest it is out of sight, and the count shows.
    expect(count).to_be_visible()
    expect(button).to_have_css("width", "1px")

    # Focus brings it up in the count's place.
    button.focus()
    expect(button).to_be_focused()
    expect(count).to_be_hidden()
    expect(button).to_have_css("width", "20px")

    # Enter pins, and the focus survives the redraw.
    page.keyboard.press("Enter")
    expect(button).to_have_attribute("aria-pressed", "true", timeout=_TIMEOUT_MS)
    expect(row.locator("[data-tree-pin]")).to_be_focused()


def test_the_sessions_row_tip_follows_the_agent_pane_and_session_rows_have_none(hub, page):
    _boot(page, hub)
    title = _active(page).locator(".osprey-tree-title")
    expect(title).not_to_have_attribute("data-tip-detail", re.compile(".*"))
    page.locator(_ROW).focus()
    page.keyboard.press("Delete")
    expect(page.locator("html")).to_have_attribute("data-agent", "closed", timeout=_TIMEOUT_MS)
    expect(page.locator(_ROW)).to_have_attribute(
        "data-tip", "Sessions · Agent pane closed — click to open"
    )
