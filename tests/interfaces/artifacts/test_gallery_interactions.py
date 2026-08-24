"""Browser Playwright suite pinning the artifacts gallery's interaction contract.

The acceptance pin for the artifacts module cluster (the state/types/
render/preview/timeseries modules plus the slim gallery.js entry) and for
logbook.js/print.js being real ES modules (direct imports, with no
`window._galleryState`/`window.inject*` global bridge). This suite drives
the whole chain through a real browser against real fixture artifacts on
disk, rather than the fake-host unit tests each module's own Vitest suite
uses -- so a wiring mistake anywhere in the module graph (a renamed
selector, a dropped import, an inject function that stopped being called)
shows up as an actual interaction failure here.

Deliberately written at arms length from the module code: these
assertions target what a user sees in the DOM (button classes, mounted
content, visible text) -- an independent check, not a rubber stamp.

Covers:
  - the toolbar filter input narrows the sidebar tree as the user types
  - the ⋯ overflow menu opens, its "All sessions" item toggles the scope
    (aria-checked + the scope pill above the list), and the pill's ✕
    returns to the session scope
  - selecting an artifact of a given type renders that type's preview
    header (title, type badge) and viewport (markdown -> the marked/KaTeX
    container, html -> a sandboxed iframe)
  - Simple mode's result card renders those same viewports, through the
    same dispatch, for every seeded type
  - the logbook/print inject buttons are present in the preview's action
    bar -- proving the direct-import wiring actually renders them (a
    grep-level check that no window bridge exists can't show that)

Unlike test_load_smokes.py's `_launch_artifacts` (an empty tmp_path -- fine
there, since it only checks the page loads clean), this suite needs real
on-disk artifacts: ArtifactStore.list_entries() reflects whatever the
`artifacts.json` index says, and the gallery has nothing to filter/select
without it. `_launch_artifacts` here first seeds three fixture artifacts
(markdown + html + json) via a throwaway ArtifactStore pointed at the same
`workspace_root` the app will use -- BaseStore.__init__ loads the index
from disk, so the app's own store picks up the seeded entries on
construction (mirrors the `_make_artifact` helper already used in
test_logbook.py for the non-browser artifact tests).

Run:
    .venv/bin/pytest tests/interfaces/artifacts/test_gallery_interactions.py -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import TYPE_CHECKING

import pytest

from tests.interfaces.conftest import _run_app_server

if TYPE_CHECKING:
    from collections.abc import Iterator

    from playwright.sync_api import Page

try:
    from playwright.sync_api import expect

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False

pytestmark = [pytest.mark.browser, pytest.mark.slow]

VIEWPORT = {"width": 1280, "height": 800}

MARKDOWN_TITLE = "Beam Current Summary"
HTML_TITLE = "Orbit Plot"
JSON_TITLE = "Corrector Readback Set"


# ---------------------------------------------------------------------------
# Live server launcher
# ---------------------------------------------------------------------------


@contextmanager
def _launch_artifacts(tmp_path, monkeypatch) -> Iterator[str]:
    """Seed three fixture artifacts (markdown + html + json), then serve the gallery.

    Mirrors test_logbook.py's `_make_artifact` helper: a throwaway
    ArtifactStore writes the index + content files, then create_app()'s own
    ArtifactStore (same workspace_root) loads that index at construction.

    Each fixture sets a real `category` (from type_registry.py's CATEGORIES,
    not ARTIFACT_TYPES) -- render.js's tree sections group by
    `category || artifact_type`, and the preview badge class reflects the
    category. Viewport selection in preview.js is unaffected -- that
    dispatch switches on the raw `artifact_type`, not `category`.
    """
    monkeypatch.chdir(tmp_path)

    from osprey.stores.artifact_store import ArtifactStore

    seed_store = ArtifactStore(workspace_root=tmp_path)
    seed_store.save_file(
        file_content=b"# Beam Current\n\nSR current held steady at 400 mA overnight.\n",
        filename="beam-current.md",
        artifact_type="markdown",
        title=MARKDOWN_TITLE,
        description="A markdown fixture artifact",
        mime_type="text/markdown",
        tool_source="test_fixture",
        category="document",
    )
    seed_store.save_file(
        file_content=b"<html><body><h1>Orbit</h1></body></html>",
        filename="orbit.html",
        artifact_type="html",
        title=HTML_TITLE,
        description="An html fixture artifact",
        mime_type="text/html",
        tool_source="test_fixture",
        category="visualization",
    )
    # A json fixture: the shape a channel-finder result lands as, and one of
    # the types Simple mode could not render while it had a viewport builder
    # of its own. Seeded last, so it is the newest -- Simple's result card
    # shows the newest artifact until the user picks another.
    seed_store.save_file(
        file_content=b'{"correctors": {"SR01C:HCM1:SP": 0.42}}',
        filename="correctors.json",
        artifact_type="json",
        title=JSON_TITLE,
        description="A json fixture artifact",
        mime_type="application/json",
        tool_source="test_fixture",
        category="channel_values",
    )

    from osprey.interfaces.artifacts.app import create_app

    app = create_app(workspace_root=tmp_path)
    with _run_app_server(app) as base_url:
        yield base_url


def _card_for_title(page: Page, title: str):
    """Locate the sidebar tree-item for a fixture artifact by its title text.

    Selecting by title (via the item's own text, not a hardcoded artifact
    id -- ArtifactStore._make_id() generates a fresh id every run) keeps
    this stable across runs; `.tree-item[data-id=...]` is the default
    (browseMode="tree", sidebarLayout="list") card markup rendered by
    render.js, expanded by default (no click-to-expand step needed).
    """
    return page.locator(".tree-item", has_text=title)


# ---------------------------------------------------------------------------
# Test 1: the toolbar filter input narrows the sidebar tree
# ---------------------------------------------------------------------------


def test_search_input_narrows_sidebar(tmp_path, monkeypatch, chromium_browser):
    """Typing in the toolbar filter hides non-matching artifacts (debounced);
    clearing it restores both fixtures.
    """
    with _launch_artifacts(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page(viewport=VIEWPORT)
        page.goto(base_url, wait_until="domcontentloaded")

        markdown_card = _card_for_title(page, MARKDOWN_TITLE)
        html_card = _card_for_title(page, HTML_TITLE)
        expect(markdown_card).to_be_visible(timeout=10_000)
        expect(html_card).to_be_visible(timeout=10_000)

        search = page.locator("#search")
        search.fill("beam current")
        expect(markdown_card).to_be_visible(timeout=5_000)
        expect(html_card).to_have_count(0, timeout=5_000)

        search.fill("orbit")
        expect(html_card).to_be_visible(timeout=5_000)
        expect(markdown_card).to_have_count(0, timeout=5_000)

        search.fill("")
        expect(markdown_card).to_be_visible(timeout=5_000)
        expect(html_card).to_be_visible(timeout=5_000)
        page.close()


# ---------------------------------------------------------------------------
# Test 1b: overflow menu — all-sessions scope toggle + scope pill
# ---------------------------------------------------------------------------


def test_overflow_menu_toggles_all_sessions_scope(tmp_path, monkeypatch, chromium_browser):
    """The ⋯ menu opens, its "All sessions" item flips aria-checked and shows
    the scope pill; the pill's ✕ clears the scope again. (With no session
    scoping active the artifact list itself is unchanged either way -- this
    pins the scope-state UI contract, not the fetch filtering, which
    state.test.mjs covers.)
    """
    with _launch_artifacts(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page(viewport=VIEWPORT)
        page.goto(base_url, wait_until="domcontentloaded")
        expect(_card_for_title(page, MARKDOWN_TITLE)).to_be_visible(timeout=10_000)

        menu = page.locator("#sidebar-menu")
        menu_btn = page.locator("#sidebar-menu-btn")
        all_sessions = page.locator("#all-sessions-btn")
        pill = page.locator("#scope-pill")

        expect(menu).to_be_hidden()
        expect(pill).to_be_hidden()

        menu_btn.click()
        expect(menu).to_be_visible()
        expect(all_sessions).to_have_attribute("aria-checked", "false")

        # Toggling the scope closes the menu and raises the pill.
        all_sessions.click()
        expect(menu).to_be_hidden()
        expect(pill).to_be_visible(timeout=5_000)
        menu_btn.click()
        expect(all_sessions).to_have_attribute("aria-checked", "true")
        menu_btn.click()

        # The pill's ✕ returns to the default this-session scope.
        page.locator("#scope-pill-clear").click()
        expect(pill).to_be_hidden(timeout=5_000)
        menu_btn.click()
        expect(all_sessions).to_have_attribute("aria-checked", "false")
        page.close()


# ---------------------------------------------------------------------------
# Test 2: selecting an artifact renders its type's preview header + viewport
# ---------------------------------------------------------------------------


def test_selecting_artifact_renders_preview_for_its_type(tmp_path, monkeypatch, chromium_browser):
    """Preview header (title, badge) and viewport differ per artifact type."""
    with _launch_artifacts(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page(viewport=VIEWPORT)
        page.goto(base_url, wait_until="domcontentloaded")

        _card_for_title(page, MARKDOWN_TITLE).click()
        expect(page.locator(".preview-header-title")).to_have_text(MARKDOWN_TITLE, timeout=10_000)
        # Badge class reflects `category` (set on the fixture, see
        # _launch_artifacts) over artifact_type; viewport dispatch below is
        # keyed on the raw artifact_type regardless.
        expect(page.locator(".preview-header .badge")).to_have_class("badge badge-document")
        expect(page.locator(".preview-viewport .md-preview-container")).to_be_visible(timeout=5_000)

        _card_for_title(page, HTML_TITLE).click()
        expect(page.locator(".preview-header-title")).to_have_text(HTML_TITLE, timeout=10_000)
        expect(page.locator(".preview-header .badge")).to_have_class("badge badge-visualization")
        expect(page.locator(".preview-viewport iframe.preview-iframe-light")).to_be_visible(
            timeout=5_000
        )
        page.close()


# ---------------------------------------------------------------------------
# Test 3: logbook/print inject buttons present post-bridge-kill (5.7 gap)
# ---------------------------------------------------------------------------


def test_logbook_and_print_buttons_present_after_bridge_kill(
    tmp_path, monkeypatch, chromium_browser
):
    """The preview's action bar still carries the logbook + print buttons.

    5.7's own validation gate only greps for the dead window bridge --
    it never confirms logbook.js/print.js's `injectLogbookButtons()`/
    `injectPrintButton()` (now called directly from preview.js as real
    imports) still land their buttons in the live DOM. This is that check.
    """
    with _launch_artifacts(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page(viewport=VIEWPORT)
        page.goto(base_url, wait_until="domcontentloaded")

        _card_for_title(page, MARKDOWN_TITLE).click()
        expect(page.locator(".preview-header-title")).to_have_text(MARKDOWN_TITLE, timeout=10_000)

        actions_bar = page.locator(".preview-header-actions")
        expect(actions_bar.locator(".logbook-action-btn")).to_be_visible(timeout=5_000)
        expect(actions_bar.locator(".print-action-btn")).to_be_visible(timeout=5_000)
        page.close()


# ---------------------------------------------------------------------------
# Test 4: Simple mode renders the same viewports as Expert
# ---------------------------------------------------------------------------


def _simple_row(page: Page, title: str):
    """Locate a row in Simple mode's "Results from this session" list."""
    return page.locator(".simple-list-item", has_text=title)


def test_simple_mode_renders_the_same_viewports_as_expert(tmp_path, monkeypatch, chromium_browser):
    """Every seeded type renders in Simple's result card, not just the
    iframe/img ones.

    Simple mode used to build its card preview from types.js's
    `thumbnailHtml()` -- the sidebar *card* builder, which knows only the
    <img>/<iframe> types and drops markdown, JSON, text, PDF and timeseries
    to a summary dump or a type-icon placeholder. Both surfaces now share
    artifact-viewport.js's single dispatch, so this asserts the two types
    that shortfall actually swallowed (markdown, JSON) reach their real
    renderers here, down to the rendered output rather than just the
    container: an empty `.md-preview-container` would satisfy a
    presence-only check while still showing the user nothing.
    """
    with _launch_artifacts(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page(viewport=VIEWPORT)
        page.goto(f"{base_url}?mode=simple", wait_until="domcontentloaded")

        preview = page.locator("#simple-result-preview")
        expect(_simple_row(page, MARKDOWN_TITLE)).to_be_visible(timeout=10_000)

        # Markdown: the marked/KaTeX pipeline ran, producing real headings.
        _simple_row(page, MARKDOWN_TITLE).click()
        expect(page.locator("#simple-result-title")).to_have_text(MARKDOWN_TITLE, timeout=5_000)
        expect(preview.locator(".osprey-md-rendered h1")).to_have_text(
            "Beam Current", timeout=5_000
        )

        # JSON: the recursive viewer ran, producing keyed rows.
        _simple_row(page, JSON_TITLE).click()
        expect(page.locator("#simple-result-title")).to_have_text(JSON_TITLE, timeout=5_000)
        expect(preview.locator(".json-viewer .json-key").first).to_have_text(
            '"correctors"', timeout=5_000
        )

        # HTML: the type that already worked, still working.
        _simple_row(page, HTML_TITLE).click()
        expect(page.locator("#simple-result-title")).to_have_text(HTML_TITLE, timeout=5_000)
        expect(preview.locator("iframe.preview-iframe-light")).to_be_visible(timeout=5_000)

        # Nothing anywhere in the card fell back to the sidebar-thumbnail
        # placeholder the old builder produced for unrenderable types.
        expect(preview.locator(".thumb-placeholder, .thumb-summary")).to_have_count(0)
        page.close()


# ---------------------------------------------------------------------------
# Test 5: an embedded artifact page inherits the gallery's theme
# ---------------------------------------------------------------------------


def test_preview_iframe_inherits_gallery_theme(tmp_path, monkeypatch, chromium_browser):
    """A freshly mounted preview boots in the gallery's theme, follows a hub
    theme change, and the "Open in new tab" link is re-stamped with it.

    The served page resolves its own theme through theme-boot.js (whose top
    rung is ``?theme=``) rather than reading the parent gallery's
    ``data-theme`` the way the old in-page bridge did -- so the gallery has
    to carry its theme onto the iframe src and re-send it on load and on
    every change. Drives the whole chain: a gallery opened in a non-main
    theme, a preview mounted afterwards, then a hub-style broadcast.
    """
    with _launch_artifacts(tmp_path, monkeypatch) as base_url:
        ctx = chromium_browser.new_context(viewport=VIEWPORT, color_scheme="dark")
        page = ctx.new_page()
        page.goto(f"{base_url}/?theme=retro-light", wait_until="domcontentloaded")
        expect(page.locator("html")).to_have_attribute("data-theme", "retro-light")

        _card_for_title(page, HTML_TITLE).click()
        iframe = page.locator(".preview-viewport iframe.preview-iframe-light")
        expect(iframe).to_be_visible(timeout=5_000)
        # Load-time half: the src carries the theme in force, and the page
        # inside painted in it (not the OS-dark it would resolve on its own).
        assert "theme=retro-light" in (iframe.get_attribute("src") or "")
        page.wait_for_function(
            "() => document.querySelector('.preview-viewport iframe')"
            "  ?.contentDocument?.documentElement.getAttribute('data-theme') === 'retro-light'",
            timeout=10_000,
        )
        link = page.locator(".preview-header a[data-theme-link]")
        assert "theme=retro-light" in (link.get_attribute("href") or "")

        # Live half: a hub broadcast re-themes the gallery, the embedded page,
        # and the new-tab link -- to an id outside the main family.
        page.evaluate(
            "() => window.postMessage({type: 'osprey-theme-change',"
            " theme: 'high-contrast-light'}, window.location.origin)"
        )
        expect(page.locator("html")).to_have_attribute("data-theme", "high-contrast-light")
        page.wait_for_function(
            "() => document.querySelector('.preview-viewport iframe')"
            "  ?.contentDocument?.documentElement.getAttribute('data-theme')"
            "  === 'high-contrast-light'",
            timeout=10_000,
        )
        expect(link).to_have_attribute("href", re.compile(r"theme=high-contrast-light"))
        page.close()
