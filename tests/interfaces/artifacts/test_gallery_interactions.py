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
  - filter chips narrow the sidebar to artifacts of the selected type
  - selecting an artifact of a given type renders that type's preview
    header (title, type badge) and viewport (markdown -> #md-viewport,
    html -> a sandboxed iframe)
  - the logbook/print inject buttons are present in the preview's action
    bar -- proving the direct-import wiring actually renders them (a
    grep-level check that no window bridge exists can't show that)

Unlike test_load_smokes.py's `_launch_artifacts` (an empty tmp_path -- fine
there, since it only checks the page loads clean), this suite needs real
on-disk artifacts: ArtifactStore.list_entries() reflects whatever the
`artifacts.json` index says, and the gallery has nothing to filter/select
without it. `_launch_artifacts` here first seeds two fixture artifacts
(markdown + html) via a throwaway ArtifactStore pointed at the same
`workspace_root` the app will use -- BaseStore.__init__ loads the index
from disk, so the app's own store picks up the seeded entries on
construction (mirrors the `_make_artifact` helper already used in
test_logbook.py for the non-browser artifact tests).

Run:
    .venv/bin/pytest tests/interfaces/artifacts/test_gallery_interactions.py -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

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


# ---------------------------------------------------------------------------
# Live server launcher
# ---------------------------------------------------------------------------


@contextmanager
def _launch_artifacts(tmp_path, monkeypatch) -> Iterator[str]:
    """Seed two fixture artifacts (markdown + html), then serve the gallery.

    Mirrors test_logbook.py's `_make_artifact` helper: a throwaway
    ArtifactStore writes the index + content files, then create_app()'s own
    ArtifactStore (same workspace_root) loads that index at construction.

    Each fixture sets a real `category` (from type_registry.py's CATEGORIES,
    not ARTIFACT_TYPES) -- render.js's filter chips group by
    `category || artifact_type`, but the type registry always returns a
    non-empty `categories` map, which wins that `||` unconditionally. An
    artifact saved without a category (as real create_document/
    create_static_plot MCP tools never do) never gets a filter chip at all,
    since its bare artifact_type never matches any CATEGORIES key. Viewport
    selection in preview.js is unaffected -- that dispatch switches on the
    raw `artifact_type`, not `category`.
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
# Test 1: filter chips narrow the sidebar to the selected type
# ---------------------------------------------------------------------------


def test_filter_chips_narrow_sidebar_by_type(tmp_path, monkeypatch, chromium_browser):
    """Clicking a type chip hides artifacts of other types and marks it active;
    the static "All" chip (index.html:89, active by default) restores both.
    """
    with _launch_artifacts(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page(viewport=VIEWPORT)
        page.goto(base_url, wait_until="domcontentloaded")

        markdown_card = _card_for_title(page, MARKDOWN_TITLE)
        html_card = _card_for_title(page, HTML_TITLE)
        expect(markdown_card).to_be_visible(timeout=10_000)
        expect(html_card).to_be_visible(timeout=10_000)

        all_chip = page.locator('.filter-chip[data-filter="all"]')
        markdown_chip = page.locator('.filter-chip.type-chip[data-filter="document"]')
        html_chip = page.locator('.filter-chip.type-chip[data-filter="visualization"]')
        expect(all_chip).to_have_class("filter-chip active")
        expect(markdown_chip).to_be_visible(timeout=5_000)
        expect(html_chip).to_be_visible(timeout=5_000)

        markdown_chip.click()
        expect(markdown_chip).to_have_class("filter-chip type-chip active")
        expect(all_chip).not_to_have_class("filter-chip active")
        expect(markdown_card).to_be_visible()
        expect(html_card).to_have_count(0)

        html_chip.click()
        expect(html_chip).to_have_class("filter-chip type-chip active")
        expect(markdown_chip).not_to_have_class("filter-chip type-chip active")
        expect(html_card).to_be_visible()
        expect(markdown_card).to_have_count(0)

        # "All" restores both.
        all_chip.click()
        expect(all_chip).to_have_class("filter-chip active")
        expect(html_chip).not_to_have_class("filter-chip type-chip active")
        expect(markdown_card).to_be_visible()
        expect(html_card).to_be_visible()
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
        expect(page.locator("#md-viewport")).to_be_visible(timeout=5_000)

        _card_for_title(page, HTML_TITLE).click()
        expect(page.locator(".preview-header-title")).to_have_text(HTML_TITLE, timeout=10_000)
        expect(page.locator(".preview-header .badge")).to_have_class("badge badge-visualization")
        expect(page.locator(".preview-viewport iframe.preview-iframe-light")).to_be_visible(timeout=5_000)
        page.close()


# ---------------------------------------------------------------------------
# Test 3: logbook/print inject buttons present post-bridge-kill (5.7 gap)
# ---------------------------------------------------------------------------


def test_logbook_and_print_buttons_present_after_bridge_kill(tmp_path, monkeypatch, chromium_browser):
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
