"""Browser Playwright suite pinning the artifacts viewer's interaction contract.

The acceptance pin for the artifacts module cluster (the state/types/preview/
timeseries modules plus the slim gallery.js entry) and for logbook.js/print.js
being real ES modules. This suite drives the whole chain through a real browser
against real fixture artifacts on disk, rather than the fake-host unit tests each
module's own Vitest suite uses -- so a wiring mistake anywhere in the module graph
shows up as an actual interaction failure here.

The page is a viewer of one artifact, framed by ``tests/interfaces/_panel_host.py``:
the artifact the hub selects (an ``osprey-artifact-select`` message) renders
through the same dispatch for every seeded type, and the hub's
``osprey-artifact-action`` messages run the gallery's own logbook and delete paths.

Unlike _launchers.py's `launch_artifacts` (an empty tmp_path -- fine
there, since it only checks the page loads clean), this suite needs real
on-disk artifacts: ArtifactStore.list_entries() reflects whatever the
`artifacts.json` index says, and the viewer has nothing to select without it. `_launch_artifacts` here first seeds three fixture artifacts
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

from contextlib import contextmanager
from typing import TYPE_CHECKING

import pytest

from osprey.interfaces._serving import run_app_server
from tests.interfaces._panel_host import open_panel_host

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

    The server derives each fixture's form from what it is (markdown and html
    are Documents, json is Data): render.js's tree sections group by that
    form, and the preview badge class reflects it. Viewport selection in
    preview.js is unaffected -- that dispatch switches on the raw
    `artifact_type`.
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
    )
    seed_store.save_file(
        file_content=b"<html><body><h1>Orbit</h1></body></html>",
        filename="orbit.html",
        artifact_type="html",
        title=HTML_TITLE,
        description="An html fixture artifact",
        mime_type="text/html",
        tool_source="test_fixture",
    )
    # A json fixture: the shape a channel-finder result lands as.
    seed_store.save_file(
        file_content=b'{"correctors": {"SR01C:HCM1:SP": 0.42}}',
        filename="correctors.json",
        artifact_type="json",
        title=JSON_TITLE,
        description="A json fixture artifact",
        mime_type="application/json",
        tool_source="test_fixture",
    )

    from osprey.interfaces.artifacts.app import create_app

    app = create_app(workspace_root=tmp_path)
    with run_app_server(app) as base_url:
        yield base_url


def _select_in_viewer(page: Page, title: str) -> str:
    """Show the artifact titled *title* in the viewer, as the hub does.

    The hub's sidebar owns the list, so the viewer is driven by the message
    the hub posts on a row click. It is sent from the page itself (same
    origin), with the id read from the listing, since ArtifactStore mints a
    fresh id every run. Returns that id.
    """
    return page.evaluate(
        """async (title) => {
            const listing = await (await fetch('/api/artifacts')).json();
            const entry = listing.artifacts.find((a) => a.title === title);
            window.postMessage(
                { type: 'osprey-artifact-select', artifact_id: entry.id },
                window.location.origin
            );
            return entry.id;
        }""",
        title,
    )


def _act_in_viewer(page: Page, artifact_id: str, action: str) -> None:
    """Send the hub's ``osprey-artifact-action`` message for *action*."""
    page.evaluate(
        """([id, action]) => window.postMessage(
            { type: 'osprey-artifact-action', artifact_id: id, action },
            window.location.origin
        )""",
        [artifact_id, action],
    )


def test_the_viewer_shows_what_the_hub_selects(tmp_path, monkeypatch, chromium_browser):
    """The page shows one artifact: no browser column and no preview header
    (the hub renders both), and every seeded type the hub selects renders
    through the shared viewport dispatch.

    Markdown and JSON are asserted down to their rendered output rather than
    their containers: an empty ``.md-preview-container`` would satisfy a
    presence-only check while showing the operator nothing.
    """
    with _launch_artifacts(tmp_path, monkeypatch) as base_url:
        page = open_panel_host(chromium_browser.new_page(viewport=VIEWPORT), base_url)

        expect(page.locator("#browse-sidebar")).to_have_count(0)
        expect(page.locator("#resize-handle")).to_have_count(0)
        expect(page.locator(".header")).to_have_count(0)
        expect(page.locator("#preview-empty")).to_be_visible()
        preview = page.locator("#preview-content")

        # Markdown: the marked/KaTeX pipeline ran, producing real headings.
        _select_in_viewer(page, MARKDOWN_TITLE)
        expect(preview.locator(".osprey-md-rendered h1")).to_have_text(
            "Beam Current", timeout=10_000
        )
        expect(preview.locator(".preview-desc")).to_have_text("A markdown fixture artifact")
        expect(page.locator(".preview-header")).to_have_count(0)

        # JSON: the recursive viewer ran, producing keyed rows.
        _select_in_viewer(page, JSON_TITLE)
        expect(preview.locator(".json-viewer .json-key").first).to_have_text(
            '"correctors"', timeout=5_000
        )

        # HTML: a sandboxed frame.
        _select_in_viewer(page, HTML_TITLE)
        expect(preview.locator(".preview-viewport iframe.preview-iframe-light")).to_be_visible(
            timeout=5_000
        )

        # A select of nothing returns the viewer to its empty state.
        page.evaluate(
            "window.postMessage({ type: 'osprey-artifact-select', artifact_id: null },"
            " window.location.origin)"
        )
        expect(page.locator("#preview-empty")).to_be_visible(timeout=5_000)
        expect(preview).to_be_hidden()
        page.close()


def test_the_viewer_runs_the_actions_the_hub_sends(tmp_path, monkeypatch, chromium_browser):
    """The hub's tile menu reaches the gallery's own code paths through
    ``osprey-artifact-action``: logbook opens the composer, and delete, once confirmed, removes the artifact and
    empties the viewer.
    """
    from osprey.stores.artifact_store import ArtifactStore

    with _launch_artifacts(tmp_path, monkeypatch) as base_url:
        page = open_panel_host(chromium_browser.new_page(viewport=VIEWPORT), base_url)
        artifact_id = _select_in_viewer(page, HTML_TITLE)
        expect(page.locator(".preview-viewport iframe")).to_be_visible(timeout=10_000)

        _act_in_viewer(page, artifact_id, "logbook")
        expect(page.locator(".logbook-overlay")).to_be_visible(timeout=5_000)

        messages: list[str] = []

        def accept(dialog):
            messages.append(dialog.message)
            dialog.accept()

        page.once("dialog", accept)
        _act_in_viewer(page, artifact_id, "delete")

        expect(page.locator("#preview-empty")).to_be_visible(timeout=5_000)
        assert messages == [f'Delete "{HTML_TITLE}"? This cannot be undone.']
        titles = [e.title for e in ArtifactStore(workspace_root=tmp_path).list_entries()]
        assert HTML_TITLE not in titles
        page.close()


# ---------------------------------------------------------------------------
# Test 5: a mounted artifact page inherits the gallery's theme
# ---------------------------------------------------------------------------


def test_the_viewer_preview_boots_in_and_follows_the_hub_theme(
    tmp_path, monkeypatch, chromium_browser
):
    """A preview the hub selects boots in the gallery's theme and follows a
    hub theme change.

    The served page resolves its own theme through theme-boot.js (whose top
    rung is ``?theme=``) rather than reading the parent gallery's
    ``data-theme`` the way the old in-page bridge did -- so the gallery has
    to carry its theme onto the iframe src and re-send it on load and on
    every change. Drives the whole chain: a gallery opened in a non-main
    theme, a preview mounted afterwards, then a hub-style broadcast.

    The preview runs in the content sandbox's opaque origin, so the gallery
    cannot read its document: the frame is inspected through Playwright's own
    frame access, and the live half rides the page's opaque-origin bridge.
    """
    with _launch_artifacts(tmp_path, monkeypatch) as base_url:
        ctx = chromium_browser.new_context(viewport=VIEWPORT, color_scheme="dark")
        page = open_panel_host(ctx.new_page(), f"{base_url}/?theme=retro-light")
        expect(page.locator("html")).to_have_attribute("data-theme", "retro-light")

        _select_in_viewer(page, HTML_TITLE)
        iframe = page.locator(".preview-viewport iframe.preview-iframe-light")
        expect(iframe).to_be_visible(timeout=5_000)
        # Load-time half: the src carries the theme in force, and the page
        # inside painted in it (not the OS-dark it would resolve on its own).
        assert "theme=retro-light" in (iframe.get_attribute("src") or "")
        # The preview frame is opaque-origin (sandboxed, no same-origin), so
        # the gallery cannot read its contentDocument; Playwright's frame
        # handle reaches into it the way the user's eyes do.
        frame_html = page.frame_locator(".preview-viewport iframe").locator("html")
        expect(frame_html).to_have_attribute("data-theme", "retro-light", timeout=10_000)

        # Live half: a hub broadcast re-themes the gallery and the embedded
        # page -- to an id outside the main family.
        page.evaluate(
            "() => window.postMessage({type: 'osprey-theme-change',"
            " theme: 'high-contrast-light'}, window.location.origin)"
        )
        expect(page.locator("html")).to_have_attribute("data-theme", "high-contrast-light")
        expect(frame_html).to_have_attribute("data-theme", "high-contrast-light", timeout=10_000)
        page.close()
