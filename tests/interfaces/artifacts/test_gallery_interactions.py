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
  - the logbook/print inject buttons are present in the preview's action
    bar -- proving the direct-import wiring actually renders them (a
    grep-level check that no window bridge exists can't show that)
  - embedded (``?embedded=true``), the page is a viewer of one artifact:
    no browser column and no preview header, the artifact the hub selects
    (an ``osprey-artifact-select`` message) renders through the same
    dispatch for every seeded type, and the hub's
    ``osprey-artifact-action`` messages run the gallery's own pin, logbook
    and delete paths

Unlike _launchers.py's `launch_artifacts` (an empty tmp_path -- fine
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

from osprey.interfaces._serving import run_app_server

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
# Short enough that one page of rows overflows the sidebar by more than the
# load-ahead margin, so the first page stands alone until the list is scrolled.
PAGING_VIEWPORT = {"width": 1280, "height": 360}

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
    # A json fixture: the shape a channel-finder result lands as.
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
    with run_app_server(app) as base_url:
        yield base_url


@contextmanager
def _launch_many_artifacts(
    tmp_path, monkeypatch, count: int, page_size: int | None = None
) -> Iterator[str]:
    """Seed ``count`` markdown artifacts with distinct titles, then serve the gallery.

    Titles are ``Paged Artifact NNN`` in save order, so ``000`` is the oldest
    and ``count - 1`` the newest; the zero padding keeps one title from being
    a substring of another. ``page_size`` overrides the listing's page size.
    """
    monkeypatch.chdir(tmp_path)
    if page_size is not None:
        monkeypatch.setattr("osprey.interfaces.artifacts.app._page_size", lambda: page_size)

    from osprey.stores.artifact_store import ArtifactStore

    seed_store = ArtifactStore(workspace_root=tmp_path)
    for i in range(count):
        seed_store.save_file(
            file_content=f"# Paged {i}\n".encode(),
            filename=f"paged-{i:03d}.md",
            artifact_type="markdown",
            title=f"Paged Artifact {i:03d}",
            mime_type="text/markdown",
            tool_source="test_fixture",
            category="document",
        )

    from osprey.interfaces.artifacts.app import create_app

    app = create_app(workspace_root=tmp_path)
    with run_app_server(app) as base_url:
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
# Test 1a: the sidebar fetches the next page as it is scrolled
# ---------------------------------------------------------------------------


def test_scrolling_the_sidebar_loads_the_next_page(tmp_path, monkeypatch, chromium_browser):
    """The gallery opens on the newest page; scrolling the list to its end
    fetches the older artifacts the first page left out.
    """
    with _launch_many_artifacts(tmp_path, monkeypatch, 25) as base_url:
        page = chromium_browser.new_page(viewport=PAGING_VIEWPORT)
        page.goto(base_url, wait_until="domcontentloaded")

        newest = _card_for_title(page, "Paged Artifact 024")
        oldest = _card_for_title(page, "Paged Artifact 000")
        expect(newest).to_be_visible(timeout=10_000)
        expect(page.locator(".tree-item")).to_have_count(20, timeout=5_000)
        expect(oldest).to_have_count(0)

        page.locator("#sidebar-body").evaluate("el => { el.scrollTop = el.scrollHeight; }")

        expect(oldest).to_have_count(1, timeout=5_000)
        expect(page.locator(".tree-item")).to_have_count(25, timeout=5_000)
        page.close()


def test_a_sidebar_taller_than_one_page_fetches_the_next_one(
    tmp_path, monkeypatch, chromium_browser
):
    """When the first page does not fill the sidebar there is nothing to
    scroll, so the gallery fetches on until the list reaches past its end.
    """
    with _launch_many_artifacts(tmp_path, monkeypatch, 25) as base_url:
        page = chromium_browser.new_page(viewport=VIEWPORT)
        page.goto(base_url, wait_until="domcontentloaded")

        expect(_card_for_title(page, "Paged Artifact 000")).to_have_count(1, timeout=10_000)
        expect(page.locator(".tree-item")).to_have_count(25, timeout=5_000)
        page.close()


def test_a_short_page_size_keeps_fetching_until_the_sidebar_overflows(
    tmp_path, monkeypatch, chromium_browser
):
    """With a page far shorter than the sidebar, the gallery fetches page
    after page, not just one more, until the list reaches past its end.
    """
    with _launch_many_artifacts(tmp_path, monkeypatch, 25, page_size=3) as base_url:
        page = chromium_browser.new_page(viewport=VIEWPORT)
        page.goto(base_url, wait_until="domcontentloaded")

        expect(_card_for_title(page, "Paged Artifact 024")).to_be_visible(timeout=10_000)
        page.wait_for_function(
            "document.querySelectorAll('.tree-item').length >= 9", timeout=10_000
        )
        page.close()


def test_escape_in_the_filter_restores_the_unfiltered_list(tmp_path, monkeypatch, chromium_browser):
    """Escape in the filter box clears it and the list shows the newest page
    again, not only the artifacts that matched.
    """
    with _launch_many_artifacts(tmp_path, monkeypatch, 25) as base_url:
        page = chromium_browser.new_page(viewport=PAGING_VIEWPORT)
        page.goto(base_url, wait_until="domcontentloaded")

        expect(page.locator(".tree-item")).to_have_count(20, timeout=10_000)
        search = page.locator("#search")
        search.fill("Paged Artifact 000")
        expect(page.locator(".tree-item")).to_have_count(1, timeout=5_000)

        search.press("Escape")

        expect(search).to_have_value("")
        expect(_card_for_title(page, "Paged Artifact 024")).to_have_count(1, timeout=5_000)
        expect(page.locator(".tree-item")).to_have_count(20, timeout=5_000)
        page.close()


def test_a_focus_on_an_unloaded_artifact_selects_it_while_a_search_is_active(
    tmp_path, monkeypatch, chromium_browser
):
    """An agent focus on an artifact outside the loaded pages opens it in the
    preview even when the filter text keeps it out of the list.
    """
    with _launch_many_artifacts(tmp_path, monkeypatch, 25) as base_url:
        page = chromium_browser.new_page(viewport=PAGING_VIEWPORT)
        page.goto(base_url, wait_until="domcontentloaded")

        expect(page.locator(".tree-item")).to_have_count(20, timeout=10_000)
        page.locator("#search").fill("Paged Artifact 024")
        expect(page.locator(".tree-item")).to_have_count(1, timeout=5_000)

        # Sent from the page itself, so the request is same-origin.
        status = page.evaluate(
            """async () => {
                const listing = await fetch('/api/artifacts?search=Paged%20Artifact%20000');
                const id = (await listing.json()).artifacts[0].id;
                const resp = await fetch('/api/focus', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ artifact_id: id }),
                });
                return resp.status;
            }"""
        )
        assert status == 200

        expect(page.locator(".preview-header-title")).to_have_text(
            "Paged Artifact 000", timeout=10_000
        )
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
# Test 4: embedded, the page is the hub's viewer of one artifact
# ---------------------------------------------------------------------------


def _select_in_viewer(page: Page, title: str) -> str:
    """Show the artifact titled *title* in the embedded viewer, as the hub does.

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


def test_the_embedded_viewer_shows_what_the_hub_selects(tmp_path, monkeypatch, chromium_browser):
    """Embedded, the page shows one artifact: no browser column and no
    preview header (the hub renders both), and every seeded type the hub
    selects renders through the shared viewport dispatch.

    Markdown and JSON are asserted down to their rendered output rather than
    their containers: an empty ``.md-preview-container`` would satisfy a
    presence-only check while showing the operator nothing.
    """
    with _launch_artifacts(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page(viewport=VIEWPORT)
        page.goto(f"{base_url}?embedded=true", wait_until="domcontentloaded")

        expect(page.locator("#browse-sidebar")).to_be_hidden()
        expect(page.locator("#resize-handle")).to_be_hidden()
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

        # Below 560 px the browse layout shows its y-axis splitter by id; the
        # viewer's rule still keeps it hidden.
        page.set_viewport_size({"width": 480, "height": 800})
        expect(page.locator("#resize-handle-y")).to_be_hidden()
        page.close()


def test_the_embedded_viewer_runs_the_actions_the_hub_sends(
    tmp_path, monkeypatch, chromium_browser
):
    """The hub's tile menu reaches the gallery's own code paths through
    ``osprey-artifact-action``: pin sets the flag on the stored entry, logbook
    opens the composer, and delete, once confirmed, removes the artifact and
    empties the viewer.
    """
    from osprey.stores.artifact_store import ArtifactStore

    with _launch_artifacts(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page(viewport=VIEWPORT)
        page.goto(f"{base_url}?embedded=true", wait_until="domcontentloaded")
        artifact_id = _select_in_viewer(page, HTML_TITLE)
        expect(page.locator(".preview-viewport iframe")).to_be_visible(timeout=10_000)

        with page.expect_response(
            lambda r: r.url.endswith(f"/api/artifacts/{artifact_id}/pin")
        ) as pinned:
            _act_in_viewer(page, artifact_id, "pin")
        assert pinned.value.ok
        entry = ArtifactStore(workspace_root=tmp_path).get_entry(artifact_id)
        assert entry is not None and entry.pinned is True

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

    The gallery is loaded ``?embedded=true``, as the hub loads it: only a
    follower applies the broadcast at all (standalone, the gallery runs
    theme-manager.js in the hub role behind its own display menu).
    """
    with _launch_artifacts(tmp_path, monkeypatch) as base_url:
        ctx = chromium_browser.new_context(viewport=VIEWPORT, color_scheme="dark")
        page = ctx.new_page()
        page.goto(f"{base_url}/?theme=retro-light&embedded=true", wait_until="domcontentloaded")
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


def test_the_open_in_new_tab_link_carries_and_follows_the_theme(
    tmp_path, monkeypatch, chromium_browser
):
    """The standalone preview header's "Open in new tab" link carries the
    theme in force, and a theme change re-stamps it in place.

    ``?theme=`` is the top rung of theme-boot.js's ladder, so a link minted
    before the change would otherwise open the new tab in the stale theme.
    Standalone, the gallery runs theme-manager.js in the hub role, so the
    change goes through the module instance the page runs, as a pick in its
    own display menu does.
    """
    with _launch_artifacts(tmp_path, monkeypatch) as base_url:
        ctx = chromium_browser.new_context(viewport=VIEWPORT, color_scheme="dark")
        page = ctx.new_page()
        page.goto(f"{base_url}/?theme=retro-light", wait_until="domcontentloaded")
        expect(page.locator("html")).to_have_attribute("data-theme", "retro-light")

        _card_for_title(page, HTML_TITLE).click()
        link = page.locator(".preview-header a[data-theme-link]")
        expect(link).to_have_attribute("href", re.compile(r"theme=retro-light"), timeout=10_000)

        page.evaluate(
            "async (theme) => (await import('/design-system/js/theme-manager.js')).setTheme(theme)",
            "high-contrast-light",
        )
        expect(page.locator("html")).to_have_attribute("data-theme", "high-contrast-light")
        expect(link).to_have_attribute("href", re.compile(r"theme=high-contrast-light"))
        page.close()


# ---------------------------------------------------------------------------
# Picking several artifacts and deleting them together
# ---------------------------------------------------------------------------


def test_shift_click_picks_a_run_and_deletes_it_together(tmp_path, monkeypatch, chromium_browser):
    """Click the first row, Shift-click the third: three rows are picked and
    the bar above the list says so. Delete, once confirmed, removes those
    three from the store and the list and leaves the fourth.
    """
    with _launch_many_artifacts(tmp_path, monkeypatch, 4) as base_url:
        page = chromium_browser.new_page(viewport=VIEWPORT)
        page.goto(base_url, wait_until="domcontentloaded")

        rows = page.locator(".tree-item")
        expect(rows).to_have_count(4, timeout=10_000)
        # Newest first: 003, 002, 001, 000.
        _card_for_title(page, "Paged Artifact 003").click()
        _card_for_title(page, "Paged Artifact 001").click(modifiers=["Shift"])

        expect(page.locator(".tree-item.picked")).to_have_count(3, timeout=5_000)
        bar = page.locator("#pick-bar")
        expect(bar).to_be_visible()
        expect(page.locator("#pick-count")).to_have_text("3 selected")

        messages: list[str] = []

        def accept(dialog):
            messages.append(dialog.message)
            dialog.accept()

        page.once("dialog", accept)
        page.locator("#pick-delete").click()

        expect(rows).to_have_count(1, timeout=5_000)
        expect(_card_for_title(page, "Paged Artifact 000")).to_have_count(1)
        expect(bar).to_be_hidden()
        assert messages == ["Delete 3 artifacts? This cannot be undone."]

        from osprey.stores.artifact_store import ArtifactStore

        titles = [e.title for e in ArtifactStore(workspace_root=tmp_path).list_entries()]
        assert titles == ["Paged Artifact 000"]
        page.close()
