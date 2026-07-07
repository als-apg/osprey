"""Browser Playwright suite pinning the scaffold gallery's detail-view interaction contract.

The acceptance pin for the scaffold gallery's module decomposition (the
scaffold utils/data/view/detail/edit modules plus the slim entry) and the
config-renderers siblings (settings-editor.js, mcp-renderer.js). Real
behavior crosses module boundaries here (detail.js's shell calling back
into edit.js/edit-form.js as `gallery.<method>()`, detail-content.js
dispatching to the sibling settings/mcp renderers) -- this suite is the one
place that drives the whole chain through a real browser rather than a fake
`gallery` host object, so a wiring mistake anywhere in the module graph (a
renamed method, a dropped import, a delegator that silently stopped
forwarding) shows up as an actual interaction failure here even if every
module's own Vitest suite still passes in isolation.

Covers:
  - opening an artifact's detail view from a gallery card
  - preview -> edit switching, including the framework-artifact "claim on
    first edit" flow (handleEditFramework's confirm + POST /claim)
  - the dirty-guard veto contract: an in-progress edit blocks the settings
    drawer's close (same composite `registerUnsavedGuard` check test_osprey_
    drawer.py's guard tests exercise with a synthetic guard -- this suite
    drives the REAL scaffold-gallery editDirty flag through an actual edit)
  - Discard restores clean state, verified against a server refetch (not
    just a client-side flag flip), and closing then succeeds without a
    dialog at all
  - the settings.json and mcp.json artifacts render through their split
    renderer modules (settings-editor.js, mcp-renderer.js) in Preview mode

Unlike test_osprey_drawer.py's launcher (an empty `tmp_path` -- fine there,
since those tests only assert that a `/api/scaffold` request fires), this
suite needs real on-disk artifacts: ScaffoldGalleryService.list_artifacts()
is filesystem-first and returns nothing for a project with no `.claude/`
tree at all. So `_launch_web_terminal` here first runs a real
`osprey build --preset hello-world` (matching the pattern already used by
the non-browser tests in test_scaffold_gallery_service.py /
test_scaffold_gallery_env_vars.py) into `tmp_path`, then serves web_terminal
with that directory as `project_cwd`.

Run:
    .venv/bin/pytest tests/interfaces/web_terminal/test_scaffold_detail.py -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from osprey.cli.build_cmd import build
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

# ---------------------------------------------------------------------------
# Selectors
# ---------------------------------------------------------------------------

# Same rationale as test_osprey_drawer.py: the settings trigger carries
# `data-drawer-trigger`, not the drawer component's own `[data-drawer]`
# marker, so the warning gate below is the sole open path.
TRIGGER_SELECTOR = '[data-drawer-trigger="settings-drawer"]'
DRAWER_SELECTOR = "#settings-drawer"
BACKDROP_SELECTOR = "#drawer-backdrop"
WARNING_PROCEED_SELECTOR = ".settings-warning-proceed"
WELCOME_OVERLAY_SELECTOR = "#welcome-overlay"

# rules/safety: a framework (never user-owned) artifact with front matter
# that has no `model` field, so Edit mode falls to the plain-text editor
# (not the front-matter form) -- the clean preview/edit contrast this suite
# needs: rendered markdown in Preview, a raw textarea in Edit.
SAFETY_CARD_SELECTOR = '.prompts-card[data-name="rules/safety"]'
SETTINGS_CARD_SELECTOR = '.prompts-card[data-name="settings-json"]'
MCP_CARD_SELECTOR = '.prompts-card[data-name="mcp-json"]'

CONFIG_TAB_SELECTOR = '.drawer-tab[data-tab="tab-config"]'

VIEWPORT = {"width": 1280, "height": 800}


# ---------------------------------------------------------------------------
# Live server launcher
# ---------------------------------------------------------------------------


@contextmanager
def _launch_web_terminal(tmp_path, monkeypatch) -> Iterator[str]:
    """Build a real hello-world project, then serve web_terminal against it."""
    runner = CliRunner()
    result = runner.invoke(
        build,
        [
            "scaffold-detail-test",
            "--preset",
            "hello-world",
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    project_dir = tmp_path / "scaffold-detail-test"

    monkeypatch.chdir(project_dir)
    with (
        patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(project_dir)},
        ),
        patch(
            "osprey.interfaces.web_terminal.app._load_panel_config",
            return_value=({"artifacts"}, [], None),
        ),
        patch(
            "osprey.interfaces.web_terminal.app._launch_artifact_server",
            side_effect=lambda a: setattr(a.state, "artifact_server_url", "http://127.0.0.1:8086"),
        ),
    ):
        from osprey.interfaces.web_terminal.app import create_app

        app = create_app(shell_command=["echo", "hello"])
        with _run_app_server(app) as base_url:
            yield base_url


def _goto(page: Page, base_url: str) -> None:
    """Navigate to the hub and dismiss the first-visit welcome banner.

    See test_osprey_drawer.py's `_goto` docstring for the retry rationale
    (the welcome modal's Enter-to-dismiss listener attaches only after its
    own `/health` fetch resolves, so a single blind keypress can race it).
    """
    page.goto(base_url, wait_until="domcontentloaded")
    overlay = page.locator(WELCOME_OVERLAY_SELECTOR)
    if overlay.count() == 0:
        return
    for _ in range(50):
        if overlay.count() == 0:
            break
        page.keyboard.press("Enter")
        page.wait_for_timeout(100)
    expect(overlay).to_have_count(0, timeout=5_000)


def _open_settings_drawer(page: Page) -> None:
    """Click the header trigger, proceeding past the first-time warning if shown."""
    page.click(TRIGGER_SELECTOR)
    proceed = page.locator(WARNING_PROCEED_SELECTOR)
    try:
        expect(proceed).to_be_visible(timeout=2_000)
        proceed.click()
    except AssertionError:
        pass  # already acked this server session -- drawer opened directly
    drawer = page.locator(DRAWER_SELECTOR)
    expect(drawer).to_have_attribute("open", "", timeout=5_000)
    expect(drawer).to_have_css("transform", "matrix(1, 0, 0, 1, 0, 0)", timeout=2_000)


def _once_dialog(page: Page, *, accept: bool) -> None:
    """Arm a one-shot handler for the next native `confirm()`/`alert()` dialog.

    scaffold-gallery's write actions (handleEditFramework, the drawer's
    composite unsaved-changes guard) all use the browser's real `confirm()`,
    not a hand-rolled DOM overlay like the settings first-time warning --
    Playwright's dialog API is the correct tool here, not a
    `window.confirm` stub (which would also shadow the guard's own call).
    """
    page.once("dialog", lambda dialog: dialog.accept() if accept else dialog.dismiss())


# ---------------------------------------------------------------------------
# Test 1: open detail from the gallery, showing the rendered preview
# ---------------------------------------------------------------------------


def test_open_detail_from_gallery_shows_preview(tmp_path, monkeypatch, chromium_browser):
    """Clicking a card opens its detail view in Preview mode with rendered markdown."""
    with _launch_web_terminal(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page(viewport=VIEWPORT)
        _goto(page, base_url)

        _open_settings_drawer(page)

        card = page.locator(SAFETY_CARD_SELECTOR)
        expect(card).to_be_visible(timeout=5_000)
        card.click()

        expect(page.locator(".prompts-detail-name")).to_have_text("rules/safety")
        expect(page.locator(".prompts-detail-header .prompts-badge")).to_have_class(
            "prompts-badge framework"
        )
        expect(page.locator(".prompts-mode-btn.active")).to_have_text("Preview")

        preview = page.locator(".osprey-md-rendered")
        expect(preview).to_be_visible(timeout=5_000)
        expect(preview).to_contain_text("Tool Confinement")
        page.close()


# ---------------------------------------------------------------------------
# Test 2/3/4: preview -> edit switch, dirty guard blocks close, discard
# restores clean state, then close succeeds
# ---------------------------------------------------------------------------


def test_edit_dirty_guard_blocks_close_and_discard_restores_clean(
    tmp_path, monkeypatch, chromium_browser
):
    """The full write-side interaction chain on a framework artifact.

    Switching to Edit on a still-framework artifact claims it first
    (handleEditFramework's confirm + POST /claim) -- this pins that flow
    end-to-end, not just the dirty-flag bookkeeping scaffold-edit.test.mjs's
    fake-host unit tests already cover.
    """
    with _launch_web_terminal(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page(viewport=VIEWPORT)
        _goto(page, base_url)

        _open_settings_drawer(page)
        page.locator(SAFETY_CARD_SELECTOR).click()
        expect(page.locator(".osprey-md-rendered")).to_be_visible(timeout=5_000)

        # Preview -> Edit: still framework-owned, so this claims the file
        # first (a real confirm() dialog + POST /claim), then switches mode.
        _once_dialog(page, accept=True)
        page.locator(".prompts-mode-btn", has_text="Edit").click()

        expect(page.locator(".prompts-detail-header .prompts-badge")).to_have_class(
            "prompts-badge user-owned", timeout=5_000
        )
        textarea = page.locator(".prompts-edit-textarea")
        expect(textarea).to_be_visible(timeout=5_000)
        original_value = textarea.input_value()
        assert "Tool Confinement" in original_value, (
            "expected the claimed file's original content in the edit textarea"
        )

        # Dirty an edit -- Discard/Save appear and enable.
        textarea.fill(original_value + "\nEDITED-BY-SCAFFOLD-DETAIL-PIN-TEST\n")
        discard_btn = page.locator(".prompts-discard-btn")
        save_btn = page.locator(".prompts-save-btn")
        expect(discard_btn).to_be_enabled(timeout=2_000)
        expect(save_btn).to_be_enabled(timeout=2_000)

        # The dirty guard vetoes a drawer close via the backdrop -- same
        # composite `registerUnsavedGuard` contract test_osprey_drawer.py
        # pins with a synthetic guard, driven here by a real edit.
        _once_dialog(page, accept=False)
        page.locator(BACKDROP_SELECTOR).click(position={"x": 10, "y": 10})
        expect(page.locator(DRAWER_SELECTOR)).to_have_attribute("open", "", timeout=2_000)

        # Discard restores clean state: back to Preview, and (crucially) the
        # content comes from a server refetch of the on-disk file, which was
        # never actually written -- proving the edit was truly discarded,
        # not just hidden client-side.
        discard_btn.click()
        expect(page.locator(".prompts-mode-btn.active")).to_have_text("Preview", timeout=5_000)
        expect(page.locator(".prompts-discard-btn")).to_have_count(0)
        preview = page.locator(".osprey-md-rendered")
        expect(preview).to_be_visible(timeout=5_000)
        expect(preview).to_contain_text("Tool Confinement")
        expect(preview).not_to_contain_text("EDITED-BY-SCAFFOLD-DETAIL-PIN-TEST")

        # Clean now -- closing succeeds with no dialog at all (the guard
        # returns true immediately without prompting).
        page.locator(BACKDROP_SELECTOR).click(position={"x": 10, "y": 10})
        expect(page.locator(DRAWER_SELECTOR)).not_to_have_attribute("open", "", timeout=5_000)
        page.close()


# ---------------------------------------------------------------------------
# Test 5: settings.json renders through the split settings-editor.js
# ---------------------------------------------------------------------------


def test_settings_json_renders_through_split_settings_editor(
    tmp_path, monkeypatch, chromium_browser
):
    """settings-json's Preview mode mounts settings-editor.js's interactive editor."""
    with _launch_web_terminal(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page(viewport=VIEWPORT)
        _goto(page, base_url)

        _open_settings_drawer(page)
        page.click(CONFIG_TAB_SELECTOR)

        card = page.locator(SETTINGS_CARD_SELECTOR)
        expect(card).to_be_visible(timeout=5_000)
        card.click()

        editor = page.locator(".config-structured-view.config-editor")
        expect(editor).to_be_visible(timeout=5_000)
        expect(page.locator(".config-edit-input")).to_have_count(1)

        for level in ("allow", "ask", "deny"):
            col = page.locator(f".config-perm-col.config-perm-{level}")
            expect(col).to_be_visible()
            expect(col.locator(".config-perm-header")).to_have_text(level.upper())
        page.close()


# ---------------------------------------------------------------------------
# Test 6: mcp.json renders through the split mcp-renderer.js
# ---------------------------------------------------------------------------


def test_mcp_json_renders_through_split_mcp_renderer(tmp_path, monkeypatch, chromium_browser):
    """mcp-json's Preview mode mounts mcp-renderer.js's server-card grid."""
    with _launch_web_terminal(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page(viewport=VIEWPORT)
        _goto(page, base_url)

        _open_settings_drawer(page)
        page.click(CONFIG_TAB_SELECTOR)

        card = page.locator(MCP_CARD_SELECTOR)
        expect(card).to_be_visible(timeout=5_000)
        card.click()

        grid = page.locator(".config-mcp-grid")
        expect(grid).to_be_visible(timeout=5_000)
        cards = page.locator(".config-mcp-card")
        expect(cards).not_to_have_count(0)
        # "controls" is the core control-system MCP server every OSPREY
        # build registers -- a stable anchor independent of preset/registry
        # churn in the rest of the server list.
        expect(page.locator(".config-mcp-card-name", has_text="controls")).to_be_visible()
        page.close()
