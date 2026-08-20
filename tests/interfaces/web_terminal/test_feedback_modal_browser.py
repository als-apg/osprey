"""Browser tests: the feedback dialog's keyboard, layering and Local round-trip.

Everything here is a claim only a real browser can settle. The focus trap needs
a layout engine (``focusableElements`` filters on ``offsetParent``, which is
null for everything under happy-dom). The Escape/Cmd-K arbitration between the
dialog, the command palette and ``<osprey-drawer>`` is three capture-phase
document listeners racing in registration order. The ``inert`` question is
about what the drawer's ``_syncBackgroundInert()`` pass did to a node that no
longer exists — the reason the dialog is destroyed on close instead of parked.
And the Local send is the only test in the tree that drives a report from a
keystroke in the textarea all the way to an ``fb-*.json`` on the server's disk.

Coverage:

  (1) Tab cycles through every control of the open dialog and wraps.
  (2) Cmd/Ctrl+K while the dialog is open does NOT open the palette — and the
      same key does open it once the dialog is gone (the control that keeps
      the negative honest).
  (3) Escape with the palette layered over the dialog closes only the palette.
  (4) A settings-drawer open/close cycle leaves no ``inert`` on a reopened
      dialog.
  (5) Local send: type, Send, and the record lands in the server's store.

Run:
    .venv/bin/pytest tests/interfaces/web_terminal/test_feedback_modal_browser.py -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

from tests.interfaces._panel_launch import publish_artifact_url
from tests.interfaces.conftest import _apply_all, _run_app_server

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

try:
    from playwright.sync_api import Browser, Page, expect

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False

pytestmark = [pytest.mark.browser, pytest.mark.slow]

#: Ceiling on the Tab walk. Comfortably above the dialog's control count, low
#: enough that a trap which lets focus escape (and therefore never returns to
#: where it started) fails instead of looping.
_MAX_TAB_STEPS = 20

#: A key stable enough to recognise one control again on the next lap. Radios
#: carry their value because they have no class of their own.
_KEY_OF = """
  (node) => {
    const cls = typeof node.className === 'string' ? node.className : '';
    return node.tagName.toLowerCase() +
      '|' + (node.type || '') +
      '|' + cls +
      (node.type === 'radio' ? '|' + node.value : '');
  }
"""

#: Reads the focused element: whether it is still inside the dialog, and its key.
_FOCUS_PROBE = f"""
() => {{
  const keyOf = {_KEY_OF};
  const dialog = document.querySelector('.feedback-modal-dialog');
  const active = document.activeElement;
  if (!active) return {{ inside: false, key: 'none' }};
  return {{ inside: dialog !== null && dialog.contains(active), key: keyOf(active) }};
}}
"""

#: Every control the browser will actually stop on, derived from the live
#: dialog rather than hardcoded, so the test states the contract ("Tab reaches
#: all of them") instead of a snapshot of today's form.
#:
#: Two filters make it the browser's list rather than the trap's. Disabled
#: controls are already excluded by the selector — which is how "Include
#: session context" legitimately drops out when the deployment has no session
#: to attach. And radios sharing a ``name`` are ONE tab stop: only the checked
#: member is reachable by Tab, the rest by arrow keys.
_EXPECTED_TAB_STOPS = f"""
() => {{
  const keyOf = {_KEY_OF};
  const dialog = document.querySelector('.feedback-modal-dialog');
  if (dialog === null) return [];
  const selector = [
    'a[href]',
    'button:not([disabled])',
    'textarea:not([disabled])',
    'input:not([disabled])',
    'select:not([disabled])',
    '[tabindex]:not([tabindex="-1"])',
    '[contenteditable="true"]',
  ].join(',');
  return Array.from(dialog.querySelectorAll(selector))
    .filter((node) => node.offsetParent !== null)
    .filter((node) => node.type !== 'radio' || node.checked)
    .map(keyOf);
}}
"""

#: Two animation frames, the deterministic alternative to sleeping before a
#: negative assertion. Both the palette and the dialog reveal themselves in a
#: ``requestAnimationFrame`` callback, so "it did not open" is only meaningful
#: once the frame it would have opened in has passed.
_SETTLE_TWO_FRAMES = (
    "() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)))"
)


@contextmanager
def _hub_server(workspace_dir: Path, data_root: Path) -> Iterator[str]:
    """Launch a real web-terminal hub with the feedback store under *data_root*.

    ``app.py``'s lifespan sites the store at ``resolve_shared_data_root() /
    "feedback"``, deliberately not on ``workspace_dir``. That resolver reads the
    deployment's config, so it is patched at its source module (the lifespan
    imports it lazily) to keep the records this suite writes inside ``tmp_path``.

    Yields:
        The hub's base URL.
    """
    patches = [
        patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace_dir)},
        ),
        patch(
            "osprey.interfaces.web_terminal.app._load_panel_config",
            return_value=({"artifacts"}, [], None),
        ),
        patch(
            "osprey.interfaces.web_terminal.app._launch_panel_server",
            side_effect=publish_artifact_url(),
        ),
        patch("osprey.utils.workspace.resolve_shared_data_root", return_value=data_root),
    ]
    with _apply_all(patches):
        from osprey.interfaces.web_terminal.app import create_app

        app = create_app(shell_command=["echo", "hello"])
        with _run_app_server(app) as base_url:
            yield base_url


def _open_hub_page(browser: Browser, url: str) -> Page:
    """Open a hub URL and wait for the Feedback control to be live."""
    page = browser.new_page()
    page.goto(url, wait_until="domcontentloaded")
    expect(page.locator('button.panel-rail-button[data-panel-id="artifacts"]')).to_be_attached(
        timeout=10_000
    )
    expect(page.locator("#panel-feedback-btn")).to_be_visible(timeout=10_000)
    return page


def _open_modal(page: Page) -> None:
    """Click the rail's Feedback control and wait until the dialog owns focus.

    The wait is on the focused textarea rather than on the overlay's presence:
    the overlay is attached one frame before it becomes visible, and focusing
    into a ``visibility: hidden`` subtree is a spec'd no-op — which is exactly
    why the dialog reveals and focuses in the same frame. Asserting on focus
    means the dialog is genuinely usable, not merely in the document.

    NOTE: the dialog node is destroyed on close and rebuilt on open, so no
    locator may be held across a cycle. Everything here re-queries.
    """
    page.locator("#panel-feedback-btn").click()
    expect(page.locator(".feedback-modal-overlay.visible")).to_have_count(1, timeout=10_000)
    expect(page.locator(".feedback-modal-dialog textarea.feedback-text")).to_be_focused(
        timeout=10_000
    )


def _close_modal(page: Page) -> None:
    """Dismiss the dialog with Escape and wait for the node to be gone."""
    page.keyboard.press("Escape")
    expect(page.locator(".feedback-modal-overlay")).to_have_count(0, timeout=10_000)


def _palette_hotkey(page: Page) -> str:
    """The palette hotkey this browser is actually wired for.

    ``palette-boot.js`` splits on the platform: Cmd+K on macOS, Ctrl+K
    elsewhere (and there, only outside the terminal, where Ctrl+K stays
    readline's kill-line). The suite runs on macOS locally and Linux in CI, so
    the key is resolved from the page rather than assumed — a hardcoded
    modifier would turn the negative assertion below into a tautology on the
    other platform.
    """
    is_mac = page.evaluate(
        "() => {"
        "  const nav = navigator;"
        "  const platform = (nav.userAgentData && nav.userAgentData.platform)"
        "    || nav.platform || '';"
        "  return /Mac|iPhone|iPad|iPod/i.test(platform);"
        "}"
    )
    return "Meta+k" if is_mac else "Control+k"


def _focus_probe(page: Page) -> dict[str, Any]:
    return page.evaluate(_FOCUS_PROBE)


def test_tab_cycles_through_every_control_and_wraps(tmp_path, chromium_browser):
    """Tab walks the dialog's controls and comes back round, never escaping.

    The wrap is the trap's whole job: the background behind an ``aria-modal``
    dialog is not reachable, so Tab past the last control must land on the
    first rather than on the page underneath.

    "Every control" is the browser's own tab order, read off the live dialog —
    see {@link _EXPECTED_TAB_STOPS} for the two things that are correctly not
    in it. The whole form is asserted separately below, so a control silently
    disappearing still fails this test rather than shrinking its expectations.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _hub_server(workspace, tmp_path / "data") as base_url:
        page = _open_hub_page(chromium_browser, base_url)
        try:
            _open_modal(page)

            # The form is there in full, including the session-context checkbox
            # that this fixture correctly greys out (no PTY session to attach).
            dialog = page.locator(".feedback-modal-dialog")
            expect(dialog.locator("button.feedback-modal-close")).to_have_count(1)
            expect(dialog.locator("textarea.feedback-text")).to_have_count(1)
            expect(dialog.locator('input[type="radio"][name="feedback-channel"]')).to_have_count(3)
            expect(dialog.locator("input.feedback-metadata-check")).to_be_enabled()
            expect(dialog.locator("input.feedback-context-check")).to_be_disabled()
            expect(dialog.locator("button.feedback-help-toggle")).to_have_count(1)
            expect(dialog.locator("button.feedback-send")).to_have_count(1)

            expected = page.evaluate(_EXPECTED_TAB_STOPS)
            # Pinned against the inventory just asserted, not a floor. Both
            # sides of the set comparison below are read off the live DOM, so a
            # control that becomes untabbable (``tabindex="-1"``, or newly
            # disabled) would drop out of BOTH and pass unnoticed; the inventory
            # catches deletion, and this count catches un-tabbing. The six: the
            # close button, the textarea, the channel radios (one stop — they
            # share a name), the metadata checkbox, the help toggle, and Send.
            # The context checkbox is disabled in this fixture and is correctly
            # not a stop.
            assert len(expected) == 6, f"tab order changed shape: {expected}"

            start = _focus_probe(page)["key"]
            visited = [start]
            for _ in range(_MAX_TAB_STEPS):
                page.keyboard.press("Tab")
                probe = _focus_probe(page)
                assert probe["inside"], (
                    f"Tab escaped the dialog to {probe['key']!r} — the focus trap is not "
                    f"installed (walk so far: {visited})"
                )
                visited.append(probe["key"])
                if probe["key"] == start:
                    break
            else:
                pytest.fail(f"Tab never returned to {start!r} in {_MAX_TAB_STEPS} steps: {visited}")

            assert set(visited) == set(expected), (
                f"the Tab cycle did not match the dialog's tab order — "
                f"missed {sorted(set(expected) - set(visited))}, "
                f"unexpected {sorted(set(visited) - set(expected))}"
            )
        finally:
            page.close()


def test_palette_hotkey_belongs_to_the_open_dialog(tmp_path, chromium_browser):
    """Cmd/Ctrl+K is inert while the dialog is open, and live once it closes.

    ``palette-boot.js`` bails on ``isFeedbackModalOpen()`` without consuming
    the event, so the key belongs to whatever is inside the dialog. The second
    half is the control: without it, a wrong modifier would pass the negative
    for the wrong reason.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _hub_server(workspace, tmp_path / "data") as base_url:
        page = _open_hub_page(chromium_browser, base_url)
        try:
            hotkey = _palette_hotkey(page)
            _open_modal(page)

            page.keyboard.press(hotkey)
            page.evaluate(_SETTLE_TWO_FRAMES)
            expect(page.locator(".command-palette-overlay.visible")).to_have_count(0)
            expect(page.locator(".feedback-modal-overlay.visible")).to_have_count(1)

            _close_modal(page)
            # Ctrl+K only reaches the palette from outside the terminal card;
            # the dialog's close path restores focus to the rail button, but
            # pin it explicitly so the control cannot fail for that reason.
            page.locator("#panel-feedback-btn").focus()
            page.keyboard.press(hotkey)
            expect(page.locator(".command-palette-overlay.visible")).to_have_count(
                1, timeout=10_000
            )
        finally:
            page.close()


def test_escape_closes_only_the_palette_layered_over_the_dialog(tmp_path, chromium_browser):
    """With the palette over the dialog, Escape dismisses the palette alone.

    Both register a capture-phase document listener; the dialog's bails while
    ``isOpen()`` reports the palette up, so the top surface closes first and
    the report underneath survives. A second Escape then closes the dialog,
    which is what proves the first one was arbitration and not a lost event.

    The palette is opened through its header trigger programmatically: the
    dialog's scrim covers the button, and pointer reachability is not what this
    test is about.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _hub_server(workspace, tmp_path / "data") as base_url:
        page = _open_hub_page(chromium_browser, base_url)
        try:
            _open_modal(page)
            page.evaluate("() => document.getElementById('command-palette-btn').click()")
            expect(page.locator(".command-palette-overlay.visible")).to_have_count(
                1, timeout=10_000
            )

            page.keyboard.press("Escape")
            expect(page.locator(".command-palette-overlay.visible")).to_have_count(
                0, timeout=10_000
            )
            expect(page.locator(".feedback-modal-overlay.visible")).to_have_count(1)

            page.keyboard.press("Escape")
            expect(page.locator(".feedback-modal-overlay")).to_have_count(0, timeout=10_000)
        finally:
            page.close()


def test_settings_drawer_leaves_no_inert_on_a_reopened_dialog(tmp_path, chromium_browser):
    """A drawer cycle must not poison the next dialog.

    ``<osprey-drawer>``'s ``_syncBackgroundInert()`` marks every top-level
    ``<body>`` child ``inert`` + ``aria-hidden`` while a drawer is open. That is
    the reason the dialog is built on open and destroyed on close rather than
    parked hidden between opens: a parked overlay would be caught by that pass
    and come back unusable. This walks the exact sequence — open, close, drawer
    open, drawer close, reopen — and insists the new dialog is live.

    ``[inert]``/``[aria-hidden]`` are asserted as retrying element counts, never
    as a bare ``get_attribute``: that call does not retry, so it would read
    whichever frame it happened to land in.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _hub_server(workspace, tmp_path / "data") as base_url:
        page = _open_hub_page(chromium_browser, base_url)
        try:
            _open_modal(page)
            _close_modal(page)

            page.evaluate("() => document.getElementById('settings-drawer').open()")
            expect(page.locator("#settings-drawer[open]")).to_have_count(1, timeout=10_000)
            page.evaluate("() => document.getElementById('settings-drawer').close()")
            expect(page.locator("#settings-drawer[open]")).to_have_count(0, timeout=10_000)

            # The drawer's own restore pass, checked before the dialog is
            # rebuilt. Everything below runs against a node built AFTER the
            # cycle, which `_syncBackgroundInert()` could never have marked —
            # so without this line the drawer half of the sequence is walked
            # but never asserted, and a drawer that left `inert` on the body's
            # children would still pass while breaking the real dialog for any
            # user who opened settings first.
            expect(page.locator("body > [inert]")).to_have_count(0)

            _open_modal(page)  # a NEW node — nothing may be held across the cycle
            expect(page.locator(".feedback-modal-overlay[inert]")).to_have_count(0)
            expect(page.locator(".feedback-modal-overlay[aria-hidden]")).to_have_count(0)
            expect(page.locator(".feedback-modal-dialog[inert]")).to_have_count(0)
            # Focus is the behavioural half of the same claim: an inert subtree
            # cannot hold it, so a focused textarea is proof the dialog is live.
            expect(page.locator(".feedback-modal-dialog textarea.feedback-text")).to_be_focused()
        finally:
            page.close()


def test_local_send_writes_a_record_and_reports_its_id(tmp_path, chromium_browser):
    """Type, Send, and the deployment has the report on disk under that id.

    The Local channel is the one that needs no configuration, so it is live the
    moment the dialog is — the outbound channels wait on ``/api/panels`` and are
    deliberately not exercised here. The notice must carry the same id the
    store filed the record under: that id is what an operator quotes and what
    ``osprey feedback show`` takes.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()
    data_root = tmp_path / "data"
    feedback_dir = data_root / "feedback"

    with _hub_server(workspace, data_root) as base_url:
        page = _open_hub_page(chromium_browser, base_url)
        try:
            _open_modal(page)
            page.locator(".feedback-modal-dialog textarea.feedback-text").fill(
                "The rail's utility cluster overlaps the workspace in top mode."
            )
            page.locator(".feedback-modal-dialog button.feedback-send").click()

            notice = page.locator(".feedback-modal-body .feedback-notice")
            expect(notice).to_contain_text("Recorded on this deployment", timeout=15_000)

            records = sorted(feedback_dir.glob("fb-*.json"))
            assert len(records) == 1, f"expected one record in {feedback_dir}, got {records}"

            reported = re.search(r"\(([^)]+)\)", notice.inner_text() or "")
            assert reported is not None, f"the notice carries no record id: {notice.inner_text()!r}"
            assert reported.group(1) == records[0].stem, (
                f"notice reports {reported.group(1)!r} but the store filed {records[0].stem!r}"
            )
        finally:
            page.close()
