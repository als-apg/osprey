"""Real-browser suite for standalone ARIEL's ``<osprey-display-menu>`` popover.

Standalone ARIEL mounts the shared display-preferences popover in its header
(``<osprey-display-menu settings-drawer="settings-drawer">``). Everything the
popover does is runtime browser behavior — custom-element upgrade, the theme
manager's live apply, the pre-paint boot scripts, ``localStorage`` persistence,
the ``?mode=`` strip — so a real Chromium page is the only oracle for it.

Covered here, one concern per test:

- the popover opens from the header trigger and renders its three fixed rows;
- a family / appearance / view pick applies immediately (``data-theme``,
  ``data-ui-mode``);
- both preferences survive a reload, including from a page first opened with a
  one-shot ``?mode=`` deep-link — the explicit pick strips the param and wins;
- a family picked on standalone ARIEL is adopted by the web-terminal hub on its
  next load (the two surfaces share one origin-scoped preference);
- embedded (``?embedded=true``), the component's own hide rule takes the
  popover out.

The cross-surface case needs both surfaces on ONE origin, since
``localStorage`` is origin-scoped: it runs the hub on the port ARIEL just
vacated (see ``_serve_on_port``), so the browser context carries the pick from
one app to the other exactly as it does in a deployment that serves them from
the same host.

Run:
    uv run pytest tests/interfaces/ariel/test_display_menu.py -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import re
import socket
import threading
import time
from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING
from unittest.mock import patch
from urllib.parse import urlparse

import pytest
import uvicorn

from tests.interfaces._browser import assert_page_loads_clean
from tests.interfaces._panel_launch import publish_artifact_url
from tests.interfaces.conftest import _apply_all

# The standalone-ARIEL launcher the seven-interface smoke suite already owns —
# imported rather than re-declared so this suite boots the app exactly the way
# that suite does (cwd isolation included).
from tests.interfaces.test_load_smokes import _launch_ariel

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from fastapi import FastAPI
    from playwright.sync_api import Browser, Locator, Page

try:
    from playwright.sync_api import expect

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False

pytestmark = [
    pytest.mark.browser,
    pytest.mark.slow,
    pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason="playwright package not installed"),
]

# Attaching to the DOM is not the same as the element having upgraded and
# rendered: `<osprey-display-menu>` builds its markup in connectedCallback,
# which only runs once app.js's module import chain has registered the element.
_MENU = "osprey-display-menu"
_TRIGGER = f"{_MENU} .display-menu-trigger"
_CARD = f"{_MENU} .display-menu-card"


# ---------------------------------------------------------------------------
# Hub on a chosen port (cross-surface only)
# ---------------------------------------------------------------------------


def _listen_on(port: int, attempts: int = 3, delay: float = 0.3) -> socket.socket:
    """Bind and listen on *port*, retrying a few times before giving up.

    ``SO_REUSEADDR`` only covers the predecessor's own connections lingering in
    TIME_WAIT. Between ARIEL releasing the port and the hub claiming it there is
    a millisecond window in which anything else on the machine can take it, and
    on a busy CI runner that is a real (if rare) event — retrying turns it into
    a pause instead of a red build.

    Raises:
        OSError: If every attempt was refused; the last refusal is the cause.
    """
    last: OSError | None = None
    for attempt in range(attempts):
        sock = socket.socket()
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as exc:
            sock.close()
            last = exc
            if attempt + 1 < attempts:
                time.sleep(delay)
            continue
        sock.listen(128)
        return sock
    raise OSError(f"Could not claim port {port} for the hub after {attempts} attempts") from last


@contextmanager
def _serve_on_port(app: FastAPI, port: int) -> Iterator[str]:
    """Run *app* on a caller-chosen port, mirroring ``run_app_server``.

    ``osprey.interfaces._serving.run_app_server`` always claims a fresh port,
    which is right for every suite that only needs *a* server. The
    cross-surface flow below instead needs the second app on the *same origin*
    as the first, so it hands over the port the first one just released.

    Yields:
        The server's base URL.
    """
    sock = _listen_on(port)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread_error: list[BaseException] = []

    def _serve() -> None:
        try:
            server.run(sockets=[sock])
        except BaseException as exc:  # noqa: BLE001 - surfaced to the caller as the cause
            # Includes the SystemExit uvicorn raises when startup fails; without
            # this the thread dies silently and the caller only learns the
            # server "never became ready".
            thread_error.append(exc)

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()

    def _shutdown() -> None:
        server.should_exit = True
        thread.join(timeout=5)
        with suppress(OSError):
            sock.close()

    try:
        for _ in range(500):
            if server.started:
                break
            if not thread.is_alive():
                raise RuntimeError(
                    f"Hub server exited during startup without listening on port {port}"
                ) from (thread_error[0] if thread_error else None)
            time.sleep(0.02)
        else:
            raise RuntimeError(f"Hub server did not become ready on port {port}")
        yield f"http://127.0.0.1:{port}"
    finally:
        _shutdown()


@contextmanager
def _launch_hub_on_port(workspace_dir: Path, port: int) -> Iterator[str]:
    """Launch a real web-terminal hub on *port*.

    Same patch set as ``design_system/test_behavioral.py``'s
    ``_hub_live_server`` (config lookup, panel config, panel-server spawn); the
    only difference is the fixed port.
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
    ]

    with _apply_all(patches):
        from osprey.interfaces.web_terminal.app import create_app

        app = create_app(shell_command=["echo", "hello"])
        with _serve_on_port(app, port) as base_url:
            yield base_url


# ---------------------------------------------------------------------------
# Shared page helpers
# ---------------------------------------------------------------------------


def _goto_ariel(page: Page, url: str) -> Locator:
    """Navigate to a standalone-ARIEL URL and wait for the menu to upgrade.

    Returns:
        The ``<osprey-display-menu>`` locator.
    """
    assert_page_loads_clean(page, url)
    expect(page.locator(_TRIGGER)).to_be_attached(timeout=10_000)
    return page.locator(_MENU)


def _seed_explicit_theme(page: Page, theme: str = "dark") -> None:
    """Pin the stored theme and reload, so no assertion rides on the OS setting.

    Without this, theme-boot.js resolves the default 'auto' preference against
    the sandbox's ``prefers-color-scheme`` and the concrete theme id (and the
    Appearance row's starting side) differs per machine.
    """
    page.evaluate("(t) => localStorage.setItem('osprey-theme', t)", theme)
    page.reload(wait_until="load")
    expect(page.locator("html")).to_have_attribute("data-theme", theme, timeout=10_000)
    expect(page.locator(_TRIGGER)).to_be_attached(timeout=10_000)


def _open_menu(page: Page) -> None:
    """Click the header trigger and wait for the popover card to show."""
    page.click(_TRIGGER)
    expect(page.locator(_CARD)).to_be_visible(timeout=5_000)


# ---------------------------------------------------------------------------
# Test 1: the popover opens and renders its three fixed rows
# ---------------------------------------------------------------------------


def test_popover_opens_with_its_three_rows(
    tmp_path, monkeypatch, chromium_browser: Browser
) -> None:
    """The trigger reveals a card carrying Appearance, View and Theme."""
    with _launch_ariel(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()
        menu = _goto_ariel(page, base_url)

        expect(page.locator(_CARD)).to_be_hidden()
        _open_menu(page)

        # Text content, not inner_text: the headings are rendered uppercase by
        # a `text-transform`, which inner_text would report back.
        assert menu.locator(".display-menu-heading").all_text_contents() == [
            "Appearance",
            "View",
            "Theme",
        ]
        expect(menu.locator(".display-seg-option[data-appearance]")).to_have_count(2)
        expect(menu.locator(".display-seg-option[data-mode]")).to_have_count(2)

        # The Theme row offers exactly the registry's families, in its order —
        # asserted against theme-manager.js itself rather than a hardcoded list,
        # so a token regeneration that adds a family can't leave the row stale.
        registry_families = page.evaluate(
            "async () => (await import('/design-system/js/theme-manager.js'))"
            ".themeFamilies().map((f) => f.id)"
        )
        rendered_families = menu.locator(".display-family-pill").evaluate_all(
            "(pills) => pills.map((p) => p.dataset.family)"
        )
        assert rendered_families == registry_families

        # ARIEL passes settings-drawer, so the optional Settings entry is built
        # and carries <osprey-drawer>'s delegated trigger contract.
        expect(menu.locator('.display-menu-settings[data-drawer="settings-drawer"]')).to_have_count(
            1
        )

        page.close()


# ---------------------------------------------------------------------------
# Test 2: every row applies its pick immediately
# ---------------------------------------------------------------------------


def test_picks_apply_immediately(tmp_path, monkeypatch, chromium_browser: Browser) -> None:
    """Family, appearance and view picks land on <html> without a reload."""
    with _launch_ariel(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()
        menu = _goto_ariel(page, base_url)
        _seed_explicit_theme(page, "dark")
        html = page.locator("html")
        expect(html).to_have_attribute("data-ui-mode", "expert", timeout=10_000)

        _open_menu(page)

        # Family: the current dark/light preference rides along (dark -> desy-dark).
        menu.locator('.display-family-pill[data-family="desy"]').click()
        expect(html).to_have_attribute("data-theme", "desy-dark", timeout=5_000)
        expect(menu.locator('.display-family-pill[data-family="desy"]')).to_have_class(
            re.compile(r"\bactive\b")
        )

        # Appearance: flips light/dark WITHIN that family.
        menu.locator('.display-seg-option[data-appearance="light"]').click()
        expect(html).to_have_attribute("data-theme", "desy-light", timeout=5_000)

        # View: the pick round-trips through the same-window mode broadcast.
        menu.locator('.display-seg-option[data-mode="simple"]').click()
        expect(html).to_have_attribute("data-ui-mode", "simple", timeout=5_000)

        # Display picks leave the card open — only the Settings entry closes it.
        expect(page.locator(_CARD)).to_be_visible()

        page.close()


# ---------------------------------------------------------------------------
# Test 3: both preferences survive a reload, and outrank a ?mode= deep-link
# ---------------------------------------------------------------------------


def test_theme_and_view_survive_reload_over_a_query_mode_deep_link(
    tmp_path, monkeypatch, chromium_browser: Browser
) -> None:
    """An explicit view pick strips a leftover ``?mode=`` and wins on reload."""
    with _launch_ariel(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()
        menu = _goto_ariel(page, f"{base_url}/?mode=simple")
        html = page.locator("html")
        expect(html).to_have_attribute("data-ui-mode", "simple", timeout=10_000)

        _seed_explicit_theme(page, "dark")
        assert page.evaluate("location.search") == "?mode=simple", (
            "the deep-link must still be on the URL when the pick is made"
        )

        _open_menu(page)
        menu.locator('.display-seg-option[data-mode="expert"]').click()
        expect(html).to_have_attribute("data-ui-mode", "expert", timeout=5_000)
        assert page.evaluate("location.search") == "", (
            "an explicit view pick must drop the one-shot ?mode= param"
        )

        menu.locator('.display-family-pill[data-family="retro"]').click()
        expect(html).to_have_attribute("data-theme", "retro-dark", timeout=5_000)

        page.reload(wait_until="load")
        expect(page.locator(_TRIGGER)).to_be_attached(timeout=10_000)
        expect(html).to_have_attribute("data-theme", "retro-dark", timeout=5_000)
        expect(html).to_have_attribute("data-ui-mode", "expert", timeout=5_000)
        assert page.evaluate("location.search") == ""

        page.close()


# ---------------------------------------------------------------------------
# Test 4: cross-surface — the hub adopts ARIEL's pick on its next load
# ---------------------------------------------------------------------------


def test_family_picked_on_ariel_is_adopted_by_the_hub(
    tmp_path, monkeypatch, chromium_browser: Browser
) -> None:
    """A family chosen in standalone ARIEL is what the hub boots into.

    One browser context spans both surfaces, and the hub takes over the exact
    origin ARIEL just released, so the shared ``osprey-theme`` key is the only
    thing carrying the preference across.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    context = chromium_browser.new_context()
    page = context.new_page()
    try:
        with _launch_ariel(tmp_path, monkeypatch) as ariel_url:
            port = urlparse(ariel_url).port
            assert port is not None
            menu = _goto_ariel(page, ariel_url)
            _seed_explicit_theme(page, "dark")

            _open_menu(page)
            menu.locator('.display-family-pill[data-family="desy"]').click()
            expect(page.locator("html")).to_have_attribute("data-theme", "desy-dark", timeout=5_000)

        with _launch_hub_on_port(workspace, port) as hub_url:
            assert hub_url == ariel_url, "the hub must serve the origin ARIEL used"
            # Two apps behind one URL is a harness-only situation, and the
            # browser has no way to know the document behind it changed: an
            # unbusted goto is answered from the HTTP cache with ARIEL's page.
            # The param is unknown to the hub (and origin-irrelevant, so the
            # stored preference is still the same one ARIEL wrote).
            page.goto(f"{hub_url}/?surface=hub", wait_until="domcontentloaded")
            expect(page.locator('button[data-panel-id="artifacts"]')).to_be_attached(timeout=10_000)
            expect(page.locator("html")).to_have_attribute("data-theme", "desy-dark", timeout=5_000)
    finally:
        page.close()
        context.close()


# ---------------------------------------------------------------------------
# Test 5: embedded ARIEL shows no popover
# ---------------------------------------------------------------------------


def test_embedded_hides_the_popover(tmp_path, monkeypatch, chromium_browser: Browser) -> None:
    """``?embedded=true`` hides the menu through the component's own rule.

    ``display`` computes independently of an ancestor's own ``display: none``
    (ARIEL's header carries one when embedded), so reading it off the element
    proves the ``body.embedded osprey-display-menu`` rule the component injects
    for itself is what took the popover out.
    """
    with _launch_ariel(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()
        _goto_ariel(page, f"{base_url}/?embedded=true")

        assert page.evaluate("document.body.classList.contains('embedded')")
        expect(page.locator(_MENU)).to_be_hidden()
        assert (
            page.evaluate("getComputedStyle(document.querySelector('osprey-display-menu')).display")
            == "none"
        )

        page.close()
