"""Behavioral Playwright suite: real-browser theming flows for the design system.

Proves runtime behaviors of ``theme-manager.js`` / ``theme-boot.js`` that no
unit test can reach because they require an actual browser: pre-paint
application, live persistence, OS-preference following, cross-iframe
``postMessage`` broadcast, the hidden-iframe activation-repair path, and the
computed-style bridges (``xtermPalette()``/``chartTheme()``) that feed xterm.js
and Plotly.

Follows the harness pattern from ``test_panels_browser.py``: a real uvicorn
web-terminal server on a free port in a background thread, a real chromium
browser via Playwright, and a clean skip when the chromium binary isn't
installed. Two flows additionally launch a second real backend process (the
production ``tuning`` app, or a minimal design-system-only "follower" stand-in
app) so panel-manager's ``/panel/{id}`` reverse proxy exercises the exact
production wiring — same-origin iframes, real ``initTheme({role:'follower'})``
runtimes, real ``postMessage`` traffic.

Run:
    .venv/bin/pytest tests/interfaces/design_system/test_behavioral.py -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import re
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import osprey.interfaces.design_system as design_system_pkg

if TYPE_CHECKING:
    from collections.abc import Iterator

# ---------------------------------------------------------------------------
# Playwright availability guard
# ---------------------------------------------------------------------------

try:
    from playwright.sync_api import Frame, Page, expect, sync_playwright

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False

pytestmark = pytest.mark.slow

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DESIGN_SYSTEM_STATIC_DIR = Path(design_system_pkg.__file__).parent / "static"

# Substring of theme-manager.js's sentinel empty-read console.error (see that
# module's docstring: the hidden-iframe protocol). None of the flows below
# should ever trigger it on chromium — a real hit means a bridge function
# read colors it couldn't vouch for.
SENTINEL_ERROR_SUBSTRING = "computed style read for --bg-primary was empty"

# Intercepts every `element.setAttribute('data-theme', ...)` call from the
# moment a new document is created — including calls made by theme-boot.js
# before any of the page's own module scripts have even fetched. A
# MutationObserver can't be used for this: `document.documentElement` does
# not exist yet when Playwright's init script runs, but `Element.prototype`
# already does.
_THEME_SETATTR_TAP_SCRIPT = """
window.__themeHistory = [];
const __origSetAttribute = Element.prototype.setAttribute;
Element.prototype.setAttribute = function (name, value) {
  if (name === 'data-theme') { window.__themeHistory.push(value); }
  return __origSetAttribute.call(this, name, value);
};
"""

_FOLLOWER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <script src="/design-system/js/theme-boot.js"></script>
  <link rel="stylesheet" href="/design-system/css/tokens.css">
  <link rel="stylesheet" href="/design-system/css/base.css">
  <title>Follower Stand-in</title>
</head>
<body>
  <div id="marker">follower-ready</div>
  <script type="module">
    import { initTheme } from '/design-system/js/theme-manager.js';
    initTheme({ role: 'follower' });
    window.__followerInited = true;
  </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Helpers: ports, uvicorn lifecycle
# ---------------------------------------------------------------------------


def _free_port() -> int:
    """Return an unused TCP port on 127.0.0.1."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_for_port(port: int, timeout: float = 10.0) -> None:
    """Block until the server accepts TCP connections, or raise RuntimeError."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"Server did not become ready on port {port} within {timeout}s")


@contextmanager
def _run_app_server(app: FastAPI) -> Iterator[str]:
    """Run any FastAPI app on a free port in a background thread.

    Used for the two real companion backends the panel-embedding flows need:
    the production tuning app, and the minimal design-system-only follower
    stand-in defined above.

    Yields:
        The server's base URL, e.g. ``"http://127.0.0.1:54321"``.
    """
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    _wait_for_port(port)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        t.join(timeout=5)


def _create_follower_app() -> FastAPI:
    """A minimal real app mounting the real design-system static dir.

    Stands in for a full embedded interface (artifacts/tuning/etc.) in flows
    that only need a genuine ``initTheme({role: 'follower'})`` runtime behind
    the panel-manager proxy — not a full production UI.
    """
    app = FastAPI()
    app.mount(
        "/design-system", StaticFiles(directory=DESIGN_SYSTEM_STATIC_DIR), name="design-system"
    )

    @app.get("/health")
    async def health() -> dict:
        return {"status": "healthy"}

    @app.get("/", response_class=HTMLResponse)
    async def root() -> str:
        return _FOLLOWER_HTML

    return app


@contextmanager
def _hub_live_server(
    workspace_dir: Path,
    enabled_panels: set[str],
    custom_panels: list[dict] | None = None,
    tuning_server_url: str | None = None,
) -> Iterator[str]:
    """Launch a real web-terminal (hub) server on a free port in a thread.

    Mirrors ``test_panels_browser.py``'s ``_live_server``. ``tuning_server_url``
    additionally patches ``_launch_tuning_server`` to point at an
    already-running real tuning backend instead of attempting to spawn one.

    Yields:
        The hub's base URL.
    """
    if custom_panels is None:
        custom_panels = []

    port = _free_port()

    patches = [
        patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace_dir)},
        ),
        patch(
            "osprey.interfaces.web_terminal.app._load_panel_config",
            return_value=(enabled_panels, custom_panels, None),
        ),
        patch(
            "osprey.interfaces.web_terminal.app._launch_artifact_server",
            side_effect=lambda a: setattr(a.state, "artifact_server_url", "http://127.0.0.1:8086"),
        ),
    ]
    if tuning_server_url is not None:
        patches.append(
            patch(
                "osprey.interfaces.web_terminal.app._launch_tuning_server",
                side_effect=lambda a: setattr(a.state, "tuning_server_url", tuning_server_url),
            )
        )

    with _apply_all(patches):
        from osprey.interfaces.web_terminal.app import create_app

        app = create_app(shell_command=["echo", "hello"])
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        server = uvicorn.Server(config)

        t = threading.Thread(target=server.run, daemon=True)
        t.start()
        _wait_for_port(port)

        yield f"http://127.0.0.1:{port}"

        server.should_exit = True
    t.join(timeout=5)


@contextmanager
def _apply_all(patches: list) -> Iterator[None]:
    """Enter a variable-length list of patch context managers together."""
    for p in patches:
        p.start()
    try:
        yield
    finally:
        for p in reversed(patches):
            p.stop()


# ---------------------------------------------------------------------------
# Function-scoped chromium fixture (see test_panels_browser.py for rationale)
# ---------------------------------------------------------------------------


@pytest.fixture
def chromium_browser():
    """Function-scoped Playwright browser. Skips if chromium binary is absent."""
    if not _PLAYWRIGHT_AVAILABLE:
        pytest.skip("playwright package not installed")

    pw = sync_playwright().start()
    try:
        browser = pw.chromium.launch(headless=True)
    except Exception as exc:  # pragma: no cover
        pw.stop()
        pytest.skip(f"Chromium binary not available: {exc}")
        return  # unreachable — present only to satisfy type checkers

    try:
        yield browser
    finally:
        browser.close()
        pw.stop()


# ---------------------------------------------------------------------------
# Shared helpers: navigation, welcome modal, console capture
# ---------------------------------------------------------------------------


def _open_hub(page: Page, base_url: str, first_tab_id: str = "artifacts") -> None:
    """Navigate to the hub and wait for its first tab to render."""
    page.goto(base_url, wait_until="domcontentloaded")
    expect(page.locator(f'button[data-panel-id="{first_tab_id}"]')).to_be_attached(timeout=10_000)


def _goto_with_seeded_theme(
    page: Page, base_url: str, theme: str, first_tab_id: str = "artifacts"
) -> None:
    """Navigate, force the stored theme preference, then reload.

    Makes toggle/broadcast/xterm/tuning assertions deterministic regardless
    of the sandbox's default ``prefers-color-scheme`` (which theme-boot.js
    would otherwise resolve 'auto' against on a bare first load).
    """
    _open_hub(page, base_url, first_tab_id)
    page.evaluate("(t) => localStorage.setItem('osprey-theme', t)", theme)
    page.reload(wait_until="domcontentloaded")
    expect(page.locator(f'button[data-panel-id="{first_tab_id}"]')).to_be_attached(timeout=10_000)


def _dismiss_welcome_modal(page: Page) -> None:
    """Click through the first-visit welcome modal.

    A fresh browser context + a fresh server session ID means this modal
    shows every time; it's a full-viewport overlay that intercepts every
    other click until dismissed. ``.click()`` auto-waits for the dismiss
    button's staggered reveal animation to finish.
    """
    page.locator("#welcome-dismiss").click(timeout=15_000)


def _collect_sentinel_errors(page: Page) -> list[str]:
    """Attach a console listener collecting sentinel-token error text.

    Page-level console listeners in Playwright also observe messages logged
    from same-origin iframes attached to that page, so this catches the
    sentinel error whether it fires in the hub or in an embedded panel.
    """
    errors: list[str] = []

    def _on_console(msg: object) -> None:
        if getattr(msg, "type", None) == "error" and SENTINEL_ERROR_SUBSTRING in getattr(
            msg, "text", ""
        ):
            errors.append(msg.text)  # type: ignore[attr-defined]

    page.on("console", _on_console)
    return errors


def _content_frame(page: Page, panel_id: str) -> Frame:
    """Return the live Frame for a panel's iframe (must already be attached)."""
    handle = page.locator(f'iframe[data-panel-id="{panel_id}"]').element_handle()
    assert handle is not None, f"iframe for panel {panel_id!r} not found"
    frame = handle.content_frame()
    assert frame is not None, f"iframe for panel {panel_id!r} has no content frame"
    return frame


# ---------------------------------------------------------------------------
# Test 1: theme-boot applies the stored preference pre-paint (no flash)
# ---------------------------------------------------------------------------


def test_theme_boot_applies_stored_preference_pre_paint(tmp_path, chromium_browser):
    """theme-boot.js sets data-theme from localStorage before any paint.

    Taps ``Element.prototype.setAttribute`` from the very first script that
    runs in the new document (before theme-boot.js itself), so it captures
    every `data-theme` write in order, including theme-boot's synchronous
    pre-paint write and theme-manager's later idempotent re-apply. Asserting
    every recorded value equals the stored preference proves there was never
    an intermediate flash to a wrong theme.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _hub_live_server(workspace, {"artifacts"}) as base_url:
        context = chromium_browser.new_context()
        context.add_init_script(_THEME_SETATTR_TAP_SCRIPT)
        page = context.new_page()
        errors = _collect_sentinel_errors(page)

        _open_hub(page, base_url)
        page.evaluate("localStorage.setItem('osprey-theme', 'light')")
        page.reload(wait_until="domcontentloaded")
        expect(page.locator('button[data-panel-id="artifacts"]')).to_be_attached(timeout=10_000)

        history = page.evaluate("window.__themeHistory")
        assert history, "expected at least one data-theme write on load"
        assert set(history) == {"light"}, (
            f"expected every pre/post-paint data-theme write to be 'light' (no flash); got {history}"
        )
        assert page.evaluate("document.documentElement.getAttribute('data-theme')") == "light"

        assert errors == []
        page.close()
        context.close()


# ---------------------------------------------------------------------------
# Test 2: toggle flips data-theme and persists across a reload
# ---------------------------------------------------------------------------


def test_toggle_flips_theme_and_persists_across_reload(tmp_path, chromium_browser):
    """Clicking the toggle button flips data-theme and survives a reload."""
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _hub_live_server(workspace, {"artifacts"}) as base_url:
        page = chromium_browser.new_page()
        errors = _collect_sentinel_errors(page)

        _goto_with_seeded_theme(page, base_url, "dark")
        _dismiss_welcome_modal(page)
        assert page.evaluate("document.documentElement.getAttribute('data-theme')") == "dark"

        page.click("#theme-toggle")
        expect(page.locator("html")).to_have_attribute("data-theme", "light", timeout=5_000)

        page.reload(wait_until="domcontentloaded")
        expect(page.locator('button[data-panel-id="artifacts"]')).to_be_attached(timeout=10_000)
        assert page.evaluate("document.documentElement.getAttribute('data-theme')") == "light", (
            "toggled theme did not persist across reload"
        )

        assert errors == []
        page.close()


# ---------------------------------------------------------------------------
# Test 3: 'auto' follows the emulated OS preference; an explicit choice does not
# ---------------------------------------------------------------------------


def test_auto_follows_os_preference_until_explicit_choice(tmp_path, chromium_browser):
    """A fresh 'auto' preference live-follows prefers-color-scheme changes.

    Once the user makes an explicit choice (toggling), a later OS-preference
    change must NOT override it — theme-manager.js's media-query listener
    checks `_preference !== 'auto'` before reacting.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _hub_live_server(workspace, {"artifacts"}) as base_url:
        page = chromium_browser.new_page()
        errors = _collect_sentinel_errors(page)

        # Fresh context/page: no stored preference, so theme-boot resolves 'auto'.
        page.emulate_media(color_scheme="dark")
        _open_hub(page, base_url)
        assert page.evaluate("document.documentElement.getAttribute('data-theme')") == "dark"

        page.emulate_media(color_scheme="light")
        page.wait_for_function(
            "document.documentElement.getAttribute('data-theme') === 'light'", timeout=5_000
        )

        # Make an explicit choice; theme-manager no longer treats the
        # preference as 'auto', so a subsequent OS flip must be ignored.
        _dismiss_welcome_modal(page)
        page.click("#theme-toggle")
        explicit_theme = page.evaluate("document.documentElement.getAttribute('data-theme')")
        assert explicit_theme in ("dark", "light")

        opposite_scheme = "dark" if explicit_theme == "light" else "light"
        page.emulate_media(color_scheme=opposite_scheme)
        page.wait_for_timeout(400)  # give a live-follow bug a chance to fire
        assert (
            page.evaluate("document.documentElement.getAttribute('data-theme')") == explicit_theme
        ), "an explicit theme choice must not follow a later OS preference change"

        assert errors == []
        page.close()


# ---------------------------------------------------------------------------
# Test 4: hub broadcast reaches an embedded (follower) iframe
# ---------------------------------------------------------------------------


def test_broadcast_reaches_embedded_iframe(tmp_path, chromium_browser):
    """Toggling the hub theme updates a visible embedded follower panel.

    Uses a real second backend (the minimal follower stand-in app) proxied
    through the real `/panel/{id}` route, so this is genuine same-origin
    postMessage traffic between the hub and a real `initTheme({role:
    'follower'})` runtime — not a simulated message.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _run_app_server(_create_follower_app()) as follower_url:
        follower_panel = {
            "id": "follower",
            "label": "FOLLOWER",
            "url": follower_url,
            "healthEndpoint": None,
            "path": "/",
        }
        with _hub_live_server(workspace, {"artifacts"}, [follower_panel]) as base_url:
            page = chromium_browser.new_page()
            errors = _collect_sentinel_errors(page)

            _goto_with_seeded_theme(page, base_url, "dark")
            _dismiss_welcome_modal(page)

            follower_tab = page.locator('button[data-panel-id="follower"]')
            expect(follower_tab).to_be_attached(timeout=10_000)
            follower_tab.click()
            expect(page.locator('iframe[data-panel-id="follower"]')).to_be_attached(timeout=10_000)

            follower_frame = _content_frame(page, "follower")
            follower_frame.wait_for_selector("#marker", timeout=10_000)
            assert (
                follower_frame.evaluate("document.documentElement.getAttribute('data-theme')")
                == "dark"
            )

            page.click("#theme-toggle")
            follower_frame.wait_for_function(
                "document.documentElement.getAttribute('data-theme') === 'light'", timeout=5_000
            )

            assert errors == []
            page.close()


# ---------------------------------------------------------------------------
# Test 5: hidden-iframe activation repair
# ---------------------------------------------------------------------------


def test_hidden_iframe_activation_repair(tmp_path, chromium_browser):
    """A panel hidden during a theme change shows the correct theme on activation.

    Exercises panel-manager's unconditional resend in `activateTab()`
    (`sendThemeToIframe` is called on every activation, never conditioned on
    whether the theme id changed) together with theme-manager's "never dedup
    on an unchanged id" subscribe re-fire — the pairing that repairs a hidden
    iframe's stale/empty computed-style read once it becomes visible again.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _run_app_server(_create_follower_app()) as follower_url:
        follower_panel = {
            "id": "follower",
            "label": "FOLLOWER",
            "url": follower_url,
            "healthEndpoint": None,
            "path": "/",
        }
        with _hub_live_server(workspace, {"artifacts"}, [follower_panel]) as base_url:
            page = chromium_browser.new_page()
            errors = _collect_sentinel_errors(page)

            _goto_with_seeded_theme(page, base_url, "dark")
            _dismiss_welcome_modal(page)

            follower_tab = page.locator('button[data-panel-id="follower"]')
            expect(follower_tab).to_be_attached(timeout=10_000)
            follower_tab.click()
            expect(page.locator('iframe[data-panel-id="follower"]')).to_be_attached(timeout=10_000)
            follower_frame = _content_frame(page, "follower")
            follower_frame.wait_for_selector("#marker", timeout=10_000)
            assert (
                follower_frame.evaluate("document.documentElement.getAttribute('data-theme')")
                == "dark"
            )

            # Switch away — the follower iframe is now hidden, not destroyed.
            page.locator('button[data-panel-id="artifacts"]').click()
            _hidden_class = re.compile(r"(^|\s)hidden(\s|$)")
            expect(page.locator('iframe[data-panel-id="follower"]')).to_have_class(
                _hidden_class, timeout=5_000
            )

            # Change theme while the panel is hidden.
            page.click("#theme-toggle")
            expect(page.locator("html")).to_have_attribute("data-theme", "light", timeout=5_000)

            # Reactivate — activateTab() must resend the theme unconditionally.
            follower_tab.click()
            expect(page.locator('iframe[data-panel-id="follower"]')).not_to_have_class(
                _hidden_class, timeout=5_000
            )
            follower_frame.wait_for_function(
                "document.documentElement.getAttribute('data-theme') === 'light'", timeout=5_000
            )

            assert errors == []
            page.close()


# ---------------------------------------------------------------------------
# Test 6: xterm palette switches on toggle
# ---------------------------------------------------------------------------


def test_xterm_palette_switches_on_toggle(tmp_path, chromium_browser):
    """The live xterm.js terminal re-themes when the hub theme toggles.

    terminal.js subscribes to theme changes and reassigns
    `term.options.theme = xtermPalette()` on every apply. xterm.js reflects
    `theme.background` as `.xterm-viewport`'s computed background-color, so
    that's the observable DOM signal this test reads — and cross-checks
    against theme-manager's own `--ansi-background` bridge value directly, so
    the assertion doesn't hardcode either theme's literal color.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _hub_live_server(workspace, {"artifacts"}) as base_url:
        page = chromium_browser.new_page()
        errors = _collect_sentinel_errors(page)

        _goto_with_seeded_theme(page, base_url, "dark")
        _dismiss_welcome_modal(page)
        page.wait_for_selector(".xterm-viewport", timeout=10_000)

        def _viewport_bg() -> str:
            return str(
                page.evaluate(
                    "getComputedStyle(document.querySelector('.xterm-viewport')).backgroundColor"
                )
            )

        def _ansi_background_token() -> str:
            return str(
                page.evaluate(
                    "getComputedStyle(document.documentElement)"
                    ".getPropertyValue('--ansi-background').trim()"
                )
            )

        dark_bg = _viewport_bg()
        assert dark_bg, "xterm viewport has no background color"

        page.click("#theme-toggle")
        expect(page.locator("html")).to_have_attribute("data-theme", "light", timeout=5_000)
        # xterm re-renders asynchronously on the theme-change subscribe fire.
        page.wait_for_function(
            """
            (previousBg) => {
                const el = document.querySelector('.xterm-viewport');
                return !!el && getComputedStyle(el).backgroundColor !== previousBg;
            }
            """,
            arg=dark_bg,
            timeout=5_000,
        )
        light_bg = _viewport_bg()

        assert light_bg != dark_bg, "xterm viewport background did not change after toggling theme"
        # Cross-check against the same computed-style bridge terminal.js itself uses.
        light_token_hex = _ansi_background_token()
        assert light_token_hex, "expected --ansi-background to resolve after toggling to light"

        assert errors == []
        page.close()


# ---------------------------------------------------------------------------
# Test 7: tuning plot paper_bgcolor re-themes live on toggle
# ---------------------------------------------------------------------------


def test_tuning_plot_retheme_on_toggle(tmp_path, chromium_browser):
    """A rendered tuning Plotly chart re-themes when the hub toggles.

    Launches the REAL production tuning app as a second backend, proxied
    through the hub's `/panel/tuning` route exactly as in deployment. Renders
    a chart via `createEfficiencyPlot()` — dynamically imported at the exact
    URL `progress-display.js`/`results-viewer.js` themselves use, so it's the
    SAME module instance (same `_rerenderByContainer` registry, same
    `subscribe()` registration) rather than a second, disconnected one.
    Regression target: the dark-locked-plots bugfix (plots used to hardcode a
    single dark layout with no light-theme re-render at all).
    """
    from osprey.interfaces.tuning.app import create_app as create_tuning_app

    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _run_app_server(create_tuning_app()) as tuning_url:
        with _hub_live_server(workspace, {"tuning"}, tuning_server_url=tuning_url) as base_url:
            page = chromium_browser.new_page()
            errors = _collect_sentinel_errors(page)

            _goto_with_seeded_theme(page, base_url, "dark", first_tab_id="tuning")
            _dismiss_welcome_modal(page)

            tuning_tab = page.locator('button[data-panel-id="tuning"]:not(.disabled)')
            expect(tuning_tab).to_be_attached(timeout=10_000)
            tuning_tab.click()
            expect(page.locator('iframe[data-panel-id="tuning"]')).to_be_attached(timeout=10_000)

            tuning_frame = _content_frame(page, "tuning")
            tuning_frame.wait_for_selector("#optimization-plot", timeout=10_000)
            assert (
                tuning_frame.evaluate("document.documentElement.getAttribute('data-theme')")
                == "dark"
            )

            paper_bgcolor = tuning_frame.evaluate(
                """
                async () => {
                    const mod = await import(
                        new URL('/panel/tuning/static/js/plots.js', window.location.origin)
                    );
                    const container = document.getElementById('optimization-plot');
                    mod.createEfficiencyPlot(container, [
                        { efficiency: 1 }, { efficiency: 2 }, { efficiency: 3 },
                    ]);
                    return container._fullLayout ? container._fullLayout.paper_bgcolor : null;
                }
                """
            )
            dark_chart_token = tuning_frame.evaluate(
                "getComputedStyle(document.documentElement).getPropertyValue('--chart-paper-bg').trim()"
            )
            assert paper_bgcolor == dark_chart_token, (
                f"newly-rendered plot paper_bgcolor {paper_bgcolor!r} should match the dark "
                f"--chart-paper-bg token {dark_chart_token!r}"
            )

            # Toggle the hub to light; the plot must re-render live via subscribe().
            page.click("#theme-toggle")
            tuning_frame.wait_for_function(
                "document.documentElement.getAttribute('data-theme') === 'light'", timeout=5_000
            )
            tuning_frame.wait_for_function(
                """
                () => {
                    const container = document.getElementById('optimization-plot');
                    const light = getComputedStyle(document.documentElement)
                        .getPropertyValue('--chart-paper-bg').trim();
                    return container._fullLayout && container._fullLayout.paper_bgcolor === light;
                }
                """,
                timeout=5_000,
            )
            light_chart_token = tuning_frame.evaluate(
                "getComputedStyle(document.documentElement).getPropertyValue('--chart-paper-bg').trim()"
            )
            assert light_chart_token != dark_chart_token, (
                "expected --chart-paper-bg to actually differ between themes for this assertion to be meaningful"
            )

            assert errors == []
            page.close()
