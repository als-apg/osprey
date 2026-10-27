"""Per-interface "loads clean in a real browser" smoke suite.

Boots each OSPREY interface ``create_app()`` factory on a real uvicorn server
(see ``_launchers.py``) and drives a real Chromium page
against ``GET /``, asserting via :func:`assert_page_loads_clean` that the page
loaded without an uncaught JS exception or a failed same-origin
script/stylesheet fetch. This is the negative-space complement to
``test_app_setup_parametrized.py`` (which only inspects the assembled FastAPI
app object, never a live page): here the browser is the oracle.

Two targets additionally assert their DOM actually rendered, since a page can
load "clean" by the above signals while its JS silently no-ops before building
any UI: Web Terminal checks the hub shell's artifacts tab button, and the
graph-mode Channel Finder checks that the explore view's class tree drew nodes.

Run:
    uv run pytest tests/interfaces/test_load_smokes.py -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.interfaces._browser import assert_page_loads_clean
from tests.interfaces._launchers import (
    launch_ariel,
    launch_artifacts,
    launch_channel_finder,
    launch_lattice_dashboard,
    launch_okf_panel,
    launch_web_terminal,
)
from tests.interfaces.conftest import launch_graph_channel_finder

if TYPE_CHECKING:
    from playwright.sync_api import Browser

try:
    from playwright.sync_api import expect

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False

pytestmark = [pytest.mark.browser, pytest.mark.slow]


# ---------------------------------------------------------------------------
# Allowlists
# ---------------------------------------------------------------------------
# Empty (None) unless a real interface turns out to have benign vendored-lib
# noise; keep entries narrow and documented with the benign source.

_ALLOWLISTS: dict[str, object] = {
    "web_terminal": None,
    "artifacts": None,
    "ariel": None,
    "channel_finder": None,
    "channel_finder_graph": None,
    "lattice_dashboard": None,
    "okf_panel": None,
    "web_terminal_session": None,
}


INTERFACE_LAUNCHERS = [
    ("web_terminal", launch_web_terminal),
    ("artifacts", launch_artifacts),
    ("ariel", launch_ariel),
    ("channel_finder", launch_channel_finder),
    # Same interface, second paradigm: the graph mode serves a different
    # explore view off the same factory, so it gets its own target. The
    # launcher (shared with the visual lane) writes the graph config and
    # points the app's store seam at the demo corpus -- no database is dialled.
    ("channel_finder_graph", launch_graph_channel_finder),
    ("lattice_dashboard", launch_lattice_dashboard),
    ("okf_panel", launch_okf_panel),
]


@pytest.mark.parametrize(
    "name,launcher",
    INTERFACE_LAUNCHERS,
    ids=[name for name, _ in INTERFACE_LAUNCHERS],
)
def test_interface_loads_clean(
    name: str,
    launcher,
    tmp_path,
    monkeypatch,
    chromium_browser: Browser,
) -> None:
    """Each interface's ``GET /`` loads with no uncaught error or failed asset."""
    with launcher(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()
        assert_page_loads_clean(page, base_url, allowlist=_ALLOWLISTS[name])

        if name == "web_terminal":
            if not _PLAYWRIGHT_AVAILABLE:  # pragma: no cover - guarded above too
                pytest.skip("playwright package not installed")
            # Hub shell DOM check: the artifacts tab button is always present
            # (artifacts is always-enabled, see _open_page in
            # test_panels_browser.py) once panel-manager.js finishes its
            # async init (fetch /api/panels, then per-panel config).
            expect(page.locator('button[data-panel-id="artifacts"]')).to_be_visible(timeout=10_000)

        if name == "channel_finder_graph":
            # Explore is the default route (no hash), and its finder renders
            # result rows only after /api/graph/search answers -- so a result
            # row is the DOM proof the page did more than load.
            row = "#graph-finder-table .result-table tbody tr[data-pv]:first-child"
            page.wait_for_selector(row, timeout=10_000)
            assert page.locator(row).count() >= 1

        page.close()


# ---------------------------------------------------------------------------
# Web-terminal static pages (session.html) -- CC-11
# ---------------------------------------------------------------------------
# A static asset under the hub's own /static mount (configure_interface_app
# mounts the whole static/ dir), not a second app -- a path-override on the
# same launch_web_terminal hub server above, mirroring test_visual.py's
# _web_terminal_static_page_server (which likewise reuses its hub server
# unchanged and only varies the VisualTarget's `path`).

_WEB_TERMINAL_STATIC_PAGES = [
    ("web_terminal_session", "/static/session.html"),
]


@pytest.mark.parametrize(
    "name,path",
    _WEB_TERMINAL_STATIC_PAGES,
    ids=[name for name, _ in _WEB_TERMINAL_STATIC_PAGES],
)
def test_web_terminal_static_page_loads_clean(
    name: str,
    path: str,
    tmp_path,
    monkeypatch,
    chromium_browser: Browser,
) -> None:
    """session.html loads with no uncaught error or failed asset."""
    with launch_web_terminal(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()
        assert_page_loads_clean(page, f"{base_url}{path}", allowlist=_ALLOWLISTS[name])
        page.close()
