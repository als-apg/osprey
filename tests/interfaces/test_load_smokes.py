"""Seven-interface "loads clean in a real browser" smoke suite.

Boots each of the seven OSPREY interface ``create_app()`` factories on a real
uvicorn server (see ``_run_app_server`` in ``conftest.py``) and drives a real
Chromium page against ``GET /``, asserting via :func:`assert_page_loads_clean`
that the page loaded without an uncaught JS exception or a failed same-origin
script/stylesheet fetch. This is the negative-space complement to
``test_app_setup_parametrized.py`` (which only inspects the assembled FastAPI
app object, never a live page): here the browser is the oracle.

Web Terminal additionally asserts the hub shell actually rendered its DOM
(the artifacts tab button), since a page can load "clean" by the above
signals while its JS silently no-ops before building any UI.

Run:
    uv run pytest tests/interfaces/test_load_smokes.py -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from tests.interfaces._browser import assert_page_loads_clean
from tests.interfaces.conftest import _run_app_server

if TYPE_CHECKING:
    from collections.abc import Iterator

    from playwright.sync_api import Browser

try:
    from playwright.sync_api import expect

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False

pytestmark = [pytest.mark.browser, pytest.mark.slow]


# ---------------------------------------------------------------------------
# Per-interface launchers
# ---------------------------------------------------------------------------
# Each is a @contextmanager yielding a live base_url, isolating cwd to
# tmp_path first (matters for ariel's config lookup and artifacts' index
# dir) and building the app with the same signature the interface's own
# tests use (see test_app_setup_parametrized.py).


@contextmanager
def _launch_web_terminal(tmp_path, monkeypatch) -> Iterator[str]:
    monkeypatch.chdir(tmp_path)
    # Patch trio bypasses companion-backend spawn; must stay active while the
    # lifespan runs (i.e. across both create_app() and _run_app_server).
    with (
        patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(tmp_path)},
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


@contextmanager
def _launch_artifacts(tmp_path, monkeypatch) -> Iterator[str]:
    monkeypatch.chdir(tmp_path)
    from osprey.interfaces.artifacts.app import create_app

    app = create_app(workspace_root=tmp_path)
    with _run_app_server(app) as base_url:
        yield base_url


@contextmanager
def _launch_ariel(tmp_path, monkeypatch) -> Iterator[str]:
    monkeypatch.chdir(tmp_path)
    from osprey.interfaces.ariel.app import create_app

    app = create_app()
    with _run_app_server(app) as base_url:
        yield base_url


@contextmanager
def _launch_channel_finder(tmp_path, monkeypatch) -> Iterator[str]:
    monkeypatch.chdir(tmp_path)
    from osprey.interfaces.channel_finder.app import create_app

    app = create_app(project_cwd=str(tmp_path))
    with _run_app_server(app) as base_url:
        yield base_url


@contextmanager
def _launch_lattice_dashboard(tmp_path, monkeypatch) -> Iterator[str]:
    monkeypatch.chdir(tmp_path)
    from osprey.interfaces.lattice_dashboard.app import create_app

    app = create_app(workspace_root=tmp_path)
    with _run_app_server(app) as base_url:
        yield base_url


@contextmanager
def _launch_okf_panel(tmp_path, monkeypatch) -> Iterator[str]:
    monkeypatch.chdir(tmp_path)
    from osprey.interfaces.okf_panel.app import create_app

    # Build over the shared on-disk fixture bundle the panel's own unit tests use
    # (tests/interfaces/okf_panel/fixtures/bundle) so GET / renders real content.
    bundle = Path(__file__).parent / "okf_panel" / "fixtures" / "bundle"
    app = create_app(bundle_path=str(bundle))
    with _run_app_server(app) as base_url:
        yield base_url


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
    "lattice_dashboard": None,
    "okf_panel": None,
    "web_terminal_session": None,
    "web_terminal_safety": None,
}


INTERFACE_LAUNCHERS = [
    ("web_terminal", _launch_web_terminal),
    ("artifacts", _launch_artifacts),
    ("ariel", _launch_ariel),
    ("channel_finder", _launch_channel_finder),
    ("lattice_dashboard", _launch_lattice_dashboard),
    ("okf_panel", _launch_okf_panel),
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

        page.close()


# ---------------------------------------------------------------------------
# Web-terminal static pages (session.html, safety.html) -- CC-11
# ---------------------------------------------------------------------------
# Both are static assets under the hub's own /static mount (configure_interface_app
# mounts the whole static/ dir), not a second app -- a path-override on the
# same _launch_web_terminal hub server above, mirroring test_visual.py's
# _web_terminal_static_page_server (which likewise reuses its hub server
# unchanged and only varies the VisualTarget's `path`).

_WEB_TERMINAL_STATIC_PAGES = [
    ("web_terminal_session", "/static/session.html"),
    ("web_terminal_safety", "/static/safety.html"),
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
    """session.html/safety.html load with no uncaught error or failed asset."""
    with _launch_web_terminal(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()
        assert_page_loads_clean(page, f"{base_url}{path}", allowlist=_ALLOWLISTS[name])
        page.close()
