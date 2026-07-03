"""Visual regression suite: per-interface x per-theme screenshot diffing.

Complements ``test_behavioral.py`` (interaction flows) with a pixel-level
check that each interface's *resting* appearance in each theme hasn't
drifted. Every target below is launched as a real standalone app (or, for
the web-terminal hub, a real hub + a real embedded artifacts backend — the
same production wiring ``test_behavioral.py`` uses) via uvicorn in a
background thread, screenshotted at a fixed 1280x800 viewport with real
chromium, and diffed against a committed baseline PNG with Pillow.

Platform split (anti-aliasing/subpixel rendering differs across OSes, so a
byte-level compare only means something when producer and baseline were
rendered on the same platform):

- Linux (CI): the authoritative comparison. A missing baseline is written
  on the spot (bootstrap); an existing one is perceptually diffed with a
  tolerance for AA noise.
- Everywhere else (e.g. a contributor's macOS machine): screenshot capture
  is still exercised and asserted valid, but the byte-compare is skipped
  with an explicit notice — see ``scripts/ci_check.sh`` for the analogous
  local-dev notice pattern.

``--regen-baselines`` (see ``conftest.py``) unconditionally overwrites the
baseline with the freshly captured screenshot, for use on Linux (CI, or a
Linux dev box) to produce the authoritative reference images.

Run:
    .venv/bin/pytest tests/interfaces/design_system/test_visual.py -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import io
import platform
import socket
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

import osprey.interfaces.design_system as design_system_pkg

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from contextlib import AbstractContextManager

    from playwright.sync_api import Browser

# ---------------------------------------------------------------------------
# Playwright availability guard
# ---------------------------------------------------------------------------

try:
    from playwright.sync_api import expect, sync_playwright

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False

pytestmark = pytest.mark.slow

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DESIGN_SYSTEM_STATIC_DIR = Path(design_system_pkg.__file__).parent / "static"
BASELINES_DIR = Path(__file__).parent / "baselines"
VIEWPORT = {"width": 1280, "height": 800}
THEMES = ("dark", "light")

# Fraction of pixels allowed to exceed PIXEL_THRESHOLD before a comparison
# fails. Both numbers are deliberately generous: this suite is a drift
# detector for real regressions (wrong token, missed re-theme, broken
# layout), not a pixel-perfect font-rendering assertion — font hinting and
# anti-aliasing shift a small fraction of edge pixels between otherwise
# identical renders even on the same OS/chromium build.
PIXEL_THRESHOLD = 30
DIFF_RATIO_TOLERANCE = 0.02

_DASHBOARD_HTML_ONLY_ON_LINUX_NOTE = (
    "browser-based visual baseline NOT byte-compared on this platform "
    "(no Linux) — CI enforces the pixel diff on ubuntu-latest"
)


# ---------------------------------------------------------------------------
# Helpers: ports, uvicorn lifecycle (mirrors test_behavioral.py)
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


@contextmanager
def _hub_live_server(workspace_dir: Path, artifact_server_url: str) -> Iterator[str]:
    """Launch a real web-terminal (hub) server pointed at a real artifacts backend.

    Mirrors ``test_behavioral.py``'s ``_hub_live_server``, simplified to the one
    configuration this suite needs: artifacts enabled and genuinely reachable
    (not the dead fake URL some other suites use), so the captured screenshot
    shows real embedded content instead of a proxy error page.
    """
    port = _free_port()
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
            "osprey.interfaces.web_terminal.app._launch_artifact_server",
            side_effect=lambda a: setattr(a.state, "artifact_server_url", artifact_server_url),
        ),
    ]
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


# ---------------------------------------------------------------------------
# The dispatch dashboard has no standalone FastAPI app of its own in
# production (it's rendered HTML mounted into the event-dispatcher server) —
# stand one up the same way test_behavioral.py's follower stand-in does: a
# minimal real app that mounts the real design-system static dir and serves
# the real ``render_dashboard_html()`` output.
# ---------------------------------------------------------------------------


def _create_dispatch_dashboard_app() -> FastAPI:
    from osprey.dispatch.dashboard import render_dashboard_html

    app = FastAPI()
    app.mount(
        "/design-system", StaticFiles(directory=DESIGN_SYSTEM_STATIC_DIR), name="design-system"
    )

    @app.get("/", response_class=HTMLResponse)
    async def root() -> str:
        return str(render_dashboard_html(facility_name="Test Facility", pv_strip_prefix="SR:"))

    return app


# ---------------------------------------------------------------------------
# Visual targets: one real app (or hub + real embedded backend) per interface
# ---------------------------------------------------------------------------


@dataclass
class VisualTarget:
    """One screenshot target: a server factory, a path, and capture options."""

    name: str
    server_factory: Callable[[Path], AbstractContextManager[str]]
    path: str = "/"
    # The web-terminal hub shows a first-visit welcome modal (full-viewport
    # overlay) that would otherwise dominate the screenshot; dismiss it so
    # the baseline shows the actual working interface.
    dismiss_welcome: bool = False
    wait_selector: str | None = None


def _artifacts_server(tmp_path: Path):
    from osprey.interfaces.artifacts.app import create_app

    return _run_app_server(create_app(workspace_root=tmp_path / "artifacts_ws"))


def _ariel_server(tmp_path: Path):
    from osprey.interfaces.ariel.app import create_app

    # No config.yml passed: the app gracefully degrades to a DB-less mode
    # (search/browse disabled, UI still renders) rather than failing to boot.
    return _run_app_server(create_app())


def _channel_finder_server(tmp_path: Path):
    from osprey.interfaces.channel_finder.app import create_app

    return _run_app_server(create_app(project_cwd=str(tmp_path / "cf_ws")))


def _tuning_server(tmp_path: Path):
    from osprey.interfaces.tuning.app import create_app

    return _run_app_server(create_app())


def _lattice_dashboard_server(tmp_path: Path):
    from osprey.interfaces.lattice_dashboard.app import create_app

    return _run_app_server(create_app(workspace_root=tmp_path / "lattice_ws"))


def _dispatch_dashboard_server(tmp_path: Path):
    return _run_app_server(_create_dispatch_dashboard_app())


@contextmanager
def _web_terminal_hub_server(tmp_path: Path) -> Iterator[str]:
    """Hub + a real artifacts backend (not the dead-URL trick other suites use)."""
    from osprey.interfaces.artifacts.app import create_app as create_artifacts_app

    workspace = tmp_path / "hub_ws"
    workspace.mkdir(exist_ok=True)
    with _run_app_server(create_artifacts_app(workspace_root=workspace)) as artifact_url:
        with _hub_live_server(workspace, artifact_url) as base_url:
            yield base_url


def _web_terminal_static_page_server(tmp_path: Path):
    """session.html/safety.html are served from the hub's /static mount."""
    return _web_terminal_hub_server(tmp_path)


TARGETS: list[VisualTarget] = [
    VisualTarget(
        "web_terminal_hub",
        _web_terminal_hub_server,
        path="/",
        dismiss_welcome=True,
        wait_selector='button[data-panel-id="artifacts"]',
    ),
    VisualTarget(
        "web_terminal_session",
        _web_terminal_static_page_server,
        path="/static/session.html",
    ),
    VisualTarget(
        "web_terminal_safety",
        _web_terminal_static_page_server,
        path="/static/safety.html",
    ),
    VisualTarget("artifacts_gallery", _artifacts_server, path="/"),
    VisualTarget("ariel", _ariel_server, path="/"),
    VisualTarget("channel_finder", _channel_finder_server, path="/"),
    VisualTarget("tuning", _tuning_server, path="/"),
    VisualTarget("lattice_dashboard", _lattice_dashboard_server, path="/"),
    # Dispatch dashboard has no live dispatcher backend behind it here, so it
    # renders its genuine no-data empty state — a legitimate, stable baseline.
    VisualTarget("dispatch_dashboard", _dispatch_dashboard_server, path="/"),
]


# ---------------------------------------------------------------------------
# Function-scoped chromium fixture (see test_behavioral.py for rationale)
# ---------------------------------------------------------------------------


@pytest.fixture
def chromium_browser() -> Iterator[Browser]:
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
# Perceptual diff + baseline handling
# ---------------------------------------------------------------------------


def _perceptual_diff_ratio(
    img_a: Image.Image, img_b: Image.Image, pixel_threshold: int = PIXEL_THRESHOLD
) -> float:
    """Fraction of pixels whose max per-channel difference exceeds ``pixel_threshold``.

    A dimension mismatch (e.g. a baseline captured at a different viewport)
    is reported as maximally different rather than raising, so callers get a
    clean assertion failure instead of a numpy broadcast error.
    """
    if img_a.size != img_b.size:
        return 1.0
    arr_a = np.asarray(img_a.convert("RGB"), dtype=np.int16)
    arr_b = np.asarray(img_b.convert("RGB"), dtype=np.int16)
    diff = np.abs(arr_a - arr_b).max(axis=-1)
    return float((diff > pixel_threshold).mean())


def _assert_matches_baseline(name: str, png_bytes: bytes, regen: bool) -> None:
    """Compare (or record) one screenshot against its committed baseline.

    ``--regen-baselines`` always writes the baseline unconditionally. The
    first-time bootstrap write (no baseline committed yet) is Linux-only —
    writing one on any other platform would mint a non-authoritative,
    non-Linux-rendered PNG that the real Linux CI comparison would then
    (wrongly) diff against. Off-Linux with no baseline just skips with the
    same explicit notice as the ordinary off-Linux compare path below.
    """
    captured = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    assert captured.size == (VIEWPORT["width"], VIEWPORT["height"]), (
        f"{name}: captured screenshot size {captured.size} != viewport {VIEWPORT}"
    )

    baseline_path = BASELINES_DIR / f"{name}.png"
    is_linux = platform.system() == "Linux"

    if regen or (is_linux and not baseline_path.exists()):
        BASELINES_DIR.mkdir(parents=True, exist_ok=True)
        captured.save(baseline_path)
        if not baseline_path.exists():  # pragma: no cover - defensive
            raise AssertionError(f"{name}: failed to write baseline to {baseline_path}")
        return

    if not is_linux:
        print(f"[visual] {name}: {_DASHBOARD_HTML_ONLY_ON_LINUX_NOTE}")
        return

    baseline = Image.open(baseline_path).convert("RGB")
    ratio = _perceptual_diff_ratio(captured, baseline)
    assert ratio <= DIFF_RATIO_TOLERANCE, (
        f"{name}: {ratio:.4%} of pixels differ from {baseline_path.name} "
        f"(tolerance {DIFF_RATIO_TOLERANCE:.4%}). If this is an intentional "
        "visual change, regenerate with --regen-baselines on Linux (or via CI) "
        "and review the resulting diff."
    )


# ---------------------------------------------------------------------------
# The suite
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", TARGETS, ids=[t.name for t in TARGETS])
def test_visual_snapshot(tmp_path, chromium_browser, target: VisualTarget, pytestconfig) -> None:
    """Capture `target` in every theme and diff each against its baseline."""
    regen = bool(pytestconfig.getoption("--regen-baselines"))

    with target.server_factory(tmp_path) as base_url:
        for theme in THEMES:
            page = chromium_browser.new_page(viewport=VIEWPORT)
            try:
                page.goto(
                    f"{base_url}{target.path}?theme={theme}",
                    wait_until="domcontentloaded",
                    timeout=15_000,
                )
                if target.wait_selector:
                    expect(page.locator(target.wait_selector)).to_be_attached(timeout=10_000)
                if target.dismiss_welcome:
                    page.locator("#welcome-dismiss").click(timeout=15_000)
                # Let async init (panel health polling, SSE-driven layout,
                # font swaps) settle before the screenshot.
                page.wait_for_timeout(600)

                applied_theme = page.evaluate("document.documentElement.getAttribute('data-theme')")
                assert applied_theme == theme, (
                    f"{target.name}: expected data-theme={theme!r}, got {applied_theme!r}"
                )

                png_bytes = page.screenshot()
            finally:
                page.close()

            _assert_matches_baseline(f"{target.name}_{theme}", png_bytes, regen)
