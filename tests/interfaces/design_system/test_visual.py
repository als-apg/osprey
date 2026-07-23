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
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

import osprey.interfaces.design_system as design_system_pkg
from tests.interfaces.conftest import _apply_all, _run_app_server

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from contextlib import AbstractContextManager

# ---------------------------------------------------------------------------
# Playwright availability guard
# ---------------------------------------------------------------------------

try:
    from playwright.sync_api import expect

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
# Helpers: patch aggregation (ports/uvicorn lifecycle live in conftest.py)
# ---------------------------------------------------------------------------


@contextmanager
def _hub_live_server(workspace_dir: Path, artifact_server_url: str) -> Iterator[str]:
    """Launch a real web-terminal (hub) server pointed at a real artifacts backend.

    Mirrors ``test_behavioral.py``'s ``_hub_live_server``, simplified to the one
    configuration this suite needs: artifacts enabled and genuinely reachable
    (not the dead fake URL some other suites use), so the captured screenshot
    shows real embedded content instead of a proxy error page.
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
            "osprey.interfaces.web_terminal.app._launch_artifact_server",
            side_effect=lambda a: setattr(a.state, "artifact_server_url", artifact_server_url),
        ),
    ]
    with _apply_all(patches):
        from osprey.interfaces.web_terminal.app import create_app

        app = create_app(shell_command=["echo", "hello"])
        with _run_app_server(app) as base_url:
            yield base_url


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
# The multi-user grouped landing page has no standalone FastAPI app either:
# in production it's a static file (nginx/landing.html) produced by
# render_web_terminals() and served by nginx, with no app route behind it.
# Stand one up the same way the dispatch dashboard does — a minimal real app
# serving the genuine render output — so the baseline exercises the real
# deploy render path, not a hand-built HTML fixture. Two users with distinct
# personas (alice->operator, bob->physicist) so both persona sublabels appear.
# ---------------------------------------------------------------------------

_MULTI_USER_LANDING_CONFIG = {
    "facility": {"name": "Demo Control Room", "prefix": "demo"},
    "registry": {"url": "registry.example.org/demo"},
    "deploy": {"fqdn": "demo.example.org"},
    "modules": {
        "web_terminals": {
            "nginx_port": 8080,
            "web_base_port": 8100,
            "artifact_base_port": 8200,
            "ariel_base_port": 8300,
            "lattice_base_port": 8400,
            "default_persona": "operator",
            "personas": {
                "operator": {"project": "demo-operator"},
                "physicist": {"project": "demo-physicist"},
            },
            "users": [
                {"name": "alice", "index": 0, "persona": "operator"},
                {"name": "bob", "index": 1, "persona": "physicist"},
            ],
        }
    },
}


def _create_multi_user_landing_app() -> FastAPI:
    from osprey.deployment.web_terminals.render import render_web_terminals

    landing_html = render_web_terminals(_MULTI_USER_LANDING_CONFIG)["nginx/landing.html"]

    app = FastAPI()

    # The landing page is fully self-contained (inline CSS, no external assets),
    # so — unlike the dispatch dashboard — it needs no /design-system static mount.
    @app.get("/", response_class=HTMLResponse)
    async def root() -> str:
        return landing_html

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
    # The web-terminal hub now boots a dockview workspace (dock-workspace.js)
    # whose default artifacts panel docks as an overlay iframe (dock-iframe.js).
    # Unlike the pre-dock split, the grid and its overlay settle a beat after the
    # rail renders, so ``dock_shell`` targets wait for the dockview grid AND the
    # auto-docked artifacts overlay iframe to be on screen before the screenshot —
    # otherwise the baseline can capture a half-built (empty) grid.
    dock_shell: bool = False
    # UI-mode axis. Mode-aware surfaces capture the full theme x mode matrix
    # (baseline ``{name}_{theme}_{mode}.png``); ``(None,)`` keeps a surface on
    # the theme-only pair with the original ``{name}_{theme}.png`` names (the
    # hub's static session/safety pages have no mode-dependent layout).
    modes: tuple[str | None, ...] = (None,)


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


def _lattice_dashboard_server(tmp_path: Path):
    from osprey.interfaces.lattice_dashboard.app import create_app

    return _run_app_server(create_app(workspace_root=tmp_path / "lattice_ws"))


def _dispatch_dashboard_server(tmp_path: Path):
    return _run_app_server(_create_dispatch_dashboard_app())


def _multi_user_landing_server(tmp_path: Path):
    return _run_app_server(_create_multi_user_landing_app())


def _okf_panel_server(tmp_path: Path):
    from osprey.interfaces.okf_panel.app import create_app

    # Reuse the okf_panel suite's on-disk fixture bundle so the baseline shows
    # a populated tree + document instead of the guarded not-configured shell.
    bundle = Path(__file__).parents[1] / "okf_panel" / "fixtures" / "bundle"
    return _run_app_server(create_app(str(bundle)))


# ---------------------------------------------------------------------------
# Scan panels (Phase-6 operator interfaces): unlike every other target above,
# the sidecar has no ``create_app()`` factory — it's a single module-level
# FastAPI singleton (see ``osprey.services.bluesky_panels.app``) that mounts both
# panel bundles (plan/results) plus the shared design-system
# assets in one process. Import the app object directly and hand it to
# ``_run_app_server`` the same way the other targets hand it a freshly
# constructed app. No bridge is running behind it here, so each panel renders
# its genuine no-bridge shell/empty state — a legitimate, stable baseline
# (mirrors the dispatch-dashboard no-data rationale above).
# ---------------------------------------------------------------------------


def _bluesky_panels_server(tmp_path: Path):
    from osprey.services.bluesky_panels.app import app as bluesky_panels_app

    return _run_app_server(bluesky_panels_app)


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


# The full expert/simple pair every mode-aware surface captures.
MODES: tuple[str, ...] = ("expert", "simple")

TARGETS: list[VisualTarget] = [
    VisualTarget(
        "web_terminal_hub",
        _web_terminal_hub_server,
        path="/",
        dismiss_welcome=True,
        # Scope to the rail button: overlay iframes now also carry
        # data-panel-id="artifacts" in the docked shell, so the bare attribute
        # selector is ambiguous.
        wait_selector='button.panel-rail-button[data-panel-id="artifacts"]',
        dock_shell=True,
        modes=MODES,
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
    VisualTarget("artifacts_gallery", _artifacts_server, path="/", modes=MODES),
    VisualTarget("ariel", _ariel_server, path="/", modes=MODES),
    VisualTarget("channel_finder", _channel_finder_server, path="/", modes=MODES),
    # D14/D15 regression guard: embedded mode must hide the standalone logo +
    # theme switcher (component self-hides via body.embedded) while keeping
    # the pipeline switcher + nav usable — see channel-finder-narrowing.
    VisualTarget(
        "channel_finder_embedded",
        _channel_finder_server,
        path="/?embedded=true",
        modes=MODES,
    ),
    VisualTarget("lattice_dashboard", _lattice_dashboard_server, path="/", modes=MODES),
    VisualTarget(
        "okf_panel",
        _okf_panel_server,
        path="/",
        wait_selector="#tree",
        modes=MODES,
    ),
    # Dispatch dashboard has no live dispatcher backend behind it here, so it
    # renders its genuine no-data empty state — a legitimate, stable baseline.
    VisualTarget("dispatch_dashboard", _dispatch_dashboard_server, path="/", modes=MODES),
    # Scan panels (Phase-6): mounted at /plan, /results by the
    # sidecar (see ``_PANEL_MOUNTS`` in ``osprey.services.bluesky_panels.app``);
    # each wait_selector is a static top-level element present in the shell's
    # initial markup (not injected by JS), so it attaches even though no
    # bridge is running behind this sidecar and every panel's fetch fails.
    VisualTarget(
        "scan_panel_plan",
        _bluesky_panels_server,
        path="/plan/",
        wait_selector="#plan-tree",
        modes=MODES,
    ),
    VisualTarget(
        "scan_panel_results",
        _bluesky_panels_server,
        path="/results/",
        wait_selector="#run-picker",
        modes=MODES,
    ),
]


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
    """Capture `target` in every theme (x mode, when mode-aware) and diff each."""
    regen = bool(pytestconfig.getoption("--regen-baselines"))

    with target.server_factory(tmp_path) as base_url:
        for theme in THEMES:
            for mode in target.modes:
                page = chromium_browser.new_page(viewport=VIEWPORT)
                try:
                    # target.path may already carry its own query string (e.g.
                    # channel_finder_embedded's "/?embedded=true"), so `theme`
                    # must be appended with "&" rather than blindly with "?".
                    separator = "&" if "?" in target.path else "?"
                    url = f"{base_url}{target.path}{separator}theme={theme}"
                    if mode is not None:
                        url += f"&mode={mode}"
                    page.goto(url, wait_until="domcontentloaded", timeout=15_000)
                    if target.wait_selector:
                        expect(page.locator(target.wait_selector)).to_be_attached(timeout=10_000)
                    if target.dismiss_welcome:
                        page.locator("#welcome-dismiss").click(timeout=15_000)
                    if target.dock_shell:
                        # The dockview grid and the auto-docked artifacts overlay
                        # iframe settle a beat after the rail renders; wait for both
                        # so the baseline captures the built default layout, not a
                        # half-constructed (empty) grid.
                        expect(page.locator(".dv-groupview").first).to_be_visible(timeout=10_000)
                        expect(
                            page.locator('.dock-iframe-overlay iframe[data-panel-id="artifacts"]')
                        ).to_be_visible(timeout=10_000)
                    # Let async init (panel health polling, SSE-driven layout,
                    # font swaps) settle before the screenshot.
                    page.wait_for_timeout(600)

                    applied_theme = page.evaluate(
                        "document.documentElement.getAttribute('data-theme')"
                    )
                    assert applied_theme == theme, (
                        f"{target.name}: expected data-theme={theme!r}, got {applied_theme!r}"
                    )
                    if mode is not None:
                        applied_mode = page.evaluate(
                            "document.documentElement.getAttribute('data-ui-mode')"
                        )
                        assert applied_mode == mode, (
                            f"{target.name}: expected data-ui-mode={mode!r}, got {applied_mode!r}"
                        )

                    png_bytes = page.screenshot()
                finally:
                    page.close()

                suffix = f"_{theme}" if mode is None else f"_{theme}_{mode}"
                _assert_matches_baseline(f"{target.name}{suffix}", png_bytes, regen)


@pytest.mark.parametrize("theme", THEMES)
def test_visual_multi_user_landing(tmp_path, chromium_browser, theme, pytestconfig) -> None:
    """Capture the multi-user grouped landing page in each theme and diff its baseline.

    The deploy-rendered landing page is not one of the app ``TARGETS`` because it
    themes differently: in production it's a static nginx-served file with no
    in-page JS, so it has no ``?theme=`` query hook or ``data-theme`` attribute and
    themes purely off the ``prefers-color-scheme`` media query. Its theme is
    therefore driven by Playwright's ``color_scheme`` emulation (the real signal
    the page responds to) rather than the query param the app targets use. Baseline
    handling is otherwise identical — shared ``_assert_matches_baseline`` — so it
    is Linux-authoritative and skips the byte-compare off-Linux like every target.
    """
    regen = bool(pytestconfig.getoption("--regen-baselines"))

    with _multi_user_landing_server(tmp_path) as base_url:
        page = chromium_browser.new_page(viewport=VIEWPORT, color_scheme=theme)
        try:
            page.goto(base_url, wait_until="domcontentloaded", timeout=15_000)
            # Both persona-badged user cards must be present before the shot, so
            # the baseline is guaranteed to show the operator/physicist sublabels.
            expect(page.locator(".landing-card-sublabel")).to_have_count(2, timeout=10_000)
            page.wait_for_timeout(300)
            png_bytes = page.screenshot()
        finally:
            page.close()

    _assert_matches_baseline(f"multi_user_landing_{theme}", png_bytes, regen)
