"""Sync-Playwright runner that turns declarative recipes into committed PNGs.

Each :class:`~docs.screenshots.recipes.DocShot` names an *environment*, a set of
*themes*, a *viewport*, and either a single view (``path``) or one view per
:class:`~docs.screenshots.recipes.SubView`. This module boots the environment,
drives a real headless chromium to every (theme, view) combination, and writes
the resulting screenshots under :func:`output_dir` with filenames that keep byte
parity with the ``.rst`` figures that consume them.

Two environments are dispatched by :func:`run`:

* ``standalone_interface`` — resolve the recipe's ``app_factory``, boot it on a
  throwaway port via :func:`osprey.interfaces._serving.run_app_server`, and
  capture. Zero container, deterministic, the default target.
* ``tutorial_stack`` — delegated to :func:`capture_tutorial_stack`, which owns
  the full container lifecycle: it builds the ``control-assistant`` tutorial
  project, brings up Postgres detached, seeds ARIEL at the frozen anchor, and
  captures either the static ARIEL views or the agentic web-terminal hero. In an
  environment without that runtime (no ``osprey`` binary, no container engine,
  Postgres never ready) it raises :class:`ScreenshotSkip` so a ``--stack`` run
  degrades to a clean one-line notice instead of a traceback.

The whole run shares one browser; missing chromium/Playwright is reported as a
one-line skip rather than a traceback.
"""

from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from docs.screenshots import recipes

from osprey.interfaces._serving import free_port, run_app_server, wait_for_port

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from docs.screenshots.recipes import DocShot
    from playwright.sync_api import Browser

# Host TCP port Postgres publishes once ``osprey deploy up -d`` is healthy.
_POSTGRES_PORT = 5432

# Project name for the throwaway tutorial build. ``osprey build <name> --preset
# control-assistant -o <dir>`` renders the project at ``<dir>/<name>``.
_TUTORIAL_PROJECT_NAME = "docshots-tutorial"

# Floor (bytes) below which a captured hero PNG is treated as too trivial to be a
# real screenshot when Pillow is unavailable to inspect its pixels.
_MIN_HERO_PNG_BYTES = 1024


class ScreenshotSkip(Exception):
    """Raised when capture cannot proceed for a benign, expected reason.

    Used for absent optional dependencies (Playwright, the chromium binary) and
    for the not-yet-available ``tutorial_stack`` provider, so callers can print a
    clear one-line notice instead of surfacing a traceback.
    """


# ---------------------------------------------------------------------------
# Output location, versioning, and the JSON manifest
# ---------------------------------------------------------------------------


def output_dir() -> Path:
    """Directory the committed screenshots (and the manifest) live in."""
    return Path(__file__).parent.parent / recipes.OUTPUT_SUBDIR


def osprey_version() -> str:
    """OSPREY version string, or ``"0+unknown"`` if undeterminable.

    Prefers ``osprey.__version__`` (the canonical source of truth in
    ``src/osprey/__init__.py``); falls back to the installed distribution
    metadata (the distribution is named ``osprey-framework``, not ``osprey``).
    """
    try:
        import osprey

        return osprey.__version__
    except (ImportError, AttributeError):
        pass

    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("osprey-framework")
    except PackageNotFoundError:
        return "0+unknown"


def stamp_manifest(name: str, kind: str) -> None:
    """Upsert one ``name`` entry in the capture manifest.

    The manifest maps each recipe ``name`` to the ``osprey`` version, the UTC
    capture instant, and the recipe ``kind``. Existing entries are overwritten;
    a missing or malformed manifest is treated as empty. The file is rewritten
    with sorted keys, two-space indent, and a trailing newline.
    """
    manifest_path = output_dir() / recipes.MANIFEST_NAME
    try:
        data = json.loads(manifest_path.read_text())
        if not isinstance(data, dict):
            data = {}
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}

    data[name] = {
        "osprey_version": osprey_version(),
        "captured_utc": datetime.now(UTC).isoformat(),
        "kind": kind,
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Headless chromium context (re-implemented, not imported from tests/)
# ---------------------------------------------------------------------------


@contextmanager
def chromium_context() -> Iterator[Browser]:
    """Yield a headless chromium ``Browser``, stopping Playwright on every exit.

    Raises :class:`ScreenshotSkip` (never a traceback) when Playwright is not
    installed or the chromium binary is unavailable. ``sync_playwright().start()``
    spins an asyncio loop on the main thread, so it is stopped on *every* exit
    path — including the skip taken when the binary is absent.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise ScreenshotSkip("playwright is not installed") from exc

    pw = sync_playwright().start()
    try:
        browser = pw.chromium.launch(headless=True)
    except Exception as exc:
        pw.stop()
        raise ScreenshotSkip(f"chromium binary not available: {exc}") from exc

    try:
        yield browser
    finally:
        browser.close()
        pw.stop()


# ---------------------------------------------------------------------------
# View expansion and filename shaping
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _View:
    """One resolved (path, hash, activation, waits, output) view within a shot."""

    path: str
    hash: str
    click_selector: str | None
    wait_selectors: tuple[str, ...]
    out: str


def _views(shot: DocShot) -> list[_View]:
    """Expand a recipe into its concrete views (implicit single view or subviews)."""
    if not shot.subviews:
        waits = tuple(s for s in (shot.wait_selector,) if s)
        return [_View(shot.path, "", None, waits, shot.name)]

    views: list[_View] = []
    for sv in shot.subviews:
        if sv.anchor.startswith("#"):
            hash_frag, click = sv.anchor, None
        else:
            hash_frag, click = "", sv.anchor
        waits = tuple(s for s in (shot.wait_selector, sv.wait_selector) if s)
        views.append(_View(shot.path, hash_frag, click, waits, sv.out))
    return views


def _output_filename(out: str, theme: str, n_themes: int) -> str:
    """Filename for one view/theme: no theme suffix for single-theme recipes."""
    if n_themes == 1:
        return f"{out}.png"
    return f"{out}_{theme}.png"


def _build_url(base_url: str, path: str, theme: str, hash_frag: str) -> str:
    """Assemble ``base + path + ?theme=... + #hash`` (query before hash)."""
    separator = "&" if "?" in path else "?"
    return f"{base_url}{path}{separator}theme={theme}{hash_frag}"


# ---------------------------------------------------------------------------
# Capture of a single running standalone/stack environment
# ---------------------------------------------------------------------------


def capture_shot(browser: Browser, base_url: str, shot: DocShot) -> list[Path]:
    """Capture one recipe's PNG(s) against a live ``browser`` + ``base_url``.

    Iterates every theme and every view, driving a fresh page per capture, and
    returns the written file paths. The full-page mode takes a *viewport*
    screenshot (not Playwright's scroll capture) so dimensions match the
    committed images.
    """
    from playwright.sync_api import expect

    dest_dir = output_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)

    n_themes = len(shot.themes)
    views = _views(shot)
    written: list[Path] = []

    # Element crops of small widgets render at 2x device scale so the committed
    # PNG stays crisp when displayed; full-page/viewport shots stay at 1x so
    # their pixel dimensions match the committed images (e.g. ARIEL's 1280x900).
    device_scale = 2 if shot.capture_mode == "element" else 1

    for theme in shot.themes:
        for view in views:
            page = browser.new_page(
                viewport={"width": shot.viewport[0], "height": shot.viewport[1]},
                device_scale_factor=device_scale,
            )
            try:
                url = _build_url(base_url, view.path, theme, view.hash)
                page.goto(url, wait_until="domcontentloaded", timeout=15_000)

                if view.click_selector:
                    page.locator(view.click_selector).click(timeout=15_000)

                for selector in view.wait_selectors:
                    expect(page.locator(selector)).to_be_attached(timeout=10_000)

                if shot.dismiss_welcome:
                    page.locator("#welcome-dismiss").click(timeout=15_000)

                # Let the theme swap and any async init settle before shooting.
                page.wait_for_timeout(600)

                if shot.capture_mode == "element":
                    png = page.locator(shot.element_selector).screenshot()
                else:
                    png = page.screenshot()

                dest = dest_dir / _output_filename(view.out, theme, n_themes)
                dest.write_bytes(png)
                written.append(dest)
            finally:
                page.close()

    return written


# ---------------------------------------------------------------------------
# Environment providers
# ---------------------------------------------------------------------------


def _resolve_app_factory(dotted: str) -> Callable[[], object]:
    """Resolve a ``"module.path:callable"`` dotted path to the callable."""
    module_path, _, attr = dotted.partition(":")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def _capture_standalone(browser: Browser, shot: DocShot) -> list[Path]:
    """Boot a ``standalone_interface`` recipe's app and capture it."""
    factory = _resolve_app_factory(shot.app_factory)
    app = factory()
    with run_app_server(app) as base_url:
        return capture_shot(browser, base_url, shot)


def _run_stack_step(cmd: list[str], *, cwd: Path | None, what: str) -> None:
    """Run one project-scoped lifecycle command, mapping failure to a skip.

    A missing ``osprey`` binary (``FileNotFoundError``) or a non-zero exit is
    reported as :class:`ScreenshotSkip` so the stack degrades gracefully rather
    than surfacing a traceback. Output is captured (never streamed) so a skipped
    ``--stack`` run stays quiet. Only ever runs the exact, project-scoped command
    it is given — never a system-wide or destructive one.
    """
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ScreenshotSkip(f"osprey CLI unavailable to {what}: {exc}") from exc
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip().splitlines()[-1:] or [""]
        raise ScreenshotSkip(f"failed to {what} (exit {result.returncode}): {tail[0]}")


@contextmanager
def _tutorial_stack(*, artifact_port: int) -> Iterator[Path]:
    """Build the tutorial project, bring up Postgres, seed ARIEL, yield the dir.

    Lifecycle order (each step's failure degrades to :class:`ScreenshotSkip`):
    make a temp build root, ``osprey build <name> --preset control-assistant``
    into it (``--skip-deps``: the capture drives ``osprey``/ARIEL from the current
    environment, so the built project needs no venv of its own) with the
    artifact-server port pinned, ``osprey deploy up -d`` (detached — the
    non-detached form would ``execvpe`` away the runner), wait for Postgres on
    :data:`_POSTGRES_PORT` *before* seeding, then ``osprey sim apply nominal``
    frozen to :data:`recipes.ANCHOR`. Yields the rendered project directory
    (``<build_root>/<name>``). The ``finally`` block always tears the project down
    with the project-scoped ``osprey deploy down`` and removes the build root — it
    never issues any prune, volume, or system-wide command.
    """
    try:
        build_root = Path(tempfile.mkdtemp(prefix="osprey-docshot-"))
    except OSError as exc:
        raise ScreenshotSkip(f"could not create a temp project dir: {exc}") from exc

    project_dir = build_root / _TUTORIAL_PROJECT_NAME
    try:
        _run_stack_step(
            [
                "osprey",
                "build",
                _TUTORIAL_PROJECT_NAME,
                "--preset",
                "control-assistant",
                "-o",
                str(build_root),
                "--skip-deps",
                "--set",
                f"config.artifact_server.port={artifact_port}",
            ],
            cwd=None,
            what="build the control-assistant tutorial",
        )
        _run_stack_step(
            ["osprey", "deploy", "up", "-d"],
            cwd=project_dir,
            what="bring up Postgres",
        )
        try:
            wait_for_port(_POSTGRES_PORT, timeout=120.0)
        except RuntimeError as exc:
            raise ScreenshotSkip(f"Postgres did not become ready: {exc}") from exc
        _run_stack_step(
            ["osprey", "sim", "apply", "nominal", "--yes", "--now", recipes.ANCHOR],
            cwd=project_dir,
            what="seed ARIEL",
        )
        yield project_dir
    finally:
        try:
            subprocess.run(
                ["osprey", "deploy", "down"],
                cwd=str(project_dir),
                capture_output=True,
                check=False,
            )
        except OSError:
            pass
        shutil.rmtree(build_root, ignore_errors=True)


def _png_dimensions(png_bytes: bytes) -> tuple[int, int]:
    """Return ``(width, height)`` read from a PNG's IHDR chunk (no decode)."""
    if len(png_bytes) < 24:
        raise AssertionError("hero capture is too short to contain a PNG header")
    width = int.from_bytes(png_bytes[16:20], "big")
    height = int.from_bytes(png_bytes[20:24], "big")
    return width, height


def assert_hero_structural(png_bytes: bytes, viewport: tuple[int, int]) -> None:
    """Assert an agentic hero PNG is non-blank and matches ``viewport``.

    Raises :class:`AssertionError` unless ``png_bytes`` is a PNG whose IHDR
    width and height equal ``viewport`` and whose pixels are not a single uniform
    color. When Pillow is available the blank check inspects the decoded pixels;
    otherwise it falls back to a minimum byte-size floor
    (:data:`_MIN_HERO_PNG_BYTES`). This is the structural success criterion for
    an agentic capture — a pure function so it can be unit-tested on fixtures.
    """
    if not png_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        raise AssertionError("hero capture is not a PNG")

    width, height = _png_dimensions(png_bytes)
    if (width, height) != (viewport[0], viewport[1]):
        raise AssertionError(f"hero PNG is {width}x{height}, expected {viewport[0]}x{viewport[1]}")

    try:
        from io import BytesIO

        from PIL import Image
    except ImportError:
        if len(png_bytes) < _MIN_HERO_PNG_BYTES:
            raise AssertionError(
                f"hero PNG is only {len(png_bytes)} bytes; expected a real capture"
            ) from None
        return

    with Image.open(BytesIO(png_bytes)) as img:
        extrema = img.getextrema()
    bands = extrema if isinstance(extrema[0], tuple) else (extrema,)
    if all(lo == hi for lo, hi in bands):
        raise AssertionError("hero PNG is a single uniform color (blank capture)")


def _wait_for_artifact(artifact_port: int, needle: str, *, timeout: float) -> None:
    """Poll the artifact server until an ``artifact_type`` contains ``needle``.

    Raises :class:`TimeoutError` with a clear message if no matching artifact
    appears within ``timeout`` seconds, so a stalled agent run fails fast instead
    of hanging forever.
    """
    url = f"http://127.0.0.1:{artifact_port}/api/artifacts"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5.0) as resp:  # noqa: S310 (loopback only)
                payload = json.loads(resp.read().decode())
        except (urllib.error.URLError, json.JSONDecodeError, OSError):
            payload = {}
        for artifact in payload.get("artifacts", []):
            if needle in (artifact.get("artifact_type") or ""):
                return
        time.sleep(1.0)
    raise TimeoutError(
        f"no artifact with artifact_type containing {needle!r} appeared on "
        f"port {artifact_port} within {timeout}s"
    )


def _capture_agentic(
    browser: Browser, project_dir: Path, artifact_port: int, shot: DocShot
) -> list[Path]:
    """Drive the live web terminal to produce the agentic hero screenshot(s).

    Launches a detached ``osprey web`` bound to a free port, then for each theme:
    opens the UI with ``?theme=``, dismisses the welcome modal, answers the PTY
    trust prompt, types the operator prompt, waits (bounded) for a matching
    artifact, reveals the artifacts panel, opens the plot, and screenshots the
    viewport. Every launched process and page is torn down in ``finally``; the
    web server is stopped with the project-scoped ``osprey web stop --project``.
    """
    web_port = free_port()
    try:
        proc = subprocess.Popen(
            [
                "osprey",
                "web",
                "--project",
                str(project_dir),
                "--detach",
                "--port",
                str(web_port),
            ]
        )
    except FileNotFoundError as exc:
        raise ScreenshotSkip(f"osprey CLI unavailable to launch web terminal: {exc}") from exc

    try:
        try:
            wait_for_port(web_port, timeout=90.0)
        except RuntimeError as exc:
            raise ScreenshotSkip(f"web terminal did not become ready: {exc}") from exc

        base_url = f"http://127.0.0.1:{web_port}"
        dest_dir = output_dir()
        dest_dir.mkdir(parents=True, exist_ok=True)
        n_themes = len(shot.themes)
        written: list[Path] = []

        for theme in shot.themes:
            page = browser.new_page(
                viewport={"width": shot.viewport[0], "height": shot.viewport[1]}
            )
            try:
                page.goto(
                    f"{base_url}/?theme={theme}",
                    wait_until="domcontentloaded",
                    timeout=30_000,
                )
                page.locator("#welcome-dismiss").click(timeout=30_000)

                # The Claude-Code trust prompt lives in the PTY, not the DOM:
                # focus the terminal and answer it, then type the operator prompt.
                page.locator("#terminal-container").click(timeout=30_000)
                page.keyboard.press("Enter")
                page.keyboard.type(shot.prompt or "")
                page.keyboard.press("Enter")

                _wait_for_artifact(artifact_port, shot.wait_for or "", timeout=240.0)

                page.locator('button[data-panel-id="artifacts"]').click(timeout=30_000)
                panel = page.frame_locator('iframe.panel-iframe[data-panel-id="artifacts"]')
                panel.locator(".gallery-card").first.click(timeout=30_000)

                # Let the preview render before shooting the viewport.
                page.wait_for_timeout(1_000)
                png = page.screenshot()
                assert_hero_structural(png, shot.viewport)

                dest = dest_dir / _output_filename(shot.name, theme, n_themes)
                dest.write_bytes(png)
                written.append(dest)
            finally:
                page.close()

        return written
    finally:
        try:
            subprocess.run(
                ["osprey", "web", "stop", "--project", str(project_dir)],
                capture_output=True,
                check=False,
            )
        except OSError:
            pass
        try:
            proc.terminate()
        except (OSError, ValueError):
            pass


def capture_tutorial_stack(
    browser_factory: Callable[[], Browser], shot: DocShot, *, agentic: bool
) -> list[Path]:
    """Capture a ``tutorial_stack`` recipe (container lifecycle owner).

    ``browser_factory`` is a zero-argument callable returning a live browser to
    reuse for the capture. Builds and seeds the tutorial stack (via
    :func:`_tutorial_stack`), then dispatches on ``shot.kind``: ``"static"``
    boots the ARIEL app on a throwaway port and reuses :func:`capture_shot`;
    ``"agentic"`` drives the live web terminal via :func:`_capture_agentic`
    (only when ``agentic`` is set). Raises :class:`ScreenshotSkip` when the
    container runtime is unavailable so a ``--stack`` run degrades gracefully.
    """
    artifact_port = free_port()
    with _tutorial_stack(artifact_port=artifact_port) as project_dir:
        browser = browser_factory()

        if shot.kind == "agentic":
            if not agentic:
                raise ScreenshotSkip(f"{shot.name}: agentic recipe requires the --agentic flag")
            try:
                return _capture_agentic(browser, project_dir, artifact_port, shot)
            except TimeoutError as exc:
                # The live agent never produced the expected artifact (e.g. no
                # provider credentials in this environment) — degrade to a clear
                # skip rather than a traceback.
                raise ScreenshotSkip(
                    f"{shot.name}: agent did not produce a {shot.wait_for!r} artifact in time"
                ) from exc

        from osprey.interfaces.ariel.app import create_app

        app = create_app(config_path=str(project_dir / "config.yml"))
        with run_app_server(app) as ariel_url:
            return capture_shot(browser, ariel_url, shot)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(shots: list[DocShot], *, stack: bool = False, agentic: bool = False) -> None:
    """Capture the selected recipes, sharing one headless browser for the run.

    ``standalone_interface`` recipes are booted and captured directly;
    ``tutorial_stack`` recipes are delegated to :func:`capture_tutorial_stack`
    and skipped per-recipe (with a clear notice) where its runtime is absent.
    Absent chromium/Playwright skips the whole run gracefully. One manifest
    entry is stamped per successfully captured recipe ``name``.
    """
    if not shots:
        print("No screenshot recipes selected; nothing to capture.")
        return

    written: list[Path] = []
    try:
        with chromium_context() as browser:
            for shot in shots:
                if shot.environment == "standalone_interface":
                    paths = _capture_standalone(browser, shot)
                else:
                    try:
                        paths = capture_tutorial_stack(lambda: browser, shot, agentic=agentic)
                    except ScreenshotSkip as exc:
                        print(f"skipped {shot.name}: {exc}", file=sys.stderr)
                        continue

                stamp_manifest(shot.name, shot.kind)
                written.extend(paths)
                print(f"captured {shot.name}: {len(paths)} file(s)")
    except ScreenshotSkip as exc:
        print(f"screenshot capture skipped: {exc}", file=sys.stderr)
        return

    print(f"Wrote {len(written)} screenshot file(s) to {output_dir()}.")
