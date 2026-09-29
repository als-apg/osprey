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

The whole run shares one browser; missing chromium/Playwright, or a Playwright driver
that does not start, is reported as a one-line skip rather than a traceback.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from docs.screenshots import recipes

from osprey.interfaces._serving import (
    authorize_browser_context,
    free_port,
    run_app_server,
    wait_for_port,
)
from osprey.interfaces.vendor import verify_all as verify_vendor_bundles
from osprey.interfaces.web_auth import OPERATOR_SECRET_ENV, mint_secret
from osprey.port_layout import default_port

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from docs.screenshots.recipes import DocShot
    from playwright.sync_api import Browser

# Host TCP port Postgres publishes once ``osprey up -d`` is healthy. This module
# builds its own throwaway control-assistant deployment below and never sets
# ``deployment.port_base``, so the layout's default base is the right one here.
_POSTGRES_PORT = default_port("postgres")

# Directory name for the throwaway tutorial deployment repo. ``osprey init
# <dir>/<name> --preset control-assistant`` creates the repo; ``osprey build``
# run inside it renders ``<dir>/<name>/build``.
_TUTORIAL_PROJECT_NAME = "docshots-tutorial"

# Provider shorthand for the tutorial deployment's agent, for a host whose
# credentials belong to a provider other than the preset's default. Unset keeps
# the preset's own provider.
PROVIDER_ENV = "OSPREY_DOCSHOTS_PROVIDER"
#: When set, a directory each live session's transcripts are copied into before
#: its config dir is removed, so a failed take can be read afterwards.
KEEP_ENV = "OSPREY_DOCSHOTS_KEEP"

# Floor (bytes) below which a captured hero PNG is treated as too trivial to be a
# real screenshot when Pillow is unavailable to inspect its pixels.
_MIN_HERO_PNG_BYTES = 1024


class ScreenshotSkip(Exception):
    """Raised when capture cannot proceed for a benign, expected reason.

    Used for absent optional dependencies (Playwright, a Playwright driver that
    starts, the chromium binary) and
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


def _driver_start_failure(exc: AttributeError | OSError) -> str:
    """Return the skip reason for a Playwright driver that did not start."""
    if isinstance(exc, OSError):
        return f"playwright driver did not start: {exc}"
    reason = "playwright driver exited before its handshake"
    node_options = os.environ.get("NODE_OPTIONS")
    if node_options:
        # The driver is a node process that inherits this environment, and NODE_OPTIONS
        # acts before any Playwright code runs, so it is the host setting worth naming.
        reason += f" (NODE_OPTIONS={node_options!r})"
    return reason


@contextmanager
def chromium_context() -> Iterator[Browser]:
    """Yield a headless chromium ``Browser``, stopping Playwright on every exit.

    Raises :class:`ScreenshotSkip` (never a traceback) when Playwright is not
    installed, its driver does not start, or the chromium binary is unavailable.
    ``sync_playwright().start()`` spins an asyncio loop on the main thread, so it
    is stopped on *every* exit path — including the skip taken when the binary is
    absent.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise ScreenshotSkip("playwright is not installed") from exc

    manager = sync_playwright()
    try:
        pw = manager.start()
    # The sync API reports a driver that exited before its handshake as a missing
    # ``_playwright`` attribute and one it cannot spawn as an OSError; only those two are
    # an absent browser stack. ``__exit__`` is best-effort: an unspawned driver has no pipe.
    except (AttributeError, OSError) as exc:
        if isinstance(exc, AttributeError) and exc.name != "_playwright":
            raise
        with suppress(Exception):
            manager.__exit__(None, None, None)
        raise ScreenshotSkip(_driver_start_failure(exc)) from exc

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
                # The interface app is gated by WebAuthMiddleware; a bare page is
                # refused with 401 and never renders. This runner serves the app
                # in-process (via run_app_server), so authorize the context
                # directly with a session cookie the in-process gate accepts.
                authorize_browser_context(page.context)
                url = _build_url(base_url, view.path, theme, view.hash)
                page.goto(url, wait_until="domcontentloaded", timeout=15_000)

                if view.click_selector:
                    page.locator(view.click_selector).click(timeout=15_000)

                for selector in view.wait_selectors:
                    expect(page.locator(selector)).to_be_attached(timeout=10_000)

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


def _capture_static_page(browser: Browser, shot: DocShot) -> list[Path]:
    """Serve a ``static_page`` recipe's committed HTML file and capture it.

    The file's parent directory is served by a throwaway threaded
    ``http.server`` on a free port (an ``http://`` origin, because chromium
    treats ``file://`` query parameters — the theme switch — inconsistently),
    and :func:`capture_shot` then drives the usual theme × view matrix against
    the file's own URL. The page maps ``?theme=`` onto its CSS itself.
    """
    import http.server
    import threading
    from dataclasses import replace
    from functools import partial

    repo_root = Path(__file__).parent.parent.parent
    source = repo_root / shot.source_file
    if not source.is_file():
        raise ScreenshotSkip(f"source file not found: {shot.source_file}")

    class _QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args: object) -> None:  # keep the run quiet
            pass

    handler = partial(_QuietHandler, directory=str(source.parent))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_address[1]}"
        return capture_shot(browser, base_url, replace(shot, path=f"/{source.name}"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _capture_hermetic_hub(browser: Browser, shot: DocShot) -> list[Path]:
    """Boot the contact sheet's hermetic web terminal and capture a recipe from it.

    One stack serves every theme. Each theme loads in ``shot.hub_mode`` and is
    driven into ``shot.stage`` before the shot, through the same routine the
    contact sheet uses, so the doc image and the review card cannot drift apart.
    """
    from docs.screenshots.contact_sheet import capture_hub_view, hermetic_hub

    dest_dir = output_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    with hermetic_hub() as hub:
        for theme in shot.themes:
            dest = dest_dir / _output_filename(shot.name, theme, len(shot.themes))
            capture_hub_view(browser, hub, theme, shot.hub_mode, dest, stage=shot.stage)
            paths.append(dest)
    return paths


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


def rendered_artifact_port(project_dir: Path) -> int:
    """The artifact-server port ``osprey build`` rendered for *project_dir*.

    The build derives it from ``deployment.port_base`` and writes it to
    ``build/config.yml``; the capture talks to whatever port that says.

    Raises:
        ScreenshotSkip: If the rendered config is missing or names no port.
    """
    import yaml

    config_path = project_dir / "build" / "config.yml"
    try:
        config = yaml.safe_load(config_path.read_text()) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ScreenshotSkip(f"no artifact_server.port: cannot read {config_path}: {exc}") from exc
    port = (config.get("artifact_server") or {}).get("port")
    if not isinstance(port, int) or isinstance(port, bool):
        raise ScreenshotSkip(f"no artifact_server.port rendered in {config_path}")
    return port


def _opus_model_id(project_dir: Path) -> str:
    """The Opus model id the tutorial repo's provider serves.

    Read from the repo's own ``profile.yml`` (which provider answers) and
    ``providers.yml`` (the ids that provider serves), the files ``osprey init``
    writes. Every capture on the tutorial stack runs on Opus, the stack
    screenshots and the demo video alike, so a provider that serves no Opus
    model is a skip rather than a silent fall back to another model.
    """
    import yaml

    try:
        profile = yaml.safe_load((project_dir / "profile.yml").read_text(encoding="utf-8"))
        catalog = yaml.safe_load((project_dir / "providers.yml").read_text(encoding="utf-8"))
        provider = profile["provider"]
        models = catalog["providers"][provider]["models"]
    except (OSError, yaml.YAMLError, KeyError, TypeError) as exc:
        raise ScreenshotSkip(f"could not read the tutorial's provider catalog: {exc!r}") from exc
    for model in models:
        if "opus" in str(model):
            return str(model)
    raise ScreenshotSkip(f"provider {provider!r} serves no Opus model: {models}")


@contextmanager
def tutorial_stack() -> Iterator[Path]:
    """Build the tutorial project, bring up Postgres, seed ARIEL, yield the dir.

    Lifecycle order (each step's failure degrades to :class:`ScreenshotSkip`):
    make a temp build root, ``osprey init <dir> --preset control-assistant``
    into it, ``osprey set model=<the provider's Opus id>``, ``osprey build --skip-deps --dev``
    (the capture drives ``osprey``/ARIEL from the current environment, so the
    deployment needs no venv of its own), ``osprey up -d --dev`` (detached — the
    non-detached form would ``execvpe`` away the runner), wait for Postgres on
    :data:`_POSTGRES_PORT` *before* seeding, then ``osprey sim apply nominal``
    frozen to :data:`recipes.ANCHOR`. Yields the deployment repo directory
    (``<build_root>/<name>``). The ``finally`` block always tears the deployment
    down with the repo-scoped ``osprey reset --yes`` — containers, volumes and
    built images carrying this repo's identity, so the next run's fresh
    credentials never meet a store initialized with this run's — and removes the
    build root. It never issues any prune or system-wide command.
    """
    try:
        build_root = Path(tempfile.mkdtemp(prefix="osprey-docshot-"))
    except OSError as exc:
        raise ScreenshotSkip(f"could not create a temp project dir: {exc}") from exc

    project_dir = build_root / _TUTORIAL_PROJECT_NAME
    try:
        # Two steps because they are two things: `init` writes the source
        # zone, `build` renders it. --skip-deps belongs to the render. The
        # containers run this checkout (a dev render and a dev start), because
        # an unreleased checkout has no published version to pin.
        init_cmd = [
            "osprey",
            "init",
            str(project_dir),
            "--preset",
            "control-assistant",
            "--no-git",
            # The capture drives a single-user `osprey web` on the host; the
            # preset's multi-user roster would own the same index-0 panel ports.
            "--set",
            "config.modules.web_terminals.enabled=false",
        ]
        provider = os.environ.get(PROVIDER_ENV)
        if provider:
            init_cmd += ["--set", f"provider={provider}"]
        _run_stack_step(
            init_cmd,
            cwd=None,
            what="create the control-assistant tutorial repo",
        )
        # Captures on this stack (the stack and agentic screenshots and the
        # demo video) run on Opus, whatever the preset defaults to. The profile
        # names a model by the id its provider serves, and only the catalog
        # init just wrote says which one that is.
        _run_stack_step(
            ["osprey", "set", f"model={_opus_model_id(project_dir)}"],
            cwd=project_dir,
            what="pin the tutorial to Opus",
        )
        _run_stack_step(
            ["osprey", "build", "--skip-deps", "--dev"],
            cwd=project_dir,
            what="build the control-assistant tutorial",
        )
        _run_stack_step(
            ["osprey", "up", "-d", "--dev"],
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
            # check=True: this is the only teardown for a real container
            # stack, and a silent failure leaks it for the rest of the run.
            # A raise here is caught below and reported rather than dropped.
            subprocess.run(
                ["osprey", "reset", "--yes"],
                cwd=str(project_dir),
                capture_output=True,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            # Reported, never raised: this runs in a `finally`, and raising
            # here would replace whatever brought us into it. But it is not
            # swallowed either — a leaked container stack outlives the run and
            # the next one inherits it.
            print(f"WARNING: `osprey reset` failed; container stack may be leaking: {exc}")
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


# The header a scripted client carries the operator secret in; the artifact
# server sits behind the same gate as the web terminal.
OPERATOR_SECRET_HEADER = "X-Osprey-Terminal-Secret"


def artifact_request(artifact_port: int, path: str, secret: str | None) -> urllib.request.Request:
    """A loopback request to the artifact server, carrying ``secret`` when given."""
    headers = {OPERATOR_SECRET_HEADER: secret} if secret else {}
    return urllib.request.Request(f"http://127.0.0.1:{artifact_port}{path}", headers=headers)


def fetch_artifacts(artifact_port: int, secret: str | None = None) -> list[dict]:
    """Return the artifact-server's current artifact list.

    An unreachable or garbled answer is an empty list, since the server may
    still be starting. A refused credential raises :class:`RuntimeError`: no
    later poll can pass, and an empty list would read as "nothing yet".
    """
    request = artifact_request(artifact_port, "/api/artifacts", secret)
    try:
        with urllib.request.urlopen(request, timeout=5.0) as resp:  # loopback only
            return json.loads(resp.read().decode()).get("artifacts", [])
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise RuntimeError(
                f"the artifact server refused {request.full_url}: HTTP {exc.code}"
            ) from exc
        return []
    except (urllib.error.URLError, json.JSONDecodeError, OSError):
        return []


def artifact_ids(artifact_port: int, secret: str | None = None) -> set[str]:
    """Snapshot the ids already present, so a later wait can ignore them."""
    return {a["id"] for a in fetch_artifacts(artifact_port, secret) if a.get("id")}


def _wait_for_artifact(
    artifact_port: int,
    needle: str,
    *,
    timeout: float,
    exclude_ids: set[str] | None = None,
    secret: str | None = None,
) -> str:
    """Poll until a *new* ``artifact_type`` contains ``needle``; return its id.

    ``exclude_ids`` are artifact ids already present before this theme's prompt
    was submitted; matches against them are ignored so each theme waits for the
    plot *its own* prompt produced (the artifact store accumulates across themes,
    so a bare match would return the previous theme's stale plot immediately).

    Returns the matched artifact's id so the caller can select *that* artifact in
    the gallery — the agent often keeps producing artifacts (e.g. a summary) after
    the plot, so the newest-auto-selected item is not reliably the plot.

    Raises :class:`TimeoutError` with a clear message if no matching artifact
    appears within ``timeout`` seconds, so a stalled agent run fails fast instead
    of hanging forever.
    """
    seen = exclude_ids or set()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for artifact in fetch_artifacts(artifact_port, secret):
            if artifact.get("id") in seen:
                continue
            if needle in (artifact.get("artifact_type") or ""):
                return artifact.get("id") or ""
        time.sleep(1.0)
    raise TimeoutError(
        f"no artifact with artifact_type containing {needle!r} appeared on "
        f"port {artifact_port} within {timeout}s"
    )


# Environment switch the web terminal reads at startup (it outranks
# ``web.tour``). Fresh browser contexts would otherwise each get the first-visit
# tour invite, which covers the page and takes focus.
WEB_TOUR_ENV = "OSPREY_WEB_TOUR"

# Seconds a stopped web terminal gets to exit before it is killed: `osprey web
# stop` only signals it, and a server still holding its panel ports would clash
# with the next capture's.
_WEB_STOP_GRACE_S = 10.0


def _now_ms() -> int:
    """Wall-clock time in epoch milliseconds, as Claude Code stamps its state."""
    return int(time.time() * 1000)


def _isolated_claude_config(project_dir: Path) -> Path:
    """A fresh Claude Code config dir, seeded for the rendered project.

    The session must not load the operator's own hooks, plugins, output style
    or memory, which would change what the agent does and what the capture
    shows. The seed marks onboarding complete, trusts ``build/`` and approves
    the provider key, as a deployed web terminal's entrypoint does.

    The session starts in auto permission mode, as an operator runs it: the
    agent hands work to background subagents, which cannot show a permission
    prompt, so in manual mode their tool calls wait on a click nobody gives.
    OSPREY's own approval hooks still ask for the tools they gate. Auto mode
    behind a gateway also opens a one-time billing notice that waits for Enter
    and holds the tool call it interrupted; no capture can answer it, so it is
    recorded as acknowledged, as Claude Code records it once an operator has.
    """
    from osprey.deployment.claude_state_seed import seed_claude_state

    config_dir = Path(tempfile.mkdtemp(prefix="osprey-docshot-claude-"))
    seed_claude_state(
        project_dir / "build", env={**os.environ, "CLAUDE_CONFIG_DIR": str(config_dir)}
    )
    settings_path = config_dir / "settings.json"
    settings = json.loads(settings_path.read_text()) if settings_path.is_file() else {}
    settings.setdefault("permissions", {})["defaultMode"] = "auto"
    settings["skipAutoPermissionPrompt"] = True
    settings_path.write_text(json.dumps(settings, indent=2))
    state_path = config_dir / ".claude.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    state["autoModeClassifierBillingNoticeAcknowledgedAt"] = _now_ms()
    state_path.write_text(json.dumps(state, indent=2))
    return config_dir


def _web_server_pid(project_dir: Path) -> int | None:
    """The PID a detached ``osprey web`` recorded for *project_dir*, if any."""
    from osprey.cli.web_cmd import PID_FILE

    try:
        return int((project_dir / PID_FILE).read_text().strip())
    except (OSError, ValueError):
        return None


def _reap_web_server(pid: int) -> None:
    """Wait up to :data:`_WEB_STOP_GRACE_S` for *pid* to exit, then kill it."""
    deadline = time.monotonic() + _WEB_STOP_GRACE_S
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        except PermissionError:
            return  # not ours
        if time.monotonic() >= deadline:
            break
        time.sleep(0.2)
    print(f"WARNING: web terminal PID {pid} survived `osprey web stop`; killing it")
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


@dataclass(frozen=True)
class WebTerminal:
    """A detached ``osprey web`` launched by :func:`web_terminal`."""

    port: int
    base_url: str
    operator_secret: str
    claude_config_dir: Path


def _keep_session_record(config_dir: Path) -> None:
    """Copy the session's transcripts under :data:`KEEP_ENV`, when it is set.

    Reported, never raised: this runs in a teardown, and a failed copy must not
    replace whatever ended the session.
    """
    keep_root = os.environ.get(KEEP_ENV)
    transcripts = config_dir / "projects"
    if not keep_root or not transcripts.is_dir():
        return
    dest = Path(keep_root) / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    try:
        shutil.copytree(transcripts, dest / "projects")
    except OSError as exc:
        print(f"WARNING: could not keep the session transcripts in {dest}: {exc}")
    else:
        print(f"Kept the session transcripts in {dest}")


@contextmanager
def web_terminal(project_dir: Path) -> Iterator[WebTerminal]:
    """Launch a detached ``osprey web`` for ``project_dir``; stop it on exit.

    The server runs in a SEPARATE process whose credential holder is its own,
    so a same-process session cookie cannot reach it. Instead a known operator
    secret is pinned into the child's environment and exposed here: a browser
    logs in with ``?token=<secret>`` (the child mints its own session and sets
    the cookie every later request and the terminal websocket carry), and
    scripted requests send it as ``X-Osprey-Terminal-Secret``.

    Raises :class:`ScreenshotSkip` when the ``osprey`` CLI is missing or the
    server never opens its port. Teardown is the repo-scoped
    ``osprey web stop --repo``; a failed stop is reported, never raised.
    """
    # The page's scripts come from this checkout's vendored bundles, never a
    # public CDN: no capture, screenshot or video, may depend on the network
    # answering mid-run.
    _ok, problems = verify_vendor_bundles()
    if problems:
        raise ScreenshotSkip(
            f"{len(problems)} vendored front-end file(s) missing or corrupt "
            f"(first: {problems[0]}); run `osprey vendor fetch` once"
        )
    web_port = free_port()
    operator_secret = mint_secret()
    config_dir = _isolated_claude_config(project_dir)
    child_env = {
        **os.environ,
        OPERATOR_SECRET_ENV: operator_secret,
        WEB_TOUR_ENV: "never",
        "CLAUDE_CONFIG_DIR": str(config_dir),
        "OSPREY_OFFLINE": "1",
    }
    try:
        proc = subprocess.Popen(
            [
                "osprey",
                "web",
                "--repo",
                str(project_dir),
                "--detach",
                "--port",
                str(web_port),
            ],
            env=child_env,
        )
    except FileNotFoundError as exc:
        shutil.rmtree(config_dir, ignore_errors=True)
        raise ScreenshotSkip(f"osprey CLI unavailable to launch web terminal: {exc}") from exc

    try:
        try:
            wait_for_port(web_port, timeout=90.0)
        except RuntimeError as exc:
            raise ScreenshotSkip(f"web terminal did not become ready: {exc}") from exc
        yield WebTerminal(
            port=web_port,
            base_url=f"http://127.0.0.1:{web_port}",
            operator_secret=operator_secret,
            claude_config_dir=config_dir,
        )
    finally:
        server_pid = _web_server_pid(project_dir)
        try:
            # check=True: a silently-failed stop leaves a detached web terminal
            # holding its port.
            subprocess.run(
                ["osprey", "web", "stop", "--repo", str(project_dir)],
                capture_output=True,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            print(f"WARNING: `osprey web stop` failed; a web terminal may still be running: {exc}")
        if server_pid is not None:
            _reap_web_server(server_pid)
        _keep_session_record(config_dir)
        shutil.rmtree(config_dir, ignore_errors=True)
        try:
            proc.terminate()
        except (OSError, ValueError):
            pass


def _capture_agentic(
    browser: Browser, project_dir: Path, artifact_port: int, shot: DocShot
) -> list[Path]:
    """Drive the live web terminal to produce the agentic hero screenshot(s).

    Launches a detached ``osprey web`` bound to a free port, then for each theme:
    opens the UI with ``?theme=``, answers the PTY trust prompt, types the
    operator prompt, waits (bounded) for a matching
    artifact, reveals the artifacts panel, opens the plot, and screenshots the
    viewport. Every launched process and page is torn down in ``finally``; the
    web server is stopped with the repo-scoped ``osprey web stop --repo``.
    """
    with web_terminal(project_dir) as web:
        base_url = web.base_url
        operator_secret = web.operator_secret
        dest_dir = output_dir()
        dest_dir.mkdir(parents=True, exist_ok=True)
        n_themes = len(shot.themes)
        written: list[Path] = []

        for theme in shot.themes:
            page = browser.new_page(
                viewport={"width": shot.viewport[0], "height": shot.viewport[1]}
            )
            try:
                # First navigation carries the operator secret as ``?token=``:
                # the gate admits this GET to ``/``, the root handler mints a
                # session, sets the session cookie, and redirects to the
                # token-stripped URL. Every later request (and the PTY websocket)
                # then authenticates with the cookie the child issued.
                token = urllib.parse.quote(operator_secret, safe="")
                page.goto(
                    f"{base_url}/?theme={theme}&token={token}",
                    wait_until="domcontentloaded",
                    timeout=30_000,
                )
                # The Claude-Code trust prompt lives in the PTY, not the DOM:
                # focus the terminal and answer it, then type the operator prompt.
                # The Enter that accepts the trust prompt kicks the CLI into its
                # REPL; typing before that transition finishes drops the prompt
                # characters, so settle briefly before typing.
                page.locator("#terminal-container").click(timeout=30_000)
                page.keyboard.press("Enter")
                page.wait_for_timeout(2_000)

                # Baseline the artifacts already present (from earlier themes) so
                # the wait below matches the plot *this* prompt produces, not a
                # stale one carried over in the shared artifact store.
                before = artifact_ids(artifact_port, operator_secret)
                page.keyboard.type(shot.prompt or "")
                page.keyboard.press("Enter")

                plot_id = _wait_for_artifact(
                    artifact_port,
                    shot.wait_for or "",
                    timeout=240.0,
                    exclude_ids=before,
                    secret=operator_secret,
                )

                # Reveal the artifacts panel and select the plot by its id. The
                # embedded gallery defaults to the tree view (rows), not the card
                # grid, and every row/card carries ``data-id`` — so a data-id
                # selector reveals the plot's preview regardless of view mode,
                # and (unlike relying on newest-auto-select) pins the preview to
                # the plot even as the agent keeps emitting later artifacts.
                page.locator('button[data-panel-id="artifacts"]').click(timeout=30_000)
                panel = page.frame_locator('iframe.panel-iframe[data-panel-id="artifacts"]')
                panel.locator(f'[data-id="{plot_id}"]').first.click(timeout=30_000)

                # Let the plot preview render (Plotly draws async) before shooting.
                page.wait_for_timeout(2_000)
                png = page.screenshot()
                assert_hero_structural(png, shot.viewport)

                dest = dest_dir / _output_filename(shot.name, theme, n_themes)
                dest.write_bytes(png)
                written.append(dest)
            finally:
                page.close()

        return written


def _capture_web_terminal_static(browser: Browser, project_dir: Path, shot: DocShot) -> list[Path]:
    """Capture a static element out of the live web terminal (no agent turn).

    Launches a detached ``osprey web`` against the built tutorial repo exactly
    as :func:`_capture_agentic` does — the ``?token=`` login, the PTY trust
    prompt — but sends no operator prompt and needs no provider credentials.
    What it waits for instead is the control-target chip reporting an
    enforceable session (``data-enforceable="true"``): the signal that the
    CLI's controls server has published its state record, so the popover will
    render real rows rather than a warning banner. It then opens the popover
    by clicking the chip and crops the recipe's ``element_selector`` at 2x,
    once per theme.
    """
    from playwright.sync_api import expect

    with web_terminal(project_dir) as web:
        base_url = web.base_url
        operator_secret = web.operator_secret
        dest_dir = output_dir()
        dest_dir.mkdir(parents=True, exist_ok=True)
        n_themes = len(shot.themes)
        written: list[Path] = []

        for theme in shot.themes:
            # Element crops render at 2x device scale, like capture_shot's.
            page = browser.new_page(
                viewport={"width": shot.viewport[0], "height": shot.viewport[1]},
                device_scale_factor=2,
            )
            try:
                token = urllib.parse.quote(operator_secret, safe="")
                page.goto(
                    f"{base_url}/?theme={theme}&token={token}",
                    wait_until="domcontentloaded",
                    timeout=30_000,
                )
                # Accept the CLI trust prompt so the session boots its MCP
                # servers; the controls server's published record is what
                # flips the chip to enforceable. On the second theme the
                # session resumes and the Enter is a harmless keypress.
                page.locator("#terminal-container").click(timeout=30_000)
                page.keyboard.press("Enter")

                chip = page.locator("#control-target-chip")
                expect(chip).to_be_visible(timeout=60_000)
                try:
                    expect(chip).to_have_attribute("data-enforceable", "true", timeout=90_000)
                except AssertionError as exc:
                    raise ScreenshotSkip(
                        f"{shot.name}: the controls server never published a record "
                        "(chip stayed unenforceable) — the capture would show a "
                        "warning banner instead of the real rows"
                    ) from exc

                chip.click()
                target = page.locator(shot.element_selector)
                expect(target).to_be_attached(timeout=10_000)
                # Let the open animation and the first re-render settle.
                page.wait_for_timeout(600)

                png = target.screenshot()
                dest = dest_dir / _output_filename(shot.name, theme, n_themes)
                dest.write_bytes(png)
                written.append(dest)
            finally:
                page.close()
        return written


def capture_tutorial_stack(
    browser_factory: Callable[[], Browser], shot: DocShot, *, agentic: bool
) -> list[Path]:
    """Capture a ``tutorial_stack`` recipe (container lifecycle owner).

    ``browser_factory`` is a zero-argument callable returning a live browser to
    reuse for the capture. Builds and seeds the tutorial stack (via
    :func:`tutorial_stack`), then dispatches on ``shot.kind`` and
    ``shot.stack_app``: ``"static"`` boots the ARIEL app on a throwaway port
    and reuses :func:`capture_shot` — or, for ``stack_app="web_terminal"``,
    drives the live web terminal statically via
    :func:`_capture_web_terminal_static`; ``"agentic"`` drives it with a real
    agent turn via :func:`_capture_agentic` (only when ``agentic`` is set).
    Raises :class:`ScreenshotSkip` when the container runtime is unavailable
    so a ``--stack`` run degrades gracefully.
    """
    with tutorial_stack() as project_dir:
        artifact_port = rendered_artifact_port(project_dir)
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

        if shot.stack_app == "web_terminal":
            return _capture_web_terminal_static(browser, project_dir, shot)

        from osprey.interfaces.ariel.app import create_app

        # The rendered config moved under build/; older layouts kept it at the repo root.
        config_path = project_dir / "build" / "config.yml"
        if not config_path.is_file():
            config_path = project_dir / "config.yml"

        app = create_app(config_path=str(config_path))
        with run_app_server(app) as ariel_url:
            return capture_shot(browser, ariel_url, shot)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(shots: list[DocShot], *, stack: bool = False, agentic: bool = False) -> None:  # noqa: ARG001 - the CLI passes both selector flags by keyword; the recipe list arrives already filtered on stack
    """Capture the selected recipes, sharing one headless browser for the run.

    ``standalone_interface`` recipes are booted and captured directly;
    ``static_page`` recipes are served from their committed HTML file;
    ``hermetic_hub`` recipes boot the contact sheet's seeded web terminal;
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
                elif shot.environment == "static_page":
                    try:
                        paths = _capture_static_page(browser, shot)
                    except ScreenshotSkip as exc:
                        print(f"skipped {shot.name}: {exc}", file=sys.stderr)
                        continue
                elif shot.environment == "hermetic_hub":
                    try:
                        paths = _capture_hermetic_hub(browser, shot)
                    except ScreenshotSkip as exc:
                        print(f"skipped {shot.name}: {exc}", file=sys.stderr)
                        continue
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
