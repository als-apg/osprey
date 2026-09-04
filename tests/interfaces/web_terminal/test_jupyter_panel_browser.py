"""Browser smoke: the JUPYTER panel, from the rail entry to a cell's output.

One flow through the whole stack a real operator touches, with nothing stubbed
below the browser: the terminal's lifespan spawns a real ``jupyter_server``
sidecar on an ephemeral loopback port, the rail entry's health poll reaches it
through the panel proxy, the panel iframe loads JupyterLab from
``/panel/jupyter``, the starter notebook opens from the file browser, and its
cells run on a real kernel started through the OSPREY kernelspec.

What only a browser can prove, and why each step is here:

  * the rail entry becomes ENABLED — the health poll really reaches the
    sidecar's ``api/status`` through the proxy, with the per-launch token
    re-issued server-side (the browser never holds it);
  * the panel iframe really renders JupyterLab, rather than the bare
    "Jupyter Server" landing page the sidecar serves without a pinned
    ``default_url``;
  * a pinned dark web theme reaches JupyterLab through the labconfig override
    the sidecar writes — the override file existing proves nothing about
    whether ``jupyterlab_server`` applied it;
  * ``getting-started.ipynb`` is listed and opens, and its import cell runs
    with no output at all — the kernel's interpreter can import
    ``osprey.runtime``, which no in-process test can establish;
  * a fresh cell round-trips ``1+1`` to ``2``, so the proxy's WebSocket leg
    carries the kernel channels in both directions;
  * nothing outlives the server: no sidecar, no kernel.

The kernel's first start runs a fresh interpreter through the launcher, so the
timeouts here are generous by design rather than by superstition.

Run:
    uv run pytest tests/interfaces/web_terminal/test_jupyter_panel_browser.py -m browser -q

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
import requests

from tests.interfaces._panel_launch import DEFAULT_ARTIFACT_URL, publish_artifact_url
from tests.interfaces.conftest import _run_app_server

# ---------------------------------------------------------------------------
# Playwright availability guard
# ---------------------------------------------------------------------------

try:
    from playwright.sync_api import Page, expect

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False

pytestmark = [pytest.mark.browser, pytest.mark.slow]

#: JupyterLab's own boot inside the panel iframe. The bundle is large and every
#: byte of it crosses the terminal's panel proxy.
_LAB_BOOT_MS = 90_000

#: A kernel's first start: a fresh interpreter that imports ``osprey.runtime``
#: before it answers anything.
_KERNEL_TIMEOUT = 180.0

#: A cell's round trip once the kernel is already idle.
_EXEC_MS = 60_000

#: Wide enough that JupyterLab keeps its left sidebar open inside the panel
#: tile. Below roughly 600px of iframe width Lab collapses to its narrow
#: layout and the file browser is no longer on screen to double-click.
_VIEWPORT = {"width": 1600, "height": 1000}

#: Seeded before load: marks the onboarding tour dismissed, so its invite card
#: cannot overlay the shell and swallow the rail click. Same seed the sibling
#: dock suite uses; the tour has its own coverage.
_DISMISS_RAIL_HINT = "try { localStorage.setItem('osprey-tour-dismissed-v1', '1') } catch (e) {}"


# ---------------------------------------------------------------------------
# Environment: a shared root and a pinned theme, both inside tmp_path
# ---------------------------------------------------------------------------


@pytest.fixture
def shared_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the sidecar at a throwaway agent-data root and pin the theme dark.

    Three seams, all of them read during the lifespan rather than by the test:

    * ``resolve_agent_data_root`` is what ``_launch_sidecar`` hands the sidecar
      as its shared root, so repointing it puts ``notebooks/`` and the sidecar's
      persistent state under *tmp_path* instead of the developer's own
      ``var/agent_data``. It is imported inside the function, so the seam is the
      attribute on its own module.
    * ``OSPREY_CONFIG`` only has to be SET — the sidecar's preflight refuses to
      launch without it and reads nothing out of it.
    * ``routes.websocket`` binds ``resolve_agent_data_root`` with a
      module-level ``from ... import``, so the name it calls is its own and the
      patch on ``operator_session`` never reaches it. That is the root
      ``write_binding`` writes under; unpatched it derives the cwd, dropping
      ``var/agent_data/jupyter/session-binding.json`` into the repository and
      failing the session posture leak guard.
    * ``OSPREY_WEB_THEME`` resolves to a pinned mode of ``dark`` during startup,
      which is the value that reaches the sidecar's labconfig override.

    ``TMPDIR`` is deliberately NOT set. The sidecar's per-launch tempdir is a
    ``tempfile.mkdtemp`` in this same process, and ``tempfile`` caches its
    directory on first use — which ``tmp_path``'s own factory has already
    triggered before any fixture runs. Setting it here would read as though the
    runtime dir were being relocated while changing nothing at all, so the
    teardown check asks the sidecar where its runtime dir actually is instead.

    Returns:
        The shared root the sidecar will use.
    """
    from osprey.interfaces.web_terminal import operator_session
    from osprey.interfaces.web_terminal.routes import websocket

    root = tmp_path / "agent_data"
    root.mkdir()
    monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "does-not-exist.yml"))
    monkeypatch.setenv("OSPREY_WEB_THEME", "dark")
    monkeypatch.setattr(operator_session, "resolve_agent_data_root", lambda app=None: str(root))
    monkeypatch.setattr(websocket, "resolve_agent_data_root", lambda app=None: str(root))
    return root


@contextmanager
def _live_server(workspace_dir: Path):
    """Launch a real web terminal with the JUPYTER panel enabled.

    The patch set is the sibling dock suite's, minus the parts this flow does
    not need: the artifacts panel is published at the shared unserved address
    (it is the default panel and would otherwise hold the boot open), and the
    panel config is handed over directly rather than read from a profile.

    Nothing about the sidecar is patched — the lifespan spawns it for real,
    which is the point of this module.

    Args:
        workspace_dir: The directory the file watcher is pointed at.

    Yields:
        ``(base_url, app)`` — the live server address and the FastAPI app.
    """
    with (
        patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace_dir)},
        ),
        patch(
            "osprey.interfaces.web_terminal.app._load_panel_config",
            return_value=({"artifacts", "jupyter"}, [], None),
        ),
        patch(
            "osprey.interfaces.web_terminal.app._launch_panel_server",
            side_effect=publish_artifact_url(DEFAULT_ARTIFACT_URL),
        ),
    ):
        from osprey.interfaces.web_terminal.app import create_app

        app = create_app(shell_command=["echo", "hello"])
        with _run_app_server(app) as base_url:
            yield base_url, app


# ---------------------------------------------------------------------------
# Page helpers
# ---------------------------------------------------------------------------


def _open_page(browser, base_url: str) -> Page:
    """Open the terminal and wait for the rail and the dock grid to render.

    Args:
        browser: The function-scoped chromium fixture.
        base_url: The live server's address.

    Returns:
        A page whose rail and dockview grid are both on screen.
    """
    page = browser.new_page(viewport=_VIEWPORT)
    page.add_init_script(_DISMISS_RAIL_HINT)
    page.goto(base_url, wait_until="domcontentloaded")
    expect(page.locator('button.panel-rail-button[data-panel-id="artifacts"]')).to_be_attached(
        timeout=10_000
    )
    expect(page.locator(".dv-groupview").first).to_be_visible(timeout=10_000)
    return page


def _rail_entry(page: Page, panel_id: str):
    """The rail button for *panel_id*, healthy or not."""
    return page.locator(f'button.panel-rail-button[data-panel-id="{panel_id}"]')


def _enabled_rail_entry(page: Page, panel_id: str):
    """The rail button for *panel_id*, only once its health poll settled healthy.

    The rail signals availability with the ``disabled`` CSS class rather than
    the HTML attribute (``panel-rail.js`` ``setEntryEnabled``), so the enabled
    state is a class assertion — the same handle the sibling dock suite waits
    on.
    """
    return page.locator(f'button.panel-rail-button[data-panel-id="{panel_id}"]:not(.disabled)')


def _lab(page: Page):
    """A frame locator scoped to the JUPYTER panel's overlay iframe.

    Service panels render into an overlay iframe layer that tracks the dock
    group's geometry (``dock-iframe.js``), so the iframe is addressed by its
    ``data-panel-id`` inside that layer rather than by position.
    """
    return page.frame_locator('.dock-iframe-overlay iframe[data-panel-id="jupyter"]')


def _wait_for_idle_kernel(base_url: str, timeout: float) -> dict:
    """Block until the sidecar reports a notebook session with an idle kernel.

    Driven from the test process rather than from the DOM. The kernel's first
    start runs a fresh interpreter through the OSPREY launcher, and the
    server's own session list is the one unambiguous signal that it came up —
    a toolbar indicator reads "idle" just as readily before any kernel was
    requested at all.

    Args:
        base_url: The live server's address; the sidecar is reached through its
            panel proxy.
        timeout: Seconds to wait before failing.

    Returns:
        The session record whose kernel reported idle.

    Raises:
        AssertionError: No session reported an idle kernel in time.
    """
    deadline = time.monotonic() + timeout
    seen: object = None
    while time.monotonic() < deadline:
        response = requests.get(f"{base_url}/panel/jupyter/api/sessions", timeout=10)
        if response.ok:
            seen = response.json()
            for session in seen:
                if (session.get("kernel") or {}).get("execution_state") == "idle":
                    return session
        time.sleep(0.5)
    raise AssertionError(f"No idle kernel within {timeout:.0f} s; sessions: {seen}")


def _assert_nothing_outlived(*patterns: str) -> None:
    """Fail if any process still names one of *patterns* after the server stopped.

    Both processes this feature starts have to be named, and they are named by
    different paths:

    * the SIDECAR by the shared root, which its argv carries as
      ``--ServerApp.root_dir=<shared root>/notebooks``;
    * every KERNEL by the sidecar's per-launch runtime directory, which its argv
      carries as ``-f <runtime dir>/kernel-<id>.json``. That directory is a
      ``tempfile.mkdtemp`` in the system temp dir, NOT under ``tmp_path``, so
      the shared root alone would never match a kernel — and a kernel is the
      process most worth checking, because ``jupyter_client`` starts it in its
      own session, outside the group the sidecar signals. It survives unless
      jupyter_server's own shutdown reaps it, which is exactly the path this
      guards.

    Both paths are unique to one run, so neither can trip over a Jupyter the
    developer is running themselves.

    Polled rather than sampled once: the server thread is joined with its own
    timeout, so the lifespan's shutdown may still be terminating processes when
    this is reached.

    Args:
        *patterns: Absolute paths to look for in full command lines.

    Raises:
        AssertionError: A process still names one of them after the grace window.
    """
    if shutil.which("pgrep") is None:  # pragma: no cover - POSIX runners have it
        return
    deadline = time.monotonic() + 15.0
    pids: list[str] = []
    while time.monotonic() < deadline:
        found = subprocess.run(
            ["pgrep", "-f", "|".join(str(p) for p in patterns)],
            capture_output=True,
            text=True,
            check=False,
        )
        if found.returncode != 0:
            return
        pids = found.stdout.split()
        time.sleep(0.5)
    raise AssertionError(f"Processes outlived the server:\n{_describe(pids)}")


def _describe(pids: list[str]) -> str:
    """Render *pids* as one ``pid: command line`` per line, for a failure message.

    ``pgrep -l`` is not enough: on Linux it prints the process *name*, so every
    survivor reads ``python`` and the report cannot say which of the two paths
    it was matched on. ``pgrep -a`` would print the command line but is not in
    macOS's pgrep, so the lookup goes through ``ps``, which both have.
    """
    described = []
    for pid in pids:
        shown = subprocess.run(
            ["ps", "-p", pid, "-o", "command="],
            capture_output=True,
            text=True,
            check=False,
        )
        described.append(f"  {pid}: {shown.stdout.strip() or '<gone>'}")
    return "\n".join(described)


# ===========================================================================
# The flow
# ===========================================================================


def test_notebooks_panel_opens_jupyterlab_and_runs_cells_on_a_real_kernel(
    tmp_path, chromium_browser, shared_root
):
    """The JUPYTER panel, end to end: rail entry, JupyterLab, a live kernel.

    Ordered so a failure names its own step. The rail entry is asserted enabled
    BEFORE it is clicked: a disabled entry is ``pointer-events: none``, so a
    click alone would wait out the health poll silently and report a timeout on
    the wrong element.

    The import cell is asserted to produce NO output rather than "no error
    output": a successful ``from osprey.runtime import ...`` renders nothing at
    all, so an empty output area is the stronger statement and a traceback
    fails it. That check is deliberately the LAST one in the flow — see the
    comment at its call site.
    """
    workspace = tmp_path / "watch"
    workspace.mkdir()

    with _live_server(workspace) as (base_url, app):
        assert getattr(app.state, "jupyter_server_url", None), (
            "the lifespan published no sidecar URL — the launch failed before the browser"
        )

        page = _open_page(chromium_browser, base_url)

        # The health poll goes through /panel/jupyter/api/status, with the
        # per-launch token re-issued server-side.
        expect(_enabled_rail_entry(page, "jupyter")).to_be_attached(timeout=60_000)

        # A rail click takes the focused tile over (one panel per tile).
        _rail_entry(page, "jupyter").click()
        expect(page.locator('.tile-tab[aria-label="JUPYTER"]')).to_have_count(1, timeout=10_000)

        lab = _lab(page)
        expect(lab.locator("#jp-main-dock-panel")).to_be_visible(timeout=_LAB_BOOT_MS)

        # The pinned dark web theme reached JupyterLab through the labconfig
        # override the sidecar wrote for this launch.
        body = lab.locator("body")
        expect(body).to_have_attribute("data-jp-theme-light", "false", timeout=_LAB_BOOT_MS)
        expect(body).to_have_attribute("data-jp-theme-name", "JupyterLab Dark")

        # The starter notebook seeded into the empty notebooks/ is listed, and
        # opens the way an operator opens it.
        starter = lab.locator(".jp-DirListing-item").filter(has_text="getting-started.ipynb")
        expect(starter).to_have_count(1, timeout=_LAB_BOOT_MS)
        starter.dblclick()
        expect(lab.locator(".jp-NotebookPanel")).to_be_visible(timeout=_LAB_BOOT_MS)

        _wait_for_idle_kernel(base_url, _KERNEL_TIMEOUT)

        # Run the import cell. Shift+Enter on the last cell also appends a
        # fresh one, which is the cell the arithmetic goes into below.
        import_cell = lab.locator(".jp-CodeCell").first
        import_cell.locator(".cm-content").click()
        page.keyboard.press("Shift+Enter")

        expect(import_cell.locator(".jp-InputArea-prompt")).to_have_text("[1]:", timeout=_EXEC_MS)

        cells = lab.locator(".jp-CodeCell")
        expect(cells).to_have_count(2, timeout=_EXEC_MS)
        arithmetic = cells.nth(1)
        arithmetic.locator(".cm-content").click()
        page.keyboard.type("1+1")
        page.keyboard.press("Shift+Enter")

        expect(arithmetic.locator(".jp-OutputArea-output")).to_have_text("2", timeout=_EXEC_MS)

        # Checked last, not beside the prompt above. The prompt comes from the
        # shell reply while a traceback arrives separately on iopub, so a check
        # taken the moment the prompt lands can read an output area that is
        # merely still empty. By the time the SECOND cell has rendered its
        # result, everything the first cell will ever emit has arrived.
        expect(import_cell.locator(".jp-OutputArea-output")).to_have_count(0)

        # Read while the sidecar is still up: stop() clears the property and
        # removes the directory.
        runtime_dir = app.state.sidecars["jupyter"].runtime_dir
        assert runtime_dir is not None, "the sidecar reported no runtime dir while running"

        page.close()

    _assert_nothing_outlived(str(shared_root), str(runtime_dir))
