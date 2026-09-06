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

THE SECOND FLOW: a popped-out notebook that follows the deployment
------------------------------------------------------------------
The two lanes at the bottom of this file are about the OTHER way an operator
holds a notebook — popped out of the hub into its own window, with the terminal
tab closed behind it — and about the control-target bar the panel proxy injects
into that page. They share this module's real sidecar and real kernel, and add
a control-context record and a three-target render on top, because the question
they answer is what a CELL is routed at:

  * a switch made from the Lab page's own chip moves the deployment's record,
    and the very next cell in the popped-out notebook comes up on the new
    machine. Nothing is restarted — that re-routing is the one thing that
    separates a notebook kernel from an executor sandbox;
  * restarting the kernel changes none of it: the fresh process reads the same
    record and lands on the same target, with no refusal hint printed, because
    nothing about a restart is a refusal.

What those two lanes do NOT assert is the VALUE a channel read comes back with
on the new machine. That needs two control systems actually answering on this
host, which is a deployed-container question and is pinned as one, in
``tests/e2e/test_jupyter_panel_lifecycle.py`` (``test_the_next_cell_follows_a_
chip_target_switch``). What is asserted here is the routing itself, read out of
the cell's own environment — the stamp every control-system call in that cell
resolves its connector from.

Run:
    uv run pytest tests/interfaces/web_terminal/test_jupyter_panel_browser.py -m browser -q

Skips cleanly when the chromium headless binary is not installed. Run it ALONE:
a real browser beside a real sidecar and a real kernel has OOM-killed the runner
here when a parallel pytest was live.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import requests
import yaml

from osprey.jupyter_kernel import HINT_TARGET_CHANGED, HINT_WRITES_OFF
from osprey_connectors import control_context, posture_store
from tests._control_context_fixtures import write_control_context, write_server_report
from tests.interfaces._panel_launch import DEFAULT_ARTIFACT_URL, publish_artifact_url
from tests.interfaces.conftest import _apply_all, _run_app_server

# The control-target render, the chip's selectors and the waits are the sibling
# browser suite's, imported rather than restated: two suites disagreeing about
# what a switchable deployment looks like is exactly the drift this feature
# exists to remove.
from tests.interfaces.web_terminal.test_posture_toggle_browser import (
    _LONG_LIVED_SHELL,
    ACTIVE_TARGET,
    CHIP,
    MODAL_CONFIRM,
    MODAL_TITLE,
    NAMES,
    OPEN_MODAL,
    POPOVER,
    RECORD_GENERATION,
    SWITCH_TARGET,
    TIMEOUT,
    _reachability_sweep,
    _reset_process_memos,
    _write_config,
)

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


# ===========================================================================
# A popped-out notebook, its own control-target bar, and what a cell is on
# ===========================================================================

#: The cell that reports what this cell is routed at. One line, because a
#: CodeMirror editor re-indents a typed block and the assertion would then be
#: about the editor. The name is the executor's own stamp — every
#: control-system call in the cell resolves its connector from it, so a cell
#: that prints it is reporting the machine it would have talked to.
_TARGET_CELL = "import os; print(os.environ.get('OSPREY_CONTROL_TARGET'))"

#: The bar the panel proxy injects into the Lab page, and the chip inside it.
_LAB_BAR = "#osprey-control-target-bar"
_LAB_CHIP = f"{_LAB_BAR} {CHIP}"

#: How long a kernel restart gets to produce a fresh, idle interpreter.
_RESTART_TIMEOUT = 180.0


@contextmanager
def _control_target_lab(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[str, Any, Path]]:
    """A live hub with the JUPYTER panel AND a control context to switch.

    This is the sibling chip suite's arrangement and this module's sidecar in
    one process, which is what the two lanes below need and what neither file
    has on its own:

    * the three-target render comes from ``_write_config`` — imported, so the
      roster the Lab bar shows is the same roster the hub's own chip shows;
    * ``agent_data.base_dir`` is pointed INTO *tmp_path* in that same config,
      rather than by patching a resolver. Everything downstream of this process
      re-derives the agent-data root from config — the sidecar's own launch,
      the session-binding writer, the record reader inside the kernel — and a
      patch reaches only the callers it names, leaving the rest writing into
      the repository's ``var/agent_data`` and tripping the leak guard;
    * ``OSPREY_AGENT_DATA_ROOT`` is stamped as well, because the record reader
      prefers the stamp over the config, and the two must not answer
      differently;
    * ``CONFIG_FILE`` is what ``osprey_connectors`` resolves a target through,
      and the KERNEL is the process that has to be able to. Without it the
      kernel's resolvability check reads whatever ``config.yml`` happens to sit
      in the working directory and the cell runs unstamped;
    * the record is written BEFORE ``create_app``, at :data:`ACTIVE_TARGET` and
      owned by this process, so the lifespan's claim is a merge and the Lab
      bar's chip comes up writable.

    Yields:
        ``(base_url, app, root)`` — the hub's address, its app, and the
        agent-data root the record and the server reports live under.
    """
    workspace = tmp_path / "watch"
    workspace.mkdir(exist_ok=True)
    root = tmp_path / "agent_data"
    (root / posture_store.STATE_DIR_NAME).mkdir(parents=True, exist_ok=True)

    config = _write_config(tmp_path / "config.yml")
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    document["agent_data"] = {"base_dir": str(root)}
    config.write_text(yaml.safe_dump(document), encoding="utf-8")

    monkeypatch.setenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, str(root))
    monkeypatch.setenv("OSPREY_CONFIG", str(config))
    monkeypatch.setenv("CONFIG_FILE", str(config))
    monkeypatch.setenv("OSPREY_WEB_THEME", "dark")
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    monkeypatch.delenv("OSPREY_POSTURE_SESSION", raising=False)

    _reset_process_memos()
    write_control_context(root, target=ACTIVE_TARGET, generation=RECORD_GENERATION)

    patches = [
        patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace)},
        ),
        patch(
            "osprey.interfaces.web_terminal.app._load_panel_config",
            return_value=({"artifacts", "jupyter"}, [], None),
        ),
        patch(
            "osprey.interfaces.web_terminal.app._launch_panel_server",
            side_effect=publish_artifact_url(DEFAULT_ARTIFACT_URL),
        ),
        patch(
            "osprey.interfaces.web_terminal.session_discovery.SessionDiscovery"
            ".snapshot_session_ids",
            side_effect=lambda *_args, **_kwargs: set(),
        ),
    ]
    try:
        with _apply_all(patches):
            from osprey.interfaces.web_terminal.app import create_app

            app = create_app(shell_command=list(_LONG_LIVED_SHELL), config_path=config)
            with _run_app_server(app) as base_url:
                yield base_url, app, root
    finally:
        _reset_process_memos()


def _publish_fleet(root: Path, *, applied_target: str, applied_generation: int) -> Path:
    """Publish one live controls server at *applied_generation*.

    Two different jobs, one call. Before a switch it carries the reachability
    sweep the gate refuses without — a report with no probe makes every switch
    answer ``reachability_unknown``, and that refusal is only visible in the
    audit log, so the browser would just show an unexplained dialog error.
    After a switch, rewriting it at the new generation is the fleet arriving,
    which is what ends ``switching…``.
    """
    import os as _os

    return write_server_report(
        root,
        _os.getpid(),
        applied_target=applied_target,
        applied_generation=applied_generation,
        reachability=_reachability_sweep(),
    )


def _record() -> Any:
    """The deployment's control-context record, re-read from disk."""
    control_context.invalidate_cache()
    return control_context.read_record()


def _standalone_notebook(browser, base_url: str) -> Page:
    """Open the starter notebook in its own window, the way a pop-out lands.

    ``/panel/jupyter/notebooks/<path>`` is JupyterLab's single-document route,
    and it is on the proxy's page allow-list, so this is a real page with the
    control-target bar injected into it — not the hub, and not an iframe inside
    the hub. That is the whole arrangement these lanes are about: the operator
    has the notebook and nothing else on screen.
    """
    page = browser.new_page(viewport=_VIEWPORT)
    page.add_init_script(_DISMISS_RAIL_HINT)
    page.goto(
        f"{base_url}/panel/jupyter/doc/tree/getting-started.ipynb",
        wait_until="domcontentloaded",
    )
    expect(page.locator(".jp-NotebookPanel")).to_be_visible(timeout=_LAB_BOOT_MS)
    return page


def _run_next_cell(page: Page, code: str) -> Any:
    """Append a fresh cell carrying *code* to the open notebook, and run it.

    The cell is INSERTED rather than reused. The starter notebook's own cells
    already have content and their own outputs, so typing into the last one
    would append to somebody else's code and then assert on somebody else's
    output. ``Escape`` drops the notebook into command mode and ``b`` inserts
    an empty code cell below the selected one, which is JupyterLab's own
    keyboard path for what an operator does with the ``+`` button.

    Args:
        page: The notebook's own page.
        code: One line of Python. Multi-line source would be re-indented by the
            editor, and the test would then be about CodeMirror.

    Returns:
        The cell that ran, so the caller can read its output.
    """
    before = page.locator(".jp-CodeCell").count()
    page.locator(".jp-CodeCell").last.locator(".cm-content").click()
    page.keyboard.press("Escape")
    page.keyboard.press("b")
    expect(page.locator(".jp-CodeCell")).to_have_count(before + 1, timeout=TIMEOUT)

    # Addressed by INDEX, never by ``.last``. A locator is resolved when it is
    # used, and ``Shift+Enter`` on the final cell appends another one — so a
    # ``.last`` handed back here would point at the fresh empty cell by the
    # time the caller read its output, and would wait out the timeout on a cell
    # that never ran.
    cell = page.locator(".jp-CodeCell").nth(before)
    cell.locator(".cm-content").click()
    page.keyboard.type(code)
    page.keyboard.press("Shift+Enter")
    return cell


def _cell_output(cell: Any) -> str:
    """Everything the cell printed, waited for rather than sampled."""
    expect(cell.locator(".jp-OutputArea-output")).to_have_count(1, timeout=_EXEC_MS)
    return cell.locator(".jp-OutputArea-output").inner_text()


def _switch_from_the_lab_bar(page: Page, target: str) -> None:
    """Drive the injected bar's chip all the way through a switch confirm."""
    expect(page.locator(_LAB_CHIP)).to_be_visible(timeout=_LAB_BOOT_MS)
    expect(page.locator(_LAB_CHIP)).to_have_attribute("data-enforceable", "true", timeout=TIMEOUT)
    page.locator(_LAB_CHIP).click()
    expect(page.locator(POPOVER)).to_be_visible(timeout=TIMEOUT)
    page.locator(f'{POPOVER} .ctc-row[data-target="{target}"] .ctc-switch').click()
    expect(page.locator(MODAL_TITLE)).to_have_text(f"Switch to {NAMES[target]}?", timeout=TIMEOUT)
    page.locator(MODAL_CONFIRM).click()
    expect(page.locator(OPEN_MODAL)).to_have_count(0, timeout=TIMEOUT)


def test_a_popped_out_notebook_follows_a_switch_made_from_its_own_bar(
    tmp_path, monkeypatch, chromium_browser
):
    """The terminal tab is gone; the notebook and its bar carry the whole gesture.

    The arrangement is the one an operator actually ends up in: the notebook is
    popped out into its own window and the hub tab is closed behind it. Nothing
    of the hub's shell is on screen — no rail, no dock, no ``terminal.js`` — so
    the control-target bar the panel proxy injects is the only way to see or
    move the deployment, and the chip inside it is the same module the hub
    header mounts.

    Ordered so a failure names its own step:

    1. the cell before the switch reports :data:`ACTIVE_TARGET`. Without this
       the assertion after the switch would pass on a deployment that had been
       on the new target all along;
    2. the switch is made from the bar's own popover and lands in the record —
       a new target and one more generation;
    3. the chip says ``switching…`` while the planted controls server is still
       reporting the old generation, and stops saying it when that server is
       rewritten at the new one. That is the fleet arriving, and it is the half
       of the pending rule a single-page suite cannot reach;
    4. the very next cell — same kernel, nothing restarted — reports the new
       target.

    Step 4 is the contract: ``pre_run_cell`` rewrites the routing names from
    the record before every cell, so a kernel is the one process that follows a
    switch instead of being invalidated by it.
    """
    with _control_target_lab(tmp_path, monkeypatch) as (base_url, app, root):
        assert getattr(app.state, "jupyter_server_url", None), (
            "the lifespan published no sidecar URL — the launch failed before the browser"
        )
        _publish_fleet(root, applied_target=ACTIVE_TARGET, applied_generation=RECORD_GENERATION)

        hub = _open_page(chromium_browser, base_url)
        notebook = _standalone_notebook(chromium_browser, base_url)
        try:
            # The pop-out is on its own; the tab it came from is closed.
            hub.close()

            _wait_for_idle_kernel(base_url, _KERNEL_TIMEOUT)

            before = _run_next_cell(notebook, _TARGET_CELL)
            assert _cell_output(before).strip() == ACTIVE_TARGET, _cell_output(before)

            _switch_from_the_lab_bar(notebook, SWITCH_TARGET)

            moved = _record()
            assert moved is not None, "the switch left no record"
            assert moved.target == SWITCH_TARGET, moved
            assert moved.generation == RECORD_GENERATION + 1, moved

            # The fleet has not caught up, and the chip says so.
            expect(notebook.locator(_LAB_CHIP)).to_have_attribute(
                "data-pending", "true", timeout=TIMEOUT
            )
            expect(notebook.locator(f"{_LAB_CHIP} .ctc-state")).to_have_text(
                "switching…", timeout=TIMEOUT
            )

            # …and stops saying it when the server arrives at the new generation.
            _publish_fleet(root, applied_target=SWITCH_TARGET, applied_generation=moved.generation)
            expect(notebook.locator(_LAB_CHIP)).not_to_have_attribute(
                "data-pending", "true", timeout=TIMEOUT
            )

            after = _run_next_cell(notebook, _TARGET_CELL)
            printed = _cell_output(after)
            assert printed.strip() == SWITCH_TARGET, printed
            assert HINT_TARGET_CHANGED not in printed, printed
        finally:
            notebook.close()


def test_a_restarted_kernel_comes_back_on_the_same_target_with_no_hint(
    tmp_path, monkeypatch, chromium_browser
):
    """A restart is not a refusal, and it is not a re-derivation either.

    The kernel is the process that outlives a switch, which raises the opposite
    question: does it also survive being replaced? A restarted kernel is a
    fresh interpreter with none of the previous one's environment, so the only
    thing that can put it back on the deployment's machine is the record — and
    the record is where it reads from, before the first cell as before every
    other.

    Two halves, and the second is the one worth having:

    * the cell after the restart reports the SAME target as the cell before it;
    * it prints nothing else. Both action lines the kernel can print are
      refusal lines — the launch pin refused the write, or the target moved
      under a running cell — and a restart is neither. A hint here would be the
      kernel telling an operator to go and fix something that is not wrong.

    The restart is driven through the sidecar's own REST API rather than
    JupyterLab's menu: it is the same code path the menu reaches, and it names
    the kernel being restarted instead of depending on which document has
    focus.
    """
    with _control_target_lab(tmp_path, monkeypatch) as (base_url, app, root):
        assert getattr(app.state, "jupyter_server_url", None)
        _publish_fleet(root, applied_target=ACTIVE_TARGET, applied_generation=RECORD_GENERATION)

        hub = _open_page(chromium_browser, base_url)
        notebook = _standalone_notebook(chromium_browser, base_url)
        try:
            hub.close()
            session = _wait_for_idle_kernel(base_url, _KERNEL_TIMEOUT)
            kernel_id = session["kernel"]["id"]

            before = _run_next_cell(notebook, _TARGET_CELL)
            assert _cell_output(before).strip() == ACTIVE_TARGET, _cell_output(before)

            restarted = requests.post(
                f"{base_url}/panel/jupyter/api/kernels/{kernel_id}/restart", timeout=60
            )
            assert restarted.ok, restarted.text
            _wait_for_idle_kernel(base_url, _RESTART_TIMEOUT)

            after = _run_next_cell(notebook, _TARGET_CELL)
            printed = _cell_output(after)
            assert printed.strip() == ACTIVE_TARGET, printed
            assert HINT_WRITES_OFF not in printed, printed
            assert HINT_TARGET_CHANGED not in printed, printed
        finally:
            notebook.close()
