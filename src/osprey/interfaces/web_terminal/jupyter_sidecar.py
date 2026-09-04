"""Launch and supervise the notebook sidecar behind the web terminal's panel.

The sidecar is a ``jupyter_server`` process the terminal starts itself, bound
to loopback on a port the OS assigns, protected by a token minted per launch.
The proxy re-issues that token on every request, so the browser never holds it.

Two interpreters are involved, and they are not the same one:

* the **sidecar** runs under the terminal's own interpreter
  (:data:`sys.executable`): it imports ``jupyter_server`` and this package's
  confined contents manager, both of which live with the terminal;
* every **kernel** runs under :func:`resolve_agent_interpreter`, the interpreter
  that runs agent-authored code, so a cell imports the same packages the
  project's own environment ships.

Paths. Persistent state lives on the agent-data shared root, next to the
session binding the kernel launcher reads::

    <shared_root>/notebooks/                  ServerApp.root_dir
    <shared_root>/jupyter/data/               JUPYTER_DATA_DIR
    <shared_root>/jupyter/data/kernels/osprey/kernel.json
    <shared_root>/jupyter/lab/settings/       JUPYTERLAB_SETTINGS_DIR
    <shared_root>/jupyter/lab/workspaces/     JUPYTERLAB_WORKSPACES_DIR

Per-launch secrets never touch the shared root, because agent tooling may read
anything under it. They live in a ``tempfile.mkdtemp`` directory (mode 0700)
that :meth:`JupyterSidecar.stop` removes::

    <tempdir>/runtime/                        JUPYTER_RUNTIME_DIR
    <tempdir>/config/                         JUPYTER_CONFIG_DIR
    <tempdir>/config/labconfig/default_setting_overrides.json
    <tempdir>/config/labconfig/page_config.json

The runtime directory holds ``jpserver-<pid>.json`` (the port) and every
kernel's connection file (its signing key); the config directory holds nothing
the shared root should keep. The override file is where a pinned web theme
reaches JupyterLab: ``jupyterlab_server`` reads it through
``ConfigManager("labconfig")``, and a user's own theme pick, which lands in
``JUPYTERLAB_SETTINGS_DIR`` on the shared root, still wins and still persists.
``page_config.json`` is read the same way; it switches off the JupyterLab
plugins that make no sense inside an embedded panel (see
:data:`LAB_DISABLED_EXTENSIONS`).

A sidecar that exits on its own — outside :meth:`JupyterSidecar.stop` — is
reported once, with its last stderr lines, and :attr:`JupyterSidecar.on_exit`
is invoked so the terminal can retract the panel. Nothing restarts it.
"""

from __future__ import annotations

import collections
import json
import logging
import os
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import IO, Any

import nbformat

from osprey.jupyter_kernel import BINDING_RELPATH
from osprey.mcp_server.python_executor.executor import resolve_agent_interpreter
from osprey.mcp_server.sandbox_env import scrub_sandbox_child_env

logger = logging.getLogger(__name__)

_NAME = "Notebook sidecar"

#: Path of the panel under the terminal's URL prefix; the sidecar's ``base_url``.
_PANEL_PATH = "panel/jupyter"

#: The one kernelspec the sidecar lists. ``KernelSpecManager.allowed_kernelspecs``
#: is pinned to exactly this name, and ``ensure_native_kernel`` is off, so the
#: interpreter's own ``python3`` spec never appears.
KERNELSPEC_NAME = "osprey"

#: The shared-root subtree that holds sidecar state. Derived from the session
#: binding's path so the sidecar and the kernel launcher agree on one subtree.
_SHARED_SUBTREE = Path(BINDING_RELPATH).parent

#: The settings plugin JupyterLab reads its theme from, and the two theme
#: names it ships. A pinned web theme is written as an override for this key.
_THEME_PLUGIN = "@jupyterlab/apputils-extension:themes"
_THEME_NAMES = {"dark": "JupyterLab Dark", "light": "JupyterLab Light"}

#: JupyterLab plugins switched off through ``labconfig/page_config.json``.
#: The extension manager offers installs into an image that cannot keep them;
#: the announcements plugin is the news prompt and the update check.
LAB_DISABLED_EXTENSIONS = (
    "@jupyterlab/extensionmanager-extension",
    "@jupyterlab/apputils-extension:announcements",
)

#: The ``LabApp`` class that answers "no update available"; named on the argv.
_NEVER_CHECK_FOR_UPDATE = "jupyterlab.handlers.announcements.NeverCheckForUpdate"

#: The notebook written into an empty ``notebooks/``, and its two cells.
STARTER_NOTEBOOK_NAME = "getting-started.ipynb"
_STARTER_MARKDOWN = "This kernel follows your terminal session. Open a chat session before writing."
_STARTER_CODE = "from osprey.runtime import read_channel, write_channel"

#: The health check that names a channel the facility declares is still
#: moving. Its whole purpose is to tell a live archive from a wedged one, so
#: the channel it watches is the one a demo read can count on to show a value.
_FRESHNESS_CHECK = "archiver_freshness"

#: The probe channel the generic project template ships unfilled. The build
#: refuses to write a real-looking one because a placeholder makes a target
#: look reachable while naming a channel nothing serves; a starter cell that
#: read it would fail on its first run, so it is skipped here for the same
#: reason.
_PROBE_PLACEHOLDER = "YOUR:PROBE:CHANNEL"


def _mapping(value: Any) -> Mapping[str, Any]:
    """Return *value* when it is a mapping, else an empty one.

    Config reaches here as whatever the deployment's YAML parsed to. A block
    that is absent, ``None`` or the wrong shape is a deployment that declares
    nothing, which is an answer rather than an error.
    """
    return value if isinstance(value, Mapping) else {}


def _deployment_config() -> Mapping[str, Any]:
    """Return the deployment's resolved configuration.

    Imported at call time: the loader pulls in the workspace machinery, which
    a sidecar has no other reason to carry.
    """
    from osprey.utils.workspace import load_osprey_config

    return load_osprey_config()


def starter_read_channel(config: Mapping[str, Any]) -> str | None:
    """Return a channel *config* says this deployment can read, or ``None``.

    The starter notebook shows a real read only where the deployment has
    already named a channel it can serve. Nothing is guessed and no facility
    name is baked in: a mock-only deployment such as the hello-world preset
    declares no channel, and gets no read line.

    Two declarations qualify, in this order:

    1. an ``archiver_freshness`` health check's ``channel`` — the facility's
       canary, chosen because it keeps moving;
    2. a control target's ``probe_channel`` — the channel the target switch
       reads to prove that target is reachable.

    Args:
        config: The deployment's resolved configuration.

    Returns:
        The channel address, or ``None`` when the deployment names none.
    """
    categories = _mapping(_mapping(config.get("health")).get("categories"))
    for category in categories.values():
        checks = _mapping(category).get("checks")
        for check in checks if isinstance(checks, list) else []:
            entry = _mapping(check)
            channel = entry.get("channel")
            if entry.get("type") == _FRESHNESS_CHECK and isinstance(channel, str) and channel:
                return channel

    targets = _mapping(_mapping(config.get("control_system")).get("connector"))
    for target in targets.values():
        probe = _mapping(target).get("probe_channel")
        if isinstance(probe, str) and probe and probe != _PROBE_PLACEHOLDER:
            return probe

    return None


#: How many stderr lines are kept for the failure report.
_STDERR_TAIL_LINES = 20

_PREFLIGHT_TIMEOUT = 30.0
_STOP_GRACE = 5.0

#: How long a straggling kernel is given to take SIGTERM before it is killed.
_REAP_GRACE = 3.0

#: How long the reap waits for a straggler to *appear* before calling it clean.
#: A kernel forked just before the server died is invisible to a command-line
#: match until it execs, so the reap cannot trust its first snapshot.
_REAP_SETTLE = 1.0


def lab_theme_override(pinned_mode: str | None) -> dict[str, dict[str, str]] | None:
    """Return the labconfig override that pins JupyterLab's theme.

    Args:
        pinned_mode: The web theme's already-resolved pinned mode. Only
            ``'dark'`` and ``'light'`` name a JupyterLab theme; a family value
            (``'retro'``, ``'desy'``) and ``None`` pin nothing.

    Returns:
        The override document, or ``None`` when nothing should be pinned.
    """
    theme = _THEME_NAMES.get(pinned_mode or "")
    return {_THEME_PLUGIN: {"theme": theme}} if theme is not None else None


def lab_page_config() -> dict[str, dict[str, bool]]:
    """Return the labconfig ``page_config`` document the sidecar is seeded with.

    ``jupyterlab_server`` merges this document into the page config it embeds
    in the ``/lab`` page. ``disabledExtensions`` must be the mapping form
    here: the list form is accepted only from the application settings
    directory, and a list from labconfig breaks the merge.
    """
    return {"disabledExtensions": dict.fromkeys(LAB_DISABLED_EXTENSIONS, True)}


def seed_starter_notebook(notebooks_dir: Path, example_channel: str | None = None) -> Path | None:
    """Write the starter notebook when *notebooks_dir* holds no notebook at all.

    Args:
        notebooks_dir: The directory JupyterLab opens.
        example_channel: A channel this deployment can read, from
            :func:`starter_read_channel`. When given, the code cell reads it,
            so the first cell run returns a value instead of importing two
            names and stopping. When ``None`` the cell is the import alone —
            a deployment that names no channel is not given one to fail on.

    Returns:
        The path written, or ``None`` when a notebook already exists anywhere
        under *notebooks_dir*. An existing file is never rewritten.
    """
    if next(notebooks_dir.rglob("*.ipynb"), None) is not None:
        return None
    code = _STARTER_CODE
    if example_channel:
        code = f'{_STARTER_CODE}\n\nread_channel("{example_channel}")'
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell(_STARTER_MARKDOWN),
            nbformat.v4.new_code_cell(code),
        ],
        metadata={
            "kernelspec": {
                "name": KERNELSPEC_NAME,
                "display_name": "OSPREY",
                "language": "python",
            }
        },
    )
    path = notebooks_dir / STARTER_NOTEBOOK_NAME
    with path.open("w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)
    return path


class _StderrTail(threading.Thread):
    """Drain a pipe on a daemon thread, keeping the last few lines."""

    def __init__(self, pipe: IO[bytes]) -> None:
        super().__init__(daemon=True, name="notebook-sidecar-stderr")
        self._pipe = pipe
        self._lines: collections.deque[str] = collections.deque(maxlen=_STDERR_TAIL_LINES)

    def run(self) -> None:
        for raw in self._pipe:
            self._lines.append(raw.decode("utf-8", "replace").rstrip("\n"))

    @property
    def text(self) -> str:
        return "\n".join(self._lines)


class JupyterSidecar:
    """One notebook sidecar process: preflight, spawn, readiness, shutdown.

    Args:
        shared_root: The agent-data shared root. Notebooks and the sidecar's
            persistent state live under it.
        outer_prefix: The terminal's URL prefix (``compute_url_prefix()``);
            empty on a single-user deployment.
        pinned_mode: The web theme's pinned mode, ``'dark'``, ``'light'`` or
            ``None``.
    """

    def __init__(self, shared_root: Path, outer_prefix: str, pinned_mode: str | None) -> None:
        self._shared_root = Path(shared_root)
        self._outer_prefix = outer_prefix
        self._pinned_mode = pinned_mode
        self._token: str | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._tail: _StderrTail | None = None
        self._tempdir: Path | None = None
        self._runtime_dir: Path | None = None
        self._port: int | None = None
        self._watcher: threading.Thread | None = None
        self._stopping = threading.Event()
        # Guards the pair below: the watcher thread records an exit, the
        # loop thread registers the callback, and whichever comes second fires it.
        self._exit_lock = threading.Lock()
        self._exit_status: int | None = None
        self._on_exit: Callable[[], None] | None = None

    # -- shared-root layout -------------------------------------------------

    @property
    def notebooks_dir(self) -> Path:
        return self._shared_root / "notebooks"

    @property
    def data_dir(self) -> Path:
        return self._shared_root / _SHARED_SUBTREE / "data"

    @property
    def kernelspec_path(self) -> Path:
        return self.data_dir / "kernels" / KERNELSPEC_NAME / "kernel.json"

    @property
    def settings_dir(self) -> Path:
        return self._shared_root / _SHARED_SUBTREE / "lab" / "settings"

    @property
    def workspaces_dir(self) -> Path:
        return self._shared_root / _SHARED_SUBTREE / "lab" / "workspaces"

    @property
    def base_url(self) -> str:
        return f"{self._outer_prefix}/{_PANEL_PATH}/"

    @property
    def runtime_dir(self) -> Path | None:
        """The per-launch ``JUPYTER_RUNTIME_DIR``; ``None`` between launches."""
        return self._runtime_dir

    # -- process facts ------------------------------------------------------

    @property
    def token(self) -> str | None:
        return self._token

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    @property
    def stderr_tail(self) -> str:
        return self._tail.text if self._tail is not None else ""

    @property
    def auth_headers(self) -> dict[str, str]:
        return {"authorization": f"Bearer {self._token}"}

    @property
    def url(self) -> str:
        """The backend URL the proxy forwards to; no trailing slash."""
        if self._port is None:
            raise RuntimeError(f"{_NAME} is not ready")
        return f"http://127.0.0.1:{self._port}{self._outer_prefix}/{_PANEL_PATH}"

    @property
    def on_exit(self) -> Callable[[], None] | None:
        """Called once, from a worker thread, when the process exits outside :meth:`stop`.

        Assigning it after the exit already happened fires it right away, so
        a caller that registers after :meth:`wait_ready` cannot miss a death
        in between. The callback runs on the watcher thread: hand loop-owned
        state to the loop from inside it.
        """
        return self._on_exit

    @on_exit.setter
    def on_exit(self, callback: Callable[[], None] | None) -> None:
        with self._exit_lock:
            self._on_exit = callback
            fire = callback is not None and self._exit_status is not None
        if fire and callback is not None:
            callback()

    # -- preflight ----------------------------------------------------------

    def preflight(self) -> None:
        """Check what a launch needs; raise ``RuntimeError`` with one line otherwise.

        Blocks for up to :data:`_PREFLIGHT_TIMEOUT` seconds while the agent
        interpreter proves it can import the kernel's two dependencies.
        """
        if not os.environ.get("OSPREY_CONFIG"):
            raise RuntimeError("OSPREY_CONFIG is not set")
        interpreter = str(resolve_agent_interpreter())
        try:
            probe = subprocess.run(
                [interpreter, "-c", "import ipykernel, osprey.runtime"],
                capture_output=True,
                text=True,
                timeout=_PREFLIGHT_TIMEOUT,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"{interpreter} did not finish the import check in {_PREFLIGHT_TIMEOUT:.0f} s"
            ) from exc
        except OSError as exc:
            raise RuntimeError(f"{interpreter} cannot be run: {exc}") from exc
        if probe.returncode != 0:
            reason = (probe.stderr.strip().splitlines() or ["no output"])[-1]
            raise RuntimeError(
                f"{interpreter} cannot import ipykernel and osprey.runtime: {reason}"
            )

    # -- launch assembly ----------------------------------------------------
    # What a launch is made of: the seeded directories, the kernelspec, the
    # command line and the child environment. Nothing here starts a process.

    def _seed(self, config_dir: Path) -> None:
        """Prepare the shared root and *config_dir* once, before the spawn.

        Args:
            config_dir: The per-launch ``JUPYTER_CONFIG_DIR``.
        """
        for directory in (
            self.notebooks_dir,
            self.kernelspec_path.parent,
            self.settings_dir,
            self.workspaces_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        labconfig = config_dir / "labconfig"
        labconfig.mkdir(parents=True, exist_ok=True)
        (labconfig / "page_config.json").write_text(
            json.dumps(lab_page_config(), indent=2) + "\n", encoding="utf-8"
        )
        override = lab_theme_override(self._pinned_mode)
        if override is not None:
            (labconfig / "default_setting_overrides.json").write_text(
                json.dumps(override, indent=2) + "\n", encoding="utf-8"
            )
        try:
            example_channel = starter_read_channel(_deployment_config())
        except Exception:  # noqa: BLE001 — the read line is cosmetic, the panel is not
            logger.debug("Starter notebook: config load failed", exc_info=True)
            example_channel = None
        seed_starter_notebook(self.notebooks_dir, example_channel)

    def _write_kernelspec(self) -> None:
        spec = {
            "argv": [
                str(resolve_agent_interpreter()),
                "-m",
                "osprey.jupyter_kernel",
                "-f",
                "{connection_file}",
            ],
            "display_name": "OSPREY",
            "language": "python",
            "env": {
                "JUPYTER_TOKEN": "",
                "OSPREY_AGENT_DATA_ROOT": str(self._shared_root),
            },
        }
        self.kernelspec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")

    def _argv(self) -> list[str]:
        """The sidecar's command line. Every ``--Class.trait`` here is a pinned behavior."""
        contents_manager = (
            "osprey.interfaces.web_terminal.jupyter_contents.ConfinedFileContentsManager"
        )
        return [
            sys.executable,
            "-m",
            "osprey.interfaces.web_terminal.jupyter_sidecar_main",
            "--ServerApp.ip=127.0.0.1",
            "--ServerApp.port=0",
            "--ServerApp.port_retries=0",
            "--ServerApp.terminals_enabled=False",
            f"--ServerApp.base_url={self.base_url}",
            f"--ServerApp.root_dir={self.notebooks_dir}",
            f"--ServerApp.contents_manager_class={contents_manager}",
            # Delete means delete, folders included: the trash would land on
            # the shared volume.
            "--FileContentsManager.delete_to_trash=False",
            "--FileContentsManager.always_delete_dir=True",
            "--KernelSpecManager.ensure_native_kernel=False",
            f"--KernelSpecManager.allowed_kernelspecs=['{KERNELSPEC_NAME}']",
            "--ServerApp.open_browser=False",
            "--ServerApp.default_url=/lab",
            # No news feed and no update check from inside an embedded panel.
            "--LabApp.news_url=None",
            f"--LabApp.check_for_updates_class={_NEVER_CHECK_FOR_UPDATE}",
        ]

    def _child_env(self, token: str, runtime_dir: Path, config_dir: Path) -> dict[str, str]:
        """The sidecar's environment: the scrubbed parent's, plus the Jupyter directories.

        Args:
            token: The per-launch token the server requires on every request.
            runtime_dir: The per-launch ``JUPYTER_RUNTIME_DIR``.
            config_dir: The per-launch ``JUPYTER_CONFIG_DIR``.
        """
        env = scrub_sandbox_child_env(os.environ)
        env.update(
            {
                "JUPYTER_TOKEN": token,
                "JUPYTER_RUNTIME_DIR": str(runtime_dir),
                "JUPYTER_CONFIG_DIR": str(config_dir),
                "JUPYTER_DATA_DIR": str(self.data_dir),
                "JUPYTERLAB_SETTINGS_DIR": str(self.settings_dir),
                "JUPYTERLAB_WORKSPACES_DIR": str(self.workspaces_dir),
            }
        )
        if "OSPREY_CONFIG" in os.environ:
            env["OSPREY_CONFIG"] = os.environ["OSPREY_CONFIG"]
        return env

    # -- process lifecycle --------------------------------------------------

    def spawn(self) -> None:
        """Start the sidecar process. Blocks for the ``Popen`` only."""
        if self._process is not None:
            raise RuntimeError(f"{_NAME} is already running (pid {self._process.pid})")
        self._stopping.clear()
        with self._exit_lock:
            self._exit_status = None
        self._token = secrets.token_urlsafe(32)
        self._tempdir = Path(tempfile.mkdtemp(prefix="osprey-notebook-sidecar-"))
        self._runtime_dir = self._tempdir / "runtime"
        config_dir = self._tempdir / "config"
        self._runtime_dir.mkdir(mode=0o700)
        config_dir.mkdir(mode=0o700)
        self._seed(config_dir)
        self._write_kernelspec()

        self._process = subprocess.Popen(
            self._argv(),
            env=self._child_env(self._token, self._runtime_dir, config_dir),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        assert self._process.stderr is not None
        self._tail = _StderrTail(self._process.stderr)
        self._tail.start()
        self._watcher = threading.Thread(
            target=self._watch_exit,
            args=(self._process, self._tail),
            daemon=True,
            name="notebook-sidecar-exit",
        )
        self._watcher.start()
        logger.info("%s launched (pid %s)", _NAME, self._process.pid)

    def _watch_exit(self, process: subprocess.Popen[bytes], tail: _StderrTail) -> None:
        """Reap *process*; report an exit that :meth:`stop` did not cause.

        The stderr pipe is inherited by every kernel, so its end of stream
        cannot stand in for the exit: an orphaned kernel keeps it open.
        ``wait()`` is what says the sidecar is gone. The report is skipped
        while the sidecar was still coming up, because :meth:`wait_ready`
        raises with the same tail and its caller logs that.
        """
        status = process.wait()
        if self._stopping.is_set():
            return
        tail.join(timeout=1)
        if self._port is not None:
            text = tail.text
            logger.warning("%s exited (rc=%s)%s", _NAME, status, f": {text}" if text else "")
        with self._exit_lock:
            self._exit_status = status
            callback = self._on_exit
        if callback is not None:
            callback()

    def _read_port(self) -> int | None:
        assert self._runtime_dir is not None and self._process is not None
        info = self._runtime_dir / f"jpserver-{self._process.pid}.json"
        try:
            return int(json.loads(info.read_text(encoding="utf-8"))["port"])
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _answers(self, port: int) -> bool:
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}{self.base_url}api/status",
            headers={"Authorization": f"token {self._token}"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=2) as response:
                return int(response.status) == 200
        except (urllib.error.URLError, OSError):
            return False

    def wait_ready(self, timeout: float) -> None:
        """Block until the sidecar answers ``api/status`` with the token.

        Raises:
            RuntimeError: the process exited, or *timeout* seconds passed,
                before it answered. The message carries the stderr tail.
        """
        if self._process is None:
            raise RuntimeError(f"{_NAME} was not spawned")
        deadline = time.monotonic() + timeout
        port: int | None = None
        while True:
            status = self._process.poll()
            if status is not None:
                raise RuntimeError(
                    f"{_NAME} exited with status {status} before it was ready\n{self.stderr_tail}"
                )
            if port is None:
                port = self._read_port()
            if port is not None and self._answers(port):
                self._port = port
                logger.info("%s ready at %s", _NAME, self.url)
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"{_NAME} did not answer within {timeout:.0f} s\n{self.stderr_tail}"
                )
            time.sleep(0.1)

    def stop(self) -> None:
        """Terminate the process group and remove the per-launch tempdir. Idempotent."""
        self._stopping.set()
        runtime_dir = self._runtime_dir
        process, self._process = self._process, None
        if process is not None:
            if process.poll() is None:
                _signal_group(process.pid, signal.SIGTERM)
                try:
                    process.wait(_STOP_GRACE)
                except subprocess.TimeoutExpired:
                    _signal_group(process.pid, signal.SIGKILL)
                    process.wait()
            # The stderr pipe is left to the tail thread: closing it here
            # blocks on the reader, and a kernel the group signal did not
            # reach (kernels start their own session) can hold the write end
            # open until the reap below collects it.
            logger.info("%s stopped (pid %s)", _NAME, process.pid)
        if runtime_dir is not None:
            _reap_stragglers(runtime_dir)
        if self._watcher is not None:
            self._watcher.join(timeout=1)
            self._watcher = None
        if self._tail is not None:
            self._tail.join(timeout=1)
            self._tail = None
        self._port = None
        tempdir, self._tempdir = self._tempdir, None
        self._runtime_dir = None
        if tempdir is not None:
            shutil.rmtree(tempdir, ignore_errors=True)


def _pids_naming(runtime_dir: str) -> list[int]:
    """Return the pids of live processes whose command line names *runtime_dir*.

    Args:
        runtime_dir: The sidecar's per-launch runtime directory.

    Returns:
        Matching pids, never including this process.
    """
    found = subprocess.run(
        ["pgrep", "-f", runtime_dir], capture_output=True, text=True, check=False
    )
    own = os.getpid()
    return [int(f) for f in found.stdout.split() if f.isdigit() and int(f) != own]


def _reap_stragglers(runtime_dir: str) -> None:
    """Terminate anything that outlived the server still naming *runtime_dir*.

    Kernels are started in their own session, so the group signal that stops
    the server never reaches them. jupyter-server closes its kernels on its own
    shutdown, but on a loaded host that shutdown can exceed ``_STOP_GRACE`` and
    the server is killed first — leaving a kernel running, with the control
    target and write posture it launched with, and no terminal above it.

    The runtime dir is a 0700 tempdir unique to this launch, and every kernel
    names its connection file inside it, so matching on it cannot reach a
    process belonging to anything else. That uniqueness is the whole safety
    argument, and it is why the match is not widened to the shared root: that
    one is shared across launches, and a signal sent on it could reach another
    deployment's kernels.

    A straggler is waited *for*, not merely sampled. A kernel the server forked
    just before it died does not name the runtime dir until it execs — its
    connection file is an argv entry — so a single snapshot taken the moment the
    server dies races that exec and, losing, leaves the kernel running with
    nothing above it. Hence the settle window: only a quiet
    ``_REAP_SETTLE`` means there is genuinely nothing to reap.

    Args:
        runtime_dir: The sidecar's per-launch runtime directory.
    """
    if shutil.which("pgrep") is None:  # pragma: no cover - POSIX hosts have it
        return
    appear_by = time.monotonic() + _REAP_SETTLE
    while not (survivors := _pids_naming(runtime_dir)):
        if time.monotonic() >= appear_by:
            return
        time.sleep(0.05)
    logger.info("%s reaping %d process(es) that outlived it", _NAME, len(survivors))
    for pid in survivors:
        _signal_pid(pid, signal.SIGTERM)
    deadline = time.monotonic() + _REAP_GRACE
    while time.monotonic() < deadline:
        if not _pids_naming(runtime_dir):
            return
        time.sleep(0.1)
    for pid in _pids_naming(runtime_dir):
        _signal_pid(pid, signal.SIGKILL)


def _signal_pid(pid: int, signum: signal.Signals) -> None:
    """Send *signum* to *pid*; a process already gone is fine."""
    try:
        os.kill(pid, signum)
    except (ProcessLookupError, PermissionError):
        pass


def _signal_group(pid: int, signum: signal.Signals) -> None:
    """Send *signum* to the process group led by *pid*; a gone group is fine."""
    try:
        os.killpg(pid, signum)
    except ProcessLookupError:
        pass
