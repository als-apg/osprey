"""One user's CANOPY, run by the web terminal as two child processes.

``FRONT`` is CANOPY's panel server (``python -m canopy_mcp --http --fd N``): the
viewer, the saved views, the MCP endpoint and the relays. ``DATA`` is CANOPY's
backend (``python -m uvicorn canopy.main:app --fd N``) with OSPREY's PV backend
loaded, so every value it serves comes through the control session of the user
the terminal belongs to.

Both listen on loopback sockets this process binds and keeps. A request that
arrives while DATA is being replaced waits in the kernel's backlog and is
answered by the replacement, and neither port changes for the sidecar's life.

DATA is supervised by how it exits:

======  =========================  =========================================
Exit    Meaning                    What follows
======  =========================  =========================================
75      the control target moved   a replacement at once, rendered again
76      the transport went stale   a replacement at once, rendered again
78      the start was refused      no replacement; the panel is retracted
other   crash                      replacements after 0.5, 1, 2, 4 s
======  =========================  =========================================

Any five exits other than 78 in 60 s retract the panel.

Its environment is rendered again for every (re)start from the control-context
record and the config as they are at that moment, so a replacement serves the
target the record names now. Nothing about writes is rendered: DATA judges
every write, and states every heartbeat, on the execution verdict it reads
then, the hold of the terminal's sessions included. The panel has no writes
setting of its own. FRONT is started once and outlives every replacement; if
FRONT itself exits, the panel is retracted.

A panel write belongs to the terminal's sessions. DATA gets the terminal's
hold store (``OSPREY_SESSION_HOLD_DIR``), and the sidecar keeps
``sessions.json`` in the key directory naming every session the terminal runs
now, rewritten within :data:`SESSIONS_POLL_S` of one starting or ending; DATA
is held while any of them is. No session, no hold.

Four secrets-bearing files live in a per-launch directory (mode 0700, files
0600) that is also both children's working directory: ``front.access`` (the
credential every FRONT request but ``GET /health`` must carry), ``front.read``
(the key FRONT sends DATA on every read), ``front.write`` (the key FRONT's PUT
relay sends DATA with a write, and nothing else) and ``data.keys`` (DATA's key
registry: the read key, scoped to the modules FRONT calls that DATA mounts,
and the write key, scoped to ``hare``). No secret is in an environment value
or an argv element; the children are told file names. Because the working directory holds nothing
else, no ``.env`` is read by CANOPY's settings classes.

This module imports nothing from CANOPY: CANOPY is needed by the children's
interpreter, not by the terminal.

POSIX only.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import math
import os
import secrets
import shutil
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from osprey.integrations.canopy import (
    ACCESS_HEADER,
    BACKEND_PATH,
    CHANNELS_FILE,
    COLLECTIONS_SUBPATH,
    FACILITY_COLLECTIONS_SUBPATH,
    HOST_STYLESHEETS,
    HOST_STYLESHEETS_ENV,
    LIBRARY_SUBPATH,
    MODULES,
    MOLE_ENV_PREFIX,
    NO_RECORD,
    PANEL_ID,
    PAYLOAD_DIRNAME,
    PERSONAL_SUBPATH,
    PUSH_SUBPATH,
    REACH_SINCE_ENV,
    RENDERED_FOR_ENV,
    SESSIONS_FILENAME,
    STATEMENT_FILENAME,
    VIEWS_SUBPATH,
    WRITE_BUDGET_ENV,
)
from osprey.integrations.canopy.library import (
    DEPLOYMENT_IGNORED_NAMES,
    HOME_SCREEN_KEY,
    IGNORED_NAMES_ENV,
    find_home_screen,
)
from osprey.interfaces.web_terminal.supervised_child import HeldSocket, SupervisedChild
from osprey.mcp_server.sandbox_env import drop_sandbox_denied_env
from osprey.runtime.budget import (
    GRACE_MARGIN_S,
    MAX_WRITE_BOUND_S,
    RELAY_MARGIN_S,
    WriteBudget,
    write_budget,
)
from osprey.runtime.session import (
    EXIT_BINDING_MOVED,
    EXIT_START_REFUSED,
    EXIT_TRANSPORT_STALE,
)
from osprey_connectors import control_context, types
from osprey_connectors.archive_targets import archive_targets
from osprey_connectors.safety_file import atomic_replace
from osprey_connectors.session_hold import HOLD_DIR_ENV_VAR
from osprey_connectors.workspace import repo_root_for_config

logger = logging.getLogger(__name__)

__all__ = [
    "CanopySidecar",
    "WriteBudget",
    "carried_reach",
    "data_keys_text",
    "mole_block",
    "rendered_for",
    "scrubbed_environment",
    "write_budget_text",
    "write_time_budget",
]

_NAME = "CANOPY sidecar"

#: How often the sidecar compares the terminal's live sessions with the
#: sessions file DATA reads, in seconds.
SESSIONS_POLL_S = 0.5

#: Inherited names with these prefixes never reach a child: what a child gets
#: under them is what this module renders, and nothing else.
STRIPPED_PREFIXES = ("CANOPY_", "OSPREY_CANOPY_")

#: The header the multi-user reverse proxy stamps on a request to the terminal,
#: spelled as ``deployment.web_terminals.render`` spells it.
PROXY_MARK_HEADER = "X-Osprey-Terminal-Secret"

#: The key directory's files.
ACCESS_FILENAME = "front.access"
READ_KEY_FILENAME = "front.read"
WRITE_KEY_FILENAME = "front.write"
DATA_KEYS_FILENAME = "data.keys"

#: The CANOPY modules FRONT (``canopy_mcp``) calls on DATA: its relays (hare,
#: formula-engine), its archiver forward (mole), and its tools -- ``read_pv`` and the
#: health summary (hare), ``search_pvs`` and lint (raccoon), ``convert_legacy``
#: (converter). The widget catalog is no call: FRONT reads it from the viewer's
#: own ``ir-map.json``. Restated from canopy_mcp's code, which this module does
#: not import; pinned against it by
#: ``tests/integrations/canopy/test_front_modules.py``.
FRONT_CALLED_MODULES = ("hare", "formula-engine", "mole", "raccoon", "converter")

#: The modules DATA's read key is scoped to: those FRONT calls that DATA
#: mounts. A module DATA does not mount answers 404 whatever the key holds.
FRONT_MODULES = tuple(module for module in FRONT_CALLED_MODULES if module in MODULES.split(","))

#: The modules DATA's write key is scoped to: FRONT relays exactly one
#: mutating route, Hare's ``PUT /hare/v1/pvs/{pv}``.
WRITE_MODULES = ("hare",)

#: The parts of the payload directory, as the build stages them. The widget
#: catalog FRONT lists is the viewer's ``ir-map.json``; the IR and prop schemas
#: come from the payload's ``spec/`` when it has one, which FRONT finds beside
#: the viewer on its own.
VIEWER_DIRNAME = "viewer"
TRACE_DIRNAME = "trace"
IR_MAP_FILENAME = "ir-map.json"

#: ``web.panels.canopy.vole_bytes_max`` when the config does not carry it.
DEFAULT_VOLE_BYTES_MAX = 268435456

#: Seconds FRONT's relays keep retrying a refused upstream connect.
RELAY_CONNECT_RETRY_S = "3"

#: The archiver instance name DATA's archive reader is configured under, and
#: the bound on one of its requests.
MOLE_ARCHIVER_NAME = "osprey"
MOLE_TIMEOUT_S = "10"
MOLE_TYPES = {
    types.MONGODB_ARCHIVER: "osprey.integrations.canopy.mongo_recorder:OspreyMongoRecorderAdapter",
    types.EPICS_ARCHIVER: "epics_archiver_appliance",
}

#: How long FRONT gets after SIGTERM; it holds nothing to drain.
FRONT_GRACE_S = 5.0

#: The stop grace a terminal showing this panel needs, at the write budget's
#: ceiling: FRONT's grace, then DATA's (``D + 2`` with ``D`` at most
#: :data:`~osprey.runtime.budget.MAX_WRITE_BOUND_S`), then the relay's margin.
#: ``SIDECAR_PANELS["canopy"].stop_grace_s`` spells it as a literal, because the
#: panel registry is a leaf that imports nothing; a test holds the two equal.
PANEL_STOP_GRACE_S = int(FRONT_GRACE_S + MAX_WRITE_BOUND_S + GRACE_MARGIN_S + RELAY_MARGIN_S)

#: One readiness request's own bound: a connect to a held socket succeeds
#: before any child accepts, so an unanswered request has to give up by itself.
_PROBE_TIMEOUT_S = 2.0

#: Changes into the directory named first, sends standard output where
#: standard error goes, then becomes the command that follows. The process the
#: supervisor watches keeps its pid and ends up with exactly the command's
#: argv. The supervisor itself starts a child in the terminal's working
#: directory and discards its standard output - which is where a refused start
#: says why, so without the redirect a retracted panel would carry no reason.
_CHDIR_EXEC = (
    "import os, sys; os.chdir(sys.argv[1]); os.dup2(2, 1); os.execv(sys.argv[2], sys.argv[2:])"
)


# -- rendering ---------------------------------------------------------------------


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _panel_settings(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """The ``web.panels.canopy`` block, or an empty one."""
    return _mapping(_mapping(_mapping(config.get("web")).get("panels")).get(PANEL_ID))


def scrubbed_environment(environ: Mapping[str, str]) -> dict[str, str]:
    """What both children inherit: *environ* without credentials and CANOPY names.

    The terminal's credential set goes first, then every ``CANOPY_*`` and
    ``OSPREY_CANOPY_*`` name: a key or an address left in the terminal's
    environment must not configure a child. Nothing else is removed. In
    particular the names a control session refuses to start under are passed
    through, so DATA refuses rather than runs with them.
    """
    env = drop_sandbox_denied_env(environ)
    for name in [name for name in env if name.startswith(STRIPPED_PREFIXES)]:
        del env[name]
    return env


def rendered_for(record: Any) -> str:
    """*record* as DATA compares it: ``<target>@<generation>``, or ``none``."""
    if record is None:
        return NO_RECORD
    return f"{record.target}@{record.generation}"


def carried_reach(key_dir: Path, record: Any) -> str | None:
    """Take DATA's last reachability statement: what :data:`REACH_SINCE_ENV` renders.

    Reads :data:`STATEMENT_FILENAME` in *key_dir* and removes it in every case,
    so a statement is handed on at most once. Its value is ``no,<since>`` or
    ``yes`` only when the statement was written for *record* as it is now
    (:func:`rendered_for`); a changed record identity, a missing file or one
    that does not hold a ``yes`` / ``no`` statement is ``None``.
    """
    path = key_dir / STATEMENT_FILENAME
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        logger.warning("Could not read DATA's reachability statement %s", path, exc_info=True)
        raw = None
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        logger.warning("Could not remove DATA's reachability statement %s", path, exc_info=True)
    try:
        payload = json.loads(raw) if raw is not None else None
    except ValueError:
        return None
    if not isinstance(payload, Mapping) or payload.get("binding_key") != rendered_for(record):
        return None
    state, since = payload.get("state"), payload.get("since")
    if state == "yes":
        return "yes"
    if (
        state == "no"
        and isinstance(since, int | float)
        and not isinstance(since, bool)
        and math.isfinite(since)
    ):
        return f"no,{float(since)!r}"
    return None


def write_time_budget(config: Mapping[str, Any]) -> WriteBudget:
    """The write time budget *config* gives: :func:`osprey.runtime.budget.write_budget`'s.

    Kept as the sidecar's one name for it. The budget fixes DATA's stop grace
    (``grace_s``), FRONT's relay timeout (``relay_timeout_s``) and the
    ``T,S,D`` DATA is told it may use at most.
    """
    return write_budget(config)


def write_budget_text(budget: WriteBudget) -> str:
    """*budget* as :data:`WRITE_BUDGET_ENV` carries it: ``T,S,D``."""
    return f"{budget.T},{budget.S},{budget.D}"


def mole_block(config: Mapping[str, Any], target: str | None) -> dict[str, str]:
    """The archiver variables for a DATA bound to *target*, or none.

    Rendered only when the deployment's archive holds *target*'s own history
    (:func:`~osprey_connectors.archive_targets.archive_targets`): the
    facility's appliance for ``live`` and the stand-in that rehearses it, the
    deployment's own store for ``va``. A DATA on any other target gets no
    archiver, and answers that it has none, rather than another machine's
    past. The store's password travels as the name of the variable that holds
    it.

    Args:
        config: The full config mapping.
        target: The control target of the record DATA will read; ``None`` when
            there is no record.
    """
    if target is None or target not in archive_targets(config):
        return {}
    section = config.get("archiver")
    mole_type = MOLE_TYPES.get(types.resolve_archiver_type(section))
    if mole_type is None:
        return {}
    try:
        settings = types.resolve_archiver_settings(section)
        url, extras, key_env = _archiver_address(mole_type, settings)
    except ValueError as exc:
        logger.warning("No archiver for the CANOPY panel: %s", exc)
        return {}
    instance = f"{MOLE_ENV_PREFIX}ARCHIVERS__{MOLE_ARCHIVER_NAME.upper()}__"
    block = {
        f"{MOLE_ENV_PREFIX}DEFAULT_ARCHIVER": MOLE_ARCHIVER_NAME,
        f"{instance}TYPE": mole_type,
        f"{instance}URL": url,
        f"{instance}TIMEOUT_S": MOLE_TIMEOUT_S,
    }
    for key, value in extras.items():
        block[f"{instance}EXTRAS__{key.upper()}"] = value
    if key_env:
        block[f"{instance}API_KEY_ENV"] = key_env
    return block


def _archiver_address(
    mole_type: str, settings: Mapping[str, Any]
) -> tuple[str, dict[str, str], str | None]:
    """``(url, extras, password variable name)`` of an archiver settings block.

    Raises:
        ValueError: the block lacks what the address is made of.
    """

    def need(mapping: Mapping[str, Any], key: str, where: str) -> str:
        value = mapping.get(key)
        if value is None or not str(value).strip():
            raise ValueError(f"archiver {where}{key} is not set")
        return str(value)

    raw_url = settings.get("url")
    url = raw_url.strip() if isinstance(raw_url, str) else ""
    if mole_type == MOLE_TYPES[types.EPICS_ARCHIVER]:
        if not url:
            raise ValueError("archiver url is not set")
        return url, {}, None

    from osprey_connectors.archiver import address_overrides

    auth = _mapping(settings.get("auth"))
    extras = {"collection": need(settings, "collection", "")}
    password_env = auth.get("password_env")
    key_env = str(password_env) if password_env else None
    if url:
        return url, extras, key_env
    # The address this process was given for the store wins over the config's,
    # which is the host-side one - the rule the store's own connector applies.
    override_host, override_port = address_overrides()
    host = override_host or need(settings, "host", "")
    port = override_port if override_port is not None else need(settings, "port", "")
    database = need(settings, "name", "")
    username = urllib.parse.quote(need(auth, "username", "auth."), safe="")
    source = urllib.parse.quote(need(auth, "source", "auth."), safe="")
    if key_env is None:
        raise ValueError("archiver auth.password_env is not set")
    return f"mongodb://{username}@{host}:{port}/{database}?authSource={source}", extras, key_env


def _read_config(path: str) -> dict[str, Any] | None:
    """The config at *path*, read from the file now; ``None`` when it cannot be read.

    Deliberately not the process-wide cached config: a replacement DATA is
    rendered from the file as it is when the replacement starts. An empty
    *path*, a file that cannot be read or parsed, and a document that is not a
    mapping are all ``None``.
    """
    if not path:
        return None
    try:
        from osprey_connectors.config import ConfigBuilder

        raw = ConfigBuilder(path, load_env=False).raw_config
    except Exception as exc:
        logger.warning("The config at %s could not be read (%s)", path, exc)
        return None
    if not isinstance(raw, dict):
        logger.warning("The config at %s is not a mapping", path)
        return None
    return raw


def _load_config(path: str) -> dict[str, Any]:
    """The config at *path* as :func:`_read_config` reads it; empty when it cannot be read."""
    config = _read_config(path)
    return config if config is not None else {}


def _read_record() -> Any:
    """The control-context record now, or ``None`` for every way of not having one."""
    try:
        return control_context.read_record()
    except Exception:
        logger.debug("Control context unreadable", exc_info=True)
        return None


def _write_secret(path: Path, text: str) -> None:
    """Create *path* with mode 0600 holding *text*."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)


def data_keys_text(read_key: str, write_key: str | None = None) -> str:
    """DATA's keys file: one line per key, ``<token>:<scopes>@<modules>``.

    FRONT's read key, scoped to :data:`FRONT_MODULES`, and -- when given --
    FRONT's write key, ``read+write`` scoped to :data:`WRITE_MODULES` (CANOPY's
    keys file refuses a key without ``read``).
    """
    lines = [f"{read_key}:read@{'+'.join(FRONT_MODULES)}"]
    if write_key is not None:
        lines.append(f"{write_key}:read+write@{'+'.join(WRITE_MODULES)}")
    return "".join(f"{line}\n" for line in lines)


# -- the sidecar -------------------------------------------------------------------


class CanopySidecar:
    """One user's CANOPY: FRONT and DATA behind two held sockets.

    Args:
        shared_root: The user's own agent-data root. Saved views, the
            personal panel store and the push requests live under it, so they
            are the user's and nobody else's.
        outer_prefix: The terminal's URL prefix. FRONT serves at its own root
            whatever the prefix, so this is kept only for the record.
        pinned_mode: The web theme's pinned mode; CANOPY takes its theme from
            ``web.panels.canopy.theme_file`` instead.
        payload_dir: The staged payload directory (``viewer/`` with its
            ``ir-map.json``, and, when they were built or staged, ``trace/``
            and ``spec/``). Found beside the config when not given.
        hold_root: The terminal's session hold store, rendered into DATA's
            environment; ``None`` when no watchdog runs here, and nothing is
            held.
        live_sessions: The session keys the terminal runs now. Read from the
            sidecar's own thread; ``None`` names none.
    """

    def __init__(
        self,
        shared_root: Path,
        outer_prefix: str,
        pinned_mode: str | None,
        *,
        payload_dir: Path | None = None,
        hold_root: Path | None = None,
        live_sessions: Callable[[], Sequence[str]] | None = None,
    ) -> None:
        self._shared_root = Path(shared_root)
        self._hold_root = Path(hold_root) if hold_root is not None else None
        self._live_sessions = live_sessions
        self._sessions_written: list[str] | None = None
        self._sessions_thread: threading.Thread | None = None
        self._outer_prefix = outer_prefix
        self._pinned_mode = pinned_mode
        self._payload_dir = Path(payload_dir) if payload_dir is not None else None
        self._python = sys.executable

        self._front_socket: HeldSocket | None = None
        self._data_socket: HeldSocket | None = None
        self._front: SupervisedChild | None = None
        self._data: SupervisedChild | None = None
        self._key_dir: Path | None = None
        self._token: str | None = None
        self._budget: WriteBudget | None = None
        self._front_environment: dict[str, str] = {}
        self._stopping = threading.Event()
        # Guards everything below. The exit is recorded by a child's watcher
        # thread, the callback is registered by the loop thread, and whichever
        # comes second fires it - once.
        self._exit_lock = threading.Lock()
        self._exited = False
        self._torn_down = False
        self._reported = False
        self._exit_status: int | None = None
        self._failure_tail = ""
        self._on_exit: Callable[[], None] | None = None

    # -- layout -------------------------------------------------------------

    @property
    def views_dir(self) -> Path:
        """Where this user's saved views live."""
        return self._shared_root.joinpath(*VIEWS_SUBPATH)

    @property
    def drafts_dir(self) -> Path:
        """This user's personal panel store: what the authoring tools write."""
        return self._shared_root.joinpath(*PERSONAL_SUBPATH)

    @property
    def push_dir(self) -> Path:
        """Where this user's push requests wait for the host-side review."""
        return self._shared_root.joinpath(*PUSH_SUBPATH)

    @property
    def collections_dir(self) -> Path:
        """This user's personal collection store: read-write, and theirs alone."""
        return self._shared_root.joinpath(*COLLECTIONS_SUBPATH)

    @property
    def key_dir(self) -> Path | None:
        """The per-launch key directory; ``None`` between launches."""
        return self._key_dir

    @property
    def payload_dir(self) -> Path:
        """The staged payload directory this sidecar serves from."""
        if self._payload_dir is not None:
            return self._payload_dir
        config = self._config_path()
        candidates = [Path.cwd() / PAYLOAD_DIRNAME]
        if config:
            candidates = [
                repo_root_for_config(config) / PAYLOAD_DIRNAME,
                Path(config).parent / PAYLOAD_DIRNAME,
                *candidates,
            ]
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
        return candidates[0]

    @staticmethod
    def _config_path() -> str:
        """The terminal's rendered config, absolute; empty when it has none."""
        configured = os.environ.get("OSPREY_CONFIG", "").strip()
        if not configured:
            return ""
        return os.path.abspath(os.path.expandvars(configured))

    # -- process facts ------------------------------------------------------

    @property
    def budget(self) -> WriteBudget | None:
        """The write time budget fixed at :meth:`spawn`."""
        return self._budget

    @property
    def token(self) -> str | None:
        """The access credential, so a failure reason can be shown without it."""
        return self._token

    @property
    def auth_headers(self) -> dict[str, str]:
        """What the terminal's proxy adds to every request it forwards to FRONT."""
        if self._token is None:
            raise RuntimeError(f"{_NAME} was not spawned")
        return {ACCESS_HEADER: self._token}

    @property
    def url(self) -> str:
        """FRONT's address, which the proxy forwards to; no trailing slash."""
        if self._front_socket is None:
            raise RuntimeError(f"{_NAME} was not spawned")
        return f"http://127.0.0.1:{self._front_socket.port}"

    @property
    def data_url(self) -> str:
        """DATA's address, which only FRONT is told."""
        if self._data_socket is None:
            raise RuntimeError(f"{_NAME} was not spawned")
        return f"http://127.0.0.1:{self._data_socket.port}"

    @property
    def pid(self) -> int | None:
        """FRONT's pid."""
        return self.front_pid

    @property
    def front_pid(self) -> int | None:
        return self._front.pid if self._front is not None else None

    @property
    def data_pid(self) -> int | None:
        """The live DATA's pid; ``None`` between two of them."""
        return self._data.pid if self._data is not None else None

    @property
    def data_starts(self) -> int:
        """How many DATA processes this launch has started."""
        return self._data.starts if self._data is not None else 0

    @property
    def exit_status(self) -> int | None:
        """The status of the exit that ended the sidecar; ``None`` while it runs.

        Also ``None`` after :meth:`stop`, which is not an exit of its own.
        """
        with self._exit_lock:
            return self._exit_status

    @property
    def stderr_tail(self) -> str:
        """The last stderr lines of the child that ended the sidecar, else of both."""
        with self._exit_lock:
            if self._exited:
                return self._failure_tail
        tails = [child.stderr_tail for child in (self._data, self._front) if child is not None]
        return "\n".join(tail for tail in tails if tail)

    @property
    def on_exit(self) -> Callable[[], None] | None:
        """Called once, from a worker thread, when the sidecar ends outside :meth:`stop`.

        That is DATA refusing to start, DATA crashing too often, or FRONT
        exiting. By then both children are gone and both ports are released.
        Assigning it after that already happened fires it right away.
        """
        return self._on_exit

    @on_exit.setter
    def on_exit(self, callback: Callable[[], None] | None) -> None:
        with self._exit_lock:
            self._on_exit = callback
            fire = callback is not None and self._torn_down and not self._reported
            if fire:
                self._reported = True
        if fire and callback is not None:
            callback()

    # -- preflight ----------------------------------------------------------

    def preflight(self) -> None:
        """Check what a launch needs; raise ``RuntimeError`` with one line otherwise."""
        config = self._config_path()
        if not config:
            raise RuntimeError("OSPREY_CONFIG is not set")
        if not Path(config).is_file():
            raise RuntimeError(f"the config {config} does not exist")
        for module in ("canopy", "canopy_mcp", "uvicorn"):
            if importlib.util.find_spec(module) is None:
                raise RuntimeError(f"{self._python} has no {module} installed")
        payload = self.payload_dir
        if not (payload / VIEWER_DIRNAME / "index.html").is_file():
            raise RuntimeError(f"no built viewer under {payload / VIEWER_DIRNAME}")
        if not (payload / VIEWER_DIRNAME / IR_MAP_FILENAME).is_file():
            raise RuntimeError(
                f"no widget catalog: {payload / VIEWER_DIRNAME / IR_MAP_FILENAME} does not exist"
            )

    # -- environments -------------------------------------------------------

    @staticmethod
    def _theme(config: Mapping[str, Any], config_path: str) -> dict[str, str]:
        """``CANOPY_THEME_FILE`` from ``web.panels.canopy.theme_file``.

        A relative value names a file in the project, so it resolves against the
        project root (the base ``payload_dir`` uses); an absolute one is passed
        as written.
        """
        theme_file = _panel_settings(config).get("theme_file")
        if not theme_file:
            return {}
        path = Path(str(theme_file))
        if not path.is_absolute() and config_path:
            path = repo_root_for_config(config_path) / path
        return {"CANOPY_THEME_FILE": str(path)}

    @staticmethod
    def _library_dir(config_path: str) -> str | None:
        """``CANOPY_LIBRARY_DIR``: the deployment's screen library when it exists.

        The library sits at :data:`LIBRARY_SUBPATH` below the project root and is
        looked for at every render. Without a config or without the directory
        there is none, and the front process serves the personal store alone.
        """
        if not config_path:
            logger.info("No screen library for CANOPY: OSPREY_CONFIG is not set")
            return None
        library = repo_root_for_config(config_path) / LIBRARY_SUBPATH
        if not library.is_dir():
            logger.info("No screen library for CANOPY: %s is not a directory", library)
            return None
        return str(library)

    @staticmethod
    def _home_screen(
        settings: Mapping[str, Any], config_path: str, library: str | None
    ) -> str | None:
        """``CANOPY_HOME_SCREEN``: the screen :data:`HOME_SCREEN_KEY` names.

        The home screen is configuration: unset or empty renders nothing, and
        CANOPY opens its first screen. A set id is rendered only when the
        library is rendered and holds that screen, as
        :func:`~osprey.integrations.canopy.library.find_home_screen` judges
        it; otherwise one warning names the key and the file looked for.
        """
        screen_id = settings.get("home_screen")
        if screen_id is None or screen_id == "":
            return None
        root = (
            Path(library)
            if library is not None
            else repo_root_for_config(config_path) / LIBRARY_SUBPATH
        )
        found = find_home_screen(root, screen_id)
        if library is not None and found.found:
            return str(screen_id)
        problem = found.problem if not found.found else "the screen library is not rendered"
        logger.warning(
            "CANOPY opens its first screen: %s is %r, but %s (%s). %s",
            HOME_SCREEN_KEY,
            screen_id,
            problem,
            found.looked_in,
            found.fix or "",
        )
        return None

    @staticmethod
    def _facility_collections_dir(config_path: str) -> str | None:
        """``CANOPY_FACILITY_COLLECTIONS_DIR``: the facility collections when they exist.

        The store sits at :data:`FACILITY_COLLECTIONS_SUBPATH` below the project
        root, beside the screen library, and is looked for at every render.
        Without a config or without the directory there is none, and the front
        process lists the personal collections alone.
        """
        if not config_path:
            logger.info("No facility collections for CANOPY: OSPREY_CONFIG is not set")
            return None
        store = repo_root_for_config(config_path) / FACILITY_COLLECTIONS_SUBPATH
        if not store.is_dir():
            logger.info("No facility collections for CANOPY: %s is not a directory", store)
            return None
        return str(store)

    def data_env(self) -> dict[str, str]:
        """DATA's complete environment, from the record and the config as they are now.

        Rendered at every (re)start; the write budget is the one fixed at
        :meth:`spawn`, so DATA's bound never outgrows the stop grace and the
        relay timeout fixed with it. The terminal's hold store is named when
        there is one. The
        reachability the previous DATA stated last is taken here too
        (:func:`carried_reach`) and rendered only while the record is the one
        it was stated for.
        """
        key_dir = self._require_key_dir()
        budget = self._require_budget()
        config_path = self._config_path()
        read = _read_config(config_path)
        config = read if read is not None else {}
        record = _read_record()
        settings = _panel_settings(config)

        env = scrubbed_environment(os.environ)
        env.update(self._theme(config, config_path))
        if config_path:
            env["CONFIG_FILE"] = config_path
            env["OSPREY_CONFIG"] = config_path
        try:
            vole_bytes_max = int(settings.get("vole_bytes_max", DEFAULT_VOLE_BYTES_MAX))
        except (TypeError, ValueError):
            vole_bytes_max = DEFAULT_VOLE_BYTES_MAX
        env.update(
            {
                "CANOPY_EPICS__BACKEND": BACKEND_PATH,
                "CANOPY_MODULES": MODULES,
                "CANOPY_API_KEYS_FILE": str(key_dir / DATA_KEYS_FILENAME),
                "CANOPY_CLOSED_META": "1",
                "CANOPY_VOLE__GLOBAL_BYTES_MAX": str(vole_bytes_max),
                RENDERED_FOR_ENV: rendered_for(record),
                WRITE_BUDGET_ENV: write_budget_text(budget),
            }
        )
        if self._hold_root is not None:
            env[HOLD_DIR_ENV_VAR] = str(self._hold_root)
        carried = carried_reach(key_dir, record)
        if carried is not None:
            env[REACH_SINCE_ENV] = carried
        env.update(mole_block(config, record.target if record is not None else None))
        return env

    def front_env(self) -> dict[str, str]:
        """FRONT's complete environment, as rendered at :meth:`spawn`.

        The PUT relay is always on: whether a write may happen is DATA's to
        refuse, per request, and whether the viewer asks a person first is
        DATA's ``asked`` claim. FRONT is not told either.
        """
        key_dir = self._require_key_dir()
        budget = self._require_budget()
        config_path = self._config_path()
        config = _load_config(config_path) if config_path else {}
        settings = _panel_settings(config)
        payload = self.payload_dir

        env = scrubbed_environment(os.environ)
        env.update(self._theme(config, config_path))
        env.update(
            {
                "CANOPY_BASE_URL": self.data_url,
                "CANOPY_SELF_URL": self.url,
                "CANOPY_ACCESS_TOKEN_FILE": str(key_dir / ACCESS_FILENAME),
                "CANOPY_API_KEY_FILE": str(key_dir / READ_KEY_FILENAME),
                # The personal store: read-write, and this user's alone.
                "CANOPY_DRAFTS_DIR": str(self.drafts_dir),
                "CANOPY_PUSH_DIR": str(self.push_dir),
                # The personal collection store: read-write, and this user's alone.
                "CANOPY_COLLECTIONS_DIR": str(self.collections_dir),
                "CANOPY_VIEWS_DIR": str(self.views_dir),
                "CANOPY_VIEWER_DIST": str(payload / VIEWER_DIRNAME),
                # The path the terminal's proxy serves this panel under.
                "CANOPY_URL_PREFIX": f"/panel/{PANEL_ID}",
                # The operator secret marks a request as having come through
                # that proxy, which keeps a browser off the agent's endpoint.
                "CANOPY_PROXY_MARK_HEADERS": PROXY_MARK_HEADER,
                # What the image build prunes: a library path may not use it.
                IGNORED_NAMES_ENV: " ".join(DEPLOYMENT_IGNORED_NAMES),
                # The authoring tools write into the personal store; the
                # library refuses every write on its own.
                "CANOPY_AUTHORING": "1",
                # The workspace hub and its two tools: this FRONT serves one
                # operator from one process, the only kind that may serve it.
                "CANOPY_WORKSPACE": "1",
                # The screenshot browser runs without Chromium's own sandbox,
                # under the container's seccomp profile alone.
                "CANOPY_BROWSER_SANDBOX": "0",
                "CANOPY_RELAY_CONNECT_RETRY_S": RELAY_CONNECT_RETRY_S,
                "CANOPY_TRACE_DEFAULT_QUERY": str(settings.get("trace_default_query") or ""),
                "CANOPY_CORRELATION_DEFAULT_QUERY": str(
                    settings.get("correlation_default_query") or ""
                ),
                "CANOPY_WAVEFORM_DEFAULT_QUERY": str(settings.get("waveform_default_query") or ""),
                # The build's channel catalog, at the one path every web
                # container mounts it at; a container without the mount leaves
                # FRONT's /channels a 404.
                "CANOPY_CHANNELS_FILE": CHANNELS_FILE,
                # The sheets that restyle the embedded pages with OSPREY's
                # tokens, in cascade order; always named, as the web terminal
                # serves all three.
                HOST_STYLESHEETS_ENV: " ".join(HOST_STYLESHEETS),
                "CANOPY_PUT_RELAY": "1",
                "CANOPY_PUT_RELAY_TIMEOUT_S": str(budget.relay_timeout_s),
                "CANOPY_WRITE_API_KEY_FILE": str(key_dir / WRITE_KEY_FILENAME),
            }
        )
        library = self._library_dir(config_path)
        if library is not None:
            env["CANOPY_LIBRARY_DIR"] = library
        home = self._home_screen(settings, config_path, library)
        if home is not None:
            env["CANOPY_HOME_SCREEN"] = home
        facility_collections = self._facility_collections_dir(config_path)
        if facility_collections is not None:
            env["CANOPY_FACILITY_COLLECTIONS_DIR"] = facility_collections
        trace = payload / TRACE_DIRNAME
        if (trace / "index.html").is_file():
            env["CANOPY_TRACE_DIST"] = str(trace)
        return env

    def _require_key_dir(self) -> Path:
        if self._key_dir is None:
            raise RuntimeError(f"{_NAME} was not spawned")
        return self._key_dir

    def _require_budget(self) -> WriteBudget:
        if self._budget is None:
            raise RuntimeError(f"{_NAME} was not spawned")
        return self._budget

    # -- commands -----------------------------------------------------------

    def _front_command(self, fd: int) -> list[str]:
        return [self._python, "-m", "canopy_mcp", "--http", "--fd", str(fd)]

    def _data_command(self, fd: int) -> list[str]:
        return [self._python, "-m", "uvicorn", "canopy.main:app", "--fd", str(fd)]

    def _in_key_dir(self, command: list[str]) -> list[str]:
        """*command*, started with the key directory as its working directory."""
        return [self._python, "-c", _CHDIR_EXEC, str(self._require_key_dir()), *command]

    # -- process lifecycle --------------------------------------------------

    def spawn(self) -> None:
        """Bind the sockets, write the key files and start both children.

        Blocks for the two forks only.
        """
        if self._front is not None or self._data is not None:
            raise RuntimeError(f"{_NAME} is already running")
        self._stopping.clear()
        with self._exit_lock:
            self._exited = self._torn_down = self._reported = False
            self._exit_status = None
            self._failure_tail = ""

        self._budget = write_time_budget(_load_config(self._config_path()))

        self._key_dir = Path(tempfile.mkdtemp(prefix="osprey-canopy-sidecar-"))
        self._token = secrets.token_urlsafe(32)
        read_key = secrets.token_urlsafe(32)
        write_key = secrets.token_urlsafe(32)
        _write_secret(self._key_dir / ACCESS_FILENAME, self._token + "\n")
        _write_secret(self._key_dir / READ_KEY_FILENAME, read_key + "\n")
        _write_secret(self._key_dir / WRITE_KEY_FILENAME, write_key + "\n")
        _write_secret(self._key_dir / DATA_KEYS_FILENAME, data_keys_text(read_key, write_key))
        self._sessions_written = None
        self._write_sessions()
        self._sessions_thread = threading.Thread(
            target=self._watch_sessions, name="canopy-sessions", daemon=True
        )
        self._sessions_thread.start()

        self._front_socket = front_socket = HeldSocket()
        self._data_socket = data_socket = HeldSocket()
        front_fd, data_fd = front_socket.fd, data_socket.fd
        self._front_environment = self.front_env()

        self._data = SupervisedChild(
            lambda: self._in_key_dir(self._data_command(data_fd)),
            self.data_env,
            grace_s=self._budget.grace_s,
            respawn_now_codes={EXIT_BINDING_MOVED, EXIT_TRANSPORT_STALE},
            never_respawn_codes={EXIT_START_REFUSED},
            on_give_up=lambda: self._child_ended(self._data),
            pass_fds=(data_fd,),
            name="CANOPY data process",
        )
        # FRONT is not replaced: its first exit ends the supervision.
        self._front = SupervisedChild(
            lambda: self._in_key_dir(self._front_command(front_fd)),
            lambda: self._front_environment,
            grace_s=FRONT_GRACE_S,
            respawn_now_codes=(),
            never_respawn_codes=(),
            on_give_up=lambda: self._child_ended(self._front),
            pass_fds=(front_fd,),
            name="CANOPY front process",
            give_up_after=1,
        )
        self._data.start()
        self._front.start()

    # -- the terminal's sessions -------------------------------------------

    def _write_sessions(self) -> None:
        """Rewrite :data:`SESSIONS_FILENAME` when the terminal's live sessions changed.

        Never raises. A read that fails (a registry changed under it) keeps
        the file as it is, for the next tick.
        """
        key_dir = self._key_dir
        if key_dir is None:
            return
        try:
            keys = sorted(set(self._live_sessions())) if self._live_sessions is not None else []
        except Exception:
            logger.debug("The terminal's sessions could not be read; retrying", exc_info=True)
            return
        if keys == self._sessions_written:
            return
        try:
            atomic_replace(
                key_dir / SESSIONS_FILENAME,
                json.dumps(keys).encode(),
                mode=0o600,
                sync="best_effort",
            )
        except OSError:
            logger.warning("Could not write the CANOPY sessions file", exc_info=True)
            return
        self._sessions_written = keys

    def _watch_sessions(self) -> None:
        """Keep the sessions file current until :meth:`stop`."""
        while not self._stopping.wait(SESSIONS_POLL_S):
            self._write_sessions()

    def _child_ended(self, child: SupervisedChild | None) -> None:
        """A child's supervision ended on its own: end the sidecar and report it.

        Runs on that child's watcher thread.
        """
        if child is None:
            return
        with self._exit_lock:
            if self._stopping.is_set() or self._exited:
                return
            self._exited = True
            status = child.exit_status
            self._exit_status = status if status is not None else -1
            self._failure_tail = child.stderr_tail
        self._stop_children()
        with self._exit_lock:
            self._torn_down = True
            callback = self._on_exit
            if callback is not None and not self._stopping.is_set():
                self._reported = True
            else:
                callback = None
        if callback is not None:
            callback()

    def _stop_children(self) -> None:
        """Stop FRONT, then DATA, then release both ports. Idempotent."""
        # FRONT first, so nothing new reaches DATA while it drains.
        for child in (self._front, self._data):
            if child is not None:
                child.stop()
        for held in (self._front_socket, self._data_socket):
            if held is not None:
                held.close()

    def _answers(self, url: str, expected: int) -> bool:
        """Whether a bare ``GET`` of *url* is answered with status *expected*."""
        try:
            with urllib.request.urlopen(url, timeout=_PROBE_TIMEOUT_S) as response:
                return int(response.status) == expected
        except urllib.error.HTTPError as exc:
            return exc.code == expected
        except (urllib.error.URLError, OSError):
            return False

    def wait_ready(self, timeout: float) -> None:
        """Block until both children serve and FRONT asks for the credential.

        Ready means DATA and FRONT each answer ``GET /health``, and FRONT
        refuses ``GET /config.json`` without the access header with 401 - a
        FRONT that would serve without the credential is not published.

        Raises:
            RuntimeError: the sidecar ended, :meth:`stop` ran, or *timeout*
                seconds passed, before that. The exit and timeout messages
                carry the stderr tail.
        """
        if self._front is None or self._data is None:
            raise RuntimeError(f"{_NAME} was not spawned")
        front, data = self.url, self.data_url
        checks = ((f"{data}/health", 200), (f"{front}/health", 200), (f"{front}/config.json", 401))
        deadline = time.monotonic() + timeout
        passed = 0
        while True:
            if self._stopping.is_set():
                raise RuntimeError(f"{_NAME} was stopped before it was ready")
            with self._exit_lock:
                exited, status = self._exited, self._exit_status
            if exited:
                raise RuntimeError(
                    f"{_NAME} exited with status {status} before it was ready\n{self.stderr_tail}"
                )
            while passed < len(checks) and self._answers(*checks[passed]):
                passed += 1
            if passed == len(checks):
                logger.info("%s ready at %s", _NAME, front)
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"{_NAME} did not answer within {timeout:.0f} s\n{self.stderr_tail}"
                )
            time.sleep(0.1)

    def stop(self) -> None:
        """Stop both children, release the ports and remove the key directory.

        DATA gets the write budget's grace to drain before it is killed.
        Idempotent, and fires no ``on_exit``.
        """
        self._stopping.set()
        self._stop_children()
        thread, self._sessions_thread = self._sessions_thread, None
        if thread is not None:
            thread.join(timeout=2 * SESSIONS_POLL_S)
        self._front = self._data = None
        key_dir, self._key_dir = self._key_dir, None
        if key_dir is not None:
            shutil.rmtree(key_dir, ignore_errors=True)
