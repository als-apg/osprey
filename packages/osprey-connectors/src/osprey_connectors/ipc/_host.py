"""The connector-host child: the process that owns the control-system client.

One target, one child. The parent (the controls MCP server, or a
:class:`~osprey_connectors.ipc.ConnectorHostPool`) holds no ``libca``
and never sets an ``EPICS_CA_*`` variable of its own; it launches this module,
which builds the real connector through the ordinary
:class:`~osprey_connectors.factory.ConnectorFactory` and serves the proxy
surface over the frame codec. Switching targets is therefore a process
lifecycle operation rather than an in-process reconfiguration, which is the
only way to change an EPICS environment that is read once, process-wide, by a
library that has already loaded.

Launch contract
---------------
The child is spawned as::

    python -m osprey_connectors.ipc._host

with **stdin** and **stdout** connected to pipes and **stderr** left free for
diagnostics. There are no command-line arguments and no configuration in
``argv``: everything arrives on the wire, so nothing about a deployment shows
up in ``ps``.

``stdout`` is the frame channel and carries nothing else. To make that true
regardless of what a control-system library decides to print, the child
duplicates its stdout to a private file descriptor at startup and then points
file descriptor 1 at stderr. A stray ``print()`` — from this module, from
libca, from anything imported below — lands on stderr and cannot desynchronise
the stream. Frames are written to the private descriptor with :func:`os.write`.

The init frame
--------------
The **first frame the parent sends** is a request whose method is ``init`` and
whose kwargs are a :class:`HostInit`'s fields, as the plain mapping
``model_dump(exclude_none=True)`` writes::

    {
      "control_system": {...},          # the full control_system config section
      "target": "live" | "va",          # the control target, never defaulted
      "config_file": "/abs/config.yml", # optional: the project config to read
      "execution_mode": "readonly"      # optional: restrict this child to reads
    }

The child validates them as a :class:`HostInit` once, before it builds
anything. ``control_system`` is passed whole rather than re-read from disk, so the child
builds from exactly the section the parent resolved against. ``target`` is
resolved to a connector type by :func:`osprey_connectors.targets.resolve_target`
— the same shared resolver the parent and the executor sandbox use, so no
holder can decide privately which machine ``live`` means.

Writes are deliberately **not** grantable by launch payload. The connector's
write guard decides every write by the write verdict
(:func:`osprey_connectors.write_switch.write_verdict`): the write tier of the
project config's ``control_system`` section — not the section in the init
frame — and the login's switch for the machine, read from the record this
child's environment names. ``execution_mode`` here can only *add* the
``OSPREY_EXECUTION_MODE=readonly`` claim, never clear an inherited one. A child
cannot be talked into writing by the way it was started.

Environment scrub
-----------------
Before anything else — before the init frame is read, before the connectors
package is imported — every inherited ``EPICS_CA_*`` and ``EPICS_PVA_*``
variable is removed from ``os.environ``. Ambient environment must never
contribute to what this child talks to: the endpoint is derived by
``connect()`` from the config section alone, and the post-connect report below
is only meaningful because anything left in the environment afterwards was put
there by ``connect()`` itself.

Start marks
-----------
A supervisor may hand the child the write end of a pipe and name its
descriptor in :data:`START_MARKS_FD_ENV`. The child then writes one line per
start phase it completes, ``"<name> <time.monotonic()>\n"``, in the order of
:data:`START_PHASES`, and closes the descriptor before it serves:

* ``main`` — the interpreter is up and this module, with everything it
  imports, is loaded;
* ``init`` — the init frame has been read;
* ``imports`` — every built-in connector and archiver module is imported;
* ``connected`` — the connector is built: its own module imported and its
  ``connect()`` returned.

The time between two marks is the time the later phase took. The marks are
diagnostics beside the frame channel, never part of it: a child handed no
descriptor writes none, and a supervisor that reads none learns nothing less
from the frames.

The post-connect report
-----------------------
The **first frame the child sends** is the result frame answering ``init``. It
carries what ``connect()`` actually configured, which is what the parent
asserts its own derivation against. The child builds it as a :class:`HostReport`
and sends its fields as the plain mapping ``model_dump(by_alias=True)`` writes,
keyed as below (``_epics_configured`` is the alias of the record's
``epics_configured``); the parent validates it as one once per start::

    {
      "selected_role": "read_only" | "write_access" | None,
      "mode":          "name_server" | "addr_list" | None,
      "host":          str | None,
      "port":          int | str | None,
      "_epics_configured": bool,
      # diagnostics, alongside the five verification fields above
      "connector_type": str,     # what `target` resolved to
      "transport":      str | None,  # the wire the built connector speaks
      "target":         str,
      "writes_enabled": bool,    # the posture the gateway selection was made
      "readonly_run":   bool,    # on, and the run mode (see below)
      "authorization":  str | None,  # the write authorization it names
      "epics_env":      {...},   # EPICS_CA_*/EPICS_PVA_* set by connect()
      "pid":            int,
    }

``mode``, ``host`` and ``port`` are read back out of the environment
``connect()`` installed, not out of the config that was requested.
``selected_role`` is the gateway role whose configured endpoint matches what
was installed; where both roles name the same endpoint — which makes them
operationally indistinguishable — the tie is broken by the rule ``connect()``
applies (``write_access`` when writes are enabled, a write gateway exists, and
the run is not readonly; otherwise ``read_only``, or ``write_access`` on a block
that configures no ``read_only`` gateway).

``writes_enabled`` is the connector instance's own posture — whether the write
verdict allows a write to this child's target — because that is the value
``connect()`` selected on, and the parent verifies the reported role against a
derivation made with the same verdict. In a readonly run, where that verdict
refuses for every type, the write tier is reported instead and ``readonly_run``
names the reason. A child that names a program's write authorization
(``OSPREY_WRITE_AUTHORIZATION``, which the pool sets for a key that names one)
reports it in ``authorization``, and its ``writes_enabled`` is the
authorization's verdict, channels aside: a child spawned under a live
authorization selects the write gateway with the machine's switch off.

**A connector that configures no CA environment reports all five as
null/false**, and that is a well-formed report rather than a degraded one: the
simulator served in process talks to no gateway, so ``selected_role``, ``mode``, ``host``
and ``port`` are ``None`` and ``_epics_configured`` is ``False``. There is
nothing for the parent to verify because there is no endpoint to get wrong.

Served methods
--------------
``read_channel``, ``read_multiple_channels``, ``write_channel``,
``write_multiple_channels``, ``validate_channel``, ``disconnect`` — forwarded
to the connector with the kwargs the frame carried, the batch read's
``addresses`` positionally as the connector takes them — plus ``spawn_probe``
and ``ping``.

``ping {}`` answers with this child's pid and touches nothing else. It is how a
supervisor tells a child that is merely slow — a batched or confirmed write can
legitimately take several of its call's timeouts — from one that is wedged:
every request is served as its own task, so a ping is answered alongside any
number of slow calls, and goes unanswered only when this process's event loop
itself is stuck.

A batched read is **one round trip**: ``read_multiple_channels`` fans out
concurrently *inside* the child through the connector's own implementation, so
N channels cost one request frame and one result frame rather than N of each.
Requests are dispatched as tasks and may complete in any order; the parent
matches replies by ``request_id``.

``spawn_probe {address, timeout_s}`` is the readiness check the parent runs
against a freshly spawned child before it swaps traffic over: a real read of
one named channel, bounded by ``timeout_s``, returning that channel's
:class:`~osprey_connectors.control_system.ChannelValue`. It is a distinct
method because a probe is allowed to fail without meaning anything is wrong
with the child — a failed probe aborts a switch, and this child keeps serving.

**A request-level failure never kills the child.** Every exception raised while
serving a request is encoded into an error frame for that ``request_id`` by
:mod:`osprey_connectors.ipc._exceptions`, and the serve loop continues. A
request frame that does not decode — a payload node the codec does not know, a
record its own class refuses — is refused the same way, in an error frame for
the request it names; a frame that names no request is logged and skipped. Only
a stream that has lost its framing, which no later frame can be read past, ends
the child.

Lifecycle
---------
The child exits:

- with 0 when stdin reaches **EOF** — the parent closed the pipe or died;
- with 0 after a ``disconnect`` request, once outstanding work has finished and
  the acknowledgement has been written;
- with :data:`EXIT_ORPHANED` when the **watchdog** thread finds it
  reparented: the parent PID differs from the one recorded at start (init or a
  subreaper), because its parent died without closing the pipe. This holds
  because the child is spawned directly by its supervisor, with no wrapper
  process between them; the recorded parent is the supervisor itself. A
  polling thread is used rather than a signal: ``PR_SET_PDEATHSIG`` does not
  exist on macOS and there is no ``/proc`` to watch, and a thread polling
  ``getppid()`` behaves identically on both platforms;
- with :data:`EXIT_INIT_FAILED` when the first frame is not a usable ``init``
  or the connector fails to connect. Either way a typed error frame naming the
  failure is written first, so the parent learns why rather than seeing only a
  dead process.

Exit is abrupt (:func:`os._exit`) after the streams are flushed: this process
holds a Channel Access context whose interpreter-shutdown handlers can block,
and a child that will not die is worse for the parent than one that skips its
atexit hooks.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import os
import stat
import sys
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Literal, Protocol, SupportsFloat, get_args

from pydantic import Field

from osprey_connectors.config_sections import ControlSystemSection, GatewayMode, SelectedRole
from osprey_connectors.connector_types import (
    CONTROL_TARGETS,
    ControlTarget,
    Transport,
    is_control_target,
)
from osprey_connectors.epics_env import EPICS_ENV_PREFIXES, scrub_epics_env
from osprey_connectors.errors import ConnectorInternalError
from osprey_connectors.execution_mode import (
    EXECUTION_MODE_ENV_VAR,
    READONLY_MODE,
    ExecutionMode,
    is_readonly_run,
)
from osprey_connectors.ipc import _frames as frames
from osprey_connectors.ipc._exceptions import encode_exception
from osprey_connectors.model import FrozenModel, ReadOnlyMapping

if TYPE_CHECKING:
    from collections.abc import Awaitable, Mapping

    from osprey_connectors.control_system._base import ChannelValue, ControlSystemConnector

__all__ = [
    "START_MARKS_FD_ENV",
    "START_PHASES",
    "WRITE_METHODS",
    "HostInit",
    "HostMethod",
    "HostReport",
    "ProxyMethod",
    "StartPhase",
    "SupervisorMethod",
]

#: Clean shutdown: pipe EOF, or an acknowledged ``disconnect``.
EXIT_OK = 0

#: The init frame was unusable, or ``connect()`` failed. An error frame
#: describing the failure is always written before exiting with this code.
EXIT_INIT_FAILED = 2

#: The watchdog found the process reparented: the parent PID differs from the
#: one recorded at start (init or a subreaper), so the parent is gone.
EXIT_ORPHANED = 3

#: Names the inherited descriptor a child writes its start marks to.
START_MARKS_FD_ENV = "OSPREY_CONNECTOR_HOST_START_MARKS_FD"

#: One start mark a child writes (see "Start marks" above).
StartPhase = Literal["main", "init", "imports", "connected"]

#: The start marks a child writes, in the order it reaches them.
START_PHASES: tuple[StartPhase, ...] = get_args(StartPhase)


class _StartMarks:
    """Writes this child's start marks to the descriptor its supervisor passed.

    The variable naming the descriptor is removed from the environment as it
    is read, so nothing this child starts inherits a number that means
    something else there. A descriptor that is not a pipe is never written to.
    """

    def __init__(self) -> None:
        self._fd: int | None = None
        raw = os.environ.pop(START_MARKS_FD_ENV, None)
        if raw is None or not raw.isdigit():
            return
        with contextlib.suppress(OSError):
            if stat.S_ISFIFO(os.fstat(int(raw)).st_mode):
                self._fd = int(raw)

    def mark(self, phase: StartPhase) -> None:
        if self._fd is None:
            return
        try:
            os.write(self._fd, f"{phase} {time.monotonic():.6f}\n".encode())
        except OSError:
            self._fd = None

    def close(self) -> None:
        if self._fd is not None:
            with contextlib.suppress(OSError):
                os.close(self._fd)
            self._fd = None


#: A connector method the child serves for the proxy, forwarded verbatim.
ProxyMethod = Literal[
    "read_channel",
    "read_multiple_channels",
    "write_channel",
    "write_multiple_channels",
    "validate_channel",
    "disconnect",
]

#: A method of the child's own that a supervisor calls: ``init`` (the
#: configuration frame, which must arrive first), ``ping`` and ``spawn_probe``.
SupervisorMethod = Literal["init", "ping", "spawn_probe"]

#: Every method word a request frame may carry.
HostMethod = ProxyMethod | SupervisorMethod

#: The connector methods the child forwards, in the order its refusal lists them.
PROXY_METHODS: tuple[ProxyMethod, ...] = get_args(ProxyMethod)

#: The connector methods that can move the machine. ``write_channel_checked``
#: is composed from ``write_channel`` in the parent, so it is that one call.
WRITE_METHODS: frozenset[ProxyMethod] = frozenset({"write_channel", "write_multiple_channels"})

#: How often the watchdog asks whether it still has a parent.
WATCHDOG_INTERVAL_S = 1.0

#: Bound applied to ``spawn_probe`` when the caller names none.
DEFAULT_PROBE_TIMEOUT_S = 5.0

_READ_CHUNK_BYTES = 65536

# Named for the public namespace, not this private module, so a logging config
# keyed on the name keeps matching whatever the module is called.
logger = logging.getLogger("osprey_connectors.ipc.host")


# --------------------------------------------------------------------------
# Startup: stdout protection
# --------------------------------------------------------------------------


def _claim_frame_channel() -> int:
    """Take stdout for frames alone, and point stray writes at stderr.

    Returns:
        A private duplicate of the original stdout, which is from here on the
        only way to reach the parent's read pipe. File descriptor 1 becomes a
        duplicate of stderr, so anything that prints — this module, libca,
        rich, a library's import-time banner — is diagnostics rather than
        stream corruption.
    """
    sys.stdout.flush()
    channel = os.dup(sys.stdout.fileno())
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    return channel


def _write_frame(fd: int, payload: bytes) -> None:
    """Write one whole frame, looping over short writes."""
    view = memoryview(payload)
    while view:
        view = view[os.write(fd, view) :]


def _parent_changed(parent_pid: int) -> bool:
    """Whether this process's parent is no longer *parent_pid*.

    A parent of init counts as changed even when it is the one recorded: a
    supervisor that died before the record was taken has already handed the
    child to init, and a supervisor is never init itself.
    """
    parent = os.getppid()
    return parent != parent_pid or parent == 1


def _start_watchdog(interval: float = WATCHDOG_INTERVAL_S, *, parent_pid: int) -> threading.Thread:
    """Start the daemon thread that exits when this process is orphaned.

    Args:
        interval: Seconds between polls.
        parent_pid: The parent PID recorded at start.
    """

    def _poll() -> None:
        while True:
            if _parent_changed(parent_pid):
                logger.warning(
                    "connector host: parent %s is gone (now reparented to %s); exiting",
                    parent_pid,
                    os.getppid(),
                )
                _exit_now(EXIT_ORPHANED)
            time.sleep(interval)

    thread = threading.Thread(target=_poll, name="connector-host-watchdog", daemon=True)
    thread.start()
    return thread


def _exit_now(code: int) -> None:
    """Flush the diagnostics and leave, without running interpreter shutdown."""
    try:
        sys.stderr.flush()
        _save_coverage()
    finally:
        os._exit(code)  # pragma: no cover - ends the process before the tracer records it


def _save_coverage() -> None:
    """Write coverage data a measured run collected in this child, if any.

    ``os._exit`` skips the atexit hook coverage.py saves from, so without this a
    test run that measures subprocesses sees none of the code a child ran. Only
    a run that already imported coverage has anything to save; everywhere else
    this is a dictionary lookup.
    """
    coverage = sys.modules.get("coverage")
    if coverage is None:  # pragma: no cover - only true outside a measured run
        return
    with contextlib.suppress(Exception):
        current = coverage.Coverage.current()
        if current is not None:
            current.stop()
            current.save()


# --------------------------------------------------------------------------
# Post-connect report
# --------------------------------------------------------------------------


def _installed_endpoint() -> tuple[GatewayMode | None, str | None, int | str | None]:
    """The CA endpoint currently installed in the environment.

    Read *after* ``connect()`` and after the startup scrub, so whatever is
    found here was put there by ``connect()`` and by nothing else.

    Returns:
        ``(mode, host, port)``, all ``None`` when no CA gateway was configured.
    """
    name_servers = os.environ.get("EPICS_CA_NAME_SERVERS")
    if name_servers:
        host, separator, port = name_servers.rpartition(":")
        if not separator:
            return "name_server", name_servers, None
        return "name_server", host, _as_port(port)

    addr_list = os.environ.get("EPICS_CA_ADDR_LIST")
    if addr_list:
        return "addr_list", addr_list, _as_port(os.environ.get("EPICS_CA_SERVER_PORT"))

    return None, None, None


def _as_port(value: str | None) -> int | str | None:
    """A port as the int it usually is, or verbatim when it is not one."""
    if value is None or value == "":
        return None
    return int(value) if value.isdigit() else value


def _writes_enabled_input() -> bool:
    """The DEPLOYMENT half of this child's write posture — the write tier alone.

    :func:`osprey_connectors.write_tier.tier_allows` over the project config's
    ``control_system`` section: whether this deployment's write tier allows
    writes at all, before the switch, the owner or the target are asked. The
    tier is one per section, so the connector type this child serves does not
    change the answer.

    This is **not** the posture ``connect()`` selected its gateway on. The
    connector's own ``_writes_enabled`` is the whole write verdict — run mode,
    tier, target, owner, the login's switch for the machine and the launch pin —
    and that property is what :func:`_selection_writes_enabled` reads for the
    report.

    What is left for this one is the readonly-run mirror. A readonly run
    collapses the instance's verdict to ``False``, which would erase the
    deployment's answer from the report; reported here instead, the pair
    ``(writes_enabled, readonly_run)`` still says that a tier allowing writes
    was held on the read gateway by the run mode. The reported role is
    unaffected, because the gateway rule (:func:`_rule_role`) never selects
    ``write_access`` for a readonly run.

    The section is read from the project config rather than taken from the init
    payload for the same reason the connector reads it there. A config that
    cannot be read allows nothing.
    """
    from osprey_connectors.config import current_config
    from osprey_connectors.write_tier import tier_allows

    try:
        section = current_config().control_system
    except (FileNotFoundError, KeyError, RuntimeError):
        return False
    return tier_allows(section)


def _selection_writes_enabled(connector: ControlSystemConnector) -> bool:
    """The write posture ``connect()`` actually selected this child's gateway on.

    The connector instance's own property rather than a second derivation from
    config: it is the whole rule — the deployment ceiling for its type, the run
    mode, and the operator's narrowing for the target the factory stamped on it
    — and it is the value ``connect()`` read to choose between the
    ``write_access`` and ``read_only`` gateways. Reading it here is what keeps
    the report and the connected role the same answer.

    Fail-closed on the same exception set ``connect()`` catches around that
    property, plus an instance that carries no such property at all: a
    connector with no posture to read went through no write gateway.
    """
    try:
        return bool(connector._writes_enabled)
    except (AttributeError, FileNotFoundError, KeyError, RuntimeError):
        return False


def _named_authorization() -> str | None:
    """The program's write authorization this child names, for its report; ``None`` for none.

    The child's ``writes_enabled`` and its gateway were decided on it
    (:func:`~osprey_connectors.write_authorization.authorized_write_verdict`);
    the connectors' write guard reads it from the environment, and this asks
    the same reader.
    """
    from osprey_connectors.control_system._base import _named_authorization

    return _named_authorization()


def _rule_role(
    gateways: dict[str, object], writes_enabled: bool, readonly_run: bool
) -> SelectedRole | None:
    """The gateway role ``connect()`` selects, given its inputs; ``None`` with no gateways.

    The rule itself is :func:`osprey_connectors.control_system._gateway_role._selected_role`,
    the one place it lives: this only adds that an empty table selects nothing.
    """
    from osprey_connectors.control_system._gateway_role import _selected_role

    if not gateways:
        return None
    return _selected_role(gateways, writes_enabled=writes_enabled, readonly_run=readonly_run)


def _endpoint_matches(gateway: object, mode: str | None, host: str | None, port: object) -> bool:
    """Whether a gateway block describes the endpoint that was installed.

    A block with no ``port`` matches on address and mode alone: an unset port
    is default-filled at connect time (the virtual accelerator follows
    ``services.virtual_accelerator.port``), so the config never carried the
    number the environment now shows.
    """
    if not isinstance(gateway, dict) or not gateway:
        return False
    if str(gateway.get("address", "")) != str(host or ""):
        return False
    if "port" in gateway and str(gateway["port"]) != str(port):
        return False
    from osprey_connectors.config import config_flag

    name_server = config_flag(gateway.get("use_name_server"), key="gateways.use_name_server")
    return ("name_server" if name_server else "addr_list") == mode


def _installed_role(
    gateways: dict[str, object],
    mode: str | None,
    host: str | None,
    port: object,
    writes_enabled: bool,
    readonly_run: bool,
) -> SelectedRole | None:
    """Which gateway role the connector actually went through.

    Evidence first: the role whose configured endpoint matches the environment
    that was installed. The rule ``connect()`` applies is used only to break a
    tie between blocks that name the same endpoint, and the answer is ``None``
    when nothing was installed at all — a connector that configures no gateway
    went through no role.
    """
    if mode is None:
        return None
    ruled = _rule_role(gateways, writes_enabled, readonly_run)
    if ruled and _endpoint_matches(gateways.get(ruled), mode, host, port):
        return ruled
    matched = [
        name for name, gateway in gateways.items() if _endpoint_matches(gateway, mode, host, port)
    ]
    if len(matched) != 1:
        return None
    # A table key that is no role names none: the report says no role was used.
    return next((role for role in get_args(SelectedRole) if role == matched[0]), None)


class HostInit(FrozenModel):
    """What a connector-host child is started for: its ``init`` frame (see the module docstring).

    A supervisor builds one and sends ``model_dump(exclude_none=True)``, the
    mapping the frame has always carried; the child validates it once.

    Attributes:
        control_system: The ``control_system`` config section, whole.
        target: The control target the child serves.
        config_file: The project config the child reads its write posture
            and limits from, or ``None`` for the one
            :func:`~osprey_connectors.config.locate_config` finds in its
            environment.
        execution_mode: ``"readonly"`` for a child that refuses every write
            whatever the config arms, or ``None``. It can only take writes
            away.
    """

    control_system: ReadOnlyMapping[str, object]
    target: ControlTarget
    config_file: str | None = None
    execution_mode: ExecutionMode | None = None


class HostReport(FrozenModel):
    """What a child's ``connect()`` configured: its answer to ``init`` (see the module docstring).

    The child builds one and sends ``model_dump(by_alias=True)``; the parent
    validates the reply as one once per start, so a reply that is not one is
    refused before anything verifies it. The field order is the order the
    child has always written the keys in.

    Attributes:
        selected_role: The gateway role whose endpoint was installed, or ``None``.
        mode: How the installed endpoint reaches EPICS, or ``None``.
        host: The installed endpoint's host, or ``None``.
        port: The installed endpoint's port, or ``None``.
        epics_configured: Whether ``connect()`` configured an EPICS gateway at
            all; ``_epics_configured`` on the wire.
        connector_type: What the target resolved to.
        transport: The wire the built connector speaks, or ``None``.
        target: The control target the child serves.
        writes_enabled: The posture the gateway selection was made on.
        readonly_run: Whether the child is in a readonly run.
        authorization: The write authorization the child names, or ``None``.
        epics_env: The ``EPICS_CA_*``/``EPICS_PVA_*`` variables ``connect()`` set.
        pid: The child's process id.
    """

    selected_role: SelectedRole | None
    mode: GatewayMode | None
    host: str | None
    port: int | str | None
    epics_configured: bool = Field(alias="_epics_configured")
    connector_type: str
    transport: Transport | None
    target: ControlTarget
    writes_enabled: bool
    readonly_run: bool
    authorization: str | None
    epics_env: ReadOnlyMapping[str, str]
    pid: int


def _init_block(value: object, where: str) -> Mapping[str, object]:
    """A nested block of the init payload's section: empty when unset, refused when no mapping.

    Raises:
        ValueError: *value* is set but is not a mapping.
    """
    if not value:
        return {}
    if not isinstance(value, dict):
        raise ValueError(
            f"connector host init: {where} must be a mapping, not a {type(value).__name__}"
        )
    return value


def _post_connect_report(
    connector: ControlSystemConnector,
    connector_type: str,
    target: ControlTarget,
    section: Mapping[str, object],
) -> HostReport:
    """Describe what ``connect()`` configured, for the parent to verify.

    Raises:
        ValueError: The section's ``connector`` block, or its block for
            *connector_type*, is set but is not a mapping.
    """
    connectors = _init_block(section.get("connector"), "control_system.connector")
    connector_block = _init_block(
        connectors.get(connector_type), f"control_system.connector.{connector_type}"
    )
    gateways = connector_block.get("gateways") or {}
    if not isinstance(gateways, dict):
        gateways = {}

    readonly_run = is_readonly_run()
    if readonly_run:
        # The instance's posture collapses to False for every type in a readonly
        # run, which would erase the deployment's answer from the report. The
        # deployment half is reported instead, so the (writes_enabled,
        # readonly_run) pair still names the run mode as what held an armed
        # target on the read gateway. The gateway rule never selects
        # write_access for a readonly run, so the role this report names is the
        # same either way.
        writes_enabled = _writes_enabled_input()
    else:
        writes_enabled = _selection_writes_enabled(connector)
    mode, host, port = _installed_endpoint()

    return HostReport(
        selected_role=_installed_role(gateways, mode, host, port, writes_enabled, readonly_run),
        mode=mode,
        host=host,
        port=port,
        epics_configured=bool(getattr(connector, "_epics_configured", False)),
        connector_type=connector_type,
        transport=getattr(connector, "transport", None),
        target=target,
        writes_enabled=writes_enabled,
        readonly_run=readonly_run,
        authorization=_named_authorization(),
        epics_env={
            name: value for name, value in os.environ.items() if name.startswith(EPICS_ENV_PREFIXES)
        },
        pid=os.getpid(),
    )


# --------------------------------------------------------------------------
# Connector construction
# --------------------------------------------------------------------------


def _host_init(kwargs: object) -> HostInit:
    """The ``init`` frame's kwargs as a :class:`HostInit`, validated once.

    The section, when it is not a mapping, and the target, when it is not a
    control target, are refused in their own sentences before the record
    validates the rest; an unset or empty section is an empty one.

    Raises:
        TypeError: *kwargs* is not a mapping.
        ValueError: The section is no mapping, the target is no control
            target, or another field is not what :class:`HostInit` holds
            (a ``pydantic.ValidationError``).
    """
    if not isinstance(kwargs, dict):
        raise TypeError(
            f"connector host 'init' kwargs are a {type(kwargs).__name__}, not a mapping"
        )
    section = kwargs.get("control_system") or {}
    if not isinstance(section, dict):
        raise ValueError("connector host init: 'control_system' must be the config section dict")
    target = kwargs.get("target")
    if not is_control_target(target):
        raise ValueError(
            f"Unknown control target {target!r} in the connector host init frame; "
            f"valid targets are {', '.join(CONTROL_TARGETS)}."
        )
    return HostInit.model_validate({**kwargs, "control_system": section})


def _install_config_file(config_file: str) -> None:
    """Make *config_file* this child's process config, and name it to anything it loads.

    The file is installed as the process default, so every lookup the
    connector makes reads it; ``OSPREY_CONFIG`` names it as well, for framework
    code the connector loads that locates its own config (and ``CONFIG_FILE``,
    which some framework readers still read). The working directory's env
    chain is applied first, as the lazy default would before its load.

    A file that does not load installs nothing: every lookup then locates the
    same file through ``OSPREY_CONFIG`` and fails as it did before the child
    installed anything, so a reader that degrades on a missing config (writes
    off, no limits validator) degrades the same way.
    """
    from osprey_connectors.config import install_config, load_config
    from osprey_connectors.dotenv import load_project_dotenv
    from osprey_connectors.errors import ConfigLoadError, ConfigurationError

    os.environ["OSPREY_CONFIG"] = config_file
    os.environ["CONFIG_FILE"] = config_file
    load_project_dotenv()
    try:
        install_config(load_config(config_file))
    except (ConfigLoadError, ConfigurationError) as exc:
        logger.debug(f"connector host: {config_file} did not load ({exc}); nothing installed")


async def _build_connector(
    init: HostInit, marks: _StartMarks | None = None
) -> tuple[ControlSystemConnector, HostReport]:
    """Apply the init payload and bring up the connector it describes.

    Args:
        init: The ``init`` frame's payload, as documented in the module
            docstring.
        marks: Where the ``imports`` and ``connected`` start marks are written,
            or ``None`` to write none.

    Returns:
        The live connector and its post-connect report.

    Raises:
        ValueError: The target cannot be resolved on this deployment.
        Exception: Whatever ``connect()`` raises. The caller turns it into an
            error frame; this child does not start serving on a failed connect.
    """
    # A plain dict, as the frame carried it: the factory is handed the section.
    section = dict(init.control_system)
    target = init.target

    if init.config_file:
        # The connector reads writes_enabled and the limits database — the home
        # of the per-channel confirm policy — from the project config, so the
        # child must be pointed at the same file the parent resolved against.
        _install_config_file(init.config_file)
    if init.execution_mode == READONLY_MODE:
        # Restriction only: an inherited readonly claim is never cleared here.
        os.environ[EXECUTION_MODE_ENV_VAR] = READONLY_MODE

    from osprey_connectors import connector_types, targets
    from osprey_connectors.factory import ConnectorFactory

    read = ControlSystemSection.read(section)
    connector_type = targets.resolve_target(read, target)
    if marks is not None:
        marks.mark("imports")

    # The section is passed whole: writes_enabled and the limits block,
    # per-channel confirm included, travel with it, and the connector sub-block
    # is already keyed by the resolved type. Its type is restated only when the
    # target's type differs from the section's own resolved type, because the
    # serving leaf is read under the section's own spelling (no stated type is
    # the simulator in process; a stated virtual_accelerator without the leaf is
    # the served one), so restating a type the section already resolves to
    # would build the other venue.
    config = (
        section
        if connector_types.resolve_control_system_type(read) == connector_type
        else {**section, "type": connector_type}
    )
    # The target the parent pointed this child at — already validated above as
    # one of the three literals — is what indexes the per-target narrowing the reference monitor
    # reads. This child serves exactly one target, so the stamp is the payload's.
    connector = await ConnectorFactory.create_control_system_connector(
        ControlSystemSection.read(config), control_target=target
    )
    if marks is not None:
        marks.mark("connected")

    report = _post_connect_report(connector, connector_type, target, section)
    logger.warning(
        "connector host: serving target %r as %r (role=%r mode=%r %s:%s)",
        target,
        connector_type,
        report.selected_role,
        report.mode,
        report.host,
        report.port,
    )
    return connector, report


# --------------------------------------------------------------------------
# Serving
# --------------------------------------------------------------------------


class _Undecodable:
    """A delimited frame this child could not decode, in its place in the stream."""

    def __init__(self, request_id: str | None, cause: Exception) -> None:
        #: The request the frame named, or ``None`` when it named none that reads.
        self.request_id = request_id
        self.cause = cause


class _FrameStream:
    """Frames arriving on stdin, one await at a time."""

    def __init__(self, reader: asyncio.StreamReader) -> None:
        self._reader = reader
        self._parser = frames.FrameReader()
        self._ready: deque[bytes] = deque()

    async def next(
        self,
    ) -> frames.RequestFrame | frames.ResultFrame | frames.ErrorFrame | _Undecodable | None:
        """The next frame, or ``None`` once the pipe reaches EOF.

        Each frame is decoded on its own, so one that does not decode comes back
        as an :class:`_Undecodable` and costs nothing past itself.

        Raises:
            FrameDecodeError: The stream has lost its framing.
        """
        while not self._ready:
            chunk = await self._reader.read(_READ_CHUNK_BYTES)
            if not chunk:
                return None
            self._ready.extend(self._parser.feed_raw(chunk))
        raw = self._ready.popleft()
        try:
            return frames.decode_frame(raw)
        except Exception as exc:
            return _Undecodable(frames.request_id_of(raw), exc)


class _FrameWriter:
    """Serialized writes to the private frame descriptor.

    The lock keeps two concurrently completing requests from interleaving their
    bytes, and the write itself runs in a thread so a parent that has stopped
    reading stalls one reply rather than the whole event loop.
    """

    def __init__(self, fd: int) -> None:
        self._fd = fd
        self._lock = asyncio.Lock()

    async def send(self, payload: bytes) -> None:
        async with self._lock:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _write_frame, self._fd, payload)


def _error_frame(request_id: str, exc: BaseException) -> bytes:
    """Encode a failure for one request, normalising unknown classes here.

    :func:`~osprey_connectors.ipc._exceptions.encode_exception` decides what
    crosses typed; a class outside its registry is turned, at this edge, into
    the nearest registered class it subclasses or else into a
    ``ConnectorInternalError`` carrying the original ``repr``, so the parent is
    never handed a tag it cannot rebuild. A value that defeats the codec itself
    still produces a frame, as the same ``ConnectorInternalError`` — losing the
    reply entirely would leave the parent waiting on a request that already
    failed.
    """
    body = encode_exception(exc)
    try:
        return frames.encode_error(request_id, body)
    except Exception as encode_failure:  # pragma: no cover - defensive
        return frames.encode_error(
            request_id,
            encode_exception(
                ConnectorInternalError(
                    f"connector host could not encode its error ({encode_failure}): {body.message}",
                    class_tag=body.class_tag,
                )
            ),
        )


async def _refuse(frame: _Undecodable, writer: _FrameWriter) -> None:
    """Answer a frame that did not decode: an error frame for its request, when it named one."""
    logger.warning(
        "connector host: refusing a frame for request %r that does not decode: %r",
        frame.request_id,
        frame.cause,
    )
    if frame.request_id is not None:
        await writer.send(_error_frame(frame.request_id, frame.cause))


async def _spawn_probe(
    connector: ControlSystemConnector,
    address: object,
    *,
    timeout_s: object = DEFAULT_PROBE_TIMEOUT_S,
) -> ChannelValue:
    """Read one named channel under a hard bound.

    Args:
        connector: The live connector.
        address: Channel address to read.
        timeout_s: Seconds to allow, defaulting to
            :data:`DEFAULT_PROBE_TIMEOUT_S`. The bound is applied here as well
            as being passed to the connector: a control-system client that
            ignores its own timeout must not be able to hang the switch that is
            waiting on this probe.

    Returns:
        The channel's :class:`ChannelValue`.

    Raises:
        TypeError: The channel is not an address, or the bound is not a number.
        ValueError: The bound is not a positive number.
        TimeoutError: The read did not finish inside the bound. The message
            always names the channel and the bound, because the two bounds race
            and ``asyncio.wait_for`` usually wins with a *bare*
            :class:`TimeoutError` — one whose empty message renders, several
            layers up, as a switch refusal ending in ": ." that names nothing
            the operator could act on.
    """
    if not isinstance(address, str):
        raise TypeError(f"spawn_probe address must be a channel address, got {address!r}")
    if timeout_s is None:
        bound = DEFAULT_PROBE_TIMEOUT_S
    elif isinstance(timeout_s, str | SupportsFloat):
        bound = float(timeout_s)
    else:
        raise TypeError(f"spawn_probe timeout_s must be a number, got {timeout_s!r}")
    if bound <= 0:
        raise ValueError(f"spawn_probe timeout_s must be positive, got {timeout_s!r}")
    try:
        return await asyncio.wait_for(
            connector.read_channel(address, timeout_s=bound), timeout=bound
        )
    except TimeoutError as exc:
        reason = f"probe read of {address!r} timed out after {bound}s"
        # A connector that reported its own timeout said something more
        # specific; it is kept, behind the subject it did not name.
        raise TimeoutError(f"{reason}: {exc}" if str(exc) else reason) from exc


class _ServedMethod(Protocol):
    """A :data:`PROXY_METHODS` member as the child calls it: keywords in, an awaitable out."""

    def __call__(self, **kwargs: object) -> Awaitable[object]: ...


async def _invoke(connector: ControlSystemConnector, method: object, kwargs: object) -> object:
    """Run one request against the connector.

    *method* and *kwargs* are the request frame's, unvalidated: a bad one is
    refused here, in that request's error frame.
    """
    if method == "ping":
        return os.getpid()
    if not isinstance(kwargs, dict):
        raise TypeError(
            f"connector host request kwargs are a {type(kwargs).__name__}, not a mapping"
        )
    if method == "spawn_probe":
        return await _spawn_probe(connector, **kwargs)
    if method == "read_multiple_channels":
        # The frame names the batch's addresses; the connector takes them positionally.
        rest = dict(kwargs)
        addresses = rest.pop("addresses", None)
        if not isinstance(addresses, list) or not all(isinstance(a, str) for a in addresses):
            raise TypeError("connector host read_multiple_channels needs a list of addresses")
        served_batch: _ServedMethod = functools.partial(connector.read_multiple_channels, addresses)
        return await served_batch(**rest)
    if isinstance(method, str) and method in PROXY_METHODS:
        served: _ServedMethod = getattr(connector, method)
        return await served(**kwargs)
    raise ValueError(
        f"connector host does not serve {method!r}; it serves "
        f"{', '.join((*PROXY_METHODS, 'spawn_probe', 'ping'))}"
    )


async def _dispatch(
    connector: ControlSystemConnector, frame: frames.RequestFrame, writer: _FrameWriter
) -> None:
    """Serve one request and write exactly one reply for it."""
    try:
        payload = frames.encode_result(
            frame.request_id, await _invoke(connector, frame.method, frame.kwargs)
        )
    except Exception as exc:
        logger.warning("connector host: %s failed: %r", frame.method, exc)
        payload = _error_frame(frame.request_id, exc)
    await writer.send(payload)


async def _serve(
    connector: ControlSystemConnector, stream: _FrameStream, writer: _FrameWriter
) -> int:
    """Dispatch requests until the pipe closes or a disconnect is served."""
    inflight: set[asyncio.Task[None]] = set()
    while True:
        frame = await stream.next()
        if frame is None:
            logger.warning("connector host: stdin closed; exiting")
            break
        if isinstance(frame, _Undecodable):
            await _refuse(frame, writer)
            continue
        if not isinstance(frame, frames.RequestFrame):
            # Results and errors travel the other way; a parent that sends one
            # is confused, and answering it would only confuse it further.
            logger.warning("connector host: ignoring unexpected %s frame", type(frame).__name__)
            continue
        if frame.method == "disconnect":
            # Finish what is already running before the connector is torn out
            # from under it, then acknowledge and go.
            if inflight:
                await asyncio.gather(*inflight, return_exceptions=True)
            await _dispatch(connector, frame, writer)
            return EXIT_OK

        task = asyncio.create_task(_dispatch(connector, frame, writer))
        inflight.add(task)
        task.add_done_callback(inflight.discard)

    if inflight:
        await asyncio.gather(*inflight, return_exceptions=True)
    return EXIT_OK


async def _stdin_frames() -> _FrameStream:
    """Wrap stdin in a stream reader without spending a thread on it."""
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin.buffer)
    return _FrameStream(reader)


async def _run(channel_fd: int, marks: _StartMarks) -> int:
    """Read the init frame, report, then serve until told otherwise."""
    stream = await _stdin_frames()
    writer = _FrameWriter(channel_fd)

    init = await stream.next()
    if init is None:
        logger.warning("connector host: stdin closed before the init frame; exiting")
        return EXIT_OK
    marks.mark("init")
    if isinstance(init, _Undecodable):
        await _refuse(init, writer)
        return EXIT_INIT_FAILED
    if not isinstance(init, frames.RequestFrame) or init.method != "init":
        got = init.method if isinstance(init, frames.RequestFrame) else type(init).__name__
        # A protocol violation is a bug in the sender, not a reachability answer.
        await writer.send(
            _error_frame(
                init.request_id,
                ConnectorInternalError(
                    f"connector host expected a 'init' request as its first frame, got {got!r}",
                    class_tag=type(init).__name__,
                ),
            )
        )
        return EXIT_INIT_FAILED

    try:
        connector, report = await _build_connector(_host_init(init.kwargs), marks)
    except Exception as exc:
        logger.warning("connector host: connect failed: %r", exc)
        await writer.send(_error_frame(init.request_id, exc))
        return EXIT_INIT_FAILED
    finally:
        # Nothing the child serves, and nothing it starts while serving, holds
        # the supervisor's pipe open.
        marks.close()

    await writer.send(frames.encode_result(init.request_id, report.model_dump(by_alias=True)))
    return await _serve(connector, stream, writer)


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m osprey_connectors.ipc._host``."""
    parent = os.getppid()
    marks = _StartMarks()
    marks.mark("main")
    del argv  # the launch contract carries no arguments; config arrives on the wire
    removed = scrub_epics_env()
    channel_fd = _claim_frame_channel()
    # Handlers on stderr, explicitly: a record that reached stdout would sit in
    # the middle of a frame. Nothing here calls the framework's
    # configure_logging(), which is an entry-point concern.
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format="%(name)s: %(message)s")
    if removed:
        logger.warning("connector host: scrubbed inherited %s", ", ".join(sorted(removed)))

    _start_watchdog(WATCHDOG_INTERVAL_S, parent_pid=parent)
    try:
        return asyncio.run(_run(channel_fd, marks))
    except KeyboardInterrupt:  # pragma: no cover - parent-initiated
        return EXIT_OK


if __name__ == "__main__":
    _exit_now(main(sys.argv[1:]))
