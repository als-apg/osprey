"""The connector-host child: the process that owns the control-system client.

One target, one child. The parent (the controls MCP server) holds no ``libca``
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

    python -m osprey_connectors.ipc.host

with **stdin** and **stdout** connected to pipes and **stderr** left free for
diagnostics. There are no command-line arguments and no configuration in
``argv``: everything arrives on the wire, so nothing about a deployment shows
up in ``ps``.

``stdout`` is the frame channel and carries nothing else. To make that true
regardless of what a control-system library decides to print, the child
duplicates its stdout to a private file descriptor at startup and then points
file descriptor 1 at stderr. A stray ``print()`` — from this module, from
pyepics, from anything imported below — lands on stderr and cannot desynchronise
the stream. Frames are written to the private descriptor with :func:`os.write`.

The init frame
--------------
The **first frame the parent sends** is a request whose method is ``init`` and
whose kwargs are::

    {
      "control_system": {...},          # the full control_system config section
      "target": "live" | "va",          # the session target, never defaulted
      "config_file": "/abs/config.yml", # optional: what CONFIG_FILE to read
      "execution_mode": "readonly"      # optional: restrict this child to reads
    }

``control_system`` is passed whole rather than re-read from disk, so the child
builds from exactly the section the parent resolved against. ``target`` is
resolved to a connector type by :func:`osprey_connectors.types.resolve_target`
— the same shared resolver the parent and the executor sandbox use, so no
holder can decide privately which machine ``live`` means.

Writes are deliberately **not** grantable by launch payload. The connector's
write guard reads the posture of its own connector type from the project config
— ``control_system.connector.<type>.writes_enabled``, which inherits
``control_system.writes_enabled`` when that block says nothing about it — and
``execution_mode`` here can only *add* the ``OSPREY_EXECUTION_MODE=readonly``
claim, never clear an inherited one. A child cannot be talked into writing by
the way it was started; only the deployment's own config enables writes.

Environment scrub
-----------------
Before anything else — before the init frame is read, before the connectors
package is imported — every inherited ``EPICS_CA_*`` and ``EPICS_PVA_*``
variable is removed from ``os.environ``. Ambient environment must never
contribute to what this child talks to: the endpoint is derived by
``connect()`` from the config section alone, and the post-connect report below
is only meaningful because anything left in the environment afterwards was put
there by ``connect()`` itself.

The post-connect report
-----------------------
The **first frame the child sends** is the result frame answering ``init``. It
carries what ``connect()`` actually configured, which is what the parent
asserts its own derivation against::

    {
      "selected_role": "read_only" | "write_access" | None,
      "mode":          "name_server" | "addr_list" | None,
      "host":          str | None,
      "port":          int | str | None,
      "_epics_configured": bool,
      # diagnostics, alongside the five verification fields above
      "connector_type": str,     # what `target` resolved to
      "target":         str,
      "writes_enabled": bool,    # that type's posture, and the run mode: the
      "readonly_run":   bool,    # two inputs the gateway selection was made with
      "epics_env":      {...},   # EPICS_CA_*/EPICS_PVA_* set by connect()
      "pid":            int,
    }

``mode``, ``host`` and ``port`` are read back out of the environment
``connect()`` installed, not out of the config that was requested.
``selected_role`` is the gateway role whose configured endpoint matches what
was installed; where both roles name the same endpoint — which makes them
operationally indistinguishable — the tie is broken by the rule ``connect()``
applies (``write_access`` when writes are enabled, a write gateway exists, and
the run is not readonly; otherwise ``read_only``).

**A connector that configures no CA environment reports all five as
null/false**, and that is a well-formed report rather than a degraded one: the
mock connector talks to no gateway, so ``selected_role``, ``mode``, ``host``
and ``port`` are ``None`` and ``_epics_configured`` is ``False``. There is
nothing for the parent to verify because there is no endpoint to get wrong.

Served methods
--------------
``read_channel``, ``read_multiple_channels``, ``write_channel``,
``write_multiple_channels``, ``disconnect`` — forwarded to the connector with
the kwargs the frame carried — plus ``spawn_probe``.

A batched read is **one round trip**: ``read_multiple_channels`` fans out
concurrently *inside* the child through the connector's own implementation, so
N channels cost one request frame and one result frame rather than N of each.
Requests are dispatched as tasks and may complete in any order; the parent
matches replies by ``request_id``.

``spawn_probe {channel, timeout}`` is the readiness check the parent runs
against a freshly spawned child before it swaps traffic over: a real read of
one named channel, bounded by ``timeout``, returning that channel's
:class:`~osprey_connectors.control_system.base.ChannelValue`. It is a distinct
method because a probe is allowed to fail without meaning anything is wrong
with the child — a failed probe aborts a switch, and this child keeps serving.

**A request-level failure never kills the child.** Every exception raised while
serving a request is encoded into an error frame for that ``request_id`` by
:mod:`osprey_connectors.ipc.exceptions`, and the serve loop continues.

Lifecycle
---------
The child exits:

- with 0 when stdin reaches **EOF** — the parent closed the pipe or died;
- with 0 after a ``disconnect`` request, once outstanding work has finished and
  the acknowledgement has been written;
- with :data:`EXIT_ORPHANED` when the **watchdog** thread sees
  ``os.getppid() == 1``, meaning it has been reparented to init because its
  parent died without closing the pipe. A polling thread is used rather than a
  signal: ``PR_SET_PDEATHSIG`` does not exist on macOS and there is no
  ``/proc`` to watch, and a thread polling ``getppid()`` behaves identically on
  both platforms;
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
import logging
import os
import sys
import threading
import time
from collections import deque
from typing import Any

from osprey_connectors.ipc import frames
from osprey_connectors.ipc.exceptions import encode_exception

__all__ = [
    "EXIT_INIT_FAILED",
    "EXIT_OK",
    "EXIT_ORPHANED",
    "INIT_METHOD",
    "PROXY_METHODS",
    "main",
    "scrub_epics_env",
]

#: Clean shutdown: pipe EOF, or an acknowledged ``disconnect``.
EXIT_OK = 0

#: The init frame was unusable, or ``connect()`` failed. An error frame
#: describing the failure is always written before exiting with this code.
EXIT_INIT_FAILED = 2

#: The watchdog found the process reparented to init: the parent is gone.
EXIT_ORPHANED = 3

#: Method name of the configuration frame, which must arrive first.
INIT_METHOD = "init"

#: Connector methods forwarded verbatim. ``spawn_probe`` is served separately
#: because it is this module's own method, not the connector's.
PROXY_METHODS = (
    "read_channel",
    "read_multiple_channels",
    "write_channel",
    "write_multiple_channels",
    "disconnect",
)

#: Every inherited variable with one of these prefixes is dropped at startup.
EPICS_ENV_PREFIXES = ("EPICS_CA_", "EPICS_PVA_")

#: How often the watchdog asks whether it still has a parent.
WATCHDOG_INTERVAL_S = 1.0

#: Bound applied to ``spawn_probe`` when the caller names none.
DEFAULT_PROBE_TIMEOUT_S = 5.0

_READ_CHUNK_BYTES = 65536

logger = logging.getLogger("osprey_connectors.ipc.host")


# --------------------------------------------------------------------------
# Startup: environment scrub and stdout protection
# --------------------------------------------------------------------------


def scrub_epics_env(environ: dict[str, str] | None = None) -> dict[str, str]:
    """Remove every inherited ``EPICS_CA_*``/``EPICS_PVA_*`` variable.

    Args:
        environ: Mapping to scrub, defaulting to :data:`os.environ`.

    Returns:
        The removed name/value pairs, for logging. Nothing is put back: an
        endpoint this child was not configured with must not be reachable from
        it, and a variable that survived the scrub would silently widen what
        ``connect()`` decided.
    """
    env = os.environ if environ is None else environ
    removed = {}
    for name in [name for name in env if name.startswith(EPICS_ENV_PREFIXES)]:
        removed[name] = env.pop(name)
    return removed


def _claim_frame_channel() -> int:
    """Take stdout for frames alone, and point stray writes at stderr.

    Returns:
        A private duplicate of the original stdout, which is from here on the
        only way to reach the parent's read pipe. File descriptor 1 becomes a
        duplicate of stderr, so anything that prints — this module, pyepics,
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


def _start_watchdog(interval: float = WATCHDOG_INTERVAL_S) -> threading.Thread:
    """Start the daemon thread that exits when this process is orphaned."""

    def _poll() -> None:
        while True:
            if os.getppid() == 1:
                logger.warning("connector host: parent is gone (getppid() == 1); exiting")
                _exit_now(EXIT_ORPHANED)
            time.sleep(interval)

    thread = threading.Thread(target=_poll, name="connector-host-watchdog", daemon=True)
    thread.start()
    return thread


def _exit_now(code: int) -> None:
    """Flush the diagnostics and leave, without running interpreter shutdown."""
    try:
        sys.stderr.flush()
    finally:
        os._exit(code)


# --------------------------------------------------------------------------
# Post-connect report
# --------------------------------------------------------------------------


def _installed_endpoint() -> tuple[str | None, str | None, Any]:
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


def _as_port(value: str | None) -> Any:
    """A port as the int it usually is, or verbatim when it is not one."""
    if value is None or value == "":
        return None
    return int(value) if value.isdigit() else value


def _writes_enabled_input(connector_type: str) -> bool:
    """Whether this child's connector type is armed for writes.

    Write posture is per connector type, so the answer depends on which type
    this child is serving. That type is the one the init frame already resolved
    and the factory stamped on the connector, threaded in rather than derived a
    second time here: a report that re-read the config for a type would be free
    to name one the child is not running.

    Deliberately the same resolver, the same type and the same fail-closed
    exception set the connector's own gateway selection uses, because this is
    the input that decided which gateway it selected. Reporting a different
    answer than the one the selection was made with would make the parent's
    verification compare two unrelated things.

    The section is read from the project config rather than taken from the init
    payload for the same reason: the connector reads its posture there.
    """
    from osprey_connectors.config import get_config_value
    from osprey_connectors.types import type_writes_enabled

    try:
        section = get_config_value("control_system", {})
    except (FileNotFoundError, KeyError, RuntimeError):
        return False
    return type_writes_enabled(section, connector_type)


def _rule_role(gateways: dict[str, Any], writes_enabled: bool, readonly_run: bool) -> str | None:
    """The gateway role ``connect()``'s rule selects, given its inputs."""
    if writes_enabled and not readonly_run and gateways.get("write_access"):
        return "write_access"
    return "read_only" if gateways.get("read_only") else None


def _endpoint_matches(gateway: Any, mode: str | None, host: str | None, port: Any) -> bool:
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
    return ("name_server" if gateway.get("use_name_server", False) else "addr_list") == mode


def _selected_role(
    gateways: dict[str, Any],
    mode: str | None,
    host: str | None,
    port: Any,
    writes_enabled: bool,
    readonly_run: bool,
) -> str | None:
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
    return matched[0] if len(matched) == 1 else None


def _post_connect_report(
    connector: Any, connector_type: str, target: str, section: dict[str, Any]
) -> dict[str, Any]:
    """Describe what ``connect()`` configured, for the parent to verify."""
    from osprey_connectors.control_system.base import is_readonly_run

    connector_block = (section.get("connector") or {}).get(connector_type) or {}
    gateways = connector_block.get("gateways") or {}
    if not isinstance(gateways, dict):
        gateways = {}

    writes_enabled = _writes_enabled_input(connector_type)
    readonly_run = is_readonly_run()
    mode, host, port = _installed_endpoint()

    return {
        "selected_role": _selected_role(gateways, mode, host, port, writes_enabled, readonly_run),
        "mode": mode,
        "host": host,
        "port": port,
        "_epics_configured": bool(getattr(connector, "_epics_configured", False)),
        "connector_type": connector_type,
        "target": target,
        "writes_enabled": writes_enabled,
        "readonly_run": readonly_run,
        "epics_env": {
            name: value for name, value in os.environ.items() if name.startswith(EPICS_ENV_PREFIXES)
        },
        "pid": os.getpid(),
    }


# --------------------------------------------------------------------------
# Connector construction
# --------------------------------------------------------------------------


async def _build_connector(payload: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Apply the init payload and bring up the connector it describes.

    Args:
        payload: The ``init`` frame's kwargs, as documented in the module
            docstring.

    Returns:
        The live connector and its post-connect report.

    Raises:
        ValueError: The payload names no usable target, or the target cannot be
            resolved on this deployment.
        Exception: Whatever ``connect()`` raises. The caller turns it into an
            error frame; this child does not start serving on a failed connect.
    """
    section = payload.get("control_system") or {}
    target = payload.get("target")
    if not isinstance(section, dict):
        raise ValueError("connector host init: 'control_system' must be the config section dict")

    config_file = payload.get("config_file")
    if config_file:
        # The connector reads writes_enabled, the limits database and the
        # verification defaults from the project config, so the child must be
        # pointed at the same file the parent resolved against.
        os.environ["CONFIG_FILE"] = str(config_file)
    if payload.get("execution_mode") == "readonly":
        # Restriction only: an inherited readonly claim is never cleared here.
        os.environ["OSPREY_EXECUTION_MODE"] = "readonly"

    from osprey_connectors import types
    from osprey_connectors.factory import ConnectorFactory, register_builtin_connectors

    connector_type = types.resolve_target(section, target)
    register_builtin_connectors()

    # The section is passed whole with only its type replaced: writes_enabled,
    # write_verification and the limits block travel with it, and the connector
    # sub-block is already keyed by the resolved type.
    config = {**section, "type": connector_type}
    connector = await ConnectorFactory.create_control_system_connector(config)

    report = _post_connect_report(connector, connector_type, str(target), section)
    logger.warning(
        "connector host: serving target %r as %r (role=%r mode=%r %s:%s)",
        target,
        connector_type,
        report["selected_role"],
        report["mode"],
        report["host"],
        report["port"],
    )
    return connector, report


# --------------------------------------------------------------------------
# Serving
# --------------------------------------------------------------------------


class _FrameStream:
    """Frames arriving on stdin, one await at a time."""

    def __init__(self, reader: asyncio.StreamReader) -> None:
        self._reader = reader
        self._parser = frames.FrameReader()
        self._ready: deque[Any] = deque()

    async def next(self) -> Any:
        """The next frame, or ``None`` once the pipe reaches EOF."""
        while not self._ready:
            chunk = await self._reader.read(_READ_CHUNK_BYTES)
            if not chunk:
                return None
            self._ready.extend(self._parser.feed(chunk))
        return self._ready.popleft()


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

    :func:`~osprey_connectors.ipc.exceptions.encode_exception` decides what
    crosses typed; a class outside its registry is turned into a
    ``ConnectionError`` carrying the original ``repr`` at this edge, so the
    parent is never handed a tag it cannot rebuild. A value that defeats the
    codec itself still produces a frame — losing the reply entirely would leave
    the parent waiting on a request that already failed.
    """
    body = encode_exception(exc)
    if body["class_tag"] != type(exc).__name__:
        exc = ConnectionError(body["message"])
    try:
        return frames.encode_error(request_id, exc)
    except Exception as encode_failure:  # pragma: no cover - defensive
        return frames.encode_error(
            request_id,
            ConnectionError(f"connector host could not encode its error ({encode_failure}): {exc}"),
        )


async def _spawn_probe(
    connector: Any, channel: str, timeout: float | None = DEFAULT_PROBE_TIMEOUT_S
) -> Any:
    """Read one named channel under a hard bound.

    Args:
        connector: The live connector.
        channel: Channel address to read.
        timeout: Seconds to allow, defaulting to
            :data:`DEFAULT_PROBE_TIMEOUT_S`. The bound is applied here as well
            as being passed to the connector: a control-system client that
            ignores its own timeout must not be able to hang the switch that is
            waiting on this probe.

    Returns:
        The channel's :class:`ChannelValue`.

    Raises:
        ValueError: The bound is not a positive number.
        TimeoutError: The read did not finish inside the bound. The message
            always names the channel and the bound, because the two bounds race
            and ``asyncio.wait_for`` usually wins with a *bare*
            :class:`TimeoutError` — one whose empty message renders, several
            layers up, as a switch refusal ending in ": ." that names nothing
            the operator could act on.
    """
    bound = DEFAULT_PROBE_TIMEOUT_S if timeout is None else float(timeout)
    if bound <= 0:
        raise ValueError(f"spawn_probe timeout must be positive, got {timeout!r}")
    try:
        return await asyncio.wait_for(connector.read_channel(channel, timeout=bound), timeout=bound)
    except TimeoutError as exc:
        reason = f"probe read of {channel!r} timed out after {bound}s"
        # A connector that reported its own timeout said something more
        # specific; it is kept, behind the subject it did not name.
        raise TimeoutError(f"{reason}: {exc}" if str(exc) else reason) from exc


async def _invoke(connector: Any, method: str, kwargs: dict[str, Any]) -> Any:
    """Run one request against the connector."""
    if method == "spawn_probe":
        return await _spawn_probe(connector, **kwargs)
    if method in PROXY_METHODS:
        return await getattr(connector, method)(**kwargs)
    raise ValueError(
        f"connector host does not serve {method!r}; it serves "
        f"{', '.join((*PROXY_METHODS, 'spawn_probe'))}"
    )


async def _dispatch(connector: Any, frame: Any, writer: _FrameWriter) -> None:
    """Serve one request and write exactly one reply for it."""
    try:
        payload = frames.encode_result(
            frame.request_id, await _invoke(connector, frame.method, frame.kwargs)
        )
    except Exception as exc:
        logger.warning("connector host: %s failed: %r", frame.method, exc)
        payload = _error_frame(frame.request_id, exc)
    await writer.send(payload)


async def _serve(connector: Any, stream: _FrameStream, writer: _FrameWriter) -> int:
    """Dispatch requests until the pipe closes or a disconnect is served."""
    inflight: set[asyncio.Task[None]] = set()
    while True:
        frame = await stream.next()
        if frame is None:
            logger.warning("connector host: stdin closed; exiting")
            break
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


async def _run(channel_fd: int) -> int:
    """Read the init frame, report, then serve until told otherwise."""
    stream = await _stdin_frames()
    writer = _FrameWriter(channel_fd)

    init = await stream.next()
    if init is None:
        logger.warning("connector host: stdin closed before the init frame; exiting")
        return EXIT_OK
    if not isinstance(init, frames.RequestFrame) or init.method != INIT_METHOD:
        request_id = getattr(init, "request_id", "init")
        await writer.send(
            _error_frame(
                request_id,
                ConnectionError(
                    f"connector host expected a {INIT_METHOD!r} request as its first frame, "
                    f"got {getattr(init, 'method', type(init).__name__)!r}"
                ),
            )
        )
        return EXIT_INIT_FAILED

    try:
        connector, report = await _build_connector(init.kwargs)
    except Exception as exc:
        logger.warning("connector host: connect failed: %r", exc)
        await writer.send(_error_frame(init.request_id, exc))
        return EXIT_INIT_FAILED

    await writer.send(frames.encode_result(init.request_id, report))
    return await _serve(connector, stream, writer)


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m osprey_connectors.ipc.host``."""
    del argv  # the launch contract carries no arguments; config arrives on the wire
    removed = scrub_epics_env()
    channel_fd = _claim_frame_channel()
    # Handlers on stderr, explicitly: a record that reached stdout would sit in
    # the middle of a frame. Nothing here calls the framework's
    # configure_logging(), which is an entry-point concern.
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format="%(name)s: %(message)s")
    if removed:
        logger.warning("connector host: scrubbed inherited %s", ", ".join(sorted(removed)))

    _start_watchdog()
    try:
        return asyncio.run(_run(channel_fd))
    except KeyboardInterrupt:  # pragma: no cover - parent-initiated
        return EXIT_OK


if __name__ == "__main__":
    _exit_now(main(sys.argv[1:]))
