"""Runtime utilities for generated Python code.

This module provides control-system-agnostic utilities for reading and writing
to control systems. It's designed to be used in generated Python code and
automatically configures itself from the global config.

Usage in generated code:
    >>> from osprey.runtime import write_channel, read_channel
    >>>
    >>> # Write to control system (synchronous, like EPICS caput)
    >>> write_channel("BEAM:CURRENT", 500.0)
    >>>
    >>> # Read from control system (synchronous, like EPICS caget)
    >>> value = read_channel("BEAM:CURRENT")
    >>> print(f"Current: {value}")

Configuration:
    The runtime uses the control system configuration from the global config.yml.

Limits Validation:
    Write operations are validated at two levels:
    1. Runtime-level: An injected LimitsValidator, set by the executor's
       execution wrapper and by nothing else, checks values before the connector
       is even called. This provides a safety net in subprocess execution where
       the connector may not be fully configured.
    2. Connector-level: The control system connector validates writes against its
       own configured limits database as a secondary check.

Write Outcomes:
    Writes go through the connector's ``write_channel_checked``, so a write that
    was refused, that failed, or whose readback did not confirm the setpoint
    raises instead of returning. Generated code never has to inspect a result
    object to find out whether the hardware took the value.

Control Target:
    The execution sandbox is launched with a *target stamp* — ``live`` or ``va``
    — in its environment, written by
    :mod:`osprey.mcp_server.python_executor.executor`. That stamp, not the
    config's own ``control_system.type``, decides which connector this runtime
    builds: the type is resolved through
    :func:`osprey_connectors.types.resolve_target`, so the factory reads
    ``control_system.connector.<that type>`` and ``connect()`` derives that
    target's gateways here rather than anywhere upstream. An unstamped process
    resolves its connector exactly as it always did.

    Writes are additionally *pinned* to the generation the stamp was taken at.
    A connector already built in this process is never re-pointed at another
    machine — holders do not reconnect across a switch — so once the session's
    target or generation moves, every further write raises
    :class:`ControlTargetChangedError` and the operator re-runs the code in a
    fresh sandbox. Reads are not pinned: reading the machine the run started on
    is the run being consistent with itself, not a hazard.

    A notebook kernel is the one process that outlives a switch, so it is the
    one exception to "one process, one stamp": its ``pre_run_cell`` rewrites the
    stamp from the deployment's record before every cell. This module follows
    that — the connector is rebuilt when the stamp moves, and a cell the kernel
    could route nowhere at all is refused by :func:`_get_connector` on its first
    control-system call.
"""

import asyncio
import atexit
import json
import os
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from osprey.connectors.control_system.limits_validator import LimitsValidator
    from osprey_connectors.control_context import ControlContext

logger = get_logger("runtime")

__all__ = [
    "write_channel",
    "read_channel",
    "write_channels",
    "cleanup_runtime",
    "ControlTargetChangedError",
    "SwitchInProgressError",
]

#: The target stamp this process was launched with. The same two literals are
#: spelled in :mod:`osprey.mcp_server.python_executor.executor`, which is their
#: only writer; ``tests/runtime/test_executor_target_stamp.py`` pins them equal.
ENV_CONTROL_TARGET = "OSPREY_CONTROL_TARGET"
ENV_CONTROL_TARGET_GENERATION = "OSPREY_CONTROL_TARGET_GENERATION"

#: Why this process may reach no control system at all, and whether a notebook
#: cell is open. Both are written by :mod:`osprey.jupyter_kernel`, their only
#: writer, and re-spelled here for the reason the two stamps above are: a
#: runtime that imported the kernel launcher would drag the notebook stack into
#: every sandbox. ``tests/runtime/test_jupyter_kernel.py`` pins them equal.
ENV_CONTROL_TARGET_REFUSAL = "OSPREY_CONTROL_TARGET_REFUSAL"
ENV_IN_CELL = "OSPREY_NOTEBOOK_IN_CELL"

#: Which kind of client the markers written here describe. Restated rather than
#: imported, under the same drift-guard rule the executor's own
#: ``INFLIGHT_SURFACE`` follows.
INFLIGHT_SURFACE = "notebook_kernel"

#: How a kernel spells itself as an audit session; a marker's ``kernel_id`` is
#: what follows the prefix.
KERNEL_SESSION_PREFIX = "kernel:"


class ControlTargetChangedError(RuntimeError):
    """This call reached no control system, because the target moved under it.

    Never a failed operation — nothing was attempted. It is raised by
    :func:`_assert_target_pin`, when the deployment's record has moved past the
    target and generation this process was stamped with, and, as
    :class:`SwitchInProgressError`, by :func:`_get_connector`, when the
    deployment is between two targets and there is nothing to route to.

    A sandbox cannot retry it: it holds its stamp for its whole life and the
    connector it built stays connected to that target's gateways. A notebook
    cell can — the kernel re-routes itself from the record before every cell,
    so re-running the cell is what picks the move up.
    """


class SwitchInProgressError(ControlTargetChangedError):
    """A control-target switch is in flight, so this call was not routed.

    Raised on the first control-system call of a notebook cell that
    ``pre_run_cell`` could route nowhere: while a controls server is applying a
    switch it is between two targets, and a connector built now could reach
    either one.

    Args:
        refusal: The value the kernel left behind —
            ``switch_in_progress:<pid>[,<pid>…]``. It opens the message, so the
            operator and any log reader meet the same token every other surface
            refuses with.
    """

    def __init__(self, refusal: str) -> None:
        self.refusal = refusal
        _, _, pids = refusal.partition(":")
        named = pids or "an unnamed server"
        super().__init__(
            f"{refusal}. A control-target switch is in progress on pid {named}; "
            "re-run the cell when the chip settles."
        )


# Module-level state
_runtime_connector: Any | None = None
#: The ``(target, generation)`` the connector above was built for. A kernel
#: re-stamps itself every cell, so this is what tells a rebuild from a reuse.
_connector_stamp: tuple[str | None, int | None] | None = None
_connector_lock = asyncio.Lock()
#: The in-flight marker this cell holds, or ``None``. Removed by
#: ``post_run_cell``, which is why the claim below re-checks the file rather
#: than trusting this name across cells.
_cell_marker: "Path | None" = None
#: The executor sandbox's own limits validator, injected by its execution
#: wrapper before user code runs. Nothing else sets it — a notebook kernel runs
#: no wrapper — so every other process validates at the connector alone.
_limits_validator: "LimitsValidator | None" = None


def _stamped_target() -> str | None:
    """The control target this sandbox was launched for, or ``None``.

    The environment is the only source consulted. The state file is not read for
    routing even though it is readable from here: the stamp is what the host
    decided this run is for, and a process that re-derived its own target could
    route somewhere the execute call never reported.
    """
    target = os.environ.get(ENV_CONTROL_TARGET, "").strip()
    return target or None


def _stamped_generation() -> int | None:
    """The generation the stamp was taken at, or ``None`` if it is unusable."""
    raw = os.environ.get(ENV_CONTROL_TARGET_GENERATION, "").strip()
    try:
        return int(raw)
    except ValueError:
        return None


def _target_connector_config() -> dict[str, Any] | None:
    """The ``control_system`` config this run's target selects, or ``None``.

    ``None`` means "no stamp": the caller passes it straight to the factory,
    which loads the section itself — byte-identical to the unstamped behaviour.

    With a stamp, the section is loaded here and its ``type`` is replaced by the
    type the target resolves to. That single substitution is what re-points the
    factory, because the resolved type is also the key of the block it reads
    settings from (``control_system.connector.<type>``). The rest of the section
    is passed through unchanged, so every target's gateways stay where the
    deployment configured them.

    Raises:
        ValueError: From :func:`~osprey_connectors.types.resolve_target` when
            the deployment has not named the machine this target means. It is
            deliberately not caught: falling back to the config's own type would
            answer "which machine am I on" with a different machine.
    """
    target = _stamped_target()
    if target is None:
        return None

    from osprey_connectors.config import get_config_value
    from osprey_connectors.types import resolve_target

    section = get_config_value("control_system", {})
    if not isinstance(section, dict):
        section = {}
    config = dict(section)
    config["type"] = resolve_target(section, target)
    return config


def _assert_not_refused() -> None:
    """Refuse the call outright when this cell was routed nowhere.

    ``IPython`` swallows an exception raised by a ``pre_run_cell`` callback, so
    the kernel's gate cannot refuse a cell where it decides to: it leaves the
    reason in the environment instead, and the cell's first control-system call
    raises it here — before the stamp is read, because a cell that was refused
    carries no stamp to read.

    Raises:
        SwitchInProgressError: If the kernel left a refusal for this cell.
    """
    refusal = os.environ.get(ENV_CONTROL_TARGET_REFUSAL, "").strip()
    if refusal:
        raise SwitchInProgressError(refusal)


def _claim_cell() -> None:
    """Hold this cell against a target switch, from its first call onwards.

    A switch waits for every live in-flight marker, and this is the notebook's.
    It is written lazily because a cell that never touches the control system
    has nothing for a switch to wait on, and only while the kernel says a cell
    is open because a thread reaching the control system between cells is not
    something an operator could be asked to interrupt — its writes are held by
    :func:`_assert_target_pin` instead.

    ``post_run_cell`` removes the marker by this kernel's id, so a kernel that
    could not name itself writes none: a marker nothing can remove would refuse
    every later switch on this deployment. For the same reason the claim is
    re-checked against the file rather than against :data:`_cell_marker` alone,
    which still names the marker the previous cell's end deleted.

    Never raises. The marker is advisory — what stops a superseded run writing
    is the generation pin — so a state directory this process cannot write to
    costs a switch its wait, not the cell its run.
    """
    global _cell_marker

    if ENV_IN_CELL not in os.environ:
        return
    try:
        if _cell_marker is not None and _cell_marker.exists():
            return

        from osprey.audit import posture
        from osprey.mcp_server.control_system import target_state
        from osprey_connectors import posture_store

        session = posture.posture_session() or ""
        kernel_id = (
            session[len(KERNEL_SESSION_PREFIX) :]
            if session.startswith(KERNEL_SESSION_PREFIX)
            else ""
        )
        if not kernel_id:
            return

        directory = target_state.state_dir()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (
            f"{target_state.INFLIGHT_FILE_PREFIX}{os.getpid()}_{uuid.uuid4().hex}"
            f"{target_state.INFLIGHT_FILE_SUFFIX}"
        )
        marker = {
            "pid": os.getpid(),
            "session": session,
            "surface": INFLIGHT_SURFACE,
            "kernel_id": kernel_id,
            "target": _stamped_target(),
            "launch_posture": os.environ.get(posture_store.LAUNCH_POSTURE_ENV_VAR),
            "started_at": datetime.now().astimezone().isoformat(),
        }
        # Temp file beside it and a rename, as the executor's marker is written:
        # a reader must never meet half of one.
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(marker), encoding="utf-8")
        os.replace(tmp, path)
        _cell_marker = path
    except Exception:  # noqa: BLE001 - an unclaimed cell is not worth the cell
        logger.warning("Could not claim this cell against a target switch", exc_info=True)


async def _get_connector():
    """Get or create the connector this process's stamp names.

    Internal function called by the runtime utilities.

    When this process carries a target stamp, the connector is built for that
    target instead of for the config's baseline type, and the same target is
    stamped onto the instance so the reference monitor inside it reads the
    session posture for the target this run was launched against. An unstamped
    process names no target, exactly as before.

    A sandbox reaches the build once and reuses it for its whole life. A
    notebook kernel does not: it is re-stamped from the deployment's record
    before every cell, so a stamp that no longer matches the connector in hand
    means the ground moved between cells, and the connector is disconnected and
    rebuilt rather than re-pointed. The disconnect goes through
    :func:`_disconnect_locked` because ``_connector_lock`` is already held here
    and is not reentrant.

    Returns:
        ControlSystemConnector instance

    Raises:
        SwitchInProgressError: If the kernel routed this cell nowhere because a
            control-target switch is in flight. Nothing is built, and the
            connector this process already holds is left alone.
    """
    global _runtime_connector, _connector_stamp

    _assert_not_refused()
    _claim_cell()

    async with _connector_lock:
        stamp = (_stamped_target(), _stamped_generation())
        if _runtime_connector is not None and stamp != _connector_stamp:
            logger.debug("Control target moved to %s; rebuilding the connector", stamp)
            await _disconnect_locked()

        if _runtime_connector is None:
            from osprey.connectors.factory import ConnectorFactory

            config = _target_connector_config()
            if config is None:
                logger.debug("Creating connector from global config")
            else:
                logger.debug(
                    "Creating connector for stamped target %s (type %s)",
                    stamp[0],
                    config["type"],
                )
            _runtime_connector = await ConnectorFactory.create_control_system_connector(
                config=config, control_target=stamp[0]
            )
            _connector_stamp = stamp

    return _runtime_connector


def _current_target_record() -> "ControlContext | None":
    """What the deployment's control context says *now*.

    One record per deployment instance, written by its owner: there is no
    identity for this process to carry and no directory for it to search. The
    file the stamp was taken from is the file the pin re-reads.

    ``None`` is returned when there is no agent-data root to resolve, when the
    record is missing, and when it is unreadable or corrupt. All of them mean
    the current generation is unknown, and the pin below treats unknown as a
    refusal: a write is the operation that cannot be taken back.
    """
    try:
        from osprey_connectors import control_context

        return control_context.read_record()
    except Exception:
        logger.debug("Control context unreadable from sandbox", exc_info=True)
        return None


def _assert_target_pin() -> None:
    """Refuse the write if the deployment is no longer on the stamped target.

    An unstamped process is not pinned at all — it never claimed a target, so
    there is nothing for it to have drifted from.

    The comparison is ``(target, generation)`` against the record and nothing
    else. Whether the fleet has finished converging on that generation is a
    question about admitting a NEW run, answered where runs are admitted (the
    executor's stamp, the kernel's cell gate); a process already holding a
    connector is judged only on whether the ground moved under it.

    This is a snapshot of the record taken just before the write, so a
    switch that lands in the window between the check and the write itself is
    not caught. That window is not a routing hole: this process's connector was
    bound to its target's gateways at ``connect()`` time and does not follow a
    switch, so the write still goes where the stamp says. What the pin bounds is
    how long a superseded process keeps writing there — the switch lifecycle's
    drain, not this check, is what makes that window closed rather than merely
    small.

    Raises:
        ControlTargetChangedError: If the stamped generation or target does not
            match what the deployment's record currently says, or if that record
            cannot be read at all.
    """
    target = _stamped_target()
    if target is None:
        return

    stamped_generation = _stamped_generation()
    record = _current_target_record()

    if (
        stamped_generation is not None
        and record is not None
        and record.target == target
        and record.generation == stamped_generation
    ):
        return

    stamped_description = (
        f"{target!r} generation {'unknown' if stamped_generation is None else stamped_generation}"
    )
    if record is not None:
        current_description = f"{record.target!r} generation {record.generation}"
    else:
        current_description = "unknown (the deployment's control context is missing or unreadable)"

    raise ControlTargetChangedError(
        "Refusing to write: this execution was started against control target "
        f"{stamped_description}, but the deployment is now on {current_description}. "
        "Connectors held by a running process never reconnect across a target "
        "change, so this process can only reach the target it started on. "
        "Re-run the code with execute() to get a sandbox on the current target."
    )


# ========================================================
# Internal async implementations
# ========================================================


async def _write_channel_async(channel_address: str, value: Any, **kwargs) -> None:
    """Internal async implementation for writing to a channel.

    The connector handles limits validation when available.
    The injected _limits_validator provides a safety net for subprocess execution
    where the connector's config-based validator may not be initialized.

    Returns normally only for a write that verifiably landed; every other outcome
    raises. See :func:`write_channel`.
    """
    # Pinned first, before the value is even validated: a write aimed at a
    # machine this process can no longer reach must not be described by an error
    # about its value.
    _assert_target_pin()

    # Safety net: validate against injected limits validator (set by execution wrapper)
    # This catches violations even when the connector's own validator isn't configured
    if _limits_validator is not None:
        _limits_validator.validate(channel_address, value)  # Raises ChannelLimitsViolationError

    connector = await _get_connector()
    # write_channel_checked is the reference monitor's denial contract: it raises
    # on a refusal, on a failed write, AND on a write whose confirming re-read
    # did not hold the setpoint. Calling write_channel directly would let an
    # unconfirmed write return silently and get logged as "Wrote ...".
    result = await connector.write_channel_checked(channel_address, value, **kwargs)

    # Reaching here means the outcome is confirmed, or confirmation was not
    # requested (unrequested) — every other outcome already raised above.
    if result.observed_value is not None:
        logger.debug(
            f"Wrote {channel_address} = {value} [{result.outcome}, "
            f"observed {result.observed_value}]"
        )
    else:
        logger.debug(f"Wrote {channel_address} = {value} [{result.outcome}]")


async def _read_channel_async(channel_address: str, **kwargs) -> Any:
    """Internal async implementation for reading from a channel."""
    connector = await _get_connector()
    channel_value = await connector.read_channel(channel_address, **kwargs)
    return channel_value.value


async def _write_channels_async(channel_values: dict[str, Any], **kwargs) -> None:
    """Internal async implementation for writing multiple channels."""
    if len(channel_values) == 1:
        [(address, value)] = channel_values.items()
        await _write_channel_async(address, value, **kwargs)
    else:
        # Same pin as the single-channel path, which the branch above reaches
        # through _write_channel_async.
        _assert_target_pin()

        # Validate all values against injected limits validator first
        if _limits_validator is not None:
            for channel_address, value in channel_values.items():
                _limits_validator.validate(channel_address, value)

        from osprey.connectors.control_system import raise_for_write_result

        connector = await _get_connector()
        results = await connector.write_multiple_channels(list(channel_values.items()), **kwargs)
        # Same denial contract as the single-channel path: a refusal or an
        # unconfirmed write must raise rather than return.
        for result in results:
            raise_for_write_result(result)


def _run_async(coro) -> Any:
    """Run async coroutine synchronously.

    Handles both subprocess and Jupyter notebook contexts correctly.

    Only the loop probe sits inside the ``try``: the ``RuntimeError`` it
    raises is the one signal that there is no running loop. Whatever the
    coroutine itself raises — including :class:`ControlTargetChangedError`,
    which is a ``RuntimeError`` too — must propagate unchanged from either
    branch, never be mistaken for that signal and retried.
    """
    try:
        # Try to get running loop (e.g., in Jupyter with nest_asyncio)
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop - we're in a subprocess, use asyncio.run()
        return asyncio.run(coro)

    # If we have a running loop, we need to run in a new thread
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(asyncio.run, coro)
        return future.result()


# ========================================================
# Public synchronous API (like EPICS caput/caget)
# ========================================================


def write_channel(channel_address: str, value: Any, **kwargs) -> None:
    """Write value to control system channel.

    Works with any configured control system (EPICS, Mock, etc.).

    Synchronous function - no 'await' needed. Works like EPICS caput().

    Args:
        channel_address: Channel/PV name to write to
        value: Value to write (will be coerced to appropriate type)
        **kwargs: Additional arguments passed to connector
                  - timeout: Operation timeout in seconds
                  - confirm: Whether to re-read the channel and compare it
                    against the value sent. Omit (or pass None) to let the
                    channel resolve its own confirm default; pass True or
                    False to override it for this write.

    Raises:
        ChannelLimitsViolationError: If value violates channel safety limits
        ChannelWriteBlockedError: If the write was refused and no value was
            written — by policy, limits, or validation (never attempted), or by
            the control system itself (CONTROL_SYSTEM_REFUSED)
        ChannelWriteFailedError: If the write was attempted but did not come
            back confirmed — the control system did not take it (FAILED), a
            confirming re-read holds a different value (MISMATCH), or the
            write was not acknowledged in time or the confirming re-read itself
            failed (UNCONFIRMED)
        ControlTargetChangedError: If the session switched control target after
            this execution started; nothing was written
        TimeoutError: If operation times out

    Examples:
        >>> from osprey.runtime import write_channel
        >>> write_channel("BEAM:CURRENT", 500.0)
        >>> write_channel("MAGNET:FIELD", 2.5, timeout=10.0)
    """
    _run_async(_write_channel_async(channel_address, value, **kwargs))


def read_channel(channel_address: str, **kwargs) -> Any:
    """Read value from control system channel.

    Works with any configured control system (EPICS, Mock, etc.).

    Synchronous function - no 'await' needed. Works like EPICS caget().

    Args:
        channel_address: Channel/PV name to read from
        **kwargs: Additional arguments passed to connector
                  - timeout: Operation timeout in seconds

    Returns:
        Current value of the channel. An enum-typed channel (EPICS mbbi/bi/bo
        and equivalents) reads as its integer state index; the matching state
        names ride on the reading's metadata as ``enum_label`` /
        ``enum_labels``, which the channel_read tool reports and which this
        value-only helper does not return.

    Raises:
        RuntimeError: If read operation fails
        TimeoutError: If operation times out

    Examples:
        >>> from osprey.runtime import read_channel
        >>> current = read_channel("BEAM:CURRENT")
        >>> print(f"Current: {current}")
    """
    return _run_async(_read_channel_async(channel_address, **kwargs))


def write_channels(channel_values: dict[str, Any], **kwargs) -> None:
    """Write multiple channels.

    Convenience function for writing multiple channels. Writes are performed
    sequentially but all use the same connector.

    Synchronous function - no 'await' needed.

    Args:
        channel_values: Dictionary mapping channel names to values
        **kwargs: Additional arguments passed to each write (timeout, confirm
                  -- see write_channel). A batch carries one confirm for
                  every channel in it; omit it (or pass None) to let each
                  channel resolve its own confirm default instead.

    Raises:
        ChannelLimitsViolationError: If a value violates channel safety limits
        ChannelWriteBlockedError: If any write was refused and no value was
            written — by policy, limits, or validation (never attempted), or by
            the control system itself (CONTROL_SYSTEM_REFUSED)
        ChannelWriteFailedError: If any write did not come back confirmed —
            FAILED, MISMATCH, or UNCONFIRMED. Writes before the failing one
            have already been applied.
        ControlTargetChangedError: If the session switched control target after
            this execution started; nothing was written

    Examples:
        >>> from osprey.runtime import write_channels
        >>> write_channels({
        ...     "MAGNET:H01": 5.0,
        ...     "MAGNET:H02": 5.2,
        ...     "MAGNET:H03": 4.8
        ... })
    """
    _run_async(_write_channels_async(channel_values, **kwargs))


async def _disconnect_locked() -> None:
    """Disconnect and drop the runtime connector. The lock must be held already.

    Split out of :func:`cleanup_runtime` for :func:`_get_connector`, which
    disconnects and rebuilds inside one critical section: ``asyncio.Lock`` is
    not reentrant, so a rebuild that called ``cleanup_runtime`` would wait on a
    lock it is holding itself.
    """
    global _runtime_connector, _connector_stamp

    if _runtime_connector is None:
        return
    try:
        # Check if connector has cleanup method
        if hasattr(_runtime_connector, "disconnect"):
            await _runtime_connector.disconnect()
        elif hasattr(_runtime_connector, "close"):
            await _runtime_connector.close()
        logger.debug("Runtime connector cleaned up")
    except Exception as e:
        logger.warning(f"Error during connector cleanup: {e}")
    finally:
        _runtime_connector = None
        _connector_stamp = None


async def cleanup_runtime() -> None:
    """Cleanup runtime resources.

    Disconnects connector and releases resources. Called automatically
    at end of execution, but can be called manually if needed.

    This is particularly useful for long-running notebook sessions to
    ensure connections don't become stale.
    """
    async with _connector_lock:
        await _disconnect_locked()


# Register cleanup on module exit
def _cleanup_on_exit() -> None:
    """Synchronous cleanup for atexit handler."""
    if _runtime_connector is not None:
        try:
            asyncio.run(cleanup_runtime())
        except Exception:
            pass  # Best effort cleanup


atexit.register(_cleanup_on_exit)
