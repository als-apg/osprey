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
    1. Runtime-level: An injected LimitsValidator (set by the execution wrapper)
       checks values before the connector is even called. This provides a safety
       net in subprocess execution where the connector may not be fully configured.
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
"""

import asyncio
import atexit
import os
from typing import TYPE_CHECKING, Any

from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from osprey.connectors.control_system.limits_validator import LimitsValidator

logger = get_logger("runtime")

__all__ = [
    "write_channel",
    "read_channel",
    "write_channels",
    "cleanup_runtime",
    "ControlTargetChangedError",
]

#: The target stamp this process was launched with. The same three literals are
#: spelled in :mod:`osprey.mcp_server.python_executor.executor`, which is their
#: only writer; ``tests/runtime/test_executor_target_stamp.py`` pins them equal.
ENV_CONTROL_TARGET = "OSPREY_CONTROL_TARGET"
ENV_CONTROL_TARGET_GENERATION = "OSPREY_CONTROL_TARGET_GENERATION"
#: PID of the controls server whose record the stamp was taken from — the
#: identity of the file the write pin re-reads. Searching the state directory
#: for it instead would be a guess: two sessions can share a checkout, so
#: "the only live record" is neither necessarily ours nor necessarily unique.
ENV_CONTROL_TARGET_STATE_PID = "OSPREY_CONTROL_TARGET_STATE_PID"


class ControlTargetChangedError(RuntimeError):
    """A write was refused because the session's control target moved under it.

    Raised by the write path only. It is not a failed write — nothing was
    attempted — and it is not retryable in this process, because the connector
    this process holds is still connected to the previous target's gateways.
    """


# Module-level state
_runtime_connector: Any | None = None
_connector_lock = asyncio.Lock()
_limits_validator: "LimitsValidator | None" = (
    None  # Injected by execution wrapper for subprocess safety
)


def _stamped_target() -> str | None:
    """The session target this sandbox was launched for, or ``None``.

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


async def _get_connector():
    """Get or create connector using global config.

    Internal function called by the runtime utilities.
    Creates the connector once and reuses it for all operations.

    When this sandbox carries a target stamp, the connector is built for that
    target instead of for the config's baseline type, and the same target is
    stamped onto the instance so the reference monitor inside it reads the
    session posture for the target this run was launched against. An unstamped
    sandbox names no target, exactly as before. The connector is created
    once either way: a process holds one connector for its whole life and never
    re-points it, which is why the write path pins rather than reconnects.

    Returns:
        ControlSystemConnector instance
    """
    global _runtime_connector

    async with _connector_lock:
        if _runtime_connector is None:
            from osprey.connectors.factory import ConnectorFactory

            config = _target_connector_config()
            if config is None:
                logger.debug("Creating connector from global config")
            else:
                logger.debug(
                    "Creating connector for stamped target %s (type %s)",
                    _stamped_target(),
                    config["type"],
                )
            _runtime_connector = await ConnectorFactory.create_control_system_connector(
                config=config, control_target=_stamped_target()
            )

    return _runtime_connector


def _stamped_state_pid() -> int | None:
    """PID of the controls server the stamp was taken from, or ``None``."""
    raw = os.environ.get(ENV_CONTROL_TARGET_STATE_PID, "").strip()
    try:
        return int(raw)
    except ValueError:
        return None


def _current_target_record() -> dict[str, Any] | None:
    """What the controls server that stamped this run publishes *now*.

    Exactly one file is consulted: the one belonging to the PID carried in the
    stamp. This process cannot re-derive that identity — its parent is the
    python-executor server, not the Claude Code process that owns the state
    file — and it must not search for it either. Two sessions can share a
    checkout, so "the only live record in the directory" would refuse every
    write in the two-session case and, worse, could match a stranger's record
    in the case where this session's own server has died.

    ``None`` is returned when there is no stamped PID, when that process is
    gone, and when its file is missing, unreadable, or corrupt. All of them mean
    the current generation is unknown, and the pin below treats unknown as a
    refusal: a write is the operation that cannot be taken back.
    """
    pid = _stamped_state_pid()
    if pid is None:
        return None
    try:
        from osprey.mcp_server.control_system import target_state

        if not target_state.is_process_alive(pid):
            return None
        return target_state.read_file(target_state.state_file_path(pid))
    except Exception:
        logger.debug("Target state unreadable from sandbox", exc_info=True)
        return None


def _assert_target_pin() -> None:
    """Refuse the write if the session is no longer on the stamped target.

    An unstamped process is not pinned at all — it never claimed a target, so
    there is nothing for it to have drifted from.

    This is a snapshot of the state file taken just before the write, so a
    switch that lands in the window between the check and the write itself is
    not caught. That window is not a routing hole: this process's connector was
    bound to its target's gateways at ``connect()`` time and does not follow a
    switch, so the write still goes where the stamp says. What the pin bounds is
    how long a superseded process keeps writing there — the switch lifecycle's
    drain, not this check, is what makes that window closed rather than merely
    small.

    Raises:
        ControlTargetChangedError: If the stamped generation or target does not
            match what the controls server currently publishes, or if the
            current state cannot be read at all.
    """
    target = _stamped_target()
    if target is None:
        return

    stamped_generation = _stamped_generation()
    record = _current_target_record()
    current_target = record.get("target") if record is not None else None
    current_generation = record.get("generation") if record is not None else None

    if (
        stamped_generation is not None
        and current_target == target
        and current_generation == stamped_generation
    ):
        return

    stamped_description = (
        f"{target!r} generation {'unknown' if stamped_generation is None else stamped_generation}"
    )
    if record is not None:
        current_description = f"{current_target!r} generation {current_generation}"
    elif (state_pid := _stamped_state_pid()) is None:
        current_description = (
            "unknown (this execution carries no state-file identity to check against)"
        )
    else:
        current_description = (
            f"unknown (the controls server this execution was stamped from, pid "
            f"{state_pid}, is gone or its state file is unreadable)"
        )

    raise ControlTargetChangedError(
        "Refusing to write: this execution was started against control target "
        f"{stamped_description}, but the session is now on {current_description}. "
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
    """
    try:
        # Try to get running loop (e.g., in Jupyter with nest_asyncio)
        asyncio.get_running_loop()
        # If we have a running loop, we need to run in a new thread
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        # No running loop - we're in a subprocess, use asyncio.run()
        return asyncio.run(coro)


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
            confirming re-read itself failed (UNCONFIRMED)
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


async def cleanup_runtime() -> None:
    """Cleanup runtime resources.

    Disconnects connector and releases resources. Called automatically
    at end of execution, but can be called manually if needed.

    This is particularly useful for long-running notebook sessions to
    ensure connections don't become stale.
    """
    global _runtime_connector

    async with _connector_lock:
        if _runtime_connector is not None:
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


# Register cleanup on module exit
def _cleanup_on_exit() -> None:
    """Synchronous cleanup for atexit handler."""
    if _runtime_connector is not None:
        try:
            asyncio.run(cleanup_runtime())
        except Exception:
            pass  # Best effort cleanup


atexit.register(_cleanup_on_exit)
