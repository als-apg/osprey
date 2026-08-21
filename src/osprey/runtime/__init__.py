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
"""

import asyncio
import atexit
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
]

# Module-level state
_runtime_connector: Any | None = None
_connector_lock = asyncio.Lock()
_limits_validator: "LimitsValidator | None" = (
    None  # Injected by execution wrapper for subprocess safety
)


async def _get_connector():
    """Get or create connector using global config.

    Internal function called by the runtime utilities.
    Creates the connector once and reuses it for all operations.

    Returns:
        ControlSystemConnector instance
    """
    global _runtime_connector

    async with _connector_lock:
        if _runtime_connector is None:
            from osprey.connectors.factory import ConnectorFactory

            logger.debug("Creating connector from global config")
            _runtime_connector = await ConnectorFactory.create_control_system_connector(config=None)

    return _runtime_connector


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
    # Safety net: validate against injected limits validator (set by execution wrapper)
    # This catches violations even when the connector's own validator isn't configured
    if _limits_validator is not None:
        _limits_validator.validate(channel_address, value)  # Raises ChannelLimitsViolationError

    connector = await _get_connector()
    # write_channel_checked is the reference monitor's denial contract: it raises
    # on a refusal, on a failed write, AND on a write whose readback did not
    # confirm the setpoint. Calling write_channel directly would let an
    # unverified write return silently and get logged as "Wrote ...".
    result = await connector.write_channel_checked(channel_address, value, **kwargs)

    # Reaching here means verified, or no verification was requested at all.
    verification = result.verification
    if verification is not None and verification.verified:
        logger.debug(f"Wrote {channel_address} = {value} [{verification.level} verified]")
    else:
        logger.debug(f"Wrote {channel_address} = {value} [no verification requested]")


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
        # Validate all values against injected limits validator first
        if _limits_validator is not None:
            for channel_address, value in channel_values.items():
                _limits_validator.validate(channel_address, value)

        from osprey.connectors.control_system import raise_for_write_result

        connector = await _get_connector()
        results = await connector.write_multiple_channels(list(channel_values.items()), **kwargs)
        # Same denial contract as the single-channel path: a refusal or an
        # unverified write must raise rather than return.
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
                  - verification_level: 'none', 'callback', or 'readback'.
                    Omit to let each channel resolve its own level (see the
                    connector's four-layer resolution)
                  - tolerance: Tolerance for readback verification

    Raises:
        ChannelLimitsViolationError: If value violates channel safety limits
        ChannelWriteBlockedError: If the write was refused (writes disabled,
            limits, or validation) and never attempted
        ChannelWriteFailedError: If the write was attempted but failed, or came
            back unverified because the readback did not match the setpoint
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
        Current value of the channel

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
        **kwargs: Additional arguments passed to each write (timeout,
                  verification_level, tolerance -- see write_channel).
                  Omit verification_level to let each channel resolve its
                  own level (see the connector's four-layer resolution)

    Raises:
        ChannelLimitsViolationError: If a value violates channel safety limits
        ChannelWriteBlockedError: If any write was refused and never attempted
        ChannelWriteFailedError: If any write failed or came back unverified.
            Writes before the failing one have already been applied.

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
