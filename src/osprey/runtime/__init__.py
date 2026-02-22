"""Runtime utilities for generated Python code.

This module provides control-system-agnostic utilities for reading and writing
to control systems. It's designed to be used in generated Python code and
automatically configures itself from execution context.

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
    The runtime automatically uses the control system configuration that was
    active when the code was generated (preserved in context.json).

    This ensures notebooks are reproducible - re-running a notebook a week
    later will use the same control system configuration it was created with.

Limits Validation:
    Write operations are validated at two levels:
    1. Runtime-level: An injected LimitsValidator (set by the execution wrapper)
       checks values before the connector is even called. This provides a safety
       net in subprocess execution where the connector may not be fully configured.
    2. Connector-level: The control system connector validates writes against its
       own configured limits database as a secondary check.
"""

import asyncio
import atexit
from typing import TYPE_CHECKING, Any

from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from osprey.services.python_executor.execution.limits_validator import LimitsValidator

logger = get_logger("runtime")

__all__ = [
    "configure_from_context",
    "write_channel",
    "read_channel",
    "write_channels",
    "cleanup_runtime",
]

# Module-level state
_runtime_connector: Any | None = None
_runtime_config: dict | None = None
_connector_lock = asyncio.Lock()
_limits_validator: "LimitsValidator | None" = (
    None  # Injected by execution wrapper for subprocess safety
)


def configure_from_context(context) -> None:
    """Configure runtime environment from execution context.

    Called automatically by execution wrapper after loading context.
    Extracts control system config snapshot and prepares connector.

    Fallback chain:
    1. Try context snapshot config
    2. If missing/invalid, fall back to global config with warning
    3. If both fail, raise clear error

    Args:
        context: ContextManager instance from load_context()

    Note:
        This function is called automatically by the execution wrapper.
        You don't need to call it manually in generated code.
    """
    global _runtime_config

    # Try context snapshot first
    if context and hasattr(context, "_data"):
        try:
            # Access the raw data dictionary
            raw_data = context._data if hasattr(context, "_data") else {}
            exec_config = raw_data.get("_execution_config")

            if exec_config and "control_system" in exec_config:
                _runtime_config = exec_config["control_system"]

                # Validate required fields
                if "type" not in _runtime_config:
                    raise ValueError("Context config missing 'type' field")

                logger.debug(f"Runtime configured from context snapshot: {_runtime_config['type']}")
                return
        except Exception as e:
            logger.debug(f"Failed to load context config: {e}")
            logger.debug("Falling back to global configuration")

    # Fallback to global config
    try:
        from osprey.utils.config import get_config_value

        _runtime_config = get_config_value("control_system", {})
        if _runtime_config and "type" in _runtime_config:
            logger.debug(f"Runtime configured from global config: {_runtime_config['type']}")
        else:
            raise RuntimeError("No control system configuration available")
    except Exception as e:
        raise RuntimeError(
            "Failed to configure runtime: No valid config in context or global settings. "
            f"Error: {e}"
        ) from e


async def _get_connector():
    """Get or create connector using context config or global config.

    Internal function called by the runtime utilities.
    Creates the connector once and reuses it for all operations.

    Returns:
        ControlSystemConnector instance
    """
    global _runtime_connector

    async with _connector_lock:
        if _runtime_connector is None:
            from osprey.connectors.factory import ConnectorFactory

            if _runtime_config:
                # Use config snapshot from context (reproducible)
                logger.debug(
                    f"Creating connector from context config: {_runtime_config.get('type')}"
                )
                _runtime_connector = await ConnectorFactory.create_control_system_connector(
                    config=_runtime_config
                )
            else:
                # Use current global config
                logger.debug("Creating connector from global config")
                _runtime_connector = await ConnectorFactory.create_control_system_connector(
                    config=None
                )

    return _runtime_connector


# ========================================================
# Internal async implementations
# ========================================================


async def _write_channel_async(channel_address: str, value: Any, **kwargs) -> None:
    """Internal async implementation for writing to a channel.

    The connector handles limits validation when available.
    The injected _limits_validator provides a safety net for subprocess execution
    where the connector's config-based validator may not be initialized.
    """
    # Safety net: validate against injected limits validator (set by execution wrapper)
    # This catches violations even when the connector's own validator isn't configured
    if _limits_validator is not None:
        _limits_validator.validate(channel_address, value)  # Raises ChannelLimitsViolationError

    connector = await _get_connector()
    result = await connector.write_channel(channel_address, value, **kwargs)

    if not result.success:
        raise RuntimeError(f"Write failed for {channel_address}: {result.error_message}")

    # Log success with verification info if available
    if result.verification and result.verification.verified:
        logger.debug(f"Wrote {channel_address} = {value} [{result.verification.level} verified]")
    else:
        logger.debug(f"Wrote {channel_address} = {value}")


async def _read_channel_async(channel_address: str, **kwargs) -> Any:
    """Internal async implementation for reading from a channel."""
    connector = await _get_connector()
    pv_value = await connector.read_channel(channel_address, **kwargs)
    return pv_value.value


async def _write_channels_async(channel_values: dict[str, Any], **kwargs) -> None:
    """Internal async implementation for writing multiple channels."""
    for channel_address, value in channel_values.items():
        await _write_channel_async(channel_address, value, **kwargs)


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
    Uses the control system configuration from when the code was generated.

    Synchronous function - no 'await' needed. Works like EPICS caput().

    Args:
        channel_address: Channel/PV name to write to
        value: Value to write (will be coerced to appropriate type)
        **kwargs: Additional arguments passed to connector
                  - timeout: Operation timeout in seconds
                  - verification_level: 'none', 'callback', or 'readback'
                  - tolerance: Tolerance for readback verification

    Raises:
        ChannelLimitsViolationError: If value violates channel safety limits
        RuntimeError: If write operation fails
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
        **kwargs: Additional arguments passed to each write

    Raises:
        RuntimeError: If any write operation fails

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
