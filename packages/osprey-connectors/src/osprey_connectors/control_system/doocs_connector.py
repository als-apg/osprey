"""
DOOCS control system connector using doocs4py.

Provides interface to the DOOCS control system.

Author: Frank Mayet (DESY, MXL)
Date: 2026-07-01
"""

import asyncio
import secrets
from collections.abc import Callable
from datetime import datetime
from typing import Any

from osprey_connectors.config import get_facility_timezone
from osprey_connectors.control_system.base import (
    ChannelMetadata,
    ChannelValue,
    ChannelWriteResult,
    ControlSystemConnector,
    WriteOutcome,
    values_match,
)
from osprey_connectors.control_system.limits_validator import LimitsValidator
from osprey_connectors.logger import get_logger

logger = get_logger("doocs_connector")


class DOOCSConnector(ControlSystemConnector):
    """
    DOOCS control system connector using doocs4py

    Provides read/write access to DOOCS properties.
    """

    def __init__(self):
        self._connected: bool = False
        self._subscriptions: dict[str, Any] = {}

    async def connect(self, config: dict[str, Any]) -> None:
        """
        Configure DOOCS environment and test connection.

        Args:
            config: No config needed for DOOCS

        Raises:
            ImportError: If doocs4py is not installed
        """
        # Import doocs4py here and give clear error if not installed
        try:
            import doocs4py

            self._doocs4py = doocs4py
            logger.debug(f"DOOCS connector: doocs4py version {self._doocs4py.__version__} loaded")
        except ImportError:
            raise ImportError("doocs4py is required for the DOOCS connector.") from None

        # Initialize limits validator for automatic validation and confirm policy
        self._limits_validator = LimitsValidator.from_config(connector_type=self._connector_type)
        if self._limits_validator:
            logger.debug("DOOCS connector: limits validator initialized")

        # Test connection using a doocs4py.names call, listing all FACILITYs
        try:
            facilities = [f[1] for f in self._doocs4py.names("*")]
            logger.debug(
                "DOOCS connector: ENS connection successful."
                f"Available FACILITIEs: {len(facilities)}"
            )
        except Exception:
            raise Exception("DOOCS connector failed to connect to the ENS.") from None

        self._connected = True
        logger.debug("DOOCS connector initialized")

    async def disconnect(self) -> None:
        """Cleanup DOOCS connections."""
        # Unsubscribe from all active subscriptions
        for sub_id in list(self._subscriptions.keys()):
            await self.unsubscribe(sub_id)

        self._connected = False
        logger.info("DOOCS connector disconnected")

    async def read_channel(
        self, channel_address: str, timeout: float | None = None
    ) -> ChannelValue:
        """
        Read current value from a DOOCS property.

        Args:
            channel_address: DOOCS address (e.g., 'FACILITY/DEVICE/LOCATION/PROPERTY')
            timeout: Not supported by doocs4py

        Returns:
            ChannelValue with current value, timestamp, and metadata

        Raises:
            ConnectionError: If channel cannot be connected
            TimeoutError: If operation times out
        """

        # Use asyncio.to_thread for blocking DOOCS operations
        read_result = await asyncio.to_thread(self._read_channel_sync, channel_address)

        return read_result

    def _read_channel_sync(self, address: str) -> ChannelValue:
        """Synchronous DOOCS read (runs in thread pool)."""

        data = self._doocs4py.get(address)  # EqData

        value = data.get_data()
        macropulse = data.macropulse
        timestamp_s, timestamp_us = data.timestamp.get_seconds_and_microseconds_since_epoch()
        timestamp_float = timestamp_s + timestamp_us / 1e6

        timestamp = datetime.fromtimestamp(timestamp_float, get_facility_timezone())

        # Compile metadata
        metadata = ChannelMetadata(
            units="",
            precision=None,
            alarm_status=None,
            timestamp=timestamp,
            raw_metadata={
                "macropulse": macropulse,
                "type": type(value),
            },
        )

        return ChannelValue(value=value, timestamp=timestamp, metadata=metadata)

    async def write_channel(
        self,
        channel_address: str,
        value: Any,
        timeout: float | None = None,
        confirm: bool | None = None,
    ) -> ChannelWriteResult:
        """
        Write a value to a DOOCS property, confirming it unless asked not to.

        DOOCS has no acknowledgement of its own — ``set()`` either returns or
        raises — so a confirmed write is the value sent followed by one fresh
        read of the same property, compared with
        :func:`~osprey_connectors.control_system.base.values_match`. DOOCS
        reports no enum labels and no alarm state, so the comparison gets no
        label and the alarm fields stay unset.

        Args:
            channel_address: DOOCS address
            value: Value to write
            timeout: Optional timeout for the confirming read
            confirm: Whether to re-read the property and compare, or ``None``
                to resolve the policy for this channel from the limits database

        Returns:
            ChannelWriteResult carrying the outcome and what the property was
            seen to hold

        Raises:
            ChannelLimitsViolationError: If limits validation fails (when enabled)
        """

        # Step 1: Validate limits (if enabled)
        if self._limits_validator:
            try:
                self._limits_validator.validate(channel_address, value)
                logger.debug(f"✓ Limits validation passed: {channel_address}={value}")
            except Exception as e:
                # Import here to avoid circular dependency
                from osprey_connectors.errors import ChannelLimitsViolationError

                # Re-raise limits violations
                if isinstance(e, ChannelLimitsViolationError):
                    raise

                # Log unexpected errors but don't block (fail-open for non-limit errors)
                logger.warning(f"Limits validation error (non-blocking): {e}")

        # Step 2: Resolve the confirmation policy. An explicit confirm — False
        # every bit as much as True — is an answer and is taken as given; only
        # an omitted one is resolved from the limits database.
        if confirm is None:
            confirm = self._resolve_confirm(channel_address)

        # Step 3: Send the value
        try:
            self._doocs4py.set(channel_address, value)
        except Exception as e:
            return ChannelWriteResult(
                channel_address=channel_address,
                value_written=value,
                outcome=WriteOutcome.FAILED,
                error_message=f"Failed to write to '{channel_address}': {e}",
                notes="DOOCS did not take the value",
            )

        if not confirm:
            logger.debug(f"DOOCS write (unconfirmed by request): {channel_address} = {value}")
            return ChannelWriteResult(
                channel_address=channel_address,
                value_written=value,
                outcome=WriteOutcome.UNREQUESTED,
                notes="No confirmation requested",
            )

        # Step 4: Confirm with one fresh read
        try:
            readback = await self.read_channel(channel_address, timeout=timeout)
        except Exception as e:
            logger.warning(f"DOOCS confirming read failed for {channel_address}: {e}")
            return ChannelWriteResult(
                channel_address=channel_address,
                value_written=value,
                outcome=WriteOutcome.UNCONFIRMED,
                error_message=f"Confirming read of '{channel_address}' failed: {e}",
                notes="The value was sent; what the property holds is unknown",
            )

        observed = readback.value
        confirmed = values_match(value, observed)

        logger.debug(
            f"DOOCS write ({'confirmed' if confirmed else 'mismatch'}): "
            f"{channel_address} = {value!r}, observed {observed!r}"
        )

        return ChannelWriteResult(
            channel_address=channel_address,
            value_written=value,
            outcome=WriteOutcome.CONFIRMED if confirmed else WriteOutcome.MISMATCH,
            observed_value=observed,
            notes=(
                f"Observed {observed!r}" if confirmed else f"Observed {observed!r}, sent {value!r}"
            ),
        )

    async def read_multiple_channels(
        self, channel_addresses: list[str], timeout: float | None = None
    ) -> dict[str, ChannelValue]:
        """Read multiple channels concurrently."""
        tasks = [self.read_channel(ch_addr, timeout) for ch_addr in channel_addresses]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            ch_addr: result
            for ch_addr, result in zip(channel_addresses, results, strict=False)
            if not isinstance(result, Exception)
        }

    async def subscribe(
        self, channel_address: str, callback: Callable[[ChannelValue], None]
    ) -> str:
        """
        Subscribe to property value changes.

        Args:
            channel_address: DOOCS address
            callback: Function to call when value changes

        Returns:
            Subscription ID for later unsubscription
        """
        loop = asyncio.get_event_loop()

        def doocs_callback(data):  # EqData
            """Wrapper to convert DOOCS callback to Osprey format."""
            value = data.get_data()
            macropulse = data.macropulse
            timestamp_s, timestamp_us = data.timestamp.get_seconds_and_microseconds_since_epoch()
            timestamp_float = timestamp_s + timestamp_us / 1e6

            timestamp = datetime.fromtimestamp(timestamp_float, get_facility_timezone())

            # Compile metadata
            metadata = ChannelMetadata(
                units="",
                precision=None,
                alarm_status=None,
                timestamp=timestamp,
                raw_metadata={
                    "macropulse": macropulse,
                    "type": type(value),
                },
            )

            prop_value = ChannelValue(value=value, timestamp=timestamp, metadata=metadata)
            # Schedule callback in event loop
            loop.call_soon_threadsafe(callback, prop_value)

        # Subscribe
        address = self._doocs4py.Address(channel_address)
        self._doocs4py.subscribe(address, doocs_callback)

        # Generate subscription ID
        sub_id = f"{channel_address}_{secrets.token_hex(8)}"
        self._subscriptions[sub_id] = self._doocs4py.Address(channel_address)

        logger.debug(f"DOOCS subscription created: {sub_id}")
        return sub_id

    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from DOOCS property changes."""
        if subscription_id in self._subscriptions:
            address = self._subscriptions[subscription_id]
            self._doocs4py.unsubscribe(address)
            del self._subscriptions[subscription_id]
            logger.debug(f"DOOCS subscription removed: {subscription_id}")

    async def get_metadata(self, channel_address: str) -> ChannelMetadata:
        """Get metadata for a channel."""
        channel_value = await self.read_channel(channel_address)
        return channel_value.metadata

    async def validate_channel(self, channel_address: str) -> bool:
        """
        Check if property exists and is accessible.

        Args:
            channel_address: DOOCS address

        Returns:
            True if channel can be accessed
        """
        try:
            await self.read_channel(channel_address)
            return True
        except Exception as e:
            logger.debug(f"DOOCS property validation failed for {channel_address}: {e}")
            return False
