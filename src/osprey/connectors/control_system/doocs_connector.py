"""
DOOCS control system connector using doocs4py.

Provides interface to the DOOCS control system.

Author: Frank Mayet (DESY, MXL)
Date: 2026-07-01
"""

########################################################################################

import asyncio
import secrets
from collections.abc import Callable
from datetime import datetime
from typing import Any

from osprey.connectors.control_system.base import (
    ChannelMetadata,
    ChannelValue,
    ChannelWriteResult,
    ControlSystemConnector,
    WriteVerification,
)
from osprey.connectors.control_system.limits_validator import LimitsValidator
from osprey.utils.config import get_facility_timezone
from osprey.utils.logger import get_logger

logger = get_logger("doocs_connector")

########################################################################################


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
            logger.debug(
                f"DOOCS connector: doocs4py version {self._doocs4py.__version__} loaded"
            )
        except ImportError:
            raise ImportError("doocs4py is required for the DOOCS connector.") from None

        # Initialize limits validator for automatic validation and verification config
        self._limits_validator = LimitsValidator.from_config()
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
        timestamp_s, timestamp_us = (
            data.timestamp.get_seconds_and_microseconds_since_epoch()
        )
        timestamp_float = timestamp_s + timestamp_us / 1e6

        timestamp = datetime.fromtimestamp(timestamp_float, get_facility_timezone())

        # Compile metadata
        metadata = ChannelMetadata(
            units="",  # TODO: Implement .EGU
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
        verification_level: str | None = None,
        tolerance: float | None = None,
    ) -> ChannelWriteResult:
        """
        Write value to DOOCS property with automatic limits validation and verification.

        The connector automatically:
        1. Validates limits (min/max/step/writable) if limits checking enabled
        2. Determines verification level from per-channel or global config
        3. Executes write with appropriate verification

        Args:
            channel_address: DOOCS address
            value: Value to write
            timeout: Not supported by DOOCS
            verification_level: Optional override for verification level
                                (auto-determined if None)
            tolerance: Optional override for tolerance (auto-calculated if None)

        Returns:
            ChannelWriteResult with write status and verification details

        Raises:
            ConnectionError: If channel cannot be connected
            TimeoutError: If operation times out
            ChannelLimitsViolationError: If limits validation fails (when enabled)
        """

        # Step 1: Validate limits (if enabled)
        if self._limits_validator:
            try:
                self._limits_validator.validate(channel_address, value)
                logger.debug(f"✓ Limits validation passed: {channel_address}={value}")
            except Exception as e:
                # Import here to avoid circular dependency
                from osprey.errors import ChannelLimitsViolationError

                # Re-raise limits violations
                if isinstance(e, ChannelLimitsViolationError):
                    raise

                # Log unexpected errors but don't block (fail-open for non-limit errors)
                logger.warning(f"Limits validation error (non-blocking): {e}")

        # Step 2: Auto-determine verification config if not provided
        if verification_level is None:
            verification_level, auto_tolerance = self._get_verification_config(
                channel_address, float(value)
            )
            if tolerance is None:
                tolerance = auto_tolerance

        if verification_level == "callback":
            logger.debug(
                "DOOCS write: callback verification not supported by DOOCS. "
                "Will perform readback verification"
            )

        # Step 3: Execute write with verification
        if verification_level == "none":
            # Fast path - no verification
            try:
                self._doocs4py.set(channel_address, value)
            except Exception as e:
                return ChannelWriteResult(
                    channel_address=channel_address,
                    value_written=value,
                    success=False,
                    verification=WriteVerification(
                        level="none", verified=False, notes="Write command failed"
                    ),
                    error_message=f"Failed to write to '{channel_address}': {e}",
                )

            logger.debug(f"DOOCS write (no verification): {channel_address} = {value}")
            return ChannelWriteResult(
                channel_address=channel_address,
                value_written=value,
                success=True,
                verification=WriteVerification(
                    level="none", verified=False, notes="No verification requested"
                ),
            )

        elif verification_level == "callback" or verification_level == "readback":
            # Full verification - readback
            try:
                self._doocs4py.set(channel_address, value)
            except Exception as e:
                return ChannelWriteResult(
                    channel_address=channel_address,
                    value_written=value,
                    success=False,
                    verification=WriteVerification(
                        level="none", verified=False, notes="Write command failed"
                    ),
                    error_message=f"Failed to write to '{channel_address}': {e}",
                )

            # Read back to verify
            try:
                readback = await self.read_channel(channel_address, timeout=timeout)

                verified = False
                value_to_report = None
                # FIXME: Osprey should also support writing strings
                # Check tolerance if numerical data
                if isinstance(readback, (float, int)):
                    diff = abs(float(readback.value) - float(value))
                    verified = diff <= (tolerance or 0.001)
                    value_to_report = float(readback.value)

                elif isinstance(readback.value, str):
                    diff = 0.0
                    verified = value == readback
                    value_to_report = 0.0

                logger.debug(
                    f"DOOCS write (readback verified={verified}): "
                    f"{channel_address} = {value}, "
                    f"readback = {readback.value}, diff = {diff:.6f}, "
                    f"tolerance = {tolerance}"
                )

                return ChannelWriteResult(
                    channel_address=channel_address,
                    value_written=value,
                    success=True,
                    verification=WriteVerification(
                        level="readback",
                        verified=verified,
                        readback_value=value_to_report,
                        tolerance_used=tolerance,
                        notes=(
                            (
                                f"Readback: {readback.value}, tolerance: ±{tolerance}, "
                                f"diff: {diff:.6f}"
                            )
                            if verified
                            else (
                                f"Readback mismatch: {readback.value} (expected "
                                f"{value}, diff: {diff:.6f} > tolerance {tolerance})"
                            )
                        ),
                    ),
                )

            except Exception as e:
                logger.warning(f"DOOCS readback failed for {channel_address}: {e}")
                return ChannelWriteResult(
                    channel_address=channel_address,
                    value_written=value,
                    success=True,  # Write succeeded, but readback failed
                    verification=WriteVerification(
                        level="readback",
                        verified=False,
                        notes=f"Readback failed: {str(e)}",
                    ),
                    error_message=f"Readback verification failed: {str(e)}",
                )

        else:
            raise ValueError(
                f"Invalid verification_level: {verification_level}. "
                "Must be 'none', or 'readback'"
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
            timestamp_s, timestamp_us = (
                data.timestamp.get_seconds_and_microseconds_since_epoch()
            )
            timestamp_float = timestamp_s + timestamp_us / 1e6

            timestamp = datetime.fromtimestamp(timestamp_float, get_facility_timezone())

            # Compile metadata
            metadata = ChannelMetadata(
                units="",  # TODO: Implement .EGU
                precision=None,
                alarm_status=None,
                timestamp=timestamp,
                raw_metadata={
                    "macropulse": macropulse,
                    "type": type(value),
                },
            )

            prop_value = ChannelValue(
                value=value, timestamp=timestamp, metadata=metadata
            )
            # Schedule callback in event loop
            loop.call_soon_threadsafe(callback, prop_value)

        # Subscribe
        address = self._doocs4py.Address(channel_address)
        self._doocs4py.subscribe(address)

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
