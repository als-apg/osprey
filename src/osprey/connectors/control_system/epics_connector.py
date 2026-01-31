"""
EPICS control system connector using pyepics.

Provides interface to EPICS Channel Access (CA) control system.
Refactored from existing EPICS integration code.

Related to Issue #18 - Control System Abstraction (Layer 2 - EPICS Implementation)
"""

import asyncio
import os
import threading
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
from osprey.utils.logger import get_logger

logger = get_logger("epics_connector")


class EPICSConnector(ControlSystemConnector):
    """
    EPICS control system connector using pyepics.

    Provides read/write access to EPICS Process Variables through
    Channel Access protocol. Supports gateway configuration for
    remote access and read-only/write-access gateways.

    Example:
        Direct gateway connection:
        >>> config = {
        >>>     'timeout': 5.0,
        >>>     'gateways': {
        >>>         'read_only': {
        >>>             'address': 'cagw-alsdmz.als.lbl.gov',
        >>>             'port': 5064
        >>>         }
        >>>     }
        >>> }
        >>> connector = EPICSConnector()
        >>> await connector.connect(config)
        >>> value = await connector.read_pv('BEAM:CURRENT')
        >>> print(f"Beam current: {value.value} {value.metadata.units}")

        SSH tunnel connection:
        >>> config = {
        >>>     'timeout': 5.0,
        >>>     'gateways': {
        >>>         'read_only': {
        >>>             'address': 'localhost',
        >>>             'port': 5074,
        >>>             'use_name_server': True
        >>>         }
        >>>     }
        >>> }
        >>> connector = EPICSConnector()
        >>> await connector.connect(config)
        >>> value = await connector.read_pv('BEAM:CURRENT')
        >>> print(f"Beam current: {value.value} {value.metadata.units}")
    """

    def __init__(self):
        self._connected = False
        self._subscriptions: dict[str, Any] = {}
        self._pv_cache: dict[str, Any] = {}
        self._pv_cache_lock = threading.Lock()  # Thread safety for PV cache
        self._epics_configured = False
        self._limits_validator = None  # Initialized during connect()

    async def connect(self, config: dict[str, Any]) -> None:
        """
        Configure EPICS environment and test connection.

        Args:
            config: Configuration with keys:
                - timeout: Default timeout in seconds (default: 5.0)
                - gateways: Gateway configuration dict with:
                    - read_only: {address, port, use_name_server} for read operations
                    - write_access: {address, port, use_name_server} for write operations

                Gateway sub-keys:
                    - address: Gateway hostname or IP
                    - port: Gateway port number
                    - use_name_server: (optional) Use EPICS_CA_NAME_SERVERS instead of
                      EPICS_CA_ADDR_LIST. Required for SSH tunnels. Default: False

        Raises:
            ImportError: If pyepics is not installed
        """
        # Import epics here to give clear error if not installed
        try:
            import epics

            self._epics = epics
        except ImportError:
            raise ImportError(
                "pyepics is required for EPICS connector. Install with: pip install pyepics"
            ) from None

        # Extract gateway configuration
        gateway_config = config.get("gateways", {}).get("read_only", {})
        if gateway_config:
            address = gateway_config.get("address", "")
            port = gateway_config.get("port", 5064)
            # Explicit configuration for connection method
            # Config system automatically converts "true"/"false" strings to booleans
            use_name_server = gateway_config.get("use_name_server", False)

            # Configure EPICS environment variables
            if use_name_server:
                # Use CA_NAME_SERVERS (required for SSH tunnels and some gateway configurations)
                os.environ["EPICS_CA_NAME_SERVERS"] = f"{address}:{port}"
                logger.debug(f"Using EPICS_CA_NAME_SERVERS: {address}:{port}")
            else:
                # Use CA_ADDR_LIST (standard gateway configuration)
                os.environ["EPICS_CA_ADDR_LIST"] = address
                os.environ["EPICS_CA_SERVER_PORT"] = str(port)
                logger.debug(f"Using EPICS_CA_ADDR_LIST: {address}, CA_SERVER_PORT: {port}")

            os.environ["EPICS_CA_AUTO_ADDR_LIST"] = "NO"

            # Clear EPICS cache to pick up new environment
            self._epics.ca.clear_cache()

            logger.debug(f"Configured EPICS gateway: {address}:{port}")
            self._epics_configured = True

        self._timeout = config.get("timeout", 5.0)

        # Initialize limits validator for automatic validation and verification config
        from osprey.services.python_executor.execution.limits_validator import LimitsValidator

        self._limits_validator = LimitsValidator.from_config()
        if self._limits_validator:
            logger.debug("EPICS connector: limits validator initialized")

        self._connected = True
        logger.debug("EPICS connector initialized")

    async def disconnect(self) -> None:
        """Cleanup EPICS connections."""
        # Unsubscribe from all active subscriptions
        for sub_id in list(self._subscriptions.keys()):
            await self.unsubscribe(sub_id)

        # Disconnect and clear cached PVs
        with self._pv_cache_lock:
            for pv in self._pv_cache.values():
                try:
                    pv.disconnect()
                except Exception:
                    pass  # Best effort cleanup
            self._pv_cache.clear()

        self._connected = False
        logger.info("EPICS connector disconnected")

    async def read_channel(
        self, channel_address: str, timeout: float | None = None
    ) -> ChannelValue:
        """
        Read current value from EPICS channel.

        Args:
            channel_address: EPICS channel address (e.g., 'BEAM:CURRENT')
            timeout: Timeout in seconds (uses default if None)

        Returns:
            ChannelValue with current value, timestamp, and metadata

        Raises:
            ConnectionError: If channel cannot be connected
            TimeoutError: If operation times out
        """
        timeout = timeout or self._timeout

        # Use asyncio.to_thread for blocking EPICS operations
        pv_result = await asyncio.to_thread(self._read_channel_sync, channel_address, timeout)

        return pv_result

    def _read_channel_sync(self, pv_address: str, timeout: float) -> ChannelValue:
        """Synchronous PV read (runs in thread pool).

        Uses PV cache to reuse PV objects for the same channel address.
        This prevents subscription floods when reading the same channel rapidly,
        which can crash soft IOCs like caproto due to race conditions.
        """
        # Get or create cached PV object (thread-safe)
        with self._pv_cache_lock:
            if pv_address not in self._pv_cache:
                self._pv_cache[pv_address] = self._epics.PV(pv_address)
            pv = self._pv_cache[pv_address]

        pv.wait_for_connection(timeout=timeout)

        if not pv.connected:
            raise ConnectionError(
                f"Failed to connect to PV '{pv_address}' (timeout after {timeout}s)"
            )

        value = pv.value

        # Get timestamp from EPICS (seconds since epoch)
        if pv.timestamp:
            timestamp = datetime.fromtimestamp(pv.timestamp)
        else:
            timestamp = datetime.now()

        # Extract metadata
        metadata = ChannelMetadata(
            units=getattr(pv, "units", "") or "",
            precision=getattr(pv, "precision", None),
            alarm_status=pv.status if hasattr(pv, "status") else None,
            timestamp=timestamp,
            raw_metadata={
                "severity": getattr(pv, "severity", None),
                "type": getattr(pv, "type", None),
                "count": getattr(pv, "count", None),
            },
        )

        return ChannelValue(value=value, timestamp=timestamp, metadata=metadata)

    def _get_verification_config(
        self, channel_address: str, value: float
    ) -> tuple[str, float | None]:
        """Get verification level and tolerance for a channel write.

        Priority:
        1. Explicit parameters (for backward compatibility / manual override)
        2. Per-channel config from limits database
        3. Global config from config.yml
        4. Fallback: callback with 0.1% tolerance

        Args:
            channel_address: Channel being written
            value: Value being written (for percentage tolerance calculation)

        Returns:
            Tuple of (verification_level, tolerance)
        """
        # Try per-channel config first (if limits validator available)
        if self._limits_validator:
            level, tolerance = self._limits_validator.get_verification_config(
                channel_address, value
            )
            if level is not None:
                logger.debug(f"Using per-channel verification for {channel_address}: {level}")
                return level, tolerance

        # Fall back to global config (or hardcoded defaults if config unavailable)
        try:
            from osprey.utils.config import get_config_value

            level = get_config_value("control_system.write_verification.default_level", "callback")

            # Calculate tolerance for readback verification
            tolerance = None
            if level == "readback":
                default_percent = get_config_value(
                    "control_system.write_verification.default_tolerance_percent", 0.1
                )
                tolerance = abs(value) * default_percent / 100.0

            logger.debug(f"Using global verification config for {channel_address}: {level}")
            return level, tolerance
        except (FileNotFoundError, KeyError, RuntimeError):
            # Config not available - use hardcoded safe defaults
            logger.debug(
                f"Using hardcoded verification defaults for {channel_address} (config unavailable)"
            )
            return "callback", None

    async def write_channel(
        self,
        channel_address: str,
        value: Any,
        timeout: float | None = None,
        verification_level: str | None = None,
        tolerance: float | None = None,
    ) -> ChannelWriteResult:
        """
        Write value to EPICS channel with automatic limits validation and verification.

        The connector automatically:
        1. Validates limits (min/max/step/writable) if limits checking enabled
        2. Determines verification level from per-channel or global config
        3. Executes write with appropriate verification

        Args:
            channel_address: EPICS channel address
            value: Value to write
            timeout: Timeout in seconds
            verification_level: Optional override for verification level (auto-determined if None)
            tolerance: Optional override for tolerance (auto-calculated if None)

        Returns:
            ChannelWriteResult with write status and verification details

        Raises:
            ConnectionError: If channel cannot be connected
            TimeoutError: If operation times out
            ChannelLimitsViolationError: If limits validation fails (when enabled)
        """
        timeout = timeout or self._timeout

        # Step 1: Validate limits (if enabled)
        if self._limits_validator:
            try:
                self._limits_validator.validate(channel_address, value)
                logger.debug(f"✓ Limits validation passed: {channel_address}={value}")
            except Exception as e:
                # Import here to avoid circular dependency
                from osprey.services.python_executor.exceptions import ChannelLimitsViolationError

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

        # Step 3: Execute write with verification
        if verification_level == "none":
            # Fast path - no verification, no wait for callback
            success = await asyncio.to_thread(
                self._epics.caput, channel_address, value, wait=False, timeout=timeout
            )

            if not success:
                return ChannelWriteResult(
                    channel_address=channel_address,
                    value_written=value,
                    success=False,
                    verification=WriteVerification(
                        level="none", verified=False, notes="Write command failed"
                    ),
                    error_message=f"Failed to write to channel '{channel_address}'",
                )

            logger.debug(f"EPICS write (no verification): {channel_address} = {value}")
            return ChannelWriteResult(
                channel_address=channel_address,
                value_written=value,
                success=True,
                verification=WriteVerification(
                    level="none", verified=False, notes="No verification requested"
                ),
            )

        elif verification_level == "callback":
            # EPICS callback - wait for IOC to confirm processing
            success = await asyncio.to_thread(
                self._epics.caput,
                channel_address,
                value,
                wait=True,  # Wait for IOC callback
                timeout=timeout,
            )

            if not success:
                return ChannelWriteResult(
                    channel_address=channel_address,
                    value_written=value,
                    success=False,
                    verification=WriteVerification(
                        level="callback", verified=False, notes="IOC callback failed or timeout"
                    ),
                    error_message=f"Failed to write to channel '{channel_address}'",
                )

            logger.debug(f"EPICS write (callback verified): {channel_address} = {value}")
            return ChannelWriteResult(
                channel_address=channel_address,
                value_written=value,
                success=True,
                verification=WriteVerification(
                    level="callback", verified=True, notes="IOC callback confirmed"
                ),
            )

        elif verification_level == "readback":
            # Full verification - callback + readback
            success = await asyncio.to_thread(
                self._epics.caput, channel_address, value, wait=True, timeout=timeout
            )

            if not success:
                return ChannelWriteResult(
                    channel_address=channel_address,
                    value_written=value,
                    success=False,
                    verification=WriteVerification(
                        level="readback", verified=False, notes="Write command failed"
                    ),
                    error_message=f"Failed to write to channel '{channel_address}'",
                )

            # Read back to verify
            try:
                readback = await self.read_channel(channel_address, timeout=timeout)

                # Check tolerance
                diff = abs(float(readback.value) - float(value))
                verified = diff <= (tolerance or 0.001)

                logger.debug(
                    f"EPICS write (readback verified={verified}): {channel_address} = {value}, "
                    f"readback = {readback.value}, diff = {diff:.6f}, tolerance = {tolerance}"
                )

                return ChannelWriteResult(
                    channel_address=channel_address,
                    value_written=value,
                    success=True,
                    verification=WriteVerification(
                        level="readback",
                        verified=verified,
                        readback_value=float(readback.value),
                        tolerance_used=tolerance,
                        notes=(
                            f"Readback: {readback.value}, tolerance: ±{tolerance}, diff: {diff:.6f}"
                            if verified
                            else f"Readback mismatch: {readback.value} (expected {value}, diff: {diff:.6f} > tolerance {tolerance})"
                        ),
                    ),
                )

            except Exception as e:
                logger.warning(f"EPICS readback failed for {channel_address}: {e}")
                return ChannelWriteResult(
                    channel_address=channel_address,
                    value_written=value,
                    success=True,  # Write succeeded, but readback failed
                    verification=WriteVerification(
                        level="readback", verified=False, notes=f"Readback failed: {str(e)}"
                    ),
                    error_message=f"Readback verification failed: {str(e)}",
                )

        else:
            raise ValueError(
                f"Invalid verification_level: {verification_level}. Must be 'none', 'callback', or 'readback'"
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
        Subscribe to channel value changes.

        Args:
            channel_address: EPICS channel address
            callback: Function to call when value changes

        Returns:
            Subscription ID for later unsubscription
        """
        loop = asyncio.get_event_loop()

        def epics_callback(pvname=None, value=None, timestamp=None, **kwargs):
            """Wrapper to convert EPICS callback to our format."""
            pv_value = ChannelValue(
                value=value,
                timestamp=datetime.fromtimestamp(timestamp) if timestamp else datetime.now(),
                metadata=ChannelMetadata(
                    units=kwargs.get("units", ""), alarm_status=kwargs.get("status")
                ),
            )
            # Schedule callback in event loop
            loop.call_soon_threadsafe(callback, pv_value)

        # Create PV and add callback
        pv = self._epics.PV(channel_address, callback=epics_callback)

        # Generate subscription ID
        sub_id = f"{channel_address}_{id(pv)}"
        self._subscriptions[sub_id] = pv

        logger.debug(f"EPICS subscription created: {sub_id}")
        return sub_id

    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from PV changes."""
        if subscription_id in self._subscriptions:
            pv = self._subscriptions[subscription_id]
            pv.clear_callbacks()
            del self._subscriptions[subscription_id]
            logger.debug(f"EPICS subscription removed: {subscription_id}")

    async def get_metadata(self, channel_address: str) -> ChannelMetadata:
        """Get metadata for a channel."""
        channel_value = await self.read_channel(channel_address)
        return channel_value.metadata

    async def validate_channel(self, channel_address: str) -> bool:
        """
        Check if channel exists and is accessible.

        Args:
            channel_address: EPICS channel address

        Returns:
            True if channel can be accessed
        """
        try:
            await self.read_channel(channel_address, timeout=2.0)
            return True
        except Exception as e:
            logger.debug(f"Channel validation failed for {channel_address}: {e}")
            return False
