"""
EPICS control system connector using pyepics.

Provides interface to EPICS Channel Access (CA) control system.
Refactored from existing EPICS integration code.

Related to Issue #18 - Control System Abstraction (Layer 2 - EPICS Implementation)
"""

import asyncio
import os
from collections.abc import Callable
from datetime import datetime
from typing import Any

from osprey.connectors.control_system.base import ControlSystemConnector, PVMetadata, PVValue
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
        self._epics_configured = False

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
                "pyepics is required for EPICS connector. "
                "Install with: pip install pyepics"
            )

        # Extract gateway configuration
        gateway_config = config.get('gateways', {}).get('read_only', {})
        if gateway_config:
            address = gateway_config.get('address', '')
            port = gateway_config.get('port', 5064)
            # Explicit configuration for connection method
            # Config system automatically converts "true"/"false" strings to booleans
            use_name_server = gateway_config.get('use_name_server', False)

            # Configure EPICS environment variables
            if use_name_server:
                # Use CA_NAME_SERVERS (required for SSH tunnels and some gateway configurations)
                os.environ['EPICS_CA_NAME_SERVERS'] = f'{address}:{port}'
                logger.debug(f"Using EPICS_CA_NAME_SERVERS: {address}:{port}")
            else:
                # Use CA_ADDR_LIST (standard gateway configuration)
                os.environ['EPICS_CA_ADDR_LIST'] = address
                os.environ['EPICS_CA_SERVER_PORT'] = str(port)
                logger.debug(f"Using EPICS_CA_ADDR_LIST: {address}, CA_SERVER_PORT: {port}")

            os.environ['EPICS_CA_AUTO_ADDR_LIST'] = 'NO'

            # Clear EPICS cache to pick up new environment
            self._epics.ca.clear_cache()

            logger.debug(f"Configured EPICS gateway: {address}:{port}")
            self._epics_configured = True

        self._timeout = config.get('timeout', 5.0)
        self._connected = True
        logger.debug("EPICS connector initialized")

    async def disconnect(self) -> None:
        """Cleanup EPICS connections."""
        # Unsubscribe from all active subscriptions
        for sub_id in list(self._subscriptions.keys()):
            await self.unsubscribe(sub_id)

        self._pv_cache.clear()
        self._connected = False
        logger.info("EPICS connector disconnected")

    async def read_pv(
        self,
        pv_address: str,
        timeout: float | None = None
    ) -> PVValue:
        """
        Read current value from EPICS PV.

        Args:
            pv_address: EPICS PV address (e.g., 'BEAM:CURRENT')
            timeout: Timeout in seconds (uses default if None)

        Returns:
            PVValue with current value, timestamp, and metadata

        Raises:
            ConnectionError: If PV cannot be connected
            TimeoutError: If operation times out
        """
        timeout = timeout or self._timeout

        # Use asyncio.to_thread for blocking EPICS operations
        pv_result = await asyncio.to_thread(
            self._read_pv_sync,
            pv_address,
            timeout
        )

        return pv_result

    def _read_pv_sync(self, pv_address: str, timeout: float) -> PVValue:
        """Synchronous PV read (runs in thread pool)."""
        pv = self._epics.PV(pv_address)
        pv.wait_for_connection(timeout=timeout)

        if not pv.connected:
            raise ConnectionError(
                f"Failed to connect to PV '{pv_address}' "
                f"(timeout after {timeout}s)"
            )

        value = pv.value

        # Get timestamp from EPICS (seconds since epoch)
        if pv.timestamp:
            timestamp = datetime.fromtimestamp(pv.timestamp)
        else:
            timestamp = datetime.now()

        # Extract metadata
        metadata = PVMetadata(
            units=getattr(pv, 'units', '') or '',
            precision=getattr(pv, 'precision', None),
            alarm_status=pv.status if hasattr(pv, 'status') else None,
            timestamp=timestamp,
            raw_metadata={
                'severity': getattr(pv, 'severity', None),
                'type': getattr(pv, 'type', None),
                'count': getattr(pv, 'count', None),
            }
        )

        return PVValue(
            value=value,
            timestamp=timestamp,
            metadata=metadata
        )

    async def write_pv(
        self,
        pv_address: str,
        value: Any,
        timeout: float | None = None
    ) -> bool:
        """
        Write value to EPICS PV.

        Args:
            pv_address: EPICS PV address
            value: Value to write
            timeout: Timeout in seconds

        Returns:
            True if write was successful

        Raises:
            ConnectionError: If PV cannot be connected
            TimeoutError: If operation times out
        """
        timeout = timeout or self._timeout

        success = await asyncio.to_thread(
            self._epics.caput,
            pv_address,
            value,
            timeout
        )

        if not success:
            raise ConnectionError(f"Failed to write to PV '{pv_address}'")

        logger.debug(f"EPICS write: {pv_address} = {value}")
        return bool(success)

    async def read_multiple_pvs(
        self,
        pv_addresses: list[str],
        timeout: float | None = None
    ) -> dict[str, PVValue]:
        """Read multiple PVs concurrently."""
        tasks = [self.read_pv(pv_addr, timeout) for pv_addr in pv_addresses]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            pv_addr: result
            for pv_addr, result in zip(pv_addresses, results)
            if not isinstance(result, Exception)
        }

    async def subscribe(
        self,
        pv_address: str,
        callback: Callable[[PVValue], None]
    ) -> str:
        """
        Subscribe to PV value changes.

        Args:
            pv_address: EPICS PV address
            callback: Function to call when value changes

        Returns:
            Subscription ID for later unsubscription
        """
        loop = asyncio.get_event_loop()

        def epics_callback(pvname=None, value=None, timestamp=None, **kwargs):
            """Wrapper to convert EPICS callback to our format."""
            pv_value = PVValue(
                value=value,
                timestamp=datetime.fromtimestamp(timestamp) if timestamp else datetime.now(),
                metadata=PVMetadata(
                    units=kwargs.get('units', ''),
                    alarm_status=kwargs.get('status')
                )
            )
            # Schedule callback in event loop
            loop.call_soon_threadsafe(callback, pv_value)

        # Create PV and add callback
        pv = self._epics.PV(pv_address, callback=epics_callback)

        # Generate subscription ID
        sub_id = f"{pv_address}_{id(pv)}"
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

    async def get_metadata(self, pv_address: str) -> PVMetadata:
        """Get metadata for a PV."""
        pv_value = await self.read_pv(pv_address)
        return pv_value.metadata

    async def validate_pv(self, pv_address: str) -> bool:
        """
        Check if PV exists and is accessible.

        Args:
            pv_address: EPICS PV address

        Returns:
            True if PV can be accessed
        """
        try:
            await self.read_pv(pv_address, timeout=2.0)
            return True
        except Exception as e:
            logger.debug(f"PV validation failed for {pv_address}: {e}")
            return False

