"""
Mock control system connector for development and testing.

Works with any PV names - generates realistic synthetic data.
Ideal for R&D and development without control room access.

Related to Issue #18 - Control System Abstraction (Layer 2 - Mock Implementation)
"""

import asyncio
from collections.abc import Callable
from datetime import datetime
from typing import Any

import numpy as np

from osprey.connectors.control_system.base import ControlSystemConnector, PVMetadata, PVValue
from osprey.utils.logger import get_logger

logger = get_logger("mock_connector")


class MockConnector(ControlSystemConnector):
    """
    Mock control system connector for development and testing.

    This connector simulates a control system without requiring real hardware.
    It generates realistic synthetic data for any PV name, making it ideal
    for R&D and development when you don't have access to the control room.

    Features:
    - Accepts any PV name
    - Generates realistic initial values based on PV naming conventions
    - Adds configurable noise to simulate real measurements
    - Maintains state between reads and writes
    - Simulates readback PVs (e.g., :SP -> :RB)

    Example:
        >>> config = {
        >>>     'response_delay_ms': 10,
        >>>     'noise_level': 0.01,
        >>>     'enable_writes': True
        >>> }
        >>> connector = MockConnector()
        >>> await connector.connect(config)
        >>> value = await connector.read_pv('BEAM:CURRENT')
        >>> print(f"Beam current: {value.value} {value.metadata.units}")
    """

    def __init__(self):
        self._connected = False
        self._state: dict[str, float] = {}
        self._subscriptions: dict[str, tuple] = {}

    async def connect(self, config: dict[str, Any]) -> None:
        """
        Initialize mock connector.

        Args:
            config: Configuration with keys:
                - response_delay_ms: Simulated response delay (default: 10)
                - noise_level: Relative noise level 0-1 (default: 0.01)
                - enable_writes: Allow write operations (default: True)
        """
        self._response_delay = config.get('response_delay_ms', 10) / 1000.0
        self._noise_level = config.get('noise_level', 0.01)
        self._enable_writes = config.get('enable_writes', True)
        self._connected = True
        logger.debug("Mock connector initialized")

    async def disconnect(self) -> None:
        """Cleanup mock connector."""
        self._state.clear()
        self._subscriptions.clear()
        self._connected = False
        logger.debug("Mock connector disconnected")

    async def read_pv(
        self,
        pv_address: str,
        timeout: float | None = None
    ) -> PVValue:
        """
        Read PV - generates realistic value if not cached.

        Args:
            pv_address: Any PV name (mock accepts all names)
            timeout: Ignored for mock connector

        Returns:
            PVValue with synthetic data
        """
        # Simulate network delay
        await asyncio.sleep(self._response_delay)

        # Get or generate initial value
        if pv_address not in self._state:
            self._state[pv_address] = self._generate_initial_value(pv_address)

        # Add noise
        base_value = self._state[pv_address]
        noise = np.random.normal(0, abs(base_value) * self._noise_level)
        value = base_value + noise

        return PVValue(
            value=value,
            timestamp=datetime.now(),
            metadata=PVMetadata(
                units=self._infer_units(pv_address),
                timestamp=datetime.now(),
                description=f"Mock PV: {pv_address}"
            )
        )

    async def write_pv(
        self,
        pv_address: str,
        value: Any,
        timeout: float | None = None
    ) -> bool:
        """
        Write PV - updates internal state.

        Args:
            pv_address: Any PV name
            value: Value to write
            timeout: Ignored for mock connector

        Returns:
            True if successful, False if writes disabled
        """
        if not self._enable_writes:
            logger.warning(f"Write to {pv_address} rejected (writes disabled)")
            return False

        # Simulate network delay
        await asyncio.sleep(self._response_delay)

        self._state[pv_address] = float(value)
        logger.debug(f"Mock write: {pv_address} = {value}")

        # Update corresponding readback PV
        readback_pv = pv_address.replace(':SP', ':RB').replace(':SET', ':GET')
        if readback_pv != pv_address:
            # Simulate small offset between setpoint and readback
            offset = np.random.normal(0, abs(float(value)) * 0.001)
            self._state[readback_pv] = float(value) + offset

        return True

    async def read_multiple_pvs(
        self,
        pv_addresses: list[str],
        timeout: float | None = None
    ) -> dict[str, PVValue]:
        """Read multiple PVs concurrently."""
        tasks = [self.read_pv(pv) for pv in pv_addresses]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            pv: result
            for pv, result in zip(pv_addresses, results, strict=True)
            if not isinstance(result, Exception)
        }

    async def subscribe(
        self,
        pv_address: str,
        callback: Callable[[PVValue], None]
    ) -> str:
        """
        Subscribe to PV changes.

        Note: Mock connector only triggers callbacks on write_pv calls.
        """
        sub_id = f"mock_{pv_address}_{id(callback)}"
        self._subscriptions[sub_id] = (pv_address, callback)
        logger.debug(f"Mock subscription created: {sub_id}")
        return sub_id

    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from PV changes."""
        if subscription_id in self._subscriptions:
            del self._subscriptions[subscription_id]
            logger.debug(f"Mock subscription removed: {subscription_id}")

    async def get_metadata(self, pv_address: str) -> PVMetadata:
        """Get PV metadata (synthetic for mock)."""
        return PVMetadata(
            units=self._infer_units(pv_address),
            description=f"Mock PV: {pv_address}",
            timestamp=datetime.now()
        )

    async def validate_pv(self, pv_address: str) -> bool:
        """All PV names are valid in mock mode."""
        return True

    def _generate_initial_value(self, pv_name: str) -> float:
        """
        Generate realistic initial value based on PV type.

        Uses naming conventions to infer reasonable values.
        """
        pv_lower = pv_name.lower()

        if 'current' in pv_lower:
            return 500.0 if 'beam' in pv_lower else 150.0
        elif 'voltage' in pv_lower:
            return 5000.0
        elif 'power' in pv_lower:
            return 50.0
        elif 'pressure' in pv_lower:
            return 1e-9
        elif 'temp' in pv_lower:
            return 25.0
        elif 'lifetime' in pv_lower:
            return 10.0
        elif 'position' in pv_lower or 'pos' in pv_lower:
            return 0.0
        elif 'energy' in pv_lower:
            return 1900.0  # MeV for typical storage ring
        else:
            return 100.0

    def _infer_units(self, pv_name: str) -> str:
        """Infer units from PV name."""
        pv_lower = pv_name.lower()

        if 'current' in pv_lower:
            return 'mA' if 'beam' in pv_lower else 'A'
        elif 'voltage' in pv_lower:
            return 'V'
        elif 'power' in pv_lower:
            return 'kW'
        elif 'pressure' in pv_lower:
            return 'Torr'
        elif 'temp' in pv_lower:
            return '°C'
        elif 'lifetime' in pv_lower:
            return 'hours'
        elif 'position' in pv_lower or 'pos' in pv_lower:
            return 'mm'
        elif 'energy' in pv_lower:
            return 'MeV'
        else:
            return ''

