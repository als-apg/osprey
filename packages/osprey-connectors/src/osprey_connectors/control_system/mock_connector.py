"""
Mock control system connector for development and testing.

Works with any PV names - generates realistic synthetic data.
Ideal for R&D and development without control room access.

"""

import asyncio
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from osprey_connectors.simulation import SimulationEngine

from osprey_connectors.channel_taxonomy import classify_channel
from osprey_connectors.config import get_facility_timezone
from osprey_connectors.control_system.base import (
    ChannelMetadata,
    ChannelValue,
    ChannelWriteResult,
    ControlSystemConnector,
    WriteOutcome,
    values_match,
)
from osprey_connectors.logger import get_logger
from osprey_connectors.simulation import engine_serves

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
        >>> }
        >>> connector = MockConnector()
        >>> await connector.connect(config)
        >>> value = await connector.read_channel('BEAM:CURRENT')
        >>> print(f"Beam current: {value.value} {value.metadata.units}")
    """

    def __init__(self):
        self._connected = False
        self._state: dict[str, float] = {}
        self._subscriptions: dict[str, tuple] = {}
        self._sim_engine: SimulationEngine | None = None

    async def connect(self, config: dict[str, Any]) -> None:
        """
        Initialize mock connector.

        Args:
            config: Configuration with keys:
                - response_delay_ms: Simulated response delay (default: 10)
                - noise_level: Relative noise level 0-1 (default: 0.01)
                - simulation_file: Optional path to a machine.json driving the
                  data-driven simulation engine (relative paths resolve against
                  the project root). Without it, every PV is served procedurally.
        """
        self._response_delay = config.get("response_delay_ms", 10) / 1000.0
        self._noise_level = config.get("noise_level", 0.01)

        # Initialize limits validator for automatic validation and confirm policy
        from osprey_connectors.control_system.limits_validator import LimitsValidator

        self._limits_validator = LimitsValidator.from_config(connector_type=self._connector_type)
        if self._limits_validator:
            logger.debug("Mock connector: limits validator initialized")

        # Optional data-driven simulation engine (machine file)
        from osprey_connectors.simulation import engine_from_connector_config

        self._sim_engine = engine_from_connector_config(config)

        self._connected = True
        logger.debug("Mock connector initialized")

    async def disconnect(self) -> None:
        """Cleanup mock connector."""
        self._state.clear()
        self._subscriptions.clear()
        self._sim_engine = None
        self._connected = False
        logger.debug("Mock connector disconnected")

    async def read_channel(
        self, channel_address: str, timeout: float | None = None
    ) -> ChannelValue:
        """
        Read channel - generates realistic value if not cached.

        Args:
            channel_address: Any channel name (mock accepts all names)
            timeout: Ignored for mock connector

        Returns:
            ChannelValue with synthetic data
        """
        # Simulate network delay
        await asyncio.sleep(self._response_delay)

        return self._read_value(channel_address, apply_noise=True)

    async def _confirming_read(self, channel_address: str) -> ChannelValue:
        """Read a channel back to confirm a write, without measurement noise.

        Confirmation reports what the simulated control system *holds*, and the
        store holds exactly what was put there. The noise ``read_channel``
        injects models the jitter of measuring a live signal, so applying it
        here would manufacture a mismatch on every write at any noise level.
        """
        await asyncio.sleep(self._response_delay)

        return self._read_value(channel_address, apply_noise=False)

    def _read_value(self, channel_address: str, *, apply_noise: bool) -> ChannelValue:
        """Build the reading for ``channel_address`` from the simulated machine.

        Args:
            channel_address: Any channel name
            apply_noise: Whether to add measurement noise to the held value.
                Engine-served channels carry whatever the machine file makes
                them report either way.

        Returns:
            ChannelValue with synthetic data
        """
        # Simulation engine serves its channels; unknown PVs fall back to procedural
        if engine_serves(self._sim_engine, channel_address):
            reading = self._sim_engine.read(channel_address)
            now = datetime.now(get_facility_timezone())
            return ChannelValue(
                value=reading.value,
                timestamp=now,
                metadata=ChannelMetadata(
                    units=reading.units,
                    timestamp=now,
                    description=reading.description,
                ),
            )

        # Get or generate initial value
        if channel_address not in self._state:
            self._state[channel_address] = self._generate_initial_value(channel_address)

        value = self._state[channel_address]
        if apply_noise:
            # Add noise, floored per kind so a 0.0 baseline is not dead-flat.
            sigma = classify_channel(channel_address).noise_sigma(value, self._noise_level)
            value = value + np.random.normal(0, sigma)

        return ChannelValue(
            value=value,
            timestamp=datetime.now(get_facility_timezone()),
            metadata=ChannelMetadata(
                units=self._infer_units(channel_address),
                timestamp=datetime.now(get_facility_timezone()),
                description=f"Mock channel: {channel_address}",
            ),
        )

    async def write_channel(
        self,
        channel_address: str,
        value: Any,
        timeout: float | None = None,
        confirm: bool | None = None,
    ) -> ChannelWriteResult:
        """
        Write a value to a channel, confirming it unless asked not to.

        The connector automatically:
        1. Validates limits (min/max/step/writable) if limits checking enabled
        2. Resolves the confirmation policy for the channel when ``confirm`` is None
        3. Puts the value, then re-reads the channel that was written and compares

        Args:
            channel_address: Any channel name
            value: Value to write
            timeout: Ignored for mock connector
            confirm: Whether to re-read the channel and compare, or ``None`` to
                resolve the policy for this channel from the limits database

        Returns:
            ChannelWriteResult carrying the outcome and what the channel was seen
            to hold. The mock has no alarm metadata to report, so the alarm
            fields stay ``None``.

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

        # Step 2: Resolve the confirmation policy for this channel.
        if confirm is None:
            confirm = self._resolve_confirm(channel_address)

        # Step 3: Put the value into the simulated control system.
        # Simulate network delay
        await asyncio.sleep(self._response_delay)

        try:
            self._put(channel_address, value)
        except Exception as e:
            logger.warning(f"Mock write failed for {channel_address}: {e}")
            return ChannelWriteResult(
                channel_address=channel_address,
                value_written=value,
                outcome=WriteOutcome.FAILED,
                error_message=f"Mock write failed: {e}",
            )

        if not confirm:
            # Fast path by contract: nothing is read, so the result carries no
            # observed value — the same reasoning as the EPICS connector.
            logger.debug(f"Mock write (unconfirmed by policy): {channel_address} = {value}")
            return ChannelWriteResult(
                channel_address=channel_address,
                value_written=value,
                outcome=WriteOutcome.UNREQUESTED,
                notes="Confirmation not requested (mock)",
            )

        # Step 4: Confirm by re-reading the channel that was written.
        try:
            observed = await self._confirming_read(channel_address)
        except Exception as e:
            logger.warning(f"Mock confirming read failed for {channel_address}: {e}")
            return ChannelWriteResult(
                channel_address=channel_address,
                value_written=value,
                outcome=WriteOutcome.UNCONFIRMED,
                error_message=f"Mock confirming read failed: {e}",
                notes=f"Confirming read raised: {e} (mock)",
            )

        # The mock reports no enum label, so the comparison is the ordinary one.
        if values_match(value, observed.value):
            logger.debug(f"Mock write confirmed: {channel_address} = {observed.value}")
            return ChannelWriteResult(
                channel_address=channel_address,
                value_written=value,
                outcome=WriteOutcome.CONFIRMED,
                observed_value=observed.value,
                notes=f"Observed {observed.value}, sent {value} (mock)",
            )

        logger.warning(
            f"Mock write mismatch: {channel_address} sent {value}, observed {observed.value}"
        )
        return ChannelWriteResult(
            channel_address=channel_address,
            value_written=value,
            outcome=WriteOutcome.MISMATCH,
            observed_value=observed.value,
            notes=f"Observed {observed.value}, sent {value} (mock)",
        )

    def _put(self, channel_address: str, value: Any) -> None:
        """Store ``value`` in the simulated control system.

        Raises whatever the store raises — a value the mock cannot hold is a
        write the control system did not take.
        """
        if engine_serves(self._sim_engine, channel_address):
            # Engine channels: :SP -> :RB mirroring is handled by expr readbacks
            # in the machine file, so no string-replace mirroring is needed here.
            self._sim_engine.write(channel_address, value)
            return

        self._state[channel_address] = float(value)

        # Update corresponding readback channel (simulate small offset)
        readback_ch = channel_address.replace(":SP", ":RB").replace(":SET", ":GET")
        if readback_ch != channel_address:
            # Simulate small offset between setpoint and readback
            offset = np.random.normal(0, abs(float(value)) * 0.001)
            self._state[readback_ch] = float(value) + offset

    async def read_multiple_channels(
        self, channel_addresses: list[str], timeout: float | None = None
    ) -> dict[str, ChannelValue]:
        """Read multiple channels concurrently."""
        tasks = [self.read_channel(ch) for ch in channel_addresses]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            ch: result
            for ch, result in zip(channel_addresses, results, strict=True)
            if not isinstance(result, Exception)
        }

    async def subscribe(
        self, channel_address: str, callback: Callable[[ChannelValue], None]
    ) -> str:
        """
        Subscribe to channel changes.

        Note: Mock connector only triggers callbacks on write_channel calls.
        """
        sub_id = f"mock_{channel_address}_{id(callback)}"
        self._subscriptions[sub_id] = (channel_address, callback)
        logger.debug(f"Mock subscription created: {sub_id}")
        return sub_id

    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from channel changes."""
        if subscription_id in self._subscriptions:
            del self._subscriptions[subscription_id]
            logger.debug(f"Mock subscription removed: {subscription_id}")

    async def get_metadata(self, channel_address: str) -> ChannelMetadata:
        """Get channel metadata (from the simulation engine when available)."""
        if engine_serves(self._sim_engine, channel_address):
            reading = self._sim_engine.read(channel_address)
            return ChannelMetadata(
                units=reading.units,
                description=reading.description,
                timestamp=datetime.now(get_facility_timezone()),
            )
        return ChannelMetadata(
            units=self._infer_units(channel_address),
            description=f"Mock channel: {channel_address}",
            timestamp=datetime.now(get_facility_timezone()),
        )

    async def validate_channel(self, channel_address: str) -> bool:
        """All channel names are valid in mock mode."""
        return True

    def _generate_initial_value(self, channel_name: str) -> float:
        """Generate a realistic initial value from the shared channel taxonomy."""
        return classify_channel(channel_name).base_value

    def _infer_units(self, channel_name: str) -> str:
        """Infer units from the shared channel taxonomy."""
        return classify_channel(channel_name).units
