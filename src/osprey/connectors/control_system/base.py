"""
Abstract base class for control system connectors.

Provides protocol-agnostic interfaces for reading/writing process variables,
subscribing to changes, and retrieving metadata from various control systems.

Related to Issue #18 - Control System Abstraction (Layer 2)
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class PVMetadata:
    """Metadata about a process variable/attribute."""

    units: str = ""
    precision: int | None = None
    alarm_status: str | None = None
    timestamp: datetime | None = None
    description: str | None = None
    min_value: float | None = None
    max_value: float | None = None
    raw_metadata: dict[str, Any] | None = field(default_factory=dict)

    def __post_init__(self):
        """Ensure raw_metadata is a dict."""
        if self.raw_metadata is None:
            self.raw_metadata = {}


@dataclass
class PVValue:
    """Value of a process variable with metadata."""

    value: Any
    timestamp: datetime
    metadata: PVMetadata = field(default_factory=PVMetadata)


class ControlSystemConnector(ABC):
    """
    Abstract base class for control system connectors.

    Implementations provide interfaces to different control systems
    (EPICS, LabVIEW, Tango, Mock, etc.) using a unified API.

    Example:
        >>> connector = await ConnectorFactory.create_control_system_connector()
        >>> try:
        >>>     pv_value = await connector.read_pv('BEAM:CURRENT')
        >>>     print(f"Beam current: {pv_value.value} {pv_value.metadata.units}")
        >>> finally:
        >>>     await connector.disconnect()
    """

    @abstractmethod
    async def connect(self, config: dict[str, Any]) -> None:
        """
        Establish connection to control system.

        Args:
            config: Control system-specific configuration

        Raises:
            ConnectionError: If connection cannot be established
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to control system and cleanup resources."""
        pass

    @abstractmethod
    async def read_pv(
        self,
        pv_address: str,
        timeout: float | None = None
    ) -> PVValue:
        """
        Read current value of a process variable.

        Args:
            pv_address: Address/name of the process variable
            timeout: Optional timeout in seconds

        Returns:
            PVValue with current value, timestamp, and metadata

        Raises:
            ConnectionError: If PV cannot be reached
            TimeoutError: If operation times out
            ValueError: If PV address is invalid
        """
        pass

    @abstractmethod
    async def write_pv(
        self,
        pv_address: str,
        value: Any,
        timeout: float | None = None
    ) -> bool:
        """
        Write value to a process variable.

        Args:
            pv_address: Address/name of the process variable
            value: Value to write
            timeout: Optional timeout in seconds

        Returns:
            True if write was successful

        Raises:
            ConnectionError: If PV cannot be reached
            TimeoutError: If operation times out
            ValueError: If value is invalid for this PV
            PermissionError: If write access is not allowed
        """
        pass

    @abstractmethod
    async def read_multiple_pvs(
        self,
        pv_addresses: list[str],
        timeout: float | None = None
    ) -> dict[str, PVValue]:
        """
        Read multiple PVs efficiently (can be optimized per control system).

        Args:
            pv_addresses: List of PV addresses to read
            timeout: Optional timeout in seconds

        Returns:
            Dictionary mapping PV address to PVValue
            (May exclude PVs that failed to read)
        """
        pass

    @abstractmethod
    async def subscribe(
        self,
        pv_address: str,
        callback: Callable[[PVValue], None]
    ) -> str:
        """
        Subscribe to PV changes.

        Args:
            pv_address: Address/name of the process variable
            callback: Function to call when PV value changes

        Returns:
            Subscription ID for later unsubscription

        Raises:
            ConnectionError: If PV cannot be reached
        """
        pass

    @abstractmethod
    async def unsubscribe(self, subscription_id: str) -> None:
        """
        Unsubscribe from PV changes.

        Args:
            subscription_id: Subscription ID returned by subscribe()
        """
        pass

    @abstractmethod
    async def get_metadata(self, pv_address: str) -> PVMetadata:
        """
        Get metadata about a PV.

        Args:
            pv_address: Address/name of the process variable

        Returns:
            PVMetadata with units, limits, description, etc.

        Raises:
            ConnectionError: If PV cannot be reached
        """
        pass

    @abstractmethod
    async def validate_pv(self, pv_address: str) -> bool:
        """
        Check if PV exists and is accessible.

        Args:
            pv_address: Address/name of the process variable

        Returns:
            True if PV is valid and accessible
        """
        pass

