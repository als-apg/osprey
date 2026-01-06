"""
Abstract base class for control system connectors.

Provides protocol-agnostic interfaces for reading/writing process variables,
subscribing to changes, and retrieving metadata from various control systems.

Related to Issue #18 - Control System Abstraction (Layer 2)
"""

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ChannelMetadata:
    """Metadata about a control system channel."""

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
class ChannelValue:
    """Value of a control system channel with metadata."""

    value: Any
    timestamp: datetime
    metadata: ChannelMetadata = field(default_factory=ChannelMetadata)


@dataclass
class WriteVerification:
    """
    Verification result from a channel write operation.

    Different control systems provide different levels of verification:
    - "none": No verification performed (fast write)
    - "callback": Control system confirmed request processing (e.g., EPICS IOC callback)
    - "readback": Full verification with readback comparison
    """

    level: str  # "none", "callback", "readback"
    verified: bool  # Whether verification succeeded
    readback_value: float | None = None  # Actual value read back (for "readback" level)
    tolerance_used: float | None = None  # Tolerance used for comparison (for "readback" level)
    notes: str | None = None  # Additional verification details


@dataclass
class ChannelWriteResult:
    """
    Result from a channel write operation with optional verification.

    This is the control-system-agnostic result type returned by all connectors.
    Provides detailed information about write success and verification status.
    """

    channel_address: str  # Channel that was written
    value_written: Any  # Value that was written
    success: bool  # Whether the write command succeeded
    verification: WriteVerification | None = None  # Verification details (if performed)
    error_message: str | None = None  # Error message if write failed


class ControlSystemConnector(ABC):
    """
    Abstract base class for control system connectors.

    Implementations provide interfaces to different control systems
    (EPICS, LabVIEW, Tango, Mock, etc.) using a unified API.

    Example:
        >>> connector = await ConnectorFactory.create_control_system_connector()
        >>> try:
        >>>     channel_value = await connector.read_channel('BEAM:CURRENT')
        >>>     print(f"Beam current: {channel_value.value} {channel_value.metadata.units}")
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
    async def read_channel(
        self, channel_address: str, timeout: float | None = None
    ) -> ChannelValue:
        """
        Read current value of a channel.

        Args:
            channel_address: Address/name of the channel
            timeout: Optional timeout in seconds

        Returns:
            ChannelValue with current value, timestamp, and metadata

        Raises:
            ConnectionError: If channel cannot be reached
            TimeoutError: If operation times out
            ValueError: If channel address is invalid
        """
        pass

    @abstractmethod
    async def write_channel(
        self,
        channel_address: str,
        value: Any,
        timeout: float | None = None,
        verification_level: str = "callback",
        tolerance: float | None = None,
    ) -> ChannelWriteResult:
        """
        Write value to a channel with configurable verification.

        Args:
            channel_address: Address/name of the channel
            value: Value to write
            timeout: Optional timeout in seconds
            verification_level: Verification strategy ("none", "callback", "readback")
            tolerance: Absolute tolerance for readback verification (only used if verification_level="readback")

        Returns:
            ChannelWriteResult with write status and verification details

        Raises:
            ConnectionError: If channel cannot be reached
            TimeoutError: If operation times out
            ValueError: If value is invalid for this channel
            PermissionError: If write access is not allowed

        Note:
            The verification_level determines what confirmation is provided:
            - "none": Fast write, no verification (success=True if command sent)
            - "callback": Control system confirms processing (e.g., EPICS IOC callback)
            - "readback": Full verification with readback value comparison

            Different control systems may interpret these levels differently based on
            their native capabilities.
        """
        pass

    @abstractmethod
    async def read_multiple_channels(
        self, channel_addresses: list[str], timeout: float | None = None
    ) -> dict[str, ChannelValue]:
        """
        Read multiple channels efficiently (can be optimized per control system).

        Args:
            channel_addresses: List of channel addresses to read
            timeout: Optional timeout in seconds

        Returns:
            Dictionary mapping channel address to ChannelValue
            (May exclude channels that failed to read)
        """
        pass

    @abstractmethod
    async def subscribe(
        self, channel_address: str, callback: Callable[[ChannelValue], None]
    ) -> str:
        """
        Subscribe to channel changes.

        Args:
            channel_address: Address/name of the channel
            callback: Function called when value changes (receives ChannelValue)

        Returns:
            Subscription ID for later unsubscribe
        """
        pass

    @abstractmethod
    async def unsubscribe(self, subscription_id: str) -> None:
        """
        Cancel subscription to channel changes.

        Args:
            subscription_id: Subscription ID returned by subscribe()
        """
        pass

    @abstractmethod
    async def get_metadata(self, channel_address: str) -> ChannelMetadata:
        """
        Get metadata about a channel.

        Args:
            channel_address: Address/name of the channel

        Returns:
            ChannelMetadata with units, limits, description, etc.

        Raises:
            ConnectionError: If channel cannot be reached
        """
        pass

    @abstractmethod
    async def validate_channel(self, channel_address: str) -> bool:
        """
        Check if channel exists and is accessible.

        Args:
            channel_address: Address/name of the channel

        Returns:
            True if channel is valid and accessible
        """
        pass

    # Deprecated method aliases for backward compatibility
    async def read_pv(self, pv_address: str, timeout: float | None = None) -> ChannelValue:
        """
        Read current value of a PV/channel.

        .. deprecated:: 0.9.5
           Use :meth:`read_channel` instead. The term "PV" is EPICS-specific;
           "channel" is control-system agnostic.
        """
        warnings.warn(
            "read_pv() is deprecated and will be removed in v0.10. "
            "Use read_channel() instead. The term 'PV' is EPICS-specific; "
            "'channel' is control-system agnostic.",
            DeprecationWarning,
            stacklevel=2,
        )
        return await self.read_channel(pv_address, timeout)

    async def write_pv(
        self,
        pv_address: str,
        value: Any,
        timeout: float | None = None,
        verification_level: str = "callback",
        tolerance: float | None = None,
    ) -> ChannelWriteResult:
        """
        Write value to a PV/channel.

        .. deprecated:: 0.9.5
           Use :meth:`write_channel` instead. The term "PV" is EPICS-specific;
           "channel" is control-system agnostic.
        """
        warnings.warn(
            "write_pv() is deprecated and will be removed in v0.10. "
            "Use write_channel() instead. The term 'PV' is EPICS-specific; "
            "'channel' is control-system agnostic.",
            DeprecationWarning,
            stacklevel=2,
        )
        return await self.write_channel(pv_address, value, timeout, verification_level, tolerance)

    async def read_multiple_pvs(
        self, pv_addresses: list[str], timeout: float | None = None
    ) -> dict[str, ChannelValue]:
        """
        Read multiple PVs/channels.

        .. deprecated:: 0.9.5
           Use :meth:`read_multiple_channels` instead.
        """
        warnings.warn(
            "read_multiple_pvs() is deprecated and will be removed in v0.10. "
            "Use read_multiple_channels() instead. The term 'PV' is EPICS-specific; "
            "'channel' is control-system agnostic.",
            DeprecationWarning,
            stacklevel=2,
        )
        return await self.read_multiple_channels(pv_addresses, timeout)

    async def validate_pv(self, pv_address: str) -> bool:
        """
        Check if PV/channel exists and is accessible.

        .. deprecated:: 0.9.5
           Use :meth:`validate_channel` instead.
        """
        warnings.warn(
            "validate_pv() is deprecated and will be removed in v0.10. "
            "Use validate_channel() instead. The term 'PV' is EPICS-specific; "
            "'channel' is control-system agnostic.",
            DeprecationWarning,
            stacklevel=2,
        )
        return await self.validate_channel(pv_address)


# Backward compatibility wrappers with deprecation warnings
# Deprecated in 0.9.5, will be removed in a future version


class PVMetadata(ChannelMetadata):
    """
    Deprecated alias for ChannelMetadata.

    .. deprecated:: 0.9.5
        Use :class:`ChannelMetadata` instead. The term "PV" (Process Variable) is
        EPICS-specific; "channel" is control-system agnostic and supports any control
        system (EPICS, Tango, LabVIEW, etc.).
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "PVMetadata is deprecated and will be removed in v0.10. "
            "Use ChannelMetadata instead. The term 'PV' is EPICS-specific; "
            "'channel' is control-system agnostic.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


class PVValue(ChannelValue):
    """
    Deprecated alias for ChannelValue.

    .. deprecated:: 0.9.5
        Use :class:`ChannelValue` instead. The term "PV" (Process Variable) is
        EPICS-specific; "channel" is control-system agnostic and supports any control
        system (EPICS, Tango, LabVIEW, etc.).
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "PVValue is deprecated and will be removed in v0.10. "
            "Use ChannelValue instead. The term 'PV' is EPICS-specific; "
            "'channel' is control-system agnostic.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
