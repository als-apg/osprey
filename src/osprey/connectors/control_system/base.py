"""
Abstract base class for control system connectors.

Provides protocol-agnostic interfaces for reading/writing process variables,
subscribing to changes, and retrieving metadata from various control systems.

"""

import functools
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger("osprey.connectors.control_system")


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

    _limits_validator: Any = None  # Initialized by subclasses in connect()

    @property
    def _writes_enabled(self) -> bool:
        """Check whether writes are enabled via global config.

        Returns False (fail-safe) when config is unavailable.
        """
        try:
            from osprey.utils.config import get_config_value

            return get_config_value("control_system.writes_enabled", False)
        except (FileNotFoundError, RuntimeError):
            return False

    def __init_subclass__(cls, **kwargs):
        """Auto-wrap write_channel() with writes_enabled pre-check.

        Any subclass that defines write_channel() gets it transparently wrapped.
        The wrapper checks _writes_enabled before calling the original method.
        When writes are disabled, returns ChannelWriteResult(success=False)
        with an operator-facing error message — no exception is raised.

        This fires before limits validation (intentional: fast-reject when
        writes are disabled, avoiding unnecessary validation work).
        """
        super().__init_subclass__(**kwargs)
        original = cls.__dict__.get("write_channel")
        if original is None:
            return

        @functools.wraps(original)
        async def _guarded(self, channel_address, value, *args, **kwargs):
            if not self._writes_enabled:
                return ChannelWriteResult(
                    channel_address=channel_address,
                    value_written=value,
                    success=False,
                    error_message=(
                        f"Write to '{channel_address}' blocked: writes are disabled. "
                        "Set control_system.writes_enabled: true in config.yml"
                    ),
                )
            return await original(self, channel_address, value, *args, **kwargs)

        cls.write_channel = _guarded

    def _get_verification_config(
        self, channel_address: str, value: float
    ) -> tuple[str, float | None]:
        """Get verification level and tolerance for a channel write.

        Priority:
        1. Per-channel config from limits database
        2. Global config from config.yml
        3. Fallback: callback with no tolerance

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

    async def write_multiple_channels(
        self,
        operations: list[tuple[str, Any]],
        timeout: float | None = None,
        verification_level: str = "callback",
        tolerance: float | None = None,
    ) -> list[ChannelWriteResult]:
        """
        Write multiple channels. Override for atomic/batched behavior.

        Default implementation writes sequentially via write_channel().
        Subclasses can override to provide transactional semantics (e.g.,
        disabling lattice recalculation between writes in a simulator).

        Args:
            operations: List of (channel_address, value) tuples
            timeout: Optional timeout in seconds
            verification_level: Verification strategy ("none", "callback", "readback")
            tolerance: Absolute tolerance for readback verification

        Returns:
            List of ChannelWriteResult in the same order as operations
        """
        results = []
        for address, value in operations:
            result = await self.write_channel(
                address,
                value,
                timeout=timeout,
                verification_level=verification_level,
                tolerance=tolerance,
            )
            results.append(result)
        return results

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
