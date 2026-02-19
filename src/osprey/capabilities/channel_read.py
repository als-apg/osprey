"""
Channel Read Capability

This capability handles reading current values of control system channels.
It provides real-time access to live channel data.

Based on ALS Assistant's channel value retrieval capability pattern.

Configuration:
    The control system connector is configured in config.yml. By default, the template
    uses the mock connector which works with any channel names.

    Development Mode (config.yml):
        control_system:
            type: mock

    Production Mode (config.yml):
        control_system:
            type: epics
            connector:
                epics:
                    timeout: 5.0
                    gateways:
                        read_only:
                            address: cagw.your-facility.edu
                            port: 5064

    The capability code remains the same - just change the config!
"""

from datetime import datetime
from typing import Any, ClassVar

from pydantic import BaseModel

from osprey.base.capability import BaseCapability
from osprey.base.decorators import capability_node
from osprey.base.errors import ErrorClassification, ErrorSeverity
from osprey.base.examples import (
    OrchestratorGuide,
    TaskClassifierGuide,
)
from osprey.connectors.factory import ConnectorFactory
from osprey.context import CapabilityContext

# ========================================================
# Context Classes
# ========================================================


class ChannelValue(BaseModel):
    """Individual channel value data - simple nested structure for Pydantic."""

    value: str
    timestamp: datetime  # Pydantic handles datetime serialization automatically
    units: str


class ChannelValuesContext(CapabilityContext):
    """
    Result from channel value retrieval operation and context for downstream capabilities.
    Based on ALS Assistant's ChannelValues pattern.
    """

    CONTEXT_TYPE: ClassVar[str] = "CHANNEL_VALUES"
    CONTEXT_CATEGORY: ClassVar[str] = "COMPUTATIONAL_DATA"

    channel_values: dict[str, ChannelValue]  # Clean structure - no DotDict needed

    @property
    def channel_count(self) -> int:
        """Number of channels retrieved."""
        return len(self.channel_values)

    def get_access_details(self, key: str) -> dict[str, Any]:
        """Rich description for LLM consumption."""
        channels_preview = list(self.channel_values.keys())[:3]
        example_channel = channels_preview[0] if channels_preview else "SR:CURRENT:RB"

        # Get example value from the ChannelValue object
        try:
            example_value = (
                self.channel_values[example_channel].value
                if example_channel in self.channel_values
                else "400.5"
            )
        except Exception:
            example_value = "400.5"

        return {
            "channel_count": self.channel_count,
            "channels": channels_preview,
            "data_structure": "Dict[channel_name -> ChannelValue] where ChannelValue has .value, .timestamp, .units fields - IMPORTANT: use bracket notation for channel names (due to special characters like colons), but dot notation for fields",
            "access_pattern": f"context.{self.CONTEXT_TYPE}.{key}.channel_values['CHANNEL_NAME'].value (NOT ['value'])",
            "example_usage": f"context.{self.CONTEXT_TYPE}.{key}.channel_values['{example_channel}'].value gives '{example_value}' (use .value not ['value'])",
            "available_fields": ["value", "timestamp", "units"],
        }

    def get_summary(self) -> dict[str, Any]:
        """
        FOR HUMAN DISPLAY: Create readable summary for UI/debugging.
        Always customize for better user experience.
        """
        channel_data = {}
        for channel_name, channel_info in self.channel_values.items():
            channel_data[channel_name] = {
                "value": channel_info.value,
                "timestamp": channel_info.timestamp,
                "units": channel_info.units,
            }

        return {
            "type": "Channel Values",
            "channel_data": channel_data,
        }


# ========================================================
# Channel-Related Errors
# ========================================================


class ChannelError(Exception):
    """Base class for all channel-related errors."""

    pass


class ChannelNotFoundError(ChannelError):
    """Raised when a channel address doesn't exist in the system."""

    pass


class ChannelTimeoutError(ChannelError):
    """Raised when channel operations time out."""

    pass


class ChannelAccessError(ChannelError):
    """Raised when there are channel access errors."""

    pass


class ChannelDependencyError(ChannelError):
    """Raised when required dependencies are missing."""

    pass


# ========================================================
# Capability Implementation
# ========================================================


@capability_node
class ChannelReadCapability(BaseCapability):
    """Channel read capability for reading current channel values."""

    name = "channel_read"
    description = "Read current values of control system channels"
    provides = ["CHANNEL_VALUES"]
    requires = ["CHANNEL_ADDRESSES"]

    async def execute(self) -> dict[str, Any]:
        """
        Execute channel read operation.

        This method reads current values from control system channels using the
        configured connector (mock or EPICS). It automatically extracts required
        channel addresses from the execution context and returns structured results.

        Returns:
            State updates containing the CHANNEL_VALUES context with current readings.

        Raises:
            ChannelDependencyError: If required CHANNEL_ADDRESSES context is missing.
            ChannelAccessError: If channel reading fails.
        """

        # Get unified logger with automatic streaming
        logger = self.get_logger()
        logger.status("Starting channel read operation...")

        # Extract channel addresses from execution context
        try:
            (channels_to_read,) = self.get_required_contexts()
        except ValueError as e:
            raise ChannelDependencyError(str(e)) from e

        logger.status(f"Reading {len(channels_to_read)} channel values...")

        # Create control system connector from configuration
        # This will use 'mock' for development or 'epics' for production
        # based on the 'control_system' section in config.yml
        connector = await ConnectorFactory.create_control_system_connector()

        try:
            # Get all channel values
            channel_values = {}
            total_channels = len(channels_to_read)

            for i, channel_address in enumerate(channels_to_read):
                logger.status(f"Reading channel {i + 1}/{total_channels}: {channel_address}")

                try:
                    # Read channel using connector (returns ChannelValue with metadata)
                    channel_result = await connector.read_channel(channel_address)

                    # Convert to ChannelValue format expected by context
                    channel_values[channel_address] = ChannelValue(
                        value=str(channel_result.value),
                        timestamp=channel_result.timestamp,
                        units=channel_result.metadata.units if channel_result.metadata else "",
                    )
                except Exception as e:
                    logger.error(f"Failed to read channel {channel_address}: {e}")
                    # Continue with other channels
                    raise ChannelAccessError(
                        f"Failed to read channel {channel_address}: {str(e)}"
                    ) from e

        finally:
            # Always disconnect connector
            await connector.disconnect()

        # Create structured result
        result = ChannelValuesContext(channel_values=channel_values)

        logger.status(f"Successfully read {result.channel_count} channel values")

        logger.info(f"Channel read result: {result.channel_count} channels read")

        # Store result in execution context
        return self.store_output_context(result)

    def process_extracted_contexts(self, contexts):
        """
        Flatten CHANNEL_ADDRESSES contexts into single list.

        Handles both single and multiple CHANNEL_ADDRESSES contexts,
        merging them into a flat list of channel addresses.
        """
        channels_raw = contexts["CHANNEL_ADDRESSES"]
        logger = self.get_logger()

        if isinstance(channels_raw, list):
            # Flatten multiple contexts into single channel list
            channels_flat = []
            for ctx in channels_raw:
                channels_flat.extend(ctx.channels)
            logger.info(
                f"Merged {len(channels_raw)} CHANNEL_ADDRESSES contexts into {len(channels_flat)} total channels"
            )
            contexts["CHANNEL_ADDRESSES"] = channels_flat
        else:
            # Single context - extract channel list
            contexts["CHANNEL_ADDRESSES"] = channels_raw.channels

        return contexts

    @staticmethod
    def classify_error(exc: Exception, context: dict) -> ErrorClassification:
        """Channel-specific error classification."""

        if isinstance(exc, ChannelTimeoutError):
            return ErrorClassification(
                severity=ErrorSeverity.RETRIABLE,
                user_message=f"Channel timeout error: {str(exc)}",
                metadata={"technical_details": str(exc)},
            )
        elif isinstance(exc, ChannelAccessError):
            return ErrorClassification(
                severity=ErrorSeverity.RETRIABLE,
                user_message=f"Channel access error: {str(exc)}",
                metadata={"technical_details": str(exc)},
            )
        elif isinstance(exc, ChannelNotFoundError):
            return ErrorClassification(
                severity=ErrorSeverity.REPLANNING,
                user_message=f"Channel access failed: {str(exc)}",
                metadata={"technical_details": str(exc)},
            )
        elif isinstance(exc, ChannelDependencyError):
            return ErrorClassification(
                severity=ErrorSeverity.REPLANNING,
                user_message=f"Missing dependency: {str(exc)}",
                metadata={"technical_details": str(exc)},
            )
        elif isinstance(exc, ChannelError):
            # Generic channel error
            return ErrorClassification(
                severity=ErrorSeverity.RETRIABLE,
                user_message=f"Channel error: {str(exc)}",
                metadata={"technical_details": str(exc)},
            )
        else:
            # Not a channel-specific error, use default classification
            return ErrorClassification(
                severity=ErrorSeverity.RETRIABLE,
                user_message=f"Unexpected error: {str(exc)}",
                metadata={"technical_details": str(exc)},
            )

    @staticmethod
    def get_retry_policy() -> dict[str, Any]:
        """Define retry policy for channel read operations."""
        return {"max_attempts": 3, "delay_seconds": 1.0, "backoff_factor": 2.0}

    def _create_orchestrator_guide(self) -> OrchestratorGuide | None:
        """Delegate orchestrator guide to prompt builder."""
        from osprey.prompts.loader import get_framework_prompts

        builder = get_framework_prompts().get_channel_read_prompt_builder()
        return builder.get_orchestrator_guide()

    def _create_classifier_guide(self) -> TaskClassifierGuide | None:
        """Delegate classifier guide to prompt builder."""
        from osprey.prompts.loader import get_framework_prompts

        builder = get_framework_prompts().get_channel_read_prompt_builder()
        return builder.get_classifier_guide()
