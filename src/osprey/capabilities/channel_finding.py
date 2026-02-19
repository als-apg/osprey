"""
Channel Finding Capability

This capability searches for control system channel addresses based on user descriptions.
It helps users find the correct channel names when they know what they want to measure
but don't know the exact channel address.

Based on ALS Assistant's PV Address Finding capability pattern.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from osprey.base.capability import BaseCapability
from osprey.base.decorators import capability_node
from osprey.base.errors import ErrorClassification, ErrorSeverity
from osprey.base.examples import (
    OrchestratorGuide,
    TaskClassifierGuide,
)
from osprey.context import CapabilityContext
from osprey.services.channel_finder.service import ChannelFinderService

# ========================================================
# Context Class
# ========================================================


class ChannelAddressesContext(CapabilityContext):
    """
    Framework context for channel finding capability results.

    This is the rich context object used throughout the framework for channel address data.
    Based on ALS Assistant's ChannelAddresses pattern.
    """

    CONTEXT_TYPE: ClassVar[str] = "CHANNEL_ADDRESSES"
    CONTEXT_CATEGORY: ClassVar[str] = "METADATA"

    channels: list[str]  # List of found channel addresses
    original_query: str  # Original natural language query that led to these channels

    def get_access_details(self, key: str) -> dict[str, Any]:
        """Rich description for LLM consumption."""
        return {
            "channels": self.channels,
            "total_available": len(self.channels),
            "original_query": self.original_query,
            "data_structure": "List of channel address strings",
            "access_pattern": f"context.{self.CONTEXT_TYPE}.{key}.channels",
            "example_usage": f"context.{self.CONTEXT_TYPE}.{key}.channels[0] gives '{self.channels[0] if self.channels else 'CHANNEL:NAME'}'",
        }

    def get_summary(self) -> dict[str, Any]:
        """
        FOR HUMAN DISPLAY: Create readable summary for UI/debugging.
        Always customize for better user experience.
        """
        return {
            "type": "Channel Addresses",
            "total_channels": len(self.channels),
            "original_query": self.original_query,
            "channel_list": self.channels,
        }


# Available pipelines based on template configuration
AVAILABLE_PIPELINES = ["in_context", "hierarchical", "middle_layer"]
DEFAULT_PIPELINE = None  # Read from config at runtime


# ===================================================================
# Private Helper Functions
# ===================================================================


def _configure_service_logging(logger, package_name: str) -> None:
    """Configure channel finder service to inherit framework log level."""
    cf_root_logger = logging.getLogger(f"{package_name}.services.channel_finder")
    current_level = logger.level if logger.level != 0 else logging.INFO
    cf_root_logger.setLevel(current_level)
    cf_root_logger.propagate = True


# ===================================================================
# Channel Finding Errors
# ===================================================================


class ChannelFindingError(Exception):
    """Base class for all channel finding errors."""

    pass


class ChannelNotFoundError(ChannelFindingError):
    """Raised when no channel addresses can be found for the given query."""

    pass


class ChannelFinderServiceError(ChannelFindingError):
    """Raised when the channel finder service is unavailable or fails."""

    pass


# ===================================================================
# Capability Implementation
# ===================================================================


@capability_node
class ChannelFindingCapability(BaseCapability):
    """
    Channel finding capability for resolving descriptions to channel addresses.

    Supports multiple pipeline modes (in_context, hierarchical, middle_layer)
    configurable via config.yml. Switch between modes based on your control
    system size and requirements.
    """

    name = "channel_finding"
    description = "Find control system channel addresses based on descriptions or search terms"
    provides = ["CHANNEL_ADDRESSES"]
    requires = []  # No dependencies - extracts from task objective

    async def execute(self) -> dict[str, Any]:
        """
        Find control system channel addresses based on search query.

        This method uses the configured channel finder service to search the
        channel database and return matching control system addresses. The search
        query is extracted from the task objective.

        Returns:
            State updates containing CHANNEL_ADDRESSES context with found channels.

        Raises:
            ChannelNotFoundError: If no matching channels are found.
            ChannelFinderServiceError: If the search service fails.
        """
        # Extract search query from task objective
        search_query = self.get_task_objective(default="unknown")

        # Get unified logger with automatic streaming
        logger = self.get_logger()

        # Log the query
        logger.info(f'Channel finding query: "{search_query}"')
        logger.status("Finding channel addresses...")

        # Configure service logging to show pipeline details
        _configure_service_logging(logger, "osprey")

        try:
            # Initialize service (reads pipeline_mode from config.yml at runtime)
            service = ChannelFinderService()

            logger.status("Searching channel database...")

            # Execute channel finding
            result = await service.find_channels(search_query)

        except Exception as e:
            error_msg = f"Channel finder service failed for query '{search_query}': {str(e)}"
            logger.error(error_msg)
            raise ChannelFinderServiceError(error_msg) from e

        # Log results
        # IMPORTANT: We extract the ADDRESS field, not the channel name
        # The database contains both:
        #   - 'channel': descriptive/user-friendly name (e.g., "BeamCurrent")
        #   - 'address': actual control system address (e.g., "BEAM:CURRENT")
        # We use the ADDRESS for all downstream operations
        if result.total_channels > 0:
            logger.info(
                f"Found {result.total_channels} channel address{'es' if result.total_channels != 1 else ''}"
            )
            # Detailed logging can be enabled via framework log level
            for ch in result.channels:
                logger.debug(f"  {ch.channel} -> {ch.address}")

        logger.status("Creating channel context...")

        # Convert service layer response to framework context
        # CRITICAL: Extract ADDRESSES, not channel names
        # These addresses are what we'll use for actual control system operations
        channel_list = [ch.address for ch in result.channels]

        # Check if no channels were found and raise appropriate error for re-planning
        if not channel_list:
            error_msg = (
                f"No channel addresses found for query: '{search_query}'. {result.processing_notes}"
            )
            logger.warning(f"Channel address not found: {error_msg}")
            raise ChannelNotFoundError(error_msg)

        # Create framework context object with original query for semantic linking
        # This allows downstream capabilities (like channel_write) to match
        # natural language requests back to the correct channel addresses
        channel_context = ChannelAddressesContext(
            channels=channel_list,
            original_query=result.query,
        )

        logger.status("Channel finding complete")

        # Store result in execution context
        return self.store_output_context(channel_context)

    @staticmethod
    def classify_error(exc: Exception, context: dict) -> ErrorClassification:
        """Channel finding error classification with defensive approach."""

        if isinstance(exc, ChannelNotFoundError):
            return ErrorClassification(
                severity=ErrorSeverity.REPLANNING,
                user_message=f"No channel addresses found: {str(exc)}",
                metadata={
                    "technical_details": str(exc),
                    "replanning_reason": f"Channel search failed to find matches: {exc}",
                    "suggestions": [
                        "Try refining the search terms to be more specific",
                        "Check if the requested channels exist in the system",
                        "Consider using different keywords or system names",
                    ],
                },
            )

        elif isinstance(exc, ChannelFinderServiceError):
            return ErrorClassification(
                severity=ErrorSeverity.CRITICAL,
                user_message=f"Channel finder service unavailable: {str(exc)}",
                metadata={
                    "technical_details": str(exc),
                    "safety_abort_reason": f"Channel finder service failure: {exc}",
                    "suggestions": [
                        "Check if the channel finder service is running and accessible",
                        "Verify channel database is loaded properly",
                        "Contact system administrator if the service appears down",
                    ],
                },
            )

        else:
            # Default: critical for unknown errors (defensive approach)
            return ErrorClassification(
                severity=ErrorSeverity.CRITICAL,
                user_message=f"Unexpected error in channel finding: {exc}",
                metadata={
                    "technical_details": str(exc),
                    "safety_abort_reason": f"Unhandled channel finding error: {exc}",
                },
            )

    def _create_orchestrator_guide(self) -> OrchestratorGuide | None:
        """Delegate orchestrator guide to prompt builder."""
        from osprey.prompts.loader import get_framework_prompts

        builder = get_framework_prompts().get_channel_finding_orchestration_prompt_builder()
        return builder.get_orchestrator_guide()

    def _create_classifier_guide(self) -> TaskClassifierGuide | None:
        """Delegate classifier guide to prompt builder."""
        from osprey.prompts.loader import get_framework_prompts

        builder = get_framework_prompts().get_channel_finding_orchestration_prompt_builder()
        return builder.get_classifier_guide()
