"""
Abstract base class for all channel finder pipelines.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

from .models import ChannelFinderResult, ChannelInfo, ExplicitChannelDetectionOutput

logger = logging.getLogger(__name__)


class BasePipeline(ABC):
    """Abstract base class for all channel finder pipelines."""

    def __init__(self, database, model_config: dict, **kwargs):
        """
        Initialize pipeline.

        Args:
            database: Database instance (specific to pipeline type)
            model_config: LLM model configuration
            **kwargs: Pipeline-specific configuration
        """
        self.database = database
        self.model_config = model_config

        # Get explicit channel validation mode from config
        from osprey.utils.config import get_config_value

        self.explicit_validation_mode = get_config_value(
            "channel_finder.explicit_validation_mode",
            "lenient",  # Default to lenient mode
        )

    @abstractmethod
    async def process_query(self, query: str) -> ChannelFinderResult:
        """
        Process a natural language query and return matching channels.

        Args:
            query: Natural language query string

        Returns:
            ChannelFinderResult with found channels and metadata
        """
        pass

    @abstractmethod
    def get_statistics(self) -> dict[str, Any]:
        """
        Return pipeline statistics.

        Returns:
            Dict with pipeline-specific statistics
        """
        pass

    @property
    @abstractmethod
    def pipeline_name(self) -> str:
        """Return the pipeline name."""
        pass

    async def _detect_explicit_channels(self, query: str) -> ExplicitChannelDetectionOutput:
        """
        Detect if the query contains explicit channel/PV addresses.

        This is a shared optimization across all pipelines. Uses LLM to identify
        when users provide specific channel addresses directly, allowing pipelines
        to skip search/navigation for efficiency.

        Args:
            query: User's natural language query

        Returns:
            ExplicitChannelDetectionOutput with detected addresses and search decision
        """
        # Import here to avoid circular dependency
        from ..llm import get_chat_completion
        from ..prompts import explicit_detection

        # Get prompt from prompts module
        prompt = explicit_detection.get_prompt(query)

        # Set caller context for API call logging
        from osprey.models import set_api_call_context

        set_api_call_context(
            function="_detect_explicit_channels",
            module="base_pipeline",
            class_name="BasePipeline",
            extra={"stage": "explicit_detection"},
        )

        # Get LLM response
        response = await asyncio.to_thread(
            get_chat_completion,
            message=prompt,
            model_config=self.model_config,
            output_model=ExplicitChannelDetectionOutput,
        )

        return response

    def _validate_explicit_channels(
        self, channel_addresses: list[str]
    ) -> tuple[list[str], list[str]]:
        """
        Validate explicit channel addresses based on configured validation mode.

        Validation modes (channel_finder.explicit_validation_mode):
        - 'strict': Only accept channels that exist in database
        - 'lenient': Accept all explicit channels, warn if not in database (default)
        - 'skip': Accept all without validation (fastest, no database lookups)

        Args:
            channel_addresses: List of explicit channel addresses to validate

        Returns:
            Tuple of (valid_channels, invalid_channels)
            - In 'lenient' or 'skip' mode, all channels go to valid_channels
            - In 'strict' mode, only database channels go to valid_channels
        """
        if self.explicit_validation_mode == "skip":
            # Skip validation entirely - trust all explicit addresses
            logger.info(
                f"  → Validation mode: skip (trusting all {len(channel_addresses)} explicit addresses)"
            )
            return (channel_addresses, [])

        # Validate against database
        valid_channels = []
        invalid_channels = []

        for address in channel_addresses:
            if self.database.validate_channel(address):
                valid_channels.append(address)
                logger.info(f"  ✓ Validated: {address}")
            else:
                invalid_channels.append(address)
                if self.explicit_validation_mode == "lenient":
                    # Lenient mode: include anyway but warn
                    logger.warning(
                        f"  ⚠  {address} not in database, but including anyway (lenient mode)"
                    )
                else:
                    # Strict mode: reject
                    logger.warning(f"  ✗ Not found in database: {address}")

        # In lenient mode, include all channels (valid + invalid)
        if self.explicit_validation_mode == "lenient":
            all_channels = valid_channels + invalid_channels
            return (all_channels, [])
        else:
            # Strict mode: only return validated channels
            return (valid_channels, invalid_channels)

    def _build_result(self, query: str, channels: list[str]) -> ChannelFinderResult:
        """
        Build final result object from channel names.

        Shared helper method across all pipelines to construct the final
        ChannelFinderResult with channel info and metadata.

        Args:
            query: Original user query
            channels: List of channel names/addresses found

        Returns:
            ChannelFinderResult with channel info
        """
        channel_infos = []

        for channel_name in channels:
            channel_data = self.database.get_channel(channel_name)
            if channel_data:
                channel_infos.append(
                    ChannelInfo(
                        channel=channel_name,
                        address=channel_data.get("address", channel_name),
                        description=channel_data.get("description"),
                    )
                )
            else:
                # Channel not in database - use as-is (can happen with explicit detection)
                channel_infos.append(
                    ChannelInfo(
                        channel=channel_name,
                        address=channel_name,
                        description=None,
                    )
                )

        notes = f"Found {len(channel_infos)} channel(s)"

        return ChannelFinderResult(
            query=query,
            channels=channel_infos,
            total_channels=len(channel_infos),
            processing_notes=notes,
        )
