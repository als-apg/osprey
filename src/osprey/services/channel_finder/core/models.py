"""
Core Data Structures for Channel Finder

Defines all Pydantic models used throughout the multi-stage processing pipeline.
These models are shared across all pipeline implementations.
"""

from pydantic import BaseModel, Field


# Stage 1: Query Splitting
class QuerySplitterOutput(BaseModel):
    """Output model for query splitting stage."""

    queries: list[str] = Field(
        description="List of atomic queries, each requesting a single channel or group"
    )


# Stage 2: Channel Matching (In-Context Pipeline)
class ChannelMatchOutput(BaseModel):
    """Output model for channel matching stage."""

    channels_found: bool = Field(description="True if any matching channels were found")
    channels: list[str] = Field(description="List of channel names that match the query")


# Stage 3: Validation and Correction (In-Context Pipeline)
class ChannelValidationEntry(BaseModel):
    """Used internally to track validation status during correction."""

    channel: str = Field(description="Channel name")
    valid: bool = Field(description="Whether channel exists in database")


class ChannelCorrectionOutput(BaseModel):
    """Output model for channel correction stage."""

    corrected_channels: list[str] = Field(description="Full corrected list of valid channels only")


# Stage 0: Explicit Channel Detection (Optimization - shared by all pipelines)
class ExplicitChannelDetectionOutput(BaseModel):
    """
    Output model for detecting explicit channel addresses in queries.

    Used to identify when users provide specific PV/channel addresses directly
    in their query, allowing pipelines to skip search/navigation for efficiency.
    """

    has_explicit_addresses: bool = Field(
        description="True if the query contains explicit channel/PV addresses"
    )
    channel_addresses: list[str] = Field(
        default_factory=list,
        description="List of explicit channel addresses found in the query (empty if none)",
    )
    needs_additional_search: bool = Field(
        description="True if channel finding pipeline should also be invoked (query contains search terms beyond explicit addresses)"
    )
    reasoning: str = Field(
        description="Brief explanation of what was detected and why additional search is/isn't needed"
    )


# Final Output (Shared by all pipelines)
class ChannelInfo(BaseModel):
    """Information about a single channel."""

    channel: str = Field(description="Channel name")
    address: str = Field(description="Channel address")
    description: str | None = Field(default=None, description="Channel description if available")


class ChannelFinderResult(BaseModel):
    """Final result from the channel finder pipeline."""

    query: str = Field(description="Original user query")
    channels: list[ChannelInfo] = Field(description="Found channels with addresses")
    total_channels: int = Field(description="Total number of unique channels found")
    processing_notes: str = Field(description="Notes about query processing and results")
