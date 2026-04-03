"""
Core Data Structures for Channel Finder

Defines Pydantic models for channel finder results.
"""

from typing import Any

from pydantic import BaseModel, Field


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
    selections_paths: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Distinct hierarchy selections paths that produced channels",
    )
