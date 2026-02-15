"""Generic JSON ingestion adapter.

This module provides a flexible adapter for testing and facilities
without custom APIs.

See 01_DATA_LAYER.md Sections 5.4, 5.10 for specification.
"""

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from osprey.services.ariel_search.exceptions import IngestionError
from osprey.services.ariel_search.ingestion.base import BaseAdapter
from osprey.services.ariel_search.models import AttachmentInfo, EnhancedLogbookEntry
from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from osprey.services.ariel_search.config import ARIELConfig

logger = get_logger("ariel")


class GenericJSONAdapter(BaseAdapter):
    """Generic JSON ingestion adapter.

    Reads entries from JSON files or HTTP endpoints with a flexible schema.
    """

    def __init__(self, config: "ARIELConfig") -> None:
        """Initialize the adapter."""
        super().__init__(config)

        if not config.ingestion or not config.ingestion.source_url:
            raise IngestionError(
                "source_url is required for generic_json adapter",
                source_system=self.source_system_name,
            )

        self.source_url = config.ingestion.source_url

    @property
    def source_system_name(self) -> str:
        """Return the source system identifier."""
        return "Generic JSON"

    async def fetch_entries(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[EnhancedLogbookEntry]:
        """Fetch entries from JSON source.

        Args:
            since: Only fetch entries after this timestamp
            until: Only fetch entries before this timestamp
            limit: Maximum number of entries to fetch

        Yields:
            EnhancedLogbookEntry objects
        """
        data = await self._load_data()
        entries = data.get("entries", [])

        count = 0
        for entry_data in entries:
            try:
                entry = self._convert_entry(entry_data)

                # Apply time filters
                if since and entry["timestamp"] <= since:
                    continue
                if until and entry["timestamp"] >= until:
                    continue

                yield entry
                count += 1

                if limit and count >= limit:
                    break

            except Exception as e:
                logger.warning(f"Failed to convert entry: {e}")
                continue

    async def _load_data(self) -> dict[str, Any]:
        """Load JSON data from source."""
        if self.source_url.startswith(("http://", "https://")):
            # HTTP mode
            try:
                import aiohttp

                async with aiohttp.ClientSession() as session:
                    async with session.get(self.source_url) as response:
                        if response.status != 200:
                            raise IngestionError(
                                f"HTTP request failed with status {response.status}",
                                source_system=self.source_system_name,
                            )
                        return cast(dict[str, Any], await response.json())
            except ImportError as err:
                raise IngestionError(
                    "aiohttp is required for HTTP sources. Install with: pip install aiohttp",
                    source_system=self.source_system_name,
                ) from err
        else:
            # File mode
            path = Path(self.source_url)
            if not path.exists():
                raise IngestionError(
                    f"Source file not found: {self.source_url}",
                    source_system=self.source_system_name,
                )
            with open(path) as f:
                return cast(dict[str, Any], json.load(f))

    def _convert_entry(self, data: dict[str, Any]) -> EnhancedLogbookEntry:
        """Convert generic JSON entry to EnhancedLogbookEntry."""
        now = datetime.now(UTC)

        # Parse timestamp
        timestamp = self._parse_timestamp(data.get("timestamp", ""))

        # Build raw_text from title + text if both present
        title = data.get("title", "")
        text = data.get("text", "")
        if title and text:
            raw_text = f"{title}\n\n{text}"
        else:
            raw_text = title or text

        # Build attachments
        attachments: list[AttachmentInfo] = []
        for att in data.get("attachments", []):
            if isinstance(att, dict) and "url" in att:
                attachments.append(
                    {
                        "url": att["url"],
                        "type": att.get("type"),
                        "filename": att.get("filename"),
                        "thumbnail_url": att.get("thumbnail_url"),
                        "caption": att.get("caption"),
                    }
                )

        # Build metadata from optional fields
        metadata: dict[str, Any] = {}
        if title:
            metadata["title"] = title
        if data.get("books"):
            metadata["books"] = data["books"]
        if data.get("tags"):
            metadata["tags"] = data["tags"]
        if data.get("linked_to"):
            metadata["linked_to"] = data["linked_to"]
        if data.get("level"):
            metadata["level"] = data["level"]
        if data.get("categories"):
            metadata["categories"] = data["categories"]
        if data.get("loto_tag"):
            metadata["loto_tag"] = data["loto_tag"]
        if data.get("event_time"):
            metadata["event_time"] = data["event_time"]
        if data.get("segment_area"):
            metadata["segment_area"] = data["segment_area"]
        if data.get("body_format"):
            metadata["body_format"] = data["body_format"]
        if data.get("entrymakers"):
            metadata["entrymakers"] = data["entrymakers"]
        if data.get("num_comments") is not None:
            metadata["num_comments"] = data["num_comments"]
        if data.get("needs_attention") is not None:
            metadata["needs_attention"] = data["needs_attention"]

        return {
            "entry_id": str(data["id"]),
            "source_system": self.source_system_name,
            "timestamp": timestamp,
            "author": data.get("author", ""),
            "raw_text": raw_text,
            "attachments": attachments,
            "metadata": metadata,
            "created_at": now,
            "updated_at": now,
        }

    def _parse_timestamp(self, value: str | int | float) -> datetime:
        """Parse timestamp from various formats."""
        if isinstance(value, (int, float)):
            # Unix epoch
            return datetime.fromtimestamp(value, tz=UTC)

        if isinstance(value, str):
            # Try ISO 8601 first
            try:
                # Handle with or without Z suffix
                if value.endswith("Z"):
                    value = value[:-1] + "+00:00"
                return datetime.fromisoformat(value)
            except ValueError:
                pass  # Not ISO 8601; try next format

            # Try Unix epoch as string
            try:
                return datetime.fromtimestamp(float(value), tz=UTC)
            except ValueError:
                pass  # Not a Unix epoch string; fall through to raise below

        raise ValueError(f"Cannot parse timestamp: {value}")
