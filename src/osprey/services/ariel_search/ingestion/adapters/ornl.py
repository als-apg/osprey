"""ORNL Logbook ingestion adapter.

This module provides the adapter for Oak Ridge National Laboratory
electronic logbook system.

See 01_DATA_LAYER.md Section 5.8 for specification.
"""

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from osprey.services.ariel_search.exceptions import IngestionError
from osprey.services.ariel_search.ingestion.base import BaseAdapter
from osprey.services.ariel_search.models import AttachmentInfo, EnhancedLogbookEntry
from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from osprey.services.ariel_search.config import ARIELConfig

logger = get_logger("ariel")


class ORNLLogbookAdapter(BaseAdapter):
    """Adapter for Oak Ridge National Laboratory electronic logbook system."""

    def __init__(self, config: "ARIELConfig") -> None:
        """Initialize the adapter."""
        super().__init__(config)

        if not config.ingestion or not config.ingestion.source_url:
            raise IngestionError(
                "source_url is required for ornl_logbook adapter",
                source_system=self.source_system_name,
            )

        self.source_url = config.ingestion.source_url

        # ORNL-specific config defaults
        self.merge_title_content = True
        self.store_event_time = True
        self.logbooks_filter: list[str] | None = None

    @property
    def source_system_name(self) -> str:
        """Return the source system identifier."""
        return "ORNL Logbook"

    async def fetch_entries(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[EnhancedLogbookEntry]:
        """Fetch entries from ORNL logbook source.

        Args:
            since: Only fetch entries after this timestamp
            until: Only fetch entries before this timestamp
            limit: Maximum number of entries to fetch

        Yields:
            EnhancedLogbookEntry objects
        """
        data = await self._load_data()

        # Handle various response formats
        if isinstance(data, dict) and "entries" in data:
            entries = data.get("entries", [])
        elif isinstance(data, list):
            entries = data
        else:
            entries = []

        count = 0
        for entry_data in entries:
            try:
                entry = self._convert_entry(entry_data)

                # Apply logbooks filter if configured
                if self.logbooks_filter:
                    entry_books = entry["metadata"].get("books", [])
                    if not any(b in self.logbooks_filter for b in entry_books):
                        continue

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

    async def _load_data(self) -> Any:
        """Load JSON data from source."""
        if self.source_url.startswith(("http://", "https://")):
            try:
                import aiohttp

                async with aiohttp.ClientSession() as session:
                    async with session.get(self.source_url) as response:
                        if response.status != 200:
                            raise IngestionError(
                                f"HTTP request failed with status {response.status}",
                                source_system=self.source_system_name,
                            )
                        return await response.json()
            except ImportError as err:
                raise IngestionError(
                    "aiohttp is required for HTTP sources",
                    source_system=self.source_system_name,
                ) from err
        else:
            path = Path(self.source_url)
            if not path.exists():
                raise IngestionError(
                    f"Source file not found: {self.source_url}",
                    source_system=self.source_system_name,
                )
            with open(path) as f:
                return json.load(f)

    def _convert_entry(self, data: dict[str, Any]) -> EnhancedLogbookEntry:
        """Convert ORNL JSON entry to EnhancedLogbookEntry.

        See 01_DATA_LAYER.md Section 5.8 for field mapping.
        """
        now = datetime.now(UTC)

        # Parse entry_time as timestamp
        entry_time = data.get("entry_time", "")
        timestamp = self._parse_timestamp(entry_time) if entry_time else now

        # Build raw_text from title + content
        title = data.get("title", "")
        content = data.get("content", "")

        if self.merge_title_content and title and content:
            raw_text = f"{title}\n\n{content}"
        else:
            raw_text = title or content

        # Transform attachments
        attachments = self._transform_attachments(data)

        # Build metadata with ORNL-specific fields
        metadata: dict[str, Any] = {}
        if title:
            metadata["title"] = title

        # Store logbook as books array for consistency with JLab
        logbook = data.get("logbook")
        if logbook:
            metadata["books"] = [logbook] if isinstance(logbook, str) else logbook

        if data.get("segment/area"):
            metadata["segment_area"] = data["segment/area"]

        # Store event_time separately from entry_time
        if self.store_event_time and data.get("event_time"):
            event_time = self._parse_timestamp(data["event_time"])
            metadata["event_time"] = event_time.isoformat()

        # Reference to related entry
        if data.get("reference"):
            metadata["linked_to"] = data["reference"]

        # Attachment headers
        if data.get("attachment_header"):
            att_headers = data["attachment_header"]
            if isinstance(att_headers, str):
                metadata["attachment_headers"] = [att_headers]
            else:
                metadata["attachment_headers"] = att_headers

        return {
            "entry_id": str(data.get("ID", data.get("id", ""))),
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
            return datetime.fromtimestamp(value, tz=UTC)

        if isinstance(value, str):
            # Try ISO 8601
            try:
                if value.endswith("Z"):
                    value = value[:-1] + "+00:00"
                return datetime.fromisoformat(value)
            except ValueError:
                pass  # Not ISO 8601; try next format

            # Try Unix epoch as string
            try:
                return datetime.fromtimestamp(float(value), tz=UTC)
            except ValueError:
                pass  # Not a Unix epoch string; fall through to default below

        return datetime.now(UTC)

    def _transform_attachments(
        self,
        data: dict[str, Any],
    ) -> list[AttachmentInfo]:
        """Transform ORNL attachments to ARIEL format."""
        result: list[AttachmentInfo] = []

        # ORNL may have attachment_header but not full attachment objects
        # This is a simplified implementation
        attachments = data.get("attachments", [])
        if not attachments and data.get("attachment") == "Y":
            # Has attachment flag but no details - create placeholder
            headers = data.get("attachment_header", [])
            if isinstance(headers, str):
                headers = [headers]
            for header in headers:
                result.append(
                    {
                        "url": "",  # URL not available
                        "filename": header,
                        "type": None,
                    }
                )
        else:
            for att in attachments:
                if not isinstance(att, dict):
                    continue
                result.append(
                    {
                        "url": att.get("url", ""),
                        "type": att.get("type"),
                        "filename": att.get("filename"),
                    }
                )

        return result
