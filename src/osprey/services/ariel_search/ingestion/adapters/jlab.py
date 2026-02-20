"""JLab Logbook ingestion adapter.

This module provides the adapter for Jefferson Lab electronic logbook system.
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


class JLabLogbookAdapter(BaseAdapter):
    """Adapter for Jefferson Lab electronic logbook system."""

    def __init__(self, config: "ARIELConfig") -> None:
        """Initialize the adapter."""
        super().__init__(config)

        if not config.ingestion or not config.ingestion.source_url:
            raise IngestionError(
                "source_url is required for jlab_logbook adapter",
                source_system=self.source_system_name,
            )

        self.source_url = config.ingestion.source_url

        self.merge_title_body = True
        self.include_thumbnails = True
        self.books_filter: list[str] | None = None

    @property
    def source_system_name(self) -> str:
        """Return the source system identifier."""
        return "JLab Logbook"

    async def fetch_entries(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[EnhancedLogbookEntry]:
        """Fetch entries from JLab logbook source.

        Args:
            since: Only fetch entries after this timestamp
            until: Only fetch entries before this timestamp
            limit: Maximum number of entries to fetch

        Yields:
            EnhancedLogbookEntry objects
        """
        data = await self._load_data()

        # JLab API returns entries in data.entries
        if isinstance(data, dict) and "data" in data:
            entries = data.get("data", {}).get("entries", [])
        elif isinstance(data, dict) and "entries" in data:
            entries = data.get("entries", [])
        else:
            entries = data if isinstance(data, list) else []

        count = 0
        for entry_data in entries:
            try:
                entry = self._convert_entry(entry_data)

                if self.books_filter:
                    entry_books = entry["metadata"].get("books", [])
                    if not any(b in self.books_filter for b in entry_books):
                        continue

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
        """Convert JLab JSON entry to EnhancedLogbookEntry."""
        now = datetime.now(UTC)

        # Parse timestamp from created.timestamp (Unix epoch string)
        created = data.get("created", {})
        timestamp_str = created.get("timestamp", "0") if isinstance(created, dict) else "0"
        try:
            timestamp_epoch = int(timestamp_str)
            timestamp = datetime.fromtimestamp(timestamp_epoch, tz=UTC)
        except (ValueError, TypeError):
            timestamp = now

        title = data.get("title", "")
        body = data.get("body", {})
        content = body.get("content", "") if isinstance(body, dict) else ""

        if self.merge_title_body and title and content:
            raw_text = f"{title}\n\n{content}"
        else:
            raw_text = title or content

        attachments = self._transform_attachments(data.get("attachments", []))

        metadata: dict[str, Any] = {}
        if title:
            metadata["title"] = title
        if data.get("books"):
            metadata["books"] = data["books"]
        if data.get("tags"):
            metadata["tags"] = data["tags"]
        if isinstance(body, dict) and body.get("format"):
            metadata["body_format"] = body["format"]
        if data.get("entrymakers"):
            metadata["entrymakers"] = data["entrymakers"]
        if data.get("numComments") is not None:
            metadata["num_comments"] = data["numComments"]
        if data.get("needsAttention") is not None:
            metadata["needs_attention"] = data["needsAttention"]

        return {
            "entry_id": str(data.get("lognumber", data.get("id", ""))),
            "source_system": self.source_system_name,
            "timestamp": timestamp,
            "author": data.get("author", ""),
            "raw_text": raw_text,
            "attachments": attachments,
            "metadata": metadata,
            "created_at": now,
            "updated_at": now,
        }

    def _transform_attachments(
        self,
        source_attachments: list[dict[str, Any]],
    ) -> list[AttachmentInfo]:
        """Transform JLab attachments to ARIEL format."""
        result: list[AttachmentInfo] = []
        for att in source_attachments:
            if not isinstance(att, dict) or "url" not in att:
                continue

            url = att["url"]
            filename = url.rsplit("/", 1)[-1] if "/" in url else url

            attachment: AttachmentInfo = {
                "url": url,
                "type": att.get("type"),
                "filename": filename,
            }

            if self.include_thumbnails and att.get("thumbnail"):
                attachment["thumbnail_url"] = att["thumbnail"]
            if att.get("caption"):
                attachment["caption"] = att["caption"]

            result.append(attachment)

        return result
