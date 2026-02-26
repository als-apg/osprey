"""Generic JSON ingestion adapter.

This module provides a flexible adapter for testing and facilities
without custom APIs. Supports both reading and writing to local JSON files.
"""

import fcntl
import json
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from osprey.services.ariel_search.exceptions import IngestionError
from osprey.services.ariel_search.ingestion.base import FacilityAdapter
from osprey.services.ariel_search.models import AttachmentInfo, EnhancedLogbookEntry
from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.models import FacilityEntryCreateRequest

logger = get_logger("ariel")


class GenericJSONAdapter(FacilityAdapter):
    """Generic JSON ingestion adapter.

    Reads entries from JSON files or HTTP endpoints with a flexible schema.
    Supports writing to local JSON files.
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

    @property
    def supports_write(self) -> bool:
        """Write is supported only for local file sources, not HTTP."""
        return not self.source_url.startswith(("http://", "https://"))

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

    async def create_entry(self, request: "FacilityEntryCreateRequest") -> str:
        """Create an entry in the local JSON file.

        Args:
            request: Entry creation request.

        Returns:
            The generated entry ID.

        Raises:
            NotImplementedError: If the source is an HTTP URL.
            IngestionError: If the write fails.
        """
        if not self.supports_write:
            raise NotImplementedError(
                "GenericJSONAdapter only supports writing to local file sources, not HTTP"
            )

        entry_id = f"local-{uuid.uuid4().hex[:12]}"
        now = datetime.now(UTC)

        entry_data: dict[str, Any] = {
            "id": entry_id,
            "timestamp": now.isoformat(),
            "author": request.author or "",
            "title": request.subject,
            "text": request.details,
            "tags": request.tags,
            "attachments": [],
            "metadata": request.metadata,
        }

        if request.logbook:
            entry_data["books"] = [request.logbook]

        self._append_entry(entry_data)

        logger.info(f"Created local JSON entry {entry_id}")
        return entry_id

    def _append_entry(self, entry_data: dict[str, Any]) -> None:
        """Append an entry to the local JSON file with file locking.

        Args:
            entry_data: Entry dict matching the JSON schema that _convert_entry reads.
        """
        path = Path(self.source_url)

        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = path.with_suffix(".tmp")

        try:
            if path.exists():
                with open(path) as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                    try:
                        data = json.load(f)
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            else:
                data = {"entries": []}

            data.setdefault("entries", [])
            data["entries"].append(entry_data)

            # Atomic write: write to tmp file, then replace
            with open(tmp_path, "w") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    json.dump(data, f, indent=2, default=str)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            # os.replace is atomic on POSIX
            tmp_path.replace(path)

        except (json.JSONDecodeError, OSError) as e:
            # Clean up tmp file on failure
            if tmp_path.exists():
                tmp_path.unlink()
            raise IngestionError(
                f"Failed to append entry to {self.source_url}: {e}",
                source_system=self.source_system_name,
            ) from e

    async def _load_data(self) -> dict[str, Any]:
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
                        return cast(dict[str, Any], await response.json())
            except ImportError as err:
                raise IngestionError(
                    "aiohttp is required for HTTP sources. Install with: pip install aiohttp",
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
                return cast(dict[str, Any], json.load(f))

    def _convert_entry(self, data: dict[str, Any]) -> EnhancedLogbookEntry:
        """Convert generic JSON entry to EnhancedLogbookEntry."""
        now = datetime.now(UTC)

        timestamp = self._parse_timestamp(data.get("timestamp", ""))

        title = data.get("title", "")
        text = data.get("text", "")
        if title and text:
            raw_text = f"{title}\n\n{text}"
        else:
            raw_text = title or text

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
            return datetime.fromtimestamp(value, tz=UTC)

        if isinstance(value, str):
            try:
                # Handle with or without Z suffix
                if value.endswith("Z"):
                    value = value[:-1] + "+00:00"
                return datetime.fromisoformat(value)
            except ValueError:
                pass  # Not ISO 8601; try next format

            try:
                return datetime.fromtimestamp(float(value), tz=UTC)
            except ValueError:
                pass  # Not a Unix epoch string; fall through to raise below

        raise ValueError(f"Cannot parse timestamp: {value}")
