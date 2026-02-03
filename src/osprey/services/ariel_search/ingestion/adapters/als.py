"""ALS Logbook ingestion adapter.

This module provides the adapter for ALS eLog system.

See 01_DATA_LAYER.md Sections 5.3, 5.5 for specification.
"""

import json
import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from osprey.services.ariel_search.exceptions import IngestionError
from osprey.services.ariel_search.ingestion.base import BaseAdapter
from osprey.services.ariel_search.models import AttachmentInfo, EnhancedLogbookEntry

if TYPE_CHECKING:
    from osprey.services.ariel_search.config import ARIELConfig

logger = logging.getLogger(__name__)


def parse_als_categories(category_str: str) -> list[str]:
    """Parse ALS comma-separated categories into array.

    Args:
        category_str: Comma-separated category string

    Returns:
        List of category names
    """
    if not category_str:
        return []
    # Handle leading/trailing commas in historical entries
    return [cat.strip() for cat in category_str.split(",") if cat.strip()]


def transform_als_attachments(
    source_attachments: list[dict[str, Any]],
    url_prefix: str,
) -> list[AttachmentInfo]:
    """Transform ALS relative attachment paths to full URLs.

    Args:
        source_attachments: List of attachment dicts from ALS logbook
        url_prefix: Base URL from config (e.g., "https://elog.als.lbl.gov/")

    Returns:
        List of AttachmentInfo dicts with full URLs
    """
    result: list[AttachmentInfo] = []
    for att in source_attachments:
        if isinstance(att, dict) and "url" in att:
            path = att["url"]
            # Extract filename from path
            filename = path.rsplit("/", 1)[-1] if "/" in path else path
            result.append({
                "url": url_prefix.rstrip("/") + "/" + path.lstrip("/"),
                "filename": filename,
                "type": None,  # ALS source doesn't include MIME type
            })
    return result


class ALSLogbookAdapter(BaseAdapter):
    """Adapter for ALS eLog system.

    Handles both file-based (JSONL) and HTTP-based sources.
    MVP implements file mode only; HTTP mode raises NotImplementedError.
    """

    def __init__(self, config: "ARIELConfig") -> None:
        """Initialize the adapter."""
        super().__init__(config)

        if not config.ingestion or not config.ingestion.source_url:
            raise IngestionError(
                "source_url is required for als_logbook adapter",
                source_system=self.source_system_name,
            )

        self.source_url = config.ingestion.source_url
        self.source_type = self._detect_source_type(self.source_url)

        # ALS-specific config defaults
        self.merge_subject_details = True
        self.attachment_url_prefix = "https://elog.als.lbl.gov/"
        self.skip_empty_entries = True

    @property
    def source_system_name(self) -> str:
        """Return the source system identifier."""
        return "ALS eLog"

    def _detect_source_type(self, source_url: str) -> Literal["file", "http"]:
        """Detect source type from URL scheme."""
        if source_url.startswith(("http://", "https://")):
            return "http"
        return "file"

    async def fetch_entries(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[EnhancedLogbookEntry]:
        """Fetch entries from ALS logbook source.

        Args:
            since: Only fetch entries after this timestamp
            until: Only fetch entries before this timestamp
            limit: Maximum number of entries to fetch

        Yields:
            EnhancedLogbookEntry objects
        """
        if self.source_type == "http":
            raise NotImplementedError(
                "HTTP mode not yet implemented. Use file-based JSONL source for MVP."
            )

        # File mode: read JSONL line by line
        path = Path(self.source_url)
        if not path.exists():
            raise IngestionError(
                f"Source file not found: {self.source_url}",
                source_system=self.source_system_name,
            )

        count = 0
        with open(path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    entry = self._convert_entry(data)

                    # Skip empty entries if configured
                    if self.skip_empty_entries and not entry["raw_text"].strip():
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

                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON at line {line_num}: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Failed to convert entry at line {line_num}: {e}")
                    continue

    def _convert_entry(self, data: dict[str, Any]) -> EnhancedLogbookEntry:
        """Convert ALS JSON entry to EnhancedLogbookEntry.

        See 01_DATA_LAYER.md Section 5.5 for field mapping.
        """
        now = datetime.now(UTC)

        # Parse timestamp - ALS uses Unix epoch STRING (not int)
        timestamp_str = data.get("timestamp", "0")
        try:
            timestamp_epoch = int(timestamp_str)
            timestamp = datetime.fromtimestamp(timestamp_epoch, tz=UTC)
        except (ValueError, TypeError):
            timestamp = now

        # Build raw_text from subject + details
        subject = data.get("subject", "")
        details = data.get("details", "")
        if self.merge_subject_details and subject and details:
            raw_text = f"{subject}\n\n{details}"
        else:
            raw_text = subject or details

        # Parse categories
        categories = parse_als_categories(data.get("category", ""))

        # Handle "0" as null for tag and linkedto
        tag = data.get("tag")
        if tag == "0":
            tag = None

        linked_to = data.get("linkedto")
        if linked_to == "0":
            linked_to = None

        # Transform attachments
        source_attachments = data.get("attachments", [])
        attachments = transform_als_attachments(
            source_attachments,
            self.attachment_url_prefix,
        )

        # Build metadata with ALS-specific fields
        metadata: dict[str, Any] = {}
        if subject:
            metadata["subject"] = subject
        if data.get("level"):
            metadata["level"] = data["level"]
        if categories:
            metadata["categories"] = categories
        if tag:
            metadata["loto_tag"] = tag
        if linked_to:
            metadata["linked_to"] = linked_to

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
