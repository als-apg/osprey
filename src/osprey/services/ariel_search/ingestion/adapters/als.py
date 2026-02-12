"""ALS Logbook ingestion adapter.

This module provides the adapter for ALS eLog system.
Supports both file-based (JSONL) and HTTP-based sources.

See 01_DATA_LAYER.md Sections 5.3, 5.5 for specification.
"""

import asyncio
import json
import os
import ssl
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import aiohttp

from osprey.services.ariel_search.exceptions import IngestionError
from osprey.services.ariel_search.ingestion.base import BaseAdapter
from osprey.services.ariel_search.models import AttachmentInfo, EnhancedLogbookEntry
from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from osprey.services.ariel_search.config import ARIELConfig

logger = get_logger("ariel")

# Default start date for full ALS logbook history
ALS_LOGBOOK_START_DATE = datetime(2003, 1, 1, tzinfo=UTC)


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
            result.append(
                {
                    "url": url_prefix.rstrip("/") + "/" + path.lstrip("/"),
                    "filename": filename,
                    "type": None,  # ALS source doesn't include MIME type
                }
            )
    return result


class ALSLogbookAdapter(BaseAdapter):
    """Adapter for ALS eLog system.

    Handles both file-based (JSONL) and HTTP-based sources.
    HTTP mode supports SOCKS proxy for access through SSH tunnels.
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

        # HTTP mode configuration
        self.proxy_url = config.ingestion.proxy_url or os.environ.get("ARIEL_SOCKS_PROXY")
        self.verify_ssl = config.ingestion.verify_ssl
        self.chunk_days = config.ingestion.chunk_days or 365
        self.request_timeout = config.ingestion.request_timeout_seconds or 60
        self.max_retries = config.ingestion.max_retries or 3
        self.retry_delay = config.ingestion.retry_delay_seconds or 5

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
            async for entry in self._fetch_entries_http(since, until, limit):
                yield entry
            return

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

    async def _fetch_entries_http(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[EnhancedLogbookEntry]:
        """Fetch entries via HTTP from the ALS logbook API.

        Uses time windowing to split large date ranges into manageable chunks.
        Deduplicates entries across windows by entry_id.

        Args:
            since: Only fetch entries after this timestamp (default: 2003-01-01)
            until: Only fetch entries before this timestamp (default: now)
            limit: Maximum number of entries to fetch

        Yields:
            EnhancedLogbookEntry objects
        """
        # Default time range: full ALS logbook history
        start_date = since or ALS_LOGBOOK_START_DATE
        end_date = until or datetime.now(UTC)

        # Generate time windows
        windows = self._generate_time_windows(start_date, end_date)
        logger.info(
            f"Fetching ALS logbook entries from {start_date.isoformat()} to {end_date.isoformat()} "
            f"in {len(windows)} window(s)"
        )

        # Track seen IDs for deduplication across windows
        seen_ids: set[str] = set()
        count = 0

        # Create connector with optional proxy
        connector = self._create_connector()

        # Create SSL context
        ssl_context: ssl.SSLContext | bool
        if self.verify_ssl:
            ssl_context = True
        else:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

        timeout = aiohttp.ClientTimeout(total=self.request_timeout)

        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
        ) as session:
            for window_start, window_end in windows:
                try:
                    entries_data = await self._fetch_window_with_retry(
                        session, window_start, window_end, ssl_context
                    )
                except IngestionError:
                    # Re-raise ingestion errors
                    raise
                except Exception as e:
                    logger.error(f"Failed to fetch window {window_start}-{window_end}: {e}")
                    continue

                for data in entries_data:
                    entry_id = str(data.get("id", ""))

                    # Deduplicate across windows
                    if entry_id in seen_ids:
                        continue
                    seen_ids.add(entry_id)

                    try:
                        entry = self._convert_entry(data)

                        # Skip empty entries if configured
                        if self.skip_empty_entries and not entry["raw_text"].strip():
                            continue

                        yield entry
                        count += 1

                        if limit and count >= limit:
                            logger.info(f"Reached limit of {limit} entries")
                            return

                    except Exception as e:
                        logger.warning(f"Failed to convert entry {entry_id}: {e}")
                        continue

        logger.info(f"Fetched {count} entries from ALS logbook API")

    def _generate_time_windows(
        self, start_date: datetime, end_date: datetime
    ) -> list[tuple[datetime, datetime]]:
        """Generate time windows for chunked API requests.

        Args:
            start_date: Start of date range
            end_date: End of date range

        Returns:
            List of (window_start, window_end) tuples
        """
        windows: list[tuple[datetime, datetime]] = []
        window_start = start_date

        while window_start < end_date:
            window_end = min(window_start + timedelta(days=self.chunk_days), end_date)
            windows.append((window_start, window_end))
            window_start = window_end

        return windows

    async def _fetch_window_with_retry(
        self,
        session: aiohttp.ClientSession,
        window_start: datetime,
        window_end: datetime,
        ssl_context: ssl.SSLContext | bool,
    ) -> list[dict[str, Any]]:
        """Fetch a single time window with exponential backoff retry.

        Args:
            session: aiohttp session
            window_start: Start of window
            window_end: End of window
            ssl_context: SSL context for HTTPS requests

        Returns:
            List of entry dictionaries from API

        Raises:
            IngestionError: After max retries exceeded or on 4xx errors
        """
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                return await self._fetch_window(session, window_start, window_end, ssl_context)
            except aiohttp.ClientResponseError as e:
                if 400 <= e.status < 500:
                    # Don't retry 4xx errors
                    raise IngestionError(
                        f"HTTP {e.status} error from ALS API: {e.message}",
                        source_system=self.source_system_name,
                    ) from e
                last_error = e
            except (aiohttp.ClientError, TimeoutError) as e:
                last_error = e

            # Exponential backoff
            delay = self.retry_delay * (2**attempt)
            logger.warning(
                f"Attempt {attempt + 1}/{self.max_retries} failed for window "
                f"{window_start.isoformat()}-{window_end.isoformat()}, "
                f"retrying in {delay}s: {last_error}"
            )
            await asyncio.sleep(delay)

        raise IngestionError(
            f"Max retries ({self.max_retries}) exceeded fetching ALS logbook: {last_error}",
            source_system=self.source_system_name,
        )

    async def _fetch_window(
        self,
        session: aiohttp.ClientSession,
        window_start: datetime,
        window_end: datetime,
        ssl_context: ssl.SSLContext | bool,
    ) -> list[dict[str, Any]]:
        """Fetch a single time window from the ALS logbook API.

        Args:
            session: aiohttp session
            window_start: Start of window
            window_end: End of window
            ssl_context: SSL context for HTTPS requests

        Returns:
            List of entry dictionaries from API
        """
        # Convert to Unix timestamps
        start_ts = int(window_start.timestamp())
        end_ts = int(window_end.timestamp())

        params = {
            "op": "retrieve",
            "start": str(start_ts),
            "end": str(end_ts),
        }

        logger.debug(
            f"Fetching ALS logbook window: {window_start.isoformat()} to {window_end.isoformat()}"
        )

        async with session.get(self.source_url, params=params, ssl=ssl_context) as response:
            response.raise_for_status()
            data = await response.json()

            if not isinstance(data, list):
                logger.warning(f"Unexpected response type: {type(data)}, expected list")
                return []

            logger.debug(f"Fetched {len(data)} entries from window")
            return data

    def _create_connector(self) -> aiohttp.BaseConnector:
        """Create aiohttp connector with optional SOCKS proxy support.

        Returns:
            aiohttp connector (with proxy if configured)

        Raises:
            IngestionError: If proxy is configured but aiohttp-socks is not installed
        """
        if not self.proxy_url:
            return aiohttp.TCPConnector()

        # Try to import aiohttp-socks for SOCKS proxy support
        try:
            from aiohttp_socks import ProxyConnector
        except ImportError as e:
            raise IngestionError(
                "SOCKS proxy configured but aiohttp-socks is not installed. "
                "Install with: pip install 'osprey-framework[ariel-proxy]'",
                source_system=self.source_system_name,
            ) from e

        logger.info(f"Using SOCKS proxy: {self.proxy_url}")
        connector: aiohttp.BaseConnector = ProxyConnector.from_url(self.proxy_url)
        return connector

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
