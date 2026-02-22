"""Extract metadata from metadata.json attachments.

Scans an entry's attachments for files named ``metadata.json``, fetches the
content (local file or HTTP), and merges the result into the entry's metadata
dict.  All failures are non-fatal — logged as warnings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from osprey.services.ariel_search.models import EnhancedLogbookEntry
from osprey.utils.logger import get_logger

logger = get_logger("ariel.ingestion")


async def extract_metadata_from_attachments(
    entry: EnhancedLogbookEntry,
    *,
    fetch_timeout: int = 5,
) -> None:
    """Scan attachments for ``metadata.json`` and merge into entry metadata.

    The function modifies *entry* in place.  If no matching attachment is
    found, or if fetching / parsing fails, the entry is left unchanged.

    Args:
        entry: The logbook entry to enrich.
        fetch_timeout: HTTP request timeout in seconds.
    """
    for att in entry.get("attachments", []):
        filename = att.get("filename") or ""
        if filename.lower() != "metadata.json":
            continue

        url = att.get("url", "")
        if not url:
            continue

        try:
            data = await _fetch_metadata(url, fetch_timeout)
        except Exception:
            logger.warning("Failed to fetch metadata.json from %s", url, exc_info=True)
            continue

        if isinstance(data, dict):
            entry["metadata"].update(data)
            logger.debug("Merged metadata.json from %s into entry %s", url, entry["entry_id"])


async def _fetch_metadata(url: str, timeout: int) -> Any:
    """Fetch and parse a metadata.json file from a URL or local path."""
    if url.startswith(("http://", "https://")):
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                resp.raise_for_status()
                return await resp.json()
    else:
        # Treat as local file path
        path = Path(url)
        if path.exists():
            return json.loads(path.read_text())
        raise FileNotFoundError(f"metadata.json not found at {url}")
