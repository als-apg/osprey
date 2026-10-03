"""Extract entry metadata from a sidecar JSON attachment.

Scans an entry's attachments for the sidecar filenames its adapter declares,
fetches each through :func:`~osprey.services.ariel_search.attachments.fetch.fetch_attachment_bytes`
--- the same origin set, file-source confinement, size cap, TLS and proxy as
every other attachment fetch --- and merges the result into the entry's
metadata dict. All failures are non-fatal --- logged as warnings.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from osprey.services.ariel_search.attachments.fetch import fetch_attachment_bytes, origins_for
from osprey.services.ariel_search.models import EnhancedLogbookEntry
from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from osprey.services.ariel_search.ingestion.base import FacilityAdapter

logger = get_logger("ariel.ingestion")

#: The filename the step matches when no adapter declares its own.
DEFAULT_SIDECAR_NAMES: tuple[str, ...] = ("metadata.json",)

#: Largest sidecar accepted, in bytes.
SIDECAR_MAX_BYTES: int = 1024 * 1024

#: Total seconds one sidecar fetch may take.
SIDECAR_FETCH_TIMEOUT: int = 5


async def extract_metadata_from_attachments(
    entry: EnhancedLogbookEntry,
    *,
    fetch_timeout: int = SIDECAR_FETCH_TIMEOUT,
    adapter: FacilityAdapter | None = None,
) -> None:
    """Merge an entry's sidecar metadata attachment into its metadata dict.

    The function modifies *entry* in place. If no attachment matches, or if
    fetching or parsing fails, the entry is left unchanged.

    Which filenames count is the adapter's answer, declared through
    :attr:`~osprey.services.ariel_search.ingestion.base.FacilityAdapter.metadata_sidecar_names`;
    ``metadata.json`` is the default.

    Where a sidecar may be fetched from is the adapter's answer too: an http(s)
    sidecar must lie in the adapter's origin set (its own origin plus
    ``ariel.attachments.allowed_origins``), and a file source reads only
    relative paths under its file base. Without an adapter there is no origin
    set and no file base, so nothing is fetched.

    Args:
        entry: The logbook entry to enrich.
        fetch_timeout: Total timeout of one sidecar fetch, in seconds.
        adapter: The adapter that produced *entry*, for its sidecar filenames,
            its origins, its file base, its TLS context and its proxy.
    """
    names = _sidecar_names(adapter)
    attachments = entry.get("attachments", [])

    matched = False
    for att in attachments:
        filename = (att.get("filename") or "").lower()
        if filename not in names:
            continue
        matched = True

        url = att.get("url", "")
        if not url:
            continue
        if adapter is None:
            logger.debug("No adapter to fetch sidecar metadata %s through; skipped", url)
            continue

        try:
            data = await _fetch_metadata(url, fetch_timeout, adapter)
        except Exception:
            logger.warning("Failed to read sidecar metadata from %s", url, exc_info=True)
            continue

        if isinstance(data, dict):
            entry["metadata"].update(data)
            logger.debug("Merged sidecar metadata from %s into entry %s", url, entry["entry_id"])

    if attachments and not matched:
        logger.debug(
            "Entry %s has %d attachment(s) but none named %s",
            entry["entry_id"],
            len(attachments),
            ", ".join(sorted(names)),
        )


def _sidecar_names(adapter: FacilityAdapter | None) -> set[str]:
    """The lower-cased filenames that count as a sidecar for *adapter*."""
    declared = getattr(adapter, "metadata_sidecar_names", None) or DEFAULT_SIDECAR_NAMES
    return {str(name).lower() for name in declared}


async def _fetch_metadata(url: str, timeout: float, adapter: FacilityAdapter) -> Any:
    """Fetch and parse one sidecar through the attachment fetcher.

    Args:
        url: The sidecar's url or relative path from upstream data.
        timeout: Total timeout of the fetch, in seconds.
        adapter: The adapter whose origins, file base and transport apply.

    Returns:
        The parsed JSON, or ``None`` when the fetcher delivered no bytes; that
        case is logged once at WARNING with the fetcher's code.

    Raises:
        ValueError: The bytes are not valid JSON.
    """
    out = await fetch_attachment_bytes(
        url,
        SIDECAR_MAX_BYTES,
        origins_for(adapter, adapter.config),
        adapter,
        total=timeout,
    )
    if out.data is None:
        if out.code == "origin_not_allowed":
            logger.warning(
                "Sidecar metadata %s not fetched (origin_not_allowed): list its origin "
                "in ariel.attachments.allowed_origins to read it",
                url,
            )
        else:
            logger.warning(
                "Sidecar metadata %s not fetched (%s)", url, out.code or "transient failure"
            )
        return None
    return json.loads(out.data)
