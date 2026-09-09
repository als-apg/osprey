"""Extract entry metadata from a sidecar JSON attachment.

Scans an entry's attachments for the sidecar filenames its adapter declares,
fetches the content (local file or HTTP), and merges the result into the
entry's metadata dict. All failures are non-fatal --- logged as warnings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from osprey.services.ariel_search.models import EnhancedLogbookEntry
from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from osprey.services.ariel_search.config import IngestionConfig
    from osprey.services.ariel_search.ingestion.base import FacilityAdapter

logger = get_logger("ariel.ingestion")

#: The filename the step matched before adapters could declare their own. Used
#: when no adapter is supplied, so a direct caller behaves as it always did.
DEFAULT_SIDECAR_NAMES: tuple[str, ...] = ("metadata.json",)


async def extract_metadata_from_attachments(
    entry: EnhancedLogbookEntry,
    *,
    fetch_timeout: int = 5,
    adapter: FacilityAdapter | None = None,
    ingestion: IngestionConfig | None = None,
) -> None:
    """Merge an entry's sidecar metadata attachment into its metadata dict.

    The function modifies *entry* in place. If no attachment matches, or if
    fetching or parsing fails, the entry is left unchanged.

    Which filenames count is the ADAPTER's answer, not this module's:
    ``metadata.json`` is one facility's convention, and a logbook that names
    its sidecar anything else was silently getting no metadata at all. An
    adapter declares its own through
    :attr:`~osprey.services.ariel_search.ingestion.base.FacilityAdapter.metadata_sidecar_names`.

    The HTTP fetch honours the ingestion config it is given --- the same TLS
    verification, site CA and SOCKS proxy the adapter itself uses. Without
    that, this step reached the logbook host over a connection configured
    differently from every other request in the same ingest: unverified where
    the adapter verified, and direct where the adapter went through a proxy
    (so, on an air-gapped site, not at all).

    Args:
        entry: The logbook entry to enrich.
        fetch_timeout: HTTP request timeout in seconds.
        adapter: The adapter that produced *entry*, for its sidecar filenames
            and its proxy connector. ``None`` falls back to
            :data:`DEFAULT_SIDECAR_NAMES` and a plain session.
        ingestion: The ingestion config, for ``verify_ssl`` and ``ca_bundle``.
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

        try:
            data = await _fetch_metadata(url, fetch_timeout, adapter, ingestion)
        except Exception:
            logger.warning("Failed to fetch sidecar metadata from %s", url, exc_info=True)
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


async def _fetch_metadata(
    url: str,
    timeout: int,
    adapter: FacilityAdapter | None = None,
    ingestion: IngestionConfig | None = None,
) -> Any:
    """Fetch and parse a sidecar metadata file from a URL or local path."""
    if url.startswith(("http://", "https://")):
        import aiohttp

        from osprey.services.ariel_search.ingestion.http import build_ssl_context

        ssl_context: Any = True
        if ingestion is not None:
            ssl_context = build_ssl_context(ingestion.verify_ssl, ingestion.ca_bundle)

        # The adapter's own connector, so a SOCKS proxy the ingest goes
        # through carries this request too. A sidecar fetched without an
        # adapter has none to inherit and goes out on aiohttp's default.
        connector = adapter._create_connector() if adapter is not None else None

        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=timeout),
                ssl=ssl_context,
            ) as resp:
                resp.raise_for_status()
                return await resp.json()
    else:
        # Treat as local file path
        path = Path(url)
        if path.exists():
            return json.loads(path.read_text())
        raise FileNotFoundError(f"sidecar metadata not found at {url}")
