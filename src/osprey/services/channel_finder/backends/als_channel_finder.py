"""ALS ChannelFinder REST API backend for PV metadata.

Queries the ChannelFinder service at controls.als.lbl.gov (265k+ PVs)
via an SSH tunnel. The tunnel must be opened externally before use::

    ./scripts/als_channel_finder_tunnel.sh

Then configure in config.yml::

    channel_finder:
      direct:
        backend: als_channel_finder
        backend_url: https://localhost:8443/ChannelFinder
"""

from __future__ import annotations

import logging
import ssl
from typing import Any

import aiohttp

from osprey.services.channel_finder.backends.base import (
    PVInfoBackend,
    PVRecord,
    SearchResult,
)

logger = logging.getLogger(__name__)

# ChannelFinder property names → PVRecord fields
_PROP_FIELD_MAP = {
    "recordType": "record_type",
    "recordDesc": "description",
    "hostName": "host",
    "iocName": "ioc",
}

# Properties consumed as PVRecord fields (everything else goes into tags)
_KNOWN_PROPS = set(_PROP_FIELD_MAP) | {"pvName"}

_MAX_PAGE_SIZE = 200
_REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=30)


def _channel_to_pvrecord(channel: dict[str, Any]) -> PVRecord:
    """Convert a ChannelFinder channel JSON object to a PVRecord."""
    props: dict[str, str] = {}
    for prop in channel.get("properties", []):
        name = prop.get("name", "")
        value = prop.get("value")
        if name and value is not None:
            props[name] = str(value)

    tags = {k: v for k, v in props.items() if k not in _KNOWN_PROPS}

    return PVRecord(
        name=channel.get("name", ""),
        record_type=props.get("recordType", ""),
        description=props.get("recordDesc", ""),
        host=props.get("hostName", ""),
        ioc=props.get("iocName", ""),
        units="",  # not available from ChannelFinder
        tags=tags,
    )


class ALSChannelFinderBackend(PVInfoBackend):
    """PV info backend that queries the ALS ChannelFinder REST API.

    Args:
        base_url: Base URL of the ChannelFinder service
            (e.g. ``https://localhost:8443/ChannelFinder``).
    """

    def __init__(self, base_url: str = "https://localhost:8443/ChannelFinder") -> None:
        self._base_url = base_url.rstrip("/")
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Lazily create and return the shared aiohttp session."""
        if self._session is None or self._session.closed:
            # Disable SSL verification for localhost tunnel
            ssl_ctx = ssl.create_default_context()
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE
            connector = aiohttp.TCPConnector(ssl=ssl_ctx)
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=_REQUEST_TIMEOUT,
            )
        return self._session

    async def close(self) -> None:
        """Close the underlying HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def _get_json(self, path: str, params: dict[str, str] | None = None) -> Any:
        """Make a GET request and return parsed JSON."""
        session = await self._get_session()
        url = f"{self._base_url}{path}"
        async with session.get(url, params=params) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def _get_count(self, params: dict[str, str]) -> int:
        """Get the count of channels matching the given query params."""
        session = await self._get_session()
        url = f"{self._base_url}/resources/channels/count"
        async with session.get(url, params=params) as resp:
            resp.raise_for_status()
            text = await resp.text()
            return int(text.strip())

    def _build_query_params(
        self,
        pattern: str,
        *,
        record_type: str | None = None,
        ioc: str | None = None,
    ) -> dict[str, str]:
        """Build ChannelFinder query params from search filters."""
        params: dict[str, str] = {"~name": pattern}
        if record_type:
            params["recordType"] = record_type
        if ioc:
            params["iocName"] = ioc
        return params

    async def search(
        self,
        pattern: str,
        *,
        record_type: str | None = None,
        ioc: str | None = None,
        description_contains: str | None = None,
        page: int = 1,
        page_size: int = 100,
    ) -> SearchResult:
        page_size = min(page_size, _MAX_PAGE_SIZE)
        page = max(page, 1)

        params = self._build_query_params(pattern, record_type=record_type, ioc=ioc)

        # Get total count (server-side filters only)
        total_count = await self._get_count(params)

        # Fetch the requested page
        offset = (page - 1) * page_size
        fetch_params = {**params, "~size": str(page_size), "~from": str(offset)}

        if total_count == 0 or offset >= total_count:
            return SearchResult(
                records=[],
                total_count=total_count,
                has_more=False,
                page=page,
                page_size=page_size,
            )

        channels = await self._get_json("/resources/channels", params=fetch_params)
        if not isinstance(channels, list):
            channels = []

        records = [_channel_to_pvrecord(ch) for ch in channels]

        # Client-side post-filter for description_contains
        if description_contains:
            lower = description_contains.lower()
            records = [r for r in records if lower in r.description.lower()]
            # With client-side filtering, total_count is approximate.
            # Report filtered count for this page; has_more is based on
            # whether the server has more unfiltered results.
            return SearchResult(
                records=records,
                total_count=len(records),
                has_more=(offset + page_size) < total_count,
                page=page,
                page_size=page_size,
            )

        has_more = (offset + page_size) < total_count
        return SearchResult(
            records=records,
            total_count=total_count,
            has_more=has_more,
            page=page,
            page_size=page_size,
        )

    async def get_metadata(self, pv_names: list[str]) -> list[PVRecord]:
        pv_names = pv_names[:100]  # cap per interface contract
        if not pv_names:
            return []

        results: list[PVRecord] = []
        # Batch into groups to avoid overly long query strings
        batch_size = 20
        for i in range(0, len(pv_names), batch_size):
            batch = pv_names[i : i + batch_size]
            for name in batch:
                try:
                    channels = await self._get_json("/resources/channels", params={"~name": name})
                    if isinstance(channels, list):
                        for ch in channels:
                            if ch.get("name") == name:
                                results.append(_channel_to_pvrecord(ch))
                                break
                except (aiohttp.ClientError, TimeoutError):
                    logger.warning("Failed to fetch metadata for PV: %s", name)

        return results
