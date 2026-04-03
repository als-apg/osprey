"""
APS Archiver Appliance connector.

Direct REST API connector for the APS EPICS Archiver Appliance at
pvarchiver.aps.anl.gov. Uses the standard getData.json endpoint with
robust error handling per PV.

Register in config.yml:
    archiver:
      type: aps_archiver_connector.APSArchiverConnector
      aps_archiver_connector:
        url: https://pvarchiver.aps.anl.gov
        timeout: 60
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import requests

from osprey.connectors.archiver.base import ArchiverConnector, ArchiverMetadata

logger = logging.getLogger(__name__)


def _build_retrieval_url(base_url: str) -> str:
    """Ensure the URL points to the retrieval getData.json endpoint."""
    if "retrieval" not in base_url:
        base_url = base_url.rstrip("/") + "/retrieval"
    return urljoin(base_url.rstrip("/") + "/", "data/getData.json")


def _to_iso8601_utc(dt: datetime) -> str:
    """Convert a datetime to ISO8601 UTC string with trailing Z."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def _parse_series(payload: list[dict]) -> dict[str, pd.Series]:
    """Convert Archiver Appliance JSON response to {pv_name: Series}.

    The archiver returns:
        [{"meta": {"name": "PV"}, "data": [{"secs": N, "nanos": N, "val": V}, ...]}, ...]
    """
    series: dict[str, pd.Series] = {}

    for entry in payload or []:
        pv_name = entry.get("meta", {}).get("name")
        data = entry.get("data", [])

        if not pv_name or not data:
            continue

        index: list[datetime] = []
        values: list = []
        for point in data:
            secs = point.get("secs")
            nanos = point.get("nanos", 0)
            val = point.get("val")
            if secs is None:
                continue
            if isinstance(val, list) and len(val) == 1:
                val = val[0]
            ts = datetime.fromtimestamp(secs + nanos / 1e9, tz=timezone.utc)
            index.append(ts)
            values.append(val)

        if index:
            series[pv_name] = pd.Series(values, index=index, dtype="object")

    return series


def _fetch_data(
    pv_list: list[str],
    start_date: datetime,
    end_date: datetime,
    base_url: str,
    timeout: int = 60,
    processing: str | None = None,
    bin_size: int | None = None,
) -> pd.DataFrame:
    """Fetch data from the Archiver Appliance REST API.

    Tries batch request first, then falls back to individual requests
    for any PVs that failed.
    """
    retrieval_url = _build_retrieval_url(base_url)
    start_iso = _to_iso8601_utc(start_date)
    end_iso = _to_iso8601_utc(end_date)

    session = requests.Session()
    all_series: dict[str, pd.Series] = {}

    # Apply processing operator if requested (e.g., mean_600(PV))
    def _pv_with_processing(pv: str) -> str:
        if processing and processing != "raw" and bin_size:
            return f"{processing}_{bin_size}({pv})"
        return pv

    # --- Batch request (all PVs in one call) ---
    params = [("from", start_iso), ("to", end_iso)]
    for pv in pv_list:
        params.append(("pv", _pv_with_processing(pv)))

    try:
        resp = session.get(retrieval_url, params=params, timeout=timeout, verify=False)
        resp.raise_for_status()
        payload = resp.json()
        all_series = _parse_series(payload)
        logger.debug(f"Batch request: {len(all_series)}/{len(pv_list)} PVs returned data")
    except Exception as e:
        logger.warning(f"Batch archiver request failed: {e}")

    # --- Individual requests for missing PVs ---
    for pv in pv_list:
        if pv in all_series:
            continue
        try:
            resp = session.get(
                retrieval_url,
                params={"from": start_iso, "to": end_iso, "pv": _pv_with_processing(pv)},
                timeout=timeout,
                verify=False,
            )
            resp.raise_for_status()
            payload = resp.json()
            parsed = _parse_series(payload)
            all_series.update(parsed)
        except Exception as e:
            logger.warning(f"Individual request for '{pv}' failed: {e}")

    if not all_series:
        return pd.DataFrame()

    df = pd.DataFrame(all_series)
    df.sort_index(inplace=True)
    return df


class APSArchiverConnector(ArchiverConnector):
    """APS Archiver Appliance connector using direct REST API.

    Works with any standard EPICS Archiver Appliance installation.
    Uses the getData.json endpoint with batch + per-PV fallback.
    """

    def __init__(self):
        self._connected = False
        self._url: str | None = None
        self._timeout: int = 60

    async def connect(self, config: dict[str, Any]) -> None:
        url = config.get("url")
        if not url:
            raise ValueError("archiver URL is required (e.g., https://pvarchiver.aps.anl.gov)")
        self._url = url
        self._timeout = config.get("timeout", 60)
        self._connected = True
        logger.info(f"APS Archiver connector initialized: {url}")

    async def disconnect(self) -> None:
        self._connected = False
        self._url = None
        logger.debug("APS Archiver connector disconnected")

    async def get_data(
        self,
        pv_list: list[str],
        start_date: datetime,
        end_date: datetime,
        precision_ms: int = 1000,
        timeout: int | None = None,
        processing: str | None = None,
        bin_size: int | None = None,
    ) -> pd.DataFrame:
        if not self._connected or not self._url:
            raise RuntimeError("Archiver not connected")

        timeout = timeout or self._timeout

        def fetch():
            return _fetch_data(
                pv_list=pv_list,
                start_date=start_date,
                end_date=end_date,
                base_url=self._url,
                timeout=timeout,
                processing=processing,
                bin_size=bin_size,
            )

        try:
            data = await asyncio.wait_for(asyncio.to_thread(fetch), timeout=timeout + 5)
            logger.debug(f"Retrieved {len(data)} points for {len(pv_list)} PVs")
            return data
        except TimeoutError as e:
            raise TimeoutError(f"Archiver request timed out after {timeout}s") from e
        except Exception as e:
            if "connection" in str(e).lower():
                raise ConnectionError(f"Cannot reach archiver: {e}") from e
            raise

    async def get_metadata(self, pv_name: str) -> ArchiverMetadata:
        return ArchiverMetadata(
            pv_name=pv_name,
            is_archived=True,
            description=f"APS Archived PV: {pv_name}",
        )

    async def check_availability(self, pv_names: list[str]) -> dict[str, bool]:
        return dict.fromkeys(pv_names, True)
