"""
APS Archiver Appliance retrieval helper.

Provides a minimal wrapper around the Archiver Appliance REST API using
`data/getData.json` and returns a pandas DataFrame indexed by UTC timestamps.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional
from urllib.parse import urljoin

import pandas as pd
import requests
import zoneinfo

logger = logging.getLogger(__name__)


class ArchiverRequestError(RuntimeError):
    """Raised when the Archiver Appliance request fails."""


def _to_iso8601_utc(dt: datetime, local_timezone: str = "America/Chicago") -> str:
    """
    Convert a datetime to ISO8601 UTC (with trailing Z).
    Assumes provided local timezone if naive (default America/Chicago).
    """
    try:
        local_tz = zoneinfo.ZoneInfo(local_timezone)
    except Exception:
        local_tz = timezone.utc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=local_tz)
    dt_utc = dt.astimezone(timezone.utc)
    return dt_utc.isoformat().replace("+00:00", "Z")


def _build_retrieval_url(base_url: str) -> str:
    """Ensure the URL points to the retrieval endpoint."""
    if not base_url:
        raise ValueError("Archiver base URL is required")
    # Common configs may omit the trailing /retrieval
    if "retrieval" not in base_url:
        base_url = base_url.rstrip("/") + "/retrieval"
    return urljoin(base_url.rstrip("/") + "/", "data/getData.json")


def _parse_series(payload: List[dict]) -> Dict[str, pd.Series]:
    """Convert Archiver JSON payload to {pv: Series}."""
    series: Dict[str, pd.Series] = {}
    for entry in payload or []:
        pv_name = entry.get("meta", {}).get("name")
        data = entry.get("data", [])
        if not pv_name or not data:
            continue

        index: List[datetime] = []
        values: List = []
        for point in data:
            secs = point.get("secs")
            nanos = point.get("nanos", 0)
            val = point.get("val")
            if secs is None:
                continue
            # Some values are wrapped in single-item lists
            if isinstance(val, list) and len(val) == 1:
                val = val[0]
            ts = datetime.fromtimestamp(secs + nanos / 1e9, tz=timezone.utc)
            index.append(ts)
            values.append(val)

        if index:
            series[pv_name] = pd.Series(values, index=index, dtype="object")
    return series


def fetch_archiver_history(
    pv_list: Iterable[str],
    start_date: datetime,
    end_date: datetime,
    base_url: str,
    timeout: int = 30,
    verify_ssl: bool = False,
    local_timezone: str = "America/Chicago",
) -> pd.DataFrame:
    """
    Fetch historical PV data from the APS Archiver Appliance.

    Args:
        pv_list: Iterable of PV names.
        start_date: Start datetime (naive assumed America/Chicago).
        end_date: End datetime (naive assumed America/Chicago).
        base_url: Base archiver URL (with or without /retrieval).
        timeout: Request timeout seconds.
        verify_ssl: Whether to verify TLS certificates (defaults False for internal endpoints).
        local_timezone: Name of local timezone to convert results to (defaults to America/Chicago).

    Returns:
        pandas.DataFrame indexed by local timezone timestamps with PV columns.

    Raises:
        ArchiverRequestError on HTTP or parsing errors.
    """
    pv_list = list(pv_list)
    if not pv_list:
        return pd.DataFrame()

    retrieval_url = _build_retrieval_url(base_url)
    start_iso = _to_iso8601_utc(start_date, local_timezone)
    end_iso = _to_iso8601_utc(end_date, local_timezone)

    # Log the exact time window and conversion being used for the query
    logger.info(
        "APS archiver query: PVs=%s, local_start=%s, local_end=%s, "
        "utc_start=%s, utc_end=%s, local_timezone=%s, retrieval_url=%s",
        pv_list,
        start_date,
        end_date,
        start_iso,
        end_iso,
        local_timezone,
        retrieval_url,
    )

    session = requests.Session()
    session.trust_env = False
    session.verify = verify_ssl

    all_series: Dict[str, pd.Series] = {}

    for pv in pv_list:
        params = {"pv": pv, "from": start_iso, "to": end_iso}
        try:
            resp = session.get(retrieval_url, params=params, timeout=(1, timeout))
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            raise ArchiverRequestError(f"Failed to retrieve data for PV '{pv}': {exc}") from exc

        pv_series = _parse_series(payload)
        if pv in pv_series:
            all_series[pv] = pv_series[pv]
        elif pv_series:
            # If the archiver returns a canonical PV name different from the request
            all_series.update(pv_series)

    if not all_series:
        return pd.DataFrame()

    df = pd.DataFrame(all_series)
    df.sort_index(inplace=True)

    # Convert index to local timezone for downstream plotting/reporting
    try:
        tz = zoneinfo.ZoneInfo(local_timezone)
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize(timezone.utc).tz_convert(tz)
        else:
            df.index = df.index.tz_convert(tz)
    except Exception as exc:
        logger.warning(f"Failed to convert archiver timestamps to {local_timezone}: {exc}")

    return df
