"""Who wrote a channel, from the facility's caPutLog search service.

The service is an Elastic Search-UI proxy, and these points of its query were
measured against one, so they stay as they are:

- The search is ``POST /search``; every GET spelling answers 404.
- Only ``queryConfig.search_fields`` filters by PV. ``state.searchTerm`` alone
  leaves the total unfiltered, and a ``state.filters`` entry on ``pv`` returns
  nothing because the field is text-analysed.
- No time range or sort is proven through the proxy, so ``since`` and the newest
  first order are applied here, to the page the service returned.
- ``wasSearched`` is false on a success, so the envelope's ``results`` says
  whether this is an answer.

The match is on analysed text, so a page can hold other PVs that share words with
the one asked about. Those rows are not its history and are dropped here.

No rows means no logged put in what the index still holds. Retention is not
known, and a change made by an IOC's own logic or a sequencer leaves no entry, so
it is never evidence that the value did not change.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from osprey_connectors._service_http import ReadOnlyTransport
from osprey_connectors.put_log._contract import PutLog, PutRow
from osprey_connectors.service_answers import BackendSpec, HistoryPage, Outcome, Unavailable

_SEARCH_PATH = "/search"
_WHERE = "pv_directory.put_log.settings"
_NO_ROWS_REASON = "no logged puts"
_MALFORMED_REASON = "malformed response"

#: The most rows one call returns, because a page ends up in a tool result.
_MAX_ROWS = 200

_EARLIEST = datetime.min.replace(tzinfo=UTC)


class CaPutLog(PutLog):
    """Read one PV's logged writes, its field puts included.

    Args:
        spec: The backend's spec; ``settings`` holds the connection block.
        project_dir: Unused; every backend takes it.
        transport: What sends the requests; by default the read-only one built from
            ``spec.settings``.
    """

    ALLOWED_REQUESTS = (("POST", _SEARCH_PATH),)

    def __init__(
        self,
        spec: BackendSpec,
        *,
        project_dir: Path | None = None,
        transport: Any = None,
        **seams: Any,
    ) -> None:
        super().__init__(spec, project_dir=project_dir, **seams)
        if transport is None:
            transport = ReadOnlyTransport(spec.settings, self.ALLOWED_REQUESTS, where=_WHERE)
        self._transport = transport

    async def history(
        self, channel: str, since: datetime | None, limit: int, deadline: float
    ) -> HistoryPage[PutRow]:
        """The logged writes of *channel*, newest first within the page the service returned."""
        limit = max(1, min(limit, _MAX_ROWS))
        # One row past the limit, so a full page differs from one that just fills.
        envelope = await self._search(f"{channel}.*", limit + 1, deadline)
        if isinstance(envelope, Unavailable):
            return self._page(Outcome.UNAVAILABLE, reason=envelope.reason)

        results = envelope["results"]
        hits = [hit for hit in results if _is_write_to(_recorded(hit, "pv"), channel)]
        if since is not None:
            floor = _as_utc(since)
            hits = [hit for hit in hits if (stamp := _stamp(hit)) is None or stamp >= floor]
        hits.sort(key=lambda hit: _stamp(hit) or _EARLIEST, reverse=True)

        rows = [_row(hit, channel) for hit in hits[:limit]]
        truncated = len(hits) > limit or _left_unread(envelope, limit)
        if not rows:
            return self._page(Outcome.NO_MATCH, truncated=truncated, reason=_NO_ROWS_REASON)
        return self._page(Outcome.FOUND, rows=rows, truncated=truncated)

    async def probe(self, deadline: float) -> str | Unavailable:
        """Whether a search answers; the root's status is not a request this backend may send."""
        envelope = await self._search("", 1, deadline)
        return envelope if isinstance(envelope, Unavailable) else "search answered"

    async def _search(
        self, term: str, rows: int, deadline: float
    ) -> Mapping[str, Any] | Unavailable:
        body = {
            "state": {"searchTerm": term, "current": 1, "resultsPerPage": rows},
            "queryConfig": {"search_fields": {"pv": {}}},
        }
        answer: Any = await self._transport.request(
            "POST", _SEARCH_PATH, json_body=body, deadline=deadline
        )
        if isinstance(answer, Unavailable):
            return answer
        if not isinstance(answer, dict):
            return Unavailable(_MALFORMED_REASON)
        results = answer.get("results")
        if not isinstance(results, list) or not all(isinstance(hit, dict) for hit in results):
            return Unavailable(_MALFORMED_REASON)
        return answer

    def _page(
        self,
        outcome: Outcome,
        *,
        rows: list[PutRow] | None = None,
        truncated: bool = False,
        reason: str | None = None,
    ) -> HistoryPage[PutRow]:
        return HistoryPage(
            outcome=outcome,
            source=self.source,
            rows=rows or [],
            truncated=truncated,
            retention_known=False,
            reason=reason,
        )


def _recorded(hit: Mapping[str, Any], name: str) -> Any:
    """One result field, out of the ``{"raw": value}`` box the service wraps it in."""
    value = hit.get(name)
    return value["raw"] if isinstance(value, dict) and "raw" in value else value


def _is_write_to(recorded: Any, pv: str) -> bool:
    return isinstance(recorded, str) and (recorded == pv or recorded.startswith(f"{pv}."))


def _at(hit: Mapping[str, Any]) -> Any:
    # The IOC's own stamp is when the put happened; @timestamp is when it was indexed.
    for name in ("ioctimestamp", "@timestamp", "timestamp"):
        if (value := _recorded(hit, name)) is not None:
            return value
    return None


def _stamp(hit: Mapping[str, Any]) -> datetime | None:
    """The instant of *hit*, or ``None`` for a time this cannot place, such as IOC local time."""
    value = _at(hit)
    if not isinstance(value, str):
        return None
    try:
        return _as_utc(datetime.fromisoformat(value))
    except ValueError:
        return None


def _as_utc(moment: datetime) -> datetime:
    # The index stamps every instant in UTC, so a time with no zone is read as one.
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)


def _left_unread(envelope: Mapping[str, Any], limit: int) -> bool:
    read = len(envelope["results"])
    total = envelope.get("totalResults")
    return total > read if isinstance(total, int) else read > limit


def _row(hit: Mapping[str, Any], channel: str) -> PutRow:
    recorded = _recorded(hit, "pv")
    return PutRow(
        at=_at(hit),
        channel=recorded,
        field=recorded[len(channel) :] or None,
        old=_recorded(hit, "old"),
        new=_recorded(hit, "new"),
        user=_recorded(hit, "user"),
        client=_recorded(hit, "client"),
    )
