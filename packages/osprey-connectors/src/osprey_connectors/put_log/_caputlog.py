"""Who wrote a channel, from the facility's caPutLog search service.

The service is an Elastic Search-UI proxy, and these points of its query were
measured against one, so they stay as they are:

- The search is ``POST /search``; every GET spelling answers 404.
- Only ``queryConfig.search_fields`` filters by PV. ``state.searchTerm`` alone
  leaves the total unfiltered, and a ``state.filters`` entry on ``pv`` returns
  nothing because the field is text-analysed.
- No time range or sort is proven through the proxy, so ``start_at`` and the
  newest first order are applied here, to the page the service returned.
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
from typing import ClassVar

from pydantic import ConfigDict, ValidationError
from typing_extensions import override

from osprey_connectors._service_http import (
    MALFORMED,
    HttpServiceSettings,
    ReadOnlyTransport,
    Transport,
)
from osprey_connectors.config_sections import ConnectorSettings
from osprey_connectors.errors import require_aware
from osprey_connectors.model import FrozenModel
from osprey_connectors.put_log._contract import PutLog, PutLogEntry
from osprey_connectors.service_answers import (
    BackendSpec,
    LogPage,
    PageOutcome,
    Unavailable,
    _instant,
    _text,
)

_SEARCH_PATH = "/search"
_WHERE = "pv_directory.put_log.settings"
_NO_ROWS_REASON = "no logged puts"

#: The most rows one call returns, because a page ends up in a tool result.
_MAX_ROWS = 200

_EARLIEST = datetime.min.replace(tzinfo=UTC)


class _Envelope(FrozenModel):
    """What an answer is read for: ``results`` must be a list of objects, the rest is ignored."""

    model_config = ConfigDict(extra="ignore")

    results: list[dict[str, object]]
    totalResults: object = None


class CaPutLogSettings(HttpServiceSettings):
    """caPutLog's block, ``pv_directory.put_log.settings``, read."""


class CaPutLog(PutLog):
    """Read one PV's logged writes, its field puts included.

    Args:
        spec: The backend's spec; ``settings`` holds the connection block.
        project_dir: Unused; every backend takes it.
        transport: What sends the requests; by default the read-only one built from
            ``spec.settings``.
    """

    settings_model: ClassVar[type[ConnectorSettings]] = CaPutLogSettings
    SETTINGS_KEY: ClassVar[str] = _WHERE

    ALLOWED_REQUESTS = (("POST", _SEARCH_PATH),)

    def __init__(
        self,
        spec: BackendSpec,
        *,
        project_dir: Path | None = None,
        transport: Transport | None = None,
        **seams: object,
    ) -> None:
        super().__init__(spec, project_dir=project_dir, **seams)
        if transport is None:
            transport = ReadOnlyTransport(
                self._settings(CaPutLogSettings).connection, self.ALLOWED_REQUESTS
            )
        self._transport = transport

    @override
    async def history(
        self,
        address: str,
        *,
        start_at: datetime | None = None,
        limit: int,
        timeout_s: float,
    ) -> LogPage[PutLogEntry] | Unavailable:
        """The logged writes of *address*, newest first within the page the service returned."""
        if start_at is not None:
            require_aware(start_at, name="start_at")
        limit = max(1, min(limit, _MAX_ROWS))
        # One row past the limit, so a full page differs from one that just fills.
        envelope = await self._search(f"{address}.*", limit + 1, timeout_s)
        if isinstance(envelope, Unavailable):
            return envelope

        written = [
            _row(hit, address)
            for hit in envelope.results
            if _is_write_to(_recorded(hit, "pv"), address)
        ]
        if start_at is not None:
            written = [row for row in written if row.put_at is None or row.put_at >= start_at]
        written.sort(key=lambda row: row.put_at or _EARLIEST, reverse=True)

        rows = written[:limit]
        truncated = len(written) > limit or _left_unread(envelope, limit)
        if not rows:
            return self._page("no_match", truncated=truncated, note=_NO_ROWS_REASON)
        return self._page("found", rows=rows, truncated=truncated)

    @override
    async def probe(self, *, timeout_s: float) -> str | Unavailable:
        """Whether a search answers; the root's status is not a request this backend may send."""
        envelope = await self._search("", 1, timeout_s)
        return envelope if isinstance(envelope, Unavailable) else "search answered"

    async def _search(self, term: str, rows: int, timeout_s: float) -> _Envelope | Unavailable:
        body = {
            "state": {"searchTerm": term, "current": 1, "resultsPerPage": rows},
            "queryConfig": {"search_fields": {"pv": {}}},
        }
        answer = await self._transport.request(
            "POST", _SEARCH_PATH, json_body=body, timeout_s=timeout_s
        )
        if isinstance(answer, Unavailable):
            return answer
        try:
            return _Envelope.model_validate(answer)
        except ValidationError:
            return MALFORMED

    def _page(
        self,
        outcome: PageOutcome,
        *,
        rows: list[PutLogEntry] | None = None,
        truncated: bool = False,
        note: str | None = None,
    ) -> LogPage[PutLogEntry]:
        return LogPage(
            outcome=outcome,
            source=self.source,
            rows=tuple(rows or ()),
            truncated=truncated,
            note=note,
            retention_known=False,
        )


def _recorded(hit: Mapping[str, object], name: str) -> object:
    """One result field, out of the ``{"raw": value}`` box the service wraps it in."""
    value = hit.get(name)
    return value["raw"] if isinstance(value, dict) and "raw" in value else value


def _is_write_to(recorded: object, address: str) -> bool:
    return isinstance(recorded, str) and (recorded == address or recorded.startswith(f"{address}."))


def _at(hit: Mapping[str, object]) -> object:
    # The IOC's own stamp is when the put happened; @timestamp is when it was indexed.
    for name in ("ioctimestamp", "@timestamp", "timestamp"):
        if (value := _recorded(hit, name)) is not None:
            return value
    return None


def _left_unread(envelope: _Envelope, limit: int) -> bool:
    read = len(envelope.results)
    total = envelope.totalResults
    return total > read if isinstance(total, int) else read > limit


def _row(hit: Mapping[str, object], address: str) -> PutLogEntry:
    """*hit* as a row; a stamp that cannot be placed, such as IOC local time, stays as text."""
    # A hit is kept only when it records a string pv, so str() changes nothing.
    recorded = str(_recorded(hit, "pv"))
    stamp = _at(hit)
    put_at = _instant(stamp)
    return PutLogEntry(
        put_at=put_at,
        recorded_stamp=_text(stamp) if put_at is None else None,
        address=recorded,
        field=recorded[len(address) :] or None,
        old=_recorded(hit, "old"),
        new=_recorded(hit, "new"),
        user=_text(_recorded(hit, "user")),
        client=_text(_recorded(hit, "client")),
    )
