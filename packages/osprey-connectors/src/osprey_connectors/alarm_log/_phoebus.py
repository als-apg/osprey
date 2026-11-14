"""Read a Phoebus alarm logger (4.7.0 or newer) over its search routes.

``GET /search/alarm?pv=`` answers a bare JSON list, newest first, that interleaves
two kinds of record, told apart by the prefix of their ``config`` path: ``state:``
rows are severity changes, and ``config:`` rows are edits to the alarm tree, whose
``config_msg`` is a JSON string. So one search answers both history and
configuration, and every row keeps its kind, because the two are not one series.

Nothing proves the service's ``pv`` match exact, so a row counts for a PV only if
its tree path ends in exactly that name. The service also ignores parameters it does
not know, and ``start`` has only been seen in its relative spelling, so the window
sent is whole days and the exact one is applied here.

No rows means the logger holds nothing for that name. It does not mean the PV never
alarmed: a facility's legacy alarm system may never reach this index. A root that
answers does not prove ingestion either, since the logger answers with a dead Kafka
consumer. ``/search/alarm/config`` has never been measured answering, so it only
ever runs after the search brought no ``config:`` row, and its failure says nothing
about the tree.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from osprey_connectors._service_http import ReadOnlyTransport
from osprey_connectors.alarm_log._contract import AlarmHistoryRow, AlarmLog
from osprey_connectors.alarm_state import AlarmConfigRecord
from osprey_connectors.service_answers import BackendSpec, HistoryPage, Outcome, Unavailable

_SEARCH_PATH = "/search/alarm"
_CONFIG_PATH = "/search/alarm/config"
_SETTINGS_KEY = "pv_directory.alarm_log.settings"

#: How far back a configuration lookup reads, so a noisy PV's state rows leave room.
_CONFIG_SEARCH_SIZE = 200

_NO_LOGGED_EVENTS = "no logged events"
_MALFORMED_REASON = "malformed response"
_UNESCAPED_SLASH = re.compile(r"(?<!\\)/")
_PV_SCHEME = re.compile(r"^(?:ca|pva)://")


class PhoebusAlarmLogger(AlarmLog):
    """The logged alarm events and last logged alarm configuration of one PV."""

    #: The root is exact because the front end answers 503 without its trailing slash.
    ALLOWED_REQUESTS = (("GET", "/"), ("GET", "/search/*"))

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
            transport = ReadOnlyTransport(spec.settings, self.ALLOWED_REQUESTS, where=_SETTINGS_KEY)
        self._transport = transport

    async def history(
        self, channel: str, since: datetime | None, limit: int, deadline: float
    ) -> HistoryPage[AlarmHistoryRow]:
        # One row more than asked for tells a full page from a longer history.
        answer = await self._search(channel, limit + 1, since, deadline)
        if isinstance(answer, Unavailable):
            return self._page(Outcome.UNAVAILABLE, reason=answer.reason)
        matching = _rows_of(answer, channel)
        rows = [row for row in matching if _in_window(row, since)]
        if not rows:
            return self._page(Outcome.NO_MATCH, reason=_NO_LOGGED_EVENTS)
        # A row older than the window means the page already reached the window's end.
        truncated = len(rows) > limit if len(rows) < len(matching) else len(answer) > limit
        return self._page(
            Outcome.FOUND, rows=[_history_row(row) for row in rows[:limit]], truncated=truncated
        )

    async def config_for(self, channel: str, deadline: float) -> AlarmConfigRecord | Unavailable:
        started = time.monotonic()
        answer = await self._search(channel, _CONFIG_SEARCH_SIZE, None, deadline)
        if isinstance(answer, Unavailable):
            return answer
        rows = _rows_of(answer, channel)
        newest = _newest_config_row(rows)
        if newest is not None:
            return _config_record(newest)
        if not rows:
            return AlarmConfigRecord(in_tree="no")
        # State rows prove a path but not the tree's entry, so nothing below answers "no".
        path = _tree_path(rows[0])
        if path is not None:
            remaining = deadline - (time.monotonic() - started)
            fallback = await self._transport.request(
                "GET", _CONFIG_PATH, params={"config": f"config:{path}"}, deadline=remaining
            )
            if isinstance(fallback, list):
                newest = _newest_config_row(_rows_of(fallback, channel))
                if newest is not None:
                    return _config_record(newest)
        return AlarmConfigRecord(in_tree="unknown", path=path)

    async def probe(self, deadline: float) -> str | Unavailable:
        answer = await self._transport.request("GET", "/", deadline=deadline)
        if isinstance(answer, Unavailable):
            return answer
        if not isinstance(answer, Mapping):
            return Unavailable(_MALFORMED_REASON)
        service = " ".join(str(answer[key]) for key in ("name", "version") if answer.get(key))
        return f"root answered ({service or 'unnamed service'}); this does not prove ingestion"

    async def _search(
        self, pv: str, size: int, since: datetime | None, deadline: float
    ) -> list[Any] | Unavailable:
        params: dict[str, Any] = {"pv": pv, "size": size}
        if since is not None:
            params.update(start=_days_since(since), end="now")
        answer = await self._transport.request(
            "GET", _SEARCH_PATH, params=params, deadline=deadline
        )
        if isinstance(answer, Unavailable | list):
            return answer
        return Unavailable(_MALFORMED_REASON)

    def _page(self, outcome: Outcome, **fields: Any) -> HistoryPage[AlarmHistoryRow]:
        return HistoryPage(outcome=outcome, source=self.source, **fields)


def _rows_of(answer: list[Any], pv: str) -> list[Mapping[str, Any]]:
    return [row for row in answer if isinstance(row, Mapping) and _belongs_to(row, pv)]


def _belongs_to(row: Mapping[str, Any], pv: str) -> bool:
    path = _tree_path(row)
    name = _UNESCAPED_SLASH.split(path)[-1] if path else row.get("pv")
    return isinstance(name, str) and _pv_name(name) == _pv_name(pv)


def _pv_name(tree_name: str) -> str:
    return _PV_SCHEME.sub("", tree_name.replace("\\/", "/"))


def _tree_path(row: Mapping[str, Any]) -> str | None:
    config = row.get("config")
    return (config.partition(":")[2] or None) if isinstance(config, str) else None


def _is_config_row(row: Mapping[str, Any]) -> bool:
    config = row.get("config")
    return isinstance(config, str) and config.startswith("config:")


def _newest_config_row(rows: list[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    return next((row for row in rows if _is_config_row(row)), None)


def _config_record(row: Mapping[str, Any]) -> AlarmConfigRecord:
    """What a ``config:`` row says; one that cannot be read leaves the answer unknown."""
    path = _tree_path(row)
    body = _config_body(row)
    if body is None:
        return AlarmConfigRecord(in_tree="unknown", path=path)
    if "delete" in body:
        return AlarmConfigRecord(in_tree="no", path=path)
    return AlarmConfigRecord(
        in_tree="yes",
        path=path,
        guidance=body.get("guidance"),
        displays=body.get("displays"),
        enabled=body.get("enabled"),
        latching=body.get("latching"),
    )


def _config_body(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    text = row.get("config_msg")
    if not isinstance(text, str | bytes | bytearray):
        return None
    try:
        body = json.loads(text)
    except (ValueError, RecursionError):
        return None
    return body if isinstance(body, Mapping) else None


def _history_row(row: Mapping[str, Any]) -> AlarmHistoryRow:
    if _is_config_row(row):
        body = _config_body(row) or {}
        return AlarmHistoryRow(
            time=row.get("message_time"),
            message=body.get("delete", body.get("description")),
            user=row.get("user"),
            kind="config",
        )
    return AlarmHistoryRow(
        time=row.get("time") or row.get("message_time"),
        severity=row.get("severity"),
        status=row.get("current_message"),
        message=row.get("message"),
        user=row.get("user"),
        kind="state",
    )


def _in_window(row: Mapping[str, Any], since: datetime | None) -> bool:
    """A row whose time cannot be read stays: nothing shows it fell outside the window."""
    if since is None:
        return True
    stamp = row.get("time") or row.get("message_time")
    if not isinstance(stamp, str):
        return True
    try:
        recorded = datetime.fromisoformat(stamp)
    except ValueError:
        return True
    return _aware(recorded) >= _aware(since)


def _days_since(since: datetime) -> str:
    age = (datetime.now(UTC) - _aware(since)).total_seconds()
    return f"{max(1, math.ceil(age / 86400))}days"


def _aware(moment: datetime) -> datetime:
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)
