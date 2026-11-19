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
from typing import ClassVar

from pydantic import ConfigDict, OnErrorOmit, TypeAdapter
from typing_extensions import override

from osprey_connectors._service_http import (
    MALFORMED,
    HttpServiceSettings,
    ReadOnlyTransport,
    Transport,
)
from osprey_connectors.alarm_log._contract import AlarmLog, AlarmLogEntry
from osprey_connectors.alarm_state import AlarmConfiguration, AlarmSeverity
from osprey_connectors.config_sections import ConnectorSettings
from osprey_connectors.errors import require_aware
from osprey_connectors.model import FrozenModel
from osprey_connectors.service_answers import (
    BackendSpec,
    LogPage,
    PageOutcome,
    Unavailable,
    _instant,
    _Recorded,
)

_SEARCH_PATH = "/search/alarm"
_CONFIG_PATH = "/search/alarm/config"
_SETTINGS_KEY = "pv_directory.alarm_log.settings"

#: How far back a configuration lookup reads, so a noisy PV's state rows leave room.
_CONFIG_SEARCH_SIZE = 200

_NO_LOGGED_EVENTS = "no logged events"
_UNESCAPED_SLASH = re.compile(r"(?<!\\)/")
_PV_SCHEME = re.compile(r"^(?:ca|pva)://")


class _LoggedRow(FrozenModel):
    """One row of a search answer, of either kind; a field of another type is not recorded.

    ``time`` and ``message_time`` stay as recorded: a row's time is whichever of the
    two is set, read once as the instant it names (:attr:`event_at`).
    """

    model_config = ConfigDict(extra="ignore")

    config: _Recorded[str] = None
    pv: _Recorded[str] = None
    config_msg: object = None
    time: object = None
    message_time: object = None
    severity: _Recorded[str] = None
    current_message: _Recorded[str] = None
    message: _Recorded[str] = None
    user: _Recorded[str] = None

    @property
    def tree_path(self) -> str | None:
        """The alarm-tree path the ``config`` key names, ``None`` for none."""
        return (self.config.partition(":")[2] or None) if self.config is not None else None

    @property
    def is_config(self) -> bool:
        """Whether this is an edit to the alarm tree rather than a severity change."""
        return self.config is not None and self.config.startswith("config:")

    @property
    def stamp(self) -> object:
        """The row's time as recorded: ``time``, or ``message_time`` when that is unset."""
        return self.time or self.message_time

    @property
    def event_at(self) -> datetime | None:
        """The instant :attr:`stamp` names; ``None`` for one with no zone or not ISO-8601."""
        return _instant(self.stamp)


class _ConfigBody(FrozenModel):
    """A ``config:`` row's ``config_msg``; a ``delete`` key, whatever its value, is a removal."""

    model_config = ConfigDict(extra="ignore")

    delete: _Recorded[str] = None
    description: _Recorded[str] = None
    guidance: object = None
    displays: object = None
    enabled: _Recorded[bool] = None
    latching: _Recorded[bool] = None

    @property
    def deleted(self) -> bool:
        return "delete" in self.model_fields_set


#: A search answer's rows, an entry that is not a JSON object skipped.
_ROWS = TypeAdapter(list[OnErrorOmit[_LoggedRow]])


class PhoebusAlarmLogSettings(HttpServiceSettings):
    """The Phoebus alarm logger's block, ``pv_directory.alarm_log.settings``, read."""


class PhoebusAlarmLogger(AlarmLog):
    """The logged alarm events and last logged alarm configuration of one PV."""

    settings_model: ClassVar[type[ConnectorSettings]] = PhoebusAlarmLogSettings
    SETTINGS_KEY: ClassVar[str] = _SETTINGS_KEY

    #: The root is exact because the front end answers 503 without its trailing slash.
    ALLOWED_REQUESTS = (("GET", "/"), ("GET", "/search/*"))

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
                self._settings(PhoebusAlarmLogSettings).connection, self.ALLOWED_REQUESTS
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
    ) -> LogPage[AlarmLogEntry] | Unavailable:
        if start_at is not None:
            require_aware(start_at, name="start_at")
        # One row more than asked for tells a full page from a longer history.
        answer = await self._search(address, limit + 1, start_at, timeout_s)
        if isinstance(answer, Unavailable):
            return answer
        matching = _rows_of(answer, address)
        rows = [row for row in matching if _in_window(row, start_at)]
        if not rows:
            return self._page("no_match", note=_NO_LOGGED_EVENTS)
        # A row older than the window means the page already reached the window's end.
        truncated = len(rows) > limit if len(rows) < len(matching) else len(answer) > limit
        return self._page(
            "found", rows=[_history_row(row) for row in rows[:limit]], truncated=truncated
        )

    @override
    async def config_for(
        self, address: str, *, timeout_s: float
    ) -> AlarmConfiguration | Unavailable:
        started = time.monotonic()
        answer = await self._search(address, _CONFIG_SEARCH_SIZE, None, timeout_s)
        if isinstance(answer, Unavailable):
            return answer
        rows = _rows_of(answer, address)
        newest = _newest_config_row(rows)
        if newest is not None:
            return _config_record(newest)
        if not rows:
            return AlarmConfiguration(in_tree="no")
        # State rows prove a path but not the tree's entry, so nothing below answers "no".
        path = rows[0].tree_path
        if path is not None:
            remaining = timeout_s - (time.monotonic() - started)
            fallback = await self._transport.request(
                "GET", _CONFIG_PATH, params={"config": f"config:{path}"}, timeout_s=remaining
            )
            if isinstance(fallback, list):
                newest = _newest_config_row(_rows_of(fallback, address))
                if newest is not None:
                    return _config_record(newest)
        return AlarmConfiguration(in_tree="unknown", path=path)

    @override
    async def probe(self, *, timeout_s: float) -> str | Unavailable:
        answer = await self._transport.request("GET", "/", timeout_s=timeout_s)
        if isinstance(answer, Unavailable):
            return answer
        if not isinstance(answer, Mapping):
            return MALFORMED
        service = " ".join(str(answer[key]) for key in ("name", "version") if answer.get(key))
        return f"root answered ({service or 'unnamed service'}); this does not prove ingestion"

    async def _search(
        self, address: str, size: int, start_at: datetime | None, timeout_s: float
    ) -> list[object] | Unavailable:
        params: dict[str, str | int] = {"pv": address, "size": size}
        if start_at is not None:
            params.update(start=_days_since(start_at), end="now")
        answer = await self._transport.request(
            "GET", _SEARCH_PATH, params=params, timeout_s=timeout_s
        )
        if isinstance(answer, Unavailable | list):
            return answer
        return MALFORMED

    def _page(
        self,
        outcome: PageOutcome,
        *,
        rows: list[AlarmLogEntry] | None = None,
        truncated: bool = False,
        note: str | None = None,
    ) -> LogPage[AlarmLogEntry]:
        return LogPage(
            outcome=outcome,
            source=self.source,
            rows=tuple(rows or ()),
            truncated=truncated,
            note=note,
        )


def _rows_of(answer: list[object], address: str) -> list[_LoggedRow]:
    return [row for row in _ROWS.validate_python(answer, strict=True) if _belongs_to(row, address)]


def _belongs_to(row: _LoggedRow, address: str) -> bool:
    path = row.tree_path
    name = _UNESCAPED_SLASH.split(path)[-1] if path else row.pv
    return name is not None and _pv_name(name) == _pv_name(address)


def _pv_name(tree_name: str) -> str:
    return _PV_SCHEME.sub("", tree_name.replace("\\/", "/"))


def _newest_config_row(rows: list[_LoggedRow]) -> _LoggedRow | None:
    return next((row for row in rows if row.is_config), None)


def _config_record(row: _LoggedRow) -> AlarmConfiguration:
    """What a ``config:`` row says; one that cannot be read leaves the answer unknown."""
    path = row.tree_path
    body = _config_body(row)
    if body is None:
        return AlarmConfiguration(in_tree="unknown", path=path)
    if body.deleted:
        return AlarmConfiguration(in_tree="no", path=path)
    return AlarmConfiguration(
        in_tree="yes",
        path=path,
        guidance=body.guidance,
        displays=body.displays,
        enabled=body.enabled,
        latching=body.latching,
    )


def _config_body(row: _LoggedRow) -> _ConfigBody | None:
    text = row.config_msg
    if not isinstance(text, str | bytes | bytearray):
        return None
    try:
        return _ConfigBody.model_validate(json.loads(text))
    except (ValueError, RecursionError):
        return None


def _history_row(row: _LoggedRow) -> AlarmLogEntry:
    if row.is_config:
        body = _config_body(row) or _ConfigBody()
        return AlarmLogEntry(
            event_at=_instant(row.message_time),
            message=body.delete if body.deleted else body.description,
            user=row.user,
            kind="config",
        )
    severity, acknowledged = AlarmSeverity.of_recorded(row.severity)
    return AlarmLogEntry(
        event_at=row.event_at,
        severity=severity,
        acknowledged=acknowledged,
        status=row.current_message,
        message=row.message,
        user=row.user,
        kind="state",
    )


def _in_window(row: _LoggedRow, start_at: datetime | None) -> bool:
    """A row whose time cannot be placed stays: nothing shows it fell outside the window."""
    if start_at is None:
        return True
    event_at = row.event_at
    return event_at is None or event_at >= start_at


def _days_since(start_at: datetime) -> str:
    age = (datetime.now(UTC) - start_at).total_seconds()
    return f"{max(1, math.ceil(age / 86400))}days"
