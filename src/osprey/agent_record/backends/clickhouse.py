"""The ClickHouse backend — the agent record in a ClickHouse store, over HTTP.

The backend talks to the store's HTTP interface through ``httpx`` as one
account, :data:`~osprey.agent_record.credentials.CLICKHOUSE_USER`, which
:mod:`.clickhouse_stage` creates with exactly the grants this module needs.
The tables are declared in :mod:`.clickhouse_ddl`.

**Entries.** The record table is a plain ``MergeTree``, which enforces no
uniqueness, so first write wins is enforced here: appends are serialized in
this process, and before each insert the ``(key, chunk_index)`` pairs of the
batch are looked up across every partition. A pair already stored — on any
arrival day, or earlier in the same batch — is not inserted and is counted as
a duplicate. One process owns the record (the agent-record service), so the
in-process lock is the whole serialization. Bodies are stored as their
canonical JSON text, ``ts`` and ``inserted_at`` as ``DateTime64(6, 'UTC')``;
an entry that arrives without ``inserted_at`` is stamped inside the lock,
each insert strictly later than every stamp before it (``max(previous + 1 µs,
now)``), so stamps follow commit order even when the clock steps back.
The UTC date of ``inserted_at`` is the arrival day and the partition.

**State** is a ``ReplacingMergeTree(version)`` read with ``FINAL``; each put
carries a version above every earlier one from this process.

**Retention** drops whole arrival-day partitions; **export** is the store's
own ``FORMAT Parquet`` of one partition, in the embedded backend's column
order.

Every method raises :class:`~osprey.agent_record.contract.RecordBackendError`
on failure (:class:`~osprey.agent_record.contract.RecordBackendUnavailable`
when the store cannot be reached), except :meth:`ClickHouseBackend.health`,
which reports a failure as ``ok=False``.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx

from osprey.agent_record.backends import clickhouse_ddl as ddl
from osprey.agent_record.contract import (
    ENTRY_COLUMNS,
    STATE_TIME_FIELD,
    AppendResult,
    BackendHealth,
    ConditionOp,
    Entry,
    EntryInvalid,
    Query,
    RecordBackendError,
    RecordBackendUnavailable,
    ScopeFilter,
    decode_state,
    encode_state,
    entry_from_row,
    entry_row,
    next_stamp,
    select_conditions,
    select_order,
    state_time,
)
from osprey.agent_record.credentials import CLICKHOUSE_USER

BACKEND_NAME = "clickhouse"

#: Pairs looked up, and rows inserted, per statement.
_BATCH = 500

#: Seconds one call may take, connect included.
_TIMEOUT_S = 60.0

_SELECT_COLUMNS = ", ".join(f"`{name}`" for name in ENTRY_COLUMNS)
_ENTRIES = ddl.qualified(ddl.ENTRIES_TABLE)
_STATE = ddl.qualified(ddl.STATE_TABLE)

#: Statuses that mean the store is not there to answer, rather than refusing.
_UNAVAILABLE_STATUSES = frozenset({502, 503, 504})

_SETTINGS = {
    "http_wait_end_of_query": "1",
    "output_format_parquet_string_as_string": "1",
    "date_time_output_format": "simple",
}


def _strings(values: Iterable[str]) -> str:
    return "(" + ", ".join(ddl.quote(v) for v in sorted(values)) + ")"


def _wire_time(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S.%f")


def _time_literal(value: datetime) -> str:
    return f"toDateTime64({ddl.quote(_wire_time(value))}, 6, 'UTC')"


def _read_time(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


def _now() -> datetime:
    """The wall clock an insert stamps from."""
    return datetime.now(UTC)


class ClickHouseBackend:
    """:class:`~osprey.agent_record.contract.RecordBackend` on a ClickHouse store.

    Args:
        url: The store's HTTP endpoint, e.g. ``http://agent_record_clickhouse:8123``.
        password: The agent-record account's password.
        user: The account; the staged agent-record account by default.
        client: An ``httpx.Client`` to use instead of one this backend opens
            (and therefore does not close).
    """

    def __init__(
        self,
        url: str,
        password: str,
        *,
        user: str = CLICKHOUSE_USER,
        client: httpx.Client | None = None,
    ) -> None:
        self.url = url.rstrip("/") + "/"
        self._user = user
        self._password = password
        self._owns_client = client is None
        self._client = client if client is not None else httpx.Client(timeout=_TIMEOUT_S)
        self._lock = threading.Lock()
        self._stamped: datetime | None = None
        self._version = 0
        self._closed = False

    # ------------------------------------------------------------------ lifecycle

    def close(self) -> None:
        """Release the HTTP client this backend opened. Idempotent."""
        self._closed = True
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> ClickHouseBackend:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------ transport

    def _post(self, sql: str, what: str, *, data: bytes = b"") -> httpx.Response:
        if self._closed:
            raise RecordBackendUnavailable(f"the agent record store {self.url} is closed")
        content = sql.encode("utf-8") + (b"\n" + data if data else b"")
        try:
            response = self._client.post(
                self.url,
                params=_SETTINGS,
                content=content,
                headers={"X-ClickHouse-User": self._user, "X-ClickHouse-Key": self._password},
            )
        except httpx.HTTPError as exc:
            raise RecordBackendUnavailable(
                f"{what} on the agent record store {self.url} failed: cannot reach it ({exc})"
            ) from exc
        if response.status_code != 200 or response.headers.get("X-ClickHouse-Exception-Code"):
            detail = response.text.strip()[:600]
            error = (
                RecordBackendUnavailable
                if response.status_code in _UNAVAILABLE_STATUSES
                else RecordBackendError
            )
            raise error(
                f"{what} on the agent record store {self.url} failed "
                f"(HTTP {response.status_code}): {detail}"
            )
        return response

    def _rows(self, sql: str, what: str) -> list[dict[str, Any]]:
        response = self._post(sql + "\nFORMAT JSONEachRow", what)
        try:
            return [json.loads(line) for line in response.text.splitlines() if line.strip()]
        except ValueError as exc:
            raise RecordBackendError(f"{what}: the store's answer is not JSON: {exc}") from exc

    # ------------------------------------------------------------------ entries

    def append(self, entries: Sequence[Entry]) -> AppendResult:
        """Store *entries*, first write wins on ``(key, chunk.index)``."""
        unique: dict[tuple[str, int], Entry] = {}
        for entry in entries:
            if not isinstance(entry, Entry):
                raise RecordBackendError(f"append takes Entry objects, not {type(entry).__name__}")
            unique.setdefault(entry.unique_key, entry)
        appended = 0
        pairs = list(unique)
        with self._lock:
            for start in range(0, len(pairs), _BATCH):
                batch = pairs[start : start + _BATCH]
                stored = self._stored_pairs(batch)
                fresh = [unique[pair] for pair in batch if pair not in stored]
                if fresh:
                    now = self._stamp(fresh)
                    data = "".join(json.dumps(self._row(e, now)) + "\n" for e in fresh)
                    self._post(
                        f"INSERT INTO {_ENTRIES} ({_SELECT_COLUMNS}) FORMAT JSONEachRow",
                        "append",
                        data=data.encode("utf-8"),
                    )
                appended += len(fresh)
        return AppendResult(appended=appended, duplicates=len(entries) - appended)

    def _stamp(self, entries: Sequence[Entry]) -> datetime:
        """The ``inserted_at`` of one insert, after every stored one. Under the lock.

        An entry that already carries ``inserted_at`` keeps it; the next stamp
        is after it.
        """
        if self._stamped is None:
            [row] = self._rows(
                f"SELECT toString(maxOrNull(inserted_at)) AS newest FROM {_ENTRIES}",
                "append's stamp",
            )
            newest = row.get("newest")
            self._stamped = _read_time(newest) if newest else None
        now, self._stamped = next_stamp(self._stamped, _now(), entries)
        return now

    def _stored_pairs(self, pairs: Sequence[tuple[str, int]]) -> set[tuple[str, int]]:
        keys = {key for key, _ in pairs}
        rows = self._rows(
            f"SELECT key, chunk_index FROM {_ENTRIES} WHERE key IN {_strings(keys)}",
            "append's existence check",
        )
        return {(str(row["key"]), int(row["chunk_index"])) for row in rows}

    @staticmethod
    def _row(entry: Entry, now: datetime) -> dict[str, Any]:
        row = entry_row(entry, now)
        row["ts"] = _wire_time(row["ts"])
        row["inserted_at"] = _wire_time(row["inserted_at"])
        return row

    def select(self, scope_filter: ScopeFilter, query: Query) -> list[Entry]:
        """The entries inside *scope_filter* that match *query*, ordered by
        ``ts`` then ``inserted_at``."""
        where: list[str] = []
        for condition in select_conditions(scope_filter, query):
            column, value = condition.column, condition.value
            if condition.op is ConditionOp.IN:
                where.append(f"`{column}` IN {_strings(value)}")
            elif condition.op is ConditionOp.STARTS_WITH:
                where.append(f"startsWith({column}, {ddl.quote(value)})")
            else:
                where.append(f"{column} {condition.op} {_time_literal(value)}")
        sql = f"SELECT {_SELECT_COLUMNS} FROM {_ENTRIES}"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += f" ORDER BY {select_order(query)}"
        if query.limit is not None:
            sql += f" LIMIT {max(int(query.limit), 0)}"
        rows = self._rows(sql, "select")
        try:
            return [self._entry(row) for row in rows]
        except (EntryInvalid, ValueError, KeyError, TypeError) as exc:
            raise RecordBackendError(f"a stored entry in {self.url} is unreadable: {exc}") from exc

    @staticmethod
    def _entry(row: Mapping[str, Any]) -> Entry:
        return entry_from_row(
            {**row, "ts": _read_time(row["ts"]), "inserted_at": _read_time(row["inserted_at"])}
        )

    def delete_arrived_before(self, day: date) -> int:
        """Drop every arrival-day partition before *day*; the count of entries dropped."""
        if not isinstance(day, date):
            raise RecordBackendError(f"delete_arrived_before needs a date, not {day!r}")
        with self._lock:
            days = self._rows(
                f"SELECT toString(toDate(inserted_at)) AS day, count() AS n FROM {_ENTRIES}"
                f" WHERE toDate(inserted_at) < toDate({ddl.quote(day.isoformat())})"
                " GROUP BY day ORDER BY day",
                "delete",
            )
            dropped = 0
            for row in days:
                self._post(
                    f"ALTER TABLE {_ENTRIES} DROP PARTITION {ddl.quote(str(row['day']))}",
                    f"delete of arrival day {row['day']}",
                )
                dropped += int(row["n"])
        return dropped

    def export_parquet(self, arrival_day: date, dest: Path) -> int:
        """Write the entries of one arrival day to *dest* as Parquet; the row count.

        *dest* is overwritten, and its parent directory created, as needed.
        """
        if not isinstance(arrival_day, date):
            raise RecordBackendError(f"export_parquet needs a date, not {arrival_day!r}")
        dest = Path(dest)
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RecordBackendError(
                f"cannot create the export directory of {dest}: {exc}"
            ) from exc
        what = f"export of arrival day {arrival_day}"
        day = f"toDate({ddl.quote(arrival_day.isoformat())})"
        with self._lock:
            [count] = self._rows(
                f"SELECT count() AS n FROM {_ENTRIES} WHERE toDate(inserted_at) = {day}", what
            )
            response = self._post(
                f"SELECT {_SELECT_COLUMNS} FROM {_ENTRIES} WHERE toDate(inserted_at) = {day}"
                " ORDER BY inserted_at, key, chunk_index FORMAT Parquet",
                what,
            )
        try:
            dest.write_bytes(response.content)
        except OSError as exc:
            raise RecordBackendError(f"{what}: cannot write {dest}: {exc}") from exc
        return int(count["n"])

    # ------------------------------------------------------------------ state

    def get_state(self, name: str) -> Mapping[str, Any] | None:
        """The last value put under *name*, or ``None``."""
        rows = self._rows(
            f"SELECT value FROM {_STATE} FINAL WHERE name = {ddl.quote(name)}", "state read"
        )
        if not rows:
            return None
        return decode_state(name, rows[0]["value"], self.url)

    def put_state(self, name: str, value: Mapping[str, Any]) -> None:
        """Store *value* under *name*, replacing any earlier value."""
        text = encode_state(name, value)
        with self._lock:
            self._version = max(self._version + 1, time.time_ns())
            row = json.dumps({"name": name, "value": text, "version": self._version})
            self._post(
                f"INSERT INTO {_STATE} (name, value, version) FORMAT JSONEachRow",
                "state write",
                data=row.encode("utf-8") + b"\n",
            )

    def count_state(self, prefix: str, since: datetime | None = None) -> int:
        """How many states are named ``<prefix>…``, stamped at or after *since* if given."""
        sql = f"SELECT count() AS n FROM {_STATE} FINAL WHERE startsWith(name, {ddl.quote(prefix)})"
        if since is not None:
            sql += (
                f" AND JSONExtractString(value, {ddl.quote(STATE_TIME_FIELD)})"
                f" >= {ddl.quote(state_time(since))}"
            )
        rows = self._rows(sql, "state count")
        return int(rows[0]["n"]) if rows else 0

    # ------------------------------------------------------------------ health

    def health(self) -> BackendHealth:
        """The entry count and arrival-day range; ``ok=False`` with the reason on failure."""
        try:
            [row] = self._rows(
                "SELECT count() AS n, toString(minOrNull(toDate(inserted_at))) AS oldest,"
                f" toString(maxOrNull(toDate(inserted_at))) AS newest FROM {_ENTRIES}",
                "health",
            )
        except (RecordBackendError, ValueError) as exc:
            return BackendHealth(ok=False, backend=BACKEND_NAME, detail=f"{self.url}: {exc}")
        return BackendHealth(
            ok=True,
            backend=BACKEND_NAME,
            detail=self.url,
            entries=int(row["n"]),
            oldest_arrival_day=date.fromisoformat(row["oldest"]) if row["oldest"] else None,
            newest_arrival_day=date.fromisoformat(row["newest"]) if row["newest"] else None,
        )
