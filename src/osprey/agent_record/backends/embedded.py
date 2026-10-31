"""The embedded backend — the whole agent record in one DuckDB file.

This is the default backend: it needs no store container, only a file whose
path the caller supplies.

**One owner.** Exactly one :class:`EmbeddedBackend` holds the file open for
writing. A second opener — another process, or a second instance in the same
process — is refused at once with :class:`RecordBackendLocked` naming the
file. DuckDB's own cross-process file lock fails fast with an ``IOException``;
that is mapped here, and an in-process registry covers the case DuckDB does
not (two connections in one process silently share one database).

**Entries** live in ``entries`` with ``PRIMARY KEY (key, chunk_index)``.
Append is first write wins: a pair that already exists — stored earlier, on
any arrival day, or earlier in the same batch — is not stored and is counted
as a duplicate. Bodies are stored as their canonical JSON text
(:func:`~osprey.agent_record.contract.serialize_body`), so chunk reassembly is
byte-identical. ``ts`` and ``inserted_at`` are ``TIMESTAMPTZ``; an entry that
arrives without ``inserted_at`` is stamped inside the append lock, strictly
later than every stamp before it (``max(previous + 1 µs, now)``), so stamps
follow commit order even when the clock steps back. The UTC date
of ``inserted_at`` is the arrival day, the unit of retention and export.

**State** lives in a separate ``state`` table keyed by name, last write wins.
It never appears in :meth:`EmbeddedBackend.select`, the entry count or an
export, and retention never deletes it.

**Scope** is the reader's job: :meth:`EmbeddedBackend.select` only narrows
rows by the :class:`~osprey.agent_record.contract.ScopeFilter` it is handed.

Every method raises :class:`~osprey.agent_record.contract.RecordBackendError`
on failure, except :meth:`EmbeddedBackend.health`, which reports a failure as
``ok=False``.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import duckdb

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

BACKEND_NAME = "embedded"

#: Rows per INSERT statement; bounds the parameter count of one statement.
_INSERT_BATCH = 500


def _now() -> datetime:
    """The wall clock an append stamps from."""
    return datetime.now(UTC)


_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS entries (
        key VARCHAR NOT NULL,
        chunk_index INTEGER NOT NULL,
        chunk_total INTEGER NOT NULL,
        kind VARCHAR NOT NULL,
        ts TIMESTAMPTZ NOT NULL,
        identity VARCHAR NOT NULL,
        on_behalf_of VARCHAR NOT NULL,
        session_id VARCHAR NOT NULL,
        terminal_session VARCHAR NOT NULL,
        run_id VARCHAR NOT NULL,
        agent_id VARCHAR NOT NULL,
        parent_key VARCHAR NOT NULL,
        tool_use_id VARCHAR NOT NULL,
        body VARCHAR NOT NULL,
        inserted_at TIMESTAMPTZ NOT NULL,
        arrival_day DATE NOT NULL,
        PRIMARY KEY (key, chunk_index)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS state (
        name VARCHAR PRIMARY KEY,
        value VARCHAR NOT NULL
    )
    """,
)

_ROW_PLACEHOLDERS = "(" + ", ".join("?" for _ in range(len(ENTRY_COLUMNS) + 1)) + ")"
_INSERT_PREFIX = f"INSERT INTO entries ({', '.join(ENTRY_COLUMNS)}, arrival_day) VALUES "
_SELECT_COLUMNS = ", ".join(ENTRY_COLUMNS)

_OPEN_PATHS: set[Path] = set()
_OPEN_PATHS_LOCK = threading.Lock()


class RecordBackendLocked(RecordBackendUnavailable):
    """The record file is already held open by another owner."""

    def __init__(self, path: Path, detail: str = "") -> None:
        self.path = path
        message = f"the agent record file {path} is already open by another process"
        if detail:
            message += f" ({detail})"
        super().__init__(message)


def _is_lock_error(exc: duckdb.Error) -> bool:
    return isinstance(exc, duckdb.IOException) and "lock" in str(exc).lower()


def _sql_string(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


class EmbeddedBackend:
    """:class:`~osprey.agent_record.contract.RecordBackend` on one DuckDB file.

    Opening creates the file, its parent directory and the tables if they do
    not exist. Use as a context manager, or call :meth:`close`.

    Raises:
        RecordBackendLocked: If the file is already open by another owner.
        RecordBackendUnavailable: If the file cannot be opened as a record.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._key = self.path.resolve()
        self._lock = threading.Lock()
        self._stamped: datetime | None = None
        self._con: duckdb.DuckDBPyConnection | None = None
        with _OPEN_PATHS_LOCK:
            if self._key in _OPEN_PATHS:
                raise RecordBackendLocked(self.path, "already open in this process")
            _OPEN_PATHS.add(self._key)
        try:
            self._con = self._connect()
        except BaseException:
            with _OPEN_PATHS_LOCK:
                _OPEN_PATHS.discard(self._key)
            raise

    def _connect(self) -> duckdb.DuckDBPyConnection:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RecordBackendUnavailable(
                f"cannot create the directory of the agent record file {self.path}: {exc}"
            ) from exc
        try:
            con = duckdb.connect(str(self.path))
        except duckdb.Error as exc:
            if _is_lock_error(exc):
                raise RecordBackendLocked(self.path, str(exc)) from exc
            raise RecordBackendUnavailable(
                f"cannot open the agent record file {self.path}: {exc}"
            ) from exc
        try:
            con.execute("SET TimeZone = 'UTC'")
            for statement in _SCHEMA:
                con.execute(statement)
        except duckdb.Error as exc:
            con.close()
            raise RecordBackendUnavailable(
                f"cannot prepare the agent record file {self.path}: {exc}"
            ) from exc
        return con

    # ------------------------------------------------------------------ lifecycle

    def close(self) -> None:
        """Release the file. Idempotent."""
        with self._lock:
            con, self._con = self._con, None
        if con is None:
            return
        try:
            con.close()
        finally:
            with _OPEN_PATHS_LOCK:
                _OPEN_PATHS.discard(self._key)

    def __enter__(self) -> EmbeddedBackend:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _connection(self) -> duckdb.DuckDBPyConnection:
        if self._con is None:
            raise RecordBackendUnavailable(f"the agent record file {self.path} is closed")
        return self._con

    # ------------------------------------------------------------------ entries

    def append(self, entries: Sequence[Entry]) -> AppendResult:
        """Store *entries*, first write wins on ``(key, chunk.index)``."""
        unique: dict[tuple[str, int], Entry] = {}
        for entry in entries:
            if not isinstance(entry, Entry):
                raise RecordBackendError(f"append takes Entry objects, not {type(entry).__name__}")
            unique.setdefault(entry.unique_key, entry)
        appended = 0
        with self._lock:
            con = self._connection()
            try:
                now = self._stamp(con, unique.values())
                rows = [self._row(entry, now) for entry in unique.values()]
                con.execute("BEGIN TRANSACTION")
                try:
                    for start in range(0, len(rows), _INSERT_BATCH):
                        batch = rows[start : start + _INSERT_BATCH]
                        sql = (
                            _INSERT_PREFIX
                            + ", ".join(_ROW_PLACEHOLDERS for _ in batch)
                            + " ON CONFLICT DO NOTHING"
                        )
                        params = [value for row in batch for value in row]
                        result = con.execute(sql, params).fetchone()
                        appended += int(result[0]) if result else 0
                    con.execute("COMMIT")
                except BaseException:
                    con.execute("ROLLBACK")
                    raise
            except duckdb.Error as exc:
                raise RecordBackendError(f"append to {self.path} failed: {exc}") from exc
        return AppendResult(appended=appended, duplicates=len(entries) - appended)

    def _stamp(self, con: duckdb.DuckDBPyConnection, entries: Iterable[Entry]) -> datetime:
        """The ``inserted_at`` of this append, after every stored one. Under the lock.

        An entry that already carries ``inserted_at`` keeps it; the next stamp
        is after it.
        """
        if self._stamped is None:
            [newest] = con.execute("SELECT max(inserted_at) FROM entries").fetchone() or [None]
            self._stamped = newest.astimezone(UTC) if newest is not None else None
        now, self._stamped = next_stamp(self._stamped, _now(), entries)
        return now

    @staticmethod
    def _row(entry: Entry, now: datetime) -> list[Any]:
        row = entry_row(entry, now)
        return [*(row[name] for name in ENTRY_COLUMNS), row["inserted_at"].astimezone(UTC).date()]

    def select(self, scope_filter: ScopeFilter, query: Query) -> list[Entry]:
        """The entries inside *scope_filter* that match *query*, ordered by
        ``ts`` then ``inserted_at``."""
        where: list[str] = []
        params: list[Any] = []
        for condition in select_conditions(scope_filter, query):
            column, value = condition.column, condition.value
            if condition.op is ConditionOp.IN:
                where.append(f"{column} IN ({', '.join('?' for _ in value)})")
                params.extend(sorted(value))
            elif condition.op is ConditionOp.STARTS_WITH:
                where.append(f"starts_with({column}, ?)")
                params.append(value)
            else:
                where.append(f"{column} {condition.op} ?")
                params.append(value)
        sql = f"SELECT {_SELECT_COLUMNS} FROM entries"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += f" ORDER BY {select_order(query)}"
        if query.limit is not None:
            sql += " LIMIT ?"
            params.append(max(int(query.limit), 0))
        with self._lock:
            con = self._connection()
            try:
                rows = con.execute(sql, params).fetchall()
            except duckdb.Error as exc:
                raise RecordBackendError(f"select from {self.path} failed: {exc}") from exc
        try:
            return [self._entry(row) for row in rows]
        except (EntryInvalid, ValueError) as exc:
            raise RecordBackendError(f"a stored entry in {self.path} is unreadable: {exc}") from exc

    @staticmethod
    def _entry(row: tuple[Any, ...]) -> Entry:
        return entry_from_row(dict(zip(ENTRY_COLUMNS, row, strict=True)))

    def delete_arrived_before(self, day: date) -> int:
        """Drop every entry whose arrival day is before *day*; the count dropped."""
        with self._lock:
            con = self._connection()
            try:
                result = con.execute("DELETE FROM entries WHERE arrival_day < ?", [day]).fetchone()
            except duckdb.Error as exc:
                raise RecordBackendError(f"delete from {self.path} failed: {exc}") from exc
        return int(result[0]) if result else 0

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
        # COPY takes no bound parameters: the day comes from date.isoformat()
        # and the path is quoted as an SQL string literal.
        sql = (
            f"COPY (SELECT {_SELECT_COLUMNS} FROM entries"
            f" WHERE arrival_day = DATE '{arrival_day.isoformat()}'"
            " ORDER BY inserted_at, key, chunk_index)"
            f" TO {_sql_string(str(dest))} (FORMAT PARQUET)"
        )
        with self._lock:
            con = self._connection()
            try:
                result = con.execute(sql).fetchone()
            except duckdb.Error as exc:
                raise RecordBackendError(
                    f"export of arrival day {arrival_day} from {self.path} failed: {exc}"
                ) from exc
        return int(result[0]) if result else 0

    # ------------------------------------------------------------------ state

    def get_state(self, name: str) -> Mapping[str, Any] | None:
        """The last value put under *name*, or ``None``."""
        with self._lock:
            con = self._connection()
            try:
                row = con.execute("SELECT value FROM state WHERE name = ?", [name]).fetchone()
            except duckdb.Error as exc:
                raise RecordBackendError(f"state read from {self.path} failed: {exc}") from exc
        if row is None:
            return None
        return decode_state(name, row[0], self.path)

    def put_state(self, name: str, value: Mapping[str, Any]) -> None:
        """Store *value* under *name*, replacing any earlier value."""
        text = encode_state(name, value)
        with self._lock:
            con = self._connection()
            try:
                con.execute(
                    "INSERT INTO state (name, value) VALUES (?, ?)"
                    " ON CONFLICT (name) DO UPDATE SET value = excluded.value",
                    [name, text],
                )
            except duckdb.Error as exc:
                raise RecordBackendError(f"state write to {self.path} failed: {exc}") from exc

    def count_state(self, prefix: str, since: datetime | None = None) -> int:
        """How many states are named ``<prefix>…``, stamped at or after *since* if given."""
        sql = "SELECT count(*) FROM state WHERE starts_with(name, ?)"
        params: list[Any] = [prefix]
        if since is not None:
            sql += f" AND coalesce(json_extract_string(value, '$.{STATE_TIME_FIELD}'), '') >= ?"
            params.append(state_time(since))
        with self._lock:
            con = self._connection()
            try:
                row = con.execute(sql, params).fetchone()
            except duckdb.Error as exc:
                raise RecordBackendError(f"state count in {self.path} failed: {exc}") from exc
        return int(row[0]) if row else 0

    # ------------------------------------------------------------------ health

    def health(self) -> BackendHealth:
        """The entry count and arrival-day range; ``ok=False`` with the reason on failure."""
        detail = str(self.path)
        try:
            with self._lock:
                con = self._connection()
                row = con.execute(
                    "SELECT count(*), min(arrival_day), max(arrival_day) FROM entries"
                ).fetchone()
        except (RecordBackendError, duckdb.Error) as exc:
            return BackendHealth(ok=False, backend=BACKEND_NAME, detail=f"{detail}: {exc}")
        count, oldest, newest = row or (0, None, None)
        return BackendHealth(
            ok=True,
            backend=BACKEND_NAME,
            detail=detail,
            entries=int(count),
            oldest_arrival_day=oldest,
            newest_arrival_day=newest,
        )
