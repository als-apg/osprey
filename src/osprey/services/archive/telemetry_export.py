"""Export completed days of the telemetry store into the archive as Parquet.

A UTC day ``D`` is exported once, by the first pass at or after ``D+1 00:10Z``,
for every completed day not yet recorded in a manifest, walking back at most
``backfill_days``. Every OSPREY table is read by **arrival day** —
``toDate(InsertedAt, 'UTC') = D`` — so a row that reached the store late (after
an outage) is exported with the day it arrived, and every stored row is
exported exactly once. The store's ``system.query_log`` is read by its own
``event_date``. Each table becomes ``<pass-day>/telemetry/<D>/<table>.parquet``
with a manifest line carrying its table, day, row count and sha256.

Every file of a day is staged before any is placed, and the day is recorded
only when all of them were read; a failure leaves the day for the next pass.
A store busy with other queries (``Code: 202``, too many simultaneous queries)
is retried with backoff before the day counts as failed.

The store is read as the ``archive`` account, whose password comes only from
the environment (``OSPREY_TELEMETRY_ARCHIVE_PASSWORD``, delivered by the
account's own credential file) — never from a flag, whose value would be
visible in the process list. The account is read-only on the store's side.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from osprey.deployment.telemetry_config import DEFAULT_RETENTION_DAYS
from osprey.deployment.telemetry_credentials import ARCHIVE_PASSWORD_ENV
from osprey.deployment.telemetry_store import ddl
from osprey.deployment.telemetry_store.accounts import ARCHIVE_SECONDS, ARCHIVE_USER
from osprey.services.archive.manifest import ArchiveState

if TYPE_CHECKING:
    from osprey.services.archive.run import ArchiveTree

#: The file stem and rows key of the store's own query log.
QUERY_LOG = "query_log"
#: Every table a day exports, in order: the OSPREY tables, then the query log.
EXPORT_TABLES: tuple[str, ...] = (*(table.name for table in ddl.TABLES), QUERY_LOG)
#: How long after midnight a day is left for the query log's flush.
LATE_GRACE = timedelta(minutes=10)
#: Most completed days a pass exports: the store's default retention.
DEFAULT_BACKFILL_DAYS = DEFAULT_RETENTION_DAYS
#: Where a day's files go, below the pass day directory.
ARCHIVE_SUBDIR = "telemetry"

_BUSY_CODE = "202"
_RETRIES = 5
_SHOWN_CHARS = 300
_PARQUET_MAGIC = b"PAR1"


class TelemetryExportError(Exception):
    """One day could not be exported."""


def pending_days(now: datetime, exported: set[str], backfill_days: int) -> list[date]:
    """The completed days not yet exported, oldest first, at most *backfill_days* back."""
    latest = (now.astimezone(UTC) - LATE_GRACE).date() - timedelta(days=1)
    days = [latest - timedelta(days=back) for back in range(max(backfill_days, 0))]
    return sorted(d for d in days if d.isoformat() not in exported)


def _source(table: str) -> tuple[str, str]:
    """The ``FROM`` target of *table* and the expression that dates its rows."""
    if table == QUERY_LOG:
        return "`system`.`query_log`", "event_date"
    if table not in ddl.TABLES_BY_NAME:
        raise ValueError(f"{table!r} is not a telemetry store table")
    return f"`{ddl.DATABASE}`.`{table}`", "toDate(InsertedAt, 'UTC')"


def render_export_sql(table: str, day: date) -> str:
    """The statement that reads one day of *table* as Parquet."""
    source, dated = _source(table)
    return f"SELECT * FROM {source} WHERE {dated} = '{day.isoformat()}' FORMAT Parquet"


def render_count_sql(table: str, day: date) -> str:
    """The statement that counts the rows :func:`render_export_sql` reads."""
    source, dated = _source(table)
    return f"SELECT count() FROM {source} WHERE {dated} = '{day.isoformat()}' FORMAT TSV"


def _busy(response: httpx.Response, body: str) -> bool:
    return (
        response.headers.get("X-ClickHouse-Exception-Code") == _BUSY_CODE
        or f"Code: {_BUSY_CODE}." in body
    )


@dataclass
class TelemetryExporter:
    """Reads the telemetry store's HTTP interface as the ``archive`` account."""

    url: str
    backfill_days: int = DEFAULT_BACKFILL_DAYS
    environ: Mapping[str, str] = field(default_factory=lambda: os.environ)
    transport: httpx.BaseTransport | None = None
    # The store stops an archive query at ARCHIVE_SECONDS; wait a little longer.
    timeout: float = ARCHIVE_SECONDS + 60.0
    sleep: Callable[[float], None] = time.sleep

    def _client(self) -> httpx.Client:
        password = self.environ.get(ARCHIVE_PASSWORD_ENV)
        if not password:
            raise TelemetryExportError(
                f"{ARCHIVE_PASSWORD_ENV} must be set to read the telemetry store"
            )
        return httpx.Client(
            base_url=self.url.rstrip("/"),
            auth=(ARCHIVE_USER, password),
            timeout=self.timeout,
            transport=self.transport,
        )

    def _query(self, client: httpx.Client, sql: str) -> Iterator[bytes]:
        """The body of *sql*, retried while the store answers 202 (busy)."""
        for attempt in range(_RETRIES + 1):
            with client.stream("POST", "/", content=sql.encode("utf-8")) as response:
                if response.status_code == 200:
                    yield from response.iter_bytes()
                    return
                body = response.read().decode("utf-8", errors="replace")
                if not _busy(response, body) or attempt == _RETRIES:
                    raise TelemetryExportError(
                        f"store answered {response.status_code}: {body[:_SHOWN_CHARS].strip()}"
                    )
            self.sleep(0.5 * 2**attempt)

    def _count(self, client: httpx.Client, table: str, day: date) -> int:
        text = b"".join(self._query(client, render_count_sql(table, day))).decode().strip()
        try:
            return int(text)
        except ValueError as exc:
            raise TelemetryExportError(f"count of {table} is not a number: {text!r}") from exc

    def _parquet(self, client: httpx.Client, table: str, day: date) -> Iterator[bytes]:
        """The Parquet body of one table's day, refused unless it is a whole file."""
        head = b""
        tail = b""
        for chunk in self._query(client, render_export_sql(table, day)):
            if len(head) < len(_PARQUET_MAGIC):
                head = (head + chunk)[: len(_PARQUET_MAGIC)]
            tail = (tail + chunk)[-len(_PARQUET_MAGIC) :]
            yield chunk
        if head != _PARQUET_MAGIC or tail != _PARQUET_MAGIC:
            raise TelemetryExportError(f"{table} for {day.isoformat()} is not a whole Parquet file")

    def _export_day(
        self, client: httpx.Client, tree: ArchiveTree, day: date
    ) -> tuple[dict[str, int], list[tuple[Path, int, str, str, str, int]]]:
        rows: dict[str, int] = {}
        staged: list[tuple[Path, int, str, str, str, int]] = []
        try:
            for table in EXPORT_TABLES:
                # The day is closed (arrival day, or the query log past its
                # flush), so the count and the export read the same rows.
                count = self._count(client, table, day)
                temp, size, sha = tree.stage(self._parquet(client, table, day))
                rel = f"{ARCHIVE_SUBDIR}/{day.isoformat()}/{table}.parquet"
                staged.append((temp, size, sha, rel, table, count))
                rows[table] = count
        except BaseException:
            for temp, *_ in staged:
                tree.discard(temp)
            raise
        return rows, staged

    @staticmethod
    def _place_day(
        tree: ArchiveTree,
        label: str,
        rows: dict[str, int],
        staged: list[tuple[Path, int, str, str, str, int]],
    ) -> None:
        """Place a staged day and record it; a failure discards what is still staged.

        One manifest line per file, then the day line last, so a day that fails
        part-way stays unrecorded and the next pass exports it again.
        """
        pending = list(staged)
        try:
            while pending:
                temp, size, sha, rel, table, count = pending[0]
                path = tree.place(temp, rel)
                pending.pop(0)
                tree.append(
                    {
                        "kind": "file",
                        "source": rel,
                        "path": path,
                        "size": size,
                        "sha256": sha,
                        "source_mtime_ns": None,
                        "table": table,
                        "day": label,
                        "rows": count,
                        "archived_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                    }
                )
            tree.append({"kind": "telemetry_day", "day": label, "rows": rows})
        except BaseException:
            for temp, *_ in pending:
                tree.discard(temp)
            raise

    def export(
        self, tree: ArchiveTree, state: ArchiveState, *, now: datetime
    ) -> tuple[list[str], list[dict[str, str]]]:
        """Export every pending day; return the days recorded and the errors met."""
        days = pending_days(now, state.telemetry_days, self.backfill_days)
        if not days:
            return [], []
        exported: list[str] = []
        errors: list[dict[str, str]] = []
        try:
            client = self._client()
        except TelemetryExportError as exc:
            return [], [{"source": ARCHIVE_SUBDIR, "error": str(exc)}]
        with client:
            for day in days:
                label = day.isoformat()
                source = f"{ARCHIVE_SUBDIR}/{label}"
                try:
                    rows, staged = self._export_day(client, tree, day)
                except (httpx.HTTPError, OSError, TelemetryExportError) as exc:
                    errors.append({"source": source, "error": str(exc)})
                    continue
                try:
                    self._place_day(tree, label, rows, staged)
                except OSError as exc:
                    errors.append({"source": source, "error": str(exc)})
                    continue
                state.telemetry_days.add(label)
                exported.append(label)
        return exported, errors
