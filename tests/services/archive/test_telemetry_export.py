"""Completed days of the telemetry store are exported once, per table, as Parquet."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pytest

from osprey.deployment.telemetry_store import ddl
from osprey.services.archive import run as archive_run
from osprey.services.archive.manifest import MANIFEST_NAME, load_state
from osprey.services.archive.run import run_pass
from osprey.services.archive.telemetry_export import (
    EXPORT_TABLES,
    QUERY_LOG,
    TelemetryExporter,
    pending_days,
    render_count_sql,
    render_export_sql,
)

ENV = {"OSPREY_TELEMETRY_ARCHIVE_PASSWORD": "archive-secret"}
# 2026-09-24 02:00Z: 2026-09-23 is complete and past the grace.
NOW = datetime(2026, 9, 24, 2, 0, 0, tzinfo=UTC)
DAY = date(2026, 9, 23)

_FROM = re.compile(r"FROM `(?P<db>[^`]+)`\.`(?P<table>[^`]+)`")


def _parquet(table: str, rows: int) -> bytes:
    return b"PAR1" + f"{table}:{rows}".encode() + b"\x00" * 64 + b"PAR1"


class FakeStore:
    """An in-memory store answering counts and Parquet exports over HTTP."""

    def __init__(
        self,
        rows: dict[str, int] | None = None,
        *,
        fail_table: str | None = None,
        busy: int = 0,
        torn_table: str | None = None,
    ):
        self.rows = rows or {}
        self.fail_table = fail_table
        self.busy = busy
        self.torn_table = torn_table
        self.queries: list[str] = []
        self.auth: set[str] = set()

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.auth.add(request.headers.get("Authorization", ""))
        sql = request.content.decode()
        self.queries.append(sql)
        if self.busy:
            self.busy -= 1
            return httpx.Response(
                503,
                headers={"X-ClickHouse-Exception-Code": "202"},
                text="Code: 202. DB::Exception: Too many simultaneous queries. (TOO_MANY_SIMULTANEOUS_QUERIES)",
            )
        match = _FROM.search(sql)
        assert match, sql
        table = "query_log" if match["db"] == "system" else match["table"]
        if table == self.fail_table:
            return httpx.Response(500, text="Code: 60. DB::Exception: boom")
        count = self.rows.get(table, 0)
        if "count()" in sql:
            return httpx.Response(200, text=f"{count}\n")
        body = _parquet(table, count)
        if table == self.torn_table:
            body = body[:-3]
        return httpx.Response(200, content=body)


def _exporter(store: FakeStore, **kwargs) -> TelemetryExporter:
    sleeps = kwargs.pop("sleeps", [])
    return TelemetryExporter(
        url="http://telemetry_store:8123",
        environ=kwargs.pop("environ", ENV),
        transport=httpx.MockTransport(store.handler),
        sleep=sleeps.append,
        **kwargs,
    )


@pytest.fixture
def trees(tmp_path):
    sources = tmp_path / "sources"
    sources.mkdir()
    dest = tmp_path / "dest"
    dest.mkdir()
    os.chmod(dest, 0o700)
    return sources, dest


def _lines(dest: Path, day: str = "2026-09-24") -> list[dict]:
    text = (dest / day / MANIFEST_NAME).read_text()
    return [json.loads(line) for line in text.splitlines()]


# ── The rendered SQL ─────────────────────────────────────────────────────────


def test_query_log_is_read_by_event_date():
    export = render_export_sql(QUERY_LOG, DAY)
    count = render_count_sql(QUERY_LOG, DAY)

    assert export == (
        "SELECT * FROM `system`.`query_log` WHERE event_date = '2026-09-23' FORMAT Parquet"
    )
    assert "WHERE event_date = '2026-09-23'" in count
    assert "InsertedAt" not in export + count


@pytest.mark.parametrize("table", [t.name for t in ddl.TABLES])
def test_every_store_table_is_read_by_arrival_day(table):
    export = render_export_sql(table, DAY)
    count = render_count_sql(table, DAY)

    assert export == (
        f"SELECT * FROM `telemetry`.`{table}` "
        "WHERE toDate(InsertedAt, 'UTC') = '2026-09-23' FORMAT Parquet"
    )
    assert "WHERE toDate(InsertedAt, 'UTC') = '2026-09-23'" in count
    assert count.startswith("SELECT count() FROM")
    assert "event_date" not in export + count


def test_every_table_and_the_query_log_are_exported():
    assert set(EXPORT_TABLES) == {t.name for t in ddl.TABLES} | {QUERY_LOG}


def test_an_unknown_table_is_refused():
    with pytest.raises(ValueError):
        render_export_sql("system.users", DAY)


# ── A pass ───────────────────────────────────────────────────────────────────


def test_a_completed_day_exports_one_parquet_per_table_with_a_manifest_line_each(trees):
    sources, dest = trees
    store = FakeStore({"otel_logs": 48, "audit_ledger": 7, "query_log": 3})

    result = run_pass(sources, dest, now=NOW, telemetry=_exporter(store, backfill_days=1))

    assert result.telemetry_days == ["2026-09-23"]
    assert result.errors == []
    base = dest / "2026-09-24/telemetry/2026-09-23"
    assert sorted(p.name for p in base.iterdir()) == sorted(f"{t}.parquet" for t in EXPORT_TABLES)
    assert (base / "otel_logs.parquet").read_bytes() == _parquet("otel_logs", 48)

    lines = _lines(dest)
    files = [r for r in lines if r["kind"] == "file"]
    assert [r["table"] for r in files] == list(EXPORT_TABLES)
    for record in files:
        content = (dest / record["path"]).read_bytes()
        assert record["day"] == "2026-09-23"
        assert record["sha256"] == hashlib.sha256(content).hexdigest()
        assert record["size"] == len(content)
        assert record["rows"] == store.rows.get(record["table"], 0)
        assert record["source"] == f"telemetry/2026-09-23/{record['table']}.parquet"

    day_lines = [r for r in lines if r["kind"] == "telemetry_day"]
    assert len(day_lines) == 1
    assert day_lines[0]["rows"] == {t: store.rows.get(t, 0) for t in EXPORT_TABLES}
    # The day line comes after every file line of the day.
    assert lines.index(day_lines[0]) > max(lines.index(r) for r in files)


def test_grace_waits_ten_minutes_past_midnight():
    assert pending_days(datetime(2026, 9, 24, 0, 9, tzinfo=UTC), set(), 1) == [date(2026, 9, 22)]
    assert pending_days(datetime(2026, 9, 24, 0, 10, tzinfo=UTC), set(), 1) == [DAY]


def test_an_exported_day_is_never_exported_again(trees):
    sources, dest = trees
    store = FakeStore()

    run_pass(sources, dest, now=NOW, telemetry=_exporter(store, backfill_days=1))
    queries = len(store.queries)
    second = run_pass(
        sources,
        dest,
        now=datetime(2026, 9, 24, 12, 0, 0, tzinfo=UTC),
        telemetry=_exporter(store, backfill_days=1),
    )

    assert second.telemetry_days == []
    assert len(store.queries) == queries
    assert load_state(dest).telemetry_days == {"2026-09-23"}


def test_a_failed_table_leaves_the_day_unrecorded(trees):
    sources, dest = trees
    store = FakeStore(fail_table="otel_traces")

    result = run_pass(sources, dest, now=NOW, telemetry=_exporter(store, backfill_days=1))

    assert result.telemetry_days == []
    assert [e["source"] for e in result.errors] == ["telemetry/2026-09-23"]
    assert "Code: 60." in result.errors[0]["error"]
    assert load_state(dest).telemetry_days == set()
    assert not (dest / "2026-09-24/telemetry").exists()
    assert list((dest / ".incoming").iterdir()) == []


def test_a_torn_parquet_body_leaves_the_day_unrecorded(trees):
    sources, dest = trees
    store = FakeStore({"otel_logs": 5}, torn_table="otel_logs")

    result = run_pass(sources, dest, now=NOW, telemetry=_exporter(store, backfill_days=1))

    assert result.telemetry_days == []
    assert "not a whole Parquet file" in result.errors[0]["error"]
    assert list((dest / ".incoming").iterdir()) == []


def test_a_busy_store_is_retried_with_backoff(trees):
    sources, dest = trees
    store = FakeStore({"otel_logs": 1}, busy=2)
    sleeps: list[float] = []

    result = run_pass(
        sources, dest, now=NOW, telemetry=_exporter(store, backfill_days=1, sleeps=sleeps)
    )

    assert result.telemetry_days == ["2026-09-23"]
    assert result.errors == []
    assert sleeps == [0.5, 1.0]


def test_a_store_that_stays_busy_fails_the_day(trees):
    sources, dest = trees
    store = FakeStore(busy=100)
    sleeps: list[float] = []

    result = run_pass(
        sources, dest, now=NOW, telemetry=_exporter(store, backfill_days=1, sleeps=sleeps)
    )

    assert result.telemetry_days == []
    assert "Code: 202." in result.errors[0]["error"]
    assert sleeps == [0.5, 1.0, 2.0, 4.0, 8.0]
    assert load_state(dest).telemetry_days == set()


def test_any_other_error_is_not_retried(trees):
    sources, dest = trees
    store = FakeStore(fail_table="otel_logs")
    sleeps: list[float] = []

    run_pass(sources, dest, now=NOW, telemetry=_exporter(store, backfill_days=1, sleeps=sleeps))

    assert sleeps == []


def test_a_day_that_cannot_be_placed_is_recorded_as_an_error_and_left_unrecorded(
    trees, monkeypatch
):
    sources, dest = trees
    store = FakeStore()

    def _full_disk(*_args):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(archive_run.ArchiveTree, "place", _full_disk)

    result = run_pass(sources, dest, now=NOW, telemetry=_exporter(store, backfill_days=1))

    assert result.telemetry_days == []
    assert [e["source"] for e in result.errors] == ["telemetry/2026-09-23"]
    assert load_state(dest).telemetry_days == set()
    assert list((dest / ".incoming").iterdir()) == []


def test_backfill_stops_at_the_limit(trees):
    sources, dest = trees

    result = run_pass(sources, dest, now=NOW, telemetry=_exporter(FakeStore(), backfill_days=3))

    assert result.telemetry_days == ["2026-09-21", "2026-09-22", "2026-09-23"]


def test_the_archive_account_reads_with_its_password_from_the_environment_only(trees):
    sources, dest = trees
    store = FakeStore()

    run_pass(sources, dest, now=NOW, telemetry=_exporter(store, backfill_days=1))
    expected = "Basic " + base64.b64encode(b"archive:archive-secret").decode()
    assert store.auth == {expected}

    store_without = FakeStore()
    result = run_pass(
        sources,
        dest,
        now=datetime(2026, 9, 25, 2, 0, 0, tzinfo=UTC),
        telemetry=_exporter(store_without, backfill_days=1, environ={}),
    )

    assert store_without.queries == []
    assert result.telemetry_days == []
    assert "OSPREY_TELEMETRY_ARCHIVE_PASSWORD" in result.errors[0]["error"]
