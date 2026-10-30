"""The embedded backend: one DuckDB file behind the record contract.

Pins first-write-wins append on ``(key, chunk_index)`` with duplicates counted
(within a batch and across arrival days), select under a scope filter and a
query, arrival-day delete and Parquet export, the last-write-wins state table,
the health probe, and that a second opener of the file is refused with
:class:`RecordBackendLocked` naming the file instead of hanging.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pytest

from osprey.agent_record import contract as c
from osprey.agent_record.backends.embedded import EmbeddedBackend, RecordBackendLocked

TS = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)
DAY1 = datetime(2026, 10, 20, 8, 0, tzinfo=UTC)
DAY2 = datetime(2026, 10, 21, 8, 0, tzinfo=UTC)
DAY3 = datetime(2026, 10, 22, 8, 0, tzinfo=UTC)


def _entry(key="sp:alice:e1", **overrides):
    fields = {
        "key": key,
        "kind": c.KIND_AUDIT,
        "ts": TS,
        "identity": "alice",
        "body": {"text": key},
    }
    fields.update(overrides)
    return c.Entry(**fields)


@pytest.fixture
def path(tmp_path: Path) -> Path:
    return tmp_path / "record.duckdb"


@pytest.fixture
def backend(path: Path):
    with EmbeddedBackend(path) as b:
        yield b


def _keys(entries):
    return [e.key for e in entries]


# --------------------------------------------------------------------------- protocol


def test_is_a_record_backend(backend):
    assert isinstance(backend, c.RecordBackend)
    assert isinstance(backend, c.StateStore)


def test_locked_is_an_unavailable_backend_error():
    assert issubclass(RecordBackendLocked, c.RecordBackendUnavailable)
    assert issubclass(RecordBackendLocked, c.RecordBackendError)


def test_creates_parent_directory(tmp_path):
    p = tmp_path / "var" / "agent_record" / "record.duckdb"
    with EmbeddedBackend(p) as b:
        assert b.path == p
        assert b.append([_entry()]) == c.AppendResult(appended=1)
    assert p.exists()


# --------------------------------------------------------------------------- append


def test_append_round_trips_every_field(backend):
    entry = _entry(
        key="cc:alice:s1:u1:0",
        kind=c.KIND_ANSWER,
        ts=datetime(2026, 10, 23, 14, 0, 0, 123456, tzinfo=timezone(timedelta(hours=2))),
        body={"text": "héllo", "nested": {"n": [1, 2.5, None, True]}, "z": 1, "a": 2},
        on_behalf_of="bob",
        session_id="s1",
        terminal_session="t1",
        run_id="r1",
        agent_id="ag1",
        parent_key="cc:alice:s1:u0:0",
        tool_use_id="tu1",
        inserted_at=DAY2,
    )
    assert backend.append([entry]) == c.AppendResult(appended=1, duplicates=0)
    [got] = backend.select(c.ScopeFilter(), c.Query())
    assert got == entry
    assert got.ts.tzinfo is not None and got.inserted_at.tzinfo is not None
    assert c.serialize_body(got.body) == c.serialize_body(entry.body)


def test_append_sets_arrival_when_absent(backend):
    before = datetime.now(UTC)
    backend.append([_entry()])
    after = datetime.now(UTC)
    [got] = backend.select(c.ScopeFilter(), c.Query())
    assert got.inserted_at is not None
    assert before <= got.inserted_at <= after


def test_first_write_wins_and_duplicate_is_counted(backend):
    assert backend.append([_entry(body={"v": "first"})]) == c.AppendResult(1, 0)
    assert backend.append([_entry(body={"v": "second"})]) == c.AppendResult(0, 1)
    [got] = backend.select(c.ScopeFilter(), c.Query())
    assert got.body == {"v": "first"}


def test_duplicates_within_one_batch_are_counted(backend):
    batch = [
        _entry(body={"v": "first"}),
        _entry(key="sp:alice:e2"),
        _entry(body={"v": "second"}),
        _entry(key="sp:alice:e2"),
    ]
    assert backend.append(batch) == c.AppendResult(appended=2, duplicates=2)
    got = {e.key: e for e in backend.select(c.ScopeFilter(), c.Query())}
    assert got["sp:alice:e1"].body == {"v": "first"}


def test_duplicate_on_a_later_arrival_day_is_not_stored(backend):
    backend.append([_entry(inserted_at=DAY1)])
    assert backend.append([_entry(inserted_at=DAY2, body={"v": "late"})]) == c.AppendResult(0, 1)
    [got] = backend.select(c.ScopeFilter(), c.Query())
    assert got.inserted_at == DAY1
    assert backend.export_parquet(DAY2.date(), backend.path.parent / "d2.parquet") == 0


def test_chunks_share_key_and_differ_in_index(backend):
    big = _entry(body={"text": "x" * 500})
    chunks = c.split_entry(big, 120)
    assert len(chunks) > 1
    assert backend.append(chunks) == c.AppendResult(appended=len(chunks))
    assert backend.append(chunks[:1]) == c.AppendResult(duplicates=1)
    got = backend.select(c.ScopeFilter(), c.Query(keys=frozenset({big.key})))
    assert sorted(e.chunk.index for e in got) == list(range(len(chunks)))
    assert c.reassemble(got).body == big.body


def test_empty_append(backend):
    assert backend.append([]) == c.AppendResult(0, 0)


def test_large_batch(backend):
    batch = [_entry(key=f"sp:alice:e{i}") for i in range(1500)]
    assert backend.append(batch) == c.AppendResult(appended=1500)
    assert backend.append(batch) == c.AppendResult(duplicates=1500)
    assert backend.health().entries == 1500


def test_append_refuses_a_non_entry(backend):
    with pytest.raises(c.RecordBackendError):
        backend.append([{"key": "x"}])
    assert backend.select(c.ScopeFilter(), c.Query()) == []


# --------------------------------------------------------------------------- select


@pytest.fixture
def populated(backend):
    backend.append(
        [
            _entry("sp:alice:a1", ts=DAY1, session_id="s1", inserted_at=DAY1, tool_use_id="t1"),
            _entry(
                "sp:alice:a2",
                kind=c.KIND_ANSWER,
                ts=DAY2,
                session_id="s2",
                run_id="r1",
                inserted_at=DAY2,
            ),
            _entry("sp:bob:b1", identity="bob", ts=DAY3, run_id="r2", inserted_at=DAY3),
        ]
    )
    return backend


def test_select_unfiltered_oldest_first(populated):
    assert _keys(populated.select(c.ScopeFilter(), c.Query())) == [
        "sp:alice:a1",
        "sp:alice:a2",
        "sp:bob:b1",
    ]


def test_select_newest_first_and_limit(populated):
    got = populated.select(c.ScopeFilter(), c.Query(order=c.ORDER_NEWEST_FIRST, limit=2))
    assert _keys(got) == ["sp:bob:b1", "sp:alice:a2"]


@pytest.mark.parametrize(
    ("scope_filter", "expected"),
    [
        (c.ScopeFilter(identities=frozenset({"bob"})), ["sp:bob:b1"]),
        (c.ScopeFilter(kinds=frozenset({c.KIND_ANSWER})), ["sp:alice:a2"]),
        (c.ScopeFilter(session_ids=frozenset({"s1", "s2"})), ["sp:alice:a1", "sp:alice:a2"]),
        (c.ScopeFilter(run_ids=frozenset({"r2"})), ["sp:bob:b1"]),
        (c.ScopeFilter(arrived_since=DAY2), ["sp:alice:a2", "sp:bob:b1"]),
        (
            c.ScopeFilter(identities=frozenset({"alice"}), arrived_since=DAY2),
            ["sp:alice:a2"],
        ),
        (c.ScopeFilter(identities=frozenset({"nobody"})), []),
    ],
)
def test_select_scope_filter(populated, scope_filter, expected):
    assert _keys(populated.select(scope_filter, c.Query())) == expected


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (c.Query(keys=frozenset({"sp:alice:a2", "sp:missing:x"})), ["sp:alice:a2"]),
        (c.Query(key_prefix="sp:alice:"), ["sp:alice:a1", "sp:alice:a2"]),
        (c.Query(key_prefix="sp:al%"), []),
        (c.Query(tool_use_ids=frozenset({"t1"})), ["sp:alice:a1"]),
        (c.Query(since=DAY2), ["sp:alice:a2", "sp:bob:b1"]),
        (c.Query(until=DAY2), ["sp:alice:a1"]),
        (c.Query(since=DAY2, until=DAY3), ["sp:alice:a2"]),
        (c.Query(limit=0), []),
    ],
)
def test_select_query(populated, query, expected):
    assert _keys(populated.select(c.ScopeFilter(), query)) == expected


def test_select_ties_on_ts_order_by_arrival(backend):
    backend.append([_entry("sp:alice:late", inserted_at=DAY2)])
    backend.append([_entry("sp:alice:early", inserted_at=DAY1)])
    assert _keys(backend.select(c.ScopeFilter(), c.Query())) == ["sp:alice:early", "sp:alice:late"]


def test_select_refuses_an_unknown_order(populated):
    with pytest.raises(c.RecordBackendError):
        populated.select(c.ScopeFilter(), c.Query(order="sideways"))


# --------------------------------------------------------------------------- delete


def test_delete_arrived_before_drops_whole_earlier_days(populated):
    assert populated.delete_arrived_before(DAY2.date()) == 1
    assert _keys(populated.select(c.ScopeFilter(), c.Query())) == ["sp:alice:a2", "sp:bob:b1"]
    assert populated.delete_arrived_before(DAY2.date()) == 0
    assert populated.delete_arrived_before(date(2030, 1, 1)) == 2
    assert populated.select(c.ScopeFilter(), c.Query()) == []


def test_delete_uses_the_utc_arrival_day(backend):
    # 23:30 at -05:00 is 04:30 UTC the next day.
    late = datetime(2026, 10, 20, 23, 30, tzinfo=timezone(timedelta(hours=-5)))
    backend.append([_entry(inserted_at=late)])
    assert backend.delete_arrived_before(date(2026, 10, 21)) == 0
    assert backend.delete_arrived_before(date(2026, 10, 22)) == 1


def test_delete_leaves_state(populated):
    populated.put_state("transcript:alice:x.jsonl", {"offset": 3})
    populated.delete_arrived_before(date(2030, 1, 1))
    assert populated.get_state("transcript:alice:x.jsonl") == {"offset": 3}


# --------------------------------------------------------------------------- export


def test_export_parquet_writes_one_arrival_day(populated, tmp_path):
    dest = tmp_path / "archive" / "2026-10-21" / "agent_record" / "record.parquet"
    assert populated.export_parquet(DAY2.date(), dest) == 1
    con = duckdb.connect()
    rows = con.execute(
        "SELECT key, chunk_index, chunk_total, kind, identity, run_id, body "
        f"FROM read_parquet('{dest}')"
    ).fetchall()
    assert rows == [("sp:alice:a2", 0, 1, c.KIND_ANSWER, "alice", "r1", '{"text":"sp:alice:a2"}')]
    columns = [
        r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{dest}')").fetchall()
    ]
    assert {"key", "chunk_index", "ts", "inserted_at", "on_behalf_of", "tool_use_id"} <= set(
        columns
    )


def test_export_parquet_empty_day(populated, tmp_path):
    dest = tmp_path / "empty.parquet"
    assert populated.export_parquet(date(2020, 1, 1), dest) == 0
    assert dest.exists()


def test_export_parquet_path_with_a_quote(populated, tmp_path):
    dest = tmp_path / "it's" / "day.parquet"
    assert populated.export_parquet(DAY1.date(), dest) == 1
    assert dest.exists()


def test_export_parquet_overwrites(populated, tmp_path):
    dest = tmp_path / "day.parquet"
    populated.export_parquet(DAY1.date(), dest)
    assert populated.export_parquet(DAY1.date(), dest) == 1


# --------------------------------------------------------------------------- state


def test_state_last_write_wins(backend):
    assert backend.get_state("spool:alice:seg:1") is None
    backend.put_state("spool:alice:seg:1", {"offset": 10})
    backend.put_state("spool:alice:seg:1", {"offset": 20, "extra": [1, "x"]})
    assert backend.get_state("spool:alice:seg:1") == {"offset": 20, "extra": [1, "x"]}


def test_state_is_not_in_the_record(backend):
    backend.put_state("transcript:alice:a.jsonl", {"offset": 1})
    assert backend.select(c.ScopeFilter(), c.Query()) == []
    assert backend.health().entries == 0


def test_state_refuses_a_non_mapping_or_unserializable_value(backend):
    with pytest.raises(c.RecordBackendError):
        backend.put_state("x", ["not", "a", "mapping"])
    with pytest.raises(c.RecordBackendError):
        backend.put_state("x", {"bad": object()})
    assert backend.get_state("x") is None


def test_state_survives_reopen(path):
    with EmbeddedBackend(path) as b:
        b.put_state("n", {"v": 1})
        b.append([_entry()])
    with EmbeddedBackend(path) as b:
        assert b.get_state("n") == {"v": 1}
        assert b.append([_entry()]) == c.AppendResult(duplicates=1)


# --------------------------------------------------------------------------- health


def test_health_empty(backend):
    h = backend.health()
    assert h == c.BackendHealth(ok=True, backend="embedded", detail=h.detail, entries=0)
    assert str(backend.path) in h.detail


def test_health_counts_and_arrival_range(populated):
    h = populated.health()
    assert h.ok and h.backend == "embedded"
    assert h.entries == 3
    assert h.oldest_arrival_day == DAY1.date()
    assert h.newest_arrival_day == DAY3.date()


def test_health_after_close_is_not_ok(path):
    b = EmbeddedBackend(path)
    b.close()
    h = b.health()
    assert not h.ok
    assert h.backend == "embedded"


def test_methods_after_close_raise_backend_error(path):
    b = EmbeddedBackend(path)
    b.close()
    b.close()  # idempotent
    with pytest.raises(c.RecordBackendError):
        b.append([_entry()])
    with pytest.raises(c.RecordBackendError):
        b.select(c.ScopeFilter(), c.Query())
    with pytest.raises(c.RecordBackendError):
        b.get_state("x")


# --------------------------------------------------------------------------- locking


@pytest.mark.usefixtures("backend")
def test_second_process_is_refused_with_a_clear_error(path):
    script = textwrap.dedent(
        f"""
        import sys
        from osprey.agent_record.backends.embedded import EmbeddedBackend, RecordBackendLocked
        try:
            EmbeddedBackend({str(path)!r})
        except RecordBackendLocked as exc:
            print("LOCKED", exc)
            sys.exit(3)
        print("OPENED")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "LOCKED" in proc.stdout
    assert str(path) in proc.stdout


@pytest.mark.usefixtures("backend")
def test_second_opener_in_the_same_process_is_refused(path):
    with pytest.raises(RecordBackendLocked) as info:
        EmbeddedBackend(path)
    assert str(path) in str(info.value)


def test_reopen_after_close(path):
    EmbeddedBackend(path).close()
    with EmbeddedBackend(path) as b:
        assert b.health().ok


def test_unopenable_file_is_unavailable(tmp_path):
    bogus = tmp_path / "not-a-db.duckdb"
    bogus.write_text("this is not a duckdb file")
    with pytest.raises(c.RecordBackendUnavailable) as info:
        EmbeddedBackend(bogus)
    assert not isinstance(info.value, RecordBackendLocked)
    assert str(bogus) in str(info.value)
