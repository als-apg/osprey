"""Acceptance rows of the picture copy, each against a real PostgreSQL schema.

Every case runs on a fresh scratch database (testcontainer or the shared test
server), so whole-store operations -- backfill, a poll's retry step, the CLI
ingest -- see only the rows the case seeds. Fetches go through the conftest
fake unless a case reads a real file source; renditions are faked so no render
worker starts. Timing cases scale the copy run's constants: a 60 s entry
deadline becomes a fraction of a second.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import Counter
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import pytest

from osprey.services.ariel_search import cli_operations as ops
from osprey.services.ariel_search.attachments import attachment_id_for
from osprey.services.ariel_search.attachments import copy as copy_mod
from osprey.services.ariel_search.attachments.fetch import FetchOutcome
from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.ingestion.base import FacilityAdapter
from osprey.services.ariel_search.ingestion.ingest import ingest_one

# xdist_group("docker"): every container-starting test file shares one worker, so a
# run has a single testcontainers session and the shared database is serialized.
pytestmark = [pytest.mark.asyncio, pytest.mark.xdist_group("docker"), pytest.mark.timeout(120)]

ORIGINS = frozenset({("https", "h.example", 443)})
PNG_MAGIC = b"\x89PNG\r\n\x1a\n" + b"\x00" * 40
PDF = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<< /Type /Catalog >>\nendobj\n"
RENDITION_SHA = "cd" * 32
BASE_TS = datetime(2026, 9, 1, tzinfo=UTC)

#: The scaled entry deadline (60 s in production).
DEADLINE_S = 0.3


# --- helpers -----------------------------------------------------------------------


def _fake_prepare(monkeypatch: pytest.MonkeyPatch) -> None:
    """Render every sniffed picture instantly; non-images answer from the sniff."""
    from osprey.imaging.formats import sniff
    from osprey.services.ariel_search.attachments import prepare as prepare_mod

    async def _prepare(data, **_kwargs):
        sniffed = sniff(data)
        if not sniffed.is_image:
            return prepare_mod.PreparedPicture(sniffed.mime, sniffed.skip_reason)
        return prepare_mod.PreparedPicture(
            mime_type=sniffed.mime,
            skip_reason=None,
            rendition_bytes=b"r",
            rendition_mime="image/png",
            rendition_w=1,
            rendition_h=1,
            rendition_sha256=RENDITION_SHA,
        )

    monkeypatch.setattr(prepare_mod, "prepare_picture", _prepare)


def _scale_copy_runs(monkeypatch: pytest.MonkeyPatch, deadline_s: float = DEADLINE_S) -> None:
    """Give every internally built ``CopyRun`` (poll, backfill) a scaled entry deadline."""

    @dataclass(eq=False)
    class _ScaledCopyRun(copy_mod.CopyRun):
        entry_deadline_s: float = deadline_s

    monkeypatch.setattr(copy_mod, "CopyRun", _ScaledCopyRun)


class _HttpAdapter(FacilityAdapter):
    """An http source whose pictures live on ``h.example``; it yields ``entries``."""

    def __init__(self, config: ARIELConfig, entries: list[dict] | None = None) -> None:
        super().__init__(config)
        self.entries = entries or []

    @property
    def source_system_name(self) -> str:
        return "Stub"

    def attachment_origins(self) -> frozenset:
        return ORIGINS

    async def fetch_entries(self, *_args, **_kwargs) -> AsyncIterator:  # type: ignore[override]
        for entry in list(self.entries):
            yield dict(entry)


class _WriteAdapter(_HttpAdapter):
    """A writable facility logbook: ``create_entry`` returns the entry it then yields."""

    @property
    def supports_write(self) -> bool:
        return True

    async def create_entry(self, _request) -> str:  # type: ignore[override]
        return str(self.entries[0]["entry_id"])


def _config(mode: str = "images", **attachments: Any) -> ARIELConfig:
    return ARIELConfig.from_dict(
        {
            "database": {"uri": "postgresql://test"},
            "attachments": {"copy_on_ingest": mode, **attachments},
            "ingestion": {
                "adapter": "generic_json",
                "source_url": "https://h.example/api",
                "watch": {"require_initial_ingest": False},
            },
        }
    )


def _url(name: str, host: str = "h.example") -> str:
    return f"https://{host}/files/{name}"


def _png(name: str, host: str = "h.example") -> dict:
    return {"url": _url(name, host), "type": "image/png", "filename": name}


def _entry(entry_id: str, attachments: list, *, minutes: int = 0) -> dict:
    return {
        "entry_id": entry_id,
        "source_system": "test",
        "timestamp": BASE_TS + timedelta(minutes=minutes),
        "author": "tester",
        "raw_text": "beam lost at 14:02",
        "attachments": attachments,
        "metadata": {},
        "enhancement_status": {},
    }


def _id(entry_id: str, url: str) -> str:
    aid = attachment_id_for(entry_id, {"url": url})
    assert aid is not None
    return aid


async def _ingest(repo, entry: dict, cfg: ARIELConfig, *, deadline_s: float | None = None):
    adapter = _HttpAdapter(cfg)
    run = copy_mod.CopyRun(adapter, ORIGINS)
    if deadline_s is not None:
        run.entry_deadline_s = deadline_s
    return await ingest_one(entry, adapter, repo, [], cfg, run)


async def _backfill(repo, cfg: ARIELConfig, **kwargs):
    return await ops.backfill_store(repo, _HttpAdapter(cfg), cfg, **kwargs)


async def _state(repo, entry_id: str) -> dict[str, dict]:
    async with repo.pool.connection() as conn:
        result = await conn.execute(
            """
            SELECT attachment_id, copy_status, skip_reason, copy_attempts, mime_type,
                   data, rendition_sha256, created_at
            FROM attachment_files WHERE entry_id = %(e)s
            """,
            {"e": entry_id},
        )
        cols = [
            "attachment_id",
            "copy_status",
            "skip_reason",
            "copy_attempts",
            "mime_type",
            "data",
            "rendition_sha256",
            "created_at",
        ]
        return {r[0]: dict(zip(cols, r, strict=True)) for r in await result.fetchall()}


async def _status(repo, entry_id: str) -> dict:
    async with repo.pool.connection() as conn:
        result = await conn.execute(
            "SELECT enhancement_status FROM enhanced_entries WHERE entry_id = %(e)s",
            {"e": entry_id},
        )
        row = await result.fetchone()
        return row[0] or {}


async def _seed_recorded(repo, entry_id: str, names: list[str], *, minutes: int = 0) -> list[str]:
    """Store an entry with one declared PNG per name and record its rows, fetching nothing."""
    entry = _entry(entry_id, [_png(n) for n in names], minutes=minutes)
    await ingest_one(entry, _HttpAdapter(_config()), repo, [], _config(), None)
    return [_id(entry_id, _url(n)) for n in names]


def _scheduler(repo, monkeypatch, cfg: ARIELConfig, adapter: FacilityAdapter, lock_factory=None):
    """A poll scheduler over *adapter* with no enhancers, on the real run ledger."""
    from osprey.services.ariel_search import enhancement
    from osprey.services.ariel_search import ingestion as ingestion_pkg
    from osprey.services.ariel_search.ingestion.scheduler import IngestionScheduler

    monkeypatch.setattr(ingestion_pkg, "get_adapter", lambda _config: adapter)
    monkeypatch.setattr(enhancement, "create_enhancers_from_config", lambda _config: [])
    return IngestionScheduler(cfg, repo, lock_factory=lock_factory)


def _waiting_copy_lock(conninfo: str, key: str):
    """The real copy advisory lock, waited for instead of tried.

    The lock is released by closing its connection, and the server finishes that
    release asynchronously, so a poll that follows another at once can still find
    the previous poll's backend holding it and skip its retry step. A test that
    counts one attempt per poll waits for the release; with no other copier in
    the test, waiting never blocks for longer than that.
    """
    from osprey.services.ariel_search.database.connection import try_advisory_lock

    return try_advisory_lock(conninfo, key, wait=True)


def _hanging(*, sent: bool):
    """A fetcher that never answers within any deadline; ``sent`` marks the request out."""

    async def _fetch(*_args, **kwargs):
        if sent:
            kwargs["on_sent"]()
        await asyncio.sleep(3600)
        return FetchOutcome(data=PNG_MAGIC)

    return _fetch


def _sleepy(seconds: float):
    async def _fetch(*_args, **kwargs):
        kwargs["on_sent"]()
        await asyncio.sleep(seconds)
        return FetchOutcome(data=PNG_MAGIC)

    return _fetch


@pytest.fixture
async def scratch_repo(scratch_database):
    """An ``ARIELRepository`` on a fresh scratch database at today's schema."""
    from osprey.services.ariel_search.database import ARIELRepository
    from osprey.services.ariel_search.database.connection import create_connection_pool
    from osprey.services.ariel_search.database.migrations import run_migrations

    cfg = ARIELConfig.from_dict({"database": {"uri": scratch_database}})
    pool = await create_connection_pool(cfg.database)
    try:
        await run_migrations(pool, cfg)
        yield ARIELRepository(pool, cfg)
    finally:
        await pool.close()


async def _migrate(config_dict: dict) -> None:
    from osprey.services.ariel_search.database.connection import create_connection_pool
    from osprey.services.ariel_search.database.migrations import run_migrations

    config = ARIELConfig.from_dict(json.loads(json.dumps(config_dict)))
    pool = await create_connection_pool(config.database)
    try:
        await run_migrations(pool, config)
    finally:
        await pool.close()


def _sync_rows(uri: str, entry_id: str) -> list[tuple]:
    with psycopg.connect(uri) as conn:
        return conn.execute(
            """
            SELECT attachment_id, copy_status, rendition_sha256, data, created_at
            FROM attachment_files WHERE entry_id = %s ORDER BY attachment_id
            """,
            (entry_id,),
        ).fetchall()


# --- re-ingest -----------------------------------------------------------------------


class TestReIngest:
    async def test_two_als_ingests_keep_one_row_with_the_same_id_and_fetch_once(
        self, monkeypatch, scratch_database, tmp_path, attachment_fetch
    ):
        _fake_prepare(monkeypatch)
        attachment_fetch.respond(FetchOutcome(data=PNG_MAGIC))
        source = tmp_path / "als.jsonl"
        picture = "attachments/2026/10/beam_profile.png"
        source.write_text(
            json.dumps(
                {
                    "id": "als-accept-1",
                    "timestamp": "1790000000",
                    "author": "operator",
                    "subject": "Beam profile",
                    "details": "Screen image of the injected beam.",
                    "category": "Operations",
                    "tag": "0",
                    "linkedto": "0",
                    "level": "entry",
                    "attachments": [{"url": picture}],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        config_dict = {
            "database": {"uri": scratch_database},
            "ingestion": {"adapter": "als_logbook", "source_url": str(source)},
        }
        await _migrate(config_dict)

        async def _run():
            return await ops.run_ingest(
                dict(config_dict),
                source=str(source),
                adapter=None,
                since=None,
                limit=None,
                dry_run=False,
            )

        assert (await _run()).count == 1
        first = _sync_rows(scratch_database, "als-accept-1")
        assert len(first) == 1
        assert first[0][1] == "copied"
        assert first[0][2] == RENDITION_SHA
        assert len([c for c in attachment_fetch.calls if c["url"].endswith(picture)]) == 1
        fetches = len(attachment_fetch.calls)

        assert (await _run()).count == 1
        second = _sync_rows(scratch_database, "als-accept-1")
        assert len(attachment_fetch.calls) == fetches
        assert [r[0] for r in second] == [first[0][0]]
        assert second[0][4] == first[0][4]
        assert second[0][1] == "copied"

    async def test_a_c_after_a_b_drops_b_keeps_a_and_copies_c(
        self, scratch_repo, attachment_fetch, monkeypatch
    ):
        _fake_prepare(monkeypatch)
        attachment_fetch.respond(FetchOutcome(data=PNG_MAGIC))
        cfg, entry_id = _config(), "reingest-ac"
        await _ingest(scratch_repo, _entry(entry_id, [_png("a.png"), _png("b.png")]), cfg)
        a, b, c = (_id(entry_id, _url(n)) for n in ("a.png", "b.png", "c.png"))
        before = await _state(scratch_repo, entry_id)
        assert set(before) == {a, b}

        await _ingest(scratch_repo, _entry(entry_id, [_png("a.png"), _png("c.png")]), cfg)

        after = await _state(scratch_repo, entry_id)
        assert set(after) == {a, c}
        assert all(r["copy_status"] == "copied" for r in after.values())
        assert after[a]["created_at"] == before[a]["created_at"]
        urls = Counter(call["url"] for call in attachment_fetch.calls)
        assert urls == {_url("a.png"): 1, _url("b.png"): 1, _url("c.png"): 1}

    @pytest.mark.real_fetch
    async def test_generic_file_source_copies_a_relative_png_and_reingest_deletes_nothing(
        self, monkeypatch, scratch_database, tmp_path
    ):
        from osprey.services.ariel_search.attachments import fetch as fetch_mod

        _fake_prepare(monkeypatch)
        calls: list[str] = []
        real = fetch_mod.fetch_attachment_bytes

        async def _counting(url, *args, **kwargs):
            calls.append(url)
            return await real(url, *args, **kwargs)

        monkeypatch.setattr(copy_mod, "fetch_attachment_bytes", _counting)
        (tmp_path / "pics").mkdir()
        (tmp_path / "pics" / "spot.png").write_bytes(PNG_MAGIC)
        source = tmp_path / "entries.json"
        source.write_text(
            json.dumps(
                {
                    "entries": [
                        {
                            "id": "file-accept-1",
                            "title": "Beam spot",
                            "text": "Screen 3",
                            "author": "operator",
                            "timestamp": "2026-09-01T10:00:00+00:00",
                            "attachments": [
                                {"url": "pics/spot.png", "type": "image/png", "filename": "spot"}
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        config_dict = {
            "database": {"uri": scratch_database},
            "ingestion": {"adapter": "generic_json", "source_url": str(source)},
        }
        await _migrate(config_dict)

        async def _run():
            return await ops.run_ingest(
                dict(config_dict),
                source=str(source),
                adapter=None,
                since=None,
                limit=None,
                dry_run=False,
            )

        assert (await _run()).count == 1
        first = _sync_rows(scratch_database, "file-accept-1")
        assert len(first) == 1
        assert first[0][1] == "copied"
        assert first[0][2] == RENDITION_SHA
        assert bytes(first[0][3]) == PNG_MAGIC
        assert len(calls) == 1

        assert (await _run()).count == 1
        second = _sync_rows(scratch_database, "file-accept-1")
        assert [(r[0], r[1], r[4]) for r in second] == [(r[0], r[1], r[4]) for r in first]
        assert len(calls) == 1


# --- deadlines -----------------------------------------------------------------------


class TestDeadlines:
    async def test_always_hanging_fetcher_costs_each_entry_at_most_the_deadline(
        self, scratch_repo, attachment_fetch, monkeypatch
    ):
        _fake_prepare(monkeypatch)
        _scale_copy_runs(monkeypatch)
        attachment_fetch.respond(_hanging(sent=False))
        cfg = _config()
        entries = [_entry(f"hang-{i}", [_png(f"h{i}.png")], minutes=i) for i in range(3)]
        scheduler = _scheduler(scratch_repo, monkeypatch, cfg, _HttpAdapter(cfg, entries))

        started = time.monotonic()
        result = await scheduler.poll_once()
        elapsed = time.monotonic() - started

        assert result.entries_added == 3
        # Each entry's copy ends at its deadline; the slack covers the database work.
        assert elapsed < 3 * DEADLINE_S + 2.0
        for i in range(3):
            (row,) = (await _state(scratch_repo, f"hang-{i}")).values()
            assert (row["copy_status"], row["copy_attempts"]) == ("pending", 0)

    async def test_fetcher_slower_than_the_deadline_ends_fetch_failed_after_five_polls(
        self, scratch_repo, attachment_fetch, monkeypatch
    ):
        _fake_prepare(monkeypatch)
        _scale_copy_runs(monkeypatch, deadline_s=0.1)
        (aid,) = await _seed_recorded(scratch_repo, "slow-1", ["slow.png"])
        attachment_fetch.respond(_sleepy(1.0))
        cfg = _config()
        scheduler = _scheduler(
            scratch_repo, monkeypatch, cfg, _HttpAdapter(cfg), lock_factory=_waiting_copy_lock
        )

        for poll in range(1, 5):
            await scheduler.poll_once()
            row = (await _state(scratch_repo, "slow-1"))[aid]
            assert (row["copy_status"], row["copy_attempts"]) == ("pending", poll)

        await scheduler.poll_once()
        row = (await _state(scratch_repo, "slow-1"))[aid]
        assert (row["copy_status"], row["skip_reason"]) == ("skipped", "fetch_failed")
        assert len(attachment_fetch.calls) == 5


# --- backfill picks up configuration changes -----------------------------------------


class TestBackfillAfterConfigChange:
    async def test_origin_skip_then_allowed_origins_then_backfill_copies(
        self, scratch_repo, attachment_fetch, monkeypatch
    ):
        _fake_prepare(monkeypatch)
        attachment_fetch.respond(FetchOutcome(data=PNG_MAGIC))
        entry_id, item = "origin-1", _png("o.png", host="other.example")
        aid = _id(entry_id, item["url"])

        await _ingest(scratch_repo, _entry(entry_id, [item]), _config())
        row = (await _state(scratch_repo, entry_id))[aid]
        assert (row["copy_status"], row["skip_reason"]) == ("skipped", "origin_not_allowed")
        assert attachment_fetch.calls == []

        widened = _config(allowed_origins=["https://other.example"])
        result = await _backfill(scratch_repo, widened)

        assert result.status == "done"
        row = (await _state(scratch_repo, entry_id))[aid]
        assert (row["copy_status"], row["skip_reason"]) == ("copied", None)
        assert row["rendition_sha256"] == RENDITION_SHA
        assert [c["url"] for c in attachment_fetch.calls] == [item["url"]]

    async def test_b1_schema_entry_reaches_copied_through_backfill_alone(
        self, scratch_database, attachment_fetch, monkeypatch
    ):
        from osprey.services.ariel_search.database.attachment_migration import (
            AttachmentFilesCopyStateMigration,
        )
        from osprey.services.ariel_search.database.attachment_text_migration import (
            AttachmentTextColumnsMigration,
        )
        from osprey.services.ariel_search.database.connection import create_connection_pool

        _fake_prepare(monkeypatch)
        attachment_fetch.respond(FetchOutcome(data=PNG_MAGIC))
        config_dict = {
            "database": {"uri": scratch_database},
            "ingestion": {"adapter": "generic_json", "source_url": "https://h.example/l.json"},
            "attachments": {"copy_on_ingest": "images"},
        }
        await ops.run_migrate(dict(config_dict))
        cfg = ARIELConfig.from_dict(config_dict)
        pool = await create_connection_pool(cfg.database)
        try:
            for migration in (
                AttachmentFilesCopyStateMigration(),
                AttachmentTextColumnsMigration(),
            ):
                async with pool.connection() as conn, conn.transaction():
                    await migration.down(conn)
                    await conn.execute(
                        "DELETE FROM ariel_migrations WHERE name = %s", (migration.name,)
                    )
        finally:
            await pool.close()

        item = _png("b1.png")
        with psycopg.connect(scratch_database, autocommit=True) as conn:
            conn.execute(
                """
                INSERT INTO enhanced_entries (
                    entry_id, source_system, timestamp, author, raw_text,
                    attachments, metadata, enhancement_status
                ) VALUES ('b1-accept', 'test', %s, 'tester', 'x', %s::jsonb, '{}'::jsonb,
                          '{}'::jsonb)
                """,
                (BASE_TS, json.dumps([item])),
            )

        await ops.run_migrate(dict(config_dict))
        result = await ops.run_backfill(dict(config_dict))

        assert result.status == "done"
        assert result.copied == 1
        rows = _sync_rows(scratch_database, "b1-accept")
        assert [(r[0], r[1], r[2]) for r in rows] == [
            (_id("b1-accept", item["url"]), "copied", RENDITION_SHA)
        ]

    async def test_none_then_images_backfill_copies_and_requeues_the_caption(
        self, scratch_repo, attachment_fetch, monkeypatch
    ):
        _fake_prepare(monkeypatch)
        attachment_fetch.respond(FetchOutcome(data=PNG_MAGIC))
        entry_id = "mode-none-1"
        aid = _id(entry_id, _url("n.png"))

        await _ingest(scratch_repo, _entry(entry_id, [_png("n.png")]), _config("none"))
        row = (await _state(scratch_repo, entry_id))[aid]
        assert (row["copy_status"], row["skip_reason"]) == ("skipped", "copy_on_ingest_mode")
        assert attachment_fetch.calls == []
        async with scratch_repo.pool.connection() as conn:
            await conn.execute(
                "UPDATE enhanced_entries SET enhancement_status = %(s)s::jsonb "
                "WHERE entry_id = %(e)s",
                {"s": json.dumps({"image_caption": {"status": "complete"}}), "e": entry_id},
            )

        result = await _backfill(scratch_repo, _config("images"))

        assert (result.status, result.copied) == ("done", 1)
        row = (await _state(scratch_repo, entry_id))[aid]
        assert (row["copy_status"], row["skip_reason"]) == ("copied", None)
        assert row["rendition_sha256"] == RENDITION_SHA
        assert "image_caption" not in await _status(scratch_repo, entry_id)

    async def test_images_then_all_backfill_copies_an_octet_stream_pdf_once(
        self, scratch_repo, attachment_fetch, monkeypatch
    ):
        _fake_prepare(monkeypatch)
        attachment_fetch.respond(FetchOutcome(data=PDF))
        entry_id = "mode-all-1"
        item = {"url": _url("doc.bin"), "type": "application/octet-stream"}
        aid = _id(entry_id, item["url"])

        await _ingest(scratch_repo, _entry(entry_id, [item]), _config("images"))
        row = (await _state(scratch_repo, entry_id))[aid]
        assert (row["copy_status"], row["skip_reason"]) == ("skipped", "copy_on_ingest_mode")
        assert row["mime_type"] == "application/pdf"
        assert row["data"] is None
        fetched_at_ingest = len(attachment_fetch.calls)
        assert fetched_at_ingest == 1

        result = await _backfill(scratch_repo, _config("all"))

        assert (result.status, result.copied) == ("done", 1)
        row = (await _state(scratch_repo, entry_id))[aid]
        assert row["copy_status"] == "copied"
        assert row["mime_type"] == "application/pdf"
        assert bytes(row["data"]) == PDF
        assert len(attachment_fetch.calls) == fetched_at_ingest + 1

        again = await _backfill(scratch_repo, _config("all"))

        assert (again.status, again.fetches, again.copied) == ("done", 0, 0)
        assert len(attachment_fetch.calls) == fetched_at_ingest + 1


# --- backfill against failing sources and a concurrent poll -------------------------


class TestBackfillRuns:
    async def test_six_gone_rows_and_one_good_row_copy_the_good_one(
        self, scratch_repo, attachment_fetch, monkeypatch
    ):
        _fake_prepare(monkeypatch)
        names = [f"gone-{i}.png" for i in range(6)] + ["good.png"]
        ids = await _seed_recorded(scratch_repo, "gone-1", names)

        def _serve(url, *_args, **_kwargs):
            if url.endswith("/good.png"):
                return FetchOutcome(data=PNG_MAGIC)
            return FetchOutcome(code="source_gone")

        attachment_fetch.respond(_serve)

        result = await _backfill(scratch_repo, _config())

        assert result.status == "done"
        assert result.copied == 1
        assert result.skipped.get("source_gone") == 6
        state = await _state(scratch_repo, "gone-1")
        assert [state[i]["skip_reason"] for i in ids[:6]] == ["source_gone"] * 6
        assert state[ids[6]]["copy_status"] == "copied"
        assert state[ids[6]]["rendition_sha256"] == RENDITION_SHA

    async def test_backfill_next_to_a_poll_retry_step_fetches_each_picture_once(
        self, scratch_repo, attachment_fetch, monkeypatch
    ):
        _fake_prepare(monkeypatch)
        attachment_fetch.respond(_sleepy(0.05))
        for i in range(3):
            await _seed_recorded(scratch_repo, f"race-{i}", [f"{i}-a.png", f"{i}-b.png"], minutes=i)
        cfg = _config()
        scheduler = _scheduler(scratch_repo, monkeypatch, cfg, _HttpAdapter(cfg))

        backfilled, _polled = await asyncio.gather(
            _backfill(scratch_repo, cfg, wait=True), scheduler.poll_once()
        )

        assert backfilled.status == "done"
        counts = Counter(call["url"] for call in attachment_fetch.calls)
        assert len(counts) == 6
        assert set(counts.values()) == {1}
        for i in range(3):
            rows = (await _state(scratch_repo, f"race-{i}")).values()
            assert all(r["copy_status"] == "copied" for r in rows)


# --- an agent-written entry --------------------------------------------------------


class TestServiceCreateEntry:
    async def test_created_entry_reingested_through_b1_upsert_is_copied_by_the_next_poll(
        self, scratch_repo, attachment_fetch, monkeypatch
    ):
        from osprey.services.ariel_search import ingestion as ingestion_pkg
        from osprey.services.ariel_search.models import FacilityEntryCreateRequest
        from osprey.services.ariel_search.service import ARIELSearchService

        _fake_prepare(monkeypatch)
        attachment_fetch.respond(FetchOutcome(data=PNG_MAGIC))
        cfg = _config()
        entry = _entry("svc-accept-1", [_png("upstream.png")])
        entry["timestamp"] = datetime.now(UTC)
        adapter = _WriteAdapter(cfg, [entry])
        monkeypatch.setattr(ingestion_pkg, "get_adapter", lambda _config: adapter)
        service = ARIELSearchService(cfg, scratch_repo.pool, scratch_repo)

        created = await service.create_entry(
            FacilityEntryCreateRequest(subject="Beam spot", details="Screen 3")
        )

        assert created.entry_id == "svc-accept-1"
        assert attachment_fetch.calls == []

        scheduler = _scheduler(scratch_repo, monkeypatch, cfg, adapter)
        await scheduler.poll_once()

        aid = _id("svc-accept-1", _url("upstream.png"))
        row = (await _state(scratch_repo, "svc-accept-1"))[aid]
        assert (row["copy_status"], row["skip_reason"]) == ("copied", None)
        assert row["rendition_sha256"] == RENDITION_SHA
        assert bytes(row["data"]) == PNG_MAGIC
