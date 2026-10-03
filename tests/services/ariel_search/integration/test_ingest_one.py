"""Integration tests for ``ingest_one``, the per-entry store of every ingest caller.

They run against the real PostgreSQL schema (testcontainer or the shared test
database): the upsert, the attachment savepoint, the picture copy and the
enhancers each on their own pool connection, the way the poll and the CLI run
them.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest

from osprey.services.ariel_search.attachments import attachment_id_for
from osprey.services.ariel_search.attachments import copy as copy_mod
from osprey.services.ariel_search.attachments.compose import (
    caption_model_id,
    compose_attachment_text,
)
from osprey.services.ariel_search.attachments.fetch import FetchOutcome
from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.database.repository import SchemaFacts, attachment_text_md5
from osprey.services.ariel_search.exceptions import DatabaseQueryError
from osprey.services.ariel_search.ingestion.base import FacilityAdapter
from osprey.services.ariel_search.ingestion.ingest import (
    SCHEMA_BEHIND_WARNING,
    EntryIngestOutcome,
    ingest_one,
)

# xdist_group("docker"): every container-starting test file shares one worker, so a
# run has a single testcontainers session and the shared database is serialized.
pytestmark = [pytest.mark.asyncio, pytest.mark.xdist_group("docker")]

#: Rows this file seeds, one family recorded before any write.
PREFIX = "ingestone-"

ORIGINS = frozenset({("https", "h.example", 443)})
PNG_MAGIC = b"\x89PNG\r\n\x1a\n" + b"\x00" * 40
MODEL = "vision-x"


@pytest.fixture(autouse=True)
def _seed_prefix(seeded_prefixes):
    seeded_prefixes.add(PREFIX)


class _HttpAdapter(FacilityAdapter):
    """An http source whose entries the tests hand to ``ingest_one`` directly."""

    @property
    def source_system_name(self) -> str:
        return "Stub"

    async def fetch_entries(self, *_args, **_kwargs) -> AsyncIterator:  # type: ignore[override]
        for entry in ():  # never yields
            yield entry


class _Enhancer:
    """An inline enhancer that records the entries it saw, or raises."""

    def __init__(self, name: str, *, fail: bool = False) -> None:
        self._name = name
        self.fail = fail
        self.seen: list[dict] = []

    @property
    def name(self) -> str:
        return self._name

    async def enhance(self, entry, _conn) -> None:
        self.seen.append(dict(entry))
        if self.fail:
            raise RuntimeError(f"{self._name} down")


def _config(mode: str = "images", *, model_id: str | None = None) -> ARIELConfig:
    data: dict = {
        "database": {"uri": "postgresql://test"},
        "attachments": {"copy_on_ingest": mode},
    }
    if model_id is not None:
        data["enhancement_modules"] = {
            "image_caption": {"enabled": False, "model": {"provider": "x", "model_id": model_id}}
        }
    return ARIELConfig.from_dict(data)


def _url(name: str) -> str:
    return f"https://h.example/files/{name}"


def _png(name: str, **extra) -> dict:
    return {"url": _url(name), "type": "image/png", "filename": name, **extra}


HEADER_ONLY = {"filename": "header", "url": ""}


def _entry(entry_id: str, attachments: list, *, text: str = "beam lost at 14:02") -> dict:
    return {
        "entry_id": entry_id,
        "source_system": "test",
        "timestamp": datetime.now(UTC),
        "author": "tester",
        "raw_text": text,
        "attachments": attachments,
        "metadata": {},
        "enhancement_status": {},
    }


def _id(entry_id: str, url: str) -> str:
    aid = attachment_id_for(entry_id, {"url": url})
    assert aid is not None
    return aid


def _fake_prepare(monkeypatch) -> None:
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
            rendition_sha256="cd" * 32,
        )

    monkeypatch.setattr(prepare_mod, "prepare_picture", _prepare)


async def _ingest(
    repository,
    entry: dict,
    cfg: ARIELConfig,
    *,
    enhancers: list | None = None,
    copy_run: copy_mod.CopyRun | None = None,
) -> EntryIngestOutcome:
    adapter = _HttpAdapter(cfg)
    run = copy_run if copy_run is not None else copy_mod.CopyRun(adapter, ORIGINS)
    return await ingest_one(entry, adapter, repository, enhancers or [], cfg, run)


async def _rows(repository, entry_id: str) -> dict[str, dict]:
    async with repository.pool.connection() as conn:
        result = await conn.execute(
            """
            SELECT attachment_id, source_url, copy_status, skip_reason, data
            FROM attachment_files WHERE entry_id = %(e)s
            """,
            {"e": entry_id},
        )
        cols = ["attachment_id", "source_url", "copy_status", "skip_reason", "data"]
        return {r[0]: dict(zip(cols, r, strict=True)) for r in await result.fetchall()}


async def _stored(repository, entry_id: str) -> dict | None:
    async with repository.pool.connection() as conn:
        result = await conn.execute(
            """
            SELECT raw_text, attachments, attachment_text, attachment_captions,
                   enhancement_status
            FROM enhanced_entries WHERE entry_id = %(e)s
            """,
            {"e": entry_id},
        )
        row = await result.fetchone()
    if row is None:
        return None
    cols = ["raw_text", "attachments", "text", "captions", "status"]
    return dict(zip(cols, row, strict=True))


async def _set_captions(repository, entry_id: str, captions: dict, text: str | None) -> None:
    async with repository.pool.connection() as conn:
        await conn.execute(
            """
            UPDATE enhanced_entries
            SET attachment_captions = %(c)s::jsonb, attachment_text = %(t)s
            WHERE entry_id = %(e)s
            """,
            {"c": json.dumps(captions), "t": text, "e": entry_id},
        )


async def _insert_native(repository, entry_id: str, attachment_id: str, data: bytes) -> None:
    """A native upload's row: source_url NULL, copied, original stored."""
    async with repository.pool.connection() as conn:
        await conn.execute(
            """
            INSERT INTO attachment_files (
                attachment_id, entry_id, filename, mime_type, data, size_bytes
            ) VALUES (%(a)s, %(e)s, 'native.png', 'image/png', %(d)s, %(s)s)
            """,
            {"a": attachment_id, "e": entry_id, "d": data, "s": len(data)},
        )


async def _backfill_pair(repository, entry_ids: list[str], cfg: ARIELConfig) -> None:
    """The backfill shape: a FOR UPDATE page transaction recording rows, then copies."""
    adapter = _HttpAdapter(cfg)
    async with repository.pool.connection() as conn, conn.transaction():
        result = await conn.execute(
            """
            SELECT entry_id, attachments, attachment_text, attachment_captions
            FROM enhanced_entries WHERE entry_id = ANY(%(ids)s)
            ORDER BY entry_id FOR UPDATE
            """,
            {"ids": entry_ids},
        )
        page = await result.fetchall()
        for entry_id, attachments, text, captions in page:
            locked = {
                "attachments": attachments,
                "attachment_text": text,
                "attachment_captions": captions,
            }
            await copy_mod.record_and_compose(conn, entry_id, locked, cfg, adapter)
    run = copy_mod.CopyRun(adapter, ORIGINS)
    for entry_id in entry_ids:
        await copy_mod.copy_entry(repository, entry_id, cfg, run, retry_skipped=True)


@pytest.mark.timeout(60)
class TestIngestOneStore:
    async def test_ingest_one_stores_text_records_rows_copies_and_runs_enhancers(
        self, repository, attachment_fetch, monkeypatch
    ):
        _fake_prepare(monkeypatch)
        attachment_fetch.respond(FetchOutcome(data=PNG_MAGIC))
        cfg = _config("images")
        entry_id = f"{PREFIX}happy"
        ok, broken = _Enhancer("ok_module"), _Enhancer("broken_module", fail=True)
        entry = _entry(entry_id, [_png("a.png", caption="dipole trip")])

        outcome = await _ingest(repository, entry, cfg, enhancers=[ok, broken])

        assert outcome == EntryIngestOutcome(
            enhanced=1, enhancer_failed=1, attachments_recorded=True
        )
        stored = await _stored(repository, entry_id)
        assert stored is not None
        assert stored["raw_text"] == "beam lost at 14:02"
        assert stored["text"] == "[picture a.png - upstream caption] dipole trip"
        assert entry["attachment_text"] == stored["text"]
        assert entry["attachment_captions"] == stored["captions"]
        row = (await _rows(repository, entry_id))[_id(entry_id, _url("a.png"))]
        assert row["copy_status"] == "copied"
        assert bytes(row["data"]) == PNG_MAGIC
        assert ok.seen[0]["attachment_text"] == stored["text"]
        assert "ok_module" in stored["status"]
        assert "broken_module" in stored["status"]

    async def test_ingest_one_upsert_raising_propagates(self, repository, monkeypatch):
        cfg = _config("none")
        entry_id = f"{PREFIX}upsert-raises"
        enhancer = _Enhancer("ok_module")

        async def _broken(*_args, **_kwargs):
            raise DatabaseQueryError("database gone", query="UPSERT")

        monkeypatch.setattr(repository, "upsert_entry_returning", _broken)
        with pytest.raises(DatabaseQueryError):
            await _ingest(repository, _entry(entry_id, [_png("a.png")]), cfg, enhancers=[enhancer])
        assert await _stored(repository, entry_id) is None
        assert enhancer.seen == []

    async def test_ingest_one_record_rows_raising_once_keeps_text_then_backfill_recovers(
        self, repository, attachment_fetch, monkeypatch, caplog
    ):
        _fake_prepare(monkeypatch)
        cfg = _config("images")
        entry_id = f"{PREFIX}savepoint"
        real_record_rows = copy_mod.record_rows
        calls = {"n": 0}

        async def _flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("record exploded")
            return await real_record_rows(*args, **kwargs)

        monkeypatch.setattr(copy_mod, "record_rows", _flaky)
        enhancer = _Enhancer("ok_module")
        entry = _entry(entry_id, [_png("a.png", caption="kicker scan")])

        with caplog.at_level(logging.WARNING, logger="ariel"):
            outcome = await _ingest(repository, entry, cfg, enhancers=[enhancer])

        assert outcome.attachments_recorded is False
        assert outcome.enhanced == 1
        stored = await _stored(repository, entry_id)
        assert stored is not None
        assert stored["raw_text"] == "beam lost at 14:02"
        assert stored["text"] is None
        assert "attachment_text" in entry and entry["attachment_text"] is None
        assert await _rows(repository, entry_id) == {}
        assert attachment_fetch.calls == []
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            f"{entry_id}: attachments not recorded (record exploded); "
            "run osprey ariel attachments backfill" in m
            for m in warnings
        )

        attachment_fetch.respond(FetchOutcome(data=PNG_MAGIC))
        await _backfill_pair(repository, [entry_id], cfg)
        row = (await _rows(repository, entry_id))[_id(entry_id, _url("a.png"))]
        assert row["copy_status"] == "copied"
        stored = await _stored(repository, entry_id)
        assert stored is not None
        assert stored["text"] == "[picture a.png - upstream caption] kicker scan"

    async def test_ingest_one_picture_timeout_keeps_text_stored_before_fetch(
        self, repository, attachment_fetch, monkeypatch
    ):
        _fake_prepare(monkeypatch)
        cfg = _config("images")
        entry_id = f"{PREFIX}timeout"
        seen_at_fetch: list = []

        async def _slow(*_args, **kwargs):
            seen_at_fetch.append(await _stored(repository, entry_id))
            kwargs["on_sent"]()
            await asyncio.sleep(5)
            return FetchOutcome(data=PNG_MAGIC)

        attachment_fetch.respond(_slow)
        run = copy_mod.CopyRun(_HttpAdapter(cfg), ORIGINS, entry_deadline_s=0.3)
        enhancer = _Enhancer("ok_module")

        outcome = await _ingest(
            repository, _entry(entry_id, [_png("a.png")]), cfg, enhancers=[enhancer], copy_run=run
        )

        assert outcome == EntryIngestOutcome(
            enhanced=1, enhancer_failed=0, attachments_recorded=True
        )
        assert seen_at_fetch and seen_at_fetch[0] is not None
        assert seen_at_fetch[0]["raw_text"] == "beam lost at 14:02"
        stored = await _stored(repository, entry_id)
        assert stored is not None and stored["raw_text"] == "beam lost at 14:02"
        row = (await _rows(repository, entry_id))[_id(entry_id, _url("a.png"))]
        assert row["copy_status"] == "pending"
        assert enhancer.seen


@pytest.mark.timeout(60)
class TestIngestOneReIngest:
    async def test_ingest_one_header_only_and_c_after_a_c_deletes_a(
        self, repository, attachment_fetch
    ):
        cfg = _config("none")
        entry_id = f"{PREFIX}header-delete"
        await _ingest(repository, _entry(entry_id, [_png("a.png"), _png("c.png")]), cfg)
        a, c = _id(entry_id, _url("a.png")), _id(entry_id, _url("c.png"))
        assert set(await _rows(repository, entry_id)) == {a, c}

        outcome = await _ingest(repository, _entry(entry_id, [HEADER_ONLY, _png("c.png")]), cfg)

        assert outcome.attachments_recorded is True
        assert set(await _rows(repository, entry_id)) == {c}
        assert attachment_fetch.calls == []

    async def test_ingest_one_pending_row_deleted_when_url_leaves_list(
        self, repository, attachment_fetch, monkeypatch
    ):
        _fake_prepare(monkeypatch)
        attachment_fetch.respond(FetchOutcome(transient=True, host_up=False))
        cfg = _config("images")
        entry_id = f"{PREFIX}pending-delete"
        await _ingest(repository, _entry(entry_id, [_png("a.png"), _png("b.png")]), cfg)
        a, b = _id(entry_id, _url("a.png")), _id(entry_id, _url("b.png"))
        rows = await _rows(repository, entry_id)
        assert rows[b]["copy_status"] == "pending"

        await _ingest(repository, _entry(entry_id, [_png("a.png")]), cfg)

        assert set(await _rows(repository, entry_id)) == {a}

    async def test_ingest_one_dropped_b_loses_its_caption_and_text_line(self, repository):
        cfg = _config("none", model_id=MODEL)
        entry_id = f"{PREFIX}caption-prune"
        await _ingest(repository, _entry(entry_id, [_png("a.png"), _png("b.png")]), cfg)
        b = _id(entry_id, _url("b.png"))
        captions = {b: {MODEL: {"caption": "orbit bump at sector 7"}}}
        stored = await _stored(repository, entry_id)
        assert stored is not None
        text = compose_attachment_text(entry_id, stored["attachments"], captions, MODEL)
        assert text is not None and "orbit bump at sector 7" in text
        await _set_captions(repository, entry_id, captions, text)

        entry = _entry(entry_id, [_png("a.png"), _png("c.png")])
        await _ingest(repository, entry, cfg)

        stored = await _stored(repository, entry_id)
        assert stored is not None
        assert b not in (stored["captions"] or {})
        assert "orbit bump at sector 7" not in (stored["text"] or "")
        assert b not in (entry["attachment_captions"] or {})
        assert entry["attachment_text"] == stored["text"]

    @pytest.mark.parametrize(
        "upstream", [[], [HEADER_ONLY]], ids=["empty-list", "header-only-list"]
    )
    async def test_ingest_one_keeps_native_png_row_and_data(
        self, repository, attachment_fetch, upstream
    ):
        cfg = _config("images")
        suffix = "empty" if not upstream else "header"
        entry_id = f"{PREFIX}native-{suffix}"
        native_id = f"{PREFIX}nat-{suffix}"
        native_item = {
            "url": f"/api/attachments/{native_id}",
            "filename": "native.png",
            "type": "image/png",
        }
        await _ingest(repository, _entry(entry_id, [native_item]), cfg)
        await _insert_native(repository, entry_id, native_id, PNG_MAGIC)

        outcome = await _ingest(repository, _entry(entry_id, list(upstream)), cfg)

        assert outcome.attachments_recorded is True
        rows = await _rows(repository, entry_id)
        assert set(rows) == {native_id}
        assert bytes(rows[native_id]["data"]) == PNG_MAGIC
        stored = await _stored(repository, entry_id)
        assert stored is not None
        assert native_item in stored["attachments"]
        assert attachment_fetch.calls == []


@pytest.mark.timeout(120)
class TestIngestOneConcurrency:
    async def test_ingest_one_two_interleaved_calls_end_consistent(
        self, repository, attachment_fetch
    ):
        cfg = _config("none", model_id=MODEL)
        entry_id = f"{PREFIX}interleave"
        first = [_png("a.png", caption="one"), _png("b.png", caption="two")]
        second = [_png("b.png", caption="two"), _png("c.png", caption="three")]
        for _round in range(5):
            await asyncio.gather(
                _ingest(repository, _entry(entry_id, list(first)), cfg),
                _ingest(repository, _entry(entry_id, list(second)), cfg),
            )
            stored = await _stored(repository, entry_id)
            assert stored is not None
            urls = {item["url"] for item in stored["attachments"]}
            rows = await _rows(repository, entry_id)
            assert {r["source_url"] for r in rows.values()} == urls
            expected = compose_attachment_text(
                entry_id, stored["attachments"], stored["captions"], caption_model_id(cfg)
            )
            assert stored["text"] == expected
        assert attachment_fetch.calls == []

    async def test_ingest_one_against_backfill_page_lock_never_deadlocks(
        self, repository, attachment_fetch, monkeypatch
    ):
        _fake_prepare(monkeypatch)
        attachment_fetch.respond(FetchOutcome(data=PNG_MAGIC))
        cfg = _config("images")
        ids = [f"{PREFIX}deadlock-1", f"{PREFIX}deadlock-2"]
        lists = [
            [_png("a.png"), _png("b.png")],
            [_png("b.png"), _png("c.png"), HEADER_ONLY],
        ]
        for entry_id in ids:
            await _ingest(repository, _entry(entry_id, list(lists[0])), cfg)

        async def _ingest_loop() -> None:
            for i in range(8):
                for entry_id in reversed(ids):
                    await _ingest(repository, _entry(entry_id, list(lists[i % 2])), cfg)

        async def _backfill_loop() -> None:
            for _ in range(8):
                await _backfill_pair(repository, ids, cfg)

        await asyncio.gather(_ingest_loop(), _backfill_loop())

        for entry_id in ids:
            stored = await _stored(repository, entry_id)
            assert stored is not None
            urls = {item["url"] for item in stored["attachments"] if item.get("url")}
            rows = await _rows(repository, entry_id)
            assert {r["source_url"] for r in rows.values()} == urls


class TestIngestOneSchemaBehind:
    async def test_ingest_one_schema_behind_runs_b1_upsert_and_warns_once_per_run(
        self, repository, attachment_fetch, monkeypatch, caplog
    ):
        cfg = _config("images")

        async def _behind() -> SchemaFacts:
            return SchemaFacts(has_v2_fts=True, has_copy_state=False)

        b1_calls: list[str] = []
        real_upsert = repository.upsert_entry

        async def _spy(entry):
            b1_calls.append(entry["entry_id"])
            await real_upsert(entry)

        async def _no_returning(*_args, **_kwargs):
            raise AssertionError("the B1 path never upserts RETURNING")

        monkeypatch.setattr(repository, "schema_facts", _behind)
        monkeypatch.setattr(repository, "upsert_entry", _spy)
        monkeypatch.setattr(repository, "upsert_entry_returning", _no_returning)
        enhancer = _Enhancer("ok_module")
        run = copy_mod.CopyRun(_HttpAdapter(cfg), ORIGINS)
        ids = [f"{PREFIX}behind-1", f"{PREFIX}behind-2"]

        with caplog.at_level(logging.WARNING, logger="ariel"):
            outcomes = [
                await _ingest(
                    repository,
                    _entry(entry_id, [_png("a.png")]),
                    cfg,
                    enhancers=[enhancer],
                    copy_run=run,
                )
                for entry_id in ids
            ]
            next_run = copy_mod.CopyRun(_HttpAdapter(cfg), ORIGINS)
            await _ingest(repository, _entry(f"{PREFIX}behind-3", []), cfg, copy_run=next_run)

        assert outcomes == [EntryIngestOutcome(1, 0, True)] * 2
        assert b1_calls == [*ids, f"{PREFIX}behind-3"]
        for entry_id in ids:
            stored = await _stored(repository, entry_id)
            assert stored is not None and stored["raw_text"] == "beam lost at 14:02"
            assert await _rows(repository, entry_id) == {}
        assert attachment_fetch.calls == []
        warned = [r for r in caplog.records if r.getMessage() == SCHEMA_BEHIND_WARNING]
        assert len(warned) == 2


async def _run_times(repository, run_id: int) -> tuple[datetime, datetime | None]:
    async with repository.pool.connection() as conn:
        result = await conn.execute(
            "SELECT started_at, completed_at FROM ingestion_runs WHERE id = %(id)s",
            {"id": run_id},
        )
        row = await result.fetchone()
    assert row is not None
    if isinstance(row, dict):
        return row["started_at"], row["completed_at"]
    return row[0], row[1]


class TestPollWatermark:
    async def test_watermark_is_the_successful_run_start_so_mid_run_entries_are_polled(
        self, repository
    ):
        source = f"{PREFIX}watermark-{uuid.uuid4().hex[:8]}"
        try:
            run_id = await repository.start_ingestion_run(source)
            started_at, _ = await _run_times(repository, run_id)

            # Written upstream while the run fetched: newer than the run's start.
            entry = _entry(f"{PREFIX}watermark-mid", [])
            entry["timestamp"] = started_at + timedelta(milliseconds=1)
            await _ingest(repository, entry, _config("none"))
            await asyncio.sleep(0.05)
            await repository.complete_ingestion_run(
                run_id, entries_added=1, entries_updated=0, entries_failed=0
            )
            _, completed_at = await _run_times(repository, run_id)

            watermark = await repository.get_last_successful_run(source)
            assert watermark == started_at
            assert completed_at is not None and completed_at > entry["timestamp"]
            assert watermark < entry["timestamp"]

            failed_id = await repository.start_ingestion_run(source)
            await repository.fail_ingestion_run(failed_id, "nothing stored")
            assert await repository.get_last_successful_run(source) == started_at
        finally:
            async with repository.pool.connection() as conn:
                await conn.execute(
                    "DELETE FROM ingestion_runs WHERE source_system = %(s)s", {"s": source}
                )


class _CaptionedEmbedder(_Enhancer):
    """A ``text_embedding`` stand-in recording the embed input it would send."""

    def __init__(self, *, on_enhance=None) -> None:
        super().__init__("text_embedding")
        self.inputs: list[str] = []
        self._on_enhance = on_enhance

    async def enhance(self, entry, conn) -> None:
        from osprey.services.ariel_search.enhancement.text_embedding.embedder import (
            embedding_input,
        )

        await super().enhance(entry, conn)
        self.inputs.append(
            embedding_input(entry.get("raw_text") or "", entry.get("attachment_text"), 512)
        )
        if self._on_enhance is not None:
            await self._on_enhance(entry)


class _OnlyThese:
    """The real repository, with ``get_incomplete_entries`` narrowed to this file's ids."""

    def __init__(self, repository, ids: set[str]) -> None:
        self._repository = repository
        self._ids = ids

    def __getattr__(self, name):
        return getattr(self._repository, name)

    async def get_incomplete_entries(self, module_name=None, status=None, limit=100, **kw):
        found = await self._repository.get_incomplete_entries(
            module_name=module_name, status=status, limit=100_000, **kw
        )
        return [entry for entry in found if entry["entry_id"] in self._ids][:limit]


CAPTION_TEXT = "[picture a.png - upstream caption] dipole trip"


@pytest.mark.timeout(60)
class TestTextModulesMarkUnderAttachmentTextMd5:
    async def test_ingest_one_marks_text_embedding_on_a_migrated_store(
        self, repository, attachment_fetch, monkeypatch
    ):
        _fake_prepare(monkeypatch)
        attachment_fetch.respond(FetchOutcome(data=PNG_MAGIC))
        entry_id = f"{PREFIX}md5-mark"
        embedder = _CaptionedEmbedder()
        marks: list[dict] = []
        real_mark = repository.mark_enhancement_complete

        async def _spy(entry_id_, module, **kwargs):
            marks.append({"module": module, **kwargs})
            await real_mark(entry_id_, module, **kwargs)

        monkeypatch.setattr(repository, "mark_enhancement_complete", _spy)
        entry = _entry(entry_id, [_png("a.png", caption="dipole trip")])

        outcome = await _ingest(repository, entry, _config("images"), enhancers=[embedder])

        assert outcome.enhanced == 1
        stored = await _stored(repository, entry_id)
        assert stored is not None and stored["text"] == CAPTION_TEXT
        assert stored["status"]["text_embedding"]["status"] == "complete"
        assert marks == [{"module": "text_embedding", "md5": attachment_text_md5(entry)}]
        assert embedder.inputs == [f"beam lost at 14:02\n{CAPTION_TEXT}"]

    async def test_a_caption_changed_during_enhance_leaves_the_text_module_owed(
        self, repository, attachment_fetch, monkeypatch
    ):
        _fake_prepare(monkeypatch)
        attachment_fetch.respond(FetchOutcome(data=PNG_MAGIC))
        entry_id = f"{PREFIX}md5-race"

        async def _recaption(_entry_seen) -> None:
            await _set_captions(repository, entry_id, {}, "[picture a.png] a newer caption")

        embedder = _CaptionedEmbedder(on_enhance=_recaption)
        other = _Enhancer("ok_module")
        entry = _entry(entry_id, [_png("a.png", caption="dipole trip")])

        outcome = await _ingest(repository, entry, _config("images"), enhancers=[embedder, other])

        assert outcome.enhanced == 2
        stored = await _stored(repository, entry_id)
        assert stored is not None
        assert "text_embedding" not in stored["status"]
        assert stored["status"]["ok_module"]["status"] == "complete"

    async def test_an_unchanged_reingest_keeps_the_caption_in_the_embedder_input(
        self, repository, attachment_fetch, monkeypatch
    ):
        _fake_prepare(monkeypatch)
        attachment_fetch.respond(FetchOutcome(data=PNG_MAGIC))
        entry_id = f"{PREFIX}md5-reingest"
        cfg = _config("images")
        first, second = _CaptionedEmbedder(), _CaptionedEmbedder()

        await _ingest(
            repository,
            _entry(entry_id, [_png("a.png", caption="dipole trip")]),
            cfg,
            enhancers=[first],
        )
        await _ingest(
            repository,
            _entry(entry_id, [_png("a.png", caption="dipole trip")]),
            cfg,
            enhancers=[second],
        )

        expected = f"beam lost at 14:02\n{CAPTION_TEXT}"
        assert first.inputs == [expected]
        assert second.inputs == [expected]
        stored = await _stored(repository, entry_id)
        assert stored is not None
        assert stored["status"]["text_embedding"]["status"] == "complete"

    async def test_run_enhance_after_a_caption_fold_completes_once(
        self, repository, attachment_fetch, monkeypatch
    ):
        import osprey.services.ariel_search.enhancement as enhancement_pkg
        from osprey.services.ariel_search import cli_operations as ops
        from tests.services.ariel_search._cli_ops_doubles import _patch_service, _StubService

        _fake_prepare(monkeypatch)
        attachment_fetch.respond(FetchOutcome(data=PNG_MAGIC))
        entry_id = f"{PREFIX}md5-run-enhance"
        await _ingest(
            repository, _entry(entry_id, [_png("a.png", caption="dipole trip")]), _config()
        )
        embedder = _CaptionedEmbedder()
        monkeypatch.setattr(
            enhancement_pkg, "create_enhancers_from_config", lambda *_a, **_k: [embedder]
        )
        _patch_service(
            monkeypatch, _StubService(_OnlyThese(repository, {entry_id}), repository.pool)
        )
        db = {"database": {"uri": "postgresql://test"}}

        first = await ops.run_enhance(dict(db), module=None, force=False, limit=10)
        second = await ops.run_enhance(dict(db), module=None, force=False, limit=10)

        assert (first.entries_processed, first.succeeded) == (1, 1)
        assert embedder.inputs == [f"beam lost at 14:02\n{CAPTION_TEXT}"]
        assert second.entries_processed == 0
        assert len(embedder.seen) == 1
        stored = await _stored(repository, entry_id)
        assert stored is not None
        assert stored["status"]["text_embedding"]["status"] == "complete"

    async def test_a_schema_behind_store_marks_without_md5(self, repository, monkeypatch):
        async def _behind() -> SchemaFacts:
            return SchemaFacts(has_v2_fts=True, has_copy_state=False)

        marks: list[tuple] = []
        real_mark = repository.mark_enhancement_complete

        async def _spy(*args, **kwargs):
            marks.append((args, kwargs))
            await real_mark(*args, **kwargs)

        monkeypatch.setattr(repository, "schema_facts", _behind)
        monkeypatch.setattr(repository, "mark_enhancement_complete", _spy)
        entry_id = f"{PREFIX}md5-behind"

        await _ingest(
            repository, _entry(entry_id, []), _config("none"), enhancers=[_CaptionedEmbedder()]
        )

        assert marks == [((entry_id, "text_embedding"), {})]
        stored = await _stored(repository, entry_id)
        assert stored is not None
        assert stored["status"]["text_embedding"]["status"] == "complete"
