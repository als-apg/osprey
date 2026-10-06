"""Image-module completion marks against a real database.

An image module is complete for an entry when no picture is still being copied
and every viewable picture is done under the module's current marker. The
per-entry mark and the set-based batch mark share one completion predicate;
these tests pin both, including the races with copy and native writers that
take the entry lock first. Every test runs on a fresh scratch database.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import psycopg
import pytest

from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.database.repository import (
    IMAGE_MARK_BATCH_SIZE,
    ARIELRepository,
    CopyRendition,
    image_completion_sql,
)

# xdist_group("docker"): every container-starting test file shares one worker, so a
# run has a single testcontainers session and the shared database is serialized.
pytestmark = [pytest.mark.asyncio, pytest.mark.xdist_group("docker"), pytest.mark.timeout(180)]

CAPTION = "image_caption"
EMBEDDING = "image_embedding"
MODEL_A = "vis-a"
MODEL_B = "vis-b"
RENDITION = CopyRendition(data=b"png", mime_type="image/png", width=1, height=1, sha256="cd" * 32)


@pytest.fixture
async def repo(scratch_database: str) -> AsyncIterator[ARIELRepository]:
    """A repository over a freshly migrated scratch database."""
    from osprey.services.ariel_search.database.connection import create_connection_pool
    from osprey.services.ariel_search.database.migrations import run_migrations

    config = ARIELConfig.from_dict({"database": {"uri": scratch_database}})
    pool = await create_connection_pool(config.database)
    try:
        await run_migrations(pool, config)
        yield ARIELRepository(pool, config)
    finally:
        await pool.close()


def _seed(
    uri: str,
    entry_id: str,
    *,
    status: dict[str, Any] | None = None,
    captions: dict[str, Any] | None = None,
) -> None:
    with psycopg.connect(uri, autocommit=True) as conn:
        conn.execute(
            """
            INSERT INTO enhanced_entries (
                entry_id, source_system, timestamp, raw_text, enhancement_status,
                attachment_captions
            ) VALUES (%s, 'test', NOW(), 'text', %s::jsonb, %s::jsonb)
            """,
            (
                entry_id,
                json.dumps(status or {}),
                None if captions is None else json.dumps(captions),
            ),
        )


def _seed_bulk(uri: str, prefix: str, count: int) -> None:
    """``count`` picture-less entries ``<prefix>00000`` … in one statement."""
    with psycopg.connect(uri, autocommit=True) as conn:
        conn.execute(
            """
            INSERT INTO enhanced_entries (entry_id, source_system, timestamp, raw_text)
            SELECT %(prefix)s || lpad(i::text, 5, '0'), 'test', NOW(), 'text'
            FROM generate_series(0, %(count)s - 1) AS i
            """,
            {"prefix": prefix, "count": count},
        )


def _picture(uri: str, attachment_id: str, entry_id: str, *, copy_status: str) -> None:
    viewable = copy_status == "copied"
    with psycopg.connect(uri, autocommit=True) as conn:
        conn.execute(
            """
            INSERT INTO attachment_files (
                attachment_id, entry_id, filename, mime_type, source_url,
                copy_status, rendition_sha256
            ) VALUES (%s, %s, 'f.png', 'image/png', 'https://h.example/f.png', %s, %s)
            """,
            (attachment_id, entry_id, copy_status, "ab" * 32 if viewable else None),
        )


def _set_captions(uri: str, entry_id: str, captions: dict[str, Any] | None) -> None:
    with psycopg.connect(uri, autocommit=True) as conn:
        conn.execute(
            "UPDATE enhanced_entries SET attachment_captions = %s::jsonb WHERE entry_id = %s",
            (None if captions is None else json.dumps(captions), entry_id),
        )


def _status(uri: str, entry_id: str) -> dict[str, Any]:
    with psycopg.connect(uri) as conn:
        row = conn.execute(
            "SELECT enhancement_status FROM enhanced_entries WHERE entry_id = %s", (entry_id,)
        ).fetchone()
    assert row is not None
    return row[0]


def _is_complete(uri: str, entry_id: str, module: str, marker: str) -> bool:
    entry = _status(uri, entry_id).get(module) or {}
    return entry.get("status") == "complete" and entry.get("marker") == marker


async def _wait_for_lock_waiter(uri: str) -> None:
    """Return once some backend is blocked on a lock (the mark waiting for the entry)."""
    async with await psycopg.AsyncConnection.connect(uri, autocommit=True) as conn:
        for _ in range(200):
            result = await conn.execute(
                "SELECT count(*) FROM pg_stat_activity"
                " WHERE datname = current_database() AND wait_event_type = 'Lock'"
            )
            row = await result.fetchone()
            if row is not None and row[0] > 0:
                return
            await asyncio.sleep(0.05)
    raise AssertionError("the mark never waited for the entry lock")


class TestCompletionPredicate:
    async def test_text_module_is_refused(self) -> None:
        with pytest.raises(ValueError):
            image_completion_sql("text_embedding", "m")

    async def test_embedding_table_must_be_an_identifier(self) -> None:
        with pytest.raises(ValueError):
            image_completion_sql(EMBEDDING, "t; DROP TABLE x")

    async def test_per_entry_mark_refuses_a_text_module(self, repo: ARIELRepository) -> None:
        with pytest.raises(ValueError):
            await repo.mark_image_module_complete("e-1", "text_embedding", "m")


class TestPerEntryMark:
    async def test_all_pictures_captioned_marks_complete(
        self, repo: ARIELRepository, scratch_database: str
    ) -> None:
        _seed(scratch_database, "e-1", captions={"a-1": {MODEL_A: {"text": "x"}}})
        _picture(scratch_database, "a-1", "e-1", copy_status="copied")

        assert await repo.mark_image_module_complete("e-1", CAPTION, MODEL_A) is True
        assert _is_complete(scratch_database, "e-1", CAPTION, MODEL_A)

    async def test_null_captions_with_one_viewable_picture_marks_nothing(
        self, repo: ARIELRepository, scratch_database: str
    ) -> None:
        _seed(scratch_database, "e-1", captions=None)
        _picture(scratch_database, "a-1", "e-1", copy_status="copied")

        assert await repo.mark_image_module_complete("e-1", CAPTION, MODEL_A) is False
        assert await repo.mark_image_module_complete_batch(CAPTION, MODEL_A) == []
        assert CAPTION not in _status(scratch_database, "e-1")

    async def test_pending_copy_keeps_the_module_owed(
        self, repo: ARIELRepository, scratch_database: str
    ) -> None:
        _seed(scratch_database, "e-1", captions={"a-1": {MODEL_A: {"text": "x"}}})
        _picture(scratch_database, "a-1", "e-1", copy_status="copied")
        _picture(scratch_database, "a-2", "e-1", copy_status="pending")

        assert await repo.mark_image_module_complete("e-1", CAPTION, MODEL_A) is False

    async def test_copy_completing_between_read_and_mark(
        self, repo: ARIELRepository, scratch_database: str
    ) -> None:
        """The module read a-1 only; a-2's copy lands before the mark: still owed."""
        _seed(scratch_database, "e-1", captions={"a-1": {MODEL_A: {"text": "x"}}})
        _picture(scratch_database, "a-1", "e-1", copy_status="copied")
        _picture(scratch_database, "a-2", "e-1", copy_status="pending")

        to_do = [row["attachment_id"] for row in await repo.get_copy_rows("e-1")]
        assert "a-2" in to_do
        written = await repo.apply_copy_outcome(
            "e-1", "a-2", copy_status="copied", data=b"png", mime_type="image/png",
            size_bytes=3, rendition=RENDITION,
        )  # fmt: skip
        assert written == "copied"

        assert await repo.mark_image_module_complete("e-1", CAPTION, MODEL_A) is False
        assert CAPTION not in _status(scratch_database, "e-1")

    async def test_native_writer_committing_while_the_mark_waits(
        self, repo: ARIELRepository, scratch_database: str
    ) -> None:
        """The mark blocks on the entry lock; the writer's new picture is seen after it."""
        _seed(scratch_database, "e-1", captions={"a-1": {MODEL_A: {"text": "x"}}})
        _picture(scratch_database, "a-1", "e-1", copy_status="copied")

        async with await psycopg.AsyncConnection.connect(scratch_database) as writer:
            async with writer.transaction():
                await repo.insert_native_attachment(
                    "e-1", "n-1", filename="n.png", mime_type="image/png", data=b"png",
                    rendition=RENDITION, conn=writer,
                )  # fmt: skip
                mark = asyncio.create_task(repo.mark_image_module_complete("e-1", CAPTION, MODEL_A))
                await _wait_for_lock_waiter(scratch_database)
                assert not mark.done()
        assert await mark is False
        assert CAPTION not in _status(scratch_database, "e-1")

    async def test_gone_entry_marks_nothing(self, repo: ARIELRepository) -> None:
        assert await repo.mark_image_module_complete("missing", CAPTION, MODEL_A) is False

    async def test_embedding_reads_the_image_table(
        self, repo: ARIELRepository, scratch_database: str
    ) -> None:
        with psycopg.connect(scratch_database, autocommit=True) as conn:
            conn.execute("CREATE TABLE img_probe (attachment_id TEXT PRIMARY KEY)")
        _seed(scratch_database, "e-1")
        _picture(scratch_database, "a-1", "e-1", copy_status="copied")

        assert await repo.mark_image_module_complete("e-1", EMBEDDING, "img_probe") is False
        with psycopg.connect(scratch_database, autocommit=True) as conn:
            conn.execute("INSERT INTO img_probe VALUES ('a-1')")
        assert await repo.mark_image_module_complete("e-1", EMBEDDING, "img_probe") is True
        assert _is_complete(scratch_database, "e-1", EMBEDDING, "img_probe")


class TestBatchMark:
    async def test_three_runs_on_1500_pictureless_entries(
        self, repo: ARIELRepository, scratch_database: str
    ) -> None:
        _seed_bulk(scratch_database, "p-", 1500)

        counts = [
            len(await repo.mark_image_module_complete_batch(CAPTION, MODEL_A)) for _ in range(3)
        ]

        assert counts == [1000, 500, 0]

    async def test_keyset_cursor_issues_four_batches_on_3000_rows(
        self, repo: ARIELRepository, scratch_database: str
    ) -> None:
        _seed_bulk(scratch_database, "k-", 3000)

        afters: list[str] = []
        after = ""
        while True:
            afters.append(after)
            ids = await repo.mark_image_module_complete_batch(CAPTION, MODEL_A, after=after)
            if ids:
                assert all(entry_id > after for entry_id in ids)
                after = ids[-1]
            if len(ids) < IMAGE_MARK_BATCH_SIZE:
                break

        assert len(afters) == 4
        assert all(a < b for a, b in zip(afters, afters[1:], strict=False))

    async def test_10k_pictureless_rows_and_one_pictured_row_in_one_pass(
        self, repo: ARIELRepository, scratch_database: str
    ) -> None:
        _seed_bulk(scratch_database, "z-", 10_000)
        _seed(scratch_database, "pic-1")
        _picture(scratch_database, "a-1", "pic-1", copy_status="copied")

        marked: list[str] = []
        batches = 0
        after = ""
        while True:
            ids = await repo.mark_image_module_complete_batch(CAPTION, MODEL_A, after=after)
            batches += 1
            marked.extend(ids)
            if ids:
                after = ids[-1]
            if len(ids) < IMAGE_MARK_BATCH_SIZE:
                break

        assert len(marked) == 10_000
        assert "pic-1" not in marked
        assert batches == 11
        assert CAPTION not in _status(scratch_database, "pic-1")

    async def test_captioned_picture_then_source_skip_completes_in_one_catchup(
        self, repo: ARIELRepository, scratch_database: str
    ) -> None:
        """(a) A captioned while B is pending; B's copy ends in a source skip."""
        _seed(scratch_database, "e-1", captions={"a-1": {MODEL_A: {"text": "x"}}})
        _picture(scratch_database, "a-1", "e-1", copy_status="copied")
        _picture(scratch_database, "b-1", "e-1", copy_status="pending")

        assert await repo.mark_image_module_complete_batch(CAPTION, MODEL_A) == []
        written = await repo.apply_copy_outcome(
            "e-1", "b-1", copy_status="skipped", skip_reason="source_gone"
        )
        assert written == "skipped"

        assert await repo.mark_image_module_complete_batch(CAPTION, MODEL_A) == ["e-1"]
        assert _is_complete(scratch_database, "e-1", CAPTION, MODEL_A)

    async def test_model_switch_back_with_captions_stored_completes(
        self, repo: ARIELRepository, scratch_database: str
    ) -> None:
        """(b) A→B→A: the A captions are still stored, so A is complete with no model call."""
        _seed(
            scratch_database,
            "e-1",
            status={CAPTION: {"status": "complete", "completed_at": "t", "marker": MODEL_B}},
            captions={"a-1": {MODEL_A: {"text": "x"}, MODEL_B: {"text": "y"}}},
        )
        _picture(scratch_database, "a-1", "e-1", copy_status="copied")

        assert await repo.mark_image_module_complete_batch(CAPTION, MODEL_A) == ["e-1"]
        assert _is_complete(scratch_database, "e-1", CAPTION, MODEL_A)

    async def test_every_picture_captioned_but_key_cleared(
        self, repo: ARIELRepository, scratch_database: str
    ) -> None:
        """(c) The walk never visits it (nothing left to do); the batch mark completes it."""
        _seed(
            scratch_database,
            "e-1",
            captions={"a-1": {MODEL_A: {"text": "x"}}, "a-2": {MODEL_A: {"text": "y"}}},
        )
        _picture(scratch_database, "a-1", "e-1", copy_status="copied")
        _picture(scratch_database, "a-2", "e-1", copy_status="copied")
        assert await repo.get_incomplete_entries(CAPTION, marker=MODEL_A) == []

        assert await repo.mark_image_module_complete_batch(CAPTION, MODEL_A) == ["e-1"]
        assert _is_complete(scratch_database, "e-1", CAPTION, MODEL_A)

    async def test_complete_under_current_marker_is_not_revisited(
        self, repo: ARIELRepository, scratch_database: str
    ) -> None:
        _seed(scratch_database, "e-1")
        assert await repo.mark_image_module_complete_batch(CAPTION, MODEL_A) == ["e-1"]
        assert await repo.mark_image_module_complete_batch(CAPTION, MODEL_A) == []
        assert await repo.mark_image_module_complete_batch(CAPTION, MODEL_B) == ["e-1"]

    async def test_locked_entry_is_skipped(
        self, repo: ARIELRepository, scratch_database: str
    ) -> None:
        _seed(scratch_database, "e-1")
        _seed(scratch_database, "e-2")
        async with await psycopg.AsyncConnection.connect(scratch_database) as holder:
            async with holder.transaction():
                assert await ARIELRepository.lock_entry(holder, "e-1")
                assert await repo.mark_image_module_complete_batch(CAPTION, MODEL_A) == ["e-2"]
        assert await repo.mark_image_module_complete_batch(CAPTION, MODEL_A) == ["e-1"]
