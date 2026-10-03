"""Integration tests for the attachment copy-state repository statements.

Each statement runs against the real PostgreSQL schema (testcontainer or the shared
test database), with named placeholders only.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from osprey.services.ariel_search.database.repository import (
    ATTACHMENT_ROW_COLUMNS,
    IMAGE_STATUS_KEYS,
    CopyRendition,
    SchemaFacts,
)

# xdist_group("docker"): every container-starting test file shares one worker, so a
# run has a single testcontainers session and the shared database is serialized.
pytestmark = [pytest.mark.asyncio, pytest.mark.xdist_group("docker")]

#: Rows this file seeds, one family recorded before any write.
PREFIX = "attstmt-"

#: Far-future timestamps put this file's entries ahead of every other seeded entry.
FUTURE = datetime(2999, 1, 1, tzinfo=UTC)

RENDITION = CopyRendition(
    data=b"\xff\xd8rendition", mime_type="image/jpeg", width=4, height=3, sha256="ab" * 32
)

STATUS_WITH_IMAGE_KEYS = {
    "image_caption": {"status": "complete"},
    "image_embedding": {"status": "complete"},
    "text_embedding": {"status": "complete"},
}


@pytest.fixture(autouse=True)
def _seed_prefix(seeded_prefixes):
    seeded_prefixes.add(PREFIX)


async def _seed_entry(repository, entry_id: str, *, timestamp=None, status=None) -> None:
    async with repository.pool.connection() as conn:
        await conn.execute(
            """
            INSERT INTO enhanced_entries (
                entry_id, source_system, timestamp, author, raw_text,
                attachments, metadata, enhancement_status
            ) VALUES (
                %(entry_id)s, 'test', %(ts)s, 'tester', 'x',
                '[]'::jsonb, '{}'::jsonb, %(status)s::jsonb
            )
            """,
            {
                "entry_id": entry_id,
                "ts": timestamp or datetime.now(UTC),
                "status": json.dumps(status or {}),
            },
        )


async def _seed_row(
    repository,
    entry_id: str,
    attachment_id: str,
    *,
    source_url: str | None = None,
    copy_status: str = "pending",
    skip_reason: str | None = None,
    data: bytes | None = None,
    size_bytes: int | None = None,
    rendition_sha256: str | None = None,
) -> None:
    async with repository.pool.connection() as conn:
        await conn.execute(
            """
            INSERT INTO attachment_files (
                attachment_id, entry_id, filename, mime_type, data, size_bytes,
                source_url, copy_status, skip_reason, rendition_sha256
            ) VALUES (
                %(attachment_id)s, %(entry_id)s, 'f.png', 'image/png', %(data)s,
                %(size_bytes)s, %(source_url)s, %(copy_status)s, %(skip_reason)s,
                %(rendition_sha256)s
            )
            """,
            {
                "attachment_id": attachment_id,
                "entry_id": entry_id,
                "data": data,
                "size_bytes": size_bytes if size_bytes is not None else (len(data or b"")),
                "source_url": source_url,
                "copy_status": copy_status,
                "skip_reason": skip_reason,
                "rendition_sha256": rendition_sha256,
            },
        )


async def _row(repository, attachment_id: str) -> dict | None:
    """The full stored row, blob included, read straight from the table."""
    from psycopg.rows import dict_row

    async with repository.pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT * FROM attachment_files WHERE attachment_id = %(a)s",
                {"a": attachment_id},
            )
            return await cur.fetchone()


async def _status(repository, entry_id: str) -> dict:
    async with repository.pool.connection() as conn:
        result = await conn.execute(
            "SELECT enhancement_status FROM enhanced_entries WHERE entry_id = %(e)s",
            {"e": entry_id},
        )
        return (await result.fetchone())[0]


class TestCopyOutcome:
    async def test_pending_row_becomes_copied_with_rendition_and_clears_image_keys(
        self, repository
    ):
        entry, att = f"{PREFIX}out-1", f"{PREFIX}out-1-a"
        await _seed_entry(repository, entry, status=STATUS_WITH_IMAGE_KEYS)
        await _seed_row(repository, entry, att, source_url="https://x/a.png")

        status = await repository.apply_copy_outcome(
            entry,
            att,
            copy_status="copied",
            data=b"PNGDATA",
            mime_type="image/png",
            size_bytes=7,
            rendition=RENDITION,
        )

        assert status == "copied"
        row = await _row(repository, att)
        assert bytes(row["data"]) == b"PNGDATA"
        assert row["size_bytes"] == 7
        assert row["rendition_sha256"] == RENDITION.sha256
        assert bytes(row["rendition_bytes"]) == RENDITION.data
        assert (row["rendition_w"], row["rendition_h"]) == (4, 3)
        assert row["skip_reason"] is None
        remaining = await _status(repository, entry)
        assert not set(IMAGE_STATUS_KEYS) & set(remaining)
        assert "text_embedding" in remaining

    async def test_config_skipped_row_is_redecided(self, repository):
        entry, att = f"{PREFIX}out-2", f"{PREFIX}out-2-a"
        await _seed_entry(repository, entry)
        await _seed_row(
            repository, entry, att, source_url="https://x/b.png",
            copy_status="skipped", skip_reason="size_cap",
        )  # fmt: skip

        status = await repository.apply_copy_outcome(
            entry, att, copy_status="skipped", mime_type="image/png", skip_reason="fetch_failed"
        )

        assert status == "skipped"
        assert (await _row(repository, att))["skip_reason"] == "fetch_failed"

    async def test_content_skip_is_terminal_zero_rows(self, repository):
        entry, att = f"{PREFIX}out-3", f"{PREFIX}out-3-a"
        await _seed_entry(repository, entry)
        await _seed_row(
            repository, entry, att, source_url="https://x/c.pdf",
            copy_status="copied", skip_reason="not_an_image", data=b"%PDF",
        )  # fmt: skip

        status = await repository.apply_copy_outcome(
            entry, att, copy_status="copied", data=b"other", mime_type="image/png", size_bytes=5
        )

        assert status is None
        row = await _row(repository, att)
        assert bytes(row["data"]) == b"%PDF"
        assert row["skip_reason"] == "not_an_image"

    async def test_zero_rows_never_resurrects_a_deleted_row(self, repository):
        entry, att = f"{PREFIX}out-4", f"{PREFIX}out-4-a"
        await _seed_entry(repository, entry, status=STATUS_WITH_IMAGE_KEYS)
        await _seed_row(repository, entry, att, source_url="https://x/d.png")
        async with repository.pool.connection() as conn:
            await conn.execute(
                "DELETE FROM attachment_files WHERE attachment_id = %(a)s", {"a": att}
            )

        status = await repository.apply_copy_outcome(
            entry,
            att,
            copy_status="copied",
            data=b"PNG",
            mime_type="image/png",
            size_bytes=3,
            rendition=RENDITION,
        )

        assert status is None
        assert await _row(repository, att) is None
        assert set(IMAGE_STATUS_KEYS) <= set(await _status(repository, entry))

    async def test_transient_keeps_pending_and_counts_attempt(self, repository):
        entry, att = f"{PREFIX}out-5", f"{PREFIX}out-5-a"
        await _seed_entry(repository, entry, status=STATUS_WITH_IMAGE_KEYS)
        await _seed_row(repository, entry, att, source_url="https://x/e.png")

        status = await repository.apply_copy_outcome(
            entry, att, copy_status="pending", mime_type="image/png", copy_attempts=1
        )

        assert status == "pending"
        row = await _row(repository, att)
        assert row["copy_attempts"] == 1
        assert row["rendition_sha256"] is None
        assert set(IMAGE_STATUS_KEYS) <= set(await _status(repository, entry))


class TestRenderOnly:
    async def test_render_only_writes_rendition_and_keeps_data(self, repository):
        entry, att = f"{PREFIX}ren-1", f"{PREFIX}ren-1-a"
        await _seed_entry(repository, entry, status=STATUS_WITH_IMAGE_KEYS)
        await _seed_row(repository, entry, att, copy_status="copied", data=b"ORIG")

        assert await repository.get_copy_source(att) == (b"ORIG", "image/png")
        assert await repository.apply_render_outcome(
            entry, att, mime_type="image/png", rendition=RENDITION
        )

        row = await _row(repository, att)
        assert bytes(row["data"]) == b"ORIG"
        assert row["copy_status"] == "copied"
        assert row["rendition_sha256"] == RENDITION.sha256
        assert not set(IMAGE_STATUS_KEYS) & set(await _status(repository, entry))
        # Already rendered: neither a source nor a second render.
        assert await repository.get_copy_source(att) is None
        assert not await repository.apply_render_outcome(
            entry, att, mime_type="image/png", rendition=RENDITION
        )

    async def test_render_only_never_touches_pending_rows(self, repository):
        entry, att = f"{PREFIX}ren-2", f"{PREFIX}ren-2-a"
        await _seed_entry(repository, entry)
        await _seed_row(repository, entry, att, source_url="https://x/f.png")

        assert await repository.get_copy_source(att) is None
        assert not await repository.apply_render_outcome(
            entry, att, mime_type="image/png", skip_reason="decoder_failed"
        )
        assert (await _row(repository, att))["skip_reason"] is None


class TestBudgetAndKeys:
    async def test_budget_counts_only_copied_rows(self, repository):
        entry = f"{PREFIX}bud-1"
        await _seed_entry(repository, entry)
        await _seed_row(repository, entry, f"{entry}-a", copy_status="copied", data=b"12345")
        await _seed_row(repository, entry, f"{entry}-b", copy_status="copied", data=b"123")
        await _seed_row(repository, entry, f"{entry}-c", source_url="https://x/g.png")

        assert await repository.count_copied_attachments(entry) == (2, 8)
        assert await repository.count_copied_attachments(f"{PREFIX}bud-none") == (0, 0)

    async def test_image_key_clear_is_guarded(self, repository):
        entry = f"{PREFIX}keys-1"
        await _seed_entry(repository, entry, status=STATUS_WITH_IMAGE_KEYS)
        async with repository.pool.connection() as conn:
            assert await repository.clear_image_status_keys(conn, entry)
            assert not await repository.clear_image_status_keys(conn, entry)
        assert set(await _status(repository, entry)) == {"text_embedding"}


class TestNativeInsert:
    async def test_native_insert_with_rendition(self, repository):
        entry, att = f"{PREFIX}nat-1", f"{PREFIX}nat-1-a"
        await _seed_entry(repository, entry, status=STATUS_WITH_IMAGE_KEYS)

        await repository.insert_native_attachment(
            entry, att, filename="up.png", mime_type="image/png", data=b"PNGBYTES",
            rendition=RENDITION,
        )  # fmt: skip

        row = await _row(repository, att)
        assert row["source_url"] is None
        assert row["copy_status"] == "copied"
        assert row["size_bytes"] == 8
        assert row["rendition_sha256"] == RENDITION.sha256
        assert not set(IMAGE_STATUS_KEYS) & set(await _status(repository, entry))

    async def test_native_non_image_keeps_data_with_skip(self, repository):
        entry, att = f"{PREFIX}nat-2", f"{PREFIX}nat-2-a"
        await _seed_entry(repository, entry)

        await repository.insert_native_attachment(
            entry, att, filename="doc.pdf", mime_type="application/pdf", data=b"%PDF-1.7",
            skip_reason="reserved_format",
        )  # fmt: skip

        row = await _row(repository, att)
        assert bytes(row["data"]) == b"%PDF-1.7"
        assert row["skip_reason"] == "reserved_format"
        assert row["rendition_sha256"] is None


class TestKeepDeletion:
    async def _seed_mixed(self, repository, entry: str) -> None:
        await _seed_entry(repository, entry)
        await _seed_row(repository, entry, f"{entry}-native", copy_status="copied", data=b"N")
        await _seed_row(repository, entry, f"{entry}-pend", source_url="https://x/p.png")
        await _seed_row(
            repository, entry, f"{entry}-skip", source_url="https://x/s.png",
            copy_status="skipped", skip_reason="size_cap",
        )  # fmt: skip
        await _seed_row(
            repository, entry, f"{entry}-kept", source_url="https://x/k.png",
            copy_status="copied", data=b"K",
        )  # fmt: skip

    async def _ids(self, repository, entry: str) -> set[str]:
        return {row["attachment_id"] for row in await repository.get_copy_rows(entry)}

    async def test_deletes_every_status_for_dropped_urls_never_native(self, repository):
        entry = f"{PREFIX}keep-1"
        await self._seed_mixed(repository, entry)

        async with repository.pool.connection() as conn, conn.transaction():
            await repository.lock_entry(conn, entry)
            deleted = await repository.delete_dropped_attachments(conn, entry, ["https://x/k.png"])

        assert set(deleted) == {f"{entry}-pend", f"{entry}-skip"}
        assert await self._ids(repository, entry) == {f"{entry}-native", f"{entry}-kept"}

    async def test_empty_keep_deletes_all_non_native(self, repository):
        entry = f"{PREFIX}keep-2"
        await self._seed_mixed(repository, entry)

        async with repository.pool.connection() as conn:
            deleted = await repository.delete_dropped_attachments(conn, entry, [])

        assert len(deleted) == 3
        assert await self._ids(repository, entry) == {f"{entry}-native"}


class TestReads:
    async def test_copy_rows_carry_state_and_no_blobs(self, repository):
        entry = f"{PREFIX}rows-1"
        await _seed_entry(repository, entry)
        await _seed_row(repository, entry, f"{entry}-a", copy_status="copied", data=b"D")
        await _seed_row(repository, entry, f"{entry}-b", source_url="https://x/b.png")

        rows = {row["attachment_id"]: row for row in await repository.get_copy_rows(entry)}

        assert set(rows) == {f"{entry}-a", f"{entry}-b"}
        for row in rows.values():
            assert "data" not in row
            assert "rendition_bytes" not in row
        assert rows[f"{entry}-a"]["has_data"] is True
        assert rows[f"{entry}-b"]["has_data"] is False
        assert rows[f"{entry}-b"]["copy_status"] == "pending"
        assert rows[f"{entry}-b"]["source_url"] == "https://x/b.png"
        assert rows[f"{entry}-b"]["copy_attempts"] == 0

    async def test_retry_candidates_newest_first_below_cursor(self, repository):
        older, newer = f"{PREFIX}retry-older", f"{PREFIX}retry-newer"
        done, skipped = f"{PREFIX}retry-done", f"{PREFIX}retry-skipped"
        t_old, t_new = FUTURE, FUTURE + timedelta(days=1)
        await _seed_entry(repository, older, timestamp=t_old)
        await _seed_entry(repository, newer, timestamp=t_new)
        await _seed_entry(repository, done, timestamp=t_new + timedelta(days=1))
        await _seed_entry(repository, skipped, timestamp=t_new + timedelta(days=2))
        await _seed_row(repository, older, f"{older}-a", source_url="https://x/o.png")
        await _seed_row(repository, newer, f"{newer}-a", copy_status="copied", data=b"R")
        await _seed_row(
            repository, done, f"{done}-a", copy_status="copied", data=b"R",
            rendition_sha256="cd" * 32,
        )  # fmt: skip
        await _seed_row(
            repository, skipped, f"{skipped}-a", source_url="https://x/s.png",
            copy_status="skipped", skip_reason="source_gone",
        )  # fmt: skip

        first = await repository.get_copy_retry_candidates(None, 2)
        assert [entry_id for _, entry_id in first] == [newer, older]
        assert first[0][0] == t_new

        rest = await repository.get_copy_retry_candidates(first[0], 10)
        assert rest[0][1] == older
        assert newer not in [entry_id for _, entry_id in rest]

    async def test_copy_counts_and_bytes(self, repository):
        before_pending, before_skipped = await repository.get_attachment_copy_counts()
        entry = f"{PREFIX}counts-1"
        await _seed_entry(repository, entry)
        await _seed_row(repository, entry, f"{entry}-p1", source_url="https://x/1.png")
        await _seed_row(repository, entry, f"{entry}-p2", source_url="https://x/2.png")
        await _seed_row(
            repository, entry, f"{entry}-s", source_url="https://x/3.png",
            copy_status="skipped", skip_reason="origin_not_allowed",
        )  # fmt: skip
        await _seed_row(
            repository, entry, f"{entry}-c", copy_status="copied", data=b"%PDF",
            skip_reason="not_an_image",
        )  # fmt: skip

        pending, skipped = await repository.get_attachment_copy_counts()

        assert pending - before_pending == 2
        for code in ("origin_not_allowed", "not_an_image"):
            assert skipped.get(code, 0) - before_skipped.get(code, 0) == 1
        assert await repository.get_attachment_bytes() > 0


async def _set_rendition(repository, attachment_id: str) -> None:
    async with repository.pool.connection() as conn:
        await conn.execute(
            """
            UPDATE attachment_files
            SET rendition_bytes = %(data)s, rendition_mime = %(mime)s,
                rendition_w = %(w)s, rendition_h = %(h)s, rendition_sha256 = %(sha)s
            WHERE attachment_id = %(attachment_id)s
            """,
            {
                "attachment_id": attachment_id,
                "data": RENDITION.data,
                "mime": RENDITION.mime_type,
                "w": RENDITION.width,
                "h": RENDITION.height,
                "sha": RENDITION.sha256,
            },
        )


class TestAttachmentReaders:
    async def test_attachment_rows_group_by_entry_without_blobs(self, repository):
        first, second = f"{PREFIX}readers-1", f"{PREFIX}readers-2"
        await _seed_entry(repository, first)
        await _seed_entry(repository, second)
        await _seed_row(repository, first, f"{first}-a", copy_status="copied", data=b"D")
        await _seed_row(repository, first, f"{first}-b", source_url="https://x/b.png")
        await _seed_row(repository, second, f"{second}-a", copy_status="copied", data=b"E")
        await _set_rendition(repository, f"{first}-a")

        mapping = await repository.get_attachment_rows([first, second, f"{PREFIX}absent"])

        assert set(mapping) == {first, second}
        assert {row["attachment_id"] for row in mapping[first]} == {f"{first}-a", f"{first}-b"}
        assert [row["attachment_id"] for row in mapping[second]] == [f"{second}-a"]
        for rows in mapping.values():
            for row in rows:
                assert tuple(row) == ATTACHMENT_ROW_COLUMNS
        by_id = {row["attachment_id"]: row for row in mapping[first]}
        assert by_id[f"{first}-a"]["rendition_mime"] == "image/jpeg"
        assert (by_id[f"{first}-a"]["rendition_w"], by_id[f"{first}-a"]["rendition_h"]) == (4, 3)
        assert by_id[f"{first}-a"]["rendition_sha256"] == RENDITION.sha256
        assert by_id[f"{first}-b"]["copy_status"] == "pending"
        assert by_id[f"{first}-b"]["source_url"] == "https://x/b.png"

    async def test_rendition_reads_the_rendition_and_not_the_original(self, repository):
        entry = f"{PREFIX}rendition-1"
        await _seed_entry(repository, entry)
        await _seed_row(repository, entry, f"{entry}-a", copy_status="copied", data=b"ORIG")
        await _seed_row(repository, entry, f"{entry}-b", copy_status="copied", data=b"ORIG")
        await _set_rendition(repository, f"{entry}-a")

        row = await repository.get_rendition(f"{entry}-a")

        assert row is not None
        assert tuple(row) == (*ATTACHMENT_ROW_COLUMNS, "rendition_bytes")
        assert bytes(row["rendition_bytes"]) == RENDITION.data
        assert row["rendition_mime"] == "image/jpeg"
        assert await repository.get_rendition(f"{entry}-b") is None
        assert await repository.get_rendition(f"{PREFIX}absent") is None

    async def test_original_reads_only_copied_rows_with_data(self, repository):
        entry = f"{PREFIX}original-1"
        await _seed_entry(repository, entry)
        await _seed_row(repository, entry, f"{entry}-a", copy_status="copied", data=b"ORIG")
        await _seed_row(repository, entry, f"{entry}-b", source_url="https://x/b.png")
        await _set_rendition(repository, f"{entry}-a")

        row = await repository.get_attachment_original(f"{entry}-a")

        assert row is not None
        assert "rendition_bytes" not in row
        assert bytes(row["data"]) == b"ORIG"
        assert row["mime_type"] == "image/png"
        assert row["filename"] == "f.png"
        assert await repository.get_attachment_original(f"{entry}-b") is None
        assert await repository.get_attachment_original(f"{PREFIX}absent") is None

    async def test_original_falls_back_to_the_b1_statement(self, repository, monkeypatch):
        entry = f"{PREFIX}original-b1"
        await _seed_entry(repository, entry)
        await _seed_row(repository, entry, f"{entry}-a", copy_status="pending", data=b"OLD")

        async def _unmigrated():
            return SchemaFacts(has_v2_fts=False, has_copy_state=False)

        monkeypatch.setattr(repository, "schema_facts", _unmigrated)

        row = await repository.get_attachment_original(f"{entry}-a")

        assert row is not None
        assert bytes(row["data"]) == b"OLD"
        assert row["filename"] == "f.png"
        assert row["mime_type"] == "image/png"
        assert await repository.get_attachment_rows([entry]) is None
        assert await repository.get_rendition(f"{entry}-a") is None
