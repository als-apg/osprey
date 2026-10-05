"""``osprey ariel ingest`` and quickstart copy an entry's pictures on real rows.

Both commands open their own service on a fresh scratch database, ingest one
reference-adapter entry carrying one PNG attachment with no enhancement module configured,
and leave the attachment ``copied`` with a rendition. A second ingest of the
same entry fetches nothing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import psycopg
import pytest

from osprey.services.ariel_search import cli_operations as ops
from osprey.services.ariel_search.attachments.fetch import FetchOutcome
from osprey.services.ariel_search.config import ARIELConfig

# xdist_group("docker"): every container-starting test file shares one worker, so a
# run has a single testcontainers session and the shared database is serialized.
pytestmark = [pytest.mark.asyncio, pytest.mark.xdist_group("docker"), pytest.mark.timeout(120)]

PNG_MAGIC = b"\x89PNG\r\n\x1a\n" + b"\x00" * 40
ENTRY_ID = "cliingest-1"
PICTURE_PATH = "attachments/2026/10/beam_profile.png"


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
            rendition_sha256="cd" * 32,
        )

    monkeypatch.setattr(prepare_mod, "prepare_picture", _prepare)


@pytest.fixture
def als_source(tmp_path: Path) -> Path:
    """A reference-adapter JSONL export holding one entry with one PNG attachment."""
    path = tmp_path / "als_entries.jsonl"
    entry = {
        "id": ENTRY_ID,
        "timestamp": "1790000000",
        "author": "operator",
        "subject": "Beam profile",
        "details": "Screen image of the injected beam.",
        "category": "Operations",
        "tag": "0",
        "linkedto": "0",
        "level": "entry",
        "attachments": [{"url": PICTURE_PATH}],
    }
    path.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    return path


def _config_dict(uri: str, source: Path) -> dict[str, Any]:
    return {
        "database": {"uri": uri},
        "ingestion": {"adapter": "als_logbook", "source_url": str(source)},
    }


async def _migrate(config_dict: dict[str, Any]) -> None:
    """Bring the scratch database to today's schema, as ``osprey ariel migrate`` does."""
    from osprey.services.ariel_search.database.connection import create_connection_pool
    from osprey.services.ariel_search.database.migrations import run_migrations

    config = ARIELConfig.from_dict(json.loads(json.dumps(config_dict)))
    pool = await create_connection_pool(config.database)
    try:
        await run_migrations(pool, config)
    finally:
        await pool.close()


def _copy_rows(uri: str) -> list[dict[str, Any]]:
    with psycopg.connect(uri) as conn:
        rows = conn.execute(
            """
            SELECT copy_status, rendition_sha256, data
            FROM attachment_files WHERE entry_id = %s
            """,
            (ENTRY_ID,),
        ).fetchall()
    return [{"copy_status": r[0], "rendition_sha256": r[1], "data": r[2]} for r in rows]


def _assert_copied(uri: str) -> None:
    rows = _copy_rows(uri)
    assert len(rows) == 1
    (row,) = rows
    assert row["copy_status"] == "copied"
    assert row["rendition_sha256"] == "cd" * 32
    assert bytes(row["data"]) == PNG_MAGIC


class TestRunIngestCopies:
    async def test_run_ingest_copies_the_picture_and_reingest_fetches_nothing(
        self, monkeypatch, scratch_database, als_source, attachment_fetch
    ):
        _fake_prepare(monkeypatch)
        attachment_fetch.respond(FetchOutcome(data=PNG_MAGIC))
        await _migrate(_config_dict(scratch_database, als_source))

        out = await ops.run_ingest(
            _config_dict(scratch_database, als_source),
            source=str(als_source),
            adapter=None,
            since=None,
            limit=None,
            dry_run=False,
        )

        assert (out.count, out.enhanced_count, out.failed_count) == (1, 0, 0)
        assert out.enhancer_names == []
        _assert_copied(scratch_database)
        picture_fetches = [c for c in attachment_fetch.calls if c["url"].endswith(PICTURE_PATH)]
        assert len(picture_fetches) == 1
        fetches_after_first = len(attachment_fetch.calls)

        again = await ops.run_ingest(
            _config_dict(scratch_database, als_source),
            source=str(als_source),
            adapter=None,
            since=None,
            limit=None,
            dry_run=False,
        )

        assert again.count == 1
        assert len(attachment_fetch.calls) == fetches_after_first
        _assert_copied(scratch_database)


class TestRunQuickstartCopies:
    async def test_run_quickstart_copies_the_picture_and_reingest_fetches_nothing(
        self, monkeypatch, scratch_database, als_source, attachment_fetch
    ):
        _fake_prepare(monkeypatch)
        attachment_fetch.respond(FetchOutcome(data=PNG_MAGIC))

        out = await ops.run_quickstart(_config_dict(scratch_database, als_source), source=None)

        assert (out.count, out.enhanced_count, out.failed_count) == (1, 0, 0)
        assert out.migrations_applied > 0
        _assert_copied(scratch_database)
        fetches_after_first = len(attachment_fetch.calls)
        assert fetches_after_first == 1

        again = await ops.run_quickstart(_config_dict(scratch_database, als_source), source=None)

        assert again.count == 1
        assert len(attachment_fetch.calls) == fetches_after_first
        _assert_copied(scratch_database)
