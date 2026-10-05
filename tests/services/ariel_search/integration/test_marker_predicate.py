"""Marker-aware enhancement status against a real database.

A marker module records, in its status object, the marker (the caption model,
the image table) the status was written under. A status under any other marker
reads as pending with no attempts: in the to-do walk, in its order, and in the
counts ``status`` reports. Every test runs on a fresh scratch database so the
counts cover exactly the rows it seeds.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import pytest

from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.database.repository import (
    MAX_ENHANCEMENT_ATTEMPTS,
    ARIELRepository,
)

# xdist_group("docker"): every container-starting test file shares one worker, so a
# run has a single testcontainers session and the shared database is serialized.
pytestmark = [pytest.mark.asyncio, pytest.mark.xdist_group("docker"), pytest.mark.timeout(120)]

BASE = datetime(2026, 9, 1, tzinfo=UTC)
CAPTION = "image_caption"
MODULE = "marker_probe"


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
    at: datetime = BASE,
    status: dict[str, Any] | None = None,
    captions: dict[str, Any] | None = None,
) -> None:
    with psycopg.connect(uri, autocommit=True) as conn:
        conn.execute(
            """
            INSERT INTO enhanced_entries (
                entry_id, source_system, timestamp, raw_text, enhancement_status,
                attachment_captions
            ) VALUES (%s, 'test', %s, 'text', %s::jsonb, %s::jsonb)
            """,
            (
                entry_id,
                at,
                json.dumps(status or {}),
                None if captions is None else json.dumps(captions),
            ),
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


def _status(uri: str, entry_id: str) -> dict[str, Any]:
    with psycopg.connect(uri) as conn:
        row = conn.execute(
            "SELECT enhancement_status FROM enhanced_entries WHERE entry_id = %s", (entry_id,)
        ).fetchone()
    assert row is not None
    return row[0]


class TestModelSwitch:
    async def test_three_failures_then_a_model_switch_restart_the_count(
        self, repo: ARIELRepository, scratch_database: str
    ) -> None:
        """Giving up under model A is forgotten under model B: pending, attempts from 1."""
        _seed(scratch_database, "e-1")
        _picture(scratch_database, "a-1", "e-1", copy_status="copied")

        attempts = [
            await repo.mark_enhancement_failed("e-1", CAPTION, "timeout", marker="vis-a")
            for _ in range(MAX_ENHANCEMENT_ATTEMPTS)
        ]

        assert attempts == [1, 2, 3]
        stored = _status(scratch_database, "e-1")[CAPTION]
        assert stored["gave_up"] is True
        assert stored["marker"] == "vis-a"
        assert await repo.get_incomplete_entries(CAPTION, marker="vis-a") == []
        stats = await repo.get_enhancement_stats(markers={CAPTION: "vis-a"})
        assert stats[CAPTION] == {"complete": 0, "failed": 1, "pending": 0, "gave_up": 1}

        walked = await repo.get_incomplete_entries(CAPTION, marker="vis-b")
        stats = await repo.get_enhancement_stats(markers={CAPTION: "vis-b"})

        assert [e["entry_id"] for e in walked] == ["e-1"]
        assert {k: stats[CAPTION][k] for k in ("failed", "gave_up", "pending")} == {
            "failed": 0,
            "gave_up": 0,
            "pending": 1,
        }
        assert await repo.mark_enhancement_failed("e-1", CAPTION, "t", marker="vis-b") == 1
        assert await repo.mark_enhancement_failed("e-1", CAPTION, "t", marker="vis-b") == 2
        stored = _status(scratch_database, "e-1")[CAPTION]
        assert stored["gave_up"] is False
        assert stored["marker"] == "vis-b"
        assert [e["entry_id"] for e in await repo.get_incomplete_entries(CAPTION, marker="vis-b")]

    async def test_complete_under_the_current_marker_leaves_the_walk(
        self, repo: ARIELRepository, scratch_database: str
    ) -> None:
        """The complete object stores the marker; a new marker brings the entry back."""
        _seed(scratch_database, "e-1")
        _picture(scratch_database, "a-1", "e-1", copy_status="copied")

        await repo.mark_enhancement_complete("e-1", CAPTION, marker="vis-a")

        stored = _status(scratch_database, "e-1")[CAPTION]
        assert stored["status"] == "complete"
        assert stored["marker"] == "vis-a"
        assert "completed_at" in stored
        assert await repo.get_incomplete_entries(CAPTION, marker="vis-a") == []
        assert len(await repo.get_incomplete_entries(CAPTION, marker="vis-b")) == 1


class TestImageTodo:
    async def test_pending_copies_are_not_walked(
        self, repo: ARIELRepository, scratch_database: str
    ) -> None:
        """500 entries whose only picture is still copying stay out of the walk."""
        with psycopg.connect(scratch_database, autocommit=True) as conn:
            conn.execute(
                """
                INSERT INTO enhanced_entries (entry_id, source_system, timestamp, raw_text)
                SELECT 'p-' || lpad(i::text, 4, '0'), 'test', %s, 'text'
                FROM generate_series(1, 500) AS i
                """,
                (BASE,),
            )
            conn.execute(
                """
                INSERT INTO attachment_files (
                    attachment_id, entry_id, filename, mime_type, source_url, copy_status
                )
                SELECT 'pa-' || lpad(i::text, 4, '0'), 'p-' || lpad(i::text, 4, '0'),
                       'f.png', 'image/png', 'https://h.example/f.png', 'pending'
                FROM generate_series(1, 500) AS i
                """
            )
        _seed(scratch_database, "c-1")
        _picture(scratch_database, "ca-1", "c-1", copy_status="copied")

        walked = await repo.get_incomplete_entries(CAPTION, marker="vis-a", limit=1000)

        assert [e["entry_id"] for e in walked] == ["c-1"]
        stats = await repo.get_enhancement_stats(markers={CAPTION: "vis-a"})
        # No entry carries the key yet, so no module row exists: all 501 stay unfinished.
        assert stats == {"total_entries": 501}

    async def test_captioned_pictures_are_done_and_null_captions_are_not(
        self, repo: ARIELRepository, scratch_database: str
    ) -> None:
        """A caption under the current model is done; a NULL caption column is not done."""
        _seed(scratch_database, "done", captions={"d-1": {"vis-a": {"caption": "x"}}})
        _picture(scratch_database, "d-1", "done", copy_status="copied")
        _seed(scratch_database, "null")
        _picture(scratch_database, "n-1", "null", copy_status="copied")

        walked = await repo.get_incomplete_entries(CAPTION, marker="vis-a")
        switched = await repo.get_incomplete_entries(CAPTION, marker="vis-b")

        assert [e["entry_id"] for e in walked] == ["null"]
        assert sorted(e["entry_id"] for e in switched) == ["done", "null"]


class TestCatchupOrder:
    async def test_pending_newest_first_then_failed_by_attempts(
        self, repo: ARIELRepository, scratch_database: str
    ) -> None:
        """Key-absent, pending and stale rows sort as pending; failed rows follow."""
        hour = timedelta(hours=1)
        _seed(scratch_database, "absent", at=BASE)
        _seed(
            scratch_database,
            "pending",
            at=BASE + hour,
            status={MODULE: {"status": "pending", "marker": "m-a"}},
        )
        _seed(
            scratch_database,
            "stale",
            at=BASE + 2 * hour,
            status={
                MODULE: {"status": "failed", "attempts": 3, "gave_up": True, "marker": "m-old"}
            },
        )
        _seed(
            scratch_database,
            "failed-2",
            at=BASE + 5 * hour,
            status={MODULE: {"status": "failed", "attempts": 2, "marker": "m-a"}},
        )
        _seed(
            scratch_database,
            "failed-1",
            at=BASE - hour,
            status={MODULE: {"status": "failed", "attempts": 1, "marker": "m-a"}},
        )
        _seed(
            scratch_database,
            "gave-up",
            at=BASE + 9 * hour,
            status={MODULE: {"status": "failed", "attempts": 3, "gave_up": True, "marker": "m-a"}},
        )
        _seed(
            scratch_database,
            "complete",
            at=BASE + 9 * hour,
            status={MODULE: {"status": "complete", "marker": "m-a"}},
        )

        walked = await repo.get_incomplete_entries(MODULE, marker="m-a")

        assert [e["entry_id"] for e in walked] == [
            "stale",
            "pending",
            "absent",
            "failed-1",
            "failed-2",
        ]


class TestStats:
    async def test_counts_partition_the_store_and_gave_up_is_a_subset_of_failed(
        self, repo: ARIELRepository, scratch_database: str
    ) -> None:
        """complete + failed + pending == total; stale rows count pending, never gave_up."""
        rows: dict[str, dict[str, Any]] = {
            "ok": {"status": "complete", "marker": "vis-a"},
            "stale-ok": {"status": "complete", "marker": "vis-old"},
            "stale-gave-up": {
                "status": "failed",
                "attempts": 3,
                "gave_up": True,
                "marker": "vis-old",
            },
            "failed": {"status": "failed", "attempts": 1, "gave_up": False, "marker": "vis-a"},
            "gave-up": {"status": "failed", "attempts": 3, "gave_up": True, "marker": "vis-a"},
            "pending": {"status": "pending", "marker": "vis-a"},
        }
        for entry_id, state in rows.items():
            _seed(
                scratch_database,
                entry_id,
                status={CAPTION: state, "text_embedding": {"status": "complete"}},
            )
        _seed(scratch_database, "absent")

        stats = await repo.get_enhancement_stats(markers={CAPTION: "vis-a"})

        counts = stats[CAPTION]
        assert counts == {"complete": 1, "failed": 2, "pending": 4, "gave_up": 1}
        assert counts["complete"] + counts["failed"] + counts["pending"] == stats["total_entries"]
        assert counts["gave_up"] <= counts["failed"]
        assert stats["text_embedding"] == {"complete": 6, "failed": 0, "pending": 1}

    async def test_a_module_absent_from_markers_keeps_b1_counts(
        self, repo: ARIELRepository, scratch_database: str
    ) -> None:
        """Without markers the counts are B1's, exactly, whatever the markers stored."""
        _seed(scratch_database, "a", status={CAPTION: {"status": "complete", "marker": "x"}})
        _seed(scratch_database, "b", status={CAPTION: {"status": "failed", "gave_up": True}})
        _seed(scratch_database, "c")

        b1 = await repo.get_enhancement_stats()
        other = await repo.get_enhancement_stats(markers={"image_embedding": "img_t"})

        assert b1 == {
            "total_entries": 3,
            CAPTION: {"complete": 1, "failed": 1, "pending": 1},
        }
        assert other == b1
