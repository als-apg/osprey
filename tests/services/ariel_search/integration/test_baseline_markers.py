"""Text-module markers: the one-shot baseline stamp and marker writes, on a real database.

A text module's status written before markers existed carries none. Such a
status is never owed or counted pending on its marker alone; the baseline stamp
gives it the current marker (and a legacy failure its ``gave_up``), so a later
marker change, and only that, makes it stale. Every test runs on a fresh
scratch database so the counts cover exactly the rows it seeds.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import psycopg
import pytest

from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.database.repository import (
    MAX_ENHANCEMENT_ATTEMPTS,
    ARIELRepository,
    attachment_text_md5,
)
from osprey.services.ariel_search.enhancement.status import qmd_export_marker
from tests.services.ariel_search.integration._support import enhancement_status

pytestmark = [pytest.mark.asyncio, pytest.mark.xdist_group("docker"), pytest.mark.timeout(120)]

SUMMARY = "semantic_processor"
QMD = "qmd_export"
BASE = datetime(2026, 9, 1, tzinfo=UTC)


def _seed(
    uri: str,
    entry_id: str,
    status: dict[str, Any] | None = None,
    *,
    attachment_text: str | None = None,
) -> None:
    with psycopg.connect(uri, autocommit=True) as conn:
        conn.execute(
            """
            INSERT INTO enhanced_entries (
                entry_id, source_system, timestamp, raw_text, enhancement_status,
                attachment_text
            ) VALUES (%s, 'test', %s, 'text', %s::jsonb, %s)
            """,
            (entry_id, BASE, json.dumps(status or {}), attachment_text),
        )


def _legacy_complete() -> dict[str, Any]:
    return {"status": "complete", "completed_at": "2026-01-01"}


def _legacy_failed(attempts: int) -> dict[str, Any]:
    return {"status": "failed", "failed_at": "2026-01-01", "error": "x", "attempts": attempts}


async def _owed(repo: ARIELRepository, module: str, marker: str) -> list[str]:
    return [e["entry_id"] for e in await repo.get_incomplete_entries(module, marker=marker)]


def _qmd_marker(mirror_path: str) -> str:
    config = ARIELConfig.from_dict(
        {
            "database": {"uri": "postgresql://unused/unused"},
            "enhancement_modules": {QMD: {"enabled": True, "mirror_path": mirror_path}},
        }
    )
    marker = qmd_export_marker(config)
    assert marker is not None
    return marker


class TestBaselineMarkerUpgrade:
    async def test_legacy_completions_are_not_stale_before_or_after_the_stamp(
        self, scratch_repo: ARIELRepository, scratch_database: str
    ) -> None:
        """Upgrade shows 0 stale; a model change afterwards makes them stale."""
        _seed(scratch_database, "e-1", {SUMMARY: _legacy_complete()})
        _seed(scratch_database, "e-2", {SUMMARY: _legacy_complete()})

        # Before the stamp: a missing text marker is never owed or counted pending.
        assert await _owed(scratch_repo, SUMMARY, "default") == []
        stats = await scratch_repo.get_enhancement_stats(markers={SUMMARY: "default"})
        assert stats[SUMMARY] == {"complete": 2, "failed": 0, "pending": 0, "gave_up": 0}

        stamped = await scratch_repo.stamp_baseline_markers({SUMMARY: "default"})

        assert stamped == {SUMMARY: 2}
        assert enhancement_status(scratch_database, "e-1")[SUMMARY]["marker"] == "default"
        assert await _owed(scratch_repo, SUMMARY, "default") == []
        stats = await scratch_repo.get_enhancement_stats(markers={SUMMARY: "default"})
        assert stats[SUMMARY]["pending"] == 0
        assert stats[SUMMARY]["complete"] == 2

        # A model change: both are now stale, owed and counted pending.
        assert sorted(await _owed(scratch_repo, SUMMARY, "gpt-new")) == ["e-1", "e-2"]
        stats = await scratch_repo.get_enhancement_stats(markers={SUMMARY: "gpt-new"})
        assert stats[SUMMARY] == {"complete": 0, "failed": 0, "pending": 2, "gave_up": 0}

    async def test_legacy_failure_that_used_its_attempts_is_not_requeued(
        self, scratch_repo: ARIELRepository, scratch_database: str
    ) -> None:
        """A legacy failure at the cap gives up; one below the cap stays owed."""
        _seed(scratch_database, "spent", {SUMMARY: _legacy_failed(MAX_ENHANCEMENT_ATTEMPTS)})
        _seed(scratch_database, "fresh", {SUMMARY: _legacy_failed(1)})

        await scratch_repo.stamp_baseline_markers({SUMMARY: "default"})

        spent = enhancement_status(scratch_database, "spent")[SUMMARY]
        assert spent["gave_up"] is True
        assert spent["marker"] == "default"
        assert spent["attempts"] == MAX_ENHANCEMENT_ATTEMPTS
        assert enhancement_status(scratch_database, "fresh")[SUMMARY]["gave_up"] is False
        assert sorted(await _owed(scratch_repo, SUMMARY, "default")) == ["fresh"]
        stats = await scratch_repo.get_enhancement_stats(markers={SUMMARY: "default"})
        assert stats[SUMMARY]["failed"] == 2
        assert stats[SUMMARY]["gave_up"] == 1

    async def test_qmd_mirror_root_change_makes_a_completed_entry_stale(
        self, scratch_repo: ARIELRepository, scratch_database: str
    ) -> None:
        """The stamped mirror root differs from the new one: the entry is owed again."""
        old, new = _qmd_marker("/mirror/a"), _qmd_marker("/mirror/b")
        assert old != new
        _seed(scratch_database, "e-1", {QMD: _legacy_complete()})

        await scratch_repo.stamp_baseline_markers({QMD: old})

        assert await _owed(scratch_repo, QMD, old) == []
        assert await _owed(scratch_repo, QMD, new) == ["e-1"]
        stats = await scratch_repo.get_enhancement_stats(markers={QMD: new})
        assert stats[QMD]["pending"] == 1

    async def test_second_stamp_is_a_no_op(
        self, scratch_repo: ARIELRepository, scratch_database: str
    ) -> None:
        """Statuses that carry a marker, pending ones and other modules are untouched."""
        _seed(scratch_database, "e-1", {SUMMARY: _legacy_complete()})
        _seed(scratch_database, "e-2", {SUMMARY: {"status": "pending"}})
        _seed(
            scratch_database,
            "e-3",
            {"image_caption": {"status": "complete"}, "text_embedding": {"status": "complete"}},
        )
        markers = {
            SUMMARY: "default",
            QMD: None,
            "image_caption": "vis-a",
            "text_embedding": "k",
        }

        first = await scratch_repo.stamp_baseline_markers(markers)
        after_first = {e: enhancement_status(scratch_database, e) for e in ("e-1", "e-2", "e-3")}
        second = await scratch_repo.stamp_baseline_markers({**markers, SUMMARY: "gpt-new"})

        assert first == {SUMMARY: 1}
        assert second == {SUMMARY: 0}
        assert {
            e: enhancement_status(scratch_database, e) for e in ("e-1", "e-2", "e-3")
        } == after_first
        assert "marker" not in after_first["e-2"][SUMMARY]
        assert "marker" not in after_first["e-3"]["image_caption"]


class TestMarkerMd5:
    async def test_marker_md5_complete_stores_the_marker_under_the_md5_guard(
        self, scratch_repo: ARIELRepository, scratch_database: str
    ) -> None:
        """``qmd_export`` completes under both: the marker is stored while the md5 holds."""
        marker = _qmd_marker("/mirror/a")
        _seed(scratch_database, "e-1", attachment_text="caption one")
        md5 = attachment_text_md5({"attachment_text": "caption one"})

        await scratch_repo.mark_enhancement_complete("e-1", QMD, marker=marker, md5=md5)

        stored = enhancement_status(scratch_database, "e-1")[QMD]
        assert stored["status"] == "complete"
        assert stored["marker"] == marker
        assert await _owed(scratch_repo, QMD, marker) == []

    async def test_marker_md5_mismatch_writes_nothing(
        self, scratch_repo: ARIELRepository, scratch_database: str
    ) -> None:
        """A caption that changed meanwhile keeps the module owed, marker or not."""
        marker = _qmd_marker("/mirror/a")
        _seed(scratch_database, "e-1", attachment_text="caption two")
        md5 = attachment_text_md5({"attachment_text": "caption one"})

        await scratch_repo.mark_enhancement_complete("e-1", QMD, marker=marker, md5=md5)

        assert QMD not in enhancement_status(scratch_database, "e-1")
        assert await _owed(scratch_repo, QMD, marker) == ["e-1"]

    async def test_marker_md5_failure_stores_the_marker_and_counts(
        self, scratch_repo: ARIELRepository, scratch_database: str
    ) -> None:
        """A text failure under a marker records it and gives up at the cap."""
        _seed(scratch_database, "e-1")
        attempts = [
            await scratch_repo.mark_enhancement_failed("e-1", SUMMARY, "boom", marker="default")
            for _ in range(MAX_ENHANCEMENT_ATTEMPTS)
        ]

        stored = enhancement_status(scratch_database, "e-1")[SUMMARY]
        assert attempts == [1, 2, 3]
        assert stored["marker"] == "default"
        assert stored["gave_up"] is True
        assert await _owed(scratch_repo, SUMMARY, "default") == []


class _FakeRepo:
    def __init__(self, fail: bool = False) -> None:
        self.calls: list[dict[str, str | None]] = []
        self.fail = fail

    async def stamp_baseline_markers(self, markers: dict[str, str | None]) -> dict[str, int]:
        self.calls.append(dict(markers))
        if self.fail:
            raise RuntimeError("db down")
        return dict.fromkeys(markers, 0)


class TestBaselineMarkerOncePerProcess:
    async def test_baseline_marker_stamp_runs_once_per_module(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from osprey.services.ariel_search import cli_operations

        monkeypatch.setattr(cli_operations, "_BASELINE_STAMPED", set())
        config = ARIELConfig.from_dict({"database": {"uri": "postgresql://h/once"}})
        repo = _FakeRepo()

        await cli_operations._stamp_baseline_markers(config, repo, {SUMMARY: "default"})
        await cli_operations._stamp_baseline_markers(
            config, repo, {SUMMARY: "default", QMD: "1:/m:none"}
        )
        await cli_operations._stamp_baseline_markers(config, repo, {SUMMARY: "x", QMD: None})

        assert repo.calls == [{SUMMARY: "default"}, {QMD: "1:/m:none"}]

    async def test_baseline_marker_stamp_failure_is_retried(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from osprey.services.ariel_search import cli_operations

        monkeypatch.setattr(cli_operations, "_BASELINE_STAMPED", set())
        config = ARIELConfig.from_dict({"database": {"uri": "postgresql://h/retry"}})

        with pytest.raises(RuntimeError):
            await cli_operations._stamp_baseline_markers(
                config, _FakeRepo(fail=True), {SUMMARY: "default"}
            )
        repo = _FakeRepo()
        await cli_operations._stamp_baseline_markers(config, repo, {SUMMARY: "default"})

        assert repo.calls == [{SUMMARY: "default"}]


class _FakeEnhancer:
    name = SUMMARY
    runs_inline = True

    def __init__(self) -> None:
        self.ran = False

    async def enhance(self, _entry: Any, _conn: Any) -> None:
        self.ran = True


class _FakeService:
    def __init__(self, repository: Any) -> None:
        self.repository = repository

    async def __aenter__(self) -> _FakeService:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None


class TestBaselineMarkerAbortsTextStage:
    async def test_baseline_marker_stamp_failure_aborts_the_text_stage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No entry is read or enhanced when the stamp fails."""
        import osprey.services.ariel_search as ariel_search
        import osprey.services.ariel_search.enhancement as enhancement
        from osprey.services.ariel_search import cli_operations

        monkeypatch.setattr(cli_operations, "_BASELINE_STAMPED", set())
        enhancer = _FakeEnhancer()
        repo = _FakeRepo(fail=True)

        async def _service(_config: Any, *, store_deadline_s=None) -> _FakeService:  # noqa: ARG001 - the signature it stands in for
            return _FakeService(repo)

        monkeypatch.setattr(ariel_search, "create_ariel_service", _service)
        monkeypatch.setattr(enhancement, "create_enhancers_from_config", lambda *a, **k: [enhancer])
        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://h/abort"},
                "enhancement_modules": {SUMMARY: {"enabled": True, "provider": "cborg"}},
            }
        )
        messages: list[str] = []

        result = await cli_operations._run_text_enhance(
            config, None, False, 10, messages.append, stop_event=None
        )

        assert repo.calls == [{SUMMARY: "default"}]
        assert result.entries_processed == 0
        assert enhancer.ran is False
        assert any("baseline marker stamp failed" in m for m in messages)
