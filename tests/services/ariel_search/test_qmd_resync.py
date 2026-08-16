"""Tests for the qmd markdown-mirror resync pass and its CLI surface.

The resync exists because several mutation paths write straight to
``enhanced_entries`` and never reach an enhancer, so the mirror the qmd sidecar
searches would silently fall behind the database. Three properties carry that
guarantee and are asserted here:

* **The watermark advances and is honoured.** A pass records the highest
  ``updated_at`` it saw; the next pass asks the database only for rows at or
  after it. The bound is inclusive on purpose -- a row sharing the watermark's
  timestamp is re-read rather than skipped.
* **Unchanged content is not rewritten.** A second pass over the same rows
  renders the same bytes, so no file is touched and the sidecar's freshness
  marker does not move. This is what makes watermark churn from bookkeeping
  updates free.
* **``--rebuild`` is the only pass that removes.** An incremental pass cannot
  know an entry was deleted; the rebuild wipes the shard tree first, which is
  what recovers a mirror after ``osprey ariel purge``.

The database is the fake pool/cursor family from ``conftest.py`` -- the
assertions are about the SQL emitted and the files that land on disk. The
mirror itself is real: writes go to ``tmp_path`` through the production writer,
because "did the bytes change" is the contract under test.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from osprey.cli.ariel import ariel_group
from osprey.services.ariel_search import cli_operations as ops
from osprey.services.ariel_search.database.migrations import KNOWN_MIGRATIONS
from osprey.services.ariel_search.database.qmd_resync_migration import QmdResyncMigration
from osprey.services.ariel_search.enhancement.qmd_export import TOUCH_MARKER_NAME
from osprey.services.ariel_search.enhancement.qmd_export.writer import encode_entry_id
from tests.services.ariel_search._cli_ops_doubles import _patch_pool
from tests.services.ariel_search.conftest import _FakePool

_DB = {"database": {"uri": "postgresql://localhost/test"}}

_EARLIER = datetime(2026, 8, 14, 9, 0, tzinfo=UTC)
_LATER = datetime(2026, 8, 15, 17, 30, tzinfo=UTC)


def _config(mirror: Path | None = None, enabled: bool = True) -> dict:
    """Build an ``ariel`` config section with qmd_export configured."""
    module: dict[str, Any] = {"enabled": enabled}
    if mirror is not None:
        module["mirror_path"] = str(mirror)
    return {**_DB, "enhancement_modules": {"qmd_export": module}}


def _row(entry_id: str, updated_at: datetime, raw_text: str = "beam dump at 09:00") -> dict:
    """Build one ``enhanced_entries`` row as ``dict_row`` would hand it over."""
    return {
        "entry_id": entry_id,
        "source_system": "als_logbook",
        "timestamp": datetime(2026, 8, 14, 8, 0, tzinfo=UTC),
        "author": "operator",
        "raw_text": raw_text,
        "attachments": [],
        "metadata": {},
        "created_at": _EARLIER,
        "updated_at": updated_at,
    }


def _mirrored(mirror: Path, entry_id: str) -> Path:
    """Path the mirror writer gives an entry timestamped 2026-08."""
    return mirror / "2026" / "08" / f"{encode_entry_id(entry_id)}.md"


class _PagingPool:
    """Pool stand-in that hands out one scripted page per ``execute``.

    The shared fake cursor is deliberately non-consuming, which cannot express
    keyset pagination, so the paging contract gets its own double. Every
    executed ``(sql, params)`` pair is recorded on ``calls`` in order.
    """

    def __init__(self, pages: list[list[dict]]) -> None:
        self.pages = list(pages)
        self.calls: list[tuple[str, Any]] = []
        self.close_calls = 0

    def connection(self) -> _PagingPool:
        return self

    def cursor(self, row_factory: Any = None) -> _PagingPool:
        return self

    async def __aenter__(self) -> _PagingPool:
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        return False

    async def execute(self, sql: str, params: Any = None) -> _PagingPool:
        self.calls.append((sql, params))
        return self

    async def fetchall(self) -> list[dict]:
        return self.pages.pop(0) if self.pages else []

    async def close(self) -> None:
        self.close_calls += 1


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------


class TestQmdResyncMigration:
    """The additive index that keeps the resync scan off a sequential scan."""

    def test_identity(self) -> None:
        migration = QmdResyncMigration()
        assert migration.name == "qmd_resync_index"
        assert migration.depends_on == ["core_schema"]

    @pytest.mark.asyncio
    async def test_up_creates_the_updated_at_index(self, ddl_conn) -> None:
        await QmdResyncMigration().up(ddl_conn)

        joined = " ".join(" ".join(ddl_conn.sql).split())
        assert "CREATE INDEX IF NOT EXISTS idx_entries_updated_at" in joined
        assert "ON enhanced_entries(updated_at)" in joined

    @pytest.mark.asyncio
    async def test_up_is_additive(self, ddl_conn) -> None:
        """No table is created, altered or dropped -- only an index appears."""
        await QmdResyncMigration().up(ddl_conn)

        for statement in ddl_conn.sql:
            upper = statement.upper()
            assert "ALTER TABLE" not in upper
            assert "CREATE TABLE" not in upper
            assert "DROP" not in upper

    @pytest.mark.asyncio
    async def test_down_drops_only_the_index(self, ddl_conn) -> None:
        await QmdResyncMigration().down(ddl_conn)

        joined = " ".join(ddl_conn.sql)
        assert "DROP INDEX IF EXISTS idx_entries_updated_at" in joined
        assert "TABLE" not in joined.upper()

    def test_registered_behind_the_qmd_export_module(self) -> None:
        """The index ships only where the mirror it serves is enabled."""
        entries = [row for row in KNOWN_MIGRATIONS if row[0] == "qmd_resync_index"]
        assert len(entries) == 1
        _, module_path, class_name, requires_module = entries[0]
        assert module_path.endswith("database.qmd_resync_migration")
        assert class_name == "QmdResyncMigration"
        assert requires_module == "qmd_export"


# ---------------------------------------------------------------------------
# Configuration gate
# ---------------------------------------------------------------------------


class TestResyncConfiguration:
    """What the pass does before it ever reaches the database."""

    @pytest.mark.asyncio
    async def test_disabled_module_is_a_no_op(self, monkeypatch, tmp_path) -> None:
        """No mirror configured means nothing to keep in sync, not an error."""
        _patch_pool(monkeypatch, _FakePool())

        assert await ops.run_qmd_resync(_config(tmp_path, enabled=False)) is None

    @pytest.mark.asyncio
    async def test_absent_module_is_a_no_op(self, monkeypatch) -> None:
        _patch_pool(monkeypatch, _FakePool())

        assert await ops.run_qmd_resync(dict(_DB)) is None

    @pytest.mark.asyncio
    async def test_enabled_without_a_mirror_path_is_an_error(self, monkeypatch) -> None:
        """An enabled exporter with nowhere to write is broken config."""
        _patch_pool(monkeypatch, _FakePool())

        with pytest.raises(ValueError, match="mirror_path"):
            await ops.run_qmd_resync(_config(None))


# ---------------------------------------------------------------------------
# Incremental pass
# ---------------------------------------------------------------------------


class TestIncrementalResync:
    """Scan, export, advance the watermark."""

    @pytest.mark.asyncio
    async def test_exports_rows_and_records_the_watermark(self, monkeypatch, tmp_path) -> None:
        pool = _FakePool(rows_for={"FROM enhanced_entries": [_row("42", _EARLIER)]})
        _patch_pool(monkeypatch, pool)

        result = await ops.run_qmd_resync(_config(tmp_path))

        assert result is not None
        assert (result.scanned, result.written, result.unchanged, result.failed) == (1, 1, 0, 0)
        assert _mirrored(tmp_path, "42").read_text(encoding="utf-8").startswith("# Entry 42")
        watermark = (tmp_path / ops.QMD_WATERMARK_NAME).read_text(encoding="utf-8").strip()
        assert datetime.fromisoformat(watermark) == _EARLIER
        assert result.watermark == _EARLIER

    @pytest.mark.asyncio
    async def test_a_write_advances_the_freshness_marker(self, monkeypatch, tmp_path) -> None:
        pool = _FakePool(rows_for={"FROM enhanced_entries": [_row("42", _EARLIER)]})
        _patch_pool(monkeypatch, pool)

        await ops.run_qmd_resync(_config(tmp_path))

        assert (tmp_path / TOUCH_MARKER_NAME).exists()

    @pytest.mark.asyncio
    async def test_first_pass_scans_everything(self, monkeypatch, tmp_path) -> None:
        """With no watermark on disk the scan carries no lower bound."""
        pool = _FakePool(rows_for={"FROM enhanced_entries": [_row("42", _EARLIER)]})
        _patch_pool(monkeypatch, pool)

        await ops.run_qmd_resync(_config(tmp_path))

        sql, params = pool.calls[0]
        assert "WHERE" not in sql
        assert params == [ops.QMD_RESYNC_PAGE_SIZE]

    @pytest.mark.asyncio
    async def test_stored_watermark_bounds_the_next_scan(self, monkeypatch, tmp_path) -> None:
        (tmp_path / ops.QMD_WATERMARK_NAME).write_text(f"{_EARLIER.isoformat()}\n", "utf-8")
        pool = _FakePool(rows_for={"FROM enhanced_entries": []})
        _patch_pool(monkeypatch, pool)

        await ops.run_qmd_resync(_config(tmp_path))

        sql, params = pool.calls[0]
        assert "updated_at >= %s" in " ".join(sql.split())
        assert params == [_EARLIER, ops.QMD_RESYNC_PAGE_SIZE]

    @pytest.mark.asyncio
    async def test_unparsable_watermark_falls_back_to_a_full_scan(
        self, monkeypatch, tmp_path
    ) -> None:
        """Slower, never wrong -- a corrupt watermark must not skip rows."""
        (tmp_path / ops.QMD_WATERMARK_NAME).write_text("not-a-timestamp\n", "utf-8")
        pool = _FakePool(rows_for={"FROM enhanced_entries": []})
        _patch_pool(monkeypatch, pool)

        await ops.run_qmd_resync(_config(tmp_path))

        assert "WHERE" not in pool.calls[0][0]

    @pytest.mark.asyncio
    async def test_empty_scan_leaves_the_watermark_alone(self, monkeypatch, tmp_path) -> None:
        (tmp_path / ops.QMD_WATERMARK_NAME).write_text(f"{_EARLIER.isoformat()}\n", "utf-8")
        _patch_pool(monkeypatch, _FakePool(rows_for={"FROM enhanced_entries": []}))

        result = await ops.run_qmd_resync(_config(tmp_path))

        assert result is not None
        assert result.scanned == 0
        stored = (tmp_path / ops.QMD_WATERMARK_NAME).read_text(encoding="utf-8").strip()
        assert datetime.fromisoformat(stored) == _EARLIER

    @pytest.mark.asyncio
    async def test_watermark_takes_the_highest_updated_at(self, monkeypatch, tmp_path) -> None:
        rows = [_row("a", _EARLIER), _row("b", _LATER)]
        _patch_pool(monkeypatch, _FakePool(rows_for={"FROM enhanced_entries": rows}))

        result = await ops.run_qmd_resync(_config(tmp_path))

        assert result is not None
        assert result.watermark == _LATER

    @pytest.mark.asyncio
    async def test_unchanged_content_is_not_rewritten(self, monkeypatch, tmp_path) -> None:
        """The whole point: bookkeeping churn re-exports nothing."""
        rows = [_row("42", _EARLIER)]
        _patch_pool(monkeypatch, _FakePool(rows_for={"FROM enhanced_entries": rows}))
        await ops.run_qmd_resync(_config(tmp_path))

        marker = tmp_path / TOUCH_MARKER_NAME
        before_marker = marker.read_text(encoding="utf-8")
        before_mtime = _mirrored(tmp_path, "42").stat().st_mtime_ns

        _patch_pool(monkeypatch, _FakePool(rows_for={"FROM enhanced_entries": rows}))
        result = await ops.run_qmd_resync(_config(tmp_path))

        assert result is not None
        assert (result.written, result.unchanged) == (0, 1)
        assert _mirrored(tmp_path, "42").stat().st_mtime_ns == before_mtime
        assert marker.read_text(encoding="utf-8") == before_marker

    @pytest.mark.asyncio
    async def test_changed_content_is_rewritten(self, monkeypatch, tmp_path) -> None:
        _patch_pool(
            monkeypatch,
            _FakePool(rows_for={"FROM enhanced_entries": [_row("42", _EARLIER, "first")]}),
        )
        await ops.run_qmd_resync(_config(tmp_path))

        _patch_pool(
            monkeypatch,
            _FakePool(rows_for={"FROM enhanced_entries": [_row("42", _LATER, "corrected")]}),
        )
        result = await ops.run_qmd_resync(_config(tmp_path))

        assert result is not None
        assert result.written == 1
        assert "corrected" in _mirrored(tmp_path, "42").read_text(encoding="utf-8")

    @pytest.mark.asyncio
    async def test_unmirrorable_entry_is_counted_not_fatal(self, monkeypatch, tmp_path) -> None:
        """One hostile identifier must not cost the rest of the pass."""
        rows = [_row("x" * 400, _EARLIER), _row("42", _LATER)]
        _patch_pool(monkeypatch, _FakePool(rows_for={"FROM enhanced_entries": rows}))

        result = await ops.run_qmd_resync(_config(tmp_path))

        assert result is not None
        assert (result.scanned, result.written, result.failed) == (2, 1, 1)
        assert _mirrored(tmp_path, "42").exists()

    @pytest.mark.asyncio
    async def test_pool_is_closed(self, monkeypatch, tmp_path) -> None:
        pool = _FakePool(rows_for={"FROM enhanced_entries": []})
        _patch_pool(monkeypatch, pool)

        await ops.run_qmd_resync(_config(tmp_path))

        assert pool.closed


class TestResyncPaging:
    """A full page means there is more to fetch."""

    @pytest.mark.asyncio
    async def test_pages_forward_on_the_keyset(self, monkeypatch, tmp_path) -> None:
        pool = _PagingPool([[_row("a", _EARLIER), _row("b", _LATER)], [_row("c", _LATER)]])
        _patch_pool(monkeypatch, pool)

        result = await ops.run_qmd_resync(_config(tmp_path), page_size=2)

        assert result is not None
        assert result.scanned == 3
        assert len(pool.calls) == 2
        second_sql, second_params = pool.calls[1]
        assert "(updated_at, entry_id) > (%s, %s)" in " ".join(second_sql.split())
        assert second_params == [_LATER, "b", 2]
        for name in ("a", "b", "c"):
            assert _mirrored(tmp_path, name).exists()

    @pytest.mark.asyncio
    async def test_short_page_ends_the_scan(self, monkeypatch, tmp_path) -> None:
        pool = _PagingPool([[_row("a", _EARLIER)]])
        _patch_pool(monkeypatch, pool)

        await ops.run_qmd_resync(_config(tmp_path), page_size=2)

        assert len(pool.calls) == 1

    @pytest.mark.asyncio
    async def test_a_page_that_advances_nothing_stops_the_scan(self, monkeypatch, tmp_path) -> None:
        """Rows with no updated_at give no key to page past; stop rather than spin."""
        undated = _row("a", _EARLIER)
        undated["updated_at"] = None
        pool = _PagingPool([[undated], [undated], [undated]])
        _patch_pool(monkeypatch, pool)

        result = await ops.run_qmd_resync(_config(tmp_path), page_size=1)

        assert result is not None
        assert len(pool.calls) == 1
        assert result.watermark is None


# ---------------------------------------------------------------------------
# Rebuild
# ---------------------------------------------------------------------------


class TestRebuild:
    """The wholesale pass that covers purge."""

    @pytest.mark.asyncio
    async def test_rebuild_drops_files_for_entries_that_are_gone(
        self, monkeypatch, tmp_path
    ) -> None:
        _patch_pool(
            monkeypatch,
            _FakePool(rows_for={"FROM enhanced_entries": [_row("42", _EARLIER)]}),
        )
        await ops.run_qmd_resync(_config(tmp_path))
        orphan = _mirrored(tmp_path, "42")
        assert orphan.exists()

        _patch_pool(monkeypatch, _FakePool(rows_for={"FROM enhanced_entries": []}))
        result = await ops.run_qmd_resync(_config(tmp_path), rebuild=True)

        assert result is not None
        assert result.rebuild is True
        assert result.removed == 1
        assert not orphan.exists()

    @pytest.mark.asyncio
    async def test_rebuild_ignores_the_stored_watermark(self, monkeypatch, tmp_path) -> None:
        (tmp_path / ops.QMD_WATERMARK_NAME).write_text(f"{_LATER.isoformat()}\n", "utf-8")
        pool = _FakePool(rows_for={"FROM enhanced_entries": [_row("42", _EARLIER)]})
        _patch_pool(monkeypatch, pool)

        await ops.run_qmd_resync(_config(tmp_path), rebuild=True)

        assert "WHERE" not in pool.calls[0][0]
        assert _mirrored(tmp_path, "42").exists()

    @pytest.mark.asyncio
    async def test_empty_rebuild_clears_the_stale_watermark(self, monkeypatch, tmp_path) -> None:
        """The shape of a rebuild after a purge: nothing left, no bound left."""
        (tmp_path / ops.QMD_WATERMARK_NAME).write_text(f"{_LATER.isoformat()}\n", "utf-8")
        _patch_pool(monkeypatch, _FakePool(rows_for={"FROM enhanced_entries": []}))

        await ops.run_qmd_resync(_config(tmp_path), rebuild=True)

        assert not (tmp_path / ops.QMD_WATERMARK_NAME).exists()

    @pytest.mark.asyncio
    async def test_rebuild_keeps_the_bookkeeping_files(self, monkeypatch, tmp_path) -> None:
        """Dotfiles at the root are markers, not corpus."""
        (tmp_path / TOUCH_MARKER_NAME).write_text("2026-01-01T00:00:00+00:00\n", "utf-8")
        _patch_pool(monkeypatch, _FakePool(rows_for={"FROM enhanced_entries": []}))

        await ops.run_qmd_resync(_config(tmp_path), rebuild=True)

        assert (tmp_path / TOUCH_MARKER_NAME).exists()

    @pytest.mark.asyncio
    async def test_rebuild_on_a_missing_mirror_is_fine(self, monkeypatch, tmp_path) -> None:
        _patch_pool(monkeypatch, _FakePool(rows_for={"FROM enhanced_entries": []}))

        result = await ops.run_qmd_resync(_config(tmp_path / "absent"), rebuild=True)

        assert result is not None
        assert result.removed == 0


# ---------------------------------------------------------------------------
# Best-effort wrapper and the pipeline pre-step
# ---------------------------------------------------------------------------


class TestBestEffortPreStep:
    """Ingest and watch must survive a mirror they cannot write."""

    @pytest.mark.asyncio
    async def test_failure_is_swallowed(self, monkeypatch) -> None:
        async def _boom(*args: Any, **kwargs: Any):
            raise RuntimeError("mirror volume is read-only")

        monkeypatch.setattr(ops, "run_qmd_resync", _boom)

        assert await ops.resync_qmd_mirror_best_effort({}) is None

    @pytest.mark.asyncio
    async def test_result_is_returned_when_it_succeeds(self, monkeypatch, tmp_path) -> None:
        _patch_pool(
            monkeypatch,
            _FakePool(rows_for={"FROM enhanced_entries": [_row("42", _EARLIER)]}),
        )

        result = await ops.resync_qmd_mirror_best_effort(_config(tmp_path))

        assert result is not None
        assert result.written == 1


class TestWatchWiring:
    """Daemon mode resyncs at the head of every poll the scheduler makes."""

    @pytest.mark.asyncio
    async def test_each_daemon_poll_is_preceded_by_a_resync(self, monkeypatch) -> None:
        import osprey.services.ariel_search as ariel_pkg
        import osprey.services.ariel_search.ingestion.scheduler as sched_mod

        order: list[str] = []

        class _Scheduler:
            def __init__(self, config: Any, repository: Any) -> None:
                self.polls = 0

            async def poll_once(self, *args: Any, **kwargs: Any):
                order.append("poll")
                self.polls += 1

            async def run_forever(self) -> None:
                for _ in range(2):
                    await self.poll_once()

            async def stop(self) -> None:
                pass

        class _Service:
            repository = object()

            async def __aenter__(self) -> _Service:
                return self

            async def __aexit__(self, *exc: object) -> bool:
                return False

        async def _create(config: Any):
            return _Service()

        async def _resync(config_dict: dict, progress: Any = None):
            order.append("resync")
            return None

        monkeypatch.setattr(ariel_pkg, "create_ariel_service", _create)
        monkeypatch.setattr(sched_mod, "IngestionScheduler", _Scheduler)
        monkeypatch.setattr(ops, "resync_qmd_mirror_best_effort", _resync)

        config_dict = {**_DB, "ingestion": {"source_url": "https://logbook.example/api"}}
        await ops.run_watch(config_dict, None, None, False, None, False)

        assert order == ["resync", "poll", "resync", "poll"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestQmdResyncCommand:
    """``osprey ariel qmd-resync`` and the pre-step it shares with ingest."""

    def test_help_names_every_bypassing_mutation_path(self) -> None:
        """The docstring is the operator's map of what this command covers."""
        help_text = ariel_group.commands["qmd-resync"].help or ""
        assert "interfaces/ariel/api/routes.py:395" in help_text
        assert "interfaces/ariel/api/routes.py:533" in help_text
        assert "services/ariel_search/service.py:431 and :439" in help_text

    def test_runs_the_pass_and_reports_counts(self, monkeypatch) -> None:
        seen: list[dict] = []

        async def _fake(config_dict, rebuild=False, page_size=0, progress=None):
            seen.append({"rebuild": rebuild})
            return ops.QmdResyncResult(
                scanned=3,
                written=2,
                unchanged=1,
                failed=0,
                removed=0,
                rebuild=rebuild,
                mirror_path="/srv/mirror",
                watermark=_LATER,
            )

        monkeypatch.setattr(ops, "run_qmd_resync", _fake)
        monkeypatch.setattr("osprey.cli.ariel.get_config_value", lambda key, default=None: _DB)

        result = CliRunner().invoke(ariel_group, ["qmd-resync"])

        assert result.exit_code == 0, result.output
        assert seen == [{"rebuild": False}]
        # The summary renders as a key/value section, whose label column is
        # padded to the widest label present -- so match the pairing, not the
        # run of spaces, which changes with the other rows.
        assert re.search(r"Written\s+2", result.output), result.output
        assert re.search(r"Unchanged\s+1", result.output), result.output

    def test_rebuild_flag_reaches_the_pass(self, monkeypatch) -> None:
        seen: list[bool] = []

        async def _fake(config_dict, rebuild=False, page_size=0, progress=None):
            seen.append(rebuild)
            return ops.QmdResyncResult(
                scanned=0,
                written=0,
                unchanged=0,
                failed=0,
                removed=7,
                rebuild=rebuild,
                mirror_path="/srv/mirror",
                watermark=None,
            )

        monkeypatch.setattr(ops, "run_qmd_resync", _fake)
        monkeypatch.setattr("osprey.cli.ariel.get_config_value", lambda key, default=None: _DB)

        result = CliRunner().invoke(ariel_group, ["qmd-resync", "--rebuild"])

        assert result.exit_code == 0, result.output
        assert seen == [True]
        assert re.search(r"Removed before rebuild\s+7", result.output), result.output

    def test_disabled_module_reports_rather_than_failing(self, monkeypatch) -> None:
        async def _fake(config_dict, rebuild=False, page_size=0, progress=None):
            return None

        monkeypatch.setattr(ops, "run_qmd_resync", _fake)
        monkeypatch.setattr("osprey.cli.ariel.get_config_value", lambda key, default=None: _DB)

        result = CliRunner().invoke(ariel_group, ["qmd-resync"])

        assert result.exit_code == 0
        assert "not enabled" in result.output

    def test_misconfiguration_exits_with_the_config_key(self, monkeypatch) -> None:
        async def _fake(config_dict, rebuild=False, page_size=0, progress=None):
            raise ValueError("enhancement_modules.qmd_export.mirror_path is required")

        monkeypatch.setattr(ops, "run_qmd_resync", _fake)
        monkeypatch.setattr("osprey.cli.ariel.get_config_value", lambda key, default=None: _DB)

        result = CliRunner().invoke(ariel_group, ["qmd-resync"])

        assert result.exit_code == 1
        assert "mirror_path" in result.stderr


class TestPipelinePreStep:
    """The pass ingest and watch run before their own work."""

    @pytest.fixture
    def _observed(self, monkeypatch) -> list[str]:
        calls: list[str] = []

        async def _fake_resync(config_dict, progress=None):
            calls.append("resync")
            return None

        monkeypatch.setattr(ops, "resync_qmd_mirror_best_effort", _fake_resync)
        monkeypatch.setattr("osprey.cli.ariel.get_config_value", lambda key, default=None: _DB)
        return calls

    def test_ingest_resyncs_first(self, monkeypatch, _observed) -> None:
        async def _fake_ingest(*args: Any, **kwargs: Any):
            _observed.append("ingest")
            return ops.IngestResult(count=0, enhanced_count=0, failed_count=0, dry_run=True)

        monkeypatch.setattr(ops, "run_ingest", _fake_ingest)

        result = CliRunner().invoke(ariel_group, ["ingest", "-s", "entries.json"])

        assert result.exit_code == 0, result.output
        assert _observed == ["resync", "ingest"]

    def test_watch_resyncs_first(self, monkeypatch, _observed) -> None:
        async def _fake_watch(*args: Any, **kwargs: Any):
            _observed.append("watch")
            return None

        monkeypatch.setattr(ops, "run_watch", _fake_watch)

        result = CliRunner().invoke(ariel_group, ["watch", "--once"])

        assert result.exit_code == 0, result.output
        assert _observed == ["resync", "watch"]
