"""Unit tests for :mod:`osprey.services.ariel_search.cli_operations`.

These are the service-layer functions behind the ``osprey ariel`` CLI. They
compose config parsing, the ARIEL service, adapters and enhancers, then return
structured result dataclasses. This module targets the *pure-logic* and
*error-translation* contracts that do not need a live Postgres:

* ``_entry_summary`` — the JSON-safe display projection (fully pure).
* ``get_status`` / ``run_search`` — status assembly, URI masking, and the
  human-facing error-message branches (connection failure, missing tables).
* ``run_enhance`` — the "no enhancers selected" short-circuit.
* ``run_ingest`` (dry-run) — adapter-driven counting with no DB writes.
* ``run_reembed`` (dry-run) — table-name derivation with no embedding calls.
* ``run_watch`` — the "no source configured" rejection.
* ``seed_logbook_entries`` / ``list_models`` — repository orchestration with
  the service boundary mocked.

The service boundary (``create_ariel_service``), the adapter factory
(``get_adapter``) and the enhancer factory (``create_enhancers_from_config``)
are monkeypatched at their source modules, since ``cli_operations`` imports
them lazily inside each function. No real database, network, or embedding
provider is ever touched.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from osprey.services.ariel_search import cli_operations as ops

# A minimal config dict accepted by ARIELConfig.from_dict.
_DB = {"database": {"uri": "postgresql://localhost/test"}}


class _StubService:
    """Async-context-manager stub standing in for the ARIEL service."""

    def __init__(self, *, repository=None, health=(True, "OK")):
        self.repository = repository or MagicMock()
        self._health = health

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def health_check(self):
        return self._health


def _patch_service(monkeypatch, service):
    """Route ``create_ariel_service`` to return *service*."""
    import osprey.services.ariel_search as ariel_pkg

    async def _fake_create(config):
        return service

    monkeypatch.setattr(ariel_pkg, "create_ariel_service", _fake_create)


def _patch_service_raises(monkeypatch, exc):
    """Route ``create_ariel_service`` to raise *exc*."""
    import osprey.services.ariel_search as ariel_pkg

    async def _fake_create(config):
        raise exc

    monkeypatch.setattr(ariel_pkg, "create_ariel_service", _fake_create)


def _embedding_table(name="text_embeddings_nomic", count=5, dim=768, active=True):
    return SimpleNamespace(table_name=name, entry_count=count, dimension=dim, is_active=active)


# ---------------------------------------------------------------------------
# _entry_summary — pure projection
# ---------------------------------------------------------------------------


class TestEntrySummary:
    def test_datetime_timestamp_is_isoformatted(self):
        ts = datetime(2026, 6, 9, 8, 1, 0, tzinfo=UTC)
        out = ops._entry_summary({"timestamp": ts, "raw_text": "hello"})
        assert out["timestamp"] == ts.isoformat()

    def test_title_is_first_line_truncated_to_100_chars(self):
        long_first = "A" * 250
        entry = {"raw_text": f"{long_first}\nsecond line"}
        out = ops._entry_summary(entry)
        assert out["title"] == "A" * 100
        assert "second line" not in out["title"]

    def test_leading_whitespace_stripped_before_title(self):
        out = ops._entry_summary({"raw_text": "   \n  Real title\nrest"})
        assert out["title"] == "Real title"

    def test_missing_fields_default_gracefully(self):
        out = ops._entry_summary({})
        assert out == {
            "entry_id": "",
            "timestamp": "",
            "author": "",
            "title": "",
            "score": None,
        }

    def test_none_raw_text_yields_empty_title(self):
        out = ops._entry_summary({"raw_text": None, "entry_id": "E1"})
        assert out["title"] == ""
        assert out["entry_id"] == "E1"

    def test_score_and_string_timestamp_preserved(self):
        out = ops._entry_summary(
            {"_score": 0.42, "timestamp": "2026-01-01T00:00:00", "author": "op"}
        )
        assert out["score"] == 0.42
        assert out["timestamp"] == "2026-01-01T00:00:00"
        assert out["author"] == "op"


# ---------------------------------------------------------------------------
# get_status
# ---------------------------------------------------------------------------


class TestGetStatus:
    async def test_empty_config_reports_not_configured(self):
        out = await ops.get_status({})
        assert out == {"status": "error", "message": "ARIEL not configured"}

    async def test_healthy_status_assembles_full_report(self, monkeypatch):
        repo = MagicMock()
        repo.get_enhancement_stats = AsyncMock(return_value={"total_entries": 42})
        repo.get_embedding_tables = AsyncMock(return_value=[_embedding_table()])
        service = _StubService(repository=repo, health=(True, "connected"))
        _patch_service(monkeypatch, service)

        out = await ops.get_status(dict(_DB))

        assert out["status"] == "healthy"
        assert out["message"] == "connected"
        assert out["entries"] == 42
        assert out["database"]["connected"] is True
        assert out["embedding_tables"][0]["table"] == "text_embeddings_nomic"
        assert out["embedding_tables"][0]["entries"] == 5
        assert "enhancement_modules" in out
        assert "search_modules" in out

    async def test_unhealthy_flag_when_health_check_fails(self, monkeypatch):
        repo = MagicMock()
        repo.get_enhancement_stats = AsyncMock(return_value={})
        repo.get_embedding_tables = AsyncMock(return_value=[])
        service = _StubService(repository=repo, health=(False, "no schema"))
        _patch_service(monkeypatch, service)

        out = await ops.get_status(dict(_DB))

        assert out["status"] == "unhealthy"
        assert out["database"]["connected"] is False
        # No stats -> entries defaults to 0.
        assert out["entries"] == 0

    async def test_uri_with_credentials_is_masked(self, monkeypatch):
        repo = MagicMock()
        repo.get_enhancement_stats = AsyncMock(return_value={"total_entries": 0})
        repo.get_embedding_tables = AsyncMock(return_value=[])
        _patch_service(monkeypatch, _StubService(repository=repo))

        out = await ops.get_status({"database": {"uri": "postgresql://u:p@dbhost:5432/ariel"}})

        assert out["database"]["uri"] == "dbhost:5432/ariel"

    async def test_uri_without_credentials_passes_through(self, monkeypatch):
        repo = MagicMock()
        repo.get_enhancement_stats = AsyncMock(return_value={"total_entries": 0})
        repo.get_embedding_tables = AsyncMock(return_value=[])
        _patch_service(monkeypatch, _StubService(repository=repo))

        out = await ops.get_status({"database": {"uri": "postgresql://localhost/ariel"}})

        assert out["database"]["uri"] == "postgresql://localhost/ariel"

    async def test_connection_error_maps_to_friendly_message(self, monkeypatch):
        _patch_service_raises(monkeypatch, RuntimeError("could not connect to server"))
        out = await ops.get_status(dict(_DB))
        assert out["status"] == "error"
        assert "osprey deploy up" in out["message"]

    async def test_generic_error_returns_raw_message(self, monkeypatch):
        _patch_service_raises(monkeypatch, RuntimeError("boom-unexpected"))
        out = await ops.get_status(dict(_DB))
        assert out == {"status": "error", "message": "boom-unexpected"}


# ---------------------------------------------------------------------------
# run_search — error-branch translation and input validation
# ---------------------------------------------------------------------------


class TestRunSearch:
    async def test_empty_config_reports_not_configured(self):
        out = await ops.run_search({}, "q", "keyword", 5)
        assert out == {"error": "ARIEL not configured"}

    async def test_invalid_mode_raises_keyerror(self):
        # SearchMode[mode.upper()] is evaluated before the try-block, so an
        # unknown mode surfaces as an uncaught KeyError.
        with pytest.raises(KeyError):
            await ops.run_search(dict(_DB), "q", "does-not-exist", 5)

    async def test_connection_error_maps_to_friendly_message(self, monkeypatch):
        _patch_service_raises(monkeypatch, RuntimeError("connection refused"))
        out = await ops.run_search(dict(_DB), "q", "keyword", 5)
        assert "osprey deploy up" in out["error"]

    async def test_missing_relation_suggests_migrate(self, monkeypatch):
        _patch_service_raises(
            monkeypatch,
            RuntimeError('relation "enhanced_entries" does not exist'),
        )
        out = await ops.run_search(dict(_DB), "q", "keyword", 5)
        assert "osprey ariel migrate" in out["error"]

    async def test_generic_error_returns_raw_message(self, monkeypatch):
        _patch_service_raises(monkeypatch, RuntimeError("weird failure"))
        out = await ops.run_search(dict(_DB), "q", "keyword", 5)
        assert out == {"error": "weird failure"}

    async def test_success_projects_answer_sources_and_entries(self, monkeypatch):
        result = SimpleNamespace(
            answer="the answer",
            sources=["S1", "S2"],
            search_modes_used=[_mode_keyword()],
            reasoning="because",
            entries=[{"entry_id": "E1", "raw_text": "First line\nmore", "_score": 0.9}],
        )
        service = MagicMock()
        service.__aenter__ = AsyncMock(return_value=service)
        service.__aexit__ = AsyncMock(return_value=False)
        service.search = AsyncMock(return_value=result)
        _patch_service(monkeypatch, service)

        out = await ops.run_search(dict(_DB), "coupler", "keyword", 3)

        assert out["answer"] == "the answer"
        assert out["sources"] == ["S1", "S2"]
        assert out["search_modes"] == ["keyword"]
        assert out["entries"][0]["title"] == "First line"
        assert out["entries"][0]["score"] == 0.9


def _mode_keyword():
    from osprey.services.ariel_search import SearchMode

    return SearchMode.KEYWORD


# ---------------------------------------------------------------------------
# run_enhance — no-enhancers short-circuit
# ---------------------------------------------------------------------------


class TestRunEnhance:
    async def test_no_enhancers_returns_empty_result(self, monkeypatch):
        import osprey.services.ariel_search.enhancement as enh

        monkeypatch.setattr(enh, "create_enhancers_from_config", lambda config: [])
        # create_ariel_service must never be reached; make it explode if it is.
        _patch_service_raises(monkeypatch, AssertionError("service should not be created"))

        out = await ops.run_enhance(dict(_DB), module=None, force=False, limit=10)

        assert out.entries_processed == 0
        assert out.module_names == []

    async def test_module_filter_narrowing_to_none_short_circuits(self, monkeypatch):
        import osprey.services.ariel_search.enhancement as enh

        enhancer = SimpleNamespace(name="text_embedding")
        monkeypatch.setattr(enh, "create_enhancers_from_config", lambda config: [enhancer])
        _patch_service_raises(monkeypatch, AssertionError("service should not be created"))

        # Selecting a module that no configured enhancer provides -> empty.
        out = await ops.run_enhance(dict(_DB), module="nonexistent", force=False, limit=10)

        assert out.entries_processed == 0
        assert out.module_names == []


# ---------------------------------------------------------------------------
# run_ingest — dry-run counts without DB writes
# ---------------------------------------------------------------------------


class TestRunIngestDryRun:
    async def test_dry_run_counts_entries_and_skips_writes(self, monkeypatch):
        import osprey.services.ariel_search.enhancement as enh
        import osprey.services.ariel_search.ingestion as ing

        class _Adapter:
            source_system_name = "TestSource"

            async def fetch_entries(self, since=None, limit=None):
                for i in range(3):
                    yield {"entry_id": f"E{i}"}

        monkeypatch.setattr(ing, "get_adapter", lambda config: _Adapter())
        monkeypatch.setattr(enh, "create_enhancers_from_config", lambda config: [])
        # Dry-run must not create the service.
        _patch_service_raises(monkeypatch, AssertionError("service should not be created"))

        out = await ops.run_ingest(
            dict(_DB),
            source="file:///x.json",
            adapter="generic_json",
            since=None,
            limit=None,
            dry_run=True,
        )

        assert out.dry_run is True
        assert out.count == 3
        assert out.enhanced_count == 0
        assert out.failed_count == 0
        assert out.enhancer_names == []

    async def test_dry_run_reports_enhancer_names_via_progress(self, monkeypatch):
        import osprey.services.ariel_search.enhancement as enh
        import osprey.services.ariel_search.ingestion as ing

        class _Adapter:
            source_system_name = "TestSource"

            async def fetch_entries(self, since=None, limit=None):
                if False:
                    yield  # empty async generator

        monkeypatch.setattr(ing, "get_adapter", lambda config: _Adapter())
        monkeypatch.setattr(
            enh,
            "create_enhancers_from_config",
            lambda config: [SimpleNamespace(name="text_embedding")],
        )
        _patch_service_raises(monkeypatch, AssertionError("service should not be created"))

        messages: list[str] = []
        out = await ops.run_ingest(
            dict(_DB),
            source="file:///x.json",
            adapter="generic_json",
            since=None,
            limit=None,
            dry_run=True,
            progress=messages.append,
        )

        assert out.enhancer_names == ["text_embedding"]
        assert any("TestSource" in m for m in messages)


# ---------------------------------------------------------------------------
# run_reembed — dry-run derives table name, no embedding calls
# ---------------------------------------------------------------------------


class TestRunReembedDryRun:
    async def test_dry_run_returns_zeroed_result(self, monkeypatch):
        # Service creation would mean real work; forbid it.
        _patch_service_raises(monkeypatch, AssertionError("service should not be created"))

        messages: list[str] = []
        out = await ops.run_reembed(
            dict(_DB),
            model="nomic-embed-text",
            dimension=768,
            batch_size=16,
            dry_run=True,
            force=False,
            progress=messages.append,
        )

        assert out.dry_run is True
        assert (out.processed, out.skipped, out.errors) == (0, 0, 0)
        # The derived table name is surfaced in the dry-run preview.
        assert any("text_embeddings_nomic_embed_text" in m for m in messages)


# ---------------------------------------------------------------------------
# run_watch — input validation
# ---------------------------------------------------------------------------


class TestRunWatch:
    async def test_missing_source_raises_valueerror(self, monkeypatch):
        # No source arg and no ingestion.source_url in config -> rejected before
        # any service is created.
        _patch_service_raises(monkeypatch, AssertionError("service should not be created"))

        with pytest.raises(ValueError, match="No ingestion source configured"):
            await ops.run_watch(
                dict(_DB),
                source=None,
                adapter=None,
                once=True,
                interval=None,
                dry_run=False,
            )


# ---------------------------------------------------------------------------
# seed_logbook_entries — repository orchestration
# ---------------------------------------------------------------------------


class TestSeedLogbookEntries:
    async def test_seeds_all_entries_and_completes_run(self, monkeypatch):
        repo = MagicMock()
        repo.start_ingestion_run = AsyncMock(return_value="run-1")
        repo.upsert_entry = AsyncMock()
        repo.complete_ingestion_run = AsyncMock()
        repo.fail_ingestion_run = AsyncMock()
        _patch_service(monkeypatch, _StubService(repository=repo))

        entries = [{"entry_id": "E1"}, {"entry_id": "E2"}]
        count = await ops.seed_logbook_entries(dict(_DB), entries)

        assert count == 2
        assert repo.upsert_entry.await_count == 2
        repo.complete_ingestion_run.assert_awaited_once()
        repo.complete_ingestion_run.assert_awaited_once_with(
            "run-1", entries_added=2, entries_updated=0, entries_failed=0
        )
        repo.fail_ingestion_run.assert_not_awaited()

    async def test_upsert_failure_marks_run_failed_and_reraises(self, monkeypatch):
        repo = MagicMock()
        repo.start_ingestion_run = AsyncMock(return_value="run-2")
        repo.upsert_entry = AsyncMock(side_effect=RuntimeError("db down"))
        repo.complete_ingestion_run = AsyncMock()
        repo.fail_ingestion_run = AsyncMock()
        _patch_service(monkeypatch, _StubService(repository=repo))

        with pytest.raises(RuntimeError, match="db down"):
            await ops.seed_logbook_entries(dict(_DB), [{"entry_id": "E1"}])

        repo.fail_ingestion_run.assert_awaited_once()
        repo.complete_ingestion_run.assert_not_awaited()


# ---------------------------------------------------------------------------
# list_models — repository projection
# ---------------------------------------------------------------------------


class TestListModels:
    async def test_projects_embedding_tables(self, monkeypatch):
        repo = MagicMock()
        repo.get_embedding_tables = AsyncMock(
            return_value=[
                _embedding_table(name="text_embeddings_a", count=3, dim=768, active=True),
                _embedding_table(name="text_embeddings_b", count=0, dim=384, active=False),
            ]
        )
        _patch_service(monkeypatch, _StubService(repository=repo))

        out = await ops.list_models(dict(_DB))

        assert out == [
            {
                "table_name": "text_embeddings_a",
                "entry_count": 3,
                "dimension": 768,
                "is_active": True,
            },
            {
                "table_name": "text_embeddings_b",
                "entry_count": 0,
                "dimension": 384,
                "is_active": False,
            },
        ]

    async def test_empty_when_no_tables(self, monkeypatch):
        repo = MagicMock()
        repo.get_embedding_tables = AsyncMock(return_value=[])
        _patch_service(monkeypatch, _StubService(repository=repo))

        out = await ops.list_models(dict(_DB))
        assert out == []
