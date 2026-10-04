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
from osprey.services.ariel_search.database.repository import SchemaFacts
from osprey.services.ariel_search.exceptions import DatabaseQueryError

# A minimal config dict accepted by ARIELConfig.from_dict.
_DB = {"database": {"uri": "postgresql://localhost/test"}}

# What ``get_status`` reports for a config with no ``vocabulary`` block at all.
_DISABLED_VOCABULARY = {"status": "disabled", "concepts": 0, "errors": []}


def _write_vocabulary(tmp_path, body: str):
    """Write *body* as a vocabulary file and return its path."""
    path = tmp_path / "vocabulary.yml"
    path.write_text(body, encoding="utf-8")
    return path


# A file with exactly three errors: an empty canonical, an unknown kind, and an
# empty forms list.
_THREE_ERRORS = """
concepts:
  - canonical: ""
    kind: acronym
    forms: ["bpm"]
  - canonical: beam position monitor
    kind: sideways
    forms: ["bpm"]
  - canonical: radio frequency
    kind: acronym
    forms: []
"""

# One form bound to two concepts: legal, one warning, no errors.
_AMBIGUOUS = """
concepts:
  - canonical: troubleshoot
    kind: shorthand
    forms: ["t/s", "ts"]
  - canonical: timing system
    kind: acronym
    forms: ["ts"]
"""

# The only stopword-valued string is a *shorthand* form, so whether it warns
# depends on canonical_to_shorthand.
_STOPWORD_SHORTHAND = """
concepts:
  - canonical: orbit correction
    kind: shorthand
    forms: ["a"]
"""


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

    async def _fake_create(_config):
        return service

    monkeypatch.setattr(ariel_pkg, "create_ariel_service", _fake_create)


def _patch_service_raises(monkeypatch, exc):
    """Route ``create_ariel_service`` to raise *exc*."""
    import osprey.services.ariel_search as ariel_pkg

    async def _fake_create(_config):
        raise exc

    monkeypatch.setattr(ariel_pkg, "create_ariel_service", _fake_create)


def _embedding_table(name="text_embeddings_nomic", count=5, dim=768, active=True):
    return SimpleNamespace(table_name=name, entry_count=count, dimension=dim, is_active=active)


def _status_repo(
    stats=None,
    tables=(),
    last_ingestion=None,
    *,
    copy_state=True,
    attachment_bytes=0,
    copy_counts=(0, {}),
):
    """Repository double answering every query ``get_status`` makes.

    Args:
        stats: ``get_enhancement_stats`` payload; an empty store by default.
        tables: ``get_embedding_tables`` payload (no image tables).
        last_ingestion: ``get_last_ingestion`` result.
        copy_state: ``schema_facts().has_copy_state``.
        attachment_bytes: ``get_attachment_bytes`` result, or an exception to raise.
        copy_counts: ``get_attachment_copy_counts`` result ``(pending, {code: n})``.
    """
    repo = MagicMock()
    repo.get_enhancement_stats = AsyncMock(
        return_value={"total_entries": 0} if stats is None else stats
    )
    repo.get_embedding_tables = AsyncMock(return_value=list(tables))
    repo.get_image_embedding_tables = AsyncMock(return_value=[])
    repo.get_last_ingestion = AsyncMock(return_value=last_ingestion)
    repo.schema_facts = AsyncMock(
        return_value=SchemaFacts(has_v2_fts=copy_state, has_copy_state=copy_state)
    )
    if isinstance(attachment_bytes, BaseException):
        repo.get_attachment_bytes = AsyncMock(side_effect=attachment_bytes)
    else:
        repo.get_attachment_bytes = AsyncMock(return_value=attachment_bytes)
    repo.get_attachment_copy_counts = AsyncMock(return_value=copy_counts)
    return repo


@pytest.fixture(autouse=True)
def render_probe(monkeypatch):
    """Answer the render probe ``ok`` without spawning a worker; tests may flip it."""
    probe = AsyncMock(return_value=True)
    monkeypatch.setattr("osprey.imaging.render.probe_render_worker", probe)
    return probe


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
        assert out == {
            "status": "error",
            "message": "ARIEL not configured",
            "vocabulary": _DISABLED_VOCABULARY,
        }

    async def test_healthy_status_assembles_full_report(self, monkeypatch):
        repo = _status_repo(stats={"total_entries": 42}, tables=[_embedding_table()])
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
        repo = _status_repo(stats={})
        service = _StubService(repository=repo, health=(False, "no schema"))
        _patch_service(monkeypatch, service)

        out = await ops.get_status(dict(_DB))

        assert out["status"] == "unhealthy"
        assert out["database"]["connected"] is False
        # No stats -> entries defaults to 0.
        assert out["entries"] == 0

    async def test_uri_with_credentials_is_masked(self, monkeypatch):
        repo = _status_repo()
        _patch_service(monkeypatch, _StubService(repository=repo))

        out = await ops.get_status({"database": {"uri": "postgresql://u:p@dbhost:5432/ariel"}})

        assert out["database"]["uri"] == "dbhost:5432/ariel"

    async def test_uri_without_credentials_passes_through(self, monkeypatch):
        repo = _status_repo()
        _patch_service(monkeypatch, _StubService(repository=repo))

        out = await ops.get_status({"database": {"uri": "postgresql://localhost/ariel"}})

        assert out["database"]["uri"] == "postgresql://localhost/ariel"

    async def test_connection_error_maps_to_friendly_message(self, monkeypatch):
        _patch_service_raises(monkeypatch, RuntimeError("could not connect to server"))
        out = await ops.get_status(dict(_DB))
        assert out["status"] == "error"
        assert "osprey up" in out["message"]

    async def test_generic_error_returns_raw_message(self, monkeypatch):
        _patch_service_raises(monkeypatch, RuntimeError("boom-unexpected"))
        out = await ops.get_status(dict(_DB))
        assert out == {
            "status": "error",
            "message": "boom-unexpected",
            "vocabulary": _DISABLED_VOCABULARY,
        }

    async def test_last_ingestion_is_the_repository_timestamp_in_iso_form(self, monkeypatch):
        repo = _status_repo(
            stats={"total_entries": 1},
            last_ingestion=datetime(2026, 3, 4, 5, 6, 7, tzinfo=UTC),
        )
        _patch_service(monkeypatch, _StubService(repository=repo))

        out = await ops.get_status(dict(_DB))

        assert out["last_ingestion"] == "2026-03-04T05:06:07+00:00"

    async def test_last_ingestion_is_null_when_nothing_has_been_ingested(self, monkeypatch):
        repo = _status_repo()
        _patch_service(monkeypatch, _StubService(repository=repo))

        out = await ops.get_status(dict(_DB))

        assert out["status"] == "healthy"
        assert out["last_ingestion"] is None

    async def test_last_ingestion_query_failure_lands_in_the_error_branch(self, monkeypatch):
        repo = _status_repo()
        repo.get_last_ingestion = AsyncMock(side_effect=DatabaseQueryError("boom-ingestion"))
        _patch_service(monkeypatch, _StubService(repository=repo))

        out = await ops.get_status(dict(_DB))

        assert out["status"] == "error"
        assert "boom-ingestion" in out["message"]

    async def test_registered_module_with_no_rows_owes_every_entry(self, monkeypatch):
        """One table over every registered module, whether the store knows it or not.

        A module that has never run has written no status key, so every entry is
        still pending for it: the same count a module that processed one entry
        reports as the rest, never a zero that reads as "nothing to do".
        """
        repo = _status_repo(stats={"total_entries": 3})
        _patch_service(monkeypatch, _StubService(repository=repo))

        out = await ops.get_status(dict(_DB))

        assert out["enhancement_modules"]["text_embedding"] == {
            "enabled": False,
            "complete": 0,
            "failed": 0,
            "pending": 3,
            "gave_up": 0,
        }

    async def test_registered_module_on_an_empty_store_reports_zeros(self, monkeypatch):
        """No entries, nothing owed: a never-run module reads all zeros."""
        repo = _status_repo(stats={"total_entries": 0})
        _patch_service(monkeypatch, _StubService(repository=repo))

        out = await ops.get_status(dict(_DB))

        assert out["enhancement_modules"]["text_embedding"] == {
            "enabled": False,
            "complete": 0,
            "failed": 0,
            "pending": 0,
            "gave_up": 0,
        }

    async def test_store_key_with_no_registered_module_is_reported_as_orphaned(self, monkeypatch):
        """Rows nothing writes any more are named, not silently folded in."""
        repo = _status_repo(
            stats={
                "total_entries": 3,
                "text_embedding": {"complete": 3, "failed": 0, "pending": 0},
                "retired_tagger": {"complete": 1, "failed": 2, "pending": 0},
            }
        )
        _patch_service(monkeypatch, _StubService(repository=repo))

        out = await ops.get_status(dict(_DB))

        assert out["orphaned_enhancement_modules"] == {
            "retired_tagger": {"complete": 1, "failed": 2, "pending": 0, "gave_up": 0}
        }
        assert "retired_tagger" not in out["enhancement_modules"]
        assert out["enhancement_modules"]["text_embedding"]["complete"] == 3

    async def test_stats_are_read_without_markers_when_no_caption_model(self, monkeypatch):
        """No caption model configured: the stats call is B1's, with no argument."""
        repo = _status_repo(stats={"total_entries": 3})
        _patch_service(monkeypatch, _StubService(repository=repo))

        await ops.get_status(dict(_DB))

        repo.get_enhancement_stats.assert_awaited_once_with()

    async def test_stats_are_read_under_the_caption_marker(self, monkeypatch):
        """A configured caption model is passed as the image_caption marker."""
        repo = _status_repo(
            stats={
                "total_entries": 3,
                "image_caption": {"complete": 1, "failed": 1, "pending": 1, "gave_up": 1},
            }
        )
        _patch_service(monkeypatch, _StubService(repository=repo))
        config = {
            **_DB,
            "enhancement_modules": {"image_caption": {"model": {"model_id": "vis-a"}}},
        }

        out = await ops.get_status(config)

        repo.get_enhancement_stats.assert_awaited_once_with(markers={"image_caption": "vis-a"})
        counts = (
            out["enhancement_modules"].get("image_caption")
            or out["orphaned_enhancement_modules"]["image_caption"]
        )
        assert {k: counts[k] for k in ("complete", "failed", "pending", "gave_up")} == {
            "complete": 1,
            "failed": 1,
            "pending": 1,
            "gave_up": 1,
        }

    async def test_total_entries_is_not_a_module_in_either_table(self, monkeypatch):
        """It is the store's own count, and it already has its own key."""
        repo = _status_repo(stats={"total_entries": 3})
        _patch_service(monkeypatch, _StubService(repository=repo))

        out = await ops.get_status(dict(_DB))

        assert "total_entries" not in out["enhancement_modules"]
        assert "total_entries" not in out["orphaned_enhancement_modules"]
        assert out["entries"] == 3


class TestGetStatusAttachments:
    """The ``attachments`` object of ``osprey ariel status``."""

    async def test_attachments_block_carries_capability_counts_and_render(self, monkeypatch):
        repo = _status_repo(attachment_bytes=4096, copy_counts=(3, {"too_large": 2}))
        _patch_service(monkeypatch, _StubService(repository=repo))

        out = await ops.get_status(dict(_DB))

        attachments = out["attachments"]
        assert list(attachments) == [
            "copy_on_ingest",
            "formats",
            "view",
            "captions",
            "picture_search",
            "picture_search_unavailable",
            "bytes",
            "pending",
            "skipped",
            "render",
        ]
        assert attachments["copy_on_ingest"] == "images"
        assert "png" in attachments["formats"]["viewable"]
        assert "svg" in attachments["formats"]["reserved"]
        assert attachments["view"] is True
        assert attachments["captions"] is False
        assert attachments["picture_search"] is False
        assert attachments["bytes"] == 4096
        assert attachments["pending"] == 3
        assert attachments["skipped"] == {"too_large": 2}
        assert attachments["render"] == "ok"

    async def test_attachments_matches_the_capability_block(self, monkeypatch):
        from osprey.services.ariel_search.capabilities import attachments_capability

        _patch_service(monkeypatch, _StubService(repository=_status_repo()))
        config = {**_DB, "attachments": {"copy_on_ingest": "all", "view": {"enabled": False}}}

        out = await ops.get_status(config)

        expected = attachments_capability(ops._ariel_config(config))
        assert {k: out["attachments"][k] for k in expected} == expected
        assert out["attachments"]["view"] is False
        assert out["attachments"]["copy_on_ingest"] == "all"

    async def test_attachments_render_unavailable_when_probe_fails(self, monkeypatch, render_probe):
        render_probe.return_value = False
        _patch_service(monkeypatch, _StubService(repository=_status_repo()))

        out = await ops.get_status(dict(_DB))

        assert out["attachments"]["render"] == "unavailable"
        render_probe.assert_awaited_once()

    async def test_attachments_schema_behind_nulls_pending_and_skipped(self, monkeypatch):
        repo = _status_repo(copy_state=False, attachment_bytes=1024)
        _patch_service(monkeypatch, _StubService(repository=repo))

        out = await ops.get_status(dict(_DB))

        assert out["status"] == "healthy"
        assert out["attachments"]["pending"] is None
        assert out["attachments"]["skipped"] is None
        assert out["attachments"]["bytes"] == 1024
        repo.get_attachment_copy_counts.assert_not_awaited()

    async def test_attachments_bytes_null_when_store_cannot_size_it(self, monkeypatch):
        repo = _status_repo(attachment_bytes=DatabaseQueryError("no attachment_files"))
        _patch_service(monkeypatch, _StubService(repository=repo))

        out = await ops.get_status(dict(_DB))

        assert out["status"] == "healthy"
        assert out["attachments"]["bytes"] is None

    async def test_attachments_absent_on_error_paths(self, monkeypatch):
        _patch_service_raises(monkeypatch, RuntimeError("could not connect to server"))

        out = await ops.get_status(dict(_DB))

        assert out["status"] == "error"
        assert "attachments" not in out


class TestStatusTextAttachments:
    """``osprey ariel status`` prints the attachments object for an operator."""

    def _run(self, monkeypatch, repo):
        from click.testing import CliRunner

        from osprey.cli.ariel import ariel_group

        _patch_service(monkeypatch, _StubService(repository=repo))
        monkeypatch.setattr("osprey.cli.ariel.get_config_value", lambda *a, **kw: dict(_DB))
        result = CliRunner().invoke(ariel_group, ["status"])
        assert result.exit_code == 0, result.output
        return result.output

    def test_attachments_text_lists_each_non_zero_skip_code_with_its_reason(self, monkeypatch):
        from osprey.imaging.formats import skip_reason_text

        repo = _status_repo(copy_counts=(2, {"too_large": 3, "fetch_failed": 0}))

        text = self._run(monkeypatch, repo)

        assert "render: ok" in text
        assert "pending: 2" in text
        assert skip_reason_text("too_large") in text
        assert "too_large: 3" in text
        assert "fetch_failed" not in text
        assert "schema behind code" not in text

    def test_attachments_text_render_unavailable(self, monkeypatch, render_probe):
        render_probe.return_value = False

        assert "render: unavailable" in self._run(monkeypatch, _status_repo())

    def test_attachments_text_schema_behind_code(self, monkeypatch):
        text = self._run(monkeypatch, _status_repo(copy_state=False))

        assert "schema behind code: run osprey ariel migrate" in text
        assert "pending:" not in text


class TestGetStatusVocabulary:
    """The ``vocabulary`` block rides on every path out of ``get_status``."""

    async def test_no_config_reports_disabled(self):
        out = await ops.get_status({})
        assert out["vocabulary"] == {"status": "disabled", "concepts": 0, "errors": []}

    async def test_legacy_pipelines_section_reports_invalid_with_parse_error(self):
        out = await ops.get_status({"pipelines": {"rag": {}}})

        assert out["vocabulary"]["status"] == "invalid"
        assert out["vocabulary"]["concepts"] == 0
        assert len(out["vocabulary"]["errors"]) == 1
        assert "ariel.pipelines section is no longer supported" in out["vocabulary"]["errors"][0]

    async def test_missing_file_reports_invalid_naming_the_key(self, monkeypatch, tmp_path):
        repo = _status_repo()
        _patch_service(monkeypatch, _StubService(repository=repo))

        config = dict(_DB)
        config["vocabulary"] = {"enabled": True, "path": str(tmp_path / "absent.yml")}
        out = await ops.get_status(config)

        assert out["status"] == "healthy"
        assert out["vocabulary"]["status"] == "invalid"
        assert "ariel.vocabulary.path" in out["vocabulary"]["errors"][0]

    async def test_valid_file_reports_ok_with_concept_count(self, monkeypatch, tmp_path):
        repo = _status_repo()
        _patch_service(monkeypatch, _StubService(repository=repo))

        path = _write_vocabulary(tmp_path, _AMBIGUOUS)
        config = dict(_DB)
        config["vocabulary"] = {"enabled": True, "path": str(path)}
        out = await ops.get_status(config)

        assert out["vocabulary"] == {"status": "ok", "concepts": 2, "errors": []}

    async def test_enabled_without_path_reports_invalid(self, monkeypatch):
        repo = _status_repo()
        _patch_service(monkeypatch, _StubService(repository=repo))

        config = dict(_DB)
        config["vocabulary"] = {"enabled": True}
        out = await ops.get_status(config)

        assert out["vocabulary"]["status"] == "invalid"
        assert "ariel.vocabulary.path is required" in out["vocabulary"]["errors"][0]

    async def test_connection_failure_still_carries_the_block(self, monkeypatch, tmp_path):
        _patch_service_raises(monkeypatch, RuntimeError("could not connect to server"))

        path = _write_vocabulary(tmp_path, _AMBIGUOUS)
        config = dict(_DB)
        config["vocabulary"] = {"enabled": True, "path": str(path)}
        out = await ops.get_status(config)

        assert out["status"] == "error"
        assert out["vocabulary"] == {"status": "ok", "concepts": 2, "errors": []}

    async def test_relative_path_resolves_against_config_dir(self, monkeypatch, tmp_path):
        repo = _status_repo()
        _patch_service(monkeypatch, _StubService(repository=repo))

        _write_vocabulary(tmp_path, _AMBIGUOUS)
        config = dict(_DB)
        config["vocabulary"] = {"enabled": True, "path": "vocabulary.yml"}
        out = await ops.get_status(config, config_dir=tmp_path)

        assert out["vocabulary"] == {"status": "ok", "concepts": 2, "errors": []}


# ---------------------------------------------------------------------------
# check_vocabulary — the DB-free file check behind `osprey ariel vocab-check`
# ---------------------------------------------------------------------------


class TestCheckVocabulary:
    def test_three_error_file_reports_all_three(self, tmp_path):
        path = _write_vocabulary(tmp_path, _THREE_ERRORS)

        out = ops.check_vocabulary({}, str(path))

        assert out["status"] == "invalid"
        assert out["path"] == str(path)
        assert out["concepts"] == 0
        assert len(out["errors"]) == 3

    def test_valid_file_reports_ok_with_count(self, tmp_path):
        path = _write_vocabulary(tmp_path, _AMBIGUOUS)

        out = ops.check_vocabulary({}, str(path))

        assert out["status"] == "ok"
        assert out["concepts"] == 2
        assert out["errors"] == []

    def test_ambiguous_form_warns_naming_every_canonical(self, tmp_path):
        path = _write_vocabulary(tmp_path, _AMBIGUOUS)

        out = ops.check_vocabulary({}, str(path))

        assert out["status"] == "ok"
        assert len(out["warnings"]) == 1
        warning = out["warnings"][0]
        assert 'form "ts"' in warning
        assert "troubleshoot" in warning
        assert "timing system" in warning

    def test_stopword_shorthand_warns_only_when_that_direction_is_enabled(self, tmp_path):
        path = _write_vocabulary(tmp_path, _STOPWORD_SHORTHAND)

        off = ops.check_vocabulary({"vocabulary": {"canonical_to_shorthand": False}}, str(path))
        on = ops.check_vocabulary({"vocabulary": {"canonical_to_shorthand": True}}, str(path))

        assert off["warnings"] == []
        assert len(on["warnings"]) == 1
        assert 'form "a"' in on["warnings"][0]

    def test_explicit_path_beats_the_configured_one(self, tmp_path):
        configured = _write_vocabulary(tmp_path, _THREE_ERRORS)
        explicit = tmp_path / "other.yml"
        explicit.write_text(_AMBIGUOUS, encoding="utf-8")

        out = ops.check_vocabulary(
            {"vocabulary": {"enabled": True, "path": str(configured)}}, str(explicit)
        )

        assert out["path"] == str(explicit)
        assert out["status"] == "ok"

    def test_configured_relative_path_resolves_against_config_dir(self, tmp_path):
        _write_vocabulary(tmp_path, _AMBIGUOUS)

        out = ops.check_vocabulary(
            {"vocabulary": {"enabled": True, "path": "vocabulary.yml"}}, config_dir=tmp_path
        )

        assert out["status"] == "ok"
        assert out["path"] == str(tmp_path / "vocabulary.yml")

    def test_disabled_block_still_checks_its_configured_path(self, tmp_path):
        path = _write_vocabulary(tmp_path, _AMBIGUOUS)

        out = ops.check_vocabulary({"vocabulary": {"enabled": False, "path": str(path)}})

        assert out["status"] == "ok"

    def test_no_path_anywhere_is_an_error(self):
        out = ops.check_vocabulary({})

        assert out["status"] == "error"
        assert out["message"] == "no vocabulary path: pass PATH or set ariel.vocabulary.path"
        assert out["path"] is None

    def test_missing_file_is_reported_as_an_error_not_raised(self, tmp_path):
        out = ops.check_vocabulary({}, str(tmp_path / "absent.yml"))

        assert out["status"] == "invalid"
        assert "not found" in out["errors"][0]

    def test_malformed_knob_is_refused_rather_than_defaulted(self, tmp_path):
        path = _write_vocabulary(tmp_path, _AMBIGUOUS)

        out = ops.check_vocabulary({"vocabulary": {"canonical_to_acronym": "yes"}}, str(path))

        assert out["status"] == "invalid"
        assert "ariel.vocabulary.canonical_to_acronym" in out["errors"][0]


# ---------------------------------------------------------------------------
# run_search — error-branch translation and input validation
# ---------------------------------------------------------------------------


class TestRunSearch:
    async def test_empty_config_reports_not_configured(self):
        out = await ops.run_search({}, "q", "keyword", 5)
        assert out == {"error": "ARIEL not configured"}

    async def test_malformed_mode_reports_error_without_calling_service(self):
        # normalize_search_mode rejects blank names before the service is built.
        out = await ops.run_search(dict(_DB), "q", "   ", 5)
        assert out == {"error": "search mode cannot be empty"}

    async def test_unroutable_mode_reports_available_modes(self, monkeypatch):
        # A well-formed but unregistered mode reaches the service, which raises
        # ConfigurationError naming the modes a caller may actually ask for.
        from osprey.services.ariel_search.exceptions import ConfigurationError

        _patch_service_raises(
            monkeypatch,
            ConfigurationError(
                "Unknown search mode 'does-not-exist'. Available modes: keyword, semantic",
                config_key="modes",
            ),
        )
        out = await ops.run_search(dict(_DB), "q", "does-not-exist", 5)
        assert "Unknown search mode 'does-not-exist'" in out["error"]
        assert "keyword, semantic" in out["error"]

    async def test_connection_error_maps_to_friendly_message(self, monkeypatch):
        _patch_service_raises(monkeypatch, RuntimeError("connection refused"))
        out = await ops.run_search(dict(_DB), "q", "keyword", 5)
        assert "osprey up" in out["error"]

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
            search_modes_used=["keyword"],
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

    async def test_hybrid_with_picture_only_hits_returns_exactly_limit_entries(self, monkeypatch):
        # The picture lane may add up to ceil(limit/3) picture-only entries
        # beyond ``limit``; the CLI shows ``limit`` and the sources of those.
        rows = [
            {"entry_id": f"E{i}", "raw_text": f"entry {i}", "_score": 1 - i / 10} for i in range(4)
        ]
        result = SimpleNamespace(
            answer=None,
            sources=[r["entry_id"] for r in rows],
            search_modes_used=["hybrid"],
            reasoning="Hybrid search: 4 results",
            entries=rows,
        )
        service = MagicMock()
        service.__aenter__ = AsyncMock(return_value=service)
        service.__aexit__ = AsyncMock(return_value=False)
        service.search = AsyncMock(return_value=result)
        _patch_service(monkeypatch, service)

        out = await ops.run_search(dict(_DB), "orbit kick", "hybrid", 3)

        assert [e["entry_id"] for e in out["entries"]] == ["E0", "E1", "E2"]
        assert out["sources"] == ["E0", "E1", "E2"]
        assert service.search.await_args.kwargs["max_results"] == 3


# ---------------------------------------------------------------------------
# run_enhance — no-enhancers short-circuit
# ---------------------------------------------------------------------------


class TestRunEnhance:
    async def test_no_enhancers_returns_empty_result(self, monkeypatch):
        import osprey.services.ariel_search.enhancement as enh

        monkeypatch.setattr(enh, "create_enhancers_from_config", lambda config, **_: [])
        # create_ariel_service must never be reached; make it explode if it is.
        _patch_service_raises(monkeypatch, AssertionError("service should not be created"))

        out = await ops.run_enhance(dict(_DB), module=None, force=False, limit=10)

        assert out.entries_processed == 0
        assert out.module_names == []

    async def test_module_filter_narrowing_to_none_short_circuits(self, monkeypatch):
        import osprey.services.ariel_search.enhancement as enh

        enhancer = SimpleNamespace(name="text_embedding")
        monkeypatch.setattr(enh, "create_enhancers_from_config", lambda config, **_: [enhancer])
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
            unreadable_entries = 0

            async def fetch_entries(self, since=None, limit=None):  # noqa: ARG002 - the ingestion adapter fetch_entries signature
                for i in range(3):
                    yield {"entry_id": f"E{i}"}

        monkeypatch.setattr(ing, "get_adapter", lambda config: _Adapter())
        monkeypatch.setattr(enh, "create_enhancers_from_config", lambda config, **_: [])
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
            unreadable_entries = 0

            async def fetch_entries(self, since=None, limit=None):  # noqa: ARG002 - the ingestion adapter fetch_entries signature
                if False:
                    yield  # empty async generator

        monkeypatch.setattr(ing, "get_adapter", lambda config: _Adapter())
        monkeypatch.setattr(
            enh,
            "create_enhancers_from_config",
            lambda config, **_: [SimpleNamespace(name="text_embedding")],
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

    async def test_dry_run_names_the_input_limit(self, monkeypatch):
        _patch_service_raises(monkeypatch, AssertionError("service should not be created"))

        messages: list[str] = []
        await ops.run_reembed(
            dict(_DB),
            model="nomic-embed-text",
            dimension=768,
            batch_size=16,
            dry_run=True,
            force=False,
            progress=messages.append,
        )

        assert "  Input limit: 512 tokens" in messages


# ---------------------------------------------------------------------------
# run_watch — input validation
# ---------------------------------------------------------------------------


class TestRunWatch:
    async def test_missing_ingestion_block_names_the_block(self, monkeypatch):
        """A config with no ``ingestion`` at all has not configured what watch
        does, so the refusal names the block rather than a field inside one the
        operator never wrote. Rejected before any service is created."""
        from osprey.services.ariel_search.exceptions import ConfigurationError

        _patch_service_raises(monkeypatch, AssertionError("service should not be created"))

        with pytest.raises(ConfigurationError, match="ariel.ingestion is not configured") as exc:
            await ops.run_watch(
                dict(_DB),
                source=None,
                adapter=None,
                once=True,
                interval=None,
                dry_run=False,
            )

        assert exc.value.config_key == "ingestion"

    async def test_a_configured_block_without_a_source_still_names_the_source(self, monkeypatch):
        """The block is there and names its adapter; what is missing is where to
        read from, and that is what the message says."""
        _patch_service_raises(monkeypatch, AssertionError("service should not be created"))

        config_dict = {**_DB, "ingestion": {"adapter": "generic_json"}}
        with pytest.raises(ValueError, match="No ingestion source configured"):
            await ops.run_watch(
                config_dict,
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

    async def test_pictures_are_stored_natively_and_linked_on_their_entry(
        self, monkeypatch, tmp_path
    ):
        """Each picture goes through the native store after its row exists, and the
        row is written again carrying the returned items; other rows are untouched."""
        import osprey.services.ariel_search.attachments as attachments

        repo = MagicMock()
        repo.start_ingestion_run = AsyncMock(return_value="run-3")
        repo.upsert_entry = AsyncMock()
        repo.complete_ingestion_run = AsyncMock()
        repo.fail_ingestion_run = AsyncMock()
        _patch_service(monkeypatch, _StubService(repository=repo))
        stored: list[tuple[str, str, str | None, bytes]] = []

        async def _store(repository, entry_id, *, filename, declared_mime, data):
            assert repository is repo
            stored.append((entry_id, filename, declared_mime, data))
            return {"url": f"/api/attachments/att-{len(stored)}", "type": declared_mime}

        monkeypatch.setattr(attachments, "store_native_attachment", _store)
        picture = tmp_path / "trend.png"
        picture.write_bytes(b"\x89PNG\r\n\x1a\nbody")
        entries = [
            {"entry_id": "E1", "attachments": []},
            {"entry_id": "E2", "attachments": []},
        ]

        count = await ops.seed_logbook_entries(dict(_DB), entries, pictures={"E2": [picture]})

        assert count == 2
        assert stored == [("E2", "trend.png", "image/png", picture.read_bytes())]
        written = [call.args[0] for call in repo.upsert_entry.await_args_list]
        assert [(w["entry_id"], w["attachments"]) for w in written] == [
            ("E1", []),
            ("E2", []),
            ("E2", [{"url": "/api/attachments/att-1", "type": "image/png"}]),
        ]
        assert entries[1]["attachments"] == []  # the caller's record is not mutated

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


# ---------------------------------------------------------------------------
# get_status: per-module health
# ---------------------------------------------------------------------------


class _HealthModule:
    """A module double whose ``health_check`` answers *verdict* (or runs *check*)."""

    def __init__(self, name, verdict=None, *, check=None, runs_inline=True, reason=None):
        self.name = name
        self.runs_inline = runs_inline
        self._verdict = verdict
        self._check = check
        self._reason = reason

    async def health_check(self):
        if self._check is not None:
            return await self._check()
        return self._verdict

    def health_reason(self):
        return self._reason

    def required_relations(self):
        return []


def _enable(*names, **blocks):
    """A config with *names* enabled; *blocks* add keys to a module's block."""
    modules = {name: {"enabled": True, **blocks.get(name, {})} for name in names}
    return {**_DB, "enhancement_modules": modules}


def _patch_modules(monkeypatch, modules):
    """Build each module from *modules* ``{name: module | exception}``; record stages."""
    import osprey.services.ariel_search.enhancement as enhancement_pkg

    stages = []

    def _create(_config, *, stage="inline", names=None):
        stages.append(stage)
        built = []
        for name in names or []:
            item = modules.get(name)
            if isinstance(item, BaseException):
                raise item
            if item is not None:
                built.append(item)
        return built

    monkeypatch.setattr(enhancement_pkg, "create_enhancers_from_config", _create)
    return stages


def _health(out, name):
    return out["enhancement_modules"][name]["health"]


class TestGetStatusModuleHealth:
    """Every enabled module carries ``health`` from its own check."""

    @pytest.mark.parametrize("reason", ["auth", "model", "unreachable", "config"])
    async def test_each_unhealthy_reason_is_reported(self, monkeypatch, reason):
        from osprey.models.providers.health import HealthResult

        _patch_service(monkeypatch, _StubService(repository=_status_repo()))
        _patch_modules(
            monkeypatch,
            {"text_embedding": _HealthModule("text_embedding", HealthResult(False, "x", reason))},
        )

        out = await ops.get_status(_enable("text_embedding"))

        assert _health(out, "text_embedding") == {
            "reachable": False,
            "reason": reason,
            "probed_from": "this process",
        }
        assert out["status"] == "healthy"

    async def test_healthy_module_reports_reachable_with_no_reason(self, monkeypatch):
        from osprey.models.providers.health import HealthResult

        _patch_service(monkeypatch, _StubService(repository=_status_repo()))
        _patch_modules(
            monkeypatch,
            {"qmd_export": _HealthModule("qmd_export", HealthResult(True, "OK", None))},
        )

        out = await ops.get_status(_enable("qmd_export"))

        assert _health(out, "qmd_export") == {
            "reachable": True,
            "reason": None,
            "probed_from": "this process",
        }

    async def test_module_without_a_health_check_reports_null(self, monkeypatch):
        from osprey.models.providers.health import HealthResult

        _patch_service(monkeypatch, _StubService(repository=_status_repo()))
        verdict = HealthResult(None, "no health check", None)
        _patch_modules(monkeypatch, {"qmd_export": _HealthModule("qmd_export", verdict)})

        out = await ops.get_status(_enable("qmd_export"))

        assert _health(out, "qmd_export")["reachable"] is None
        assert _health(out, "qmd_export")["reason"] is None

    async def test_a_legacy_tuple_is_normalised(self, monkeypatch):
        _patch_service(monkeypatch, _StubService(repository=_status_repo()))
        _patch_modules(
            monkeypatch, {"qmd_export": _HealthModule("qmd_export", (False, "mirror gone"))}
        )

        out = await ops.get_status(_enable("qmd_export"))

        assert _health(out, "qmd_export")["reachable"] is False
        assert _health(out, "qmd_export")["reason"] == "unreachable"

    async def test_health_reason_overrides_and_keeps_reachable(self, monkeypatch):
        from osprey.models.providers.health import HealthResult

        _patch_service(monkeypatch, _StubService(repository=_status_repo()))
        module = _HealthModule("qmd_export", HealthResult(True, "OK", None), reason="no_reader")
        _patch_modules(monkeypatch, {"qmd_export": module})

        out = await ops.get_status(_enable("qmd_export"))

        assert _health(out, "qmd_export")["reachable"] is True
        assert _health(out, "qmd_export")["reason"] == "no_reader"

    async def test_disabled_modules_carry_no_health(self, monkeypatch):
        _patch_service(monkeypatch, _StubService(repository=_status_repo()))
        _patch_modules(monkeypatch, {})

        out = await ops.get_status(dict(_DB))

        assert all("health" not in entry for entry in out["enhancement_modules"].values())

    async def test_each_module_is_built_alone_with_stage_all(self, monkeypatch):
        from osprey.models.providers.health import HealthResult

        _patch_service(monkeypatch, _StubService(repository=_status_repo()))
        ok = HealthResult(True, "OK", None)
        stages = _patch_modules(
            monkeypatch,
            {
                "qmd_export": _HealthModule("qmd_export", ok),
                "text_embedding": _HealthModule("text_embedding", ok),
            },
        )

        await ops.get_status(_enable("qmd_export", "text_embedding"))

        assert stages == ["all", "all"]

    async def test_configure_error_reports_config_and_status_stays_healthy(self, monkeypatch):
        from osprey.models.providers.health import HealthResult
        from osprey.services.ariel_search.exceptions import ModuleConfigError

        _patch_service(monkeypatch, _StubService(repository=_status_repo()))
        _patch_modules(
            monkeypatch,
            {
                "image_caption": ModuleConfigError("provider is required", key="x.provider"),
                "text_embedding": _HealthModule("text_embedding", HealthResult(True, "OK", None)),
            },
        )

        out = await ops.get_status(_enable("image_caption", "text_embedding"))

        assert out["status"] == "healthy"
        assert _health(out, "image_caption") == {
            "reachable": False,
            "reason": "config",
            "probed_from": "this process",
        }
        assert _health(out, "text_embedding")["reachable"] is True

    async def test_misconfigured_image_caption_through_the_real_factory(self, monkeypatch):
        """The real module refuses a provider-less block; the store's status is untouched."""
        _patch_service(monkeypatch, _StubService(repository=_status_repo()))

        out = await ops.get_status(
            _enable("image_caption", image_caption={"model": {"model_id": "vis"}})
        )

        assert out["status"] == "healthy"
        assert _health(out, "image_caption")["reason"] == "config"
        assert _health(out, "image_caption")["reachable"] is False

    async def test_a_check_that_raises_is_classified(self, monkeypatch):
        _patch_service(monkeypatch, _StubService(repository=_status_repo()))

        async def boom():
            raise ConnectionError("refused")

        _patch_modules(monkeypatch, {"text_embedding": _HealthModule("text_embedding", check=boom)})

        out = await ops.get_status(_enable("text_embedding"))

        assert _health(out, "text_embedding")["reason"] == "unreachable"

    async def test_a_synchronously_sleeping_check_returns_within_6_s(self, monkeypatch, tmp_path):
        """The real qmd_export check runs off the loop, so the 5 s timeout returns."""
        import time

        from osprey.services.ariel_search.enhancement.qmd_export.exporter import (
            QmdExportModule,
        )

        def slow(_root):
            time.sleep(30)

        monkeypatch.setattr(QmdExportModule, "_mirror_health", staticmethod(slow))
        _patch_service(monkeypatch, _StubService(repository=_status_repo()))
        config = _enable("qmd_export", qmd_export={"mirror_path": str(tmp_path / "m")})

        started = time.monotonic()
        out = await ops.get_status(config)

        assert time.monotonic() - started < 6
        assert _health(out, "qmd_export") == {
            "reachable": False,
            "reason": "unreachable",
            "probed_from": "this process",
        }

    async def test_checks_run_concurrently(self, monkeypatch):
        """Two checks of 4 s each finish in under 6 s together."""
        import asyncio
        import time

        from osprey.models.providers.health import HealthResult

        async def four_seconds():
            await asyncio.sleep(4)
            return HealthResult(True, "OK", None)

        _patch_service(monkeypatch, _StubService(repository=_status_repo()))
        _patch_modules(
            monkeypatch,
            {
                "qmd_export": _HealthModule("qmd_export", check=four_seconds),
                "text_embedding": _HealthModule("text_embedding", check=four_seconds),
            },
        )

        started = time.monotonic()
        out = await ops.get_status(_enable("qmd_export", "text_embedding"))

        assert time.monotonic() - started < 6
        assert _health(out, "qmd_export")["reachable"] is True
        assert _health(out, "text_embedding")["reachable"] is True


@pytest.fixture
def caption_isolation(monkeypatch):
    """Fresh availability/offload/local-server state; no real Ollama fallback answers."""
    from osprey.models.providers import _local_server
    from osprey.models.providers.ollama import OllamaProviderAdapter
    from osprey.services.ariel_search.enhancement import _offload, availability

    availability.reset_availability()
    _offload.reset_offload_state()
    _local_server.reset_cache()
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.setattr(_local_server, "container_fallback_urls", lambda *_a, **_k: [])
    monkeypatch.setattr(OllamaProviderAdapter, "_get_fallback_urls", staticmethod(lambda u: []))
    yield
    availability.reset_availability()
    _offload.reset_offload_state()
    _local_server.reset_cache()


@pytest.mark.usefixtures("caption_isolation")
class TestGetStatusImageModuleHealth:
    """Picture modules answer through the catch-up's own pre-pass check."""

    @staticmethod
    def _caption_config(monkeypatch, base_url):
        monkeypatch.setattr(
            "osprey.models.config.get_provider_config",
            lambda name: {"base_url": base_url} if name == "ollama" else {},
        )
        return _enable(
            "image_caption",
            image_caption={"provider": "ollama", "model": {"model_id": "qwen3-vl:4b"}},
        )

    async def test_model_without_vision_reports_model(self, monkeypatch):
        from tests.services.ariel_search.test_image_caption import StubOllama

        stub = StubOllama({"qwen3-vl:4b": ["completion"]})
        try:
            _patch_service(monkeypatch, _StubService(repository=_status_repo()))
            out = await ops.get_status(self._caption_config(monkeypatch, stub.url))
        finally:
            stub.stop()

        assert out["enhancement_modules"]["image_caption"]["health"]["reason"] == "model"
        assert out["status"] == "healthy"

    async def test_schema_behind_reports_config_and_store_stays_healthy(self, monkeypatch):
        _patch_service(monkeypatch, _StubService(repository=_status_repo(copy_state=False)))

        out = await ops.get_status(self._caption_config(monkeypatch, "http://127.0.0.1:9"))

        assert out["status"] == "healthy"
        assert out["enhancement_modules"]["image_caption"]["health"] == {
            "reachable": False,
            "reason": "config",
            "probed_from": "this process",
        }


class TestStatusTextModuleHealth:
    """``osprey ariel status`` prints one line per enabled module it found unusable."""

    def _run(self, monkeypatch, verdicts, config):
        from click.testing import CliRunner

        from osprey.cli.ariel import ariel_group

        _patch_service(monkeypatch, _StubService(repository=_status_repo()))
        _patch_modules(monkeypatch, {n: _HealthModule(n, v) for n, v in verdicts.items()})
        monkeypatch.setattr(
            "osprey.cli.ariel.get_config_value",
            lambda key, default=None: (
                config
                if key == "ariel"
                else ("http://user:secret@gpu:8080/x?k=1" if key.endswith("base_url") else default)
            ),
        )
        result = CliRunner().invoke(ariel_group, ["status"])
        assert result.exit_code == 0, result.output
        return result.output

    def test_model_line_names_model_provider_and_fix(self, monkeypatch):
        from osprey.models.providers.health import HealthResult

        config = _enable(
            "image_caption",
            image_caption={"provider": "ollama", "model": {"model_id": "qwen3-vl:4b"}},
        )
        text = self._run(
            monkeypatch, {"image_caption": HealthResult(False, "no vision", "model")}, config
        )

        assert (
            "image_caption: skipped, model qwen3-vl:4b not available on ollama (pull it, "
            "or set ariel.enhancement_modules.image_caption.enabled: false)"
        ) in text

    def test_unreachable_line_redacts_the_base_url(self, monkeypatch):
        from osprey.models.providers.health import HealthResult

        config = _enable("text_embedding", text_embedding={"provider": "llama-cpp"})
        text = self._run(
            monkeypatch, {"text_embedding": HealthResult(False, "down", "unreachable")}, config
        )

        assert "text_embedding: skipped, llama-cpp not reachable at http://gpu:8080/x" in text
        assert "start llama-server, see the picture-search guide" in text
        assert "secret" not in text
        assert "ariel.enhancement_modules.text_embedding.enabled: false" in text

    def test_auth_line(self, monkeypatch):
        from osprey.models.providers.health import HealthResult

        config = _enable("semantic_processor", semantic_processor={"provider": "cborg"})
        text = self._run(
            monkeypatch, {"semantic_processor": HealthResult(False, "401", "auth")}, config
        )

        assert "semantic_processor: skipped, cborg refused the API key" in text
        assert "api.providers.cborg" in text

    def test_config_line_names_migrate(self, monkeypatch):
        from osprey.models.providers.health import HealthResult

        config = _enable("image_caption", image_caption={"provider": "ollama"})
        text = self._run(
            monkeypatch, {"image_caption": HealthResult(False, "schema", "config")}, config
        )

        assert "image_caption: skipped," in text
        assert "osprey ariel migrate" in text

    def test_no_reader_line(self):
        from osprey.cli.ariel import module_skip_line

        line = module_skip_line(
            "image_caption",
            "no_reader",
            _enable("image_caption", image_caption={"provider": "x", "base_url": "http://h"}),
        )

        assert line.startswith("image_caption: skipped, x cannot read pictures")

    def test_healthy_and_unchecked_modules_print_nothing(self, monkeypatch):
        from osprey.models.providers.health import HealthResult

        config = _enable("qmd_export", "text_embedding")
        text = self._run(
            monkeypatch,
            {
                "qmd_export": HealthResult(True, "OK", None),
                "text_embedding": HealthResult(None, "no health check", None),
            },
            config,
        )

        assert "skipped," not in text
