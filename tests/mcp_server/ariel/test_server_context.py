"""Tests for ARIEL MCP server context."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp.exceptions import ToolError

from osprey.mcp_server.ariel.server_context import (
    ARIELContext,
    get_ariel_context,
    initialize_ariel_context,
    reset_ariel_context,
)
from osprey.services.ariel_search.exceptions import ConfigurationError, VocabularyError
from tests.mcp_server.conftest import assert_raises_error


def test_initialize_loads_config(tmp_path, monkeypatch):
    """Registry initialization loads ariel config from config.yml."""
    monkeypatch.chdir(tmp_path)
    config = {
        "ariel": {
            "database": {"uri": "postgresql://localhost:5432/ariel"},
            "search_modules": {"keyword": {"enabled": True}},
        }
    }
    (tmp_path / "config.yml").write_text(json.dumps(config))

    registry = initialize_ariel_context()
    assert registry.config is not None
    assert registry.config.database.uri == "postgresql://localhost:5432/ariel"


def test_initialize_connection_string_compat(tmp_path, monkeypatch):
    """Registry maps connection_string to uri for DatabaseConfig compatibility."""
    monkeypatch.chdir(tmp_path)
    config = {
        "ariel": {
            "database": {"connection_string": "postgresql://localhost:5432/ariel"},
        }
    }
    (tmp_path / "config.yml").write_text(json.dumps(config))

    registry = initialize_ariel_context()
    assert registry.config.database.uri == "postgresql://localhost:5432/ariel"


def test_initialize_missing_ariel_section(tmp_path, monkeypatch):
    """Missing ariel section warns but doesn't crash."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("control_system: {}\n")

    registry = initialize_ariel_context()
    with pytest.raises(RuntimeError, match="ARIEL config not available"):
        _ = registry.config


def test_get_registry_before_init():
    """get_ariel_context raises before initialization."""
    reset_ariel_context()
    with pytest.raises(RuntimeError, match="not initialized"):
        get_ariel_context()


def test_reset_clears_singleton():
    """reset_ariel_context clears the singleton."""
    # Initialize first
    with patch.object(ARIELContext, "_load_config", return_value={}):
        initialize_ariel_context()

    # Should work
    get_ariel_context()

    # Reset
    reset_ariel_context()

    # Should fail now
    with pytest.raises(RuntimeError):
        get_ariel_context()


async def test_service_caches(tmp_path, monkeypatch):
    """service() creates the service once and caches it."""
    monkeypatch.chdir(tmp_path)
    config = {
        "ariel": {
            "database": {"uri": "postgresql://localhost:5432/ariel"},
        }
    }
    (tmp_path / "config.yml").write_text(json.dumps(config))

    registry = initialize_ariel_context()

    mock_service = AsyncMock()
    with patch(
        "osprey.services.ariel_search.service.create_ariel_service",
        new=AsyncMock(return_value=mock_service),
    ) as mock_create:
        svc1 = await registry.service()
        svc2 = await registry.service()

    assert svc1 is svc2
    assert mock_create.call_count == 1


# ---------------------------------------------------------------------------
# Eager validation: the MCP server refuses to start on a broken config
# ---------------------------------------------------------------------------

VOCABULARY_YML = """\
concepts:
  - canonical: beam position monitor
    kind: acronym
    forms: [bpm, bpms]
"""


def _write_project(project_dir, vocabulary_block):
    """Write a config.yml carrying the given ariel.vocabulary block.

    Args:
        project_dir: Directory to treat as the project root.
        vocabulary_block: The ``ariel.vocabulary`` mapping.

    Returns:
        Path to the written config.yml.
    """
    config = {
        "ariel": {
            "database": {"uri": "postgresql://localhost:5432/ariel"},
            "vocabulary": vocabulary_block,
        }
    }
    config_path = project_dir / "config.yml"
    config_path.write_text(json.dumps(config))
    return config_path


def test_initialize_refuses_on_missing_vocabulary_file(tmp_path, monkeypatch):
    """A vocabulary that cannot be loaded refuses startup, naming the failing key."""
    monkeypatch.chdir(tmp_path)
    _write_project(tmp_path, {"enabled": True, "path": "data/missing.yml"})

    with pytest.raises(ConfigurationError) as excinfo:
        initialize_ariel_context()

    assert isinstance(excinfo.value, VocabularyError)
    assert "ariel.vocabulary.path" in str(excinfo.value)
    assert excinfo.value.config_key == "ariel.vocabulary.path"


def test_initialize_refuses_when_enabled_without_path(tmp_path, monkeypatch):
    """enabled: true with no path refuses startup naming ariel.vocabulary.path."""
    monkeypatch.chdir(tmp_path)
    _write_project(tmp_path, {"enabled": True})

    with pytest.raises(ConfigurationError) as excinfo:
        initialize_ariel_context()

    assert "ariel.vocabulary.path" in str(excinfo.value)


def test_initialize_resolves_relative_path_against_config_dir(tmp_path, monkeypatch):
    """A relative vocabulary path resolves against the config file's directory, not the CWD."""
    project = tmp_path / "project"
    (project / "data").mkdir(parents=True)
    (project / "data" / "vocabulary.yml").write_text(VOCABULARY_YML)
    config_path = _write_project(project, {"enabled": True, "path": "data/vocabulary.yml"})

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setenv("OSPREY_CONFIG", str(config_path))

    registry = initialize_ariel_context()

    assert registry.config.loaded_vocabulary is not None
    assert registry.config.loaded_vocabulary.concept_count > 0


def test_initialize_disabled_vocabulary_with_missing_file_starts(tmp_path, monkeypatch):
    """A disabled vocabulary reads nothing, so a stale path does not refuse startup."""
    monkeypatch.chdir(tmp_path)
    _write_project(tmp_path, {"enabled": False, "path": "data/missing.yml"})

    registry = initialize_ariel_context()

    assert registry.config.loaded_vocabulary is None
    assert registry.config.vocabulary_errors == []


def test_initialize_missing_ariel_section_still_does_not_raise(tmp_path, monkeypatch):
    """Eager validation does not turn a missing ariel section into a startup crash."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("control_system: {}\n")

    registry = initialize_ariel_context()

    assert registry.raw_config.get("ariel") is None


def test_initialize_refuses_a_malformed_entry_text_budget(tmp_path, monkeypatch):
    """A malformed entry-text budget refuses startup and names the key."""
    monkeypatch.chdir(tmp_path)
    config = {
        "ariel": {
            "database": {"uri": "postgresql://localhost:5432/ariel"},
            "entry_text": {"listing_chars": "lots"},
        }
    }
    (tmp_path / "config.yml").write_text(json.dumps(config))

    with pytest.raises(ValueError, match=r"ariel\.entry_text\.listing_chars"):
        initialize_ariel_context()


# ---------------------------------------------------------------------------
# Service open: a missing embedding table is refused, never cached
# ---------------------------------------------------------------------------

_TABLE_MISSING = (
    "search_modules.semantic.model: semantic search is enabled for model 'm', "
    "but this database has no embedding table for it ('text_embeddings_m')"
)
_TABLE_REMEDY = "Run `osprey ariel quickstart`, or set search_modules.semantic.enabled: false"


def _write_semantic_project(project_dir, *, semantic_enabled=True):
    """Write a config.yml whose ARIEL section enables (or not) semantic search."""
    config = {
        "ariel": {
            "database": {"uri": "postgresql://localhost:5432/ariel"},
            "search_modules": {
                "keyword": {"enabled": True},
                "semantic": {"enabled": semantic_enabled, "model": "m"},
            },
        }
    }
    (project_dir / "config.yml").write_text(json.dumps(config))


def _service_factory(outcomes):
    """Patch target for ``create_ariel_service`` building real services.

    Each built service's repository answers the table probe with the next item
    of ``outcomes`` (an exception to raise, or ``None`` to pass), and its
    ``aclose`` is recorded instead of closing pools that were never opened.
    """
    from osprey.services.ariel_search.service import ARIELSearchService

    built: list[ARIELSearchService] = []
    repository = MagicMock()
    repository.validate_search_model_table = AsyncMock(side_effect=outcomes)

    async def create(config, **kwargs):
        # Yield once, as a real pool open does, so racing callers interleave.
        await asyncio.sleep(0)
        service = ARIELSearchService(config=config, pool=MagicMock(), repository=repository)
        service.aclose = AsyncMock()  # type: ignore[method-assign]
        built.append(service)
        return service

    return create, built, repository


def _table_missing(remedy=_TABLE_REMEDY):
    return ConfigurationError(
        _TABLE_MISSING, config_key="search_modules.semantic.model", remedy=remedy
    )


async def test_service_refuses_missing_embedding_table_and_caches_nothing(tmp_path, monkeypatch):
    """A refused probe closes the built service, raises the tool refusal, caches nothing."""
    monkeypatch.chdir(tmp_path)
    _write_semantic_project(tmp_path)
    ctx = initialize_ariel_context()
    create, built, _ = _service_factory([_table_missing()])

    with patch("osprey.services.ariel_search.service.create_ariel_service", new=create):
        with assert_raises_error(error_type="service_unavailable") as captured:
            await ctx.service()

    assert captured["envelope"]["error_message"] == _TABLE_MISSING
    assert captured["envelope"]["suggestions"] == [_TABLE_REMEDY]
    assert len(built) == 1
    built[0].aclose.assert_awaited_once()
    assert ctx._service is None


async def test_service_refusal_without_a_remedy_suggests_nothing(tmp_path, monkeypatch, caplog):
    """A refusal that names no fix is rendered without one, and no restart is advised."""
    monkeypatch.chdir(tmp_path)
    _write_semantic_project(tmp_path)
    ctx = initialize_ariel_context()
    create, _, _ = _service_factory([_table_missing(remedy=None)])

    with patch("osprey.services.ariel_search.service.create_ariel_service", new=create):
        with assert_raises_error(error_type="service_unavailable") as captured:
            await ctx.service()

    assert captured["envelope"]["suggestions"] == []
    assert "Remedy" not in caplog.text
    assert "restart" not in caplog.text


async def test_service_serves_once_the_table_exists(tmp_path, monkeypatch):
    """After the operator runs the fix, the next call builds and caches a service."""
    monkeypatch.chdir(tmp_path)
    _write_semantic_project(tmp_path)
    ctx = initialize_ariel_context()
    create, built, _ = _service_factory([_table_missing(), None])

    with patch("osprey.services.ariel_search.service.create_ariel_service", new=create):
        with assert_raises_error(error_type="service_unavailable"):
            await ctx.service()
        service = await ctx.service()
        again = await ctx.service()

    assert service is again is built[1]
    assert ctx._service is service
    built[1].aclose.assert_not_awaited()


async def test_service_with_semantic_disabled_asks_no_table(tmp_path, monkeypatch):
    """With semantic search off the service opens without a table probe."""
    monkeypatch.chdir(tmp_path)
    _write_semantic_project(tmp_path, semantic_enabled=False)
    ctx = initialize_ariel_context()
    create, built, repository = _service_factory([_table_missing()])

    with patch("osprey.services.ariel_search.service.create_ariel_service", new=create):
        service = await ctx.service()

    assert service is built[0]
    repository.validate_search_model_table.assert_not_called()


async def test_concurrent_first_calls_build_one_service(tmp_path, monkeypatch):
    """Two first calls racing against a passing probe build exactly one service."""
    monkeypatch.chdir(tmp_path)
    _write_semantic_project(tmp_path)
    ctx = initialize_ariel_context()
    create, built, _ = _service_factory([None, None])

    with patch("osprey.services.ariel_search.service.create_ariel_service", new=create):
        first, second = await asyncio.gather(ctx.service(), ctx.service())

    assert first is second
    assert len(built) == 1


async def test_concurrent_first_calls_both_refused(tmp_path, monkeypatch):
    """Two first calls racing against a refusing probe both raise; each build is closed."""
    monkeypatch.chdir(tmp_path)
    _write_semantic_project(tmp_path)
    ctx = initialize_ariel_context()
    create, built, _ = _service_factory([_table_missing(), _table_missing()])

    with patch("osprey.services.ariel_search.service.create_ariel_service", new=create):
        results = await asyncio.gather(ctx.service(), ctx.service(), return_exceptions=True)

    assert all(isinstance(r, ToolError) for r in results)
    assert len(built) == 2
    for service in built:
        service.aclose.assert_awaited_once()
    assert ctx._service is None


async def test_search_tool_answers_with_the_refusal(tmp_path, monkeypatch):
    """A search tool called while the table is missing returns the error envelope."""
    from osprey.mcp_server.ariel.tools.keyword_search import keyword_search
    from tests.mcp_server.ariel.conftest import get_tool_fn

    monkeypatch.chdir(tmp_path)
    _write_semantic_project(tmp_path)
    initialize_ariel_context()
    create, _, _ = _service_factory([_table_missing()])

    with patch("osprey.services.ariel_search.service.create_ariel_service", new=create):
        with assert_raises_error(error_type="service_unavailable") as captured:
            await get_tool_fn(keyword_search)(query="beam current")

    assert captured["envelope"]["error_message"] == _TABLE_MISSING
