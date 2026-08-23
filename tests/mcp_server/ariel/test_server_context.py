"""Tests for ARIEL MCP server context."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from osprey.mcp_server.ariel.server_context import (
    ARIELContext,
    get_ariel_context,
    initialize_ariel_context,
    reset_ariel_context,
)
from osprey.services.ariel_search.exceptions import ConfigurationError, VocabularyError


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
def test_initialize_missing_ariel_section(tmp_path, monkeypatch):
    """Missing ariel section warns but doesn't crash."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("control_system: {}\n")

    registry = initialize_ariel_context()
    with pytest.raises(RuntimeError, match="ARIEL config not available"):
        _ = registry.config


@pytest.mark.unit
def test_get_registry_before_init():
    """get_ariel_context raises before initialization."""
    reset_ariel_context()
    with pytest.raises(RuntimeError, match="not initialized"):
        get_ariel_context()


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
def test_initialize_refuses_on_missing_vocabulary_file(tmp_path, monkeypatch):
    """A vocabulary that cannot be loaded refuses startup, naming the failing key."""
    monkeypatch.chdir(tmp_path)
    _write_project(tmp_path, {"enabled": True, "path": "data/missing.yml"})

    with pytest.raises(ConfigurationError) as excinfo:
        initialize_ariel_context()

    assert isinstance(excinfo.value, VocabularyError)
    assert "ariel.vocabulary.path" in str(excinfo.value)
    assert excinfo.value.config_key == "ariel.vocabulary.path"


@pytest.mark.unit
def test_initialize_refuses_when_enabled_without_path(tmp_path, monkeypatch):
    """enabled: true with no path refuses startup naming ariel.vocabulary.path."""
    monkeypatch.chdir(tmp_path)
    _write_project(tmp_path, {"enabled": True})

    with pytest.raises(ConfigurationError) as excinfo:
        initialize_ariel_context()

    assert "ariel.vocabulary.path" in str(excinfo.value)


@pytest.mark.unit
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


@pytest.mark.unit
def test_initialize_disabled_vocabulary_with_missing_file_starts(tmp_path, monkeypatch):
    """A disabled vocabulary reads nothing, so a stale path does not refuse startup."""
    monkeypatch.chdir(tmp_path)
    _write_project(tmp_path, {"enabled": False, "path": "data/missing.yml"})

    registry = initialize_ariel_context()

    assert registry.config.loaded_vocabulary is None
    assert registry.config.vocabulary_errors == []


@pytest.mark.unit
def test_initialize_missing_ariel_section_still_does_not_raise(tmp_path, monkeypatch):
    """Eager validation does not turn a missing ariel section into a startup crash."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("control_system: {}\n")

    registry = initialize_ariel_context()

    assert registry.raw_config.get("ariel") is None
