"""Tests for the ARIEL web panel's configuration lifecycle.

The panel is always-on: a broken ``config.yml`` must never stop it from
starting, must be reported as a configuration problem rather than a database
problem, and must not prevent the search service from being constructed when
the database is actually reachable.

Every test drives the REAL lifespan through ``TestClient(app)`` (the pattern of
``test_app.py::test_app_with_lifespan``) with only ``create_ariel_service``
patched, so the classification under test is the one the panel really performs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml
from fastapi.testclient import TestClient

DSN = "postgresql://user:pass@localhost:5432/ariel"

VALID_VOCABULARY = """
concepts:
  - canonical: troubleshoot
    kind: shorthand
    forms:
      - t/s
      - ts
"""


@pytest.fixture
def service_double():
    """A stand-in for a live ARIEL service."""
    service = AsyncMock()
    service.health_check = AsyncMock(return_value=(True, "Service healthy"))
    service.__aenter__ = AsyncMock(return_value=service)
    service.__aexit__ = AsyncMock(return_value=None)
    return service


def _write_config(tmp_path: Path, ariel_section: dict) -> Path:
    """Write a real config.yml carrying the given ``ariel`` section."""
    config_file = tmp_path / "config.yml"
    config_file.write_text(yaml.dump({"ariel": ariel_section}))
    return config_file


def _banner(caplog: pytest.LogCaptureFixture, needle: str) -> str:
    """Return the single logged message containing ``needle``."""
    for record in caplog.records:
        message = record.getMessage()
        if needle in message:
            return message
    raise AssertionError(f"no log record containing {needle!r}. Captured:\n{caplog.text}")


def _assert_no_banner(caplog: pytest.LogCaptureFixture, needle: str) -> None:
    """Assert no logged message contains ``needle``."""
    for record in caplog.records:
        assert needle not in record.getMessage(), f"unexpected {needle!r} banner"


def _start(config_file: Path | None, create_ariel_service, caplog):
    """Run the real lifespan and return the started app."""
    from osprey.interfaces.ariel import create_app

    caplog.set_level(logging.INFO, logger="ariel")
    with patch("osprey.services.ariel_search.create_ariel_service", create_ariel_service):
        app = create_app(config_file)
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200
    return app


def test_missing_vocabulary_file_without_database(tmp_path: Path, caplog):
    """enabled: true + missing file + dead database: invalid config, panel up."""
    config_file = _write_config(
        tmp_path,
        {
            "database": {"uri": DSN},
            "vocabulary": {"enabled": True, "path": "vocabulary.yml"},
        },
    )
    dead_db = AsyncMock(side_effect=RuntimeError("connection refused"))

    app = _start(config_file, dead_db, caplog)

    from osprey.services.ariel_search.exceptions import VocabularyError

    assert app.state.config_status == "configuration_invalid"
    assert app.state.ariel_config is not None
    assert len(app.state.config_errors) == 1
    assert "ariel.vocabulary.path" in app.state.config_errors[0]
    assert app.state.config_remedy == VocabularyError.remedy
    assert app.state.config_path == config_file

    banner = _banner(caplog, "CONFIGURATION INVALID")
    assert "ariel.vocabulary.path" in banner
    assert VocabularyError.remedy in banner
    # The configuration banner never blames the database.
    assert "DATABASE NOT AVAILABLE" not in banner


def test_service_is_constructed_despite_config_errors(tmp_path: Path, service_double, caplog):
    """A vocabulary error degrades search only — the service is still built."""
    config_file = _write_config(
        tmp_path,
        {
            "database": {"uri": DSN},
            "vocabulary": {"enabled": True, "path": "vocabulary.yml"},
        },
    )
    create = AsyncMock(return_value=service_double)

    app = _start(config_file, create, caplog)

    assert app.state.config_status == "configuration_invalid"
    assert app.state.config_errors
    assert app.state.ariel_service is not None
    create.assert_awaited_once()
    _assert_no_banner(caplog, "DATABASE NOT AVAILABLE")


def test_no_config_file_at_all(tmp_path: Path, monkeypatch, caplog):
    """No config.yml anywhere: the panel starts and names CONFIG_FILE."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CONFIG_FILE", raising=False)
    create = AsyncMock()

    app = _start(None, create, caplog)

    assert app.state.config_status == "configuration_invalid"
    assert app.state.ariel_config is None
    assert app.state.config_path is None
    assert app.state.ariel_service is None
    assert "No config.yml found" in app.state.config_errors[0]
    assert "CONFIG_FILE" in app.state.config_remedy
    assert "ariel.vocabulary.path" not in app.state.config_remedy
    # No config means no database attempt, so no database banner.
    create.assert_not_awaited()
    _banner(caplog, "CONFIGURATION INVALID")
    _assert_no_banner(caplog, "DATABASE NOT AVAILABLE")


def test_legacy_pipelines_section(tmp_path: Path, caplog):
    """A legacy ariel.pipelines section is a configuration error, not a crash."""
    config_file = _write_config(
        tmp_path,
        {"database": {"uri": DSN}, "pipelines": {"rag": {"enabled": True}}},
    )
    create = AsyncMock()

    app = _start(config_file, create, caplog)

    assert app.state.config_status == "configuration_invalid"
    assert app.state.ariel_config is None
    assert app.state.config_path == config_file
    assert "pipelines" in app.state.config_errors[0]
    assert app.state.config_remedy == "fix the named key in config.yml and restart"
    create.assert_not_awaited()
    _banner(caplog, "CONFIGURATION INVALID")


def test_pre_existing_validate_error_is_a_warning(tmp_path: Path, service_double, caplog):
    """A missing semantic model warns; the service is still constructed."""
    vocabulary_file = tmp_path / "vocabulary.yml"
    vocabulary_file.write_text(VALID_VOCABULARY)
    config_file = _write_config(
        tmp_path,
        {
            "database": {"uri": DSN},
            "search_modules": {"semantic": {"enabled": True}},
            "vocabulary": {"enabled": True, "path": "vocabulary.yml"},
        },
    )
    create = AsyncMock(return_value=service_double)

    app = _start(config_file, create, caplog)

    assert app.state.config_status == "configuration_warning"
    assert app.state.config_errors == [
        "search_modules.semantic.model is required when semantic search is enabled"
    ]
    assert app.state.config_remedy == "fix the named key in config.yml and restart"
    assert app.state.ariel_service is not None

    banner = _banner(caplog, "CONFIGURATION WARNING")
    assert "search_modules.semantic.model" in banner
    _assert_no_banner(caplog, "CONFIGURATION INVALID")


def test_clean_config_is_ok(tmp_path: Path, service_double, caplog):
    """A clean config leaves no errors and records the file it came from."""
    config_file = _write_config(tmp_path, {"database": {"uri": DSN}})
    create = AsyncMock(return_value=service_double)

    app = _start(config_file, create, caplog)

    assert app.state.config_status == "ok"
    assert app.state.config_errors == []
    assert app.state.config_remedy is None
    assert app.state.config_path == config_file
    assert app.state.ariel_service is not None
    _assert_no_banner(caplog, "CONFIGURATION")


def test_relative_vocabulary_path_resolves_against_the_config_dir(
    tmp_path: Path, monkeypatch, service_double, caplog
):
    """CONFIG_FILE outside the CWD: a relative vocabulary path follows the file."""
    project = tmp_path / "project"
    project.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (project / "vocabulary.yml").write_text(VALID_VOCABULARY)
    config_file = _write_config(
        project,
        {
            "database": {"uri": DSN},
            "vocabulary": {"enabled": True, "path": "vocabulary.yml"},
        },
    )

    monkeypatch.chdir(elsewhere)
    monkeypatch.setenv("CONFIG_FILE", str(config_file))
    create = AsyncMock(return_value=service_double)

    # No explicit config_path: discovery must find CONFIG_FILE and resolve the
    # relative vocabulary path against ITS directory, not the process CWD.
    app = _start(None, create, caplog)

    assert app.state.config_status == "ok"
    assert app.state.config_errors == []
    assert app.state.config_path == config_file
    assert app.state.ariel_config.loaded_vocabulary is not None
    assert app.state.ariel_config.vocabulary_active is True


def test_database_down_with_clean_config(tmp_path: Path, caplog):
    """A dead database with a clean config keeps the existing degraded banner."""
    config_file = _write_config(tmp_path, {"database": {"uri": DSN}})
    dead_db = AsyncMock(side_effect=RuntimeError("connection refused"))

    app = _start(config_file, dead_db, caplog)

    assert app.state.config_status == "ok"
    assert app.state.config_errors == []
    assert app.state.ariel_service is None

    banner = _banner(caplog, "DATABASE NOT AVAILABLE")
    assert "DEGRADED MODE" in banner
    assert "connection refused" in banner
    _assert_no_banner(caplog, "CONFIGURATION INVALID")


def test_load_ariel_config_with_path_returns_the_resolved_file(tmp_path: Path):
    """The sibling loader reports the file the dictionary was read from."""
    from osprey.interfaces.ariel.app import load_ariel_config, load_ariel_config_with_path

    config_file = _write_config(tmp_path, {"database": {"uri": DSN}})

    config_dict, path = load_ariel_config_with_path(config_file)

    assert path == config_file
    assert config_dict["database"]["uri"] == DSN
    # The old entry point stays a pure delegation.
    assert load_ariel_config(config_file) == config_dict
