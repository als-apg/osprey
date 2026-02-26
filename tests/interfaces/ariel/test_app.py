"""Tests for ARIEL web interface app factory."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def mock_ariel_config():
    """Mock ARIEL configuration."""
    return {
        "database": {
            "uri": "postgresql://user:pass@localhost:5432/ariel",
        },
        "embeddings": {
            "provider": "openai",
            "model": "text-embedding-3-small",
        },
    }


@pytest.fixture
def mock_ariel_service():
    """Mock ARIEL service."""
    service = AsyncMock()
    service.health_check = AsyncMock(return_value=(True, "Service healthy"))
    service.get_status = AsyncMock()
    service.search = AsyncMock()
    service.repository = AsyncMock()
    service.__aenter__ = AsyncMock(return_value=service)
    service.__aexit__ = AsyncMock(return_value=None)
    return service


def test_create_app_basic():
    """Test basic app creation."""
    from osprey.interfaces.ariel import create_app

    with patch("osprey.interfaces.ariel.app.load_ariel_config") as mock_load:
        mock_load.return_value = {}

        # Mock the service creation to avoid actual initialization
        with patch("osprey.services.ariel_search.create_ariel_service"):
            # Don't actually run the lifespan - just check app creation
            app = create_app()

            assert isinstance(app, FastAPI)
            assert app.title == "ARIEL Search Interface"
            assert app.version == "1.0.0"


def test_load_ariel_config_from_file(tmp_path: Path, mock_ariel_config):
    """Test loading config from explicit path."""
    from osprey.interfaces.ariel.app import load_ariel_config

    # Create temp config file
    config_file = tmp_path / "config.yml"
    import yaml

    config_file.write_text(yaml.dump({"ariel": mock_ariel_config}))

    # Load config
    result = load_ariel_config(config_file)

    assert result == mock_ariel_config


def test_load_ariel_config_not_found(tmp_path: Path, monkeypatch):
    """Test config loading fails when file not found."""
    from osprey.interfaces.ariel.app import load_ariel_config

    # Ensure fallback paths (CWD config.yml, /app/config.yml) don't match
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CONFIG_FILE", raising=False)

    nonexistent = tmp_path / "nonexistent" / "config.yml"

    with pytest.raises(RuntimeError, match="No config.yml found"):
        load_ariel_config(nonexistent)


def test_load_ariel_config_db_host_override(tmp_path: Path, mock_ariel_config, monkeypatch):
    """Test database host override via environment variable."""
    from osprey.interfaces.ariel.app import load_ariel_config

    # Set environment variable
    monkeypatch.setenv("ARIEL_DATABASE_HOST", "postgresql-service")

    # Create temp config file
    config_file = tmp_path / "config.yml"
    import yaml

    config_file.write_text(yaml.dump({"ariel": mock_ariel_config}))

    # Load config
    result = load_ariel_config(config_file)

    # Check that localhost was replaced
    assert "postgresql-service" in result["database"]["uri"]
    assert "localhost" not in result["database"]["uri"]


def test_static_dir_exists():
    """Test that static directory exists."""
    from osprey.interfaces.ariel.app import STATIC_DIR

    assert STATIC_DIR.exists()
    assert STATIC_DIR.is_dir()
    # Check for expected files
    assert (STATIC_DIR / "index.html").exists()
    assert (STATIC_DIR / "css").is_dir()
    assert (STATIC_DIR / "js").is_dir()


@pytest.mark.asyncio
async def test_app_with_lifespan(mock_ariel_config, mock_ariel_service, tmp_path: Path):
    """Test app with lifespan events."""
    from osprey.interfaces.ariel import create_app

    # Create temp config
    config_file = tmp_path / "config.yml"
    import yaml

    config_file.write_text(yaml.dump({"ariel": mock_ariel_config}))

    with patch("osprey.services.ariel_search.create_ariel_service") as mock_create:
        # Mock ARIELConfig
        mock_config_class = MagicMock()
        mock_config_instance = MagicMock()
        mock_config_instance.validate.return_value = []  # No errors
        mock_config_class.from_dict.return_value = mock_config_instance

        with patch("osprey.services.ariel_search.ARIELConfig", mock_config_class):
            mock_create.return_value = mock_ariel_service

            app = create_app(config_file)

            # Use test client to trigger lifespan
            with TestClient(app) as client:
                # Check health endpoint
                response = client.get("/health")
                assert response.status_code == 200
                assert "status" in response.json()

                # Check root endpoint
                response = client.get("/")
                assert response.status_code == 200


def test_run_web_basic():
    """Test run_web function."""
    from osprey.interfaces.ariel import run_web

    with patch("uvicorn.run") as mock_run:
        run_web(host="0.0.0.0", port=8080, reload=False)

        # Check uvicorn.run was called
        mock_run.assert_called_once()
        # In non-reload mode, app instance is passed directly
        assert not isinstance(mock_run.call_args.args[0], str)


def test_run_web_reload_mode():
    """Test run_web in reload mode."""
    from osprey.interfaces.ariel import run_web

    with patch("uvicorn.run") as mock_run:
        run_web(host="127.0.0.1", port=8085, reload=True)

        # Check uvicorn.run was called with string reference
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["factory"] is True
        assert call_kwargs["reload"] is True
        assert call_kwargs["host"] == "127.0.0.1"
        assert call_kwargs["port"] == 8085
