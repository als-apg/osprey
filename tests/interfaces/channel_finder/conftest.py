"""Fixtures for Channel Finder web interface tests."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def mock_config():
    """Return a minimal channel finder config dict."""
    return {
        "channel_finder": {
            "pipeline_mode": "in_context",
            "pipelines": {
                "in_context": {
                    "database": {"path": "/tmp/test_db.json", "type": "flat"},
                },
            },
        },
    }


@pytest.fixture()
def mock_registry():
    """Mock the in-context registry initialization."""
    mock_reg = MagicMock()
    mock_reg.database = MagicMock()
    mock_reg.facility_name = "TEST"

    with patch(
        "osprey.services.channel_finder.mcp.in_context.registry.initialize_cf_ic_registry",
        return_value=mock_reg,
    ) as init_mock:
        yield init_mock


@pytest.fixture()
def app(mock_config, mock_registry):
    """Create a test Channel Finder FastAPI app with mocked dependencies."""
    with patch(
        "osprey.mcp_server.common.load_osprey_config",
        return_value=mock_config,
    ):
        from osprey.interfaces.channel_finder.app import create_app

        application = create_app(project_cwd="/tmp/test-project")
        yield application


@pytest.fixture()
def client(app):
    """Create a TestClient for the channel finder app."""
    with TestClient(app) as c:
        yield c
