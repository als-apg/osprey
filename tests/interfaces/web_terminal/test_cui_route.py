"""Tests for the /api/cui-server route."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.app import create_app


@pytest.fixture
def _mock_web_config():
    with patch(
        "osprey.interfaces.web_terminal.app._load_web_config",
        return_value={},
    ):
        yield


@pytest.fixture
def client_with_cui(_mock_web_config):
    with TestClient(create_app(shell_command="echo")) as c:
        c.app.state.cui_server_url = "http://127.0.0.1:3001"
        yield c


@pytest.fixture
def client_without_cui(_mock_web_config):
    with patch(
        "osprey.interfaces.web_terminal.app._launch_cui_server",
    ) as mock_launch:
        mock_launch.side_effect = lambda app: setattr(app.state, "cui_server_url", None)
        with TestClient(create_app(shell_command="echo")) as c:
            yield c


def _make_mock_response(status=200, body=None):
    """Create a mock urllib response."""
    resp = MagicMock()
    resp.status = status
    resp.read.return_value = json.dumps(body or {}).encode()
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestCUIServerRoute:
    def test_cui_server_config_available(self, client_with_cui):
        health_resp = _make_mock_response(200)
        config_resp = _make_mock_response(200, {"authToken": "abc123"})

        with patch("urllib.request.urlopen", side_effect=[health_resp, config_resp]):
            resp = client_with_cui.get("/api/cui-server")
        assert resp.status_code == 200
        data = resp.json()
        assert data["url"] == "http://127.0.0.1:3001"
        assert data["available"] is True
        assert data["authToken"] == "abc123"

    def test_cui_server_config_not_available(self, client_without_cui):
        resp = client_without_cui.get("/api/cui-server")
        assert resp.status_code == 200
        data = resp.json()
        assert data["url"] is None
        assert data["available"] is False
        assert data["authToken"] is None

    def test_cui_server_unhealthy(self, client_with_cui):
        """URL is set but CUI server is not responding."""
        with patch("urllib.request.urlopen", side_effect=ConnectionRefusedError):
            resp = client_with_cui.get("/api/cui-server")
        assert resp.status_code == 200
        data = resp.json()
        assert data["url"] == "http://127.0.0.1:3001"
        assert data["available"] is False
        assert data["authToken"] is None
