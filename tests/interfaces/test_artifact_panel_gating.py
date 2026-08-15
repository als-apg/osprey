"""``/api/artifact-server`` is what enables or disables the WORKSPACE tab.

The front end computes panel availability from this endpoint alone, so a URL
published for a server that was never started hands the operator an enabled tab
whose iframe returns a bare 502. These tests pin the endpoint's half of that
contract, end to end from config to JSON body.

The launch half — auto-launch gating, the resolved address, failure retraction —
is not artifact-specific: one launcher serves all six companion panels, and
``test_panel_launch_gating.py`` pins it parametrized over the whole registry.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.infrastructure import server_launcher
from osprey.interfaces.web_terminal.app import _launch_panel_server
from osprey.interfaces.web_terminal.routes import panels
from osprey.utils import workspace


@pytest.fixture
def config(monkeypatch) -> dict:
    """A mutable config seen by both the launcher and the web terminal."""
    cfg: dict = {}
    monkeypatch.setattr(server_launcher, "load_osprey_config", lambda: cfg)
    monkeypatch.setattr(workspace, "load_osprey_config", lambda: cfg)
    # The port resolver honours this override; an inherited value would decide
    # the port out from under the config these tests set.
    monkeypatch.delenv("OSPREY_ARTIFACT_SERVER_PORT", raising=False)
    return cfg


@pytest.fixture(autouse=True)
def ensure_artifact_server(monkeypatch) -> MagicMock:
    """Stub out the real launch so no server is started."""
    mock = MagicMock()
    monkeypatch.setattr(server_launcher, "ensure_web_server", mock)
    return mock


def _api() -> FastAPI:
    api = FastAPI()
    api.include_router(panels.router)
    return api


def test_reports_unavailable_when_auto_launch_is_false(config):
    config["artifact_server"] = {"auto_launch": False}
    api = _api()
    _launch_panel_server(api, "artifact")

    body = TestClient(api).get("/api/artifact-server").json()

    assert body == {"url": None, "available": False}


def test_reports_available_when_auto_launch_is_on(config):
    config["artifact_server"] = {"auto_launch": True}
    api = _api()
    _launch_panel_server(api, "artifact")

    body = TestClient(api).get("/api/artifact-server").json()

    assert body["available"] is True
    assert body["url"] == "/panel/artifacts"
