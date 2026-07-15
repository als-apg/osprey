"""Unit tests for the scan panels sidecar app skeleton (task 1.1).

Exercises the assembled FastAPI app (`scan_panels/app.py`) rather than its
helpers directly: the healthcheck route, the three panel static mounts, and
import-cleanliness (no `bluesky`/`ophyd`/`tiled` at module scope).
"""

from __future__ import annotations

import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from osprey.services.scan_panels.app import _PANEL_MOUNTS, app

_HEAVY_MODULES = ("bluesky", "ophyd", "tiled")


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def test_health_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_panel_mounts_registered() -> None:
    assert _PANEL_MOUNTS == {"plan": "/plan", "results": "/results", "health": "/health-panel"}

    mounted_paths = {route.path for route in app.routes if hasattr(route, "path")}
    for mount_path in _PANEL_MOUNTS.values():
        assert mount_path in mounted_paths


def test_panel_mounts_do_not_collide_with_healthcheck(client: TestClient) -> None:
    # The health-panel bundle is mounted at /health-panel, never /health --
    # a request to /health must always hit the JSON healthcheck route, not
    # a static file lookup.
    response = client.get("/health")
    assert response.json() == {"status": "ok"}
    assert "/health-panel" in {route.path for route in app.routes if hasattr(route, "path")}
    assert "/health" != "/health-panel"


@pytest.mark.parametrize("mount_path", ["/plan", "/results", "/health-panel"])
def test_panel_mount_responds_not_as_health_json(client: TestClient, mount_path: str) -> None:
    # The panel static mount must not shadow the /health JSON route. Depending
    # on whether the panel bundle has been authored yet, the mount responds
    # either 404 (empty directory) or 200 with the panel's own HTML -- never
    # the /health route's {"status": "ok"} JSON body.
    response = client.get(mount_path)
    assert response.status_code in (200, 404)
    if response.status_code == 200:
        assert response.headers.get("content-type", "").startswith("text/html")


def test_import_is_clean_of_heavy_control_system_deps() -> None:
    # Must run in a FRESH interpreter: an in-process ``sys.modules`` check is
    # polluted by any other test in the session that imported bluesky/ophyd/
    # tiled (e.g. the bluesky_bridge suite in a full ``pytest tests/`` run), so
    # proving scan_panels.app's OWN import graph is clean requires isolation.
    check = (
        "import sys, osprey.services.scan_panels.app; "
        f"leaked=[m for m in {_HEAVY_MODULES!r} if m in sys.modules]; "
        "sys.exit('leaked: ' + ','.join(leaked) if leaked else 0)"
    )
    result = subprocess.run([sys.executable, "-c", check], capture_output=True, text=True)
    assert result.returncode == 0, (
        "osprey.services.scan_panels.app must not import a heavy control-system "
        f"dependency at module scope: {result.stdout}{result.stderr}"
    )
