"""Unit tests for the bluesky panels sidecar app skeleton (task 1.1).

Exercises the assembled FastAPI app (`bluesky_panels/app.py`) rather than its
helpers directly: the healthcheck route, the three panel static mounts, and
import-cleanliness (no `bluesky`/`ophyd`/`tiled` at module scope).
"""

from __future__ import annotations

import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.bluesky_panels.app import _PANEL_MOUNTS, _PANELS_ROOT, app

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


def test_every_mounted_panel_ships_a_bundle() -> None:
    """A registered mount is not evidence the panel exists.

    ``app.py`` runs ``os.makedirs(panel_dir, exist_ok=True)`` before mounting,
    so a panel whose bundle never made it into the tree still mounts cleanly —
    against an empty directory it just fabricated — and serves 404s to
    operators while the nav still links to it. The mount-registration check
    above cannot see that, so assert the bundle's real entry file instead of
    the directory (the directory always exists by the time this imports).

    This is a source-tree guard, not a style check: the results panel was
    silently dropped this way once already, when an unanchored ``results/``
    rule in .gitignore re-swallowed it at its new path during a package move
    and git staged the deletions with no matching additions.
    """
    for panel_name in _PANEL_MOUNTS:
        entry = _PANELS_ROOT / panel_name / "index.html"
        assert entry.is_file(), (
            f"panel {panel_name!r} is mounted at {_PANEL_MOUNTS[panel_name]} but ships no "
            f"bundle entry at {entry} — either the bundle is missing from the tree or "
            f"an ignore rule is keeping it untracked (check `git check-ignore -v`)"
        )


def test_every_mounted_panel_actually_serves(client: TestClient) -> None:
    """The operator-visible half of the guard above: each mount must return a
    real HTML shell, not the 404 an empty fabricated bundle directory yields."""
    for panel_name, mount_path in _PANEL_MOUNTS.items():
        response = client.get(f"{mount_path}/")
        assert response.status_code == 200, (
            f"panel {panel_name!r} at {mount_path}/ returned {response.status_code}"
        )
        assert "text/html" in response.headers["content-type"], panel_name


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
    # proving bluesky_panels.app's OWN import graph is clean requires isolation.
    check = (
        "import sys, osprey.interfaces.bluesky_panels.app; "
        f"leaked=[m for m in {_HEAVY_MODULES!r} if m in sys.modules]; "
        "sys.exit('leaked: ' + ','.join(leaked) if leaked else 0)"
    )
    result = subprocess.run([sys.executable, "-c", check], capture_output=True, text=True)
    assert result.returncode == 0, (
        "osprey.interfaces.bluesky_panels.app must not import a heavy control-system "
        f"dependency at module scope: {result.stdout}{result.stderr}"
    )
