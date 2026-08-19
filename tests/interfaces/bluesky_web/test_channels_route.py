"""Tests for the bluesky-web sidecar's channel-catalog route and the caching
contract it depends on.

The sidecar's blanket ``_no_cache`` middleware fills the header in with
``setdefault`` rather than overwriting it, so a route can make its own caching
decision (the channel catalog revalidates with an ETag) while everything else
keeps the full no-store string. Both halves of that contract are pinned here:
this module owns the caching guard for a route-level decision, and
``test_app_integration.py::test_every_response_forbids_browser_caching`` owns
the blanket guard across representative paths.

Exercises the package-level ``osprey.interfaces.bluesky_web.app:app`` -- the
object actually served in production -- via ``TestClient(app)`` entered as a
context manager so the real ``_lifespan`` runs, matching the house pattern in
``test_app_integration.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.bluesky_web.app import app

_NO_STORE = "no-cache, no-store, must-revalidate"


_NO_CACHE = "no-cache"
_CATALOG = b'["BR:DIAG:BPM:01:POSITION:X", "BR:DIAG:BPM:02:POSITION:X"]'


# Config isolation (OSPREY_CONFIG pointed at a nonexistent file, launch token
# cleared) comes from this directory's conftest autouse fixture.


@pytest.fixture
def project_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A stand-in for the container's ``/app/project`` mount point.

    The compose template hands the sidecar ``CONFIG_FILE=/app/project/config.yml``
    and mounts the catalog as that file's sibling, which is exactly what the
    route resolves against -- so pointing ``CONFIG_FILE`` at a temp config.yml
    reproduces the deployed layout. The root conftest's ``restore_environ``
    clears ``CONFIG_FILE`` around every test, so this never leaks sideways.
    """
    project = tmp_path / "project"
    project.mkdir()
    (project / "config.yml").write_text("connector:\n  type: mock\n")
    monkeypatch.setenv("CONFIG_FILE", str(project / "config.yml"))
    return project


# ---------------------------------------------------------------------------
# Caching contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/bluesky/", "/bluesky/index.html"])
def test_panel_bundle_response_still_forbids_browser_caching(path: str) -> None:
    """Regression guard for the ``setdefault`` middleware.

    Panel bundles ship unversioned filenames (``panel.js``, not
    ``panel-<hash>.js``), so a cached copy silently survives a container
    rebuild and the operator keeps seeing last week's UI. The static mount sets
    no ``Cache-Control`` of its own, so the middleware must still fill in the
    full no-store string -- ``setdefault`` may only yield to a route that made
    a deliberate caching decision.
    """
    with TestClient(app) as client:
        response = client.get(path)

    assert response.status_code == 200, path
    assert response.headers["cache-control"] == _NO_STORE, path


# ---------------------------------------------------------------------------
# GET /channels
# ---------------------------------------------------------------------------


def test_catalog_is_served_with_a_revalidating_etag(project_dir: Path) -> None:
    """The catalog comes back verbatim, cacheable only against its ETag.

    ``no-cache`` (not ``no-store``) is the point: the browser must ask every
    time, but the ETag lets the answer be an empty 304 for a catalog that only
    changes on a rebuild.
    """
    (project_dir / "channels.json").write_bytes(_CATALOG)

    with TestClient(app) as client:
        response = client.get("/channels")

    assert response.status_code == 200
    assert response.json() == ["BR:DIAG:BPM:01:POSITION:X", "BR:DIAG:BPM:02:POSITION:X"]
    assert response.headers["cache-control"] == _NO_CACHE
    assert response.headers["etag"].startswith('"')


def test_matching_if_none_match_gets_an_empty_304(project_dir: Path) -> None:
    (project_dir / "channels.json").write_bytes(_CATALOG)

    with TestClient(app) as client:
        first = client.get("/channels")
        etag = first.headers["etag"]
        second = client.get("/channels", headers={"If-None-Match": etag})

    assert second.status_code == 304
    assert second.content == b""
    assert second.headers["etag"] == etag
    assert second.headers["cache-control"] == _NO_CACHE


def test_rebuilt_catalog_invalidates_a_stale_etag(project_dir: Path) -> None:
    """A rebuild that changes the catalog must not be served from cache."""
    catalog = project_dir / "channels.json"
    catalog.write_bytes(_CATALOG)

    with TestClient(app) as client:
        stale_etag = client.get("/channels").headers["etag"]
        catalog.write_bytes(b'["SR:DIAG:BPM:07:POSITION:Y"]')
        response = client.get("/channels", headers={"If-None-Match": stale_etag})

    assert response.status_code == 200
    assert response.json() == ["SR:DIAG:BPM:07:POSITION:Y"]
    assert response.headers["etag"] != stale_etag


def test_absent_catalog_is_a_404(project_dir: Path) -> None:
    """An absent file is the "this build emitted none" signal, not an empty list.

    Panels treat the 404 exactly as they treat an empty catalog -- no
    autocomplete -- so the body shape carries no contract; only the status does.
    The response must still not be cached, or the 404 would outlive the rebuild
    that fixes it.
    """
    assert not (project_dir / "channels.json").exists()

    with TestClient(app) as client:
        response = client.get("/channels")

    assert response.status_code == 404
    assert response.headers["cache-control"] == _NO_STORE
