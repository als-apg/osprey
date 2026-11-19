"""Unit tests for the optimizer-web sidecar app shell: its healthcheck, the
panel index served as a static file, the service URL it relays to, and an
import graph free of the optimizer package."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.optimizer_web.app import app, resolve_service_url


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def test_health_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


_PANEL_DIR = Path(__file__).parents[3] / "src/osprey/interfaces/optimizer_web/panels/optimizer"


@pytest.mark.parametrize("path", ["/optimizer/", "/optimizer/index.html"])
def test_the_panel_index_is_served_unchanged(client: TestClient, path: str) -> None:
    response = client.get(path, follow_redirects=False)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.content == (_PANEL_DIR / "index.html").read_bytes()


def test_the_bare_panel_path_redirects_to_the_bundle_root(client: TestClient) -> None:
    """The page's ``css/`` and ``js/`` references are relative, so the bundle is
    only addressable with the trailing slash."""
    response = client.get("/optimizer", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"].endswith("/optimizer/")


def test_the_panel_files_are_served_with_their_pinned_types(client: TestClient) -> None:
    response = client.get("/optimizer/css/shell.css")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/css")
    assert response.content == (_PANEL_DIR / "css/shell.css").read_bytes()


def test_everything_it_serves_is_uncached(client: TestClient) -> None:
    response = client.get("/health")

    assert response.headers["cache-control"] == "no-cache, no-store, must-revalidate"


def test_the_service_url_is_the_compose_set_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPTIMIZER_URL", "http://optimizer:8082/")

    assert resolve_service_url() == "http://optimizer:8082"


def test_an_unreadable_config_still_resolves_a_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.delenv("OPTIMIZER_URL", raising=False)
    monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "does-not-exist.yml"))

    assert resolve_service_url().startswith("http://127.0.0.1:")


def test_import_is_clean_of_the_optimizer_package() -> None:
    """The sidecar relays over HTTP; it must not drag the optimizer's search
    stack (or the package at all) into its image."""
    check = (
        "import sys, osprey.interfaces.optimizer_web.app; "
        "leaked=[m for m in ('osprey_optimizer', 'xopt', 'torch') if m in sys.modules]; "
        "sys.exit('leaked: ' + ','.join(leaked) if leaked else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", check], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, f"{result.stdout}{result.stderr}"
