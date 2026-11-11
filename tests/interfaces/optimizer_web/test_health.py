"""Unit tests for the optimizer-web sidecar app shell: its healthcheck, the
rendered panel index, the service URL it relays to, and an import graph free
of the optimizer package."""

from __future__ import annotations

import subprocess
import sys

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


def test_the_panel_index_is_rendered_at_optimizer(client: TestClient) -> None:
    response = client.get("/optimizer")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert 'id="optimizer-root"' in response.text
    assert "{{" not in response.text


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
