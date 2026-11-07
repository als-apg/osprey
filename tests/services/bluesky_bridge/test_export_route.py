"""Tests for the run-export route's refusals that need no Tiled catalog.

`GET /runs/{id}/export` reads the catalog only; a deployment with no catalog
configured answers before any network read. These tests run with
`BLUESKY_TILED_URI` unset, so no container is involved. The catalog-backed
refusals are covered against a real server in
`tests/integration/test_tiled_export.py`.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from osprey.services.bluesky_bridge import app as app_module
from osprey.services.bluesky_bridge import queue_backend as qb

_TILED_URI_ENV = "BLUESKY_TILED_URI"
_TILED_API_KEY_ENV = "BLUESKY_TILED_API_KEY"


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(_TILED_URI_ENV, raising=False)
    monkeypatch.delenv(_TILED_API_KEY_ENV, raising=False)
    yield


@pytest.fixture
def from_uri_calls(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Every `tiled.client.from_uri` call the request makes."""
    import tiled.client

    calls: list[Any] = []

    def _record(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        raise AssertionError("no Tiled client may be built when no catalog is configured")

    monkeypatch.setattr(tiled.client, "from_uri", _record)
    return calls


def test_an_export_with_no_catalog_is_a_settled_refusal(from_uri_calls: list[Any]) -> None:
    response = TestClient(app_module.app).get("/runs/some-run/export")

    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail["code"] == qb.REASON_TILED_NOT_CONFIGURED
    assert detail["retryable"] is False
    assert "no tiled catalog is configured" in detail["detail"].lower()
    assert from_uri_calls == []


def test_an_unsupported_format_is_refused_before_the_catalog_check(
    from_uri_calls: list[Any],
) -> None:
    response = TestClient(app_module.app).get("/runs/some-run/export?format=bogus")

    assert response.status_code == 400, response.text
    assert response.json()["detail"]["code"] == qb.REASON_UNSUPPORTED_FORMAT
    assert response.json()["detail"]["retryable"] is False
    assert from_uri_calls == []
