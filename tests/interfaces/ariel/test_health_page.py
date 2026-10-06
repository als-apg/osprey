"""The ARIEL panel's open ``/health`` page.

``/health`` is the one panel route the sign-in gate leaves open, so it is what
a probe holding no operator secret can read. It carries the safe status facts
— entry count, last ingestion, enabled modules — and nothing that names the
store's address or login. Every test drives the real app with the gate
installed and the suite's credential seam switched off, and the first one
proves the gate is really there by being refused at ``/api/status``.
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from osprey.interfaces.ariel.app import (
    HEALTH_NO_SERVICE,
    HEALTH_OK,
    HEALTH_STORE_NOT_ANSWERING,
)
from tests.interfaces.ariel._health_app import (
    STORE_HOST,
    STORE_PASSWORD,
    StoreDouble,
    ariel_app,
)

pytestmark = pytest.mark.no_auth_seam

_FACT_KEYS = {
    "entry_count",
    "last_ingestion",
    "enabled_search_modules",
    "enabled_enhancement_modules",
}


async def _get(app, path: str) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://ariel.test") as client:
        return await client.get(path)


async def test_health_answers_without_a_credential_with_the_status_facts(tmp_path) -> None:
    ingested = datetime(2026, 9, 30, 8, 0, tzinfo=UTC)
    async with ariel_app(tmp_path, StoreDouble(last_ingestion=ingested)) as app:
        assert (await _get(app, "/api/status")).status_code == 401
        resp = await _get(app, "/health")

    assert resp.status_code == 200
    assert resp.json() == {
        "status": "healthy",
        "message": HEALTH_OK,
        "config_status": "ok",
        "service": {
            "entry_count": 48291,
            "last_ingestion": "2026-09-30T08:00:00Z",
            "enabled_search_modules": ["keyword"],
            "enabled_enhancement_modules": ["text_embedding"],
        },
    }


@pytest.mark.parametrize(
    "store",
    [
        pytest.param(StoreDouble(last_ingestion=datetime.now(UTC)), id="healthy"),
        pytest.param(StoreDouble(failing=True), id="store-not-answering"),
        pytest.param(None, id="no-service"),
    ],
)
async def test_health_carries_no_store_address_or_credential(tmp_path, store) -> None:
    async with ariel_app(tmp_path, store) as app:
        resp = await _get(app, "/health")

    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"status", "message", "config_status", "service"}
    assert body["service"] is None or set(body["service"]) == _FACT_KEYS
    for secret in (STORE_PASSWORD, STORE_HOST, "postgresql://", "database_uri", "errors"):
        assert secret not in resp.text


async def test_a_store_that_does_not_answer_reads_degraded_without_its_error(tmp_path) -> None:
    async with ariel_app(tmp_path, StoreDouble(failing=True)) as app:
        body = (await _get(app, "/health")).json()

    assert body["status"] == "degraded"
    assert body["message"] == HEALTH_STORE_NOT_ANSWERING
    assert body["service"] is None


async def test_a_panel_without_its_service_reads_degraded(tmp_path) -> None:
    async with ariel_app(tmp_path, None) as app:
        body = (await _get(app, "/health")).json()

    assert body["status"] == "degraded"
    assert body["message"] == HEALTH_NO_SERVICE
    assert body["service"] is None


async def test_a_store_never_ingested_reports_no_last_ingestion(tmp_path) -> None:
    async with ariel_app(tmp_path, StoreDouble(last_ingestion=None)) as app:
        body = (await _get(app, "/health")).json()

    assert body["status"] == "healthy"
    assert body["service"]["last_ingestion"] is None
