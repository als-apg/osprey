"""Tests for the core ``ariel`` health category.

The probe reads the panel's open ``/health`` page, so every test here runs it
against the real ARIEL app — its lifespan, its search service and the sign-in
gate ``configure_interface_app`` installs — with only the database stood in
(:mod:`tests.interfaces.ariel._health_app`). The module opts out of the suite's
credential seam: the probe holds no operator secret in a deployment, and the
first test proves the gate is installed by reading ``/api/status`` refused.

The presence gate and the address derivation need no panel; the answer shapes
the panel never produces (a non-200, a non-JSON body, an unparseable
timestamp) are served by a small stand-in ASGI app.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route

from osprey.health.core.ariel import ariel
from osprey.health.models import CheckResult, Status
from osprey.interfaces._serving import free_port
from osprey.interfaces.ariel.app import HEALTH_NO_SERVICE, HEALTH_STORE_NOT_ANSWERING
from osprey.port_layout import default_port
from osprey.services.ariel_search.config import IngestionConfig, WatchConfig
from tests.interfaces.ariel._health_app import (
    ARIEL_SECTION,
    StoreDouble,
    ariel_app,
)

pytestmark = pytest.mark.no_auth_seam


class _Recording(httpx.AsyncBaseTransport):
    """Pass every request to ``inner`` and keep the URL it was sent to."""

    def __init__(self, inner: httpx.AsyncBaseTransport) -> None:
        self.inner = inner
        self.urls: list[str] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.urls.append(str(request.url))
        return await self.inner.handle_async_request(request)


async def _run(config, *, transport=None) -> dict[str, CheckResult]:
    results = await ariel(config, transport=transport)()
    assert isinstance(results, list)
    return {r.name: r for r in results}


def _cfg(
    *, web: dict | None = None, deployment: dict | None = None, ingestion: dict | None = None
) -> dict:
    """A config with a non-empty top-level ``ariel`` block (the presence gate)."""
    ariel_block: dict = {"database": {"uri": "postgresql://ariel@localhost/ariel"}}
    if web is not None:
        ariel_block["web"] = web
    if ingestion is not None:
        ariel_block["ingestion"] = ingestion
    cfg: dict = {"ariel": ariel_block}
    if deployment is not None:
        cfg["deployment"] = deployment
    return cfg


@asynccontextmanager
async def _panel(tmp_path, store, **kwargs):
    """The real panel over ``store``, as a transport the probe can be handed."""
    async with ariel_app(tmp_path, store, **kwargs) as app:
        yield httpx.ASGITransport(app=app)


async def _probe(tmp_path, store=None, *, config=None, **kwargs) -> dict[str, CheckResult]:
    store = (
        StoreDouble(last_ingestion=datetime.now(UTC) - timedelta(hours=2))
        if store is None
        else store
    )
    async with _panel(tmp_path, store, **kwargs) as transport:
        return await _run(_cfg() if config is None else config, transport=transport)


def _stand_in(response: Response) -> httpx.ASGITransport:
    """A panel stand-in whose ``/health`` answers ``response``."""

    async def health(_request) -> Response:
        return response

    return httpx.ASGITransport(app=Starlette(routes=[Route("/health", health)]))


# --------------------------------------------------------------------------- #
# No credential
# --------------------------------------------------------------------------- #


async def test_the_probe_reads_the_open_page_of_a_gated_panel(tmp_path) -> None:
    async with _panel(tmp_path, StoreDouble(last_ingestion=datetime.now(UTC))) as transport:
        async with httpx.AsyncClient(transport=transport, base_url="http://ariel.test") as client:
            assert (await client.get("/api/status")).status_code == 401
        recording = _Recording(transport)
        by_name = await _run(_cfg(), transport=recording)

    assert [url.rsplit("/", 1)[1] for url in recording.urls] == ["health"]
    assert by_name["ariel_status"].status is Status.OK
    assert by_name["ariel_entries"].value == "48,291 entries"


# --------------------------------------------------------------------------- #
# Presence gate
# --------------------------------------------------------------------------- #


async def test_no_rows_when_no_ariel_block() -> None:
    by_name = await _run({"deployment": {"bind_address": "127.0.0.1"}})
    assert by_name == {}


async def test_no_rows_when_ariel_block_empty() -> None:
    by_name = await _run({"ariel": {}})
    assert by_name == {}


async def test_no_rows_when_config_none() -> None:
    by_name = await _run(None)
    assert by_name == {}


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


async def test_configured_emits_all_rows(tmp_path) -> None:
    by_name = await _probe(tmp_path)
    assert set(by_name) == {
        "ariel_status",
        "ariel_entries",
        "ariel_last_ingestion",
        "ariel_search_modules",
        "ariel_enhancement_modules",
    }
    assert all(r.category == "ariel" for r in by_name.values())


async def test_status_ok_and_has_latency(tmp_path) -> None:
    row = (await _probe(tmp_path))["ariel_status"]
    assert row.status is Status.OK
    assert "reachable" in row.message
    assert row.latency_ms >= 0.0


async def test_entries_value_formatted(tmp_path) -> None:
    row = (await _probe(tmp_path))["ariel_entries"]
    assert row.status is Status.OK
    assert row.value == "48,291 entries"


async def test_last_ingestion_reports_age(tmp_path) -> None:
    row = (await _probe(tmp_path))["ariel_last_ingestion"]
    assert row.status is Status.OK
    assert row.value == "2 h ago"


async def test_module_rows_list_names(tmp_path) -> None:
    by_name = await _probe(tmp_path)
    search = by_name["ariel_search_modules"]
    assert search.status is Status.OK
    assert "1 search module(s)" in search.message
    assert search.value == "keyword"
    enh = by_name["ariel_enhancement_modules"]
    assert enh.status is Status.OK
    assert enh.value == "text_embedding"


# --------------------------------------------------------------------------- #
# Last-ingestion staleness threshold
# --------------------------------------------------------------------------- #

#: An ``ariel.ingestion`` block with an explicit 30-minute threshold
#: (``poll_interval_seconds`` + ``watch.max_interval_seconds``).
_INGESTION_30_MIN = {
    "adapter": "generic_json",
    "poll_interval_seconds": 600,
    "watch": {"max_interval_seconds": 1200},
}


def _ingested(delta: timedelta) -> StoreDouble:
    """A store whose last ingestion is ``delta`` in the past."""
    return StoreDouble(last_ingestion=datetime.now(UTC) - delta)


async def test_fresh_ingestion_is_ok_under_threshold(tmp_path) -> None:
    cfg = _cfg(ingestion=_INGESTION_30_MIN)
    row = (await _probe(tmp_path, _ingested(timedelta(minutes=5)), config=cfg))[
        "ariel_last_ingestion"
    ]
    assert row.status is Status.OK
    assert row.value == "5 m ago"


async def test_stale_ingestion_warns_with_age_and_threshold(tmp_path) -> None:
    cfg = _cfg(ingestion=_INGESTION_30_MIN)
    row = (await _probe(tmp_path, _ingested(timedelta(hours=6)), config=cfg))[
        "ariel_last_ingestion"
    ]
    assert row.status is Status.WARNING
    assert "6 h" in row.message
    assert "30 m" in row.message
    assert row.value == "6 h ago"


async def test_threshold_falls_back_to_dataclass_defaults(tmp_path) -> None:
    """Absent keys take :class:`IngestionConfig`/:class:`WatchConfig` defaults."""
    cfg = _cfg(ingestion={"adapter": "generic_json"})
    default_threshold = IngestionConfig.poll_interval_seconds + WatchConfig.max_interval_seconds

    stale = _ingested(timedelta(seconds=default_threshold + 3600))
    fresh = _ingested(timedelta(seconds=default_threshold - 3600))
    (tmp_path / "stale").mkdir()
    (tmp_path / "fresh").mkdir()
    assert (await _probe(tmp_path / "stale", stale, config=cfg))[
        "ariel_last_ingestion"
    ].status is Status.WARNING
    assert (await _probe(tmp_path / "fresh", fresh, config=cfg))[
        "ariel_last_ingestion"
    ].status is Status.OK


async def test_no_ingestion_block_never_warns_on_age(tmp_path) -> None:
    row = (await _probe(tmp_path, _ingested(timedelta(days=30))))["ariel_last_ingestion"]
    assert row.status is Status.OK
    assert row.value == "30 d ago"


# --------------------------------------------------------------------------- #
# Degradation
# --------------------------------------------------------------------------- #


async def test_configured_but_unreachable_emits_single_warning(monkeypatch) -> None:
    monkeypatch.delenv("OSPREY_ARIEL_PORT", raising=False)
    # A port just released by the OS has no listener, so the connect is refused.
    by_name = await _run(_cfg(web={"host": "127.0.0.1", "port": free_port()}))
    assert set(by_name) == {"ariel_status"}
    row = by_name["ariel_status"]
    assert row.status is Status.WARNING
    assert "unreachable" in row.message
    assert "osprey web" in row.details


async def test_a_store_that_does_not_answer_is_the_status_row_alone(tmp_path) -> None:
    by_name = await _probe(tmp_path, StoreDouble(failing=True))
    assert set(by_name) == {"ariel_status"}
    row = by_name["ariel_status"]
    assert row.status is Status.WARNING
    assert row.details == HEALTH_STORE_NOT_ANSWERING


async def test_a_panel_without_its_service_is_the_status_row_alone(tmp_path) -> None:
    async with ariel_app(tmp_path, None) as app:
        by_name = await _run(_cfg(), transport=httpx.ASGITransport(app=app))
    assert set(by_name) == {"ariel_status"}
    assert by_name["ariel_status"].status is Status.WARNING
    assert by_name["ariel_status"].details == HEALTH_NO_SERVICE


async def test_zero_entries_warns(tmp_path) -> None:
    store = StoreDouble(entry_count=0, last_ingestion=datetime.now(UTC))
    assert (await _probe(tmp_path, store))["ariel_entries"].status is Status.WARNING


async def test_missing_last_ingestion_warns(tmp_path) -> None:
    store = StoreDouble(last_ingestion=None)
    assert (await _probe(tmp_path, store))["ariel_last_ingestion"].status is Status.WARNING


async def test_empty_search_modules_warns(tmp_path) -> None:
    section = {**ARIEL_SECTION, "search_modules": {"keyword": {"enabled": False}}}
    by_name = await _probe(tmp_path, section=section)
    assert by_name["ariel_search_modules"].status is Status.WARNING


async def test_empty_enhancement_modules_is_ok(tmp_path) -> None:
    section = {k: v for k, v in ARIEL_SECTION.items() if k != "enhancement_modules"}
    by_name = await _probe(tmp_path, section=section)
    assert by_name["ariel_enhancement_modules"].status is Status.OK


# --------------------------------------------------------------------------- #
# Answers the panel never gives
# --------------------------------------------------------------------------- #


async def test_non_200_emits_single_warning() -> None:
    by_name = await _run(_cfg(), transport=_stand_in(Response(status_code=503)))
    assert set(by_name) == {"ariel_status"}
    assert by_name["ariel_status"].status is Status.WARNING
    assert "503" in by_name["ariel_status"].message


async def test_non_json_body_emits_single_warning() -> None:
    by_name = await _run(_cfg(), transport=_stand_in(PlainTextResponse("not json")))
    assert set(by_name) == {"ariel_status"}
    assert by_name["ariel_status"].status is Status.WARNING


async def test_unparseable_timestamp_stays_ok_with_ingestion_block() -> None:
    body = {
        "status": "healthy",
        "message": "ARIEL service healthy",
        "config_status": "ok",
        "service": {
            "entry_count": 1,
            "last_ingestion": "whenever",
            "enabled_search_modules": ["keyword"],
            "enabled_enhancement_modules": [],
        },
    }
    cfg = _cfg(ingestion=_INGESTION_30_MIN)
    row = (await _run(cfg, transport=_stand_in(JSONResponse(body))))["ariel_last_ingestion"]
    assert row.status is Status.OK
    assert row.value == "whenever"


# --------------------------------------------------------------------------- #
# Endpoint construction
# --------------------------------------------------------------------------- #


async def _probed_url(config) -> list[str]:
    recording = _Recording(_stand_in(Response(status_code=503)))
    await _run(config, transport=recording)
    return recording.urls


async def test_health_url_uses_the_panels_host_and_port(monkeypatch) -> None:
    """The probe targets ``ariel.web.host``/``port`` — what the panel binds."""
    monkeypatch.delenv("OSPREY_ARIEL_PORT", raising=False)
    assert await _probed_url(_cfg(web={"host": "10.0.0.5", "port": 9999})) == [
        "http://10.0.0.5:9999/health"
    ]


async def test_health_url_defaults(monkeypatch) -> None:
    monkeypatch.delenv("OSPREY_ARIEL_PORT", raising=False)
    assert await _probed_url(_cfg()) == [f"http://127.0.0.1:{default_port('ariel')}/health"]


async def test_health_url_honours_the_multi_user_port_override(monkeypatch) -> None:
    """``OSPREY_ARIEL_PORT`` — exported per user by the multi-user compose
    render because the per-user containers share the host network namespace —
    is the port the panel binds, so it is the port the probe knocks on."""
    monkeypatch.setenv("OSPREY_ARIEL_PORT", "10301")
    assert await _probed_url(_cfg(web={"port": 9999})) == ["http://127.0.0.1:10301/health"]


async def test_misplaced_address_key_is_reported_not_probed() -> None:
    """``ariel.port`` (correct: ``ariel.web.port``) is a warning naming the key,
    with no probe issued — the same verdict the ``web_panels`` category gives."""
    config = _cfg()
    config["ariel"]["port"] = 9999
    recording = _Recording(_stand_in(Response(status_code=503)))
    by_name = await _run(config, transport=recording)
    assert recording.urls == []
    assert set(by_name) == {"ariel_status"}
    assert by_name["ariel_status"].status is Status.WARNING
    assert "ariel.web.port" in by_name["ariel_status"].details
