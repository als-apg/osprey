"""Tests for the core ``ariel`` health category.

Drives the category's async ``/api/status`` probe through an injected
:class:`httpx.MockTransport`, exercising the presence gate (a top-level ``ariel``
config block), endpoint construction from ``bind_address`` + ``ariel.web.port``,
and every derived row (reachability, entry count, last-ingestion age, and the
search/enhancement module rows).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import httpx

from osprey.health.core.ariel import ariel
from osprey.health.models import CheckResult, Status


async def _run(config, *, transport=None) -> dict[str, CheckResult]:
    results = await ariel(config, transport=transport)()
    assert isinstance(results, list)
    return {r.name: r for r in results}


def _cfg(*, web: dict | None = None, deployment: dict | None = None) -> dict:
    """A config with a non-empty top-level ``ariel`` block (the presence gate)."""
    ariel_block: dict = {"database": {"uri": "postgresql://ariel@localhost/ariel"}}
    if web is not None:
        ariel_block["web"] = web
    cfg: dict = {"ariel": ariel_block}
    if deployment is not None:
        cfg["deployment"] = deployment
    return cfg


def _status_payload(**overrides) -> dict:
    payload = {
        "healthy": True,
        "database_connected": True,
        "database_uri": "postgresql://ariel@localhost/ariel",
        "entry_count": 48291,
        "enabled_search_modules": ["keyword", "semantic"],
        "enabled_enhancement_modules": ["text_embedding"],
        "last_ingestion": (datetime.now() - timedelta(hours=2)).isoformat(),
        "errors": [],
    }
    payload.update(overrides)
    return payload


def _ok_transport(payload: dict | None = None, captured: list[str] | None = None):
    body = _status_payload() if payload is None else payload

    def handler(req: httpx.Request) -> httpx.Response:
        if captured is not None:
            captured.append(str(req.url))
        return httpx.Response(200, json=body)

    return httpx.MockTransport(handler)


# --------------------------------------------------------------------------- #
# Presence gate
# --------------------------------------------------------------------------- #


async def test_no_rows_when_no_ariel_block() -> None:
    by_name = await _run({"deployment": {"bind_address": "127.0.0.1"}}, transport=_ok_transport())
    assert by_name == {}


async def test_no_rows_when_ariel_block_empty() -> None:
    by_name = await _run({"ariel": {}}, transport=_ok_transport())
    assert by_name == {}


async def test_no_rows_when_config_none() -> None:
    by_name = await _run(None, transport=_ok_transport())
    assert by_name == {}


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


async def test_configured_emits_all_rows() -> None:
    by_name = await _run(_cfg(), transport=_ok_transport())
    assert set(by_name) == {
        "ariel_status",
        "ariel_entries",
        "ariel_last_ingestion",
        "ariel_search_modules",
        "ariel_enhancement_modules",
    }
    assert all(r.category == "ariel" for r in by_name.values())


async def test_status_ok_and_has_latency() -> None:
    row = (await _run(_cfg(), transport=_ok_transport()))["ariel_status"]
    assert row.status is Status.OK
    assert "reachable" in row.message
    assert row.latency_ms >= 0.0


async def test_entries_value_formatted() -> None:
    row = (await _run(_cfg(), transport=_ok_transport()))["ariel_entries"]
    assert row.status is Status.OK
    assert row.value == "48,291 entries"


async def test_last_ingestion_reports_age() -> None:
    row = (await _run(_cfg(), transport=_ok_transport()))["ariel_last_ingestion"]
    assert row.status is Status.OK
    assert row.value.endswith("ago")


async def test_module_rows_list_names() -> None:
    by_name = await _run(_cfg(), transport=_ok_transport())
    search = by_name["ariel_search_modules"]
    assert search.status is Status.OK
    assert "2 search module(s)" in search.message
    assert search.value == "keyword, semantic"
    enh = by_name["ariel_enhancement_modules"]
    assert enh.status is Status.OK
    assert enh.value == "text_embedding"


# --------------------------------------------------------------------------- #
# Degradation
# --------------------------------------------------------------------------- #


async def test_configured_but_unreachable_emits_single_warning() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused", request=req)

    by_name = await _run(_cfg(), transport=httpx.MockTransport(handler))
    assert set(by_name) == {"ariel_status"}
    row = by_name["ariel_status"]
    assert row.status is Status.WARNING
    assert "unreachable" in row.message
    assert "osprey web" in row.details


async def test_non_200_emits_single_warning() -> None:
    transport = httpx.MockTransport(lambda req: httpx.Response(503))
    by_name = await _run(_cfg(), transport=transport)
    assert set(by_name) == {"ariel_status"}
    assert by_name["ariel_status"].status is Status.WARNING
    assert "503" in by_name["ariel_status"].message


async def test_non_json_body_emits_single_warning() -> None:
    transport = httpx.MockTransport(lambda req: httpx.Response(200, text="not json"))
    by_name = await _run(_cfg(), transport=transport)
    assert set(by_name) == {"ariel_status"}
    assert by_name["ariel_status"].status is Status.WARNING


async def test_unhealthy_warns_but_still_derives_rows() -> None:
    payload = _status_payload(healthy=False, errors=["db pool exhausted"])
    by_name = await _run(_cfg(), transport=_ok_transport(payload))
    status = by_name["ariel_status"]
    assert status.status is Status.WARNING
    assert "db pool exhausted" in status.details
    # The other rows are still derived from the same payload.
    assert by_name["ariel_entries"].status is Status.OK


async def test_zero_entries_warns() -> None:
    by_name = await _run(_cfg(), transport=_ok_transport(_status_payload(entry_count=0)))
    assert by_name["ariel_entries"].status is Status.WARNING


async def test_missing_entry_count_warns() -> None:
    by_name = await _run(_cfg(), transport=_ok_transport(_status_payload(entry_count=None)))
    assert by_name["ariel_entries"].status is Status.WARNING


async def test_missing_last_ingestion_warns() -> None:
    by_name = await _run(_cfg(), transport=_ok_transport(_status_payload(last_ingestion=None)))
    assert by_name["ariel_last_ingestion"].status is Status.WARNING


async def test_empty_search_modules_warns() -> None:
    by_name = await _run(
        _cfg(), transport=_ok_transport(_status_payload(enabled_search_modules=[]))
    )
    assert by_name["ariel_search_modules"].status is Status.WARNING


async def test_empty_enhancement_modules_is_ok() -> None:
    by_name = await _run(
        _cfg(), transport=_ok_transport(_status_payload(enabled_enhancement_modules=[]))
    )
    assert by_name["ariel_enhancement_modules"].status is Status.OK


# --------------------------------------------------------------------------- #
# Endpoint construction
# --------------------------------------------------------------------------- #


async def test_status_url_uses_bind_and_port() -> None:
    config = _cfg(web={"port": 9999}, deployment={"bind_address": "10.0.0.5"})
    captured: list[str] = []
    await _run(config, transport=_ok_transport(captured=captured))
    assert captured == ["http://10.0.0.5:9999/api/status"]


async def test_status_url_defaults() -> None:
    captured: list[str] = []
    await _run(_cfg(), transport=_ok_transport(captured=captured))
    assert captured == ["http://127.0.0.1:8085/api/status"]
