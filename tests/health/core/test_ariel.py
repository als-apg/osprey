"""Tests for the core ``ariel`` health category.

Drives the category's async ``/api/status`` probe through an injected
:class:`httpx.MockTransport`, exercising the presence gate (a top-level ``ariel``
config block), endpoint construction through the web-server registry resolver
(``ariel.web.host``/``port``, with the multi-user ``OSPREY_ARIEL_PORT`` override),
and every derived row (reachability, entry count, last-ingestion age, and the
search/enhancement module rows).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import httpx

from osprey.health.core.ariel import ariel
from osprey.health.models import CheckResult, Status
from osprey.port_layout import default_port


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


async def test_status_url_uses_the_panels_host_and_port(monkeypatch) -> None:
    """The probe targets ``ariel.web.host``/``port`` — what the panel binds."""
    monkeypatch.delenv("OSPREY_ARIEL_PORT", raising=False)
    config = _cfg(web={"host": "10.0.0.5", "port": 9999})
    captured: list[str] = []
    await _run(config, transport=_ok_transport(captured=captured))
    assert captured == ["http://10.0.0.5:9999/api/status"]


async def test_status_url_defaults(monkeypatch) -> None:
    monkeypatch.delenv("OSPREY_ARIEL_PORT", raising=False)
    captured: list[str] = []
    await _run(_cfg(), transport=_ok_transport(captured=captured))
    assert captured == [f"http://127.0.0.1:{default_port('ariel')}/api/status"]


async def test_status_url_honours_the_multi_user_port_override(monkeypatch) -> None:
    """``OSPREY_ARIEL_PORT`` — exported per user by the multi-user compose
    render because the per-user containers share the host network namespace —
    is the port the panel binds, so it is the port the probe knocks on."""
    monkeypatch.setenv("OSPREY_ARIEL_PORT", "10301")
    captured: list[str] = []
    await _run(_cfg(web={"port": 9999}), transport=_ok_transport(captured=captured))
    assert captured == ["http://127.0.0.1:10301/api/status"]


async def test_misplaced_address_key_is_reported_not_probed() -> None:
    """``ariel.port`` (correct: ``ariel.web.port``) is a warning naming the key,
    with no probe issued — the same verdict the ``web_panels`` category gives."""
    config = _cfg()
    config["ariel"]["port"] = 9999
    captured: list[str] = []
    by_name = await _run(config, transport=_ok_transport(captured=captured))
    assert captured == []
    assert set(by_name) == {"ariel_status"}
    assert by_name["ariel_status"].status is Status.WARNING
    assert "ariel.web.port" in by_name["ariel_status"].details
