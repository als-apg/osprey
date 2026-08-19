"""Smoke tests for ``conftest.py``'s ``bluesky_live_server`` fixture.

The fixture exists for the real-browser Playwright suites, but its own
contract — the server boots, the temp channel catalog is served, the
no-catalog variant 404s, the bridge stub answers through the proxy, and the
panel bundle is actually mounted — is pinned here with plain HTTP, so it runs
in the ordinary unit lane and a broken fixture fails fast instead of as an
opaque browser-test timeout.
"""

from __future__ import annotations

import httpx

from tests.interfaces.bluesky_web.conftest import (
    DEFAULT_CHANNEL_CATALOG,
    STUB_PLAN_NAME,
)


def test_fixture_boots_and_answers_the_healthcheck(bluesky_live_server) -> None:
    with bluesky_live_server() as base_url:
        response = httpx.get(f"{base_url}/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_channels_serves_the_temp_catalog_with_an_etag(bluesky_live_server) -> None:
    """The default launch serves the fixture's catalog under the real caching contract."""
    with bluesky_live_server() as base_url:
        response = httpx.get(f"{base_url}/channels")

    assert response.status_code == 200
    assert response.json() == DEFAULT_CHANNEL_CATALOG
    assert response.headers["etag"].startswith('"')
    assert response.headers["cache-control"] == "no-cache"


def test_an_explicit_catalog_replaces_the_default(bluesky_live_server) -> None:
    catalog = ["LINAC:GUN:PHASE:SP", "LINAC:GUN:PHASE:RB"]

    with bluesky_live_server(catalog=catalog) as base_url:
        response = httpx.get(f"{base_url}/channels")

    assert response.status_code == 200
    assert response.json() == catalog


def test_no_catalog_variant_404s(bluesky_live_server) -> None:
    """``catalog=None`` is the "this deployment has no catalog" path."""
    with bluesky_live_server(catalog=None) as base_url:
        response = httpx.get(f"{base_url}/channels")

    assert response.status_code == 404


def test_bridge_stub_serves_role_tagged_plans_through_the_proxy(bluesky_live_server) -> None:
    """The stubbed ``/plans`` answer reaches the browser side of the proxy.

    This is what the browser suites' plan form is built from, so the two
    field shapes they exercise are pinned: a plain string channel field and a
    channel-list field, each carrying ``x-channel-role`` on the field itself.
    """
    with bluesky_live_server() as base_url:
        response = httpx.get(f"{base_url}/plans")

    assert response.status_code == 200
    plans = response.json()
    assert [plan["name"] for plan in plans] == [STUB_PLAN_NAME]

    properties = plans[0]["schema"]["properties"]
    assert properties["target"]["x-channel-role"] == "movable"
    assert properties["monitors"]["x-channel-role"] == "readable"
    assert properties["monitors"]["x-widget"] == "channel-list"


def test_panel_bundle_is_served(bluesky_live_server) -> None:
    """The real panel bundle (not a stub) is mounted at /bluesky/.

    The browser suites navigate here; a fixture that served an empty
    directory would let them fail with an unhelpful blank page.
    """
    with bluesky_live_server() as base_url:
        response = httpx.get(f"{base_url}/bluesky/")

    assert response.status_code == 200
    assert "<html" in response.text.lower()
