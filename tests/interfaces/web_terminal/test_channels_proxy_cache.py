"""The channel-catalog caching contract must survive the hub's proxy hop.

The bluesky panel is embedded in the web-terminal hub behind ``/panel/{id}/``,
so the browser never talks to the sidecar directly — every ``/channels``
request rides through the hub's generic forwarder. That forwarder applies its
own caching default (``no-cache, no-store, must-revalidate``) via
``setdefault``, which must yield to the route's deliberate ``no-cache`` +
ETag decision; and the conditional-request cycle (``If-None-Match`` → empty
``304``) must relay intact in both directions.

This module pins that end to end over the real network path: a live
``bluesky_web`` uvicorn instance on a free localhost port, proxied by the hub
app through its real ``httpx`` client — no mocks on the wire. The route-level
halves of the contract are pinned in
``tests/interfaces/bluesky_web/test_channels_route.py``; this is the
proxy-transparency half.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces._serving import run_app_server
from osprey.interfaces.bluesky_web.app import app as bluesky_app
from osprey.interfaces.web_terminal.app import UNIVERSAL_PANELS, create_app

_CATALOG = ["BR:DIAG:BPM:01:POSITION:X", "BR:DIAG:BPM:02:POSITION:X"]

# The hub proxy's own default, applied only when the upstream set no
# Cache-Control. Appearing on /channels would mean setdefault clobbered the
# route's decision.
_HUB_DEFAULT = "no-cache, no-store, must-revalidate"


def _make_client(workspace_dir, custom_panels):
    """Create a TestClient with custom panels configured."""
    enabled = set(UNIVERSAL_PANELS)
    with (
        patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace_dir)},
        ),
        patch(
            "osprey.interfaces.web_terminal.app._load_panel_config",
            return_value=(enabled, custom_panels, None),
        ),
    ):
        app = create_app(shell_command="echo")
        with TestClient(app) as c:
            yield app, c


@pytest.fixture
def bluesky_url(monkeypatch, tmp_path):
    """A live bluesky_web server on a free port, catalog in place.

    Environment isolation mirrors ``test_channels_route.py``: config
    resolution is pointed at a nonexistent file so ambient repo/user config
    can't leak a launch token or bridge URL into the sidecar's lifespan, and
    ``CONFIG_FILE`` reproduces the deployed layout — the compose template
    mounts channels.json as config.yml's sibling, which is what the /channels
    route resolves against at request time. The root conftest's
    ``restore_environ`` clears both around every test.
    """
    monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "does-not-exist.yml"))
    monkeypatch.delenv("BLUESKY_LAUNCH_TOKEN", raising=False)

    project = tmp_path / "project"
    project.mkdir()
    (project / "config.yml").write_text("connector:\n  type: mock\n")
    (project / "channels.json").write_text(json.dumps(_CATALOG))
    monkeypatch.setenv("CONFIG_FILE", str(project / "config.yml"))

    with run_app_server(bluesky_app) as url:
        yield url


@pytest.fixture
def hub_client(tmp_path, bluesky_url):
    """Hub app whose ``bluesky`` custom panel points at the live sidecar."""
    ws = tmp_path / "_agent_data"
    ws.mkdir()
    custom = [{"id": "bluesky", "label": "Bluesky", "url": bluesky_url}]
    for _app, client in _make_client(ws, custom):
        yield client


def test_route_cache_decision_survives_the_proxy(hub_client):
    """First fetch: 200 with the route's ``no-cache``, not the hub default."""
    resp = hub_client.get("/panel/bluesky/channels")

    assert resp.status_code == 200
    assert resp.json() == _CATALOG
    assert resp.headers["cache-control"] == "no-cache"
    assert resp.headers["cache-control"] != _HUB_DEFAULT
    assert resp.headers["etag"].startswith('"')


def test_conditional_request_cycle_relays_through_the_proxy(hub_client):
    """``If-None-Match`` reaches the sidecar; its empty 304 comes back intact."""
    first = hub_client.get("/panel/bluesky/channels")
    etag = first.headers["etag"]

    second = hub_client.get("/panel/bluesky/channels", headers={"If-None-Match": etag})

    assert second.status_code == 304
    assert second.content == b""
    assert second.headers["etag"] == etag
    assert second.headers["cache-control"] == "no-cache"
