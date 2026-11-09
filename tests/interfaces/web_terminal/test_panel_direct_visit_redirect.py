"""A direct visit to a discovered bundle's entry document opens the web terminal.

Bundles do not call ``applyEmbedded``, so the server answers for them: a 303 to
``<prefix>/?open=<id>&at=<at>``. ``at`` is the contract ``frame-params.js``
holds on the client side. Only the bundle route redirects: the generic proxy
serves arbitrary documents under a sidecar mount, and those must not bounce.

The ``Location`` under a set user is pinned verbatim here and read back against
``isUrlWithinMount`` in ``frame-params.test.js``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.app import create_app
from osprey.interfaces.web_terminal.panel_discovery import DiscoveredPanel
from osprey.interfaces.web_terminal.routes.panels import router

from .conftest import bare_route_app

_DOCUMENT = {"sec-fetch-dest": "document", "sec-fetch-mode": "navigate"}
_IFRAME = {"sec-fetch-dest": "iframe", "sec-fetch-mode": "navigate"}
_HTML = {"accept": "text/html,application/xhtml+xml"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("OSPREY_TERMINAL_USER", raising=False)
    bundle = tmp_path / "mine"
    bundle.mkdir()
    (bundle / "index.html").write_text("<html><body>mine</body></html>", encoding="utf-8")
    (bundle / "page.html").write_text("<html><body>page</body></html>", encoding="utf-8")
    other = tmp_path / "other"
    other.mkdir()
    (other / "main.html").write_text("<html><body>other</body></html>", encoding="utf-8")
    app = bare_route_app(
        router,
        discovered_panel_dirs={
            "mine": DiscoveredPanel("mine", "Mine", "index.html", bundle),
            "other": DiscoveredPanel("other", "Other", "main.html", other),
        },
    )
    with TestClient(app, follow_redirects=False) as test_client:
        yield test_client


def _visit(resp) -> dict[str, list[str]]:
    assert resp.status_code == 303
    parts = urlsplit(resp.headers["location"])
    return {"path": [parts.path], **parse_qs(parts.query)}


@pytest.mark.parametrize("url", ["/panel-static/mine/", "/panel-static/mine/index.html"])
def test_a_document_navigation_to_the_bundle_root_redirects(client, url):
    assert _visit(client.get(url, headers=_DOCUMENT)) == {
        "path": ["/"],
        "open": ["mine"],
        "at": [url],
    }


def test_a_bundle_named_entry_counts_as_its_root(client):
    assert _visit(client.get("/panel-static/other/main.html", headers=_DOCUMENT))["at"] == [
        "/panel-static/other/main.html"
    ]


def test_no_sec_fetch_headers_but_a_page_request_redirects(client):
    assert _visit(client.get("/panel-static/mine/", headers=_HTML))["open"] == ["mine"]


def test_the_query_stays_on_at(client):
    resp = client.get("/panel-static/mine/?x=1&y=a%26b", headers=_DOCUMENT)
    assert _visit(resp)["at"] == ["/panel-static/mine/?x=1&y=a%26b"]


def test_under_a_user_the_location_carries_the_prefix_twice(client, monkeypatch):
    monkeypatch.setenv("OSPREY_TERMINAL_USER", "alice")
    resp = client.get("/panel-static/mine/?x=1", headers=_DOCUMENT)
    assert resp.status_code == 303
    assert (
        resp.headers["location"]
        == "/u/alice/?open=mine&at=%2Fu%2Falice%2Fpanel-static%2Fmine%2F%3Fx%3D1"
    )


@pytest.mark.parametrize(
    ("url", "headers"),
    [
        pytest.param("/panel-static/mine/", _IFRAME, id="an-iframe"),
        pytest.param("/panel-static/mine/", {}, id="a-fetch-with-no-page-accept"),
        pytest.param("/panel-static/mine/?embedded=true", _HTML, id="embedded-no-sec-fetch"),
        pytest.param("/panel-static/mine/?embedded=true", _DOCUMENT, id="embedded-document"),
        pytest.param("/panel-static/mine/page.html", _DOCUMENT, id="a-non-root-file"),
        pytest.param("/panel-static/other/index.html", _DOCUMENT, id="a-missing-index"),
    ],
)
def test_what_does_not_redirect(client, url, headers):
    assert client.get(url, headers=headers).status_code != 303


def test_an_unknown_bundle_is_a_404_and_not_a_trip_to_the_hub(client):
    assert client.get("/panel-static/nobody/", headers=_DOCUMENT).status_code == 404


@pytest.fixture
def full_app_client(tmp_path, monkeypatch):
    monkeypatch.delenv("OSPREY_TERMINAL_USER", raising=False)
    app = create_app(shell_command="echo test", project_dir=str(tmp_path))
    with TestClient(app, follow_redirects=False) as test_client:
        app.state.jupyter_server_url = "http://127.0.0.1:9"
        app.state.proxy_client.request = AsyncMock(
            return_value=httpx.Response(
                200, headers={"content-type": "text/html"}, content=b"<html><body>x</body></html>"
            )
        )
        yield test_client


def test_a_jupyter_document_is_relayed_and_never_bounced(full_app_client):
    """A user's own ``.html`` under a sidecar mount: Jupyter has the client guard."""
    resp = full_app_client.get("/panel/jupyter/files/x.html", headers=_DOCUMENT)
    assert resp.status_code == 200
    assert "location" not in resp.headers
