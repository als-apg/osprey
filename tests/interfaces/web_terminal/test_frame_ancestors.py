"""Console-origin HTML carries ``frame-ancestors 'self'`` (Q1-C6).

The web terminal app answers every HTML response that has no CSP of its own
with ``Content-Security-Policy: frame-ancestors 'self'`` and
``X-Frame-Options: SAMEORIGIN``, so no other origin can frame the console or a
first-party panel proxied onto it. The header is set once, where the app is
configured, not per route; the hub frames its panels at the same origin, so
they keep loading. A response that already carries a CSP (opaque agent
content, ARIEL attachments, the Lab page) keeps exactly the one it brought.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.testclient import TestClient

from osprey.interfaces._app_setup import (
    FRAME_ANCESTORS_CSP,
    FrameAncestorsMiddleware,
    configure_interface_app,
)
from osprey.interfaces.common_middleware import WebAuthMiddleware
from osprey.interfaces.opaque_origin import OPAQUE_CSP

from ._proxy_fakes import panel_app

FRAME_CSP = "frame-ancestors 'self'"


@pytest.fixture
def workspace_dir(tmp_path):
    ws = tmp_path / "_agent_data"
    ws.mkdir()
    return ws


@pytest.fixture
def app_and_client(workspace_dir):
    # Config-defined, as the config loader stamps it: a first-party panel keeps
    # the upstream's headers. A runtime registration (no marker) is opaque and
    # carries the sandbox CSP instead — that is test_opaque_panel_browser.py's.
    custom = [
        {"id": "my-dash", "label": "DASH", "url": "http://localhost:9000", "configDefined": True}
    ]
    yield from panel_app(workspace_dir, custom)


def _upstream(app: FastAPI, *, body: str, headers: dict[str, str]) -> None:
    """Make every proxied request answer 200 with *body* and *headers*."""

    async def fake_request(*, method, url, headers, content, **_kwargs):  # noqa: ARG001
        return httpx.Response(status_code=200, text=body, headers=upstream_headers)

    upstream_headers = headers
    app.state.proxy_client.request = AsyncMock(side_effect=fake_request)


def test_constant_is_q1_c6_spelling():
    assert FRAME_ANCESTORS_CSP == FRAME_CSP


class TestConsolePage:
    def test_console_page_carries_frame_ancestors(self, app_and_client):
        _app, client = app_and_client
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")
        assert resp.headers["content-security-policy"] == FRAME_CSP
        assert resp.headers["x-frame-options"] == "SAMEORIGIN"

    def test_headers_appear_once(self, app_and_client):
        _app, client = app_and_client
        resp = client.get("/")
        assert len(resp.headers.get_list("content-security-policy")) == 1
        assert len(resp.headers.get_list("x-frame-options")) == 1


class TestProxiedPanels:
    def test_hub_framed_first_party_panel_still_loads(self, app_and_client):
        app, client = app_and_client
        _upstream(
            app,
            body="<html><head></head><body><h1>dash</h1></body></html>",
            headers={"content-type": "text/html; charset=utf-8"},
        )
        resp = client.get("/panel/my-dash/")
        assert resp.status_code == 200
        assert "<h1>dash</h1>" in resp.text
        # Same-origin framing (the hub's /panel/<id>/ iframes) stays allowed.
        assert resp.headers["content-security-policy"] == FRAME_CSP
        assert resp.headers["x-frame-options"] == "SAMEORIGIN"

    def test_opaque_response_csp_is_not_overwritten(self, app_and_client):
        app, client = app_and_client
        _upstream(
            app,
            body="<html><script>agent()</script></html>",
            headers={"content-type": "text/html", "content-security-policy": OPAQUE_CSP},
        )
        resp = client.get("/panel/my-dash/api/files/report.html")
        assert resp.status_code == 200
        assert resp.headers.get_list("content-security-policy") == [OPAQUE_CSP]
        assert "x-frame-options" not in resp.headers

    def test_attachment_csp_is_not_overwritten(self, app_and_client):
        app, client = app_and_client
        attachment_csp = "default-src 'none'; sandbox"
        _upstream(
            app,
            body="<html></html>",
            headers={"content-type": "text/html", "content-security-policy": attachment_csp},
        )
        resp = client.get("/panel/my-dash/attachments/1")
        assert resp.headers.get_list("content-security-policy") == [attachment_csp]

    def test_non_html_panel_response_untouched(self, app_and_client):
        app, client = app_and_client
        _upstream(app, body='{"ok": true}', headers={"content-type": "application/json"})
        resp = client.get("/panel/my-dash/api/state")
        assert resp.status_code == 200
        assert "content-security-policy" not in resp.headers
        assert "x-frame-options" not in resp.headers


def _bare_app(*, frame_ancestors: bool, tmp_path) -> FastAPI:
    app = FastAPI()
    configure_interface_app(app, static_dir=tmp_path, frame_ancestors=frame_ancestors)
    return app


class TestOptIn:
    """configure_interface_app is shared by ten apps; only the console opts in."""

    def test_default_off(self, tmp_path):
        app = _bare_app(frame_ancestors=False, tmp_path=tmp_path)
        assert not any(m.cls is FrameAncestorsMiddleware for m in app.user_middleware)

    def test_opted_in_sits_directly_inside_the_gate(self, tmp_path):
        app = _bare_app(frame_ancestors=True, tmp_path=tmp_path)
        classes = [m.cls for m in app.user_middleware]
        assert classes[:2] == [WebAuthMiddleware, FrameAncestorsMiddleware]


class TestMiddleware:
    """The layer in isolation: HTML without a CSP only."""

    @pytest.fixture
    def client(self):
        app = FastAPI()

        @app.get("/page")
        def page():
            return HTMLResponse("<html></html>")

        @app.get("/own-csp")
        def own_csp():
            return HTMLResponse("<html></html>", headers={"Content-Security-Policy": OPAQUE_CSP})

        @app.get("/own-xfo")
        def own_xfo():
            return HTMLResponse("<html></html>", headers={"X-Frame-Options": "DENY"})

        @app.get("/json")
        def as_json():
            return JSONResponse({"a": 1})

        @app.get("/text")
        def as_text():
            return PlainTextResponse("hi")

        app.add_middleware(FrameAncestorsMiddleware)
        return TestClient(app)

    def test_html_gets_both(self, client):
        resp = client.get("/page")
        assert resp.headers["content-security-policy"] == FRAME_CSP
        assert resp.headers["x-frame-options"] == "SAMEORIGIN"

    def test_own_csp_kept_and_no_xfo_added(self, client):
        resp = client.get("/own-csp")
        assert resp.headers.get_list("content-security-policy") == [OPAQUE_CSP]
        assert "x-frame-options" not in resp.headers

    def test_own_xfo_not_duplicated(self, client):
        resp = client.get("/own-xfo")
        assert resp.headers["content-security-policy"] == FRAME_CSP
        assert resp.headers.get_list("x-frame-options") == ["DENY"]

    @pytest.mark.parametrize("path", ["/json", "/text"])
    def test_non_html_untouched(self, client, path):
        resp = client.get(path)
        assert "content-security-policy" not in resp.headers
        assert "x-frame-options" not in resp.headers
