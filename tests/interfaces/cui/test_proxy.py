"""Tests for the CUI reverse proxy."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces.cui.proxy import _rewrite_html, create_cui_proxy_mount


@pytest.fixture
def app():
    """Create a minimal FastAPI app with the CUI proxy mounted."""
    app = FastAPI()
    app.state.cui_server_url = "http://127.0.0.1:3001"
    app.state.project_cwd = "/home/user/my-project"
    app.routes.append(create_cui_proxy_mount())
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


def _mock_response(status_code=200, json_data=None, content=None):
    """Build a real httpx.Response with the given payload."""
    if json_data is not None and content is None:
        content = json.dumps(json_data).encode()
    return httpx.Response(
        status_code=status_code,
        content=content,
        request=httpx.Request("GET", "http://test"),
    )


def _make_mock_client(response):
    """Create a mock httpx.AsyncClient that works as an async context manager."""
    mock_client = AsyncMock()
    mock_client.request.return_value = response
    mock_client.get.return_value = response
    mock_client.post.return_value = response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


class TestConversationListFiltering:
    def test_injects_project_path_filter(self, client):
        """GET /cui/api/conversations should add projectPath query param."""
        resp_data = {"conversations": [], "total": 0}
        mock_client = _make_mock_client(_mock_response(json_data=resp_data))

        with patch("osprey.interfaces.cui.proxy.httpx.AsyncClient", return_value=mock_client):
            resp = client.get("/cui/api/conversations")

        assert resp.status_code == 200
        # The conversation interception calls _forward which uses client.request(url=...)
        call_url = mock_client.request.call_args.kwargs.get(
            "url", mock_client.request.call_args[1].get("url", "")
        )
        assert "projectPath=%2Fhome%2Fuser%2Fmy-project" in call_url

    def test_preserves_existing_query_params(self, client):
        """Should keep user-provided query params alongside projectPath."""
        resp_data = {"conversations": [], "total": 0}
        mock_client = _make_mock_client(_mock_response(json_data=resp_data))

        with patch("osprey.interfaces.cui.proxy.httpx.AsyncClient", return_value=mock_client):
            client.get("/cui/api/conversations?limit=10&sortBy=updated")

        call_url = mock_client.request.call_args.kwargs.get(
            "url", mock_client.request.call_args[1].get("url", "")
        )
        assert "limit=10" in call_url
        assert "projectPath=" in call_url


class TestWorkingDirectoriesFiltering:
    def test_filters_to_project_only(self, client):
        """GET /cui/api/working-directories should return only our project."""
        upstream_data = {
            "directories": [
                {"path": "/home/user/my-project", "shortname": "my-project", "lastDate": "2025-01-01", "conversationCount": 5},
                {"path": "/home/user/other-project", "shortname": "other-project", "lastDate": "2025-01-02", "conversationCount": 3},
            ],
            "totalCount": 2,
        }
        mock_client = _make_mock_client(_mock_response(json_data=upstream_data))

        with patch("osprey.interfaces.cui.proxy.httpx.AsyncClient", return_value=mock_client):
            resp = client.get("/cui/api/working-directories")

        data = resp.json()
        assert len(data["directories"]) == 1
        assert data["directories"][0]["path"] == "/home/user/my-project"
        assert data["totalCount"] == 1

    def test_adds_project_if_missing(self, client):
        """If project has no sessions yet, still return it in the list."""
        upstream_data = {
            "directories": [
                {"path": "/home/user/other", "shortname": "other", "lastDate": "2025-01-01", "conversationCount": 1},
            ],
            "totalCount": 1,
        }
        mock_client = _make_mock_client(_mock_response(json_data=upstream_data))

        with patch("osprey.interfaces.cui.proxy.httpx.AsyncClient", return_value=mock_client):
            resp = client.get("/cui/api/working-directories")

        data = resp.json()
        assert len(data["directories"]) == 1
        assert data["directories"][0]["path"] == "/home/user/my-project"


class TestConversationStartOverride:
    def test_forces_working_directory(self, client):
        """POST /cui/api/conversations/start should override workingDirectory."""
        mock_client = _make_mock_client(
            _mock_response(json_data={"streamingId": "abc", "sessionId": "xyz"})
        )

        with patch("osprey.interfaces.cui.proxy.httpx.AsyncClient", return_value=mock_client):
            client.post(
                "/cui/api/conversations/start",
                json={
                    "workingDirectory": "/some/other/path",
                    "initialPrompt": "hello",
                },
            )

        # Verify upstream call used our project path
        call_json = mock_client.post.call_args.kwargs.get("json", {})
        assert call_json["workingDirectory"] == "/home/user/my-project"
        assert call_json["initialPrompt"] == "hello"


class TestPassthrough:
    def test_other_routes_pass_through(self, client):
        """Other API routes should proxy without modification."""
        mock_client = _make_mock_client(_mock_response(json_data={"status": "ok"}))

        with patch("osprey.interfaces.cui.proxy.httpx.AsyncClient", return_value=mock_client):
            resp = client.get("/cui/api/system/health")

        assert resp.status_code == 200

    def test_returns_502_when_cui_unavailable(self):
        """Should return 502 if CUI server URL is not set."""
        app = FastAPI()
        app.state.cui_server_url = None
        app.state.project_cwd = "/some/path"
        app.routes.append(create_cui_proxy_mount())
        client = TestClient(app)

        resp = client.get("/cui/api/conversations")
        assert resp.status_code == 502

    def test_html_responses_are_rewritten(self, client):
        """HTML responses should have asset paths rewritten for sub-path proxy."""
        cui_html = b'<html><head><script src="/assets/main.js"></script></head></html>'
        resp = _mock_response(
            status_code=200,
            content=cui_html,
        )
        # Add text/html content-type header
        resp.headers["content-type"] = "text/html; charset=UTF-8"
        mock_client = _make_mock_client(resp)

        with patch("osprey.interfaces.cui.proxy.httpx.AsyncClient", return_value=mock_client):
            result = client.get("/cui/some-page")

        body = result.text
        assert 'src="/cui/assets/main.js"' in body
        assert "data-osprey-proxy" in body  # runtime patch injected

    def test_json_responses_are_not_rewritten(self, client):
        """JSON API responses should NOT be rewritten."""
        mock_client = _make_mock_client(
            _mock_response(json_data={"href": "/assets/something"})
        )

        with patch("osprey.interfaces.cui.proxy.httpx.AsyncClient", return_value=mock_client):
            result = client.get("/cui/api/system/health")

        body = result.json()
        assert body["href"] == "/assets/something"  # unchanged


class TestHTMLRewriting:
    """Unit tests for _rewrite_html path rewriting."""

    def test_rewrites_script_src(self):
        html = b'<script src="/assets/main-abc.js"></script>'
        result = _rewrite_html(html).decode()
        assert 'src="/cui/assets/main-abc.js"' in result

    def test_rewrites_link_href(self):
        html = b'<link rel="stylesheet" href="/assets/main-abc.css">'
        result = _rewrite_html(html).decode()
        assert 'href="/cui/assets/main-abc.css"' in result

    def test_rewrites_registerSW(self):
        html = b'<script src="/registerSW.js"></script>'
        result = _rewrite_html(html).decode()
        assert 'src="/cui/registerSW.js"' in result

    def test_rewrites_icon_paths(self):
        html = b'<link rel="apple-touch-icon" href="/icon-192x192.png" />'
        result = _rewrite_html(html).decode()
        assert 'href="/cui/icon-192x192.png"' in result

    def test_preserves_external_urls(self):
        html = b'<script src="https://cdn.example.com/lib.js"></script>'
        result = _rewrite_html(html).decode()
        assert 'src="https://cdn.example.com/lib.js"' in result

    def test_preserves_protocol_relative_urls(self):
        html = b'<script src="//cdn.example.com/lib.js"></script>'
        result = _rewrite_html(html).decode()
        assert 'src="//cdn.example.com/lib.js"' in result

    def test_does_not_double_prefix(self):
        """Paths already starting with /cui/ should not be prefixed again."""
        html = b'<script src="/cui/assets/main.js"></script>'
        result = _rewrite_html(html).decode()
        assert 'src="/cui/assets/main.js"' in result
        assert "/cui/cui/" not in result

    def test_injects_runtime_patch(self):
        html = b"<html><head><title>CUI</title></head></html>"
        result = _rewrite_html(html).decode()
        assert "data-osprey-proxy" in result
        assert "window.fetch" in result
        assert "window.EventSource" in result

    def test_injects_history_api_patches(self):
        """Runtime patch should wrap pushState/replaceState for sub-path routing."""
        html = b"<html><head><title>CUI</title></head></html>"
        result = _rewrite_html(html).decode()
        assert "history.pushState" in result
        assert "history.replaceState" in result

    def test_injects_url_rewrite(self):
        """Runtime patch should rewrite /cui/ to / via replaceState before React."""
        html = b"<html><head><title>CUI</title></head></html>"
        result = _rewrite_html(html).decode()
        # The URL rewrite uses replaceState to strip /cui prefix
        assert "location.pathname" in result
        assert "p.slice(P.length)" in result

    def test_injects_popstate_handler(self):
        """Runtime patch should handle popstate for back/forward navigation."""
        html = b"<html><head><title>CUI</title></head></html>"
        result = _rewrite_html(html).decode()
        assert "popstate" in result

    def test_runtime_patch_before_other_scripts(self):
        html = b"<html><head><script src='/cui/assets/main.js'></script></head></html>"
        result = _rewrite_html(html).decode()
        patch_pos = result.index("data-osprey-proxy")
        script_pos = result.index("main.js")
        assert patch_pos < script_pos

    def test_full_cui_html(self):
        """Test with realistic CUI HTML matching the actual npm package output."""
        html = b"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <link rel="apple-touch-icon" href="/icon-192x192.png" />
    <link rel="icon" type="image/svg+xml" href="/favicon.svg" />
    <link rel="manifest" href="/manifest.json" />
    <title>CUI</title>
    <script src="https://unpkg.com/@alenaksu/json-viewer@2.1.0/dist/json-viewer.bundle.js"></script>
    <script type="module" crossorigin src="/assets/main-g7SA0I8d.js"></script>
    <link rel="stylesheet" crossorigin href="/assets/main-tWwRjkl8.css">
    <script src="/registerSW.js"></script>
</head>
<body><div id="root"></div></body>
</html>"""
        result = _rewrite_html(html).decode()

        # Assets rewritten
        assert 'src="/cui/assets/main-g7SA0I8d.js"' in result
        assert 'href="/cui/assets/main-tWwRjkl8.css"' in result
        assert 'src="/cui/registerSW.js"' in result
        assert 'href="/cui/icon-192x192.png"' in result
        assert 'href="/cui/favicon.svg"' in result
        assert 'href="/cui/manifest.json"' in result

        # External URL preserved
        assert "https://unpkg.com/" in result

        # Runtime patch injected
        assert "data-osprey-proxy" in result
