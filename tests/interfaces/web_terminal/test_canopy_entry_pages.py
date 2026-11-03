"""CANOPY's entry pages through the terminal's panel proxy.

The integration line declares CANOPY's own application documents on its
sidecar registry entry: the viewer's shell (``/``), a screen (``/s/*``), an
embedded screen (``/embed/*``) and Trace (``/trace/*``). Those are relayed
with no sandbox of the terminal's. Every other document under
``/panel/canopy`` -- a draft an agent may have written, an SVG, any spelling
that percent-decodes or dot-segments its way out of an entry page -- gets the
terminal's own sandbox policy and ``nosniff``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest

from osprey.interfaces.web_terminal.panel_sandbox import builtin_entry_pages, matches_entry_page
from osprey.interfaces.web_terminal.routes.proxy import _PANEL_STATE_MAP
from osprey.profiles.sidecar_panels import CANOPY_PANEL_ID

from ._proxy_fakes import panel_app

POLICY = "sandbox allow-scripts allow-modals allow-popups allow-downloads"
UPSTREAM = "http://127.0.0.1:9999"


@pytest.fixture
def client(tmp_path):
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()
    for app, test_client in panel_app(workspace, []):
        for attr in _PANEL_STATE_MAP.values():
            setattr(app.state, attr, UPSTREAM)
        test_client.upstream = _upstream(app)
        yield test_client


def _upstream(app):
    """Answer every proxied request with the content type the test sets on the mock."""

    async def fake_request(*, method, url, headers, content, follow_redirects=True):  # noqa: ARG001
        reply = {} if mock.content_type is None else {"content-type": mock.content_type}
        return httpx.Response(
            status_code=200, content=b"<html><body>x</body></html>", headers=reply
        )

    mock = AsyncMock(side_effect=fake_request)
    mock.content_type = "text/html; charset=utf-8"
    app.state.proxy_client.request = mock
    return mock


def _sandboxed(response) -> bool:
    return POLICY in response.headers.get_list("content-security-policy")


def test_canopy_is_proxied_as_a_registered_sidecar():
    assert CANOPY_PANEL_ID in _PANEL_STATE_MAP
    assert builtin_entry_pages(CANOPY_PANEL_ID) == ("/", "/s/*", "/embed/*", "/trace/*")


@pytest.mark.parametrize(
    "url",
    [
        "/panel/canopy",
        "/panel/canopy/",
        "/panel/canopy/s/sr/overview",
        "/panel/canopy/s/gtl",
        "/panel/canopy/embed/sr/overview",
        "/panel/canopy/trace/",
        "/panel/canopy/trace/index.html",
    ],
)
def test_canopys_entry_pages_carry_no_sandbox(client, url):
    resp = client.get(url)
    assert resp.status_code == 200, url
    assert not _sandboxed(resp), url
    assert "x-content-type-options" not in resp.headers, url
    assert client.upstream.await_count == 1


@pytest.mark.parametrize(
    "url",
    [
        "/panel/canopy/drafts/x.html",
        "/panel/canopy/drafts/sr/overview",
        "/panel/canopy/drafts/",
        "/panel/canopy/index.html",
        "/panel/canopy/viewer/x.html",
    ],
)
def test_a_draft_or_any_other_document_is_sandboxed(client, url):
    resp = client.get(url)
    assert _sandboxed(resp), url
    assert resp.headers.get_list("x-content-type-options") == ["nosniff"], url


@pytest.mark.parametrize(
    "url",
    [
        "/panel/canopy/drafts/drawing.svg",
        "/panel/canopy/drafts/sr/drawing.svg",
        "/panel/canopy/drawing.svg",
        "/panel/canopy/widgets/icon.svg",
    ],
)
def test_an_svg_off_the_entry_pages_is_a_document_and_sandboxed(client, url):
    """An SVG runs script like HTML does; off CANOPY's own pages it is sandboxed."""
    client.upstream.content_type = "image/svg+xml"
    resp = client.get(url)
    assert _sandboxed(resp), url
    assert resp.headers.get_list("x-content-type-options") == ["nosniff"], url


@pytest.mark.parametrize(
    "url",
    [
        "/panel/canopy/s/%2e%2e/drafts/x.html",
        "/panel/canopy/s/%2E%2E/drafts/x.html",
        "/panel/canopy/embed/%2e%2e/drafts/x.html",
        "/panel/canopy/trace/%2e%2e/%2e%2e/drafts/x.html",
        "/panel/canopy/s%2f..%2fdrafts%2fx.html",
        "/panel/canopy/s%5c..%5cdrafts%5cx.html",
        "/panel/canopy/s/%252e%252e/drafts/x.html",
        "/panel/canopy/%2e%2e/canopy/drafts/x.html",
        "/panel/canopy/%64rafts/x.html",
    ],
)
def test_an_escaped_spelling_that_leaves_an_entry_page_is_sandboxed(client, url):
    resp = client.get(url)
    assert _sandboxed(resp), url


@pytest.mark.parametrize("path", ["drafts/x.html", "s/../drafts/x.html", "s/%2e%2e/drafts/x"])
def test_no_draft_path_matches_an_entry_page(path):
    assert not matches_entry_page(path, builtin_entry_pages(CANOPY_PANEL_ID))


@pytest.mark.parametrize(
    "content_type", ["application/json", "text/css", "text/javascript", "image/png"]
)
def test_a_non_document_below_a_draft_keeps_its_type_and_no_sandbox(client, content_type):
    client.upstream.content_type = content_type
    resp = client.get("/panel/canopy/drafts/asset")
    assert not _sandboxed(resp)
