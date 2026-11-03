"""The navigation guard: the server half.

Every ``POST /api/panel-op/open`` URL and every runtime registration URL is read
the way the browser's WHATWG URL parser reads it (leading and trailing C0
controls and spaces stripped, tabs and newlines removed, ``\\`` read as ``/``)
and must then be a root-relative path or an http(s) URL. An ``open`` URL must
then stay with its panel: a path under the panel's own mount (encoded ``..``,
``//`` and ``\\`` read as a server reads them), or an http(s) URL on the panel's
own server, which is rewritten into the mount. Anything else is a 422 and no
frame is relayed to the dock client. The agent's ``open_panel`` tool applies
the scheme rule before it posts the op, so that refusal reaches the agent as an
error. The client half (``buildEmbedSrc``, ``isUrlWithinMount``) is pinned by
``static/panel-nav-guard.test.mjs``.

Each route case runs with an empty URL prefix (single-user) and a ``/u/alice``
prefix (multi-user).
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces.panel_urls import (
    member_navigation_url,
    navigation_url_refusal,
    normalise_navigation_url,
)
from osprey.interfaces.web_terminal.dock_layout import normalize_layout
from osprey.interfaces.web_terminal.file_watcher import FileEventBroadcaster
from osprey.interfaces.web_terminal.routes.panel_ops import router as ops_router
from osprey.interfaces.web_terminal.routes.panels import router
from tests.mcp_server.conftest import assert_raises_error, get_tool_fn

from .conftest import bare_route_app

_LAN_ADDR = [(2, 1, 6, "", ("10.0.0.5", 0))]
_GETADDRINFO_TARGET = "osprey.interfaces.web_terminal.routes.panels.socket.getaddrinfo"
_OWN_ADDRS_TARGET = "osprey.interfaces.web_terminal.routes.panels._host_interface_addresses"

# Every URL here must be refused, by the server and by the client alike.
HOSTILE_URLS = [
    "javascript:alert(1)",
    "JavaScript:alert(1)",
    "JAVASCRIPT:alert(1)",
    "  javascript:alert(1)",
    "\x00\x01javascript:alert(1)",
    "\tjava\tscript:alert(1)",
    "java\nscript:alert(1)",
    "java\r\nscript:alert(1)",
    "javascript:alert(1)  ",
    "data:text/html,<script>alert(1)</script>",
    " DATA:text/html,x",
    "vbscript:msgbox(1)",
    "VBScript:msgbox(1)",
    "blob:http://localhost/0b1c",
    "file:///etc/passwd",
    "//evil.example/x",
    "  //evil.example/x",
    "/\\evil.example/x",
    "\\\\evil.example/x",
    "/\t/evil.example/x",
    "/\n/evil.example/x",
    "panel/ariel",
    "#fragment",
    "?q=1",
    "",
    "   ",
    "http:///no-host",
    "http:grafana.lan/d",
    "https:",
]

# Every URL here must still focus.
NAVIGABLE_URLS = [
    "/panel/ariel/",
    "/panel/ariel/#/draft/7",
    "/ariel/logbook/draft/7",
    "/panel-static/reference/index.html?x=1",
    "http://grafana.lan:3000/d/abc",
    "https://grafana.lan:3000/d/abc",
    "HTTPS://grafana.lan/d/abc",
    "  https://grafana.lan/d/abc  ",
    "http://127.0.0.1:10300/#draft",
]

PREFIXES = [
    pytest.param((None, ""), id="single-user"),
    pytest.param(("alice", "/u/alice"), id="u-alice"),
]


@pytest.fixture(params=PREFIXES)
def prefix(request, monkeypatch):
    """Run the case under an empty prefix and under ``/u/alice``."""
    user, value = request.param
    if user is None:
        monkeypatch.delenv("OSPREY_TERMINAL_USER", raising=False)
    else:
        monkeypatch.setenv("OSPREY_TERMINAL_USER", user)
    return value


def _client(**state) -> TestClient:
    defaults = {
        "enabled_panels": {"ariel"},
        "custom_panels": [],
        "active_panel": None,
        "visible_panels": ["ariel"],
        "allow_runtime_panels": True,
        "runtime_panel_allowlist": None,
    }
    defaults.update(state)
    return TestClient(bare_route_app(router, **defaults))


# ---- the rule itself ----


class TestNormalise:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("  JavaScript:alert(1)", "javascript:alert(1)"),
            ("\x00\x1fhttp://h/", "http://h/"),
            ("java\tscr\nipt:x", "javascript:x"),
            ("/\\evil.example", "//evil.example"),
            ("HTTPS://Host/Path", "https://Host/Path"),
            ("/panel/ariel/#/x", "/panel/ariel/#/x"),
            ("/a b", "/a b"),
        ],
    )
    def test_reads_as_whatwg_does(self, raw, expected):
        assert normalise_navigation_url(raw) == expected

    @pytest.mark.parametrize("raw", HOSTILE_URLS)
    def test_hostile_urls_refused(self, raw):
        assert navigation_url_refusal(normalise_navigation_url(raw)) is not None

    @pytest.mark.parametrize("raw", NAVIGABLE_URLS)
    def test_navigable_urls_admitted(self, raw):
        assert navigation_url_refusal(normalise_navigation_url(raw)) is None


# ---- /api/panel-op/open ----

_DOCK_CLIENT = "page-1"
_LAYOUT = normalize_layout(
    {
        "split": "row",
        "items": [
            {"tile": "t1", "tabs": ["ariel"], "showing": "ariel", "size": 0.6},
            {"tile": "t2", "tabs": ["terminal"], "showing": "terminal", "size": 0.4},
        ],
    }
)


def _op_app(**state) -> FastAPI:
    """The panels and ops routers with a current dock client the test plays."""
    defaults = {
        "enabled_panels": {"ariel"},
        "custom_panels": [],
        "active_panel": None,
        "visible_panels": ["ariel"],
        "broadcaster": FileEventBroadcaster(),
        "dock_layout": _LAYOUT,
        "dock_layout_rev": 3,
        "dock_client_id": _DOCK_CLIENT,
        "layout_dock": True,
    }
    defaults.update(state)
    return bare_route_app(router, ops_router, **defaults)


def _open(
    app: FastAPI, url, *, answer: bool = True, member: str = "ariel"
) -> tuple[httpx.Response, dict | None]:
    """POST an ``open`` of ``member`` (ARIEL by default) with ``url``; play the
    dock client when ``answer``.

    Returns the write's response and the relayed frame, ``None`` when the
    route relayed nothing.
    """

    async def run():
        page = app.state.broadcaster.subscribe(_DOCK_CLIENT)
        transport = httpx.ASGITransport(app=app)
        body: dict = {"request_id": "r1", "member": member}
        if url is not None:
            body["url"] = url
        async with httpx.AsyncClient(transport=transport, base_url="http://wt") as client:
            task = asyncio.create_task(client.post("/api/panel-op/open", json=body))
            frame = None
            if answer:
                frame = await asyncio.wait_for(page.get(), timeout=2)
                await client.post(
                    "/api/panel-op-result",
                    json={"request_id": "r1", "client_id": _DOCK_CLIENT, "layout": _LAYOUT},
                )
            resp = await task
        if frame is None and not page.empty():
            frame = page.get_nowait()
        app.state.broadcaster.unsubscribe(page)
        return resp, frame

    return asyncio.run(run())


@pytest.mark.usefixtures("prefix")
class TestPanelOpenGuard:
    """The guard on the one navigating op, the agent's ``open`` with a url."""

    @pytest.mark.parametrize("url", HOSTILE_URLS)
    def test_hostile_url_is_refused_and_never_relayed(self, url):
        app = _op_app()

        resp, frame = _open(app, url, answer=False)

        assert resp.status_code == 422, resp.text
        assert resp.json()["reason"] == "invalid_url"
        assert resp.json()["detail"] == navigation_url_refusal(normalise_navigation_url(url))
        assert frame is None
        assert app.state.active_panel is None
        assert app.state.dock_layout_rev == 3
        assert len(app.state.agent_activity_buffer) == 0

    def test_hostile_url_is_refused_before_membership_is_written(self):
        app = _op_app(visible_panels=[])

        resp, _ = _open(app, "javascript:alert(1)", answer=False)

        assert resp.status_code == 422
        assert app.state.visible_panels == []

    @pytest.mark.parametrize("path", ["/panel/ariel", "/panel/ariel/", "/panel/ariel/#/draft/7"])
    def test_relative_path_still_opens_and_is_prefixed(self, prefix, path):
        """A path under the member's own mount opens, prefixed into the user's.

        Paths elsewhere on the origin (``/ariel/logbook/...``, another panel's
        ``/panel-static/...``) pass the scheme rule but not the member rule;
        ``TestPanelOpenMemberGuard`` holds those refusals.
        """
        app = _op_app()

        resp, frame = _open(app, path)

        assert resp.status_code == 200, resp.text
        assert frame["type"] == "panel_op"
        assert frame["args"]["url"] == f"{prefix}{path}"

    def test_padded_relative_path_is_relayed_normalised(self, prefix):
        app = _op_app()

        resp, frame = _open(app, " \t/panel/ar\tiel/#/dr\naft/7\n ")

        assert resp.status_code == 200, resp.text
        assert frame["args"]["url"] == f"{prefix}/panel/ariel/#/draft/7"

    @pytest.mark.parametrize(
        "url",
        [
            "http://grafana.lan:3000/d/abc",
            "HTTPS://grafana.lan/d/abc",
            "  https://grafana.lan/d/abc  ",
        ],
    )
    def test_http_url_on_a_foreign_origin_is_refused(self, url):
        """An http(s) URL passes the scheme rule but names a host that is not
        this panel's server, so it is refused and never reaches the browser.

        Before the member rule these opened as given; a panel could be pointed
        at any host. ``TestPanelOpenMemberGuard`` holds the admitted form (an
        URL on the panel's own server, rewritten into its mount).
        """
        app = _op_app()

        resp, frame = _open(app, url, answer=False)

        assert resp.status_code == 422, resp.text
        assert resp.json()["reason"] == "invalid_url"
        assert "/panel/ariel" in resp.json()["detail"]
        assert frame is None
        assert app.state.dock_layout_rev == 3

    def test_no_url_still_opens(self):
        app = _op_app()

        resp, frame = _open(app, None)

        assert resp.status_code == 200
        assert "url" not in frame["args"]


# ---- the member rule: a url stays with its panel ----

_ARIEL_UPSTREAM = "http://127.0.0.1:10300"
_GRAFANA = {
    "id": "grafana",
    "label": "GRAFANA",
    "url": "http://grafana.lan:3000/grafana",
    "healthEndpoint": None,
    "path": "/",
}
_REFERENCE = {
    "id": "reference",
    "label": "REF",
    "url": "/panel-static/reference/",
    "discovered": True,
    "healthEndpoint": None,
    "path": "/",
}

# Paths that pass the scheme rule and leave ARIEL's mount: elsewhere on this
# origin, another panel, a sibling prefix, and every spelling of ``..``, ``//``
# and ``\\`` a server might read differently from the guard.
OUTSIDE_ARIEL_PATHS = [
    "/api/panels",
    "/",
    "/panel/grafana/d/1",
    "/panel/arielx/",
    "/panel/ariel/../grafana/",
    "/panel/ariel/%2e%2e/grafana",
    "/panel/ariel/..%2Fgrafana",
    "/panel/ariel/%2e./grafana",
    "/panel/ariel/%252e%252e/grafana",
    "/panel/ariel//evil",
    "/panel/ariel/%2F%2Fevil.example/x",
    "/panel/ariel/%252F%252Fevil.example/x",
    "/panel/ariel/%5C%5Cevil.example/x",
    "/panel/ariel/%5Cevil.example/x",
    "/panel/ariel/x;/../../api",
    "/panel/ariel/./%2e%2e/%2e%2e/api/panels",
]

# http(s) URLs that pass the scheme rule and are not ARIEL's own server.
FOREIGN_ORIGIN_URLS = [
    "http://evil.example/",
    "https://127.0.0.1:10300/",
    "http://127.0.0.1:10301/",
    "http://127.0.0.1/",
    "http://grafana.lan:3000/grafana/d/1",
    "http://127.0.0.1:10300@evil.example/",
    "http://evil.example#@127.0.0.1:10300/",
]


class TestMemberNavigationUrl:
    """The pure rule, as the route and the tests of the route rely on it."""

    @pytest.mark.parametrize("path", OUTSIDE_ARIEL_PATHS)
    def test_a_path_leaving_the_mount_is_refused(self, path):
        admitted, refusal = member_navigation_url(
            normalise_navigation_url(path), mount="/panel/ariel", upstream=_ARIEL_UPSTREAM
        )
        assert admitted is None
        assert "/panel/ariel" in refusal

    @pytest.mark.parametrize("url", FOREIGN_ORIGIN_URLS)
    def test_a_foreign_origin_is_refused(self, url):
        admitted, refusal = member_navigation_url(
            normalise_navigation_url(url), mount="/panel/ariel", upstream=_ARIEL_UPSTREAM
        )
        assert admitted is None
        assert "/panel/ariel" in refusal

    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("/panel/ariel", "/panel/ariel"),
            ("/panel/ariel/", "/panel/ariel/"),
            ("/panel/ariel/#/draft/7", "/panel/ariel/#/draft/7"),
            ("/panel/ariel/a%20b/?x=1&y=2", "/panel/ariel/a%20b/?x=1&y=2"),
            ("/panel/ariel/a%3Fb", "/panel/ariel/a%3Fb"),
            ("http://127.0.0.1:10300", "/panel/ariel/"),
            ("http://127.0.0.1:10300/#draft", "/panel/ariel/#draft"),
            ("HTTP://127.0.0.1:10300/logbook/7?x=1", "/panel/ariel/logbook/7?x=1"),
        ],
    )
    def test_a_url_within_the_panel_is_spelled_in_its_mount(self, url, expected):
        assert member_navigation_url(
            normalise_navigation_url(url), mount="/panel/ariel", upstream=_ARIEL_UPSTREAM
        ) == (expected, None)

    def test_the_upstream_base_path_is_stripped(self):
        kw = {"mount": "/panel/grafana", "upstream": _GRAFANA["url"]}
        assert member_navigation_url("http://grafana.lan:3000/grafana/d/abc", **kw) == (
            "/panel/grafana/d/abc",
            None,
        )
        admitted, _ = member_navigation_url("http://grafana.lan:3000/d/abc", **kw)
        assert admitted is None

    def test_a_default_port_matches_its_scheme(self):
        assert member_navigation_url(
            "http://grafana.lan/x", mount="/panel/g", upstream="http://grafana.lan:80"
        ) == ("/panel/g/x", None)

    def test_a_panel_without_a_server_admits_only_its_mount(self):
        kw = {"mount": "/panel-static/reference/", "upstream": None}
        assert member_navigation_url("/panel-static/reference/index.html?x=1", **kw) == (
            "/panel-static/reference/index.html?x=1",
            None,
        )
        assert member_navigation_url("/panel-static/reference", **kw) == (
            "/panel-static/reference/",
            None,
        )
        for url in ("/panel-static/ref/index.html", "http://127.0.0.1:10300/"):
            admitted, refusal = member_navigation_url(url, **kw)
            assert admitted is None
            assert "/panel-static/reference/" in refusal


def _member_app(**state) -> FastAPI:
    defaults = {
        "enabled_panels": {"ariel"},
        "custom_panels": [_GRAFANA, _REFERENCE],
        "visible_panels": ["ariel", "grafana", "reference"],
        "ariel_server_url": _ARIEL_UPSTREAM,
    }
    defaults.update(state)
    return _op_app(**defaults)


def _assert_refused_unchanged(app: FastAPI, resp: httpx.Response, frame, mount: str) -> None:
    assert resp.status_code == 422, resp.text
    assert resp.json()["reason"] == "invalid_url"
    assert mount in resp.json()["detail"]
    assert frame is None
    assert app.state.active_panel is None
    assert app.state.dock_layout_rev == 3
    assert len(app.state.agent_activity_buffer) == 0


@pytest.mark.usefixtures("prefix")
class TestPanelOpenMemberGuard:
    """The second step of the guard on ``open``: the url stays with its panel."""

    @pytest.mark.parametrize("path", OUTSIDE_ARIEL_PATHS)
    def test_a_path_outside_the_members_mount_is_refused_and_never_relayed(self, path):
        app = _member_app()

        resp, frame = _open(app, path, answer=False)

        _assert_refused_unchanged(app, resp, frame, "/panel/ariel")

    @pytest.mark.parametrize("url", FOREIGN_ORIGIN_URLS)
    def test_a_foreign_origin_is_refused_and_never_relayed(self, url):
        app = _member_app()

        resp, frame = _open(app, url, answer=False)

        _assert_refused_unchanged(app, resp, frame, "/panel/ariel")

    def test_another_members_mount_is_refused(self):
        app = _member_app()

        resp, frame = _open(app, "/panel/ariel/", answer=False, member="grafana")

        _assert_refused_unchanged(app, resp, frame, "/panel/grafana")

    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("http://127.0.0.1:10300/#draft", "/panel/ariel/#draft"),
            ("HTTP://127.0.0.1:10300/logbook/7?x=1", "/panel/ariel/logbook/7?x=1"),
            ("  http://127.0.0.1:10300/logbook/7  ", "/panel/ariel/logbook/7"),
        ],
    )
    def test_a_url_on_the_members_own_server_opens_through_its_mount(self, prefix, url, expected):
        app = _member_app()

        resp, frame = _open(app, url)

        assert resp.status_code == 200, resp.text
        assert frame["args"]["url"] == f"{prefix}{expected}"

    def test_a_custom_panels_server_url_opens_through_its_mount(self, prefix):
        app = _member_app()

        resp, frame = _open(app, "http://grafana.lan:3000/grafana/d/abc", member="grafana")

        assert resp.status_code == 200, resp.text
        assert frame["args"]["url"] == f"{prefix}/panel/grafana/d/abc"

    def test_a_discovered_panel_admits_only_its_bundle(self, prefix):
        app = _member_app()

        resp, frame = _open(app, "/panel-static/reference/index.html?x=1", member="reference")
        assert resp.status_code == 200, resp.text
        assert frame["args"]["url"] == f"{prefix}/panel-static/reference/index.html?x=1"

        for url in ("/panel-static/ref/index.html", "/panel/reference/", "http://127.0.0.1:10300/"):
            app = _member_app()
            resp, frame = _open(app, url, answer=False, member="reference")
            _assert_refused_unchanged(app, resp, frame, "/panel-static/reference/")

    def test_a_member_without_a_server_refuses_every_http_url(self):
        app = _member_app(ariel_server_url=None)

        resp, frame = _open(app, "http://127.0.0.1:10300/", answer=False)

        _assert_refused_unchanged(app, resp, frame, "/panel/ariel")

    def test_an_encoded_segment_is_relayed_in_its_canonical_spelling(self, prefix):
        app = _member_app()

        resp, frame = _open(app, "/panel/ariel/a%20b/c%3Fd?x=1#f")

        assert resp.status_code == 200, resp.text
        assert frame["args"]["url"] == f"{prefix}/panel/ariel/a%20b/c%3Fd?x=1#f"


# ---- /api/panels/register ----


@pytest.mark.usefixtures("prefix")
class TestRuntimeRegistrationGuard:
    @pytest.mark.parametrize(
        "url",
        [
            "javascript:alert(1)",
            " JavaScript:alert(1)",
            "\tjava\nscript:alert(1)",
            "data:text/html,x",
            "vbscript:x",
            "blob:http://grafana.lan/0b1c",
            "//grafana.lan/d",
            "/\\grafana.lan/d",
            "/panel/x",
            "",
        ],
    )
    def test_hostile_or_non_http_url_is_422(self, url):
        client = _client()

        with patch(_GETADDRINFO_TARGET, return_value=_LAN_ADDR):
            resp = client.post(
                "/api/panels/register", json={"id": "graf", "label": "Graf", "url": url}
            )

        assert resp.status_code == 422
        assert client.app.state.custom_panels == []
        client.app.state.broadcaster.broadcast.assert_not_called()

    @pytest.mark.parametrize(
        ("url", "stored"),
        [
            ("http://grafana.lan:3000", "http://grafana.lan:3000"),
            ("  HTTP://grafana.lan:3000 ", "http://grafana.lan:3000"),
            ("ht\ttp://grafana.lan:3000", "http://grafana.lan:3000"),
            ("https:\\\\grafana.lan\\d", "https://grafana.lan/d"),
        ],
    )
    def test_http_url_registers_in_normalised_form(self, prefix, url, stored):
        client = _client()

        with (
            patch(_GETADDRINFO_TARGET, return_value=_LAN_ADDR) as gai,
            patch(_OWN_ADDRS_TARGET, return_value=frozenset()),
        ):
            resp = client.post(
                "/api/panels/register", json={"id": "graf", "label": "Graf", "url": url}
            )

        assert resp.status_code == 200, resp.text
        assert resp.json()["url"] == f"{prefix}/panel/graf"
        assert client.app.state.custom_panels[0]["url"] == stored
        # The host resolved is the one the normalised URL names.
        assert gai.call_args[0][0] == "grafana.lan"


# ---- open_panel (MCP) ----


def _open_panel():
    from osprey.mcp_server.workspace.tools.panel_tools import open_panel

    return get_tool_fn(open_panel)


_PANEL_OP = "osprey.mcp_server.workspace.tools.panel_tools.panel_op"
_APPLIED = {"ok": True, "layout": _LAYOUT, "popouts": [], "rev": 4}


class TestOpenPanelGuard:
    @pytest.mark.parametrize(
        "url", ["javascript:alert(1)", " DATA:text/html,x", "//evil.example/x"]
    )
    async def test_refused_url_is_an_error_and_never_posted(self, url):
        with patch(_PANEL_OP) as op:
            with assert_raises_error(error_type="invalid_url") as ctx:
                await _open_panel()("ariel", url=url)

        assert "URL refused" in ctx["envelope"]["error_message"]
        op.assert_not_called()

    @pytest.mark.parametrize("url", [None, "/panel/ariel/#/draft/7", "http://127.0.0.1:10300/#d"])
    async def test_navigable_url_is_posted(self, url):
        with patch(_PANEL_OP, return_value=_APPLIED) as op:
            result = json.loads(await _open_panel()("ariel", url=url))

        assert result["status"] == "success"
        assert result["panel"] == "ariel"
        assert result["rev"] == 4
        args = {"member": "ariel"} if url is None else {"member": "ariel", "url": url}
        op.assert_called_once_with("open", args, base_rev=None)
