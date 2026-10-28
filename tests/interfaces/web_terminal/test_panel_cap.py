"""C-CAP: an opaque panel reads its own files through a capability path.

An agent-writable panel — a discovered bundle at ``/panel-static/<id>/`` or a
runtime registration proxied at ``/panel/<id>/`` — runs at an opaque origin, so
its own requests carry ``Origin: null`` and no session cookie. The hub frames it
at ``…/<id>/~c/<cap>/`` instead, and the cap admits those requests in place of
the cookie. These tests pin what a cap grants and what it never does:

* a valid cap reads only its own panel; a forged, expired, other-panel or
  foreign-login cap answers ``403``, and a cap path never serves another
  panel's bytes;
* cap responses answer ``Access-Control-Allow-Origin: *`` without credentials
  and carry ``opaque_headers()`` on every non-inert type;
* the proxy's root-path rewrite and forwarded prefix target the cap prefix, and
  nothing credential-bearing reaches the backend;
* a cap websocket reaches its own panel's backend from ``Origin: null``, while
  an ``Origin: null`` websocket without a cap is still refused;
* the hub mints the cap for the login that asked, and only for opaque panels.

Requests that must succeed WITHOUT a credential carry ``no_auth_seam``: the
suite-wide seam would otherwise authenticate every TestClient request, and a
pass would prove nothing about the exemption.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
from starlette.websockets import WebSocketDisconnect

from osprey.interfaces.common_middleware import (
    PANEL_CAP_SEGMENT,
    is_exempt_path,
    is_panel_asset_path,
    is_panel_cap_path,
    session_cookie_name,
)
from osprey.interfaces.opaque_origin import OPAQUE_CSP
from osprey.interfaces.web_auth import WebCredentials, get_web_credentials
from osprey.interfaces.web_terminal import panel_cap
from osprey.interfaces.web_terminal.panel_cap import (
    CAP_TTL_SECONDS,
    OPERATOR_LOGIN,
    mint_cap,
    verify_cap,
)
from osprey.interfaces.web_terminal.panel_discovery import DiscoveredPanel

from ._proxy_fakes import _lower, panel_app
from .test_proxy_panel import _echo_upstream

ACAO = "access-control-allow-origin"
ACAC = "access-control-allow-credentials"
NULL_ORIGIN = {"Origin": "null"}

_BUNDLE_HTML = (
    '<!doctype html><html><head><script type="module" src="./app.js"></script></head>'
    '<body><div id="out">ALPHA</div></body></html>'
)
_BUNDLE_JS = "fetch('./data.json').then(r => r.json()).then(d => { out.textContent = d.word; });"
_BUNDLE_JSON = {"word": "alpha-data"}


def _credentials(secret: str = "s" * 43) -> WebCredentials:
    return WebCredentials(operator_secret=secret, panel_token="p" * 43)


# ---------------------------------------------------------------------------
# The cap itself
# ---------------------------------------------------------------------------


class TestMintAndVerify:
    def test_a_minted_cap_verifies_for_its_panel(self):
        creds = _credentials()
        cap = mint_cap(creds, OPERATOR_LOGIN, "alpha")
        assert verify_cap(creds, cap, "alpha")

    def test_a_cap_is_one_path_segment(self):
        cap = mint_cap(_credentials(), OPERATOR_LOGIN, "alpha")
        assert "/" not in cap
        assert is_panel_cap_path(f"/panel/alpha/{PANEL_CAP_SEGMENT}/{cap}/x")

    def test_another_panels_cap_is_refused(self):
        creds = _credentials()
        cap = mint_cap(creds, OPERATOR_LOGIN, "alpha")
        assert not verify_cap(creds, cap, "beta")

    def test_an_expired_cap_is_refused(self):
        creds = _credentials()
        cap = mint_cap(creds, OPERATOR_LOGIN, "alpha", ttl=-1)
        assert not verify_cap(creds, cap, "alpha")

    def test_a_cap_expires_at_its_deadline(self):
        creds = _credentials()
        cap = mint_cap(creds, OPERATOR_LOGIN, "alpha", now=1000.0, ttl=60)
        assert verify_cap(creds, cap, "alpha", now=1059.0)
        assert not verify_cap(creds, cap, "alpha", now=1060.0)

    def test_a_cap_promising_more_than_twelve_hours_is_refused(self):
        creds = _credentials()
        cap = mint_cap(creds, OPERATOR_LOGIN, "alpha", ttl=CAP_TTL_SECONDS + 60)
        assert not verify_cap(creds, cap, "alpha")
        assert CAP_TTL_SECONDS == 12 * 60 * 60

    @pytest.mark.parametrize(
        "mutate",
        [
            lambda c: c[:-2] + ("AA" if not c.endswith("AA") else "BB"),  # signature
            lambda c: f"{int(c.split('.')[0], 16) + 3600:x}." + c.split(".", 1)[1],  # expiry
            lambda c: c.split(".")[0] + ".AAAAAAAAAAAAAAAA." + c.split(".")[2],  # login tag
            lambda c: c + ".x",
            lambda c: c.replace(".", ""),
            lambda c: "zz." + c.split(".", 1)[1],
            lambda c: "0" + c,  # same expiry value, different spelling
            lambda c: "",
        ],
        ids=["sig", "expiry", "login", "extra-part", "no-parts", "bad-hex", "padded", "empty"],
    )
    def test_a_forged_cap_is_refused(self, mutate):
        creds = _credentials()
        cap = mint_cap(creds, OPERATOR_LOGIN, "alpha")
        assert not verify_cap(creds, mutate(cap), "alpha")

    def test_a_cap_from_another_deployments_secret_is_refused(self):
        """Another user's container holds another secret: its caps fail here."""
        theirs = mint_cap(_credentials("t" * 43), OPERATOR_LOGIN, "alpha")
        assert not verify_cap(_credentials(), theirs, "alpha")

    def test_a_session_login_lives_only_as_long_as_its_session(self):
        creds = _credentials()
        session_id = creds.create_session(persist=False)
        login = _session_login(session_id)
        cap = mint_cap(creds, login, "alpha")
        assert verify_cap(creds, cap, "alpha")

        creds.revoke_session(session_id)
        assert not verify_cap(creds, cap, "alpha")

    def test_a_login_this_process_never_issued_is_refused(self):
        """A foreign login: a session id that is not one of this process's."""
        creds = _credentials()
        creds.create_session(persist=False)
        cap = mint_cap(creds, _session_login("not-a-session-of-this-process"), "alpha")
        assert not verify_cap(creds, cap, "alpha")

    def test_an_expired_session_login_is_refused(self):
        creds = _credentials()
        session_id = creds.create_session(ttl_seconds=-1, persist=False)
        cap = mint_cap(creds, _session_login(session_id), "alpha")
        assert not verify_cap(creds, cap, "alpha")


def _session_login(session_id: str) -> str:
    """The login :func:`panel_cap.login_for_request` names for a session cookie."""
    from osprey.interfaces.web_auth import _digest

    return _digest(session_id)


# ---------------------------------------------------------------------------
# The gate exemption
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "/panel/alpha/~c/1a2b.tag.sig/",
        "/panel/alpha/~c/1a2b.tag.sig/index.html",
        "/panel/alpha/~c/1a2b.t-g_.s-g_/ws",
        "/panel-static/alpha/~c/1a2b.tag.sig/",
        "/panel-static/alpha/~c/1a2b.tag.sig/js/app.js",
    ],
)
def test_cap_paths_are_recognised(path):
    assert is_panel_cap_path(path)


@pytest.mark.parametrize(
    "path",
    [
        "/panel/alpha/~c/1a2b.tag.sig",  # bare: not the minted form
        "/panel/alpha/~c//x",
        "/panel/alpha/~c/../api/panels",
        "/panel/alpha/~c/1a2b/../../../api/panels",
        "/panel/alpha/~d/1a2b/x",
        "/panel/alpha/c/1a2b/x",
        "/panel/alpha/index.html",
        "/panel-static/alpha/index.html",
        "/panel/.x/~c/1a2b/x",
        "/panel/alpha/~c/-x/y",
        "/api/panel/alpha/~c/1a2b/x",
        "/u/alice/panel/alpha/~c/1a2b/x",
        "/files/alpha/~c/1a2b/x",
        "panel/alpha/~c/1a2b/x",
    ],
)
def test_other_paths_are_not_cap_paths(path):
    assert not is_panel_cap_path(path)


def test_a_cap_path_is_neither_c_assets_nor_a_static_mount():
    """The exemption is its own: nothing else in the gate opened these paths."""
    path = "/panel/alpha/~c/1a2b.tag.sig/static/js/x.js"
    assert not is_exempt_path(path)
    assert not is_panel_asset_path(path)


# ---------------------------------------------------------------------------
# Discovered bundles: /panel-static/<id>/~c/<cap>/…
# ---------------------------------------------------------------------------


def _bundle(root: Path, panel_id: str, word: str) -> DiscoveredPanel:
    directory = root / panel_id
    directory.mkdir(parents=True)
    (directory / "index.html").write_text(_BUNDLE_HTML.replace("ALPHA", word.upper()))
    (directory / "app.js").write_text(_BUNDLE_JS)
    (directory / "data.json").write_text(json.dumps({"word": f"{word}-data"}))
    (directory / "manifest.json").write_text(json.dumps({"id": panel_id, "entry": "index.html"}))
    return DiscoveredPanel(id=panel_id, label=word.upper(), entry="index.html", directory=directory)


def _discovered_entry(panel_id: str) -> dict:
    return {
        "id": panel_id,
        "label": panel_id.upper(),
        "url": f"/panel-static/{panel_id}/",
        "healthEndpoint": None,
        "path": "/",
        "discovered": True,
    }


_CONFIG_PANEL = {
    "id": "facility-dash",
    "label": "DASH",
    "url": "http://127.0.0.1:9",
    "path": "/",
    "configDefined": True,
}
_RUNTIME_PANEL = {"id": "agent-dash", "label": "AGENT", "url": "http://127.0.0.1:9", "path": "/"}


@pytest.fixture
def hub(tmp_path):
    """The whole app with two discovered bundles, a runtime and a config panel."""
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()
    bundles = tmp_path / "panels"
    alpha = _bundle(bundles, "alpha", "alpha")
    beta = _bundle(bundles, "beta", "beta")
    custom = [_discovered_entry("alpha"), _discovered_entry("beta"), _CONFIG_PANEL, _RUNTIME_PANEL]
    for app, client in panel_app(workspace, custom):
        app.state.discovered_panel_dirs = {"alpha": alpha, "beta": beta}
        yield app, client


def _cap(app, panel_id: str, **kwargs) -> str:
    return mint_cap(get_web_credentials(app), OPERATOR_LOGIN, panel_id, **kwargs)


def _forged(cap: str) -> str:
    """``cap`` with its signature's last two characters changed, never to themselves.

    A fixed replacement equals the real tail for one signature in 4096, and that
    cap is then valid.
    """
    return cap[:-2] + ("zz" if not cap.endswith("zz") else "AA")


def _static(panel_id: str, cap: str, file: str = "") -> str:
    return f"/panel-static/{panel_id}/{PANEL_CAP_SEGMENT}/{cap}/{file}"


def _assert_cap_cors(resp: httpx.Response) -> None:
    assert resp.headers.get(ACAO) == "*"
    assert ACAC not in resp.headers
    assert "set-cookie" not in resp.headers


class TestStaticCapRoute:
    @pytest.mark.no_auth_seam
    def test_a_valid_cap_reads_the_whole_bundle_without_a_cookie(self, hub):
        app, client = hub
        cap = _cap(app, "alpha")

        entry = client.get(_static("alpha", cap), headers=NULL_ORIGIN)
        module = client.get(_static("alpha", cap, "app.js"), headers=NULL_ORIGIN)
        data = client.get(_static("alpha", cap, "data.json"), headers=NULL_ORIGIN)

        assert entry.status_code == 200
        assert "ALPHA" in entry.text
        assert module.status_code == 200
        assert module.text == _BUNDLE_JS
        assert data.status_code == 200
        assert data.json() == {"word": "alpha-data"}
        for resp in (entry, module, data):
            _assert_cap_cors(resp)

    @pytest.mark.no_auth_seam
    def test_documents_are_opaque_and_inert_types_get_nosniff(self, hub):
        app, client = hub
        cap = _cap(app, "alpha")

        entry = client.get(_static("alpha", cap, "index.html"))
        module = client.get(_static("alpha", cap, "app.js"))
        data = client.get(_static("alpha", cap, "data.json"))

        assert entry.headers["content-security-policy"] == OPAQUE_CSP
        assert entry.headers["x-content-type-options"] == "nosniff"
        assert module.headers["content-security-policy"] == OPAQUE_CSP
        assert "content-security-policy" not in data.headers
        assert data.headers["x-content-type-options"] == "nosniff"

    @pytest.mark.no_auth_seam
    def test_without_a_cap_the_bundle_stays_behind_the_gate(self, hub):
        _app, client = hub
        assert client.get("/panel-static/alpha/index.html").status_code == 401
        assert client.get("/panel-static/alpha/~c/").status_code == 401

    @pytest.mark.no_auth_seam
    @pytest.mark.parametrize(
        "make",
        [
            lambda app: _cap(app, "beta"),
            lambda app: _cap(app, "alpha", ttl=-1),
            lambda app: _forged(_cap(app, "alpha")),
            lambda app: mint_cap(_credentials("t" * 43), OPERATOR_LOGIN, "alpha"),
            lambda app: mint_cap(get_web_credentials(app), _session_login("foreign"), "alpha"),
        ],
        ids=["other-panel", "expired", "forged", "foreign-deployment", "foreign-login"],
    )
    def test_an_invalid_cap_is_forbidden(self, hub, make):
        app, client = hub
        resp = client.get(_static("alpha", make(app), "index.html"), headers=NULL_ORIGIN)
        assert resp.status_code == 403
        assert "ALPHA" not in resp.text

    @pytest.mark.no_auth_seam
    def test_a_cap_path_never_serves_another_panels_bytes(self, hub):
        app, client = hub
        alpha_cap = _cap(app, "alpha")

        # The other panel's id under this cap: refused, not served.
        assert client.get(_static("beta", alpha_cap, "data.json")).status_code == 403
        # Climbing out of the bundle by an encoded traversal: not a cap path at all.
        climb = client.get(_static("alpha", alpha_cap, "%2e%2e/beta/data.json"))
        assert climb.status_code in (401, 404)
        assert "beta-data" not in climb.text
        # A symlink-free sibling name is just a missing file in this bundle.
        assert client.get(_static("alpha", alpha_cap, "beta/data.json")).status_code == 404

    @pytest.mark.no_auth_seam
    def test_a_cap_never_crosses_into_the_proxy_tree(self, hub):
        """A discovered panel's cap does not open ``/panel/<id>/~c/…``."""
        app, client = hub
        app.state.proxy_client.request = AsyncMock(
            side_effect=AssertionError("a static cap reached a backend")
        )
        resp = client.get(f"/panel/alpha/{PANEL_CAP_SEGMENT}/{_cap(app, 'alpha')}/index.html")
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Runtime panels: /panel/<id>/~c/<cap>/…
# ---------------------------------------------------------------------------


def _proxied(panel_id: str, cap: str, path: str = "") -> str:
    return f"/panel/{panel_id}/{PANEL_CAP_SEGMENT}/{cap}/{path}"


class _Upstream:
    """A recording ``proxy_client.request`` double."""

    def __init__(self, *, body: str, content_type: str, headers: dict | None = None):
        self.calls: list[dict] = []
        self._body = body
        self._headers = {"content-type": content_type, **(headers or {})}

    def __call__(self, *, method, url, headers, content, follow_redirects=True):  # noqa: ARG002
        self.calls.append({"method": method, "url": url, "headers": _lower(headers)})
        return httpx.Response(status_code=200, text=self._body, headers=self._headers)


class TestProxyCapRoute:
    @pytest.mark.no_auth_seam
    def test_a_valid_cap_relays_to_the_panels_backend_without_credentials(self, hub):
        app, client = hub
        upstream = _Upstream(body="<html><body>RUNTIME</body></html>", content_type="text/html")
        app.state.proxy_client.request = AsyncMock(side_effect=upstream)
        cap = _cap(app, "agent-dash")

        resp = client.get(
            _proxied("agent-dash", cap, "view?q=1"),
            headers={**NULL_ORIGIN, "Cookie": "x=1", "Authorization": "Bearer y"},
        )

        assert resp.status_code == 200
        assert "RUNTIME" in resp.text
        (call,) = upstream.calls
        assert call["url"] == "http://127.0.0.1:9/view?q=1"
        assert call["headers"]["x-forwarded-prefix"] == f"/panel/agent-dash/~c/{cap}"
        for name in ("cookie", "authorization", "x-osprey-terminal-secret", "origin"):
            assert name not in call["headers"], name
        _assert_cap_cors(resp)
        assert OPAQUE_CSP in resp.headers.get_list("content-security-policy")

    @pytest.mark.no_auth_seam
    def test_the_root_path_rewrite_targets_the_cap_prefix(self, hub):
        app, client = hub
        body = (
            '<script type="module" src="/static/js/app.js"></script>'
            "<script>fetch('/api/data')</script><a href=\"/\">home</a>"
        )
        app.state.proxy_client.request = AsyncMock(
            side_effect=_Upstream(body=body, content_type="text/html")
        )
        cap = _cap(app, "agent-dash")

        text = client.get(_proxied("agent-dash", cap)).text

        base = f"/panel/agent-dash/~c/{cap}"
        assert f'src="{base}/static/js/app.js"' in text
        assert f"fetch('{base}/api/data')" in text
        assert f'href="{base}/"' in text
        assert '"/panel/agent-dash/static' not in text

    @pytest.mark.no_auth_seam
    def test_the_upstreams_own_cors_and_csp(self, hub):
        """Upstream CORS never survives; its CSP stands beside the sandbox."""
        app, client = hub
        app.state.proxy_client.request = AsyncMock(
            side_effect=_Upstream(
                body="<p>x</p>",
                content_type="text/html",
                headers={
                    "access-control-allow-origin": "https://attacker.test",
                    "access-control-allow-credentials": "true",
                    "content-security-policy": "default-src 'self'",
                    "set-cookie": "osprey_terminal_session=evil",
                },
            )
        )
        resp = client.get(_proxied("agent-dash", _cap(app, "agent-dash")))
        _assert_cap_cors(resp)
        policies = resp.headers.get_list("content-security-policy")
        assert OPAQUE_CSP in policies
        assert "default-src 'self'" in policies

    @pytest.mark.no_auth_seam
    def test_a_json_answer_is_inert(self, hub):
        app, client = hub
        app.state.proxy_client.request = AsyncMock(
            side_effect=_Upstream(body='{"a": 1}', content_type="application/json")
        )
        resp = client.post(
            _proxied("agent-dash", _cap(app, "agent-dash"), "api/save"),
            json={"a": 1},
            headers=NULL_ORIGIN,
        )
        assert resp.status_code == 200
        _assert_cap_cors(resp)
        assert "content-security-policy" not in resp.headers
        assert resp.headers["x-content-type-options"] == "nosniff"

    @pytest.mark.no_auth_seam
    @pytest.mark.parametrize(
        "make",
        [
            lambda app: _cap(app, "alpha"),
            lambda app: _cap(app, "agent-dash", ttl=-1),
            lambda app: _forged(_cap(app, "agent-dash")),
            lambda app: mint_cap(_credentials("t" * 43), OPERATOR_LOGIN, "agent-dash"),
            lambda app: mint_cap(get_web_credentials(app), _session_login("foreign"), "agent-dash"),
        ],
        ids=["other-panel", "expired", "forged", "foreign-deployment", "foreign-login"],
    )
    def test_an_invalid_cap_is_forbidden_before_any_upstream_call(self, hub, make):
        app, client = hub
        app.state.proxy_client.request = AsyncMock(
            side_effect=AssertionError("an invalid cap reached the backend")
        )
        assert client.get(_proxied("agent-dash", make(app))).status_code == 403

    @pytest.mark.no_auth_seam
    @pytest.mark.parametrize("panel_id", ["facility-dash", "artifacts", "jupyter"])
    def test_first_party_panels_are_never_served_under_a_cap(self, hub, panel_id):
        """A config or built-in panel earns credentials upstream: no cap opens it."""
        app, client = hub
        app.state.proxy_client.request = AsyncMock(
            side_effect=AssertionError("a first-party panel was reached by a cap")
        )
        assert client.get(_proxied(panel_id, _cap(app, panel_id))).status_code == 403

    @pytest.mark.no_auth_seam
    def test_without_a_cap_the_runtime_panel_stays_behind_the_gate(self, hub):
        _app, client = hub
        assert client.get("/panel/agent-dash/").status_code == 401
        assert client.get("/panel/agent-dash/~c/x").status_code == 401


# ---------------------------------------------------------------------------
# Websockets
# ---------------------------------------------------------------------------


class TestCapWebsocket:
    @pytest.mark.no_auth_seam
    def test_a_cap_websocket_reaches_its_own_sidecar_from_a_null_origin(self, tmp_path):
        workspace = tmp_path / "_agent_data"
        workspace.mkdir()
        with _echo_upstream() as port:
            custom = [{"id": "agent-ws", "label": "WS", "url": f"http://127.0.0.1:{port}"}]
            for app, client in panel_app(workspace, custom):
                cap = _cap(app, "agent-ws")
                with client.websocket_connect(
                    _proxied("agent-ws", cap, "ws"), headers=NULL_ORIGIN
                ) as session:
                    session.send_text("ping")
                    assert session.receive_text() == "ping"

    @pytest.mark.no_auth_seam
    def test_a_null_origin_websocket_without_a_cap_is_refused(self, tmp_path):
        workspace = tmp_path / "_agent_data"
        workspace.mkdir()
        with _echo_upstream() as port:
            custom = [{"id": "agent-ws", "label": "WS", "url": f"http://127.0.0.1:{port}"}]
            for _app, client in panel_app(workspace, custom):
                with (
                    pytest.raises(WebSocketDisconnect),
                    client.websocket_connect("/panel/agent-ws/ws", headers=NULL_ORIGIN),
                ):
                    pass

    def test_a_null_origin_websocket_with_the_operators_credential_is_refused(self, tmp_path):
        """The seam's operator credential does not lift the Origin rule either."""
        workspace = tmp_path / "_agent_data"
        workspace.mkdir()
        with _echo_upstream() as port:
            custom = [{"id": "agent-ws", "label": "WS", "url": f"http://127.0.0.1:{port}"}]
            for _app, client in panel_app(workspace, custom):
                with (
                    pytest.raises(WebSocketDisconnect),
                    client.websocket_connect("/panel/agent-ws/ws", headers=NULL_ORIGIN),
                ):
                    pass

    @pytest.mark.no_auth_seam
    @pytest.mark.parametrize("which", ["other-panel", "expired", "forged"])
    def test_an_invalid_cap_websocket_is_refused(self, tmp_path, which):
        workspace = tmp_path / "_agent_data"
        workspace.mkdir()
        with _echo_upstream() as port:
            custom = [
                {"id": "agent-ws", "label": "WS", "url": f"http://127.0.0.1:{port}"},
                {"id": "other-ws", "label": "WS2", "url": f"http://127.0.0.1:{port}"},
            ]
            for app, client in panel_app(workspace, custom):
                if which == "other-panel":
                    cap = _cap(app, "other-ws")
                elif which == "expired":
                    cap = _cap(app, "agent-ws", ttl=-1)
                else:
                    cap = _forged(_cap(app, "agent-ws"))
                with (
                    pytest.raises(WebSocketDisconnect),
                    client.websocket_connect(_proxied("agent-ws", cap, "ws"), headers=NULL_ORIGIN),
                ):
                    pass


# ---------------------------------------------------------------------------
# Minting
# ---------------------------------------------------------------------------


class TestMinting:
    @pytest.mark.parametrize(
        ("panel_id", "prefix"),
        [("alpha", "/panel-static/alpha/~c/"), ("agent-dash", "/panel/agent-dash/~c/")],
    )
    def test_the_hub_mints_a_cap_url_for_each_opaque_panel(self, hub, panel_id, prefix):
        _app, client = hub
        body = client.get(f"/api/panels/{panel_id}/cap").json()
        assert body["id"] == panel_id
        assert body["cap_url"].startswith(prefix)
        assert body["cap_url"].endswith("/")

    def test_get_panels_is_unchanged(self, hub):
        """A cap is per login, so the shared panel payload never carries one."""
        _app, client = hub
        custom = {cp["id"]: cp for cp in client.get("/api/panels").json()["custom"]}
        assert custom["alpha"]["url"] == "/panel-static/alpha/"
        assert all("cap" not in key for cp in custom.values() for key in cp)

    def test_the_minted_url_serves_the_panel(self, hub):
        _app, client = hub
        url = client.get("/api/panels/alpha/cap").json()["cap_url"]
        resp = client.get(f"{url}data.json")
        assert resp.status_code == 200
        assert resp.json() == {"word": "alpha-data"}

    @pytest.mark.parametrize("panel_id", ["facility-dash", "artifacts", "no-such-panel"])
    def test_no_cap_is_minted_for_a_first_party_or_unknown_panel(self, hub, panel_id):
        _app, client = hub
        assert client.get(f"/api/panels/{panel_id}/cap").status_code == 404

    @pytest.mark.no_auth_seam
    def test_minting_needs_a_login(self, hub):
        _app, client = hub
        assert client.get("/api/panels/alpha/cap").status_code == 401

    @pytest.mark.no_auth_seam
    def test_a_cookie_login_mints_a_cap_that_dies_with_the_session(self, hub):
        app, client = hub
        creds = get_web_credentials(app)
        session_id = creds.create_session(persist=False)
        cookie = {"Cookie": f"{session_cookie_name()}={session_id}"}

        url = client.get("/api/panels/alpha/cap", headers=cookie).json()["cap_url"]
        assert client.get(f"{url}data.json").status_code == 200

        creds.revoke_session(session_id)
        assert client.get(f"{url}data.json").status_code == 403

    def test_the_cap_url_follows_the_per_user_prefix(self, hub, monkeypatch):
        _app, client = hub
        monkeypatch.setattr(panel_cap, "compute_url_prefix", lambda: "/u/alice")
        url = client.get("/api/panels/agent-dash/cap").json()["cap_url"]
        assert url.startswith("/u/alice/panel/agent-dash/~c/")
