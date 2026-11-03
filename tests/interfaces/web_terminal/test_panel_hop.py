"""A sidecar that opts in is reachable by the agent at its MCP transport.

A sidecar may serve an MCP transport beside its pages (``SidecarPanel.mcp_path``).
Declaring the path alone keeps it closed to every caller. Declaring
``agent_hop`` as well opens exactly one door:

* the terminal's gate places ``GET|POST|DELETE /panel/<id><mcp_path>`` in the
  panel tier — three exact pairs and nothing beside them;
* the proxy forwards that request only when the gate admitted it on the panel
  token. A browser session or the operator secret gets 403 there, because the
  transport belongs to the agent's client and not to a page;
* what is forwarded is the conversation as sent: the answer is relayed byte for
  byte, and nothing the caller used to authenticate at the terminal — the
  bearer, a cookie, the operator secret — travels upstream. The sidecar's own
  launch credential is the one credential the upstream sees.

Every other spelling of the path stays refused, whoever asks.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from osprey.interfaces import web_auth
from osprey.interfaces.common_middleware import (
    OPERATOR_SECRET_HEADER,
    PANEL_ADMISSION_STATE_KEY,
    WebAuthMiddleware,
)
from osprey.interfaces.web_auth import WebCredentials, reset_web_credentials
from osprey.interfaces.web_terminal.routes import proxy
from osprey.profiles import web_panels
from osprey.profiles.web_panels import SidecarPanel
from osprey.registry.web import panel_url_state_attr
from osprey_connectors.session_hold import RECORD_NAME, ensure_session, open_flag

from ._proxy_fakes import _FakeConnect, _FakeStreamResponse, _lower, _patch_connect

BACKEND_URL = "http://127.0.0.1:9500"

#: Declares a transport and opts in to the agent hop.
HOP_PANEL = "agent-hop"
#: Declares a transport and does not opt in.
CLOSED_PANEL = "mcp-no-hop"

HOP_ROUTE = f"/panel/{HOP_PANEL}/mcp"
HOP_PAIRS = {("GET", HOP_ROUTE), ("POST", HOP_ROUTE), ("DELETE", HOP_ROUTE)}

PANEL_TOKEN = "panel-token-value"
OPERATOR_SECRET = "operator-secret-value"
#: The panel token of some other user's terminal.
FOREIGN_PANEL_TOKEN = "panel-token-of-another-terminal"
COOKIE_NAME = "osprey_terminal_session_8080"

#: The sidecar's own launch credential, as the launcher publishes it.
LAUNCH_HEADER = "x-test-sidecar-access"
LAUNCH_CREDENTIAL = "launch-credential-value"

MCP_ACCEPT = "application/json, text/event-stream"
MCP_CALL_BODY = b'{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'
#: Carries a root-absolute path in quotes, the shape the page rewrite moves.
MCP_ANSWER_BODY = (
    b'{"jsonrpc":"2.0","id":1,"result":{"tools":[{"name":"get_screen",'
    b'"inputSchema":{"$ref":"/api/schema"}}]}}'
)

BEARER = {"authorization": f"Bearer {PANEL_TOKEN}"}


def _declare(monkeypatch, panel_id: str, **fields) -> None:
    """Put one synthetic sidecar into the registry every reader shares."""
    monkeypatch.setitem(
        web_panels.SIDECAR_PANELS,
        panel_id,
        SidecarPanel("a test sidecar", "module:Class", **fields),
    )
    monkeypatch.setitem(proxy._PANEL_STATE_MAP, panel_id, panel_url_state_attr(panel_id))


@pytest.fixture
def sidecars(monkeypatch):
    """One sidecar that opted in, one that only declared its transport."""
    _declare(
        monkeypatch,
        HOP_PANEL,
        mcp_path="/mcp",
        agent_hop=True,
        mcp_allow=("get_screen",),
        mcp_ask=("push_screen",),
    )
    _declare(monkeypatch, CLOSED_PANEL, mcp_path="/mcp")


class TestTheRouteTable:
    """What the opt-in adds to the panel tier: three exact pairs."""

    def test_with_no_opted_in_panel_the_table_is_the_static_one(self):
        # The shipped registry has no panel that opts in.
        assert web_auth.panel_tier_routes() == web_auth.PANEL_TIER_ROUTES

    @pytest.mark.usefixtures("sidecars")
    def test_opting_in_opens_exactly_the_three_pairs(self):
        added = web_auth.panel_tier_routes() - web_auth.PANEL_TIER_ROUTES
        assert added == HOP_PAIRS
        assert web_auth.PANEL_TIER_ROUTES <= web_auth.panel_tier_routes()

    @pytest.mark.usefixtures("sidecars")
    @pytest.mark.parametrize("method", ["GET", "POST", "DELETE"])
    def test_the_gate_classifies_the_three_pairs_as_panel_tier(self, method):
        assert web_auth.classify(method, HOP_ROUTE, True) is web_auth.Tier.PANEL

    @pytest.mark.usefixtures("sidecars")
    @pytest.mark.parametrize("method", ["PUT", "PATCH", "HEAD", "OPTIONS", "TRACE"])
    def test_any_other_method_stays_operator_only(self, method):
        assert web_auth.classify(method, HOP_ROUTE, False) is web_auth.Tier.OPERATOR

    @pytest.mark.usefixtures("sidecars")
    @pytest.mark.parametrize(
        "path",
        [
            f"{HOP_ROUTE}/",
            f"{HOP_ROUTE}/x",
            f"{HOP_ROUTE}?x=1",
            f"/panel/{HOP_PANEL}",
            f"/panel/{HOP_PANEL}/",
            f"/panel/{HOP_PANEL}/MCP",
            f"/panel/{HOP_PANEL}//mcp",
            f"/panel/{HOP_PANEL}/page",
            f"/u/alice{HOP_ROUTE}",
        ],
    )
    def test_the_match_is_exact_and_never_a_prefix(self, path):
        assert web_auth.classify("POST", path, False) is web_auth.Tier.OPERATOR

    @pytest.mark.usefixtures("sidecars")
    def test_a_transport_without_the_opt_in_opens_nothing(self):
        closed = f"/panel/{CLOSED_PANEL}/mcp"

        assert not {path for _, path in web_auth.panel_tier_routes() if CLOSED_PANEL in path}
        for method in ("GET", "POST", "DELETE"):
            assert web_auth.classify(method, closed, False) is web_auth.Tier.OPERATOR

    @pytest.mark.parametrize("mcp_path", [None, "", "mcp", "x/mcp"])
    def test_an_opt_in_without_a_rooted_transport_path_opens_nothing(self, monkeypatch, mcp_path):
        """No path, or one that would splice into the panel id, adds no route."""
        before = web_auth.panel_tier_routes()
        _declare(monkeypatch, HOP_PANEL, mcp_path=mcp_path, agent_hop=True)

        assert web_auth.panel_tier_routes() == before
        assert not {path for _, path in web_auth.panel_tier_routes() if HOP_PANEL in path}

    def test_the_table_is_derived_at_call_time(self, monkeypatch):
        before = web_auth.panel_tier_routes()
        _declare(monkeypatch, HOP_PANEL, mcp_path="/mcp", agent_hop=True)

        assert web_auth.panel_tier_routes() - before == HOP_PAIRS

    @pytest.mark.usefixtures("sidecars")
    def test_the_answer_is_immutable_and_leaves_the_static_table_alone(self):
        static = set(web_auth.PANEL_TIER_ROUTES)
        routes = web_auth.panel_tier_routes()

        assert isinstance(routes, frozenset)
        assert set(web_auth.PANEL_TIER_ROUTES) == static
        assert not HOP_PAIRS & web_auth.PANEL_TIER_ROUTES


class _Scope:
    """What the proxy reads of a request to decide the hop."""

    def __init__(self, raw_path: bytes | None, *, admitted: bool):
        self.scope: dict[str, object] = {}
        if raw_path is not None:
            self.scope["raw_path"] = raw_path
        if admitted:
            self.scope["state"] = {PANEL_ADMISSION_STATE_KEY: True}


@pytest.mark.usefixtures("sidecars")
class TestTheRule:
    """``_mcp_path_refusal`` for a sidecar that opted in."""

    @staticmethod
    def _refusal(
        path: str,
        *,
        admitted: bool,
        panel_id: str = HOP_PANEL,
        raw_path: bytes | None = HOP_ROUTE.encode(),
    ):
        scope = _Scope(raw_path, admitted=admitted)
        return proxy._mcp_path_refusal(scope, panel_id, path, f"{BACKEND_URL}/{path}")

    def test_the_declared_hop_passes_for_a_panel_token_admission(self):
        assert self._refusal("mcp", admitted=True) is None

    def test_the_declared_hop_is_forbidden_without_that_admission(self):
        assert self._refusal("mcp", admitted=False) == 403

    def test_a_request_with_no_state_at_all_is_not_admitted(self):
        scope = _Scope(HOP_ROUTE.encode(), admitted=False)
        assert "state" not in scope.scope

        assert proxy._mcp_path_refusal(scope, HOP_PANEL, "mcp", f"{BACKEND_URL}/mcp") == 403

    @pytest.mark.parametrize("value", [False, None, 1, "true", "yes"])
    def test_only_the_gates_own_mark_counts(self, value):
        scope = _Scope(HOP_ROUTE.encode(), admitted=False)
        scope.scope["state"] = {PANEL_ADMISSION_STATE_KEY: value}

        assert proxy._mcp_path_refusal(scope, HOP_PANEL, "mcp", f"{BACKEND_URL}/mcp") == 403

    def test_the_admission_opens_nothing_on_a_panel_that_did_not_opt_in(self):
        raw_path = f"/panel/{CLOSED_PANEL}/mcp".encode()

        assert self._refusal("mcp", admitted=True, panel_id=CLOSED_PANEL, raw_path=raw_path) == 403

    @pytest.mark.parametrize(
        "raw_path",
        [
            f"/panel/{HOP_PANEL}/%6dcp".encode(),
            f"/panel/{HOP_PANEL}/%6Dcp".encode(),
            f"/panel/{HOP_PANEL}/m%63p".encode(),
            f"/panel/{HOP_PANEL}/mcp/".encode(),
            f"/panel/{HOP_PANEL}/mcp?x=1".encode(),
            f"/panel/{CLOSED_PANEL}/mcp".encode(),
            b"/mcp",
            b"",
            None,
        ],
    )
    def test_an_admitted_request_spelled_any_other_way_is_forbidden(self, raw_path):
        """The decoded path reads ``mcp``; only the raw path tells the spellings apart."""
        assert self._refusal("mcp", admitted=True, raw_path=raw_path) == 403

    @pytest.mark.parametrize("path", ["mcp/", "/mcp", "mcp;x", "mcp/x", "///mcp//x"])
    def test_an_admitted_request_on_a_neighbouring_spelling_is_forbidden(self, path):
        assert self._refusal(path, admitted=True) == 403

    @pytest.mark.parametrize("path", ["../mcp", "a/../mcp", "./mcp", "m%2fcp"])
    def test_a_dot_segment_or_escape_is_not_found_even_when_admitted(self, path):
        assert self._refusal(path, admitted=True) == 404

    def test_an_escaped_separator_in_the_raw_path_is_not_found_even_when_admitted(self):
        raw_path = f"/panel/{HOP_PANEL}/x%2f..%2fmcp".encode()

        assert self._refusal("mcp", admitted=True, raw_path=raw_path) == 404

    @pytest.mark.parametrize("path", ["", "index.html", "api/mcp", "mcpx"])
    def test_a_page_path_passes_for_either_caller(self, path):
        assert self._refusal(path, admitted=False) is None
        assert self._refusal(path, admitted=True) is None


def _record_request(app, *, body: bytes = MCP_ANSWER_BODY, content_type: str = "application/json"):
    """Stand in for the upstream on the standard branch; return the recorder."""

    async def fake_request(*, method, url, headers, content, follow_redirects=True):  # noqa: ARG001
        return httpx.Response(200, content=body, headers={"content-type": content_type})

    request = AsyncMock(side_effect=fake_request)
    send = AsyncMock(side_effect=AssertionError("no streamed upstream request expected"))
    app.state.proxy_client.request = request
    app.state.proxy_client.send = send
    return request


def _record_stream(app, upstream):
    """Stand in for the upstream on the streaming branch; return what was built."""
    captured: dict[str, object] = {}
    real_build = app.state.proxy_client.build_request

    def spy_build(**kwargs):
        captured.update(kwargs)
        return real_build(**kwargs)

    app.state.proxy_client.build_request = spy_build
    app.state.proxy_client.send = AsyncMock(return_value=upstream)
    app.state.proxy_client.request = AsyncMock(
        side_effect=AssertionError("no buffered upstream request expected")
    )
    return captured


def _refuse_upstream(app) -> tuple[AsyncMock, AsyncMock]:
    """Fail the test if the proxy reaches an upstream by either branch."""
    request = AsyncMock(side_effect=AssertionError("no upstream request expected"))
    send = AsyncMock(side_effect=AssertionError("no upstream request expected"))
    app.state.proxy_client.request = request
    app.state.proxy_client.send = send
    return request, send


def _refuse_rewrite(*_args, **_kwargs):
    raise AssertionError("the agent hop must not rewrite the body")


@pytest.fixture
def gated(sidecars):  # noqa: ARG001 - the registry entries must exist before the app
    """The proxy router behind the real gate, with both sidecars running.

    Every row that uses this carries ``no_auth_seam``: the suite-wide seam
    stamps the operator secret onto any client aimed at a gated app, and these
    rows are about which credential the caller holds.
    """
    reset_web_credentials()
    app = FastAPI()
    app.include_router(proxy.router)
    app.state.custom_panels = []
    for panel_id in (HOP_PANEL, CLOSED_PANEL):
        setattr(app.state, panel_url_state_attr(panel_id), BACKEND_URL)
    app.state.panel_auth_headers = {HOP_PANEL: {LAUNCH_HEADER: LAUNCH_CREDENTIAL}}
    app.state.proxy_client = httpx.AsyncClient()
    credentials = WebCredentials(operator_secret=OPERATOR_SECRET, panel_token=PANEL_TOKEN)
    app.state.web_credentials = credentials
    app.add_middleware(WebAuthMiddleware, cookie_name=COOKIE_NAME)
    with TestClient(app, client=("127.0.0.1", 54321)) as client:
        yield app, client, credentials
    asyncio.run(app.state.proxy_client.aclose())
    reset_web_credentials()


def _credential_free(forwarded: dict[str, str]) -> None:
    """Nothing the caller authenticated with at the terminal went upstream."""
    names = {name.lower().replace("_", "-") for name in forwarded}
    assert "authorization" not in names
    assert "cookie" not in names
    assert OPERATOR_SECRET_HEADER.lower() not in names
    carried = " ".join(forwarded.values())
    for secret in (PANEL_TOKEN, OPERATOR_SECRET, "claimed-secret", "theme=dark"):
        assert secret not in carried


class TestTheAgentsCall:
    """The panel token of this terminal on the declared hop."""

    #: What an agent's client sends, plus every header the proxy must not relay.
    #: The cookie is not a session and the underscore header is not the gate's
    #: spelling, so the request is still admitted on the bearer alone.
    INBOUND = {
        **BEARER,
        "content-type": "application/json",
        "cookie": "theme=dark",
        "X_Osprey_Terminal_Secret": "claimed-secret",
    }

    @pytest.mark.no_auth_seam
    @pytest.mark.parametrize("method", ["GET", "POST", "DELETE"])
    def test_each_of_the_three_methods_reaches_the_transport(self, gated, method):
        app, client, _credentials = gated
        request = _record_request(app)

        resp = client.request(method, HOP_ROUTE, headers=BEARER)

        assert resp.status_code == 200
        assert request.await_args.kwargs["method"] == method
        assert request.await_args.kwargs["url"] == f"{BACKEND_URL}/mcp"

    @pytest.mark.no_auth_seam
    def test_a_buffered_answer_is_relayed_byte_for_byte(self, gated, monkeypatch):
        """Even one typed as a page, which the proxy would otherwise rewrite."""
        app, client, _credentials = gated
        monkeypatch.setattr(proxy, "_rewrite_content", _refuse_rewrite)
        monkeypatch.setattr(proxy, "_inject_control_target_bar", _refuse_rewrite)
        _record_request(app, content_type="text/html")

        resp = client.post(HOP_ROUTE, headers=self.INBOUND, content=MCP_CALL_BODY)

        assert resp.status_code == 200
        assert resp.content == MCP_ANSWER_BODY
        assert resp.headers["content-type"].startswith("text/html")

    @pytest.mark.no_auth_seam
    def test_a_streamed_answer_is_relayed_byte_for_byte(self, gated, monkeypatch):
        app, client, _credentials = gated
        monkeypatch.setattr(proxy, "_rewrite_content", _refuse_rewrite)
        captured = _record_stream(
            app,
            _FakeStreamResponse(
                headers={"content-type": "application/json"}, chunks=(MCP_ANSWER_BODY,)
            ),
        )

        resp = client.post(
            HOP_ROUTE, headers={**self.INBOUND, "accept": MCP_ACCEPT}, content=MCP_CALL_BODY
        )

        assert resp.status_code == 200
        assert resp.content == MCP_ANSWER_BODY
        assert resp.headers["content-type"] == "application/json"
        assert captured["url"] == f"{BACKEND_URL}/mcp"
        assert captured["content"] == MCP_CALL_BODY

    @pytest.mark.no_auth_seam
    def test_the_call_body_reaches_the_upstream_unchanged(self, gated):
        app, client, _credentials = gated
        request = _record_request(app)

        client.post(HOP_ROUTE, headers=self.INBOUND, content=MCP_CALL_BODY)

        assert request.await_args.kwargs["content"] == MCP_CALL_BODY

    @pytest.mark.no_auth_seam
    def test_the_upstream_receives_none_of_the_callers_credentials(self, gated):
        """Buffered branch: the recording upstream sees the launch credential only."""
        app, client, _credentials = gated
        request = _record_request(app)

        resp = client.post(HOP_ROUTE, headers=self.INBOUND, content=MCP_CALL_BODY)

        assert resp.status_code == 200
        forwarded = _lower(request.await_args.kwargs["headers"])
        _credential_free(forwarded)
        assert forwarded[LAUNCH_HEADER] == LAUNCH_CREDENTIAL

    @pytest.mark.no_auth_seam
    def test_the_streamed_upstream_receives_none_of_the_callers_credentials(self, gated):
        app, client, _credentials = gated
        captured = _record_stream(
            app, _FakeStreamResponse(headers={"content-type": "application/json"})
        )

        resp = client.post(
            HOP_ROUTE, headers={**self.INBOUND, "accept": MCP_ACCEPT}, content=MCP_CALL_BODY
        )

        assert resp.status_code == 200
        forwarded = _lower(captured["headers"])
        _credential_free(forwarded)
        assert forwarded[LAUNCH_HEADER] == LAUNCH_CREDENTIAL

    @pytest.mark.no_auth_seam
    def test_the_operator_secret_is_not_issued_to_the_transport(self, gated):
        """The sidecar earns the secret on its pages; a panel-token call carries none.

        ``operator_secret`` is on by default for a sidecar, so an operator's
        page request is forwarded with the secret. The hop is not one: the
        caller holds the weak credential, and handing the upstream the strong
        one on its behalf would raise what that caller can reach.
        """
        app, client, _credentials = gated
        request = _record_request(app)

        client.get(f"/panel/{HOP_PANEL}/page", headers={OPERATOR_SECRET_HEADER: OPERATOR_SECRET})
        page = _lower(request.await_args.kwargs["headers"])
        client.post(HOP_ROUTE, headers=BEARER, content=MCP_CALL_BODY)
        hop = _lower(request.await_args.kwargs["headers"])

        assert page[OPERATOR_SECRET_HEADER.lower()] == OPERATOR_SECRET
        assert OPERATOR_SECRET_HEADER.lower() not in hop


class TestOtherCallers:
    """Everyone who is not this terminal's agent."""

    @pytest.mark.no_auth_seam
    @pytest.mark.parametrize("method", ["GET", "POST", "DELETE"])
    def test_the_operator_secret_is_forbidden_on_the_hop(self, gated, method):
        app, client, _credentials = gated
        request, send = _refuse_upstream(app)

        resp = client.request(method, HOP_ROUTE, headers={OPERATOR_SECRET_HEADER: OPERATOR_SECRET})

        assert resp.status_code == 403
        assert request.await_count == 0
        assert send.await_count == 0

    @pytest.mark.no_auth_seam
    @pytest.mark.parametrize("method", ["GET", "POST", "DELETE"])
    def test_a_browser_session_is_forbidden_on_the_hop(self, gated, method):
        app, client, credentials = gated
        request, send = _refuse_upstream(app)
        headers = {
            "cookie": f"{COOKIE_NAME}={credentials.create_session()}",
            "origin": "http://testserver",
        }

        resp = client.request(method, HOP_ROUTE, headers=headers)

        assert resp.status_code == 403
        assert request.await_count == 0
        assert send.await_count == 0

    @pytest.mark.no_auth_seam
    def test_an_operator_who_also_sends_the_panel_token_is_still_forbidden(self, gated):
        """The gate admits on the stronger credential, so the mark is never set."""
        app, client, credentials = gated
        request, _send = _refuse_upstream(app)
        with_secret = {**BEARER, OPERATOR_SECRET_HEADER: OPERATOR_SECRET}
        with_cookie = {
            **BEARER,
            "cookie": f"{COOKIE_NAME}={credentials.create_session()}",
            "origin": "http://testserver",
        }

        assert client.post(HOP_ROUTE, headers=with_secret).status_code == 403
        assert client.post(HOP_ROUTE, headers=with_cookie).status_code == 403
        assert request.await_count == 0

    @pytest.mark.no_auth_seam
    @pytest.mark.parametrize("method", ["GET", "POST", "DELETE"])
    def test_a_token_minted_by_another_terminal_is_unauthorised(self, gated, method):
        app, client, _credentials = gated
        request, send = _refuse_upstream(app)

        resp = client.request(
            method, HOP_ROUTE, headers={"authorization": f"Bearer {FOREIGN_PANEL_TOKEN}"}
        )

        assert resp.status_code == 401
        assert request.await_count == 0
        assert send.await_count == 0

    @pytest.mark.no_auth_seam
    def test_no_credential_is_unauthorised(self, gated):
        app, client, _credentials = gated
        request, _send = _refuse_upstream(app)

        assert client.post(HOP_ROUTE, content=MCP_CALL_BODY).status_code == 401
        assert request.await_count == 0

    @pytest.mark.no_auth_seam
    def test_an_operator_still_reaches_the_sidecars_pages(self, gated):
        app, client, _credentials = gated
        request = _record_request(app)

        resp = client.get(
            f"/panel/{HOP_PANEL}/index.html", headers={OPERATOR_SECRET_HEADER: OPERATOR_SECRET}
        )

        assert resp.status_code == 200
        assert request.await_args.kwargs["url"] == f"{BACKEND_URL}/index.html"


class TestThePanelTokenElsewhere:
    """The token opens the declared spelling and nothing near it."""

    @pytest.mark.no_auth_seam
    @pytest.mark.parametrize(
        ("path", "status"),
        [
            ("%6dcp", 403),
            ("m%63p", 403),
            ("mcp/", 401),
            ("/mcp", 401),
            ("mcp;x", 401),
            ("mcp/x", 401),
            ("MCP", 401),
            ("%2e%2e/mcp", 401),
            ("%252e%252e/mcp", 401),
            ("m%2fcp", 401),
            ("", 401),
            ("index.html", 401),
            ("api/tools", 401),
        ],
    )
    def test_any_other_path_below_the_panel_is_refused_with_no_upstream_request(
        self, gated, path, status
    ):
        """An escaped letter decodes to the route, so only the proxy can refuse it."""
        app, client, _credentials = gated
        request, send = _refuse_upstream(app)

        resp = client.post(f"/panel/{HOP_PANEL}/{path}", headers=BEARER, content=MCP_CALL_BODY)

        assert resp.status_code == status
        assert request.await_count == 0
        assert send.await_count == 0

    @pytest.mark.no_auth_seam
    @pytest.mark.parametrize("method", ["PUT", "PATCH", "HEAD", "OPTIONS"])
    def test_any_other_method_on_the_hop_is_unauthorised(self, gated, method):
        app, client, _credentials = gated
        request, _send = _refuse_upstream(app)

        assert client.request(method, HOP_ROUTE, headers=BEARER).status_code == 401
        assert request.await_count == 0

    @pytest.mark.no_auth_seam
    @pytest.mark.parametrize("method", ["GET", "POST", "DELETE"])
    def test_a_transport_without_the_opt_in_is_closed_to_the_token(self, gated, method):
        app, client, _credentials = gated
        request, _send = _refuse_upstream(app)

        resp = client.request(method, f"/panel/{CLOSED_PANEL}/mcp", headers=BEARER)

        assert resp.status_code == 401
        assert request.await_count == 0

    @pytest.mark.no_auth_seam
    def test_a_transport_without_the_opt_in_is_closed_to_the_operator(self, gated):
        app, client, _credentials = gated
        request, _send = _refuse_upstream(app)

        resp = client.post(
            f"/panel/{CLOSED_PANEL}/mcp", headers={OPERATOR_SECRET_HEADER: OPERATOR_SECRET}
        )

        assert resp.status_code == 403
        assert request.await_count == 0


class TestOtherSpellingsForAnOperator:
    """Opting in loosens nothing for the callers the hop is closed to."""

    @pytest.mark.no_auth_seam
    @pytest.mark.parametrize(
        ("path", "status"),
        [
            ("mcp", 403),
            ("mcp/", 403),
            ("/mcp", 403),
            ("mcp;x", 403),
            ("mcp/x", 403),
            ("%6dcp", 403),
            ("%2e%2e/mcp", 404),
            ("%252e%252e/mcp", 404),
            ("m%2fcp", 404),
            ("page%5c..%5cmcp", 404),
        ],
    )
    def test_every_spelling_is_refused_with_no_upstream_request(self, gated, path, status):
        app, client, _credentials = gated
        request, send = _refuse_upstream(app)

        resp = client.get(
            f"/panel/{HOP_PANEL}/{path}", headers={OPERATOR_SECRET_HEADER: OPERATOR_SECRET}
        )

        assert resp.status_code == status
        assert request.await_count == 0
        assert send.await_count == 0


class TestWebSocketLeg:
    """No websocket reaches the transport: the token opens none, an operator is refused."""

    @pytest.mark.no_auth_seam
    def test_the_panel_token_opens_no_websocket_on_the_hop(self, gated):
        _app, client, _credentials = gated
        fake = _FakeConnect()

        with _patch_connect(fake):
            with pytest.raises(WebSocketDisconnect):
                with client.websocket_connect(HOP_ROUTE, headers=BEARER):
                    pass  # pragma: no cover - the handshake is refused

        assert fake.target is None

    @pytest.mark.no_auth_seam
    def test_an_operator_websocket_on_the_hop_is_forbidden(self, gated):
        _app, client, _credentials = gated
        fake = _FakeConnect()

        with _patch_connect(fake):
            with pytest.raises(WebSocketDisconnect) as closed:
                with client.websocket_connect(
                    HOP_ROUTE, headers={OPERATOR_SECRET_HEADER: OPERATOR_SECRET}
                ):
                    pass  # pragma: no cover - the handshake is refused

        assert closed.value.code == 4403
        assert fake.target is None


# ---------------------------------------------------------------------------
# The hold: the hop is a route the agent has, so a watchdog hold covers it
# ---------------------------------------------------------------------------

SESSION_KEY = "1f5a9c7e-0b44-4a1d-9cc2-0e5b6d2f7a31"  # gitleaks:allow - a test session id
OTHER_SESSION_KEY = "2f5a9c7e-0b44-4a1d-9cc2-0e5b6d2f7a32"  # gitleaks:allow - a test session id

#: A call to the one tool the panel asks the operator about.
ASK_CALL = json.dumps(
    {"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {"name": "push_screen"}}
).encode()
#: A call to a tool the panel lets the agent call unasked.
READ_CALL = json.dumps(
    {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "get_screen"}}
).encode()


class _Pool:
    """A session registry that knows the sessions it was given."""

    def __init__(self, *keys):
        self.keys = set(keys)

    def get_session(self, key):
        return object() if key in self.keys else None

    def has_chat_key(self, _key):
        return False


def _hold(root: Path, key: str = SESSION_KEY) -> None:
    assert open_flag(
        key,
        flag_id="flag-a",
        source="wd",
        reason="risky",
        prompt_ref="0f6d7c2e-1a2b-4c3d-8e9f-0123456789ab:flag-a",
        prompt_excerpt="please loosen the write limits",
        root=root,
    )


@pytest.fixture
def held_hop(gated, tmp_path):
    """The hop under a hold store, with a live session and a recording upstream."""
    app, client, _credentials = gated
    root = tmp_path / "session_holds"
    app.state.session_hold_root = root
    app.state.pty_registry = _Pool(SESSION_KEY)
    request = _record_request(app)
    return app, client, root, request


def _call(client, body, *, session=SESSION_KEY):
    headers = {**BEARER, "content-type": "application/json"}
    if session is not None:
        headers["x-osprey-session"] = session
    return client.post(HOP_ROUTE, headers=headers, content=body)


class TestTheHold:
    """The same hold the dispatcher hop applies, on the sidecar's ask tools."""

    @pytest.mark.no_auth_seam
    def test_an_ask_tool_is_refused_while_the_session_is_held(self, held_hop):
        _app, client, root, request = held_hop
        _hold(root)

        resp = _call(client, ASK_CALL)

        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == 7
        assert "[osprey-refusal: session_held]" in body["error"]["message"]
        assert request.await_count == 0

    @pytest.mark.no_auth_seam
    def test_the_same_call_passes_once_nothing_holds_the_session(self, held_hop):
        """A session with a record and no open flag; a session with no record
        at all is held, fail closed, like a job on the dispatcher hop."""
        _app, client, root, request = held_hop
        ensure_session(SESSION_KEY, root=root)

        resp = _call(client, ASK_CALL)

        assert resp.status_code == 200
        assert resp.content == MCP_ANSWER_BODY
        assert request.await_count == 1

    @pytest.mark.no_auth_seam
    @pytest.mark.parametrize("session", [None, "", "not-live", OTHER_SESSION_KEY, "../x"])
    def test_an_ask_tool_needs_a_live_session_under_a_store(self, held_hop, session):
        """A key nobody runs cannot act past a hold it has no record of."""
        _app, client, _root, request = held_hop

        resp = _call(client, ASK_CALL, session=session)

        assert "error" in resp.json()
        assert request.await_count == 0

    @pytest.mark.no_auth_seam
    def test_a_corrupt_record_refuses_the_call(self, held_hop):
        _app, client, root, request = held_hop
        (root / SESSION_KEY).mkdir(parents=True)
        (root / SESSION_KEY / RECORD_NAME).write_text("garbage")

        assert "session_held" in _call(client, ASK_CALL).json()["error"]["message"]
        assert request.await_count == 0

    @pytest.mark.no_auth_seam
    @pytest.mark.parametrize("body", [READ_CALL, MCP_CALL_BODY, b"not json"])
    def test_a_call_that_acts_on_nothing_passes_while_held(self, held_hop, body):
        _app, client, root, request = held_hop
        _hold(root)

        _call(client, body, session=None)

        assert request.await_count == 1

    @pytest.mark.no_auth_seam
    def test_a_batch_with_an_ask_tool_is_refused_whole(self, held_hop):
        _app, client, root, request = held_hop
        _hold(root)
        batch = json.dumps([json.loads(MCP_CALL_BODY), json.loads(ASK_CALL)]).encode()

        answer = _call(client, batch).json()

        assert [entry["id"] for entry in answer] == [1, 7]
        assert request.await_count == 0

    @pytest.mark.no_auth_seam
    def test_with_no_store_nothing_is_checked(self, gated):
        app, client, _credentials = gated
        app.state.session_hold_root = None
        request = _record_request(app)

        _call(client, ASK_CALL, session=None)

        assert request.await_count == 1
