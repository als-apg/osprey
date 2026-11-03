"""Who can reach CANOPY's write route, and the panel ops, through the terminal.

A panel write carries ``confirmed``, a boolean the viewer sets when a person
answers its dialog. Nothing downstream can tell a person's click from a
caller that set the flag itself, so what makes the flag a person's is who can
send the write at all. The terminal's gate is that boundary
(:class:`~osprey.interfaces.common_middleware.WebAuthMiddleware`):

* the browser session's cookie, on a request whose ``Origin`` is the
  terminal's own (or, with no ``Origin``, ``Sec-Fetch-Site: same-origin``), is
  the one caller admitted. That is the viewer's own page, CANOPY's entry
  document, whose dialog sets the flag;
* a foreign ``Origin``, the opaque ``null`` origin of a sandboxed document
  (every document a panel serves outside its entry pages, PR 2's sandbox:
  ``test_canopy_entry_pages.py``), a cookie no browser vouches for, and a
  request with no credential at all are refused by name at the gate, before
  the proxy, so the write never reaches CANOPY and ``session.write`` never runs;
* the panel token, which the agent's process holds, is refused on the write
  route whatever ``Origin`` it carries: the route is not in the panel tier.

The panel ops take the panel token as their agent caller; a browser-borne
call to them meets the same Origin rule as the write.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces.common_middleware import WebAuthMiddleware
from osprey.interfaces.web_auth import Tier, WebCredentials, classify, reset_web_credentials
from osprey.interfaces.web_terminal.routes import panel_ops, proxy
from osprey.profiles.sidecar_panels import CANOPY_PANEL_ID
from osprey.registry.web import panel_url_state_attr

SIDECAR_URL = "http://127.0.0.1:45678"
PANEL_TOKEN = "panel-token-value"
OPERATOR_SECRET = "operator-secret-value"
COOKIE_NAME = "osprey_terminal_session_8080"

#: The terminal's own origin, as the test client addresses it.
OWN_ORIGIN = "http://testserver"

#: CANOPY's PV write, as FRONT serves it under the panel's mount.
WRITE_PATH = f"/panel/{CANOPY_PANEL_ID}/hare/v1/pvs/SR:C01:HCM1:Setpoint"
WRITE_BODY = b'{"value": 1.5, "binding": "simulator@3", "confirmed": true}'
JSON = {"content-type": "application/json"}

ORIGIN_REFUSED = "cross-origin request refused"
NO_CREDENTIAL = "authentication required"
NOT_PANEL_TIER = "this route requires operator credentials"

#: Browser-borne callers that are not the viewer's own page.
FOREIGN_BROWSER_HEADERS = [
    pytest.param({"origin": "https://evil.example"}, id="foreign-origin"),
    pytest.param({"origin": "null"}, id="null-origin-sandboxed-document"),
    pytest.param({"origin": "http://testserver.evil.example"}, id="origin-prefix-lookalike"),
    pytest.param({"origin": "http://testserver:8081"}, id="same-host-other-port"),
    pytest.param({"sec-fetch-site": "cross-site"}, id="no-origin-cross-site"),
    pytest.param({"sec-fetch-site": "same-site"}, id="no-origin-same-site"),
    pytest.param({}, id="no-origin-no-fetch-metadata"),
]

#: The panel ops a browser-borne request could aim at.
PANEL_OPS = [
    ("/api/panel-op/open", {"request_id": "r1", "member": CANOPY_PANEL_ID}),
    ("/api/panel-op/open-set", {"request_id": "r1", "members": [CANOPY_PANEL_ID], "as": "tabs"}),
    ("/api/panel-op/open-arrangement", {"request_id": "r1", "name": "x"}),
    ("/api/panel-op/close-tab", {"request_id": "r1", "key": CANOPY_PANEL_ID}),
]


@pytest.fixture
def gated():
    """The proxy and the panel ops behind the real gate, CANOPY's sidecar up.

    Every upstream call and every op handler fails the test unless a row
    replaces it, so a refused request is shown never to have got past the gate.
    """
    reset_web_credentials()
    app = FastAPI()
    app.include_router(proxy.router)
    app.include_router(panel_ops.router)
    app.state.custom_panels = []
    setattr(app.state, panel_url_state_attr(CANOPY_PANEL_ID), SIDECAR_URL)
    app.state.proxy_client = httpx.AsyncClient()
    app.state.proxy_client.request = AsyncMock(side_effect=AssertionError("reached CANOPY"))
    app.state.proxy_client.send = AsyncMock(side_effect=AssertionError("reached CANOPY"))
    credentials = WebCredentials(operator_secret=OPERATOR_SECRET, panel_token=PANEL_TOKEN)
    app.state.web_credentials = credentials
    app.add_middleware(WebAuthMiddleware, cookie_name=COOKIE_NAME)
    with TestClient(app, client=("127.0.0.1", 54321)) as client:
        yield app, client, credentials
    asyncio.run(app.state.proxy_client.aclose())
    reset_web_credentials()


@pytest.fixture
def no_op_runs(monkeypatch):
    async def refuse(*_args, **_kwargs):
        raise AssertionError("an op handler ran")

    monkeypatch.setattr(panel_ops, "_run_op", refuse)


def _cookie(credentials: WebCredentials) -> dict[str, str]:
    return {"cookie": f"{COOKIE_NAME}={credentials.create_session(persist=False)}"}


def _assert_never_forwarded(app: FastAPI) -> None:
    app.state.proxy_client.request.assert_not_awaited()
    app.state.proxy_client.send.assert_not_awaited()


@pytest.mark.no_auth_seam
class TestTheWriteRoute:
    def test_the_viewers_own_page_is_forwarded_once_with_its_body(self, gated):
        """The control row: cookie and own Origin reach CANOPY, ``confirmed`` as sent."""
        app, client, credentials = gated

        async def landed(*, method, url, headers, content, follow_redirects=True):  # noqa: ARG001
            return httpx.Response(200, json={"pv": "SR:C01:HCM1:Setpoint"})

        app.state.proxy_client.request = AsyncMock(side_effect=landed)
        app.state.proxy_client.send = AsyncMock(side_effect=AssertionError("not streamed"))

        resp = client.put(
            WRITE_PATH,
            headers={**_cookie(credentials), "origin": OWN_ORIGIN, **JSON},
            content=WRITE_BODY,
        )

        assert resp.status_code == 200, resp.text
        call = app.state.proxy_client.request.await_args.kwargs
        assert call["method"] == "PUT"
        assert call["url"] == f"{SIDECAR_URL}/hare/v1/pvs/SR:C01:HCM1:Setpoint"
        assert call["content"] == WRITE_BODY

    def test_same_origin_fetch_metadata_stands_in_for_a_missing_origin(self, gated):
        app, client, credentials = gated
        app.state.proxy_client.request = AsyncMock(return_value=httpx.Response(200, json={}))

        resp = client.put(
            WRITE_PATH,
            headers={**_cookie(credentials), "sec-fetch-site": "same-origin", **JSON},
            content=WRITE_BODY,
        )

        assert resp.status_code == 200, resp.text
        app.state.proxy_client.request.assert_awaited_once()

    @pytest.mark.parametrize("headers", FOREIGN_BROWSER_HEADERS)
    def test_a_session_cookie_from_anywhere_else_is_refused_by_name(self, gated, headers):
        app, client, credentials = gated

        resp = client.put(
            WRITE_PATH, headers={**_cookie(credentials), **headers, **JSON}, content=WRITE_BODY
        )

        assert resp.status_code == 403
        assert resp.json()["detail"] == ORIGIN_REFUSED
        _assert_never_forwarded(app)

    @pytest.mark.parametrize(
        "headers",
        [
            pytest.param({"origin": OWN_ORIGIN}, id="own-origin"),
            pytest.param({"origin": "null"}, id="null-origin"),
            pytest.param({}, id="nothing"),
        ],
    )
    def test_a_write_with_no_credential_is_refused_by_name(self, gated, headers):
        app, client, _credentials = gated

        resp = client.put(WRITE_PATH, headers={**headers, **JSON}, content=WRITE_BODY)

        assert resp.status_code == 401
        assert resp.json()["detail"] == NO_CREDENTIAL
        _assert_never_forwarded(app)

    def test_an_expired_or_forged_session_is_refused(self, gated):
        app, client, _credentials = gated

        resp = client.put(
            WRITE_PATH,
            headers={"cookie": f"{COOKIE_NAME}=forged", "origin": OWN_ORIGIN, **JSON},
            content=WRITE_BODY,
        )

        assert resp.status_code == 401
        _assert_never_forwarded(app)


@pytest.mark.no_auth_seam
class TestThePanelTokenAlone:
    """The agent's credential does not pass the write route, however it is sent."""

    def test_the_write_route_is_not_in_the_panel_tier(self):
        assert classify("PUT", WRITE_PATH, False) is Tier.OPERATOR
        assert classify("POST", WRITE_PATH, False) is Tier.OPERATOR

    @pytest.mark.parametrize(
        "headers",
        [
            pytest.param({}, id="no-origin"),
            pytest.param({"origin": OWN_ORIGIN}, id="own-origin"),
            pytest.param({"sec-fetch-site": "same-origin"}, id="same-origin-metadata"),
        ],
    )
    def test_the_token_is_refused_on_the_write_route(self, gated, headers):
        app, client, _credentials = gated

        resp = client.put(
            WRITE_PATH,
            headers={"authorization": f"Bearer {PANEL_TOKEN}", **headers, **JSON},
            content=WRITE_BODY,
        )

        assert resp.status_code == 401
        assert resp.json()["detail"] == NOT_PANEL_TIER
        _assert_never_forwarded(app)

    def test_the_token_with_a_foreign_origin_is_refused_for_its_origin(self, gated):
        app, client, _credentials = gated

        resp = client.put(
            WRITE_PATH,
            headers={
                "authorization": f"Bearer {PANEL_TOKEN}",
                "origin": "https://evil.example",
                **JSON,
            },
            content=WRITE_BODY,
        )

        assert resp.status_code == 403
        assert resp.json()["detail"] == ORIGIN_REFUSED
        _assert_never_forwarded(app)


@pytest.mark.no_auth_seam
@pytest.mark.usefixtures("no_op_runs")
class TestThePanelOps:
    @pytest.mark.parametrize(("path", "body"), PANEL_OPS)
    @pytest.mark.parametrize("headers", FOREIGN_BROWSER_HEADERS)
    def test_a_session_cookie_from_anywhere_else_is_refused_by_name(
        self, gated, path, body, headers
    ):
        _app, client, credentials = gated

        resp = client.post(path, headers={**_cookie(credentials), **headers}, json=body)

        assert resp.status_code == 403
        assert resp.json()["detail"] == ORIGIN_REFUSED

    @pytest.mark.parametrize(("path", "body"), PANEL_OPS)
    def test_an_op_with_no_credential_is_refused_by_name(self, gated, path, body):
        _app, client, _credentials = gated

        resp = client.post(path, headers={"origin": OWN_ORIGIN}, json=body)

        assert resp.status_code == 401
        assert resp.json()["detail"] == NO_CREDENTIAL

    @pytest.mark.parametrize(("path", "body"), PANEL_OPS)
    @pytest.mark.parametrize("origin", ["https://evil.example", "null"])
    def test_the_panel_token_from_a_browser_elsewhere_is_refused(self, gated, path, body, origin):
        _app, client, _credentials = gated

        resp = client.post(
            path, headers={"authorization": f"Bearer {PANEL_TOKEN}", "origin": origin}, json=body
        )

        assert resp.status_code == 403
        assert resp.json()["detail"] == ORIGIN_REFUSED
