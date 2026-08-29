"""Contract tests for the ``?token=`` → session-cookie exchange.

A browser handed a login URL arrives at an exchange page (``/`` or
``/static/session.html``) carrying ``?token=<operator secret>``.
:class:`~osprey.interfaces.common_middleware.WebAuthMiddleware` answers that
request itself: it mints a session, sets it as an ``HttpOnly``,
``SameSite=Lax`` cookie carrying the configured lifetime as ``Max-Age`` and
redirects to the token-stripped URL, and the cookie carries authentication from
then on.

The exchange lives in the **gate**, not in a route, and the first module
section is what pins that: it runs the exchange against *every* app that
installs the gate. Six entry points print a login URL (``osprey web``, ``osprey
artifacts web``, ``osprey channel-finder web``, ``osprey theme-lab``, ARIEL's
``run_web``, the artifact gallery's ``run_server``) and only one of them used to
own an exchange handler, so on the other five the page rendered, no cookie was
minted, and every later ``/api/*`` call was 401 — with the secret left in the
address bar to leak through ``Referer`` in the meantime. A per-app handler is
something an interface has to remember; a gate is something it cannot avoid.

These tests set auth up **explicitly** — a known operator secret on
``app.state.web_credentials``, cookies constructed by hand — rather than leaning
on the shared TestClient auth seam, because the flow under test *is* the auth
flow. The whole module opts out of that seam (``pytestmark`` below): the seam
force-sets the operator header on every request, which would silently admit the
very unauthenticated requests these tests assert are refused.

The apps are driven without the ``TestClient`` lifespan (no ``with`` block): the
exchange needs only ``app.state.web_credentials``, which is seeded directly, and
skipping startup keeps the tests off the PTY/file-watcher machinery the flow
does not touch.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from starlette.testclient import TestClient

from osprey.interfaces.common_middleware import (
    EXTERNAL_ORIGIN_ENV,
    TERMINAL_USER_ENV,
    WebAuthMiddleware,
    _exchange_query_token,
    session_cookie_name,
)
from osprey.interfaces.web_auth import WebCredentials
from osprey.interfaces.web_terminal.app import create_app
from tests.interfaces.test_app_setup_parametrized import INTERFACE_BUILDERS

pytestmark = pytest.mark.no_auth_seam

OPERATOR_SECRET = "operator-secret-value"
PANEL_TOKEN = "panel-token-value"
WEB_PORT = "8899"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _client(app, **kwargs) -> TestClient:
    """A fresh client with an empty cookie jar."""
    client = TestClient(app, **kwargs)
    client.cookies.clear()
    return client


def _set_cookie_header(response) -> str:
    header = response.headers.get("set-cookie")
    assert header is not None, "expected a Set-Cookie header on the exchange"
    return header


def _cookie_pair(response) -> tuple[str, str]:
    """The ``name=value`` a Set-Cookie carries, split into a pair."""
    first = _set_cookie_header(response).split(";", 1)[0]
    name, _, value = first.partition("=")
    return name, value


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch):
    """A real web-terminal app with a known operator secret on its gate.

    The port is pinned through the environment so the gate derives the cookie
    name the assertions expect, and the credential holder is set directly on
    ``app.state`` — the object the gate reads and mints sessions on.
    """
    monkeypatch.setenv("OSPREY_WEB_PORT", WEB_PORT)
    application = create_app(shell_command="echo")
    application.state.web_credentials = WebCredentials(
        operator_secret=OPERATOR_SECRET, panel_token=PANEL_TOKEN
    )
    return application


# --------------------------------------------------------------------------- #
# The exchange is structural: every app that installs the gate performs it
# --------------------------------------------------------------------------- #


@pytest.fixture(params=INTERFACE_BUILDERS, ids=[name for name, _ in INTERFACE_BUILDERS])
def gated_app(request, tmp_path, monkeypatch: pytest.MonkeyPatch):
    """One real app per interface that calls ``configure_interface_app``.

    The credential holder is swapped in through ``monkeypatch`` rather than
    assigned, because one of these apps — ``bluesky_web`` — is assembled at
    module scope and is therefore a process-wide singleton: an assignment would
    outlive this test and leave every later suite that drives that app (the
    Playwright visual snapshots authorize a browser against the *process*
    holder) authenticating against a holder only this module knows about.
    """
    monkeypatch.setenv("OSPREY_WEB_PORT", WEB_PORT)
    _, builder = request.param
    application = builder(tmp_path)
    monkeypatch.setattr(
        application.state,
        "web_credentials",
        WebCredentials(operator_secret=OPERATOR_SECRET, panel_token=PANEL_TOKEN),
        raising=False,
    )
    return application


def test_every_gated_app_exchanges_a_token_for_a_cookie(gated_app):
    """``GET /?token=<secret>`` is a 303 carrying the session cookie.

    The assertion that would have caught the original defect on five of the six
    entry points: the gate answers the login URL on every app that installs it,
    so no interface can ship a login URL that renders a page and hands out no
    session.
    """
    response = _client(gated_app).get(f"/?token={OPERATOR_SECRET}", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/"
    name, value = _cookie_pair(response)
    assert name == session_cookie_name(WEB_PORT)
    assert value


def test_every_gated_app_admits_the_cookie_it_handed_out(gated_app):
    """The exchanged cookie clears the gate on the next request.

    ``/`` is the one path every one of these apps answers *something* on, so the
    assertion is about the gate rather than about any app's routing: without the
    cookie the gate refuses with 401; with it, whatever the app itself does
    (render, redirect, 404) is reached.
    """
    exchange = _client(gated_app).get(f"/?token={OPERATOR_SECRET}", follow_redirects=False)
    name, value = _cookie_pair(exchange)

    assert _client(gated_app).get("/", follow_redirects=False).status_code == 401

    authenticated = _client(gated_app)
    authenticated.cookies.set(name, value)
    assert authenticated.get("/", follow_redirects=False).status_code != 401


def test_no_gated_app_renders_a_page_at_the_token_url(gated_app):
    """The token URL never renders: it redirects, so no subresource sees it.

    A page rendered *at* ``/?token=SECRET`` makes the browser attach that URL as
    the ``Referer`` of every stylesheet, script and font it then fetches — which
    is how a one-time secret ends up in an access log it was never meant to
    reach.
    """
    response = _client(gated_app).get(f"/?token={OPERATOR_SECRET}", follow_redirects=False)
    assert response.status_code == 303
    assert response.content == b""


# --------------------------------------------------------------------------- #
# The cookie the exchange hands out
# --------------------------------------------------------------------------- #


def test_exchange_cookie_carries_the_prescribed_attributes(app):
    """HttpOnly + SameSite=Lax + Path=/ + Max-Age, and — loopback http — no Secure.

    Attributes are asserted as substrings because their *order* in the header is
    not load-bearing — a browser parses ``Set-Cookie`` as an unordered
    attribute list. ``Max-Age`` is the one exception to substring-only checking:
    it must appear exactly once, since a second copy would leave the cookie's
    expiry decided by whichever the browser happens to keep.
    """
    response = _client(app).get(f"/?token={OPERATOR_SECRET}", follow_redirects=False)
    header = _set_cookie_header(response).lower()

    assert "httponly" in header
    assert "samesite=lax" in header
    assert "samesite=strict" not in header  # Strict would drop the login navigation
    assert "path=/" in header
    # The default 12-hour lifetime, so the cookie survives a browser restart and
    # expires on the same schedule the server expires the session.
    assert "max-age=43200" in header
    assert header.count("max-age=") == 1
    # Secure would make the browser refuse the cookie over the single-user
    # loopback http origin, breaking the exchange the moment it is set.
    assert "secure" not in header


def test_exchange_cookie_max_age_follows_the_configured_lifetime(
    monkeypatch: pytest.MonkeyPatch,
):
    """A deployment that shortens the session shortens the cookie with it.

    The number is not a constant in the cookie renderer: it is read off the
    credential holder, which is where the deployment's configured lifetime
    lands. A cookie that outlived the server-side deadline would be presented
    long after the session was forgotten, and the operator would meet a 401
    rather than the login page.
    """
    monkeypatch.setenv("OSPREY_WEB_PORT", WEB_PORT)
    application = create_app(shell_command="echo")
    application.state.web_credentials = WebCredentials(
        operator_secret=OPERATOR_SECRET,
        panel_token=PANEL_TOKEN,
        session_ttl_seconds=60,
    )

    response = _client(application).get(f"/?token={OPERATOR_SECRET}", follow_redirects=False)
    header = _set_cookie_header(response).lower()

    assert "max-age=60" in header
    assert header.count("max-age=") == 1


def test_cookie_is_secure_over_an_https_request(app):
    """A request that arrived over TLS gets a ``Secure`` cookie."""
    client = _client(app, base_url="https://testserver")
    response = client.get(f"/?token={OPERATOR_SECRET}", follow_redirects=False)

    assert "secure" in _set_cookie_header(response).lower()


def test_cookie_is_secure_behind_an_https_external_origin(app, monkeypatch):
    """The proxy shape derives ``Secure`` from the browser-facing origin.

    nginx terminates TLS and forwards over plain HTTP, so the request's own
    scheme says ``http`` and only ``OSPREY_TERMINAL_EXTERNAL_ORIGIN`` knows the
    browser is on ``https://``. Deriving from it is what keeps a real
    deployment's session cookie off plain HTTP.
    """
    monkeypatch.setenv(EXTERNAL_ORIGIN_ENV, "https://osprey.example.org")
    response = _client(app).get(f"/?token={OPERATOR_SECRET}", follow_redirects=False)

    assert "secure" in _set_cookie_header(response).lower()


def test_cookie_is_not_secure_behind_an_http_external_origin(app, monkeypatch):
    """A plain-http external origin must not get ``Secure`` — the browser would drop it."""
    monkeypatch.setenv(EXTERNAL_ORIGIN_ENV, "http://osprey.example.org")
    response = _client(app).get(f"/?token={OPERATOR_SECRET}", follow_redirects=False)

    assert "secure" not in _set_cookie_header(response).lower()


def test_exchanged_cookie_authenticates_a_subsequent_request(app):
    """The cookie the exchange sets admits a later, token-less request."""
    exchange = _client(app).get(f"/?token={OPERATOR_SECRET}", follow_redirects=False)
    name, value = _cookie_pair(exchange)

    follow_up = _client(app)
    follow_up.cookies.set(name, value)
    landed = follow_up.get("/", follow_redirects=False)

    assert landed.status_code == 200


def test_exchanged_cookie_clears_a_gated_api_route(app):
    """The cookie is a full session, not merely a pass for the landing page."""
    exchange = _client(app).get(f"/?token={OPERATOR_SECRET}", follow_redirects=False)
    name, value = _cookie_pair(exchange)

    assert _client(app).get("/api/session").status_code == 401

    authenticated = _client(app)
    authenticated.cookies.set(name, value)
    assert authenticated.get("/api/session").status_code == 200


# --------------------------------------------------------------------------- #
# Where the exchange sends the browser
# --------------------------------------------------------------------------- #


def test_exchange_preserves_other_query_parameters(app):
    """Only ``token`` is stripped; any sibling query parameters survive."""
    response = _client(app).get(f"/?theme=dark&token={OPERATOR_SECRET}", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/?theme=dark"
    assert "token" not in response.headers["location"]


def test_exchange_redirect_carries_the_multi_user_url_prefix(monkeypatch):
    """nginx strips ``/u/<user>``, so the redirect has to put it back.

    The app sees a bare ``/``; a relative ``Location`` built from that path is
    resolved by the browser against the *front door's* root, landing it outside
    its own container's mount with a cookie it never gets to use.
    """
    monkeypatch.setenv("OSPREY_WEB_PORT", WEB_PORT)
    monkeypatch.setenv(TERMINAL_USER_ENV, "alice")
    application = create_app(shell_command="echo")
    application.state.web_credentials = WebCredentials(
        operator_secret=OPERATOR_SECRET, panel_token=PANEL_TOKEN
    )

    response = _client(application).get(f"/?token={OPERATOR_SECRET}", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/u/alice/"


def test_session_html_exchanges_a_valid_token(app):
    """The exempt ``session.html`` landing page also performs the exchange.

    Exempt paths skip the credential ladder, but not the exchange: a login URL
    may point here, and skipping it would leave the secret in the address bar of
    the very page whose job is to trade it away.
    """
    response = _client(app).get(
        f"/static/session.html?token={OPERATOR_SECRET}", follow_redirects=False
    )
    assert response.status_code == 303
    name, value = _cookie_pair(response)
    assert name == session_cookie_name(WEB_PORT)
    assert value
    assert response.headers["location"] == "/static/session.html"


def test_duplicate_token_keys_are_resolved_consistently(app):
    """A duplicated ``?token=`` admits on the first value and strips every copy.

    ``?token=<valid>&token=`` is admitted by ``_exchange_query_token``'s
    first-value read; the redirect must drop *both* keys, or the secret survives
    in the address bar of the page the browser lands on.
    """
    response = _client(app).get(f"/?token={OPERATOR_SECRET}&token=", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/"
    name, value = _cookie_pair(response)
    assert name == session_cookie_name(WEB_PORT)
    assert value


# --------------------------------------------------------------------------- #
# Refusals and non-exchanges
# --------------------------------------------------------------------------- #


def test_wrong_token_is_refused(app):
    """A ``?token=`` that is not the operator secret is refused at the gate."""
    response = _client(app).get("/?token=not-the-secret", follow_redirects=False)
    assert response.status_code == 401
    assert response.headers.get("set-cookie") is None


def test_valid_cookie_without_a_token_renders(app):
    """A live session cookie and no token renders the page normally."""
    session_id = app.state.web_credentials.create_session()
    client = _client(app)
    client.cookies.set(session_cookie_name(WEB_PORT), session_id)

    response = client.get("/", follow_redirects=False)
    assert response.status_code == 200


def test_no_token_and_no_cookie_is_refused(app):
    """Neither credential means a 401 — the gate's default answer."""
    response = _client(app).get("/", follow_redirects=False)
    assert response.status_code == 401


def test_empty_token_query_is_not_a_credential(app):
    """``?token=`` with no value is treated as absent, not as an empty secret."""
    response = _client(app).get("/?token=", follow_redirects=False)
    assert response.status_code == 401


def test_token_on_a_non_exchange_path_does_not_authenticate(app):
    """A secret in the URL of an ordinary route is not a credential there."""
    response = _client(app).get(f"/api/config?token={OPERATOR_SECRET}", follow_redirects=False)
    assert response.status_code == 401


def test_token_via_a_non_get_method_does_not_authenticate(app):
    """A URL-borne secret can never authenticate a mutating request."""
    response = _client(app).post(f"/?token={OPERATOR_SECRET}", follow_redirects=False)
    assert response.status_code == 401


def test_session_html_without_a_token_renders(app):
    """``session.html`` is exempt: no token, no cookie, still served."""
    response = _client(app).get("/static/session.html", follow_redirects=False)
    assert response.status_code == 200


def test_session_html_with_a_wrong_token_falls_through_to_render(app):
    """On the exempt page a bad token does not exchange — it just renders.

    ``session.html`` is reachable with no credential, so refusing a wrong
    ``?token=`` there would invent a refusal for a route that is open by design.
    The exchange declines and the page is served.
    """
    response = _client(app).get("/static/session.html?token=not-the-secret", follow_redirects=False)
    assert response.status_code == 200
    assert response.headers.get("set-cookie") is None


def test_unconfigured_credentials_render_instead_of_crashing(app, monkeypatch):
    """On the exempt page an unpopulatable holder degrades to a rendered page.

    The container shape that declares a bind host but supplies no operator
    secret makes ``get_web_credentials`` raise ``RuntimeError``. The gate skips
    the credential ladder on an exempt path, so the only code that would touch
    credentials there is the exchange — and it must decline rather than turn an
    open page into a 500 or a 503.
    """

    def _raise(_app=None):
        raise RuntimeError("web credentials are unavailable")

    monkeypatch.setattr("osprey.interfaces.common_middleware.get_web_credentials", _raise)
    response = _client(app).get(
        f"/static/session.html?token={OPERATOR_SECRET}", follow_redirects=False
    )
    assert response.status_code == 200
    assert response.headers.get("set-cookie") is None


# --------------------------------------------------------------------------- #
# Middleware unit: _exchange_query_token scoping and one-comparison property
# --------------------------------------------------------------------------- #


def _http_scope(method: str, path: str, query_string: bytes) -> dict[str, Any]:
    return {"type": "http", "method": method, "path": path, "query_string": query_string}


@pytest.mark.parametrize(
    ("scope", "expected"),
    [
        (_http_scope("GET", "/", b"token=abc"), "abc"),
        (_http_scope("GET", "/static/session.html", b"token=abc"), "abc"),
        # Later params, and a leading unrelated one, do not confuse the read.
        (_http_scope("GET", "/", b"foo=1&token=abc"), "abc"),
        # Non-GET, non-exchange-path, blank and whitespace-only tokens: no credential.
        (_http_scope("POST", "/", b"token=abc"), None),
        (_http_scope("GET", "/api/config", b"token=abc"), None),
        (_http_scope("GET", "/", b""), None),
        (_http_scope("GET", "/", b"token="), None),
        (_http_scope("GET", "/", b"token=%20%20"), None),
    ],
)
def test_exchange_query_token_scoping(scope, expected):
    """The extractor accepts a token only for a GET to an exchange path."""
    assert _exchange_query_token(scope) == expected


def test_exchange_query_token_ignores_a_websocket():
    """A websocket handshake synthesises GET; a URL token must not open one."""
    scope = {
        "type": "websocket",
        "method": "GET",
        "path": "/",
        "query_string": f"token={OPERATOR_SECRET}".encode(),
    }
    assert _exchange_query_token(scope) is None


def _run_gate(scope, credentials, downstream=None) -> list[dict[str, Any]]:
    """Drive the gate over one raw ASGI scope, returning what it sent."""
    stub_app = SimpleNamespace(state=SimpleNamespace(web_credentials=credentials))
    scope = {**scope, "app": stub_app}

    async def _default(scope, receive, send):  # pragma: no cover - overridden below
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    sent: list[dict[str, Any]] = []

    async def _receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def _send(message):
        sent.append(message)

    asyncio.run(WebAuthMiddleware(downstream or _default)(scope, _receive, _send))
    return sent


def test_valid_query_token_does_exactly_one_operator_comparison():
    """The gate answers a ``?token=`` GET after a single constant-time compare."""
    credentials = WebCredentials(operator_secret=OPERATOR_SECRET, panel_token=PANEL_TOKEN)
    calls: list[str | None] = []
    original = credentials.verify_operator

    def _counting(candidate: str | None) -> bool:
        calls.append(candidate)
        return original(candidate)

    credentials.verify_operator = _counting  # type: ignore[method-assign]

    reached: list[bool] = []

    async def _downstream(scope, receive, send):  # pragma: no cover - must not run
        reached.append(True)

    sent = _run_gate(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "query_string": f"token={OPERATOR_SECRET}".encode(),
            "headers": [],
        },
        credentials,
        _downstream,
    )

    # The gate answers the exchange itself; nothing downstream ever runs.
    assert reached == []
    assert calls == [OPERATOR_SECRET]  # exactly one operator comparison
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    assert status == 303


def test_wrong_query_token_is_refused_by_the_gate():
    """A ``?token=`` GET whose secret is wrong is refused, not passed through."""
    credentials = WebCredentials(operator_secret=OPERATOR_SECRET, panel_token=PANEL_TOKEN)
    reached: list[bool] = []

    async def _downstream(scope, receive, send):  # pragma: no cover - must not run
        reached.append(True)

    sent = _run_gate(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "query_string": b"token=wrong",
            "headers": [],
        },
        credentials,
        _downstream,
    )

    assert reached == []
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    assert status == 401
