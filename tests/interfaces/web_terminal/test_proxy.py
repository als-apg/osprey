"""The panel proxy as a credential boundary, in both directions.

Everything reaching ``/panel/<id>/...`` was authenticated as the operator at
the terminal's origin, and the browser attaches that proof — session cookie,
``Authorization``, and behind the multi-user reverse proxy an
``X-Osprey-Terminal-Secret`` header — to panel requests as well, because a
panel shares that origin. The backend on the far side of the proxy hop does
not share the trust: it may be a facility Grafana, or a URL an agent
registered at runtime.

These tests pin every side of that boundary:

* nothing that identifies the operator crosses the hop outbound, for any panel;
* a request reaches a backend carrying the owner *this process* vouches for and
  never the one the browser claimed — and that minted name is the name the
  terminal's own gate gives the very same credential, in both deployment
  shapes, because a backend filing work under one and a chip looked up under
  the other would gate the wrong user's writes;
* the operator secret is re-issued only toward a backend that is *both*
  config-declared *and* addressed at loopback — with every ambiguous case
  (unresolvable host, runtime registration, off-box address) resolving to no
  injection rather than to a leak;
* nothing a backend sends back can act on the terminal's origin: no
  ``Set-Cookie`` (which would land in the operator's jar for the terminal, on
  the terminal's own session-cookie name), no CORS grant; and
* no proxied request follows a redirect, so a backend can steer neither the
  secret nor this process's network reach.

**Auth here is explicit.** The module opts out of the shared TestClient auth
seam (``pytestmark`` below), which force-sets the operator header on every
request: with the seam on, a test that puts its own value in that header never
gets it onto the wire — the app sees the seam's — so the strip assertions pass
whether or not the proxy strips anything. Requests instead authenticate the way
a real caller does, and which way is load-bearing per test:

* :data:`OPERATOR_HEADERS` — the multi-user shape, nginx's operator-secret
  header plus what the browser attached. Used where the point is that an
  *inbound* credential does not survive the hop.
* a **session cookie** (:func:`_cookie_client`) — the single-user browser shape,
  carrying no secret header at all. Used where the point is that the secret
  observed upstream was *injected by this process*, which a request that
  carried one inbound could never show.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.common_middleware import session_cookie_name
from osprey.interfaces.web_auth import BIND_HOST_ENV, WebCredentials
from osprey.interfaces.web_terminal.app import UNIVERSAL_PANELS, create_app
from osprey.interfaces.web_terminal.routes.proxy import (
    _PANEL_STATE_MAP,
    TERMINAL_SECRET_HEADER,
    UPSTREAM_OPEN_TIMEOUT,
    _backend_is_loopback,
    _is_stripped_header,
)
from osprey.utils.identity import AUDIT_IDENTITY_ENV, TERMINAL_USER_ENV, acting_identity
from osprey.utils.owner_header import OWNER_HEADER

pytestmark = pytest.mark.no_auth_seam

#: The secret this process holds, and the only value a trusted backend may see.
OPERATOR_SECRET = "operator-secret-value"

#: What a multi-user operator request carries on the wire: the browser's cookie
#: and ``Authorization``, plus the secret nginx stamps on.
#:
#: The secret is the *live* one, not a stand-in. The terminal's own gate checks
#: this header before any other credential and refuses a value that does not
#: verify (see ``TestForgedSecretStopsAtTheGate``), so the live secret is the
#: only one that can reach the proxy at all — which makes it exactly the value
#: that must not cross the hop to a backend that has not earned it.
OPERATOR_HEADERS = {
    # Mixed case on purpose: the strip is case-insensitive, a header name is not.
    "Cookie": "osprey_session=live-session-id",
    "Authorization": "Bearer operator-bearer",
    "X-Osprey-Terminal-Secret": OPERATOR_SECRET,
}

#: One framework panel, taken from the registry-derived map rather than named,
#: so a registry that grows or renames a panel does not silently skip this.
FRAMEWORK_PANEL_ID, FRAMEWORK_STATE_ATTR = sorted(_PANEL_STATE_MAP.items())[0]

#: The panels that must never be handed the operator secret.
UNTRUSTED_PANELS = ("facility", "registered")

#: The owner header's wire spelling — what :func:`_lower` leaves behind.
#: Derived from the shared constant so a rename cannot leave the assertions
#: below watching a header name nothing sends any more.
OWNER_WIRE_NAME = OWNER_HEADER.lower()

#: One named roster user's secret, distinct from the deployment-wide one: the
#: gate names an account from the secret that matched, so the two readings of
#: "operator" have to be told apart by which value was presented.
ROSTER_SECRET = "roster-alice-b-secret-value"

#: The account that secret belongs to, as the build renders the roster.
ROSTER_ACCOUNT = "alice-b"


def _build_app(workspace_dir, custom_panels, framework_url=None):
    """Create an app whose panel set is exactly ``custom_panels``.

    The client is entered *inside* the config patches because the lifespan
    re-reads the panel config: leaving it outside means the app starts with the
    real (empty) panel set and every proxied request 404s.
    """
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
        with TestClient(app) as client:
            # Applied after the lifespan, which writes both of these itself.
            # Pinning the credentials here gives the proxy — and the gate in
            # front of it — a known secret without populating (and popping)
            # anything from the environment.
            app.state.web_credentials = WebCredentials(
                operator_secret=OPERATOR_SECRET,
                panel_token="panel-token-value",
            )
            if framework_url is not None:
                setattr(app.state, FRAMEWORK_STATE_ATTR, framework_url)
            yield app, client


def _cookie_client(app) -> TestClient:
    """A client authenticated by a live session cookie and nothing else.

    The single-user browser shape. It carries no ``X-Osprey-Terminal-Secret``
    inbound, so a secret seen on the upstream hop can only have been injected by
    the proxy — the distinction the old inbound-secret tests could not draw.

    The cookie name is derived exactly as the gate derives it, so the two agree
    whatever ``OSPREY_WEB_PORT`` holds in this process.
    """
    client = TestClient(app)
    client.cookies.clear()
    client.cookies.set(session_cookie_name(), app.state.web_credentials.create_session())
    return client


@pytest.fixture
def identity_markers(monkeypatch):
    """Clear the process identity markers; each test sets only what it means.

    Both sides of the owner question read them per call — the proxy through
    ``acting_identity()``, the gate through ``web_auth._own_secret_operator``,
    which is what lets one image serve a ``web-<user>`` container, a shared
    sidecar and a single-user host. A marker inherited from the ambient
    environment would silently decide which of those shapes a test exercises.
    """
    monkeypatch.delenv(BIND_HOST_ENV, raising=False)
    monkeypatch.delenv(TERMINAL_USER_ENV, raising=False)
    monkeypatch.delenv(AUDIT_IDENTITY_ENV, raising=False)
    return monkeypatch


def _roster_credentials(app, account=ROSTER_ACCOUNT):
    """Re-pin ``app``'s credentials with one named roster user beside the own secret.

    The multi-user holder: nginx injects each user's own secret into their
    ``web-<user>`` container, and the same secret is that user's roster entry,
    which is what lets the gate answer with an account name rather than with
    the coarser own-secret reading.
    """
    credentials = WebCredentials(
        operator_secret=OPERATOR_SECRET,
        panel_token="panel-token-value",
        roster_secrets=(ROSTER_SECRET,),
        roster_owners=(account,),
    )
    app.state.web_credentials = credentials
    return credentials


def _owners_of(captured):
    """Every header key in ``captured`` that a backend reads as the owner."""
    return [name for name in captured if name.lower() == OWNER_WIRE_NAME]


@pytest.fixture
def workspace_dir(tmp_path):
    ws = tmp_path / "_agent_data"
    ws.mkdir()
    return ws


@pytest.fixture
def panels_app(workspace_dir):
    """An app carrying one panel of every trust shape the boundary must tell apart."""
    custom = [
        # Config-declared, loopback → the only shape that earns the secret.
        {
            "id": "trusted",
            "label": "TRUSTED",
            "url": "http://127.0.0.1:9000",
            "configDefined": True,
        },
        # Config-declared but off-box: a facility Grafana the operator listed.
        {
            "id": "facility",
            "label": "FACILITY",
            "url": "http://grafana.facility.lan:3000",
            "configDefined": True,
        },
        # Loopback but registered at runtime — the shape an agent can create.
        {"id": "registered", "label": "REGISTERED", "url": "http://127.0.0.1:9001"},
    ]
    yield from _build_app(workspace_dir, custom, framework_url="http://127.0.0.1:9200")


def _capture_request(app, response_factory=None):
    """Stub ``proxy_client.request`` and return the dict it records headers into.

    ``follow_redirects`` is accepted (and defaulted) because the proxy now sends
    every request with it — a stub that rejected the keyword would be silently
    retried without it by ``_request_no_redirect``'s ``TypeError`` fallback, and
    the redirect assertions would then be testing the fallback.
    """
    captured: dict[str, str] = {}

    async def fake_request(*, method, url, headers, content, follow_redirects=True):
        captured.update(headers)
        if response_factory is not None:
            return response_factory()
        return httpx.Response(200, json={"ok": True}, headers={"content-type": "application/json"})

    app.state.proxy_client.request = AsyncMock(side_effect=fake_request)
    return captured


def _lower(headers):
    return {k.lower(): v for k, v in headers.items()}


class _FakeStreamResponse:
    """A streamed upstream response, as ``client.send(stream=True)`` returns one."""

    def __init__(self, *, status_code=200, headers=None, chunks=(b"data: hello\n\n",)):
        self.status_code = status_code
        self.headers = httpx.Headers(headers or {"content-type": "text/event-stream"})
        self._chunks = chunks
        self.closed = False

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk

    async def aclose(self):
        self.closed = True


def _capture_sse(app, upstream):
    """Stub the SSE branch's ``build_request``/``send``; return the header dict."""
    captured: dict[str, str] = {}
    real_build = app.state.proxy_client.build_request

    def spy_build(**kwargs):
        captured.update(kwargs["headers"])
        return real_build(**kwargs)

    app.state.proxy_client.build_request = spy_build
    app.state.proxy_client.send = AsyncMock(return_value=upstream)
    return captured


class TestInboundCredentialStrip:
    """No header identifying the operator crosses the proxy hop."""

    @pytest.mark.parametrize("panel_id", ["trusted", "facility", "registered"])
    def test_operator_headers_never_reach_backend(self, panels_app, panel_id):
        app, client = panels_app
        captured = _capture_request(app)

        resp = client.get(f"/panel/{panel_id}/api/status", headers=OPERATOR_HEADERS)

        assert resp.status_code == 200
        seen = _lower(captured)
        assert "cookie" not in seen
        # No inbound credential value survives, under this header name or any
        # other. The secret is asserted separately, per panel trust shape:
        # the trusted panel legitimately receives it (injected, not relayed).
        forwarded = " ".join(captured.values())
        assert "live-session-id" not in forwarded
        assert "operator-bearer" not in forwarded

    @pytest.mark.parametrize("panel_id", UNTRUSTED_PANELS)
    def test_inbound_secret_does_not_reach_an_untrusted_backend(self, panels_app, panel_id):
        """The live secret arriving inbound must not ride the hop to a backend
        that has not earned it — under its own name or any other."""
        app, client = panels_app
        captured = _capture_request(app)

        client.get(f"/panel/{panel_id}/api/status", headers=OPERATOR_HEADERS)

        assert "x-osprey-terminal-secret" not in _lower(captured)
        assert OPERATOR_SECRET not in " ".join(captured.values())

    def test_authorization_is_dropped_not_relayed(self, panels_app):
        """An inbound Authorization header is dropped for an ordinary panel."""
        app, client = panels_app
        captured = _capture_request(app)

        client.get("/panel/registered/api/status", headers=OPERATOR_HEADERS)

        assert "authorization" not in _lower(captured)

    def test_benign_headers_still_forwarded(self, panels_app):
        """The strip is targeted — ordinary request headers still cross."""
        app, client = panels_app
        captured = _capture_request(app)

        client.get(
            "/panel/registered/api/status",
            headers={**OPERATOR_HEADERS, "x-custom-thing": "kept", "accept": "application/json"},
        )

        seen = _lower(captured)
        assert seen.get("x-custom-thing") == "kept"
        assert seen.get("x-forwarded-prefix") == "/panel/registered"

    def test_sse_path_strips_too(self, panels_app):
        """The streaming branch builds its own request and must strip identically."""
        app, client = panels_app
        captured = _capture_sse(app, _FakeStreamResponse())

        resp = client.get(
            "/panel/registered/events",
            headers={**OPERATOR_HEADERS, "accept": "text/event-stream"},
        )

        assert resp.status_code == 200
        seen = _lower(captured)
        assert "cookie" not in seen
        assert "authorization" not in seen
        assert "x-osprey-terminal-secret" not in seen
        assert OPERATOR_SECRET not in " ".join(captured.values())

    def test_underscore_spelled_secret_is_stripped(self, panels_app):
        """The underscore spelling folds to the same WSGI var and must go too.

        Authenticated by cookie, because that is the only way this request
        exists: the gate matches the header name exactly, so the underscore
        spelling is not a credential to it — it is a caller who authenticated
        some other way trying to smuggle a header that reconstitutes as the
        secret on a WSGI/CGI backend.
        """
        app, _client = panels_app
        captured = _capture_request(app)

        _cookie_client(app).get(
            "/panel/registered/api/status",
            headers={"X_Osprey_Terminal_Secret": "underscore-secret"},
        )

        assert "underscore-secret" not in " ".join(captured.values())


class TestBrowserOriginIsNotRelayed:
    """The browser's ``Origin`` stops at the terminal; the hop carries none.

    ``Origin`` names the page that made the request — the terminal — and the
    terminal's own gate has already held it against the terminal's origin
    before this route runs. Relayed onward it reaches a backend whose gate
    compares it against *that backend's* address (the bluesky sidecar's own
    slot, say), which the terminal's origin never equals: every write
    from a proxied panel is then refused as cross-origin while every read
    succeeds. On this hop the proxy is a new client, and a client that sends
    no ``Origin`` is what a gated backend admits on the operator secret (see
    ``test_operator_header_with_no_origin_is_allowed`` in the gate's tests).
    """

    #: What the terminal's own gate demands on a write from this TestClient:
    #: its Host is ``testserver``, so this is the terminal's origin.
    TERMINAL_ORIGIN = "http://testserver"

    def test_write_from_a_cookie_session_carries_no_origin(self, panels_app):
        """Single-user shape: cookie-authenticated POST, Origin dropped."""
        app, _ = panels_app
        client = _cookie_client(app)
        captured = _capture_request(app)

        resp = client.post(
            "/panel/trusted/queue/items",
            json={"draft_revision": 1},
            headers={"Origin": self.TERMINAL_ORIGIN},
        )

        assert resp.status_code == 200
        assert "origin" not in _lower(captured)

    def test_write_in_the_multi_user_shape_carries_no_origin(self, panels_app):
        """Multi-user shape: nginx-stamped operator header, Origin dropped."""
        app, client = panels_app
        captured = _capture_request(app)

        resp = client.post(
            "/panel/trusted/queue/start",
            headers={**OPERATOR_HEADERS, "Origin": self.TERMINAL_ORIGIN},
        )

        assert resp.status_code == 200
        assert "origin" not in _lower(captured)

    def test_sse_path_carries_no_origin(self, panels_app):
        """The streaming branch builds its own request and must drop it too."""
        app, client = panels_app
        captured = _capture_sse(app, _FakeStreamResponse())

        resp = client.get(
            "/panel/trusted/queue/events",
            headers={
                **OPERATOR_HEADERS,
                "accept": "text/event-stream",
                "Origin": self.TERMINAL_ORIGIN,
            },
        )

        assert resp.status_code == 200
        assert "origin" not in _lower(captured)


class TestStripPredicate:
    """``_is_stripped_header`` folds ``_``→``-`` before matching."""

    @pytest.mark.parametrize(
        "name",
        [
            "cookie",
            "Cookie",
            "authorization",
            "AUTHORIZATION",
            "x-osprey-terminal-secret",
            "X-Osprey-Terminal-Secret",
            "X_Osprey_Terminal_Secret",
            "x_osprey_terminal_secret",
            # Spelled out rather than derived: this is the wire name the gate,
            # the MCP layer and the bridge each reach independently, and the
            # strip is only closed if the proxy answers to that same text.
            "x-osprey-owner",
            "X-Osprey-Owner",
            "X_Osprey_Owner",
        ],
    )
    def test_stripped(self, name):
        assert _is_stripped_header(name) is True

    @pytest.mark.parametrize(
        "name",
        ["accept", "x-forwarded-prefix", "content-type", "x-osprey-terminal", "user-agent"],
    )
    def test_not_stripped(self, name):
        assert _is_stripped_header(name) is False


class TestSecretInjection:
    """The secret is re-issued only to a config-declared loopback backend.

    Driven through :func:`_cookie_client`: the request carries no secret
    inbound, so a secret on the upstream hop is proof of injection rather than
    of relay.
    """

    def test_config_defined_loopback_panel_gets_secret(self, panels_app):
        app, _client = panels_app
        captured = _capture_request(app)

        _cookie_client(app).get("/panel/trusted/api/status")

        assert _lower(captured).get("x-osprey-terminal-secret") == OPERATOR_SECRET

    def test_framework_panel_gets_secret(self, panels_app):
        """A framework sidecar's URL comes from app.state, so it is declared."""
        app, _client = panels_app
        captured = _capture_request(app)

        _cookie_client(app).get(f"/panel/{FRAMEWORK_PANEL_ID}/api/status")

        assert _lower(captured).get("x-osprey-terminal-secret") == OPERATOR_SECRET

    def test_config_defined_offbox_panel_gets_nothing(self, panels_app):
        """Config-declared is not enough: the secret must not leave the machine."""
        app, _client = panels_app
        captured = _capture_request(app)

        _cookie_client(app).get("/panel/facility/api/status")

        assert "x-osprey-terminal-secret" not in _lower(captured)
        assert OPERATOR_SECRET not in " ".join(captured.values())

    def test_runtime_registered_loopback_panel_gets_nothing(self, panels_app):
        """Loopback is not enough either: runtime registration is agent-reachable."""
        app, _client = panels_app
        captured = _capture_request(app)

        _cookie_client(app).get("/panel/registered/api/status")

        assert "x-osprey-terminal-secret" not in _lower(captured)
        assert OPERATOR_SECRET not in " ".join(captured.values())

    def test_websocket_injection_needs_no_inbound_secret(self, panels_app):
        """The WS leg injects for a trusted panel from a cookie-only handshake.

        The HTTP tests above cannot borrow this one's client, so it is pinned
        here: nothing in the handshake carried a secret, and the upstream one
        still appears.
        """
        app, _client = panels_app
        fake = _FakeConnect()

        with patch("websockets.connect", fake):
            with _cookie_client(app).websocket_connect(
                "/panel/trusted/ws/stream", headers={"origin": "http://testserver"}
            ):
                pass

        assert fake.kwargs["additional_headers"] == {TERMINAL_SECRET_HEADER: OPERATOR_SECRET}

    def test_credentials_failure_forwards_no_secret(self, panels_app, caplog):
        """If the process cannot produce credentials, load unauthenticated — never crash."""
        app, client = panels_app
        captured = _capture_request(app)

        # Only the proxy's own lookup is broken; the gate in front keeps its
        # working one, so the request still reaches the route.
        with patch(
            "osprey.interfaces.web_terminal.routes.proxy.get_web_credentials",
            side_effect=RuntimeError("no credentials in this process"),
        ):
            with caplog.at_level(logging.WARNING):
                resp = client.get("/panel/trusted/api/status", headers=OPERATOR_HEADERS)

        assert resp.status_code == 200
        assert "x-osprey-terminal-secret" not in _lower(captured)
        assert any("without the operator secret" in r.message for r in caplog.records)


class TestOwnerMint:
    """Every hop carries the owner this process vouches for, or none at all.

    ``X-Osprey-Owner`` is the name a backend files queued work under, and the
    name whose per-target chip then gates that work's writes — so unlike the
    secret it is not a credential being protected but an assertion being made.
    The rows here pin the two halves that make it worth believing: a value the
    browser sent never reaches a backend, and the value that does is minted
    from this container's own identity.

    Driven through :func:`_cookie_client` wherever the point is the mint alone:
    a cookie authorises without naming anyone, so the gate in front stamps no
    owner and the header observed upstream can only have come from this hop.
    """

    def test_declared_loopback_backend_is_told_who_is_driving(self, panels_app, identity_markers):
        app, _client = panels_app
        identity_markers.setenv(TERMINAL_USER_ENV, "alice")
        captured = _capture_request(app)

        _cookie_client(app).get("/panel/trusted/api/status")

        assert _lower(captured).get(OWNER_WIRE_NAME) == "alice"

    def test_framework_sidecar_is_told_who_is_driving(self, panels_app, identity_markers):
        """The shape this exists for: the Bluesky sidecar is a framework panel.

        Its URL is published on ``app.state`` by the launcher that started it,
        which is what makes it declared, and it listens on loopback — so it is
        handed both a credential and the name of the human behind the request
        it will queue a plan for.
        """
        app, _client = panels_app
        identity_markers.setenv(TERMINAL_USER_ENV, "alice")
        captured = _capture_request(app)

        _cookie_client(app).get(f"/panel/{FRAMEWORK_PANEL_ID}/queue/items")

        assert _lower(captured).get(OWNER_WIRE_NAME) == "alice"

    def test_inbound_owner_never_reaches_the_backend(self, panels_app, identity_markers):
        """From alice's terminal, a browser claiming ``bob`` arrives as alice.

        Exactly one owner header goes upstream, under the minted spelling. Two
        would be as bad as the wrong one: a backend reading the first of a pair
        would be back to reading the browser's claim, and which of the two a
        framework reads is not a property this proxy should have to rely on.
        """
        app, client = panels_app
        identity_markers.setenv(TERMINAL_USER_ENV, "alice")
        captured = _capture_request(app)

        resp = client.get(
            "/panel/trusted/api/status",
            headers={**OPERATOR_HEADERS, "X-Osprey-Owner": "bob"},
        )

        assert resp.status_code == 200
        assert _owners_of(captured) == [OWNER_HEADER]
        assert _lower(captured)[OWNER_WIRE_NAME] == "alice"
        assert "bob" not in " ".join(captured.values())

    def test_underscore_spelled_owner_does_not_survive(self, panels_app, identity_markers):
        """The WSGI spelling folds to the same header and must not ride along.

        A backend behind a WSGI/CGI layer reads both spellings as the one
        ``HTTP_X_OSPREY_OWNER`` key, so a claim sent with underscores would
        reconstitute on the far side exactly as the dash spelling does.
        """
        app, _client = panels_app
        identity_markers.setenv(TERMINAL_USER_ENV, "alice")
        captured = _capture_request(app)

        _cookie_client(app).get(
            "/panel/trusted/api/status",
            headers={"X_Osprey_Owner": "bob"},
        )

        assert _owners_of(captured) == [OWNER_HEADER]
        assert "bob" not in " ".join(captured.values())

    @pytest.mark.parametrize("panel_id", UNTRUSTED_PANELS)
    def test_a_backend_that_earns_no_credential_is_told_nothing(
        self, panels_app, identity_markers, panel_id
    ):
        """Off-box and runtime-registered backends learn no account name.

        Neither is owed it: a registered listener is a surface the agent's own
        sandbox can create, and an off-box host is one nobody at this console
        authenticated to. A claim sent inbound does not reach them either.
        """
        app, _client = panels_app
        identity_markers.setenv(TERMINAL_USER_ENV, "alice")
        captured = _capture_request(app)

        _cookie_client(app).get(
            f"/panel/{panel_id}/api/status",
            headers={"X-Osprey-Owner": "bob"},
        )

        assert _owners_of(captured) == []
        assert "bob" not in " ".join(captured.values())

    def test_a_name_the_guard_refuses_is_sent_as_no_owner_at_all(
        self, panels_app, identity_markers, caplog
    ):
        """An unspellable account costs its owner attribution, never the request.

        The value rides a client that encodes headers as ASCII and raises on
        anything else, so a name outside the allowlist the reading end applies
        would fail every proxied request for that user. It is dropped here
        instead — with the guard's own one warning naming the shape — and the
        work the backend files is owner-less, exactly as it is for a login that
        named nobody.

        The unspellable account reaches the mint through the local-account rung
        rather than through ``OSPREY_TERMINAL_USER``, because that marker also
        spells the container's URL prefix: setting it here would send the same
        characters out in ``x-forwarded-prefix`` as well, and the row would then
        be pinning two different header failures at once.
        """
        app, _client = panels_app
        identity_markers.setattr("getpass.getuser", lambda: "renée")
        captured = _capture_request(app)

        with caplog.at_level(logging.WARNING):
            resp = _cookie_client(app).get("/panel/trusted/api/status")

        assert resp.status_code == 200
        assert _owners_of(captured) == []
        logged = [r.getMessage() for r in caplog.records]
        assert any(OWNER_HEADER.lower() in message.lower() for message in logged)
        assert not any("renée" in message for message in logged)

    def test_sse_branch_mints_too(self, panels_app, identity_markers):
        """The streaming branch sends the same built header set, so it must agree."""
        app, _client = panels_app
        identity_markers.setenv(TERMINAL_USER_ENV, "alice")
        captured = _capture_sse(app, _FakeStreamResponse())

        resp = _cookie_client(app).get(
            "/panel/trusted/events",
            headers={"accept": "text/event-stream", "X-Osprey-Owner": "bob"},
        )

        assert resp.status_code == 200
        assert _owners_of(captured) == [OWNER_HEADER]
        assert _lower(captured)[OWNER_WIRE_NAME] == "alice"


class TestOwnerDriftAgainstTheGate:
    """The proxy mints the name the gate would give the same credential.

    Two independent implementations answer "who is this": the proxy asks the
    identity ladder, and the sidecar's own gate asks which secret matched. They
    must agree, because they attribute the same work — the proxy names the
    request that queues a plan while the gate names the request that reaches
    the panel's own address, and a plan filed under one name while its chip is
    looked up under the other is gated by the wrong user's posture. These rows
    compare the two values rather than either against a literal, so a change to
    one ladder that the other does not follow fails here.
    """

    def test_roster_user_matches(self, panels_app, identity_markers):
        """The multi-user shape: alice's own container, alice's roster secret."""
        app, client = panels_app
        identity_markers.setenv(BIND_HOST_ENV, "0.0.0.0")
        identity_markers.setenv(TERMINAL_USER_ENV, ROSTER_ACCOUNT)
        credentials = _roster_credentials(app)
        captured = _capture_request(app)

        resp = client.get(
            "/panel/trusted/api/status",
            headers={TERMINAL_SECRET_HEADER: ROSTER_SECRET},
        )

        gate_value = credentials.identify_operator(ROSTER_SECRET)
        assert resp.status_code == 200
        assert gate_value == ROSTER_ACCOUNT
        assert _lower(captured).get(OWNER_WIRE_NAME) == gate_value

    def test_single_user_host_matches(self, panels_app, identity_markers):
        """``osprey up``: one account runs the terminal and answers its own gate.

        No bind host is the tell that nothing fronts this process, so the own
        secret names the human at the console rather than nobody — and it has
        to be the same human the proxy mints, which is the whole point of the
        single-user row: here the two ladders reach the account by different
        rungs and still have to land on one name.

        The local account is pinned rather than read: unpinned, the row asserts
        against whatever account happens to be running the suite, and a name
        the header guard refuses would fail it for a reason it is not about.
        """
        app, client = panels_app
        identity_markers.setattr("getpass.getuser", lambda: "console-account")
        credentials = app.state.web_credentials
        captured = _capture_request(app)

        resp = client.get(
            "/panel/trusted/api/status",
            headers={TERMINAL_SECRET_HEADER: OPERATOR_SECRET},
        )

        gate_value = credentials.identify_operator(OPERATOR_SECRET)
        assert resp.status_code == 200
        # A name, not the authorised-but-unattributable sentinel.
        assert isinstance(gate_value, str)
        assert _lower(captured).get(OWNER_WIRE_NAME) == gate_value == acting_identity()


class TestForgedSecretStopsAtTheGate:
    """A wrong inbound secret never reaches the proxy to be relayed or dropped.

    This is why the strip tests above authenticate with the *live* secret rather
    than a forged stand-in: the gate checks the presented credential and refuses
    a value that does not verify, without falling through to the cookie. A test
    that sent a forged value would be asserting the gate's refusal, not the
    proxy's strip.
    """

    def test_forged_secret_is_refused_before_any_upstream_call(self, panels_app):
        app, client = panels_app
        captured = _capture_request(app)

        resp = client.get(
            "/panel/trusted/api/status",
            headers={"X-Osprey-Terminal-Secret": "forged-value"},
        )

        assert resp.status_code == 401
        assert app.state.proxy_client.request.await_count == 0
        assert captured == {}

    def test_forged_secret_is_refused_even_with_a_live_cookie(self, panels_app):
        """The gate judges the credential presented, not the best one available."""
        app, _client = panels_app
        _capture_request(app)

        resp = _cookie_client(app).get(
            "/panel/trusted/api/status",
            headers={"X-Osprey-Terminal-Secret": "forged-value"},
        )

        assert resp.status_code == 401
        assert app.state.proxy_client.request.await_count == 0


#: A response header set in which every entry acts on the *embedding* origin
#: rather than on the panel's own document — the class the response filter
#: exists to refuse. Used as the upstream headers at every emit site.
#:
#: * ``Set-Cookie`` lands in the operator's jar for the terminal (cookies ignore
#:   ports, and ``osprey_terminal_session_<port>`` is a derivable name).
#: * ``Clear-Site-Data`` reaches the same outcome without naming a cookie: the
#:   browser clears the *response* origin's cookies and storage.
#: * ``Access-Control-*`` publishes an attacker-chosen origin as allowed to read
#:   credentialed responses from the console.
#: * ``Refresh`` navigates the top-level browsing context off-site on any
#:   status, no ``30x`` required.
#: * ``WWW-Authenticate`` raises a native credential dialog attributed to the
#:   console's own origin.
HOSTILE_RESPONSE_HEADERS = {
    "set-cookie": "osprey_terminal_session_8765=evicted; Path=/",
    "clear-site-data": '"cookies", "storage"',
    "access-control-allow-origin": "https://attacker.test",
    "access-control-allow-credentials": "true",
    "refresh": "0; url=https://attacker.test/",
    "www-authenticate": 'Basic realm="OSPREY console login"',
}


def _assert_origin_acting_headers_dropped(headers):
    """Assert no header that acts on the terminal's origin survived the hop."""
    seen = _lower(headers)
    for name in HOSTILE_RESPONSE_HEADERS:
        assert name not in seen, f"{name} was relayed onto the terminal's origin"
    assert not any(name.startswith("access-control-") for name in seen)


class TestResponseBoundary:
    """Nothing a backend sends back may act on the terminal's own origin.

    A proxied response is emitted at ``/panel/<id>/...`` on the terminal's
    origin, so every header in :data:`HOSTILE_RESPONSE_HEADERS` is a header the
    backend gets to aim at the *console* rather than at its own document: it can
    evict the operator's session (``Set-Cookie``, ``Clear-Site-Data``), relax the
    console's origin for an attacker's (``Access-Control-*``), navigate the
    operator off-site (``Refresh``), or raise a credential prompt wearing the
    console's origin (``WWW-Authenticate``). A same-origin-proxied panel needs
    none of them, and each emit site is checked against the whole set rather
    than against the one or two headers that were topical when it was written.
    """

    HOSTILE = {"content-type": "application/json", **HOSTILE_RESPONSE_HEADERS}

    @pytest.mark.parametrize("panel_id", ["trusted", "registered"])
    def test_standard_response_drops_cookies_and_cors(self, panels_app, panel_id):
        app, client = panels_app
        _capture_request(app, lambda: httpx.Response(200, json={"ok": True}, headers=self.HOSTILE))

        resp = client.get(f"/panel/{panel_id}/api/status", headers=OPERATOR_HEADERS)

        assert resp.status_code == 200
        _assert_origin_acting_headers_dropped(resp.headers)
        seen = _lower(resp.headers)
        # The pass-through itself still works.
        assert seen["content-type"].startswith("application/json")

    def test_rewritten_html_response_drops_cookies_and_cors(self, panels_app):
        """The HTML branch builds its own Response and must use the same filter."""
        app, client = panels_app
        _capture_request(
            app,
            lambda: httpx.Response(
                200,
                text="<a href='/static/x.js'>x</a>",
                headers={**self.HOSTILE, "content-type": "text/html"},
            ),
        )

        resp = client.get("/panel/trusted/index.html", headers=OPERATOR_HEADERS)

        _assert_origin_acting_headers_dropped(resp.headers)
        assert "/panel/trusted/static/x.js" in resp.text

    def test_relayed_redirect_drops_cookies_and_cors(self, panels_app):
        """A 30x is relayed, so it is an emit site like any other."""
        app, client = panels_app
        _capture_request(
            app,
            lambda: httpx.Response(302, headers={**self.HOSTILE, "location": "/dashboard"}),
        )

        resp = client.get(
            "/panel/trusted/api/thing", headers=OPERATOR_HEADERS, follow_redirects=False
        )

        assert resp.status_code == 302
        _assert_origin_acting_headers_dropped(resp.headers)
        seen = _lower(resp.headers)
        assert seen["location"] == "/panel/trusted/dashboard"

    def test_sse_response_drops_cookies_and_cors(self, panels_app):
        """The streaming branch emits its own header set too."""
        app, client = panels_app
        _capture_sse(
            app,
            _FakeStreamResponse(headers={**self.HOSTILE, "content-type": "text/event-stream"}),
        )

        resp = client.get(
            "/panel/registered/events",
            headers={**OPERATOR_HEADERS, "accept": "text/event-stream"},
        )

        assert resp.status_code == 200
        _assert_origin_acting_headers_dropped(resp.headers)

    def test_backend_cannot_evict_the_operator_session(self, panels_app):
        """End to end: the browser's terminal session survives a hostile panel.

        The backend sets the terminal's *own* session cookie name — the concrete
        attack the header strip exists to stop — and the client's jar is checked
        afterwards, so this fails if the header reaches the browser at all.

        The same response also carries ``Clear-Site-Data``, which evicts the
        session without naming it: the directive applies to the origin of the
        *response*, which here is the terminal's. The test client has no
        implementation of it, so the jar cannot show that second attack — the
        header simply must not reach a browser that does.
        """
        app, _client = panels_app
        client = _cookie_client(app)
        # The hostile Set-Cookie names whatever the gate's cookie is called here.
        cookie_name = session_cookie_name()
        session_before = client.cookies.get(cookie_name)
        assert session_before
        _capture_request(
            app,
            lambda: httpx.Response(
                200,
                json={"ok": True},
                headers={
                    "content-type": "application/json",
                    "set-cookie": f"{cookie_name}=evicted; Path=/",
                    "clear-site-data": '"cookies", "storage"',
                },
            ),
        )

        resp = client.get("/panel/registered/api/status")

        assert resp.status_code == 200
        seen = _lower(resp.headers)
        assert "set-cookie" not in seen
        assert "clear-site-data" not in seen
        assert client.cookies.get(cookie_name) == session_before

    def test_websocket_handshake_relays_nothing_from_the_backend(self, panels_app):
        """The WS accept is built by this app, so no upstream header can ride it.

        The upstream connects first, and ``websocket.accept()`` then carries
        only the negotiated subprotocol, no headers — pinned here so a later
        change that forwarded the upstream handshake's response headers
        (``Set-Cookie`` included) would be caught.
        """
        app, _client = panels_app
        fake = _FakeConnect()

        with patch("websockets.connect", fake):
            with _cookie_client(app).websocket_connect(
                "/panel/trusted/ws/stream", headers={"origin": "http://testserver"}
            ) as session:
                accepted = session.extra_headers or []

        names = {name.decode("latin-1").lower() for name, _value in accepted}
        assert "set-cookie" not in names
        assert not any(name.startswith("access-control-") for name in names)


class TestRedirectContainment:
    """No proxied request follows a redirect, and none aims the browser off-site.

    Two separate failures ride a *followed* redirect. With the secret injected,
    httpx re-sends every header but Authorization/Cookie to the new origin, so a
    loopback backend answering 30x with an off-box Location exfiltrates it. With
    no secret at all, the request still has this process's *position* to lend: it
    sits inside the container next to services the browser cannot address, so an
    off-box panel answering ``302 Location: http://127.0.0.1:<port>/`` would have
    the proxy fetch that on its behalf. Both are refused the same way — the 30x
    is relayed to the browser, whose re-request goes back through the gate.

    Relaying is not unconditional. A ``Location`` naming a *third* origin is not
    relayed either: it would make ``/panel/<id>/...`` on the console's own origin
    into an open redirect, so it is answered 502 with no ``Location`` at all. The
    two shapes that stay inside the panel — the backend's own origin, and a
    root-absolute or relative path — are re-based into ``<prefix>/panel/<id>``.
    """

    def _record_calls(self, app, response_factory):
        calls: list[dict] = []

        async def fake_request(*, method, url, headers, content, follow_redirects=True):
            calls.append(
                {"url": str(url), "headers": dict(headers), "follow_redirects": follow_redirects}
            )
            return response_factory()

        app.state.proxy_client.request = AsyncMock(side_effect=fake_request)
        return calls

    def test_offbox_redirect_is_not_followed(self, panels_app):
        app, client = panels_app
        calls = self._record_calls(
            app,
            lambda: httpx.Response(
                302,
                headers={"location": "http://evil.example/steal", "content-type": "text/plain"},
            ),
        )

        resp = client.get(
            "/panel/trusted/api/thing", headers=OPERATOR_HEADERS, follow_redirects=False
        )

        # Exactly one upstream call: the loopback hop. The proxy did not make a
        # second request to the off-box Location, so the secret never left.
        assert len(calls) == 1
        assert calls[0]["follow_redirects"] is False
        assert "127.0.0.1" in calls[0]["url"]
        assert _lower(calls[0]["headers"]).get("x-osprey-terminal-secret") == OPERATOR_SECRET
        # Nor is it relayed: handing the browser a third-origin Location from a
        # console URL is an open redirect wearing the console's own origin.
        assert resp.status_code == 502
        assert "location" not in _lower(resp.headers)
        assert "evil.example" not in resp.text

    @pytest.mark.parametrize("panel_id", UNTRUSTED_PANELS)
    def test_uncredentialed_redirect_into_loopback_is_not_followed(self, panels_app, panel_id):
        """SSRF, not credential leak: an untrusted panel must not aim the proxy
        at a service only this process can reach."""
        app, client = panels_app
        calls = self._record_calls(
            app,
            lambda: httpx.Response(302, headers={"location": "http://127.0.0.1:6379/admin"}),
        )

        resp = client.get(
            f"/panel/{panel_id}/api/thing", headers=OPERATOR_HEADERS, follow_redirects=False
        )

        assert len(calls) == 1
        assert calls[0]["follow_redirects"] is False
        # Not followed, and not relayed either: the admin port is a third origin
        # as far as this panel is concerned, whichever side of the box it is on.
        assert resp.status_code == 502
        assert "location" not in _lower(resp.headers)

    def test_sse_redirect_is_not_followed(self, panels_app):
        """The streaming branch has its own follow flag and its own relay."""
        app, client = panels_app
        upstream = _FakeStreamResponse(
            status_code=302, headers={"location": "http://evil.example/steal"}
        )
        _capture_sse(app, upstream)

        resp = client.get(
            "/panel/registered/events",
            headers={**OPERATOR_HEADERS, "accept": "text/event-stream"},
            follow_redirects=False,
        )

        assert app.state.proxy_client.send.await_args.kwargs["follow_redirects"] is False
        assert upstream.closed is True
        assert resp.status_code == 502
        assert "location" not in _lower(resp.headers)

    def test_loopback_internal_redirect_location_is_rebased(self, panels_app):
        app, client = panels_app
        self._record_calls(app, lambda: httpx.Response(302, headers={"location": "/dashboard"}))

        resp = client.get(
            "/panel/trusted/api/thing", headers=OPERATOR_HEADERS, follow_redirects=False
        )

        assert resp.status_code == 302
        # Root-absolute Location re-based into the panel namespace so the
        # browser's re-request comes back through the proxy (and is re-gated).
        assert resp.headers["location"] == "/panel/trusted/dashboard"

    def test_backend_origin_redirect_location_is_rebased(self, panels_app):
        app, client = panels_app
        self._record_calls(
            app,
            lambda: httpx.Response(
                302, headers={"location": "http://127.0.0.1:9000/dashboard/state"}
            ),
        )

        resp = client.get(
            "/panel/trusted/api/thing", headers=OPERATOR_HEADERS, follow_redirects=False
        )

        assert resp.status_code == 302
        assert resp.headers["location"] == "/panel/trusted/dashboard/state"

    def test_offbox_backend_own_origin_location_is_rebased_not_refused(self, panels_app):
        """The refusal is about *third* origins, not about being off-box.

        A config-declared facility panel redirecting to its own origin is the
        ordinary in-panel redirect, so it re-bases like a loopback one; only a
        Location that is neither the backend nor the console is refused.
        """
        app, client = panels_app
        self._record_calls(
            app,
            lambda: httpx.Response(
                302, headers={"location": "http://grafana.facility.lan:3000/d/abc"}
            ),
        )

        resp = client.get(
            "/panel/facility/api/thing", headers=OPERATOR_HEADERS, follow_redirects=False
        )

        assert resp.status_code == 302
        assert resp.headers["location"] == "/panel/facility/d/abc"

    def test_protocol_relative_location_is_refused(self, panels_app):
        """``//evil.test/x`` is an absolute URL wearing the request's scheme."""
        app, client = panels_app
        self._record_calls(
            app, lambda: httpx.Response(302, headers={"location": "//evil.test/steal"})
        )

        resp = client.get(
            "/panel/trusted/api/thing", headers=OPERATOR_HEADERS, follow_redirects=False
        )

        assert resp.status_code == 502
        assert "location" not in _lower(resp.headers)

    def test_unparseable_location_is_refused(self, panels_app):
        """A Location the proxy cannot classify is refused, not emitted.

        ``urlparse`` raises on a malformed IPv6 authority; without the guard the
        panel request would 500, and relaying the value unclassified would be
        the open redirect the refusal exists to close.
        """
        app, client = panels_app
        self._record_calls(
            app, lambda: httpx.Response(302, headers={"location": "http://[::1/steal"})
        )

        resp = client.get(
            "/panel/trusted/api/thing", headers=OPERATOR_HEADERS, follow_redirects=False
        )

        assert resp.status_code == 502
        assert "location" not in _lower(resp.headers)

    def test_terminal_origin_location_is_relayed(self, panels_app):
        """An absolute Location naming the console itself is not off-site.

        It is the same destination a root-absolute path would name, so refusing
        it would break a backend that spells its in-console redirect out in full
        while granting nothing: the browser lands back on the terminal, which
        gates the re-request like any other.
        """
        app, client = panels_app
        self._record_calls(
            app,
            lambda: httpx.Response(
                302, headers={"location": "http://testserver/panel/trusted/dashboard"}
            ),
        )

        resp = client.get(
            "/panel/trusted/api/thing", headers=OPERATOR_HEADERS, follow_redirects=False
        )

        assert resp.status_code == 302
        assert resp.headers["location"] == "http://testserver/panel/trusted/dashboard"

    @pytest.mark.parametrize("request_path", ["/panel/registered", "/panel/registered/"])
    def test_relative_location_from_the_panel_root_stays_in_the_panel(
        self, panels_app, request_path
    ):
        """``Location: dashboard/`` on ``/panel/<id>`` must not lose the id.

        The browser resolves a relative Location against the *request* path, and
        ``/panel/registered`` has no trailing slash — so an unrewritten
        ``dashboard/`` resolves to ``/panel/dashboard/``, a different panel id.
        The proxy resolves it here instead, treating the panel prefix as the
        directory it is.
        """
        app, client = panels_app
        self._record_calls(app, lambda: httpx.Response(302, headers={"location": "dashboard/"}))

        resp = client.get(request_path, headers=OPERATOR_HEADERS, follow_redirects=False)

        assert resp.status_code == 302
        assert resp.headers["location"] == "/panel/registered/dashboard/"

    def test_relative_location_under_a_subpath_resolves_against_it(self, panels_app):
        """Deeper paths keep the browser's own resolution, which was correct."""
        app, client = panels_app
        self._record_calls(app, lambda: httpx.Response(302, headers={"location": "next"}))

        resp = client.get(
            "/panel/registered/wizard/step1", headers=OPERATOR_HEADERS, follow_redirects=False
        )

        assert resp.status_code == 302
        assert resp.headers["location"] == "/panel/registered/wizard/next"

    def test_not_modified_is_not_treated_as_a_redirect(self, panels_app):
        """304 shares the 3xx range but is a cache answer, not a redirect.

        Relaying it through the redirect path would attach a rewritten Location
        and lose the conditional-request semantics the channel catalog depends
        on (``tests/interfaces/web_terminal/test_channels_proxy_cache.py``).
        """
        app, client = panels_app
        self._record_calls(
            app,
            lambda: httpx.Response(304, headers={"etag": '"abc"', "cache-control": "no-cache"}),
        )

        resp = client.get(
            "/panel/trusted/channels", headers=OPERATOR_HEADERS, follow_redirects=False
        )

        assert resp.status_code == 304
        assert "location" not in _lower(resp.headers)
        assert resp.headers["etag"] == '"abc"'
        assert resp.headers["cache-control"] == "no-cache"


class TestDispatcherTokenGate:
    """The events dispatcher token follows the same gate as the operator secret.

    It is a deployment-wide bearer token: config origin alone would still send
    it over the network to whatever host an off-box ``events`` panel names.
    """

    @pytest.fixture
    def events_app(self, workspace_dir, request):
        custom = [
            {
                "id": "events",
                "label": "EVENTS",
                "url": request.param,
                "configDefined": True,
            },
        ]
        yield from _build_app(workspace_dir, custom)

    @pytest.mark.parametrize("events_app", ["http://localhost:8020"], indirect=True)
    def test_loopback_events_panel_gets_the_bearer(self, events_app, monkeypatch):
        app, client = events_app
        monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", "dispatcher-token")
        captured = _capture_request(app)

        client.get("/panel/events/dashboard/state", headers=OPERATOR_HEADERS)

        seen = _lower(captured)
        assert seen.get("authorization") == "Bearer dispatcher-token"
        assert "cookie" not in seen
        # localhost is loopback and the panel is config-defined, so it is also
        # entitled to the operator secret.
        assert seen.get("x-osprey-terminal-secret") == OPERATOR_SECRET

    @pytest.mark.parametrize("events_app", ["http://events.facility.lan:8020"], indirect=True)
    def test_offbox_events_panel_gets_neither_credential(self, events_app, monkeypatch):
        app, client = events_app
        monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", "dispatcher-token")
        captured = _capture_request(app)

        client.get("/panel/events/dashboard/state", headers=OPERATOR_HEADERS)

        seen = _lower(captured)
        assert "authorization" not in seen
        assert "x-osprey-terminal-secret" not in seen
        assert "dispatcher-token" not in " ".join(captured.values())


class TestLoopbackClassification:
    """``_backend_is_loopback`` decides the injection gate; ambiguity means no."""

    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1:9000",
            "http://127.0.0.53:8080/base",
            "https://localhost",
            "http://localhost:8020/",
            "http://[::1]:9000",
            "http://[::ffff:127.0.0.1]:9000",
        ],
    )
    def test_loopback_hosts(self, url):
        assert _backend_is_loopback(url) is True

    @pytest.mark.parametrize(
        "url",
        [
            "http://grafana.facility.lan:3000",
            "http://10.0.0.5:9000",
            "http://0.0.0.0:9000",
            "http://[::]:9000",
            "http://127.0.0.1.attacker.example:9000",
            "http://localhost.attacker.example:9000",
            "http://attacker.example/?h=localhost",
            # *.localhost is NOT trusted: the name is reserved for loopback by
            # RFC 6761, but the socket still resolves it through the system
            # resolver, which some stacks forward upstream — so the spelling
            # cannot be allowed to earn the secret.
            "http://panel.localhost:3000",
            "",
            "not a url at all",
            "http://",
        ],
    )
    def test_non_loopback_hosts(self, url):
        assert _backend_is_loopback(url) is False


class TestHeaderSpellingPin:
    """The proxy and the nginx renderer must name the same header."""

    def test_matches_deployment_renderer(self):
        from osprey.deployment.web_terminals.render import TERMINAL_SECRET_HEADER as rendered

        assert TERMINAL_SECRET_HEADER == rendered

    def test_stripped_name_matches_injected_name(self):
        from osprey.interfaces.web_terminal.routes.proxy import _STRIPPED_REQUEST_HEADERS

        assert TERMINAL_SECRET_HEADER.lower() in _STRIPPED_REQUEST_HEADERS

    def test_the_owner_name_minted_here_is_the_one_the_gate_drops(self):
        """One spelling for the header, wherever it is minted or refused.

        The proxy mints it, the terminal's gate drops every inbound one, and
        the bridge reads it — three modules whose agreement is what makes a
        forged claim unusable rather than merely unlikely. They agree by
        importing the same constant, and this says so out loud.
        """
        from osprey.interfaces.common_middleware import _OWNER_HEADER_WIRE

        assert _OWNER_HEADER_WIRE.decode("latin-1") == OWNER_HEADER.lower()
        assert _is_stripped_header(OWNER_HEADER) is True


class _FakeUpstreamSocket:
    """A websocket upstream that stays open until the relay task is cancelled."""

    def __init__(self):
        self.sent: list[object] = []

    async def send(self, data):
        self.sent.append(data)

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.Event().wait()  # pragma: no cover - cancelled at teardown
        raise AssertionError("unreachable")


class _FakeConnect:
    """Stands in for ``websockets.connect``, recording the handshake arguments."""

    def __init__(self):
        self.target = None
        self.kwargs = None

    def __call__(self, target, **kwargs):
        self.target = target
        self.kwargs = kwargs
        return self

    async def __aenter__(self):
        return _FakeUpstreamSocket()

    async def __aexit__(self, *exc_info):
        return False


class TestWebSocketBoundary:
    """The WS upstream handshake carries the secret, or nothing at all."""

    def _connect(self, client, path):
        fake = _FakeConnect()
        with patch("websockets.connect", fake):
            with client.websocket_connect(
                path, headers={"X-Osprey-Terminal-Secret": OPERATOR_SECRET}
            ):
                pass
        return fake

    def test_trusted_panel_ws_carries_secret(self, panels_app):
        _app, client = panels_app

        fake = self._connect(client, "/panel/trusted/ws/stream")

        assert fake.kwargs["additional_headers"] == {TERMINAL_SECRET_HEADER: OPERATOR_SECRET}

    def test_trusted_panel_ws_refuses_redirects(self, panels_app):
        """Carrying the secret, the handshake must not follow a redirect off loopback."""
        _app, client = panels_app

        fake = self._connect(client, "/panel/trusted/ws/stream")

        # process_redirect returning the exception unchanged declines the
        # redirect, so additional_headers never rides one to a new origin.
        override = getattr(fake, "process_redirect", None)
        assert callable(override)
        sentinel = RuntimeError("redirect")
        assert override(sentinel) is sentinel

    def test_registered_panel_ws_carries_no_headers(self, panels_app):
        _app, client = panels_app

        fake = self._connect(client, "/panel/registered/ws/stream")

        assert fake.kwargs["additional_headers"] is None
        # No secret in flight → the default redirect handling is left in place.
        assert getattr(fake, "process_redirect", None) is None

    def test_upstream_handshake_outlasts_a_kernel_start(self, panels_app):
        """The upstream open timeout covers a handshake the panel holds on purpose.

        jupyter-server answers a kernel-channel upgrade only once the kernel
        behind it has replied to a kernel-info request, and it allows that reply
        its ``MappingKernelManager.kernel_info_timeout``. A kernel that joins a
        bound session imports the framework before it answers anything, so on a
        loaded host that reply lands after ``websockets``' default ten-second
        opening timeout — and the proxy, not the panel, hung up: the browser got
        a normal close before its first frame, with the kernel alive and well
        behind it. The proxy's budget for an upstream handshake must therefore
        be no shorter than the longest one a panel is allowed to hold.
        """
        from jupyter_server.services.kernels.kernelmanager import MappingKernelManager

        _app, client = panels_app

        fake = self._connect(client, "/panel/trusted/ws/stream")

        assert fake.kwargs["open_timeout"] == UPSTREAM_OPEN_TIMEOUT
        assert UPSTREAM_OPEN_TIMEOUT >= MappingKernelManager.kernel_info_timeout.default_value

    def test_offbox_panel_ws_carries_no_headers(self, panels_app):
        _app, client = panels_app

        fake = self._connect(client, "/panel/facility/ws/stream")

        assert fake.kwargs["additional_headers"] is None
        assert getattr(fake, "process_redirect", None) is None

    def test_real_websockets_exposes_process_redirect(self):
        """The redirect refusal above is only real if the library still has the hook.

        ``_FakeConnect`` accepts any attribute, so the guard the proxy installs
        would look identical against a ``websockets`` too old to have
        ``process_redirect`` (added in 14.0) — the handshake would quietly follow
        redirects with the secret attached. Asserted against the real object, and
        backed by the ``websockets>=14`` floor declared in ``pyproject.toml``.
        """
        import websockets

        assert callable(getattr(websockets.connect, "process_redirect", None))


class TestRedirectPrefixIdempotence:
    """A prefix-AWARE backend's Location is never double-prefixed.

    Backends that honor the ``X-Forwarded-Prefix`` this proxy sends emit
    Locations already inside the panel namespace; re-basing one of those
    again yields ``/u/a/panel/x/u/a/panel/x/...`` and a terminal 404.
    Prefix-naive backends keep the old re-base.
    """

    PREFIX = "/u/alice/panel/dataview"
    BACKEND = "http://192.168.6.14:8200"

    def _rewrite(self, location: str) -> str | None:
        from osprey.interfaces.web_terminal.routes.proxy import _rewrite_redirect_location

        return _rewrite_redirect_location(
            location,
            self.BACKEND,
            self.PREFIX,
            base_path=self.PREFIX + "/",
            terminal_origin="http://127.0.0.1:10000",
        )

    def test_naive_root_absolute_location_still_gains_the_prefix(self):
        assert self._rewrite("/day/2026-08-31") == self.PREFIX + "/day/2026-08-31"

    def test_prefix_aware_root_absolute_location_is_relayed_unchanged(self):
        aware = self.PREFIX + "/day/2026-08-31"
        assert self._rewrite(aware) == aware

    def test_prefix_aware_location_with_query_is_relayed_unchanged(self):
        aware = self.PREFIX + "?experiment=Undulator"
        assert self._rewrite(aware) == aware

    def test_bare_prefix_location_is_relayed_unchanged(self):
        assert self._rewrite(self.PREFIX) == self.PREFIX

    def test_a_panel_id_sharing_a_prefix_is_not_conflated(self):
        # `/u/alice/panel/dataview-2/...` is NOT inside `/u/alice/panel/dataview`.
        other = self.PREFIX + "-2/day/2026-08-31"
        assert self._rewrite(other) == self.PREFIX + other

    def test_prefix_aware_absolute_backend_location_is_rebasded_without_doubling(self):
        aware = self.BACKEND + self.PREFIX + "/day/2026-08-31"
        assert self._rewrite(aware) == self.PREFIX + "/day/2026-08-31"
