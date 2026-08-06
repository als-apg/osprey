"""Tests for the auth sidecar's OIDC login and callback routes.

Authlib is replaced at the route boundary — ``app.state.oidc_client`` — so these
tests pin *this* module's decisions rather than the library's: which roster user
a callback can unlock, what happens when the asserted identity maps to nobody or
to somebody else, and what the re-issued session cookie carries. The handshake
against a real (mock) IdP is a separate test.

The load-bearing assertion, repeated in several shapes: a callback never unlocks
a user other than the one whose card was clicked, whatever identity the IdP
asserts.
"""

from __future__ import annotations

import json
import logging
from base64 import b64encode
from typing import Any

import httpx
import itsdangerous
import pytest
from authlib.integrations.starlette_client import OAuthError
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.services.auth_sidecar.app import STATE_COOKIE_NAME, create_app
from osprey.services.auth_sidecar.return_to import MAX_RETURN_TO_LENGTH
from osprey.services.auth_sidecar.routes.oidc import (
    CALLBACK_PATH,
    LOGIN_PATH,
    PENDING_FLOW_SESSION_KEY,
)
from osprey.services.auth_sidecar.sessions import SESSION_COOKIE_NAME, SessionCodec, SessionState

SESSION_SECRET = "session-secret-value"
STATE_SECRET = "state-secret-value"
EXTERNAL_ORIGIN = "https://terminals.example.org"
SESSION_LIFETIME = 3600

ALICE_SUBJECT = "idp|alice"
CAROL_SUBJECT = "idp|carol"
FLOW_STATE = "handshake-state-value"
IDP_AUTHORIZE_URL = "https://idp.example.org/authorize?client_id=client-id&state=" + FLOW_STATE

# alice and carol are mapped; bob is on the roster with no mapped identity, so
# he is the "unmapped" case every refusal test needs.
OIDC_ENV = {
    "OSPREY_AUTH_METHOD": "oidc",
    "OSPREY_AUTH_SESSION_SECRET": SESSION_SECRET,
    "OSPREY_AUTH_STATE_SECRET": STATE_SECRET,
    "OSPREY_AUTH_SESSION_LIFETIME": str(SESSION_LIFETIME),
    "OSPREY_AUTH_USERS": "alice,bob,carol",
    "OSPREY_AUTH_OIDC_ISSUER": "https://idp.example.org",
    "OSPREY_AUTH_OIDC_CLIENT_ID": "client-id",
    "OSPREY_AUTH_OIDC_CLIENT_SECRET": "client-secret-value",
    "OSPREY_AUTH_OIDC_SUBJECT_ALICE": ALICE_SUBJECT,
    "OSPREY_AUTH_OIDC_SUBJECT_CAROL": CAROL_SUBJECT,
    "OSPREY_AUTH_EXTERNAL_ORIGIN": EXTERNAL_ORIGIN,
    "OSPREY_AUTH_TLS_ENABLED": "true",
}

PASSWORD_ENV = {
    "OSPREY_AUTH_METHOD": "password",
    "OSPREY_AUTH_SESSION_SECRET": SESSION_SECRET,
    "OSPREY_AUTH_USERS": "alice",
    "OSPREY_AUTH_EXTERNAL_ORIGIN": EXTERNAL_ORIGIN,
}


class FakeOIDCClient:
    """Stands in for Authlib's Starlette client at the route boundary.

    Records what the login route hands it (the ``redirect_uri`` above all, which
    the IdP registration has to match) and returns whatever the test wants the
    callback to see.
    """

    def __init__(
        self,
        *,
        userinfo: dict[str, Any] | None = None,
        token: dict[str, Any] | None = None,
        authorize_error: BaseException | None = None,
        callback_error: BaseException | None = None,
        state: str = FLOW_STATE,
    ) -> None:
        self.state = state
        self.token = token if token is not None else {"userinfo": userinfo or {}}
        self.authorize_error = authorize_error
        self.callback_error = callback_error
        self.redirect_uri: str | None = None
        self.saved: dict[str, Any] | None = None

    async def create_authorization_url(self, redirect_uri: str | None = None) -> dict[str, Any]:
        if self.authorize_error is not None:
            raise self.authorize_error
        self.redirect_uri = redirect_uri
        return {"url": IDP_AUTHORIZE_URL, "state": self.state, "nonce": "nonce-value"}

    async def save_authorize_data(self, request: Any, **kwargs: Any) -> None:
        self.saved = kwargs

    async def authorize_access_token(self, request: Any, **kwargs: Any) -> dict[str, Any]:
        # Authlib's own signature takes keyword arguments the callback supplies
        # (claims_options above all, which is what makes it check the audience).
        # Accepting and recording them keeps this stand-in callable wherever the
        # real client is; refusing them would fail every callback with a
        # TypeError the broad except arm reports as an unvalidatable token.
        self.token_kwargs = kwargs
        if self.callback_error is not None:
            raise self.callback_error
        return self.token


def _app(env: dict[str, str] | None = None, client: FakeOIDCClient | None = None) -> FastAPI:
    """An app in OIDC mode with Authlib replaced."""
    app = create_app(env if env is not None else OIDC_ENV)
    app.state.oidc_client = client if client is not None else FakeOIDCClient()
    return app


def _client(app: FastAPI, *, tls: bool = True) -> TestClient:
    """A test client whose origin matches the deployment's.

    Not cosmetic: the state cookie the factory pins carries ``Secure`` whenever
    the deployment is on TLS, and a client on an ``http://`` origin drops it
    silently — every handshake would then fail as "no login in flight", for a
    reason that has nothing to do with this module.
    """
    return TestClient(app, base_url="https://testserver" if tls else "http://testserver")


def _osprey_log(caplog: pytest.LogCaptureFixture) -> str:
    """Only this service's log records.

    The test client's own HTTP logger echoes full request URLs, so asserting
    against the whole capture would test httpx rather than what this module
    writes.
    """
    return "\n".join(
        record.getMessage() for record in caplog.records if record.name.startswith("osprey.")
    )


def _state_cookie(data: dict[str, Any]) -> str:
    """Forge a Starlette session cookie the way ``SessionMiddleware`` signs one.

    Lets a callback be exercised without first driving the login route, which is
    the only way to test a pending flow the login route would never create (an
    unmapped user, a stale state).
    """
    signer = itsdangerous.TimestampSigner(STATE_SECRET)
    return signer.sign(b64encode(json.dumps(data).encode("utf-8"))).decode("utf-8")


def _pending(user: str, *, state: str = FLOW_STATE, next_target: str | None = None) -> str:
    """A state cookie carrying one in-flight handshake for ``user``.

    One caveat for anything built on this helper: a cookie planted here lands in
    the test jar at path ``/``, while the response's own ``Set-Cookie`` carries
    the pinned ``path=/auth``, so the jar keeps the planted value instead of
    replacing it. The route still consumes the flow — the *server* pops it — but
    a second request from the same client presents the forged cookie again.
    Assert on consumption only through the real login route (see
    ``test_callback_does_not_reuse_a_completed_flow``), never through this one.
    """
    return _state_cookie(
        {
            PENDING_FLOW_SESSION_KEY: {
                "state": state,
                "user": user,
                "next": next_target or f"/u/{user}/",
            }
        }
    )


def _session_from(response: httpx.Response) -> SessionState:
    """Decode the auth session cookie the response set."""
    codec = SessionCodec(SESSION_SECRET, max_age=SESSION_LIFETIME)
    return codec.decode(response.cookies[SESSION_COOKIE_NAME])


def _set_cookie_header(response: httpx.Response, name: str) -> str:
    """The raw ``Set-Cookie`` header for ``name``; attributes are asserted on it."""
    for header in response.headers.get_list("set-cookie"):
        if header.startswith(f"{name}="):
            return header
    raise AssertionError(f"no Set-Cookie for {name!r} in {response.headers.get_list('set-cookie')}")


# --- the routes exist and belong to this mode --------------------------------


def test_factory_includes_the_oidc_routes() -> None:
    """The factory picks the module up with no edit of its own."""
    paths = create_app(OIDC_ENV).openapi()["paths"]
    assert LOGIN_PATH in paths
    assert CALLBACK_PATH in paths


@pytest.mark.parametrize("path", [LOGIN_PATH, CALLBACK_PATH])
def test_oidc_routes_are_absent_in_password_mode(path: str) -> None:
    """A password deployment has no OIDC surface, even though the module loads."""
    with _client(_app(PASSWORD_ENV)) as client:
        response = client.get(path, params={"user": "alice"}, follow_redirects=False)
    assert response.status_code == 404


# --- login -------------------------------------------------------------------


def test_login_redirects_to_the_identity_provider() -> None:
    """The clicked card starts a handshake whose redirect_uri is the deployment's."""
    fake = FakeOIDCClient()
    with _client(_app(client=fake)) as client:
        response = client.get(LOGIN_PATH, params={"user": "alice"}, follow_redirects=False)

    assert response.status_code == 302
    assert response.headers["location"] == IDP_AUTHORIZE_URL
    assert fake.redirect_uri == f"{EXTERNAL_ORIGIN}{CALLBACK_PATH}"
    assert fake.saved is not None
    assert fake.saved["redirect_uri"] == f"{EXTERNAL_ORIGIN}{CALLBACK_PATH}"
    # The clicked user rides in the signed state cookie, never on the wire.
    assert STATE_COOKIE_NAME in response.cookies
    assert "alice" not in response.headers["location"]


def test_login_refuses_a_user_with_no_mapped_identity(caplog: pytest.LogCaptureFixture) -> None:
    """A handshake that could only end in a refusal never starts."""
    with caplog.at_level(logging.WARNING), _client(_app()) as client:
        response = client.get(LOGIN_PATH, params={"user": "bob"}, follow_redirects=False)

    assert response.status_code == 403
    assert "bob" in caplog.text


def test_login_refuses_a_user_who_is_not_on_the_roster() -> None:
    with _client(_app()) as client:
        response = client.get(LOGIN_PATH, params={"user": "mallory"}, follow_redirects=False)
    assert response.status_code == 404


def test_login_reports_an_unreachable_issuer_as_502() -> None:
    """Discovery failure is the IdP's fault, not a crash in the sidecar."""
    fake = FakeOIDCClient(authorize_error=httpx.ConnectError("no route to issuer"))
    with _client(_app(client=fake)) as client:
        response = client.get(LOGIN_PATH, params={"user": "alice"}, follow_redirects=False)
    assert response.status_code == 502


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError('Missing "authorize_url" value'),
        ValueError("Expecting value: line 1 column 1 (char 0)"),
    ],
    ids=["no-authorization-endpoint", "not-json"],
)
def test_login_reports_an_unusable_discovery_document_as_502(failure: Exception) -> None:
    """A wrong issuer or a captive portal serves a document that parses badly
    rather than failing to arrive; that is still the IdP's problem, not a 500."""
    with _client(_app(client=FakeOIDCClient(authorize_error=failure))) as client:
        response = client.get(LOGIN_PATH, params={"user": "alice"}, follow_redirects=False)
    assert response.status_code == 502


# --- return-to validation ----------------------------------------------------


@pytest.mark.parametrize(
    "target",
    [
        "https://evil.example.org/",
        "//evil.example.org/",
        "/\\evil.example.org/",
        "/u/alice/\\..\\..",
        "u/alice/",
        "/u/alice/\r\nSet-Cookie: x=y",
    ],
    ids=["absolute", "protocol-relative", "backslash-authority", "backslash", "relative", "crlf"],
)
def test_login_discards_an_off_origin_return_to(target: str) -> None:
    """A tampered return-to sends the operator to their own terminal instead."""
    with _client(_app(client=FakeOIDCClient(userinfo={"sub": ALICE_SUBJECT}))) as client:
        client.get(LOGIN_PATH, params={"user": "alice", "next": target}, follow_redirects=False)
        response = client.get(
            CALLBACK_PATH, params={"code": "auth-code", "state": FLOW_STATE}, follow_redirects=False
        )

    assert response.status_code == 303
    assert response.headers["location"] == "/u/alice/"


@pytest.mark.parametrize(
    "target",
    [
        "/u/alice/artifacts%2f..%2f..%2fu%2fbob%2f",
        "/u/alice/" + "a" * MAX_RETURN_TO_LENGTH,
    ],
    ids=["encoded-separator", "over-length"],
)
def test_login_discards_a_return_to_only_the_shared_rule_rejects(target: str) -> None:
    """Both flows hold a return-to to one rule.

    An encoded separator (which this service, nginx and the browser each resolve
    differently) and a value past the cap (which becomes a ``Location`` header
    the proxy answers with a 502) are rejected on the OIDC leg exactly as they
    are on the password one.
    """
    with _client(_app(client=FakeOIDCClient(userinfo={"sub": ALICE_SUBJECT}))) as client:
        client.get(LOGIN_PATH, params={"user": "alice", "next": target}, follow_redirects=False)
        response = client.get(
            CALLBACK_PATH, params={"code": "auth-code", "state": FLOW_STATE}, follow_redirects=False
        )

    assert response.status_code == 303
    assert response.headers["location"] == "/u/alice/"


def test_login_keeps_a_same_origin_return_to() -> None:
    with _client(_app(client=FakeOIDCClient(userinfo={"sub": ALICE_SUBJECT}))) as client:
        client.get(
            LOGIN_PATH,
            params={"user": "alice", "next": "/u/alice/tuning?panel=orbit"},
            follow_redirects=False,
        )
        response = client.get(
            CALLBACK_PATH, params={"code": "auth-code", "state": FLOW_STATE}, follow_redirects=False
        )

    assert response.status_code == 303
    assert response.headers["location"] == "/u/alice/tuning?panel=orbit"


def test_callback_survives_a_return_to_that_is_not_a_string() -> None:
    """The stored return-to is read back out of a cookie payload, so its shape
    is whatever was written, not whatever was declared."""
    app = _app(client=FakeOIDCClient(userinfo={"sub": ALICE_SUBJECT}))
    with _client(app) as client:
        client.cookies.set(
            STATE_COOKIE_NAME,
            _state_cookie(
                {PENDING_FLOW_SESSION_KEY: {"state": FLOW_STATE, "user": "alice", "next": 17}}
            ),
        )
        response = client.get(
            CALLBACK_PATH, params={"code": "auth-code", "state": FLOW_STATE}, follow_redirects=False
        )

    assert response.status_code == 303
    assert response.headers["location"] == "/u/alice/"


# --- callback: the identity must be the clicked user's -----------------------


def test_callback_unlocks_the_clicked_user() -> None:
    """The whole point: a mapped identity unlocks that user and nobody else."""
    with _client(_app(client=FakeOIDCClient(userinfo={"sub": ALICE_SUBJECT}))) as client:
        client.get(LOGIN_PATH, params={"user": "alice"}, follow_redirects=False)
        response = client.get(
            CALLBACK_PATH, params={"code": "auth-code", "state": FLOW_STATE}, follow_redirects=False
        )

    assert response.status_code == 303
    assert response.headers["location"] == "/u/alice/"
    session = _session_from(response)
    assert session.unlocked_usernames(now=0.0) == ("alice",)
    entry = session.entry("alice")
    assert entry is not None
    # No generation tag: that is the password mode's rotation signal, and an
    # empty one is what verify refuses to authorise a password session on.
    assert entry.generation_tag == ""


def test_callback_expiry_comes_from_the_configured_lifetime() -> None:
    codec = SessionCodec(SESSION_SECRET, max_age=SESSION_LIFETIME)
    with _client(_app(client=FakeOIDCClient(userinfo={"sub": ALICE_SUBJECT}))) as client:
        client.get(LOGIN_PATH, params={"user": "alice"}, follow_redirects=False)
        response = client.get(
            CALLBACK_PATH, params={"code": "auth-code", "state": FLOW_STATE}, follow_redirects=False
        )

    entry = _session_from(response).entry("alice")
    assert entry is not None
    assert entry.expires_at == pytest.approx(codec.now() + SESSION_LIFETIME, abs=5)


def test_callback_cookie_carries_the_pinned_attributes() -> None:
    """HttpOnly always, SameSite=Lax, Secure under TLS, and path ``/`` so the
    cookie reaches the ``/u/<user>/`` locations it authorises."""
    with _client(_app(client=FakeOIDCClient(userinfo={"sub": ALICE_SUBJECT}))) as client:
        client.get(LOGIN_PATH, params={"user": "alice"}, follow_redirects=False)
        response = client.get(
            CALLBACK_PATH, params={"code": "auth-code", "state": FLOW_STATE}, follow_redirects=False
        )

    header = _set_cookie_header(response, SESSION_COOKIE_NAME).lower()
    assert "httponly" in header
    assert "samesite=lax" in header
    assert "secure" in header
    assert "path=/;" in header or header.rstrip().endswith("path=/")


def test_callback_does_not_inherit_a_revoked_session_id() -> None:
    """A logout/login race must not lock the browser out of what it just unlocked.

    Logout revokes by session id and verify refuses every cookie carrying a
    revoked one, so a callback that kept the id would re-sign a cookie verify
    then rejects — and the next handshake would inherit it again.
    """
    app = _app(client=FakeOIDCClient(userinfo={"sub": ALICE_SUBJECT}))
    codec = SessionCodec(SESSION_SECRET, max_age=SESSION_LIFETIME)
    retired = SessionState.new(codec.now())
    app.state.revocation_store.revoke(retired.session_id, codec.now() + SESSION_LIFETIME)

    with _client(app) as client:
        client.cookies.set(SESSION_COOKIE_NAME, codec.encode(retired))
        client.get(LOGIN_PATH, params={"user": "alice"}, follow_redirects=False)
        response = client.get(
            CALLBACK_PATH, params={"code": "auth-code", "state": FLOW_STATE}, follow_redirects=False
        )

        assert response.status_code == 303
        assert _session_from(response).session_id != retired.session_id
        # The proof that matters: the terminal this handshake was for opens.
        assert client.get("/verify", params={"user": "alice"}).status_code == 200


def test_callback_without_tls_issues_no_secure_cookie() -> None:
    env = {
        **OIDC_ENV,
        "OSPREY_AUTH_TLS_ENABLED": "false",
        "OSPREY_AUTH_EXTERNAL_ORIGIN": "http://x",
    }
    app = _app(env, FakeOIDCClient(userinfo={"sub": ALICE_SUBJECT}))
    with _client(app, tls=False) as client:
        client.get(LOGIN_PATH, params={"user": "alice"}, follow_redirects=False)
        response = client.get(
            CALLBACK_PATH, params={"code": "auth-code", "state": FLOW_STATE}, follow_redirects=False
        )

    assert "secure" not in _set_cookie_header(response, SESSION_COOKIE_NAME).lower()


def test_callback_honours_the_configured_claim() -> None:
    """``oidc_claim`` selects which claim carries the identity."""
    env = {**OIDC_ENV, "OSPREY_AUTH_OIDC_CLAIM": "email"}
    userinfo = {"sub": "some-opaque-id", "email": ALICE_SUBJECT}
    with _client(_app(env, FakeOIDCClient(userinfo=userinfo))) as client:
        client.get(LOGIN_PATH, params={"user": "alice"}, follow_redirects=False)
        response = client.get(
            CALLBACK_PATH, params={"code": "auth-code", "state": FLOW_STATE}, follow_redirects=False
        )

    assert response.status_code == 303
    assert _session_from(response).unlocked_usernames(now=0.0) == ("alice",)


def test_callback_refuses_an_identity_mapped_to_another_user(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """carol's identity does not open alice's terminal — and does not open
    carol's either. The clicked card is the only user a login can unlock."""
    with (
        caplog.at_level(logging.WARNING),
        _client(_app(client=FakeOIDCClient(userinfo={"sub": CAROL_SUBJECT}))) as client,
    ):
        client.get(LOGIN_PATH, params={"user": "alice"}, follow_redirects=False)
        response = client.get(
            CALLBACK_PATH, params={"code": "auth-code", "state": FLOW_STATE}, follow_redirects=False
        )

    assert response.status_code == 403
    assert SESSION_COOKIE_NAME not in response.cookies
    assert "carol" not in caplog.text


def test_callback_refuses_an_unmapped_identity(caplog: pytest.LogCaptureFixture) -> None:
    """The acceptance gate: an IdP identity mapped to no roster user is refused.

    Driven through a forged pending flow for bob, who the login route would
    never have sent to the IdP — the callback must refuse on its own, not
    because something upstream did.
    """
    app = _app(client=FakeOIDCClient(userinfo={"sub": "idp|stranger"}))
    with caplog.at_level(logging.WARNING), _client(app) as client:
        client.cookies.set(STATE_COOKIE_NAME, _pending("bob"))
        response = client.get(
            CALLBACK_PATH, params={"code": "auth-code", "state": FLOW_STATE}, follow_redirects=False
        )

    assert response.status_code == 403
    assert SESSION_COOKIE_NAME not in response.cookies
    assert "bob" in caplog.text


def test_callback_ignores_a_user_query_parameter() -> None:
    """The clicked card is the username, and the callback's URL is not a vote.

    The pending flow says alice; the query string says carol, whose identity the
    IdP is asserting. If the query parameter were read, this would succeed as
    carol — so the refusal is the whole binding, stated in one request.
    """
    app = _app(client=FakeOIDCClient(userinfo={"sub": CAROL_SUBJECT}))
    with _client(app) as client:
        client.cookies.set(STATE_COOKIE_NAME, _pending("alice"))
        response = client.get(
            CALLBACK_PATH,
            params={"code": "auth-code", "state": FLOW_STATE, "user": "carol"},
            follow_redirects=False,
        )

    assert response.status_code == 403
    assert SESSION_COOKIE_NAME not in response.cookies


def test_callback_compares_a_non_ascii_identity_without_raising() -> None:
    """A mapped identity carrying a diacritic must be able to log in.

    Compared as ``str``, ``secrets.compare_digest`` raises on exactly this
    input, which would turn a login that should succeed into a 500 and lock the
    operator out permanently.
    """
    env = {**OIDC_ENV, "OSPREY_AUTH_OIDC_SUBJECT_ALICE": "jörg@example.org"}
    app = _app(env, FakeOIDCClient(userinfo={"sub": "jörg@example.org"}))
    with _client(app) as client:
        client.cookies.set(STATE_COOKIE_NAME, _pending("alice"))
        response = client.get(
            CALLBACK_PATH, params={"code": "auth-code", "state": FLOW_STATE}, follow_redirects=False
        )

    assert response.status_code == 303
    assert _session_from(response).unlocked_usernames(now=0.0) == ("alice",)


def test_callback_refuses_a_differing_non_ascii_identity() -> None:
    """The same comparison still denies when the values differ."""
    env = {**OIDC_ENV, "OSPREY_AUTH_OIDC_SUBJECT_ALICE": "jörg@example.org"}
    app = _app(env, FakeOIDCClient(userinfo={"sub": "jorg@example.org"}))
    with _client(app) as client:
        client.cookies.set(STATE_COOKIE_NAME, _pending("alice"))
        response = client.get(
            CALLBACK_PATH, params={"code": "auth-code", "state": FLOW_STATE}, follow_redirects=False
        )

    assert response.status_code == 403
    assert SESSION_COOKIE_NAME not in response.cookies


def test_callback_refuses_a_non_ascii_state_without_raising() -> None:
    """``state`` is unauthenticated input; an accent in it is a denial, not a 500."""
    app = _app(client=FakeOIDCClient(userinfo={"sub": ALICE_SUBJECT}))
    with _client(app) as client:
        client.cookies.set(STATE_COOKIE_NAME, _pending("alice"))
        response = client.get(
            CALLBACK_PATH, params={"code": "auth-code", "state": "ü"}, follow_redirects=False
        )

    assert response.status_code == 400
    assert SESSION_COOKIE_NAME not in response.cookies


def test_callback_refuses_when_the_claim_is_absent() -> None:
    """A token carrying no usable claim asserts no identity, so it authorises none."""
    app = _app(client=FakeOIDCClient(userinfo={"email": ALICE_SUBJECT}))
    with _client(app) as client:
        client.cookies.set(STATE_COOKIE_NAME, _pending("alice"))
        response = client.get(
            CALLBACK_PATH, params={"code": "auth-code", "state": FLOW_STATE}, follow_redirects=False
        )

    assert response.status_code == 403
    assert SESSION_COOKIE_NAME not in response.cookies


def test_callback_refuses_a_non_string_claim() -> None:
    """A claim of the wrong shape is not compared against, it is refused."""
    app = _app(client=FakeOIDCClient(userinfo={"sub": ["idp|alice"]}))
    with _client(app) as client:
        client.cookies.set(STATE_COOKIE_NAME, _pending("alice"))
        response = client.get(
            CALLBACK_PATH, params={"code": "auth-code", "state": FLOW_STATE}, follow_redirects=False
        )

    assert response.status_code == 403


def test_callback_refuses_a_token_with_no_userinfo() -> None:
    app = _app(client=FakeOIDCClient(token={"access_token": "opaque"}))
    with _client(app) as client:
        client.cookies.set(STATE_COOKIE_NAME, _pending("alice"))
        response = client.get(
            CALLBACK_PATH, params={"code": "auth-code", "state": FLOW_STATE}, follow_redirects=False
        )

    assert response.status_code == 403


# --- callback: handshake integrity -------------------------------------------


def test_callback_without_a_pending_flow_is_refused() -> None:
    """A callback nobody started is not a login."""
    with _client(_app()) as client:
        response = client.get(
            CALLBACK_PATH, params={"code": "auth-code", "state": FLOW_STATE}, follow_redirects=False
        )

    assert response.status_code == 400
    assert SESSION_COOKIE_NAME not in response.cookies


def test_callback_refuses_a_state_the_browser_did_not_start() -> None:
    """The returned state must be the one this browser's pending flow carries."""
    app = _app(client=FakeOIDCClient(userinfo={"sub": ALICE_SUBJECT}))
    with _client(app) as client:
        client.cookies.set(STATE_COOKIE_NAME, _pending("alice", state="a-different-state"))
        response = client.get(
            CALLBACK_PATH, params={"code": "auth-code", "state": FLOW_STATE}, follow_redirects=False
        )

    assert response.status_code == 400
    assert SESSION_COOKIE_NAME not in response.cookies


def test_callback_does_not_reuse_a_completed_flow() -> None:
    """The pending flow is consumed, so a replayed callback finds nothing."""
    with _client(_app(client=FakeOIDCClient(userinfo={"sub": ALICE_SUBJECT}))) as client:
        client.get(LOGIN_PATH, params={"user": "alice"}, follow_redirects=False)
        first = client.get(
            CALLBACK_PATH, params={"code": "auth-code", "state": FLOW_STATE}, follow_redirects=False
        )
        replay = client.get(
            CALLBACK_PATH, params={"code": "auth-code", "state": FLOW_STATE}, follow_redirects=False
        )

    assert first.status_code == 303
    assert replay.status_code == 400


def test_authorization_failure_is_a_4xx_not_a_crash() -> None:
    """Authlib's own state/IdP error path surfaces as a refusal."""
    app = _app(client=FakeOIDCClient(callback_error=OAuthError(error="access_denied")))
    with _client(app) as client:
        client.cookies.set(STATE_COOKIE_NAME, _pending("alice"))
        response = client.get(
            CALLBACK_PATH, params={"code": "auth-code", "state": FLOW_STATE}, follow_redirects=False
        )

    assert response.status_code == 400
    assert SESSION_COOKIE_NAME not in response.cookies


def test_token_validation_failure_is_reported_as_a_bad_gateway() -> None:
    """An ID token that does not validate is the IdP's problem, not a 500."""
    app = _app(client=FakeOIDCClient(callback_error=ValueError("bad signature")))
    with _client(app) as client:
        client.cookies.set(STATE_COOKIE_NAME, _pending("alice"))
        response = client.get(
            CALLBACK_PATH, params={"code": "auth-code", "state": FLOW_STATE}, follow_redirects=False
        )

    assert response.status_code == 502
    assert SESSION_COOKIE_NAME not in response.cookies


def test_unreachable_token_endpoint_is_reported_as_a_bad_gateway() -> None:
    app = _app(client=FakeOIDCClient(callback_error=httpx.ConnectError("token endpoint down")))
    with _client(app) as client:
        client.cookies.set(STATE_COOKIE_NAME, _pending("alice"))
        response = client.get(
            CALLBACK_PATH, params={"code": "auth-code", "state": FLOW_STATE}, follow_redirects=False
        )

    assert response.status_code == 502


# --- the session the callback re-issues --------------------------------------


def test_callback_keeps_other_unlocked_users() -> None:
    """A shared control-room browser: alice logging in leaves bob unlocked."""
    codec = SessionCodec(SESSION_SECRET, max_age=SESSION_LIFETIME)
    existing = codec.new_state().with_user(
        "bob", expires_at=codec.now() + SESSION_LIFETIME, generation_tag="bobtag"
    )
    app = _app(client=FakeOIDCClient(userinfo={"sub": ALICE_SUBJECT}))
    with _client(app) as client:
        client.cookies.set(SESSION_COOKIE_NAME, codec.encode(existing))
        client.get(LOGIN_PATH, params={"user": "alice"}, follow_redirects=False)
        response = client.get(
            CALLBACK_PATH, params={"code": "auth-code", "state": FLOW_STATE}, follow_redirects=False
        )

    session = _session_from(response)
    assert set(session.unlocked_usernames(now=0.0)) == {"alice", "bob"}
    bob = session.entry("bob")
    assert bob is not None
    assert bob.generation_tag == "bobtag"
    assert session.session_id == existing.session_id


def test_callback_treats_an_unreadable_cookie_as_a_fresh_session() -> None:
    """A tampered session cookie carries no authorisation, so it is replaced."""
    app = _app(client=FakeOIDCClient(userinfo={"sub": ALICE_SUBJECT}))
    with _client(app) as client:
        client.cookies.set(SESSION_COOKIE_NAME, "not-a-signed-cookie")
        client.get(LOGIN_PATH, params={"user": "alice"}, follow_redirects=False)
        response = client.get(
            CALLBACK_PATH, params={"code": "auth-code", "state": FLOW_STATE}, follow_redirects=False
        )

    assert response.status_code == 303
    assert _session_from(response).unlocked_usernames(now=0.0) == ("alice",)


# --- nothing secret leaves the process ---------------------------------------


@pytest.mark.parametrize(
    "client_factory",
    [
        lambda: FakeOIDCClient(userinfo={"sub": CAROL_SUBJECT}),
        lambda: FakeOIDCClient(callback_error=OAuthError(error="access_denied")),
    ],
    ids=["mismatch", "authorization-error"],
)
def test_failures_never_echo_credentials_or_tokens(
    client_factory: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """Refusals name a category; they never carry the client secret, the code,
    the token, or the asserted identity."""
    app = _app(client=client_factory())
    with caplog.at_level(logging.DEBUG), _client(app) as client:
        client.cookies.set(STATE_COOKIE_NAME, _pending("alice"))
        response = client.get(
            CALLBACK_PATH,
            params={"code": "secret-auth-code", "state": FLOW_STATE},
            follow_redirects=False,
        )

    leaked = ("client-secret-value", "secret-auth-code", CAROL_SUBJECT, ALICE_SUBJECT)
    for value in leaked:
        assert value not in response.text
        assert value not in _osprey_log(caplog)
