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

from osprey.services.auth_sidecar import audit
from osprey.services.auth_sidecar.app import STATE_COOKIE_NAME, create_app
from osprey.services.auth_sidecar.return_to import MAX_RETURN_TO_LENGTH
from osprey.services.auth_sidecar.routes.oidc import (
    CALLBACK_PATH,
    LOGIN_PATH,
    PENDING_FLOW_SESSION_KEY,
    REASON_AMBIGUOUS_IDENTITY,
    REASON_HOSTED_DOMAIN_MISMATCH,
    REASON_IDENTITY_MISMATCH,
    REASON_NO_COVERING_PRINCIPAL,
    REASON_UNMAPPED_USER,
    REASON_UNSAFE_ASSERTED_IDENTITY,
    REASON_UNVERIFIED_EMAIL,
    token_admissible,
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

# The rule-based cards are exercised against a deployment mapping identities on
# `email`, because that is the only claim under which a `domain:` principal
# means anything (a `sub` names no domain). dana is on nobody's roster: an
# identity a rule admits need not be a roster entry at all, which is the whole
# point of naming one.
DANA_EMAIL = "dana@example.org"
DANA_DOMAIN = "example.org"
# bob's own mapped identity, outside the domain his card is usually shared
# with, so `self` is the only principal that can admit him.
BOB_EMAIL = "bob@lbl.gov"

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
        # The default carries an `id_token` because the real client only fills
        # `userinfo` when the token response had one, and the callback now
        # requires it. An explicit `token=` is passed through untouched, which
        # is how a test probes a token response of its own shape.
        self.token = (
            token
            if token is not None
            else {"id_token": "header.payload.signature", "userinfo": userinfo or {}}
        )
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


def _rule_env(access: str, **overrides: str) -> dict[str, str]:
    """The OIDC roster with bob's card carrying a rendered access rule.

    Mapping on ``email`` throughout, so alice's and carol's ``idp|`` subjects
    match nothing a rule test asserts and the reverse match cannot quietly
    supply an answer a principal was supposed to give.
    """
    return {
        **OIDC_ENV,
        "OSPREY_AUTH_OIDC_CLAIM": "email",
        "OSPREY_AUTH_ROSTER_ACCESS_BOB": access,
        **overrides,
    }


@pytest.fixture
def refusals(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Every login refusal the route files, exactly as the ledger receives it.

    The ledger itself needs an audit zone to write into; what these tests are
    pinning is which category a refusal is filed under and what reaches the
    record beside it, so the writer is replaced and the envelope is not built.
    """
    recorded: list[dict[str, Any]] = []

    def _record(**fields: Any) -> None:
        recorded.append(fields)

    monkeypatch.setattr(audit, "record_login_refusal", _record)
    return recorded


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


def test_login_starts_a_shared_cards_handshake_when_any_entry_is_mapped() -> None:
    """bob's shared card has no mapped identity of its own — but alice and
    carol do, and either of them could open it, so the handshake starts. The
    unmapped-user refusal above is the own-card rule; a shared card only asks
    whether SOMEBODY on the roster is mapped."""
    env = {**OIDC_ENV, "OSPREY_AUTH_ROSTER_ACCESS_BOB": "any"}
    with _client(_app(env)) as client:
        response = client.get(LOGIN_PATH, params={"user": "bob"}, follow_redirects=False)

    assert response.status_code == 302
    assert response.headers["location"] == IDP_AUTHORIZE_URL


def test_login_refuses_a_shared_card_when_no_entry_is_mapped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A roster on which nobody carries a mapped identity cannot open a shared
    card either: the handshake could only end in a refusal, so it never
    starts — same category as the own-card rule, its own message."""
    env = {
        key: value
        for key, value in OIDC_ENV.items()
        if not key.startswith("OSPREY_AUTH_OIDC_SUBJECT_")
    }
    env["OSPREY_AUTH_ROSTER_ACCESS_BOB"] = "any"
    with caplog.at_level(logging.WARNING), _client(_app(env)) as client:
        response = client.get(LOGIN_PATH, params={"user": "bob"}, follow_redirects=False)

    assert response.status_code == 403
    assert "no roster entry carries a mapped identity" in caplog.text


@pytest.mark.parametrize(
    "access",
    ['["domain:example.org"]', '["user:dana@example.org"]', '["self","domain:example.org"]'],
)
def test_login_starts_a_handshake_for_a_card_that_names_an_identity(access: str) -> None:
    """A card naming an identity or a domain needs no roster mapping at all.

    The roster here maps nobody, which is precisely the case the older gate
    refused: it asked whether SOMEBODY was mapped, because the roster was the
    only thing that could open a shared card. A `user:`/`domain:` principal is
    answered by the token instead, so the handshake has somewhere to go and
    the gate must let it start.
    """
    env = {
        key: value
        for key, value in _rule_env(access).items()
        if not key.startswith("OSPREY_AUTH_OIDC_SUBJECT_")
    }
    with _client(_app(env)) as client:
        response = client.get(LOGIN_PATH, params={"user": "bob"}, follow_redirects=False)

    assert response.status_code == 302
    assert response.headers["location"] == IDP_AUTHORIZE_URL


def test_login_refuses_a_card_whose_access_rule_admits_nobody(
    caplog: pytest.LogCaptureFixture, refusals: list[dict[str, Any]]
) -> None:
    """An unreadable rule degrades to a set admitting nobody, not to an own
    card, so the handshake could only end in a refusal — and the refusal says
    the rule is what has to be fixed, not the roster's mappings."""
    env = {**OIDC_ENV, "OSPREY_AUTH_ROSTER_ACCESS_BOB": "sometimes"}
    with caplog.at_level(logging.WARNING), _client(_app(env)) as client:
        response = client.get(LOGIN_PATH, params={"user": "bob"}, follow_redirects=False)

    assert response.status_code == 403
    assert response.json()["detail"] == "this card's access rule admits nobody"
    assert "admits nobody" in _osprey_log(caplog)
    assert [record["reason"] for record in refusals] == [REASON_UNMAPPED_USER]


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


def test_callback_stores_the_asserted_subject() -> None:
    """The re-issued session carries the provider subject the login accepted, so
    a later verify can name the account without re-contacting the IdP."""
    with _client(_app(client=FakeOIDCClient(userinfo={"sub": ALICE_SUBJECT}))) as client:
        client.get(LOGIN_PATH, params={"user": "alice"}, follow_redirects=False)
        response = client.get(
            CALLBACK_PATH, params={"code": "auth-code", "state": FLOW_STATE}, follow_redirects=False
        )

    entry = _session_from(response).entry("alice")
    assert entry is not None
    assert entry.oidc_subject == ALICE_SUBJECT


def test_callback_stores_the_configured_subject_when_the_claim_differs() -> None:
    """The stored subject is the deployment's canonical mapping, not whatever
    secondary claim the token also happens to carry."""
    env = {**OIDC_ENV, "OSPREY_AUTH_OIDC_CLAIM": "email"}
    userinfo = {"sub": "some-opaque-id", "email": ALICE_SUBJECT}
    with _client(_app(env, FakeOIDCClient(userinfo=userinfo))) as client:
        client.get(LOGIN_PATH, params={"user": "alice"}, follow_redirects=False)
        response = client.get(
            CALLBACK_PATH, params={"code": "auth-code", "state": FLOW_STATE}, follow_redirects=False
        )

    entry = _session_from(response).entry("alice")
    assert entry is not None
    assert entry.oidc_subject == ALICE_SUBJECT


def test_callback_logs_the_accepted_subject(caplog: pytest.LogCaptureFixture) -> None:
    """The subject is an opaque account identifier, not a credential, so the
    success line records it — unlike a refusal, which names only a category."""
    with (
        caplog.at_level(logging.INFO),
        _client(_app(client=FakeOIDCClient(userinfo={"sub": ALICE_SUBJECT}))) as client,
    ):
        client.get(LOGIN_PATH, params={"user": "alice"}, follow_redirects=False)
        response = client.get(
            CALLBACK_PATH, params={"code": "auth-code", "state": FLOW_STATE}, follow_redirects=False
        )

    assert response.status_code == 303
    log = _osprey_log(caplog)
    assert "succeeded" in log
    assert ALICE_SUBJECT in log


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


def test_callback_refuses_a_mapped_identity_that_cannot_be_carried() -> None:
    """A mapped identity carrying a diacritic fails the login *closed*.

    The identity has to reach the terminal as an ``X-Osprey-Auth-Subject``
    header, which is latin-1 on the wire; Osprey requires the stricter ASCII an
    OIDC ``sub`` has by specification. Authorising this login would forward a
    mangled identity or silently no identity at all, so it is refused — with its
    own category, because the fault is in the deployment's claim mapping rather
    than in the operator or the IdP. See
    ``tests/services/auth_sidecar/test_role_payload.py`` for the audited
    category and the ordering against the token exchange.
    """
    env = {**OIDC_ENV, "OSPREY_AUTH_OIDC_SUBJECT_ALICE": "jörg@example.org"}
    app = _app(env, FakeOIDCClient(userinfo={"sub": "jörg@example.org"}))
    with _client(app) as client:
        client.cookies.set(STATE_COOKIE_NAME, _pending("alice"))
        response = client.get(
            CALLBACK_PATH, params={"code": "auth-code", "state": FLOW_STATE}, follow_redirects=False
        )

    assert response.status_code == 403
    assert SESSION_COOKIE_NAME not in response.cookies


def test_callback_compares_a_non_ascii_assertion_without_raising() -> None:
    """A non-ASCII value on the *asserted* side still refuses rather than 500s.

    The mapped identity is ordinary ASCII here, so the uncarryable-identity
    guard does not fire and the comparison itself runs. Compared as ``str``,
    ``secrets.compare_digest`` raises ``TypeError`` on exactly this input — and
    the asserted claim is IdP-supplied, so that would be an unhandled 500 on
    unauthenticated input rather than a refusal.
    """
    app = _app(client=FakeOIDCClient(userinfo={"sub": "jörg@example.org"}))
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


def test_callback_refuses_a_token_with_no_id_token() -> None:
    """An OAuth2 provider where an OIDC one was configured.

    Authlib parses no ID token, so nothing it put in the token can be trusted as
    claims — the same class of fault as an ID token that fails validation, and
    refused the same way rather than continuing on the access token alone.
    """
    app = _app(client=FakeOIDCClient(token={"access_token": "opaque"}))
    with _client(app) as client:
        client.cookies.set(STATE_COOKIE_NAME, _pending("alice"))
        response = client.get(
            CALLBACK_PATH, params={"code": "auth-code", "state": FLOW_STATE}, follow_redirects=False
        )

    assert response.status_code == 502


def test_callback_refuses_a_validated_token_with_no_userinfo() -> None:
    """The ID token was parsed but carried no identity claim: a refusal on the
    identity itself, which is a 403 and not a provider fault."""
    app = _app(
        client=FakeOIDCClient(
            token={"access_token": "opaque", "id_token": "header.payload.signature"}
        )
    )
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


# --- the login-only token gate -----------------------------------------------
#
# `token_admissible` is a pure predicate over already-validated claims, so it is
# exercised directly rather than through a handshake: what it decides has to be
# readable without a token endpoint in the way.

HOSTED_DOMAIN = "lbl.gov"


def test_token_gate_accepts_an_identity_from_the_hosted_domain() -> None:
    """The ordinary Workspace token: both halves name the same domain."""
    admission = token_admissible(
        {"email": "alice@lbl.gov", "hd": HOSTED_DOMAIN}, identity_claim="email"
    )

    assert admission.admissible
    assert admission.reason == ""


def test_token_gate_accepts_a_token_with_no_hosted_domain() -> None:
    """Most providers never emit `hd`; its absence corroborates nothing and
    refuses nothing."""
    admission = token_admissible({"email": "alice@example.org"}, identity_claim="email")

    assert admission.admissible


@pytest.mark.parametrize(
    "hosted",
    ["", None, 0, ["lbl.gov"]],
    ids=["empty", "null", "number", "array"],
)
def test_token_gate_reads_an_unusable_hosted_domain_as_absent(hosted: Any) -> None:
    """A claim carrying no usable string asserts no hosted domain.

    The same reading `_claim_values` gives an unusable group claim: a shape this
    service cannot read is one the provider did not say anything in.
    """
    admission = token_admissible(
        {"email": "alice@example.org", "hd": hosted}, identity_claim="email"
    )

    assert admission.admissible


@pytest.mark.parametrize(
    "hosted",
    ["@lbl.gov", " lbl.gov", "lbl.gov "],
    ids=["at-sign", "leading-space", "trailing-space"],
)
def test_token_gate_refuses_a_hosted_domain_it_cannot_read_as_a_domain(hosted: Any) -> None:
    """A non-empty string `hd` this service cannot read as a domain is a
    contradiction, not silence: the provider said something, and it does not
    corroborate the address. Reading it as absent would admit an address from
    any domain next to a garbage `hd`, which is the fail-open direction."""
    admission = token_admissible({"email": "alice@lbl.gov", "hd": hosted}, identity_claim="email")

    assert not admission.admissible
    assert admission.reason == REASON_HOSTED_DOMAIN_MISMATCH


def test_token_gate_refuses_a_hosted_domain_the_identity_contradicts() -> None:
    """One token, two answers about where its subject belongs."""
    admission = token_admissible(
        {"email": "alice@example.org", "hd": HOSTED_DOMAIN}, identity_claim="email"
    )

    assert not admission.admissible
    assert admission.reason == REASON_HOSTED_DOMAIN_MISMATCH


def test_token_gate_refuses_a_subdomain_of_the_hosted_domain() -> None:
    """The comparison is exact, never by suffix: `als.lbl.gov` is a different
    domain and often a differently delegated one."""
    admission = token_admissible(
        {"email": "alice@als.lbl.gov", "hd": HOSTED_DOMAIN}, identity_claim="email"
    )

    assert not admission.admissible
    assert admission.reason == REASON_HOSTED_DOMAIN_MISMATCH


@pytest.mark.parametrize(
    ("asserted", "hosted"),
    [
        ("Alice@LBL.GOV", "lbl.gov"),
        ("alice@lbl.gov", "LBL.GOV"),
        ("alice@LbL.gOv", "lBl.GoV"),
    ],
    ids=["identity-uppercase", "hosted-uppercase", "both-mixed"],
)
def test_token_gate_folds_case_on_both_sides(asserted: str, hosted: str) -> None:
    """Case in either half is the same domain — a provider that upper-cases one
    of them must not be read as a contradiction."""
    admission = token_admissible({"email": asserted, "hd": hosted}, identity_claim="email")

    assert admission.admissible


def test_token_gate_folds_only_ascii() -> None:
    """Two spellings agreeing only outside A-Z are different domains here.

    The authored `domain:` value went through an ASCII-only fold at render time,
    and this comparison stays in step with it rather than inventing a
    translation between a Unicode-authored domain and a punycode-asserted one.
    """
    admission = token_admissible(
        {"email": "alice@MÜNCHEN.de", "hd": "münchen.de"}, identity_claim="email"
    )

    assert not admission.admissible
    assert admission.reason == REASON_HOSTED_DOMAIN_MISMATCH


@pytest.mark.parametrize(
    "claims",
    [
        {"hd": HOSTED_DOMAIN},
        {"hd": HOSTED_DOMAIN, "email": ""},
        {"hd": HOSTED_DOMAIN, "email": None},
        {"hd": HOSTED_DOMAIN, "email": 42},
    ],
    ids=["absent", "empty", "null", "number"],
)
def test_token_gate_refuses_a_hosted_domain_with_no_asserted_identity(
    claims: dict[str, Any],
) -> None:
    """A token asserting a hosted domain and no identity is refused.

    The closed answer: there is nothing for the hosted domain to corroborate.
    The callback resolves the asserted identity before this gate runs, so a
    login never reaches here on such a token in the first place.
    """
    admission = token_admissible(claims, identity_claim="email")

    assert not admission.admissible
    assert admission.reason == REASON_HOSTED_DOMAIN_MISMATCH


@pytest.mark.parametrize(
    "asserted",
    ["idp|alice", "8f2c1d34-0b6e-4a11-9c77-2f0e5a9b1d43", "alice@", "@lbl.gov"],
    ids=["opaque-subject", "guid", "no-domain", "no-mailbox"],
)
def test_token_gate_accepts_an_identity_that_names_no_domain(asserted: str) -> None:
    """A deployment mapping on `sub` still logs in against a Workspace IdP.

    An identity naming no mailbox in a domain contradicts the hosted domain in
    nothing — and no `domain:` principal can admit such a value either, so
    there is nothing here for the cross-check to protect. Refusing it would
    lock out every deployment whose identity claim is not an address.
    """
    admission = token_admissible({"sub": asserted, "hd": HOSTED_DOMAIN}, identity_claim="sub")

    assert admission.admissible


def test_token_gate_reads_the_configured_identity_claim() -> None:
    """The cross-check is against the claim this deployment maps on, and no
    other claim the token happens to carry."""
    claims = {"sub": "alice@example.org", "email": "alice@lbl.gov", "hd": HOSTED_DOMAIN}

    assert token_admissible(claims, identity_claim="email").admissible
    assert not token_admissible(claims, identity_claim="sub").admissible


def test_token_gate_accepts_a_verified_address() -> None:
    """The provider vouched for it."""
    admission = token_admissible(
        {"email": "alice@lbl.gov", "email_verified": True}, identity_claim="email"
    )

    assert admission.admissible


def test_token_gate_accepts_a_token_that_says_nothing_about_verification() -> None:
    """An absent claim is silence, not a denial: reading it as one would refuse
    every deployment whose IdP does not emit it."""
    admission = token_admissible({"email": "alice@lbl.gov"}, identity_claim="email")

    assert admission.admissible


@pytest.mark.parametrize(
    "verified",
    [False, "false", "False", "FALSE", " false "],
    ids=["boolean", "string", "capitalised", "uppercase", "padded"],
)
def test_token_gate_refuses_an_explicitly_unverified_address(verified: Any) -> None:
    """An explicit false refuses, in either spelling.

    The string is the one non-boolean shape whose meaning is unambiguous, and
    admitting a login the provider declined to vouch for on an encoding
    difference is the wrong direction to be wrong in.
    """
    admission = token_admissible(
        {"email": "alice@lbl.gov", "email_verified": verified}, identity_claim="email"
    )

    assert not admission.admissible
    assert admission.reason == REASON_UNVERIFIED_EMAIL


@pytest.mark.parametrize(
    "verified",
    [True, "true", None, 0, "", "no", ["false"]],
    ids=["boolean", "string", "null", "zero", "empty", "word", "array"],
)
def test_token_gate_accepts_every_shape_that_is_not_an_explicit_false(verified: Any) -> None:
    """Only a false says the address is unverified; every other shape says
    nothing, and nothing is not a denial."""
    admission = token_admissible(
        {"email": "alice@lbl.gov", "email_verified": verified}, identity_claim="email"
    )

    assert admission.admissible


def test_token_gate_names_the_disagreement_when_both_checks_fail() -> None:
    """The contradiction is the finding an operator can act on, so it is the
    one the caller files."""
    admission = token_admissible(
        {"email": "alice@example.org", "hd": HOSTED_DOMAIN, "email_verified": False},
        identity_claim="email",
    )

    assert not admission.admissible
    assert admission.reason == REASON_HOSTED_DOMAIN_MISMATCH


def test_token_gate_accepts_a_token_carrying_neither_claim() -> None:
    """Neither check applies to a bare token, and a gate that refuses what it
    was not asked about would deny every non-Google deployment."""
    admission = token_admissible({"sub": ALICE_SUBJECT}, identity_claim="sub")

    assert admission.admissible
    assert admission.reason == ""


# --- callback: admission by the card's access rule ---------------------------
#
# The roster arm above answers a card that names `roster`. These pin the other
# principals: what a `user:`/`domain:` rule admits, what it refuses, and what
# each refusal leaves behind in the ledger. Every one of them drives the real
# login route first, because a card naming a principal now starts a handshake
# the older gate would have refused.


def _rule_login(app: FastAPI, userinfo: dict[str, Any]) -> httpx.Response:
    """Drive bob's shared card through a full handshake asserting ``userinfo``."""
    app.state.oidc_client = FakeOIDCClient(userinfo=userinfo)
    with _client(app) as client:
        client.get(LOGIN_PATH, params={"user": "bob"}, follow_redirects=False)
        return client.get(
            CALLBACK_PATH, params={"code": "auth-code", "state": FLOW_STATE}, follow_redirects=False
        )


@pytest.mark.parametrize(
    "access",
    ['["domain:example.org"]', '["user:dana@example.org"]', '["self","domain:example.org"]'],
)
def test_callback_admits_an_identity_a_principal_covers(access: str) -> None:
    """The feature in one assertion: an identity on nobody's roster opens a
    shared card because the card's rule names it, or names its domain.

    The session records the ASSERTED identity rather than an opener — there is
    no roster entry behind this login to name — and that field is what verify
    re-runs the rule against on every later subrequest.
    """
    response = _rule_login(_app(_rule_env(access)), {"email": DANA_EMAIL})

    assert response.status_code == 303
    session = _session_from(response)
    assert session.unlocked_usernames(now=0.0) == ("bob",)
    entry = session.entry("bob")
    assert entry is not None
    assert entry.opener == ""
    assert entry.admitted_identity == DANA_EMAIL


def test_callback_prefers_the_opener_when_the_roster_is_what_admits() -> None:
    """A `roster` card is unchanged: the reverse match wins, the opener is
    recorded, and no admitted identity is stored — the opener already says by
    what authority the card was opened, and verify re-validates it that way."""
    env = _rule_env("any", OSPREY_AUTH_OIDC_SUBJECT_ALICE=DANA_EMAIL)
    response = _rule_login(_app(env), {"email": DANA_EMAIL})

    assert response.status_code == 303
    entry = _session_from(response).entry("bob")
    assert entry is not None
    assert entry.opener == "alice"
    assert entry.admitted_identity == ""


def test_callback_refuses_a_roster_identity_the_card_does_not_admit(
    refusals: list[dict[str, Any]],
) -> None:
    """The narrowing that makes `roster` a member worth writing.

    alice is on the roster and her mapped identity is exactly what the IdP
    asserts — under `access: any` that opens bob's card. This card names a
    domain instead, and hers is not in it, so nothing admits her: a card that
    still reverse-matched the roster would make `roster` an unremovable member
    of every rule and there would be no way to say "only this domain".
    """
    env = _rule_env('["domain:example.org"]', OSPREY_AUTH_OIDC_SUBJECT_ALICE="alice@lbl.gov")
    response = _rule_login(_app(env), {"email": "alice@lbl.gov"})

    assert response.status_code == 403
    assert SESSION_COOKIE_NAME not in response.cookies
    assert [record["reason"] for record in refusals] == [REASON_NO_COVERING_PRINCIPAL]


def test_callback_refuses_an_identity_no_principal_covers(
    refusals: list[dict[str, Any]],
) -> None:
    """A neighbouring domain is a different domain: there is no subdomain or
    suffix reading of a `domain:` principal."""
    env = _rule_env('["domain:example.org"]')
    response = _rule_login(_app(env), {"email": "dana@corp.example.org"})

    assert response.status_code == 403
    assert SESSION_COOKIE_NAME not in response.cookies
    assert [record["reason"] for record in refusals] == [REASON_NO_COVERING_PRINCIPAL]


@pytest.mark.parametrize(
    ("userinfo", "reason"),
    [
        (
            {"email": DANA_EMAIL, "hd": "elsewhere.org"},
            REASON_HOSTED_DOMAIN_MISMATCH,
        ),
        (
            {"email": DANA_EMAIL, "email_verified": False},
            REASON_UNVERIFIED_EMAIL,
        ),
    ],
)
def test_callback_refuses_a_token_it_cannot_admit_on(
    userinfo: dict[str, Any], reason: str, refusals: list[dict[str, Any]]
) -> None:
    """The rule would admit this identity; the token's own story is what
    refuses it. Asked once, here, because nothing downstream sees these claims
    again — and filed under the check that objected, which is the half an
    operator can act on."""
    response = _rule_login(_app(_rule_env('["domain:example.org"]')), userinfo)

    assert response.status_code == 403
    assert SESSION_COOKIE_NAME not in response.cookies
    assert [record["reason"] for record in refusals] == [reason]


def test_callback_refuses_an_admitted_identity_that_cannot_be_carried(
    refusals: list[dict[str, Any]],
) -> None:
    """The rule covers this identity — the diacritic is in the local part, so
    the domain still matches — but it has to reach the terminal in an identity
    header, and it cannot. Refused under its own category rather than left to
    `with_user`, whose ValueError would answer a denial with a 500.
    """
    response = _rule_login(_app(_rule_env('["domain:example.org"]')), {"email": "jörg@example.org"})

    assert response.status_code == 403
    assert SESSION_COOKIE_NAME not in response.cookies
    assert [record["reason"] for record in refusals] == [REASON_UNSAFE_ASSERTED_IDENTITY]


@pytest.mark.parametrize(
    ("access", "overrides"),
    [
        pytest.param(
            '["roster","user:dana@example.org"]',
            {"OSPREY_AUTH_OIDC_SUBJECT_CAROL": DANA_EMAIL},
            id="a-roster-entry-and-a-named-principal",
        ),
        pytest.param(
            '["user:dana@example.org","user:DANA@example.org"]',
            {},
            id="two-colliding-named-principals",
        ),
    ],
)
def test_callback_refuses_an_identity_two_authorities_admit(
    access: str, overrides: dict[str, str], refusals: list[dict[str, Any]]
) -> None:
    """Two authorities admitting one identity is a configuration fault, not a
    tie to be broken. Which one won would decide by declaration order what the
    session records — an opener, or an admitted identity — and those are
    re-validated on different terms every subrequest.
    """
    response = _rule_login(_app(_rule_env(access, **overrides)), {"email": DANA_EMAIL})

    assert response.status_code == 403
    assert SESSION_COOKIE_NAME not in response.cookies
    assert [record["reason"] for record in refusals] == [REASON_AMBIGUOUS_IDENTITY]


@pytest.mark.parametrize(
    "asserted",
    [
        pytest.param(DANA_EMAIL, id="a-mapped-roster-identity"),
        pytest.param(BOB_EMAIL, id="the-own"),
    ],
)
def test_callback_admits_nobody_on_an_unreadable_access_rule(
    asserted: str, refusals: list[dict[str, Any]]
) -> None:
    """The fail-closed end of warn-and-degrade, pinned at the callback as well
    as at the gate: an unreadable rule admits nobody — not the identity a
    correctly spelled rule would have named, and not the card's own user
    either, because a set that could not be read carries no `self`. Driven
    through a forged pending flow, because the login route refuses this card
    before the IdP is ever reached.
    """
    env = {
        **_rule_env("sometimes"),
        "OSPREY_AUTH_OIDC_SUBJECT_ALICE": DANA_EMAIL,
        "OSPREY_AUTH_OIDC_SUBJECT_BOB": BOB_EMAIL,
    }
    app = _app(env, FakeOIDCClient(userinfo={"email": asserted}))
    with _client(app) as client:
        client.cookies.set(STATE_COOKIE_NAME, _pending("bob"))
        response = client.get(
            CALLBACK_PATH, params={"code": "auth-code", "state": FLOW_STATE}, follow_redirects=False
        )

    assert response.status_code == 403
    assert SESSION_COOKIE_NAME not in response.cookies
    assert [record["reason"] for record in refusals] == [REASON_IDENTITY_MISMATCH]


# Every way a shared card can refuse a login that reached the callback, so the
# two properties below are asserted over the same closed set: the answer the
# browser gets, and what is left behind in the log and the ledger.
REFUSAL_CASES = [
    pytest.param('["domain:example.org"]', {}, {"email": "dana@corp.example.org"}, id="no-rule"),
    pytest.param(
        '["domain:example.org"]',
        {},
        {"email": DANA_EMAIL, "hd": "elsewhere.org"},
        id="token-not-admissible",
    ),
    pytest.param(
        '["domain:example.org"]',
        {},
        {"email": DANA_EMAIL, "email_verified": False},
        id="unverified-address",
    ),
    pytest.param(
        '["domain:example.org"]', {}, {"email": "jörg@example.org"}, id="admitted-uncarryable"
    ),
    pytest.param(
        '["domain:example.org"]',
        {"OSPREY_AUTH_OIDC_SUBJECT_BOB": BOB_EMAIL},
        {"email": BOB_EMAIL},
        id="owner-without-self",
    ),
    pytest.param("any", {}, {"email": DANA_EMAIL}, id="roster-miss"),
]

AMBIGUOUS_CASE = pytest.param(
    '["user:dana@example.org","user:DANA@example.org"]',
    {},
    {"email": DANA_EMAIL},
    id="ambiguous",
)
"""Kept out of the set above: an ambiguity names a configuration fault rather
than a decision about the caller, so it is answered in its own words — but it
is held to the same hygiene as every other refusal."""


@pytest.mark.parametrize(("access", "overrides", "userinfo"), [*REFUSAL_CASES, AMBIGUOUS_CASE])
def test_a_refused_rule_login_leaks_no_identity(
    access: str,
    overrides: dict[str, str],
    userinfo: dict[str, Any],
    refusals: list[dict[str, Any]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Neither the ledger nor the log carries the value that was refused.

    The record names a category and the roster user whose card was clicked;
    the asserted address and its domain are claim values, and an operator has
    them in their own IdP. A record that carried them would make the ledger a
    list of who tried to open which terminal — including, in the owner case,
    an address that is not on the roster under any name.
    """
    with caplog.at_level(logging.WARNING):
        response = _rule_login(_app(_rule_env(access, **overrides)), userinfo)

    assert response.status_code == 403
    assert refusals and all(record["user"] == "bob" for record in refusals)
    written = _osprey_log(caplog) + repr(refusals)
    address = str(userinfo["email"])
    for value in (address, address.rpartition("@")[2], "corp.example.org", DANA_DOMAIN):
        assert value not in written


@pytest.mark.parametrize(("access", "overrides", "userinfo"), REFUSAL_CASES)
def test_rule_refusals_answer_the_caller_identically(
    access: str, overrides: dict[str, str], userinfo: dict[str, Any]
) -> None:
    """A shared card is not an oracle for which principals a deployment names.

    Whatever refused — no covering principal, a token the gate would not admit
    on, an identity the rule covered but could not CARRY, or one mapped to no
    roster entry — the browser is told the same thing with the same status.
    Only the ledger says which. The uncarryable case is the one that has to be
    said out loud: a body naming the carrying problem would confirm that the
    rule covered the caller, which is the fact this answer exists to withhold.
    """
    response = _rule_login(_app(_rule_env(access, **overrides)), userinfo)

    assert response.status_code == 403
    assert response.json()["detail"] == "this identity is not permitted for this user"


# --- callback: the owner of a shared card ------------------------------------
#
# `self` is a principal like any other, which is what makes it worth writing
# beside the others: it is how an operator shares a terminal without handing it
# over. These pin that the owner's own login survives exactly where the rule
# says it should, and stops where it says it should not.


def test_callback_admits_the_owner_of_a_card_that_keeps_self() -> None:
    """`[self, domain:x]` shares bob's card with a domain and keeps bob's own
    login working, even though his mapped identity is outside that domain.

    The session is the own-card shape: no opener, because nobody else opened
    it, and no admitted identity, because his own mapping is what proved it.
    The stored subject is the CONFIGURED spelling, as on every own-card login.
    """
    env = _rule_env('["self","domain:example.org"]', OSPREY_AUTH_OIDC_SUBJECT_BOB=BOB_EMAIL)
    response = _rule_login(_app(env), {"email": "BOB@lbl.gov"})

    assert response.status_code == 303
    entry = _session_from(response).entry("bob")
    assert entry is not None
    assert entry.opener == ""
    assert entry.admitted_identity == ""
    assert entry.oidc_subject == BOB_EMAIL


def test_callback_refuses_the_owner_of_a_card_that_drops_self(
    refusals: list[dict[str, Any]],
) -> None:
    """`[domain:x]` alone hands the card over: the owner is outside that
    domain, `self` is not a member, and nothing else covers him. The
    distinction from the test above is the whole reason `self` is readable in a
    principal list rather than implied by every rule."""
    env = _rule_env('["domain:example.org"]', OSPREY_AUTH_OIDC_SUBJECT_BOB=BOB_EMAIL)
    response = _rule_login(_app(env), {"email": BOB_EMAIL})

    assert response.status_code == 403
    assert SESSION_COOKIE_NAME not in response.cookies
    assert [record["reason"] for record in refusals] == [REASON_NO_COVERING_PRINCIPAL]


def test_callback_admits_the_owner_by_rule_when_the_rule_covers_them() -> None:
    """An owner the rule happens to cover is admitted BY THE RULE, not as the
    owner: `[domain:x]` names no `self`, so what admits bob here is the domain
    principal, and the session records the admitted identity accordingly. That
    is the shape verify re-runs the rule against — an owner arm would have
    recorded nothing for it to re-check."""
    env = _rule_env('["domain:example.org"]', OSPREY_AUTH_OIDC_SUBJECT_BOB="bob@example.org")
    response = _rule_login(_app(env), {"email": "bob@example.org"})

    assert response.status_code == 303
    entry = _session_from(response).entry("bob")
    assert entry is not None
    assert entry.opener == ""
    assert entry.admitted_identity == "bob@example.org"


def test_callback_refuses_an_owner_mapping_that_cannot_be_carried(
    refusals: list[dict[str, Any]],
) -> None:
    """The own arm's counterpart to the roster arm's post-match check. A shared
    card's own mapping decides nothing until the callback, so the pre-exchange
    gate that would have caught this on an own card never ran — and `with_user`
    would answer a denial with a 500."""
    env = _rule_env('["self","domain:example.org"]', OSPREY_AUTH_OIDC_SUBJECT_BOB="jörg@lbl.gov")
    response = _rule_login(_app(env), {"email": "jörg@lbl.gov"})

    assert response.status_code == 403
    assert SESSION_COOKIE_NAME not in response.cookies
    assert [record["reason"] for record in refusals] == [audit.REASON_NON_ASCII_SUBJECT]
