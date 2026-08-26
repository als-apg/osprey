"""The auth sidecar's OIDC handshake, driven against a real signing IdP.

``test_oidc.py`` replaces Authlib at ``app.state.oidc_client``; these tests
replace nothing. The sidecar builds its own client from
``OSPREY_AUTH_OIDC_ISSUER``, fetches a discovery document over a socket, and
hands Authlib tokens signed by :mod:`tests.services.auth_sidecar.mock_idp` — so
what is under test here is the half the fake stands in for: discovery, the
nonce, the ID token's signature, and its ``iss``/``aud``/``exp`` claims. Every
refusal below is Authlib's, arriving at the callback's deliberately broad
``except`` arm; every acceptance is a token that genuinely verified.

**The flow is walked, never assembled.** Each test starts at the login route,
follows the 302 to the provider with a second client (a real request to the
listening stub — the sidecar's ASGI app cannot serve it), and hands the
provider's redirect back to the sidecar as a top-level cross-site GET, cookie
jar and all. Nothing here forges a state cookie: a planted cookie lands at path
``/`` while the server's own carries ``path=/auth``, so the jar would keep
answering with the forgery and any claim about a consumed flow would be
vacuous. Walking the flow is also the only way SameSite=Lax is genuinely
exercised — the callback is a cross-site navigation, and a ``strict`` cookie
would simply not arrive.
"""

from __future__ import annotations

import json
import logging
from base64 import b64decode
from collections.abc import Iterator
from typing import Any

import httpx
import itsdangerous
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces._serving import run_app_server
from osprey.services.auth_sidecar.app import STATE_COOKIE_NAME, create_app
from osprey.services.auth_sidecar.routes.oidc import (
    CALLBACK_PATH,
    CLIENT_NAME,
    DEFAULT_SCOPE,
    LOGIN_PATH,
    PENDING_FLOW_SESSION_KEY,
)
from osprey.services.auth_sidecar.sessions import SESSION_COOKIE_NAME, SessionCodec, SessionState
from tests.services.auth_sidecar.mock_idp import (
    DISCOVERY_AS_HTML,
    DISCOVERY_WITHOUT_AUTHORIZATION_ENDPOINT,
    MockIdP,
)

SESSION_SECRET = "session-secret-value"
STATE_SECRET = "state-secret-value"
EXTERNAL_ORIGIN = "https://testserver"
"""The deployment's public origin. It matches the test client's base URL so the
callback the provider redirects to is one this client can actually be asked
for, exactly as a browser would be."""

SESSION_LIFETIME = 3600

ALICE_SUBJECT = "idp|alice"
CAROL_SUBJECT = "idp|carol"

TOP_LEVEL_NAVIGATION = {
    "Sec-Fetch-Site": "cross-site",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Dest": "document",
}
"""What a browser sends when the IdP redirects it back: a cross-site top-level
GET, which is precisely the request shape SameSite=Lax admits and SameSite=Strict
would strip the state cookie from."""


# --- the provider ------------------------------------------------------------


@pytest.fixture(scope="module")
def idp() -> Iterator[MockIdP]:
    """One stub provider, listening on a real port for the whole module.

    Served once rather than per test: the sidecar reaches it over TCP, so it
    cannot be an ASGI object handed to a client, and standing a server up for
    each of thirty tests costs more than resetting one.
    """
    provider = MockIdP()
    with run_app_server(provider.app) as base_url:
        provider.issuer = base_url
        yield provider


@pytest.fixture(autouse=True)
def _reset_idp(idp: MockIdP) -> Iterator[None]:
    """Give every test an unmodified provider with no recorded requests."""
    idp.reset()
    yield


# --- the sidecar -------------------------------------------------------------


def _env(idp: MockIdP, **overrides: str) -> dict[str, str]:
    """The sidecar's environment, pointed at the running provider.

    alice and carol are mapped; bob is on the roster with no mapped identity.
    """
    env = {
        "OSPREY_AUTH_METHOD": "oidc",
        "OSPREY_AUTH_SESSION_SECRET": SESSION_SECRET,
        "OSPREY_AUTH_STATE_SECRET": STATE_SECRET,
        "OSPREY_AUTH_SESSION_LIFETIME": str(SESSION_LIFETIME),
        "OSPREY_AUTH_USERS": "alice,bob,carol",
        "OSPREY_AUTH_OIDC_ISSUER": idp.issuer,
        "OSPREY_AUTH_OIDC_CLIENT_ID": idp.client_id,
        "OSPREY_AUTH_OIDC_CLIENT_SECRET": idp.client_secret,
        "OSPREY_AUTH_OIDC_SUBJECT_ALICE": ALICE_SUBJECT,
        "OSPREY_AUTH_OIDC_SUBJECT_CAROL": CAROL_SUBJECT,
        "OSPREY_AUTH_EXTERNAL_ORIGIN": EXTERNAL_ORIGIN,
        "OSPREY_AUTH_TLS_ENABLED": "true",
    }
    env.update(overrides)
    return env


def _sidecar(idp: MockIdP, **overrides: str) -> FastAPI:
    """A sidecar app with no test double anywhere in it."""
    return create_app(_env(idp, **overrides))


def _browser(app: FastAPI) -> TestClient:
    """A client on the deployment's own origin, keeping its cookies.

    The origin matters: both cookies carry ``Secure`` under TLS, and a client on
    ``http://`` would drop them and turn every handshake into "no login in
    flight" for a reason unrelated to what is being tested.
    """
    return TestClient(app, base_url=EXTERNAL_ORIGIN)


# --- walking the handshake ---------------------------------------------------


def _start_login(
    client: TestClient, user: str, *, next_target: str | None = None
) -> httpx.Response:
    """Click ``user``'s card. Returns the sidecar's redirect to the provider."""
    params = {"user": user}
    if next_target is not None:
        params["next"] = next_target
    return client.get(LOGIN_PATH, params=params, follow_redirects=False)


def _visit_identity_provider(location: str) -> httpx.Response:
    """Follow the redirect to the provider with a real HTTP request.

    A separate client on purpose: the provider is a listening server, not the
    ASGI app the sidecar's test client is bound to, and routing this hop through
    that client would send the authorize request to the sidecar instead.
    """
    with httpx.Client(follow_redirects=False, timeout=10.0) as browser:
        return browser.get(location)


def _return_to_sidecar(client: TestClient, location: str) -> httpx.Response:
    """Hand the provider's redirect back to the sidecar as a browser would."""
    return client.get(location, follow_redirects=False, headers=TOP_LEVEL_NAVIGATION)


def _log_in(client: TestClient, user: str, *, next_target: str | None = None) -> httpx.Response:
    """Walk a whole handshake for ``user``; returns the callback's response."""
    login = _start_login(client, user, next_target=next_target)
    assert login.status_code == 302, login.text
    handoff = _visit_identity_provider(login.headers["location"])
    assert handoff.status_code == 302, handoff.text
    return _return_to_sidecar(client, handoff.headers["location"])


# --- reading what the cookies carry ------------------------------------------


def _state_payload(client: TestClient) -> dict[str, Any]:
    """The decoded Starlette session the state cookie currently holds.

    Read out of the client's jar, so it reflects what the browser would send
    next — the only honest place to assert that a handshake was consumed.
    """
    raw = client.cookies.get(STATE_COOKIE_NAME)
    if not raw:
        return {}
    unsigned = itsdangerous.TimestampSigner(STATE_SECRET).unsign(raw)
    payload: dict[str, Any] = json.loads(b64decode(unsigned))
    return payload


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


def _osprey_log(caplog: pytest.LogCaptureFixture) -> str:
    """Only this service's log records, never the test clients' request logs."""
    return "\n".join(
        record.getMessage() for record in caplog.records if record.name.startswith("osprey.")
    )


# --- the round trip ----------------------------------------------------------


def test_a_real_handshake_unlocks_the_clicked_user(idp: MockIdP) -> None:
    """The whole flow against a real IdP: discovery, redirect, signed token, session.

    Everything Authlib validates is validated here for real — this is the case
    every refusal below is a single-value deviation from.
    """
    with _browser(_sidecar(idp)) as client:
        response = _log_in(client, "alice", next_target="/u/alice/tuning?panel=orbit")

    assert response.status_code == 303
    assert response.headers["location"] == "/u/alice/tuning?panel=orbit"

    session = _session_from(response)
    assert session.unlocked_usernames(now=0.0) == ("alice",)
    entry = session.entry("alice")
    assert entry is not None
    # OIDC sessions carry no credential-generation tag; there is no stored hash
    # to compute one from.
    assert entry.generation_tag == ""

    authorize = idp.authorize_requests[-1]
    assert authorize["client_id"] == idp.client_id
    assert authorize["response_type"] == "code"
    assert authorize["scope"] == DEFAULT_SCOPE
    # `openid` in the scope is what makes Authlib generate and later check a nonce.
    assert authorize["nonce"]


def test_the_redirect_uri_is_identical_on_both_legs(idp: MockIdP) -> None:
    """Authorization and token exchange must present the same ``redirect_uri``.

    An IdP compares the two byte for byte and refuses the exchange when they
    differ, so this is the assertion that the deployment's registered callback
    is what actually travels — the value the sidecar derives from its own
    origin, never from the inbound request.
    """
    expected = f"{EXTERNAL_ORIGIN}{CALLBACK_PATH}"
    with _browser(_sidecar(idp)) as client:
        response = _log_in(client, "alice")

    assert response.status_code == 303
    assert idp.authorize_requests[-1]["redirect_uri"] == expected
    assert idp.token_requests[-1]["redirect_uri"] == expected
    assert idp.token_requests[-1]["grant_type"] == "authorization_code"


def test_authlibs_state_entry_is_written_and_then_cleared(idp: MockIdP) -> None:
    """The state cookie holds Authlib's own entry beside the sidecar's, briefly.

    ``_state_osprey_oidc_<state>`` is the shape the pinned cookie is sized and
    scoped for, and both entries have to be gone once the handshake completes or
    the next login inherits a half-finished one.
    """
    with _browser(_sidecar(idp)) as client:
        login = _start_login(client, "alice")
        assert login.status_code == 302

        # Read out of the redirect the browser is about to follow, not out of
        # the provider's records: the state cookie has to be readable at exactly
        # this point, before the IdP has seen anything.
        outbound = httpx.URL(login.headers["location"]).params
        started = _state_payload(client)
        authlib_key = f"_state_{CLIENT_NAME}_{outbound['state']}"
        assert authlib_key in started
        assert started[authlib_key]["data"]["nonce"] == outbound["nonce"]
        assert started[authlib_key]["data"]["redirect_uri"] == f"{EXTERNAL_ORIGIN}{CALLBACK_PATH}"
        assert started[PENDING_FLOW_SESSION_KEY]["user"] == "alice"
        assert started[PENDING_FLOW_SESSION_KEY]["state"] == outbound["state"]

        handoff = _visit_identity_provider(login.headers["location"])
        response = _return_to_sidecar(client, handoff.headers["location"])

    assert response.status_code == 303
    finished = _state_payload(client)
    assert authlib_key not in finished
    assert PENDING_FLOW_SESSION_KEY not in finished


def test_the_callback_arrives_as_a_lax_cross_site_navigation(idp: MockIdP) -> None:
    """The state cookie is scoped for the redirect that actually comes back.

    ``SameSite=Lax`` and ``path=/auth`` are what let a top-level GET from the
    IdP carry the handshake; ``Strict`` would drop it and the login would fail
    as "no login is in progress".
    """
    with _browser(_sidecar(idp)) as client:
        login = _start_login(client, "alice")
        header = _set_cookie_header(login, STATE_COOKIE_NAME).lower()
        assert "samesite=lax" in header
        assert "path=/auth" in header
        assert "max-age=300" in header
        assert "httponly" in header

        handoff = _visit_identity_provider(login.headers["location"])
        response = _return_to_sidecar(client, handoff.headers["location"])

    assert response.status_code == 303


def test_a_completed_handshake_cannot_be_replayed(idp: MockIdP) -> None:
    """The consumed flow is gone from the browser's own cookie, not just the server's."""
    with _browser(_sidecar(idp)) as client:
        login = _start_login(client, "alice")
        handoff = _visit_identity_provider(login.headers["location"])
        callback_url = handoff.headers["location"]

        first = _return_to_sidecar(client, callback_url)
        replay = _return_to_sidecar(client, callback_url)

    assert first.status_code == 303
    assert replay.status_code == 400
    assert SESSION_COOKIE_NAME not in replay.cookies


def test_identity_comes_from_the_id_token_not_the_userinfo_endpoint(idp: MockIdP) -> None:
    """The userinfo endpoint asserts carol; the token asserts alice; alice wins.

    A sidecar that fell back to the userinfo endpoint would either unlock the
    wrong user or refuse the right one, and either way this counter would move.
    """
    with _browser(_sidecar(idp)) as client:
        response = _log_in(client, "alice")

    assert response.status_code == 303
    assert idp.userinfo_requests == 0


# --- the ID token has to validate --------------------------------------------


def test_a_replayed_nonce_is_refused(idp: MockIdP, caplog: pytest.LogCaptureFixture) -> None:
    """A token whose nonce is not the one this browser's handshake carries.

    The nonce is the binding between the authorization request and the token;
    without this check a token minted for one login could be planted into
    another.
    """
    idp.nonce_claim = "a-nonce-from-another-handshake"
    with caplog.at_level(logging.WARNING), _browser(_sidecar(idp)) as client:
        response = _log_in(client, "alice")

    assert response.status_code == 502
    assert SESSION_COOKIE_NAME not in response.cookies
    assert "did not validate" in _osprey_log(caplog)


def test_a_token_signed_by_an_unpublished_key_is_refused(
    idp: MockIdP, caplog: pytest.LogCaptureFixture
) -> None:
    """Signed with the provider's ``kid`` but not its key: the signature fails.

    The matching ``kid`` is what makes this a signature test — an unknown one
    would be refused as a missing key, which is a different path.
    """
    idp.sign_with_intruder_key = True
    with caplog.at_level(logging.WARNING), _browser(_sidecar(idp)) as client:
        response = _log_in(client, "alice")

    assert response.status_code == 502
    assert SESSION_COOKIE_NAME not in response.cookies
    assert "did not validate" in _osprey_log(caplog)


def test_a_token_from_another_issuer_is_refused(
    idp: MockIdP, caplog: pytest.LogCaptureFixture
) -> None:
    """``iss`` must be the issuer the discovery document named."""
    idp.issuer_claim = "https://idp.evil.example.org"
    with caplog.at_level(logging.WARNING), _browser(_sidecar(idp)) as client:
        response = _log_in(client, "alice")

    assert response.status_code == 502
    assert SESSION_COOKIE_NAME not in response.cookies
    assert "did not validate" in _osprey_log(caplog)


def test_a_token_for_another_audience_is_refused(
    idp: MockIdP, caplog: pytest.LogCaptureFixture
) -> None:
    """A token naming another relying party as its audience does not log anyone in.

    The refusal has to be the audience check itself. Before the callback named
    ``aud`` in its ``claims_options`` this same token was still refused, but by
    the OIDC ``azp`` rule — an ``aud`` that is not this client makes ``azp``
    required, and it was absent — which is why the logged class name is asserted
    here: ``InvalidClaimError`` is the audience comparison, ``MissingClaimError``
    would mean the audience was never compared and a claim the IdP chooses
    whether to send did the work.
    """
    idp.audience_claim = "some-other-relying-party"
    with caplog.at_level(logging.WARNING), _browser(_sidecar(idp)) as client:
        response = _log_in(client, "alice")

    assert response.status_code == 502
    assert SESSION_COOKIE_NAME not in response.cookies
    log = _osprey_log(caplog)
    assert "did not validate" in log
    assert "InvalidClaimError" in log
    assert "MissingClaimError" not in log


def test_a_token_for_another_audience_is_refused_even_when_azp_names_this_client(
    idp: MockIdP,
) -> None:
    """The case the ``azp`` rule cannot catch, and the reason ``aud`` is checked.

    ``azp`` naming this client satisfies every OIDC rule Authlib applies on its
    own, so this token — minted for a *different* relying party — was accepted
    outright until the audience was named in ``claims_options``: 303, with a
    session cookie. Nothing but the audience comparison refuses it, which makes
    this the one test that fails the moment that argument is dropped.
    """
    idp.audience_claim = "some-other-relying-party"
    idp.extra_claims = {"azp": idp.client_id}
    with _browser(_sidecar(idp)) as client:
        response = _log_in(client, "alice")

    assert response.status_code == 502
    assert SESSION_COOKIE_NAME not in response.cookies


@pytest.mark.parametrize("slash_in_token", [False, True], ids=["bare-iss", "slashed-iss"])
def test_a_trailing_slash_on_the_configured_issuer_locks_nobody_out(
    idp: MockIdP, slash_in_token: bool
) -> None:
    """Naming the claims to check must not invent an issuer requirement.

    Supplying ``claims_options`` replaces Authlib's default ``iss`` entry rather
    than extending it, so the callback rebuilds that entry — and an issuer
    configured with a trailing slash is the same issuer, because the discovery
    URL is built by stripping exactly that character. Which spelling reaches the
    token is the IdP's choice (Auth0 publishes the slashed form, Keycloak the
    bare one), so an operator who configured it with the slash has to be able to
    log in against either. Both cases are here because each one is what fails
    when the other spelling is dropped from the list.
    """
    if slash_in_token:
        idp.issuer_claim = f"{idp.issuer}/"
    app = _sidecar(idp, OSPREY_AUTH_OIDC_ISSUER=f"{idp.issuer}/")
    with _browser(app) as client:
        response = _log_in(client, "alice")

    assert response.status_code == 303
    assert _session_from(response).unlocked_usernames(now=0.0) == ("alice",)


def test_an_expired_token_is_refused(idp: MockIdP, caplog: pytest.LogCaptureFixture) -> None:
    """Well past Authlib's 120-second leeway, so this is expiry and not clock skew."""
    idp.issued_at_offset = -3600
    idp.token_lifetime = 300
    with caplog.at_level(logging.WARNING), _browser(_sidecar(idp)) as client:
        response = _log_in(client, "alice")

    assert response.status_code == 502
    assert SESSION_COOKIE_NAME not in response.cookies
    assert "did not validate" in _osprey_log(caplog)


def test_a_token_response_without_an_id_token_is_refused(idp: MockIdP) -> None:
    """An OAuth2 provider where an OIDC one was configured.

    Authlib parses no ID token, so nothing in the token response has been
    validated and there is no identity to compare. The callback refuses on the
    absent ID token itself — a provider fault of the same class as one that
    fails validation, and never a login on the access token alone.
    """
    idp.omit_id_token = True
    with _browser(_sidecar(idp)) as client:
        response = _log_in(client, "alice")

    assert response.status_code == 502
    assert SESSION_COOKIE_NAME not in response.cookies


# --- the provider misbehaves -------------------------------------------------


@pytest.mark.parametrize(
    "posture",
    [DISCOVERY_WITHOUT_AUTHORIZATION_ENDPOINT, DISCOVERY_AS_HTML],
    ids=["no-authorization-endpoint", "not-json"],
)
def test_an_unusable_discovery_document_is_a_bad_gateway(idp: MockIdP, posture: str) -> None:
    """Reachable but wrong is still the IdP's problem, not a 500.

    A mistyped issuer lands on something that answers 200 — a captive portal, a
    plain web server — and the failure surfaces inside Authlib rather than as an
    httpx error.
    """
    idp.discovery = posture
    with _browser(_sidecar(idp)) as client:
        response = _start_login(client, "alice")

    assert response.status_code == 502
    assert idp.authorize_requests == []


def test_a_rejected_consent_screen_is_a_refusal(idp: MockIdP) -> None:
    """The IdP sends the browser back with ``error=`` instead of a code."""
    idp.authorize_error = "access_denied"
    with _browser(_sidecar(idp)) as client:
        response = _log_in(client, "alice")

    assert response.status_code == 400
    assert SESSION_COOKIE_NAME not in response.cookies


def test_a_wrong_client_secret_is_a_refusal_not_a_crash(idp: MockIdP) -> None:
    """The token endpoint really checks the credentials the sidecar presents."""
    app = _sidecar(idp, OSPREY_AUTH_OIDC_CLIENT_SECRET="not-the-registered-secret")
    with _browser(app) as client:
        response = _log_in(client, "alice")

    assert response.status_code == 400
    assert SESSION_COOKIE_NAME not in response.cookies


# --- the identity has to be the clicked user's -------------------------------


def test_an_identity_mapped_to_nobody_is_refused(
    idp: MockIdP, caplog: pytest.LogCaptureFixture
) -> None:
    """A verified token is not an authorisation: the subject must be alice's."""
    idp.subject = "idp|stranger"
    with caplog.at_level(logging.WARNING), _browser(_sidecar(idp)) as client:
        response = _log_in(client, "alice")

    assert response.status_code == 403
    assert SESSION_COOKIE_NAME not in response.cookies
    assert "alice" in _osprey_log(caplog)


def test_an_identity_mapped_to_another_user_is_refused(idp: MockIdP) -> None:
    """carol's real, verified identity does not open alice's terminal.

    Nor carol's: the clicked card is the only user a handshake can unlock, so no
    session cookie is issued at all.
    """
    idp.subject = CAROL_SUBJECT
    with _browser(_sidecar(idp)) as client:
        response = _log_in(client, "alice")

    assert response.status_code == 403
    assert SESSION_COOKIE_NAME not in response.cookies


def test_a_user_with_no_mapped_identity_never_reaches_the_provider(idp: MockIdP) -> None:
    """bob's handshake could only end in a refusal, so it does not start."""
    with _browser(_sidecar(idp)) as client:
        response = _start_login(client, "bob")

    assert response.status_code == 403
    assert idp.authorize_requests == []


def test_a_non_ascii_identity_is_refused_over_the_wire(idp: MockIdP) -> None:
    """An identity the boundary cannot carry is refused, not half-forwarded.

    A mapped identity travels onward to the terminal as an
    ``X-Osprey-Auth-Subject`` header, which is latin-1 on the wire and ASCII by
    Osprey's stricter rule — which an OIDC ``sub`` already satisfies by
    specification. This deployment mapped a claim spelling that does not, so the
    handshake completes against the IdP and the login is then refused with a
    category pointing at the configuration. The refusal is a clean 403 rather
    than the ``TypeError`` a ``str`` comparison would raise on this input, which
    is why :func:`~osprey.services.auth_sidecar.routes.oidc._same_value`
    compares UTF-8 bytes.
    """
    idp.subject = "jörg@example.org"
    app = _sidecar(idp, OSPREY_AUTH_OIDC_SUBJECT_ALICE="jörg@example.org")
    with _browser(app) as client:
        response = _log_in(client, "alice")

    assert response.status_code == 403
    assert SESSION_COOKIE_NAME not in response.cookies


def test_a_configured_claim_other_than_sub_is_honoured(idp: MockIdP) -> None:
    """Facilities commonly map on ``preferred_username`` rather than ``sub``."""
    idp.subject = "an-opaque-provider-id"
    idp.extra_claims = {"preferred_username": "alice@example.org"}
    app = _sidecar(
        idp,
        OSPREY_AUTH_OIDC_CLAIM="preferred_username",
        OSPREY_AUTH_OIDC_SUBJECT_ALICE="alice@example.org",
    )
    with _browser(app) as client:
        response = _log_in(client, "alice")

    assert response.status_code == 303
    assert _session_from(response).unlocked_usernames(now=0.0) == ("alice",)


def test_the_session_cookie_carries_the_pinned_attributes(idp: MockIdP) -> None:
    """HttpOnly, SameSite=Lax, Secure under TLS, and path ``/`` so it reaches
    the ``/u/<user>/`` locations it authorises."""
    with _browser(_sidecar(idp)) as client:
        response = _log_in(client, "alice")

    header = _set_cookie_header(response, SESSION_COOKIE_NAME).lower()
    assert "httponly" in header
    assert "samesite=lax" in header
    assert "secure" in header
    assert "path=/;" in header or header.rstrip().endswith("path=/")


def test_a_refusal_never_echoes_the_token_or_the_client_secret(
    idp: MockIdP, caplog: pytest.LogCaptureFixture
) -> None:
    """A real signed token is in flight here, so this is the case that matters:
    neither it, nor the identity it asserts, nor the client secret may appear in
    a response or a log line."""
    idp.subject = CAROL_SUBJECT
    with caplog.at_level(logging.DEBUG), _browser(_sidecar(idp)) as client:
        response = _log_in(client, "alice")

    assert response.status_code == 403
    issued = idp.token_requests[-1]["code"]
    for value in (idp.client_secret, issued, CAROL_SUBJECT):
        assert value not in response.text
        assert value not in _osprey_log(caplog)
