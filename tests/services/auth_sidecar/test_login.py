"""Tests for the auth sidecar's password login page and credential POST.

Three properties carry most of the weight here, and each is pinned from several
directions:

* **The page is addressed per user.** It states the username and never asks for
  one, and the username it unlocks travels into the session cookie byte for byte
  — verify exact-matches it against the roster, so any normalisation here would
  break the chain open rather than closed.
* **Every refusal is the same refusal.** A wrong password, a roster user with no
  provisioned credential and a username that is not on the roster at all produce
  the same status, the same page and the same call into the throttle. The
  identity of those three responses is asserted byte for byte.
* **The throttle runs before the credential.** An attempt arriving inside an open
  window is refused with the *correct* password in hand, and it does not grow
  the window it was refused by.

Apps are always built from an explicit env mapping, so a variable left in the
real process environment cannot make one of these pass.
"""

from __future__ import annotations

import logging
import re
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from osprey.services.auth_sidecar.app import create_app
from osprey.services.auth_sidecar.passwords import generation_tag, hash_password
from osprey.services.auth_sidecar.return_to import MAX_RETURN_TO_LENGTH
from osprey.services.auth_sidecar.routes.login import (
    DENIAL_MESSAGE,
    LOGIN_PATH,
    MAX_LOCATION_LENGTH,
    MAX_USERNAME_LENGTH,
    OIDC_LOGIN_PATH,
    THROTTLE_MESSAGE,
)
from osprey.services.auth_sidecar.sessions import SESSION_COOKIE_NAME, SessionCodec, SessionState
from osprey.services.auth_sidecar.throttle import AttemptThrottle

SESSION_SECRET = "session-secret-value"
STATE_SECRET = "state-secret-value"
SESSION_LIFETIME = 3600
EXTERNAL_ORIGIN = "https://terminals.example.org"

ALICE_PASSWORD = "alice-password"
ALICE_HASH = hash_password(ALICE_PASSWORD)

# Three five-character usernames, so the three refusal bodies can be compared
# byte for byte after substituting the name each one echoes back:
#   alice — on the roster, with a provisioned credential
#   carol — on the roster, with none
#   mally — not on the roster at all
PASSWORD_ENV = {
    "OSPREY_AUTH_METHOD": "password",
    "OSPREY_AUTH_SESSION_SECRET": SESSION_SECRET,
    "OSPREY_AUTH_SESSION_LIFETIME": str(SESSION_LIFETIME),
    "OSPREY_AUTH_USERS": "alice,bob,carol",
    "OSPREY_AUTH_PW_HASH_ALICE": ALICE_HASH,
    "OSPREY_AUTH_PW_HASH_BOB": hash_password("bob-password"),
    "OSPREY_AUTH_EXTERNAL_ORIGIN": EXTERNAL_ORIGIN,
    "OSPREY_AUTH_TLS_ENABLED": "true",
}

OIDC_ENV = {
    "OSPREY_AUTH_METHOD": "oidc",
    "OSPREY_AUTH_SESSION_SECRET": SESSION_SECRET,
    "OSPREY_AUTH_STATE_SECRET": STATE_SECRET,
    "OSPREY_AUTH_SESSION_LIFETIME": str(SESSION_LIFETIME),
    "OSPREY_AUTH_USERS": "alice,bob",
    "OSPREY_AUTH_OIDC_ISSUER": "https://idp.example.org",
    "OSPREY_AUTH_OIDC_CLIENT_ID": "client-id",
    "OSPREY_AUTH_OIDC_CLIENT_SECRET": "client-secret-value",
    "OSPREY_AUTH_EXTERNAL_ORIGIN": EXTERNAL_ORIGIN,
    "OSPREY_AUTH_TLS_ENABLED": "true",
}


BROWSER_ACCEPT = {"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}
"""What a top-level navigation sends. The header Chrome, Firefox and Safari all
put on a link click, which is the only thing the route's HTML branch keys on."""

JSON_ACCEPT = {"Accept": "application/json"}
"""What a client that will parse the answer sends, and must keep getting."""


def _client(env: dict[str, str] | None = None) -> TestClient:
    """A test client over a sidecar built from ``env`` (password mode by default).

    Addressed over ``https``, matching the TLS deployment the default env
    describes: the cookie this route issues carries ``Secure`` there, and a
    client talking http would silently drop it — the same failure an operator
    sees when ``tls_enabled`` disagrees with the real origin.
    """
    app = create_app(env if env is not None else PASSWORD_ENV)
    return TestClient(app, base_url="https://testserver")


def _issued_cookie(response: httpx.Response) -> str:
    """The session cookie value a response set.

    Read off the header rather than out of the client's jar: a test that seeds a
    cookie of its own would otherwise leave two entries under one name, and the
    assertion would depend on which the jar hands back.
    """
    header = response.headers["set-cookie"]
    return header.split(";", 1)[0].split("=", 1)[1]


def _codec(secret: str = SESSION_SECRET) -> SessionCodec:
    """A client-side codec for reading (or forging) the cookie the route issues."""
    return SessionCodec(secret, max_age=SESSION_LIFETIME)


def _post(client: TestClient, data: dict[str, object], **kwargs: object) -> httpx.Response:
    """POST to the login route the way a browser on this deployment would.

    Every POST carries an ``Origin``, because the route refuses one that does not
    — a form that will not say where it came from cannot be told from a
    cross-site submission. Tests about that check override the header.
    """
    headers = {"Origin": EXTERNAL_ORIGIN, **dict(kwargs.pop("headers", {}) or {})}
    return client.post(LOGIN_PATH, data=data, headers=headers, follow_redirects=False, **kwargs)


def _login(
    client: TestClient,
    *,
    user: str = "alice",
    password: str = ALICE_PASSWORD,
    next_value: str = "",
) -> httpx.Response:
    """POST one credential attempt, without following the success redirect."""
    return _post(client, {"user": user, "next": next_value, "password": password})


def _cookie_attributes(header: str) -> dict[str, str]:
    """The Set-Cookie attributes, lowercased, as ``{name: value}`` (flags map to "")."""
    attributes: dict[str, str] = {}
    for chunk in header.split(";")[1:]:
        name, _, value = chunk.strip().partition("=")
        attributes[name.lower()] = value
    return attributes


def _inputs(body: str) -> list[str]:
    """Every ``<input>`` tag in the page, as raw markup."""
    return re.findall(r"<input[^>]*>", body, flags=re.IGNORECASE | re.DOTALL)


class FakeClock:
    """A monotonic clock the test moves by hand.

    The throttle's windows are seconds long, so tests that let them lapse in real
    time would be pinning the machine's speed rather than the route's behaviour —
    and tests that stay inside one would be assuming two requests land within a
    second of each other.
    """

    def __init__(self) -> None:
        self.time = 1000.0

    def advance(self, seconds: float) -> None:
        self.time += seconds

    def __call__(self) -> float:
        return self.time


def _throttled_client(
    env: dict[str, str] | None = None,
) -> tuple[TestClient, AttemptThrottle, FakeClock]:
    """A client whose sidecar throttles on a clock the test controls.

    The throttle is replaced on ``app.state`` — the one place every route reaches
    it through — so the route under test is unmodified.
    """
    app = create_app(env if env is not None else PASSWORD_ENV)
    clock = FakeClock()
    throttle = AttemptThrottle(clock=clock)
    app.state.attempt_throttle = throttle
    return TestClient(app, base_url="https://testserver"), throttle, clock


# --- The page ---------------------------------------------------------------


def test_page_states_the_user_and_asks_only_for_a_password() -> None:
    """The acceptance gate: "Enter password for <user>" and no username field."""
    response = _client().get(LOGIN_PATH, params={"user": "alice", "next": "/u/alice/"})

    assert response.status_code == 200
    assert "Enter password for" in response.text
    assert "alice" in response.text

    fields = _inputs(response.text)
    # One password input, and every other field hidden: nothing on this page
    # invites typing a username.
    assert sum('type="password"' in field for field in fields) == 1
    editable = [field for field in fields if 'type="hidden"' not in field]
    assert len(editable) == 1
    assert 'type="password"' in editable[0]
    assert 'name="user"' not in editable[0]
    # The username still travels with the form, as a value the browser cannot edit.
    assert any('type="hidden"' in field and 'name="user"' in field for field in fields)


def test_page_is_never_cached() -> None:
    """A shared control-room browser must not re-serve a previous prompt."""
    response = _client().get(LOGIN_PATH, params={"user": "alice", "next": ""})

    assert response.headers["cache-control"] == "no-store"


def test_page_escapes_the_username_it_echoes() -> None:
    """A roster name carrying markup renders as text, not as markup."""
    env = {**PASSWORD_ENV, "OSPREY_AUTH_USERS": "alice,<script>x</script>"}
    response = _client(env).get(LOGIN_PATH, params={"user": "<script>x</script>"})

    assert response.status_code == 200
    assert "<script>x</script>" not in response.text
    assert "&lt;script&gt;" in response.text


def test_empty_next_falls_back_to_the_users_own_terminal() -> None:
    """nginx emits an empty ``next`` when the original path fails its allowlist."""
    response = _client().get(LOGIN_PATH, params={"user": "alice", "next": ""})

    assert response.status_code == 200
    assert 'name="next" value="/u/alice/"' in response.text


def test_a_same_origin_next_is_carried_into_the_form() -> None:
    response = _client().get(LOGIN_PATH, params={"user": "alice", "next": "/u/alice/files/log"})

    assert 'name="next" value="/u/alice/files/log"' in response.text


@pytest.mark.parametrize(
    "hostile",
    [
        "https://evil.example/",
        "//evil.example/",
        "/\\evil.example/",
        "/u/alice/\\..\\evil",
        "/u/alice/%2f%2fevil.example",
        "/u/alice/%5Cevil",
        "u/alice/",
        "/u/alice/\r\nSet-Cookie: x=y",
    ],
)
def test_an_off_origin_next_is_discarded(hostile: str) -> None:
    """Anything that could leave this origin is replaced, never repaired."""
    response = _client().get(LOGIN_PATH, params={"user": "alice", "next": hostile})

    assert response.status_code == 200
    assert 'name="next" value="/u/alice/"' in response.text


@pytest.mark.parametrize("params", [{}, {"user": ""}, {"user": ["alice", "bob"]}])
def test_a_link_without_exactly_one_user_is_refused(params: dict[str, object]) -> None:
    """Never a 422: a malformed link gets this route's own refusal."""
    response = _client().get(LOGIN_PATH, params=params)

    assert response.status_code == 400


@pytest.mark.parametrize("params", [{}, {"user": ""}])
def test_a_browser_that_arrives_without_a_user_is_sent_to_the_landing_page(
    params: dict[str, object],
) -> None:
    """A person cannot act on a JSON body; the roster cards are where they belong."""
    response = _client().get(
        LOGIN_PATH, params=params, headers=BROWSER_ACCEPT, follow_redirects=False
    )

    assert response.status_code == 302
    # Origin-relative, matching what logout answers with: nginx serves the
    # landing page at exactly this path in front of the sidecar.
    assert response.headers["location"] == "/"
    assert response.headers["cache-control"] == "no-store"


def test_a_browser_link_naming_two_users_is_refused_rather_than_redirected() -> None:
    """Ambiguous is not empty: a landing page here would be a guess between them."""
    response = _client().get(
        LOGIN_PATH,
        params={"user": ["alice", "bob"]},
        headers=BROWSER_ACCEPT,
        follow_redirects=False,
    )

    assert response.status_code == 400


@pytest.mark.parametrize("params", [{}, {"user": ["alice", "bob"]}])
def test_a_client_that_did_not_ask_for_html_gets_the_unchanged_refusal(
    params: dict[str, object],
) -> None:
    """The JSON 400 is byte-identical to what this route answered before."""
    asked_for_json = _client().get(
        LOGIN_PATH, params=params, headers=JSON_ACCEPT, follow_redirects=False
    )
    stated_no_preference = _client().get(
        LOGIN_PATH, params=params, headers={"Accept": "*/*"}, follow_redirects=False
    )

    assert asked_for_json.status_code == 400
    assert asked_for_json.json() == {"detail": "the login link must name exactly one user"}
    assert (stated_no_preference.status_code, stated_no_preference.content) == (
        asked_for_json.status_code,
        asked_for_json.content,
    )


def test_a_request_that_states_no_media_preference_is_not_treated_as_a_browser() -> None:
    """The HTML branch is opt-in: silence is not a navigation."""
    response = _client().get(LOGIN_PATH, headers={"Accept": ""}, follow_redirects=False)

    assert response.status_code == 400


def test_a_user_who_is_not_on_the_roster_has_no_login_page() -> None:
    """No credential is evaluated here, and the roster is public."""
    response = _client().get(LOGIN_PATH, params={"user": "mally"})

    assert response.status_code == 404


def test_the_refusals_that_are_not_pages_are_uncacheable_too() -> None:
    """The page carries no-store; so must the answers that replace it."""
    client = _client()

    for response in (
        client.get(LOGIN_PATH),
        client.get(LOGIN_PATH, params={"user": "mally"}),
        _post(client, {"password": "x"}),
        client.post(LOGIN_PATH, data={"password": "x"}, follow_redirects=False),
    ):
        assert response.status_code in (400, 404)
        assert response.headers["cache-control"] == "no-store"


# --- Nothing a client sends is assumed to be small --------------------------


def test_an_over_long_next_never_reaches_a_location_header() -> None:
    """nginx bounds what a return-to may contain, not how much of it there is.

    A return-to of request-line size becomes a ``Location`` wider than the
    proxy's buffer, which the operator meets as a 502 rather than a login.
    """
    enormous = "/u/alice/" + "a" * 8000

    page = _client().get(LOGIN_PATH, params={"user": "alice", "next": enormous})
    redirect = _login(_client(), next_value=enormous)

    assert 'name="next" value="/u/alice/"' in page.text
    assert redirect.status_code == 303
    assert redirect.headers["location"] == "/u/alice/"


def test_a_next_at_the_cap_is_still_honoured() -> None:
    """The bound is generous, not a second allowlist: a real deep link survives."""
    deep = "/u/alice/" + "a" * (MAX_RETURN_TO_LENGTH - len("/u/alice/"))

    response = _client().get(LOGIN_PATH, params={"user": "alice", "next": deep})

    assert f'name="next" value="{deep}"' in response.text


def test_an_over_long_user_does_not_enlarge_the_refusal_page() -> None:
    """The response's size is not an anonymous, pre-credential caller's to choose."""
    client = _client()
    ordinary = _login(client, user="mally", password="x")

    huge = _login(client, user="m" * 50_000, password="x")

    assert huge.status_code == ordinary.status_code == 401
    assert DENIAL_MESSAGE in huge.text
    # The name is echoed in a few places (title, prompt, hidden field, default
    # return-to), so the page grows by a small multiple of the cap — never by the
    # 50 KB the caller sent.
    assert "m" * (MAX_USERNAME_LENGTH + 1) not in huge.text
    assert len(huge.text) < len(ordinary.text) + 8 * MAX_USERNAME_LENGTH


def test_an_over_long_user_is_never_looked_up_as_a_roster_user() -> None:
    """Truncating for display must not turn into truncating for authorisation.

    A roster name is at most the cap, so a *prefix* of an over-long submission
    could match one exactly — and the browser would have unlocked a user it never
    named.
    """
    env = {
        **PASSWORD_ENV,
        "OSPREY_AUTH_USERS": "a" * MAX_USERNAME_LENGTH,
        f"OSPREY_AUTH_PW_HASH_{'A' * MAX_USERNAME_LENGTH}": ALICE_HASH,
    }
    client = _client(env)

    response = _login(client, user="a" * (MAX_USERNAME_LENGTH + 1), password=ALICE_PASSWORD)

    assert response.status_code == 401
    assert "set-cookie" not in response.headers


def test_an_over_long_user_has_no_login_page() -> None:
    response = _client().get(LOGIN_PATH, params={"user": "m" * 50_000})

    assert response.status_code == 400
    assert len(response.text) < 1000


# --- OIDC dispatch ----------------------------------------------------------


def test_the_oidc_hand_off_names_the_route_that_exists() -> None:
    """The login module spells the OIDC path out rather than importing it, so
    that a deployment with no OIDC in it does not import Authlib. This is the
    check that keeps the two copies honest."""
    from osprey.services.auth_sidecar.routes import oidc

    assert OIDC_LOGIN_PATH == oidc.LOGIN_PATH


def test_oidc_deployments_are_sent_to_the_oidc_login_route() -> None:
    """nginx's 401 handler sends every method's browsers here; only one can answer.

    Without this hand-off an OIDC deployment would show a password form that no
    password could ever satisfy.
    """
    response = _client(OIDC_ENV).get(
        LOGIN_PATH, params={"user": "alice", "next": "/u/alice/files"}, follow_redirects=False
    )

    assert response.status_code == 302
    assert response.headers["location"] == f"{OIDC_LOGIN_PATH}?user=alice&next=%2Fu%2Falice%2Ffiles"
    assert "Enter password for" not in response.text


def test_oidc_dispatch_validates_the_return_to_before_forwarding_it() -> None:
    response = _client(OIDC_ENV).get(
        LOGIN_PATH,
        params={"user": "alice", "next": "https://evil.example/"},
        follow_redirects=False,
    )

    assert response.headers["location"] == f"{OIDC_LOGIN_PATH}?user=alice&next=%2Fu%2Falice%2F"


def test_the_oidc_hand_off_bounds_the_location_it_emits() -> None:
    """Percent-encoding triples a path of separators, so a return-to inside its
    own cap can still become a header nginx will not carry."""
    separators = "/" * (MAX_RETURN_TO_LENGTH - 1)

    response = _client(OIDC_ENV).get(
        LOGIN_PATH, params={"user": "alice", "next": f"/{separators}"}, follow_redirects=False
    )

    assert response.status_code == 302
    assert len(response.headers["location"]) <= MAX_LOCATION_LENGTH
    assert response.headers["location"].endswith("next=%2Fu%2Falice%2F")


def test_oidc_deployments_serve_no_password_form() -> None:
    """A POST under OIDC is not a refusal of a credential — there is no form."""
    response = _login(_client(OIDC_ENV))

    assert response.status_code == 404


# --- Successful login -------------------------------------------------------


def test_login_unlocks_the_user_and_redirects_to_the_validated_target() -> None:
    client = _client()
    before = time.time()

    response = _login(client, next_value="/u/alice/files")

    assert response.status_code == 303
    assert response.headers["location"] == "/u/alice/files"

    state = _codec().decode(_issued_cookie(response))
    entry = state.entry("alice")
    assert entry is not None
    assert entry.generation_tag == generation_tag(ALICE_HASH)
    assert before + SESSION_LIFETIME <= entry.expires_at <= time.time() + SESSION_LIFETIME


def test_login_carries_the_username_verbatim_and_verify_accepts_it() -> None:
    """No normalisation anywhere in the chain: verify exact-matches the roster."""
    env = {
        **PASSWORD_ENV,
        "OSPREY_AUTH_USERS": "Ann",
        "OSPREY_AUTH_PW_HASH_ANN": ALICE_HASH,
    }
    client = _client(env)

    response = _login(client, user="Ann")
    assert response.status_code == 303

    state = _codec().decode(_issued_cookie(response))
    assert [user.username for user in state.users] == ["Ann"]
    # The cookie the login issued authorises exactly the user it named — and the
    # differently-cased name is a different user, which is why the entry may not
    # be folded.
    assert client.get("/verify", params={"user": "Ann"}).status_code == 200
    assert client.get("/verify", params={"user": "ann"}).status_code == 401


def test_an_off_origin_target_in_the_form_is_discarded_too() -> None:
    """The hidden field is client input like any other, so it is re-validated."""
    response = _login(_client(), next_value="https://evil.example/")

    assert response.status_code == 303
    assert response.headers["location"] == "/u/alice/"


def test_login_leaves_other_unlocked_users_alone() -> None:
    """One browser may hold several terminals; a second login is not a reset."""
    codec = _codec()
    existing = SessionState.new(codec.now()).with_user(
        "bob", expires_at=codec.now() + SESSION_LIFETIME, generation_tag="bobs-tag"
    )
    client = _client()
    client.cookies.set(SESSION_COOKIE_NAME, codec.encode(existing))

    response = _login(client)

    state = codec.decode(_issued_cookie(response))
    assert sorted(user.username for user in state.users) == ["alice", "bob"]
    assert state.session_id == existing.session_id


def test_an_unreadable_cookie_is_replaced_by_a_fresh_session() -> None:
    """A cookie that carries no authorisation is not an error on the way in."""
    client = _client()
    client.cookies.set(SESSION_COOKIE_NAME, "not-a-signed-payload")

    response = _login(client)

    assert response.status_code == 303
    state = _codec().decode(_issued_cookie(response))
    assert state.unlocked_usernames(state.issued_at) == ("alice",)


def test_the_session_cookie_matches_the_oidc_routes_attributes() -> None:
    """A divergence here would mean login behaved differently per method.

    ``Path=/`` because the cookie must reach the verify subrequest nginx issues
    from ``/u/<user>/``; no ``Max-Age``, so the browser holds it for the session
    and the signed expiry inside it stays the only authority on its lifetime.
    """
    response = _login(_client())

    header = response.headers["set-cookie"]
    assert header.startswith(f"{SESSION_COOKIE_NAME}=")
    attributes = _cookie_attributes(header)
    assert attributes["path"] == "/"
    assert attributes["samesite"] == "lax"
    assert "httponly" in attributes
    assert "secure" in attributes
    assert "max-age" not in attributes
    assert "expires" not in attributes


def test_the_session_cookie_drops_secure_on_a_cleartext_deployment() -> None:
    """A Secure cookie over http is silently discarded — every login would
    appear to succeed and unlock nothing."""
    env = {**PASSWORD_ENV, "OSPREY_AUTH_TLS_ENABLED": "false"}

    response = _login(_client(env))

    assert "secure" not in _cookie_attributes(response.headers["set-cookie"])


def test_a_multipart_form_is_parsed_like_any_other() -> None:
    """The regression guard for the form-parsing dependency.

    Starlette needs ``python-multipart`` for every form body, so a build that
    lost it would serve a login page whose form can never be submitted — with a
    green unit suite behind it if nothing exercised a real POST.
    """
    response = _post(
        _client(),
        {"user": "alice", "next": "/u/alice/", "password": ALICE_PASSWORD},
        files={"unused": ("empty.txt", b"")},
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/u/alice/"


def test_a_revoked_session_id_is_not_carried_into_the_new_cookie() -> None:
    """Otherwise a logout/login race locks the browser out of what it just unlocked.

    Logout revokes by session id and verify refuses every cookie carrying a
    revoked one, so a login that inherited the id would hand back a freshly
    signed cookie that verify rejects — and the next login would inherit it
    again, with nothing in either response saying why.
    """
    app = create_app(PASSWORD_ENV)
    client = TestClient(app, base_url="https://testserver")
    codec = _codec()
    retired = SessionState.new(codec.now())
    app.state.revocation_store.revoke(retired.session_id, codec.now() + SESSION_LIFETIME)
    client.cookies.set(SESSION_COOKIE_NAME, codec.encode(retired))

    response = _login(client)

    assert response.status_code == 303
    assert codec.decode(_issued_cookie(response)).session_id != retired.session_id
    # The proof that matters: the terminal this login was for actually opens.
    assert client.get("/verify", params={"user": "alice"}).status_code == 200


# --- Cross-site submissions --------------------------------------------------


def test_a_form_submitted_from_another_origin_is_not_a_login() -> None:
    """``SameSite=Lax`` does not stop a browser storing the cookie a top-level
    cross-site POST comes back with, so an attacker page could otherwise log an
    operator into the attacker's roster user — and because a login adds rather
    than replaces, the operator would keep their own terminals and have little
    reason to notice which one they are typing into."""
    client, throttle, _ = _throttled_client()

    response = _post(
        client,
        {"user": "alice", "next": "", "password": ALICE_PASSWORD},
        headers={"Origin": "https://evil.example"},
    )

    assert response.status_code == 400
    assert "set-cookie" not in response.headers
    assert SESSION_COOKIE_NAME not in client.cookies
    # Refused before the throttle: a cross-site page must not be able to burn an
    # operator's window either.
    assert throttle.retry_after("alice") == 0.0


@pytest.mark.parametrize("origin", [EXTERNAL_ORIGIN, EXTERNAL_ORIGIN.upper() + "/"])
def test_the_deployments_declared_origin_is_accepted(origin: str) -> None:
    """Case and a trailing slash do not make it a different site."""
    response = _post(
        _client(),
        {"user": "alice", "next": "", "password": ALICE_PASSWORD},
        headers={"Origin": origin},
    )

    assert response.status_code == 303


def test_the_declared_origin_outranks_the_host_the_request_arrived_on() -> None:
    """``external_origin`` is the deployment's own word about where it lives;
    ``Host`` is the client's. Where the two disagree, the client does not win."""
    response = _post(
        _client(),
        {"user": "alice", "next": "", "password": ALICE_PASSWORD},
        headers={"Origin": "https://testserver"},
    )

    assert response.status_code == 400


def test_a_deployment_with_no_declared_origin_falls_back_to_the_host() -> None:
    """Password mode is not required to declare one, and must still log people in.

    The fallback is weaker — ``Host`` is client-supplied — but not weak against
    this attack: a cross-site form makes the browser derive ``Host`` from the URL
    it posts *to* while ``Origin`` names the page it posts *from*, and an
    attacker chooses neither.
    """
    env = {key: value for key, value in PASSWORD_ENV.items() if "EXTERNAL_ORIGIN" not in key}
    client = _client(env)

    accepted = _post(
        client,
        {"user": "alice", "next": "", "password": ALICE_PASSWORD},
        headers={"Origin": "https://testserver"},
    )
    refused = _post(
        client,
        {"user": "alice", "next": "", "password": ALICE_PASSWORD},
        headers={"Origin": "https://evil.example"},
    )

    assert accepted.status_code == 303
    assert refused.status_code == 400


def test_the_scheme_comes_from_the_proxys_forwarded_header() -> None:
    """nginx terminates TLS, so the sidecar's own scheme is not the browser's."""
    env = {key: value for key, value in PASSWORD_ENV.items() if "EXTERNAL_ORIGIN" not in key}
    client = TestClient(create_app(env), base_url="http://terminals.example.org")

    response = _post(
        client,
        {"user": "alice", "next": "", "password": ALICE_PASSWORD},
        headers={"Origin": "https://terminals.example.org", "X-Forwarded-Proto": "https"},
    )

    assert response.status_code == 303


def test_a_post_that_declares_no_origin_at_all_is_refused() -> None:
    """Fail closed. Every browser this page can be used from sends one, so the
    only callers a permissive branch would serve are scripted clients — which
    have no victim's cookie jar for a minted session to act on."""
    client = _client()

    response = client.post(
        LOGIN_PATH,
        data={"user": "alice", "next": "", "password": ALICE_PASSWORD},
        follow_redirects=False,
    )

    assert response.status_code == 400
    assert "set-cookie" not in response.headers


def test_a_null_origin_is_not_an_origin() -> None:
    """Sandboxed iframes and some redirect chains send the literal string."""
    response = _post(
        _client(),
        {"user": "alice", "next": "", "password": ALICE_PASSWORD},
        headers={"Origin": "null"},
    )

    assert response.status_code == 400


@pytest.mark.parametrize(
    ("referer", "expected"),
    [
        (f"{EXTERNAL_ORIGIN}/auth/login?user=alice", 303),
        ("https://evil.example/attack.html", 400),
        ("not-a-url", 400),
    ],
)
def test_referer_stands_in_where_a_browser_omits_the_origin(referer: str, expected: int) -> None:
    """Older browsers omitted ``Origin`` on same-origin form posts. Only the
    referrer's origin is read; its path is neither needed nor trusted."""
    client = _client()

    response = client.post(
        LOGIN_PATH,
        data={"user": "alice", "next": "", "password": ALICE_PASSWORD},
        headers={"Referer": referer},
        follow_redirects=False,
    )

    assert response.status_code == expected


# --- Refusals ---------------------------------------------------------------


def test_a_wrong_password_is_refused_without_a_session() -> None:
    client = _client()

    response = _login(client, password="not-the-password")

    assert response.status_code == 401
    assert DENIAL_MESSAGE in response.text
    assert "set-cookie" not in response.headers
    assert SESSION_COOKIE_NAME not in client.cookies


def test_an_empty_password_is_refused_like_any_other() -> None:
    response = _login(_client(), password="")

    assert response.status_code == 401
    assert DENIAL_MESSAGE in response.text


def test_every_refusal_is_the_same_refusal() -> None:
    """Wrong password, no provisioned credential and no roster entry are one answer.

    The three responses are compared byte for byte once each one's echoed
    username is substituted — the only thing that may differ between them is the
    caller's own input coming back.
    """
    client = _client()
    wrong = _login(client, user="alice", password="not-the-password")
    uncredentialed = _login(client, user="carol", password="anything")
    unknown = _login(client, user="mally", password="anything")

    assert wrong.status_code == uncredentialed.status_code == unknown.status_code == 401
    for other, name in ((uncredentialed, "carol"), (unknown, "mally")):
        assert other.text.replace(name, "alice") == wrong.text
        assert other.headers["content-type"] == wrong.headers["content-type"]
        assert other.headers["cache-control"] == wrong.headers["cache-control"]
        assert "set-cookie" not in other.headers


def test_the_throttle_fires_for_a_user_who_is_not_on_the_roster() -> None:
    """A non-roster username must not be the cheap way past the throttle."""
    client, _, _ = _throttled_client()

    first = _login(client, user="mally", password="anything")
    second = _login(client, user="mally", password="anything")

    assert first.status_code == 401
    assert second.status_code == 429
    assert int(second.headers["retry-after"]) >= 1


def test_a_throttled_attempt_is_refused_before_the_credential_is_evaluated() -> None:
    """The correct password inside an open window is still a 429, not a login."""
    client, _, _ = _throttled_client()
    assert _login(client, password="not-the-password").status_code == 401

    response = _login(client, password=ALICE_PASSWORD)

    assert response.status_code == 429
    assert THROTTLE_MESSAGE in response.text
    assert int(response.headers["retry-after"]) >= 1
    assert "set-cookie" not in response.headers
    assert SESSION_COOKIE_NAME not in client.cookies


def test_a_throttled_attempt_does_not_grow_the_window() -> None:
    """Otherwise a flood of parallel guesses would ratchet the window against
    the operator the throttle exists to protect."""
    client, throttle, _ = _throttled_client()

    _login(client, password="not-the-password")
    opened = throttle.retry_after("alice")
    assert _login(client, password="not-the-password").status_code == 429

    assert throttle.retry_after("alice") <= opened


def test_the_window_lifts_and_the_credential_is_evaluated_again() -> None:
    """The refusal delays an operator; it never latches into a lockout."""
    client, throttle, clock = _throttled_client()
    _login(client, password="not-the-password")
    assert _login(client, password=ALICE_PASSWORD).status_code == 429

    clock.advance(throttle.max_delay + 1)

    assert _login(client, password=ALICE_PASSWORD).status_code == 303


def test_a_successful_login_clears_the_window() -> None:
    """An operator who mistypes once pays a second, not a minute."""
    client, throttle, clock = _throttled_client()

    _login(client, password="not-the-password")
    clock.advance(throttle.max_delay + 1)
    assert _login(client, password=ALICE_PASSWORD).status_code == 303

    assert throttle.retry_after("alice") == 0.0


@pytest.mark.parametrize(
    "form",
    [
        {"password": ALICE_PASSWORD},
        {"user": "", "password": ALICE_PASSWORD},
        {"user": ["alice", "bob"], "password": ALICE_PASSWORD},
    ],
)
def test_a_form_without_exactly_one_user_is_refused(form: dict[str, object]) -> None:
    """Never a 422, and never resolved to whichever value a parser picks first."""
    response = _post(_client(), form)

    assert response.status_code == 400
    assert "set-cookie" not in response.headers


@pytest.mark.parametrize(
    "form",
    [{"password": ALICE_PASSWORD}, {"user": "", "password": ALICE_PASSWORD}],
)
def test_a_browser_posting_a_form_without_a_user_is_sent_to_the_landing_page(
    form: dict[str, object],
) -> None:
    """No user means no credential to evaluate — the roster page, not a JSON body."""
    response = _post(_client(), form, headers=BROWSER_ACCEPT)

    assert response.status_code == 302
    assert response.headers["location"] == "/"
    assert response.headers["cache-control"] == "no-store"
    # Nothing was evaluated, so nothing may be unlocked.
    assert "set-cookie" not in response.headers


def test_a_browser_posting_two_users_is_refused_rather_than_redirected() -> None:
    """The form is ambiguous, and this route resolves neither name."""
    response = _post(
        _client(), {"user": ["alice", "bob"], "password": ALICE_PASSWORD}, headers=BROWSER_ACCEPT
    )

    assert response.status_code == 400


@pytest.mark.parametrize(
    "form",
    [{"password": ALICE_PASSWORD}, {"user": ["alice", "bob"], "password": ALICE_PASSWORD}],
)
def test_a_json_client_posting_without_exactly_one_user_gets_the_unchanged_refusal(
    form: dict[str, object],
) -> None:
    """The JSON 400 is byte-identical to what this route answered before."""
    asked_for_json = _post(_client(), form, headers=JSON_ACCEPT)
    stated_no_preference = _post(_client(), form, headers={"Accept": "*/*"})

    assert asked_for_json.status_code == 400
    assert asked_for_json.json() == {"detail": "the login form must name exactly one user"}
    assert (stated_no_preference.status_code, stated_no_preference.content) == (
        asked_for_json.status_code,
        asked_for_json.content,
    )


def test_a_cross_site_post_is_still_refused_when_the_browser_asks_for_html() -> None:
    """The origin check runs first: a hostile form cannot be answered with a redirect."""
    response = _post(
        _client(),
        {"password": ALICE_PASSWORD},
        headers={**BROWSER_ACCEPT, "Origin": "https://evil.example"},
    )

    assert response.status_code == 400


# --- Logging ----------------------------------------------------------------


def test_a_successful_login_is_logged_and_the_password_is_not(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The positive assertion comes first, so this cannot pass vacuously."""
    caplog.set_level(logging.DEBUG)

    assert _login(_client()).status_code == 303

    assert "password login succeeded for 'alice'" in caplog.text
    assert ALICE_PASSWORD not in caplog.text


def test_a_refusal_names_the_user_but_never_the_credential(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)

    assert _login(_client(), password="hunter2-and-a-secret").status_code == 401

    assert "login refused for 'alice'" in caplog.text
    assert "hunter2-and-a-secret" not in caplog.text


def test_a_username_carrying_newlines_cannot_forge_a_log_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Every caller-supplied value reaching a log line is ``%r``-formatted."""
    caplog.set_level(logging.DEBUG)

    assert _login(_client(), user="mally\nWARNING forged", password="x").status_code == 401

    assert "'mally\\nWARNING forged'" in caplog.text
    assert "\nWARNING forged" not in caplog.text
