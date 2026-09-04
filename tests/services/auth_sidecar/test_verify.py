"""Tests for the auth sidecar's ``auth_request`` target.

The endpoint answers one question — may this browser reach ``/u/<user>/`` right
now — and everything pinned here is about it answering "no" often enough. The
acceptance case is the isolation the whole feature exists for: one browser's
valid cookie for alice authorizes alice and *only* alice, on the same request
otherwise unchanged.

Cookies are minted here with a client-side
:class:`~osprey.services.auth_sidecar.sessions.SessionCodec` rather than by
driving the (not yet landed) login route, so these tests exercise the verify
path alone. The app is always built from an explicit env mapping, so a variable
left in the real process environment cannot make one pass.
"""

from __future__ import annotations

import logging

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.services.auth_sidecar.app import create_app
from osprey.services.auth_sidecar.passwords import generation_tag, hash_password
from osprey.services.auth_sidecar.sessions import (
    SESSION_COOKIE_NAME,
    SessionCodec,
    SessionState,
    UnlockedUser,
)

SESSION_SECRET = "session-secret-value"
STATE_SECRET = "state-secret-value"
SESSION_LIFETIME = 3600

ALICE_HASH = hash_password("alice-password")
ROTATED_ALICE_HASH = hash_password("alice-new-password")

PASSWORD_ENV = {
    "OSPREY_AUTH_METHOD": "password",
    "OSPREY_AUTH_SESSION_SECRET": SESSION_SECRET,
    "OSPREY_AUTH_SESSION_LIFETIME": str(SESSION_LIFETIME),
    "OSPREY_AUTH_USERS": "alice,bob",
    "OSPREY_AUTH_PW_HASH_ALICE": ALICE_HASH,
    "OSPREY_AUTH_EXTERNAL_ORIGIN": "https://terminals.example.org",
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
    "OSPREY_AUTH_OIDC_CLIENT_SECRET": "client-secret",
    "OSPREY_AUTH_EXTERNAL_ORIGIN": "https://terminals.example.org",
    "OSPREY_AUTH_TLS_ENABLED": "true",
}

SESSION_ID = "test-session-id"


def _mint(
    *entries: UnlockedUser,
    secret: str = SESSION_SECRET,
    session_id: str = SESSION_ID,
    issued_at: float | None = None,
) -> str:
    """Sign a session cookie the way the login route will.

    Args:
        entries: The unlocked-user entries the cookie carries.
        secret: Signing secret. A different one stands in for a forged cookie.
        session_id: The session id, which logout revokes by.
        issued_at: Stamp for the age check; defaults to now.
    """
    codec = SessionCodec(secret)
    now = codec.now()
    state = SessionState(
        session_id=session_id,
        issued_at=now if issued_at is None else issued_at,
        users=entries,
    )
    return codec.encode(state)


def _unlocked(
    username: str,
    *,
    stored: str | None = ALICE_HASH,
    ttl: float = 600.0,
    subject: str = "",
    opener: str = "",
    admitted_identity: str = "",
) -> UnlockedUser:
    """An unlocked-user entry expiring ``ttl`` seconds from now.

    Args:
        username: The roster user the entry unlocks.
        stored: The stored hash the generation tag is derived from, or ``None``
            for an OIDC entry, which carries no tag.
        ttl: Seconds until the entry lapses; negative for an expired one.
        subject: The OIDC subject the entry carries, or ``""`` for a password
            entry and any session minted before the subject was carried.
        opener: The roster user whose credential proved the login, for an entry
            on a shared card, or ``""`` for an own-card entry.
        admitted_identity: The identity a card principal admitted, for an entry
            belonging to somebody the roster does not name, or ``""`` for an
            own-card or roster login. A roster entry never carries both this
            and an opener.
    """
    return UnlockedUser(
        username=username,
        expires_at=SessionCodec(SESSION_SECRET).now() + ttl,
        generation_tag="" if stored is None else generation_tag(stored),
        oidc_subject=subject,
        opener=opener,
        admitted_identity=admitted_identity,
    )


def _app(env: dict[str, str] | None = None) -> FastAPI:
    """A sidecar app built from an explicit environment."""
    return create_app(PASSWORD_ENV if env is None else env)


def _verify(
    client: TestClient,
    user: str | list[str] | None,
    cookie: str | None = None,
    *,
    method: str = "GET",
) -> httpx.Response:
    """Issue one subrequest the way nginx's internal auth location does.

    The cookie goes in as a raw header rather than through the client's jar, so
    each request carries exactly the cookie the test named and nothing carries
    over between them.

    Args:
        client: The test client.
        user: The ``user`` query parameter — a list repeats it, ``None`` omits
            it entirely.
        cookie: The session cookie value, if the request carries one.
        method: HTTP method, so the non-GET refusals can be exercised.
    """
    headers = {} if cookie is None else {"Cookie": f"{SESSION_COOKIE_NAME}={cookie}"}
    params = {} if user is None else {"user": user}
    return client.request(method, "/verify", params=params, headers=headers)


class TestRegistration:
    """The factory picks the route up with no edit of its own."""

    def test_verify_is_registered_by_the_factory(self) -> None:
        assert "/verify" in _app().openapi()["paths"]


class TestAcceptance:
    """The isolation guarantee, on one cookie."""

    def test_alice_cookie_authorizes_alice_and_not_bob(self) -> None:
        cookie = _mint(_unlocked("alice"))
        with TestClient(_app()) as client:
            assert _verify(client, "alice", cookie).status_code == 200
            assert _verify(client, "bob", cookie).status_code == 401

    def test_authorization_carries_no_body(self) -> None:
        """nginx reads only the status; anything else is surface for nothing."""
        cookie = _mint(_unlocked("alice"))
        with TestClient(_app()) as client:
            assert _verify(client, "alice", cookie).content == b""


class TestDenials:
    """Every way in must be closed, and closed identically."""

    def test_missing_cookie_is_denied(self) -> None:
        with TestClient(_app()) as client:
            assert _verify(client, "alice").status_code == 401

    def test_tampered_cookie_is_denied(self) -> None:
        """A cookie signed with another secret is a forgery, not a session."""
        forged = _mint(_unlocked("alice"), secret="not-the-session-secret")
        with TestClient(_app()) as client:
            assert _verify(client, "alice", forged).status_code == 401

    def test_truncated_cookie_is_denied(self) -> None:
        cookie = _mint(_unlocked("alice"))
        with TestClient(_app()) as client:
            assert _verify(client, "alice", cookie[:-4]).status_code == 401

    def test_expired_entry_is_denied(self) -> None:
        cookie = _mint(_unlocked("alice", ttl=-1.0))
        with TestClient(_app()) as client:
            assert _verify(client, "alice", cookie).status_code == 401

    def test_over_age_cookie_is_denied(self) -> None:
        """Past ``session_lifetime`` the whole cookie lapses, entries or not."""
        codec = SessionCodec(SESSION_SECRET)
        cookie = _mint(_unlocked("alice"), issued_at=codec.now() - SESSION_LIFETIME - 60)
        with TestClient(_app()) as client:
            assert _verify(client, "alice", cookie).status_code == 401

    def test_revoked_session_is_denied(self) -> None:
        """Logout kills a cookie that is otherwise entirely valid."""
        entry = _unlocked("alice")
        cookie = _mint(entry)
        app = _app()
        with TestClient(app) as client:
            assert _verify(client, "alice", cookie).status_code == 200
            app.state.revocation_store.revoke(SESSION_ID, entry.expires_at)
            assert _verify(client, "alice", cookie).status_code == 401

    def test_rotated_password_is_denied(self) -> None:
        """The generation tag is what makes ``osprey users passwd`` take effect."""
        cookie = _mint(_unlocked("alice"))
        rotated = {**PASSWORD_ENV, "OSPREY_AUTH_PW_HASH_ALICE": ROTATED_ALICE_HASH}
        with TestClient(_app(rotated)) as client:
            assert _verify(client, "alice", cookie).status_code == 401

    def test_password_mode_entry_without_a_tag_is_denied(self) -> None:
        """An OIDC-shaped entry must not authorize a password deployment."""
        cookie = _mint(_unlocked("alice", stored=None))
        with TestClient(_app()) as client:
            assert _verify(client, "alice", cookie).status_code == 401

    def test_roster_user_without_a_credential_is_denied(self) -> None:
        """An individual 401 — the deployment still serves everyone else."""
        cookie = _mint(_unlocked("bob"))
        with TestClient(_app()) as client:
            assert _verify(client, "bob", cookie).status_code == 401
            assert _verify(client, "alice", _mint(_unlocked("alice"))).status_code == 200

    def test_non_roster_user_is_denied(self) -> None:
        """Even holding a cookie that claims to unlock them."""
        cookie = _mint(_unlocked("carol"))
        with TestClient(_app()) as client:
            assert _verify(client, "carol", cookie).status_code == 401

    def test_missing_user_parameter_is_denied(self) -> None:
        """Never a 200, and never FastAPI's 422 validation body either."""
        cookie = _mint(_unlocked("alice"))
        with TestClient(_app()) as client:
            response = _verify(client, None, cookie)
        assert response.status_code == 401
        assert response.content == b""

    def test_empty_user_parameter_is_denied(self) -> None:
        cookie = _mint(_unlocked("alice"))
        with TestClient(_app()) as client:
            assert _verify(client, "", cookie).status_code == 401

    @pytest.mark.parametrize("users", [["bob", "alice"], ["alice", "bob"]])
    def test_a_repeated_user_parameter_is_denied(self, users: list[str]) -> None:
        """Both orderings, so nothing rides on which value a parser picks.

        nginx's exact-match internal locations cannot produce this query today.
        The point is that if some future seam ever does, the endpoint refuses
        instead of resolving the ambiguity in whichever direction is convenient
        — including the direction that would authorize alice's valid cookie
        against a request naming bob.
        """
        cookie = _mint(_unlocked("alice"))
        with TestClient(_app()) as client:
            # The single-parameter request proves the cookie is good, so the
            # refusal below is the repetition and not some unrelated denial.
            assert _verify(client, "alice", cookie).status_code == 200
            assert _verify(client, users, cookie).status_code == 401

    @pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
    def test_only_get_is_served(self, method: str) -> None:
        """nginx's auth subrequest is a GET; nothing else may authorize.

        FastAPI does not add HEAD to a GET route the way bare Starlette does,
        so even the body-less twin of the served method is refused here.
        """
        cookie = _mint(_unlocked("alice"))
        with TestClient(_app()) as client:
            response = _verify(client, "alice", cookie, method=method)
        assert response.status_code >= 400

    @pytest.mark.parametrize(
        ("user", "cookie"),
        [
            ("carol", None),
            ("alice", None),
            ("alice", "forged"),
        ],
    )
    def test_denials_are_indistinguishable(self, user: str, cookie: str | None) -> None:
        """An unknown user and a bad cookie must look the same to the client.

        Same status, same empty body, and no ``WWW-Authenticate`` to hint that
        a credential would help here — the login flow lives at nginx's
        ``error_page``, not behind a challenge header on the subrequest.
        """
        value = None if cookie is None else _mint(_unlocked("alice"), secret=cookie)
        with TestClient(_app()) as client:
            response = _verify(client, user, value)
        assert response.status_code == 401
        assert response.content == b""
        assert "www-authenticate" not in response.headers


class TestOidcMode:
    """OIDC sessions carry no generation tag, and must not be asked for one."""

    def test_entry_without_a_tag_authorizes(self) -> None:
        cookie = _mint(_unlocked("alice", stored=None))
        with TestClient(_app(OIDC_ENV)) as client:
            assert _verify(client, "alice", cookie).status_code == 200

    def test_a_non_empty_tag_is_ignored(self) -> None:
        """Informational, and deliberate: OIDC never consults the tag.

        There is no stored hash in this mode to compare a tag against, and a
        cookie carrying one at all could only have been minted by a holder of
        the signing secret — which is strictly more than the tag would prove.
        Pinned so a later reader does not mistake this for an oversight.
        """
        cookie = _mint(_unlocked("alice", stored=ROTATED_ALICE_HASH))
        with TestClient(_app(OIDC_ENV)) as client:
            assert _verify(client, "alice", cookie).status_code == 200

    def test_unmapped_roster_user_still_authorizes(self) -> None:
        """No stored hash exists in this mode, so its absence denies nothing."""
        cookie = _mint(_unlocked("bob", stored=None))
        with TestClient(_app(OIDC_ENV)) as client:
            assert _verify(client, "bob", cookie).status_code == 200

    def test_revocation_and_expiry_still_apply(self) -> None:
        expired = _mint(_unlocked("alice", stored=None, ttl=-1.0))
        entry = _unlocked("alice", stored=None)
        app = _app(OIDC_ENV)
        with TestClient(app) as client:
            assert _verify(client, "alice", expired).status_code == 401
            app.state.revocation_store.revoke(SESSION_ID, entry.expires_at)
            assert _verify(client, "alice", _mint(entry)).status_code == 401


class TestSubjectHeader:
    """An authorized request reports the account behind it, in either method."""

    SUBJECT_HEADER = "X-Osprey-Auth-Subject"
    ALICE_SUBJECT = "idp|alice"

    def test_oidc_authorization_returns_the_subject_header(self) -> None:
        cookie = _mint(_unlocked("alice", stored=None, subject=self.ALICE_SUBJECT))
        with TestClient(_app(OIDC_ENV)) as client:
            response = _verify(client, "alice", cookie)
        assert response.status_code == 200
        assert response.headers[self.SUBJECT_HEADER] == self.ALICE_SUBJECT

    def test_password_session_names_the_roster_user(self) -> None:
        """A password session has no provider account, so the roster username
        *is* the account behind it — and that is what the header names. The
        header's meaning is unchanged: present still means a known account."""
        cookie = _mint(_unlocked("alice"))
        with TestClient(_app()) as client:
            response = _verify(client, "alice", cookie)
        assert response.status_code == 200
        assert response.headers[self.SUBJECT_HEADER] == "alice"

    def test_oidc_session_without_a_subject_still_verifies_and_omits_the_header(self) -> None:
        """Backward compat: a session minted before the subject was carried
        authorizes exactly as before and simply names no account."""
        cookie = _mint(_unlocked("alice", stored=None, subject=""))
        with TestClient(_app(OIDC_ENV)) as client:
            response = _verify(client, "alice", cookie)
        assert response.status_code == 200
        assert self.SUBJECT_HEADER.lower() not in response.headers

    def test_a_denied_request_never_carries_the_subject_header(self) -> None:
        """The header rides only on the 200, never on a refusal."""
        cookie = _mint(_unlocked("alice", stored=None, subject=self.ALICE_SUBJECT))
        with TestClient(_app(OIDC_ENV)) as client:
            response = _verify(client, "bob", cookie)
        assert response.status_code == 401
        assert self.SUBJECT_HEADER.lower() not in response.headers


class TestAccountHeader:
    """An authorized request always names the roster card it is on."""

    ACCOUNT_HEADER = "X-Osprey-Auth-Account"
    SUBJECT_HEADER = "X-Osprey-Auth-Subject"
    ALICE_SUBJECT = "idp|alice"

    def test_password_authorization_names_the_roster_account(self) -> None:
        """In password mode the card and the proof are the same name, so both
        headers say ``alice`` — the account still rides on its own header
        rather than leaving a consumer to infer it from the subject."""
        cookie = _mint(_unlocked("alice"))
        with TestClient(_app()) as client:
            response = _verify(client, "alice", cookie)
        assert response.status_code == 200
        assert response.headers[self.ACCOUNT_HEADER] == "alice"

    def test_oidc_authorization_names_the_card_not_the_person(self) -> None:
        """Under OIDC the two headers answer different questions: the account is
        the roster card the request is on, the subject is whoever proved the
        login. A shared card is exactly where they diverge."""
        cookie = _mint(_unlocked("alice", stored=None, subject=self.ALICE_SUBJECT))
        with TestClient(_app(OIDC_ENV)) as client:
            response = _verify(client, "alice", cookie)
        assert response.status_code == 200
        assert response.headers[self.ACCOUNT_HEADER] == "alice"
        assert response.headers[self.SUBJECT_HEADER] == self.ALICE_SUBJECT

    def test_an_oidc_session_naming_nobody_still_names_the_account(self) -> None:
        """The account does not depend on the session holding an identity: the
        request is on a card whether or not the cookie says who opened it."""
        cookie = _mint(_unlocked("alice", stored=None, subject=""))
        with TestClient(_app(OIDC_ENV)) as client:
            response = _verify(client, "alice", cookie)
        assert response.status_code == 200
        assert response.headers[self.ACCOUNT_HEADER] == "alice"
        assert self.SUBJECT_HEADER.lower() not in response.headers

    def test_a_denied_request_never_carries_the_account_header(self) -> None:
        """The header rides only on the 200. A refusal says nothing at all, so a
        consumer can never read a card off a request that was not authorized."""
        cookie = _mint(_unlocked("alice"))
        with TestClient(_app()) as client:
            response = _verify(client, "bob", cookie)
        assert response.status_code == 401
        assert self.ACCOUNT_HEADER.lower() not in response.headers

    def test_a_non_ascii_roster_username_denies(self) -> None:
        """Defense in depth, and unreachable in a real deployment: the render
        holds roster names to ``USERNAME_CHARSET_RE``, so a username that could
        not cross the nginx boundary never reaches a rendered sidecar. The
        roster username is also the one identity value that arrives here without
        having passed through the session codec's own charset refusal, so this
        route checks it itself and denies rather than emitting a header that
        would arrive mangled — or dropping it and leaving the terminal running
        with less identity than the deployment believes it forwarded."""
        env = {**OIDC_ENV, "OSPREY_AUTH_USERS": "j\u00f6rg,bob"}
        cookie = _mint(_unlocked("j\u00f6rg", stored=None, subject=""))
        with TestClient(_app(env)) as client:
            response = _verify(client, "j\u00f6rg", cookie)
        assert response.status_code == 401
        assert self.ACCOUNT_HEADER.lower() not in response.headers


class TestSharedCardOpener:
    """A shared card's session lives and dies with what admitted it.

    A card admitting the whole roster is opened by a roster entry, and its
    session names that opener — the entry whose credential proved the login —
    which verify re-validates against the roster's current mapping on every
    subrequest. A card naming particular identities or domains is opened by
    somebody the roster need not name at all, and its session records the
    identity that was matched instead; that match is re-run against the card's
    *current* principals on every subrequest, so withdrawing a principal closes
    the card on the next request rather than at expiry.

    Every direction is pinned: a shared card refuses a session naming neither,
    a card no longer shared refuses one naming an opener, a card narrowed away
    from the roster refuses one too, and no session may name both an opener and
    an admitted identity. Editing an access rule therefore retires the sessions
    minted under the old one instead of letting them lapse.

    One case deliberately does not appear here: an admitted identity that could
    not ride an identity header never reaches this route at all, because the
    codec refuses it on the way in and again on the way out
    (``test_an_uncarryable_admitted_identity_invalidates_the_cookie`` and
    ``test_an_uncarryable_admitted_identity_is_refused_at_login`` in
    ``test_sessions.py``). Repeating it here would pin the codec's rule in a
    second place rather than this route's.
    """

    ACCOUNT_HEADER = "X-Osprey-Auth-Account"
    SUBJECT_HEADER = "X-Osprey-Auth-Subject"
    BOB_SUBJECT = "idp|bob"
    CAROL_IDENTITY = "carol@example.org"

    SHARED_ENV = {
        **OIDC_ENV,
        "OSPREY_AUTH_ROSTER_ACCESS_ALICE": "any",
        "OSPREY_AUTH_OIDC_SUBJECT_BOB": BOB_SUBJECT,
    }

    RULE_ENV = {
        **OIDC_ENV,
        "OSPREY_AUTH_OIDC_CLAIM": "email",
        "OSPREY_AUTH_ROSTER_ACCESS_ALICE": '["domain:example.org"]',
    }
    """Alice's card admits a domain and nothing else — no roster principal, so
    nobody is admitted for merely being on the roster."""

    def test_a_valid_opener_authorizes_and_names_both_identities(self) -> None:
        """The account is the card the request is on, the subject the opener's
        login — the shared card is exactly where the two headers diverge."""
        cookie = _mint(_unlocked("alice", stored=None, subject=self.BOB_SUBJECT, opener="bob"))
        with TestClient(_app(self.SHARED_ENV)) as client:
            response = _verify(client, "alice", cookie)
        assert response.status_code == 200
        assert response.headers[self.ACCOUNT_HEADER] == "alice"
        assert response.headers[self.SUBJECT_HEADER] == self.BOB_SUBJECT

    def test_an_opener_whose_mapping_was_removed_is_denied(self) -> None:
        """Taking the opener's subject out of the config closes the shared card
        immediately, not at each outstanding session's expiry."""
        cookie = _mint(_unlocked("alice", stored=None, subject=self.BOB_SUBJECT, opener="bob"))
        env = {k: v for k, v in self.SHARED_ENV.items() if k != "OSPREY_AUTH_OIDC_SUBJECT_BOB"}
        with TestClient(_app(env)) as client:
            assert _verify(client, "alice", cookie).status_code == 401

    def test_an_opener_whose_mapping_changed_is_denied(self) -> None:
        """Remapping the opener's suffix to another IdP identity is a new
        person; the sessions the old one opened stop verifying."""
        cookie = _mint(_unlocked("alice", stored=None, subject=self.BOB_SUBJECT, opener="bob"))
        env = {**self.SHARED_ENV, "OSPREY_AUTH_OIDC_SUBJECT_BOB": "idp|somebody-else"}
        with TestClient(_app(env)) as client:
            assert _verify(client, "alice", cookie).status_code == 401

    def test_an_email_opener_re_spelled_in_another_case_is_still_the_opener(self) -> None:
        """Under ``claim: email`` an operator changing only the case of the
        opener's mapping has remapped nobody — the same comparator the login
        went through keeps the card open; under ``sub`` the same edit is a new
        identity and closes it."""
        cookie = _mint(_unlocked("alice", stored=None, subject="bob@example.org", opener="bob"))
        env = {
            **self.SHARED_ENV,
            "OSPREY_AUTH_OIDC_CLAIM": "email",
            "OSPREY_AUTH_OIDC_SUBJECT_BOB": "Bob@Example.ORG",
        }
        with TestClient(_app(env)) as client:
            response = _verify(client, "alice", cookie)
        assert response.status_code == 200
        assert response.headers[self.SUBJECT_HEADER] == "bob@example.org"

        env_sub = {**self.SHARED_ENV, "OSPREY_AUTH_OIDC_SUBJECT_BOB": self.BOB_SUBJECT.upper()}
        with TestClient(_app(env_sub)) as client:
            assert _verify(client, "alice", cookie).status_code == 401

    def test_a_shared_card_session_naming_no_opener_is_denied(self) -> None:
        """Flipping a card to ``any`` must not widen sessions minted while it
        was an own card — they carry no opener to re-validate."""
        cookie = _mint(_unlocked("alice", stored=None, subject=""))
        with TestClient(_app(self.SHARED_ENV)) as client:
            assert _verify(client, "alice", cookie).status_code == 401

    def test_an_opener_on_a_card_no_longer_shared_is_denied(self) -> None:
        """Flipping ``any`` back to own retires every shared session at once."""
        cookie = _mint(_unlocked("alice", stored=None, subject=self.BOB_SUBJECT, opener="bob"))
        with TestClient(_app(OIDC_ENV)) as client:
            assert _verify(client, "alice", cookie).status_code == 401

    def test_only_the_shared_session_dies_when_the_opener_is_remapped(self) -> None:
        """The asymmetry, pinned side by side. An own card's subject is read
        off the session and never re-derived, so changing alice's configured
        subject leaves her own session verifying until it lapses — the
        documented lapse-with-the-session rule. A shared session keyed on that
        same opener dies on the next subrequest, because a shared card
        multiplies one login across the roster."""
        env = {
            **OIDC_ENV,
            "OSPREY_AUTH_OIDC_SUBJECT_ALICE": "idp|changed",
            "OSPREY_AUTH_ROSTER_ACCESS_BOB": "any",
        }
        own = _mint(_unlocked("alice", stored=None, subject="idp|alice"))
        shared = _mint(_unlocked("bob", stored=None, subject="idp|alice", opener="alice"))
        with TestClient(_app(env)) as client:
            assert _verify(client, "alice", own).status_code == 200
            assert _verify(client, "bob", shared).status_code == 401

    def test_a_rule_admitted_session_authorizes_and_names_the_admitted_identity(self) -> None:
        """The card's ``domain:`` principal admitted somebody the roster does
        not name, so there is no mapped roster entry and no opener to report —
        the subject header names the identity that was matched, and the account
        header the card it was matched against. The two diverge, and neither is
        the other: the person behind the request is reported as themselves and
        not as the roster name whose card they are on.

        Minted the way the callback mints one: a rule-admitted login stores the
        ASSERTED identity as the session's OIDC subject and again as the
        admitted identity, so both fields carry it (``oidc.py``'s
        ``proved_subject = asserted``). A fixture with one of them empty would
        pin a shape no login produces.
        """
        cookie = _mint(
            _unlocked(
                "alice",
                stored=None,
                subject=self.CAROL_IDENTITY,
                admitted_identity=self.CAROL_IDENTITY,
            )
        )
        with TestClient(_app(self.RULE_ENV)) as client:
            response = _verify(client, "alice", cookie)
        assert response.status_code == 200
        assert response.headers[self.ACCOUNT_HEADER] == "alice"
        assert response.headers[self.SUBJECT_HEADER] == self.CAROL_IDENTITY

    def test_the_admitting_principal_is_re_matched_on_every_subrequest(self) -> None:
        """Nothing about the admission is remembered, so asking twice is the
        same question twice and the second answer is not a cached first."""
        cookie = _mint(
            _unlocked(
                "alice",
                stored=None,
                subject=self.CAROL_IDENTITY,
                admitted_identity=self.CAROL_IDENTITY,
            )
        )
        with TestClient(_app(self.RULE_ENV)) as client:
            assert _verify(client, "alice", cookie).status_code == 200
            assert _verify(client, "alice", cookie).status_code == 200

    def test_withdrawing_the_covering_principal_closes_the_card(self) -> None:
        """The revocation contract: re-pointing the card at another domain
        denies the very next subrequest of a session the old one admitted."""
        cookie = _mint(
            _unlocked(
                "alice",
                stored=None,
                subject=self.CAROL_IDENTITY,
                admitted_identity=self.CAROL_IDENTITY,
            )
        )
        env = {**self.RULE_ENV, "OSPREY_AUTH_ROSTER_ACCESS_ALICE": '["domain:elsewhere.example"]'}
        with TestClient(_app(env)) as client:
            assert _verify(client, "alice", cookie).status_code == 401

    def test_a_rule_admitted_session_on_a_card_no_longer_shared_is_denied(self) -> None:
        """Removing the rule outright is the same withdrawal: an own card
        admits its own user by logging in, never by a principal."""
        cookie = _mint(
            _unlocked(
                "alice",
                stored=None,
                subject=self.CAROL_IDENTITY,
                admitted_identity=self.CAROL_IDENTITY,
            )
        )
        with TestClient(_app(OIDC_ENV)) as client:
            assert _verify(client, "alice", cookie).status_code == 401

    def test_a_roster_opener_still_authorizes_on_a_mixed_card(self) -> None:
        """A card naming both ``roster`` and a domain keeps admitting roster
        logins: the opener path is unchanged by the principal beside it."""
        env = {
            **OIDC_ENV,
            "OSPREY_AUTH_ROSTER_ACCESS_ALICE": '["domain:example.org","roster"]',
            "OSPREY_AUTH_OIDC_SUBJECT_BOB": self.BOB_SUBJECT,
        }
        cookie = _mint(_unlocked("alice", stored=None, subject=self.BOB_SUBJECT, opener="bob"))
        with TestClient(_app(env)) as client:
            response = _verify(client, "alice", cookie)
        assert response.status_code == 200
        assert response.headers[self.ACCOUNT_HEADER] == "alice"
        assert response.headers[self.SUBJECT_HEADER] == self.BOB_SUBJECT

    def test_an_opener_on_a_card_narrowed_away_from_the_roster_is_denied(self) -> None:
        """``roster`` is the only principal that admits a login for being on
        the roster, so dropping it retires the sessions it opened — the same
        rule as flipping the card back to own, one principal finer."""
        env = {**self.RULE_ENV, "OSPREY_AUTH_OIDC_SUBJECT_BOB": self.BOB_SUBJECT}
        cookie = _mint(_unlocked("alice", stored=None, subject=self.BOB_SUBJECT, opener="bob"))
        with TestClient(_app(env)) as client:
            assert _verify(client, "alice", cookie).status_code == 401

    def test_coverage_moving_between_principals_keeps_the_session_open(self) -> None:
        """Withdrawal is about coverage, not about the principal that happened
        to do the admitting. Replacing the domain member with a ``user:`` one
        naming the same identity leaves the session verifying — nothing in the
        entry remembers which member matched at login, precisely so that this
        is one question asked afresh rather than two answers to reconcile."""
        cookie = _mint(
            _unlocked(
                "alice",
                stored=None,
                subject=self.CAROL_IDENTITY,
                admitted_identity=self.CAROL_IDENTITY,
            )
        )
        moved = {
            **self.RULE_ENV,
            "OSPREY_AUTH_ROSTER_ACCESS_ALICE": f'["user:{self.CAROL_IDENTITY}"]',
        }
        with TestClient(_app(self.RULE_ENV)) as client:
            assert _verify(client, "alice", cookie).status_code == 200
        with TestClient(_app(moved)) as client:
            response = _verify(client, "alice", cookie)
        assert response.status_code == 200
        assert response.headers[self.SUBJECT_HEADER] == self.CAROL_IDENTITY

    def test_a_mixed_card_admits_an_opener_and_a_rule_admission_at_once(self) -> None:
        """Both kinds of session on one card under one set of settings, which
        is what a ``[roster, domain:…]`` card is for. The account header names
        the card either way; the subject names whoever is behind the request —
        the opener's mapped identity on the roster session, exactly the stored
        admitted identity on the other, and neither is the roster username."""
        env = {
            **OIDC_ENV,
            "OSPREY_AUTH_OIDC_CLAIM": "email",
            "OSPREY_AUTH_ROSTER_ACCESS_ALICE": '["domain:example.org","roster"]',
            "OSPREY_AUTH_OIDC_SUBJECT_BOB": "bob@example.org",
        }
        opened = _mint(_unlocked("alice", stored=None, subject="bob@example.org", opener="bob"))
        admitted = _mint(
            _unlocked(
                "alice",
                stored=None,
                subject=self.CAROL_IDENTITY,
                admitted_identity=self.CAROL_IDENTITY,
            )
        )
        with TestClient(_app(env)) as client:
            by_opener = _verify(client, "alice", opened)
            by_rule = _verify(client, "alice", admitted)
        assert by_opener.status_code == 200
        assert by_opener.headers[self.ACCOUNT_HEADER] == "alice"
        assert by_opener.headers[self.SUBJECT_HEADER] == "bob@example.org"
        assert by_rule.status_code == 200
        assert by_rule.headers[self.ACCOUNT_HEADER] == "alice"
        assert by_rule.headers[self.SUBJECT_HEADER] == self.CAROL_IDENTITY

    def test_a_session_naming_both_an_opener_and_an_admitted_identity_is_denied(self) -> None:
        """No login mints that shape. Refused whole rather than resolved in
        whichever direction happens to be checked first."""
        env = {
            **OIDC_ENV,
            "OSPREY_AUTH_ROSTER_ACCESS_ALICE": '["domain:example.org","roster"]',
            "OSPREY_AUTH_OIDC_SUBJECT_BOB": self.BOB_SUBJECT,
        }
        cookie = _mint(
            _unlocked(
                "alice",
                stored=None,
                subject=self.BOB_SUBJECT,
                opener="bob",
                admitted_identity=self.CAROL_IDENTITY,
            )
        )
        with TestClient(_app(env)) as client:
            assert _verify(client, "alice", cookie).status_code == 401

    @pytest.mark.parametrize(
        "entry_kwargs",
        [
            {"subject": CAROL_IDENTITY, "admitted_identity": CAROL_IDENTITY},
            {"subject": BOB_SUBJECT, "opener": "bob"},
            {},
        ],
        ids=["rule-admitted", "opener", "neither"],
    )
    def test_an_unreadable_access_rule_denies_every_session_on_the_card(
        self, entry_kwargs: dict[str, str]
    ) -> None:
        """A malformed ``OSPREY_AUTH_ROSTER_ACCESS_*`` admits nobody at all —
        not the identities the operator meant to name, not the roster, and not
        the card's own user. The card is visibly unusable until it is fixed,
        which is the point: a value nobody can read must not degrade into a
        working card of any width."""
        env = {
            **OIDC_ENV,
            "OSPREY_AUTH_ROSTER_ACCESS_ALICE": "domain:example.org",
            "OSPREY_AUTH_OIDC_SUBJECT_BOB": self.BOB_SUBJECT,
        }
        cookie = _mint(_unlocked("alice", stored=None, **entry_kwargs))
        with TestClient(_app(env)) as client:
            assert _verify(client, "alice", cookie).status_code == 401

    def test_a_rule_denial_looks_like_every_other_denial(self) -> None:
        """A session the card no longer admits is refused with the same bare
        401 as an unknown user: no body, no challenge header, and none of the
        identity headers — so nothing tells the caller that this card exists,
        or that the identity it holds was once admitted to it."""
        cookie = _mint(
            _unlocked(
                "alice",
                stored=None,
                subject=self.CAROL_IDENTITY,
                admitted_identity=self.CAROL_IDENTITY,
            )
        )
        env = {**self.RULE_ENV, "OSPREY_AUTH_ROSTER_ACCESS_ALICE": '["domain:elsewhere.example"]'}
        with TestClient(_app(env)) as client:
            response = _verify(client, "alice", cookie)
        assert response.status_code == 401
        assert response.content == b""
        assert "www-authenticate" not in response.headers
        assert self.ACCOUNT_HEADER.lower() not in response.headers
        assert self.SUBJECT_HEADER.lower() not in response.headers


class TestOwnerAdmissionOnASharedCard:
    """``self`` beside other principals keeps the card's own user logging in.

    A card's owner is admitted by their own credential exactly when ``self`` is
    one of the card's principals — trivially on an own card, and just as much on
    ``[self, domain:…]``, which is how an operator opens a terminal to a domain
    *without* shutting its owner out of it. A card that names only other
    principals does not admit the owner, and neither does one whose rule cannot
    be read: in both cases the owner's own session is refused like anyone
    else's, on the next subrequest.

    The owner's session is the own shape — no opener and no admitted identity,
    because neither is what let them in — so it is that shape, under both
    methods, these pin. Everything else about an own-card session is unchanged:
    the credential checked is the card's own, and there is no opener to
    re-derive.
    """

    ACCOUNT_HEADER = "X-Osprey-Auth-Account"
    SUBJECT_HEADER = "X-Osprey-Auth-Subject"
    ALICE_IDENTITY = "alice@example.org"

    def test_the_owner_of_a_self_and_domain_card_still_authorizes(self) -> None:
        """Password mode: the card's own hash still proves the card's own user,
        and the headers name that user on both — the domain principal beside
        ``self`` widens who else may come in, never who the owner is."""
        cookie = _mint(_unlocked("alice"))
        env = {**PASSWORD_ENV, "OSPREY_AUTH_ROSTER_ACCESS_ALICE": '["domain:example.org","self"]'}
        with TestClient(_app(env)) as client:
            response = _verify(client, "alice", cookie)
        assert response.status_code == 200
        assert response.headers[self.ACCOUNT_HEADER] == "alice"
        assert response.headers[self.SUBJECT_HEADER] == "alice"

    def test_the_owner_of_a_domain_only_card_is_denied(self) -> None:
        """Dropping ``self`` is a real narrowing and not a formality: the owner
        reaches this card only by being admitted like everybody else, which an
        own-shape session is not evidence of."""
        cookie = _mint(_unlocked("alice"))
        env = {**PASSWORD_ENV, "OSPREY_AUTH_ROSTER_ACCESS_ALICE": '["domain:example.org"]'}
        with TestClient(_app(env)) as client:
            assert _verify(client, "alice", cookie).status_code == 401

    def test_the_owner_of_a_self_and_domain_card_still_authorizes_under_oidc(self) -> None:
        """The same rule under the federated method, where the session names
        the owner's own provider account rather than the roster username."""
        cookie = _mint(_unlocked("alice", stored=None, subject=self.ALICE_IDENTITY))
        env = {
            **OIDC_ENV,
            "OSPREY_AUTH_OIDC_CLAIM": "email",
            "OSPREY_AUTH_ROSTER_ACCESS_ALICE": '["domain:example.org","self"]',
            "OSPREY_AUTH_OIDC_SUBJECT_ALICE": self.ALICE_IDENTITY,
        }
        with TestClient(_app(env)) as client:
            response = _verify(client, "alice", cookie)
        assert response.status_code == 200
        assert response.headers[self.ACCOUNT_HEADER] == "alice"
        assert response.headers[self.SUBJECT_HEADER] == self.ALICE_IDENTITY

    def test_the_owner_of_a_domain_only_card_is_denied_under_oidc(self) -> None:
        cookie = _mint(_unlocked("alice", stored=None, subject=self.ALICE_IDENTITY))
        env = {
            **OIDC_ENV,
            "OSPREY_AUTH_OIDC_CLAIM": "email",
            "OSPREY_AUTH_ROSTER_ACCESS_ALICE": '["domain:example.org"]',
            "OSPREY_AUTH_OIDC_SUBJECT_ALICE": self.ALICE_IDENTITY,
        }
        with TestClient(_app(env)) as client:
            assert _verify(client, "alice", cookie).status_code == 401

    def test_the_owner_denial_looks_like_every_other_denial(self) -> None:
        """No body, no challenge, no identity headers — a card that stopped
        admitting its owner tells the browser nothing it did not already know."""
        cookie = _mint(_unlocked("alice"))
        env = {**PASSWORD_ENV, "OSPREY_AUTH_ROSTER_ACCESS_ALICE": '["domain:example.org"]'}
        with TestClient(_app(env)) as client:
            response = _verify(client, "alice", cookie)
        assert response.status_code == 401
        assert response.content == b""
        assert "www-authenticate" not in response.headers
        assert self.ACCOUNT_HEADER.lower() not in response.headers
        assert self.SUBJECT_HEADER.lower() not in response.headers

    def test_a_whole_roster_card_still_refuses_an_own_shape_session(self) -> None:
        """``any`` resolves to ``[roster]`` and names no ``self``, so the owner
        reaches their own card there as a roster member — with an opener — and
        an own-shape session on it stays refused exactly as before."""
        cookie = _mint(_unlocked("alice"))
        env = {**PASSWORD_ENV, "OSPREY_AUTH_ROSTER_ACCESS_ALICE": "any"}
        with TestClient(_app(env)) as client:
            assert _verify(client, "alice", cookie).status_code == 401


class TestSharedCardPassword:
    """A password shared card lives and dies with the *opener's* stored hash.

    The login mints a shared card's generation tag from the opener's hash, so
    verify checks the tag against that hash and never against the card's own —
    which may still exist as a stale leftover from before the card was shared.
    Everything that can happen to the opener's credential (rotation, removal,
    the opener leaving the roster) therefore closes the card, and nothing that
    happens to the card's own leftover hash changes the answer either way.
    """

    ACCOUNT_HEADER = "X-Osprey-Auth-Account"
    SUBJECT_HEADER = "X-Osprey-Auth-Subject"

    SHARED_ENV = {
        **PASSWORD_ENV,
        "OSPREY_AUTH_ROSTER_ACCESS_BOB": "any",
    }
    STALE_BOB_HASH = hash_password("bob-old-password")

    def test_a_valid_opener_authorizes_and_names_both_identities(self) -> None:
        """The account is the card the request is on, the subject the opener —
        the shared card is where the two headers diverge under password too."""
        cookie = _mint(_unlocked("bob", opener="alice"))
        with TestClient(_app(self.SHARED_ENV)) as client:
            response = _verify(client, "bob", cookie)
        assert response.status_code == 200
        assert response.headers[self.ACCOUNT_HEADER] == "bob"
        assert response.headers[self.SUBJECT_HEADER] == "alice"

    def test_a_rotated_opener_password_is_denied(self) -> None:
        """Rotating the opener's password retires every shared session it
        opened, exactly as it retires the opener's own."""
        cookie = _mint(_unlocked("bob", opener="alice"))
        env = {**self.SHARED_ENV, "OSPREY_AUTH_PW_HASH_ALICE": ROTATED_ALICE_HASH}
        with TestClient(_app(env)) as client:
            assert _verify(client, "bob", cookie).status_code == 401

    def test_an_opener_whose_hash_was_removed_is_denied(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The log names the actual fault: the opener's missing credential,
        not the card's."""
        cookie = _mint(_unlocked("bob", opener="alice"))
        env = {k: v for k, v in self.SHARED_ENV.items() if k != "OSPREY_AUTH_PW_HASH_ALICE"}
        caplog.set_level(logging.DEBUG)
        with TestClient(_app(env)) as client:
            assert _verify(client, "bob", cookie).status_code == 401
        assert "the opener has no stored credential" in caplog.text

    def test_an_opener_removed_from_the_roster_is_denied(self) -> None:
        """Stored hashes are roster-scoped, so taking the opener off the
        roster closes the shared card even while its hash lingers in the env."""
        cookie = _mint(_unlocked("bob", opener="alice"))
        env = {**self.SHARED_ENV, "OSPREY_AUTH_USERS": "bob"}
        with TestClient(_app(env)) as client:
            assert _verify(client, "bob", cookie).status_code == 401

    def test_a_stale_card_hash_does_not_close_a_valid_shared_session(self) -> None:
        """The card's own leftover hash disagrees with the tag, and it must not
        matter: the tag was minted from the opener's hash and that still
        matches."""
        cookie = _mint(_unlocked("bob", opener="alice"))
        env = {**self.SHARED_ENV, "OSPREY_AUTH_PW_HASH_BOB": self.STALE_BOB_HASH}
        with TestClient(_app(env)) as client:
            assert _verify(client, "bob", cookie).status_code == 200

    def test_a_stale_card_hash_does_not_keep_a_shared_session_open(self) -> None:
        """The sharpest pin: the tag *matches* the card's own leftover hash,
        and the opener's hash is gone. An implementation consulting the card's
        hash would answer 200 here; the opener's hash is the only authority,
        so the answer is 401."""
        cookie = _mint(_unlocked("bob", stored=self.STALE_BOB_HASH, opener="alice"))
        env = {
            k: v
            for k, v in {**self.SHARED_ENV, "OSPREY_AUTH_PW_HASH_BOB": self.STALE_BOB_HASH}.items()
            if k != "OSPREY_AUTH_PW_HASH_ALICE"
        }
        with TestClient(_app(env)) as client:
            assert _verify(client, "bob", cookie).status_code == 401


class TestLogging:
    """Denials are diagnosable without disclosing anything."""

    def test_denial_logs_a_reason_but_no_secret_material(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        entry = _unlocked("alice")
        cookie = _mint(entry)
        rotated = {**PASSWORD_ENV, "OSPREY_AUTH_PW_HASH_ALICE": ROTATED_ALICE_HASH}
        caplog.set_level(logging.DEBUG)

        with TestClient(_app(rotated)) as client:
            assert _verify(client, "alice", cookie).status_code == 401

        assert "generation tag" in caplog.text
        for secret in (cookie, entry.generation_tag, ALICE_HASH, ROTATED_ALICE_HASH):
            assert secret not in caplog.text

    def test_a_forged_cookie_is_never_echoed(self, caplog: pytest.LogCaptureFixture) -> None:
        forged = _mint(_unlocked("alice"), secret="not-the-session-secret")
        caplog.set_level(logging.DEBUG)

        with TestClient(_app()) as client:
            assert _verify(client, "alice", forged).status_code == 401

        assert "session cookie rejected" in caplog.text
        assert forged not in caplog.text

    def test_authorization_logs_nothing(self, caplog: pytest.LogCaptureFixture) -> None:
        """One line per allowed request would be one line per keystroke."""
        cookie = _mint(_unlocked("alice"))
        caplog.set_level(logging.DEBUG, logger="osprey.services.auth_sidecar.routes.verify")

        with TestClient(_app()) as client:
            assert _verify(client, "alice", cookie).status_code == 200

        assert caplog.text == ""
