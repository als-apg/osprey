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
) -> UnlockedUser:
    """An unlocked-user entry expiring ``ttl`` seconds from now.

    Args:
        username: The roster user the entry unlocks.
        stored: The stored hash the generation tag is derived from, or ``None``
            for an OIDC entry, which carries no tag.
        ttl: Seconds until the entry lapses; negative for an expired one.
        subject: The OIDC subject the entry carries, or ``""`` for a password
            entry and any session minted before the subject was carried.
    """
    return UnlockedUser(
        username=username,
        expires_at=SessionCodec(SESSION_SECRET).now() + ttl,
        generation_tag="" if stored is None else generation_tag(stored),
        oidc_subject=subject,
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
    """An authorized OIDC request reports the account behind it."""

    SUBJECT_HEADER = "X-Osprey-Auth-Subject"
    ALICE_SUBJECT = "idp|alice"

    def test_oidc_authorization_returns_the_subject_header(self) -> None:
        cookie = _mint(_unlocked("alice", stored=None, subject=self.ALICE_SUBJECT))
        with TestClient(_app(OIDC_ENV)) as client:
            response = _verify(client, "alice", cookie)
        assert response.status_code == 200
        assert response.headers[self.SUBJECT_HEADER] == self.ALICE_SUBJECT

    def test_password_session_omits_the_subject_header(self) -> None:
        """A password entry has no subject, so the header is absent, not blank."""
        cookie = _mint(_unlocked("alice"))
        with TestClient(_app()) as client:
            response = _verify(client, "alice", cookie)
        assert response.status_code == 200
        assert self.SUBJECT_HEADER.lower() not in response.headers

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
