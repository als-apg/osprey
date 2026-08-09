"""Tests for the auth sidecar's logout route.

The acceptance case is the one a shared control-room machine depends on: alice
logs out, bob — unlocked in the same browser, in the same cookie — is still
unlocked afterwards. Everything else here is either the other half of that
(the surrendered cookie really is dead) or the discipline that keeps a logout
from reporting on a session it was not given.

Cookies are minted with a client-side
:class:`~osprey.services.auth_sidecar.sessions.SessionCodec` rather than by
driving the login route, so these tests exercise logout alone. Sessions are
verified through the real ``/verify`` endpoint, because "revoked" means nothing
except as verify sees it. The app is always built from an explicit env mapping,
so a variable left in the real process environment cannot make one pass.
"""

from __future__ import annotations

import logging

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.services.auth_sidecar.app import create_app
from osprey.services.auth_sidecar.passwords import generation_tag, hash_password
from osprey.services.auth_sidecar.routes.logout import LANDING_PATH, LOGOUT_PATH
from osprey.services.auth_sidecar.sessions import (
    SESSION_COOKIE_NAME,
    SessionCodec,
    SessionState,
    UnlockedUser,
)

SESSION_SECRET = "session-secret-value"
SESSION_LIFETIME = 3600

ALICE_HASH = hash_password("alice-password")
BOB_HASH = hash_password("bob-password")

PASSWORD_ENV = {
    "OSPREY_AUTH_METHOD": "password",
    "OSPREY_AUTH_SESSION_SECRET": SESSION_SECRET,
    "OSPREY_AUTH_SESSION_LIFETIME": str(SESSION_LIFETIME),
    "OSPREY_AUTH_USERS": "alice,bob",
    "OSPREY_AUTH_PW_HASH_ALICE": ALICE_HASH,
    "OSPREY_AUTH_PW_HASH_BOB": BOB_HASH,
    "OSPREY_AUTH_EXTERNAL_ORIGIN": "https://terminals.example.org",
    "OSPREY_AUTH_TLS_ENABLED": "true",
}

SESSION_ID = "test-session-id"

_STORED = {"alice": ALICE_HASH, "bob": BOB_HASH}


def _unlocked(username: str, *, ttl: float = 600.0) -> UnlockedUser:
    """An unlocked-user entry expiring ``ttl`` seconds from now."""
    return UnlockedUser(
        username=username,
        expires_at=SessionCodec(SESSION_SECRET).now() + ttl,
        generation_tag=generation_tag(_STORED[username]),
    )


def _mint(*entries: UnlockedUser, session_id: str = SESSION_ID) -> str:
    """Sign a session cookie carrying ``entries``."""
    codec = SessionCodec(SESSION_SECRET)
    return codec.encode(SessionState(session_id=session_id, issued_at=codec.now(), users=entries))


def _app(env: dict[str, str] | None = None) -> FastAPI:
    """A sidecar app built from an explicit environment."""
    return create_app(PASSWORD_ENV if env is None else env)


def _logout(
    client: TestClient, user: str | list[str] | None, cookie: str | None = None
) -> httpx.Response:
    """Call logout the way the in-app control's navigation does.

    The cookie goes in as a raw header rather than through the client's jar, so
    each request carries exactly the cookie the test named. Redirects are not
    followed: the landing page is nginx's to serve, not this app's.
    """
    headers = {} if cookie is None else {"Cookie": f"{SESSION_COOKIE_NAME}={cookie}"}
    params = {} if user is None else {"user": user}
    return client.get(LOGOUT_PATH, params=params, headers=headers, follow_redirects=False)


def _verify(client: TestClient, user: str, cookie: str) -> int:
    """Ask the real verify endpoint whether ``cookie`` still authorizes ``user``."""
    return client.get(
        "/verify",
        params={"user": user},
        headers={"Cookie": f"{SESSION_COOKIE_NAME}={cookie}"},
    ).status_code


def _reissued(response: httpx.Response) -> str:
    """The session cookie value the logout response set."""
    jar = httpx.Cookies()
    jar.extract_cookies(response)
    value = jar.get(SESSION_COOKIE_NAME)
    assert value is not None, "logout did not reissue a session cookie"
    return value


class TestRegistration:
    """The factory picks the route up with no edit of its own."""

    def test_logout_is_registered_by_the_factory(self) -> None:
        assert LOGOUT_PATH in _app().openapi()["paths"]


class TestAcceptance:
    """One user leaves; the others stay exactly as they were."""

    def test_alice_logs_out_and_bob_stays_unlocked(self) -> None:
        cookie = _mint(_unlocked("alice"), _unlocked("bob"))
        with TestClient(_app()) as client:
            assert _verify(client, "alice", cookie) == 200
            assert _verify(client, "bob", cookie) == 200

            reissued = _reissued(_logout(client, "alice", cookie))

            assert _verify(client, "bob", reissued) == 200
            assert _verify(client, "alice", reissued) == 401

    def test_the_surrendered_cookie_stops_authorizing_everything(self) -> None:
        """The other half of the acceptance case: bob survives on the cookie he
        was *given*, never on the one that was handed in — that one is revoked by
        session id, which retires it for every user it carried."""
        cookie = _mint(_unlocked("alice"), _unlocked("bob"))
        with TestClient(_app()) as client:
            _logout(client, "alice", cookie)

            assert _verify(client, "alice", cookie) == 401
            assert _verify(client, "bob", cookie) == 401

    def test_logout_redirects_to_the_landing_page(self) -> None:
        cookie = _mint(_unlocked("alice"))
        with TestClient(_app()) as client:
            response = _logout(client, "alice", cookie)

        assert response.status_code == 303
        assert response.headers["location"] == LANDING_PATH
        # Like every login response: a cached logout would replay the redirect
        # without the revocation behind it.
        assert response.headers["cache-control"] == "no-store"

    def test_the_reissued_session_carries_a_fresh_session_id(self) -> None:
        """Why bob survives at all: revocation is recorded per session id, and
        verify refuses every user on a revoked one, so reissuing the same id
        would log bob out as a side effect of alice leaving."""
        cookie = _mint(_unlocked("alice"), _unlocked("bob"))
        with TestClient(_app()) as client:
            reissued = _reissued(_logout(client, "alice", cookie))

        codec = SessionCodec(SESSION_SECRET)
        assert codec.decode(reissued).session_id != SESSION_ID

    def test_the_reissued_session_does_not_extend_its_own_age(self) -> None:
        """Rotating the id must not reset the issued-at stamp, or logging out
        repeatedly would extend a session past `session_lifetime` indefinitely."""
        codec = SessionCodec(SESSION_SECRET)
        issued_at = codec.now() - 120
        cookie = codec.encode(
            SessionState(session_id=SESSION_ID, issued_at=issued_at, users=(_unlocked("bob"),))
        )
        with TestClient(_app()) as client:
            reissued = _reissued(_logout(client, "alice", cookie))

        assert codec.decode(reissued).issued_at == pytest.approx(issued_at)


class TestLoginAgainAfterLogout:
    """The self-inflicted-lockout regression the id rotation exists to prevent."""

    def _log_back_in(self, cookie: str, username: str) -> str:
        """Mint the cookie a login route would issue on top of ``cookie``.

        This is what every login path in this service does with the browser's
        current session: decode it, add the entry, keep the session id (see
        ``routes/oidc.py``'s ``_current_session`` followed by ``with_user``).
        The real routes are not driven here — the password login route has not
        landed, and the OIDC callback needs an IdP — but the session handling
        under test is theirs, character for character, and it is precisely the
        step that inherits whatever session id logout left behind.
        """
        codec = SessionCodec(SESSION_SECRET)
        session = codec.decode(cookie).with_user(
            username,
            expires_at=codec.now() + 600,
            generation_tag=generation_tag(_STORED[username]),
        )
        return codec.encode(session)

    def test_logging_in_again_in_the_same_browser_works(self) -> None:
        """Had logout reissued the revoked session id, this login would mint a
        fresh entry under an id the store already refuses: the user would get a
        303 and a cookie, then every verify would deny for up to
        `session_lifetime` — a silent lockout that looks like a broken password.
        """
        cookie = _mint(_unlocked("alice"))
        with TestClient(_app()) as client:
            reissued = _reissued(_logout(client, "alice", cookie))

            back_in = self._log_back_in(reissued, "alice")

            assert _verify(client, "alice", back_in) == 200

    def test_logging_in_again_does_not_revive_the_surrendered_cookie(self) -> None:
        """The other direction: a new login must not un-revoke the cookie that
        was handed in, which an attacker may have captured before the logout."""
        cookie = _mint(_unlocked("alice"))
        with TestClient(_app()) as client:
            reissued = _reissued(_logout(client, "alice", cookie))
            self._log_back_in(reissued, "alice")

            assert _verify(client, "alice", cookie) == 401

    def test_a_second_logout_after_logging_back_in_still_revokes(self) -> None:
        """Rotation must hold across cycles, not just the first one: the second
        logout revokes the id the second session actually carried."""
        cookie = _mint(_unlocked("alice"))
        app = _app()
        with TestClient(app) as client:
            reissued = _reissued(_logout(client, "alice", cookie))
            back_in = self._log_back_in(reissued, "alice")
            assert _verify(client, "alice", back_in) == 200

            second = _reissued(_logout(client, "alice", back_in))

            assert _verify(client, "alice", back_in) == 401
            assert _verify(client, "alice", self._log_back_in(second, "alice")) == 200


class TestRevocation:
    """Logout writes to the store verify reads."""

    def test_the_old_session_id_is_revoked_in_the_shared_store(self) -> None:
        """Revoking into a store nothing checks would look identical from the
        route's own return value, so this asserts on the app's store itself."""
        cookie = _mint(_unlocked("alice"))
        app = _app()
        with TestClient(app) as client:
            assert len(app.state.revocation_store) == 0
            _logout(client, "alice", cookie)

        assert app.state.revocation_store.is_revoked(SESSION_ID)

    def test_a_logout_for_a_user_who_was_not_unlocked_revokes_nothing(self) -> None:
        """Bob is on the roster but not in this cookie: there is no session of
        his to retire, and alice's must not be collateral."""
        cookie = _mint(_unlocked("alice"))
        app = _app()
        with TestClient(app) as client:
            _logout(client, "bob", cookie)

            assert len(app.state.revocation_store) == 0
            assert _verify(client, "alice", cookie) == 200


class TestBenignPaths:
    """Nothing about a logout reveals whether that user was unlocked."""

    def test_a_logout_with_no_cookie_redirects_without_error(self) -> None:
        with TestClient(_app()) as client:
            response = _logout(client, "alice")

        assert response.status_code == 303
        assert response.headers["location"] == LANDING_PATH

    def test_a_logout_with_no_user_parameter_redirects_without_error(self) -> None:
        """Never a 422: FastAPI's validation body is a different status, a
        different shape, and a description of this endpoint nothing needs."""
        cookie = _mint(_unlocked("alice"))
        with TestClient(_app()) as client:
            response = _logout(client, None, cookie)

        assert response.status_code == 303

    def test_a_repeated_user_parameter_acts_on_neither(self) -> None:
        """As in verify: refused outright rather than resolved to whichever value
        a parser picks first."""
        cookie = _mint(_unlocked("alice"), _unlocked("bob"))
        app = _app()
        with TestClient(app) as client:
            assert _logout(client, ["alice", "bob"], cookie).status_code == 303

            assert len(app.state.revocation_store) == 0
            assert _verify(client, "alice", cookie) == 200
            assert _verify(client, "bob", cookie) == 200

    def test_a_logout_for_an_unknown_user_redirects_without_error(self) -> None:
        """Not on the roster at all — still a plain redirect, with no hint that
        the name was unrecognised rather than merely locked."""
        cookie = _mint(_unlocked("alice"))
        with TestClient(_app()) as client:
            response = _logout(client, "mallory", cookie)

        assert response.status_code == 303

    def test_an_unlocked_and_a_locked_logout_are_shaped_identically(self) -> None:
        """The information-leak assertion: status, location and the presence of a
        reissued cookie are the same whether or not that user was unlocked."""
        cookie = _mint(_unlocked("alice"))
        with TestClient(_app()) as client:
            real = _logout(client, "alice", cookie)
            noop = _logout(client, "bob", cookie)

        assert (real.status_code, real.headers["location"]) == (
            noop.status_code,
            noop.headers["location"],
        )
        assert _reissued(real) and _reissued(noop)

    def test_an_unreadable_cookie_redirects_without_error(self) -> None:
        cookie = _mint(_unlocked("alice"))
        with TestClient(_app()) as client:
            assert _logout(client, "alice", cookie[:-4]).status_code == 303

    def test_logging_out_the_last_user_leaves_an_empty_session(self) -> None:
        cookie = _mint(_unlocked("alice"))
        with TestClient(_app()) as client:
            reissued = _reissued(_logout(client, "alice", cookie))

        assert SessionCodec(SESSION_SECRET).decode(reissued).users == ()


class TestCookieAttributes:
    """Parity with the cookie the login and OIDC callback routes issue.

    A cookie differing in any attribute below is a *second* cookie to the
    browser: the original stays in place and the logout logs out of nothing.
    """

    def _set_cookie_header(self, tls: str) -> str:
        env = {**PASSWORD_ENV, "OSPREY_AUTH_TLS_ENABLED": tls}
        cookie = _mint(_unlocked("alice"))
        with TestClient(_app(env)) as client:
            response = _logout(client, "alice", cookie)
        header = response.headers.get("set-cookie")
        assert header is not None
        return header

    def test_the_reissued_cookie_is_httponly_lax_and_root_pathed(self) -> None:
        header = self._set_cookie_header("true")

        assert header.startswith(f"{SESSION_COOKIE_NAME}=")
        assert "HttpOnly" in header
        assert "SameSite=lax" in header
        assert "Path=/" in header

    def test_the_reissued_cookie_is_secure_under_tls(self) -> None:
        assert "Secure" in self._set_cookie_header("true")

    def test_the_reissued_cookie_drops_secure_without_tls(self) -> None:
        """`allow_insecure_http` deployments would otherwise have the browser
        discard a cookie it was told to send only over HTTPS."""
        assert "Secure" not in self._set_cookie_header("false")

    def test_the_reissued_cookie_carries_no_max_age(self) -> None:
        """A session cookie: its real lifetime is the signed expiry inside it,
        and a Max-Age would be a second, unsigned answer to the same question."""
        header = self._set_cookie_header("true")

        assert "Max-Age" not in header
        assert "Expires" not in header


class TestLogging:
    """Caller-supplied values reach the log only through `%r`."""

    def test_a_username_carrying_a_newline_cannot_forge_a_log_line(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        cookie = _mint(_unlocked("alice"))
        with caplog.at_level(logging.DEBUG, logger="osprey.services.auth_sidecar.routes.logout"):
            with TestClient(_app()) as client:
                _logout(client, "mallory\nINFO forged", cookie)

        assert caplog.records
        assert "\\nINFO forged" in caplog.text
        assert "\nINFO forged" not in caplog.text
