"""Tests for the role field and its source, the identity headers, and what may
travel in them.

Three things are pinned here, and they are one property seen from three sides:
*an identity the deployment cannot carry across the nginx boundary is not an
identity this sidecar will authorise with.*

* **The payload carries a role, and where it came from.** ``UnlockedUser``
  gained a ``role`` and then a ``role_source``, both additively and deny-safe:
  absent means ``""``, a cookie minted before either field existed still
  decodes, and no payload version was burned for either.
* **All four identity headers are emitted, or no value is invented.** The
  account is the one that always rides — an authorized request is by definition
  on a roster card — so an answer without it means a sidecar older than this
  release, not a session with nothing to say. The rest are omitted rather than
  blank: a password session names the roster user in ``X-Osprey-Auth-Subject``
  (presence still means a known account); an OIDC session names the provider
  subject; a session that holds neither leaves the header absent rather than
  blank.
* **Only header-safe values ever exist.** The mint path refuses to store a
  non-ASCII subject or role, the decode path refuses to return one, and the
  verify path refuses to authorise on one. A non-ASCII OIDC subject therefore
  fails the *login* closed, with its own audited category — it is not a header
  quietly dropped at the end of a successful authorisation.

Apps are built from explicit env mappings, and the OIDC client is replaced at
the route boundary, so nothing here depends on the real process environment or
on reaching an IdP.
"""

from __future__ import annotations

import json
import logging
from base64 import b64encode
from typing import Any

import httpx
import itsdangerous
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from itsdangerous import URLSafeSerializer

from osprey.services.auth_sidecar import audit
from osprey.services.auth_sidecar.app import STATE_COOKIE_NAME, AuthSettings, create_app
from osprey.services.auth_sidecar.exceptions import InvalidSessionError
from osprey.services.auth_sidecar.identity_headers import (
    ROLE_HEADER,
    ROLE_SOURCE_HEADER,
    SUBJECT_HEADER,
    is_header_safe,
)
from osprey.services.auth_sidecar.passwords import generation_tag, hash_password
from osprey.services.auth_sidecar.routes.oidc import CALLBACK_PATH, PENDING_FLOW_SESSION_KEY
from osprey.services.auth_sidecar.routes.verify import _authorized
from osprey.services.auth_sidecar.sessions import (
    PAYLOAD_VERSION,
    SESSION_COOKIE_NAME,
    SIGNATURE_SALT,
    SessionCodec,
    SessionState,
    UnlockedUser,
)

SESSION_SECRET = "session-secret-value"
STATE_SECRET = "state-secret-value"
SESSION_LIFETIME = 3600
EXTERNAL_ORIGIN = "https://terminals.example.org"
SESSION_ID = "test-session-id"
FLOW_STATE = "handshake-state-value"

ALICE_PASSWORD = "alice-password"
ALICE_HASH = hash_password(ALICE_PASSWORD)
ALICE_SUBJECT = "idp|alice"

#: A subject a latin-1 HTTP header cannot carry unchanged across the boundary.
#: The OIDC ``sub`` is ASCII by specification, so this can only arrive from a
#: deployment mapping a non-ASCII claim spelling onto a roster user.
NON_ASCII_SUBJECT = "jörg@example.org"

PASSWORD_ENV = {
    "OSPREY_AUTH_METHOD": "password",
    "OSPREY_AUTH_SESSION_SECRET": SESSION_SECRET,
    "OSPREY_AUTH_SESSION_LIFETIME": str(SESSION_LIFETIME),
    "OSPREY_AUTH_USERS": "alice,bob",
    "OSPREY_AUTH_PW_HASH_ALICE": ALICE_HASH,
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
    "OSPREY_AUTH_OIDC_SUBJECT_ALICE": ALICE_SUBJECT,
    "OSPREY_AUTH_EXTERNAL_ORIGIN": EXTERNAL_ORIGIN,
    "OSPREY_AUTH_TLS_ENABLED": "true",
}


# --- harness ----------------------------------------------------------------


def _mint(*entries: UnlockedUser, session_id: str = SESSION_ID) -> str:
    """Sign a session cookie carrying ``entries``, the way a login route does."""
    codec = SessionCodec(SESSION_SECRET)
    return codec.encode(SessionState(session_id=session_id, issued_at=codec.now(), users=entries))


def _raw_cookie(users: dict[str, dict[str, Any]]) -> str:
    """Sign a payload by hand, so a test can omit a key the codec always writes.

    This is how a cookie minted by an *older* sidecar is reproduced: the codec
    has no way to leave ``role`` out of what it encodes, and the whole point of
    an additive field is that a payload without it still decodes.
    """
    payload = {
        "v": PAYLOAD_VERSION,
        "sid": SESSION_ID,
        "iat": SessionCodec(SESSION_SECRET).now(),
        "users": users,
    }
    return URLSafeSerializer(SESSION_SECRET, salt=SIGNATURE_SALT).dumps(payload)


def _password_entry(
    username: str = "alice", *, role: str = "", role_source: str = "", ttl: float = 600.0
) -> UnlockedUser:
    """A password-mode entry: a generation tag, no subject."""
    return UnlockedUser(
        username=username,
        expires_at=SessionCodec(SESSION_SECRET).now() + ttl,
        generation_tag=generation_tag(ALICE_HASH),
        role=role,
        role_source=role_source,
    )


def _oidc_entry(
    username: str = "alice",
    *,
    subject: str = ALICE_SUBJECT,
    role: str = "",
    role_source: str = "",
    ttl: float = 600.0,
) -> UnlockedUser:
    """An OIDC entry: a subject, no generation tag."""
    return UnlockedUser(
        username=username,
        expires_at=SessionCodec(SESSION_SECRET).now() + ttl,
        generation_tag="",
        oidc_subject=subject,
        role=role,
        role_source=role_source,
    )


def _verify(client: TestClient, user: str, cookie: str) -> httpx.Response:
    """Issue one subrequest the way nginx's internal auth location does.

    The cookie goes in as a raw header rather than through the client's jar, so
    each request carries exactly the cookie the test named.
    """
    return client.get(
        "/verify",
        params={"user": user},
        headers={"Cookie": f"{SESSION_COOKIE_NAME}={cookie}"},
    )


def _issued_cookie(response: httpx.Response) -> str:
    """The session cookie value a response set, read off the header."""
    return response.headers["set-cookie"].split(";", 1)[0].split("=", 1)[1]


class FakeOIDCClient:
    """Stands in for Authlib's Starlette client at the route boundary.

    Deliberately local rather than shared with ``test_oidc.py``: these tests
    assert that the callback refuses *before* it reaches the token exchange, and
    a stand-in that records whether it was called at all is what proves it.
    """

    def __init__(self, *, userinfo: dict[str, Any] | None = None) -> None:
        # An `id_token` alongside the claims: the callback refuses a token
        # response without one, because that is the only case in which Authlib
        # actually parsed and validated what it put in `userinfo`.
        self.token = {
            "id_token": "header.payload.signature",
            "userinfo": userinfo or {"sub": ALICE_SUBJECT},
        }
        self.exchanged = False

    async def create_authorization_url(self, redirect_uri: str | None = None) -> dict[str, Any]:
        return {
            "url": f"https://idp.example.org/authorize?state={FLOW_STATE}",
            "state": FLOW_STATE,
            "nonce": "nonce-value",
        }

    async def save_authorize_data(self, request: Any, **kwargs: Any) -> None:
        return None

    async def authorize_access_token(self, request: Any, **kwargs: Any) -> dict[str, Any]:
        self.exchanged = True
        return self.token


def _oidc_app(env: dict[str, str] | None = None, client: FakeOIDCClient | None = None) -> FastAPI:
    """An app in OIDC mode with Authlib replaced."""
    app = create_app(env if env is not None else OIDC_ENV)
    app.state.oidc_client = client if client is not None else FakeOIDCClient()
    return app


def _pending(user: str) -> str:
    """A forged state cookie carrying one in-flight handshake for ``user``."""
    signer = itsdangerous.TimestampSigner(STATE_SECRET)
    data = {PENDING_FLOW_SESSION_KEY: {"state": FLOW_STATE, "user": user, "next": f"/u/{user}/"}}
    return signer.sign(b64encode(json.dumps(data).encode("utf-8"))).decode("utf-8")


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Capture what the sidecar hands its audit seam."""
    written: list[Any] = []
    monkeypatch.setattr(audit, "write_envelope", written.append)
    return written


# --- what may travel in an identity header ----------------------------------


def test_the_role_source_header_spelling() -> None:
    """The nginx template writes the name as a literal and the terminal
    re-spells it rather than importing it, so nothing but an assertion holds
    the three copies together. A rename that stopped at the constant would
    leave the provenance forwarded under a name nothing reads."""
    assert ROLE_SOURCE_HEADER == "X-Osprey-Auth-Role-Source"


class TestHeaderSafety:
    """The predicate that decides whether a value can cross the boundary."""

    @pytest.mark.parametrize(
        "value",
        [
            "alice",
            "idp|alice",
            "8f14e45f-ceea-4a5c-9c76-01dd8f7f56a2",
            "cn=Alice Smith, ou=Users",  # spaces inside are ordinary in a DN
            "operator-2",
        ],
    )
    def test_ascii_identifiers_are_carryable(self, value: str) -> None:
        assert is_header_safe(value)

    @pytest.mark.parametrize(
        ("value", "why"),
        [
            ("jörg@example.org", "non-ASCII: latin-1 carries it, nginx does not"),
            ("идентификатор", "not even latin-1 encodable"),
            ("alice\r\nX-Osprey-Auth-Role: admin", "header injection"),
            ("alice\n", "a bare newline still splits the header"),
            ("alice\tbob", "control characters are not header text"),
            (" alice", "leading space is stripped by an HTTP parser"),
            ("alice ", "trailing space likewise"),
            ("", "an empty value names nobody"),
        ],
    )
    def test_unsafe_values_are_refused(self, value: str, why: str) -> None:
        assert not is_header_safe(value), why


# --- the payload ------------------------------------------------------------


class TestRoleOnThePayload:
    """``role`` is additive and deny-safe."""

    def test_role_defaults_to_empty(self) -> None:
        """Deny-safe: an entry that names no role holds ``""``, never ``None``
        and never a stand-in default that some table would answer."""
        entry = UnlockedUser(username="alice", expires_at=1.0)
        assert entry.role == ""

    def test_role_round_trips_through_the_cookie(self) -> None:
        codec = SessionCodec(SESSION_SECRET)
        state = codec.new_state().with_user("alice", expires_at=codec.now() + 600, role="operator")
        entry = codec.decode(codec.encode(state)).entry("alice")
        assert entry is not None
        assert entry.role == "operator"

    def test_a_cookie_minted_before_the_field_existed_still_decodes(self) -> None:
        """Additive means additive: no payload version was burned, so an
        outstanding session keeps working and simply names no role."""
        raw = _raw_cookie(
            {"alice": {"exp": SessionCodec(SESSION_SECRET).now() + 600, "tag": "0123456789abcdef"}}
        )
        entry = SessionCodec(SESSION_SECRET).decode(raw).entry("alice")
        assert entry is not None
        assert entry.role == ""

    def test_the_payload_version_was_not_bumped(self) -> None:
        assert PAYLOAD_VERSION == 1

    def test_a_fresh_login_replaces_the_role_wholesale(self) -> None:
        """Replacement is not a merge — the same rule the tag and the subject
        already follow. A login that names no role clears the previous one
        rather than leaving a privilege the operator no longer holds."""
        codec = SessionCodec(SESSION_SECRET)
        state = (
            codec.new_state()
            .with_user("alice", expires_at=codec.now() + 600, role="operator")
            .with_user("alice", expires_at=codec.now() + 600)
        )
        entry = state.entry("alice")
        assert entry is not None
        assert entry.role == ""

    def test_a_non_string_role_is_refused(self) -> None:
        raw = _raw_cookie(
            {"alice": {"exp": SessionCodec(SESSION_SECRET).now() + 600, "tag": "", "role": ["ops"]}}
        )
        with pytest.raises(InvalidSessionError):
            SessionCodec(SESSION_SECRET).decode(raw)


class TestRoleSourceOnThePayload:
    """``role_source`` is additive and deny-safe, on the role's own terms."""

    def test_role_source_defaults_to_empty(self) -> None:
        """Provenance for no role is no provenance — ``""``, never ``None`` and
        never a guess at which end resolved a role the entry does not hold."""
        entry = UnlockedUser(username="alice", expires_at=1.0)
        assert entry.role_source == ""

    def test_role_source_round_trips_through_the_cookie(self) -> None:
        codec = SessionCodec(SESSION_SECRET)
        state = codec.new_state().with_user(
            "alice", expires_at=codec.now() + 600, role="operator", role_source="roster"
        )
        entry = codec.decode(codec.encode(state)).entry("alice")
        assert entry is not None
        assert entry.role == "operator"
        assert entry.role_source == "roster"

    def test_a_cookie_minted_before_the_field_existed_still_decodes(self) -> None:
        """The upgrade case this field is shaped around: a session already open
        keeps its role and simply shows no provenance until the next login."""
        raw = _raw_cookie(
            {
                "alice": {
                    "exp": SessionCodec(SESSION_SECRET).now() + 600,
                    "tag": "0123456789abcdef",
                    "role": "operator",
                }
            }
        )
        entry = SessionCodec(SESSION_SECRET).decode(raw).entry("alice")
        assert entry is not None
        assert entry.role == "operator"
        assert entry.role_source == ""

    def test_the_payload_version_was_not_bumped_for_the_source_either(self) -> None:
        assert PAYLOAD_VERSION == 1

    def test_a_fresh_login_replaces_the_role_source_wholesale(self) -> None:
        """The source cannot outlive the role it qualifies: a login naming no
        role clears both, so provenance never names the origin of a privilege
        this session no longer holds."""
        codec = SessionCodec(SESSION_SECRET)
        state = (
            codec.new_state()
            .with_user("alice", expires_at=codec.now() + 600, role="operator", role_source="claim")
            .with_user("alice", expires_at=codec.now() + 600)
        )
        entry = state.entry("alice")
        assert entry is not None
        assert entry.role == ""
        assert entry.role_source == ""

    def test_a_non_string_role_source_is_refused(self) -> None:
        raw = _raw_cookie(
            {
                "alice": {
                    "exp": SessionCodec(SESSION_SECRET).now() + 600,
                    "tag": "",
                    "role": "operator",
                    "source": ["roster"],
                }
            }
        )
        with pytest.raises(InvalidSessionError):
            SessionCodec(SESSION_SECRET).decode(raw)

    def test_with_user_refuses_an_uncarryable_role_source(self) -> None:
        """The mint-side half of the same guard the subject and the role get:
        a value the boundary would mangle is refused where the caller can
        still be told, not dropped at the end of a successful login."""
        codec = SessionCodec(SESSION_SECRET)
        with pytest.raises(ValueError, match="role source"):
            codec.new_state().with_user(
                "alice", expires_at=codec.now() + 600, role="operator", role_source="röster"
            )

    def test_decoding_refuses_an_uncarryable_role_source(self) -> None:
        """And the read-side half. Only this sidecar could have signed such a
        cookie, so it is refused rather than returned with the source dropped."""
        raw = _raw_cookie(
            {
                "alice": {
                    "exp": SessionCodec(SESSION_SECRET).now() + 600,
                    "tag": "",
                    "role": "operator",
                    "source": NON_ASCII_SUBJECT,
                }
            }
        )
        with pytest.raises(InvalidSessionError):
            SessionCodec(SESSION_SECRET).decode(raw)


class TestOnlyCarryableIdentitiesAreStored:
    """The mint path refuses what the boundary cannot carry."""

    def test_with_user_refuses_a_non_ascii_subject(self) -> None:
        codec = SessionCodec(SESSION_SECRET)
        with pytest.raises(ValueError, match="subject"):
            codec.new_state().with_user(
                "alice", expires_at=codec.now() + 600, oidc_subject=NON_ASCII_SUBJECT
            )

    def test_with_user_refuses_a_non_ascii_role(self) -> None:
        codec = SessionCodec(SESSION_SECRET)
        with pytest.raises(ValueError, match="role"):
            codec.new_state().with_user("alice", expires_at=codec.now() + 600, role="öps")

    def test_with_user_refuses_a_role_carrying_a_newline(self) -> None:
        codec = SessionCodec(SESSION_SECRET)
        with pytest.raises(ValueError, match="role"):
            codec.new_state().with_user(
                "alice", expires_at=codec.now() + 600, role="ops\r\nX-Osprey-Auth-Role: admin"
            )

    @pytest.mark.parametrize("key", ["sub", "role"])
    def test_decoding_refuses_an_uncarryable_value(self, key: str) -> None:
        """Belt and braces on the read side. A cookie carrying one of these
        could only come from a sidecar that signed it, which the mint-side guard
        prevents — so it is refused rather than returned, and verify's answer
        stays a plain 401 instead of an encoding failure on the hot path."""
        raw = _raw_cookie(
            {
                "alice": {
                    "exp": SessionCodec(SESSION_SECRET).now() + 600,
                    "tag": "",
                    key: NON_ASCII_SUBJECT,
                }
            }
        )
        with pytest.raises(InvalidSessionError):
            SessionCodec(SESSION_SECRET).decode(raw)


# --- what verify emits ------------------------------------------------------


class TestSubjectHeader:
    """Presence means a known account — in both methods now."""

    def test_a_password_session_names_the_roster_user(self) -> None:
        """New in this task: the roster username *is* the account behind a
        password session, and downstream authorization needs a subject to key
        on in both methods."""
        cookie = _mint(_password_entry())
        with TestClient(create_app(PASSWORD_ENV)) as client:
            response = _verify(client, "alice", cookie)
        assert response.status_code == 200
        assert response.headers[SUBJECT_HEADER] == "alice"

    def test_an_oidc_session_still_names_the_provider_account(self) -> None:
        cookie = _mint(_oidc_entry())
        with TestClient(_oidc_app()) as client:
            response = _verify(client, "alice", cookie)
        assert response.status_code == 200
        assert response.headers[SUBJECT_HEADER] == ALICE_SUBJECT

    def test_a_pre_subject_oidc_session_still_omits_the_header(self) -> None:
        """The compatibility case this task keeps: a session minted before the
        subject was carried authorises exactly as before and names no account,
        so an absent header never has to be told apart from a blank one."""
        cookie = _mint(_oidc_entry(subject=""))
        with TestClient(_oidc_app()) as client:
            response = _verify(client, "alice", cookie)
        assert response.status_code == 200
        assert SUBJECT_HEADER.lower() not in response.headers

    def test_a_denial_carries_no_identity_header(self) -> None:
        cookie = _mint(_oidc_entry(role="operator", role_source="claim"))
        with TestClient(_oidc_app()) as client:
            response = _verify(client, "bob", cookie)
        assert response.status_code == 401
        assert SUBJECT_HEADER.lower() not in response.headers
        assert ROLE_HEADER.lower() not in response.headers
        assert ROLE_SOURCE_HEADER.lower() not in response.headers


class TestRoleHeader:
    """The role rides the same 200, on the same terms."""

    def test_a_carried_role_is_emitted(self) -> None:
        cookie = _mint(_password_entry(role="operator"))
        with TestClient(create_app(PASSWORD_ENV)) as client:
            response = _verify(client, "alice", cookie)
        assert response.status_code == 200
        assert response.headers[ROLE_HEADER] == "operator"

    def test_an_entry_with_no_role_omits_the_header(self) -> None:
        """Deny-safe by absence: no role header means no role, which is what a
        downstream consumer must read as "no privileges", never as a default."""
        cookie = _mint(_password_entry())
        with TestClient(create_app(PASSWORD_ENV)) as client:
            response = _verify(client, "alice", cookie)
        assert response.status_code == 200
        assert ROLE_HEADER.lower() not in response.headers

    def test_an_oidc_session_carries_its_role_too(self) -> None:
        cookie = _mint(_oidc_entry(role="observer"))
        with TestClient(_oidc_app()) as client:
            response = _verify(client, "alice", cookie)
        assert response.status_code == 200
        assert response.headers[SUBJECT_HEADER] == ALICE_SUBJECT
        assert response.headers[ROLE_HEADER] == "observer"


class TestRoleSourceHeader:
    """The source rides beside the role, and only beside it."""

    def test_a_roster_role_names_its_source(self) -> None:
        cookie = _mint(_password_entry(role="operator", role_source="roster"))
        with TestClient(create_app(PASSWORD_ENV)) as client:
            response = _verify(client, "alice", cookie)
        assert response.status_code == 200
        assert response.headers[ROLE_HEADER] == "operator"
        assert response.headers[ROLE_SOURCE_HEADER] == "roster"

    def test_a_claimed_role_names_its_source(self) -> None:
        cookie = _mint(_oidc_entry(role="observer", role_source="claim"))
        with TestClient(_oidc_app()) as client:
            response = _verify(client, "alice", cookie)
        assert response.status_code == 200
        assert response.headers[ROLE_HEADER] == "observer"
        assert response.headers[ROLE_SOURCE_HEADER] == "claim"

    def test_an_entry_with_no_role_omits_the_source_too(self) -> None:
        """Nothing to explain: the header names the origin of a privilege, so
        it cannot be present where the privilege is not."""
        cookie = _mint(_password_entry())
        with TestClient(create_app(PASSWORD_ENV)) as client:
            response = _verify(client, "alice", cookie)
        assert response.status_code == 200
        assert ROLE_HEADER.lower() not in response.headers
        assert ROLE_SOURCE_HEADER.lower() not in response.headers

    def test_a_source_signed_without_a_role_forwards_neither(self) -> None:
        """A payload no login route can mint, signed by hand the way an older
        cookie is: the reader derives the source from the role, so a source
        standing alone authorises the same 200 and forwards nothing about it."""
        raw = _raw_cookie(
            {
                "alice": {
                    "exp": SessionCodec(SESSION_SECRET).now() + 600,
                    "tag": generation_tag(ALICE_HASH),
                    "source": "claim",
                }
            }
        )
        with TestClient(create_app(PASSWORD_ENV)) as client:
            response = _verify(client, "alice", raw)
        assert response.status_code == 200
        assert ROLE_HEADER.lower() not in response.headers
        assert ROLE_SOURCE_HEADER.lower() not in response.headers

    def test_a_source_that_cannot_be_carried_is_refused(self) -> None:
        """The source's own branch of the carriage guard, reached directly for
        the reason the role's is: the codec refuses to decode such a value and
        ``with_user`` refuses to mint one, and a guard no test reaches is one
        refactor away from being deleted as dead code."""
        settings = AuthSettings.from_env(PASSWORD_ENV)
        entry = UnlockedUser(
            username="alice",
            expires_at=SessionCodec(SESSION_SECRET).now() + 600.0,
            generation_tag=generation_tag(ALICE_HASH),
            role="operator",
            role_source="roster\r\nX-Osprey-Auth-Role: admin",
        )
        response = _authorized("alice", entry, settings)
        assert response.status_code == 401
        assert ROLE_SOURCE_HEADER.lower() not in response.headers
        assert ROLE_HEADER.lower() not in response.headers
        assert SUBJECT_HEADER.lower() not in response.headers


class TestAnUncarryableIdentityDenies:
    """Verify never emits a value it did not validate."""

    #: The env key this roster name's credential is provisioned under.
    #: ``env_var_suffix("jörg")`` upper-cases without transliterating, so the
    #: key carries the same non-ASCII character the name does. Spelling it any
    #: other way leaves the user with no stored credential, and the 401 that
    #: follows is the credential-missing one — the earlier branch, which would
    #: make every test below pass without the guard they exist to protect.
    JORG_HASH_VAR = "OSPREY_AUTH_PW_HASH_JÖRG"

    def test_the_roster_name_does_have_a_credential(self) -> None:
        """The control for the test below: this user is fully provisioned, so
        the refusal there can only be the identity-header one."""
        env = {**PASSWORD_ENV, "OSPREY_AUTH_USERS": "jörg", self.JORG_HASH_VAR: ALICE_HASH}
        assert AuthSettings.from_env(env).password_hash("jörg") == ALICE_HASH

    def test_a_roster_name_that_cannot_be_carried_is_refused(self) -> None:
        """A username outside the render gate's charset (a ``--no-lint`` render)
        would become the subject header in password mode. Authorising it would
        either fail the response encoding on the hot path or forward a mangled
        identity, so the subrequest is denied instead — the same bare 401 as
        every other refusal."""
        env = {**PASSWORD_ENV, "OSPREY_AUTH_USERS": "jörg", self.JORG_HASH_VAR: ALICE_HASH}
        cookie = _mint(_password_entry("jörg"))
        with TestClient(create_app(env)) as client:
            response = _verify(client, "jörg", cookie)
        assert response.status_code == 401
        assert SUBJECT_HEADER.lower() not in response.headers

    def test_a_role_that_cannot_be_carried_is_refused(self) -> None:
        """The role branch of the same guard, reached directly.

        Unreachable through a real request — the codec refuses to decode such a
        role, and ``with_user`` refuses to mint one — so the entry is built as
        the dataclass it is. Defence in depth is still defence: an untested
        guard is one refactor away from being deleted as dead code, which is
        exactly when it stops being defence."""
        settings = AuthSettings.from_env(PASSWORD_ENV)
        entry = UnlockedUser(
            username="alice",
            expires_at=SessionCodec(SESSION_SECRET).now() + 600.0,
            generation_tag=generation_tag(ALICE_HASH),
            role="öps",
        )
        response = _authorized("alice", entry, settings)
        assert response.status_code == 401
        assert ROLE_HEADER.lower() not in response.headers
        assert SUBJECT_HEADER.lower() not in response.headers


# --- the password login path ------------------------------------------------


class TestPasswordLogin:
    """What a real password login mints, and what verify then reports."""

    def _login(self, client: TestClient) -> httpx.Response:
        """POST one credential attempt the way a browser on this deployment does.

        The ``Origin`` is not optional: the route refuses a POST that will not
        say where it came from.
        """
        return client.post(
            "/auth/login",
            data={"user": "alice", "next": "", "password": ALICE_PASSWORD},
            headers={"Origin": EXTERNAL_ORIGIN},
            follow_redirects=False,
        )

    def test_a_login_then_a_verify_reports_the_roster_user(self) -> None:
        with TestClient(create_app(PASSWORD_ENV), base_url="https://testserver") as client:
            login = self._login(client)
            assert login.status_code == 303
            response = _verify(client, "alice", _issued_cookie(login))
        assert response.status_code == 200
        assert response.headers[SUBJECT_HEADER] == "alice"

    def test_the_minted_entry_carries_no_role_yet(self) -> None:
        """Deny-safe until the roster role is resolved into the session by the
        login re-check task: the field exists, the value is ``""``, and verify
        emits no role header for it."""
        with TestClient(create_app(PASSWORD_ENV), base_url="https://testserver") as client:
            login = self._login(client)
            response = _verify(client, "alice", _issued_cookie(login))
        entry = (
            SessionCodec(SESSION_SECRET, max_age=SESSION_LIFETIME)
            .decode(_issued_cookie(login))
            .entry("alice")
        )
        assert login.status_code == 303
        assert entry is not None
        assert entry.role == ""
        assert ROLE_HEADER.lower() not in response.headers


# --- a non-ASCII OIDC subject fails the login closed ------------------------


class TestNonAsciiSubjectFailsClosed:
    """The debt the verify docstring used to defer, paid at the login."""

    def _callback(self, app: FastAPI, user: str) -> httpx.Response:
        with TestClient(app, base_url=EXTERNAL_ORIGIN) as client:
            client.cookies.set(STATE_COOKIE_NAME, _pending(user))
            return client.get(
                CALLBACK_PATH,
                params={"code": "auth-code", "state": FLOW_STATE},
                follow_redirects=False,
            )

    def test_the_login_is_refused_not_the_header_dropped(self, recorded: list[Any]) -> None:
        env = {**OIDC_ENV, "OSPREY_AUTH_OIDC_SUBJECT_ALICE": NON_ASCII_SUBJECT}
        response = self._callback(_oidc_app(env), "alice")
        assert response.status_code == 403
        assert SESSION_COOKIE_NAME not in response.cookies
        # The refusal is the audited one, not a login that quietly went missing:
        # the fixture is here for the assertion as well as for its stubbing.
        assert [record.to_dict()["decision"] for record in recorded] == ["refused"]

    def test_the_refusal_precedes_the_token_exchange(self) -> None:
        """A mapping this deployment could never carry is refused before the
        IdP is contacted at all — the login could only have ended in a refusal."""
        env = {**OIDC_ENV, "OSPREY_AUTH_OIDC_SUBJECT_ALICE": NON_ASCII_SUBJECT}
        fake = FakeOIDCClient()
        self._callback(_oidc_app(env, fake), "alice")
        assert not fake.exchanged

    def test_the_refusal_has_its_own_audited_category(self, recorded: list[Any]) -> None:
        env = {**OIDC_ENV, "OSPREY_AUTH_OIDC_SUBJECT_ALICE": NON_ASCII_SUBJECT}
        self._callback(_oidc_app(env), "alice")
        assert len(recorded) == 1
        record = recorded[0].to_dict()
        assert record["reason"] == audit.REASON_NON_ASCII_SUBJECT
        assert record["decision"] == "refused"
        assert record["subject"] == "alice"

    def test_the_audited_record_never_carries_the_offending_value(
        self, recorded: list[Any]
    ) -> None:
        """The category is the evidence. The claim spelling itself is
        configuration the operator can read from their own config, and a
        ledger line is not where it belongs."""
        env = {**OIDC_ENV, "OSPREY_AUTH_OIDC_SUBJECT_ALICE": NON_ASCII_SUBJECT}
        self._callback(_oidc_app(env), "alice")
        assert NON_ASCII_SUBJECT not in json.dumps(recorded[0].to_dict(), ensure_ascii=False)

    def test_the_log_line_names_the_category_and_not_the_value(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        env = {**OIDC_ENV, "OSPREY_AUTH_OIDC_SUBJECT_ALICE": NON_ASCII_SUBJECT}
        with caplog.at_level(logging.WARNING, logger="osprey"):
            self._callback(_oidc_app(env), "alice")
        logged = "\n".join(
            record.getMessage() for record in caplog.records if record.name.startswith("osprey.")
        )
        assert "alice" in logged
        assert NON_ASCII_SUBJECT not in logged

    def test_an_ascii_subject_is_unaffected(self, recorded: list[Any]) -> None:
        """The control: the ordinary mapping still completes the login, and the
        one record it files is the success — no refusal is audited alongside a
        login that happened."""
        response = self._callback(_oidc_app(), "alice")
        assert response.status_code == 303
        assert SESSION_COOKIE_NAME in response.cookies
        assert [record.to_dict()["decision"] for record in recorded] == ["allowed"]

    def test_the_audit_seam_never_costs_the_refusal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A ledger that cannot be written degrades the audit trail, never the
        decision: the login is still refused when the seam raises."""

        def boom(envelope: Any) -> None:
            raise OSError("the ledger is unwritable")

        monkeypatch.setattr(audit, "write_envelope", boom)
        env = {**OIDC_ENV, "OSPREY_AUTH_OIDC_SUBJECT_ALICE": NON_ASCII_SUBJECT}
        assert self._callback(_oidc_app(env), "alice").status_code == 403
