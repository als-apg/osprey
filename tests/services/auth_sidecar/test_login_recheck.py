"""Tests for the login re-check: one identity matrix, enforced in one place.

The sidecar is the only component that knows *how* a browser proved who it is,
so it is the only component that can say what that proof is worth. These tests
pin the matrix of :mod:`osprey.services.auth_sidecar.routes.recheck` row by row:

* ``none`` — no login method at all. The sidecar mints nothing and records no
  login event, because a deployment with no login has no login to record; the
  identity a single-user install runs under comes from
  :func:`~osprey.utils.identity.acting_identity`, which this service never
  reaches for.
* ``password`` — the roster username is the subject and the roster entry's own
  ``role:`` is the role. Static, per user, and read by *key* rather than found
  by search.
* ``oidc`` with no claims map — the provider subject is the subject, and the
  role is the roster's: nothing asked the provider about privilege, so the role
  is the one the render bound this user's persona from.
* ``oidc`` with a claims map — the role comes from the validated ID token, by
  the category set task 4.4 established, and is *cross-checked* against the
  roster's own. A token mapping someone into a role other than the one their
  terminal was built as is refused (``role_mismatch``), because that session
  would name one role while sitting in another's container.

Two invariants cut across every row. **Fail closed:** a combination the matrix
does not describe refuses the login rather than picking a plausible reading of
it. **Anti-lookup:** nothing here ever searches the roster for the entry that
*matches* an identity or a role — every resolution is keyed by the username
whose card was clicked, so a mis-click can never become someone else's session.
"""

from __future__ import annotations

import inspect
import json
from base64 import b64encode
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import httpx
import itsdangerous
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.deployment.web_terminals.personas import env_var_suffix
from osprey.services.auth_sidecar import audit
from osprey.services.auth_sidecar.app import STATE_COOKIE_NAME, create_app
from osprey.services.auth_sidecar.identity_headers import (
    ROLE_HEADER,
    ROLE_SOURCE_HEADER,
    SUBJECT_HEADER,
)
from osprey.services.auth_sidecar.passwords import hash_password
from osprey.services.auth_sidecar.routes import recheck
from osprey.services.auth_sidecar.routes import verify as verify_module
from osprey.services.auth_sidecar.routes.oidc import (
    CALLBACK_PATH,
    PENDING_FLOW_SESSION_KEY,
    RoleBinding,
)
from osprey.services.auth_sidecar.routes.recheck import (
    METHOD_OIDC,
    METHOD_PASSWORD,
    ROLE_SOURCE_CLAIM,
    ROLE_SOURCE_ROSTER,
    LoginGrant,
    RecheckRefused,
    RosterRoles,
    recheck_login,
    session_role,
    session_role_source,
    session_subject,
)
from osprey.services.auth_sidecar.routes.verify import VERIFY_PATH
from osprey.services.auth_sidecar.sessions import (
    SESSION_COOKIE_NAME,
    SessionCodec,
    UnlockedUser,
)
from osprey.utils.identity import AUDIT_IDENTITY_ENV, TERMINAL_USER_ENV

SESSION_SECRET = "session-secret-value"
STATE_SECRET = "state-secret-value"
SESSION_LIFETIME = 3600
EXTERNAL_ORIGIN = "https://terminals.example.org"
FLOW_STATE = "handshake-state-value"

ALICE_PASSWORD = "alice-password"
ALICE_HASH = hash_password(ALICE_PASSWORD)
ALICE_SUBJECT = "idp|alice"
BOB_SUBJECT = "idp|bob"

SIDECAR_IDENTITY = "sidecar"

GROUP_CLAIM = "groups"
OPERATOR_GROUP = "als-operators"
CLAIM_MAP = {OPERATOR_GROUP: "operator"}

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
    "OSPREY_AUTH_OIDC_SUBJECT_BOB": BOB_SUBJECT,
    "OSPREY_AUTH_EXTERNAL_ORIGIN": EXTERNAL_ORIGIN,
    "OSPREY_AUTH_TLS_ENABLED": "true",
}


# --- harness ----------------------------------------------------------------


@pytest.fixture
def zone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The sidecar's own bound audit subdirectory, as compose gives it."""
    directory = tmp_path / "var" / "audit" / SIDECAR_IDENTITY
    directory.mkdir(parents=True)
    monkeypatch.setenv(audit.AUDIT_DIR_ENV, str(directory))
    monkeypatch.setenv(AUDIT_IDENTITY_ENV, SIDECAR_IDENTITY)
    monkeypatch.delenv(TERMINAL_USER_ENV, raising=False)
    return directory


def _records(zone: Path) -> list[dict[str, Any]]:
    path = zone / f"{audit.SURFACE}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]


def _bind_roster_roles(monkeypatch: pytest.MonkeyPatch, **roles: str) -> None:
    """Bind a static role to each named user, the way the container's env does.

    Set on the *process* environment rather than in the mapping passed to
    ``create_app``: like
    :class:`~osprey.services.auth_sidecar.routes.oidc.RoleBinding`, this table
    is read from ``os.environ`` by the module that owns it rather than from
    :class:`~osprey.services.auth_sidecar.app.AuthSettings`, which is what the
    container actually runs on and what an explicit test mapping is not.
    """
    for user, role in roles.items():
        # Derived the ONE way, never re-spelled: `env_var_suffix` is what the
        # render emits the variable under and what the service reads it back
        # by, so a fixture that hard-coded `upper()` would keep passing after
        # the two drifted apart (`alice-b` -> `ALICE_B`, not `ALICE-B`).
        monkeypatch.setenv(f"{recheck.ENV_ROSTER_ROLE_PREFIX}{env_var_suffix(user)}", role)


def _login(
    client: TestClient, user: str = "alice", password: str = ALICE_PASSWORD
) -> httpx.Response:
    return client.post(
        "/auth/login",
        data={"user": user, "next": "", "password": password},
        headers={"Origin": EXTERNAL_ORIGIN},
        follow_redirects=False,
    )


def _issued_cookie(response: httpx.Response) -> str:
    for header in response.headers.get_list("set-cookie"):
        if header.startswith(f"{SESSION_COOKIE_NAME}="):
            return header.split(";", 1)[0].split("=", 1)[1]
    raise AssertionError("the login issued no session cookie")


def _minted_entry(cookie: str, user: str) -> UnlockedUser:
    """The entry a login actually signed, read back out of the cookie it issued.

    The role a session carries is visible from verify's headers; the source it
    carries is only visible beside a role, so the cookie itself is where the two
    halves can be seen to have been minted together.
    """
    entry = SessionCodec(SESSION_SECRET).decode(cookie).entry(user)
    assert entry is not None, "the login minted no entry for this user"
    return entry


def _verify(client: TestClient, user: str, cookie: str) -> httpx.Response:
    return client.get(VERIFY_PATH, params={"user": user}, cookies={SESSION_COOKIE_NAME: cookie})


class FakeOIDCClient:
    """Stands in for Authlib's Starlette client at the route boundary."""

    def __init__(self, *, userinfo: dict[str, Any] | None = None) -> None:
        self.token = {
            "id_token": "header.payload.signature",
            "userinfo": userinfo if userinfo is not None else {"sub": ALICE_SUBJECT},
        }

    async def create_authorization_url(self, redirect_uri: str | None = None) -> dict[str, Any]:
        return {
            "url": f"https://idp.example.org/authorize?state={FLOW_STATE}",
            "state": FLOW_STATE,
            "nonce": "nonce-value",
        }

    async def save_authorize_data(self, request: Any, **kwargs: Any) -> None:
        return None

    async def authorize_access_token(self, request: Any, **kwargs: Any) -> dict[str, Any]:
        return self.token


def _oidc_app(
    env: dict[str, str] | None = None,
    *,
    userinfo: dict[str, Any] | None = None,
    binding: RoleBinding | None = None,
) -> FastAPI:
    app = create_app(env if env is not None else OIDC_ENV)
    app.state.oidc_client = FakeOIDCClient(userinfo=userinfo)
    app.state.role_binding = binding if binding is not None else RoleBinding()
    return app


def _pending(user: str) -> str:
    signer = itsdangerous.TimestampSigner(STATE_SECRET)
    data = {PENDING_FLOW_SESSION_KEY: {"state": FLOW_STATE, "user": user, "next": f"/u/{user}/"}}
    return signer.sign(b64encode(json.dumps(data).encode("utf-8"))).decode("utf-8")


def _callback(app: FastAPI, user: str = "alice") -> httpx.Response:
    with TestClient(app, base_url=EXTERNAL_ORIGIN) as client:
        client.cookies.set(STATE_COOKIE_NAME, _pending(user))
        return client.get(
            CALLBACK_PATH,
            params={"code": "auth-code", "state": FLOW_STATE},
            follow_redirects=False,
        )


class RecordingMapping(Mapping[str, str]):
    """A mapping that remembers how it was read.

    The instrument behind the anti-lookup tests: a *lookup* touches
    ``__getitem__``/``get`` with one key, while a *search* has to iterate. The
    two are distinguishable only from the mapping's side, which is why this
    exists rather than an assertion about the answer.
    """

    def __init__(self, values: Mapping[str, str]) -> None:
        self._values = dict(values)
        self.keys_read: list[str] = []
        self.iterations = 0

    def __getitem__(self, key: str) -> str:
        self.keys_read.append(key)
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        self.iterations += 1
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


# --- the matrix, row by row -------------------------------------------------


class TestTheNoneRow:
    """No login method: nothing is minted, and nothing is recorded."""

    def test_the_matrix_admits_no_grant(self) -> None:
        with pytest.raises(RecheckRefused) as refused:
            recheck_login(method="none", user="alice", roster_roles=RosterRoles())
        assert refused.value.reason == recheck.REASON_UNSUPPORTED_METHOD

    def test_no_login_event_is_recorded(self, zone: Path) -> None:
        """The single-user install's identity comes from ``acting_identity()``,
        not from a login — so a deployment with no login method has no login
        events, and this service must not invent any."""
        env = dict(PASSWORD_ENV)
        env["OSPREY_AUTH_METHOD"] = "none"
        with TestClient(create_app(env), base_url="https://testserver") as client:
            assert _login(client).status_code == 503
        assert _records(zone) == []


class TestThePasswordRow:
    """The roster username is the subject; the roster entry's role is the role,
    and the grant says so — nothing else was asked."""

    def test_the_grant_names_the_roster_user_and_their_role(self) -> None:
        grant = recheck_login(
            method=METHOD_PASSWORD,
            user="alice",
            roster_roles=RosterRoles({"alice": "operator"}),
        )
        assert grant == LoginGrant(subject="alice", role="operator", role_source=ROLE_SOURCE_ROSTER)

    def test_a_roster_entry_with_no_role_grants_none(self) -> None:
        grant = recheck_login(
            method=METHOD_PASSWORD, user="alice", roster_roles=RosterRoles({"bob": "operator"})
        )
        assert grant == LoginGrant(subject="alice", role="", role_source="")

    def test_the_login_mints_the_roster_role_into_the_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _bind_roster_roles(monkeypatch, alice="operator")
        with TestClient(create_app(PASSWORD_ENV), base_url="https://testserver") as client:
            login = _login(client)
            assert login.status_code == 303
            cookie = _issued_cookie(login)
            response = _verify(client, "alice", cookie)
        assert response.status_code == 200
        assert response.headers[SUBJECT_HEADER] == "alice"
        assert response.headers[ROLE_HEADER] == "operator"
        assert _minted_entry(cookie, "alice").role_source == ROLE_SOURCE_ROSTER

    def test_the_ledger_reports_the_role_the_login_granted(
        self, zone: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _bind_roster_roles(monkeypatch, alice="operator")
        with TestClient(create_app(PASSWORD_ENV), base_url="https://testserver") as client:
            assert _login(client).status_code == 303
        assert [(r["decision"], r["subject"], r.get("role")) for r in _records(zone)] == [
            ("allowed", "alice", "operator")
        ]

    def test_a_roleless_deployment_is_unchanged(self) -> None:
        """Every password deployment written before roles existed: the field is
        there, the value is empty, and verify emits no role header for it."""
        with TestClient(create_app(PASSWORD_ENV), base_url="https://testserver") as client:
            login = _login(client)
            response = _verify(client, "alice", _issued_cookie(login))
        assert response.status_code == 200
        assert ROLE_HEADER.lower() not in response.headers


class TestTheOidcRowWithoutAClaimsMap:
    """No claim binding: nothing asked the provider about privilege, so the role
    is the one the RENDER bound — the roster entry this user's persona came
    from, which is what the grant names as its source. Audited like every other
    login."""

    def test_the_grant_carries_the_provider_subject_and_the_roster_role(self) -> None:
        grant = recheck_login(
            method=METHOD_OIDC,
            user="alice",
            roster_roles=RosterRoles({"alice": "observer"}),
            asserted_subject=ALICE_SUBJECT,
            claim_role="",
        )
        assert grant == LoginGrant(
            subject=ALICE_SUBJECT, role="observer", role_source=ROLE_SOURCE_ROSTER
        )

    def test_a_roster_entry_with_no_role_still_grants_none(self) -> None:
        grant = recheck_login(
            method=METHOD_OIDC,
            user="alice",
            roster_roles=RosterRoles({"bob": "operator"}),
            asserted_subject=ALICE_SUBJECT,
            claim_role="",
        )
        assert grant == LoginGrant(subject=ALICE_SUBJECT, role="", role_source="")

    def test_the_session_carries_the_roster_role(
        self, zone: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End to end, through the callback: alice's roster ``role: observer``
        is what the render built her terminal from, and with no claim binding to
        ask the provider about privilege it is what her session names too."""
        _bind_roster_roles(monkeypatch, alice="observer")
        response = _callback(_oidc_app())
        assert response.status_code == 303
        assert [record.get("role") for record in _records(zone)] == ["observer"]
        assert _minted_entry(_issued_cookie(response), "alice").role_source == ROLE_SOURCE_ROSTER

    def test_the_login_is_recorded(self, zone: Path) -> None:
        assert _callback(_oidc_app()).status_code == 303
        assert [(r["decision"], r["reason"]) for r in _records(zone)] == [
            ("allowed", audit.REASON_OIDC_LOGIN)
        ]


class TestTheOidcRowWithAClaimsMap:
    """The role comes from the validated token — and must be the one this user's
    terminal was rendered as. The token is what decided it, so the token is what
    the grant credits."""

    def test_the_grant_carries_the_resolved_role(self) -> None:
        grant = recheck_login(
            method=METHOD_OIDC,
            user="alice",
            roster_roles=RosterRoles({"alice": "operator"}),
            asserted_subject=ALICE_SUBJECT,
            claim_role="operator",
        )
        assert grant == LoginGrant(
            subject=ALICE_SUBJECT, role="operator", role_source=ROLE_SOURCE_CLAIM
        )

    def test_a_claim_role_that_is_not_the_rendered_one_refuses(self) -> None:
        """SC4's first clause. alice's container was built from ``observer``;
        a token mapping her to ``operator`` describes a terminal she is not
        about to land in, so the login is refused rather than reconciled."""
        with pytest.raises(RecheckRefused) as refused:
            recheck_login(
                method=METHOD_OIDC,
                user="alice",
                roster_roles=RosterRoles({"alice": "observer"}),
                asserted_subject=ALICE_SUBJECT,
                claim_role="operator",
            )
        assert refused.value.reason == audit.REASON_ROLE_MISMATCH

    def test_an_entry_with_no_rendered_role_carries_the_claims(self) -> None:
        """The one honest gap, stated in the module docstring: a ``persona:``-
        pinned entry (or one riding the default persona) has no rendered role
        for the claim to disagree with, so the claim's role is what it carries."""
        grant = recheck_login(
            method=METHOD_OIDC,
            user="alice",
            roster_roles=RosterRoles({"bob": "operator"}),
            asserted_subject=ALICE_SUBJECT,
            claim_role="operator",
        )
        assert grant == LoginGrant(
            subject=ALICE_SUBJECT, role="operator", role_source=ROLE_SOURCE_CLAIM
        )

    def test_a_mapped_group_becomes_the_session_role(
        self, zone: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _bind_roster_roles(monkeypatch, alice="operator")
        app = _oidc_app(
            userinfo={"sub": ALICE_SUBJECT, GROUP_CLAIM: [OPERATOR_GROUP]},
            binding=RoleBinding(claim=GROUP_CLAIM, claim_map=CLAIM_MAP),
        )
        response = _callback(app)
        assert response.status_code == 303
        assert [record.get("role") for record in _records(zone)] == ["operator"]
        assert _minted_entry(_issued_cookie(response), "alice").role_source == ROLE_SOURCE_CLAIM

    def test_an_unmapped_group_fails_closed(self, zone: Path) -> None:
        app = _oidc_app(
            userinfo={"sub": ALICE_SUBJECT, GROUP_CLAIM: ["nobody-maps-this"]},
            binding=RoleBinding(claim=GROUP_CLAIM, claim_map=CLAIM_MAP),
        )
        assert _callback(app).status_code == 403
        assert [record["reason"] for record in _records(zone)] == [audit.REASON_UNMAPPED_ROLE_CLAIM]


class TestTheClaimIsCrossCheckedAgainstTheRenderedRole:
    """SC4, end to end through the OIDC callback.

    The deviation this class exists to close: alice's roster entry says
    ``role: observer``, so ``effective_persona`` built her container from the
    observer persona — while her IdP group maps to ``operator``. Before the
    cross-check, that login was a 303 whose session, forwarded header and every
    ledger line said ``operator`` about a container running as ``observer``.
    """

    @staticmethod
    def _app(userinfo_group: str) -> FastAPI:
        return _oidc_app(
            userinfo={"sub": ALICE_SUBJECT, GROUP_CLAIM: [userinfo_group]},
            binding=RoleBinding(
                claim=GROUP_CLAIM,
                claim_map={OPERATOR_GROUP: "operator", "als-observers": "observer"},
            ),
        )

    def test_a_login_into_someone_elses_role_is_refused_and_filed(
        self, zone: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _bind_roster_roles(monkeypatch, alice="observer")

        result = _callback(self._app(OPERATOR_GROUP))

        assert result.status_code == 403
        assert [(r["decision"], r["subject"], r["reason"]) for r in _records(zone)] == [
            ("refused", "alice", audit.REASON_ROLE_MISMATCH)
        ]

    def test_the_refused_login_mints_no_session(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _bind_roster_roles(monkeypatch, alice="observer")
        result = _callback(self._app(OPERATOR_GROUP))
        assert not [
            header
            for header in result.headers.get_list("set-cookie")
            if header.startswith(f"{SESSION_COOKIE_NAME}=")
        ]

    def test_the_matching_claim_is_admitted_and_carries_that_role(
        self, zone: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The control: the same deployment, the same user, the group that maps
        to the role her terminal was actually built from.

        The grant is watched as well as the ledger, because the two authorities
        agreeing is exactly the case where the answer could plausibly be filed
        under either one. It is the token's: the roster was the cross-check
        here, not the decision.
        """
        _bind_roster_roles(monkeypatch, alice="observer")
        grants: list[LoginGrant] = []

        def recording(**kwargs: Any) -> LoginGrant:
            grant = recheck_login(**kwargs)
            grants.append(grant)
            return grant

        monkeypatch.setattr("osprey.services.auth_sidecar.routes.oidc.recheck_login", recording)

        result = _callback(self._app("als-observers"))

        assert result.status_code == 303
        assert [(r["decision"], r.get("role")) for r in _records(zone)] == [("allowed", "observer")]
        assert [(g.role, g.role_source) for g in grants] == [("observer", ROLE_SOURCE_CLAIM)]

    def test_the_admitted_role_reaches_the_verify_subrequest(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """What nginx actually asks for: the header the terminal is entered
        with names the role the roster and the token agree on."""
        _bind_roster_roles(monkeypatch, alice="observer")
        app = self._app("als-observers")
        with TestClient(app, base_url=EXTERNAL_ORIGIN) as client:
            client.cookies.set(STATE_COOKIE_NAME, _pending("alice"))
            login = client.get(
                CALLBACK_PATH,
                params={"code": "auth-code", "state": FLOW_STATE},
                follow_redirects=False,
            )
            response = _verify(client, "alice", _issued_cookie(login))
        assert response.status_code == 200
        assert response.headers[ROLE_HEADER] == "observer"
        assert response.headers[ROLE_SOURCE_HEADER] == ROLE_SOURCE_CLAIM


# --- fail closed ------------------------------------------------------------


class TestFailClosed:
    """A combination the matrix does not describe refuses, never guesses."""

    @pytest.mark.parametrize("method", ["", "none", "None", "ldap", "PASSWORD"])
    def test_an_unsupported_method_grants_nothing(self, method: str) -> None:
        """Including the spellings that *nearly* work: the method arrives from
        the environment, and a case difference must not be read as a match."""
        with pytest.raises(RecheckRefused) as refused:
            recheck_login(method=method, user="alice", roster_roles=RosterRoles())
        assert refused.value.reason == recheck.REASON_UNSUPPORTED_METHOD

    def test_a_password_login_carrying_a_provider_subject_refuses(self) -> None:
        """The impossible combination: a password login has no IdP behind it, so
        a caller that has one has confused two flows."""
        with pytest.raises(RecheckRefused) as refused:
            recheck_login(
                method=METHOD_PASSWORD,
                user="alice",
                roster_roles=RosterRoles(),
                asserted_subject=ALICE_SUBJECT,
            )
        assert refused.value.reason == recheck.REASON_METHOD_MISMATCH

    def test_a_password_login_carrying_a_claim_role_refuses(self) -> None:
        with pytest.raises(RecheckRefused) as refused:
            recheck_login(
                method=METHOD_PASSWORD,
                user="alice",
                roster_roles=RosterRoles(),
                claim_role="operator",
            )
        assert refused.value.reason == recheck.REASON_METHOD_MISMATCH

    @pytest.mark.parametrize("subject", [None, ""])
    def test_an_oidc_login_with_no_asserted_subject_refuses(self, subject: str | None) -> None:
        with pytest.raises(RecheckRefused) as refused:
            recheck_login(
                method=METHOD_OIDC,
                user="alice",
                roster_roles=RosterRoles(),
                asserted_subject=subject,
                claim_role="",
            )
        assert refused.value.reason == recheck.REASON_METHOD_MISMATCH

    def test_an_oidc_login_that_never_asked_about_the_role_refuses(self) -> None:
        """``claim_role=""`` is "this deployment binds no roles"; ``None`` is
        "nobody asked", which is not an answer the matrix accepts."""
        with pytest.raises(RecheckRefused) as refused:
            recheck_login(
                method=METHOD_OIDC,
                user="alice",
                roster_roles=RosterRoles(),
                asserted_subject=ALICE_SUBJECT,
                claim_role=None,
            )
        assert refused.value.reason == recheck.REASON_METHOD_MISMATCH

    def test_a_login_for_nobody_refuses(self) -> None:
        with pytest.raises(RecheckRefused) as refused:
            recheck_login(method=METHOD_PASSWORD, user="", roster_roles=RosterRoles())
        assert refused.value.reason == recheck.REASON_METHOD_MISMATCH

    @pytest.mark.parametrize("role", ["öps", "ops\r\nX-Osprey-Auth-Role: admin", " ops"])
    def test_an_uncarryable_roster_role_refuses_the_login(self, role: str) -> None:
        """Refused here rather than left to ``with_user``, whose ``ValueError``
        would surface as a 500 on what is a denial — the same reasoning the
        OIDC path's ``unsafe_role`` category is written down under."""
        with pytest.raises(RecheckRefused) as refused:
            recheck_login(
                method=METHOD_PASSWORD, user="alice", roster_roles=RosterRoles({"alice": role})
            )
        assert refused.value.reason == audit.REASON_UNSAFE_ROLE

    def test_an_uncarryable_roster_role_is_a_403_and_not_a_500(
        self, zone: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _bind_roster_roles(monkeypatch, alice="öps")
        with TestClient(create_app(PASSWORD_ENV), base_url="https://testserver") as client:
            response = _login(client)
        assert response.status_code == 403
        assert [(r["decision"], r["reason"]) for r in _records(zone)] == [
            ("refused", audit.REASON_UNSAFE_ROLE)
        ]

    def test_a_refused_re_check_mints_no_cookie(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The re-check runs before the session is minted, so a matrix failure
        cannot leave a browser holding a session the matrix refused."""
        _bind_roster_roles(monkeypatch, alice="öps")
        with TestClient(create_app(PASSWORD_ENV), base_url="https://testserver") as client:
            response = _login(client)
        assert not [
            header
            for header in response.headers.get_list("set-cookie")
            if header.startswith(f"{SESSION_COOKIE_NAME}=")
        ]


# --- the anti-lookup invariant ----------------------------------------------


class TestTheAntiLookupInvariant:
    """No resolution here ever searches the roster for a matching entry."""

    def test_the_role_lookup_reads_one_key_and_iterates_nothing(self) -> None:
        """A search would have to iterate; a lookup cannot. Measured from the
        mapping's own side, because the answer looks identical either way."""
        table = RecordingMapping({"alice": "operator", "bob": "observer"})
        grant = recheck_login(method=METHOD_PASSWORD, user="alice", roster_roles=RosterRoles(table))
        assert grant.role == "operator"
        assert table.keys_read == ["alice"]
        assert table.iterations == 0

    def test_an_unknown_user_is_not_searched_for_either(self) -> None:
        table = RecordingMapping({"bob": "observer"})
        grant = recheck_login(method=METHOD_PASSWORD, user="alice", roster_roles=RosterRoles(table))
        assert grant.role == ""
        assert table.iterations == 0

    def test_an_identity_mapped_to_another_user_unlocks_nobody(self, zone: Path) -> None:
        """Bob's subject arriving on alice's handshake is refused, not resolved
        into bob's session: the card that was clicked is the only user a login
        can unlock."""
        response = _callback(_oidc_app(userinfo={"sub": BOB_SUBJECT}), user="alice")
        assert response.status_code == 403
        assert not [
            header
            for header in response.headers.get_list("set-cookie")
            if header.startswith(f"{SESSION_COOKIE_NAME}=")
        ]
        assert [(r["subject"], r["reason"]) for r in _records(zone)] == [
            ("alice", audit.REASON_IDENTITY_MISMATCH)
        ]

    #: Every public name the matrix module is allowed to define.
    #:
    #: Deliberately the whole surface rather than a pattern. A guard that
    #: matched only the spellings someone thought of — ``user_for``,
    #: ``user_with`` — let a working reverse lookup named ``owner_of_subject()``
    #: through untouched. Anything new has to be added to this set, which is a
    #: place a reader stops and asks what it resolves and from which direction.
    EXPECTED_SURFACE = frozenset(
        {
            "ENV_ROSTER_ROLE_PREFIX",
            "METHOD_OIDC",
            "METHOD_PASSWORD",
            "REASON_METHOD_MISMATCH",
            "REASON_ROLE_MISMATCH",
            "REASON_UNSUPPORTED_METHOD",
            "ROLE_SOURCE_CLAIM",
            "ROLE_SOURCE_ROSTER",
            "SUPPORTED_METHODS",
            "LoginGrant",
            "RecheckRefused",
            "RosterRoles",
            "recheck_login",
            "roster_roles",
            "session_role",
            "session_role_source",
            "session_subject",
        }
    )

    @staticmethod
    def _defined_here() -> set[str]:
        """The module's public names, minus everything it merely imported.

        An import is not part of this module's surface — pinning ``Mapping`` or
        ``dataclass`` would make the assertion a diff of the import block. What
        is left is what ``recheck.py`` itself defines: its constants (which
        carry no ``__module__``) and its own classes and functions.
        """
        surface: set[str] = set()
        for name in dir(recheck):
            if name.startswith("_"):
                continue
            value = getattr(recheck, name)
            if inspect.ismodule(value):
                continue
            origin = getattr(value, "__module__", None)
            if origin is not None and origin != recheck.__name__:
                continue
            surface.add(name)
        return surface

    def test_the_matrix_exposes_no_reverse_lookup(self) -> None:
        """A helper answering "which user holds this role" is what an anti-lookup
        invariant forbids, so the module must not grow one by accident.

        Structural, not a naming canary: the surface is compared as a set, so a
        reverse lookup fails this test whatever it is called.
        """
        assert self._defined_here() == set(self.EXPECTED_SURFACE)

    def test_the_declared_surface_is_the_defined_one(self) -> None:
        """``__all__`` is checked against reality in both directions, so a new
        helper cannot hide by being left out of it (or by being listed and never
        written)."""
        assert set(recheck.__all__) == self._defined_here()


# --- what the table tells a reader ------------------------------------------


class TestTheTableSaysWhatEachEndDoes:
    """The properties an operator reads off the docstrings, pinned as text.

    Both are invisible from behaviour alone — one is a row that mint and verify
    dispose of differently, the other is a role that outlives the config change
    that retired it — and both are the kind of thing a reader has to be *told*.
    A canary, deliberately: if the sentence goes, so does the only place the
    property is written down.
    """

    def test_the_matrix_qualifies_its_last_row(self) -> None:
        """ "anything else" is refused at MINT; verify names the locally verified
        identity instead, and the table says so rather than leaving it to
        ``session_subject``'s own docstring."""
        table = recheck.__doc__ or ""
        assert "refused at mint" in table
        assert "locally verified identity" in table
        assert "session_subject` does not re-check the closed set" in table

    def test_verify_says_the_role_is_the_one_the_login_granted(self) -> None:
        """An operator who removes a privilege reasonably expects it gone; what
        actually happens is that it lapses with the session."""
        doc = verify_module._authorized.__doc__ or ""
        assert "the one the login granted" in doc
        assert "lapses with the session" in doc
        assert "generation tag" in doc


# --- the minted subject is the matrix's answer ------------------------------


class TestTheMintedSubjectIsTheMatrixAnswer:
    """Both mint sites carry the grant's ``subject``, not their own reading of it.

    On the password row the two are the same string — the row grants the roster
    username, which is what the route passed in — so the route agreeing with
    the table proves nothing on its own. What proves it is changing the table's
    answer and watching the session follow: a row whose subject the route
    re-derives instead of reading would silently not reach the browser.
    """

    def _mint_under_a_moved_row(
        self, monkeypatch: pytest.MonkeyPatch, subject: str
    ) -> tuple[FastAPI, httpx.Response]:
        """Log alice in with the matrix granting ``subject`` instead."""
        app = create_app(PASSWORD_ENV)

        def moved_row(**kwargs: Any) -> LoginGrant:
            return LoginGrant(subject=subject, role="")

        monkeypatch.setattr("osprey.services.auth_sidecar.routes.login.recheck_login", moved_row)
        with TestClient(app, base_url=EXTERNAL_ORIGIN) as client:
            return app, _login(client)

    def test_the_session_names_the_subject_the_matrix_granted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, response = self._mint_under_a_moved_row(monkeypatch, "mallory")
        assert response.status_code == 303
        codec = app.state.session_codec
        state = codec.decode(_issued_cookie(response))
        assert state.unlocked_usernames(codec.now()) == ("mallory",)

    def test_a_moved_row_stops_authorizing_the_name_that_logged_in(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The consequence, on the surface nginx actually asks: the session no
        longer unlocks alice, because the matrix no longer says it should."""
        app, response = self._mint_under_a_moved_row(monkeypatch, "mallory")
        cookie = _issued_cookie(response)
        with TestClient(app, base_url=EXTERNAL_ORIGIN) as client:
            assert _verify(client, "alice", cookie).status_code == 401

    def test_the_unmoved_row_still_mints_the_roster_username(self) -> None:
        """The control: with the real table, the password row grants the roster
        username and nothing about the minted session changed."""
        app = create_app(PASSWORD_ENV)
        with TestClient(app, base_url=EXTERNAL_ORIGIN) as client:
            response = _login(client)
            assert response.status_code == 303
            cookie = _issued_cookie(response)
            assert _verify(client, "alice", cookie).status_code == 200
        codec = app.state.session_codec
        assert codec.decode(cookie).unlocked_usernames(codec.now()) == ("alice",)


# --- the roster role source -------------------------------------------------


class TestTheRosterRoleSource:
    """Per-user, env-only, and keyed by the same suffix every other per-user
    variable in this service is keyed by."""

    def test_a_role_is_read_for_each_roster_user(self) -> None:
        roles = RosterRoles.from_env(
            ("alice", "bob"),
            {
                f"{recheck.ENV_ROSTER_ROLE_PREFIX}ALICE": "operator",
                f"{recheck.ENV_ROSTER_ROLE_PREFIX}BOB": "observer",
            },
        )
        assert roles.role_for("alice") == "operator"
        assert roles.role_for("bob") == "observer"

    def test_the_key_is_derived_by_the_shared_suffix_mapping(self) -> None:
        """A hyphenated username, which is where ``upper()`` alone stops working.

        The same bug the password hashes already have a pin against
        (``test_app.py::test_per_user_hash_uses_the_shared_suffix_mapping``),
        on the variable that decides a *role*: the render emits
        ``OSPREY_AUTH_ROSTER_ROLE_ALICE_B``, and a reader that upper-cased
        without replacing the hyphen would look for ``ALICE-B``, find nothing,
        and hand the user no privileges without a word.

        The literal is asserted alongside, so this cannot pass by re-deriving
        the same wrong answer on both sides.
        """
        assert env_var_suffix("alice-b") == "ALICE_B"
        roles = RosterRoles.from_env(
            ("alice-b",),
            {f"{recheck.ENV_ROSTER_ROLE_PREFIX}{env_var_suffix('alice-b')}": "operator"},
        )
        assert roles.role_for("alice-b") == "operator"

    def test_a_user_outside_the_roster_has_no_role(self) -> None:
        roles = RosterRoles.from_env(
            ("alice",), {f"{recheck.ENV_ROSTER_ROLE_PREFIX}CAROL": "admin"}
        )
        assert roles.role_for("carol") == ""

    @pytest.mark.parametrize("raw", ["", "   "])
    def test_a_blank_value_is_no_role_at_all(self, raw: str) -> None:
        """Never a role named ``""``: the deny-safe reading of a variable that
        was rendered from an empty config key."""
        roles = RosterRoles.from_env(("alice",), {f"{recheck.ENV_ROSTER_ROLE_PREFIX}ALICE": raw})
        assert roles.role_for("alice") == ""
        assert "alice" not in roles.roles

    def test_the_value_is_stripped(self) -> None:
        roles = RosterRoles.from_env(
            ("alice",), {f"{recheck.ENV_ROSTER_ROLE_PREFIX}ALICE": "  operator  "}
        )
        assert roles.role_for("alice") == "operator"

    def test_the_prefix_cannot_collide_with_the_claim_binding(self) -> None:
        """``OSPREY_AUTH_ROLE_`` was unavailable: ``OSPREY_AUTH_ROLE_CLAIM`` and
        ``OSPREY_AUTH_ROLE_MAP`` already mean something else, and a roster user
        named ``claim`` would have read the group binding as their own role."""
        from osprey.services.auth_sidecar.routes.oidc import ENV_ROLE_CLAIM, ENV_ROLE_MAP

        assert not ENV_ROLE_CLAIM.startswith(recheck.ENV_ROSTER_ROLE_PREFIX)
        assert not ENV_ROLE_MAP.startswith(recheck.ENV_ROSTER_ROLE_PREFIX)
        roles = RosterRoles.from_env(
            ("claim", "map"), {ENV_ROLE_CLAIM: "groups", ENV_ROLE_MAP: "{}"}
        )
        assert roles.role_for("claim") == ""
        assert roles.role_for("map") == ""

    def test_the_table_is_parsed_once_and_cached_on_the_app(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _bind_roster_roles(monkeypatch, alice="operator")
        app = create_app(PASSWORD_ENV)
        with TestClient(app, base_url="https://testserver") as client:
            assert _login(client).status_code == 303
        cached = app.state.roster_roles
        assert isinstance(cached, RosterRoles)
        assert cached.role_for("alice") == "operator"

    def test_two_apps_do_not_share_one_table(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The ``RoleBinding`` reasoning, for the same reason: two apps in one
        process must not run on the first one's authorization table."""
        _bind_roster_roles(monkeypatch, alice="operator")
        first = create_app(PASSWORD_ENV)
        first.state.roster_roles = RosterRoles({"alice": "admin"})
        second = create_app(PASSWORD_ENV)
        with TestClient(first, base_url="https://testserver") as client:
            login = _login(client)
            assert _verify(client, "alice", _issued_cookie(login)).headers[ROLE_HEADER] == "admin"
        with TestClient(second, base_url="https://testserver") as client:
            login = _login(client)
            response = _verify(client, "alice", _issued_cookie(login))
        assert response.headers[ROLE_HEADER] == "operator"


# --- one matrix, both ends --------------------------------------------------


class TestVerifyReadsTheSameMatrix:
    """The subject a verify subrequest forwards is the matrix's, not a second
    reading of the same question."""

    def test_a_password_session_names_the_roster_user(self) -> None:
        entry = UnlockedUser(username="alice", expires_at=0, generation_tag="", oidc_subject="")
        assert session_subject(method=METHOD_PASSWORD, username="alice", entry=entry) == "alice"

    def test_an_oidc_session_names_the_provider_account(self) -> None:
        entry = UnlockedUser(
            username="alice", expires_at=0, generation_tag="", oidc_subject=ALICE_SUBJECT
        )
        assert session_subject(method=METHOD_OIDC, username="alice", entry=entry) == ALICE_SUBJECT

    def test_a_pre_subject_oidc_session_names_nobody(self) -> None:
        """A session minted before the subject was carried: the header is
        omitted rather than filled with the roster name, which would report a
        provider account this login never asserted."""
        entry = UnlockedUser(username="alice", expires_at=0, generation_tag="", oidc_subject="")
        assert session_subject(method=METHOD_OIDC, username="alice", entry=entry) == ""

    def test_a_missing_entry_names_nobody(self) -> None:
        assert session_subject(method=METHOD_OIDC, username="alice", entry=None) == ""

    def test_an_unexpected_method_names_the_locally_verified_identity(self) -> None:
        """Written as "not OIDC" so the fallback branch is the one naming an
        identity this service verified itself, never a stored claim."""
        entry = UnlockedUser(
            username="alice", expires_at=0, generation_tag="", oidc_subject=BOB_SUBJECT
        )
        assert session_subject(method="ldap", username="alice", entry=entry) == "alice"


class TestAMethodChangeUnderAnOutstandingCookie:
    """Switching ``auth.method`` retires the roles those cookies were granted.

    The signing secret survives a re-render, so a password session stays
    readable after a switch to ``oidc`` — and under ``oidc`` the generation-tag
    check that a password rotation would have tripped is skipped. Without this
    rule the cookie kept forwarding the roster role it was minted with, under a
    method that never asserted it and with the rotation check out of reach.
    """

    @staticmethod
    def _password_entry(
        role: str = "operator", role_source: str = ROLE_SOURCE_ROSTER
    ) -> UnlockedUser:
        return UnlockedUser(
            username="alice",
            expires_at=0,
            generation_tag="tag",
            oidc_subject="",
            role=role,
            role_source=role_source,
        )

    def test_a_password_session_keeps_its_role_under_password(self) -> None:
        """The control: nothing changed for a deployment that did not switch."""
        assert session_role(method=METHOD_PASSWORD, entry=self._password_entry()) == "operator"

    def test_a_password_session_read_under_oidc_holds_no_role(self) -> None:
        assert session_role(method=METHOD_OIDC, entry=self._password_entry()) == ""

    def test_an_oidc_session_keeps_the_role_its_token_granted(self) -> None:
        entry = UnlockedUser(
            username="alice",
            expires_at=0,
            generation_tag="",
            oidc_subject=ALICE_SUBJECT,
            role="operator",
        )
        assert session_role(method=METHOD_OIDC, entry=entry) == "operator"

    def test_a_missing_entry_holds_no_role(self) -> None:
        assert session_role(method=METHOD_OIDC, entry=None) == ""

    def test_a_password_session_keeps_its_source_under_password(self) -> None:
        """The source rides with the role it explains, on the unchanged path."""
        entry = self._password_entry()
        assert session_role_source(method=METHOD_PASSWORD, entry=entry) == ROLE_SOURCE_ROSTER

    def test_a_password_session_read_under_oidc_holds_no_source(self) -> None:
        """Lockstep: the flip that retires the role retires its provenance too,
        so nothing is left naming where a lapsed privilege came from."""
        assert session_role_source(method=METHOD_OIDC, entry=self._password_entry()) == ""

    def test_a_missing_entry_holds_no_source(self) -> None:
        assert session_role_source(method=METHOD_OIDC, entry=None) == ""

    def test_a_source_without_a_role_explains_nothing(self) -> None:
        """No grant can mint this shape — only a payload signed by something
        other than :func:`_grant` — and the derived reader drops it rather than
        forwarding provenance for a role the session does not hold."""
        entry = self._password_entry(role="", role_source=ROLE_SOURCE_ROSTER)
        assert session_role_source(method=METHOD_PASSWORD, entry=entry) == ""

    def test_the_flipped_deployment_forwards_no_role_header(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole path, over one cookie: mint under ``password``, then
        present the same cookie to the same secret rendered as ``oidc``."""
        _bind_roster_roles(monkeypatch, alice="operator")
        with TestClient(create_app(PASSWORD_ENV), base_url="https://testserver") as client:
            login = _login(client)
            assert login.status_code == 303
            cookie = _issued_cookie(login)
            before = _verify(client, "alice", cookie)
        assert before.headers[ROLE_HEADER] == "operator"
        assert before.headers[ROLE_SOURCE_HEADER] == ROLE_SOURCE_ROSTER

        flipped = {**OIDC_ENV, "OSPREY_AUTH_SESSION_SECRET": SESSION_SECRET}
        with TestClient(create_app(flipped), base_url="https://testserver") as client:
            after = _verify(client, "alice", cookie)

        assert after.status_code == 200
        assert ROLE_HEADER.lower() not in after.headers
        # The source is derived from the role, so it lapses in the same breath —
        # never left naming where a retired privilege came from.
        assert ROLE_SOURCE_HEADER.lower() not in after.headers
        # The subject half was already closed; asserted beside it so the two
        # halves of one identity cannot drift back apart.
        assert SUBJECT_HEADER.lower() not in after.headers
