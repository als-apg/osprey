"""Tests for the sidecar's login ledger: what it records, and where it lands.

The sidecar decides one thing — may this browser become this roster user — and
that decision is a safety decision like any tool call. These tests pin the four
properties that make its records usable next to the rest of the ledger:

* **Where.** Every record is one ``audit`` entry written through the one
  record writer into the spool directory ``OSPREY_SPOOL_DIR`` names. The
  sidecar is not an Osprey project (one uvicorn app, no ``osprey.yml`` to
  resolve a root from), so that variable is the *only* source of the path — a
  project-root resolution here would either raise or invent a directory outside
  the one the container binds.
* **Who.** The service is the actor and the roster user is the SUBJECT. Every
  entry is the sidecar's — the record takes the identity from the spool
  directory — and carries the person in ``on_behalf_of``, which the record
  keeps because the sidecar is on its trusted on-behalf list.
* **What.** ``posture_source`` is the app's own stamp and ``session`` is null:
  a login has no posture store and no session key, and inventing either would
  join these records to a session that never existed.
* **What it costs.** Nothing. An unwritable spool degrades to a log line and a
  health report, and both a 303 and a 403 arrive exactly as — and as fast as —
  they would have with the audit trail working.

Apps are built from explicit env mappings and the OIDC client is replaced at
the route boundary, so nothing here reaches an IdP or reads the real process
environment for anything but the audit variables the fixtures set.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.agent_record import writer as record_writer
from osprey.agent_record.contract import (
    KIND_AUDIT,
    RECORD_OFF_ENV,
    RECORD_OFF_VALUE,
    SPOOL_DIR_ENV,
    WRITER_ERRORS_FILE,
)
from osprey.audit import writer
from osprey.audit.envelope import POSTURE_SOURCE_APP, AuditEnvelope
from osprey.deployment.web_terminals.roster import USERNAME_CHARSET_RE
from osprey.services.auth_sidecar import audit, revocation
from osprey.services.auth_sidecar.app import (
    create_app,
    get_attempt_throttle,
    get_audit_throttle,
)
from osprey.services.auth_sidecar.passwords import hash_password
from osprey.services.auth_sidecar.revocation import STATE_DIR_ENV, state_directory
from osprey.services.auth_sidecar.routes.oidc import (
    FOLDED_DETAIL_KEY,
    REASON_AMBIGUOUS_ROLE_CLAIM,
    REASON_IDENTITY_MISMATCH,
    REASON_MISSING_ROLE_CLAIM,
    REASON_NO_ASSERTED_IDENTITY,
    REASON_UNMAPPED_ROLE_CLAIM,
    REASON_UNMAPPED_USER,
    REASON_UNSAFE_ROLE,
    RoleBinding,
)
from osprey.services.auth_sidecar.routes.recheck import RosterRoles
from osprey.services.auth_sidecar.throttle import AttemptThrottle
from osprey_connectors.identity import TERMINAL_USER_ENV
from tests.services.auth_sidecar._helpers import (
    SESSION_LIFETIME,
    SESSION_SECRET,
    SIDECAR_IDENTITY,
    STATE_SECRET,
    SidecarZone,
)
from tests.services.auth_sidecar._oidc import (
    ALICE_SUBJECT,
    EXTERNAL_ORIGIN,
    FakeOIDCClient,
    callback,
)

ALICE_PASSWORD = "alice-password"
ALICE_HASH = hash_password(ALICE_PASSWORD)
BOB_SUBJECT = "idp|bob"

GROUP_CLAIM = "groups"
OPERATOR_GROUP = "als-operators"
OBSERVER_GROUP = "als-observers"
CLAIM_MAP = {OPERATOR_GROUP: "operator", OBSERVER_GROUP: "observer"}

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


def _spool_text(spool: Path) -> str:
    """Every byte the sidecar has put in its spool directory, as text."""
    if not spool.exists():
        return ""
    return "".join(
        path.read_text("utf-8", errors="replace")
        for path in sorted(spool.rglob("*"))
        if path.is_file()
    )


def _unfiled(caplog: pytest.LogCaptureFixture) -> list[str]:
    """The records this module put on the log because they could not be stored."""
    return [record.getMessage() for record in caplog.records if "unfiled" in record.getMessage()]


def _blocked(tmp_path: Path) -> Path:
    """A spool path that is a file, not a directory — a bind that never materialised."""
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("", encoding="utf-8")
    return blocked


def _login(
    client: TestClient, user: str = "alice", password: str = ALICE_PASSWORD
) -> httpx.Response:
    """POST one credential attempt the way a browser on this deployment does."""
    return client.post(
        "/auth/login",
        data={"user": user, "next": "", "password": password},
        headers={"Origin": EXTERNAL_ORIGIN},
        follow_redirects=False,
    )


def _password_client() -> TestClient:
    return TestClient(create_app(PASSWORD_ENV), base_url="https://testserver")


def _oidc_app(
    env: dict[str, str] | None = None,
    *,
    userinfo: dict[str, Any] | None = None,
    binding: RoleBinding | None = None,
) -> FastAPI:
    """An app in OIDC mode with Authlib replaced and the role binding injected."""
    app = create_app(env if env is not None else OIDC_ENV)
    app.state.oidc_client = FakeOIDCClient(userinfo=userinfo)
    app.state.role_binding = binding if binding is not None else RoleBinding()
    return app


# --- where a record lands ---------------------------------------------------


class TestWhereARecordLands:
    """The spool variable is the whole path resolution, and the entry names the surface."""

    def test_a_login_writes_one_audit_entry_in_the_sidecars_spool(self, zone: SidecarZone) -> None:
        with _password_client() as client:
            assert _login(client).status_code == 303
        assert any(path.is_file() for path in zone.spool.iterdir())
        entries = zone.entries()
        assert len(entries) == 1
        assert entries[0].kind == KIND_AUDIT
        assert entries[0].identity == SIDECAR_IDENTITY

    def test_the_entry_carries_the_trusted_on_behalf(self, zone: SidecarZone) -> None:
        """Written under the sidecar's own identity with ``on_behalf_of`` set
        to the person logging in — and kept by the record, because the sidecar
        is on the trusted on-behalf list, where any other source's claim is
        dropped."""
        with _password_client() as client:
            assert _login(client).status_code == 303
        (entry,) = zone.entries()
        assert entry.identity == SIDECAR_IDENTITY
        assert entry.on_behalf_of == "alice"

    def test_the_entry_names_the_surface(self, zone: SidecarZone) -> None:
        with _password_client() as client:
            _login(client)
        assert zone.records()[0]["surface"] == audit.SURFACE

    def test_nothing_is_filed_in_the_state_directory(self, zone: SidecarZone) -> None:
        """The state directory holds the revoked-session file and nothing else:
        a record filed there would be a second copy outside the record."""
        with _password_client() as client:
            _login(client)
            _login(client, password="wrong")
        assert list(zone.state.iterdir()) == []

    def test_the_path_never_asks_for_a_project_root(
        self, zone: SidecarZone, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The sidecar image holds one uvicorn app and no project. Resolving a
        root here would either raise or point at a directory the container does
        not bind, so the emitter must never reach for one."""

        def _explode(*args: Any, **kwargs: Any) -> Path:
            raise AssertionError("the sidecar must not resolve a project root")

        monkeypatch.setattr("osprey_connectors.workspace.resolve_project_root", _explode)
        with _password_client() as client:
            assert _login(client).status_code == 303
        assert len(zone.records()) == 1

    def test_each_decision_is_one_entry(self, zone: SidecarZone) -> None:
        with _password_client() as client:
            _login(client, password="wrong")
            _login(client, user="bob", password="wrong")
        assert len(zone.records()) == 2

    def test_an_unset_spool_variable_degrades_to_a_log_line(
        self, zone: SidecarZone, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Without ``OSPREY_SPOOL_DIR`` the writer would have to resolve a
        project root the image does not have; the write fails, and the record
        is still readable on the log — just not durable."""

        def _no_root(*args: Any, **kwargs: Any) -> Path:
            raise RuntimeError("no project root in this image")

        monkeypatch.delenv(SPOOL_DIR_ENV, raising=False)
        monkeypatch.setattr("osprey_connectors.workspace.resolve_project_root", _no_root)
        with (
            caplog.at_level(logging.INFO, logger=audit.logger.name),
            _password_client() as client,
        ):
            assert _login(client).status_code == 303
        logged = _unfiled(caplog)
        assert len(logged) == 1
        assert "alice" in logged[0]
        assert not zone.spool.exists()

    def test_the_state_directory_is_not_needed_to_record(
        self, zone: SidecarZone, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unset ``OSPREY_STATE_DIR`` costs the revocation file its
        home, never the record its entry."""
        monkeypatch.delenv(STATE_DIR_ENV, raising=False)
        with _password_client() as client:
            assert _login(client).status_code == 303
        assert len(zone.records()) == 1


# --- what a successful login records ----------------------------------------


class TestASuccessfulLogin:
    """A login that happened is a decision the ledger has to carry too."""

    def test_it_is_recorded_as_allowed(self, zone: SidecarZone) -> None:
        with _password_client() as client:
            _login(client)
        record = zone.records()[0]
        assert record["decision"] == "allowed"
        assert record["reason"] == audit.REASON_PASSWORD_LOGIN

    def test_the_user_is_the_subject_and_the_service_is_the_actor(self, zone: SidecarZone) -> None:
        """The user did not write this record; the sidecar did, on their behalf.
        Their name is what was acted on, which is exactly why the file lives
        under the service's directory and not under theirs."""
        with _password_client() as client:
            _login(client)
        record = zone.records()[0]
        assert record["subject"] == "alice"
        assert record["actor"] == SIDECAR_IDENTITY

    def test_the_record_says_whom_the_service_decided_for(self, zone: SidecarZone) -> None:
        """Filed under the service, so the person is named in the one field a
        reader of the leading columns finds: directly after ``session``."""
        with _password_client() as client:
            _login(client)
        (entry,) = zone.entries()
        record = dict(entry.body)
        keys = list(record)
        assert record["on_behalf_of"] == "alice"
        assert keys.index("on_behalf_of") == keys.index("session") + 1
        assert entry.identity == SIDECAR_IDENTITY

    def test_the_posture_is_the_apps_own_stamp(self, zone: SidecarZone) -> None:
        """The sidecar governs entry, not writes: it holds no posture store and
        spawns no session, so it stamps its own constant and says so."""
        with _password_client() as client:
            _login(client)
        record = zone.records()[0]
        assert record["posture"] == audit.SIDECAR_POSTURE
        assert record["posture_source"] == POSTURE_SOURCE_APP

    def test_no_session_key_is_invented(self, zone: SidecarZone) -> None:
        """``session`` is the audit session id that governed the record. A
        login has none — the browser has no session yet — and the field is
        present as null rather than dropped, so every line carries one shape."""
        with _password_client() as client:
            _login(client)
        record = zone.records()[0]
        assert "session" in record
        assert record["session"] is None

    def test_the_credential_never_reaches_the_ledger(self, zone: SidecarZone) -> None:
        with _password_client() as client:
            _login(client)
            _login(client, password="a-guess-that-failed")
        stored = _spool_text(zone.spool)
        assert len(zone.records()) == 2
        assert ALICE_PASSWORD not in stored
        assert "a-guess-that-failed" not in stored

    def test_an_oidc_login_records_the_role_it_resolved(self, zone: SidecarZone) -> None:
        app = _oidc_app(
            userinfo={"sub": ALICE_SUBJECT, GROUP_CLAIM: [OPERATOR_GROUP]},
            binding=RoleBinding(claim=GROUP_CLAIM, claim_map=CLAIM_MAP),
        )
        assert callback(app).status_code == 303
        record = zone.records()[0]
        assert record["decision"] == "allowed"
        assert record["reason"] == audit.REASON_OIDC_LOGIN
        assert record["subject"] == "alice"
        assert record["role"] == "operator"

    def test_a_roleless_deployment_records_no_role(self, zone: SidecarZone) -> None:
        """Deny-safe by absence here too: no role means the key is omitted, not
        blank, so nobody reads an empty string as a role named ''."""
        assert callback(_oidc_app()).status_code == 303
        assert "role" not in zone.records()[0]

    def test_a_detail_is_carried_when_given(self, zone: SidecarZone) -> None:
        """A shared-card login names its opener in ``detail`` — identifiers
        only, on the same terms as a refusal's detail."""
        audit.record_login_success(user="alice", method="password", detail="opener=alice")
        assert zone.records()[0]["detail"] == "opener=alice"

    def test_no_detail_means_the_key_is_omitted(self, zone: SidecarZone) -> None:
        """Absent, not null: a record with nothing to add carries no key, so a
        reader never has to distinguish ``null`` from missing."""
        audit.record_login_success(user="alice", method="password")
        assert "detail" not in zone.records()[0]


# --- what a refused login records -------------------------------------------


class TestARefusedPasswordLogin:
    """One category for every way a password attempt fails to unlock a user."""

    @pytest.mark.parametrize(
        ("user", "password"),
        [
            ("alice", "not-alices-password"),  # a real user, a wrong credential
            ("bob", ALICE_PASSWORD),  # a roster user with no provisioned hash
            ("carol", ALICE_PASSWORD),  # not on the roster at all
            ("a" * 200, ALICE_PASSWORD),  # too long to be either
        ],
    )
    def test_every_refusal_is_recorded(self, zone: SidecarZone, user: str, password: str) -> None:
        with _password_client() as client:
            assert _login(client, user=user, password=password).status_code == 401
        record = zone.records()[0]
        assert record["decision"] == "refused"
        assert record["reason"] == audit.REASON_BAD_CREDENTIAL

    def test_the_ledger_does_not_say_which_users_exist(self, zone: SidecarZone) -> None:
        """The anti-lookup invariant reaches the ledger as well as the page: a
        wrong password, an unprovisioned roster user and a name that was never
        on the roster are one category, so a reader of the file cannot use it
        to enumerate accounts either."""
        with _password_client() as client:
            _login(client, user="alice", password="wrong")
            _login(client, user="bob", password="wrong")
            _login(client, user="carol", password="wrong")
        assert {record["reason"] for record in zone.records()} == {audit.REASON_BAD_CREDENTIAL}

    def test_the_recorded_subject_is_bounded(self, zone: SidecarZone) -> None:
        """The name the ledger names is the bounded one the page echoes, never
        the caller-sized string: neither the page nor the ledger is sized by
        whoever posted the form."""
        with _password_client() as client:
            _login(client, user="a" * 500, password="wrong")
        assert len(zone.records()[0]["subject"]) <= 256

    def test_an_unevaluated_attempt_is_not_recorded(self, zone: SidecarZone) -> None:
        """A second attempt inside an open throttle window is refused without
        the credential ever being evaluated, and the ledger says nothing about
        it. Recording it would let an unauthenticated caller append to the
        audit trail at request rate, and the decision it would describe — this
        user's credential did not verify — is already on the line above."""
        with _password_client() as client:
            assert _login(client, password="wrong").status_code == 401
            assert _login(client, password="wrong").status_code == 429
        assert len(zone.records()) == 1


SHARED_ENV = {**PASSWORD_ENV, "OSPREY_AUTH_ROSTER_ACCESS_BOB": "any"}
"""The password roster with bob's card opened to the whole roster, so a shared
login's ledger records can be pinned: alice's credential opens bob's card."""


def _shared_login(
    client: TestClient, opener: str, password: str = ALICE_PASSWORD
) -> httpx.Response:
    """POST one shared-card attempt against bob's card."""
    return client.post(
        "/auth/login",
        data={"user": "bob", "username": opener, "next": "", "password": password},
        headers={"Origin": EXTERNAL_ORIGIN},
        follow_redirects=False,
    )


class TestASharedCardInTheLedger:
    """A shared card's records are the card's, with the opener in ``detail``.

    The subject stays the card — that is whose terminal was entered — and the
    opener rides in ``detail`` as an identifier, so "who is in bob's terminal,
    and whose credential put them there" are both one line's answer.
    """

    def test_a_success_names_its_opener(self, zone: SidecarZone) -> None:
        with TestClient(create_app(SHARED_ENV), base_url="https://testserver") as client:
            assert _shared_login(client, "alice").status_code == 303
        record = zone.records()[0]
        assert record["decision"] == "allowed"
        assert record["reason"] == audit.REASON_PASSWORD_LOGIN
        assert record["subject"] == "bob"
        assert record["detail"] == "opener=alice"

    def test_a_refusal_names_its_opener(self, zone: SidecarZone) -> None:
        """Same category as every other bad credential — the record must not
        say whether the opener exists any more than the page does — but the
        bounded opener name is carried so the ledger can answer who was
        knocking."""
        with TestClient(create_app(SHARED_ENV), base_url="https://testserver") as client:
            assert _shared_login(client, "mally", password="a-guess").status_code == 401
        record = zone.records()[0]
        assert record["decision"] == "refused"
        assert record["reason"] == audit.REASON_BAD_CREDENTIAL
        assert record["subject"] == "bob"
        assert record["detail"] == "opener=mally"

    def test_an_empty_opener_refusal_carries_no_detail(self, zone: SidecarZone) -> None:
        """An empty opener is refused like any other miss, but ``opener=`` with
        nothing after it would put an empty value in an identifiers-only field
        — so that refusal carries no detail key at all."""
        with TestClient(create_app(SHARED_ENV), base_url="https://testserver") as client:
            assert _shared_login(client, "", password="a-guess").status_code == 401
        record = zone.records()[0]
        assert record["decision"] == "refused"
        assert record["reason"] == audit.REASON_BAD_CREDENTIAL
        assert "detail" not in record


BROKEN_HASH = "scrypt.16384.8.1.c2FsdA"
"""A stored hash cut to five fields: provisioned, and impossible to evaluate."""

BROKEN_ALICE_ENV = {**PASSWORD_ENV, "OSPREY_AUTH_PW_HASH_ALICE": BROKEN_HASH}


class TestAnUnevaluableCredential:
    """A provisioned credential the service cannot read is a configuration fault,
    recorded under its own category and never quoted."""

    def test_it_is_recorded_under_its_own_category(self, zone: SidecarZone) -> None:
        with TestClient(create_app(BROKEN_ALICE_ENV), base_url="https://testserver") as client:
            assert _login(client, user="alice", password=ALICE_PASSWORD).status_code == 401
        records = zone.records()
        assert len(records) == 1
        assert records[0]["decision"] == "refused"
        assert records[0]["reason"] == audit.REASON_CREDENTIAL_UNEVALUABLE
        assert records[0]["subject"] == "alice"

    def test_the_stored_value_never_reaches_the_ledger(self, zone: SidecarZone) -> None:
        with TestClient(create_app(BROKEN_ALICE_ENV), base_url="https://testserver") as client:
            _login(client, user="alice", password=ALICE_PASSWORD)
        text = _spool_text(zone.spool)
        assert len(zone.records()) == 1
        assert "c2FsdA" not in text
        assert "scrypt.16384" not in text

    def test_a_well_formed_neighbour_keeps_the_ordinary_category(self, zone: SidecarZone) -> None:
        with TestClient(create_app(BROKEN_ALICE_ENV), base_url="https://testserver") as client:
            assert _login(client, user="bob", password="wrong").status_code == 401
            assert _login(client, user="carol", password="wrong").status_code == 401
        assert [record["reason"] for record in zone.records()] == [
            audit.REASON_BAD_CREDENTIAL,
            audit.REASON_BAD_CREDENTIAL,
        ]

    @pytest.mark.usefixtures("zone")
    def test_the_log_names_the_fix(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.ERROR, logger="osprey.services.auth_sidecar.routes.login")
        with TestClient(create_app(BROKEN_ALICE_ENV), base_url="https://testserver") as client:
            _login(client, user="alice", password=ALICE_PASSWORD)
        assert "osprey users passwd alice" in caplog.text
        assert "c2FsdA" not in caplog.text

    def test_a_shared_card_names_its_opener(self, zone: SidecarZone) -> None:
        env = {**SHARED_ENV, "OSPREY_AUTH_PW_HASH_ALICE": BROKEN_HASH}
        with TestClient(create_app(env), base_url="https://testserver") as client:
            assert _shared_login(client, "alice").status_code == 401
        record = zone.records()[0]
        assert record["reason"] == audit.REASON_CREDENTIAL_UNEVALUABLE
        assert record["subject"] == "bob"
        assert record["detail"] == "opener=alice"


class TestARefusedOidcLogin:
    """Every category the OIDC path refuses under reaches the same ledger."""

    def test_an_unmapped_user_is_recorded(self, zone: SidecarZone) -> None:
        env = {key: value for key, value in OIDC_ENV.items() if "SUBJECT_ALICE" not in key}
        with TestClient(_oidc_app(env), base_url=EXTERNAL_ORIGIN) as client:
            response = client.get(
                "/auth/oidc/login", params={"user": "alice"}, follow_redirects=False
            )
        assert response.status_code == 403
        assert zone.records()[0]["reason"] == REASON_UNMAPPED_USER

    def test_a_mapped_identity_that_cannot_be_carried_is_recorded(self, zone: SidecarZone) -> None:
        env = {**OIDC_ENV, "OSPREY_AUTH_OIDC_SUBJECT_ALICE": "jörg@example.org"}
        assert callback(_oidc_app(env)).status_code == 403
        assert zone.records()[0]["reason"] == audit.REASON_NON_ASCII_SUBJECT

    def test_an_absent_asserted_identity_is_recorded(self, zone: SidecarZone) -> None:
        assert callback(_oidc_app(userinfo={})).status_code == 403
        assert zone.records()[0]["reason"] == REASON_NO_ASSERTED_IDENTITY

    def test_an_identity_belonging_to_another_user_is_recorded(self, zone: SidecarZone) -> None:
        assert callback(_oidc_app(userinfo={"sub": BOB_SUBJECT})).status_code == 403
        assert zone.records()[0]["reason"] == REASON_IDENTITY_MISMATCH

    @pytest.mark.parametrize(
        ("groups", "claim_map", "reason"),
        [
            (None, CLAIM_MAP, REASON_MISSING_ROLE_CLAIM),
            ([OBSERVER_GROUP + "-elsewhere"], CLAIM_MAP, REASON_UNMAPPED_ROLE_CLAIM),
            ([OPERATOR_GROUP, OBSERVER_GROUP], CLAIM_MAP, REASON_AMBIGUOUS_ROLE_CLAIM),
            ([OPERATOR_GROUP], {OPERATOR_GROUP: "öperator"}, REASON_UNSAFE_ROLE),
        ],
    )
    def test_every_role_category_is_recorded(
        self,
        zone: SidecarZone,
        groups: list[str] | None,
        claim_map: dict[str, str],
        reason: str,
    ) -> None:
        userinfo: dict[str, Any] = {"sub": ALICE_SUBJECT}
        if groups is not None:
            userinfo[GROUP_CLAIM] = groups
        app = _oidc_app(
            userinfo=userinfo, binding=RoleBinding(claim=GROUP_CLAIM, claim_map=claim_map)
        )
        assert callback(app).status_code == 403
        record = zone.records()[0]
        assert record["decision"] == "refused"
        assert record["reason"] == reason
        assert record["subject"] == "alice"

    def test_exactly_one_record_per_refusal(self, zone: SidecarZone) -> None:
        """The refusal seam is one call wide: no category records twice, and a
        refused login never also files the success record."""
        assert callback(_oidc_app(userinfo={"sub": BOB_SUBJECT})).status_code == 403
        assert len(zone.records()) == 1


class TestAnOidcLoginResolvedFromSeveralRoles:
    """A login whose token mapped to several roles says which ones in its record."""

    @staticmethod
    def _app(groups: list[str]) -> FastAPI:
        app = _oidc_app(
            userinfo={"sub": ALICE_SUBJECT, GROUP_CLAIM: groups},
            binding=RoleBinding(claim=GROUP_CLAIM, claim_map=CLAIM_MAP),
        )
        app.state.roster_roles = RosterRoles({"alice": "observer"})
        return app

    def test_the_success_record_names_the_mapped_roles(self, zone: SidecarZone) -> None:
        assert callback(self._app([OPERATOR_GROUP, OBSERVER_GROUP])).status_code == 303
        records = zone.records()
        assert len(records) == 1
        assert records[0]["decision"] == "allowed"
        assert records[0]["role"] == "observer"
        assert records[0]["detail"] == "mapped_roles=observer,operator"

    def test_one_mapped_role_records_no_detail(self, zone: SidecarZone) -> None:
        assert callback(self._app([OBSERVER_GROUP])).status_code == 303
        assert "detail" not in zone.records()[0]


# --- the emitter never costs the decision -----------------------------------


class TestTheEmitterNeverCostsTheDecision:
    """An audit gap must never become an outage, nor a slower login."""

    @pytest.mark.usefixtures("zone")
    def test_an_unwritable_spool_still_lets_a_login_succeed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The spool variable points at something that is not a directory — a
        bind that never materialised. The login is still a login."""
        monkeypatch.setenv(SPOOL_DIR_ENV, str(_blocked(tmp_path)))
        with _password_client() as client:
            assert _login(client).status_code == 303

    @pytest.mark.usefixtures("zone")
    def test_an_unwritable_spool_still_refuses(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(SPOOL_DIR_ENV, str(_blocked(tmp_path)))
        assert callback(_oidc_app(userinfo={"sub": BOB_SUBJECT})).status_code == 403

    @pytest.mark.skipif(
        hasattr(os, "geteuid") and os.geteuid() == 0, reason="root writes a read-only directory"
    )
    def test_an_unwritable_spool_does_not_delay_the_login(self, zone: SidecarZone) -> None:
        """The write is one local append with no retry and no wait, so a spool
        that refuses it answers as fast as one that takes it: the login's time
        is the login's, never the audit trail's."""

        def _timed_login() -> float:
            with _password_client() as client:
                started = time.monotonic()
                assert _login(client).status_code == 303
                return time.monotonic() - started

        _timed_login()  # warm the app and the writer before anything is measured
        usual = min(_timed_login() for _ in range(3))
        record_writer.seal_all()
        zone.spool.chmod(0o500)
        try:
            unwritable = min(_timed_login() for _ in range(3))
            assert record_writer.write(KIND_AUDIT, {"probe": True}) is None
        finally:
            zone.spool.chmod(0o700)
        assert unwritable <= usual + 0.25

    def test_a_failed_write_is_reported_to_health(
        self, zone: SidecarZone, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The writer appends the failure to the spool directory's ``.errors``,
        which the record reports on its health, and the login goes on."""

        def _explode(_directory: Path, _data: bytes) -> None:
            raise OSError("the spool is full")

        monkeypatch.setattr(record_writer, "_append", _explode)
        with _password_client() as client:
            assert _login(client).status_code == 303
        errors = (zone.spool / WRITER_ERRORS_FILE).read_text("utf-8").splitlines()
        assert len(errors) == 1
        assert json.loads(errors[0])["kind"] == KIND_AUDIT
        assert "the spool is full" in json.loads(errors[0])["error"]

    def test_a_missing_spool_directory_is_created(self, zone: SidecarZone) -> None:
        """The deploy path provisions the spool directory; the writer creating
        it is the laptop and first-run case."""
        assert not zone.spool.exists()
        with _password_client() as client:
            assert _login(client).status_code == 303
        assert len(zone.records()) == 1

    @pytest.mark.usefixtures("zone")
    def test_a_seam_that_raises_costs_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _explode(*args: Any, **kwargs: Any) -> None:
            raise OSError("the audit zone is gone")

        monkeypatch.setattr(audit, "write_envelope", _explode)
        with _password_client() as client:
            assert _login(client).status_code == 303
        assert callback(_oidc_app(userinfo={"sub": BOB_SUBJECT})).status_code == 403

    def test_a_record_that_cannot_be_built_costs_nothing(self, zone: SidecarZone) -> None:
        """An empty subject is a programming error in a caller, not a reason to
        raise into the decision it describes."""
        audit.record_login_refusal(user="", reason=audit.REASON_BAD_CREDENTIAL)
        audit.record_login_success(user="", method="password")
        assert zone.records() == []


# --- the shared writer does the writing -------------------------------------


class TestTheSharedWriterDoesTheWriting:
    """The sidecar supplies the envelope; the one record writer supplies the
    body shaping and the append, so neither is re-implemented one service over."""

    def test_the_reused_seam_exists(self) -> None:
        """A rename in the writer must fail here, not in production."""
        assert callable(writer.record_envelope)
        assert callable(record_writer.write)

    @pytest.mark.usefixtures("zone")
    def test_the_write_goes_through_the_record_writer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        written: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

        def _capture(kind: str, body: Any, **ids: Any) -> str:
            written.append((kind, dict(body), ids))
            return "entry-1"

        monkeypatch.setattr(record_writer, "write", _capture)
        with _password_client() as client:
            _login(client)
        assert len(written) == 1
        kind, body, ids = written[0]
        assert kind == KIND_AUDIT
        assert body["subject"] == "alice"
        assert ids["on_behalf_of"] == "alice"

    @pytest.mark.usefixtures("zone")
    def test_the_seam_returns_the_entry_id(self) -> None:
        envelope = AuditEnvelope(
            surface=audit.SURFACE,
            actor=SIDECAR_IDENTITY,
            posture=audit.SIDECAR_POSTURE,
            posture_source=POSTURE_SOURCE_APP,
            session=None,
            on_behalf_of="alice",
            subject="alice",
            decision="allowed",
            reason=audit.REASON_PASSWORD_LOGIN,
        )
        entry_id = audit.write_envelope(envelope)
        assert isinstance(entry_id, str)
        assert entry_id


# --- the degrade ladder has no missing rung ---------------------------------


class TestTheDegradeLadderIsMonotone:
    """*File, else log* — for every way the write can fail, not just the easy one.

    The failure that is actually likely in a deployment — a bind that mounted
    read-only, a full disk, a directory this process cannot open — must not
    erase the record: whatever rung gave way, the record reaches the log.
    """

    @pytest.mark.usefixtures("zone")
    def test_an_unwritable_spool_still_logs_the_record(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setenv(SPOOL_DIR_ENV, str(_blocked(tmp_path)))
        with (
            caplog.at_level(logging.INFO, logger=audit.logger.name),
            _password_client() as client,
        ):
            assert _login(client).status_code == 303
        unfiled = _unfiled(caplog)
        assert len(unfiled) == 1
        stored = json.loads(unfiled[0].split("unfiled): ", 1)[1])
        assert stored["subject"] == "alice"
        assert stored["decision"] == "allowed"
        assert stored["reason"] == audit.REASON_PASSWORD_LOGIN

    @pytest.mark.usefixtures("zone")
    def test_a_refusal_in_an_unwritable_spool_keeps_its_category(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The category is the half a human cannot reconstruct from the route's
        own log line, which says only that the login was refused."""
        monkeypatch.setenv(SPOOL_DIR_ENV, str(_blocked(tmp_path)))
        with (
            caplog.at_level(logging.INFO, logger=audit.logger.name),
            _password_client() as client,
        ):
            assert _login(client, password="wrong").status_code == 401
        unfiled = _unfiled(caplog)
        assert len(unfiled) == 1
        assert json.loads(unfiled[0].split("unfiled): ", 1)[1])["reason"] == (
            audit.REASON_BAD_CREDENTIAL
        )

    @pytest.mark.usefixtures("zone")
    def test_the_record_off_switch_logs_the_record_too(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The quietest failure: nothing raises and nothing is stored."""
        monkeypatch.setenv(RECORD_OFF_ENV, RECORD_OFF_VALUE)
        with (
            caplog.at_level(logging.INFO, logger=audit.logger.name),
            _password_client() as client,
        ):
            assert _login(client).status_code == 303
        assert _unfiled(caplog)

    @pytest.mark.usefixtures("zone")
    def test_an_append_that_raises_logs_the_record_too(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        def _explode(_directory: Path, _data: bytes) -> None:
            raise OSError("the spool is gone")

        monkeypatch.setattr(record_writer, "_append", _explode)
        with (
            caplog.at_level(logging.INFO, logger=audit.logger.name),
            _password_client() as client,
        ):
            assert _login(client).status_code == 303
        assert _unfiled(caplog)


class TestThePathTheVariableNames:
    """What the state-directory variable may hold, and what it never changes."""

    def test_a_relative_state_directory_is_refused_and_costs_no_record(
        self, zone: SidecarZone, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Resolved against the process's working directory it would name a path
        inside the image that the host binds nothing at; it is refused like a
        blank one, and the record does not depend on it."""
        monkeypatch.setenv(STATE_DIR_ENV, "relative/audit")
        monkeypatch.chdir(tmp_path)
        assert state_directory() is None
        with _password_client() as client:
            assert _login(client).status_code == 303
        assert len(zone.records()) == 1
        assert not (tmp_path / "relative").exists()

    def test_a_dot_relative_directory_is_refused_too(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(STATE_DIR_ENV, "./audit")
        monkeypatch.chdir(tmp_path)
        assert state_directory() is None

    def test_state_directory_reads_the_mapping_it_is_given(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An explicit mapping is the whole source: the process environment is
        read only when no mapping is given."""
        monkeypatch.setenv(STATE_DIR_ENV, str(tmp_path / "process"))
        given = tmp_path / "given"
        assert state_directory({STATE_DIR_ENV: f"  {given}  "}) == given
        assert state_directory({}) is None
        assert state_directory({STATE_DIR_ENV: "   "}) is None
        assert state_directory() == tmp_path / "process"

    def test_state_directory_refuses_a_relative_value(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=revocation.logger.name):
            assert state_directory({STATE_DIR_ENV: "relative/audit"}) is None
        assert [
            record.getMessage() for record in caplog.records if STATE_DIR_ENV in record.getMessage()
        ]

    def test_the_writer_marker_cannot_move_this_record(
        self, zone: SidecarZone, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A stray ``OSPREY_AUDIT_WRITER`` in this environment changes nothing:
        the entry is still the sidecar's, on the sidecar's surface."""
        monkeypatch.setenv("OSPREY_AUDIT_WRITER", "maintenance")
        with _password_client() as client:
            assert _login(client).status_code == 303
        (entry,) = zone.entries()
        assert entry.identity == SIDECAR_IDENTITY
        assert entry.body["surface"] == audit.SURFACE


class TestTheCategorySetIsClosed:
    """One place enumerates what an operator may find in ``reason``."""

    def test_every_category_this_service_files_is_in_the_set(self) -> None:
        named = {
            value
            for name, value in vars(audit).items()
            if name.startswith("REASON_") and isinstance(value, str)
        }
        assert named <= audit.LOGIN_REASONS
        # Categories no route files yet are pinned here by name, so the
        # vars() sweep above is not their only tie to the closed set.
        assert audit.REASON_AMBIGUOUS_IDENTITY in audit.LOGIN_REASONS
        for reason in RULE_ADMISSION_REASONS:
            assert reason in audit.LOGIN_REASONS

    def test_every_category_is_exported_under_its_own_name(self) -> None:
        """``__all__`` is checked against the categories in both directions.

        The closed set is the vocabulary an operator reads off ``reason``; the
        export list is how the routes reach the spellings. A category in one
        and not the other is a category that either cannot be named at a call
        site or is named there under a symbol nothing documents.
        """
        defined = {
            name
            for name, value in vars(audit).items()
            if name.startswith("REASON_") and isinstance(value, str)
        }
        exported = {
            name
            for name in audit.__all__
            if name.startswith("REASON_") and isinstance(getattr(audit, name), str)
        }
        assert exported == defined

    def test_the_sign_in_subject_can_never_be_a_roster_name(self) -> None:
        """The card-less subject sits outside the roster charset, so a record
        filed under it can never be read as one about a roster user."""
        assert not USERNAME_CHARSET_RE.match(audit.SIGN_IN_SUBJECT)

    def test_the_oidc_routes_name_the_same_categories(self) -> None:
        """The route module keeps its own spellings for readability at the point
        of decision; they must be the very same strings, not a second copy."""
        from osprey.services.auth_sidecar.routes import oidc as oidc_routes

        for name, value in vars(oidc_routes).items():
            if name.startswith("REASON_") and isinstance(value, str):
                assert value in audit.LOGIN_REASONS, name


RULE_ADMISSION_REASONS = (
    audit.REASON_HOSTED_DOMAIN_MISMATCH,
    audit.REASON_NO_COVERING_PRINCIPAL,
    audit.REASON_UNSAFE_ASSERTED_IDENTITY,
    audit.REASON_UNVERIFIED_EMAIL,
)
"""The categories a card's principal set refuses a login under."""


class TestTheRuleAdmissionCategories:
    """The four categories a principal-set refusal is filed under.

    Pinned by behaviour and not only by the closed-set sweep above, because
    each one exists to be *distinguishable* in the ledger: an operator counting
    them is deciding whether to widen a card's principal set, fix a claim their
    IdP emits, or leave a refusal alone. A category that files under a
    neighbour's spelling answers none of those.
    """

    @pytest.mark.parametrize("reason", RULE_ADMISSION_REASONS)
    def test_each_category_reaches_the_ledger_under_its_own_name(
        self, zone: SidecarZone, reason: str
    ) -> None:
        audit.record_login_refusal(user="bob", reason=reason, detail="hd")
        stored = zone.records()
        assert len(stored) == 1
        assert stored[0]["decision"] == "refused"
        assert stored[0]["reason"] == reason
        assert stored[0]["subject"] == "bob"
        assert stored[0]["detail"] == "hd"

    @pytest.mark.parametrize("reason", RULE_ADMISSION_REASONS)
    def test_no_category_is_a_spelling_of_another(self, reason: str) -> None:
        """Four distinct strings, so a reader can tell the four apart."""
        assert sum(other == reason for other in RULE_ADMISSION_REASONS) == 1

    @pytest.mark.parametrize("reason", RULE_ADMISSION_REASONS)
    def test_the_detail_is_stored_as_the_identifier_the_caller_passed(
        self, zone: SidecarZone, reason: str
    ) -> None:
        """``detail`` is echoed verbatim, so what a caller passes is what an
        operator reads. The discipline that a caller passes the claim's NAME
        and never the address or domain the token carried is the caller's to
        keep; the refusal sites pin it where the claim values enter.
        """
        audit.record_login_refusal(user="bob", reason=reason, detail="email_verified")
        assert zone.records()[0]["detail"] == "email_verified"


class TestTheWriterShapesTheLine:
    """The body shaping is the half the reuse exists for, so it is pinned by
    BEHAVIOUR rather than by the symbol's existence.

    :func:`osprey.audit.writer.record_envelope` carries the record bound, the
    oversize-``detail`` replacement, and the identifiers-only escape hatch. A
    local re-implementation that still reached the record writer would pass a
    test that only asserted the name is callable — and would not show up until
    a long ``detail`` (the ambiguity category joins role names into one) grew
    an entry past its bound in production.
    """

    def test_an_over_budget_detail_is_replaced_by_the_writers_marker(
        self, zone: SidecarZone, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(TERMINAL_USER_ENV, "a" * 300)
        audit.record_login_refusal(
            user="u" * 300, reason="r" * 300, detail="d" * 4096, role="R" * 300
        )
        stored = zone.records()
        assert len(stored) == 1
        assert stored[0]["detail"] == writer.DETAIL_DROPPED

    def test_the_stored_body_stays_inside_the_record_bound(
        self, zone: SidecarZone, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The property the replacement buys: one record stays a bounded
        entry, whatever the caller handed it."""
        monkeypatch.setenv(TERMINAL_USER_ENV, "a" * 300)
        audit.record_login_refusal(
            user="u" * 300, reason="r" * 300, detail="d" * 4096, role="R" * 300
        )
        (stored,) = zone.records()
        assert len(writer._encode(stored)) <= writer.MAX_RECORD_BYTES

    def test_an_ordinary_detail_is_stored_whole(self, zone: SidecarZone) -> None:
        """The control: the degrade is for records that do not fit, not for
        every record that has a detail."""
        audit.record_login_refusal(
            user="alice", reason=audit.REASON_MISSING_ROLE_CLAIM, detail="groups"
        )
        assert zone.records()[0]["detail"] == "groups"


# --- what bounds the ledger's growth ----------------------------------------


UNMAPPED_OIDC_ENV = {**OIDC_ENV, "OSPREY_AUTH_OIDC_SUBJECT_BOB": ""}
"""bob is on the roster and has no mapped IdP identity: every login for him is
refused BEFORE the token exchange, which is what makes those refusals free for
an unauthenticated caller to generate."""

UNCARRIABLE_OIDC_ENV = {**OIDC_ENV, "OSPREY_AUTH_OIDC_SUBJECT_BOB": "idp|böb"}
"""The *other* pre-exchange refusal, and the same cost to reach.

bob's mapped identity is spelled in a way the ASCII-only identity header cannot
carry, so the callback refuses it before the token exchange too — a
configuration fault rather than a missing mapping, and its own audited
category."""


def _oidc_start(app: FastAPI, user: str) -> httpx.Response:
    """One unauthenticated GET of the OIDC login route — no cookie, no code."""
    with TestClient(app, base_url=EXTERNAL_ORIGIN) as client:
        return client.get("/auth/oidc/login", params={"user": user}, follow_redirects=False)


class TestThePreExchangeRefusalsAreBounded:
    """The refusals reachable before the token exchange cost one GET apiece.

    The ledger is root-owned in a directory nothing else in the deployment
    binds, so nobody inside it can rotate or truncate the file: unbounded append
    at request rate is disk pressure against the audit zone itself, and it
    buries the genuine refusals an operator greps for. These refusals therefore
    run through the same attempt throttle that bounds ``bad_credential`` on the
    password path — the ANSWER is unchanged every time, only the number of times
    the ledger repeats it is bounded.
    """

    def test_a_flood_of_login_starts_appends_once(self, zone: SidecarZone) -> None:
        app = _oidc_app(UNMAPPED_OIDC_ENV)
        # The flood has to land in ONE window for "appends once" to be the claim
        # being made, and the window is a real duration the throttle reads off a
        # clock. Held still, so what is asserted is the fold and not how long
        # fifty requests happened to take on the machine running them.
        _with_movable_audit_window(app)
        for _ in range(50):
            assert _oidc_start(app, "bob").status_code == 403
        assert len(zone.records()) == 1
        assert zone.records()[0]["reason"] == REASON_UNMAPPED_USER

    @pytest.mark.usefixtures("zone")
    def test_the_refusal_itself_is_unchanged(self) -> None:
        """Bounding the record must not bound the decision: the caller gets the
        same 403 and the same message however many times they ask."""
        app = _oidc_app(UNMAPPED_OIDC_ENV)
        answers = {
            (response.status_code, response.json()["detail"])
            for response in (_oidc_start(app, "bob") for _ in range(5))
        }
        assert len(answers) == 1

    @pytest.mark.parametrize(
        ("env", "reason"),
        [
            (UNMAPPED_OIDC_ENV, REASON_UNMAPPED_USER),
            (UNCARRIABLE_OIDC_ENV, audit.REASON_NON_ASCII_SUBJECT),
        ],
        ids=["unmapped_user", "non_ascii_subject"],
    )
    def test_the_callbacks_pre_exchange_refusals_are_bounded_too(
        self, zone: SidecarZone, env: dict[str, str], reason: str
    ) -> None:
        """Reached at two requests per record without the bound — the state
        cookie that gets here is itself minted by a free GET.

        Both pre-exchange arms, because they are two separate ``bound=`` at two
        separate call sites: the misconfigured-subject deployment is exactly as
        cheap to drive as the unmapped-user one, and pinning only the first left
        the second free to lose its bound unnoticed.
        """
        app = _oidc_app(env)
        # Held still, so what is asserted is the fold and not how long twenty
        # callbacks happened to take on the machine running them.
        _with_movable_audit_window(app)
        for _ in range(20):
            assert callback(app, user="bob").status_code == 403
        assert [record["reason"] for record in zone.records()] == [reason]

    def test_each_user_has_their_own_window(self, zone: SidecarZone) -> None:
        """Keyed on the roster user, like the password path's window, so one
        name's refusals cannot silence another's."""
        env = {**OIDC_ENV, "OSPREY_AUTH_OIDC_SUBJECT_ALICE": "", "OSPREY_AUTH_OIDC_SUBJECT_BOB": ""}
        app = _oidc_app(env)
        _oidc_start(app, "alice")
        _oidc_start(app, "bob")
        assert [record["subject"] for record in zone.records()] == ["alice", "bob"]

    def test_a_post_exchange_refusal_is_recorded_every_time(self, zone: SidecarZone) -> None:
        """No throttle there, deliberately: reaching one costs a full IdP round
        trip, so it is not free to generate, and it describes a login that
        actually authenticated."""
        app = _oidc_app(userinfo={"sub": BOB_SUBJECT})
        for _ in range(3):
            assert callback(app, user="alice").status_code == 403
        assert [record["reason"] for record in zone.records()] == [REASON_IDENTITY_MISMATCH] * 3


class _MovableClock:
    """A monotonic-shaped clock a test can step without sleeping."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _with_movable_audit_window(app: FastAPI) -> _MovableClock:
    """Replace the app's ledger window with one whose clock the test drives."""
    clock = _MovableClock()
    app.state.audit_throttle = AttemptThrottle(clock=clock)
    return clock


class TestTheLedgerWindowIsNotTheLoginThrottle:
    """The window that bounds the FILE is a different object from the one that
    delays a LOGIN, and it has to be.

    ``throttle.py`` states the rule in writing: only attempts that were actually
    evaluated may grow the login window, because growing it on unevaluated ones
    is the ratchet that shuts a legitimate operator out. Every refusal bounded
    here is unevaluated by construction — they are reachable *before* the token
    exchange, which is exactly what makes them free to generate. Sharing one
    object would therefore mean one instance under two opposite rules, and an
    unauthenticated caller spamming ``/auth/oidc/login?user=<roster user>``
    would be delaying that named operator's real login.
    """

    def test_the_app_builds_two_separate_windows(self) -> None:
        state = create_app(OIDC_ENV).state
        assert isinstance(state.audit_throttle, AttemptThrottle)
        assert state.audit_throttle is not state.attempt_throttle

    def test_an_unconfigured_app_builds_neither(self) -> None:
        assert create_app({}).state.audit_throttle is None

    @pytest.mark.usefixtures("zone")
    def test_unevaluated_refusals_never_touch_the_login_window(self) -> None:
        """The reviewer's measurement, inverted: eight free GETs used to leave
        ``retry_after('bob') > 0`` on the window that decides logins."""
        app = _oidc_app(UNMAPPED_OIDC_ENV)
        for _ in range(8):
            assert _oidc_start(app, "bob").status_code == 403
        login_window = get_attempt_throttle(SimpleNamespace(app=app))
        assert login_window.retry_after("bob") == 0.0
        assert len(login_window) == 0

    def test_a_paced_caller_cannot_walk_the_login_window_to_the_cap(
        self, zone: SidecarZone
    ) -> None:
        """The version that survives pacing: each refusal that is actually filed
        would have grown the shared window one more step toward its 30 s cap."""
        app = _oidc_app(UNMAPPED_OIDC_ENV)
        clock = _with_movable_audit_window(app)
        for _ in range(10):
            assert _oidc_start(app, "bob").status_code == 403
            clock.now += 60.0
        assert len(zone.records()) == 10
        login_window = get_attempt_throttle(SimpleNamespace(app=app))
        assert login_window.retry_after("bob") == 0.0
        assert len(login_window) == 0

    def test_the_ledger_window_is_what_grew_instead(self, zone: SidecarZone) -> None:
        """The bound still holds — it is simply held by the other object."""
        app = _oidc_app(UNMAPPED_OIDC_ENV)
        for _ in range(8):
            _oidc_start(app, "bob")
        assert len(zone.records()) == 1
        assert get_audit_throttle(SimpleNamespace(app=app)).retry_after("bob") > 0


class TestAFoldedRefusalIsCountable:
    """Suppressing the record must not suppress the fact that it happened.

    While a user's window is held open, genuine refusals for that name are not
    filed and their CATEGORY is lost — a stream of ``unmapped_user`` masks a
    real ``non_ascii_subject`` for the same user. The bound is still the right
    trade against unbounded append to a root-owned, unrotatable file, but the
    ledger says how much it swallowed, so a reader can tell one refusal from
    four hundred.
    """

    def test_the_next_record_names_how_many_were_folded(self, zone: SidecarZone) -> None:
        app = _oidc_app(UNMAPPED_OIDC_ENV)
        clock = _with_movable_audit_window(app)
        for _ in range(5):
            assert _oidc_start(app, "bob").status_code == 403
        clock.now += 60.0
        assert _oidc_start(app, "bob").status_code == 403

        records = zone.records()
        assert len(records) == 2
        assert records[0].get("detail") is None
        assert records[1]["detail"] == f"{FOLDED_DETAIL_KEY}=4"

    def test_a_record_that_folded_nothing_carries_no_note(self, zone: SidecarZone) -> None:
        """The key appears only where something WAS suppressed, so an ordinary
        record reads exactly as it did before."""
        app = _oidc_app(UNMAPPED_OIDC_ENV)
        clock = _with_movable_audit_window(app)
        assert _oidc_start(app, "bob").status_code == 403
        clock.now += 60.0
        assert _oidc_start(app, "bob").status_code == 403

        assert [record.get("detail") for record in zone.records()] == [None, None]

    def test_the_count_restarts_after_it_is_filed(self, zone: SidecarZone) -> None:
        """Each record names the folds since the PREVIOUS one, so the counts
        partition the stream rather than accumulating."""
        app = _oidc_app(UNMAPPED_OIDC_ENV)
        clock = _with_movable_audit_window(app)
        for _ in range(3):
            _oidc_start(app, "bob")
        clock.now += 60.0
        for _ in range(2):
            _oidc_start(app, "bob")
        clock.now += 60.0
        _oidc_start(app, "bob")

        assert [record.get("detail") for record in zone.records()] == [
            None,
            f"{FOLDED_DETAIL_KEY}=2",
            f"{FOLDED_DETAIL_KEY}=1",
        ]

    def test_each_user_folds_separately(self, zone: SidecarZone) -> None:
        """Keyed on the roster user like the window itself: one name's folds
        cannot be attributed to another's record."""
        env = {**OIDC_ENV, "OSPREY_AUTH_OIDC_SUBJECT_ALICE": "", "OSPREY_AUTH_OIDC_SUBJECT_BOB": ""}
        app = _oidc_app(env)
        clock = _with_movable_audit_window(app)
        for _ in range(4):
            _oidc_start(app, "bob")
        _oidc_start(app, "alice")
        clock.now += 60.0
        _oidc_start(app, "bob")

        assert [(r["subject"], r.get("detail")) for r in zone.records()] == [
            ("bob", None),
            ("alice", None),
            ("bob", f"{FOLDED_DETAIL_KEY}=3"),
        ]

    @pytest.mark.usefixtures("zone")
    def test_a_folded_refusal_still_costs_the_caller_the_same_answer(self) -> None:
        """Counting what was folded changes the ledger, never the response."""
        app = _oidc_app(UNMAPPED_OIDC_ENV)
        _with_movable_audit_window(app)
        answers = {
            (response.status_code, response.json()["detail"])
            for response in (_oidc_start(app, "bob") for _ in range(6))
        }
        assert len(answers) == 1


class TestATokenWithNoIdToken:
    """The one refusal driven by a hostile or substituted token endpoint."""

    def test_it_is_recorded_under_its_own_category(self, zone: SidecarZone) -> None:
        app = _oidc_app()
        app.state.oidc_client.token = {"userinfo": {"sub": ALICE_SUBJECT}}
        assert callback(app).status_code == 502
        assert [(r["decision"], r["reason"]) for r in zone.records()] == [
            ("refused", audit.REASON_UNVALIDATED_TOKEN)
        ]

    def test_the_record_does_not_change_the_answer(self, zone: SidecarZone) -> None:
        """502, not 403: what failed is the provider's response, not this user's
        login, and the ledger entry does not move the status."""
        app = _oidc_app()
        app.state.oidc_client.token = {"access_token": "opaque"}
        response = callback(app)
        assert response.status_code == 502
        assert zone.records()[0]["subject"] == "alice"

    @pytest.mark.usefixtures("zone")
    def test_no_session_is_minted(self) -> None:
        app = _oidc_app()
        app.state.oidc_client.token = {"userinfo": {"sub": ALICE_SUBJECT}}
        response = callback(app)
        assert not [
            header
            for header in response.headers.get_list("set-cookie")
            if header.startswith("osprey_auth_session=")
        ]
