"""Tests for the sidecar's login ledger: what it records, and where it lands.

The sidecar decides one thing — may this browser become this roster user — and
that decision is a safety decision like any tool call. These tests pin the four
properties that make its records usable next to the rest of the ledger:

* **Where.** Records land in the directory ``OSPREY_AUTH_AUDIT_DIR`` names, in
  a file named for the surface. The sidecar is not an Osprey project (one
  uvicorn app, no ``osprey.yml`` to resolve a root from), so the env var is the
  *only* source of that path — a project-root resolution here would either
  raise or invent a directory outside the one subdirectory the container binds.
* **Who.** The service is the actor and the roster user is the SUBJECT. Every
  record files under the sidecar's own identity, so a user's own audit
  subdirectory can never be made to look as though they wrote their own login
  history.
* **What.** ``posture_source`` is the app's own stamp and ``session`` is null:
  a login has no posture store and no session key, and inventing either would
  join these records to a session that never existed.
* **What it costs.** Nothing. An unset variable degrades to a log line, an
  unwritable ledger degrades to a log line, and both a 303 and a 403 arrive
  exactly as they would have with the audit trail working.

Apps are built from explicit env mappings and the OIDC client is replaced at
the route boundary, so nothing here reaches an IdP or reads the real process
environment for anything but the audit variables the fixtures set.
"""

from __future__ import annotations

import json
import logging
from base64 import b64encode
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import itsdangerous
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.audit import writer
from osprey.audit.envelope import POSTURE_SOURCE_APP
from osprey.services.auth_sidecar import audit
from osprey.services.auth_sidecar.app import (
    STATE_COOKIE_NAME,
    create_app,
    get_attempt_throttle,
    get_audit_throttle,
)
from osprey.services.auth_sidecar.passwords import hash_password
from osprey.services.auth_sidecar.routes.oidc import (
    CALLBACK_PATH,
    FOLDED_DETAIL_KEY,
    PENDING_FLOW_SESSION_KEY,
    REASON_AMBIGUOUS_ROLE_CLAIM,
    REASON_IDENTITY_MISMATCH,
    REASON_MISSING_ROLE_CLAIM,
    REASON_NO_ASSERTED_IDENTITY,
    REASON_UNMAPPED_ROLE_CLAIM,
    REASON_UNMAPPED_USER,
    REASON_UNSAFE_ROLE,
    RoleBinding,
)
from osprey.services.auth_sidecar.throttle import AttemptThrottle
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

#: The identity the sidecar's container is given, and therefore the directory
#: component its records file under.
SIDECAR_IDENTITY = "sidecar"

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


@pytest.fixture
def zone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The sidecar's own bound audit subdirectory, as compose gives it.

    Also pins the identity ladder to what the sidecar's container env supplies,
    so a developer's local account can never become the actor in an assertion.
    """
    directory = tmp_path / "var" / "audit" / SIDECAR_IDENTITY
    directory.mkdir(parents=True)
    monkeypatch.setenv(audit.AUDIT_DIR_ENV, str(directory))
    monkeypatch.setenv(AUDIT_IDENTITY_ENV, SIDECAR_IDENTITY)
    monkeypatch.delenv(TERMINAL_USER_ENV, raising=False)
    return directory


def _ledger(zone: Path) -> Path:
    """The one file the sidecar's records land in."""
    return zone / f"{audit.SURFACE}{writer.LEDGER_SUFFIX}"


def _records(zone: Path) -> list[dict[str, Any]]:
    """Every record stored so far, decoded in the order it was appended."""
    path = _ledger(zone)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]


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


class FakeOIDCClient:
    """Stands in for Authlib's Starlette client at the route boundary."""

    def __init__(self, *, userinfo: dict[str, Any] | None = None) -> None:
        # The `id_token` is what the real client parsed `userinfo` from, and the
        # callback refuses a token response without one — see
        # ``test_claim_resolution.py``'s TestTheClaimsMustComeFromAnIdToken.
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
    """An app in OIDC mode with Authlib replaced and the role binding injected."""
    app = create_app(env if env is not None else OIDC_ENV)
    app.state.oidc_client = FakeOIDCClient(userinfo=userinfo)
    app.state.role_binding = binding if binding is not None else RoleBinding()
    return app


def _pending(user: str) -> str:
    """A forged state cookie carrying one in-flight handshake for ``user``."""
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


# --- where a record lands ---------------------------------------------------


class TestWhereARecordLands:
    """The env var is the whole path resolution, and the file names the surface."""

    def test_records_land_in_the_directory_the_env_names(self, zone: Path) -> None:
        with _password_client() as client:
            assert _login(client).status_code == 303
        assert _ledger(zone).exists()
        assert len(_records(zone)) == 1

    def test_the_ledger_is_named_for_the_surface(self, zone: Path) -> None:
        """One file per surface inside the identity's directory — the same
        ``<identity>/<surface>.jsonl`` shape every other emitter files under,
        so the sidecar's records read next to the rest without a special case."""
        with _password_client() as client:
            _login(client)
        assert [path.name for path in zone.iterdir()] == ["auth_sidecar.jsonl"]
        assert audit.ledger_path() == _ledger(zone)

    def test_the_path_never_asks_for_a_project_root(
        self, zone: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The sidecar image holds one uvicorn app and no project. Resolving a
        root here would either raise or point at a directory the container does
        not bind, so the emitter must never reach for one."""

        def _explode(*args: Any, **kwargs: Any) -> Path:
            raise AssertionError("the sidecar must not resolve a project root")

        monkeypatch.setattr("osprey.utils.workspace.resolve_project_root", _explode)
        with _password_client() as client:
            assert _login(client).status_code == 303
        assert len(_records(zone)) == 1

    def test_each_record_is_one_line(self, zone: Path) -> None:
        with _password_client() as client:
            _login(client, password="wrong")
            _login(client, user="bob", password="wrong")
        assert len(_ledger(zone).read_text("utf-8").splitlines()) == 2

    def test_an_unset_variable_degrades_to_a_log_line(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A sidecar rendered before the audit mount existed still logs what it
        could not file — the record is readable, just not durable."""
        monkeypatch.delenv(audit.AUDIT_DIR_ENV, raising=False)
        monkeypatch.chdir(tmp_path)
        with caplog.at_level(logging.INFO, logger=audit.logger.name), _password_client() as client:
            assert _login(client).status_code == 303
        assert not list(tmp_path.rglob("*.jsonl"))
        logged = [
            record.getMessage() for record in caplog.records if "unfiled" in record.getMessage()
        ]
        assert len(logged) == 1
        assert "alice" in logged[0]

    def test_a_blank_variable_degrades_the_same_way(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A variable rendered from an empty ctx key is unset, not a relative
        path to the process's working directory."""
        monkeypatch.setenv(audit.AUDIT_DIR_ENV, "   ")
        monkeypatch.chdir(tmp_path)
        with _password_client() as client:
            assert _login(client).status_code == 303
        assert audit.ledger_path() is None
        assert not list(tmp_path.rglob("*.jsonl"))


# --- what a successful login records ----------------------------------------


class TestASuccessfulLogin:
    """A login that happened is a decision the ledger has to carry too."""

    def test_it_is_recorded_as_allowed(self, zone: Path) -> None:
        with _password_client() as client:
            _login(client)
        record = _records(zone)[0]
        assert record["decision"] == "allowed"
        assert record["reason"] == audit.REASON_PASSWORD_LOGIN

    def test_the_user_is_the_subject_and_the_service_is_the_actor(self, zone: Path) -> None:
        """The user did not write this record; the sidecar did, on their behalf.
        Their name is what was acted on, which is exactly why the file lives
        under the service's directory and not under theirs."""
        with _password_client() as client:
            _login(client)
        record = _records(zone)[0]
        assert record["subject"] == "alice"
        assert record["actor"] == SIDECAR_IDENTITY
        assert _ledger(zone).parent.name == SIDECAR_IDENTITY

    def test_the_posture_is_the_apps_own_stamp(self, zone: Path) -> None:
        """The sidecar governs entry, not writes: it holds no posture store and
        spawns no session, so it stamps its own constant and says so."""
        with _password_client() as client:
            _login(client)
        record = _records(zone)[0]
        assert record["posture"] == audit.SIDECAR_POSTURE
        assert record["posture_source"] == POSTURE_SOURCE_APP

    def test_no_session_key_is_invented(self, zone: Path) -> None:
        """``session`` is the posture-store key that governed the record. A
        login has none — the browser has no session yet — and the field is
        present as null rather than dropped, so every line carries one shape."""
        with _password_client() as client:
            _login(client)
        record = _records(zone)[0]
        assert "session" in record
        assert record["session"] is None

    def test_the_credential_never_reaches_the_ledger(self, zone: Path) -> None:
        with _password_client() as client:
            _login(client)
            _login(client, password="a-guess-that-failed")
        stored = _ledger(zone).read_text("utf-8")
        assert ALICE_PASSWORD not in stored
        assert "a-guess-that-failed" not in stored

    def test_an_oidc_login_records_the_role_it_resolved(self, zone: Path) -> None:
        app = _oidc_app(
            userinfo={"sub": ALICE_SUBJECT, GROUP_CLAIM: [OPERATOR_GROUP]},
            binding=RoleBinding(claim=GROUP_CLAIM, claim_map=CLAIM_MAP),
        )
        assert _callback(app).status_code == 303
        record = _records(zone)[0]
        assert record["decision"] == "allowed"
        assert record["reason"] == audit.REASON_OIDC_LOGIN
        assert record["subject"] == "alice"
        assert record["role"] == "operator"

    def test_a_roleless_deployment_records_no_role(self, zone: Path) -> None:
        """Deny-safe by absence here too: no role means the key is omitted, not
        blank, so nobody reads an empty string as a role named ''."""
        assert _callback(_oidc_app()).status_code == 303
        assert "role" not in _records(zone)[0]


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
    def test_every_refusal_is_recorded(self, zone: Path, user: str, password: str) -> None:
        with _password_client() as client:
            assert _login(client, user=user, password=password).status_code == 401
        record = _records(zone)[0]
        assert record["decision"] == "refused"
        assert record["reason"] == audit.REASON_BAD_CREDENTIAL

    def test_the_ledger_does_not_say_which_users_exist(self, zone: Path) -> None:
        """The anti-lookup invariant reaches the ledger as well as the page: a
        wrong password, an unprovisioned roster user and a name that was never
        on the roster are one category, so a reader of the file cannot use it
        to enumerate accounts either."""
        with _password_client() as client:
            _login(client, user="alice", password="wrong")
            _login(client, user="bob", password="wrong")
            _login(client, user="carol", password="wrong")
        assert {record["reason"] for record in _records(zone)} == {audit.REASON_BAD_CREDENTIAL}

    def test_the_recorded_subject_is_bounded(self, zone: Path) -> None:
        """The name the ledger names is the bounded one the page echoes, never
        the caller-sized string: neither the page nor the ledger is sized by
        whoever posted the form."""
        with _password_client() as client:
            _login(client, user="a" * 500, password="wrong")
        assert len(_records(zone)[0]["subject"]) <= 256

    def test_an_unevaluated_attempt_is_not_recorded(self, zone: Path) -> None:
        """A second attempt inside an open throttle window is refused without
        the credential ever being evaluated, and the ledger says nothing about
        it. Recording it would let an unauthenticated caller append to the
        audit trail at request rate, and the decision it would describe — this
        user's credential did not verify — is already on the line above."""
        with _password_client() as client:
            assert _login(client, password="wrong").status_code == 401
            assert _login(client, password="wrong").status_code == 429
        assert len(_records(zone)) == 1


class TestARefusedOidcLogin:
    """Every category the OIDC path refuses under reaches the same ledger."""

    def test_an_unmapped_user_is_recorded(self, zone: Path) -> None:
        env = {key: value for key, value in OIDC_ENV.items() if "SUBJECT_ALICE" not in key}
        with TestClient(_oidc_app(env), base_url=EXTERNAL_ORIGIN) as client:
            response = client.get(
                "/auth/oidc/login", params={"user": "alice"}, follow_redirects=False
            )
        assert response.status_code == 403
        assert _records(zone)[0]["reason"] == REASON_UNMAPPED_USER

    def test_a_mapped_identity_that_cannot_be_carried_is_recorded(self, zone: Path) -> None:
        env = {**OIDC_ENV, "OSPREY_AUTH_OIDC_SUBJECT_ALICE": "jörg@example.org"}
        assert _callback(_oidc_app(env)).status_code == 403
        assert _records(zone)[0]["reason"] == audit.REASON_NON_ASCII_SUBJECT

    def test_an_absent_asserted_identity_is_recorded(self, zone: Path) -> None:
        assert _callback(_oidc_app(userinfo={})).status_code == 403
        assert _records(zone)[0]["reason"] == REASON_NO_ASSERTED_IDENTITY

    def test_an_identity_belonging_to_another_user_is_recorded(self, zone: Path) -> None:
        assert _callback(_oidc_app(userinfo={"sub": BOB_SUBJECT})).status_code == 403
        assert _records(zone)[0]["reason"] == REASON_IDENTITY_MISMATCH

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
        zone: Path,
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
        assert _callback(app).status_code == 403
        record = _records(zone)[0]
        assert record["decision"] == "refused"
        assert record["reason"] == reason
        assert record["subject"] == "alice"

    def test_exactly_one_record_per_refusal(self, zone: Path) -> None:
        """The refusal seam is one call wide: no category records twice, and a
        refused login never also files the success record."""
        assert _callback(_oidc_app(userinfo={"sub": BOB_SUBJECT})).status_code == 403
        assert len(_records(zone)) == 1


# --- the emitter never costs the decision -----------------------------------


class TestTheEmitterNeverCostsTheDecision:
    """An audit gap must never become an outage."""

    def test_an_unwritable_zone_still_lets_a_login_succeed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The variable points at something that is not a directory — a bind
        that never materialised. The login is still a login."""
        blocked = tmp_path / "not-a-directory"
        blocked.write_text("", encoding="utf-8")
        monkeypatch.setenv(audit.AUDIT_DIR_ENV, str(blocked))
        with _password_client() as client:
            assert _login(client).status_code == 303

    def test_an_unwritable_zone_still_refuses(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        blocked = tmp_path / "not-a-directory"
        blocked.write_text("", encoding="utf-8")
        monkeypatch.setenv(audit.AUDIT_DIR_ENV, str(blocked))
        assert _callback(_oidc_app(userinfo={"sub": BOB_SUBJECT})).status_code == 403

    def test_a_missing_directory_is_created(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The deploy path provisions this directory; the emitter creating it
        is the laptop and first-run case, not a second owner of the mode."""
        zone = tmp_path / "var" / "audit" / SIDECAR_IDENTITY
        monkeypatch.setenv(audit.AUDIT_DIR_ENV, str(zone))
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, SIDECAR_IDENTITY)
        with _password_client() as client:
            assert _login(client).status_code == 303
        assert len(_records(zone)) == 1

    def test_a_seam_that_raises_costs_nothing(
        self, zone: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _explode(*args: Any, **kwargs: Any) -> None:
            raise OSError("the audit zone is gone")

        monkeypatch.setattr(audit, "write_envelope", _explode)
        with _password_client() as client:
            assert _login(client).status_code == 303
        assert _callback(_oidc_app(userinfo={"sub": BOB_SUBJECT})).status_code == 403

    def test_a_record_that_cannot_be_built_costs_nothing(self, zone: Path) -> None:
        """An empty subject is a programming error in a caller, not a reason to
        raise into the decision it describes."""
        audit.record_login_refusal(user="", reason=audit.REASON_BAD_CREDENTIAL)
        audit.record_login_success(user="", method="password")
        assert _records(zone) == []


# --- the shared writer does the writing -------------------------------------


class TestTheSharedWriterDoesTheWriting:
    """The sidecar supplies the directory; the ledger's own writer supplies the
    line shaping and the atomic append, so its byte budget and its append-only
    guarantee are not re-implemented one service over."""

    def test_the_reused_seam_exists(self) -> None:
        """A rename in the writer must fail here, not in production: the seam
        is the writer's, and the sidecar bypasses its routing entry points only
        because they resolve a project root it does not have."""
        assert callable(writer.append_envelope)
        assert callable(writer._line_for)
        assert callable(writer._append)

    def test_the_append_goes_through_the_writer(
        self, zone: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        appended: list[tuple[Path, bytes]] = []

        def _capture(path: Path, line: bytes) -> bool:
            appended.append((path, line))
            return True

        monkeypatch.setattr(writer, "_append", _capture)
        with _password_client() as client:
            _login(client)
        assert len(appended) == 1
        path, line = appended[0]
        assert path == _ledger(zone)
        assert line.endswith(b"\n")
        assert json.loads(line)["subject"] == "alice"


# --- the degrade ladder has no missing rung ---------------------------------


class TestTheDegradeLadderIsMonotone:
    """*File, else log* — for every way the write can fail, not just the easy one.

    A CONFIGURED-but-unwritable ledger must not be a worse degrade than a
    variable that was never set. The failure that is actually likely in a
    deployment — a bind that mounted read-only, a full disk, a root-owned file
    this process cannot open — is precisely the one that would otherwise erase
    the record, while the merely cosmetic one (an unset variable on a laptop)
    kept it.
    """

    def test_an_unwritable_zone_still_logs_the_record(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        blocked = tmp_path / "not-a-directory"
        blocked.write_text("", encoding="utf-8")
        monkeypatch.setenv(audit.AUDIT_DIR_ENV, str(blocked))
        with caplog.at_level(logging.INFO, logger=audit.logger.name), _password_client() as client:
            assert _login(client).status_code == 303
        unfiled = [
            record.getMessage() for record in caplog.records if "unfiled" in record.getMessage()
        ]
        assert len(unfiled) == 1
        stored = json.loads(unfiled[0].split("unfiled): ", 1)[1])
        assert stored["subject"] == "alice"
        assert stored["decision"] == "allowed"
        assert stored["reason"] == audit.REASON_PASSWORD_LOGIN

    def test_a_refusal_in_an_unwritable_zone_keeps_its_category(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The category is the half a human cannot reconstruct from the route's
        own log line, which says only that the login was refused."""
        blocked = tmp_path / "not-a-directory"
        blocked.write_text("", encoding="utf-8")
        monkeypatch.setenv(audit.AUDIT_DIR_ENV, str(blocked))
        with caplog.at_level(logging.INFO, logger=audit.logger.name), _password_client() as client:
            assert _login(client, password="wrong").status_code == 401
        unfiled = [
            record.getMessage() for record in caplog.records if "unfiled" in record.getMessage()
        ]
        assert len(unfiled) == 1
        assert json.loads(unfiled[0].split("unfiled): ", 1)[1])["reason"] == (
            audit.REASON_BAD_CREDENTIAL
        )

    def test_a_torn_append_logs_the_record_too(
        self, zone: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The quietest failure of the three: ``_append`` returns False, nothing
        raises, and the writer's own warning names the path rather than the
        record it lost."""
        monkeypatch.setattr(writer, "_append", lambda path, line: False)
        with caplog.at_level(logging.INFO, logger=audit.logger.name), _password_client() as client:
            assert _login(client).status_code == 303
        assert [
            record.getMessage() for record in caplog.records if "unfiled" in record.getMessage()
        ]

    def test_a_seam_that_raises_logs_the_record_too(
        self, zone: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        def _explode(path: Path, line: bytes) -> bool:
            raise OSError("the audit zone is gone")

        monkeypatch.setattr(writer, "_append", _explode)
        with caplog.at_level(logging.INFO, logger=audit.logger.name), _password_client() as client:
            assert _login(client).status_code == 303
        assert [
            record.getMessage() for record in caplog.records if "unfiled" in record.getMessage()
        ]


class TestThePathTheVariableNames:
    """What the variable may hold, and what the file may be called."""

    def test_a_relative_directory_is_refused_like_a_blank_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Resolved against the process's working directory it would name a path
        inside the image that the host binds nothing at: the records would
        accumulate in the container's writable layer and vanish with it, while
        the emitter reported them as durably stored."""
        monkeypatch.setenv(audit.AUDIT_DIR_ENV, "relative/audit")
        monkeypatch.chdir(tmp_path)
        with caplog.at_level(logging.INFO, logger=audit.logger.name), _password_client() as client:
            assert _login(client).status_code == 303
        assert audit.ledger_path() is None
        assert not list(tmp_path.rglob("*.jsonl"))
        assert [
            record.getMessage() for record in caplog.records if "unfiled" in record.getMessage()
        ]

    def test_a_dot_relative_directory_is_refused_too(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(audit.AUDIT_DIR_ENV, "./audit")
        monkeypatch.chdir(tmp_path)
        assert audit.ledger_path() is None

    def test_the_writer_marker_cannot_rename_this_ledger(
        self, zone: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``writer.ledger_name`` routes on ``OSPREY_AUDIT_WRITER``, whose whole
        purpose — one uid per file across the project image's root phase — has
        no analogue in a single-uvicorn container with no entrypoint. Borrowing
        it would let a stray marker in this environment silently rename the file
        the e2e grep and an operator's ``tail`` both look for."""
        monkeypatch.setenv(writer.AUDIT_WRITER_ENV, writer.WRITER_MAINTENANCE)
        assert writer.ledger_name(audit.SURFACE) != audit.SURFACE
        with _password_client() as client:
            assert _login(client).status_code == 303
        assert [path.name for path in zone.iterdir()] == ["auth_sidecar.jsonl"]


class TestTheCategorySetIsClosed:
    """One place enumerates what an operator may find in ``reason``."""

    def test_every_category_this_service_files_is_in_the_set(self) -> None:
        named = {
            value
            for name, value in vars(audit).items()
            if name.startswith("REASON_") and isinstance(value, str)
        }
        assert named <= audit.LOGIN_REASONS

    def test_the_oidc_routes_name_the_same_categories(self) -> None:
        """The route module keeps its own spellings for readability at the point
        of decision; they must be the very same strings, not a second copy."""
        from osprey.services.auth_sidecar.routes import oidc as oidc_routes

        for name, value in vars(oidc_routes).items():
            if name.startswith("REASON_") and isinstance(value, str):
                assert value in audit.LOGIN_REASONS, name


class TestTheWriterShapesTheLine:
    """The line shaping is the half the reuse exists for, so it is pinned by
    BEHAVIOUR rather than by the symbol's existence.

    ``_line_for`` carries the 2 KB budget that makes concurrent appends
    non-interleaving, the oversize-``detail`` replacement, and the
    identifiers-only escape hatch. A local re-implementation that still called
    ``writer._append`` would pass a test that only asserted the name is
    callable — and would not show up until a long ``detail`` (the ambiguity
    category joins role names into one) tore a line in production.
    """

    def test_an_over_budget_detail_is_replaced_by_the_writers_marker(
        self, zone: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(TERMINAL_USER_ENV, "a" * 300)
        audit.record_login_refusal(
            user="u" * 300, reason="r" * 300, detail="d" * 4096, role="R" * 300
        )
        stored = _records(zone)
        assert len(stored) == 1
        assert stored[0]["detail"] == writer.DETAIL_DROPPED

    def test_the_stored_line_stays_inside_the_append_bound(
        self, zone: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The property the replacement buys: one record is still one atomic
        write, so two containers appending at once cannot interleave."""
        monkeypatch.setenv(TERMINAL_USER_ENV, "a" * 300)
        audit.record_login_refusal(
            user="u" * 300, reason="r" * 300, detail="d" * 4096, role="R" * 300
        )
        assert len(_ledger(zone).read_bytes()) <= writer.MAX_RECORD_BYTES

    def test_an_ordinary_detail_is_stored_whole(self, zone: Path) -> None:
        """The control: the degrade is for records that do not fit, not for
        every record that has a detail."""
        audit.record_login_refusal(
            user="alice", reason=audit.REASON_MISSING_ROLE_CLAIM, detail="groups"
        )
        assert _records(zone)[0]["detail"] == "groups"


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

    def test_a_flood_of_login_starts_appends_once(self, zone: Path) -> None:
        app = _oidc_app(UNMAPPED_OIDC_ENV)
        for _ in range(50):
            assert _oidc_start(app, "bob").status_code == 403
        assert len(_records(zone)) == 1
        assert _records(zone)[0]["reason"] == REASON_UNMAPPED_USER

    def test_the_refusal_itself_is_unchanged(self, zone: Path) -> None:
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
        self, zone: Path, env: dict[str, str], reason: str
    ) -> None:
        """Reached at two requests per record without the bound — the state
        cookie that gets here is itself minted by a free GET.

        Both pre-exchange arms, because they are two separate ``bound=`` at two
        separate call sites: the misconfigured-subject deployment is exactly as
        cheap to drive as the unmapped-user one, and pinning only the first left
        the second free to lose its bound unnoticed.
        """
        app = _oidc_app(env)
        for _ in range(20):
            assert _callback(app, user="bob").status_code == 403
        assert [record["reason"] for record in _records(zone)] == [reason]

    def test_each_user_has_their_own_window(self, zone: Path) -> None:
        """Keyed on the roster user, like the password path's window, so one
        name's refusals cannot silence another's."""
        env = {**OIDC_ENV, "OSPREY_AUTH_OIDC_SUBJECT_ALICE": "", "OSPREY_AUTH_OIDC_SUBJECT_BOB": ""}
        app = _oidc_app(env)
        _oidc_start(app, "alice")
        _oidc_start(app, "bob")
        assert [record["subject"] for record in _records(zone)] == ["alice", "bob"]

    def test_a_post_exchange_refusal_is_recorded_every_time(self, zone: Path) -> None:
        """No throttle there, deliberately: reaching one costs a full IdP round
        trip, so it is not free to generate, and it describes a login that
        actually authenticated."""
        app = _oidc_app(userinfo={"sub": BOB_SUBJECT})
        for _ in range(3):
            assert _callback(app, user="alice").status_code == 403
        assert [record["reason"] for record in _records(zone)] == [REASON_IDENTITY_MISMATCH] * 3


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

    def test_unevaluated_refusals_never_touch_the_login_window(self, zone: Path) -> None:
        """The reviewer's measurement, inverted: eight free GETs used to leave
        ``retry_after('bob') > 0`` on the window that decides logins."""
        app = _oidc_app(UNMAPPED_OIDC_ENV)
        for _ in range(8):
            assert _oidc_start(app, "bob").status_code == 403
        login_window = get_attempt_throttle(SimpleNamespace(app=app))
        assert login_window.retry_after("bob") == 0.0
        assert len(login_window) == 0

    def test_a_paced_caller_cannot_walk_the_login_window_to_the_cap(self, zone: Path) -> None:
        """The version that survives pacing: each refusal that is actually filed
        would have grown the shared window one more step toward its 30 s cap."""
        app = _oidc_app(UNMAPPED_OIDC_ENV)
        clock = _with_movable_audit_window(app)
        for _ in range(10):
            assert _oidc_start(app, "bob").status_code == 403
            clock.now += 60.0
        assert len(_records(zone)) == 10
        login_window = get_attempt_throttle(SimpleNamespace(app=app))
        assert login_window.retry_after("bob") == 0.0
        assert len(login_window) == 0

    def test_the_ledger_window_is_what_grew_instead(self, zone: Path) -> None:
        """The bound still holds — it is simply held by the other object."""
        app = _oidc_app(UNMAPPED_OIDC_ENV)
        for _ in range(8):
            _oidc_start(app, "bob")
        assert len(_records(zone)) == 1
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

    def test_the_next_record_names_how_many_were_folded(self, zone: Path) -> None:
        app = _oidc_app(UNMAPPED_OIDC_ENV)
        clock = _with_movable_audit_window(app)
        for _ in range(5):
            assert _oidc_start(app, "bob").status_code == 403
        clock.now += 60.0
        assert _oidc_start(app, "bob").status_code == 403

        records = _records(zone)
        assert len(records) == 2
        assert records[0].get("detail") is None
        assert records[1]["detail"] == f"{FOLDED_DETAIL_KEY}=4"

    def test_a_record_that_folded_nothing_carries_no_note(self, zone: Path) -> None:
        """The key appears only where something WAS suppressed, so an ordinary
        record reads exactly as it did before."""
        app = _oidc_app(UNMAPPED_OIDC_ENV)
        clock = _with_movable_audit_window(app)
        assert _oidc_start(app, "bob").status_code == 403
        clock.now += 60.0
        assert _oidc_start(app, "bob").status_code == 403

        assert [record.get("detail") for record in _records(zone)] == [None, None]

    def test_the_count_restarts_after_it_is_filed(self, zone: Path) -> None:
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

        assert [record.get("detail") for record in _records(zone)] == [
            None,
            f"{FOLDED_DETAIL_KEY}=2",
            f"{FOLDED_DETAIL_KEY}=1",
        ]

    def test_each_user_folds_separately(self, zone: Path) -> None:
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

        assert [(r["subject"], r.get("detail")) for r in _records(zone)] == [
            ("bob", None),
            ("alice", None),
            ("bob", f"{FOLDED_DETAIL_KEY}=3"),
        ]

    def test_a_folded_refusal_still_costs_the_caller_the_same_answer(self, zone: Path) -> None:
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

    def test_it_is_recorded_under_its_own_category(self, zone: Path) -> None:
        app = _oidc_app()
        app.state.oidc_client.token = {"userinfo": {"sub": ALICE_SUBJECT}}
        assert _callback(app).status_code == 502
        assert [(r["decision"], r["reason"]) for r in _records(zone)] == [
            ("refused", audit.REASON_UNVALIDATED_TOKEN)
        ]

    def test_the_record_does_not_change_the_answer(self, zone: Path) -> None:
        """502, not 403: what failed is the provider's response, not this user's
        login, and the ledger entry does not move the status."""
        app = _oidc_app()
        app.state.oidc_client.token = {"access_token": "opaque"}
        response = _callback(app)
        assert response.status_code == 502
        assert _records(zone)[0]["subject"] == "alice"

    def test_no_session_is_minted(self, zone: Path) -> None:
        app = _oidc_app()
        app.state.oidc_client.token = {"userinfo": {"sub": ALICE_SUBJECT}}
        response = _callback(app)
        assert not [
            header
            for header in response.headers.get_list("set-cookie")
            if header.startswith("osprey_auth_session=")
        ]
