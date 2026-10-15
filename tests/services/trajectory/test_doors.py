"""The trajectory service's gated app: the panel door, the agent door and ``/health``.

The app is the real one behind the real shared web gate; each request names
its caller by the roster secret it presents, the way the terminal's proxy hop
does. The admission gate reads a real control-context tree and the run lock is
the real one under a throwaway deployment root; the connector is a stand-in
and the limits validator is stubbed, as in the sessions' own tests.

Every row carries ``no_auth_seam``: the suite-wide seam would overwrite the
roster secret each row presents with the app's own secret.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from tests._guarded_run import reset_run_state

from osprey.audit import writer
from osprey.audit.envelope import (
    DECISION_ALLOWED,
    DECISION_REFUSED,
    SURFACE_TRAJECTORY_SERVICE,
)
from osprey.interfaces.common_middleware import OPERATOR_SECRET_HEADER
from osprey.interfaces.web_auth import WebCredentials
from osprey.runtime.guarded_run import ENV_CONTROL_IDENTITY
from osprey.services.trajectory import actions
from osprey.services.trajectory.__main__ import service_port
from osprey.services.trajectory.app import CLIENT_HEADER, LAUNCH_TOKEN_ENV, create_app
from osprey.services.trajectory.startup import TrajectoryCommands, TrajectoryStartup
from osprey.utils.launch_token import LAUNCH_TOKEN_HEADER
from osprey.utils.owner_header import OWNER_HEADER
from osprey_connectors import control_context, posture_store
from osprey_connectors.identity import AUDIT_IDENTITY_ENV

pytestmark = pytest.mark.no_auth_seam

TARGET = "va"
TOKEN = "launch-token-value"
OWN_SECRET = "own-secret-" + "x" * 32
ALICE_SECRET = "alice-secret-" + "a" * 32
BOB_SECRET = "bob-secret-" + "b" * 32
UNNAMED_SECRET = "unnamed-secret-" + "u" * 32
REFUSED_SECRET = "refused-secret-" + "r" * 32
COMMANDS = TrajectoryCommands(
    model="SR", arm=("SR:BPM:ARM",), injection_request="SR:INJ:REQ", injection_busy="SR:INJ:BUSY"
)
DOORS = ("/panel/api", "/api")


class _Connector:
    def __init__(self) -> None:
        self.outcome: Any = 0
        self._limits_validator: Any = None

    async def read_channel(self, address: str, timeout: float | None = None) -> Any:  # noqa: ARG002
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


class _Validator:
    failsafe_reason = None


@pytest.fixture(autouse=True)
def _deployment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for name in (
        posture_store.CONTROL_CONTEXT_TREE_ENV_VAR,
        posture_store.CONTROL_CONTEXT_DIR_ENV_VAR,
        posture_store.CONTROL_OWNER_ENV_VAR,
        posture_store.LAUNCH_POSTURE_ENV_VAR,
        ENV_CONTROL_IDENTITY,
        AUDIT_IDENTITY_ENV,
        "OSPREY_EXECUTION_MODE",
        LAUNCH_TOKEN_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    root = tmp_path / "deploy"
    (root / "build").mkdir(parents=True)
    (root / "build" / "config.yml").write_text(
        "control_system:\n  writes_enabled: true\n", encoding="utf-8"
    )
    monkeypatch.chdir(root)
    from osprey_connectors.workspace import reset_config_cache

    reset_config_cache()
    reset_run_state(monkeypatch)
    tree = root / posture_store.STATE_DIR_NAME
    tree.mkdir(parents=True, exist_ok=True)
    (tree / posture_store.CONTROL_TREE_MARKER_NAME).write_text("provisioned\n", encoding="utf-8")
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, str(tree))
    posture_store.invalidate_cache()
    monkeypatch.setattr(
        actions.LimitsValidator, "from_config", staticmethod(lambda **_kw: _Validator())
    )
    yield root
    posture_store.invalidate_cache()
    reset_config_cache()


@pytest.fixture
def records(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Every audit record filed, by any layer, as the fields it was filed with."""
    seen: list[dict[str, Any]] = []

    def capture(**fields: Any) -> Path:
        seen.append(fields)
        return Path("/dev/null")

    monkeypatch.setattr(writer, "record", capture)
    return seen


@pytest.fixture
def connector() -> _Connector:
    return _Connector()


@pytest.fixture
def client(connector: _Connector, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(LAUNCH_TOKEN_ENV, TOKEN)

    async def start() -> TrajectoryStartup:
        return TrajectoryStartup(
            target=TARGET, connector=connector, commands=COMMANDS, guarded_run_dir=Path("/unused")
        )

    app = create_app(start=start)
    app.state.web_credentials = WebCredentials(
        operator_secret=OWN_SECRET,
        panel_token="panel-token",
        roster_secrets=(ALICE_SECRET, BOB_SECRET, UNNAMED_SECRET, REFUSED_SECRET),
        roster_owners=("alice", "bob", "", "björn"),
    )
    with TestClient(app) as test_client:
        yield test_client


def _as(secret: str, door: str, **extra: str) -> dict[str, str]:
    headers = {OPERATOR_SECRET_HEADER: secret, **extra}
    if door == "/api":
        headers.setdefault(LAUNCH_TOKEN_HEADER, TOKEN)
    return headers


def _narrow(owner: str) -> None:
    tree = Path(os.environ[posture_store.CONTROL_CONTEXT_TREE_ENV_VAR])
    control_context.write_record(
        control_context.ControlContext(target=TARGET, generation=1, posture={TARGET: "sandbox"}),
        path=tree / owner / control_context.RECORD_FILENAME,
    )
    posture_store.invalidate_cache()


def _ours(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in records if r.get("surface") == SURFACE_TRAJECTORY_SERVICE]


def _code(response: Any) -> str:
    return response.json()["detail"]["code"]


# -- owner and doors -----------------------------------------------------------


@pytest.mark.parametrize("door", DOORS)
def test_open_grants_the_secrets_account_on_both_doors(client: TestClient, door: str) -> None:
    response = client.post(f"{door}/session", headers=_as(ALICE_SECRET, door))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["owner"] == "alice"
    assert body["target"] == TARGET
    assert body["client_id"]
    state = client.get(f"{door}/state", headers=_as(ALICE_SECRET, door)).json()
    assert state["owner"] == "alice"
    assert state["kind"] == ("panel" if door == "/panel/api" else "agent")
    assert state["caller"] == "alice"
    assert "client_id" not in state


@pytest.mark.parametrize("door", DOORS)
def test_an_inbound_owner_claim_is_not_the_owner(client: TestClient, door: str) -> None:
    response = client.post(
        f"{door}/session", headers=_as(ALICE_SECRET, door, **{OWNER_HEADER: "bob"})
    )
    assert response.status_code == 200
    assert response.json()["owner"] == "alice"


@pytest.mark.parametrize("door", DOORS)
def test_a_roster_account_owner_from_header_refuses_is_no_owner(
    client: TestClient, door: str, records: list[dict[str, Any]]
) -> None:
    response = client.post(f"{door}/session", headers=_as(REFUSED_SECRET, door))
    assert response.status_code == 403
    assert _code(response) == "no_operator_identity"
    assert client.get(f"{door}/state", headers=_as(REFUSED_SECRET, door)).json()["caller"] is None
    (record,) = _ours(records)
    assert record["decision"] == DECISION_REFUSED
    assert record["reason"] == "no_operator_identity"


@pytest.mark.parametrize("door", DOORS)
def test_a_secret_naming_nobody_is_refused(client: TestClient, door: str) -> None:
    response = client.post(f"{door}/session", headers=_as(UNNAMED_SECRET, door))
    assert response.status_code == 403
    assert _code(response) == "no_operator_identity"


@pytest.mark.parametrize("door", DOORS)
def test_a_request_without_a_credential_is_refused_by_the_gate(
    client: TestClient, door: str
) -> None:
    headers = {LAUNCH_TOKEN_HEADER: TOKEN} if door == "/api" else {}
    assert client.post(f"{door}/session", headers=headers).status_code == 401


@pytest.mark.parametrize("door", DOORS)
def test_an_owner_in_the_body_is_refused(client: TestClient, door: str) -> None:
    response = client.post(
        f"{door}/session", headers=_as(ALICE_SECRET, door), json={"owner": "bob"}
    )
    assert response.status_code == 400
    assert _code(response) == "owner_in_body"


# -- launch token --------------------------------------------------------------


def test_the_agent_door_refuses_a_missing_or_wrong_token(client: TestClient) -> None:
    missing = client.post("/api/session", headers={OPERATOR_SECRET_HEADER: ALICE_SECRET})
    wrong = client.post(
        "/api/session", headers=_as(ALICE_SECRET, "/api", **{LAUNCH_TOKEN_HEADER: "no"})
    )
    state = client.get("/api/state", headers={OPERATOR_SECRET_HEADER: ALICE_SECRET})
    for response in (missing, wrong, state):
        assert response.status_code == 403
        assert _code(response) == "launch_token_invalid"


def test_the_agent_door_refuses_when_no_token_is_configured(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(LAUNCH_TOKEN_ENV)
    response = client.post("/api/session", headers=_as(ALICE_SECRET, "/api"))
    assert response.status_code == 503
    assert _code(response) == "launch_token_required"


def test_the_panel_door_never_checks_the_token(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(LAUNCH_TOKEN_ENV)
    response = client.post("/panel/api/session", headers={OPERATOR_SECRET_HEADER: ALICE_SECRET})
    assert response.status_code == 200


# -- sessions through the doors ------------------------------------------------


def test_a_second_open_is_refused_naming_the_holder(client: TestClient) -> None:
    client.post("/panel/api/session", headers=_as(ALICE_SECRET, "/panel/api"))
    response = client.post("/api/session", headers=_as(BOB_SECRET, "/api"))
    assert response.status_code == 409
    assert _code(response) == "session_held"


def test_take_over_voids_the_old_client(client: TestClient) -> None:
    old = client.post("/panel/api/session", headers=_as(ALICE_SECRET, "/panel/api")).json()
    taken = client.post("/api/takeover", headers=_as(BOB_SECRET, "/api"))
    assert taken.status_code == 200
    assert taken.json()["owner"] == "bob"
    voided = client.delete(
        "/panel/api/session",
        headers=_as(ALICE_SECRET, "/panel/api", **{CLIENT_HEADER: old["client_id"]}),
    )
    assert voided.status_code == 409
    assert _code(voided) == "session_taken_over"


def test_take_over_with_no_holder_is_refused(client: TestClient) -> None:
    response = client.post("/panel/api/takeover", headers=_as(ALICE_SECRET, "/panel/api"))
    assert response.status_code == 409
    assert _code(response) == "no_session"


def test_take_over_by_a_narrowed_owner_is_refused(client: TestClient) -> None:
    client.post("/panel/api/session", headers=_as(ALICE_SECRET, "/panel/api"))
    _narrow("bob")
    response = client.post("/api/takeover", headers=_as(BOB_SECRET, "/api"))
    assert response.status_code == 403
    assert _code(response) == "narrowing"


def test_release_without_a_client_header_is_bad_request(client: TestClient) -> None:
    client.post("/panel/api/session", headers=_as(ALICE_SECRET, "/panel/api"))
    response = client.delete("/panel/api/session", headers=_as(ALICE_SECRET, "/panel/api"))
    assert response.status_code == 400
    assert _code(response) == "bad_request"


def test_release_by_another_owner_is_refused(client: TestClient) -> None:
    grant = client.post("/panel/api/session", headers=_as(ALICE_SECRET, "/panel/api")).json()
    response = client.delete(
        "/panel/api/session",
        headers=_as(BOB_SECRET, "/panel/api", **{CLIENT_HEADER: grant["client_id"]}),
    )
    assert response.status_code == 409
    assert _code(response) == "not_session_holder"


def test_a_narrowed_holder_releases_through_the_door(
    client: TestClient, records: list[dict[str, Any]]
) -> None:
    grant = client.post("/api/session", headers=_as(ALICE_SECRET, "/api")).json()
    _narrow("alice")
    response = client.delete(
        "/api/session", headers=_as(ALICE_SECRET, "/api", **{CLIENT_HEADER: grant["client_id"]})
    )
    assert response.status_code == 200, response.text
    assert response.json() == {"owner": "alice", "target": TARGET, "client_id": grant["client_id"]}
    assert client.get("/api/state", headers=_as(ALICE_SECRET, "/api")).json()["owner"] is None
    release = _ours(records)[-1]
    assert release["subject"] == "session_release"
    assert release["decision"] == DECISION_ALLOWED
    assert release["actor"] == "alice"
    assert release["detail"] == f"owner=alice target={TARGET} client_id={grant['client_id']}"


def test_an_unreachable_target_refuses_an_open(client: TestClient, connector: _Connector) -> None:
    connector.outcome = OSError("no answer")
    response = client.post("/panel/api/session", headers=_as(ALICE_SECRET, "/panel/api"))
    assert response.status_code == 503
    assert _code(response) == "target_unreachable"


# -- audit ---------------------------------------------------------------------


def test_each_writing_action_is_recorded_once_with_the_owner_as_actor(
    client: TestClient, records: list[dict[str, Any]]
) -> None:
    grant = client.post("/panel/api/session", headers=_as(ALICE_SECRET, "/panel/api")).json()
    taken = client.post("/api/takeover", headers=_as(BOB_SECRET, "/api")).json()
    client.delete(
        "/api/session", headers=_as(BOB_SECRET, "/api", **{CLIENT_HEADER: taken["client_id"]})
    )
    client.post(
        "/api/session",
        headers=_as(ALICE_SECRET, "/api", **{OWNER_HEADER: "x"}),
        json={"owner": "x"},
    )

    assert [r.get("surface") for r in records] == [SURFACE_TRAJECTORY_SERVICE] * 4
    opened, took, released, refused = records
    assert (opened["subject"], opened["actor"], opened["decision"]) == (
        "session_open",
        "alice",
        DECISION_ALLOWED,
    )
    assert (took["subject"], took["actor"], took["decision"]) == (
        "session_takeover",
        "bob",
        DECISION_ALLOWED,
    )
    assert (released["subject"], released["actor"]) == ("session_release", "bob")
    assert (refused["decision"], refused["reason"]) == (DECISION_REFUSED, "owner_in_body")
    assert grant["client_id"] not in opened["detail"]


def test_a_refused_take_over_is_recorded_under_the_posture_that_held_it(
    client: TestClient, records: list[dict[str, Any]]
) -> None:
    client.post("/panel/api/session", headers=_as(ALICE_SECRET, "/panel/api"))
    _narrow("bob")
    client.post("/api/takeover", headers=_as(BOB_SECRET, "/api"))
    took = _ours(records)[-1]
    assert (took["decision"], took["reason"], took["posture"]) == (
        DECISION_REFUSED,
        "narrowing",
        "sandbox",
    )


# -- health, bundle, entry point -----------------------------------------------


def test_health_answers_without_a_credential_and_asks_the_target(
    client: TestClient, connector: _Connector
) -> None:
    assert client.get("/health").json()["target"] == "reachable"
    connector.outcome = OSError("gone")
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["target"] == "target_unreachable"


def test_the_panel_bundle_is_served(client: TestClient) -> None:
    response = client.get("/panel/", headers={OPERATOR_SECRET_HEADER: ALICE_SECRET})
    assert response.status_code == 200
    assert "Trajectory" in response.text


def test_the_port_is_the_services_port_or_the_layout_slot() -> None:
    from osprey.port_layout import default_port, resolve_port_base

    assert service_port({"services": {"trajectory": {"port": 9123}}}) == 9123
    assert service_port({}) == default_port("trajectory", base=resolve_port_base({}))
