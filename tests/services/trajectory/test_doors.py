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

import asyncio
import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel
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
from osprey.services.trajectory.state import (
    ShotResult,
    Snapshot,
    SnapshotEvent,
    TakenOverEvent,
    parse_event,
)
from osprey.services.trajectory.step import TrajectoryTool, compute_step
from osprey.services.trajectory.store import STATE_DIR_ENV_VAR, TrajectoryStore
from osprey.utils.launch_token import LAUNCH_TOKEN_HEADER
from osprey.utils.owner_header import OWNER_HEADER
from osprey_connectors import control_context, posture_store
from osprey_connectors.control_system.base import ChannelValue, ChannelWriteResult, WriteOutcome
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

    def validate(self, *_args: Any, **_kwargs: Any) -> None:
        return None


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
    assert state["holder"] == {
        "owner": "alice",
        "kind": "panel" if door == "/panel/api" else "agent",
    }
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
    assert client.get("/api/state", headers=_as(ALICE_SECRET, "/api")).json()["holder"] is None
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


# -- the threading-loop routes -------------------------------------------------
#
# A richer stand-in: a machine whose checked write lands, a fake pyAML tool
# answering a set step, and a state directory for shot files and references.

X_ROWS = ("SR:BPM1:X", "SR:BPM2:X", "SR:BPM3:X")
Y_ROWS = ("SR:BPM1:Y", "SR:BPM2:Y", "SR:BPM3:Y")
SUMS = ("SR:BPM1:SUM", "SR:BPM2:SUM", "SR:BPM3:SUM")
HCM1, HCM2, VCM1 = "SR:HCM1", "SR:HCM2", "SR:VCM1"


class _Machine(_Connector):
    """Holds channel values; a checked write lands and is listed."""

    def __init__(self) -> None:
        super().__init__()
        self.values: dict[str, Any] = {HCM1: 1.0, HCM2: -2.0, VCM1: 0.5}
        self.writes: list[tuple[str, Any]] = []

    async def read_channel(self, address: str, timeout: float | None = None) -> Any:  # noqa: ARG002
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return ChannelValue(value=self.values.get(address, 0), timestamp=datetime.now(UTC))

    async def write_channel_checked(self, address: str, value: Any, **_: Any) -> Any:
        self.writes.append((address, value))
        self.values[address] = value
        return ChannelWriteResult(
            channel_address=address, value_written=value, outcome=WriteOutcome.CONFIRMED
        )


class _Correct:
    """A fake pyAML tool answering ``correct`` with a set step."""

    def correct(self, x: Any, y: Any, reference: Any = None, **_: Any) -> Any:  # noqa: ARG002
        inverse = SimpleNamespace(
            no_response_names=(),
            rank_discarded=0,
            singular_values=np.array([1.0]),
            inverse_weights=np.array([1.0]),
            spectrum_kind=(SimpleNamespace(value="LIVE"),),
            n_discard=0,
        )
        return SimpleNamespace(
            step=np.array([0.2, -0.1, 0.05]),
            names=(HCM1, HCM2, VCM1),
            reason=SimpleNamespace(value="OK"),
            status="",
            all_zero=False,
            zeroed_h=(),
            zeroed_v=(),
            pseudo_inverse=inverse,
        )


TOOL = TrajectoryTool(
    model="SR",
    tool=_Correct(),
    layout=SimpleNamespace(
        bpm_names_h=X_ROWS, bpm_names_v=Y_ROWS, hcorr_names=(HCM1, HCM2), vcorr_names=(VCM1,)
    ),
    bpm_index=(1.0, 2.0, 3.0, 1.0, 2.0, 3.0),
    sum_names=(*SUMS, *SUMS),
    hcorr_index=(0.5, 1.5),
    vcorr_index=(2.5,),
    unsolved={},
    alpha=0.0,
    n_discard=0,
    gain=1.0,
)


def _shot(shot_id: str = "20260101T000000000000Z-shot0001") -> ShotResult:
    """A shot keyed the way the service's own row layout keys one."""
    return ShotResult(
        shot_id=shot_id,
        model="SR",
        x=dict(zip(X_ROWS, (0.1, 0.2, 0.3), strict=True)),
        y=dict(zip(Y_ROWS, (-0.1, -0.2, -0.3), strict=True)),
        sum=dict(zip(SUMS, (1.0, 1.0, 1.0), strict=True)),
        bpm_order=list(X_ROWS),
        lost=[],
        last_bpm=X_ROWS[-1],
        i0=0,
        window=(0, 10),
        per_shot_lost_at=[None],
        shots_included=1,
        shots_failed=[],
        commands_fired=[],
    )


def _app(connector: Any, *, tool: Any = None) -> FastAPI:
    async def start() -> TrajectoryStartup:
        return TrajectoryStartup(
            target=TARGET,
            connector=connector,
            commands=COMMANDS,
            guarded_run_dir=Path("/unused"),
            tool=tool,
        )

    app = create_app(start=start)
    app.state.web_credentials = WebCredentials(
        operator_secret=OWN_SECRET,
        panel_token="panel-token",
        roster_secrets=(ALICE_SECRET, BOB_SECRET),
        roster_owners=("alice", "bob"),
    )
    return app


@pytest.fixture
def state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / "state"
    monkeypatch.setenv(STATE_DIR_ENV_VAR, str(directory))
    return directory


@pytest.fixture
def machine() -> _Machine:
    return _Machine()


@pytest.fixture
def loop_client(machine: _Machine, state_dir: Path, monkeypatch: pytest.MonkeyPatch):  # noqa: ARG001
    monkeypatch.setenv(LAUNCH_TOKEN_ENV, TOKEN)
    with TestClient(_app(machine, tool=TOOL)) as test_client:
        yield test_client


def _pend(client: TestClient, client_id: str) -> Any:
    """Make a step for a stored-looking shot pending, as a holder's shot would."""
    sessions = client.app.state.sessions  # type: ignore[attr-defined]
    shot = _shot()
    step = compute_step(TOOL, shot, None, target=TARGET, client_id=client_id, bounds={})
    sessions.step.replace(step, shot)
    sessions.latest_shot_id = shot.shot_id
    return step


def test_state_is_the_snapshot(client: TestClient) -> None:
    client.post("/api/session", headers=_as(ALICE_SECRET, "/api"))
    state = client.get("/panel/api/state", headers=_as(BOB_SECRET, "/panel/api")).json()
    snapshot = Snapshot.model_validate(state)
    assert snapshot.caller == "bob"
    assert snapshot.holder is not None and snapshot.holder.owner == "alice"
    assert snapshot.target == TARGET
    assert snapshot.mode == "idle"
    assert snapshot.stop_budget_s > 0


@pytest.mark.parametrize("door", DOORS)
def test_stop_answers_202_with_the_bound_and_is_recorded_with_the_stopper(
    client: TestClient, door: str, records: list[dict[str, Any]]
) -> None:
    client.post("/panel/api/session", headers=_as(ALICE_SECRET, "/panel/api"))
    response = client.post(f"{door}/stop", headers=_as(BOB_SECRET, door))
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["owner"] == "bob"
    assert body["action"] is None
    assert body["stop_budget_s"] > 0
    state = client.get(f"{door}/state", headers=_as(BOB_SECRET, door)).json()
    assert state["holder"]["owner"] == "alice", "Stop is not a take-over"
    stopped = _ours(records)[-1]
    assert (stopped["subject"], stopped["actor"], stopped["decision"]) == (
        "stop",
        "bob",
        DECISION_ALLOWED,
    )


def test_a_narrowed_owner_cannot_stop(client: TestClient) -> None:
    _narrow("bob")
    response = client.post("/api/stop", headers=_as(BOB_SECRET, "/api"))
    assert response.status_code == 403
    assert _code(response) == "narrowing"


@pytest.mark.parametrize("door", DOORS)
def test_step_needs_only_the_door_owner(loop_client: TestClient, door: str) -> None:
    missing = loop_client.get(f"{door}/step", headers=_as(BOB_SECRET, door))
    assert missing.status_code == 404
    assert _code(missing) == "no_step"

    grant = loop_client.post("/panel/api/session", headers=_as(ALICE_SECRET, "/panel/api")).json()
    step = _pend(loop_client, grant["client_id"])
    # Bob holds no session and still reads the pending step.
    response = loop_client.get(f"{door}/step", headers=_as(BOB_SECRET, door))
    assert response.status_code == 200, response.text
    assert response.json()["digest"] == step.digest


def test_a_shot_names_its_session(loop_client: TestClient) -> None:
    response = loop_client.post("/api/shot", headers=_as(ALICE_SECRET, "/api"))
    assert response.status_code == 400
    assert _code(response) == "bad_request"


def test_a_shot_without_a_sequence_module_is_refused_and_recorded(
    loop_client: TestClient, records: list[dict[str, Any]]
) -> None:
    grant = loop_client.post("/api/session", headers=_as(ALICE_SECRET, "/api")).json()
    response = loop_client.post(
        "/api/shot",
        headers=_as(ALICE_SECRET, "/api", **{CLIENT_HEADER: grant["client_id"]}),
        json={"n_shots": 2},
    )
    assert response.status_code == 503
    assert _code(response) == "no_sequence_module"
    shot = _ours(records)[-1]
    assert (shot["subject"], shot["actor"], shot["reason"]) == (
        "shot",
        "alice",
        "no_sequence_module",
    )


@pytest.mark.parametrize(
    "body", [{"n_shots": 0}, {"n_shots": True}, {"average": "median"}, {"owner": "x"}, {"x": 1}]
)
def test_a_bad_shot_body_is_refused(loop_client: TestClient, body: dict[str, Any]) -> None:
    grant = loop_client.post("/api/session", headers=_as(ALICE_SECRET, "/api")).json()
    response = loop_client.post(
        "/api/shot",
        headers=_as(ALICE_SECRET, "/api", **{CLIENT_HEADER: grant["client_id"]}),
        json=body,
    )
    assert response.status_code == 400
    assert _code(response) in {"bad_request", "owner_in_body"}


def test_the_owner_is_on_the_apply_audit_line(
    loop_client: TestClient, machine: _Machine, records: list[dict[str, Any]]
) -> None:
    grant = loop_client.post("/api/session", headers=_as(ALICE_SECRET, "/api")).json()
    step = _pend(loop_client, grant["client_id"])
    response = loop_client.post(
        "/api/apply",
        headers=_as(ALICE_SECRET, "/api", **{CLIENT_HEADER: grant["client_id"]}),
        json={"step_digest": step.digest},
    )
    assert response.status_code == 200, response.text
    done = response.json()
    assert done["stopped"] is False
    assert [moved["name"] for moved in done["moved"]] == [HCM1, HCM2, VCM1]
    assert [address for address, _ in machine.writes] == [HCM1, HCM2, VCM1]

    applied = [r for r in _ours(records) if r["subject"] == "apply"]
    assert len(applied) == 1, "an apply is recorded once"
    (line,) = applied
    assert line["actor"] == "alice"
    assert line["decision"] == DECISION_ALLOWED
    assert "owner=alice" in line["detail"]
    assert f"step_digest={step.digest}" in line["detail"]
    for name in (HCM1, HCM2, VCM1):
        assert name in line["detail"]

    again = loop_client.post(
        "/api/apply",
        headers=_as(ALICE_SECRET, "/api", **{CLIENT_HEADER: grant["client_id"]}),
        json={"step_digest": step.digest},
    )
    assert again.status_code == 409
    assert _code(again) == "step_consumed"


def test_a_put_back_is_recorded_as_one(
    loop_client: TestClient, machine: _Machine, records: list[dict[str, Any]]
) -> None:
    grant = loop_client.post("/api/session", headers=_as(ALICE_SECRET, "/api")).json()
    response = loop_client.post(
        "/api/apply",
        headers=_as(ALICE_SECRET, "/api", **{CLIENT_HEADER: grant["client_id"]}),
        json={"model": "SR", "values": [{"name": HCM1, "before": 0.25}]},
    )
    assert response.status_code == 200, response.text
    assert machine.values[HCM1] == 0.25
    (line,) = [r for r in _ours(records) if r["subject"] == "put_back"]
    assert line["actor"] == "alice"
    assert "model=SR" in line["detail"]


def test_references_save_load_select_and_list(loop_client: TestClient) -> None:
    grant = loop_client.post("/panel/api/session", headers=_as(ALICE_SECRET, "/panel/api")).json()
    holder = _as(ALICE_SECRET, "/panel/api", **{CLIENT_HEADER: grant["client_id"]})
    _pend(loop_client, grant["client_id"])

    saved = loop_client.post("/panel/api/references", headers=holder, json={"name": "golden"})
    assert saved.status_code == 200, saved.text
    assert saved.json()["name"] == "golden"

    loaded = loop_client.get("/panel/api/references/golden", headers=holder)
    assert loaded.status_code == 200, loaded.text
    assert loaded.json()["matched"] == list(X_ROWS)
    assert loaded.json()["missing_in_reference"] == []

    selected = loop_client.post(
        "/panel/api/references/select", headers=holder, json={"name": "golden"}
    )
    assert selected.status_code == 200, selected.text
    assert selected.json()["reference"]["name"] == "golden"
    assert selected.json()["step"]["reference"] == "golden"

    listed = loop_client.get("/panel/api/references", headers=holder).json()
    assert listed == {"model": "SR", "names": ["golden"], "current": "golden"}
    state = loop_client.get("/panel/api/state", headers=holder).json()
    assert state["reference"]["name"] == "golden"


@pytest.mark.parametrize(
    ("name", "status", "code"),
    [("no-such", 404, "reference_not_found"), ("bad name!", 400, "invalid_name")],
)
def test_a_reference_that_cannot_load_is_refused(
    loop_client: TestClient, name: str, status: int, code: str
) -> None:
    response = loop_client.get(
        f"/panel/api/references/{name}", headers=_as(ALICE_SECRET, "/panel/api")
    )
    assert response.status_code == status
    assert _code(response) == code


def test_a_reference_matching_no_bpm_is_refused(loop_client: TestClient, state_dir: Path) -> None:
    asyncio.run(TrajectoryStore(state_dir).save_reference("SR", "elsewhere", {"OTHER": (0.0, 0.0)}))
    response = loop_client.get(
        "/panel/api/references/elsewhere", headers=_as(ALICE_SECRET, "/panel/api")
    )
    assert response.status_code == 409
    assert _code(response) == "no_matching_bpms"


def test_the_shots_are_listed(loop_client: TestClient, state_dir: Path) -> None:
    asyncio.run(TrajectoryStore(state_dir).save_shot(_shot()))
    response = loop_client.get("/panel/api/shots", headers=_as(ALICE_SECRET, "/panel/api"))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model"] == "SR"
    assert [shot["shot_id"] for shot in body["shots"]] == [_shot().shot_id]


def test_a_continuous_run_the_service_cannot_shoot_is_refused_at_start(
    loop_client: TestClient, records: list[dict[str, Any]]
) -> None:
    grant = loop_client.post("/panel/api/session", headers=_as(ALICE_SECRET, "/panel/api")).json()
    response = loop_client.post(
        "/panel/api/continuous",
        headers=_as(ALICE_SECRET, "/panel/api", **{CLIENT_HEADER: grant["client_id"]}),
        json={"max_shots": 3},
    )
    assert response.status_code == 503
    assert _code(response) == "no_sequence_module"
    state = loop_client.get("/panel/api/state", headers=_as(ALICE_SECRET, "/panel/api")).json()
    assert state["continuous"] is None
    (line,) = [r for r in _ours(records) if r["subject"] == "continuous_start"]
    assert (line["actor"], line["reason"]) == ("alice", "no_sequence_module")


def test_a_heartbeat_without_a_run_answers_none(loop_client: TestClient) -> None:
    grant = loop_client.post("/panel/api/session", headers=_as(ALICE_SECRET, "/panel/api")).json()
    response = loop_client.post(
        "/panel/api/continuous/heartbeat",
        headers=_as(ALICE_SECRET, "/panel/api", **{CLIENT_HEADER: grant["client_id"]}),
    )
    assert response.status_code == 200, response.text
    assert response.json() == {"continuous": None}


# -- the state stream ------------------------------------------------------------


class _Stream:
    """One ``GET /panel/api/events`` driven straight through the ASGI app."""

    def __init__(self, app: FastAPI, secret: str) -> None:
        self.messages: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._gone = asyncio.Event()
        self._asked = False
        self._buffer = ""
        path = "/panel/api/events"
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "path": path,
            "raw_path": path.encode(),
            "root_path": "",
            "query_string": b"",
            "scheme": "http",
            "headers": [(OPERATOR_SECRET_HEADER.lower().encode(), secret.encode())],
            "client": ("127.0.0.1", 50000),
            "server": ("testserver", 80),
        }
        self.task = asyncio.create_task(app(scope, self._receive, self._send))

    async def _receive(self) -> dict[str, Any]:
        if not self._asked:
            self._asked = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await self._gone.wait()
        return {"type": "http.disconnect"}

    async def _send(self, message: dict[str, Any]) -> None:
        await self.messages.put(message)

    async def event(self) -> BaseModel:
        """The next event; it must validate against the state contract."""
        while True:
            if "\n\n" in self._buffer:
                frame, self._buffer = self._buffer.split("\n\n", 1)
                if frame.startswith("data: "):
                    return parse_event(frame.removeprefix("data: "))
                continue
            message = await asyncio.wait_for(self.messages.get(), 5)
            if message["type"] == "http.response.start":
                assert message["status"] == 200
                continue
            self._buffer += message.get("body", b"").decode()
            if not message.get("more_body", False):
                raise EOFError("the stream ended")

    def close(self) -> None:
        self._gone.set()


@pytest.fixture
async def served(connector: _Connector, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[FastAPI]:
    monkeypatch.setenv(LAUNCH_TOKEN_ENV, TOKEN)
    app = _app(connector)
    async with app.router.lifespan_context(app):
        yield app


def _http(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver")


async def test_an_open_stream_survives_stop_and_take_over_and_receives_the_take_over(
    served: FastAPI,
) -> None:
    stream = _Stream(served, ALICE_SECRET)
    first = await stream.event()
    assert isinstance(first, SnapshotEvent)
    assert first.snapshot.caller == "alice"
    assert first.snapshot.holder is None

    async with _http(served) as http:
        opened = await http.post("/panel/api/session", headers=_as(ALICE_SECRET, "/panel/api"))
        assert opened.status_code == 200, opened.text
        stopped = await http.post("/api/stop", headers=_as(BOB_SECRET, "/api"))
        assert stopped.status_code == 202, stopped.text
        taken = await http.post("/api/takeover", headers=_as(BOB_SECRET, "/api"))
        assert taken.status_code == 200, taken.text

    seen: list[BaseModel] = []
    while not any(isinstance(event, TakenOverEvent) for event in seen):
        seen.append(await stream.event())
    (event,) = [event for event in seen if isinstance(event, TakenOverEvent)]
    assert (event.previous.owner, event.previous.kind) == ("alice", "panel")
    assert (event.by.owner, event.by.kind) == ("bob", "agent")
    after = await stream.event()
    while not isinstance(after, SnapshotEvent):
        after = await stream.event()
    assert after.snapshot.holder is not None and after.snapshot.holder.owner == "bob"
    assert after.snapshot.taken_over is not None
    assert not stream.task.done(), "Stop and take-over must not end a stream"

    served.state.shutdown_signal.set()
    with pytest.raises(EOFError):
        while True:
            await stream.event()
    await asyncio.wait_for(stream.task, 5)
    stream.close()


async def test_a_stream_ends_on_the_shutdown_signal_only(served: FastAPI) -> None:
    stream = _Stream(served, BOB_SECRET)
    assert isinstance(await stream.event(), SnapshotEvent)
    async with _http(served) as http:
        await http.post("/panel/api/session", headers=_as(ALICE_SECRET, "/panel/api"))
        stopped = await http.post("/panel/api/stop", headers=_as(ALICE_SECRET, "/panel/api"))
        assert stopped.status_code == 202
    await asyncio.sleep(0.05)
    assert not stream.task.done()
    served.state.shutdown_signal.set()
    await asyncio.wait_for(stream.task, 5)
    stream.close()
