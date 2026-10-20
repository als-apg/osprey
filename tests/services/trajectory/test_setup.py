"""Setup as a server-checked fact: the setup route, its persistence and its re-check.

The app is the real one behind the real shared web gate, its connector a
stand-in machine whose setpoints drive their readbacks (unless a readback is
held stuck), its tool a fake pyAML tool. The model declares a device-scope
attenuation on each of three BPMs and a global turns tracked.

Every row carries ``no_auth_seam``: the suite-wide seam would overwrite the
roster secret each row presents with the app's own secret.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from tests._guarded_run import reset_run_state

from osprey.audit import writer
from osprey.audit.envelope import SURFACE_TRAJECTORY_SERVICE
from osprey.interfaces.common_middleware import OPERATOR_SECRET_HEADER
from osprey.interfaces.web_auth import WebCredentials
from osprey.runtime.guarded_run import ENV_CONTROL_IDENTITY
from osprey.services.trajectory import actions
from osprey.services.trajectory import app as app_module
from osprey.services.trajectory.acquisition import ChainRun, ShotRows, chain_action
from osprey.services.trajectory.app import CLIENT_HEADER, LAUNCH_TOKEN_ENV, create_app
from osprey.services.trajectory.setup import (
    SetupKeeper,
    declared_settings,
    matches,
)
from osprey.services.trajectory.solve_config import SolveUpdate
from osprey.services.trajectory.startup import (
    TrajectoryCommands,
    TrajectoryDevice,
    TrajectoryStartup,
)
from osprey.services.trajectory.state import (
    ChainResult,
    SetupReadbacks,
    SetupResult,
    SetupState,
    ShotResult,
    Snapshot,
)
from osprey.services.trajectory.step import TrajectoryTool, compute_step
from osprey.services.trajectory.store import STATE_DIR_ENV_VAR, TrajectoryStore
from osprey.utils.launch_token import LAUNCH_TOKEN_HEADER
from osprey_connectors import posture_store
from osprey_connectors.control_system.base import ChannelValue, ChannelWriteResult, WriteOutcome
from osprey_connectors.identity import AUDIT_IDENTITY_ENV

pytestmark = pytest.mark.no_auth_seam

TARGET = "va"
TOKEN = "launch-token-value"
OWN_SECRET = "own-secret-" + "x" * 32
ALICE_SECRET = "alice-secret-" + "a" * 32
BOB_SECRET = "bob-secret-" + "b" * 32
PANEL = "/panel/api"

BPMS = (1, 2, 3)
X_ROWS = tuple(f"SR:BPM{n}:X" for n in BPMS)
Y_ROWS = tuple(f"SR:BPM{n}:Y" for n in BPMS)
SUMS = tuple(f"SR:BPM{n}:SUM" for n in BPMS)
DEVICES = tuple(f"SR/BPM{n}" for n in BPMS)
ATT_SP = {device: f"SR:BPM{n}:ATT:SP" for device, n in zip(DEVICES, BPMS, strict=True)}
ATT_RB = {device: f"SR:BPM{n}:ATT:RB" for device, n in zip(DEVICES, BPMS, strict=True)}
TURNS_SP, TURNS_RB = "SR:DIAG:TBT:TURNS:SP", "SR:DIAG:TBT:TURNS:RB"
HCM1, HCM2, VCM1 = "SR:HCM1", "SR:HCM2", "SR:VCM1"

COMMANDS = TrajectoryCommands(
    model="SR",
    devices=tuple(
        TrajectoryDevice(
            name=device,
            x_row=x,
            y_row=y,
            sum_row=total,
            arm="SR:BPM:ARM",
            armed="SR:BPM:ARMED",
        )
        for device, x, y, total in zip(DEVICES, X_ROWS, Y_ROWS, SUMS, strict=True)
    ),
    injection_request="SR:INJ:REQ",
    injection_busy="SR:INJ:BUSY",
    trigger_sample=1,
)

#: The model's ``trajectory_slots`` entry, shaped as the facts view writes it.
SLOTS: dict[str, Any] = {
    "settings": [
        {
            "id": "attenuation",
            "label": "Attenuation",
            "description": None,
            "scope": "device",
            "setting": dict(ATT_SP),
            "readback": dict(ATT_RB),
            "tolerance": dict.fromkeys(DEVICES, 0.01),
            "timeout_s": 0.5,
        },
        {
            "id": "turns_tracked",
            "label": "Turns tracked",
            "description": None,
            "scope": "global",
            "setting": TURNS_SP,
            "readback": TURNS_RB,
            "tolerance": None,
            "timeout_s": 0.5,
        },
    ],
}


class _Machine:
    """Channel values; a setpoint written drives its readback unless the readback is stuck."""

    def __init__(self) -> None:
        self.values: dict[str, Any] = {HCM1: 1.0, HCM2: -2.0, VCM1: 0.5}
        for device in DEVICES:
            self.values[ATT_SP[device]] = 10.0
            self.values[ATT_RB[device]] = 10.0
        self.values[TURNS_SP] = 4
        self.values[TURNS_RB] = 4
        for row in (*X_ROWS, *Y_ROWS, *SUMS):
            self.values[row] = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.readback_of = {
            **{ATT_SP[device]: ATT_RB[device] for device in DEVICES},
            TURNS_SP: TURNS_RB,
        }
        self.stuck: set[str] = set()
        self.writes: list[tuple[str, Any]] = []
        self.batches: list[list[str]] = []
        self._limits_validator: Any = None

    async def read_channel(self, address: str, timeout: float | None = None) -> Any:  # noqa: ARG002
        return ChannelValue(value=self.values.get(address, 0), timestamp=datetime.now(UTC))

    async def read_multiple_channels(
        self, addresses: list[str], timeout: float | None = None
    ) -> dict[str, Any]:
        self.batches.append(list(addresses))
        return {address: await self.read_channel(address, timeout) for address in addresses}

    async def write_channel_checked(self, address: str, value: Any, **_: Any) -> Any:
        self.writes.append((address, value))
        self.values[address] = value
        readback = self.readback_of.get(address)
        if readback is not None and readback not in self.stuck:
            self.values[readback] = value
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
    hcorr_index=(0.5, 1.5),
    vcorr_index=(2.5,),
    unsolved={},
    alpha=0.0,
    n_discard=0,
    gain=1.0,
)


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
    monkeypatch.setenv(LAUNCH_TOKEN_ENV, TOKEN)
    monkeypatch.setattr(app_module, "_trajectory_slots", lambda _config, _model: SLOTS)
    yield root
    posture_store.invalidate_cache()
    reset_config_cache()


@pytest.fixture
def records(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []

    def capture(**fields: Any) -> Path:
        seen.append(fields)
        return Path("/dev/null")

    monkeypatch.setattr(writer, "record", capture)
    return seen


@pytest.fixture
def state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / "state"
    monkeypatch.setenv(STATE_DIR_ENV_VAR, str(directory))
    return directory


@pytest.fixture
def machine() -> _Machine:
    return _Machine()


def _app(connector: Any) -> FastAPI:
    async def start() -> TrajectoryStartup:
        return TrajectoryStartup(
            target=TARGET,
            connector=connector,
            commands=COMMANDS,
            guarded_run_dir=Path("/unused"),
            tool=TOOL,
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
def client(machine: _Machine, state_dir: Path):  # noqa: ARG001
    with TestClient(_app(machine)) as test_client:
        yield test_client


def _as(secret: str, door: str = PANEL, **extra: str) -> dict[str, str]:
    headers = {OPERATOR_SECRET_HEADER: secret, **extra}
    if door == "/api":
        headers.setdefault(LAUNCH_TOKEN_HEADER, TOKEN)
    return headers


def _hold(client: TestClient, door: str = PANEL) -> dict[str, str]:
    grant = client.post(f"{door}/session", headers=_as(ALICE_SECRET, door)).json()
    return _as(ALICE_SECRET, door, **{CLIENT_HEADER: grant["client_id"]})


def _code(response: Any) -> str:
    return response.json()["detail"]["code"]


def _state(client: TestClient) -> Snapshot:
    return Snapshot.model_validate(client.get(f"{PANEL}/state", headers=_as(ALICE_SECRET)).json())


def _setup(client: TestClient, headers: dict[str, str], body: dict[str, Any]) -> SetupResult:
    response = client.post(f"{PANEL}/setup", headers=headers, json=body)
    assert response.status_code == 200, response.text
    return SetupResult.model_validate(response.json())


def _pend(client: TestClient, client_id: str) -> None:
    """Make a step pending, as a holder's shot would."""
    sessions = client.app.state.sessions  # type: ignore[attr-defined]
    shot = ShotResult(
        shot_id="20260101T000000000000Z-shot0001",
        model="SR",
        x=dict(zip(X_ROWS, (0.1, 0.2, 0.3), strict=True)),
        y=dict(zip(Y_ROWS, (-0.1, -0.2, -0.3), strict=True)),
        sum=dict(zip(SUMS, (1.0, 1.0, 1.0), strict=True)),
        bpm_order=list(X_ROWS),
        lost=[],
        last_bpm=X_ROWS[-1],
        i0=0,
        window=(0, 4),
        per_shot_lost_at=[None],
        shots_included=1,
        shots_failed=[],
        commands_fired=[],
    )
    step = compute_step(TOOL, shot, None, target=TARGET, client_id=client_id, bounds={})
    sessions.step.replace(step, shot)


# -- the setup route ---------------------------------------------------------------------


def test_a_setup_writes_only_what_differs_and_answers_the_moves_in_put_back_shape(
    client: TestClient, machine: _Machine, records: list[dict[str, Any]]
) -> None:
    machine.values[ATT_RB["SR/BPM2"]] = 12.0
    machine.values[ATT_SP["SR/BPM2"]] = 12.0
    headers = _hold(client)
    machine.batches.clear()

    result = _setup(client, headers, {"settings": {"attenuation": 12.0, "turns_tracked": 250}})

    # Every declared setting of every included BPM is read in one batch first.
    first = machine.batches[0]
    assert set(first) == {*ATT_SP.values(), *ATT_RB.values(), TURNS_SP, TURNS_RB}
    # BPM2 already reads 12: it is not written; the global is written once.
    assert machine.writes == [
        (ATT_SP["SR/BPM1"], 12.0),
        (ATT_SP["SR/BPM3"], 12.0),
        (TURNS_SP, 250),
    ]
    assert [(m.name, m.before, m.after, m.outcome) for m in result.moved] == [
        (ATT_SP["SR/BPM1"], 10.0, 12.0, "confirmed"),
        (ATT_SP["SR/BPM3"], 10.0, 12.0, "confirmed"),
        (TURNS_SP, 4, 250, "confirmed"),
    ]
    assert result.unsettled == []
    setup = result.setup
    assert setup is not None and not setup.void
    assert setup.by == "alice" and setup.turns_tracked == 250
    assert len(setup.settings_sha) == 64 and int(setup.settings_sha, 16) >= 0

    state = _state(client)
    assert state.setup == setup
    assert state.last_action is not None
    assert state.last_action.name == "setup" and not state.last_action.stopped
    assert state.last_action.moved == result.moved
    (line,) = [
        r
        for r in records
        if r.get("surface") == SURFACE_TRAJECTORY_SERVICE and r["subject"] == "setup"
    ]
    assert line["actor"] == "alice" and line["reason"] == "setup"
    assert setup.settings_sha in line["detail"]


def test_a_setup_is_a_partial_update_and_idempotent(client: TestClient, machine: _Machine) -> None:
    headers = _hold(client)
    first = _setup(client, headers, {"settings": {"turns_tracked": 100}})
    assert [m.name for m in first.moved] == [TURNS_SP]
    # Absent keys are kept as they read.
    assert all(machine.values[ATT_SP[d]] == 10.0 for d in DEVICES)
    machine.writes.clear()

    again = _setup(client, headers, {"settings": {"turns_tracked": 100}})
    assert again.moved == [] and machine.writes == []
    assert again.setup is not None and first.setup is not None
    assert again.setup.settings_sha == first.setup.settings_sha

    other = _setup(client, headers, {"settings": {"attenuation": 11.0}})
    assert other.setup is not None and other.setup.turns_tracked == 100
    assert other.setup.settings_sha != first.setup.settings_sha


def test_a_readback_that_does_not_settle_is_named_and_the_setup_does_not_take(
    client: TestClient, machine: _Machine
) -> None:
    headers = _hold(client)
    _setup(client, headers, {"settings": {}})
    machine.stuck.add(ATT_RB["SR/BPM3"])

    result = _setup(client, headers, {"settings": {"attenuation": 20.0}})

    assert result.unsettled == [ATT_SP["SR/BPM3"]]
    assert result.setup is None
    state = _state(client)
    assert state.setup is not None and state.setup.void
    assert any("did not settle" in notice for notice in state.notices)
    # The moves are listed for put-back all the same.
    assert {m.name for m in result.moved} == set(ATT_SP.values())
    refused = client.post(f"{PANEL}/shot", headers=headers, json={"steps": ["readout"]})
    assert (refused.status_code, _code(refused)) == (409, "setup_required")


def test_a_setup_that_moves_a_setting_voids_the_step(client: TestClient) -> None:
    headers = _hold(client)
    _pend(client, headers[CLIENT_HEADER])
    sessions = client.app.state.sessions  # type: ignore[attr-defined]
    _setup(client, headers, {"settings": {}})
    assert sessions.step.step is not None  # nothing moved: the step stands

    _setup(client, headers, {"settings": {"turns_tracked": 9}})
    assert sessions.step.step is None


class _Solve:
    """The SolveConfig writer's stand-in: records each update and the action it ran in."""

    def __init__(self, sessions: Any) -> None:
        self.sessions = sessions
        self.values: dict[str, Any] = {"excluded_bpms": []}
        self.notices: list[str] = []
        self.solve = None
        self.updates: list[tuple[Any, str, Any]] = []

    async def update(self, partial: Any, cause: str) -> SolveUpdate:
        self.updates.append((partial, cause, self.sessions.running_action()))
        changed: frozenset[str] = frozenset()
        if (
            partial.excluded_bpms is not None
            and partial.excluded_bpms != self.values["excluded_bpms"]
        ):
            self.values["excluded_bpms"] = list(partial.excluded_bpms)
            changed = frozenset({"excluded_bpms"})
        return SolveUpdate(solve=None, changed=changed, recalculated=False)  # type: ignore[arg-type]


def _solve(client: TestClient) -> _Solve:
    sessions = client.app.state.sessions  # type: ignore[attr-defined]
    sessions.solve = _Solve(sessions)
    return sessions.solve


def test_excluded_bpms_reach_the_solve_config_writer_inside_the_setup_action(
    client: TestClient, machine: _Machine, state_dir: Path
) -> None:
    solve = _solve(client)
    headers = _hold(client)
    _pend(client, headers[CLIENT_HEADER])
    machine.batches.clear()

    result = _setup(
        client, headers, {"settings": {"attenuation": 15.0}, "excluded_bpms": [X_ROWS[1]]}
    )

    ((partial, cause, running),) = solve.updates
    assert partial.excluded_bpms == [X_ROWS[1]] and cause == "setup"
    assert running is not None and running.name == "setup"
    # An exclusion change voids the step, as the writer's step rule says.
    assert client.app.state.sessions.step.step is None  # type: ignore[attr-defined]
    # The excluded BPM's settings are neither read nor written.
    assert ATT_SP["SR/BPM2"] not in machine.batches[0]
    assert ATT_SP["SR/BPM2"] not in {m.name for m in result.moved}
    assert machine.values[ATT_SP["SR/BPM2"]] == 10.0
    # Setup never stores exclusions itself.
    persisted = (state_dir / "setup" / "SR.json").read_text(encoding="utf-8")
    assert "excluded" not in persisted and X_ROWS[1] not in persisted


def test_a_setup_without_excluded_bpms_leaves_the_solve_config_alone(client: TestClient) -> None:
    solve = _solve(client)
    _setup(client, _hold(client), {"settings": {"turns_tracked": 6}})
    assert solve.updates == []


def test_a_setup_during_a_continuous_run_is_refused_continuous_running(
    client: TestClient, machine: _Machine, records: list[dict[str, Any]]
) -> None:
    headers = _hold(client)
    sessions = client.app.state.sessions  # type: ignore[attr-defined]
    sessions._continuous = SimpleNamespace(running=True)
    try:
        response = client.post(
            f"{PANEL}/setup", headers=headers, json={"settings": {"turns_tracked": 5}}
        )
    finally:
        sessions._continuous = None
    assert (response.status_code, _code(response)) == (409, "continuous_running")
    assert machine.writes == []
    assert [r["reason"] for r in records if r.get("subject") == "setup"] == ["continuous_running"]


@pytest.mark.parametrize(
    ("body", "status", "code"),
    [
        ({"settings": {"gain": 1}}, 400, "unknown_setting"),
        ({"settings": {"turns_tracked": [1]}}, 400, "bad_request"),
        ({"settings": {}, "owner": "x"}, 400, "owner_in_body"),
        ({"bpms": []}, 400, "bad_request"),
    ],
)
def test_a_bad_setup_body_is_refused_before_any_write(
    client: TestClient, machine: _Machine, body: dict, status: int, code: str
) -> None:
    headers = _hold(client)
    response = client.post(f"{PANEL}/setup", headers=headers, json=body)
    assert (response.status_code, _code(response)) == (status, code)
    assert machine.writes == []


def test_a_setup_by_a_non_holder_is_refused(client: TestClient, machine: _Machine) -> None:
    _hold(client)
    response = client.post(
        f"{PANEL}/setup",
        headers=_as(BOB_SECRET, **{CLIENT_HEADER: "nope"}),
        json={"settings": {"turns_tracked": 5}},
    )
    assert response.status_code == 409
    assert machine.writes == []


def test_the_setup_route_is_panel_only(client: TestClient) -> None:
    headers = _hold(client, "/api")
    response = client.post("/api/setup", headers=headers, json={"settings": {}})
    assert response.status_code in (404, 405)


def test_get_setup_answers_the_typed_readbacks(client: TestClient, machine: _Machine) -> None:
    machine.values[ATT_RB["SR/BPM1"]] = 10.5
    response = client.get(f"{PANEL}/setup", headers=_as(BOB_SECRET))
    assert response.status_code == 200, response.text
    read = SetupReadbacks.model_validate(response.json())
    assert [(s.id, s.scope) for s in read.settings] == [
        ("attenuation", "device"),
        ("turns_tracked", "global"),
    ]
    assert read.settings[0].values == {"SR/BPM1": 10.5, "SR/BPM2": 10.0, "SR/BPM3": 10.0}
    assert read.settings[1].value == 4
    assert read.read_at is not None and read.setup is None

    headers = _hold(client)
    applied = _setup(client, headers, {"settings": {}})
    again = SetupReadbacks.model_validate(
        client.get(f"{PANEL}/setup", headers=_as(BOB_SECRET)).json()
    )
    assert again.setup == applied.setup


# -- persistence and the re-check at start ------------------------------------------------


def test_the_setup_is_persisted_and_held_again_at_start(machine: _Machine, state_dir: Path) -> None:
    with TestClient(_app(machine)) as first:
        applied = _setup(first, _hold(first), {"settings": {"turns_tracked": 7}})
    assert (state_dir / "setup" / "SR.json").is_file()

    with TestClient(_app(machine)) as second:
        state = _state(second)
    assert state.setup == applied.setup
    assert state.notices == []


def test_a_setup_the_machine_left_while_the_service_was_down_is_void_at_start(
    machine: _Machine, state_dir: Path
) -> None:
    with TestClient(_app(machine)) as first:
        _setup(first, _hold(first), {"settings": {"turns_tracked": 7}})
    machine.values[TURNS_RB] = 8

    with TestClient(_app(machine)) as second:
        state = _state(second)
        headers = _hold(second)
        refused = second.post(f"{PANEL}/shot", headers=headers, json={"steps": ["readout"]})
    assert state.setup is not None and state.setup.void
    assert any(TURNS_RB in notice for notice in state.notices)
    assert (refused.status_code, _code(refused)) == (409, "setup_required")
    document = json.loads((state_dir / "setup" / "SR.json").read_text(encoding="utf-8"))
    assert document["setup"]["void"] is True


def test_an_unreadable_setup_file_holds_no_setup_with_a_notice(
    machine: _Machine, state_dir: Path
) -> None:
    (state_dir / "setup").mkdir(parents=True)
    (state_dir / "setup" / "SR.json").write_text("{not json", encoding="utf-8")
    with TestClient(_app(machine)) as client:
        state = _state(client)
    assert state.setup is None
    assert any("could not be read" in notice for notice in state.notices)


# -- the re-check before every chain turn -------------------------------------------------


def test_a_chain_turn_stamps_the_setup_it_ran_under(client: TestClient, machine: _Machine) -> None:
    headers = _hold(client)
    applied = _setup(client, headers, {"settings": {"turns_tracked": 3}})
    machine.batches.clear()

    response = client.post(f"{PANEL}/shot", headers=headers, json={"steps": ["readout"]})

    assert response.status_code == 200, response.text
    shot = ChainResult.model_validate(response.json()).shot
    assert shot is not None and applied.setup is not None
    assert shot.setup_sha == applied.setup.settings_sha
    # The re-check is folded into the pre-inject batch read.
    assert set(ATT_RB.values()) <= set(machine.batches[0])
    assert set(X_ROWS) <= set(machine.batches[0])


def test_a_divergence_before_a_turn_voids_setup_and_step_and_the_shot_carries_no_sha(
    client: TestClient, machine: _Machine
) -> None:
    headers = _hold(client)
    _setup(client, headers, {"settings": {"turns_tracked": 3}})
    _pend(client, headers[CLIENT_HEADER])
    # An outside write moves one BPM's attenuation.
    machine.values[ATT_RB["SR/BPM3"]] = 30.0

    response = client.post(f"{PANEL}/shot", headers=headers, json={"steps": ["readout"]})

    assert response.status_code == 200, response.text
    shot = ChainResult.model_validate(response.json()).shot
    assert shot is not None and shot.setup_sha is None
    state = _state(client)
    assert state.setup is not None and state.setup.void
    assert state.step is None
    assert any(ATT_RB["SR/BPM3"] in notice for notice in state.notices)
    refused = client.post(f"{PANEL}/shot", headers=headers, json={"steps": ["readout"]})
    assert (refused.status_code, _code(refused)) == (409, "setup_required")


def test_a_re_check_ignores_an_excluded_bpm(client: TestClient, machine: _Machine) -> None:
    _solve(client)
    headers = _hold(client)
    _setup(client, headers, {"settings": {"turns_tracked": 3}})
    _setup(client, headers, {"settings": {}, "excluded_bpms": [X_ROWS[2]]})
    machine.values[ATT_RB["SR/BPM3"]] = 30.0

    response = client.post(f"{PANEL}/shot", headers=headers, json={"steps": ["readout"]})

    assert response.status_code == 200, response.text
    state = _state(client)
    assert state.setup is not None and not state.setup.void


# -- the chain's hook, on its own -----------------------------------------------------------


class _Check:
    def __init__(self, answers: list[str | None]) -> None:
        self.answers = list(answers)
        self.seen: list[dict[str, Any]] = []

    def addresses(self) -> list[str]:
        return [TURNS_RB]

    async def check(self, values: Any) -> str | None:
        self.seen.append(dict(values))
        return self.answers.pop(0)


def _rows() -> ShotRows:
    return ShotRows(bpms=X_ROWS, x=X_ROWS, y=Y_ROWS, sum=SUMS)


def _chain(machine: _Machine, tmp_path: Path, check: Any, n_shots: int = 1) -> ChainResult:
    startup = TrajectoryStartup(
        target=TARGET, connector=machine, commands=COMMANDS, guarded_run_dir=Path("/unused")
    )
    action = chain_action(
        startup,
        TrajectoryStore(tmp_path / "chain"),
        rows=_rows(),
        # Built directly: a door refuses repeated readouts, the chain itself does not.
        run=ChainRun(steps=("readout",), n_shots=n_shots),
        setup_check=check,
    )
    return asyncio.run(action(None, asyncio.Event()))  # type: ignore[arg-type]


def test_the_chain_stamps_the_sha_its_re_check_held(machine: _Machine, tmp_path: Path) -> None:
    check = _Check(["a" * 64])
    result = _chain(machine, tmp_path, check)
    assert result.shot is not None and result.shot.setup_sha == "a" * 64
    assert TURNS_RB in check.seen[0]


def test_the_chain_stamps_none_unless_every_injection_held_the_setup(
    machine: _Machine, tmp_path: Path
) -> None:
    check = _Check(["a" * 64, None])
    result = _chain(machine, tmp_path, check, n_shots=2)
    assert len(check.seen) == 2
    assert result.shot is not None and result.shot.setup_sha is None


def test_a_chain_without_a_re_check_stamps_none(machine: _Machine, tmp_path: Path) -> None:
    result = _chain(machine, tmp_path, None)
    assert result.shot is not None and result.shot.setup_sha is None


# -- the pieces ---------------------------------------------------------------------------


def test_the_declared_settings_resolve_readbacks_falling_back_to_the_setpoint() -> None:
    slots = {
        "settings": [
            {
                "id": "mode",
                "scope": "device",
                "setting": {"A": "A:SP", "B": "B:SP"},
                "readback": {"A": "A:RB", "B": None},
                "tolerance": {"A": None, "B": None},
                "timeout_s": None,
            },
            {"id": "turns", "scope": "global", "setting": "T:SP", "readback": None},
        ]
    }
    mode, turns = declared_settings(slots)
    assert mode.readback == {"A": "A:RB", "B": "B:SP"}
    assert mode.timeout_s == 5.0
    assert (turns.setting, turns.readback, turns.tolerance) == ("T:SP", "T:SP", None)


@pytest.mark.parametrize(
    ("read", "wanted", "tolerance", "held"),
    [
        (10.004, 10.0, 0.01, True),
        (10.02, 10.0, 0.01, False),
        (4, 4.0, None, True),
        (4.000001, 4, None, False),
        ("tbt", "tbt", None, True),
        (None, 1, None, False),
        (float("nan"), 1.0, 1.0, False),
        (True, 1, None, True),
    ],
)
def test_a_readback_matches_within_its_tolerance(
    read: Any, wanted: Any, tolerance: float | None, held: bool
) -> None:
    assert matches(read, wanted, tolerance) is held


def test_a_keeper_without_a_setup_reads_nothing_and_vouches_for_nothing() -> None:
    keeper = SetupKeeper(declared_settings(SLOTS), model="SR")
    assert keeper.addresses() == []
    assert asyncio.run(keeper.check({})) is None
    keeper.state = SetupState(applied_at=0.0, by=None, settings_sha="e" * 64, turns_tracked=1)
    # A setup held with nothing to re-check holds as it is.
    assert keeper.addresses() == []
    assert asyncio.run(keeper.check({})) == "e" * 64
