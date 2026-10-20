"""Acquisition: the agent door's default chain arms, injects, reads and averages in one turn.

The shot runs through the real :class:`ActionLayer` -- real admission on a
real control-context tree, the real run lock -- with the fixture sequence
module, under an ``optional`` validator with no records: the arm and the
request are commands, and a command needs no limits record. The stand-in
connector models two arming devices and three BPMs whose x/y/sum records the
injection fills with a fresh shot, stamped later than anything before it.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from tests._guarded_run import reset_run_state

from osprey.runtime.guarded_run import ENV_CONTROL_IDENTITY
from osprey.services.trajectory import actions
from osprey.services.trajectory.acquisition import (
    NO_SEQUENCE_MODULE,
    OVER_BUDGET,
    AcquisitionRefused,
    ShotRows,
    ShotStopped,
    shot_action,
)
from osprey.services.trajectory.actions import ActionLayer
from osprey.services.trajectory.admit import AdmitGate
from osprey.services.trajectory.constants import CHECKPOINT_POLL_S, budgets
from osprey.services.trajectory.sequence import SequenceModule, load_sequence
from osprey.services.trajectory.startup import (
    TrajectoryCommands,
    TrajectoryDevice,
    TrajectorySettings,
    TrajectoryStartup,
)
from osprey.services.trajectory.state import ShotFailure, ShotResult
from osprey.services.trajectory.store import TrajectoryStore
from osprey_connectors import posture_store
from osprey_connectors.control_system.base import (
    ChannelValue,
    ChannelWriteResult,
    WriteOutcome,
)
from osprey_connectors.control_system.limits_validator import LimitsValidator
from osprey_connectors.identity import AUDIT_IDENTITY_ENV

FIXTURE_DIR = Path(__file__).parent / "sequences" / "fixture"

OWNER = "alice"
KIND = "panel"
TARGET = "va"
ARMED = {"writes_enabled": True}
REQUEST = "SR:INJ:REQUEST"
BUSY = "SR:INJ:BUSY"


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
    ):
        monkeypatch.delenv(name, raising=False)
    root = tmp_path / "deploy"
    (root / "build").mkdir(parents=True)
    (root / "build" / "config.yml").write_text("control_system: {}\n", encoding="utf-8")
    monkeypatch.chdir(root)
    from osprey_connectors.workspace import reset_config_cache

    reset_config_cache()
    reset_run_state(monkeypatch)
    tree = root / posture_store.STATE_DIR_NAME
    tree.mkdir(parents=True, exist_ok=True)
    (tree / posture_store.CONTROL_TREE_MARKER_NAME).write_text("provisioned\n", encoding="utf-8")
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, str(tree))
    posture_store.invalidate_cache()
    yield root
    posture_store.invalidate_cache()
    reset_config_cache()


SHOT_ARM = ("SR:BPM:ARM:A", "SR:BPM:ARM:B")
SHOT_ARMED = ("SR:BPM:ARMED:A", "SR:BPM:ARMED:B")
#: The sample of every turn series the injected shot arrives at.
TRIGGER_SAMPLE = 2
SHOT_COMMANDS = TrajectoryCommands(
    model="SR",
    devices=tuple(
        TrajectoryDevice(
            name=f"SR/BPM{device}",
            x_row=f"SR:BPM{device}:X",
            y_row=f"SR:BPM{device}:Y",
            sum_row=f"SR:BPM{device}:SUM",
            arm=arm,
            armed=armed,
        )
        for device, arm, armed in zip("AB", SHOT_ARM, SHOT_ARMED, strict=True)
    ),
    injection_request=REQUEST,
    injection_busy=BUSY,
    trigger_sample=TRIGGER_SAMPLE,
)
BPMS = ("BPM1", "BPM2", "BPM3")
ROWS = ShotRows(
    bpms=BPMS,
    x=tuple(f"{bpm}:X:TBT" for bpm in BPMS),
    y=tuple(f"{bpm}:Y:TBT" for bpm in BPMS),
    sum=tuple(f"{bpm}:SUM:TBT" for bpm in BPMS),
)
SAMPLES = 8
#: The sample the beam arrives at every BPM.
ARRIVAL = 3
EPOCH = datetime(2026, 1, 1, tzinfo=UTC)
CALL_TIMEOUT_S = 0.2
#: Fast settings: every armed wait ends in well under a second.
FAST = TrajectorySettings(armed_timeout_s=0.3)


def _series(level: float, *, beam: bool) -> list[float]:
    return [0.0] * ARRIVAL + [level if beam else 0.0] * (SAMPLES - ARRIVAL)


class _ShotMachine:
    """Two arming devices and three BPMs; an injection fills the BPM records.

    ``drop_arm`` arms nothing (the arm write is lost); ``stale`` holds addresses
    the injection leaves untouched; ``never_clear`` holds armed flags that stay
    1; ``lose_from`` is the first BPM the beam does not reach on each shot.
    """

    _control_target = TARGET

    def __init__(self) -> None:
        self.values: dict[str, Any] = {BUSY: 0}
        self.stamps: dict[str, datetime] = {BUSY: EPOCH}
        for address in (*ROWS.x, *ROWS.y, *ROWS.sum):
            self._post(address, [0.0] * SAMPLES, EPOCH)
        for flag in SHOT_ARMED:
            self._post(flag, 0, EPOCH)
        self.shots = 0
        self.reads: list[str] = []
        self.commands: list[tuple[list[tuple[str, Any]], dict[str, Any]]] = []
        self.drop_arm = False
        self.drop_arm_on: set[int] = set()
        self.stale: set[str] = set()
        self.stale_on: set[int] = set()
        self.never_clear: set[str] = set()
        self.lose_from: list[int | None] = []
        self.x_by_shot: list[float] = []
        self.on_read: Any = None
        self._limits_validator: Any = "never-assigned"

    def _post(self, address: str, value: Any, stamp: datetime) -> None:
        self.values[address] = value
        self.stamps[address] = stamp

    def _current_value_reader(self) -> None:
        return None

    async def read_channel(self, address: str, timeout: float | None = None) -> ChannelValue:  # noqa: ARG002
        self.reads.append(address)
        if self.on_read is not None:
            self.on_read(address)
        return ChannelValue(value=self.values[address], timestamp=self.stamps[address])

    async def read_multiple_channels(
        self, channel_addresses: list[str], timeout: float | None = None
    ) -> dict[str, ChannelValue]:
        return {address: await self.read_channel(address, timeout) for address in channel_addresses}

    async def write_channel_checked(self, address: str, *_: Any, **__: Any) -> Any:
        raise AssertionError(f"a shot wrote the setting {address}")

    async def write_multiple_channels(
        self, operations: list[tuple[str, Any]], **kwargs: Any
    ) -> list[ChannelWriteResult]:
        self.commands.append((list(operations), kwargs))
        for address, value in operations:
            if address in SHOT_ARM and value == 1:
                if not (self.drop_arm or self.shots in self.drop_arm_on):
                    self._post(SHOT_ARMED[SHOT_ARM.index(address)], 1, EPOCH)
            elif address == REQUEST and value == 1:
                self._inject()
        return [
            ChannelWriteResult(
                channel_address=address, value_written=value, outcome=WriteOutcome.UNREQUESTED
            )
            for address, value in operations
        ]

    def _inject(self) -> None:
        if not all(self.values[flag] == 1 for flag in SHOT_ARMED):
            return
        shot = self.shots
        self.shots += 1
        stamp = EPOCH + timedelta(seconds=10 * (shot + 1))
        lose_from = self.lose_from[shot] if shot < len(self.lose_from) else None
        x_level = self.x_by_shot[shot] if shot < len(self.x_by_shot) else 1.0
        for index, _bpm in enumerate(BPMS):
            beam = lose_from is None or index < lose_from
            for address, level in (
                (ROWS.x[index], x_level + index),
                (ROWS.y[index], -x_level - index),
                (ROWS.sum[index], 2.0),
            ):
                if address in self.stale or (shot in self.stale_on and address == ROWS.sum[1]):
                    continue
                self._post(address, _series(level, beam=beam), stamp)
        for flag in SHOT_ARMED:
            if flag not in self.never_clear:
                self._post(flag, 0, stamp)


@pytest.fixture
def optional_validator(monkeypatch: pytest.MonkeyPatch) -> LimitsValidator:
    """An ``optional`` validator with no records: commands pass with no limits."""
    built = LimitsValidator({}, {"mode": "optional", "on_violation": "error"}, {})

    def from_config(*, connector_type: str | None = None, target: str | None = None) -> Any:  # noqa: ARG001
        return built

    monkeypatch.setattr(actions.LimitsValidator, "from_config", staticmethod(from_config))
    return built


def _shot_layer(
    settings: TrajectorySettings = FAST,
    module: SequenceModule | None | str = "fixture",
) -> tuple[ActionLayer, _ShotMachine]:
    machine = _ShotMachine()
    startup = TrajectoryStartup(
        target=TARGET,
        connector=machine,
        commands=SHOT_COMMANDS,
        guarded_run_dir=Path("/unused"),
        settings=settings,
        sequence=load_sequence(FIXTURE_DIR) if module == "fixture" else module,
    )
    return ActionLayer(startup, AdmitGate(target=TARGET, section=ARMED)), machine


async def _shoot(layer: ActionLayer, store: TrajectoryStore, **kwargs: Any) -> ShotResult:
    kwargs.setdefault("rows", ROWS)
    kwargs.setdefault("call_timeout_s", CALL_TIMEOUT_S)
    kwargs.setdefault("poll_s", 0.01)
    try:
        return await layer.submit(OWNER, KIND, shot_action(layer.startup, store, **kwargs))
    finally:
        await layer.close()


@pytest.fixture
def store(tmp_path: Path) -> TrajectoryStore:
    return TrajectoryStore(tmp_path / "state")


# -- a good shot -----------------------------------------------------------------


@pytest.mark.usefixtures("optional_validator")
async def test_a_shot_arms_every_device_in_one_batch_injects_and_reads_the_fresh_first_turn(
    store: TrajectoryStore,
) -> None:
    layer, machine = _shot_layer()

    shot = await _shoot(layer, store)

    # One arm batch over every device, then the module's request; nothing confirmed.
    assert [ops for ops, _ in machine.commands] == [
        [(SHOT_ARM[0], 1), (SHOT_ARM[1], 1)],
        [(REQUEST, 1)],
    ]
    assert all(kwargs == {"confirm": False} for _, kwargs in machine.commands)
    assert shot.commands_fired == [*SHOT_ARM, REQUEST]
    assert shot.model == "SR"
    assert shot.shots_included == 1 and shot.shots_failed == []
    assert shot.bpm_order == list(BPMS)
    assert shot.x == {ROWS.x[0]: 1.0, ROWS.x[1]: 2.0, ROWS.x[2]: 3.0}
    assert shot.y == {ROWS.y[0]: -1.0, ROWS.y[1]: -2.0, ROWS.y[2]: -3.0}
    assert shot.sum == dict.fromkeys(ROWS.sum, 2.0)
    assert shot.lost == [] and shot.last_bpm == "BPM3"
    assert shot.i0 == ARRIVAL
    assert shot.window == (ARRIVAL - TRIGGER_SAMPLE, SAMPLES)
    assert shot.per_shot_lost_at == [None]
    # Armed and injected in the same action: fresh, read turn by turn. Without a
    # turns-tracked readback the raw buffer is what the rows held.
    assert shot.fresh and shot.readout == "tbt" and shot.n_turns is None
    assert shot.raw_samples == SAMPLES
    assert shot.beam_current is None


@pytest.mark.usefixtures("optional_validator")
async def test_a_shot_writes_its_shot_file(store: TrajectoryStore) -> None:
    layer, _ = _shot_layer()

    shot = await _shoot(layer, store)

    path = store.root / "shots" / "SR" / f"{shot.shot_id}.json"
    assert ShotResult.model_validate_json(path.read_text(encoding="utf-8")) == shot


@pytest.mark.usefixtures("optional_validator")
async def test_shots_are_averaged_and_lost_bpms_read_none_at_the_earliest_loss(
    store: TrajectoryStore,
) -> None:
    layer, machine = _shot_layer()
    machine.x_by_shot = [1.0, 3.0]
    machine.lose_from = [None, 2]

    shot = await _shoot(layer, store, n_shots=2)

    assert machine.shots == 2
    assert shot.shots_included == 2
    assert shot.x == {ROWS.x[0]: 2.0, ROWS.x[1]: 3.0, ROWS.x[2]: None}
    assert shot.lost == ["BPM3"] and shot.last_bpm == "BPM2"
    assert shot.per_shot_lost_at == [None, "BPM3"]
    assert shot.commands_fired == [*SHOT_ARM, REQUEST, *SHOT_ARM, REQUEST]


# -- failed shots: reported, excluded, never retried -------------------------------


@pytest.mark.usefixtures("optional_validator")
async def test_a_dropped_arm_write_fails_the_shot_armed_never_set_naming_every_straggler(
    store: TrajectoryStore,
) -> None:
    layer, machine = _shot_layer()
    machine.drop_arm = True

    shot = await _shoot(layer, store)

    assert shot.shots_included == 0
    assert shot.shots_failed == [
        ShotFailure(index=0, reason="armed never set", stragglers=list(SHOT_ARMED))
    ]
    # Never injected, never retried: one arm batch, no request.
    assert [ops for ops, _ in machine.commands] == [[(SHOT_ARM[0], 1), (SHOT_ARM[1], 1)]]
    assert shot.commands_fired == list(SHOT_ARM)
    assert shot.last_bpm is None and shot.i0 is None and shot.window is None
    assert set(shot.x.values()) == {None}


@pytest.mark.usefixtures("optional_validator")
async def test_an_armed_flag_that_never_clears_fails_the_shot_naming_it(
    store: TrajectoryStore,
) -> None:
    layer, machine = _shot_layer()
    machine.never_clear = {SHOT_ARMED[1]}

    shot = await _shoot(layer, store)

    assert shot.shots_failed == [
        ShotFailure(index=0, reason="armed never cleared", stragglers=[SHOT_ARMED[1]])
    ]
    assert shot.shots_included == 0


@pytest.mark.usefixtures("optional_validator")
async def test_a_record_not_refreshed_by_the_shot_fails_it_as_a_stale_buffer(
    store: TrajectoryStore,
) -> None:
    layer, machine = _shot_layer()
    machine.stale = {ROWS.y[1]}

    shot = await _shoot(layer, store)

    assert shot.shots_failed == [ShotFailure(index=0, reason="stale buffer at BPM2")]
    assert shot.shots_included == 0


@pytest.mark.usefixtures("optional_validator")
async def test_a_failed_shot_is_excluded_and_the_rest_are_averaged(
    store: TrajectoryStore,
) -> None:
    layer, machine = _shot_layer()
    machine.x_by_shot = [1.0, 5.0, 3.0]
    machine.stale_on = {1}

    shot = await _shoot(layer, store, n_shots=3)

    assert machine.shots == 3
    assert shot.shots_failed == [ShotFailure(index=1, reason="stale buffer at BPM2")]
    assert shot.shots_included == 2
    assert shot.x[ROWS.x[0]] == 2.0


def test_a_shot_without_a_trajectory_model_refuses_trajectory_unavailable(
    store: TrajectoryStore,
) -> None:
    layer, _ = _shot_layer()
    layer.startup.commands = None

    with pytest.raises(AcquisitionRefused) as caught:
        shot_action(layer.startup, store, rows=ROWS)

    assert caught.value.code == "trajectory_unavailable"
    assert caught.value.status == 503


@pytest.mark.usefixtures("optional_validator")
async def test_a_shot_whose_beam_never_reaches_the_first_bpm_fails_extraction(
    store: TrajectoryStore,
) -> None:
    layer, machine = _shot_layer()
    machine.lose_from = [0]

    shot = await _shoot(layer, store)

    assert shot.shots_failed == [ShotFailure(index=0, reason="no BPM rose")]


# -- the post-shot hook ------------------------------------------------------------


@pytest.mark.usefixtures("optional_validator")
async def test_the_hook_is_called_once_per_shot_action_with_the_averaged_result(
    store: TrajectoryStore,
) -> None:
    layer, machine = _shot_layer()
    machine.x_by_shot = [1.0, 3.0]
    seen: list[ShotResult] = []

    async def hook(result: ShotResult) -> None:
        seen.append(result)

    shot = await _shoot(layer, store, n_shots=2, post_shot=hook)

    assert seen == [shot]
    assert seen[0].shots_included == 2 and seen[0].x[ROWS.x[0]] == 2.0


@pytest.mark.usefixtures("optional_validator")
async def test_the_hook_is_not_called_when_every_shot_fails(store: TrajectoryStore) -> None:
    layer, machine = _shot_layer()
    machine.drop_arm = True
    seen: list[ShotResult] = []

    shot = await _shoot(layer, store, n_shots=2, post_shot=seen.append)

    assert seen == []
    assert len(shot.shots_failed) == 2
    assert (store.root / "shots" / "SR" / f"{shot.shot_id}.json").is_file()


# -- refusals before the turn ------------------------------------------------------


def test_a_shot_over_the_action_budget_is_refused_before_the_turn(
    store: TrajectoryStore,
) -> None:
    # The default chain's injection: readout 2·5, arm 5 + 5, inject 5 + 0.5 + 5 = 30.5 s,
    # so 2 × 30.5 = 61 s > 60 s; the fixture module pauses at most 0.5 s.
    layer, machine = _shot_layer(TrajectorySettings(armed_timeout_s=5.0, action_timeout_s=60.0))

    with pytest.raises(AcquisitionRefused) as caught:
        shot_action(layer.startup, store, rows=ROWS, n_shots=2, call_timeout_s=5.0)

    assert caught.value.code == OVER_BUDGET
    assert caught.value.status == 400
    assert machine.reads == [] and machine.commands == []
    # One shot fits: 30.5 s.
    shot_action(layer.startup, store, rows=ROWS, n_shots=1, call_timeout_s=5.0)


def test_a_shot_without_a_sequence_module_refuses_no_sequence_module(
    store: TrajectoryStore,
) -> None:
    layer, _ = _shot_layer(module=None)

    with pytest.raises(AcquisitionRefused) as caught:
        shot_action(layer.startup, store, rows=ROWS)

    assert caught.value.code == NO_SEQUENCE_MODULE
    assert caught.value.status == 503


# -- a stop mid armed-wait -------------------------------------------------------


@pytest.mark.usefixtures("optional_validator")
async def test_a_stop_mid_armed_wait_ends_the_shot_within_the_stop_budget(
    store: TrajectoryStore,
) -> None:
    # Arming never takes, so the shot sits in its armed wait for armed_timeout_s.
    layer, machine = _shot_layer(TrajectorySettings(armed_timeout_s=5.0))
    machine.drop_arm = True
    module = layer.startup.sequence
    budget = budgets(
        max_pause_s=module.metadata.max_pause_s, call_timeout_s=CALL_TIMEOUT_S
    ).stop_budget_s
    stopped_at: list[float] = []

    def on_read(address: str) -> None:
        if address in SHOT_ARMED and not stopped_at:
            stopped_at.append(time.monotonic())
            layer.renew_stop_event()

    machine.on_read = on_read
    seen: list[ShotResult] = []

    with pytest.raises(ShotStopped) as caught:
        await _shoot(layer, store, post_shot=seen.append, poll_s=CHECKPOINT_POLL_S)

    assert time.monotonic() - stopped_at[0] < budget
    assert caught.value.commands_fired == list(SHOT_ARM)
    assert [ops for ops, _ in machine.commands] == [[(SHOT_ARM[0], 1), (SHOT_ARM[1], 1)]]
    assert seen == []
    assert not (store.root / "shots").exists()
