"""Exclusions at acquisition, and shots derived at another loss cutoff.

A chain turn arms, waits on and reads the included BPMs only, so a dead
excluded BPM never fails a shot; the shot records its ``excluded_bpms`` and
the ``loss_cutoff`` it was extracted with. Editing the loss cutoff outside a
continuous run re-extracts the latest shot's raw buffer as a derived shot that
names its parent, inherits its freshness and mode, fires nothing and keeps no
buffer of its own; the raw read path resolves it to the parent. A pruned
buffer refuses the derivation (410 ``raw_evicted``), and so do exclusions
that changed since the parent was read (409 ``exclusions_changed``).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from tests.services.trajectory import test_acquisition as acquisition_tests
from tests.services.trajectory import test_solve_config as solve_config_tests
from tests.services.trajectory.test_acquisition import (
    BUSY,
    CALL_TIMEOUT_S,
    EPOCH,
    FAST,
    FIXTURE_DIR,
    KIND,
    OWNER,
    REQUEST,
    SAMPLES,
    TARGET,
    TRIGGER_SAMPLE,
    _series,
)

from osprey.services.trajectory.acquisition import (
    ARM,
    DERIVED_SUFFIX,
    EXCLUSIONS_CHANGED,
    INJECT,
    INVALID_CHAIN,
    READOUT,
    AcquisitionRefused,
    ChainRun,
    ShotRows,
    chain_action,
    derive_shot,
)
from osprey.services.trajectory.actions import ActionLayer
from osprey.services.trajectory.admit import AdmitGate
from osprey.services.trajectory.raw import RAW_EVICTED, RawBuffer, RawUnits
from osprey.services.trajectory.sequence import load_sequence
from osprey.services.trajectory.startup import (
    TrajectoryCommands,
    TrajectoryDevice,
    TrajectorySettings,
    TrajectoryStartup,
)
from osprey.services.trajectory.state import ShotResult
from osprey.services.trajectory.store import TrajectoryStore
from osprey_connectors.control_system.base import ChannelValue, ChannelWriteResult, WriteOutcome

# The acquisition tests' deployment and validator, the SolveConfig tests' build.
_deployment = acquisition_tests._deployment
optional_validator = acquisition_tests.optional_validator
goldens = solve_config_tests.goldens
shared = solve_config_tests.shared
state = solve_config_tests.state

#: Three BPMs named by their x rows, as the service's row layout names them.
BPMS = ("BPM1:X", "BPM2:X", "BPM3:X")
ROWS = ShotRows(
    bpms=BPMS,
    x=BPMS,
    y=tuple(bpm.replace(":X", ":Y") for bpm in BPMS),
    sum=tuple(bpm.replace(":X", ":SUM") for bpm in BPMS),
)
#: BPM1 and BPM2 share one arm command and one armed flag; BPM3 has its own.
ARMS = ("ARM:A", "ARM:A", "ARM:C")
FLAGS = ("ARMED:A", "ARMED:A", "ARMED:C")
COMMANDS = TrajectoryCommands(
    model="SR",
    devices=tuple(
        TrajectoryDevice(
            name=f"SR/{bpm}",
            x_row=bpm,
            y_row=y,
            sum_row=total,
            arm=arm,
            armed=armed,
        )
        for bpm, y, total, arm, armed in zip(BPMS, ROWS.y, ROWS.sum, ARMS, FLAGS, strict=True)
    ),
    injection_request=REQUEST,
    injection_busy=BUSY,
    trigger_sample=TRIGGER_SAMPLE,
)


class _Machine:
    """Three BPMs; BPM3's acquisition is dead when ``dead`` is set.

    A dead BPM3 never sets its armed flag and its records cannot be read. Each
    injection fills BPM ``b``'s sum with ``levels[b]``.
    """

    _control_target = TARGET

    def __init__(self, *, dead: bool = False, levels: tuple[float, ...] = (1.0, 1.0, 1.0)):
        self.dead = dead
        self.levels = levels
        self.values: dict[str, Any] = {BUSY: 0}
        self.stamps: dict[str, Any] = {BUSY: EPOCH}
        for address in ROWS.records:
            self._post(address, [0.0] * SAMPLES, EPOCH)
        for flag in dict.fromkeys(FLAGS):
            self._post(flag, 0, EPOCH)
        self.reads: list[str] = []
        self.written: list[str] = []
        self.shots = 0
        self._limits_validator: Any = "never-assigned"

    def _post(self, address: str, value: Any, stamp: Any) -> None:
        self.values[address] = value
        self.stamps[address] = stamp

    def _current_value_reader(self) -> None:
        return None

    def _gone(self, address: str) -> bool:
        return self.dead and (address in ROWS.records[2::3] or address == "ARMED:C")

    async def read_channel(self, address: str, timeout: float | None = None) -> ChannelValue:
        del timeout
        self.reads.append(address)
        return ChannelValue(value=self.values[address], timestamp=self.stamps[address])

    async def read_multiple_channels(
        self, channel_addresses: list[str], timeout: float | None = None
    ) -> dict[str, ChannelValue]:
        del timeout
        self.reads.extend(channel_addresses)
        return {
            address: ChannelValue(value=self.values[address], timestamp=self.stamps[address])
            for address in channel_addresses
            if not self._gone(address)
        }

    async def write_channel_checked(self, address: str, *_: Any, **__: Any) -> Any:
        raise AssertionError(f"a shot wrote the setting {address}")

    async def write_multiple_channels(
        self, operations: list[tuple[str, Any]], **_: Any
    ) -> list[ChannelWriteResult]:
        for address, value in operations:
            self.written.append(address)
            if address in ARMS and value == 1:
                flag = FLAGS[ARMS.index(address)]
                if not self._gone(flag):
                    self._post(flag, 1, EPOCH)
            elif address == REQUEST and value == 1:
                self._inject()
        return [
            ChannelWriteResult(
                channel_address=address, value_written=value, outcome=WriteOutcome.UNREQUESTED
            )
            for address, value in operations
        ]

    def _inject(self) -> None:
        self.shots += 1
        stamp = EPOCH.replace(second=10 * self.shots)
        for index in range(len(BPMS)):
            for address, level in (
                (ROWS.x[index], 0.1 * (index + 1)),
                (ROWS.y[index], -0.1 * (index + 1)),
                (ROWS.sum[index], self.levels[index]),
            ):
                self._post(address, _series(level, beam=True), stamp)
        for flag in dict.fromkeys(FLAGS):
            self._post(flag, 0, stamp)


def _layer(machine: _Machine, settings: TrajectorySettings = FAST) -> ActionLayer:
    startup = TrajectoryStartup(
        target=TARGET,
        connector=machine,
        commands=COMMANDS,
        guarded_run_dir=Path("/unused"),
        settings=settings,
        sequence=load_sequence(FIXTURE_DIR),
    )
    return ActionLayer(startup, AdmitGate(target=TARGET, section={"writes_enabled": True}))


RUN = ChainRun(steps=(ARM, INJECT, READOUT))


async def _chain(
    layer: ActionLayer, store: TrajectoryStore, run: ChainRun = RUN, **kwargs: Any
) -> ShotResult:
    kwargs.setdefault("call_timeout_s", CALL_TIMEOUT_S)
    kwargs.setdefault("poll_s", 0.01)
    try:
        result = await layer.submit(
            OWNER, KIND, chain_action(layer.startup, store, rows=ROWS, run=run, **kwargs)
        )
    finally:
        await layer.close()
    assert result.shot is not None
    return result.shot


@pytest.fixture
def store(tmp_path: Path) -> TrajectoryStore:
    return TrajectoryStore(tmp_path / "shots")


# -- exclusions reach acquisition ----------------------------------------------------


@pytest.mark.usefixtures("optional_validator")
async def test_a_dead_bpm_fails_every_shot_while_it_is_included(store: TrajectoryStore) -> None:
    shot = await _chain(_layer(_Machine(dead=True)), store)

    assert shot.shots_included == 0
    assert shot.shots_failed[0].reason == "no reading at BPM3:X"


@pytest.mark.usefixtures("optional_validator")
async def test_an_excluded_dead_bpm_is_neither_armed_waited_on_nor_read(
    store: TrajectoryStore,
) -> None:
    machine = _Machine(dead=True)

    shot = await _chain(_layer(machine), store, excluded_bpms=["BPM3:X"])

    assert shot.shots_included == 1 and shot.shots_failed == []
    assert "ARM:C" not in machine.written and "ARM:A" in machine.written
    assert not {"ARMED:C", *ROWS.records[2::3]} & set(machine.reads)
    assert shot.excluded_bpms == ["BPM3:X"]
    assert shot.bpm_order == list(BPMS)
    assert shot.x == {"BPM1:X": pytest.approx(0.1), "BPM2:X": pytest.approx(0.2), "BPM3:X": None}
    assert shot.sum["BPM3:SUM"] is None
    assert shot.lost == [] and shot.last_bpm == "BPM2:X"
    _root, raw = await store.load_raw("SR", shot.shot_id)
    assert raw.header.shape[2] == 2


@pytest.mark.usefixtures("optional_validator")
async def test_a_shared_arm_stays_while_one_of_its_devices_is_included(
    store: TrajectoryStore,
) -> None:
    machine = _Machine()

    await _chain(_layer(machine), store, excluded_bpms=["BPM1:X"])

    assert machine.written.count("ARM:A") == 1 and "ARM:C" in machine.written
    assert "ARMED:A" in machine.reads


@pytest.mark.usefixtures("optional_validator")
async def test_excluding_every_bpm_is_refused_before_the_turn(store: TrajectoryStore) -> None:
    layer = _layer(_Machine())
    try:
        with pytest.raises(AcquisitionRefused) as caught:
            chain_action(layer.startup, store, rows=ROWS, run=RUN, excluded_bpms=BPMS)
    finally:
        await layer.close()

    assert caught.value.code == INVALID_CHAIN


@pytest.mark.usefixtures("optional_validator")
async def test_a_shot_records_its_mode_and_the_loss_cutoff_it_was_extracted_with(
    store: TrajectoryStore,
) -> None:
    machine = _Machine(levels=(1.0, 0.6, 1.0))

    default = await _chain(_layer(machine), store)
    raised = await _chain(_layer(machine), store, loss_cutoff=0.7)

    assert default.loss_cutoff == FAST.loss_cutoff_fraction and default.lost == []
    assert raised.loss_cutoff == 0.7 and raised.lost == ["BPM2:X", "BPM3:X"]
    assert (raised.readout, raised.average, raised.n_turns) == ("tbt", "mean", None)
    assert raised.excluded_bpms == []


# -- derived shots -------------------------------------------------------------------


@pytest.mark.usefixtures("optional_validator")
async def test_a_derived_shot_re_extracts_the_parent_at_the_new_cutoff(
    store: TrajectoryStore,
) -> None:
    run = ChainRun(steps=(ARM, INJECT, READOUT), n_shots=2, average="nanmean")
    parent = await _chain(_layer(_Machine(levels=(1.0, 0.6, 1.0))), store, run=run)
    assert parent.lost == []

    derived = await derive_shot(
        store, "SR", parent.shot_id, loss_cutoff=0.7, excluded_bpms=[], settings=FAST
    )

    assert derived.shot_id == f"{parent.shot_id}{DERIVED_SUFFIX}"
    assert derived.parent_id == parent.shot_id
    assert derived.loss_cutoff == 0.7
    assert derived.lost == ["BPM2:X", "BPM3:X"] and derived.last_bpm == "BPM1:X"
    assert derived.shots_included == 2 and derived.average == "nanmean"
    assert (derived.fresh, derived.readout) == (parent.fresh, parent.readout)
    assert derived.commands_fired == []
    assert derived.timestamp == parent.timestamp
    assert derived.raw_samples == parent.raw_samples
    assert await store.load_shot("SR", derived.shot_id) == derived


@pytest.mark.usefixtures("optional_validator")
async def test_the_raw_read_path_resolves_a_derived_shot_to_its_parent(
    store: TrajectoryStore,
) -> None:
    parent = await _chain(_layer(_Machine()), store)
    derived = await derive_shot(
        store, "SR", parent.shot_id, loss_cutoff=0.3, excluded_bpms=[], settings=FAST
    )

    root, raw = await store.load_raw("SR", derived.shot_id)
    _, parent_raw = await store.load_raw("SR", parent.shot_id)

    assert root == parent.shot_id
    np.testing.assert_array_equal(raw.samples, parent_raw.samples)


@pytest.mark.usefixtures("optional_validator")
async def test_a_second_edit_replaces_the_derived_shot_of_the_same_parent(
    store: TrajectoryStore,
) -> None:
    parent = await _chain(_layer(_Machine(levels=(1.0, 0.6, 1.0))), store)

    await derive_shot(store, "SR", parent.shot_id, loss_cutoff=0.7, excluded_bpms=[], settings=FAST)
    again = await derive_shot(
        store, "SR", parent.shot_id, loss_cutoff=0.2, excluded_bpms=[], settings=FAST
    )

    listed = await store.list_shots("SR")
    assert sorted(shot.shot_id for shot in listed) == sorted([parent.shot_id, again.shot_id])
    assert again.lost == []


@pytest.mark.usefixtures("optional_validator")
async def test_a_derived_shot_keeps_the_parents_exclusions(store: TrajectoryStore) -> None:
    parent = await _chain(
        _layer(_Machine(dead=True, levels=(1.0, 0.6, 1.0))), store, excluded_bpms=["BPM3:X"]
    )

    derived = await derive_shot(
        store, "SR", parent.shot_id, loss_cutoff=0.7, excluded_bpms=["BPM3:X"], settings=FAST
    )

    assert derived.excluded_bpms == ["BPM3:X"]
    assert derived.lost == ["BPM2:X"] and derived.x["BPM3:X"] is None


@pytest.mark.usefixtures("optional_validator")
async def test_changed_exclusions_refuse_the_derivation(store: TrajectoryStore) -> None:
    parent = await _chain(_layer(_Machine()), store)

    with pytest.raises(AcquisitionRefused) as caught:
        await derive_shot(
            store, "SR", parent.shot_id, loss_cutoff=0.7, excluded_bpms=["BPM2:X"], settings=FAST
        )

    assert (caught.value.code, caught.value.status) == (EXCLUSIONS_CHANGED, 409)
    assert [shot.shot_id for shot in await store.list_shots("SR")] == [parent.shot_id]


@pytest.mark.usefixtures("optional_validator")
async def test_a_pruned_buffer_refuses_the_derivation(tmp_path: Path) -> None:
    store = TrajectoryStore(tmp_path / "pruned", shots_per_model=1)
    first = await _chain(_layer(_Machine()), store)
    await _chain(_layer(_Machine()), store)

    with pytest.raises(AcquisitionRefused) as caught:
        await derive_shot(
            store, "SR", first.shot_id, loss_cutoff=0.7, excluded_bpms=[], settings=FAST
        )

    assert (caught.value.code, caught.value.status) == (RAW_EVICTED, 410)


# -- the loss-cutoff hook ------------------------------------------------------------


def _parent(model: str, shot_id: str = "20260101T000000000000Z-parent") -> ShotResult:
    return ShotResult(
        shot_id=shot_id,
        model=model,
        x=dict.fromkeys(ROWS.x),
        y=dict.fromkeys(ROWS.y),
        sum=dict.fromkeys(ROWS.sum),
        bpm_order=list(BPMS),
        lost=[],
        last_bpm=None,
        i0=None,
        window=None,
        per_shot_lost_at=[],
        shots_included=1,
        shots_failed=[],
        commands_fired=["ARM:A"],
        loss_cutoff=0.5,
        raw_samples=SAMPLES,
    )


def _raw(levels: tuple[float, ...]) -> RawBuffer:
    planes = [np.zeros((len(BPMS), SAMPLES)) for _ in range(3)]
    for index, level in enumerate(levels):
        planes[2][index] = _series(level, beam=True)
    return RawBuffer.of([planes], trigger_sample=TRIGGER_SAMPLE, units=RawUnits())


def _sessions(latest: str | None, *, running: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        continuous_running=running,
        latest_shot_id=latest,
        layer=SimpleNamespace(startup=SimpleNamespace(settings=TrajectorySettings())),
    )


async def test_a_cutoff_edit_outside_a_run_publishes_a_derived_shot(
    shared: Any, state: Path, tmp_path: Path
) -> None:
    config = await solve_config_tests._load(shared, state)
    store = TrajectoryStore(tmp_path / "hook")
    parent = _parent(config.model)
    await store.save_shot(parent, _raw((1.0, 0.6, 1.0)))
    published: list[ShotResult] = []
    config.derive_on_cutoff(_sessions(parent.shot_id), store, published.append)

    update = await config.update({"loss_cutoff": 0.7}, "operator")

    assert update.changed == {"loss_cutoff"}
    assert [shot.parent_id for shot in published] == [parent.shot_id]
    assert published[0].lost == ["BPM2:X", "BPM3:X"] and published[0].loss_cutoff == 0.7


async def test_a_cutoff_edit_during_a_run_persists_and_derives_nothing(
    shared: Any, state: Path, tmp_path: Path
) -> None:
    config = await solve_config_tests._load(shared, state)
    store = TrajectoryStore(tmp_path / "hook")
    parent = _parent(config.model)
    await store.save_shot(parent, _raw((1.0, 1.0, 1.0)))
    published: list[ShotResult] = []
    config.derive_on_cutoff(_sessions(parent.shot_id, running=True), store, published.append)

    await config.update({"loss_cutoff": 0.7}, "operator")

    assert published == []
    assert config.values["loss_cutoff"] == 0.7
    assert [shot.shot_id for shot in await store.list_shots(config.model)] == [parent.shot_id]


async def test_a_cutoff_edit_without_a_shot_derives_nothing(
    shared: Any, state: Path, tmp_path: Path
) -> None:
    config = await solve_config_tests._load(shared, state)
    published: list[ShotResult] = []
    config.derive_on_cutoff(_sessions(None), TrajectoryStore(tmp_path / "hook"), published.append)

    await config.update({"loss_cutoff": 0.7}, "operator")

    assert published == [] and config.values["loss_cutoff"] == 0.7


async def test_changed_exclusions_persist_the_edit_and_refuse_the_derivation_only(
    shared: Any, state: Path, tmp_path: Path
) -> None:
    config = await solve_config_tests._load(shared, state)
    store = TrajectoryStore(tmp_path / "hook")
    parent = _parent(config.model).model_copy(update={"excluded_bpms": ["BPM3:X"]})
    await store.save_shot(parent, _raw((1.0, 1.0)))
    published: list[ShotResult] = []
    config.derive_on_cutoff(_sessions(parent.shot_id), store, published.append)

    with pytest.raises(AcquisitionRefused) as caught:
        await config.update({"loss_cutoff": 0.7}, "operator")

    assert (caught.value.code, caught.value.status) == (EXCLUSIONS_CHANGED, 409)
    assert config.values["loss_cutoff"] == 0.7
    assert config.overrides == {"loss_cutoff": 0.7}
    assert published == []


async def test_an_async_publisher_is_awaited(shared: Any, state: Path, tmp_path: Path) -> None:
    config = await solve_config_tests._load(shared, state)
    store = TrajectoryStore(tmp_path / "hook")
    parent = _parent(config.model)
    await store.save_shot(parent, _raw((1.0, 1.0, 1.0)))
    published: list[str] = []

    async def publish(shot: ShotResult) -> None:
        await asyncio.sleep(0)
        published.append(shot.shot_id)

    config.derive_on_cutoff(_sessions(parent.shot_id), store, publish)
    await config.update({"loss_cutoff": 0.3}, "operator")

    assert published == [f"{parent.shot_id}{DERIVED_SUFFIX}"]
