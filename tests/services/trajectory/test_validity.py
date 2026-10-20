"""Step validity: a step is appliable while its shot's setpoint snapshot still matches.

A shot that reads out stores the full RM-column corrector setpoint vector and
its ``setup_sha``, read in one batch after the readout; with ``n_shots > 1`` the
vector is also read before the first injection, and a difference between the
two reads makes the shot not fresh, so it makes no step. An apply re-reads the
same vector in one batch with the setup re-check's readbacks, before the
limits check, and refuses with 409 ``machine_moved`` when a setpoint moved by
more than ``1e-6 × max_step`` or the setup identity changed.

Everything runs through real :class:`Sessions` on a real :class:`ActionLayer`
with the real chain action, its post-shot hook computing the step over a fake
pyAML tool, and the real apply. The connector is the acquisition tests'
stand-in machine extended with corrector setpoints whose checked write runs the
turn's limits validator, the way every connector does.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from tests.services.trajectory import test_acquisition as acquisition_tests
from tests.services.trajectory.test_acquisition import (
    CALL_TIMEOUT_S,
    FAST,
    FIXTURE_DIR,
    KIND,
    OWNER,
    ROWS,
    SHOT_COMMANDS,
    TARGET,
    _ShotMachine,
)

from osprey.services.trajectory import actions
from osprey.services.trajectory.acquisition import (
    OVER_BUDGET,
    SNAPSHOT_TOLERANCE,
    AcquisitionRefused,
    chain_budget_s,
    chain_declaration,
    derive_shot,
    makes_step,
    moved_setpoints,
    n_shots_max,
    setpoint_names,
    shot_action,
    snapshot_budget_s,
)
from osprey.services.trajectory.actions import ActionLayer
from osprey.services.trajectory.admit import AdmitGate
from osprey.services.trajectory.apply import (
    LIMITS_VIOLATION,
    MACHINE_MOVED,
    ApplyRefused,
    apply,
)
from osprey.services.trajectory.sequence import load_sequence
from osprey.services.trajectory.sessions import Sessions
from osprey.services.trajectory.solve_config import SolveUpdate
from osprey.services.trajectory.startup import TrajectorySettings, TrajectoryStartup
from osprey.services.trajectory.state import ShotResult, Step
from osprey.services.trajectory.step import TrajectoryTool
from osprey.services.trajectory.store import TrajectoryStore
from osprey_connectors.control_system.base import ChannelValue, ChannelWriteResult, WriteOutcome
from osprey_connectors.control_system.call_timeout import DEFAULT_TIMEOUT_S
from osprey_connectors.control_system.limits_validator import ChannelLimitsConfig, LimitsValidator
from osprey_connectors.errors import ChannelLimitsViolationError, ChannelWriteBlockedError

# The acquisition tests' deployment and store fixtures, shared here.
_deployment = acquisition_tests._deployment
store = acquisition_tests.store

ARMED = {"writes_enabled": True}
HCM1, HCM2, VCM1 = "SR:HCM1", "SR:HCM2", "SR:VCM1"
H_CORR = (HCM1, HCM2)
V_CORR = (VCM1,)
CORRECTORS = (*H_CORR, *V_CORR)
SETUP_RB = "SR:SETUP:RB"

#: The corrector records: HCM1 and HCM2 state ``max_step``, VCM1 states none.
RECORDS = {
    HCM1: {"min_value": -10.0, "max_value": 10.0, "max_step": 0.5},
    HCM2: {"min_value": -10.0, "max_value": 10.0, "max_step": 1.0},
    VCM1: {"min_value": -10.0, "max_value": 10.0},
}
#: The setpoints the machine starts at.
START = {HCM1: 1.0, HCM2: -2.0, VCM1: 0.5}
#: The step the fake tool answers, HCM1 first; inside every ``max_step``.
STEP = (0.2, -0.3, 0.1)


# -- the machine, the tool and the setup --------------------------------------------


class _Machine(_ShotMachine):
    """The acquisition stand-in with corrector setpoints and checked writes.

    ``batches`` lists every batch read; ``kick_on_inject`` moves setpoints when
    an injection fires, as another writer would mid-run.
    """

    def __init__(self) -> None:
        super().__init__()
        for name, value in START.items():
            self._post(name, value, acquisition_tests.EPOCH)
        self._post(SETUP_RB, 1, acquisition_tests.EPOCH)
        self.batches: list[list[str]] = []
        self.setpoint_writes: list[tuple[str, Any]] = []
        self.kick_on_inject: dict[str, float] = {}
        self.fail_batch_with: set[str] = set()

    async def read_multiple_channels(
        self, channel_addresses: list[str], timeout: float | None = None
    ) -> dict[str, ChannelValue]:
        self.batches.append(list(channel_addresses))
        if self.fail_batch_with & set(channel_addresses):
            raise TimeoutError("the batch read timed out")
        return await super().read_multiple_channels(channel_addresses, timeout)

    async def write_channel_checked(self, address: str, value: Any, **_: Any) -> Any:
        validator = self._limits_validator
        if validator is not None:
            try:
                validator.validate(address, value, read_current=lambda a: self.values[a])
            except ChannelLimitsViolationError as exc:
                raise ChannelWriteBlockedError(address, "LIMITS", message=str(exc)) from exc
        self.setpoint_writes.append((address, value))
        self._post(address, value, acquisition_tests.EPOCH)
        return ChannelWriteResult(
            channel_address=address, value_written=value, outcome=WriteOutcome.CONFIRMED
        )

    def kick(self, name: str, delta: float) -> None:
        """A ``channel_write`` from outside the service."""
        self._post(name, self.values[name] + delta, acquisition_tests.EPOCH)

    def _inject(self) -> None:
        super()._inject()
        for name, delta in self.kick_on_inject.items():
            self.kick(name, delta)

    def snapshot_batches(self) -> list[list[str]]:
        return [batch for batch in self.batches if HCM1 in batch]


class _Correct:
    """A fake pyAML tool answering ``correct`` with a set step."""

    def __init__(self, step: tuple[float, ...] = STEP) -> None:
        self.step = np.asarray(step, dtype=float)

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
            step=self.step,
            names=CORRECTORS,
            reason=SimpleNamespace(value="OK"),
            status="",
            all_zero=not np.any(self.step),
            zeroed_h=(),
            zeroed_v=(),
            pseudo_inverse=inverse,
        )


def _tool() -> TrajectoryTool:
    return TrajectoryTool(
        model="SR",
        tool=_Correct(),
        layout=SimpleNamespace(
            bpm_names_h=ROWS.x, bpm_names_v=ROWS.y, hcorr_names=H_CORR, vcorr_names=V_CORR
        ),
        bpm_index=(1.0, 2.0, 3.0, 1.0, 2.0, 3.0),
        hcorr_index=(0.5, 1.5),
        vcorr_index=(2.5,),
        unsolved={},
        alpha=0.0,
        n_discard=0,
        gain=1.0,
    )


class _Setup:
    """The setup re-check: holds ``sha-1`` while its readback reads 1."""

    def __init__(self) -> None:
        self.checks: list[Any] = []

    def addresses(self) -> list[str]:
        return [SETUP_RB]

    async def check(self, values: Any) -> str | None:
        self.checks.append(values)
        reading = values.get(SETUP_RB)
        return "sha-1" if reading is not None and reading.value == 1 else None


@pytest.fixture(autouse=True)
def validator(monkeypatch: pytest.MonkeyPatch) -> LimitsValidator:
    """The validator every action turn builds; commands and unlisted channels pass (optional)."""
    built = LimitsValidator(
        {
            address: ChannelLimitsConfig(
                address,
                record["min_value"],
                record["max_value"],
                max_step=record.get("max_step"),
                writable=True,
            )
            for address, record in RECORDS.items()
        },
        {"mode": "optional", "on_violation": "error"},
        {address: {**record, "confirm": True} for address, record in RECORDS.items()},
    )
    monkeypatch.setattr(actions.LimitsValidator, "from_config", staticmethod(lambda **_kw: built))
    return built


def _sessions(settings: TrajectorySettings = FAST) -> tuple[Sessions, _Machine]:
    machine = _Machine()
    startup = TrajectoryStartup(
        target=TARGET,
        connector=machine,
        commands=SHOT_COMMANDS,
        guarded_run_dir=Path("/unused"),
        settings=settings,
        sequence=load_sequence(FIXTURE_DIR),
        tool=_tool(),
    )
    layer = ActionLayer(startup, AdmitGate(target=TARGET, section=ARMED))
    return Sessions(layer, call_timeout_s=CALL_TIMEOUT_S), machine


async def _no_plans(_target: str) -> bool:
    return False


async def _shoot(
    sessions: Sessions, client_id: str, store: TrajectoryStore, **kwargs: Any
) -> ShotResult:
    startup = sessions.layer.startup

    def build(hook: Any) -> Any:
        return shot_action(
            startup,
            store,
            rows=ROWS,
            post_shot=hook,
            call_timeout_s=CALL_TIMEOUT_S,
            poll_s=0.01,
            **kwargs,
        )

    return await sessions.shot(OWNER, KIND, client_id, store, build)


async def _apply(sessions: Sessions, client_id: str, step: Step, **kwargs: Any) -> Any:
    kwargs.setdefault("plan_running", _no_plans)
    return await apply(sessions, OWNER, KIND, client_id, {"step_digest": step.digest}, **kwargs)


async def _refused(coro: Any) -> ApplyRefused:
    with pytest.raises(ApplyRefused) as caught:
        await coro
    return caught.value


def _pending(sessions: Sessions) -> Step:
    step = sessions.step.step
    assert step is not None, "no step is pending"
    return step


# -- the shot's snapshot ------------------------------------------------------------


async def test_a_shot_stores_the_full_rm_column_setpoint_vector_read_in_one_batch_after_readout(
    store: TrajectoryStore,
) -> None:
    sessions, machine = _sessions()
    try:
        client_id = (await sessions.open(OWNER, KIND)).client_id
        shot = await _shoot(sessions, client_id, store)
    finally:
        await sessions.layer.close()

    assert shot.setpoints == START
    assert list(shot.setpoints) == list(CORRECTORS)
    # One shot, one snapshot batch: the vector and nothing else, after the readout.
    assert machine.snapshot_batches() == [list(CORRECTORS)]
    assert machine.batches[-1] == list(CORRECTORS)
    assert shot.fresh and makes_step(shot)
    stored = await store.load_shot("SR", shot.shot_id)
    assert stored.setpoints == START


async def test_the_agent_doors_shot_carries_no_setup_sha_beside_its_setpoints(
    store: TrajectoryStore,
) -> None:
    sessions, _machine = _sessions()
    try:
        client_id = (await sessions.open(OWNER, KIND)).client_id
        shot = await _shoot(sessions, client_id, store, n_shots=1)
    finally:
        await sessions.layer.close()
    # The agent door's chain runs without a setup check, so its identity is no setup.
    assert shot.setup_sha is None and shot.setpoints == START


async def test_n_shots_above_one_reads_the_vector_before_the_first_shot_too(
    store: TrajectoryStore,
) -> None:
    sessions, machine = _sessions()
    try:
        client_id = (await sessions.open(OWNER, KIND)).client_id
        shot = await _shoot(sessions, client_id, store, n_shots=3)
    finally:
        await sessions.layer.close()

    assert machine.snapshot_batches() == [list(CORRECTORS), list(CORRECTORS)]
    # The first batch of the turn is the snapshot, before any injection's pre-inject read.
    assert machine.batches[0] == list(CORRECTORS)
    assert machine.batches[-1] == list(CORRECTORS)
    assert shot.fresh and shot.setpoints == START and shot.shots_included == 3
    assert sessions.step.step is not None


async def test_a_setpoint_that_moved_between_the_two_reads_makes_the_shot_not_fresh_with_no_step(
    store: TrajectoryStore,
) -> None:
    sessions, machine = _sessions()
    machine.kick_on_inject = {HCM2: 0.01}
    try:
        client_id = (await sessions.open(OWNER, KIND)).client_id
        shot = await _shoot(sessions, client_id, store, n_shots=2)
    finally:
        await sessions.layer.close()

    assert shot.fresh is False
    assert not makes_step(shot)
    assert sessions.step.step is None
    # The stored vector is the one read after the readout.
    assert shot.setpoints is not None and shot.setpoints[HCM2] == pytest.approx(-1.98)


async def test_a_drift_within_the_tolerance_between_the_two_reads_keeps_the_shot_fresh(
    store: TrajectoryStore,
) -> None:
    sessions, machine = _sessions()
    # HCM2's max_step is 1.0: half the tolerance per injection, two injections.
    machine.kick_on_inject = {HCM2: 0.25 * SNAPSHOT_TOLERANCE}
    try:
        client_id = (await sessions.open(OWNER, KIND)).client_id
        shot = await _shoot(sessions, client_id, store, n_shots=2)
    finally:
        await sessions.layer.close()

    assert shot.fresh and sessions.step.step is not None


async def test_a_failed_snapshot_read_leaves_no_setpoints_and_makes_no_step(
    store: TrajectoryStore,
) -> None:
    sessions, machine = _sessions()
    machine.fail_batch_with = {HCM1}
    try:
        client_id = (await sessions.open(OWNER, KIND)).client_id
        shot = await _shoot(sessions, client_id, store)
    finally:
        await sessions.layer.close()

    assert shot.setpoints is None and shot.fresh is False
    assert shot.shots_included == 1  # the readout itself was good
    assert sessions.step.step is None


async def test_a_derived_shot_inherits_its_parents_setpoints_and_setup(
    store: TrajectoryStore,
) -> None:
    sessions, _machine = _sessions()
    try:
        client_id = (await sessions.open(OWNER, KIND)).client_id
        parent = await _shoot(sessions, client_id, store)
    finally:
        await sessions.layer.close()

    derived = await derive_shot(
        store, "SR", parent.shot_id, loss_cutoff=0.3, excluded_bpms=(), settings=FAST
    )

    assert derived.parent_id == parent.shot_id
    assert derived.setpoints == parent.setpoints == START
    assert derived.setup_sha == parent.setup_sha
    assert derived.fresh == parent.fresh


def test_the_snapshot_names_are_the_tools_rm_columns_and_none_without_a_tool() -> None:
    sessions, _machine = _sessions()
    assert setpoint_names(sessions.layer.startup) == CORRECTORS
    bare = TrajectoryStartup(
        target=TARGET,
        connector=_Machine(),
        commands=SHOT_COMMANDS,
        guarded_run_dir=Path("/unused"),
        settings=FAST,
    )
    assert setpoint_names(bare) == ()


def test_a_setpoint_matches_within_the_tolerance_of_its_max_step_and_exactly_without_one() -> None:
    snapshot = {HCM1: 1.0, VCM1: 0.5}
    bounds = {HCM1: 0.5}
    tolerance = SNAPSHOT_TOLERANCE * 0.5

    assert moved_setpoints(snapshot, {HCM1: 1.0 + 0.9 * tolerance, VCM1: 0.5}, bounds) == []
    assert moved_setpoints(snapshot, {HCM1: 1.0 + 1.1 * tolerance, VCM1: 0.5}, bounds) == [HCM1]
    assert moved_setpoints(snapshot, {HCM1: 1.0, VCM1: 0.5 + 1e-12}, bounds) == [VCM1]
    # A name the read lacks, or that reads no number, moved.
    assert moved_setpoints(snapshot, {HCM1: 1.0}, bounds) == [VCM1]
    assert moved_setpoints(snapshot, {HCM1: None, VCM1: 0.5}, bounds) == [HCM1]


# -- the apply's re-read ------------------------------------------------------------


async def test_an_unmoved_machine_applies_reading_the_vector_in_one_batch_before_the_writes(
    store: TrajectoryStore,
) -> None:
    sessions, machine = _sessions()
    try:
        client_id = (await sessions.open(OWNER, KIND)).client_id
        await _shoot(sessions, client_id, store)
        machine.batches.clear()
        machine.reads.clear()
        step = _pending(sessions)
        done = await _apply(sessions, client_id, step)
    finally:
        await sessions.layer.close()

    assert not done.stopped
    assert machine.batches == [list(CORRECTORS)]
    # Each corrector was read once, inside that batch, and never one at a time after it.
    assert [read for read in machine.reads if read in CORRECTORS] == list(CORRECTORS)
    assert [name for name, _ in machine.setpoint_writes] == list(CORRECTORS)
    assert machine.values[HCM1] == pytest.approx(START[HCM1] + step.deltas[HCM1])


async def test_apply_then_reselect_then_apply_is_refused_machine_moved(
    store: TrajectoryStore,
) -> None:
    sessions, machine = _sessions()
    try:
        client_id = (await sessions.open(OWNER, KIND)).client_id
        await _shoot(sessions, client_id, store)
        await _apply(sessions, client_id, _pending(sessions))
        # Reselecting the reference recomputes a fresh, unconsumed step on the same shot.
        again = await sessions.select_reference(OWNER, KIND, client_id, store, None)
        assert again is not None and not sessions.step.consumed
        written = list(machine.setpoint_writes)

        refused = await _refused(_apply(sessions, client_id, again))
    finally:
        await sessions.layer.close()

    assert refused.code == MACHINE_MOVED and refused.status == 409
    assert HCM1 in refused.detail
    assert machine.setpoint_writes == written


async def test_a_channel_write_kick_after_the_shot_refuses_the_apply_machine_moved(
    store: TrajectoryStore,
) -> None:
    sessions, machine = _sessions()
    try:
        client_id = (await sessions.open(OWNER, KIND)).client_id
        await _shoot(sessions, client_id, store)
        machine.kick(VCM1, 0.05)

        refused = await _refused(_apply(sessions, client_id, _pending(sessions)))
    finally:
        await sessions.layer.close()

    assert refused.code == MACHINE_MOVED
    assert VCM1 in refused.detail
    assert machine.setpoint_writes == []
    assert not sessions.step.consumed


async def test_a_kick_on_a_corrector_the_step_leaves_alone_still_refuses_the_apply(
    store: TrajectoryStore,
) -> None:
    # The snapshot is the full RM-column vector, not just the correctors the step moves.
    sessions, machine = _sessions()
    try:
        client_id = (await sessions.open(OWNER, KIND)).client_id
        await _shoot(sessions, client_id, store)
        step = _pending(sessions)
        sessions.step.replace(
            step.model_copy(update={"deltas": {**step.deltas, HCM2: 0.0}}),
            sessions.step.shot,  # type: ignore[arg-type]
        )
        machine.kick(HCM2, 0.1)

        refused = await _refused(_apply(sessions, client_id, _pending(sessions)))
    finally:
        await sessions.layer.close()

    assert refused.code == MACHINE_MOVED and HCM2 in refused.detail


async def test_a_drift_within_the_tolerance_still_applies(store: TrajectoryStore) -> None:
    sessions, machine = _sessions()
    try:
        client_id = (await sessions.open(OWNER, KIND)).client_id
        await _shoot(sessions, client_id, store)
        machine.kick(HCM1, 0.5 * SNAPSHOT_TOLERANCE * RECORDS[HCM1]["max_step"])

        done = await _apply(sessions, client_id, _pending(sessions))
    finally:
        await sessions.layer.close()

    assert not done.stopped and len(machine.setpoint_writes) == 3


async def test_a_shot_excluded_cms_change_then_keep_live_change_then_apply_succeeds(
    store: TrajectoryStore,
) -> None:
    sessions, machine = _sessions()
    try:
        client_id = (await sessions.open(OWNER, KIND)).client_id
        shot = await _shoot(sessions, client_id, store)
        # A corrector-exclusion change voids the step and keeps its shot ...
        excluded = SolveUpdate(solve=None, changed=frozenset({"excluded_cms"}), recalculated=False)  # type: ignore[arg-type]
        assert await sessions.after_solve(excluded, store, client_id) is None
        assert sessions.step.shot is shot
        # ... and a keep_live change recomputes a step on that same shot.
        tuned = SolveUpdate(solve=None, changed=frozenset({"keep_live"}), recalculated=False)  # type: ignore[arg-type]
        step = await sessions.after_solve(tuned, store, client_id)
        assert step is not None and step.shot_id == shot.shot_id

        done = await _apply(sessions, client_id, step)
    finally:
        await sessions.layer.close()

    assert not done.stopped and len(machine.setpoint_writes) == 3


async def test_a_step_whose_shot_carries_no_snapshot_is_refused_machine_moved(
    store: TrajectoryStore,
) -> None:
    sessions, machine = _sessions()
    try:
        client_id = (await sessions.open(OWNER, KIND)).client_id
        shot = await _shoot(sessions, client_id, store)
        step = _pending(sessions)
        sessions.step.replace(step, shot.model_copy(update={"setpoints": None}))

        refused = await _refused(_apply(sessions, client_id, step))
    finally:
        await sessions.layer.close()

    assert refused.code == MACHINE_MOVED
    assert machine.setpoint_writes == []


async def test_the_machine_moved_check_runs_before_the_limits_check(
    store: TrajectoryStore,
) -> None:
    sessions, machine = _sessions()
    try:
        client_id = (await sessions.open(OWNER, KIND)).client_id
        await _shoot(sessions, client_id, store)
        # HCM1 kicked to its max: the step would break [min, max] too.
        machine.kick(HCM1, 10.0 - START[HCM1])

        refused = await _refused(_apply(sessions, client_id, _pending(sessions)))
    finally:
        await sessions.layer.close()

    assert refused.code == MACHINE_MOVED and refused.code != LIMITS_VIOLATION


async def test_the_setup_readbacks_join_the_snapshot_batch_and_an_unchanged_setup_applies(
    store: TrajectoryStore,
) -> None:
    sessions, machine = _sessions()
    setup = _Setup()
    try:
        client_id = (await sessions.open(OWNER, KIND)).client_id
        shot = await _shoot(sessions, client_id, store)
        sessions.step.replace(_pending(sessions), shot.model_copy(update={"setup_sha": "sha-1"}))
        machine.batches.clear()

        done = await _apply(sessions, client_id, _pending(sessions), setup_check=setup)
    finally:
        await sessions.layer.close()

    assert not done.stopped
    assert machine.batches == [[*CORRECTORS, SETUP_RB]]
    assert len(setup.checks) == 1


async def test_a_changed_setup_identity_refuses_the_apply_machine_moved(
    store: TrajectoryStore,
) -> None:
    sessions, machine = _sessions()
    setup = _Setup()
    try:
        client_id = (await sessions.open(OWNER, KIND)).client_id
        shot = await _shoot(sessions, client_id, store)
        sessions.step.replace(_pending(sessions), shot.model_copy(update={"setup_sha": "sha-1"}))
        machine._post(SETUP_RB, 2, acquisition_tests.EPOCH)

        refused = await _refused(_apply(sessions, client_id, _pending(sessions), setup_check=setup))
    finally:
        await sessions.layer.close()

    assert refused.code == MACHINE_MOVED and "setup" in refused.detail
    assert machine.setpoint_writes == []


async def test_a_setup_applied_since_a_shot_without_one_refuses_the_apply(
    store: TrajectoryStore,
) -> None:
    sessions, machine = _sessions()
    try:
        client_id = (await sessions.open(OWNER, KIND)).client_id
        await _shoot(sessions, client_id, store)  # no setup held: setup_sha None

        refused = await _refused(
            _apply(sessions, client_id, _pending(sessions), setup_check=_Setup())
        )
    finally:
        await sessions.layer.close()

    assert refused.code == MACHINE_MOVED
    assert machine.setpoint_writes == []


# -- the budgets ---------------------------------------------------------------------


def test_the_snapshot_reads_are_the_last_terms_of_the_shot_budget(store: TrajectoryStore) -> None:
    assert snapshot_budget_s(1, call_timeout_s=5.0) == 5.0
    assert snapshot_budget_s(4, call_timeout_s=5.0) == 10.0

    # The default chain's injection: readout 2·5, arm 5 + 5, inject 5 + 0.5 + 5 = 30.5 s.
    # One shot: 30.5 + 5 = 35.5 s fits 36 s; two: 61 + 10 = 71 s fits 71 s, not 70 s.
    sessions, machine = _sessions(TrajectorySettings(armed_timeout_s=5.0, action_timeout_s=36.0))
    startup = sessions.layer.startup
    shot_action(startup, store, rows=ROWS, n_shots=1, call_timeout_s=5.0)
    tight = TrajectorySettings(armed_timeout_s=5.0, action_timeout_s=35.0)
    sessions, machine = _sessions(tight)
    with pytest.raises(AcquisitionRefused) as caught:
        shot_action(sessions.layer.startup, store, rows=ROWS, n_shots=1, call_timeout_s=5.0)
    assert caught.value.code == OVER_BUDGET and "setpoint reads" in caught.value.detail

    sessions, _ = _sessions(TrajectorySettings(armed_timeout_s=5.0, action_timeout_s=71.0))
    shot_action(sessions.layer.startup, store, rows=ROWS, n_shots=2, call_timeout_s=5.0)
    sessions, machine = _sessions(TrajectorySettings(armed_timeout_s=5.0, action_timeout_s=70.0))
    with pytest.raises(AcquisitionRefused):
        shot_action(sessions.layer.startup, store, rows=ROWS, n_shots=2, call_timeout_s=5.0)
    assert machine.batches == [] and machine.commands == []


def test_n_shots_max_admits_the_snapshot_reads() -> None:
    declared = chain_declaration(None)  # arm, inject, readout
    per_shot = chain_budget_s(
        declared, declared.defaults, armed_timeout_s=5.0, max_pause_s=0.5, call_timeout_s=5.0
    )
    assert per_shot == 30.5

    def most(timeout: float) -> int:
        settings = TrajectorySettings(armed_timeout_s=5.0, action_timeout_s=timeout)
        return n_shots_max(declared, settings, max_pause_s=0.5, call_timeout_s=5.0)

    assert most(3 * 30.5 + 10.0) == 3
    assert most(3 * 30.5 + 9.9) == 2
    assert most(30.5 + 5.0) == 1
    assert most(30.5 + 4.9) == 0


def test_the_shipped_preset_still_publishes_n_shots_max_of_at_least_three() -> None:
    from osprey.cli.build_injectors import _locate_pkg_services
    from osprey.facility.build import build_facility
    from osprey.facility.views.pyaml import trajectory_slots

    example = Path(__file__).resolve().parents[3] / "src/osprey/templates/facilities/example"
    document = build_facility(example, project_name="demo")
    declared = chain_declaration(trajectory_slots(document, "SR"))
    module = load_sequence(_locate_pkg_services() / "trajectory" / "sequence")
    assert module is not None

    most = n_shots_max(
        declared,
        TrajectorySettings(),
        max_pause_s=module.metadata.max_pause_s,
        call_timeout_s=DEFAULT_TIMEOUT_S,
    )

    assert most >= 3
