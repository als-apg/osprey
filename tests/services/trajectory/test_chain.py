"""The chain as data: the declared steps, a run's validation, its budget and its turn.

The declaration is parsed from a ``trajectory_slots`` entry shaped as the facts
view writes it. Chain turns run through the real :class:`ActionLayer` on the
acquisition tests' stand-in machine, extended with a cleaning command and its
done readback, a turns-tracked readback and a stored-current readback.
"""

from __future__ import annotations

from typing import Any

import pytest
from tests.services.trajectory import test_acquisition as acquisition_tests
from tests.services.trajectory.test_acquisition import (
    ARRIVAL,
    CALL_TIMEOUT_S,
    EPOCH,
    REQUEST,
    ROWS,
    SAMPLES,
    SHOT_ARM,
    SHOT_ARMED,
    TRIGGER_SAMPLE,
    _shot_layer,
)

from osprey.services.trajectory.acquisition import (
    DEFAULT_DECLARATION,
    INVALID_CHAIN,
    OVER_BUDGET,
    UNKNOWN_STEP,
    AcquisitionRefused,
    ChainFailed,
    ChainRun,
    chain_action,
    chain_budget_s,
    chain_declaration,
    chain_run,
    default_run,
    makes_step,
    n_shots_max,
)
from osprey.services.trajectory.actions import ActionLayer
from osprey.services.trajectory.startup import TrajectorySettings
from osprey.services.trajectory.state import ChainBody, ChainResult, ShotFailure, ShotResult
from osprey.services.trajectory.store import TrajectoryStore

# The acquisition tests' deployment, validator and store fixtures, shared here.
_deployment = acquisition_tests._deployment
optional_validator = acquisition_tests.optional_validator
store = acquisition_tests.store

CLEAN = "SR:DIAG:CLEANING:REQUEST"
CLEAN_DONE = "SR:DIAG:CLEANING:DONE"
TURNS_SP = "SR:DIAG:TBT:TURNS:SP"
TURNS_RB = "SR:DIAG:TBT:TURNS:RB"
RF_ON = "SR:RF:ON:RB"
CURRENT = "SR:DCCT:CURRENT"
TURNS = 4


def _step(step_id: str, *, default: bool = True, command: Any = None, **extra: Any) -> dict:
    return {
        "id": step_id,
        "label": step_id.title(),
        "description": f"The {step_id} step.",
        "command": command,
        "default": default,
        "position": 99,  # the parser places every step by its index
        "timeout_s": extra.pop("timeout_s", None),
        **extra,
    }


def _slots(*, clean_done: bool = False, clean_default: bool = True) -> dict[str, Any]:
    """A ``trajectory_slots`` entry as the facts view writes the example's."""
    clean = _step(
        "clean",
        default=clean_default,
        command={"address": CLEAN, "value": 1},
        timeout_s=1.0,
        **({"done": {"address": CLEAN_DONE, "value": 1}} if clean_done else {}),
    )
    return {
        "devices": {},
        "trigger_sample": TRIGGER_SAMPLE,
        "injection_request": REQUEST,
        "injection_busy": "SR:INJ:BUSY",
        "machine": {"rf_on": RF_ON, "stored_current": CURRENT},
        "settings": [
            {"id": "attenuation", "scope": "device", "setting": {}, "readback": {}},
            {
                "id": "turns_tracked",
                "scope": "global",
                "setting": TURNS_SP,
                "readback": TURNS_RB,
                "tolerance": None,
            },
        ],
        "steps": [_step("arm"), _step("inject"), clean, _step("readout")],
    }


# -- the declaration -----------------------------------------------------------------


def test_the_declaration_carries_the_steps_in_order_with_their_defaults_and_readbacks() -> None:
    declared = chain_declaration(_slots(clean_default=False))

    assert declared.ids == ("arm", "inject", "clean", "readout")
    assert [step.position for step in declared.steps] == [0, 1, 2, 3]
    assert declared.defaults == ("arm", "inject", "readout")
    assert declared.commands == frozenset({CLEAN})
    clean = declared.steps[2]
    assert clean.command is not None and (clean.command.address, clean.timeout_s) == (CLEAN, 1.0)
    # The turns-tracked readback is the setting's pair; the header reads RF and current.
    assert declared.turns_tracked == TURNS_RB
    assert declared.machine == {"rf_on": RF_ON, "stored_current": CURRENT}


def test_the_turns_tracked_setting_without_a_readback_is_read_at_its_setpoint() -> None:
    slots = _slots()
    slots["settings"][1]["readback"] = None

    assert chain_declaration(slots).turns_tracked == TURNS_SP


@pytest.mark.parametrize("slots", [None, {}, {"steps": []}])
def test_a_model_declaring_no_steps_runs_arm_inject_readout(slots: Any) -> None:
    declared = chain_declaration(slots)

    assert declared.steps == DEFAULT_DECLARATION.steps
    assert declared.defaults == ("arm", "inject", "readout")
    assert declared.machine == {} and declared.turns_tracked is None


@pytest.mark.parametrize(
    ("steps", "words"),
    [
        ([_step("arm"), _step("inject"), _step("wash"), _step("readout")], "declares a command"),
        ([_step("arm"), _step("inject"), _step("inject"), _step("readout")], "more than once"),
        ([_step("arm"), _step("inject")], "no readout"),
        (
            [_step("arm", command={"address": "X", "value": 1}), _step("inject"), _step("readout")],
            "takes no command",
        ),
        ([_step("arm"), _step("inject"), {"id": "readout"}], "malformed"),
    ],
)
def test_a_malformed_chain_is_refused_naming_the_problem(steps: list, words: str) -> None:
    with pytest.raises(ValueError, match=words):
        chain_declaration({"steps": steps})


# -- a run ---------------------------------------------------------------------------


def test_a_run_takes_the_ticked_steps_in_chain_order() -> None:
    declared = chain_declaration(_slots())

    run = chain_run(declared, ChainBody(steps=["readout", "arm", "inject", "arm"], n_shots=2))

    assert run == ChainRun(steps=("arm", "inject", "readout"), n_shots=2)
    assert run.reads and run.fresh


def test_a_readout_without_arm_and_inject_is_not_fresh() -> None:
    declared = chain_declaration(_slots())

    assert not chain_run(declared, ChainBody(steps=["inject", "readout"])).fresh
    assert not chain_run(declared, ChainBody(steps=["readout"])).fresh


def test_a_step_the_chain_does_not_declare_is_refused_unknown_step() -> None:
    with pytest.raises(AcquisitionRefused) as caught:
        chain_run(chain_declaration(_slots()), ChainBody(steps=["arm", "wash"]))

    assert (caught.value.code, caught.value.status) == (UNKNOWN_STEP, 400)
    assert "'wash'" in caught.value.detail


@pytest.mark.parametrize(
    "body",
    [
        ChainBody(steps=["arm", "inject"], readout="pseudo_orbit"),
        ChainBody(steps=["arm", "inject", "readout"], n_turns=5),
        ChainBody(steps=["readout"], n_shots=3),
        ChainBody(steps=["arm"], n_shots=2),
    ],
)
def test_an_invalid_combination_is_refused_invalid_chain(body: ChainBody) -> None:
    with pytest.raises(AcquisitionRefused) as caught:
        chain_run(chain_declaration(_slots()), body)

    assert (caught.value.code, caught.value.status) == (INVALID_CHAIN, 400)


def test_the_agent_doors_run_is_the_declared_default_chain() -> None:
    declared = chain_declaration(_slots())

    run = default_run(declared, n_shots=3, average="nanmean")

    assert run == ChainRun(
        steps=("arm", "inject", "clean", "readout"), n_shots=3, average="nanmean"
    )


def test_a_default_chain_that_reads_nothing_is_refused_for_the_agent_door() -> None:
    slots = _slots()
    slots["steps"][-1]["default"] = False

    with pytest.raises(AcquisitionRefused) as caught:
        default_run(chain_declaration(slots), n_shots=1, average="mean")

    assert caught.value.code == INVALID_CHAIN


# -- the budget ----------------------------------------------------------------------


def test_the_budget_sums_every_ticked_steps_declared_bound() -> None:
    declared = chain_declaration(_slots())
    bounds = {"armed_timeout_s": 10.0, "max_pause_s": 0.5, "call_timeout_s": 5.0}

    # readout 2·5; arm 5 + 10; inject 5 + 0.5 + 5 (current) + 10 (clear); clean 5 + 1.
    assert chain_budget_s(declared, declared.ids, **bounds) == 51.5
    # A lone arm: its write and its wait.
    assert chain_budget_s(declared, ["arm"], **bounds) == 15.0
    # A lone readout: the pre-inject batch and the read.
    assert chain_budget_s(declared, ["readout"], **bounds) == 10.0


def test_n_shots_max_is_the_most_default_chain_injections_the_timeout_admits() -> None:
    declared = chain_declaration(_slots())
    settings = TrajectorySettings(armed_timeout_s=10.0, action_timeout_s=300.0)

    assert n_shots_max(declared, settings, max_pause_s=0.5, call_timeout_s=5.0) == 5  # 300 // 51.5
    bare = n_shots_max(DEFAULT_DECLARATION, settings, max_pause_s=0.5, call_timeout_s=5.0)
    assert bare == 7  # 300 // 40.5


@pytest.mark.usefixtures("optional_validator")
def test_a_chain_over_the_action_budget_is_refused_before_the_turn(
    store: TrajectoryStore,
) -> None:
    layer, machine = _shot_layer(TrajectorySettings(armed_timeout_s=5.0, action_timeout_s=60.0))
    declared = chain_declaration(_slots())
    run = chain_run(declared, ChainBody(steps=list(declared.ids), n_shots=2))

    with pytest.raises(AcquisitionRefused) as caught:
        chain_action(
            layer.startup, store, rows=ROWS, run=run, declaration=declared, call_timeout_s=5.0
        )

    assert caught.value.code == OVER_BUDGET
    assert machine.reads == [] and machine.commands == []


# -- a chain turn --------------------------------------------------------------------


def _machine_layer(**kwargs: Any) -> tuple[ActionLayer, Any]:
    """The acquisition stand-in, with cleaning, turns-tracked and stored-current readbacks."""
    layer, machine = _shot_layer(**kwargs)
    machine._post(TURNS_RB, TURNS, EPOCH)
    machine._post(CURRENT, 1.5, EPOCH)
    machine._post(CLEAN_DONE, 0, EPOCH)
    machine.batches = []
    machine.done_on_clean = True
    plain_read = machine.read_multiple_channels
    plain_write = machine.write_multiple_channels

    async def read_multiple_channels(addresses: list[str], timeout: float | None = None) -> Any:
        machine.batches.append(list(addresses))
        return await plain_read(addresses, timeout)

    async def write_multiple_channels(operations: list[tuple[str, Any]], **kw: Any) -> Any:
        for address, _value in operations:
            if address == REQUEST:
                # The injection adds stored charge; the chain read the current before it.
                machine._post(CURRENT, machine.values[CURRENT] + 1.0, EPOCH)
            if address == CLEAN and machine.done_on_clean:
                machine._post(CLEAN_DONE, 1, EPOCH)
        return await plain_write(operations, **kw)

    machine.read_multiple_channels = read_multiple_channels
    machine.write_multiple_channels = write_multiple_channels
    return layer, machine


async def _run(
    layer: ActionLayer,
    store: TrajectoryStore,
    body: ChainBody,
    *,
    slots: dict[str, Any] | None = None,
    **kwargs: Any,
) -> ChainResult:
    declared = chain_declaration(slots if slots is not None else _slots())
    kwargs.setdefault("call_timeout_s", CALL_TIMEOUT_S)
    kwargs.setdefault("poll_s", 0.01)
    action = chain_action(
        layer.startup,
        store,
        rows=ROWS,
        run=chain_run(declared, body),
        declaration=declared,
        **kwargs,
    )
    try:
        return await layer.submit("alice", "panel", action)
    finally:
        await layer.close()


@pytest.mark.usefixtures("optional_validator")
async def test_the_full_chain_arms_injects_cleans_and_reads_a_fresh_shot(
    store: TrajectoryStore,
) -> None:
    layer, machine = _machine_layer()
    seen: list[ShotResult] = []

    result = await _run(
        layer,
        store,
        ChainBody(steps=["arm", "inject", "clean", "readout"]),
        post_shot=seen.append,
    )

    assert result.steps_run == ["arm", "inject", "clean", "readout"]
    assert [ops for ops, _ in machine.commands] == [
        [(SHOT_ARM[0], 1), (SHOT_ARM[1], 1)],
        [(REQUEST, 1)],
        [(CLEAN, 1)],
    ]
    assert result.commands_fired == [*SHOT_ARM, REQUEST, CLEAN]
    shot = result.shot
    assert shot is not None and shot.fresh and shot.readout == "tbt"
    assert shot.shots_included == 1 and shot.i0 == ARRIVAL
    # The stored current read just before inject, not after it.
    assert shot.beam_current == 1.5
    # The raw buffer: the trigger sample, then the turns tracked.
    assert shot.raw_samples == TRIGGER_SAMPLE + TURNS
    # The pre-inject batch read every record and the turns-tracked readback in one call.
    assert machine.batches[0] == [*ROWS.records, TURNS_RB]
    assert seen == [shot]
    assert (store.root / "shots" / "SR" / f"{shot.shot_id}.json").is_file()
    assert makes_step(shot)


@pytest.mark.usefixtures("optional_validator")
async def test_the_ticked_chain_runs_once_per_injection(store: TrajectoryStore) -> None:
    layer, machine = _machine_layer()

    result = await _run(
        layer, store, ChainBody(steps=["arm", "inject", "clean", "readout"], n_shots=2)
    )

    assert machine.shots == 2
    assert result.commands_fired == [*SHOT_ARM, REQUEST, CLEAN, *SHOT_ARM, REQUEST, CLEAN]
    assert result.steps_run == ["arm", "inject", "clean", "readout"]
    assert result.shot is not None and result.shot.shots_included == 2
    # Each injection read the current before it fired: 1.5, then 2.5.
    assert result.shot.beam_current == 2.0


@pytest.mark.usefixtures("optional_validator")
async def test_a_readout_alone_reads_what_the_buffers_hold_and_is_not_fresh(
    store: TrajectoryStore,
) -> None:
    layer, machine = _machine_layer()
    # The buffers hold an old shot, stamped before anything this action read.
    for index, _bpm in enumerate(ROWS.bpms):
        for address, level in ((ROWS.x[index], 1.0), (ROWS.y[index], -1.0), (ROWS.sum[index], 2.0)):
            machine._post(address, [0.0] * ARRIVAL + [level] * (SAMPLES - ARRIVAL), EPOCH)
    seen: list[ShotResult] = []

    result = await _run(layer, store, ChainBody(steps=["readout"]), post_shot=seen.append)

    assert machine.commands == [] and result.commands_fired == []
    shot = result.shot
    assert shot is not None and not shot.fresh
    # Not stale-checked: an unrefreshed buffer is the point of a lone readout.
    assert shot.shots_included == 1 and shot.shots_failed == []
    assert not makes_step(shot) and seen == []
    assert (store.root / "shots" / "SR" / f"{shot.shot_id}.json").is_file()


@pytest.mark.usefixtures("optional_validator")
async def test_a_chain_without_readout_stores_no_shot_and_a_lone_arm_leaves_the_bpms_armed(
    store: TrajectoryStore,
) -> None:
    layer, machine = _machine_layer()
    seen: list[ShotResult] = []

    result = await _run(layer, store, ChainBody(steps=["arm"]), post_shot=seen.append)

    assert result == ChainResult(steps_run=["arm"], commands_fired=list(SHOT_ARM), shot=None)
    assert all(machine.values[flag] == 1 for flag in SHOT_ARMED)
    assert seen == []
    assert not (store.root / "shots").exists()


@pytest.mark.usefixtures("optional_validator")
async def test_a_pseudo_orbit_readout_averages_n_turns_and_makes_no_step(
    store: TrajectoryStore,
) -> None:
    layer, machine = _machine_layer()
    # Each turn from the arrival on reads the turn count above the first-turn level.
    original = machine._inject

    def inject_ramp() -> None:
        original()
        for index in range(len(ROWS.bpms)):
            ramp = [0.0] * ARRIVAL + [float(k) for k in range(SAMPLES - ARRIVAL)]
            machine._post(ROWS.x[index], ramp, machine.stamps[ROWS.x[index]])

    machine._inject = inject_ramp
    seen: list[ShotResult] = []

    result = await _run(
        layer,
        store,
        ChainBody(steps=["arm", "inject", "readout"], readout="pseudo_orbit", n_turns=3),
        post_shot=seen.append,
    )

    shot = result.shot
    assert shot is not None and shot.readout == "pseudo_orbit" and shot.n_turns == 3
    assert shot.x == dict.fromkeys(ROWS.x, 1.0)  # mean of turns 0, 1, 2
    assert shot.fresh and not makes_step(shot) and seen == []


@pytest.mark.usefixtures("optional_validator")
async def test_a_pseudo_orbit_over_the_turns_tracked_fails_the_injection(
    store: TrajectoryStore,
) -> None:
    layer, _machine = _machine_layer()

    result = await _run(
        layer,
        store,
        ChainBody(steps=["arm", "inject", "readout"], readout="pseudo_orbit", n_turns=TURNS + 1),
    )

    assert result.shot is not None and result.shot.shots_included == 0
    assert result.shot.shots_failed == [
        ShotFailure(index=0, reason=f"n_turns {TURNS + 1} is over the {TURNS} turns tracked")
    ]


@pytest.mark.usefixtures("optional_validator")
async def test_a_command_step_waits_for_its_done_readback(store: TrajectoryStore) -> None:
    layer, machine = _machine_layer()
    slots = _slots(clean_done=True)

    result = await _run(
        layer, store, ChainBody(steps=["arm", "inject", "clean", "readout"]), slots=slots
    )

    assert result.shot is not None and result.shot.shots_included == 1
    assert CLEAN_DONE in machine.reads


@pytest.mark.usefixtures("optional_validator")
async def test_a_command_step_whose_done_never_comes_fails_the_injection(
    store: TrajectoryStore,
) -> None:
    layer, machine = _machine_layer()
    machine.done_on_clean = False

    result = await _run(
        layer,
        store,
        ChainBody(steps=["arm", "inject", "clean", "readout"]),
        slots=_slots(clean_done=True),
    )

    assert result.shot is not None
    assert result.shot.shots_failed == [
        ShotFailure(index=0, reason="clean did not finish within 1 s", stragglers=[CLEAN_DONE])
    ]


@pytest.mark.usefixtures("optional_validator")
async def test_a_failing_step_of_a_chain_without_readout_ends_it_naming_what_fired(
    store: TrajectoryStore,
) -> None:
    layer, machine = _machine_layer()
    machine.done_on_clean = False

    with pytest.raises(ChainFailed) as caught:
        await _run(
            layer, store, ChainBody(steps=["arm", "inject", "clean"]), slots=_slots(clean_done=True)
        )

    assert caught.value.reason == "clean did not finish within 1 s"
    assert caught.value.steps_run == ["arm", "inject"]
    assert caught.value.commands_fired == [*SHOT_ARM, REQUEST, CLEAN]
    assert not (store.root / "shots").exists()


@pytest.mark.usefixtures("optional_validator")
async def test_a_chain_that_does_not_inject_needs_no_sequence_module(
    store: TrajectoryStore,
) -> None:
    layer, machine = _machine_layer(module=None)

    result = await _run(layer, store, ChainBody(steps=["arm"]))

    assert result.steps_run == ["arm"]
    assert [ops for ops, _ in machine.commands] == [[(SHOT_ARM[0], 1), (SHOT_ARM[1], 1)]]
