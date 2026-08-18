"""Bluesky-marked coverage for the shipped plans' parameter schemas and for
the `orm` and `orbit_bump_sweep` plans' abort-safe restore behavior.

There is no hand-built plan set: a single registry owns them (see
`plan_loader.get_facility_plans`), and `orm`/`grid_scan` are
plain `plans_core/` files, discovered through the layered directory loader.
Their end-to-end registration + RunEngine round trip (through the real
loader, against mock devices) is `test_exemplar_plans.py`'s job — this file
covers what that one doesn't: each plan's own `PARAMS` schema in isolation,
and (task 2.3, CC-1) the `orm` and `orbit_bump_sweep` plans'
restore-in-`finally` abort safety, which only their own generator bodies can
exercise.

The bluesky stack is a core dependency, so these run in the normal unit lane;
the `pytest.importorskip` guard only skips them in a slimmed install where
bluesky is absent. To run this file in an isolated venv:

    uv venv /tmp/bluesky-scratch
    /tmp/bluesky-scratch/bin/pip install -e .
    /tmp/bluesky-scratch/bin/python -m pytest \
        tests/services/bluesky_bridge/test_builtin_plans.py -q
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Callable
from typing import Any

import pytest

pytest.importorskip("bluesky")
pytest.importorskip("ophyd_async")

from bluesky import RunEngine  # noqa: E402
from bluesky.utils import FailedStatus, RequestAbort  # noqa: E402
from ophyd_async.core import AsyncStatus  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from osprey.services.bluesky_bridge import plan_loader  # noqa: E402
from osprey.services.bluesky_bridge.devices._connect import connect_all  # noqa: E402
from osprey.services.bluesky_bridge.devices.mock import (  # noqa: E402
    MockDetector,
    MockMotor,
    build_devices,
)
from osprey.services.bluesky_bridge.plan_metadata import parse_plan_metadata_dict  # noqa: E402
from osprey.services.bluesky_bridge.plans_core import orbit_bump_sweep  # noqa: E402
from osprey.services.bluesky_bridge.plans_core.grid_scan import (
    PARAMS as GridScanParams,  # noqa: E402
)
from osprey.services.bluesky_bridge.plans_core.orbit_bump_sweep import (  # noqa: E402
    PARAMS as BumpParams,
)
from osprey.services.bluesky_bridge.plans_core.orbit_bump_sweep import (  # noqa: E402
    build_plan as bump_plan,
)
from osprey.services.bluesky_bridge.plans_core.orm import PARAMS as ORMParams  # noqa: E402
from osprey.services.bluesky_bridge.plans_core.orm import build_plan as orm_plan  # noqa: E402

# =========================================================================
# GridScanParams (plans_core/grid_scan.py)
# =========================================================================


def test_grid_scan_params_accepts_a_well_formed_axis_set() -> None:
    params = GridScanParams(
        detectors=["det1"],
        axes=[
            {"setpoint": "m1", "start": 0.0, "stop": 1.0, "num_points": 3},
            {"setpoint": "m2", "start": 0.0, "stop": 2.0, "num_points": 5},
        ],
    )
    assert len(params.axes) == 2


def test_grid_scan_params_rejects_overlapping_setpoints_and_detectors() -> None:
    """The `model_validator(mode="after")` cross-field check: a device named
    as both a driven setpoint and a read detector is a configuration
    mistake, caught at schema-validation time rather than mid-run."""
    with pytest.raises(ValidationError, match="disjoint"):
        GridScanParams(
            detectors=["shared"],
            axes=[{"setpoint": "shared", "start": 0.0, "stop": 1.0, "num_points": 3}],
        )


def test_grid_scan_schema_validation_path_matches_reinitialize() -> None:
    """The bridge validates plan_args via `spec.schema.model_validate(...)` in
    `reinitialize()`; a malformed grid_scan payload must fail there too."""
    with pytest.raises(ValidationError):
        GridScanParams.model_validate({"detectors": ["det1"], "axes": []})


# =========================================================================
# ORMParams (task 3.3): must fail closed on every bad input — reinitialize()
# calls `spec.schema.model_validate(...)` and turns any ValidationError into
# `error_message` + `return False`, never lets it raise out of the bridge.
# =========================================================================


def test_orm_params_accepts_a_valid_set() -> None:
    params = ORMParams(
        correctors=["hcm1", "hcm2"],
        detectors=["bpm1", "bpm2", "bpm3"],
        span_a=2.0,
        num=5,
    )
    assert params.correctors == ["hcm1", "hcm2"]
    assert params.detectors == ["bpm1", "bpm2", "bpm3"]
    assert params.span_a == 2.0
    assert params.num == 5


def test_orm_params_rejects_an_empty_corrector_list() -> None:
    with pytest.raises(ValidationError):
        ORMParams(correctors=[], detectors=["bpm1"], span_a=2.0, num=5)


def test_orm_params_rejects_an_empty_detector_list() -> None:
    with pytest.raises(ValidationError):
        ORMParams(correctors=["hcm1"], detectors=[], span_a=2.0, num=5)


def test_orm_params_accepts_a_span_larger_than_any_one_facilitys_band() -> None:
    """`span_a` carries no schema-level magnitude cap.

    It is an *excursion* about each corrector's own pre-scan working point,
    expressed in whatever unit that corrector's channel speaks — so any
    literal ceiling here would be one facility's number standing in for
    every facility's. The real bound is the deployment's own
    `channel_limits.json`, enforced by the connector's reference monitor
    when the plan actually writes: an out-of-band setpoint is refused
    there, aborting the run, rather than being guessed at here.
    """
    params = ORMParams(correctors=["hcm1"], detectors=["bpm1"], span_a=120.0, num=5)
    assert params.span_a == 120.0


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_orm_params_rejects_a_non_finite_span(value: float) -> None:
    """With the upper bound gone, `gt=0` is the only magnitude constraint —
    and `inf > 0` is true, so infinity would sail through and generate a
    sweep of non-finite setpoints. Non-finite spans are rejected outright."""
    with pytest.raises(ValidationError):
        ORMParams(correctors=["hcm1"], detectors=["bpm1"], span_a=value, num=5)


def test_orm_params_rejects_a_non_positive_span() -> None:
    with pytest.raises(ValidationError):
        ORMParams(correctors=["hcm1"], detectors=["bpm1"], span_a=0.0, num=5)


def test_orm_params_rejects_too_few_points() -> None:
    with pytest.raises(ValidationError):
        ORMParams(correctors=["hcm1"], detectors=["bpm1"], span_a=2.0, num=2)


def test_orm_params_rejects_overlapping_correctors_and_detectors() -> None:
    """The `model_validator(mode="after")` cross-field check: a device named
    as both a driven corrector and a read BPM is a configuration mistake,
    caught uniformly at schema-validation time rather than mid-run."""
    with pytest.raises(ValidationError, match="disjoint"):
        ORMParams(correctors=["shared"], detectors=["shared"], span_a=2.0, num=5)


def test_orm_schema_validation_path_matches_reinitialize() -> None:
    """Mirrors `test_grid_scan_schema_validation_path_matches_reinitialize`:
    the bridge validates plan_args via `spec.schema.model_validate(...)` in
    `reinitialize()`, so a bad orm payload must fail there too — this is what
    lets `reinitialize()` return `False` + set `error_message` instead of
    raising."""
    with pytest.raises(ValidationError):
        ORMParams.model_validate({"correctors": [], "detectors": ["bpm1"], "span_a": 2.0, "num": 5})


def test_orm_params_sweep_defaults_to_bidirectional() -> None:
    """`sweep` is optional: an omitting payload (every pre-existing caller)
    keeps the symmetric two-sided sweep."""
    params = ORMParams(correctors=["hcm1"], detectors=["bpm1"], span_a=2.0, num=5)
    assert params.sweep == "bidirectional"


def test_orm_params_accepts_monodirectional_sweep() -> None:
    params = ORMParams(
        correctors=["hcm1"], detectors=["bpm1"], span_a=2.0, num=5, sweep="monodirectional"
    )
    assert params.sweep == "monodirectional"


def test_orm_params_rejects_an_unknown_sweep() -> None:
    with pytest.raises(ValidationError):
        ORMParams(correctors=["hcm1"], detectors=["bpm1"], span_a=2.0, num=5, sweep="sideways")


def _corrector_sweep_setpoints(params: ORMParams, working_point: float = 0.0) -> list[float]:
    """Run `orm` against a corrector idling at *working_point* and collect,
    in order, the current values it commands onto that corrector — the sweep
    points followed by the restore in the `finally`.

    Driven by a real `RunEngine` with a `msg_hook`, NOT by iterating the
    generator by hand. The plan reads each corrector's pre-scan working point
    with `bps.rd`, and `bps.rd` walked by hand runs in bluesky's "list-ify"
    mode: nothing answers its `read`, so it silently returns its
    `default_value` of 0 instead of the device's real value. A hand-walked
    stream would therefore report a sweep centred on zero no matter what the
    plan does — it would keep passing against an absolute sweep and pin
    nothing. The `msg_hook` gets the identical `Msg` stream with a live
    device on the other end of it.
    """
    corrector = MockMotor("hcm1", initial_value=working_point)
    devices = asyncio.run(connect_all({"hcm1": corrector, "bpm1": MockDetector("bpm1")}))

    setpoints: list[float] = []

    def _record(msg) -> None:
        if msg.command == "set" and msg.obj is corrector:
            setpoints.append(msg.args[0])

    RE = RunEngine(context_managers=[])
    RE.msg_hook = _record
    RE(orm_plan(devices, params))
    return setpoints


@pytest.mark.parametrize("working_point", [0.0, 2.5])
def test_orm_bidirectional_sweep_is_symmetric_about_the_working_point(
    working_point: float,
) -> None:
    """The default sweep kicks the corrector both ways about wherever it was
    already sitting — `working_point ± span_a` — then puts it back there.

    The `0.0` case is the virtual accelerator, whose correctors do idle at
    zero; the `2.5` case is a real ring, whose correctors hold an
    orbit-correction working point. Both are the same arithmetic, which is
    the point: the plan no longer has a privileged origin.
    """
    params = ORMParams(correctors=["hcm1"], detectors=["bpm1"], span_a=3.0, num=4)
    setpoints = _corrector_sweep_setpoints(params, working_point)
    assert setpoints == [
        working_point - 3.0,
        working_point - 1.0,
        working_point + 1.0,
        working_point + 3.0,
        working_point,
    ]


@pytest.mark.parametrize("working_point", [0.0, 2.5])
def test_orm_monodirectional_sweep_never_kicks_below_the_working_point(
    working_point: float,
) -> None:
    """A monodirectional sweep is one-sided about the working point: it spans
    `[working_point, working_point + span_a]` and never drives the corrector
    below where it started, then restores it."""
    params = ORMParams(
        correctors=["hcm1"], detectors=["bpm1"], span_a=3.0, num=4, sweep="monodirectional"
    )
    setpoints = _corrector_sweep_setpoints(params, working_point)
    assert setpoints == [
        working_point,
        working_point + 1.0,
        working_point + 2.0,
        working_point + 3.0,
        working_point,
    ]
    assert all(value >= working_point for value in setpoints)


# =========================================================================
# `orm`'s restore-in-`finally` abort safety (task 2.3, CC-1): a refused
# restore write must never replace the original in-flight exception that
# triggered the `finally` in the first place. Distinct from
# `test_exemplar_plans.py`'s happy-path round trip — this drives the FAILURE
# path, which only `plans_core/orm.py`'s own generator body can exercise.
# =========================================================================


class _FailOnValueMotor(MockMotor):
    """A `MockMotor` whose `set()` raises a chosen error for chosen values.

    Mirrors `MockMotor.set`'s body exactly for every value not in
    `fail_values`, so passing an empty mapping makes this behave identically
    to a plain `MockMotor` — only the configured values diverge, simulating
    a `write_channel_checked`-raised refusal/failure on a real corrector
    device without needing a real connector.
    """

    def __init__(
        self,
        name: str,
        fail_values: dict[float, Exception],
        initial_value: float = 0.0,
    ) -> None:
        super().__init__(name=name, initial_value=initial_value)
        self._fail_values = fail_values

    @AsyncStatus.wrap
    async def set(self, value: float) -> None:
        if value in self._fail_values:
            raise self._fail_values[value]
        await self.setpoint.set(value)
        self._set_readback(value)


def test_orm_plan_restore_refusal_does_not_mask_the_original_sweep_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """CC-1: if BOTH a mid-sweep move and the cleanup restore raise, the
    exception that surfaces from the RunEngine must be the ORIGINAL sweep
    failure, not the restore's — and the restore failure must be logged, not
    silently dropped, so the operator still learns the corrector was left
    away from its working point.

    The RunEngine wraps a device `set()` error in its own
    `bluesky.utils.FailedStatus` (which carries the underlying exception's
    message), so the check below matches on that wrapper's message rather
    than the bare `RuntimeError` type.
    """
    # The corrector idles at 2.5, so span_a=3.0, num=4 sweeps
    # [-0.5, 1.5, 3.5, 5.5] and restores to 2.5. Failing the FIRST swept
    # point and the restore target covers both writes, and 2.5 is
    # deliberately NOT among the swept currents, so the two failures below
    # are unambiguous distinct write attempts. A nonzero idle value is what
    # makes the restore target observable at all: at zero it would coincide
    # with the value the plan used to hard-code.
    hcm1 = _FailOnValueMotor(
        "hcm1",
        fail_values={
            -0.5: RuntimeError("ORIGINAL sweep failure"),
            2.5: RuntimeError("RESTORE refused"),
        },
        initial_value=2.5,
    )
    devices = asyncio.run(connect_all({"hcm1": hcm1, "bpm1": MockDetector("bpm1")}))

    params = ORMParams(correctors=["hcm1"], detectors=["bpm1"], span_a=3.0, num=4)
    plan = orm_plan(devices, params)

    RE = RunEngine(context_managers=[])
    with caplog.at_level(logging.WARNING, logger="osprey.services.bluesky_bridge.plans_core.orm"):
        with pytest.raises(FailedStatus, match="ORIGINAL sweep failure") as excinfo:
            RE(plan)

    # The restore's own error never replaces the original in the exception
    # that reaches the caller.
    assert "RESTORE refused" not in str(excinfo.value)

    # ...but it WAS caught and logged, not swallowed silently, so the
    # operator still learns the corrector was left off-zero.
    assert "failed to restore corrector" in caplog.text
    assert "hcm1" in caplog.text
    assert "RESTORE refused" in caplog.text


def test_orm_plan_restores_every_corrector_to_zero_when_no_refusal_occurs() -> None:
    """The ordinary (non-error) path on a machine whose correctors idle at
    zero — the virtual accelerator: with no write refused, every corrector
    ends its own sweep back at 0 A."""
    devices = asyncio.run(build_devices(motor_names=["hcm1", "hcm2"], detector_names=["bpm1"]))
    params = ORMParams(correctors=["hcm1", "hcm2"], detectors=["bpm1"], span_a=2.0, num=3)
    plan = orm_plan(devices, params)

    RE = RunEngine(context_managers=[])
    RE(plan)

    for name in ("hcm1", "hcm2"):
        readback = asyncio.run(devices[name].readback.get_value())
        assert readback == 0.0


def test_orm_plan_refuses_to_sweep_a_corrector_reading_back_non_finite() -> None:
    """A corrector whose readback is NaN — dead, disconnected, or unscaled —
    has no working point to sweep about, and every setpoint derived from it
    would be NaN too.

    That must fail before the first write, not at it: a NaN demand passes a
    limits check (`nan < low` and `nan > high` are both false, so the
    reference monitor has nothing to refuse), reaches the IOC, and only then
    surfaces as a readback-settle timeout — having already written. The plan
    refuses up front, and no `set` is ever issued.
    """
    hcm1 = MockMotor("hcm1", initial_value=float("nan"))
    devices = asyncio.run(connect_all({"hcm1": hcm1, "bpm1": MockDetector("bpm1")}))
    params = ORMParams(correctors=["hcm1"], detectors=["bpm1"], span_a=2.0, num=3)

    writes: list[float] = []

    def _record(msg) -> None:
        if msg.command == "set" and msg.obj is hcm1:
            writes.append(msg.args[0])

    RE = RunEngine(context_managers=[])
    RE.msg_hook = _record
    with pytest.raises(ValueError, match="non-finite"):
        RE(orm_plan(devices, params))

    assert writes == []


def test_orm_plan_restores_every_corrector_to_its_own_pre_scan_working_point() -> None:
    """The same path on a real ring: correctors holding an orbit-correction
    working point are each put back where THEY were, not where the plan
    assumed they were.

    Two different nonzero working points, because a single shared one could
    be passed by a plan that recorded one corrector's value and restored all
    of them to it. Parking these at 0 A — what the plan used to do — is what
    would destroy a stored-beam orbit.
    """
    working_points = {"hcm1": 2.5, "hcm2": -1.25}
    devices = asyncio.run(
        connect_all(
            {
                **{
                    name: MockMotor(name, initial_value=value)
                    for name, value in working_points.items()
                },
                "bpm1": MockDetector("bpm1"),
            }
        )
    )
    params = ORMParams(correctors=["hcm1", "hcm2"], detectors=["bpm1"], span_a=2.0, num=3)

    RE = RunEngine(context_managers=[])
    RE(orm_plan(devices, params))

    for name, value in working_points.items():
        assert asyncio.run(devices[name].readback.get_value()) == value


# =========================================================================
# BumpParams (plans_core/orbit_bump_sweep.py): the bump is parameterized in
# ORBIT space — "this much displacement at these BPMs, none at these" — so
# the schema's job is to refuse a bump that is not a bump: too few (or too
# many) correctors to close one, fewer orbit constraints than corrector
# kicks, a BPM asked to move and hold still at once, or half a beam-current
# guard. What it deliberately does NOT refuse is a *small* bump: whether the
# machine is being asked for anything real is the runtime demand gate's call,
# since it alone sees the reference orbit, the response, and the noise floor.
# =========================================================================


def _bump_payload(**overrides: object) -> dict[str, object]:
    """A minimal valid `orbit_bump_sweep` payload, with *overrides* applied.

    Three correctors against three orbit constraints (one target, two closure
    BPMs) — the smallest bump the schema admits, so any single relaxation in a
    test is the thing under test rather than slack elsewhere.
    """
    payload: dict[str, object] = {
        "correctors": ["hcm1", "hcm2", "hcm3"],
        "targets": [{"bpm": "bpm1", "value": 0.3}],
        "closure_bpms": ["bpm2", "bpm3"],
        "bpms": ["bpm4"],
        "num": 5,
        "probe_amplitude": 0.05,
        "tolerance": 0.001,
        "max_trim_iterations": 3,
        "settle_s": 0.2,
    }
    payload.update(overrides)
    return payload


def test_bump_params_accepts_a_well_formed_set_and_defaults_the_rest() -> None:
    params = BumpParams(**_bump_payload())

    assert params.correctors == ["hcm1", "hcm2", "hcm3"]
    assert [target.bpm for target in params.targets] == ["bpm1"]
    assert params.targets[0].value == 0.3
    assert params.closure_bpms == ["bpm2", "bpm3"]
    assert params.bpms == ["bpm4"]
    # The optional half of the schema: a bump asked for with none of these
    # named still runs, one-sided, off the orbit the machine is already on.
    assert params.mode == "relative"
    assert params.sweep == "monodirectional"
    assert params.baseline_reads == 5
    assert params.best_effort is False
    assert params.detectors == []
    assert params.beam_current_device is None
    assert params.min_beam_current is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("correctors", ["hcm1", "hcm1", "hcm2"]),
        ("closure_bpms", ["bpm2", "bpm2", "bpm3"]),
        ("bpms", ["bpm4", "bpm4"]),
        ("detectors", ["dcct", "dcct"]),
    ],
)
def test_bump_params_rejects_a_device_named_twice_in_one_list(field: str, value: list[str]) -> None:
    """A repeat is never what the operator meant — and a repeated corrector
    duplicates a column of the solve, which is singular in exactly the way the
    constraint-count check exists to prevent."""
    with pytest.raises(ValidationError, match="unique"):
        BumpParams(**_bump_payload(**{field: value}))


def test_bump_params_rejects_two_demands_on_one_target_bpm() -> None:
    """Two displacements asked for at the same BPM are two answers to one
    question; there is no rule that picks between them."""
    with pytest.raises(ValidationError, match="at most once"):
        BumpParams(
            **_bump_payload(targets=[{"bpm": "bpm1", "value": 0.3}, {"bpm": "bpm1", "value": 0.5}])
        )


def test_bump_params_rejects_a_bpm_that_is_both_bumped_and_held() -> None:
    """A closure BPM is one the bump must leave alone; a target BPM is one it
    must move. Naming a BPM as both asks the solve for two contradictory rows."""
    with pytest.raises(ValidationError, match="disjoint"):
        BumpParams(
            **_bump_payload(
                targets=[{"bpm": "bpm2", "value": 0.3}],
                closure_bpms=["bpm2", "bpm3"],
            )
        )


@pytest.mark.parametrize("count", [3, 4])
def test_bump_params_accepts_three_or_four_correctors(count: int) -> None:
    """Three kicks is the smallest set that can close a local bump; four is the
    other standard arrangement, and buys a degree of freedom."""
    params = BumpParams(
        **_bump_payload(
            correctors=[f"hcm{i}" for i in range(1, count + 1)],
            closure_bpms=["bpm2", "bpm3", "bpm5"],
        )
    )
    assert len(params.correctors) == count


@pytest.mark.parametrize("count", [2, 5])
def test_bump_params_rejects_a_corrector_count_that_cannot_be_a_bump(count: int) -> None:
    """Two correctors cannot cancel their own kicks downstream — what leaks out
    is an oscillation around the whole ring, not a local bump. Five correctors
    is a different diagnostic, not this one."""
    with pytest.raises(ValidationError):
        BumpParams(
            **_bump_payload(
                correctors=[f"hcm{i}" for i in range(1, count + 1)],
                closure_bpms=["bpm2", "bpm3", "bpm5", "bpm6", "bpm7"],
            )
        )


def test_bump_params_rejects_an_empty_target_list() -> None:
    """With nothing asked for anywhere, there is no bump to solve for — the
    plan would be an elaborate way to read the orbit."""
    with pytest.raises(ValidationError):
        BumpParams(**_bump_payload(targets=[]))


def test_bump_params_accepts_targets_whose_values_are_all_zero() -> None:
    """A zero-offset bump is legal, and deliberately so.

    Whether the machine is actually being asked for anything is decided at
    runtime by the demand gate, which alone sees the reference orbit, the
    measured response, and the BPM noise floor — a schema-level "is this big
    enough" test has none of those. It would also reject the all-zero dry run
    against mock devices, which is precisely the run that proves the plan safe
    before it is ever pointed at a ring.
    """
    params = BumpParams(
        **_bump_payload(targets=[{"bpm": "bpm1", "value": 0.0}, {"bpm": "bpm5", "value": 0.0}])
    )
    assert [target.value for target in params.targets] == [0.0, 0.0]


def test_bump_params_accepts_a_demand_larger_than_any_one_facilitys_band() -> None:
    """Target values and `probe_amplitude` carry no schema-level magnitude cap.

    They are expressed in whatever units the BPM and corrector channels speak,
    so a literal ceiling here would be one facility's number standing in for
    every facility's. The real bound is the deployment's own
    `channel_limits.json`, enforced by the connector's reference monitor when
    the plan writes: an out-of-band setpoint is refused there, aborting the
    run, rather than being guessed at here.
    """
    params = BumpParams(
        **_bump_payload(targets=[{"bpm": "bpm1", "value": 5000.0}], probe_amplitude=400.0)
    )
    assert params.targets[0].value == 5000.0
    assert params.probe_amplitude == 400.0


def test_bump_params_rejects_fewer_orbit_constraints_than_correctors() -> None:
    """Each target and closure BPM is a row of the solve; each corrector is an
    unknown. Fewer rows than unknowns leaves a whole direction in kick space,
    every point of which fits the request equally well — so the operator picks,
    not the numerics."""
    with pytest.raises(ValidationError, match="underdetermined"):
        BumpParams(**_bump_payload(closure_bpms=["bpm2"]))


@pytest.mark.parametrize(
    "guard",
    [
        {"beam_current_device": "dcct"},
        {"min_beam_current": 50.0},
    ],
)
def test_bump_params_rejects_half_a_beam_current_guard(guard: dict[str, object]) -> None:
    """A device with no threshold has nothing to compare against and a
    threshold with no device has nothing to read — either alone is a guard that
    reads as armed while being inert."""
    with pytest.raises(ValidationError, match="together or not at all"):
        BumpParams(**_bump_payload(**guard))


def test_bump_params_accepts_a_complete_beam_current_guard() -> None:
    params = BumpParams(**_bump_payload(beam_current_device="dcct", min_beam_current=50.0))
    assert params.beam_current_device == "dcct"
    assert params.min_beam_current == 50.0


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
@pytest.mark.parametrize("field", ["probe_amplitude", "tolerance", "settle_s", "min_beam_current"])
def test_bump_params_rejects_a_non_finite_float(field: str, value: float) -> None:
    """A non-finite demand cannot be refused by a limits check (`nan < low` and
    `nan > high` are both false), so it would reach the machine and fail only
    as a settle timeout, having already written. The magnitude bounds do not
    catch it either: `inf > 0` is true."""
    overrides: dict[str, object] = {field: value}
    if field == "min_beam_current":
        overrides["beam_current_device"] = "dcct"
    with pytest.raises(ValidationError):
        BumpParams(**_bump_payload(**overrides))


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_bump_params_rejects_a_non_finite_target_value(value: float) -> None:
    with pytest.raises(ValidationError):
        BumpParams(**_bump_payload(targets=[{"bpm": "bpm1", "value": value}]))


def test_bump_params_rejects_an_absolute_bidirectional_sweep() -> None:
    """A bidirectional sweep reflects the requested bump about the working
    orbit. An absolute target names an orbit position instead, so there is no
    working orbit to reflect about and the negated amplitude names no orbit at
    all."""
    with pytest.raises(ValidationError, match="absolute"):
        BumpParams(**_bump_payload(mode="absolute", sweep="bidirectional"))


@pytest.mark.parametrize(
    ("mode", "sweep"),
    [
        ("absolute", "monodirectional"),
        ("relative", "bidirectional"),
        ("relative", "monodirectional"),
    ],
)
def test_bump_params_accepts_every_other_mode_and_sweep_pairing(mode: str, sweep: str) -> None:
    params = BumpParams(**_bump_payload(mode=mode, sweep=sweep))
    assert (params.mode, params.sweep) == (mode, sweep)


@pytest.mark.parametrize("field", ["mode", "sweep"])
def test_bump_params_rejects_an_unknown_mode_or_sweep(field: str) -> None:
    with pytest.raises(ValidationError):
        BumpParams(**_bump_payload(**{field: "sideways"}))


def test_bump_params_rejects_a_single_amplitude_point() -> None:
    """One point is not a sweep: with nothing to compare it against, neither
    the achieved-versus-requested slope nor the closure residual has a trend."""
    with pytest.raises(ValidationError):
        BumpParams(**_bump_payload(num=1))


def test_bump_params_rejects_too_few_baseline_reads() -> None:
    """The baseline is both the reference orbit and the BPM noise estimate the
    demand gate compares against; two samples cannot carry a spread."""
    with pytest.raises(ValidationError):
        BumpParams(**_bump_payload(baseline_reads=2))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("probe_amplitude", 0.0),
        ("tolerance", 0.0),
        ("settle_s", -0.1),
        ("max_trim_iterations", 0),
    ],
)
def test_bump_params_rejects_a_non_positive_bound(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        BumpParams(**_bump_payload(**{field: value}))


def test_bump_params_schema_validation_path_matches_reinitialize() -> None:
    """Mirrors the orm/grid_scan cases: the bridge validates plan_args via
    `spec.schema.model_validate(...)` in `reinitialize()`, so a bad bump
    payload must fail there too — that is what lets `reinitialize()` return
    `False` and set `error_message` instead of raising out of the bridge."""
    with pytest.raises(ValidationError):
        BumpParams.model_validate(_bump_payload(correctors=["hcm1"]))


def test_bump_plan_metadata_parses_alongside_its_params_schema() -> None:
    """`PLAN_METADATA` goes through the same fail-closed parser every plan does.

    `required_devices` names the PARAMS fields carrying device names, matched
    by the queue's add-time pre-check on the trailing-`s`-stripped form — so
    `bpms` covers both the `bpms` field and the `bpm` inside each target, and
    `beam_current_devices` covers `beam_current_device`.
    """
    metadata = parse_plan_metadata_dict(
        orbit_bump_sweep.PLAN_METADATA, source="plans_core/orbit_bump_sweep.py"
    )
    assert metadata.name == "orbit_bump_sweep"
    assert metadata.category == "accelerator"
    assert metadata.writes is True
    assert metadata.required_devices == [
        "correctors",
        "bpms",
        "closure_bpms",
        "detectors",
        "beam_current_devices",
    ]

    stripped = {name.removesuffix("s") for name in metadata.required_devices}
    for field in ("correctors", "closure_bpms", "bpms", "detectors", "beam_current_device"):
        assert field in BumpParams.model_fields
        assert field.removesuffix("s") in stripped
    assert "bpm" in stripped  # reaches TargetPoint.bpm nested inside `targets`


def test_bump_plan_loads_as_shipped_with_its_params_schema() -> None:
    """The plans_core loader discovers every file in this package immediately,
    so a half-written plan is not a dormant file — it is a quarantined plan in
    the live catalog. This is what checks the file landed complete."""
    plan_loader.reset_facility_plans()
    try:
        facility = plan_loader.get_facility_plans()
        assert "orbit_bump_sweep" in facility.plans
        spec = facility.plans["orbit_bump_sweep"]
        assert spec.provenance == "shipped"
        assert spec.metadata is not None
        assert spec.metadata.writes is True
        # Not `is BumpParams`: the loader execs the file by path under a
        # synthetic module name, so the registered schema is a distinct class
        # object built from the same source. Its field set is what the bridge
        # actually validates payloads against.
        assert spec.schema is not None
        assert set(spec.schema.model_fields) == set(BumpParams.model_fields)
    finally:
        plan_loader.reset_facility_plans()


# =========================================================================
# `orbit_bump_sweep`'s abort safety (task 3.3). The schema cases above decide
# what the plan will accept; these decide what its generator body does when a
# run goes wrong — every one of them on a machine holding a nonzero
# orbit-correction working point, because "restored" means "back where each
# corrector was", never "back to zero".
#
# Same shape as the `orm` abort tests above: drive the real generator against
# mock devices, break one thing, and check both what surfaces to the caller
# and where the correctors were left. The one difference is that a bump needs
# a machine with beam physics in it to solve anything, and mock devices have
# none — a probe moves no BPM, so the fitted response is identically zero.
# That is not a limitation here: with zero-valued targets the demand gate
# (`demand_is_negligible`) declares the run trivially converged, no solve is
# ever attempted, and the plan still walks its whole profile — probes,
# verification reads, one data row per step, and the restore. The failure
# paths below all live in that walk, so they are exercised in full.
# =========================================================================


#: The correctors' pre-scan working points: three distinct nonzero values, so
#: a plan that restored all of them to one recorded value (or to zero) fails
#: every restore assertion below.
_BUMP_WORKING_POINTS = {"hcm1": 2.5, "hcm2": -1.25, "hcm3": 0.75}

#: Every BPM's constant readback. Any finite value works — the plan measures
#: the reference orbit rather than assuming one — so a nonzero one is used to
#: keep "the orbit" and "zero" distinguishable in the arithmetic.
_BUMP_BPM_VALUE = 0.4

_BUMP_LOGGER = "osprey.services.bluesky_bridge.plans_core.orbit_bump_sweep"


def _bump_devices(**replacements: Any) -> dict[str, Any]:
    """Connect a bump-sized device set, with *replacements* swapped in by name.

    BPMs are `MockMotor`s rather than `MockDetector`s: a detector counts up on
    every trigger, which a baseline reads as noise (σ = 1.0 over three reads),
    and the plan correctly refuses a tolerance narrower than that before it
    writes anything — see the fail-fast test, which is the one place that
    counting behavior is the point. A motor's soft readback is the steady
    number a quiet BPM reports.
    """
    devices: dict[str, Any] = {
        name: MockMotor(name, initial_value=value) for name, value in _BUMP_WORKING_POINTS.items()
    }
    for name in ("bpm1", "bpm2", "bpm3", "bpm4"):
        devices[name] = MockMotor(name, initial_value=_BUMP_BPM_VALUE)
    devices.update(replacements)
    return asyncio.run(connect_all(devices))


def _bump_run_params(**overrides: object) -> BumpParams:
    """A short `orbit_bump_sweep` run over `_bump_devices`'s device set.

    Zero-valued targets, so the demand gate takes the trivially-converged path
    described in this section's header; `settle_s=0.0` so the sweep does not
    spend real time sleeping; `num=2` for a four-step monodirectional profile.
    """
    payload = _bump_payload(
        targets=[{"bpm": "bpm1", "value": 0.0}],
        num=2,
        baseline_reads=3,
        tolerance=0.01,
        settle_s=0.0,
    )
    payload.update(overrides)
    return BumpParams(**payload)


def _bump_restore_writes() -> list[tuple[str, float]]:
    """The write stream the `finally` alone produces: each corrector, in plan
    order, commanded back to its own working point."""
    return list(_BUMP_WORKING_POINTS.items())


def _record_writes(commands: list[tuple[str, float]]) -> Callable[[Any], None]:
    """A `msg_hook` collecting every `(device, value)` the plan commands.

    A `msg_hook` on a live `RunEngine`, not a hand-walked generator: `bps.rd`
    walked by hand answers from its `default_value` of 0 instead of the
    device, so a hand-walked stream would report working points of zero no
    matter what the correctors hold — see `_corrector_sweep_setpoints` above.
    """

    def _record(msg: Any) -> None:
        if msg.command == "set":
            commands.append((msg.obj.name, msg.args[0]))

    return _record


class _RefusingMotor(MockMotor):
    """A `MockMotor` that refuses every write while `refusing` is set.

    The flag is flipped by the test's own `msg_hook`, so a refusal is pinned to
    a *moment* in the run — this probe write, the cleanup restore after the
    last data row — rather than to a float value the plan happened to compute
    for that step. That keeps a test saying what it means ("fail the restore")
    instead of encoding the arithmetic that produced the value.
    """

    def __init__(self, name: str, initial_value: float = 0.0, message: str = "write refused"):
        super().__init__(name=name, initial_value=initial_value)
        self.refusing = False
        self._message = message

    @AsyncStatus.wrap
    async def set(self, value: float) -> None:
        if self.refusing:
            raise RuntimeError(self._message)
        await self.setpoint.set(value)
        self._set_readback(value)


class _StickyReadbackMotor(MockMotor):
    """A corrector whose readback never follows its setpoint.

    This is the shape a *clamped* write has under the real device layer: the
    demand is accepted at the setpoint channel, the readback never reaches it,
    and `ConnectorSettable.set` — which polls the readback and raises
    `TimeoutError` when the settle deadline passes — is what turns that into an
    abort. Reproduced here rather than mocked at the connector, so the plan
    sees exactly what it sees on a real machine: a `set()` that raises.
    """

    @AsyncStatus.wrap
    async def set(self, value: float) -> None:
        await self.setpoint.set(value)
        readback = await self.readback.get_value()
        if readback != value:
            raise TimeoutError(
                f"Readback {self.name!r} did not settle to {value} (last read: {readback})"
            )


class _NegativeZeroMotor(MockMotor):
    """A corrector whose readback reports negative zero.

    Signed zero is the one float where `value + 0.0` is not `value`, which is
    what makes "commanded verbatim" distinguishable from "commanded through the
    arithmetic" at all. It has to be reported from `read()` — the call the
    plan's `bps.rd` makes — because `MockMotor`'s soft signal normalizes `-0.0`
    to `+0.0` on the way in.
    """

    async def read(self) -> dict[str, Any]:
        reading = await super().read()
        return {key: {**entry, "value": -0.0} for key, entry in reading.items()}


class _ArmableBpm(MockMotor):
    """A BPM whose readback jumps to `armed_value` once the test arms it.

    Arming from a `msg_hook` pins the change to a moment in the run — "the
    first read after the baseline is complete", "the terminal step" — rather
    than to a count of how many reads the plan spent getting there, which is an
    implementation detail no test should own. `armed_value` is NaN for the
    unreadable-BPM case and a finite off-reference number for the
    machine-did-not-come-back case.
    """

    def __init__(
        self,
        name: str,
        initial_value: float = 0.0,
        armed_value: float = float("nan"),
    ) -> None:
        super().__init__(name=name, initial_value=initial_value)
        self.armed = False
        self._armed_value = armed_value

    async def read(self) -> dict[str, Any]:
        reading = await super().read()
        if not self.armed:
            return reading
        return {key: {**entry, "value": self._armed_value} for key, entry in reading.items()}


def test_bump_plan_restores_every_corrector_after_a_clean_sweep() -> None:
    """The ordinary path: every corrector ends the run back at its own pre-scan
    working point, and the run's rows are the pinned layout.

    Three different nonzero working points, because a single shared one could
    be passed by a plan that recorded one corrector's value and restored all of
    them to it — and parking a ring's correctors at 0 A is what would drop the
    orbit correction the machine was holding.
    """
    devices = _bump_devices()
    params = _bump_run_params()
    rows: list[Any] = []

    RE = RunEngine(context_managers=[])
    RE(bump_plan(devices, params), lambda name, doc: rows.append(name) if name == "event" else None)

    for name, working_point in _BUMP_WORKING_POINTS.items():
        assert asyncio.run(devices[name].readback.get_value()) == working_point
    # `baseline_reads` baseline rows, then one row per monodirectional step.
    assert len(rows) == params.baseline_reads + 2 * params.num


def test_bump_plan_restores_every_corrector_after_a_mid_run_failure() -> None:
    """A refused write mid-probe aborts the run, and the `finally` brings the
    corrector back from where the abort left it.

    The refusal is the *second* probe write — the `-probe` — so at the moment
    the run fails, `hcm1` is genuinely parked at `working_point + probe`, not
    sitting where it started. Restoring it is a real move, which is what makes
    the assertion below mean something.
    """
    hcm1 = _RefusingMotor(
        "hcm1", initial_value=_BUMP_WORKING_POINTS["hcm1"], message="PROBE write refused"
    )
    devices = _bump_devices(hcm1=hcm1)

    commands: list[tuple[str, float]] = []
    record = _record_writes(commands)

    def _hook(msg: Any) -> None:
        record(msg)
        if msg.command == "set" and msg.obj is hcm1:
            hcm1.refusing = len([1 for name, _ in commands if name == "hcm1"]) == 2

    RE = RunEngine(context_managers=[])
    RE.msg_hook = _hook
    with pytest.raises(FailedStatus, match="PROBE write refused"):
        RE(bump_plan(devices, _bump_run_params()))

    # The +probe landed, the -probe was refused, and the cleanup put it back.
    assert commands[0] == ("hcm1", _BUMP_WORKING_POINTS["hcm1"] + 0.05)
    for name, working_point in _BUMP_WORKING_POINTS.items():
        assert asyncio.run(devices[name].readback.get_value()) == working_point
    assert commands[-3:] == _bump_restore_writes()


def test_bump_plan_restores_every_corrector_when_the_run_is_aborted() -> None:
    """An abort mid-run still restores every corrector.

    Driven by throwing `bluesky.utils.RequestAbort` into the generator, which
    is what a RunEngine abort actually delivers to a running plan. Not
    `GeneratorExit`: CPython forbids a generator from yielding while a
    `GeneratorExit` is being handled ("generator ignored GeneratorExit"), so a
    `close()`-shaped abort could never carry the restore messages out to the
    RunEngine in the first place — restore-on-abort is a claim about the abort
    path plans are actually aborted through, and this is it.

    Hand-driven, so the reads are answered here rather than by a RunEngine:
    the run is aborted at the baseline's first `create`, which is the first
    message inside the `try` — the earliest point at which the plan owes a
    restore at all.
    """
    devices = _bump_devices()
    plan = bump_plan(devices, _bump_run_params())

    def _reading(device: Any) -> dict[str, Any]:
        value = _BUMP_WORKING_POINTS.get(device.name, _BUMP_BPM_VALUE)
        return {field: {"value": value, "timestamp": 0.0} for field in device.hints["fields"]}

    reply: Any = None
    while True:
        msg = plan.send(reply)
        reply = _reading(msg.obj) if msg.command == "read" else None
        if msg.command == "create":
            break

    cleanup: list[Any] = []
    with pytest.raises(RequestAbort):
        msg = plan.throw(RequestAbort())
        while True:
            cleanup.append(msg)
            msg = plan.send(None)

    assert [
        (msg.obj.name, msg.args[0]) for msg in cleanup if msg.command == "set"
    ] == _bump_restore_writes()


def test_bump_plan_aborts_and_restores_when_a_bpm_reads_back_non_finite() -> None:
    """A BPM that goes non-finite mid-run stops the run rather than trimming
    toward a NaN, and the correctors come back.

    A NaN poisons every mean, fit, and band check downstream while satisfying
    none of their comparisons: `nan < low` and `nan > high` are both false, so
    nothing downstream refuses it and the run would grind through its whole
    profile "not converged" for a reason nobody could read off the data.

    The BPM is armed the moment the baseline is complete, so the first read to
    come back NaN is one of `hcm1`'s probe reads — taken while `hcm1` sits at
    `working_point + probe`, which is what makes the restore assertion a real
    move rather than a no-op.
    """
    bpm1 = _ArmableBpm("bpm1", initial_value=_BUMP_BPM_VALUE, armed_value=float("nan"))
    devices = _bump_devices(bpm1=bpm1)
    params = _bump_run_params()

    commands: list[tuple[str, float]] = []
    record = _record_writes(commands)
    saves: list[int] = []

    def _hook(msg: Any) -> None:
        record(msg)
        if msg.command == "save":
            saves.append(1)
            bpm1.armed = len(saves) >= params.baseline_reads

    RE = RunEngine(context_managers=[])
    RE.msg_hook = _hook
    with pytest.raises(ValueError, match="non-finite") as excinfo:
        RE(bump_plan(devices, params))

    assert "+probe" in str(excinfo.value)
    assert commands[0] == ("hcm1", _BUMP_WORKING_POINTS["hcm1"] + 0.05)
    assert commands[-3:] == _bump_restore_writes()
    for name, working_point in _BUMP_WORKING_POINTS.items():
        assert asyncio.run(devices[name].readback.get_value()) == working_point


def test_bump_plan_beam_current_guard_stops_the_sweep_before_it_writes() -> None:
    """A beam-current reading below `min_beam_current` stops the run before the
    first probe write, and the correctors are left where they were.

    The guard runs before every write batch, and the first batch is `hcm1`'s
    probe — so on a machine that has already lost its beam, the only writes the
    whole run issues are the `finally`'s restores, each commanding a corrector
    to exactly where it already is.
    """
    dcct = MockMotor("dcct", initial_value=12.0)
    devices = _bump_devices(dcct=dcct)
    params = _bump_run_params(beam_current_device="dcct", min_beam_current=50.0)

    commands: list[tuple[str, float]] = []
    RE = RunEngine(context_managers=[])
    RE.msg_hook = _record_writes(commands)
    with pytest.raises(RuntimeError, match="below min_beam_current"):
        RE(bump_plan(devices, params))

    assert commands == _bump_restore_writes()
    for name, working_point in _BUMP_WORKING_POINTS.items():
        assert asyncio.run(devices[name].readback.get_value()) == working_point


def test_bump_plan_fail_fast_refuses_a_tolerance_below_the_baseline_noise() -> None:
    """A tolerance narrower than twice the measured BPM noise fails the run at
    the baseline — before the first probe write.

    The BPMs here are `MockDetector`s, whose counter advances on every trigger,
    so three baseline reads of 1, 2, 3 carry σ = 1.0 — a noisier machine than
    any tolerance of 0.001 could be verified on. That band is one a perfectly
    converged orbit falls outside of by chance alone, so trimming into it would
    burn every iteration chasing noise and then report a failure the machine
    never had.

    The only writes the run issues are the `finally`'s restores, which command
    each corrector to exactly where it already is: no probe, no step, nothing
    that moves the machine.
    """
    devices = _bump_devices(
        **{name: MockDetector(name) for name in ("bpm1", "bpm2", "bpm3", "bpm4")}
    )
    params = _bump_run_params(tolerance=0.001)

    commands: list[tuple[str, float]] = []
    RE = RunEngine(context_managers=[])
    RE.msg_hook = _record_writes(commands)
    with pytest.raises(ValueError, match="narrower than"):
        RE(bump_plan(devices, params))

    assert commands == _bump_restore_writes()


def test_bump_plan_fail_fast_refuses_a_corrector_reading_back_non_finite() -> None:
    """A corrector with no readable working point is refused before the `try`
    is ever entered, so not one write is attempted — not even a restore.

    Every setpoint the run would derive from a NaN working point is NaN too,
    and a NaN demand is not something a limits check can refuse (`nan < low`
    and `nan > high` are both false): it would reach the IOC and surface only
    as a readback-settle timeout, having already written.
    """
    devices = _bump_devices(hcm2=MockMotor("hcm2", initial_value=float("nan")))

    commands: list[tuple[str, float]] = []
    RE = RunEngine(context_managers=[])
    RE.msg_hook = _record_writes(commands)
    with pytest.raises(ValueError, match="non-finite working point"):
        RE(bump_plan(devices, _bump_run_params()))

    assert commands == []


def test_bump_plan_clamped_corrector_write_aborts_and_restores_the_rest() -> None:
    """A corrector whose readback never reaches the demand aborts the run, and
    every corrector — the clamped one included — is left at its working point.

    This is what a clamped write looks like from inside the plan: the setpoint
    is accepted, the readback stays put, and the device layer's settle wait
    times out, which the RunEngine surfaces as a `FailedStatus`. The plan must
    not read a clamped probe as a measurement — its whole response fit divides
    by the *commanded* amplitude, so a probe that only half-arrived would be
    fitted as a real slope and the bump solved through it would be wrong.
    """
    hcm2 = _StickyReadbackMotor("hcm2", initial_value=_BUMP_WORKING_POINTS["hcm2"])
    devices = _bump_devices(hcm2=hcm2)

    commands: list[tuple[str, float]] = []
    RE = RunEngine(context_managers=[])
    RE.msg_hook = _record_writes(commands)
    with pytest.raises(FailedStatus, match="did not settle"):
        RE(bump_plan(devices, _bump_run_params()))

    assert commands[-3:] == _bump_restore_writes()
    for name, working_point in _BUMP_WORKING_POINTS.items():
        assert asyncio.run(devices[name].readback.get_value()) == working_point


def test_bump_plan_restore_failure_on_a_clean_sweep_is_raised_not_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A sweep that finishes cleanly but cannot put a corrector back must not
    read as a success.

    The mirror image of the `orm` case above, and the reason the plan tracks
    whether an error is already in flight: mid-run, a failed restore is logged
    so it cannot replace the exception that actually explains the run; on a
    clean run there is no such exception, and a corrector left off its working
    point is itself the finding.

    The refusal is armed once the last data row is saved, so the only write it
    can catch is the cleanup restore.
    """
    hcm2 = _RefusingMotor(
        "hcm2", initial_value=_BUMP_WORKING_POINTS["hcm2"], message="RESTORE refused"
    )
    devices = _bump_devices(hcm2=hcm2)
    params = _bump_run_params()
    rows = params.baseline_reads + 2 * params.num

    saves: list[int] = []

    def _hook(msg: Any) -> None:
        if msg.command == "save":
            saves.append(1)
            hcm2.refusing = len(saves) >= rows

    RE = RunEngine(context_managers=[])
    RE.msg_hook = _hook
    with caplog.at_level(logging.WARNING, logger=_BUMP_LOGGER):
        with pytest.raises(RuntimeError, match="could not be restored") as excinfo:
            RE(bump_plan(devices, params))

    assert "hcm2" in str(excinfo.value)
    assert "failed to restore corrector" in caplog.text
    assert "RESTORE refused" in caplog.text
    # The other two were still put back: one corrector's refusal never stops
    # the rest of the `finally` from running.
    for name in ("hcm1", "hcm3"):
        assert asyncio.run(devices[name].readback.get_value()) == _BUMP_WORKING_POINTS[name]


def test_bump_plan_terminal_step_restores_the_working_points_bit_verbatim() -> None:
    """The terminal scale-0 step commands the recorded working points
    themselves, not `working_point + 0.0`.

    Signed zero is what makes the difference observable: `-0.0 + 0.0` is `+0.0`,
    the one case where the arithmetic path does not return the value it was
    handed. A corrector reading back `-0.0` is an ordinary thing — any scaling
    of a small negative raw count produces one — and the plan's promise is that
    the run ends by commanding exactly what it read, so the machine is left
    bit-identical to how it was found rather than merely numerically equal.

    The sign sequence below is the whole story: the probe restore, the terminal
    step, and the cleanup restore carry the working point through verbatim,
    while the three ordinary profile steps go through `working_point + offset`
    and come back positive.
    """
    devices = _bump_devices(hcm1=_NegativeZeroMotor("hcm1", initial_value=0.0))

    commands: list[tuple[str, float]] = []
    RE = RunEngine(context_managers=[])
    RE.msg_hook = _record_writes(commands)
    RE(bump_plan(devices, _bump_run_params()))

    zeros = [value for name, value in commands if name == "hcm1" and value == 0.0]
    assert [math.copysign(1.0, value) for value in zeros] == [
        -1.0,  # the probe loop's restore: `bps.mv(corrector, working_point)`
        1.0,  # profile step 1/4, via `working_point + offset`
        1.0,  # profile step 2/4
        1.0,  # profile step 3/4
        -1.0,  # the terminal step: verbatim, no arithmetic
        -1.0,  # the `finally`'s restore
    ]
    # The assertion above is on what was *commanded*, not on the readback: the
    # mock's soft signal normalizes the sign on write, so only the message
    # stream can carry the distinction here.
    assert asyncio.run(devices["hcm1"].readback.get_value()) == 0.0


def test_bump_plan_does_not_trim_the_terminal_restore_step() -> None:
    """An out-of-band orbit at the terminal step is raised, never trimmed away.

    Trimming there would mean correcting the machine's failure to come back by
    moving the correctors somewhere other than where the run found them — which
    is precisely the outcome the terminal step exists to detect. So the step is
    commanded once and, if the orbit is not back, the run says so.

    The BPM is knocked off the reference orbit only after the last ordinary
    data row, so the terminal step is the first one that sees it: exactly two
    write batches follow, the terminal step's own and the `finally`'s restore,
    where a trim pass would have added a third.
    """
    bpm1 = _ArmableBpm("bpm1", initial_value=_BUMP_BPM_VALUE, armed_value=_BUMP_BPM_VALUE + 5.0)
    devices = _bump_devices(bpm1=bpm1)
    params = _bump_run_params()
    ordinary_rows = params.baseline_reads + 2 * params.num - 1

    commands: list[tuple[str, float]] = []
    record = _record_writes(commands)
    armed_at: list[int] = []

    def _hook(msg: Any) -> None:
        record(msg)
        if msg.command == "save" and not bpm1.armed:
            armed_at.append(len(commands))
            bpm1.armed = len(armed_at) >= ordinary_rows

    RE = RunEngine(context_managers=[])
    RE.msg_hook = _hook
    with pytest.raises(RuntimeError, match="did not come back to its reference orbit"):
        RE(bump_plan(devices, params))

    after_arming = commands[armed_at[-1] :]
    assert len(after_arming) == 2 * len(_BUMP_WORKING_POINTS)
    assert after_arming[-3:] == _bump_restore_writes()
    for name, working_point in _BUMP_WORKING_POINTS.items():
        assert asyncio.run(devices[name].readback.get_value()) == working_point
