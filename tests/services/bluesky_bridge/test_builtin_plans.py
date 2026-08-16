"""Bluesky-marked coverage for the shipped `orm`/`grid_scan` plans'
parameter schemas and the `orm` plan's abort-safe restore behavior.

There is no hand-built plan set: a single registry owns them (see
`plan_loader.get_facility_plans`), and `orm`/`grid_scan` are
plain `plans_core/` files, discovered through the layered directory loader.
Their end-to-end registration + RunEngine round trip (through the real
loader, against mock devices) is `test_exemplar_plans.py`'s job — this file
covers what that one doesn't: each plan's own `PARAMS` schema in isolation,
and (task 2.3, CC-1) the `orm` plan's restore-in-`finally` abort safety,
which only its own generator body can exercise.

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

import pytest

pytest.importorskip("bluesky")
pytest.importorskip("ophyd_async")

from bluesky import RunEngine  # noqa: E402
from bluesky.utils import FailedStatus  # noqa: E402
from ophyd_async.core import AsyncStatus  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from osprey.services.bluesky_bridge.devices._connect import connect_all  # noqa: E402
from osprey.services.bluesky_bridge.devices.mock import (  # noqa: E402
    MockDetector,
    MockMotor,
    build_devices,
)
from osprey.services.bluesky_bridge.plans_core.grid_scan import (
    PARAMS as GridScanParams,  # noqa: E402
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
