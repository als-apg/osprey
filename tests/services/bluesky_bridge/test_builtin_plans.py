"""Bluesky-marked coverage for the shipped `orm`/`grid_scan` plans'
parameter schemas and the `orm` plan's abort-safe restore behavior.

`plans.py`'s hand-built plan set is gone (the single-registry migration —
see `plan_loader.get_facility_plans`); `orm`/`grid_scan` are now
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


def test_orm_params_rejects_a_span_beyond_the_channel_limits_band() -> None:
    """`channel_limits.json` bounds a corrector `:SP` to +-12 A; ORMParams'
    own `span_a` bound (+-10 A, the response model's typical linear-kick
    range — see `lattice/response.py`) is tighter and rejected first."""
    with pytest.raises(ValidationError):
        ORMParams(correctors=["hcm1"], detectors=["bpm1"], span_a=12.0, num=5)


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


def _corrector_sweep_setpoints(params: ORMParams) -> list[float]:
    """Drive the `orm` generator directly (no RunEngine) and collect, in
    order, the current values it commands onto its single corrector — the
    sweep points followed by the restore-to-0 in the `finally`. Iterating a
    plan just walks its `Msg` stream; none of `orm`'s control flow branches on
    a yielded value, so this needs no RunEngine."""
    devices = asyncio.run(build_devices(motor_names=["hcm1"], detector_names=["bpm1"]))
    corrector = devices["hcm1"]
    setpoints: list[float] = []
    for msg in orm_plan(devices, params):
        if msg.command == "set" and msg.obj is corrector:
            setpoints.append(msg.args[0])
    return setpoints


def test_orm_bidirectional_sweep_spans_symmetric_range() -> None:
    """The default sweep drives the corrector across the symmetric
    `[-span_a, +span_a]` (both signs present), then restores it to 0."""
    params = ORMParams(correctors=["hcm1"], detectors=["bpm1"], span_a=3.0, num=4)
    setpoints = _corrector_sweep_setpoints(params)
    assert setpoints == [-3.0, -1.0, 1.0, 3.0, 0.0]


def test_orm_monodirectional_sweep_spans_zero_to_span() -> None:
    """A monodirectional sweep never drives the corrector negative: it spans
    `[0, +span_a]` only, then restores to 0."""
    params = ORMParams(
        correctors=["hcm1"], detectors=["bpm1"], span_a=3.0, num=4, sweep="monodirectional"
    )
    setpoints = _corrector_sweep_setpoints(params)
    assert setpoints == [0.0, 1.0, 2.0, 3.0, 0.0]
    assert all(value >= 0.0 for value in setpoints)


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

    def __init__(self, name: str, fail_values: dict[float, Exception]) -> None:
        super().__init__(name=name)
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
    """CC-1: if BOTH a mid-sweep move and the cleanup restore-to-0 raise, the
    exception that surfaces from the RunEngine must be the ORIGINAL sweep
    failure, not the restore's — and the restore failure must be logged, not
    silently dropped, so the operator still learns the corrector was left
    off-zero.

    The RunEngine wraps a device `set()` error in its own
    `bluesky.utils.FailedStatus` (which carries the underlying exception's
    message), so the check below matches on that wrapper's message rather
    than the bare `RuntimeError` type.
    """
    # span_a=3.0, num=4 -> currents = [-3.0, -1.0, 1.0, 3.0]; 0.0 (the
    # restore target) is deliberately NOT among the swept currents, so the
    # two failures below are unambiguous distinct write attempts.
    hcm1 = _FailOnValueMotor(
        "hcm1",
        fail_values={
            -3.0: RuntimeError("ORIGINAL sweep failure"),
            0.0: RuntimeError("RESTORE refused"),
        },
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
    """The ordinary (non-error) path: with no write refused, every corrector
    still ends its own sweep restored to 0 A."""
    devices = asyncio.run(build_devices(motor_names=["hcm1", "hcm2"], detector_names=["bpm1"]))
    params = ORMParams(correctors=["hcm1", "hcm2"], detectors=["bpm1"], span_a=2.0, num=3)
    plan = orm_plan(devices, params)

    RE = RunEngine(context_managers=[])
    RE(plan)

    for name in ("hcm1", "hcm2"):
        readback = asyncio.run(devices[name].readback.get_value())
        assert readback == 0.0
