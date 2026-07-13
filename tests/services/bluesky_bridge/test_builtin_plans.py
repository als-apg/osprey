"""Bluesky-marked coverage for the v1 built-in plan set (`plans.py`).

`plans.py` imports `bluesky`, so it lives behind the optional
`osprey-framework[bluesky-bridge]` extra — every test here is skipped via
`pytest.importorskip` when bluesky is absent, keeping the import-clean fast
lane green. To run these:

    uv venv /tmp/bluesky-scratch
    /tmp/bluesky-scratch/bin/pip install -e '.[bluesky-bridge]'
    /tmp/bluesky-scratch/bin/python -m pytest \
        tests/services/bluesky_bridge/test_builtin_plans.py -q

Mostly focuses on the parameter schemas' contract (which are pure pydantic
and need no RunEngine) — broader RunEngine-driven execution of these plans is
covered by `test_runengine_integration.py`/`test_orm_plan_integration.py`.
The task 2.3 (CC-1) tests below are the exception: they drive the `orm`
plan's abort-safe restore-in-`finally` behavior through a real RunEngine
directly, since that behavior lives entirely in `plans.py`'s own generator.
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
from osprey.services.bluesky_bridge.plans import (  # noqa: E402
    BUILTIN_PLANS,
    GridScanParams,
    ORMParams,
    list_plan_specs,
)


def test_builtin_registry_exposes_the_v1_plan_set() -> None:
    assert set(BUILTIN_PLANS) == {"count", "scan", "grid_scan", "orm"}
    for name, spec in BUILTIN_PLANS.items():
        assert spec.name == name
        assert callable(spec.plan)


def test_list_plan_specs_serializes_each_plan_with_its_schema() -> None:
    served = {p["name"]: p for p in list_plan_specs()}
    assert set(served) == {"count", "scan", "grid_scan", "orm"}
    for entry in served.values():
        assert entry["description"]  # every built-in ships a description
        assert entry["schema"]["type"] == "object"  # a real JSON schema


def test_grid_scan_params_accepts_matching_num_points_and_axes() -> None:
    params = GridScanParams(
        detectors=["det1"],
        axes=[
            {"motor": "m1", "start": 0.0, "stop": 1.0},
            {"motor": "m2", "start": 0.0, "stop": 2.0},
        ],
        num_points=[3, 5],
    )
    assert len(params.num_points) == len(params.axes) == 2


def test_grid_scan_params_rejects_mismatched_num_points_and_axes() -> None:
    """The `model_validator` makes the length mismatch a schema-level
    validation error (raised on construction / `model_validate`), not a bare
    ValueError deferred to plan-build time."""
    with pytest.raises(ValidationError, match="num_points must have one entry per axis"):
        GridScanParams(
            detectors=["det1"],
            axes=[{"motor": "m1", "start": 0.0, "stop": 1.0}],
            num_points=[3, 5],  # 2 points, 1 axis
        )


def test_grid_scan_schema_validation_path_matches_reinitialize() -> None:
    """The bridge validates plan_args via `spec.schema.model_validate(...)` in
    `reinitialize()`; a mismatched grid_scan payload must fail there too."""
    spec = BUILTIN_PLANS["grid_scan"]
    with pytest.raises(ValidationError):
        spec.schema.model_validate(
            {
                "detectors": ["det1"],
                "axes": [{"motor": "m1", "start": 0.0, "stop": 1.0}],
                "num_points": [3, 5],
            }
        )


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
    spec = BUILTIN_PLANS["orm"]
    with pytest.raises(ValidationError):
        spec.schema.model_validate(
            {"correctors": [], "detectors": ["bpm1"], "span_a": 2.0, "num": 5}
        )


# =========================================================================
# `_orm_plan` restore-in-`finally` abort safety (task 2.3, CC-1): a refused
# restore write must never replace the original in-flight exception that
# triggered the `finally` in the first place.
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
    plan = BUILTIN_PLANS["orm"].plan(devices, params)

    RE = RunEngine(context_managers=[])
    with caplog.at_level(logging.WARNING, logger="osprey.services.bluesky_bridge.plans"):
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
    plan = BUILTIN_PLANS["orm"].plan(devices, params)

    RE = RunEngine(context_managers=[])
    RE(plan)

    for name in ("hcm1", "hcm2"):
        readback = asyncio.run(devices[name].readback.get_value())
        assert readback == 0.0
