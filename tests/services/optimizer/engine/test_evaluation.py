"""Tests for optimization/evaluation.py — the Evaluator.

Canonical home for Evaluator behavior, consolidating what was previously split
across test_evaluation_sync.py, test_error_handling.py, the A8 failure-mode
tests, test_optimization.py::TestEvaluationFunction, and test_coercion.py.

Covers:
- the reading_to_float coercion seam (bool-not-numeric, NaN/disconnected ->
  None) that the evaluator reads through,
- setpoint/measurement synchronization (the Evaluator returns measurements
  matching the CURRENT setpoint; the mock env simulates one-step hardware lag),
- grid snapping (backend bookkeeping keys are filtered by the run loop, not
  here — see test_runner),
- tolerance of unreadable/non-numeric objective readings (retry then NaN, no
  abort — control-system errors abort immediately),
- FakeInjectorEnvironment-driven smoke and input-validation errors.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from unittest.mock import MagicMock

import pytest

from osprey.services.optimizer.models.exceptions import (
    ControlSystemUnreachableError,
    VariableOutOfBoundsError,
)
from osprey.services.optimizer.models.protocols import EnvironmentProtocol, PVReading
from osprey.services.optimizer.models.testing import bounds_variables, valid_variables
from osprey.services.optimizer.models.types import VariableInputs
from osprey.services.optimizer.optimization.coercion import reading_to_float
from osprey.services.optimizer.optimization.evaluation import Evaluator, VariableGrid

from ._fakes import (  # relative import — app_tests is not under the importable package
    EchoEnv,
    FakeInjectorEnvironment,
)

OBJECTIVE = "charge_ltb"


class TestReadingToFloat:
    """optimization.coercion.reading_to_float — the evaluator's read->float seam."""

    @pytest.mark.parametrize(
        ("reading", "expected"),
        [
            pytest.param(PVReading(value=3), 3.0, id="int"),
            pytest.param(PVReading(value=2.5), 2.5, id="float"),
            pytest.param(PVReading(value="open"), None, id="string"),
            pytest.param(PVReading(value=[1.0, 2.0]), None, id="list"),
            pytest.param(PVReading(value=None), None, id="none"),
            pytest.param(PVReading(value=float("nan")), None, id="nan"),
            # bool is an int subclass; a digital channel is not an optimization scalar.
            pytest.param(PVReading(value=True), None, id="bool"),
            pytest.param(PVReading(value=3.0, connected=False), None, id="disconnected"),
        ],
    )
    def test_numeric_values_become_float_others_none(
        self, reading: PVReading, expected: float | None
    ) -> None:
        assert reading_to_float(reading) == expected


class _StaleMeasurementEnvironment:
    """Mock environment that models hardware lag.

    Before the shot boundary the measurement is stale (reflects the previous
    setpoint).  The shot-boundary wait inside ``set_variables`` advances the
    internal state so subsequent ``get_observables`` returns fresh data.
    """

    def __init__(self) -> None:
        self._active_setpoint: VariableInputs | None = None

    def set_variables(
        self,
        inputs: VariableInputs,
        *,
        check_cancellation: Callable[[], None] | None = None,
    ) -> None:
        if check_cancellation is not None:
            check_cancellation()
        # Simulate the shot-boundary wait: the machine cycles and the active
        # setpoint catches up to the requested one before observables are read.
        self._active_setpoint = inputs

    def get_observables(
        self,
        observable_names: list[str],
        *,
        check_cancellation: Callable[[], None] | None = None,
    ) -> dict[str, PVReading]:
        if check_cancellation is not None:
            check_cancellation()
        assert self._active_setpoint is not None, "set_variables must be called first"
        value = sum(self._active_setpoint.values())
        return {name: PVReading(value=value) for name in observable_names}

    def evaluate(
        self,
        variable_inputs: VariableInputs,
        observable_names: list[str],
        *,
        check_cancellation: Callable[[], None] | None = None,
    ) -> dict[str, PVReading]:
        self.set_variables(variable_inputs, check_cancellation=check_cancellation)
        return self.get_observables(
            observable_names,
            check_cancellation=check_cancellation,
        )


def _expected_measurement(inputs: VariableInputs) -> float:
    """Expected measurement if the hardware correctly responded to *these* inputs."""
    return sum(inputs.values())


class TestStaleMeasurementRace:
    """Assert that each evaluation returns a measurement matching its OWN setpoint."""

    @pytest.fixture
    def evaluator(self) -> Evaluator:
        env = _StaleMeasurementEnvironment()
        return Evaluator(env, OBJECTIVE)  # type: ignore[arg-type]

    def test_repeated_evaluations_match_setpoint(self, evaluator: Evaluator) -> None:
        """Repeated evaluations with distinct setpoints should each return their own
        measurement — the value for inputs_B reflects inputs_B, not the prior inputs_A."""
        distinct_inputs = [{"var1": float(i), "var2": float(i * 100)} for i in range(1, 6)]

        for i, inputs in enumerate(distinct_inputs):
            result = evaluator(inputs)
            expected = _expected_measurement(inputs)
            assert result[OBJECTIVE] == pytest.approx(expected), (
                f"Evaluation {i}: expected {expected} for setpoint {inputs}, got {result[OBJECTIVE]}"
            )


class TestGridSnapping:
    """A gridded variable's value is snapped to its step before reaching hardware."""

    @pytest.mark.parametrize(
        ("grid", "expected_var1"),
        [
            pytest.param(
                {"var1": VariableGrid(min=0.0, max=10.0, step=1.0)}, 3.0, id="step_1.0_snaps_up"
            ),
            pytest.param(
                {"var1": VariableGrid(min=0.0, max=10.0, step=0.5)}, 2.5, id="step_0.5_snaps_down"
            ),
            pytest.param(None, 2.6, id="no_grid_passes_through"),
        ],
    )
    def test_inputs_snapped_to_their_grid(
        self, grid: dict[str, VariableGrid] | None, expected_var1: float
    ) -> None:
        env = _StaleMeasurementEnvironment()
        evaluator = Evaluator(env, OBJECTIVE, grid=grid)  # type: ignore[arg-type]

        # var1=2.6 snaps to its own step; ungridded var2 always passes through.
        result = evaluator({"var1": 2.6, "var2": 4.0})

        assert env._active_setpoint == {"var1": expected_var1, "var2": 4.0}
        # The returned inputs are the SNAPPED (on-grid) values actually applied,
        # not the proposed off-grid ones, so the loop records the point evaluated.
        assert result == {
            "var1": expected_var1,
            "var2": 4.0,
            OBJECTIVE: pytest.approx(expected_var1 + 4.0),
        }


class TestEvaluationGridAndNone:
    def test_grid_entry_absent_from_inputs_skips_snap(self) -> None:
        """A grid key not present in the inputs is skipped (no snap)."""
        env = EchoEnv({"obj": PVReading(value=1.0)})
        # "var_missing" is in the grid but never passed as an input.
        grid = {"var_missing": VariableGrid(min=0.0, max=10.0, step=1.0)}
        evaluator = Evaluator(env, "obj", grid=grid)  # type: ignore[arg-type]

        result = evaluator({"var_present": 2.6})

        # var_present is ungridded → passes through untouched; var_missing absent.
        # The evaluator returns the (snapped) inputs alongside the objective.
        assert env.last_inputs == {"var_present": 2.6}
        assert result == {"var_present": 2.6, "obj": 1.0}

    @pytest.mark.parametrize(
        "reading",
        [
            pytest.param(PVReading(value=None, connected=False), id="none_disconnected"),
            pytest.param(PVReading(value="open"), id="non_numeric"),
        ],
    )
    def test_unreadable_objective_returns_nan(self, reading: PVReading) -> None:
        """An objective that reads as None or as a non-numeric value is retried then
        returns NaN (never raised) — the loop records the point instead of aborting."""
        env = EchoEnv({"obj": reading})
        evaluator = Evaluator(env, "obj")  # type: ignore[arg-type]

        result = evaluator({"x": 1.0})
        assert math.isnan(result["obj"])


class TestA8NaNFlowsThroughEvaluator:
    """A8: a ``None``-valued reading (the PVReading "no data" signal) is
    retried up to ``objective_retries`` times and then tolerated as NaN."""

    def test_nan_objective_flows_through_evaluator(self) -> None:
        mock_env = MagicMock(spec=EnvironmentProtocol)
        mock_env.evaluate.return_value = {"charge_ltb": PVReading(value=None, connected=True)}

        evaluator = Evaluator(environment=mock_env, objective_name="charge_ltb")

        result = evaluator({"x": 0.5})

        assert math.isnan(result["charge_ltb"])
        # default objective_retries=2 → 3 total attempts
        assert mock_env.evaluate.call_count == evaluator.objective_retries + 1

    def test_objective_retried_then_succeeds(self) -> None:
        """First read returns unreadable; second returns a finite value.
        Evaluator must return the finite value and call evaluate exactly twice.
        """
        mock_env = MagicMock(spec=EnvironmentProtocol)
        mock_env.evaluate.side_effect = [
            {"charge_ltb": PVReading(value=None, connected=True)},
            {"charge_ltb": PVReading(value=1.5, connected=True)},
        ]

        evaluator = Evaluator(
            environment=mock_env, objective_name="charge_ltb", objective_retries=2
        )

        result = evaluator({"x": 0.5})

        assert result == {"x": 0.5, "charge_ltb": 1.5}
        assert mock_env.evaluate.call_count == 2

    def test_control_system_error_not_retried(self) -> None:
        """A ControlSystemUnreachableError from the environment aborts
        immediately — it is never retried and evaluate is called exactly once.
        """
        mock_env = MagicMock(spec=EnvironmentProtocol)
        mock_env.evaluate.side_effect = ControlSystemUnreachableError("unreachable")

        evaluator = Evaluator(
            environment=mock_env, objective_name="charge_ltb", objective_retries=2
        )

        with pytest.raises(ControlSystemUnreachableError):
            evaluator({"x": 0.5})

        assert mock_env.evaluate.call_count == 1


class TestConstraintMeasurement:
    """Constraint observables are measured alongside the objective in one
    ``evaluate`` transaction and returned in the outcome dict."""

    def test_objective_and_constraints_measured_in_one_call(self) -> None:
        mock_env = MagicMock(spec=EnvironmentProtocol)
        mock_env.evaluate.return_value = {
            "charge_ltb": PVReading(value=2.0),
            "loss": PVReading(value=0.3),
        }
        evaluator = Evaluator(mock_env, "charge_ltb", constraint_names=["loss"])

        result = evaluator({"x": 1.0})

        # A single evaluate call requested BOTH the objective and the constraint,
        # in order — constraints ride the same hardware transaction, no extra read.
        assert mock_env.evaluate.call_count == 1
        requested = mock_env.evaluate.call_args.args[1]
        assert requested == ["charge_ltb", "loss"]
        assert result == {"x": 1.0, "charge_ltb": 2.0, "loss": 0.3}

    def test_unreadable_constraint_triggers_retry_then_succeeds(self) -> None:
        """The whole measurement is retried while ANY outcome is unreadable; a
        constraint that reads on the second attempt is honored."""
        mock_env = MagicMock(spec=EnvironmentProtocol)
        mock_env.evaluate.side_effect = [
            {"charge_ltb": PVReading(value=2.0), "loss": PVReading(value=None, connected=True)},
            {"charge_ltb": PVReading(value=2.0), "loss": PVReading(value=0.3)},
        ]
        evaluator = Evaluator(
            mock_env, "charge_ltb", constraint_names=["loss"], objective_retries=2
        )

        result = evaluator({"x": 1.0})

        assert mock_env.evaluate.call_count == 2
        assert result == {"x": 1.0, "charge_ltb": 2.0, "loss": 0.3}

    def test_persistently_unreadable_constraint_becomes_nan(self) -> None:
        """A constraint unreadable after all retries is recorded as NaN (the
        objective still finite), so the loop records the point without aborting."""
        mock_env = MagicMock(spec=EnvironmentProtocol)
        mock_env.evaluate.return_value = {
            "charge_ltb": PVReading(value=2.0),
            "loss": PVReading(value=None, connected=True),
        }
        evaluator = Evaluator(
            mock_env, "charge_ltb", constraint_names=["loss"], objective_retries=2
        )

        result = evaluator({"x": 1.0})

        assert mock_env.evaluate.call_count == evaluator.objective_retries + 1
        assert result["charge_ltb"] == 2.0
        assert math.isnan(result["loss"])


class TestEvaluationFunction:
    """FakeInjectorEnvironment-driven smoke tests for the evaluation function."""

    @pytest.fixture(autouse=True)
    def setup_evaluator(self) -> None:
        """Set up test fixtures before each test."""
        self.env = FakeInjectorEnvironment()
        self.objective_name = "charge_ltb"
        self.evaluate_injector = Evaluator(self.env, self.objective_name)

    def test_evaluate_injector_efficiency_range(
        self, test_variable_bounds: dict[str, list[float]]
    ) -> None:
        """The objective is a finite, strictly-positive measurement.

        A bare ``>= 0.0`` also passes for a constant/zero evaluator; assert
        finite and ``> 0`` so a NaN, zero, or negative reading fails.
        """
        inputs = valid_variables(test_variable_bounds)

        result = self.evaluate_injector(inputs)
        charge = result[self.objective_name]

        assert math.isfinite(charge)
        assert charge > 0.0

    def test_evaluate_injector_multiple_calls(
        self, test_variable_bounds: dict[str, list[float]]
    ) -> None:
        """Repeated evaluation of the SAME input is deterministic.

        The old ``>= 0.0`` loop could not catch drift or a random evaluator;
        pin that all five reads return the identical value.
        """
        inputs = valid_variables(test_variable_bounds)

        results = [self.evaluate_injector(inputs)[self.objective_name] for _ in range(5)]

        assert all(math.isfinite(c) for c in results)
        assert len(set(results)) == 1, f"expected identical reads, got {results}"

    def test_evaluate_at_bounds(self, test_variable_bounds: dict[str, list[float]]) -> None:
        """At the bounds, the upper-corner objective strictly exceeds the lower.

        The old test only asserted each was ``>= 0.0``; a constant evaluator
        passed. Pin the ordering so the boundary evaluation actually responds.
        """
        lower_inputs = bounds_variables(test_variable_bounds, use_lower=True)
        result_lower = self.evaluate_injector(lower_inputs)[self.objective_name]

        upper_inputs = bounds_variables(test_variable_bounds, use_lower=False)
        result_upper = self.evaluate_injector(upper_inputs)[self.objective_name]

        assert math.isfinite(result_lower) and math.isfinite(result_upper)
        assert result_lower > 0.0
        assert result_upper > result_lower


class TestEvaluationErrors:
    """Input-validation errors raised by the evaluation function."""

    @pytest.fixture(autouse=True)
    def setup_evaluator(self) -> None:
        """Set up test fixtures before each test."""
        self.env = FakeInjectorEnvironment()
        self.evaluator = Evaluator(self.env, "charge_ltb")

    @pytest.mark.parametrize(
        ("var_name", "over_bound_by"),
        [
            pytest.param("INVALID_VAR", None, id="unknown_variable"),
            pytest.param("LN______MASTPH_AC00", 1000.0, id="out_of_bounds"),
        ],
    )
    def test_evaluation_with_rejected_variable_raises_error(
        self,
        test_variable_bounds: dict[str, list[float]],
        var_name: str,
        over_bound_by: float | None,
    ) -> None:
        """An unknown variable or an out-of-bounds setpoint is refused, not clamped."""
        inputs = valid_variables(test_variable_bounds)
        inputs[var_name] = (
            5.0 if over_bound_by is None else test_variable_bounds[var_name][1] + over_bound_by
        )

        with pytest.raises(VariableOutOfBoundsError):
            self.evaluator(inputs)
