from enum import Enum
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from osprey.services.optimizer.models.api import (
    EnvironmentDetails,
    HistoricalRunDetails,
    JobResponse,
    PointDelta,
    RestoreResponse,
    ResultSummary,
)
from osprey.services.optimizer.models.config import (
    ConstraintSpec,
    OptimizationConfig,
    Preset,
    VariableOverride,
)
from osprey.services.optimizer.models.enums import (
    AlgorithmType,
    ConstraintType,
    EnvironmentType,
    JobStatus,
    JobType,
)
from osprey.services.optimizer.models.protocols import (
    CompositeObservable,
    ObservableMetadata,
    OptimizerProtocol,
    PVObservable,
    VariableMetadata,
)
from osprey.services.optimizer.models.state import OptimizationState
from osprey.services.optimizer.models.types import (
    DIRECTIONS,
    Direction,
    parse_direction,
    require_finite,
    require_finite_positive,
    validate_min_le_max,
)


class TestConfig:
    """Tests for configuration logic and safety defaults."""

    def test_optimization_config_safety_defaults(self) -> None:
        """Ensure default configuration is valid."""
        config = OptimizationConfig()
        assert config.environment_config is not None

    def test_optimization_config_preserves_provided_environment_config(self) -> None:
        """A supplied environment_config is left untouched (validator False branch)."""
        config = OptimizationConfig(environment_config={"foo": "bar"})
        assert config.environment_config == {"foo": "bar"}

    def test_optimization_config_validation(self) -> None:
        """Test custom validators in OptimizationConfig."""
        config = OptimizationConfig(
            environment_name="test", variable_overrides={"var1": VariableOverride(min=0.0)}
        )
        assert config.variable_overrides is not None
        assert config.variable_overrides["var1"].min == 0.0

        # variable_overrides lives only at the top level — it is not copied into
        # environment_config (no environment consumer reads the nested copy).
        assert "variable_overrides" not in (config.environment_config or {})

        with pytest.raises(ValidationError):
            OptimizationConfig(algorithm="invalid_algo")  # type: ignore[arg-type]

    def test_optimization_config_rejects_negative_n_iterations(self) -> None:
        """Negative n_iterations is meaningless; 0 stays valid (init-only run)."""
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            OptimizationConfig(n_iterations=-1)
        assert OptimizationConfig(n_iterations=0).n_iterations == 0

    def test_optimization_config_rejects_n_initial_samples_below_one(self) -> None:
        """n_initial_samples must be >= 1; 0 is rejected, 1 is accepted, None means auto."""
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            OptimizationConfig(n_initial_samples=0)
        assert OptimizationConfig(n_initial_samples=1).n_initial_samples == 1
        assert OptimizationConfig(n_initial_samples=None).n_initial_samples is None

    def test_optimization_config_rejects_empty_variables(self) -> None:
        """An explicit empty list is rejected so it can't be silently read as 'all variables'."""
        with pytest.raises(ValidationError, match="variables must be non-empty"):
            OptimizationConfig(variables=[])
        # None (omitted) is still allowed and means "use all variables".
        assert OptimizationConfig(variables=None).variables is None

    def test_variable_override_window_fields_accepted_alone(self) -> None:
        """Window fields are accepted on their own (separately) without min/max."""
        assert VariableOverride(window_fraction=0.1).window_fraction == 0.1
        assert VariableOverride(window_abs=2.0).window_abs == 2.0

    def test_variable_override_rejects_min_with_window(self) -> None:
        """min/max and window_* are mutually exclusive."""
        with pytest.raises(ValidationError, match="mutually exclusive"):
            VariableOverride(min=0.0, window_fraction=0.1)
        with pytest.raises(ValidationError, match="mutually exclusive"):
            VariableOverride(max=10.0, window_abs=2.0)

    def test_variable_override_rejects_both_window_modes(self) -> None:
        """window_fraction and window_abs cannot be combined."""
        with pytest.raises(ValidationError, match="mutually exclusive"):
            VariableOverride(window_fraction=0.1, window_abs=2.0)

    def test_variable_override_rejects_non_positive_window(self) -> None:
        """Window values must be > 0 (Pydantic gt=0 enforces it)."""
        with pytest.raises(ValidationError):
            VariableOverride(window_fraction=0.0)
        with pytest.raises(ValidationError):
            VariableOverride(window_abs=-1.0)

    def test_variable_override_rejects_inverted_min_max(self) -> None:
        """min > max is meaningless and otherwise silently collapses to a single
        safety-edge point in resolve_bounds; reject it up front. min == max is allowed."""
        with pytest.raises(ValidationError, match="cannot exceed max"):
            VariableOverride(min=104.0, max=96.0)
        assert VariableOverride(min=5.0, max=5.0).min == 5.0

    def test_variable_override_rejects_non_finite_bounds(self) -> None:
        """NaN slips past the min > max check (NaN comparisons are False) and
        silently propagates into the resolved bounds / xopt VOCS; inf is nonsensical
        as a bound. Both must be rejected at the model boundary."""
        for bad in (float("nan"), float("inf"), float("-inf")):
            with pytest.raises(ValidationError, match="min must be a finite number"):
                VariableOverride(min=bad)
            with pytest.raises(ValidationError, match="max must be a finite number"):
                VariableOverride(max=bad)

    def test_variable_override_rejects_non_positive_or_non_finite_step_size(self) -> None:
        """step_size constrains the optimizer to a grid, so a zero/negative/non-finite
        step is meaningless and must be rejected at the boundary. A positive step is fine."""
        for bad in (0.0, -0.5, float("nan"), float("inf")):
            with pytest.raises(ValidationError, match="step_size must be a finite positive number"):
                VariableOverride(step_size=bad)
        assert VariableOverride(step_size=0.25).step_size == 0.25

    def test_variable_override_rejects_unknown_fields(self) -> None:
        """Forbid unknown keys so a typo like ``windowfraction`` surfaces instead of being silently dropped."""
        with pytest.raises(ValidationError):
            VariableOverride(windowfraction=0.1)  # type: ignore[call-arg]

    def test_constraints_default_empty(self) -> None:
        """constraints defaults to an empty dict (no constraints), not None."""
        assert OptimizationConfig().constraints == {}

    def test_constraint_spec_round_trip(self) -> None:
        """A ConstraintSpec survives JSON round-trip inside OptimizationConfig."""
        config = OptimizationConfig(
            objective_name="charge",
            constraints={"loss": ConstraintSpec(type=ConstraintType.LESS_THAN, value=0.5)},
        )
        restored = OptimizationConfig.model_validate_json(config.model_dump_json())
        spec = restored.constraints["loss"]
        assert spec.type is ConstraintType.LESS_THAN
        assert spec.value == 0.5

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_constraint_spec_rejects_non_finite_value(self, bad: float) -> None:
        """A NaN/inf threshold would silently propagate into the VOCS; reject it."""
        with pytest.raises(ValidationError, match="constraint value must be a finite number"):
            ConstraintSpec(type=ConstraintType.GREATER_THAN, value=bad)

    def test_constraint_spec_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            ConstraintSpec(type=ConstraintType.GREATER_THAN, value=1.0, threshold=2.0)  # type: ignore[call-arg]

    def test_constraint_on_objective_rejected(self) -> None:
        """A constraint keyed on the objective itself is unsupported in v1."""
        with pytest.raises(ValidationError, match="constraining the objective is not supported"):
            OptimizationConfig(
                objective_name="charge",
                constraints={"charge": ConstraintSpec(type=ConstraintType.GREATER_THAN, value=1.0)},
            )

    def test_constraint_on_non_objective_observable_accepted(self) -> None:
        """A constraint on a different observable than the objective is fine."""
        config = OptimizationConfig(
            objective_name="charge",
            constraints={"loss": ConstraintSpec(type=ConstraintType.LESS_THAN, value=0.1)},
        )
        assert "loss" in config.constraints

    def test_preset_accepts_algorithm_and_iterations(self) -> None:
        preset = Preset(algorithm=AlgorithmType.EXTREMUM_SEEKING, n_iterations=100)
        assert preset.algorithm is AlgorithmType.EXTREMUM_SEEKING
        assert preset.n_iterations == 100

    def test_preset_rejects_unknown_fields(self) -> None:
        """Forbid unknown keys so YAML typos surface instead of being silently dropped."""
        with pytest.raises(ValidationError):
            Preset(strategy="extremum_seeking")  # type: ignore[call-arg]

    @pytest.mark.parametrize(
        "fields",
        [
            pytest.param({"n_iterations": 0}, id="non_positive_iterations"),
            pytest.param({"variables": []}, id="empty_variables"),
            pytest.param(
                {"variables": ["a"], "variable_overrides": {"b": {"window_abs": 1.0}}},
                id="override_outside_variables",
            ),
        ],
    )
    def test_preset_rejects(self, fields: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            Preset.model_validate(fields)


class TestProtocols:
    """Tests for hardware safety validation logic."""

    def test_variable_metadata_logic(self) -> None:
        """Verify custom validators for hardware bounds."""
        VariableMetadata(
            min_value=0.0,
            max_value=10.0,
            readback_pv="RBV",
            setpoint_pv="SP",
        )

        # Inverted bounds are a safety risk.
        with pytest.raises(ValidationError, match="cannot exceed max"):
            VariableMetadata(min_value=10.0, max_value=0.0, readback_pv="RBV", setpoint_pv="SP")

    def test_pv_observable_bounds_validation(self) -> None:
        """PVObservable shares the _ObservableBase bounds validator: min must not exceed max."""
        valid = PVObservable(
            description="Valid",
            units="A",
            min_value=0.0,
            max_value=10.0,
            readback_pv="TEST:PV",
        )
        assert valid.units == "A"

        with pytest.raises(ValidationError, match=r"min \(10\.0\) cannot exceed max \(5\.0\)"):
            PVObservable(
                units="A",
                min_value=10.0,
                max_value=5.0,
                readback_pv="TEST:PV",
            )


class TestOptimizerProtocol:
    """The runtime-checkable OptimizerProtocol matches an object that has both
    ask/tell methods and rejects one missing either of them."""

    def test_minimal_implementation_passes_isinstance(self) -> None:
        class FakeOptimizer:
            def suggest(self, num_points: int | None = None) -> list[dict[str, object]]:
                return [{"x": 0.0}]

            def ingest(self, points: list[dict[str, object]]) -> None:
                pass

        assert isinstance(FakeOptimizer(), OptimizerProtocol)

    def test_missing_method_fails_isinstance(self) -> None:
        class MissingIngest:
            def suggest(self, num_points: int | None = None) -> list[dict[str, object]]:
                return []

        assert not isinstance(MissingIngest(), OptimizerProtocol)


class TestApiContract:
    """Tests for API Contract Serialization (Over-the-wire safety)."""

    def test_optimization_state_serialization(self) -> None:
        """Ensure complex state objects can actually be serialized to JSON."""
        state = JobResponse(
            job_id="job1",
            job_type=JobType.OPTIMIZATION,
            status=JobStatus.RUNNING,
            data=[
                {"x": 1.0, "y": 0.5, "valid": True},
                {"x": 2.0, "y": None, "valid": False},
            ],
            snapshots={"initial": {"x": 1.0}},
        )
        json_str = state.model_dump_json()
        restored = JobResponse.model_validate_json(json_str)
        assert restored.job_id == "job1"
        assert restored.data is not None
        assert restored.data[1]["y"] is None
        assert restored.data[0]["y"] == 0.5
        assert restored.snapshots == {"initial": {"x": 1.0}}

    def test_optimization_state_variable_bounds_round_trips(self) -> None:
        """variable_bounds (resolved [min, max] per variable) survives JSON."""
        state = JobResponse(
            job_id="job_b",
            job_type=JobType.OPTIMIZATION,
            status=JobStatus.RUNNING,
            variable_bounds={"x": [0.0, 10.0], "y": [-1.5, 1.5]},
        )
        restored = JobResponse.model_validate_json(state.model_dump_json())
        assert restored.variable_bounds == {"x": [0.0, 10.0], "y": [-1.5, 1.5]}

    def test_optimization_state_variable_bounds_defaults_none(self) -> None:
        """variable_bounds is optional and defaults to None."""
        state = OptimizationState(job_id="job_c", status=JobStatus.CREATED)
        assert state.variable_bounds is None

    def test_job_response_error_message_round_trips(self) -> None:
        resp = JobResponse(
            job_id="j3",
            job_type=JobType.OPTIMIZATION,
            status=JobStatus.ERROR,
            error_message="boom",
        )
        dumped = resp.model_dump_json()
        assert "message" not in resp.model_dump()  # ensure no old key
        restored = JobResponse.model_validate_json(dumped)
        assert restored.error_message == "boom"

    def test_historical_run_serialization(self) -> None:
        """Test serialization of historical data contracts."""
        details = HistoricalRunDetails(
            timestamp="2023-01-01",
            config={"param": 1, "nested": {"a": 1}},
            best_point={"x": 0.5},
            initial_point=None,
            data_records=[],
        )

        json_str = details.model_dump_json()
        restored = HistoricalRunDetails.model_validate_json(json_str)
        assert restored.config["nested"]["a"] == 1


class TestResultSummaryModels:
    """Tests for PointDelta, ResultSummary, and target_value on metadata."""

    def test_result_summary_serialization(self) -> None:
        summary = ResultSummary(
            best_points=[
                PointDelta(
                    values={"var1": 1.0, "obj": 0.9},
                    delta_vs_initial={"var1": 0.5, "obj": 0.4},
                    delta_vs_target={"obj": -0.1},
                )
            ],
            initial_point={"var1": 0.5, "obj": 0.5},
            target_value={"obj": 1.0},
        )
        json_str = summary.model_dump_json()
        restored = ResultSummary.model_validate_json(json_str)
        assert len(restored.best_points) == 1
        assert restored.best_points[0].values["obj"] == 0.9
        assert restored.initial_point is not None
        assert restored.target_value == {"obj": 1.0}

    def test_result_summary_empty(self) -> None:
        summary = ResultSummary(best_points=[])
        json_str = summary.model_dump_json()
        restored = ResultSummary.model_validate_json(json_str)
        assert restored.best_points == []
        assert restored.initial_point is None
        assert restored.target_value is None

    def test_optimization_status_response_with_summary(self) -> None:
        resp = JobResponse(
            job_id="j1",
            job_type=JobType.OPTIMIZATION,
            status=JobStatus.RUNNING,
            result_summary=ResultSummary(
                best_points=[PointDelta(values={"x": 1.0})],
            ),
        )
        json_str = resp.model_dump_json()
        restored = JobResponse.model_validate_json(json_str)
        assert restored.result_summary is not None
        assert len(restored.result_summary.best_points) == 1

    def test_optimization_status_response_without_summary(self) -> None:
        resp = JobResponse(job_id="j2", job_type=JobType.OPTIMIZATION, status=JobStatus.IDLE)
        assert resp.result_summary is None

    def test_observable_metadata_target_value(self) -> None:
        meta = PVObservable(
            readback_pv="PV:TEST",
            target_value=5.0,
        )
        assert meta.target_value == 5.0

    def test_observable_metadata_target_value_default_none(self) -> None:
        meta = PVObservable(readback_pv="PV:TEST")
        assert meta.target_value is None

    def test_pv_observable_rejects_source_observables(self) -> None:
        with pytest.raises(ValidationError):
            PVObservable(readback_pv="PV:X", source_observables=["a"])  # type: ignore[call-arg]

    def test_composite_observable_rejects_readback_pv(self) -> None:
        with pytest.raises(ValidationError):
            CompositeObservable(source_observables=["a"], readback_pv="PV:X")  # type: ignore[call-arg]

    def test_observable_metadata_json_round_trip_variants(self) -> None:
        adapter: TypeAdapter[PVObservable | CompositeObservable] = TypeAdapter(ObservableMetadata)
        pv = PVObservable(readback_pv="PV:X", units="nC")
        parsed_pv = adapter.validate_json(pv.model_dump_json())
        assert isinstance(parsed_pv, PVObservable)
        assert parsed_pv.readback_pv == "PV:X"

        comp = CompositeObservable(source_observables=["a", "b"], units="nC")
        parsed_comp = adapter.validate_json(comp.model_dump_json())
        assert isinstance(parsed_comp, CompositeObservable)
        assert parsed_comp.source_observables == ["a", "b"]

    def test_environment_details_json_round_trip_embedded(self) -> None:
        details = EnvironmentDetails(
            name="mock",
            environment_type=EnvironmentType.SIMULATION,
            variables={},
            observables={
                "direct": PVObservable(readback_pv="PV:X"),
                "computed": CompositeObservable(source_observables=["direct"]),
            },
            available_objectives=["direct"],
            default_objective="direct",
        )
        restored = EnvironmentDetails.model_validate_json(details.model_dump_json())
        assert isinstance(restored.observables["direct"], PVObservable)
        assert isinstance(restored.observables["computed"], CompositeObservable)


class TestValidationHelpers:
    """Tests for the shared validation helpers in ``types``."""

    def test_validate_min_le_max_accepts_ordered_and_equal(self) -> None:
        validate_min_le_max(1.0, 2.0)
        validate_min_le_max(5.0, 5.0)

    def test_validate_min_le_max_rejects_inverted(self) -> None:
        with pytest.raises(ValueError, match="cannot exceed max"):
            validate_min_le_max(2.0, 1.0)

    def test_validate_min_le_max_label_in_message(self) -> None:
        with pytest.raises(ValueError, match="safety min"):
            validate_min_le_max(2.0, 1.0, label="safety min")

    def test_require_finite_accepts_finite(self) -> None:
        require_finite(0.0)
        require_finite(-1e9)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_require_finite_rejects_non_finite(self, bad: float) -> None:
        with pytest.raises(ValueError, match="must be a finite number"):
            require_finite(bad, label="step")

    def test_require_finite_positive_accepts_positive(self) -> None:
        require_finite_positive(0.001)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_require_finite_positive_rejects(self, bad: float) -> None:
        with pytest.raises(ValueError, match="must be a finite positive number"):
            require_finite_positive(bad)

    def test_directions_set_matches_literal(self) -> None:
        assert {"MAXIMIZE", "MINIMIZE"} == DIRECTIONS

    def test_direction_is_str_enum(self) -> None:
        assert issubclass(Direction, Enum)
        assert Direction.MAXIMIZE.value == "MAXIMIZE"
        assert Direction.MINIMIZE.value == "MINIMIZE"
        assert Direction.MAXIMIZE == "MAXIMIZE"

    def test_direction_serializes_as_plain_string(self) -> None:
        from pydantic import BaseModel

        class _M(BaseModel):
            d: Direction

        assert _M(d=Direction.MAXIMIZE).model_dump(mode="json") == {"d": "MAXIMIZE"}
        assert _M.model_validate({"d": "MINIMIZE"}).d is Direction.MINIMIZE

    @pytest.mark.parametrize("value", ["MAXIMIZE", "MINIMIZE"])
    def test_parse_direction_accepts_canonical(self, value: str) -> None:
        assert parse_direction(value) == value

    @pytest.mark.parametrize("bad", ["min", "max", "minimize", "maximize", "Maximize", "sideways"])
    def test_parse_direction_rejects_non_canonical(self, bad: str) -> None:
        """Only the exact canonical spellings are accepted; aliases/casings are rejected."""
        with pytest.raises(ValueError, match="Invalid objective direction"):
            parse_direction(bad)


class TestRestoreResponse:
    def test_roundtrips(self) -> None:
        r = RestoreResponse(
            environment="als_injector",
            control_target="live",
            restored={"v1": 1.5, "v2": 3.0},
            count=2,
        )
        assert r.environment == "als_injector"
        assert r.restored == {"v1": 1.5, "v2": 3.0}
        assert r.count == 2
        assert RestoreResponse.model_validate_json(r.model_dump_json()) == r


class TestEnums:
    def test_created_status_exists_and_is_not_terminal(self) -> None:
        assert JobStatus.CREATED.value == "created"
        assert JobStatus.CREATED.is_terminal is False


def test_current_job_response_allows_null_and_value() -> None:
    from osprey.services.optimizer.models.api import CurrentJobResponse

    assert CurrentJobResponse().job_id is None
    assert CurrentJobResponse(job_id="j1").job_id == "j1"


def test_current_job_request_requires_job_id() -> None:
    import pytest
    from pydantic import ValidationError

    from osprey.services.optimizer.models.api import CurrentJobRequest

    assert CurrentJobRequest(job_id="j1").job_id == "j1"
    with pytest.raises(ValidationError):
        CurrentJobRequest()  # type: ignore[call-arg]


def test_result_info_job_id_defaults_none_and_round_trips() -> None:
    from osprey.services.optimizer.models.api import ResultInfo

    legacy = ResultInfo(run_id="env/ts", environment="env", timestamp="ts", has_model=False)
    assert legacy.job_id is None

    linked = ResultInfo(
        run_id="env/ts", environment="env", timestamp="ts", has_model=False, job_id="job-123"
    )
    assert linked.model_dump()["job_id"] == "job-123"
