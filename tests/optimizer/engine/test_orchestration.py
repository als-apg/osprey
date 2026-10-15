"""Unit tests for OptimizationEngine."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from osprey_optimizer.events import (
    EventBus,
    EventType,
    OptimizationEvent,
    OptimizationStartedData,
)
from osprey_optimizer.models.config import (
    ConstraintSpec,
    OptimizationConfig,
    VariableOverride,
)
from osprey_optimizer.models.enums import AlgorithmType, ConstraintType
from osprey_optimizer.models.exceptions import ConnectionError as OptimizerConnectionError
from osprey_optimizer.models.protocols import PVReading
from osprey_optimizer.optimization.runner import PhaseResult
from osprey_optimizer.services.draft_resolution import resolve_draft_config
from osprey_optimizer.services.orchestration import (
    OptimizationEngine,
    RunnerServices,
    resolve_environment,
)
from osprey_optimizer.services.plugin_registry import EnvironmentRegistry

from ._fakes import FakeInjectorEnvironment
from .conftest import make_mock_environment


def _config(**overrides: Any) -> OptimizationConfig:
    """Build an ``OptimizationConfig`` with this file's recurring defaults, overridden as needed."""
    defaults: dict[str, Any] = {
        "objective_name": "charge_ltb",
        "algorithm": AlgorithmType.RANDOM,
        "n_iterations": 5,
        "environment_name": "als_injector",
        "control_target": "va",
        "environment_config": {},
    }
    defaults.update(overrides)
    return OptimizationConfig(**defaults)


def test_top_level_fields_drive_run_when_environment_config_is_empty(
    mock_plugin_env: FakeInjectorEnvironment,
) -> None:
    """Selection/objective come from the top-level config, not ``environment_config``.

    Web/API-built jobs leave ``environment_config`` empty (the model no longer
    copies ``variable_overrides`` into it), so the plugin env is constructed with
    no active-variable filter and the class-default objective. The orchestration
    engine must still optimize only the top-level ``variables`` subset against the
    top-level ``objective_name`` — proving the run is driven by the top-level
    fields, and that an empty ``environment_config`` does not silently widen the
    run to all variables or fall back to the default objective.
    """
    selected = ["LN______MASTPH_AC00", "LN______AS2_PH_AC00"]
    excluded = "EG______BIAS___AC01"  # a real env variable deliberately left out

    config = _config(variables=selected)

    registry = EnvironmentRegistry()
    # The env exposes far more than the subset, so a run that honored
    # environment_config (empty -> no filter -> all vars) would optimize all of them.
    full_env = registry.load_environment("als_injector", {})
    assert excluded in full_env.variables
    assert len(full_env.variables) > len(selected)

    with mock_plugin_env:
        resolve_draft_config(config, mock_plugin_env)

    result = OptimizationEngine(config, environment_factory=registry.load_environment).run()

    # Only the selected variables are decision variables in the optimization data...
    for name in selected:
        assert name in result.data.columns
    # ...and a real-but-unselected variable never entered the run.
    assert excluded not in result.data.columns
    # The objective came from the top-level field.
    assert "charge_ltb" in result.data.columns
    # The resolved config still carries exactly the selected subset.
    assert set(result.metadata.config.variables or []) == set(selected)
    # The run completed against the top-level objective and produced a best point.
    assert result.metadata.config.objective_name == "charge_ltb"
    assert result.metadata.config is config
    # 1 initial sample (RANDOM default) + the 5 configured iterations.
    assert result.data.height == 6
    assert isinstance(result.best_point, dict)


def test_optimization_engine_uses_factory(patched_engine_collaborators: MagicMock) -> None:
    """Verify that OptimizationEngine uses the provided factory to load plugins."""
    config = _config(
        environment_name="test_plugin",
        environment_config={"foo": "bar"},
        variables=["test_var"],
        variable_overrides={"test_var": VariableOverride(min=0.0, max=1.0, step_size=0.1)},
    )

    mock_env = make_mock_environment()
    mock_factory = MagicMock(return_value=mock_env)

    engine = OptimizationEngine(config, environment_factory=mock_factory)

    mock_data = MagicMock()
    patched_engine_collaborators.return_value = PhaseResult(data=mock_data)

    result = engine.run()

    # The job's target rides into the environment config beside its own keys.
    mock_factory.assert_called_once_with(
        "test_plugin", {"foo": "bar", "control_target": "va", "execution_mode": None}
    )
    assert result.data is mock_data


def _env_efficiency_only() -> MagicMock:
    """Env exposing a single ``efficiency`` observable (no ``charge_ltb``)."""
    env = MagicMock()
    env.observables = {"efficiency": MagicMock()}
    env.available_objectives = ["efficiency"]
    return env


class TestOptimizationEngineValidation:
    @pytest.mark.usefixtures("patched_engine_collaborators")
    @pytest.mark.parametrize(
        ("environment_factory", "config", "match"),
        [
            pytest.param(
                MagicMock(),
                OptimizationConfig(environment_name=None),
                "environment_name must be set",
                id="environment_name_none",
            ),
            pytest.param(
                MagicMock(return_value=_env_efficiency_only()),
                OptimizationConfig(
                    environment_name="test_env", control_target="va", objective_name="nonexistent"
                ),
                "not found in environment",
                id="objective_not_in_observables",
            ),
            pytest.param(
                MagicMock(return_value=_env_efficiency_only()),
                OptimizationConfig(
                    environment_name="test_env", control_target="va", objective_name=None
                ),
                "objective_name must be set",
                id="missing_objective",
            ),
            pytest.param(
                MagicMock(
                    return_value=make_mock_environment(
                        variables={"var_a": MagicMock(min_value=0.0, max_value=10.0, units="A")},
                    )
                ),
                OptimizationConfig(
                    environment_name="test_env",
                    control_target="va",
                    objective_name="charge_ltb",
                    variables=["var_a", "ghost_var"],
                ),
                r"ghost_var.*not found in environment",
                id="unknown_variable",
            ),
            pytest.param(
                MagicMock(return_value=make_mock_environment()),
                OptimizationConfig(
                    environment_name="test_env",
                    control_target="va",
                    objective_name="charge_ltb",
                    variables=["test_var"],
                    constraints={
                        "ghost_obs": ConstraintSpec(type=ConstraintType.LESS_THAN, value=1.0)
                    },
                ),
                r"ghost_obs.*not found in environment",
                id="unknown_constraint_observable",
            ),
            pytest.param(
                MagicMock(return_value=make_mock_environment()),
                OptimizationConfig(
                    environment_name="test_env", control_target="va", objective_name="charge_ltb"
                ),
                "not resolved",
                id="run_rejects_unresolved_config",
            ),
        ],
    )
    def test_run_raises_value_error(
        self,
        environment_factory: MagicMock,
        config: OptimizationConfig,
        match: str,
    ) -> None:
        engine = OptimizationEngine(config, environment_factory=environment_factory)
        with pytest.raises(ValueError, match=match):
            engine.run()

    def test_override_step_size_builds_grid_and_snaps_bounds(self) -> None:
        """Each variable's grid uses exactly the resolved override's own step_size
        (draft resolution has already snapped bounds to that grid before run() sees it),
        and the resolved bounds thread through to the VOCS verbatim (two-phase contract:
        run() consumes the overrides, it does not compute them)."""
        mock_env = make_mock_environment(
            variables={
                "var_a": MagicMock(min_value=0.0, max_value=10.0, units="A", step_size=0.1),
                "var_b": MagicMock(min_value=0.0, max_value=10.0, units="V", step_size=0.5),
            },
        )
        mock_factory = MagicMock(return_value=mock_env)

        config = _config(
            environment_name="test_env",
            variables=["var_a", "var_b"],
            variable_overrides={
                # Already resolved: [1,9] step 2 anchored at safety_min=0 snaps to [2,8].
                "var_a": VariableOverride(min=2.0, max=8.0, step_size=2.0),
                # No override at authoring time -> resolved to the full hardware step.
                "var_b": VariableOverride(min=0.0, max=10.0, step_size=0.5),
            },
        )

        engine = OptimizationEngine(config, environment_factory=mock_factory)

        with (
            patch("osprey_optimizer.services.orchestration.run_optimization_phase") as mock_run,
            patch("osprey_optimizer.services.orchestration.report_results"),
            patch("osprey_optimizer.services.orchestration.Evaluator") as mock_evaluator,
        ):
            mock_run.return_value = PhaseResult(data=MagicMock())
            engine.run()

        grid = mock_evaluator.call_args.kwargs["grid"]
        assert set(grid) == {"var_a", "var_b"}
        assert (grid["var_a"].min, grid["var_a"].max, grid["var_a"].step) == (2.0, 8.0, 2.0)
        assert (grid["var_b"].min, grid["var_b"].max, grid["var_b"].step) == (0.0, 10.0, 0.5)
        bounds = mock_run.call_args.kwargs["bounds"]
        assert bounds["var_a"] == [2.0, 8.0]
        assert bounds["var_b"] == [0.0, 10.0]

    def test_constraints_threaded_into_collaborators(self) -> None:
        """A known constraint is passed to the Evaluator (as names), the phase runner,
        and report_results (as specs)."""
        env = make_mock_environment(objectives={"charge_ltb": "MAXIMIZE", "loss": "MINIMIZE"})
        constraints = {"loss": ConstraintSpec(type=ConstraintType.LESS_THAN, value=0.5)}
        config = _config(
            environment_name="test_env",
            variables=["test_var"],
            variable_overrides={"test_var": VariableOverride(min=0.0, max=1.0, step_size=0.1)},
            constraints=constraints,
        )
        engine = OptimizationEngine(config, environment_factory=MagicMock(return_value=env))

        with (
            patch("osprey_optimizer.services.orchestration.run_optimization_phase") as mock_run,
            patch("osprey_optimizer.services.orchestration.report_results") as mock_report,
            patch("osprey_optimizer.services.orchestration.Evaluator") as mock_evaluator,
        ):
            mock_run.return_value = PhaseResult(data=MagicMock())
            engine.run()

        assert mock_evaluator.call_args.kwargs["constraint_names"] == ["loss"]
        assert mock_run.call_args.kwargs["constraints"] == constraints
        assert mock_report.call_args.kwargs["constraints"] == constraints

    @pytest.mark.parametrize(
        ("variables", "variable_overrides", "initial_point", "expected_bounds"),
        [
            pytest.param(
                None,  # variables=None -> resolves to every env variable
                {
                    "var_a": VariableOverride(max=8.0),  # min=None -> falls back to safety_min
                    "missing_var": VariableOverride(
                        min=1.0, max=2.0
                    ),  # not an active variable -> ignored
                },
                None,  # initial_point=None -> the engine reads the env
                {"var_a": [0.0, 8.0], "var_b": [-5.0, 5.0]},
                id="unknown_var_skipped_partial_bounds_filled",
            ),
            pytest.param(
                ["var_a", "var_b"],
                {"var_a": VariableOverride(window_abs=1.5)},
                {"var_a": 5.0, "var_b": 0.0},
                # var_a centered on its 5.0 readback; var_b has no override -> full safety range.
                {"var_a": [3.5, 6.5], "var_b": [-5.0, 5.0]},
                id="window_abs_centered_on_readback",
            ),
        ],
    )
    def test_resolved_overrides_thread_into_bounds(
        self,
        patched_engine_collaborators: MagicMock,
        variables: list[str] | None,
        variable_overrides: dict[str, VariableOverride],
        initial_point: dict[str, float] | None,
        expected_bounds: dict[str, list[float]],
    ) -> None:
        """Draft resolution owns the override semantics — ignoring unknown names, filling
        partial bounds from the safety range, and centering a ``window_abs`` on the current
        readback — and the resolved result threads through run() unchanged."""
        mock_env = make_mock_environment(
            variables={
                "var_a": MagicMock(min_value=0.0, max_value=10.0, units="A", step_size=0.1),
                "var_b": MagicMock(min_value=-5.0, max_value=5.0, units="V", step_size=0.1),
            },
        )
        mock_env.get_variables.return_value = {
            "var_a": PVReading(value=5.0),
            "var_b": PVReading(value=0.0),
        }
        mock_factory = MagicMock(return_value=mock_env)

        config = _config(
            environment_name="test_env",
            variables=variables,
            variable_overrides=variable_overrides,
        )
        with mock_env:
            resolve_draft_config(config, mock_env)

        engine = OptimizationEngine(
            config, environment_factory=mock_factory, initial_point=initial_point
        )

        patched_engine_collaborators.return_value = PhaseResult(data=MagicMock())
        engine.run()

        assert patched_engine_collaborators.call_args.kwargs["bounds"] == expected_bounds

    def test_initial_point_none_reads_env_and_raises_on_failed_reads(
        self, patched_engine_collaborators: MagicMock
    ) -> None:
        """When initial_point is None, the engine reads env vars and raises if any are missing."""
        del patched_engine_collaborators  # patch side-effects only

        env = make_mock_environment(
            variables={"var_a": MagicMock(min_value=0.0, max_value=1.0, units="A")}
        )
        env.get_variables.return_value = {"var_a": PVReading(value=None, connected=False)}

        config = _config(
            environment_name="test_env",
            variables=["var_a"],
            variable_overrides={"var_a": VariableOverride(min=0.0, max=1.0, step_size=0.1)},
        )
        engine = OptimizationEngine(
            config,
            environment_factory=MagicMock(return_value=env),
            initial_point=None,
        )

        with pytest.raises(OptimizerConnectionError, match="Cannot read variables"):
            engine.run()

    def test_optimization_started_carries_resolved_variable_bounds(
        self, patched_engine_collaborators: MagicMock
    ) -> None:
        """OPTIMIZATION_STARTED carries the resolved [min, max] per variable (the VOCS bounds).

        var_a's override exceeds its safety range, so draft resolution clamps it before
        run() ever sees the config — proving the event is sourced from the resolved grid,
        not a raw request run() would have to compute itself.
        """
        mock_env = make_mock_environment(
            variables={
                "var_a": MagicMock(min_value=0.0, max_value=10.0, units="A", step_size=0.1),
                "var_b": MagicMock(min_value=-5.0, max_value=5.0, units="V", step_size=0.1),
            },
        )
        mock_factory = MagicMock(return_value=mock_env)

        config = _config(
            environment_name="test_env",
            variables=["var_a", "var_b"],
            variable_overrides={"var_a": VariableOverride(min=-3.0, max=15.0)},  # exceeds [0, 10]
        )
        with mock_env:
            resolve_draft_config(config, mock_env)

        bus = EventBus()
        captured: list[OptimizationEvent] = []
        bus.subscribe(EventType.OPTIMIZATION_STARTED, captured.append)

        engine = OptimizationEngine(
            config,
            environment_factory=mock_factory,
            services=RunnerServices(event_bus=bus),
            initial_point={"var_a": 5.0, "var_b": 0.0},
        )
        patched_engine_collaborators.return_value = PhaseResult(data=MagicMock())
        engine.run()

        assert len(captured) == 1
        data = captured[0].data
        assert isinstance(data, OptimizationStartedData)
        assert data.variable_bounds == {"var_a": [0.0, 10.0], "var_b": [-5.0, 5.0]}

    def test_subscribe_all_registers_listener_for_every_event_type(self) -> None:
        """``EventBus.subscribe_all`` must wire a listener to every type, including new ones.

        Regression guard for the bug where hard-coded subscription tuples in CLI/service
        forwarders silently dropped any event added later (see BOUNDS_CLAMPED).
        """
        bus = EventBus()

        def callback(_event: object) -> None:
            return None

        bus.subscribe_all(callback)
        for event_type in EventType:
            assert callback in bus._subscribers[event_type], f"{event_type} not subscribed"


def test_a_config_without_a_target_never_reaches_the_factory() -> None:
    """The worker's own refusal: no target named, no environment built."""
    config = _config(control_target=None)
    factory = MagicMock()
    with pytest.raises(ValueError, match="control_target"):
        resolve_environment(config, factory)
    factory.assert_not_called()
