"""Tests for auto-set fixed_features with pre-trained surrogate models.

Covers:
- _compute_fixed_features logic (runner.py)
- _resolve_n_initial with model_file
- model bundle saved via result pipeline (results.py)
- orchestration threading of current_env_values
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import polars as pl
import pytest
from sklearn.ensemble import RandomForestRegressor
from xopt import VOCS
from xopt.generators import get_generator
from xopt.generators.bayesian.bayesian_generator import BayesianGenerator

from osprey_optimizer.models.config import OptimizationConfig, VariableOverride
from osprey_optimizer.models.enums import AlgorithmType, SurrogateModelType
from osprey_optimizer.models.protocols import PVReading
from osprey_optimizer.optimization.runner import (
    PhaseResult,
)
from osprey_optimizer.optimization.xopt_backend import (
    _compute_fixed_features,
    _resolve_n_initial,
)
from osprey_optimizer.results import ResultMetadata, save_optimization_results
from osprey_optimizer.services.orchestration import OptimizationEngine

from ._fakes import EchoEnv
from .conftest import make_mock_environment, run_phase_with_optimizer


def _dump(tmp_path: Path, obj: Any) -> str:
    """joblib-dump *obj* into the tmp dir and return the path _compute_fixed_features takes."""
    path = tmp_path / "model.joblib"
    joblib.dump(obj, path)
    return str(path)


class TestComputeFixedFeatures:
    def test_returns_fixed_for_extra_vars(self, tmp_path: Path) -> None:
        path = _dump(tmp_path, {"model": "fake", "variables": ["x", "y", "z"]})

        vocs = VOCS(variables={"x": [0, 1]}, objectives={"f": "MAXIMIZE"})
        env_vals = {"x": 0.5, "y": 1.0, "z": 2.0}
        result = _compute_fixed_features(path, vocs, env_vals)
        assert result == {"y": 1.0, "z": 2.0}

    @pytest.mark.parametrize(
        ("dump_obj", "vocs_vars", "env_vals"),
        [
            pytest.param(
                {"model": "fake", "variables": ["x", "y"]},
                {"x": [0, 1], "y": [0, 1]},
                {"x": 0.5, "y": 1.0},
                id="no_extra_vars",
            ),
        ],
    )
    def test_returns_none(
        self,
        tmp_path: Path,
        dump_obj: Any,
        vocs_vars: dict[str, list[int]],
        env_vals: dict[str, float],
    ) -> None:
        vocs = VOCS(variables=vocs_vars, objectives={"f": "MAXIMIZE"})
        result = _compute_fixed_features(_dump(tmp_path, dump_obj), vocs, env_vals)
        assert result is None

    @pytest.mark.parametrize(
        ("dump_obj", "vocs_vars", "env_vals", "match"),
        [
            pytest.param(
                RandomForestRegressor(n_estimators=5),
                {"x": [0, 1]},
                {"x": 0.5},
                "Not a model bundle",
                id="not_a_bundle",
            ),
            pytest.param(
                {"model": "fake", "variables": []},
                {"x": [0, 1]},
                {"x": 0.5},
                "does not cover",
                id="empty_variables_list",
            ),
            pytest.param(
                {"model": "fake", "variables": ["x"]},
                {"x": [0, 1], "y": [0, 1]},
                {"x": 0.5, "y": 1.0},
                "does not cover",
                id="model_missing_vocs_var",
            ),
            pytest.param(
                {"model": "fake", "variables": ["x", "y", "z"]},
                {"x": [0, 1]},
                {"x": 0.5, "y": 1.0},
                "not available in the current environment",
                id="extra_model_var_missing_env_value",
            ),
        ],
    )
    def test_raises(
        self,
        tmp_path: Path,
        dump_obj: Any,
        vocs_vars: dict[str, list[int]],
        env_vals: dict[str, float],
        match: str,
    ) -> None:
        """A model_file that is not a bundle, whose variables do not cover the
        VOCS variables, or whose extra variables have no current environment
        value, fails the job instead of silently running without fixed
        features."""
        vocs = VOCS(variables=vocs_vars, objectives={"f": "MAXIMIZE"})
        with pytest.raises(ValueError, match=match):
            _compute_fixed_features(_dump(tmp_path, dump_obj), vocs, env_vals)


class TestResolveNInitialWithModelFile:
    @pytest.fixture
    def bayesian_gen(self) -> BayesianGenerator:
        vocs = VOCS(variables={"x": [0, 1]}, objectives={"f": "MAXIMIZE"})
        gen = get_generator("upper_confidence_bound")(vocs=vocs)
        assert isinstance(gen, BayesianGenerator)
        return gen

    @pytest.mark.parametrize(
        ("use_bayesian", "n_initial_samples", "model_file", "expected", "expect_warning"),
        [
            pytest.param(
                True, None, "/some/model.joblib", 1, False, id="defaults_to_1_with_model_file"
            ),
            pytest.param(True, 3, "/some/model.joblib", 3, True, id="explicit_value_respected"),
            pytest.param(True, None, None, max(2, 5 + 1), False, id="no_model_file_uses_default"),
            pytest.param(
                False, None, "/some/model.joblib", 1, False, id="non_bayesian_ignores_model_file"
            ),
            pytest.param(False, 0, None, 1, False, id="explicit_below_minimum_clamps_to_one"),
            pytest.param(True, 10, "/some/model.joblib", 10, True, id="explicit_high_value_warns"),
            pytest.param(
                True, 1, "/some/model.joblib", 1, False, id="explicit_value_of_1_is_silent"
            ),
        ],
    )
    def test_resolve_n_initial(
        self,
        bayesian_gen: BayesianGenerator,
        caplog: pytest.LogCaptureFixture,
        use_bayesian: bool,
        n_initial_samples: int | None,
        model_file: str | None,
        expected: int,
        expect_warning: bool,
    ) -> None:
        """The five distinct exits of _resolve_n_initial: without model_file a Bayesian
        generator still auto-computes n_initial; model_file only affects Bayesian
        generators, others default to 1; an explicit value below 1 clamps to 1.
        With a model_file, an explicit value above 1 logs the prior-knowledge
        advisory; 1 (or None) stays silent.
        """
        with caplog.at_level(logging.WARNING):
            result = _resolve_n_initial(
                n_initial_samples=n_initial_samples,
                generator=bayesian_gen if use_bayesian else MagicMock(),
                n_variables=5,
                model_file=model_file,
            )
        assert result == expected
        assert ("prior knowledge" in caplog.text) is expect_warning


class TestModelBundleSaving:
    def test_model_without_variables_raises(self, tmp_path: Path) -> None:
        """A trained model must come with its variable names — never a raw dump."""
        rf = RandomForestRegressor(n_estimators=5)

        with pytest.raises(ValueError, match="model_variables is required"):
            save_optimization_results(
                data=pl.DataFrame({"x": [1.0, 2.0], "obj": [0.1, 0.2]}),
                best_point={"x": 2.0},
                config=OptimizationConfig(
                    environment_name="test_env", control_target="va", n_iterations=5
                ),
                output_dir=tmp_path / "results",
                metadata=ResultMetadata(model=rf, model_variables=None),
            )


class TestOrchestrationCurrentEnvValues:
    @pytest.mark.parametrize(
        "model_file",
        [
            pytest.param("/some/model.joblib", id="with_model_file"),
            pytest.param(None, id="without_model_file"),
        ],
    )
    def test_passes_current_env_values(
        self, patched_engine_collaborators: MagicMock, model_file: str | None
    ) -> None:
        """Engine reads current env values and passes them to phase.run() —
        always, whether or not a model_file is configured."""
        mock_env = make_mock_environment(
            variables={
                "var_a": MagicMock(
                    max_step=None, min_value=0.0, max_value=10.0, units="A", step_size=0.1
                ),
                "var_b": MagicMock(
                    max_step=None, min_value=-5.0, max_value=5.0, units="V", step_size=0.1
                ),
            },
        )
        mock_env.get_setpoints.return_value = {
            "var_a": PVReading(value=3.0),
            "var_b": PVReading(value=-1.5),
        }
        mock_factory = MagicMock(return_value=mock_env)

        config = OptimizationConfig(
            environment_name="test_env",
            control_target="va",
            objective_name="charge_ltb",
            variables=["var_a"],
            variable_overrides={"var_a": VariableOverride(min=0.0, max=10.0, step_size=0.1)},
            model_file=model_file,
        )

        engine = OptimizationEngine(config, environment_factory=mock_factory)

        mock_data = MagicMock()
        patched_engine_collaborators.return_value = PhaseResult(data=mock_data)
        engine.run()

        call_kwargs = patched_engine_collaborators.call_args.kwargs
        assert call_kwargs["current_env_values"] == {"var_a": 3.0, "var_b": -1.5}

    def test_model_variables_threaded_to_result(
        self, patched_engine_collaborators: MagicMock
    ) -> None:
        """model_variables from the optimizer flows into OptimizationResult."""
        mock_env = make_mock_environment()
        mock_factory = MagicMock(return_value=mock_env)

        config = OptimizationConfig(
            environment_name="test_env",
            control_target="va",
            objective_name="charge_ltb",
            variables=["test_var"],
            variable_overrides={"test_var": VariableOverride(min=0.0, max=1.0, step_size=0.1)},
        )

        engine = OptimizationEngine(config, environment_factory=mock_factory)

        mock_optimizer = MagicMock()
        mock_optimizer.resolve_n_initial.return_value = 1
        mock_optimizer.fixed_features = None
        mock_optimizer.trained_model.return_value = None
        mock_optimizer.model_variables.return_value = ["x", "y", "z"]

        with patch(
            "osprey_optimizer.services.orchestration.XoptOptimizer",
            return_value=mock_optimizer,
        ):
            mock_data = MagicMock()
            patched_engine_collaborators.return_value = PhaseResult(data=mock_data)
            result = engine.run()

        assert result.model_variables == ["x", "y", "z"]


class TestRunnerFixedFeatures:
    def test_fixed_features_computed_and_injected(self, tmp_path: Path) -> None:
        """A Bayesian gen with a model_file that has an extra variable computes
        fixed_features (exposed on the optimizer) and, threaded through
        static_columns, injects the fixed-feature column into the recorded data."""
        # Model trained on x + extra → fixed_features = {extra: current value}.
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 1, (8, 2))
        y = X[:, 0]
        rf = RandomForestRegressor(n_estimators=4, random_state=0)
        rf.fit(X, y)

        config = OptimizationConfig(
            algorithm=AlgorithmType.UPPER_CONFIDENCE_BOUND,
            n_iterations=0,
            n_initial_samples=2,
            objective_name="y",
            surrogate_model=SurrogateModelType.RANDOM_FOREST,
            model_file=_dump(tmp_path, {"model": rf, "variables": ["x", "extra"]}),
        )

        optimizer, result = run_phase_with_optimizer(
            bounds={"x": [0.0, 1.0]},
            objective_name="y",
            config=config,
            current_env_values={"x": 0.5, "extra": 0.7},
            env=EchoEnv({"y": PVReading(value=1.0)}),
        )
        assert optimizer.fixed_features == {"extra": 0.7}
        assert "extra" in result.data.columns
        assert all(v == pytest.approx(0.7) for v in result.data["extra"].to_list())

    def test_model_file_without_extra_vars_skips_injection(self, tmp_path: Path) -> None:
        """model_file present but the model covers exactly the VOCS variables →
        _compute_fixed_features returns None, so no column is injected."""
        rng = np.random.default_rng(1)
        X = rng.uniform(0, 1, (8, 1))
        y = X[:, 0]
        rf = RandomForestRegressor(n_estimators=4, random_state=1)
        rf.fit(X, y)

        config = OptimizationConfig(
            algorithm=AlgorithmType.UPPER_CONFIDENCE_BOUND,
            n_iterations=0,
            n_initial_samples=2,
            objective_name="y",
            surrogate_model=SurrogateModelType.RANDOM_FOREST,
            model_file=_dump(tmp_path, {"model": rf, "variables": ["x"]}),
        )
        optimizer, result = run_phase_with_optimizer(
            bounds={"x": [0.0, 1.0]},
            objective_name="y",
            config=config,
            current_env_values={"x": 0.5},
            env=EchoEnv({"y": PVReading(value=1.0)}),
        )
        assert optimizer.fixed_features is None
        # No extra model variable → nothing injected beyond x.
        assert "extra" not in result.data.columns
