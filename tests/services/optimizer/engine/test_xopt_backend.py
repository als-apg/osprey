"""Tests for the Xopt implementation of OptimizerProtocol.

Covers generator resolution (custom generators, LHS→EXPLORE rewrite, sklearn
surrogate wiring), trained-model extraction, and the XoptOptimizer ask/tell
adapter itself — confirming real xopt generators (Random, LHS, UCB, NSGA2)
drive correctly through ``suggest`` / ``ingest``.
"""

from __future__ import annotations

import logging
import operator
import warnings
from collections.abc import Callable
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
from pydantic import ValidationError
from xopt import VOCS
from xopt.errors import DataError, GeneratorWarning, XoptError
from xopt.generators.bayesian.bayesian_generator import BayesianGenerator
from xopt.generators.random import RandomGenerator

from osprey.services.optimizer.models.config import ConstraintSpec, OptimizationConfig
from osprey.services.optimizer.models.enums import AlgorithmType, ConstraintType, SurrogateModelType
from osprey.services.optimizer.models.exceptions import InvalidSuggestionError, OptimizerError
from osprey.services.optimizer.models.protocols import OptimizerProtocol
from osprey.services.optimizer.models.types import Direction
from osprey.services.optimizer.optimization.generators.baxus import BAxUSGenerator
from osprey.services.optimizer.optimization.surrogate.wrapper import SklearnBoTorchModel
from osprey.services.optimizer.optimization.xopt_backend import (
    XoptOptimizer,
    _extract_trained_model,
    _resolve_generator,
)


@pytest.fixture
def vocs() -> VOCS:
    return VOCS(variables={"x": [0.0, 1.0], "y": [0.0, 1.0]}, objectives={"f": "MAXIMIZE"})


def _config(
    algorithm: AlgorithmType,
    *,
    n_iterations: int = 1,
    n_initial_samples: int | None = None,
    objective_name: str = "f",
    algorithm_params: dict[str, Any] | None = None,
) -> OptimizationConfig:
    return OptimizationConfig(
        algorithm=algorithm,
        n_iterations=n_iterations,
        n_initial_samples=n_initial_samples,
        objective_name=objective_name,
        algorithm_params=algorithm_params,
    )


# the one outcome constraint every constrained test uses (read-only: the backend
# stores it by reference and never mutates it), and a row that violates it
_LOSS_CONSTRAINT = {"loss": ConstraintSpec(type=ConstraintType.LESS_THAN, value=0.5)}
_INFEASIBLE_ROW = {"x": 0.1, "y": 0.1, "f": 1.0, "loss": 0.9}


def _optimizer(
    algorithm: AlgorithmType,
    *,
    constraints: dict[str, ConstraintSpec] | None = None,
    **cfg: object,
) -> XoptOptimizer:
    return XoptOptimizer(
        variables={"x": [0.0, 1.0], "y": [0.0, 1.0]},
        objective_name="f",
        direction=Direction.MAXIMIZE,
        config=_config(algorithm, **cfg),  # type: ignore[arg-type]
        current_env_values={"x": 0.5, "y": 0.5},
        constraints=constraints,
    )


class TestXoptOptimizerAskTell:
    def test_satisfies_protocol(self) -> None:
        assert isinstance(_optimizer(AlgorithmType.RANDOM), OptimizerProtocol)

    @pytest.mark.parametrize(
        "algorithm",
        [
            AlgorithmType.RANDOM,
            AlgorithmType.LATIN_HYPERCUBE,
            AlgorithmType.UPPER_CONFIDENCE_BOUND,
            AlgorithmType.NSGA2,
        ],
    )
    def test_suggest_ingest_cycle(self, algorithm: AlgorithmType) -> None:
        """Each generator drives through the ask/tell cycle: seed via ingest, then
        suggest → ingest for a few steps."""
        opt = _optimizer(algorithm)
        # Seed with the current state so Bayesian/NSGA2 have training data.
        opt.ingest([{"x": 0.5, "y": 0.5, "f": 1.0}])
        for _ in range(3):
            suggestions = opt.suggest()
            assert isinstance(suggestions, list)
            for s in suggestions:
                assert isinstance(s, dict)
                assert 0.0 <= s["x"] <= 1.0  # setpoints stay within the variable bounds
                assert 0.0 <= s["y"] <= 1.0
            # Feed back a finite objective (variable values echoed as the outcome).
            opt.ingest([{**s, "f": s["x"] + s["y"]} for s in suggestions])

    def test_nsga2_bookkeeping_keys_round_trip(self) -> None:
        """NSGA2 stamps xopt_* bookkeeping onto suggestions; ingesting them back
        (as the loop does) keeps the generator working and preserves the keys."""
        opt = _optimizer(AlgorithmType.NSGA2)
        opt.ingest([{"x": 0.5, "y": 0.5, "f": 1.0}])

        suggestions = opt.suggest()
        assert suggestions
        assert any(k.startswith("xopt_") for k in suggestions[0])

        # Round-trip the bookkeeping keys back untouched.
        opt.ingest([{**s, "f": 2.0} for s in suggestions])
        # A further suggest still works (generator consumed the fed-back candidates).
        assert opt.suggest()

    def test_resolve_n_initial_bayesian_auto(self) -> None:
        # 2 variables → max(2, n_vars + 1) = 3.
        assert _optimizer(AlgorithmType.UPPER_CONFIDENCE_BOUND).resolve_n_initial() == 3

    def test_trained_model_and_variables_none_for_random(self) -> None:
        opt = _optimizer(AlgorithmType.RANDOM)
        opt.ingest([{"x": 0.5, "y": 0.5, "f": 1.0}])
        assert opt.trained_model() is None
        assert opt.model_variables() is None

    def test_model_variables_captured_for_bayesian(self) -> None:
        opt = _optimizer(AlgorithmType.UPPER_CONFIDENCE_BOUND)
        opt.ingest(
            [
                {"x": 0.2, "y": 0.3, "f": 1.0},
                {"x": 0.6, "y": 0.7, "f": 2.0},
                {"x": 0.5, "y": 0.1, "f": 1.5},
            ]
        )
        opt.suggest()  # forces the GP to fit, populating model_input_names
        assert opt.model_variables() == ["x", "y"]

    def test_model_variables_none_for_baxus(self) -> None:
        """BAxUS trains in a target subspace; model_variables must stay None
        even if the generator happens to expose model_input_names."""
        opt = _optimizer(AlgorithmType.BAXUS)
        opt._generator = MagicMock(spec=BAxUSGenerator)
        opt._generator.model_input_names = ["z0", "z1"]
        assert opt.model_variables() is None


class TestSuggestValidation:
    """The generator is untyped (Any); ``suggest`` is the only seam between its
    output and hardware setpoints, so it must reject non-numeric / non-finite
    variable values and coerce numpy scalars to builtin floats."""

    def _stubbed(self, candidate: dict[str, Any]) -> XoptOptimizer:
        opt = _optimizer(AlgorithmType.RANDOM)
        opt._generator = MagicMock()
        opt._generator.generate.return_value = [candidate]
        return opt

    @pytest.mark.parametrize(
        "candidate",
        [
            {"x": float("nan"), "y": 0.5},
            {"x": float("inf"), "y": 0.5},
            {"x": float("-inf"), "y": 0.5},
            {"x": "1.5", "y": 0.5},
            {"x": None, "y": 0.5},
            {"x": True, "y": 0.5},
            {"y": 0.5},
        ],
        ids=["nan", "inf", "-inf", "string", "none", "bool", "missing"],
    )
    def test_unusable_variable_value_raises(self, candidate: dict[str, Any]) -> None:
        opt = self._stubbed(candidate)
        with pytest.raises(InvalidSuggestionError, match="'x'"):
            opt.suggest()

    def test_numpy_scalars_coerced_and_bookkeeping_untouched(self) -> None:
        """Variable values are coerced to builtin floats; xopt's own bookkeeping
        keys pass through unvalidated."""
        opt = self._stubbed(
            {"x": np.float32(0.25), "y": np.int64(1), "xopt_candidate_idx": "not-a-number"}
        )
        [suggestion] = opt.suggest()
        assert suggestion["x"] == 0.25
        assert suggestion["y"] == 1.0
        assert type(suggestion["x"]) is float
        assert type(suggestion["y"]) is float
        assert suggestion["xopt_candidate_idx"] == "not-a-number"

    def test_real_generator_suggestions_are_builtin_floats(self) -> None:
        opt = _optimizer(AlgorithmType.RANDOM)
        for suggestion in opt.suggest(3):
            assert all(type(suggestion[name]) is float for name in ("x", "y"))


class TestErrorTranslation:
    """xopt exception types must not cross the OptimizerProtocol seam: construction,
    suggest, and ingest translate them to ``OptimizerError``, message and cause kept."""

    def test_construction_failure_becomes_optimizer_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(_algorithm: object) -> None:
            raise XoptError("No generator named nope")

        monkeypatch.setattr(
            "osprey.services.optimizer.optimization.xopt_backend.get_generator", _boom
        )
        with pytest.raises(OptimizerError, match="No generator named nope") as exc_info:
            _optimizer(AlgorithmType.RANDOM)
        assert isinstance(exc_info.value.__cause__, XoptError)

    @pytest.mark.parametrize(
        ("method", "error", "call", "match"),
        [
            pytest.param(
                "generate",
                XoptError("reference point must contain all objective names"),
                lambda opt: opt.suggest(),
                "reference point",
                id="suggest",
            ),
            pytest.param(
                "add_data",
                DataError("bad data"),
                lambda opt: opt.ingest([{"x": 0.1, "y": 0.2, "f": 1.0}]),
                "bad data",
                id="ingest",
            ),
        ],
    )
    def test_generator_failure_becomes_optimizer_error(
        self, method: str, error: Exception, call: Callable[[XoptOptimizer], object], match: str
    ) -> None:
        opt = _optimizer(AlgorithmType.RANDOM)
        opt._generator = MagicMock()
        getattr(opt._generator, method).side_effect = error
        with pytest.raises(OptimizerError, match=match) as exc_info:
            call(opt)
        assert exc_info.value.__cause__ is error

    def test_non_xopt_generator_errors_pass_through(self) -> None:
        """Only the xopt family is translated — BAxUS's own NotImplementedError
        for multi-point suggest must reach the run loop unchanged."""
        opt = _optimizer(AlgorithmType.BAXUS)
        with pytest.raises(NotImplementedError):
            opt.suggest(2)


class TestXoptOptimizerConstraints:
    """Outcome constraints: VOCS wiring, unsupported-algorithm rejection, the UCB
    caveat warning, LHS+EXPLORE composition, and feasibility-aware best selection."""

    def test_constraints_wired_into_vocs(self) -> None:
        opt = _optimizer(AlgorithmType.RANDOM, constraints=_LOSS_CONSTRAINT)
        vocs = opt._vocs
        assert vocs.constraint_names == ["loss"]
        assert vocs.constraints["loss"].value == 0.5

    @pytest.mark.parametrize(
        "algorithm",
        [
            AlgorithmType.NELDER_MEAD,
            AlgorithmType.EXTREMUM_SEEKING,
            AlgorithmType.RCDS,
            AlgorithmType.BAXUS,
        ],
    )
    def test_unsupported_algorithm_rejected(self, algorithm: AlgorithmType) -> None:
        """Constraints + an algorithm that can't honor them fails with our operator
        message (naming the algorithm and the supported ones) — not xopt's VOCSError."""
        with pytest.raises(
            ValueError, match=f"'{algorithm.value}' does not support outcome constraints"
        ) as exc_info:
            _optimizer(algorithm, constraints=_LOSS_CONSTRAINT)
        # A couple of the supported algorithms are named so the operator can pick one.
        assert "random" in str(exc_info.value)
        assert "nsga2" in str(exc_info.value)
        assert "expected_improvement" in str(exc_info.value)

    @pytest.mark.parametrize(
        ("algorithm", "expect_warning"),
        [
            pytest.param(AlgorithmType.UPPER_CONFIDENCE_BOUND, True, id="ucb_warns"),
            pytest.param(AlgorithmType.EXPECTED_IMPROVEMENT, False, id="ei_clean"),
        ],
    )
    def test_constraint_caveat_warns_only_for_ucb(
        self, caplog: pytest.LogCaptureFixture, algorithm: AlgorithmType, expect_warning: bool
    ) -> None:
        """UCB carries the caveat warning; a non-UCB supported algorithm builds
        cleanly with no caveat warning."""
        with caplog.at_level(logging.WARNING):
            _optimizer(algorithm, constraints=_LOSS_CONSTRAINT)
        assert (
            any(
                "can misbehave" in rec.message and "expected_improvement" in rec.message
                for rec in caplog.records
            )
        ) is expect_warning

    def test_lhs_explore_composes_with_constraints(self) -> None:
        """The LHS→EXPLORE objective rewrite still accepts a constrained VOCS."""
        opt = _optimizer(AlgorithmType.LATIN_HYPERCUBE, constraints=_LOSS_CONSTRAINT)
        gen_vocs: VOCS = cast(Any, opt._generator).vocs
        assert [type(o).__name__ for o in gen_vocs.objectives.values()] == ["ExploreObjective"]
        assert gen_vocs.constraint_names == ["loss"]
        # The generator drives without raising over the constrained, rewritten VOCS.
        opt.ingest([{"x": 0.5, "y": 0.5, "f": 1.0, "loss": 0.2}])
        assert opt.suggest()


class TestXoptOptimizerFeasibilitySearch:
    """Constrained EI must not hard-fail when no ingested row is feasible: bypass the
    generator with space-filling random points (feasibility search) until one is,
    then resume the generator. Non-EI constrained algorithms are unaffected."""

    def _ei(self) -> XoptOptimizer:
        return _optimizer(AlgorithmType.EXPECTED_IMPROVEMENT, constraints=_LOSS_CONSTRAINT)

    @pytest.mark.parametrize(
        ("rows", "n_points"),
        [
            pytest.param([_INFEASIBLE_ROW], 1, id="violating_reading"),
            pytest.param(
                [{**_INFEASIBLE_ROW, "loss": float("nan")}], 1, id="unverifiable_nan_reading"
            ),
            pytest.param([], 1, id="empty_record"),
            pytest.param([_INFEASIBLE_ROW], 3, id="multiple_fallback_points"),
        ],
    )
    def test_ei_all_infeasible_suggests_random_within_bounds(
        self, caplog: pytest.LogCaptureFixture, rows: list[dict[str, float]], n_points: int
    ) -> None:
        """Reproduces the issue: constrained EI with only infeasible rows would raise
        xopt's "No feasible points found" — the gate returns bounded random points and
        logs the feasibility search instead. A NaN constraint reading can't be shown
        satisfied, so it counts as infeasible and keeps the search active too; an empty
        record counts as no-feasible for the same reason. Every fallback point is logged
        (no silent fallback), one record per point.
        """
        opt = self._ei()
        opt.ingest(rows)  # infeasible: loss !< 0.5
        with caplog.at_level(logging.WARNING):
            suggestions = opt.suggest(n_points)
        assert len(suggestions) == n_points
        for point in suggestions:
            assert set(point) == {"x", "y"}
            assert 0.0 <= point["x"] <= 1.0
            assert 0.0 <= point["y"] <= 1.0
        fallback_warnings = [
            rec
            for rec in caplog.records
            if "expected_improvement" in rec.message and "feasibility search" in rec.message.lower()
        ]
        assert len(fallback_warnings) == n_points

    def test_feasibility_search_points_are_validated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Feasibility-search points become hardware setpoints just like generator
        candidates, so they pass the same last-stop validation rather than being
        trusted because they were drawn locally — the invariant holds on both
        return paths of ``suggest``."""
        opt = self._ei()
        opt.ingest([_INFEASIBLE_ROW])
        monkeypatch.setattr(opt, "_random_within_bounds", lambda: {"x": float("nan"), "y": 0.5})
        with pytest.raises(InvalidSuggestionError, match="'x'"):
            opt.suggest()

    @pytest.mark.parametrize(
        ("algorithm", "constraints", "rows"),
        [
            # After a feasible row is ingested the generator path is used again.
            pytest.param(
                AlgorithmType.EXPECTED_IMPROVEMENT,
                _LOSS_CONSTRAINT,
                [
                    _INFEASIBLE_ROW,
                    {"x": 0.2, "y": 0.2, "f": 2.0, "loss": 0.2},
                ],  # infeasible, then feasible
                id="ei_resumes_after_feasible_row",
            ),
            # Algorithms that don't need a feasible incumbent are untouched: they
            # drive the generator normally even on all-infeasible data.
            pytest.param(
                AlgorithmType.RANDOM, _LOSS_CONSTRAINT, [_INFEASIBLE_ROW], id="random_unaffected"
            ),
            pytest.param(
                AlgorithmType.LATIN_HYPERCUBE,
                _LOSS_CONSTRAINT,
                [_INFEASIBLE_ROW],
                id="lhs_unaffected",
            ),
            pytest.param(
                AlgorithmType.UPPER_CONFIDENCE_BOUND,
                _LOSS_CONSTRAINT,
                [_INFEASIBLE_ROW],
                id="ucb_unaffected",
            ),
            # Without constraints the gate is inert — EI drives the generator as before.
            pytest.param(
                AlgorithmType.EXPECTED_IMPROVEMENT,
                None,
                [
                    {"x": 0.2, "y": 0.3, "f": 1.0},
                    {"x": 0.6, "y": 0.7, "f": 2.0},
                    {"x": 0.5, "y": 0.1, "f": 1.5},
                ],
                id="unconstrained_ei_inert",
            ),
        ],
    )
    def test_generator_drives_without_feasibility_search(
        self,
        caplog: pytest.LogCaptureFixture,
        algorithm: AlgorithmType,
        constraints: dict[str, ConstraintSpec] | None,
        rows: list[dict[str, float]],
    ) -> None:
        """Cases where the fallback must stay out of the way: the generator produces the
        suggestion and no feasibility-search warning is logged."""
        opt = _optimizer(algorithm, constraints=constraints)
        opt.ingest(rows)
        with caplog.at_level(logging.WARNING):
            suggestions = opt.suggest()
        assert suggestions
        assert not any("feasibility search" in r.message.lower() for r in caplog.records)


class TestBAxUSEvalBudgetWiring:
    """The backend aligns BAxUS's seed quota and eval_budget with the run shape."""

    @pytest.mark.parametrize(
        ("cfg_kwargs", "expected_eval_budget", "expected_n_initial_sobol"),
        [
            pytest.param(
                {"n_iterations": 30, "n_initial_samples": 5}, 35, 5, id="derived_from_config"
            ),
            # Bayesian default seeding is max(2, n_variables + 1) = 3 for 2 variables.
            pytest.param({"n_iterations": 10}, 13, 3, id="default_initial_samples_included"),
            pytest.param(
                {"n_iterations": 30, "algorithm_params": {"eval_budget": 7, "n_initial_sobol": 2}},
                7,
                2,
                id="explicit_algorithm_params_win",
            ),
        ],
    )
    def test_eval_budget_and_n_initial_sobol(
        self,
        cfg_kwargs: dict[str, Any],
        expected_eval_budget: int,
        expected_n_initial_sobol: int,
    ) -> None:
        """The runner always wires eval_budget right after construction, so the
        generator's missing-budget advisory must not escape to the job logs."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            opt = _optimizer(AlgorithmType.BAXUS, **cfg_kwargs)
        assert [w for w in caught if issubclass(w.category, GeneratorWarning)] == []
        assert opt._generator.eval_budget == expected_eval_budget
        assert opt._generator.n_initial_sobol == expected_n_initial_sobol

    def test_other_bayesian_generators_unaffected(self) -> None:
        opt = _optimizer(AlgorithmType.UPPER_CONFIDENCE_BOUND, n_iterations=30)
        assert not hasattr(opt._generator, "eval_budget")

    def test_explicit_budget_below_seeding_fails_at_job_creation(self) -> None:
        """An explicit eval_budget the run's seeding alone would exceed is
        contradictory — rejected loudly when the backend aligns the quota."""
        with pytest.raises(ValidationError, match="must cover the seed quota"):
            _optimizer(
                AlgorithmType.BAXUS,
                n_iterations=30,
                n_initial_samples=17,
                algorithm_params={"eval_budget": 7},
            )


class TestResolveGenerator:
    @pytest.mark.parametrize(
        ("algorithm", "algorithm_params", "expected_type", "attr", "expected"),
        [
            pytest.param(
                AlgorithmType.BAXUS,
                {"target_dim_init": 1},
                BAxUSGenerator,
                operator.attrgetter("embedding.target_dim"),
                1,
                id="baxus_target_dim",
            ),
            pytest.param(
                AlgorithmType.UPPER_CONFIDENCE_BOUND,
                {"beta": 0.1},
                BayesianGenerator,
                operator.attrgetter("beta"),
                0.1,
                id="ucb_beta",
            ),
            pytest.param(
                AlgorithmType.BAXUS,
                None,
                BAxUSGenerator,
                operator.attrgetter("target_dim_init"),
                2,  # default
                id="none_uses_defaults",
            ),
            pytest.param(
                AlgorithmType.RANDOM,
                None,
                RandomGenerator,
                operator.attrgetter("name"),
                "random",
                id="builtin_random",
            ),
        ],
    )
    def test_algorithm_params_applied(
        self,
        vocs: VOCS,
        algorithm: AlgorithmType,
        algorithm_params: dict[str, Any] | None,
        expected_type: type,
        attr: Callable[[Any], Any],
        expected: float,
    ) -> None:
        gen = _resolve_generator(algorithm, vocs, algorithm_params=algorithm_params)
        assert isinstance(gen, expected_type)
        assert attr(gen) == expected

    def test_algorithm_params_invalid_raises(self, vocs: VOCS) -> None:
        with pytest.raises(ValidationError):
            _resolve_generator(
                AlgorithmType.BAXUS, vocs, algorithm_params={"not_a_real_field": 999}
            )

    @pytest.mark.parametrize(
        ("algorithm", "expected_type", "surrogate_model_type", "model_file"),
        [
            pytest.param(
                AlgorithmType.UPPER_CONFIDENCE_BOUND,
                BayesianGenerator,
                SurrogateModelType.RANDOM_FOREST,
                None,
                id="bayesian",
            ),
            pytest.param(
                AlgorithmType.BAXUS,
                BAxUSGenerator,
                SurrogateModelType.EXTRA_TREES,
                None,
                id="baxus",
            ),
            pytest.param(
                AlgorithmType.UPPER_CONFIDENCE_BOUND,
                BayesianGenerator,
                SurrogateModelType.RANDOM_FOREST,
                "fake_model.joblib",
                id="with_model_file",
            ),
        ],
    )
    def test_sklearn_surrogate_attached(
        self,
        vocs: VOCS,
        algorithm: AlgorithmType,
        expected_type: type[BayesianGenerator],
        surrogate_model_type: SurrogateModelType,
        model_file: str | None,
    ) -> None:
        gen = _resolve_generator(
            algorithm, vocs, surrogate_model=surrogate_model_type, model_file=model_file
        )
        assert isinstance(gen, expected_type)
        assert gen.gp_constructor is not None
        assert gen.gp_constructor.name == "sklearn"
        assert gen.gp_constructor.surrogate_model_type == surrogate_model_type
        assert gen.gp_constructor.model_file == model_file

    @pytest.mark.parametrize(
        "surrogate_model", [SurrogateModelType.DEFAULT, SurrogateModelType.XOPT_GP]
    )
    def test_builtin_surrogate_skips_constructor(
        self, vocs: VOCS, surrogate_model: SurrogateModelType
    ) -> None:
        gen = _resolve_generator(
            AlgorithmType.UPPER_CONFIDENCE_BOUND, vocs, surrogate_model=surrogate_model
        )
        assert isinstance(gen, BayesianGenerator)
        assert gen.gp_constructor.name != "sklearn"

    def test_non_bayesian_algorithm_warns(
        self, vocs: VOCS, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            gen = _resolve_generator(
                AlgorithmType.NELDER_MEAD, vocs, surrogate_model=SurrogateModelType.RANDOM_FOREST
            )
        # The requested generator is still built; the surrogate is genuinely dropped.
        assert type(gen).__name__ == "NelderMeadGenerator"
        assert not hasattr(gen, "gp_constructor")
        assert any("ignored for non-Bayesian" in rec.message for rec in caplog.records)

    def test_lhs_objective_rewritten_to_explore(self, vocs: VOCS) -> None:
        """LatinHypercubeGenerator rejects optimizing objectives; the resolver swaps
        them to EXPLORE so it accepts the same VOCS."""
        gen = _resolve_generator(AlgorithmType.LATIN_HYPERCUBE, vocs)
        gen_vocs: VOCS = cast(Any, gen).vocs
        # xopt materializes "EXPLORE" as an ExploreObjective instance.
        assert [type(o).__name__ for o in gen_vocs.objectives.values()] == ["ExploreObjective"]


class TestExtractTrainedModel:
    def test_returns_none_for_non_bayesian(self, vocs: VOCS) -> None:
        gen = _resolve_generator(AlgorithmType.RANDOM, vocs)
        assert _extract_trained_model(gen) is None

    def test_returns_none_when_model_not_set(self, vocs: VOCS) -> None:
        gen = _resolve_generator(AlgorithmType.UPPER_CONFIDENCE_BOUND, vocs)
        assert isinstance(gen, BayesianGenerator)
        assert _extract_trained_model(gen) is None

    def test_returns_none_for_baxus_even_with_trained_model(self) -> None:
        """BAxUS trains in a target subspace (z0..zk), not vocs space; exporting
        its model here would mislabel target-space features as input-space ones,
        so it must stay None even when a model is present."""
        gen = MagicMock(spec=BAxUSGenerator)
        gen.model = MagicMock()
        assert _extract_trained_model(gen) is None

    def test_extracts_sklearn_model_from_wrapper(self) -> None:
        fake_sklearn = MagicMock()
        gen = MagicMock(spec=BayesianGenerator)
        gen.model = SklearnBoTorchModel(fake_sklearn)
        assert _extract_trained_model(gen) is fake_sklearn

    def test_extracts_from_model_list(self) -> None:
        fake_sklearn = MagicMock()
        wrapper = SklearnBoTorchModel(fake_sklearn)
        model_list = MagicMock()
        model_list.models = [wrapper]
        gen = MagicMock(spec=BayesianGenerator)
        gen.model = model_list
        assert _extract_trained_model(gen) is fake_sklearn

    def test_returns_raw_botorch_model_when_no_sklearn(self) -> None:
        raw_model = MagicMock()
        raw_model.models = None
        gen = MagicMock(spec=BayesianGenerator)
        gen.model = raw_model
        assert _extract_trained_model(gen) is raw_model
