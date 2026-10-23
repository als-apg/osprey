"""Xopt implementation of :class:`OptimizerProtocol`.

All xopt-specific concerns live here, behind the backend-agnostic ask/tell
seam: the optimizer never sets variables, reads hardware, or raises
evaluation errors (snap-to-grid, readback verification, and NaN policy stay
in the ``Evaluator`` and run loop). This is also the only module that names
xopt exception types — backend failures are translated to ``OptimizerError``
so nothing above the seam imports xopt.
"""

import logging
import random
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, TypeGuard

import pandas as pd
import polars as pl
from xopt.errors import (
    DataError,
    FeasibilityError,
    GeneratorError,
    GeneratorWarning,
    VOCSError,
    XoptError,
)
from xopt.generator import Generator
from xopt.generators import get_generator
from xopt.generators.bayesian.bayesian_generator import BayesianGenerator
from xopt.vocs import VOCS

from osprey_optimizer.models.config import ConstraintSpec, OptimizationConfig
from osprey_optimizer.models.enums import (
    CONSTRAINT_UNSUPPORTED_ALGORITHMS,
    AlgorithmType,
    SurrogateModelType,
)
from osprey_optimizer.models.exceptions import InvalidSuggestionError, OptimizerError
from osprey_optimizer.models.types import Direction, VariableBounds

from ..utils.numeric import is_usable_setpoint_number
from .best_point import feasible_filter
from .generators import BAxUSGenerator
from .surrogate.constructor import SklearnModelConstructor, load_model_metadata

logger = logging.getLogger(__name__)


_CUSTOM_GENERATORS: dict[AlgorithmType, type[Generator]] = {
    AlgorithmType.BAXUS: BAxUSGenerator,
}

# Listed in full because xopt's error classes share no common base. Not a
# blanket ``except Exception``: xopt also raises plain builtins, which must
# propagate as themselves rather than be relabelled as backend failures.
_XOPT_ERRORS = (XoptError, GeneratorError, VOCSError, DataError, FeasibilityError)


@contextmanager
def _translate_xopt_errors() -> Iterator[None]:
    """Re-raise xopt-specific failures as ``OptimizerError``."""
    try:
        yield
    except _XOPT_ERRORS as e:
        raise OptimizerError(f"Optimizer backend failed: {e}") from e


# Algorithms that need a feasible incumbent to score candidates (xopt's EI
# raises "No feasible points found in the data" otherwise). The others in
# ``_CONSTRAINT_SUPPORTED`` weight infeasible points down but still generate
# (verified against xopt 3.0.0), so only these need the feasibility-search
# fallback.
_REQUIRES_FEASIBLE_INCUMBENT: frozenset[AlgorithmType] = frozenset(
    {AlgorithmType.EXPECTED_IMPROVEMENT}
)

# Algorithms that accept and honor outcome constraints; named in the
# rejection message so the operator can pick a working algorithm.
_CONSTRAINT_SUPPORTED: tuple[AlgorithmType, ...] = (
    AlgorithmType.RANDOM,
    AlgorithmType.LATIN_HYPERCUBE,
    AlgorithmType.UPPER_CONFIDENCE_BOUND,
    AlgorithmType.EXPECTED_IMPROVEMENT,
    AlgorithmType.NSGA2,
    AlgorithmType.CNSGA,
)


def _check_constraint_support(algorithm: AlgorithmType) -> None:
    """Reject algorithms that cannot honor outcome constraints; warn for UCB.

    Raises before generator construction, so no hardware is touched. UCB works
    but can misbehave when the acquisition goes negative under constraints
    (xopt's own caveat, mirrored here).
    """
    # CONSTRAINT_UNSUPPORTED_ALGORITHMS is sourced from osprey_optimizer.models
    # so the web UI blocks the same set before submit.
    if algorithm in CONSTRAINT_UNSUPPORTED_ALGORITHMS:
        supported = ", ".join(a.value for a in _CONSTRAINT_SUPPORTED)
        raise ValueError(
            f"Algorithm '{algorithm.value}' does not support outcome constraints. Choose one of: {supported}."
        )
    if algorithm is AlgorithmType.UPPER_CONFIDENCE_BOUND:
        logger.warning(
            "Algorithm '%s' with outcome constraints can misbehave when the acquisition "
            "function goes negative; prefer '%s' for constrained optimization.",
            algorithm.value,
            AlgorithmType.EXPECTED_IMPROVEMENT.value,
        )


def _vocs_for_algorithm(vocs: VOCS, algorithm: AlgorithmType) -> VOCS:
    # LatinHypercubeGenerator is a pure space-filling sampler and rejects optimizing
    # objectives (MINIMIZE/MAXIMIZE) in xopt>=3.0; swap to EXPLORE so it accepts them.
    if algorithm is not AlgorithmType.LATIN_HYPERCUBE:
        return vocs
    new_vocs = vocs.model_copy()
    new_vocs.objectives = dict.fromkeys(vocs.objectives, "EXPLORE")
    return new_vocs


def _resolve_generator(
    algorithm: AlgorithmType,
    vocs: VOCS,
    surrogate_model: SurrogateModelType = SurrogateModelType.DEFAULT,
    model_file: str | None = None,
    algorithm_params: dict[str, Any] | None = None,
) -> Generator:
    """Create a generator by name, checking custom generators first.

    When *surrogate_model* is not DEFAULT/XOPT_GP, builds an
    ``SklearnModelConstructor`` and attaches it to Bayesian generators
    (BAxUS included).
    """
    extra = algorithm_params or {}
    if algorithm in _CUSTOM_GENERATORS:
        # The runner wires eval_budget immediately after construction (see
        # __init__), so BAxUS's missing-budget advisory is always spurious here.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="BAxUS: eval_budget is not set",
                category=GeneratorWarning,
            )
            gen = _CUSTOM_GENERATORS[algorithm](vocs=vocs, **extra)
    else:
        gen_vocs = _vocs_for_algorithm(vocs, algorithm)
        gen = get_generator(algorithm)(vocs=gen_vocs, **extra)

    if surrogate_model in (SurrogateModelType.DEFAULT, SurrogateModelType.XOPT_GP):
        return gen

    constructor = SklearnModelConstructor(
        surrogate_model_type=surrogate_model,
        model_file=model_file,
    )

    if isinstance(gen, BayesianGenerator):
        gen.gp_constructor = constructor
    else:
        logger.warning(
            "surrogate_model=%s ignored for non-Bayesian algorithm '%s'",
            surrogate_model.value,
            algorithm,
        )

    return gen


def _resolve_n_initial(
    n_initial_samples: int | None,
    generator: Generator,
    n_variables: int,
    model_file: str | None = None,
) -> int:
    """Determine the number of initial random samples for a given algorithm/generator.

    Every algorithm gets at least 1 initial sample (the current machine state).
    """
    if isinstance(generator, BayesianGenerator) and model_file:
        if n_initial_samples is not None and n_initial_samples > 1:
            logger.warning(
                "n_initial_samples=%d with a pre-trained model — "
                "the model already provides prior knowledge, consider using n_initial_samples=1",
                n_initial_samples,
            )
        return n_initial_samples if n_initial_samples is not None else 1
    if n_initial_samples is not None:
        return max(1, n_initial_samples)
    if isinstance(generator, BayesianGenerator):
        return max(2, n_variables + 1)
    return 1


def _compute_fixed_features(
    model_file: str,
    vocs: VOCS,
    current_env_values: dict[str, float],
) -> dict[str, float] | None:
    """Compute fixed_features for extra model variables not in the current VOCS.

    Returns a dict mapping extra variable names to their current environment
    values, or ``None`` if there are no extra variables. Raises ``ValueError``
    when the model and the available values cannot be reconciled — a
    pre-trained surrogate must never run on an incomplete feature set.
    """
    model_vars, _ = load_model_metadata(model_file)

    vocs_vars = set(vocs.variable_names)
    model_var_set = set(model_vars)

    missing_vars = vocs_vars - model_var_set
    if missing_vars:
        raise ValueError(
            f"Model was trained on {sorted(model_var_set)} but VOCS requires {sorted(vocs_vars)}. "
            f"The model does not cover variables: {sorted(missing_vars)}. "
            f"Select only variables the model was trained on, or use a different model."
        )

    extra_vars = [v for v in model_vars if v not in vocs_vars]
    if not extra_vars:
        return None

    missing = [v for v in extra_vars if v not in current_env_values]
    if missing:
        raise ValueError(
            f"Model was trained on variables {sorted(missing)} that are not available in the "
            f"current environment values, so they cannot be fixed for the surrogate. "
            f"Use a different model, or an environment that exposes these variables."
        )

    fixed = {v: current_env_values[v] for v in extra_vars}
    logger.info("Auto-fixing extra model variables: %s", fixed)
    return fixed


def _is_exportable_bayesian(generator: Generator) -> TypeGuard[BayesianGenerator]:
    """True for Bayesian generators whose surrogate lives in vocs space.

    BAxUS trains in a target subspace (z0..zk), not vocs space; exporting
    would mislabel target-space features with input-space names.
    """
    return not isinstance(generator, BAxUSGenerator) and isinstance(generator, BayesianGenerator)


def _extract_trained_model(generator: Generator) -> Any | None:
    """Extract the trained surrogate model from the generator after optimization.

    For Bayesian generators with ``SklearnBoTorchModel`` wrappers, extracts the
    raw sklearn estimator (compatible with ``model_file`` reload).  Otherwise
    returns the BoTorch model as-is.
    """
    from .surrogate.wrapper import SklearnBoTorchModel

    if not _is_exportable_bayesian(generator):
        return None

    model = getattr(generator, "model", None)
    if model is None:
        return None

    if isinstance(model, SklearnBoTorchModel):
        return model.sklearn_model

    # ModelList wrappers (multi-output): check each sub-model.
    sub_models = getattr(model, "models", None)
    if sub_models:
        sklearn_models = [m.sklearn_model for m in sub_models if isinstance(m, SklearnBoTorchModel)]
        if sklearn_models:
            return sklearn_models[0] if len(sklearn_models) == 1 else sklearn_models

    return model


class XoptOptimizer:
    """Xopt-backed :class:`OptimizerProtocol` implementation.

    ``suggest``/``ingest`` map onto the generator's ``generate``/``add_data``.
    The non-protocol members expose the wiring the run loop and orchestration
    need.
    """

    def __init__(
        self,
        *,
        variables: VariableBounds,
        objective_name: str,
        direction: Direction,
        config: OptimizationConfig,
        current_env_values: dict[str, float],
        constraints: dict[str, ConstraintSpec] | None = None,
    ) -> None:
        self._algorithm = config.algorithm
        self._n_initial_samples = config.n_initial_samples
        self._model_file = config.model_file
        self._constraints = constraints or {}

        # Fail fast, before touching hardware.
        if self._constraints:
            _check_constraint_support(config.algorithm)

        with _translate_xopt_errors():
            self._vocs = VOCS(
                variables=dict(variables),
                objectives={objective_name: direction},
                constraints={
                    name: [spec.type.value, spec.value] for name, spec in self._constraints.items()
                },
            )
            self._generator = _resolve_generator(
                config.algorithm,
                self._vocs,
                config.surrogate_model,
                config.model_file,
                config.algorithm_params,
            )

        # BAxUS paces its embedding expansions against the run's total
        # evaluation budget (seeding + iterations); an explicit
        # algorithm_params eval_budget wins. The generator derives its failure
        # tolerance on demand, so post-construction assignment is honored.
        if isinstance(self._generator, BAxUSGenerator):
            n_seed = self.resolve_n_initial()
            # Falsy (absent or 0) means "auto", mirroring the field's own
            # semantics — only a real explicit quota blocks the alignment.
            if not (config.algorithm_params or {}).get("n_initial_sobol"):
                # Align the generator's seed quota with the runner's actual
                # seeding phase: every runner-injected initial point must count
                # as seeding, not as a BO-phase trust-region update.
                self._generator.n_initial_sobol = n_seed
            if self._generator.eval_budget is None:
                self._generator.eval_budget = n_seed + config.n_iterations

        # Pre-trained model with extra variables: fix them at their current
        # values so the surrogate sees the full feature set it was trained on.
        self.fixed_features: dict[str, float] | None = None
        if config.model_file and isinstance(self._generator, BayesianGenerator):
            fixed = _compute_fixed_features(config.model_file, self._vocs, current_env_values)
            if fixed:
                self._generator.fixed_features = fixed
                self.fixed_features = fixed

        # Feeds _needs_feasibility_search only; the runner owns the full
        # evaluated-point record.
        self._has_feasible_point = False

    def suggest(self, num_points: int | None = None) -> list[dict[str, Any]]:
        """Ask the generator for the next candidate point(s).

        When the algorithm needs a feasible incumbent and no ingested row is
        feasible yet, the generator is bypassed and random points within
        bounds are returned instead (a feasibility search, always logged).
        """
        n = num_points or 1
        if self._needs_feasibility_search():
            return [self._validated_candidate(self._random_within_bounds()) for _ in range(n)]
        with _translate_xopt_errors():
            candidates = self._generator.generate(n)
        return [self._validated_candidate(dict(candidate)) for candidate in candidates]

    def _validated_candidate(self, candidate: dict[str, Any]) -> dict[str, Any]:
        """Coerce each variable value to a finite builtin float, in place.

        Last stop before a suggestion becomes a hardware setpoint: reject
        NaN/inf, non-numeric, and missing variable values. Non-variable
        bookkeeping keys (e.g. NSGA2's ``xopt_*``) pass through untouched.
        """
        for name in self._vocs.variables:
            value = candidate.get(name)
            if not is_usable_setpoint_number(value):
                raise InvalidSuggestionError(
                    f"Generator suggested an unusable value for variable '{name}': {value!r}"
                )
            candidate[name] = float(value)
        return candidate

    def _needs_feasibility_search(self) -> bool:
        """True when constraints are set, the algorithm requires a feasible incumbent, and no ingested row is feasible.

        Nothing ingested counts as no-feasible.
        """
        if not self._constraints or self._algorithm not in _REQUIRES_FEASIBLE_INCUMBENT:
            return False
        return not self._has_feasible_point

    def _random_within_bounds(self) -> dict[str, Any]:
        """A uniform space-filling point over the VOCS variable bounds, logged."""
        logger.warning(
            "Algorithm '%s' has no feasible point yet under the configured constraints; "
            "returning a random space-filling suggestion (feasibility search active).",
            self._algorithm.value,
        )
        # Not security-sensitive (S311).
        return {
            name: random.uniform(*self._vocs.variables[name].domain)
            for name in self._vocs.variable_names
        }

    def ingest(self, points: list[dict[str, Any]]) -> None:
        """Tell the generator the outcomes of evaluated point(s).

        Fixed-feature columns (if any) are appended to each row so the surrogate
        trains on the full feature set.
        """
        defaults = self.fixed_features or {}
        rows: list[dict[str, Any]] = [{**defaults, **point} for point in points]
        if self._constraints and not self._has_feasible_point and rows:
            feasible = pl.DataFrame(rows).filter(feasible_filter(self._constraints))
            self._has_feasible_point = not feasible.is_empty()
        with _translate_xopt_errors():
            self._generator.add_data(pd.DataFrame(rows))

    def resolve_n_initial(self) -> int:
        """Number of initial samples for this generator (see :func:`_resolve_n_initial`)."""
        return _resolve_n_initial(
            self._n_initial_samples,
            self._generator,
            len(self._vocs.variables),
            self._model_file,
        )

    def trained_model(self) -> Any | None:
        """The trained surrogate model, if the generator produced one."""
        return _extract_trained_model(self._generator)

    def model_variables(self) -> list[str] | None:
        """Input names the surrogate was trained on (Bayesian generators only)."""
        if _is_exportable_bayesian(self._generator):
            return list(self._generator.model_input_names)
        return None
