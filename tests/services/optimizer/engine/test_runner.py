"""Canonical home for optimization/runner.py — the backend-neutral phase loop.

Covers ``run_optimization_phase`` (seeding order, event emission, control
checkpoints, grid handling, snapped-inputs recording, bookkeeping-key filtering
and round-trip,
no-finite failure, abort paths) and ``report_results`` value selection.

The loop is driven two ways: with the real :class:`XoptOptimizer` +
:class:`Evaluator` over the fake injector env (integration realism — proves
xopt generators work under the ask/tell loop), and with a plain
``FakeOptimizer`` + fake ``evaluate`` (pure-loop unit coverage independent of
any backend).
"""

from __future__ import annotations

import logging
import math
from typing import Any

import polars as pl
import pytest

from osprey.services.optimizer.events import (
    EventBus,
    EventType,
    IterationCompletedData,
    OptimizationEvent,
    PhaseCompletedData,
    PhaseFailedData,
    PhaseStartedData,
)
from osprey.services.optimizer.models.config import ConstraintSpec, OptimizationConfig
from osprey.services.optimizer.models.enums import AlgorithmType, ConstraintType
from osprey.services.optimizer.models.exceptions import (
    CancellationError,
    ControlSystemUnreachableError,
    EvaluationError,
    WriteRefusedError,
)
from osprey.services.optimizer.models.protocols import PVReading
from osprey.services.optimizer.models.types import Direction, VariableBounds
from osprey.services.optimizer.optimization.best_point import best_feasible_row, build_best_point
from osprey.services.optimizer.optimization.evaluation import Evaluator as AppEvaluator
from osprey.services.optimizer.optimization.evaluation import VariableGrid
from osprey.services.optimizer.optimization.runner import (
    ControlContext,
    report_results,
    run_optimization_phase,
)
from osprey.services.optimizer.optimization.xopt_backend import XoptOptimizer
from osprey.services.optimizer.services.bounds_resolver import snap_value_to_grid

from ._fakes import FakeInjectorEnvironment
from .conftest import make_abort_phase, make_complete_phase

OBJECTIVE = "charge_ltb"


@pytest.fixture
def env() -> FakeInjectorEnvironment:
    return FakeInjectorEnvironment()


@pytest.fixture
def env_bounds(env: FakeInjectorEnvironment) -> VariableBounds:
    """Two-variable search space drawn from the fake env's real bounds."""
    all_vars = env.variables
    names = list(all_vars)[:2]
    return {n: [all_vars[n].min_value, all_vars[n].max_value] for n in names}


@pytest.fixture
def current_env_values(env_bounds: VariableBounds) -> dict[str, float]:
    """Mid-range values for all env_bounds variables."""
    return {name: (lo + hi) / 2 for name, (lo, hi) in env_bounds.items()}


@pytest.fixture
def event_collector() -> tuple[EventBus, list[OptimizationEvent]]:
    bus = EventBus()
    collected: list[OptimizationEvent] = []

    def _collect(event: OptimizationEvent) -> None:
        collected.append(event)

    for et in EventType:
        bus.subscribe(et, _collect)
    return bus, collected


def _config(
    algorithm: AlgorithmType = AlgorithmType.UPPER_CONFIDENCE_BOUND,
    n_iterations: int = 1,
    n_initial_samples: int | None = None,
    objective_name: str = OBJECTIVE,
) -> OptimizationConfig:
    return OptimizationConfig(
        algorithm=algorithm,
        n_iterations=n_iterations,
        n_initial_samples=n_initial_samples,
        objective_name=objective_name,
    )


def run_real_phase(
    env: FakeInjectorEnvironment,
    bounds: VariableBounds,
    current: dict[str, float],
    config: OptimizationConfig,
    *,
    context: ControlContext | None = None,
    direction: Direction = Direction.MAXIMIZE,
    objective_name: str = OBJECTIVE,
    grid: dict[str, VariableGrid] | None = None,
) -> Any:
    """Drive run_optimization_phase with the real XoptOptimizer + Evaluator,
    exactly as orchestration wires them."""
    evaluate = AppEvaluator(env, objective_name, grid=grid)
    optimizer = XoptOptimizer(
        variables=bounds,
        objective_name=objective_name,
        direction=direction,
        config=config,
        current_env_values=current,
    )
    return run_optimization_phase(
        optimizer=optimizer,
        evaluate=evaluate,
        bounds=bounds,
        objective_name=objective_name,
        direction=direction,
        config=config,
        context=context or ControlContext(),
        current_env_values=current,
        n_initial=optimizer.resolve_n_initial(),
        static_columns=optimizer.fixed_features,
    )


class FakeOptimizer:
    """OptimizerProtocol test double.

    Returns canned suggestions (one per ``suggest`` call, falling back to a
    fixed point once exhausted) and records every ingested point so tests can
    assert on what the loop told the optimizer.
    """

    def __init__(self, suggestions: list[dict[str, Any]] | None = None) -> None:
        self._suggestions = [dict(s) for s in (suggestions or [])]
        self._i = 0
        self.ingested: list[dict[str, Any]] = []

    def suggest(self, num_points: int | None = None) -> list[dict[str, Any]]:
        if self._i < len(self._suggestions):
            point = self._suggestions[self._i]
            self._i += 1
            return [dict(point)]
        return [{"x": 0.5}]

    def ingest(self, points: list[dict[str, Any]]) -> None:
        self.ingested.extend(dict(p) for p in points)


def _run_pure_loop(
    optimizer: Any,
    evaluate: Any,
    *,
    n_iterations: int = 1,
    n_initial: int = 1,
    n_initial_samples: int | None = None,
    bounds: VariableBounds | None = None,
    current: dict[str, float] | None = None,
    objective_name: str = "f",
    context: ControlContext | None = None,
    constraints: dict[str, ConstraintSpec] | None = None,
    static_columns: dict[str, float] | None = None,
) -> Any:
    """Drive run_optimization_phase over a test-double optimizer + fake ``evaluate``.

    Pure-loop counterpart of :func:`run_real_phase` (no backend, no hardware).
    Defaults are this file's recurring single-variable setup: ``x`` in [0, 1]
    with the machine currently at 0.5, MAXIMIZE-ing objective ``"f"``.
    """
    return run_optimization_phase(
        optimizer=optimizer,
        evaluate=evaluate,
        bounds={"x": [0.0, 1.0]} if bounds is None else bounds,
        objective_name=objective_name,
        direction=Direction.MAXIMIZE,
        config=_config(
            AlgorithmType.RANDOM,
            n_iterations=n_iterations,
            n_initial_samples=n_initial_samples,
            objective_name=objective_name,
        ),
        context=context or ControlContext(),
        current_env_values={"x": 0.5} if current is None else current,
        n_initial=n_initial,
        static_columns=static_columns,
        constraints=constraints,
    )


class TestNInitialSamplesResolution:
    """The real XoptOptimizer.resolve_n_initial drives the recorded row count."""

    @pytest.mark.parametrize(
        "algorithm,n_iterations,n_initial_samples,expected_initial",
        [
            (AlgorithmType.UPPER_CONFIDENCE_BOUND, 1, None, None),
            (AlgorithmType.NELDER_MEAD, 2, None, 1),
            (AlgorithmType.RANDOM, 3, None, 1),
            (AlgorithmType.LATIN_HYPERCUBE, 3, None, 1),
            (AlgorithmType.UPPER_CONFIDENCE_BOUND, 1, 5, 5),
            (AlgorithmType.RANDOM, 2, 3, 3),
        ],
        ids=[
            "bayesian_auto_computes",
            "sequential_defaults_to_one",
            "random_defaults_to_one",
            "lhs_defaults_to_one",
            "explicit_value_respected",
            "explicit_n_initial_respected",
        ],
    )
    def test_resolution(
        self,
        env: FakeInjectorEnvironment,
        env_bounds: VariableBounds,
        current_env_values: dict[str, float],
        algorithm: AlgorithmType,
        n_iterations: int,
        n_initial_samples: int | None,
        expected_initial: int | None,
    ) -> None:
        if expected_initial is None:
            expected_initial = max(2, len(env_bounds) + 1)
        result = run_real_phase(
            env,
            env_bounds,
            current_env_values,
            _config(
                algorithm=algorithm, n_iterations=n_iterations, n_initial_samples=n_initial_samples
            ),
        )
        assert len(result.data) == expected_initial + n_iterations


class TestEventBusEmissions:
    @pytest.mark.parametrize(
        ("algorithm", "n_iterations", "n_initial_samples", "expected_phase", "expected_n_samples"),
        [
            pytest.param(AlgorithmType.RANDOM, 2, None, "random", 3, id="random_default_seeding"),
            pytest.param(
                AlgorithmType.UPPER_CONFIDENCE_BOUND,
                1,
                None,
                "upper_confidence_bound",
                4,
                id="ucb_auto_seeding",
            ),
        ],
    )
    def test_phase_started_and_completed_emitted(
        self,
        env: FakeInjectorEnvironment,
        env_bounds: VariableBounds,
        current_env_values: dict[str, float],
        event_collector: tuple[EventBus, list[OptimizationEvent]],
        algorithm: AlgorithmType,
        n_iterations: int,
        n_initial_samples: int | None,
        expected_phase: str,
        expected_n_samples: int,
    ) -> None:
        """PHASE_STARTED fires once, naming the algorithm, with n_samples covering
        the auto-resolved seeding plus n_iterations; PHASE_COMPLETED then closes the
        phase once, carrying the best value found."""
        bus, events = event_collector
        run_real_phase(
            env,
            env_bounds,
            current_env_values,
            _config(
                algorithm=algorithm, n_iterations=n_iterations, n_initial_samples=n_initial_samples
            ),
            context=ControlContext(event_bus=bus),
        )
        started = [e for e in events if e.type == EventType.PHASE_STARTED]
        assert len(started) == 1
        assert isinstance(started[0].data, PhaseStartedData)
        assert started[0].data.phase == expected_phase
        assert started[0].data.n_samples == expected_n_samples

        completed = [e for e in events if e.type == EventType.PHASE_COMPLETED]
        assert len(completed) == 1
        assert isinstance(completed[0].data, PhaseCompletedData)
        assert completed[0].data.best_value is not None
        assert completed[0].data.best_value > 0

    def test_iteration_completed_per_sample_reports_running_best(
        self,
        env: FakeInjectorEnvironment,
        env_bounds: VariableBounds,
        current_env_values: dict[str, float],
        event_collector: tuple[EventBus, list[OptimizationEvent]],
    ) -> None:
        """Each initial sample emits its own ITERATION_COMPLETED so the interface
        updates incrementally during seeding instead of jumping at the end, and every
        event reports the real best-so-far (max of the MAXIMIZE objective over the data
        seen) rather than a 0.0 placeholder."""
        bus, events = event_collector
        n_initial = 4
        n_iter = 2
        run_real_phase(
            env,
            env_bounds,
            current_env_values,
            _config(
                algorithm=AlgorithmType.RANDOM, n_iterations=n_iter, n_initial_samples=n_initial
            ),
            context=ControlContext(event_bus=bus),
        )
        iters = [e for e in events if e.type == EventType.ITERATION_COMPLETED]
        assert len(iters) == n_initial + n_iter
        for e in iters:
            assert isinstance(e.data, IterationCompletedData)
            assert e.data.best_value == pytest.approx(e.data.data[OBJECTIVE].max())
        # Initial-phase events carry a growing dataset (1, 2, ..., n_initial rows),
        # proving the data is published after each evaluation rather than once.
        initial_events = iters[:n_initial]
        assert [len(e.data.data) for e in initial_events] == list(range(1, n_initial + 1))  # type: ignore[union-attr]

    def test_no_events_without_bus(
        self,
        env: FakeInjectorEnvironment,
        env_bounds: VariableBounds,
        current_env_values: dict[str, float],
    ) -> None:
        result = run_real_phase(
            env,
            env_bounds,
            current_env_values,
            _config(algorithm=AlgorithmType.RANDOM, n_iterations=2),
            context=ControlContext(event_bus=None),
        )
        assert isinstance(result.data, pl.DataFrame)
        assert len(result.data) == 3  # 1 initial + 2 iterations


class TestControlCheckpoints:
    """run_optimization_phase must invoke the pause/cancel checkpoints
    (handle_control_callbacks) once before initial seeding and once at the top
    of every iteration."""

    def test_checkpoints_fire_in_real_loop(
        self,
        env: FakeInjectorEnvironment,
        env_bounds: VariableBounds,
        current_env_values: dict[str, float],
    ) -> None:
        cc_calls: list[int] = []
        wp_calls: list[int] = []
        ctx = ControlContext(
            check_cancellation=lambda: cc_calls.append(1),
            wait_if_paused=lambda: wp_calls.append(1),
        )
        n_iter = 2
        run_real_phase(
            env,
            env_bounds,
            current_env_values,
            _config(algorithm=AlgorithmType.RANDOM, n_iterations=n_iter, n_initial_samples=1),
            context=ctx,
        )
        # Exactly 1 checkpoint before seeding + 1 at the top of each iteration —
        # no extra checkpoints, and none skipped.
        assert len(cc_calls) == n_iter + 1
        assert len(wp_calls) == n_iter + 1

    def test_cancellation_propagates_and_aborts_early(
        self,
        env: FakeInjectorEnvironment,
        env_bounds: VariableBounds,
        current_env_values: dict[str, float],
        event_collector: tuple[EventBus, list[OptimizationEvent]],
    ) -> None:
        """A CancellationError raised at the 2nd checkpoint propagates out of the
        loop AND stops it early — the remaining iterations never run."""
        bus, events = event_collector
        cc_calls = {"n": 0}

        def _cancel_on_second() -> None:
            cc_calls["n"] += 1
            if cc_calls["n"] == 2:
                raise CancellationError("cancelled")

        n_iter = 5
        ctx = ControlContext(event_bus=bus, check_cancellation=_cancel_on_second)
        with pytest.raises(CancellationError):
            run_real_phase(
                env,
                env_bounds,
                current_env_values,
                _config(algorithm=AlgorithmType.RANDOM, n_iterations=n_iter, n_initial_samples=1),
                context=ctx,
            )
        # Aborted at the 2nd checkpoint (top of iteration 0), before any step:
        # only the seeding sample produced an ITERATION_COMPLETED event, and no
        # PHASE_FAILED is emitted (cancellation is not a control-system fault).
        assert cc_calls["n"] == 2
        iters = [e for e in events if e.type == EventType.ITERATION_COMPLETED]
        assert len(iters) == 1
        assert [e for e in events if e.type == EventType.PHASE_FAILED] == []

    def test_pause_checkpoint_runs_before_step(
        self,
        env: FakeInjectorEnvironment,
        env_bounds: VariableBounds,
        current_env_values: dict[str, float],
        event_collector: tuple[EventBus, list[OptimizationEvent]],
    ) -> None:
        """wait_if_paused fires at the top of each iteration BEFORE the step adds
        the next row. Each call sees the pre-step evaluation count, proving the
        pause seam precedes the step rather than following it."""
        bus, events = event_collector
        counts_seen: list[int] = []

        def _record_pre_step_count() -> None:
            counts_seen.append(len([e for e in events if e.type == EventType.ITERATION_COMPLETED]))

        n_initial = 1
        n_iter = 2
        ctx = ControlContext(event_bus=bus, wait_if_paused=_record_pre_step_count)
        run_real_phase(
            env,
            env_bounds,
            current_env_values,
            _config(
                algorithm=AlgorithmType.RANDOM, n_iterations=n_iter, n_initial_samples=n_initial
            ),
            context=ctx,
        )
        # Before seeding: 0 events. Seeding emits n_initial. Each iteration's
        # checkpoint runs before its own step, so it sees exactly the prior count.
        assert counts_seen == [0, n_initial, n_initial + 1]


class TestSeeding:
    def test_first_seed_is_current_state_rest_random(self) -> None:
        opt = FakeOptimizer()
        seen: list[dict[str, float]] = []

        def evaluate(point: dict[str, float]) -> dict[str, float]:
            seen.append(dict(point))
            return {**point, "f": 1.0}

        result = _run_pure_loop(
            opt,
            evaluate,
            bounds={"x": [0.0, 1.0], "y": [10.0, 20.0]},
            current={"x": 0.5, "y": 15.0},
            n_iterations=0,
            n_initial_samples=4,
            n_initial=4,
        )
        assert len(result.data) == 4
        # The very first evaluated point is the current machine state.
        assert seen[0] == {"x": 0.5, "y": 15.0}
        # The remaining seeds are uniform-random within bounds.
        for pt in seen[1:]:
            assert 0.0 <= pt["x"] <= 1.0
            assert 10.0 <= pt["y"] <= 20.0

    @pytest.mark.parametrize(
        ("current_env_values", "match"),
        [
            pytest.param({}, "missing variables", id="missing"),
            pytest.param({"x": 5.0}, "outside bounds", id="out_of_bounds"),
        ],
    )
    def test_unusable_current_env_value_raises(
        self, current_env_values: dict[str, float], match: str
    ) -> None:
        """Seeding refuses to start from a state it cannot trust: a variable with no
        reading, or one already outside the resolved bounds."""
        with pytest.raises(ValueError, match=match):
            _run_pure_loop(FakeOptimizer(), lambda p: {**p, "f": 1.0}, current=current_env_values)

    def test_zero_initial_samples_skips_seeding(self) -> None:
        """n_initial == 0 bypasses the seeding block entirely; only the
        optimization steps are recorded and the current state is never read."""
        opt = FakeOptimizer(suggestions=[{"x": 0.8}, {"x": 0.9}])
        seen: list[dict[str, float]] = []

        def evaluate(point: dict[str, float]) -> dict[str, float]:
            seen.append(dict(point))
            return {**point, "f": 1.0}

        result = _run_pure_loop(opt, evaluate, n_iterations=2, n_initial=0)
        assert len(result.data) == 2
        # Only the two suggested points were evaluated; the current state was not.
        assert seen == [{"x": 0.8}, {"x": 0.9}]


class TestLoopRecordSemantics:
    def test_snapped_inputs_are_recorded_not_proposed(self) -> None:
        """The loop records the inputs the evaluator returns (on-grid), not the
        proposed off-grid values."""
        opt = FakeOptimizer(suggestions=[{"x": 0.26}])

        def evaluate(point: dict[str, float]) -> dict[str, float]:
            # Emulate the app evaluator snapping 0.26 -> 0.5 before returning.
            return {"x": 0.5, "f": 1.0}

        result = _run_pure_loop(opt, evaluate, n_initial_samples=1, current={"x": 0.3})
        # Seed and iteration both recorded the snapped value the evaluator applied.
        assert result.data["x"].to_list() == [0.5, 0.5]
        # The optimizer was told the snapped input too (outcome wins the merge).
        assert opt.ingested[-1]["x"] == 0.5

    @pytest.mark.parametrize(
        ("suggestion", "ingest_key", "ingest_value", "recorded_col"),
        [
            pytest.param(
                {"x": 0.7, "xopt_candidate_idx": 3},
                "xopt_candidate_idx",
                3,
                "xopt_candidate_idx",
                id="xopt_prefixed",
            ),
            pytest.param(
                {"x": 0.7, "trial_index": 4, "arm_name": 1},
                "trial_index",
                4,
                "arm_name",
                id="ax_style_no_prefix",
            ),
        ],
    )
    def test_bookkeeping_kept_from_evaluate_but_survives_to_ingest_and_record(
        self,
        suggestion: dict[str, Any],
        ingest_key: str,
        ingest_value: int,
        recorded_col: str,
    ) -> None:
        """Canonical home for the bookkeeping contract: a backend key never reaches
        the evaluator, yet round-trips into ingest and the persisted record. The
        filter is by setpoint name, not by key convention, so an Ax-style key with
        no ``xopt_`` prefix is kept from hardware just the same."""
        opt = FakeOptimizer(suggestions=[suggestion])
        seen: list[dict[str, float]] = []

        def evaluate(point: dict[str, float]) -> dict[str, float]:
            seen.append(dict(point))
            return {**point, "f": 2.0}

        result = _run_pure_loop(opt, evaluate, n_initial_samples=1)
        # The evaluator only ever saw real setpoints — no bookkeeping key.
        assert seen and all(set(s) == {"x"} for s in seen)
        # ingest for the iteration carried the bookkeeping key untouched.
        assert opt.ingested[-1][ingest_key] == ingest_value
        # ...and it survives into the recorded frame.
        assert recorded_col in result.data.columns

    def test_bookkeeping_column_first_seen_after_100_rows_survives(
        self, event_collector: tuple[EventBus, list[OptimizationEvent]]
    ) -> None:
        """A column that first appears late in the record must still reach the frame.

        polars infers the schema from a bounded prefix of the records by default
        (100 rows), dropping keys it never saw there. Bookkeeping columns appear
        only once the optimizer starts suggesting, so any run seeded with more
        than 100 initial samples used to lose them from the published and
        persisted frames.
        """
        bus, events = event_collector
        n_initial = 101
        opt = FakeOptimizer(suggestions=[{"x": 0.7, "xopt_candidate_idx": 3}])

        result = _run_pure_loop(
            opt,
            lambda p: {**p, "f": 1.0},
            n_iterations=1,
            n_initial=n_initial,
            n_initial_samples=n_initial,
            context=ControlContext(event_bus=bus),
        )

        assert result.data.height == n_initial + 1
        assert result.data["xopt_candidate_idx"].to_list() == [None] * n_initial + [3]
        # The interface renders the published frame, so it must carry it too.
        last_published = [e for e in events if e.type == EventType.ITERATION_COMPLETED][-1]
        assert isinstance(last_published.data, IterationCompletedData)
        assert "xopt_candidate_idx" in last_published.data.data.columns

    def test_fixed_feature_columns_reach_the_evaluator_and_every_record_row(self) -> None:
        """Fixed-feature columns are real setpoints — a pre-trained surrogate assumes
        the machine is held at those values — so they must NOT be filtered out, and
        static_columns (fixed-feature columns) land on every recorded row."""
        opt = FakeOptimizer(suggestions=[{"x": 0.7, "extra": 0.42}])
        seen: list[dict[str, float]] = []

        def evaluate(point: dict[str, float]) -> dict[str, float]:
            seen.append(dict(point))
            return {**point, "f": 2.0}

        result = _run_pure_loop(opt, evaluate, n_initial_samples=1, static_columns={"extra": 0.42})
        assert seen[-1] == {"x": 0.7, "extra": 0.42}
        assert result.data["extra"].to_list() == [pytest.approx(0.42), pytest.approx(0.42)]

    def test_no_finite_objective_raises_evaluation_error(self) -> None:
        opt = FakeOptimizer()
        bus = EventBus()
        emitted: list[OptimizationEvent] = []
        bus.subscribe(EventType.PHASE_FAILED, emitted.append)

        with pytest.raises(EvaluationError, match="no finite"):
            _run_pure_loop(
                opt,
                lambda p: {**p, "f": math.nan},
                n_iterations=2,
                n_initial_samples=1,
                context=ControlContext(event_bus=bus),
            )
        assert len(emitted) == 1
        assert isinstance(emitted[0].data, PhaseFailedData)


class TestGridSnappingEndToEnd:
    """A gridded variable's recorded values are all on its step grid after a run."""

    @pytest.mark.parametrize(
        ("algorithm", "n_iterations", "n_initial_samples", "n_gridded", "step_divisor"),
        [
            # Coarse grid over one variable (6 points), so snapping is plainly observable.
            pytest.param(AlgorithmType.RANDOM, 3, 2, 1, 5.0, id="random_coarse_grid"),
            # A Bayesian (UCB) run over gridded variables must complete and stay on the
            # grid, proving post-hoc snapping does not wedge xopt's BO acquisition under
            # ask/tell. ~21 points: snapping active but not degenerate.
            pytest.param(
                AlgorithmType.UPPER_CONFIDENCE_BOUND, 4, 3, 2, 20.0, id="bayesian_fine_grid"
            ),
        ],
    )
    def test_recorded_values_land_on_grid(
        self,
        env: FakeInjectorEnvironment,
        algorithm: AlgorithmType,
        n_iterations: int,
        n_initial_samples: int,
        n_gridded: int,
        step_divisor: float,
    ) -> None:
        all_vars = env.variables
        names = list(all_vars.keys())[:2]
        bounds: VariableBounds = {n: [all_vars[n].min_value, all_vars[n].max_value] for n in names}
        grid: dict[str, VariableGrid] = {}
        for n in names[:n_gridded]:
            vmin, vmax = bounds[n]
            grid[n] = VariableGrid(min=vmin, max=vmax, step=(vmax - vmin) / step_divisor)
        # Midpoint start is deliberately off-grid to prove even the first sample snaps.
        current = {n: (all_vars[n].min_value + all_vars[n].max_value) / 2 for n in names}

        result = run_real_phase(
            env,
            bounds,
            current,
            _config(
                algorithm=algorithm, n_iterations=n_iterations, n_initial_samples=n_initial_samples
            ),
            grid=grid,
        )

        assert result.data.height == n_initial_samples + n_iterations
        for n, g in grid.items():
            for v in result.data[n].to_list():
                assert v == pytest.approx(
                    snap_value_to_grid(v, vmin=g.min, vmax=g.max, step=g.step)
                )


class TestFeasibilityFilter:
    """best_feasible_row applies each constraint as a strict inequality and treats
    an unverifiable (NaN/null) constraint reading as infeasible."""

    @pytest.mark.parametrize(
        "data,constraints,expected_x",
        [
            (
                pl.DataFrame({"x": [0.1, 0.2, 0.3], "f": [1.0, 2.0, 3.0], "c": [0.4, 0.5, 0.6]}),
                {"c": ConstraintSpec(type=ConstraintType.GREATER_THAN, value=0.5)},
                0.3,
            ),
            (
                pl.DataFrame({"x": [0.1, 0.2, 0.3], "f": [3.0, 2.0, 1.0], "c": [0.4, 0.5, 0.6]}),
                {"c": ConstraintSpec(type=ConstraintType.LESS_THAN, value=0.5)},
                0.1,
            ),
            (
                pl.DataFrame({"x": [0.1, 0.2], "f": [9.0, 3.0], "c": [float("nan"), 0.9]}),
                {"c": ConstraintSpec(type=ConstraintType.GREATER_THAN, value=0.5)},
                0.2,
            ),
            (
                pl.DataFrame({"x": [0.1, 0.2], "f": [9.0, 3.0], "c": [None, 0.9]}),
                {"c": ConstraintSpec(type=ConstraintType.GREATER_THAN, value=0.5)},
                0.2,
            ),
            (
                pl.DataFrame({"x": [0.1, 0.2], "f": [9.0, 3.0]}),
                {},
                0.1,
            ),
            (
                pl.DataFrame({"x": [0.1, 0.2], "f": [9.0, 3.0], "c": [0.1, 0.2]}),
                {"c": ConstraintSpec(type=ConstraintType.GREATER_THAN, value=0.5)},
                None,
            ),
        ],
        ids=[
            "greater_than_excludes_rows_at_or_below_threshold",
            "less_than_excludes_rows_at_or_above_threshold",
            "nan_constraint_value_is_infeasible",
            "null_constraint_value_is_infeasible",
            "empty_constraints_is_plain_best_finite",
            "no_feasible_row_returns_none",
        ],
    )
    def test_best_feasible_row_selects_expected_x(
        self, data: pl.DataFrame, constraints: dict[str, ConstraintSpec], expected_x: float | None
    ) -> None:
        best = best_feasible_row(data, "f", constraints, direction=Direction.MAXIMIZE)
        # No feasible row at all -> None, rather than a row that violates a constraint.
        selected_x = None if best is None else pytest.approx(best["x"])
        assert selected_x == expected_x


class TestBuildBestPoint:
    """build_best_point excludes rows where the objective *or any selected
    variable* is non-finite, so the partial-save/cancellation path never
    reports a non-finite setpoint as safe to apply to hardware."""

    @pytest.mark.parametrize(
        "data,expected",
        [
            (
                pl.DataFrame({"x": [0.1, 0.2, 0.3], "f": [float("inf"), 2.0, float("nan")]}),
                {"x": 0.2},
            ),
            (
                pl.DataFrame({"x": [float("nan"), 0.2], "f": [9.0, 3.0]}),
                {"x": 0.2},
            ),
            (
                pl.DataFrame({"x": [0.1, 0.2], "f": [float("nan"), float("inf")]}),
                {},
            ),
        ],
        ids=[
            "non_finite_objective_rows_excluded",
            "non_finite_variable_row_excluded",
            "all_non_finite_returns_empty",
        ],
    )
    def test_non_finite_rows_never_selected(
        self, data: pl.DataFrame, expected: dict[str, float]
    ) -> None:
        assert build_best_point(data, "f", ["x"], direction=Direction.MAXIMIZE) == expected


class TestConstrainedPhase:
    """run_optimization_phase honors constraints in best-so-far, final selection,
    and the no-feasible-row failure path."""

    @staticmethod
    def _evaluate_from_map(outcomes: dict[float, tuple[float, float]]) -> Any:
        """Return an evaluate() that maps a point's ``x`` to ``(f, loss)`` outcomes."""

        def evaluate(point: dict[str, float]) -> dict[str, float]:
            f, loss = outcomes[point["x"]]
            return {**point, "f": f, "loss": loss}

        return evaluate

    def test_final_best_prefers_feasible_over_better_infeasible(self) -> None:
        """A higher objective that violates the constraint is not applied; the best
        feasible point is selected instead."""
        opt = FakeOptimizer(suggestions=[{"x": 0.8}, {"x": 0.9}])
        # current (0.5): feasible f=3; 0.8: infeasible better f=9; 0.9: feasible f=4.
        evaluate = self._evaluate_from_map(
            {0.5: (3.0, 0.2), 0.8: (9.0, 0.9), 0.9: (4.0, 0.1)},
        )
        constraints = {"loss": ConstraintSpec(type=ConstraintType.LESS_THAN, value=0.5)}
        bus = EventBus()
        completed: list[OptimizationEvent] = []
        bus.subscribe(EventType.PHASE_COMPLETED, completed.append)

        result = _run_pure_loop(
            opt,
            evaluate,
            n_iterations=2,
            context=ControlContext(event_bus=bus),
            constraints=constraints,
        )
        best = report_results(
            result.data,
            {"x": [0.0, 1.0]},
            "f",
            direction=Direction.MAXIMIZE,
            constraints=constraints,
        )
        assert best["x"] == pytest.approx(0.9)  # feasible f=4, not infeasible f=9
        # PHASE_COMPLETED best_value reflects the feasible best, not the infeasible 9.
        assert isinstance(completed[0].data, PhaseCompletedData)
        assert completed[0].data.best_value == pytest.approx(4.0)

    def test_best_so_far_ignores_infeasible_during_seeding(self) -> None:
        """The running best reported per iteration is the best FEASIBLE value seen,
        so an infeasible-but-better step does not inflate it. n_initial=1 evaluates
        only the current state; the two off-current points arrive as suggestions so
        their outcomes are pinned (seeding fill is uniform-random)."""
        evaluate = self._evaluate_from_map(
            {0.5: (2.0, 0.1), 0.7: (9.0, 0.9), 0.9: (4.0, 0.2)},
        )
        constraints = {"loss": ConstraintSpec(type=ConstraintType.LESS_THAN, value=0.5)}
        bus = EventBus()
        events: list[OptimizationEvent] = []
        bus.subscribe(EventType.ITERATION_COMPLETED, events.append)

        opt = FakeOptimizer(suggestions=[{"x": 0.7}, {"x": 0.9}])
        _run_pure_loop(
            opt,
            evaluate,
            n_iterations=2,
            context=ControlContext(event_bus=bus),
            constraints=constraints,
        )
        best_values = [e.data.best_value for e in events]  # type: ignore[union-attr]
        # After current (feasible f=2): 2.0. After x=0.7 (infeasible f=9): still 2.0.
        # After x=0.9 (feasible f=4): 4.0.
        assert best_values == [pytest.approx(2.0), pytest.approx(2.0), pytest.approx(4.0)]

    def test_no_feasible_row_raises_and_emits_phase_failed(self) -> None:
        """Objective measured everywhere but no point feasible → EvaluationError +
        PHASE_FAILED; an infeasible setpoint must never be applied."""
        opt = FakeOptimizer(suggestions=[{"x": 0.8}, {"x": 0.9}])
        evaluate = self._evaluate_from_map(
            {0.5: (3.0, 0.9), 0.8: (9.0, 0.8), 0.9: (4.0, 0.7)},
        )
        constraints = {"loss": ConstraintSpec(type=ConstraintType.LESS_THAN, value=0.5)}
        bus = EventBus()
        emitted: list[OptimizationEvent] = []
        bus.subscribe(EventType.PHASE_FAILED, emitted.append)

        with pytest.raises(EvaluationError, match="no point satisfying the constraints"):
            _run_pure_loop(
                opt,
                evaluate,
                n_iterations=2,
                context=ControlContext(event_bus=bus),
                constraints=constraints,
            )
        assert len(emitted) == 1
        assert isinstance(emitted[0].data, PhaseFailedData)
        assert "no point satisfying the constraints" in emitted[0].data.error_message


class TestConstrainedEIFeasibilitySearch:
    """Acceptance test for issue #80: a constrained ExpectedImprovement run that
    starts entirely outside the feasible region must not hard-fail on the first BO
    suggest. The real XoptOptimizer's feasibility-search fallback returns random
    space-filling points until a feasible row is ingested, then the phase completes
    with a feasible best."""

    def test_infeasible_start_recovers_and_completes(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        constraints = {"loss": ConstraintSpec(type=ConstraintType.LESS_THAN, value=0.5)}
        calls = {"n": 0}

        def evaluate(point: dict[str, float]) -> dict[str, float]:
            # First two evaluations violate the constraint (loss high), then feasible —
            # simulating a machine that starts outside the constraint region.
            calls["n"] += 1
            loss = 0.9 if calls["n"] <= 2 else 0.2
            return {**point, "f": float(calls["n"]), "loss": loss}

        bounds: VariableBounds = {"x": [0.0, 1.0], "y": [0.0, 1.0]}
        current = {"x": 0.5, "y": 0.5}
        config = _config(
            AlgorithmType.EXPECTED_IMPROVEMENT,
            n_iterations=4,
            n_initial_samples=1,
            objective_name="f",
        )
        optimizer = XoptOptimizer(
            variables=bounds,
            objective_name="f",
            direction=Direction.MAXIMIZE,
            config=config,
            current_env_values=current,
            constraints=constraints,
        )
        bus = EventBus()
        completed: list[OptimizationEvent] = []
        bus.subscribe(EventType.PHASE_COMPLETED, completed.append)

        with caplog.at_level(logging.WARNING):
            result = run_optimization_phase(
                optimizer=optimizer,
                evaluate=evaluate,
                bounds=bounds,
                objective_name="f",
                direction=Direction.MAXIMIZE,
                config=config,
                context=ControlContext(event_bus=bus),
                current_env_values=current,
                n_initial=optimizer.resolve_n_initial(),
                constraints=constraints,
            )

        # Phase completed instead of crashing on the first constrained-EI suggest.
        assert len(completed) == 1
        # A feasible best exists and satisfies the constraint.
        best_row = best_feasible_row(result.data, "f", constraints, direction=Direction.MAXIMIZE)
        assert best_row is not None
        assert best_row["loss"] < 0.5
        # The feasibility search engaged (and was logged) while all rows were infeasible.
        assert any("feasibility search" in rec.message.lower() for rec in caplog.records)


class TestReportResults:
    @pytest.mark.parametrize(
        ("rows", "env_vars", "objective", "direction", "expected_x"),
        [
            pytest.param(
                {"x": [0.1, 0.5, 0.9], "y": [0.2, 0.6, 0.8], "f": [1.0, 3.0, 2.0]},
                {"x": [0.0, 1.0], "y": [0.0, 1.0]},
                "f",
                "MAXIMIZE",
                {"x": 0.5, "y": 0.6},
                id="returns_best_point_variables",
            ),
            pytest.param(
                {"x": [0.1, 0.2, 0.3, 0.4], "f": [10.0, 50.0, 30.0, 20.0]},
                {"x": [0.0, 1.0]},
                "f",
                "MAXIMIZE",
                {"x": 0.2},
                id="maximizes_objective",
            ),
            pytest.param(
                {"x": [0.1, 0.2, 0.3, 0.4], "f": [10.0, 50.0, 30.0, 20.0]},
                {"x": [0.0, 1.0]},
                "f",
                "MINIMIZE",
                {"x": 0.1},
                id="minimizes_objective",
            ),
            pytest.param(
                {"x": [0.1, 0.2, 0.3], "f": [-10.0, -1.0, -5.0]},
                {"x": [0.0, 1.0]},
                "f",
                "MAXIMIZE",
                {"x": 0.2},
                id="negative_objective_values",
            ),
            pytest.param(
                {"x": [0.1, 0.5, 0.9], "f": [1.0, float("inf"), 2.0]},
                {"x": [0.0, 1.0]},
                "f",
                "MAXIMIZE",
                {"x": 0.9},
                id="filters_inf_values",
            ),
            pytest.param(
                {"x": [0.1, 0.5, 0.9], "f": [float("-inf"), 3.0, 2.0]},
                {"x": [0.0, 1.0]},
                "f",
                "MINIMIZE",
                {"x": 0.9},
                id="minimize_excludes_neg_inf",
            ),
            pytest.param(
                {"x": [0.1, float("inf")], "f": [1.0, 99.0]},
                {"x": [0.0, 1.0]},
                "f",
                "MAXIMIZE",
                {"x": 0.1},
                id="filters_inf_in_variables",
            ),
            pytest.param(
                {"x": [0.1, 0.2, 0.3], "f": [0.5, None, 0.7]},
                {"x": [0.0, 1.0]},
                "f",
                "MAXIMIZE",
                {"x": 0.3},
                id="filters_null_objective",
            ),
            pytest.param(
                {"x": [0.1, None], "f": [0.5, 0.9]},
                {"x": [0.0, 1.0]},
                "f",
                "MAXIMIZE",
                {"x": 0.1},
                id="filters_null_variable",
            ),
            pytest.param(
                {"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0], "obj": [10.0, 20.0]},
                {"a": [0.0, 10.0], "b": [0.0, 10.0], "c": [0.0, 10.0]},
                "obj",
                "MAXIMIZE",
                {"a": 2.0, "b": 4.0, "c": 6.0},
                id="all_env_variables_returned",
            ),
        ],
    )
    def test_best_point_selection(
        self,
        rows: dict[str, list[float]],
        env_vars: dict[str, list[float]],
        objective: str,
        direction: Direction,
        expected_x: dict[str, float],
    ) -> None:
        """report_results picks the best row for the direction, filtering non-finite,
        and returns every environment variable of that row."""
        best = report_results(pl.DataFrame(rows), env_vars, objective, direction=direction)
        assert set(best) == set(expected_x)
        for key, value in expected_x.items():
            assert best[key] == pytest.approx(value)

    def test_all_non_finite_returns_empty(self, caplog: pytest.LogCaptureFixture) -> None:
        data = pl.DataFrame({"x": [float("inf")], "f": [float("inf")]})
        with caplog.at_level(logging.WARNING):
            best = report_results(data, {"x": [0.0, 1.0]}, "f", direction=Direction.MAXIMIZE)
        assert best == {}
        assert any("No finite" in rec.message for rec in caplog.records)

    def test_constrained_summary_reports_feasible_count(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """With constraints, the summary logs how many finite evaluations were feasible.

        The mean spans all finite evaluations (labeled as such), so the feasible
        count is what lets an operator relate it to the feasible-only "Best".
        """
        data = pl.DataFrame({"x": [0.1, 0.2, 0.3], "obj": [1.0, 2.0, 3.0], "c": [0.0, 1.0, 1.0]})
        constraints = {"c": ConstraintSpec(type=ConstraintType.GREATER_THAN, value=0.5)}
        with caplog.at_level(logging.INFO):
            best = report_results(
                data,
                {"x": [0.0, 1.0]},
                "obj",
                direction=Direction.MAXIMIZE,
                constraints=constraints,
            )
        assert best["x"] == pytest.approx(0.3)
        assert "Feasible evaluations: 2 of 3 finite" in caplog.text


class TestHardFailAbortsRun:
    """The loop aborts (PHASE_FAILED + re-raise) on a control-system fault and
    on total persistent objective loss, but leaves non-control errors unmasked.

    These reuse the ``make_abort_phase`` / ``make_complete_phase`` scaffolds
    (conftest): a MagicMock env whose ``evaluate`` is driven per the given
    ``exc``, wired through the real app Evaluator + a single-variable
    XoptOptimizer + a 2-iteration config + an EventBus.
    """

    def test_missing_objective_name_raises(self) -> None:
        with pytest.raises(ValueError, match="objective_name must be set"):
            _run_pure_loop(FakeOptimizer(), lambda p: {**p, "f": 1.0}, objective_name="")

    @pytest.mark.parametrize(
        ("evaluate_side_effect", "exc_type", "match", "failed_fragments"),
        [
            pytest.param(
                ControlSystemUnreachableError("unreachable"),
                ControlSystemUnreachableError,
                "unreachable",
                ("unreachable",),
                id="persistent_control_loss",
            ),
            pytest.param(
                lambda *_a, **_k: {"y": PVReading(value=None, connected=True)},
                EvaluationError,
                "no finite",
                ("no finite",),
                id="all_objectives_nan",
            ),
            pytest.param(
                RuntimeError("generator blew up"),
                RuntimeError,
                "generator blew up",
                (),
                id="plain_error_not_wrapped",
            ),
        ],
    )
    def test_abort_paths_emit_phase_failed_only_for_control_faults(
        self,
        evaluate_side_effect: Any,
        exc_type: type[Exception],
        match: str,
        failed_fragments: tuple[str, ...],
    ) -> None:
        """A control-system fault, and total control loss degraded to all-NaN
        observations, both fail the phase with the reason in PHASE_FAILED. A
        non-control-system error surfaces unchanged and emits NO PHASE_FAILED (the
        loop only wraps ControlSystemUnreachableError / EvaluationError /
        WriteRefusedError)."""
        phase_kwargs, emitted = make_abort_phase(evaluate_side_effect)
        with pytest.raises(exc_type, match=match):
            run_optimization_phase(**phase_kwargs)
        assert len(emitted) == len(failed_fragments)
        for event, fragment in zip(emitted, failed_fragments, strict=True):
            assert isinstance(event.data, PhaseFailedData)
            assert fragment in event.data.error_message

    @pytest.mark.parametrize(
        ("exc", "failed_fragment", "reason"),
        [
            pytest.param(
                EvaluationError("Shot-gated read: timed out after 10.0s waiting for fresh samples"),
                "timed out",
                None,
                id="evaluation_error",
            ),
            pytest.param(
                WriteRefusedError(
                    "Write to GTL:Q1 refused: writes disabled", reason="WRITES_DISABLED"
                ),
                "writes disabled",
                "WRITES_DISABLED",
                id="write_refused",
            ),
        ],
    )
    def test_mid_run_failure_aborts_on_first_failing_step(
        self, exc: Exception, failed_fragment: str, reason: str | None
    ) -> None:
        """A mid-run failure aborts on the first failing step and does not keep
        evaluating through the remaining iteration budget.

        For a policy-refused write (blocked) that also means it is never retried —
        policy does not become true by retrying, and retrying a refused write looks
        like probing the reference monitor. Either way the operator-facing reason
        reaches PHASE_FAILED.
        """
        calls = {"n": 0}

        def evaluate(*_a: Any, **_k: Any) -> dict[str, PVReading]:
            calls["n"] += 1
            if calls["n"] == 1:
                return {"y": PVReading(value=1.0, connected=True)}
            raise exc

        phase_kwargs, emitted = make_abort_phase(evaluate)
        with pytest.raises(type(exc)) as exc_info:
            run_optimization_phase(**phase_kwargs)
        assert str(exc) in str(exc_info.value)
        assert getattr(exc_info.value, "reason", None) == reason
        assert len(emitted) == 1
        assert isinstance(emitted[0].data, PhaseFailedData)
        assert failed_fragment in emitted[0].data.error_message
        # 1 clean seed + 1 failing step, then abort — no further evaluations.
        assert calls["n"] == 2

    @pytest.mark.parametrize(
        ("per_call", "fallback", "direction", "expected_best"),
        [
            pytest.param({1: math.inf}, 5.0, Direction.MAXIMIZE, 5.0, id="maximize_excludes_inf"),
            pytest.param(
                {1: -math.inf, 2: 10.0},
                3.0,
                Direction.MINIMIZE,
                3.0,
                id="minimize_excludes_neg_inf",
            ),
        ],
    )
    def test_best_comes_from_the_finite_filtered_frame(
        self,
        per_call: dict[int, float],
        fallback: float,
        direction: Direction,
        expected_best: float,
    ) -> None:
        """The reported best comes from the finite-filtered frame: an inf sentinel is
        excluded for either direction, so MINIMIZE reports the finite minimum rather
        than -inf (and MAXIMIZE the finite maximum rather than +inf)."""
        calls = {"n": 0}

        def _evaluate(*_args: Any, **_kwargs: Any) -> dict[str, PVReading]:
            calls["n"] += 1
            return {"y": PVReading(value=per_call.get(calls["n"], fallback))}

        phase_kwargs, completed = make_complete_phase(_evaluate, direction=direction)
        run_optimization_phase(**phase_kwargs)
        assert len(completed) == 1
        data = completed[0].data
        assert isinstance(data, PhaseCompletedData)
        assert data.best_value is not None
        assert math.isfinite(data.best_value)
        assert data.best_value == expected_best
