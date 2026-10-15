"""The in-process simulator and the benchmark harness stay runnable."""

from tests.optimizer.sim.benchmark import best_so_far, run_one


def test_a_seeded_run_is_reproducible():
    first = run_one("osprey", "random", seed=3, n_initial=3, n_iterations=2)
    second = run_one("osprey", "random", seed=3, n_initial=3, n_iterations=2)
    assert len(first) == 5
    assert first == second


def test_best_so_far_ignores_nan():
    assert best_so_far([1.0, float("nan"), 0.5, 2.0]) == [1.0, 1.0, 1.0, 2.0]
