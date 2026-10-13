"""Simulator benchmark: per-seed best objective at half and full budget.

Runs the engine in-process against the tuning simulator's physics, so the same
script measures OSPREY's port and tuning_scripts itself::

    # OSPREY (this repo)
    uv run python -m tests.services.optimizer.sim.benchmark --package osprey -o osprey.json
    # tuning_scripts, from its checkout, with this repo on PYTHONPATH
    PYTHONPATH=/path/to/osprey uv run python -m tests.services.optimizer.sim.benchmark \\
        --package tuning_scripts -o tuning_scripts.json
    # compare
    uv run python -m tests.services.optimizer.sim.benchmark --compare tuning_scripts.json osprey.json

The protocol is the one tuning_scripts' optimization rule sets: identical seeds,
one fixed budget at least twice the initial-sample count, and per seed the best
at half and full budget plus the evaluation at which the candidate first
reaches the baseline's final best.
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
import statistics
import sys
from typing import Any

PACKAGES = {
    "osprey": ("osprey.services.optimizer", "osprey.services.optimizer.models"),
    "tuning_scripts": ("tuning_scripts_app", "tuning_scripts_models"),
}


def run_one(
    package: str, algorithm: str, seed: int, n_initial: int, n_iterations: int
) -> list[float]:
    """The objective at every evaluation of one seeded run."""
    import numpy as np
    import torch

    from .environment import INITIAL_VALUES, PHASE_VARIABLES, VARIABLES, make_environment_class

    app_pkg, models_pkg = PACKAGES[package]
    config_mod = importlib.import_module(f"{models_pkg}.config")
    orchestration = importlib.import_module(f"{app_pkg}.services.orchestration")
    draft = importlib.import_module(f"{app_pkg}.services.draft_resolution")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env_cls = make_environment_class(models_pkg)
    fields: dict[str, Any] = {
        "environment_name": "sim_injector",
        "objective_name": "charge_ltb",
        "algorithm": algorithm,
        "n_initial_samples": n_initial,
        "n_iterations": n_iterations,
        "variables": list(PHASE_VARIABLES),
        "variable_overrides": {name: {"window_fraction": 0.1} for name in PHASE_VARIABLES},
    }
    if package == "osprey":
        fields["control_target"] = "va"
    config = config_mod.OptimizationConfig(**fields)

    def factory(_name: str, env_config: dict[str, Any]) -> Any:
        return env_cls({**env_config, "sim_seed": seed})

    env = factory("sim_injector", {})
    with env:
        draft.resolve_draft_config(config, env)
    for name in PHASE_VARIABLES:
        assert VARIABLES[name].min_value <= INITIAL_VALUES[name] <= VARIABLES[name].max_value

    result = orchestration.OptimizationEngine(config, environment_factory=factory).run()
    return [float(v) for v in result.data["charge_ltb"].to_list()]


def best_so_far(values: list[float]) -> list[float]:
    out, best = [], float("-inf")
    for v in values:
        if v == v:  # NaN never improves the best
            best = max(best, v)
        out.append(best)
    return out


def measure(args: argparse.Namespace) -> dict[str, Any]:
    runs: dict[str, Any] = {}
    for algorithm in args.algorithms:
        for seed in args.seeds:
            trace = run_one(args.package, algorithm, seed, args.n_initial, args.n_iterations)
            runs[f"{algorithm}/{seed}"] = trace
            curve = best_so_far(trace)
            print(
                f"{algorithm:>24} seed {seed}: half {curve[len(curve) // 2 - 1]:.4f}  full {curve[-1]:.4f}",
                file=sys.stderr,
            )
    return {
        "package": args.package,
        "n_initial": args.n_initial,
        "n_iterations": args.n_iterations,
        "runs": runs,
    }


def compare(baseline: dict[str, Any], candidate: dict[str, Any]) -> int:
    """Print the per-seed table; non-zero exit when the pass bar fails."""
    assert (baseline["n_initial"], baseline["n_iterations"]) == (
        candidate["n_initial"],
        candidate["n_iterations"],
    )
    keys = sorted(set(baseline["runs"]) & set(candidate["runs"]))
    print(
        "| run | baseline half | candidate half | baseline full | candidate full | candidate reaches baseline best at |"
    )
    print("|---|---|---|---|---|---|")
    algorithms: dict[str, list[tuple[float, float]]] = {}
    for key in keys:
        b, c = best_so_far(baseline["runs"][key]), best_so_far(candidate["runs"][key])
        half = len(b) // 2 - 1
        reached = next((i + 1 for i, v in enumerate(c) if v >= b[-1]), None)
        print(
            f"| {key} | {b[half]:.4f} | {c[half]:.4f} | {b[-1]:.4f} | {c[-1]:.4f} | {reached or 'never'} |"
        )
        algorithms.setdefault(key.split("/")[0], []).append((b[-1], c[-1]))
    ok = True
    for algorithm, pairs in algorithms.items():
        b_med = statistics.median(p[0] for p in pairs)
        c_med = statistics.median(p[1] for p in pairs)
        verdict = "ok" if c_med >= b_med else "WORSE"
        ok &= c_med >= b_med
        print(
            f"\n{algorithm}: median full-budget best baseline {b_med:.4f}, candidate {c_med:.4f} ({verdict})"
        )
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--package", choices=sorted(PACKAGES), default="osprey")
    parser.add_argument("--algorithms", nargs="+", default=["random", "upper_confidence_bound"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-initial", type=int, default=5)
    parser.add_argument("--n-iterations", type=int, default=20)
    parser.add_argument("-o", "--output")
    parser.add_argument("--compare", nargs=2, metavar=("BASELINE", "CANDIDATE"))
    args = parser.parse_args()
    if args.compare:
        with open(args.compare[0]) as b, open(args.compare[1]) as c:
            return compare(json.load(b), json.load(c))
    result = measure(args)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
