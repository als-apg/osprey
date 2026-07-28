#!/usr/bin/env python3
"""Startup coverage guard for the model-matrix custom e2e runners (issue #259).

The matrix runners do NOT hand-maintain a pytest `--ignore` list inline; they
derive it from ``scripts/benchmark/matrix_e2e_config.json`` and call this script at
startup. Purpose: guarantee the runner executes the FULL ``tests/e2e/`` kit
EXCEPT the files we explicitly exclude — and shout if that invariant slips.

It emits (to STDERR) an audit of every e2e test file marked RUN or EXCLUDED,
and a ``WARNING:`` line when:
  * a configured exclusion no longer matches a real file (stale exclusion), or
  * the number of files that WILL run drops below ``baseline_expected_run_min``
    (a silent-coverage-loss tripwire).

With ``--print-ignore-args`` it prints the ``--ignore=<path>`` tokens to STDOUT
so the bash runner can splice them straight into its pytest invocation —
keeping the exclusion list single-sourced in the JSON config.

With ``--check-lanes`` it additionally collects the in-scope suite (subprocess
``pytest --collect-only`` with the exclusion ``--ignore``s, using the
``OSPREY_E2E_LANES`` manifest hook in tests/e2e/conftest.py) and verifies every
collected test carries EXACTLY ONE benchmark-lane marker — ``agentic_benchmark``
(model-capability score) or ``harness_benchmark`` (harness-integrity score).
This closes the drift hole where a new e2e file silently joins the matrix scope
without declaring what its pass/fail measures.

Exit code: 0 in the default/audit modes (warnings only, per design) — but
``--check-lanes`` IS a gate and exits 1 on unmarked or double-marked tests, so
the matrix runner refuses to score a suite whose lane split is ambiguous.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def check_lanes(root: Path, excluded: dict[str, str]) -> list[str]:
    """Collect the in-scope suite and return lane-marker violations (empty = OK).

    Runs ``pytest tests/e2e/ --collect-only`` in a subprocess with the same
    ``--ignore`` set the matrix runner uses, pointing ``OSPREY_E2E_LANES`` at a
    temp manifest so the conftest hook records each collected nodeid's lane.
    Collection failure is itself a violation — the matrix could not run either.
    """
    with tempfile.TemporaryDirectory() as td:
        lanes_path = Path(td) / "lanes.json"
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "tests/e2e/",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            "-o",
            "addopts=",
            *[f"--ignore={p}" for p in excluded],
        ]
        env = {**os.environ, "OSPREY_E2E_LANES": str(lanes_path)}
        proc = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True)
        if not lanes_path.exists():
            tail = (proc.stdout + proc.stderr)[-800:]
            return [f"lane collection produced no manifest (rc={proc.returncode}): {tail}"]
        lanes = json.loads(lanes_path.read_text(encoding="utf-8"))

    violations = []
    unmarked = sorted(n for n, lane in lanes.items() if lane == "unmarked")
    double = sorted(n for n, lane in lanes.items() if lane == "both")
    if unmarked:
        violations.append(
            f"{len(unmarked)} in-scope test(s) carry NO benchmark-lane marker — add "
            "@pytest.mark.agentic_benchmark or @pytest.mark.harness_benchmark (or add the "
            "file to excluded_files in matrix_e2e_config.json):"
        )
        violations.extend(f"  {n}" for n in unmarked)
    if double:
        violations.append(f"{len(double)} test(s) carry BOTH lane markers — pick one:")
        violations.extend(f"  {n}" for n in double)
    if not lanes:
        violations.append("lane manifest is empty — collection found no in-scope tests")
    return violations


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".", help="OSPREY checkout root")
    ap.add_argument(
        "--config",
        default="scripts/benchmark/matrix_e2e_config.json",
        help="path to matrix_e2e_config.json (relative to --repo-root or absolute)",
    )
    ap.add_argument(
        "--print-ignore-args",
        action="store_true",
        help="print `--ignore=<path>` tokens to stdout for the runner to consume",
    )
    ap.add_argument(
        "--check-lanes",
        action="store_true",
        help="collect the in-scope suite and GATE (exit 1) on tests missing a "
        "benchmark-lane marker (agentic_benchmark/harness_benchmark)",
    )
    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    suite_root = root / cfg["suite_root"]
    excluded = {e["path"]: e["reason"] for e in cfg["excluded_files"]}
    baseline_min = int(cfg.get("baseline_expected_run_min", 0))

    # Every test file actually present in the suite (recursive).
    all_files = sorted(str(p.relative_to(root)) for p in suite_root.rglob("test_*.py"))

    run_files = [f for f in all_files if f not in excluded]
    warnings: list[str] = []

    # Tripwire 1: stale exclusions (configured to exclude a file that's gone).
    for path in excluded:
        if not (root / path).exists():
            warnings.append(f"stale exclusion in config — file does not exist: {path}")

    # Tripwire 2: silent coverage loss.
    if len(run_files) < baseline_min:
        warnings.append(
            f"only {len(run_files)} e2e files will RUN, below baseline "
            f"{baseline_min} — coverage may have regressed (deleted tests? "
            f"over-broad exclusion?)"
        )

    # --- audit to stderr ---
    print(
        f"[e2e-coverage] suite={cfg['suite_root']}  files: {len(all_files)} total, "
        f"{len(run_files)} RUN, {len(excluded)} EXCLUDED",
        file=sys.stderr,
    )
    for f in all_files:
        mark = "EXCLUDED" if f in excluded else "run"
        note = f"  ({excluded[f]})" if f in excluded else ""
        print(f"  [{mark:8}] {f}{note}", file=sys.stderr)
    for w in warnings:
        print(f"WARNING: [e2e-coverage] {w}", file=sys.stderr)
    if not warnings:
        print("[e2e-coverage] OK — full kit covered except explicit exclusions.", file=sys.stderr)

    if args.print_ignore_args:
        for path in excluded:
            print(f"--ignore={path}")

    if args.check_lanes:
        violations = check_lanes(root, excluded)
        for v in violations:
            print(f"LANE-GATE: {v}", file=sys.stderr)
        if violations:
            return 1
        print(
            "[e2e-coverage] lane markers OK — every in-scope test declares its lane.",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
