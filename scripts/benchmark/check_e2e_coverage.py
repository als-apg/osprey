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

Always exits 0: this is a warning, never a gate (per design).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


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

    return 0


if __name__ == "__main__":
    sys.exit(main())
