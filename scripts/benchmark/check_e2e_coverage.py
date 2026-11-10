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

With ``--print-run-args`` it prints the board's pytest selection to STDOUT, one
token per line, so the bash runner can splice it straight into its pytest
invocation: the ``--ignore=<path>`` tokens (the exclusion list, single-sourced
in the JSON config), ``-m agentic`` (the board runs the tiered agentic tests
only), ``-p no:rerunfailures`` (the board scores each test's first attempt;
the ``flaky`` reruns PR CI relies on are off) and the two options that keep a
result about the model: the per-test timeout and the reported duration cover
the test's call, not a fixture's stack build or a queue for a host-wide lock.

With ``--check-board-images`` it is a gate on the images the in-scope tests need
and never build (``board_images`` in the config): each must exist, and must be
the one ``build_board_images.sh`` built at this checkout's revision. A stale image
is the point: its simulator view format can differ from this checkout's render,
and the tests would die in setup. ``--select EXPR`` limits the gate to a smoke run
whose ``-k`` selection reaches a test that needs them.
``--record-board-images`` is that script's last step.

With ``--check-lanes`` it additionally collects the in-scope suite (subprocess
``pytest --collect-only`` with the exclusion ``--ignore``s, using the
``OSPREY_E2E_LANES`` manifest hook in tests/e2e/conftest.py) and verifies every
collected test carries EXACTLY ONE benchmark-lane marker — ``agentic`` (an
agentic test with its tier; ``tests/e2e/agentic_tiers.py``) or
``harness_benchmark`` (an agent may run, but the assertion is OSPREY's). This
closes the drift hole where a new e2e file silently joins the matrix scope
without declaring what its pass/fail measures. An incomplete ``agentic`` marker
fails the collection itself, which the gate reports.

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

#: The board's selection beside the exclusions: the tiered agentic tests, first
#: attempt only, timed over the test's call.
BOARD_SELECTION = (
    "-m",
    "agentic",
    "-p",
    "no:rerunfailures",
    "-o",
    "timeout_func_only=true",
    "-o",
    "junit_duration_report=call",
)

#: What ``build_board_images.sh`` built: the revision and each image's id.
BOARD_IMAGES_STAMP = Path(tempfile.gettempdir()) / "osprey-board-images.json"
BUILD_IMAGES_COMMAND = "scripts/benchmark/build_board_images.sh"


def _image_name(entry: dict) -> str:
    """The image a ``board_images`` entry names, after its override variable."""
    return os.environ.get(entry.get("override_env", ""), "") or entry["image"]


def _image_id(image: str) -> str | None:
    """The image's id on this host, or None when it is not there."""
    runtime = os.environ.get("OSPREY_E2E_RUNTIME", "docker")
    try:
        proc = subprocess.run(
            [runtime, "image", "inspect", image, "--format", "{{.Id}}"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def _revision(root: Path) -> str | None:
    """The checkout's commit, or None when it cannot be read."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def check_board_images(root: Path, entries: list[dict], stamp: Path) -> list[str]:
    """What stops a full cell from running its image-bound tests (empty = OK)."""
    try:
        recorded = json.loads(stamp.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        recorded = {}
    problems = []
    for entry in entries:
        image = _image_name(entry)
        found = _image_id(image)
        if found is None:
            problems.append(f"image {image} is not on this host")
        elif recorded.get("images", {}).get(image) != found:
            problems.append(f"image {image} was not built at this checkout's revision")
    revision = _revision(root)
    if not problems and (revision is None or recorded.get("revision") != revision):
        problems.append(
            f"the images were built at {recorded.get('revision')}, and this checkout is {revision}"
        )
    return [f"{p}; build them with {BUILD_IMAGES_COMMAND}" for p in problems]


def selection_needs_images(root: Path, cfg: dict, select: str) -> bool:
    """Whether a smoke run's ``-k`` selection reaches a test the images serve.

    Anything short of a clean answer counts as yes: the gate is the cheap failure.
    """
    excluded = {e["path"]: e["reason"] for e in cfg["excluded_files"]}
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/e2e/", "--collect-only", "-q"]
        + ["-p", "no:cacheprovider", "-o", "addopts=", "--assert=plain"]
        + [*run_args(excluded), "-k", select],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if proc.returncode not in (0, 5):  # 5: the selection matched nothing
        return True
    needed_by = {path for e in cfg.get("board_images", []) for path in e["needed_by"]}
    return any(line.split("::", 1)[0] in needed_by for line in proc.stdout.splitlines())


def record_board_images(root: Path, entries: list[dict], stamp: Path) -> list[str]:
    """Record the checkout's revision and each image's id; return any image missing."""
    images = {_image_name(entry): _image_id(_image_name(entry)) for entry in entries}
    missing = [image for image, found in images.items() if found is None]
    if not missing:
        stamp.write_text(
            json.dumps({"revision": _revision(root), "images": images}), encoding="utf-8"
        )
    return missing


def run_args(excluded: dict[str, str]) -> list[str]:
    """The pytest arguments a board cell runs ``tests/e2e/`` with."""
    return [f"--ignore={p}" for p in excluded] + list(BOARD_SELECTION)


def check_lanes(root: Path, excluded: dict[str, str]) -> list[str]:
    """Collect the in-scope suite and return lane-marker violations (empty = OK).

    Runs ``pytest tests/e2e/ --collect-only`` in a subprocess with the same
    ``--ignore`` set the matrix runner uses, pointing ``OSPREY_E2E_LANES`` at a
    temp manifest so the conftest hook records each collected nodeid's lane.
    The manifest is written from a collection hook that runs even when
    collection was refused, so the collector's own exit status decides whether
    what it left behind can be scored.
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
            # Collection reads markers and runs no assert, so rewriting each
            # module's asserts on import buys nothing here; on a checkout with
            # no bytecode cache it is a quarter of the collection's time.
            "--assert=plain",
            *[f"--ignore={p}" for p in excluded],
        ]
        env = {**os.environ, "OSPREY_E2E_LANES": str(lanes_path)}
        proc = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True)
        tail = (proc.stdout + proc.stderr)[-800:]
        if not lanes_path.exists():
            return [f"lane collection produced no manifest (rc={proc.returncode}): {tail}"]
        if proc.returncode != 0:
            return [f"lane collection failed (rc={proc.returncode}): {tail}"]
        manifest = json.loads(lanes_path.read_text(encoding="utf-8"))

    lanes = {n: entry["lane"] for n, entry in manifest.items()}
    violations = []
    unmarked = sorted(n for n, lane in lanes.items() if lane == "unmarked")
    double = sorted(n for n, lane in lanes.items() if lane == "both")
    if unmarked:
        violations.append(
            f"{len(unmarked)} in-scope test(s) carry NO benchmark-lane marker — add "
            "@pytest.mark.agentic(tier=..., title=..., asks=..., passes_if=...) or "
            "@pytest.mark.harness_benchmark (or add the file to excluded_files in "
            "matrix_e2e_config.json):"
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
        "--print-run-args",
        action="store_true",
        help="print the board's pytest selection (exclusion --ignores, -m agentic, reruns "
        "off) to stdout, one token per line, for the runner to consume",
    )
    ap.add_argument(
        "--check-lanes",
        action="store_true",
        help="collect the in-scope suite and GATE (exit 1) on tests missing a "
        "benchmark-lane marker (agentic/harness_benchmark)",
    )
    ap.add_argument(
        "--check-board-images",
        action="store_true",
        help="GATE (exit 1) on the images the in-scope tests need and never build: "
        "each must exist, built at this checkout's revision",
    )
    ap.add_argument(
        "--record-board-images",
        action="store_true",
        help="record the images just built, with this checkout's revision",
    )
    ap.add_argument(
        "--select",
        default="",
        help="with --check-board-images: the smoke run's -k expression; the gate applies "
        "only when it selects a test that needs the images",
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

    # Tripwire 1: stale exclusions (configured to exclude a file that's gone).
    warnings: list[str] = [
        f"stale exclusion in config — file does not exist: {path}"
        for path in excluded
        if not (root / path).exists()
    ]

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

    if args.record_board_images:
        missing = record_board_images(root, cfg.get("board_images", []), BOARD_IMAGES_STAMP)
        for image in missing:
            print(f"IMAGE-GATE: image {image} was not built", file=sys.stderr)
        return 1 if missing else 0

    if args.check_board_images and (
        not args.select or selection_needs_images(root, cfg, args.select)
    ):
        problems = check_board_images(root, cfg.get("board_images", []), BOARD_IMAGES_STAMP)
        for problem in problems:
            print(f"IMAGE-GATE: {problem}", file=sys.stderr)
        if problems:
            return 1

    if args.print_run_args:
        for token in run_args(excluded):
            print(token)

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
