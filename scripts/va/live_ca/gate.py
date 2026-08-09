#!/usr/bin/env python3
"""Run the live Channel Access suites and fail unless they really ran.

The suites this drives guard their pcaspy import with ``pytest.importorskip``.
That is the right behaviour on a host that cannot load pcaspy -- an honest,
loud skip beats a hollow pass -- but it means the ordinary pytest exit code
cannot distinguish "the Channel Access contract holds" from "nothing was
exercised at all". Both are 0.

This script closes that gap. It is the container image's entry point, and the
container is by construction the one venue where a skip is not an acceptable
outcome: it is linux/amd64 with the ``virtual-accelerator`` extra installed
from the repo's own lockfile, so pcaspy is present or the image is broken.

The gate is deliberately *not* a text match on pytest's summary line. It reads
the terminal reporter's own outcome counts, so it cannot be fooled by a
formatting change, by ``-q`` versus ``-v``, or by a colour code landing in the
middle of the word it was grepping for.

Three conditions, all required, applied to every module and to the total:

* pytest itself exited 0 -- nothing failed or errored;
* nothing skipped -- pcaspy loaded and every live class ran;
* something passed -- an empty or fully deselected run is not a green run.

**Each module runs in a process of its own.** This is the one thing about the
gate's shape that is not obvious, so: the live modules each pick a Channel
Access port at import time with
``os.environ.setdefault("EPICS_CA_SERVER_PORT", _free_port())``. ``setdefault``
means the FIRST module imported in a process wins, and every later one inherits
its port -- so in a single combined pytest run, the second module stands a
second ``pcaspy.SimpleServer`` up on a port the process-wide libca client has
already latched onto the first server. The observed symptom was every test in
the second module erroring at fixture setup with "the Channel Access server
never became reachable", intermittently: twice in roughly twenty-seven combined
runs, while the same module alone in a fresh process went twenty for twenty.

Running one module per process is what makes that ``setdefault`` do what it
looks like it does -- each module gets its own free port, its own server and
its own libca -- and it is also the shape this repo's Channel Access notes
already point at (one server per process, unique PV names, no ``clear_cache``).

Counts still come from pytest's own terminal reporter rather than its printed
summary. Each child is this same file re-entered with ``--run-module``, which
runs ``pytest.main`` under the tally plugin and prints one machine-readable
sentinel line; the parent sums those. A child that prints no sentinel is
treated as a failure, so a hard crash cannot be mistaken for a zero-count pass.

Run it directly (``scripts/va/live_ca/run_live_ca.sh``) rather than invoking
pytest by hand in the container, or the second and third conditions go
unchecked. On a developer's Mac, run the suites with the worktree's own pytest
instead: the skip there is expected and this gate would (correctly) reject it.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys

import pytest

#: The suites whose assertions are only observable over a real CA wire.
LIVE_SUITES = (
    "tests/va/test_record_factory.py",
    "tests/va/test_apply_fault.py",
)

#: Added in --pva mode. Its served-boot branch needs the PVA transport and the
#: serving package on top of pcaspy.
PVA_SUITES = ("tests/va/test_facility_seam.py",)

#: Modules that must import in --pva mode. This is a precondition, not a
#: preference, and it is checked *before* pytest runs.
#:
#: test_facility_seam.py's boot test passes on either of two outcomes: the boot
#: served, or it stopped on a missing server extension. Both are legitimate --
#: on a host lacking the transport, stopping there is the honest result -- so
#: a green tells you nothing about which branch ran, and the served branch is
#: the one that had never executed. Asserting these import first is what makes
#: the fallback unreachable, and therefore what makes the green mean "the
#: served path ran". Without it, --pva mode would quietly re-certify the same
#: fallback the CA-only venue already covers.
PVA_REQUIRED_MODULES = ("pcaspy", "p4p", "lume_pva_apg")

#: ``-o addopts=`` drops the repo's default ``-v``; ``-p no:cacheprovider``
#: keeps pytest from writing to /work, which is mounted read-only. ``-ra``
#: reports every non-passing outcome including skip reasons, so a rejected run
#: says *why* it was empty instead of just reporting a count.
PYTEST_ARGS = ("-q", "-ra", "--tb=short", "-p", "no:cacheprovider", "-o", "addopts=")

#: Internal flag: run exactly one module and report its counts. Not part of the
#: gate's interface -- ``main`` re-enters this file with it, once per module.
RUN_ONE_FLAG = "--run-module"

#: Prefix of the one line a child writes for its parent to read. Distinctive
#: enough that no pytest output or test print can be mistaken for it.
COUNTS_SENTINEL = "__GATE_COUNTS__ "

#: The outcomes the verdict is built from. Named so a module that reports none
#: of them still produces a full, explicit zero row rather than a sparse dict.
OUTCOMES = ("passed", "skipped", "failed", "error")


class _Tally:
    """Capture the run's outcome counts from the terminal reporter."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    def pytest_terminal_summary(self, terminalreporter: object) -> None:
        stats = terminalreporter.stats  # type: ignore[attr-defined]
        # The empty-string key holds reports with no outcome of interest.
        self.counts = {outcome: len(reports) for outcome, reports in stats.items() if outcome}


def _check_required_modules(modules: tuple[str, ...]) -> list[str]:
    """Return the subset of ``modules`` that cannot be imported."""
    missing = []
    for name in modules:
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)
    return missing


def _run_one(target: str) -> int:
    """Child entry point: run one module and print its counts for the parent.

    The tally plugin reads the terminal reporter directly, so the numbers the
    parent sums are pytest's own -- crossing a process boundary does not
    downgrade them to a summary-line scrape.
    """
    tally = _Tally()
    status = int(pytest.main([target, *PYTEST_ARGS], plugins=[tally]))
    counts = {outcome: tally.counts.get(outcome, 0) for outcome in OUTCOMES}
    print(f"{COUNTS_SENTINEL}{json.dumps(counts)}")
    return status


def _run_module(target: str) -> tuple[dict[str, int], int]:
    """Run one module in a fresh process; return its counts and exit status.

    A child that dies without printing the sentinel -- a segfault in libca, an
    OOM kill, an import that took the interpreter down -- is reported as one
    error rather than as zeroes, because zeroes here would read as "nothing
    went wrong" to the caller summing them.
    """
    proc = subprocess.run(
        [sys.executable, "-u", __file__, RUN_ONE_FLAG, target],
        capture_output=True,
        text=True,
    )
    counts: dict[str, int] | None = None
    for line in proc.stdout.splitlines():
        if line.startswith(COUNTS_SENTINEL):
            # Defensive only: the child writes this line itself, and pytest's
            # fd-level capture keeps test output from interleaving into it. But
            # a mangled sentinel should fall through to the honest "no counts"
            # report below -- a JSON traceback from the gate would be a worse
            # description of what went wrong than the message that branch gives.
            try:
                parsed = json.loads(line[len(COUNTS_SENTINEL) :])
                counts = {outcome: int(parsed[outcome]) for outcome in OUTCOMES}
            except (ValueError, TypeError, KeyError):
                counts = None
        else:
            print(line)
    if proc.stderr.strip():
        print(proc.stderr, end="" if proc.stderr.endswith("\n") else "\n")

    if counts is None:
        print(f"  !! {target} produced no counts -- the child died before reporting.")
        return ({"passed": 0, "skipped": 0, "failed": 0, "error": 1}, proc.returncode or 1)
    return (counts, proc.returncode)


def main(argv: list[str]) -> int:
    """Run the suites and return the gate's exit status."""
    pva = "--pva" in argv
    argv = [arg for arg in argv if arg != "--pva"]

    # Every remaining argument is fanned out as its own module path, so a
    # pytest flag would become a target: `-k foo` would hand one child `-k` and
    # another `foo`, and pytest exits 4 on the usage error while its siblings
    # run UNFILTERED -- the opposite of what the flag asked for, reported as a
    # gate failure. Reject them by name instead of guessing an intent.
    flags = [arg for arg in argv if arg.startswith("-")]
    if flags:
        print(f"gate.py takes module paths only; got {', '.join(flags)}.")
        print("pytest flags are not passed through: each argument becomes its own")
        print("pytest process, so a flag would be run as a test path while the")
        print("other modules ran unfiltered. To filter, invoke pytest directly")
        print("inside the container -- but note the gate's skip and per-module")
        print("checks do not apply to a run it did not drive.")
        return 1

    if pva:
        missing = _check_required_modules(PVA_REQUIRED_MODULES)
        if missing:
            print("=" * 72)
            print("live Channel Access gate (--pva)")
            print(f"  VERDICT: FAIL -- cannot import {', '.join(missing)}.")
            print("           The served-boot branch cannot run without these, and the")
            print("           seam test PASSES on the fallback branch instead, so")
            print("           continuing would certify the served path without")
            print("           exercising it. The `virtual-accelerator` extra installs")
            print("           all three, so this image is not what it claims to be.")
            print("=" * 72)
            return 1

    targets = argv or [*LIVE_SUITES, *(PVA_SUITES if pva else ())]

    # One process per target, and every target is run even after one fails:
    # the point of the venue is to say what the whole contract did, and
    # stopping early would hide a second module's state behind the first's.
    totals = dict.fromkeys(OUTCOMES, 0)
    status = 0
    silent: list[str] = []
    for target in targets:
        print(f"--- {target} ---")
        counts, target_status = _run_module(target)
        for outcome in OUTCOMES:
            totals[outcome] += counts.get(outcome, 0)
        status = status or target_status
        if counts["passed"] == 0:
            silent.append(target)
        print(
            f"  {target}: passed={counts['passed']} skipped={counts['skipped']} "
            f"failed={counts['failed']} errors={counts['error']} exit={target_status}"
        )

    passed = totals["passed"]
    skipped = totals["skipped"]
    failed = totals["failed"]
    errors = totals["error"]

    print()
    print("=" * 72)
    print(f"live Channel Access gate{' (--pva)' if pva else ''}")
    print(
        f"  passed={passed} skipped={skipped} failed={failed} errors={errors} pytest_exit={status}"
    )

    if status != 0:
        print("  VERDICT: FAIL -- the live suites did not pass.")
        print("=" * 72)
        return status

    if skipped:
        print(f"  VERDICT: FAIL -- {skipped} test(s) SKIPPED in the venue that must run them.")
        print("           A skipped live suite proves nothing. The usual cause is")
        print("           pcaspy missing from this image: check that the")
        print("           `virtual-accelerator` extra still carries it and that")
        print("           this really is a linux/amd64 container.")
        print("=" * 72)
        return 1

    # Per module, not just in total: a module that contributed nothing is a
    # module whose contract went unchecked, and the other modules' passes would
    # otherwise carry it to a green aggregate. This also subsumes the old
    # "nothing ran at all" check: `targets` cannot be empty (argv falls back to
    # a non-empty constant, and a flags-only argv is rejected above), so an
    # empty run can only reach here as every module reporting zero passes.
    if silent:
        print(f"  VERDICT: FAIL -- {len(silent)} module(s) ran nothing: {', '.join(silent)}.")
        print("           A module that reports no passes proves nothing about the")
        print("           contract it owns, whatever the other modules did.")
        print("=" * 72)
        return 1

    print(f"  VERDICT: PASS -- {passed} live Channel Access test(s) ran, none skipped.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == RUN_ONE_FLAG:
        sys.exit(_run_one(sys.argv[2]))
    sys.exit(main(sys.argv[1:]))
