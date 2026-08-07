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

Three conditions, all required:

* pytest itself exited 0 -- nothing failed or errored;
* nothing skipped -- pcaspy loaded and every live class ran;
* something passed -- an empty or fully deselected run is not a green run.

Run it directly (``scripts/va/live_ca/run_live_ca.sh``) rather than invoking
pytest by hand in the container, or the second and third conditions go
unchecked. On a developer's Mac, run the suites with the worktree's own pytest
instead: the skip there is expected and this gate would (correctly) reject it.
"""

from __future__ import annotations

import sys

import pytest

#: The suites whose assertions are only observable over a real CA wire.
LIVE_SUITES = (
    "tests/va/test_record_factory.py",
    "tests/va/test_apply_fault.py",
)

#: ``-o addopts=`` drops the repo's default ``-v``; ``-p no:cacheprovider``
#: keeps pytest from writing to /work, which is mounted read-only. ``-ra``
#: reports every non-passing outcome including skip reasons, so a rejected run
#: says *why* it was empty instead of just reporting a count.
PYTEST_ARGS = ("-q", "-ra", "--tb=short", "-p", "no:cacheprovider", "-o", "addopts=")


class _Tally:
    """Capture the run's outcome counts from the terminal reporter."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    def pytest_terminal_summary(self, terminalreporter: object) -> None:
        stats = terminalreporter.stats  # type: ignore[attr-defined]
        # The empty-string key holds reports with no outcome of interest.
        self.counts = {outcome: len(reports) for outcome, reports in stats.items() if outcome}


def main(argv: list[str]) -> int:
    """Run the suites and return the gate's exit status."""
    targets = argv or list(LIVE_SUITES)
    tally = _Tally()
    status = int(pytest.main([*targets, *PYTEST_ARGS], plugins=[tally]))

    passed = tally.counts.get("passed", 0)
    skipped = tally.counts.get("skipped", 0)
    failed = tally.counts.get("failed", 0)
    errors = tally.counts.get("error", 0)

    print()
    print("=" * 72)
    print("live Channel Access gate")
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

    if passed == 0:
        print("  VERDICT: FAIL -- nothing ran. An empty run is not a green run.")
        print("=" * 72)
        return 1

    print(f"  VERDICT: PASS -- {passed} live Channel Access test(s) ran, none skipped.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
