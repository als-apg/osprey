#!/usr/bin/env python3
"""Fail a CI lane whose junit reports show a skipped test, or no test at all.

pytest exits 0 when every test it collected skipped, so a lane whose tests all
skip for want of a runtime, an extra or a key reports a green check over
nothing. ci.yml's skip-gate steps ("Fail the lane on any skipped test") run
this on the reports their pytest steps write with ``--junitxml=`` instead of
trusting pytest's exit code.

A report fails the lane when it is missing or unreadable, when it collected no
tests, or when any of its tests skipped. ``--allow-partial-skips`` is the
lane-wide gate of ``privilege-split-e2e``, whose audit module carries documented
per-test guards: a report fails only when every one of its tests skipped, and a
partial skip is a warning.

Usage::

    uv run python scripts/ci/junit_no_skips.py REPORT [REPORT ...] \\
        [--allow-partial-skips] [--message TEXT]

Every report is checked and its counts printed, with a GitHub ``::error::`` or
``::warning::`` annotation per finding. Exits 1 if any report failed; otherwise
prints ``--message`` and exits 0. Stdlib-only.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

DEFAULT_MESSAGE = "every test ran; none skipped"


def counts(report: Path) -> tuple[int, int]:
    """Return ``(tests, skipped)`` summed over every ``testsuite`` in *report*."""
    total = skipped = 0
    for suite in ET.parse(report).getroot().iter("testsuite"):
        total += int(suite.get("tests", 0))
        skipped += int(suite.get("skipped", 0))
    return total, skipped


def check(report: Path, *, allow_partial_skips: bool = False) -> bool:
    """Print *report*'s counts and annotations; return whether it passes."""
    if not report.exists():
        print(f"::error::{report} was never written — the step that writes it did not finish")
        return False
    try:
        total, skipped = counts(report)
    except (ET.ParseError, OSError, ValueError) as exc:
        print(f"::error::{report} is not a readable junit report: {exc}")
        return False
    print(f"{report}: {total} test(s), {skipped} skipped")
    if total == 0:
        print(f"::error::{report} collected no tests")
        return False
    if not skipped:
        return True
    if not allow_partial_skips:
        print(f"::error::{report} reports {skipped} skipped test(s)")
        return False
    if skipped >= total:
        print(f"::error::{report} skipped every one of its {total} test(s)")
        return False
    print(f"::warning::{report} skipped {skipped} of {total} test(s)")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reports", nargs="+", type=Path, help="junit XML reports to read")
    parser.add_argument(
        "--allow-partial-skips",
        action="store_true",
        help="fail only a report whose every test skipped; warn on a partial skip",
    )
    parser.add_argument(
        "--message", default=DEFAULT_MESSAGE, help="printed when every report passes"
    )
    args = parser.parse_args(argv)
    # A list, not a generator: every report is checked and reported, not just the first failure.
    results = [check(r, allow_partial_skips=args.allow_partial_skips) for r in args.reports]
    if not all(results):
        return 1
    print(args.message)
    return 0


if __name__ == "__main__":
    sys.exit(main())
