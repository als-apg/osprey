#!/usr/bin/env python3
"""Turn a CI diagnostics directory into a one-screen answer.

``tests/ci_diagnostics.py`` writes an event log per xdist worker while the suite
runs. When a lane is killed rather than failing — a step timeout, a runner OOM,
a wedged worker — those logs hold the one fact the job log cannot show: which
test each worker was inside at the moment everything stopped.

This reads them and says so, in the run's step summary, so nobody has to
download the artifact to learn whether a freeze was one stuck test or the whole
runner going away.

The lane also tees pytest's own output into the same directory as
``pytest.log``. When that file holds a ``--durations`` report, its top entries
are rendered here too: the ranking is the one fact that tells "one group of
tests got expensive" apart from "the whole suite is uniformly slower", and the
job log it would otherwise live in is truncated by ``gh run view --log`` to a
fraction of its length, so the table was unreachable from the CLI.

Deliberately standalone — no imports from ``tests/`` — so the reader cannot be
broken by the writer, and so it still runs in a job that never installed the
project. Usage::

    python scripts/ci/diag_summary.py [ci-diag-dir]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def _events(path: Path) -> list[dict]:
    """Parse one JSONL log, skipping records a signal cut in half.

    A killed process is routinely interrupted mid-``write``, so a malformed
    final line is the normal case, not corruption. Dropping the report over it
    would discard the diagnosis this file exists to carry.
    """
    records = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return records
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def stalled_tests(directory: Path | str) -> dict[str, str]:
    """Map worker -> the test it started and never finished.

    Workers that finished everything they started are absent, so an empty result
    means no test was in flight — the process died between tests, or exited
    cleanly and something else failed the job.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return {}

    stalled: dict[str, str] = {}
    for path in sorted(directory.glob("events-*.jsonl")):
        worker = path.stem.removeprefix("events-")
        in_flight: str | None = None
        for record in _events(path):
            event = record.get("event")
            if event == "start":
                in_flight = record.get("nodeid")
            elif event in ("finish", "session_end"):
                in_flight = None
        if in_flight:
            stalled[worker] = in_flight
    return stalled


PYTEST_LOG = "pytest.log"
"""File name the lane tees pytest's output into, inside the diagnostics dir."""

SLOWEST_ROWS = 15
"""Rows of the durations report shown in the summary. The full report stays in
``pytest.log`` in the uploaded artifact; the summary is a screen, not a log."""

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_DURATIONS_HEADER = re.compile(r"slowest \d+ durations")
_DURATION_ROW = re.compile(r"^\s*(\d+(?:\.\d+)?)s\s+(call|setup|teardown)\s+(\S.*?)\s*$")


def slowest_tests(directory: Path | str) -> list[tuple[float, str, str]] | None:
    """The ``--durations`` report from ``pytest.log`` as (seconds, phase, nodeid).

    ``None`` when there is no log to read; an empty list when the log exists
    but never reached the report (the lane was killed first). pytest writes
    the report with ``--color=yes`` escapes on its rule lines, so the text is
    stripped of ANSI sequences before matching.
    """
    path = Path(directory) / PYTEST_LOG
    try:
        text = _ANSI.sub("", path.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return None
    rows: list[tuple[float, str, str]] = []
    in_report = False
    for line in text.splitlines():
        if not in_report:
            in_report = bool(_DURATIONS_HEADER.search(line))
            continue
        match = _DURATION_ROW.match(line)
        if match is None:
            break
        rows.append((float(match.group(1)), match.group(2), match.group(3)))
    return rows


def _counts(directory: Path) -> dict[str, int]:
    return {
        path.stem.removeprefix("events-"): sum(
            1 for record in _events(path) if record.get("event") == "finish"
        )
        for path in sorted(directory.glob("events-*.jsonl"))
    }


def render_report(directory: Path | str) -> str:
    """Markdown for ``$GITHUB_STEP_SUMMARY``."""
    directory = Path(directory)
    stalled = stalled_tests(directory)
    completed = _counts(directory)

    lines = ["### Test lane diagnostics", ""]

    if not completed:
        lines.append("No diagnostics were recorded (the lane never started the suite).")
        return "\n".join(lines) + "\n"

    if stalled:
        # Every worker stopping at once points at the runner rather than at any
        # one test; a single stalled worker points at the test it names.
        scope = "all workers" if len(stalled) == len(completed) else "some workers"
        lines += [
            f"**In flight when the lane stopped ({scope}):**",
            "",
            "| Worker | Test | Completed before it |",
            "| --- | --- | --- |",
        ]
        lines += [
            f"| `{worker}` | `{nodeid}` | {completed.get(worker, 0)} |"
            for worker, nodeid in sorted(stalled.items())
        ]
        if len(stalled) == len(completed) and len(stalled) > 1:
            lines += [
                "",
                "Every worker stopped with a test in flight — consistent with the "
                "runner stalling or being killed, not with one deadlocked test. "
                "Check `runner-state.txt` and the container `OOMKilled` flags.",
            ]
    else:
        lines.append("No stalled tests: every started test also finished.")

    stacks = sorted(directory.glob("stacks-*.txt"))
    if stacks:
        lines += ["", f"**Stack dumps captured:** {', '.join(p.name for p in stacks)}"]

    slowest = slowest_tests(directory)
    if slowest:
        lines += [
            "",
            f"**Slowest tests (top {min(SLOWEST_ROWS, len(slowest))} of {len(slowest)} "
            f"reported; the full report is `{PYTEST_LOG}` in the artifact):**",
            "",
            "| Seconds | Phase | Test |",
            "| ---: | --- | --- |",
        ]
        lines += [
            f"| {seconds:.1f} | {phase} | `{nodeid}` |"
            for seconds, phase, nodeid in slowest[:SLOWEST_ROWS]
        ]
    elif slowest is not None:
        lines += [
            "",
            f"No durations report in `{PYTEST_LOG}`: the suite never reached its final summary.",
        ]

    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    directory = Path(argv[1]) if len(argv) > 1 else Path("ci-diag")
    sys.stdout.write(render_report(directory))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
