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

The same log is read for the lane's per-test timeout, so a test the cap killed
is named in the summary rather than left to be found by scrolling a 60k-line
log (#743). The cap fails the hung test and lets the suite finish, so this is
usually a *red* lane rather than a killed one — but it is still the fact you
want first, and it is the one the durations table cannot show.

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

#: The short-summary line naming a test the cap killed, and the primary
#: evidence here. Matched on pytest-timeout's own ``from pytest-timeout``
#: signature and NOT on the bare word "timeout": a test's own name or assertion
#: text is echoed into this line too, and mislabelling an ordinary failure as a
#: hang sends the next reader after the wrong thing. Any ``[gwN] [ 45%]``
#: prefixes xdist adds are skipped. The node id is closed on whitespace and not
#: on a word boundary: a parametrised id ends in ``]``, and ``\b`` after a
#: non-word character would backtrack into the id and truncate it.
_TIMEOUT_FAILURE = re.compile(
    r"^(?:\[[^\]]*\]\s*)*(?:FAILED|ERROR)\s+(\S+)\s.*from pytest-timeout", re.MULTILINE
)

#: The rule pytest-timeout writes around its stack dump when it fires
#: (``terminal.sep("+", title="Timeout")``), used only as a fallback when the
#: run never reached the summary line above. A fallback and not the primary
#: signal because pytest-timeout emits it only when
#: ``len(threading.enumerate()) > 1``, and an xdist worker's execnet receiver
#: is started with ``_thread``, so it does not count: a hung test with no
#: threads of its own is killed with no banner at all. Verified against
#: pytest-timeout 2.4.0 under ``-n 2 --dist loadgroup``.
_TIMEOUT_BANNER = re.compile(r"\+{3,}\s*Timeout\s*\+{3,}")


def _pytest_log(directory: Path | str) -> str | None:
    """``pytest.log`` with ANSI escapes stripped, or ``None`` if there is none.

    pytest writes its rules with ``--color=yes`` escapes, so every reader below
    wants the stripped text; sharing one reader also means one file read per
    report rather than one per question.
    """
    try:
        text = (Path(directory) / PYTEST_LOG).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return _ANSI.sub("", text)


def timed_out_tests(directory: Path | str) -> list[str] | None:
    """Node ids the lane's per-test cap killed, in the order pytest named them.

    ``None`` when there is no log to read. An empty list when the log exists
    and no test was named — either nothing timed out, or one did and the run
    never reached the short summary that names it, which ``timeout_banner_seen``
    still reports.
    """
    text = _pytest_log(directory)
    if text is None:
        return None
    seen: dict[str, None] = {}
    for match in _TIMEOUT_FAILURE.finditer(text):
        seen.setdefault(match.group(1), None)
    return list(seen)


def timeout_banner_seen(directory: Path | str) -> bool:
    """Whether pytest-timeout's banner is in the log, named test or not.

    The banner is written while the test is still hung; the line that names it
    comes from the final summary. A lane killed in between has the first and
    not the second, and "a timeout fired" is still the answer to why it stopped.
    False is not proof that nothing timed out — see ``_TIMEOUT_BANNER``.
    """
    text = _pytest_log(directory)
    return text is not None and bool(_TIMEOUT_BANNER.search(text))


def slowest_tests(directory: Path | str) -> list[tuple[float, str, str]] | None:
    """The ``--durations`` report from ``pytest.log`` as (seconds, phase, nodeid).

    ``None`` when there is no log to read; an empty list when the log exists
    but never reached the report (the lane was killed first). pytest writes
    the report with ``--color=yes`` escapes on its rule lines, so the text is
    stripped of ANSI sequences before matching.
    """
    text = _pytest_log(directory)
    if text is None:
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

    # Above the stack dumps on purpose: when the per-test cap fired, this names
    # the test outright and the reader needs nothing else.
    timed_out = timed_out_tests(directory)
    if timed_out:
        lines += [
            "",
            "**Killed by the per-test timeout:**",
            "",
            *[f"- `{nodeid}`" for nodeid in timed_out],
        ]
    elif timeout_banner_seen(directory):
        lines += [
            "",
            "A per-test timeout fired, but the run never reached the summary line that "
            f"names the test — the banner is in `{PYTEST_LOG}` and the table above is "
            "the closest thing to a name.",
        ]

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
