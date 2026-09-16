#!/usr/bin/env python3
"""The type check scored against a written-down set of errors, over the declared trees.

A checker whose exit status nothing reads reports to no one. This gate gives the
type check a verdict the build can fail on, and it holds two things while doing so.

**The tree is checked against a written-down error set.** ``scripts/mypy_baseline.json``
enumerates every error the tree reports today. A run is compared against it, and only an
error the baseline does not carry fails the gate. The invariant is not that the count is
zero — it is that the count is recorded and that no change may raise it. An error that
exists is a known error; an error that arrives unannounced is the failure mode. Errors
the baseline lists that a run no longer reports are *stale*, and stale is not a failure:
a gate that reds on an improvement teaches people to stop improving.

The baseline is keyed on ``(path, code, message)`` and never on a line number. An edit
anywhere above an error moves its line, and a baseline keyed on lines would be invalid
after any change at all. The message text carries the symbol and the types, which is what
identifies the error; the count is a multiset, so an error traded for a different one in
the same file is still an addition and still fails.

**The trees checked in the build are the trees ``[tool.mypy] files`` names.** The gate
reads that list and passes it to the checker explicitly, so the build's targets and a
bare local ``mypy``'s targets are one list rather than two that a test has to keep equal.

Usage::

    uv run python scripts/mypy_gate.py            # score this tree against the baseline
    uv run python scripts/mypy_gate.py --update   # rewrite the baseline from this run

Two conditions are refused rather than scored, both exiting ``2``, because a run taken
under different conditions is a different measurement: an environment missing the stub
distributions the ``dev`` extra declares, and a checker that exits with anything but
``0`` or ``1``.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess  # noqa: S404 - replaced wholesale by the tests; see main()
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
BASELINE = REPO_ROOT / "scripts" / "mypy_baseline.json"

REFRESH_COMMAND = "uv run python scripts/mypy_gate.py --update"
SYNC_COMMAND = "uv sync --extra dev"

#: ``path:line: error: message`` and ``path:line:column: error: message``.
_DIAGNOSTIC = re.compile(r"^(?P<path>[^\s:]+):\d+(?::\d+)?: (?P<severity>[a-z]+): (?P<rest>.*)$")

#: mypy appends the error code as ``  [code]`` — two spaces, lowercase, hyphenated.
_CODE = re.compile(r"^(?P<message>.*?)\s{2}\[(?P<code>[a-z][a-z0-9-]*)\]$")

Error = tuple[str, str, str]


def declared_targets(pyproject_path: Path) -> list[str]:
    """The trees ``[tool.mypy] files`` names, in the order it names them."""
    with pyproject_path.open("rb") as handle:
        pyproject = tomllib.load(handle)
    return list(pyproject["tool"]["mypy"]["files"])


def parse_errors(output: str) -> list[Error]:
    """Every ``error`` diagnostic in *output*, as ``(path, code, message)``.

    ``note`` lines and the line and column numbers are discarded. An error mypy emits
    without a code is recorded with the code ``""`` rather than dropped.
    """
    errors: list[Error] = []
    for line in output.splitlines():
        match = _DIAGNOSTIC.match(line)
        if match is None or match["severity"] != "error":
            continue
        rest = match["rest"]
        coded = _CODE.match(rest)
        if coded is None:
            errors.append((match["path"], "", rest.strip()))
        else:
            errors.append((match["path"], coded["code"], coded["message"].strip()))
    return errors


def tally(errors: Sequence[Error]) -> collections.Counter[Error]:
    """Count the errors as a multiset keyed on ``(path, code, message)``."""
    return collections.Counter(errors)


def describe(error: Error, count: int = 1) -> str:
    """One error rendered for a human: ``path: [code] message``."""
    path, code, message = error
    rendered = f"{path}: [{code}] {message}" if code else f"{path}: {message}"
    return f"{rendered}  (x{count})" if count > 1 else rendered


def compare(
    current: collections.Counter[Error], baseline: collections.Counter[Error]
) -> tuple[list[str], list[str]]:
    """``(added, stale)`` — what the run reports and the baseline does not, and the reverse.

    ``added`` is the failure. ``stale`` is a notice: a run may legitimately report fewer
    errors than the baseline, on another platform or after an unrelated fix.
    """
    added = [describe(error, count) for error, count in sorted((current - baseline).items())]
    stale = [describe(error, count) for error, count in sorted((baseline - current).items())]
    return added, stale


def load_baseline(path: Path) -> collections.Counter[Error]:
    """Read a baseline file back into a multiset. ``total`` is derived, so it is ignored."""
    document = json.loads(path.read_text(encoding="utf-8"))
    counted: collections.Counter[Error] = collections.Counter()
    for entry in document["errors"]:
        counted[(entry["path"], entry["code"], entry["message"])] += int(entry.get("count", 1))
    return counted


def write_baseline(path: Path, counted: collections.Counter[Error]) -> None:
    """Write *counted* as the baseline, sorted so a diff of the file reads as a diff of the tree."""
    errors = [
        {"path": error[0], "code": error[1], "message": error[2], "count": count}
        for error, count in sorted(counted.items())
    ]
    document = {"total": sum(counted.values()), "errors": errors}
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def _missing_stubs(errors: Sequence[Error]) -> list[str]:
    """The stub distributions the run went without, named by the imports that wanted them."""
    return [
        message
        for _path, code, message in errors
        if code == "import-untyped" and message.startswith("Library stubs not installed for")
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score the type check against its baseline.")
    parser.add_argument(
        "--update",
        action="store_true",
        help="rewrite the baseline from this run instead of scoring against it",
    )
    args = parser.parse_args(argv)

    # `subprocess` is read off this module so a test can replace the attribute rather
    # than the process-global `subprocess.run`.
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "mypy", *declared_targets(PYPROJECT), "--no-error-summary"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    errors = parse_errors(completed.stdout)

    missing = _missing_stubs(errors)
    if missing:
        print("The type check ran without the stub distributions the `dev` extra declares.")
        for message in sorted(set(missing)):
            print(f"  {message}")
        print(f"Its report is not the measurement the baseline holds. Fix: {SYNC_COMMAND}")
        return 2

    if completed.returncode not in {0, 1}:
        print(f"mypy exited {completed.returncode}; nothing was scored.")
        print(completed.stderr.rstrip())
        return 2

    current = tally(errors)

    if args.update:
        write_baseline(BASELINE, current)
        total = sum(current.values())
        print(f"Wrote {BASELINE.relative_to(REPO_ROOT)}: {total} errors.")
        return 0

    added, stale = compare(current, load_baseline(BASELINE))

    if added:
        print(f"The type check reports {len(added)} error(s) the baseline does not carry:")
        for line in added:
            print(f"  {line}")
        print(f"Fix them, or record them deliberately with: {REFRESH_COMMAND}")
        return 1

    if stale:
        print(f"{len(stale)} baseline error(s) are no longer reported:")
        for line in stale:
            print(f"  {line}")
        print(f"Drop them from the baseline with: {REFRESH_COMMAND}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
