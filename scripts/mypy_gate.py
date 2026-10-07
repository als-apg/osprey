#!/usr/bin/env python3
"""The type check over the declared trees, with a verdict the build can fail on.

The type check runs over the trees ``[tool.mypy] files`` names, and the build fails on
any error it reports. No list of tolerated errors sits beside the tree: an error is
fixed where mypy reports it, and a deliberate exception is a ``# type: ignore[code]``
on its own line, where a reviewer sees it.

The targets are read from ``[tool.mypy] files`` and passed to the checker explicitly, so
the build and a bare ``uv run mypy`` check one list of trees rather than two that a test
has to keep equal.

Two conditions are refused rather than judged, both exiting ``2``. A run without the
stub distributions the ``dev`` extra declares is a weaker report, in which every value
from those libraries is ``Any``, and its fix is an environment command, not a code
change. A checker that exits with anything but ``0`` or ``1`` checked nothing.

Usage::

    uv run python scripts/mypy_gate.py

Exit codes: ``0`` clean, ``1`` errors reported, ``2`` refused.
"""

from __future__ import annotations

import argparse
import re
import subprocess  # replaced wholesale by the tests; see main()
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"

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


def _missing_stubs(errors: Sequence[Error]) -> list[str]:
    """The stub distributions the run went without, named by the imports that wanted them."""
    return [
        message
        for _path, code, message in errors
        if code == "import-untyped" and message.startswith("Library stubs not installed for")
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the type check over the declared trees; any error fails."
    )
    parser.parse_args(argv)

    # `subprocess` is read off this module so a test can replace the attribute rather
    # than the process-global `subprocess.run`.
    completed = subprocess.run(
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
        print(f"A report without them is not the type check. Fix: {SYNC_COMMAND}")
        return 2

    if completed.returncode not in {0, 1}:
        print(f"mypy exited {completed.returncode}; nothing was checked.")
        print(completed.stderr.rstrip())
        return 2

    if errors or completed.returncode == 1:
        print(completed.stdout.rstrip())
        if completed.stderr:
            print(completed.stderr.rstrip())
        if errors:
            print(f"The type check reports {len(errors)} error(s); the tree must report none.")
        else:
            print("mypy exited 1 without a located error; the tree must report none.")
        return 1

    print("The type check reports no errors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
