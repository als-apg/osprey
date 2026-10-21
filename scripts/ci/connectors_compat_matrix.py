#!/usr/bin/env python3
"""Which framework releases the connectors compatibility lanes run against.

``osprey-connectors`` releases on its own tags, and every framework release
pins it with a floor and no ceiling, so a rebuild of a released framework
resolves the newest connectors. Two lanes hold that up:

* **Lane A** (``connectors-compat`` in ci.yml, and the nightly
  ``connectors-compat-nightly.yml``) runs the connectors built at HEAD under
  each framework release listed in ``[tool.osprey-connectors.compat]
  frameworks`` of ``packages/osprey-connectors/pyproject.toml``.
* **Lane B** (``framework-at-connectors-floor``) runs the framework at HEAD
  with the connectors release its ``osprey-connectors>=`` floor names.

Both start out with nothing to run. No framework release is supported yet
(the 2026.9.0 betas, the only ones that depend on this distribution, are
superseded at the cut-over), and the floor still names a connectors build
from before the distribution released on its own, which HEAD has outgrown.
Until :data:`FIRST_INDEPENDENT_CONNECTORS` is the floor, Lane B stands down,
and until a release is listed, Lane A does; each says so in its log and
succeeds. :func:`problems` refuses the one inconsistent state, a floor at or
past the first independent release with no framework listed, so the gates
cannot outlive the cut-over.

Usage::

    python3 scripts/ci/connectors_compat_matrix.py github-output   # the plan job
    python3 scripts/ci/connectors_compat_matrix.py matrix
    python3 scripts/ci/connectors_compat_matrix.py floor
    python3 scripts/ci/connectors_compat_matrix.py check

Standard library only, so the plan job runs it with the runner's ``python3``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tomllib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT_PYPROJECT = REPO_ROOT / "pyproject.toml"
CONNECTORS_PYPROJECT = REPO_ROOT / "packages" / "osprey-connectors" / "pyproject.toml"

#: The first ``osprey-connectors`` released from its own tag. A floor below it
#: names a build cut from the framework's tags, older than anything HEAD of
#: the framework is written against.
FIRST_INDEPENDENT_CONNECTORS = "2026.10.0"

#: What Lane A's single cell prints when no framework release is listed.
NO_FRAMEWORK_LISTED = "no supported framework release listed yet"

#: The release forms this repository tags: ``YYYY.M.P``, optionally a
#: pre-release (``a``/``b``/``rc`` + n), a ``.postN`` and a ``.devN``.
_VERSION_RE = re.compile(
    r"^(?P<release>\d+(?:\.\d+)*)"
    r"(?:(?P<pre>a|b|rc)(?P<pre_n>\d+))?"
    r"(?:\.post(?P<post>\d+))?"
    r"(?:\.dev(?P<dev>\d+))?$"
)
_PRE_ORDER = {"a": 0, "b": 1, "rc": 2}
_FLOOR_RE = re.compile(r"^osprey-connectors\s*>=\s*(?P<floor>[^,;\s]+)\s*$")


def version_key(version: str) -> tuple[Any, ...]:
    """A sort key for ``version`` that orders as PEP 440 does for these forms.

    Raises:
        ValueError: ``version`` is not one of the forms this repository tags.
    """
    match = _VERSION_RE.match(version)
    if match is None:
        raise ValueError(f"not a release version: {version!r}")
    release = tuple(int(part) for part in match["release"].split("."))
    release += (0,) * (3 - len(release))
    dev = int(match["dev"]) if match["dev"] is not None else None
    if match["pre"] is not None:
        pre: tuple[int, int] = (_PRE_ORDER[match["pre"]], int(match["pre_n"]))
    elif dev is not None and match["post"] is None:
        pre = (-1, 0)  # 1.0.dev1 sorts before 1.0a1
    else:
        pre = (3, 0)  # a final release sorts after its pre-releases
    post = int(match["post"]) if match["post"] is not None else -1
    return (release, pre, post, dev if dev is not None else float("inf"))


def _read(path: Path) -> Mapping[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def supported_frameworks(connectors: Mapping[str, Any] | None = None) -> list[str]:
    """The framework releases Lane A runs against, as listed.

    Raises:
        ValueError: The list is missing, holds a non-version, or repeats one.
    """
    document = connectors if connectors is not None else _read(CONNECTORS_PYPROJECT)
    try:
        listed = document["tool"]["osprey-connectors"]["compat"]["frameworks"]
    except KeyError as error:
        raise ValueError("[tool.osprey-connectors.compat] frameworks is not declared") from error
    if not isinstance(listed, list) or not all(isinstance(item, str) for item in listed):
        raise ValueError(f"frameworks must be a list of version strings, not {listed!r}")
    for version in listed:
        version_key(version)
    if len(set(listed)) != len(listed):
        raise ValueError(f"frameworks lists a release twice: {listed}")
    return list(listed)


def connectors_floor(root: Mapping[str, Any] | None = None) -> str:
    """The version the framework's ``osprey-connectors>=`` dependency names.

    Raises:
        ValueError: The dependency is absent, or is not a bare floor.
    """
    document = root if root is not None else _read(ROOT_PYPROJECT)
    floors = [
        match["floor"]
        for requirement in document["project"]["dependencies"]
        if (match := _FLOOR_RE.match(requirement.strip()))
    ]
    if len(floors) != 1:
        raise ValueError(f"expected one 'osprey-connectors>=' dependency, found {len(floors)}")
    version_key(floors[0])
    return floors[0]


def floor_applicable(floor: str) -> bool:
    """Whether the floor names an independently released connectors."""
    return version_key(floor) >= version_key(FIRST_INDEPENDENT_CONNECTORS)


def matrix(frameworks: Iterable[str]) -> dict[str, list[dict[str, str]]]:
    """Lane A's ``strategy.matrix``: one cell per release, or one empty cell.

    GitHub refuses a matrix with no cell, so an empty list becomes a single
    cell whose ``framework`` is empty, the one the lane's stand-down step
    reads.
    """
    cells = [{"framework": version} for version in frameworks]
    return {"include": cells or [{"framework": ""}]}


def problems(frameworks: list[str], floor: str) -> list[str]:
    """Why the declared state is inconsistent, or nothing."""
    found = []
    if floor_applicable(floor) and not frameworks:
        found.append(
            f"the connectors floor {floor} is an independent release (>= "
            f"{FIRST_INDEPENDENT_CONNECTORS}) but no framework release is listed in "
            "[tool.osprey-connectors.compat] frameworks: list the cut-over release"
        )
    return found


def _github_output(frameworks: list[str], floor: str) -> str:
    applicable = floor_applicable(floor)
    return "\n".join(
        [
            f"matrix={json.dumps(matrix(frameworks), separators=(',', ':'))}",
            f"listed={'true' if frameworks else 'false'}",
            f"floor={floor}",
            f"floor-applicable={'true' if applicable else 'false'}",
        ]
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("github-output", "matrix", "floor", "check"))
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        frameworks = supported_frameworks()
        floor = connectors_floor()
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    found = problems(frameworks, floor)
    for problem in found:
        print(f"error: {problem}", file=sys.stderr)
    if found:
        return 1

    if args.command == "matrix":
        print(json.dumps(matrix(frameworks)))
    elif args.command == "floor":
        print(floor)
    elif args.command == "github-output":
        text = _github_output(frameworks, floor)
        target = os.environ.get("GITHUB_OUTPUT")
        if target:
            with open(target, "a", encoding="utf-8") as handle:
                handle.write(text + "\n")
        print(text)
        print(f"frameworks listed: {', '.join(frameworks) or NO_FRAMEWORK_LISTED}", file=sys.stderr)
        if not floor_applicable(floor):
            print(
                f"connectors floor {floor} predates the first independent release "
                f"{FIRST_INDEPENDENT_CONNECTORS}: the floor lane stands down",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
