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

Lane A stands down while ``frameworks`` is empty, and Lane B while the floor
predates :data:`FIRST_INDEPENDENT_CONNECTORS`; each says so in its log and
succeeds. :func:`problems` holds the declarations to the tags reachable from
HEAD, so neither stand-down can outlive the releases that end it:

* **R1** The floor is a final release, and from
  :data:`FIRST_INDEPENDENT_CONNECTORS` on its ``osprey-connectors-v`` tag is
  reachable: below a floor this history does not carry, paired wheel
  installs and ``osprey up --dev`` fail.
* **R2** Every framework release tagged after :data:`LAST_PAIRED_FRAMEWORK`
  is listed in ``frameworks`` or in ``retired``, except the newest: its
  publish and the PR that lists it necessarily follow its tag. ``check
  --before-tagging vX.Y.Z`` applies the rule as if that tag were the newest,
  so the release about to be cut may not leave its predecessor unlisted.
* **R3** Every listed and retired release is such a tag, and none is both.

Usage::

    python3 scripts/ci/connectors_compat_matrix.py github-output [--framework X]  # the plan job
    python3 scripts/ci/connectors_compat_matrix.py matrix
    python3 scripts/ci/connectors_compat_matrix.py floor
    python3 scripts/ci/connectors_compat_matrix.py check [--before-tagging vX.Y.Z]
    python3 scripts/ci/connectors_compat_matrix.py view-required <framework version>
    python3 scripts/ci/connectors_compat_matrix.py source-text-tests

Every command checks the declarations first, against ``git tag --merged
HEAD``, and refuses a shallow checkout, whose tags it cannot see. Standard
library only, so the plan job runs it with the runner's ``python3``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
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

#: The last framework release whose connectors were cut from the framework's
#: own tag. Every framework tagged after it pairs with independent connectors.
LAST_PAIRED_FRAMEWORK = "2026.9.0b3"

#: The prefix of an independent connectors release's tag.
CONNECTORS_TAG_PREFIX = "osprey-connectors-v"

#: Why Lane A's single cell stands down; the cell carries it to the log.
NO_FRAMEWORK_LISTED = (
    "no supported framework release listed yet: [tool.osprey-connectors.compat] "
    "frameworks in packages/osprey-connectors/pyproject.toml is empty"
)

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
_FRAMEWORK_TAG_RE = re.compile(r"^v(?P<version>[0-9].*)$")
_NODE_ID_RE = re.compile(r"^tests/[\w/.-]+\.py(?:::[A-Za-z_]\w*)?$")


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


def is_final(version: str) -> bool:
    """Whether ``version`` carries no pre-release or development segment."""
    match = _VERSION_RE.match(version)
    if match is None:
        raise ValueError(f"not a release version: {version!r}")
    return match["pre"] is None and match["dev"] is None


def _read(path: Path) -> Mapping[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _compat_table(connectors: Mapping[str, Any] | None) -> Mapping[str, Any]:
    document = connectors if connectors is not None else _read(CONNECTORS_PYPROJECT)
    try:
        return document["tool"]["osprey-connectors"]["compat"]
    except KeyError as error:
        raise ValueError("[tool.osprey-connectors.compat] frameworks is not declared") from error


def _version_list(table: Mapping[str, Any], key: str) -> list[str]:
    listed = table[key]
    if not isinstance(listed, list) or not all(isinstance(item, str) for item in listed):
        raise ValueError(f"{key} must be a list of version strings, not {listed!r}")
    for version in listed:
        version_key(version)
    if len(set(listed)) != len(listed):
        raise ValueError(f"{key} lists a release twice: {listed}")
    return list(listed)


def supported_frameworks(connectors: Mapping[str, Any] | None = None) -> list[str]:
    """The framework releases Lane A runs against, as listed.

    Raises:
        ValueError: The list is missing, holds a non-version, or repeats one.
    """
    table = _compat_table(connectors)
    if "frameworks" not in table:
        raise ValueError("[tool.osprey-connectors.compat] frameworks is not declared")
    return _version_list(table, "frameworks")


def retired_frameworks(connectors: Mapping[str, Any] | None = None) -> list[str]:
    """The framework releases no lane runs against any more (empty if undeclared).

    Raises:
        ValueError: The list holds a non-version, or repeats one.
    """
    table = _compat_table(connectors)
    return _version_list(table, "retired") if "retired" in table else []


def source_text_tests(connectors: Mapping[str, Any] | None = None) -> list[str]:
    """The node ids Lane B and the nightly deselect: tests that read a source tree.

    Raises:
        ValueError: An entry is not a ``tests/...py[::name]`` node id.
    """
    listed = _compat_table(connectors).get("source-text-tests", [])
    if not isinstance(listed, list) or not all(
        isinstance(item, str) and _NODE_ID_RE.match(item) for item in listed
    ):
        raise ValueError(f"source-text-tests must be test node ids, not {listed!r}")
    return list(listed)


def connectors_floor(root: Mapping[str, Any] | None = None) -> str:
    """The version the ``osprey-connectors>=`` dependency of ``root`` names.

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


def view_required(framework: str) -> bool:
    """Whether ``framework`` must write the simulator view the connectors read.

    Every release that can be listed (R3: tagged after
    :data:`LAST_PAIRED_FRAMEWORK`) ships the view writer; only the paired
    releases up to it predate the view.
    """
    return version_key(framework) > version_key(LAST_PAIRED_FRAMEWORK)


def reachable_tags(repo: Path = REPO_ROOT) -> list[str]:
    """``git tag --merged HEAD`` in ``repo``.

    Raises:
        ValueError: The checkout is shallow (its tags are not all visible), or
            git cannot answer.
    """

    def _git(*argv: str) -> str:
        try:
            result = subprocess.run(
                ["git", *argv], cwd=repo, capture_output=True, text=True, check=True, timeout=60
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise ValueError(f"git {' '.join(argv)} failed: {error}") from error
        return result.stdout

    if _git("rev-parse", "--is-shallow-repository").strip() != "false":
        raise ValueError(
            "the checkout is shallow, so the tags reachable from HEAD are not all visible: "
            "check out with full history (fetch-depth: 0)"
        )
    return _git("tag", "--merged", "HEAD").split()


def framework_releases(tags: Iterable[str]) -> set[str]:
    """The framework versions tagged after :data:`LAST_PAIRED_FRAMEWORK`.

    A ``v``-tag that is not a release version (no tag this repository cuts)
    is not a framework release.
    """
    found = set()
    for tag in tags:
        match = _FRAMEWORK_TAG_RE.match(tag)
        if match is None or _VERSION_RE.match(match["version"]) is None:
            continue
        if version_key(match["version"]) > version_key(LAST_PAIRED_FRAMEWORK):
            found.add(match["version"])
    return found


def problems(
    frameworks: list[str],
    retired: list[str],
    floor: str,
    tags: Iterable[str] | None,
    *,
    before_tagging: str | None = None,
) -> list[str]:
    """Why the declared state is inconsistent with ``tags``, or nothing.

    Args:
        frameworks: ``[tool.osprey-connectors.compat] frameworks``.
        retired: ``[tool.osprey-connectors.compat] retired``.
        floor: The framework's connectors floor.
        tags: The tags reachable from HEAD, or ``None`` when they are unknown;
            the rules that read them are then skipped.
        before_tagging: A framework tag about to be cut (``vX.Y.Z``): R2 then
            treats it as the newest release.

    Raises:
        ValueError: ``before_tagging`` is not a framework release tag.
    """
    tags = None if tags is None else set(tags)
    found = []
    # R1: the floor is a release this history carries.
    if not is_final(floor):
        found.append(f"the connectors floor {floor} is a pre-release: the floor names a release")
    elif (
        tags is not None
        and floor_applicable(floor)
        and f"{CONNECTORS_TAG_PREFIX}{floor}" not in tags
    ):
        found.append(
            f"the connectors floor {floor} names a connectors release this history does not "
            f"carry (no tag {CONNECTORS_TAG_PREFIX}{floor}): cut and push its tag first "
            "(paired wheel installs and `osprey up --dev` would fail below it)"
        )
    # R3: the two lists are disjoint ...
    found.extend(
        f"framework {version} is both listed and retired"
        for version in sorted(set(frameworks) & set(retired), key=version_key)
    )
    if tags is None:
        return found
    released = framework_releases(tags)
    # ... and name only releases tagged after the last paired one.
    found.extend(
        f"framework {version} is in [tool.osprey-connectors.compat] {key} but no "
        f"tag v{version} after {LAST_PAIRED_FRAMEWORK} is reachable from HEAD"
        for key, versions in (("frameworks", frameworks), ("retired", retired))
        for version in versions
        if version not in released
    )
    # R2: every release but the newest is accounted for.
    if before_tagging is not None:
        match = _FRAMEWORK_TAG_RE.match(before_tagging)
        if match is None or _VERSION_RE.match(match["version"]) is None:
            raise ValueError(f"not a framework release tag: {before_tagging!r} (expected vX.Y.Z)")
        owed = released - {match["version"]}
    else:
        owed = released - {max(released, key=version_key)} if released else set()
    found.extend(
        f"framework {version} is tagged and a newer release follows it, but it is neither "
        "listed in [tool.osprey-connectors.compat] frameworks nor retired: list or retire it"
        for version in sorted(owed - set(frameworks) - set(retired), key=version_key)
    )
    return found


def matrix(frameworks: Iterable[str]) -> dict[str, list[dict[str, str]]]:
    """Lane A's ``strategy.matrix``: one cell per release, or one empty cell.

    GitHub refuses a matrix with no cell, so an empty list becomes a single
    cell whose ``framework`` is empty and whose ``reason`` the lane's
    stand-down step prints.
    """
    cells = [{"framework": version} for version in frameworks]
    return {"include": cells or [{"framework": "", "reason": NO_FRAMEWORK_LISTED}]}


def _github_output(frameworks: list[str], floor: str) -> str:
    applicable = floor_applicable(floor)
    return "\n".join(
        [
            f"matrix={json.dumps(matrix(frameworks), separators=(',', ':'))}",
            f"floor={floor}",
            f"floor-applicable={'true' if applicable else 'false'}",
        ]
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "command",
        choices=("github-output", "matrix", "floor", "check", "view-required", "source-text-tests"),
    )
    parser.add_argument("version", nargs="?", help="view-required: the framework version")
    parser.add_argument(
        "--framework",
        default="",
        help="github-output: one release to run instead of the listed ones (a dispatched dry run)",
    )
    parser.add_argument(
        "--before-tagging",
        metavar="TAG",
        help="check: hold the lists to R2 as if the framework tag TAG were the newest",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if (args.command == "view-required") != (args.version is not None):
        parser.error("view-required takes a framework version, and no other command does")
    try:
        frameworks = supported_frameworks()
        retired = retired_frameworks()
        floor = connectors_floor()
        tests = source_text_tests()
        found = problems(
            frameworks, retired, floor, reachable_tags(), before_tagging=args.before_tagging
        )
        for version in (args.framework, args.version):
            if version:
                version_key(version)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    for problem in found:
        print(f"error: {problem}", file=sys.stderr)
    if found:
        return 1

    if args.command == "matrix":
        print(json.dumps(matrix(frameworks)))
    elif args.command == "floor":
        print(floor)
    elif args.command == "view-required":
        print("true" if view_required(args.version) else "false")
    elif args.command == "source-text-tests":
        print("\n".join(tests))
    elif args.command == "github-output":
        text = _github_output([args.framework] if args.framework else frameworks, floor)
        target = os.environ.get("GITHUB_OUTPUT")
        if target:
            with open(target, "a", encoding="utf-8") as handle:
                handle.write(text + "\n")
        print(text)
        if args.framework:
            print(f"dispatched for framework {args.framework}", file=sys.stderr)
        elif frameworks:
            print(f"frameworks listed: {', '.join(frameworks)}", file=sys.stderr)
        else:
            print(NO_FRAMEWORK_LISTED, file=sys.stderr)
        if not floor_applicable(floor):
            print(
                f"connectors floor {floor} predates the first independent release "
                f"{FIRST_INDEPENDENT_CONNECTORS}: the floor lane stands down",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
