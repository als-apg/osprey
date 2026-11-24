#!/usr/bin/env python3
"""Which framework releases a component package's compatibility lanes run against.

``osprey-connectors`` and ``osprey-facility`` release on their own tags, and
every framework release that depends on one pins it with a floor and no
ceiling, so a rebuild of a released framework resolves the newest package.
Two lanes per package hold that up:

* **Lane A** (``connectors-compat`` / ``facility-compat``, and the nightly)
  runs the package built at HEAD under each framework release listed in
  ``[tool.<dist>.compat] frameworks`` of the package's ``pyproject.toml``.
* **Lane B** (``framework-at-connectors-floor`` /
  ``framework-at-facility-floor``) runs the framework at HEAD with the
  package release its ``<dist>>=`` floor names.

Lane A stands down while ``frameworks`` is empty, and Lane B while the floor
does not apply: for connectors while it predates
:data:`FIRST_INDEPENDENT_CONNECTORS`, for facility while the root lists
``osprey-facility`` with no version specifier (:attr:`Package.floor_optional`).
Each says so in its log and succeeds. :func:`problems` holds the declarations
to the tags reachable from HEAD, so neither stand-down can outlive the
releases that end it:

* **R1** The floor is a final release, and once it applies its
  ``<dist>-v`` tag is reachable: below a floor this history does not carry,
  paired wheel installs and ``osprey up --dev`` fail.
* **R2** Every framework release tagged after the package's cutoff
  (:attr:`Package.cutoff`: :data:`LAST_PAIRED_FRAMEWORK` for connectors,
  :data:`LAST_FRAMEWORK_WITHOUT` for facility) is listed in ``frameworks`` or
  in ``retired``, except the newest: its publish and the PR that lists it
  necessarily follow its tag. ``check --before-tagging vX.Y.Z`` applies the
  rule as if that tag were the newest, so the release about to be cut may not
  leave its predecessor unlisted.
* **R3** Every listed and retired release is such a tag, and none is both.

Usage::

    python3 scripts/ci/compat_matrix.py --package P github-output [--framework X]  # the plan job
    python3 scripts/ci/compat_matrix.py --package P matrix
    python3 scripts/ci/compat_matrix.py --package P floor
    python3 scripts/ci/compat_matrix.py --package P check [--before-tagging vX.Y.Z]
    python3 scripts/ci/compat_matrix.py --package connectors view-required <framework version>
    python3 scripts/ci/compat_matrix.py --package connectors source-text-tests

``P`` is ``connectors`` or ``facility``; ``--root DIR`` reads the pyprojects
and the tags of another tree. Every command checks the declarations first,
against ``git tag --merged HEAD``, and refuses a shallow checkout, whose tags
it cannot see. Standard library only, so the plan job runs it with the
runner's ``python3``.
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
from typing import Any, NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The first ``osprey-connectors`` released from its own tag. A floor below it
#: names a build cut from the framework's tags, older than anything HEAD of
#: the framework is written against.
FIRST_INDEPENDENT_CONNECTORS = "2026.10.0"

#: The last framework release whose connectors were cut from the framework's
#: own tag. Every framework tagged after it pairs with independent connectors.
LAST_PAIRED_FRAMEWORK = "2026.9.0b3"

#: The newest framework release whose root ``[project].dependencies`` has no
#: ``osprey-facility``. Every framework tagged after it depends on the package.
LAST_FRAMEWORK_WITHOUT = "2026.11.1"


class Package(NamedTuple):
    """The constants one package's compat machinery reads."""

    key: str
    dist: str
    package_dir: str
    #: The last framework release that does not pair with an independent
    #: release of this package: R2 and R3 read the tags after it.
    cutoff: str
    #: A floor below this names no independent release, and Lane B stands
    #: down; ``None`` when every release of the package is its own.
    first_independent: str | None
    #: Whether the root may list the package with no version specifier, which
    #: stands Lane B down. A required floor that is absent is an error.
    floor_optional: bool
    #: Subcommands this package refuses, each with the reason it prints.
    refused: Mapping[str, str]

    @property
    def tag_prefix(self) -> str:
        return f"{self.dist}-v"

    @property
    def table(self) -> str:
        return f"[tool.{self.dist}.compat]"

    def pyproject(self, root: Path) -> Path:
        return root / self.package_dir / "pyproject.toml"


CONNECTORS = Package(
    key="connectors",
    dist="osprey-connectors",
    package_dir="packages/osprey-connectors",
    cutoff=LAST_PAIRED_FRAMEWORK,
    first_independent=FIRST_INDEPENDENT_CONNECTORS,
    floor_optional=False,
    refused={},
)

FACILITY = Package(
    key="facility",
    dist="osprey-facility",
    package_dir="packages/osprey-facility",
    cutoff=LAST_FRAMEWORK_WITHOUT,
    first_independent=None,
    floor_optional=True,
    refused={
        "view-required": (
            "view-required is connectors-only: the simulator view is what the connectors "
            "read, and no framework release writes one for osprey-facility"
        ),
        "source-text-tests": (
            "source-text-tests is connectors-only from the command line: the facility "
            "lanes read [tool.osprey-facility.compat] source-text-tests through the module"
        ),
    },
)

PACKAGES: dict[str, Package] = {package.key: package for package in (CONNECTORS, FACILITY)}

#: The release forms this repository tags: ``YYYY.M.P``, optionally a
#: pre-release (``a``/``b``/``rc`` + n), a ``.postN`` and a ``.devN``.
_VERSION_RE = re.compile(
    r"^(?P<release>\d+(?:\.\d+)*)"
    r"(?:(?P<pre>a|b|rc)(?P<pre_n>\d+))?"
    r"(?:\.post(?P<post>\d+))?"
    r"(?:\.dev(?P<dev>\d+))?$"
)
_PRE_ORDER = {"a": 0, "b": 1, "rc": 2}
#: A requirement's project name, then whatever follows it.
_REQUIREMENT_RE = re.compile(r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)(?P<rest>.*)$")
_BARE_FLOOR_RE = re.compile(r"^\s*>=\s*(?P<floor>[^,;\s]+)\s*$")
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


def _normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _read(path: Path) -> Mapping[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def no_framework_listed(package: Package) -> str:
    """Why Lane A's single cell stands down; the cell carries it to the log."""
    return (
        f"no supported framework release listed yet: {package.table} "
        f"frameworks in {package.package_dir}/pyproject.toml is empty"
    )


def _compat_table(
    package: Package, document: Mapping[str, Any] | None, root: Path
) -> Mapping[str, Any]:
    document = document if document is not None else _read(package.pyproject(root))
    try:
        return document["tool"][package.dist]["compat"]
    except KeyError as error:
        raise ValueError(f"{package.table} frameworks is not declared") from error


def _version_list(table: Mapping[str, Any], key: str) -> list[str]:
    listed = table[key]
    if not isinstance(listed, list) or not all(isinstance(item, str) for item in listed):
        raise ValueError(f"{key} must be a list of version strings, not {listed!r}")
    for version in listed:
        version_key(version)
    if len(set(listed)) != len(listed):
        raise ValueError(f"{key} lists a release twice: {listed}")
    return list(listed)


def supported_frameworks(
    package: Package, document: Mapping[str, Any] | None = None, *, root: Path = REPO_ROOT
) -> list[str]:
    """The framework releases Lane A runs against, as listed.

    Raises:
        ValueError: The list is missing, holds a non-version, or repeats one.
    """
    table = _compat_table(package, document, root)
    if "frameworks" not in table:
        raise ValueError(f"{package.table} frameworks is not declared")
    return _version_list(table, "frameworks")


def retired_frameworks(
    package: Package, document: Mapping[str, Any] | None = None, *, root: Path = REPO_ROOT
) -> list[str]:
    """The framework releases no lane runs against any more (empty if undeclared).

    Raises:
        ValueError: The list holds a non-version, or repeats one.
    """
    table = _compat_table(package, document, root)
    return _version_list(table, "retired") if "retired" in table else []


def source_text_tests(
    package: Package, document: Mapping[str, Any] | None = None, *, root: Path = REPO_ROOT
) -> list[str]:
    """The node ids Lane B and the nightly deselect: tests that read a source tree.

    Raises:
        ValueError: An entry is not a ``tests/...py[::name]`` node id.
    """
    listed = _compat_table(package, document, root).get("source-text-tests", [])
    if not isinstance(listed, list) or not all(
        isinstance(item, str) and _NODE_ID_RE.match(item) for item in listed
    ):
        raise ValueError(f"source-text-tests must be test node ids, not {listed!r}")
    return list(listed)


def floor(
    package: Package, document: Mapping[str, Any] | None = None, *, root: Path = REPO_ROOT
) -> str | None:
    """The version the ``<dist>>=`` dependency of the root ``pyproject.toml`` names.

    ``None`` is the stand-down of an optional floor: the root lists the
    package with no version specifier at all.

    Raises:
        ValueError: The dependency is absent or listed more than once, or it is
            not one bare floor (a pin, a ceiling, a marker), or it names no
            version and the package's floor is required.
    """
    document = document if document is not None else _read(root / "pyproject.toml")
    entries = []
    for requirement in document["project"]["dependencies"]:
        match = _REQUIREMENT_RE.match(requirement.strip())
        if match is not None and _normalized(match["name"]) == package.dist:
            entries.append(match["rest"])
    expected = f"expected one '{package.dist}>=' dependency"
    if len(entries) != 1:
        raise ValueError(f"{expected}, found {len(entries)}")
    (rest,) = entries
    if package.floor_optional and not rest.strip():
        return None
    bare = _BARE_FLOOR_RE.match(rest)
    if bare is None:
        raise ValueError(f"{expected}, found {package.dist}{rest.rstrip()!r}")
    version_key(bare["floor"])
    return bare["floor"]


def floor_stand_down(package: Package) -> str:
    """Why Lane B stands down while an optional floor is not declared."""
    return (
        f"no '{package.dist}>=' floor in the root pyproject.toml ({package.dist} is listed "
        "with no version): the floor lane stands down"
    )


def floor_applicable(package: Package, floor: str | None) -> bool:
    """Whether the floor names an independently released package."""
    if floor is None:
        return False
    if package.first_independent is None:
        return True
    return version_key(floor) >= version_key(package.first_independent)


def view_required(framework: str) -> bool:
    """Whether ``framework`` must write the simulator view the connectors read.

    Connectors only. Every release that can be listed (R3: tagged after
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


def _framework_version(tag: str) -> str | None:
    """The release version a framework ``v``-tag names, or ``None`` for any other tag."""
    match = _FRAMEWORK_TAG_RE.match(tag)
    if match is None or _VERSION_RE.match(match["version"]) is None:
        return None
    return match["version"]


def framework_releases(package: Package, tags: Iterable[str]) -> set[str]:
    """The framework versions tagged after the package's :attr:`Package.cutoff`.

    A ``v``-tag that is not a release version (no tag this repository cuts)
    is not a framework release.
    """
    cutoff = version_key(package.cutoff)
    versions = (_framework_version(tag) for tag in tags)
    return {version for version in versions if version and version_key(version) > cutoff}


def problems(
    package: Package,
    frameworks: list[str],
    retired: list[str],
    floor: str | None,
    tags: Iterable[str] | None,
    *,
    before_tagging: str | None = None,
) -> list[str]:
    """Why the declared state is inconsistent with ``tags``, or nothing.

    Args:
        package: Whose constants the rules read.
        frameworks: ``[tool.<dist>.compat] frameworks``.
        retired: ``[tool.<dist>.compat] retired``.
        floor: The framework's floor on the package; ``None`` for the
            stand-down of an optional floor, which leaves R1 nothing to hold.
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
    if floor is None:
        pass
    elif not is_final(floor):
        found.append(f"the {package.key} floor {floor} is a pre-release: the floor names a release")
    elif (
        tags is not None
        and floor_applicable(package, floor)
        and f"{package.tag_prefix}{floor}" not in tags
    ):
        found.append(
            f"the {package.key} floor {floor} names a {package.key} release this history does "
            f"not carry (no tag {package.tag_prefix}{floor}): cut and push its tag first "
            "(paired wheel installs and `osprey up --dev` would fail below it)"
        )
    # R3: the two lists are disjoint ...
    found.extend(
        f"framework {version} is both listed and retired"
        for version in sorted(set(frameworks) & set(retired), key=version_key)
    )
    if tags is None:
        return found
    released = framework_releases(package, tags)
    # ... and name only releases tagged after the cutoff.
    found.extend(
        f"framework {version} is in {package.table} {key} but no "
        f"tag v{version} after {package.cutoff} is reachable from HEAD"
        for key, versions in (("frameworks", frameworks), ("retired", retired))
        for version in versions
        if version not in released
    )
    # R2: every release but the newest is accounted for.
    if before_tagging is not None:
        about_to_tag = _framework_version(before_tagging)
        if about_to_tag is None:
            raise ValueError(f"not a framework release tag: {before_tagging!r} (expected vX.Y.Z)")
        owed = released - {about_to_tag}
    else:
        owed = released - {max(released, key=version_key)} if released else set()
    found.extend(
        f"framework {version} is tagged and a newer release follows it, but it is neither "
        f"listed in {package.table} frameworks nor retired: list or retire it"
        for version in sorted(owed - set(frameworks) - set(retired), key=version_key)
    )
    return found


def matrix(package: Package, frameworks: Iterable[str]) -> dict[str, list[dict[str, str]]]:
    """Lane A's ``strategy.matrix``: one cell per release, or one empty cell.

    GitHub refuses a matrix with no cell, so an empty list becomes a single
    cell whose ``framework`` is empty and whose ``reason`` the lane's
    stand-down step prints.
    """
    cells = [{"framework": version} for version in frameworks]
    return {"include": cells or [{"framework": "", "reason": no_framework_listed(package)}]}


def _github_output(package: Package, frameworks: list[str], floor: str | None) -> str:
    applicable = floor_applicable(package, floor)
    return "\n".join(
        [
            f"matrix={json.dumps(matrix(package, frameworks), separators=(',', ':'))}",
            f"floor={floor or ''}",
            f"floor-applicable={'true' if applicable else 'false'}",
        ]
    )


def _floor_stand_down_reason(package: Package, floor: str | None) -> str | None:
    if floor is None:
        return floor_stand_down(package)
    if not floor_applicable(package, floor):
        return (
            f"{package.key} floor {floor} predates the first independent release "
            f"{package.first_independent}: the floor lane stands down"
        )
    return None


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--package", required=True, choices=tuple(PACKAGES))
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="the tree whose pyprojects and reachable tags are read (default: this repository)",
    )
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
    package = PACKAGES[args.package]
    if args.command in package.refused:
        print(f"error: {package.refused[args.command]}", file=sys.stderr)
        return 2
    if (args.command == "view-required") != (args.version is not None):
        parser.error("view-required takes a framework version, and no other command does")
    root: Path = args.root
    try:
        frameworks = supported_frameworks(package, root=root)
        retired = retired_frameworks(package, root=root)
        declared = floor(package, root=root)
        tests = source_text_tests(package, root=root)
        found = problems(
            package,
            frameworks,
            retired,
            declared,
            reachable_tags(root),
            before_tagging=args.before_tagging,
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
        print(json.dumps(matrix(package, frameworks)))
    elif args.command == "floor":
        if declared is None:
            print(floor_stand_down(package), file=sys.stderr)
        else:
            print(declared)
    elif args.command == "view-required":
        print("true" if view_required(args.version) else "false")
    elif args.command == "source-text-tests":
        print("\n".join(tests))
    elif args.command == "github-output":
        text = _github_output(package, [args.framework] if args.framework else frameworks, declared)
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
            print(no_framework_listed(package), file=sys.stderr)
        reason = _floor_stand_down_reason(package, declared)
        if reason is not None:
            print(reason, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
