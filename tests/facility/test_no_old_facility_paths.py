"""No description module is referenced under its old ``osprey.facility`` path.

The facility description moved out of the framework into the top-level
``osprey_facility`` distribution; only the framework side (``views``,
``panels``, ``settings``, ``bundled``) stays under ``osprey.facility``. No shim
re-exports the old paths, so a reference left behind breaks at run time or
leaves a dead link in the docs. This sweep reads the shipped roots for every
form a module reference takes:

- an import statement, ``import osprey.facility.build`` or
  ``from osprey.facility.records import ...``;
- a name imported from the package itself, ``from osprey.facility import
  DEFAULT_ROLE``, that is not one of the surviving framework modules;
- a dotted string anywhere: an ``importlib`` argument, an entry-point target
  ``osprey.facility.kinds:x``, the old entry-point group, a Sphinx role or
  ``automodule`` directive;
- a file path, ``src/osprey/facility/build.py``, or the retired
  ``scripts/facility_schema/`` tool directory.

A document identifier ``osprey.facility.<name>/<N>`` (``scenarios/3``,
``facility/4``) names a file format, not a module, and is not a reference.

Exempt as whole files: the root and package changelogs and this phase's
retirement fragment (they record what was removed), and the frozen floor
schema (its text is fixed at the version it describes). A line that must name
an old path on purpose is listed in :data:`LINE_ALLOWLIST` with its reason;
an entry that no longer matches its line fails, so the list cannot rot.
"""

from __future__ import annotations

import ast
import functools
import re
from collections.abc import Collection, Iterable, Mapping
from pathlib import Path

from tests._text_sweep import SHIPPED_TEXT_SUFFIXES, text_files

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The description package whose top-level modules were once ``osprey.facility.*``.
_DESCRIPTION_PACKAGE = REPO_ROOT / "packages/osprey-facility/src/osprey_facility"

#: The framework modules that stay under ``osprey.facility``.
SURVIVING_MODULES = frozenset({"views", "panels", "settings", "bundled"})


def _description_modules(package: Path) -> frozenset[str]:
    """The top-level module and subpackage names of the description package."""
    names = {
        path.stem if path.is_file() else path.name
        for path in package.iterdir()
        if (path.suffix == ".py" or (path / "__init__.py").is_file())
        and not path.name.startswith("_")
    }
    return frozenset(names)


def _package_exports(package: Path) -> frozenset[str]:
    """The ``__all__`` of the description package, once ``osprey.facility``'s own."""
    tree = ast.parse((package / "__init__.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            return frozenset(ast.literal_eval(node.value))
    return frozenset()


#: Every name a module of the old description tree went by.
DESCRIPTION_MODULES = _description_modules(_DESCRIPTION_PACKAGE)

#: The names a caller once imported from ``osprey.facility`` itself and now
#: imports from ``osprey_facility``: its modules and its package exports.
DESCRIPTION_NAMES = DESCRIPTION_MODULES | _package_exports(_DESCRIPTION_PACKAGE)

#: The shipped roots the sweep reads (FR9: source, tests, scripts, templates,
#: root config, workflows, connectors, docs).
ROOTS = (
    "src",
    "tests",
    "scripts",
    "packages",
    "plugins",
    "docs/source",
    ".github",
    "pyproject.toml",
)

#: Diagrams and pages ship as HTML under ``docs/source``.
SUFFIXES = (*SHIPPED_TEXT_SUFFIXES, ".html", ".cfg", ".ini")

#: Whole files that record history and must name what was removed.
EXEMPT_FILES = frozenset(
    {
        "changelog.d/facility-package.removed.md",
        "packages/osprey-facility/src/osprey_facility/schema/_generated/facility-4.schema.json",
        "tests/facility/test_no_old_facility_paths.py",
        "tests/facility/test_no_old_facility_paths_detects.py",
    }
)

#: Whole files exempt by name wherever they sit.
EXEMPT_NAMES = frozenset({"CHANGELOG.md"})

#: ``(path, text the line contains) -> reason`` for lines that name an old path
#: on purpose.
LINE_ALLOWLIST: Mapping[tuple[str, str], str] = {
    (
        "tests/facility/test_measurement_kinds_registry.py",
        "``osprey.facility.kinds``",
    ): "documents the old entry-point group the registry must ignore (FR7)",
    (
        "tests/facility/test_measurement_kinds_registry.py",
        '("osprey.facility.kinds", False)',
    ): "plants a kind under the old group to prove it is not read (FR7)",
    (
        "tests/facility/test_graph_determinism.py",
        "from osprey.facility import DEFAULT_ROLE, PN_LOCAL",
    ): "respells the base commit's writer source, which used the old path",
    (
        "tests/facility/test_graph_determinism.py",
        "from osprey.facility.records import",
    ): "respells the base commit's writer source, which used the old path",
    (
        "tests/facility/test_graph_determinism.py",
        "from osprey.facility.validate import",
    ): "respells the base commit's writer source, which used the old path",
    (
        "tests/facility/test_package_boundary_detects.py",
        "osprey.facility.build",
    ): "plants an old path to prove the boundary check rejects it",
}


def _patterns(modules: Collection[str]) -> tuple[re.Pattern[str], re.Pattern[str]]:
    """The dotted/path reference pattern and the package-import pattern."""
    alternation = "|".join(sorted((re.escape(m) for m in modules), key=len, reverse=True))
    dotted = re.compile(
        rf"(?<![\w.])osprey(?P<sep>[./])facility[./](?:{alternation})(?![\w])"
        r"(?P<ident>/(?:\d+|N)\b)?"
        r"|scripts[./]facility_schema\b"
    )
    package_import = re.compile(r"\bfrom\s+osprey\.facility\s+import\s+(?P<names>.*)$")
    return dotted, package_import


def _imported_names(first: str, following: list[str]) -> list[str]:
    """The names of a ``from osprey.facility import`` clause, parenthesised or not."""
    text = first
    if text.lstrip().startswith("(") and ")" not in text:
        for line in following:
            text += " " + line
            if ")" in line:
                break
    text = text.split("#", 1)[0].replace("(", " ").replace(")", " ").replace("\\", " ")
    names = []
    for part in text.split(","):
        words = part.split()
        if words:
            names.append(re.sub(r"\W.*$", "", words[0]))
    return [n for n in names if n]


def _line_offenses(
    line: str,
    following: list[str],
    dotted: re.Pattern[str],
    package_import: re.Pattern[str],
    names: Collection[str],
) -> list[str]:
    """The old-path references on one line."""
    found = []
    for match in dotted.finditer(line):
        if match.group("ident") and match.group("sep") == ".":
            continue
        found.append(match.group(0))
    imported = package_import.search(line)
    if imported:
        for name in _imported_names(imported.group("names"), following):
            if name in names:
                found.append(f"from osprey.facility import {name}")
    return found


def old_path_offenders(
    repo_root: Path,
    roots: Iterable[str] = ROOTS,
    *,
    modules: Collection[str] = DESCRIPTION_MODULES,
    names: Collection[str] = DESCRIPTION_NAMES,
    exempt_files: Collection[str] = EXEMPT_FILES,
    line_allowlist: Mapping[tuple[str, str], str] = LINE_ALLOWLIST,
) -> tuple[list[str], set[tuple[str, str]]]:
    """Every old-path reference under ``roots``, and the allowlist entries used.

    Returns:
        ``path:line: reference`` offenders, and the allowlist keys that matched
        a line holding a reference.
    """
    dotted, package_import = _patterns(modules)

    def skip(path: Path) -> bool:
        rel = path.relative_to(repo_root).as_posix()
        return rel in exempt_files or path.name in EXEMPT_NAMES

    offenders: list[str] = []
    used: set[tuple[str, str]] = set()
    for path in text_files(repo_root, roots, suffixes=SUFFIXES, skip=skip):
        rel = path.relative_to(repo_root).as_posix()
        if rel in exempt_files:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(lines, start=1):
            offenses = _line_offenses(
                line, lines[number : number + 40], dotted, package_import, names
            )
            if not offenses:
                continue
            allowed = [key for key in line_allowlist if key[0] == rel and key[1] in line]
            if allowed:
                used.update(allowed)
                continue
            offenders.extend(f"{rel}:{number}: {offense}" for offense in offenses)
    return offenders, used


@functools.cache
def _repo_sweep() -> tuple[tuple[str, ...], frozenset[tuple[str, str]]]:
    """:func:`old_path_offenders` over the real repository, read once per process."""
    offenders, used = old_path_offenders(REPO_ROOT)
    return tuple(offenders), frozenset(used)


def test_description_modules_are_known() -> None:
    """The sweep knows the names it looks for; an empty list would pass anything."""
    assert {"build", "kinds", "records", "validate", "schema", "layers"} <= DESCRIPTION_MODULES
    assert {"DEFAULT_ROLE", "PN_LOCAL"} <= DESCRIPTION_NAMES
    assert not DESCRIPTION_NAMES & SURVIVING_MODULES


def test_no_old_facility_paths() -> None:
    offenders, _ = _repo_sweep()
    assert offenders == (), (
        "description modules live in osprey_facility; rewrite these references "
        "(or, for a line that must name the old path, add it to LINE_ALLOWLIST "
        "with its reason):\n" + "\n".join(offenders)
    )


def test_line_allowlist_entries_are_live() -> None:
    """Every allowlisted line still holds an old-path reference."""
    _, used = _repo_sweep()
    assert sorted(set(LINE_ALLOWLIST) - used) == []
