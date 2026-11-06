"""Which test files carry the ``wallclock`` marker, read off their source.

The image-free wall-clock budgets run alone, in a pytest process that collects
only the files this module names (``scripts/repo_checks.py``,
``pytest-wallclock-unit``), not the whole tree it would otherwise import to
pick a handful of tests. A budget added anywhere is still timed with no new
wiring, because the files are found when the check runs.

A text scan can miss a marker applied out of sight, by a conftest hook or a
mark imported under another name. :func:`unscanned` holds the scan to what
pytest actually marks: the root conftest calls it on every collection, before
any deselection, so a sweep that collects a marked test in a file the scan does
not name stops.

Standard library only: ``scripts/repo_checks.py`` loads this module by path,
where pytest may not be importable. The leading underscore keeps it out of
collection.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

MARKER = "wallclock"
REPO_ROOT = Path(__file__).resolve().parents[1]
#: Where the scan looks.
ROOT = "tests"
#: Left out of the scan: their budgets need the sidecar image, and the
#: qmd-sidecar-e2e job and the ``pytest-wallclock`` check time them.
EXCLUDED = ("tests/e2e", "tests/integration")
#: The marker applied in the file: as a decorator, on a ``pytestmark`` line, or
#: as a line of its own inside a ``pytestmark`` list. Anchored at the start of
#: a line so a test that only spells the marker in a string is not read as a
#: carrier.
MARK_RE = re.compile(
    rb"^[ \t]*(?:@|pytestmark\b.*?)?pytest\.mark\." + MARKER.encode() + rb"\b", re.MULTILINE
)


def _excluded(relative: str) -> bool:
    return any(relative == d or relative.startswith(f"{d}/") for d in EXCLUDED)


def carries_marker(path: Path) -> bool:
    """Whether the source of *path* applies the marker, as the scan reads it."""
    return MARK_RE.search(path.read_bytes()) is not None


def marked_files(root: Path = REPO_ROOT) -> list[str]:
    """The ``test_*.py`` files under ``tests/`` outside :data:`EXCLUDED` whose
    source applies the marker, repository-relative and sorted."""
    found = []
    for path in sorted((root / ROOT).rglob("test_*.py")):
        relative = path.relative_to(root).as_posix()
        if "__pycache__" in path.parts or _excluded(relative):
            continue
        if carries_marker(path):
            found.append(relative)
    return found


def unscanned(marked: Iterable[Path], root: Path = REPO_ROOT) -> list[str]:
    """The files of *marked* (the files of the tests pytest gave the marker)
    that the scan does not name, each with what to do about it.

    Only files under ``tests/`` outside :data:`EXCLUDED` count: the others are
    timed by checks that select over their directories. Empty means every
    marked test collected is in a file :func:`marked_files` names.
    """
    problems = []
    for path in sorted(set(marked)):
        try:
            relative = path.resolve().relative_to(root).as_posix()
        except ValueError:
            continue
        if not relative.startswith(f"{ROOT}/") or _excluded(relative):
            continue
        if not (path.name.startswith("test_") and carries_marker(path)):
            problems.append(
                f"{relative} holds a {MARKER}-marked test that the scan in "
                f"tests/_wallclock_files.py does not find, so pytest-wallclock-unit "
                f"would not time it: apply the marker in the file, as a decorator or "
                f"on a pytestmark line"
            )
    return problems
