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

The scan covers every root the parallel sweep collects, read from the sweep's
one statement (``UNIT_SELECTION`` in ``scripts/repo_checks.py``), so a budget in
the connectors package's suite is timed alone like one under ``tests/``.

Standard library only: ``scripts/repo_checks.py`` loads this module by path,
where pytest may not be importable. The leading underscore keeps it out of
collection.
"""

from __future__ import annotations

import functools
import importlib.util
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType

MARKER = "wallclock"
REPO_ROOT = Path(__file__).resolve().parents[1]
#: The module that states the sweep's selection, whose roots the scan covers.
SWEEP_MODULE = "scripts/repo_checks.py"
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


def _load(path: Path, name: str) -> ModuleType:
    # Registered before it runs, so its dataclasses resolve their module.
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@functools.cache
def sweep_roots() -> tuple[str, ...]:
    """The paths the parallel sweep collects, read from its selection.

    ``scripts/`` is not a package; its checks module imports ``check_kit`` by
    name, so that is loaded first, as ``scripts/check.py`` loads it.
    """
    scripts = REPO_ROOT / "scripts"
    if "check_kit" not in sys.modules:
        _load(scripts / "check_kit.py", "check_kit")
    checks = _load(REPO_ROOT / SWEEP_MODULE, "_wallclock_sweep_selection")
    return tuple(root.rstrip("/") for root in checks.UNIT_SELECTION.roots)


def _scanned(relative: str, roots: Iterable[str]) -> bool:
    """Whether *relative* lies under one of *roots* and outside :data:`EXCLUDED`."""
    under = any(relative.startswith(f"{root.rstrip('/')}/") for root in roots)
    return under and not any(relative == d or relative.startswith(f"{d}/") for d in EXCLUDED)


def carries_marker(path: Path) -> bool:
    """Whether the source of *path* applies the marker, as the scan reads it."""
    return MARK_RE.search(path.read_bytes()) is not None


def marked_files(root: Path = REPO_ROOT, roots: Iterable[str] | None = None) -> list[str]:
    """The ``test_*.py`` files under *roots* (the sweep's, by default) outside
    :data:`EXCLUDED` whose source applies the marker, repository-relative and
    sorted."""
    roots = tuple(sweep_roots() if roots is None else roots)
    found = set()
    for top in roots:
        for path in (root / top).rglob("test_*.py"):
            relative = path.relative_to(root).as_posix()
            if "__pycache__" in path.parts or not _scanned(relative, roots):
                continue
            if carries_marker(path):
                found.add(relative)
    return sorted(found)


def unscanned(
    marked: Iterable[Path], root: Path = REPO_ROOT, roots: Iterable[str] | None = None
) -> list[str]:
    """The files of *marked* (the files of the tests pytest gave the marker)
    that the scan does not name, each with what to do about it.

    Only files under *roots* (the sweep's, by default) outside :data:`EXCLUDED`
    count: the others are timed by checks that select over their directories.
    Empty means every marked test collected is in a file :func:`marked_files`
    names.
    """
    roots = tuple(sweep_roots() if roots is None else roots)
    problems = []
    for path in sorted(set(marked)):
        try:
            relative = path.resolve().relative_to(root).as_posix()
        except ValueError:
            continue
        if not _scanned(relative, roots):
            continue
        if not (path.name.startswith("test_") and carries_marker(path)):
            problems.append(
                f"{relative} holds a {MARKER}-marked test that the scan in "
                f"tests/_wallclock_files.py does not find, so pytest-wallclock-unit "
                f"would not time it: apply the marker in the file, as a decorator or "
                f"on a pytestmark line"
            )
    return problems
