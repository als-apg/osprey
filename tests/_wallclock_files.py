"""Which test files hold an image-free ``wallclock`` budget, read off their source.

A wall-clock budget runs alone, in one of two serial checks placed by the image
marker. ``pytest-wallclock-unit`` (``scripts/repo_checks.py``) times the
image-free budgets (:data:`IMAGE_FREE`) and collects only the files this module
names, not the whole tree it would otherwise import to pick a handful of tests.
``pytest-wallclock`` times the budgets that need the sidecar image
(:data:`IMAGE_BOUND`) under :data:`IMAGE_ROOT`, in the job that builds it. A
budget added anywhere is still timed with no new wiring, because the files are
found when the check runs.

A text scan can miss a marker applied out of sight, by a conftest hook or a
mark imported under another name. :func:`unscanned` holds the scan to what
pytest actually marks: the root conftest calls it on every collection, before
any deselection, so a run that collects a marked test neither check would time,
or one whose file the scan places in the wrong check, stops.

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
#: The marker of a test that needs the sidecar image built.
IMAGE_MARKER = "dockerbuild"
#: The one path the image check collects; an image-bound budget lives under it.
IMAGE_ROOT = "tests/integration"
#: The two checks' selections. Complementary, so each marked test is selected
#: by exactly one of them.
IMAGE_FREE = f"{MARKER} and not {IMAGE_MARKER}"
IMAGE_BOUND = f"{MARKER} and {IMAGE_MARKER}"
REPO_ROOT = Path(__file__).resolve().parents[1]
#: The module that states the sweep's selection, whose roots the scan covers.
SWEEP_MODULE = "scripts/repo_checks.py"


def _mark_re(marker: str) -> re.Pattern[bytes]:
    """*marker* applied in a file: as a decorator, on a ``pytestmark`` line, or
    as a line of its own inside a ``pytestmark`` list. Anchored at the start of
    a line so a test that only spells the marker in a string is not read as a
    carrier."""
    return re.compile(
        rb"^[ \t]*(?:@|pytestmark\b.*?)?pytest\.mark\." + marker.encode() + rb"\b", re.MULTILINE
    )


MARK_RE = _mark_re(MARKER)
IMAGE_MARK_RE = _mark_re(IMAGE_MARKER)


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


def _under(relative: str, top: str) -> bool:
    top = top.rstrip("/")
    return relative == top or relative.startswith(f"{top}/")


def _scanned(relative: str, roots: Iterable[str]) -> bool:
    """Whether *relative* lies under one of *roots*."""
    return any(_under(relative, root) for root in roots)


def _applies(data: bytes, marker: str, pattern: re.Pattern[bytes]) -> bool:
    # The literal every match spells; a file without it skips the regex.
    return b"pytest.mark." + marker.encode() in data and pattern.search(data) is not None


def carries_marker(path: Path) -> bool:
    """Whether the source of *path* applies the marker, as the scan reads it."""
    return _applies(path.read_bytes(), MARKER, MARK_RE)


def image_free(path: Path) -> bool:
    """Whether the source of *path* applies the marker and not the image marker:
    a file the image-free check collects."""
    data = path.read_bytes()
    return _applies(data, MARKER, MARK_RE) and not _applies(data, IMAGE_MARKER, IMAGE_MARK_RE)


def marked_files(root: Path = REPO_ROOT, roots: Iterable[str] | None = None) -> list[str]:
    """The ``test_*.py`` files under *roots* (the sweep's, by default) that the
    image-free check collects, placed by the image marker: their source applies
    the marker and not the image marker. Repository-relative and sorted."""
    roots = tuple(sweep_roots() if roots is None else roots)
    found = set()
    for top in roots:
        for path in (root / top).rglob("test_*.py"):
            if "__pycache__" in path.parts:
                continue
            if image_free(path):
                found.add(path.relative_to(root).as_posix())
    return sorted(found)


def unscanned(
    marked: Iterable[tuple[Path, bool]], root: Path = REPO_ROOT, roots: Iterable[str] | None = None
) -> list[str]:
    """The files of *marked* that neither check, placed by the image marker,
    would time once, each with what to do about it.

    *marked* holds one ``(path, image_bound)`` pair per test pytest gave the
    marker: the test's file and whether pytest also gave it the image marker.
    Only files under *roots* (the sweep's, by default) count. An image-free test
    belongs in a file :func:`marked_files` names, and an image-bound one in a
    file it does not name, under :data:`IMAGE_ROOT`. Empty means every marked
    test collected is where its check looks.
    """
    roots = tuple(sweep_roots() if roots is None else roots)
    problems = []
    for path, image_bound in sorted(set(marked)):
        try:
            relative = path.resolve().relative_to(root).as_posix()
        except ValueError:
            continue
        if not _scanned(relative, roots):
            continue
        is_test = path.name.startswith("test_")
        named = is_test and image_free(path)
        if image_bound and not _under(relative, IMAGE_ROOT):
            problems.append(
                f"{relative} holds a {MARKER} test marked {IMAGE_MARKER} outside "
                f"{IMAGE_ROOT}/, where pytest-wallclock does not look, so no check "
                f"times it: move the file under {IMAGE_ROOT}/"
            )
        elif image_bound and named:
            problems.append(
                f"{relative} holds a {MARKER} test pytest marks {IMAGE_MARKER} out of "
                f"the sight of the scan in tests/_wallclock_files.py, which hands the "
                f"file to pytest-wallclock-unit, a check that builds no image: apply "
                f"{IMAGE_MARKER} in the file, on its pytestmark line"
            )
        elif not image_bound and not named and is_test and carries_marker(path):
            problems.append(
                f"{relative} mixes {MARKER} tests with and without {IMAGE_MARKER}, so "
                f"no check times the ones without it: put {IMAGE_MARKER} on the "
                f"module's pytestmark line"
            )
        elif not image_bound and not named:
            problems.append(
                f"{relative} holds a {MARKER}-marked test that the scan in "
                f"tests/_wallclock_files.py does not find, so pytest-wallclock-unit "
                f"would not time it: apply the marker in the file, as a decorator or "
                f"on a pytestmark line"
            )
    return problems
