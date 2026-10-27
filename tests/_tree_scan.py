"""Reading the source tree as a fact about the suite.

Some invariants are properties of the tree rather than of any run: that a
container is built in one place, that a superseded import path is unused, that
every stand-in for an upstream gap has a canary. A
host with no container engine skips every test that could prove them by
running, so they are read off the source instead, which works wherever the
suite is collected.

The leading underscore keeps the module out of pytest collection:
``python_files`` matches ``test_*.py``/``*_test.py`` only, and a helper module
that got collected would report its imports as test failures.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable, Collection, Iterable
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TESTS_ROOT.parent


def tracked_files() -> list[str]:
    """Every path git tracks, repository-relative with forward slashes.

    The listing is NUL-separated, so a path with a space or a newline in it
    comes back whole.

    Returns:
        The tracked paths in git's order.
    """
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout
    return [path for path in out.split("\0") if path]


def python_sources(exempt: Path) -> list[tuple[str, str]]:
    """Every Python file under ``tests/`` with its text, as ``(relative path, text)``.

    Args:
        exempt: A file to leave out — a guard that names what it forbids would
            otherwise report itself.

    Returns:
        Pairs of path relative to ``tests/`` and file text, sorted by path.
    """
    skip = exempt.resolve()
    found: list[tuple[str, str]] = []
    for path in sorted(TESTS_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts or path.resolve() == skip:
            continue
        found.append((str(path.relative_to(TESTS_ROOT)), path.read_text(encoding="utf-8")))
    return found


def product_sources(root: Path) -> list[tuple[str, str]]:
    """Every Python file of the shipped code under ``root`` with its text.

    The shipped code is ``root/src`` and each ``root/packages/*/src``.

    Args:
        root: The repository root, or a stand-in tree laid out like it.

    Returns:
        Pairs of path relative to ``root`` (POSIX form) and file text, sorted
        by path.
    """
    found: list[tuple[str, str]] = []
    for source in [root / "src", *sorted((root / "packages").glob("*/src"))]:
        for path in source.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            found.append((path.relative_to(root).as_posix(), path.read_text(encoding="utf-8")))
    return sorted(found)


#: The extensions shipped prose, code and configuration are written in — the
#: default admission rule of the repository text sweeps.
SHIPPED_TEXT_SUFFIXES = (
    ".py",
    ".md",
    ".rst",
    ".txt",
    ".j2",
    ".yml",
    ".yaml",
    ".json",
    ".toml",
    ".sh",
)

#: Shipped files with no extension, admitted by name. The scaffold emits its
#: ``gitignore``/``dockerignore``/``Dockerfile`` templates verbatim, and a
#: suffix-only rule cannot see them.
EXTENSIONLESS_SHIPPED_NAMES = frozenset({"gitignore", "dockerignore", "Dockerfile"})


def text_files(
    repo_root: Path,
    roots: Iterable[str],
    *,
    suffixes: Collection[str] = SHIPPED_TEXT_SUFFIXES,
    names: Collection[str] = frozenset(),
    skip: Callable[[Path], bool] | None = None,
) -> list[Path]:
    """Every file a repository text sweep reads under ``roots``.

    A root names a directory or one file. A directory is walked recursively
    with ``__pycache__`` pruned, and admits a file whose suffix is in
    ``suffixes`` or whose name, any leading dot stripped, is in ``names`` (the
    same kind of file ships both as ``gitignore`` and ``.dockerignore``). A
    file root is admitted as named: ``rglob`` on a file yields nothing, which
    would silently drop it. A root that does not exist contributes nothing.

    Args:
        repo_root: The repository root, or a stand-in tree laid out like it.
        roots: Paths relative to ``repo_root``.
        suffixes: The admitted extensions.
        names: The admitted extensionless file names.
        skip: Leaves out a walked file it returns true for.

    Returns:
        The admitted files, root by root in ``rglob`` order.
    """
    files: list[Path] = []
    for root in roots:
        base = repo_root / root
        if base.is_file():
            files.append(base)
            continue
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.suffix not in suffixes and path.name.lstrip(".") not in names:
                continue
            if skip is not None and skip(path):
                continue
            files.append(path)
    return files


def matching_lines(
    repo_root: Path, files: Iterable[Path], pattern: re.Pattern[str]
) -> list[tuple[str, int, str]]:
    """Every line of ``files`` that ``pattern`` matches.

    Files are read as UTF-8 with undecodable bytes ignored; one that cannot be
    read is passed over.

    Args:
        repo_root: The root the reported paths are relative to.
        files: The files to read, typically from :func:`text_files`.
        pattern: Searched in each line.

    Returns:
        ``(path relative to repo_root in POSIX form, line number, line)``
        triples, the line as written.
    """
    found: list[tuple[str, int, str]] = []
    for path in files:
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if not pattern.search(content):
            continue
        relative = path.relative_to(repo_root).as_posix()
        for number, line in enumerate(content.splitlines(), start=1):
            if pattern.search(line):
                found.append((relative, number, line))
    return found
