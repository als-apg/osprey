"""Reading the shipped text a repository sweep checks.

The removal gates under ``tests/docs/`` and the top-level literal guards sweep
shipped roots (``src``, ``docs``, ``plugins`` and the like) for words, ports and
spellings. They collect and match files through this module. It is kept apart
from :mod:`tests._tree_scan` on purpose: importing that module marks a test as a
reader of the whole tests tree (``tests/scripts/test_check.py``), and these
sweeps read only the roots their caller names.

The leading underscore keeps the module out of pytest collection.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Iterable, Mapping
from pathlib import Path
from typing import TypeVar

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
    file root is admitted as named, without consulting ``suffixes``, ``names``
    or ``skip``: ``rglob`` on a file yields nothing, which would silently drop
    it. A root that does not exist contributes nothing.

    Args:
        repo_root: The repository root, or a stand-in tree laid out like it.
        roots: Paths relative to ``repo_root``.
        suffixes: The admitted extensions.
        names: The admitted extensionless file names.
        skip: Leaves out a walked file it returns true for; never a file root.

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


#: The key a caller of :func:`matching_lines_each` names each pattern by.
_Key = TypeVar("_Key")


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
    return matching_lines_each(repo_root, files, {None: pattern})[None]


def matching_lines_each(
    repo_root: Path, files: Iterable[Path], patterns: Mapping[_Key, re.Pattern[str]]
) -> dict[_Key, list[tuple[str, int, str]]]:
    """:func:`matching_lines` for each of ``patterns``, reading each file once.

    A sweep with several patterns over the same files spends most of its time
    reading them, so it asks for all its patterns in one pass.

    Args:
        repo_root: The root the reported paths are relative to.
        files: The files to read, typically from :func:`text_files`.
        patterns: Each searched in each line, by a key of the caller's.

    Returns:
        For each key, what :func:`matching_lines` returns for its pattern.
    """
    found: dict[_Key, list[tuple[str, int, str]]] = {key: [] for key in patterns}
    for path in files:
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        lines: list[str] | None = None
        for key, pattern in patterns.items():
            if not pattern.search(content):
                continue
            if lines is None:
                lines = content.splitlines()
            relative = path.relative_to(repo_root).as_posix()
            found[key] += (
                (relative, number, line)
                for number, line in enumerate(lines, start=1)
                if pattern.search(line)
            )
    return found
