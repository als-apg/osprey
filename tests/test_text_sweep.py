"""The repository text sweeps read the files their roots name, and only those.

Every removal gate under ``tests/docs/`` and the top-level literal guards
collect their files through :func:`tests._text_sweep.text_files`. A collector
that silently dropped a file root or an extensionless template would leave each
of them reporting a clean tree it never opened.
"""

from __future__ import annotations

import re
from pathlib import Path

from tests._text_sweep import EXTENSIONLESS_SHIPPED_NAMES, matching_lines, text_files


def _plant(root: Path, relative: str, text: str = "x\n") -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_the_collector_admits_what_its_rule_names(tmp_path: Path) -> None:
    for relative in (
        "src/pkg/module.py",
        "src/pkg/__pycache__/module.py",
        "src/pkg/image.png",
        "src/pkg/gitignore",
        "src/pkg/.dockerignore",
        "src/pkg/LICENSE",
        "src/pkg/skipped.py",
        "README",
    ):
        _plant(tmp_path, relative)

    found = text_files(
        tmp_path,
        ("src", "README", "missing"),
        names=EXTENSIONLESS_SHIPPED_NAMES,
        skip=lambda path: path.name == "skipped.py",
    )

    assert sorted(path.relative_to(tmp_path).as_posix() for path in found) == [
        # A file root is admitted as named, whatever its suffix.
        "README",
        # Extensionless names match with or without a leading dot.
        "src/pkg/.dockerignore",
        "src/pkg/gitignore",
        "src/pkg/module.py",
    ]


def test_matching_lines_reports_each_matching_line_as_written(tmp_path: Path) -> None:
    _plant(tmp_path, "docs/a.rst", "clean\n  retired name\nretired twice: retired\n")
    _plant(tmp_path, "docs/b.rst", "clean\n")

    hits = matching_lines(tmp_path, text_files(tmp_path, ("docs",)), re.compile("retired"))

    assert hits == [
        ("docs/a.rst", 2, "  retired name"),
        ("docs/a.rst", 3, "retired twice: retired"),
    ]
