"""Reading the test tree as a fact about the suite.

Some invariants are properties of the tree rather than of any run: that a
container is built in one place, that a superseded import path is unused. A
host with no container engine skips every test that could prove them by
running, so they are read off the source instead, which works wherever the
suite is collected.

The leading underscore keeps the module out of pytest collection:
``python_files`` matches ``test_*.py``/``*_test.py`` only, and a helper module
that got collected would report its imports as test failures.
"""

from __future__ import annotations

from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parent


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
