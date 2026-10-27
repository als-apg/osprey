"""Removal gate: the MML converter script is gone from the documentation.

``osprey mml import`` / ``map`` / ``emit`` replaced the one-shot converter
module ``osprey.services.channel_finder.utils.mml_converter`` and its
``MMLConverter`` class. The module was deleted, so a page that still tells a
reader to run it hands out a command that fails.

Scope of the sweep
------------------
``ROOTS`` is the documentation tree alone. The converter's name survives on
purpose in three places outside it:

* ``changelog.d/`` and ``CHANGELOG.md`` --- history records that the script
  existed and that it was removed.
* ``tests/`` --- including this file, which spells both retired names once in
  the pattern below. It is under ``tests/``, so it is not swept, exactly as the
  sibling gate (``test_skills_cli_retired.py``) arranges for its own literals.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests._tree_scan import SHIPPED_TEXT_SUFFIXES, matching_lines, text_files

#: Both retired names, spelled once, as a single alternation: the module path a
#: page told the reader to run, and the class it exported.
RETIRED_PATTERN = re.compile(r"mml_converter|MMLConverter")

#: Directory roots and file roots, mixed. Only the documentation tree: the
#: shipped package and the plugin are covered by the deletion itself, and by
#: their own import-time test suites.
ROOTS = ("docs/source",)

#: The Sphinx output tree is ``docs/build``, which sits outside
#: ``docs/source``, so no build skip is needed. Binary files are excluded by
#: the suffix allowlist, not by a name rule.
SCAN_SUFFIXES = (*SHIPPED_TEXT_SUFFIXES, ".html")

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _sweep(repo_root: Path, roots: tuple[str, ...]) -> list[str]:
    """Report ``path:line:token`` for every retired name still present."""
    files = text_files(repo_root, roots, suffixes=SCAN_SUFFIXES)
    return [
        f"{rel}:{lineno}:{match.group(0)}"
        for rel, lineno, line in matching_lines(repo_root, files, RETIRED_PATTERN)
        for match in RETIRED_PATTERN.finditer(line)
    ]


def _assert_clean(repo_root: Path, roots: tuple[str, ...]) -> None:
    offenders = _sweep(repo_root, roots)
    assert not offenders, (
        "The retired MML converter is still documented. The module "
        "`osprey.services.channel_finder.utils.mml_converter` and its "
        "`MMLConverter` class were removed; a middle-layer database now comes "
        "from `osprey mml import`, `osprey mml map` and `osprey mml emit`:\n" + "\n".join(offenders)
    )


def test_no_live_reference_to_the_retired_converter() -> None:
    _assert_clean(_REPO_ROOT, ROOTS)


@pytest.mark.parametrize(
    ("relative_path", "root"),
    [
        # A directory root, swept recursively.
        ("docs/source/how-to/regress.rst", "docs/source"),
        # A file root, admitted as named.
        ("docs/source/index.rst", "docs/source/index.rst"),
    ],
)
def test_the_sweep_would_catch_a_regression(tmp_path: Path, relative_path: str, root: str) -> None:
    """The guard is only worth having if it fails on a reintroduced name."""
    planted = tmp_path / relative_path
    planted.parent.mkdir(parents=True, exist_ok=True)
    planted.write_text(
        "Run ``python -m osprey.services.channel_finder.utils.mml_converter``.\n",
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="retired MML converter"):
        _assert_clean(tmp_path, (root,))


def test_both_retired_spellings_are_covered() -> None:
    """The module path and the class name are both caught; the new verbs are not."""
    assert RETIRED_PATTERN.search("python -m osprey...utils.mml_converter --input x")
    assert RETIRED_PATTERN.search("the MMLConverter class reads the export")
    assert not RETIRED_PATTERN.search("run osprey mml import on the export")
