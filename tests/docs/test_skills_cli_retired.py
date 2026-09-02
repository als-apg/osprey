"""Removal gate: the ``osprey skills`` CLI and the old skill names are gone.

The six skills are no longer installed by an OSPREY subcommand copying files out
of ``src/osprey/templates/skills/``. They ship as one Claude Code plugin under
``plugins/osprey/skills/`` and are invoked as ``/osprey:<name>`` --- so the
``osprey skills`` command group, the ``osprey-`` prefixed slash-command names it
created, and the template directory it copied from must not come back.

Scope of the sweep
------------------
``ROOTS`` is the shipped package, the docs tree, the plugin, and the two
repository-root documents plus ``.github`` (workflows are executable
instructions like any other). Three trees are deliberately outside it:

* ``CHANGELOG.md``, ``RELEASE_NOTES.md`` and ``changelog.d/`` --- released
  history records that the command existed and that it was removed. Rewriting
  shipped release notes to satisfy a grep would be worse than the grep failing.
* ``tests/`` --- the pinned literals in the test suite were flipped by hand, and
  a guard that swept its own siblings would fail on their fixtures.
* This file --- it spells every retired name once, in the pattern below. It is
  under ``tests/``, so it is not swept, exactly as the sibling gate
  (``test_deploy_ops_retired.py``) arranges for its own literal.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

#: Every retired name, spelled once, as a single alternation.
#:
#: The ``(?<!started)`` lookbehind on ``/osprey-build-interview`` is the one
#: tolerated survival. The deployer-facing page kept its filename when the skill
#: was renamed, so Sphinx ``:doc:`` roles and toctree entries legitimately read
#: ``getting-started/osprey-build-interview``. That is a page path, not a slash
#: command. The retired *command* is ``/osprey-build-interview`` at a word
#: boundary, and only the page path is ever preceded by ``started``.
RETIRED_PATTERN = re.compile(
    r"osprey skills"
    r"|(?<!started)/osprey-build-interview"
    r"|osprey-contribute"
    r"|osprey-pre-commit"
    r"|osprey-release"
    r"|osprey-design-philosophy"
    r"|creating-an-osprey-panel"
    r"|templates/skills"
)

#: Directory roots and file roots, mixed. ``_collect`` admits either.
ROOTS = (
    "src/osprey",
    "docs/source",
    "plugins",
    "README.md",
    "CONTRIBUTING.md",
    ".github",
)

SCAN_SUFFIXES = (
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

#: Shipped files with no extension, admitted by name --- the same clause as the
#: sibling gates: a suffix-only rule cannot see the ``gitignore``/
#: ``dockerignore``/``Dockerfile`` templates the scaffold emits verbatim.
#: Matched with any leading dot stripped, because the same kind of file ships
#: both ways.
SCAN_STEMS = frozenset({"gitignore", "dockerignore", "Dockerfile"})

#: ``__pycache__`` is the only directory pruned by name. There is no ``build/``
#: skip on purpose: ``src/osprey/build/`` is a real shipped package, and the
#: Sphinx output tree is ``docs/build``, which sits outside ``docs/source``.
#: Binary files are excluded by the suffix allowlist, not by a name rule.
PRUNED_DIRS = frozenset({"__pycache__"})

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _admits(path: Path) -> bool:
    if any(part in PRUNED_DIRS for part in path.parts):
        return False
    return path.suffix in SCAN_SUFFIXES or path.name.lstrip(".") in SCAN_STEMS


def _collect(repo_root: Path, roots: tuple[str, ...]) -> list[Path]:
    """Every scannable file under ``roots``, which may name files or directories."""
    files: list[Path] = []
    for root in roots:
        base = repo_root / root
        if not base.exists():
            continue
        if base.is_file():
            # A file root is named explicitly, so it is admitted as named.
            # ``rglob`` on a file yields nothing, which would silently drop
            # README.md and CONTRIBUTING.md from the sweep.
            files.append(base)
            continue
        for path in base.rglob("*"):
            if path.is_file() and _admits(path):
                files.append(path)
    return files


def _sweep(repo_root: Path, roots: tuple[str, ...]) -> list[str]:
    """Report ``path:line:token`` for every retired name still present."""
    offenders: list[str] = []
    for path in _collect(repo_root, roots):
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rel = path.relative_to(repo_root)
        for lineno, line in enumerate(content.splitlines(), start=1):
            for match in RETIRED_PATTERN.finditer(line):
                offenders.append(f"{rel}:{lineno}:{match.group(0)}")
    return offenders


def _assert_clean(repo_root: Path, roots: tuple[str, ...]) -> None:
    offenders = _sweep(repo_root, roots)
    assert not offenders, (
        "Retired skill-install names remain. The `osprey skills` command group "
        "and `src/osprey/templates/skills/` were removed; the six skills ship "
        "as the `plugins/osprey` plugin and are invoked as `/osprey:<name>`:\n"
        + "\n".join(offenders)
    )


def test_no_live_reference_to_the_retired_skills_cli() -> None:
    _assert_clean(_REPO_ROOT, ROOTS)


@pytest.mark.parametrize(
    ("relative_path", "root"),
    [
        # A directory root, swept recursively.
        ("docs/source/regress.rst", "docs/source"),
        # A file root, admitted directly by ``_collect``.
        ("README.md", "README.md"),
    ],
)
def test_the_sweep_would_catch_a_regression(tmp_path: Path, relative_path: str, root: str) -> None:
    """The guard is only worth having if it fails on a reintroduced name."""
    planted = tmp_path / relative_path
    planted.parent.mkdir(parents=True, exist_ok=True)
    planted.write_text(
        "Run ``osprey skills install osprey-design-philosophy`` to install it.\n",
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="Retired skill-install names remain"):
        _assert_clean(tmp_path, (root,))


def test_the_documentation_page_filename_is_tolerated() -> None:
    """``:doc:`` roles and toctree entries keep the old page filename."""
    kept = "- :doc:`/getting-started/osprey-build-interview` --- the deployer page"
    assert not RETIRED_PATTERN.search(kept)
    assert RETIRED_PATTERN.search("Invoke /osprey-build-interview to start")
