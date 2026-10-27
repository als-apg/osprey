"""Removal gate: the ``osprey-deploy-ops`` skill is gone, with no successor name.

Operating a deployment is the lifecycle commands themselves — ``osprey up``,
``osprey status``, ``osprey logs``, ``osprey health``, ``osprey down`` — not a
separate installable runbook. The skill was removed outright, so no shipped
source or documentation page may still tell someone to install it.

The sweep covers ``src/osprey/`` (including the bundled skills and app README
templates, which are shipped instructions) and ``docs/source/``. ``CHANGELOG.md``
is out of scope by design — it records the removal and the historical entries
that introduced the skill, and rewriting released history to satisfy a grep would
be worse than the grep failing.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests._tree_scan import EXTENSIONLESS_SHIPPED_NAMES, matching_lines, text_files

# Spelled once. This file is not itself swept — the roots below are the shipped
# package and the docs tree, not tests/ — so the literal is safe here.
RETIRED_SKILL = "osprey-deploy-ops"

# plugins/ holds the shipped skills.
ROOTS = ("src/osprey", "docs/source", "plugins")

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_no_live_reference_to_the_retired_skill() -> None:
    files = text_files(_REPO_ROOT, ROOTS, names=EXTENSIONLESS_SHIPPED_NAMES)
    hits = matching_lines(_REPO_ROOT, files, re.compile(re.escape(RETIRED_SKILL)))
    offenders = [f"{path}:{number}: {line.strip()}" for path, number, line in hits]

    assert not offenders, (
        f"The {RETIRED_SKILL} skill was removed — operating a deployment is the "
        "lifecycle commands (`osprey up` / `status` / `logs` / `health`). "
        "Stale references remain:\n" + "\n".join(offenders)
    )


def test_the_sweep_would_catch_a_regression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard is only worth having if it fails on a reintroduced reference."""
    fake_root = tmp_path / "repo"
    (fake_root / "docs" / "source").mkdir(parents=True)
    (fake_root / "docs" / "source" / "regress.rst").write_text(
        f"   osprey skills install {RETIRED_SKILL} --target .claude/skills/\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(f"{__name__}._REPO_ROOT", fake_root, raising=True)

    with pytest.raises(AssertionError, match="Stale references remain"):
        test_no_live_reference_to_the_retired_skill()
