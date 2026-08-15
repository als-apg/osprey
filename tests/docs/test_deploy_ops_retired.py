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

from pathlib import Path

import pytest

# Spelled once. This file is not itself swept — the roots below are the shipped
# package and the docs tree, not tests/ — so the literal is safe here.
RETIRED_SKILL = "osprey-deploy-ops"

ROOTS = ("src/osprey", "docs/source")
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
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _shipped_sources() -> list[Path]:
    files: list[Path] = []
    for root in ROOTS:
        base = _REPO_ROOT / root
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix not in SCAN_SUFFIXES:
                continue
            if "__pycache__" in path.parts:
                continue
            files.append(path)
    return files


def test_no_live_reference_to_the_retired_skill() -> None:
    offenders: list[str] = []
    for path in _shipped_sources():
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if RETIRED_SKILL not in content:
            continue
        hits = [
            f"{i}: {line.strip()}"
            for i, line in enumerate(content.splitlines(), start=1)
            if RETIRED_SKILL in line
        ][:3]
        offenders.append(f"{path.relative_to(_REPO_ROOT)}\n  " + "\n  ".join(hits))

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
