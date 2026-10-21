"""WORKSPACE_MEMBERS stays in step with the root uv workspace."""

from __future__ import annotations

import tomllib
from pathlib import Path

from osprey.deployment.members import WORKSPACE_MEMBERS

_ROOT = Path(__file__).resolve().parents[2]


def _declared_members() -> list[str]:
    pyproject = tomllib.loads((_ROOT / "pyproject.toml").read_text())
    return pyproject["tool"]["uv"]["workspace"]["members"]


def test_members_match_the_root_workspace() -> None:
    """Every declared member is named, in the declared order, and nothing else."""
    assert WORKSPACE_MEMBERS == tuple(Path(m).name for m in _declared_members())


def test_every_member_directory_is_a_package_with_its_own_name() -> None:
    """The basename doubles as the distribution name the plumbing installs."""
    for member in WORKSPACE_MEMBERS:
        member_pyproject = tomllib.loads(
            (_ROOT / "packages" / member / "pyproject.toml").read_text()
        )
        assert member_pyproject["project"]["name"] == member
