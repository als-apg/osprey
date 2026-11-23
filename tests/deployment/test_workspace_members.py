"""WORKSPACE_MEMBERS and NOT_SHIPPED stay in step with the root uv workspace."""

from __future__ import annotations

import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from osprey.deployment.members import NOT_SHIPPED, WORKSPACE_MEMBERS

_ROOT = Path(__file__).resolve().parents[2]


def _root_pyproject() -> dict:
    return tomllib.loads((_ROOT / "pyproject.toml").read_text())


def _declared_members() -> list[str]:
    return _root_pyproject()["tool"]["uv"]["workspace"]["members"]


def test_members_match_the_root_workspace() -> None:
    """Every declared member is named, in the declared order, and nothing else."""
    assert WORKSPACE_MEMBERS + NOT_SHIPPED == tuple(Path(m).name for m in _declared_members())


def test_shipped_and_not_shipped_are_disjoint() -> None:
    """A member either ships with the framework or does not."""
    assert not set(WORKSPACE_MEMBERS) & set(NOT_SHIPPED)


def test_every_member_directory_is_a_package_with_its_own_name() -> None:
    """The basename doubles as the distribution name the plumbing installs."""
    for member in WORKSPACE_MEMBERS + NOT_SHIPPED:
        member_pyproject = tomllib.loads(
            (_ROOT / "packages" / member / "pyproject.toml").read_text()
        )
        assert member_pyproject["project"]["name"] == member


def test_the_framework_never_depends_on_an_unshipped_member() -> None:
    """No [project] dependency or extra of the framework names a NOT_SHIPPED member."""
    project = _root_pyproject()["project"]
    lists = {"dependencies": project["dependencies"]}
    lists.update(
        {f"optional-dependencies.{k}": v for k, v in project["optional-dependencies"].items()}
    )
    unshipped = {canonicalize_name(m) for m in NOT_SHIPPED}
    named = {
        (where, spec)
        for where, specs in lists.items()
        for spec in specs
        if canonicalize_name(Requirement(spec).name) in unshipped
    }
    assert not named


def test_unshipped_members_reach_the_checkout_through_the_dev_group() -> None:
    """The root dev group and a workspace source give the checkout every NOT_SHIPPED member."""
    pyproject = _root_pyproject()
    dev = {canonicalize_name(Requirement(s).name) for s in pyproject["dependency-groups"]["dev"]}
    sources = pyproject["tool"]["uv"]["sources"]
    for member in NOT_SHIPPED:
        assert canonicalize_name(member) in dev
        assert sources[member] == {"workspace": True}
