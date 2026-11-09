"""The pinned CANOPY and PyDMConverter install under the versions OSPREY locks.

The project image installs the staged wheels as ``pip install -c constraints.txt
*.whl``, where the constraints are the framework's frozen install minus the two
wheels' own lines (:mod:`osprey.deployment.constraints`). A requirement of either
wheel that excludes the version OSPREY's lock holds for that distribution makes the
image build unresolvable. This lane installs CANOPY under the same kind of
constraints, ``uv export`` of the lock, so such a requirement already fails its
install, but only for the lock's versions on the runner's platform and Python.
So each base requirement (the image installs no extras) is held here to every
version ``uv.lock`` records for its distribution, on every platform, and the
requirement that breaks it is named; a distribution the lock does not hold is the
wheel's to bring and is not judged.
"""

from __future__ import annotations

import importlib.metadata
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

from osprey.integrations.canopy.proven import PROVEN_COMMITS

pytestmark = pytest.mark.canopy

LOCK = Path(__file__).resolve().parents[3] / "uv.lock"


def _locked_versions() -> dict[str, set[Version]]:
    locked: dict[str, set[Version]] = {}
    for package in tomllib.loads(LOCK.read_text(encoding="utf-8"))["package"]:
        if "version" in package:
            name = canonicalize_name(package["name"])
            locked.setdefault(name, set()).add(Version(package["version"]))
    return locked


def unsatisfied(requirements: list[str], locked: dict[str, set[Version]]) -> list[str]:
    """``<requirement> excludes <name>==<version>`` for each base requirement the lock breaks."""
    found: list[str] = []
    for text in requirements:
        requirement = Requirement(text)
        if requirement.marker is not None and not requirement.marker.evaluate({"extra": ""}):
            continue
        name = canonicalize_name(requirement.name)
        for version in sorted(locked.get(name, ())):
            if not requirement.specifier.contains(version, prereleases=True):
                found.append(f"{text} excludes {name}=={version}")
    return found


@pytest.mark.parametrize("distribution", sorted(PROVEN_COMMITS))
def test_the_pinned_wheel_installs_under_the_locked_versions(distribution: str) -> None:
    requirements = importlib.metadata.requires(distribution) or []
    assert unsatisfied(requirements, _locked_versions()) == []


def test_a_requirement_the_lock_excludes_is_named() -> None:
    locked = {"pydantic-settings": {Version("2.11.0")}}
    assert unsatisfied(
        ["pydantic-settings>=2.14.2", "pydantic-settings>=2", "other>=9", "x>=1; extra == 'mcp'"],
        locked,
    ) == ["pydantic-settings>=2.14.2 excludes pydantic-settings==2.11.0"]
