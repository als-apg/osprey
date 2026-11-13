"""Tests for the profile's ``machine:`` key and its resolution into facility roots.

``machine: <dir>`` names the facility directory of a mounted machine repository,
relative to the profile directory. Unset, ``<data>/facility`` is the one tree
playing both roles. ``BuildProfile.resolved_facility_roots`` is the single
resolution point the build, ``osprey facility`` and ``osprey validate`` share.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_profile import (
    _KNOWN_PROFILE_KEYS,
    BuildProfile,
    load_profile,
)
from osprey.cli.build_profile_load import ROOT_ONLY_KEYS, TIER_KIND_KEYS
from osprey.cli.init_cmd import init
from osprey.cli.validate_cmd import check_profile_file, validate
from osprey.facility.errors import KINDS, FacilityBuildError
from osprey.facility.roots import FacilityRoots
from osprey_connectors.errors import BuildProfileError


def _write_yaml(path: Path, body: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return path


def _machine(root: Path) -> Path:
    """A minimal machine root: a facility directory holding ``identity.yaml``."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "identity.yaml").write_text("code: m\nname: M\n", encoding="utf-8")
    return root


def _profile(tmp_path: Path, **keys: Any) -> BuildProfile:
    (tmp_path / "data" / "facility").mkdir(parents=True, exist_ok=True)
    return BuildProfile(name="p", data="data", **keys)


# ---------------------------------------------------------------------------
# The key itself
# ---------------------------------------------------------------------------


def test_machine_is_a_known_root_only_key() -> None:
    assert "machine" in _KNOWN_PROFILE_KEYS
    assert "machine" in ROOT_ONLY_KEYS
    assert "machine" not in TIER_KIND_KEYS


def test_machine_errors_are_registered_kinds() -> None:
    assert "machine-missing" in KINDS
    assert "machine-restated" in KINDS


def test_profile_file_carries_machine(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    _machine(tmp_path / "va" / "facility")
    profile = _write_yaml(
        tmp_path / "profile.yml", {"name": "p", "data": "data", "machine": "va/facility"}
    )

    assert load_profile(profile).machine == "va/facility"


@pytest.mark.parametrize("value", [42, "", "   ", ["va"]])
def test_non_path_machine_values_are_rejected(tmp_path: Path, value: Any) -> None:
    (tmp_path / "data").mkdir()
    profile = _write_yaml(tmp_path / "profile.yml", {"name": "p", "data": "data", "machine": value})

    with pytest.raises(BuildProfileError, match="machine must be a non-empty directory path"):
        load_profile(profile)


def test_a_tier_stating_machine_is_refused(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    _machine(tmp_path / "va")
    _write_yaml(tmp_path / "profile.yml", {"name": "root", "data": "data"})
    tier = _write_yaml(tmp_path / "tiers" / "alice.yml", {"name": "alice", "machine": "va"})

    with pytest.raises((BuildProfileError, FacilityBuildError)) as excinfo:
        load_profile(tier)

    message = str(excinfo.value)
    assert "machine" in message
    assert "tiers/alice.yml" in message


# ---------------------------------------------------------------------------
# resolved_facility_roots
# ---------------------------------------------------------------------------


def test_no_data_gives_no_roots(tmp_path: Path) -> None:
    assert BuildProfile(name="p").resolved_facility_roots(tmp_path) is None


def test_unset_machine_gives_a_single_root(tmp_path: Path) -> None:
    roots = _profile(tmp_path).resolved_facility_roots(tmp_path)

    facility = (tmp_path / "data").resolve() / "facility"
    assert roots == FacilityRoots.single(facility)
    assert not roots.mounted


def test_set_machine_gives_a_split(tmp_path: Path) -> None:
    machine = _machine(tmp_path / "va" / "facility")

    roots = _profile(tmp_path, machine="va/facility").resolved_facility_roots(tmp_path)

    assert roots is not None
    assert roots.mounted
    assert roots.machine == machine.resolve()
    assert roots.content == (tmp_path / "data").resolve() / "facility"


def test_an_empty_machine_directory_is_machine_missing(tmp_path: Path) -> None:
    (tmp_path / "va").mkdir()

    with pytest.raises(FacilityBuildError) as excinfo:
        _profile(tmp_path, machine="va").resolved_facility_roots(tmp_path)

    assert excinfo.value.kind == "machine-missing"
    assert "va/identity.yaml" in str(excinfo.value)


def test_an_absent_machine_directory_is_machine_missing(tmp_path: Path) -> None:
    with pytest.raises(FacilityBuildError) as excinfo:
        _profile(tmp_path, machine="va").resolved_facility_roots(tmp_path)

    assert excinfo.value.kind == "machine-missing"


def test_a_machine_without_identity_is_machine_missing(tmp_path: Path) -> None:
    (tmp_path / "va" / "records").mkdir(parents=True)
    (tmp_path / "va" / "classes.yaml").write_text("{}\n", encoding="utf-8")

    with pytest.raises(FacilityBuildError) as excinfo:
        _profile(tmp_path, machine="va").resolved_facility_roots(tmp_path)

    assert excinfo.value.kind == "machine-missing"
    assert "va/identity.yaml" in str(excinfo.value)
    assert "data/facility exists" not in str(excinfo.value)


def test_naming_the_repository_top_gets_the_hint(tmp_path: Path) -> None:
    _machine(tmp_path / "machine" / "data" / "facility")

    with pytest.raises(FacilityBuildError) as excinfo:
        _profile(tmp_path, machine="machine").resolved_facility_roots(tmp_path)

    assert excinfo.value.kind == "machine-missing"
    message = str(excinfo.value)
    assert "machine/data/facility exists" in message
    assert "machine: machine/data/facility" in message


def test_a_machine_inside_the_data_tree_is_refused(tmp_path: Path) -> None:
    _machine(tmp_path / "data" / "machine")

    with pytest.raises(FacilityBuildError) as excinfo:
        _profile(tmp_path, machine="data/machine").resolved_facility_roots(tmp_path)

    assert excinfo.value.kind == "profile-invalid"
    assert "overlap" in str(excinfo.value)


def test_a_data_tree_inside_the_machine_root_is_refused(tmp_path: Path) -> None:
    _machine(tmp_path / "va")
    (tmp_path / "va" / "data" / "facility").mkdir(parents=True)
    profile = BuildProfile(name="p", data="va/data", machine="va")

    with pytest.raises(FacilityBuildError) as excinfo:
        profile.resolved_facility_roots(tmp_path)

    assert excinfo.value.kind == "profile-invalid"


def test_the_data_tree_itself_as_machine_is_refused(tmp_path: Path) -> None:
    _machine(tmp_path / "data")

    with pytest.raises(FacilityBuildError) as excinfo:
        _profile(tmp_path, machine="data").resolved_facility_roots(tmp_path)

    assert excinfo.value.kind == "profile-invalid"


# ---------------------------------------------------------------------------
# osprey validate
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def materialized(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One real materialization of the demo preset."""
    root = tmp_path_factory.mktemp("machine-key") / "demo"
    result = CliRunner().invoke(init, [str(root), "--preset", "demo", "--no-git"])
    assert result.exit_code == 0, result.output
    return root


@pytest.fixture
def repo(materialized: Path, tmp_path: Path) -> Path:
    target = tmp_path / "demo"
    shutil.copytree(materialized, target)
    return target


def _state_machine(repo: Path, location: str) -> Path:
    profile = repo / "profile.yml"
    text = profile.read_text(encoding="utf-8")
    profile.write_text(text.rstrip("\n") + f"\nmachine: {location}\n", encoding="utf-8")
    return profile


def test_validate_reports_an_empty_submodule(repo: Path) -> None:
    (repo / "va").mkdir()
    profile = _state_machine(repo, "va")

    result = CliRunner().invoke(validate, [str(profile)])

    assert result.exit_code != 0
    assert "machine-missing" in result.output
    assert "va/identity.yaml" in result.output


def test_a_materialized_profile_stating_machine_passes(repo: Path) -> None:
    _machine(repo / "va" / "facility")
    profile = _state_machine(repo, "va/facility")

    check_profile_file(profile)
