"""The limits-database remedy names where the limits actually live.

With ``machine:`` in the profile, the facility's ``limits.yaml`` lives in the
machine repository, so the remedy for a profile that names its own limits
database points there. Without ``machine:``, it names the deployment's own
``facility/limits.yaml``.
"""

from __future__ import annotations

from typing import Any

import pytest

from osprey.cli.build_profile_load import _check_limits_database_path, _parse_profile
from osprey_facility.errors import FacilityBuildError

WIDE = "control_system.limits_checking.database_path"


def _line(key: str, limits: str) -> str:
    return (
        f"facility: profile-invalid: path {key} — the build writes data/channel_limits.json "
        f"from {limits} and names it as the limits database; fix: move its "
        f"limits into {limits} and remove {key} from the profile"
    )


def _refusal(raw: dict[str, Any]) -> str:
    with pytest.raises(FacilityBuildError) as excinfo:
        _check_limits_database_path(raw)
    return str(excinfo.value)


@pytest.mark.parametrize(
    ("machine", "limits"),
    [
        ("machine", "machine/limits.yaml"),
        ("machine/", "machine/limits.yaml"),
        ("vendor/als-machine", "vendor/als-machine/limits.yaml"),
        ("./machine", "machine/limits.yaml"),
    ],
)
def test_with_machine_the_remedy_names_the_machine_limits(machine: str, limits: str) -> None:
    raw = {"machine": machine, "config": {WIDE: "x.json"}}

    assert _refusal(raw) == _line(WIDE, limits)


def test_with_machine_the_remedy_does_not_name_the_facility_tree() -> None:
    raw = {"machine": "machine", "config": {WIDE: "x.json"}}

    text = _refusal(raw)

    assert "facility/limits.yaml" not in text
    assert "machine/limits.yaml" in text


@pytest.mark.parametrize(
    "raw_extra",
    [{}, {"machine": None}, {"machine": ""}],
    ids=["no-machine", "machine-null", "machine-empty"],
)
def test_without_machine_the_remedy_names_the_facility_limits(raw_extra: dict[str, Any]) -> None:
    raw = {**raw_extra, "config": {WIDE: "x.json"}}

    assert _refusal(raw) == _line(WIDE, "facility/limits.yaml")


def test_the_profile_parse_carries_the_machine_remedy() -> None:
    with pytest.raises(FacilityBuildError) as excinfo:
        _parse_profile({"machine": "machine", "config": {WIDE: "x.json"}})

    assert str(excinfo.value) == _line(WIDE, "machine/limits.yaml")


def test_the_docstring_records_why_runtime_strings_stay() -> None:
    doc = _check_limits_database_path.__doc__ or ""

    assert "machine" in doc
    assert "D7" in doc or "render" in doc
