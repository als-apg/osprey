"""A profile that names its own limits database stops the profile parse.

The build writes ``data/channel_limits.json`` from ``<data>/facility/limits.yaml``
and names it as the limits database, so a profile that states
``control_system.limits_checking.database_path``, at either scope and in any
spelling, is refused with one ``profile-invalid`` line naming the rendered key.
"""

from __future__ import annotations

from typing import Any

import pytest

from osprey.cli.build_profile_load import _parse_profile
from osprey.cli.build_profile_merge import merge_persona_delta
from osprey.cli.build_profile_resolve import apply_cli_edits
from osprey.facility.errors import FacilityBuildError

WIDE = "control_system.limits_checking.database_path"
PER_TYPE = "control_system.connector.epics.limits_checking.database_path"


def _line(key: str, data: str = "data") -> str:
    return (
        f"facility: profile-invalid: path {key} — the build writes data/channel_limits.json "
        f"from {data}/facility/limits.yaml and names it as the limits database; fix: move its "
        f"limits into {data}/facility/limits.yaml and remove {key} from the profile"
    )


def _refusal(raw: dict[str, Any]) -> str:
    with pytest.raises(FacilityBuildError) as excinfo:
        _parse_profile(raw)
    return str(excinfo.value)


@pytest.mark.parametrize(
    ("config", "key"),
    [
        ({WIDE: "data/my_limits.json"}, WIDE),
        ({"control_system.limits_checking": {"database_path": "x.json"}}, WIDE),
        ({"control_system": {"limits_checking": {"database_path": "x.json"}}}, WIDE),
        ({PER_TYPE: "data/my_limits.json"}, PER_TYPE),
        (
            {"control_system.connector": {"epics": {"limits_checking": {"database_path": "x"}}}},
            PER_TYPE,
        ),
        (
            {
                "control_system": {
                    "connector": {"epics": {"limits_checking": {"database_path": "x"}}}
                }
            },
            PER_TYPE,
        ),
    ],
    ids=["wide-dotted", "wide-prefix", "wide-nested", "type-dotted", "type-prefix", "type-nested"],
)
def test_every_spelling_is_refused_with_the_rendered_key(config: dict[str, Any], key: str) -> None:
    assert _refusal({"config": config}) == _line(key)


def test_the_deployment_wide_key_is_named_before_a_per_type_one() -> None:
    config = {PER_TYPE: "a.json", WIDE: "b.json"}

    assert _refusal({"config": config}) == _line(WIDE)


def test_per_type_keys_are_named_in_sorted_type_order() -> None:
    config = {
        "control_system.connector.virtual_accelerator.limits_checking.database_path": "a.json",
        PER_TYPE: "b.json",
    }

    assert _refusal({"config": config}) == _line(PER_TYPE)


def test_a_persona_delta_stating_the_key_is_refused_when_merged() -> None:
    root = {"config": {"control_system.limits_checking.enabled": True}}
    delta = {"config": {WIDE: "data/persona_limits.json"}}

    assert _refusal(merge_persona_delta(root, delta)) == _line(WIDE)


def test_a_set_edit_stating_the_key_is_refused() -> None:
    edited = apply_cli_edits({"config": {}}, (f"config.{WIDE}=x",))

    assert _refusal(edited) == _line(WIDE)


def test_a_block_stating_only_the_posture_passes() -> None:
    config = {
        "control_system.limits_checking.enabled": True,
        "control_system.limits_checking.mode": "optional",
        "control_system.connector.epics.limits_checking.enabled": True,
        "control_system.connector.epics.limits_checking.mode": "exclusive",
    }

    _parse_profile({"config": config})


@pytest.mark.parametrize(
    ("data", "shown"),
    [("./site-data", "site-data"), ("facility_data/", "facility_data"), (None, "data")],
)
def test_the_data_tree_follows_the_profile(data: str | None, shown: str) -> None:
    raw: dict[str, Any] = {"config": {WIDE: "x.json"}}
    if data is not None:
        raw["data"] = data

    assert _refusal(raw) == _line(WIDE, shown)
