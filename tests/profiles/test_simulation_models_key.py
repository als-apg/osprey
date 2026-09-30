"""The ``simulation.models`` key: which models a render serves.

``resolve_served`` reads the key from a rendered config and the model names
from the facility file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from osprey.facility import TEXTURE
from osprey.facility.build import build_facility
from osprey.facility.errors import FacilityBuildError
from osprey.facility.served import resolve_served

_DEMO_FACILITY = (
    Path(__file__).resolve().parents[2]
    / "src/osprey/templates/apps/control_assistant/data/facility"
)


@pytest.fixture(scope="module")
def demo() -> dict[str, Any]:
    return build_facility(_DEMO_FACILITY, project_name="demo")


def _config(models: Any) -> dict[str, Any]:
    return {"simulation": {"models": models}}


def test_null_and_absent_resolve_byte_equal_lists(demo: dict[str, Any]) -> None:
    absent = resolve_served({}, demo)
    null = resolve_served(_config(None), demo)
    block_without_key = resolve_served({"simulation": {}}, demo)
    assert json.dumps(null).encode() == json.dumps(absent).encode()
    assert block_without_key == absent
    assert absent == ["SR", TEXTURE]


@pytest.mark.parametrize("models", [[], [TEXTURE]])
def test_texture_or_empty_serves_no_physics(demo: dict[str, Any], models: list[str]) -> None:
    assert resolve_served(_config(models), demo) == [TEXTURE]


def test_a_named_model_is_served_with_texture_last(demo: dict[str, Any]) -> None:
    assert resolve_served(_config([TEXTURE, "SR", "SR"]), demo) == ["SR", TEXTURE]


def test_an_unknown_model_stops_naming_every_valid_name(demo: dict[str, Any]) -> None:
    with pytest.raises(FacilityBuildError) as excinfo:
        resolve_served(_config(["SR", "NOPE"]), demo)
    error = excinfo.value
    assert error.kind == "profile-invalid"
    assert error.record_id == "simulation.models"
    assert "`NOPE`" in error.detail
    assert "`SR`, `texture`" in error.detail


@pytest.mark.parametrize("models", ["SR", 3, [["SR"]], {"SR": True}])
def test_a_value_that_is_not_a_list_of_names_stops(demo: dict[str, Any], models: Any) -> None:
    with pytest.raises(FacilityBuildError) as excinfo:
        resolve_served(_config(models), demo)
    assert excinfo.value.kind == "profile-invalid"
    assert excinfo.value.record_id == "simulation.models"
