"""The ``simulation.models`` key: which models a render serves.

``resolve_served`` reads the key from a rendered config and the model names
from the facility file; a persona delta that sets the key on a deployment whose
baseline target is a VA instance stops at resolution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from osprey.cli.build_profile_archiver import _expand_dotted
from osprey.cli.build_profile_load import load_profile_document
from osprey.cli.build_profile_resolve import resolve_build_document
from osprey.facility import TEXTURE
from osprey.facility.build import build_facility
from osprey.facility.errors import FacilityBuildError
from osprey.facility.served import resolve_served

_DEMO_FACILITY = (
    Path(__file__).resolve().parents[2]
    / "src/osprey/templates/apps/control_assistant/data/facility"
)

#: The four shipped presets every all-templates key must appear in.
_PRESETS = ("hello-world", "ariel-standalone", "channel-finder-standalone", "control-assistant")


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


@pytest.mark.parametrize("preset", _PRESETS)
def test_every_preset_spells_the_key_as_null(preset: str) -> None:
    config = _expand_dotted(resolve_build_document(None, preset).profile.config)
    assert "models" in config.get("simulation", {})
    assert config["simulation"]["models"] is None


def _repo(tmp_path: Path, connector: str, personas: dict[str, str]) -> Path:
    (tmp_path / "data").mkdir()
    (tmp_path / "profile.yml").write_text(
        f"name: demo\ndata: data\nconfig:\n  control_system.type: {connector}\n"
        "  simulation.models: null\n"
    )
    (tmp_path / "personas").mkdir()
    for name, body in personas.items():
        (tmp_path / "personas" / f"{name}.yml").write_text(f"name: {name}\n{body}")
    return tmp_path


def _served_for(delta: Path, facility: dict[str, Any]) -> list[str]:
    config = _expand_dotted(load_profile_document(delta).profile.config)
    return resolve_served(config, facility)


def test_two_mock_personas_differing_only_in_the_key_resolve_different_lists(
    tmp_path: Path, demo: dict[str, Any]
) -> None:
    repo = _repo(
        tmp_path,
        "mock",
        {
            "physics": "config:\n  simulation.models: [SR]\n",
            "plain": "config:\n  simulation.models: [texture]\n",
        },
    )
    physics = _served_for(repo / "personas" / "physics.yml", demo)
    plain = _served_for(repo / "personas" / "plain.yml", demo)
    assert physics == ["SR", TEXTURE]
    assert plain == [TEXTURE]


@pytest.mark.parametrize("connector", ["virtual_accelerator", "live_standin"])
@pytest.mark.parametrize(
    "body",
    ["config:\n  simulation.models: [texture]\n", "config:\n  simulation:\n    models: null\n"],
)
def test_a_persona_setting_the_key_on_a_va_baseline_stops(
    tmp_path: Path, connector: str, body: str
) -> None:
    repo = _repo(tmp_path, connector, {"reader": body})
    with pytest.raises(FacilityBuildError) as excinfo:
        load_profile_document(repo / "personas" / "reader.yml")
    error = excinfo.value
    assert error.kind == "profile-invalid"
    assert error.record_id == "personas/reader.yml"
    assert "`simulation.models`" in error.detail


def test_a_persona_that_moves_itself_onto_a_va_baseline_stops(tmp_path: Path) -> None:
    repo = _repo(
        tmp_path,
        "mock",
        {
            "reader": (
                "config:\n  control_system.type: virtual_accelerator\n  simulation.models: [SR]\n"
            )
        },
    )
    with pytest.raises(FacilityBuildError) as excinfo:
        resolve_build_document(repo / "personas" / "reader.yml", None)
    assert excinfo.value.kind == "profile-invalid"


def test_a_persona_leaving_the_key_alone_on_a_va_baseline_resolves(tmp_path: Path) -> None:
    import yaml

    from osprey.cli.build_profile_merge import resolve_profile_document

    repo = _repo(tmp_path, "virtual_accelerator", {"reader": "config:\n  web.theme: dark\n"})
    delta = repo / "personas" / "reader.yml"
    document = resolve_profile_document(yaml.safe_load(delta.read_text()), delta)
    assert document.is_persona_delta
