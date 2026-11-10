"""A kind's settings transform patches its model's engine settings, once, at combine.

A kind may declare ``settings_from_measurement(model_entry, measurement_record)``
returning ``{<engine>: {<setting>: value}}``. The build applies the patches of
the kinds the measurement file lists, in the registry's order, to the combined
document before the wiring and compute checks, so the engine's own checks, the
simulator view and every other reader see one settings map. The fixture kind
patches ``pyat.rest_mass_gev``, a setting the pyAT engine validates.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from osprey.facility import kinds
from osprey.facility.errors import FacilityBuildError
from osprey.facility.scenarios import fault_roster
from osprey.facility.views.simulator import write_simulator_view
from osprey_connectors.simulation.view import VARIABLES_FILE
from tests.facility._pyaml_trees import (
    built_document,
    measured_tree,
    view_inputs,
    with_correctors,
    with_rf,
)
from tests.facility.fixtures.registry_kind import REGISTRY_KIND, install

pytest.importorskip("pyaml")

#: What the fixture kind patches for a trigger sample of 12: the sample plus its window.
_PATCHED = {"pyat": {"rest_mass_gev": 12.0 + 64}}

_SOURCE = "package registry-kind-fixture"


def _tree(kinds_listed: list[str] | None = None, **block: Any) -> dict[str, Any]:
    tree = with_correctors(with_rf(measured_tree()))
    tree["measurement/SR.yaml"] |= {
        "kinds": kinds_listed or [REGISTRY_KIND],
        "corrector_delta": 1.0e-5,
        "kind_blocks": {REGISTRY_KIND: {"trigger_sample": 12, **block}},
    }
    return tree


def _sr(document: dict[str, Any]) -> dict[str, Any]:
    return next(model for model in document["models"] if model["name"] == "SR")


def _stop(tmp_path: Path, tree: dict[str, Any]) -> str:
    with pytest.raises(FacilityBuildError) as stopped:
        built_document(tmp_path / "tree", tree)
    return str(stopped.value.format_message())


def _hooks(monkeypatch: pytest.MonkeyPatch, **hooks: Callable[..., Any]) -> None:
    """The fixture kind installed, each named kind's transform replaced by its hook."""
    install(monkeypatch)
    found, sources = kinds._load()
    patched = {
        name: dataclasses.replace(declaration, settings_from_measurement=hooks[name])
        if name in hooks
        else declaration
        for name, declaration in found.items()
    }
    monkeypatch.setattr(kinds, "_load", lambda: (patched, sources))


def test_every_built_in_kind_patches_nothing(tmp_path):
    document, _facility = built_document(tmp_path / "tree", measured_tree())
    sr = _sr(document)

    assert {
        name: kinds.settings_patch(declaration, sr)
        for name, declaration in kinds.builtin_kinds().items()
    } == dict.fromkeys(("chromaticity_monitor", "crm", "dispersion", "orm", "trm"), {})
    assert "settings" not in sr
    assert sr["provenance"]["defaults"] == []


def test_a_listed_kinds_patch_reaches_the_document_and_the_simulator_view(tmp_path, monkeypatch):
    install(monkeypatch)
    tree = _tree()

    document, _facility = built_document(tmp_path / "tree", tree)

    sr = _sr(document)
    assert sr["settings"] == _PATCHED
    assert sr["provenance"]["defaults"] == ["settings.pyat.rest_mass_gev"]
    view = tmp_path / "view"
    write_simulator_view(view, view_inputs(tmp_path / "again", tree))
    variables = json.loads((view / VARIABLES_FILE).read_text("utf-8"))
    (served,) = [model for model in variables["models"] if model["name"] == "SR"]
    assert served["settings"] == _PATCHED


def test_the_fault_roster_reads_the_patched_settings(tmp_path, monkeypatch):
    error_elements = {"pyat": {"error_elements": [{"element": "SX"}]}}
    _hooks(monkeypatch, **{REGISTRY_KIND: lambda model, measurement: error_elements})

    document, _facility = built_document(tmp_path / "tree", _tree())

    sr = _sr(document)
    assert sr["settings"] == error_elements
    roster = fault_roster(sr, {})
    assert roster is not None
    assert {"SX/dx", "SX/dy", "SX/roll"} <= set(roster.slots)


def test_the_patch_merges_under_the_engine_beside_the_hand_written_settings(tmp_path, monkeypatch):
    install(monkeypatch)
    tree = _tree()
    tree["models.yaml"][0]["settings"] = {"pyat": {"solve": "periodic"}}

    document, _facility = built_document(tmp_path / "tree", tree)

    assert _sr(document)["settings"] == {"pyat": {"solve": "periodic", **_PATCHED["pyat"]}}


def test_a_patch_of_a_hand_written_setting_stops_the_build_naming_both(tmp_path, monkeypatch):
    install(monkeypatch)
    tree = _tree()
    tree["models.yaml"][0]["settings"] = {"pyat": {"rest_mass_gev": 0.000511}}

    assert _stop(tmp_path, tree) == (
        f"facility: value-invalid: model SR — kind {REGISTRY_KIND} ({_SOURCE}) sets "
        "`settings.pyat.rest_mass_gev`, which models.yaml states by hand; fix: remove "
        f"`settings.pyat.rest_mass_gev` from models.yaml, or remove {REGISTRY_KIND} from `kinds`"
    )


def test_two_kinds_patching_one_setting_stop_the_build_in_the_registrys_order(
    tmp_path, monkeypatch
):
    _hooks(monkeypatch, trm=lambda model, measurement: {"pyat": {"rest_mass_gev": 1.0}})

    assert _stop(tmp_path, _tree([REGISTRY_KIND, "trm"])) == (
        f"facility: value-invalid: measurement SR — kinds trm (built-in) and {REGISTRY_KIND} "
        f"({_SOURCE}) both set `settings.pyat.rest_mass_gev`; fix: list one of trm and "
        f"{REGISTRY_KIND} in `kinds`"
    )


def test_an_engine_rejected_patch_stops_the_build(tmp_path, monkeypatch):
    _hooks(
        monkeypatch,
        **{REGISTRY_KIND: lambda model, measurement: {"pyat": {"rest_mass_gev": -1.0}}},
    )

    message = _stop(tmp_path, _tree())

    assert message.startswith("facility: engine-invalid: model SR")
    assert "settings key pyat.rest_mass_gev is -1.0" in message


def test_the_transform_is_handed_the_model_entry_and_its_measurement_record(tmp_path, monkeypatch):
    seen: list[tuple[dict[str, Any], dict[str, Any]]] = []

    def hook(model: dict[str, Any], measurement: dict[str, Any]) -> dict[str, Any]:
        seen.append((model, measurement))
        model["engine"] = "edited"
        return {}

    _hooks(monkeypatch, **{REGISTRY_KIND: hook})

    document, _facility = built_document(tmp_path / "tree", _tree())

    ((model, measurement),) = seen
    assert model["name"] == "SR"
    assert measurement["kind_blocks"][REGISTRY_KIND] == {"trigger_sample": 12}
    assert _sr(document)["engine"] == "pyat"
    assert "settings" not in _sr(document)


def test_a_block_the_kinds_schema_refuses_is_never_handed_to_the_transform(tmp_path, monkeypatch):
    def hook(*handed: Any) -> Any:
        raise AssertionError(f"the transform ran on a refused block: {handed}")

    _hooks(monkeypatch, **{REGISTRY_KIND: hook})
    tree = _tree()
    tree["measurement/SR.yaml"]["kind_blocks"][REGISTRY_KIND] = {"trigger_sample": -1}

    assert _stop(tmp_path, tree) == (
        f"facility: value-invalid: measurement SR — `kind_blocks.{REGISTRY_KIND}.trigger_sample: "
        "Input should be greater than or equal to 0`; fix: correct "
        f"`kind_blocks.{REGISTRY_KIND}` as kind {REGISTRY_KIND}'s measurement schema states"
    )


@pytest.mark.parametrize(
    ("hook", "cause"),
    [
        pytest.param(
            lambda model, measurement: measurement["kind_blocks"]["absent"],
            "KeyError: 'absent'",
            id="raises",
        ),
        pytest.param(
            lambda model, measurement: ["pyat"],
            "TypeError: the settings transform returned a list, not a mapping",
            id="not-a-mapping",
        ),
        pytest.param(
            lambda model, measurement: {"pyat": 1.0},
            "TypeError: the settings transform's patch of pyat is a float, not a mapping",
            id="engine-not-a-mapping",
        ),
        pytest.param(
            lambda model, measurement: {"elegant": {"turns": 4}},
            "ValueError: the settings transform patches engine elegant; model SR runs pyat",
            id="other-engine",
        ),
    ],
)
def test_a_transform_that_fails_stops_the_build_with_one_line(tmp_path, monkeypatch, hook, cause):
    _hooks(monkeypatch, **{REGISTRY_KIND: hook})

    assert _stop(tmp_path, _tree()) == (
        f"facility: value-invalid: measurement SR — kind {REGISTRY_KIND}'s settings transform "
        f"fails on this file's block: {cause}; fix: make the settings transform of kind "
        f"{REGISTRY_KIND} ({_SOURCE}) accept every block its measurement schema admits"
    )


def test_block_admitted_states_which_sub_block_a_kind_takes(monkeypatch):
    install(monkeypatch)
    found, _sources = kinds._load()
    plain, schema = found["orm"], found[REGISTRY_KIND]

    assert kinds.block_admitted(plain, None)
    assert not kinds.block_admitted(plain, {"trigger_sample": 12})
    assert kinds.block_admitted(schema, {"trigger_sample": 12})
    assert not kinds.block_admitted(schema, {"trigger_sample": -1})
