"""The new-format mapping each MML fixture tree carries at ``imported/mml/mapping.yaml``.

Every file answers the whole export beside it: it loads through the mml
layer's reader with no problem, leaves no slot undecided, gives every family
field the export carries a direction, and names the wiring of every model the
tree wires. A test that imports a tree copies its file to
``data/facility/imported/mml/mapping.yaml`` first, which is what the
``load_or_draft`` case does here.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from osprey.facility.layers.mml.mapping import (
    MAPPING_FILE,
    EngineBlock,
    Mapping,
    check_mapping,
    field_roles,
    load_or_draft,
    read_mapping,
    require_decided,
)
from osprey.services.mml.family import family_views, system_bodies
from osprey.services.mml.loaders.json_any import VA_SUFFIX, load_json, load_sibling
from osprey.services.mml.systems import merge_inputs

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "mml"

#: Each tree's exports, by the stem every file of one export is named after.
TREES: dict[str, tuple[str, ...]] = {
    "spear3": ("spear3.storagering",),
    "nsls2": ("nsls2.storagering", "nsls2.ltb"),
    "synthetic": ("quokka.sr",),
}


def _export(tree: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """The tree's merged export: ``(ao, ad, va)``, each keyed by system."""
    loaded = [load_json(FIXTURES / tree / f"{stem}.ao.json") for stem in TREES[tree]]
    ao, ad = merge_inputs([(one, None) for one in loaded])
    va = {
        one.export["submachine"]: load_sibling(FIXTURES / tree / f"{stem}{VA_SUFFIX}")
        for stem, one in zip(TREES[tree], loaded, strict=True)
        if one.export is not None
    }
    return ao, ad, va


def _mapping(tree: str) -> Mapping:
    return read_mapping(FIXTURES / tree / MAPPING_FILE)


@pytest.mark.parametrize("tree", sorted(TREES))
def test_the_mapping_checks_clean_against_its_export(tree: str) -> None:
    ao, _ad, _va = _export(tree)
    mapping = _mapping(tree)
    assert [str(problem) for problem in check_mapping(mapping, ao)] == []
    require_decided(mapping, ao)


@pytest.mark.parametrize("tree", sorted(TREES))
def test_every_exported_family_field_has_a_decided_direction(tree: str) -> None:
    ao, _ad, _va = _export(tree)
    mapping = _mapping(tree)
    carried = {
        f"{view.raw_name}.{name}"
        for system, body in system_bodies(ao)
        for view in family_views(system, body)
        for name in view.fields
    }
    assert carried
    assert carried <= set(mapping.directions)
    assert {mapping.directions[key].direction for key in carried} <= {"read", "write"}
    assert set(field_roles(mapping)) == set(mapping.directions)


@pytest.mark.parametrize("tree", sorted(TREES))
def test_a_copy_under_data_facility_loads_without_a_draft(tree: str, tmp_path: Path) -> None:
    ao, ad, va = _export(tree)
    facility_dir = tmp_path / "data" / "facility"
    target = facility_dir / MAPPING_FILE
    target.parent.mkdir(parents=True)
    shutil.copyfile(FIXTURES / tree / MAPPING_FILE, target)
    assert load_or_draft(facility_dir, ao, ad, va) == _mapping(tree)
    assert target.read_bytes() == (FIXTURES / tree / MAPPING_FILE).read_bytes()


def test_spear3_carries_the_facility_identity() -> None:
    identity = _mapping("spear3").identity
    assert identity is not None and identity.code == "SPEAR3"


@pytest.mark.parametrize("tree", ["nsls2", "synthetic"])
def test_the_other_trees_carry_no_facility_block(tree: str) -> None:
    assert _mapping(tree).identity is None


@pytest.mark.parametrize(
    ("tree", "models"),
    [
        ("spear3", ["StorageRing"]),
        ("nsls2", ["StorageRing", "LTB"]),
        ("synthetic", ["SR"]),
    ],
)
def test_models_are_keyed_and_named_by_the_export_s_system_token(
    tree: str, models: list[str]
) -> None:
    mapping = _mapping(tree)
    assert list(mapping.models) == models
    assert [model.name for model in mapping.models.values()] == models
    assert list(mapping.section_order) == models


@pytest.mark.parametrize(
    ("tree", "model"),
    [("spear3", "StorageRing"), ("nsls2", "StorageRing"), ("synthetic", "SR")],
)
def test_the_wired_model_names_its_orbit_corrector_and_cavity_wiring(tree: str, model: str) -> None:
    wiring = _mapping(tree).models[model].wiring
    assert wiring["BPMx"].engine == EngineBlock(axis="x")
    assert wiring["BPMy"].engine == EngineBlock(axis="y")
    corrector = "HC" if tree == "synthetic" else "HCM"
    assert wiring[corrector].engine == EngineBlock(attribute="KickAngle", index=0)
    assert wiring["RF"].engine == EngineBlock(attribute="Frequency")
    assert wiring["RF"].element_field == "Setpoint"


def test_the_synthetic_out_of_range_family_is_wired_like_any_other() -> None:
    hc = _mapping("synthetic").models["SR"].wiring["HC"]
    assert (hc.element_field, hc.calibration) == ("Setpoint", "table")
