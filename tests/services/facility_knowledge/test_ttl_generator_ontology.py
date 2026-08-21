"""Tests for the demo-machine ontology table.

The table is the join between the channel database's FAMILY tokens and the
NARAD class vocabulary the graph store is queried through, so these tests pin
both halves: every family the real tier-3 database produces resolves to a class,
and the class hierarchy that carries the rollup queries stays intact.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osprey.services.channel_finder.databases.hierarchical import HierarchicalChannelDatabase
from osprey.services.facility_knowledge.ttl_generator.model import NARAD_SEM_NS, build_model
from osprey.services.facility_knowledge.ttl_generator.ontology_map import (
    ROOT_CLASS,
    ClassDef,
    OntologyMap,
    OntologyMapError,
    UnknownFamilyError,
    load_demo_ontology,
    load_ontology,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TIER3_DB = (
    _REPO_ROOT
    / "src/osprey/templates/apps/control_assistant/data/channel_databases/tiers/tier3"
    / "hierarchical.json"
)

#: Every FAMILY token the demo machine's tier-3 database expands to.  Asserted
#: literally as well as derived, so a table that quietly loses a family fails
#: even if the database loses it at the same time.
EXPECTED_FAMILIES = frozenset(
    {
        "BPM",
        "CAVITY",
        "DCCT",
        "DIPOLE",
        "GAMMA",
        "GAUGE",
        "HCM",
        "ION-PUMP",
        "KLYSTRON",
        "NEUTRON",
        "QD",
        "QF",
        "QFA",
        "SD",
        "SF",
        "SHD",
        "SHF",
        "VALVE",
        "VCM",
    }
)

#: The families that must land under narad_sem:Magnet.
MAGNET_FAMILIES = frozenset({"DIPOLE", "HCM", "QD", "QF", "QFA", "SD", "SF", "SHD", "SHF", "VCM"})

#: Devices of those families in the tier-3 database — the known answer to the
#: "every magnet" rollup query.
EXPECTED_MAGNET_DEVICES = 382


@pytest.fixture(scope="module")
def demo_model():
    """The graph model derived from the real tier-3 channel database."""
    database = HierarchicalChannelDatabase(str(_TIER3_DB))
    return build_model(database.channel_map)


@pytest.fixture(scope="module")
def ontology() -> OntologyMap:
    """The shipped demo ontology table."""
    return load_demo_ontology()


def _make_map(classes: dict[str, ClassDef], families: dict[str, str] | None = None) -> OntologyMap:
    """Build a synthetic table straight from dataclasses, bypassing validation."""
    return OntologyMap(
        family_to_class=families if families is not None else {},
        classes=classes,
        source_path=Path("/tmp/synthetic_ontology.json"),
    )


# ---------------------------------------------------------------------------
# Family coverage
# ---------------------------------------------------------------------------


def test_tier3_family_set_is_the_expected_nineteen(demo_model) -> None:
    """The database still produces exactly the 19 families the table covers."""
    families = {device.family for device in demo_model.devices}
    assert families == EXPECTED_FAMILIES
    assert len(EXPECTED_FAMILIES) == 19


def test_every_tier3_family_resolves_to_a_class(demo_model, ontology: OntologyMap) -> None:
    """Every family present in the real database has a class in the table."""
    for family in sorted({device.family for device in demo_model.devices}):
        klass = ontology.class_for(family)
        assert klass.name
        assert klass.iri == f"{NARAD_SEM_NS}{klass.name}"


def test_table_maps_no_family_the_database_lacks(ontology: OntologyMap) -> None:
    """The table does not carry entries for families that no longer exist."""
    assert set(ontology.family_to_class) == EXPECTED_FAMILIES


def test_mapping_matches_the_narad_vocabulary(ontology: OntologyMap) -> None:
    """The FAMILY-to-class assignments are the ones the ALS corpus would use."""
    expected = {
        "BPM": "BeamPositionMonitor",
        "CAVITY": "AcceleratingCavity",
        "DCCT": "BeamCurrentMonitor",
        "DIPOLE": "Dipole",
        "GAMMA": "BeamLossMonitor",
        "GAUGE": "Gauge",
        "HCM": "HCorrector",
        "ION-PUMP": "Pump",
        "KLYSTRON": "Modulator",
        "NEUTRON": "BeamLossMonitor",
        "QD": "Quadrupole",
        "QF": "Quadrupole",
        "QFA": "Quadrupole",
        "SD": "Sextupole",
        "SF": "Sextupole",
        "SHD": "Sextupole",
        "SHF": "Sextupole",
        "VALVE": "Valve",
        "VCM": "VCorrector",
    }
    assert dict(ontology.family_to_class) == expected


def test_unknown_family_names_the_family_and_the_table(ontology: OntologyMap) -> None:
    """A miss points the operator at the family and the file to edit."""
    with pytest.raises(UnknownFamilyError) as excinfo:
        ontology.class_for("SEPTUM")
    message = str(excinfo.value)
    assert "SEPTUM" in message
    assert str(ontology.source_path) in message
    assert isinstance(excinfo.value, KeyError)


# ---------------------------------------------------------------------------
# Hierarchy
# ---------------------------------------------------------------------------


def test_hierarchy_mirrors_the_als_corpus(ontology: OntologyMap) -> None:
    """The subClassOf edges are copied from als_gtb.ttl, edge for edge."""
    expected_parents = {
        "AcceleratorDevice": None,
        "Magnet": "AcceleratorDevice",
        "Instrumentation": "AcceleratorDevice",
        "Vacuum": "AcceleratorDevice",
        "RadioFrequency": "AcceleratorDevice",
        "Dipole": "Magnet",
        "Quadrupole": "Magnet",
        "Sextupole": "Magnet",
        "Corrector": "Magnet",
        "HCorrector": "Corrector",
        "VCorrector": "Corrector",
        "BeamPositionMonitor": "Instrumentation",
        "BeamCurrentMonitor": "Instrumentation",
        "BeamLossMonitor": "Instrumentation",
        "Gauge": "Vacuum",
        "Pump": "Vacuum",
        "Valve": "Vacuum",
        "AcceleratingCavity": "RadioFrequency",
        "Modulator": "RadioFrequency",
    }
    actual = {name: klass.parent for name, klass in ontology.classes.items()}
    assert actual == expected_parents


def test_ancestors_walk_to_the_root(ontology: OntologyMap) -> None:
    """ancestors() returns the chain nearest-first, ending at the root."""
    assert ontology.ancestors("HCorrector") == ("Corrector", "Magnet", ROOT_CLASS)
    assert ontology.ancestors("Gauge") == ("Vacuum", ROOT_CLASS)
    assert ontology.ancestors(ROOT_CLASS) == ()


def test_used_classes_are_subclassof_closed(demo_model, ontology: OntologyMap) -> None:
    """The emitted class set contains every parent of every class in it."""
    families = sorted({device.family for device in demo_model.devices})
    used = ontology.used_classes(families)
    names = {klass.name for klass in used}

    for klass in used:
        if klass.parent is not None:
            assert klass.parent in names, f"{klass.name} declared without its parent"
    assert ROOT_CLASS in names
    assert [klass.name for klass in used] == sorted(names)


def test_used_classes_covers_the_mapped_classes_and_nothing_stray(
    demo_model, ontology: OntologyMap
) -> None:
    """Every mapped class is emitted, plus exactly the ancestors they need."""
    families = sorted({device.family for device in demo_model.devices})
    names = {klass.name for klass in ontology.used_classes(families)}

    mapped = {ontology.class_for(family).name for family in families}
    assert mapped <= names
    # Corrector is not mapped by any family; it is pulled in by HCM and VCM.
    assert "Corrector" in names
    assert names == set(ontology.classes)


def test_used_classes_ignores_duplicate_families(ontology: OntologyMap) -> None:
    """Repeating a family does not repeat its classes."""
    once = ontology.used_classes(["QF"])
    twice = ontology.used_classes(["QF", "QF", "QD"])
    assert [klass.name for klass in once] == ["AcceleratorDevice", "Magnet", "Quadrupole"]
    assert once == twice


def test_used_classes_rejects_an_unknown_family(ontology: OntologyMap) -> None:
    """The closure refuses to silently drop a family it cannot type."""
    with pytest.raises(UnknownFamilyError):
        ontology.used_classes(["QF", "SEPTUM"])


def test_magnet_branch_holds_exactly_the_magnet_families(demo_model, ontology: OntologyMap) -> None:
    """The 'every magnet' rollup has a known answer on the demo corpus."""
    magnet_families = {
        family
        for family in ontology.family_to_class
        if "Magnet" in ontology.ancestors(ontology.class_for(family).name)
    }
    assert magnet_families == MAGNET_FAMILIES

    magnet_devices = [device for device in demo_model.devices if device.family in magnet_families]
    assert len(magnet_devices) == EXPECTED_MAGNET_DEVICES
    assert len(magnet_devices) < len(demo_model.devices)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_shipped_table_validates(ontology: OntologyMap) -> None:
    """The table that ships is well formed."""
    ontology.validate()


def test_validate_rejects_a_dangling_parent() -> None:
    """A parent no class declares is an error, and the message names it."""
    broken = _make_map(
        {
            ROOT_CLASS: ClassDef(ROOT_CLASS, None, ()),
            "Magnet": ClassDef("Magnet", "Subsystem", ()),
        }
    )
    with pytest.raises(OntologyMapError, match="Subsystem"):
        broken.validate()


def test_validate_rejects_a_cycle() -> None:
    """Two classes that parent each other never reach the root."""
    cyclic = _make_map(
        {
            ROOT_CLASS: ClassDef(ROOT_CLASS, None, ()),
            "Magnet": ClassDef("Magnet", "Dipole", ()),
            "Dipole": ClassDef("Dipole", "Magnet", ()),
        }
    )
    with pytest.raises(OntologyMapError, match="cycle"):
        cyclic.validate()


def test_validate_rejects_a_second_root() -> None:
    """Only AcceleratorDevice may be parentless."""
    two_roots = _make_map(
        {
            ROOT_CLASS: ClassDef(ROOT_CLASS, None, ()),
            "Orphan": ClassDef("Orphan", None, ()),
        }
    )
    with pytest.raises(OntologyMapError, match="parentless"):
        two_roots.validate()


def test_validate_rejects_a_family_mapped_to_an_undeclared_class() -> None:
    """A family may not point at a class the table never declares."""
    broken = _make_map(
        {ROOT_CLASS: ClassDef(ROOT_CLASS, None, ())},
        {"DIPOLE": "Dipole"},
    )
    with pytest.raises(OntologyMapError, match="Dipole"):
        broken.validate()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_load_demo_ontology_names_the_shipped_file(ontology: OntologyMap) -> None:
    """The map carries the path of the file an operator would edit."""
    assert ontology.source_path.name == "demo_ontology.json"
    assert ontology.source_path.exists()


def test_load_demo_ontology_is_cached() -> None:
    """Repeated loads hand back the same immutable table."""
    assert load_demo_ontology() is load_demo_ontology()


def test_load_ontology_reads_a_facility_table(tmp_path: Path) -> None:
    """A facility can point the loader at its own table."""
    table = {
        "_comment": "Facility table with one family.",
        "root": ROOT_CLASS,
        "family_to_class": {"BEND": "Dipole"},
        "classes": {
            ROOT_CLASS: {"parent": None},
            "Magnet": {"parent": ROOT_CLASS, "altLabels": ["magnet"]},
            "Dipole": {"parent": "Magnet", "altLabels": ["bend"]},
        },
    }
    path = tmp_path / "facility_ontology.json"
    path.write_text(json.dumps(table), encoding="utf-8")

    loaded = load_ontology(path)
    assert loaded.source_path == path
    assert loaded.class_for("BEND").alt_labels == ("bend",)
    assert loaded.ancestors("Dipole") == ("Magnet", ROOT_CLASS)


def test_load_ontology_rejects_a_broken_table(tmp_path: Path) -> None:
    """A malformed table fails at load, not at the first query."""
    path = tmp_path / "broken_ontology.json"
    path.write_text(
        json.dumps(
            {
                "root": ROOT_CLASS,
                "family_to_class": {"BEND": "Dipole"},
                "classes": {ROOT_CLASS: {"parent": None}, "Dipole": {"parent": "Subsystem"}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(OntologyMapError, match="Subsystem"):
        load_ontology(path)


def test_load_ontology_rejects_an_unknown_top_level_key(tmp_path: Path) -> None:
    """A typo in a top-level key is caught rather than silently ignored."""
    path = tmp_path / "typo_ontology.json"
    path.write_text(
        json.dumps(
            {
                "root": ROOT_CLASS,
                "familyToClass": {"BEND": "Dipole"},
                "classes": {ROOT_CLASS: {"parent": None}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(OntologyMapError, match="familyToClass"):
        load_ontology(path)
