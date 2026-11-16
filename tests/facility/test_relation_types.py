"""The relation type table and the infrastructure classes of the vocabulary.

``vocabulary.yaml`` states the thirteen relation types as one closed enum whose
values carry their ends, symmetry and single-valued key as annotations;
``vocabulary.json`` carries them as the ``relations`` table, and
:func:`osprey.facility.validate.relation_types` reads it without LinkML.
"""

from __future__ import annotations

from typing import Any

from osprey.facility.validate import relation_types, vocabulary

D = ["device"]
DP = ["device", "place"]

#: The relation table of the design, row for row: subject kinds, object kinds,
#: symmetric, and the key fields of "at most one object per".
EXPECTED: dict[str, tuple[list[str], list[str], bool, list[str]]] = {
    "part_of": (D, D, False, ["subject"]),
    "located_in": (D, ["place"], False, ["subject"]),
    "powered_by": (DP, DP, False, ["subject", "port"]),
    "cabled_to": (DP, DP, True, ["attributes.cable"]),
    "read_out_by": (D, DP, False, ["subject", "port"]),
    "controlled_by": (D, DP, False, []),
    "triggered_by": (DP, DP, False, ["subject", "port"]),
    "interlocked_by": (D, DP, False, []),
    "bounded_by": (["group", "place"], D, False, []),
    "networked_to": (DP, DP, True, ["subject", "port"]),
    "supplied_by": (D, D, False, ["subject", "attributes.medium"]),
    "same_as": (D, D, True, []),
    "runs_on": (D, D, False, ["subject"]),
}

INFRASTRUCTURE_CHILDREN = {
    "PowerSupply",
    "ElectronicsChassis",
    "Host",
    "NetworkSwitch",
    "ControlApplication",
    "Circuit",
    "UtilityLoop",
}


def _classes() -> dict[str, dict[str, Any]]:
    return {row["name"]: row for row in vocabulary()["classes"]}


def test_relation_types_equal_the_design_table_row_for_row() -> None:
    types = relation_types()
    assert set(types) == set(EXPECTED)
    assert len(types) == 13
    for name, (subject, object_, symmetric, single_per) in EXPECTED.items():
        row = types[name]
        assert set(row) == {"name", "description", "subject", "object", "symmetric", "single_per"}
        assert row["name"] == name
        assert row["subject"] == subject, name
        assert row["object"] == object_, name
        assert row["symmetric"] is symmetric, name
        assert row["single_per"] == single_per, name
        assert isinstance(row["description"], str) and row["description"].strip(), name


def test_relations_table_is_sorted_by_name() -> None:
    names = [row["name"] for row in vocabulary()["relations"]]
    assert names == sorted(names)


def test_interlock_zone_is_not_a_vocabulary_class() -> None:
    assert "InterlockZone" not in _classes()


def test_infrastructure_is_abstract_with_seven_concrete_children() -> None:
    classes = _classes()
    assert classes["Infrastructure"]["parent"] == "AcceleratorDevice"
    assert classes["Infrastructure"]["abstract"] is True
    children = {name for name, row in classes.items() if row["parent"] == "Infrastructure"}
    assert children == INFRASTRUCTURE_CHILDREN
    assert all(classes[name]["abstract"] is False for name in children)


def test_unidentified_is_concrete_under_the_root() -> None:
    row = _classes()["Unidentified"]
    assert row["parent"] == "AcceleratorDevice"
    assert row["abstract"] is False


def test_vocabulary_has_one_class_root() -> None:
    roots = [name for name, row in _classes().items() if row["parent"] is None]
    assert roots == ["AcceleratorDevice"]
