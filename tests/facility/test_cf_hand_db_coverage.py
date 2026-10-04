"""Every fact of the demo's four hand channel databases survives in its records.

The hand databases are the tier-1 in_context database and the tier-3
in_context, hierarchical and middle-layer databases. Each description, level
description, common name and ``DeviceList`` entry they hold must appear in the
demo's committed records, as a channel, group, place or device description,
name, ``signals`` sentence or device attribute.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.facility.test_generator_records import CA_DATA, committed, records_by_id

TIERS = CA_DATA / "channel_databases/tiers"
TIER1_IN_CONTEXT = TIERS / "tier1/in_context.json"
TIER3_IN_CONTEXT = TIERS / "tier3/in_context.json"
TIER3_HIERARCHICAL = TIERS / "tier3/hierarchical.json"
TIER3_MIDDLE_LAYER = TIERS / "tier3/middle_layer.json"


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _children(node: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    for key, value in node.items():
        if not key.startswith("_"):
            yield key, value


def _hierarchical_families() -> Iterator[tuple[str, str, str, dict[str, Any]]]:
    """(machine, system, family, family node) for every hierarchical family."""
    for machine, machine_node in _children(_json(TIER3_HIERARCHICAL)["tree"]):
        for system, system_node in _children(machine_node):
            for family, family_node in _children(system_node):
                yield machine, system, family, family_node


def _field_sentences() -> Iterator[tuple[str, str, str]]:
    """(group id, signals key, sentence) for every hierarchical field and subfield."""
    for machine, _system, family, family_node in _hierarchical_families():
        gid = f"{machine}/{family}"
        for field, field_node in _children(family_node["DEVICE"]):
            yield gid, field, field_node["_description"]
            for subfield, subfield_node in _children(field_node):
                yield gid, f"{field}/{subfield}", subfield_node["_description"]


def _descriptions_and_names(kind: str) -> set[str]:
    texts = set()
    for record in committed(f"records/{kind}s.yaml"):
        if "description" in record:
            texts.add(record["description"])
        texts.update(record.get("names") or [])
    return texts


def test_every_field_and_subfield_sentence_sits_in_its_family_group_signals() -> None:
    groups = records_by_id("group")
    sentences = list(_field_sentences())
    assert len({sentence for _gid, _key, sentence in sentences}) == 161
    for gid, key, sentence in sentences:
        assert groups[gid]["signals"][key] == sentence, (gid, key)
    for gid, group in groups.items():
        expected = {key for g, key, _s in sentences if g == gid}
        assert set(group.get("signals") or {}) == expected, gid


def test_family_groups_keep_each_machine_s_own_wording() -> None:
    groups = records_by_id("group")
    assert groups["SR/DIPOLE"]["signals"]["CURRENT"] != groups["BR/DIPOLE"]["signals"]["CURRENT"]


def test_hierarchical_level_descriptions_are_place_or_group_descriptions() -> None:
    tree = _json(TIER3_HIERARCHICAL)["tree"]
    places = records_by_id("place")
    groups = records_by_id("group")
    for machine, machine_node in _children(tree):
        assert places[machine]["description"] == machine_node["_description"]
        for system, system_node in _children(machine_node):
            assert groups[f"{machine}/{system}"]["description"] == system_node["_description"]
    for machine, _system, family, family_node in _hierarchical_families():
        assert groups[f"{machine}/{family}"]["description"] == family_node["_description"]


@pytest.mark.parametrize("path", [TIER1_IN_CONTEXT, TIER3_IN_CONTEXT], ids=["tier1", "tier3"])
def test_in_context_names_and_descriptions_are_channel_names_and_descriptions(
    path: Path,
) -> None:
    channels = records_by_id("channel")
    for row in _json(path)["channels"]:
        channel = channels[row["address"]]
        assert row["channel"] in channel["names"], row["address"]
        assert channel["description"] == row["description"], row["address"]


def test_middle_layer_machine_labels_are_place_descriptions_or_names() -> None:
    texts = _descriptions_and_names("place")
    labels = [node["_description"] for _m, node in _children(_json(TIER3_MIDDLE_LAYER))]
    assert len(labels) == 3
    assert set(labels) <= texts


def test_middle_layer_family_labels_are_their_group_description_or_names() -> None:
    groups = records_by_id("group")
    labels = set()
    for machine, machine_node in _children(_json(TIER3_MIDDLE_LAYER)):
        for family, family_node in _children(machine_node):
            group = groups[f"{machine}/{family}"]
            label = family_node["_description"]
            labels.add(label)
            assert label == group.get("description") or label in group.get("names", []), (
                machine,
                family,
            )
    assert len(labels) == 19


def test_middle_layer_field_labels_occur_in_a_channel_description() -> None:
    descriptions = [c["description"].lower() for c in committed("records/channels.yaml")]
    labels = set()
    for _machine, machine_node in _children(_json(TIER3_MIDDLE_LAYER)):
        for _family, family_node in _children(machine_node):
            for _field, field_node in _children(family_node):
                labels.add(field_node["_description"])
                for _subfield, leaf in _children(field_node):
                    labels.add(leaf["_description"])
    assert len(labels) == 34
    for label in sorted(labels):
        assert any(label.lower() in text for text in descriptions), label


def test_middle_layer_setup_is_on_the_devices_its_channels_name() -> None:
    channels = records_by_id("channel")
    devices = records_by_id("device")
    for machine, machine_node in _children(_json(TIER3_MIDDLE_LAYER)):
        for family, family_node in _children(machine_node):
            setup = family_node["_setup"]
            for _field, field_node in _children(family_node):
                for _subfield, leaf in _children(field_node):
                    for index, address in enumerate(leaf["ChannelNames"]):
                        device = devices[channels[address]["on"]["device"]]
                        where = (machine, family, address)
                        assert setup["CommonNames"][index] in device["names"], where
                        assert device["attributes"]["DeviceList"] == setup["DeviceList"][index]
                        assert device["attributes"]["ElementList"] == setup["ElementList"][index]
