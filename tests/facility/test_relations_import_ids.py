"""Claim ids of imported relation claims (p5-r5)."""

from __future__ import annotations

import copy
import re

import pytest

from osprey.facility.layers.relations.ids import claim_id
from osprey.facility.validate import relation_types

TYPES = relation_types()


def _claim(**changes):
    claim = {
        "type": "powered_by",
        "subject": {"device": "SR01C:QF1", "port": "A"},
        "object": {"device": "SR01C:PS1", "port": "3"},
        "attributes": {"note": "first"},
        "source": {
            "document": "SR power schedule",
            "version": "rev B",
            "state": "released",
            "locator": "p. 4",
            "imported": "2026-11-01",
        },
    }
    claim.update(changes)
    return claim


def _id(claim):
    return claim_id("cables", claim, TYPES)


class TestShape:
    def test_the_id_names_layer_type_and_a_hash(self):
        assert re.fullmatch(r"cables/powered_by/[0-9a-f]{12}", _id(_claim()))

    def test_the_id_is_stable(self):
        assert _id(_claim()) == _id(copy.deepcopy(_claim()))

    def test_the_layer_is_part_of_the_id(self):
        assert claim_id("other", _claim(), TYPES) != _id(_claim())


class TestEnds:
    def test_an_object_only_difference_gives_two_ids(self):
        other = _claim(object={"device": "SR01C:PS2", "port": "3"})
        assert _id(other) != _id(_claim())

    def test_a_port_difference_gives_two_ids(self):
        other = _claim(object={"device": "SR01C:PS1", "port": "4"})
        assert _id(other) != _id(_claim())

    def test_the_unit_counts_as_part_of_the_object_end(self):
        base = _claim(type="located_in", subject={"device": "D1"}, object={"place": "SR/S01"})
        unit = _claim(
            type="located_in", subject={"device": "D1"}, object={"place": "SR/S01", "unit": 7}
        )
        assert _id(base) != _id(unit)

    def test_an_end_kind_difference_gives_two_ids(self):
        device = _claim(object={"device": "SR01", "port": "3"})
        place = _claim(object={"place": "SR01", "port": "3"})
        assert _id(device) != _id(place)

    def test_a_port_read_as_a_number_is_the_same_port(self):
        number = _claim(object={"device": "SR01C:PS1", "port": 3})
        assert _id(number) == _id(_claim())

    def test_a_reversed_directed_row_is_another_claim(self):
        claim = _claim()
        reversed_ = _claim(subject=claim["object"], object=claim["subject"])
        assert _id(reversed_) != _id(claim)

    @pytest.mark.parametrize("type_", ["cabled_to", "networked_to", "same_as"])
    def test_a_reversed_symmetric_row_is_one_claim(self, type_):
        claim = _claim(type=type_, attributes={"cable": "C-17"})
        reversed_ = _claim(
            type=type_,
            subject=claim["object"],
            object=claim["subject"],
            attributes={"cable": "C-17"},
        )
        assert _id(reversed_) == _id(claim)


class TestCitation:
    @pytest.mark.parametrize("field", ["locator", "version", "imported", "state"])
    def test_a_changed_citation_detail_keeps_the_id(self, field):
        other = _claim()
        other["source"][field] = "changed"
        assert _id(other) == _id(_claim())

    def test_a_missing_citation_detail_keeps_the_id(self):
        other = _claim(source={"document": "SR power schedule"})
        assert _id(other) == _id(_claim())

    def test_a_changed_document_changes_the_id(self):
        other = _claim()
        other["source"]["document"] = "SR cable plant"
        assert _id(other) != _id(_claim())


class TestAttributes:
    def test_a_non_key_attribute_change_keeps_the_id(self):
        assert _id(_claim(attributes={"note": "second"})) == _id(_claim())

    def test_dropping_non_key_attributes_keeps_the_id(self):
        assert _id(_claim(attributes={})) == _id(_claim())

    def test_a_key_attribute_change_changes_the_id(self):
        base = _claim(type="supplied_by", attributes={"medium": "water"})
        other = _claim(type="supplied_by", attributes={"medium": "air"})
        assert _id(base) != _id(other)

    def test_a_key_attribute_beside_other_attributes_is_read_alone(self):
        base = _claim(type="cabled_to", attributes={"cable": "C-17"})
        other = _claim(type="cabled_to", attributes={"cable": "C-17", "length": "4 m"})
        assert _id(base) == _id(other)
        assert _id(_claim(type="cabled_to", attributes={"cable": "C-18"})) != _id(base)


class TestDuplicates:
    def test_an_exact_duplicate_row_is_one_claim(self):
        assert _id(_claim()) == _id(_claim())

    def test_an_unknown_type_still_gets_an_id(self):
        assert _id(_claim(type="no_such_type")).startswith("cables/no_such_type/")
