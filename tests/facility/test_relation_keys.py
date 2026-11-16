"""Relation claim rules: shape problems, keys, coarse reasons, against the real type rows."""

from __future__ import annotations

from typing import Any

import pytest

from osprey.facility.relations import (
    COARSE_KEY_ATTRIBUTE,
    COARSE_PLACE,
    COARSE_PORT,
    COARSE_UNIT,
    claim_keys,
    coarse_reasons,
    shape_problems,
    spell_key,
)
from osprey.facility.validate import relation_types


@pytest.fixture(scope="module")
def types() -> dict[str, dict[str, Any]]:
    return relation_types()


def _claim(rid: str, rtype: str, subject: dict, object_: dict, **extra: Any) -> dict[str, Any]:
    return {"id": rid, "type": rtype, "subject": subject, "object": object_, **extra}


class TestKeys:
    def test_directed_type_keys_on_subject_and_port(self, types) -> None:
        claim = _claim(
            "a", "powered_by", {"device": "Q1", "port": "IN"}, {"device": "PS1", "port": "OUT1"}
        )
        keys = claim_keys(claim, types)
        assert keys == [("powered_by", ("subject", ("device", "Q1")), ("port", "IN"))]
        assert spell_key(keys[0]) == "device Q1, port IN"

    def test_directed_type_keyed_on_subject_only(self, types) -> None:
        claim = _claim("a", "part_of", {"device": "M1"}, {"device": "CH1"})
        assert claim_keys(claim, types) == [("part_of", ("subject", ("device", "M1")))]

    def test_symmetric_type_keys_once_from_each_end(self, types) -> None:
        claim = _claim(
            "a", "networked_to", {"device": "H1", "port": "eth0"}, {"device": "SW1", "port": "p7"}
        )
        assert claim_keys(claim, types) == [
            ("networked_to", ("subject", ("device", "H1")), ("port", "eth0")),
            ("networked_to", ("subject", ("device", "SW1")), ("port", "p7")),
        ]

    def test_symmetric_pair_in_opposite_directions_shares_keys(self, types) -> None:
        one = _claim("a", "networked_to", {"device": "H1"}, {"device": "SW1", "port": "p7"})
        two = _claim("b", "networked_to", {"device": "SW1", "port": "p7"}, {"device": "H1"})
        assert set(claim_keys(one, types)) == set(claim_keys(two, types))

    def test_cabled_to_gives_one_key_from_both_ends(self, types) -> None:
        claim = _claim(
            "a", "cabled_to", {"device": "A"}, {"place": "R1"}, attributes={"cable": "C-12"}
        )
        keys = claim_keys(claim, types)
        assert keys == [("cabled_to", ("attributes.cable", "C-12"))]
        assert spell_key(keys[0]) == "cable C-12"

    def test_portless_end_keys_with_no_port(self, types) -> None:
        one = _claim("a", "powered_by", {"device": "Q1"}, {"device": "PS1"})
        two = _claim("b", "powered_by", {"device": "Q1"}, {"device": "PS2"})
        (key,) = claim_keys(one, types)
        assert key == ("powered_by", ("subject", ("device", "Q1")), ("port", None))
        assert claim_keys(two, types) == [key]
        assert spell_key(key) == "device Q1, no port"

    def test_portless_key_differs_from_a_stated_port(self, types) -> None:
        bare = _claim("a", "powered_by", {"device": "Q1"}, {"device": "PS1"})
        ported = _claim("b", "powered_by", {"device": "Q1", "port": "IN"}, {"device": "PS1"})
        assert claim_keys(bare, types) != claim_keys(ported, types)

    def test_attribute_key_field_after_subject(self, types) -> None:
        claim = _claim(
            "a", "supplied_by", {"device": "M1"}, {"device": "LCW1"}, attributes={"medium": "water"}
        )
        (key,) = claim_keys(claim, types)
        assert key == ("supplied_by", ("subject", ("device", "M1")), ("attributes.medium", "water"))
        assert spell_key(key) == "device M1, medium water"

    @pytest.mark.parametrize("rtype", ["controlled_by", "interlocked_by", "bounded_by", "same_as"])
    def test_type_without_single_per_has_no_key(self, types, rtype) -> None:
        subject = {"group": "G"} if rtype == "bounded_by" else {"device": "A"}
        claim = _claim("a", rtype, subject, {"device": "B"})
        assert claim_keys(claim, types) == []

    def test_missing_key_attribute_leaves_the_claim_keyless(self, types) -> None:
        claim = _claim("a", "cabled_to", {"device": "A"}, {"device": "B"})
        assert claim_keys(claim, types) == []

    def test_unknown_type_has_no_key(self, types) -> None:
        assert claim_keys(_claim("a", "nope", {"device": "A"}, {"device": "B"}), types) == []

    def test_keys_read_from_the_rows_argument(self, types) -> None:
        table = {**types, "part_of": {**types["part_of"], "single_per": []}}
        claim = _claim("a", "part_of", {"device": "M1"}, {"device": "CH1"})
        assert claim_keys(claim, table) == []


class TestCoarse:
    def test_sound_claim_is_not_coarse(self, types) -> None:
        claim = _claim("a", "powered_by", {"device": "Q1"}, {"device": "PS1", "port": "OUT1"})
        assert coarse_reasons(claim, [claim], types) == []

    def test_place_where_device_allowed(self, types) -> None:
        claim = _claim("a", "powered_by", {"device": "Q1"}, {"place": "SR/S01"})
        assert coarse_reasons(claim, [], types) == [COARSE_PLACE]

    def test_place_on_subject_end_where_device_allowed(self, types) -> None:
        claim = _claim("a", "triggered_by", {"place": "SR/S01"}, {"device": "EVR1"})
        assert coarse_reasons(claim, [], types) == [COARSE_PLACE]

    def test_place_where_only_place_allowed_is_not_coarse(self, types) -> None:
        claim = _claim("a", "located_in", {"device": "Q1"}, {"place": "R1", "unit": 5})
        assert coarse_reasons(claim, [], types) == []

    def test_missing_port_beside_a_claim_stating_it(self, types) -> None:
        bare = _claim("a", "powered_by", {"device": "Q1"}, {"device": "PS1"})
        ported = _claim("b", "powered_by", {"device": "Q1"}, {"device": "PS1", "port": "OUT1"})
        assert coarse_reasons(bare, [bare, ported], types) == [COARSE_PORT]
        assert coarse_reasons(ported, [bare, ported], types) == []

    def test_missing_port_against_a_different_object_is_not_coarse(self, types) -> None:
        bare = _claim("a", "powered_by", {"device": "Q1"}, {"device": "PS1"})
        other = _claim("b", "powered_by", {"device": "Q1"}, {"device": "PS2", "port": "OUT1"})
        assert coarse_reasons(bare, [bare, other], types) == []

    def test_two_portless_claims_are_not_coarse(self, types) -> None:
        one = _claim("a", "powered_by", {"device": "Q1"}, {"device": "PS1"})
        two = _claim("b", "powered_by", {"device": "Q1"}, {"device": "PS2"})
        assert coarse_reasons(one, [one, two], types) == []
        assert coarse_reasons(two, [one, two], types) == []

    def test_missing_port_on_a_symmetric_pair_stated_reversed(self, types) -> None:
        bare = _claim("a", "networked_to", {"device": "H1", "port": "eth0"}, {"device": "SW1"})
        ported = _claim(
            "b", "networked_to", {"device": "SW1", "port": "p7"}, {"device": "H1", "port": "eth0"}
        )
        assert coarse_reasons(bare, [bare, ported], types) == [COARSE_PORT]

    def test_missing_key_attribute(self, types) -> None:
        claim = _claim("a", "cabled_to", {"device": "A"}, {"device": "B"})
        assert coarse_reasons(claim, [], types) == [COARSE_KEY_ATTRIBUTE]

    def test_located_in_without_unit_beside_one_with(self, types) -> None:
        bare = _claim("a", "located_in", {"device": "Q1"}, {"place": "R1"})
        u5 = _claim("b", "located_in", {"device": "Q1"}, {"place": "R1", "unit": 5})
        u6 = _claim("c", "located_in", {"device": "Q1"}, {"place": "R1", "unit": 6})
        on_key = [bare, u5, u6]
        assert coarse_reasons(bare, on_key, types) == [COARSE_UNIT]
        assert coarse_reasons(u5, on_key, types) == []
        assert coarse_reasons(u6, on_key, types) == []

    def test_located_in_without_unit_alone_is_not_coarse(self, types) -> None:
        bare = _claim("a", "located_in", {"device": "Q1"}, {"place": "R1"})
        other = _claim("b", "located_in", {"device": "Q1"}, {"place": "R2", "unit": 5})
        assert coarse_reasons(bare, [bare, other], types) == []

    def test_reasons_come_in_fixed_order(self, types) -> None:
        bare = _claim(
            "a", "cabled_to", {"place": "R1"}, {"device": "B"}, attributes={"cable": "C-1"}
        )
        ported = _claim(
            "b",
            "cabled_to",
            {"place": "R1"},
            {"device": "B", "port": "J1"},
            attributes={"cable": "C-1"},
        )
        assert coarse_reasons(bare, [bare, ported], types) == [COARSE_PLACE, COARSE_PORT]

    def test_end_kinds_read_from_the_rows_argument(self, types) -> None:
        table = {**types, "powered_by": {**types["powered_by"], "object": ["place"]}}
        claim = _claim("a", "powered_by", {"device": "Q1"}, {"place": "SR/S01"})
        assert coarse_reasons(claim, [], table) == []


def _details(claim: dict[str, Any], types: dict[str, dict[str, Any]]) -> list[str]:
    return [detail for detail, _remedy in shape_problems(claim, types)]


class TestShapeProblems:
    def test_sound_claim_has_none(self, types) -> None:
        claim = _claim("a", "powered_by", {"device": "Q1"}, {"device": "PS1", "port": "OUT1"})
        assert shape_problems(claim, types) == []

    def test_end_naming_no_kind(self, types) -> None:
        claim = _claim("a", "powered_by", {"port": "IN"}, {"device": "PS1"})
        assert shape_problems(claim, types) == [
            (
                "`subject` names no device, place or group",
                "name one of `device`, `place`, `group` in `subject`",
            )
        ]

    def test_end_naming_two_kinds(self, types) -> None:
        claim = _claim("a", "powered_by", {"device": "Q1"}, {"device": "PS1", "place": "R1"})
        assert _details(claim, types) == ["`object` names more than one of device, place, group"]

    def test_unit_on_a_device_end(self, types) -> None:
        claim = _claim("a", "powered_by", {"device": "Q1", "unit": 3}, {"device": "PS1"})
        assert _details(claim, types) == ["`subject` gives `unit` on an end that names no place"]

    def test_unit_on_a_place_end_is_sound(self, types) -> None:
        claim = _claim("a", "located_in", {"device": "Q1"}, {"place": "R1", "unit": 3})
        assert shape_problems(claim, types) == []

    def test_end_kind_outside_the_type(self, types) -> None:
        claim = _claim("a", "powered_by", {"group": "SR/QUAD"}, {"device": "PS1"})
        assert shape_problems(claim, types) == [
            (
                "`subject` names a group; powered_by allows device or place there",
                "name a device or place as the subject of powered_by",
            )
        ]

    def test_end_kind_read_from_the_rows_argument(self, types) -> None:
        table = {**types, "powered_by": {**types["powered_by"], "subject": ["group"]}}
        claim = _claim("a", "powered_by", {"group": "SR/QUAD"}, {"device": "PS1"})
        assert shape_problems(claim, table) == []

    def test_inferred_without_rule(self, types) -> None:
        claim = _claim("a", "part_of", {"device": "M1"}, {"device": "CH1"}, basis="inferred")
        assert _details(claim, types) == ["`basis` is inferred and `rule` is missing"]
        claim["rule"] = "slot-in-chassis"
        assert shape_problems(claim, types) == []

    def test_supplied_by_without_medium(self, types) -> None:
        claim = _claim("a", "supplied_by", {"device": "M1"}, {"device": "LCW1"})
        assert shape_problems(claim, types) == [
            ("supplied_by gives no `attributes.medium`", "state the medium in `attributes.medium`")
        ]

    def test_cabled_to_without_cable_is_not_a_shape_problem(self, types) -> None:
        claim = _claim("a", "cabled_to", {"device": "A"}, {"device": "B"})
        assert shape_problems(claim, types) == []

    def test_subject_equal_to_object(self, types) -> None:
        claim = _claim(
            "a", "networked_to", {"device": "SW1", "port": "p1"}, {"device": "SW1", "port": "p1"}
        )
        assert _details(claim, types) == ["`subject` and `object` name the same end"]

    def test_same_device_on_two_ports_is_sound(self, types) -> None:
        claim = _claim(
            "a", "networked_to", {"device": "SW1", "port": "p1"}, {"device": "SW1", "port": "p2"}
        )
        assert shape_problems(claim, types) == []

    def test_problems_come_in_fixed_order(self, types) -> None:
        claim = _claim("a", "supplied_by", {"group": "G", "unit": 1}, {}, basis="inferred")
        assert _details(claim, types) == [
            "`subject` gives `unit` on an end that names no place",
            "`subject` names a group; supplied_by allows device there",
            "`object` names no device, place or group",
            "`basis` is inferred and `rule` is missing",
            "supplied_by gives no `attributes.medium`",
        ]

    def test_remedies_name_no_records_file(self, types) -> None:
        claim = _claim(
            "a", "supplied_by", {"group": "G", "device": "X", "unit": 1}, {}, basis="inferred"
        )
        remedies = [remedy for _detail, remedy in shape_problems(claim, types)]
        assert remedies and all("records/" not in remedy for remedy in remedies)
