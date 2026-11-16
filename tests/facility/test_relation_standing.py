"""Relation claim standing: disputes per key, ranking by policy, and order independence."""

from __future__ import annotations

import itertools
from typing import Any

import pytest

from osprey.facility.relations import compute_standing
from osprey.facility.validate import relation_types

_DOCUMENTS = [
    {"id": "spec-1", "title": "Spec", "kind": "spec", "system": "s"},
    {"id": "spec-2", "title": "Spec two", "kind": "spec", "system": "s"},
    {"id": "dwg-1", "title": "Drawing", "kind": "drawing", "system": "s"},
    {"id": "note-1", "title": "Note", "kind": "note", "system": "s"},
]

_RANKED = {"relations": {"precedence": {"default": ["spec", "drawing"]}}}


@pytest.fixture(scope="module")
def types() -> dict[str, dict[str, Any]]:
    return relation_types()


def _claim(rid: str, rtype: str, subject: dict, object_: dict, **extra: Any) -> dict[str, Any]:
    return {"id": rid, "type": rtype, "subject": subject, "object": object_, **extra}


def _cited(document: str) -> dict[str, Any]:
    return {"source": {"document": document}}


def _power(rid: str, supply: str, **extra: Any) -> dict[str, Any]:
    return _claim(rid, "powered_by", {"device": "Q1"}, {"device": supply}, **extra)


class TestUndisputed:
    def test_single_claim_is_accepted_without_contests(self, types) -> None:
        claims = [_power("a", "PS1")]
        assert compute_standing(claims, types, None, []) == {"a": ("accepted", [])}

    def test_agreeing_claims_are_accepted(self, types) -> None:
        claims = [_power("a", "PS1"), _power("b", "PS1", **_cited("dwg-1"))]
        assert compute_standing(claims, types, _RANKED, _DOCUMENTS) == {
            "a": ("accepted", []),
            "b": ("accepted", []),
        }

    def test_different_ports_on_one_subject_are_different_keys(self, types) -> None:
        claims = [
            _claim("a", "powered_by", {"device": "Q1", "port": "A"}, {"device": "PS1"}),
            _claim("b", "powered_by", {"device": "Q1", "port": "B"}, {"device": "PS2"}),
        ]
        assert compute_standing(claims, types, None, []) == {
            "a": ("accepted", []),
            "b": ("accepted", []),
        }

    def test_no_claims_gives_no_standing(self, types) -> None:
        assert compute_standing([], types, None, []) == {}


class TestNoPolicy:
    def test_two_objects_on_one_key_contest_each_other(self, types) -> None:
        claims = [_power("a", "PS1", **_cited("spec-1")), _power("b", "PS2", **_cited("dwg-1"))]
        assert compute_standing(claims, types, None, _DOCUMENTS) == {
            "a": ("contested", ["b"]),
            "b": ("contested", ["a"]),
        }

    def test_objects_differing_only_by_port_contest(self, types) -> None:
        claims = [
            _claim("a", "powered_by", {"device": "Q1"}, {"device": "PS1", "port": "OUT1"}),
            _claim("b", "powered_by", {"device": "Q1"}, {"device": "PS1", "port": "OUT2"}),
        ]
        assert compute_standing(claims, types, None, []) == {
            "a": ("contested", ["b"]),
            "b": ("contested", ["a"]),
        }


class TestRanked:
    def test_better_kind_wins(self, types) -> None:
        claims = [_power("a", "PS1", **_cited("spec-1")), _power("b", "PS2", **_cited("dwg-1"))]
        assert compute_standing(claims, types, _RANKED, _DOCUMENTS) == {
            "a": ("accepted", ["b"]),
            "b": ("superseded", ["a"]),
        }

    def test_uncited_undeclared_and_unlisted_rank_after_listed(self, types) -> None:
        for loser in ({}, _cited("nowhere"), _cited("note-1")):
            claims = [_power("a", "PS1", **_cited("dwg-1")), _power("b", "PS2", **loser)]
            assert compute_standing(claims, types, _RANKED, _DOCUMENTS) == {
                "a": ("accepted", ["b"]),
                "b": ("superseded", ["a"]),
            }

    def test_type_list_overrides_default(self, types) -> None:
        policy = {
            "relations": {
                "precedence": {"default": ["spec", "drawing"], "powered_by": ["drawing", "spec"]}
            }
        }
        claims = [_power("a", "PS1", **_cited("spec-1")), _power("b", "PS2", **_cited("dwg-1"))]
        assert compute_standing(claims, types, policy, _DOCUMENTS) == {
            "a": ("superseded", ["b"]),
            "b": ("accepted", ["a"]),
        }

    def test_tie_at_top_with_weaker_third(self, types) -> None:
        claims = [
            _power("a", "PS1", **_cited("spec-1")),
            _power("b", "PS2", **_cited("spec-2")),
            _power("c", "PS3", **_cited("dwg-1")),
        ]
        assert compute_standing(claims, types, _RANKED, _DOCUMENTS) == {
            "a": ("contested", ["b", "c"]),
            "b": ("contested", ["a", "c"]),
            "c": ("superseded", ["a", "b"]),
        }

    def test_group_ranks_by_its_best_claim(self, types) -> None:
        claims = [
            _power("a", "PS1", **_cited("note-1")),
            _power("a2", "PS1", **_cited("spec-1")),
            _power("b", "PS2", **_cited("dwg-1")),
        ]
        assert compute_standing(claims, types, _RANKED, _DOCUMENTS) == {
            "a": ("accepted", ["b"]),
            "a2": ("accepted", ["b"]),
            "b": ("superseded", ["a", "a2"]),
        }


class TestBasis:
    def test_stated_over_inferred_breaks_a_rank_tie(self, types) -> None:
        claims = [
            _power("a", "PS1", **_cited("spec-1")),
            _power("b", "PS2", basis="inferred", rule="r", **_cited("spec-2")),
        ]
        assert compute_standing(claims, types, _RANKED, _DOCUMENTS) == {
            "a": ("accepted", ["b"]),
            "b": ("superseded", ["a"]),
        }

    def test_explicit_stated_basis_counts_as_stated(self, types) -> None:
        claims = [
            _power("a", "PS1", basis="stated"),
            _power("b", "PS2", basis="inferred", rule="r"),
        ]
        assert compute_standing(claims, types, _RANKED, _DOCUMENTS)["a"] == ("accepted", ["b"])

    def test_basis_does_nothing_without_policy(self, types) -> None:
        claims = [_power("a", "PS1"), _power("b", "PS2", basis="inferred", rule="r")]
        assert compute_standing(claims, types, None, _DOCUMENTS) == {
            "a": ("contested", ["b"]),
            "b": ("contested", ["a"]),
        }

    def test_basis_does_nothing_when_turned_off(self, types) -> None:
        policy = {
            "relations": {
                "precedence": {"default": ["spec", "drawing"]},
                "stated_over_inferred": False,
            }
        }
        claims = [_power("a", "PS1"), _power("b", "PS2", basis="inferred", rule="r")]
        assert compute_standing(claims, types, policy, _DOCUMENTS) == {
            "a": ("contested", ["b"]),
            "b": ("contested", ["a"]),
        }

    def test_kind_outranks_basis(self, types) -> None:
        claims = [
            _power("a", "PS1", **_cited("dwg-1")),
            _power("b", "PS2", basis="inferred", rule="r", **_cited("spec-1")),
        ]
        assert compute_standing(claims, types, _RANKED, _DOCUMENTS)["b"] == ("accepted", ["a"])


class TestUndecided:
    _POLICY = {"relations": {"precedence": {"default": "undecided"}}}

    def test_undecided_default_ranks_no_kind(self, types) -> None:
        claims = [_power("a", "PS1", **_cited("spec-1")), _power("b", "PS2", **_cited("note-1"))]
        assert compute_standing(claims, types, self._POLICY, _DOCUMENTS) == {
            "a": ("contested", ["b"]),
            "b": ("contested", ["a"]),
        }

    def test_undecided_default_still_orders_stated_over_inferred(self, types) -> None:
        # The policy file exists, so `stated_over_inferred` (absent means true) is in force.
        claims = [_power("a", "PS1"), _power("b", "PS2", basis="inferred", rule="r")]
        assert compute_standing(claims, types, self._POLICY, _DOCUMENTS) == {
            "a": ("accepted", ["b"]),
            "b": ("superseded", ["a"]),
        }

    def test_type_list_ranks_beside_undecided_default(self, types) -> None:
        policy = {"relations": {"precedence": {"default": "undecided", "powered_by": ["spec"]}}}
        claims = [_power("a", "PS1", **_cited("spec-1")), _power("b", "PS2", **_cited("dwg-1"))]
        assert compute_standing(claims, types, policy, _DOCUMENTS)["a"] == ("accepted", ["b"])


class TestCoarse:
    def test_unitless_place_is_coarse_and_units_contest(self, types) -> None:
        claims = [
            _claim("r", "located_in", {"device": "D1"}, {"place": "R1"}),
            _claim("u5", "located_in", {"device": "D1"}, {"place": "R1", "unit": 5}),
            _claim("u6", "located_in", {"device": "D1"}, {"place": "R1", "unit": 6}),
        ]
        assert compute_standing(claims, types, None, []) == {
            "r": ("accepted", []),
            "u5": ("contested", ["u6"]),
            "u6": ("contested", ["u5"]),
        }

    def test_portless_object_is_coarse_beside_a_ported_one(self, types) -> None:
        claims = [
            _claim("a", "powered_by", {"device": "Q1"}, {"device": "PS1"}),
            _claim("b", "powered_by", {"device": "Q1"}, {"device": "PS1", "port": "OUT1"}),
        ]
        assert compute_standing(claims, types, None, []) == {
            "a": ("accepted", []),
            "b": ("accepted", []),
        }

    def test_place_end_where_device_allowed_is_coarse(self, types) -> None:
        claims = [
            _claim("a", "powered_by", {"device": "Q1"}, {"place": "SR"}),
            _claim("b", "powered_by", {"device": "Q1"}, {"device": "PS1"}),
        ]
        assert compute_standing(claims, types, None, []) == {
            "a": ("accepted", []),
            "b": ("accepted", []),
        }


class TestSymmetric:
    def test_pair_in_both_directions_agrees(self, types) -> None:
        claims = [
            _claim(
                "a", "networked_to", {"device": "H1", "port": "e0"}, {"device": "SW", "port": "p7"}
            ),
            _claim(
                "b", "networked_to", {"device": "SW", "port": "p7"}, {"device": "H1", "port": "e0"}
            ),
        ]
        assert compute_standing(claims, types, None, []) == {
            "a": ("accepted", []),
            "b": ("accepted", []),
        }

    def test_cabled_to_agrees_on_unordered_ends(self, types) -> None:
        cable = {"attributes": {"cable": "C-1"}}
        claims = [
            _claim("a", "cabled_to", {"device": "A"}, {"device": "B"}, **cable),
            _claim("b", "cabled_to", {"device": "B"}, {"device": "A"}, **cable),
        ]
        assert compute_standing(claims, types, None, []) == {
            "a": ("accepted", []),
            "b": ("accepted", []),
        }

    def test_cabled_to_disagreeing_ends_contest(self, types) -> None:
        cable = {"attributes": {"cable": "C-1"}}
        claims = [
            _claim("a", "cabled_to", {"device": "A"}, {"device": "B"}, **cable),
            _claim("b", "cabled_to", {"device": "A"}, {"device": "C"}, **cable),
        ]
        assert compute_standing(claims, types, None, []) == {
            "a": ("contested", ["b"]),
            "b": ("contested", ["a"]),
        }

    def test_cabled_to_without_cable_is_keyless(self, types) -> None:
        claims = [
            _claim("a", "cabled_to", {"device": "A"}, {"device": "B"}),
            _claim("b", "cabled_to", {"device": "A"}, {"device": "C"}),
        ]
        assert compute_standing(claims, types, None, []) == {
            "a": ("accepted", []),
            "b": ("accepted", []),
        }


class TestKeyless:
    def test_keyless_type_never_disputes(self, types) -> None:
        claims = [
            _claim("a", "controlled_by", {"device": "Q1"}, {"device": "IOC1"}),
            _claim("b", "controlled_by", {"device": "Q1"}, {"device": "IOC2"}),
            _claim("c", "same_as", {"device": "Q1"}, {"device": "Q9"}),
        ]
        assert compute_standing(claims, types, None, []) == {
            "a": ("accepted", []),
            "b": ("accepted", []),
            "c": ("accepted", []),
        }


class TestAcrossKeys:
    def test_winning_one_key_and_losing_another_is_superseded(self, types) -> None:
        claims = [
            _claim(
                "a",
                "networked_to",
                {"device": "X", "port": "p1"},
                {"device": "Y", "port": "p2"},
                **_cited("dwg-1"),
            ),
            _claim(
                "b",
                "networked_to",
                {"device": "X", "port": "p1"},
                {"device": "Z", "port": "p3"},
                **_cited("note-1"),
            ),
            _claim(
                "c",
                "networked_to",
                {"device": "Y", "port": "p2"},
                {"device": "W", "port": "p4"},
                **_cited("spec-1"),
            ),
        ]
        assert compute_standing(claims, types, _RANKED, _DOCUMENTS) == {
            "a": ("superseded", ["b", "c"]),
            "b": ("superseded", ["a"]),
            "c": ("accepted", ["a"]),
        }

    def test_contested_on_one_key_accepted_on_another_is_contested(self, types) -> None:
        claims = [
            _claim("a", "networked_to", {"device": "X", "port": "p1"}, {"device": "Y"}),
            _claim("b", "networked_to", {"device": "X", "port": "p1"}, {"device": "Z"}),
        ]
        assert compute_standing(claims, types, None, []) == {
            "a": ("contested", ["b"]),
            "b": ("contested", ["a"]),
        }


def _mixed_claims() -> list[dict[str, Any]]:
    return [
        _power("a", "PS1", **_cited("spec-1")),
        _power("b", "PS2", **_cited("spec-2")),
        _power("c", "PS3", **_cited("dwg-1")),
        _claim("r", "located_in", {"device": "D1"}, {"place": "R1"}),
        _claim("u5", "located_in", {"device": "D1"}, {"place": "R1", "unit": 5}),
        _claim("u6", "located_in", {"device": "D1"}, {"place": "R1", "unit": 6}),
        _claim("n1", "networked_to", {"device": "X", "port": "p1"}, {"device": "Y", "port": "p2"}),
        _claim("n2", "networked_to", {"device": "Y", "port": "p2"}, {"device": "X", "port": "p1"}),
        _claim(
            "n3",
            "networked_to",
            {"device": "Y", "port": "p2"},
            {"device": "W"},
            basis="inferred",
            rule="r",
        ),
    ]


class TestOrder:
    @pytest.mark.parametrize("policy", [None, _RANKED])
    def test_every_claim_order_gives_one_result(self, types, policy) -> None:
        claims = _mixed_claims()
        expected = compute_standing(claims, types, policy, _DOCUMENTS)
        assert {standing for standing, _ in expected.values()} >= {"accepted", "contested"}
        # 9! orders is too many for a unit test; every order of the disputed tail is covered,
        # and every rotation of the whole list.
        for order in itertools.permutations(claims[3:]):
            assert compute_standing([*claims[:3], *order], types, policy, _DOCUMENTS) == expected
        for order in itertools.permutations(claims[:4]):
            assert compute_standing([*order, *claims[4:]], types, policy, _DOCUMENTS) == expected
        for shift in range(len(claims)):
            rotated = claims[shift:] + claims[:shift]
            assert compute_standing(rotated, types, policy, _DOCUMENTS) == expected

    @pytest.mark.parametrize("policy", [None, _RANKED])
    def test_every_order_of_three_layers_gives_one_result(self, types, policy) -> None:
        claims = _mixed_claims()
        layers = [claims[0:3], claims[3:6], claims[6:9]]
        expected = compute_standing(claims, types, policy, _DOCUMENTS)
        for order in itertools.permutations(layers):
            for reverse in (False, True):
                flat = [claim for layer in order for claim in (layer[::-1] if reverse else layer)]
                assert compute_standing(flat, types, policy, _DOCUMENTS) == expected

    def test_document_order_does_not_matter(self, types) -> None:
        claims = _mixed_claims()
        expected = compute_standing(claims, types, _RANKED, _DOCUMENTS)
        assert compute_standing(claims, types, _RANKED, _DOCUMENTS[::-1]) == expected

    def test_input_is_not_mutated(self, types) -> None:
        claims = _mixed_claims()
        before = repr(claims)
        compute_standing(claims, types, _RANKED, _DOCUMENTS)
        assert repr(claims) == before
