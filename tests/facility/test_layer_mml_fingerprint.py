"""The deck-to-export pairing check of the mml layer.

A 2.0 export states four facts about the lattice every number in it was
sampled over, and the deck travels beside the export as its own file. Nothing
but those facts ties the two together, so they are recomputed from the deck.
These cases hold the recomputation to the exporter's spelling -- the digest
over the newline-joined family names, an element count with the parameter
element in it, GeV rather than eV, one-based indices -- and the comparison to
the tolerances it is allowed.

Every key set asserted here comes from ``tests/templates/mml_export_contract``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from osprey.facility.layers.mml.fingerprint import (
    ENERGY_TOLERANCE_GEV,
    FINGERPRINT_KEYS,
    check_fingerprint,
    lattice_fingerprint,
)
from osprey.facility.layers.mml.loaders.mat import load_lattice
from tests.templates.mml_export_contract import VA_LATTICE_KEYS

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "mml"
SYNTHETIC = FIXTURES / "synthetic"

#: The system the synthetic export carries, from its own ``_export``.
SYSTEM = "SR"

#: The deck the synthetic export was sampled over, and the deck of another
#: ring that agrees with it on every fact but the family names.
DECK = "quokka.sr.lattice.mat"
OTHER_DECK = "mismatched.lattice.mat"


def _document(name: str) -> dict:
    """One committed document of the synthetic export."""
    return json.loads((SYNTHETIC / name).read_text(encoding="utf-8"))


def _stated() -> dict:
    """The four facts the synthetic export states about its ring."""
    return _document("quokka.sr.va.json")["lattice"]


def _ring(name: str):
    """One committed deck, as the import reads it."""
    return load_lattice(SYNTHETIC / name)


def _facts(**overrides: Any) -> dict:
    """A recomputed fingerprint, with the named facts replaced."""
    facts = {
        "elements": 41,
        "famname_sha256": "a" * 64,
        "energy_gev": 2.0,
        "ringparam_indices": (1,),
    }
    facts.update(overrides)
    return facts


class _Element:
    """A deck element, carrying whichever attributes the deck spells it with."""

    def __init__(self, fam_name: str, **attributes: Any) -> None:
        self.FamName = fam_name
        for name, value in attributes.items():
            setattr(self, name, value)


class _Ring(list):
    """A ring stand-in: its elements in order and the energy it holds, in eV."""

    def __init__(self, elements: list[_Element], energy: float) -> None:
        super().__init__(elements)
        self.energy = energy


class TestLatticeFingerprint:
    """The four facts, recomputed from the deck the way the exporter took them."""

    def test_the_committed_deck_answers_the_export_it_was_sampled_for(self):
        """The pair the whole check exists to accept."""
        recomputed = lattice_fingerprint(_ring(DECK))

        assert set(recomputed) == set(VA_LATTICE_KEYS)
        assert check_fingerprint(_stated(), recomputed) is None

    def test_the_digest_is_over_the_newline_joined_names_with_none_trailing(self):
        """The join is the digest's whole spelling, and a trailing newline is not in it."""
        ring = _ring(DECK)
        names = [element.FamName for element in ring]

        recomputed = lattice_fingerprint(ring)

        joined = "\n".join(names)
        assert recomputed["famname_sha256"] == hashlib.sha256(joined.encode("utf-8")).hexdigest()
        trailing = hashlib.sha256((joined + "\n").encode("utf-8")).hexdigest()
        assert recomputed["famname_sha256"] != trailing

    def test_the_element_count_holds_the_parameter_element(self):
        """The count is the deck's own, so the indices the Middle Layer carries hold."""
        ring = _ring(DECK)

        recomputed = lattice_fingerprint(ring)

        assert recomputed["elements"] == len(ring)
        assert recomputed["ringparam_indices"] == (1,)
        assert ring[0].FamName == ring.name

    def test_the_energy_is_stated_in_gev(self):
        """The deck holds eV and the export states GeV; the check compares GeV."""
        ring = _ring(DECK)

        recomputed = lattice_fingerprint(ring)

        assert recomputed["energy_gev"] == ring.energy / 1e9
        assert recomputed["energy_gev"] == pytest.approx(2.0)

    def test_a_parameter_element_is_found_under_either_spelling(self):
        """A deck names its class; a ring read back from one tags it."""
        by_class = _Ring([_Element("ring", Class="RingParam"), _Element("QF")], 2e9)
        by_tag = _Ring([_Element("ring", tag="RingParam"), _Element("QF")], 2e9)

        assert lattice_fingerprint(by_class)["ringparam_indices"] == (1,)
        assert lattice_fingerprint(by_tag)["ringparam_indices"] == (1,)

    def test_a_ring_carrying_no_parameter_element_states_no_index(self):
        """Nothing is invented for a ring that has none."""
        recomputed = lattice_fingerprint(_Ring([_Element("QF"), _Element("QD")], 3e9))

        assert recomputed["ringparam_indices"] == ()
        assert recomputed["elements"] == 2

    def test_the_other_deck_differs_in_the_digest_and_in_nothing_else(self):
        """The counter-example is caught by the family names alone."""
        sampled = lattice_fingerprint(_ring(DECK))
        other = lattice_fingerprint(_ring(OTHER_DECK))

        assert other["famname_sha256"] != sampled["famname_sha256"]
        assert {key: other[key] for key in FINGERPRINT_KEYS if key != "famname_sha256"} == {
            key: sampled[key] for key in FINGERPRINT_KEYS if key != "famname_sha256"
        }
        mismatch = check_fingerprint(_stated(), other)
        assert mismatch is not None
        assert mismatch.field == "famname_sha256"


class TestCheckFingerprint:
    """What the comparison accepts, and which field it names when it does not."""

    def test_an_agreeing_pair_is_no_mismatch(self):
        """The recomputed facts, spelled as the exporter spells them, pair."""
        stated = {
            "elements": 41,
            "famname_sha256": "a" * 64,
            "energy_gev": 2.0,
            "ringparam_indices": 1,
        }

        assert check_fingerprint(stated, _facts()) is None

    def test_one_index_is_the_one_entry_list_it_is(self):
        """A single index is written as a number, and read as the list of one."""
        stated = {**_facts(), "ringparam_indices": 1}

        assert check_fingerprint(stated, _facts(ringparam_indices=(1,))) is None

    def test_several_indices_are_read_in_the_order_they_are_written(self):
        """A ring with more than one parameter element states them as a list."""
        stated = {**_facts(), "ringparam_indices": [1, 22]}

        assert check_fingerprint(stated, _facts(ringparam_indices=(1, 22))) is None
        assert check_fingerprint(stated, _facts(ringparam_indices=(22, 1))) is not None

    def test_an_empty_list_of_indices_pairs_with_a_ring_that_has_none(self):
        """No parameter element is a fact like any other, stated on both sides."""
        stated = {**_facts(), "ringparam_indices": []}

        assert check_fingerprint(stated, _facts(ringparam_indices=())) is None

    def test_a_count_written_as_a_whole_number_of_doubles_agrees(self):
        """MATLAB counts in doubles; 41 elements is 41 either way."""
        stated = {**_facts(), "elements": 41.0, "ringparam_indices": 1}

        assert check_fingerprint(stated, _facts()) is None

    def test_an_energy_inside_the_tolerance_agrees(self):
        """Two paths to the same number differ in the last places, not in the machine."""
        stated = {**_facts(), "energy_gev": 2.0 + ENERGY_TOLERANCE_GEV / 2, "ringparam_indices": 1}

        assert check_fingerprint(stated, _facts()) is None

    def test_an_energy_outside_the_tolerance_names_the_energy(self):
        """A different operating point is a different set of calibrations."""
        stated = {**_facts(), "energy_gev": 2.5, "ringparam_indices": 1}

        mismatch = check_fingerprint(stated, _facts())

        assert mismatch is not None
        assert mismatch.field == "energy_gev"
        assert mismatch.expected == 2.5
        assert mismatch.actual == 2.0

    def test_a_digest_of_another_ring_names_the_digest(self):
        """The field the message must name for the pair the check exists to refuse."""
        stated = {**_facts(), "famname_sha256": "b" * 64, "ringparam_indices": 1}

        mismatch = check_fingerprint(stated, _facts())

        assert mismatch is not None
        assert mismatch.field == "famname_sha256"

    def test_a_count_disagreement_names_the_count(self):
        """A deck of another length is refused before its names are compared."""
        mismatch = check_fingerprint({**_facts(), "elements": 40}, _facts())

        assert mismatch is not None
        assert mismatch.field == "elements"

    def test_a_fact_the_export_does_not_state_is_a_disagreement(self):
        """A fingerprint missing a fact cannot pair: there is nothing to pair with."""
        stated = {key: value for key, value in _facts().items() if key != "famname_sha256"}

        mismatch = check_fingerprint(stated, _facts())

        assert mismatch is not None
        assert mismatch.field == "famname_sha256"
        assert mismatch.expected is None

    def test_the_first_fact_of_the_stated_order_is_the_one_named(self):
        """One message, one field: the comparison walks the facts in their own order."""
        stated = {**_facts(), "elements": 40, "famname_sha256": "b" * 64}

        mismatch = check_fingerprint(stated, _facts())

        assert mismatch is not None
        assert mismatch.field == FINGERPRINT_KEYS[0]
