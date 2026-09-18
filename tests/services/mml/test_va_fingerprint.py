"""The deck-to-export pairing check of ``osprey mml import``.

A 2.0 export states four facts about the ring every number in it was sampled
over, and the deck travels beside the export as its own file. Nothing but
those facts ties the two together, so the import recomputes them from the deck
it is given. The unit lanes hold the recomputation to the exporter's spelling
-- the digest over the newline-joined family names, an element count with the
parameter element in it, GeV rather than eV, one-based indices -- and the
comparison to the tolerances it is allowed.

The command lanes run the real ``mml import`` on the committed synthetic
export. Its own deck pairs; the mismatched deck, which agrees on every other
fact, is refused by the digest alone, and nothing is written. The stale-sibling
lane holds the other half of the tree's self-consistency: ``data/mml``
describes the last import and nothing else, so a sibling that import does not
carry does not survive it.

Every key set asserted here comes from ``tests/templates/mml_export_contract``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from osprey.cli.main import cli
from osprey.services.mml.loaders.mat import load_lattice
from osprey.services.mml.va.fingerprint import (
    ENERGY_TOLERANCE_GEV,
    FINGERPRINT_KEYS,
    check_fingerprint,
    lattice_fingerprint,
)
from tests.templates.mml_export_contract import VA_LATTICE_KEYS

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "mml"
SYNTHETIC = FIXTURES / "synthetic"

#: The system the synthetic export carries, from its own ``_export``.
SYSTEM = "SR"

#: The deck the synthetic export was sampled over, and the deck of another
#: ring that agrees with it on every fact but the family names.
DECK = "quokka.sr.lattice.mat"
OTHER_DECK = "mismatched.lattice.mat"

#: The files an export with no virtual-accelerator half writes, and all it writes.
ONE_ZERO_OUTPUTS = ["PROFILE.md", "ad.json", "ao.json"]


def _document(name: str) -> dict:
    """One committed document of the synthetic export."""
    return json.loads((SYNTHETIC / name).read_text(encoding="utf-8"))


def _stated() -> dict:
    """The four facts the synthetic export states about its ring."""
    return _document("quokka.sr.va.json")["lattice"]


def _ring(name: str):
    """One committed deck, as the import reads it."""
    return load_lattice(SYNTHETIC / name)


def _copy(repo: Path, *names: str) -> None:
    """Put the named synthetic export files into the deployment repo."""
    for name in names:
        shutil.copy(SYNTHETIC / name, repo / name)


def _import(*args: str):
    """Run ``osprey mml import`` with *args* in the current directory."""
    return CliRunner().invoke(cli, ["mml", "import", *args], catch_exceptions=False)


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


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A minimal deployment repo (a ``profile.yml`` marker) as the cwd."""
    root = tmp_path / "deploy"
    root.mkdir()
    (root / "profile.yml").write_text("name: scratch\n", encoding="utf-8")
    monkeypatch.chdir(root)
    return root


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


class TestImportChecksTheDeck:
    """``osprey mml import`` pairs every deck it files with the export beside it."""

    def test_the_export_and_its_own_deck_import_together(self, repo: Path):
        """The accepted pair, through the real command."""
        _copy(repo, "quokka.sr.ao.json", "quokka.sr.ad.json", "quokka.sr.va.json", DECK)

        result = _import("quokka.sr.ao.json")

        assert result.exit_code == 0, result.output
        assert (repo / "data" / "mml" / "lattice" / f"{SYSTEM}.mat").exists()

    def test_a_deck_of_another_ring_is_refused_by_name(self, repo: Path):
        """The deck sits where the export's deck belongs, and disagrees only in its names."""
        _copy(repo, "quokka.sr.ao.json", "quokka.sr.ad.json", "quokka.sr.va.json")
        shutil.copy(SYNTHETIC / OTHER_DECK, repo / DECK)

        result = _import("quokka.sr.ao.json")

        assert result.exit_code != 0
        assert "famname_sha256" in result.output
        assert SYSTEM in result.output

    def test_a_refused_pair_leaves_the_tree_untouched(self, repo: Path):
        """The refusal comes before anything is written, as every import refusal does."""
        _copy(repo, "quokka.sr.ao.json", "quokka.sr.ad.json", "quokka.sr.va.json")
        shutil.copy(SYNTHETIC / OTHER_DECK, repo / DECK)

        _import("quokka.sr.ao.json")

        assert not (repo / "data" / "mml").exists()

    def test_a_deck_with_no_virtual_accelerator_export_is_filed_unchecked(self, repo: Path):
        """There is nothing to pair a deck with when the export states no facts."""
        _copy(repo, "quokka.sr.ao.json", "quokka.sr.ad.json")
        shutil.copy(SYNTHETIC / OTHER_DECK, repo / DECK)

        result = _import("quokka.sr.ao.json")

        assert result.exit_code == 0, result.output
        assert (repo / "data" / "mml" / "lattice" / f"{SYSTEM}.mat").exists()

    def test_an_export_that_could_not_fingerprint_its_ring_is_warned_about(self, repo: Path):
        """An unchecked pair is imported, and says so."""
        _copy(repo, "quokka.sr.ao.json", "quokka.sr.ad.json", DECK)
        document = _document("quokka.sr.va.json")
        document["lattice"] = {"refused": "getenergymodel gave [] for the model energy"}
        (repo / "quokka.sr.va.json").write_text(json.dumps(document), encoding="utf-8")

        result = _import("quokka.sr.ao.json")

        assert result.exit_code == 0, result.output
        assert SYSTEM in result.output
        assert (repo / "data" / "mml" / "lattice" / f"{SYSTEM}.mat").exists()


class TestStaleSiblings:
    """``data/mml`` describes the last import: what it does not carry does not survive."""

    def test_an_export_without_a_virtual_accelerator_half_sweeps_the_earlier_one(self, repo: Path):
        """The documents of the replaced export would otherwise be read against a new one."""
        _copy(
            repo,
            "quokka.sr.ao.json",
            "quokka.sr.ad.json",
            "quokka.sr.va.json",
            "quokka.sr.response.json",
            DECK,
        )
        _import("quokka.sr.ao.json")

        result = _import(str(FIXTURES / "paired" / "quokka.ring.ao.json"))

        assert result.exit_code == 0, result.output
        written = repo / "data" / "mml"
        assert sorted(path.name for path in written.iterdir()) == ONE_ZERO_OUTPUTS

    def test_the_swept_files_are_named(self, repo: Path):
        """A file the import removes is a file the operator is told about."""
        _copy(repo, "quokka.sr.ao.json", "quokka.sr.ad.json", "quokka.sr.va.json", DECK)
        _import("quokka.sr.ao.json")

        result = _import(str(FIXTURES / "paired" / "quokka.ring.ao.json"))

        assert "va.json" in result.output
        assert f"{SYSTEM}.mat" in result.output

    def test_a_deck_of_a_system_the_import_does_not_carry_is_swept(self, repo: Path):
        """A deck the import leaves behind would be paired with an export that is gone."""
        _copy(repo, "quokka.sr.ao.json", "quokka.sr.ad.json", "quokka.sr.va.json", DECK)
        _import("quokka.sr.ao.json")
        stale = repo / "data" / "mml" / "lattice" / "OLD.mat"
        shutil.copy(SYNTHETIC / DECK, stale)

        result = _import("quokka.sr.ao.json")

        assert result.exit_code == 0, result.output
        assert not stale.exists()
        assert (repo / "data" / "mml" / "lattice" / f"{SYSTEM}.mat").exists()

    def test_a_reimport_of_the_same_export_sweeps_nothing(self, repo: Path):
        """The digests stamped into the emitted artifacts survive a re-import."""
        _copy(
            repo,
            "quokka.sr.ao.json",
            "quokka.sr.ad.json",
            "quokka.sr.va.json",
            "quokka.sr.response.json",
            DECK,
        )
        _import("quokka.sr.ao.json")
        written = repo / "data" / "mml"
        names = ("va.json", "response.json")
        before = {name: (written / name).stat().st_mtime_ns for name in names}

        result = _import("quokka.sr.ao.json")

        assert result.exit_code == 0, result.output
        assert {name: (written / name).stat().st_mtime_ns for name in names} == before
        assert (written / "lattice" / f"{SYSTEM}.mat").exists()
