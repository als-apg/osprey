"""The rules that decide what the virtual accelerator does with each family.

The committed synthetic export is the fixture these lanes are built on: it
carries one family per rule -- a strength, a sliced kick, a monitor of each
plane, a cavity, an energy candidate with a knob and one without, a dipole trim
that takes no kick, an escape hatch, an unknown type and three families with no
lattice element at all -- and a deck to read them against. Every rule is pinned
against it, and the cases it does not carry are pinned against a copy of it
altered in the one place the rule turns on, so that what a lane proves is the
rule rather than the fixture.

The exports of the two real facilities pin the same rules against family names
and type tokens nobody invented. Their re-exported 2.0 trees are not committed,
so the lanes that need a full export find it or skip, naming what is missing;
the lanes that need only the type token and the indices beside it run against
the committed exports as they stand.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from osprey.services.mml.family import family_views
from osprey.services.mml.loaders.mat import load_lattice
from osprey.services.mml.va.verdicts import (
    ATTYPE_TABLE,
    CORRECTOR_MEMBERSHIP,
    ENERGY_TOLERANCE_GEV,
    attype_slot_opens,
    propose,
    resolve_attype,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "mml"
SYNTHETIC = FIXTURES / "synthetic"

#: The stem the synthetic export's five files share.
STEM = "quokka.sr"

#: The system the synthetic export describes.
SYSTEM = "SR"

#: The committed exports of the two facilities whose verdicts are pinned, and
#: the sub-machine of each whose families the rules decide.
REAL_EXPORTS = {
    "spear3": "spear3.storagering",
    "nsls2": "nsls2.storagering",
}


def _document(name: str) -> dict:
    """One committed document, freshly read so a lane may alter its copy."""
    return json.loads((SYNTHETIC / name).read_text(encoding="utf-8"))


def _views(ao: dict) -> dict:
    """The computed grain of one export's families, keyed as the export keys them."""
    return {view.raw_name: view for view in family_views(SYSTEM, ao)}


@pytest.fixture(scope="module")
def deck():
    """The deck the synthetic export was sampled over."""
    return load_lattice(SYNTHETIC / f"{STEM}.lattice.mat")


@pytest.fixture
def export() -> dict:
    """The synthetic export's virtual-accelerator block."""
    return _document(f"{STEM}.va.json")


@pytest.fixture
def objects() -> dict:
    """The synthetic export's accelerator objects."""
    return _document(f"{STEM}.ao.json")


@pytest.fixture
def verdicts(export: dict, deck, objects: dict) -> dict:
    """What the rules make of the synthetic export as it stands."""
    return propose(export, deck, _views(objects))


def _element(name: str = "E", pass_method: str = "IdentityPass", **attributes: Any):
    """One stand-in element, carrying only the attributes a lane is about."""
    return SimpleNamespace(FamName=name, PassMethod=pass_method, **attributes)


class _Deck(list):
    """A deck of stand-in elements, carrying an energy the way a ring does."""

    def __init__(self, elements: list, energy_gev: float = 2.0) -> None:
        super().__init__(elements)
        self.energy = energy_gev * 1e9


class TestTheSyntheticExport:
    """Every family of the committed export, decided against its own deck."""

    def test_every_family_of_the_export_is_answered_exactly_once(
        self, verdicts: dict, export: dict
    ) -> None:
        assert list(verdicts) == list(export["families"])

    @pytest.mark.parametrize(
        ("family", "element_field"),
        [
            ("QF", "PolynomB[1]"),
            ("QD", "PolynomB[1]"),
            ("SF", "PolynomB[2]"),
            ("SQ", "PolynomA[1]"),
        ],
    )
    def test_a_magnet_couples_on_the_multipole_its_type_names(
        self, verdicts: dict, family: str, element_field: str
    ) -> None:
        verdict = verdicts[family]
        assert (verdict.verdict, verdict.kind, verdict.element_field) == (
            "couple",
            "strength",
            element_field,
        )
        assert verdict.slot is None

    @pytest.mark.parametrize(
        ("family", "element_field"), [("HC", "KickAngle[0]"), ("VC", "KickAngle[1]")]
    )
    def test_a_corrector_couples_on_the_plane_its_type_names(
        self, verdicts: dict, family: str, element_field: str
    ) -> None:
        verdict = verdicts[family]
        assert (verdict.verdict, verdict.kind, verdict.element_field) == (
            "couple",
            "kick",
            element_field,
        )

    def test_a_sliced_kick_couples_although_one_device_is_short_a_slice(
        self, verdicts: dict, export: dict
    ) -> None:
        stated = export["families"]["HC"]["nominals"]["Setpoint"]["at_index"]
        assert any("NaN" in row for row in stated)
        assert verdicts["HC"].verdict == "couple"

    @pytest.mark.parametrize(("family", "plane"), [("BPMx", "x"), ("BPMy", "y")])
    def test_a_beam_monitor_couples_on_the_plane_it_reads(
        self, verdicts: dict, family: str, plane: str
    ) -> None:
        verdict = verdicts[family]
        assert (verdict.verdict, verdict.kind, verdict.element_field) == (
            "couple",
            "monitor",
            plane,
        )
        assert verdict.nominal_source == "Monitor"

    def test_the_energy_candidate_with_a_ramp_becomes_the_knob(self, verdicts: dict) -> None:
        verdict = verdicts["BEND"]
        assert (verdict.verdict, verdict.kind) == ("couple", "energy")
        assert verdict.element_field is None
        assert verdict.calibration == "table"

    def test_the_energy_candidate_with_a_constant_ramp_latches(self, verdicts: dict) -> None:
        verdict = verdicts["BSOFT"]
        assert (verdict.verdict, verdict.reason) == (
            "latch",
            "bend2gev is constant at this facility",
        )
        assert verdict.slot is None

    def test_a_dipole_trim_whose_elements_take_no_kick_latches_on_the_element(
        self, verdicts: dict
    ) -> None:
        verdict = verdicts["BDM"]
        assert verdict.verdict == "latch"
        assert verdict.reason == "element BD1 (BndMPoleSymplectic4Pass) takes no KickAngle"
        assert verdict.slot is None

    def test_the_cavity_couples_on_its_class_and_not_on_its_index(
        self, verdicts: dict, export: dict
    ) -> None:
        assert export["families"]["RF"]["nominals"]["Setpoint"]["at_index"] == 41
        verdict = verdicts["RF"]
        assert (verdict.verdict, verdict.kind, verdict.element_field) == ("couple", "rf", None)

    def test_a_family_reached_through_a_hook_asks_before_it_is_driven(self, verdicts: dict) -> None:
        verdict = verdicts["IDGAP"]
        assert verdict.verdict == "latch"
        assert verdict.slot is not None
        assert verdict.slot.kind == "escape_hatch"
        assert verdict.slot.answer is None
        assert "SpecialFunctionSet" in verdict.slot.question

    def test_an_unknown_type_over_bound_elements_asks_what_the_family_drives(
        self, verdicts: dict
    ) -> None:
        verdict = verdicts["SEPTUM"]
        assert verdict.verdict == "latch"
        assert verdict.slot is not None
        assert verdict.slot.kind == "attype"
        assert "Septum" in verdict.slot.question

    @pytest.mark.parametrize("family", ["DCCT", "TUNE", "Version"])
    def test_a_family_with_no_lattice_element_latches_on_one_shared_reason(
        self, verdicts: dict, family: str
    ) -> None:
        verdict = verdicts[family]
        assert (verdict.verdict, verdict.reason, verdict.slot) == (
            "latch",
            "no lattice element",
            None,
        )

    def test_the_export_leaves_exactly_two_questions_open(self, verdicts: dict) -> None:
        open_slots = {
            name: verdict.slot.kind
            for name, verdict in verdicts.items()
            if verdict.slot is not None
        }
        assert open_slots == {"IDGAP": "escape_hatch", "SEPTUM": "attype"}

    def test_nothing_that_carries_a_question_is_coupled(self, verdicts: dict) -> None:
        assert not [
            name
            for name, verdict in verdicts.items()
            if verdict.slot is not None and verdict.verdict != "latch"
        ]

    def test_a_coupled_family_names_the_field_its_nominal_was_read_from(
        self, verdicts: dict, export: dict
    ) -> None:
        for name, verdict in verdicts.items():
            if verdict.verdict != "couple":
                continue
            assert verdict.nominal_source in export["families"][name]["nominals"]


class TestTheEnergyRule:
    """The candidate set decides itself before any type token is read."""

    def test_a_ramp_that_disagrees_with_the_deck_latches_naming_both_energies(
        self, export: dict, deck, objects: dict
    ) -> None:
        table = export["families"]["BEND"]["energy_table"]
        table["energy_at_nominal"] = deck.energy / 1e9 + 2 * ENERGY_TOLERANCE_GEV
        verdict = propose(export, deck, _views(objects))["BEND"]
        assert verdict.verdict == "latch"
        assert "2.002 GeV" in verdict.reason
        assert "2 GeV" in verdict.reason

    def test_a_ramp_inside_the_tolerance_still_becomes_the_knob(
        self, export: dict, deck, objects: dict
    ) -> None:
        table = export["families"]["BEND"]["energy_table"]
        table["energy_at_nominal"] = deck.energy / 1e9 + ENERGY_TOLERANCE_GEV / 2
        assert propose(export, deck, _views(objects))["BEND"].kind == "energy"

    def test_a_second_family_with_a_ramp_is_asked_about_rather_than_bound(
        self, export: dict, deck, objects: dict
    ) -> None:
        table = export["families"]["BSOFT"]["energy_table"]
        table["values"] = [2.0 + step / 100 for step in range(len(table["values"]))]
        table["energy_at_nominal"] = deck.energy / 1e9
        verdicts = propose(export, deck, _views(objects))
        assert verdicts["BEND"].kind == "energy"
        assert verdicts["BSOFT"].verdict == "latch"
        assert verdicts["BSOFT"].slot.kind == "attype"
        assert "BEND" in verdicts["BSOFT"].slot.question

    def test_a_candidate_never_reaches_the_type_table(
        self, export: dict, deck, objects: dict
    ) -> None:
        objects["BSOFT"]["AT"] = {"ATType": "SEXT", "ATIndex": [6, 16]}
        export["families"]["BSOFT"]["nominals"]["Setpoint"]["at_type"] = "SEXT"
        export["families"]["BSOFT"]["nominals"]["Setpoint"]["at_index"] = [6, 16]
        verdict = propose(export, deck, _views(objects))["BSOFT"]
        assert verdict.reason == "bend2gev is constant at this facility"
        assert verdict.kind is None

    def test_a_candidate_with_no_ramp_at_all_latches(
        self, export: dict, deck, objects: dict
    ) -> None:
        del export["families"]["BEND"]["energy_table"]
        verdict = propose(export, deck, _views(objects))["BEND"]
        assert (verdict.verdict, verdict.reason) == ("latch", "the export states no energy table")

    def test_a_dipole_the_export_left_out_of_the_candidate_set_never_takes_the_knob(
        self, export: dict, deck, objects: dict
    ) -> None:
        objects["BDM"]["MemberOf"] = ["BEND", "Magnet"]
        verdicts = propose(export, deck, _views(objects))
        assert verdicts["BEND"].kind == "energy"
        assert verdicts["BDM"].verdict == "latch"
        assert verdicts["BDM"].slot.kind == "attype"
        assert "BEND" in verdicts["BDM"].slot.question

    def test_a_dipole_with_no_ramp_behind_it_and_no_knob_to_answer_to_latches(
        self, export: dict, deck, objects: dict
    ) -> None:
        objects["BDM"]["MemberOf"] = ["BEND", "Magnet"]
        export["families"]["BEND"]["energy_candidate"] = 0
        verdict = propose(export, deck, _views(objects))["BDM"]
        assert (verdict.verdict, verdict.reason) == ("latch", "the export states no energy table")
        assert verdict.slot is None


class TestTheCavityRule:
    """A cavity is bound by class, so the indices beside it decide nothing."""

    def test_the_cavity_couples_although_the_export_binds_no_element(
        self, export: dict, deck, objects: dict
    ) -> None:
        export["families"]["RF"]["nominals"]["Setpoint"]["at_index"] = []
        assert propose(export, deck, _views(objects))["RF"].verdict == "couple"

    def test_a_deck_with_no_cavity_latches_the_family_that_asks_for_one(
        self, export: dict, objects: dict
    ) -> None:
        verdict = propose(export, _Deck([_element()]), _views(objects))["RF"]
        assert (verdict.verdict, verdict.reason) == ("latch", "the deck holds no cavity")

    def test_a_cavity_is_recognised_by_the_class_a_deck_tags_it_with(
        self, export: dict, objects: dict
    ) -> None:
        deck = _Deck([_element("RFC", Class="RFCavity")])
        assert propose(export, deck, _views(objects))["RF"].verdict == "couple"


class TestTheDipoleTrim:
    """A dipole-typed family whose membership also names the correctors."""

    def test_a_trim_whose_elements_take_a_kick_is_asked_which_plane_it_drives(
        self, export: dict, deck, objects: dict
    ) -> None:
        export["families"]["BDM"]["nominals"]["Setpoint"]["at_index"] = [9, 19]
        verdict = propose(export, deck, _views(objects))["BDM"]
        assert verdict.verdict == "latch"
        assert verdict.slot.kind == "attype"
        assert "which plane of KickAngle" in verdict.slot.question

    def test_the_plane_is_never_read_out_of_the_family_name(
        self, export: dict, deck, objects: dict
    ) -> None:
        export["families"]["BDM"]["nominals"]["Setpoint"]["at_index"] = [9, 19]
        assert propose(export, deck, _views(objects))["BDM"].element_field is None

    def test_a_trim_that_binds_no_element_latches_without_a_question(
        self, export: dict, deck, objects: dict
    ) -> None:
        export["families"]["BDM"]["nominals"]["Setpoint"]["at_index"] = []
        verdict = propose(export, deck, _views(objects))["BDM"]
        assert (verdict.verdict, verdict.reason, verdict.slot) == (
            "latch",
            "no lattice element",
            None,
        )


class TestTheTypeTable:
    """What a type token resolves to, and what an unresolved one asks."""

    @pytest.mark.parametrize(
        ("token", "resolved"),
        [
            ("QUAD", ("strength", "PolynomB[1]")),
            ("quadrupole", ("strength", "PolynomB[1]")),
            ("K2", ("strength", "PolynomB[2]")),
            ("OCTU", ("strength", "PolynomB[3]")),
            ("SKEWQUAD", ("strength", "PolynomA[1]")),
            ("SkewQ", ("strength", "PolynomA[1]")),
            ("HCOR", ("kick", "KickAngle[0]")),
            ("VCM", ("kick", "KickAngle[1]")),
            ("BEND", ("energy", None)),
            ("RF Cavity", ("rf", None)),
            ("xTurns", ("monitor", "x")),
            ("Y", ("monitor", "y")),
        ],
    )
    def test_a_token_resolves_whatever_case_it_is_written_in(
        self, token: str, resolved: tuple
    ) -> None:
        assert resolve_attype(token) == resolved
        assert resolve_attype(token.swapcase()) == resolved
        assert resolve_attype(f"  {token} ") == resolved

    @pytest.mark.parametrize("token", ["", None, "Septum", "KickerAmp", "MachineParameters"])
    def test_a_token_the_table_does_not_know_resolves_to_nothing(self, token: Any) -> None:
        assert resolve_attype(token) is None

    def test_an_unknown_token_asks_only_when_elements_are_bound(self) -> None:
        assert attype_slot_opens("Septum", 146)
        assert attype_slot_opens("Septum", [4, 14])
        assert not attype_slot_opens("Septum", [])
        assert not attype_slot_opens("Septum", "")
        assert not attype_slot_opens("Septum", None)

    def test_a_known_token_never_asks(self) -> None:
        assert not attype_slot_opens("QUAD", [2, 12])

    def test_an_unknown_token_with_no_elements_latches_without_a_question(
        self, export: dict, deck, objects: dict
    ) -> None:
        export["families"]["SEPTUM"]["nominals"]["Monitor"]["at_index"] = []
        verdict = propose(export, deck, _views(objects))["SEPTUM"]
        assert (verdict.verdict, verdict.reason, verdict.slot) == (
            "latch",
            "no lattice element",
            None,
        )


class TestTheElementRules:
    """The checks a family passes before the model is allowed to drive it."""

    def test_units_on_either_field_carry_the_family(
        self, export: dict, deck, objects: dict
    ) -> None:
        objects["QF"]["Setpoint"]["PhysicsUnits"] = "rad"
        verdict = propose(export, deck, _views(objects))["QF"]
        assert verdict.verdict == "couple"
        assert verdict.reason == "Setpoint states rad where Monitor states its units"

    def test_units_neither_field_states_ask_what_the_family_drives(
        self, export: dict, deck, objects: dict
    ) -> None:
        objects["QF"]["Setpoint"]["PhysicsUnits"] = ""
        objects["QF"]["Monitor"]["PhysicsUnits"] = ""
        verdict = propose(export, deck, _views(objects))["QF"]
        assert verdict.verdict == "latch"
        assert verdict.slot.kind == "attype"
        assert "strength units" in verdict.slot.question

    def test_a_corrector_stating_a_length_on_both_fields_asks(
        self, export: dict, deck, objects: dict
    ) -> None:
        objects["HC"]["Setpoint"]["PhysicsUnits"] = "mm"
        objects["HC"]["Monitor"]["PhysicsUnits"] = "Meter"
        verdict = propose(export, deck, _views(objects))["HC"]
        assert verdict.slot.kind == "attype"
        assert "kick units" in verdict.slot.question

    def test_a_multipole_shorter_than_the_index_latches_on_the_element(
        self, export: dict, objects: dict
    ) -> None:
        deck = _Deck([_element("QF1", "StrMPoleSymplectic4Pass", PolynomB=[0.0, 1.2])])
        export["families"]["SF"]["nominals"]["Setpoint"]["at_index"] = [1]
        verdict = propose(export, deck, _views(objects))["SF"]
        assert verdict.verdict == "latch"
        assert verdict.reason == "element QF1 (StrMPoleSymplectic4Pass) takes no PolynomB"

    def test_an_index_past_the_end_of_the_deck_latches(
        self, export: dict, deck, objects: dict
    ) -> None:
        export["families"]["QF"]["nominals"]["Setpoint"]["at_index"] = [2, len(deck) + 1]
        verdict = propose(export, deck, _views(objects))["QF"]
        assert verdict.verdict == "latch"
        assert f"past the end of a ring of {len(deck)}" in verdict.reason

    def test_two_families_driving_one_field_of_one_element_both_ask_which_binds_it(
        self, export: dict, deck, objects: dict
    ) -> None:
        objects["SQ"]["AT"]["ATType"] = "K2"
        export["families"]["SQ"]["nominals"]["Setpoint"]["at_type"] = "K2"
        verdicts = propose(export, deck, _views(objects))
        for name, other in (("SF", "SQ"), ("SQ", "SF")):
            verdict = verdicts[name]
            assert verdict.verdict == "latch"
            assert verdict.slot.kind == "shared_field"
            assert other in verdict.slot.question
            assert "PolynomB[2]" in verdict.slot.question

    def test_families_sharing_an_element_through_different_fields_ask_nothing(
        self, verdicts: dict, export: dict
    ) -> None:
        for pair in (("SF", "SQ"), ("HC", "VC"), ("BPMx", "BPMy")):
            shared = {
                family: export["families"][family]["nominals"]["Setpoint"]["at_index"]
                if "Setpoint" in export["families"][family]["nominals"]
                else export["families"][family]["nominals"]["Monitor"]["at_index"]
                for family in pair
            }
            assert shared[pair[0]] == shared[pair[1]]
            assert all(verdicts[family].slot is None for family in pair)

    def test_the_energy_knob_sharing_an_element_asks_nothing(self, verdicts: dict) -> None:
        assert verdicts["BEND"].slot is None
        assert verdicts["BDM"].slot is None

    def test_a_hook_on_the_coupled_field_stops_a_family_the_table_resolved(
        self, export: dict, deck, objects: dict
    ) -> None:
        objects["QF"]["Setpoint"]["AT"] = {"ATParameterGroup": "PolynomB"}
        verdict = propose(export, deck, _views(objects))["QF"]
        assert verdict.verdict == "latch"
        assert verdict.slot.kind == "escape_hatch"
        assert "ATParameterGroup" in verdict.slot.question

    def test_a_hook_on_a_field_the_family_does_not_couple_through_is_no_question(
        self, export: dict, deck, objects: dict
    ) -> None:
        objects["QF"]["Monitor"]["AT"] = {"SpecialFunctionGet": {"$fn": "read_it", "file": ""}}
        assert propose(export, deck, _views(objects))["QF"].verdict == "couple"

    def test_the_cavity_is_never_stopped_by_a_hook(self, export: dict, deck, objects: dict) -> None:
        objects["RF"]["AT"]["SpecialFunctionSet"] = {"$fn": "set_it", "file": ""}
        assert propose(export, deck, _views(objects))["RF"].verdict == "couple"


class TestTheDeckIsTheAuthority:
    """Every fact about the ring is read from the deck, not from the export."""

    def test_an_export_refusing_to_state_its_lattice_facts_decides_the_same_families(
        self, export: dict, deck, objects: dict
    ) -> None:
        stated = propose(copy.deepcopy(export), deck, _views(objects))
        export["lattice"] = {"refused": "the ring was not saved"}
        assert propose(export, deck, _views(objects)) == stated

    def test_the_deck_energy_rather_than_the_stated_one_decides_the_knob(
        self, export: dict, objects: dict, deck
    ) -> None:
        export["lattice"]["energy_gev"] = 3.0
        assert propose(export, deck, _views(objects))["BEND"].kind == "energy"

    def test_a_family_with_no_view_of_its_own_asks_rather_than_couples(
        self, export: dict, deck, objects: dict
    ) -> None:
        views = _views(objects)
        del views["QF"]
        assert propose(export, deck, views)["QF"].slot.kind == "attype"

    def test_a_block_with_no_families_decides_nothing(self, deck) -> None:
        assert propose({"lattice": {}}, deck, {}) == {}


def _real_export(facility: str) -> dict:
    """The committed accelerator objects of one facility's storage ring."""
    path = FIXTURES / facility / f"{REAL_EXPORTS[facility]}.ao.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _stated_type(family: dict) -> tuple[Any, Any]:
    """The lattice type and indices a family states at family level."""
    block = family.get("AT")
    if not isinstance(block, dict):
        return None, None
    return block.get("ATType"), block.get("ATIndex")


def _two_zero_tree(facility: str) -> tuple[dict, Path] | None:
    """A committed 2.0 export of one facility, once its re-export lands."""
    for document in sorted((FIXTURES / facility).rglob("*.va.json")):
        deck = document.with_name(document.name[: -len(".va.json")] + ".lattice.mat")
        objects = document.with_name(document.name[: -len(".va.json")] + ".ao.json")
        if deck.exists() and objects.exists():
            return json.loads(document.read_text(encoding="utf-8")), deck
    return None


def _real_verdicts(facility: str) -> dict:
    """What the rules make of a facility's 2.0 export, skipping until it exists."""
    found = _two_zero_tree(facility)
    if found is None:
        pytest.skip(
            f"no 2.0 export under {FIXTURES / facility}: the re-exported "
            f"{REAL_EXPORTS[facility]}.va.json and .lattice.mat are not committed"
        )
    document, deck = found
    block = document if "families" in document else next(iter(document.values()))
    return propose(block, load_lattice(deck), _views(_real_export(facility)))


class TestTheFirstFacility:
    """The rules against a storage ring nobody wrote for this test."""

    FACILITY = "spear3"

    def test_the_families_it_asks_about_are_the_ones_with_an_unknown_type(self) -> None:
        objects = _real_export(self.FACILITY)
        asking = {
            name
            for name, family in objects.items()
            if isinstance(family, dict) and attype_slot_opens(*_stated_type(family))
        }
        assert asking == {
            "HCMCurrReference",
            "VCMCurrReference",
            "KickerAmp",
            "KickerDelay",
            "Septum",
        }

    def test_a_family_with_an_unknown_type_and_no_elements_is_not_one_of_them(self) -> None:
        objects = _real_export(self.FACILITY)
        token, indices = _stated_type(objects["MachineParameters"])
        assert resolve_attype(token) is None
        assert not indices
        assert not attype_slot_opens(token, indices)

    def test_the_dipole_trims_are_the_families_the_corrector_rule_claims(self) -> None:
        objects = _real_export(self.FACILITY)
        trims = {
            name
            for name, family in objects.items()
            if isinstance(family, dict)
            and resolve_attype(_stated_type(family)[0]) == ATTYPE_TABLE["bend"]
            and CORRECTOR_MEMBERSHIP in (family.get("MemberOf") or [])
        }
        assert trims == {"BDM", "CD"}

    @pytest.mark.parametrize(
        ("family", "resolved"),
        [
            ("HCM", ("kick", "KickAngle[0]")),
            ("VCM", ("kick", "KickAngle[1]")),
            ("RF", ("rf", None)),
            ("SkewQuad", ("strength", "PolynomA[1]")),
            ("BPMx", ("monitor", "x")),
        ],
    )
    def test_the_families_a_rule_decides_resolve_as_the_table_says(
        self, family: str, resolved: tuple
    ) -> None:
        token, _ = _stated_type(_real_export(self.FACILITY)[family])
        assert resolve_attype(token) == resolved

    def test_the_whole_export_decides_as_pinned(self) -> None:
        verdicts = _real_verdicts(self.FACILITY)
        asking = {
            name: verdict.slot.kind
            for name, verdict in verdicts.items()
            if verdict.slot is not None
        }
        assert asking == {
            "HCMCurrReference": "attype",
            "VCMCurrReference": "attype",
            "KickerAmp": "attype",
            "KickerDelay": "attype",
            "Septum": "attype",
        }
        for family in ("BDM", "CD"):
            assert verdicts[family].verdict == "latch"
            assert "takes no KickAngle" in verdicts[family].reason
        assert verdicts["HCM"].kind == "kick"
        assert verdicts["VCM"].kind == "kick"
        assert verdicts["RF"].kind == "rf"


class TestTheSecondFacility:
    """A storage ring whose dipoles are typed as sextupoles."""

    FACILITY = "nsls2"

    def test_it_asks_about_no_family_at_all(self) -> None:
        objects = _real_export(self.FACILITY)
        assert not [
            name
            for name, family in objects.items()
            if isinstance(family, dict) and attype_slot_opens(*_stated_type(family))
        ]

    def test_the_dipoles_would_read_as_sextupoles_if_the_table_saw_them_first(self) -> None:
        token, _ = _stated_type(_real_export(self.FACILITY)["BEND"])
        assert resolve_attype(token) == ("strength", "PolynomB[2]")

    def test_the_cavity_states_no_element_and_is_bound_all_the_same(self) -> None:
        objects = _real_export(self.FACILITY)
        token, indices = _stated_type(objects["RF"])
        assert resolve_attype(token) == ("rf", None)
        assert indices == []

    def test_the_skew_quadrupoles_state_an_angle_on_the_field_they_set(self) -> None:
        objects = _real_export(self.FACILITY)
        assert objects["SQ"]["Setpoint"]["PhysicsUnits"].strip().lower() == "rad"

    def test_the_whole_export_decides_as_pinned(self) -> None:
        verdicts = _real_verdicts(self.FACILITY)
        assert not [name for name, verdict in verdicts.items() if verdict.slot is not None]
        assert verdicts["BEND"].verdict == "latch"
        assert verdicts["BEND"].reason == "bend2gev is constant at this facility"
        assert verdicts["RF"].kind == "rf"
        skew = verdicts["SQ"]
        assert (skew.verdict, skew.element_field) == ("couple", "PolynomA[1]")
        assert "rad" in skew.reason
