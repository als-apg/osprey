"""Addressing the elements a coupled family drives, and naming them.

The Middle Layer addresses a lattice element by its one-based position in the
saved ring; the model addresses it by name, and a real deck names a hundred
elements ``DR``. These lanes pin the join between the two: the positions a
family states are read against the deck it was sampled over, every element a
family drives is named for the one family that owns it, and a device split
over several elements keeps the slot each piece was stated in.

The committed synthetic export is the fixture: it carries a strength, a pair
of skew and normal sextupoles sharing one element, a kick split over two
elements with a device missing its second, a monitor of each plane on one
element, a cavity and an energy candidate -- every ownership contest the rank
decides, against a deck that holds them. The cases it does not carry are
pinned against a copy of it altered in the one place the rule turns on.

The first facility's export is the ring nobody wrote for this test. Its 2.0
re-export is not committed, so the lane builds the deck its stated positions
imply and reads the type tokens and indices of the committed 1.0 export
against it; what it proves is the ownership arithmetic over a real family
list, not a rule the synthetic fixture already pins.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import at
import pytest

from osprey.services.mml.family import family_views
from osprey.services.mml.loaders.mat import load_lattice
from osprey.services.mml.mapping.schema import VAFamily
from osprey.services.mml.va.elements import (
    CARRIED_FIELDS,
    OWNER_RANK,
    Addressing,
    ServedMarker,
    address_elements,
)
from osprey.services.mml.va.verdicts import BuiltCavity, cavity_to_build, propose, resolve_attype

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "mml"
SYNTHETIC = FIXTURES / "synthetic"

#: The stem the synthetic export's files share, and the system it describes.
STEM = "quokka.sr"
SYSTEM = "SR"


def _document(name: str) -> dict:
    """One committed document, freshly read so a lane may alter its copy."""
    return json.loads((SYNTHETIC / name).read_text(encoding="utf-8"))


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
    views = {view.raw_name: view for view in family_views(SYSTEM, objects)}
    return propose(export, deck, views)


@pytest.fixture
def addressed(export: dict, deck, verdicts: dict) -> Addressing:
    """The synthetic export addressed against its own deck."""
    return address_elements(export, deck, verdicts)


def _stated(export: dict, family: str) -> dict:
    """The nominal block a family states its lattice positions under."""
    nominals = export["families"][family]["nominals"]
    return next(iter(nominals.values()))


def _names(binding) -> list[str]:
    """The element names one device is bound to, in slice order."""
    return [slice_.element for slice_ in binding.slices]


class TestTheSyntheticExport:
    """Every coupled family of the committed export, against its own deck."""

    def test_only_the_families_a_rule_couples_to_an_element_are_addressed(
        self, addressed: Addressing, verdicts: dict
    ) -> None:
        coupled = {name for name, verdict in verdicts.items() if verdict.verdict == "couple"}
        assert set(addressed.bindings) <= coupled
        assert set(addressed.bindings) == {
            "QF",
            "QD",
            "SF",
            "SQ",
            "HC",
            "VC",
            "BPMx",
            "BPMy",
            "BEND",
            "RF",
        }

    def test_a_latched_family_is_addressed_nowhere(self, addressed: Addressing) -> None:
        for family in ("BDM", "BSOFT", "IDGAP", "SEPTUM", "DCCT", "TUNE", "Version"):
            assert family not in addressed.bindings

    def test_one_binding_per_device_in_the_order_the_export_lists_them(
        self, addressed: Addressing, export: dict
    ) -> None:
        for family, bindings in addressed.bindings.items():
            if family == "RF":  # the cavity binds by class, over one device
                continue
            assert [binding.device for binding in bindings] == [
                tuple(row) for row in export["families"][family]["device_list"]
            ]

    def test_a_stated_position_addresses_the_element_one_before_it(
        self, addressed: Addressing, export: dict, deck
    ) -> None:
        for family, bindings in addressed.bindings.items():
            if family == "RF":
                continue
            stated = [
                position
                for row in _stated(export, family)["at_index"]
                for position in (row if isinstance(row, list) else [row])
                if position != "NaN"
            ]
            read = [slice_.position for binding in bindings for slice_ in binding.slices]
            assert read == [int(position) - 1 for position in stated]
            assert all(0 <= position < len(deck) for position in read)

    def test_a_quadrupole_names_the_field_it_writes_and_the_element_it_writes_it_on(
        self, addressed: Addressing
    ) -> None:
        first = addressed.bindings["QF"][0]
        assert (first.kind, first.attribute, first.index) == ("strength", "PolynomB", 1)
        assert (first.element, first.owner, first.device) == ("QF_1_1", "QF", (1, 1))
        assert [slice_.position for slice_ in first.slices] == [1]

    def test_a_monitor_names_the_axis_it_reads_and_no_component_of_it(
        self, addressed: Addressing
    ) -> None:
        assert [(binding.attribute, binding.index) for binding in addressed.bindings["BPMx"]] == [
            ("x", None)
        ] * 4
        assert [(binding.attribute, binding.index) for binding in addressed.bindings["BPMy"]] == [
            ("y", None)
        ] * 4

    def test_the_cavity_is_bound_by_class_whatever_position_the_export_states(
        self, addressed: Addressing, deck
    ) -> None:
        cavities = [
            index for index, element in enumerate(deck) if type(element).__name__ == "RFCavity"
        ]
        (binding,) = addressed.bindings["RF"]
        assert (binding.kind, binding.attribute, binding.index) == ("rf", "Frequency", None)
        assert [slice_.position for slice_ in binding.slices] == cavities
        assert binding.element == "RF_1_1"

    def test_the_energy_candidate_addresses_the_dipoles_it_states(
        self, addressed: Addressing
    ) -> None:
        bindings = addressed.bindings["BEND"]
        assert [binding.kind for binding in bindings] == ["energy"] * 4
        assert [binding.attribute for binding in bindings] == [None] * 4
        assert [binding.element for binding in bindings] == [
            "BEND_1_1",
            "BEND_2_1",
            "BEND_3_1",
            "BEND_4_1",
        ]


class TestWhoOwnsAnElement:
    """The one family a shared element is named after, by the rank."""

    def test_the_rank_is_the_one_the_owner_rule_states(self) -> None:
        assert OWNER_RANK == ("monitor", "PolynomB", "PolynomA", "KickAngle", "energy", "rf")

    def test_a_normal_sextupole_outranks_the_skew_that_shares_its_element(
        self, addressed: Addressing
    ) -> None:
        shared = {
            slice_.position for binding in addressed.bindings["SF"] for slice_ in binding.slices
        }
        assert shared == {
            slice_.position for binding in addressed.bindings["SQ"] for slice_ in binding.slices
        }
        assert all(addressed.owners[position] == "SF" for position in shared)
        assert [binding.element for binding in addressed.bindings["SQ"]] == [
            "SF_1_1",
            "SF_2_1",
            "SF_3_1",
            "SF_4_1",
        ]

    def test_the_skew_still_writes_its_own_field_on_the_element_it_does_not_own(
        self, addressed: Addressing
    ) -> None:
        skew = addressed.bindings["SQ"][0]
        assert (skew.attribute, skew.index, skew.owner) == ("PolynomA", 1, "SF")

    def test_two_correctors_of_one_element_are_named_for_the_first_by_name(
        self, addressed: Addressing
    ) -> None:
        assert _names(addressed.bindings["HC"][0]) == ["HC_1_1_1", "HC_1_1_2"]
        assert _names(addressed.bindings["VC"][0]) == ["HC_1_1_1", "HC_1_1_2"]
        assert addressed.bindings["VC"][0].owner == "HC"

    def test_a_monitor_outranks_every_field_of_the_element_it_reads(
        self, addressed: Addressing
    ) -> None:
        assert [binding.element for binding in addressed.bindings["BPMy"]] == [
            binding.element for binding in addressed.bindings["BPMx"]
        ]
        assert all(owner != "BPMy" for owner in addressed.owners.values())

    def test_an_element_no_coupled_family_binds_has_no_owner(
        self, addressed: Addressing, deck
    ) -> None:
        drifts = {index for index, element in enumerate(deck) if element.FamName == "DR"}
        assert not drifts & set(addressed.owners)

    def test_the_rank_decides_before_the_name_does(
        self, export: dict, deck, verdicts: dict
    ) -> None:
        # Renaming the skew family so it sorts first leaves the rank in charge.
        moved = copy.deepcopy(verdicts)
        moved["AAA"] = moved.pop("SQ")
        block = copy.deepcopy(export)
        block["families"]["AAA"] = block["families"].pop("SQ")
        assert address_elements(block, deck, moved).bindings["AAA"][0].owner == "SF"


class TestTheNamesItEmits:
    """The name a bound element carries in the ring the lane returns."""

    def test_a_device_on_one_element_is_named_for_its_sector_and_number(
        self, addressed: Addressing
    ) -> None:
        assert [binding.element for binding in addressed.bindings["QD"]] == [
            "QD_1_1",
            "QD_2_1",
            "QD_3_1",
            "QD_4_1",
        ]

    def test_a_split_device_numbers_its_pieces_by_the_slot_each_was_stated_in(
        self, addressed: Addressing
    ) -> None:
        assert [_names(binding) for binding in addressed.bindings["HC"]] == [
            ["HC_1_1_1", "HC_1_1_2"],
            ["HC_2_1_1", "HC_2_1_2"],
            ["HC_3_1_1", "HC_3_1_2"],
            ["HC_4_1_1"],
        ]

    def test_a_slot_the_device_does_not_have_is_skipped_and_still_counted(
        self, export: dict, deck, verdicts: dict
    ) -> None:
        block = copy.deepcopy(export)
        _stated(block, "HC")["at_index"] = [[9, "NaN", 10], [19, 20], [29, 30], [39, "NaN"]]
        addressed = address_elements(block, deck, verdicts)
        first = addressed.bindings["HC"][0]
        assert _names(first) == ["HC_1_1_1", "HC_1_1_3"]
        assert [slice_.slot for slice_ in first.slices] == [1, 3]
        assert [slice_.position for slice_ in first.slices] == [8, 9]

    def test_the_ring_it_returns_carries_the_names_the_bindings_use(
        self, addressed: Addressing
    ) -> None:
        for bindings in addressed.bindings.values():
            for binding in bindings:
                for slice_ in binding.slices:
                    assert addressed.ring[slice_.position].FamName == slice_.element

    def test_the_deck_it_was_given_is_left_as_it_was(self, addressed: Addressing, deck) -> None:
        assert addressed.ring is not deck
        assert [element.FamName for element in deck][:5] == [
            "quokka_sr",
            "QF1",
            "DR",
            "BPM1",
            "BD1",
        ]
        assert len(addressed.ring) == len(deck)
        assert addressed.ring.energy == deck.energy

    def test_an_element_no_family_binds_keeps_the_name_the_deck_gave_it(
        self, addressed: Addressing, deck
    ) -> None:
        untouched = [index for index, _ in enumerate(deck) if index not in addressed.owners]
        assert [addressed.ring[index].FamName for index in untouched] == [
            deck[index].FamName for index in untouched
        ]


class TestTheMarkersItConverts:
    """A beam monitor the deck saved as a marker reads nothing until it is one."""

    def _deck(self, deck, element):
        """The synthetic deck with one element put in place of its first monitor."""
        ring = copy.deepcopy(deck)
        ring[3] = element
        return ring

    def test_a_marker_a_monitor_family_reads_becomes_a_monitor(
        self, export: dict, deck, verdicts: dict
    ) -> None:
        ring = self._deck(deck, at.Marker("BPM1"))
        addressed = address_elements(export, ring, verdicts)
        converted = addressed.ring[3]
        assert isinstance(converted, at.Monitor)
        assert (converted.FamName, converted.Length) == ("BPMx_1_1", 0.0)

    def test_a_monitor_the_deck_already_carries_is_only_renamed(
        self, addressed: Addressing
    ) -> None:
        assert isinstance(addressed.ring[3], at.Monitor)
        assert addressed.ring[3].FamName == "BPMx_1_1"

    def test_an_element_with_a_length_is_never_shortened_into_a_monitor(
        self, export: dict, deck, verdicts: dict
    ) -> None:
        ring = self._deck(deck, at.Quadrupole("QX", 0.3, 0.5))
        addressed = address_elements(export, ring, verdicts)
        kept = addressed.ring[3]
        assert isinstance(kept, at.Quadrupole)
        assert (kept.FamName, kept.Length) == ("BPMx_1_1", 0.3)

    def test_nothing_no_monitor_family_reads_is_converted(
        self, addressed: Addressing, deck
    ) -> None:
        classes = [
            type(element).__name__
            for index, element in enumerate(addressed.ring)
            if index != 3 and not isinstance(deck[index], at.Monitor)
        ]
        assert "Monitor" not in classes


class TestTheMonitorsItServesAsMarkers:
    """A monitor nothing reads, under a name the deck repeats, reads nothing.

    A facility marks the structure of its ring -- girder ends, straights --
    with the monitor type and one name for all of them, and the model refuses
    a deck that names two monitors alike because a reading is addressed by the
    name it is read from. Those elements carry no reading to address, so the
    served deck carries them as markers of the same name.
    """

    def _deck(self, deck, *names: str) -> tuple[Any, dict[int, str]]:
        """The synthetic deck with a monitor of each name where nothing is bound.

        The drifts are what no family of this export reaches, so putting the
        elements there leaves every stated position where the export states
        it.
        """
        ring = copy.deepcopy(deck)
        free = [index for index, element in enumerate(ring) if isinstance(element, at.Drift)]
        assert len(free) >= len(names), "the deck holds too few unbound elements"
        placed = dict(zip(free, names, strict=False))
        for index, name in placed.items():
            ring[index] = at.Monitor(name)
        return ring, placed

    def test_a_repeated_name_no_family_reads_becomes_a_plain_marker(
        self, export: dict, deck, verdicts: dict
    ) -> None:
        ring, placed = self._deck(deck, "GE", "GE")

        addressed = address_elements(export, ring, verdicts)

        for index in placed:
            served = addressed.ring[index]
            assert isinstance(served, at.Marker)
            assert (served.FamName, served.Length) == ("GE", 0.0)

    def test_every_other_element_of_the_deck_keeps_its_class(
        self, export: dict, deck, verdicts: dict
    ) -> None:
        ring, placed = self._deck(deck, "GE", "GE")

        addressed = address_elements(export, ring, verdicts)

        untouched = [index for index in range(len(ring)) if index not in placed]
        assert [type(addressed.ring[index]).__name__ for index in untouched] == [
            type(ring[index]).__name__ for index in untouched
        ]

    def test_a_unique_name_no_family_reads_stays_a_monitor(
        self, export: dict, deck, verdicts: dict
    ) -> None:
        """One name addresses one reading, whether or not anything reads it."""
        ring, placed = self._deck(deck, "MK4G1C30A")

        addressed = address_elements(export, ring, verdicts)

        served = addressed.ring[next(iter(placed))]
        assert isinstance(served, at.Monitor)
        assert served.FamName == "MK4G1C30A"
        assert addressed.markers == ()

    def test_a_monitor_a_family_reads_is_never_converted(
        self, export: dict, deck, verdicts: dict
    ) -> None:
        """The deck's name for it is gone by then: it is named for its device."""
        ring, placed = self._deck(deck, "BPM1", "BPM1")

        addressed = address_elements(export, ring, verdicts)

        read = addressed.ring[3]
        assert isinstance(read, at.Monitor)
        assert read.FamName == "BPMx_1_1"
        assert [type(addressed.ring[index]).__name__ for index in placed] == ["Marker", "Marker"]
        assert addressed.markers == (ServedMarker(name="BPM1", elements=2),)

    def test_it_names_every_converted_name_with_its_count(
        self, export: dict, deck, verdicts: dict
    ) -> None:
        ring, _ = self._deck(deck, "GS", "GE", "GS", "GE", "GS")

        addressed = address_elements(export, ring, verdicts)

        assert addressed.markers == (
            ServedMarker(name="GE", elements=2),
            ServedMarker(name="GS", elements=3),
        )

    def test_the_served_ring_is_exactly_as_long_as_the_deck(
        self, export: dict, deck, verdicts: dict
    ) -> None:
        """A marker occupies nothing, so the conversion may only take nothing."""
        ring, _ = self._deck(deck, "GS", "GE", "GS", "GE", "GS")

        addressed = address_elements(export, ring, verdicts)

        assert sum(element.Length for element in addressed.ring) == sum(
            element.Length for element in ring
        )

    def test_a_repeated_monitor_carrying_a_length_is_refused_by_name(
        self, export: dict, deck, verdicts: dict
    ) -> None:
        """Serving it as a marker would take half a metre out of the ring."""
        ring, placed = self._deck(deck, "GE", "GE")
        index = max(placed)
        ring[index] = at.Monitor("GE", Length=0.5)

        with pytest.raises(ValueError, match=r"2 monitor-type elements named 'GE'.*0.5 m long"):
            address_elements(export, ring, verdicts)

    def test_a_repeated_monitor_that_acts_on_the_beam_is_refused_by_name(
        self, export: dict, deck, verdicts: dict
    ) -> None:
        """A marker passes the beam through untouched, and so must its twin."""
        ring, placed = self._deck(deck, "GE", "GE")
        ring[max(placed)].PassMethod = "DriftPass"

        with pytest.raises(ValueError, match=r"named 'GE'.*passes the beam as 'DriftPass'"):
            address_elements(export, ring, verdicts)

    def test_a_repeated_name_that_is_no_monitor_is_left_alone(
        self, addressed: Addressing, deck
    ) -> None:
        """The deck calls a dozen drifts ``DR`` and nothing addresses any of them."""
        assert addressed.markers == ()
        assert sum(1 for element in deck if element.FamName == "DR") > 1


class TestWhatItRefuses:
    """The exports it will not address, each named by what is wrong with it."""

    def test_a_position_past_the_end_of_the_ring(self, export: dict, deck, verdicts: dict) -> None:
        block = copy.deepcopy(export)
        _stated(block, "BEND")["at_index"] = [5, 15, 25, 999]
        with pytest.raises(ValueError) as refusal:
            address_elements(block, deck, verdicts)
        assert "BEND" in str(refusal.value)
        assert "999" in str(refusal.value)
        assert str(len(deck)) in str(refusal.value)

    def test_a_position_at_or_below_the_start_of_the_ring(
        self, export: dict, deck, verdicts: dict
    ) -> None:
        block = copy.deepcopy(export)
        _stated(block, "BEND")["at_index"] = [0, 15, 25, 35]
        with pytest.raises(ValueError, match="BEND"):
            address_elements(block, deck, verdicts)

    def test_a_family_that_binds_elements_and_lists_no_device(
        self, export: dict, deck, verdicts: dict
    ) -> None:
        block = copy.deepcopy(export)
        block["families"]["QF"]["device_list"] = []
        with pytest.raises(ValueError) as refusal:
            address_elements(block, deck, verdicts)
        assert "QF" in str(refusal.value)
        assert "device" in str(refusal.value)

    def test_a_family_stating_more_element_rows_than_it_has_devices(
        self, export: dict, deck, verdicts: dict
    ) -> None:
        block = copy.deepcopy(export)
        block["families"]["QF"]["device_list"] = [[1, 1], [2, 1]]
        with pytest.raises(ValueError) as refusal:
            address_elements(block, deck, verdicts)
        assert "QF" in str(refusal.value)
        assert "4" in str(refusal.value)

    def test_two_elements_the_renaming_would_give_one_name(
        self, export: dict, deck, verdicts: dict
    ) -> None:
        block = copy.deepcopy(export)
        block["families"]["QF"]["device_list"] = [[1, 1], [1, 1], [3, 1], [4, 1]]
        with pytest.raises(ValueError) as refusal:
            address_elements(block, deck, verdicts)
        assert "QF_1_1" in str(refusal.value)

    def test_a_name_the_deck_already_carries_elsewhere(
        self, export: dict, deck, verdicts: dict
    ) -> None:
        ring = copy.deepcopy(deck)
        ring[7].FamName = "QF_1_1"  # a drift the export binds nowhere
        with pytest.raises(ValueError) as refusal:
            address_elements(export, ring, verdicts)
        assert "QF_1_1" in str(refusal.value)

    def test_an_export_with_no_families_at_all(self, deck) -> None:
        addressed = address_elements({"lattice": {}}, deck, {})
        assert addressed.bindings == {}
        assert addressed.owners == {}
        assert [element.FamName for element in addressed.ring] == [
            element.FamName for element in deck
        ]


def _standin_ring(length: int, monitors: set[int], correctors: set[int]) -> list:
    """A deck of the shape a facility's stated positions imply.

    The first facility's 2.0 re-export is not committed, so its lattice is
    not either. What the ownership arithmetic reads off a deck is its length
    and the class of each bound element, so the lane builds exactly that.
    """
    ring: list[Any] = []
    for position in range(1, length + 1):
        if position in monitors:
            ring.append(at.Marker(f"BPM{position}"))
        elif position in correctors:
            ring.append(at.Corrector(f"COR{position}", 0.0, [0.0, 0.0]))
        else:
            ring.append(at.Drift("DR", 0.1))
    return ring


class TestTheFirstFacility:
    """The ownership arithmetic over a family list nobody wrote for this test."""

    FACILITY = "spear3"
    EXPORT = "spear3.storagering"

    #: The families whose shared elements the addressing is pinned over.
    FAMILIES = ("HCM", "VCM", "BPMx", "BPMy")

    @pytest.fixture(scope="class")
    def objects(self) -> dict:
        """The committed accelerator objects of the facility's storage ring."""
        path = FIXTURES / self.FACILITY / f"{self.EXPORT}.ao.json"
        return json.loads(path.read_text(encoding="utf-8"))

    @pytest.fixture(scope="class")
    def rows(self, objects: dict) -> dict:
        """The positions each pinned family states, as the export states them."""
        return {family: objects[family]["AT"]["ATIndex"] for family in self.FAMILIES}

    @pytest.fixture(scope="class")
    def block(self, objects: dict, rows: dict) -> dict:
        """The export block the stated types and positions amount to."""
        families = {}
        for family in self.FAMILIES:
            kind, _ = resolve_attype(objects[family]["AT"]["ATType"])
            field = "Monitor" if kind == "monitor" else "Setpoint"
            families[family] = {
                "device_list": objects[family]["DeviceList"],
                "nominals": {
                    field: {
                        "at_type": objects[family]["AT"]["ATType"],
                        "at_index": rows[family],
                    }
                },
            }
        return {"families": families}

    @pytest.fixture(scope="class")
    def verdicts(self, objects: dict) -> dict:
        """What the rules make of those families, as task 2.3 pins them."""
        decided = {}
        for family in self.FAMILIES:
            kind, element_field = resolve_attype(objects[family]["AT"]["ATType"])
            decided[family] = VAFamily(
                verdict="couple",
                kind=kind,
                element_field=element_field,
                nominal_source="Monitor" if kind == "monitor" else "Setpoint",
            )
        return decided

    @pytest.fixture(scope="class")
    def addressed(self, block: dict, rows: dict, verdicts: dict) -> Addressing:
        """The facility's correctors and monitors addressed against that deck."""
        monitors = {int(position) for position in rows["BPMx"]}
        correctors = {int(position) for position in rows["HCM"]}
        length = max(int(position) for stated in rows.values() for position in stated)
        ring = _standin_ring(length, monitors, correctors)
        return address_elements(block, ring, verdicts)

    def test_both_corrector_planes_drive_the_same_elements(self, rows: dict) -> None:
        # 78, not the 72 the plan's gate text estimated: the committed export
        # states 78 corrector positions per plane, every one of them shared and
        # every device row distinct. The monitor count, 117, is as estimated.
        assert len(rows["HCM"]) == len(rows["VCM"]) == 78
        assert set(rows["HCM"]) == set(rows["VCM"])

    def test_every_shared_corrector_element_is_owned_by_the_horizontal_plane(
        self, addressed: Addressing, rows: dict
    ) -> None:
        shared = {int(position) - 1 for position in rows["HCM"]}
        assert len(shared) == 78
        assert {addressed.owners[position] for position in shared} == {"HCM"}
        assert [binding.element for binding in addressed.bindings["VCM"]] == [
            binding.element for binding in addressed.bindings["HCM"]
        ]

    def test_every_shared_monitor_element_is_owned_by_the_horizontal_plane(
        self, addressed: Addressing, rows: dict
    ) -> None:
        shared = {int(position) - 1 for position in rows["BPMx"]}
        assert len(shared) == 117
        assert {addressed.owners[position] for position in shared} == {"BPMx"}
        assert [binding.element for binding in addressed.bindings["BPMy"]] == [
            binding.element for binding in addressed.bindings["BPMx"]
        ]

    def test_the_elements_it_names_are_named_for_the_owner_and_its_device(
        self, addressed: Addressing, objects: dict
    ) -> None:
        devices = objects["HCM"]["DeviceList"]
        assert [binding.element for binding in addressed.bindings["HCM"]][:3] == [
            f"HCM_{sector}_{number}" for sector, number in devices[:3]
        ]

    def test_every_monitor_marker_it_names_is_a_monitor_in_the_ring_it_returns(
        self, addressed: Addressing, rows: dict
    ) -> None:
        read = [addressed.ring[int(position) - 1] for position in rows["BPMx"]]
        assert all(isinstance(element, at.Monitor) for element in read)
        assert len({element.FamName for element in read}) == 117


#: How many buckets the synthetic ring is given, and the volts a reviewer
#: answered with. The ring is 23.2 m round, so 40 of them is about 517 MHz.
BUILT_HARMONIC = 40
BUILT_VOLTAGE = 3.0e6


def _without_a_cavity(deck) -> Any:
    """The synthetic deck with its cavity taken out, as a facility exports one.

    A facility whose Middle Layer holds the radio frequency never saves the
    cavity, so the deck arrives one element short and the ring solves at fixed
    energy until the emit lane builds one.
    """
    ring = copy.deepcopy(deck)
    del ring[-1]
    assert not [element for element in ring if isinstance(element, at.RFCavity)]
    return ring


class TestTheCavityItBuilds:
    """A deck that carries no cavity is served one built from the export."""

    @pytest.fixture
    def cavity_less(self, export: dict, deck, objects: dict):
        """The cavity-less deck, the verdicts over it, and the cavity to build."""
        ring = _without_a_cavity(deck)
        export["families"]["RF"]["nominals"]["Setpoint"]["at_index"] = []
        views = {view.raw_name: view for view in family_views(SYSTEM, objects)}
        ad = {"HarmonicNumber": BUILT_HARMONIC}
        built = cavity_to_build(export, ring, ad, voltage=BUILT_VOLTAGE)
        return ring, propose(export, ring, views, ad), built

    def test_the_built_cavity_is_the_last_element_and_occupies_no_space(
        self, export: dict, cavity_less
    ) -> None:
        ring, verdicts, built = cavity_less

        addressed = address_elements(export, ring, verdicts, cavity=built)

        cavity = addressed.ring[-1]
        assert isinstance(cavity, at.RFCavity)
        assert len(addressed.ring) == len(ring) + 1
        assert (cavity.Length, cavity.PassMethod) == (0.0, "RFCavityPass")

    def test_it_carries_the_harmonic_the_voltage_and_the_deck_energy(
        self, export: dict, cavity_less
    ) -> None:
        ring, verdicts, built = cavity_less

        cavity = address_elements(export, ring, verdicts, cavity=built).ring[-1]

        assert cavity.HarmNumber == BUILT_HARMONIC
        assert cavity.Voltage == BUILT_VOLTAGE
        assert cavity.Energy == ring.energy

    def test_it_is_built_on_the_harmonic_rather_than_at_the_stated_frequency(
        self, export: dict, cavity_less
    ) -> None:
        """The stated frequency is the real ring's, to the figures it is quoted to.

        Reading it onto a deck of a slightly different circumference starts
        the beam off momentum, which is the error a cavity is built to remove.
        """
        ring, verdicts, built = cavity_less

        addressed = address_elements(export, ring, verdicts, cavity=built)

        cavity = addressed.ring[-1]
        assert cavity.Frequency == pytest.approx(
            BUILT_HARMONIC * addressed.ring.revolution_frequency
        )
        assert addressed.cavity.frequency_hz == cavity.Frequency
        assert addressed.cavity.nominal_hz == pytest.approx(built.nominal_hz)

    def test_the_served_ring_solves_six_dimensionally(self, export: dict, cavity_less) -> None:
        """The deck as it arrives cannot close an orbit through a bucket at all."""
        ring, verdicts, built = cavity_less
        with pytest.raises(at.AtError):
            copy.deepcopy(ring).find_orbit6()

        addressed = address_elements(export, ring, verdicts, cavity=built)

        assert addressed.ring.is_6d
        orbit, _ = addressed.ring.find_orbit6()
        assert len(orbit) == 6

    def test_the_family_binds_it_as_it_binds_a_cavity_the_deck_carried(
        self, export: dict, cavity_less
    ) -> None:
        ring, verdicts, built = cavity_less

        addressed = address_elements(export, ring, verdicts, cavity=built)

        bound = addressed.bindings["RF"]
        assert len(bound) == 1
        assert bound[0].attribute == "Frequency"
        assert bound[0].slices[0].position == len(addressed.ring) - 1
        assert addressed.ring[-1].FamName == bound[0].element

    def test_the_deck_it_was_given_is_left_as_it_was(self, export: dict, cavity_less) -> None:
        ring, verdicts, built = cavity_less
        before = len(ring)

        address_elements(export, ring, verdicts, cavity=built)

        assert len(ring) == before
        assert not [element for element in ring if isinstance(element, at.RFCavity)]

    def test_every_stated_position_still_points_where_it_did(
        self, export: dict, cavity_less
    ) -> None:
        """The cavity goes on the end, so no index the export states moves."""
        ring, verdicts, built = cavity_less

        addressed = address_elements(export, ring, verdicts, cavity=built)

        for family, bound in addressed.bindings.items():
            if family == "RF":
                continue
            for device in bound:
                for piece in device.slices:
                    assert piece.position < len(ring)

    def test_a_cavity_with_no_answered_voltage_is_refused(self, export: dict, cavity_less) -> None:
        ring, verdicts, _ = cavity_less
        unanswered = BuiltCavity(family="RF", nominal_hz=5.0e8, harmonic=BUILT_HARMONIC)

        with pytest.raises(ValueError, match="no voltage"):
            address_elements(export, ring, verdicts, cavity=unanswered)

    def test_a_deck_saved_as_one_period_of_a_ring_is_refused_a_cavity(
        self, export: dict, cavity_less
    ) -> None:
        """A period is not a ring, and the facts the cavity is built from are the ring's.

        pyAT reads a cavity's harmonic number as the count per period and
        multiplies it by the lattice periodicity, so the whole-ring harmonic
        the accelerator data states, built into one period of four, is a
        cavity at four times the right frequency. The element positions the
        export indexes are the whole ring's too, so the deck is refused by
        name rather than served a cavity nobody could read.
        """
        ring, verdicts, built = cavity_less
        periodic = copy.deepcopy(ring)
        periodic.periodicity = 4

        with pytest.raises(ValueError, match="saved as 4 periods of the ring"):
            address_elements(export, periodic, verdicts, cavity=built)

    def test_a_deck_saved_as_the_whole_ring_is_served_one(self, export: dict, cavity_less) -> None:
        ring, verdicts, built = cavity_less

        assert ring.periodicity == 1
        assert isinstance(
            address_elements(export, ring, verdicts, cavity=built).ring[-1], at.RFCavity
        )

    def test_a_deck_given_no_cavity_is_addressed_as_before(
        self, export: dict, deck, verdicts: dict, addressed: Addressing
    ) -> None:
        again = address_elements(export, deck, verdicts, cavity=None)

        assert again.cavity is None
        assert len(again.ring) == len(addressed.ring)


#: An aperture and a transformation a deck may put on a zero-length element.
#: Neither is implied by the marker class or the monitor class, and both say
#: where the element stands rather than what it reads.
APERTURES = [1.0e-3, 2.0e-3]
DISPLACEMENT = [1.0e-4, 0.0, 2.0e-4, 0.0, 0.0, 0.0]


class TestWhatAConvertedElementKeeps:
    """A conversion between the marker class and the monitor class moves the class.

    An aperture is where the beam is lost and a transformation is where the
    element stands. Both belong to the position, which the conversion does not
    touch, so an element that loses them on the way through has been moved to
    make a name work.
    """

    def _placed(self, deck, element, at_index: int):
        """The synthetic deck with one element put where nothing is bound."""
        ring = copy.deepcopy(deck)
        ring[at_index] = element
        return ring

    def test_a_monitor_served_as_a_marker_keeps_its_apertures(
        self, export: dict, deck, verdicts: dict
    ) -> None:
        free = [index for index, element in enumerate(deck) if isinstance(element, at.Drift)][:2]
        ring = copy.deepcopy(deck)
        for index in free:
            ring[index] = at.Monitor("GE", EApertures=APERTURES, T1=DISPLACEMENT)

        addressed = address_elements(export, ring, verdicts)

        for index in free:
            served = addressed.ring[index]
            assert isinstance(served, at.Marker)
            assert list(served.EApertures) == APERTURES
            assert list(served.T1) == DISPLACEMENT

    def test_a_marker_read_as_a_monitor_keeps_its_apertures(
        self, export: dict, deck, verdicts: dict
    ) -> None:
        ring = self._placed(deck, at.Marker("BPM1", EApertures=APERTURES, T1=DISPLACEMENT), 3)

        addressed = address_elements(export, ring, verdicts)

        served = addressed.ring[3]
        assert isinstance(served, at.Monitor)
        assert list(served.EApertures) == APERTURES
        assert list(served.T1) == DISPLACEMENT

    def test_a_field_the_deck_never_stated_stays_unstated(
        self, export: dict, deck, verdicts: dict
    ) -> None:
        """A class carries these only where the deck did; an absent one is absent."""
        free = [index for index, element in enumerate(deck) if isinstance(element, at.Drift)][:2]
        ring = copy.deepcopy(deck)
        for index in free:
            ring[index] = at.Monitor("GE")

        addressed = address_elements(export, ring, verdicts)

        served = addressed.ring[free[0]]
        assert isinstance(served, at.Marker)
        assert not [field for field in CARRIED_FIELDS if hasattr(served, field)]
