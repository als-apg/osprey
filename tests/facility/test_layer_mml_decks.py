"""Tests for the mml layer's deck pass (``osprey.facility.layers.mml.decks``).

The layer is driven over the new-format spear3 and nsls2 mappings and the
decks their exports saved, and over small decks built here for the rules a
fixture does not reach.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from osprey.facility.layers.mml.decks import (
    DECKS_DIR,
    Addressing,
    address_elements,
    served_deck,
    write_deck,
)
from osprey.facility.layers.mml.mapping import (
    EngineBlock,
    ImportStop,
    MappingError,
    Model,
    WiringFamily,
    read_mapping,
)

at = pytest.importorskip("at")

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "mml"

#: (tree, export stem, raw system token) of each imported storage deck.
STORAGE = (
    ("spear3", "spear3.storagering", "StorageRing"),
    ("nsls2", "nsls2.storagering", "StorageRing"),
)


def _fixture(tree: str, stem: str, raw: str) -> tuple[Model, Any, dict, dict]:
    from osprey.services.mml.loaders.mat import load_lattice

    mapping = read_mapping(FIXTURES / tree / "imported" / "mml" / "mapping.yaml")
    deck = load_lattice(FIXTURES / tree / f"{stem}.lattice.mat")
    va = json.loads((FIXTURES / tree / f"{stem}.va.json").read_text(encoding="utf-8"))
    ad = json.loads((FIXTURES / tree / f"{stem}.ad.json").read_text(encoding="utf-8"))
    return mapping.models[raw], deck, va, ad


@pytest.fixture(scope="module", params=STORAGE, ids=[tree for tree, _, _ in STORAGE])
def addressed(request: pytest.FixtureRequest) -> Addressing:
    model, deck, va, ad = _fixture(*request.param)
    return address_elements(model, deck, va, ad)


def _wired(addressing: Addressing) -> set[str]:
    return {
        piece.element
        for bindings in addressing.bindings.values()
        for binding in bindings
        for piece in binding.slices
    }


# --- the fixture decks --------------------------------------------------------------


def test_every_wired_element_carries_a_unique_name(addressed: Addressing) -> None:
    names = Counter(element.FamName for element in addressed.deck)
    wired = _wired(addressed)
    assert wired
    assert sorted(name for name in wired if names[name] != 1) == []


def test_every_monitor_name_is_unique(addressed: Addressing) -> None:
    names = Counter(e.FamName for e in addressed.deck if isinstance(e, at.Monitor))
    assert addressed.monitors == sum(names.values()) > 0
    assert sorted(name for name, count in names.items() if count > 1) == []


def test_every_wired_family_the_export_places_binds_elements(addressed: Addressing) -> None:
    assert {"BPMx", "BPMy", "HCM", "VCM"} <= set(addressed.bindings)


def test_a_device_is_named_for_its_family_and_row() -> None:
    model, deck, va, ad = _fixture(*STORAGE[0])
    addressing = address_elements(model, deck, va, ad)
    first = addressing.bindings["BPMx"][0]
    assert (first.device, first.element, first.owner) == ((1, 1), "BPMx_1_1", "BPMx")


def test_the_horizontal_corrector_names_the_magnet_both_planes_drive() -> None:
    model, deck, va, ad = _fixture(*STORAGE[0])
    addressing = address_elements(model, deck, va, ad)
    horizontal, vertical = addressing.bindings["HCM"][0], addressing.bindings["VCM"][0]
    assert horizontal.element == vertical.element == "HCM_1_1"
    assert vertical.owner == "HCM"
    assert vertical.engine == EngineBlock(attribute="KickAngle", index=1)


def test_a_deck_holding_its_cavity_is_served_that_cavity() -> None:
    model, deck, va, ad = _fixture(*STORAGE[0])
    addressing = address_elements(model, deck, va, ad)
    assert addressing.cavity is None
    assert len(addressing.deck) == len(deck)
    (rf,) = addressing.bindings["RF"]
    assert (rf.element, rf.slices[0].position) == ("RF_1_1", 0)


def test_a_deck_without_a_cavity_is_built_one_from_the_answered_voltage() -> None:
    model, deck, va, ad = _fixture(*STORAGE[1])
    addressing = address_elements(model, deck, va, ad)
    built = addressing.deck
    cavity = built[-1]
    assert len(built) == len(deck) + 1
    assert isinstance(cavity, at.RFCavity)
    assert (cavity.FamName, cavity.Length, cavity.Voltage) == ("RF_1_1", 0.0, 3000000.0)
    assert cavity.HarmNumber == 1320
    revolution = at.clight / built.circumference
    assert cavity.Frequency == pytest.approx(1320 * revolution)
    assert addressing.cavity is not None
    assert (addressing.cavity.family, addressing.cavity.harmonic) == ("RF", 1320)
    assert addressing.cavity.frequency_hz == pytest.approx(cavity.Frequency)
    (rf,) = addressing.bindings["RF"]
    assert rf.slices[0].position == len(deck)


def test_a_cavity_to_build_without_a_voltage_stops_the_import() -> None:
    model, deck, va, ad = _fixture(*STORAGE[1])
    unanswered = replace(
        model, wiring={**model.wiring, "RF": replace(model.wiring["RF"], voltage=None)}
    )
    with pytest.raises(ImportStop) as caught:
        address_elements(unanswered, deck, va, ad)
    assert caught.value.lines == (
        "models.StorageRing.wiring.RF.voltage: answer the cavity voltage in volts; "
        "the deck holds no cavity",
    )


def test_a_voltage_on_a_deck_holding_its_cavity_is_refused() -> None:
    model, deck, va, ad = _fixture(*STORAGE[0])
    answered = replace(
        model, wiring={**model.wiring, "RF": replace(model.wiring["RF"], voltage=1.0)}
    )
    with pytest.raises(MappingError) as caught:
        address_elements(answered, deck, va, ad)
    assert str(caught.value) == (
        "models.StorageRing.wiring.RF.voltage: the deck holds a cavity; remove voltage"
    )


def test_a_cavity_is_not_built_without_a_harmonic_number() -> None:
    model, deck, va, _ad = _fixture(*STORAGE[1])
    with pytest.raises(ValueError, match="family RF drives a cavity the deck does not hold"):
        address_elements(model, deck, va, {"HarmonicNumber": []})


def test_a_cavity_is_not_built_into_one_period_of_a_deck() -> None:
    model, deck, va, ad = _fixture(*STORAGE[1])
    deck.periodicity = 2
    with pytest.raises(ValueError, match="saved as 2 periods"):
        address_elements(model, deck, va, ad)


def test_the_deck_passed_in_is_left_as_it_was() -> None:
    model, deck, va, ad = _fixture(*STORAGE[0])
    before = [element.FamName for element in deck]
    address_elements(model, deck, va, ad)
    assert [element.FamName for element in deck] == before


def test_the_served_deck_moves_in_six_dimensions(addressed: Addressing) -> None:
    served = served_deck(addressed)
    assert served.is_6d
    cavities = [e for e in served if isinstance(e, at.RFCavity)]
    assert cavities
    assert {cavity.PassMethod for cavity in cavities} == {"RFCavityPass"}


def test_every_wired_corrector_is_served_zeroed_polynomials(addressed: Addressing) -> None:
    served = served_deck(addressed)
    correctors = {
        piece.position
        for bindings in addressed.bindings.values()
        for binding in bindings
        if binding.engine.attribute == "KickAngle"
        for piece in binding.slices
        if addressed.owners[piece.position] == piece.owner == binding.family
    }
    assert correctors
    for position in correctors:
        element = served[position]
        width = int(element.MaxOrder) + 1
        for name in ("PolynomA", "PolynomB"):
            assert list(getattr(element, name)) == [0.0] * width, (position, name)


def test_serving_leaves_the_addressed_deck_as_it_was(addressed: Addressing) -> None:
    before = [(e.FamName, e.PassMethod) for e in addressed.deck]
    served_deck(addressed)
    assert [(e.FamName, e.PassMethod) for e in addressed.deck] == before


def test_the_written_deck_reads_back_served(addressed: Addressing, tmp_path: Path) -> None:
    path = write_deck(served_deck(addressed), tmp_path, "StorageRing")
    assert path == tmp_path / DECKS_DIR / "StorageRing.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    assert "at_version" not in document
    read = at.load_lattice(str(path))
    assert read.is_6d
    names = Counter(element.FamName for element in read)
    assert sorted(name for name in _wired(addressed) if names[name] != 1) == []


def test_writing_a_deck_twice_writes_the_same_bytes(tmp_path: Path) -> None:
    model, deck, va, ad = _fixture(*STORAGE[0])
    served = served_deck(address_elements(model, deck, va, ad))
    first = write_deck(served, tmp_path, "StorageRing").read_bytes()
    assert write_deck(served, tmp_path, "StorageRing").read_bytes() == first


# --- small decks --------------------------------------------------------------------


def _model(**wiring: EngineBlock) -> Model:
    return Model(
        raw="SR",
        name="SR",
        description=None,
        provenance="stated",
        wiring={
            family: WiringFamily(
                element_field="Monitor" if engine.axis else "Setpoint",
                engine=engine,
                calibration="linear",
            )
            for family, engine in wiring.items()
        },
    )


def _family(field: str, at_index: Any, devices: Any) -> dict[str, Any]:
    return {"device_list": devices, "nominals": {field: {"at_index": at_index}}}


def _deck(*elements: Any) -> Any:
    return at.Lattice(list(elements), energy=3e9, periodicity=1)


def test_a_normal_multipole_outranks_the_skew_one_wound_on_it() -> None:
    deck = _deck(at.Drift("D", 1.0), at.Sextupole("S", 0.2, 1.0), at.Drift("D", 1.0))
    va = {
        "families": {
            "SQ": _family("Setpoint", [2], [[1, 1]]),
            "SX": _family("Setpoint", [2], [[1, 1]]),
        }
    }
    model = _model(
        SQ=EngineBlock(attribute="PolynomA", index=1),
        SX=EngineBlock(attribute="PolynomB", index=2),
    )
    addressing = address_elements(model, deck, va)
    assert addressing.deck[1].FamName == "SX_1_1"
    assert addressing.owners == {1: "SX"}
    assert addressing.bindings["SQ"][0].element == "SX_1_1"


def test_a_monitor_outranks_every_magnet() -> None:
    deck = _deck(at.Drift("D", 1.0), at.Quadrupole("Q", 0.2, 1.0))
    va = {
        "families": {
            "Q": _family("Setpoint", [2], [[1, 1]]),
            "BPMx": _family("Monitor", [2], [[1, 1]]),
        }
    }
    model = _model(Q=EngineBlock(attribute="PolynomB", index=1), BPMx=EngineBlock(axis="x"))
    assert address_elements(model, deck, va).deck[1].FamName == "BPMx_1_1"


def test_a_marker_a_monitor_family_reads_becomes_a_monitor() -> None:
    deck = _deck(at.Drift("D", 1.0), at.Marker("M"))
    va = {"families": {"BPMx": _family("Monitor", [2], [[3, 7]])}}
    addressing = address_elements(_model(BPMx=EngineBlock(axis="x")), deck, va)
    assert isinstance(addressing.deck[1], at.Monitor)
    assert addressing.deck[1].FamName == "BPMx_3_7"


def test_a_split_device_names_each_piece_by_its_stated_slot() -> None:
    deck = _deck(at.Quadrupole("Q", 0.1, 1.0), at.Drift("D", 1.0), at.Quadrupole("Q", 0.1, 1.0))
    va = {"families": {"QF": _family("Setpoint", [[1, float("nan"), 3]], [[2, 1]])}}
    model = _model(QF=EngineBlock(attribute="PolynomB", index=1))
    (binding,) = address_elements(model, deck, va).bindings["QF"]
    assert [(piece.element, piece.slot) for piece in binding.slices] == [
        ("QF_2_1_1", 1),
        ("QF_2_1_3", 3),
    ]


def test_a_corrector_is_served_polynomials_as_wide_as_it_carries() -> None:
    corrector = at.Corrector("C", 0.1, [0.0, 0.0], PolynomB=[0.0, 0.5, 0.2], MaxOrder=1)
    deck = _deck(at.Drift("D", 1.0), corrector)
    va = {"families": {"HCM": _family("Setpoint", [2], [[1, 1]])}}
    model = _model(HCM=EngineBlock(attribute="KickAngle", index=0))
    served = served_deck(address_elements(model, deck, va))
    assert list(served[1].PolynomB) == [0.0, 0.0, 0.0]
    assert list(served[1].PolynomA) == [0.0, 0.0]


def test_a_corrector_winding_leaves_the_magnet_it_is_wound_on_alone() -> None:
    sextupole = at.Sextupole("S", 0.2, 1.5)
    deck = _deck(at.Drift("D", 1.0), sextupole)
    va = {
        "families": {
            "HCM": _family("Setpoint", [2], [[1, 1]]),
            "SX": _family("Setpoint", [2], [[1, 1]]),
        }
    }
    model = _model(
        HCM=EngineBlock(attribute="KickAngle", index=0),
        SX=EngineBlock(attribute="PolynomB", index=2),
    )
    served = served_deck(address_elements(model, deck, va))
    assert served[1].FamName == "SX_1_1"
    assert served[1].PolynomB[2] == 1.5


def test_a_repeated_monitor_nothing_reads_is_served_as_a_marker() -> None:
    deck = _deck(at.Monitor("G"), at.Drift("D", 1.0), at.Monitor("G"), at.Monitor("B"))
    va = {"families": {"BPMx": _family("Monitor", [4], [[1, 1]])}}
    addressing = address_elements(_model(BPMx=EngineBlock(axis="x")), deck, va)
    assert [type(e).__name__ for e in addressing.deck] == ["Marker", "Drift", "Marker", "Monitor"]
    assert [(m.name, m.elements) for m in addressing.markers] == [("G", 2)]
    assert addressing.monitors == 1


def test_a_family_the_export_does_not_place_binds_nothing() -> None:
    deck = _deck(at.Quadrupole("Q", 0.1, 1.0))
    model = _model(QF=EngineBlock(attribute="PolynomB", index=1))
    assert address_elements(model, deck, {"families": {}}).bindings == {}


@pytest.mark.parametrize(
    ("va", "engine", "message"),
    [
        (
            {"families": {"QF": _family("Setpoint", [5], [[1, 1]])}},
            EngineBlock(attribute="PolynomB", index=1),
            "family QF binds ATIndex 5 of a deck of 1 elements",
        ),
        (
            {"families": {"QF": _family("Setpoint", [1], None)}},
            EngineBlock(attribute="PolynomB", index=1),
            "family QF binds 1 element rows and lists no device to name them after",
        ),
        (
            {"families": {"QF": _family("Setpoint", [1], [[1, 1]])}},
            EngineBlock(attribute="K", index=1),
            "family QF drives K; wire it to an axis or to PolynomB, PolynomA, KickAngle "
            "or Frequency",
        ),
    ],
)
def test_what_the_deck_cannot_address_is_refused(
    va: dict[str, Any], engine: EngineBlock, message: str
) -> None:
    deck = _deck(at.Quadrupole("Q", 0.1, 1.0))
    with pytest.raises(ValueError, match=message):
        address_elements(_model(QF=engine), deck, va)


def test_two_elements_named_alike_are_refused() -> None:
    deck = _deck(at.Quadrupole("QF_1_1", 0.1, 1.0), at.Quadrupole("Q", 0.1, 1.0))
    va = {"families": {"QF": _family("Setpoint", [2], [[1, 1]])}}
    model = _model(QF=EngineBlock(attribute="PolynomB", index=1))
    with pytest.raises(ValueError, match="2 elements named 'QF_1_1', at positions 1, 2"):
        address_elements(model, deck, va)
