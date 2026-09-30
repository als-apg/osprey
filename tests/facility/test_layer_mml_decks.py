"""Tests for the mml layer's deck pass (``osprey.facility.layers.mml.decks``).

The layer is driven over the new-format spear3 and nsls2 mappings and the
decks their exports saved, and over small decks built here for the rules a
fixture does not reach.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from osprey.facility.layers.mml.decks import Addressing, address_elements
from osprey.facility.layers.mml.mapping import EngineBlock, Model, WiringFamily, read_mapping

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
    return address_elements(model, deck, va)


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
    addressing = address_elements(model, deck, va)
    first = addressing.bindings["BPMx"][0]
    assert (first.device, first.element, first.owner) == ((1, 1), "BPMx_1_1", "BPMx")


def test_the_horizontal_corrector_names_the_magnet_both_planes_drive() -> None:
    model, deck, va, ad = _fixture(*STORAGE[0])
    addressing = address_elements(model, deck, va)
    horizontal, vertical = addressing.bindings["HCM"][0], addressing.bindings["VCM"][0]
    assert horizontal.element == vertical.element == "HCM_1_1"
    assert vertical.owner == "HCM"
    assert vertical.engine == EngineBlock(attribute="KickAngle", index=1)


def test_the_deck_passed_in_is_left_as_it_was() -> None:
    model, deck, va, ad = _fixture(*STORAGE[0])
    before = [element.FamName for element in deck]
    address_elements(model, deck, va)
    assert [element.FamName for element in deck] == before


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
