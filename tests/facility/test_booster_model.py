"""The example facility's booster ring ``BR``, from MML's booster lattice.

``scripts/facility_demo/_booster.py`` builds the booster from MML: the deck is
``Booster/boosterlattice.m`` element for element, solved as a periodic ring at
one energy, and the records are MML's families, 24 bends on one supply, 16 + 16
quadrupoles on a supply per family, 16 + 16 correctors and 20 monitors. The
ring closes and is stable. Each supply's golden current maps onto the deck's
MML strengths, a wired channel carries no seed nominal, place ``BR`` spans the
deck, and the measurement file lists the orbit response alone. The lattice is
read from an MML checkout when one is present and otherwise from the
checked-in excerpt, which must then agree with MML.
"""

from __future__ import annotations

import importlib.util
import json
import math
import shutil
import sys
from collections import Counter
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from osprey.facility.views.simulator import simulator_wiring
from osprey_connectors.simulation.engines import curve_from_record, to_hardware, to_physics
from osprey_facility import source_tree
from osprey_facility.build import build_facility
from osprey_facility.records import FacilityRecords
from osprey_facility.sources import read_yaml
from tests._builds import FRAMEWORK_CHECKS
from tests.facility.conftest import BuiltProject

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCES = REPO_ROOT / "scripts/facility_demo"
EXAMPLE = REPO_ROOT / "src/osprey/templates/facilities/example"
MODELS = source_tree.path("models")
SEEDS = source_tree.path("seeds")
DECKS = source_tree.path("decks")
MEASUREMENT = source_tree.keyed_file("measurement", "BR")
EXCERPT = Path(__file__).resolve().parent / "fixtures/mml/boosterlattice_excerpt.m"

#: MML's booster families: the devices of each (``Booster/boosterinit.m``).
FAMILY_COUNTS = {"DIPOLE": 24, "QF": 16, "QD": 16, "HCM": 16, "VCM": 16, "BPM": 20}

#: The deck's elements by MML token (``Booster/boosterlattice.m``).
TOKEN_COUNTS = {
    "BU": 24,
    "BD": 24,
    "QF": 32,
    "QD": 32,
    "BPM": 32,
    "HCM": 16,
    "VCM": 16,
    "SF": 8,
    "SD": 12,
    "TV": 3,
}

#: The supply device of each family on one supply.
SUPPLIES = {"DIPOLE": "BR/DIPOLE-RAMP01", "QF": "BR/QF-RAMP01", "QD": "BR/QD-RAMP01"}

#: The booster devices this model does not emit: the three supplies, the kicker, the two
#: septa, the five screens and the current monitor.
OTHER_DEVICES = 12


def _module(name: str, filename: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    # ``_booster`` imports its siblings by bare name, as generate.py does.
    spec = importlib.util.spec_from_file_location(name, SOURCES / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # import-time required because the ``needs_mml`` skip condition below reads the loaded module, and a mark is evaluated at import; scripts/ is not a package, so the
    # module is registered for its dataclasses and its siblings are found on a
    # sys.path entry held only while it executes.
    sys.modules[name] = module
    added = str(SOURCES) not in sys.path
    if added:
        sys.path.insert(0, str(SOURCES))
    try:
        spec.loader.exec_module(module)
    finally:
        if added:
            sys.path.remove(str(SOURCES))
    return module


@cache
def booster() -> ModuleType:
    return _module("facility_demo_booster", "_booster.py")


def _mml() -> ModuleType:
    return booster()._mml


def _mml_file() -> Path | None:
    root = _mml().MML_ROOT
    path = None if root is None else root / _mml().LATTICES["BR"][0]
    return path if path is not None and path.is_file() else None


needs_mml = pytest.mark.skipif(_mml_file() is None, reason="no MML checkout")


@cache
def _section(prefer_mml: bool = True) -> Any:
    path = _mml_file() if prefer_mml and _mml_file() is not None else EXCERPT
    return _mml().parse(path.read_text(encoding="latin-1"), "BR")


@cache
def _committed(rel: str) -> Any:
    return read_yaml((EXAMPLE / rel).read_text(encoding="utf-8"))


def _model() -> dict[str, Any]:
    (model,) = [model for model in _committed(MODELS) if model["name"] == "BR"]
    return model


@cache
def _deck() -> dict[str, Any]:
    return json.loads((EXAMPLE / DECKS / "BR.json").read_text(encoding="utf-8"))


def _deck_elements() -> list[dict[str, Any]]:
    return _deck()["elements"]


def _token(name: str) -> str:
    return name.rstrip("0123456789")


def _golden() -> dict[str, float]:
    """Each supply current address's golden current."""
    return {
        booster().supply_address(family, sub): current
        for family, current in booster().GOLDEN.items()
        for sub in ("SP", "RB")
    }


def _corrector_currents() -> list[str]:
    return [
        f"BR:MAG:{family}:{n:02d}:CURRENT:{sub}"
        for family in ("HCM", "VCM")
        for n in range(1, 17)
        for sub in ("SP", "RB")
    ]


def _positions(axes: str = "XY") -> list[str]:
    return [f"BR:DIAG:BPM:{n:02d}:POSITION:{axis}" for n in range(1, 21) for axis in axes]


def _ring() -> Any:
    at = pytest.importorskip("at")
    return at.load_lattice(str(EXAMPLE / DECKS / "BR.json"))


def _ring4() -> Any:
    ring = _ring()
    ring.disable_6d()
    return ring


@pytest.fixture(scope="module")
def built() -> dict[str, Any]:
    return build_facility(EXAMPLE, project_name="demo", extra_checks=FRAMEWORK_CHECKS)


# --- the deck is MML's lattice ---------------------------------------------------------


def test_the_committed_deck_and_model_are_the_generated_ones() -> None:
    assert (EXAMPLE / DECKS / "BR.json").read_text(encoding="utf-8") == booster().deck_text(
        _section()
    )
    assert _model() == booster().model(_deck())


@needs_mml
def test_the_excerpt_parses_like_mml() -> None:
    excerpt, original = _section(False), _section(True)
    assert excerpt.energy == original.energy
    # The excerpt drops blank and comment lines, so only the source lines differ.
    assert [(e.name, e.occurrence, e.definition) for e in excerpt.elements] == [
        (e.name, e.occurrence, e.definition) for e in original.elements
    ]


def test_the_deck_holds_the_mml_ring_in_order() -> None:
    section = _section()
    tokens = [element.name for element in section.elements]
    names = [element["FamName"] for element in _deck_elements()]

    assert [_token(name) if _token(name) in TOKEN_COUNTS else name for name in names] == tokens
    assert {t: n for t, n in Counter(tokens).items() if t in TOKEN_COUNTS} == TOKEN_COUNTS
    assert names[:3] == ["START", "AP", "CAV"] and names[-1] == "END"
    for token, count in TOKEN_COUNTS.items():
        assert [n for n in names if _token(n) == token] == [
            f"{token}{i}" for i in range(1, count + 1)
        ]


def test_every_length_angle_and_strength_is_mmls() -> None:
    section = _section()
    assert len(section.elements) == len(_deck_elements())
    for source, element in zip(section.elements, _deck_elements(), strict=True):
        assert element["Length"] == source.length, element["FamName"]
        if source.type == "quadrupole":
            assert element["PolynomB"] == [0.0, source.k]
        elif source.type == "rbend":
            assert element["BendingAngle"] == source.angle == pytest.approx(math.radians(15.0))
            assert element["EntranceAngle"] == source.definition.entrance_angle == 0.0
            assert element["ExitAngle"] == source.definition.exit_angle == 0.0
            assert element["PolynomB"] == [0.0, 0.0]
        elif source.type == "sextupole":
            assert (source.length, source.k, element["Class"]) == (0.0, 0.0, "Marker")
    quads = {e["FamName"]: e for e in _deck_elements() if e["Class"] == "Quadrupole"}
    assert quads["QF1"]["PolynomB"][1] == pytest.approx(2.7682214 * 0.96)
    assert quads["QD1"]["PolynomB"][1] == pytest.approx(-2.5401249 * 0.99)
    assert (quads["QF1"]["Length"], quads["QD1"]["Length"]) == (0.15, 0.10)


def test_a_sextupole_with_a_strength_is_refused() -> None:
    text = EXCERPT.read_text(encoding="latin-1").replace("Ksf =  0;", "Ksf =  1;")
    with pytest.raises(ValueError, match="a sextupole with a length or a strength"):
        booster().deck(_mml().parse(text, "BR"))


def test_the_cavity_is_mmls_at_the_demo_energy() -> None:
    (cavity,) = [e for e in _deck_elements() if e["Class"] == "RFCavity"]
    assert cavity["Voltage"] == 1.2e6
    assert cavity["HarmNumber"] == 125
    assert cavity["Frequency"] == pytest.approx(125 * 299792458 / 75)
    assert cavity["Energy"] == _deck()["properties"]["energy"] == booster().ENERGY == 2.0e9
    assert _deck()["properties"]["harmonic_number"] == 125
    # MML's file defaults to another energy; the optics are kept, the energy is the demo's.
    assert _section().energy == 1.9e9


def test_the_model_is_a_periodic_pyat_ring_at_one_energy() -> None:
    model = _model()
    assert model["engine"] == "pyat"
    assert model["deck"] == f"{DECKS}/BR.json"
    assert model["settings"] == {"pyat": {"solve": "periodic"}}
    ring = _ring()
    assert ring.energy == 2.0e9
    assert ring.is_6d
    assert ring.harmonic_number == 125


def test_the_24_bends_close_the_75_m_ring() -> None:
    bends = [e for e in _deck_elements() if e["Class"] == "Bend"]
    assert len(bends) == 24
    assert sum(e["BendingAngle"] for e in bends) == pytest.approx(2.0 * math.pi, rel=1e-14)
    assert _ring().circumference == pytest.approx(75.0, rel=1e-14)
    assert _ring().get_rf_frequency() == pytest.approx(125 * 299792458 / 75)


def test_the_periodic_solution_is_closed_and_stable() -> None:
    at = pytest.importorskip("at")
    ring = _ring4()

    orbit, _ = at.find_orbit4(ring)
    one_turn = at.find_m44(ring)[0]
    _, ringdata, along = at.get_optics(ring, refpts=range(len(ring) + 1), get_chrom=True)
    tunes = along.mu[-1] / (2.0 * np.pi)

    assert np.allclose(orbit, 0.0, atol=1e-12)
    assert abs(one_turn[0, 0] + one_turn[1, 1]) < 1.1
    assert abs(one_turn[2, 2] + one_turn[3, 3]) < 1.1
    assert np.linalg.det(one_turn) == pytest.approx(1.0, abs=1e-9)
    assert tunes == pytest.approx([6.3151, 2.1662], abs=5e-4)
    assert ringdata.tune == pytest.approx(tunes % 1.0, abs=1e-9)
    # No sextupole acts, so the chromaticity is the natural one.
    assert ringdata.chromaticity == pytest.approx([-10.895, -4.720], abs=5e-3)
    assert np.all(along.beta > 0.4) and np.all(along.beta < 12.5)
    assert np.allclose(along.beta[0], along.beta[-1], rtol=1e-9)
    assert along.dispersion[:, 0].max() == pytest.approx(1.044, abs=1e-3)


def test_the_ring_with_its_cavity_holds_the_design_orbit() -> None:
    orbit, _ = _ring().find_orbit()
    assert np.allclose(orbit, 0.0, atol=1e-10)


# --- the records are MML's families ------------------------------------------------------


def test_the_devices_are_mmls_families() -> None:
    every = [d for d in _committed("records/devices.yaml") if d["id"].startswith("BR/")]
    counts = Counter(d["id"].removeprefix("BR/").rstrip("0123456789") for d in every)

    assert {family: counts[family] for family in FAMILY_COUNTS} == FAMILY_COUNTS
    assert len(every) == sum(FAMILY_COUNTS.values()) + OTHER_DEVICES == 120
    assert every == sorted(every, key=lambda d: d["id"])
    emitted = [
        d for d in every if d["id"].removeprefix("BR/").rstrip("0123456789") in FAMILY_COUNTS
    ]
    assert emitted == booster().devices(_deck())
    assert {d["place"] for d in emitted} == {"BR"}


def test_each_device_is_its_deck_elements_in_beam_order() -> None:
    order = [element["FamName"] for element in _deck_elements()]
    bound = booster().bound(_deck())

    assert {family: len(devices) for family, devices in bound.items()} == FAMILY_COUNTS
    for family, devices in bound.items():
        positions = [order.index(names[0]) for names in devices.values()]
        assert positions == sorted(positions), family
        assert list(devices) == [f"{n:02d}" for n in range(1, len(devices) + 1)]
    assert bound["DIPOLE"]["24"] == ("BU24",)
    assert bound["QF"]["01"] == ("QF1", "QF2") and bound["QD"]["16"] == ("QD31", "QD32")
    assert bound["HCM"]["16"] == ("HCM16",)


def test_the_20_monitors_are_the_ones_mml_binds() -> None:
    """MML's ``BPMx`` lists devices 2, 3, 4, 6 and 8 of each sector, lattice monitor 8(s-1)+d."""
    bound = booster().bound(_deck())["BPM"]
    monitors = sorted(int(names[0].removeprefix("BPM")) for names in bound.values())
    assert monitors == [8 * sector + device for sector in range(4) for device in (2, 3, 4, 6, 8)]
    devices = {d["id"]: d for d in _committed("records/devices.yaml")}
    assert devices["BR/BPM01"]["names"] == ["BPM01", "BPM(1,2)"]
    assert devices["BR/BPM20"]["names"] == ["BPM20", "BPM(4,8)"]
    assert len([e for e in _deck_elements() if e["Class"] == "Monitor"]) == 32


def test_each_supply_current_is_on_its_supply_alone() -> None:
    channels = {record["id"]: record for record in _committed("records/channels.yaml")}
    for family, supply in SUPPLIES.items():
        setpoint = channels[booster().supply_address(family, "SP")]
        readback = channels[booster().supply_address(family, "RB")]
        assert setpoint["on"] == readback["on"] == {"device": supply}
        # One device per address: the wiring names the magnets a supply feeds.
        assert "endpoint_of" not in setpoint and "endpoint_of" not in readback
        assert setpoint["role"] == "setpoint" and setpoint["pair"] == readback["id"]
        assert setpoint["unit"] == readback["unit"] == "A"
        assert "tolerance" in setpoint
    # A magnet on a shared supply has no channel of its own.
    assert [
        a for a in channels if a.startswith(("BR:MAG:DIPOLE:", "BR:MAG:QF:", "BR:MAG:QD:"))
    ] == []


def test_the_channels_are_the_generated_ones() -> None:
    channels = [r for r in _committed("records/channels.yaml") if booster().owned(r["id"])]
    generated = booster().channels(_deck())
    assert [r["id"] for r in channels] == [r["id"] for r in generated]
    assert len(generated) == 6 + 32 * 3 + 20 * 3
    for record, expected in zip(channels, generated, strict=True):
        # The settle tolerance is the seed rules', stamped after generation.
        assert {**record, "tolerance": None} == {**expected, "tolerance": None}, record["id"]
    every = [r for r in _committed("records/channels.yaml") if r["id"].startswith("BR:")]
    assert len(every) == 203


def test_the_groups_hold_their_families() -> None:
    groups = {record["id"]: record for record in _committed("records/groups.yaml")}
    for family, count in FAMILY_COUNTS.items():
        assert groups[f"BR/{family}"]["members"] == [
            f"BR/{family}{n:02d}" for n in range(1, count + 1)
        ]
    assert len(groups["BR/MAG"]["members"]) == 24 + 16 + 16 + 16 + 16
    assert len(groups["BR/DIAG"]["members"]) == 21 and "BR/DCCT01" in groups["BR/DIAG"]["members"]
    assert [groups[r["id"]] for r in booster().groups(_deck())] == booster().groups(_deck())


def test_every_corrector_setpoint_carries_mmls_range() -> None:
    limits = {
        record["address"]: record
        for record in _committed("limits.yaml")["records"]
        if record["address"].startswith("BR:")
    }
    assert sorted(limits) == sorted(a for a in _corrector_currents() if a.endswith(":SP"))
    assert list(limits.values()) == booster().limits(_deck())
    for record in limits.values():
        assert (record["min_value"], record["max_value"]) == (-6.0, 6.0)


# --- the wiring --------------------------------------------------------------------


def test_every_supply_corrector_and_monitor_is_wired_and_no_other_channel() -> None:
    channels = {record["id"]: record for record in _committed("records/channels.yaml")}
    wired = {record["address"] for record in _model()["wiring"]}

    assert wired == {*_golden(), *_corrector_currents(), *_positions()}
    assert len(wired) == 6 + 64 + 40
    assert wired <= set(channels)


def test_each_supply_drives_every_element_of_its_family_a_slice_each() -> None:
    by_address = {record["address"]: record for record in _model()["wiring"]}
    elements = {element["FamName"]: element for element in _deck_elements()}
    for family, token in (("DIPOLE", "BU"), ("QF", "QF"), ("QD", "QD")):
        record = by_address[booster().supply_address(family, "SP")]
        assert "element" not in record
        assert [piece["element"] for piece in record["slices"]] == [
            f"{token}{i}" for i in range(1, TOKEN_COUNTS[token] + 1)
        ]
        assert all(set(piece) == {"element", "device"} for piece in record["slices"])
        named = list(dict.fromkeys(piece["device"] for piece in record["slices"]))
        assert named == [f"BR/{family}{n:02d}" for n in range(1, FAMILY_COUNTS[family] + 1)]
        strengths = {
            (elements[p["element"]]["Length"], tuple(elements[p["element"]]["PolynomB"]))
            for p in record["slices"]
        }
        assert len(strengths) == 1, family
        assert record == {**by_address[booster().supply_address(family, "RB")], **record}
    driven = {
        piece["element"]
        for record in _model()["wiring"]
        for piece in record.get("slices", [{"element": record.get("element")}])
    }
    magnets = {
        e["FamName"] for e in _deck_elements() if e["Class"] in ("Bend", "Quadrupole", "Corrector")
    }
    assert magnets <= driven


def test_setpoint_and_readback_share_elements_engine_and_calibration() -> None:
    by_address = {record["address"]: record for record in _model()["wiring"]}
    pairs = [a for a in by_address if a.endswith(":CURRENT:SP")]
    assert len(pairs) == 3 + 32
    for setpoint in pairs:
        readback = by_address[setpoint.removesuffix("SP") + "RB"]
        assert {**by_address[setpoint], "address": None} == {**readback, "address": None}


def test_at_golden_currents_the_deck_holds_mmls_optics() -> None:
    elements = {element["FamName"]: element for element in _deck_elements()}
    golden = _golden()
    by_address = {record["address"]: record for record in _model()["wiring"]}
    for address, current in golden.items():
        record = by_address[address]
        assert record["calibration"]["energy_scaling"] == "brho"
        physics = to_physics(curve_from_record(record["calibration"]["curve"]), current)
        for piece in record["slices"]:
            element = elements[piece["element"]]
            if record["engine"]["index"] == 1:
                assert physics == pytest.approx(element["PolynomB"][1], rel=1e-12), address
            else:
                assert element["Class"] == "Bend"
                assert physics == pytest.approx(0.0, abs=1e-12), address


def test_the_golden_currents_sit_inside_mmls_supply_ranges() -> None:
    golden = booster().GOLDEN
    assert 0.0 < golden["DIPOLE"] < 1000.0
    assert 0.0 < golden["QF"] < 130.0 and 0.0 < golden["QD"] < 130.0


def test_a_corrector_kicks_by_its_gain_and_rests_at_zero() -> None:
    by_address = {record["address"]: record for record in _model()["wiring"]}
    for address in _corrector_currents():
        record = by_address[address]
        assert record["engine"] == {
            "attribute": "KickAngle",
            "index": 0 if ":HCM:" in address else 1,
        }
        assert record["calibration"]["energy_scaling"] == "none"
        curve = curve_from_record(record["calibration"]["curve"])
        assert to_physics(curve, 0.0) == 0.0
        assert to_physics(curve, 1.0) == pytest.approx(booster().CORRECTOR_GAIN)


def test_the_held_setpoints_are_the_golden_currents(built: Any) -> None:
    golden = _golden()
    held = {
        entry["address"]: entry["default"]
        for entry in simulator_wiring(FacilityRecords(built), "BR")
        if entry["address"] in golden
    }
    assert sorted(held) == sorted(golden)
    for address, value in held.items():
        assert value == pytest.approx(golden[address], rel=1e-9, abs=1e-12), address


def test_the_monitors_read_millimetres() -> None:
    channels = {record["id"]: record for record in _committed("records/channels.yaml")}
    monitors = [record for record in _model()["wiring"] if "axis" in record["engine"]]
    assert len(monitors) == 40
    for record in monitors:
        assert channels[record["address"]]["unit"] == "mm", record["address"]
        assert record["calibration"]["energy_scaling"] == "none"
        curve = curve_from_record(record["calibration"]["curve"])
        assert to_hardware(curve, None, 1.0e-3) == pytest.approx(1.0), record["address"]


# --- the seeds, the place and the measurement file ---------------------------------


def test_no_wired_channel_carries_a_seed_nominal_and_each_readback_moves() -> None:
    seeds = _committed(SEEDS)
    wired = {record["address"] for record in _model()["wiring"]}
    assert sorted(a for a in wired if "nominal" in seeds.get(a, {})) == []
    assert seeds["BR:DIAG:BPM:20:POSITION:X"] == {"noise": {"absolute": 0.005}}
    for family in SUPPLIES:
        assert "noise" in seeds[booster().supply_address(family, "RB")]
    assert "noise" in seeds["BR:MAG:HCM:16:CURRENT:RB"]


def test_the_unwired_booster_channels_carry_their_nominals() -> None:
    seeds = _committed(SEEDS)
    for address in (
        "BR:MAG:DIPOLE-RAMP:01:GAIN:SP",
        "BR:MAG:HCM:01:STATUS:ON",
        "BR:DIAG:DCCT:01:CURRENT:RB",
        "BR:DIAG:BPM:20:STATUS:VALID",
    ):
        assert "nominal" in seeds[address], address


def test_no_seed_outlives_a_removed_channel() -> None:
    channels = {record["id"] for record in _committed("records/channels.yaml")}
    assert sorted(a for a in _committed(SEEDS) if a.startswith("BR:") and a not in channels) == []


def test_the_booster_place_spans_its_deck() -> None:
    places = {record["id"]: record for record in _committed("records/places.yaml")}
    assert places["BR"]["span"] == {"model": "BR", "from_marker": "START"}
    assert places["BR"]["level"] == "machine"


def test_every_lattice_device_takes_its_s_from_the_deck(built: Any) -> None:
    devices = [d for d in built["devices"] if d["id"].startswith("BR/")]
    modelled = {d["id"]: d for d in devices if d.get("model") == "BR"}

    assert len(modelled) == sum(FAMILY_COUNTS.values())
    assert not set(SUPPLIES.values()) & set(modelled)
    assert [d.get("model") for d in devices if d["id"] not in modelled] == [None] * OTHER_DEVICES
    bends = [modelled[f"BR/DIPOLE{n:02d}"]["s"] for n in range(1, 25)]
    assert bends == sorted(bends) and 0.0 < bends[0] and bends[-1] < 75.0


def test_the_measurement_file_lists_the_orbit_response_alone() -> None:
    measurement = _committed(MEASUREMENT)
    assert measurement == booster().measurement()
    assert measurement == {
        "kinds": ["orm"],
        "groups": {"bpm": "BR/BPM", "hcor": "BR/HCM", "vcor": "BR/VCM"},
        "corrector_delta": 1.0e-05,
    }
    # No tune and no RF frequency channel: nothing to measure a dispersion or a tune with.
    channels = [r["id"] for r in _committed("records/channels.yaml") if r["id"].startswith("BR:")]
    assert [a for a in channels if ":RF:" in a or ":TUNE:" in a] == []


@needs_mml
def test_generation_rewrites_the_committed_files(tmp_path: Path) -> None:
    tree = tmp_path / "example"
    shutil.copytree(EXAMPLE, tree)
    booster().write_deck(tree)
    booster().write_sources(tree)
    for rel in (
        f"{DECKS}/BR.json",
        MODELS,
        MEASUREMENT,
        "records/places.yaml",
        "records/devices.yaml",
        "records/groups.yaml",
        "limits.yaml",
        SEEDS,
    ):
        assert (tree / rel).read_bytes() == (EXAMPLE / rel).read_bytes(), rel
    written = read_yaml((tree / "records/channels.yaml").read_text(encoding="utf-8"))
    committed = _committed("records/channels.yaml")
    assert [{**r, "tolerance": None} for r in written] == [
        {**r, "tolerance": None} for r in committed
    ]


def test_generation_replaces_the_records_the_booster_had_before(tmp_path: Path) -> None:
    tree = tmp_path / "example"
    shutil.copytree(EXAMPLE, tree)
    channels = tree / "records/channels.yaml"
    channels.write_text(
        channels.read_text(encoding="utf-8")
        + "- id: BR:MAG:QF:01:STATUS:FAULT\n  on:\n    device: BR/QF01\n",
        encoding="utf-8",
    )
    seeds = tree / SEEDS
    seeds.write_text(
        seeds.read_text(encoding="utf-8") + "BR:MAG:QF:01:STATUS:FAULT:\n  nominal: 'FALSE'\n",
        encoding="utf-8",
    )
    booster().write_sources(tree)
    assert "BR:MAG:QF:01:STATUS:FAULT" not in channels.read_text(encoding="utf-8")
    assert (tree / SEEDS).read_bytes() == (EXAMPLE / SEEDS).read_bytes()


# --- the served ring ---------------------------------------------------------------


class Served:
    """The composite the in-process simulator builds from the demo view."""

    def __init__(self, view: Path) -> None:
        from osprey_connectors.simulation.composite import Composite

        self.composite = Composite(view, model_log=False)


@pytest.fixture(scope="module")
def served(built_demo: BuiltProject) -> Served:
    pytest.importorskip("at")
    return Served(built_demo.build_dir / "data" / "simulator")


def _monitor_orbit(ring: Any) -> Any:
    """The solved orbit in mm at the 20 device monitors, in device order."""
    names = [names[0] for names in booster().bound(_deck())["BPM"].values()]
    refpts = [int(ring.get_uint32_index(name)[0]) for name in names]
    _, orbit = ring.find_orbit(refpts=refpts)
    return 1.0e3 * orbit


def test_the_served_ring_starts_on_its_closed_orbit_at_golden_currents(served: Served) -> None:
    golden = _golden()
    positions = _positions()

    held = served.composite.held(sorted([*golden, *positions]))

    for address in positions:
        assert held[address] == pytest.approx(0.0, abs=0.05), address
    for address, current in golden.items():
        assert held[address] == pytest.approx(current, rel=1e-3), address


def test_a_corrector_moves_the_served_orbit_as_the_solved_ring_says(served: Served) -> None:
    setpoint = "BR:MAG:HCM:03:CURRENT:SP"
    positions = _positions("X")
    ring = _ring()
    (index,) = ring.get_uint32_index("HCM3")
    ring[index].KickAngle = [booster().CORRECTOR_GAIN, 0.0]
    expected_mm = _monitor_orbit(ring)[:, 0]

    served.composite.set({setpoint: 1.0})
    try:
        held = served.composite.held(positions)
    finally:
        served.composite.set({setpoint: 0.0})

    assert max(abs(expected_mm)) > 0.3
    assert [held[a] for a in positions] == pytest.approx(list(expected_mm), abs=0.05)
    back = served.composite.held(positions)
    assert [back[a] for a in positions] == pytest.approx([0.0] * 20, abs=0.05)


def test_a_quadrupole_supply_moves_all_its_magnets_of_the_served_ring(served: Served) -> None:
    """With one corrector on, the orbit follows the focusing of all 32 ``QF`` elements."""
    corrector = "BR:MAG:HCM:03:CURRENT:SP"
    supply = booster().supply_address("QF", "SP")
    golden = booster().GOLDEN["QF"]
    positions = _positions("X")
    ring = _ring()
    (index,) = ring.get_uint32_index("HCM3")
    ring[index].KickAngle = [booster().CORRECTOR_GAIN, 0.0]
    kicked_mm = _monitor_orbit(ring)[:, 0]
    for index in ring.get_uint32_index("QF*"):
        ring[index].PolynomB[1] *= 1.01
    expected_mm = _monitor_orbit(ring)[:, 0]

    served.composite.set({corrector: 1.0, supply: golden * 1.01})
    try:
        held = served.composite.held(positions)
    finally:
        served.composite.set({corrector: 0.0, supply: golden})

    assert max(abs(expected_mm - kicked_mm)) > 0.1
    assert [held[a] for a in positions] == pytest.approx(list(expected_mm), abs=0.02)


def test_the_cavity_holds_the_orbit_against_a_bend_supply_error(served: Served) -> None:
    """All 24 bends 0.1 % strong: without RF the orbit moves; with it the beam gains energy."""
    setpoint = booster().supply_address("DIPOLE", "SP")
    golden = booster().GOLDEN["DIPOLE"]
    positions = _positions("X")
    ring = _ring()
    for index in ring.get_uint32_index("BU*"):
        ring[index].PolynomB[0] = 1.0e-3 * ring[index].BendingAngle / ring[index].Length
    expected_mm = _monitor_orbit(ring)[:, 0]
    free = ring.deepcopy()
    free.disable_6d()

    served.composite.set({setpoint: golden * 1.001})
    try:
        held = served.composite.held(positions)
    finally:
        served.composite.set({setpoint: golden})

    assert max(abs(_monitor_orbit(free)[:, 0])) > 0.1
    assert max(abs(expected_mm)) < 1.0e-6
    assert [held[a] for a in positions] == pytest.approx(list(expected_mm), abs=0.05)
