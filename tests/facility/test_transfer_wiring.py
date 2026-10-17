"""The example facility's injector transfer-line models ``LTB`` and ``BTS``.

``scripts/facility_demo/_transfer.py`` emits one pyAT model per line: its deck
solved single-pass from the launch Twiss, wired from the element table. The
calibration maps each magnet's golden current (the table nominal) onto the
deck's own strength, so at golden currents every quadrupole holds its deck
``K``, every bend its design angle and every corrector a zero kick; the BPMs
read millimetres, ``1e3`` times the solved orbit. Every deck corrector,
quadrupole, monitor and bend is wired or a fixed element, and an uncovered one
stops generation, named. A wired channel carries no seed nominal, places LTB
and BTS span their decks, and each line has an orbit-response measurement file.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from osprey.facility.build import build_facility
from osprey.facility.sources import read_yaml
from osprey.facility.views.simulator import simulator_wiring
from osprey.simulation.engines.calibration import (
    curve_from_record,
    field,
    to_hardware,
    to_physics,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCES = REPO_ROOT / "scripts/facility_demo"
EXAMPLE = REPO_ROOT / "src/osprey/templates/facilities/example"

SECTIONS = ("LTB", "BTS")


def _module(name: str, filename: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    # ``_transfer`` imports its siblings by bare name, as generate.py does.
    spec = importlib.util.spec_from_file_location(name, SOURCES / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # import-time required because the section parametrizes below read the loaded module, and decorators are evaluated at import; scripts/ is not a package, so the
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


transfer = _module("facility_demo_transfer", "_transfer.py")
elements_table = transfer._elements


@cache
def _committed(rel: str) -> Any:
    return read_yaml((EXAMPLE / rel).read_text(encoding="utf-8"))


def _models() -> dict[str, dict[str, Any]]:
    return {model["name"]: model for model in _committed("models.yaml")}


def _deck(section: str) -> dict[str, Any]:
    return json.loads((EXAMPLE / "decks" / f"{section}.json").read_text(encoding="utf-8"))


def _deck_elements(section: str) -> dict[str, dict[str, Any]]:
    return {element["FamName"]: element for element in _deck(section)["elements"]}


def _golden() -> dict[str, float]:
    """Each wired setpoint's golden current, the table nominal."""
    return {
        row.demo_name: float(row.nominal)
        for row in transfer.table_rows()
        if row.wired and row.template.writable
    }


@pytest.fixture(scope="module")
def built() -> dict[str, Any]:
    return build_facility(EXAMPLE, project_name="demo")


# --- the models --------------------------------------------------------------------


@pytest.mark.parametrize("section", SECTIONS)
def test_the_committed_model_is_the_generated_one(section: str) -> None:
    assert _models()[section] == transfer.model(section, _deck(section))


@pytest.mark.parametrize("section", SECTIONS)
def test_each_line_is_a_single_pass_pyat_model_from_its_launch_twiss(section: str) -> None:
    model = _models()[section]
    assert model["engine"] == "pyat"
    assert model["deck"] == f"decks/{section}.json"
    pyat = model["settings"]["pyat"]
    assert pyat["solve"] == "single_pass"
    assert pyat["twiss_in"] == transfer.TWISS_IN[section]


def test_the_models_are_sorted_by_name() -> None:
    assert [model["name"] for model in _committed("models.yaml")] == ["BTS", "LINE", "LTB", "SR"]


@pytest.mark.parametrize("section", SECTIONS)
def test_every_wired_table_row_is_wired_and_nothing_else(section: str) -> None:
    wired = {row.demo_name for row in transfer.table_rows() if row.section == section and row.wired}
    assert {record["address"] for record in _models()[section]["wiring"]} == wired


def test_every_wired_address_is_a_channel() -> None:
    channels = {record["id"] for record in _committed("records/channels.yaml")}
    for section in SECTIONS:
        missing = [
            record["address"]
            for record in _models()[section]["wiring"]
            if record["address"] not in channels
        ]
        assert missing == [], section


def test_the_frozen_bts_rows_bind_through_their_deck_element() -> None:
    elements = {record["address"]: record["element"] for record in _models()["BTS"]["wiring"]}
    assert elements["BTS:MAG:QF:01:CURRENT:SP"] == "Q1"
    assert elements["BTS:MAG:QF:01:CURRENT:RB"] == "Q1"
    assert elements["BTS:MAG:HCM:06:CURRENT:SP"] == "HCM6"
    assert elements["BTS:MAG:VCM:06:CURRENT:RB"] == "VCM6"
    assert elements["BTS:DIAG:BPM:01:POSITION:X"] == "BPM1"
    assert elements["BTS:DIAG:BPM:06:POSITION:Y"] == "BPM6"
    frozen = [row for row in transfer.table_rows() if transfer.frozen(row) and row.wired]
    assert len(frozen) == 4 * 6 * 2
    assert all(row.demo_name in elements for row in frozen)


@pytest.mark.parametrize("section", SECTIONS)
def test_setpoint_and_readback_share_element_engine_and_calibration(section: str) -> None:
    by_address = {record["address"]: record for record in _models()[section]["wiring"]}
    setpoints = [address for address in by_address if address.endswith(":SP")]
    assert setpoints
    for address in setpoints:
        setpoint, readback = by_address[address], by_address[f"{address[:-3]}:RB"]
        assert {k: v for k, v in setpoint.items() if k != "address"} == {
            k: v for k, v in readback.items() if k != "address"
        }, address


# --- held == golden ----------------------------------------------------------------


@pytest.mark.parametrize("section", SECTIONS)
def test_at_golden_currents_the_deck_holds_its_own_optics(section: str) -> None:
    elements = _deck_elements(section)
    golden = _golden()
    checked = 0
    for record in _models()[section]["wiring"]:
        if record["address"] not in golden:
            continue
        curve = curve_from_record(field(field(record, "calibration"), "curve"))
        physics = to_physics(curve, golden[record["address"]])
        element = elements[record["element"]]
        engine = record["engine"]
        if engine["attribute"] == "KickAngle":
            assert physics == pytest.approx(0.0, abs=1e-15), record["address"]
        elif engine["index"] == 1:
            assert physics == pytest.approx(element["PolynomB"][1], rel=1e-12), record["address"]
        else:
            assert element["Class"] == "Bend"
            assert physics == pytest.approx(0.0, abs=1e-12), record["address"]
        checked += 1
    assert checked == sum(1 for address in golden if address.startswith(f"{section}:"))


@pytest.mark.parametrize("section", SECTIONS)
def test_the_held_setpoints_are_the_golden_currents(section: str, built: Any) -> None:
    golden = _golden()
    held = {
        entry["address"]: entry["default"]
        for entry in simulator_wiring(built, section)
        if entry["address"] in golden
    }
    assert held, section
    assert sorted(held) == sorted(a for a in golden if a.startswith(f"{section}:"))
    for address, value in held.items():
        assert value == pytest.approx(golden[address], rel=1e-9, abs=1e-12), address


@pytest.mark.parametrize("section", SECTIONS)
def test_a_corrector_kick_is_gain_times_the_current_change(section: str) -> None:
    golden = _golden()
    for record in _models()[section]["wiring"]:
        if record["engine"].get("attribute") != "KickAngle":
            continue
        curve = curve_from_record(record["calibration"]["curve"])
        kick = to_physics(curve, golden[record["address"].replace(":RB", ":SP")] + 1.0)
        assert kick == pytest.approx(transfer.CORRECTOR_GAIN[section]), record["address"]


# --- the BPMs ----------------------------------------------------------------------


@pytest.mark.parametrize("section", SECTIONS)
def test_the_bpms_read_millimetres(section: str) -> None:
    channels = {record["id"]: record for record in _committed("records/channels.yaml")}
    bpms = [record for record in _models()[section]["wiring"] if "axis" in record["engine"]]
    assert len(bpms) == 12
    for record in bpms:
        assert channels[record["address"]]["unit"] == "mm", record["address"]
        calibration = record["calibration"]
        assert calibration["energy_scaling"] == "none"
        curve = curve_from_record(calibration["curve"])
        assert to_hardware(curve, None, 1.0e-3) == pytest.approx(1.0), record["address"]
        assert to_hardware(curve, None, -2.5e-4) == pytest.approx(-0.25), record["address"]


@pytest.mark.parametrize("section", SECTIONS)
def test_a_kicked_orbit_reads_in_millimetres_at_the_bpms(section: str) -> None:
    at = pytest.importorskip("at")
    lattice = at.load_lattice(str(EXAMPLE / "decks" / f"{section}.json"))
    wiring = _models()[section]["wiring"]
    corrector = next(r for r in wiring if r["engine"].get("attribute") == "KickAngle")
    curve = curve_from_record(corrector["calibration"]["curve"])
    golden = _golden()[corrector["address"].replace(":RB", ":SP")]
    (index,) = lattice.get_uint32_index(corrector["element"])
    kick = to_physics(curve, golden + 0.5)
    lattice[index].KickAngle[corrector["engine"]["index"]] = kick
    monitors = lattice.get_uint32_index(at.Monitor)
    particle = np.zeros((6, 1))
    tracked, *_ = lattice.track(particle, refpts=monitors)
    plane = 0 if corrector["engine"]["index"] == 0 else 2
    orbit_m = tracked[plane, 0, :, 0]
    assert max(abs(orbit_m)) > 0.0
    bpm = next(r for r in wiring if "axis" in r["engine"])
    bpm_curve = curve_from_record(bpm["calibration"]["curve"])
    for value in orbit_m:
        assert to_hardware(bpm_curve, None, float(value)) == pytest.approx(1.0e3 * value)


# --- coverage ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("section", "dropped", "element"),
    [
        ("LTB", "LTB:MAG:HCM:01:CURRENT", "HCM1"),
        ("LTB", "LTB:MAG:QD:01:CURRENT", "Q11"),
        ("BTS", "BTS:DIAG:BPM:03:POSITION", "BPM3"),
        ("BTS", "BTS:MAG:DIPOLE:01:CURRENT", "B1"),
    ],
)
def test_an_uncovered_deck_element_fails_generation_named(
    section: str, dropped: str, element: str
) -> None:
    table = [row for row in transfer.table_rows() if not row.demo_name.startswith(dropped)]
    with pytest.raises(ValueError, match=rf"neither wired nor fixed: .*\b{element}\b"):
        transfer.wiring(section, _deck(section), table)


@pytest.mark.parametrize("section", SECTIONS)
def test_every_deck_magnet_and_monitor_is_wired_or_fixed(section: str) -> None:
    wired = {record["element"] for record in _models()[section]["wiring"]}
    names = set(_deck_elements(section))
    fixed = {
        transfer.deck_element_name(key, names)
        for key in elements_table.FIXED_ELEMENTS
        if key.startswith(f"{section}:")
    }
    for name, element in _deck_elements(section).items():
        if element["Class"] in transfer.COVERED_CLASSES:
            assert (name in wired) != (name in fixed), name


def test_a_deck_element_maps_to_its_deck_name() -> None:
    names = set(_deck_elements("LTB"))
    assert transfer.deck_element_name("LTB:Q11#1", names) == "Q11"
    assert transfer.deck_element_name("LTB:BPM#3", names) == "BPM3"
    assert transfer.deck_element_name("LTB:BS#1", names) == "BS"
    with pytest.raises(ValueError, match="LTB:Q99#1"):
        transfer.deck_element_name("LTB:Q99#1", names)


# --- seeds, places, measurement ----------------------------------------------------


def test_no_wired_channel_carries_a_seed_nominal() -> None:
    seeds = _committed("seeds.yaml")
    wired = {record["address"] for section in SECTIONS for record in _models()[section]["wiring"]}
    assert sorted(a for a in wired if "nominal" in seeds.get(a, {})) == []


def test_the_frozen_bts_readbacks_keep_their_motion() -> None:
    seeds = _committed("seeds.yaml")
    assert "nominal" not in seeds["BTS:MAG:QF:01:CURRENT:RB"]
    assert "noise" in seeds["BTS:MAG:QF:01:CURRENT:RB"]


def test_the_unwired_transfer_rows_keep_their_nominals() -> None:
    seeds = _committed("seeds.yaml")
    for address in ("LTB:MAG:QF:01:STATUS:ON", "BTS:DIAG:PADDLES:01:COUNT:INSERTED"):
        assert "nominal" in seeds[address], address


@pytest.mark.parametrize("section", SECTIONS)
def test_each_line_place_spans_its_deck(section: str) -> None:
    places = {record["id"]: record for record in _committed("records/places.yaml")}
    assert places[section]["span"] == {"model": section, "from_marker": transfer.START_MARKER}
    assert places[section]["level"] == "machine"
    assert places[section]["description"]


@pytest.mark.parametrize("section", SECTIONS)
def test_each_line_measures_its_orbit_response(section: str) -> None:
    measurement = _committed(f"measurement/{section}.yaml")
    assert measurement == transfer.measurement(section)
    assert measurement["kinds"] == ["orm"]
    groups = {record["id"] for record in _committed("records/groups.yaml")}
    assert set(measurement["groups"].values()) <= groups


def test_generation_rewrites_the_committed_files(tmp_path: Path) -> None:
    tree = tmp_path / "example"
    shutil.copytree(EXAMPLE, tree)
    transfer.write_sources(tree)
    for rel in (
        "models.yaml",
        "measurement/LTB.yaml",
        "measurement/BTS.yaml",
        "records/places.yaml",
        "seeds.yaml",
    ):
        assert (tree / rel).read_bytes() == (EXAMPLE / rel).read_bytes(), rel
