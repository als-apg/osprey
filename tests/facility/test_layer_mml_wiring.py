"""The mml layer's wiring pass: each imported model's deck and wiring records.

Each case imports a fixture tree's exports into a fresh ``data/facility/``
under the tree's new-format mapping. The parity cases also run the old chain
(``osprey mml import``, ``map`` and ``emit``) over the same exports and hold
every imported wiring record to the binding that chain emits for its address.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.facility.layers.mml import decks
from osprey.facility.layers.mml.importer import LAYER_DIR, import_mml
from osprey.facility.layers.mml.mapping import MAPPING_FILE, ImportStop, read_mapping
from osprey.facility.layers.mml.wiring import _one_way
from osprey.simulation.engines.calibration import Linear, Table

at = pytest.importorskip("at")

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "mml"

#: Each tree's exports, by the stem every file of one export is named after.
TREES: dict[str, tuple[str, ...]] = {
    "spear3": ("spear3.storagering",),
    "nsls2": ("nsls2.storagering", "nsls2.ltb"),
}

#: The system each export stem carries, which names its model in both trees.
SYSTEMS: dict[str, str] = {
    "spear3.storagering": "StorageRing",
    "nsls2.storagering": "StorageRing",
    "nsls2.ltb": "LTB",
}

#: The model the old chain emits bindings for, in both trees.
STORAGE = "StorageRing"

Edit = Callable[[dict[str, Any]], None]


def _facility(root: Path, tree: str, edit: Edit | None = None) -> Path:
    facility = root / "data" / "facility"
    target = facility / MAPPING_FILE
    target.parent.mkdir(parents=True)
    document = yaml.safe_load((FIXTURES / tree / MAPPING_FILE).read_text(encoding="utf-8"))
    if edit is not None:
        edit(document)
    target.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return facility


def _exports(root: Path, tree: str, stems: tuple[str, ...], drop: tuple[str, ...] = ()) -> list:
    """The tree's exports copied under ``root``, without the siblings ``drop`` names."""
    exports = root / "exports"
    exports.mkdir()
    for stem in stems:
        for source in sorted((FIXTURES / tree).glob(f"{stem}.*")):
            if not source.name.endswith(drop):
                shutil.copyfile(source, exports / source.name)
    return [exports / f"{stem}.ao.json" for stem in stems]


def _models(facility: Path) -> dict[str, dict[str, Any]]:
    rows = yaml.safe_load((facility / LAYER_DIR / "models.yaml").read_text(encoding="utf-8"))
    return {row["name"]: row for row in rows}


def _layer_files(facility: Path) -> list[str]:
    return sorted(p.relative_to(facility).as_posix() for p in facility.rglob("*") if p.is_file())


def _stops(root: Path, tree: str, stems: tuple[str, ...], edit: Edit, drop: tuple = ()) -> str:
    """Import with an edited mapping, expecting a stop that writes nothing."""
    facility = _facility(root, tree, edit)
    with pytest.raises(ImportStop) as stop:
        import_mml(_exports(root, tree, stems, drop), facility)
    assert _layer_files(facility) == [MAPPING_FILE]
    return stop.value.format_message()


@pytest.fixture(scope="module", params=sorted(TREES))
def tree(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(scope="module")
def imported(tree: str, tmp_path_factory: pytest.TempPathFactory) -> Path:
    facility = _facility(tmp_path_factory.mktemp(tree), tree)
    import_mml([FIXTURES / tree / f"{stem}.ao.json" for stem in TREES[tree]], facility)
    return facility


# --- parity with the old chain ------------------------------------------------------


def _emitted_bindings(root: Path, tree: str) -> list[dict[str, Any]]:
    """Run the old chain over a tree's exports and return the bindings it emits."""
    pytest.importorskip("linkml_runtime")
    from click.testing import CliRunner

    from osprey.cli.main import cli

    def run(*args: str) -> None:
        result = CliRunner().invoke(cli, [*args, "--repo", str(root)], catch_exceptions=False)
        assert result.exit_code == 0, f"osprey {' '.join(args)}:\n{result.output}"

    root.mkdir(parents=True, exist_ok=True)
    (root / "profile.yml").write_text("name: scratch\n", encoding="utf-8")
    run("mml", "import", *(str(FIXTURES / tree / f"{stem}.ao.json") for stem in TREES[tree]))
    run("mml", "map", "--init")
    shutil.copy(FIXTURES / tree / "mapping.yaml", root / "data" / "mml" / "mapping.yaml")
    run("mml", "emit")
    document = json.loads((root / "data" / "simulation" / "va_bindings.json").read_text())
    assert document["system"] == STORAGE
    return document["bindings"]


def _curve(curve: dict[str, Any] | None) -> dict[str, Any] | None:
    """A record's curve in the bindings document's spelling."""
    if curve is None:
        return None
    ((kind, body),) = curve.items()
    return {"kind": kind, **body}


def _slices(record: dict[str, Any]) -> list[tuple[str, float]]:
    pieces = record.get("slices") or [{"element": record["element"]}]
    return [(piece["element"], piece.get("weight", 1.0)) for piece in pieces]


def test_every_wiring_record_equals_the_emitted_binding_for_its_address(
    tree: str, imported: Path, tmp_path: Path
) -> None:
    bindings = _emitted_bindings(tmp_path / "old", tree)
    emitted: dict[str, dict[str, Any]] = {}
    for binding in bindings:
        if binding["element"] is None:
            continue
        for key in ("setpoint_address", "readback_address"):
            if binding[key]:
                emitted[binding[key]] = binding
    wiring = {record["address"]: record for record in _models(imported)[STORAGE]["wiring"]}

    assert sorted(wiring) == sorted(emitted)
    for address, record in wiring.items():
        binding = emitted[address]
        engine, calibration = record["engine"], record["calibration"]
        assert _slices(record)[0][0] == binding["element"], address
        assert engine.get("attribute", engine.get("axis")) == binding["attribute"], address
        assert engine.get("index") == binding["index"], address
        assert _curve(calibration["curve"]) == binding["calibration"], address
        assert calibration["energy_scaling"] == binding["energy_scaling"], address
        if binding["monitor_inverse"] is not None:
            assert _curve(calibration["inverse"]) == binding["monitor_inverse"], address
        expected = [(piece["element"], piece["weight"]) for piece in binding["slices"]]
        assert [name for name, _ in _slices(record)] == [name for name, _ in expected], address
        assert [weight for _, weight in _slices(record)] == pytest.approx(
            [weight for _, weight in expected], rel=1e-12
        ), address


# --- the entry and its deck ---------------------------------------------------------


def test_models_yaml_holds_one_entry_per_imported_system(tree: str, imported: Path) -> None:
    rows = yaml.safe_load((imported / LAYER_DIR / "models.yaml").read_text(encoding="utf-8"))
    assert sorted(row["name"] for row in rows) == sorted(SYSTEMS[stem] for stem in TREES[tree])


def test_every_entry_names_its_deck_between_engine_and_settings(imported: Path) -> None:
    for name, entry in _models(imported).items():
        assert entry["deck"] == f"imported/mml/decks/{name}.json"
        assert [key for key in entry if key != "settings"] == ["name", "engine", "deck", "wiring"]
        assert list(entry)[-1] == "wiring"
        assert (imported / entry["deck"]).is_file()


def test_every_written_deck_is_the_served_deck_of_its_export(tree: str, imported: Path) -> None:
    from osprey.services.mml.loaders.mat import load_lattice

    mapping = read_mapping(FIXTURES / tree / MAPPING_FILE)
    for stem in TREES[tree]:
        system = SYSTEMS[stem]
        va = json.loads((FIXTURES / tree / f"{stem}.va.json").read_text(encoding="utf-8"))
        ad = json.loads((FIXTURES / tree / f"{stem}.ad.json").read_text(encoding="utf-8"))
        addressing = decks.address_elements(
            mapping.models[system], load_lattice(FIXTURES / tree / f"{stem}.lattice.mat"), va, ad
        )
        written = imported / decks.DECKS_DIR / f"{system}.json"
        assert written.read_text(encoding="utf-8") == decks.deck_text(decks.served_deck(addressing))


def test_every_wired_element_is_in_the_written_deck_exactly_once(imported: Path) -> None:
    for entry in _models(imported).values():
        document = json.loads((imported / entry["deck"]).read_text(encoding="utf-8"))
        names = Counter(element["FamName"] for element in document["elements"])
        wired = {name for record in entry["wiring"] for name, _ in _slices(record)}
        assert wired
        assert sorted(name for name in wired if names[name] != 1) == []


def test_models_yaml_states_every_record_in_full(imported: Path) -> None:
    text = (imported / LAYER_DIR / "models.yaml").read_text(encoding="utf-8")
    assert [line for line in text.splitlines() if "&id" in line or "*id" in line] == []


def test_every_wired_address_is_a_channel_record(imported: Path) -> None:
    rows = yaml.safe_load((imported / LAYER_DIR / "channels.yaml").read_text(encoding="utf-8"))
    channels = {row["id"] for row in rows}
    for entry in _models(imported).values():
        addresses = [record["address"] for record in entry["wiring"]]
        assert addresses == sorted(set(addresses))
        assert set(addresses) <= channels


def test_every_device_of_a_shared_endpoint_is_named_by_a_slice(imported: Path) -> None:
    rows = yaml.safe_load((imported / LAYER_DIR / "channels.yaml").read_text(encoding="utf-8"))
    shared = {row["id"]: row["endpoint_of"] for row in rows if "endpoint_of" in row}
    wired = 0
    for entry in _models(imported).values():
        for record in entry["wiring"]:
            if record["address"] not in shared:
                continue
            wired += 1
            named = [piece.get("device") for piece in record["slices"]]
            assert sorted(set(named)) == sorted(shared[record["address"]]), record["address"]
    assert wired


def test_a_second_import_writes_the_same_decks_and_wiring(
    tree: str, imported: Path, tmp_path: Path
) -> None:
    again = _facility(tmp_path, tree)
    import_mml([FIXTURES / tree / f"{stem}.ao.json" for stem in TREES[tree]], again)
    names = ["models.yaml", *(f"decks/{SYSTEMS[stem]}.json" for stem in TREES[tree])]
    for name in names:
        assert (again / LAYER_DIR / name).read_bytes() == (imported / LAYER_DIR / name).read_bytes()


def test_the_import_returns_each_deck_it_wrote(tmp_path: Path) -> None:
    facility = _facility(tmp_path, "spear3")
    written = import_mml([FIXTURES / "spear3" / "spear3.storagering.ao.json"], facility)
    assert facility / decks.DECKS_DIR / "StorageRing.json" in written


def test_a_deck_of_a_model_the_import_does_not_carry_is_removed(tmp_path: Path) -> None:
    facility = _facility(tmp_path, "nsls2")
    import_mml([FIXTURES / "nsls2" / f"{stem}.ao.json" for stem in TREES["nsls2"]], facility)
    assert (facility / decks.DECKS_DIR / "LTB.json").is_file()
    import_mml([FIXTURES / "nsls2" / "nsls2.storagering.ao.json"], facility)
    assert sorted(p.name for p in (facility / decks.DECKS_DIR).iterdir()) == ["StorageRing.json"]


# --- stops --------------------------------------------------------------------------


def test_a_wired_family_the_export_places_no_device_of_stops_per_family(tmp_path: Path) -> None:
    def unplaced(document: dict[str, Any]) -> None:
        wiring = document["models"]["LTB"].setdefault("wiring", {})
        for family, axis in (("BPMx", "x"), ("BPMy", "y")):
            wiring[family] = {
                "element_field": "Monitor",
                "engine": {"axis": axis},
                "calibration": "linear",
            }

    assert _stops(tmp_path, "nsls2", ("nsls2.ltb",), unplaced).splitlines() == [
        "import mml: export-invalid: LTB: family BPMx is wired through Monitor "
        "and the export places none of its devices",
        "import mml: export-invalid: LTB: family BPMy is wired through Monitor "
        "and the export places none of its devices",
    ]


def test_what_the_deck_pass_refuses_stops_the_import(tmp_path: Path) -> None:
    def unranked(document: dict[str, Any]) -> None:
        document["models"]["StorageRing"]["wiring"]["QF"]["engine"]["attribute"] = "K"

    assert _stops(tmp_path, "spear3", TREES["spear3"], unranked) == (
        "import mml: export-invalid: StorageRing: family QF drives K; wire it to an axis "
        "or to PolynomB, PolynomA, KickAngle or Frequency"
    )


def test_wired_families_without_a_saved_deck_stop_the_import(tmp_path: Path) -> None:
    message = _stops(tmp_path, "spear3", TREES["spear3"], lambda _: None, drop=(".lattice.mat",))
    assert message == (
        "import mml: export-invalid: StorageRing: the mapping wires families "
        "and the export saved no deck"
    )


def test_a_calibration_of_another_shape_than_the_mapping_names_stops(tmp_path: Path) -> None:
    def table(document: dict[str, Any]) -> None:
        document["models"]["StorageRing"]["wiring"]["HCM"]["calibration"] = "table"

    assert _stops(tmp_path, "spear3", TREES["spear3"], table) == (
        "import mml: export-invalid: StorageRing: family HCM device 1: the mapping names "
        "a table calibration and the export states a linear one"
    )


def test_a_wired_address_no_channel_record_carries_stops(tmp_path: Path) -> None:
    def skipped(document: dict[str, Any]) -> None:
        document["families"]["RF"]["channels"] = 0

    lines = _stops(tmp_path, "spear3", TREES["spear3"], skipped).splitlines()
    assert lines
    for line in lines:
        assert line.startswith("import mml: reference-missing: wiring StorageRing/")
        assert line.endswith(": no channel record carries the address")


def test_a_driven_readback_the_export_states_no_way_back_for_stops(tmp_path: Path) -> None:
    facility = _facility(tmp_path, "nsls2")
    exports = _exports(tmp_path, "nsls2", ("nsls2.ltb",))
    sibling = exports[0].with_name("nsls2.ltb.va.json")
    document = json.loads(sibling.read_text(encoding="utf-8"))
    del document["families"]["Q"]["Monitor"]["monitor_inverse"]
    sibling.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ImportStop) as stop:
        import_mml(exports, facility)

    assert _layer_files(facility) == [MAPPING_FILE]
    lines = stop.value.format_message().splitlines()
    assert lines
    for line in lines:
        assert line.startswith("import mml: export-invalid: LTB: family Q device ")
        assert ": serves its readback on " in line
        assert line.endswith(" and states no monitor_inverse")


def test_an_unanswered_cavity_voltage_stays_mapping_undecided(tmp_path: Path) -> None:
    def unanswered(document: dict[str, Any]) -> None:
        del document["models"]["StorageRing"]["wiring"]["RF"]["voltage"]

    assert _stops(tmp_path, "nsls2", ("nsls2.storagering",), unanswered) == (
        "import mml: mapping-undecided: models.StorageRing.wiring.RF.voltage: "
        "answer the cavity voltage in volts; the deck holds no cavity"
    )


# --- curves -------------------------------------------------------------------------


def test_a_curve_that_turns_back_keeps_the_stretch_holding_the_operating_point() -> None:
    curve = Table(grid=(0.0, 1.0, 2.0, 1.5, 1.0), values=(0.0, 10.0, 20.0, 30.0, 40.0))
    assert _one_way(curve, "family Q device 1", "calibration", hardware=0.9) == Table(
        grid=(0.0, 1.0, 2.0), values=(0.0, 10.0, 20.0)
    )


def test_a_curve_back_to_hardware_is_placed_on_its_values() -> None:
    curve = Table(grid=(0.0, 1.0, 2.0, 1.5, 1.0), values=(0.0, 10.0, 20.0, 30.0, 40.0))
    kept = _one_way(curve, "family Q device 1", "monitor_inverse", hardware=38.0, on_values=True)
    assert kept == Table(grid=(2.0, 1.5, 1.0), values=(20.0, 30.0, 40.0))


def test_a_curve_that_reads_one_way_and_a_straight_line_are_kept_whole() -> None:
    table = Table(grid=(0.0, 1.0, 2.0), values=(0.0, 10.0, 20.0))
    line = Linear(gain=2.0, offset=1.0)
    assert _one_way(table, "family Q device 1", "calibration", hardware=5.0) is table
    assert _one_way(line, "family Q device 1", "calibration", hardware=5.0) is line


def test_a_curve_that_stands_still_everywhere_is_refused() -> None:
    curve = Table(grid=(1.0, 1.0, 1.0), values=(0.0, 1.0, 2.0))
    with pytest.raises(ValueError, match="repeats one sampled point across the whole grid"):
        _one_way(curve, "family Q device 1", "calibration", hardware=1.0)


# --- the transfer line --------------------------------------------------------------


@pytest.fixture(scope="module")
def transfer(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    facility = _facility(tmp_path_factory.mktemp("transfer"), "nsls2")
    import_mml([FIXTURES / "nsls2" / f"{stem}.ao.json" for stem in TREES["nsls2"]], facility)
    return _models(facility)["LTB"]


def test_the_transfer_line_keeps_its_single_pass_settings_beside_its_wiring(
    transfer: dict[str, Any],
) -> None:
    assert transfer["settings"]["pyat"]["solve"] == "single_pass"
    assert set(transfer["settings"]["pyat"]) == {"solve", "twiss_in"}
    assert transfer["wiring"]


def test_the_transfer_line_wires_its_correctors_and_quadrupoles_only(
    transfer: dict[str, Any],
) -> None:
    ao = json.loads((FIXTURES / "nsls2" / "nsls2.ltb.ao.json").read_text(encoding="utf-8"))

    def addresses(*families: str) -> set[str]:
        return {
            name.strip()
            for family in families
            for field in ("Setpoint", "Monitor")
            for name in ao[family].get(field, {}).get("ChannelNames", [])
            if name.strip()
        }

    wired = {record["address"] for record in transfer["wiring"]}
    assert wired == addresses("HCM", "VCM", "Q")
    assert not wired & addresses("BEND", "Screen", "BPMx", "BPMy")
    engines = Counter(
        (record["engine"]["attribute"], record["engine"]["index"]) for record in transfer["wiring"]
    )
    assert engines == {("KickAngle", 0): 16, ("KickAngle", 1): 16, ("PolynomB", 1): 30}


def test_the_transfer_line_carries_the_calibrations_its_export_states(
    transfer: dict[str, Any],
) -> None:
    ao = json.loads((FIXTURES / "nsls2" / "nsls2.ltb.ao.json").read_text(encoding="utf-8"))
    va = json.loads((FIXTURES / "nsls2" / "nsls2.ltb.va.json").read_text(encoding="utf-8"))
    wiring = {record["address"]: record for record in transfer["wiring"]}
    for family in ("HCM", "VCM", "Q"):
        facts = va["families"][family]
        stated = facts["Setpoint"]["calibration"]
        inverse = facts["Monitor"]["monitor_inverse"]
        for device, name in enumerate(ao[family]["Setpoint"]["ChannelNames"]):
            calibration = wiring[name.strip()]["calibration"]
            assert calibration == {
                "curve": {
                    "linear": {"gain": stated["gain"][device], "offset": stated["offset"][device]}
                },
                "inverse": {
                    "linear": {"gain": inverse["gain"][device], "offset": inverse["offset"][device]}
                },
                "energy_scaling": facts["Setpoint"]["energy_scaling"],
            }
