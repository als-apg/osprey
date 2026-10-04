"""The mml layer's seed-once files: authored files an import creates when absent.

Each case imports a fixture tree's exports into a fresh ``data/facility/``
under the tree's new-format mapping and reads the authored files the import
seeded beside the layer's records. The build cases run the in-process build
over the imported tree, and the writable case runs the old command chain over
the same exports.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.facility.build import LATER_STAGES, build_facility
from osprey.facility.errors import FacilityBuildError
from osprey.facility.layers.mml.importer import (
    LAYER_DIR,
    import_mml,
    read_exports,
    write_records,
)
from osprey.facility.layers.mml.mapping import MAPPING_FILE, read_mapping
from osprey.facility.layers.mml.seed import HEADER, READOUT_FILE, TUNING, band
from osprey.facility.validate import known_classes, run_stages

at = pytest.importorskip("at")

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "mml"

#: The spear3 energy knob's setpoint, which the old chain binds to no element.
ENERGY_KNOB = "MS1-BD:CurrSetpt"

#: Each tree's exports, by the stem every file of one export is named after.
TREES: dict[str, tuple[str, ...]] = {
    "spear3": ("spear3.storagering",),
    "nsls2": ("nsls2.storagering", "nsls2.ltb"),
    "synthetic": ("quokka.sr",),
}

#: The trees the build case runs to a clean exit.
BUILT = ("nsls2", "spear3")

#: The corrector setpoint the synthetic export starts outside its own ``Range``.
OUTSIDE = "QK:HC:1:CUR:SP"


def _facility(root: Path, tree: str) -> Path:
    facility = root / "data" / "facility"
    target = facility / MAPPING_FILE
    target.parent.mkdir(parents=True)
    shutil.copyfile(FIXTURES / tree / MAPPING_FILE, target)
    return facility


def _sources(tree: str) -> list[Path]:
    return [FIXTURES / tree / f"{stem}.ao.json" for stem in TREES[tree]]


def _import(root: Path, tree: str) -> Path:
    facility = _facility(root, tree)
    import_mml(_sources(tree), facility)
    return facility


def _edited(root: Path, tree: str, stem: str, edit: Any) -> Path:
    """Copy one export of a tree, apply ``edit`` to its AO document and return the AO."""
    exports = root / "exports"
    exports.mkdir()
    for source in sorted((FIXTURES / tree).glob(f"{stem}.*")):
        shutil.copyfile(source, exports / source.name)
    ao = exports / f"{stem}.ao.json"
    document = json.loads(ao.read_text(encoding="utf-8"))
    edit(document)
    ao.write_text(json.dumps(document), encoding="utf-8")
    return ao


def _load(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _limits(facility: Path) -> dict[str, dict[str, Any]]:
    return {row["address"]: row for row in _load(facility / "limits.yaml")["records"]}


def _wired(facility: Path) -> set[str]:
    return {
        record["address"]
        for model in _load(facility / LAYER_DIR / "models.yaml")
        for record in model.get("wiring", [])
    }


def _roles(facility: Path) -> dict[str, str]:
    channels = _load(facility / LAYER_DIR / "channels.yaml")
    return {channel["id"]: channel.get("role", "readback") for channel in channels}


def _authored(facility: Path) -> dict[str, bytes]:
    """Every file outside the layer directory, by its path under ``data/facility``."""
    return {
        path.relative_to(facility).as_posix(): path.read_bytes()
        for path in sorted(facility.rglob("*"))
        if path.is_file() and LAYER_DIR not in path.relative_to(facility).as_posix()
    }


@pytest.fixture(scope="module", params=sorted(TREES))
def tree(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(scope="module")
def imported(tree: str, tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _import(tmp_path_factory.mktemp(tree), tree)


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _import(tmp_path_factory.mktemp("synthetic-limits"), "synthetic")


@pytest.fixture(scope="module")
def spear3(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _import(tmp_path_factory.mktemp("spear3-seeded"), "spear3")


# --- every seeded file --------------------------------------------------------------


def test_every_seeded_file_opens_with_the_header(imported: Path) -> None:
    authored = _authored(imported)
    assert {"limits.yaml", "seeds.yaml", "classes.yaml"} <= set(authored)
    for name, content in authored.items():
        assert content.decode("utf-8").splitlines()[0] == HEADER, name
    assert HEADER.startswith("# ")


def test_the_import_returns_each_file_it_seeded(tmp_path: Path) -> None:
    facility = _facility(tmp_path, "spear3")
    written = import_mml(_sources("spear3"), facility)
    seeded = {path.relative_to(facility).as_posix() for path in written} - {
        path.relative_to(facility).as_posix() for path in (facility / LAYER_DIR).rglob("*")
    }
    assert seeded == set(_authored(facility))


def test_a_second_import_leaves_every_seeded_file_as_it_is(tree: str, imported: Path) -> None:
    before = _authored(imported)
    mapping = (imported / MAPPING_FILE).read_bytes()
    written = import_mml(_sources(tree), imported)
    assert _authored(imported) == before
    assert (imported / MAPPING_FILE).read_bytes() == mapping
    assert not [path for path in written if LAYER_DIR not in path.as_posix()]


def test_a_file_a_person_wrote_is_never_overwritten(tmp_path: Path) -> None:
    facility = _facility(tmp_path, "synthetic")
    authored = {
        "limits.yaml": "records: []\n",
        "seeds.yaml": "{}\n",
        "classes.yaml": "- {class: SkewQuadrupole, parent: Quadrupole}\n",
        "measurement/SR.yaml": "kinds: []\n",
        "scenarios/readout.yaml": "faults: {}\n",
    }
    for name, text in authored.items():
        (facility / name).parent.mkdir(parents=True, exist_ok=True)
        (facility / name).write_text(text, encoding="utf-8")
    import_mml(_sources("synthetic"), facility)
    for name, text in authored.items():
        assert (facility / name).read_text(encoding="utf-8") == text, name


def test_one_import_judges_each_export_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from osprey.services.mml import judgments

    judged: list[str] = []
    judge = judgments.judged_family_views

    def counted(system: str, *args: Any, **kwargs: Any) -> Any:
        judged.append(system)
        return judge(system, *args, **kwargs)

    monkeypatch.setattr(judgments, "judged_family_views", counted)
    facility = _facility(tmp_path, "nsls2")

    import_mml(_sources("nsls2"), facility)

    assert len(judged) == len(TREES["nsls2"])
    assert len(set(judged)) == len(judged)


# --- limits.yaml --------------------------------------------------------------------


def test_limits_hold_records_and_no_other_key(imported: Path) -> None:
    document = _load(imported / "limits.yaml")
    assert list(document) == ["records"]
    addresses = [row["address"] for row in document["records"]]
    assert addresses == sorted(set(addresses))
    for row in document["records"]:
        assert set(row) <= {"address", "min_value", "max_value", "writable"}, row["address"]


def test_only_setpoints_carry_a_limits_record(imported: Path) -> None:
    roles = _roles(imported)
    assert {roles[address] for address in _limits(imported)} == {"setpoint"}


def test_every_wired_setpoint_is_writable_inside_its_band(imported: Path) -> None:
    roles, limits = _roles(imported), _limits(imported)
    wired = {address for address in _wired(imported) if roles[address] == "setpoint"}
    assert wired
    writable = {address for address, row in limits.items() if row.get("writable") is True}
    assert writable == wired
    for address in wired:
        assert limits[address]["min_value"] <= limits[address]["max_value"], address


def _emitted_setpoints(root: Path, tree: str) -> set[str]:
    """Run the old chain over a tree's exports and return the setpoints it binds.

    A monitor binding carries the address it serves under the same key, so the
    setpoints are the bindings of every other kind.
    """
    pytest.importorskip("linkml_runtime")
    from click.testing import CliRunner

    from osprey.cli.main import cli

    def run(*args: str) -> None:
        result = CliRunner().invoke(cli, [*args, "--repo", str(root)], catch_exceptions=False)
        assert result.exit_code == 0, f"osprey {' '.join(args)}:\n{result.output}"

    root.mkdir(parents=True, exist_ok=True)
    (root / "profile.yml").write_text("name: scratch\n", encoding="utf-8")
    run("mml", "import", *(str(source) for source in _sources(tree)))
    run("mml", "map", "--init")
    shutil.copy(FIXTURES / tree / "mapping.yaml", root / "data" / "mml" / "mapping.yaml")
    run("mml", "emit")
    document = json.loads((root / "data" / "simulation" / "va_bindings.json").read_text())
    return {
        binding["setpoint_address"]
        for binding in document["bindings"]
        if binding["element"] is not None and binding["kind"] != "monitor"
    }


def test_the_writable_set_is_the_setpoints_the_old_chain_binds(
    spear3: Path, tmp_path: Path
) -> None:
    writable = {address for address, row in _limits(spear3).items() if row.get("writable") is True}
    assert writable == _emitted_setpoints(tmp_path / "old", "spear3") | {ENERGY_KNOB}
    assert len(writable) == 300


def test_a_setpoint_an_earlier_read_field_named_is_writable_inside_its_band(spear3: Path) -> None:
    assert _roles(spear3)["SPEAR:RFFreqSetpt"] == "setpoint"
    assert _limits(spear3)["SPEAR:RFFreqSetpt"] == {
        "address": "SPEAR:RFFreqSetpt",
        "min_value": 0.0,
        "max_value": 2500.0,
        "writable": True,
    }


def test_an_unwired_setpoint_carries_its_band_only(synthetic: Path) -> None:
    limits = _limits(synthetic)
    assert "QK:IDGAP:1:CUR:SP" not in _wired(synthetic)
    assert limits["QK:IDGAP:1:CUR:SP"] == {
        "address": "QK:IDGAP:1:CUR:SP",
        "min_value": 0.0,
        "max_value": 60.0,
    }
    assert limits["QK:QF:1:CUR:SP"] == {
        "address": "QK:QF:1:CUR:SP",
        "min_value": 0.0,
        "max_value": 200.0,
        "writable": True,
    }


def test_a_band_is_the_range_as_stated_never_widened_to_the_nominal(synthetic: Path) -> None:
    va = json.loads((FIXTURES / "synthetic" / "quokka.sr.va.json").read_text(encoding="utf-8"))
    nominal = va["families"]["HC"]["nominals"]["Setpoint"]["values"][0]
    row = _limits(synthetic)[OUTSIDE]
    assert (row["min_value"], row["max_value"]) == (-1.0, 1.0)
    assert nominal > row["max_value"]


@pytest.mark.parametrize(
    ("declared", "indices", "devices", "expected"),
    [
        ([0, 200], [0], 2, (0.0, 200.0)),
        ([200, 0], [1], 2, (0.0, 200.0)),
        ([[0, 100], [0, 200]], [1], 2, (0.0, 200.0)),
        ([[0, 100], [-5, 200]], [0, 1], 2, (0.0, 100.0)),
        (["-Inf", 5], [0], 1, (None, 5.0)),
        ([[0, 100], ["NaN", "NaN"]], [1], 2, (None, None)),
        ([[0, 100]], [1], 2, (None, None)),
        (None, [0], 1, (None, None)),
    ],
)
def test_band_reads_a_flat_pair_or_the_rows_of_the_devices_on_the_address(
    declared: Any, indices: list[int], devices: int, expected: tuple[Any, Any]
) -> None:
    assert band(declared, indices, devices) == expected


def test_the_build_stops_on_a_nominal_outside_its_range(synthetic: Path) -> None:
    with pytest.raises(FacilityBuildError) as stop:
        build_facility(synthetic, project_name="demo")
    assert (stop.value.kind, stop.value.record_kind, stop.value.record_id) == (
        "seed-invalid",
        "channel",
        OUTSIDE,
    )
    assert "limits.yaml" in stop.value.sources


def test_the_planted_nominal_is_the_only_stop_of_the_synthetic_tree(synthetic: Path) -> None:
    report = run_stages(synthetic, project_name="demo", later=LATER_STAGES)
    assert [(error.kind, error.record_kind, error.record_id) for error in report.errors] == [
        ("seed-invalid", "channel", OUTSIDE)
    ]


def test_an_unseeded_setpoint_whose_band_holds_zero_is_no_stop(synthetic: Path) -> None:
    row = _limits(synthetic)["QK:IDGAP:1:CUR:SP"]
    assert row["min_value"] <= 0.0 <= row["max_value"]
    assert "QK:IDGAP:1:CUR:SP" not in _load(synthetic / "seeds.yaml")


def test_a_later_import_reports_each_differing_range_and_applies_none(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    facility = _import(tmp_path, "synthetic")
    before = _authored(facility)

    def widen(document: dict[str, Any]) -> None:
        document["IDGAP"]["Setpoint"]["Range"] = [0, 80]

    ao = _edited(tmp_path, "synthetic", "quokka.sr", widen)
    capsys.readouterr()

    import_mml([ao], facility)

    lines = [line for line in capsys.readouterr().out.splitlines() if "limits differ" in line]
    assert lines == [
        "limits differ: QK:IDGAP:1:CUR:SP file [0,60] export [0,80]",
        "limits differ: QK:IDGAP:2:CUR:SP file [0,60] export [0,80]",
    ]
    assert _authored(facility) == before


def test_a_wired_setpoint_without_a_band_on_both_edges_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def unband(document: dict[str, Any]) -> None:
        document["QF"]["Setpoint"]["Range"] = ["NaN", "NaN"]
        document["SF"]["Setpoint"]["Range"] = [0, "Inf"]
        document["IDGAP"]["Setpoint"]["Range"] = ["NaN", "NaN"]

    ao = _edited(tmp_path, "synthetic", "quokka.sr", unband)
    facility = _facility(tmp_path, "synthetic")
    capsys.readouterr()

    import_mml([ao], facility)

    lines = [line for line in capsys.readouterr().out.splitlines() if "limits unbanded" in line]
    wired = _wired(facility)
    quads = sorted(address for address in wired if address.startswith("QK:QF:") and "SP" in address)
    sexts = sorted(address for address in wired if address.startswith("QK:SF:") and "SP" in address)
    assert quads and sexts
    assert lines == sorted(
        [f"limits unbanded: {address} export [-,-]" for address in quads]
        + [f"limits unbanded: {address} export [0,-]" for address in sexts]
    )
    limits = _limits(facility)
    for address in quads:
        assert limits[address] == {"address": address, "writable": False}
    assert "QK:IDGAP:1:CUR:SP" not in limits
    for address in sexts:
        assert limits[address] == {"address": address, "min_value": 0.0, "writable": False}


def test_an_import_beside_a_limits_file_reports_no_unbanded_setpoint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def unband(document: dict[str, Any]) -> None:
        document["QF"]["Setpoint"]["Range"] = ["NaN", "NaN"]

    ao = _edited(tmp_path, "synthetic", "quokka.sr", unband)
    facility = _facility(tmp_path, "synthetic")
    (facility / "limits.yaml").write_text("records: []\n", encoding="utf-8")
    capsys.readouterr()

    import_mml([ao], facility)

    assert "limits unbanded" not in capsys.readouterr().out


# --- seeds.yaml ---------------------------------------------------------------------


def test_seeds_hold_golden_values_of_unwired_channels_only(imported: Path) -> None:
    seeds = _load(imported / "seeds.yaml")
    assert seeds
    assert list(seeds) == sorted(seeds)
    assert not set(seeds) & _wired(imported)
    assert set(seeds) <= set(_roles(imported))
    for address, seed in seeds.items():
        assert list(seed) == ["nominal"], address
        assert isinstance(seed["nominal"], float), address


def test_a_seed_is_the_nominal_the_export_states_for_its_device(synthetic: Path) -> None:
    va = json.loads((FIXTURES / "synthetic" / "quokka.sr.va.json").read_text(encoding="utf-8"))
    values = va["families"]["BSOFT"]["nominals"]["Setpoint"]["values"]
    seeds = _load(synthetic / "seeds.yaml")
    assert [seeds[f"QK:BSOFT:{n}:CUR:SP"]["nominal"] for n in (1, 2)] == values


def test_a_nominal_that_is_no_number_is_no_seed(synthetic: Path) -> None:
    va = json.loads((FIXTURES / "synthetic" / "quokka.sr.va.json").read_text(encoding="utf-8"))
    assert va["families"]["IDGAP"]["nominals"]["Setpoint"]["values"] == ["NaN", "NaN"]
    assert not [address for address in _load(synthetic / "seeds.yaml") if "IDGAP" in address]


def test_a_nominal_in_physics_units_is_no_seed(synthetic: Path) -> None:
    va = json.loads((FIXTURES / "synthetic" / "quokka.sr.va.json").read_text(encoding="utf-8"))
    assert va["families"]["SEPTUM"]["nominals"]["Monitor"]["units"] == "Physics"
    assert not [address for address in _load(synthetic / "seeds.yaml") if "SEPTUM" in address]


def test_the_import_counts_the_wired_channels_whose_golden_value_it_skips(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    facility = _facility(tmp_path, "synthetic")
    import_mml(_sources("synthetic"), facility)
    lines = [line for line in capsys.readouterr().out.splitlines() if "golden" in line]
    va = json.loads((FIXTURES / "synthetic" / "quokka.sr.va.json").read_text(encoding="utf-8"))
    wired = _wired(facility)
    stated = sum(
        1
        for name in ("QF", "QD", "SF", "SQ", "HC", "VC", "RF", "BPMx", "BPMy", "BEND")
        for nominal in va["families"][name]["nominals"].values()
        for value in (
            nominal["values"] if isinstance(nominal["values"], list) else [nominal["values"]]
        )
        if isinstance(value, (int, float))
    )
    assert wired
    assert lines == [f"golden skipped: {stated} wired channels"]


# --- measurement/<model>.yaml -------------------------------------------------------


def test_every_wired_model_has_a_measurement_file(imported: Path) -> None:
    wired = sorted(
        model["name"]
        for model in _load(imported / LAYER_DIR / "models.yaml")
        if model.get("wiring")
    )
    assert sorted(path.stem for path in (imported / "measurement").glob("*.yaml")) == wired


def test_a_measurement_file_carries_the_step_and_settle_keys(imported: Path) -> None:
    for path in sorted((imported / "measurement").glob("*.yaml")):
        document = _load(path)
        assert {key: document[key] for key in TUNING} == TUNING, path.name
        assert list(document)[:2] == ["kinds", "groups"], path.name
    assert TUNING == {
        "n_step": 5,
        "n_avg_meas": 1,
        "fit_order": 2,
        "singular_values": 16,
        "sleep_between_step": 0.0,
        "sleep_between_meas": 0.0,
        "corrector_delta": 1.0e-5,
        "quad_delta": 1.0e-3,
        "sextu_delta": 1.0e-2,
        "frequency_delta": 100.0,
    }


def test_a_periodic_model_measures_orbit_response_and_dispersion(synthetic: Path) -> None:
    document = _load(synthetic / "measurement" / "SR.yaml")
    assert document["kinds"] == ["orm", "dispersion"]
    assert document["groups"] == {
        "bpm": "BPMx",
        "hcor": "HC",
        "vcor": "VC",
        "quad": "QF",
        "sext": "SF",
    }
    assert document["instruments"] == {"rf": "QK:RF:1:CUR:SP"}


def test_every_group_and_instrument_of_a_measurement_file_exists(imported: Path) -> None:
    groups = {group["id"] for group in _load(imported / LAYER_DIR / "groups.yaml")}
    channels = set(_roles(imported))
    for path in sorted((imported / "measurement").glob("*.yaml")):
        document = _load(path)
        assert set(document["groups"].values()) <= groups, path.name
        assert set(document.get("instruments", {}).values()) <= channels, path.name


def test_a_single_pass_model_measures_orbit_response_at_most(tmp_path: Path) -> None:
    facility = _import(tmp_path, "nsls2")
    document = _load(facility / "measurement" / "LTB.yaml")
    assert document["kinds"] == []
    assert document["groups"] == {"hcor": "HCM", "vcor": "VCM", "quad": "Q"}
    assert "instruments" not in document


def test_the_spear3_measurement_reads_the_tune_on_its_wired_readback(spear3: Path) -> None:
    document = _load(spear3 / "measurement" / "StorageRing.yaml")
    assert document["instruments"]["tune"] == "MeasTune"
    assert document["groups"]["quad"] == "QF"
    assert document["kinds"] == ["orm", "dispersion", "trm"]


def test_the_import_names_the_families_a_group_role_was_not_seeded_from(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _import(tmp_path, "spear3")
    lines = [line for line in capsys.readouterr().out.splitlines() if "seeded from" in line]
    assert lines == [
        "measurement StorageRing: quad seeded from QF; also wired: QD, QFC, QDX, QFX, QDY, "
        "QFY, QDZ, QFZ, Q9S",
        "measurement StorageRing: sext seeded from SF; also wired: SD, SFM, SDM",
    ]


# --- scenarios/readout.yaml ---------------------------------------------------------

#: Millimetres per metre: the spear3 monitors publish in mm, the deck solves in m.
MM_PER_M = 1000.0


def _readout_rows(facility: Path) -> dict[str, dict[str, float]]:
    document = _load(facility / READOUT_FILE)
    assert list(document) == ["description", "faults"]
    (rows,) = document["faults"].values()
    return rows


def test_the_spear3_offsets_are_in_the_conversion_and_seed_no_scenario(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    facility = _import(tmp_path, "spear3")
    assert not [line for line in capsys.readouterr().out.splitlines() if "readout" in line]
    assert not (facility / READOUT_FILE).exists()


def test_a_gain_beside_an_offset_the_conversion_holds_is_not_carried(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exports = tmp_path / "exports"
    exports.mkdir()
    for stem in TREES["spear3"]:
        for source in sorted((FIXTURES / "spear3").glob(f"{stem}.*")):
            shutil.copyfile(source, exports / source.name)
    va = exports / "spear3.storagering.va.json"
    document = json.loads(va.read_text(encoding="utf-8"))
    document["families"]["BPMx"]["Monitor"]["readout"]["gain"][0] = 2.0
    va.write_text(json.dumps(document), encoding="utf-8")

    facility = _facility(tmp_path, "spear3")
    import_mml([exports / source.name for source in _sources("spear3")], facility)

    lines = [line for line in capsys.readouterr().out.splitlines() if "readout" in line]
    assert lines == ["import mml: readout not carried: StorageRing: family BPMx (gain)"]
    assert not (facility / READOUT_FILE).exists()


def test_the_middle_layer_correction_of_a_served_spear3_reading_is_the_model_position(
    tmp_path: Path,
) -> None:
    from osprey.facility.views.simulator import simulator_wiring
    from osprey.simulation.engines import pyat as engine

    va = json.loads((FIXTURES / "spear3" / "spear3.storagering.va.json").read_text("utf-8"))
    ao = json.loads((FIXTURES / "spear3" / "spear3.storagering.ao.json").read_text("utf-8"))
    stated = {
        address: (gain, offset)
        for family in ("BPMx", "BPMy")
        for address, gain, offset in zip(
            ao[family]["Monitor"]["ChannelNames"],
            va["families"][family]["Monitor"]["readout"]["gain"],
            va["families"][family]["Monitor"]["readout"]["offset"],
            strict=True,
        )
    }
    assert sum(offset != 0 for _, offset in stated.values()) == 107

    facility = _import(tmp_path, "spear3")
    _widen(facility, WIDENED["spear3"])
    document = build_facility(facility, project_name="demo")
    (model,) = [entry for entry in document["models"] if entry["name"] == "StorageRing"]
    wiring = simulator_wiring(document, model["name"])
    deck = facility / model["deck"]
    monitors = {
        record["address"]: record
        for record in wiring
        if "axis" in record["engine"] and "attribute" not in record["engine"]
    }
    assert set(stated) <= set(monitors)
    assert {monitors[address]["unit"] for address in stated} == {"mm"}

    seeded = _readout_rows(facility) if (facility / READOUT_FILE).exists() else {}
    applied = {
        f"{address}/{name}": value
        for address, fields in seeded.items()
        for name, value in fields.items()
    }

    def read(records: list[dict[str, Any]], active: dict[str, float]) -> dict[str, float]:
        built = engine.build(model["name"], records, deck, model.get("settings"), active)
        return engine.readout(built, built.get(sorted(stated)), 0)

    raw = read(wiring, applied)
    position = read(
        [
            {key: value for key, value in record.items() if key != "calibration"}
            if record["address"] in monitors
            else record
            for record in wiring
        ],
        {},
    )
    # The Middle Layer corrects a reading as Gain x (Raw - Offset); with whatever
    # readout the import seeded applied, the served reading carries the offset once.
    for address, (gain, offset) in stated.items():
        corrected = gain * (raw[address] - offset)
        assert corrected == pytest.approx(MM_PER_M * position[address], abs=1e-9)


def test_a_readout_the_engine_cannot_carry_is_named_and_not_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    facility = _import(tmp_path, "synthetic")
    lines = [line for line in capsys.readouterr().out.splitlines() if "readout" in line]
    # The quokka conversions hold neither the stated gain nor the stated offset.
    assert lines == [
        "import mml: readout not carried: SR: family BPMx (crunch, gain, offset)",
        "import mml: readout not carried: SR: family BPMy (crunch, gain, roll)",
    ]
    rows = _readout_rows(facility)
    assert rows
    assert {field for fields in rows.values() for field in fields} == {"roll"}
    assert not [address for address in rows if "Y" in address]


# --- classes.yaml -------------------------------------------------------------------


def test_classes_hold_one_row_per_mapping_class_the_vocabulary_lacks(
    tree: str, imported: Path
) -> None:
    mapping = read_mapping(FIXTURES / tree / MAPPING_FILE)
    vocabulary = known_classes()
    expected = {
        family.class_: family.branch
        for family in mapping.families.values()
        if family.class_ is not None and family.class_ not in vocabulary
    }
    rows = _load(imported / "classes.yaml")
    assert {row["class"]: row["parent"] for row in rows} == expected
    assert [row["class"] for row in rows] == sorted(expected)
    assert all(list(row) == ["class", "parent"] for row in rows)


def test_the_classes_of_the_storage_tree_land(spear3: Path) -> None:
    rows = {row["class"]: row["parent"] for row in _load(spear3 / "classes.yaml")}
    assert rows == {
        "BeamlineMonitor": "Instrumentation",
        "BendTrim": "Corrector",
        "CorrectorCurrentReference": "Corrector",
        "InjectionKicker": "Magnet",
        "InjectionSeptum": "Magnet",
        "MachineStatus": "Instrumentation",
        "OrbitInterlock": "Instrumentation",
        "QuadrupoleShunt": "Instrumentation",
        "SkewQuadrupole": "Quadrupole",
        "TuneMonitor": "Instrumentation",
    }


def test_a_declared_branch_is_a_row_ahead_of_the_classes_that_extend_it(tmp_path: Path) -> None:
    facility = _facility(tmp_path, "synthetic")
    path = facility / MAPPING_FILE
    document = _load(path)
    document["branches"] = {
        "Trim": {"parent": "Steering", "description": None},
        "Steering": {"parent": "Corrector", "description": None},
    }
    document["families"]["BDM"]["branch"] = "Trim"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    import_mml(_sources("synthetic"), facility)

    rows = [(row["class"], row["parent"]) for row in _load(facility / "classes.yaml")]
    assert rows[:2] == [("Steering", "Corrector"), ("Trim", "Steering")]
    assert ("BendTrim", "Trim") in rows[2:]
    assert [name for name, _ in rows[2:]] == sorted(name for name, _ in rows[2:])


# --- identity.yaml ------------------------------------------------------------------


def test_identity_is_seeded_from_the_facility_block_which_then_leaves_the_mapping(
    spear3: Path,
) -> None:
    block = _load(FIXTURES / "spear3" / MAPPING_FILE)["facility"]
    assert _load(spear3 / "identity.yaml") == block
    assert (block["code"], block["name"]) == ("SPEAR3", "SPEAR3")
    mapping = _load(spear3 / MAPPING_FILE)
    assert "facility" not in mapping
    assert mapping == {
        key: value
        for key, value in _load(FIXTURES / "spear3" / MAPPING_FILE).items()
        if key != "facility"
    }
    assert read_mapping(spear3 / MAPPING_FILE).identity is None


def test_removing_the_facility_block_keeps_every_other_line_of_the_mapping(tmp_path: Path) -> None:
    facility = _facility(tmp_path, "spear3")
    path = facility / MAPPING_FILE
    stated = path.read_text(encoding="utf-8")
    assert stated.startswith("facility:\n")
    models = stated.index("models:\n")
    block, rest = stated[:models], stated[models:]
    rest = rest.replace("  StorageRing:\n", "  StorageRing:  # the one model\n", 1)
    assert "# the one model" in rest
    rest += "# Last line.\n"
    path.write_text(f"# Reviewed by hand.\n{block}{rest}", encoding="utf-8")

    import_mml(_sources("spear3"), facility)

    assert (
        _load(facility / "identity.yaml") == _load(FIXTURES / "spear3" / MAPPING_FILE)["facility"]
    )
    assert path.read_text(encoding="utf-8") == f"# Reviewed by hand.\n{rest}"


def test_a_facility_block_in_another_spelling_still_leaves_the_mapping(tmp_path: Path) -> None:
    facility = _facility(tmp_path, "spear3")
    path = facility / MAPPING_FILE
    stated = path.read_text(encoding="utf-8")
    models = stated.index("models:\n")
    path.write_text(f'"facility": {{code: LAB}}\n{stated[models:]}', encoding="utf-8")

    import_mml(_sources("spear3"), facility)

    assert _load(facility / "identity.yaml") == {"code": "LAB"}
    assert _load(path) == yaml.safe_load(stated[models:])


def test_a_mapping_without_a_facility_block_seeds_no_identity(tmp_path: Path) -> None:
    facility = _import(tmp_path, "nsls2")
    assert "facility" not in _load(FIXTURES / "nsls2" / MAPPING_FILE)
    assert not (facility / "identity.yaml").exists()
    assert (facility / MAPPING_FILE).read_bytes() == (
        FIXTURES / "nsls2" / MAPPING_FILE
    ).read_bytes()


def test_a_facility_block_beside_an_identity_file_is_reported_and_left(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    facility = _facility(tmp_path, "spear3")
    (facility / "identity.yaml").write_text("code: LAB\n", encoding="utf-8")
    mapping = (facility / MAPPING_FILE).read_bytes()

    import_mml(_sources("spear3"), facility)

    assert "facility: block ignored; identity.yaml exists" in capsys.readouterr().out.splitlines()
    assert (facility / "identity.yaml").read_text(encoding="utf-8") == "code: LAB\n"
    assert (facility / MAPPING_FILE).read_bytes() == mapping


# --- the build ----------------------------------------------------------------------

#: The devices each tree's build stops on while ``classes.yaml`` is not seeded.
UNKNOWN_CLASS_DEVICES = {"spear3": 130, "nsls2": 32}

#: The setpoints each tree's build starts outside the band its export states,
#: each with the edge that widens its limits record to hold the build's
#: operating point. A wired setpoint starts where its calibration puts the
#: deck's strength. For spear3 that is the nominal the export states, outside
#: the export's own ``Range``. nsls2 widens nothing.
WIDENED: dict[str, dict[str, tuple[str, float]]] = {
    "spear3": {
        "09S-QD1:CurrSetpt": ("min_value", -60.0),
        "MS1-BDMT:CurrSetpt": ("max_value", 600.0),
    },
    "nsls2": {},
}

#: The nsls2 transport quadrupoles the deck holds at the other sign from the
#: export: the export states their currents positive and inside the band, the
#: deck holds them at a negative strength and their curve has a positive gain.
POLARITY = tuple(f"LTB-MG{{Quad:{number}}}I:Sp1-SP" for number in (1, 3, 4, 6, 9, 11, 14))


def _widen(facility: Path, edges: dict[str, tuple[str, float]]) -> None:
    """Apply the ``seed-invalid`` remedy: widen each named limits record by hand."""
    path = facility / "limits.yaml"
    document = _load(path)
    for row in document["records"]:
        if row["address"] in edges:
            key, value = edges[row["address"]]
            row[key] = value
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def test_a_transport_quadrupole_the_deck_holds_the_other_way_carries_its_polarity(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    facility = _import(tmp_path, "nsls2")
    ao = json.loads((FIXTURES / "nsls2" / "nsls2.ltb.ao.json").read_text(encoding="utf-8"))
    va = json.loads((FIXTURES / "nsls2" / "nsls2.ltb.va.json").read_text(encoding="utf-8"))
    addresses = [address.strip() for address in ao["Q"]["Setpoint"]["ChannelNames"]]
    stated = dict(
        zip(addresses, va["families"]["Q"]["nominals"]["Setpoint"]["values"], strict=True)
    )
    low, high = ao["Q"]["Setpoint"]["Range"]
    records = {
        record["address"]: record
        for model in _load(facility / LAYER_DIR / "models.yaml")
        for record in model.get("wiring", [])
    }
    assert [line for line in capsys.readouterr().out.splitlines() if "polarity" in line] == [
        f"import mml: polarity: LTB: family Q device {addresses.index(address) + 1}; "
        "the deck holds the other sign"
        for address in POLARITY
    ]
    for address in addresses:
        record = records[address]
        if address in POLARITY:
            assert [piece["weight"] for piece in record["slices"]] == [-1.0], address
            assert "element" not in record, address
        else:
            assert "slices" not in record, address
        assert record["calibration"]["curve"]["linear"]["gain"] > 0, address
        assert low <= stated[address] <= high, address

    document = build_facility(facility, project_name="demo")
    defaults = {
        record["address"]: record["default"]
        for model in document["models"]
        for record in model.get("wiring", [])
    }
    for address in POLARITY:
        assert defaults[address] == pytest.approx(stated[address], rel=1e-9), address


@pytest.mark.parametrize("name", BUILT)
def test_the_build_exits_clean_once_classes_are_seeded_and_each_named_band_is_widened(
    name: str, tmp_path: Path
) -> None:
    facility = _facility(tmp_path, name)
    exports = read_exports(_sources(name))
    write_records(exports, read_mapping(facility / MAPPING_FILE), facility)

    report = run_stages(facility, project_name="demo", later=LATER_STAGES)

    assert Counter(error.kind for error in report.errors) == {
        "class-unknown": UNKNOWN_CLASS_DEVICES[name]
    }

    import_mml(_sources(name), facility)

    seeded = run_stages(facility, project_name="demo", later=LATER_STAGES)
    assert sorted((error.kind, error.record_kind, error.record_id) for error in seeded.errors) == [
        ("seed-invalid", "channel", address) for address in sorted(WIDENED[name])
    ]
    for error in seeded.errors:
        assert "limits.yaml" in error.sources, error.record_id
        assert "widen the limits record" in error.remedy, error.record_id
    if WIDENED[name]:
        with pytest.raises(FacilityBuildError):
            build_facility(facility, project_name="demo")

    _widen(facility, WIDENED[name])

    document = build_facility(facility, project_name="demo")
    assert document is not None
    assert not run_stages(facility, project_name="demo", later=LATER_STAGES).errors
