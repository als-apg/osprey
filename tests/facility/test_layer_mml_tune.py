"""The mml layer's tune block: a model's tune readback wired to the engine's tunes.

The spear3 tree's mapping states a waveform block, ``MeasTune`` read on the
planes x, y and s; each case imports the tree under that mapping, or under an
edit of it, into a fresh ``data/facility/``.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.facility.layers.mml.importer import LAYER_DIR, import_mml
from osprey.facility.layers.mml.mapping import (
    MAPPING_FILE,
    MappingError,
    TuneBlock,
    parse_mapping,
    read_mapping,
)

at = pytest.importorskip("at")

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "mml"
SPEAR3 = FIXTURES / "spear3"
EXPORT = SPEAR3 / "spear3.storagering.ao.json"
MODEL = "StorageRing"


def _facility(root: Path, tune: Any = None) -> Path:
    facility = root / "data" / "facility"
    target = facility / MAPPING_FILE
    target.parent.mkdir(parents=True)
    if tune is None:
        shutil.copyfile(SPEAR3 / MAPPING_FILE, target)
        return facility
    document = yaml.safe_load((SPEAR3 / MAPPING_FILE).read_text(encoding="utf-8"))
    document["models"][MODEL]["tune"] = tune
    target.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return facility


def _rows(facility: Path, name: str) -> list[dict[str, Any]]:
    return yaml.safe_load((facility / LAYER_DIR / name).read_text(encoding="utf-8"))


def _wiring(facility: Path) -> dict[str, dict[str, Any]]:
    (model,) = [row for row in _rows(facility, "models.yaml") if row["name"] == MODEL]
    return {record["address"]: record for record in model["wiring"]}


def _channels(facility: Path) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in _rows(facility, "channels.yaml")}


@pytest.fixture(scope="module")
def spear3(tmp_path_factory: pytest.TempPathFactory) -> Path:
    facility = _facility(tmp_path_factory.mktemp("spear3-tune"))
    import_mml([EXPORT], facility)
    return facility


def test_the_spear3_mapping_reads_measured_tune_as_one_waveform() -> None:
    tune = read_mapping(SPEAR3 / MAPPING_FILE).models[MODEL].tune
    assert tune == TuneBlock(
        planes={"x": "MeasTune", "y": "MeasTune", "s": "MeasTune"}, address="MeasTune"
    )


def test_the_waveform_block_types_the_channel_the_family_wrote(spear3: Path) -> None:
    channel = _channels(spear3)["MeasTune"]
    assert (channel["role"], channel["value_type"], channel["shape"]) == (
        "readback",
        "waveform",
        [3],
    )
    assert "on" in channel or "endpoint_of" in channel


def test_the_waveform_block_wires_one_record_to_the_whole_output(spear3: Path) -> None:
    assert _wiring(spear3)["MeasTune"] == {"address": "MeasTune", "engine": {"attribute": "tune"}}


def test_a_scalar_block_wires_one_record_per_plane(tmp_path: Path) -> None:
    facility = _facility(tmp_path, {"x": "SPEAR:TuneX", "y": "SPEAR:TuneY"})
    import_mml([EXPORT], facility)
    wiring = _wiring(facility)
    assert wiring["SPEAR:TuneX"]["engine"] == {"attribute": "tune", "index": 0}
    assert wiring["SPEAR:TuneY"]["engine"] == {"attribute": "tune", "index": 1}
    channels = _channels(facility)
    for address in ("SPEAR:TuneX", "SPEAR:TuneY"):
        assert channels[address]["role"] == "readback"
        assert "value_type" not in channels[address]
    assert "value_type" not in channels["MeasTune"]


def test_a_tune_address_a_family_wires_stops_the_import(tmp_path: Path) -> None:
    from osprey.facility.layers.mml.mapping import ImportStop

    facility = _facility(tmp_path, {"address": "SPEAR:RFFreqSetpt", "planes": ["x", "y"]})
    with pytest.raises(ImportStop) as stop:
        import_mml([EXPORT], facility)
    assert stop.value.format_message() == (
        "import mml: export-invalid: StorageRing: address SPEAR:RFFreqSetpt is wired by a "
        "family and by the tune block"
    )


@pytest.mark.parametrize(
    ("tune", "message"),
    [
        ({"address": "T", "planes": ["y", "x"]}, "must be [x, y] or [x, y, s], got [y, x]"),
        ({"address": "T", "planes": ["x"]}, "must be [x, y] or [x, y, s], got [x]"),
        ({"x": "T", "z": "U"}, "unknown key"),
        ({}, "must name an address per plane, or an address and its planes"),
    ],
)
def test_a_tune_block_off_its_shape_is_refused(tune: dict[str, Any], message: str) -> None:
    document = yaml.safe_load((SPEAR3 / MAPPING_FILE).read_text(encoding="utf-8"))
    document["models"][MODEL]["tune"] = tune
    with pytest.raises(MappingError, match=message.replace("[", r"\[").replace("]", r"\]")):
        parse_mapping(document)


def test_the_built_spear3_tree_serves_measured_tune_as_the_deck_s_tunes(tmp_path: Path) -> None:
    from osprey.facility.build import build_facility
    from osprey.facility.views.simulator import simulator_wiring
    from osprey.simulation.engines import pyat as engine
    from tests.facility.test_mml_layer_seed_once import WIDENED, _widen

    facility = _facility(tmp_path)
    import_mml([EXPORT], facility)
    _widen(facility, WIDENED["spear3"])
    document = build_facility(facility, project_name="demo")
    (model,) = [entry for entry in document["models"] if entry["name"] == MODEL]
    wiring = simulator_wiring(document, MODEL)
    deck = facility / model["deck"]

    built = engine.build(MODEL, wiring, deck, model.get("settings"))
    served = built.get(["MeasTune"])["MeasTune"]

    assert built.supported_variables["MeasTune"].read_only
    expected = at.get_optics(built.lattice.deepcopy(), get_chrom=False)[1].tune
    assert served.shape == (3,) == expected.shape
    assert served == pytest.approx(expected, abs=1e-12)


def test_the_built_measured_tune_starts_from_a_waveform_of_its_shape(tmp_path: Path) -> None:
    from osprey.facility.build import build_facility
    from osprey.facility.views.simulator import simulator_wiring
    from tests.facility.test_mml_layer_seed_once import WIDENED, _widen

    values = pytest.importorskip("osprey_connectors.simulation.values")
    facility = _facility(tmp_path)
    import_mml([EXPORT], facility)
    _widen(facility, WIDENED["spear3"])
    document = build_facility(facility, project_name="demo")
    (record,) = [r for r in simulator_wiring(document, MODEL) if r["address"] == "MeasTune"]

    assert record["default"] == [0.0, 0.0, 0.0]
    assert values.coerce(record["default"], "waveform", None, [3]) is not None
