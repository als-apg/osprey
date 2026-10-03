"""The demo generator's ``models.yaml`` against the virtual accelerator's bindings.

``scripts/facility_demo/generate.py`` writes the demo's one deck model, ``SR``,
with a wiring record per address the virtual accelerator's bindings wire, plus
the optics readbacks and one cavity's frequency pair. These tests read the
model back from the YAML the generator writes and hold it to the bindings and
to the committed deck.
"""

from __future__ import annotations

import json
import shutil
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from osprey.facility.sources import load_sources, read_yaml
from tests.facility.test_generator_records import (
    CA_DATA,
    VA_BINDINGS,
    generated,
    generated_files,
    generator,
)

SR_DECK = CA_DATA / "facility/decks/SR.json"

OPTICS = {
    "SR:DIAG:CHROM:X": {"attribute": "chromaticity", "axis": "x"},
    "SR:DIAG:CHROM:Y": {"attribute": "chromaticity", "axis": "y"},
    "SR:DIAG:TUNE:X": {"attribute": "tune", "axis": "x"},
    "SR:DIAG:TUNE:Y": {"attribute": "tune", "axis": "y"},
}
CAVITY_PAIR = ("SR:RF:CAVITY:01:FREQUENCY:RB", "SR:RF:CAVITY:01:FREQUENCY:SP")
UNWIRED_CAVITY_PAIR = ("SR:RF:CAVITY:02:FREQUENCY:RB", "SR:RF:CAVITY:02:FREQUENCY:SP")


@cache
def bindings() -> list[dict[str, Any]]:
    """The virtual accelerator's bindings."""
    document = json.loads(VA_BINDINGS.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = document["bindings"]
    return rows


def models() -> list[dict[str, Any]]:
    """``models.yaml`` as the facility loader parses it."""
    rows: list[dict[str, Any]] = generated("models.yaml")
    return rows


def wiring() -> dict[str, dict[str, Any]]:
    """The ``SR`` model's wiring records keyed by address."""
    return {record["address"]: record for record in models()[0]["wiring"]}


def roles() -> dict[str, str]:
    """Each generated channel's role, schema default filled."""
    return {c["id"]: c.get("role", "readback") for c in generated("records/channels.yaml")}


def test_models_yaml_is_the_one_deck_model() -> None:
    [model] = models()
    assert {k: v for k, v in model.items() if k != "wiring"} == {
        "name": "SR",
        "engine": "pyat",
        "deck": "decks/SR.json",
        "settings": {"pyat": {"solve": "periodic"}},
    }


def test_wiring_is_the_bound_addresses_plus_the_six_sorted() -> None:
    addresses = [record["address"] for record in models()[0]["wiring"]]
    bound = {
        binding[key]
        for binding in bindings()
        for key in ("setpoint_address", "readback_address")
        if binding[key]
    }
    assert len(bound) == 840
    assert set(addresses) == bound | set(OPTICS) | set(CAVITY_PAIR)
    assert len(addresses) == 846
    assert addresses == sorted(addresses)
    assert set(addresses) <= set(roles())
    assert not set(addresses) & set(UNWIRED_CAVITY_PAIR)


def test_wiring_records_state_no_id_and_no_computed_slot() -> None:
    allowed = {"address", "element", "engine", "calibration"}
    for address, record in wiring().items():
        assert set(record) <= allowed, address


def test_bound_records_re_express_their_binding() -> None:
    records = wiring()
    for binding in bindings():
        engine: dict[str, Any]
        if binding["kind"] == "monitor":
            engine = {"axis": binding["attribute"]}
        else:
            engine = {"attribute": binding["attribute"]}
            if binding["index"] is not None:
                engine["index"] = binding["index"]
        curve = binding["calibration"]
        calibration = {
            "curve": {"linear": {"gain": curve["gain"], "offset": curve["offset"]}},
            "energy_scaling": binding["energy_scaling"],
        }
        expected = {"element": binding["element"], "engine": engine, "calibration": calibration}
        for key in ("setpoint_address", "readback_address"):
            if binding[key]:
                record = records[binding[key]]
                assert {k: v for k, v in record.items() if k != "address"} == expected, record


def test_monitor_inverses_are_the_algebraic_inverse_of_their_curve() -> None:
    for binding in bindings():
        inverse = binding["monitor_inverse"]
        if inverse is None:
            continue
        curve = binding["calibration"]
        assert inverse["kind"] == curve["kind"] == "linear"
        assert inverse["gain"] * curve["gain"] == 1.0
        assert inverse["offset"] == curve["offset"] == 0.0


def test_optics_readbacks_read_the_solve_on_their_axis() -> None:
    records = wiring()
    channel_roles = roles()
    for address, engine in OPTICS.items():
        assert records[address] == {"address": address, "engine": engine}
        assert channel_roles[address] == "readback"


def test_one_cavity_pair_drives_the_deck_cavity_frequency_in_mhz() -> None:
    at = pytest.importorskip("at")

    [cavity] = [e for e in at.load_lattice(str(SR_DECK)) if isinstance(e, at.RFCavity)]
    records = wiring()
    for address in CAVITY_PAIR:
        assert records[address] == {
            "address": address,
            "element": cavity.FamName,
            "engine": {"attribute": "Frequency"},
            "calibration": {
                "curve": {"linear": {"gain": 1000000.0, "offset": 0.0}},
                "energy_scaling": "none",
            },
        }


def test_start_values_equal_the_binding_nominals() -> None:
    pytest.importorskip("at")
    from osprey.simulation.engines import pyat

    channel_roles = roles()
    records = list(wiring().values())
    readbacks = {r["address"]: None for r in records if channel_roles[r["address"]] != "setpoint"}
    values = pyat.start_values(SR_DECK, records, models()[0]["settings"], readbacks=readbacks)
    nominal = {
        b["setpoint_address"]: b["nominal"]
        for b in bindings()
        if channel_roles[b["setpoint_address"]] == "setpoint"
    }
    assert len(nominal) == 348
    for address, expected in nominal.items():
        assert "slices" not in wiring()[address]
        assert values[address] == pytest.approx(expected, abs=1e-9), address
    assert values[CAVITY_PAIR[1]] == pytest.approx(500.41692828147894, abs=1e-9)


@pytest.mark.parametrize(
    ("where", "record_kind", "record_id"),
    [("model", "model", "SR"), ("wiring", "wiring", "SR/SR:DIAG:TUNE:X")],
)
def test_models_yaml_rejects_a_kind_key(
    tmp_path: Path, where: str, record_kind: str, record_id: str
) -> None:
    assert generator().main(["--out", str(tmp_path)]) == 0
    document = read_yaml(generated_files()["models.yaml"])
    if where == "model":
        document[0]["kind"] = "deck"
    else:
        next(r for r in document[0]["wiring"] if r["address"] == "SR:DIAG:TUNE:X")["kind"] = "tune"
    (tmp_path / "models.yaml").write_text(generator().dump(document), encoding="utf-8")
    errors = load_sources(tmp_path).errors
    assert [(e.kind, e.record_kind, e.record_id, e.detail) for e in errors] == [
        ("source-invalid", record_kind, record_id, "unknown key `kind`")
    ]


def test_written_tree_fills_every_wiring_slot(tmp_path: Path) -> None:
    pytest.importorskip("at")
    from osprey.facility.validate import run_stages
    from osprey.facility.wiring import fill_wiring_slots

    assert generator().main(["--out", str(tmp_path)]) == 0
    (tmp_path / "decks").mkdir()
    shutil.copyfile(SR_DECK, tmp_path / "decks/SR.json")
    report = run_stages(tmp_path, project_name="ca", later=(("wiring", fill_wiring_slots),))
    assert report.errors == []
