"""The computed wiring slots, filled from each model's channel, limits and engine."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
from typing import Any

import pytest
import yaml

at = pytest.importorskip("at")

from osprey.facility import wiring as wiring_module  # noqa: E402
from osprey.facility.validate import StageReport, run_stages  # noqa: E402
from osprey.facility.wiring import fill_wiring_slots  # noqa: E402

GAIN = 0.37
OFFSET = -0.012
K_A = 1.234
K_B = -0.456
SLICES = [{"element": "A", "weight": 2}, {"element": "B", "weight": 1}]
ENGINE = {"attribute": "PolynomB", "index": 1}
CALIBRATION = {"curve": {"linear": {"gain": GAIN, "offset": OFFSET}}}
SLOTS = ("direction", "unit", "default", "value_range")


def _save_deck(path: Path) -> None:
    """Elements A at s 1.0-1.2 and B at s 3.0-3.2, one monitor at the end."""
    lattice = at.Lattice(
        [
            at.Drift("D0", 1.0),
            at.Quadrupole("A", 0.2, K_A),
            at.Drift("D1", 1.8),
            at.Quadrupole("B", 0.2, K_B),
            at.Drift("D2", 0.5),
            at.Monitor("BPM1"),
        ],
        energy=3e9,
        particle="electron",
        periodicity=1,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    at.save_lattice(lattice, str(path))


def _setpoint() -> dict[str, Any]:
    return {
        "address": "Q1:SP",
        "slices": SLICES,
        "engine": ENGINE,
        "calibration": CALIBRATION,
    }


def _files(*extra_wiring: dict[str, Any], models: list[dict[str, Any]] | None = None):
    channels = [
        {
            "id": "Q1:SP",
            "role": "setpoint",
            "pair": "Q1:RB",
            "unit": "A",
            "on": {"device": "SR/Q1"},
        },
        {"id": "Q1:RB", "unit": "A", "on": {"device": "SR/Q1"}},
        {"id": "U:RB", "on": {"device": "SR/Q1"}},
        {"id": "K", "role": "setpoint", "on": {"device": "SR/Q1"}},
        {"id": "T:X", "unit": "mm"},
    ]
    sr = {
        "name": "SR",
        "engine": "pyat",
        "deck": "decks/sr.json",
        "wiring": [
            _setpoint(),
            {"address": "Q1:RB", "element": "A", "engine": ENGINE},
            {"address": "U:RB", "element": "B", "engine": ENGINE},
            *extra_wiring,
        ],
    }
    texture = {"name": "texture", "engine": "texture", "wiring": [{"address": "T:X"}]}
    return {
        "records/devices.yaml": [{"id": "SR/Q1", "class": "Quadrupole"}],
        "records/channels.yaml": channels,
        "models.yaml": [sr, texture, *(models or [])],
        "limits.yaml": {"records": [{"address": "Q1:SP", "min_value": -3, "max_value": 4.5}]},
    }


def _run(tmp_path: Path, files: dict[str, Any]) -> StageReport:
    root = tmp_path / "facility"
    for rel, data in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    _save_deck(root / "decks" / "sr.json")
    return run_stages(root, project_name="p", later=[("wiring", fill_wiring_slots)])


def _lines(result: StageReport) -> list[str]:
    return [error.format_message() for error in result.errors]


def _records(result: StageReport) -> dict[str, dict[str, Any]]:
    document = result.validated.document
    assert document is not None
    return {record["id"]: record for model in document["models"] for record in model["wiring"]}


def _filled_defaults(record: dict[str, Any]) -> list[str]:
    """The computed slots a record's provenance says the build filled."""
    return [slot for slot in record["provenance"]["defaults"] if slot in SLOTS]


@pytest.fixture
def filled(tmp_path: Path) -> dict[str, dict[str, Any]]:
    result = _run(tmp_path, _files())
    assert result.ok, _lines(result)
    return _records(result)


class TestDefault:
    def test_the_setpoint_takes_the_first_slice_inverted(self, filled):
        expected = (K_A / 2 - OFFSET) / GAIN
        assert filled["SR/Q1:SP"]["default"] == pytest.approx(expected, abs=1e-12)

    def test_the_paired_readback_takes_its_setpoint_value(self, filled):
        assert filled["SR/Q1:RB"]["default"] == filled["SR/Q1:SP"]["default"]

    def test_an_unpaired_readback_starts_at_zero(self, filled):
        assert filled["SR/U:RB"]["default"] == 0.0


class TestDirection:
    def test_a_setpoint_is_written(self, filled):
        assert filled["SR/Q1:SP"]["direction"] == "write"

    def test_a_readback_is_read(self, filled):
        assert filled["SR/Q1:RB"]["direction"] == "read"
        assert filled["SR/U:RB"]["direction"] == "read"


class TestUnit:
    def test_the_channel_unit_is_carried(self, filled):
        assert filled["SR/Q1:SP"]["unit"] == "A"
        assert filled["SR/Q1:RB"]["unit"] == "A"

    def test_a_channel_without_a_unit_leaves_it_absent(self, filled):
        assert "unit" not in filled["SR/U:RB"]


class TestValueRange:
    def test_the_limits_record_bounds_the_range(self, filled):
        value_range = filled["SR/Q1:SP"]["value_range"]
        assert value_range == [-3.0, 4.5]
        assert all(type(bound) is float for bound in value_range)

    def test_no_limits_record_leaves_it_absent(self, filled):
        assert "value_range" not in filled["SR/Q1:RB"]
        assert "value_range" not in filled["SR/U:RB"]

    def test_a_record_with_one_bound_leaves_it_absent(self, tmp_path):
        files = _files()
        files["limits.yaml"] = {"records": [{"address": "Q1:SP", "min_value": -3}]}
        result = _run(tmp_path, files)
        assert result.ok, _lines(result)
        assert "value_range" not in _records(result)["SR/Q1:SP"]


class TestProvenance:
    def test_every_filled_slot_is_recorded(self, filled):
        for record_id in ("SR/Q1:SP", "SR/Q1:RB", "SR/U:RB"):
            record = filled[record_id]
            written = sorted(slot for slot in SLOTS if slot in record)
            assert _filled_defaults(record) == written, record_id

    def test_the_setpoint_records_all_four(self, filled):
        assert _filled_defaults(filled["SR/Q1:SP"]) == sorted(SLOTS)

    def test_a_deck_less_model_is_untouched(self, filled):
        record = filled["texture/T:X"]
        assert not set(SLOTS) & set(record)
        assert _filled_defaults(record) == []


class TestStops:
    def test_a_record_naming_no_element_is_engine_invalid(self, tmp_path):
        result = _run(tmp_path, _files({"address": "K", "engine": ENGINE}))
        assert result.failed == "wiring"
        assert _lines(result) == [
            "facility: engine-invalid: wiring SR/K — K: no element to read a start value "
            "from; fix: name an element or slices for the record, or leave the channel unwired"
        ]

    def test_the_other_records_are_still_filled(self, tmp_path):
        result = _run(tmp_path, _files({"address": "K", "engine": ENGINE}))
        records = _records(result)
        assert "default" in records["SR/Q1:SP"]
        assert not set(SLOTS) & set(records["SR/K"])

    def test_an_unregistered_engine_is_engine_missing(self, tmp_path):
        other = {
            "name": "LINE",
            "engine": "no_such_engine",
            "deck": "decks/sr.json",
            "wiring": [{"address": "T:X", "element": "A"}],
        }
        files = _files(models=[other])
        files["models.yaml"][1]["wiring"] = []
        result = _run(tmp_path, files)
        assert result.failed == "wiring"
        assert [(e.kind, e.record_kind, e.record_id) for e in result.errors] == [
            ("engine-missing", "model", "LINE")
        ]
        assert "no_such_engine" in _lines(result)[0]


class TestEntryPoint:
    def test_the_engine_is_found_through_the_entry_point_group(self, tmp_path, monkeypatch):
        groups: list[str | None] = []
        real = metadata.entry_points

        def spy(**kwargs: Any) -> Any:
            groups.append(kwargs.get("group"))
            return real(**kwargs)

        monkeypatch.setattr(wiring_module.metadata, "entry_points", spy)
        result = _run(tmp_path, _files())
        assert result.ok, _lines(result)
        assert groups == ["osprey.simulation.engines"]


def test_an_authored_computed_slot_is_never_overwritten() -> None:
    record = {"id": "w1", "default": 3.0}
    with pytest.raises(
        RuntimeError, match=r"wiring w1 already carries computed slots \['default'\]"
    ):
        wiring_module._fill_record(record, 0.0, {"role": "setpoint"}, None)
