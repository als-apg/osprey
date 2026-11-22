"""The computed wiring slots, filled from each model's channel, limits and engine."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
from typing import Any

import at
import pytest
import yaml

from osprey.facility import wiring as wiring_module
from osprey.facility.validate import StageReport, run_stages
from osprey.facility.wiring import fill_wiring_slots
from osprey_connectors.simulation.engines import ENGINE_GROUP

GAIN = 0.37
OFFSET = -0.012
K_A = 1.234
K_B = -0.456
SLICES = [{"element": "A", "weight": 2}, {"element": "B", "weight": 1}]
ENGINE = {"attribute": "PolynomB", "index": 1}
CALIBRATION = {"curve": {"linear": {"gain": GAIN, "offset": OFFSET}}}
SLOTS = ("direction", "unit", "default", "value_range", "engine_role", "plane", "refresh")


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
    plug = {
        "name": "plug",
        "engine": "pyat",
        "wiring": [{"address": "T:X", "element": "BPM1", "engine": {"axis": "x"}}],
    }
    return {
        "records/devices.yaml": [{"id": "SR/Q1", "class": "Quadrupole"}],
        "records/channels.yaml": channels,
        "models.yaml": [sr, plug, *(models or [])],
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


class TestDefaultThroughAReadbacksOwnCalibration:
    """A readback stating another calibration than the setpoint on its target."""

    VOLT_GAIN, VOLT_OFFSET = 0.02, 0.003

    def _voltage(self, **calibration: Any) -> dict[str, Any]:
        own = {"curve": {"linear": {"gain": self.VOLT_GAIN, "offset": self.VOLT_OFFSET}}}
        return {
            "address": "U:RB",
            "slices": SLICES,
            "engine": ENGINE,
            "calibration": {**own, **calibration},
        }

    def _filled(self, tmp_path: Path, record: dict[str, Any]) -> StageReport:
        files = _files()
        sr = files["models.yaml"][0]
        sr["wiring"] = [r for r in sr["wiring"] if r["address"] != "U:RB"] + [record]
        return _run(tmp_path, files)

    def test_it_takes_the_setpoints_physics_value_through_its_own(self, tmp_path):
        result = self._filled(tmp_path, self._voltage())
        assert result.ok, _lines(result)
        expected = (K_A / 2 - self.VOLT_OFFSET) / self.VOLT_GAIN
        assert _records(result)["SR/U:RB"]["default"] == pytest.approx(expected, abs=1e-9)

    def test_stating_the_setpoints_calibration_it_starts_at_the_setpoint(self, tmp_path):
        result = self._filled(tmp_path, {**self._voltage(), "calibration": CALIBRATION})
        assert result.ok, _lines(result)
        records = _records(result)
        assert records["SR/U:RB"]["default"] == pytest.approx(
            records["SR/Q1:SP"]["default"], abs=1e-12
        )

    def test_a_table_with_no_inverse_stops_on_the_readback(self, tmp_path):
        table = {"table": {"grid": [0.0, 1.0], "values": [0.0, 2.0]}}
        result = self._filled(tmp_path, self._voltage(curve=table))
        assert result.failed == "wiring"
        assert _lines(result) == [
            "facility: engine-invalid: wiring SR/U:RB — a table calibration has no inverse "
            "to serve the reading; fix: add calibration.inverse"
        ]

    def _paired(self, tmp_path: Path, calibration: dict[str, Any], **channel: Any) -> StageReport:
        """The setpoint's own pair, ``Q1:RB``, wired to the setpoint's slices."""
        files = _files()
        sr = files["models.yaml"][0]
        row = {"address": "Q1:RB", "slices": SLICES, "engine": ENGINE, "calibration": calibration}
        sr["wiring"] = [r for r in sr["wiring"] if r["address"] != "Q1:RB"] + [row]
        channels = files["records/channels.yaml"]
        channels[1].update(channel.pop("readback", {}))
        channels.extend(channel.pop("more", []))
        return _run(tmp_path, files)

    @pytest.mark.parametrize(
        "channel",
        [
            pytest.param({}, id="followed"),
            pytest.param({"readback": {"simulation": {"follows": "none"}}}, id="follows-none"),
            pytest.param(
                {"more": [{"id": "Q1:RESET", "role": "setpoint", "pair": "Q1:RB"}]},
                id="shared",
            ),
        ],
    )
    def test_it_starts_through_its_own_whether_or_not_its_pair_is_followed(self, tmp_path, channel):
        result = self._paired(tmp_path, self._voltage()["calibration"], **channel)
        assert result.ok, _lines(result)
        expected = (K_A / 2 - self.VOLT_OFFSET) / self.VOLT_GAIN
        assert _records(result)["SR/Q1:RB"]["default"] == pytest.approx(expected, abs=1e-9)

    def test_a_pair_not_followed_still_starts_at_the_setpoint_on_its_target(self, tmp_path):
        """The model serves it from that setpoint, so that is where it starts."""
        result = self._paired(tmp_path, CALIBRATION, readback={"simulation": {"follows": "none"}})
        assert result.ok, _lines(result)
        records = _records(result)
        assert records["SR/Q1:RB"]["default"] == pytest.approx(
            records["SR/Q1:SP"]["default"], abs=1e-12
        )


class TestDefaultThroughADeckPropertyReadbacksOwnCalibration:
    """A readback of the deck energy stating another calibration than the energy setpoint."""

    ENERGY = {"attribute": "energy"}
    KNOB_GAIN = 1.0e7
    VOLT_GAIN, VOLT_OFFSET = 2.0e6, 1.0e6

    def _voltage(self, **calibration: Any) -> dict[str, Any]:
        own = {
            "curve": {"linear": {"gain": self.VOLT_GAIN, "offset": self.VOLT_OFFSET}},
            "energy_scaling": "none",
        }
        return {"address": "U:RB", "engine": self.ENERGY, "calibration": {**own, **calibration}}

    def _filled(self, tmp_path: Path, record: dict[str, Any], **channel: Any) -> StageReport:
        knob = {
            "address": "K",
            "engine": self.ENERGY,
            "calibration": {"curve": {"linear": {"gain": self.KNOB_GAIN, "offset": 0.0}}},
        }
        files = _files(knob)
        sr = files["models.yaml"][0]
        sr["wiring"] = [r for r in sr["wiring"] if r["address"] != "U:RB"] + [record]
        channels = {row["id"]: row for row in files["records/channels.yaml"]}
        channels["K"].update(channel.pop("setpoint", {}))
        channels["U:RB"].update(channel.pop("readback", {}))
        return _run(tmp_path, files)

    @pytest.mark.parametrize(
        "channel",
        [
            pytest.param({}, id="unpaired"),
            pytest.param({"setpoint": {"pair": "U:RB"}}, id="followed"),
            pytest.param(
                {"setpoint": {"pair": "U:RB"}, "readback": {"simulation": {"follows": "none"}}},
                id="follows-none",
            ),
        ],
    )
    def test_it_takes_the_setpoints_curve_value_through_its_own(self, tmp_path, channel):
        result = self._filled(tmp_path, self._voltage(), **channel)
        assert result.ok, _lines(result)
        records = _records(result)
        curve_value = self.KNOB_GAIN * records["SR/K"]["default"]
        assert records["SR/U:RB"]["default"] == pytest.approx(
            (curve_value - self.VOLT_OFFSET) / self.VOLT_GAIN, abs=1e-9
        )
        assert records["SR/U:RB"]["direction"] == "read"

    def test_stating_the_setpoints_calibration_it_is_a_readback_no_setpoint_pairs_with(
        self, tmp_path
    ):
        """It reads the setpoint that states its engine block, and starts at its value."""
        same = {"curve": {"linear": {"gain": self.KNOB_GAIN, "offset": 0.0}}}
        result = self._filled(tmp_path, {**self._voltage(), "calibration": same})
        assert result.ok, _lines(result)
        records = _records(result)
        record = records["SR/U:RB"]
        assert (record["engine_role"], record["direction"]) == ("readback", "read")
        assert record["default"] == records["SR/K"]["default"]

    def test_a_record_of_the_whole_deck_no_setpoint_drives_is_still_an_output(self, tmp_path):
        files = _files()
        sr = files["models.yaml"][0]
        sr["wiring"] = [r for r in sr["wiring"] if r["address"] != "U:RB"] + [
            {"address": "U:RB", "engine": self.ENERGY}
        ]
        result = _run(tmp_path, files)
        assert _lines(result) == [
            "facility: engine-invalid: model SR — the engine describes U:RB as a readback; it "
            "is a output: it names no element; fix: make the engine's describe() call U:RB a "
            "output"
        ]

        result = self._filled(tmp_path, self._voltage(energy_scaling="brho"))
        assert result.failed == "wiring"
        assert _lines(result) == [
            "facility: engine-invalid: wiring SR/U:RB — U:RB states energy_scaling brho on its "
            "own calibration; a readback of a deck property reads the curve value of its "
            "setpoint K, which no rigidity scales; fix: state energy_scaling: none on the "
            "readback"
        ]

    def test_a_table_with_no_inverse_stops_on_the_readback(self, tmp_path):
        table = {"table": {"grid": [0.0, 1.0], "values": [0.0, 2.0]}}
        result = self._filled(tmp_path, self._voltage(curve=table))
        assert result.failed == "wiring"
        assert _lines(result) == [
            "facility: engine-invalid: wiring SR/U:RB — a table calibration has no inverse "
            "to serve the reading; fix: add calibration.inverse"
        ]


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

    def test_the_setpoint_records_all_seven(self, filled):
        assert _filled_defaults(filled["SR/Q1:SP"]) == sorted(SLOTS)

    def test_a_deck_less_model_gets_direction_unit_and_range(self, filled):
        record = filled["plug/T:X"]
        assert (record["direction"], record["unit"]) == ("read", "mm")
        assert "default" not in record
        assert "value_range" not in record
        assert record["provenance"]["defaults"] == [
            "direction",
            "engine_role",
            "plane",
            "refresh",
            "unit",
        ]
        assert (record["engine_role"], record["plane"], record["refresh"]) == (
            "monitor",
            "x",
            "pass",
        )


class TestDescription:
    def test_each_record_carries_what_the_engine_says_it_is(self, filled):
        described = {
            record_id: (filled[record_id]["engine_role"], filled[record_id]["refresh"])
            for record_id in ("SR/Q1:SP", "SR/Q1:RB", "SR/U:RB")
        }
        assert described == {
            "SR/Q1:SP": ("setpoint", "pass"),
            "SR/Q1:RB": ("readback", "pass"),
            "SR/U:RB": ("readback", "pass"),
        }

    def test_a_record_steering_no_plane_carries_a_null_plane(self, filled):
        assert "plane" in filled["SR/Q1:SP"]
        assert filled["SR/Q1:SP"]["plane"] is None


class TestStops:
    def test_a_record_naming_no_element_is_engine_invalid(self, tmp_path):
        result = _run(tmp_path, _files({"address": "K", "engine": ENGINE}))
        assert result.failed == "wiring"
        assert _lines(result) == [
            "facility: engine-invalid: wiring SR/K — K: no element to read a start value "
            "from; fix: name an element or slices for the record, or leave the channel unwired"
        ]

    def test_a_deck_property_record_reads_its_start_value_from_the_deck(self, tmp_path):
        energy = {
            "address": "K",
            "engine": {"attribute": "energy"},
            "calibration": {"curve": {"linear": {"gain": 1.0e7, "offset": 0.0}}},
        }
        result = _run(tmp_path, _files(energy))
        assert result.ok, _lines(result)
        assert _records(result)["SR/K"]["default"] == pytest.approx(300.0, abs=1e-12)

    def test_an_unknown_deck_property_still_stops(self, tmp_path):
        result = _run(tmp_path, _files({"address": "K", "engine": {"attribute": "voltage"}}))
        assert _lines(result) == [
            "facility: engine-invalid: wiring SR/K — K: no element to read a start value "
            "from; fix: name an element or slices for the record, or leave the channel unwired"
        ]

    def test_a_stop_on_a_fix_added_record_names_fixes_yaml(self, tmp_path):
        files = _files()
        files["fixes.yaml"] = {
            "schema": "osprey.facility.fixes/1",
            "fixes": [
                {
                    "op": "add",
                    "kind": "wiring",
                    "id": "SR/K",
                    "record": {"address": "K", "engine": ENGINE},
                    "why": "Not exported.",
                }
            ],
        }
        result = _run(tmp_path, files)
        assert result.failed == "wiring"
        assert [(e.kind, e.record_id, e.sources) for e in result.errors] == [
            ("engine-invalid", "SR/K", ("fixes.yaml",))
        ]

    def test_a_stop_on_a_models_yaml_record_names_models_yaml(self, tmp_path):
        result = _run(tmp_path, _files({"address": "K", "engine": ENGINE}))
        assert [(e.kind, e.record_id, e.sources) for e in result.errors] == [
            ("engine-invalid", "SR/K", ("models.yaml",))
        ]

    def test_an_engine_stop_on_a_record_names_the_file_stating_it(self, tmp_path):
        table = {"curve": {"table": {"grid": [0.0, 1.0], "values": [0.0, 2.0]}}}
        record = {"address": "K", "element": "A", "engine": ENGINE, "calibration": table}
        result = _run(tmp_path, _files(record))
        assert [(e.kind, e.record_id, e.sources) for e in result.errors] == [
            ("engine-invalid", "SR/K", ("models.yaml",))
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

    def test_a_record_the_engine_cannot_describe_stops_on_the_record(self, tmp_path):
        files = _files()
        files["models.yaml"][0]["wiring"][2] = {"address": "U:RB", "element": "B"}
        result = _run(tmp_path, files)
        assert [(e.kind, e.record_kind, e.record_id, e.sources) for e in result.errors] == [
            ("engine-invalid", "wiring", "SR/U:RB", ("models.yaml",))
        ]
        assert _lines(result) == [
            "facility: engine-invalid: wiring SR/U:RB — U:RB is no setpoint, monitor, readback "
            "or optics output; fix: name an element, slices or an engine block the engine describes"
        ]
        setpoint = _records(result)["SR/Q1:SP"]
        assert "default" in setpoint
        assert setpoint["engine_role"] == "setpoint"

    def test_an_engine_that_contradicts_the_file_stops(self, tmp_path, monkeypatch):
        from osprey.simulation.engines import pyat

        describe = pyat.describe

        def write_as_readback(record: Any) -> Any:
            found = dict(describe(record))
            if record.get("direction") == "write":
                found["engine_role"] = "readback"
            return found

        monkeypatch.setattr(pyat, "describe", write_as_readback)
        result = _run(tmp_path, _files())
        assert [(e.kind, e.record_kind, e.record_id) for e in result.errors] == [
            ("engine-invalid", "model", "SR")
        ]
        assert result.errors[0].remedy == "make the engine's describe() call Q1:SP a setpoint"

    def test_an_engine_without_describe_stops_the_model(self, tmp_path, monkeypatch):
        from osprey.simulation.engines import pyat

        monkeypatch.delattr(pyat, "describe")
        files = _files()
        files["models.yaml"] = files["models.yaml"][:1]
        result = _run(tmp_path, files)
        assert [(e.kind, e.record_kind, e.record_id) for e in result.errors] == [
            ("engine-invalid", "model", "SR")
        ]
        assert result.errors[0].remedy == "add describe() to the engine plug-in"

    def test_a_deck_less_physics_model_with_an_unregistered_engine_is_engine_missing(
        self, tmp_path
    ):
        files = _files()
        files["models.yaml"][1]["engine"] = "no_such_engine"
        result = _run(tmp_path, files)
        assert [(e.kind, e.record_kind, e.record_id) for e in result.errors] == [
            ("engine-missing", "model", "plug")
        ]


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
        assert groups == [ENGINE_GROUP]


def test_an_authored_computed_slot_is_never_overwritten() -> None:
    record = {"id": "w1", "default": 3.0}
    with pytest.raises(
        RuntimeError, match=r"wiring w1 already carries computed slots \['default'\]"
    ):
        wiring_module._fill_record(record, {"engine_role": "setpoint"}, None, 0.0)


@pytest.mark.parametrize(
    ("slot", "value"), [("engine_role", "monitor"), ("plane", "x"), ("refresh", "pass")]
)
def test_an_authored_description_slot_is_refused(tmp_path: Path, slot: str, value: str) -> None:
    files = _files()
    files["models.yaml"][0]["wiring"][2] = {
        "address": "U:RB",
        "element": "B",
        "engine": ENGINE,
        slot: value,
    }
    result = _run(tmp_path, files)
    assert [(e.kind, e.record_id) for e in result.errors] == [("source-invalid", "SR/U:RB")]
    assert f"`{slot}` is computed by the build" in _lines(result)[0]
