"""Stage S6 and the in-memory build, on synthetic decks."""

from __future__ import annotations

import copy
import io
from pathlib import Path
from typing import Any

import pytest
import yaml

at = pytest.importorskip("at")

from osprey.facility import compute  # noqa: E402
from osprey.facility.build import LATER_STAGES, build_facility  # noqa: E402
from osprey.facility.errors import FacilityBuildError  # noqa: E402
from osprey.facility.validate import StageReport, run_stages, validate  # noqa: E402

QUAD = "Quadrupole"
BPM = "BeamPositionMonitor"
SETTING = {"attribute": "PolynomB", "index": 1}
READING = {"attribute": "PolynomB", "index": 0}

#: The periodic deck: QF split in two halves at its start and end.
QF_HALF = 0.25
SR_LENGTH = 3.8


def _sr_deck(path: Path) -> None:
    """QFA 0-0.25, M1, D, QD 1.25-1.55, D, M2 + BPM1 at 2.55, D, M3, QFB 3.55-3.8.

    The three drifts share the name ``D``; nothing wires them.
    """
    _save(
        path,
        [
            at.Quadrupole("QFA", QF_HALF, 1.1),
            at.Marker("M1"),
            at.Drift("D", 1.0),
            at.Quadrupole("QD", 0.3, -1.0),
            at.Drift("D", 1.0),
            at.Marker("M2"),
            at.Monitor("BPM1"),
            at.Drift("D", 1.0),
            at.Marker("M3"),
            at.Quadrupole("QFB", QF_HALF, 1.1),
        ],
    )


def _line_deck(path: Path) -> None:
    """M0 at 0, Q1 at 1.0-1.2, BPM at 2.2."""
    _save(
        path,
        [
            at.Marker("M0"),
            at.Drift("DL", 1.0),
            at.Quadrupole("Q1", 0.2, 0.9),
            at.Drift("DL2", 1.0),
            at.Monitor("LBPM"),
        ],
    )


def _save(path: Path, elements: list[Any]) -> None:
    lattice = at.Lattice(elements, energy=3e9, particle="electron", periodicity=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    at.save_lattice(lattice, str(path))


def _setpoint(address: str, device: str) -> dict[str, Any]:
    return {"id": address, "role": "setpoint", "on": {"device": device}}


def _tree() -> dict[str, Any]:
    """Model SR (periodic) with spans SR > SR/A, SR/B; model LINE (single_pass)."""
    return {
        "records/places.yaml": [
            {"id": "SR", "span": {"model": "SR", "from_marker": "M1"}},
            {"id": "SR/A", "span": {"model": "SR", "from_marker": "M1", "to_marker": "M2"}},
            {"id": "SR/B", "span": {"model": "SR", "from_marker": "M2", "to_marker": "M1"}},
            {"id": "LINE", "span": {"model": "LINE", "from_marker": "M0"}},
        ],
        "records/devices.yaml": [
            {"id": "SR/QF", "class": QUAD},
            {"id": "SR/QD", "class": QUAD},
            {"id": "SR/BPM1", "class": BPM},
            {"id": "LINE/Q1", "class": QUAD},
            {"id": "SR/SPARE", "class": QUAD},
        ],
        "records/channels.yaml": [
            _setpoint("QF:SP", "SR/QF"),
            _setpoint("QD:SP", "SR/QD"),
            {"id": "BPM1:X", "on": {"device": "SR/BPM1"}},
            _setpoint("LQ:SP", "LINE/Q1"),
        ],
        "records/groups.yaml": [{"id": "SR/QUADS", "members": ["SR/QF", "SR/QD"]}],
        "models.yaml": [
            {
                "name": "SR",
                "engine": "pyat",
                "deck": "decks/sr.json",
                "wiring": [
                    {
                        "address": "QF:SP",
                        "slices": [{"element": "QFA"}, {"element": "QFB"}],
                        "engine": SETTING,
                    },
                    {"address": "QD:SP", "element": "QD", "engine": SETTING},
                    {"address": "BPM1:X", "element": "BPM1", "engine": READING},
                ],
            },
            {
                "name": "LINE",
                "engine": "pyat",
                "deck": "decks/line.json",
                "settings": {
                    "pyat": {
                        "solve": "single_pass",
                        "twiss_in": {"beta": [5.0, 3.0], "alpha": [0.0, 0.0]},
                    }
                },
                "wiring": [{"address": "LQ:SP", "element": "Q1", "engine": SETTING}],
            },
        ],
    }


def _write(root: Path, files: dict[str, Any]) -> Path:
    for rel, data in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    _sr_deck(root / "decks" / "sr.json")
    _line_deck(root / "decks" / "line.json")
    return root


def _run(tmp_path: Path, files: dict[str, Any]) -> StageReport:
    return run_stages(_write(tmp_path / "facility", files), project_name="p", later=LATER_STAGES)


def _lines(result: StageReport) -> list[str]:
    return [error.format_message() for error in result.errors]


def _ok(tmp_path: Path, files: dict[str, Any]) -> dict[str, Any]:
    result = _run(tmp_path, files)
    assert result.ok, _lines(result)
    document = result.validated.document
    assert document is not None
    return document


def _devices(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {device["id"]: device for device in document["devices"]}


def _stops(tmp_path: Path, files: dict[str, Any]) -> list[tuple[str, str, str]]:
    result = _run(tmp_path, files)
    assert result.failed == "compute", _lines(result)
    return [(e.kind, e.record_kind, e.record_id) for e in result.errors]


def _model(files: dict[str, Any], name: str) -> dict[str, Any]:
    return next(model for model in files["models.yaml"] if model["name"] == name)


@pytest.fixture
def built(tmp_path: Path) -> dict[str, dict[str, Any]]:
    return _devices(_ok(tmp_path, _tree()))


class TestPositions:
    def test_a_split_magnet_takes_the_short_arc_through_the_end(self, built):
        qf = built["SR/QF"]
        assert qf["model"] == "SR"
        assert qf["s"] == pytest.approx(SR_LENGTH - QF_HALF, abs=1e-12)
        assert qf["length"] == pytest.approx(2 * QF_HALF, abs=1e-12)

    def test_one_element_gives_its_entrance_and_length(self, built):
        assert (built["SR/QD"]["s"], built["SR/QD"]["length"]) == pytest.approx((1.25, 0.3))

    def test_a_single_pass_model_takes_lowest_entrance_to_highest_exit(self, tmp_path):
        files = _tree()
        files["records/devices.yaml"].append({"id": "LINE/PAIR", "class": QUAD})
        files["records/channels.yaml"].append(_setpoint("LP:SP", "LINE/PAIR"))
        _model(files, "LINE")["wiring"].append(
            {
                "address": "LP:SP",
                "slices": [{"element": "Q1"}, {"element": "M0"}],
                "engine": SETTING,
            }
        )
        pair = _devices(_ok(tmp_path, files))["LINE/PAIR"]
        assert (pair["s"], pair["length"]) == pytest.approx((0.0, 1.2), abs=1e-12)

    def test_a_device_without_wiring_has_no_position(self, built):
        assert not {"model", "s", "length"} & set(built["SR/SPARE"])

    def test_repeated_names_of_unwired_elements_build(self, built):
        assert built["SR/BPM1"]["s"] == pytest.approx(2.55, abs=1e-12)

    def test_computed_slots_are_recorded(self, built):
        defaults = built["SR/QD"]["provenance"]["defaults"]
        assert {"model", "s", "length", "ordinalInModel", "ordinalInPlace"} <= set(defaults)


class TestSpans:
    def test_the_deepest_containing_span_places_the_device(self, built):
        assert built["SR/QD"]["place"] == "SR/A"
        assert built["SR/QD"]["provenance"]["place_from"] == "span"

    def test_a_span_wraps_on_a_periodic_model(self, built):
        assert built["SR/QF"]["place"] == "SR/B"

    def test_spans_are_half_open(self, built):
        assert built["SR/BPM1"]["place"] == "SR/B"

    def test_a_span_places_only_its_own_model_devices(self, built):
        assert built["LINE/Q1"]["s"] == pytest.approx(1.0, abs=1e-12)
        assert built["LINE/Q1"]["place"] == "LINE"

    def test_overlapping_spans_of_one_level_stop(self, tmp_path):
        files = _tree()
        files["records/places.yaml"].append(
            {"id": "SR/C", "span": {"model": "SR", "from_marker": "M3"}}
        )
        assert _stops(tmp_path, files) == [("span-invalid", "place", "SR/C")]

    def test_an_absent_marker_stops(self, tmp_path):
        files = _tree()
        files["records/places.yaml"][1]["span"]["to_marker"] = "NOPE"
        result = _run(tmp_path, files)
        assert _lines(result) == [
            "facility: span-invalid: place SR/A — `span.to_marker`: element NOPE is not in "
            "the deck of model SR; fix: name a marker that appears once in the deck"
        ]

    def test_a_repeated_marker_stops(self, tmp_path):
        files = _tree()
        files["records/places.yaml"][1]["span"]["to_marker"] = "D"
        assert _stops(tmp_path, files) == [("span-invalid", "place", "SR/A")]

    def test_a_span_on_a_deckless_model_stops(self, tmp_path):
        files = _tree()
        files["models.yaml"].append({"name": "PLAIN", "engine": "pyat", "wiring": []})
        files["records/places.yaml"].append(
            {"id": "OTHER", "span": {"model": "PLAIN", "from_marker": "M0"}}
        )
        assert _stops(tmp_path, files) == [("span-invalid", "place", "OTHER")]

    def test_a_backwards_span_on_a_single_pass_model_stops(self, tmp_path):
        files = _tree()
        files["records/places.yaml"][3]["span"] = {
            "model": "LINE",
            "from_marker": "LBPM",
            "to_marker": "M0",
        }
        assert _stops(tmp_path, files) == [("span-invalid", "place", "LINE")]

    def test_validate_prints_the_span_stop(self, tmp_path):
        files = _tree()
        files["records/places.yaml"][1]["span"]["to_marker"] = "NOPE"
        stream = io.StringIO()
        code = validate(_write(tmp_path / "facility", files), project_name="p", file=stream)
        assert code == 1
        assert stream.getvalue().splitlines() == [
            "facility: span-invalid: place SR/A — `span.to_marker`: element NOPE is not in "
            "the deck of model SR; fix: name a marker that appears once in the deck"
        ]


class TestPlaces:
    def test_an_ancestor_hand_place_is_refined(self, tmp_path):
        files = _tree()
        files["records/devices.yaml"][1]["place"] = "SR"
        qd = _devices(_ok(tmp_path, files))["SR/QD"]
        assert qd["place"] == "SR/A"
        assert qd["provenance"]["place_from"] == "span"

    def test_an_authored_contradiction_builds_and_is_recorded(self, tmp_path):
        files = _tree()
        files["records/devices.yaml"][1]["place"] = "SR/B"
        qd = _devices(_ok(tmp_path, files))["SR/QD"]
        assert qd["place"] == "SR/B"
        assert qd["provenance"]["place_from"] == "authored"

    def test_an_imported_contradiction_stops(self, tmp_path):
        files = _tree()
        files["imported/mml/devices.yaml"] = [{"id": "SR/QD", "place": "SR/B"}]
        result = _run(tmp_path, files)
        assert _lines(result) == [
            "facility: place-conflict: device SR/QD — layer mml states place SR/B, but the "
            "span of place SR/A holds the device at s 1.25 in model SR; fix: drop `place` "
            "from the layer, or add a fix `set` of place SR/B"
        ]

    def test_a_fix_keeps_an_imported_place(self, tmp_path):
        files = _tree()
        files["imported/mml/devices.yaml"] = [{"id": "SR/QD", "place": "SR/B"}]
        files["fixes.yaml"] = {
            "schema": "osprey.facility.fixes/1",
            "fixes": [
                {
                    "op": "set",
                    "kind": "device",
                    "id": "SR/QD",
                    "fields": {"place": "SR/B"},
                    "was": {"place": {"mml": "SR/B"}},
                    "why": "The magnet is counted with its downstream neighbours.",
                }
            ],
        }
        qd = _devices(_ok(tmp_path, files))["SR/QD"]
        assert qd["place"] == "SR/B"
        assert qd["provenance"]["place_from"] == "fix"
        assert [fix["op"] for fix in qd["provenance"]["fixes"]] == ["set"]

    def test_a_device_without_s_or_place_has_no_place_from(self, built):
        assert "place" not in built["SR/SPARE"]
        assert "place_from" not in built["SR/SPARE"]["provenance"]


class TestOrdinals:
    def test_per_class_in_model_and_in_place(self, built):
        assert built["SR/QD"]["ordinalInModel"] == 1
        assert built["SR/QF"]["ordinalInModel"] == 2
        assert built["SR/QD"]["ordinalInPlace"] == 1
        assert built["SR/QF"]["ordinalInPlace"] == 1

    def test_no_ordinal_crosses_models(self, built):
        assert built["LINE/Q1"]["ordinalInModel"] == 1

    def test_a_device_without_s_has_no_ordinal(self, built):
        assert not {"ordinalInModel", "ordinalInPlace"} & set(built["SR/SPARE"])

    def test_ties_break_by_id(self):
        devices = [
            {"id": "B", "class": QUAD, "model": "SR", "s": 1.0, "place": "SR"},
            {"id": "A", "class": QUAD, "model": "SR", "s": 1.0, "place": "SR"},
            {"id": "C", "class": BPM, "model": "SR", "s": 0.5},
        ]
        assert compute.compute_ordinals(devices) == {
            "A": {"ordinalInModel": 1, "ordinalInPlace": 1},
            "B": {"ordinalInModel": 2, "ordinalInPlace": 2},
            "C": {"ordinalInModel": 1},
        }


class TestGroups:
    def test_a_device_lists_the_groups_naming_it(self, built):
        assert built["SR/QD"]["groups"] == ["SR/QUADS"]
        assert "groups" not in built["SR/BPM1"]

    def test_resolve_groups_sorts(self):
        groups = [{"id": "Z", "members": ["A"]}, {"id": "Y", "members": ["A", "B"]}]
        assert compute.resolve_groups(groups) == {"A": ["Y", "Z"], "B": ["Y"]}


class TestWiringConflicts:
    def test_an_address_wired_by_two_models(self, tmp_path):
        files = _tree()
        _model(files, "LINE")["wiring"].append(
            {"address": "QD:SP", "element": "Q1", "engine": SETTING}
        )
        assert ("wiring-conflict", "channel", "QD:SP") in _stops(tmp_path, files)

    def test_a_device_wired_in_two_models(self, tmp_path):
        files = _tree()
        files["records/channels.yaml"].append(_setpoint("QD2:SP", "SR/QD"))
        _model(files, "LINE")["wiring"].append(
            {"address": "QD2:SP", "element": "Q1", "engine": SETTING}
        )
        assert _stops(tmp_path, files) == [("wiring-conflict", "device", "SR/QD")]

    def test_a_wired_element_repeated_in_its_deck(self, tmp_path):
        files = _tree()
        _model(files, "SR")["wiring"][0]["slices"].append({"element": "D"})
        assert _stops(tmp_path, files) == [("wiring-conflict", "wiring", "SR/QF:SP")]


def _wire_element(files: dict[str, Any], where: str, element: str) -> str:
    """Point one wiring record's element at ``element``; return the record's id."""
    wiring = _model(files, "SR")["wiring"]
    if where == "element":
        wiring[1]["element"] = element
        return "SR/QD:SP"
    if where == "first slice":
        wiring[0]["slices"][0]["element"] = element
        return "SR/QF:SP"
    if where == "later slice":
        wiring[0]["slices"].append({"element": element})
        return "SR/QF:SP"
    wiring[2]["element"] = element
    return "SR/BPM1:X"


_PLACES = ("element", "first slice", "later slice", "readback")


class TestWiredElementStops:
    """One line for a wired element, whatever slice or role it sits on."""

    @pytest.mark.parametrize("where", _PLACES)
    def test_a_repeated_element_is_a_wiring_conflict(self, tmp_path, where):
        files = _tree()
        record = _wire_element(files, where, "D")
        assert _lines(_run(tmp_path, files)) == [
            f"facility: wiring-conflict: wiring {record} — element D appears 3 times in the "
            "deck of model SR; fix: give the element a unique name in the deck"
        ]

    @pytest.mark.parametrize("where", _PLACES)
    def test_a_missing_element_names_the_wiring_record(self, tmp_path, where):
        files = _tree()
        record = _wire_element(files, where, "GHOST")
        assert _lines(_run(tmp_path, files)) == [
            f"facility: engine-invalid: wiring {record} — element GHOST is not in the deck "
            "of model SR; fix: name an element the deck holds"
        ]


class TestEngineStops:
    def test_the_locate_stop_names_the_wiring_record_not_the_deck(self, tmp_path):
        files = _tree()
        sr = _model(files, "SR")
        sr["deck"] = "decks/lattice_v2.json"
        sr["wiring"][0]["slices"].append({"element": "GHOST"})
        root = _write(tmp_path / "facility", files)
        _sr_deck(root / "decks" / "lattice_v2.json")
        result = run_stages(root, project_name="p", later=LATER_STAGES)
        assert _lines(result) == [
            "facility: engine-invalid: wiring SR/QF:SP — element GHOST is not in the deck "
            "of model SR; fix: name an element the deck holds"
        ]

    def test_prepare_runs_for_every_deck(self, tmp_path):
        files = _tree()
        _model(files, "LINE")["settings"]["pyat"]["solve"] = "sideways"
        result = _run(tmp_path, files)
        assert [(e.kind, e.record_id) for e in result.errors] == [("engine-invalid", "LINE")]


class TestModelConflicts:
    def test_a_declared_texture_model(self, tmp_path):
        files = _tree()
        files["models.yaml"].append({"name": "texture", "engine": "texture", "wiring": []})
        assert _stops(tmp_path, files) == [("model-conflict", "model", "texture")]

    def test_a_channel_on_a_status_address(self, tmp_path):
        files = _tree()
        files["records/channels.yaml"].append({"id": "p:SIM:SR:STATUS"})
        assert _stops(tmp_path, files) == [("model-conflict", "channel", "p:SIM:SR:STATUS")]

    def test_texture_is_listed_last(self, tmp_path):
        document = _ok(tmp_path, _tree())
        assert [m["name"] for m in document["models"]] == ["LINE", "SR", "texture"]
        assert document["models"][-1] == {"name": "texture", "engine": "texture"}


class TestNominalBand:
    def test_a_default_outside_its_band_stops(self, tmp_path):
        files = _tree()
        files["limits.yaml"] = {"records": [{"address": "QD:SP", "min_value": 0, "max_value": 1}]}
        result = _run(tmp_path, files)
        assert [(e.kind, e.record_id) for e in result.errors] == [("seed-invalid", "QD:SP")]
        assert "below `min_value` 0" in result.errors[0].detail

    def test_an_unwired_nominal_outside_its_band_stops(self, tmp_path):
        files = _tree()
        files["records/channels.yaml"].append({"id": "T:X", "simulation": {"nominal": 7.0}})
        files["limits.yaml"] = {"records": [{"address": "T:X", "min_value": 0, "max_value": 5}]}
        result = _run(tmp_path, files)
        assert [(e.kind, e.record_id) for e in result.errors] == [("seed-invalid", "T:X")]
        assert "nominal 7 lies above `max_value` 5" in result.errors[0].detail
        assert result.errors[0].sources == ("limits.yaml", "records/channels.yaml")

    def test_a_nominal_inside_its_band_builds(self, tmp_path):
        files = _tree()
        files["limits.yaml"] = {"records": [{"address": "QD:SP", "min_value": -2, "max_value": 0}]}
        _ok(tmp_path, files)


# --- the in-memory build on the synthetic A/B slice deck ------------------------------

GAIN = 0.37
OFFSET = -0.012
K_A = 1.234
K_B = -0.456
INVERTED = (K_A / 2 - OFFSET) / GAIN
TWISS_IN = {"beta": [5.0, 3.0], "alpha": [0.0, 0.0]}


def _ab_tree(root: Path, **extra: Any) -> Path:
    """Elements A at s 1.0-1.2 and B at 3.0-3.2 of a line, sliced into one setpoint."""
    files: dict[str, Any] = {
        "records/devices.yaml": [{"id": "SR/Q1", "class": QUAD}],
        "records/channels.yaml": [
            {"id": "Q1:SP", "role": "setpoint", "pair": "Q1:RB", "on": {"device": "SR/Q1"}},
            {"id": "Q1:RB", "on": {"device": "SR/Q1"}},
        ],
        "models.yaml": [
            {
                "name": "SR",
                "engine": "pyat",
                "deck": "decks/ab.json",
                "settings": {"pyat": {"solve": "single_pass", "twiss_in": TWISS_IN}},
                "wiring": [
                    {
                        "address": "Q1:SP",
                        "slices": [{"element": "A", "weight": 2}, {"element": "B"}],
                        "engine": SETTING,
                        "calibration": {"curve": {"linear": {"gain": GAIN, "offset": OFFSET}}},
                    }
                ],
            }
        ],
    }
    files.update(extra)
    for rel, data in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    _save(
        root / "decks" / "ab.json",
        [
            at.Drift("D0", 1.0),
            at.Quadrupole("A", 0.2, K_A),
            at.Drift("D1", 1.8),
            at.Quadrupole("B", 0.2, K_B),
            at.Drift("D2", 0.5),
            at.Monitor("BPM1"),
        ],
    )
    return root


def _with_readback_nominal(root: Path, nominal: float) -> Path:
    tree = _ab_tree(root)
    channels = yaml.safe_load((tree / "records/channels.yaml").read_text())
    channels[1]["simulation"] = {"nominal": nominal}
    (tree / "records/channels.yaml").write_text(yaml.safe_dump(channels), encoding="utf-8")
    return tree


class TestBuildFacility:
    def test_the_wiring_default_is_the_inverted_first_slice(self, tmp_path):
        document = build_facility(_ab_tree(tmp_path / "facility"), project_name="p")
        record = document["models"][0]["wiring"][0]
        assert record["default"] == pytest.approx(INVERTED, abs=1e-12)
        device = document["devices"][0]
        assert (device["s"], device["length"]) == pytest.approx((1.0, 2.2), abs=1e-12)

    def test_a_nominal_outside_the_band_around_the_default_stops(self, tmp_path):
        band = {"records": [{"address": "Q1:RB", "min_value": INVERTED + 1, "max_value": 9}]}
        root = _ab_tree(tmp_path / "facility", **{"limits.yaml": band})
        with pytest.raises(FacilityBuildError) as caught:
            build_facility(root, project_name="p")
        assert (caught.value.kind, caught.value.record_id) == ("seed-invalid", "Q1:RB")

    def test_an_unwired_readback_at_its_setpoint_default_builds(self, tmp_path):
        root = _with_readback_nominal(tmp_path / "facility", INVERTED)
        build_facility(root, project_name="p")

    def test_an_unwired_readback_off_its_setpoint_default_stops(self, tmp_path):
        root = _with_readback_nominal(tmp_path / "facility", 0.5)
        report = run_stages(root, project_name="p", later=LATER_STAGES)
        assert _lines(report) == [
            f"facility: seed-invalid: channel Q1:RB — `nominal` 0.5 differs from its setpoint "
            f"Q1:SP's {INVERTED}; fix: remove `nominal` from Q1:RB; a paired readback starts "
            "at its setpoint's value"
        ]

    def test_a_missing_directory_is_the_zero_source_file(self, tmp_path):
        document = build_facility(tmp_path / "absent", project_name="my proj")
        assert document["identity"] == {"code": "my_proj", "name": "my proj"}
        assert document["models"] == [{"name": "texture", "engine": "texture"}]
        assert document["devices"] == document["channels"] == document["places"] == []

    def test_the_first_error_is_raised(self, tmp_path):
        files = _tree()
        files["records/places.yaml"][1]["span"]["to_marker"] = "NOPE"
        with pytest.raises(FacilityBuildError, match="span-invalid: place SR/A"):
            build_facility(_write(tmp_path / "facility", copy.deepcopy(files)), project_name="p")


class TestPortedHelpers:
    def test_flatten_aliases(self):
        vocabulary = {
            "classes": [{"name": "Quadrupole", "aliases": ["Quad", "quad "]}],
            "signal_roles": [{"name": "position", "aliases": ["Pos"]}],
        }
        classes = [{"class": "SkewQuad", "aliases": ["Skew"]}]
        assert compute.flatten_aliases(vocabulary, classes) == [
            {
                "term": "quad",
                "scope": "device_class",
                "target": "Quadrupole",
                "source": "vocabulary",
            },
            {"term": "skew", "scope": "device_class", "target": "SkewQuad", "source": "facility"},
            {"term": "pos", "scope": "signal_role", "target": "position", "source": "vocabulary"},
        ]

    @pytest.mark.parametrize(
        ("place", "code"), [("SR/sect1", "SECT1"), ("LINE", "LINE"), ("", None), (None, None)]
    )
    def test_section_code(self, place, code):
        assert compute._section_code(place) == code
