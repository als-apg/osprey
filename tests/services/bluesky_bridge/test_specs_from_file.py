"""Unit tests for the device-file loader (``devices/_specs_from_file.py``).

Covers parsing, fail-soft degradation and de-duplication directly, plus the
``validate_device_document`` problem list the build uses to refuse a malformed
file. No module-level ``importorskip`` guard: the loader imports only
``yaml``, ``logging`` and the dependency-free ``devices/specs.py``, so it runs
in a slimmed install without ophyd-async. The two tests that carry a parsed
spec on into a *built* device guard ophyd-async themselves, in-function, so the
rest of the file stays runnable without it.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.services.bluesky_bridge.devices._specs_from_file import (
    VALUE_TYPES,
    specs_from_file,
    validate_device_document,
)
from osprey.services.bluesky_bridge.devices.specs import DeviceSpecs, TriggerableSpec
from osprey_connectors.command import CommandAct

_NO_DEVICES = DeviceSpecs([], [], [])


def _write(tmp_path: Path, doc: object, name: str = "devices.yml") -> Path:
    """Write ``doc`` as YAML into ``tmp_path`` and return the path."""
    path = tmp_path / name
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return path


# --- duplicate names --------------------------------------------------------


def test_duplicate_settable_name_drops_the_later_entry(tmp_path: Path) -> None:
    """Two settable entries with the same device name keep only the first."""
    path = _write(
        tmp_path,
        {
            "settables": [
                {"name": "m", "setpoint": "A:SP", "readback": "A:RB"},
                {"name": "m", "setpoint": "B:SP", "readback": "B:RB"},
            ]
        },
    )
    settables = specs_from_file(path).settables
    assert [s.name for s in settables] == ["m"]
    assert settables[0].setpoint_pv == "A:SP"  # first wins, not the shadowing second


def test_readable_name_colliding_with_a_settable_is_dropped(tmp_path: Path) -> None:
    """A readable reusing a settable's device name is dropped (settables resolve
    first); device names become event-data column keys, so a collision must not
    stand."""
    path = _write(
        tmp_path,
        {
            "settables": [{"name": "dev", "setpoint": "A:SP", "readback": "A:RB"}],
            "readables": [{"name": "dev", "pv": "B:RB"}, {"name": "other", "pv": "C:RB"}],
        },
    )
    found = specs_from_file(path)
    settables, readables = found.settables, found.readables
    assert [s.name for s in settables] == ["dev"]
    assert [s.name for s in readables] == ["other"]  # 'dev' readable dropped


def test_distinct_names_are_all_kept(tmp_path: Path) -> None:
    """No collision -> every parsed device survives (dedup is not over-eager)."""
    path = _write(
        tmp_path,
        {
            "settables": [{"name": "m1", "setpoint": "A:SP"}, {"name": "m2", "setpoint": "B:SP"}],
            "readables": [{"name": "d1", "pv": "C:RB"}, {"name": "d2", "pv": "D:RB"}],
        },
    )
    found = specs_from_file(path)
    settables, readables = found.settables, found.readables
    assert [s.name for s in settables] == ["m1", "m2"]
    assert [s.name for s in readables] == ["d1", "d2"]


# --- the reason this file format exists at all ------------------------------


def test_comma_bearing_pv_survives_the_round_trip(tmp_path: Path) -> None:
    """A PV containing a comma round-trips intact -- the env format could not
    carry one (comma was its entry separator), which is why the file exists."""
    comma_pv = "BTS:QF,1:CURRENT:SP"
    path = _write(
        tmp_path,
        {
            "settables": [{"name": "qf1", "setpoint": comma_pv, "readback": "BTS:QF,1:CURRENT:RB"}],
            "readables": [{"name": "bpm", "pv": "BTS:BPM,1:X:RB"}],
        },
    )
    found = specs_from_file(path)
    settables, readables = found.settables, found.readables
    assert settables[0].setpoint_pv == comma_pv
    assert settables[0].readback_pv == "BTS:QF,1:CURRENT:RB"
    assert readables[0].read_pv == "BTS:BPM,1:X:RB"


def test_colon_bearing_device_name_survives(tmp_path: Path) -> None:
    """Device names may be full colon-delimited addresses (the generator uses
    address-as-name), so nothing may split a name on ``:``."""
    path = _write(
        tmp_path,
        {"settables": [{"name": "SR:C01:QF:1", "setpoint": "SR:C01:QF:1:SP"}]},
    )
    settables = specs_from_file(path).settables
    assert [s.name for s in settables] == ["SR:C01:QF:1"]


def test_json_document_parses(tmp_path: Path) -> None:
    """``yaml.safe_load`` covers JSON, so a ``.json`` device file needs no
    separate reader."""
    path = tmp_path / "devices.json"
    path.write_text(
        json.dumps(
            {
                "settables": [{"name": "m", "setpoint": "A:SP", "readback": "A:RB"}],
                "readables": [{"name": "d", "pv": "C:RB"}],
            }
        ),
        encoding="utf-8",
    )
    found = specs_from_file(path)
    settables, readables = found.settables, found.readables
    assert [s.name for s in settables] == ["m"]
    assert settables[0].readback_pv == "A:RB"
    assert [s.name for s in readables] == ["d"]


def test_omitted_readback_becomes_none(tmp_path: Path) -> None:
    """``readback`` is optional; omitting it leaves ``readback_pv`` unset so the
    device falls back to the setpoint PV for reads."""
    path = _write(tmp_path, {"settables": [{"name": "m", "setpoint": "A:SP"}]})
    settables = specs_from_file(path).settables
    assert settables[0].readback_pv is None


def test_path_may_be_a_string(tmp_path: Path) -> None:
    """The container passes an env-var string; ``str`` is accepted alongside
    ``Path``."""
    path = _write(tmp_path, {"readables": [{"name": "d", "pv": "C:RB"}]})
    readables = specs_from_file(str(path)).readables
    assert [s.name for s in readables] == ["d"]


# --- fail-soft degradation --------------------------------------------------


def test_malformed_entry_is_skipped_and_the_rest_survive(tmp_path: Path) -> None:
    """One bad entry must not cost every other device its connection: malformed
    entries are skipped with a warning, good neighbours are kept."""
    path = _write(
        tmp_path,
        {
            "settables": [
                {"name": "good", "setpoint": "A:SP"},
                {"name": "no_setpoint"},
                {"setpoint": "B:SP"},
                {"name": "", "setpoint": "C:SP"},
                {"name": "bad_type", "setpoint": 17},
                "not-a-mapping",
                {"name": "empty_readback", "setpoint": "D:SP", "readback": ""},
                {"name": "also_good", "setpoint": "E:SP"},
            ],
            "readables": [
                {"name": "rb_good", "pv": "F:RB"},
                {"name": "rb_no_pv"},
                None,
                {"name": "rb_also_good", "pv": "G:RB"},
            ],
        },
    )
    found = specs_from_file(path)
    settables, readables = found.settables, found.readables
    assert [s.name for s in settables] == ["good", "also_good"]
    assert [s.name for s in readables] == ["rb_good", "rb_also_good"]


def test_missing_file_yields_no_devices(tmp_path: Path) -> None:
    """An unreadable path degrades to an empty device set rather than raising --
    the caller warns that the worker will expose no plans."""
    assert specs_from_file(tmp_path / "does-not-exist.yml") == _NO_DEVICES


def test_directory_path_yields_no_devices(tmp_path: Path) -> None:
    """A path that is a directory is unreadable too, and degrades the same way."""
    assert specs_from_file(tmp_path) == _NO_DEVICES


@pytest.mark.parametrize(
    "text",
    [
        # An empty file parses to ``None`` -- a valid, if useless, empty device set.
        pytest.param("", id="empty-file"),
        # A YAML syntax error degrades to an empty device set, not an exception.
        pytest.param("settables: [unclosed\n", id="unparseable-yaml"),
    ],
)
def test_file_without_a_usable_document_yields_no_devices(tmp_path: Path, text: str) -> None:
    path = tmp_path / "devices.yml"
    path.write_text(text, encoding="utf-8")
    assert specs_from_file(path) == _NO_DEVICES


def test_non_mapping_document_yields_no_devices_and_one_warning(tmp_path: Path, caplog) -> None:
    """A top-level list (a plausible mis-authoring) yields no devices and exactly
    one warning -- not one per entry."""
    path = _write(tmp_path, [{"name": "m", "setpoint": "A:SP"}])
    with caplog.at_level("WARNING"):
        assert specs_from_file(path) == _NO_DEVICES
    assert len(caplog.records) == 1


def test_unknown_top_level_key_rejects_the_document(tmp_path: Path, caplog) -> None:
    """A typo'd top-level key rejects the whole document with one warning: a
    silently-empty device set would look like a working deployment with no
    plans."""
    path = _write(
        tmp_path,
        {"settables": [{"name": "m", "setpoint": "A:SP"}], "readable": [{"name": "d", "pv": "C"}]},
    )
    with caplog.at_level("WARNING"):
        assert specs_from_file(path) == _NO_DEVICES
    assert len(caplog.records) == 1


def test_non_list_device_section_is_skipped(tmp_path: Path) -> None:
    """A section that is not a list contributes nothing, and the other section
    still loads."""
    path = _write(
        tmp_path,
        {
            "settables": {"name": "m", "setpoint": "A:SP"},
            "readables": [{"name": "d", "pv": "C:RB"}],
        },
    )
    found = specs_from_file(path)
    settables, readables = found.settables, found.readables
    assert settables == []
    assert [s.name for s in readables] == ["d"]


# --- settle_tolerance ---------------------------------------------------------


def test_a_settable_carries_its_settle_tolerance(tmp_path: Path) -> None:
    """``settle_tolerance`` is read as a float onto the spec; an int is a number."""
    path = _write(
        tmp_path,
        {
            "settables": [
                {"name": "m", "setpoint": "A:SP", "readback": "A:RB", "settle_tolerance": 0.011},
                {"name": "z", "setpoint": "B:SP", "settle_tolerance": 0},
                {"name": "n", "setpoint": "C:SP"},
            ]
        },
    )
    settables = specs_from_file(path).settables
    assert [(s.name, s.settle_tolerance) for s in settables] == [
        ("m", 0.011),
        ("z", 0.0),
        ("n", None),
    ]
    assert isinstance(settables[1].settle_tolerance, float)


@pytest.mark.parametrize("bad", [True, "0.01", -0.5, [0.1], float("nan")])
def test_a_bad_settle_tolerance_skips_the_entry(tmp_path: Path, caplog, bad: Any) -> None:
    """A bool, a non-number or a negative tolerance skips the entry with a warning,
    the same way a bad ``readback`` does."""
    path = _write(
        tmp_path,
        {
            "settables": [
                {"name": "bad", "setpoint": "A:SP", "settle_tolerance": bad},
                {"name": "good", "setpoint": "B:SP"},
            ]
        },
    )
    with caplog.at_level("WARNING"):
        settables = specs_from_file(path).settables
    assert [s.name for s in settables] == ["good"]
    assert "settle_tolerance" in caplog.text


@pytest.mark.parametrize("bad", [True, "0.01", -0.5, [0.1], float("nan")])
def test_validate_reports_a_bad_settle_tolerance(bad: Any) -> None:
    """The validator names the entry and the key for every value the loader skips."""
    problems = validate_device_document(
        {"settables": [{"name": "m", "setpoint": "A:SP", "settle_tolerance": bad}]}
    )
    assert problems == [
        "settables[0]: 'settle_tolerance' must be a number >= 0 or {relative: <number >= 0>} "
        "when present"
    ]


def test_validate_accepts_a_settle_tolerance() -> None:
    doc = {
        "settables": [
            {"name": "m", "setpoint": "A:SP", "readback": "A:RB", "settle_tolerance": 3.91},
            {"name": "z", "setpoint": "B:SP", "settle_tolerance": 0},
        ]
    }
    assert validate_device_document(doc) == []


def test_a_settable_carries_a_relative_tolerance_and_a_motion_band(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        {
            "settables": [
                {"name": "r", "setpoint": "A:SP", "settle_tolerance": {"relative": 1e-4}},
                {"name": "b", "setpoint": "B:SP", "readback": "B:RB", "motion_band": 0.011},
            ]
        },
    )
    settables = specs_from_file(path).settables
    assert [(s.name, s.settle_tolerance, s.settle_relative, s.motion_band) for s in settables] == [
        ("r", None, 1e-4, None),
        ("b", None, None, 0.011),
    ]


@pytest.mark.parametrize(
    ("key", "bad"),
    [
        ("settle_tolerance", {"relative": -1.0}),
        ("settle_tolerance", {"absolute": 0.1}),
        ("settle_tolerance", {"relative": 0.1, "absolute": 0.1}),
        ("motion_band", -0.5),
        ("motion_band", {"relative": 0.1}),
    ],
)
def test_a_bad_relative_tolerance_or_motion_band_skips_the_entry(
    tmp_path: Path, caplog, key: str, bad: Any
) -> None:
    path = _write(
        tmp_path,
        {
            "settables": [
                {"name": "bad", "setpoint": "A:SP", key: bad},
                {"name": "ok", "setpoint": "B:SP"},
            ]
        },
    )
    with caplog.at_level("WARNING"):
        settables = specs_from_file(path).settables
    assert [s.name for s in settables] == ["ok"]
    assert key in caplog.text
    assert validate_device_document({"settables": [{"name": "m", "setpoint": "A:SP", key: bad}]})


def test_validate_accepts_a_relative_tolerance_and_a_motion_band() -> None:
    doc = {
        "settables": [
            {"name": "r", "setpoint": "A:SP", "settle_tolerance": {"relative": 1e-4}},
            {"name": "b", "setpoint": "B:SP", "readback": "B:RB", "motion_band": 0.011},
        ]
    }
    assert validate_device_document(doc) == []


# --- unit -------------------------------------------------------------------


def test_a_device_carries_the_units_its_entry_states(tmp_path: Path) -> None:
    """``unit`` and ``setpoint_unit`` are read onto the specs; absent leaves ``None``."""
    path = _write(
        tmp_path,
        {
            "settables": [
                {
                    "name": "p",
                    "setpoint": "A:SP",
                    "readback": "A:RB",
                    "unit": "u1",
                    "setpoint_unit": "u2",
                },
                {"name": "q", "setpoint": "B:SP", "unit": "u3"},
                {"name": "r", "setpoint": "C:SP"},
            ],
            "readables": [{"name": "t", "pv": "T", "unit": "u4"}, {"name": "s", "pv": "S"}],
        },
    )
    found = specs_from_file(path)
    settables, readables = found.settables, found.readables
    assert [(s.name, s.unit, s.setpoint_unit) for s in settables] == [
        ("p", "u1", "u2"),
        ("q", "u3", None),
        ("r", None, None),
    ]
    assert [(r.name, r.unit) for r in readables] == [("t", "u4"), ("s", None)]


_SETTABLE = {"name": "bad", "setpoint": "A:SP"}
_READABLE = {"name": "bad", "pv": "A"}


@pytest.mark.parametrize(
    ("section", "key", "bad"),
    [
        ("settables", "unit", 3),
        ("settables", "unit", ""),
        ("settables", "setpoint_unit", ["u"]),
        ("readables", "unit", True),
        ("readables", "unit", "  "),
    ],
)
def test_a_bad_unit_skips_the_entry_and_is_a_validation_problem(
    tmp_path: Path, caplog, section: str, key: str, bad: Any
) -> None:
    entry = _SETTABLE if section == "settables" else _READABLE
    good = {**entry, "name": "ok"}
    path = _write(tmp_path, {section: [{**entry, key: bad}, good]})
    with caplog.at_level("WARNING"):
        found = specs_from_file(path)
        settables, readables = found.settables, found.readables
    assert [s.name for s in (settables if section == "settables" else readables)] == ["ok"]
    assert key in caplog.text
    assert validate_device_document({section: [{**entry, key: bad}]}) == [
        f"{section}[0]: {key!r} must be a non-empty string when present"
    ]


def test_validate_accepts_units() -> None:
    doc = {
        "settables": [
            {"name": "p", "setpoint": "A:SP", "readback": "A:RB", "unit": "u", "setpoint_unit": "v"}
        ],
        "readables": [{"name": "t", "pv": "T", "unit": "u"}],
    }
    assert validate_device_document(doc) == []


# --- value type -------------------------------------------------------------


def test_each_entry_reads_its_value_type(tmp_path: Path) -> None:
    """``value_type`` is read onto every spec; absent leaves ``None``."""
    path = _write(
        tmp_path,
        {
            "settables": [
                {
                    "name": "p",
                    "setpoint": "A:SP",
                    "readback": "A:RB",
                    "value_type": "int",
                    "setpoint_value_type": "float",
                },
                {"name": "q", "setpoint": "B:SP"},
            ],
            "triggerables": [
                {
                    "name": "go",
                    "command": "GO",
                    "status": "BUSY",
                    "idle": "DONE",
                    "timeout_s": 5.0,
                    "value_type": "enum",
                }
            ],
            "readables": [
                {"name": "w", "pv": "W", "value_type": "string"},
                {"name": "s", "pv": "S"},
            ],
        },
    )
    found = specs_from_file(path)

    assert [(s.name, s.value_type, s.setpoint_value_type) for s in found.settables] == [
        ("p", "int", "float"),
        ("q", None, None),
    ]
    assert [t.value_type for t in found.triggerables] == ["enum"]
    assert [(r.name, r.value_type) for r in found.readables] == [("w", "string"), ("s", None)]


@pytest.mark.parametrize(
    ("section", "entry", "key"),
    [
        ("settables", _SETTABLE, "value_type"),
        ("settables", _SETTABLE, "setpoint_value_type"),
        ("readables", _READABLE, "value_type"),
    ],
)
def test_a_value_type_that_is_no_value_type_skips_the_entry(
    tmp_path: Path, caplog, section: str, entry: dict[str, Any], key: str
) -> None:
    path = _write(tmp_path, {section: [{**entry, key: "complex"}, {**entry, "name": "ok"}]})
    with caplog.at_level("WARNING"):
        found = specs_from_file(path)
    kept = found.settables if section == "settables" else found.readables
    assert [spec.name for spec in kept] == ["ok"]
    assert key in caplog.text
    assert validate_device_document({section: [{**entry, key: "complex"}]}) == [
        f"{section}[0]: {key!r} must be one of {list(VALUE_TYPES)!r} when present"
    ]


def test_each_entry_reads_the_words_of_an_enum_channel(tmp_path: Path) -> None:
    """``options`` and ``setpoint_options`` are read onto every spec as tuples;
    absent leaves ``None``."""
    words = ["IDLE", "BUSY"]
    path = _write(
        tmp_path,
        {
            "settables": [
                {
                    "name": "p",
                    "setpoint": "A:SP",
                    "readback": "A:RB",
                    "value_type": "enum",
                    "options": words,
                    "setpoint_value_type": "enum",
                    "setpoint_options": ["OFF", "ON", "AUTO"],
                },
                {"name": "q", "setpoint": "B:SP"},
            ],
            "triggerables": [
                {
                    "name": "go",
                    "command": "GO",
                    "status": "BUSY",
                    "idle": 0,
                    "timeout_s": 5.0,
                    "value_type": "enum",
                    "options": words,
                }
            ],
            "readables": [{"name": "w", "pv": "W", "value_type": "enum", "options": words}],
        },
    )
    found = specs_from_file(path)

    assert [(s.options, s.setpoint_options) for s in found.settables] == [
        (("IDLE", "BUSY"), ("OFF", "ON", "AUTO")),
        (None, None),
    ]
    assert [t.options for t in found.triggerables] == [("IDLE", "BUSY")]
    assert [r.options for r in found.readables] == [("IDLE", "BUSY")]


@pytest.mark.parametrize("words", [[], ["IDLE", ""], "IDLE", [1, 2]])
@pytest.mark.parametrize(
    ("section", "entry", "key"),
    [
        ("settables", _SETTABLE, "options"),
        ("settables", _SETTABLE, "setpoint_options"),
        ("readables", _READABLE, "options"),
    ],
)
def test_options_that_are_no_list_of_words_skip_the_entry(
    tmp_path: Path, caplog, section: str, entry: dict[str, Any], key: str, words: Any
) -> None:
    path = _write(tmp_path, {section: [{**entry, key: words}, {**entry, "name": "ok"}]})
    with caplog.at_level("WARNING"):
        found = specs_from_file(path)
    kept = found.settables if section == "settables" else found.readables
    assert [spec.name for spec in kept] == ["ok"]
    assert key in caplog.text
    assert validate_device_document({section: [{**entry, key: words}]}) == [
        f"{section}[0]: {key!r} must be a non-empty list of non-empty strings when present"
    ]


def test_the_value_types_are_the_facility_value_types() -> None:
    from osprey_facility.schema import core

    assert list(VALUE_TYPES) == [member.value for member in core.ValueTypeEnum]


# --- shape ------------------------------------------------------------------


def test_a_readable_reads_its_shape(tmp_path: Path) -> None:
    """``shape`` is read onto the readable spec as a tuple; absent leaves ``None``."""
    path = _write(
        tmp_path,
        {"readables": [{"name": "w", "pv": "W", "shape": [500]}, {"name": "s", "pv": "S"}]},
    )
    readables = specs_from_file(path).readables
    assert [(r.name, r.shape) for r in readables] == [("w", (500,)), ("s", None)]
    assert validate_device_document({"readables": [{"name": "w", "pv": "W", "shape": [500]}]}) == []


_MALFORMED_SHAPES = [[], [0], [2, 3], [True], "500", [1.5]]


@pytest.mark.parametrize("bad", _MALFORMED_SHAPES)
def test_a_readable_with_a_malformed_shape_is_skipped_with_a_warning(
    tmp_path: Path, caplog, bad: Any
) -> None:
    path = _write(
        tmp_path, {"readables": [{**_READABLE, "shape": bad}, {**_READABLE, "name": "ok"}]}
    )
    with caplog.at_level("WARNING"):
        readables = specs_from_file(path).readables
    assert [r.name for r in readables] == ["ok"]
    assert "'bad'" in caplog.text
    assert "shape" in caplog.text


@pytest.mark.parametrize("bad", _MALFORMED_SHAPES)
def test_validate_refuses_a_readable_shape_that_is_not_one_positive_int(bad: Any) -> None:
    assert validate_device_document({"readables": [{**_READABLE, "shape": bad}]}) == [
        "readables[0]: 'shape' must be a list of one int >= 1 when present"
    ]


# --- validate_device_document ----------------------------------------------


def test_validate_accepts_a_well_formed_document() -> None:
    """A document the loader keeps whole reports no problems."""
    doc = {
        "settables": [
            {"name": "m", "setpoint": "A:SP", "readback": "A:RB"},
            {"name": "m2", "setpoint": "B:SP"},
        ],
        "readables": [{"name": "d", "pv": "C:RB"}],
    }
    assert validate_device_document(doc) == []


def test_validate_accepts_an_empty_or_partial_document() -> None:
    """An empty document and a one-section document are valid (if useless): the
    build refuses malformed files, not empty ones."""
    assert validate_device_document(None) == []
    assert validate_device_document({}) == []
    assert validate_device_document({"readables": []}) == []


def test_validate_reports_a_non_mapping_document() -> None:
    """A top-level non-mapping is a single, named problem."""
    problems = validate_device_document([{"name": "m"}])
    assert len(problems) == 1
    assert "mapping" in problems[0]


def test_validate_reports_unknown_top_level_keys() -> None:
    """An unknown top-level key is named, so the typo is fixable from the
    message alone."""
    problems = validate_device_document({"settables": [], "readable": []})
    assert len(problems) == 1
    assert "readable" in problems[0]


def test_validate_names_each_malformed_entry() -> None:
    """Every malformed entry is reported, with its section and index, so a
    13k-line file can be repaired without bisecting it."""
    problems = validate_device_document(
        {
            "settables": [
                {"name": "ok", "setpoint": "A:SP"},
                {"name": "no_setpoint"},
                {"setpoint": "B:SP"},
                {"name": "bad_type", "setpoint": 17},
                "not-a-mapping",
                {"name": "typo", "setpoint": "C:SP", "setpoint_pv": "C:SP"},
            ],
            "readables": [{"name": "rb", "pv": "D:RB"}, {"name": "rb_no_pv"}],
        }
    )
    assert len(problems) == 6
    joined = "\n".join(problems)
    assert "settables[1]" in joined and "setpoint" in joined
    assert "settables[2]" in joined and "name" in joined
    assert "settables[3]" in joined
    assert "settables[4]" in joined
    assert "settables[5]" in joined and "setpoint_pv" in joined  # unknown entry key
    assert "readables[1]" in joined


def test_validate_reports_a_non_list_section() -> None:
    """A section that is not a list is named as such."""
    problems = validate_device_document({"settables": {"name": "m"}})
    assert len(problems) == 1
    assert "settables" in problems[0] and "list" in problems[0]


def test_validate_reports_duplicate_names() -> None:
    """Duplicate device names are reported: the loader drops the later entry, so
    an unrefused file would silently ship fewer devices than it lists."""
    problems = validate_device_document(
        {
            "settables": [{"name": "m", "setpoint": "A:SP"}, {"name": "m", "setpoint": "B:SP"}],
            "readables": [{"name": "m", "pv": "C:RB"}],
        }
    )
    assert len(problems) == 2
    assert all("m" in problem for problem in problems)


def test_validate_never_raises_on_arbitrary_input() -> None:
    """``validate_device_document`` reports, never raises -- the build turns its
    list into a refusal message and needs no try/except around it."""
    for doc in ("a string", 17, [], {"settables": [[]]}, object()):
        problems = validate_device_document(doc)
        assert isinstance(problems, list)
        assert problems, f"{doc!r} was accepted"
    # A section set to null is an absent section, and a partial document is valid.
    assert validate_device_document({"settables": None}) == []


# --- from a parsed spec to a built device -----------------------------------
#
# Everything above stops at the parsed spec. The two tests below carry a spec
# on through ``devices/connector.py`` into a live device, because that is where
# the format's promise is actually kept or broken: a comma-bearing PV that
# parses correctly but is mangled on the way to the connector would still leave
# the deployment unable to address the channel.


class _FakeConnector:
    """A minimal async double for ``ControlSystemConnector``.

    Records the channel addresses the built devices actually hand to the
    control system, which is what the comma test asserts on. Reads answer
    ``0.0`` so a ``set(0.0)`` settles on its first readback poll instead of
    spinning until the settle timeout.
    """

    def __init__(self) -> None:
        self.reads: list[str] = []
        self.writes: list[tuple[str, Any]] = []

    async def read_channel(self, address: str) -> Any:
        self.reads.append(address)
        return _FakeChannelValue(0.0)

    async def write_channel_checked(self, address: str, value: Any, **_: Any) -> Any:
        self.writes.append((address, value))
        # The real result type, not a stand-in, so this double cannot drift from
        # the contract. ``unrequested`` (nothing confirmed) is what leaves the
        # settle poll above as the only check, which is what this fake wants.
        # Imported in-function: the module level here stays dependency-light so
        # the parsing tests run in a slimmed install.
        from osprey_connectors.control_system._base import ChannelWriteResult, WriteOutcome

        return ChannelWriteResult(
            channel_address=address, value_written=value, outcome=WriteOutcome.UNREQUESTED
        )


class _FakeChannelValue:
    """Stand-in for ``osprey_connectors.control_system._base.ChannelValue``."""

    def __init__(self, value: float) -> None:
        self.value = value
        self.timestamp = time.time()


def test_comma_bearing_pv_reaches_the_connector_of_a_built_device(tmp_path: Path) -> None:
    """A comma-bearing PV survives parse *and* device construction: the built
    device hands the connector the address verbatim.

    The round-trip test above proves the loader keeps the comma; this proves
    nothing downstream re-splits it, which is the property the env format could
    not offer at all.
    """
    pytest.importorskip("ophyd_async")
    from osprey.services.bluesky_bridge.devices.connector import build_devices

    setpoint_pv = "BTS:QF,1:CURRENT:SP"
    readback_pv = "BTS:QF,1:CURRENT:RB"
    read_pv = "BTS:BPM,1:X:RB"
    path = _write(
        tmp_path,
        {
            "settables": [{"name": "qf1", "setpoint": setpoint_pv, "readback": readback_pv}],
            "readables": [{"name": "bpm1", "pv": read_pv}],
        },
    )
    found = specs_from_file(path)
    settables, readables = found.settables, found.readables

    async def build_and_exercise() -> _FakeConnector:
        connector = _FakeConnector()
        devices = await build_devices(settables, readables, connector=connector)
        await devices["qf1"].set(0.0)
        await devices["bpm1"].read()
        return connector

    connector = asyncio.run(build_and_exercise())
    assert connector.writes == [(setpoint_pv, 0.0)]
    assert readback_pv in connector.reads  # the settable's readback poll
    assert read_pv in connector.reads  # the readable's read


# --- scale: a facility-sized device file ------------------------------------


_SCALE_SETTABLES = 1_000
"""Settable entries in the synthetic scale document."""

_SCALE_READABLES = 12_000
"""Readable entries in the synthetic scale document. Together with the
settables that is ~13k entries -- the facility-scale projection, an order of
magnitude above the 512 settable / 822 readable set the deployment starts
from, and covering the ~12.8k fields the env format left out."""


def _scale_document(settables: int, readables: int) -> dict[str, list[dict[str, str]]]:
    """Build a synthetic facility-sized device document in memory.

    Generated rather than committed as a golden file: a 13k-entry fixture is
    pure noise in review and in the repository, and nothing here depends on the
    exact addresses. One entry carries a comma so the scale case exercises the
    wide document *and* the format's reason for existing at once.
    """
    return {
        "settables": [
            {
                "name": f"SR:C{index // 40:02d}:MAG:CORR:{index:05d}",
                "setpoint": f"SR:C{index // 40:02d}:MAG:CORR:{index:05d}:CUR:SP",
                "readback": f"SR:C{index // 40:02d}:MAG:CORR:{index:05d}:CUR:RB",
            }
            for index in range(settables)
        ]
        + [{"name": "comma_bearing", "setpoint": "BTS:QF,1:CURRENT:SP"}],
        "readables": [
            {
                "name": f"SR:C{index // 400:02d}:DIAG:{index:05d}",
                "pv": f"SR:C{index // 400:02d}:DIAG:{index:05d}:VAL:RB",
            }
            for index in range(readables)
        ],
    }


@pytest.mark.slow
def test_facility_scale_document_parses_and_builds_within_the_worker_budget(
    tmp_path: Path,
) -> None:
    """A ~13k-entry device file parses and builds, and the wall-time it costs
    is reported.

    The worker awaits parse plus construction inside a single
    ``qserver_startup.CONNECT_TIMEOUT`` window, so this is a soft measurement
    of how much of that budget a facility-sized file consumes. No threshold is
    asserted: the number is machine-dependent, and a timing assertion here
    would be a flake generator rather than a guard. Run with ``-s`` to see the
    report line.
    """
    pytest.importorskip("ophyd_async")
    from osprey.services.bluesky_bridge.devices.connector import build_devices
    from osprey.services.bluesky_bridge.qserver_startup import CONNECT_TIMEOUT

    path = _write(tmp_path, _scale_document(_SCALE_SETTABLES, _SCALE_READABLES))

    parse_start = time.perf_counter()
    found = specs_from_file(path)
    settables, readables = found.settables, found.readables
    parse_seconds = time.perf_counter() - parse_start

    # Built and exercised inside ONE event loop: an ophyd-async device is bound
    # to the loop it was connected on, so a `set()` issued from a second
    # `asyncio.run` would not be the same device the build produced.
    connector = _FakeConnector()

    async def build_and_probe() -> tuple[dict[str, Any], float]:
        start = time.perf_counter()
        built = await build_devices(settables, readables, connector=connector)
        elapsed = time.perf_counter() - start
        # The comma-bearing entry, exercised the way a plan would: what the
        # device hands the connector is the assertion, never its private
        # attributes -- the same public form the round-trip test above uses.
        await built["comma_bearing"].set(0.0)
        return built, elapsed

    devices, build_seconds = asyncio.run(build_and_probe())

    total_seconds = parse_seconds + build_seconds
    print(
        f"\n[scale] {len(devices)} devices "
        f"({len(settables)} settable, {len(readables)} readable): "
        f"parse {parse_seconds:.2f}s + build {build_seconds:.2f}s "
        f"= {total_seconds:.2f}s of the {CONNECT_TIMEOUT:.0f}s worker budget "
        f"({total_seconds / CONNECT_TIMEOUT:.0%})"
    )

    # The deterministic half: nothing was dropped on the way through, and the
    # comma survived the whole path at scale.
    assert len(settables) == _SCALE_SETTABLES + 1
    assert len(readables) == _SCALE_READABLES
    assert len(devices) == _SCALE_SETTABLES + 1 + _SCALE_READABLES
    assert connector.writes == [("BTS:QF,1:CURRENT:SP", 0.0)]


# --- triggerables -------------------------------------------------------------


_TRIGGERABLE = {
    "name": "ES1:MCA01:ACQ",
    "command": "ES1:MCA01:ACQ",
    "status": "ES1:MCA01:ACQ_STATUS",
    "idle": "Idle",
    "timeout_s": 5.0,
}


def test_a_triggerable_entry_parses(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        {"triggerables": [_TRIGGERABLE, {**_TRIGGERABLE, "name": "b", "idle": 0, "timeout_s": 2}]},
    )

    found = specs_from_file(path)

    assert found.triggerables == [
        TriggerableSpec("ES1:MCA01:ACQ", "ES1:MCA01:ACQ", "ES1:MCA01:ACQ_STATUS", "Idle", 5.0),
        TriggerableSpec("b", "ES1:MCA01:ACQ", "ES1:MCA01:ACQ_STATUS", 0, 2.0),
    ]
    assert validate_device_document(yaml.safe_load(path.read_text())) == []


@pytest.mark.parametrize(
    ("override", "problem"),
    [
        ({"command": ""}, "'command' must be a non-empty string"),
        ({"status": None}, "'status' must be a non-empty string"),
        ({"idle": None}, "'idle' must be a non-null scalar"),
        ({"idle": [0]}, "'idle' must be a non-null scalar"),
        ({"timeout_s": 0}, "'timeout_s' must be a finite number > 0"),
        ({"timeout_s": float("inf")}, "'timeout_s' must be a finite number > 0"),
        ({"timeout_s": True}, "'timeout_s' must be a finite number > 0"),
    ],
)
def test_a_malformed_triggerable_is_skipped_and_reported(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, override: dict[str, Any], problem: str
) -> None:
    bad = {**_TRIGGERABLE, "name": "bad", **override}
    doc = {"triggerables": [bad, _TRIGGERABLE]}
    path = _write(tmp_path, doc)

    with caplog.at_level("WARNING"):
        found = specs_from_file(path)

    assert [spec.name for spec in found.triggerables] == ["ES1:MCA01:ACQ"]
    assert "'bad'" in caplog.text
    assert f"triggerables[0]: {problem}" in validate_device_document(doc)


def test_a_document_without_triggerables_still_loads(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        {
            "settables": [{"name": "m", "setpoint": "A:SP"}],
            "readables": [{"name": "d", "pv": "B:RB"}],
        },
    )

    found = specs_from_file(path)

    assert [spec.name for spec in found.settables] == ["m"]
    assert [spec.name for spec in found.readables] == ["d"]
    assert found.triggerables == []


def test_a_triggerable_name_claimed_by_a_settable_is_dropped(tmp_path: Path) -> None:
    doc = {
        "settables": [{"name": "ES1:MCA01:ACQ", "setpoint": "A:SP"}],
        "triggerables": [_TRIGGERABLE],
        "readables": [{"name": "ES1:MCA01:ACQ", "pv": "B:RB"}],
    }
    path = _write(tmp_path, doc)

    found = specs_from_file(path)

    assert [spec.name for spec in found.settables] == ["ES1:MCA01:ACQ"]
    assert found.triggerables == [] and found.readables == []
    assert len([p for p in validate_device_document(doc) if "already claimed" in p]) == 2


# --- a settable's start act --------------------------------------------------

_START_ACT = {
    "command": "ES1:STAGE:MOVE",
    "status": "ES1:STAGE:STATUS",
    "idle": "Done",
    "timeout_s": 10.0,
}
_POSITIONER = {
    "name": "ES1:STAGE:POS:SP",
    "setpoint": "ES1:STAGE:POS:SP",
    "readback": "ES1:STAGE:POS:RB",
    "started_by": _START_ACT,
}


def test_a_settable_reads_its_start_act(tmp_path: Path) -> None:
    path = _write(tmp_path, {"settables": [_POSITIONER, {"name": "m", "setpoint": "A:SP"}]})

    positioner, plain = specs_from_file(path).settables

    assert positioner.started_by == CommandAct("ES1:STAGE:MOVE", "ES1:STAGE:STATUS", "Done", 10.0)
    assert plain.started_by is None
    assert validate_device_document(yaml.safe_load(path.read_text())) == []


@pytest.mark.parametrize(
    "bad",
    [
        "ES1:STAGE:MOVE",
        {key: value for key, value in _START_ACT.items() if key != "status"},
        {**_START_ACT, "command": ""},
        {**_START_ACT, "timeout_s": 0},
    ],
    ids=["not-a-mapping", "missing-subkey", "empty-address", "non-positive-bound"],
)
def test_a_malformed_start_act_is_refused_by_the_build_and_skipped_by_the_loader(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, bad: Any
) -> None:
    doc = {"settables": [{**_POSITIONER, "name": "bad", "started_by": bad}, _POSITIONER]}
    path = _write(tmp_path, doc)

    with caplog.at_level("WARNING"):
        found = specs_from_file(path)

    assert [spec.name for spec in found.settables] == ["ES1:STAGE:POS:SP"]
    assert "'bad'" in caplog.text
    (problem,) = validate_device_document(doc)
    assert problem.startswith("settables[0]: 'started_by' must be a mapping of")


def test_a_settable_reads_its_stop_channel(tmp_path: Path) -> None:
    stopping = {**_POSITIONER, "stopped_by": "ES1:STAGE:STOP"}
    path = _write(tmp_path, {"settables": [stopping, {"name": "m", "setpoint": "A:SP"}]})

    positioner, plain = specs_from_file(path).settables

    assert positioner.stopped_by == "ES1:STAGE:STOP"
    assert plain.stopped_by is None
    assert validate_device_document(yaml.safe_load(path.read_text())) == []


@pytest.mark.parametrize(
    "entry",
    [
        {"name": "bad", "setpoint": "A:SP", "stopped_by": "ES1:STAGE:STOP"},
        {**_POSITIONER, "name": "bad", "stopped_by": ""},
    ],
    ids=["without-a-start-act", "empty-address"],
)
def test_a_stop_without_a_start_act_is_refused_by_the_build_and_skipped_by_the_loader(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, entry: dict[str, Any]
) -> None:
    doc = {"settables": [entry, _POSITIONER]}
    path = _write(tmp_path, doc)

    with caplog.at_level("WARNING"):
        found = specs_from_file(path)

    assert [spec.name for spec in found.settables] == ["ES1:STAGE:POS:SP"]
    assert "'bad'" in caplog.text
    (problem,) = validate_device_document(doc)
    assert problem.startswith("settables[0]: 'stopped_by' ")
