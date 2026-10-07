"""The model runner's configuration, decided from a hand-built simulator view.

``apply_safety`` takes a configuration shaped as ``Runner.generate_config``
returns it and fixes the five write-path keys, each setpoint's band and each
variable's PV mode; ``chromaticity_addresses`` names the channels wired to the
chromaticity output; ``as_declared`` hands each waveform over as the array its
variable declares. All three are pure, so these run on any host.
"""

from __future__ import annotations

import ast
import copy
from pathlib import Path
from typing import Any

import numpy as np
from lume.variables import NDVariable, ScalarVariable, StrVariable

from osprey.services.virtual_accelerator.serving import runner_config
from osprey.services.virtual_accelerator.serving.runner_config import (
    apply_safety,
    as_declared,
    chromaticity_addresses,
)

STATUS = "T:SIM:M:STATUS"


def _channel(address: str, **fields: Any) -> dict[str, Any]:
    role = fields.pop("role", "readback")
    return {
        "address": address,
        "role": role,
        "pair": address if role == "setpoint" else None,
        "value_type": fields.pop("value_type", "float"),
        "unit": None,
        "description": None,
        "writable": fields.pop("writable", False),
        "value_range": fields.pop("value_range", None),
        "owner": fields.pop("owner", "texture"),
        **fields,
    }


VIEW: dict[str, Any] = {
    "schema": "osprey.facility.simulator/1",
    "code": "T",
    "models": [
        {
            "name": "M",
            "engine": "pyat",
            "served": True,
            "settings": {},
            "deck": None,
            "wiring": [
                {
                    "id": "1",
                    "address": "M:HCM:SP",
                    "element": "HCM1",
                    "engine": {"attribute": "KickAngle"},
                },
                {"id": "2", "address": "M:BPM:X", "element": "BPM1", "engine": {"axis": "x"}},
                {"id": "3", "address": "M:TUNE:X", "engine": {"attribute": "tune", "axis": "x"}},
                {
                    "id": "4",
                    "address": "M:CHROM:X",
                    "engine": {"attribute": "chromaticity", "axis": "x"},
                },
                {
                    "id": "5",
                    "address": "M:CHROM:Y",
                    "engine": {"attribute": "chromaticity", "index": 1},
                },
                {"id": "6", "address": "M:CHROM", "engine": {"attribute": "chromaticity"}},
                {
                    "id": "7",
                    "address": "M:SLICED",
                    "slices": [{"element": "Q1"}],
                    "engine": {"attribute": "chromaticity"},
                },
            ],
        },
        {
            "name": "texture",
            "engine": "texture",
            "served": True,
            "settings": {},
            "deck": None,
            "wiring": [],
        },
    ],
    "channels": [
        _channel("M:BPM:X", owner="M"),
        _channel("M:CHROM", owner="M", value_type="waveform", shape=[2]),
        _channel("M:CHROM:X", owner="M"),
        _channel("M:CHROM:Y", owner="M"),
        _channel(
            "M:HCM:SP",
            owner="M",
            role="setpoint",
            writable=True,
            value_range=[-2.0, 2.0],
            precision=4,
        ),
        _channel("T:COUNT", value_type="int", precision=2),
        _channel("M:SLICED", owner="M"),
        _channel("M:TUNE:X", owner="M"),
        _channel("T:LOCKED:SP", role="setpoint", writable=False, value_range=[0.0, 1.0]),
        _channel("T:OPEN:SP", role="setpoint", writable=True),
        _channel("T:RB"),
    ],
}


def _generated(view: dict[str, Any]) -> dict[str, Any]:
    """A configuration in ``Runner.generate_config``'s shape for ``view``'s composite."""
    addresses = [channel["address"] for channel in view["channels"]] + [STATUS]
    return {
        "description": "",
        "prefix": "",
        "max_array_bytes": "80000000",
        "variables": {
            address: {"name": address, "pv": address, "mode": "rw"} for address in addresses
        },
    }


def test_the_five_write_path_keys_are_set() -> None:
    config = apply_safety(_generated(VIEW), VIEW)

    assert config["update_rate"] == 0.0
    assert config["echo_unconfirmed_writes"] is False
    assert config["alarm_on_refused_write"] is True
    assert config["clamp_writes"] is True
    assert config["control_pvs"] is False


def test_max_array_bytes_stays_a_string() -> None:
    config = apply_safety(_generated(VIEW), VIEW)

    assert isinstance(config["max_array_bytes"], str)
    assert config["max_array_bytes"] == "80000000"


def test_every_setpoint_carries_the_view_band() -> None:
    config = apply_safety(_generated(VIEW), VIEW)

    setpoints = [channel for channel in VIEW["channels"] if channel["role"] == "setpoint"]
    assert len(setpoints) == 3
    for channel in setpoints:
        assert config["variables"][channel["address"]]["value_range"] == channel["value_range"]


def test_a_non_setpoint_carries_no_band() -> None:
    config = apply_safety(_generated(VIEW), VIEW)

    assert "value_range" not in config["variables"]["M:BPM:X"]
    assert "value_range" not in config["variables"][STATUS]


def test_only_a_writable_setpoint_is_read_write() -> None:
    config = apply_safety(_generated(VIEW), VIEW)

    modes = {address: entry["mode"] for address, entry in config["variables"].items()}
    assert {address for address, mode in modes.items() if mode == "rw"} == {
        "M:HCM:SP",
        "T:OPEN:SP",
    }
    assert set(modes.values()) == {"rw", "ro"}
    assert modes[STATUS] == "ro"
    assert modes["T:LOCKED:SP"] == "ro"


def test_the_given_configuration_is_left_as_it_was() -> None:
    given = _generated(VIEW)
    before = copy.deepcopy(given)

    apply_safety(given, VIEW)

    assert given == before


def test_a_float_channel_stating_precision_hands_it_to_the_runner() -> None:
    config = apply_safety(_generated(VIEW), VIEW)

    assert config["variables"]["M:HCM:SP"]["precision"] == 4
    stated = {address for address, entry in config["variables"].items() if "precision" in entry}
    assert stated == {"M:HCM:SP"}


def test_chromaticity_addresses_are_those_wired_to_the_chromaticity_output() -> None:
    assert chromaticity_addresses(VIEW) == {"M:CHROM", "M:CHROM:X", "M:CHROM:Y"}


def test_a_view_without_chromaticity_wiring_names_none() -> None:
    view = copy.deepcopy(VIEW)
    view["models"][0]["wiring"] = view["models"][0]["wiring"][:3]

    assert chromaticity_addresses(view) == frozenset()


DECLARED = {
    "M:TUNES": NDVariable(name="M:TUNES", shape=(3,)),
    "M:GRID": NDVariable(name="M:GRID", shape=(2, 3)),
    "M:BPM:X": ScalarVariable(name="M:BPM:X"),
    "T:NAME": StrVariable(name="T:NAME"),
}


def test_a_flat_list_becomes_a_float_array_of_the_declared_shape() -> None:
    served = as_declared(DECLARED, {"M:TUNES": [0.13, 0.22, 0.0086]})

    tunes = served["M:TUNES"]
    assert isinstance(tunes, np.ndarray)
    assert tunes.dtype == np.float64
    assert tunes.shape == (3,)
    assert tunes.tolist() == [0.13, 0.22, 0.0086]


def test_a_flat_list_takes_a_two_dimensional_declared_shape() -> None:
    served = as_declared(DECLARED, {"M:GRID": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})

    assert served["M:GRID"].dtype == np.float64
    assert served["M:GRID"].tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_an_array_a_scalar_and_a_string_are_left_untouched() -> None:
    array = np.array([0.1, 0.2, 0.3])
    values = {"M:TUNES": array, "M:BPM:X": 1.5e-5, "T:NAME": "nominal"}

    served = as_declared(DECLARED, values)

    assert served["M:TUNES"] is array
    assert served["M:BPM:X"] == 1.5e-5
    assert served["T:NAME"] == "nominal"


def test_an_undefined_waveform_stays_none() -> None:
    assert as_declared(DECLARED, {"M:TUNES": None}) == {"M:TUNES": None}


def test_a_name_with_no_variable_is_left_untouched() -> None:
    value = [1.0, 2.0]

    served = as_declared(DECLARED, {"T:UNDECLARED": value})

    assert served["T:UNDECLARED"] is value


def test_the_module_imports_no_server_library() -> None:
    tree = ast.parse(Path(runner_config.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    roots = {name.split(".")[0] for name in imported}
    assert roots.isdisjoint({"lume_pva_apg", "pcaspy", "p4p"})
