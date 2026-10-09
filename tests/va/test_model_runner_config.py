"""The model runner's configuration, decided from a hand-built simulator view.

``apply_safety`` takes a configuration shaped as ``Runner.generate_config``
returns it and fixes the five write-path keys, each setpoint's band and each
variable's PV mode; ``periodic_addresses`` names the channels whose binding
refreshes ``periodic``. Both read the view through ``SimulatorView`` and are
pure, so these run on any host.
"""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from typing import Any

import pytest

from osprey.services.virtual_accelerator.serving import runner_config
from osprey.services.virtual_accelerator.serving.runner_config import (
    apply_safety,
    periodic_addresses,
)
from osprey_connectors.simulation.view import (
    ADDRESSES_FILE,
    SCHEMAS,
    VARIABLES_FILE,
    SimulatorView,
)
from tests._builds import BuiltProject

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
        "on": None,
        **fields,
    }


def _binding(record: dict[str, Any], role: str, plane: str | None, refresh: str) -> dict[str, Any]:
    """A wiring record stamped as the build stamps it."""
    return {**record, "role": role, "plane": plane, "refresh": refresh}


VIEW: dict[str, Any] = {
    "schema": SCHEMAS[VARIABLES_FILE],
    "code": "T",
    "models": [
        {
            "name": "M",
            "engine": "pyat",
            "served": True,
            "settings": {},
            "deck": None,
            "wiring": [
                _binding(
                    {
                        "id": "1",
                        "address": "M:HCM:SP",
                        "direction": "write",
                        "element": "HCM1",
                        "engine": {"attribute": "KickAngle", "index": 0},
                    },
                    "setpoint",
                    "x",
                    "pass",
                ),
                _binding(
                    {"id": "2", "address": "M:BPM:X", "element": "BPM1", "engine": {"axis": "x"}},
                    "monitor",
                    "x",
                    "pass",
                ),
                _binding(
                    {
                        "id": "3",
                        "address": "M:TUNE:X",
                        "engine": {"attribute": "tune", "axis": "x"},
                    },
                    "output",
                    "x",
                    "pass",
                ),
                _binding(
                    {
                        "id": "4",
                        "address": "M:CHROM:X",
                        "engine": {"attribute": "chromaticity", "axis": "x"},
                    },
                    "output",
                    "x",
                    "periodic",
                ),
                _binding(
                    {
                        "id": "5",
                        "address": "M:CHROM:Y",
                        "engine": {"attribute": "chromaticity", "index": 1},
                    },
                    "output",
                    "y",
                    "periodic",
                ),
                _binding(
                    {"id": "6", "address": "M:CHROM", "engine": {"attribute": "chromaticity"}},
                    "output",
                    None,
                    "periodic",
                ),
                _binding(
                    {
                        "id": "7",
                        "address": "M:SLICED",
                        "slices": [{"element": "Q1"}],
                        "engine": {"attribute": "chromaticity"},
                    },
                    "readback",
                    None,
                    "pass",
                ),
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


def _write_view(directory: Path, variables: dict[str, Any]) -> Path:
    """Write ``variables`` and its ``addresses.json`` as a view under ``directory``."""
    addresses = {
        "schema": SCHEMAS[ADDRESSES_FILE],
        "channels": sorted(channel["address"] for channel in variables["channels"]),
        "status": [STATUS],
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / ADDRESSES_FILE).write_text(json.dumps(addresses), encoding="utf-8")
    (directory / VARIABLES_FILE).write_text(json.dumps(variables), encoding="utf-8")
    return directory


@pytest.fixture
def view(tmp_path: Path) -> SimulatorView:
    return SimulatorView.open(_write_view(tmp_path / "view", VIEW))


def _generated(addresses: list[str]) -> dict[str, Any]:
    """A configuration in ``Runner.generate_config``'s shape over ``addresses``."""
    return {
        "description": "",
        "prefix": "",
        "max_array_bytes": "80000000",
        "variables": {
            address: {"name": address, "pv": address, "mode": "rw"} for address in addresses
        },
    }


#: The served addresses of :data:`VIEW`'s composite.
ADDRESSES = [channel["address"] for channel in VIEW["channels"]] + [STATUS]


def test_the_five_write_path_keys_are_set(view: SimulatorView) -> None:
    config = apply_safety(_generated(ADDRESSES), view)

    assert config["update_rate"] == 0.0
    assert config["echo_unconfirmed_writes"] is False
    assert config["alarm_on_refused_write"] is True
    assert config["clamp_writes"] is True
    assert config["control_pvs"] is False


def test_max_array_bytes_stays_a_string(view: SimulatorView) -> None:
    config = apply_safety(_generated(ADDRESSES), view)

    assert isinstance(config["max_array_bytes"], str)
    assert config["max_array_bytes"] == "80000000"


def test_every_setpoint_carries_the_view_band(view: SimulatorView) -> None:
    config = apply_safety(_generated(ADDRESSES), view)

    setpoints = [channel for channel in VIEW["channels"] if channel["role"] == "setpoint"]
    assert len(setpoints) == 3
    for channel in setpoints:
        assert config["variables"][channel["address"]]["value_range"] == channel["value_range"]


def test_a_non_setpoint_carries_no_band(view: SimulatorView) -> None:
    config = apply_safety(_generated(ADDRESSES), view)

    assert "value_range" not in config["variables"]["M:BPM:X"]
    assert "value_range" not in config["variables"][STATUS]


def test_only_a_writable_setpoint_is_read_write(view: SimulatorView) -> None:
    config = apply_safety(_generated(ADDRESSES), view)

    modes = {address: entry["mode"] for address, entry in config["variables"].items()}
    assert {address for address, mode in modes.items() if mode == "rw"} == {
        "M:HCM:SP",
        "T:OPEN:SP",
    }
    assert set(modes.values()) == {"rw", "ro"}
    assert modes[STATUS] == "ro"
    assert modes["T:LOCKED:SP"] == "ro"


def test_the_given_configuration_is_left_as_it_was(view: SimulatorView) -> None:
    given = _generated(ADDRESSES)
    before = copy.deepcopy(given)

    apply_safety(given, view)

    assert given == before


def test_a_float_channel_stating_precision_hands_it_to_the_runner(view: SimulatorView) -> None:
    config = apply_safety(_generated(ADDRESSES), view)

    assert config["variables"]["M:HCM:SP"]["precision"] == 4
    stated = {address for address, entry in config["variables"].items() if "precision" in entry}
    assert stated == {"M:HCM:SP"}


def test_periodic_addresses_are_the_bindings_refreshing_periodic(view: SimulatorView) -> None:
    assert periodic_addresses(view) == {"M:CHROM", "M:CHROM:X", "M:CHROM:Y"}


def test_a_view_without_periodic_bindings_names_none(tmp_path: Path) -> None:
    document = copy.deepcopy(VIEW)
    document["models"][0]["wiring"] = document["models"][0]["wiring"][:3]

    assert periodic_addresses(SimulatorView.open(_write_view(tmp_path, document))) == frozenset()


def _demo_view(built: BuiltProject) -> tuple[SimulatorView, dict[str, Any], dict[str, Any]]:
    """The built demo's view, and its ``variables.json`` and ``addresses.json`` as written."""
    view = SimulatorView.of_render(built.build_dir)
    variables = json.loads((view.path / VARIABLES_FILE).read_text(encoding="utf-8"))
    addresses = json.loads((view.path / ADDRESSES_FILE).read_text(encoding="utf-8"))
    return view, variables, addresses


def test_apply_safety_modes_unchanged_on_the_built_demo(
    built_control_assistant: BuiltProject,
) -> None:
    view, variables, addresses = _demo_view(built_control_assistant)
    served = [*addresses["channels"], *addresses["status"]]

    config = apply_safety(_generated(served), view)

    writable = {
        channel["address"]
        for channel in variables["channels"]
        if channel["role"] == "setpoint" and channel["writable"] is True
    }
    assert writable
    assert {address for address, entry in config["variables"].items() if entry["mode"] == "rw"} == (
        writable
    )


def test_periodic_addresses_equal_the_chromaticity_channels_on_the_demo(
    built_control_assistant: BuiltProject,
) -> None:
    view, variables, _ = _demo_view(built_control_assistant)

    chromaticity = {
        record["address"]
        for model in variables["models"]
        for record in model.get("wiring") or []
        if record.get("element") is None
        and not record.get("slices")
        and (record.get("engine") or {}).get("attribute") == "chromaticity"
    }
    assert chromaticity
    assert periodic_addresses(view) == chromaticity


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


def test_the_failed_pass_tolerance_is_a_runner_config_field() -> None:
    assert runner_config.HEALTH_KEYS == {"failed_pass_tolerance": 3}
    assert "HEALTH_KEYS" in runner_config.__all__
