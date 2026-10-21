"""The OSPREY control system binds a pyAML configuration to the OSPREY runtime."""

from __future__ import annotations

import asyncio
import copy
import subprocess
import sys
import textwrap
import warnings
from typing import Any
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from pyaml.accelerator import Accelerator
from pyaml.common.exception import PyAMLException

import osprey.runtime
from pyaml_cs_osprey.controlsystem import PYAMLCLASS, OspreyControlSystem
from pyaml_cs_osprey.device import OspreyDevice
from pyaml_cs_osprey.devices import OspreyDeviceList
from tests.pyaml_cs_osprey.conftest import DictConnector

BPM_REFS = {
    "BPM_001": ("BPM_001:x[m]", "BPM_001:y[m]"),
    "BPM_002": ("BPM_002:x[m]", "BPM_002:y[m]"),
}

CONFIG: dict[str, Any] = {
    "type": "pyaml.accelerator",
    "facility": "Test",
    "machine": "sr",
    "energy": 1.0e9,
    "controls": [{"type": "pyaml_cs_osprey.controlsystem", "name": "live"}],
    "arrays": [
        {"type": "pyaml.arrays.bpm", "name": "BPMS", "elements": list(BPM_REFS)},
    ],
    "devices": [
        {
            "type": "pyaml.magnet.quadrupole",
            "name": "QF_001",
            "model": {
                "type": "pyaml.magnet.identity_model",
                "physics": "(QF_001:Cm:rdbk, QF_001:Cm:set)[1/m]",
                "unit": "1/m",
            },
        },
        {
            "type": "pyaml.magnet.quadrupole",
            "name": "QD_001",
            "model": {
                "type": "pyaml.magnet.identity_model",
                "physics": "(QD_001:Cm:rdbk, QD_001:Cm:set)[1/m]",
                "unit": "1/m",
            },
        },
        *(
            {"type": "pyaml.bpm.bpm", "name": name, "x_pos": x, "y_pos": y}
            for name, (x, y) in BPM_REFS.items()
        ),
    ],
}


def _config() -> dict[str, Any]:
    return copy.deepcopy(CONFIG)


@pytest.fixture
def no_runtime(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Fail loudly if anything reaches the OSPREY runtime; record the attempts."""
    calls: list[str] = []

    def refuse(name: str):
        def call(*args: Any, **kwargs: Any) -> Any:
            calls.append(name)
            raise AssertionError(f"building a configuration must not call {name}")

        return call

    for name in ("read_channel", "read_channels", "write_channel", "write_channels"):
        monkeypatch.setattr(osprey.runtime, name, refuse(name))
    return calls


# --- module contract -----------------------------------------------------------


def test_pyamlclass_names_the_control_system() -> None:
    import pyaml_cs_osprey.controlsystem as module

    assert PYAMLCLASS == "OspreyControlSystem"
    assert getattr(module, PYAMLCLASS) is OspreyControlSystem


def test_name_is_the_only_configuration() -> None:
    cs = OspreyControlSystem("live")
    assert cs.name() == "live"
    with pytest.raises(TypeError):
        OspreyControlSystem("live", catalog={})  # type: ignore[call-arg]


# --- catalog rules ---------------------------------------------------------------


def test_none_reference_is_no_device() -> None:
    assert OspreyControlSystem("live").get_device_access(None) is None


def test_one_device_per_reference_text() -> None:
    cs = OspreyControlSystem("live")
    ref = "(QF_001:Cm:rdbk, QF_001:Cm:set)[1/m]"
    first = cs.get_device_access(ref)
    assert isinstance(first, OspreyDevice)
    assert cs.get_device_access(ref) is first
    assert first.name() == "QF_001:Cm:set"
    assert first.measure_name() == "QF_001:Cm:rdbk"
    assert cs.get_device_access("(QD_001:Cm:rdbk, QD_001:Cm:set)[1/m]") is not first


@pytest.mark.parametrize("ref", ["(wave:set)@0[A]", "(RB, SP)@0[A]"])
def test_indexed_references_are_refused_by_name(ref: str) -> None:
    cs = OspreyControlSystem("live")
    with pytest.raises(PyAMLException) as info:
        cs.get_device_access(ref)
    assert ref in str(info.value)
    assert cs._devices == {}


@pytest.mark.parametrize("ref", ["beam:orbit:x@3[m]", "beam:orbit:x @ 12"])
def test_bare_index_loads_read_only_parenthesised_refused(ref: str, no_runtime: list[str]) -> None:
    cs = OspreyControlSystem("live")
    device = cs.get_device_access(ref)
    assert isinstance(device, OspreyDevice)
    assert cs.get_device_access(ref) is device
    with pytest.raises(PyAMLException) as info:
        device.set(1.0)
    assert ref in str(info.value)
    assert "write_channel" not in no_runtime
    assert no_runtime == []
    for refused in ("(wave:set)@0[A]", "(RB, SP)@0[A]"):
        with pytest.raises(PyAMLException) as refusal:
            cs.get_device_access(refused)
        assert refused in str(refusal.value)


def test_malformed_reference_raises_pyaml_exception() -> None:
    with pytest.raises(PyAMLException):
        OspreyControlSystem("live").get_device_access("(unterminated[A]")


def test_get_devices_access_maps_the_list_in_order() -> None:
    cs = OspreyControlSystem("live")
    refs = ["A:x[m]", None, "A:x[m]", "(B:rb, B:sp)[A]"]
    devices = cs.get_devices_access(refs)
    assert devices[1] is None
    assert devices[0] is devices[2]
    assert [d.name() for d in (devices[0], devices[3])] == ["A:x", "B:sp"]


def test_get_devices_access_refuses_a_non_list() -> None:
    with pytest.raises(PyAMLException):
        OspreyControlSystem("live").get_devices_access("A:x[m]")  # type: ignore[arg-type]


def test_every_aggregator_is_new_and_empty() -> None:
    cs = OspreyControlSystem("live")
    first, second = cs.get_aggregator(), cs.get_aggregator()
    assert isinstance(first, OspreyDeviceList)
    assert first is not second
    assert first.len() == 0 and second.len() == 0


def test_three_bpm_aggregators_share_no_devices(no_runtime: list[str]) -> None:
    sr = Accelerator.from_dict(_config())
    bpms = [sr.live.bpm.get(name) for name in BPM_REFS]
    aggregators = sr.live.create_bpm_aggregators(bpms)
    assert len({id(x) for x in aggregators}) == 3
    inner = [_inner_list(agg) for agg in aggregators]
    assert len({id(x) for x in inner}) == 3
    assert [x.len() for x in inner] == [4, 2, 2]
    assert [repr(d) for d in _devices(inner[1])] == ["BPM_001:x[m]", "BPM_002:x[m]"]
    assert [repr(d) for d in _devices(inner[2])] == ["BPM_001:y[m]", "BPM_002:y[m]"]
    assert no_runtime == []


def _inner_list(aggregator: Any) -> OspreyDeviceList:
    """The OSPREY device list a pyAML scalar aggregator wraps."""
    found = [v for v in vars(aggregator).values() if isinstance(v, OspreyDeviceList)]
    assert len(found) == 1, vars(aggregator)
    return found[0]


def _devices(device_list: OspreyDeviceList) -> list[Any]:
    return [device_list.get_device_at(i) for i in range(device_list.len())]


# --- building an accelerator -----------------------------------------------------


def test_from_dict_builds_live_without_an_event_loop(no_runtime: list[str]) -> None:
    sr = Accelerator.from_dict(_config())
    assert isinstance(sr.live, OspreyControlSystem)
    assert sr.live.name() == "live"
    with pytest.raises(RuntimeError):
        asyncio.get_running_loop()
    assert no_runtime == []


def test_bpm_array_reads_positions_over_a_connector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(osprey.runtime, "_limits_validator", None)
    sr = Accelerator.from_dict(_config())
    connector = DictConnector(
        {"BPM_001:x": 1.0e-3, "BPM_001:y": -2.0e-3, "BPM_002:x": 3.0e-3, "BPM_002:y": 4.0e-3}
    )
    with patch("osprey.runtime._get_connector", new_callable=AsyncMock) as get:
        get.return_value = connector
        positions = sr.live.bpms.get("BPMS").positions.get()
    np.testing.assert_allclose(positions, [[1.0e-3, -2.0e-3], [3.0e-3, 4.0e-3]])


# --- schema discovery --------------------------------------------------------------


def test_entry_point_discovery_registers_the_control_system() -> None:
    script = textwrap.dedent(
        """
        import sys
        from importlib.metadata import entry_points

        assert "pyaml_cs_osprey" not in sys.modules
        (ep,) = [e for e in entry_points(group="pyaml.schemas") if e.name == "pyaml_cs_osprey"]
        assert ep.attr is None, ep.attr
        from pyaml.validation.registry import SchemaRegistry

        registry = SchemaRegistry()
        registry.discover()
        key = "pyaml_cs_osprey.controlsystem.OspreyControlSystem"
        assert key in list(registry.keys()), sorted(registry.keys())
        print("ok")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_validated_build_knows_the_schema() -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Unknown schema")
        sr = Accelerator.from_dict(_config(), validate=True)
    assert sr.live.name() == "live"
