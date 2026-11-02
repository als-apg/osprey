"""An engine plug-in's ``scenario_stops``: the scenario check's engine-side refusals."""

from __future__ import annotations

import sys
import types
from importlib import metadata
from typing import Any

import pytest

from osprey.facility.scenarios import check_scenario_engines
from osprey.simulation.engines import ENTRY_POINT_GROUP, EnginePlugin

FAKE = "fake_scenario_stops_engine"


def _function(*args: Any, **kwargs: Any) -> None:
    return None


def _engine(**declared: Any) -> types.ModuleType:
    module = types.ModuleType(FAKE)
    for name in ("locate", "prepare", "start_values", "describe", "build"):
        setattr(module, name, _function)
    for name, value in declared.items():
        setattr(module, name, value)
    return module


@pytest.fixture
def register(monkeypatch: pytest.MonkeyPatch):
    """Register a module as engine ``fake`` and nothing else."""

    def _register(module: types.ModuleType) -> None:
        monkeypatch.setitem(sys.modules, FAKE, module)
        found = metadata.EntryPoints(
            [metadata.EntryPoint(name="fake", value=FAKE, group=ENTRY_POINT_GROUP)]
        )

        def entry_points(**kwargs: Any) -> Any:
            return found

        monkeypatch.setattr(metadata, "entry_points", entry_points)

    return _register


WIRING = [{"id": "w1", "address": "M:CMD", "element": "E"}]


def _document(faults: dict[str, Any], *, engine: str = "fake") -> dict[str, Any]:
    return {
        "channels": [{"id": "M:CMD", "role": "setpoint"}],
        "models": [{"name": "M", "engine": engine, "wiring": WIRING}],
        "scenarios": [{"name": "drift", "faults": {"M": faults}}],
    }


def test_the_hook_is_no_protocol_member() -> None:
    assert isinstance(_engine(), EnginePlugin)


def test_an_engine_without_the_hook_yields_no_stops(register) -> None:
    register(_engine())
    assert check_scenario_engines(_document({"M:CMD": 1.0})) == []


def test_a_stop_refuses_the_build_naming_fault_and_remedy(register) -> None:
    def scenario_stops(wiring: Any, faults: Any) -> list[tuple[str, str, str]]:
        assert wiring
        return [
            (key, "a kick on a command record", "seed the kick on its readback") for key in faults
        ]

    register(_engine(scenario_stops=scenario_stops))
    (error,) = check_scenario_engines(_document({"M:CMD": 1.0}))
    assert error.kind == "engine-invalid"
    assert (error.record_kind, error.record_id) == ("scenario", "drift")
    assert error.sources == ("scenarios/drift.yaml",)
    assert error.detail == "`faults.M.M:CMD`: a kick on a command record"
    assert error.remedy == "seed the kick on its readback"


def test_the_hook_sees_every_fault_and_the_directed_wiring(register) -> None:
    seen: list[tuple[Any, Any]] = []

    def scenario_stops(wiring: Any, faults: Any) -> list[tuple[str, str, str]]:
        seen.append((wiring, faults))
        return []

    register(_engine(scenario_stops=scenario_stops))
    assert check_scenario_engines(_document({"M:CMD": 1.0})) == []
    assert seen == [([{**WIRING[0], "direction": "write"}], {"M:CMD": 1.0})]


def test_an_unregistered_engine_yields_no_stops(register) -> None:
    def scenario_stops(*args: Any) -> list[tuple[str, str, str]]:
        raise AssertionError("an unregistered engine is never asked")

    register(_engine(scenario_stops=scenario_stops))
    assert check_scenario_engines(_document({"M:CMD": 1.0}, engine="absent")) == []
