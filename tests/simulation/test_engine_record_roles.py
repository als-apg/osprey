"""An engine plug-in's record roles and marks, from its declaration to the view's bindings."""

from __future__ import annotations

import sys
import types
from importlib import metadata
from typing import Any

import pytest

from osprey.facility.errors import FacilityBuildError
from osprey.facility.views.simulator import simulator_wiring
from osprey.simulation.engines import (
    ENTRY_POINT_GROUP,
    EnginePlugin,
    record_marks,
    record_roles,
)
from osprey_connectors.simulation.view import COMMAND_MARK, ROLES, Binding

FAKE = "fake_record_roles_engine"
FOUR_ROLES = ("setpoint", "readback", "monitor", "output")


def _function(*args: Any, **kwargs: Any) -> None:
    return None


def _engine(describe: Any, **declared: Any) -> types.ModuleType:
    module = types.ModuleType(FAKE)
    for name in ("locate", "prepare", "start_values", "build"):
        setattr(module, name, _function)
    module.describe = describe  # type: ignore[attr-defined]
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


def _facility(*wiring: dict[str, Any]) -> dict[str, Any]:
    return {
        "channels": [{"id": str(record["address"])} for record in wiring],
        "models": [{"name": "M", "engine": "fake", "wiring": list(wiring)}],
    }


WRITE = {"id": "w1", "address": "M:CMD", "direction": "write", "element": "E"}
READ = {"id": "w2", "address": "M:TBT", "direction": "read", "element": "E"}


def _describing(**by_address: dict[str, Any]) -> Any:
    def describe(record: Any) -> dict[str, Any]:
        return {"plane": None, "refresh": "pass", **by_address[record["address"]]}

    return describe


def test_the_declarations_are_no_protocol_members() -> None:
    module = _engine(_function)
    assert isinstance(module, EnginePlugin)
    assert (record_roles(module), record_marks(module)) == (FOUR_ROLES, ())


def test_a_declaring_module_serves_what_it_declares() -> None:
    module = _engine(
        _function, RECORD_ROLES=(*FOUR_ROLES, "turn_series"), RECORD_MARKS=("command",)
    )
    assert isinstance(module, EnginePlugin)
    assert record_roles(module) == (*FOUR_ROLES, "turn_series")
    assert record_marks(module) == ("command",)


def test_pyat_declares_the_four_roles_and_no_mark() -> None:
    from osprey.simulation.engines import pyat

    assert (record_roles(pyat), record_marks(pyat)) == (FOUR_ROLES, ())


def test_the_view_names_the_roles_an_undeclaring_engine_serves() -> None:
    assert ROLES == FOUR_ROLES
    assert COMMAND_MARK == "command"


def test_a_declared_mark_reaches_the_wiring_entry(register) -> None:
    register(
        _engine(
            _describing(**{"M:CMD": {"role": "setpoint", "command": True, "kind": "kick"}}),
            RECORD_MARKS=("command",),
        )
    )
    (entry,) = simulator_wiring(_facility(WRITE), "M")
    assert entry["marks"] == ["command"]
    assert "kind" not in entry and "command" not in entry


def test_an_undeclared_mark_is_dropped(register) -> None:
    register(_engine(_describing(**{"M:CMD": {"role": "setpoint", "command": True}})))
    (entry,) = simulator_wiring(_facility(WRITE), "M")
    assert "marks" not in entry


def test_a_false_mark_is_not_carried(register) -> None:
    register(
        _engine(
            _describing(**{"M:CMD": {"role": "setpoint", "command": False}}),
            RECORD_MARKS=("command",),
        )
    )
    (entry,) = simulator_wiring(_facility(WRITE), "M")
    assert "marks" not in entry


def test_only_a_write_record_may_carry_the_command_mark(register) -> None:
    register(
        _engine(
            _describing(**{"M:TBT": {"role": "monitor", "command": True}}),
            RECORD_MARKS=("command",),
        )
    )
    with pytest.raises(FacilityBuildError) as stop:
        simulator_wiring(_facility(READ), "M")
    assert (stop.value.kind, stop.value.record_id) == ("engine-invalid", "M")
    assert "M:TBT" in str(stop.value) and "command" in str(stop.value)


def test_a_declared_role_is_served(register) -> None:
    register(
        _engine(
            _describing(**{"M:TBT": {"role": "turn_series"}}),
            RECORD_ROLES=(*FOUR_ROLES, "turn_series"),
        )
    )
    (entry,) = simulator_wiring(_facility(READ), "M")
    assert entry["role"] == "turn_series"


def test_an_undeclared_role_stops_the_build(register) -> None:
    register(_engine(_describing(**{"M:TBT": {"role": "turn_series"}})))
    with pytest.raises(FacilityBuildError) as stop:
        simulator_wiring(_facility(READ), "M")
    assert stop.value.kind == "engine-invalid"
    assert "turn_series" in str(stop.value) and "RECORD_ROLES" in str(stop.value)


def test_a_binding_carries_the_records_marks() -> None:
    record = {"id": "w1", "address": "M:CMD", "direction": "write", "role": "setpoint"}
    record |= {"plane": None, "refresh": "pass"}
    assert Binding.from_record("M", record).marks == frozenset()
    marked = Binding.from_record("M", {**record, "marks": ["command"]})
    assert marked.marks == frozenset({"command"})
