"""A wiring record's role and marks, from an engine's description to the view's bindings."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

import osprey
from osprey.facility import wiring
from osprey.facility.errors import FacilityBuildError
from osprey.facility.views.simulator import simulator_wiring
from osprey_connectors.simulation import view as view_module
from osprey_connectors.simulation.view import COMMAND_MARK, REFRESH, ROLES, Binding

MODEL = {"name": "M", "engine": "fake"}
WRITE = {"id": "M/M:CMD", "address": "M:CMD", "direction": "write", "element": "E"}
READ = {"id": "M/M:TBT", "address": "M:TBT", "direction": "read", "element": "E"}


def _describe(record: dict[str, Any], **said: Any) -> tuple[dict[str, Any], list[Any]]:
    """The record as the wiring stage leaves it, and the stops it reports."""
    filled = dict(record)
    errors = wiring._describe_record(
        lambda _: {"plane": None, "refresh": "pass", **said}, MODEL, filled, frozenset()
    )
    return filled, errors


def test_a_listed_mark_is_written_onto_the_record() -> None:
    filled, errors = _describe(WRITE, role="setpoint", marks=["command"], kind="kick")
    assert errors == []
    assert filled["marks"] == ["command"]
    assert "kind" not in filled
    assert "marks" in filled["provenance"]["defaults"]


def test_a_record_without_marks_carries_none() -> None:
    filled, errors = _describe(WRITE, role="setpoint")
    assert errors == []
    assert "marks" not in filled


def test_only_a_write_record_may_carry_the_command_mark() -> None:
    filled, errors = _describe(READ, role="monitor", marks=["command"])
    (stop,) = errors
    assert (stop.kind, stop.record_id) == ("engine-invalid", "M")
    assert "M:TBT" in str(stop) and "command" in str(stop)
    assert "role" not in filled


@pytest.mark.parametrize(
    ("said", "word"),
    [
        ({"role": "turn_series"}, "turn_series"),
        ({"role": "monitor", "marks": ["kick"]}, "kick"),
        ({"role": "monitor", "plane": "z"}, "z"),
        ({"role": "monitor", "refresh": "never"}, "never"),
    ],
)
def test_a_word_the_schema_does_not_state_stops_the_build(said: dict[str, Any], word: str) -> None:
    filled, errors = _describe(READ, **said)
    (stop,) = errors
    assert isinstance(stop, FacilityBuildError)
    assert stop.kind == "engine-invalid"
    assert word in str(stop) and "facility schema" in str(stop)
    assert "role" not in filled


def test_the_view_copies_the_marks_from_the_built_record() -> None:
    record = {**WRITE, "role": "setpoint", "plane": None, "refresh": "pass"}
    facility = {
        "channels": [{"id": "M:CMD"}],
        "models": [{**MODEL, "wiring": [{**record, "marks": ["command"]}]}],
    }
    (entry,) = simulator_wiring(facility, "M")
    assert entry["marks"] == ["command"]


def test_a_binding_carries_the_records_marks() -> None:
    record = {"id": "w1", "address": "M:CMD", "direction": "write", "role": "setpoint"}
    record |= {"plane": None, "refresh": "pass"}
    assert Binding.from_record("M", record).marks == frozenset()
    marked = Binding.from_record("M", {**record, "marks": [COMMAND_MARK]})
    assert marked.marks == frozenset({COMMAND_MARK})


#: The modules that read roles and refresh from the view.
ROLE_READERS = (
    Path(view_module.__file__).parent / "composite.py",
    Path(view_module.__file__).parent / "texture.py",
    Path(osprey.__file__).parent / "services/virtual_accelerator/serving/runner_config.py",
)


@pytest.mark.parametrize("path", ROLE_READERS, ids=lambda path: path.name)
def test_the_role_readers_name_no_role_of_their_own(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    named = [
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign | ast.AnnAssign)
        and isinstance(node.value, ast.Constant)
        and node.value.value in (*ROLES, *REFRESH)
    ]
    assert named == []
