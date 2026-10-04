"""The simulation package root and its state helpers."""

import json
import subprocess
import sys

import pytest

from osprey_connectors.simulation import state, values


def test_engine_takes_the_state_names_from_the_state_module():
    from osprey_connectors.simulation import engine

    assert engine.ACTIVE_SCENARIOS_FILENAME is state.ACTIVE_SCENARIOS_FILENAME
    assert engine.resolve_active_scenarios is state.resolve_active_scenarios


def test_two_scenarios_writing_one_address_give_one_overlap_naming_it():
    view = {"a": {"SR:BPM1:X", "SR:Q1:I"}, "b": {"SR:BPM1:X", "SR:Q2:I"}}

    overlaps = state.validate_composition(view, ["nominal", "a", "b"])

    assert overlaps == [state.Overlap(target="SR:BPM1:X", first="a", second="b")]
    assert "SR:BPM1:X" in str(overlaps[0])


def test_disjoint_scenarios_compose():
    view = {"a": {"X"}, "b": {"Y"}}

    assert state.validate_composition(view, ["a", "b"]) == []


def test_an_unknown_scenario_is_refused_by_name():
    with pytest.raises(ValueError, match="'c'"):
        state.validate_composition({"a": {"X"}}, ["a", "c"])


def test_overlap_record_prints_its_log_origin():
    overlap = state.Overlap(target="SR:BPM1:X", first="a", second="b")

    record = state.overlap_record(overlap, instance="live_standin", pid=4242)
    line = state.format_overlap_record("SR", record)

    assert record == {
        "instance": "live_standin",
        "pid": 4242,
        "event": state.OVERLAP_EVENT,
        "target": "SR:BPM1:X",
    }
    assert line.startswith("SR (log, instance live_standin, pid 4242): ")
    assert "SR:BPM1:X" in line


def _modules_after(statement: str) -> set[str]:
    """The modules a fresh interpreter holds after running ``statement``."""
    code = f"import sys; {statement}; print('\\n'.join(sorted(sys.modules)))"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return set(result.stdout.split())


def test_the_package_root_imports_no_lume_and_no_engine():
    modules = _modules_after("import osprey_connectors.simulation")

    assert not {m for m in modules if m == "lume" or m.startswith(("lume.", "lume_"))}
    assert not {
        m
        for m in modules
        if m.startswith("osprey_connectors.simulation.")
        and m.rsplit(".", 1)[1] in {"engine", "composite", "texture", "archive"}
    }
    assert "numpy" not in modules


def test_the_values_module_imports_no_numpy_and_no_lume():
    modules = _modules_after("import osprey_connectors.simulation.values")

    assert "numpy" not in modules
    assert not {m for m in modules if m == "lume" or m.startswith(("lume.", "lume_"))}


def test_the_package_root_reexports_the_state_helpers():
    import osprey_connectors.simulation as package

    for name in state.__all__:
        assert getattr(package, name) is getattr(state, name)
    assert package.coerce is values.coerce
    assert not hasattr(package, "SimulationEngine")


@pytest.mark.asyncio
async def test_the_mock_connector_still_connects_on_a_machine_file(tmp_path, monkeypatch):
    from osprey_connectors.control_system.mock_connector import MockConnector
    from osprey_connectors.simulation import engine

    monkeypatch.setattr(engine, "default_state_dir", lambda: tmp_path)
    machine = tmp_path / "machine.json"
    machine.write_text(
        json.dumps({"name": "m", "channels": {"A:B": {"value": 1.0, "units": "mm"}}})
    )
    connector = MockConnector()
    await connector.connect({"response_delay_ms": 0, "simulation_file": str(machine)})
    try:
        assert connector._sim_engine is not None
        assert connector._sim_engine.has_channel("A:B")
    finally:
        await connector.disconnect()
