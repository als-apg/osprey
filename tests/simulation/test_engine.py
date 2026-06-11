"""Tests for the SimulationEngine: schema, precedence, scenarios, noise."""

import os

import pytest

from osprey.simulation import SimulationEngine

QUAD_DRIFT_TRANS = 98.5 - 0.85 * abs(28.4 - 42.0)  # 86.94


class TestMachineFileLoading:
    """Schema load and validation errors."""

    def test_load_and_metadata(self, machine_file):
        engine = SimulationEngine.from_file(machine_file)
        assert engine.name == "TestRig"
        assert engine.has_channel("T:Q1:CUR:SP")
        assert not engine.has_channel("NOT:A:CHANNEL")
        scenarios = engine.list_scenarios()
        assert set(scenarios) == {"nominal", "quad-drift", "vac-leak"}
        assert scenarios["quad-drift"] == "Q1 left at a stale setpoint."

    def test_from_file_cached_by_path_and_mtime(self, machine_file):
        engine1 = SimulationEngine.from_file(machine_file)
        engine2 = SimulationEngine.from_file(machine_file)
        assert engine1 is engine2

        # Touching the file invalidates the cache
        os.utime(machine_file, ns=(1, 1))
        engine3 = SimulationEngine.from_file(machine_file)
        assert engine3 is not engine1

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            SimulationEngine.from_file(tmp_path / "missing.json")

    def test_nominal_injected_when_absent(self, machine_dict, make_machine_file):
        del machine_dict["scenarios"]
        engine = SimulationEngine.from_file(make_machine_file(machine_dict))
        assert engine.active_scenario() == "nominal"
        assert "nominal" in engine.list_scenarios()

    def test_value_and_expr_both_rejected(self, machine_dict, make_machine_file):
        machine_dict["channels"]["T:BAD"] = {"value": 1.0, "expr": "1 + 1"}
        with pytest.raises(ValueError, match="exactly one of 'value' or 'expr'"):
            SimulationEngine.from_file(make_machine_file(machine_dict))

    def test_neither_value_nor_expr_rejected(self, machine_dict, make_machine_file):
        machine_dict["channels"]["T:BAD"] = {"units": "A"}
        with pytest.raises(ValueError, match="exactly one of 'value' or 'expr'"):
            SimulationEngine.from_file(make_machine_file(machine_dict))

    def test_invalid_expression_rejected(self, machine_dict, make_machine_file):
        machine_dict["channels"]["T:BAD"] = {"expr": "__import__('os')"}
        with pytest.raises(ValueError, match="T:BAD"):
            SimulationEngine.from_file(make_machine_file(machine_dict))

    def test_unknown_reference_rejected(self, machine_dict, make_machine_file):
        machine_dict["channels"]["T:BAD"] = {"expr": "ch('NO:SUCH:PV')"}
        with pytest.raises(ValueError, match="unknown channel 'NO:SUCH:PV'"):
            SimulationEngine.from_file(make_machine_file(machine_dict))

    def test_reference_cycle_rejected(self, machine_dict, make_machine_file):
        machine_dict["channels"]["T:A"] = {"expr": "ch('T:B') + 1"}
        machine_dict["channels"]["T:B"] = {"expr": "ch('T:A') + 1"}
        with pytest.raises(ValueError, match="cycle"):
            SimulationEngine.from_file(make_machine_file(machine_dict))

    def test_negative_noise_rejected(self, machine_dict, make_machine_file):
        machine_dict["channels"]["T:BAD"] = {"value": 1.0, "noise": -0.1}
        with pytest.raises(ValueError, match="noise"):
            SimulationEngine.from_file(make_machine_file(machine_dict))

    def test_override_for_unknown_channel_rejected(self, machine_dict, make_machine_file):
        machine_dict["scenarios"]["nominal"]["overrides"] = {"NO:SUCH:PV": 1.0}
        with pytest.raises(ValueError, match="override for unknown channel"):
            SimulationEngine.from_file(make_machine_file(machine_dict))

    def test_archiver_events_unknown_channel_rejected(self, machine_dict, make_machine_file):
        machine_dict["scenarios"]["nominal"]["archiver"] = [
            {"channel": "NO:SUCH:PV", "events": [{"shape": "step", "at": 0.5, "to": 1.0}]}
        ]
        with pytest.raises(ValueError, match="unknown channel"):
            SimulationEngine.from_file(make_machine_file(machine_dict))

    def test_bad_event_shape_rejected(self, machine_dict, make_machine_file):
        machine_dict["scenarios"]["nominal"]["archiver"] = [
            {"channel": "T:VAC", "events": [{"shape": "wiggle", "at": 0.5}]}
        ]
        with pytest.raises(ValueError, match="shape"):
            SimulationEngine.from_file(make_machine_file(machine_dict))

    def test_ramp_missing_until_rejected(self, machine_dict, make_machine_file):
        machine_dict["scenarios"]["nominal"]["archiver"] = [
            {"channel": "T:VAC", "events": [{"shape": "ramp", "at": 0.1, "to": 1.0}]}
        ]
        with pytest.raises(ValueError, match="missing keys"):
            SimulationEngine.from_file(make_machine_file(machine_dict))


class TestReadsAndPrecedence:
    """Value precedence: session write > scenario override > baseline."""

    def test_baseline_value_read(self, machine_file):
        engine = SimulationEngine.from_file(machine_file)
        reading = engine.read("T:Q1:CUR:SP")
        assert reading.value == 42.0
        assert reading.units == "A"
        assert "nominal 42.0 A" in reading.description

    def test_baseline_expr_read(self, machine_file):
        engine = SimulationEngine.from_file(machine_file)
        assert engine.read("T:Q1:CUR:RB").value == 42.0
        assert engine.read("T:TRANS").value == pytest.approx(98.5)

    def test_scenario_override_beats_baseline(self, machine_file):
        engine = SimulationEngine.from_file(machine_file)
        engine.set_active_scenario("quad-drift")
        assert engine.read("T:Q1:CUR:SP").value == 28.4
        # Override propagates through derived channels
        assert engine.read("T:TRANS").value == pytest.approx(QUAD_DRIFT_TRANS)
        assert engine.read("T:Q1:CUR:RB").value == 28.4

    def test_write_beats_scenario_override(self, machine_file):
        engine = SimulationEngine.from_file(machine_file)
        engine.set_active_scenario("quad-drift")
        engine.write("T:Q1:CUR:SP", 42.0)
        assert engine.read("T:Q1:CUR:SP").value == 42.0
        assert engine.read("T:TRANS").value == pytest.approx(98.5)

    def test_status_override_propagates(self, machine_file):
        engine = SimulationEngine.from_file(machine_file)
        engine.set_active_scenario("vac-leak")
        assert engine.read("T:RF:STATUS").value == 0.0
        assert engine.read("T:TRANS").value == 0.0

    def test_scenario_switch_clears_writes(self, machine_file):
        engine = SimulationEngine.from_file(machine_file)
        engine.write("T:Q1:CUR:SP", 10.0)
        assert engine.read("T:Q1:CUR:SP").value == 10.0

        engine.set_active_scenario("quad-drift")
        assert engine.read("T:Q1:CUR:SP").value == 28.4  # write cleared, override applies

        engine.set_active_scenario("nominal")
        assert engine.read("T:Q1:CUR:SP").value == 42.0  # fresh machine

    def test_string_channel(self, machine_file):
        engine = SimulationEngine.from_file(machine_file)
        assert engine.read("T:MODE").value == "CW"
        engine.set_active_scenario("vac-leak")
        assert engine.read("T:MODE").value == "FAULT"

    def test_unknown_channel_raises(self, machine_file):
        engine = SimulationEngine.from_file(machine_file)
        with pytest.raises(KeyError):
            engine.read("NO:SUCH:PV")
        with pytest.raises(KeyError):
            engine.write("NO:SUCH:PV", 1.0)

    def test_set_unknown_scenario_raises(self, machine_file):
        engine = SimulationEngine.from_file(machine_file)
        with pytest.raises(ValueError, match="Unknown scenario"):
            engine.set_active_scenario("does-not-exist")


class TestNoise:
    """Noise semantics: value * (1 + N(0, noise)); strings and noise=0 untouched."""

    def test_noise_zero_is_exact(self, machine_file):
        engine = SimulationEngine.from_file(machine_file)
        values = {engine.read("T:Q1:CUR:SP").value for _ in range(20)}
        assert values == {42.0}

    def test_noisy_channel_varies(self, machine_file):
        engine = SimulationEngine.from_file(machine_file)
        values = [engine.read("T:NOISY").value for _ in range(50)]
        assert len(set(values)) > 1
        mean = sum(values) / len(values)
        assert mean == pytest.approx(100.0, rel=0.1)

    def test_string_channel_never_noisy(self, machine_dict, make_machine_file):
        machine_dict["channels"]["T:MODE"]["noise"] = 0.5
        engine = SimulationEngine.from_file(make_machine_file(machine_dict))
        assert engine.read("T:MODE").value == "CW"


class TestActiveScenarioStateFile:
    """Plain-text state file next to the machine file, mtime-based re-read."""

    def test_missing_file_means_nominal(self, machine_file):
        engine = SimulationEngine.from_file(machine_file)
        assert engine.active_scenario() == "nominal"

    def test_state_file_read_on_mtime_change(self, machine_file):
        engine = SimulationEngine.from_file(machine_file)
        assert engine.active_scenario() == "nominal"

        state_file = machine_file.parent / "active_scenario"
        state_file.write_text("quad-drift\n")
        os.utime(state_file, ns=(10**9, 10**9))
        assert engine.active_scenario() == "quad-drift"
        assert engine.read("T:Q1:CUR:SP").value == 28.4

        state_file.write_text("nominal\n")
        os.utime(state_file, ns=(2 * 10**9, 2 * 10**9))
        assert engine.active_scenario() == "nominal"

    def test_unknown_name_falls_back_to_nominal_with_warning(self, machine_file, caplog):
        engine = SimulationEngine.from_file(machine_file)
        state_file = machine_file.parent / "active_scenario"
        state_file.write_text("bogus-scenario\n")
        os.utime(state_file, ns=(10**9, 10**9))
        with caplog.at_level("WARNING"):
            assert engine.active_scenario() == "nominal"
        assert "bogus-scenario" in caplog.text

    def test_external_switch_clears_writes(self, machine_file):
        engine = SimulationEngine.from_file(machine_file)
        engine.write("T:Q1:CUR:SP", 10.0)

        state_file = machine_file.parent / "active_scenario"
        state_file.write_text("quad-drift\n")
        os.utime(state_file, ns=(10**9, 10**9))
        assert engine.read("T:Q1:CUR:SP").value == 28.4

    def test_set_active_scenario_writes_state_file(self, machine_file):
        engine = SimulationEngine.from_file(machine_file)
        engine.set_active_scenario("vac-leak")
        state_file = machine_file.parent / "active_scenario"
        assert state_file.read_text().strip() == "vac-leak"


class TestWriteCoercion:
    """MCP/CLI write paths deliver strings; numeric strings must be coerced."""

    def test_numeric_string_write_coerced(self, machine_file):
        engine = SimulationEngine.from_file(machine_file)
        engine.write("T:Q1:CUR:SP", "37.5")
        assert engine.read("T:Q1:CUR:SP").value == 37.5
        # Derived channels referencing the written one still evaluate
        assert engine.read("T:Q1:CUR:RB").value == 37.5

    def test_integer_string_write_coerced(self, machine_file):
        engine = SimulationEngine.from_file(machine_file)
        engine.write("T:RF:STATUS", "0")
        assert engine.read("T:TRANS").value == 0.0

    def test_non_numeric_string_stays_string(self, machine_file):
        engine = SimulationEngine.from_file(machine_file)
        engine.write("T:MODE", "FAULT")
        assert engine.read("T:MODE").value == "FAULT"


class TestSameScenarioReset:
    """Re-asserting the active scenario resets session writes (fresh machine)."""

    def test_set_active_scenario_same_name_clears_writes(self, machine_file):
        engine = SimulationEngine.from_file(machine_file)
        engine.set_active_scenario("quad-drift")
        engine.write("T:Q1:CUR:SP", 99.0)
        assert engine.read("T:Q1:CUR:SP").value == 99.0

        engine.set_active_scenario("quad-drift")
        assert engine.read("T:Q1:CUR:SP").value == 28.4  # write cleared

    def test_state_file_reassert_clears_writes(self, machine_file):
        engine = SimulationEngine.from_file(machine_file)
        state_file = machine_file.parent / "active_scenario"
        state_file.write_text("quad-drift\n")
        os.utime(state_file, ns=(10**9, 10**9))
        assert engine.active_scenario() == "quad-drift"

        engine.write("T:Q1:CUR:SP", 99.0)
        state_file.write_text("quad-drift\n")
        os.utime(state_file, ns=(2 * 10**9, 2 * 10**9))
        assert engine.read("T:Q1:CUR:SP").value == 28.4  # write cleared
