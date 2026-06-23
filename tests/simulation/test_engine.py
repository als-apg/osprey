"""Tests for the SimulationEngine: schema, precedence, scenarios, noise."""

import os

import pytest

from osprey.simulation import SimulationEngine, engine_serves
from osprey.simulation.expressions import ExpressionError

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


class TestChannelBoundsValidation:
    """Optional ``min``/``max`` physical bounds are validated at load time."""

    def test_non_number_min_rejected(self, machine_dict, make_machine_file):
        machine_dict["channels"]["T:BAD"] = {"value": 1.0, "min": "zero"}
        with pytest.raises(ValueError, match="T:BAD.*'min'.*'zero'"):
            SimulationEngine.from_file(make_machine_file(machine_dict))

    def test_bool_max_rejected(self, machine_dict, make_machine_file):
        machine_dict["channels"]["T:BAD"] = {"value": 1.0, "max": True}
        with pytest.raises(ValueError, match="T:BAD.*'max'"):
            SimulationEngine.from_file(make_machine_file(machine_dict))

    def test_min_not_less_than_max_rejected(self, machine_dict, make_machine_file):
        machine_dict["channels"]["T:BAD"] = {"value": 1.0, "min": 5.0, "max": 5.0}
        with pytest.raises(ValueError, match="'min'.*must be less than 'max'"):
            SimulationEngine.from_file(make_machine_file(machine_dict))

    def test_bounds_on_string_channel_rejected(self, machine_dict, make_machine_file):
        machine_dict["channels"]["T:STR"] = {"value": "CW", "min": 0.0}
        with pytest.raises(ValueError, match="not supported on string-valued"):
            SimulationEngine.from_file(make_machine_file(machine_dict))


class TestChannelBoundsReads:
    """``min``/``max`` clamp live reads on the way out, not stored state."""

    def test_override_below_min_is_clamped_on_read(self, machine_dict, make_machine_file):
        machine_dict["channels"]["T:PWR"] = {"value": 5.0, "noise": 0.0, "min": 0.0}
        machine_dict["scenarios"]["quad-drift"]["overrides"]["T:PWR"] = -10.0
        engine = SimulationEngine.from_file(make_machine_file(machine_dict))
        engine.set_active_scenario("quad-drift")
        assert engine.read("T:PWR").value == 0.0  # clamped output
        engine.set_active_scenario("nominal")
        assert engine.read("T:PWR").value == 5.0  # stored override intact

    def test_max_clamps_read(self, machine_dict, make_machine_file):
        machine_dict["channels"]["T:PWR"] = {"value": 5.0, "noise": 0.0, "max": 3.0}
        engine = SimulationEngine.from_file(make_machine_file(machine_dict))
        assert engine.read("T:PWR").value == 3.0

    def test_derived_read_sees_clamped_inputs(self, machine_dict, make_machine_file):
        machine_dict["channels"]["T:FWD"] = {"value": 5.0, "noise": 0.0, "min": 0.0}
        machine_dict["channels"]["T:NET"] = {"expr": "ch('T:FWD') - 2.0", "noise": 0.0}
        machine_dict["scenarios"]["quad-drift"]["overrides"]["T:FWD"] = -100.0
        engine = SimulationEngine.from_file(make_machine_file(machine_dict))
        engine.set_active_scenario("quad-drift")
        # T:FWD clamps to 0.0 before the expression sees it: NET = 0.0 - 2.0.
        assert engine.read("T:NET").value == pytest.approx(-2.0)


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

    def test_set_active_scenario_writes_canonical_state_file(self, machine_file):
        engine = SimulationEngine.from_file(machine_file)
        engine.set_active_scenario("vac-leak")
        # Writes always target the canonical multi-scenario file (nominal implicit).
        state_file = machine_file.parent / "active_scenarios"
        assert state_file.read_text().strip() == "vac-leak"
        assert engine.active_scenarios() == ("nominal", "vac-leak")


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


class TestExpressionRuntimeErrorContext:
    """Runtime math errors surface as ExpressionError naming the channel."""

    def test_read_division_by_zero_names_channel(self, machine_dict, make_machine_file):
        machine_dict["channels"]["T:ZERO"] = {"value": 0.0, "description": "zero"}
        machine_dict["channels"]["T:RATIO"] = {
            "expr": "100.0 / ch('T:ZERO')",
            "description": "ratio",
        }
        engine = SimulationEngine.from_file(make_machine_file(machine_dict))
        with pytest.raises(ExpressionError, match=r"T:RATIO.*100\.0 / ch"):
            engine.read("T:RATIO")

    def test_read_sqrt_of_negative_names_channel(self, machine_dict, make_machine_file):
        machine_dict["channels"]["T:NEG"] = {"value": -1.0, "description": "negative"}
        machine_dict["channels"]["T:ROOT"] = {
            "expr": "sqrt(ch('T:NEG'))",
            "description": "root",
        }
        engine = SimulationEngine.from_file(make_machine_file(machine_dict))
        with pytest.raises(ExpressionError, match="T:ROOT"):
            engine.read("T:ROOT")

    def test_series_synthesis_error_names_channel(self, machine_dict, make_machine_file):
        machine_dict["channels"]["T:ZERO"] = {"value": 0.0, "description": "zero"}
        machine_dict["channels"]["T:RATIO"] = {
            "expr": "100.0 / ch('T:ZERO')",
            "description": "ratio",
        }
        engine = SimulationEngine.from_file(make_machine_file(machine_dict))
        with pytest.raises(ExpressionError, match="T:RATIO"):
            engine.synthesize_series("T:RATIO", list(range(10)))

    def test_string_ref_error_names_outer_channel(self, machine_dict, make_machine_file):
        machine_dict["channels"]["T:DERIVED"] = {
            "expr": "2.0 * ch('T:MODE')",
            "description": "derived from string channel",
        }
        engine = SimulationEngine.from_file(make_machine_file(machine_dict))
        with pytest.raises(ExpressionError, match="T:DERIVED.*T:MODE"):
            engine.read("T:DERIVED")


class TestEventParamValidation:
    """Archiver event params are type/range-checked at load time."""

    def _machine_with_event(self, machine_dict, channel, event):
        machine_dict["scenarios"]["nominal"]["archiver"] = [{"channel": channel, "events": [event]}]
        return machine_dict

    def test_non_numeric_at_rejected(self, machine_dict, make_machine_file):
        bad = self._machine_with_event(
            machine_dict, "T:VAC", {"shape": "step", "at": "0.5", "to": 1.0}
        )
        with pytest.raises(ValueError, match="'at' must be a number"):
            SimulationEngine.from_file(make_machine_file(bad))

    def test_bool_at_rejected(self, machine_dict, make_machine_file):
        bad = self._machine_with_event(
            machine_dict, "T:VAC", {"shape": "step", "at": True, "to": 1.0}
        )
        with pytest.raises(ValueError, match="'at' must be a number"):
            SimulationEngine.from_file(make_machine_file(bad))

    def test_at_out_of_window_rejected(self, machine_dict, make_machine_file):
        bad = self._machine_with_event(
            machine_dict, "T:VAC", {"shape": "step", "at": 1.5, "to": 1.0}
        )
        with pytest.raises(ValueError, match="between 0 and 1"):
            SimulationEngine.from_file(make_machine_file(bad))

    def test_zero_width_spike_rejected(self, machine_dict, make_machine_file):
        bad = self._machine_with_event(
            machine_dict, "T:VAC", {"shape": "spike", "at": 0.5, "amplitude": 1.0, "width": 0}
        )
        with pytest.raises(ValueError, match="'width' must be .* > 0"):
            SimulationEngine.from_file(make_machine_file(bad))

    def test_ramp_on_string_channel_rejected(self, machine_dict, make_machine_file):
        bad = self._machine_with_event(
            machine_dict, "T:MODE", {"shape": "ramp", "at": 0.1, "until": 0.5, "to": 1.0}
        )
        with pytest.raises(ValueError, match="string-valued"):
            SimulationEngine.from_file(make_machine_file(bad))

    def test_spike_on_string_channel_rejected(self, machine_dict, make_machine_file):
        bad = self._machine_with_event(
            machine_dict, "T:MODE", {"shape": "spike", "at": 0.5, "amplitude": 1.0, "width": 0.1}
        )
        with pytest.raises(ValueError, match="string-valued"):
            SimulationEngine.from_file(make_machine_file(bad))

    def test_non_numeric_to_on_numeric_channel_rejected(self, machine_dict, make_machine_file):
        bad = self._machine_with_event(
            machine_dict, "T:VAC", {"shape": "step", "at": 0.5, "to": "FAULT"}
        )
        with pytest.raises(ValueError, match="'to' must be a number"):
            SimulationEngine.from_file(make_machine_file(bad))

    def test_string_step_on_string_channel_allowed(self, machine_dict, make_machine_file):
        good = self._machine_with_event(
            machine_dict, "T:MODE", {"shape": "step", "at": 0.5, "to": "FAULT"}
        )
        engine = SimulationEngine.from_file(make_machine_file(good))
        assert engine.has_channel("T:MODE")


class TestOffsetEventValidation:
    """at_offset / until_offset variants are validated at load time."""

    def _machine_with_event(self, machine_dict, event):
        machine_dict["scenarios"]["nominal"]["archiver"] = [{"channel": "T:VAC", "events": [event]}]
        return machine_dict

    def test_at_and_at_offset_together_rejected(self, machine_dict, make_machine_file):
        bad = self._machine_with_event(
            machine_dict, {"shape": "step", "at": 0.5, "at_offset": -10, "to": 1.0}
        )
        with pytest.raises(ValueError, match="exactly one"):
            SimulationEngine.from_file(make_machine_file(bad))

    def test_neither_at_nor_at_offset_rejected(self, machine_dict, make_machine_file):
        bad = self._machine_with_event(machine_dict, {"shape": "step", "to": 1.0})
        with pytest.raises(ValueError, match="exactly one"):
            SimulationEngine.from_file(make_machine_file(bad))

    def test_ramp_mixing_offset_and_fraction_rejected(self, machine_dict, make_machine_file):
        bad = self._machine_with_event(
            machine_dict, {"shape": "ramp", "at_offset": -60, "until": 0.9, "to": 1.0}
        )
        with pytest.raises(ValueError, match="mix"):
            SimulationEngine.from_file(make_machine_file(bad))

    def test_ramp_at_offset_missing_until_offset_rejected(self, machine_dict, make_machine_file):
        bad = self._machine_with_event(machine_dict, {"shape": "ramp", "at_offset": -60, "to": 1.0})
        with pytest.raises(ValueError, match="missing keys"):
            SimulationEngine.from_file(make_machine_file(bad))

    def test_non_numeric_at_offset_rejected(self, machine_dict, make_machine_file):
        bad = self._machine_with_event(
            machine_dict, {"shape": "step", "at_offset": "-60", "to": 1.0}
        )
        with pytest.raises(ValueError, match="'at_offset' must be a number"):
            SimulationEngine.from_file(make_machine_file(bad))


class TestAtTimeEventValidation:
    """at_time (daily time-of-day) variants are validated at load time."""

    def _machine_with_event(self, machine_dict, event):
        machine_dict["scenarios"]["nominal"]["archiver"] = [{"channel": "T:VAC", "events": [event]}]
        return machine_dict

    @pytest.mark.parametrize(
        ("event", "match"),
        [
            (
                {"shape": "spike", "at": 0.5, "at_time": "14:32:08", "amplitude": 1, "width": 1},
                "exactly one of",
            ),
            (
                {"shape": "ramp", "at_time": "14:32:08", "to": 1, "until": 0.9},
                "'ramp'.*'at_time'",
            ),
            (
                {"shape": "spike", "at_time": "14:99:00", "amplitude": 1, "width": 1},
                "'at_time'.*'14:99:00'",
            ),
            (
                {"shape": "spike", "at_time": "noon", "amplitude": 1, "width": 1},
                "'at_time'.*'noon'",
            ),
            (
                {"shape": "spike", "at_time": 1432, "amplitude": 1, "width": 1},
                "'at_time'.*1432",
            ),
            (
                {"shape": "spike", "at_time": "14:32:08+02:00", "amplitude": 1, "width": 1},
                "timezone",
            ),
        ],
    )
    def test_at_time_validation_errors(self, machine_dict, make_machine_file, event, match):
        bad = self._machine_with_event(machine_dict, event)
        with pytest.raises(ValueError, match=match):
            SimulationEngine.from_file(make_machine_file(bad))

    def test_valid_at_time_spike_accepted(self, machine_dict, make_machine_file):
        good = self._machine_with_event(
            machine_dict, {"shape": "spike", "at_time": "14:32:08", "amplitude": 1.0, "width": 15}
        )
        engine = SimulationEngine.from_file(make_machine_file(good))
        assert engine.has_channel("T:VAC")


class TestEngineServes:
    """The optional-engine guard shared by the mock connectors."""

    def test_none_engine_never_serves(self):
        assert engine_serves(None, "T:Q1:CUR:SP") is False

    def test_known_channel_served(self, machine_file):
        engine = SimulationEngine.from_file(machine_file)
        assert engine_serves(engine, "T:Q1:CUR:SP") is True

    def test_unknown_channel_not_served(self, machine_file):
        engine = SimulationEngine.from_file(machine_file)
        assert engine_serves(engine, "NOPE:UNKNOWN") is False
