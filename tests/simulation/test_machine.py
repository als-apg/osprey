"""Direct unit tests for the machine-description parser entry point.

The individual validation rules are exercised end-to-end through
``SimulationEngine`` construction in ``test_engine.py``; this file locks the
contract of the extracted ``parse_machine`` entry point and the ``ParsedMachine``
container it returns.
"""

from pathlib import Path

import pytest

from osprey.simulation.machine import (
    DEFAULT_SCENARIO,
    ParsedMachine,
    Scenario,
    SimChannel,
    _require_event_number,
    _validate_at_time,
    _validate_position_keys,
    parse_machine,
)

_PATH = Path("machine.json")


def _machine(**overrides):
    base = {
        "name": "TestMachine",
        "description": "fixture",
        "channels": {
            "PV:A": {"value": 10.0, "units": "mA"},
            "PV:B": {"expr": "ch('PV:A') * 2"},
        },
        "scenarios": {
            "fault": {"description": "a fault", "overrides": {"PV:A": 1.0}},
        },
    }
    base.update(overrides)
    return base


class TestParseMachineHappyPath:
    def test_returns_parsed_machine(self):
        model = parse_machine(_machine(), _PATH)
        assert isinstance(model, ParsedMachine)
        assert model.name == "TestMachine"
        assert model.description == "fixture"
        assert set(model.channels) == {"PV:A", "PV:B"}
        assert isinstance(model.channels["PV:A"], SimChannel)

    def test_expression_refs_are_extracted(self):
        model = parse_machine(_machine(), _PATH)
        assert model.channels["PV:B"].refs == ("PV:A",)

    def test_default_nominal_scenario_injected(self):
        model = parse_machine(_machine(), _PATH)
        assert DEFAULT_SCENARIO in model.scenarios
        assert isinstance(model.scenarios["fault"], Scenario)

    def test_explicit_nominal_not_overwritten(self):
        machine = _machine(scenarios={"nominal": {"description": "custom nominal"}})
        model = parse_machine(machine, _PATH)
        assert model.scenarios["nominal"].description == "custom nominal"

    def test_metadata_defaults_when_absent(self):
        machine = {"channels": {"PV:A": {"value": 1.0}}}
        model = parse_machine(machine, _PATH)
        assert model.name == ""
        assert model.description == ""
        assert set(model.scenarios) == {DEFAULT_SCENARIO}


class TestParseMachineValidation:
    def test_missing_channels_mapping(self):
        with pytest.raises(ValueError, match="must define a 'channels' mapping"):
            parse_machine({"name": "x"}, _PATH)

    def test_non_dict_machine(self):
        with pytest.raises(ValueError, match="must define a 'channels' mapping"):
            parse_machine([], _PATH)

    def test_unknown_reference_propagates(self):
        machine = {"channels": {"PV:B": {"expr": "ch('PV:MISSING')"}}}
        with pytest.raises(ValueError, match="references unknown channel 'PV:MISSING'"):
            parse_machine(machine, _PATH)

    def test_reference_cycle_propagates(self):
        machine = {
            "channels": {
                "PV:A": {"expr": "ch('PV:B')"},
                "PV:B": {"expr": "ch('PV:A')"},
            }
        }
        with pytest.raises(ValueError, match="reference cycle detected"):
            parse_machine(machine, _PATH)

    def test_invalid_event_propagates(self):
        machine = _machine(
            scenarios={
                "fault": {
                    "archiver": [{"channel": "PV:A", "events": [{"shape": "bogus", "at": 0.5}]}]
                }
            }
        )
        with pytest.raises(ValueError, match="event shape must be one of"):
            parse_machine(machine, _PATH)


_PREFIX = "Scenario 'x', channel 'PV:A'"


class TestRequireEventNumber:
    def test_accepts_number(self):
        _require_event_number(_PREFIX, {"at": 0.5}, "at", 0.0, 1.0)  # no raise

    def test_rejects_non_number(self):
        with pytest.raises(ValueError, match="must be a number"):
            _require_event_number(_PREFIX, {"to": "high"}, "to")

    def test_rejects_bool(self):
        # bool is an int subclass but must not pass the numeric check.
        with pytest.raises(ValueError, match="must be a number"):
            _require_event_number(_PREFIX, {"to": True}, "to")

    def test_closed_interval_violation(self):
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            _require_event_number(_PREFIX, {"at": 1.5}, "at", 0.0, 1.0)

    def test_strict_minimum_violation(self):
        with pytest.raises(ValueError, match="must be a number > 0"):
            _require_event_number(_PREFIX, {"width": 0.0}, "width", minimum=0.0)


class TestValidatePositionKeys:
    def test_exactly_one_required_none(self):
        with pytest.raises(ValueError, match="exactly one of"):
            _validate_position_keys(_PREFIX, {"shape": "step", "to": 1.0}, "step")

    def test_exactly_one_required_two(self):
        with pytest.raises(ValueError, match="exactly one of"):
            _validate_position_keys(_PREFIX, {"at": 0.5, "at_offset": 1.0}, "step")

    def test_single_key_ok(self):
        _validate_position_keys(_PREFIX, {"at": 0.5}, "step")  # no raise

    def test_ramp_rejects_at_time(self):
        with pytest.raises(ValueError, match="do not support 'at_time'"):
            _validate_position_keys(_PREFIX, {"at_time": "12:00:00"}, "ramp")

    def test_ramp_rejects_mixed_flavors(self):
        with pytest.raises(ValueError, match="must not mix"):
            _validate_position_keys(_PREFIX, {"at": 0.1, "until_offset": 5.0}, "ramp")

    def test_ramp_requires_until(self):
        with pytest.raises(ValueError, match=r"missing keys \['until'\]"):
            _validate_position_keys(_PREFIX, {"at": 0.1}, "ramp")


class TestValidateAtTime:
    def test_valid(self):
        _validate_at_time(_PREFIX, "08:30:00")  # no raise

    def test_non_string(self):
        with pytest.raises(ValueError, match="must be an 'HH:MM:SS' time string"):
            _validate_at_time(_PREFIX, 830)

    def test_bad_format(self):
        with pytest.raises(ValueError, match="must be a valid 'HH:MM:SS' time of day"):
            _validate_at_time(_PREFIX, "25:99:99")

    def test_timezone_offset_rejected(self):
        with pytest.raises(ValueError, match="must not carry a"):
            _validate_at_time(_PREFIX, "08:30:00+02:00")
