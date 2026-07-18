"""Direct unit tests for the machine-description parser entry point.

The individual validation rules are exercised end-to-end through
``SimulationEngine`` construction in ``test_engine.py``; this file locks the
contract of the extracted ``parse_machine`` entry point and the ``ParsedMachine``
container it returns.
"""

import json
from pathlib import Path

import pytest

from osprey.simulation.machine import (
    DEFAULT_SCENARIO,
    BpmErrorSpec,
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


class TestParsePhysicsFault:
    def test_absent_block_is_none(self):
        model = parse_machine(_machine(), _PATH)
        assert model.scenarios["fault"].physics is None

    def test_bpm_errors_defaults_and_overrides(self):
        machine = _machine(
            scenarios={
                "fault": {
                    "physics": {
                        "bpm_errors": {
                            "BPM12": {"polarity": -1},
                            "BPM03": {"offset": 1e-4, "gain": 1.05, "roll": 0.01, "noise": 2e-5},
                        }
                    }
                }
            }
        )
        errors = parse_machine(machine, _PATH).scenarios["fault"].physics.bpm_errors
        assert errors["BPM12"] == BpmErrorSpec(polarity=-1)
        assert errors["BPM03"] == BpmErrorSpec(
            offset=1e-4, gain=1.05, polarity=1, roll=0.01, noise=2e-5
        )

    def test_corrector_gain_parses(self):
        machine = _machine(scenarios={"fault": {"physics": {"corrector_gain": {"HCM01": 1.15}}}})
        physics = parse_machine(machine, _PATH).scenarios["fault"].physics
        assert physics.corrector_gain == {"HCM01": 1.15}

    def test_physics_device_ids_are_not_checked_against_channels(self):
        # Device ids are lattice ids ("HCM01"), not EPICS channel names -- unlike
        # `overrides`, they must never be validated against `channels`.
        machine = _machine(scenarios={"fault": {"physics": {"corrector_gain": {"HCM01": 1.1}}}})
        parse_machine(machine, _PATH)  # no raise

    def test_non_mapping_physics_rejected(self):
        machine = _machine(scenarios={"fault": {"physics": []}})
        with pytest.raises(ValueError, match="'physics' must be a mapping"):
            parse_machine(machine, _PATH)

    def test_non_mapping_corrector_gain_rejected(self):
        machine = _machine(scenarios={"fault": {"physics": {"corrector_gain": [1, 2]}}})
        with pytest.raises(ValueError, match="'corrector_gain' must be a mapping"):
            parse_machine(machine, _PATH)

    def test_corrector_gain_rejects_non_number(self):
        machine = _machine(scenarios={"fault": {"physics": {"corrector_gain": {"HCM01": "x"}}}})
        with pytest.raises(ValueError, match=r"corrector_gain\['HCM01'\] must be a number"):
            parse_machine(machine, _PATH)

    def test_corrector_gain_rejects_bool(self):
        machine = _machine(scenarios={"fault": {"physics": {"corrector_gain": {"HCM01": True}}}})
        with pytest.raises(ValueError, match="must be a number"):
            parse_machine(machine, _PATH)

    def test_bpm_errors_rejects_non_mapping_entry(self):
        machine = _machine(scenarios={"fault": {"physics": {"bpm_errors": {"BPM01": 5}}}})
        with pytest.raises(ValueError, match=r"bpm_errors\['BPM01'\] must be a mapping"):
            parse_machine(machine, _PATH)

    def test_bpm_errors_rejects_bad_polarity(self):
        machine = _machine(
            scenarios={"fault": {"physics": {"bpm_errors": {"BPM01": {"polarity": 2}}}}}
        )
        with pytest.raises(ValueError, match="'polarity' must be 1 or -1"):
            parse_machine(machine, _PATH)

    def test_bpm_errors_rejects_negative_noise(self):
        machine = _machine(
            scenarios={"fault": {"physics": {"bpm_errors": {"BPM01": {"noise": -1.0}}}}}
        )
        with pytest.raises(ValueError, match="'noise' must be >= 0"):
            parse_machine(machine, _PATH)

    def test_empty_device_id_rejected(self):
        machine = _machine(scenarios={"fault": {"physics": {"corrector_gain": {"": 1.1}}}})
        with pytest.raises(ValueError, match="non-empty device id strings"):
            parse_machine(machine, _PATH)


_TEMPLATE_SIM = (
    Path(__file__).parents[2] / "src/osprey/templates/apps/control_assistant/data/simulation"
)


class TestSeededDiscoveryScenarioBundles:
    """The shipped bpm-polarity bundle parses under the physics schema.

    Loads the real ``control_assistant`` machine.json + scenarios/ tree (not the
    inline fixture) so a malformed bundle is caught here, not only downstream in
    the render step or the agentic-discovery e2e.
    """

    @staticmethod
    def _load() -> ParsedMachine:
        machine_path = _TEMPLATE_SIM / "machine.json"
        machine = json.loads(machine_path.read_text())
        return parse_machine(machine, machine_path)

    def test_bpm_polarity_bundle_parses(self):
        scenario = self._load().scenarios["bpm-polarity"]
        assert scenario.physics is not None
        assert scenario.physics.corrector_gain == {}
        assert set(scenario.physics.bpm_errors) == {"BPM17"}
        assert scenario.physics.bpm_errors["BPM17"].polarity == -1
        # No rest symptom: no mock-channel overrides or archiver telemetry --
        # only the real ORM measurement reveals it.
        assert scenario.overrides == {}
        assert scenario.archiver == {}
        assert [e.entry_id for e in scenario.logbook] == ["DEMO-031"]


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
