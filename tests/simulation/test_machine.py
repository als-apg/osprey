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
    TextureSpec,
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


def _channels(specs):
    """A minimal machine wrapping the given ``pv -> spec`` channel mapping."""
    return {"channels": specs}


def _one(spec):
    """Parse a single-channel machine and return the parsed ``PV:A``."""
    return parse_machine(_channels({"PV:A": spec}), _PATH).channels["PV:A"]


class TestParseNoiseAbs:
    """``noise_abs``: additive sigma in the channel's declared units."""

    def test_defaults_to_zero_when_absent(self):
        assert _one({"value": 0.0}).noise_abs == 0.0

    def test_round_trips(self):
        assert _one({"value": 0.0, "noise_abs": 0.02}).noise_abs == 0.02

    def test_zero_is_allowed(self):
        assert _one({"value": 0.0, "noise_abs": 0}).noise_abs == 0.0

    def test_int_is_coerced_to_float(self):
        parsed = _one({"value": 0.0, "noise_abs": 3})
        assert parsed.noise_abs == 3.0
        assert isinstance(parsed.noise_abs, float)

    def test_allowed_on_expression_channel(self):
        machine = _channels(
            {"PV:A": {"value": 1.0}, "PV:B": {"expr": "ch('PV:A')", "noise_abs": 1}}
        )
        assert parse_machine(machine, _PATH).channels["PV:B"].noise_abs == 1.0

    def test_rejects_negative(self):
        with pytest.raises(ValueError, match="'noise_abs' must be a non-negative number"):
            _one({"value": 0.0, "noise_abs": -1e-6})

    def test_rejects_non_number(self):
        with pytest.raises(ValueError, match="'noise_abs' must be a non-negative number"):
            _one({"value": 0.0, "noise_abs": "0.02"})

    def test_rejects_bool(self):
        with pytest.raises(ValueError, match="'noise_abs' must be a non-negative number"):
            _one({"value": 0.0, "noise_abs": True})

    def test_composes_with_relative_noise(self):
        parsed = _one({"value": 5.0, "noise": 0.01, "noise_abs": 0.2})
        assert (parsed.noise, parsed.noise_abs) == (0.01, 0.2)


class TestParseTexture:
    """``texture``: the declarative baseline-motion primitive."""

    def test_defaults_to_none_when_absent(self):
        assert _one({"value": 0.0}).texture is None

    def test_round_trips_into_texture_spec(self):
        texture = _one(
            {"value": 0.0, "texture": {"kind": "wander", "amplitude": 0.05, "period_s": 3600}}
        ).texture
        assert texture == TextureSpec(kind="wander", amplitude=0.05, period_s=3600.0)
        assert isinstance(texture.amplitude, float)
        assert isinstance(texture.period_s, float)

    def test_texture_spec_is_frozen(self):
        texture = _one(
            {"value": 0.0, "texture": {"kind": "wander", "amplitude": 1.0, "period_s": 60.0}}
        ).texture
        with pytest.raises(Exception):  # noqa: B017 - FrozenInstanceError
            texture.amplitude = 2.0

    def test_rejects_non_mapping(self):
        with pytest.raises(ValueError, match="'texture' must be a mapping"):
            _one({"value": 0.0, "texture": [1, 2]})

    def test_rejects_missing_kind(self):
        with pytest.raises(ValueError, match=r"'texture' missing keys \['kind'\]"):
            _one({"value": 0.0, "texture": {"amplitude": 1.0, "period_s": 60.0}})

    def test_rejects_missing_amplitude_and_period(self):
        with pytest.raises(ValueError, match=r"'texture' missing keys \['amplitude', 'period_s'\]"):
            _one({"value": 0.0, "texture": {"kind": "wander"}})

    def test_rejects_unknown_kind(self):
        with pytest.raises(ValueError, match=r"'texture' kind must be one of \['wander'\]"):
            _one({"value": 0.0, "texture": {"kind": "drift", "amplitude": 1.0, "period_s": 60.0}})

    def test_rejects_unknown_key_inside_texture(self):
        with pytest.raises(ValueError, match=r"'texture' has unknown keys \['octaves'\]"):
            _one(
                {
                    "value": 0.0,
                    "texture": {
                        "kind": "wander",
                        "amplitude": 1.0,
                        "period_s": 60.0,
                        "octaves": 4,
                    },
                }
            )

    def test_rejects_non_positive_amplitude(self):
        with pytest.raises(ValueError, match="'texture' amplitude must be a number > 0"):
            _one({"value": 0.0, "texture": {"kind": "wander", "amplitude": 0.0, "period_s": 60.0}})

    def test_rejects_non_positive_period(self):
        with pytest.raises(ValueError, match="'texture' period_s must be a number > 0"):
            _one({"value": 0.0, "texture": {"kind": "wander", "amplitude": 1.0, "period_s": -1}})

    def test_rejects_bool_amplitude(self):
        with pytest.raises(ValueError, match="'texture' amplitude must be a number > 0"):
            _one({"value": 0.0, "texture": {"kind": "wander", "amplitude": True, "period_s": 60.0}})

    def test_rejects_non_string_kind(self):
        with pytest.raises(ValueError, match=r"'texture' kind must be one of \['wander'\]"):
            _one({"value": 0.0, "texture": {"kind": 1, "amplitude": 1.0, "period_s": 60.0}})

    def test_allowed_on_expression_channel(self):
        machine = _channels(
            {
                "PV:A": {"value": 1.0},
                "PV:B": {
                    "expr": "ch('PV:A')",
                    "texture": {"kind": "wander", "amplitude": 0.1, "period_s": 120.0},
                },
            }
        )
        assert parse_machine(machine, _PATH).channels["PV:B"].texture is not None


class TestStringChannelRejectsSignalKeys:
    """String channels have no numeric signal model (mirrors min/max rejection)."""

    def test_rejects_noise_abs(self):
        with pytest.raises(
            ValueError, match="'noise_abs' is not supported on string-valued channels"
        ):
            _one({"value": "CW", "noise_abs": 0.1})

    def test_rejects_texture(self):
        with pytest.raises(
            ValueError, match="'texture' is not supported on string-valued channels"
        ):
            _one({"value": "CW", "texture": {"kind": "wander", "amplitude": 1.0, "period_s": 60.0}})

    def test_string_channel_without_signal_keys_parses(self):
        assert _one({"value": "CW"}).value == "CW"


class TestUnknownTopLevelKeyLeniency:
    """FR8: adding the two keys does not tighten top-level unknown-key handling."""

    def test_unknown_top_level_key_is_ignored(self):
        assert _one({"value": 1.0, "wibble": "whatever"}).value == 1.0


class TestSimChannelDefaults:
    """FR: existing fixtures must construct unchanged (both new fields default)."""

    def test_constructs_without_new_fields(self):
        channel = SimChannel(
            name="PV:A",
            value=1.0,
            expr=None,
            refs=(),
            units="A",
            noise=0.01,
            description="d",
        )
        assert channel.noise_abs == 0.0
        assert channel.texture is None

    def test_still_frozen(self):
        channel = SimChannel("PV:A", 1.0, None, (), "A", 0.0, "d")
        with pytest.raises(Exception):  # noqa: B017 - FrozenInstanceError
            channel.noise_abs = 1.0


def _warnings(caplog):
    """WARNING-level records captured during a parse."""
    return [r for r in caplog.records if r.levelno >= 30]


class TestDeadConfigParseGuard:
    """FR7: one aggregated warning per parsed file for the dead configuration.

    Dead configuration = ``value == 0.0`` with relative ``noise > 0`` and neither
    additive key, which multiplies to a constant 0.0.
    """

    DEAD = {"value": 0.0, "noise": 0.05}

    def test_warns_exactly_once_for_many_dead_channels(self, caplog):
        machine = _channels({f"PV:{i}": dict(self.DEAD) for i in range(20)})
        with caplog.at_level("WARNING"):
            parse_machine(machine, _PATH)
        assert len(_warnings(caplog)) == 1

    def test_warning_names_the_count(self, caplog):
        machine = _channels({f"PV:{i}": dict(self.DEAD) for i in range(20)})
        with caplog.at_level("WARNING"):
            parse_machine(machine, _PATH)
        assert "20 channel(s)" in caplog.text

    def test_warning_lists_up_to_five_examples_then_elides(self, caplog):
        machine = _channels({f"PV:{i}": dict(self.DEAD) for i in range(20)})
        with caplog.at_level("WARNING"):
            parse_machine(machine, _PATH)
        for i in range(5):
            assert f"PV:{i}" in caplog.text
        assert "PV:5" not in caplog.text
        assert "(+15 more)" in caplog.text

    def test_no_elision_when_five_or_fewer(self, caplog):
        machine = _channels({f"PV:{i}": dict(self.DEAD) for i in range(3)})
        with caplog.at_level("WARNING"):
            parse_machine(machine, _PATH)
        assert "more)" not in caplog.text
        assert "3 channel(s)" in caplog.text

    def test_warning_states_the_remedy_and_the_file(self, caplog):
        with caplog.at_level("WARNING"):
            parse_machine(_channels({"PV:A": dict(self.DEAD)}), _PATH)
        assert "noise_abs" in caplog.text
        assert "texture" in caplog.text
        assert "machine.json" in caplog.text

    def test_never_raises(self):
        parse_machine(_channels({"PV:A": dict(self.DEAD)}), _PATH)  # no raise

    @pytest.mark.parametrize(
        "spec",
        [
            pytest.param({"value": 0.0, "noise": 0.05, "noise_abs": 0.01}, id="has-noise-abs"),
            pytest.param(
                {
                    "value": 0.0,
                    "noise": 0.05,
                    "texture": {"kind": "wander", "amplitude": 1.0, "period_s": 60.0},
                },
                id="has-texture",
            ),
            pytest.param({"value": 0.0, "noise": 0.0}, id="no-relative-noise"),
            pytest.param({"value": 0.0}, id="noise-absent"),
            pytest.param({"value": 100.0, "noise": 0.05}, id="non-zero-baseline"),
            pytest.param({"value": "CW"}, id="string-channel"),
            pytest.param({"value": -0.0, "noise": 0.0}, id="negative-zero-no-noise"),
        ],
    )
    def test_no_warning_for_healthy_configurations(self, spec, caplog):
        with caplog.at_level("WARNING"):
            parse_machine(_channels({"PV:A": spec}), _PATH)
        assert _warnings(caplog) == []

    def test_expression_channel_is_never_flagged(self, caplog):
        machine = _channels(
            {"PV:A": {"value": 1.0}, "PV:B": {"expr": "ch('PV:A') * 0", "noise": 0.05}}
        )
        with caplog.at_level("WARNING"):
            parse_machine(machine, _PATH)
        assert _warnings(caplog) == []

    def test_only_dead_channels_are_counted(self, caplog):
        machine = _channels(
            {
                "PV:DEAD": dict(self.DEAD),
                "PV:OK": {"value": 0.0, "noise": 0.05, "noise_abs": 0.01},
                "PV:LIVE": {"value": 100.0, "noise": 0.05},
            }
        )
        with caplog.at_level("WARNING"):
            parse_machine(machine, _PATH)
        assert "1 channel(s)" in caplog.text
        assert "PV:DEAD" in caplog.text
        assert "PV:OK" not in caplog.text


class TestZeroBaselineFixtures:
    """The shared conftest zero-baseline channels are healthy by construction."""

    def test_fixture_channels_parse_with_signal_keys(self, machine_dict, caplog):
        with caplog.at_level("WARNING"):
            channels = parse_machine(machine_dict, _PATH).channels
        assert channels["T:ZERO:NOISY"].value == 0.0
        assert channels["T:ZERO:NOISY"].noise_abs == 0.02
        assert channels["T:ZERO:NOISY"].texture is None
        assert channels["T:ZERO:TEXTURED"].texture == TextureSpec("wander", 0.05, 3600.0)
        assert _warnings(caplog) == []


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
