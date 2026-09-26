"""Direct unit tests for the machine-description parser entry point.

The individual validation rules are exercised end-to-end through
``SimulationEngine`` construction in ``test_engine.py``; this file locks the
contract of the extracted ``parse_machine`` entry point and the ``ParsedMachine``
container it returns.
"""

import dataclasses
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
from tests.simulation.conftest import TEMPLATE_SIM

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
        with pytest.raises(dataclasses.FrozenInstanceError):
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
        with pytest.raises(dataclasses.FrozenInstanceError):
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
        # The machine has no "HCM01" channel, and that is fine: device ids are
        # lattice ids, not EPICS channel names -- unlike `overrides`, they must
        # never be validated against `channels`.
        machine = _machine(scenarios={"fault": {"physics": {"corrector_gain": {"HCM01": 1.15}}}})
        physics = parse_machine(machine, _PATH).scenarios["fault"].physics
        assert physics.corrector_gain == {"HCM01": 1.15}

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


class TestSeededDiscoveryScenarioBundles:
    """The shipped bundles parse under the physics schema.

    Loads the real ``control_assistant`` machine.json + scenarios/ tree (not the
    inline fixture) so a malformed bundle is caught here, not only downstream in
    the render step or the agentic-discovery e2e.
    """

    @staticmethod
    def _load() -> ParsedMachine:
        machine_path = TEMPLATE_SIM / "machine.json"
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

    @pytest.mark.parametrize(
        ("name", "bpm_errors", "corrector_gain"),
        [
            ("bpm-polarity", {"BPM17": BpmErrorSpec(polarity=-1)}, {}),
            # Two faults on disjoint devices: a BPM 17 polarity flip plus a
            # bounded HCM01 gain deficit.
            ("orm-dual-fault", {"BPM17": BpmErrorSpec(polarity=-1)}, {"HCM01": 0.5}),
            # A bundle with no ``physics`` block still parses, with ``None``.
            ("rf-thermal", None, None),
        ],
        ids=["bpm-polarity", "orm-dual-fault", "rf-thermal-no-physics"],
    )
    def test_physics_block_parses(self, name, bpm_errors, corrector_gain):
        physics = self._load().scenarios[name].physics
        if bpm_errors is None:
            assert physics is None
            return
        assert physics is not None
        assert physics.bpm_errors == bpm_errors
        assert physics.corrector_gain == corrector_gain


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


def _scenario(spec):
    """A machine whose one scenario, ``s``, is *spec*."""
    return _machine(scenarios={"s": spec})


def _bpm(errors):
    return _scenario({"physics": {"bpm_errors": errors}})


def _event(event):
    return _scenario({"archiver": [{"channel": "PV:A", "events": [event]}]})


class TestParseMachineRejectsMalformedSpecs:
    """Every malformed shape the in-memory parser refuses, with the reason it gives."""

    @pytest.mark.parametrize(
        ("machine", "message"),
        [
            pytest.param(
                _channels({"PV:A": 5.0}),
                "Channel 'PV:A': entry must be a mapping, got float",
                id="channel-not-a-mapping",
            ),
            pytest.param(
                _channels({"PV:A": {"value": [1, 2]}}),
                "Channel 'PV:A': 'value' must be a number or string",
                id="channel-value-a-list",
            ),
            pytest.param(
                _channels({"PV:A": {"value": True}}),
                "Channel 'PV:A': 'value' must be a number or string",
                id="channel-value-a-bool",
            ),
            pytest.param(
                _channels({"PV:A": {"expr": 3}}),
                "Channel 'PV:A': 'expr' must be a string",
                id="channel-expr-not-a-string",
            ),
            pytest.param(
                _machine(scenarios=["fault"]),
                "'scenarios' must be a mapping of scenario name to definition",
                id="scenarios-not-a-mapping",
            ),
            pytest.param(
                _scenario("a fault"),
                "Scenario 's': definition must be a mapping",
                id="scenario-not-a-mapping",
            ),
            pytest.param(
                _scenario({"overrides": {"PV:A": None}}),
                "Scenario 's': override for 'PV:A' must be a number or string",
                id="override-none",
            ),
            pytest.param(
                _scenario({"overrides": {"PV:A": False}}),
                "Scenario 's': override for 'PV:A' must be a number or string",
                id="override-bool",
            ),
            pytest.param(
                _scenario({"archiver": ["PV:A"]}),
                "Scenario 's': archiver entries must be mappings",
                id="archiver-entry-not-a-mapping",
            ),
            pytest.param(
                _scenario({"physics": {"bpm_errors": ["BPM01"]}}),
                "Scenario 's' physics: 'bpm_errors' must be a mapping of device id to an error spec",
                id="bpm-errors-not-a-mapping",
            ),
            pytest.param(
                _bpm({"": {"polarity": -1}}),
                "Scenario 's' physics: 'bpm_errors' keys must be non-empty device id strings",
                id="bpm-errors-empty-device-id",
            ),
            pytest.param(
                _bpm({"BPM01": {"offset": "1e-4"}}),
                "Scenario 's' physics bpm_errors['BPM01']: 'offset' must be a number, got '1e-4'",
                id="bpm-error-offset-a-string",
            ),
            pytest.param(
                _bpm({"BPM01": {"gain": True}}),
                "Scenario 's' physics bpm_errors['BPM01']: 'gain' must be a number, got True",
                id="bpm-error-gain-a-bool",
            ),
            pytest.param(
                _event("step at 0.5"),
                "Scenario 's', channel 'PV:A': event must be a mapping",
                id="event-not-a-mapping",
            ),
            pytest.param(
                _event({"shape": "step"}),
                "Scenario 's', channel 'PV:A': 'step' event missing keys ['to']",
                id="event-missing-keys",
            ),
        ],
    )
    def test_rejects_with_the_specific_reason(self, machine, message):
        with pytest.raises(ValueError) as caught:
            parse_machine(machine, _PATH)

        assert message in str(caught.value)

    def test_a_channel_listed_after_the_one_it_references_parses(self):
        """The reference walk reaches PV:A through PV:B first, then skips it."""
        machine = _channels({"PV:B": {"expr": "ch('PV:A') + 1"}, "PV:A": {"value": 1.0}})

        channels = parse_machine(machine, _PATH).channels

        assert channels["PV:B"].refs == ("PV:A",)
        assert channels["PV:A"].value == 1.0


_LOG_ENTRY = {
    "entry_id": "E-1",
    "when": {"days_ago": 1, "time": "08:30:00"},
    "author": "operator",
    "title": "Beam lost",
    "text": "RF trip at 08:29.",
}


def _entry(**changes):
    entry = {**_LOG_ENTRY, **changes}
    return {key: value for key, value in entry.items() if value is not _DROP}


_DROP = object()


def _write_bundle(root: Path, *, scenario=None, logbook=None) -> Path:
    """A machine file beside a ``scenarios/fault`` bundle; returns the machine path.

    ``scenario``/``logbook`` are written verbatim when a ``str`` (to plant bad
    JSON), as JSON otherwise; a ``scenario`` of ``None`` leaves scenario.json out.
    """
    bundle = root / "scenarios" / "fault"
    bundle.mkdir(parents=True)
    for name, content in (("scenario.json", scenario), ("logbook.json", logbook)):
        if content is not None:
            text = content if isinstance(content, str) else json.dumps(content)
            (bundle / name).write_text(text)
    machine_path = root / "machine.json"
    machine_path.write_text(json.dumps(_machine(scenarios={})))
    return machine_path


def _load_bundle(machine_path: Path) -> ParsedMachine:
    return parse_machine(json.loads(machine_path.read_text()), machine_path)


class TestScenarioBundleValidation:
    """``scenarios/<name>/`` bundles: the files themselves, then each logbook entry."""

    def test_a_valid_bundle_carries_its_logbook(self, tmp_path):
        machine_path = _write_bundle(
            tmp_path, scenario={"description": "rf trip"}, logbook=[_entry(tags=["rf"])]
        )

        scenario = _load_bundle(machine_path).scenarios["fault"]

        assert scenario.description == "rf trip"
        [entry] = scenario.logbook
        assert (entry.entry_id, entry.author, entry.tags) == ("E-1", "operator", ("rf",))
        assert (entry.when.days_ago, entry.when.time.isoformat()) == (1, "08:30:00")

    @pytest.mark.parametrize(
        ("scenario", "logbook", "message"),
        [
            pytest.param(
                None,
                None,
                "Scenario bundle 'fault' is missing scenario.json",
                id="no-scenario-json",
            ),
            pytest.param(
                "{not json",
                None,
                "Scenario bundle 'fault': invalid scenario.json",
                id="scenario-bad-json",
            ),
            pytest.param(
                {}, "[{", "Scenario bundle 'fault': invalid logbook.json", id="logbook-bad-json"
            ),
            pytest.param(
                {},
                {"entries": []},
                "Scenario bundle 'fault': logbook.json must be a JSON array",
                id="logbook-not-an-array",
            ),
        ],
    )
    def test_a_malformed_bundle_file_is_refused(self, tmp_path, scenario, logbook, message):
        machine_path = _write_bundle(tmp_path, scenario=scenario, logbook=logbook)

        with pytest.raises(ValueError) as caught:
            _load_bundle(machine_path)

        assert message in str(caught.value)

    @pytest.mark.parametrize(
        ("entry", "message"),
        [
            pytest.param("E-1", "logbook: each entry must be a mapping", id="entry-not-a-mapping"),
            pytest.param(
                _entry(entry_id=""),
                "logbook: 'entry_id' must be a non-empty string, got ''",
                id="entry-id-empty",
            ),
            pytest.param(
                _entry(entry_id=_DROP),
                "logbook: 'entry_id' must be a non-empty string, got None",
                id="entry-id-missing",
            ),
            pytest.param(
                _entry(when="yesterday"),
                "entry 'E-1': 'when' must be a mapping with 'days_ago' and 'time'",
                id="when-not-a-mapping",
            ),
            pytest.param(
                _entry(when={"days_ago": -1, "time": "08:30:00"}),
                "entry 'E-1': 'days_ago' must be a non-negative integer, got -1",
                id="days-ago-negative",
            ),
            pytest.param(
                _entry(when={"days_ago": True, "time": "08:30:00"}),
                "entry 'E-1': 'days_ago' must be a non-negative integer, got True",
                id="days-ago-bool",
            ),
            pytest.param(
                _entry(author=7),
                "entry 'E-1': 'author' must be a string, got 7",
                id="author-not-a-string",
            ),
            pytest.param(
                _entry(text=_DROP),
                "entry 'E-1': 'text' must be a string, got None",
                id="text-missing",
            ),
            pytest.param(
                _entry(tags="rf"),
                "entry 'E-1': 'tags' must be a list of strings, got 'rf'",
                id="tags-a-string",
            ),
            pytest.param(
                _entry(categories=["ops", 3]),
                "entry 'E-1': 'categories' must be a list of strings, got ['ops', 3]",
                id="categories-mixed",
            ),
            pytest.param(
                _entry(loto_tag=42),
                "entry 'E-1': 'loto_tag' must be a string or null, got 42",
                id="loto-tag-a-number",
            ),
            pytest.param(
                _entry(extra=["k", "v"]),
                "entry 'E-1': 'extra' must be a mapping, got ['k', 'v']",
                id="extra-not-a-mapping",
            ),
        ],
    )
    def test_a_malformed_logbook_entry_is_refused(self, tmp_path, entry, message):
        machine_path = _write_bundle(tmp_path, scenario={}, logbook=[entry])

        with pytest.raises(ValueError) as caught:
            _load_bundle(machine_path)

        assert "Scenario 'fault' logbook" in str(caught.value)
        assert message in str(caught.value)
