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
    load_narratives,
    load_scenario_bundles,
    parse_machine,
    read_machine_json,
)
from osprey_connectors.simulation.logbook import PlotSeries, PlotSpec, parse_plot_spec

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

    def test_at_when_is_one_position_key(self):
        when = {"days_ago": 4, "time": "03:05:00"}
        _validate_position_keys(_PREFIX, {"at_when": when}, "spike")  # no raise
        with pytest.raises(ValueError, match="exactly one of"):
            _validate_position_keys(_PREFIX, {"at_when": when, "at_offset": -60}, "spike")

    def test_ramp_rejects_at_when(self):
        with pytest.raises(ValueError, match="do not support 'at_when'"):
            _validate_position_keys(_PREFIX, {"at_when": {"days_ago": 1}}, "ramp")


class TestAtWhenEvents:
    """``at_when`` is validated by the logbook's ``when`` rules, under its own name."""

    @staticmethod
    def _parse(at_when):
        event = {"shape": "spike", "at_when": at_when, "amplitude": 1.0, "width": 60.0}
        scenarios = {"fault": {"archiver": [{"channel": "PV:A", "events": [event]}]}}
        return parse_machine(_machine(scenarios=scenarios), _PATH)

    def test_a_calendar_event_parses(self):
        model = self._parse({"days_ago": 4, "time": "03:05:00"})
        (event,) = model.scenarios["fault"].archiver["PV:A"]
        assert event["at_when"] == {"days_ago": 4, "time": "03:05:00"}

    @pytest.mark.parametrize(
        ("at_when", "message"),
        [
            ("4 days", "'at_when' must be a mapping"),
            ({"days_ago": -1, "time": "03:05:00"}, "'days_ago' must be a non-negative integer"),
            ({"days_ago": 4, "time": "25:00:00"}, "'at_when.time' must be a valid"),
            ({"days_ago": 4, "time": "03:05:00+02:00"}, "'at_when.time' is local time"),
        ],
    )
    def test_a_malformed_calendar_event_is_refused_by_its_own_name(self, at_when, message):
        with pytest.raises(ValueError, match=message):
            self._parse(at_when)


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


_EVENT_SUBJECT = "event key 'at_time'"


class TestValidateAtTime:
    def test_valid(self):
        _validate_at_time(_PREFIX, "08:30:00", subject=_EVENT_SUBJECT)  # no raise

    def test_non_string(self):
        with pytest.raises(ValueError, match="must be an 'HH:MM:SS' time string"):
            _validate_at_time(_PREFIX, 830, subject=_EVENT_SUBJECT)

    def test_bad_format(self):
        with pytest.raises(ValueError, match="must be a valid 'HH:MM:SS' time of day"):
            _validate_at_time(_PREFIX, "25:99:99", subject=_EVENT_SUBJECT)

    def test_timezone_offset_rejected(self):
        with pytest.raises(ValueError, match="must not carry a"):
            _validate_at_time(_PREFIX, "08:30:00+02:00", subject=_EVENT_SUBJECT)

    def test_subject_names_the_key(self):
        with pytest.raises(ValueError) as info:
            _validate_at_time(_PREFIX, 830, subject="'when.time'")
        assert "'when.time' must be an 'HH:MM:SS' time string" in str(info.value)
        assert "at_time" not in str(info.value)


class TestLogbookTimeRefusal:
    @pytest.mark.parametrize("raw", [830, "25:99:99", "08:30:00+02:00"])
    def test_refusal_names_when_time(self, tmp_path, raw):
        bundle = tmp_path / "scenarios" / "fault"
        bundle.mkdir(parents=True)
        (bundle / "scenario.json").write_text("{}")
        entry = {
            "entry_id": "E1",
            "when": {"days_ago": 1, "time": raw},
            "author": "a",
            "title": "t",
            "text": "x",
        }
        (bundle / "logbook.json").write_text(json.dumps([entry]))
        with pytest.raises(ValueError) as info:
            load_scenario_bundles(tmp_path / "scenarios", {})
        assert "Scenario 'fault' logbook entry 'E1': 'when.time'" in str(info.value)
        assert "at_time" not in str(info.value)


_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24


class TestLogbookAttachments:
    """An entry's ``attachments`` names pictures inside its scenario directory."""

    @staticmethod
    def _bundle(tmp_path: Path, attachments, files: dict[str, bytes] | None = None) -> Path:
        bundle = tmp_path / "scenarios" / "fault"
        bundle.mkdir(parents=True)
        (bundle / "scenario.json").write_text("{}")
        for rel, data in (files or {}).items():
            (bundle / rel).parent.mkdir(parents=True, exist_ok=True)
            (bundle / rel).write_bytes(data)
        entry = {
            "entry_id": "E1",
            "when": {"days_ago": 1, "time": "08:00:00"},
            "author": "a",
            "title": "t",
            "text": "x",
            "attachments": attachments,
        }
        (bundle / "logbook.json").write_text(json.dumps([entry]))
        return bundle

    def _load_error(self, tmp_path: Path) -> str:
        with pytest.raises(ValueError) as info:
            load_scenario_bundles(tmp_path / "scenarios", {})
        return str(info.value)

    def test_picture_resolves_to_a_file_in_the_bundle(self, tmp_path):
        bundle = self._bundle(tmp_path, [{"path": "plots/a.png"}], {"plots/a.png": _PNG})
        entry = load_scenario_bundles(tmp_path / "scenarios", {})["fault"].logbook[0]
        assert entry.attachments == ((bundle / "plots" / "a.png").resolve(),)

    def test_entry_without_attachments_carries_none(self, tmp_path):
        bundle = self._bundle(tmp_path, [])
        raw = json.loads((bundle / "logbook.json").read_text())
        del raw[0]["attachments"]
        (bundle / "logbook.json").write_text(json.dumps(raw))
        entry = load_scenario_bundles(tmp_path / "scenarios", {})["fault"].logbook[0]
        assert entry.attachments == ()

    def test_unknown_key_is_refused(self, tmp_path):
        self._bundle(tmp_path, [{"path": "a.png", "caption": "orbit"}], {"a.png": _PNG})
        message = self._load_error(tmp_path)
        assert "Scenario 'fault' logbook entry 'E1'" in message
        assert "unknown keys ['caption']" in message

    def test_missing_file_names_the_path(self, tmp_path):
        self._bundle(tmp_path, [{"path": "plots/gone.png"}])
        message = self._load_error(tmp_path)
        assert "'plots/gone.png' not found" in message

    def test_non_image_suffix_is_refused(self, tmp_path):
        self._bundle(tmp_path, [{"path": "notes.txt"}], {"notes.txt": b"hello"})
        assert "'notes.txt' is not a picture" in self._load_error(tmp_path)

    def test_image_suffix_without_image_data_is_refused(self, tmp_path):
        self._bundle(tmp_path, [{"path": "fake.png"}], {"fake.png": b"not a png at all"})
        assert "'fake.png' does not hold .png image data" in self._load_error(tmp_path)

    @pytest.mark.parametrize("rel", ["../outside.png", "/etc/outside.png"])
    def test_path_outside_the_bundle_is_refused(self, tmp_path, rel):
        (tmp_path / "scenarios").mkdir()
        (tmp_path / "scenarios" / "outside.png").write_bytes(_PNG)
        self._bundle(tmp_path, [{"path": rel}])
        assert "must be relative to the scenario directory" in self._load_error(tmp_path)

    @pytest.mark.parametrize("raw", ["a.png", {"path": "a.png"}])
    def test_attachments_must_be_a_list_of_mappings(self, tmp_path, raw):
        self._bundle(tmp_path, raw if isinstance(raw, dict) else [raw], {"a.png": _PNG})
        message = self._load_error(tmp_path)
        assert "'attachments' must be a list" in message or "must be a mapping" in message

    def test_plot_spec_parses_into_the_entrys_attachments_in_order(self, tmp_path):
        spec = json.dumps(_plot_spec()).encode()
        bundle = self._bundle(
            tmp_path,
            [{"plot": "plots/orbit.json"}, {"path": "plots/a.png"}],
            {"plots/orbit.json": spec, "plots/a.png": _PNG},
        )
        entry = load_scenario_bundles(tmp_path / "scenarios", {})["fault"].logbook[0]
        drawn, shipped = entry.attachments
        assert shipped == (bundle / "plots" / "a.png").resolve()
        assert drawn == PlotSpec(
            filename="orbit_rms.png",
            title="SR orbit RMS",
            ylabel="µm",
            hours_before=(2.0, 1.0, 0.0),
            series=(PlotSeries("X", (1.0, 2.0, 3.0)), PlotSeries("Y", (3.0, 2.0, 1.0))),
            ylim=(0.0, 20.0),
        )

    def test_narratives_carry_plot_specs_too(self, tmp_path):
        self._bundle(
            tmp_path,
            [{"plot": "orbit.json"}],
            {"orbit.json": json.dumps(_plot_spec()).encode()},
        )
        (entry,) = load_narratives(tmp_path / "scenarios")["fault"]
        assert isinstance(entry.attachments[0], PlotSpec)

    def test_an_item_naming_both_a_path_and_a_plot_is_refused(self, tmp_path):
        self._bundle(
            tmp_path,
            [{"path": "a.png", "plot": "orbit.json"}],
            {"a.png": _PNG, "orbit.json": json.dumps(_plot_spec()).encode()},
        )
        assert "exactly one of 'path' or 'plot'" in self._load_error(tmp_path)

    def test_an_empty_item_is_refused(self, tmp_path):
        self._bundle(tmp_path, [{}])
        assert "exactly one of 'path' or 'plot'" in self._load_error(tmp_path)

    def test_missing_plot_spec_names_the_path(self, tmp_path):
        self._bundle(tmp_path, [{"plot": "plots/gone.json"}])
        assert "plot spec 'plots/gone.json' not found" in self._load_error(tmp_path)

    @pytest.mark.parametrize("rel", ["../outside.json", "/etc/outside.json"])
    def test_plot_spec_outside_the_bundle_is_refused(self, tmp_path, rel):
        (tmp_path / "scenarios").mkdir()
        (tmp_path / "scenarios" / "outside.json").write_text(json.dumps(_plot_spec()))
        self._bundle(tmp_path, [{"plot": rel}])
        assert "must be relative to the scenario directory" in self._load_error(tmp_path)

    def test_plot_spec_must_be_a_json_file(self, tmp_path):
        self._bundle(tmp_path, [{"plot": "orbit.png"}], {"orbit.png": _PNG})
        assert "plot spec 'orbit.png' must be a .json file" in self._load_error(tmp_path)

    def test_plot_spec_that_is_not_json_is_refused(self, tmp_path):
        self._bundle(tmp_path, [{"plot": "orbit.json"}], {"orbit.json": b"{not json"})
        assert "plot spec 'orbit.json' is not valid JSON" in self._load_error(tmp_path)

    def test_an_invalid_plot_spec_is_refused_with_the_entry_named(self, tmp_path):
        spec = _plot_spec(colour="red")
        self._bundle(tmp_path, [{"plot": "orbit.json"}], {"orbit.json": json.dumps(spec).encode()})
        message = self._load_error(tmp_path)
        assert "Scenario 'fault' logbook entry 'E1': plot spec 'orbit.json'" in message
        assert "unknown keys ['colour']" in message

    def test_two_pictures_with_one_name_are_refused(self, tmp_path):
        spec = json.dumps(_plot_spec(filename="a.png")).encode()
        self._bundle(
            tmp_path,
            [{"path": "a.png"}, {"plot": "orbit.json"}],
            {"a.png": _PNG, "orbit.json": spec},
        )
        assert "two attachments are both named 'a.png'" in self._load_error(tmp_path)


def _plot_spec(**overrides):
    raw = {
        "filename": "orbit_rms.png",
        "title": "SR orbit RMS",
        "ylabel": "µm",
        "hours_before": [2, 1, 0],
        "series": [{"label": "X", "values": [1, 2, 3]}, {"label": "Y", "values": [3, 2, 1]}],
        "ylim": [0, 20],
    }
    raw.update(overrides)
    return raw


class TestPlotSpec:
    """A plot spec's schema is closed and every rule is a load-time refusal."""

    def test_ylim_is_optional(self):
        raw = _plot_spec()
        del raw["ylim"]
        assert parse_plot_spec(raw).ylim is None

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"colour": "red"}, "unknown keys ['colour']"),
            ({"filename": "plots/orbit.png"}, "'filename' must be a .png file name"),
            ({"filename": "orbit.jpg"}, "'filename' must be a .png file name"),
            ({"filename": ".png"}, "'filename' must be a .png file name"),
            ({"title": 3}, "'title' must be a string"),
            ({"ylabel": None}, "'ylabel' must be a string"),
            ({"hours_before": [0]}, "at least two points"),
            ({"hours_before": [1, 2, 0]}, "non-increasing"),
            ({"hours_before": [3, 2, 1]}, "must end at 0"),
            ({"hours_before": [2, "1", 0]}, "'hours_before' must be a list of finite numbers"),
            ({"hours_before": [2, True, 0]}, "'hours_before' must be a list of finite numbers"),
            ({"series": []}, "'series' must be a non-empty list"),
            ({"series": [{"label": "X"}]}, "exactly 'label' and 'values'"),
            ({"series": [{"label": "X", "values": [1, 2, 3], "c": 1}]}, "exactly 'label'"),
            ({"series": [{"label": "", "values": [1, 2, 3]}]}, "'label' must be a non-empty"),
            ({"series": [{"label": "X", "values": [1, 2]}]}, "has 2 values for 3"),
            (
                {"series": [{"label": "X", "values": [1, float("nan"), 3]}]},
                "series 'X' 'values' must be a list of finite numbers",
            ),
            (
                {"series": [{"label": "X", "values": [1, 2, 3]}] * 2},
                "two series are both labelled 'X'",
            ),
            ({"ylim": [20, 0]}, "'ylim' must be [low, high]"),
            ({"ylim": [0, 10, 20]}, "'ylim' must be [low, high]"),
        ],
    )
    def test_a_rule_broken_is_refused(self, overrides, message):
        with pytest.raises(ValueError) as info:
            parse_plot_spec(_plot_spec(**overrides), "spec")
        assert message in str(info.value)
        assert str(info.value).startswith("spec: ")

    @pytest.mark.parametrize("key", ["filename", "title", "ylabel", "hours_before", "series"])
    def test_every_key_but_ylim_is_required(self, key):
        raw = _plot_spec()
        del raw[key]
        with pytest.raises(ValueError, match=rf"missing keys \['{key}'\]"):
            parse_plot_spec(raw)

    def test_a_spec_must_be_an_object(self):
        with pytest.raises(ValueError, match="must be a JSON object"):
            parse_plot_spec([1, 2])


class TestReadMachineJson:
    def test_syntax_error_names_the_file_and_position(self, tmp_path):
        path = tmp_path / "machine.json"
        text = '{"channels": {"A": {"value": 1},}}'
        path.write_text(text)
        # The decoder's column for a trailing comma differs across Python
        # versions; the message must carry whatever position this one reports.
        with pytest.raises(json.JSONDecodeError) as decoded:
            json.loads(text)
        position = f"line {decoded.value.lineno} column {decoded.value.colno}"
        with pytest.raises(ValueError, match=rf"is not valid JSON: .*{position}") as info:
            read_machine_json(path)
        assert str(path) in str(info.value)
        assert isinstance(info.value.__cause__, json.JSONDecodeError)

    def test_valid_file_decodes(self, tmp_path):
        path = tmp_path / "machine.json"
        machine = {"channels": {"A": {"value": 1}}}
        path.write_text(json.dumps(machine))
        assert read_machine_json(path) == machine


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
