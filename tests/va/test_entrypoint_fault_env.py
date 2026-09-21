"""Tests for entrypoint.py's physics-fault env-var parsing.

Exercises the parse helpers directly (not `main()`, which also needs a real
`machine.json` and softioc) -- mirrors `VA_STUCK_SETPOINTS`'s own untested-at-
main()-level shape. Each helper reads `os.environ` itself (matching
`VA_STUCK_SETPOINTS`'s own style), so tests set env vars via `monkeypatch`.
"""

from __future__ import annotations

import pytest

from osprey.services.virtual_accelerator import entrypoint


class TestParseDeviceFloatMap:
    """Backs VA_CORR_GAIN."""

    def test_absent_env_var_yields_empty_map(self, monkeypatch):
        monkeypatch.delenv("VA_CORR_GAIN", raising=False)
        assert entrypoint._parse_device_float_map("VA_CORR_GAIN", bound=1e-2) == {}

    def test_empty_env_var_yields_empty_map(self, monkeypatch):
        monkeypatch.setenv("VA_CORR_GAIN", "")
        assert entrypoint._parse_device_float_map("VA_CORR_GAIN", bound=1e-2) == {}

    def test_parses_a_single_entry(self, monkeypatch):
        monkeypatch.setenv("VA_CORR_GAIN", "QF07=0.9")
        result = entrypoint._parse_device_float_map("VA_CORR_GAIN", bound=5.0)
        assert result == {"QF07": pytest.approx(0.9)}

    def test_parses_multiple_comma_separated_entries(self, monkeypatch):
        monkeypatch.setenv("VA_CORR_GAIN", "HCM01=0.9,VCM03=-1")
        result = entrypoint._parse_device_float_map("VA_CORR_GAIN", bound=5.0)
        assert result == {"HCM01": pytest.approx(0.9), "VCM03": pytest.approx(-1.0)}

    def test_tolerates_incidental_whitespace(self, monkeypatch):
        monkeypatch.setenv("VA_CORR_GAIN", " HCM01 = 0.9 , VCM03=-1 ")
        result = entrypoint._parse_device_float_map("VA_CORR_GAIN", bound=5.0)
        assert result == {"HCM01": pytest.approx(0.9), "VCM03": pytest.approx(-1.0)}

    def test_magnitude_within_bound_is_accepted(self, monkeypatch):
        monkeypatch.setenv("VA_CORR_GAIN", "HCM01=5.0")
        result = entrypoint._parse_device_float_map("VA_CORR_GAIN", bound=5.0)
        assert result == {"HCM01": pytest.approx(5.0)}

    def test_magnitude_beyond_bound_is_rejected(self, monkeypatch):
        monkeypatch.setenv("VA_CORR_GAIN", "HCM01=10")
        with pytest.raises(SystemExit, match="HCM01"):
            entrypoint._parse_device_float_map("VA_CORR_GAIN", bound=5.0)

    def test_negative_magnitude_beyond_bound_is_rejected(self, monkeypatch):
        monkeypatch.setenv("VA_CORR_GAIN", "HCM01=-10")
        with pytest.raises(SystemExit, match="HCM01"):
            entrypoint._parse_device_float_map("VA_CORR_GAIN", bound=5.0)

    def test_non_numeric_value_is_rejected(self, monkeypatch):
        monkeypatch.setenv("VA_CORR_GAIN", "HCM01=not-a-number")
        with pytest.raises(SystemExit, match="non-numeric"):
            entrypoint._parse_device_float_map("VA_CORR_GAIN", bound=5.0)

    def test_missing_equals_sign_is_rejected(self, monkeypatch):
        monkeypatch.setenv("VA_CORR_GAIN", "HCM01")
        with pytest.raises(SystemExit, match="HCM01"):
            entrypoint._parse_device_float_map("VA_CORR_GAIN", bound=5.0)


class TestParseBpmErrors:
    def test_absent_env_var_yields_empty_map(self, monkeypatch):
        monkeypatch.delenv("VA_BPM_ERRORS", raising=False)
        assert entrypoint._parse_bpm_errors() == {}

    def test_parses_a_single_device_single_field(self, monkeypatch):
        monkeypatch.setenv("VA_BPM_ERRORS", "BPM01:offset_x=50e-6")
        result = entrypoint._parse_bpm_errors()
        assert result == {"BPM01": {"offset_x": pytest.approx(50e-6)}}

    def test_parses_a_single_device_multiple_fields(self, monkeypatch):
        monkeypatch.setenv("VA_BPM_ERRORS", "BPM01:offset_x=50e-6,gain_y=1.05")
        result = entrypoint._parse_bpm_errors()
        assert result == {
            "BPM01": {"offset_x": pytest.approx(50e-6), "gain_y": pytest.approx(1.05)}
        }

    def test_parses_multiple_semicolon_separated_devices(self, monkeypatch):
        monkeypatch.setenv("VA_BPM_ERRORS", "BPM01:offset_x=50e-6;BPM07:polarity_x=-1")
        result = entrypoint._parse_bpm_errors()
        assert result == {
            "BPM01": {"offset_x": pytest.approx(50e-6)},
            "BPM07": {"polarity_x": pytest.approx(-1.0)},
        }

    def test_polarity_accepts_plus_one(self, monkeypatch):
        monkeypatch.setenv("VA_BPM_ERRORS", "BPM07:polarity_x=1")
        result = entrypoint._parse_bpm_errors()
        assert result == {"BPM07": {"polarity_x": pytest.approx(1.0)}}

    def test_polarity_rejects_a_non_unit_value(self, monkeypatch):
        monkeypatch.setenv("VA_BPM_ERRORS", "BPM07:polarity_x=0.5")
        with pytest.raises(SystemExit, match="polarity_x"):
            entrypoint._parse_bpm_errors()

    def test_an_offset_of_any_size_is_accepted(self, monkeypatch):
        # A displacement is not a property of a monitor, it is the magnitude
        # the simulator was asked to seed, so no size makes it absurd: the
        # number is carried through in the unit the monitor publishes.
        monkeypatch.setenv("VA_BPM_ERRORS", "BPM01:offset_x=5")
        assert entrypoint._parse_bpm_errors() == {"BPM01": {"offset_x": pytest.approx(5.0)}}

    def test_a_noise_amplitude_of_any_size_is_accepted(self, monkeypatch):
        monkeypatch.setenv("VA_BPM_ERRORS", "BPM01:noise_y=5")
        assert entrypoint._parse_bpm_errors() == {"BPM01": {"noise_y": pytest.approx(5.0)}}

    def test_gain_below_bound_is_rejected(self, monkeypatch):
        monkeypatch.setenv("VA_BPM_ERRORS", "BPM01:gain_x=0.001")
        with pytest.raises(SystemExit, match="gain_x"):
            entrypoint._parse_bpm_errors()

    def test_negative_noise_is_rejected(self, monkeypatch):
        # A noise amplitude is a standard deviation, so a negative one names no
        # distribution at all -- a well-formedness refusal, not a size limit.
        monkeypatch.setenv("VA_BPM_ERRORS", "BPM01:noise_x=-1e-6")
        with pytest.raises(SystemExit, match="noise_x"):
            entrypoint._parse_bpm_errors()

    @pytest.mark.parametrize("spelling", ["nan", "inf", "-inf", "NaN", "Infinity"])
    @pytest.mark.parametrize("field", ["offset_x", "noise_y"])
    def test_a_non_finite_magnitude_is_rejected(self, monkeypatch, field, spelling):
        # Also well-formedness, not a size limit: a seeded magnitude may be as
        # large as it likes, but nan and inf name no magnitude at all. Refused
        # by name here because nothing downstream would refuse them -- a
        # non-finite standard deviation draws as nan rather than raising, and
        # the monitor would publish nan on every read while the boot log
        # reports the seed as applied.
        monkeypatch.setenv("VA_BPM_ERRORS", f"BPM01:{field}={spelling}")
        with pytest.raises(SystemExit, match="not a finite number"):
            entrypoint._parse_bpm_errors()

    def test_a_non_finite_instrument_property_is_rejected(self, monkeypatch):
        # One rule for every field, so a bounded one is refused the same way.
        monkeypatch.setenv("VA_BPM_ERRORS", "BPM01:gain_x=nan,polarity_y=inf")
        with pytest.raises(SystemExit, match="not a finite number"):
            entrypoint._parse_bpm_errors()

    def test_a_very_large_finite_magnitude_is_not_confused_with_infinity(self, monkeypatch):
        # The refusal is about being a number, so the largest float a seed can
        # carry still parses.
        monkeypatch.setenv("VA_BPM_ERRORS", "BPM01:offset_x=1e308")
        assert entrypoint._parse_bpm_errors() == {"BPM01": {"offset_x": pytest.approx(1e308)}}

    def test_unknown_field_is_rejected(self, monkeypatch):
        monkeypatch.setenv("VA_BPM_ERRORS", "BPM01:not_a_field=1.0")
        with pytest.raises(SystemExit, match="not_a_field"):
            entrypoint._parse_bpm_errors()

    def test_missing_colon_is_rejected(self, monkeypatch):
        monkeypatch.setenv("VA_BPM_ERRORS", "BPM01offset_x=50e-6")
        with pytest.raises(SystemExit, match="BPM01offset_x=50e-6"):
            entrypoint._parse_bpm_errors()

    def test_non_numeric_field_value_is_rejected(self, monkeypatch):
        monkeypatch.setenv("VA_BPM_ERRORS", "BPM01:offset_x=not-a-number")
        with pytest.raises(SystemExit, match="non-numeric"):
            entrypoint._parse_bpm_errors()

    def test_a_seeded_magnitude_reaches_the_model_exactly_as_written(self, monkeypatch):
        # Never clamped and never converted: the number the env var carries is
        # the number the error model receives, whatever its size. A seeded
        # offset is in the unit the monitor publishes (millimetres wherever the
        # exported monitor_inverse carries the m->mm gain), and the parser
        # neither knows that unit nor needs to.
        monkeypatch.setenv("VA_BPM_ERRORS", "BPM01:offset_x=5e-3,offset_y=-40,noise_x=1e3")
        assert entrypoint._parse_bpm_errors() == {
            "BPM01": {
                "offset_x": pytest.approx(5e-3),
                "offset_y": pytest.approx(-40.0),
                "noise_x": pytest.approx(1e3),
            }
        }

    def test_an_instrument_property_is_still_bounded(self, monkeypatch):
        # What a monitor can BE stays bounded; this is the refusal path for
        # those fields.
        monkeypatch.setenv("VA_BPM_ERRORS", "BPM01:roll=1.0")
        with pytest.raises(SystemExit, match="outside bound"):
            entrypoint._parse_bpm_errors()

    def test_a_device_spelled_as_an_address_is_one_token(self, monkeypatch):
        # The grammar splits `DEV:field=value` at the LAST colon, so a device
        # spelled as the address its reading is published on -- colon
        # separated at every level -- survives as one token and reaches the
        # resolver, which accepts that spelling as readily as the element's.
        # A field list carries no colon, so nothing about the plain spelling
        # or the missing-colon refusal changes with it.
        monkeypatch.setenv("VA_BPM_ERRORS", "SR:DIAG:BPM:12:POSITION:X:offset_x=50e-6")
        assert entrypoint._parse_bpm_errors() == {
            "SR:DIAG:BPM:12:POSITION:X": {"offset_x": pytest.approx(50e-6)}
        }

    def test_an_address_spelled_device_still_carries_a_field_list(self, monkeypatch):
        monkeypatch.setenv("VA_BPM_ERRORS", "SR:DIAG:BPM:12:X:offset_x=50e-6,polarity_y=-1")
        assert entrypoint._parse_bpm_errors() == {
            "SR:DIAG:BPM:12:X": {
                "offset_x": pytest.approx(50e-6),
                "polarity_y": pytest.approx(-1.0),
            }
        }


class TestTheFieldRegistryIsTheOneAuthority:
    """The two sides of a rendered field list, and the bounds table beside it.

    A scenario is rendered into ``VA_BPM_ERRORS`` on the host by
    ``simulation.apply``, which cannot import this module -- it pulls in the
    whole serving stack -- so it keeps its own copy of the field list. A copy
    that drifts is a field the render emits and the container refuses, at a
    deploy nobody is watching.
    """

    def test_the_render_emits_the_fields_the_container_accepts(self) -> None:
        from osprey.simulation import apply

        assert apply._BPM_ERROR_FIELD_ORDER == entrypoint._BPM_ERROR_FIELDS

    def test_every_bounded_field_is_a_field_the_parser_knows(self) -> None:
        # A bound on a field the registry omits would be unreachable: the
        # unknown-field refusal comes first.
        assert set(entrypoint._BPM_ERROR_FIELD_BOUNDS) <= set(entrypoint._BPM_ERROR_FIELDS)
