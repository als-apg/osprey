"""Tests for entrypoint.py's FR4 physics-fault env-var parsing.

Exercises the parse helpers directly (not `main()`, which also needs a real
`machine.json` and softioc) -- mirrors `VA_STUCK_SETPOINTS`'s own untested-at-
main()-level shape. Each helper reads `os.environ` itself (matching
`VA_STUCK_SETPOINTS`'s own style), so tests set env vars via `monkeypatch`.
"""

from __future__ import annotations

import pytest

from osprey.services.virtual_accelerator import entrypoint


class TestParseDeviceFloatMap:
    """Backs both VA_QUAD_MISALIGN and VA_CORR_GAIN."""

    def test_absent_env_var_yields_empty_map(self, monkeypatch):
        monkeypatch.delenv("VA_QUAD_MISALIGN", raising=False)
        assert entrypoint._parse_device_float_map("VA_QUAD_MISALIGN", bound=1e-2) == {}

    def test_empty_env_var_yields_empty_map(self, monkeypatch):
        monkeypatch.setenv("VA_QUAD_MISALIGN", "")
        assert entrypoint._parse_device_float_map("VA_QUAD_MISALIGN", bound=1e-2) == {}

    def test_parses_a_single_entry(self, monkeypatch):
        monkeypatch.setenv("VA_QUAD_MISALIGN", "QF07=300e-6")
        result = entrypoint._parse_device_float_map("VA_QUAD_MISALIGN", bound=1e-2)
        assert result == {"QF07": pytest.approx(300e-6)}

    def test_parses_multiple_comma_separated_entries(self, monkeypatch):
        monkeypatch.setenv("VA_CORR_GAIN", "HCM01=0.9,VCM03=-1")
        result = entrypoint._parse_device_float_map("VA_CORR_GAIN", bound=5.0)
        assert result == {"HCM01": pytest.approx(0.9), "VCM03": pytest.approx(-1.0)}

    def test_tolerates_incidental_whitespace(self, monkeypatch):
        monkeypatch.setenv("VA_QUAD_MISALIGN", " QF07 = 300e-6 , QD03=-150e-6 ")
        result = entrypoint._parse_device_float_map("VA_QUAD_MISALIGN", bound=1e-2)
        assert result == {"QF07": pytest.approx(300e-6), "QD03": pytest.approx(-150e-6)}

    def test_magnitude_within_bound_is_accepted(self, monkeypatch):
        monkeypatch.setenv("VA_QUAD_MISALIGN", "QF07=1e-2")
        result = entrypoint._parse_device_float_map("VA_QUAD_MISALIGN", bound=1e-2)
        assert result == {"QF07": pytest.approx(1e-2)}

    def test_magnitude_beyond_bound_is_rejected(self, monkeypatch):
        monkeypatch.setenv("VA_QUAD_MISALIGN", "QF07=5e-2")
        with pytest.raises(SystemExit, match="QF07"):
            entrypoint._parse_device_float_map("VA_QUAD_MISALIGN", bound=1e-2)

    def test_negative_magnitude_beyond_bound_is_rejected(self, monkeypatch):
        monkeypatch.setenv("VA_CORR_GAIN", "HCM01=-10")
        with pytest.raises(SystemExit, match="HCM01"):
            entrypoint._parse_device_float_map("VA_CORR_GAIN", bound=5.0)

    def test_non_numeric_value_is_rejected(self, monkeypatch):
        monkeypatch.setenv("VA_QUAD_MISALIGN", "QF07=not-a-number")
        with pytest.raises(SystemExit, match="non-numeric"):
            entrypoint._parse_device_float_map("VA_QUAD_MISALIGN", bound=1e-2)

    def test_missing_equals_sign_is_rejected(self, monkeypatch):
        monkeypatch.setenv("VA_QUAD_MISALIGN", "QF07")
        with pytest.raises(SystemExit, match="QF07"):
            entrypoint._parse_device_float_map("VA_QUAD_MISALIGN", bound=1e-2)


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

    def test_offset_beyond_bound_is_rejected(self, monkeypatch):
        monkeypatch.setenv("VA_BPM_ERRORS", "BPM01:offset_x=5")
        with pytest.raises(SystemExit, match="BPM01"):
            entrypoint._parse_bpm_errors()

    def test_gain_below_bound_is_rejected(self, monkeypatch):
        monkeypatch.setenv("VA_BPM_ERRORS", "BPM01:gain_x=0.001")
        with pytest.raises(SystemExit, match="gain_x"):
            entrypoint._parse_bpm_errors()

    def test_negative_noise_is_rejected(self, monkeypatch):
        monkeypatch.setenv("VA_BPM_ERRORS", "BPM01:noise_x=-1e-6")
        with pytest.raises(SystemExit, match="noise_x"):
            entrypoint._parse_bpm_errors()

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
