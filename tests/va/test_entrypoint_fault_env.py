"""Tests for entrypoint.py's FR4 physics-fault env-var parsing.

Exercises the parse helpers directly (not `main()`, which also needs a real
`machine.json` and softioc) -- mirrors `VA_STUCK_SETPOINTS`'s own untested-at-
main()-level shape. Each helper reads `os.environ` itself (matching
`VA_STUCK_SETPOINTS`'s own style), so tests set env vars via `monkeypatch`.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from osprey.services.virtual_accelerator import entrypoint
from osprey.services.virtual_accelerator.model import fault_bounds

# The fields `_parse_bpm_errors` accepts, written out so a field added to or
# dropped from the bounds table is a deliberate, visible change.
PARSER_FIELDS = frozenset(
    {
        "offset_x",
        "offset_y",
        "gain_x",
        "gain_y",
        "polarity_x",
        "polarity_y",
        "roll",
        "noise_x",
        "noise_y",
    }
)

# Fields bounded by an interval (the polarity fields are a sign, not a range).
_RANGED_FIELDS = sorted(PARSER_FIELDS - {"polarity_x", "polarity_y"})


class TestFaultBounds:
    """The shared bound table the env parser and the model both check."""

    def test_bpm_bounds_table_covers_exactly_the_nine_parser_fields(self):
        assert set(fault_bounds.BPM_ERROR_FIELD_BOUNDS) == PARSER_FIELDS

    def test_polarity_fields_are_bounded_fields_pinned_to_unit_magnitude(self):
        assert fault_bounds.BPM_POLARITY_FIELDS == frozenset({"polarity_x", "polarity_y"})
        for field in fault_bounds.BPM_POLARITY_FIELDS:
            assert fault_bounds.BPM_ERROR_FIELD_BOUNDS[field] == (-1.0, 1.0)

    def test_every_bound_is_an_ordered_interval(self):
        tables = (fault_bounds.BPM_ERROR_FIELD_BOUNDS, fault_bounds.MAGNET_CAL_BOUNDS)
        for table in tables:
            for field, (low, high) in table.items():
                assert low < high, field

    def test_magnet_cal_bounds_are_symmetric_about_identity_offset(self):
        """cal_factor shares the VA_CORR_GAIN bound, so a seed that parses at
        boot is a value the model accepts, and vice versa."""
        assert fault_bounds.MAX_MAGNET_CAL_OFFSET_A == 10.0
        assert fault_bounds.MAGNET_CAL_BOUNDS == {
            "cal_factor": (-fault_bounds.MAX_CORR_GAIN_FACTOR, fault_bounds.MAX_CORR_GAIN_FACTOR),
            "cal_offset": (-10.0, 10.0),
        }

    def test_bound_values_are_unchanged(self):
        """The magnitudes seeds are checked against -- a change here changes
        which fault scenarios parse."""
        assert fault_bounds.MAX_BPM_OFFSET_M == 1e-2
        assert fault_bounds.MIN_BPM_GAIN == 0.1
        assert fault_bounds.MAX_BPM_GAIN == 10.0
        assert fault_bounds.MAX_BPM_ROLL_RAD == 0.1
        assert fault_bounds.MAX_BPM_NOISE_M == 1e-2
        assert fault_bounds.MAX_CORR_GAIN_FACTOR == 5.0

    def test_entrypoint_parses_against_the_shared_table(self):
        """One table, not a copy: the parser and the model cannot drift."""
        assert entrypoint._BPM_ERROR_FIELD_BOUNDS is fault_bounds.BPM_ERROR_FIELD_BOUNDS
        assert entrypoint._BPM_POLARITY_FIELDS is fault_bounds.BPM_POLARITY_FIELDS
        assert entrypoint.MAX_CORR_GAIN_FACTOR == fault_bounds.MAX_CORR_GAIN_FACTOR

    def test_module_imports_nothing_but_future(self):
        """Pure constants: the VA_LATTICE=none boot path imports this module
        and must never reach the lattice, lume_pyat, at or lume."""
        tree = ast.parse(Path(fault_bounds.__file__).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
        assert imported <= {"__future__"}


@pytest.mark.parametrize("field", _RANGED_FIELDS)
def test_parser_accepts_each_bound_and_rejects_beyond_it(field, monkeypatch):
    """Every ranged field is checked against its entry in the shared table,
    inclusive at both ends."""
    low, high = fault_bounds.BPM_ERROR_FIELD_BOUNDS[field]
    width = high - low
    for edge in (low, high):
        monkeypatch.setenv("VA_BPM_ERRORS", f"BPM01:{field}={edge!r}")
        assert entrypoint._parse_bpm_errors() == {"BPM01": {field: edge}}
    for beyond in (low - width, high + width):
        monkeypatch.setenv("VA_BPM_ERRORS", f"BPM01:{field}={beyond!r}")
        with pytest.raises(SystemExit, match=field):
            entrypoint._parse_bpm_errors()


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

    def test_polarity_accepts_exactly_the_two_values_the_bounds_table_carries(self, monkeypatch):
        """The parser reads the polarity pair off the shared table rather
        than carrying its own copy, so a table the model changed cannot
        leave the two ends disagreeing about what a polarity may be."""
        for field in sorted(fault_bounds.BPM_POLARITY_FIELDS):
            for allowed in fault_bounds.BPM_ERROR_FIELD_BOUNDS[field]:
                monkeypatch.setenv("VA_BPM_ERRORS", f"BPM07:{field}={allowed}")
                assert entrypoint._parse_bpm_errors() == {"BPM07": {field: pytest.approx(allowed)}}

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


class TestResolveModelWriteToken:
    """Backs VA_MODEL_WRITE_TOKEN, the secret a model RPC write must present.

    No test here asserts on a token's contents beyond the value it set: the
    parser is the one place the secret is read, and what it must guarantee is
    only that an absent one disarms model writes rather than arming them with
    something an empty string could match.
    """

    def test_absent_env_var_disables_model_writes(self, monkeypatch):
        monkeypatch.delenv("VA_MODEL_WRITE_TOKEN", raising=False)
        assert entrypoint._resolve_model_write_token() is None

    def test_empty_env_var_disables_model_writes(self, monkeypatch):
        """The compose passthrough sends "" when the host var is absent, so an
        empty token is an unset one and must never be a token an empty
        credential matches."""
        monkeypatch.setenv("VA_MODEL_WRITE_TOKEN", "")
        assert entrypoint._resolve_model_write_token() is None

    def test_whitespace_only_env_var_disables_model_writes(self, monkeypatch):
        monkeypatch.setenv("VA_MODEL_WRITE_TOKEN", "   \t ")
        assert entrypoint._resolve_model_write_token() is None

    def test_a_token_is_returned_as_given(self, monkeypatch):
        monkeypatch.setenv("VA_MODEL_WRITE_TOKEN", "s3cret")
        assert entrypoint._resolve_model_write_token() == "s3cret"

    def test_surrounding_whitespace_is_part_of_the_token(self, monkeypatch):
        """A secret is matched byte for byte against what a client presents,
        so the deployment's value is never rewritten on the way through --
        only whether it is blank decides whether writes are armed."""
        monkeypatch.setenv("VA_MODEL_WRITE_TOKEN", " s3cret ")
        assert entrypoint._resolve_model_write_token() == " s3cret "
