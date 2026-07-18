"""Tests for the control-system LimitsValidator.

Covers two contracts:

1. `defaults` block inheritance - a channel inherits values from the top-level
   `defaults` block unless it overrides them, including the safety-critical
   `writable` lockdown and per-channel `verification` config.
2. Fail-closed invariant - a limits violation always raises; there is no policy
   value that turns enforcement off. (`on_violation` was removed as a knob; this
   guards against re-introducing a fail-open path.)
"""

import json
import sys
from pathlib import Path

import pytest

import osprey.connectors.control_system.limits_validator as limits_validator_module
from osprey.connectors.control_system.limits_validator import (
    DEFAULTS_FIELD,
    ChannelLimitsConfig,
    LimitsValidator,
)
from osprey.errors import ChannelLimitsViolationError


def _make_validator(tmp_path, db: dict, policy: dict | None = None) -> LimitsValidator:
    """Load a LimitsValidator from an on-disk JSON limits database."""
    limits_file = tmp_path / "limits.json"
    limits_file.write_text(json.dumps(db))
    limits_db, raw_db = LimitsValidator._load_limits_database(str(limits_file))
    return LimitsValidator(limits_db, policy or {"allow_unlisted_channels": False}, raw_db)


# ---------------------------------------------------------------------------
# `defaults` block inheritance
# ---------------------------------------------------------------------------


def test_defaults_writable_lockdown_is_inherited(tmp_path):
    """A channel that omits `writable` inherits `defaults.writable = false`.

    Safety: a defaults-level read-only lockdown must block writes to channels
    that do not re-declare `writable`, instead of silently defaulting to True.
    """
    validator = _make_validator(
        tmp_path,
        {
            "defaults": {"writable": False},
            "FOO": {"min_value": 0.0, "max_value": 10.0},  # omits `writable`
        },
    )

    with pytest.raises(ChannelLimitsViolationError) as exc:
        validator.validate("FOO", 5.0)

    assert exc.value.violation_type == "READ_ONLY_CHANNEL"


def test_channel_writable_overrides_defaults(tmp_path):
    """A channel may override `defaults.writable = false` with its own `true`."""
    validator = _make_validator(
        tmp_path,
        {
            "defaults": {"writable": False},
            "FOO": {"writable": True, "min_value": 0.0, "max_value": 10.0},
        },
    )

    # Should not raise: channel's explicit writable=True wins over defaults.
    validator.validate("FOO", 5.0)


def test_defaults_min_max_are_inherited(tmp_path):
    """A channel omitting `max_value` inherits the `defaults` bound."""
    validator = _make_validator(
        tmp_path,
        {
            "defaults": {"min_value": 0.0, "max_value": 10.0},
            "FOO": {},  # inherits both bounds
        },
    )

    with pytest.raises(ChannelLimitsViolationError) as exc:
        validator.validate("FOO", 999.0)

    assert exc.value.violation_type == "MAX_EXCEEDED"


def test_defaults_verification_is_inherited(tmp_path):
    """A channel omitting `verification` inherits the `defaults` verification.

    This deliberately makes the `defaults` level ("readback") differ from any
    plausible global fallback so the test proves true inheritance rather than a
    coincidental global default.
    """
    validator = _make_validator(
        tmp_path,
        {
            "defaults": {"verification": {"level": "readback", "tolerance_percent": 0.5}},
            "FOO": {"min_value": 0.0, "max_value": 100.0},  # omits `verification`
        },
    )

    level, tolerance = validator.get_verification_config("FOO", 50.0)

    assert level == "readback"
    assert tolerance == pytest.approx(50.0 * 0.5 / 100.0)


# ---------------------------------------------------------------------------
# Fail-closed invariant (on_violation removed as a behavioral knob)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("on_violation_value", ["skip", "error", "warn", None])
def test_violation_always_raises_regardless_of_policy(tmp_path, on_violation_value):
    """A limits violation always raises; no policy value makes it fail-open.

    `on_violation` was never honoured for control flow and has been removed as a
    config knob. This locks the fail-closed invariant: even if a policy dict
    carries a legacy `on_violation` value, enforcement still blocks.
    """
    validator = _make_validator(
        tmp_path,
        {"FOO": {"min_value": 0.0, "max_value": 10.0}},
        policy={"allow_unlisted_channels": False, "on_violation": on_violation_value},
    )

    with pytest.raises(ChannelLimitsViolationError) as exc:
        validator.validate("FOO", 999.0)  # exceeds max

    assert exc.value.violation_type == "MAX_EXCEEDED"


# ---------------------------------------------------------------------------
# resolve_database_path
# ---------------------------------------------------------------------------


class TestResolveDatabasePath:
    def test_absolute_path_is_returned_unchanged(self, monkeypatch):
        monkeypatch.delenv("CONFIG_FILE", raising=False)
        abs_path = str(Path("/etc/osprey/limits.json"))

        assert LimitsValidator.resolve_database_path(abs_path, "/some/root") == abs_path

    def test_config_file_directory_wins(self, monkeypatch):
        """A relative path resolves against CONFIG_FILE's directory when set."""
        monkeypatch.setenv("CONFIG_FILE", "/app/project/config.yml")

        resolved = LimitsValidator.resolve_database_path("limits.json", "/host/build/path")

        assert resolved == str(Path("/app/project/limits.json"))

    def test_project_root_fallback_when_no_config_file(self, monkeypatch):
        monkeypatch.delenv("CONFIG_FILE", raising=False)

        resolved = LimitsValidator.resolve_database_path("limits.json", "/proj/root")

        assert resolved == str(Path("/proj/root/limits.json"))

    def test_relative_path_unchanged_when_no_bases(self, monkeypatch):
        monkeypatch.delenv("CONFIG_FILE", raising=False)

        assert LimitsValidator.resolve_database_path("limits.json", None) == "limits.json"


# ---------------------------------------------------------------------------
# from_config
# ---------------------------------------------------------------------------


def _patch_config(monkeypatch, values: dict, raise_exc: Exception | None = None):
    """Patch get_config_value with a key->value map (default fallback otherwise)."""

    def fake_get_config_value(key, default=None):
        if raise_exc is not None:
            raise raise_exc
        return values.get(key, default)

    monkeypatch.setattr("osprey.utils.config.get_config_value", fake_get_config_value)


class TestFromConfig:
    def test_returns_none_when_disabled(self, monkeypatch):
        _patch_config(monkeypatch, {"control_system.limits_checking.enabled": False})

        assert LimitsValidator.from_config() is None

    def test_missing_db_path_yields_blocking_failsafe_validator(self, monkeypatch):
        """Enabled but no database path -> an empty validator that blocks all writes."""
        _patch_config(
            monkeypatch,
            {
                "control_system.limits_checking.enabled": True,
                "control_system.limits_checking.database_path": None,
            },
        )

        validator = LimitsValidator.from_config()

        assert isinstance(validator, LimitsValidator)
        # Empty DB is a failsafe: every channel is unlisted and therefore blocked.
        with pytest.raises(ChannelLimitsViolationError) as exc:
            validator.validate("ANY:CHANNEL", 1.0)
        assert exc.value.violation_type == "UNLISTED_CHANNEL"

    def test_loads_absolute_database(self, monkeypatch, tmp_path):
        db_file = tmp_path / "limits.json"
        db_file.write_text(json.dumps({"FOO": {"min_value": 0.0, "max_value": 10.0}}))
        _patch_config(
            monkeypatch,
            {
                "control_system.limits_checking.enabled": True,
                "control_system.limits_checking.database_path": str(db_file),
                "project_root": None,
                "control_system.limits_checking.allow_unlisted_channels": False,
            },
        )

        validator = LimitsValidator.from_config()

        assert isinstance(validator, LimitsValidator)
        assert "FOO" in validator.limits

    def test_resolves_relative_path_against_project_root(self, monkeypatch, tmp_path):
        """A relative database_path is resolved via project_root (debug branch)."""
        monkeypatch.delenv("CONFIG_FILE", raising=False)
        (tmp_path / "limits.json").write_text(json.dumps({"BAR": {"max_value": 5.0}}))
        _patch_config(
            monkeypatch,
            {
                "control_system.limits_checking.enabled": True,
                "control_system.limits_checking.database_path": "limits.json",
                "project_root": str(tmp_path),
                "control_system.limits_checking.allow_unlisted_channels": False,
            },
        )

        validator = LimitsValidator.from_config()

        assert "BAR" in validator.limits

    def test_resolves_relative_path_against_config_file(self, monkeypatch, tmp_path):
        """CONFIG_FILE's directory takes priority for relative paths (debug branch)."""
        (tmp_path / "limits.json").write_text(json.dumps({"BAZ": {"max_value": 5.0}}))
        monkeypatch.setenv("CONFIG_FILE", str(tmp_path / "config.yml"))
        _patch_config(
            monkeypatch,
            {
                "control_system.limits_checking.enabled": True,
                "control_system.limits_checking.database_path": "limits.json",
                "project_root": "/nonexistent/host/path",
                "control_system.limits_checking.allow_unlisted_channels": False,
            },
        )

        validator = LimitsValidator.from_config()

        assert "BAZ" in validator.limits

    def test_returns_none_when_config_unavailable(self, monkeypatch):
        _patch_config(monkeypatch, {}, raise_exc=RuntimeError("no config"))

        assert LimitsValidator.from_config() is None


# ---------------------------------------------------------------------------
# get_limits_config
# ---------------------------------------------------------------------------


class TestGetLimitsConfig:
    def test_known_channel_returns_full_dict(self, tmp_path):
        validator = _make_validator(
            tmp_path,
            {"FOO": {"min_value": 1.0, "max_value": 9.0, "max_step": 2.0, "writable": True}},
        )

        config = validator.get_limits_config("FOO")

        assert config == {
            "channel_address": "FOO",
            "min_value": 1.0,
            "max_value": 9.0,
            "max_step": 2.0,
            "writable": True,
        }

    def test_unknown_channel_returns_none(self, tmp_path):
        validator = _make_validator(tmp_path, {"FOO": {"max_value": 9.0}})

        assert validator.get_limits_config("MISSING") is None


# ---------------------------------------------------------------------------
# get_verification_config (tolerance math + fallbacks)
# ---------------------------------------------------------------------------


class TestGetVerificationConfig:
    def test_no_raw_db_returns_none_none(self):
        validator = LimitsValidator({}, {}, raw_db=None)

        assert validator.get_verification_config("FOO", 10.0) == (None, None)

    def test_channel_absolute_tolerance_takes_priority(self, tmp_path):
        validator = _make_validator(
            tmp_path,
            {
                "FOO": {
                    "max_value": 100.0,
                    "verification": {
                        "level": "readback",
                        "tolerance_absolute": 0.25,
                        "tolerance_percent": 5.0,
                    },
                }
            },
        )

        level, tolerance = validator.get_verification_config("FOO", 50.0)

        assert level == "readback"
        assert tolerance == 0.25  # absolute wins over percent

    def test_default_permille_tolerance_when_none_specified(self, tmp_path):
        validator = _make_validator(
            tmp_path,
            {"FOO": {"max_value": 100.0, "verification": {"level": "readback"}}},
        )

        level, tolerance = validator.get_verification_config("FOO", 200.0)

        assert level == "readback"
        assert tolerance == pytest.approx(200.0 * 0.1 / 100.0)  # 0.1% default

    def test_non_readback_level_has_no_tolerance(self, tmp_path):
        validator = _make_validator(
            tmp_path,
            {"FOO": {"max_value": 100.0, "verification": {"level": "callback"}}},
        )

        assert validator.get_verification_config("FOO", 50.0) == ("callback", None)

    def test_no_verification_config_returns_none_none(self, tmp_path):
        validator = _make_validator(tmp_path, {"FOO": {"max_value": 100.0}})

        assert validator.get_verification_config("FOO", 50.0) == (None, None)

    def test_non_dict_defaults_block_is_ignored(self):
        """A malformed (non-dict) defaults block does not crash lookups."""
        validator = LimitsValidator({}, {}, raw_db={"defaults": "oops", "FOO": {}})

        assert validator.get_verification_config("FOO", 1.0) == (None, None)


# ---------------------------------------------------------------------------
# _validate_channel_config
# ---------------------------------------------------------------------------


def _capture_warnings(monkeypatch) -> list[str]:
    """Record limits_validator warning messages without depending on logging config.

    caplog is unreliable here: get_logger() reconfigures the root logger and can
    drop pytest's capture handler depending on test order, so we patch the
    module logger directly (the pattern used in test_epics_gateway_selection.py).
    """
    messages: list[str] = []
    monkeypatch.setattr(
        limits_validator_module.logger,
        "warning",
        lambda msg, *a, **k: messages.append(str(msg)),
    )
    return messages


class TestValidateChannelConfig:
    def test_unknown_field_warns_but_does_not_raise(self, monkeypatch):
        # Unknown fields are a warning, not an error — the config still loads.
        warnings = _capture_warnings(monkeypatch)

        LimitsValidator._validate_channel_config("FOO", {"bogus_field": 1})

        assert any("unknown fields" in m for m in warnings), warnings

    def test_non_numeric_bound_raises(self):
        with pytest.raises(ValueError, match="must be numeric"):
            LimitsValidator._validate_channel_config("FOO", {"min_value": "low"})

    def test_non_bool_writable_raises(self):
        with pytest.raises(ValueError, match="must be boolean"):
            LimitsValidator._validate_channel_config("FOO", {"writable": "yes"})

    def test_verification_must_be_dict(self):
        with pytest.raises(ValueError, match="must be a dictionary"):
            LimitsValidator._validate_channel_config("FOO", {"verification": "readback"})

    def test_unknown_verification_field_warns(self, monkeypatch):
        warnings = _capture_warnings(monkeypatch)

        LimitsValidator._validate_channel_config(
            "FOO", {"verification": {"level": "callback", "bogus": 1}}
        )

        assert any("verification has unknown fields" in m for m in warnings), warnings

    def test_invalid_verification_level_raises(self):
        with pytest.raises(ValueError, match="verification.level must be"):
            LimitsValidator._validate_channel_config(
                "FOO", {"verification": {"level": "sometimes"}}
            )

    def test_verification_without_level_is_accepted(self):
        # A verification block may omit 'level' (a default is applied elsewhere) — no raise.
        LimitsValidator._validate_channel_config(
            "FOO", {"verification": {"tolerance_absolute": 0.1}}
        )


# ---------------------------------------------------------------------------
# _load_limits_database (error + skip branches)
# ---------------------------------------------------------------------------


class TestLoadDatabase:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ValueError, match="database not found"):
            LimitsValidator._load_limits_database(str(tmp_path / "nope.json"))

    def test_non_dict_root_raises(self, tmp_path):
        f = tmp_path / "limits.json"
        f.write_text(json.dumps(["not", "a", "dict"]))

        with pytest.raises(ValueError, match="must be a JSON object"):
            LimitsValidator._load_limits_database(str(f))

    def test_non_dict_defaults_raises(self, tmp_path):
        f = tmp_path / "limits.json"
        f.write_text(json.dumps({DEFAULTS_FIELD: 5}))

        with pytest.raises(ValueError, match="must be a dictionary"):
            LimitsValidator._load_limits_database(str(f))

    def test_invalid_defaults_config_raises(self, tmp_path):
        f = tmp_path / "limits.json"
        f.write_text(json.dumps({DEFAULTS_FIELD: {"min_value": "not-numeric"}}))

        with pytest.raises(ValueError, match="Invalid 'defaults' configuration"):
            LimitsValidator._load_limits_database(str(f))

    def test_metadata_and_non_dict_channels_are_skipped(self, tmp_path):
        f = tmp_path / "limits.json"
        f.write_text(
            json.dumps(
                {
                    "_comment": "ignored metadata",
                    "BADCHAN": 42,  # non-dict -> skipped
                    "GOOD": {"max_value": 10.0},
                }
            )
        )

        limits_db, _ = LimitsValidator._load_limits_database(str(f))

        assert set(limits_db) == {"GOOD"}

    def test_channel_with_invalid_field_is_skipped(self, tmp_path):
        f = tmp_path / "limits.json"
        f.write_text(json.dumps({"BADFIELD": {"min_value": "x"}, "GOOD": {"max_value": 10.0}}))

        limits_db, _ = LimitsValidator._load_limits_database(str(f))

        assert set(limits_db) == {"GOOD"}

    def test_max_step_channel_loads(self, tmp_path):
        f = tmp_path / "limits.json"
        f.write_text(json.dumps({"STEP": {"max_value": 100.0, "max_step": 5.0}}))

        limits_db, _ = LimitsValidator._load_limits_database(str(f))

        assert limits_db["STEP"].max_step == 5.0

    def test_invalid_json_raises(self, tmp_path):
        f = tmp_path / "limits.json"
        f.write_text("{ this is not valid json ")

        with pytest.raises(ValueError, match="Invalid JSON"):
            LimitsValidator._load_limits_database(str(f))

    def test_unreadable_path_raises_generic(self, tmp_path):
        # A directory exists but cannot be opened as a file -> generic failure branch.
        with pytest.raises(ValueError, match="Failed to load"):
            LimitsValidator._load_limits_database(str(tmp_path))


# ---------------------------------------------------------------------------
# max_step check (Check 4) — the I/O safety path
# ---------------------------------------------------------------------------


def _step_validator(max_step: float = 5.0) -> LimitsValidator:
    """A validator with one channel that has max_step configured (triggers caget)."""
    limits = {
        "FOO": ChannelLimitsConfig(
            channel_address="FOO", min_value=0.0, max_value=100.0, max_step=max_step
        )
    }
    return LimitsValidator(limits, {"allow_unlisted_channels": False}, {})


class TestMaxStepCheck:
    def test_current_none_blocks_write(self, monkeypatch):
        monkeypatch.setattr("epics.caget", lambda *a, **k: None)
        validator = _step_validator()

        with pytest.raises(ChannelLimitsViolationError) as exc:
            validator.validate("FOO", 50.0)

        assert exc.value.violation_type == "STEP_CHECK_FAILED"

    def test_step_exceeded_blocks_with_details(self, monkeypatch):
        monkeypatch.setattr("epics.caget", lambda *a, **k: 10.0)
        validator = _step_validator(max_step=5.0)

        with pytest.raises(ChannelLimitsViolationError) as exc:
            validator.validate("FOO", 100.0)  # step of 90 >> 5

        assert exc.value.violation_type == "MAX_STEP_EXCEEDED"
        assert exc.value.current_value == 10.0
        assert exc.value.max_step == 5.0

    def test_step_within_limit_passes(self, monkeypatch):
        monkeypatch.setattr("epics.caget", lambda *a, **k: 48.0)
        validator = _step_validator(max_step=5.0)

        # Step of 2.0 is within max_step=5.0 -> no raise.
        validator.validate("FOO", 50.0)

    def test_non_numeric_current_skips_step_check(self, monkeypatch):
        monkeypatch.setattr("epics.caget", lambda *a, **k: "not-a-number")
        validator = _step_validator(max_step=1.0)

        # Current value can't be coerced to float -> step check is skipped, write allowed.
        validator.validate("FOO", 50.0)

    def test_missing_pyepics_blocks_write(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "epics", None)  # import epics -> ImportError
        validator = _step_validator()

        with pytest.raises(ChannelLimitsViolationError) as exc:
            validator.validate("FOO", 50.0)

        assert exc.value.violation_type == "STEP_CHECK_FAILED"
        assert "pyepics not available" in exc.value.violation_reason

    def test_caget_error_blocks_write(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("CA timeout")

        monkeypatch.setattr("epics.caget", boom)
        validator = _step_validator()

        with pytest.raises(ChannelLimitsViolationError) as exc:
            validator.validate("FOO", 50.0)

        assert exc.value.violation_type == "STEP_CHECK_FAILED"
        assert "CA timeout" in exc.value.violation_reason


# ---------------------------------------------------------------------------
# validate() — non-step branches (unlisted policy, non-numeric, min bound)
# ---------------------------------------------------------------------------


class TestValidate:
    def test_unlisted_allowed_when_policy_permits(self, tmp_path):
        """allow_unlisted_channels=True lets an unknown channel through."""
        validator = _make_validator(
            tmp_path,
            {"FOO": {"max_value": 10.0}},
            policy={"allow_unlisted_channels": True},
        )

        # No raise: the channel is unlisted but policy allows it.
        validator.validate("NOT:IN:DB", 5.0)

    def test_non_numeric_value_skips_numeric_checks(self, tmp_path):
        """A non-coercible value skips min/max/step checks rather than crashing."""
        validator = _make_validator(tmp_path, {"FOO": {"min_value": 0.0, "max_value": 10.0}})

        # No raise: "on" can't be floated, so numeric bounds are not applied.
        validator.validate("FOO", "on")

    def test_below_minimum_raises(self, tmp_path):
        validator = _make_validator(tmp_path, {"FOO": {"min_value": 0.0, "max_value": 10.0}})

        with pytest.raises(ChannelLimitsViolationError) as exc:
            validator.validate("FOO", -5.0)

        assert exc.value.violation_type == "MIN_EXCEEDED"
