"""Tests for the control-system LimitsValidator.

Covers two contracts:

1. `defaults` block inheritance - a channel inherits values from the top-level
   `defaults` block unless it overrides them, including the safety-critical
   `writable` lockdown and the per-channel `confirm` write policy.
2. Fail-closed invariant - a limits violation always raises; there is no policy
   value that turns enforcement off. (`on_violation` was removed as a knob; this
   guards against re-introducing a fail-open path.)
"""

import json
import sys
from pathlib import Path

import pytest

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


def test_defaults_confirm_is_inherited(tmp_path):
    """A channel omitting `confirm` inherits the `defaults` block's `confirm`.

    The defaults value is deliberately `false`, the opposite of the fleet
    default, so the test proves true inheritance rather than a coincidental
    fallback.
    """
    validator = _make_validator(
        tmp_path,
        {
            "defaults": {"confirm": False},
            "FOO": {"min_value": 0.0, "max_value": 100.0},  # omits `confirm`
        },
    )

    assert validator.resolve_confirm("FOO") is False


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

    def test_loaded_config_directory_beats_env_and_project_root(self, monkeypatch):
        """The config actually loaded is the anchor, ahead of CONFIG_FILE and project_root."""
        monkeypatch.setenv("CONFIG_FILE", "/somewhere/else/config.yml")

        resolved = LimitsValidator.resolve_database_path(
            "data/limits.json", "/repo", config_path="/repo/build/config.yml"
        )

        assert resolved == str(Path("/repo/build/data/limits.json"))

    def test_loaded_config_anchor_ignores_absolute_paths(self, monkeypatch):
        monkeypatch.delenv("CONFIG_FILE", raising=False)
        abs_path = str(Path("/etc/osprey/limits.json"))

        resolved = LimitsValidator.resolve_database_path(
            abs_path, "/repo", config_path="/repo/build/config.yml"
        )

        assert resolved == abs_path


# ---------------------------------------------------------------------------
# from_config
# ---------------------------------------------------------------------------


def _patch_config(
    monkeypatch,
    values: dict,
    raise_exc: Exception | None = None,
    config_path: str | None = None,
):
    """Patch get_config_value with a key->value map (default fallback otherwise).

    ``config_path`` stands in for ``default_config_path()`` — the path of the
    config the singleton actually loaded. ``None`` (the default) models a
    process whose config came from somewhere the singleton can't name, which
    keeps the env-var/project_root fallback branches testable in isolation.
    """

    def fake_get_config_value(key, default=None):
        if raise_exc is not None:
            raise raise_exc
        return values.get(key, default)

    monkeypatch.setattr("osprey.utils.config.get_config_value", fake_get_config_value)
    monkeypatch.setattr("osprey.utils.config.default_config_path", lambda: config_path)


class TestFromConfig:
    def test_returns_none_when_disabled(self, monkeypatch):
        _patch_config(monkeypatch, {"control_system.limits_checking.enabled": False})

        assert LimitsValidator.from_config() is None

    def test_missing_db_path_yields_blocking_failsafe_validator(self, monkeypatch):
        """Enabled but no database path -> an empty validator that blocks all writes.

        The refusal must say the database is unavailable, not that the channel
        is unlisted — an agent reading "not in limits database" narrates a data
        problem when the actual failure is configuration (#636).
        """
        _patch_config(
            monkeypatch,
            {
                "control_system.limits_checking.enabled": True,
                "control_system.limits_checking.database_path": None,
            },
        )

        validator = LimitsValidator.from_config()

        assert isinstance(validator, LimitsValidator)
        with pytest.raises(ChannelLimitsViolationError) as exc:
            validator.validate("ANY:CHANNEL", 1.0)
        assert exc.value.violation_type == "LIMITS_DATABASE_UNAVAILABLE"
        assert "failsafe" in exc.value.violation_reason

    def test_unreadable_db_failsafe_is_distinguishable_from_unlisted_channel(
        self, monkeypatch, tmp_path
    ):
        """A database that fails to load blocks with a load-failure refusal (#636)."""
        db_file = tmp_path / "limits.json"
        db_file.write_text("{not valid json")
        _patch_config(
            monkeypatch,
            {
                "control_system.limits_checking.enabled": True,
                "control_system.limits_checking.database_path": str(db_file),
            },
        )

        validator = LimitsValidator.from_config()

        assert isinstance(validator, LimitsValidator)
        with pytest.raises(ChannelLimitsViolationError) as exc:
            validator.validate("ANY:CHANNEL", 1.0)
        assert exc.value.violation_type == "LIMITS_DATABASE_UNAVAILABLE"
        assert "not in limits database" not in exc.value.violation_reason

    def test_genuinely_unlisted_channel_keeps_the_unlisted_refusal(self, monkeypatch, tmp_path):
        """A loaded database still refuses unlisted channels with the unlisted message."""
        db_file = tmp_path / "limits.json"
        db_file.write_text(json.dumps({"FOO": {"min_value": 0.0, "max_value": 10.0}}))
        _patch_config(
            monkeypatch,
            {
                "control_system.limits_checking.enabled": True,
                "control_system.limits_checking.database_path": str(db_file),
            },
        )

        validator = LimitsValidator.from_config()

        with pytest.raises(ChannelLimitsViolationError) as exc:
            validator.validate("NOT:LISTED", 1.0)
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

    def test_four_zone_hook_resolves_database_beside_loaded_config(self, monkeypatch, tmp_path):
        """Regression (#636): a relative database_path anchors on the config actually loaded.

        The four-zone hook scenario: the render lives at <repo>/build (config +
        data/ side by side), the config's ``project_root`` names <repo>, and the
        hook process has no CONFIG_FILE in its environment. The database must be
        found next to the loaded config — not resolved against project_root,
        where it does not exist and every write would hit the empty-DB failsafe.
        """
        monkeypatch.delenv("CONFIG_FILE", raising=False)
        build = tmp_path / "build"
        (build / "data").mkdir(parents=True)
        (build / "data" / "channel_limits.json").write_text(
            json.dumps({"SR:C1:HCM:SP": {"min_value": -1.0, "max_value": 1.0}})
        )
        _patch_config(
            monkeypatch,
            {
                "control_system.limits_checking.enabled": True,
                "control_system.limits_checking.database_path": "data/channel_limits.json",
                "project_root": str(tmp_path),
                "control_system.limits_checking.allow_unlisted_channels": False,
            },
            config_path=str(build / "config.yml"),
        )

        validator = LimitsValidator.from_config()

        assert "SR:C1:HCM:SP" in validator.limits
        validator.validate("SR:C1:HCM:SP", 0.5)  # listed and in range: allowed


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
# resolve_confirm (channel entry -> defaults -> True)
# ---------------------------------------------------------------------------


class TestResolveConfirm:
    def test_no_raw_db_confirms(self):
        # No policy to read is not a licence to skip the check.
        validator = LimitsValidator({}, {}, raw_db=None)

        assert validator.resolve_confirm("FOO") is True

    def test_channel_entry_wins_over_defaults(self, tmp_path):
        validator = _make_validator(
            tmp_path,
            {
                "defaults": {"confirm": True},
                "FOO": {"max_value": 100.0, "confirm": False},
            },
        )

        assert validator.resolve_confirm("FOO") is False

    def test_channel_entry_can_re_enable_what_defaults_switched_off(self, tmp_path):
        validator = _make_validator(
            tmp_path,
            {
                "defaults": {"confirm": False},
                "FOO": {"max_value": 100.0, "confirm": True},
            },
        )

        assert validator.resolve_confirm("FOO") is True

    def test_defaults_apply_when_the_channel_is_silent(self, tmp_path):
        validator = _make_validator(
            tmp_path,
            {"defaults": {"confirm": False}, "FOO": {"max_value": 100.0}},
        )

        assert validator.resolve_confirm("FOO") is False

    def test_silence_everywhere_confirms(self, tmp_path):
        validator = _make_validator(tmp_path, {"FOO": {"max_value": 100.0}})

        assert validator.resolve_confirm("FOO") is True

    def test_unlisted_channel_confirms(self, tmp_path):
        validator = _make_validator(
            tmp_path, {"defaults": {"confirm": True}, "FOO": {"max_value": 100.0}}
        )

        assert validator.resolve_confirm("MISSING") is True

    def test_unlisted_channel_still_inherits_defaults(self, tmp_path):
        validator = _make_validator(
            tmp_path, {"defaults": {"confirm": False}, "FOO": {"max_value": 100.0}}
        )

        assert validator.resolve_confirm("MISSING") is False

    def test_non_dict_defaults_block_is_ignored(self):
        """A malformed (non-dict) defaults block does not crash the lookup."""
        validator = LimitsValidator({}, {}, raw_db={"defaults": "oops", "FOO": {}})

        assert validator.resolve_confirm("FOO") is True

    def test_non_dict_channel_entry_is_ignored(self):
        validator = LimitsValidator({}, {}, raw_db={"defaults": {"confirm": False}, "FOO": "oops"})

        assert validator.resolve_confirm("FOO") is False


# ---------------------------------------------------------------------------
# _validate_channel_config
# ---------------------------------------------------------------------------


class TestValidateChannelConfig:
    def test_unknown_field_raises_and_names_the_key(self):
        with pytest.raises(ValueError, match="bogus_field"):
            LimitsValidator._validate_channel_config("FOO", {"bogus_field": 1})

    def test_underscore_prefixed_metadata_is_accepted(self):
        # Any '_'-prefixed key is documentation, not a closed set of four.
        LimitsValidator._validate_channel_config(
            "FOO", {"max_value": 1.0, "_units": "mA", "_owner": "APG"}
        )

    def test_retired_verification_block_raises_with_the_migration_message(self):
        with pytest.raises(ValueError) as exc:
            LimitsValidator._validate_channel_config("FOO", {"verification": {"level": "none"}})

        assert "'verification' was replaced by 'confirm: true|false'" in str(exc.value)

    def test_non_numeric_bound_raises(self):
        with pytest.raises(ValueError, match="must be numeric"):
            LimitsValidator._validate_channel_config("FOO", {"min_value": "low"})

    def test_non_bool_writable_raises(self):
        with pytest.raises(ValueError, match="must be boolean"):
            LimitsValidator._validate_channel_config("FOO", {"writable": "yes"})

    def test_non_bool_confirm_raises(self):
        with pytest.raises(ValueError, match="'confirm' must be boolean"):
            LimitsValidator._validate_channel_config("FOO", {"confirm": "yes"})

    def test_bool_confirm_is_accepted(self):
        LimitsValidator._validate_channel_config("FOO", {"confirm": False})


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

    def test_metadata_fields_are_skipped(self, tmp_path):
        f = tmp_path / "limits.json"
        f.write_text(json.dumps({"_comment": "ignored metadata", "GOOD": {"max_value": 10.0}}))

        limits_db, _ = LimitsValidator._load_limits_database(str(f))

        assert set(limits_db) == {"GOOD"}

    def test_non_dict_channel_raises(self, tmp_path):
        f = tmp_path / "limits.json"
        f.write_text(json.dumps({"BADCHAN": 42, "GOOD": {"max_value": 10.0}}))

        with pytest.raises(ValueError, match="BADCHAN"):
            LimitsValidator._load_limits_database(str(f))

    def test_channel_with_invalid_field_raises(self, tmp_path):
        f = tmp_path / "limits.json"
        f.write_text(json.dumps({"BADFIELD": {"min_value": "x"}, "GOOD": {"max_value": 10.0}}))

        with pytest.raises(ValueError, match="BADFIELD"):
            LimitsValidator._load_limits_database(str(f))

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
