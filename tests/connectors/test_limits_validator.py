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

import pytest

from osprey.connectors.control_system.limits_validator import LimitsValidator
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
