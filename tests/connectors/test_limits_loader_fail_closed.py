"""The limits database loader fails closed on any malformed entry.

A limits file is a safety artifact: a key the loader does not recognise is a
key whose intent was not applied. Before, an unknown field only warned and a
broken channel entry was skipped, so a typo silently downgraded a channel to
"unlisted" — and with ``allow_unlisted_channels: true`` that means unlimited.
Now one bad entry refuses the whole load, and the operator-facing refusal names
the offending key.
"""

import json

import pytest

from osprey.connectors.control_system.limits_validator import LimitsValidator
from osprey.errors import ChannelLimitsViolationError


def _patch_config(monkeypatch, db_file, allow_unlisted: bool = False):
    """Point from_config at a limits file on disk.

    ``from_config`` reads the nested ``control_system`` section to resolve the
    posture and the dotted key for the database path, so the shim answers both
    spellings of the one deployment these tests describe.
    """
    values = {
        "control_system": {
            "limits_checking": {
                "enabled": True,
                "allow_unlisted_channels": allow_unlisted,
            },
        },
        "control_system.limits_checking.enabled": True,
        "control_system.limits_checking.database_path": str(db_file),
        "control_system.limits_checking.allow_unlisted_channels": allow_unlisted,
        "project_root": None,
    }
    monkeypatch.setattr(
        "osprey.utils.config.get_config_value", lambda key, default=None: values.get(key, default)
    )
    monkeypatch.setattr("osprey.utils.config.default_config_path", lambda: None)


def _write_db(tmp_path, db: dict):
    limits_file = tmp_path / "limits.json"
    limits_file.write_text(json.dumps(db))
    return limits_file


def _refusal(monkeypatch, tmp_path, db: dict, allow_unlisted: bool = False) -> str:
    """Load through from_config and return the refusal an operator would see."""
    _patch_config(monkeypatch, _write_db(tmp_path, db), allow_unlisted=allow_unlisted)

    validator = LimitsValidator.from_config()

    assert isinstance(validator, LimitsValidator)
    with pytest.raises(ChannelLimitsViolationError) as exc:
        validator.validate("FOO", 1.0)
    assert exc.value.violation_type == "LIMITS_DATABASE_UNAVAILABLE"
    return exc.value.violation_reason


class TestRetiredVerificationBlock:
    def test_verification_block_fails_the_load(self, tmp_path):
        db_file = _write_db(
            tmp_path, {"FOO": {"max_value": 10.0, "verification": {"level": "readback"}}}
        )

        with pytest.raises(ValueError) as exc:
            LimitsValidator._load_limits_database(str(db_file))

        assert "verification" in str(exc.value)
        assert "confirm" in str(exc.value)

    def test_refusal_points_at_confirm(self, monkeypatch, tmp_path):
        reason = _refusal(
            monkeypatch, tmp_path, {"FOO": {"max_value": 10.0, "verification": {"level": "none"}}}
        )

        assert "confirm" in reason
        assert "verification" in reason

    def test_verification_in_defaults_fails_the_load(self, tmp_path):
        db_file = _write_db(tmp_path, {"defaults": {"verification": {"level": "callback"}}})

        with pytest.raises(ValueError, match="confirm"):
            LimitsValidator._load_limits_database(str(db_file))


class TestUnknownFields:
    def test_typo_fails_the_load_and_is_named(self, tmp_path):
        db_file = _write_db(tmp_path, {"FOO": {"max_value": 10.0, "confrim": True}})

        with pytest.raises(ValueError, match="confrim"):
            LimitsValidator._load_limits_database(str(db_file))

    def test_typo_refusal_names_the_key_even_when_unlisted_channels_are_allowed(
        self, monkeypatch, tmp_path
    ):
        # allow_unlisted_channels used to be the dangerous half of this bug: the
        # typo'd channel was dropped from the database, and an unlisted channel
        # was then waved through with no limits at all.
        reason = _refusal(
            monkeypatch,
            tmp_path,
            {"FOO": {"max_value": 10.0, "confrim": True}},
            allow_unlisted=True,
        )

        assert "confrim" in reason

    def test_underscore_metadata_is_legal(self, tmp_path):
        db_file = _write_db(
            tmp_path,
            {
                "_comment": "top-level metadata",
                "FOO": {"max_value": 10.0, "_units": "mA", "_owner": "APG"},
            },
        )

        limits_db, _ = LimitsValidator._load_limits_database(str(db_file))

        assert limits_db["FOO"].max_value == 10.0

    def test_the_same_key_without_the_underscore_fails_the_load(self, tmp_path):
        db_file = _write_db(tmp_path, {"FOO": {"max_value": 10.0, "units": "mA"}})

        with pytest.raises(ValueError, match="units"):
            LimitsValidator._load_limits_database(str(db_file))


class TestMalformedEntries:
    def test_non_bool_confirm_fails_the_load(self, tmp_path):
        db_file = _write_db(tmp_path, {"FOO": {"confirm": "yes"}})

        with pytest.raises(ValueError, match="must be boolean"):
            LimitsValidator._load_limits_database(str(db_file))

    def test_non_dict_channel_entry_fails_the_load(self, tmp_path):
        db_file = _write_db(tmp_path, {"BADCHAN": 42, "FOO": {"max_value": 10.0}})

        with pytest.raises(ValueError, match="BADCHAN"):
            LimitsValidator._load_limits_database(str(db_file))

    def test_one_bad_entry_takes_down_the_whole_load(self, tmp_path):
        # The neighbouring good channel must not load either — a partially
        # loaded database is exactly the fail-open state being removed.
        db_file = _write_db(tmp_path, {"GOOD": {"max_value": 10.0}, "BAD": {"min_value": "x"}})

        with pytest.raises(ValueError, match="BAD"):
            LimitsValidator._load_limits_database(str(db_file))

    def test_bool_confirm_loads(self, tmp_path):
        db_file = _write_db(tmp_path, {"FOO": {"max_value": 10.0, "confirm": False}})

        limits_db, raw_db = LimitsValidator._load_limits_database(str(db_file))

        assert set(limits_db) == {"FOO"}
        assert raw_db["FOO"]["confirm"] is False
