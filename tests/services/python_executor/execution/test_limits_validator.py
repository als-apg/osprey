"""Unit tests for Channel Limits Validator."""

import json
from unittest.mock import MagicMock

import pytest
import yaml
from pydantic import ValidationError

from osprey_connectors.control_system._limits_validator import (
    ChannelLimitsConfig,
    LimitsValidator,
    mapping_config_lookup,
)
from osprey_connectors.errors import ChannelLimitsViolationError
from osprey_connectors.testing import config_of
from osprey_connectors_tests._config_trees import serve_config, serve_control_system


class TestChannelLimitsConfig:
    """Test ChannelLimitsConfig dataclass."""

    def test_create_basic_config(self):
        """Test creating a basic channel config."""
        config = ChannelLimitsConfig(
            address="TEST:PV", min_value=0.0, max_value=100.0, writable=True
        )
        assert config.address == "TEST:PV"
        assert config.min_value == 0.0
        assert config.max_value == 100.0
        assert config.writable is True
        assert config.max_step is None

    def test_create_config_with_max_step(self):
        """Test creating config with max_step."""
        config = ChannelLimitsConfig(address="TEST:PV", max_step=5.0, writable=True)
        assert config.max_step == 5.0

    def test_writable_has_no_default(self):
        """A record states its own writability; none is assumed for it."""
        with pytest.raises(ValidationError, match="writable"):
            ChannelLimitsConfig(address="TEST:PV")

        config = ChannelLimitsConfig(address="TEST:PV", writable=False)
        assert config.writable is False


class TestLimitsValidator:
    """Test LimitsValidator class."""

    @pytest.fixture
    def basic_validator(self):
        """Create validator with basic test database."""
        test_db = {
            "TEST:PV": ChannelLimitsConfig(
                address="TEST:PV", min_value=0.0, max_value=100.0, writable=True
            ),
            "TEST:PV:READONLY": ChannelLimitsConfig(address="TEST:PV:READONLY", writable=False),
            "TEST:PV:NO_LIMITS": ChannelLimitsConfig(address="TEST:PV:NO_LIMITS", writable=True),
        }
        policy = {"mode": "exclusive"}
        return LimitsValidator(test_db, policy)

    @pytest.fixture
    def permissive_validator(self):
        """Create validator with permissive policy."""
        test_db = {
            "TEST:PV": ChannelLimitsConfig(
                address="TEST:PV", min_value=0.0, max_value=100.0, writable=True
            )
        }
        policy = {"mode": "optional"}
        return LimitsValidator(test_db, policy)

    @pytest.fixture
    def step_validator(self):
        """Create validator with max_step configured."""
        test_db = {
            "TEST:PV:STEP": ChannelLimitsConfig(
                address="TEST:PV:STEP",
                min_value=0.0,
                max_value=100.0,
                max_step=10.0,
                writable=True,
            )
        }
        policy = {"mode": "exclusive"}
        return LimitsValidator(test_db, policy)

    # =========================================================================
    # Valid Write Tests
    # =========================================================================

    @pytest.mark.parametrize(
        ("channel", "value"),
        [
            pytest.param("TEST:PV", 50.0, id="in-range"),
            pytest.param("TEST:PV", 0.0, id="at-min-boundary"),
            pytest.param("TEST:PV", 100.0, id="at-max-boundary"),
            # A PV with no limits allows any value.
            pytest.param("TEST:PV:NO_LIMITS", 9999.0, id="no-limits"),
            # No numeric limit, nothing numeric to hold the value to.
            pytest.param(
                "TEST:PV:NO_LIMITS", "some_string", id="non-numeric-without-numeric-limits"
            ),
            # This PV has no max_step configured - so it needs no reader at all.
            # Large change, but no max_step check.
            pytest.param("TEST:PV", 99.0, id="no-max-step-needs-no-reader"),
        ],
    )
    def test_a_write_the_limits_allow_passes_validation(self, basic_validator, channel, value):
        basic_validator.validate(channel, value)  # Should not raise

    def test_non_numeric_value_is_refused_on_a_numeric_limited_channel(self, basic_validator):
        """A value min/max cannot be applied to is refused, not skipped."""
        for value in ("some_string", None):
            with pytest.raises(ChannelLimitsViolationError) as exc_info:
                basic_validator.validate("TEST:PV", value)
            assert exc_info.value.violation_type == "INVALID_NUMERIC_VALUE"

    # =========================================================================
    # Min/Max Violation Tests
    # =========================================================================

    def test_below_min_raises_error(self, basic_validator):
        """Test that value below minimum raises exception."""
        with pytest.raises(ChannelLimitsViolationError) as exc_info:
            basic_validator.validate("TEST:PV", -10.0)

        error = exc_info.value
        assert error.violation_type == "MIN_EXCEEDED"
        assert error.address == "TEST:PV"
        assert error.attempted_value == -10.0
        assert error.min_value == 0.0
        assert error.max_value == 100.0

    def test_above_max_raises_error(self, basic_validator):
        """Test that value above maximum raises exception."""
        with pytest.raises(ChannelLimitsViolationError) as exc_info:
            basic_validator.validate("TEST:PV", 150.0)

        error = exc_info.value
        assert error.violation_type == "MAX_EXCEEDED"
        assert error.address == "TEST:PV"
        assert error.attempted_value == 150.0

    def test_min_exceeded_error_message(self, basic_validator):
        """Test error message for min violation."""
        with pytest.raises(ChannelLimitsViolationError) as exc_info:
            basic_validator.validate("TEST:PV", -10.0)

        error_msg = str(exc_info.value)
        assert "TEST:PV" in error_msg
        assert "-10.0" in error_msg
        assert "BLOCKED" in error_msg

    # =========================================================================
    # Read-Only PV Tests
    # =========================================================================

    @pytest.mark.parametrize(
        ("channel", "violation_type"),
        [
            # A read-only PV blocks all writes.
            pytest.param("TEST:PV:READONLY", "READ_ONLY_CHANNEL", id="readonly-pv"),
            # An unlisted PV fails when policy blocks them.
            pytest.param("UNKNOWN:PV", "UNLISTED_CHANNEL", id="unlisted-pv-in-strict-mode"),
        ],
    )
    def test_a_write_to_a_channel_the_policy_blocks_is_refused(
        self, basic_validator, channel, violation_type
    ):
        with pytest.raises(ChannelLimitsViolationError) as exc_info:
            basic_validator.validate(channel, 42.0)

        error = exc_info.value
        assert error.violation_type == violation_type
        assert error.address == channel

    # =========================================================================
    # Unlisted PV Tests
    # =========================================================================

    def test_unlisted_pv_passes_in_permissive_mode(self, permissive_validator):
        """Test that unlisted PV passes when policy allows them."""
        permissive_validator.validate("UNKNOWN:PV", 42.0)  # Should not raise

    # =========================================================================
    # Step Size Tests (with I/O mocking)
    # =========================================================================

    def test_max_step_violation(self, step_validator):
        """Test that excessive step size raises exception."""
        # The caller supplies the read; here it answers a known current value.
        reader = MagicMock(return_value=50.0)

        # This should work: step=5 (from 50 to 55)
        step_validator.validate("TEST:PV:STEP", 55.0, read_current=reader)

        # This should fail: step=45 (from 50 to 95)
        with pytest.raises(ChannelLimitsViolationError) as exc_info:
            step_validator.validate("TEST:PV:STEP", 95.0, read_current=reader)

        error = exc_info.value
        assert error.violation_type == "MAX_STEP_EXCEEDED"
        assert error.current_value == 50.0
        assert error.max_step == 10.0

    @pytest.mark.parametrize(
        "current_value",
        [
            # A channel read failure blocks the write (failsafe).
            pytest.param(None, id="read-failure"),
            # A non-numeric current value leaves no step to measure: fail closed.
            pytest.param("invalid", id="non-numeric-current-value"),
        ],
    )
    def test_max_step_unusable_current_value_blocks_write(self, step_validator, current_value):
        reader = MagicMock(return_value=current_value)

        with pytest.raises(ChannelLimitsViolationError) as exc_info:
            step_validator.validate("TEST:PV:STEP", 55.0, read_current=reader)

        assert exc_info.value.violation_type == "STEP_CHECK_FAILED"

    def test_max_step_without_a_reader_blocks_write(self, step_validator):
        """Test that a caller with no way to read the channel is refused."""
        with pytest.raises(ChannelLimitsViolationError) as exc_info:
            step_validator.validate("TEST:PV:STEP", 55.0)

        assert exc_info.value.violation_type == "STEP_CHECK_FAILED"

    # =========================================================================
    # Database Loading Tests
    # =========================================================================

    def test_load_valid_database(self, tmp_path):
        """Test loading valid database from JSON file."""
        db_file = tmp_path / "test_boundaries.json"
        db_content = {
            "TEST:PV1": {"min_value": 0.0, "max_value": 100.0, "writable": True},
            "TEST:PV2": {"writable": False},
            "TEST:PV3": {"min_value": -10.0, "max_value": 10.0, "max_step": 2.0, "writable": True},
        }
        db_file.write_text(json.dumps(db_content))

        db, raw_db = LimitsValidator._load_limits_database(str(db_file))

        assert len(db) == 3
        assert "TEST:PV1" in db
        assert db["TEST:PV1"].min_value == 0.0
        assert db["TEST:PV1"].max_value == 100.0
        assert "TEST:PV2" in db
        assert db["TEST:PV2"].writable is False
        assert "TEST:PV3" in db
        assert db["TEST:PV3"].max_step == 2.0

    def test_load_database_skips_comments(self, tmp_path):
        """Test that database loader skips entries starting with underscore."""
        db_file = tmp_path / "test_boundaries.json"
        db_content = {
            "_comment": "This is a comment",
            "_note": "Another comment",
            "TEST:PV": {"writable": True, "min_value": 0.0, "max_value": 100.0},
        }
        db_file.write_text(json.dumps(db_content))

        db, raw_db = LimitsValidator._load_limits_database(str(db_file))

        assert len(db) == 1
        assert "TEST:PV" in db
        assert "_comment" not in db
        assert "_note" not in db

    def test_load_database_invalid_json_raises_error(self, tmp_path):
        """Test that invalid JSON raises ValueError with clear message."""
        db_file = tmp_path / "invalid.json"
        db_file.write_text("{ invalid json }")

        # Should raise ValueError with clear error message
        with pytest.raises(ValueError) as exc_info:
            LimitsValidator._load_limits_database(str(db_file))

        error_msg = str(exc_info.value)
        assert "Invalid JSON in channel limits database" in error_msg
        assert "JSONDecodeError" in str(type(exc_info.value.__cause__))

    def test_load_database_invalid_json_with_trailing_comma(self, tmp_path):
        """Test that trailing comma in JSON raises clear error."""
        db_file = tmp_path / "trailing_comma.json"
        # This is the actual error case we had - trailing comma
        db_file.write_text('{\n  "PV1": {"min_value": 0.0},\n}')

        with pytest.raises(ValueError) as exc_info:
            LimitsValidator._load_limits_database(str(db_file))

        error_msg = str(exc_info.value)
        assert "Invalid JSON" in error_msg

    def test_load_database_missing_file_raises_error(self):
        """Test that missing database file raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            LimitsValidator._load_limits_database("/nonexistent/file.json")

        error_msg = str(exc_info.value)
        assert "Failed to load channel limits database" in error_msg

    def test_load_database_fails_closed_on_an_invalid_entry(self, tmp_path):
        """One bad entry fails the whole load, naming the channel.

        Skipping the entry used to drop the channel from the database, which
        under ``mode: optional`` then waved its writes through with no
        limits at all. The load now raises instead.
        """
        db_file = tmp_path / "test_boundaries.json"
        db_content = {
            "VALID:PV": {"writable": True, "min_value": 0.0, "max_value": 100.0},
            "INVALID:PV1": {"writable": True, "min_value": "not_a_number"},  # Invalid type
            "VALID:PV2": {"writable": False},
        }
        db_file.write_text(json.dumps(db_content))

        with pytest.raises(ValueError) as exc_info:
            LimitsValidator._load_limits_database(str(db_file))

        assert "INVALID:PV1" in str(exc_info.value)

    # =========================================================================
    # Configuration Loading Tests
    # =========================================================================

    def test_from_config_disabled(self):
        """Test that from_config returns None when disabled.

        The ``control_system`` section is ``False``: a section that is not a
        mapping is a deployment that stated no posture, so the answer is still
        no validator rather than a crash on the shape.
        """
        serve_config(config_of({"control_system": False}, environ={}))

        validator = LimitsValidator.from_config()

        assert validator is None

    @pytest.mark.parametrize(
        "database_path",
        [
            pytest.param(None, id="no-database-path"),
            # Same contract as the invalid-JSON case above.
            pytest.param("/nonexistent/path/to/limits.json", id="missing-file"),
        ],
    )
    def test_from_config_without_a_usable_database_blocks_all_writes(self, database_path):
        """from_config degrades to the empty block-all validator."""
        serve_control_system({"limits_checking": {"enabled": True, "database_path": database_path}})

        validator = LimitsValidator.from_config()

        assert validator is not None
        assert validator.limits == {}  # Empty database (blocks all writes)

    def test_from_config_loads_database(self, tmp_path):
        """Test that from_config loads database from configured path."""
        db_file = tmp_path / "test_boundaries.json"
        db_content = {"TEST:PV": {"min_value": 0.0, "max_value": 100.0, "writable": True}}
        db_file.write_text(json.dumps(db_content))

        serve_control_system(
            {
                "limits_checking": {
                    "enabled": True,
                    "database_path": str(db_file),
                    "mode": "exclusive",
                }
            }
        )

        validator = LimitsValidator.from_config()

        assert validator is not None
        assert len(validator.limits) == 1
        assert "TEST:PV" in validator.limits
        assert validator.policy["mode"] == "exclusive"

    def test_from_config_invalid_json_blocks_all_writes(self, tmp_path):
        """An unparseable database degrades to the empty block-all validator.

        Same failsafe as the missing-database-path case: never crash the
        connector (a read-only deployment needs no limits), never return None
        (writes would go unchecked) — an empty DB refuses every write instead.
        """
        db_file = tmp_path / "invalid.json"
        db_file.write_text('{"PV1": {"min_value": 0.0},}')  # Trailing comma

        serve_control_system(
            {
                "limits_checking": {
                    "enabled": True,
                    "database_path": str(db_file),
                    "mode": "exclusive",
                }
            }
        )

        validator = LimitsValidator.from_config()

        assert validator is not None
        assert validator.limits == {}  # Empty database (blocks all writes)

    # =========================================================================
    # Relative database_path resolution base (R4 container fix)
    # =========================================================================

    def test_from_config_resolves_relative_path_against_config_file_dir(
        self, monkeypatch, tmp_path
    ):
        """A relative database_path resolves against the loaded config's
        directory (here the file OSPREY_CONFIG names), NOT against project_root.

        Simulates the container deploy: config.yml (and OSPREY_CONFIG) live
        under the mounted /app/project, while project_root is the flattened
        HOST build path — which does not exist in the container and must
        NOT be used as the resolution base.
        """
        # OSPREY_CONFIG dir: where the mounted config.yml actually lives.
        container_dir = tmp_path / "app_project"
        container_dir.mkdir()
        db_file = container_dir / "data" / "channel_limits.json"
        db_file.parent.mkdir()
        db_file.write_text(
            json.dumps({"TEST:PV": {"writable": True, "min_value": 0.0, "max_value": 100.0}})
        )

        # project_root: a HOST path that does NOT exist in this environment
        # either — proves resolution did not fall through to it.
        bogus_project_root = tmp_path / "host_build_path_does_not_exist"
        config_file = container_dir / "config.yml"
        config_file.write_text(
            yaml.safe_dump(
                {
                    "project_root": str(bogus_project_root),
                    "control_system": {
                        "limits_checking": {
                            "enabled": True,
                            "database_path": "data/channel_limits.json",
                            "mode": "exclusive",
                        }
                    },
                }
            )
        )
        monkeypatch.setenv("OSPREY_CONFIG", str(config_file))
        monkeypatch.chdir(tmp_path)

        validator = LimitsValidator.from_config()

        assert validator is not None
        assert len(validator.limits) == 1
        assert "TEST:PV" in validator.limits

    def test_a_lookup_without_a_config_file_resolves_against_project_root(
        self, monkeypatch, tmp_path
    ):
        """Last resort: a caller holding the config as a dict, naming no config
        file, with OSPREY_CONFIG unset, resolves a relative database_path
        against the config's project_root.

        ``from_config`` itself always loads a config first and anchors on that
        file's directory, so this fallback is reached only by a caller that
        supplies its own lookup and no anchor.
        """
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)

        project_root = tmp_path / "project"
        project_root.mkdir()
        db_file = project_root / "data" / "channel_limits.json"
        db_file.parent.mkdir()
        db_file.write_text(
            json.dumps({"TEST:PV": {"writable": True, "min_value": 0.0, "max_value": 100.0}})
        )
        lookup = mapping_config_lookup(
            {
                "project_root": str(project_root),
                "control_system": {
                    "limits_checking": {
                        "enabled": True,
                        "database_path": "data/channel_limits.json",
                        "mode": "exclusive",
                    }
                },
            }
        )

        loaded, reason = LimitsValidator.load_configured_database(config_lookup=lookup)

        assert reason is None
        assert loaded is not None
        limits, _raw = loaded
        assert list(limits) == ["TEST:PV"]


class TestChannelLimitsViolationError:
    """Test ChannelLimitsViolationError exception class."""

    def test_error_message_format(self):
        """Test that error message is properly formatted."""
        error = ChannelLimitsViolationError(
            address="TEST:PV",
            value=150.0,
            violation_type="MAX_EXCEEDED",
            violation_reason="Value 150.0 above maximum 100.0",
            min_value=0.0,
            max_value=100.0,
        )

        error_msg = str(error)
        assert "TEST:PV" in error_msg
        assert "150.0" in error_msg
        assert "BLOCKED" in error_msg
        assert "[0.0, 100.0]" in error_msg

    def test_error_with_step_info(self):
        """Test error message includes step information."""
        error = ChannelLimitsViolationError(
            address="TEST:PV",
            value=95.0,
            violation_type="MAX_STEP_EXCEEDED",
            violation_reason="Step size 45.0 exceeds maximum 10.0",
            current_value=50.0,
            max_step=10.0,
            min_value=0.0,
            max_value=100.0,
        )

        error_msg = str(error)
        assert "Current Value: 50.0" in error_msg
        assert "Maximum Step Size: 10.0" in error_msg

    def test_inherits_from_exception(self):
        """Test that exception inherits from Exception (framework-level)."""
        error = ChannelLimitsViolationError(
            address="TEST:PV",
            value=150.0,
            violation_type="MAX_EXCEEDED",
            violation_reason="Test",
        )

        assert isinstance(error, Exception)
        assert error.address == "TEST:PV"
        assert error.violation_type == "MAX_EXCEEDED"
