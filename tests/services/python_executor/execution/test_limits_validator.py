"""Unit tests for Channel Limits Validator."""
import json
from unittest.mock import MagicMock, patch

import pytest

from osprey.services.python_executor.exceptions import ChannelLimitsViolationError
from osprey.services.python_executor.execution.limits_validator import (
    ChannelLimitsConfig,
    LimitsValidator,
)


class TestChannelLimitsConfig:
    """Test ChannelLimitsConfig dataclass."""

    def test_create_basic_config(self):
        """Test creating a basic channel config."""
        config = ChannelLimitsConfig(
            channel_address="TEST:PV",
            min_value=0.0,
            max_value=100.0,
            writable=True
        )
        assert config.channel_address == "TEST:PV"
        assert config.min_value == 0.0
        assert config.max_value == 100.0
        assert config.writable is True
        assert config.max_step is None

    def test_create_config_with_max_step(self):
        """Test creating config with max_step."""
        config = ChannelLimitsConfig(
            channel_address="TEST:PV",
            max_step=5.0,
            writable=True
        )
        assert config.max_step == 5.0

    def test_default_writable_value(self):
        """Test default writable value is True."""
        config = ChannelLimitsConfig(channel_address="TEST:PV")
        assert config.writable is True


class TestLimitsValidator:
    """Test LimitsValidator class."""

    @pytest.fixture
    def basic_validator(self):
        """Create validator with basic test database."""
        test_db = {
            "TEST:PV": ChannelLimitsConfig(
                channel_address="TEST:PV",
                min_value=0.0,
                max_value=100.0,
                writable=True
            ),
            "TEST:PV:READONLY": ChannelLimitsConfig(
                channel_address="TEST:PV:READONLY",
                writable=False
            ),
            "TEST:PV:NO_LIMITS": ChannelLimitsConfig(
                channel_address="TEST:PV:NO_LIMITS",
                writable=True
            )
        }
        policy = {"allow_unlisted_pvs": False}
        return LimitsValidator(test_db, policy)

    @pytest.fixture
    def permissive_validator(self):
        """Create validator with permissive policy."""
        test_db = {
            "TEST:PV": ChannelLimitsConfig(
                channel_address="TEST:PV",
                min_value=0.0,
                max_value=100.0,
                writable=True
            )
        }
        policy = {"allow_unlisted_channels": True}
        return LimitsValidator(test_db, policy)

    @pytest.fixture
    def step_validator(self):
        """Create validator with max_step configured."""
        test_db = {
            "TEST:PV:STEP": ChannelLimitsConfig(
                channel_address="TEST:PV:STEP",
                min_value=0.0,
                max_value=100.0,
                max_step=10.0,
                writable=True
            )
        }
        policy = {"allow_unlisted_pvs": False}
        return LimitsValidator(test_db, policy)

    # =========================================================================
    # Valid Write Tests
    # =========================================================================

    def test_valid_write_passes(self, basic_validator):
        """Test that valid write passes validation without exception."""
        basic_validator.validate("TEST:PV", 50.0)  # Should not raise

    def test_valid_write_at_min_boundary(self, basic_validator):
        """Test value exactly at minimum boundary."""
        basic_validator.validate("TEST:PV", 0.0)  # Should not raise

    def test_valid_write_at_max_boundary(self, basic_validator):
        """Test value exactly at maximum boundary."""
        basic_validator.validate("TEST:PV", 100.0)  # Should not raise

    def test_valid_write_no_limits(self, basic_validator):
        """Test PV with no limits allows any value."""
        basic_validator.validate("TEST:PV:NO_LIMITS", 9999.0)  # Should not raise

    def test_non_numeric_value_skips_checks(self, basic_validator):
        """Test that non-numeric values skip min/max checks."""
        basic_validator.validate("TEST:PV", "some_string")  # Should not raise
        basic_validator.validate("TEST:PV", None)  # Should not raise

    # =========================================================================
    # Min/Max Violation Tests
    # =========================================================================

    def test_below_min_raises_error(self, basic_validator):
        """Test that value below minimum raises exception."""
        with pytest.raises(ChannelLimitsViolationError) as exc_info:
            basic_validator.validate("TEST:PV", -10.0)

        error = exc_info.value
        assert error.violation_type == "MIN_EXCEEDED"
        assert error.channel_address == "TEST:PV"
        assert error.attempted_value == -10.0
        assert error.min_value == 0.0
        assert error.max_value == 100.0

    def test_above_max_raises_error(self, basic_validator):
        """Test that value above maximum raises exception."""
        with pytest.raises(ChannelLimitsViolationError) as exc_info:
            basic_validator.validate("TEST:PV", 150.0)

        error = exc_info.value
        assert error.violation_type == "MAX_EXCEEDED"
        assert error.channel_address == "TEST:PV"
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

    def test_readonly_pv_blocks_write(self, basic_validator):
        """Test that read-only PV blocks all writes."""
        with pytest.raises(ChannelLimitsViolationError) as exc_info:
            basic_validator.validate("TEST:PV:READONLY", 42.0)

        error = exc_info.value
        assert error.violation_type == "READ_ONLY_CHANNEL"
        assert error.channel_address == "TEST:PV:READONLY"

    # =========================================================================
    # Unlisted PV Tests
    # =========================================================================

    def test_unlisted_pv_fails_in_strict_mode(self, basic_validator):
        """Test that unlisted PV fails when policy blocks them."""
        with pytest.raises(ChannelLimitsViolationError) as exc_info:
            basic_validator.validate("UNKNOWN:PV", 42.0)

        error = exc_info.value
        assert error.violation_type == "UNLISTED_CHANNEL"
        assert error.channel_address == "UNKNOWN:PV"

    def test_unlisted_pv_passes_in_permissive_mode(self, permissive_validator):
        """Test that unlisted PV passes when policy allows them."""
        permissive_validator.validate("UNKNOWN:PV", 42.0)  # Should not raise

    # =========================================================================
    # Step Size Tests (with I/O mocking)
    # =========================================================================

    def test_max_step_violation(self, step_validator):
        """Test that excessive step size raises exception."""
        # Mock epics module with caget that returns known current value
        mock_epics = MagicMock()
        mock_epics.caget = MagicMock(return_value=50.0)

        with patch.dict('sys.modules', {'epics': mock_epics}):
            # This should work: step=5 (from 50 to 55)
            step_validator.validate("TEST:PV:STEP", 55.0)

            # Reset mock for next call
            mock_epics.caget.return_value = 50.0

            # This should fail: step=45 (from 50 to 95)
            with pytest.raises(ChannelLimitsViolationError) as exc_info:
                step_validator.validate("TEST:PV:STEP", 95.0)

            error = exc_info.value
            assert error.violation_type == "MAX_STEP_EXCEEDED"
            assert error.current_value == 50.0
            assert error.max_step == 10.0

    def test_max_step_read_failure_blocks_write(self, step_validator):
        """Test that PV read failure blocks write (failsafe)."""
        # Mock epics.caget to return None (read failed)
        mock_epics = MagicMock()
        mock_epics.caget = MagicMock(return_value=None)

        with patch.dict('sys.modules', {'epics': mock_epics}):
            # Should fail because we can't read current value
            with pytest.raises(ChannelLimitsViolationError) as exc_info:
                step_validator.validate("TEST:PV:STEP", 55.0)

            assert exc_info.value.violation_type == "STEP_CHECK_FAILED"

    def test_max_step_epics_import_failure(self, step_validator):
        """Test that epics import failure blocks write when step checking required."""
        # Remove epics from sys.modules to simulate ImportError
        with patch.dict('sys.modules', {'epics': None}):
            with pytest.raises(ChannelLimitsViolationError) as exc_info:
                step_validator.validate("TEST:PV:STEP", 55.0)

            assert exc_info.value.violation_type == "STEP_CHECK_FAILED"

    def test_max_step_no_io_without_config(self, basic_validator):
        """Test that max_step checking is skipped if not configured (no I/O)."""
        # This PV has no max_step configured - should NOT trigger any epics.caget()
        # If it does, this test will fail because epics is not mocked
        basic_validator.validate("TEST:PV", 99.0)  # Large change, but no max_step check

    def test_max_step_non_numeric_current_value(self, step_validator):
        """Test that non-numeric current value skips step check."""
        mock_epics = MagicMock()
        mock_epics.caget = MagicMock(return_value="invalid")  # Non-numeric current value

        with patch.dict('sys.modules', {'epics': mock_epics}):
            # Should not raise because current value is non-numeric
            step_validator.validate("TEST:PV:STEP", 55.0)

    # =========================================================================
    # Database Loading Tests
    # =========================================================================

    def test_load_valid_database(self, tmp_path):
        """Test loading valid database from JSON file."""
        db_file = tmp_path / "test_boundaries.json"
        db_content = {
            "TEST:PV1": {
                "min_value": 0.0,
                "max_value": 100.0,
                "writable": True
            },
            "TEST:PV2": {
                "writable": False
            },
            "TEST:PV3": {
                "min_value": -10.0,
                "max_value": 10.0,
                "max_step": 2.0,
                "writable": True
            }
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
            "TEST:PV": {
                "min_value": 0.0,
                "max_value": 100.0
            }
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

    def test_load_database_skips_invalid_entries(self, tmp_path):
        """Test that database loader skips entries with invalid config."""
        db_file = tmp_path / "test_boundaries.json"
        db_content = {
            "VALID:PV": {
                "min_value": 0.0,
                "max_value": 100.0
            },
            "INVALID:PV1": {
                "min_value": "not_a_number"  # Invalid type
            },
            "INVALID:PV2": {
                "max_step": "also_invalid"  # Invalid type
            },
            "VALID:PV2": {
                "writable": False
            }
        }
        db_file.write_text(json.dumps(db_content))

        db, raw_db = LimitsValidator._load_limits_database(str(db_file))

        # Should load only valid entries
        assert len(db) == 2
        assert "VALID:PV" in db
        assert "VALID:PV2" in db
        assert "INVALID:PV1" not in db
        assert "INVALID:PV2" not in db

    # =========================================================================
    # Configuration Loading Tests
    # =========================================================================

    @patch('osprey.utils.config.get_config_value')
    def test_from_config_disabled(self, mock_get_config):
        """Test that from_config returns None when disabled."""
        mock_get_config.return_value = False

        validator = LimitsValidator.from_config()

        assert validator is None

    @patch('osprey.utils.config.get_config_value')
    def test_from_config_no_database_path(self, mock_get_config):
        """Test that from_config returns empty validator when no database path."""
        def config_side_effect(key, default):
            if key == 'control_system.limits_checking.enabled':
                return True
            elif key == 'control_system.limits_checking.database_path':
                return None
            elif key == 'control_system.limits_checking.allow_unlisted_pvs':
                return False
            return default

        mock_get_config.side_effect = config_side_effect

        validator = LimitsValidator.from_config()

        assert validator is not None
        assert validator.limits == {}  # Empty database (blocks all writes)

    @patch('osprey.utils.config.get_config_value')
    def test_from_config_loads_database(self, mock_get_config, tmp_path):
        """Test that from_config loads database from configured path."""
        db_file = tmp_path / "test_boundaries.json"
        db_content = {
            "TEST:PV": {
                "min_value": 0.0,
                "max_value": 100.0,
                "writable": True
            }
        }
        db_file.write_text(json.dumps(db_content))

        def config_side_effect(key, default):
            if key == 'control_system.limits_checking.enabled':
                return True
            elif key == 'control_system.limits_checking.database_path':
                return str(db_file)
            elif key == 'control_system.limits_checking.allow_unlisted_channels':
                return False
            return default

        mock_get_config.side_effect = config_side_effect

        validator = LimitsValidator.from_config()

        assert validator is not None
        assert len(validator.limits) == 1
        assert "TEST:PV" in validator.limits
        assert validator.policy['allow_unlisted_channels'] is False

    @patch('osprey.utils.config.get_config_value')
    def test_from_config_invalid_json_fails_fast(self, mock_get_config, tmp_path):
        """Test that from_config raises error on invalid JSON (fail-fast at init)."""
        db_file = tmp_path / "invalid.json"
        db_file.write_text('{"PV1": {"min_value": 0.0},}')  # Trailing comma

        def config_side_effect(key, default):
            if key == 'control_system.limits_checking.enabled':
                return True
            elif key == 'control_system.limits_checking.database_path':
                return str(db_file)
            elif key == 'control_system.limits_checking.allow_unlisted_channels':
                return False
            return default

        mock_get_config.side_effect = config_side_effect

        # Should raise ValueError during initialization (fail-fast)
        with pytest.raises(ValueError) as exc_info:
            LimitsValidator.from_config()

        error_msg = str(exc_info.value)
        assert "Invalid JSON" in error_msg

    @patch('osprey.utils.config.get_config_value')
    def test_from_config_missing_file_fails_fast(self, mock_get_config):
        """Test that from_config raises error on missing file (fail-fast at init)."""
        def config_side_effect(key, default):
            if key == 'control_system.limits_checking.enabled':
                return True
            elif key == 'control_system.limits_checking.database_path':
                return "/nonexistent/path/to/limits.json"
            elif key == 'control_system.limits_checking.allow_unlisted_pvs':
                return False
            return default

        mock_get_config.side_effect = config_side_effect

        # Should raise ValueError during initialization (fail-fast)
        with pytest.raises(ValueError) as exc_info:
            LimitsValidator.from_config()

        error_msg = str(exc_info.value)
        assert "Failed to load" in error_msg


class TestChannelLimitsViolationError:
    """Test ChannelLimitsViolationError exception class."""

    def test_error_message_format(self):
        """Test that error message is properly formatted."""
        error = ChannelLimitsViolationError(
            channel_address="TEST:PV",
            value=150.0,
            violation_type="MAX_EXCEEDED",
            violation_reason="Value 150.0 above maximum 100.0",
            min_value=0.0,
            max_value=100.0
        )

        error_msg = str(error)
        assert "TEST:PV" in error_msg
        assert "150.0" in error_msg
        assert "BLOCKED" in error_msg
        assert "[0.0, 100.0]" in error_msg

    def test_error_with_step_info(self):
        """Test error message includes step information."""
        error = ChannelLimitsViolationError(
            channel_address="TEST:PV",
            value=95.0,
            violation_type="MAX_STEP_EXCEEDED",
            violation_reason="Step size 45.0 exceeds maximum 10.0",
            current_value=50.0,
            max_step=10.0,
            min_value=0.0,
            max_value=100.0
        )

        error_msg = str(error)
        assert "Current Value: 50.0" in error_msg
        assert "Maximum Step Size: 10.0" in error_msg

    def test_error_category_is_code_related(self):
        """Test that exception category is CODE_RELATED."""
        from osprey.services.python_executor.exceptions import ErrorCategory

        error = ChannelLimitsViolationError(
            channel_address="TEST:PV",
            value=150.0,
            violation_type="MAX_EXCEEDED",
            violation_reason="Test"
        )

        assert error.category == ErrorCategory.CODE_RELATED
        assert error.is_code_error() is True
        assert error.should_retry_code_generation() is True

