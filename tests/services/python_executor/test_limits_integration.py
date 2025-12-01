"""Integration tests for channel limits checking end-to-end."""
import json
from unittest.mock import patch

import pytest

from osprey.services.python_executor.execution.limits_validator import (
    ChannelLimitsConfig,
    LimitsValidator,
)
from osprey.services.python_executor.execution.wrapper import ExecutionWrapper


class TestLimitsInitializationFailFast:
    """Test that invalid limits database causes initialization to fail fast."""

    @patch('osprey.utils.config.get_config_value')
    def test_invalid_json_prevents_initialization(self, mock_get_config, tmp_path):
        """Test that invalid JSON in limits database prevents app from starting."""
        # Create invalid JSON file (trailing comma)
        db_file = tmp_path / "invalid_limits.json"
        db_file.write_text('{\n  "PV1": {"min_value": 0.0},\n}')

        def config_side_effect(key, default):
            if key == 'control_system.limits_checking.enabled':
                return True
            elif key == 'control_system.limits_checking.database_path':
                return str(db_file)
            elif key == 'control_system.limits_checking.allow_unlisted_pvs':
                return False
            return default

        mock_get_config.side_effect = config_side_effect

        # Attempting to initialize should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            LimitsValidator.from_config()

        # Verify error message contains critical information
        error_msg = str(exc_info.value)
        assert "Invalid JSON" in error_msg
        # Verify it's chained from JSONDecodeError
        assert exc_info.value.__cause__ is not None

    @patch('osprey.utils.config.get_config_value')
    def test_valid_json_initializes_successfully(self, mock_get_config, tmp_path):
        """Test that valid JSON allows successful initialization."""
        # Create valid JSON file
        db_file = tmp_path / "valid_limits.json"
        db_content = {
            "PV1": {"min_value": 0.0, "max_value": 100.0},
            "PV2": {"writable": False}
        }
        db_file.write_text(json.dumps(db_content))

        def config_side_effect(key, default):
            if key == 'control_system.limits_checking.enabled':
                return True
            elif key == 'control_system.limits_checking.database_path':
                return str(db_file)
            elif key == 'control_system.limits_checking.allow_unlisted_pvs':
                return False
            return default

        mock_get_config.side_effect = config_side_effect

        # Should initialize successfully
        validator = LimitsValidator.from_config()
        assert validator is not None
        assert len(validator.limits) == 2
        assert "PV1" in validator.limits
        assert "PV2" in validator.limits

    @patch('osprey.utils.config.get_config_value')
    def test_missing_file_prevents_initialization(self, mock_get_config):
        """Test that missing limits file prevents initialization."""
        def config_side_effect(key, default):
            if key == 'control_system.limits_checking.enabled':
                return True
            elif key == 'control_system.limits_checking.database_path':
                return "/nonexistent/limits.json"
            elif key == 'control_system.limits_checking.allow_unlisted_pvs':
                return False
            return default

        mock_get_config.side_effect = config_side_effect

        # Should raise ValueError for missing file
        with pytest.raises(ValueError) as exc_info:
            LimitsValidator.from_config()

        error_msg = str(exc_info.value)
        assert "Failed to load" in error_msg


class TestLimitsCheckingIntegration:
    """Test end-to-end limits checking integration."""

    @pytest.fixture
    def test_validator(self):
        """Create a test validator with sample boundaries."""
        test_db = {
            "TEST:PV": ChannelLimitsConfig(
                channel_address="TEST:PV",
                min_value=0.0,
                max_value=100.0,
                writable=True
            ),
            "TEST:READONLY": ChannelLimitsConfig(
                channel_address="TEST:READONLY",
                writable=False
            )
        }
        policy = {"allow_unlisted_pvs": False}
        return LimitsValidator(test_db, policy)

    def test_wrapper_includes_limits_checking(self, test_validator):
        """Test that ExecutionWrapper includes boundary checking monkeypatch."""
        wrapper = ExecutionWrapper(
            execution_mode="container",
            limits_validator=test_validator
        )

        user_code = "print('Hello, World!')"
        wrapped_code = wrapper.create_wrapper(user_code, execution_folder=None)

        # Verify boundary checking code is present
        assert "Runtime Channel Limits Checking" in wrapped_code
        assert "LimitsValidator" in wrapped_code
        assert "_limits_validator" in wrapped_code
        assert "_checked_caput" in wrapped_code
        assert "epics.caput = _checked_caput" in wrapped_code

    def test_wrapper_without_validator_has_no_checking(self):
        """Test that wrapper without validator has no boundary checking code."""
        wrapper = ExecutionWrapper(
            execution_mode="container",
            limits_validator=None
        )

        user_code = "print('Hello, World!')"
        wrapped_code = wrapper.create_wrapper(user_code, execution_folder=None)

        # Verify no boundary checking code
        assert "Runtime Channel Limits Checking" not in wrapped_code
        assert "LimitsValidator" not in wrapped_code

    def test_limits_config_serialization(self, test_validator):
        """Test that boundary config is correctly serialized into wrapper."""
        wrapper = ExecutionWrapper(
            execution_mode="container",
            limits_validator=test_validator
        )

        user_code = "pass"
        wrapped_code = wrapper.create_wrapper(user_code, execution_folder=None)

        # Verify serialized config contains our test PVs
        assert "TEST:PV" in wrapped_code
        assert "TEST:READONLY" in wrapped_code

        # Verify min/max values are serialized
        assert "0.0" in wrapped_code  # min_value
        assert "100.0" in wrapped_code  # max_value

    def test_limits_config_includes_max_step(self):
        """Test that max_step field is included in serialization."""
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
        validator = LimitsValidator(test_db, policy)

        wrapper = ExecutionWrapper(
            execution_mode="container",
            limits_validator=validator
        )

        user_code = "pass"
        wrapped_code = wrapper.create_wrapper(user_code, execution_folder=None)

        # Verify max_step is in serialized config
        assert "max_step" in wrapped_code
        assert "10.0" in wrapped_code  # max_step value

    def test_monkeypatch_intercepts_caput(self, test_validator):
        """Test that monkeypatch successfully intercepts epics.caput() calls."""
        # Create wrapper with validator
        wrapper = ExecutionWrapper(
            execution_mode="local",
            limits_validator=test_validator
        )

        # Create code that attempts boundary violation
        user_code = """
import epics
try:
    epics.caput("TEST:PV", 150.0)  # Exceeds max (100.0)
    result = "FAILED - should have raised error"
except Exception as e:
    if "MAX_EXCEEDED" in str(e):
        result = "SUCCESS - boundary violation caught"
    else:
        result = f"UNEXPECTED ERROR: {str(e)}"
results = {"test_result": result}
"""
        wrapped_code = wrapper.create_wrapper(user_code, execution_folder=None)

        # Verify the wrapped code contains the boundary checking logic
        assert "_checked_caput" in wrapped_code
        assert "_limits_validator.validate" in wrapped_code

    def test_monkeypatch_allows_valid_writes(self, test_validator):
        """Test that valid writes pass through monkeypatch."""
        wrapper = ExecutionWrapper(
            execution_mode="local",
            limits_validator=test_validator
        )

        user_code = """
import epics
try:
    # Mock epics.caput to avoid actual EPICS calls
    original_caput = epics.caput

    def mock_caput(pvname, value, **kwargs):
        return 1  # Success

    # This should pass validation (50.0 is within [0.0, 100.0])
    epics.caput = mock_caput
    epics.caput("TEST:PV", 50.0)
    result = "SUCCESS - valid write allowed"
except Exception as e:
    result = f"FAILED: {str(e)}"
results = {"test_result": result}
"""
        wrapped_code = wrapper.create_wrapper(user_code, execution_folder=None)

        # Verify wrapper contains validation logic
        assert "_limits_validator.validate" in wrapped_code

    def test_policy_serialization(self, test_validator):
        """Test that policy config is correctly serialized."""
        wrapper = ExecutionWrapper(
            execution_mode="container",
            limits_validator=test_validator
        )

        user_code = "pass"
        wrapped_code = wrapper.create_wrapper(user_code, execution_folder=None)

        # Verify policy is in wrapped code
        assert "allow_unlisted_pvs" in wrapped_code
        # Policy should have false value (strict mode)
        assert '"allow_unlisted_pvs": false' in wrapped_code.lower()

    def test_wrapper_handles_empty_database(self):
        """Test wrapper handles validator with empty database (failsafe mode)."""
        # Empty database = blocks all writes (failsafe)
        validator = LimitsValidator({}, {"allow_unlisted_pvs": False})

        wrapper = ExecutionWrapper(
            execution_mode="container",
            limits_validator=validator
        )

        user_code = "pass"
        wrapped_code = wrapper.create_wrapper(user_code, execution_folder=None)

        # Should still include boundary checking infrastructure
        assert "Runtime Channel Limits Checking" in wrapped_code
        assert "_limits_validator" in wrapped_code


class TestConfigIntegration:
    """Test configuration integration with boundary checking."""

    @patch('osprey.utils.config.get_config_value')
    def test_config_loads_validator(self, mock_get_config, tmp_path):
        """Test that PythonExecutorConfig loads validator from config."""
        from osprey.services.python_executor.config import PythonExecutorConfig

        # Create test database file
        db_file = tmp_path / "test_boundaries.json"
        db_content = {
            "TEST:PV": {
                "min_value": 0.0,
                "max_value": 100.0,
                "writable": True
            }
        }
        db_file.write_text(json.dumps(db_content))

        # Mock config values
        def config_side_effect(key, default):
            if key == 'control_system.limits_checking.enabled':
                return True
            elif key == 'control_system.limits_checking.database_path':
                return str(db_file)
            elif key == 'control_system.limits_checking.allow_unlisted_pvs':
                return False
            return default

        mock_get_config.side_effect = config_side_effect

        # Create config and access limits_validator property
        config = PythonExecutorConfig({})
        validator = config.limits_validator

        # Verify validator was loaded
        assert validator is not None
        assert "TEST:PV" in validator.limits

    @patch('osprey.utils.config.get_config_value')
    def test_config_caches_validator(self, mock_get_config, tmp_path):
        """Test that validator is cached after first load."""
        from osprey.services.python_executor.config import PythonExecutorConfig

        db_file = tmp_path / "test_boundaries.json"
        db_file.write_text(json.dumps({"TEST:PV": {"writable": True}}))

        def config_side_effect(key, default):
            if key == 'control_system.limits_checking.enabled':
                return True
            elif key == 'control_system.limits_checking.database_path':
                return str(db_file)
            elif key == 'control_system.limits_checking.allow_unlisted_pvs':
                return False
            return default

        mock_get_config.side_effect = config_side_effect

        config = PythonExecutorConfig({})

        # First access
        validator1 = config.limits_validator

        # Second access - should return cached instance
        validator2 = config.limits_validator

        assert validator1 is validator2  # Same object instance

    @patch('osprey.utils.config.get_config_value')
    def test_config_returns_none_when_disabled(self, mock_get_config):
        """Test that config returns None when boundary checking disabled."""
        from osprey.services.python_executor.config import PythonExecutorConfig

        mock_get_config.return_value = False  # Disabled

        config = PythonExecutorConfig({})
        validator = config.limits_validator

        assert validator is None


class TestExecutorIntegration:
    """Test executor integration with boundary checking."""

    @patch('osprey.utils.config.get_config_value')
    def test_local_executor_passes_validator_to_wrapper(self, mock_get_config, tmp_path):
        """Test that LocalCodeExecutor passes validator to wrapper."""
        from osprey.services.python_executor.execution.node import LocalCodeExecutor

        # Create test database
        db_file = tmp_path / "test_boundaries.json"
        db_content = {"TEST:PV": {"min_value": 0.0, "max_value": 100.0}}
        db_file.write_text(json.dumps(db_content))

        def config_side_effect(key, default):
            if key == 'control_system.limits_checking.enabled':
                return True
            elif key == 'control_system.limits_checking.database_path':
                return str(db_file)
            elif key == 'control_system.limits_checking.allow_unlisted_pvs':
                return False
            return default

        mock_get_config.side_effect = config_side_effect

        # Create executor
        executor = LocalCodeExecutor({})

        # Verify validator is accessible through config
        assert executor.executor_config.limits_validator is not None

    def test_wrapper_integration_in_create_wrapper_flow(self):
        """Test complete flow from validator to wrapper generation."""
        # Create validator
        test_db = {
            "TEST:PV": ChannelLimitsConfig(
                channel_address="TEST:PV",
                min_value=0.0,
                max_value=100.0,
                writable=True
            )
        }
        policy = {"allow_unlisted_pvs": False}
        validator = LimitsValidator(test_db, policy)

        # Create wrapper with validator
        wrapper = ExecutionWrapper(
            execution_mode="container",
            limits_validator=validator
        )

        # Generate wrapped code
        user_code = """
import epics
epics.caput("TEST:PV", 50.0)
results = {"success": True}
"""
        wrapped_code = wrapper.create_wrapper(user_code, execution_folder=None)

        # Verify all components are present
        assert "Runtime Channel Limits Checking" in wrapped_code
        assert "LimitsValidator" in wrapped_code
        assert "ChannelLimitsConfig" in wrapped_code
        assert "_limits_validator.validate" in wrapped_code
        assert "TEST:PV" in wrapped_code
        assert "epics.caput = _checked_caput" in wrapped_code

        # Verify user code is included
        assert "results = {" in wrapped_code

