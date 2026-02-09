"""
Safety tests for channel_write capability.

Related to: CRITICAL_CHANNEL_WRITE_SAFETY_BYPASS.md
"""

from unittest.mock import patch

import pytest

from osprey.services.python_executor.exceptions import ChannelLimitsViolationError


# Unit tests for boundary validator
class TestLimitsValidator:
    """Unit tests for LimitsValidator class."""

    def test_validator_from_config_disabled(self):
        """Test that from_config returns None when disabled."""
        from osprey.services.python_executor.execution.limits_validator import LimitsValidator

        with patch("osprey.utils.config.get_config_value") as mock_config:
            mock_config.return_value = False  # enabled=False
            validator = LimitsValidator.from_config()
            assert validator is None

    def test_validator_max_exceeded(self):
        """Test that max value violations are caught."""
        from osprey.services.python_executor.execution.limits_validator import (
            ChannelLimitsConfig,
            LimitsValidator,
        )

        limits_db = {
            "TEST:PV": ChannelLimitsConfig(
                channel_address="TEST:PV", min_value=0.0, max_value=100.0, writable=True
            )
        }
        policy = {"allow_unlisted_channels": False}
        validator = LimitsValidator(limits_db, policy)

        # Test exceeding max
        with pytest.raises(ChannelLimitsViolationError) as exc_info:
            validator.validate("TEST:PV", 150.0)

        assert "above maximum" in exc_info.value.violation_reason
        assert exc_info.value.violation_type == "MAX_EXCEEDED"

    def test_validator_min_exceeded(self):
        """Test that min value violations are caught."""
        from osprey.services.python_executor.execution.limits_validator import (
            ChannelLimitsConfig,
            LimitsValidator,
        )

        limits_db = {
            "TEST:PV": ChannelLimitsConfig(
                channel_address="TEST:PV", min_value=0.0, max_value=100.0, writable=True
            )
        }
        policy = {"allow_unlisted_channels": False}
        validator = LimitsValidator(limits_db, policy)

        # Test below min
        with pytest.raises(ChannelLimitsViolationError) as exc_info:
            validator.validate("TEST:PV", -10.0)

        assert "below minimum" in exc_info.value.violation_reason
        assert exc_info.value.violation_type == "MIN_EXCEEDED"

    def test_validator_read_only_pv(self):
        """Test that read-only PVs are blocked."""
        from osprey.services.python_executor.execution.limits_validator import (
            ChannelLimitsConfig,
            LimitsValidator,
        )

        limits_db = {
            "TEST:PV": ChannelLimitsConfig(
                channel_address="TEST:PV",
                min_value=0.0,
                max_value=100.0,
                writable=False,  # Read-only
            )
        }
        policy = {"allow_unlisted_channels": False}
        validator = LimitsValidator(limits_db, policy)

        with pytest.raises(ChannelLimitsViolationError) as exc_info:
            validator.validate("TEST:PV", 50.0)

        assert "read-only" in exc_info.value.violation_reason
        assert exc_info.value.violation_type == "READ_ONLY_CHANNEL"

    def test_validator_unlisted_pv_strict_mode(self):
        """Test that unlisted PVs are blocked in strict mode."""
        from osprey.services.python_executor.execution.limits_validator import (
            LimitsValidator,
        )

        limits_db = {}  # Empty - no PVs listed
        policy = {"allow_unlisted_channels": False}  # Strict mode
        validator = LimitsValidator(limits_db, policy)

        with pytest.raises(ChannelLimitsViolationError) as exc_info:
            validator.validate("UNLISTED:PV", 50.0)

        assert "not in limits database" in exc_info.value.violation_reason
        assert exc_info.value.violation_type == "UNLISTED_CHANNEL"

    def test_validator_unlisted_pv_permissive_mode(self):
        """Test that unlisted PVs are allowed when allow_unlisted_channels=True."""
        from osprey.services.python_executor.execution.limits_validator import (
            LimitsValidator,
        )

        limits_db = {}  # Empty - no PVs listed
        policy = {"allow_unlisted_channels": True}  # Permissive mode
        validator = LimitsValidator(limits_db, policy)

        # Should not raise
        validator.validate("UNLISTED:PV", 50.0)

    def test_validator_valid_write(self):
        """Test that valid writes pass validation."""
        from osprey.services.python_executor.execution.limits_validator import (
            ChannelLimitsConfig,
            LimitsValidator,
        )

        limits_db = {
            "TEST:PV": ChannelLimitsConfig(
                channel_address="TEST:PV", min_value=0.0, max_value=100.0, writable=True
            )
        }
        policy = {"allow_unlisted_channels": False}
        validator = LimitsValidator(limits_db, policy)

        # Should not raise
        validator.validate("TEST:PV", 50.0)


# Unit tests for approval interrupt creation
class TestChannelWriteApprovalInterrupt:
    """Unit tests for channel write approval interrupt creation."""

    def test_create_channel_write_approval_interrupt(self):
        """Test that approval interrupt data is created correctly."""
        from osprey.approval.approval_system import create_channel_write_approval_interrupt

        # Mock write operations
        class MockWriteOperation:
            def __init__(self, channel_address, value, units=None, notes=None):
                self.channel_address = channel_address
                self.value = value
                self.units = units
                self.notes = notes

        operations = [
            MockWriteOperation("TEST:PV1", 50.0, "A"),
            MockWriteOperation("TEST:PV2", 100.0, "V"),
        ]

        analysis_details = {
            "operation_count": 2,
            "channels": ["TEST:PV1", "TEST:PV2"],
            "values": [("TEST:PV1", 50.0), ("TEST:PV2", 100.0)],
            "safety_level": "high",
        }

        safety_concerns = [
            "Direct hardware write: TEST:PV1 = 50.0",
            "Direct hardware write: TEST:PV2 = 100.0",
        ]

        interrupt_data = create_channel_write_approval_interrupt(
            operations=operations,
            analysis_details=analysis_details,
            safety_concerns=safety_concerns,
            step_objective="Test write operation",
        )

        # Verify structure
        assert "user_message" in interrupt_data
        assert "resume_payload" in interrupt_data

        # Verify user message content
        assert "HUMAN APPROVAL REQUIRED" in interrupt_data["user_message"]
        assert "Test write operation" in interrupt_data["user_message"]
        assert "TEST:PV1" in interrupt_data["user_message"]
        assert "TEST:PV2" in interrupt_data["user_message"]

        # Verify resume payload
        payload = interrupt_data["resume_payload"]
        assert payload["approval_type"] == "channel_write"
        assert payload["step_objective"] == "Test write operation"
        assert len(payload["operations"]) == 2
        assert payload["operations"][0]["channel_address"] == "TEST:PV1"
        assert payload["operations"][0]["value"] == 50.0
        assert payload["safety_concerns"] == safety_concerns
