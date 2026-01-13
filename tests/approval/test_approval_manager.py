"""Tests for ApprovalManager and get_approval_manager function.

Tests cover initialization scenarios including missing optional config fields.
See GitHub issue #79: get_approval_manager crashes on missing global_mode key.
"""

from unittest.mock import patch

import pytest

import osprey.approval.approval_manager as approval_module
from osprey.approval.approval_manager import ApprovalManager, get_approval_manager


@pytest.fixture(autouse=True)
def reset_approval_singleton():
    """Reset approval manager singleton before and after each test."""
    approval_module._approval_manager = None
    yield
    approval_module._approval_manager = None


class TestGetApprovalManagerWithDefaults:
    """Tests for get_approval_manager handling of missing optional config fields.

    Regression tests for GitHub issue #79.
    """

    def test_get_approval_manager_without_global_mode(self):
        """get_approval_manager should not crash when global_mode is omitted.

        The global_mode field is optional and should default to 'selective'.
        The logger should use the initialized object's properties, not raw config.

        Regression test for: https://github.com/als-apg/osprey/issues/79
        """
        # Config without global_mode - should use default 'selective'
        minimal_config = {
            "capabilities": {
                "python_execution": {"enabled": True, "mode": "control_writes"},
                "memory": {"enabled": False},
            }
        }

        with patch("osprey.utils.config.get_config_value", return_value=minimal_config):
            # This should not raise KeyError
            manager = get_approval_manager()

            # Verify defaults were applied correctly
            assert manager.config.global_mode == "selective"
            assert manager.config.python_execution.enabled is True
            assert manager.config.memory.enabled is False

    def test_get_approval_manager_without_capabilities(self):
        """get_approval_manager should not crash when capabilities is omitted.

        The capabilities field is optional and should default to empty dict,
        which then gets populated with secure defaults.
        """
        # Config without capabilities - should use defaults
        minimal_config = {"global_mode": "selective"}

        with patch("osprey.utils.config.get_config_value", return_value=minimal_config):
            manager = get_approval_manager()

            # Verify secure defaults were applied
            assert manager.config.global_mode == "selective"
            # Secure defaults: enabled=True, mode=all_code
            assert manager.config.python_execution.enabled is True
            assert manager.config.memory.enabled is True

    def test_get_approval_manager_empty_config(self):
        """get_approval_manager should handle completely empty config with defaults."""
        empty_config = {}

        with patch("osprey.utils.config.get_config_value", return_value=empty_config):
            manager = get_approval_manager()

            # Verify all defaults were applied
            assert manager.config.global_mode == "selective"
            assert manager.config.python_execution.enabled is True
            assert manager.config.memory.enabled is True

    def test_get_approval_manager_full_config(self):
        """get_approval_manager should work with fully specified config."""
        full_config = {
            "global_mode": "all_capabilities",
            "capabilities": {
                "python_execution": {"enabled": False, "mode": "disabled"},
                "memory": {"enabled": True},
            },
        }

        with patch("osprey.utils.config.get_config_value", return_value=full_config):
            manager = get_approval_manager()

            assert manager.config.global_mode == "all_capabilities"
            assert manager.config.python_execution.enabled is False
            assert manager.config.memory.enabled is True


class TestApprovalManagerInit:
    """Tests for ApprovalManager.__init__ with various config inputs."""

    def test_init_with_minimal_config(self):
        """ApprovalManager should initialize with minimal config using defaults."""
        minimal_config = {}
        manager = ApprovalManager(minimal_config)

        assert manager.config.global_mode == "selective"

    def test_init_with_partial_capabilities(self):
        """ApprovalManager should handle partial capabilities config."""
        config = {
            "capabilities": {
                "python_execution": {"enabled": True, "mode": "control_writes"}
                # memory is missing - should use default
            }
        }
        manager = ApprovalManager(config)

        assert manager.config.python_execution.enabled is True
        assert manager.config.memory.enabled is True  # secure default
