"""
Unit tests for capability slash command helper functions.

Tests the slash_command() module-level function and BaseCapability.slash_command() method
for reading capability-specific slash commands from agent state.
"""

import pytest

from osprey.base.capability import BaseCapability, slash_command


class TestSlashCommandHelper:
    """Tests for the module-level slash_command() helper function."""

    def test_returns_value_for_command_with_value(self):
        """slash_command returns the string value when command has a value."""
        state = {"_capability_slash_commands": {"beam": "diagnostic"}}
        result = slash_command("beam", state)
        assert result == "diagnostic"

    def test_returns_true_for_flag_command(self):
        """slash_command returns True when command is a flag (no value)."""
        state = {"_capability_slash_commands": {"verbose": True}}
        result = slash_command("verbose", state)
        assert result is True

    def test_returns_none_for_missing_command(self):
        """slash_command returns None when command is not present."""
        state = {"_capability_slash_commands": {"beam": "diagnostic"}}
        result = slash_command("other", state)
        assert result is None

    def test_returns_none_for_empty_commands(self):
        """slash_command returns None when commands dict is empty."""
        state = {"_capability_slash_commands": {}}
        result = slash_command("beam", state)
        assert result is None

    def test_returns_none_when_field_missing(self):
        """slash_command returns None when _capability_slash_commands field is missing."""
        state = {}
        result = slash_command("beam", state)
        assert result is None

    def test_multiple_commands(self):
        """slash_command correctly retrieves from state with multiple commands."""
        state = {
            "_capability_slash_commands": {
                "beam": "production",
                "verbose": True,
                "format": "json",
            }
        }
        assert slash_command("beam", state) == "production"
        assert slash_command("verbose", state) is True
        assert slash_command("format", state) == "json"
        assert slash_command("missing", state) is None


class TestBaseCapabilitySlashCommandMethod:
    """Tests for the BaseCapability.slash_command() instance method."""

    def test_method_reads_from_state(self):
        """BaseCapability.slash_command() reads commands from injected state."""

        class TestCapability(BaseCapability):
            name = "test"
            description = "Test capability"

            async def execute(self):
                return {}

        cap = TestCapability()
        cap._state = {"_capability_slash_commands": {"beam": "diagnostic", "verbose": True}}

        assert cap.slash_command("beam") == "diagnostic"
        assert cap.slash_command("verbose") is True
        assert cap.slash_command("missing") is None

    def test_method_raises_without_state(self):
        """BaseCapability.slash_command() raises RuntimeError when state not injected."""

        class TestCapability(BaseCapability):
            name = "test"
            description = "Test capability"

            async def execute(self):
                return {}

        cap = TestCapability()
        # _state is None by default

        with pytest.raises(RuntimeError) as exc_info:
            cap.slash_command("beam")

        assert "called before state injection" in str(exc_info.value)
        assert "TestCapability" in str(exc_info.value)

    def test_method_handles_empty_commands(self):
        """BaseCapability.slash_command() handles empty commands dict."""

        class TestCapability(BaseCapability):
            name = "test"
            description = "Test capability"

            async def execute(self):
                return {}

        cap = TestCapability()
        cap._state = {"_capability_slash_commands": {}}

        assert cap.slash_command("beam") is None

    def test_method_handles_missing_field(self):
        """BaseCapability.slash_command() handles missing _capability_slash_commands field."""

        class TestCapability(BaseCapability):
            name = "test"
            description = "Test capability"

            async def execute(self):
                return {}

        cap = TestCapability()
        cap._state = {}

        assert cap.slash_command("beam") is None
