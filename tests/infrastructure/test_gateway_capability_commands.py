"""
Tests for Gateway capability command parsing and storage.

Tests the Gateway's ability to detect unregistered slash commands
and store them in state for downstream capability handling.
"""

import pytest

from osprey.commands import CommandContext
from osprey.infrastructure.gateway import Gateway


@pytest.fixture
def gateway():
    """Create a fresh Gateway instance for testing."""
    return Gateway()


@pytest.fixture
def context():
    """Create a minimal command context for testing."""
    return CommandContext(interface_type="gateway", config={}, gateway=None, agent_state=None)


class TestGatewayCapabilityCommands:
    """Tests for Gateway's capability command handling."""

    @pytest.mark.asyncio
    async def test_unregistered_command_stored_in_capability_commands(self, gateway):
        """Unregistered commands are stored in capability_commands dict."""
        # /beam is not a registered command
        result = await gateway._process_slash_commands("/beam:diagnostic", config={})

        agent_control_changes, remaining_message, exit_requested, capability_commands = result

        assert capability_commands == {"beam": "diagnostic"}
        assert remaining_message == ""
        assert not exit_requested

    @pytest.mark.asyncio
    async def test_flag_command_stored_as_true(self, gateway):
        """Unregistered commands without values are stored as True."""
        # /verbose is not a registered command
        result = await gateway._process_slash_commands("/verbose", config={})

        _, _, _, capability_commands = result

        assert capability_commands == {"verbose": True}

    @pytest.mark.asyncio
    async def test_multiple_capability_commands(self, gateway):
        """Multiple unregistered commands are all stored."""
        result = await gateway._process_slash_commands(
            "/beam:diagnostic /verbose /format:json", config={}
        )

        _, _, _, capability_commands = result

        assert capability_commands == {
            "beam": "diagnostic",
            "verbose": True,
            "format": "json",
        }

    @pytest.mark.asyncio
    async def test_registered_commands_not_in_capability_commands(self, gateway):
        """Registered commands (like /help) are NOT stored in capability_commands."""
        # /help is a registered command
        result = await gateway._process_slash_commands("/help", config={})

        _, _, _, capability_commands = result

        # capability_commands should be empty - /help was handled by registry
        assert capability_commands == {}

    @pytest.mark.asyncio
    async def test_mixed_registered_and_unregistered_commands(self, gateway):
        """Mix of registered and unregistered commands are handled correctly."""
        # /help is registered, /beam is not
        result = await gateway._process_slash_commands("/help /beam:production", config={})

        agent_control_changes, remaining_message, exit_requested, capability_commands = result

        # Only unregistered command should be in capability_commands
        assert capability_commands == {"beam": "production"}

    @pytest.mark.asyncio
    async def test_no_commands_returns_empty(self, gateway):
        """Non-command input returns empty capability_commands."""
        result = await gateway._process_slash_commands("regular message", config={})

        _, remaining_message, _, capability_commands = result

        assert capability_commands == {}
        assert remaining_message == "regular message"

    @pytest.mark.asyncio
    async def test_command_with_remaining_text(self, gateway):
        """Commands followed by text preserve the remaining text."""
        result = await gateway._process_slash_commands(
            "/beam:diagnostic show me the data", config={}
        )

        _, remaining_message, _, capability_commands = result

        assert capability_commands == {"beam": "diagnostic"}
        assert remaining_message == "show me the data"


class TestGatewayProcessMessageFlow:
    """Tests for capability commands flowing through process_message."""

    @pytest.mark.asyncio
    async def test_capability_commands_in_fresh_state(self, gateway):
        """Capability commands flow to fresh_state via process_message."""
        # Process a message with capability commands (no graph needed for basic test)
        result = await gateway.process_message("/beam:diagnostic hello")

        assert result.agent_state is not None
        assert result.agent_state.get("_capability_slash_commands") == {"beam": "diagnostic"}

    @pytest.mark.asyncio
    async def test_no_capability_commands_when_none_provided(self, gateway):
        """Fresh state has empty capability commands when none provided."""
        result = await gateway.process_message("hello world")

        assert result.agent_state is not None
        # Field should be empty dict (initialized by StateManager.create_fresh_state)
        assert result.agent_state.get("_capability_slash_commands") == {}
