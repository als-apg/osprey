"""
Integration tests for capability slash commands.

Tests the full flow of capability slash commands from Gateway through to
capabilities reading them during execution.
"""

import pytest

from osprey.base.capability import BaseCapability
from osprey.infrastructure.gateway import Gateway
from osprey.state import StateManager


class TestCapabilityCommandsIntegration:
    """Integration tests for capability slash command flow."""

    @pytest.mark.asyncio
    async def test_capability_commands_flow_to_fresh_state(self):
        """Capability commands parsed by Gateway flow to fresh_state correctly."""
        gateway = Gateway()

        # Process message with capability commands
        result = await gateway.process_message("/beam:diagnostic /verbose check data")

        assert result.agent_state is not None
        capability_commands = result.agent_state.get("_capability_slash_commands", {})

        assert capability_commands == {
            "beam": "diagnostic",
            "verbose": True,
        }

    @pytest.mark.asyncio
    async def test_capability_can_read_commands_in_execute(self):
        """Capability can read slash commands during execute() via instance method."""

        class BeamlineCapability(BaseCapability):
            name = "beamline"
            description = "Test beamline capability"

            async def execute(self):
                # Read capability commands
                beam_mode = self.slash_command("beam")
                is_verbose = self.slash_command("verbose")

                return {
                    "beam_mode": beam_mode,
                    "is_verbose": is_verbose,
                }

        # Simulate state with capability commands
        state = StateManager.create_fresh_state("test query")
        state["_capability_slash_commands"] = {"beam": "diagnostic", "verbose": True}

        # Create capability and inject state (simulating decorator behavior)
        cap = BeamlineCapability()
        cap._state = state
        cap._step = {"context_key": "test_key"}

        # Execute and verify commands are readable
        result = await cap.execute()

        assert result["beam_mode"] == "diagnostic"
        assert result["is_verbose"] is True

    @pytest.mark.asyncio
    async def test_commands_reset_between_invocations(self):
        """Capability commands reset each conversation turn (execution-scoped)."""
        gateway = Gateway()

        # First message with commands
        result1 = await gateway.process_message("/beam:diagnostic hello")
        assert result1.agent_state["_capability_slash_commands"] == {"beam": "diagnostic"}

        # Second message without commands - simulate passing previous state
        result2 = await gateway.process_message("another query")

        # Commands should be reset (not persisted from previous turn)
        assert result2.agent_state.get("_capability_slash_commands") == {}

    @pytest.mark.asyncio
    async def test_module_level_helper_function(self):
        """Module-level slash_command() function works with state dict."""
        from osprey.base.capability import slash_command

        state = {"_capability_slash_commands": {"format": "json", "debug": True}}

        assert slash_command("format", state) == "json"
        assert slash_command("debug", state) is True
        assert slash_command("missing", state) is None

    def test_state_manager_initializes_field(self):
        """StateManager.create_fresh_state() initializes _capability_slash_commands."""
        state = StateManager.create_fresh_state("test query")

        assert "_capability_slash_commands" in state
        assert state["_capability_slash_commands"] == {}

    @pytest.mark.asyncio
    async def test_capability_commands_do_not_interfere_with_registered(self):
        """Registered commands work normally alongside capability commands."""
        gateway = Gateway()

        # /help is registered, /custom is not
        result = await gateway.process_message("/help /custom:value test message")

        capability_commands = result.agent_state.get("_capability_slash_commands", {})

        # Only unregistered command should be in capability_commands
        assert "custom" in capability_commands
        assert capability_commands["custom"] == "value"
        # /help should have been handled by command registry (not in capability_commands)
        assert "help" not in capability_commands
