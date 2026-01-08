"""Tests for /chat and /exit command handlers.

This module tests the CLI commands for entering and exiting
direct chat mode with capabilities.
"""


class TestChatCommand:
    """Tests for /chat command handler function."""

    def test_chat_command_enters_direct_mode(self, monkeypatch):
        """Chat handler with capability name should return session_state update."""
        from unittest.mock import MagicMock

        from osprey.commands import CommandContext, get_command_registry

        # Get the global command registry (already has commands registered)
        registry = get_command_registry()

        # Mock get_registry at the osprey.registry level (imported locally in handler)
        mock_reg_instance = MagicMock()

        # Setup capability with direct_chat_enabled
        mock_cap_class = MagicMock()
        mock_cap_class.direct_chat_enabled = True
        mock_reg_instance.capabilities = {
            "weather_mcp": {
                "class": mock_cap_class,
                "description": "Weather operations",
            }
        }

        monkeypatch.setattr("osprey.registry.get_registry", lambda: mock_reg_instance)

        mock_console = MagicMock()
        context = CommandContext(interface_type="cli", console=mock_console)

        cmd = registry.get_command("chat")
        result = cmd.handler("weather_mcp", context)

        # Should return session_state update
        assert isinstance(result, dict)
        assert "session_state" in result
        assert result["session_state"]["direct_chat_capability"] == "weather_mcp"

    def test_chat_command_lists_capabilities_when_no_args(self, monkeypatch):
        """Chat handler without args should list available direct-chat capabilities."""
        from unittest.mock import MagicMock

        from osprey.commands import CommandContext, CommandResult, get_command_registry

        registry = get_command_registry()

        # Mock registry with capabilities
        mock_reg_instance = MagicMock()

        # Create mock capability instances
        mock_cap1 = MagicMock()
        mock_cap1.name = "weather_mcp"
        mock_cap1.description = "Weather operations"
        mock_cap1.direct_chat_enabled = True

        mock_cap2 = MagicMock()
        mock_cap2.name = "normal_cap"
        mock_cap2.description = "Normal capability"
        mock_cap2.direct_chat_enabled = False

        mock_reg_instance.get_all_capabilities = MagicMock(return_value=[mock_cap1, mock_cap2])

        monkeypatch.setattr("osprey.registry.get_registry", lambda: mock_reg_instance)

        mock_console = MagicMock()
        context = CommandContext(interface_type="cli", console=mock_console)

        cmd = registry.get_command("chat")
        result = cmd.handler("", context)

        # Should return HANDLED (list was displayed)
        assert result == CommandResult.HANDLED

        # Should have printed the table
        assert mock_console.print.called


class TestExitCommand:
    """Tests for /exit command in direct chat mode."""

    def test_exit_clears_direct_chat_mode(self):
        """Exit handler should clear direct chat mode if active."""
        from unittest.mock import MagicMock

        from langchain_core.messages import SystemMessage

        from osprey.commands import CommandContext, get_command_registry

        registry = get_command_registry()

        mock_console = MagicMock()

        # Context with agent_state in direct chat mode
        context = CommandContext(
            interface_type="cli",
            console=mock_console,
            agent_state={"session_state": {"direct_chat_capability": "weather_mcp"}},
        )

        cmd = registry.get_command("exit")
        result = cmd.handler("", context)

        # Should return session_state update to clear direct chat
        assert isinstance(result, dict)
        assert "session_state" in result
        assert result["session_state"]["direct_chat_capability"] is None

        # Should include transition marker message for task extraction context
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], SystemMessage)
        assert "weather_mcp" in result["messages"][0].content
        assert "direct chat session" in result["messages"][0].content.lower()

    def test_exit_terminates_without_direct_chat(self):
        """Exit handler should terminate session when not in direct chat mode."""
        from unittest.mock import MagicMock

        from osprey.commands import CommandContext, CommandResult, get_command_registry

        registry = get_command_registry()

        mock_console = MagicMock()

        # Context without direct chat mode
        context = CommandContext(interface_type="cli", console=mock_console, agent_state={})

        cmd = registry.get_command("exit")
        result = cmd.handler("", context)

        # Should return EXIT
        assert result == CommandResult.EXIT

    def test_exit_terminates_with_empty_session_state(self):
        """Exit handler should terminate when session_state has no direct_chat_capability."""
        from unittest.mock import MagicMock

        from osprey.commands import CommandContext, CommandResult, get_command_registry

        registry = get_command_registry()

        mock_console = MagicMock()

        # Context with session_state but no direct chat
        context = CommandContext(
            interface_type="cli",
            console=mock_console,
            agent_state={"session_state": {"some_other_key": "value"}},
        )

        cmd = registry.get_command("exit")
        result = cmd.handler("", context)

        # Should return EXIT (not in direct chat mode)
        assert result == CommandResult.EXIT
