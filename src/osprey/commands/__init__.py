"""Centralized Slash Command System for Osprey Framework.

This module provides a unified, extensible slash command system that handles
all command types across different interfaces (CLI, OpenWebUI, etc.) and
execution contexts (UI operations, agent control, service-specific).

Architecture:
    - Unified command registry with categorization
    - Pluggable command handlers with context awareness
    - Consistent parsing and validation across all interfaces
    - Rich autocompletion and help system
    - Future-proof extensibility for custom commands

Usage:
    from osprey.commands import get_command_registry, execute_command

    registry = get_command_registry()
    result = await execute_command("/task:off", context)
"""

from .categories import (
    register_agent_control_commands,
    register_cli_commands,
    register_service_commands,
)
from .registry import (
    CommandRegistry,
    execute_command,
    get_command_registry,
    parse_command_line,
    register_command,
)
from .types import Command, CommandCategory, CommandContext, CommandHandler, CommandResult

__all__ = [
    # Core system
    "CommandRegistry",
    "get_command_registry",
    "register_command",
    "execute_command",
    "parse_command_line",
    # Types
    "Command",
    "CommandResult",
    "CommandCategory",
    "CommandContext",
    "CommandHandler",
    # Command categories
    "register_cli_commands",
    "register_agent_control_commands",
    "register_service_commands",
]
