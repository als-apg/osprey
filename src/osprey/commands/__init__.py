"""Slash command system: registry, types, and built-in categories.

Usage::

    from osprey.commands import get_command_registry, execute_command

    registry = get_command_registry()
    result = await execute_command("/task:off", context)
"""

from .categories import (
    register_agent_control_commands,
    register_cli_commands,
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
    "CommandRegistry",
    "get_command_registry",
    "register_command",
    "execute_command",
    "parse_command_line",
    "Command",
    "CommandResult",
    "CommandCategory",
    "CommandContext",
    "CommandHandler",
    "register_cli_commands",
    "register_agent_control_commands",
]
