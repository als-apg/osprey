"""Type definitions for the slash command system."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class CommandCategory(Enum):
    """Organizational categories for slash commands."""

    CLI = "cli"  # Interface/UI commands (help, clear, exit)
    AGENT_CONTROL = "agent"  # Agent behavior control (planning, approval, debug)
    SERVICE = "service"  # Service-specific commands (logs, metrics)
    CUSTOM = "custom"  # User-defined custom commands


class CommandResult(Enum):
    """Return values indicating how the interface should proceed after command execution."""

    CONTINUE = "continue"  # Continue normal processing
    HANDLED = "handled"  # Command was handled, stop processing
    EXIT = "exit"  # Exit the current interface
    AGENT_STATE_CHANGED = "agent_state_changed"  # Agent control state was modified


@dataclass
class CommandContext:
    """Execution context passed to command handlers.

    Fields are populated based on the calling interface.
    """

    interface_type: str = "unknown"
    user_id: str | None = None
    session_id: str | None = None

    cli_instance: Any | None = None
    console: Any | None = None

    agent_state: dict[str, Any] | None = None
    gateway: Any | None = None
    config: dict[str, Any] | None = None

    service_instance: Any | None = None

    extra: dict[str, Any] = field(default_factory=dict)


class CommandHandler(Protocol):
    """Callable that handles a slash command."""

    async def __call__(self, args: str, context: CommandContext) -> CommandResult | dict[str, Any]:
        """Execute the command.

        Args:
            args: Command arguments as string
            context: Execution context

        Returns:
            CommandResult or dict of state changes for agent control commands
        """
        ...


@dataclass
class Command:
    """Specification for a slash command: name, handler, constraints, and metadata."""

    name: str
    category: CommandCategory
    description: str
    handler: CommandHandler

    aliases: list[str] = field(default_factory=list)
    help_text: str | None = None
    syntax: str | None = None

    requires_args: bool = False
    valid_options: list[str] | None = None
    interface_restrictions: list[str] | None = None  # ["cli", "openwebui"]

    # When True, command is routed through Gateway for consistent state management
    # across interfaces (CLI, OpenWebUI, API). Typically set for commands that modify
    # agent state (/chat, /planning, /exit in direct-chat) vs. local UI commands (/help).
    gateway_handled: bool = False

    hidden: bool = False
    deprecated: bool = False

    def __post_init__(self):
        """Auto-generate help_text and syntax from fields if not set."""
        if self.help_text is None:
            self.help_text = self.description

        if self.syntax is None:
            if self.valid_options:
                options = "|".join(self.valid_options)
                self.syntax = (
                    f"/{self.name}:{options}" if self.requires_args else f"/{self.name}[:{options}]"
                )
            else:
                self.syntax = f"/{self.name}:<value>" if self.requires_args else f"/{self.name}"

    def is_valid_for_interface(self, interface_type: str) -> bool:
        """Return True if command is available in the given interface type."""
        if self.interface_restrictions is None:
            return True
        return interface_type in self.interface_restrictions

    def validate_option(self, option: str | None) -> bool:
        """Return True if option satisfies requires_args and valid_options constraints."""
        if self.requires_args and option is None:
            return False
        if self.valid_options and option is not None:
            return option in self.valid_options
        return True


@dataclass
class ParsedCommand:
    """Result of parsing a command line."""

    command_name: str
    option: str | None = None
    remaining_text: str = ""
    is_valid: bool = True
    error_message: str | None = None


class CommandExecutionError(Exception):
    """Exception raised during command execution."""

    def __init__(self, message: str, command_name: str, suggestion: str | None = None):
        super().__init__(message)
        self.command_name = command_name
        self.suggestion = suggestion
