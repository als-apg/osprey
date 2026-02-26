"""Command registry: stores, validates, and executes slash commands."""

import asyncio
import re
from typing import Any

from osprey.cli.styles import Styles, console

from .types import (
    Command,
    CommandCategory,
    CommandContext,
    CommandExecutionError,
    CommandResult,
    ParsedCommand,
)


class CommandRegistry:
    """Central registry mapping command names/aliases to Command objects.

    Singleton via get_command_registry().
    """

    def __init__(self):
        self.commands: dict[str, Command] = {}
        self.aliases: dict[str, str] = {}  # alias -> command_name mapping
        self.console = console

        self._register_default_commands()

    def register(self, command: Command) -> None:
        """Register a command. Raises ValueError on name/alias conflicts."""
        if not command.name:
            raise ValueError("Command name cannot be empty")

        if command.name in self.commands:
            raise ValueError(f"Command '{command.name}' already registered")

        self.commands[command.name] = command

        for alias in command.aliases:
            if alias in self.aliases or alias in self.commands:
                raise ValueError(f"Alias '{alias}' conflicts with existing command")
            self.aliases[alias] = command.name

    def get_command(self, name: str) -> Command | None:
        """Get a command by name or alias."""
        name = name.lstrip("/")

        if name in self.commands:
            return self.commands[name]

        if name in self.aliases:
            return self.commands[self.aliases[name]]

        return None

    def get_commands_by_category(self, category: CommandCategory) -> list[Command]:
        """Get all commands in a specific category."""
        return [cmd for cmd in self.commands.values() if cmd.category == category]

    def get_all_commands(self, include_hidden: bool = False) -> list[Command]:
        """Get all registered commands."""
        commands = list(self.commands.values())
        if not include_hidden:
            commands = [cmd for cmd in commands if not cmd.hidden]
        return sorted(commands, key=lambda x: (x.category.value, x.name))

    def get_completions(self, prefix: str, context: CommandContext | None = None) -> list[str]:
        """Get command completions for a given prefix."""
        prefix = prefix.lstrip("/")

        if not prefix:
            # Return all commands available for this interface
            commands = self.get_all_commands()
            if context and context.interface_type:
                commands = [
                    cmd for cmd in commands if cmd.is_valid_for_interface(context.interface_type)
                ]
            return [f"/{cmd.name}" for cmd in commands]

        matches = []
        for cmd_name, cmd in self.commands.items():
            # Check if command is valid for this interface
            if (
                context
                and context.interface_type
                and not cmd.is_valid_for_interface(context.interface_type)
            ):
                continue

            if cmd_name.startswith(prefix) and not cmd.hidden:
                matches.append(f"/{cmd_name}")

        for alias, cmd_name in self.aliases.items():
            cmd = self.commands[cmd_name]
            if (
                context
                and context.interface_type
                and not cmd.is_valid_for_interface(context.interface_type)
            ):
                continue

            if alias.startswith(prefix) and not cmd.hidden:
                matches.append(f"/{alias}")

        return sorted(set(matches))

    async def execute(
        self, command_line: str, context: CommandContext
    ) -> CommandResult | dict[str, Any]:
        """Execute a command from a command line."""
        parsed = parse_command_line(command_line)

        if not parsed.is_valid:
            self.console.print(f"❌ {parsed.error_message}", style=Styles.ERROR)
            return CommandResult.HANDLED

        command = self.get_command(parsed.command_name)
        if not command:
            self.console.print(f"❌ Unknown command: /{parsed.command_name}", style=Styles.ERROR)
            self.console.print("💡 Type /help to see available commands", style=Styles.DIM)
            return CommandResult.HANDLED

        # Validate interface restrictions
        if not command.is_valid_for_interface(context.interface_type):
            self.console.print(
                f"❌ Command /{command.name} not available in {context.interface_type}",
                style=Styles.ERROR,
            )
            return CommandResult.HANDLED

        # Validate options
        if not command.validate_option(parsed.option):
            if command.requires_args:
                self.console.print(
                    f"❌ Command /{command.name} requires an argument", style=Styles.ERROR
                )
            elif command.valid_options:
                valid_opts = ", ".join(command.valid_options)
                self.console.print(
                    f"❌ Invalid option '{parsed.option}' for /{command.name}. Valid options: {valid_opts}",
                    style=Styles.ERROR,
                )
            else:
                self.console.print(f"❌ Invalid option for /{command.name}", style=Styles.ERROR)
            return CommandResult.HANDLED

        try:
            if asyncio.iscoroutinefunction(command.handler):
                result = await command.handler(parsed.option or "", context)
            else:
                result = command.handler(parsed.option or "", context)

            if isinstance(result, CommandResult):
                return result
            elif isinstance(result, dict):
                return result
            else:
                return CommandResult.HANDLED

        except CommandExecutionError as e:
            self.console.print(f"❌ {e}", style=Styles.ERROR)
            if e.suggestion:
                self.console.print(f"💡 {e.suggestion}", style=Styles.DIM)
            return CommandResult.HANDLED
        except Exception as e:
            self.console.print(f"❌ Error executing /{command.name}: {e}", style=Styles.ERROR)
            return CommandResult.HANDLED

    def _register_default_commands(self):
        """Register built-in commands that are always available."""
        from .categories import (
            register_agent_control_commands,
            register_cli_commands,
        )

        register_cli_commands(self)
        register_agent_control_commands(self)


def parse_command_line(command_line: str) -> ParsedCommand:
    """Parse a command line into components.

    Supports formats:
    - /command
    - /command:option
    """
    if not command_line.startswith("/"):
        return ParsedCommand("", is_valid=False, error_message="Commands must start with /")

    line = command_line[1:]

    if not line:
        return ParsedCommand("", is_valid=False, error_message="Empty command")

    parts = line.split(" ", 1)
    first_part = parts[0]
    remaining_text = parts[1] if len(parts) > 1 else ""

    if ":" in first_part:
        match = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*):(.+)$", first_part)
        if match:
            command_name, option = match.groups()
            return ParsedCommand(
                command_name=command_name,
                option=option,
                remaining_text=remaining_text,
                is_valid=True,
            )
        else:
            return ParsedCommand(
                "", is_valid=False, error_message=f"Invalid command format: /{first_part}"
            )

    match = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)$", first_part)
    if match:
        command_name = match.group(1)
        return ParsedCommand(
            command_name=command_name, option=None, remaining_text=remaining_text, is_valid=True
        )

    return ParsedCommand("", is_valid=False, error_message=f"Invalid command format: /{first_part}")


_registry = CommandRegistry()


def get_command_registry() -> CommandRegistry:
    """Get the global command registry."""
    return _registry


def register_command(command: Command) -> None:
    """Register a command globally."""
    _registry.register(command)


async def execute_command(
    command_line: str, context: CommandContext
) -> CommandResult | dict[str, Any]:
    """Execute a command globally."""
    return await _registry.execute(command_line, context)
