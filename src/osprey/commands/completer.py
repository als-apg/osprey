"""
Command Completion System for Interactive Interfaces

This module provides intelligent autocompletion for slash commands in interactive
interfaces using prompt_toolkit integration. The completion system offers context-aware
suggestions, rich formatting, and seamless integration with the centralized command registry.

Features:
    - Context-aware command completion based on interface type
    - Rich formatted suggestions with command descriptions
    - Intelligent argument completion for command options
    - Category-based filtering for relevant commands
    - Real-time completion updates as user types

The completion system integrates with CLI interfaces to provide a modern, IDE-like
experience for command discovery and usage.
"""

from collections.abc import Iterable

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import HTML

from osprey.cli.styles import get_active_theme

from .registry import get_command_registry
from .types import CommandContext


class UnifiedCommandCompleter(Completer):
    """Unified completer for all slash commands with rich formatting and context awareness.

    The UnifiedCommandCompleter provides intelligent autocompletion for slash commands
    in interactive interfaces. It integrates with the centralized command registry to
    offer context-aware suggestions, rich formatting, and real-time completion updates
    based on the current interface type and user context.

    Key Features:
        - Context-aware filtering based on interface type and user permissions
        - Rich formatted completions with command descriptions and syntax hints
        - Intelligent argument completion for command options and parameters
        - Category-based organization for improved command discovery
        - Real-time completion updates with minimal latency

    :param context: Command execution context for filtering and personalization
    :type context: CommandContext
    :param registry: Command registry instance for command lookup
    :type registry: CommandRegistry

    .. note::
       The completer automatically filters commands based on the interface type
       specified in the CommandContext to show only relevant commands.

    Examples:
        CLI integration::

            context = CommandContext(interface_type="cli", console=console)
            completer = UnifiedCommandCompleter(context)

            # Use with prompt_toolkit
            session = PromptSession(completer=completer)
            user_input = session.prompt("You: ")

        Custom interface integration::

            context = CommandContext(
                interface_type="custom",
                user_id="user123",
                extra={"permissions": ["admin"]}
            )
            completer = UnifiedCommandCompleter(context)
    """

    def __init__(self, context: CommandContext):
        self.registry = get_command_registry()
        self.context = context

    def get_completions(self, document: Document, complete_event) -> Iterable[Completion]:
        """Get completions for the current input."""
        text = document.text_before_cursor

        # Find the current command being typed (could be after other commands)
        current_command = self._extract_current_command(text)

        if not current_command:
            return

        # Get matching commands from registry
        completions = self.registry.get_completions(current_command, self.context)

        for completion in completions:
            # Calculate how much to replace (only the current command part)
            start_position = -len(current_command)

            # Get the command for description and styling
            cmd_name = completion[1:]  # Remove leading /
            command = self.registry.get_command(cmd_name)

            if command:
                # Get colors from active theme
                theme = get_active_theme()

                # Map categories to theme colors
                category_colors = {
                    "cli": theme.info,  # info/blue for CLI commands
                    "agent": theme.success,  # success/green for agent commands
                    "service": theme.warning,  # warning/yellow for service commands
                    "custom": theme.accent,  # accent/pink for custom commands
                }

                color = category_colors.get(command.category.value, theme.text_primary)

                # Use theme's dim text color for descriptions
                display_html = (
                    f'<completion style="fg:{color}">{completion}</completion> '
                    f'<description style="fg:{theme.text_dim}">- {command.description}</description>'
                )

                # Add syntax hint using theme's dim text color
                if command.valid_options:
                    options_hint = f" [{'/'.join(command.valid_options[:2])}{'...' if len(command.valid_options) > 2 else ''}]"
                    display_html += (
                        f'<syntax style="fg:{theme.text_dim} italic">{options_hint}</syntax>'
                    )

                yield Completion(
                    text=completion,
                    start_position=start_position,
                    display=HTML(display_html),
                    style="class:completion",
                )
            else:
                # Fallback for commands without full metadata
                yield Completion(
                    text=completion,
                    start_position=start_position,
                    display=HTML(f"<completion>{completion}</completion>"),
                    style="class:completion",
                )

    def _extract_current_command(self, text: str) -> str:
        """Extract the current command being typed from the full text.

        Handles cases like:
        - "/help" -> "/help"
        - "/task:off /plan" -> "/plan"
        - "/task:off /planning:o" -> "/planning:o"
        - "some text /help" -> "/help"
        """
        if not text:
            return ""

        # Split by spaces and find the last part that starts with /
        parts = text.split()

        for part in reversed(parts):
            if part.startswith("/"):
                return part

        # If no slash command found, check if we're in the middle of typing one
        # This handles cases where cursor is right after a /
        if text.endswith("/"):
            return "/"

        return ""
