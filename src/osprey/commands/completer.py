"""Slash-command autocompletion for prompt_toolkit interfaces."""

from collections.abc import Iterable

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import HTML

from osprey.cli.styles import get_active_theme

from .registry import get_command_registry
from .types import CommandContext


class UnifiedCommandCompleter(Completer):
    """Completer that offers slash-command suggestions filtered by interface type."""

    def __init__(self, context: CommandContext):
        self.registry = get_command_registry()
        self.context = context

    def get_completions(self, document: Document, complete_event) -> Iterable[Completion]:
        """Get completions for the current input."""
        text = document.text_before_cursor

        current_command = self._extract_current_command(text)

        if not current_command:
            return

        completions = self.registry.get_completions(current_command, self.context)

        for completion in completions:
            start_position = -len(current_command)

            cmd_name = completion[1:]

            command = self.registry.get_command(cmd_name)

            if command:
                theme = get_active_theme()

                category_colors = {
                    "cli": theme.info,  # info/blue for CLI commands
                    "agent": theme.success,  # success/green for agent commands
                    "service": theme.warning,  # warning/yellow for service commands
                    "custom": theme.accent,  # accent/pink for custom commands
                }

                color = category_colors.get(command.category.value, theme.text_primary)

                display_html = (
                    f'<completion style="fg:{color}">{completion}</completion> '
                    f'<description style="fg:{theme.text_dim}">- {command.description}</description>'
                )

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

        parts = text.split()

        for part in reversed(parts):
            if part.startswith("/"):
                return part

        if text.endswith("/"):
            return "/"

        return ""
