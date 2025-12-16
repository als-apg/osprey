"""TUI Widget components."""

from osprey.interfaces.tui.widgets.blocks import (
    ClassificationBlock,
    ExecutionStepBlock,
    OrchestrationBlock,
    ProcessingBlock,
    TaskExtractionBlock,
)
from osprey.interfaces.tui.widgets.chat_display import ChatDisplay
from osprey.interfaces.tui.widgets.command_palette import CommandPalette
from osprey.interfaces.tui.widgets.theme_picker import ThemePicker
from osprey.interfaces.tui.widgets.debug import DebugBlock
from osprey.interfaces.tui.widgets.input import (
    ChatInput,
    CommandDropdown,
    StatusPanel,
)
from osprey.interfaces.tui.widgets.messages import ChatMessage
from osprey.interfaces.tui.widgets.welcome import WelcomeBanner, WelcomeScreen

__all__ = [
    "ChatMessage",
    "CommandPalette",
    "DebugBlock",
    "ProcessingBlock",
    "TaskExtractionBlock",
    "ClassificationBlock",
    "OrchestrationBlock",
    "ExecutionStepBlock",
    "ChatDisplay",
    "ChatInput",
    "StatusPanel",
    "CommandDropdown",
    "ThemePicker",
    "WelcomeBanner",
    "WelcomeScreen",
]
