"""TUI Widget components."""

from osprey.interfaces.tui.widgets.blocks import (
    ClassificationBlock,
    ExecutionStepBlock,
    LogsLink,
    OrchestrationBlock,
    ProcessingBlock,
    ProcessingStep,
    TaskExtractionBlock,
    TaskExtractionStep,
)
from osprey.interfaces.tui.widgets.chat_display import ChatDisplay
from osprey.interfaces.tui.widgets.command_palette import CommandPalette
from osprey.interfaces.tui.widgets.theme_picker import ThemePicker
from osprey.interfaces.tui.widgets.debug import DebugBlock
from osprey.interfaces.tui.widgets.log_viewer import LogViewer
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
    "LogsLink",
    "LogViewer",
    "ProcessingBlock",
    "ProcessingStep",
    "TaskExtractionBlock",
    "TaskExtractionStep",
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
