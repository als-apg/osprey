"""TUI Widget components."""

from osprey.interfaces.tui.widgets.artifact_viewer import ArtifactViewer
from osprey.interfaces.tui.widgets.artifacts import ArtifactGallery, ArtifactItem
from osprey.interfaces.tui.widgets.blocks import (
    ClassificationBlock,
    ClassificationStep,
    ExecutionStep,
    LogsLink,
    OrchestrationBlock,
    OrchestrationStep,
    ProcessingBlock,
    ProcessingStep,
    PromptLink,
    ResponseLink,
    TaskExtractionBlock,
    TaskExtractionStep,
    TodoItem,
    TodoList,
    TodoUpdateStep,
    WrappedLabel,
    WrappedStatic,
)
from osprey.interfaces.tui.widgets.chat_display import ChatDisplay
from osprey.interfaces.tui.widgets.command_palette import CommandPalette
from osprey.interfaces.tui.widgets.content_viewer import ContentViewer
from osprey.interfaces.tui.widgets.debug import DebugBlock
from osprey.interfaces.tui.widgets.input import (
    ChatInput,
    CommandDropdown,
    StatusPanel,
)
from osprey.interfaces.tui.widgets.log_viewer import LogViewer
from osprey.interfaces.tui.widgets.messages import ChatMessage
from osprey.interfaces.tui.widgets.theme_picker import ThemePicker
from osprey.interfaces.tui.widgets.welcome import WelcomeBanner, WelcomeScreen

__all__ = [
    # Artifact widgets
    "ArtifactGallery",
    "ArtifactItem",
    "ArtifactViewer",
    # Chat and display
    "ChatMessage",
    "ChatDisplay",
    "ChatInput",
    # Modals
    "CommandPalette",
    "ContentViewer",
    "LogViewer",
    "ThemePicker",
    # Processing blocks and steps
    "DebugBlock",
    "LogsLink",
    "ProcessingBlock",
    "ProcessingStep",
    "PromptLink",
    "ResponseLink",
    "TaskExtractionBlock",
    "TaskExtractionStep",
    "ClassificationBlock",
    "ClassificationStep",
    "OrchestrationBlock",
    "OrchestrationStep",
    "TodoItem",
    "TodoList",
    "TodoUpdateStep",
    "ExecutionStep",
    # Input widgets
    "StatusPanel",
    "CommandDropdown",
    # Welcome screen
    "WelcomeBanner",
    "WelcomeScreen",
    # Utilities
    "WrappedLabel",
    "WrappedStatic",
]
