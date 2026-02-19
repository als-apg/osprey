"""TUI Widget components."""

from osprey.interfaces.tui.widgets.artifact_viewer import ArtifactViewer
from osprey.interfaces.tui.widgets.artifacts import (
    ArtifactGallery,
    ArtifactItem,
    ArtifactSection,
)
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
    InfoBar,
    StatusPanel,
)
from osprey.interfaces.tui.widgets.log_viewer import LogViewer
from osprey.interfaces.tui.widgets.messages import ChatMessage
from osprey.interfaces.tui.widgets.plan_progress import PlanProgressBar
from osprey.interfaces.tui.widgets.theme_picker import ThemePicker
from osprey.interfaces.tui.widgets.welcome import WelcomeBanner, WelcomeScreen

__all__ = [
    # Artifact widgets
    "ArtifactGallery",
    "ArtifactItem",
    "ArtifactSection",
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
    "LogViewer",
    "PlanProgressBar",
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
    "ExecutionStep",
    # Input widgets
    "InfoBar",
    "StatusPanel",
    "CommandDropdown",
    # Welcome screen
    "WelcomeBanner",
    "WelcomeScreen",
    # Utilities
    "WrappedLabel",
    "WrappedStatic",
]
