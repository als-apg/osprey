"""Abstract base class and domain objects for screen capture backends.

Defines the platform-agnostic interface that macOS and Linux backends implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class WindowInfo:
    """Information about a visible window.

    Attributes:
        wid: Window ID (int on macOS, hex string on X11).
        app: Application/owner name.
        title: Window title.
        x: X position in pixels.
        y: Y position in pixels.
        width: Width in pixels.
        height: Height in pixels.
    """

    wid: int | str
    app: str
    title: str
    x: int
    y: int
    width: int
    height: int

    def to_dict(self) -> dict[str, Any]:
        """Serialise for JSON responses."""
        return {
            "wid": self.wid,
            "app": self.app,
            "title": self.title,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class ImageInfo:
    """Metadata about a captured screenshot.

    Attributes:
        filepath: Absolute path to the saved PNG file.
        width: Image width in pixels.
        height: Image height in pixels.
        size_bytes: File size in bytes.
    """

    filepath: str
    width: int
    height: int
    size_bytes: int


class WindowNotFoundError(Exception):
    """Raised when a target window cannot be resolved."""


class BackendUnavailableError(Exception):
    """Raised when no suitable screen capture backend can be initialised.

    Attributes:
        suggestions: Install/config hints to include in the MCP error response.
    """

    def __init__(self, message: str, suggestions: list[str] | None = None):
        super().__init__(message)
        self.suggestions = suggestions or []


class ScreenCaptureBackend(ABC):
    """Platform-agnostic interface for screen capture and window management."""

    @abstractmethod
    async def capture_full(self, filepath: str) -> ImageInfo:
        """Capture the entire screen (all monitors combined)."""

    @abstractmethod
    async def capture_display(self, display: str, filepath: str) -> ImageInfo:
        """Capture a specific display/monitor."""

    @abstractmethod
    async def capture_region(self, x: int, y: int, w: int, h: int, filepath: str) -> ImageInfo:
        """Capture a rectangular region."""

    @abstractmethod
    async def capture_window(self, target: str, filepath: str) -> ImageInfo:
        """Capture a specific window by WID or app name."""

    @abstractmethod
    async def list_windows(self, app_filter: str | None = None) -> list[WindowInfo]:
        """List visible windows, optionally filtered by app name."""

    @abstractmethod
    async def bring_to_front(self, app: str) -> None:
        """Bring an application's window to the front."""

    @abstractmethod
    async def move_window(self, app: str, x: int, y: int) -> None:
        """Move an application's front window to the given position."""

    @abstractmethod
    async def resize_window(self, app: str, width: int, height: int) -> None:
        """Resize an application's front window."""
