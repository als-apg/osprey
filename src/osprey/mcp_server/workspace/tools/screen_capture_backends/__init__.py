"""Screen capture backend factory.

Detects the current platform, validates dependencies, and returns a cached
backend singleton.  The MCP tool layer calls ``get_backend()`` and never
interacts with platform-specific code directly.
"""

import logging
import sys

from osprey.mcp_server.workspace.tools.screen_capture_backends.base import (
    BackendUnavailableError,
    ImageInfo,
    ScreenCaptureBackend,
    WindowInfo,
    WindowNotFoundError,
)

__all__ = [
    "BackendUnavailableError",
    "ImageInfo",
    "ScreenCaptureBackend",
    "WindowInfo",
    "WindowNotFoundError",
    "get_backend",
    "reset_backend",
]

logger = logging.getLogger("osprey.mcp_server.tools.screen_capture_backends")

_backend: ScreenCaptureBackend | None = None


def get_backend() -> ScreenCaptureBackend:
    """Return a platform-appropriate :class:`ScreenCaptureBackend` singleton.

    Raises:
        BackendUnavailableError: If the current platform is unsupported or
            required dependencies are missing.
    """
    global _backend
    if _backend is not None:
        return _backend

    if sys.platform == "darwin":
        from osprey.mcp_server.workspace.tools.screen_capture_backends.macos import (
            MacOSBackend,
        )

        _backend = MacOSBackend()
    elif sys.platform == "linux":
        _backend = _create_linux_backend()
    else:
        raise BackendUnavailableError(
            f"Unsupported platform: {sys.platform}",
            suggestions=[
                "Screen capture is supported on macOS and Linux (X11).",
                "On macOS: no extra dependencies needed.",
                "On Linux: pip install mss python-xlib",
            ],
        )

    return _backend


def _create_linux_backend() -> ScreenCaptureBackend:
    """Validate Linux X11 dependencies and return a backend instance."""
    import os

    if not os.environ.get("DISPLAY"):
        raise BackendUnavailableError(
            "Linux X11 backend requires DISPLAY environment variable.",
            suggestions=[
                "Ensure an X11 display server is running.",
                "Set DISPLAY, e.g. export DISPLAY=:0",
            ],
        )

    missing: list[str] = []
    try:
        import mss  # noqa: F401
    except ImportError:
        missing.append("mss")
    try:
        import Xlib  # noqa: F401
    except ImportError:
        missing.append("python-xlib")

    if missing:
        raise BackendUnavailableError(
            f"Linux X11 backend requires: {', '.join(missing)}",
            suggestions=[f"pip install {' '.join(missing)}"],
        )

    from osprey.mcp_server.workspace.tools.screen_capture_backends.linux_x11 import (
        LinuxX11Backend,
    )

    return LinuxX11Backend()


def reset_backend() -> None:
    """Clear the cached backend singleton — used between tests."""
    global _backend
    _backend = None
