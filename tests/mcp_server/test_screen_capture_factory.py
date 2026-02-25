"""Tests for the screen capture backend factory (get_backend / reset_backend).

Validates platform detection, singleton caching, dependency checking, and
error messages.  All platform/import checks are mocked — runs on any OS.
"""

import importlib
from unittest.mock import patch

import pytest

from osprey.mcp_server.workspace.tools.screen_capture_backends import (
    reset_backend,
)
from osprey.mcp_server.workspace.tools.screen_capture_backends.base import (
    BackendUnavailableError,
)


@pytest.mark.unit
def test_darwin_returns_macos_backend():
    """On macOS, get_backend() returns a MacOSBackend."""
    reset_backend()

    with patch("osprey.mcp_server.workspace.tools.screen_capture_backends.sys") as mock_sys:
        mock_sys.platform = "darwin"
        # Re-import to pick up the patched sys
        import osprey.mcp_server.workspace.tools.screen_capture_backends as factory

        importlib.reload(factory)

    # After reload with darwin, calling get_backend should return MacOSBackend
    reset_backend()
    with patch("osprey.mcp_server.workspace.tools.screen_capture_backends.sys") as mock_sys:
        mock_sys.platform = "darwin"
        from osprey.mcp_server.workspace.tools.screen_capture_backends import get_backend
        from osprey.mcp_server.workspace.tools.screen_capture_backends.macos import (
            MacOSBackend,
        )

        backend = get_backend()
        assert isinstance(backend, MacOSBackend)


@pytest.mark.unit
def test_singleton_caching():
    """get_backend() returns the same instance on repeated calls."""
    reset_backend()

    with patch("osprey.mcp_server.workspace.tools.screen_capture_backends.sys") as mock_sys:
        mock_sys.platform = "darwin"
        from osprey.mcp_server.workspace.tools.screen_capture_backends import get_backend

        b1 = get_backend()
        b2 = get_backend()
        assert b1 is b2


@pytest.mark.unit
def test_reset_clears_cache():
    """reset_backend() clears the singleton so next call creates a new one."""
    reset_backend()

    with patch("osprey.mcp_server.workspace.tools.screen_capture_backends.sys") as mock_sys:
        mock_sys.platform = "darwin"
        from osprey.mcp_server.workspace.tools.screen_capture_backends import get_backend

        b1 = get_backend()
        reset_backend()
        b2 = get_backend()
        assert b1 is not b2


@pytest.mark.unit
def test_linux_no_display():
    """Linux without DISPLAY raises BackendUnavailableError."""
    reset_backend()

    with (
        patch("osprey.mcp_server.workspace.tools.screen_capture_backends.sys") as mock_sys,
        patch.dict("os.environ", {}, clear=True),
    ):
        mock_sys.platform = "linux"
        from osprey.mcp_server.workspace.tools.screen_capture_backends import get_backend

        with pytest.raises(BackendUnavailableError, match="DISPLAY"):
            get_backend()


@pytest.mark.unit
def test_linux_missing_mss():
    """Linux with DISPLAY but missing mss raises BackendUnavailableError."""
    reset_backend()

    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "mss":
            raise ImportError("No module named 'mss'")
        return real_import(name, *args, **kwargs)

    with (
        patch("osprey.mcp_server.workspace.tools.screen_capture_backends.sys") as mock_sys,
        patch.dict("os.environ", {"DISPLAY": ":0"}),
        patch("builtins.__import__", side_effect=mock_import),
    ):
        mock_sys.platform = "linux"
        from osprey.mcp_server.workspace.tools.screen_capture_backends import get_backend

        with pytest.raises(BackendUnavailableError, match="mss"):
            get_backend()


@pytest.mark.unit
def test_linux_missing_xlib():
    """Linux with DISPLAY but missing python-xlib raises BackendUnavailableError."""
    reset_backend()

    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "Xlib":
            raise ImportError("No module named 'Xlib'")
        return real_import(name, *args, **kwargs)

    with (
        patch("osprey.mcp_server.workspace.tools.screen_capture_backends.sys") as mock_sys,
        patch.dict("os.environ", {"DISPLAY": ":0"}),
        patch("builtins.__import__", side_effect=mock_import),
    ):
        mock_sys.platform = "linux"
        from osprey.mcp_server.workspace.tools.screen_capture_backends import get_backend

        with pytest.raises(BackendUnavailableError, match="python-xlib"):
            get_backend()


@pytest.mark.unit
def test_unsupported_platform():
    """Unsupported platform raises BackendUnavailableError."""
    reset_backend()

    with patch("osprey.mcp_server.workspace.tools.screen_capture_backends.sys") as mock_sys:
        mock_sys.platform = "win32"
        from osprey.mcp_server.workspace.tools.screen_capture_backends import get_backend

        with pytest.raises(BackendUnavailableError, match="win32"):
            get_backend()


@pytest.mark.unit
def test_error_carries_suggestions():
    """BackendUnavailableError includes helpful suggestions."""
    reset_backend()

    with patch("osprey.mcp_server.workspace.tools.screen_capture_backends.sys") as mock_sys:
        mock_sys.platform = "win32"
        from osprey.mcp_server.workspace.tools.screen_capture_backends import get_backend

        with pytest.raises(BackendUnavailableError) as exc_info:
            get_backend()
        assert len(exc_info.value.suggestions) >= 1
        assert any("macOS" in s for s in exc_info.value.suggestions)
