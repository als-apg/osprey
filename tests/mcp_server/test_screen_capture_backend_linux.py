"""Tests for the Linux X11 screen capture backend (LinuxX11Backend).

Mocks mss and python-xlib via module-level patching so tests run on macOS CI.
"""

import os
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from osprey.mcp_server.workspace.tools.screen_capture_backends.base import (
    WindowNotFoundError,
)

# ---------------------------------------------------------------------------
# Fixtures: mock mss and Xlib modules
# ---------------------------------------------------------------------------


def _make_mock_mss_module():
    """Create a mock mss module with grab/to_png."""
    mss_mod = ModuleType("mss")
    mss_tools_mod = ModuleType("mss.tools")

    class MockScreenShot:
        def __init__(self, rgb, size, width, height):
            self.rgb = rgb
            self.size = size
            self.width = width
            self.height = height

    class MockMSS:
        monitors = [
            {"left": 0, "top": 0, "width": 3840, "height": 1080},  # 0 = all
            {"left": 0, "top": 0, "width": 1920, "height": 1080},  # 1
            {"left": 1920, "top": 0, "width": 1920, "height": 1080},  # 2
        ]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def grab(self, region):
            w = region.get("width", region.get("width", 1920))
            h = region.get("height", region.get("height", 1080))
            if isinstance(region, dict):
                w = region["width"]
                h = region["height"]
            return MockScreenShot(
                rgb=b"\x00" * (w * h * 3),
                size=(w, h),
                width=w,
                height=h,
            )

    mss_mod.mss = MockMSS
    mss_tools_mod.to_png = MagicMock(side_effect=lambda rgb, size, output: _create_file(output))
    mss_mod.tools = mss_tools_mod

    return mss_mod, mss_tools_mod


def _create_file(filepath):
    """Helper: create a small dummy file."""
    with open(filepath, "wb") as f:
        f.write(b"FAKE_PNG_DATA")


def _make_mock_xlib_modules():
    """Create mock Xlib module hierarchy."""
    xlib = ModuleType("Xlib")
    xlib_x = ModuleType("Xlib.X")
    xlib_display = ModuleType("Xlib.display")
    xlib_protocol = ModuleType("Xlib.protocol")
    xlib_protocol_rq = ModuleType("Xlib.protocol.rq")

    # X constants
    xlib_x.AnyPropertyType = 0
    xlib_x.SubstructureRedirectMask = 1 << 20
    xlib_x.SubstructureNotifyMask = 1 << 19
    xlib.X = xlib_x

    # Mock window
    class MockGeometry:
        width = 800
        height = 600

    class MockCoords:
        x = -100
        y = -200

    class MockProperty:
        def __init__(self, value):
            self.value = value

    class MockWindow:
        def __init__(self, wid):
            self._wid = wid

        def get_wm_class(self):
            return ("app", "TestApp")

        def get_wm_name(self):
            return "Test Window"

        def get_full_property(self, atom, prop_type):
            return MockProperty(b"Test Window")

        def get_geometry(self):
            return MockGeometry()

        def translate_coords(self, root, x, y):
            return MockCoords()

        def configure(self, **kwargs):
            pass

    # Mock display
    class MockScreen:
        def root(self_inner):
            pass

    class MockRoot:
        def get_full_property(self, atom, prop_type):
            return MockProperty([0x100001])

        def send_event(self, event, event_mask):
            pass

    class MockDisplay:
        def __init__(self, *args, **kwargs):
            self._root = MockRoot()

        def screen(self):
            mock_screen = MagicMock()
            mock_screen.root = self._root
            return mock_screen

        def intern_atom(self, name):
            return hash(name) & 0xFFFF

        def create_resource_object(self, type_name, wid):
            return MockWindow(wid)

        def flush(self):
            pass

        def close(self):
            pass

    xlib_display.Display = MockDisplay
    xlib.display = xlib_display
    xlib.protocol = xlib_protocol
    xlib_protocol.rq = xlib_protocol_rq

    return {
        "Xlib": xlib,
        "Xlib.X": xlib_x,
        "Xlib.display": xlib_display,
        "Xlib.protocol": xlib_protocol,
        "Xlib.protocol.rq": xlib_protocol_rq,
    }


@pytest.fixture
def mock_linux_env(tmp_path):
    """Set up mocked mss + Xlib modules and DISPLAY env var."""
    mss_mod, mss_tools_mod = _make_mock_mss_module()
    xlib_mods = _make_mock_xlib_modules()

    modules = {
        "mss": mss_mod,
        "mss.tools": mss_tools_mod,
        **xlib_mods,
    }

    with patch.dict(sys.modules, modules):
        with patch.dict(os.environ, {"DISPLAY": ":0"}):
            # Import after mocking
            from osprey.mcp_server.workspace.tools.screen_capture_backends.linux_x11 import (
                LinuxX11Backend,
            )

            yield LinuxX11Backend()


# ---------------------------------------------------------------------------
# Screenshot tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_capture_full(mock_linux_env, tmp_path):
    """capture_full captures all monitors combined."""
    backend = mock_linux_env
    filepath = str(tmp_path / "full.png")
    info = await backend.capture_full(filepath)

    assert info.width == 3840
    assert info.height == 1080
    assert info.filepath == filepath
    assert info.size_bytes > 0


@pytest.mark.unit
async def test_capture_display_valid(mock_linux_env, tmp_path):
    """capture_display captures a specific monitor."""
    backend = mock_linux_env
    filepath = str(tmp_path / "display1.png")

    # Mock the monitor count check
    mss_mod, _ = _make_mock_mss_module()
    with patch.dict(sys.modules, {"mss": mss_mod}):
        info = await backend.capture_display("1", filepath)

    assert info.width == 1920
    assert info.height == 1080


@pytest.mark.unit
async def test_capture_display_invalid(mock_linux_env, tmp_path):
    """capture_display raises ValueError for out-of-range display."""
    backend = mock_linux_env
    filepath = str(tmp_path / "bad.png")

    mss_mod, _ = _make_mock_mss_module()
    with patch.dict(sys.modules, {"mss": mss_mod}):
        with pytest.raises(ValueError, match="Invalid display"):
            await backend.capture_display("99", filepath)


@pytest.mark.unit
async def test_capture_region(mock_linux_env, tmp_path):
    """capture_region captures a specific rectangle."""
    backend = mock_linux_env
    filepath = str(tmp_path / "region.png")
    info = await backend.capture_region(100, 200, 400, 300, filepath)

    assert info.width == 400
    assert info.height == 300


# ---------------------------------------------------------------------------
# Window listing tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_list_windows_basic(mock_linux_env):
    """list_windows returns WindowInfo objects from EWMH."""
    backend = mock_linux_env
    windows = await backend.list_windows()

    assert len(windows) == 1
    assert windows[0].app == "TestApp"
    assert windows[0].title == "Test Window"
    assert windows[0].width == 800
    assert windows[0].height == 600


@pytest.mark.unit
async def test_list_windows_filter(mock_linux_env):
    """list_windows filters by app name."""
    backend = mock_linux_env

    # Match
    windows = await backend.list_windows(app_filter="test")
    assert len(windows) == 1

    # No match
    windows = await backend.list_windows(app_filter="nonexistent")
    assert len(windows) == 0


# ---------------------------------------------------------------------------
# Window not found
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_capture_window_not_found(mock_linux_env, tmp_path):
    """capture_window raises WindowNotFoundError for unknown app."""
    backend = mock_linux_env
    filepath = str(tmp_path / "window.png")

    with pytest.raises(WindowNotFoundError):
        await backend.capture_window("NonExistentApp", filepath)


# ---------------------------------------------------------------------------
# Window management tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_bring_to_front(mock_linux_env):
    """bring_to_front sends _NET_ACTIVE_WINDOW event."""
    backend = mock_linux_env
    # Should not raise
    await backend.bring_to_front("TestApp")


@pytest.mark.unit
async def test_move_window(mock_linux_env):
    """move_window configures window position."""
    backend = mock_linux_env
    await backend.move_window("TestApp", 100, 200)


@pytest.mark.unit
async def test_resize_window(mock_linux_env):
    """resize_window configures window size."""
    backend = mock_linux_env
    await backend.resize_window("TestApp", 1024, 768)


@pytest.mark.unit
async def test_window_not_found_management(mock_linux_env):
    """Window management raises WindowNotFoundError for unknown app."""
    backend = mock_linux_env

    with pytest.raises(WindowNotFoundError):
        await backend.bring_to_front("NonExistentApp")
