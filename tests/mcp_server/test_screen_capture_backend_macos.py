"""Tests for the macOS screen capture backend (MacOSBackend).

Mocks asyncio.create_subprocess_exec to test the macOS-specific Swift,
screencapture, sips, and osascript interactions without running actual
system commands.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from osprey.mcp_server.workspace.tools.screen_capture_backends.base import (
    WindowNotFoundError,
)
from osprey.mcp_server.workspace.tools.screen_capture_backends.macos import MacOSBackend

_SUBPROCESS_PATCH = (
    "osprey.mcp_server.workspace.tools.screen_capture_backends.macos.asyncio.create_subprocess_exec"
)


def _make_mock_process(returncode=0, stdout=b"", stderr=b""):
    """Create a mock asyncio subprocess."""
    proc = AsyncMock()
    proc.returncode = returncode
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    return proc


def _sips_output(width=1920, height=1080):
    """Build mock sips output for image dimensions."""
    return (f"/path/to/file.png\n  pixelWidth: {width}\n  pixelHeight: {height}\n").encode()


async def _with_fake_tools(call, *, sips=None, windows=None):
    """Await ``call()`` with every spawned macOS tool faked; return its result and the argvs.

    ``swift`` lists ``windows`` as JSON, ``screencapture`` writes PNG bytes to the
    file it was given last, ``sips`` reports ``sips`` (1920x1080 by default), and any
    other command succeeds silently.
    """
    spawned = []

    async def mock_exec(*args, **kwargs):
        spawned.append(args)
        if args[0] == "swift" and windows is not None:
            return _make_mock_process(returncode=0, stdout=json.dumps(windows).encode())
        if args[0] == "screencapture":
            open(args[-1], "wb").write(b"PNG_DATA")
            return _make_mock_process(returncode=0)
        if args[0] == "sips":
            return _make_mock_process(returncode=0, stdout=sips or _sips_output())
        return _make_mock_process(returncode=0)

    with patch(_SUBPROCESS_PATCH, side_effect=mock_exec):
        result = await call()
    return result, spawned


def _screencapture_args(spawned):
    return [a for a in spawned if a[0] == "screencapture"][0]


_WINDOWS = [
    {"wid": 1, "app": "Finder", "title": "", "x": 0, "y": 0, "width": 800, "height": 600},
    {"wid": 2, "app": "Terminal", "title": "bash", "x": 100, "y": 100, "width": 600, "height": 400},
]


@pytest.fixture
def backend():
    return MacOSBackend()


# ---------------------------------------------------------------------------
# capture_full
# ---------------------------------------------------------------------------


async def test_capture_full(backend, tmp_path):
    """capture_full runs screencapture and sips, returns ImageInfo."""
    filepath = str(tmp_path / "test.png")

    info, _ = await _with_fake_tools(
        lambda: backend.capture_full(filepath), sips=_sips_output(2560, 1440)
    )

    assert info.width == 2560
    assert info.height == 1440
    assert info.filepath == filepath
    assert info.size_bytes > 0


# ---------------------------------------------------------------------------
# capture_display
# ---------------------------------------------------------------------------


async def test_capture_display(backend, tmp_path):
    """capture_display passes -D flag."""
    filepath = str(tmp_path / "test.png")

    info, spawned = await _with_fake_tools(lambda: backend.capture_display("2", filepath))

    assert info.width == 1920
    assert "-D2" in _screencapture_args(spawned)


# ---------------------------------------------------------------------------
# capture_region
# ---------------------------------------------------------------------------


async def test_capture_region(backend, tmp_path):
    """capture_region passes -R flag with coordinates."""
    filepath = str(tmp_path / "test.png")

    info, spawned = await _with_fake_tools(
        lambda: backend.capture_region(100, 200, 800, 600, filepath), sips=_sips_output(800, 600)
    )

    assert info.width == 800
    assert info.height == 600
    assert "-R100,200,800,600" in _screencapture_args(spawned)


# ---------------------------------------------------------------------------
# capture_window
# ---------------------------------------------------------------------------


async def test_capture_window_by_wid(backend, tmp_path):
    """capture_window with numeric WID uses -l flag."""
    filepath = str(tmp_path / "test.png")

    _, spawned = await _with_fake_tools(lambda: backend.capture_window("12345", filepath))

    sc_args = _screencapture_args(spawned)
    assert "-l12345" in sc_args
    assert "-o" in sc_args


async def test_capture_window_by_name(backend, tmp_path):
    """capture_window with app name resolves WID via Swift."""
    filepath = str(tmp_path / "test.png")
    windows = [
        {"wid": 42, "app": "Phoebus", "title": "Main", "x": 0, "y": 0, "width": 800, "height": 600},
    ]

    _, spawned = await _with_fake_tools(
        lambda: backend.capture_window("Phoebus", filepath), windows=windows
    )

    assert "-l42" in _screencapture_args(spawned)


async def test_capture_window_not_found(backend, tmp_path):
    """capture_window raises WindowNotFoundError for unknown app."""
    filepath = str(tmp_path / "test.png")

    with pytest.raises(WindowNotFoundError):
        await _with_fake_tools(lambda: backend.capture_window("NonExistent", filepath), windows=[])


# ---------------------------------------------------------------------------
# list_windows
# ---------------------------------------------------------------------------


async def test_list_windows_basic(backend):
    """list_windows returns WindowInfo objects."""
    windows, _ = await _with_fake_tools(backend.list_windows, windows=_WINDOWS)

    assert len(windows) == 2
    assert windows[0].app == "Finder"
    assert windows[1].wid == 2


async def test_list_windows_filter(backend):
    """list_windows filters by app name."""
    windows, _ = await _with_fake_tools(
        lambda: backend.list_windows(app_filter="terminal"), windows=_WINDOWS
    )

    assert len(windows) == 1
    assert windows[0].app == "Terminal"


# ---------------------------------------------------------------------------
# Window management
# ---------------------------------------------------------------------------


async def test_bring_to_front(backend):
    """bring_to_front runs osascript with activate."""
    _, spawned = await _with_fake_tools(lambda: backend.bring_to_front("Phoebus"))

    assert spawned[0][0] == "osascript"
    assert "activate" in spawned[0][2]
    assert "Phoebus" in spawned[0][2]


async def test_injection_prevention(backend):
    """App names with special characters raise ValueError."""
    with pytest.raises(ValueError, match="Invalid app name"):
        await backend.bring_to_front('Finder"; do shell script "rm -rf /')


async def test_move_window(backend):
    """move_window runs osascript with position."""
    _, spawned = await _with_fake_tools(lambda: backend.move_window("Terminal", 100, 200))

    assert "position" in spawned[0][2]
    assert "{100, 200}" in spawned[0][2]


async def test_resize_window(backend):
    """resize_window runs osascript with size."""
    _, spawned = await _with_fake_tools(lambda: backend.resize_window("Terminal", 1024, 768))

    assert "size" in spawned[0][2]
    assert "{1024, 768}" in spawned[0][2]
