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


@pytest.fixture
def backend():
    return MacOSBackend()


# ---------------------------------------------------------------------------
# capture_full
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_capture_full(backend, tmp_path):
    """capture_full runs screencapture and sips, returns ImageInfo."""
    filepath = str(tmp_path / "test.png")

    async def mock_exec(*args, **kwargs):
        if args[0] == "screencapture":
            open(args[-1], "wb").write(b"PNG_DATA")
            return _make_mock_process(returncode=0)
        elif args[0] == "sips":
            return _make_mock_process(returncode=0, stdout=_sips_output(2560, 1440))
        return _make_mock_process()

    with patch(_SUBPROCESS_PATCH, side_effect=mock_exec):
        info = await backend.capture_full(filepath)

    assert info.width == 2560
    assert info.height == 1440
    assert info.filepath == filepath
    assert info.size_bytes > 0


# ---------------------------------------------------------------------------
# capture_display
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_capture_display(backend, tmp_path):
    """capture_display passes -D flag."""
    filepath = str(tmp_path / "test.png")
    captured_args = []

    async def mock_exec(*args, **kwargs):
        captured_args.append(args)
        if args[0] == "screencapture":
            open(filepath, "wb").write(b"PNG_DATA")
            return _make_mock_process(returncode=0)
        elif args[0] == "sips":
            return _make_mock_process(returncode=0, stdout=_sips_output())
        return _make_mock_process()

    with patch(_SUBPROCESS_PATCH, side_effect=mock_exec):
        info = await backend.capture_display("2", filepath)

    assert info.width == 1920
    sc_args = [a for a in captured_args if a[0] == "screencapture"][0]
    assert "-D2" in sc_args


# ---------------------------------------------------------------------------
# capture_region
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_capture_region(backend, tmp_path):
    """capture_region passes -R flag with coordinates."""
    filepath = str(tmp_path / "test.png")
    captured_args = []

    async def mock_exec(*args, **kwargs):
        captured_args.append(args)
        if args[0] == "screencapture":
            open(filepath, "wb").write(b"PNG_DATA")
            return _make_mock_process(returncode=0)
        elif args[0] == "sips":
            return _make_mock_process(returncode=0, stdout=_sips_output(800, 600))
        return _make_mock_process()

    with patch(_SUBPROCESS_PATCH, side_effect=mock_exec):
        info = await backend.capture_region(100, 200, 800, 600, filepath)

    assert info.width == 800
    assert info.height == 600
    sc_args = [a for a in captured_args if a[0] == "screencapture"][0]
    assert "-R100,200,800,600" in sc_args


# ---------------------------------------------------------------------------
# capture_window
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_capture_window_by_wid(backend, tmp_path):
    """capture_window with numeric WID uses -l flag."""
    filepath = str(tmp_path / "test.png")
    captured_args = []

    async def mock_exec(*args, **kwargs):
        captured_args.append(args)
        if args[0] == "screencapture":
            open(filepath, "wb").write(b"PNG_DATA")
            return _make_mock_process(returncode=0)
        elif args[0] == "sips":
            return _make_mock_process(returncode=0, stdout=_sips_output())
        return _make_mock_process()

    with patch(_SUBPROCESS_PATCH, side_effect=mock_exec):
        await backend.capture_window("12345", filepath)

    sc_args = [a for a in captured_args if a[0] == "screencapture"][0]
    assert "-l12345" in sc_args
    assert "-o" in sc_args


@pytest.mark.unit
async def test_capture_window_by_name(backend, tmp_path):
    """capture_window with app name resolves WID via Swift."""
    filepath = str(tmp_path / "test.png")
    mock_windows = [
        {"wid": 42, "app": "Phoebus", "title": "Main", "x": 0, "y": 0, "width": 800, "height": 600},
    ]
    captured_args = []

    async def mock_exec(*args, **kwargs):
        captured_args.append(args)
        if args[0] == "swift":
            return _make_mock_process(returncode=0, stdout=json.dumps(mock_windows).encode())
        elif args[0] == "screencapture":
            open(filepath, "wb").write(b"PNG_DATA")
            return _make_mock_process(returncode=0)
        elif args[0] == "sips":
            return _make_mock_process(returncode=0, stdout=_sips_output())
        return _make_mock_process()

    with patch(_SUBPROCESS_PATCH, side_effect=mock_exec):
        await backend.capture_window("Phoebus", filepath)

    sc_args = [a for a in captured_args if a[0] == "screencapture"][0]
    assert "-l42" in sc_args


@pytest.mark.unit
async def test_capture_window_not_found(backend, tmp_path):
    """capture_window raises WindowNotFoundError for unknown app."""
    filepath = str(tmp_path / "test.png")

    async def mock_exec(*args, **kwargs):
        if args[0] == "swift":
            return _make_mock_process(returncode=0, stdout=b"[]")
        return _make_mock_process()

    with patch(_SUBPROCESS_PATCH, side_effect=mock_exec):
        with pytest.raises(WindowNotFoundError):
            await backend.capture_window("NonExistent", filepath)


# ---------------------------------------------------------------------------
# list_windows
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_list_windows_basic(backend):
    """list_windows returns WindowInfo objects."""
    mock_windows = [
        {"wid": 1, "app": "Finder", "title": "", "x": 0, "y": 0, "width": 800, "height": 600},
        {
            "wid": 2,
            "app": "Terminal",
            "title": "bash",
            "x": 100,
            "y": 100,
            "width": 600,
            "height": 400,
        },
    ]

    async def mock_exec(*args, **kwargs):
        return _make_mock_process(returncode=0, stdout=json.dumps(mock_windows).encode())

    with patch(_SUBPROCESS_PATCH, side_effect=mock_exec):
        windows = await backend.list_windows()

    assert len(windows) == 2
    assert windows[0].app == "Finder"
    assert windows[1].wid == 2


@pytest.mark.unit
async def test_list_windows_filter(backend):
    """list_windows filters by app name."""
    mock_windows = [
        {"wid": 1, "app": "Finder", "title": "", "x": 0, "y": 0, "width": 800, "height": 600},
        {
            "wid": 2,
            "app": "Terminal",
            "title": "bash",
            "x": 100,
            "y": 100,
            "width": 600,
            "height": 400,
        },
    ]

    async def mock_exec(*args, **kwargs):
        return _make_mock_process(returncode=0, stdout=json.dumps(mock_windows).encode())

    with patch(_SUBPROCESS_PATCH, side_effect=mock_exec):
        windows = await backend.list_windows(app_filter="terminal")

    assert len(windows) == 1
    assert windows[0].app == "Terminal"


# ---------------------------------------------------------------------------
# Window management
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_bring_to_front(backend):
    """bring_to_front runs osascript with activate."""
    captured_args = []

    async def mock_exec(*args, **kwargs):
        captured_args.append(args)
        return _make_mock_process(returncode=0)

    with patch(_SUBPROCESS_PATCH, side_effect=mock_exec):
        await backend.bring_to_front("Phoebus")

    assert captured_args[0][0] == "osascript"
    assert "activate" in captured_args[0][2]
    assert "Phoebus" in captured_args[0][2]


@pytest.mark.unit
async def test_injection_prevention(backend):
    """App names with special characters raise ValueError."""
    with pytest.raises(ValueError, match="Invalid app name"):
        await backend.bring_to_front('Finder"; do shell script "rm -rf /')


@pytest.mark.unit
async def test_move_window(backend):
    """move_window runs osascript with position."""
    captured_args = []

    async def mock_exec(*args, **kwargs):
        captured_args.append(args)
        return _make_mock_process(returncode=0)

    with patch(_SUBPROCESS_PATCH, side_effect=mock_exec):
        await backend.move_window("Terminal", 100, 200)

    assert "position" in captured_args[0][2]
    assert "{100, 200}" in captured_args[0][2]


@pytest.mark.unit
async def test_resize_window(backend):
    """resize_window runs osascript with size."""
    captured_args = []

    async def mock_exec(*args, **kwargs):
        captured_args.append(args)
        return _make_mock_process(returncode=0)

    with patch(_SUBPROCESS_PATCH, side_effect=mock_exec):
        await backend.resize_window("Terminal", 1024, 768)

    assert "size" in captured_args[0][2]
    assert "{1024, 768}" in captured_args[0][2]
