"""Tests for screen capture MCP tools (tool-layer delegation).

Tests the thin MCP tool layer in screen_capture.py: validation, delegation to
the backend, exception-to-make_error mapping, and ArtifactStore integration.

All backend calls are mocked — no actual screencapture/swift/osascript execution.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from osprey.mcp_server.workspace.tools.screen_capture_backends.base import (
    BackendUnavailableError,
    ImageInfo,
    WindowInfo,
    WindowNotFoundError,
)
from tests.mcp_server.conftest import assert_error, extract_response_dict, get_tool_fn


def _get_screenshot_capture():
    from osprey.mcp_server.workspace.tools.screen_capture import screenshot_capture

    return get_tool_fn(screenshot_capture)


def _get_list_windows():
    from osprey.mcp_server.workspace.tools.screen_capture import list_windows

    return get_tool_fn(list_windows)


def _get_manage_window():
    from osprey.mcp_server.workspace.tools.screen_capture import manage_window

    return get_tool_fn(manage_window)


def _mock_image_info(filepath="/tmp/test.png", width=1920, height=1080, size_bytes=8192):
    return ImageInfo(filepath=filepath, width=width, height=height, size_bytes=size_bytes)


def _make_mock_backend(**overrides):
    """Create a mock ScreenCaptureBackend with sensible defaults."""
    from osprey.mcp_server.workspace.tools.screen_capture_backends.base import (
        ScreenCaptureBackend,
    )

    backend = AsyncMock(spec=ScreenCaptureBackend)
    backend.capture_full.return_value = _mock_image_info()
    backend.capture_display.return_value = _mock_image_info()
    backend.capture_region.return_value = _mock_image_info()
    backend.capture_window.return_value = _mock_image_info()
    backend.list_windows.return_value = []
    for k, v in overrides.items():
        setattr(backend, k, v)
    return backend


_BACKEND_PATCH = "osprey.mcp_server.workspace.tools.screen_capture.get_backend"


# ---------------------------------------------------------------------------
# screenshot_capture tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_screenshot_full_mode(tmp_path, monkeypatch):
    """Full mode captures entire screen and returns ArtifactStore entry."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(
        "screen_capture:\n  output_dir: '" + str(tmp_path / "screenshots") + "'\n"
    )

    filepath_holder = []

    async def mock_capture_full(filepath):
        filepath_holder.append(filepath)
        open(filepath, "wb").write(b"PNG_DATA")
        return _mock_image_info(filepath=filepath, width=2560, height=1440)

    backend = _make_mock_backend()
    backend.capture_full = AsyncMock(side_effect=mock_capture_full)

    with patch(_BACKEND_PATCH, return_value=backend):
        fn = _get_screenshot_capture()
        result = await fn(mode="full")

    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert "artifact_id" in data
    assert data["summary"]["mode"] == "full"
    assert data["summary"]["dimensions"] == "2560x1440"
    assert "screenshots" in data["summary"]["filepath"]


@pytest.mark.unit
async def test_screenshot_region_mode(tmp_path, monkeypatch):
    """Region mode parses x,y,w,h and delegates to backend."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(
        "screen_capture:\n  output_dir: '" + str(tmp_path / "screenshots") + "'\n"
    )

    async def mock_capture_region(x, y, w, h, filepath):
        open(filepath, "wb").write(b"PNG_DATA")
        return _mock_image_info(filepath=filepath, width=800, height=600)

    backend = _make_mock_backend()
    backend.capture_region = AsyncMock(side_effect=mock_capture_region)

    with patch(_BACKEND_PATCH, return_value=backend):
        fn = _get_screenshot_capture()
        result = await fn(mode="region", target="100,200,800,600")

    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert "artifact_id" in data
    backend.capture_region.assert_called_once()
    call_args = backend.capture_region.call_args
    assert call_args[0][:4] == (100, 200, 800, 600)


@pytest.mark.unit
async def test_screenshot_display_mode(tmp_path, monkeypatch):
    """Display mode passes display number to backend."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(
        "screen_capture:\n  output_dir: '" + str(tmp_path / "screenshots") + "'\n"
    )

    async def mock_capture_display(display, filepath):
        open(filepath, "wb").write(b"PNG_DATA")
        return _mock_image_info(filepath=filepath)

    backend = _make_mock_backend()
    backend.capture_display = AsyncMock(side_effect=mock_capture_display)

    with patch(_BACKEND_PATCH, return_value=backend):
        fn = _get_screenshot_capture()
        result = await fn(mode="display", target="2")

    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert "artifact_id" in data
    backend.capture_display.assert_called_once()
    assert backend.capture_display.call_args[0][0] == "2"


@pytest.mark.unit
async def test_screenshot_window_by_wid(tmp_path, monkeypatch):
    """Window mode with numeric WID delegates to backend."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(
        "screen_capture:\n  output_dir: '" + str(tmp_path / "screenshots") + "'\n"
    )

    async def mock_capture_window(target, filepath):
        open(filepath, "wb").write(b"PNG_DATA")
        return _mock_image_info(filepath=filepath)

    backend = _make_mock_backend()
    backend.capture_window = AsyncMock(side_effect=mock_capture_window)

    with patch(_BACKEND_PATCH, return_value=backend):
        fn = _get_screenshot_capture()
        result = await fn(mode="window", target="12345")

    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert "artifact_id" in data
    backend.capture_window.assert_called_once()
    assert backend.capture_window.call_args[0][0] == "12345"


@pytest.mark.unit
async def test_screenshot_window_by_name(tmp_path, monkeypatch):
    """Window mode with app name delegates to backend."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(
        "screen_capture:\n  output_dir: '" + str(tmp_path / "screenshots") + "'\n"
    )

    async def mock_capture_window(target, filepath):
        open(filepath, "wb").write(b"PNG_DATA")
        return _mock_image_info(filepath=filepath)

    backend = _make_mock_backend()
    backend.capture_window = AsyncMock(side_effect=mock_capture_window)

    with patch(_BACKEND_PATCH, return_value=backend):
        fn = _get_screenshot_capture()
        result = await fn(mode="window", target="Phoebus")

    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert "artifact_id" in data
    backend.capture_window.assert_called_once()
    assert backend.capture_window.call_args[0][0] == "Phoebus"


@pytest.mark.unit
async def test_screenshot_window_name_not_found(tmp_path, monkeypatch):
    """Window mode returns error when backend raises WindowNotFoundError."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(
        "screen_capture:\n  output_dir: '" + str(tmp_path / "screenshots") + "'\n"
    )

    backend = _make_mock_backend()
    backend.capture_window = AsyncMock(
        side_effect=WindowNotFoundError("No window found for app 'NonExistentApp'.")
    )

    with patch(_BACKEND_PATCH, return_value=backend):
        fn = _get_screenshot_capture()
        result = await fn(mode="window", target="NonExistentApp")

    data = assert_error(result, error_type="window_not_found")


@pytest.mark.unit
async def test_screenshot_custom_filename(tmp_path, monkeypatch):
    """Custom filename is used in the output path."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(
        "screen_capture:\n  output_dir: '" + str(tmp_path / "screenshots") + "'\n"
    )

    async def mock_capture_full(filepath):
        open(filepath, "wb").write(b"PNG_DATA")
        return _mock_image_info(filepath=filepath)

    backend = _make_mock_backend()
    backend.capture_full = AsyncMock(side_effect=mock_capture_full)

    with patch(_BACKEND_PATCH, return_value=backend):
        fn = _get_screenshot_capture()
        result = await fn(mode="full", filename="my_screenshot")

    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert "artifact_id" in data
    assert "my_screenshot.png" in data["summary"]["filepath"]


@pytest.mark.unit
async def test_screenshot_default_filename(tmp_path, monkeypatch):
    """Default filename uses capture_YYYYMMDD_HHMMSS pattern."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(
        "screen_capture:\n  output_dir: '" + str(tmp_path / "screenshots") + "'\n"
    )

    async def mock_capture_full(filepath):
        open(filepath, "wb").write(b"PNG_DATA")
        return _mock_image_info(filepath=filepath)

    backend = _make_mock_backend()
    backend.capture_full = AsyncMock(side_effect=mock_capture_full)

    with patch(_BACKEND_PATCH, return_value=backend):
        fn = _get_screenshot_capture()
        result = await fn(mode="full")

    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert "artifact_id" in data
    assert "capture_" in data["summary"]["filepath"]
    assert data["summary"]["filepath"].endswith(".png")


@pytest.mark.unit
async def test_screenshot_invalid_mode(tmp_path, monkeypatch):
    """Invalid mode returns a validation error."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(
        "screen_capture:\n  output_dir: '" + str(tmp_path / "screenshots") + "'\n"
    )

    fn = _get_screenshot_capture()
    result = await fn(mode="invalid")

    data = assert_error(result, error_type="validation_error")
    assert "invalid" in data["error_message"].lower()


@pytest.mark.unit
async def test_screenshot_command_failure(tmp_path, monkeypatch):
    """Backend RuntimeError returns capture_error."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(
        "screen_capture:\n  output_dir: '" + str(tmp_path / "screenshots") + "'\n"
    )

    backend = _make_mock_backend()
    backend.capture_full = AsyncMock(
        side_effect=RuntimeError("screencapture failed (rc=1): permission denied")
    )

    with patch(_BACKEND_PATCH, return_value=backend):
        fn = _get_screenshot_capture()
        result = await fn(mode="full")

    data = assert_error(result, error_type="capture_error")


@pytest.mark.unit
async def test_screenshot_backend_unavailable(tmp_path, monkeypatch):
    """BackendUnavailableError returns platform_error."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(
        "screen_capture:\n  output_dir: '" + str(tmp_path / "screenshots") + "'\n"
    )

    with patch(
        _BACKEND_PATCH,
        side_effect=BackendUnavailableError(
            "Unsupported platform: win32",
            suggestions=["Screen capture is supported on macOS and Linux."],
        ),
    ):
        fn = _get_screenshot_capture()
        result = await fn(mode="full")

    data = assert_error(result, error_type="platform_error")
    assert "win32" in data["error_message"]
    assert len(data["suggestions"]) >= 1


# ---------------------------------------------------------------------------
# list_windows tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_list_windows_basic(tmp_path, monkeypatch):
    """list_windows returns all visible windows from backend."""
    monkeypatch.chdir(tmp_path)

    mock_windows = [
        WindowInfo(wid=1, app="Finder", title="", x=0, y=0, width=800, height=600),
        WindowInfo(wid=2, app="Terminal", title="bash", x=100, y=100, width=600, height=400),
    ]

    backend = _make_mock_backend()
    backend.list_windows = AsyncMock(return_value=mock_windows)

    with patch(_BACKEND_PATCH, return_value=backend):
        fn = _get_list_windows()
        result = await fn()

    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert data["window_count"] == 2
    assert len(data["windows"]) == 2


@pytest.mark.unit
async def test_list_windows_app_filter(tmp_path, monkeypatch):
    """list_windows passes filter to backend."""
    monkeypatch.chdir(tmp_path)

    mock_windows = [
        WindowInfo(wid=2, app="Terminal", title="bash", x=100, y=100, width=600, height=400),
    ]

    backend = _make_mock_backend()
    backend.list_windows = AsyncMock(return_value=mock_windows)

    with patch(_BACKEND_PATCH, return_value=backend):
        fn = _get_list_windows()
        result = await fn(app_filter="terminal")

    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert data["window_count"] == 1
    assert data["windows"][0]["app"] == "Terminal"
    backend.list_windows.assert_called_once_with(app_filter="terminal")


@pytest.mark.unit
async def test_list_windows_backend_failure(tmp_path, monkeypatch):
    """Backend exception returns internal_error."""
    monkeypatch.chdir(tmp_path)

    backend = _make_mock_backend()
    backend.list_windows = AsyncMock(side_effect=RuntimeError("Swift window listing failed"))

    with patch(_BACKEND_PATCH, return_value=backend):
        fn = _get_list_windows()
        result = await fn()

    data = assert_error(result, error_type="internal_error")


# ---------------------------------------------------------------------------
# manage_window tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_manage_window_bring_to_front(tmp_path, monkeypatch):
    """bring_to_front delegates to backend."""
    monkeypatch.chdir(tmp_path)

    backend = _make_mock_backend()
    backend.bring_to_front = AsyncMock()

    with patch(_BACKEND_PATCH, return_value=backend):
        fn = _get_manage_window()
        result = await fn(app="Phoebus", action="bring_to_front")

    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert data["action"] == "bring_to_front"
    backend.bring_to_front.assert_called_once_with("Phoebus")


@pytest.mark.unit
async def test_manage_window_move(tmp_path, monkeypatch):
    """Move action delegates to backend with x, y."""
    monkeypatch.chdir(tmp_path)

    backend = _make_mock_backend()
    backend.move_window = AsyncMock()

    with patch(_BACKEND_PATCH, return_value=backend):
        fn = _get_manage_window()
        result = await fn(app="Terminal", action="move", x=200, y=300)

    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert data["action"] == "move"
    assert data["details"]["x"] == 200
    assert data["details"]["y"] == 300
    backend.move_window.assert_called_once_with("Terminal", 200, 300)


@pytest.mark.unit
async def test_manage_window_resize(tmp_path, monkeypatch):
    """Resize action delegates to backend with width, height."""
    monkeypatch.chdir(tmp_path)

    backend = _make_mock_backend()
    backend.resize_window = AsyncMock()

    with patch(_BACKEND_PATCH, return_value=backend):
        fn = _get_manage_window()
        result = await fn(app="Terminal", action="resize", width=1024, height=768)

    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert data["action"] == "resize"
    assert data["details"]["width"] == 1024
    assert data["details"]["height"] == 768
    backend.resize_window.assert_called_once_with("Terminal", 1024, 768)


@pytest.mark.unit
async def test_manage_window_invalid_action(tmp_path, monkeypatch):
    """Invalid action returns validation_error."""
    monkeypatch.chdir(tmp_path)

    fn = _get_manage_window()
    result = await fn(app="Terminal", action="maximize")

    data = assert_error(result, error_type="validation_error")
    assert "maximize" in data["error_message"]


@pytest.mark.unit
async def test_manage_window_move_missing_params(tmp_path, monkeypatch):
    """Move without x/y returns validation_error."""
    monkeypatch.chdir(tmp_path)

    fn = _get_manage_window()
    result = await fn(app="Terminal", action="move", x=100)

    data = assert_error(result, error_type="validation_error")
    assert "x and y" in data["error_message"]


@pytest.mark.unit
async def test_manage_window_resize_missing_params(tmp_path, monkeypatch):
    """Resize without width/height returns validation_error."""
    monkeypatch.chdir(tmp_path)

    fn = _get_manage_window()
    result = await fn(app="Terminal", action="resize", width=800)

    data = assert_error(result, error_type="validation_error")
    assert "width and height" in data["error_message"]


@pytest.mark.unit
async def test_manage_window_injection_prevention(tmp_path, monkeypatch):
    """App names with special characters cause ValueError from backend."""
    monkeypatch.chdir(tmp_path)

    backend = _make_mock_backend()
    backend.bring_to_front = AsyncMock(
        side_effect=ValueError(
            "Invalid app name. Only alphanumeric characters, spaces, dots, "
            "underscores, and hyphens are allowed."
        )
    )

    with patch(_BACKEND_PATCH, return_value=backend):
        fn = _get_manage_window()
        result = await fn(app='Finder"; do shell script "rm -rf /', action="bring_to_front")

    data = assert_error(result, error_type="validation_error")
    assert "Invalid app name" in data["error_message"]


@pytest.mark.unit
async def test_screenshot_region_missing_target(tmp_path, monkeypatch):
    """Region mode without target returns validation error."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(
        "screen_capture:\n  output_dir: '" + str(tmp_path / "screenshots") + "'\n"
    )

    fn = _get_screenshot_capture()
    result = await fn(mode="region")

    data = assert_error(result, error_type="validation_error")
    assert "x,y,w,h" in data["error_message"]
