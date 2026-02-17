"""Tests for the HTML-to-image converter module."""

import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osprey.mcp_server.export.converter import (
    PlaywrightNotInstalledError,
    convert_html_to_image,
)


def _make_playwright_mock():
    """Build a mock playwright module with async_playwright context manager."""
    mock_page = AsyncMock()
    mock_browser = AsyncMock()
    mock_browser.new_page.return_value = mock_page

    mock_pw = AsyncMock()
    mock_pw.chromium.launch.return_value = mock_browser

    mock_ctx = AsyncMock()
    mock_ctx.__aenter__.return_value = mock_pw
    mock_ctx.__aexit__.return_value = False

    mock_async_playwright = MagicMock(return_value=mock_ctx)

    # Build a fake playwright.async_api module
    mod = ModuleType("playwright.async_api")
    mod.async_playwright = mock_async_playwright  # type: ignore[attr-defined]

    return mod, mock_page, mock_browser


@pytest.mark.unit
async def test_convert_html_to_png(tmp_path):
    """Successful conversion calls Playwright with correct arguments."""
    html_file = tmp_path / "plot.html"
    html_file.write_text("<html><body><h1>Plot</h1></body></html>")
    output_file = tmp_path / "plot.png"
    output_file.write_bytes(b"fake png")

    mock_mod, mock_page, mock_browser = _make_playwright_mock()

    with patch.dict(sys.modules, {"playwright.async_api": mock_mod, "playwright": MagicMock()}):
        result = await convert_html_to_image(html_file, output_file)

    assert result == output_file.resolve()
    mock_browser.new_page.assert_called_once_with(viewport={"width": 1200, "height": 800})
    mock_page.goto.assert_called_once()
    assert "file://" in mock_page.goto.call_args[0][0]
    mock_page.screenshot.assert_called_once_with(path=str(output_file), type="png", full_page=True)
    mock_browser.close.assert_called_once()


@pytest.mark.unit
async def test_playwright_not_installed(tmp_path):
    """Missing playwright module raises PlaywrightNotInstalledError."""
    html_file = tmp_path / "plot.html"
    html_file.write_text("<html></html>")
    output_file = tmp_path / "plot.png"

    # Remove playwright from sys.modules to trigger ImportError
    with patch.dict(sys.modules, {"playwright.async_api": None, "playwright": None}):
        with pytest.raises(PlaywrightNotInstalledError, match="not installed"):
            await convert_html_to_image(html_file, output_file)


@pytest.mark.unit
async def test_invalid_format(tmp_path):
    """Unsupported format raises ValueError."""
    html_file = tmp_path / "plot.html"
    html_file.write_text("<html></html>")

    with pytest.raises(ValueError, match="Unsupported format"):
        await convert_html_to_image(html_file, tmp_path / "out.bmp", fmt="bmp")


@pytest.mark.unit
async def test_file_not_found(tmp_path):
    """Non-existent HTML file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="not found"):
        await convert_html_to_image(tmp_path / "missing.html", tmp_path / "out.png")


@pytest.mark.unit
async def test_custom_viewport(tmp_path):
    """Custom width/height are passed through to Playwright."""
    html_file = tmp_path / "plot.html"
    html_file.write_text("<html></html>")
    output_file = tmp_path / "plot.png"
    output_file.write_bytes(b"fake")

    mock_mod, mock_page, mock_browser = _make_playwright_mock()

    with patch.dict(sys.modules, {"playwright.async_api": mock_mod, "playwright": MagicMock()}):
        await convert_html_to_image(html_file, output_file, width=800, height=600)

    mock_browser.new_page.assert_called_once_with(viewport={"width": 800, "height": 600})
