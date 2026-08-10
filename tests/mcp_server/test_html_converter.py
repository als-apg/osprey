"""Tests for the HTML-to-image converter module."""

import subprocess
import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osprey.mcp_server.export import converter as converter_mod
from osprey.mcp_server.export.converter import (
    PlaywrightNotInstalledError,
    convert_html_to_image,
)

# The launch error Playwright raises when the browser binary is absent — the ONLY
# condition that may trigger an auto-install.
MISSING_BROWSER_ERROR = (
    "browserType.launch: Executable doesn't exist at "
    "/root/.cache/ms-playwright/chromium-1091/chrome-linux/chrome"
)


def _make_playwright_mock(launch_side_effect=None):
    """Build a mock playwright module with async_playwright context manager.

    Args:
        launch_side_effect: Optional ``side_effect`` for ``chromium.launch`` — a
            list whose exception entries are raised and whose other entries are
            returned, used to simulate a failed-then-successful launch.
    """
    mock_page = AsyncMock()
    mock_browser = AsyncMock()
    mock_browser.new_page.return_value = mock_page

    mock_pw = AsyncMock()
    if launch_side_effect is None:
        mock_pw.chromium.launch.return_value = mock_browser
    else:
        mock_pw.chromium.launch.side_effect = launch_side_effect

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


# ---------------------------------------------------------------------------
# Chromium auto-install
#
# The browser is only ever installed in reaction to a launch that actually
# failed for a missing binary. Any other outcome — a successful launch, or a
# launch that failed for another reason — must not shell out to the network.
# ---------------------------------------------------------------------------


class TestChromiumAutoInstall:
    """Auto-install is reactive, at-most-once, and never masks other errors."""

    @pytest.fixture(autouse=True)
    def _install_calls(self, monkeypatch, tmp_path):
        """Record install subprocesses instead of running them; reset the cache."""
        calls: list[list[str]] = []

        def _fake_run(cmd, **kwargs):
            calls.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(converter_mod.subprocess, "run", _fake_run)
        monkeypatch.setattr(converter_mod, "_install_attempted", False)
        monkeypatch.setattr(converter_mod, "_install_error", None)
        return calls

    @staticmethod
    def _html(tmp_path):
        html_file = tmp_path / "plot.html"
        html_file.write_text("<html></html>")
        (tmp_path / "plot.png").write_bytes(b"fake")
        return html_file, tmp_path / "plot.png"

    @pytest.mark.unit
    async def test_successful_launch_never_installs(self, tmp_path, _install_calls):
        """A browser that launches is a browser that is installed — no subprocess.

        Regression test for the sync-in-async availability probe: it raised inside
        a running event loop, was misread as "browser missing", and installed
        Chromium over the network on every single conversion.
        """
        html_file, output_file = self._html(tmp_path)
        mock_mod, _, _ = _make_playwright_mock()

        with patch.dict(sys.modules, {"playwright.async_api": mock_mod}):
            await convert_html_to_image(html_file, output_file)

        assert _install_calls == []

    @pytest.mark.unit
    async def test_missing_browser_installs_then_retries(self, tmp_path, _install_calls):
        """A launch that fails for a missing binary installs once and retries."""
        html_file, output_file = self._html(tmp_path)
        _, _, mock_browser = _make_playwright_mock()
        mock_mod, _, _ = _make_playwright_mock(
            launch_side_effect=[Exception(MISSING_BROWSER_ERROR), mock_browser]
        )

        with patch.dict(sys.modules, {"playwright.async_api": mock_mod}):
            result = await convert_html_to_image(html_file, output_file)

        assert result == output_file.resolve()
        assert _install_calls == [[sys.executable, "-m", "playwright", "install", "chromium"]]

    @pytest.mark.unit
    async def test_install_attempted_at_most_once_per_process(self, tmp_path, _install_calls):
        """A second conversion with a missing browser does not re-install."""
        html_file, output_file = self._html(tmp_path)
        mock_mod, _, _ = _make_playwright_mock(
            launch_side_effect=[Exception(MISSING_BROWSER_ERROR)] * 4
        )

        with patch.dict(sys.modules, {"playwright.async_api": mock_mod}):
            for _ in range(2):
                with pytest.raises(PlaywrightNotInstalledError):
                    await convert_html_to_image(html_file, output_file)

        assert len(_install_calls) == 1

    @pytest.mark.unit
    async def test_failed_install_reports_stderr(self, tmp_path, monkeypatch):
        """A nonzero install surfaces as PlaywrightNotInstalledError with its stderr."""
        html_file, output_file = self._html(tmp_path)

        def _failing_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="proxy blocked the CDN")

        monkeypatch.setattr(converter_mod.subprocess, "run", _failing_run)
        monkeypatch.setattr(converter_mod, "_install_attempted", False)
        monkeypatch.setattr(converter_mod, "_install_error", None)

        mock_mod, _, _ = _make_playwright_mock(
            launch_side_effect=[Exception(MISSING_BROWSER_ERROR)] * 2
        )
        with patch.dict(sys.modules, {"playwright.async_api": mock_mod}):
            with pytest.raises(PlaywrightNotInstalledError, match="proxy blocked the CDN"):
                await convert_html_to_image(html_file, output_file)

    @pytest.mark.unit
    async def test_unrelated_launch_failure_propagates(self, tmp_path, _install_calls):
        """A launch failure that is not a missing binary surfaces as itself."""
        html_file, output_file = self._html(tmp_path)
        mock_mod, _, _ = _make_playwright_mock(
            launch_side_effect=[
                Exception("browserType.launch: Host system is missing dependencies")
            ]
        )

        with patch.dict(sys.modules, {"playwright.async_api": mock_mod}):
            with pytest.raises(Exception, match="missing dependencies") as excinfo:
                await convert_html_to_image(html_file, output_file)

        assert not isinstance(excinfo.value, PlaywrightNotInstalledError)
        assert _install_calls == []
