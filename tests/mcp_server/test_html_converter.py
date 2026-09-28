"""Tests for the HTML-to-image converter module."""

import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
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


class FakePlaywrightTimeoutError(Exception):
    """Stand-in for ``playwright.async_api.TimeoutError`` in the fake module."""


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
    mod.TimeoutError = FakePlaywrightTimeoutError  # type: ignore[attr-defined]

    return mod, mock_page, mock_browser


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


async def test_playwright_not_installed(tmp_path):
    """Missing playwright module raises PlaywrightNotInstalledError."""
    html_file = tmp_path / "plot.html"
    html_file.write_text("<html></html>")
    output_file = tmp_path / "plot.png"

    # Remove playwright from sys.modules to trigger ImportError
    with patch.dict(sys.modules, {"playwright.async_api": None, "playwright": None}):
        with pytest.raises(PlaywrightNotInstalledError, match="not installed"):
            await convert_html_to_image(html_file, output_file)


async def test_invalid_format(tmp_path):
    """Unsupported format raises ValueError."""
    html_file = tmp_path / "plot.html"
    html_file.write_text("<html></html>")

    with pytest.raises(ValueError, match="Unsupported format"):
        await convert_html_to_image(html_file, tmp_path / "out.bmp", fmt="bmp")


async def test_file_not_found(tmp_path):
    """Non-existent HTML file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="not found"):
        await convert_html_to_image(tmp_path / "missing.html", tmp_path / "out.png")


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
    def _install_calls(self, monkeypatch):
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


# ---------------------------------------------------------------------------
# _plotly_script_src — where the rendered page loads Plotly from
# ---------------------------------------------------------------------------


def test_plotly_script_src_uses_vendored_file_when_present(tmp_path, monkeypatch):
    """A checkout that tracks the vendored bundle gets a file:// URI to it."""
    bundle = tmp_path / "plotly-3.3.1.min.js"
    bundle.write_text("/* plotly */")
    monkeypatch.setattr(converter_mod, "_plotly_vendor_path", lambda: bundle)

    src = converter_mod._plotly_script_src()

    assert src == bundle.resolve().as_uri()
    assert src.startswith("file://")


def test_plotly_script_src_falls_back_to_cdn_when_vendor_missing(tmp_path, monkeypatch):
    """A wheel install (no vendor dir) gets the absolute CDN URL, never a relative path."""
    from osprey.interfaces.vendor import asset_cdn_url

    monkeypatch.setattr(
        converter_mod, "_plotly_vendor_path", lambda: tmp_path / "absent" / "plotly.min.js"
    )
    # Offline mode must not turn the fallback into a server-relative path.
    monkeypatch.setenv("OSPREY_OFFLINE", "1")

    src = converter_mod._plotly_script_src()

    assert src == asset_cdn_url("Plotly.js")
    assert src.startswith("https://")


def test_plotly_script_src_vendor_path_points_at_interfaces_bundle():
    """The vendored path is the artifacts bundle under osprey.interfaces."""
    import osprey.interfaces

    expected = (
        Path(osprey.interfaces.__file__).parent / "artifacts/static/js/vendor/plotly-3.3.1.min.js"
    )
    assert converter_mod._plotly_vendor_path() == expected


# ---------------------------------------------------------------------------
# Plotly injection — a figure exported without its library renders blank, so
# the converter renders a temp copy that loads Plotly and waits for the draw.
# ---------------------------------------------------------------------------

_PLOTLY_CALL = "<div id='g'></div><script>Plotly.newPlot('g', [{y: [1, 2]}]);</script>"
_FAKE_PLOTLY_SRC = "file:///vendor/plotly-3.3.1.min.js"


def _plot_page(head: str = "") -> str:
    return f"<html><head>{head}</head><body>{_PLOTLY_CALL}</body></html>"


class TestPlotlyInjection:
    """Injected render: temp copy, base href + script, load wait, plot-drawn wait."""

    @pytest.fixture(autouse=True)
    def _fixed_script_src(self, monkeypatch):
        monkeypatch.setattr(converter_mod, "_plotly_script_src", lambda: _FAKE_PLOTLY_SRC)

    @staticmethod
    def _files(tmp_path, html: str):
        src_dir = tmp_path / "artifacts"
        src_dir.mkdir()
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        html_file = src_dir / "plot.html"
        html_file.write_text(html)
        return html_file, out_dir / "plot.png"

    async def _run(self, tmp_path, mock_page, html=None, mock_mod=None):
        html_file, output_file = self._files(tmp_path, html or _plot_page())
        seen: dict = {}

        async def _goto(url, **kwargs):
            path = Path(url.removeprefix("file://"))
            seen["url"] = url
            seen["kwargs"] = kwargs
            seen["path"] = path
            seen["content"] = path.read_text() if path.exists() else None

        if mock_page.goto.side_effect is None:
            mock_page.goto.side_effect = _goto
        with patch.dict(sys.modules, {"playwright.async_api": mock_mod, "playwright": MagicMock()}):
            result = await convert_html_to_image(html_file, output_file)
        return html_file, output_file, result, seen

    async def test_injected_copy_loads_plotly_and_base_href(self, tmp_path):
        mock_mod, mock_page, _ = _make_playwright_mock()
        html_file, output_file, result, seen = await self._run(
            tmp_path, mock_page, mock_mod=mock_mod
        )

        assert result == output_file.resolve()
        # Rendered a copy in the destination directory, not the artifact itself.
        assert seen["path"] != html_file
        assert seen["path"].parent == output_file.parent
        assert seen["path"].suffix == ".html"
        content = seen["content"]
        base = f'<base href="{html_file.parent.as_uri()}/">'
        script = f'<script src="{_FAKE_PLOTLY_SRC}"></script>'
        # Both land right after <head>, before the page's own content.
        assert content.startswith("<html><head>" + base)
        assert base + script in content
        assert content.index(script) < content.index("Plotly.newPlot")
        assert seen["kwargs"] == {"wait_until": "load", "timeout": 10_000}

    async def test_injected_render_waits_for_drawn_plot(self, tmp_path):
        mock_mod, mock_page, _ = _make_playwright_mock()
        await self._run(tmp_path, mock_page, mock_mod=mock_mod)

        mock_page.wait_for_function.assert_called_once()
        expr = mock_page.wait_for_function.call_args[0][0]
        assert ".js-plotly-plot" in expr
        assert "_fullLayout" in expr
        assert "_transitioning" in expr
        timeout = mock_page.wait_for_function.call_args.kwargs["timeout"]
        assert 0 <= timeout <= 10_000
        # Two animation frames after the draw, before the screenshot.
        mock_page.evaluate.assert_called_once()
        assert mock_page.evaluate.call_args[0][0].count("requestAnimationFrame") == 2
        mock_page.screenshot.assert_called_once()

    async def test_temp_copy_removed_and_artifact_unchanged(self, tmp_path):
        mock_mod, mock_page, _ = _make_playwright_mock()
        original = _plot_page()
        html_file, output_file, _, seen = await self._run(tmp_path, mock_page, mock_mod=mock_mod)

        assert not seen["path"].exists()
        assert html_file.read_text() == original
        assert sorted(p.name for p in output_file.parent.iterdir()) == []

    async def test_plot_wait_timeout_warns_and_screenshots(self, tmp_path, caplog):
        mock_mod, mock_page, _ = _make_playwright_mock()
        mock_page.wait_for_function.side_effect = FakePlaywrightTimeoutError("timed out")

        with caplog.at_level("WARNING", logger="osprey.mcp_server.export.converter"):
            _, _, _, seen = await self._run(tmp_path, mock_page, mock_mod=mock_mod)

        mock_page.screenshot.assert_called_once()
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1
        assert not seen["path"].exists()

    async def test_goto_timeout_warns_and_screenshots(self, tmp_path, caplog):
        mock_mod, mock_page, _ = _make_playwright_mock()
        mock_page.goto.side_effect = FakePlaywrightTimeoutError("load timed out")

        with caplog.at_level("WARNING", logger="osprey.mcp_server.export.converter"):
            _, output_file, _, _ = await self._run(tmp_path, mock_page, mock_mod=mock_mod)

        mock_page.screenshot.assert_called_once()
        assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 1
        # No temp copy left behind even though the load timed out.
        assert list(output_file.parent.glob("*.html")) == []

    async def test_non_timeout_error_still_removes_copy(self, tmp_path):
        mock_mod, mock_page, _ = _make_playwright_mock()
        mock_page.wait_for_function.side_effect = RuntimeError("page crashed")

        with pytest.raises(RuntimeError, match="page crashed"):
            await self._run(tmp_path, mock_page, mock_mod=mock_mod)

        assert list((tmp_path / "out").glob("*.html")) == []

    async def test_page_with_bundle_is_not_injected(self, tmp_path):
        """A page that loads Plotly itself keeps the unchanged networkidle path."""
        mock_mod, mock_page, _ = _make_playwright_mock()
        html = _plot_page('<script src="https://cdn.plot.ly/plotly-3.3.1.min.js"></script>')
        html_file, _, _, seen = await self._run(tmp_path, mock_page, html=html, mock_mod=mock_mod)

        assert seen["path"] == html_file
        assert seen["kwargs"] == {"wait_until": "networkidle"}
        mock_page.wait_for_function.assert_not_called()

    async def test_page_without_head_injects_after_html(self, tmp_path):
        mock_mod, mock_page, _ = _make_playwright_mock()
        html = f"<html><body>{_PLOTLY_CALL}</body></html>"
        _, _, _, seen = await self._run(tmp_path, mock_page, html=html, mock_mod=mock_mod)

        assert seen["content"].startswith("<html><base href=")
        assert f'<script src="{_FAKE_PLOTLY_SRC}"></script><body>' in seen["content"]

    async def test_exhausted_budget_never_disables_the_plot_wait(self, tmp_path, monkeypatch):
        """A slow load leaves a positive plot-wait timeout (0 means wait forever)."""
        mock_mod, mock_page, _ = _make_playwright_mock()
        ticks = iter([0.0, 11.0])
        # Replace only the converter's clock; the event loop keeps the real one.
        monkeypatch.setattr(converter_mod, "time", SimpleNamespace(monotonic=lambda: next(ticks)))

        await self._run(tmp_path, mock_page, mock_mod=mock_mod)

        assert mock_page.wait_for_function.call_args.kwargs["timeout"] == 1
