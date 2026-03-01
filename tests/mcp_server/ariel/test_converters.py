"""Tests for artifact-to-attachment converter registry."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# _wrap_in_html
# ---------------------------------------------------------------------------


class TestWrapInHtml:
    """Tests for the HTML template wrapper."""

    def test_produces_valid_html(self):
        from osprey.mcp_server.ariel.converters import _wrap_in_html

        html = _wrap_in_html("Test Title", "<p>Hello</p>")
        assert "<!DOCTYPE html>" in html
        assert "<title>Test Title</title>" in html
        assert "<p>Hello</p>" in html
        assert "</html>" in html

    def test_includes_styling(self):
        from osprey.mcp_server.ariel.converters import _wrap_in_html

        html = _wrap_in_html("Styled", "<p>Content</p>")
        assert "<style>" in html
        assert "font-family" in html

    def test_body_content_preserved(self):
        from osprey.mcp_server.ariel.converters import _wrap_in_html

        body = "<h2>data.json</h2>\n<pre>{}</pre>"
        html = _wrap_in_html("data", body)
        assert body in html


# ---------------------------------------------------------------------------
# passthrough converter
# ---------------------------------------------------------------------------


class TestPassthrough:
    """Tests for the passthrough converter."""

    @pytest.mark.unit
    async def test_returns_source_unchanged(self, tmp_path):
        from osprey.mcp_server.ariel.converters import passthrough

        source = tmp_path / "image.png"
        source.write_bytes(b"\x89PNG fake")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = await passthrough(source, output_dir)
        assert result == source

    @pytest.mark.unit
    async def test_does_not_create_files(self, tmp_path):
        from osprey.mcp_server.ariel.converters import passthrough

        source = tmp_path / "doc.pdf"
        source.write_bytes(b"%PDF-1.4 fake")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        await passthrough(source, output_dir)
        assert list(output_dir.iterdir()) == []


# ---------------------------------------------------------------------------
# html_to_png converter
# ---------------------------------------------------------------------------


class TestHtmlToPng:
    """Tests for the HTML-to-PNG converter."""

    @pytest.mark.unit
    async def test_calls_convert_html_to_image(self, tmp_path):
        from osprey.mcp_server.ariel.converters import html_to_png

        source = tmp_path / "plot.html"
        source.write_text("<html><body>Plot</body></html>")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        async def fake_convert(html_path, output_path, **kwargs):
            Path(output_path).write_bytes(b"\x89PNG converted")
            return Path(output_path).resolve()

        with patch(
            "osprey.mcp_server.export.converter.convert_html_to_image",
            side_effect=fake_convert,
        ):
            result = await html_to_png(source, output_dir)

        assert result == output_dir / "plot.png"
        assert result.read_bytes() == b"\x89PNG converted"

    @pytest.mark.unit
    async def test_output_in_output_dir(self, tmp_path):
        from osprey.mcp_server.ariel.converters import html_to_png

        source = tmp_path / "page.html"
        source.write_text("<html></html>")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        async def fake_convert(html_path, output_path, **kwargs):
            Path(output_path).write_bytes(b"\x89PNG")
            return Path(output_path).resolve()

        with patch(
            "osprey.mcp_server.export.converter.convert_html_to_image",
            side_effect=fake_convert,
        ):
            result = await html_to_png(source, output_dir)

        assert result.parent == output_dir


# ---------------------------------------------------------------------------
# markdown_to_png converter
# ---------------------------------------------------------------------------


class TestMarkdownToPng:
    """Tests for the Markdown-to-PNG converter."""

    @pytest.mark.unit
    async def test_renders_markdown_to_png(self, tmp_path):
        from osprey.mcp_server.ariel.converters import markdown_to_png

        source = tmp_path / "readme.md"
        source.write_text("# Hello\n\nSome **bold** text.", encoding="utf-8")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        async def fake_convert(html_path, output_path, **kwargs):
            # Verify the intermediate HTML has rendered markdown
            html_content = Path(html_path).read_text()
            assert "<h1>" in html_content or "<h1" in html_content
            assert "<strong>bold</strong>" in html_content
            Path(output_path).write_bytes(b"\x89PNG md")
            return Path(output_path).resolve()

        with patch(
            "osprey.mcp_server.export.converter.convert_html_to_image",
            side_effect=fake_convert,
        ):
            result = await markdown_to_png(source, output_dir)

        assert result == output_dir / "readme.png"

    @pytest.mark.unit
    async def test_intermediate_html_has_styling(self, tmp_path):
        from osprey.mcp_server.ariel.converters import markdown_to_png

        source = tmp_path / "notes.md"
        source.write_text("Some notes", encoding="utf-8")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        captured_html = {}

        async def fake_convert(html_path, output_path, **kwargs):
            captured_html["content"] = Path(html_path).read_text()
            Path(output_path).write_bytes(b"\x89PNG")
            return Path(output_path).resolve()

        with patch(
            "osprey.mcp_server.export.converter.convert_html_to_image",
            side_effect=fake_convert,
        ):
            await markdown_to_png(source, output_dir)

        assert "<!DOCTYPE html>" in captured_html["content"]
        assert "font-family" in captured_html["content"]


# ---------------------------------------------------------------------------
# notebook_to_png converter
# ---------------------------------------------------------------------------


class TestNotebookToPng:
    """Tests for the Notebook-to-PNG converter."""

    @pytest.mark.unit
    async def test_renders_notebook(self, tmp_path):
        from osprey.mcp_server.ariel.converters import notebook_to_png

        source = tmp_path / "analysis.ipynb"
        source.write_text('{"cells": [], "metadata": {}}', encoding="utf-8")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        async def fake_convert(html_path, output_path, **kwargs):
            Path(output_path).write_bytes(b"\x89PNG nb")
            return Path(output_path).resolve()

        with (
            patch(
                "osprey.mcp_server.notebook_renderer.render_notebook_to_html",
                return_value="<html><body>Notebook</body></html>",
            ),
            patch(
                "osprey.mcp_server.export.converter.convert_html_to_image",
                side_effect=fake_convert,
            ),
        ):
            result = await notebook_to_png(source, output_dir)

        assert result == output_dir / "analysis.png"

    @pytest.mark.unit
    async def test_writes_intermediate_html(self, tmp_path):
        from osprey.mcp_server.ariel.converters import notebook_to_png

        source = tmp_path / "nb.ipynb"
        source.write_text("{}", encoding="utf-8")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        async def fake_convert(html_path, output_path, **kwargs):
            Path(output_path).write_bytes(b"\x89PNG")
            return Path(output_path).resolve()

        with (
            patch(
                "osprey.mcp_server.notebook_renderer.render_notebook_to_html",
                return_value="<html>NB</html>",
            ),
            patch(
                "osprey.mcp_server.export.converter.convert_html_to_image",
                side_effect=fake_convert,
            ),
        ):
            await notebook_to_png(source, output_dir)

        html_file = output_dir / "nb.html"
        assert html_file.exists()
        assert "NB" in html_file.read_text()


# ---------------------------------------------------------------------------
# json_to_png converter
# ---------------------------------------------------------------------------


class TestJsonToPng:
    """Tests for the JSON-to-PNG converter."""

    @pytest.mark.unit
    async def test_pretty_prints_json(self, tmp_path):
        from osprey.mcp_server.ariel.converters import json_to_png

        data = {"key": "value", "list": [1, 2, 3]}
        source = tmp_path / "data.json"
        source.write_text(json.dumps(data), encoding="utf-8")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        captured_html = {}

        async def fake_convert(html_path, output_path, **kwargs):
            captured_html["content"] = Path(html_path).read_text()
            Path(output_path).write_bytes(b"\x89PNG json")
            return Path(output_path).resolve()

        with patch(
            "osprey.mcp_server.export.converter.convert_html_to_image",
            side_effect=fake_convert,
        ):
            result = await json_to_png(source, output_dir)

        assert result == output_dir / "data.png"
        # Should have pretty-printed JSON in a <pre> tag
        assert '"key": "value"' in captured_html["content"]
        assert "<pre>" in captured_html["content"]

    @pytest.mark.unit
    async def test_handles_invalid_json_gracefully(self, tmp_path):
        from osprey.mcp_server.ariel.converters import json_to_png

        source = tmp_path / "broken.json"
        source.write_text("{not valid json", encoding="utf-8")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        async def fake_convert(html_path, output_path, **kwargs):
            html = Path(html_path).read_text()
            # Falls back to raw text
            assert "{not valid json" in html
            Path(output_path).write_bytes(b"\x89PNG")
            return Path(output_path).resolve()

        with patch(
            "osprey.mcp_server.export.converter.convert_html_to_image",
            side_effect=fake_convert,
        ):
            await json_to_png(source, output_dir)


# ---------------------------------------------------------------------------
# text_to_png converter
# ---------------------------------------------------------------------------


class TestTextToPng:
    """Tests for the text-to-PNG converter."""

    @pytest.mark.unit
    async def test_wraps_in_pre(self, tmp_path):
        from osprey.mcp_server.ariel.converters import text_to_png

        source = tmp_path / "log.txt"
        source.write_text("Line 1\nLine 2\n", encoding="utf-8")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        captured_html = {}

        async def fake_convert(html_path, output_path, **kwargs):
            captured_html["content"] = Path(html_path).read_text()
            Path(output_path).write_bytes(b"\x89PNG txt")
            return Path(output_path).resolve()

        with patch(
            "osprey.mcp_server.export.converter.convert_html_to_image",
            side_effect=fake_convert,
        ):
            result = await text_to_png(source, output_dir)

        assert result == output_dir / "log.png"
        assert "<pre>" in captured_html["content"]
        assert "Line 1\nLine 2" in captured_html["content"]

    @pytest.mark.unit
    async def test_escapes_html_entities(self, tmp_path):
        from osprey.mcp_server.ariel.converters import text_to_png

        source = tmp_path / "code.py"
        source.write_text("if x < 10 & y > 5:\n    pass", encoding="utf-8")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        captured_html = {}

        async def fake_convert(html_path, output_path, **kwargs):
            captured_html["content"] = Path(html_path).read_text()
            Path(output_path).write_bytes(b"\x89PNG")
            return Path(output_path).resolve()

        with patch(
            "osprey.mcp_server.export.converter.convert_html_to_image",
            side_effect=fake_convert,
        ):
            await text_to_png(source, output_dir)

        assert "&lt;" in captured_html["content"]
        assert "&amp;" in captured_html["content"]
        assert "&gt;" in captured_html["content"]


# ---------------------------------------------------------------------------
# get_converter registry lookup
# ---------------------------------------------------------------------------


class TestGetConverter:
    """Tests for the converter registry lookup."""

    def test_known_mime_types(self):
        from osprey.mcp_server.ariel.converters import (
            get_converter,
            html_to_png,
            json_to_png,
            markdown_to_png,
            notebook_to_png,
            passthrough,
            text_to_png,
        )

        assert get_converter("image/png") is passthrough
        assert get_converter("image/jpeg") is passthrough
        assert get_converter("application/pdf") is passthrough
        assert get_converter("application/octet-stream") is passthrough
        assert get_converter("text/html") is html_to_png
        assert get_converter("text/markdown") is markdown_to_png
        assert get_converter("text/x-markdown") is markdown_to_png
        assert get_converter("application/x-ipynb+json") is notebook_to_png
        assert get_converter("application/json") is json_to_png
        assert get_converter("text/plain") is text_to_png
        assert get_converter("text/x-python") is text_to_png

    def test_unknown_image_type_passthrough(self):
        from osprey.mcp_server.ariel.converters import get_converter, passthrough

        assert get_converter("image/tiff") is passthrough
        assert get_converter("image/bmp") is passthrough

    def test_unknown_mime_type_falls_back_to_text(self):
        from osprey.mcp_server.ariel.converters import get_converter, text_to_png

        assert get_converter("application/x-custom") is text_to_png
        assert get_converter("video/mp4") is text_to_png
