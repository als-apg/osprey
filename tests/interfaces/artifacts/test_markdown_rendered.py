"""Tests for GET /api/markdown/{id}/rendered endpoint.

Verifies that the markdown rendered endpoint returns a standalone HTML page
with CDN links, proper escaping, and correct error responses.
"""

import pytest


class TestMarkdownRenderedAPI:
    """Tests for the markdown rendered endpoint."""

    def test_renders_markdown_to_html_page(self, app_client):
        """Markdown artifact returns a full HTML page with CDN links."""
        store = app_client.app.state.artifact_store

        entry = store.save_file(
            file_content=b"# Hello World\n\nSome **bold** text.",
            filename="test.md",
            artifact_type="markdown",
            title="Test Markdown",
            description="test",
            mime_type="text/markdown",
            tool_source="test",
        )

        resp = app_client.get(f"/api/markdown/{entry.id}/rendered")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

        html = resp.text
        assert "marked.min.js" in html
        assert "katex.min.js" in html
        assert "highlight.min.js" in html
        assert "osprey-md-rendered" in html
        assert "md-source" in html

    def test_non_markdown_returns_400(self, app_client):
        """Non-markdown artifact type returns 400."""
        store = app_client.app.state.artifact_store

        entry = store.save_file(
            file_content=b"<html><body>Hello</body></html>",
            filename="test.html",
            artifact_type="html",
            title="HTML artifact",
            description="not markdown",
            mime_type="text/html",
            tool_source="test",
        )

        resp = app_client.get(f"/api/markdown/{entry.id}/rendered")
        assert resp.status_code == 400
        assert "not a markdown" in resp.json()["detail"]

    def test_missing_artifact_returns_404(self, app_client):
        """Non-existent artifact ID returns 404."""

        resp = app_client.get("/api/markdown/nonexistent-id/rendered")
        assert resp.status_code == 404

    def test_script_tag_in_markdown_is_escaped(self, app_client):
        """Markdown containing </script> is safely escaped in the JSON embed."""
        store = app_client.app.state.artifact_store

        evil_md = '# Title\n\n</script><script>alert("xss")</script>'
        entry = store.save_file(
            file_content=evil_md.encode(),
            filename="evil.md",
            artifact_type="markdown",
            title="Script Test",
            description="test escaping",
            mime_type="text/markdown",
            tool_source="test",
        )

        resp = app_client.get(f"/api/markdown/{entry.id}/rendered")
        assert resp.status_code == 200

        html = resp.text
        # The raw </script> must NOT appear unescaped in the page
        # (it would break the <script type="application/json"> tag)
        # Instead it should be escaped as <\/script> inside the JSON
        assert r"<\/script>" in html
        # The literal unescaped </script> should only appear as the
        # closing tags for the actual script elements, not in the JSON
        # content area between <script type="application/json"> and </script>
        json_block_start = html.index('id="md-source">')
        json_block_end = html.index("</script>", json_block_start)
        json_content = html[json_block_start:json_block_end]
        assert "</script>" not in json_content

    def test_markdown_file_missing_on_disk_returns_404(self, app_client):
        """A registered entry whose file vanished from disk returns 404, not 500."""
        store = app_client.app.state.artifact_store
        entry = store.save_file(
            file_content=b"# Gone",
            filename="gone.md",
            artifact_type="markdown",
            title="Gone Markdown",
            description="",
            mime_type="text/markdown",
            tool_source="test",
        )
        store.get_file_path(entry.id).unlink()

        resp = app_client.get(f"/api/markdown/{entry.id}/rendered")
        assert resp.status_code == 404
        assert "not found on disk" in resp.json()["detail"]


class TestMarkdownPageIsOpaque:
    """The rendered page shows agent-written markdown, so it runs on an opaque
    origin, loads the sanitizer md-render.js needs, and carries C-MSG."""

    @pytest.fixture
    def page(self, app_client):
        entry = app_client.app.state.artifact_store.save_file(
            file_content=b"# hi\n\n<script>fetch('/api/x')</script>",
            filename="hi.md",
            artifact_type="markdown",
            title="hi",
            description="",
            mime_type="text/markdown",
            tool_source="test",
        )
        return app_client.get(f"/api/markdown/{entry.id}/rendered")

    def test_the_rendered_page_carries_the_opaque_sandbox(self, page):
        from osprey.interfaces.opaque_origin import OPAQUE_CSP

        assert page.status_code == 200
        assert page.headers["content-security-policy"] == OPAQUE_CSP
        assert page.headers["x-content-type-options"] == "nosniff"

    def test_the_sanitizer_loads_before_the_renderer(self, page):
        body = page.text
        assert "purify.min.js" in body
        assert body.index("purify.min.js") < body.index("/static/js/md-render.js")

    def test_the_sanitizer_is_the_local_copy_offline(self, monkeypatch):
        from osprey.interfaces.artifacts.app import _build_markdown_page

        monkeypatch.setenv("OSPREY_OFFLINE", "1")
        assert '<script src="/static/js/vendor/purify.min.js"></script>' in (
            _build_markdown_page("x", "t")
        )

    def test_the_page_carries_the_listener_and_the_theme_follower(self, page):
        body = page.text
        assert "osprey-artifact-ready" in body
        assert "osprey-artifact-print" in body
        assert "window.opener || window.parent" in body
        assert "from '/design-system/js/theme-manager.js'" in body
        assert "initTheme({ role: 'follower' })" in body
