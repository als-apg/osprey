"""Tests for GET /api/markdown/{id}/rendered endpoint.

Verifies that the markdown rendered endpoint returns a standalone HTML page
with CDN links, proper escaping, and correct error responses.
"""

import pytest


class TestMarkdownRenderedAPI:
    """Tests for the markdown rendered endpoint."""

    @pytest.fixture
    def app_client(self, tmp_path):
        from fastapi.testclient import TestClient

        from osprey.interfaces.artifacts.app import create_app

        app = create_app(workspace_root=tmp_path)
        return TestClient(app), tmp_path

    @pytest.mark.unit
    def test_renders_markdown_to_html_page(self, app_client):
        """Markdown artifact returns a full HTML page with CDN links."""
        client, workspace = app_client
        store = client.app.state.artifact_store

        entry = store.save_file(
            file_content=b"# Hello World\n\nSome **bold** text.",
            filename="test.md",
            artifact_type="markdown",
            title="Test Markdown",
            description="test",
            mime_type="text/markdown",
            tool_source="test",
        )

        resp = client.get(f"/api/markdown/{entry.id}/rendered")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

        html = resp.text
        assert "marked@12.0.2" in html
        assert "katex@0.16.11" in html
        assert "highlightjs" in html
        assert "osprey-md-rendered" in html
        assert "md-source" in html

    @pytest.mark.unit
    def test_non_markdown_returns_400(self, app_client):
        """Non-markdown artifact type returns 400."""
        client, workspace = app_client
        store = client.app.state.artifact_store

        entry = store.save_file(
            file_content=b"<html><body>Hello</body></html>",
            filename="test.html",
            artifact_type="html",
            title="HTML artifact",
            description="not markdown",
            mime_type="text/html",
            tool_source="test",
        )

        resp = client.get(f"/api/markdown/{entry.id}/rendered")
        assert resp.status_code == 400
        assert "not a markdown" in resp.json()["detail"]

    @pytest.mark.unit
    def test_missing_artifact_returns_404(self, app_client):
        """Non-existent artifact ID returns 404."""
        client, workspace = app_client

        resp = client.get("/api/markdown/nonexistent-id/rendered")
        assert resp.status_code == 404

    @pytest.mark.unit
    def test_script_tag_in_markdown_is_escaped(self, app_client):
        """Markdown containing </script> is safely escaped in the JSON embed."""
        client, workspace = app_client
        store = client.app.state.artifact_store

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

        resp = client.get(f"/api/markdown/{entry.id}/rendered")
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
