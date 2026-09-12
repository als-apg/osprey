"""Tests for GET /api/notebooks/{id}/rendered endpoint.

The gallery serves rendered notebooks inside a sandboxed iframe, and nbconvert's
default template links MathJax / RequireJS / jQuery / widget / Mermaid assets
from public CDNs. Whether those links may stay is the deployment's own
``offline`` posture: on an isolated deployment the loads fail and the notebook
renders broken, so the HTML has to be fully self-contained; on a connected one
the same blanking silently drops inline LaTeX, interactive widgets and Mermaid
diagrams from every notebook. These tests pin the contract that the endpoint
renders the notebook's cells AND follows the posture in both directions.
"""

import re

import nbformat
import pytest

# Hosts nbconvert's default HTMLExporter template pulls assets from.
_CDN_HOSTS = ("cdnjs.cloudflare.com", "unpkg.com")


def _make_notebook_bytes() -> bytes:
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_markdown_cell("# Demo Notebook\n\nInline math $x^2$."),
        nbformat.v4.new_code_cell("print('UNIQUE_CELL_TOKEN')"),
    ]
    return nbformat.writes(nb).encode()


class TestNotebookRenderedAPI:
    @pytest.fixture
    def app_client(self, tmp_path):
        from fastapi.testclient import TestClient

        from osprey.interfaces.artifacts.app import create_app

        app = create_app(workspace_root=tmp_path)
        return TestClient(app), tmp_path

    def _save_notebook(self, client, content: bytes | None = None):
        store = client.app.state.artifact_store
        return store.save_file(
            file_content=content or _make_notebook_bytes(),
            filename="demo.ipynb",
            artifact_type="notebook",
            title="Demo Notebook",
            description="demo",
            mime_type="application/x-ipynb+json",
            tool_source="test",
        )

    @pytest.mark.unit
    def test_renders_notebook_cells_to_html(self, app_client):
        """A notebook artifact renders its cell content to an HTML page."""
        client, _ = app_client
        entry = self._save_notebook(client)

        resp = client.get(f"/api/notebooks/{entry.id}/rendered")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        # Code cell content is present (the render actually happened, and the
        # token survives syntax-highlight span-splitting because it is a single
        # identifier).
        assert "UNIQUE_CELL_TOKEN" in resp.text

    @pytest.mark.unit
    def test_an_offline_deployments_render_is_self_contained(self, app_client, monkeypatch):
        """Rendered HTML must not reference external CDN assets when isolated.

        Where external hosts are unreachable those loads fail and the notebook
        renders broken inside the gallery's sandboxed iframe.
        """
        monkeypatch.setenv("OSPREY_OFFLINE", "1")
        client, _ = app_client
        entry = self._save_notebook(client)

        html = client.get(f"/api/notebooks/{entry.id}/rendered").text

        # No src=/href= pointing at an external http(s) resource.
        external = re.findall(r'(?:src|href)\s*=\s*["\'](https?://[^"\']+)["\']', html)
        assert external == [], f"external resource refs leaked: {sorted(set(external))}"

        # Belt-and-suspenders: the known CDN hosts must not appear anywhere.
        for host in _CDN_HOSTS:
            assert host not in html, f"rendered notebook references CDN host {host}"

    @pytest.mark.unit
    def test_a_connected_deployments_render_keeps_its_assets(self, app_client, monkeypatch):
        """A deployment that can reach the CDNs gets the fuller document.

        Blanking them here is not free: it is what makes the inline LaTeX in
        this notebook's markdown cell render as raw ``$x^2$``.
        """
        monkeypatch.setenv("OSPREY_OFFLINE", "0")
        client, _ = app_client
        entry = self._save_notebook(client)

        html = client.get(f"/api/notebooks/{entry.id}/rendered").text

        assert "cdnjs.cloudflare.com" in html

    @pytest.mark.unit
    def test_non_notebook_returns_400(self, app_client):
        client, _ = app_client
        store = client.app.state.artifact_store
        entry = store.save_file(
            file_content=b"# not a notebook",
            filename="x.md",
            artifact_type="markdown",
            title="md",
            description="",
            mime_type="text/markdown",
            tool_source="test",
        )
        resp = client.get(f"/api/notebooks/{entry.id}/rendered")
        assert resp.status_code == 400

    @pytest.mark.unit
    def test_missing_artifact_returns_404(self, app_client):
        client, _ = app_client
        resp = client.get("/api/notebooks/nonexistent-id/rendered")
        assert resp.status_code == 404

    @pytest.mark.unit
    def test_notebook_file_missing_on_disk_returns_404(self, app_client):
        """A registered entry whose .ipynb vanished from disk returns 404, not 500."""
        client, _ = app_client
        entry = self._save_notebook(client)
        client.app.state.artifact_store.get_file_path(entry.id).unlink()

        resp = client.get(f"/api/notebooks/{entry.id}/rendered")
        assert resp.status_code == 404
        assert "not found on disk" in resp.json()["detail"]

    @pytest.mark.unit
    def test_unparseable_notebook_returns_500_with_detail(self, app_client):
        """A file that nbformat cannot parse surfaces as a 500 with the render
        error in the detail, rather than an unhandled exception."""
        client, _ = app_client
        entry = self._save_notebook(client, content=b"this is not JSON at all {")

        resp = client.get(f"/api/notebooks/{entry.id}/rendered")
        assert resp.status_code == 500
        assert "Notebook rendering failed" in resp.json()["detail"]
