"""Tests for artifact gallery app endpoints and helpers not covered elsewhere.

Covers:
  - GET/DELETE /api/artifacts/{id} (single-entry fetch and deletion)
  - GET/POST /api/focus (focus state, stale-focus fallback, fullscreen flag)
  - the focus_state.txt file the CLI hook reads (pin/unpin/focus lifecycle)
  - GET /files/{id}/{filename} for HTML artifacts (Plotly injection, offline
    CDN rewriting, missing-file 404s)
  - the _SSEBroadcaster queue bookkeeping
  - config.yml priming in create_app and run_server's uvicorn handoff
"""

import asyncio

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.artifacts.app import (
    _inject_html_snippet,
    _SSEBroadcaster,
    create_app,
    run_server,
)

pytestmark = pytest.mark.unit


def _save_text_artifact(store, title="Plain text"):
    return store.save_file(
        file_content=b"hello",
        filename="note.txt",
        artifact_type="text",
        title=title,
        description="",
        mime_type="text/plain",
        tool_source="test",
    )


def _save_html_artifact(store, content: str):
    return store.save_file(
        file_content=content.encode(),
        filename="plot.html",
        artifact_type="plot_html",
        title="Plot",
        description="",
        mime_type="text/html",
        tool_source="test",
    )


@pytest.fixture
def app_client(tmp_path):
    return TestClient(create_app(workspace_root=tmp_path)), tmp_path


class TestArtifactEntryAPI:
    """GET and DELETE /api/artifacts/{artifact_id}."""

    def test_get_artifact_returns_entry(self, app_client):
        client, _ = app_client
        entry = _save_text_artifact(client.app.state.artifact_store, title="Fetch Me")

        resp = client.get(f"/api/artifacts/{entry.id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == entry.id
        assert resp.json()["title"] == "Fetch Me"

    def test_delete_route_runs_under_the_human_actor(self, app_client):
        """A gallery delete is a human action: the store-level activity
        listener must be able to tell it apart from an agent delete."""
        from osprey.stores.artifact_store import (
            current_artifact_mutation_actor,
            register_artifact_delete_listener,
            unregister_artifact_delete_listener,
        )

        client, _ = app_client
        entry = _save_text_artifact(client.app.state.artifact_store)

        actors = []

        def record_actor(_entry):
            actors.append(current_artifact_mutation_actor())

        register_artifact_delete_listener(record_actor)
        try:
            assert client.delete(f"/api/artifacts/{entry.id}").status_code == 200
        finally:
            unregister_artifact_delete_listener(record_actor)

        assert actors == ["human"]

    def test_delete_artifact_removes_entry(self, app_client):
        client, _ = app_client
        entry = _save_text_artifact(client.app.state.artifact_store)

        resp = client.delete(f"/api/artifacts/{entry.id}")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok", "artifact_id": entry.id}
        assert client.get(f"/api/artifacts/{entry.id}").status_code == 404
        assert client.delete(f"/api/artifacts/{entry.id}").status_code == 404


class TestFocus:
    """GET/POST /api/focus and the focus_state.txt file the CLI hook reads."""

    def test_focus_lifecycle(self, app_client):
        client, _ = app_client
        store = client.app.state.artifact_store
        assert client.get("/api/focus").json() == {"focused": False, "artifact": None}

        first = _save_text_artifact(store, title="First")
        _save_text_artifact(store, title="Second")
        assert client.post("/api/focus", json={"artifact_id": "nonexistent"}).status_code == 404

        resp = client.post("/api/focus", json={"artifact_id": first.id})
        assert resp.json() == {"status": "ok", "artifact_id": first.id}
        focus = client.get("/api/focus").json()
        assert focus["focused"] is True
        assert focus["artifact"]["id"] == first.id  # not the latest artifact

        resp = client.post("/api/focus", json={"artifact_id": first.id, "fullscreen": True})
        assert resp.status_code == 200
        assert resp.json()["artifact_id"] == first.id

    def test_stale_focus_cleared_and_falls_back_to_latest(self, app_client):
        """A focused artifact deleted out from under the app contributes no
        line to focus_state.txt; GET /api/focus clears the stale id and falls
        back to the newest remaining entry."""
        client, workspace = app_client
        store = client.app.state.artifact_store
        doomed = _save_text_artifact(store, title="Doomed")
        client.post("/api/focus", json={"artifact_id": doomed.id})
        store.delete_entry(doomed.id)
        survivor = _save_text_artifact(store, title="Survivor")

        # Rewriting the file while the stale id is still set omits the artifact line.
        client.post(f"/api/artifacts/{survivor.id}/pin", json={"pinned": True})
        text = (workspace / "focus_state.txt").read_text()
        assert "artifact:" not in text
        assert 'pinned:   "Survivor"' in text

        focus = client.get("/api/focus").json()
        assert focus["focused"] is False
        assert focus["artifact"]["id"] == survivor.id
        assert client.app.state.focused_artifact_id is None  # stale id was cleared

    def test_focus_file_pin_unpin_and_dedupe(self, app_client):
        """Pinning writes a pinned line; unpinning empties (not deletes) the
        file; an artifact both focused and pinned appears only as the focused
        line, not duplicated as a pinned line."""
        client, workspace = app_client
        entry = _save_text_artifact(client.app.state.artifact_store, title="Both")
        focus_file = workspace / "focus_state.txt"

        client.post(f"/api/artifacts/{entry.id}/pin", json={"pinned": True})
        assert 'pinned:   "Both"' in focus_file.read_text()

        client.post(f"/api/artifacts/{entry.id}/pin", json={"pinned": False})
        # The file is emptied, not deleted -- the hook treats both as "no focus".
        assert focus_file.read_text() == ""

        client.post(f"/api/artifacts/{entry.id}/pin", json={"pinned": True})
        client.post("/api/focus", json={"artifact_id": entry.id})
        text = focus_file.read_text()
        assert f'artifact: "Both" (id={entry.id})' in text
        assert "pinned:" not in text

    def test_delete_clears_focus_only_for_the_focused_artifact(self, tmp_path):
        """With the lifespan listeners registered, deleting another artifact
        must not disturb the focus; deleting the focused one clears the focus
        state and empties the file."""
        with TestClient(create_app(workspace_root=tmp_path)) as client:
            store = client.app.state.artifact_store
            focused = _save_text_artifact(store, title="Focused")
            other = _save_text_artifact(store, title="Other")
            client.post("/api/focus", json={"artifact_id": focused.id})

            assert client.delete(f"/api/artifacts/{other.id}").status_code == 200
            assert client.app.state.focused_artifact_id == focused.id
            focus = client.get("/api/focus").json()
            assert focus["focused"] is True
            assert focus["artifact"]["id"] == focused.id

            assert client.delete(f"/api/artifacts/{focused.id}").status_code == 200
            assert client.app.state.focused_artifact_id is None
            assert client.get("/api/focus").json() == {"focused": False, "artifact": None}
            assert (tmp_path / "focus_state.txt").read_text() == ""


class TestServeFileEndpoint:
    """GET /files/{artifact_id}/{filename}."""

    def test_missing_artifact_or_file_returns_404(self, app_client):
        client, _ = app_client
        assert client.get("/files/nonexistent/whatever.html").status_code == 404

        store = client.app.state.artifact_store
        entry = _save_text_artifact(store)
        store.get_file_path(entry.id).unlink()
        resp = client.get(f"/files/{entry.id}/{entry.filename}")
        assert resp.status_code == 404
        assert "not found on disk" in resp.json()["detail"]

    def test_plotly_bundle_injected_only_when_absent(self, app_client, monkeypatch):
        """A plot_html artifact saved with include_plotlyjs=False gets the
        local Plotly bundle plus the responsive snippet injected (offline mode
        pins the injected src to the local vendor path); one already carrying
        the bundle must not get a second 4.8MB script tag."""
        client, _ = app_client
        store = client.app.state.artifact_store

        monkeypatch.setenv("OSPREY_OFFLINE", "1")
        bare = _save_html_artifact(
            store, '<html><head></head><body><div class="plotly-graph-div"></div></body></html>'
        )
        resp = client.get(f"/files/{bare.id}/{bare.filename}")
        assert resp.status_code == 200
        assert resp.headers["content-disposition"].startswith("inline")
        assert 'src="/static/js/vendor/plotly-3.3.1.min.js"' in resp.text
        assert ".js-plotly-plot" in resp.text  # responsive/theming snippet

        monkeypatch.setenv("OSPREY_OFFLINE", "0")
        bundled = _save_html_artifact(
            store,
            "<html><head>"
            '<script src="/static/js/vendor/plotly-3.3.1.min.js"></script>'
            "</head><body></body></html>",
        )
        resp = client.get(f"/files/{bundled.id}/{bundled.filename}")
        assert resp.status_code == 200
        assert resp.text.count("plotly-3.3.1.min.js") == 1
        assert ".js-plotly-plot" in resp.text  # snippet still injected

    def test_offline_rewrites_cdn_plotly_to_local(self, app_client, monkeypatch):
        """In offline mode a CDN Plotly reference is rewritten to the bundled
        copy and its SRI attributes are stripped (the local file may differ
        from the CDN version, and SRI would block it)."""
        monkeypatch.setenv("OSPREY_OFFLINE", "1")
        client, _ = app_client
        entry = _save_html_artifact(
            client.app.state.artifact_store,
            "<html><head>"
            '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"'
            ' integrity="sha384-abc123" crossorigin="anonymous"></script>'
            "</head><body></body></html>",
        )

        resp = client.get(f"/files/{entry.id}/{entry.filename}")
        assert resp.status_code == 200
        body = resp.text
        assert "cdn.plot.ly" not in body
        assert "integrity=" not in body
        assert "crossorigin" not in body
        # The rewritten reference also satisfies the dedup check -- no second copy.
        assert body.count("plotly-3.3.1.min.js") == 1


class TestServedPlotThemeBridge:
    """The theme bridge injected into served Plotly artifacts.

    Regression pins for the dark-locked-artifact bug: the bridge used to carry
    its own two-entry ``{dark, light}`` hex palette and resolve the theme by
    reading ``window.parent``'s ``data-theme`` with a hardcoded ``'dark'``
    fallback -- so a plot opened in its own tab (no parent) always rendered
    dark, and any theme outside the ``main`` family (``high-contrast-light``,
    ``retro-light``, ...) fell to the dark palette even inside the gallery.
    The served page must instead resolve its theme the way every other
    design-system page does (theme-boot.js pre-paint, theme-manager.js at
    runtime) and take every color from the ``--chart-*`` tokens.
    """

    def test_bridge_loads_design_system_instead_of_reading_parent(self, app_client):
        client, _ = app_client
        entry = _save_html_artifact(
            client.app.state.artifact_store,
            '<html><head></head><body><div class="plotly-graph-div"></div></body></html>',
        )
        body = client.get(f"/files/{entry.id}/{entry.filename}").text
        assert '/design-system/js/theme-boot.js"' in body
        assert '/design-system/css/tokens.css"' in body
        assert "/design-system/js/theme-manager.js" in body
        assert "window.parent" not in body

    def test_snippets_carry_no_hardcoded_palette(self):
        """No injected snippet may carry a color literal or a baked theme id --
        the hygiene scanner cannot see HTML/CSS/JS inside Python strings, so
        this is the guard that keeps the old two-entry hex map from coming
        back in any of them."""
        import re

        from osprey.interfaces.artifacts.app import _RESPONSIVE_SNIPPETS

        for artifact_type, snippet in _RESPONSIVE_SNIPPETS.items():
            literals = re.findall(r"#[0-9a-fA-F]{3,8}\b|\brgba?\(|\bhsla?\(", snippet)
            assert literals == [], f"color literals in the {artifact_type} snippet: {literals}"
            assert "'dark'" not in snippet and '"dark"' not in snippet

    def test_html_and_table_artifacts_get_design_system_wiring(self, app_client):
        """table_html / html / dashboard_html pages join the design system too:
        theme-boot pre-paint, tokens.css, follower role -- an unstyled table
        must not render as a black-on-white browser-default island."""
        client, _ = app_client
        store = client.app.state.artifact_store
        entry = store.save_file(
            file_content=b'<table class="artifact-table"><tr><td>1</td></tr></table>',
            filename="table.html",
            artifact_type="table_html",
            title="Table",
            description="",
            mime_type="text/html",
            tool_source="test",
        )
        body = client.get(f"/files/{entry.id}/{entry.filename}").text
        assert '/design-system/js/theme-boot.js"' in body
        assert '/design-system/css/tokens.css"' in body
        assert "initTheme({ role: 'follower' })" in body
        assert "artifact-table" in body  # the DataFrame hook finally has CSS behind it

    def test_gallery_shell_stamps_pinned_web_theme(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OSPREY_WEB_THEME", "light")
        client = TestClient(create_app(workspace_root=tmp_path))
        body = client.get("/").text
        assert 'data-theme="light"' in body

        monkeypatch.setenv("OSPREY_WEB_THEME", "main")
        client = TestClient(create_app(workspace_root=tmp_path))
        body = client.get("/").text
        assert "data-theme=" not in body.split("<head>")[0]  # the <html> tag carries no pin

    def test_pinned_web_theme_is_server_rendered_on_served_html(self, tmp_path, monkeypatch):
        """A deployment that pinned ``web.theme`` (a concrete id) gets that id
        stamped as ``<html data-theme>`` on every served HTML artifact page, so
        a first visit with nothing in localStorage and no ``?theme=`` paints the
        configured theme -- the same server rung the web terminal renders."""
        monkeypatch.setenv("OSPREY_WEB_THEME", "high-contrast-light")
        client = TestClient(create_app(workspace_root=tmp_path))
        store = client.app.state.artifact_store
        plot = _save_html_artifact(store, "<html><head></head><body></body></html>")
        body = client.get(f"/files/{plot.id}/{plot.filename}").text
        assert 'data-theme="high-contrast-light"' in body

        md = store.save_file(
            file_content=b"# hi",
            filename="hi.md",
            artifact_type="markdown",
            title="hi",
            description="",
            mime_type="text/markdown",
            tool_source="test",
        )
        body = client.get(f"/api/markdown/{md.id}/rendered").text
        assert 'data-theme="high-contrast-light"' in body

    def test_family_web_theme_is_not_pinned_on_served_html(self, tmp_path, monkeypatch):
        """A bare family (``main``) states a palette but not a mode: no
        ``data-theme`` is stamped, so the page follows the viewer's own
        preference (localStorage / OS) exactly as an unpinned terminal does."""
        monkeypatch.setenv("OSPREY_WEB_THEME", "main")
        client = TestClient(create_app(workspace_root=tmp_path))
        plot = _save_html_artifact(
            client.app.state.artifact_store, "<html><head></head><body></body></html>"
        )
        body = client.get(f"/files/{plot.id}/{plot.filename}").text
        assert "data-theme=" not in body


def test_inject_snippet_without_head_or_body_prepends():
    result = _inject_html_snippet(b"<div>fragment</div>", "<style>s</style>")
    assert result == b"<style>s</style><div>fragment</div>"


def test_sse_broadcaster_queue_bookkeeping():
    """Full queues drop events rather than raise; unsubscribing unknown or
    already-removed queues is a no-op."""
    b = _SSEBroadcaster()
    slow_client = b.subscribe()
    while not slow_client.full():
        slow_client.put_nowait({"filler": True})
    capacity = slow_client.qsize()

    b.broadcast({"type": "artifact"})  # must not raise
    assert slow_client.qsize() == capacity  # dropped, not queued

    b.unsubscribe(asyncio.Queue())  # never subscribed: no-op
    b.unsubscribe(slow_client)
    b.unsubscribe(slow_client)  # double-unsubscribe must not raise either


class TestAppLifecycle:
    """create_app config priming and run_server's uvicorn handoff."""

    def test_config_yml_priming_is_best_effort(self, tmp_path, monkeypatch):
        """A config.yml in the workspace root is loaded at app construction
        (custom artifact categories come from it), while one the loader
        rejects must not take the gallery down with it."""
        import osprey.utils.config as config_module

        # Snapshot the config singletons; monkeypatch restores them at teardown
        # so priming the default config cannot leak into other tests.
        monkeypatch.setattr(config_module, "_default_config", config_module._default_config)
        monkeypatch.setattr(
            config_module, "_default_configurable", config_module._default_configurable
        )

        good = tmp_path / "good"
        good.mkdir()
        (good / "config.yml").write_text("project_name: artifacts-coverage-test\n")
        assert TestClient(create_app(workspace_root=good)).get("/health").status_code == 200

        bad = tmp_path / "bad"
        bad.mkdir()
        (bad / "config.yml").write_text(":\n  - not valid yaml: [unclosed\n")
        assert TestClient(create_app(workspace_root=bad)).get("/health").status_code == 200

    def test_run_server_builds_app_and_hands_it_to_uvicorn(self, tmp_path, monkeypatch):
        import uvicorn

        calls = {}

        def fake_run(app, **kwargs):
            calls["app"] = app
            calls.update(kwargs)

        monkeypatch.setattr(uvicorn, "run", fake_run)
        run_server(host="127.0.0.1", port=59999, workspace_root=tmp_path)

        assert calls["host"] == "127.0.0.1"
        assert calls["port"] == 59999
        assert calls["app"].title == "OSPREY Artifact Gallery"


class TestIndexPlotlyVendorMeta:
    """GET / carries the resolved Plotly URL for timeseries.js's lazy loader.

    Plotly is the one vendor asset loaded lazily from JS rather than from a
    template tag, so it cannot call the ``vendor_url`` Jinja global itself —
    index.html hands it the resolved URL in a meta tag instead. Without the
    tag, online (CDN-mode) deployments request the never-vendored local path,
    404, and every timeseries preview dies with "Failed to load Plotly".
    """

    def test_online_mode_carries_the_cdn_url(self, app_client, monkeypatch):
        from osprey.interfaces.vendor import asset_cdn_url

        monkeypatch.setenv("OSPREY_OFFLINE", "0")
        client, _ = app_client
        html = client.get("/").text
        assert f'name="osprey-vendor-plotly" content="{asset_cdn_url("Plotly.js")}"' in html

    def test_offline_mode_carries_the_vendored_path(self, app_client, monkeypatch):
        monkeypatch.setenv("OSPREY_OFFLINE", "1")
        client, _ = app_client
        html = client.get("/").text
        assert 'name="osprey-vendor-plotly" content="/static/js/vendor/plotly-3.3.1.min.js"' in html
