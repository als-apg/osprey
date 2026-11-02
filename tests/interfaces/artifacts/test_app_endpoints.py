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
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.artifacts.app import (
    _agent_artifact_dir,
    _inject_html_snippet,
    _SSEBroadcaster,
    create_app,
    run_server,
)
from osprey.interfaces.opaque_origin import INERT_TYPES, OPAQUE_CSP


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


class TestArtifactEntryAPI:
    """GET and DELETE /api/artifacts/{artifact_id}."""

    def test_get_artifact_returns_entry(self, app_client):
        entry = _save_text_artifact(app_client.app.state.artifact_store, title="Fetch Me")

        resp = app_client.get(f"/api/artifacts/{entry.id}")
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

        entry = _save_text_artifact(app_client.app.state.artifact_store)

        actors = []

        def record_actor(_entry):
            actors.append(current_artifact_mutation_actor())

        register_artifact_delete_listener(record_actor)
        try:
            assert app_client.delete(f"/api/artifacts/{entry.id}").status_code == 200
        finally:
            unregister_artifact_delete_listener(record_actor)

        assert actors == ["human"]

    def test_delete_artifact_removes_entry(self, app_client):
        entry = _save_text_artifact(app_client.app.state.artifact_store)

        resp = app_client.delete(f"/api/artifacts/{entry.id}")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok", "artifact_id": entry.id}
        assert app_client.get(f"/api/artifacts/{entry.id}").status_code == 404
        assert app_client.delete(f"/api/artifacts/{entry.id}").status_code == 404


class TestBulkDelete:
    """POST /api/artifacts/delete."""

    def test_bulk_delete_removes_the_listed_ids_and_reports_missing(self, app_client):
        store = app_client.app.state.artifact_store
        a = _save_text_artifact(store, title="A")
        b = _save_text_artifact(store, title="B")
        keep = _save_text_artifact(store, title="Keep")

        resp = app_client.post("/api/artifacts/delete", json={"ids": [a.id, b.id, "nope"]})
        assert resp.status_code == 200
        assert resp.json() == {"deleted": [a.id, b.id], "missing": ["nope"]}
        assert app_client.get(f"/api/artifacts/{a.id}").status_code == 404
        assert app_client.get(f"/api/artifacts/{b.id}").status_code == 404
        assert app_client.get(f"/api/artifacts/{keep.id}").status_code == 200

    def test_bulk_delete_reports_a_repeated_missing_id_once(self, app_client):
        resp = app_client.post("/api/artifacts/delete", json={"ids": ["x", "y", "x"]})
        assert resp.status_code == 200
        assert resp.json() == {"deleted": [], "missing": ["x", "y"]}

    def test_bulk_delete_runs_under_the_human_actor(self, app_client):
        from osprey.stores.artifact_store import (
            current_artifact_mutation_actor,
            register_artifact_delete_listener,
            unregister_artifact_delete_listener,
        )

        store = app_client.app.state.artifact_store
        a = _save_text_artifact(store)
        b = _save_text_artifact(store)

        actors = []

        def record_actor(_entry):
            actors.append(current_artifact_mutation_actor())

        register_artifact_delete_listener(record_actor)
        try:
            resp = app_client.post("/api/artifacts/delete", json={"ids": [a.id, b.id]})
            assert resp.status_code == 200
        finally:
            unregister_artifact_delete_listener(record_actor)

        assert actors == ["human", "human"]

    def test_bulk_delete_clears_focus_on_a_deleted_artifact(self, tmp_path):
        with TestClient(create_app(workspace_root=tmp_path)) as client:
            store = client.app.state.artifact_store
            focused = _save_text_artifact(store, title="Focused")
            other = _save_text_artifact(store, title="Other")
            client.post("/api/focus", json={"artifact_id": focused.id})

            resp = client.post("/api/artifacts/delete", json={"ids": [focused.id, other.id]})
            assert resp.status_code == 200
            assert client.app.state.focused_artifact_id is None
            assert client.get("/api/focus").json() == {"focused": False, "artifact": None}
            assert (tmp_path / "focus_state.txt").read_text() == ""

    def test_bulk_delete_refuses_an_empty_list(self, app_client):
        assert app_client.post("/api/artifacts/delete", json={"ids": []}).status_code == 422

    def test_bulk_delete_refuses_more_than_the_cap(self, app_client):
        ids = [f"id{i}" for i in range(1001)]
        assert app_client.post("/api/artifacts/delete", json={"ids": ids}).status_code == 422


class TestFocus:
    """GET/POST /api/focus and the focus_state.txt file the CLI hook reads."""

    def test_focus_lifecycle(self, app_client):
        store = app_client.app.state.artifact_store
        assert app_client.get("/api/focus").json() == {"focused": False, "artifact": None}

        first = _save_text_artifact(store, title="First")
        _save_text_artifact(store, title="Second")
        assert app_client.post("/api/focus", json={"artifact_id": "nonexistent"}).status_code == 404

        resp = app_client.post("/api/focus", json={"artifact_id": first.id})
        assert resp.json() == {"status": "ok", "artifact_id": first.id}
        focus = app_client.get("/api/focus").json()
        assert focus["focused"] is True
        assert focus["artifact"]["id"] == first.id  # not the latest artifact

        resp = app_client.post("/api/focus", json={"artifact_id": first.id, "fullscreen": True})
        assert resp.status_code == 200
        assert resp.json()["artifact_id"] == first.id

    @pytest.mark.parametrize(
        ("body", "flag"),
        [({}, None), ({"automatic": True}, True)],
        ids=["chosen", "automatic"],
    )
    def test_a_focus_event_says_whether_it_was_automatic(self, app_client, body, flag):
        """A listener may treat a focus its sender made on its own (the approval
        hook) unlike a chosen one, so the broadcast must carry the flag exactly
        when the sender set it."""
        store = app_client.app.state.artifact_store
        entry = _save_text_artifact(store, title="Review")
        with patch.object(_SSEBroadcaster, "broadcast") as broadcast:
            app_client.post("/api/focus", json={"artifact_id": entry.id, **body})
        (event,), _ = broadcast.call_args
        assert event.get("automatic") is flag

    def test_stale_focus_cleared_and_falls_back_to_latest(self, app_client, tmp_path):
        """A focused artifact deleted out from under the app contributes no
        line to focus_state.txt; GET /api/focus clears the stale id and falls
        back to the newest remaining entry."""
        store = app_client.app.state.artifact_store
        doomed = _save_text_artifact(store, title="Doomed")
        app_client.post("/api/focus", json={"artifact_id": doomed.id})
        store.delete_entry(doomed.id)
        survivor = _save_text_artifact(store, title="Survivor")

        # Rewriting the file while the stale id is still set omits the artifact line.
        app_client.post(f"/api/artifacts/{survivor.id}/pin", json={"pinned": True})
        text = (tmp_path / "focus_state.txt").read_text()
        assert "artifact:" not in text
        assert 'pinned:   "Survivor"' in text

        focus = app_client.get("/api/focus").json()
        assert focus["focused"] is False
        assert focus["artifact"]["id"] == survivor.id
        assert app_client.app.state.focused_artifact_id is None  # stale id was cleared

    def test_focus_file_pin_unpin_and_dedupe(self, app_client, tmp_path):
        """Pinning writes a pinned line; unpinning empties (not deletes) the
        file; an artifact both focused and pinned appears only as the focused
        line, not duplicated as a pinned line."""
        entry = _save_text_artifact(app_client.app.state.artifact_store, title="Both")
        focus_file = tmp_path / "focus_state.txt"

        app_client.post(f"/api/artifacts/{entry.id}/pin", json={"pinned": True})
        assert 'pinned:   "Both"' in focus_file.read_text()

        app_client.post(f"/api/artifacts/{entry.id}/pin", json={"pinned": False})
        # The file is emptied, not deleted -- the hook treats both as "no focus".
        assert focus_file.read_text() == ""

        app_client.post(f"/api/artifacts/{entry.id}/pin", json={"pinned": True})
        app_client.post("/api/focus", json={"artifact_id": entry.id})
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


def _save_run_record(store, title="Run"):
    """A run record the way the executor writes one: the metadata key, no category."""
    return store.save_file(
        file_content=b"# run",
        filename=f"{title}.md",
        artifact_type="markdown",
        title=title,
        mime_type="text/markdown",
        tool_source="execute",
        metadata={"run_record": True},
    )


class TestRunRecordVerdict:
    """The server's run-record verdict: ``?run=`` on the listing, ``run_record`` everywhere.

    The web terminal's sidebar sorts Script runs from Artifacts by what the
    server says, never by re-deriving the rule, so every entry the gallery
    sends to a browser has to carry the verdict.
    """

    @staticmethod
    def _seed(store) -> tuple[set[str], set[str]]:
        """Five run records and seven results, interleaved; return their id sets."""
        runs: set[str] = set()
        others: set[str] = set()
        for i in range(12):
            if i in (1, 3, 4, 8, 10):
                runs.add(_save_run_record(store, title=f"Run {i}").id)
            else:
                others.add(_save_text_artifact(store, title=f"Result {i}").id)
        assert (len(runs), len(others)) == (5, 7)
        return runs, others

    @staticmethod
    def _walk(client, **params) -> tuple[list[dict], set[int]]:
        """Follow ``next_cursor`` to the end; return every entry and the totals seen."""
        entries: list[dict] = []
        totals: set[int] = set()
        cursor = None
        while True:
            query = dict(params, limit=3)
            if cursor:
                query["cursor"] = cursor
            body = client.get("/api/artifacts", params=query).json()
            entries.extend(body["artifacts"])
            totals.add(body["total"])
            cursor = body["next_cursor"]
            if cursor is None:
                return entries, totals

    def test_run_true_lists_only_run_records(self, app_client):
        runs, _ = self._seed(app_client.app.state.artifact_store)
        entries, totals = self._walk(app_client, run="true")
        assert {e["id"] for e in entries} == runs
        assert len(entries) == 5
        assert totals == {5}

    def test_run_false_lists_everything_else(self, app_client):
        _, others = self._seed(app_client.app.state.artifact_store)
        entries, totals = self._walk(app_client, run="false")
        assert {e["id"] for e in entries} == others
        assert len(entries) == 7
        assert totals == {7}

    def test_no_run_param_lists_all_of_them(self, app_client):
        runs, others = self._seed(app_client.app.state.artifact_store)
        entries, totals = self._walk(app_client)
        assert {e["id"] for e in entries} == runs | others
        assert totals == {12}

    def test_the_filtered_first_page_names_its_cursor_and_count(self, app_client):
        self._seed(app_client.app.state.artifact_store)
        body = app_client.get("/api/artifacts", params={"run": "true", "limit": 3}).json()
        assert (body["count"], body["total"]) == (3, 5)
        assert body["next_cursor"] is not None

    def test_every_listed_entry_carries_the_boolean_verdict(self, app_client):
        runs, others = self._seed(app_client.app.state.artifact_store)
        body = app_client.get("/api/artifacts", params={"limit": 50}).json()
        assert len(body["artifacts"]) == 12
        for entry in body["artifacts"]:
            assert entry["run_record"] is (entry["id"] in runs)

    def test_a_legacy_run_record_is_known_by_what_its_writer_gave_it(self, app_client):
        """Saved before the metadata key existed: the category alone marks it."""
        store = app_client.app.state.artifact_store
        legacy = store.save_file(
            file_content=b"out",
            filename="out.txt",
            artifact_type="text",
            title="Output",
            mime_type="text/plain",
            tool_source="execute",
            category="code_output",
        )
        body = app_client.get("/api/artifacts", params={"run": "true"}).json()
        assert [e["id"] for e in body["artifacts"]] == [legacy.id]

    def test_get_artifact_carries_the_verdict(self, app_client):
        store = app_client.app.state.artifact_store
        run = _save_run_record(store)
        result = _save_text_artifact(store)
        assert app_client.get(f"/api/artifacts/{run.id}").json()["run_record"] is True
        assert app_client.get(f"/api/artifacts/{result.id}").json()["run_record"] is False

    def test_focus_carries_the_verdict_for_a_focused_and_a_fallback_artifact(self, app_client):
        store = app_client.app.state.artifact_store
        run = _save_run_record(store)
        app_client.post("/api/focus", json={"artifact_id": run.id})
        focus = app_client.get("/api/focus").json()
        assert focus["focused"] is True
        assert focus["artifact"]["run_record"] is True

        result = _save_text_artifact(store, title="Newest")
        store.delete_entry(run.id)
        fallback = app_client.get("/api/focus").json()
        assert fallback["focused"] is False
        assert fallback["artifact"]["id"] == result.id
        assert fallback["artifact"]["run_record"] is False

    def test_the_pin_frame_carries_the_verdict(self, app_client):
        run = _save_run_record(app_client.app.state.artifact_store)
        with patch.object(_SSEBroadcaster, "broadcast") as broadcast:
            app_client.post(f"/api/artifacts/{run.id}/pin", json={"pinned": True})
        (event,), _ = broadcast.call_args
        assert event["type"] == "artifact_updated"
        assert event["id"] == run.id
        assert event["run_record"] is True

    def test_the_saved_frame_carries_the_verdict(self, tmp_path):
        """The lifespan registers the saved listener; a save then broadcasts its frame."""
        with TestClient(create_app(workspace_root=tmp_path)) as client:
            store = client.app.state.artifact_store
            with patch.object(_SSEBroadcaster, "broadcast") as broadcast:
                run = _save_run_record(store)
                result = _save_text_artifact(store)
            frames = {
                event["id"]: event
                for (event,), _ in broadcast.call_args_list
                if event["type"] == "artifact"
            }
        assert frames[run.id]["run_record"] is True
        assert frames[result.id]["run_record"] is False


class TestServeFileEndpoint:
    """GET /files/{artifact_id}/{filename}."""

    def test_missing_artifact_or_file_returns_404(self, app_client):
        assert app_client.get("/files/nonexistent/whatever.html").status_code == 404

        store = app_client.app.state.artifact_store
        entry = _save_text_artifact(store)
        store.get_file_path(entry.id).unlink()
        resp = app_client.get(f"/files/{entry.id}/{entry.filename}")
        assert resp.status_code == 404
        assert "not found on disk" in resp.json()["detail"]

    def test_plotly_bundle_injected_only_when_absent(self, app_client, monkeypatch):
        """A plot_html artifact saved with include_plotlyjs=False gets the
        local Plotly bundle plus the responsive snippet injected (offline mode
        pins the injected src to the local vendor path); one already carrying
        the bundle must not get a second 4.8MB script tag."""
        store = app_client.app.state.artifact_store

        monkeypatch.setenv("OSPREY_OFFLINE", "1")
        bare = _save_html_artifact(
            store, '<html><head></head><body><div class="plotly-graph-div"></div></body></html>'
        )
        resp = app_client.get(f"/files/{bare.id}/{bare.filename}")
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
        resp = app_client.get(f"/files/{bundled.id}/{bundled.filename}")
        assert resp.status_code == 200
        assert resp.text.count("plotly-3.3.1.min.js") == 1
        assert ".js-plotly-plot" in resp.text  # snippet still injected

    def test_plotly_injection_for_include_plotlyjs_false(self, app_client, monkeypatch):
        """A figure exported with include_plotlyjs=False calls Plotly but never
        loads it; serving it injects exactly one local bundle script tag."""
        monkeypatch.setenv("OSPREY_OFFLINE", "1")
        entry = _save_html_artifact(
            app_client.app.state.artifact_store,
            "<html><head></head><body>"
            '<div id="fig" class="plotly-graph-div"></div>'
            '<script>Plotly.newPlot("fig", [{"y": [1, 2, 3]}], {});</script>'
            "</body></html>",
        )
        resp = app_client.get(f"/files/{entry.id}/{entry.filename}")
        assert resp.status_code == 200
        assert resp.text.count('src="/static/js/vendor/plotly-3.3.1.min.js"') == 1

    def test_plotly_injection_skipped_for_inline_bundle(self, app_client, monkeypatch):
        """A figure exported with include_plotlyjs=True inlines the bundle
        (banner ``plotly.js v...``); no second copy is injected, whatever its
        version, while the responsive snippet still is."""
        monkeypatch.setenv("OSPREY_OFFLINE", "1")
        entry = _save_html_artifact(
            app_client.app.state.artifact_store,
            "<html><head></head><body>"
            "<script>/**\n* plotly.js v2.35.2\n* Copyright 2012-2024, Plotly, Inc.\n*/"
            "window.Plotly = {};</script>"
            '<div id="fig" class="plotly-graph-div"></div>'
            '<script>Plotly.newPlot("fig", [{"y": [1, 2, 3]}], {});</script>'
            "</body></html>",
        )
        resp = app_client.get(f"/files/{entry.id}/{entry.filename}")
        assert resp.status_code == 200
        assert "plotly-3.3.1.min.js" not in resp.text
        assert "cdn.plot.ly" not in resp.text
        assert ".js-plotly-plot" in resp.text  # snippet still injected

    def test_offline_rewrites_cdn_plotly_to_local(self, app_client, monkeypatch):
        """In offline mode a CDN Plotly reference is rewritten to the bundled
        copy and its SRI attributes are stripped (the local file may differ
        from the CDN version, and SRI would block it)."""
        monkeypatch.setenv("OSPREY_OFFLINE", "1")
        entry = _save_html_artifact(
            app_client.app.state.artifact_store,
            "<html><head>"
            '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"'
            ' integrity="sha384-abc123" crossorigin="anonymous"></script>'
            "</head><body></body></html>",
        )

        resp = app_client.get(f"/files/{entry.id}/{entry.filename}")
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
        entry = _save_html_artifact(
            app_client.app.state.artifact_store,
            '<html><head></head><body><div class="plotly-graph-div"></div></body></html>',
        )
        body = app_client.get(f"/files/{entry.id}/{entry.filename}").text
        assert '/design-system/js/theme-boot.js"' in body
        assert '/design-system/css/tokens.css"' in body
        assert "/design-system/js/theme-manager.js" in body
        # The C-MSG listener posts to window.parent, but nothing reads the
        # parent's document or theme -- on an opaque origin it could not.
        assert "parent.document" not in body
        assert "parent.localStorage" not in body

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
        store = app_client.app.state.artifact_store
        entry = store.save_file(
            file_content=b'<table class="artifact-table"><tr><td>1</td></tr></table>',
            filename="table.html",
            artifact_type="table_html",
            title="Table",
            description="",
            mime_type="text/html",
            tool_source="test",
        )
        body = app_client.get(f"/files/{entry.id}/{entry.filename}").text
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


class TestGalleryFacilityTimezone:
    """The gallery shell carries the facility zone on ``<html>``."""

    @staticmethod
    def _html_tag(body: str) -> str:
        return body.split("<head>", 1)[0]

    def test_gallery_shell_stamps_the_facility_zone(self, tmp_path):
        from unittest.mock import patch
        from zoneinfo import ZoneInfo

        with patch(
            "osprey.interfaces.artifacts.app.get_facility_timezone",
            return_value=ZoneInfo("Asia/Tokyo"),
        ):
            client = TestClient(create_app(workspace_root=tmp_path))
        body = client.get("/").text
        assert 'data-facility-timezone="Asia/Tokyo"' in self._html_tag(body)

    def test_zone_comes_from_the_primed_config(self, tmp_path, monkeypatch):
        """The zone is resolved after config priming points the resolver at
        this deployment's config."""
        import osprey_connectors.config as config_module

        for name in ("_default_config", "_default_configurable", "_tz_drift_warned"):
            monkeypatch.setattr(config_module, name, getattr(config_module, name))

        cfg = tmp_path / "cfg"
        cfg.mkdir()
        config_file = cfg / "config.yml"
        config_file.write_text("project_name: tz-test\nsystem:\n  timezone: Asia/Tokyo\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(config_file))

        ws = tmp_path / "ws"
        ws.mkdir()
        client = TestClient(create_app(workspace_root=ws))
        body = client.get("/").text
        assert 'data-facility-timezone="Asia/Tokyo"' in self._html_tag(body)


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
        import osprey_connectors.config as config_module

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

    def test_create_app_without_a_root_serves_the_shared_data_root(self, tmp_path, monkeypatch):
        """With no root passed, the gallery, its store and the focus file all use
        the deployment's shared agent-data root."""
        import osprey_connectors.workspace as workspace_module

        shared = tmp_path / "shared"
        monkeypatch.setattr(workspace_module, "resolve_shared_data_root", lambda: shared)

        client = TestClient(create_app())
        entry = _save_text_artifact(client.app.state.artifact_store)

        assert client.post("/api/focus", json={"artifact_id": entry.id}).status_code == 200
        assert (shared / "focus_state.txt").is_file()

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
        html = app_client.get("/").text
        assert f'name="osprey-vendor-plotly" content="{asset_cdn_url("Plotly.js")}"' in html

    def test_offline_mode_carries_the_vendored_path(self, app_client, monkeypatch):
        monkeypatch.setenv("OSPREY_OFFLINE", "1")
        html = app_client.get("/").text
        assert 'name="osprey-vendor-plotly" content="/static/js/vendor/plotly-3.3.1.min.js"' in html


class TestIndexArtifactDirMeta:
    """GET / carries the store's artifact directory for types.js's ``artifactPath``.

    The path chip, the copy-path button and the drag-to-terminal text all name
    an artifact by this directory, so it has to be the one the deployment's
    store writes to: repo-relative when an honest repo-relative spelling
    exists, absolute otherwise.
    """

    @staticmethod
    def _client(tmp_path, monkeypatch, root: Path, base_dir: str | None) -> TestClient:
        repo_root = tmp_path / "repo"
        (repo_root / "build").mkdir(parents=True)
        config = f"project_root: {repo_root}\n"
        if base_dir is not None:
            config += f"agent_data:\n  base_dir: {base_dir}\n"
        config_path = repo_root / "build" / "config.yml"
        config_path.write_text(config)
        monkeypatch.setenv("OSPREY_CONFIG", str(config_path))
        monkeypatch.chdir(repo_root)
        return TestClient(create_app(workspace_root=root))

    def test_default_layout_is_stamped_repo_relative(self, tmp_path, monkeypatch):
        root = tmp_path / "repo" / "var" / "agent_data"
        html = self._client(tmp_path, monkeypatch, root, None).get("/").text
        assert 'name="osprey-artifact-dir" content="var/agent_data/artifacts"' in html

    def test_relocated_base_dir_is_stamped_repo_relative(self, tmp_path, monkeypatch):
        root = tmp_path / "repo" / "state" / "agent"
        html = self._client(tmp_path, monkeypatch, root, "state/agent").get("/").text
        assert 'name="osprey-artifact-dir" content="state/agent/artifacts"' in html

    def test_absolute_base_dir_is_stamped_absolute(self, tmp_path, monkeypatch):
        root = tmp_path / "abs" / "agent"
        html = self._client(tmp_path, monkeypatch, root, str(root)).get("/").text
        assert f'name="osprey-artifact-dir" content="{root}/artifacts"' in html
        assert 'content="agent/artifacts"' not in html

    @pytest.mark.parametrize(
        ("root", "base_dir", "expected"),
        [
            ("/r/var/agent_data", "var/agent_data", "var/agent_data/artifacts"),
            ("/r/state/agent", "state/agent", "state/agent/artifacts"),
            ("/r/data", "data", "data/artifacts"),
            ("/abs/agent", "/abs/agent", "/abs/agent/artifacts"),
            ("/home/u/osprey-data", "~/osprey-data", "/home/u/osprey-data/artifacts"),
            ("/r/tests/root", "var/agent_data", "/r/tests/root/artifacts"),
        ],
        ids=["default", "two-segment", "single-segment", "absolute", "home", "mismatched"],
    )
    def test_agent_artifact_dir_spells_each_layout(self, root, base_dir, expected):
        from osprey.utils.workspace import repo_root_for_agent_data

        root_path = Path(root)
        repo_root = repo_root_for_agent_data(root_path, base_dir)
        assert _agent_artifact_dir(root_path / "artifacts", repo_root, base_dir) == expected


def _save_typed_artifact(store, *, artifact_type, mime_type, filename, content=b"<x/>"):
    return store.save_file(
        file_content=content,
        filename=filename,
        artifact_type=artifact_type,
        title=filename,
        description="",
        mime_type=mime_type,
        tool_source="test",
    )


class TestServedArtifactsAreOpaque:
    """Agent-written bytes that a browser could run are served on an opaque
    origin (C-OPAQUE); inert types keep their headers plus ``nosniff``; the
    gallery's own pages carry neither."""

    @pytest.mark.parametrize(
        ("mime_type", "filename"),
        [
            ("image/svg+xml", "figure.svg"),
            ("application/xhtml+xml", "page.xhtml"),
            ("application/atom+xml", "feed.xml"),
            ("application/xslt+xml", "style.xsl"),
            ("application/octet-stream", "blob.bin"),
        ],
    )
    def test_a_served_file_of_a_scriptable_type_carries_the_opaque_sandbox(
        self, app_client, mime_type, filename
    ):
        entry = _save_typed_artifact(
            app_client.app.state.artifact_store,
            artifact_type="file",
            mime_type=mime_type,
            filename=filename,
        )
        resp = app_client.get(f"/files/{entry.id}/{entry.filename}")
        assert resp.status_code == 200
        assert resp.headers["content-security-policy"] == OPAQUE_CSP
        assert resp.headers["x-content-type-options"] == "nosniff"

    @pytest.mark.parametrize(
        "mime_type",
        sorted(INERT_TYPES),
    )
    def test_a_served_file_of_an_inert_type_gets_nosniff_and_no_sandbox(
        self, app_client, mime_type
    ):
        entry = _save_typed_artifact(
            app_client.app.state.artifact_store,
            artifact_type="file",
            mime_type=mime_type,
            filename="data.bin",
            content=b"\x00\x01",
        )
        resp = app_client.get(f"/files/{entry.id}/{entry.filename}")
        assert resp.status_code == 200
        assert "content-security-policy" not in resp.headers
        assert resp.headers["x-content-type-options"] == "nosniff"

    def test_a_text_artifact_is_served_as_inert_plain_text(self, app_client):
        """A ``text`` artifact (a .tex file, say) is served as text/plain, which
        is inert whatever type it was saved under."""
        entry = _save_typed_artifact(
            app_client.app.state.artifact_store,
            artifact_type="text",
            mime_type="application/x-tex",
            filename="paper.tex",
            content=b"\\section{x}",
        )
        resp = app_client.get(f"/files/{entry.id}/{entry.filename}")
        assert resp.headers["content-type"].startswith("text/plain")
        assert "content-security-policy" not in resp.headers
        assert resp.headers["x-content-type-options"] == "nosniff"

    @pytest.mark.parametrize("artifact_type", ["plot_html", "html", "table_html", "dashboard_html"])
    def test_an_injected_html_artifact_carries_the_opaque_sandbox(self, app_client, artifact_type):
        entry = _save_typed_artifact(
            app_client.app.state.artifact_store,
            artifact_type=artifact_type,
            mime_type="text/html",
            filename="page.html",
            content=b"<html><head></head><body><script>fetch('/x')</script></body></html>",
        )
        resp = app_client.get(f"/files/{entry.id}/{entry.filename}")
        assert resp.status_code == 200
        assert resp.headers["content-security-policy"] == OPAQUE_CSP
        assert resp.headers["x-content-type-options"] == "nosniff"
        assert resp.headers["content-disposition"].startswith("inline")

    @pytest.mark.parametrize("path", ["/", "/static/js/md-render.js", "/api/artifacts"])
    def test_the_gallerys_own_pages_carry_no_sandbox(self, app_client, path):
        resp = app_client.get(path)
        assert resp.status_code == 200
        assert "content-security-policy" not in resp.headers


class TestArtifactMessageListener:
    """C-MSG: every injection point carries the print/ready listener and keeps
    following the theme through the design-system module."""

    def test_the_listener_posts_ready_and_prints_on_request(self):
        from osprey.interfaces.artifacts.app import _ARTIFACT_MESSAGE_LISTENER

        assert "osprey-artifact-ready" in _ARTIFACT_MESSAGE_LISTENER
        assert "osprey-artifact-print" in _ARTIFACT_MESSAGE_LISTENER
        assert "window.opener || window.parent" in _ARTIFACT_MESSAGE_LISTENER
        assert "window.print()" in _ARTIFACT_MESSAGE_LISTENER
        # print is accepted only from the frame's opener or parent
        assert "event.source" in _ARTIFACT_MESSAGE_LISTENER

    @pytest.mark.parametrize("artifact_type", ["plot_html", "html", "table_html", "dashboard_html"])
    def test_every_html_snippet_carries_the_listener_and_the_theme_follower(self, artifact_type):
        from osprey.interfaces.artifacts.app import (
            _ARTIFACT_MESSAGE_LISTENER,
            _RESPONSIVE_SNIPPETS,
        )

        snippet = _RESPONSIVE_SNIPPETS[artifact_type]
        assert _ARTIFACT_MESSAGE_LISTENER in snippet
        assert "from '/design-system/js/theme-manager.js'" in snippet
        assert "initTheme({ role: 'follower' })" in snippet

    def test_a_served_html_artifact_carries_the_listener(self, app_client):
        entry = _save_html_artifact(
            app_client.app.state.artifact_store, "<html><head></head><body></body></html>"
        )
        body = app_client.get(f"/files/{entry.id}/{entry.filename}").text
        assert "osprey-artifact-ready" in body
        assert "osprey-artifact-print" in body
