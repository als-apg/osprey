"""Tests for local static-panel discovery-by-convention.

A panel bundle is a directory with ``manifest.json`` + the ``entry`` HTML.
:func:`discover_panels` finds compliant bundles under ``<project>/panels`` and
:func:`apply_discovered_panels` wires them into the same hub surfaces
runtime-registered panels use — gated on ``web.allow_runtime_panels`` (the human
opt-in) and **fail-closed** on any malformed/non-compliant bundle.

Covers:
    - discover_panels: valid bundle found; missing root, non-panel dir,
      malformed manifest, missing entry, raw hex color, built-in id collision,
      duplicate id — each skipped without affecting the others; never raises.
    - apply_discovered_panels: gated off by default; on → appended to
      custom_panels / visible_panels / discovered_panel_dirs; existing-id skip.
    - The /panel-static/{id}/{path} serving route: entry served for bare path,
      asset served, unknown id / missing file / symlink-escape → 404; and
      GET /api/panels keeps a same-origin URL as-is while rewriting raw ones.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.panel_discovery import (
    DiscoveredPanel,
    apply_discovered_panels,
    discover_panels,
)
from osprey.interfaces.web_terminal.routes.panels import router

# A minimal but fully compliant entry HTML: links the token stylesheet and the
# shared font stylesheet, loads the pre-paint boot script, and uses only
# var(--…) tokens (no raw hex).
_VALID_HTML = """<!DOCTYPE html>
<html lang="en"><head>
<script src="/design-system/js/theme-boot.js"></script>
<link rel="stylesheet" href="/design-system/css/tokens.css">
<link rel="stylesheet" href="/static/fonts/fonts.css">
<title>Demo</title>
<style>body { background: var(--bg-primary); color: var(--text-primary); }</style>
</head><body><h1>Demo panel</h1></body></html>
"""


def _write_panel(root, dirname, *, manifest, html=_VALID_HTML, entry="index.html"):
    """Create ``root/dirname`` with a manifest and (optionally) an entry file.

    ``manifest`` is written as JSON verbatim (pass a str for malformed JSON).
    ``html=None`` skips writing the entry file (to exercise a missing entry).
    """
    panel_dir = root / dirname
    panel_dir.mkdir(parents=True)
    manifest_text = manifest if isinstance(manifest, str) else json.dumps(manifest)
    (panel_dir / "manifest.json").write_text(manifest_text, encoding="utf-8")
    if html is not None:
        (panel_dir / entry).write_text(html, encoding="utf-8")
    return panel_dir


def _manifest(pid="demo", label="Demo", entry="index.html"):
    return {"id": pid, "label": label, "entry": entry, "version": 1}


class TestDiscoverPanels:
    """The pure discovery scan — compliant bundles in, fail-closed on the rest."""

    def test_missing_root_returns_empty(self, tmp_path):
        assert discover_panels(tmp_path / "does-not-exist") == []

    def test_empty_root_returns_empty(self, tmp_path):
        assert discover_panels(tmp_path) == []

    def test_valid_panel_is_discovered(self, tmp_path):
        _write_panel(tmp_path, "demo", manifest=_manifest())
        result = discover_panels(tmp_path)
        assert len(result) == 1
        panel = result[0]
        assert isinstance(panel, DiscoveredPanel)
        assert (panel.id, panel.label, panel.entry) == ("demo", "Demo", "index.html")
        assert panel.directory == (tmp_path / "demo").resolve()

    def test_dir_without_manifest_is_silently_skipped(self, tmp_path):
        (tmp_path / "notapanel").mkdir()
        (tmp_path / "notapanel" / "README.md").write_text("hi", encoding="utf-8")
        assert discover_panels(tmp_path) == []

    def test_loose_file_in_root_is_ignored(self, tmp_path):
        (tmp_path / "stray.txt").write_text("x", encoding="utf-8")
        _write_panel(tmp_path, "demo", manifest=_manifest())
        assert [p.id for p in discover_panels(tmp_path)] == ["demo"]

    def test_malformed_manifest_json_is_skipped_and_logged(self, tmp_path, caplog):
        _write_panel(tmp_path, "broken", manifest="{ not valid json ")
        with caplog.at_level(logging.WARNING):
            result = discover_panels(tmp_path)
        assert result == []
        assert any("broken" in r.message for r in caplog.records)

    def test_manifest_missing_required_field_is_skipped(self, tmp_path):
        _write_panel(tmp_path, "noid", manifest={"label": "No id", "entry": "index.html"})
        assert discover_panels(tmp_path) == []

    def test_missing_entry_file_is_skipped(self, tmp_path):
        # Manifest declares index.html but no such file is written.
        _write_panel(tmp_path, "noentry", manifest=_manifest(), html=None)
        assert discover_panels(tmp_path) == []

    def test_raw_hex_color_fails_closed(self, tmp_path):
        bad_html = _VALID_HTML.replace("var(--bg-primary)", "#ff0000")
        _write_panel(tmp_path, "hexy", manifest=_manifest(), html=bad_html)
        assert discover_panels(tmp_path) == []

    def test_builtin_id_collision_is_skipped(self, tmp_path):
        _write_panel(tmp_path, "arti", manifest=_manifest(pid="artifacts", label="X"))
        assert discover_panels(tmp_path) == []

    def test_duplicate_ids_keep_first_and_log(self, tmp_path, caplog):
        _write_panel(tmp_path, "a-first", manifest=_manifest(pid="dup", label="First"))
        _write_panel(tmp_path, "b-second", manifest=_manifest(pid="dup", label="Second"))
        with caplog.at_level(logging.WARNING):
            result = discover_panels(tmp_path)
        # Sorted by dir name: "a-first" wins, "b-second" is dropped.
        assert [p.label for p in result] == ["First"]
        assert any("b-second" in r.message for r in caplog.records)

    def test_one_bad_panel_does_not_break_the_good_ones(self, tmp_path):
        _write_panel(tmp_path, "good", manifest=_manifest(pid="good", label="Good"))
        _write_panel(tmp_path, "bad", manifest="{ broken ")
        assert [p.id for p in discover_panels(tmp_path)] == ["good"]

    def test_non_utf8_asset_is_skipped_not_fatal(self, tmp_path):
        # A non-UTF-8 sibling asset makes validate_panel's strict read_text
        # raise UnicodeDecodeError; discovery must skip that bundle, not crash.
        panel = _write_panel(tmp_path, "latin1", manifest=_manifest(pid="latin1"))
        (panel / "styles.css").write_bytes(b"body{}/* \xff\xfe not utf-8 */")
        result = discover_panels(tmp_path)  # must not raise
        assert result == []

    def test_non_utf8_asset_does_not_sink_good_panels(self, tmp_path):
        _write_panel(tmp_path, "aaa-good", manifest=_manifest(pid="good"))
        bad = _write_panel(tmp_path, "zzz-bad", manifest=_manifest(pid="bad"))
        (bad / "styles.css").write_bytes(b"\xff\xfe\xfa\xfb")
        assert [p.id for p in discover_panels(tmp_path)] == ["good"]

    def test_parent_relative_entry_is_skipped(self, tmp_path):
        # entry with a '..' component: the bundle validates (the file exists and
        # is linked), but the entry-shape guard rejects it → not served.
        _write_panel(tmp_path, "esc", manifest=_manifest(pid="esc", entry="../evil.html"))
        assert discover_panels(tmp_path) == []

    def test_never_raises_on_garbage(self, tmp_path):
        _write_panel(tmp_path, "x", manifest="not json at all")
        (tmp_path / "empty").mkdir()
        try:
            discover_panels(tmp_path)
        except Exception as exc:  # pragma: no cover - failure path
            pytest.fail(f"discover_panels raised unexpectedly: {exc}")


def _fake_app(*, allow, project_cwd, custom=None, visible=None):
    """A stand-in FastAPI app object exposing only the state discovery reads."""
    state = SimpleNamespace(
        allow_runtime_panels=allow,
        project_cwd=str(project_cwd) if project_cwd is not None else None,
        custom_panels=list(custom or []),
        visible_panels=list(visible or []),
        discovered_panel_dirs={},
    )
    return SimpleNamespace(state=state)


class TestApplyDiscoveredPanels:
    """Gating + wiring into app.state."""

    def test_gate_off_is_noop(self, tmp_path):
        _write_panel(tmp_path / "panels", "demo", manifest=_manifest())
        app = _fake_app(allow=False, project_cwd=tmp_path)
        assert apply_discovered_panels(app) == []
        assert app.state.custom_panels == []
        assert app.state.visible_panels == []
        assert app.state.discovered_panel_dirs == {}

    def test_no_project_cwd_is_noop(self, tmp_path):
        app = _fake_app(allow=True, project_cwd=None)
        assert apply_discovered_panels(app) == []

    def test_gate_on_wires_panel_into_state(self, tmp_path):
        _write_panel(tmp_path / "panels", "demo", manifest=_manifest())
        app = _fake_app(allow=True, project_cwd=tmp_path)
        applied = apply_discovered_panels(app)
        assert [p.id for p in applied] == ["demo"]
        # custom_panels entry has the same-origin static URL and discovered flag.
        entry = next(cp for cp in app.state.custom_panels if cp["id"] == "demo")
        assert entry["url"] == "/panel-static/demo/"
        assert entry["discovered"] is True
        assert "demo" in app.state.visible_panels
        assert isinstance(app.state.discovered_panel_dirs["demo"], DiscoveredPanel)

    def test_existing_custom_id_is_not_clobbered(self, tmp_path):
        _write_panel(tmp_path / "panels", "demo", manifest=_manifest())
        app = _fake_app(
            allow=True,
            project_cwd=tmp_path,
            custom=[{"id": "demo", "label": "Live", "url": "http://x/"}],
        )
        assert apply_discovered_panels(app) == []
        # The pre-existing registration is left untouched.
        assert [cp["label"] for cp in app.state.custom_panels] == ["Live"]


@pytest.fixture
def serving_client(tmp_path):
    """A bare app with the panels router and one discovered panel in state."""
    panel_dir = tmp_path / "demo"
    panel_dir.mkdir()
    (panel_dir / "index.html").write_text(_VALID_HTML, encoding="utf-8")
    (panel_dir / "app.js").write_text("export const x = 1;\n", encoding="utf-8")

    app = FastAPI()
    app.include_router(router)
    app.state.enabled_panels = set()
    app.state.custom_panels = [
        {"id": "demo", "label": "Demo", "url": "/panel-static/demo/", "discovered": True},
        {"id": "grafana", "label": "Grafana", "url": "http://grafana.lan:3000/"},
        # A non-discovered custom panel that happens to carry a leading-slash
        # URL — must still route through the proxy (keyed off the discovered
        # flag, not URL shape), never treated as same-origin.
        {"id": "legacy", "label": "Legacy", "url": "//evil.example/x"},
    ]
    app.state.visible_panels = ["demo", "grafana", "legacy"]
    app.state.discovered_panel_dirs = {
        "demo": DiscoveredPanel(
            id="demo", label="Demo", entry="index.html", directory=panel_dir.resolve()
        )
    }
    return TestClient(app), tmp_path


class TestServingRoute:
    """The /panel-static/{id}/{path} local-file route + api/panels URL rewrite."""

    def test_bare_path_serves_entry(self, serving_client):
        client, _ = serving_client
        resp = client.get("/panel-static/demo/")
        assert resp.status_code == 200
        assert "Demo panel" in resp.text

    def test_named_asset_is_served(self, serving_client):
        client, _ = serving_client
        resp = client.get("/panel-static/demo/app.js")
        assert resp.status_code == 200
        assert "export const x" in resp.text

    def test_unknown_panel_id_is_404(self, serving_client):
        client, _ = serving_client
        assert client.get("/panel-static/ghost/").status_code == 404

    def test_missing_file_is_404(self, serving_client):
        client, _ = serving_client
        assert client.get("/panel-static/demo/nope.js").status_code == 404

    def test_absolute_path_segment_cannot_escape(self, serving_client):
        client, _ = serving_client
        # A path that pathlib would treat as absolute (resetting the join base)
        # must not serve a file outside the bundle.
        resp = client.get("/panel-static/demo//etc/hosts")
        assert resp.status_code == 404
        assert "localhost" not in resp.text

    def test_symlink_escape_is_blocked(self, serving_client):
        client, tmp_path = serving_client
        secret = tmp_path / "secret.txt"
        secret.write_text("top secret", encoding="utf-8")
        link = tmp_path / "demo" / "escape.txt"
        try:
            link.symlink_to(secret)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unsupported on this platform")
        resp = client.get("/panel-static/demo/escape.txt")
        assert resp.status_code == 404
        assert "top secret" not in resp.text

    def test_api_panels_keeps_static_url_and_rewrites_raw(self, serving_client):
        client, _ = serving_client
        body = client.get("/api/panels").json()
        by_id = {cp["id"]: cp for cp in body["custom"]}
        # Discovered static panel: same-origin URL preserved.
        assert by_id["demo"]["url"] == "/panel-static/demo/"
        # URL-backed panel: raw upstream rewritten to the reverse-proxy path.
        assert by_id["grafana"]["url"] == "/panel/grafana"
        # Non-discovered panel with a protocol-relative URL: still proxied, never
        # passed through as same-origin.
        assert by_id["legacy"]["url"] == "/panel/legacy"
