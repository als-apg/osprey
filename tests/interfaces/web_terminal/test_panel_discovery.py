"""Tests for local static-panel discovery-by-convention.

A panel bundle is a directory with ``manifest.json`` + the ``entry`` HTML.
:func:`discover_panels` finds compliant bundles under ``<project>/panels`` and
:func:`apply_discovered_panels` wires them into the same hub surfaces
runtime-registered panels use — gated on ``web.allow_runtime_panels`` (the human
opt-in) and **fail-closed** on any malformed/non-compliant bundle.

Covers:
    - discover_panels: valid bundle found; missing root, non-panel dir,
      any validation error (the validator's own suite owns which errors
      exist), built-in id collision, duplicate id, configured id,
      parent-relative entry, an unreadable asset — each refused with its
      reasons without affecting the others.
    - apply_discovered_panels: gated off by default; on → appended to
      custom_panels / visible_panels / discovered_panel_dirs, every refusal
      logged and recorded in refused_panels; existing-id skip.
    - The /panel-static/{id}/{path} serving route: entry served for bare path,
      asset served, unknown id / missing file / symlink-escape → 404; and
      GET /api/panels keeps a same-origin URL as-is while rewriting raw ones;
      every non-inert bundle file carries ``opaque_headers()`` on the plain
      (cookie-gated) path too, inert ones only ``nosniff``.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.design_system.panels.page_ready import WRITER_TAG
from osprey.interfaces.design_system.panels.validator import validate_panel
from osprey.interfaces.opaque_origin import OPAQUE_CSP
from osprey.interfaces.web_terminal.panel_discovery import (
    DiscoveredPanel,
    RefusedPanel,
    apply_build_panels,
    apply_discovered_panels,
    discover_panels,
)
from osprey.interfaces.web_terminal.routes.panels import router

from .conftest import bare_route_app

# A minimal but fully compliant entry HTML: links the design-system sheets and the
# shared font stylesheet, loads the pre-paint boot script, declares the page-ready
# mark pending, loads its writer first and sets it, and uses only var(--…) tokens (no raw hex).
_MARK = (
    '<script type="module">import { markPageReady } from '
    "'/design-system/js/page-ready.js'; markPageReady();</script>"
)
_VALID_HTML = f"""<!DOCTYPE html>
<html lang="en" data-page-ready="pending"><head>
<script src="/design-system/js/theme-boot.js"></script>
{WRITER_TAG}
<link rel="stylesheet" href="/design-system/css/tokens.css">
<link rel="stylesheet" href="/design-system/css/base.css">
<link rel="stylesheet" href="/design-system/css/primitives.css">
<link rel="stylesheet" href="/static/fonts/fonts.css">
{_MARK}
<title>Demo</title>
<style>body {{ background: var(--bg-primary); color: var(--text-primary); }}</style>
</head><body><h1>Demo panel</h1></body></html>
"""


_UNMARKED_HTML = _VALID_HTML.replace(' data-page-ready="pending"', "").replace(_MARK, "")


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
        result = discover_panels(tmp_path / "does-not-exist")
        assert result.served == ()
        assert result.refused == ()

    def test_valid_panel_is_discovered(self, tmp_path):
        _write_panel(tmp_path, "demo", manifest=_manifest())
        result = discover_panels(tmp_path).served
        assert len(result) == 1
        panel = result[0]
        assert isinstance(panel, DiscoveredPanel)
        assert (panel.id, panel.label, panel.entry) == ("demo", "Demo", "index.html")
        assert panel.directory == (tmp_path / "demo").resolve()

    def test_a_panel_that_never_marks_itself_ready_is_not_served(self, tmp_path):
        _write_panel(tmp_path, "unmarked", manifest=_manifest(), html=_UNMARKED_HTML)
        result = discover_panels(tmp_path)
        assert result.served == ()
        [refused] = result.refused
        assert refused.id == "unmarked"
        assert "page-ready mark" in refused.findings[0].message

    def test_a_refused_bundle_is_returned_with_the_validators_findings(self, tmp_path):
        _write_panel(tmp_path, "unmarked", manifest=_manifest(), html=_UNMARKED_HTML)
        result = discover_panels(tmp_path)
        assert result.served == ()
        [refused] = result.refused
        assert refused.id == "unmarked"
        [finding] = refused.findings
        # The validator's own record, not a second judge's restatement of it.
        assert finding == validate_panel(tmp_path / "unmarked")[0]

    @pytest.mark.parametrize(
        ("rule", "write"),
        [
            pytest.param(
                "manifest_invalid",
                lambda root: _write_panel(root, "bad", manifest="{ not valid json "),
                id="malformed-manifest",
            ),
            pytest.param(
                "entry_outside_bundle",
                lambda root: _write_panel(
                    root,
                    "bad",
                    manifest=_manifest(pid="esc", entry="../evil.html"),
                    entry="../evil.html",
                ),
                id="entry-outside-bundle",
            ),
            pytest.param(
                "id_reserved",
                lambda root: _write_panel(root, "bad", manifest=_manifest(pid="artifacts")),
                id="builtin-id",
            ),
            pytest.param(
                "id_taken",
                lambda root: (
                    _write_panel(root, "a-first", manifest=_manifest(pid="dup")),
                    _write_panel(root, "bad", manifest=_manifest(pid="dup")),
                ),
                id="duplicate-id",
            ),
            pytest.param(
                "uninspectable",
                lambda root: (
                    _write_panel(root, "bad", manifest=_manifest(pid="bad")) / "styles.css"
                ).write_bytes(b"\xff\xfe\xfa\xfb"),
                id="non-utf8-asset",
            ),
        ],
    )
    def test_every_way_discovery_refuses_is_named(self, tmp_path, rule, write):
        write(tmp_path)
        result = discover_panels(tmp_path)
        [refused] = result.refused
        assert refused.id == "bad"
        assert [str(f.rule) for f in refused.findings] == [rule]
        if rule == "id_taken":
            assert "'dup'" in refused.findings[0].message

    def test_an_id_a_configured_panel_holds_is_refused(self, tmp_path):
        _write_panel(tmp_path, "demo", manifest=_manifest())
        result = discover_panels(tmp_path, taken={"demo"})
        assert result.served == ()
        [finding] = result.refused[0].findings
        assert finding.rule == "id_taken"
        assert "web.panels" in finding.message

    def test_served_and_refused_together_are_every_bundle(self, tmp_path):
        _write_panel(tmp_path, "good-a", manifest=_manifest(pid="good-a"))
        _write_panel(tmp_path, "good-b", manifest=_manifest(pid="good-b"))
        _write_panel(tmp_path, "bad-a", manifest=_manifest(pid="bad-a"), html=_UNMARKED_HTML)
        _write_panel(tmp_path, "bad-b", manifest="{ broken ")
        (tmp_path / "notapanel").mkdir()
        result = discover_panels(tmp_path)
        assert {p.id for p in result.served} | {r.id for r in result.refused} == {
            "good-a",
            "good-b",
            "bad-a",
            "bad-b",
        }

    def test_discovery_logs_nothing_of_its_own(self, tmp_path, caplog):
        _write_panel(tmp_path, "unmarked", manifest=_manifest(), html=_UNMARKED_HTML)
        with caplog.at_level(logging.DEBUG):
            discover_panels(tmp_path)
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []

    def test_dir_without_manifest_is_silently_skipped(self, tmp_path):
        (tmp_path / "notapanel").mkdir()
        (tmp_path / "notapanel" / "README.md").write_text("hi", encoding="utf-8")
        result = discover_panels(tmp_path)
        assert result.served == ()
        assert result.refused == ()

    def test_loose_file_in_root_is_ignored(self, tmp_path):
        (tmp_path / "stray.txt").write_text("x", encoding="utf-8")
        _write_panel(tmp_path, "demo", manifest=_manifest())
        assert [p.id for p in discover_panels(tmp_path).served] == ["demo"]

    def test_malformed_manifest_json_is_refused(self, tmp_path):
        _write_panel(tmp_path, "broken", manifest="{ not valid json ")
        result = discover_panels(tmp_path)
        assert result.served == ()
        assert [r.id for r in result.refused] == ["broken"]

    def test_builtin_id_collision_is_skipped(self, tmp_path):
        _write_panel(tmp_path, "arti", manifest=_manifest(pid="artifacts", label="X"))
        assert discover_panels(tmp_path).served == ()

    def test_duplicate_ids_keep_first_and_refuse_the_second(self, tmp_path):
        _write_panel(tmp_path, "a-first", manifest=_manifest(pid="dup", label="First"))
        _write_panel(tmp_path, "b-second", manifest=_manifest(pid="dup", label="Second"))
        result = discover_panels(tmp_path)
        # Sorted by dir name: "a-first" wins, "b-second" is dropped.
        assert [p.label for p in result.served] == ["First"]
        assert [r.id for r in result.refused] == ["b-second"]

    def test_one_bad_panel_does_not_break_the_good_ones(self, tmp_path):
        _write_panel(tmp_path, "good", manifest=_manifest(pid="good", label="Good"))
        _write_panel(tmp_path, "bad", manifest="{ broken ")
        assert [p.id for p in discover_panels(tmp_path).served] == ["good"]

    def test_non_utf8_asset_does_not_sink_good_panels(self, tmp_path):
        _write_panel(tmp_path, "aaa-good", manifest=_manifest(pid="good"))
        bad = _write_panel(tmp_path, "zzz-bad", manifest=_manifest(pid="bad"))
        (bad / "styles.css").write_bytes(b"\xff\xfe\xfa\xfb")
        assert [p.id for p in discover_panels(tmp_path).served] == ["good"]

    def test_parent_relative_entry_is_skipped(self, tmp_path):
        # entry with a '..' component: the bundle validates (the file exists and
        # is linked), but the entry-shape guard rejects it → not served.
        _write_panel(
            tmp_path,
            "esc",
            manifest=_manifest(pid="esc", entry="../evil.html"),
            entry="../evil.html",
        )
        assert discover_panels(tmp_path).served == ()


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

    def test_gate_off_records_no_refusal(self, tmp_path):
        _write_panel(tmp_path / "panels", "unmarked", manifest=_manifest(), html=_UNMARKED_HTML)
        app = _fake_app(allow=False, project_cwd=tmp_path)
        assert apply_discovered_panels(app) == []
        assert app.state.custom_panels == []
        assert app.state.refused_panels == []

    def test_a_refused_panel_is_logged_with_the_line_of_today(self, tmp_path, caplog):
        _write_panel(tmp_path / "panels", "unmarked", manifest=_manifest(), html=_UNMARKED_HTML)
        app = _fake_app(allow=True, project_cwd=tmp_path)
        with caplog.at_level(logging.WARNING):
            apply_discovered_panels(app)
        [record] = [r for r in caplog.records if "unmarked" in r.getMessage()]
        assert record.levelno == logging.WARNING
        assert "is not served: 1 validation error(s)" in record.getMessage()
        assert "page-ready mark" in record.getMessage()

    def test_a_malformed_manifest_is_logged_naming_its_directory(self, tmp_path, caplog):
        _write_panel(tmp_path / "panels", "broken", manifest="{ not valid json ")
        app = _fake_app(allow=True, project_cwd=tmp_path)
        with caplog.at_level(logging.WARNING):
            apply_discovered_panels(app)
        assert any(
            "broken" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records
        )

    def test_a_duplicate_id_is_logged_naming_its_directory(self, tmp_path, caplog):
        _write_panel(tmp_path / "panels", "a-first", manifest=_manifest(pid="dup"))
        _write_panel(tmp_path / "panels", "b-second", manifest=_manifest(pid="dup"))
        app = _fake_app(allow=True, project_cwd=tmp_path)
        with caplog.at_level(logging.WARNING):
            apply_discovered_panels(app)
        [record] = [r for r in caplog.records if "b-second" in r.getMessage()]
        assert record.levelno == logging.WARNING
        assert "is not served: id 'dup' already discovered in another directory" in (
            record.getMessage()
        )

    def test_gate_on_records_the_refused_panels(self, tmp_path):
        _write_panel(tmp_path / "panels", "demo", manifest=_manifest())
        _write_panel(
            tmp_path / "panels", "unmarked", manifest=_manifest(pid="other"), html=_UNMARKED_HTML
        )
        app = _fake_app(allow=True, project_cwd=tmp_path)
        assert [p.id for p in apply_discovered_panels(app)] == ["demo"]
        [refused] = app.state.refused_panels
        assert isinstance(refused, RefusedPanel)
        assert refused.id == "unmarked"
        entry = next(cp for cp in app.state.custom_panels if cp["id"] == "demo")
        assert entry["url"] == "/panel-static/demo/"
        assert entry["discovered"] is True
        assert "demo" in app.state.visible_panels
        assert isinstance(app.state.discovered_panel_dirs["demo"], DiscoveredPanel)

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
        assert app.state.refused_panels[0].findings[0].rule == "id_taken"


class TestApplyBuildPanels:
    """A panel the build wrote is wired from the config's block, gate or no gate."""

    _BLOCK = {"id": "roster", "label": "Roster", "url": "", "configDefined": True}

    def test_a_declared_panel_with_its_bundle_is_wired(self, tmp_path):
        _write_panel(
            tmp_path / ".osprey" / "panels", "roster", manifest=_manifest("roster", "Roster")
        )
        app = _fake_app(
            allow=False, project_cwd=tmp_path, custom=[dict(self._BLOCK)], visible=["roster"]
        )

        assert [p.id for p in apply_build_panels(app)] == ["roster"]
        entry = app.state.custom_panels[0]
        assert entry["url"] == "/panel-static/roster/"
        assert entry["discovered"] is True
        assert entry["configDefined"] is True
        assert app.state.discovered_panel_dirs["roster"].entry == "index.html"
        assert app.state.visible_panels == ["roster"]

    def test_a_declared_panel_without_a_bundle_leaves_the_rail(self, tmp_path, caplog):
        app = _fake_app(
            allow=False, project_cwd=tmp_path, custom=[dict(self._BLOCK)], visible=["roster"]
        )
        with caplog.at_level(logging.WARNING):
            assert apply_build_panels(app) == []
        assert app.state.custom_panels == []
        assert app.state.visible_panels == []
        assert "rebuild the project" in caplog.text

    def test_a_block_with_its_own_url_is_the_deployments_panel(self, tmp_path):
        _write_panel(
            tmp_path / ".osprey" / "panels", "roster", manifest=_manifest("roster", "Roster")
        )
        own = {"id": "roster", "label": "Ours", "url": "http://roster.lan/"}
        app = _fake_app(allow=False, project_cwd=tmp_path, custom=[own], visible=["roster"])

        assert apply_build_panels(app) == []
        assert app.state.custom_panels == [own]
        assert app.state.discovered_panel_dirs == {}

    def test_a_bundle_nothing_declares_is_not_served(self, tmp_path):
        _write_panel(
            tmp_path / ".osprey" / "panels", "roster", manifest=_manifest("roster", "Roster")
        )
        app = _fake_app(allow=True, project_cwd=tmp_path)
        assert apply_build_panels(app) == []
        assert apply_discovered_panels(app) == []
        assert app.state.custom_panels == []


@pytest.fixture
def serving_client(tmp_path):
    """A bare app with the panels router and one discovered panel in state."""
    panel_dir = tmp_path / "demo"
    panel_dir.mkdir()
    (panel_dir / "index.html").write_text(_VALID_HTML, encoding="utf-8")
    (panel_dir / "app.js").write_text("export const x = 1;\n", encoding="utf-8")
    (panel_dir / "data.json").write_text('{"word": "hi"}', encoding="utf-8")

    app = bare_route_app(router)
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

    @pytest.mark.parametrize(
        ("path", "text"),
        [
            pytest.param("/panel-static/demo/", "Demo panel", id="bare-path-serves-entry"),
            pytest.param("/panel-static/demo/app.js", "export const x", id="named-asset"),
        ],
    )
    def test_a_panel_static_path_is_served(self, serving_client, path, text):
        client, _ = serving_client
        resp = client.get(path)
        assert resp.status_code == 200
        assert text in resp.text

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


class TestBundleBytesAreOpaque:
    """A ``<project>/panels`` bundle is agent-writable: opaque on its plain path too."""

    @pytest.mark.parametrize("path", ["", "index.html", "app.js"])
    def test_a_document_or_script_carries_the_sandbox(self, serving_client, path):
        client, _ = serving_client
        resp = client.get(f"/panel-static/demo/{path}")
        assert resp.status_code == 200
        assert resp.headers["content-security-policy"] == OPAQUE_CSP
        assert "allow-same-origin" not in resp.headers["content-security-policy"]
        assert resp.headers["x-content-type-options"] == "nosniff"

    def test_an_inert_file_gets_nosniff_and_no_sandbox(self, serving_client):
        client, _ = serving_client
        resp = client.get("/panel-static/demo/data.json")
        assert resp.status_code == 200
        assert resp.headers["x-content-type-options"] == "nosniff"
        assert "content-security-policy" not in resp.headers

    def test_the_plain_path_grants_no_cross_origin_read(self, serving_client):
        # Only the cap path answers ``Access-Control-Allow-Origin``; the plain
        # path stays a same-origin, cookie-gated URL.
        client, _ = serving_client
        resp = client.get("/panel-static/demo/data.json", headers={"Origin": "null"})
        assert "access-control-allow-origin" not in resp.headers
