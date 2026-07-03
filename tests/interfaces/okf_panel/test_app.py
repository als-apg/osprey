"""End-to-end integration tests for the okf_panel FastAPI factory.

Exercises the whole ``create_app`` over a shared on-disk fixture bundle
(``fixtures/bundle``) via ``TestClient`` — the integration layer that completes
the test-first work begun in ``test_helpers`` / ``test_validation`` (unit) and
``tests/registry/test_okf_panel_registration`` (wiring). Precedent:
``tests/interfaces/artifacts/test_type_registry_api.py``.

Covers the PROPOSAL success-criteria endpoint contracts plus every edge case:
slash-id round-trip, missing-id 404, empty bundle, guarded None bundle, and the
broken-cross-link bundle-health path. A final test drives the real launch chain
(registry definition → factory resolution → create_app) so the panel is proven
launchable the same way ``ServerLauncher`` builds it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.okf_panel.app import create_app

BUNDLE = Path(__file__).parent / "fixtures" / "bundle"


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(str(BUNDLE)))


# ---------------------------------------------------------------------------
# Configured panel over the fixture bundle
# ---------------------------------------------------------------------------


def test_health_ok_and_configured(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "service": "okf-panel", "configured": True}


def test_root_serves_spa_shell(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "<div id=\"app\">" in r.text


def test_concepts_grouped_and_sorted(client):
    groups = client.get("/api/concepts").json()["groups"]
    ids = [g["id"] for g in groups]
    assert ids == ["control-system", "devices", "references"]  # alpha sorted
    devices = next(g for g in groups if g["id"] == "devices")
    assert [c["title"] for c in devices["concepts"]] == ["Beam Position Monitor", "RF System"]
    # Per-concept dicts never leak the type field.
    assert all(set(c) == {"id", "title", "description"} for c in devices["concepts"])


def test_structure_markdown_overview(client):
    md = client.get("/api/structure").json()["markdown"]
    assert md.startswith("# Facility Knowledge Base")
    assert "_4 concepts across 3 groups._" in md
    assert "## Devices" in md
    assert "[Beam Position Monitor](/devices/bpm.md)" in md


def test_concept_slash_id_round_trips(client):
    r = client.get("/api/concept", params={"id": "control-system/channel-finding"})
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == "control-system/channel-finding"
    assert body["frontmatter"]["title"] == "Channel Finding"
    assert "GEECS" in body["body"]


def test_concept_missing_id_returns_404(client):
    r = client.get("/api/concept", params={"id": "devices/nope"})
    assert r.status_code == 404
    assert r.json()["error"] == "not found"


def test_concept_path_traversal_returns_404(client):
    # resolve_concept_path enforces bundle-root containment → OKFBundleError →
    # caught as 404. The panel never reads outside the bundle.
    for evil in ["../../../../etc/passwd", "/etc/passwd", "..%2f..%2fsecret"]:
        r = client.get("/api/concept", params={"id": evil})
        assert r.status_code == 404, evil


def test_search_returns_title_and_snippet(client):
    data = client.get("/api/search", params={"q": "GEECS"}).json()
    ids = [r["id"] for r in data["results"]]
    assert "control-system/channel-finding" in ids
    hit = next(r for r in data["results"] if r["id"] == "control-system/channel-finding")
    assert hit["title"] == "Channel Finding"
    assert "GEECS" in hit["snippet"]


def test_search_empty_query_short_circuits(client):
    assert client.get("/api/search", params={"q": "   "}).json() == {
        "query": "   ",
        "results": [],
    }


def test_search_snippet_falls_back_to_description_when_only_frontmatter_matches(client):
    # "terminology" appears only in the glossary's frontmatter description, not
    # its body → make_snippet returns "" and the handler falls back to the
    # frontmatter description string.
    data = client.get("/api/search", params={"q": "terminology"}).json()
    hit = next(r for r in data["results"] if r["id"] == "references/glossary")
    assert hit["title"] == "Glossary"
    assert hit["snippet"] == "Facility terminology."


def test_bundle_health_reports_broken_cross_link(client):
    health = client.get("/api/bundle_health").json()
    assert health["ok"] is False
    assert health["counts"].get("broken-link", 0) >= 1
    broken = [w for w in health["warnings"] if w["kind"] == "broken-link"]
    assert any(w["concept_id"] == "control-system/channel-finding" for w in broken)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_bundle(tmp_path):
    c = TestClient(create_app(str(tmp_path)))
    assert c.get("/api/concepts").json() == {"groups": []}
    assert "_0 concepts across 0 groups._" in c.get("/api/structure").json()["markdown"]
    assert c.get("/api/bundle_health").json() == {
        "ok": True,
        "total": 0,
        "counts": {},
        "warnings": [],
    }


def test_none_bundle_is_guarded_not_crashed():
    c = TestClient(create_app(bundle_path=None))
    # /health still 200 (panel process is up) but reports unconfigured.
    h = c.get("/health").json()
    assert h["configured"] is False
    # Every data endpoint returns the clear JSON "not configured" error, 503.
    for path, params in [
        ("/api/concepts", None),
        ("/api/structure", None),
        ("/api/concept", {"id": "x"}),
        ("/api/search", {"q": "x"}),
        ("/api/bundle_health", None),
    ]:
        r = c.get(path, params=params)
        assert r.status_code == 503, path
        assert r.json()["error"] == "facility_knowledge.bundle_path not configured"


def test_bad_bundle_path_degrades_to_guarded(tmp_path):
    # A configured-but-missing directory must degrade (not raise out of factory).
    missing = tmp_path / "does-not-exist"
    c = TestClient(create_app(str(missing)))
    assert c.get("/health").json()["configured"] is False
    assert c.get("/api/concepts").status_code == 503


# ---------------------------------------------------------------------------
# Launch path: build the app exactly as ServerLauncher does (registry factory)
# ---------------------------------------------------------------------------


def test_launchable_via_registry_factory(monkeypatch):
    """Drive registry def → _make_app_factory → _resolve_dotted → create_app."""
    from osprey.infrastructure import server_launcher
    from osprey.registry.web import FRAMEWORK_WEB_SERVERS

    fake_config = {"facility_knowledge": {"bundle_path": str(BUNDLE)}}
    monkeypatch.setattr(server_launcher, "load_osprey_config", lambda: fake_config)

    factory = server_launcher._make_app_factory(FRAMEWORK_WEB_SERVERS["okf"])
    app = factory()  # resolves bundle_path from config and imports the real factory
    c = TestClient(app)

    assert c.get("/health").json()["configured"] is True
    r = c.get("/api/concept", params={"id": "control-system/channel-finding"})
    assert r.status_code == 200


def test_launch_factory_with_unconfigured_bundle_is_guarded(monkeypatch):
    """Section present but bundle_path null → factory passes None → guarded app."""
    from osprey.infrastructure import server_launcher
    from osprey.registry.web import FRAMEWORK_WEB_SERVERS

    monkeypatch.setattr(
        server_launcher, "load_osprey_config", lambda: {"facility_knowledge": {}}
    )
    factory = server_launcher._make_app_factory(FRAMEWORK_WEB_SERVERS["okf"])
    app = factory()
    c = TestClient(app)
    assert c.get("/health").json()["configured"] is False
    assert c.get("/api/concepts").status_code == 503


# ---------------------------------------------------------------------------
# bundle_path normalization — must match the MCP server + CLI (shared config key)
# ---------------------------------------------------------------------------


def _write_min_concept(root: Path, cid: str) -> None:
    p = root / f"{cid}.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("---\ntype: concept\ntitle: X\ndescription: d\n---\n\nbody\n", encoding="utf-8")


def test_relative_bundle_path_resolves_against_config_dir(monkeypatch, tmp_path):
    """A relative bundle_path resolves against the config.yml dir, not process CWD.

    (Regression for the review finding: MCP/CLI resolve it this way; the panel
    must agree or a valid relative bundle_path silently yields an empty tab.)
    """
    import osprey.utils.workspace as workspace

    config_yml = tmp_path / "config.yml"
    config_yml.write_text("", encoding="utf-8")
    _write_min_concept(tmp_path / "kb", "devices/bpm")
    # app.py imports resolve_config_path lazily from this module → patch here.
    monkeypatch.setattr(workspace, "resolve_config_path", lambda: config_yml)

    c = TestClient(create_app("kb"))  # relative
    assert c.get("/health").json()["configured"] is True
    assert any(g["id"] == "devices" for g in c.get("/api/concepts").json()["groups"])


def test_tilde_bundle_path_is_expanded(monkeypatch, tmp_path):
    """A ``~/...`` bundle_path is expanded (matches the CLI), not treated literally."""
    home = tmp_path / "home"
    _write_min_concept(home / "kb", "refs/glossary")
    monkeypatch.setenv("HOME", str(home))

    c = TestClient(create_app("~/kb"))
    assert c.get("/health").json()["configured"] is True
    assert any(g["id"] == "refs" for g in c.get("/api/concepts").json()["groups"])
