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

Search is covered in both of its backends — ranked through a stub qmd sidecar
and unranked through the substring fallback — since the panel renders the two
differently and both are normal states.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.okf_panel.app import create_app
from osprey.services.facility_knowledge.okf.bundle import OKF_COLLECTION
from osprey.services.qmd import QMDSearchResult

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
    assert '<div id="app">' in r.text


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


def test_search_without_sidecar_scores_every_hit_null(client):
    """No sidecar → substring backend → every hit carries an explicit null score.

    The key is present rather than absent: the front end decides between the
    ranked and unranked presentations by reading it, so a missing key would be
    a different contract than an unranked one.
    """
    results = client.get("/api/search", params={"q": "GEECS"}).json()["results"]
    assert results
    assert all(r["score"] is None for r in results)
    assert all(set(r) == {"id", "title", "snippet", "score"} for r in results)


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
# Ranked search (qmd sidecar answering)
#
# The sidecar itself is out of scope here — 3.7 covers the real container. What
# these tests pin is the panel's half of the contract: an available client
# makes /api/search return qmd's order, qmd's scores and qmd's excerpts, and an
# unavailable one leaves the payload exactly as the block above asserts it.
# ---------------------------------------------------------------------------


class _StubQMDClient:
    """Stand-in for :class:`QMDClient` that answers with a fixed result set.

    Only the three members :meth:`OKFBundle.search` touches are implemented.
    """

    is_configured = True

    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def is_available(self) -> bool:
        return True

    def query(self, collection, query, **kwargs):
        self.calls.append((collection, query, kwargs))
        return list(self.hits)


def _hit(concept_id: str, score: float, snippet: str = "") -> QMDSearchResult:
    """Build one qmd hit for *concept_id*, in the shape the daemon returns."""
    return QMDSearchResult(
        docid="#" + concept_id[:6],
        file=f"{concept_id}.md",  # collection prefix already stripped by the client
        collection="okf",
        title=concept_id,
        score=score,
        line=1,
        snippet=snippet,
    )


def _ranked_client(app) -> _StubQMDClient:
    """Attach a stub sidecar to *app*'s bundle and return it."""
    client = _StubQMDClient(
        [
            _hit("devices/rf-system", 1.0, "1: @@ -1,3 @@ ...\n2: The RF system provides"),
            _hit("devices/bpm", 0.5),
        ]
    )
    app.state.bundle.qmd_client = client
    return client


def test_search_ranked_results_keep_qmd_order_and_scores():
    app = create_app(str(BUNDLE))
    _ranked_client(app)
    c = TestClient(app)

    results = c.get("/api/search", params={"q": "how is the beam accelerated"}).json()["results"]

    # qmd's order is preserved verbatim — the panel never re-sorts.
    assert [r["id"] for r in results] == ["devices/rf-system", "devices/bpm"]
    assert [r["score"] for r in results] == [1.0, 0.5]
    assert [r["title"] for r in results] == ["RF System", "Beam Position Monitor"]


def test_search_ranked_hit_uses_the_qmd_excerpt():
    """qmd's match-centered excerpt is shown, stripped of its line markup."""
    app = create_app(str(BUNDLE))
    _ranked_client(app)
    c = TestClient(app)

    results = c.get("/api/search", params={"q": "acceleration"}).json()["results"]
    assert results[0]["snippet"] == "The RF system provides"


def test_search_ranked_hit_without_excerpt_falls_back_to_description():
    """A semantic hit need not contain the query text, so both local snippet
    sources can come up empty; the concept's own description is the last resort
    and keeps every result card populated."""
    app = create_app(str(BUNDLE))
    _ranked_client(app)
    c = TestClient(app)

    results = c.get("/api/search", params={"q": "where is the beam"}).json()["results"]
    bpm = next(r for r in results if r["id"] == "devices/bpm")
    assert bpm["snippet"] == "Measures transverse beam position."


def test_search_ranked_hit_with_markup_only_excerpt_uses_the_local_snippet():
    """Middle rung of the snippet chain: qmd's excerpt collapses to nothing, so
    the local substring snippet supplies the text — not the description, which
    is the last resort and carries less of the match."""
    app = create_app(str(BUNDLE))
    app.state.bundle.qmd_client = _StubQMDClient(
        [_hit("devices/rf-system", 1.0, "1: @@ -1,3 @@ ...")]
    )
    c = TestClient(app)

    results = c.get("/api/search", params={"q": "accelerating voltage"}).json()["results"]
    assert results[0]["snippet"] == "The RF system provides the accelerating voltage."


def test_search_forwards_the_query_to_the_okf_collection():
    app = create_app(str(BUNDLE))
    client = _ranked_client(app)
    TestClient(app).get("/api/search", params={"q": "rf cavities"})

    collection, query, _kwargs = client.calls[0]
    assert (collection, query) == (OKF_COLLECTION, "rf cavities")


def test_unavailable_sidecar_falls_back_to_substring_search():
    """A configured-but-down sidecar renders the unranked payload, not an error."""
    app = create_app(str(BUNDLE))
    client = _ranked_client(app)
    client.is_available = lambda: False  # type: ignore[method-assign]
    c = TestClient(app)

    results = c.get("/api/search", params={"q": "GEECS"}).json()["results"]
    assert [r["score"] for r in results] == [None] * len(results)
    assert results


async def test_slow_search_does_not_block_the_rest_of_the_panel():
    """A slow sidecar must cost the search and nothing else.

    ``bundle.search`` blocks on HTTP once a sidecar answers — up to the client's
    30 s timeout when one hangs, and seconds at a time with ``rerank: true`` in
    a perfectly healthy deployment. On the event loop that stalls every other
    request in the process, so ``/api/search`` is a sync route and Starlette
    runs it in a threadpool. Both halves are asserted: the shape, which cannot
    flake, and the behaviour it exists for.
    """
    app = create_app(str(BUNDLE))
    route = next(r for r in app.routes if getattr(r, "path", "") == "/api/search")
    assert not inspect.iscoroutinefunction(route.endpoint)  # type: ignore[attr-defined]

    blocked = 1.0
    real_search = app.state.bundle.search

    def slow_search(query, **kwargs):
        time.sleep(blocked)
        return real_search(query, **kwargs)

    app.state.bundle.search = slow_search

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://panel") as client:
        started = time.monotonic()

        async def search():
            await client.get("/api/search", params={"q": "GEECS"})
            return time.monotonic() - started

        async def concept():
            await client.get("/api/concept", params={"id": "devices/bpm"})
            return time.monotonic() - started

        search_done, concept_done = await asyncio.gather(search(), concept())

    assert search_done >= blocked  # the stub really did block
    # Generous margin: the point is "did not wait for the search", not a
    # latency budget. A blocking handler lands at ~1.0 s here.
    assert concept_done < blocked / 2


# ---------------------------------------------------------------------------
# Ranked-backend wiring: the factory reads the project config itself
# ---------------------------------------------------------------------------


def test_factory_wires_a_qmd_client_from_the_project_config(monkeypatch):
    """``services.qmd`` in config.yml reaches the bundle the panel serves."""
    import osprey.utils.workspace as workspace

    monkeypatch.setattr(
        workspace,
        "load_osprey_config",
        lambda: {"services": {"qmd": {"port": 8180}}},
    )
    app = create_app(str(BUNDLE))
    assert app.state.bundle.qmd_client.is_configured is True


def test_factory_without_services_qmd_opens_no_socket(monkeypatch):
    """No sidecar is a supported configuration: the client stays unconfigured."""
    import osprey.utils.workspace as workspace

    monkeypatch.setattr(workspace, "load_osprey_config", lambda: {})
    app = create_app(str(BUNDLE))
    assert app.state.bundle.qmd_client.is_configured is False


def test_malformed_search_settings_degrade_to_substring_not_to_a_dead_panel(monkeypatch, caplog):
    """A bad search knob must not cost the operator the whole reading pane."""
    import osprey.utils.workspace as workspace

    monkeypatch.setattr(
        workspace,
        "load_osprey_config",
        lambda: {"facility_knowledge": {"search": {"rerank": "yes"}}},
    )
    with caplog.at_level(logging.WARNING):
        app = create_app(str(BUNDLE))
    c = TestClient(app)

    assert app.state.bundle.qmd_client is None
    assert c.get("/health").json()["configured"] is True
    assert c.get("/api/search", params={"q": "GEECS"}).json()["results"]
    assert any("misconfigured" in r.message for r in caplog.records)


def test_unreadable_config_degrades_to_substring_not_to_a_dead_panel(monkeypatch, caplog):
    """The same guarantee for a failure that is not a ValueError.

    The backend resolver runs inside the factory's own catch-all, so anything
    escaping it costs the whole panel rather than just ranked search — the
    reading pane included. This pins that nothing gets that far, whatever the
    config layer raises.
    """
    import osprey.utils.workspace as workspace

    def boom():
        raise TypeError("config layer changed shape")

    monkeypatch.setattr(workspace, "load_osprey_config", boom)
    with caplog.at_level(logging.WARNING):
        app = create_app(str(BUNDLE))
    c = TestClient(app)

    assert app.state.bundle.qmd_client is None
    assert c.get("/health").json()["configured"] is True
    assert c.get("/api/search", params={"q": "GEECS"}).json()["results"]
    assert any("misconfigured" in r.message for r in caplog.records)


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

    monkeypatch.setattr(server_launcher, "load_osprey_config", lambda: {"facility_knowledge": {}})
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
