"""Tests for the facility knowledge MCP tools: list_concepts, read_concept, search.

Tools are tested by patching the ``_bundle`` singleton in the server module
so that tests run against an in-process OKFBundle backed by a tmp_path fixture
bundle — no live MCP transport needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import pytest

from osprey.deployment.qmd_service import QMDServiceConfig
from osprey.services.qmd import QMDClient, QMDSearchResult
from tests.mcp_server.conftest import assert_raises_error, get_tool_fn

# ---------------------------------------------------------------------------
# Bundle fixtures
# ---------------------------------------------------------------------------


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(body).lstrip(), encoding="utf-8")


@pytest.fixture
def fixture_bundle(tmp_path: Path) -> Path:
    """Minimal OKF bundle used across all tool tests."""
    root = tmp_path / "bundle"
    _write(
        root / "index.md",
        """\
        # Facility Bundle
        * [Accelerator Overview](/accelerator_overview.md) - High-level accelerator overview.
        """,
    )
    _write(
        root / "accelerator_overview.md",
        """\
        ---
        type: facility_overview
        title: Accelerator Overview
        description: Overview of the accelerator complex.
        ---

        The facility uses a synchrotron ring to produce bright X-ray light.
        """,
    )
    _write(
        root / "tables" / "beam_params.md",
        """\
        ---
        type: data_table
        title: Beam Parameters
        description: Live beam parameter PVs from the EPICS control system.
        tags: [epics, beam]
        ---

        Contains PV names for all beam parameters.
        """,
    )
    return root


class _FakeQMDClient(QMDClient):
    """A configured client that answers every query with canned hits.

    Subclasses the real client so the bundle is driven through the type it
    actually holds; the MCP conversation itself is proven in
    ``tests/services/qmd/test_client.py``.

    Args:
        hits: Results :meth:`query` returns.
    """

    def __init__(self, hits: list[QMDSearchResult]) -> None:
        super().__init__(QMDServiceConfig(port=8180))
        self._hits = hits

    def is_available(self) -> bool:
        """Report the sidecar as up without probing it."""
        return True

    def query(self, *args: object, **kwargs: object) -> list[QMDSearchResult]:
        """Return the canned hits."""
        return self._hits


@pytest.fixture(autouse=True)
def _patch_bundle(fixture_bundle: Path, monkeypatch):
    """Inject the fixture bundle into the server module's ``_bundle`` global."""
    import osprey.mcp_server.facility_knowledge.server as srv
    from osprey.services.facility_knowledge.okf.bundle import OKFBundle

    monkeypatch.setattr(srv, "_bundle", OKFBundle(fixture_bundle))


# ---------------------------------------------------------------------------
# list_concepts
# ---------------------------------------------------------------------------


class TestListConcepts:
    """list_concepts returns the index-level listing without reading doc bodies."""

    @pytest.mark.asyncio
    async def test_returns_concepts_list(self):
        from osprey.mcp_server.facility_knowledge.server import list_concepts

        result = json.loads(await get_tool_fn(list_concepts)())

        assert "concepts" in result
        assert "count" in result
        assert result["count"] == len(result["concepts"])

    @pytest.mark.asyncio
    async def test_includes_expected_concept_ids(self):
        from osprey.mcp_server.facility_knowledge.server import list_concepts

        result = json.loads(await get_tool_fn(list_concepts)())
        ids = {c["concept_id"] for c in result["concepts"]}

        assert "accelerator_overview" in ids
        assert "tables/beam_params" in ids

    @pytest.mark.asyncio
    async def test_each_entry_has_required_fields(self):
        from osprey.mcp_server.facility_knowledge.server import list_concepts

        result = json.loads(await get_tool_fn(list_concepts)())

        for entry in result["concepts"]:
            assert "concept_id" in entry
            assert "title" in entry
            assert "description" in entry

    @pytest.mark.asyncio
    async def test_index_md_not_in_results(self):
        """Reserved files must never appear as concepts."""
        from osprey.mcp_server.facility_knowledge.server import list_concepts

        result = json.loads(await get_tool_fn(list_concepts)())
        ids = {c["concept_id"] for c in result["concepts"]}

        assert "index" not in ids

    @pytest.mark.asyncio
    async def test_returns_error_when_bundle_not_initialised(self, monkeypatch):
        import osprey.mcp_server.facility_knowledge.server as srv
        from osprey.mcp_server.facility_knowledge.server import list_concepts

        monkeypatch.setattr(srv, "_bundle", None)

        with assert_raises_error(error_type="server_not_initialised"):
            await get_tool_fn(list_concepts)()


# ---------------------------------------------------------------------------
# read_concept
# ---------------------------------------------------------------------------


class TestReadConcept:
    """read_concept loads a concept document by OKF §2 ID."""

    @pytest.mark.asyncio
    async def test_reads_existing_concept(self):
        from osprey.mcp_server.facility_knowledge.server import read_concept

        result = json.loads(await get_tool_fn(read_concept)(concept_id="accelerator_overview"))

        assert result["concept_id"] == "accelerator_overview"
        assert result["frontmatter"]["type"] == "facility_overview"
        assert "synchrotron" in result["body"]

    @pytest.mark.asyncio
    async def test_reads_nested_concept(self):
        from osprey.mcp_server.facility_knowledge.server import read_concept

        result = json.loads(await get_tool_fn(read_concept)(concept_id="tables/beam_params"))

        assert result["frontmatter"]["title"] == "Beam Parameters"

    @pytest.mark.asyncio
    async def test_not_found_returns_error_envelope(self):
        from osprey.mcp_server.facility_knowledge.server import read_concept

        with assert_raises_error(error_type="not_found") as ctx:
            await get_tool_fn(read_concept)(concept_id="no/such/concept")

        assert (
            "no/such/concept" in ctx["envelope"]["error_message"]
            or len(ctx["envelope"]["suggestions"]) > 0
        )

    @pytest.mark.asyncio
    async def test_broken_cross_links_still_load(self, fixture_bundle: Path):
        """A concept with dangling cross-links must load (OKF §9 tolerance)."""
        _write(
            fixture_bundle / "dangling.md",
            """\
            ---
            type: note
            title: Dangling
            description: Has a broken link.
            ---
            See [missing](/tables/ghost.md) for details.
            """,
        )
        from osprey.mcp_server.facility_knowledge.server import read_concept

        result = json.loads(await get_tool_fn(read_concept)(concept_id="dangling"))

        assert result["frontmatter"]["title"] == "Dangling"
        assert "ghost" in result["body"]

    @pytest.mark.asyncio
    async def test_returns_error_when_bundle_not_initialised(self, monkeypatch):
        import osprey.mcp_server.facility_knowledge.server as srv
        from osprey.mcp_server.facility_knowledge.server import read_concept

        monkeypatch.setattr(srv, "_bundle", None)

        with assert_raises_error(error_type="server_not_initialised"):
            await get_tool_fn(read_concept)(concept_id="accelerator_overview")


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


class TestSearch:
    """search matches across frontmatter values and body text."""

    @pytest.mark.asyncio
    async def test_matches_body_text(self):
        from osprey.mcp_server.facility_knowledge.server import search

        result = json.loads(await get_tool_fn(search)(query="synchrotron"))

        ids = {r["concept_id"] for r in result["results"]}
        assert "accelerator_overview" in ids

    @pytest.mark.asyncio
    async def test_matches_frontmatter_value(self):
        from osprey.mcp_server.facility_knowledge.server import search

        result = json.loads(await get_tool_fn(search)(query="epics"))

        ids = {r["concept_id"] for r in result["results"]}
        assert "tables/beam_params" in ids

    @pytest.mark.asyncio
    async def test_case_insensitive(self):
        from osprey.mcp_server.facility_knowledge.server import search

        lower = {
            r["concept_id"]
            for r in json.loads(await get_tool_fn(search)(query="synchrotron"))["results"]
        }
        upper = {
            r["concept_id"]
            for r in json.loads(await get_tool_fn(search)(query="SYNCHROTRON"))["results"]
        }

        assert lower == upper

    @pytest.mark.asyncio
    async def test_no_match_returns_empty_results(self):
        from osprey.mcp_server.facility_knowledge.server import search

        result = json.loads(await get_tool_fn(search)(query="zyxwvutsrqponmlkjihgfedcba"))

        assert result["results"] == []
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_result_has_required_fields(self):
        from osprey.mcp_server.facility_knowledge.server import search

        result = json.loads(await get_tool_fn(search)(query="beam"))

        assert result["results"]
        for r in result["results"]:
            assert "concept_id" in r
            assert "title" in r
            assert "description" in r
            assert "score" in r
            assert "snippet" in r

    @pytest.mark.asyncio
    async def test_fallback_results_score_null(self):
        """No sidecar means no ranking, and the response has to say so."""
        from osprey.mcp_server.facility_knowledge.server import search

        result = json.loads(await get_tool_fn(search)(query="beam"))

        assert result["results"]
        assert all(r["score"] is None for r in result["results"])

    @pytest.mark.asyncio
    async def test_fallback_snippet_is_the_body_prefix(self):
        """Without a sidecar excerpt the snippet leads with the body."""
        from osprey.mcp_server.facility_knowledge.server import search

        result = json.loads(await get_tool_fn(search)(query="synchrotron"))
        by_id = {r["concept_id"]: r for r in result["results"]}

        assert by_id["accelerator_overview"]["snippet"].startswith(
            "The facility uses a synchrotron"
        )

    @pytest.mark.asyncio
    async def test_snippet_is_truncated(self, fixture_bundle: Path):
        """Body snippets must be at most 200 characters."""
        long_body = "word " * 200
        _write(
            fixture_bundle / "long_doc.md",
            f"---\ntype: note\ntitle: Long\ndescription: A long doc.\n---\n\n{long_body}\n",
        )
        from osprey.mcp_server.facility_knowledge.server import search

        result = json.loads(await get_tool_fn(search)(query="word"))
        by_id = {r["concept_id"]: r for r in result["results"]}

        assert "long_doc" in by_id
        assert len(by_id["long_doc"]["snippet"]) <= 200

    @pytest.mark.asyncio
    async def test_returns_query_in_response(self):
        from osprey.mcp_server.facility_knowledge.server import search

        result = json.loads(await get_tool_fn(search)(query="beam"))

        assert result["query"] == "beam"

    @pytest.mark.asyncio
    async def test_returns_error_when_bundle_not_initialised(self, monkeypatch):
        import osprey.mcp_server.facility_knowledge.server as srv
        from osprey.mcp_server.facility_knowledge.server import search

        monkeypatch.setattr(srv, "_bundle", None)

        with assert_raises_error(error_type="server_not_initialised"):
            await get_tool_fn(search)(query="anything")


class TestSearchRanked:
    """What the tool reports when a sidecar ranks the hits.

    The bundle's own mapping of qmd hits onto concepts is covered in
    ``tests/services/facility_knowledge/test_okf_ranked_search.py``; what is
    proven here is only what the MCP response adds on top of it.
    """

    @pytest.fixture
    def ranked_bundle(self, fixture_bundle: Path, monkeypatch):
        """Install a bundle whose sidecar returns caller-supplied hits."""
        import osprey.mcp_server.facility_knowledge.server as srv
        from osprey.services.facility_knowledge.okf.bundle import OKF_COLLECTION, OKFBundle

        def install(*hits: QMDSearchResult) -> None:
            monkeypatch.setattr(
                srv, "_bundle", OKFBundle(fixture_bundle, qmd_client=_FakeQMDClient(list(hits)))
            )

        def hit(file: str, *, score: float, snippet: str) -> QMDSearchResult:
            return QMDSearchResult(
                docid="#abc123",
                file=file,
                collection=OKF_COLLECTION,
                title="",
                score=score,
                line=1,
                snippet=snippet,
            )

        install.hit = hit  # type: ignore[attr-defined]
        return install

    @pytest.mark.asyncio
    async def test_ranked_results_carry_the_score(self, ranked_bundle):
        from osprey.mcp_server.facility_knowledge.server import search

        ranked_bundle(
            ranked_bundle.hit("accelerator_overview.md", score=1.0, snippet="1: ring"),
            ranked_bundle.hit("tables/beam_params.md", score=0.5, snippet="1: pvs"),
        )

        result = json.loads(await get_tool_fn(search)(query="what steers the beam"))

        assert [(r["concept_id"], r["score"]) for r in result["results"]] == [
            ("accelerator_overview", 1.0),
            ("tables/beam_params", 0.5),
        ]

    @pytest.mark.asyncio
    async def test_ranked_snippet_is_the_sidecar_excerpt(self, ranked_bundle):
        """The match-centred excerpt wins over the body prefix."""
        from osprey.mcp_server.facility_knowledge.server import search

        ranked_bundle(
            ranked_bundle.hit(
                "accelerator_overview.md",
                score=1.0,
                snippet="7: bright X-ray light is produced here",
            )
        )

        result = json.loads(await get_tool_fn(search)(query="x-rays"))

        assert result["results"][0]["snippet"] == "7: bright X-ray light is produced here"

    @pytest.mark.asyncio
    async def test_ranked_snippet_falls_back_to_body_when_absent(self, ranked_bundle):
        """A ranked hit with no excerpt still gets a usable snippet."""
        from osprey.mcp_server.facility_knowledge.server import search

        ranked_bundle(ranked_bundle.hit("accelerator_overview.md", score=1.0, snippet=""))

        result = json.loads(await get_tool_fn(search)(query="x-rays"))

        assert result["results"][0]["snippet"].startswith("The facility uses a synchrotron")

    @pytest.mark.asyncio
    async def test_ranked_entry_reports_concept_id_and_frontmatter(self, ranked_bundle):
        """A hit names a qmd path; the response names the concept it is."""
        from osprey.mcp_server.facility_knowledge.server import search

        ranked_bundle(ranked_bundle.hit("tables/beam_params.md", score=0.33, snippet="1: epics"))

        result = json.loads(await get_tool_fn(search)(query="control system"))

        assert result["results"][0]["concept_id"] == "tables/beam_params"
        assert result["results"][0]["title"] == "Beam Parameters"


# ---------------------------------------------------------------------------
# create_server() lifecycle paths
# ---------------------------------------------------------------------------


class TestCreateServer:
    """create_server() success and not-initialised paths."""

    def test_success_path_sets_bundle(self, fixture_bundle: Path, monkeypatch):
        """create_server() with a valid config sets _bundle to an OKFBundle."""
        import osprey.mcp_server.facility_knowledge.server as srv
        import osprey.utils.workspace as ws
        from osprey.services.facility_knowledge.okf.bundle import OKFBundle

        monkeypatch.setattr(srv, "_bundle", None)

        config_file = fixture_bundle.parent / "config.yml"
        config_file.write_text("", encoding="utf-8")
        config = {"facility_knowledge": {"bundle_path": str(fixture_bundle)}}

        monkeypatch.setattr(ws, "resolve_config_path", lambda: config_file)
        monkeypatch.setattr(ws, "load_osprey_config", lambda: config)

        result = srv.create_server()

        assert srv._bundle is not None
        assert isinstance(srv._bundle, OKFBundle)
        assert result is srv.mcp

    def test_missing_key_sets_none(self, tmp_path: Path, monkeypatch):
        """create_server() with no facility_knowledge key leaves _bundle=None (no crash)."""
        import osprey.mcp_server.facility_knowledge.server as srv
        import osprey.utils.workspace as ws

        monkeypatch.setattr(srv, "_bundle", None)

        config_file = tmp_path / "config.yml"
        config_file.write_text("", encoding="utf-8")

        monkeypatch.setattr(ws, "resolve_config_path", lambda: config_file)
        monkeypatch.setattr(ws, "load_osprey_config", lambda: {})  # no key

        srv.create_server()

        assert srv._bundle is None

    def test_bad_bundle_path_sets_none(self, tmp_path: Path, monkeypatch):
        """create_server() with a bundle_path that doesn't exist logs an error, _bundle=None."""
        import osprey.mcp_server.facility_knowledge.server as srv
        import osprey.utils.workspace as ws

        monkeypatch.setattr(srv, "_bundle", None)

        config_file = tmp_path / "config.yml"
        config_file.write_text("", encoding="utf-8")
        config = {"facility_knowledge": {"bundle_path": str(tmp_path / "nonexistent")}}

        monkeypatch.setattr(ws, "resolve_config_path", lambda: config_file)
        monkeypatch.setattr(ws, "load_osprey_config", lambda: config)

        srv.create_server()

        assert srv._bundle is None

    @pytest.mark.asyncio
    async def test_tools_work_after_create_server(self, fixture_bundle: Path, monkeypatch):
        """After create_server() succeeds, list_concepts/read_concept/search all work."""
        import osprey.mcp_server.facility_knowledge.server as srv
        import osprey.utils.workspace as ws
        from osprey.mcp_server.facility_knowledge.server import list_concepts

        monkeypatch.setattr(srv, "_bundle", None)

        config_file = fixture_bundle.parent / "config.yml"
        if not config_file.exists():
            config_file.write_text("", encoding="utf-8")
        config = {"facility_knowledge": {"bundle_path": str(fixture_bundle)}}

        monkeypatch.setattr(ws, "resolve_config_path", lambda: config_file)
        monkeypatch.setattr(ws, "load_osprey_config", lambda: config)

        srv.create_server()

        result = json.loads(await get_tool_fn(list_concepts)())
        assert "concepts" in result


# ---------------------------------------------------------------------------
# Exception handler coverage (internal_error paths)
# ---------------------------------------------------------------------------


class TestToolInternalErrors:
    """Exercise the internal_error except-Exception branches in each tool."""

    @pytest.mark.asyncio
    async def test_list_concepts_internal_error(self, monkeypatch):
        """list_concepts returns internal_error when bundle.list_concepts raises unexpectedly."""
        import osprey.mcp_server.facility_knowledge.server as srv
        from osprey.mcp_server.facility_knowledge.server import list_concepts

        monkeypatch.setattr(
            srv._bundle, "list_concepts", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        )

        with assert_raises_error(error_type="internal_error"):
            await get_tool_fn(list_concepts)()

    @pytest.mark.asyncio
    async def test_read_concept_internal_error(self, monkeypatch):
        """read_concept returns internal_error for unexpected (non-OKFBundleError) exceptions."""
        import osprey.mcp_server.facility_knowledge.server as srv
        from osprey.mcp_server.facility_knowledge.server import read_concept

        monkeypatch.setattr(
            srv._bundle,
            "read_concept",
            lambda concept_id: (_ for _ in ()).throw(RuntimeError("unexpected")),
        )

        with assert_raises_error(error_type="internal_error"):
            await get_tool_fn(read_concept)(concept_id="accelerator_overview")

    @pytest.mark.asyncio
    async def test_search_internal_error(self, monkeypatch):
        """search returns internal_error when bundle.search raises unexpectedly."""
        import osprey.mcp_server.facility_knowledge.server as srv
        from osprey.mcp_server.facility_knowledge.server import search

        monkeypatch.setattr(
            srv._bundle,
            "search",
            lambda query: (_ for _ in ()).throw(RuntimeError("search boom")),
        )

        with assert_raises_error(error_type="internal_error"):
            await get_tool_fn(search)(query="anything")

    @pytest.mark.asyncio
    async def test_read_concept_not_found_via_okf_error(self, monkeypatch):
        """read_concept maps OKFBundleError → not_found (distinct from internal_error)."""
        import osprey.mcp_server.facility_knowledge.server as srv
        from osprey.mcp_server.facility_knowledge.server import read_concept
        from osprey.services.facility_knowledge.okf.bundle import OKFBundleError

        monkeypatch.setattr(
            srv._bundle,
            "read_concept",
            lambda concept_id: (_ for _ in ()).throw(OKFBundleError("missing")),
        )

        with assert_raises_error(error_type="not_found"):
            await get_tool_fn(read_concept)(concept_id="no/such")
