"""Tests for the Channel Finder graph-paradigm REST endpoints.

The explorer's search, class tree and statistics read the search index the app
opened at startup; the device card still reads the store. That puts three
things under test that the file-backed routes never face: the reads must not
run on the event loop, an app without a usable index must answer with the
build remedy, and an index that binds nothing must be told apart from one that
is missing.

The index is real: ``graph_fixture`` emits the demo corpus as Turtle and runs
it through the same parse -> derive -> write path ``osprey build`` runs, so a
number asserted here is a number a builder actually wrote. The awkward shapes —
cycles, deep chains, shared parents, a corpus with no bindings — are built the
same way from the small corpora the index package's own tests use. The store is
faked, as before, for the one read that still goes to it.
"""

from __future__ import annotations

import asyncio
import hashlib
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from osprey.deployment.graphdb_service import GRAPHDB_BUILD_INDEX_COMMAND
from osprey.interfaces.channel_finder import database_api
from osprey.interfaces.channel_finder.app import _open_graph_index
from osprey.interfaces.channel_finder.database_api import UNRESOLVED_INDEX_PATH_REMEDY
from osprey.mcp_server.graph.server_context import GraphUnreachable
from osprey.services.channel_finder.graph_index.builder import (
    build_from_rows,
    channels_from_rows,
    parse_corpus,
)
from osprey.services.channel_finder.graph_index.reader import (
    BUILD_TTL_COMMAND,
    GraphIndex,
    GraphIndexAbsence,
    open_graph_index,
)
from osprey.services.channel_finder.graph_queries import GRAPH_DEVICE_CYPHER
from tests.interfaces.channel_finder.graph_fixture import (
    DEMO_DEVICE_ROW,
    DEMO_PAGE_COUNT,
    DEMO_ROWS,
    SEARCH_FACET_CAP,
    SEARCH_PAGE_SIZE,
    FakeGraphContext,
    class_uri,
    demo_context,
    demo_page,
    device_uri,
    install_graph_paradigm,
    open_demo_index,
    signal_uri,
)
from tests.services.channel_finder.graph_index import corpora

#: The devices the demo corpus binds — what the root class rolls up to and what
#: an unfiltered search counts.
DEMO_PAGE_DEVICES = {row["device"] for row in DEMO_ROWS}

#: A corpus whose one device is typed by a class with two parents: the tree is
#: a DAG, and a class must appear once carrying both.
TWO_PARENTS = (
    corpora.PREFIXES
    + """
narad_sem:ChannelBinding a owl:Class .
narad_sem:AcceleratorDevice a owl:Class .
narad_sem:Magnet a owl:Class ;
    rdfs:subClassOf narad_sem:AcceleratorDevice .
narad_sem:SteeringDevice a owl:Class ;
    rdfs:subClassOf narad_sem:AcceleratorDevice .
narad_sem:Corrector a owl:Class ;
    rdfs:subClassOf narad_sem:Magnet, narad_sem:SteeringDevice .

<https://narad.example.org/device/demo_HCM1> a narad_sem:Corrector ;
    narad_p:hasBinding <https://narad.example.org/binding/HCM1_SP> ;
    narad_p:sourceName "HCM1" .

<https://narad.example.org/binding/HCM1_SP> a narad_sem:ChannelBinding ;
    narad_p:fullPv "SR:HCM1:SP" .
"""
)


def _sem(name: str) -> str:
    """Return the URI of a class in the small corpora's namespace."""
    return corpora.NARAD_SEM + name


@contextmanager
def _index_over(text: str, directory: Path) -> Iterator[GraphIndex]:
    """Build an index over the corpus *text* into *directory* and open it.

    The same path the fixture's demo index takes, on a corpus small enough to
    reason about by hand.
    """
    parsed = parse_corpus(text)
    index_path = directory / "graph.duckdb"
    build_from_rows(
        parsed.binding_rows,
        parsed.class_rows,
        channels_from_rows(parsed.binding_rows),
        index_path,
        {
            "corpus_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "corpus_filename": "small.ttl",
            "binding_count": len(parsed.binding_rows),
            "device_count": len({row.device_uri for row in parsed.binding_rows}),
            "class_count": len(parsed.class_rows),
            "signal_count": parsed.signal_count,
            "section_count": len(parsed.section_codes),
        },
    )
    opened = open_graph_index(index_path)
    assert isinstance(opened, GraphIndex), opened
    with opened:
        yield opened


def _loop_thread_id(client) -> int:
    """Return the thread the app's event loop runs on, measured from inside it.

    The loop runs on a thread of the test client's own choosing, so the thread
    an index read must not land on is captured from a coroutine the same client
    drives rather than assumed to be the calling one.
    """
    seen: list[int] = []

    @client.app.get("/api/_loop_thread_probe")
    async def _loop_thread_probe():  # pragma: no cover - trivial probe
        seen.append(threading.get_ident())
        return {"ok": True}

    assert client.get("/api/_loop_thread_probe").status_code == 200
    assert seen, "probe route never ran"
    return seen[0]


def _loop_is_running() -> bool:
    """Whether the calling thread is inside a running event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _record_cursor_calls(index: GraphIndex) -> list[tuple[int, bool]]:
    """Note the thread every cursor the index opens is opened on.

    Every index read that scans the file opens one cursor, so the cursor is
    where "the read ran off the loop" can be observed without faking the
    driver.
    """
    seen: list[tuple[int, bool]] = []
    original = index.cursor

    def cursor():
        seen.append((threading.get_ident(), _loop_is_running()))
        return original()

    index.cursor = cursor  # type: ignore[method-assign]
    return seen


def _spy_search(index: GraphIndex) -> list[dict[str, Any]]:
    """Record the keyword arguments of every search the index is asked for.

    The real search still runs, so the answer stays the index's own; only the
    request contract between route and reader is captured.
    """
    calls: list[dict[str, Any]] = []
    original = index.search

    def search(**kwargs: Any):
        calls.append(kwargs)
        return original(**kwargs)

    index.search = search  # type: ignore[method-assign]
    return calls


@pytest.fixture
def graph_config(monkeypatch):
    """Point the config seam at a project that has a corpus.

    The 503 remedy depends on whether a corpus is configured; pinning one makes
    the remedy the build command rather than the configure-a-corpus sentence.
    """
    monkeypatch.setattr(
        "osprey.utils.workspace.load_osprey_config",
        lambda: {"services": {"graphdb": {"ttl_path": "./data/facility.ttl"}}},
    )


class TestOntologyPayload:
    """What the endpoint draws from the index the demo corpus builds."""

    def test_demo_corpus_yields_the_device_taxonomy(self, client):
        install_graph_paradigm(client, demo_context())

        resp = client.get("/api/graph/ontology")

        assert resp.status_code == 200
        data = resp.json()
        # Every class the build kept, and nothing else: the badge the
        # statistics route shows counts the same rows.
        assert len(data["classes"]) == client.app.state.graph_index.meta.class_count
        names = {entry["name"] for entry in data["classes"]}
        # Real classes in the corpus, but about signals and bindings rather
        # than devices — pruned at build time.
        assert "SemanticSignal" not in names
        assert "ChannelBinding" not in names
        # A leaf no device on the page is typed by is pruned the same way.
        assert "Undulator" not in names
        assert "Quadrupole" in names
        assert data["empty"] is False
        assert data["truncated"] is False
        assert data["suggestions"] == []

    def test_root_carries_the_whole_device_population(self, client):
        install_graph_paradigm(client, demo_context())

        data = client.get("/api/graph/ontology").json()

        by_name = {entry["name"]: entry for entry in data["classes"]}
        assert by_name["AcceleratorDevice"]["rollup"] == len(DEMO_PAGE_DEVICES)
        assert by_name["AcceleratorDevice"]["parents"] == []
        assert by_name["Magnet"]["parents"] == [class_uri("AcceleratorDevice")]
        assert by_name["Quadrupole"]["parents"] == [class_uri("Magnet")]
        assert by_name["BeamPositionMonitor"]["altLabel"] == ["BPM"]

    def test_an_abstract_branch_rolls_up_what_the_class_filter_finds(self, client):
        install_graph_paradigm(client, demo_context())

        tree = client.get("/api/graph/ontology").json()
        by_name = {entry["name"]: entry for entry in tree["classes"]}
        under_magnet = client.get("/api/graph/search", params={"cls": class_uri("Magnet")}).json()

        # Nothing is typed as a Magnet, yet the branch carries every device
        # under its subclasses — the same population a selection on it finds.
        assert by_name["Magnet"]["rollup"] > 0
        assert by_name["Magnet"]["rollup"] == under_magnet["devices"]
        assert by_name["Magnet"]["rollup"] == sum(
            by_name[leaf]["rollup"] for leaf in ("Dipole", "Quadrupole", "Sextupole", "Corrector")
        )

    def test_a_branch_is_marked_abstract_and_a_leaf_is_not(self, client):
        install_graph_paradigm(client, demo_context())

        data = client.get("/api/graph/ontology").json()

        by_name = {entry["name"]: entry for entry in data["classes"]}
        assert by_name["Magnet"]["abstract"] is True
        assert by_name["Corrector"]["abstract"] is True
        assert by_name["Quadrupole"]["abstract"] is False
        assert by_name["VacuumGauge"]["abstract"] is False

    def test_the_tree_is_the_index_rows_and_is_not_pruned_again(self, client):
        install_graph_paradigm(client, demo_context())

        tree = client.get("/api/graph/ontology").json()
        badge = client.get("/api/statistics").json()

        by_name = {entry["name"]: entry for entry in tree["classes"]}
        # InsertionDevice is an abstract parent whose only leaf, Undulator, the
        # build pruned. The build kept the parent; a second pruning pass over
        # the kept rows would drop it, and the tree would no longer be the
        # population the badge counted.
        assert by_name["InsertionDevice"]["rollup"] == 0
        assert by_name["InsertionDevice"]["abstract"] is True
        assert badge["total_classes"] == len(tree["classes"])

    def test_the_relationship_vocabulary_is_not_read_from_the_index(self, client):
        install_graph_paradigm(client, demo_context())

        data = client.get("/api/graph/ontology").json()

        # No browser surface draws it; an agent asks the store through
        # get_schema. The key stays so the payload shape does not change.
        assert data["relationship_types"] == []
        assert data["truncated"] is False

    def test_the_read_never_touches_the_store(self, client):
        ctx = demo_context()
        install_graph_paradigm(client, ctx)

        assert client.get("/api/graph/ontology").status_code == 200

        assert ctx.calls == []
        assert ctx.empty_checks == 0


class TestAwkwardOntologies:
    """Corpus shapes the build can legitimately produce that must not derail the endpoint."""

    def test_a_class_with_two_parents_appears_once_carrying_both(self, client, tmp_path):
        with _index_over(TWO_PARENTS, tmp_path) as index:
            install_graph_paradigm(client, demo_context(), index=index)

            resp = client.get("/api/graph/ontology")

        assert resp.status_code == 200
        classes = resp.json()["classes"]
        corrector = [c for c in classes if c["name"] == "Corrector"]
        assert len(corrector) == 1
        assert corrector[0]["parents"] == [_sem("Magnet"), _sem("SteeringDevice")]
        # The one device is counted once under each parent, not twice at root.
        by_name = {entry["name"]: entry for entry in classes}
        assert by_name["Magnet"]["rollup"] == 1
        assert by_name["SteeringDevice"]["rollup"] == 1
        assert by_name["AcceleratorDevice"]["rollup"] == 1

    def test_a_subclass_cycle_is_answered_rather_than_hung(self, client, tmp_path):
        # A corpus whose subclass edges close a loop is malformed, but it is the
        # corpus — the endpoint answers it instead of spinning or 500ing.
        with _index_over(corpora.CLASS_CYCLE, tmp_path) as index:
            install_graph_paradigm(client, demo_context(), index=index)

            resp = client.get("/api/graph/ontology")

        assert resp.status_code == 200
        data = resp.json()
        names = {entry["name"] for entry in data["classes"]}
        assert {"Loop_A", "Loop_B", "Selfish"} <= names
        assert data["empty"] is False

    def test_a_chain_deeper_than_the_walk_is_answered_to_the_bound(self, client, tmp_path):
        with _index_over(corpora.DEEP_CHAIN, tmp_path) as index:
            install_graph_paradigm(client, demo_context(), index=index)

            resp = client.get("/api/graph/ontology")

        assert resp.status_code == 200
        by_name = {entry["name"]: entry for entry in resp.json()["classes"]}
        # The one device is typed Deep_00; ten hops up reach Deep_10, and the
        # class one hop beyond the bound is still drawn as an empty parent.
        assert by_name["Deep_00"]["rollup"] == 1
        assert by_name["Deep_10"]["rollup"] == 1
        assert by_name["Deep_11"]["rollup"] == 0


class TestEmptyIndex:
    """An index that binds nothing is a different answer from one that is missing."""

    def test_an_empty_index_answers_200_naming_the_corpus_and_the_command(self, client, tmp_path):
        with _index_over(corpora.NO_BINDINGS, tmp_path) as index:
            install_graph_paradigm(client, demo_context(), index=index)

            resp = client.get("/api/graph/ontology")

        assert resp.status_code == 200
        data = resp.json()
        assert data["empty"] is True
        assert data["classes"] == []
        assert data["relationship_types"] == []
        assert data["truncated"] is False
        assert any(BUILD_TTL_COMMAND in hint for hint in data["suggestions"])
        assert any("small.ttl" in hint for hint in data["suggestions"])

    def test_the_empty_answer_blanks_the_class_rows_the_index_still_holds(self, client, tmp_path):
        with _index_over(corpora.NO_BINDINGS, tmp_path) as index:
            install_graph_paradigm(client, demo_context(), index=index)

            held = index.ontology()["classes"]
            served = client.get("/api/graph/ontology").json()["classes"]

        # The corpus declares an ontology, so the build kept its abstract
        # parents; the explorer shows the remedy instead of an empty tree, and
        # the answer keeps the shape the empty answer always had.
        assert held != []
        assert served == []

    def test_an_unfiltered_search_of_an_empty_index_reports_it(self, client, tmp_path):
        ctx = demo_context()
        with _index_over(corpora.NO_BINDINGS, tmp_path) as index:
            install_graph_paradigm(client, ctx, index=index)

            resp = client.get("/api/graph/search")

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["devices"] == 0
        assert data["rows"] == []
        assert data["pages"] == 0
        assert data["empty"] is True
        assert any(BUILD_TTL_COMMAND in hint for hint in data["suggestions"])
        # Emptiness is a fact of the index; the store is never asked.
        assert ctx.empty_checks == 0

    def test_a_filtered_search_of_an_empty_index_still_reports_it(self, client, tmp_path):
        with _index_over(corpora.NO_BINDINGS, tmp_path) as index:
            install_graph_paradigm(client, demo_context(), index=index)

            data = client.get("/api/graph/search", params={"q": "quadrupole"}).json()

        # Whether the question was filtered says nothing about the corpus:
        # an index that binds nothing is empty for every question.
        assert data["total"] == 0
        assert data["empty"] is True
        assert data["suggestions"] != []

    @pytest.mark.parametrize(
        "params",
        [
            {"q": "nosuchthing"},
            {"section": "SR09C"},
            {"cls": class_uri("Undulator")},
        ],
    )
    def test_no_match_in_a_populated_index_is_not_a_remedy(self, client, params):
        install_graph_paradigm(client, demo_context())

        data = client.get("/api/graph/search", params=params).json()

        # A corpus that happens to match nothing is an ordinary answer.
        assert data["total"] == 0
        assert data["rows"] == []
        assert data["empty"] is False
        assert data["suggestions"] == []


class TestIndexUnavailable:
    """Every way the app can lack a usable index is a 503 carrying the build remedy."""

    ROUTES = ["/api/graph/ontology", "/api/graph/search"]

    @pytest.mark.parametrize("path", ROUTES)
    def test_an_absent_index_answers_the_reason_and_the_remedy(
        self, client, graph_config, tmp_path, path
    ):
        absence = GraphIndexAbsence("missing", tmp_path / "graph.duckdb", "No search index at g.")
        install_graph_paradigm(client, demo_context(), index=absence)

        resp = client.get(path)

        assert resp.status_code == 503
        body = resp.json()
        # Not nested under FastAPI's "detail" envelope: the web UI reads all
        # three keys off the body it is handed.
        assert body["detail"] == "No search index at g."
        assert body["error_type"] == "service_unavailable"
        assert any(GRAPHDB_BUILD_INDEX_COMMAND in line for line in body["suggestions"])

    @pytest.mark.parametrize("path", ROUTES)
    def test_an_app_holding_no_index_at_all_still_answers_the_remedy(
        self, client, graph_config, path
    ):
        install_graph_paradigm(client, demo_context(), index=None)

        resp = client.get(path)

        assert resp.status_code == 503
        body = resp.json()
        assert body["detail"] == "The search index is not open."
        assert body["error_type"] == "service_unavailable"
        assert any(GRAPHDB_BUILD_INDEX_COMMAND in line for line in body["suggestions"])

    @pytest.mark.parametrize("path", ROUTES)
    def test_a_project_with_an_index_location_but_no_corpus_is_told_to_configure_one(
        self, client, monkeypatch, tmp_path, path
    ):
        # An index location is not a build source: the build refuses without a
        # corpus, so the remedy is the corpus key, as the MCP tool says too.
        monkeypatch.setattr(
            "osprey.utils.workspace.load_osprey_config",
            lambda: {"services": {"graphdb": {"index_path": "./data/graph.duckdb"}}},
        )
        absence = GraphIndexAbsence("missing", tmp_path / "graph.duckdb", "No search index at g.")
        install_graph_paradigm(client, demo_context(), index=absence)

        body = client.get(path).json()

        assert len(body["suggestions"]) == 1
        assert "services.graphdb.ttl_path" in body["suggestions"][0]
        assert "Turtle file" in body["suggestions"][0]
        assert GRAPHDB_BUILD_INDEX_COMMAND not in body["suggestions"][0]

    @pytest.mark.parametrize("path", ROUTES)
    def test_a_malformed_index_path_names_the_key_and_no_build_step(
        self, client, graph_config, path
    ):
        # The absence the app builds for a config typo carries the fix in its
        # own sentence; a build would read the same malformed key, so none is
        # suggested on top.
        absence = _open_graph_index({"services": {"graphdb": {"index_path": 42}}})
        assert isinstance(absence, GraphIndexAbsence)
        install_graph_paradigm(client, demo_context(), index=absence)

        resp = client.get(path)

        assert resp.status_code == 503
        body = resp.json()
        assert body["detail"].endswith(UNRESOLVED_INDEX_PATH_REMEDY)
        assert "services.graphdb.index_path" in body["detail"]
        assert body["suggestions"] == []

    @pytest.mark.parametrize("path", ROUTES)
    def test_a_closed_index_answers_503_rather_than_500(self, client, graph_config, path):
        # A request racing shutdown is unavailability, not a bug in the route.
        index = open_demo_index()
        index.close()
        install_graph_paradigm(client, demo_context(), index=index)

        resp = client.get(path)

        assert resp.status_code == 503
        body = resp.json()
        assert body["error_type"] == "service_unavailable"
        assert str(index.path) in body["detail"]
        # The index was built; telling the operator to build it would be
        # untrue. Retry first, rebuild only if it keeps failing.
        assert len(body["suggestions"]) == 1
        assert body["suggestions"][0].startswith("Retry the request")
        assert GRAPHDB_BUILD_INDEX_COMMAND in body["suggestions"][0]

    @pytest.mark.parametrize("path", ROUTES)
    def test_a_read_that_fails_on_an_open_index_is_a_500_not_a_remedy(self, client, path):
        # Only the closed-index RuntimeError is unavailability; any other one
        # is a defect and must not hide behind an operator remedy.
        install_graph_paradigm(client, demo_context())
        index = client.app.state.graph_index

        def broken(*args, **kwargs):
            raise RuntimeError("a bug in the read")

        index.search = broken  # type: ignore[method-assign]
        index.ontology = broken  # type: ignore[method-assign]

        resp = client.get(path)

        assert resp.status_code == 500
        assert resp.json() == {"detail": "a bug in the read"}

    @pytest.mark.parametrize("path", ROUTES)
    def test_a_store_that_is_down_does_not_reach_the_index_reads(self, client, path):
        # The explorer no longer depends on the store for these answers: an
        # unreachable store is the device card's problem, not the finder's.
        ctx = FakeGraphContext(
            raises=GraphUnreachable(
                "Graph store at bolt://localhost:7687 is unreachable.",
                ["Start the graphdb service."],
            )
        )
        install_graph_paradigm(client, ctx)

        resp = client.get(path)

        assert resp.status_code == 200
        assert ctx.calls == []

    @pytest.mark.parametrize("path", ROUTES)
    def test_an_app_without_a_store_context_still_answers(self, client, path):
        install_graph_paradigm(client, None)

        assert client.get(path).status_code == 200


class TestOffLoopReads:
    """DuckDB's driver is synchronous; the event loop must never wait on it."""

    def test_the_ontology_scan_runs_off_the_event_loop(self, client):
        install_graph_paradigm(client, demo_context())
        cursors = _record_cursor_calls(client.app.state.graph_index)
        loop_thread_id = _loop_thread_id(client)

        assert client.get("/api/graph/ontology").status_code == 200

        # One cursor for the one scan, opened on a worker thread that is not
        # running a loop.
        assert len(cursors) == 1
        assert cursors[0][0] != loop_thread_id
        assert cursors[0][1] is False

    def test_the_search_runs_off_the_event_loop(self, client):
        install_graph_paradigm(client, demo_context())
        cursors = _record_cursor_calls(client.app.state.graph_index)
        loop_thread_id = _loop_thread_id(client)

        assert client.get("/api/graph/search", params={"q": "qfa"}).status_code == 200

        # The seven statements of a search share one cursor.
        assert len(cursors) == 1
        assert cursors[0][0] != loop_thread_id
        assert cursors[0][1] is False


class TestPipelineGating:
    """The route belongs to one paradigm, and refuses on behalf of the others."""

    def test_file_backed_paradigm_gets_404(self, client):
        # The fixture app serves in_context; the graph route is not its route.
        resp = client.get("/api/graph/ontology")

        assert resp.status_code == 404
        assert resp.json()["detail"] == "Not available for this pipeline type"

    @pytest.mark.parametrize("pipeline_type", [None, "not_a_paradigm"])
    def test_unconfigured_shell_gets_400(self, client, pipeline_type):
        client.app.state.pipeline_type = pipeline_type

        resp = client.get("/api/graph/ontology")

        assert resp.status_code == 400
        assert "channel_finder.pipeline_mode" in resp.json()["detail"]


class TestSearch:
    """The faceted finder: what the request sends and what the page answers."""

    def test_query_and_facets_reach_the_index_as_the_reader_expects(self, client):
        install_graph_paradigm(client, demo_context())
        calls = _spy_search(client.app.state.graph_index)

        resp = client.get(
            "/api/graph/search",
            params=[
                ("q", "QFA Current"),
                ("section", "SR01C"),
                ("section", "SR02C"),
                ("system", "MG"),
                ("signal", "current"),
                ("dir", "R"),
                ("cls", class_uri("Quadrupole")),
                ("page", "3"),
            ],
        )

        assert resp.status_code == 200
        assert len(calls) == 1
        kwargs = calls[0]
        # The route lower-cases its tokens because the index matches folded text.
        assert kwargs["tokens"] == ["qfa", "current"]
        # A facet given twice arrives as a list, not as the last value.
        assert kwargs["sections"] == ["SR01C", "SR02C"]
        assert kwargs["systems"] == ["MG"]
        assert kwargs["signals"] == ["current"]
        assert kwargs["dirs"] == ["R"]
        assert kwargs["cls"] == class_uri("Quadrupole")
        # A 1-based page becomes the row offset the index takes.
        assert kwargs["skip"] == 2 * SEARCH_PAGE_SIZE
        assert kwargs["page_size"] == SEARCH_PAGE_SIZE
        # The cap itself, not one over it: the index asks for the extra entry
        # that tells a full list from a clipped one.
        assert kwargs["facet_cap"] == SEARCH_FACET_CAP

    def test_an_unfiltered_search_sends_empty_lists_and_no_class(self, client):
        install_graph_paradigm(client, demo_context())
        calls = _spy_search(client.app.state.graph_index)

        assert client.get("/api/graph/search").status_code == 200

        kwargs = calls[0]
        assert kwargs["tokens"] == []
        assert kwargs["sections"] == []
        assert kwargs["systems"] == []
        assert kwargs["signals"] == []
        assert kwargs["dirs"] == []
        assert kwargs["cls"] is None
        assert kwargs["skip"] == 0

    def test_a_blank_class_is_sent_as_no_filter_rather_than_an_empty_string(self, client):
        install_graph_paradigm(client, demo_context())
        calls = _spy_search(client.app.state.graph_index)

        assert client.get("/api/graph/search", params={"cls": ""}).status_code == 200

        # '' would match no class at all; the no-filter value is null.
        assert calls[0]["cls"] is None

    def test_the_page_carries_the_corpus_rows_and_the_paging_around_them(self, client):
        install_graph_paradigm(client, demo_context())

        data = client.get("/api/graph/search").json()

        first_page = demo_page(1)
        assert data["total"] == len(DEMO_ROWS)
        assert data["devices"] == len(DEMO_PAGE_DEVICES)
        assert data["page"] == 1
        assert data["page_size"] == SEARCH_PAGE_SIZE
        # The corpus is longer than one page, so the finder really paginates.
        assert data["pages"] == DEMO_PAGE_COUNT > 1
        assert data["truncated"] is False
        assert data["empty"] is False
        assert data["suggestions"] == []
        assert len(data["rows"]) == len(first_page) == SEARCH_PAGE_SIZE
        # Every row comes back exactly as the corpus described it.
        by_pv = {row["fullPv"]: row for row in first_page}
        assert [row["fullPv"] for row in data["rows"]] == [row["fullPv"] for row in first_page]
        assert all(row == by_pv[row["fullPv"]] for row in data["rows"])

    def test_facets_arrive_in_the_order_the_rail_draws_them(self, client):
        install_graph_paradigm(client, demo_context())

        data = client.get("/api/graph/search").json()

        assert list(data["facets"]) == ["section", "system", "class", "signal", "dir"]
        assert all(
            set(entry) == {"value", "count"}
            for entries in data["facets"].values()
            for entry in entries
        )

    def test_a_page_past_the_matches_is_empty_but_still_paged(self, client):
        install_graph_paradigm(client, demo_context())

        past_the_end = DEMO_PAGE_COUNT + 1

        data = client.get("/api/graph/search", params={"page": str(past_the_end)}).json()

        assert data["rows"] == []
        assert data["page"] == past_the_end
        assert data["pages"] == DEMO_PAGE_COUNT
        assert data["total"] == len(DEMO_ROWS)

    def test_filters_are_anded_across_facets(self, client):
        install_graph_paradigm(client, demo_context())

        data = client.get(
            "/api/graph/search",
            params=[
                ("q", "QFA Current"),
                ("section", "SR01C"),
                ("section", "SR02C"),
                ("system", "MG"),
                ("signal", "current"),
                ("dir", "R"),
                ("cls", class_uri("Quadrupole")),
            ],
        ).json()

        # Two sections ORed, everything else ANDed: every QFA readback in each.
        expected = [
            row
            for row in DEMO_ROWS
            if "QFA" in row["device"]
            and row["section"] in {"SR01C", "SR02C"}
            and "READSSIGNAL" in row["edges"]
        ]
        assert [row["fullPv"] for row in data["rows"]] == [row["fullPv"] for row in expected]
        assert data["total"] == len(expected)
        assert data["devices"] == len({row["device"] for row in expected})

    def test_direction_facet_counts_a_read_and_set_address_under_three_values(self, client):
        install_graph_paradigm(client, demo_context())

        data = client.get("/api/graph/search").json()

        rows = DEMO_ROWS
        counts = {entry["value"]: entry["count"] for entry in data["facets"]["dir"]}
        # The one address read and set through the same channel is counted
        # under R, under W and under RW; the one with no signal edge is 'none'.
        assert set(counts) == {"R", "W", "RW", "none"}
        assert counts["RW"] == 1
        assert counts["none"] == 1
        assert counts["R"] == sum(1 for row in rows if "READSSIGNAL" in row["edges"])
        assert counts["W"] == sum(1 for row in rows if "WRITESSIGNAL" in row["edges"])

    def test_class_facet_values_are_uris_the_class_filter_can_be_sent_back(self, client):
        install_graph_paradigm(client, demo_context())

        data = client.get("/api/graph/search").json()

        values = [entry["value"] for entry in data["facets"]["class"]]
        assert all(value.startswith("https://") for value in values)
        # An abstract branch is in the list, carrying the devices under it.
        assert class_uri("Magnet") in values

    def test_a_facet_the_index_clipped_is_reported_and_cut_to_the_cap(self, client, monkeypatch):
        # The demo page has five signals; a cap of two makes the index clip
        # that list, the way a real corpus with hundreds of values would.
        monkeypatch.setattr(database_api, "_GRAPH_EXPLORE_MAX_ROWS", 2)
        install_graph_paradigm(client, demo_context())

        data = client.get("/api/graph/search").json()

        assert data["truncated"] is True
        assert len(data["facets"]["signal"]) == 2
        assert all(len(entries) <= 2 for entries in data["facets"].values())

    @pytest.mark.parametrize(
        "params",
        [
            {"page": "0"},
            {"page": "-1"},
            {"dir": "X"},
            {"dir": "r"},
        ],
    )
    def test_a_request_outside_the_contract_is_refused_before_the_index_is_read(
        self, client, params
    ):
        ctx = demo_context()
        install_graph_paradigm(client, ctx)
        calls = _spy_search(client.app.state.graph_index)

        resp = client.get("/api/graph/search", params=params)

        assert resp.status_code == 422
        assert calls == []
        assert ctx.calls == []

    def test_the_read_never_touches_the_store(self, client):
        ctx = demo_context()
        install_graph_paradigm(client, ctx)

        assert client.get("/api/graph/search").status_code == 200

        assert ctx.calls == []
        assert ctx.empty_checks == 0

    def test_file_backed_paradigm_gets_404(self, client):
        # The fixture app serves in_context; the finder is not its route.
        resp = client.get("/api/graph/search")

        assert resp.status_code == 404
        assert resp.json()["detail"] == "Not available for this pipeline type"


class TestDevice:
    """One device's card: what the store is asked, and what a miss looks like."""

    def test_the_uri_reaches_the_store_and_the_card_carries_the_device(self, client):
        ctx = demo_context()
        install_graph_paradigm(client, ctx)
        uri = DEMO_DEVICE_ROW["uri"]

        resp = client.get("/api/graph/device", params={"uri": uri})

        assert resp.status_code == 200
        assert ctx.last_params(GRAPH_DEVICE_CYPHER) == {"uri": uri}
        # The device read answers in a single row, and is bounded to it.
        assert ctx.reads_of(GRAPH_DEVICE_CYPHER)[-1][1] == 1
        data = resp.json()
        assert data["uri"] == uri
        assert data["device"] == DEMO_DEVICE_ROW["device"]
        # The card shows the class by name and keeps the URI to filter by.
        assert data["class"] == "Quadrupole"
        assert data["class_uri"] == class_uri("Quadrupole")
        assert data["rawType"] == "QFA"
        assert data["section"] == "SR01C"
        assert data["system"] == "MG"
        assert data["sPositionM"] == DEMO_DEVICE_ROW["sPositionM"]
        assert data["ordinalInSection"] == 1
        assert data["systemDescription"] == DEMO_DEVICE_ROW["systemDescription"]
        assert data["familyDescription"] == DEMO_DEVICE_ROW["familyDescription"]
        assert data["ringDescription"] == DEMO_DEVICE_ROW["ringDescription"]

    def test_bindings_stay_grouped_under_their_signal_and_carry_their_edges(self, client):
        install_graph_paradigm(client, demo_context())

        data = client.get("/api/graph/device", params={"uri": DEMO_DEVICE_ROW["uri"]}).json()

        assert [group["name"] for group in data["signals"]] == ["current"]
        assert data["signals"][0]["uri"] == signal_uri("current")
        bindings = data["signals"][0]["bindings"]
        # The readback and the setpoint are two addresses of one signal.
        assert [binding["edges"] for binding in bindings] == [["READSSIGNAL"], ["WRITESSIGNAL"]]
        assert [binding["fullPv"] for binding in bindings] == [
            f"{DEMO_DEVICE_ROW['device']}AM00",
            f"{DEMO_DEVICE_ROW['device']}SP00",
        ]
        assert bindings[0]["description"] == "current readback"
        assert bindings[0]["fieldDescription"] == "Current"
        assert bindings[0]["subfieldDescription"] == "Readback"
        assert set(bindings[0]) == {
            "fullPv",
            "edges",
            "description",
            "subfieldDescription",
            "fieldDescription",
        }

    def test_a_binding_without_a_semantic_signal_reports_no_edges(self, client):
        device = DEMO_DEVICE_ROW["device"]
        row = {
            **DEMO_DEVICE_ROW,
            "signals": [
                {
                    "uri": signal_uri("cavityVoltage"),
                    "name": "cavityVoltage",
                    "bindings": [
                        {
                            "fullPv": f"{device}AM00",
                            "description": "read and set on one address",
                            "fieldDescription": "Voltage",
                            "subfieldDescription": None,
                            "edges": ["READSSIGNAL", "WRITESSIGNAL"],
                        },
                        {
                            "fullPv": f"{device}ST00",
                            "description": "status word, no semantic signal",
                            "fieldDescription": None,
                            "subfieldDescription": None,
                            "edges": [],
                        },
                    ],
                }
            ],
        }
        install_graph_paradigm(client, demo_context(device_result=row))

        data = client.get("/api/graph/device", params={"uri": DEMO_DEVICE_ROW["uri"]}).json()

        assert [binding["edges"] for binding in data["signals"][0]["bindings"]] == [
            ["READSSIGNAL", "WRITESSIGNAL"],
            [],
        ]

    def test_an_untyped_device_reports_no_class_rather_than_failing(self, client):
        row = {**DEMO_DEVICE_ROW, "class": None, "classes": []}
        install_graph_paradigm(client, demo_context(device_result=row))

        resp = client.get("/api/graph/device", params={"uri": DEMO_DEVICE_ROW["uri"]})

        assert resp.status_code == 200
        assert resp.json()["class"] is None
        assert resp.json()["class_uri"] is None

    def test_a_uri_the_store_does_not_hold_answers_404_the_panel_can_branch_on(self, client):
        ctx = demo_context(device_result=None)
        install_graph_paradigm(client, ctx)
        missing = device_uri("SR09C___NOPE___")

        resp = client.get("/api/graph/device", params={"uri": missing})

        assert resp.status_code == 404
        body = resp.json()
        # Not FastAPI's bare {"detail": ...}: the panel branches on error_type
        # and shows the suggestion, the same three keys every graph answer has.
        assert missing in body["detail"]
        assert body["error_type"] == "not_found"
        assert body["suggestions"]
        # A device the store does not hold says nothing about the corpus.
        assert ctx.empty_checks == 0

    @pytest.mark.parametrize("params", [{}, {"uri": ""}])
    def test_a_missing_or_empty_uri_is_refused_before_the_store_is_read(self, client, params):
        ctx = demo_context()
        install_graph_paradigm(client, ctx)

        resp = client.get("/api/graph/device", params=params)

        assert resp.status_code == 422
        assert ctx.calls == []

    def test_the_device_read_runs_off_the_event_loop(self, client):
        ctx = demo_context()
        install_graph_paradigm(client, ctx)
        loop_thread_id = _loop_thread_id(client)

        resp = client.get("/api/graph/device", params={"uri": DEMO_DEVICE_ROW["uri"]})

        assert resp.status_code == 200
        assert ctx.saw_running_loop == [False]
        assert ctx.thread_ids[0] != loop_thread_id

    def test_an_unreachable_store_answers_503_carrying_its_own_remedy(self, client):
        install_graph_paradigm(
            client,
            FakeGraphContext(
                raises=GraphUnreachable(
                    "Graph store at bolt://localhost:7687 is unreachable.",
                    ["Start the graphdb service."],
                )
            ),
        )

        resp = client.get("/api/graph/device", params={"uri": DEMO_DEVICE_ROW["uri"]})

        assert resp.status_code == 503
        body = resp.json()
        assert body["error_type"] == "service_unavailable"
        assert body["suggestions"] == ["Start the graphdb service."]

    def test_missing_graph_context_returns_503_with_a_configuration_remedy(self, client):
        install_graph_paradigm(client, None)

        resp = client.get("/api/graph/device", params={"uri": DEMO_DEVICE_ROW["uri"]})

        assert resp.status_code == 503
        body = resp.json()
        assert body["error_type"] == "service_unavailable"
        assert "graphdb" in " ".join(body["suggestions"])

    def test_file_backed_paradigm_gets_404(self, client):
        resp = client.get("/api/graph/device", params={"uri": DEMO_DEVICE_ROW["uri"]})

        assert resp.status_code == 404
        assert resp.json()["detail"] == "Not available for this pipeline type"
