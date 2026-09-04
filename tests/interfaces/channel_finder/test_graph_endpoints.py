"""Tests for the Channel Finder graph-paradigm REST endpoints.

The graph routes read a store instead of a database file, which puts three
things under test that the file-backed routes never face: the reads must not
run on the event loop, an unreachable store must answer with the remedy the
store itself supplies, and an empty store must be told apart from a broken one.

The store is faked rather than dialled. The corpus, the fake and the numbers
they imply live in ``graph_fixture`` — one copy shared with the launcher and
the browser lanes — so a number asserted here means the same thing there. The
fake is keyed by query text, so one installed context answers both reads a
single request makes with different rows, and it records the thread each call
arrives on so the off-loop contract can be asserted rather than assumed.
"""

from __future__ import annotations

import threading

import pytest

from osprey.mcp_server.graph.server_context import GraphUnreachable
from osprey.services.channel_finder.graph_queries import (
    GRAPH_DEVICE_CYPHER,
    GRAPH_SEARCH_CYPHER,
)
from tests.interfaces.channel_finder.graph_fixture import (
    DEMO_DEVICE_ROW,
    DEMO_SEARCH_FACET_OVERFLOW,
    DEMO_SEARCH_ROW,
    SEARCH_FACET_CAP,
    SEARCH_PAGE_SIZE,
    SEED_COMMAND,
    FakeGraphContext,
    class_row,
    class_uri,
    demo_context,
    device_uri,
    install_graph_paradigm,
    signal_uri,
)


def _loop_thread_id(client) -> int:
    """Return the thread the app's event loop runs on, measured from inside it.

    The loop runs on a thread of the test client's own choosing, so the thread
    a store read must not land on is captured from a coroutine the same client
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


class TestOntologyPayload:
    """What the endpoint draws from a store that answers normally."""

    def test_demo_corpus_yields_the_device_taxonomy(self, client):
        ctx = demo_context()
        install_graph_paradigm(client, ctx)

        resp = client.get("/api/graph/ontology")

        assert resp.status_code == 200
        data = resp.json()
        # Two of the 21 stored classes are about signals and bindings; the
        # taxonomy an operator came for is the other 19.
        assert len(data["classes"]) == 19
        names = {entry["name"] for entry in data["classes"]}
        assert "SemanticSignal" not in names
        assert "ChannelBinding" not in names
        assert "Quadrupole" in names
        assert data["empty"] is False
        assert data["truncated"] is False
        assert data["suggestions"] == []

    def test_root_carries_the_whole_device_population(self, client):
        install_graph_paradigm(client, demo_context())

        data = client.get("/api/graph/ontology").json()

        by_name = {entry["name"]: entry for entry in data["classes"]}
        assert by_name["AcceleratorDevice"]["rollup"] == 512
        assert by_name["AcceleratorDevice"]["parents"] == []
        # An abstract branch still carries the devices under its subclasses.
        assert by_name["Magnet"]["rollup"] == 382
        assert by_name["Magnet"]["parents"] == [class_uri("AcceleratorDevice")]
        assert by_name["Dipole"]["rollup"] == 44
        assert by_name["Corrector"]["rollup"] == 156
        assert by_name["BeamPositionMonitor"]["altLabel"] == ["BPM"]

    def test_a_branch_is_marked_abstract_and_a_leaf_is_not(self, client):
        install_graph_paradigm(client, demo_context())

        data = client.get("/api/graph/ontology").json()

        by_name = {entry["name"]: entry for entry in data["classes"]}
        # Magnet groups 382 devices without being a kind of device itself.
        assert by_name["Magnet"]["abstract"] is True
        assert by_name["Corrector"]["abstract"] is True
        assert by_name["Quadrupole"]["abstract"] is False
        assert by_name["Undulator"]["abstract"] is False

    def test_relationship_vocabulary_is_returned_flat_and_unfiltered(self, client):
        install_graph_paradigm(client, demo_context())

        data = client.get("/api/graph/ontology").json()

        assert data["relationship_types"] == [
            "HASBINDING",
            "READSSIGNAL",
            "SUBCLASSOF",
            "TYPE",
            "WRITESSIGNAL",
        ]

    def test_truncation_of_either_read_is_reported(self, client):
        install_graph_paradigm(client, demo_context(relationship_truncated=True))

        data = client.get("/api/graph/ontology").json()

        assert data["truncated"] is True

    def test_both_reads_are_bounded_by_an_explicit_row_cap(self, client):
        ctx = demo_context()
        install_graph_paradigm(client, ctx)

        client.get("/api/graph/ontology")

        assert len(ctx.calls) == 2
        assert all(max_rows == 500 for _, max_rows in ctx.calls)
        # The common path does not pay for an emptiness probe.
        assert ctx.empty_checks == 0


class TestAwkwardOntologies:
    """Rows the store can legitimately hold that must not derail the endpoint."""

    def test_a_class_with_two_parents_appears_once_carrying_both(self, client):
        rows = [
            class_row("AcceleratorDevice", 20),
            class_row("Magnet", 20, ["AcceleratorDevice"]),
            class_row("SteeringDevice", 12, ["AcceleratorDevice"]),
            # Multiple inheritance: a corrector is both a magnet and a steerer.
            class_row("Corrector", 12, ["Magnet", "SteeringDevice"]),
        ]
        install_graph_paradigm(client, FakeGraphContext(class_rows=rows))

        resp = client.get("/api/graph/ontology")

        assert resp.status_code == 200
        corrector = [c for c in resp.json()["classes"] if c["name"] == "Corrector"]
        assert len(corrector) == 1
        assert corrector[0]["parents"] == [class_uri("Magnet"), class_uri("SteeringDevice")]

    def test_a_subclass_cycle_is_answered_rather_than_hung(self, client):
        # A corpus whose SUBCLASSOF edges close a loop is malformed, but it is
        # the store's data — the endpoint answers it instead of spinning or 500ing.
        rows = [
            class_row("Alpha", 0, ["Beta"]),
            class_row("Beta", 0, ["Alpha"]),
        ]
        install_graph_paradigm(client, FakeGraphContext(class_rows=rows))

        resp = client.get("/api/graph/ontology")

        assert resp.status_code == 200
        data = resp.json()
        assert {entry["name"] for entry in data["classes"]} == {"Alpha", "Beta"}
        assert data["empty"] is False


class TestEmptyStore:
    """A store that is up but unseeded is a different answer from one that is down."""

    def test_empty_store_answers_200_naming_the_seed_command(self, client):
        ctx = FakeGraphContext(class_rows=[], relationship_rows=[], empty=True)
        install_graph_paradigm(client, ctx)

        resp = client.get("/api/graph/ontology")

        assert resp.status_code == 200
        data = resp.json()
        assert data["empty"] is True
        assert data["classes"] == []
        assert data["relationship_types"] == []
        assert data["truncated"] is False
        assert any(SEED_COMMAND in hint for hint in data["suggestions"])
        assert ctx.empty_checks == 1

    def test_a_store_that_fails_the_emptiness_probe_answers_503(self, client):
        ctx = FakeGraphContext(
            class_rows=[],
            empty_raises=GraphUnreachable("Graph store is unreachable.", ["Start the store."]),
        )
        install_graph_paradigm(client, ctx)

        resp = client.get("/api/graph/ontology")

        assert resp.status_code == 503
        assert resp.json()["error_type"] == "service_unavailable"


class TestStoreFailures:
    """Every way the store fails to serve is a 503 carrying its own remedy."""

    def test_unreachable_store_returns_the_error_payload_at_the_top_level(self, client):
        install_graph_paradigm(
            client,
            FakeGraphContext(
                raises=GraphUnreachable(
                    "Graph store at bolt://localhost:7687 is unreachable.",
                    ["Start the graphdb service."],
                )
            ),
        )

        resp = client.get("/api/graph/ontology")

        assert resp.status_code == 503
        body = resp.json()
        # Not nested under FastAPI's "detail" envelope: the web UI reads all
        # three keys off the body it is handed.
        assert "unreachable" in body["detail"]
        assert body["error_type"] == "service_unavailable"
        assert body["suggestions"] == ["Start the graphdb service."]

    def test_missing_graph_context_returns_503_with_a_configuration_remedy(self, client):
        install_graph_paradigm(client, None)

        resp = client.get("/api/graph/ontology")

        assert resp.status_code == 503
        body = resp.json()
        assert body["error_type"] == "service_unavailable"
        assert "graphdb" in " ".join(body["suggestions"])


class TestOffLoopReads:
    """The store's driver is synchronous; the event loop must never wait on it."""

    def test_store_calls_run_off_the_event_loop(self, client):
        ctx = demo_context(class_rows=[], relationship_rows=[], empty=True)
        install_graph_paradigm(client, ctx)

        # The loop runs on a thread of the test client's own choosing, so the
        # thread to compare against is captured from inside a coroutine the
        # same client drives rather than assumed to be this one.
        loop_thread_ids: list[int] = []

        @client.app.get("/api/_loop_thread_probe")
        async def _loop_thread_probe():  # pragma: no cover - trivial probe
            loop_thread_ids.append(threading.get_ident())
            return {"ok": True}

        assert client.get("/api/_loop_thread_probe").status_code == 200
        assert client.get("/api/graph/ontology").status_code == 200

        # Both reads plus the emptiness probe: three store calls, none of them
        # on the loop thread and none inside a running loop.
        assert len(ctx.thread_ids) == 3
        assert ctx.saw_running_loop == [False, False, False]
        assert loop_thread_ids, "probe route never ran"
        assert all(tid != loop_thread_ids[0] for tid in ctx.thread_ids)


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

    def test_query_and_facets_reach_the_store_as_the_cypher_expects(self, client):
        ctx = demo_context()
        install_graph_paradigm(client, ctx)

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
        params = ctx.last_params(GRAPH_SEARCH_CYPHER)
        # The route lower-cases its tokens because the Cypher does not.
        assert params["tokens"] == ["qfa", "current"]
        # A facet given twice arrives as a list, not as the last value.
        assert params["sections"] == ["SR01C", "SR02C"]
        assert params["systems"] == ["MG"]
        assert params["signals"] == ["current"]
        assert params["dirs"] == ["R"]
        assert params["cls"] == class_uri("Quadrupole")
        assert params["skip"] == 2 * SEARCH_PAGE_SIZE
        # One over the cap: a facet list that comes back full is the only way
        # the store can say it had more values than the explorer asked for.
        assert params["facet_cap"] == SEARCH_FACET_CAP + 1
        # The search answers in a single row, and the read is bounded to it.
        assert ctx.reads_of(GRAPH_SEARCH_CYPHER)[-1][1] == 1

    def test_an_unfiltered_search_sends_empty_lists_and_no_class(self, client):
        ctx = demo_context()
        install_graph_paradigm(client, ctx)

        assert client.get("/api/graph/search").status_code == 200

        params = ctx.last_params(GRAPH_SEARCH_CYPHER)
        # Every list parameter travels as a real list — the Cypher has no
        # defaults and calls size() on all of them.
        assert params["tokens"] == []
        assert params["sections"] == []
        assert params["systems"] == []
        assert params["signals"] == []
        assert params["dirs"] == []
        assert params["cls"] is None
        assert params["skip"] == 0

    def test_a_blank_class_is_sent_as_no_filter_rather_than_an_empty_string(self, client):
        ctx = demo_context()
        install_graph_paradigm(client, ctx)

        assert client.get("/api/graph/search", params={"cls": ""}).status_code == 200

        # '' would match no class at all; the no-filter value is null.
        assert ctx.last_params(GRAPH_SEARCH_CYPHER)["cls"] is None

    def test_the_page_carries_the_store_row_and_the_paging_around_it(self, client):
        install_graph_paradigm(client, demo_context())

        data = client.get("/api/graph/search").json()

        assert data["total"] == DEMO_SEARCH_ROW["total"]
        assert data["devices"] == DEMO_SEARCH_ROW["devices"]
        assert data["page"] == 1
        assert data["page_size"] == SEARCH_PAGE_SIZE
        # 128 matches over fifty-row pages is three pages, the last one short.
        assert data["pages"] == 3
        assert data["truncated"] is False
        assert data["empty"] is False
        assert data["suggestions"] == []
        assert len(data["rows"]) == SEARCH_PAGE_SIZE
        assert data["rows"][0] == DEMO_SEARCH_ROW["rows"][0]
        assert set(data["facets"]) == {"section", "system", "class", "signal", "dir"}

    def test_direction_facet_counts_a_read_and_set_address_under_three_values(self, client):
        install_graph_paradigm(client, demo_context())

        data = client.get("/api/graph/search").json()

        rows = DEMO_SEARCH_ROW["rows"]
        counts = {entry["value"]: entry["count"] for entry in data["facets"]["dir"]}
        # The one address read and set through the same channel is counted
        # under R, under W and under RW; the one with no signal edge is 'none'.
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

    def test_a_facet_the_store_clipped_is_reported_and_cut_to_the_cap(self, client):
        overflowing = {**DEMO_SEARCH_ROW, "facets": DEMO_SEARCH_FACET_OVERFLOW}
        install_graph_paradigm(client, demo_context(search_result=overflowing))

        data = client.get("/api/graph/search").json()

        assert data["truncated"] is True
        assert len(data["facets"]["signal"]) == SEARCH_FACET_CAP
        # Only the clipped facet is cut; the others arrive whole.
        assert len(data["facets"]["section"]) == len(DEMO_SEARCH_ROW["facets"]["section"])

    @pytest.mark.parametrize(
        "params",
        [
            {"page": "0"},
            {"page": "-1"},
            {"dir": "X"},
            {"dir": "r"},
        ],
    )
    def test_a_request_outside_the_contract_is_refused_before_the_store_is_read(
        self, client, params
    ):
        ctx = demo_context()
        install_graph_paradigm(client, ctx)

        resp = client.get("/api/graph/search", params=params)

        assert resp.status_code == 422
        assert ctx.calls == []

    def test_an_unfiltered_empty_result_reports_the_unseeded_store(self, client):
        ctx = demo_context(search_result=None, empty=True)
        install_graph_paradigm(client, ctx)

        resp = client.get("/api/graph/search")

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["devices"] == 0
        assert data["rows"] == []
        assert data["pages"] == 0
        assert data["empty"] is True
        assert any(SEED_COMMAND in hint for hint in data["suggestions"])
        assert ctx.empty_checks == 1

    def test_an_unfiltered_empty_result_from_a_seeded_store_is_not_a_remedy(self, client):
        ctx = demo_context(search_result=None, empty=False)
        install_graph_paradigm(client, ctx)

        data = client.get("/api/graph/search").json()

        # The store was asked and said it holds a corpus, so nothing here is a
        # seeding gap — it is a corpus that happens to match nothing.
        assert data["empty"] is False
        assert data["suggestions"] == []
        assert ctx.empty_checks == 1

    @pytest.mark.parametrize(
        "params",
        [
            {"q": "nosuchthing"},
            {"section": "SR09C"},
            {"cls": class_uri("Undulator")},
        ],
    )
    def test_a_filtered_empty_result_never_pays_for_the_emptiness_probe(self, client, params):
        ctx = demo_context(search_result=None, empty=True)
        install_graph_paradigm(client, ctx)

        data = client.get("/api/graph/search", params=params).json()

        # A search that filtered and matched nothing says so plainly: the
        # store is not unseeded just because this question had no answer.
        assert data["total"] == 0
        assert data["empty"] is False
        assert data["suggestions"] == []
        assert ctx.empty_checks == 0

    def test_the_search_read_runs_off_the_event_loop(self, client):
        ctx = demo_context()
        install_graph_paradigm(client, ctx)
        loop_thread_id = _loop_thread_id(client)

        assert client.get("/api/graph/search").status_code == 200

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

        resp = client.get("/api/graph/search")

        assert resp.status_code == 503
        body = resp.json()
        assert "unreachable" in body["detail"]
        assert body["error_type"] == "service_unavailable"
        assert body["suggestions"] == ["Start the graphdb service."]

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

    def test_file_backed_paradigm_gets_404(self, client):
        resp = client.get("/api/graph/device", params={"uri": DEMO_DEVICE_ROW["uri"]})

        assert resp.status_code == 404
        assert resp.json()["detail"] == "Not available for this pipeline type"
