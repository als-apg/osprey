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
from tests.interfaces.channel_finder.graph_fixture import (
    SEED_COMMAND,
    FakeGraphContext,
    class_row,
    class_uri,
    demo_context,
    install_graph_paradigm,
)


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
