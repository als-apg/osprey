"""The fake store answers what the routes ask, and the launcher serves it.

Two contracts meet in this file. The fake store in ``graph_fixture`` is what
every graph-paradigm test reads through, so its dispatch is asserted directly:
each query the routes send has an answer of its own, and a query it has never
been taught fails loudly rather than being served another read's rows. The
launcher is asserted the same way — the browser and visual lanes reach the
graph UI through ``launch_graph_channel_finder``, and when it is wrong they
fail as a photograph that does not match or a page that renders the
unconfigured shell, symptoms several layers away from the cause.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from osprey.deployment.graphdb_service import GRAPHDB_BUILD_INDEX_COMMAND
from osprey.interfaces.channel_finder.app import _open_graph_index
from osprey.interfaces.channel_finder.database_api import (
    UNRESOLVED_INDEX_PATH_REMEDY,
    _serve_index_read,
)
from osprey.services.channel_finder.graph_index.reader import GraphIndexAbsence
from osprey.services.channel_finder.graph_queries import GRAPH_DEVICE_CYPHER
from tests.interfaces.channel_finder.graph_fixture import (
    AGENT_FACET_CAP,
    DEMO_CLASS_COUNT,
    DEMO_DEVICE_CLASSES,
    DEMO_DEVICE_ROW,
    DEMO_DEVICES,
    DEMO_FACETS,
    DEMO_FAMILIES,
    DEMO_PAGE_COUNT,
    DEMO_ROWS,
    DEMO_SECTIONS,
    DEMO_STATISTICS,
    ORDINALS_PER_FAMILY,
    SEARCH_FACET_CAP,
    SEARCH_PAGE_SIZE,
    FakeGraphContext,
    build_demo_index,
    class_uri,
    demo_context,
    demo_facets,
    demo_index_path,
    demo_page,
    device_card,
    device_uri,
    install_graph_paradigm,
    open_demo_index,
    synth_rows,
)
from tests.interfaces.conftest import launch_graph_channel_finder


class TestGraphLauncher:
    """What a lane gets when it boots the Channel Finder through the launcher."""

    def test_served_app_reports_the_graph_paradigm(self, tmp_path, monkeypatch):
        with launch_graph_channel_finder(tmp_path, monkeypatch) as base_url:
            info = httpx.get(f"{base_url}/api/info", timeout=10.0)

        assert info.status_code == 200
        payload = info.json()
        assert payload["pipeline_type"] == "graph"
        # Store-backed rather than file-backed: the paradigm is the answer, and
        # there is no database path to name.
        assert payload["graph_backed"] is True
        assert payload["db_path"] is None

    def test_ontology_route_answers_from_the_demo_corpus(self, tmp_path, monkeypatch):
        with launch_graph_channel_finder(tmp_path, monkeypatch) as base_url:
            resp = httpx.get(f"{base_url}/api/graph/ontology", timeout=10.0)

        assert resp.status_code == 200
        data = resp.json()
        # The launcher wrote the index path into the config: the built
        # taxonomy comes back whole rather than the unavailable answer.
        with open_demo_index() as index:
            assert len(data["classes"]) == index.meta.class_count
        assert data["relationship_types"] == []
        assert data["empty"] is False


class TestFakeStoreDispatch:
    """Every query the routes send has an answer of its own, and only its own."""

    def test_the_device_read_gets_the_demo_device(self):
        assert demo_context().run_read(GRAPH_DEVICE_CYPHER).rows == [DEMO_DEVICE_ROW]

    def test_a_query_the_fake_does_not_know_fails_loudly(self):
        """A read nobody taught it is a drifted test, not a fallthrough."""
        with pytest.raises(AssertionError, match=r"unkeyed cypher: MATCH \(x:Nowhere\)"):
            demo_context().run_read("MATCH (x:Nowhere) RETURN x")

    def test_every_read_records_its_parameters_and_row_cap(self):
        """A route's request contract is assertable from the store side."""
        ctx = demo_context()

        ctx.run_read(GRAPH_DEVICE_CYPHER, {"uri": "urn:first"}, max_rows=1)
        ctx.run_read(GRAPH_DEVICE_CYPHER, {"uri": DEMO_DEVICE_ROW["uri"]}, max_rows=1)

        assert ctx.reads_of(GRAPH_DEVICE_CYPHER) == [
            ({"uri": "urn:first"}, 1),
            ({"uri": DEMO_DEVICE_ROW["uri"]}, 1),
        ]
        assert ctx.last_params(GRAPH_DEVICE_CYPHER) == {"uri": DEMO_DEVICE_ROW["uri"]}
        # The older recording the endpoint tests read stays exactly as it was.
        assert [cypher for cypher, _ in ctx.calls].count(GRAPH_DEVICE_CYPHER) == 2

    def test_asking_for_parameters_of_a_read_that_never_happened_fails(self):
        with pytest.raises(AssertionError, match="no read sent"):
            demo_context().last_params(GRAPH_DEVICE_CYPHER)


class TestSynthesizedCorpus:
    """The grid the corpus is laid out on, and the census it implies."""

    def test_the_grid_is_a_section_family_ordinal_product(self):
        rows = synth_rows()

        assert rows == DEMO_ROWS, "the corpus is not stable between calls"
        assert rows == sorted(rows, key=lambda row: row["fullPv"])
        # Every family has the same population in every section; the only
        # devices beyond the grid are the one-off cavity.
        assert (
            len(DEMO_DEVICES) == len(DEMO_SECTIONS) * len(DEMO_FAMILIES) * ORDINALS_PER_FAMILY + 1
        )
        for section in DEMO_SECTIONS:
            for family in DEMO_FAMILIES:
                here = {
                    spec.name
                    for spec in DEMO_DEVICES.values()
                    if spec.section == section and spec.family is family
                }
                assert len(here) == ORDINALS_PER_FAMILY, (section, family.stem)

    def test_every_row_carries_the_columns_the_search_projects(self):
        for row in DEMO_ROWS:
            assert set(row) == {
                "fullPv",
                "description",
                "device",
                "device_uri",
                "section",
                "system",
                "edges",
                "signals",
            }
            assert row["fullPv"].startswith(row["device"])
            assert all(set(signal) == {"uri", "name"} for signal in row["signals"])

    def test_a_settable_family_carries_a_setpoint_beside_its_readback(self):
        writes: dict[str, list[str]] = {}
        for row in DEMO_ROWS:
            if "WRITESSIGNAL" in row["edges"]:
                writes.setdefault(row["device"], []).append(row["fullPv"])

        for name, spec in DEMO_DEVICES.items():
            if spec.family not in DEMO_FAMILIES:
                continue  # the off-grid cavity, which is read and set at once
            if spec.family.settable:
                assert writes[name] == [f"{name}SP00"], name
            else:
                assert name not in writes, name

    def test_exactly_one_channel_both_reads_and_writes(self):
        both = [row for row in DEMO_ROWS if {"READSSIGNAL", "WRITESSIGNAL"} <= set(row["edges"])]

        assert len(both) == 1
        assert both[0]["edges"] == ["READSSIGNAL", "WRITESSIGNAL"]

    def test_exactly_one_channel_carries_no_signal_at_all(self):
        bare = [row for row in DEMO_ROWS if not row["edges"]]

        assert len(bare) == 1
        assert bare[0]["signals"] == []

    def test_the_corpus_spans_more_than_one_page(self):
        """A finder that never paginates in its own fixture proves nothing."""
        assert DEMO_PAGE_COUNT > 1
        assert len(demo_page(1)) == SEARCH_PAGE_SIZE
        assert demo_page(DEMO_PAGE_COUNT + 1) == []
        # The pages partition the corpus, in order and without a gap.
        paged = [row for page in range(1, DEMO_PAGE_COUNT + 1) for row in demo_page(page)]
        assert paged == DEMO_ROWS

    def test_the_census_is_read_off_the_rows(self):
        assert DEMO_STATISTICS == {
            "devices": len({row["device"] for row in DEMO_ROWS}),
            "channels": len(DEMO_ROWS),
            "classes": DEMO_CLASS_COUNT,
            "signals": len({entry["name"] for row in DEMO_ROWS for entry in row["signals"]}),
            "sections": len(DEMO_SECTIONS),
        }

    def test_the_read_write_channel_is_counted_under_all_three_directions(self):
        counts = {entry["value"]: entry["count"] for entry in DEMO_FACETS["dir"]}
        reads = sum(1 for row in DEMO_ROWS if "READSSIGNAL" in row["edges"])
        writes = sum(1 for row in DEMO_ROWS if "WRITESSIGNAL" in row["edges"])

        assert counts["RW"] == 1
        assert counts["none"] == 1
        assert counts["R"] == reads
        assert counts["W"] == writes

    def test_the_class_facet_names_classes_by_uri_including_an_abstract_one(self):
        entries = {entry["value"]: entry["count"] for entry in DEMO_FACETS["class"]}

        assert all(value.startswith("https://") for value in entries)
        # Magnet is a branch nothing in the corpus is typed by directly, so its
        # count can only have come from rolling its subclasses up.
        assert entries[class_uri("Magnet")] > entries[class_uri("Quadrupole")]

    def test_one_rail_runs_past_what_the_agent_tool_will_show(self):
        """Facet truncation is exercisable from the same corpus as pagination."""
        assert len(DEMO_FACETS["class"]) > AGENT_FACET_CAP
        # Still far under the browser rail's own budget, so nothing the
        # explorer draws is clipped.
        assert all(0 < len(entries) <= SEARCH_FACET_CAP for entries in DEMO_FACETS.values())

    def test_every_facet_is_ordered_the_way_the_index_orders_it(self):
        for name, entries in DEMO_FACETS.items():
            assert entries == sorted(entries, key=lambda e: (-e["count"], e["value"])), name

    def test_facets_over_a_subset_count_only_that_subset(self):
        one_section = [row for row in DEMO_ROWS if row["section"] == DEMO_SECTIONS[0]]

        facets = demo_facets(one_section)

        assert [entry["value"] for entry in facets["section"]] == [DEMO_SECTIONS[0]]
        assert facets["section"][0]["count"] == len(one_section)


class TestDemoDeviceRow:
    """The device read answers one device, or nothing at all."""

    def test_the_default_device_is_shaped_as_the_device_query_returns_it(self):
        row = demo_context().run_read(GRAPH_DEVICE_CYPHER).rows[0]

        assert row == DEMO_DEVICE_ROW
        assert set(row) == {
            "uri",
            "device",
            "class",
            "classes",
            "rawType",
            "section",
            "system",
            "sPositionM",
            "ordinalInSection",
            "systemDescription",
            "familyDescription",
            "ringDescription",
            "signals",
        }
        group = row["signals"][0]
        assert set(group) == {"uri", "name", "bindings"}
        assert group["bindings"] == sorted(group["bindings"], key=lambda b: b["fullPv"])
        assert {b["edges"][0] for b in group["bindings"]} == {"READSSIGNAL", "WRITESSIGNAL"}

    def test_a_card_addresses_exactly_the_channels_the_corpus_binds(self):
        """Every device's card is built from the rows, so the two cannot drift."""
        for device in DEMO_DEVICES:
            card = device_card(device)
            addressed = [
                binding["fullPv"] for group in card["signals"] for binding in group["bindings"]
            ]
            bound = [row["fullPv"] for row in DEMO_ROWS if row["device"] == device and row["edges"]]

            assert sorted(addressed) == sorted(bound), device
            assert card["uri"] == device_uri(device)
            assert card["class"] == class_uri(DEMO_DEVICE_CLASSES[device])

    def test_a_card_matches_the_channels_the_built_index_holds(self):
        device = DEMO_DEVICE_ROW["device"]
        card = device_card(device)

        with open_demo_index() as index:
            rows = index.search(tokens=[device.lower()])["rows"]

        assert {row["fullPv"] for row in rows} == {
            binding["fullPv"] for group in card["signals"] for binding in group["bindings"]
        }
        assert {row["section"] for row in rows} == {card["section"]}
        assert {row["system"] for row in rows} == {card["system"]}

    def test_a_device_the_store_does_not_hold_answers_no_rows(self):
        """Absence is an empty result, which is how a route reads a miss."""
        ctx = demo_context(device_result=None)

        assert ctx.run_read(GRAPH_DEVICE_CYPHER).rows == []

    def test_a_test_can_hand_the_device_read_one_row_of_its_own(self):
        ctx = FakeGraphContext(device_result={"uri": "urn:x", "device": "X", "signals": []})

        assert ctx.run_read(GRAPH_DEVICE_CYPHER).rows == [
            {"uri": "urn:x", "device": "X", "signals": []}
        ]


class TestDemoSearchIndex:
    """The index the fixture builds holds the corpus this module describes."""

    def test_the_index_counts_the_page_it_was_built_from(self):
        with open_demo_index() as index:
            meta = index.meta

        assert meta.binding_count == len(DEMO_ROWS)
        assert meta.section_count == len(DEMO_SECTIONS)
        assert meta.device_count == len({row["device"] for row in DEMO_ROWS})
        assert meta.corpus_sha256

    def test_every_address_on_the_page_is_a_channel_in_the_index(self):
        """The roster the index answers is the page the fake store serves."""
        with open_demo_index() as index:
            addresses = {
                row[0] for row in index.cursor().execute("SELECT address FROM channels").fetchall()
            }

        assert addresses == {row["fullPv"] for row in DEMO_ROWS}

    def test_a_search_finds_the_device_the_device_read_answers(self):
        with open_demo_index() as index:
            page = index.search(tokens=["qfa"])

        assert page["total"] > 0
        assert all("QFA" in row["fullPv"] for row in page["rows"])

    def test_the_taxonomy_keeps_the_classes_the_page_uses(self):
        """Bound classes survive; the ontology's unbound branches are pruned."""
        with open_demo_index() as index:
            names = {entry["uri"] for entry in index.ontology()["classes"]}

        assert class_uri("Quadrupole") in names
        # A branch nothing on the page is typed by directly still earns its row
        # from the subclasses that roll up to it.
        assert class_uri("Magnet") in names
        # Neither of the two non-device classes is part of a device taxonomy.
        assert class_uri("SemanticSignal") not in names

    def test_the_file_is_built_once_and_reopened(self):
        assert demo_index_path() == demo_index_path()
        first, second = open_demo_index(), open_demo_index()
        try:
            assert first is not second
            assert first.path == second.path
        finally:
            first.close()
            second.close()

    def test_building_into_a_directory_writes_a_readable_index(self, tmp_path):
        path = build_demo_index(tmp_path)

        assert path.parent == tmp_path
        assert path.stat().st_size > 0


class TestGraphParadigmInstallsAnIndex:
    """``install_graph_paradigm`` stages the index the lifespan would have."""

    def test_the_app_gets_an_open_index_by_default(self, client):
        install_graph_paradigm(client, demo_context())

        index = client.app.state.graph_index
        assert not index.closed
        assert index.meta.binding_count == len(DEMO_ROWS)

    def test_an_absence_can_be_staged_instead(self, client, tmp_path):
        absence = GraphIndexAbsence("missing", tmp_path / "graph.duckdb", "No search index.")

        install_graph_paradigm(client, demo_context(), index=absence)

        assert client.app.state.graph_index is absence

    def test_passing_none_leaves_no_index_on_the_app(self, client):
        install_graph_paradigm(client, demo_context())
        install_graph_paradigm(client, demo_context(), index=None)

        assert not hasattr(client.app.state, "graph_index")


#: The app module's logger, which the absence lines are written to.
_APP_LOGGER = "osprey.interfaces.channel_finder.app"


class TestOpenGraphIndexAtStartup:
    """What the lifespan puts on ``app.state.graph_index``."""

    def test_a_configured_index_is_opened(self, tmp_path):
        index_path = build_demo_index(tmp_path)

        opened = _open_graph_index({"services": {"graphdb": {"index_path": str(index_path)}}})

        try:
            assert not isinstance(opened, GraphIndexAbsence)
            assert opened.path == index_path
        finally:
            opened.close()

    def test_an_index_that_was_never_built_is_an_absence(self, tmp_path):
        missing = tmp_path / "graph.duckdb"

        opened = _open_graph_index({"services": {"graphdb": {"index_path": str(missing)}}})

        assert isinstance(opened, GraphIndexAbsence)
        assert opened.reason == "missing"
        assert str(missing) in opened.detail

    def test_a_malformed_index_path_is_an_absence_rather_than_a_refusal(self):
        """A config typo must not stop the app from serving everything else."""
        opened = _open_graph_index({"services": {"graphdb": {"index_path": 42}}})

        assert isinstance(opened, GraphIndexAbsence)
        assert opened.reason == "unreadable"
        assert "Cannot resolve the search index's path" in opened.detail
        # The remedy travels in the sentence itself, so every surface that
        # shows the detail — 503 body, health row, log line — shows the fix.
        assert opened.detail.endswith(UNRESOLVED_INDEX_PATH_REMEDY)

    def test_an_unbuilt_index_is_logged_at_info(self, tmp_path, caplog):
        """Not built yet is an ordinary state, not a warning."""
        missing = tmp_path / "graph.duckdb"

        with caplog.at_level(logging.DEBUG, logger=_APP_LOGGER):
            _open_graph_index({"services": {"graphdb": {"index_path": str(missing)}}})

        records = [r for r in caplog.records if r.name == _APP_LOGGER]
        assert records, "the absence was not logged"
        assert {r.levelno for r in records} == {logging.INFO}

    def test_an_index_the_driver_refuses_is_logged_at_warning(self, tmp_path, caplog):
        """A file that is there but cannot be read is worth a warning."""
        garbage = tmp_path / "graph.duckdb"
        garbage.write_bytes(b"not a duckdb file")

        with caplog.at_level(logging.DEBUG, logger=_APP_LOGGER):
            opened = _open_graph_index({"services": {"graphdb": {"index_path": str(garbage)}}})

        assert isinstance(opened, GraphIndexAbsence)
        assert opened.reason == "unreadable"
        warnings = [
            r for r in caplog.records if r.name == _APP_LOGGER and r.levelno == logging.WARNING
        ]
        assert any(opened.detail in r.getMessage() for r in warnings)


def _probe_app(index: Any) -> FastAPI:
    """A one-route app that answers through :func:`_serve_index_read`.

    The route records the thread its read ran on and whether an event loop was
    running there, which is how the "index reads never run on the event loop"
    contract is asserted rather than assumed.
    """
    app = FastAPI()
    app.state.reads = []

    def read(handle: Any) -> dict[str, Any]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            on_loop = False
        else:
            on_loop = True
        app.state.reads.append((threading.get_ident(), on_loop))
        # A real query rather than a look at the meta row: taking a cursor is
        # what a closed index refuses, and that refusal is half of what this
        # helper is asserted on.
        return {"channels": handle.cursor().execute("SELECT count(*) FROM channels").fetchone()[0]}

    @app.get("/probe")
    async def probe(request: Request) -> Any:
        return await _serve_index_read(request, "probe", read)

    if index is not None:
        app.state.graph_index = index
    return app


class TestServeIndexRead:
    """Every index-backed route shares one answer for "there is no index"."""

    @pytest.fixture()
    def graph_config(self, monkeypatch):
        """Point the config seam at a project that has a corpus."""

        def install(config: dict[str, Any]) -> None:
            monkeypatch.setattr("osprey.utils.workspace.load_osprey_config", lambda: config)

        install({"services": {"graphdb": {"ttl_path": "./data/facility.ttl"}}})
        return install

    def test_a_read_answers_off_the_event_loop(self, graph_config):
        with open_demo_index() as index:
            app = _probe_app(index)
            resp = TestClient(app).get("/probe")

        assert resp.status_code == 200
        assert resp.json() == {"channels": len(DEMO_ROWS)}
        assert len(app.state.reads) == 1
        assert app.state.reads[0][1] is False

    def test_an_absence_answers_503_with_the_build_remedy(self, graph_config, tmp_path):
        absence = GraphIndexAbsence("missing", tmp_path / "g.duckdb", "No search index at g.")

        resp = TestClient(_probe_app(absence)).get("/probe")

        assert resp.status_code == 503
        body = resp.json()
        assert body["detail"] == "No search index at g."
        assert body["error_type"] == "service_unavailable"
        assert any(GRAPHDB_BUILD_INDEX_COMMAND in line for line in body["suggestions"])
        assert any("osprey build" in line for line in body["suggestions"])

    def test_a_project_with_no_corpus_is_told_to_configure_one(self, graph_config, tmp_path):
        graph_config({"services": {"graphdb": {"uri": "bolt://localhost:7687"}}})
        absence = GraphIndexAbsence("missing", tmp_path / "g.duckdb", "No search index at g.")

        body = TestClient(_probe_app(absence)).get("/probe").json()

        assert body["suggestions"] == [
            "No corpus is configured: set services.graphdb.ttl_path to the facility's "
            "Turtle file, then build the index."
        ]

    def test_an_app_holding_no_index_at_all_still_answers_the_remedy(self, graph_config):
        resp = TestClient(_probe_app(None)).get("/probe")

        assert resp.status_code == 503
        body = resp.json()
        assert body["detail"] == "The search index is not open."
        assert any(GRAPHDB_BUILD_INDEX_COMMAND in line for line in body["suggestions"])

    def test_a_closed_index_answers_503_rather_than_500(self, graph_config):
        """A request racing shutdown is unavailability, not a bug in the route."""
        index = open_demo_index()
        index.close()

        resp = TestClient(_probe_app(index)).get("/probe")

        assert resp.status_code == 503
        assert resp.json()["error_type"] == "service_unavailable"
        assert str(index.path) in resp.json()["detail"]
