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

import httpx
import pytest

from osprey.interfaces.channel_finder.database_api import (
    GRAPH_CHANNEL_COUNT_CYPHER,
    GRAPH_DEVICE_COUNT_CYPHER,
    GRAPH_ONTOLOGY_CYPHER,
    GRAPH_SECTION_COUNT_CYPHER,
    GRAPH_SIGNAL_COUNT_CYPHER,
)
from osprey.services.channel_finder.graph_queries import (
    GRAPH_DEVICE_CYPHER,
    GRAPH_SEARCH_CYPHER,
)
from osprey.services.facility_knowledge.seeder.prompt_snapshot import RELATIONSHIP_TYPES_CYPHER
from tests.interfaces.channel_finder.graph_fixture import (
    DEMO_CLASS_COUNT,
    DEMO_CLASS_ROWS,
    DEMO_COUNTS,
    DEMO_DEVICE_ROW,
    DEMO_RELATIONSHIP_ROWS,
    DEMO_RELATIONSHIP_TYPES,
    DEMO_SEARCH_FACET_OVERFLOW,
    DEMO_SEARCH_ROW,
    SEARCH_FACET_CAP,
    SEARCH_PAGE_SIZE,
    FakeGraphContext,
    class_uri,
    demo_context,
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
        # The fake store is installed and reachable: the seeded taxonomy comes
        # back whole rather than the empty-store or unavailable answer.
        assert len(data["classes"]) == DEMO_CLASS_COUNT
        assert data["relationship_types"] == DEMO_RELATIONSHIP_TYPES
        assert data["empty"] is False


class TestFakeStoreDispatch:
    """Every query the routes send has an answer of its own, and only its own."""

    @pytest.mark.parametrize(
        ("cypher", "expected"),
        [
            (GRAPH_ONTOLOGY_CYPHER, DEMO_CLASS_ROWS),
            (RELATIONSHIP_TYPES_CYPHER, DEMO_RELATIONSHIP_ROWS),
            (GRAPH_DEVICE_COUNT_CYPHER, [{"n": DEMO_COUNTS["devices"]}]),
            (GRAPH_CHANNEL_COUNT_CYPHER, [{"n": DEMO_COUNTS["channels"]}]),
            (GRAPH_SIGNAL_COUNT_CYPHER, [{"n": DEMO_COUNTS["signals"]}]),
            (GRAPH_SECTION_COUNT_CYPHER, [{"n": DEMO_COUNTS["sections"]}]),
            (GRAPH_SEARCH_CYPHER, [DEMO_SEARCH_ROW]),
            (GRAPH_DEVICE_CYPHER, [DEMO_DEVICE_ROW]),
        ],
    )
    def test_each_query_gets_its_own_canned_answer(self, cypher, expected):
        assert demo_context().run_read(cypher).rows == expected

    def test_the_four_censuses_are_told_apart_from_each_other(self):
        """No two counts collide: four queries, four different numbers."""
        ctx = demo_context()

        counts = [
            ctx.run_read(cypher).rows[0]["n"]
            for cypher in (
                GRAPH_DEVICE_COUNT_CYPHER,
                GRAPH_CHANNEL_COUNT_CYPHER,
                GRAPH_SIGNAL_COUNT_CYPHER,
                GRAPH_SECTION_COUNT_CYPHER,
            )
        ]

        assert counts == [
            DEMO_COUNTS["devices"],
            DEMO_COUNTS["channels"],
            DEMO_COUNTS["signals"],
            DEMO_COUNTS["sections"],
        ]

    def test_a_query_the_fake_does_not_know_fails_loudly(self):
        """A read nobody taught it is a drifted test, not a fallthrough."""
        with pytest.raises(AssertionError, match=r"unkeyed cypher: MATCH \(x:Nowhere\)"):
            demo_context().run_read("MATCH (x:Nowhere) RETURN x")

    def test_an_unlisted_count_answers_no_rows(self):
        ctx = demo_context(counts={"devices": 7})

        assert ctx.run_read(GRAPH_DEVICE_COUNT_CYPHER).rows == [{"n": 7}]
        assert ctx.run_read(GRAPH_SIGNAL_COUNT_CYPHER).rows == []

    def test_every_read_records_its_parameters_and_row_cap(self):
        """A route's request contract is assertable from the store side."""
        ctx = demo_context()

        ctx.run_read(GRAPH_SEARCH_CYPHER, {"tokens": ["qfa"], "skip": 0}, max_rows=1)
        ctx.run_read(GRAPH_SEARCH_CYPHER, {"tokens": ["bpm"], "skip": 50}, max_rows=1)
        ctx.run_read(GRAPH_DEVICE_CYPHER, {"uri": DEMO_DEVICE_ROW["uri"]}, max_rows=1)

        assert ctx.reads_of(GRAPH_SEARCH_CYPHER) == [
            ({"tokens": ["qfa"], "skip": 0}, 1),
            ({"tokens": ["bpm"], "skip": 50}, 1),
        ]
        assert ctx.last_params(GRAPH_SEARCH_CYPHER) == {"tokens": ["bpm"], "skip": 50}
        assert ctx.last_params(GRAPH_DEVICE_CYPHER) == {"uri": DEMO_DEVICE_ROW["uri"]}
        # The older recording the endpoint tests read stays exactly as it was.
        assert [cypher for cypher, _ in ctx.calls].count(GRAPH_SEARCH_CYPHER) == 2

    def test_asking_for_parameters_of_a_read_that_never_happened_fails(self):
        with pytest.raises(AssertionError, match="no read sent"):
            demo_context().last_params(GRAPH_SEARCH_CYPHER)


class TestDemoSearchPage:
    """The page the fake serves looks like a page an operator would be shown."""

    def test_the_page_is_one_full_page_of_a_longer_result(self):
        rows = DEMO_SEARCH_ROW["rows"]

        assert len(rows) == SEARCH_PAGE_SIZE
        # More matches than fit on the page, so the finder really paginates.
        assert DEMO_SEARCH_ROW["total"] > SEARCH_PAGE_SIZE
        assert rows == sorted(rows, key=lambda row: row["fullPv"])

    def test_every_row_carries_the_columns_the_search_query_projects(self):
        for row in DEMO_SEARCH_ROW["rows"]:
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

    def test_exactly_one_channel_both_reads_and_writes(self):
        both = [
            row
            for row in DEMO_SEARCH_ROW["rows"]
            if {"READSSIGNAL", "WRITESSIGNAL"} <= set(row["edges"])
        ]

        assert len(both) == 1
        assert both[0]["edges"] == ["READSSIGNAL", "WRITESSIGNAL"]

    def test_exactly_one_channel_carries_no_signal_at_all(self):
        bare = [row for row in DEMO_SEARCH_ROW["rows"] if not row["edges"]]

        assert len(bare) == 1
        assert bare[0]["signals"] == []

    def test_the_read_write_channel_is_counted_under_all_three_directions(self):
        counts = {entry["value"]: entry["count"] for entry in DEMO_SEARCH_ROW["facets"]["dir"]}
        reads = sum(1 for row in DEMO_SEARCH_ROW["rows"] if "READSSIGNAL" in row["edges"])
        writes = sum(1 for row in DEMO_SEARCH_ROW["rows"] if "WRITESSIGNAL" in row["edges"])

        assert counts["RW"] == 1
        assert counts["none"] == 1
        assert counts["R"] == reads
        assert counts["W"] == writes

    def test_the_class_facet_names_classes_by_uri_including_an_abstract_one(self):
        entries = {entry["value"]: entry["count"] for entry in DEMO_SEARCH_ROW["facets"]["class"]}

        assert all(value.startswith("https://") for value in entries)
        # Magnet is a branch nothing on the page is typed by directly, so its
        # count can only have come from rolling its subclasses up.
        magnet = class_uri("Magnet")
        assert entries[magnet] > entries[class_uri("Quadrupole")]

    def test_every_facet_of_the_default_page_is_well_under_the_cap(self):
        for name, entries in DEMO_SEARCH_ROW["facets"].items():
            assert 0 < len(entries) <= SEARCH_FACET_CAP, name
            assert entries == sorted(entries, key=lambda e: (-e["count"], e["value"])), name

    def test_the_overflow_export_runs_one_entry_past_the_cap(self):
        assert len(DEMO_SEARCH_FACET_OVERFLOW["signal"]) == SEARCH_FACET_CAP + 1
        # Only the one list overflows; the rest are the page's own facets.
        assert DEMO_SEARCH_FACET_OVERFLOW["section"] == DEMO_SEARCH_ROW["facets"]["section"]
        assert set(DEMO_SEARCH_FACET_OVERFLOW) == set(DEMO_SEARCH_ROW["facets"])


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

    def test_a_device_the_store_does_not_hold_answers_no_rows(self):
        """Absence is an empty result, which is how a route reads a miss."""
        ctx = demo_context(device_result=None)

        assert ctx.run_read(GRAPH_DEVICE_CYPHER).rows == []

    def test_a_test_can_hand_the_device_read_one_row_of_its_own(self):
        ctx = FakeGraphContext(device_result={"uri": "urn:x", "device": "X", "signals": []})

        assert ctx.run_read(GRAPH_DEVICE_CYPHER).rows == [
            {"uri": "urn:x", "device": "X", "signals": []}
        ]

    def test_a_search_that_matches_nothing_answers_no_rows(self):
        assert demo_context(search_result=None).run_read(GRAPH_SEARCH_CYPHER).rows == []
