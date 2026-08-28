"""Tests for the graph-paradigm query constants and pure helpers.

These cover the two pieces of the graph explorer that hold logic rather than
plumbing: the taxonomy pruner that decides which ``:Class`` rows are worth
drawing, and the error mapper that turns a failed store read into a response
body. Both are pure — no driver, no store, no app — so the rows here are
hand-written in the shape :data:`GRAPH_ONTOLOGY_CYPHER` returns, and the
demo-shaped case pins the count an operator actually sees.
"""

from __future__ import annotations

import pytest

from osprey.interfaces.channel_finder.database_api import (
    GRAPH_CHANNEL_COUNT_CYPHER,
    GRAPH_DEVICE_COUNT_CYPHER,
    GRAPH_ONTOLOGY_CYPHER,
    GRAPH_SECTION_COUNT_CYPHER,
    GRAPH_SIGNAL_COUNT_CYPHER,
    RELATIONSHIP_TYPES_CYPHER,
    _class_name,
    _graph_error_payload,
    _prune_device_taxonomy,
)
from osprey.mcp_server.graph.server_context import (
    GraphNotConfigured,
    GraphQueryTimeout,
    GraphUnreachable,
)

_SEM = "https://narad.example.org/schema/shared_semantics/"


def _row(name: str, rollup: int, parents: list[str], alt: list[str] | None = None) -> dict:
    """Build one ontology row in the shape the ontology query returns."""
    return {
        "uri": _SEM + name,
        "altLabel": alt,
        "parents": [_SEM + parent for parent in parents],
        "rollup": rollup,
    }


def _demo_rows() -> list[dict]:
    """Return a demo-shaped corpus: 21 classes, two of them non-device leaves."""
    return [
        _row("AcceleratorDevice", 100, []),
        _row("Magnet", 50, ["AcceleratorDevice"]),
        _row("Instrumentation", 30, ["AcceleratorDevice"]),
        _row("Vacuum", 12, ["AcceleratorDevice"]),
        _row("RFSystem", 8, ["AcceleratorDevice"]),
        _row("Quadrupole", 20, ["Magnet"], ["QF", "Quad"]),
        _row("Dipole", 15, ["Magnet"], ["Bend"]),
        _row("HCorrector", 8, ["Magnet"]),
        _row("VCorrector", 7, ["Magnet"]),
        _row("BPM", 25, ["Instrumentation"], ["Beam Position Monitor"]),
        _row("Screen", 5, ["Instrumentation"]),
        _row("IonPump", 7, ["Vacuum"]),
        _row("Valve", 5, ["Vacuum"]),
        _row("Cavity", 8, ["RFSystem"]),
        # Abstract branch: no devices of its own, but a class child keeps it.
        _row("Cryostat", 0, ["AcceleratorDevice"]),
        _row("ColdBox", 2, ["Cryostat"]),
        # Two parents — the tree is a DAG, not a strict hierarchy.
        _row("SteeringMagnet", 4, ["Magnet", "AcceleratorDevice"]),
        _row("Undulator", 6, ["AcceleratorDevice"]),
        _row("Septum", 3, ["Magnet"]),
        # Not devices: zero rollup and nothing declares them a parent.
        _row("SemanticSignal", 0, []),
        _row("ChannelBinding", 0, []),
    ]


class TestCypherConstants:
    def test_ontology_query_defines_devices_as_bound_resources(self):
        assert "(d:Resource)-[:TYPE]->(sub)" in GRAPH_ONTOLOGY_CYPHER
        assert "(d)-[:HASBINDING]->(:ChannelBinding)" in GRAPH_ONTOLOGY_CYPHER

    def test_ontology_query_bounds_the_subclass_walk(self):
        # An unbounded walk would never terminate on a SUBCLASSOF cycle.
        assert "[:SUBCLASSOF*0..10]" in GRAPH_ONTOLOGY_CYPHER

    def test_ontology_query_returns_the_four_columns_the_pruner_reads(self):
        assert "AS uri" in GRAPH_ONTOLOGY_CYPHER
        assert "AS altLabel" in GRAPH_ONTOLOGY_CYPHER
        assert "parents" in GRAPH_ONTOLOGY_CYPHER
        assert "AS rollup" in GRAPH_ONTOLOGY_CYPHER

    @pytest.mark.parametrize(
        "cypher",
        [
            GRAPH_DEVICE_COUNT_CYPHER,
            GRAPH_SIGNAL_COUNT_CYPHER,
            GRAPH_SECTION_COUNT_CYPHER,
            GRAPH_CHANNEL_COUNT_CYPHER,
        ],
    )
    def test_count_queries_return_a_single_aliased_count(self, cypher):
        assert cypher.startswith("MATCH ")
        assert "count(" in cypher
        assert cypher.rstrip().endswith("AS n")

    def test_device_count_matches_the_ontology_definition_of_a_device(self):
        assert "(d:Resource)-[:HASBINDING]->(:ChannelBinding)" in GRAPH_DEVICE_COUNT_CYPHER
        assert "count(DISTINCT d)" in GRAPH_DEVICE_COUNT_CYPHER

    def test_section_count_skips_devices_with_no_section(self):
        assert "d.sectionCode IS NOT NULL" in GRAPH_SECTION_COUNT_CYPHER
        assert "count(DISTINCT d.sectionCode)" in GRAPH_SECTION_COUNT_CYPHER

    def test_reexported_constants_match_their_source(self):
        from osprey.services.channel_finder.benchmarks.runner import (
            GRAPH_CHANNEL_COUNT_CYPHER as source_channels,
        )
        from osprey.services.facility_knowledge.seeder.prompt_snapshot import (
            RELATIONSHIP_TYPES_CYPHER as source_rels,
        )

        assert GRAPH_CHANNEL_COUNT_CYPHER == source_channels
        assert RELATIONSHIP_TYPES_CYPHER == source_rels


class TestClassName:
    @pytest.mark.parametrize(
        ("uri", "expected"),
        [
            ("https://narad.example.org/schema/shared_semantics/Quadrupole", "Quadrupole"),
            ("http://example.org/ontology#BPM", "BPM"),
            ("http://example.org/ontology/nested#Valve", "Valve"),
            ("Bare", "Bare"),
        ],
    )
    def test_name_is_the_trailing_fragment(self, uri, expected):
        assert _class_name(uri) == expected


class TestPruneDeviceTaxonomy:
    def test_demo_shaped_corpus_keeps_nineteen_of_twenty_one(self):
        kept = _prune_device_taxonomy(_demo_rows())

        assert len(kept) == 19
        names = {entry["name"] for entry in kept}
        assert "SemanticSignal" not in names
        assert "ChannelBinding" not in names

    def test_abstract_parent_with_a_class_child_is_kept(self):
        kept = _prune_device_taxonomy(_demo_rows())

        cryostat = next(entry for entry in kept if entry["name"] == "Cryostat")
        assert cryostat["rollup"] == 0

    def test_class_with_two_parents_keeps_both(self):
        kept = _prune_device_taxonomy(_demo_rows())

        steering = next(entry for entry in kept if entry["name"] == "SteeringMagnet")
        assert steering["parents"] == [_SEM + "Magnet", _SEM + "AcceleratorDevice"]

    def test_orphan_zero_rollup_leaf_is_pruned(self):
        rows = [
            _row("Magnet", 4, []),
            _row("Orphan", 0, []),
        ]

        kept = _prune_device_taxonomy(rows)

        assert [entry["name"] for entry in kept] == ["Magnet"]

    def test_self_referential_parent_does_not_rescue_an_empty_class(self):
        rows = [_row("Loop", 0, ["Loop"])]

        assert _prune_device_taxonomy(rows) == []

    def test_result_is_sorted_by_name(self):
        kept = _prune_device_taxonomy(_demo_rows())

        assert [entry["name"] for entry in kept] == sorted(entry["name"] for entry in kept)

    def test_missing_alt_labels_normalise_to_a_list(self):
        kept = _prune_device_taxonomy(_demo_rows())

        quadrupole = next(entry for entry in kept if entry["name"] == "Quadrupole")
        magnet = next(entry for entry in kept if entry["name"] == "Magnet")
        assert quadrupole["altLabel"] == ["QF", "Quad"]
        assert magnet["altLabel"] == []

    def test_every_kept_entry_carries_the_full_shape(self):
        kept = _prune_device_taxonomy(_demo_rows())

        for entry in kept:
            assert set(entry) == {"uri", "name", "altLabel", "parents", "rollup"}

    def test_empty_input_yields_no_classes(self):
        assert _prune_device_taxonomy([]) == []

    def test_rows_without_a_uri_are_skipped(self):
        rows = [{"uri": None, "altLabel": None, "parents": [], "rollup": 3}]

        assert _prune_device_taxonomy(rows) == []


class TestGraphErrorPayload:
    def test_unreachable_store_maps_to_503_with_its_own_remedy(self):
        exc = GraphUnreachable("The graph store did not answer.")

        status, body = _graph_error_payload(exc)

        assert status == 503
        assert body["detail"] == "The graph store did not answer."
        assert body["error_type"] == "service_unavailable"
        assert body["suggestions"] == GraphUnreachable.default_suggestions()

    def test_missing_context_maps_to_503_with_a_usable_remedy(self):
        status, body = _graph_error_payload(None)

        assert status == 503
        assert body["detail"] == "Graph store is not available."
        assert body["error_type"] == "service_unavailable"
        assert body["suggestions"]
        assert any("graphdb" in suggestion for suggestion in body["suggestions"])

    @pytest.mark.parametrize(
        ("exc", "error_type"),
        [
            (GraphNotConfigured("no store"), "not_configured"),
            (GraphQueryTimeout("too slow"), "timeout_error"),
        ],
    )
    def test_every_cause_answers_503_carrying_its_own_error_type(self, exc, error_type):
        status, body = _graph_error_payload(exc)

        assert status == 503
        assert body["error_type"] == error_type

    def test_suggestions_are_copied_not_shared(self):
        exc = GraphUnreachable("down")

        _, body = _graph_error_payload(exc)
        body["suggestions"].append("mutated")

        assert "mutated" not in exc.suggestions
