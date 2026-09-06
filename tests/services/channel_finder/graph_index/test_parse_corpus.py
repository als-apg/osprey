"""Tests for the corpus parse that derives the search index's rows.

The rows must reproduce what the n10s-seeded store answers through
``GRAPH_SEARCH_CYPHER`` and ``GRAPH_ONTOLOGY_CYPHER``; a parity lane checks that
against a live store, and these tests pin each rule the parse copies from the
Cypher on corpora small enough to state the expected rows by hand. The shipped
demo corpus is then parsed against the counts the store integration tests
verified against Neo4j.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import time
from importlib.resources import as_file, files
from pathlib import Path

import pytest
from tests.services.channel_finder.graph_index import corpora

from osprey.channel_roster import RosterSource, RosterSourceKind
from osprey.channel_roster.graph import _corpus_readbacks, _records, read_graph_roster
from osprey.services.channel_finder.core.exceptions import GraphIndexBuildError
from osprey.services.channel_finder.graph_index.builder import (
    EDGE_READS,
    EDGE_WRITES,
    BindingRow,
    ClassRow,
    ParsedCorpus,
    build_graph_index,
    parse_corpus,
)

SEM = corpora.NARAD_SEM
DEVICE = corpora.DEVICE
BINDING = corpora.BINDING

#: Store-verified counts for the shipped demo corpus, as
#: ``tests/integration/test_graph_mcp.py`` pins them against Neo4j.
DEMO_DEVICES = 512
DEMO_BINDINGS = 2908
DEMO_WRITE_ONLY = 396
DEMO_READ_ONLY = 2512
DEMO_MAGNETS = 382
#: 21 ``owl:Class`` subjects less the ``SemanticSignal`` and ``ChannelBinding``
#: leaves that pruning drops.
DEMO_CLASS_COUNT = 19


def _rows_by_pv(parsed: ParsedCorpus) -> dict[str, BindingRow]:
    rows = {row.full_pv: row for row in parsed.binding_rows}
    assert len(rows) == len(parsed.binding_rows), "fixture repeats a fullPv; index by uri"
    return rows


def _classes_by_name(parsed: ParsedCorpus) -> dict[str, ClassRow]:
    return {row.name: row for row in parsed.class_rows}


@pytest.fixture(scope="module")
def demo_path():
    resource = (
        files("osprey.templates")
        .joinpath("apps")
        .joinpath("control_assistant")
        .joinpath("data")
        .joinpath("demo_machine.ttl")
    )
    with as_file(resource) as path:
        yield path


@pytest.fixture(scope="module")
def demo(demo_path: Path) -> ParsedCorpus:
    return parse_corpus(demo_path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def demo_index(demo_path: Path, tmp_path_factory) -> Path:
    """The demo corpus built into an index, for the one test that reads one.

    The roster reader opens an index rather than a corpus, so reaching it as
    an oracle means writing one first.
    """
    index_path = tmp_path_factory.mktemp("demo_index") / "graph.duckdb"
    build_graph_index(demo_path, index_path)
    return index_path


class TestSubclassChain:
    @pytest.fixture(scope="class")
    def parsed(self) -> ParsedCorpus:
        return parse_corpus(corpora.SUBCLASS_CHAIN)

    def test_one_row_per_binding_ordered_by_full_pv(self, parsed):
        assert [row.full_pv for row in parsed.binding_rows] == [
            "SR:MAG:QF1:CURRENT:RB",
            "SR:MAG:QF1:CURRENT:SP",
            "SR:MAG:QF1:NOTE",
        ]

    def test_field_order_matches_the_bindings_table(self):
        assert list(BindingRow.__dataclass_fields__) == [
            "binding_uri",
            "full_pv",
            "description",
            "device_uri",
            "device_name",
            "section",
            "system",
            "edges",
            "signal_uris",
            "signal_names",
            "class_uris",
            "haystack",
        ]
        assert list(ClassRow.__dataclass_fields__) == [
            "uri",
            "name",
            "alt_labels",
            "parents",
            "direct_devices",
            "rollup_devices",
        ]

    def test_a_write_binding(self, parsed):
        row = _rows_by_pv(parsed)["SR:MAG:QF1:CURRENT:SP"]
        assert row.binding_uri == BINDING + "QF1_SP"
        assert row.description == "Quadrupole QF1 current setpoint"
        assert row.device_uri == DEVICE + "demo_SR_QF1"
        assert row.device_name == "QF1"
        assert row.section == "SR"
        assert row.system == "MAG"
        assert row.edges == [EDGE_WRITES]
        assert row.signal_uris == [SEM + "quad_current_sp"]
        assert row.signal_names == ["quad_current_sp"]

    def test_a_read_binding(self, parsed):
        row = _rows_by_pv(parsed)["SR:MAG:QF1:CURRENT:RB"]
        assert row.edges == [EDGE_READS]
        assert row.signal_names == ["quad_current_rb"]

    def test_a_binding_without_a_signal_edge(self, parsed):
        row = _rows_by_pv(parsed)["SR:MAG:QF1:NOTE"]
        assert row.description is None
        assert row.edges == []
        assert row.signal_uris == []
        assert row.signal_names == []

    def test_class_uris_are_the_type_and_every_class_ancestor(self, parsed):
        row = _rows_by_pv(parsed)["SR:MAG:QF1:CURRENT:SP"]
        assert row.class_uris == [
            SEM + "AcceleratorDevice",
            SEM + "Magnet",
            SEM + "Quadrupole",
        ]

    def test_owl_thing_is_no_ancestor_because_the_store_does_not_label_it_a_class(self, parsed):
        for row in parsed.binding_rows:
            assert "http://www.w3.org/2002/07/owl#Thing" not in row.class_uris
        for row in parsed.class_rows:
            assert "http://www.w3.org/2002/07/owl#Thing" not in row.parents

    def test_haystack_is_full_pv_description_device_signals_then_class_names(self, parsed):
        row = _rows_by_pv(parsed)["SR:MAG:QF1:CURRENT:SP"]
        assert row.haystack == (
            "sr:mag:qf1:current:sp "
            "quadrupole qf1 current setpoint "
            "qf1 "
            "quad_current_sp "
            "acceleratordevice magnet quadrupole "
            "magnet "
            "focusing magnet quad quadrupole"
        )

    def test_haystack_skips_absent_fields_rather_than_writing_none(self, parsed):
        row = _rows_by_pv(parsed)["SR:MAG:QF1:NOTE"]
        assert row.haystack == (
            "sr:mag:qf1:note qf1 acceleratordevice magnet quadrupole "
            "magnet focusing magnet quad quadrupole"
        )
        assert "none" not in row.haystack

    def test_alt_labels_of_every_ancestor_lower_cased_match_a_token(self, parsed):
        row = _rows_by_pv(parsed)["SR:MAG:QF1:NOTE"]
        assert "focusing magnet" in row.haystack
        assert "quad" in row.haystack

    def test_class_rows_are_the_pruned_taxonomy_sorted_by_name(self, parsed):
        assert [row.name for row in parsed.class_rows] == [
            "AcceleratorDevice",
            "Magnet",
            "Quadrupole",
        ]

    def test_class_row_counts_and_parents(self, parsed):
        by_name = _classes_by_name(parsed)
        root = by_name["AcceleratorDevice"]
        assert (root.direct_devices, root.rollup_devices, root.parents) == (0, 1, [])
        magnet = by_name["Magnet"]
        assert (magnet.direct_devices, magnet.rollup_devices) == (0, 1)
        assert magnet.parents == [SEM + "AcceleratorDevice"]
        quad = by_name["Quadrupole"]
        assert (quad.direct_devices, quad.rollup_devices) == (1, 1)
        assert quad.parents == [SEM + "Magnet"]

    def test_alt_labels_are_kept_verbatim_and_sorted(self, parsed):
        by_name = _classes_by_name(parsed)
        assert by_name["Quadrupole"].alt_labels == ["Focusing Magnet", "quad", "quadrupole"]
        assert by_name["Magnet"].alt_labels == ["magnet"]
        assert by_name["AcceleratorDevice"].alt_labels == []

    def test_a_device_class_nothing_is_typed_as_is_pruned(self, parsed):
        assert "Sextupole" not in _classes_by_name(parsed)

    def test_signal_and_binding_classes_are_pruned(self, parsed):
        names = _classes_by_name(parsed)
        assert "ChannelBinding" not in names
        assert "SemanticSignal" not in names

    def test_roster_raw_material_is_shaped_as_the_roster_reader_shapes_it(self, parsed):
        assert parsed.writes == {corpora_uri("QF1_SP")}
        assert parsed.reads == {corpora_uri("QF1_RB")}
        assert sorted(parsed.bindings) == [
            ("SR:MAG:QF1:CURRENT:RB", corpora_uri("QF1_RB")),
            ("SR:MAG:QF1:CURRENT:SP", corpora_uri("QF1_SP")),
            ("SR:MAG:QF1:NOTE", corpora_uri("QF1_NOTE")),
        ]

    def test_census_extras(self, parsed):
        assert parsed.section_codes == frozenset({"SR"})
        assert parsed.signal_count == 2


def corpora_uri(binding: str):
    from rdflib import URIRef

    return URIRef(BINDING + binding)


class TestBothEdges:
    @pytest.fixture(scope="class")
    def parsed(self) -> ParsedCorpus:
        return parse_corpus(corpora.BOTH_EDGES)

    def test_edges_are_in_fixed_ascending_order(self, parsed):
        row = _rows_by_pv(parsed)["SR:MAG:QF2:CURRENT"]
        assert row.edges == [EDGE_READS, EDGE_WRITES]
        assert row.edges == sorted(row.edges)

    def test_signals_are_ordered_by_name(self, parsed):
        row = _rows_by_pv(parsed)["SR:MAG:QF2:CURRENT"]
        assert row.signal_names == ["quad_current_rb", "quad_current_sp"]
        assert row.signal_uris == [SEM + "quad_current_rb", SEM + "quad_current_sp"]

    def test_one_signal_reached_both_ways_is_listed_once_with_two_edges(self, parsed):
        row = _rows_by_pv(parsed)["SR:MAG:QF2:SAME"]
        assert row.edges == [EDGE_READS, EDGE_WRITES]
        assert row.signal_names == ["quad_current_sp"]


class TestSharedFullPv:
    @pytest.fixture(scope="class")
    def parsed(self) -> ParsedCorpus:
        return parse_corpus(corpora.SHARED_FULL_PV)

    def test_two_binding_nodes_under_one_full_pv_are_two_rows(self, parsed):
        rows = [row for row in parsed.binding_rows if row.full_pv == "SR:MAG:SHARED:CURRENT"]
        assert len(rows) == 2
        assert [row.device_name for row in rows] == ["QF3", "QF4"]
        assert [row.edges for row in rows] == [[EDGE_READS], [EDGE_WRITES]]

    def test_rows_are_ordered_by_full_pv_then_device_uri(self, parsed):
        keys = [(row.full_pv, row.device_uri) for row in parsed.binding_rows]
        assert keys == sorted(keys)

    def test_the_roster_vote_collapses_them_to_one_address_with_no_direction(
        self, parsed, tmp_path
    ):
        source = RosterSource(kind=RosterSourceKind.GRAPH, path=tmp_path / "shared.ttl")
        readbacks = _corpus_readbacks(parsed.graph, parsed.writes, parsed.reads, parsed.bindings)
        records = _records(parsed.bindings, source, parsed.writes, parsed.reads, readbacks)
        assert [(r.address, r.direction) for r in records] == [("SR:MAG:SHARED:CURRENT", None)]


class TestBindingUnderTwoDevices:
    @pytest.fixture(scope="class")
    def parsed(self) -> ParsedCorpus:
        return parse_corpus(corpora.BINDING_UNDER_TWO_DEVICES)

    def test_one_binding_node_hung_under_two_devices_is_two_rows(self, parsed):
        assert len(parsed.binding_rows) == 2
        assert {row.binding_uri for row in parsed.binding_rows} == {BINDING + "TWICE"}
        assert [(row.device_name, row.section) for row in parsed.binding_rows] == [
            ("QF6", "BR"),
            ("QF5", "SR"),
        ]

    def test_the_device_is_counted_once_per_class(self, parsed):
        by_name = _classes_by_name(parsed)
        assert by_name["Quadrupole"].direct_devices == 2
        assert by_name["Magnet"].rollup_devices == 2

    def test_the_roster_sees_one_binding(self, parsed):
        assert len(parsed.bindings) == 1


class TestDeviceWithoutSectionOrSystem:
    @pytest.fixture(scope="class")
    def parsed(self) -> ParsedCorpus:
        return parse_corpus(corpora.DEVICE_WITHOUT_SECTION_OR_SYSTEM)

    def test_absent_columns_are_none_never_empty_strings(self, parsed):
        [row] = parsed.binding_rows
        assert row.section is None
        assert row.system is None
        assert row.device_name is None

    def test_haystack_carries_only_what_the_corpus_states(self, parsed):
        [row] = parsed.binding_rows
        assert row.haystack == (
            "nowhere:rb unplaced readback quad_current_rb "
            "acceleratordevice magnet quadrupole magnet focusing magnet quad quadrupole"
        )

    def test_section_census_is_empty(self, parsed):
        assert parsed.section_codes == frozenset()


class TestClassCycle:
    @pytest.fixture(scope="class")
    def parsed(self) -> ParsedCorpus:
        return parse_corpus(corpora.CLASS_CYCLE)

    def test_a_cycle_terminates_and_both_classes_are_ancestors(self, parsed):
        row = _rows_by_pv(parsed)["LOOP:RB"]
        assert row.class_uris == [SEM + "Loop_A", SEM + "Loop_B"]

    def test_both_loop_classes_roll_the_device_up(self, parsed):
        by_name = _classes_by_name(parsed)
        assert (by_name["Loop_A"].direct_devices, by_name["Loop_A"].rollup_devices) == (1, 1)
        assert (by_name["Loop_B"].direct_devices, by_name["Loop_B"].rollup_devices) == (0, 1)
        assert by_name["Loop_A"].parents == [SEM + "Loop_B"]
        assert by_name["Loop_B"].parents == [SEM + "Loop_A"]

    def test_a_self_parent_is_kept_as_the_store_reports_it(self, parsed):
        selfish = _classes_by_name(parsed)["Selfish"]
        assert selfish.parents == [SEM + "Selfish"]
        assert (selfish.direct_devices, selfish.rollup_devices) == (1, 1)

    def test_a_signal_without_a_label_is_named_by_its_uri_tail(self, parsed):
        row = _rows_by_pv(parsed)["LOOP:RB"]
        assert row.signal_names == ["loop_signal"]
        assert "loop_signal" in row.haystack


class TestDeepChain:
    @pytest.fixture(scope="class")
    def parsed(self) -> ParsedCorpus:
        return parse_corpus(corpora.DEEP_CHAIN)

    def test_ancestors_stop_at_ten_hops(self, parsed):
        [row] = parsed.binding_rows
        assert row.class_uris == [SEM + f"Deep_{i:02d}" for i in range(11)]
        assert SEM + "Deep_11" not in row.class_uris

    def test_rollup_stops_at_ten_hops_but_the_parent_class_survives_pruning(self, parsed):
        by_name = _classes_by_name(parsed)
        assert all(by_name[f"Deep_{i:02d}"].rollup_devices == 1 for i in range(11))
        assert by_name["Deep_11"].rollup_devices == 0
        assert by_name["Deep_00"].direct_devices == 1
        assert by_name["Deep_01"].direct_devices == 0


class TestUntypedTargets:
    @pytest.fixture(scope="class")
    def parsed(self) -> ParsedCorpus:
        return parse_corpus(corpora.UNTYPED_TARGETS)

    def test_a_has_binding_target_not_typed_channel_binding_is_no_row(self, parsed):
        assert "SR:MAG:QF7:UNTYPED" not in _rows_by_pv(parsed)
        assert len(parsed.binding_rows) == 2

    def test_but_the_roster_raw_material_keeps_it(self, parsed):
        addresses = sorted(address for address, _ in parsed.bindings)
        assert addresses == ["SR:MAG:QF7:LABELLESS", "SR:MAG:QF7:RB", "SR:MAG:QF7:UNTYPED"]
        assert len(parsed.reads) == 3
        assert parsed.writes == set()

    def test_an_edge_to_a_node_not_typed_semantic_signal_is_no_edge(self, parsed):
        row = _rows_by_pv(parsed)["SR:MAG:QF7:RB"]
        assert row.edges == []
        assert row.signal_names == []

    def test_a_typed_signal_without_a_label_is_named_by_its_uri_tail(self, parsed):
        row = _rows_by_pv(parsed)["SR:MAG:QF7:LABELLESS"]
        assert row.edges == [EDGE_READS]
        assert row.signal_names == ["unlabelled_signal"]

    def test_a_type_not_declared_a_class_contributes_no_ancestor(self, parsed):
        row = _rows_by_pv(parsed)["SR:MAG:QF7:RB"]
        assert SEM + "NotAClass" not in row.class_uris
        assert row.class_uris == [SEM + "AcceleratorDevice", SEM + "Magnet", SEM + "Quadrupole"]

    def test_censuses_count_every_subject_the_store_would(self, parsed):
        assert parsed.section_codes == frozenset({"SR", "LTB"})
        assert parsed.signal_count == 3


class TestNoBindings:
    def test_is_not_an_error(self):
        parsed = parse_corpus(corpora.NO_BINDINGS)
        assert parsed.binding_rows == []
        assert parsed.bindings == []
        assert parsed.writes == set() and parsed.reads == set()

    def test_abstract_parents_survive_with_zero_counts(self):
        parsed = parse_corpus(corpora.NO_BINDINGS)
        assert [
            (row.name, row.direct_devices, row.rollup_devices) for row in parsed.class_rows
        ] == [
            ("AcceleratorDevice", 0, 0),
            ("Magnet", 0, 0),
        ]

    def test_empty_text_is_an_empty_corpus(self):
        parsed = parse_corpus("")
        assert parsed.binding_rows == [] and parsed.class_rows == []


class TestInvalidTurtle:
    def test_raises_graph_index_build_error_with_the_parser_message(self):
        with pytest.raises(GraphIndexBuildError) as info:
            parse_corpus(corpora.INVALID_TURTLE)
        assert "Turtle" in str(info.value)
        assert info.value.__cause__ is not None
        assert str(info.value.__cause__) in str(info.value)


class TestModuleImport:
    def test_importing_the_builder_does_not_import_rdflib_or_duckdb(self):
        script = textwrap.dedent(
            """
            import sys

            import osprey.services.channel_finder.graph_index.builder as builder

            forbidden = sorted(
                name
                for name in ("duckdb", "rdflib", "neo4j", "osprey.services.qmd")
                if name in sys.modules
            )
            assert not forbidden, forbidden
            assert callable(builder.parse_corpus)
            print("ok")
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout


class TestTheShippedDemoCorpus:
    def test_parses_in_well_under_the_build_budget(self, demo_path):
        text = demo_path.read_text(encoding="utf-8")
        started = time.perf_counter()
        parse_corpus(text)
        assert time.perf_counter() - started < 10.0

    def test_binding_and_device_counts_match_the_store(self, demo):
        assert len(demo.binding_rows) == DEMO_BINDINGS
        assert len({row.device_uri for row in demo.binding_rows}) == DEMO_DEVICES
        assert len({row.binding_uri for row in demo.binding_rows}) == DEMO_BINDINGS

    def test_directions_match_the_store(self, demo):
        writes = sum(1 for row in demo.binding_rows if row.edges == [EDGE_WRITES])
        reads = sum(1 for row in demo.binding_rows if row.edges == [EDGE_READS])
        assert (writes, reads) == (DEMO_WRITE_ONLY, DEMO_READ_ONLY)

    def test_every_row_invariant(self, demo):
        for row in demo.binding_rows:
            assert row.haystack == row.haystack.lower()
            assert row.full_pv.lower() in row.haystack
            assert set(row.edges) <= {EDGE_READS, EDGE_WRITES}
            assert row.edges == sorted(row.edges)
            assert row.section != "" and row.system != ""
            assert row.section is not None and row.system is not None
            assert len(row.signal_uris) == len(row.signal_names)
            assert row.signal_names == sorted(row.signal_names)
            assert row.class_uris == sorted(row.class_uris)
            assert row.class_uris, row.full_pv

    def test_class_rows_match_the_store(self, demo):
        assert len(demo.class_rows) == DEMO_CLASS_COUNT
        by_name = _classes_by_name(demo)
        assert by_name["AcceleratorDevice"].rollup_devices == DEMO_DEVICES
        assert by_name["AcceleratorDevice"].direct_devices == 0
        assert by_name["Magnet"].rollup_devices == DEMO_MAGNETS
        assert by_name["Magnet"].parents == [SEM + "AcceleratorDevice"]
        assert "ChannelBinding" not in by_name and "SemanticSignal" not in by_name
        assert [row.name for row in demo.class_rows] == sorted(row.name for row in demo.class_rows)

    def test_direct_counts_sum_to_the_rollup_of_the_root(self, demo):
        assert sum(row.direct_devices for row in demo.class_rows) == DEMO_DEVICES

    def test_censuses(self, demo):
        assert demo.signal_count == 113
        assert demo.section_codes == frozenset({"SR", "BR", "BTS"})

    def test_roster_raw_material_reproduces_the_roster_reader(self, demo, demo_index):
        """The parse's bindings, run through the roster's own rules, are what
        the reader answers off an index built from the same corpus.

        Both sides are attributed to the index, so the comparison is record for
        record -- provenance included -- rather than field by field.
        """
        source = RosterSource(kind=RosterSourceKind.GRAPH, path=demo_index)
        readbacks = _corpus_readbacks(demo.graph, demo.writes, demo.reads, demo.bindings)
        records = _records(demo.bindings, source, demo.writes, demo.reads, readbacks)
        assert records == read_graph_roster(source).records
