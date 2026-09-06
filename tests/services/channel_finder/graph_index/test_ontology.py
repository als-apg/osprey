"""Tests for the class tree and the badge counts an index answers with.

Both reads are cheap by construction — the tree is the ``classes`` table the
build already pruned, and the badges are the ``meta`` row it already counted —
so what these tests pin is that neither is recomputed here. A tree pruned twice
and a badge counted a second way are exactly the drifts that let the rail and
the numbers above it disagree.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.services.channel_finder.graph_index import corpora

from osprey.services.channel_finder.graph_index.builder import (
    CALLER_META_KEYS,
    ParsedCorpus,
    build_from_rows,
    channels_from_rows,
    parse_corpus,
)
from osprey.services.channel_finder.graph_index.reader import GraphIndex, open_graph_index
from osprey.services.channel_finder.graph_index.taxonomy import prune_device_taxonomy

SEM = corpora.NARAD_SEM

#: The keys one class row carries. The first six are the ones the ontology
#: route already answers with; ``direct`` travels alongside them so a parity
#: check can compare the count the store reports rather than only its sign.
CLASS_KEYS = {"uri", "name", "altLabel", "parents", "rollup", "abstract", "direct"}

#: The badge counts, named as the statistics route already names them.
BADGE_KEYS = {
    "total_devices",
    "total_channels",
    "total_classes",
    "total_signals",
    "total_sections",
}


def _meta(parsed: ParsedCorpus, **overrides: object) -> dict:
    """The ``meta`` mapping a corpus build states for *parsed*."""
    values = {
        "corpus_sha256": "c" * 64,
        "corpus_filename": "demo_machine.ttl",
        "binding_count": len(parsed.binding_rows),
        "device_count": len({row.device_uri for row in parsed.binding_rows}),
        "class_count": len(parsed.class_rows),
        "signal_count": parsed.signal_count,
        "section_count": len(parsed.section_codes),
    }
    values.update(overrides)
    assert set(values) == set(CALLER_META_KEYS)
    return values


def _open(text: str, index_path: Path, **overrides: object) -> GraphIndex:
    """Build an index over *text* and open it read-only."""
    parsed = parse_corpus(text)
    build_from_rows(
        parsed.binding_rows,
        parsed.class_rows,
        channels_from_rows(parsed.binding_rows),
        index_path,
        _meta(parsed, **overrides),
    )
    index = open_graph_index(index_path)
    assert isinstance(index, GraphIndex), index
    return index


@pytest.fixture
def chain_index(tmp_path: Path) -> GraphIndex:
    """One quadrupole under ``Quadrupole ⊂ Magnet ⊂ AcceleratorDevice``."""
    with _open(corpora.SUBCLASS_CHAIN, tmp_path / "graph.duckdb") as index:
        yield index


@pytest.fixture
def empty_index(tmp_path: Path) -> GraphIndex:
    """An index over a corpus that declares an ontology and binds nothing."""
    with _open(corpora.NO_BINDINGS, tmp_path / "graph.duckdb") as index:
        yield index


class TestOntologyShape:
    def test_the_answer_carries_the_keys_the_rail_reads(self, chain_index: GraphIndex):
        payload = chain_index.ontology()

        assert set(payload) == {
            "classes",
            "relationship_types",
            "truncated",
            "empty",
            "suggestions",
        }
        assert all(set(entry) == CLASS_KEYS for entry in payload["classes"])

    def test_the_vocabulary_is_not_derived_and_a_scan_cannot_truncate(
        self, chain_index: GraphIndex
    ):
        payload = chain_index.ontology()

        assert payload["relationship_types"] == []
        assert payload["truncated"] is False

    def test_the_chain_is_answered_whole_and_ordered_by_name(self, chain_index: GraphIndex):
        classes = chain_index.ontology()["classes"]

        assert [entry["name"] for entry in classes] == [
            "AcceleratorDevice",
            "Magnet",
            "Quadrupole",
        ]

    def test_a_class_carries_its_name_labels_parents_and_rollup(self, chain_index: GraphIndex):
        classes = {entry["name"]: entry for entry in chain_index.ontology()["classes"]}

        assert classes["Quadrupole"] == {
            "uri": f"{SEM}Quadrupole",
            "name": "Quadrupole",
            "altLabel": ["Focusing Magnet", "quad", "quadrupole"],
            "parents": [f"{SEM}Magnet"],
            "rollup": 1,
            "abstract": False,
            "direct": 1,
        }
        assert classes["Magnet"]["parents"] == [f"{SEM}AcceleratorDevice"]
        assert classes["Magnet"]["altLabel"] == ["magnet"]
        assert classes["AcceleratorDevice"]["parents"] == []

    def test_a_class_nothing_is_typed_as_is_abstract(self, chain_index: GraphIndex):
        classes = {entry["name"]: entry for entry in chain_index.ontology()["classes"]}

        # Nothing is typed a Magnet, yet one device rolls up to it.
        assert classes["Magnet"]["abstract"] is True
        assert classes["Magnet"]["direct"] == 0
        assert classes["Magnet"]["rollup"] == 1
        # The quadrupole is typed directly, so it is a kind of device.
        assert classes["Quadrupole"]["abstract"] is False


class TestTaxonomyIsAlreadyPruned:
    def test_a_class_holding_nothing_is_not_in_the_tree(self, chain_index: GraphIndex):
        # Sextupole is declared and nothing is typed as it, and no class names
        # it a parent, so the build dropped it.
        names = [entry["name"] for entry in chain_index.ontology()["classes"]]

        assert "Sextupole" not in names
        assert "Thing" not in names

    def test_pruning_the_answer_again_would_change_nothing(self, chain_index: GraphIndex):
        classes = chain_index.ontology()["classes"]

        pruned = prune_device_taxonomy(classes)

        assert pruned == [
            {key: value for key, value in entry.items() if key != "direct"} for entry in classes
        ]


class TestStatistics:
    def test_the_badges_are_the_meta_row_the_build_wrote(self, chain_index: GraphIndex):
        meta = chain_index.meta

        assert chain_index.statistics() == {
            "total_devices": meta.device_count,
            "total_channels": meta.binding_count,
            "total_classes": meta.class_count,
            "total_signals": meta.signal_count,
            "total_sections": meta.section_count,
        }

    def test_the_badges_are_named_as_the_statistics_route_names_them(self, chain_index: GraphIndex):
        assert set(chain_index.statistics()) == BADGE_KEYS

    def test_the_counts_are_the_populations_the_corpus_states(self, chain_index: GraphIndex):
        assert chain_index.statistics() == {
            "total_devices": 1,
            "total_channels": 3,
            "total_classes": 3,
            "total_signals": 2,
            "total_sections": 1,
        }

    def test_the_class_badge_counts_the_tree_the_rail_draws(self, chain_index: GraphIndex):
        assert chain_index.statistics()["total_classes"] == len(chain_index.ontology()["classes"])


class TestEmptyIndex:
    def test_an_index_that_binds_nothing_says_so_and_names_the_corpus(
        self, empty_index: GraphIndex
    ):
        payload = empty_index.ontology()

        assert payload["empty"] is True
        (suggestion,) = payload["suggestions"]
        assert "demo_machine.ttl" in suggestion
        assert "osprey knowledge build-ttl" in suggestion

    def test_the_ontology_it_does_declare_is_still_answered(self, empty_index: GraphIndex):
        # The corpus declares a class tree and binds no device to it. The tree
        # is not evidence of channels, so it is drawn beside the remedy rather
        # than instead of it.
        classes = empty_index.ontology()["classes"]

        assert [entry["name"] for entry in classes] == ["AcceleratorDevice", "Magnet"]
        assert all(entry["rollup"] == 0 for entry in classes)
        assert all(entry["abstract"] is True for entry in classes)

    def test_the_badges_of_an_empty_index_are_zeros_not_an_absence(self, empty_index: GraphIndex):
        assert empty_index.statistics() == {
            "total_devices": 0,
            "total_channels": 0,
            "total_classes": 2,
            "total_signals": 2,
            "total_sections": 0,
        }

    def test_an_index_that_binds_something_carries_no_suggestion(self, chain_index: GraphIndex):
        payload = chain_index.ontology()

        assert payload["empty"] is False
        assert payload["suggestions"] == []


class TestClosedIndex:
    def test_an_ontology_read_on_a_closed_index_is_refused(self, tmp_path: Path):
        index = _open(corpora.SUBCLASS_CHAIN, tmp_path / "graph.duckdb")
        index.close()

        with pytest.raises(RuntimeError, match="closed"):
            index.ontology()

    def test_the_badges_survive_a_close(self, tmp_path: Path):
        # They are the meta row this process already read; nothing is queried.
        index = _open(corpora.SUBCLASS_CHAIN, tmp_path / "graph.duckdb")
        index.close()

        assert index.statistics()["total_channels"] == 3
