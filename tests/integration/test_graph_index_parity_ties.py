"""The search index against the store, on a corpus built out of the awkward cases.

``tests/integration/test_graph_mcp.py`` replays the whole filter matrix over
the shipped demo corpus, which is a well-behaved machine: every address is
bound exactly once, every binding either reads or writes, and every device sits
in a section.  That is the corpus the finder ships against and the one the
parity lane has to prove first — but it is also a corpus that never exercises
the four shapes the index and the store are most likely to answer differently.

So this module builds a second one out of the pieces in
``tests/services/channel_finder/graph_index/corpora.py`` and asks the same
questions of it:

* one address bound under **two devices**, each with its own binding node, one
  reading and one writing — two search rows, one channel;
* one binding carrying **both edges**, which the direction facet counts three
  times, under ``R``, under ``W`` and under ``RW``;
* one binding node hung under **two devices**, which the store matches once per
  device — the case where the channel census and the binding count part;
* a device **placed nowhere**: no section, no system, no name.

The store is the reference here as it is there.  Where the two disagree by
construction rather than by accident — the channel census — the difference is
pinned with the number it has, not hidden.

Skips are loud and only ever about the host: without a reachable Docker daemon
the plugin resolver skips the whole module with its reason.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from osprey.services.channel_finder.graph_queries import GRAPH_CHANNEL_COUNT_CYPHER
from tests._graphdb_container import (
    GRAPHDB_TEST_PASSWORD,
    GRAPHDB_TEST_USERNAME,
    graphdb_store,
)
from tests.integration._graph_oracles import (
    GRAPH_DEVICE_COUNT_CYPHER,
    GRAPH_ONTOLOGY_CYPHER,
    GRAPH_SEARCH_CYPHER,
    GRAPH_SECTION_COUNT_CYPHER,
    GRAPH_SIGNAL_COUNT_CYPHER,
    normalise_rows,
    oracle_search,
    shape_id,
)
from tests.services.channel_finder.graph_index import corpora

logger = logging.getLogger(__name__)

# xdist_group("docker"): this module starts a real container, and the docker
# group is what keeps every such file on one xdist worker.
pytestmark = [pytest.mark.integration, pytest.mark.xdist_group("docker")]


# ---------------------------------------------------------------------------
# The corpus
# ---------------------------------------------------------------------------

#: The prefixes and the class tree every corpus in ``corpora`` opens with.
_HEAD = corpora.PREFIXES + corpora.SHARED_ONTOLOGY


def _body(corpus: str) -> str:
    """The devices and bindings of *corpus*, without its shared preamble."""
    return corpus.removeprefix(_HEAD)


#: Four awkward corpora under one ontology, which is a valid NARAD corpus in
#: its own right: seven ``(device, binding)`` pairs over six devices, five
#: distinct addresses and six binding nodes.
TIES_CORPUS = (
    _HEAD
    + _body(corpora.SHARED_FULL_PV)
    + _body(corpora.BOTH_EDGES)
    + _body(corpora.BINDING_UNDER_TWO_DEVICES)
    + _body(corpora.DEVICE_WITHOUT_SECTION_OR_SYSTEM)
)

#: The address two devices bind, one reading it and one writing it.
SHARED_ADDRESS = "SR:MAG:SHARED:CURRENT"

#: The address of the one binding node that hangs under two devices.
TWICE_BOUND_ADDRESS = "SR:MAG:TWICE:CURRENT"

_SEM = corpora.NARAD_SEM
MAGNET_CLASS_URI = _SEM + "Magnet"
QUADRUPOLE_CLASS_URI = _SEM + "Quadrupole"

#: Every filter shape this corpus can say something about, its values its own.
#:
#: None of them pages: the store orders its hit list by address alone, and this
#: corpus binds two addresses twice, so a page cut at row two could fall either
#: side of a tie and both backends still be right.  Every shape here therefore
#: asks for the whole result, and the paging the index does over a tie is
#: proven against the index's own order instead.
TIE_SHAPES: list[dict[str, Any]] = [
    {},
    {"tokens": ["quad"]},
    {"tokens": ["shared"]},
    {"tokens": ["current"]},
    {"tokens": ["nowhere"]},
    {"sections": ["SR"]},
    {"sections": ["BR"]},
    {"sections": ["SR", "BR"]},
    {"systems": ["MAG"]},
    {"sections": ["SR"], "systems": ["MAG"]},
    {"cls": MAGNET_CLASS_URI},
    {"cls": QUADRUPOLE_CLASS_URI},
    {"signals": ["quad_current_rb"]},
    {"signals": ["quad_current_sp"]},
    {"dirs": ["R"]},
    {"dirs": ["W"]},
    {"dirs": ["RW"]},
    {"dirs": ["none"]},
    {"dirs": ["R", "W"]},
    {"facet_cap": 1},
]


# ---------------------------------------------------------------------------
# The store and the index, both from that corpus
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True, scope="module")
def _requires_a_real_store(graphdb_plugin_dir: Path) -> None:
    """Skip the whole module, once and with its reason, on a host without Docker.

    Three tests here read only the search index. They would otherwise pass on a
    host that cannot run a store at all, reporting a parity lane as green when
    the half of it that proves parity never ran.
    """


@contextmanager
def _session(uri: str) -> Iterator[Any]:
    """A driver session on *uri*, closed with the block."""
    from osprey.services.facility_knowledge.seeder import graph_seeder

    with graph_seeder.open_session(uri, GRAPHDB_TEST_USERNAME, GRAPHDB_TEST_PASSWORD) as session:
        yield session


@pytest.fixture(scope="module")
def ties_store(graphdb_plugin_dir: Path) -> Iterator[str]:
    """A store of its own, seeded with :data:`TIES_CORPUS`.

    Its own container rather than the demo lane's: n10s imports into whatever
    is already there, and a second corpus in the demo store would move every
    count the module beside this one asserts.

    The seeding goes through the real seeder, which is the path ``osprey
    knowledge seed-graph`` takes.
    """
    from osprey.services.facility_knowledge.seeder import graph_seeder

    with graphdb_store(graphdb_plugin_dir) as uri:
        with _session(uri) as session:
            bootstrap = graph_seeder.bootstrap(session)
            assert bootstrap.ok, bootstrap.message
            imported = graph_seeder.import_ttl(session, TIES_CORPUS)
            assert imported.termination_status == graph_seeder.TERMINATION_OK, imported.extra_info
            graph_seeder.write_marker(
                session,
                graph_seeder.ttl_sha256(TIES_CORPUS),
                graph_seeder.parse_direction_source(TIES_CORPUS),
            )
            logger.info(
                f"ties: seeded {imported.triples_loaded} triples, "
                f"{graph_seeder.resource_count(session)} Resource nodes"
            )
        yield uri


@pytest.fixture(scope="module")
def ties_session(ties_store: str) -> Iterator[Any]:
    """One session on the seeded store, held for the module."""
    with _session(ties_store) as session:
        yield session


@pytest.fixture(scope="module")
def ties_index_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The index built from the same corpus text the store was seeded with."""
    from tests._graph_index import build_index_from_ttl

    directory = tmp_path_factory.mktemp("graph-index-ties")
    corpus = directory / "ties.ttl"
    corpus.write_text(TIES_CORPUS, encoding="utf-8")
    return build_index_from_ttl(corpus, index_path=directory / "ties.duckdb")


@pytest.fixture(scope="module")
def ties_index(ties_index_path: Path) -> Iterator[Any]:
    """The opened index, closed with the module."""
    from osprey.services.channel_finder.graph_index import GraphIndexAbsence, open_graph_index

    index = open_graph_index(ties_index_path)
    if isinstance(index, GraphIndexAbsence):
        pytest.fail(f"the tie corpus's index could not be opened: {index.detail}")
    try:
        yield index
    finally:
        index.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _oracle(session: Any) -> Any:
    """A runner for :func:`oracle_search` bound to *session*'s store."""

    def run(params: Mapping[str, Any]) -> Mapping[str, Any]:
        record = session.run(GRAPH_SEARCH_CYPHER, dict(params)).single()
        assert record is not None, "the search must answer in exactly one row"
        return record.data()

    return run


def _count(session: Any, cypher: str) -> int:
    """Run a counting query that returns a single ``n``."""
    record = session.run(cypher).single()
    assert record is not None, cypher
    return int(record["n"])


def _store_taxonomy(session: Any) -> list[dict[str, Any]]:
    """The taxonomy the retired route served: the store's rows, pruned.

    ``direct`` is carried back in from the unpruned rows because the pruning
    spends it on ``abstract`` and drops it, and the index reports both.
    """
    from osprey.services.channel_finder.graph_index import prune_device_taxonomy

    rows = [record.data() for record in session.run(GRAPH_ONTOLOGY_CYPHER)]
    direct = {row["uri"]: int(row.get("direct") or 0) for row in rows}
    return [{**entry, "direct": direct[entry["uri"]]} for entry in prune_device_taxonomy(rows)]


def _normalise_classes(classes: Any) -> list[dict[str, Any]]:
    """One class row's comparable fields, with its two lists in one order."""
    return [
        {
            "uri": entry["uri"],
            "name": entry["name"],
            "altLabel": sorted(entry.get("altLabel") or []),
            "parents": sorted(entry.get("parents") or []),
            "rollup": int(entry["rollup"]),
            "direct": int(entry["direct"]),
            "abstract": bool(entry["abstract"]),
        }
        for entry in classes
    ]


# ---------------------------------------------------------------------------
# The corpus is what it claims to be
# ---------------------------------------------------------------------------


def test_the_tie_corpus_carries_every_case_it_claims(ties_index: Any) -> None:
    """The four cases are all present, and the numbers below rest on them.

    A corpus assembled out of four other corpora could lose one of them to a
    stray edit and the parity assertions would still pass — on a corpus that
    no longer proves anything.
    """
    page = ties_index.search()

    rows = page["rows"]
    assert page["total"] == 7, rows
    assert page["devices"] == 6, rows

    shared = [row for row in rows if row["fullPv"] == SHARED_ADDRESS]
    assert {row["device"] for row in shared} == {"QF3", "QF4"}, shared
    assert {frozenset(row["edges"]) for row in shared} == {
        frozenset({"READSSIGNAL"}),
        frozenset({"WRITESSIGNAL"}),
    }, shared

    twice = [row for row in rows if row["fullPv"] == TWICE_BOUND_ADDRESS]
    assert {row["device"] for row in twice} == {"QF5", "QF6"}, twice

    both = [row for row in rows if set(row["edges"]) == {"READSSIGNAL", "WRITESSIGNAL"}]
    assert len(both) == 2, both
    # Both rows hang under one device: the "bound twice under one class" shape
    # the class facet's distinct-device count exists for.
    assert {row["device"] for row in both} == {"QF2"}, both

    unplaced = [row for row in rows if row["section"] is None]
    assert [row["fullPv"] for row in unplaced] == ["NOWHERE:RB"], unplaced
    assert unplaced[0]["system"] is None, unplaced
    assert unplaced[0]["device"] is None, unplaced


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", TIE_SHAPES, ids=[shape_id(shape) for shape in TIE_SHAPES])
def test_the_index_answers_what_the_store_answers(
    ties_session: Any, ties_index: Any, shape: dict[str, Any]
) -> None:
    """One filter shape over the tie corpus, both backends, the same answer.

    The shapes that matter most: a direction filter over a binding carrying
    both edges has to keep it under ``R``, under ``W`` and under ``RW``; a
    class filter has to roll two devices up through one shared address; and a
    section filter has to leave the unplaced device out of every one of them.
    """
    expected = oracle_search(_oracle(ties_session), shape)
    actual = ties_index.search(**shape)

    assert actual["total"] == expected["total"], shape
    assert actual["devices"] == expected["devices"], shape
    assert normalise_rows(actual["rows"]) == normalise_rows(expected["rows"]), shape
    assert actual["facets"] == expected["facets"], shape
    assert actual["truncated"] == expected["truncated"], shape


def test_the_index_taxonomy_is_the_stores_taxonomy(ties_session: Any, ties_index: Any) -> None:
    """The class tree matches, including the two abstract branches above it.

    ``Sextupole`` is in the corpus's ontology and nothing is typed as it, so
    both backends have to drop it; ``Magnet`` and ``AcceleratorDevice`` hold no
    device directly and have to survive as branches.
    """
    expected = _normalise_classes(_store_taxonomy(ties_session))
    actual = _normalise_classes(ties_index.ontology()["classes"])

    assert actual == expected
    assert [entry["name"] for entry in actual] == ["AcceleratorDevice", "Magnet", "Quadrupole"]


def test_the_index_statistics_are_the_stores_census(ties_session: Any, ties_index: Any) -> None:
    """Four of the five badges match the store, and the fifth parts by one.

    ``total_channels`` counts ``(device, binding)`` pairs, because that is what
    a search row is and what the finder pages through.  The store's census
    counts ``:ChannelBinding`` nodes.  This corpus hangs one binding under two
    devices, so the index counts seven where the store counts six — a
    deliberate difference, pinned here with its cause rather than papered over
    by comparing something else.
    """
    stats = ties_index.statistics()

    assert stats["total_devices"] == _count(ties_session, GRAPH_DEVICE_COUNT_CYPHER)
    assert stats["total_signals"] == _count(ties_session, GRAPH_SIGNAL_COUNT_CYPHER)
    assert stats["total_sections"] == _count(ties_session, GRAPH_SECTION_COUNT_CYPHER)
    assert stats["total_classes"] == len(_store_taxonomy(ties_session))

    nodes = _count(ties_session, GRAPH_CHANNEL_COUNT_CYPHER)
    assert nodes == 6, nodes
    assert stats["total_channels"] == 7, stats
    assert stats["total_channels"] == nodes + 1, (stats["total_channels"], nodes)
    assert stats["total_channels"] == ties_index.search()["total"], stats


# ---------------------------------------------------------------------------
# What the roster reads out of the same index
# ---------------------------------------------------------------------------


def test_the_roster_collapses_the_shared_address_to_one_channel(ties_index_path: Path) -> None:
    """Two devices binding one address are one channel, with no direction.

    The search rows keep the two apart, because an operator looking at the
    graph is looking at devices.  The roster cannot: a channel is an address,
    and an address that one device reads and another writes has no single
    direction to offer, so it is offered as neither rather than as whichever
    binding happened to sort first.
    """
    from osprey.channel_roster.graph import read_graph_roster
    from osprey.channel_roster.records import RosterSource, RosterSourceKind

    result = read_graph_roster(RosterSource(kind=RosterSourceKind.GRAPH, path=ties_index_path))
    directions = {record.address: record.direction for record in result.records}

    assert len(result.records) == 5, result.records
    assert len(directions) == 5, "an address is one channel"
    assert directions[SHARED_ADDRESS] is None, directions
    # The binding under two devices is one address with one direction: both
    # rows say the same thing about it, so there is nothing to reconcile.
    assert directions[TWICE_BOUND_ADDRESS] == "read", directions
    # And the binding carrying both edges is the other way of having none.
    assert directions["SR:MAG:QF2:CURRENT"] is None, directions


def test_paging_a_tie_is_stable_and_complete(ties_index: Any) -> None:
    """The index's own paging never loses or repeats a row of a tie group.

    This is the assertion the store cannot join in on: it orders by address
    alone, so it may cut a shared address either way.  The index orders by
    address and then device, which is why paging through a corpus with tied
    addresses returns each row exactly once.
    """
    whole = ties_index.search(page_size=50)["rows"]
    paged = [
        row for skip in range(0, 7, 2) for row in ties_index.search(skip=skip, page_size=2)["rows"]
    ]

    assert [(row["fullPv"], row["device_uri"]) for row in paged] == [
        (row["fullPv"], row["device_uri"]) for row in whole
    ]
