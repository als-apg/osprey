"""The demo graph and the demo channel finder must describe the same machine.

The control-assistant preset carries two descriptions of one demo accelerator:
the hierarchical channel database the channel finder searches, and the graph view the
build writes and the graph store is seeded from. The agent reaches for whichever fits the question,
and it has no way to notice when the two disagree — a channel it finds in the
graph but cannot read, or a setpoint the graph calls a readback, looks like a
control-system fault rather than a stale corpus.

Nothing else in the suite compares them. The sibling guard in
``tests/templates/test_control_assistant_demo_ttl.py`` pins where the corpus
counts what is in the view; counting catches a corpus that shrank, not one
that drifted sideways. This file pins the *set equalities* across the
artifacts:

- every ``narad_p:fullPv`` in the corpus is a channel the database expands to
  or one of the addresses the facility serves beyond it, and every channel the
  database expands to has a binding (graph ≡ channel finder, in both
  directions);
- the corpus carries the binding, family and system prose on the right node. The prose is the whole point of searching a graph by meaning: a
  corpus whose bindings carry no description is one the agent can only query
  by address, which is what the channel finder already does better.

``rdflib`` is imported inside the fixtures and tests rather than at module
scope, matching the discipline the runtime modules keep — see
``test_import_isolation.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests._builds import BuiltProject

#: Repo root — this file sits at ``tests/services/facility_knowledge/``.
REPO_ROOT = Path(__file__).resolve().parents[3]

#: The frozen copy of the demo's hierarchical channel database, read by an
#: explicit path: resolving it through the config would let a misconfigured
#: project quietly compare something else.
CHANNEL_DB_PATH = REPO_ROOT / "tests/facility/golden/cf_index_pre_line/hierarchical.json"

#: The addresses the demo facility serves beyond the channel database, as rows
#: of the frozen demo fingerprint's shape.
ADDITIONS_PATH = REPO_ROOT / "tests/facility/golden/demo_fingerprint_additions.json"

#: NARAD property namespace — the emitter binds it as ``narad_p:``.
NARAD_PROPERTY = "https://narad.example.org/property/"

#: Expected sizes, spelled out rather than derived, so a change to *both* sides
#: at once still trips something.
#: ``EXPECTED_CHANNELS`` counts the channel database; the view carries those and
#: ``EXPECTED_BINDINGS`` in all, the four addresses the facility adds included.
EXPECTED_CHANNELS = 2908
EXPECTED_BINDINGS = 2912
EXPECTED_DEVICES = 512

#: Prose predicates on ``narad_sem:ChannelBinding``, one per binding: the
#: channel's own sentence.
BINDING_DESCRIPTION_PREDICATES = ("description",)

#: Prose predicates on every device node: its family's and its system's.
DEVICE_DESCRIPTION_PREDICATES = ("familyDescription", "systemDescription")

#: Distinct texts behind the device predicates — the facility's whole family
#: and system vocabulary. Counting the distinct values, not just the triples,
#: is what says the texts were joined by device rather than broadcast.
EXPECTED_DISTINCT_DEVICE_TEXTS = {
    "familyDescription": 28,
    "systemDescription": 3,
}

#: How many members of a set difference a failure message names before eliding.
_MAX_REPORTED = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample(values: set[str]) -> str:
    """Render up to :data:`_MAX_REPORTED` sorted members for a message."""
    ordered = sorted(values)
    shown = ", ".join(ordered[:_MAX_REPORTED])
    remainder = len(ordered) - _MAX_REPORTED
    return f"{shown} (+{remainder} more)" if remainder > 0 else shown or "<none>"


def _difference_report(left: set[str], right: set[str], left_name: str, right_name: str) -> str:
    """Describe both directions of a set difference for an assertion message."""
    only_left = left - right
    only_right = right - left
    return (
        f"{len(only_left)} only in {left_name}: {_sample(only_left)}\n"
        f"{len(only_right)} only in {right_name}: {_sample(only_right)}"
    )


def _objects_by_predicate(graph: Any, local_name: str) -> Any:
    """``(subject, object)`` pairs for one ``narad_p:`` predicate."""
    from rdflib import URIRef

    return graph.subject_objects(URIRef(NARAD_PROPERTY + local_name))


def _pvs_of_bindings_with(graph: Any, predicate_local_name: str) -> set[str]:
    """Full PVs of every binding carrying *predicate_local_name*.

    The emitter puts exactly one of ``readsSignal`` / ``writesSignal`` on each
    ``ChannelBinding``, so reading the predicate off the binding is exact — no
    need to hop through the signal individual to the group it stands for.
    """
    from rdflib import URIRef

    full_pv = URIRef(NARAD_PROPERTY + "fullPv")
    return {
        str(pv)
        for binding in graph.subjects(URIRef(NARAD_PROPERTY + predicate_local_name), None)
        for pv in graph.objects(binding, full_pv)
    }


# ---------------------------------------------------------------------------
# Fixtures — every expensive input is parsed once for the module
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def channel_map() -> dict[str, dict]:
    """The hierarchical database expanded to its flat colon-grammar channel map.

    Expansion goes through the loader the channel finder itself uses, so the
    reference set is what the finder would answer with — not a re-derivation of
    the tree that could drift from it.
    """
    from osprey.services.channel_finder.databases.hierarchical import (
        HierarchicalChannelDatabase,
    )

    return HierarchicalChannelDatabase(str(CHANNEL_DB_PATH)).channel_map


@pytest.fixture(scope="module")
def committed_graph(built_control_assistant: BuiltProject) -> Any:
    """The graph view the control-assistant build writes, parsed once."""
    from rdflib import Graph

    graph = Graph()
    graph.parse(
        built_control_assistant.build_dir / "data" / "graph" / "facility.ttl", format="turtle"
    )
    return graph


@pytest.fixture(scope="module")
def additions() -> set[str]:
    """The addresses the facility serves beyond the channel database."""
    rows = json.loads(ADDITIONS_PATH.read_text(encoding="utf-8"))["rows"]
    return {row["address"] for row in rows}


@pytest.fixture(scope="module")
def corpus_pvs(committed_graph: Any) -> set[str]:
    """Every ``narad_p:fullPv`` in the view."""
    return {str(pv) for _, pv in _objects_by_predicate(committed_graph, "fullPv")}


# ---------------------------------------------------------------------------
# graph ≡ channel finder
# ---------------------------------------------------------------------------


def test_demo_ttl_bindings_equal_the_channel_database(
    corpus_pvs: set[str], channel_map: dict[str, dict], additions: set[str]
) -> None:
    """Every documented channel has a binding, and every binding a channel.

    The channels are the database's and the addresses the facility adds beyond
    it, named in ``demo_fingerprint_additions.json``.

    Set equality in both directions is the point. A subset in either direction
    is a real defect: a channel with no binding is invisible to a graph search
    the agent was told to trust, and a binding with no channel is a PV the agent
    would offer that the control system does not have.
    """
    database_pvs = set(channel_map)

    assert len(database_pvs) == EXPECTED_CHANNELS, (
        f"The hierarchical database expands to {len(database_pvs)} channels, not "
        f"{EXPECTED_CHANNELS}. If the demo machine really did change, update the "
        "corpus and this count together."
    )
    assert corpus_pvs == database_pvs | additions, (
        "The graph corpus and the channel database describe different machines.\n"
        + _difference_report(
            corpus_pvs, database_pvs | additions, "the graph view", "the channel database"
        )
    )
    assert len(corpus_pvs) == EXPECTED_CHANNELS + len(additions)


# ---------------------------------------------------------------------------
# Every binding carries one direction, except the facility's additions
# ---------------------------------------------------------------------------


def test_demo_ttl_gives_every_channel_binding_exactly_one_direction(
    committed_graph: Any, corpus_pvs: set[str], additions: set[str]
) -> None:
    """No channel is both read and written, and none is neither.

    The addresses the facility adds beyond the channel database sit on a place
    and name no signal, so they are the only bindings with no direction.
    """
    read = _pvs_of_bindings_with(committed_graph, "readsSignal")
    written = _pvs_of_bindings_with(committed_graph, "writesSignal")

    assert not (read & written), f"Bindings carrying both directions: {_sample(read & written)}"
    assert corpus_pvs - (read | written) == additions, (
        f"Bindings with no direction at all: {_sample(corpus_pvs - (read | written))}"
    )


# ---------------------------------------------------------------------------
# The corpus carries the prose both databases hold
# ---------------------------------------------------------------------------


def test_every_binding_carries_its_prose(committed_graph: Any) -> None:
    """Every binding carries its own sentence.

    One value per predicate per binding: neosemantics keeps a single value for a
    property it is not told is multi-valued, so two texts under one predicate
    would import as whichever arrived last.
    """

    for predicate in BINDING_DESCRIPTION_PREDICATES:
        pairs = list(_objects_by_predicate(committed_graph, predicate))
        subjects = {subject for subject, _ in pairs}
        assert len(pairs) == EXPECTED_BINDINGS, (
            f"narad_p:{predicate} appears {len(pairs)} times, not {EXPECTED_BINDINGS}."
        )
        assert len(subjects) == EXPECTED_BINDINGS, (
            f"narad_p:{predicate} carries more than one value on some binding."
        )


def test_every_device_carries_its_family_and_system_prose(committed_graph: Any) -> None:
    """Device prose comes from the device's family and its system.

    The distinct-text counts are the assertion that matters: 28 families and 3
    systems is the facility's vocabulary, and a join that fell back to a single
    default would still put a text on all 512 devices.
    """

    for predicate in DEVICE_DESCRIPTION_PREDICATES:
        pairs = list(_objects_by_predicate(committed_graph, predicate))
        assert len(pairs) == EXPECTED_DEVICES, (
            f"narad_p:{predicate} appears {len(pairs)} times, not once per device."
        )
        assert len({subject for subject, _ in pairs}) == EXPECTED_DEVICES
        distinct = len({str(text) for _, text in pairs})
        assert distinct == EXPECTED_DISTINCT_DEVICE_TEXTS[predicate], (
            f"narad_p:{predicate} has {distinct} distinct texts, not "
            f"{EXPECTED_DISTINCT_DEVICE_TEXTS[predicate]}."
        )


def test_semantic_signals_carry_no_prose(committed_graph: Any) -> None:
    """Signals are deliberately text-free.

    A ``SemanticSignal`` is shared by every channel that reads or writes it, so
    text on one would be one channel's wording standing in for all of them. The
    bindings carry that text instead, and a description turning up here means
    something started guessing.
    """
    from rdflib import RDF, URIRef

    signals = set(
        committed_graph.subjects(
            RDF.type, URIRef("https://narad.example.org/schema/shared_semantics/SemanticSignal")
        )
    )
    assert signals, "The corpus declares no semantic signals at all"

    described = {
        signal
        for signal in signals
        for predicate in BINDING_DESCRIPTION_PREDICATES + DEVICE_DESCRIPTION_PREDICATES
        if (signal, URIRef(NARAD_PROPERTY + predicate), None) in committed_graph
    }
    assert not described, f"{len(described)} semantic signals carry description prose: " + _sample(
        {str(signal) for signal in described}
    )
