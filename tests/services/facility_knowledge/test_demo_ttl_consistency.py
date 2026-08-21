"""The demo graph and the demo channel finder must describe the same machine.

The control-assistant preset ships two descriptions of one demo accelerator: the
tier-3 channel database the channel finder searches, and the Turtle corpus the
graph store is seeded from. The agent reaches for whichever fits the question,
and it has no way to notice when the two disagree — a channel it finds in the
graph but cannot read, or a setpoint the graph calls a readback, looks like a
control-system fault rather than a stale corpus.

Nothing else in the suite compares them. The sibling guard in
``tests/templates/test_control_assistant_demo_ttl.py`` pins where the corpus
lands in a render and counts what is in it; counting catches a corpus that
shrank, not one that drifted sideways. This file pins the *set equalities*
across the artifacts:

- every ``narad_p:fullPv`` in the corpus is a channel the database expands to,
  and every channel the database expands to has a binding (graph ≡ channel
  finder, in both directions);
- every channel the virtual accelerator simulates is documented in the corpus
  (a subset — the VA models a documented slice, not the whole machine);
- the corpus' devices are exactly the devices the PV grammar implies, so a
  device cannot appear in the graph without channels behind it;
- the channels the corpus marks ``narad_p:writesSignal`` are exactly the ones
  ``channel_limits.json`` grants write access to. This is the load-bearing one:
  the limits file is what the write path actually enforces, so a corpus that
  disagrees would have the agent proposing writes the connector refuses, or
  worse, describing an enforced setpoint as read-only.

The direction *source* is asserted too. The generator has a grammar fallback
(``:SP`` writes, everything else reads) that happens to agree with the shipped
limits file, so a guard that only compared the two sets would stay green if the
limits file went missing entirely and the fallback silently took over. Asserting
``DirectionSource.LIMITS`` is what tells those two worlds apart.

Finally the corpus is regenerated from its inputs and compared to the committed
file *semantically* — same triples, via ``rdflib.compare.isomorphic``, not the
same bytes. Byte equality would be a guard on the serializer's whitespace and on
whatever pre-commit does to a ``.ttl``; what matters here is that the committed
corpus still says what regenerating it today would say.

``rdflib`` is imported inside the fixtures and tests rather than at module
scope, matching the discipline the runtime modules keep — see
``test_import_isolation.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

#: Repo root — this file sits at ``tests/services/facility_knowledge/``.
REPO_ROOT = Path(__file__).resolve().parents[3]

#: The control-assistant preset's data tree. Every artifact compared here is
#: read from this one directory by an explicit path: the guard is about what
#: ships side by side, so resolving anything through the config would let a
#: misconfigured project quietly compare something else.
DEMO_DATA = REPO_ROOT / "src/osprey/templates/apps/control_assistant/data"

#: Tier-3 hierarchical channel database — the colon-grammar source of truth.
CHANNEL_DB_PATH = DEMO_DATA / "channel_databases/tiers/tier3/hierarchical.json"

#: Channel limits — the write-safety layer the runtime actually enforces.
LIMITS_PATH = DEMO_DATA / "channel_limits.json"

#: Virtual-accelerator / mock machine model.
MACHINE_PATH = DEMO_DATA / "simulation/machine.json"

#: The committed corpus the preset seeds its graph store from.
TTL_PATH = DEMO_DATA / "demo_machine.ttl"

#: NARAD property namespace — the emitter binds it as ``narad_p:``.
NARAD_PROPERTY = "https://narad.example.org/property/"

#: Expected sizes, spelled out rather than derived, so a change to *both* sides
#: at once still trips something.
EXPECTED_CHANNELS = 2908
EXPECTED_WRITABLE = 396

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
    """The tier-3 database expanded to its flat colon-grammar channel map.

    Expansion goes through the loader the channel finder itself uses, so the
    reference set is what the finder would answer with — not a re-derivation of
    the tree that could drift from it.
    """
    from osprey.services.channel_finder.databases.hierarchical import (
        HierarchicalChannelDatabase,
    )

    return HierarchicalChannelDatabase(str(CHANNEL_DB_PATH)).channel_map


@pytest.fixture(scope="module")
def directed_model_and_report(channel_map: dict[str, dict]) -> tuple[Any, Any]:
    """The generator's model, with directions resolved from the limits file."""
    from osprey.services.facility_knowledge.ttl_generator import direction, model

    return direction.resolve_and_assign(model.build_model(channel_map), LIMITS_PATH)


@pytest.fixture(scope="module")
def directed_model(directed_model_and_report: tuple[Any, Any]) -> Any:
    return directed_model_and_report[0]


@pytest.fixture(scope="module")
def committed_graph() -> Any:
    """The committed corpus, parsed once."""
    from rdflib import Graph

    graph = Graph()
    graph.parse(TTL_PATH, format="turtle")
    return graph


@pytest.fixture(scope="module")
def regenerated_graph(directed_model: Any) -> Any:
    """The corpus as generating it from today's inputs would write it."""
    from rdflib import Graph

    from osprey.services.facility_knowledge.ttl_generator import emitter, ontology_map

    graph = Graph()
    graph.parse(
        data=emitter.serialize_turtle(directed_model, ontology_map.load_demo_ontology()),
        format="turtle",
    )
    return graph


@pytest.fixture(scope="module")
def corpus_pvs(committed_graph: Any) -> set[str]:
    """Every ``narad_p:fullPv`` in the committed corpus."""
    return {str(pv) for _, pv in _objects_by_predicate(committed_graph, "fullPv")}


# ---------------------------------------------------------------------------
# graph ≡ channel finder
# ---------------------------------------------------------------------------


def test_demo_ttl_bindings_equal_the_channel_database(
    corpus_pvs: set[str], channel_map: dict[str, dict]
) -> None:
    """Every documented channel has a binding, and every binding a channel.

    Set equality in both directions is the point. A subset in either direction
    is a real defect: a channel with no binding is invisible to a graph search
    the agent was told to trust, and a binding with no channel is a PV the agent
    would offer that the control system does not have.
    """
    database_pvs = set(channel_map)

    assert len(database_pvs) == EXPECTED_CHANNELS, (
        f"The tier-3 database expands to {len(database_pvs)} channels, not "
        f"{EXPECTED_CHANNELS}. If the demo machine really did change, regenerate "
        "the corpus with `osprey knowledge build-ttl` and update this count."
    )
    assert corpus_pvs == database_pvs, (
        "The graph corpus and the channel database describe different machines.\n"
        + _difference_report(corpus_pvs, database_pvs, "the TTL corpus", "the channel database")
    )
    assert len(corpus_pvs) == EXPECTED_CHANNELS


def test_demo_ttl_documents_every_simulated_channel(corpus_pvs: set[str]) -> None:
    """The virtual accelerator simulates a subset of the documented machine.

    Subset, not equality: ``machine.json`` models the channels the demo needs to
    behave physically, which is fewer than the machine documents. What must not
    happen is the reverse — a channel the VA serves that the graph has never
    heard of, which the agent would find by reading and then fail to explain.
    """
    machine = json.loads(MACHINE_PATH.read_text(encoding="utf-8"))
    simulated = set(machine["channels"])

    extras = simulated - corpus_pvs
    assert not extras, (
        f"{len(extras)} channels in {MACHINE_PATH.name} have no binding in the "
        f"graph corpus: {_sample(extras)}"
    )
    assert simulated, "machine.json declares no channels at all"


def test_demo_ttl_devices_match_the_pv_grammar(committed_graph: Any, directed_model: Any) -> None:
    """Corpus devices are exactly the devices the address grammar implies.

    The device layer is derived, not authored: a device exists because channels
    named after it exist. Comparing the corpus against a fresh derivation from
    the same channel map is what keeps a hand-edit — or a half-finished
    regeneration — from leaving a device node with nothing behind it.
    """
    from rdflib import URIRef

    section_code = URIRef(NARAD_PROPERTY + "sectionCode")
    source_names = list(_objects_by_predicate(committed_graph, "sourceName"))

    corpus_iris = {str(subject) for subject, _ in source_names}
    model_iris = {device.iri for device in directed_model.devices}
    assert corpus_iris == model_iris, (
        "The corpus' device nodes are not the ones the PV grammar derives.\n"
        + _difference_report(corpus_iris, model_iris, "the TTL corpus", "the derived model")
    )

    corpus_identities = {
        (str(section), str(name))
        for subject, name in source_names
        for section in committed_graph.objects(subject, section_code)
    }
    model_identities = {
        (device.section_code, device.source_name) for device in directed_model.devices
    }
    assert corpus_identities == model_identities, (
        "Device (sectionCode, sourceName) identities disagree with the grammar.\n"
        + _difference_report(
            {f"{section}/{name}" for section, name in corpus_identities},
            {f"{section}/{name}" for section, name in model_identities},
            "the TTL corpus",
            "the derived model",
        )
    )


# ---------------------------------------------------------------------------
# The corpus and the write-safety layer agree on what is writable
# ---------------------------------------------------------------------------


def test_demo_ttl_writes_exactly_the_limits_writable_set(committed_graph: Any) -> None:
    """``writesSignal`` bindings are exactly the limits file's writable set.

    ``load_writable_set`` reads the file the way the runtime does — merging the
    ``defaults`` block, where the demo machine grants write access by *omitting*
    the key. Re-deriving it here from raw JSON would risk agreeing with the
    corpus about the wrong thing.
    """
    from osprey.services.facility_knowledge.ttl_generator.direction import load_writable_set

    writable = set(load_writable_set(LIMITS_PATH))
    written = _pvs_of_bindings_with(committed_graph, "writesSignal")

    assert len(writable) == EXPECTED_WRITABLE, (
        f"{LIMITS_PATH.name} grants write access to {len(writable)} channels, not "
        f"{EXPECTED_WRITABLE}."
    )
    assert written == writable, (
        "The graph describes different channels as written than the write-safety "
        "layer permits.\n"
        + _difference_report(written, writable, "narad_p:writesSignal", "channel_limits.json")
    )
    assert all(pv.endswith(":SP") for pv in written), (
        "A written binding is not a setpoint: "
        f"{_sample({pv for pv in written if not pv.endswith(':SP')})}"
    )


def test_demo_ttl_reads_everything_the_limits_file_withholds(
    committed_graph: Any, corpus_pvs: set[str]
) -> None:
    """The read set is the complement — no channel is both, and none is neither."""
    from osprey.services.facility_knowledge.ttl_generator.direction import load_writable_set

    writable = set(load_writable_set(LIMITS_PATH))
    read = _pvs_of_bindings_with(committed_graph, "readsSignal")
    written = _pvs_of_bindings_with(committed_graph, "writesSignal")

    assert not (read & written), f"Bindings carrying both directions: {_sample(read & written)}"
    assert read | written == corpus_pvs, (
        f"Bindings with no direction at all: {_sample(corpus_pvs - (read | written))}"
    )
    assert read == corpus_pvs - writable


def test_demo_ttl_direction_came_from_the_limits_file(
    directed_model_and_report: tuple[Any, Any],
) -> None:
    """The generator read the limits file, not its grammar fallback.

    On the demo machine the two agree exactly, so every set assertion above
    would stay green if the limits file vanished and the ``:SP`` heuristic took
    over. This is the assertion that can tell the difference.
    """
    from osprey.services.facility_knowledge.ttl_generator.direction import DirectionSource

    _, report = directed_model_and_report
    assert report.source is DirectionSource.LIMITS, (
        f"Directions came from {report.source!r}, not the limits file: {report.message}"
    )
    assert report.limits_path == LIMITS_PATH


def test_demo_ttl_generator_refuses_a_mixed_direction_group(tmp_path: Path) -> None:
    """A signal group whose members disagree is refused, not averaged.

    One ``(FAMILY, FIELD, SUBFIELD)`` group mints one ``SemanticSignal``, so its
    members must share a direction. If a future limits file made one member of a
    group read-only, silently picking a direction would publish a graph that
    contradicts the write path for the other members — the exact failure the
    cross-artifact assertions above exist to catch, only invisible because the
    corpus and the limits file would still be *self*-consistent.
    """
    from osprey.services.facility_knowledge.ttl_generator import direction, model

    addresses = [f"SR:MAG:DIPOLE:{n:02d}:CURRENT:SP" for n in range(1, 5)]
    dissenter = addresses[-1]

    limits_path = tmp_path / "mixed_limits.json"
    limits_path.write_text(
        json.dumps(
            {
                "defaults": {"writable": True},
                **{address: {} for address in addresses[:-1]},
                dissenter: {"writable": False},
            }
        ),
        encoding="utf-8",
    )

    synthetic = model.build_model({address: {} for address in addresses})

    with pytest.raises(direction.DirectionConflictError) as excinfo:
        direction.resolve_and_assign(synthetic, limits_path)

    error = excinfo.value
    assert error.group_key == ("DIPOLE", "CURRENT", "SP")
    assert error.dissenting == (dissenter,)
    message = str(error)
    assert "(DIPOLE, CURRENT, SP)" in message
    assert dissenter in message


# ---------------------------------------------------------------------------
# The committed corpus is what regenerating it would produce
# ---------------------------------------------------------------------------


def test_demo_ttl_regenerates_to_the_same_triple_count(
    committed_graph: Any, regenerated_graph: Any
) -> None:
    """A cheap first read on drift, and a useful one when isomorphism fails."""
    assert len(committed_graph) == len(regenerated_graph), (
        f"The committed corpus has {len(committed_graph)} triples; regenerating it "
        f"from today's inputs gives {len(regenerated_graph)}. Re-run "
        "`osprey knowledge build-ttl` and commit the result."
    )


def test_demo_ttl_regeneration_is_isomorphic_to_the_committed_corpus(
    committed_graph: Any, regenerated_graph: Any
) -> None:
    """Semantic equality, deliberately not byte equality.

    The corpus is a committed build artifact, and the thing worth guarding is
    that it still means what its inputs say — not that the serializer emits the
    same whitespace, nor that pre-commit left the file alone.
    """
    from rdflib.compare import graph_diff, isomorphic

    if not isomorphic(committed_graph, regenerated_graph):
        _, only_committed, only_regenerated = graph_diff(committed_graph, regenerated_graph)
        pytest.fail(
            "The committed corpus is not what regenerating it produces.\n"
            f"{len(only_committed)} triples only in the committed file, "
            f"{len(only_regenerated)} only in the regenerated one.\n"
            "Committed-only sample: "
            + "\n  ".join(f"{s} {p} {o}" for s, p, o in list(only_committed)[:5])
            + "\nRegenerated-only sample: "
            + "\n  ".join(f"{s} {p} {o}" for s, p, o in list(only_regenerated)[:5])
        )
