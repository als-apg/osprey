"""The relation edges of the graph view answer the design's Appendix B queries.

Each Appendix A corpus — the facility without a policy, the facility with its
``located_in`` precedence policy, and the facility with the readout control
claim the trace needs — is rendered by the graph view and seeded into the real
Neo4j + neosemantics store through ``seeder.import_ttl``, the same path
``osprey up`` takes. The three Appendix B queries then run exactly as the design
writes them:

- **Trip**: the devices whose power chain ends on ``HALL/PANEL-3A``.
- **Rack**: the devices located in rack ``RK08``, and in rack ``RK07``, where
  the two rack claims for ``SR/QF05-PS`` contest each other unless the policy
  settles them.
- **Trace**: the path from ``SR/BPM13`` through readout, control and
  runs-on edges to a ``:Host``.

Every expected answer is derived from the built facility document's claims and
their standings, never from the writer, and is also checked against the answer
the design states, so a derivation that went vacuous fails too. The store must
also carry the uppercased relationship types n10s gives ``poweredBy``,
``locatedInContested`` and ``hasMember``, and the ``:Claim`` and ``:Group``
labels.

A skipped run is not a pass: the store fixture turns a container skip into a
failure that says why the store could not start.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pytest

from osprey.facility.records import FacilityRecords
from tests._graphdb_container import (
    GRAPHDB_TEST_DATABASE,
    GRAPHDB_TEST_PASSWORD,
    GRAPHDB_TEST_USERNAME,
    WatchedSession,
    WatchedStore,
    wiped_graphdb_store,
)
from tests.facility._appendix_a import appendix_a_tree, build_appendix_a, with_readout_control

# xdist_group("docker"): this module seeds the session's shared container, and
# the docker group keeps every such module on one worker.
pytestmark = [pytest.mark.xdist_group("docker")]

#: What the store is called in its failure messages.
STORE_LABEL = "graphdb (neo4j + n10s) seeded with the Appendix A relations"

# ---------------------------------------------------------------------------
# The Appendix B queries, as the design writes them
# ---------------------------------------------------------------------------

TRIP_QUERY = "MATCH (d)-[:POWEREDBY*1..]->(s {sourceName: 'HALL/PANEL-3A'}) RETURN d.sourceName"
RACK_QUERY = "MATCH (d)-[:LOCATEDIN]->(r {placePath: 'HALL/ROW3/RK08'}) RETURN d.sourceName"
#: The rack query for the other rack: the design's text with the rack swapped.
RACK_RK07_QUERY = RACK_QUERY.replace("HALL/ROW3/RK08", "HALL/ROW3/RK07")
TRACE_QUERY = (
    "MATCH p = (b {sourceName: 'SR/BPM13'})"
    "-[:READOUTBY|CONTROLLEDBY|RUNSON*1..4]->(h:Host) RETURN p"
)

TRIP_SUPPLY = "HALL/PANEL-3A"
RACK_RK08 = "HALL/ROW3/RK08"
RACK_RK07 = "HALL/ROW3/RK07"
TRACE_START = "SR/BPM13"
TRACE_TYPES = ("read_out_by", "controlled_by", "runs_on")
TRACE_MAX_HOPS = 4
HOST_CLASS = "Host"

#: The seeded corpora: (Appendix A variant, whether the readout control claim is added).
CORPORA: dict[str, tuple[str, bool]] = {
    "no_policy": ("no_policy", False),
    "located_in_precedence": ("located_in_precedence", False),
    "with_readout_control": ("no_policy", True),
}

TYPES_CYPHER = "MATCH ()-[r]->() RETURN DISTINCT type(r) AS t"
CLAIMS_CYPHER = "MATCH (c:Claim) RETURN c.claimId AS id"
GROUP_MEMBERS_CYPHER = (
    "MATCH (g:Group) OPTIONAL MATCH (g)-[:HASMEMBER]->(m) RETURN g.groupId AS g, m.sourceName AS m"
)


# ---------------------------------------------------------------------------
# Expectations from the facility document
# ---------------------------------------------------------------------------


def _accepted(document: Mapping[str, Any], relation_type: str) -> list[tuple[str, str]]:
    """The (subject id, object id) of every accepted claim of ``relation_type``."""
    pairs = []
    for claim in document.get("relations") or []:
        if claim.get("type") != relation_type or claim.get("standing") != "accepted":
            continue
        subject = claim["subject"].get("device") or claim["subject"].get("place")
        obj = claim["object"].get("device") or claim["object"].get("place")
        pairs.append((subject, obj))
    return pairs


def expected_trip(document: Mapping[str, Any], supply: str) -> set[str]:
    """Devices with a chain of accepted ``powered_by`` claims ending on ``supply``."""
    edges = _accepted(document, "powered_by")
    found: set[str] = set()
    frontier = {supply}
    while frontier:
        step = {subject for subject, obj in edges if obj in frontier} - found
        found |= step
        frontier = step
    return found


def expected_rack(document: Mapping[str, Any], rack: str) -> set[str]:
    """Devices whose own place is ``rack`` or with an accepted ``located_in`` to it."""
    authored = {device["id"] for device in document["devices"] if device.get("place") == rack}
    claimed = {subject for subject, obj in _accepted(document, "located_in") if obj == rack}
    return authored | claimed


def expected_traces(document: Mapping[str, Any]) -> set[tuple[str, ...]]:
    """Every path of accepted trace claims from the BPM to a host, as node ids."""
    edges = [pair for kind in TRACE_TYPES for pair in _accepted(document, kind)]
    hosts = {d["id"] for d in document["devices"] if d.get("class") == HOST_CLASS}
    paths: set[tuple[str, ...]] = set()

    def walk(path: tuple[str, ...], used: frozenset[int]) -> None:
        if len(path) > 1 and path[-1] in hosts:
            paths.add(path)
        if len(path) - 1 == TRACE_MAX_HOPS:
            return
        for index, (subject, obj) in enumerate(edges):
            if subject == path[-1] and index not in used:
                walk((*path, obj), used | {index})

    walk((TRACE_START,), frozenset())
    return paths


def _has_contested(document: Mapping[str, Any], relation_type: str) -> bool:
    return any(
        claim.get("type") == relation_type and claim.get("standing") == "contested"
        for claim in document.get("relations") or []
    )


def _claim_group_members(document: Mapping[str, Any]) -> set[tuple[str, str]]:
    """(group id, member id) of every group that is an end of a claim."""
    named = {
        end.get("group")
        for claim in document.get("relations") or []
        for end in (claim.get("subject") or {}, claim.get("object") or {})
        if end.get("group")
    }
    return {
        (group["id"], member)
        for group in document.get("groups") or []
        if group["id"] in named
        for member in group.get("members") or []
    }


# ---------------------------------------------------------------------------
# The seeded store
# ---------------------------------------------------------------------------


def _shared_store(request: pytest.FixtureRequest) -> WatchedStore:
    """The session's store; a store that cannot start fails this module, never skips it."""
    try:
        return request.getfixturevalue("graphdb_session_store")
    except pytest.skip.Exception as exc:
        pytest.fail(f"{STORE_LABEL} could not start, and a skipped run is not a pass: {exc}")


def _strings(rows: Iterable[Any], key: str) -> list[str]:
    return [row[key] for row in rows]


def _seed_and_ask(store: WatchedStore, ttl: str) -> dict[str, Any]:
    """Wipe ``store``, seed ``ttl`` and return the answers to every query."""
    from osprey.services.graph_store import seeder as graph_seeder

    store = wiped_graphdb_store(store, label=STORE_LABEL)
    with (
        store.reading(),
        graph_seeder.open_session(
            store.uri,
            GRAPHDB_TEST_USERNAME,
            GRAPHDB_TEST_PASSWORD,
            database=GRAPHDB_TEST_DATABASE,
        ) as raw,
    ):
        boot = graph_seeder.bootstrap(raw)
        assert boot.ok, boot.message
        imported = graph_seeder.import_ttl(raw, ttl)
        assert imported.termination_status == graph_seeder.TERMINATION_OK, imported.extra_info
        session = WatchedSession(raw, store)
        traces = []
        for row in session.records(TRACE_QUERY):
            path = row["p"]
            traces.append(
                {
                    "nodes": tuple(node["sourceName"] for node in path.nodes),
                    "types": tuple(rel.type for rel in path.relationships),
                    "end_labels": frozenset(path.end_node.labels),
                }
            )
        return {
            "trip": _strings(session.records(TRIP_QUERY), "d.sourceName"),
            "rk08": _strings(session.records(RACK_QUERY), "d.sourceName"),
            "rk07": _strings(session.records(RACK_RK07_QUERY), "d.sourceName"),
            "trace": traces,
            "types": set(_strings(session.records(TYPES_CYPHER), "t")),
            "claims": _strings(session.records(CLAIMS_CYPHER), "id"),
            "group_members": {
                (row["g"], row["m"]) for row in session.records(GROUP_MEMBERS_CYPHER)
            },
        }


@pytest.fixture(scope="module")
def seeded(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    """Per corpus: its built document and the store's answers once it alone is seeded.

    The corpora are seeded one after another into the one shared store, each
    after a wipe, and every query runs before the next wipe, so no test
    depends on which corpus the store holds when it runs.
    """
    from osprey.facility.views.graph import graph_text

    store = _shared_store(request)
    out: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for name, (variant, readout) in CORPORA.items():
        tree = appendix_a_tree(variant)  # type: ignore[arg-type]
        if readout:
            tree = with_readout_control(tree)
        root: Path = tmp_path_factory.mktemp(f"appendix-a-{name}") / "data" / "facility"
        document = build_appendix_a(root, tree=tree)
        out[name] = (document, _seed_and_ask(store, graph_text(FacilityRecords(document))))
    return out


# ---------------------------------------------------------------------------
# The tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("corpus", list(CORPORA))
def test_trip_returns_the_power_chain(seeded, corpus: str) -> None:
    document, answers = seeded[corpus]
    expected = expected_trip(document, TRIP_SUPPLY)
    assert expected == {"SR/QF05", "SR/QF05-PS"}, "the derived power chain is not the design's"
    assert sorted(answers["trip"]) == sorted(expected)


@pytest.mark.parametrize("corpus", list(CORPORA))
def test_rack_rk08_returns_its_devices(seeded, corpus: str) -> None:
    document, answers = seeded[corpus]
    expected = expected_rack(document, RACK_RK08)
    assert expected == {"SR/BPM13-EL"}, "the derived RK08 devices are not the design's"
    assert sorted(answers["rk08"]) == sorted(expected)


def test_rack_rk07_is_empty_while_its_claim_is_contested(seeded) -> None:
    document, answers = seeded["no_policy"]
    assert _has_contested(document, "located_in"), "the rack claims should contest each other"
    assert expected_rack(document, RACK_RK07) == set()
    assert answers["rk07"] == []


def test_rack_rk07_holds_the_supply_under_the_policy(seeded) -> None:
    document, answers = seeded["located_in_precedence"]
    expected = expected_rack(document, RACK_RK07)
    assert expected == {"SR/QF05-PS"}, "the policy should accept the rack-layout claim"
    assert sorted(answers["rk07"]) == sorted(expected)


@pytest.mark.parametrize("corpus", ["no_policy", "located_in_precedence"])
def test_trace_finds_no_path_on_the_plain_fixture(seeded, corpus: str) -> None:
    document, answers = seeded[corpus]
    assert expected_traces(document) == set()
    assert answers["trace"] == []


def test_trace_ends_on_a_host_with_readout_control(seeded) -> None:
    document, answers = seeded["with_readout_control"]
    expected = expected_traces(document)
    assert expected == {("SR/BPM13", "SR/BPM13-EL", "CTRL/sr-mag-03", "CTRL/srioc03")}
    assert {trace["nodes"] for trace in answers["trace"]} == expected
    assert len(answers["trace"]) == len(expected)
    for trace in answers["trace"]:
        assert "Host" in trace["end_labels"]
        assert trace["types"] == ("READOUTBY", "CONTROLLEDBY", "RUNSON")


@pytest.mark.parametrize("corpus", list(CORPORA))
def test_relationship_types_are_uppercased(seeded, corpus: str) -> None:
    document, answers = seeded[corpus]
    assert {"POWEREDBY", "HASMEMBER", "LOCATEDIN"} <= answers["types"]
    contested = _has_contested(document, "located_in")
    assert ("LOCATEDINCONTESTED" in answers["types"]) is contested
    if corpus == "no_policy":
        assert contested, "the plain fixture's rack claims should be contested"


@pytest.mark.parametrize("corpus", list(CORPORA))
def test_claims_are_claim_nodes(seeded, corpus: str) -> None:
    document, answers = seeded[corpus]
    claim_ids = [claim["id"] for claim in document["relations"]]
    assert claim_ids, "the fixture should carry relation claims"
    assert sorted(answers["claims"]) == sorted(claim_ids)


@pytest.mark.parametrize("corpus", list(CORPORA))
def test_claimed_groups_are_group_nodes_with_members(seeded, corpus: str) -> None:
    document, answers = seeded[corpus]
    expected = _claim_group_members(document)
    assert {group for group, _ in expected} == {"SR/VS03"}
    assert answers["group_members"] == expected
