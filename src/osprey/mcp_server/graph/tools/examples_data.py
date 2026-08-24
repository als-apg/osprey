"""Curated Cypher examples for the graph MCP server.

Pure data: no imports beyond the standard library, so this module can be read by
the ``example_queries`` tool, by tests and by documentation without touching the
driver or the store.

Every query is written against the shape neosemantics produces from a
NARAD-convention Turtle corpus with ``applyNeo4jNaming`` on, which is what
``osprey knowledge seed-graph`` loads:

* node labels are the RDF class local names — ``:Resource`` on every node,
  plus ``:ChannelBinding``, ``:Class``, and one device-class label per device;
* relationship types are UPPERCASED — ``:HASBINDING``, ``:READSSIGNAL``,
  ``:WRITESSIGNAL``, ``:SUBCLASSOF``, ``:TYPE``;
* property names keep their ``narad_p:`` local spelling — ``.uri``, ``.fullPv``,
  ``.sourceName``, ``.sectionCode``, ``.sPositionM``, ``.ordinalInSection``,
  ``.confidence``.

Those uppercase relationship names belong to the *graph projection* only. The
Turtle corpus itself spells them ``narad_p:hasBinding`` and friends.

The queries are corpus-agnostic: nothing about a particular facility is baked
into the Cypher, only into the parameter values. Each example carries its
parameter sets keyed by corpus; the one shipped corpus is ``"demo"``, the
generated demo machine, so every example runs as-is against a deployment that
seeded it, and on any other corpus the values are read off the rows the
structural queries return.

Every query is bounded: it ends in a ``LIMIT`` even where an aggregate already
guarantees a handful of rows, so no example can be the reason a result comes
back truncated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["EXAMPLE_QUERIES", "ExampleQuery"]

_SEM = "https://narad.example.org/schema/shared_semantics/"

_ACCELERATOR_DEVICE_URI = _SEM + "AcceleratorDevice"
_MAGNET_URI = _SEM + "Magnet"


@dataclass(frozen=True)
class ExampleQuery:
    """One curated, runnable example.

    Attributes:
        key: Stable identifier, used to name the example in tool output and docs.
        title: One-line operator-facing name for what the query answers.
        description: What an operator learns from the result, plus what each
            parameter means and how to vary it.
        cypher: The query, parameterized. Safe to run as written once paired
            with one of the parameter sets below.
        parameters: Parameter sets keyed by corpus. ``"demo"`` maps the
            query's ``$name`` placeholders to values that exist in the
            generated demo machine. An example that takes no parameters
            carries an empty mapping.
    """

    key: str
    title: str
    description: str
    cypher: str
    parameters: dict[str, dict[str, Any]]


_Q1A = ExampleQuery(
    key="q1a",
    title="Device count by concrete class",
    description=(
        "How many devices of each concrete kind the graph holds — Quadrupoles, "
        "BPMs, valves and so on — counted from the class label each device node "
        "carries. Run this first on an unfamiliar corpus: it is the fastest "
        "picture of what the facility is made of, and the row count tells you "
        "how many distinct device classes are in play. Only devices that have at "
        "least one channel binding are counted, so the total matches the number "
        "of devices an operator can actually address. Takes no parameters."
    ),
    cypher="""
MATCH (d:Resource)-[:HASBINDING]->(:ChannelBinding)
WITH DISTINCT d
RETURN [l IN labels(d) WHERE l <> "Resource"][0] AS device_class,
       count(d) AS device_count
ORDER BY device_count DESC, device_class
LIMIT 100
""".strip(),
    parameters={"demo": {}},
)

_Q1B = ExampleQuery(
    key="q1b",
    title="Device count rolled up to top-level branches",
    description=(
        "The same census as q1a, but aggregated up the class hierarchy to the "
        "branches directly under a root class — magnets, instrumentation, "
        "vacuum, RF. This is the query that proves the ontology is traversable "
        "rather than merely present: a device typed as an HCorrector is counted "
        "under Magnet without the query naming HCorrector anywhere.\n"
        "\n"
        "$root_uri — the class whose immediate children become the branches. "
        "Use the ontology root to see the whole facility; pass a narrower class "
        "to break just that subtree down one level."
    ),
    cypher="""
MATCH (branch_cls:Class)-[:SUBCLASSOF]->(:Class {uri: $root_uri})
MATCH (cls:Class)-[:SUBCLASSOF*0..]->(branch_cls)
MATCH (d:Resource)-[:TYPE]->(cls)
RETURN last(split(branch_cls.uri, "/")) AS branch,
       count(DISTINCT d) AS device_count
ORDER BY device_count DESC, branch
LIMIT 100
""".strip(),
    parameters={"demo": {"root_uri": _ACCELERATOR_DEVICE_URI}},
)

_Q1C = ExampleQuery(
    key="q1c",
    title="Every device of a class, including its subclasses",
    description=(
        "Answers 'give me every magnet' — or every instrument, every vacuum "
        "device — by walking the class hierarchy down from one class and "
        "listing the devices typed as it or any of its descendants. Each row is "
        "a device with its concrete class and the section it sits in. This is "
        "the payoff of having an ontology: the caller names one class and the "
        "graph supplies the subclasses.\n"
        "\n"
        "$class_uri — the class to roll up. Any class URI works; q4b lists the "
        "ones this corpus has."
    ),
    cypher="""
MATCH (cls:Class)-[:SUBCLASSOF*0..]->(:Class {uri: $class_uri})
MATCH (d:Resource)-[:TYPE]->(cls)
RETURN [l IN labels(d) WHERE l <> "Resource"][0] AS device_class,
       d.sectionCode AS section,
       d.sourceName AS device
ORDER BY device_class, section, device
LIMIT 500
""".strip(),
    parameters={"demo": {"class_uri": _MAGNET_URI}},
)

_Q2 = ExampleQuery(
    key="q2",
    title="Walk one section in beamline order",
    description=(
        "Lists the devices in a single section in the order the beam meets "
        "them, which is the sequence operators reason in when they talk about a "
        "region of the machine. Rows are ordered by longitudinal position "
        "(``sPositionM``) where the corpus records it and fall back to the "
        "device's ordinal within the section where it does not, so the ordering "
        "is meaningful on a corpus that records only ordinals.\n"
        "\n"
        "$section — the section code to walk, exactly as the corpus spells it "
        "(a transfer line, a ring, a sector). Values come from the ``section`` "
        "column of q1c."
    ),
    cypher="""
MATCH (d:Resource)
WHERE d.sectionCode = $section
RETURN d.sourceName AS device,
       [l IN labels(d) WHERE l <> "Resource"][0] AS device_class,
       d.sPositionM AS s_m,
       d.ordinalInSection AS ordinal
ORDER BY s_m, ordinal, device
LIMIT 200
""".strip(),
    parameters={"demo": {"section": "SR"}},
)

_Q3 = ExampleQuery(
    key="q3",
    title="Every PV of one device, split by direction",
    description=(
        "The channel-finding query: given a device, return every process "
        "variable bound to it, the semantic signal each one carries, whether it "
        "is read or written, and how confident the binding is. Use it to go "
        "from a device an operator named to the addresses a control-system "
        "connector can actually talk to.\n"
        "\n"
        "$name — the device's source name as the corpus records it.\n"
        "$section — the section that device sits in. Both are needed because a "
        "source name can repeat across sections."
    ),
    cypher="""
MATCH (d:Resource {sourceName: $name, sectionCode: $section})-[:HASBINDING]->(b:ChannelBinding)
OPTIONAL MATCH (b)-[:READSSIGNAL]->(rs)
OPTIONAL MATCH (b)-[:WRITESSIGNAL]->(ws)
RETURN b.fullPv AS pv,
       last(split(coalesce(rs.uri, ws.uri), "/")) AS signal,
       CASE WHEN ws IS NULL THEN "read" ELSE "write" END AS direction,
       b.confidence AS confidence
ORDER BY pv
LIMIT 200
""".strip(),
    parameters={"demo": {"name": "DIPOLE01", "section": "SR"}},
)

_Q4B = ExampleQuery(
    key="q4b",
    title="Class hierarchy as a flat table",
    description=(
        "Every (class, parent) edge in the ontology, one per row, with both the "
        "short names and the full URIs. This is how to find out which classes a "
        "corpus actually defines — take a ``class_uri`` from here into q1c, or "
        "read the table top to bottom to audit the taxonomy. Takes no "
        "parameters."
    ),
    cypher="""
MATCH (sub:Class)-[:SUBCLASSOF]->(parent:Class)
RETURN last(split(sub.uri, "/")) AS subclass,
       last(split(parent.uri, "/")) AS superclass,
       sub.uri AS subclass_uri,
       parent.uri AS superclass_uri
ORDER BY superclass, subclass
LIMIT 500
""".strip(),
    parameters={"demo": {}},
)

_Q4C = ExampleQuery(
    key="q4c",
    title="Inheritance chains from a root class",
    description=(
        "The same taxonomy as q4b, but as whole chains instead of single edges: "
        "each row is the path from one class up to the root, deepest first. It "
        "shows the multi-hop structure a single-edge listing hides — that a "
        "horizontal corrector is a corrector is a magnet — which is what makes "
        "the rollups in q1b and q1c reach as far as they do.\n"
        "\n"
        "$root_uri — the class every chain must end at. Pass the ontology root "
        "for the full taxonomy, or a branch class to see only that subtree."
    ),
    cypher="""
MATCH path = (sub:Class)-[:SUBCLASSOF*1..6]->(:Class {uri: $root_uri})
RETURN [n IN nodes(path) | last(split(n.uri, "/"))] AS chain,
       size(nodes(path)) AS depth
ORDER BY depth DESC, chain
LIMIT 200
""".strip(),
    parameters={"demo": {"root_uri": _ACCELERATOR_DEVICE_URI}},
)

_Q5 = ExampleQuery(
    key="q5",
    title="Read/write split across all bindings",
    description=(
        "One row summarising every channel binding in the corpus: how many are "
        "write-only, how many read-only, how many are both, and the total. It "
        "is the cheapest sanity check that a corpus imported completely and "
        "that its direction facts survived — a readwrite count that should be "
        "zero but is not, or a total that does not match what was seeded, "
        "points at the corpus rather than at the query. Takes no parameters."
    ),
    cypher="""
MATCH (b:ChannelBinding)
WITH b,
     EXISTS { (b)-[:WRITESSIGNAL]->() } AS is_write,
     EXISTS { (b)-[:READSSIGNAL]->() }  AS is_read
RETURN
  sum(CASE WHEN is_write AND NOT is_read THEN 1 ELSE 0 END) AS write_only,
  sum(CASE WHEN is_read  AND NOT is_write THEN 1 ELSE 0 END) AS read_only,
  sum(CASE WHEN is_read  AND is_write     THEN 1 ELSE 0 END) AS readwrite,
  count(b)                                                  AS total
LIMIT 1
""".strip(),
    parameters={"demo": {}},
)

_Q6 = ExampleQuery(
    key="q6",
    title="Devices sharing one PV",
    description=(
        "Which devices are bound to a given process variable. Normally the "
        "answer is one, and the query doubles as a reverse lookup from an "
        "address back to the device that owns it. More than one device on the "
        "row is the interesting case: it means those devices share a single "
        "control endpoint — magnets on a common power supply, for instance — so "
        "writing that address moves all of them at once. The generated demo "
        "corpus has no shared PVs by construction, so ``device_count`` is "
        "always 1 there; a facility corpus built from real wiring can have "
        "them.\n"
        "\n"
        "$pv — the full process-variable address to look up, exactly as the "
        "corpus records it in ``fullPv``. Addresses come from q3."
    ),
    cypher="""
MATCH (d:Resource)-[:HASBINDING]->(b:ChannelBinding {fullPv: $pv})
RETURN b.fullPv AS pv,
       collect(DISTINCT d.sectionCode) AS sections,
       collect(DISTINCT d.sourceName) AS devices,
       count(DISTINCT d) AS device_count
LIMIT 5
""".strip(),
    parameters={"demo": {"pv": "SR:MAG:DIPOLE:01:CURRENT:SP"}},
)


EXAMPLE_QUERIES: tuple[ExampleQuery, ...] = (
    _Q1A,
    _Q1B,
    _Q1C,
    _Q2,
    _Q3,
    _Q4B,
    _Q4C,
    _Q5,
    _Q6,
)
"""The curated set, ordered from broad census to narrow lookup.

The order is the reading order: q1a/q1b/q1c size up the corpus, q2 and q3 walk
into it, q4b/q4c describe its ontology, and q5/q6 answer integrity questions
about the bindings.
"""
