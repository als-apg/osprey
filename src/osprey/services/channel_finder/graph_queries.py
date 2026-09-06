"""Cypher the graph paradigm's explorer and census are read with.

The census is asked by two callers that share nothing else: the benchmark
runner, which opens its own driver against a project directory it was handed,
and the web explorer, which reads the running app's store context. Both must
count the same population or the number an operator reads in the browser and
the number a benchmark reports would quietly disagree.

The device read serves the explorer's graph-mode finder. It lives here rather
than in the web route so the query text has one home that a test can import
without starting the app. The finder's search itself is no longer answered by
a scan-everything Cypher query — the explorer reads a DuckDB search index
built ahead of time — so the faceted-search query and its ontology and census
siblings that used to live alongside the explorer's routes now live only as
the reference implementation a parity lane checks the index against, in
``tests/integration/_graph_oracles.py``.

This module is deliberately dependency-free — constants and their docstrings,
no imports at all. The explorer reaches it from
``osprey.interfaces.channel_finder.database_api``, which is imported whenever
the web app starts; pulling the benchmark runner in for one string dragged the
whole agent SDK into that import closure.

Store shape every query here assumes (n10s-imported, Neo4j 5.26 community, no
APOC): ``(:Resource)-[:HASBINDING]->(:ChannelBinding)``, a device typed by
``(:Resource)-[:TYPE]->(:Class)`` with ``:Class`` nodes chained by
``[:SUBCLASSOF]``, and a binding reaching its meaning through
``(:ChannelBinding)-[:READSSIGNAL|WRITESSIGNAL]->(:SemanticSignal)``. Devices
carry ``uri``, ``sourceName``, ``sectionCode``, ``system``, ``rawType``,
``sPositionM``, ``ordinalInSection``, ``systemDescription``,
``familyDescription`` and ``ringDescription``; bindings carry ``fullPv``,
``description``, ``fieldDescription``, ``subfieldDescription``, ``protocol``
and ``confidence``; signals carry ``uri`` and ``label``; classes carry ``uri``
and a list-valued ``altLabel``. Every node also wears n10s's ``Resource``
label, which is why nothing below matches on node labels.
"""

from __future__ import annotations

#: The graph paradigm's channel census.
#:
#: ``ChannelBinding`` is the graph's channel: one node per address the facility
#: exposes, so counting them answers the same question the file-backed
#: paradigms answer by counting rows in their database.
GRAPH_CHANNEL_COUNT_CYPHER = "MATCH (b:ChannelBinding) RETURN count(b) AS n"

#: One device, with its channels grouped by the signal they carry.
#:
#: Keyed by ``uri`` — the ``device_uri`` a search row carries. Returns no row
#: at all when the store holds no such device, so a caller reads absence from
#: an empty result rather than from a null field.
#:
#: The one row is ``{uri, device, class, classes, rawType, section, system,
#: sPositionM, ordinalInSection, systemDescription, familyDescription,
#: ringDescription, signals}``. ``class`` is the device's own class URI (its
#: ``TYPE`` target, not an ancestor; ``null`` when untyped) and ``classes``
#: every such URI, for a corpus that types a device twice. ``signals`` is a
#: list ordered by signal name of ``{uri, name, bindings}``, each ``bindings``
#: a list ordered by ``fullPv`` of ``{fullPv, description, fieldDescription,
#: subfieldDescription, protocol, confidence, edges}``. A binding with no
#: signal edge lands in a group whose ``uri`` and ``name`` are ``null``. A
#: device with no bindings answers with ``signals: []``.
GRAPH_DEVICE_CYPHER = """
MATCH (d:Resource {uri: $uri})
OPTIONAL MATCH (d)-[:TYPE]->(c:Class)
WITH d, collect(DISTINCT c.uri) AS classes
OPTIONAL MATCH (d)-[:HASBINDING]->(b:ChannelBinding)
OPTIONAL MATCH (b)-[e:READSSIGNAL|WRITESSIGNAL]->(s:SemanticSignal)
WITH d, classes, b, s, collect(DISTINCT type(e)) AS edges
ORDER BY b.fullPv
WITH d, classes, s.uri AS signal_uri,
     coalesce(s.label, last(split(s.uri, '/'))) AS signal_name,
     [x IN collect({fullPv: b.fullPv, description: b.description,
                    fieldDescription: b.fieldDescription,
                    subfieldDescription: b.subfieldDescription,
                    protocol: b.protocol, confidence: b.confidence, edges: edges})
      WHERE x.fullPv IS NOT NULL] AS bindings
ORDER BY signal_name
WITH d, classes,
     [g IN collect({uri: signal_uri, name: signal_name, bindings: bindings})
      WHERE size(g.bindings) > 0] AS signals
RETURN d.uri AS uri,
       d.sourceName AS device,
       head(classes) AS class,
       classes,
       d.rawType AS rawType,
       d.sectionCode AS section,
       d.system AS system,
       d.sPositionM AS sPositionM,
       d.ordinalInSection AS ordinalInSection,
       d.systemDescription AS systemDescription,
       d.familyDescription AS familyDescription,
       d.ringDescription AS ringDescription,
       signals
""".strip()
