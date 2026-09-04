"""Cypher the graph paradigm's explorer and census are read with.

The census is asked by two callers that share nothing else: the benchmark
runner, which opens its own driver against a project directory it was handed,
and the web explorer, which reads the running app's store context. Both must
count the same population or the number an operator reads in the browser and
the number a benchmark reports would quietly disagree.

The search and device reads serve the explorer's graph-mode finder. They live
here rather than in the web route so the query text has one home that a test
can import without starting the app.

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

#: One faceted search over the graph's channels, answered in a single row.
#:
#: Parameters (the route lower-cases the tokens; nothing here does):
#:
#: * ``tokens`` — list of strings; every token must ``CONTAINS``-match one of
#:   the binding's ``fullPv`` or ``description``, its device's ``sourceName``,
#:   one of its signal names, or a name of any class the device rolls up to
#:   (the URI's trailing fragment, the way ``_class_name`` in the explorer
#:   derives it, plus every ``altLabel``). ``[]`` matches everything.
#: * ``sections`` / ``systems`` — lists of ``sectionCode`` / ``system``
#:   values, OR within the list, ``[]`` for no filter.
#: * ``cls`` — one class URI or ``null``; matches a device whose class is that
#:   class or any subclass of it, so a parent class rolls its children up.
#: * ``signals`` — list of signal names (the ``name`` the rows carry), ``[]``
#:   for no filter.
#: * ``dirs`` — list drawn from ``'R'``, ``'W'``, ``'RW'``, ``'none'``: ``R``
#:   matches a binding with a ``READSSIGNAL`` edge, ``W`` one with
#:   ``WRITESSIGNAL``, ``RW`` one with both, ``none`` one with neither. ``[]``
#:   for no filter.
#: * ``skip`` — page offset in rows; the page is fifty rows ordered by
#:   ``fullPv``.
#: * ``facet_cap`` — how many entries each facet list carries.
#:
#: The one row returned is ``{total, devices, rows, facets}``: ``total`` is
#: the number of bindings passing every filter, ``devices`` the distinct
#: devices among them, ``rows`` the page as
#: ``{fullPv, description, device, device_uri, section, system, edges,
#: signals}`` with ``edges`` the edge types present and ``signals`` a list of
#: ``{uri, name}``, and ``facets`` a map ``{section, system, class, signal,
#: dir}`` whose lists hold ``{value, count}`` entries ordered by count
#: descending then value ascending. Each facet is counted with its own filter
#: lifted and every other filter applied, which is what lets an operator see
#: what a second selection in the same facet would add. Section, system and
#: signal count bindings; class counts distinct devices per class URI, rolled
#: up over ancestors; direction counts a binding under every value that would
#: match it. A facet lists only values with a non-zero count — an ``RW`` or
#: ``none`` entry appears only once such a binding exists.
#:
#: Why the shape is what it is: the token predicate and the per-row facet
#: flags are evaluated once per binding, then the survivors are collected and
#: each ``CALL {}`` block re-reads that list for one facet — one pass over the
#: store, however many facets. The facet lists are sorted *after* the blocks,
#: at the top level, rather than inside them: Neo4j's transaction memory
#: tracker charges a sort buffered inside a subquery for the imported list on
#: every buffered row, so a fifty-row ``ORDER BY … LIMIT`` inside a block was
#: booked as fifty copies of the whole hit list and tripped the store's
#: transaction cap on a corpus of three thousand bindings. Sorting the small
#: per-facet lists once the hit list has left scope keeps the charge to a few
#: entries. The page is sliced from the hit list, which was ordered by
#: ``fullPv`` before it was collected, for the same reason.
GRAPH_SEARCH_CYPHER = """
MATCH (d:Resource)-[:HASBINDING]->(:ChannelBinding)
WITH DISTINCT d
OPTIONAL MATCH (d)-[:TYPE]->(:Class)-[:SUBCLASSOF*0..10]->(anc:Class)
WITH d,
     coalesce(collect(DISTINCT anc.uri), []) AS ancestors,
     coalesce(
       [u IN collect(DISTINCT anc.uri) | toLower(last(split(last(split(u, '/')), '#')))]
       + reduce(acc = [], c IN collect(DISTINCT anc) |
                acc + [x IN coalesce(c.altLabel, []) | toLower(x)]),
       []) AS classNames
MATCH (d)-[:HASBINDING]->(b:ChannelBinding)
OPTIONAL MATCH (b)-[e:READSSIGNAL|WRITESSIGNAL]->(s:SemanticSignal)
WITH d, ancestors, classNames, b,
     collect(DISTINCT type(e)) AS edges,
     [x IN collect(DISTINCT {uri: s.uri, name: coalesce(s.label, last(split(s.uri, '/')))})
      WHERE x.uri IS NOT NULL] AS signals
WHERE all(t IN $tokens WHERE
      toLower(coalesce(b.fullPv, '')) CONTAINS t
      OR toLower(coalesce(b.description, '')) CONTAINS t
      OR toLower(coalesce(d.sourceName, '')) CONTAINS t
      OR any(x IN signals WHERE toLower(x.name) CONTAINS t)
      OR any(c IN classNames WHERE c CONTAINS t))
WITH {
       fullPv: b.fullPv,
       description: b.description,
       device: d.sourceName,
       device_uri: d.uri,
       section: d.sectionCode,
       system: d.system,
       edges: edges,
       signals: signals,
       ancestors: ancestors,
       mSec: size($sections) = 0 OR coalesce(d.sectionCode IN $sections, false),
       mSys: size($systems) = 0 OR coalesce(d.system IN $systems, false),
       mCls: $cls IS NULL OR $cls IN ancestors,
       mSig: size($signals) = 0 OR any(x IN signals WHERE x.name IN $signals),
       mDir: size($dirs) = 0 OR any(v IN $dirs WHERE
               (v = 'R' AND 'READSSIGNAL' IN edges)
               OR (v = 'W' AND 'WRITESSIGNAL' IN edges)
               OR (v = 'RW' AND 'READSSIGNAL' IN edges AND 'WRITESSIGNAL' IN edges)
               OR (v = 'none' AND size(edges) = 0))
     } AS h
ORDER BY h.fullPv
WITH collect(h) AS hits
WITH hits, [h IN hits WHERE h.mSec AND h.mSys AND h.mCls AND h.mSig AND h.mDir] AS matched
CALL {
  WITH hits
  UNWIND [h IN hits WHERE h.mSys AND h.mCls AND h.mSig AND h.mDir] AS h
  WITH h.section AS value, count(*) AS n
  WHERE value IS NOT NULL
  RETURN collect({value: value, count: n}) AS sectionRaw
}
CALL {
  WITH hits
  UNWIND [h IN hits WHERE h.mSec AND h.mCls AND h.mSig AND h.mDir] AS h
  WITH h.system AS value, count(*) AS n
  WHERE value IS NOT NULL
  RETURN collect({value: value, count: n}) AS systemRaw
}
CALL {
  WITH hits
  UNWIND [h IN hits WHERE h.mSec AND h.mSys AND h.mSig AND h.mDir] AS h
  UNWIND h.ancestors AS value
  WITH value, count(DISTINCT h.device_uri) AS n
  RETURN collect({value: value, count: n}) AS classRaw
}
CALL {
  WITH hits
  UNWIND [h IN hits WHERE h.mSec AND h.mSys AND h.mCls AND h.mDir] AS h
  UNWIND h.signals AS x
  WITH x.name AS value, count(*) AS n
  RETURN collect({value: value, count: n}) AS signalRaw
}
CALL {
  WITH hits
  UNWIND [h IN hits WHERE h.mSec AND h.mSys AND h.mCls AND h.mSig] AS h
  UNWIND (CASE WHEN 'READSSIGNAL' IN h.edges THEN ['R'] ELSE [] END
          + CASE WHEN 'WRITESSIGNAL' IN h.edges THEN ['W'] ELSE [] END
          + CASE WHEN 'READSSIGNAL' IN h.edges AND 'WRITESSIGNAL' IN h.edges
                 THEN ['RW'] ELSE [] END
          + CASE WHEN size(h.edges) = 0 THEN ['none'] ELSE [] END) AS value
  WITH value, count(*) AS n
  RETURN collect({value: value, count: n}) AS dirRaw
}
CALL {
  WITH matched
  UNWIND matched AS h
  RETURN count(h) AS total, count(DISTINCT h.device_uri) AS devices
}
WITH sectionRaw, systemRaw, classRaw, signalRaw, dirRaw, total, devices,
     [h IN matched[$skip..$skip + 50] |
      {fullPv: h.fullPv, description: h.description, device: h.device,
       device_uri: h.device_uri, section: h.section, system: h.system,
       edges: h.edges, signals: h.signals}] AS rows
UNWIND (CASE WHEN size(dirRaw) = 0 THEN [null] ELSE dirRaw END) AS f
WITH sectionRaw, systemRaw, classRaw, signalRaw, total, devices, rows, f
ORDER BY f.count DESC, f.value ASC LIMIT $facet_cap
WITH sectionRaw, systemRaw, classRaw, signalRaw, total, devices, rows,
     [x IN collect(f) WHERE x IS NOT NULL] AS dirFacet
UNWIND (CASE WHEN size(sectionRaw) = 0 THEN [null] ELSE sectionRaw END) AS f
WITH systemRaw, classRaw, signalRaw, dirFacet, total, devices, rows, f
ORDER BY f.count DESC, f.value ASC LIMIT $facet_cap
WITH systemRaw, classRaw, signalRaw, dirFacet, total, devices, rows,
     [x IN collect(f) WHERE x IS NOT NULL] AS sectionFacet
UNWIND (CASE WHEN size(systemRaw) = 0 THEN [null] ELSE systemRaw END) AS f
WITH classRaw, signalRaw, dirFacet, sectionFacet, total, devices, rows, f
ORDER BY f.count DESC, f.value ASC LIMIT $facet_cap
WITH classRaw, signalRaw, dirFacet, sectionFacet, total, devices, rows,
     [x IN collect(f) WHERE x IS NOT NULL] AS systemFacet
UNWIND (CASE WHEN size(classRaw) = 0 THEN [null] ELSE classRaw END) AS f
WITH signalRaw, dirFacet, sectionFacet, systemFacet, total, devices, rows, f
ORDER BY f.count DESC, f.value ASC LIMIT $facet_cap
WITH signalRaw, dirFacet, sectionFacet, systemFacet, total, devices, rows,
     [x IN collect(f) WHERE x IS NOT NULL] AS classFacet
UNWIND (CASE WHEN size(signalRaw) = 0 THEN [null] ELSE signalRaw END) AS f
WITH dirFacet, sectionFacet, systemFacet, classFacet, total, devices, rows, f
ORDER BY f.count DESC, f.value ASC LIMIT $facet_cap
WITH dirFacet, sectionFacet, systemFacet, classFacet, total, devices, rows,
     [x IN collect(f) WHERE x IS NOT NULL] AS signalFacet
RETURN total, devices, rows,
       {section: sectionFacet, system: systemFacet, class: classFacet,
        signal: signalFacet, dir: dirFacet} AS facets
""".strip()

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
