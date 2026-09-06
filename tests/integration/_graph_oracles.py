"""Reference Cypher the graph search index's parity lane checks against.

The graph explorer used to answer search, ontology and census reads by
running these queries directly against the store on every request. Now it
reads a DuckDB search index built ahead of time from the same Turtle corpus,
and these queries survive only as the oracle: a parity lane runs both the
index's ``search`` / ``ontology`` / ``statistics`` reads and these Cypher
queries against a live store seeded from the same corpus, and compares the
answers. If the index and this Cypher ever disagree, the oracle is right and
the index has a bug.

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

from collections.abc import Callable, Iterable, Mapping
from typing import Any

#: The class tree the explorer draws, one row per ``:Class`` node.
#:
#: ``rollup`` counts the devices that fall under a class *including* its
#: subclasses, which is what makes an abstract branch like ``Magnet`` show a
#: number even though nothing is typed directly as one. A "device" is defined
#: exactly as the curated example queries define it — a ``:Resource`` with at
#: least one channel binding — so the counts the explorer shows and the counts
#: the agent computes cannot disagree.
#:
#: ``direct`` counts only the devices typed as the class itself, taken from the
#: same traversal by counting the ones whose subclass *is* the class. It is the
#: difference between a branch and a leaf: a class with a rollup but no direct
#: devices is an abstract grouping, which the rail draws differently.
#:
#: The descent is bounded at ten hops rather than left unbounded: a corpus whose
#: ``SUBCLASSOF`` edges contain a cycle would otherwise walk forever, and no
#: real ontology nests device classes ten deep.
GRAPH_ONTOLOGY_CYPHER = """
MATCH (c:Class)
OPTIONAL MATCH (c)-[:SUBCLASSOF]->(p:Class)
WITH c, collect(DISTINCT p.uri) AS parents
OPTIONAL MATCH (sub:Class)-[:SUBCLASSOF*0..10]->(c)
OPTIONAL MATCH (d:Resource)-[:TYPE]->(sub)
WHERE (d)-[:HASBINDING]->(:ChannelBinding)
WITH c, parents, count(DISTINCT d) AS rollup,
     count(DISTINCT CASE WHEN sub = c THEN d END) AS direct
RETURN c.uri AS uri, c.altLabel AS altLabel, parents, rollup, direct
ORDER BY uri
""".strip()

#: Devices in the store, counted the same way :data:`GRAPH_ONTOLOGY_CYPHER`
#: rolls them up: a bound ``:Resource``. Counted ``DISTINCT`` because a device
#: with several channels binds several times.
GRAPH_DEVICE_COUNT_CYPHER = (
    "MATCH (d:Resource)-[:HASBINDING]->(:ChannelBinding) RETURN count(DISTINCT d) AS n"
)

#: Semantic signals in the store — the readings and settings devices expose,
#: which is a different population from the channels that address them.
GRAPH_SIGNAL_COUNT_CYPHER = "MATCH (s:SemanticSignal) RETURN count(s) AS n"

#: How many sections of the machine the corpus covers. Devices without a
#: section code — anything not placed along the ring — are simply not counted.
GRAPH_SECTION_COUNT_CYPHER = (
    "MATCH (d:Resource) WHERE d.sectionCode IS NOT NULL RETURN count(DISTINCT d.sectionCode) AS n"
)

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


# ---------------------------------------------------------------------------
# Driving the oracle the way the retired route drove it
# ---------------------------------------------------------------------------

#: The five facets, in the order the explorer's rail draws them.
SEARCH_FACETS = ("section", "system", "class", "signal", "dir")

#: The page :data:`GRAPH_SEARCH_CYPHER` slices. Fifty rows is written into the
#: query rather than passed to it, so a parity shape asking for a different
#: page size is answered by running the oracle once per fifty rows.
ORACLE_PAGE_SIZE = 50

#: What the retired route showed in a facet, and what it asked the store for:
#: the cap plus one, so a list that came back at the longer length is a list
#: the store had more values for.
ROUTE_FACET_CAP = 500


def oracle_params(shape: Mapping[str, Any], *, skip: int | None = None) -> dict[str, Any]:
    """Turn one ``GraphIndex.search`` shape into the query's parameters.

    Every parameter is passed on every call, the way the route passed them:
    the query declares no defaults, and a list left out arrives as null, which
    is not the same as the empty list that means "no filter". ``facet_cap``
    carries the route's one-over convention rather than the shape's own cap.

    Args:
        shape: Keyword arguments as ``PARITY_MATRIX`` spells them.
        skip: Page offset to use instead of the shape's own.

    Returns:
        The parameter map :data:`GRAPH_SEARCH_CYPHER` takes.
    """
    return {
        # The route folded case before the query (``q.lower().split()``).
        "tokens": [str(token).lower() for token in shape.get("tokens", [])],
        "sections": list(shape.get("sections", [])),
        "systems": list(shape.get("systems", [])),
        "cls": shape.get("cls"),
        "signals": list(shape.get("signals", [])),
        "dirs": list(shape.get("dirs", [])),
        "skip": int(shape.get("skip", 0)) if skip is None else skip,
        "facet_cap": int(shape.get("facet_cap", ROUTE_FACET_CAP)) + 1,
    }


def oracle_search(
    run: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    shape: Mapping[str, Any],
) -> dict[str, Any]:
    """Answer one search shape from the store, shaped like the route's payload.

    Two things the route did around the query are done here too, because a
    comparison against the raw query row would compare the index against
    something no browser ever received: the facets are asked for one entry
    over the cap and clipped back to it, with the clipping reported as
    ``truncated``, and the page is assembled to the shape's own size. The
    query's page is fifty rows wide, so a wider page is stitched from
    successive fifties and a narrower one is cut out of the first — exact only
    because every offset in the matrix is a whole number of fifties.

    Args:
        run: Runs the query with the given parameters and returns its one row.
        shape: Keyword arguments as ``PARITY_MATRIX`` spells them.

    Returns:
        ``total``, ``devices``, the ``rows`` of the page, the clipped
        ``facets``, and whether any facet was ``truncated``.

    Raises:
        ValueError: If the shape's offset is not a whole number of the query's
            own fifty-row pages.
    """
    page_size = int(shape.get("page_size", ORACLE_PAGE_SIZE))
    facet_cap = int(shape.get("facet_cap", ROUTE_FACET_CAP))
    skip = int(shape.get("skip", 0))
    if skip % ORACLE_PAGE_SIZE:
        raise ValueError(
            f"the oracle pages in {ORACLE_PAGE_SIZE}s, so it cannot start at row {skip}."
        )

    rows: list[Any] = []
    first: Mapping[str, Any] | None = None
    offset = skip
    while len(rows) < page_size:
        payload = run(oracle_params(shape, skip=offset))
        if first is None:
            first = payload
        chunk = list(payload.get("rows") or [])
        rows.extend(chunk)
        if len(chunk) < ORACLE_PAGE_SIZE:
            break
        offset += ORACLE_PAGE_SIZE
    assert first is not None, "a page holds at least one row, so the query ran at least once"

    raw = first.get("facets") or {}
    facets: dict[str, list[dict[str, Any]]] = {}
    truncated = False
    for name in SEARCH_FACETS:
        entries = list(raw.get(name) or [])
        if len(entries) > facet_cap:
            truncated = True
            entries = entries[:facet_cap]
        facets[name] = [
            {"value": entry.get("value"), "count": int(entry.get("count") or 0)}
            for entry in entries
        ]

    return {
        "total": int(first.get("total") or 0),
        "devices": int(first.get("devices") or 0),
        "rows": rows[:page_size],
        "facets": facets,
        "truncated": truncated,
    }


def normalise_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Reduce search rows to what both backends must agree on, in one order.

    ``edges`` and ``signals`` are compared as sets: both are collected per
    binding, and neither backend promises the order it collects them in. The
    rows are ordered by address and then device, which is a total order over a
    page however the two backends broke a tie inside it.

    Args:
        rows: Search rows, from either backend.

    Returns:
        The rows, each carrying its eight comparable fields, sorted.
    """
    normalised = [
        {
            "fullPv": row.get("fullPv"),
            "description": row.get("description"),
            "device": row.get("device"),
            "device_uri": row.get("device_uri"),
            "section": row.get("section"),
            "system": row.get("system"),
            "edges": frozenset(row.get("edges") or ()),
            "signals": frozenset(
                (signal.get("uri"), signal.get("name")) for signal in row.get("signals") or ()
            ),
        }
        for row in rows
    ]
    return sorted(normalised, key=lambda row: (str(row["fullPv"]), str(row["device_uri"])))


def shape_id(shape: Mapping[str, Any]) -> str:
    """Name one filter shape for a test id, e.g. ``sections=SR+skip=50``."""
    if not shape:
        return "unfiltered"
    parts = []
    for key, value in shape.items():
        if isinstance(value, list):
            spelled = ",".join(str(item).rsplit("/", 1)[-1] for item in value)
        else:
            spelled = str(value).rsplit("/", 1)[-1]
        parts.append(f"{key}={spelled}")
    return "+".join(parts)
