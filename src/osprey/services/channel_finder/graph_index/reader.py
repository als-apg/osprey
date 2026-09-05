"""Opening the flat search index a build wrote, or saying why it cannot open.

A deployment reads this file; it never writes one. ``osprey build`` derives the
index from the Turtle corpus, and everything the explorer asks on a click —
search, the ontology rail, the badge counts and the channel roster — is a scan
over the rows :mod:`.schema` declares.

The index is a file, so its absence is ordinary and is answered rather than
raised: :func:`open_graph_index` returns a :class:`GraphIndexAbsence` for a
build that has not run, a file the driver cannot open, and an index written by
a different schema version. Callers turn that into the sentence their surface
shows — a 503 on the search routes, a roster absence, a health-check row —
without a ``try`` around the open.

``duckdb`` is imported inside :func:`open_graph_index`. The roster and the
health check import this package on paths where dragging the driver in would be
a regression, and a subprocess test in this package's test module pins it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from .schema import META_KEYS, SCHEMA_VERSION

if TYPE_CHECKING:  # pragma: no cover - typing only; duckdb stays a lazy import
    from types import TracebackType

    import duckdb

#: Why an index could not be opened. ``missing`` is a build that has not run;
#: ``unreadable`` is anything the driver refused, including a path that is not a
#: DuckDB file at all and a DuckDB file without our tables; ``schema_mismatch``
#: is an index this build's reader would misread.
AbsenceReason = Literal["missing", "unreadable", "schema_mismatch"]

#: The ``meta`` row, selected by name in :data:`~.schema.META_KEYS` order.
_META_SELECT = f"SELECT {', '.join(META_KEYS)} FROM meta"

#: Command that regenerates the Turtle corpus an index is built from, named in
#: the suggestion an index that binds nothing carries. Spelled here rather than
#: imported from :mod:`osprey.deployment.graphdb_service`: the roster and the
#: health check import this module on paths where pulling the deployment
#: package into the process would be the regression.
BUILD_TTL_COMMAND = "osprey knowledge build-ttl"

#: The two edge types a binding reaches its signal through, as n10s imports the
#: corpus's predicates and as the ``edges`` column spells them.
EDGE_READS = "READSSIGNAL"
EDGE_WRITES = "WRITESSIGNAL"

#: Rows one page of the finder holds. THE page width: the HTTP route and the
#: agent's keyword tool both read it here rather than each spelling fifty, so
#: the offset a caller computes and the slice :meth:`GraphIndex.search` cuts
#: cannot disagree. Both import this module already, and it pulls no driver.
DEFAULT_PAGE_SIZE = 50

#: The facets a search answers with, in the order the rail draws them. Named
#: here rather than read off a result, so a facet that counted nothing arrives
#: as an empty list instead of vanishing from the payload and taking the rail's
#: control with it.
SEARCH_FACETS: tuple[str, ...] = ("section", "system", "class", "signal", "dir")

#: The direction values one binding is counted under, derived from its edges.
#: ``R``, ``W``, ``RW`` and ``none`` are the spellings the rail filters on and
#: ``directionOf`` in ``graph-finder-render.js`` derives the pill from; a
#: binding carrying both edges is counted under all three of ``R``, ``W`` and
#: ``RW``, exactly as the store's dir facet unwinds it.
_DIR_VALUES_SQL = f"""
        CASE
            WHEN list_contains(edges, '{EDGE_READS}')
                 AND list_contains(edges, '{EDGE_WRITES}') THEN ['R', 'W', 'RW']
            WHEN list_contains(edges, '{EDGE_READS}') THEN ['R']
            WHEN list_contains(edges, '{EDGE_WRITES}') THEN ['W']
            WHEN len(edges) = 0 THEN ['none']
            ELSE []::VARCHAR[]
        END"""

#: The class tree, ordered the way the taxonomy the explorer draws is ordered.
_CLASSES_SELECT = (
    "SELECT uri, name, alt_labels, parents, rollup_devices, direct_devices "
    "FROM classes ORDER BY name, uri"
)

#: Every match, counted twice: the bindings, and the devices they hang under.
_TOTALS_TAIL = "SELECT count(*), count(DISTINCT device_uri) FROM matched"

#: One page of matches. ``device_uri`` breaks the ``full_pv`` tie a corpus that
#: binds one address under two devices creates, so a page is reproducible.
_ROWS_TAIL = """
SELECT full_pv, description, device_name, device_uri, section, system,
       edges, signal_uris, signal_names
FROM matched
ORDER BY full_pv, device_uri
LIMIT ? OFFSET ?"""

#: One tail per facet, each lifting its own filter and applying the other four:
#: what the rail shows is what a *second* selection in that facet would add,
#: which a facet counted under its own filter could never say. Section, system
#: and signal count bindings; class counts distinct devices, because a device
#: with forty channels is one device under its class however many rows it has;
#: dir counts a binding under every value that would match it. A NULL value is
#: no value — a device placed nowhere is not a section the rail can offer — so
#: it is dropped rather than drawn as a blank entry.
_FACET_TAILS: dict[str, str] = {
    "section": """
SELECT section AS value, count(*) AS n
FROM base
WHERE m_sys AND m_cls AND m_sig AND m_dir AND section IS NOT NULL
GROUP BY value
ORDER BY n DESC, value ASC
LIMIT ?""",
    "system": """
SELECT system AS value, count(*) AS n
FROM base
WHERE m_sec AND m_cls AND m_sig AND m_dir AND system IS NOT NULL
GROUP BY value
ORDER BY n DESC, value ASC
LIMIT ?""",
    "class": """
SELECT value, count(DISTINCT device_uri) AS n
FROM (
    SELECT unnest(class_uris) AS value, device_uri
    FROM base
    WHERE m_sec AND m_sys AND m_sig AND m_dir
)
WHERE value IS NOT NULL
GROUP BY value
ORDER BY n DESC, value ASC
LIMIT ?""",
    "signal": """
SELECT value, count(*) AS n
FROM (
    SELECT unnest(signal_names) AS value
    FROM base
    WHERE m_sec AND m_sys AND m_cls AND m_dir
)
WHERE value IS NOT NULL
GROUP BY value
ORDER BY n DESC, value ASC
LIMIT ?""",
    "dir": """
SELECT value, count(*) AS n
FROM (
    SELECT unnest(dir_values) AS value
    FROM base
    WHERE m_sec AND m_sys AND m_cls AND m_sig
)
WHERE value IS NOT NULL
GROUP BY value
ORDER BY n DESC, value ASC
LIMIT ?""",
}


def _placeholders(count: int) -> str:
    """Return ``?, ?, ?`` for *count* bound parameters."""
    return ", ".join(["?"] * count)


def _membership(column: str, values: Sequence[str], params: list[Any]) -> str:
    """Return the flag SQL for an OR-within-facet filter on a scalar column.

    A NULL column never matches: ``NULL IN (…)`` is NULL, and a device placed
    in no section is not in the section an operator selected. The ``coalesce``
    says so rather than letting a three-valued flag travel into the facet
    blocks, where a NULL would be counted differently in each.

    Args:
        column: The column to test.
        values: The selected values; empty means no filter at all.
        params: Parameter list, extended in place with *values*.

    Returns:
        A boolean SQL expression, ``TRUE`` when nothing is selected.
    """
    if not values:
        return "TRUE"
    params.extend(values)
    return f"coalesce({column} IN ({_placeholders(len(values))}), FALSE)"


def _overlaps(column: str, values: Sequence[str], params: list[Any]) -> str:
    """Return the flag SQL for an OR-within-facet filter on a list column.

    Args:
        column: The list column to intersect with the selection.
        values: The selected values; empty means no filter at all.
        params: Parameter list, extended in place with *values*.

    Returns:
        A boolean SQL expression, ``TRUE`` when nothing is selected.
    """
    if not values:
        return "TRUE"
    params.extend(values)
    return f"len(list_intersect({column}, list_value({_placeholders(len(values))}))) > 0"


def _search_cte(
    *,
    tokens: Sequence[str],
    sections: Sequence[str],
    systems: Sequence[str],
    cls: str | None,
    signals: Sequence[str],
    dirs: Sequence[str],
) -> tuple[str, list[Any]]:
    """Build the common table expressions every statement of one search reads.

    The token filter is the only one applied as a ``WHERE``: it is the one an
    operator typed, it cuts the population hardest, and no facet lifts it. The
    five facet filters travel as per-row flags instead, because each facet is
    counted with its own flag lifted and the other four applied, and deciding
    them once per row is what lets the seven statements of a search agree on
    the same rows without re-deriving a filter in each.

    Every value is bound, never formatted into the text: a section code and a
    search token are operator input arriving over HTTP.

    Args:
        tokens: Search tokens, ANDed. Lower-cased here, because ``haystack`` is
            written lower-cased and the route folds the query the same way.
        sections: Section codes to keep, empty for no filter.
        systems: System codes to keep, empty for no filter.
        cls: One class URI to keep, or ``None``. It matches a device whose
            rolled-up classes contain it, so a parent class stands for every
            device under it.
        signals: Signal names to keep, empty for no filter.
        dirs: Directions to keep, drawn from ``R``, ``W``, ``RW`` and ``none``.

    Returns:
        The ``WITH`` prefix, ending in a ``matched`` relation, and the
        parameters it binds, in the order the statement reads them.
    """
    params: list[Any] = []
    terms = []
    for token in tokens:
        terms.append("contains(haystack, ?)")
        params.append(token.lower())
    token_where = " AND ".join(terms) if terms else "TRUE"

    m_sec = _membership("section", sections, params)
    m_sys = _membership("system", systems, params)
    if cls is None:
        m_cls = "TRUE"
    else:
        params.append(cls)
        m_cls = "list_contains(class_uris, ?)"
    m_sig = _overlaps("signal_names", signals, params)
    m_dir = _overlaps("dir_values", dirs, params)

    sql = f"""
WITH scanned AS (
    SELECT
        full_pv,
        description,
        device_uri,
        device_name,
        section,
        system,
        coalesce(edges, []::VARCHAR[]) AS edges,
        coalesce(signal_uris, []::VARCHAR[]) AS signal_uris,
        coalesce(signal_names, []::VARCHAR[]) AS signal_names,
        coalesce(class_uris, []::VARCHAR[]) AS class_uris
    FROM bindings
    WHERE {token_where}
),
flagged AS (
    SELECT
        *,{_DIR_VALUES_SQL} AS dir_values,
        {m_sec} AS m_sec,
        {m_sys} AS m_sys,
        {m_cls} AS m_cls,
        {m_sig} AS m_sig
    FROM scanned
),
base AS (
    SELECT *, {m_dir} AS m_dir FROM flagged
),
matched AS (
    SELECT * FROM base WHERE m_sec AND m_sys AND m_cls AND m_sig AND m_dir
)"""
    return sql, params


def _search_row(record: Sequence[Any]) -> dict[str, Any]:
    """Shape one row of :data:`_ROWS_TAIL` into what the result table reads.

    The signal URIs and names are stored as two parallel lists, one column
    each, and are zipped back into the ``{uri, name}`` entries the table and
    the device card both draw. Direction is not a field: the table derives it
    from ``edges``, which is why the pill and the dir facet cannot disagree.

    Args:
        record: One row, in :data:`_ROWS_TAIL` column order.

    Returns:
        The row, keyed as ``graph-finder-render.js`` reads it.
    """
    (
        full_pv,
        description,
        device_name,
        device_uri,
        section,
        system,
        edges,
        signal_uris,
        signal_names,
    ) = record
    return {
        "fullPv": full_pv,
        "description": description,
        "device": device_name,
        "device_uri": device_uri,
        "section": section,
        "system": system,
        "edges": list(edges or []),
        "signals": [
            {"uri": uri, "name": name}
            for uri, name in zip(signal_uris or [], signal_names or [], strict=False)
        ],
    }


@dataclass(slots=True, frozen=True)
class GraphIndexMeta:
    """The index's single ``meta`` row: what it was built from, and how much.

    The counts are the badge numbers the explorer shows and the populations the
    build reported, so a surface reads them here instead of counting rows.
    """

    schema_version: int
    corpus_sha256: str
    corpus_filename: str
    binding_count: int
    device_count: int
    class_count: int
    signal_count: int
    section_count: int


@dataclass(slots=True, frozen=True)
class GraphIndexAbsence:
    """Why :func:`open_graph_index` returned no index.

    Attributes:
        reason: Which of the three absences this is, for callers that branch on
            it (the roster maps ``missing`` and the other two to different
            absence kinds).
        path: Where the index was looked for, always the caller's path.
        detail: One sentence naming the path and, where there is one, the
            driver's own message or the two schema versions. Surfaces show it
            verbatim and add their own remedy.
    """

    reason: AbsenceReason
    path: Path
    detail: str


class GraphIndex:
    """An open, read-only handle on a built index.

    One connection is opened per process and every query takes its own
    :meth:`cursor`. A DuckDB cursor is an independent connection over the same
    database, so concurrent threads — the app runs each query under
    ``asyncio.to_thread`` — may hold one each without serialising on a shared
    one.

    The query methods land on this class as they are written: ``search()``,
    ``ontology()`` and ``statistics()`` each take a fresh cursor, run their SQL
    and shape the payload their route already reads. Nothing here caches a
    result; the file does not change under a running process.
    """

    def __init__(
        self, connection: duckdb.DuckDBPyConnection, meta: GraphIndexMeta, path: Path
    ) -> None:
        """Wrap an open read-only connection. Use :func:`open_graph_index`.

        Args:
            connection: A read-only connection on ``path`` whose ``meta`` row
                has already been read and version-checked.
            meta: That row.
            path: The index file, kept for the messages callers write.
        """
        self._con: duckdb.DuckDBPyConnection | None = connection
        self.meta = meta
        self.path = path

    @property
    def closed(self) -> bool:
        """Whether :meth:`close` has already run."""
        return self._con is None

    def cursor(self) -> duckdb.DuckDBPyConnection:
        """Return an independent cursor on the index, safe to use off-thread.

        Raises:
            RuntimeError: If the index has been closed.
        """
        if self._con is None:
            raise RuntimeError(f"The search index at {self.path} is closed.")
        return self._con.cursor()

    def close(self) -> None:
        """Close the connection. Idempotent, so a lifespan may close twice."""
        connection, self._con = self._con, None
        if connection is not None:
            connection.close()

    def _empty_suggestions(self) -> list[str]:
        """What an operator is told when the index binds no channels at all.

        An index with a ``meta`` row and no bindings is not a broken file: it
        is a corpus that describes an ontology and no devices, so the remedy is
        to regenerate the corpus rather than to touch the deployment. The
        sentence names the file the index was built from, because a project
        with several corpora needs to know which one came back empty.
        """
        return [
            f"The index was built from {self.meta.corpus_filename}, which binds no "
            f"channels. Regenerate the corpus with `{BUILD_TTL_COMMAND}` and build "
            "the index again."
        ]

    def search(
        self,
        *,
        tokens: Sequence[str] = (),
        sections: Sequence[str] = (),
        systems: Sequence[str] = (),
        cls: str | None = None,
        signals: Sequence[str] = (),
        dirs: Sequence[str] = (),
        skip: int = 0,
        page_size: int = DEFAULT_PAGE_SIZE,
        facet_cap: int = 500,
    ) -> dict[str, Any]:
        """Answer one page of the finder, with the facets around it.

        The payload is the one the finder's markup builders read, and the
        semantics are the store's: filters are ORed within a facet and ANDed
        across facets, ``cls`` rolls its subclasses up, and every facet is
        counted with its own filter lifted so the rail says what a second
        selection in it would add.

        Seven statements run on one cursor over the same common table
        expressions — the totals, the page, and one per facet — rather than one
        statement aggregating everything: DuckDB reads the flat ``bindings``
        table once per statement, and seven scans of a few thousand rows cost
        less than the single query that would have to carry every facet's
        grouping through one pipeline.

        Args:
            tokens: Search tokens; a binding matches when its ``haystack``
                contains every one of them. Empty matches everything.
            sections: Section codes to keep, empty for no filter. A device
                without a section is never kept by a section filter.
            systems: System codes to keep, empty for no filter.
            cls: One class URI, or ``None``. A device matches when the class is
                among the ones it rolls up to.
            signals: Signal names to keep, empty for no filter.
            dirs: Directions to keep, drawn from ``R``, ``W``, ``RW`` and
                ``none``; empty for no filter.
            skip: How many matching rows to skip before the page.
            page_size: How many rows the page holds.
            facet_cap: How many entries each facet list carries. One more is
                asked for, so a list that comes back longer is a list the index
                had more values for, which is reported as ``truncated``.

        Returns:
            ``total`` and ``devices`` over every match, the ``page`` of
            ``rows`` and the ``pages`` and ``page_size`` around it, the five
            ``facets``, whether a facet was ``truncated``, and — for an index
            that binds nothing — ``empty`` with the ``suggestions`` naming the
            corpus and the command that regenerates it.

        Raises:
            ValueError: If ``page_size`` is below one or ``skip`` or
                ``facet_cap`` is negative.
            RuntimeError: If the index has been closed.
        """
        if page_size < 1:
            raise ValueError(f"page_size must be at least 1, not {page_size}.")
        if skip < 0:
            raise ValueError(f"skip cannot be negative, not {skip}.")
        if facet_cap < 0:
            raise ValueError(f"facet_cap cannot be negative, not {facet_cap}.")

        cte, params = _search_cte(
            tokens=tokens,
            sections=sections,
            systems=systems,
            cls=cls,
            signals=signals,
            dirs=dirs,
        )

        cursor = self.cursor()
        try:
            totals: Any = cursor.execute(cte + _TOTALS_TAIL, list(params)).fetchone()
            page_rows = cursor.execute(cte + _ROWS_TAIL, [*params, page_size, skip]).fetchall()
            facets: dict[str, list[dict[str, Any]]] = {}
            truncated = False
            for name in SEARCH_FACETS:
                entries = cursor.execute(
                    cte + _FACET_TAILS[name], [*params, facet_cap + 1]
                ).fetchall()
                if len(entries) > facet_cap:
                    truncated = True
                    entries = entries[:facet_cap]
                facets[name] = [{"value": value, "count": int(n)} for value, n in entries]
        finally:
            cursor.close()

        total = int(totals[0]) if totals else 0
        devices = int(totals[1]) if totals else 0
        empty = self.meta.binding_count == 0
        return {
            "total": total,
            "devices": devices,
            "page": skip // page_size + 1,
            # No matches is no pages. The finder clamps a page into [1, pages]
            # with a floor of 1, so a zero here reads as "the one page there is".
            "pages": (total + page_size - 1) // page_size,
            "page_size": page_size,
            "truncated": truncated,
            "rows": [_search_row(record) for record in page_rows],
            "facets": facets,
            "empty": empty,
            "suggestions": self._empty_suggestions() if empty else [],
        }

    def ontology(self) -> dict[str, Any]:
        """Return the device class tree the explorer's rail draws.

        The rows are the pruned taxonomy: the build already dropped the classes
        that are neither held by a device nor an abstract parent of one, so
        nothing is pruned again here and the tree the rail draws is the tree
        the ``class_count`` badge counted.

        ``relationship_types`` is empty: no browser surface reads it, and an
        agent asks the store's vocabulary through ``get_schema`` instead.
        ``truncated`` is always false — a scan has no row cap to hit.

        Returns:
            ``classes``, ``relationship_types``, ``truncated``, ``empty`` and
            ``suggestions``.

        Raises:
            RuntimeError: If the index has been closed.
        """
        cursor = self.cursor()
        try:
            rows = cursor.execute(_CLASSES_SELECT).fetchall()
        finally:
            cursor.close()

        classes = [
            {
                "uri": uri,
                "name": name,
                "altLabel": list(alt_labels or []),
                "parents": list(parents or []),
                "rollup": int(rollup),
                # A class nothing is typed directly as is a grouping rather
                # than a kind of device, whatever its subclasses roll up to it.
                "abstract": int(direct) == 0,
                "direct": int(direct),
            }
            for uri, name, alt_labels, parents, rollup, direct in rows
        ]
        empty = self.meta.binding_count == 0
        return {
            "classes": classes,
            "relationship_types": [],
            "truncated": False,
            "empty": empty,
            "suggestions": self._empty_suggestions() if empty else [],
        }

    def statistics(self) -> dict[str, int]:
        """Return the badge counts, as the ``meta`` row already states them.

        Nothing is counted here. The build wrote these five numbers from the
        corpus it parsed, so the badges, the class tree and the search cannot
        report different populations of the same index.

        Returns:
            ``total_devices``, ``total_channels``, ``total_classes``,
            ``total_signals`` and ``total_sections``.
        """
        meta = self.meta
        return {
            "total_devices": meta.device_count,
            "total_channels": meta.binding_count,
            "total_classes": meta.class_count,
            "total_signals": meta.signal_count,
            "total_sections": meta.section_count,
        }

    def __enter__(self) -> GraphIndex:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def open_graph_index(path: Path) -> GraphIndex | GraphIndexAbsence:
    """Open the index at ``path`` read-only, or say why it could not be opened.

    Args:
        path: Where the index should be, as
            :func:`~osprey.deployment.graphdb_service.resolve_graph_index_path`
            resolved it. It need not exist.

    Returns:
        A :class:`GraphIndex` the caller owns and must :meth:`~GraphIndex.close`,
        or a :class:`GraphIndexAbsence`: ``missing`` when nothing is at ``path``,
        ``unreadable`` when the driver refuses the file or its ``meta`` row, and
        ``schema_mismatch`` when the index was written by a different version of
        :data:`~.schema.SCHEMA_VERSION`.
    """
    if not path.exists():
        return GraphIndexAbsence("missing", path, f"No search index at {path}.")

    import duckdb

    try:
        connection = duckdb.connect(str(path), read_only=True)
    except Exception as exc:  # noqa: BLE001 - any driver refusal is an absence
        return GraphIndexAbsence("unreadable", path, f"Could not open {path}: {exc}")

    try:
        row: Any = connection.execute(_META_SELECT).fetchone()
    except Exception as exc:  # noqa: BLE001 - a file that is not one of our indexes
        connection.close()
        return GraphIndexAbsence(
            "unreadable", path, f"Could not read the meta row of {path}: {exc}"
        )

    if row is None:
        connection.close()
        return GraphIndexAbsence("unreadable", path, f"{path} carries no meta row.")

    meta = GraphIndexMeta(*row)
    if meta.schema_version != SCHEMA_VERSION:
        connection.close()
        return GraphIndexAbsence(
            "schema_mismatch",
            path,
            f"{path} was built for schema version {meta.schema_version}, "
            f"and this build reads version {SCHEMA_VERSION}.",
        )
    return GraphIndex(connection, meta, path)
