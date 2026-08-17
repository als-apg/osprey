"""ARIEL hybrid search module, backed by the qmd sidecar.

Answers a search request from the qmd sidecar's ARIEL mirror collection rather
than from Postgres. The sidecar indexes the markdown tree that the ``qmd_export``
enhancement writes — one ``<mirror>/YYYY/MM/<percent-encoded entry_id>.md`` file
per row — so a hit names a file, not a database row, and this module's job is to
turn the one into the other.

Four properties of that job are load-bearing:

**qmd ranks, Postgres is authoritative.** A hit contributes exactly two things:
its position in the ranking and its snippet. Every field the caller sees is
re-read from ``enhanced_entries`` by primary key, so a mirror file that has
drifted from its row cannot put stale text in front of an operator. A hit whose
row is gone is dropped: the index is a copy of the database and is allowed to lag
it.

**The document names itself; the reported path does not.** Hydration reads the
``entry_id`` out of the hit's title, which is the ``# Entry <entry_id>`` heading
the mirror writer put in the document. It deliberately does **not** invert the
path qmd reports, because qmd does not report the filename it indexed: it
slugifies it, mapping both ``%`` and ``_`` to ``-``, collapsing runs and dropping
a leading one. Inverting that is lossy twice over — an id needing any escape
comes back as a key no row has, and ``beam_current_setpoint`` and
``beam-current-setpoint`` collapse onto a single reported path, so inverting it
hydrates one of them to the *other's* row. A wrong entry in front of an operator
is worse than a missing one, so a hit whose title declares no identifier is
dropped rather than guessed at, and a path that disagrees with the title is
logged.

**Post-filters over-fetch.** Date, author and source-system filters cannot be
pushed into the qmd query — the sidecar indexes prose, not columns — so they run
after hydration. Fetching exactly ``max_results`` candidates and then filtering
would starve the result set, so a filtered query asks the sidecar for
:data:`OVERFETCH_FACTOR` times as many hits, capped at
:data:`MAX_FETCH_LIMIT`.

**Reranking is the deployment's choice, not this module's.** qmd's LLM reranker
dominates query latency — roughly 4x — and its cost is near-constant in corpus
size. This is an agent-facing tool with no interactive budget to blow, so the
default is qmd's own (``True``), and a deployment that wants the fast path sets
``search_modules.hybrid.settings.rerank: false``.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from osprey.services.ariel_search.enhancement.qmd_export.writer import entry_id_from_path
from osprey.services.ariel_search.search.base import ParameterDescriptor, SearchToolDescriptor
from osprey.services.qmd import QMDClient, QMDUnavailableError
from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.database.repository import ARIELRepository
    from osprey.services.ariel_search.models import EnhancedLogbookEntry
    from osprey.services.qmd import QMDSearchResult

logger = get_logger("ariel")

#: qmd collection the ARIEL mirror is indexed under. One sidecar serves several
#: corpora side by side and the collection filter is what keeps them apart:
#: without it, OKF concept hits would come back here and fail hydration. Not a
#: config key — it is fixed by the sidecar's rendered index, not chosen per
#: deployment, exactly as ``OKF_COLLECTION`` is on the facility-knowledge side.
ARIEL_COLLECTION = "ariel"

#: Whether the reranker runs when config says nothing. Matches qmd's own default.
#: See the module docstring for why the agent-facing tool keeps the slow path.
DEFAULT_RERANK = True

#: Candidates the reranker considers (qmd's ``candidateLimit``) when config says
#: nothing. Matches qmd's own default; lowering it trades recall for latency.
DEFAULT_CANDIDATE_LIMIT = 40

#: How many extra hits a filtered query asks for, so that post-filtering has
#: something left to return. Four covers a filter that rejects three quarters of
#: the ranking, which is well past what a date or author filter usually does.
OVERFETCH_FACTOR = 4

#: Ceiling on the over-fetch. Reranking cost grows with the candidate set, so an
#: unbounded multiplier would let ``max_results=50`` turn one query into a very
#: slow one for a filter that may reject nothing at all.
MAX_FETCH_LIMIT = 200

#: Prefix the mirror writer puts before the identifier in each document's first
#: heading (``# Entry <entry_id>``), and therefore the prefix qmd reports back in
#: a hit's title. This is a coupling to ``render_entry``'s output rather than a
#: shared constant, because the writer belongs to the export pipeline; the drift
#: is pinned by a test that renders a row and reads its first line.
TITLE_ENTRY_PREFIX = "Entry "

#: Config block the query knobs are read from.
_SETTINGS_PREFIX = "search_modules.hybrid.settings"

# Built once and reused: the client caches its health verdict and its MCP
# session, and a fresh client per query would re-handshake every time.
_client_lock = threading.Lock()
_cached_client: QMDClient | None = None


@dataclass(frozen=True)
class HybridSearchSettings:
    """Query knobs read from ``search_modules.hybrid.settings``.

    Attributes:
        rerank: Whether qmd reranks candidates with its LLM reranker. Defaults
            to :data:`DEFAULT_RERANK`.
        candidate_limit: qmd's ``candidateLimit``. Defaults to
            :data:`DEFAULT_CANDIDATE_LIMIT`; ``None`` leaves qmd's own default
            in place.
        collection: qmd collection the ARIEL mirror is indexed under. Not a
            config key — see :data:`ARIEL_COLLECTION`.
    """

    rerank: bool = DEFAULT_RERANK
    candidate_limit: int | None = DEFAULT_CANDIDATE_LIMIT
    collection: str = ARIEL_COLLECTION

    @classmethod
    def from_ariel_config(cls, config: ARIELConfig | None) -> HybridSearchSettings:
        """Read the module's ``settings`` block, defaults filled in.

        An absent block is the normal case and yields the defaults. A *present*
        key of the wrong type is refused rather than defaulted: silently
        ignoring ``rerank: "false"`` would leave the deployment on the slow path
        it explicitly asked to leave, which is far harder to diagnose than a
        refusal.

        Args:
            config: The loaded ARIEL configuration, or ``None``.

        Returns:
            The resolved settings.

        Raises:
            ValueError: If ``rerank`` is present but not a boolean, or
                ``candidate_limit`` is present but not a positive integer.
        """
        module = config.search_modules.get("hybrid") if config is not None else None
        settings = module.settings if module is not None else None
        if not isinstance(settings, dict):
            return cls()

        rerank = settings.get("rerank", cls.rerank)
        if not isinstance(rerank, bool):
            raise ValueError(f"{_SETTINGS_PREFIX}.rerank must be a boolean, got {rerank!r}")

        candidate_limit = settings.get("candidate_limit", cls.candidate_limit)
        if candidate_limit is not None and (
            not isinstance(candidate_limit, int)
            or isinstance(candidate_limit, bool)
            or candidate_limit < 1
        ):
            raise ValueError(
                f"{_SETTINGS_PREFIX}.candidate_limit must be a positive integer, "
                f"got {candidate_limit!r}"
            )

        return cls(rerank=rerank, candidate_limit=candidate_limit)


def _default_client() -> QMDClient:
    """Return the process-wide client for this deployment's sidecar.

    Built from the project config on first use and reused afterwards, so the
    health verdict and MCP session survive between queries. A deployment with no
    ``services.qmd`` block yields a client that reports itself unavailable
    without ever opening a socket.

    Returns:
        The cached client.
    """
    global _cached_client
    with _client_lock:
        if _cached_client is None:
            from osprey.deployment.qmd_service import resolve_qmd_service_config
            from osprey.utils.workspace import load_osprey_config

            _cached_client = QMDClient(resolve_qmd_service_config(load_osprey_config()))
        return _cached_client


def _resolve_client(client: QMDClient | None) -> tuple[QMDClient, bool]:
    """Pick the client for this query and probe whether it can answer.

    Both halves are blocking I/O and both belong off the event loop, which is
    why they are one function rather than two awaits. :func:`_default_client`
    takes a lock and loads the project config; :meth:`QMDClient.is_available`
    issues a synchronous ``GET /health`` that costs the client's full 30 s
    timeout against a sidecar which accepts the connection and then does not
    answer — an *expected* state while the daemon finishes its startup index
    pass, not an exotic one. Left on the loop, a probe like that stalls every
    other request in the process, including the keyword and semantic searches
    this module's own unavailable-message steers the caller towards.

    Args:
        client: Caller-supplied client, or ``None`` to use the process-wide one.

    Returns:
        The client, and whether it reported itself able to answer.
    """
    qmd = client if client is not None else _default_client()
    return qmd, qmd.is_available()


def _reset_client_cache() -> None:
    """Drop the cached client so the next query rebuilds it from config.

    Exists for tests and for a config reload; production code has no reason to
    call it.
    """
    global _cached_client
    with _client_lock:
        _cached_client = None


async def hybrid_search(
    query: str,
    repository: ARIELRepository,
    config: ARIELConfig,
    *,
    max_results: int = 10,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    author: str | None = None,
    source_system: str | None = None,
    rerank: bool | None = None,
    candidate_limit: int | None = None,
    client: QMDClient | None = None,
    **kwargs: Any,
) -> list[tuple[EnhancedLogbookEntry, float, list[str]]]:
    """Execute a hybrid keyword+semantic search against the qmd sidecar.

    The sidecar ranks; Postgres answers. Hits are decoded back to ``entry_id``s,
    hydrated by primary key, filtered, and returned in qmd's ranking order.

    Args:
        query: Natural-language or keyword query.
        repository: ARIEL database repository, used to hydrate hits.
        config: ARIEL configuration, read for the module's query knobs.
        max_results: Maximum entries to return (default: 10).
        start_date: Keep entries at or after this time.
        end_date: Keep entries at or before this time.
        author: Keep entries whose author contains this text, case-insensitively.
        source_system: Keep entries from this source system (exact match).
        rerank: Per-query override of ``settings.rerank``. ``None`` uses config.
        candidate_limit: Per-query override of ``settings.candidate_limit``.
            ``None`` uses config.
        client: Client for the sidecar. Defaults to the process-wide client
            built from the project's ``services.qmd`` block; tests pass a stub.

    Returns:
        List of ``(entry, score, snippets)`` tuples in descending relevance
        order. ``score`` is qmd's 0-1 relevance, which is an **ordering signal,
        not a calibrated probability** — do not threshold it. ``snippets``
        carries qmd's line-numbered, match-centered excerpt, or is empty when the
        sidecar supplied none.

    Raises:
        QMDUnavailableError: If no sidecar is configured, or the configured one
            is not answering. This is deliberately not an empty result: "search
            is down" and "nothing matched" must not look alike to the agent.
        QMDClientError: If the sidecar was reached but could not answer.
        ValueError: If the module's config block holds a malformed knob.
    """
    if not query.strip():
        return []

    settings = HybridSearchSettings.from_ariel_config(config)
    effective_rerank = settings.rerank if rerank is None else bool(rerank)
    effective_candidates = settings.candidate_limit if candidate_limit is None else candidate_limit

    qmd, available = await asyncio.to_thread(_resolve_client, client)
    if not available:
        raise QMDUnavailableError(
            "no qmd sidecar is configured for this deployment"
            if not qmd.is_configured
            else f"the qmd sidecar at {qmd.base_url} is not answering"
        )

    filters = _Filters(start_date, end_date, author, source_system)
    fetch_limit = _fetch_limit(max_results, filters.any_active)

    logger.info(
        f"hybrid_search: query={query!r}, max_results={max_results}, fetch_limit={fetch_limit}, "
        f"rerank={effective_rerank}, candidate_limit={effective_candidates}"
    )

    # The client is synchronous by design (stdlib urllib, no new dependency), and
    # a reranked query is measured in seconds — running it inline would stall the
    # event loop for every other request in the process.
    hits = await asyncio.to_thread(
        qmd.query,
        settings.collection,
        query,
        limit=fetch_limit,
        rerank=effective_rerank,
        candidate_limit=effective_candidates,
    )

    ranked = _decode_hits(hits)
    if not ranked:
        logger.info("hybrid_search: returning 0 results")
        return []

    entries = await repository.get_entries_by_ids([entry_id for entry_id, _ in ranked])
    by_id = {entry["entry_id"]: entry for entry in entries}

    results: list[tuple[EnhancedLogbookEntry, float, list[str]]] = []
    for entry_id, hit in ranked:
        entry = by_id.get(entry_id)
        if entry is None:
            # The mirror is a copy of the database and may lag a deletion.
            logger.debug(f"hybrid_search: no row for mirrored entry {entry_id!r}, dropping hit")
            continue
        if not filters.accepts(entry):
            continue
        results.append((entry, hit.score, [hit.snippet] if hit.snippet else []))
        if len(results) >= max_results:
            break

    logger.info(f"hybrid_search: returning {len(results)} results")
    return results


def _fetch_limit(max_results: int, has_filters: bool) -> int:
    """Decide how many hits to ask the sidecar for.

    Args:
        max_results: How many results the caller wants back.
        has_filters: Whether any post-filter is active.

    Returns:
        ``max_results`` when nothing will be filtered out, otherwise the
        over-fetched count, bounded by :data:`MAX_FETCH_LIMIT` and never below
        ``max_results`` — a cap that fetched fewer than the caller asked for
        would starve the query it was meant to protect.
    """
    wanted = max(1, max_results)
    if not has_filters:
        return wanted
    return max(wanted, min(wanted * OVERFETCH_FACTOR, MAX_FETCH_LIMIT))


def _decode_hits(hits: list[QMDSearchResult]) -> list[tuple[str, QMDSearchResult]]:
    """Map ranked qmd hits onto ``entry_id``s, keeping the first of each.

    Only the basename of a hit's path carries the identifier, so a hit decodes
    the same whether the daemon reported it relative to the collection root or
    with a leading directory this client did not strip.

    Args:
        hits: Hits as the client returned them, in ranking order.

    Returns:
        ``(entry_id, hit)`` pairs in ranking order. A hit whose title declares no
        identifier is dropped with a warning: hydrating a guess would serve some
        other entry's row under this hit's rank, and a wrong answer is worse than
        a missing one.
    """
    ranked: list[tuple[str, QMDSearchResult]] = []
    seen: set[str] = set()
    for hit in hits:
        entry_id = _entry_id_from_title(hit.title)
        if entry_id is None:
            logger.warning(
                f"hybrid_search: hit declares no entry in its title, dropping it: "
                f"title={hit.title!r} file={hit.file!r}"
            )
            continue
        _warn_on_path_disagreement(entry_id, hit)
        if entry_id in seen:
            continue
        seen.add(entry_id)
        ranked.append((entry_id, hit))
    return ranked


def _entry_id_from_title(title: str) -> str | None:
    """Recover the ``entry_id`` a mirrored document declares in its own heading.

    This is the hydration channel, in place of inverting the reported path,
    because **qmd does not report the filename it indexed** — it slugifies it,
    turning both ``%`` and ``_`` into ``-``, collapsing runs and dropping a
    leading one. That makes the path lossy in two ways: an id needing any escape
    inverts to a key no row has, and — worse — ``beam_current_setpoint`` and
    ``beam-current-setpoint`` collapse onto one reported path, so inverting it
    hydrates one of them to the *other's* row.

    The title does not go through that transformation:
    :func:`~osprey.services.ariel_search.enhancement.qmd_export.writer.render_entry`
    writes ``# Entry <entry_id>`` as the document's first line and qmd surfaces
    that heading verbatim, so the identifier arrives as the writer spelled it.

    The value is used **as-is and never percent-decoded**. The heading carries
    the raw identifier, not the encoded filename, so decoding it would corrupt
    every id that merely looks encoded — ``a%2Fb`` would silently become ``a/b``.

    Args:
        title: The hit's title, as the daemon reported it.

    Returns:
        The declared identifier, or ``None`` when the title is not one this
        exporter wrote. ``None`` means drop the hit, never fall back to the
        path: the path is the channel that is known to be wrong.
    """
    if not title.startswith(TITLE_ENTRY_PREFIX):
        return None
    return title[len(TITLE_ENTRY_PREFIX) :].strip() or None


def _warn_on_path_disagreement(entry_id: str, hit: QMDSearchResult) -> None:
    """Log when the reported path would have hydrated a different row.

    The path is no longer trusted, but it is free to look at, and a mismatch is
    the fingerprint of qmd's slugification — including the collision case, where
    the path names a real *other* entry. Logging it turns a corruption that used
    to be silent into one an operator can find.

    Args:
        entry_id: The identifier taken from the title, which wins.
        hit: The hit being decoded.
    """
    try:
        from_path = entry_id_from_path(hit.file)
    except ValueError:
        from_path = None
    if from_path is not None and from_path != entry_id:
        logger.warning(
            f"hybrid_search: reported path disagrees with the document's own title; "
            f"trusting the title. title={entry_id!r} path={from_path!r} file={hit.file!r}"
        )


@dataclass(frozen=True)
class _Filters:
    """The post-hydration filters, and whether any of them is active.

    Attributes:
        start_date: Lower bound on ``timestamp``, or ``None``.
        end_date: Upper bound on ``timestamp``, or ``None``.
        author: Case-insensitive substring the author must contain, or ``None``.
        source_system: Exact source system, or ``None``.
    """

    start_date: datetime | None
    end_date: datetime | None
    author: str | None
    source_system: str | None

    @property
    def any_active(self) -> bool:
        """Whether any filter will run, and therefore whether to over-fetch."""
        return any(
            (
                self.start_date is not None,
                self.end_date is not None,
                bool(self.author),
                bool(self.source_system),
            )
        )

    def accepts(self, entry: EnhancedLogbookEntry) -> bool:
        """Report whether an entry survives every active filter.

        Args:
            entry: A hydrated ``enhanced_entries`` row.

        Returns:
            ``True`` when the entry passes. An entry with no usable timestamp is
            rejected by an active date filter rather than admitted: an
            unorderable row cannot be shown to satisfy a date range.
        """
        if self.source_system and entry.get("source_system") != self.source_system:
            return False
        if self.author and self.author.casefold() not in str(entry.get("author") or "").casefold():
            return False
        if self.start_date is None and self.end_date is None:
            return True

        stamp = _as_utc(entry.get("timestamp"))
        if stamp is None:
            return False
        if self.start_date is not None:
            bound = _as_utc(self.start_date)
            if bound is not None and stamp < bound:
                return False
        if self.end_date is not None:
            bound = _as_utc(self.end_date)
            if bound is not None and stamp > bound:
                return False
        return True


def _as_utc(value: Any) -> datetime | None:
    """Normalize a timestamp to an aware UTC ``datetime``.

    Naive values are read as UTC, matching the ``TIMESTAMPTZ`` column's storage,
    so a naive bound from a request and an aware value from the database stay
    comparable instead of raising.

    Args:
        value: A ``datetime``, an ISO-8601 string, or anything else.

    Returns:
        The UTC-normalized datetime, or ``None`` if it cannot be interpreted.
    """
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class HybridSearchInput(BaseModel):
    """Input schema for the qmd search tool."""

    query: str = Field(
        description="Natural language description or keywords describing what to find"
    )
    max_results: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum results to return",
    )
    start_date: datetime | None = Field(
        default=None,
        description="Filter entries created after this time (inclusive)",
    )
    end_date: datetime | None = Field(
        default=None,
        description="Filter entries created before this time (inclusive)",
    )


def format_qmd_result(
    entry: EnhancedLogbookEntry,
    score: float,
    highlights: list[str],
) -> dict[str, Any]:
    """Format a qmd search result for agent consumption.

    Args:
        entry: EnhancedLogbookEntry hydrated from Postgres.
        score: qmd's relevance score, an ordering signal rather than a
            calibrated probability.
        highlights: qmd's match-centered snippet, or an empty list.

    Returns:
        Formatted dict for agent
    """
    from osprey.services.ariel_search.models import _format_entry_base

    return {**_format_entry_base(entry), "score": score, "highlights": highlights}


def get_parameter_descriptors() -> list[ParameterDescriptor]:
    """Return tunable parameter descriptors for the capabilities API."""
    return [
        ParameterDescriptor(
            name="rerank",
            label="Rerank Results",
            description=(
                "Re-order candidates with the sidecar's reranker model. "
                "Improves ranking quality at roughly 4x the query latency."
            ),
            param_type="bool",
            default=DEFAULT_RERANK,
            section="Retrieval",
        ),
        ParameterDescriptor(
            name="candidate_limit",
            label="Candidate Limit",
            description=(
                "How many candidates the reranker considers. Lowering it trades recall for latency."
            ),
            param_type="int",
            default=DEFAULT_CANDIDATE_LIMIT,
            min_value=1,
            max_value=MAX_FETCH_LIMIT,
            step=1,
            section="Retrieval",
        ),
    ]


def get_tool_descriptor() -> SearchToolDescriptor:
    """Return the descriptor for auto-discovery by the agent executor."""
    return SearchToolDescriptor(
        name="hybrid_search",
        description=(
            "Hybrid keyword and semantic search over the mirrored logbook corpus, "
            "ranked by the qmd sidecar. Use when a query mixes specific terms with "
            "a described situation, or when keyword search returns too little."
        ),
        search_mode="hybrid",
        args_schema=HybridSearchInput,
        execute=hybrid_search,
        format_result=format_qmd_result,
    )
