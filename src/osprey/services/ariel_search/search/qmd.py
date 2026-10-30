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

**The hybrid strategy is explicit.** The compatible ``qmd`` strategy sends both
lexical and vector sub-queries to the sidecar. The opt-in ``rrf`` strategy asks
qmd only for BM25, obtains the dense ranking from ARIEL's configured semantic
provider, and combines their positions with equal-weight RRF. Keeping that merge
in ARIEL makes it deterministic and testable without qmd's vector model or
opaque weighted multi-search ordering.

**Reranking is the deployment's choice, not this module's.** qmd's LLM reranker
reads every candidate, so a reranked query is markedly slower than a plain one,
at a cost driven by the candidate pool that grows far more slowly than the
corpus does. One key — ``search_modules.hybrid.settings.rerank`` — governs both
surfaces this module serves: the agent-facing tool, whose ``rerank`` argument
may turn an enabled reranker off for one call, and the search panel, whose
toggle exists only when the deployment enables reranking. ``false`` is a hard
deployment disable. The default is qmd's own (``True``), because an agent has no
interactive budget to blow. The ``rrf`` strategy is always unreranked, so it
never loads qmd's reranker model. Reranking is also not load-bearing for
correctness: a query whose rerank fails degrades to qmd's fast ordering rather
than failing.
"""

from __future__ import annotations

import asyncio
import math
import threading
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, Field

from osprey.services.ariel_search.database.search_fts import MIN_TERM_COVERAGE, coverage_terms
from osprey.services.ariel_search.enhancement.qmd_export.writer import entry_id_from_path
from osprey.services.ariel_search.exceptions import (
    SearchConfigurationError,
    SearchTimeoutError,
)
from osprey.services.ariel_search.models import (
    END_BOUND_DESCRIPTION,
    START_BOUND_DESCRIPTION,
    DiagnosticLevel,
    SearchDiagnostic,
    TimeBoundText,
)
from osprey.services.ariel_search.search import fusion
from osprey.services.ariel_search.search.base import (
    ModuleOutput,
    ParameterDescriptor,
    SearchToolDescriptor,
    module_result,
)
from osprey.services.qmd import QMDClient, QMDUnavailableError
from osprey_connectors.logger import get_logger

if TYPE_CHECKING:
    from osprey.models.providers.base import BaseProvider
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.database.repository import ARIELRepository
    from osprey.services.ariel_search.models import EnhancedLogbookEntry
    from osprey.services.ariel_search.search._offload import SearchCallPool
    from osprey.services.ariel_search.search.base import QueryExpansion
    from osprey.services.qmd import QMDSearchResult

logger = get_logger("ariel")

#: qmd corpus the ARIEL mirror is indexed as: the name its sidecar's port is
#: resolved by and the collection its one index holds. Not a config key — it is
#: fixed by the deployment's corpus derivation, not chosen per deployment,
#: exactly as ``OKF_COLLECTION`` is on the facility-knowledge side.
ARIEL_COLLECTION = "ariel"

#: Whether the reranker runs when config says nothing. Matches qmd's own default.
#: See the module docstring for why the agent-facing tool keeps the slow path.
DEFAULT_RERANK = True

#: Candidates the reranker considers (qmd's ``candidateLimit``) when config says
#: nothing. Matches qmd's own default; lowering it trades recall for latency.
DEFAULT_CANDIDATE_LIMIT = 40

#: The existing all-qmd implementation remains the shipped behavior.  RRF is
#: opt-in because it deliberately replaces qmd's baked-in vector model with
#: ARIEL's configured semantic provider.
DEFAULT_STRATEGY = "qmd"
_HYBRID_STRATEGIES = frozenset(("qmd", "rrf"))

#: Standard reciprocal-rank-fusion constant.  At this value agreement between
#: the BM25 and Nomic lanes matters more than small score-scale differences
#: within either backend, while the leading positions still contribute most.
RRF_K = 60

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
        strategy: ``qmd`` retains qmd's combined lexical/vector query. ``rrf``
            combines qmd's lexical BM25 ranking with ARIEL's configured dense
            ranking using reciprocal-rank fusion.
        rerank: Whether qmd reranks candidates with its LLM reranker. Defaults
            to :data:`DEFAULT_RERANK`; ignored by the lightweight ``rrf``
            strategy.
        candidate_limit: Candidate depth for each lane in ``rrf`` and qmd's
            ``candidateLimit`` in ``qmd``.
            Defaults to :data:`DEFAULT_CANDIDATE_LIMIT`.
        collection: qmd collection the ARIEL mirror is indexed under. Not a
            config key — see :data:`ARIEL_COLLECTION`.
    """

    strategy: str = DEFAULT_STRATEGY
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
            ValueError: If ``strategy`` is not supported, ``rerank`` is present
                but not a boolean, or
                ``candidate_limit`` is present but not a positive integer.
        """
        module = config.search_modules.get("hybrid") if config is not None else None
        settings = module.settings if module is not None else None
        if not isinstance(settings, dict):
            return cls()

        strategy = settings.get("strategy", cls.strategy)
        if strategy not in _HYBRID_STRATEGIES:
            choices = ", ".join(sorted(_HYBRID_STRATEGIES))
            raise ValueError(
                f"{_SETTINGS_PREFIX}.strategy must be one of {choices}, got {strategy!r}"
            )

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

        return cls(strategy=strategy, rerank=rerank, candidate_limit=candidate_limit)


@dataclass(frozen=True)
class _RankedTextHit:
    """One backend-neutral text hit after identity recovery."""

    entry_id: str
    score: float
    snippet: str = ""


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
            from osprey.deployment.qmd_service import resolve_qmd_corpus_config
            from osprey_connectors.workspace import load_osprey_config

            _cached_client = QMDClient(
                resolve_qmd_corpus_config(load_osprey_config(), ARIEL_COLLECTION)
            )
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
    embedder: BaseProvider | None = None,
    *,
    max_results: int = 10,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    author: str | None = None,
    source_system: str | None = None,
    rerank: bool | None = None,
    candidate_limit: int | None = None,
    client: QMDClient | None = None,
    query_expansion: QueryExpansion | None = None,
    include_images: bool | None = None,
    search_calls: SearchCallPool,
    **kwargs: Any,
) -> list[tuple[EnhancedLogbookEntry, float, list[str]]] | ModuleOutput:
    """Execute the configured qmd or RRF hybrid strategy.

    The sidecar ranks; Postgres answers. Hits are decoded back to ``entry_id``s,
    hydrated by primary key, filtered, and returned in qmd's ranking order.

    With picture search on, the picture lane runs concurrently with the text
    lane and the two rankings are fused: entries whose nearest picture is close
    to the query move up, and up to ``ceil(max_results / 3)`` entries matched
    only through a picture join the result. Each fused entry carries
    ``_matched_via`` (a sorted subset of ``{"image", "text"}``) and, when a
    picture matched, that picture's id at the end of
    ``_matched_attachment_ids``. A picture lane that fails or is cooling down
    leaves the text result exactly as it would be with the lane off, plus a
    "Picture search unavailable" diagnostic.

    Args:
        query: Natural-language or keyword query.
        repository: ARIEL database repository, used to hydrate hits.
        config: ARIEL configuration, read for the module's query knobs.
        embedder: Provider used by ARIEL's existing semantic search, for the
            ``rrf`` strategy only. When omitted, the ``rrf`` path resolves the
            configured provider class; the ``qmd`` strategy never touches it.
        max_results: Maximum text-lane entries to return (default: 10).
            Image-only entries are counted separately, at most
            ``ceil(max_results / 3)`` of them.
        start_date: Keep entries at or after this time.
        end_date: Keep entries at or before this time.
        author: Keep entries whose author contains this text, case-insensitively.
        source_system: Keep entries from this source system (exact match).
        rerank: Per-query override when the ``qmd`` strategy's
            ``settings.rerank`` enables the
            reranker. A deployment setting of ``false`` is a hard disable and
            cannot be overridden by a request. The ``rrf`` strategy never uses
            the qmd reranker.
        candidate_limit: Per-query override of ``settings.candidate_limit``;
            RRF fetches this many candidates per lane. ``None`` uses config.
        client: Client for the sidecar. Defaults to the process-wide client
            built from the project's ``services.qmd`` block; tests pass a stub.
        query_expansion: Resolved vocabulary expansion for this query, passed by
            the service only when one was actually resolved. When present, its
            ``flattened_text`` is sent to the sidecar in place of `query` — the
            whole query is the matching text here, and it is never truncated.
            The expanded text is what the sidecar's reranker sees, which is why
            a deployment whose reranked ordering degrades drops ``hybrid`` from
            ``ariel.vocabulary.expand_modes``. The picture lane always sees the
            original `query`.
        include_images: Whether the picture lane runs. ``None`` means on
            exactly when ``image_embedding`` is enabled.
        search_calls: The search service's search-call pool. Every provider
            call of every lane runs on it: the picture lane embeds the query
            there, and so does the ``rrf`` strategy's semantic lane.

    Returns:
        Without `query_expansion`: a list of ``(entry, score, snippets)`` tuples
        in descending relevance order — the bare-list contract every direct
        caller relies on today, returned unchanged. ``score`` is an **ordering
        signal, not a calibrated probability** — do not threshold it. It is
        qmd's score for the ``qmd`` strategy and the normalised RRF score for
        the ``rrf`` strategy. A picture match likewise changes it to the fused
        reciprocal-rank score divided by the best one, so the top entry scores
        1.0. ``snippets``
        carries qmd's line-numbered, match-centered excerpt, or is empty when
        the sidecar supplied none or the entry matched only through a picture.

        With `query_expansion`: a `ModuleOutput` carrying those same tuples as
        `entries` and the applied expansion groups as `expansion`. The shape is
        conditional so that direct callers keep the return value they have
        always received; the service unwraps either form.

        A `ModuleOutput` is also returned — with or without an expansion —
        when the query produced a diagnostic: the reranked query failed and the
        results come from the unreranked retry, the picture lane failed, or
        the sidecar is down and the results are picture matches only.

    Raises:
        QMDUnavailableError: If no sidecar is configured, or the configured one
            is not answering, and the picture lane matched nothing (or did not
            run). This is deliberately not an empty result: "search is down"
            and "nothing matched" must not look alike to the agent.
        QMDClientError: If the sidecar was reached but could not answer. A
            reranked query that fails is retried once without the reranker, so
            what surfaces here is the retry's failure, not the first attempt's.
        SearchConfigurationError: If the module's config block holds a
            malformed knob. Typed so the service can report it as a
            configuration problem rather than a generic module failure.
    """
    if not query.strip():
        return module_result([], query_expansion)

    try:
        settings = HybridSearchSettings.from_ariel_config(config)
    except ValueError as exc:
        # The parse refuses a bad knob with a ValueError naming the key. On the
        # query path that has to reach the service as a *configuration* failure:
        # an untyped crash is reported to the agent as a broken sidecar, and it
        # goes looking at a service that is perfectly healthy. Message verbatim.
        raise SearchConfigurationError(str(exc)) from exc
    rrf = settings.strategy == "rrf"
    effective_rerank = False if rrf else settings.rerank and (rerank is None or bool(rerank))
    effective_candidates = settings.candidate_limit if candidate_limit is None else candidate_limit

    filters = _Filters(start_date, end_date, author, source_system)
    fetch_limit = _fetch_limit(max_results, filters.any_active)

    lane_on = (
        config.is_enhancement_module_enabled("image_embedding")
        if include_images is None
        else bool(include_images)
    )
    # Started before the sidecar is resolved so the two lanes overlap; the
    # picture lane sees the typed query, never a vocabulary expansion.
    img_task: asyncio.Task[dict[str, fusion.ImageHit] | None] | None = None
    if lane_on:
        # Imported here: the lane's dependencies import the config module,
        # which imports this package.
        from osprey.services.ariel_search.search import image_lane

        img_task = asyncio.create_task(
            image_lane.search_images(
                query, repository, config, fetch_limit=fetch_limit, search_calls=search_calls
            )
        )
    try:
        return await _ranked_search(
            query,
            repository,
            config,
            settings=settings,
            filters=filters,
            fetch_limit=fetch_limit,
            max_results=max_results,
            rerank=effective_rerank,
            rrf=rrf,
            candidate_limit=effective_candidates,
            client=client,
            embedder=embedder,
            query_expansion=query_expansion,
            img_task=img_task,
            search_calls=search_calls,
        )
    finally:
        if img_task is not None:
            if not img_task.done():
                img_task.cancel()
            await asyncio.gather(img_task, return_exceptions=True)


#: Message of the diagnostic a failed or cooling-down picture lane adds.
PICTURE_UNAVAILABLE_MESSAGE = (
    "Picture search unavailable — results are matched on text only, so entries "
    "known only by their pictures are missing."
)

#: Message of the diagnostic a timed-out semantic lane adds to a hybrid answer.
SEMANTIC_UNAVAILABLE_MESSAGE = (
    "Semantic ranking unavailable — the query embedding timed out, so results "
    "are ranked on keywords only."
)

#: Opening of the diagnostic an image-only answer to a sidecar outage carries;
#: :func:`_text_unavailable_message` completes it with the sidecar dialled.
TEXT_UNAVAILABLE_MESSAGE = "Text ranking unavailable — picture matches only"


def _sidecar_cause(qmd: QMDClient) -> str:
    """Why *qmd* cannot rank text: no sidecar configured, or the one dialled is silent.

    The URL is the one this process dialled, which inside a container is not
    the address the config block publishes, so it is named rather than implied.
    """
    if not qmd.is_configured:
        return "no qmd sidecar is configured for this deployment"
    return f"the qmd sidecar at {qmd.base_url} is not answering"


def _text_unavailable_message(qmd: QMDClient) -> str:
    """The picture-only diagnostic, naming why the text lane is missing."""
    return (
        f"{TEXT_UNAVAILABLE_MESSAGE}; {_sidecar_cause(qmd)}, "
        "so entries matching on text are missing."
    )


async def _ranked_search(
    query: str,
    repository: ARIELRepository,
    config: ARIELConfig,
    *,
    settings: HybridSearchSettings,
    filters: _Filters,
    fetch_limit: int,
    max_results: int,
    rerank: bool,
    rrf: bool,
    candidate_limit: int | None,
    client: QMDClient | None,
    embedder: BaseProvider | None,
    query_expansion: QueryExpansion | None,
    img_task: asyncio.Task[dict[str, fusion.ImageHit] | None] | None,
    search_calls: SearchCallPool,
) -> list[tuple[EnhancedLogbookEntry, float, list[str]]] | ModuleOutput:
    """Run the text lane, join the picture lane when it ran, and shape the result.

    See :func:`hybrid_search`; `img_task` is the running picture lane, or
    ``None`` when picture search is off for this call.
    """
    qmd, available = await asyncio.to_thread(_resolve_client, client)
    if not available:
        image_hits = await img_task if img_task is not None else None
        if image_hits:
            results = await _fused_results(
                [],
                image_hits,
                repository,
                config,
                filters=filters,
                max_results=max_results,
                query=query,
                query_expansion=query_expansion,
            )
            logger.info(f"hybrid_search: sidecar down, returning {len(results)} picture matches")
            return ModuleOutput(
                entries=results,
                diagnostics=(
                    SearchDiagnostic(
                        level=DiagnosticLevel.WARNING,
                        source="hybrid",
                        message=_text_unavailable_message(qmd),
                        category="text_ranking",
                    ),
                ),
                expansion=query_expansion.groups if query_expansion else (),
            )
        raise QMDUnavailableError(_sidecar_cause(qmd))

    logger.info(
        f"hybrid_search: query={query!r}, max_results={max_results}, fetch_limit={fetch_limit}, "
        f"strategy={settings.strategy}, rerank={rerank}, candidate_limit={candidate_limit}, "
        f"picture_lane={img_task is not None}"
    )

    lane_limit = max(fetch_limit, candidate_limit or fetch_limit)
    semantic_task: asyncio.Task[list[_RankedTextHit]] | None = None
    if rrf:
        semantic_task = asyncio.create_task(
            _semantic_hits(
                query,
                repository,
                config,
                embedder,
                limit=lane_limit,
                filters=filters,
                query_expansion=query_expansion,
                search_calls=search_calls,
            )
        )
    try:
        hits, diagnostics = await _fetch_hits(
            qmd,
            settings.collection,
            query_expansion.flattened_text if query_expansion else query,
            limit=fetch_limit,
            rerank=rerank,
            lexical_only=rrf,
            candidate_limit=candidate_limit,
        )
    except BaseException:
        if semantic_task is not None:
            semantic_task.cancel()
            await asyncio.gather(semantic_task, return_exceptions=True)
        raise
    image_hits = await img_task if img_task is not None else None
    if img_task is not None and image_hits is None:
        diagnostics = (
            *diagnostics,
            SearchDiagnostic(
                level=DiagnosticLevel.WARNING,
                source="hybrid",
                message=PICTURE_UNAVAILABLE_MESSAGE,
                category="picture_search",
            ),
        )

    def shaped(
        entries: list[tuple[EnhancedLogbookEntry, float, list[str]]],
    ) -> list[tuple[EnhancedLogbookEntry, float, list[str]]] | ModuleOutput:
        """Return `entries` in the shape this call's contract asks for.

        Same rule as `module_result`, with diagnostics folded in: a caller that
        gets neither an expansion nor a diagnostic keeps the bare list it has
        always received, and anything richer travels as one `ModuleOutput`.
        """
        if not diagnostics:
            return module_result(entries, query_expansion)
        return ModuleOutput(
            entries=entries,
            diagnostics=diagnostics,
            expansion=query_expansion.groups if query_expansion else (),
        )

    ranked = _decode_hits(hits)
    if semantic_task is not None:
        try:
            semantic = await semantic_task
        except SearchTimeoutError as e:
            # A hung or saturated embedding provider costs hybrid its semantic
            # lane, never the keyword answer the sidecar already gave.
            logger.warning(f"hybrid_search: semantic lane skipped: {e}")
            semantic = []
            diagnostics = (
                *diagnostics,
                SearchDiagnostic(
                    level=DiagnosticLevel.WARNING,
                    source="hybrid",
                    message=SEMANTIC_UNAVAILABLE_MESSAGE,
                    category="semantic_search",
                ),
            )
        ranked = _rrf_hits(ranked, semantic)

    if image_hits is not None:
        results = await _fused_results(
            ranked,
            image_hits,
            repository,
            config,
            filters=filters,
            max_results=max_results,
            query=query,
            query_expansion=query_expansion,
        )
        logger.info(f"hybrid_search: returning {len(results)} fused results")
        return shaped(results)

    if not ranked:
        logger.info("hybrid_search: returning 0 results")
        return shaped([])

    entries = await repository.get_entries_by_ids([hit.entry_id for hit in ranked])
    by_id = {entry["entry_id"]: entry for entry in entries}

    results = []
    for hit in ranked:
        entry = by_id.get(hit.entry_id)
        if entry is None:
            # The mirror is a copy of the database and may lag a deletion.
            logger.debug(f"hybrid_search: no row for ranked entry {hit.entry_id!r}, dropping hit")
            continue
        if not filters.accepts(entry):
            continue
        results.append((entry, hit.score, [hit.snippet] if hit.snippet else []))
        if len(results) >= max_results:
            break

    if results:
        await _attach_caption_matches(
            results,
            repository,
            config,
            terms=coverage_terms(query, query_expansion),
        )

    logger.info(f"hybrid_search: returning {len(results)} results")
    return shaped(results)


async def _fused_results(
    ranked: list[_RankedTextHit],
    image_hits: dict[str, fusion.ImageHit],
    repository: ARIELRepository,
    config: ARIELConfig,
    *,
    filters: _Filters,
    max_results: int,
    query: str,
    query_expansion: QueryExpansion | None,
) -> list[tuple[EnhancedLogbookEntry, float, list[str]]]:
    """Hydrate both lanes in one read, fuse them, and keep the page.

    Every filter-accepted text hit takes part in the fusion. The result keeps
    at most `max_results` text-lane entries and, counted separately, at most
    ``ceil(max_results / 3)`` image-only entries, so a picture match never
    pushes a fetched text hit out. Each kept entry is marked ``_matched_via``;
    an entry whose picture matched gets that picture's id after its caption
    ids in ``_matched_attachment_ids``.
    """
    text_ids = [hit.entry_id for hit in ranked]
    seen = set(text_ids)
    ids = text_ids + [entry_id for entry_id in image_hits if entry_id not in seen]
    if not ids:
        return []
    entries = await repository.get_entries_by_ids(ids)
    by_id = {entry["entry_id"]: entry for entry in entries}

    text_hits: list[tuple[str, float]] = []
    snippets: dict[str, list[str]] = {}
    for hit in ranked:
        entry = by_id.get(hit.entry_id)
        if entry is None:
            logger.debug(f"hybrid_search: no row for ranked entry {hit.entry_id!r}, dropping hit")
            continue
        if not filters.accepts(entry):
            continue
        text_hits.append((hit.entry_id, hit.score))
        snippets[hit.entry_id] = [hit.snippet] if hit.snippet else []

    accepted_images = {
        entry_id: hit
        for entry_id, hit in image_hits.items()
        if entry_id in by_id and filters.accepts(by_id[entry_id])
    }
    image_cap = math.ceil(max_results / 3)
    fused = fusion.fuse_lanes(text_hits, accepted_images, cap=image_cap)

    kept: list[fusion.FusedHit] = []
    text_kept = image_kept = 0
    for fused_hit in fused:
        if fused_hit.matched_via == ["image"]:
            if image_kept >= image_cap:
                continue
            image_kept += 1
        else:
            if text_kept >= max_results:
                continue
            text_kept += 1
        kept.append(fused_hit)

    results: list[tuple[EnhancedLogbookEntry, float, list[str]]] = []
    for fused_hit in kept:
        entry = by_id[fused_hit.entry_id]
        cast("dict[str, Any]", entry)["_matched_via"] = list(fused_hit.matched_via)
        results.append((entry, fused_hit.score, snippets.get(fused_hit.entry_id, [])))

    if results:
        await _attach_caption_matches(
            results,
            repository,
            config,
            terms=coverage_terms(query, query_expansion),
        )
    for fused_hit, (entry, _score, _snippets) in zip(kept, results, strict=True):
        if fused_hit.attachment_id is None:
            continue
        row = cast("dict[str, Any]", entry)
        matched = list(row.get("_matched_attachment_ids") or ())
        if fused_hit.attachment_id not in matched:
            matched.append(fused_hit.attachment_id)
        row["_matched_attachment_ids"] = matched
    return results


async def _attach_caption_matches(
    results: list[tuple[EnhancedLogbookEntry, float, list[str]]],
    repository: ARIELRepository,
    config: ARIELConfig,
    *,
    terms: list[tuple[str, tuple[str, ...]]],
) -> None:
    """Mark each result with the ids of its attachments whose caption matched.

    One ``caption_matches`` call over the result ids in its coverage form: a
    caption matches when it holds at least :data:`MIN_TERM_COVERAGE` of the
    query's words (`terms`), a word holding when it or one of its vocabulary
    alternatives does. A result whose captions matched
    gets ``_matched_attachment_ids``; ordering and scores are untouched. The
    ids are supplementary evidence: a timeout or database failure logs one
    WARNING and marks nothing, so it never fails a search the sidecar answered.
    """
    from osprey.services.ariel_search.attachments.compose import caption_model_id
    from osprey.services.ariel_search.exceptions import (
        DatabaseQueryError,
        SearchTimeoutError,
    )

    try:
        matched = await repository.caption_matches(
            [entry["entry_id"] for entry, _score, _highlights in results],
            caption_model_id(config),
            terms=terms,
            min_fraction=MIN_TERM_COVERAGE,
        )
    except (SearchTimeoutError, DatabaseQueryError) as e:
        logger.warning(f"hybrid_search: caption matching skipped: {e}")
        return
    for entry, _score, _highlights in results:
        ids = matched.get(entry["entry_id"])
        if ids:
            cast("dict[str, Any]", entry)["_matched_attachment_ids"] = list(ids)


async def _fetch_hits(
    qmd: QMDClient,
    collection: str,
    text: str,
    *,
    limit: int,
    rerank: bool,
    lexical_only: bool,
    candidate_limit: int | None,
) -> tuple[list[QMDSearchResult], tuple[SearchDiagnostic, ...]]:
    """Ask the sidecar to rank, degrading to the unreranked ordering if it must.

    Reranking is not load-bearing for correctness — it improves an ordering that
    is already usable — so a reranker fault costs the query its ranking quality
    rather than its results.

    Args:
        qmd: The resolved sidecar client, already known to be answering.
        collection: qmd collection to search.
        text: The query text as the sidecar should see it.
        limit: How many hits to ask for.
        rerank: Whether to run the reranker at all.
        lexical_only: Ask qmd for BM25 results only, for ARIEL's RRF strategy.
        candidate_limit: Candidate depth for the RRF BM25 lane and qmd's
            ``candidateLimit`` for a qmd query.

    Returns:
        The hits, and the diagnostics to carry back with them — one WARNING
        when the hits came from the unreranked retry, otherwise empty.

    Raises:
        Exception: Whatever the client raised, when the unreranked query fails
            too. One retry, not a loop: if the fast path is down, search is down.
    """

    async def run(
        *,
        rerank: bool,
        lexical_only: bool = False,
        query_limit: int | None = None,
    ) -> list[QMDSearchResult]:
        # The client is synchronous by design (stdlib urllib, no new dependency),
        # and a reranked query is measured in seconds — running it inline would
        # stall the event loop for every other request in the process.
        return await asyncio.to_thread(
            qmd.query,
            collection,
            text,
            limit=query_limit or limit,
            search_types=("lex",) if lexical_only else ("lex", "vec"),
            rerank=rerank,
            candidate_limit=candidate_limit,
        )

    if not rerank:
        lane_limit = max(limit, candidate_limit or limit)
        return await run(
            rerank=False,
            lexical_only=lexical_only,
            query_limit=lane_limit if lexical_only else None,
        ), ()

    try:
        return await run(rerank=True), ()
    except Exception as exc:  # any reranker fault falls back
        # The client already proved /health answers at resolve time, so a
        # failure here is the reranker itself: the model is still loading
        # after a sidecar restart and the query outran the timeout, or the
        # reranked query took the daemon down with it. Either way the
        # unreranked path is a different code path in the daemon and is
        # worth one try — a dead daemon refuses it fast.
        logger.warning(
            f"hybrid_search: reranked query failed ({type(exc).__name__}: {exc}); "
            "retrying without the reranker"
        )
        return await run(rerank=False), (
            SearchDiagnostic(
                level=DiagnosticLevel.WARNING,
                source="hybrid",
                message=(
                    "Reranker unavailable — results are not reranked, so the "
                    "ordering may be less relevant than usual."
                ),
                category="rerank",
            ),
        )


async def _semantic_hits(
    query: str,
    repository: ARIELRepository,
    config: ARIELConfig,
    embedder: BaseProvider | None,
    *,
    limit: int,
    filters: _Filters,
    query_expansion: QueryExpansion | None,
    search_calls: SearchCallPool,
) -> list[_RankedTextHit]:
    """Run the existing Nomic/pgvector search and reduce it to an RRF lane."""
    if not config.get_search_model():
        return []
    if embedder is None:
        from osprey.services.ariel_search.search.semantic import semantic_provider_class

        embedder = semantic_provider_class(config)()

    from osprey.services.ariel_search.search.semantic import semantic_search

    output = await semantic_search(
        query,
        repository,
        config,
        embedder,
        max_results=limit,
        # Hybrid needs a candidate ranking rather than semantic mode's
        # operator-facing cutoff; RRF itself decides the final order.
        similarity_threshold=0.0,
        start_date=filters.start_date,
        end_date=filters.end_date,
        author=filters.author,
        source_system=filters.source_system,
        query_expansion=query_expansion,
        search_calls=search_calls,
    )
    rows = output.entries if isinstance(output, ModuleOutput) else output
    return [_RankedTextHit(entry_id=entry["entry_id"], score=float(score)) for entry, score in rows]


def _rrf_hits(
    lexical: list[_RankedTextHit],
    semantic: list[_RankedTextHit],
    *,
    k: int = RRF_K,
) -> list[_RankedTextHit]:
    """Fuse qmd BM25 and ARIEL/Nomic rankings by stable entry ID.

    Raw BM25 and cosine scores are deliberately ignored because they live on
    unrelated scales. Only rank positions contribute. The lexical hit is
    retained when a document appears in both lists because it carries qmd's
    matching snippet. Scores are normalised so the first result remains 1.0,
    matching the existing UI contract.
    """
    if k < 1:
        raise ValueError("RRF k must be positive")

    hits: dict[str, _RankedTextHit] = {}
    ranks: dict[str, list[int | None]] = {}
    scores: dict[str, float] = {}

    for lane, ranking in enumerate((lexical, semantic)):
        seen: set[str] = set()
        for rank, hit in enumerate(ranking, start=1):
            key = hit.entry_id
            if key in seen:
                continue
            seen.add(key)
            hits.setdefault(key, hit)
            ranks.setdefault(key, [None, None])[lane] = rank
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

    if not scores:
        return []

    def order(key: str) -> tuple[float, int, int, str]:
        lexical_rank, semantic_rank = ranks[key]
        missing = len(lexical) + len(semantic) + 1
        return (
            -scores[key],
            lexical_rank if lexical_rank is not None else missing,
            semantic_rank if semantic_rank is not None else missing,
            key,
        )

    ordered = sorted(scores, key=order)
    top = scores[ordered[0]]
    return [replace(hits[key], score=scores[key] / top) for key in ordered]


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


def _decode_hits(hits: list[QMDSearchResult]) -> list[_RankedTextHit]:
    """Map ranked qmd hits onto ``entry_id``s, keeping the first of each.

    Only the basename of a hit's path carries the identifier, so a hit decodes
    the same whether the daemon reported it relative to the collection root or
    with a leading directory this client did not strip.

    Args:
        hits: Hits as the client returned them, in ranking order.

    Returns:
        Backend-neutral hits in ranking order. A hit whose title declares no
        identifier is dropped with a warning: hydrating a guess would serve
        some other entry's row under this hit's rank, and a wrong answer is
        worse than a missing one.
    """
    ranked: list[_RankedTextHit] = []
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
        ranked.append(_RankedTextHit(entry_id, hit.score, hit.snippet))
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
    start_date: TimeBoundText = Field(default=None, description=START_BOUND_DESCRIPTION)
    end_date: TimeBoundText = Field(default=None, description=END_BOUND_DESCRIPTION)
    expand_query: bool | None = Field(
        default=None,
        description=(
            "Apply the facility vocabulary expansion (shorthand/acronyms). "
            "None = the configured default (capabilities().vocabulary.expand_by_default)"
        ),
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


def get_parameter_descriptors(config: ARIELConfig | None = None) -> list[ParameterDescriptor]:
    """Return tunable parameter descriptors for the capabilities API.

    The defaults reported here are the deployment's, not the shipped ones: a
    panel that opened on ``rerank: true`` while the deployment had turned
    reranking off would invite an operator to "leave the default alone" and get
    the opposite of it. Reading them from the same
    ``search_modules.hybrid.settings`` block the query path reads keeps the two
    in step.

    Describing the module never fails on bad config. ``/api/capabilities``, the
    search page and the MCP capabilities tool all walk this function, and a
    deployment with one malformed key should still see a described, usable
    module; the key itself is reported by startup validation, which is the
    surface that can explain it. So a refusal from the settings parser falls
    back to the shipped defaults here rather than propagating.

    Args:
        config: The loaded ARIEL configuration. ``None`` — the case for a
            caller that has no config in hand — yields the shipped defaults.

    Returns:
        One descriptor per tunable knob, in panel order.
    """
    try:
        settings = HybridSearchSettings.from_ariel_config(config)
    except ValueError:
        settings = HybridSearchSettings()

    descriptors = [
        ParameterDescriptor(
            name="candidate_limit",
            label="Candidate Limit",
            description=(
                "How many candidates each RRF lane considers, or qmd's candidate "
                "pool for its native strategy. Lowering it trades recall for latency."
            ),
            param_type="int",
            default=settings.candidate_limit,
            min_value=1,
            max_value=MAX_FETCH_LIMIT,
            step=1,
            section="Retrieval",
        ),
    ]
    # A deployment setting of false is a hard disable, not merely the panel's
    # initial toggle position. Omitting the descriptor keeps the panel on its
    # one-phase RRF path and gives operators no control that can only no-op.
    if settings.strategy == "qmd" and settings.rerank:
        descriptors.insert(
            0,
            ParameterDescriptor(
                name="rerank",
                label="Rerank Results",
                description=(
                    "Re-order candidates with the sidecar's reranker model for better "
                    "ordering. Significantly slower — an LLM reviews each result. If "
                    "reranking fails, the results are shown without it."
                ),
                param_type="bool",
                default=True,
                section="Retrieval",
            ),
        )
    # Offered only where picture search can actually run: a knob that could
    # only ever no-op would invite an operator to believe it did something.
    if (
        config is not None
        and config.is_search_module_enabled("hybrid")
        and config.is_enhancement_module_enabled("image_embedding")
    ):
        descriptors.append(
            ParameterDescriptor(
                name="include_images",
                label="Search Pictures",
                description=(
                    "Also match entries by what their attached plots, screenshots and photos show."
                ),
                param_type="bool",
                default=True,
                section="Retrieval",
            )
        )
    return descriptors


def get_tool_descriptor() -> SearchToolDescriptor:
    """Return the descriptor for auto-discovery by the agent executor."""
    return SearchToolDescriptor(
        name="hybrid_search",
        description=(
            "Hybrid logbook search that fuses qmd BM25 and ARIEL/Nomic semantic "
            "rankings. Use when a query mixes specific terms with a described "
            "situation, or when keyword search returns too little."
        ),
        search_mode="hybrid",
        args_schema=HybridSearchInput,
        execute=hybrid_search,
        format_result=format_qmd_result,
        # Only the rrf strategy embeds, and it resolves its own provider, so the
        # default qmd strategy depends on the sidecar alone; every lane's provider
        # call runs on the pool the service hands over.
        needs_embedder=False,
        needs_search_calls=True,
        accepts_expansion=True,
        accepts_include_images=True,
    )
