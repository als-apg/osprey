"""ARIEL ``jev`` search module — lexical retrieval, semantic re-ranking.

An opt-in alternative to the search bar's other modes, modelled on the
two-stage shape that makes instant search possible at all:

1. **Lexical retrieval.** PostgreSQL full-text search returns a *candidate
   pool* — more rows than the operator asked for, ranked by BM25-ish relevance.
   This is the same :func:`~osprey.services.ariel_search.search.keyword.keyword_search`
   the ``keyword`` mode runs, with the same syntax, filters and vocabulary
   expansion. Nothing here re-implements retrieval.

2. **Semantic re-ranking.** One batched request asks Jev a relevance question
   per candidate plus two questions about the query itself. The answers reorder
   the pool; the top ``max_results`` of the reordered pool is what the operator
   sees.

Why this is a mode rather than a phase bolted onto ``keyword``: the two differ
in what a result *means*. A ``keyword`` hit contains the operator's terms. A
``jev`` hit was judged to be about what the operator asked, which is a claim a
model made and can be wrong about. Keeping them separate keeps that distinction
visible in the mode tab, and keeps the module disabled — and unreachable —
until a deployment turns it on.

Three properties are load-bearing:

**The lexical stage is authoritative about membership.** Re-ranking reorders a
pool; it never adds to it. An entry Postgres did not return cannot appear,
whatever the model thinks, so the ceiling on what this mode can surface is the
ceiling of the underlying full-text search — a fact an operator tuning
``candidate_limit`` should know.

**A model that cannot be reached is a degradation, not a failure.** Every path
out of the Jev call — no API key, a timeout, an error status, a malformed
answer — lands on the lexical ranking with a WARNING diagnostic saying so. A
search bar that returns nothing because a third-party endpoint was slow is
worse than one that returns the fast ranking.

**The query-level answers are confidence-gated.** Jev reports a confidence with
every choice, and this module ignores a classification below
``classification_confidence``. An unsure guess about whether the operator wants
"the most recent" would otherwise reshuffle results for no reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from osprey.services.ariel_search.models import DiagnosticLevel, SearchDiagnostic
from osprey.services.ariel_search.search.base import (
    ModuleOutput,
    ParameterDescriptor,
    SearchToolDescriptor,
)
from osprey.services.ariel_search.search.keyword import keyword_search, parse_keyword_query
from osprey.services.jev import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_ENDPOINT,
    DEFAULT_MAX_IN_FLIGHT,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    JevClient,
    JevError,
    JevSettings,
    choice_question,
    score_question,
)
from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.database.repository import ARIELRepository
    from osprey.services.ariel_search.models import EnhancedLogbookEntry
    from osprey.services.ariel_search.search.base import ParsedKeywordQuery, QueryExpansion
    from osprey.services.jev import JevAnswer, JevResult

logger = get_logger("ariel")

#: Config block the knobs below are read from.
_SETTINGS_PREFIX = "search_modules.jev.settings"

#: How many rows the lexical stage fetches for the model to reorder. Thirty is
#: the reference figure: large enough that a mediocre lexical rank can be
#: rescued, small enough that the batched request stays inside a keystroke.
DEFAULT_CANDIDATE_LIMIT = 30

#: Hard ceiling on the pool. One question per candidate, so this is also the
#: ceiling on questions per request.
MAX_CANDIDATE_LIMIT = 60

#: Normalized relevance a candidate must reach to survive. Low by design: this
#: drops the pool's obvious noise rather than second-guessing the ordering.
DEFAULT_MIN_RELEVANCE = 0.25

#: How much of the final score the lexical rank keeps when the model has no
#: opinion about query shape. Small — the model's judgement is the point — but
#: not zero, so a tie between two candidates breaks the way Postgres ranked
#: them rather than arbitrarily.
DEFAULT_LEXICAL_WEIGHT = 0.2

#: The lexical weight used instead when the model is confident the query is
#: made of exact terms (a PV name, an error string). There, the operator typed
#: the literal text they want and containing it is most of relevance.
DEFAULT_EXACT_LEXICAL_WEIGHT = 0.45

#: How much a recency prior contributes when the model is confident the query
#: asks for the latest of something ("last beam dump").
DEFAULT_RECENCY_WEIGHT = 0.3

#: Confidence a query-level classification must reach before it changes any
#: weight. Below it, the answer is discarded and the defaults stand.
DEFAULT_CLASSIFICATION_CONFIDENCE = 0.6

#: Characters of each entry's text placed in the state. Enough to judge
#: relevance, bounded so a pool of thirty long entries cannot blow the request
#: past its latency budget.
DEFAULT_SNIPPET_CHARS = 600

#: The relevance rubric, weakest level first. Written about logbook entries
#: specifically: a generic "how relevant, 1-5" rubric scores a same-subsystem
#: entry the same as one describing the event asked about.
RELEVANCE_LEVELS: tuple[str, ...] = (
    "Unrelated: the entry has nothing to do with what the operator asked about.",
    "Weak: the entry shares a subsystem or a term with the question but does not "
    "discuss what was asked.",
    "Partial: the entry is about the right subsystem and touches the question, but "
    "records a different event or measurement.",
    "Strong: the entry records the kind of event, fault or measurement the question "
    "asks about, on the right subsystem.",
    "Exact: the entry is about precisely the event, fault or measurement the question "
    "asks about, and an operator would open it first.",
)

#: Question id of the "is this query literal text?" classification.
_SHAPE_QUESTION = "query_shape"

#: Question id of the "does this query want the latest one?" classification.
_ORDER_QUESTION = "query_order"

#: Prefix of the per-candidate relevance question ids. The suffix is the
#: candidate's index in the pool, which is how an answer is mapped back.
_CANDIDATE_PREFIX = "cand_"


#: One client per distinct transport configuration, per event loop.
#:
#: A per-keystroke search bar builds a client on every request, and a fresh
#: ``httpx.AsyncClient`` is a fresh connection pool: without this cache, every
#: keystroke pays a TCP and TLS handshake it could have reused, which is most
#: of the latency budget the mode exists to stay inside. Keyed on the loop as
#: well as the settings because a pool belongs to the loop that opened it —
#: a CLI process running one ``asyncio.run`` per search must not be handed the
#: previous run's dead pool.
_CLIENTS: dict[tuple[JevSettings, int], JevClient] = {}


def _shared_client(settings: JevSettings) -> JevClient:
    """Return the cached client for these settings on this event loop.

    Args:
        settings: The transport configuration the module resolved.

    Returns:
        A client whose connection pool is reused across searches.
    """
    import asyncio

    try:
        loop_key = id(asyncio.get_running_loop())
    except RuntimeError:
        loop_key = 0
    key = (settings, loop_key)
    client = _CLIENTS.get(key)
    if client is None:
        # A loop that is not this one is a loop that has ended: its pool holds
        # nothing live, and keeping the entry would grow the cache once per
        # `asyncio.run` in a CLI process.
        for stale in [k for k in _CLIENTS if k[1] != loop_key]:
            del _CLIENTS[stale]
        client = JevClient(settings)
        _CLIENTS[key] = client
    return client


class JevSearchError(Exception):
    """Raised internally when the re-ranking stage cannot produce an order.

    Never escapes the module: :func:`jev_search` turns it into the lexical
    ranking plus a WARNING diagnostic.
    """


@dataclass(frozen=True)
class JevSearchSettings:
    """Knobs read from ``search_modules.jev.settings``.

    Attributes:
        endpoint: System One endpoint URL.
        model: Model alias to ask.
        api_key_env: Environment variable holding the API key.
        timeout_seconds: Per-request timeout for the re-ranking call.
        max_in_flight: Re-ranking requests allowed to be in flight at once.
        candidate_limit: Rows the lexical stage fetches for re-ranking.
        min_relevance: Normalized relevance a candidate must reach to survive.
        lexical_weight: Share of the final score the lexical rank keeps.
        exact_lexical_weight: The share used instead for a confident
            literal-text query.
        recency_weight: Share a recency prior takes for a confident
            latest-first query.
        classification_confidence: Floor below which a query-level answer is
            ignored.
        snippet_chars: Characters of entry text placed in the state.
        classify: Whether to ask the two query-level questions at all.
    """

    endpoint: str = DEFAULT_ENDPOINT
    model: str = DEFAULT_MODEL
    api_key_env: str = DEFAULT_API_KEY_ENV
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_in_flight: int = DEFAULT_MAX_IN_FLIGHT
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT
    min_relevance: float = DEFAULT_MIN_RELEVANCE
    lexical_weight: float = DEFAULT_LEXICAL_WEIGHT
    exact_lexical_weight: float = DEFAULT_EXACT_LEXICAL_WEIGHT
    recency_weight: float = DEFAULT_RECENCY_WEIGHT
    classification_confidence: float = DEFAULT_CLASSIFICATION_CONFIDENCE
    snippet_chars: int = DEFAULT_SNIPPET_CHARS
    classify: bool = True

    @classmethod
    def from_ariel_config(cls, config: ARIELConfig | None) -> JevSearchSettings:
        """Read the module's ``settings`` block, defaults filled in.

        A present key of the wrong type is refused rather than defaulted, the
        rule the other built-in modules follow: a ``candidate_limit`` written
        as ``"30"`` used to travel unexamined into the request and into the
        capabilities panel, where it reads as an empty box rather than as the
        configuration error it is.

        Args:
            config: The loaded ARIEL configuration, or ``None``.

        Returns:
            The resolved settings.

        Raises:
            ValueError: If a present key has the wrong type or is out of range.
                The message names the full dotted key.
        """
        module = config.search_modules.get("jev") if config is not None else None
        settings = module.settings if module is not None else None
        if not isinstance(settings, dict):
            return cls()

        defaults = cls()
        return cls(
            endpoint=_text(settings, "endpoint", defaults.endpoint),
            model=_text(settings, "model", defaults.model),
            api_key_env=_text(settings, "api_key_env", defaults.api_key_env),
            timeout_seconds=_number(settings, "timeout_seconds", defaults.timeout_seconds, 0.1, 60),
            max_in_flight=_integer(settings, "max_in_flight", defaults.max_in_flight, 1, 32),
            candidate_limit=_integer(
                settings, "candidate_limit", defaults.candidate_limit, 1, MAX_CANDIDATE_LIMIT
            ),
            min_relevance=_number(settings, "min_relevance", defaults.min_relevance, 0, 1),
            lexical_weight=_number(settings, "lexical_weight", defaults.lexical_weight, 0, 1),
            exact_lexical_weight=_number(
                settings, "exact_lexical_weight", defaults.exact_lexical_weight, 0, 1
            ),
            recency_weight=_number(settings, "recency_weight", defaults.recency_weight, 0, 1),
            classification_confidence=_number(
                settings,
                "classification_confidence",
                defaults.classification_confidence,
                0,
                1,
            ),
            snippet_chars=_integer(settings, "snippet_chars", defaults.snippet_chars, 80, 4000),
            classify=_flag(settings, "classify", defaults.classify),
        )

    def to_client_settings(self) -> JevSettings:
        """Project the transport knobs onto the client's own settings object.

        Returns:
            The client settings this module's configuration implies.
        """
        return JevSettings(
            endpoint=self.endpoint,
            model=self.model,
            api_key_env=self.api_key_env,
            timeout_seconds=self.timeout_seconds,
            max_in_flight=self.max_in_flight,
        )


def _text(settings: dict[str, Any], key: str, default: str) -> str:
    """Read a string knob, refusing a present value of another type."""
    value = settings.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{_SETTINGS_PREFIX}.{key} must be a non-empty string, got {value!r}")
    return value


def _flag(settings: dict[str, Any], key: str, default: bool) -> bool:
    """Read a boolean knob, refusing the string ``"false"`` and friends."""
    value = settings.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{_SETTINGS_PREFIX}.{key} must be a boolean, got {value!r}")
    return value


def _number(settings: dict[str, Any], key: str, default: float, low: float, high: float) -> float:
    """Read a numeric knob within an inclusive range."""
    value = settings.get(key, default)
    # ``bool`` is a subclass of ``int``, so ``True`` would otherwise pass as 1.
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not low <= value <= high:
        raise ValueError(
            f"{_SETTINGS_PREFIX}.{key} must be a number in [{low}, {high}], got {value!r}"
        )
    return float(value)


def _integer(settings: dict[str, Any], key: str, default: int, low: int, high: int) -> int:
    """Read an integer knob within an inclusive range."""
    value = settings.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
        raise ValueError(
            f"{_SETTINGS_PREFIX}.{key} must be an integer in [{low}, {high}], got {value!r}"
        )
    return value


def build_state(
    query: str,
    candidates: Sequence[EnhancedLogbookEntry],
    *,
    snippet_chars: int,
) -> dict[str, Any]:
    """Build the one state object every question is asked against.

    One state, not one per candidate: the endpoint evaluates questions in
    parallel over a shared state, so sending the pool once is what keeps thirty
    relevance judgements inside one request's latency.

    Args:
        query: The operator's raw query, as typed.
        candidates: The lexical stage's pool, in its ranked order.
        snippet_chars: Characters of each entry's text to include.

    Returns:
        A JSON-serializable state with the query and a compact record per
        candidate, each keyed by the index its question id carries.
    """
    return {
        "query": query,
        "candidates": [
            {
                "index": index,
                "entry_id": entry.get("entry_id"),
                "author": entry.get("author") or "",
                "timestamp": _isoformat(entry.get("timestamp")),
                "summary": entry.get("summary") or "",
                "keywords": list(entry.get("keywords") or []),
                "text": (entry.get("raw_text") or "")[:snippet_chars],
            }
            for index, entry in enumerate(candidates)
        ],
    }


def build_questions(
    candidate_count: int,
    *,
    classify: bool,
) -> dict[str, dict[str, Any]]:
    """Build the batched question map for one pool.

    Args:
        candidate_count: How many candidates the state carries.
        classify: Whether to include the two query-level questions.

    Returns:
        Question id to question object: one relevance ``score`` per candidate,
        plus the query-level ``choice`` questions when asked for.
    """
    questions: dict[str, dict[str, Any]] = {
        f"{_CANDIDATE_PREFIX}{index}": score_question(
            f"How well does candidate {index} answer the operator's query?",
            RELEVANCE_LEVELS,
        )
        for index in range(candidate_count)
    }
    if not classify:
        return questions

    questions[_SHAPE_QUESTION] = choice_question(
        "Is the operator's query literal text they expect to find verbatim, or a "
        "description of a situation?",
        {
            "literal": "The query is a name, identifier, channel, error string or quoted "
            "phrase the operator expects to appear verbatim in the entry.",
            "conceptual": "The query describes an event, symptom or situation in the "
            "operator's own words rather than the logbook's.",
        },
    )
    questions[_ORDER_QUESTION] = choice_question(
        "Does the operator want the most recent matching entries, or the best matching "
        "ones regardless of when they were written?",
        {
            "recency": "The query asks for the latest, last, current or most recent of "
            "something, or is about the state of the machine right now.",
            "relevance": "The query asks about a kind of event without saying when, so "
            "the best match matters more than the newest.",
        },
    )
    return questions


def normalized_score(answer: JevAnswer, levels: int) -> float:
    """Map a ``score`` answer onto 0-1.

    The endpoint reports the level it picked plus the legend it scored against.
    The legend is read rather than assumed, because a rubric's levels may be
    reported 0-indexed or 1-indexed and guessing wrong compresses every result
    toward one end of the scale. A legend that cannot be read falls back to
    ``1..levels``, the documented shape.

    Args:
        answer: The answer to map.
        levels: How many levels the rubric declared.

    Returns:
        The level's position in its own range, 0-1. A rubric with one level, or
        an unreadable score, yields 0.0.
    """
    try:
        raw = float(answer.value)
    except (TypeError, ValueError):
        return 0.0

    keys: list[float] = []
    for key in answer.legend:
        try:
            keys.append(float(key))
        except (TypeError, ValueError):
            keys = []
            break
    low, high = (min(keys), max(keys)) if len(keys) >= 2 else (1.0, float(levels))

    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (raw - low) / (high - low)))


def _confident_choice(result: JevResult, question_id: str, floor: float) -> str | None:
    """Return a choice answer only when the model was confident enough.

    Args:
        result: The decoded reply.
        question_id: Which question to read.
        floor: Minimum confidence.

    Returns:
        The chosen option, or None when the question was not asked, was not a
        choice, or came back below the floor.
    """
    answer = result.answers.get(question_id)
    if answer is None or answer.type != "choice":
        return None
    if answer.confidence is not None and answer.confidence < floor:
        return None
    return str(answer.value) if answer.value is not None else None


def blend_scores(
    relevances: Sequence[float],
    *,
    lexical_weight: float,
    recency_weight: float,
    recency_ranks: Sequence[float] | None = None,
) -> list[float]:
    """Combine the model's relevance with the priors the lexical stage supplies.

    Kept pure, and separate from the request, because this is the part an
    operator will argue with: it is the whole of how a Jev answer becomes an
    ordering, and it is testable without an endpoint.

    The lexical prior is the candidate's position in the pool, normalized so
    the first row scores 1 and the last scores 0. The recency prior is the same
    shape over the pool's timestamps. Both are *within-pool* priors: they say
    nothing across queries, which is exactly what they are used for.

    Args:
        relevances: Normalized model relevance per candidate, in pool order.
        lexical_weight: Share of the final score the lexical prior takes.
        recency_weight: Share the recency prior takes. Zero disables it.
        recency_ranks: Normalized recency prior per candidate, in pool order,
            or None when recency is not in play.

    Returns:
        One blended score per candidate, in pool order.
    """
    count = len(relevances)
    if count == 0:
        return []

    weights_total = 1.0 + (recency_weight if recency_ranks else 0.0)
    blended: list[float] = []
    for index, relevance in enumerate(relevances):
        lexical_prior = 1.0 - (index / (count - 1)) if count > 1 else 1.0
        score = (1.0 - lexical_weight) * relevance + lexical_weight * lexical_prior
        if recency_ranks:
            score += recency_weight * recency_ranks[index]
        blended.append(score / weights_total)
    return blended


def recency_priors(candidates: Sequence[EnhancedLogbookEntry]) -> list[float]:
    """Rank the pool by timestamp, newest scoring 1 and oldest 0.

    Rank-based rather than time-based on purpose: the gap between two entries
    in a logbook is wildly uneven, and a prior built from raw elapsed time
    would let one old outlier flatten the whole pool.

    Args:
        candidates: The pool, in lexical order.

    Returns:
        One prior per candidate, in pool order. An entry with no timestamp
        sorts oldest.
    """
    count = len(candidates)
    if count == 0:
        return []
    if count == 1:
        return [1.0]

    order = sorted(
        range(count),
        key=lambda index: _sort_key(candidates[index].get("timestamp")),
        reverse=True,
    )
    priors = [0.0] * count
    for position, index in enumerate(order):
        priors[index] = 1.0 - (position / (count - 1))
    return priors


def _sort_key(timestamp: Any) -> float:
    """Return a sortable epoch for a timestamp, oldest-first for a missing one."""
    if isinstance(timestamp, datetime):
        try:
            return timestamp.timestamp()
        except (OverflowError, OSError, ValueError):
            return float("-inf")
    return float("-inf")


def _isoformat(timestamp: Any) -> str | None:
    """Render a timestamp for the state, or None when there is none."""
    return timestamp.isoformat() if isinstance(timestamp, datetime) else None


async def jev_search(
    query: str,
    repository: ARIELRepository,
    config: ARIELConfig,
    *,
    max_results: int = 10,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    author: str | None = None,
    source_system: str | None = None,
    candidate_limit: int | None = None,
    min_relevance: float | None = None,
    rerank: bool = True,
    parsed: ParsedKeywordQuery | None = None,
    query_expansion: QueryExpansion | None = None,
    client: JevClient | None = None,
    **kwargs: Any,
) -> ModuleOutput:
    """Retrieve lexically, then reorder with Jev.

    Args:
        query: The operator's query, in the ``keyword`` mode's syntax — the
            lexical stage is that module, so ``author:``, ``date:``, quoted
            phrases and pattern tokens all mean what they mean there.
        repository: ARIEL database repository.
        config: ARIEL configuration.
        max_results: Entries to return after re-ranking.
        start_date: Filter entries after this time.
        end_date: Filter entries before this time.
        author: Filter by author name.
        source_system: Filter by source system.
        candidate_limit: Rows to fetch for re-ranking. Omitted, the configured
            default. Capped at :data:`MAX_CANDIDATE_LIMIT`, and never below
            ``max_results`` — a pool smaller than the answer would make the
            re-ranking decide nothing.
        min_relevance: Normalized relevance floor. Omitted, the configured one.
        rerank: When false, return the lexical ranking and make no request.
            This is what the search panel's two-phase path sends for its first
            phase, and what an operator turns off to compare the two orderings.
        parsed: The service's parse of ``query``, passed straight through.
        query_expansion: The resolved vocabulary expansion, passed straight
            through to the lexical stage.
        client: Injected client, for tests. Omitted, one is built from config.

    Returns:
        A :class:`ModuleOutput` whose ``entries`` are ``(entry, score,
        highlights)`` tuples in final order, carrying diagnostics that say
        which ordering the operator is looking at and what it cost.
    """
    settings = JevSearchSettings.from_ariel_config(config)
    pool_size = min(
        MAX_CANDIDATE_LIMIT,
        max(max_results, candidate_limit or settings.candidate_limit),
    )

    lexical = await keyword_search(
        query,
        repository,
        config,
        max_results=pool_size,
        start_date=start_date,
        end_date=end_date,
        author=author,
        source_system=source_system,
        parsed=parsed,
        query_expansion=query_expansion,
    )
    rows = list(lexical.entries) if isinstance(lexical, ModuleOutput) else list(lexical)
    diagnostics: tuple[SearchDiagnostic, ...] = (
        lexical.diagnostics if isinstance(lexical, ModuleOutput) else ()
    )
    expansion = lexical.expansion if isinstance(lexical, ModuleOutput) else ()

    if not rows or not rerank:
        return ModuleOutput(
            entries=rows[:max_results], diagnostics=diagnostics, expansion=expansion
        )

    candidates = [row[0] for row in rows]
    try:
        ordered, rerank_diagnostics = await _rerank(
            query,
            rows,
            candidates,
            settings=settings,
            max_results=max_results,
            min_relevance=(
                settings.min_relevance if min_relevance is None else float(min_relevance)
            ),
            client=client,
        )
    except JevSearchError as exc:
        logger.warning(f"jev_search: falling back to the lexical ranking: {exc}")
        return ModuleOutput(
            entries=rows[:max_results],
            diagnostics=(*diagnostics, _fallback_diagnostic(str(exc))),
            expansion=expansion,
        )

    return ModuleOutput(
        entries=ordered,
        diagnostics=(*diagnostics, *rerank_diagnostics),
        expansion=expansion,
    )


async def _rerank(
    query: str,
    rows: list[Any],
    candidates: list[EnhancedLogbookEntry],
    *,
    settings: JevSearchSettings,
    max_results: int,
    min_relevance: float,
    client: JevClient | None,
) -> tuple[list[Any], tuple[SearchDiagnostic, ...]]:
    """Ask Jev about the pool and return it reordered.

    Args:
        query: The operator's query.
        rows: The lexical stage's result tuples, in its order.
        candidates: The entries of those tuples, in the same order.
        settings: Resolved module settings.
        max_results: How many rows to return.
        min_relevance: Normalized relevance floor.
        client: Injected client, or None to build one.

    Returns:
        ``(rows, diagnostics)`` — the surviving rows in final order, each
        carrying the blended score in place of the lexical one, and the
        diagnostics describing the call.

    Raises:
        JevSearchError: If no usable ordering could be produced. The caller
            turns this into the lexical ranking plus a warning.
    """
    jev = client or _shared_client(settings.to_client_settings())
    if not jev.is_available():
        raise JevSearchError(
            f"no API key ({settings.api_key_env} is unset), so the ranking was not reviewed"
        )

    state = build_state(query, candidates, snippet_chars=settings.snippet_chars)
    questions = build_questions(len(candidates), classify=settings.classify)

    try:
        result = await jev.ask(state, questions)
    except JevError as exc:
        raise JevSearchError(str(exc)) from exc
    except Exception as exc:  # Any failure in the rerank stage is a degradation, never a 500.
        raise JevSearchError(f"unexpected failure asking Jev: {exc}") from exc

    relevances: list[float] = []
    answered = 0
    for index in range(len(candidates)):
        answer = result.answers.get(f"{_CANDIDATE_PREFIX}{index}")
        if answer is None or answer.type != "score":
            relevances.append(0.0)
            continue
        answered += 1
        relevances.append(normalized_score(answer, len(RELEVANCE_LEVELS)))

    if answered == 0:
        raise JevSearchError("the reply carried no relevance answers")

    shape = _confident_choice(result, _SHAPE_QUESTION, settings.classification_confidence)
    order = _confident_choice(result, _ORDER_QUESTION, settings.classification_confidence)

    lexical_weight = (
        settings.exact_lexical_weight if shape == "literal" else settings.lexical_weight
    )
    wants_recency = order == "recency"
    blended = blend_scores(
        relevances,
        lexical_weight=lexical_weight,
        recency_weight=settings.recency_weight if wants_recency else 0.0,
        recency_ranks=recency_priors(candidates) if wants_recency else None,
    )

    kept = [
        (row, score)
        for row, score, relevance in zip(rows, blended, relevances, strict=True)
        if relevance >= min_relevance
    ]
    # Every candidate scoring below the floor is a pool the model rejected
    # wholesale. Returning nothing there would hide results the lexical stage
    # did find, so the floor is dropped and the reordering still stands.
    if not kept:
        kept = list(zip(rows, blended, strict=True))

    kept.sort(key=lambda pair: pair[1], reverse=True)
    ordered = [(row[0], score, *row[2:]) for row, score in kept[:max_results]]

    return ordered, _rerank_diagnostics(
        result,
        pool=len(candidates),
        answered=answered,
        dropped=len(candidates) - len(kept),
        shape=shape,
        order=order,
    )


def _rerank_diagnostics(
    result: JevResult,
    *,
    pool: int,
    answered: int,
    dropped: int,
    shape: str | None,
    order: str | None,
) -> tuple[SearchDiagnostic, ...]:
    """Describe a completed re-ranking for the operator.

    Both lines are INFO and both are things the operator cannot see from the
    results themselves: what the reordering cost, and which query-level reading
    changed the weights.

    Args:
        result: The decoded reply.
        pool: Candidates sent.
        answered: Candidates the model actually scored.
        dropped: Candidates that fell below the relevance floor.
        shape: The confident query-shape answer, or None.
        order: The confident ordering answer, or None.

    Returns:
        The diagnostics, in display order.
    """
    detail = (
        f"Reranked {answered}/{pool} candidates with {result.model} in "
        f"{result.latency_ms} ms ({result.input_tokens} input tokens)"
    )
    if dropped:
        detail += f"; dropped {dropped} below the relevance floor"

    diagnostics = [
        SearchDiagnostic(
            level=DiagnosticLevel.INFO,
            source="search.jev",
            message=detail,
            category="rerank",
        )
    ]

    readings = [name for name in (shape, order) if name]
    if readings:
        diagnostics.append(
            SearchDiagnostic(
                level=DiagnosticLevel.INFO,
                source="search.jev",
                message=f"Query read as: {', '.join(readings)}",
                category="rerank",
            )
        )
    return tuple(diagnostics)


def _fallback_diagnostic(reason: str) -> SearchDiagnostic:
    """Build the WARNING that says the operator is looking at the fast ranking.

    Uses the ``rerank`` category the search panel already watches for, so the
    existing "could not improve the ranking" status line covers this module
    without a frontend change.

    Args:
        reason: Why the re-ranking did not happen, in a clause.

    Returns:
        The diagnostic.
    """
    return SearchDiagnostic(
        level=DiagnosticLevel.WARNING,
        source="search.jev",
        message=f"Showing the fast keyword ranking: {reason}",
        category="rerank",
    )


class JevSearchInput(BaseModel):
    """Input schema for the Jev search tool."""

    query: str = Field(description="What to find, in keyword-search syntax")
    max_results: int = Field(default=10, ge=1, le=50, description="Maximum results to return")
    candidate_limit: int = Field(
        default=DEFAULT_CANDIDATE_LIMIT,
        ge=1,
        le=MAX_CANDIDATE_LIMIT,
        description="Rows the keyword stage fetches for the model to reorder",
    )
    min_relevance: float = Field(
        default=DEFAULT_MIN_RELEVANCE,
        ge=0.0,
        le=1.0,
        description="Normalized relevance a candidate must reach to be returned",
    )
    rerank: bool = Field(
        default=True,
        description="Reorder with Jev. False returns the keyword ranking unchanged.",
    )
    start_date: datetime | None = Field(
        default=None, description="Filter entries created after this time (inclusive)"
    )
    end_date: datetime | None = Field(
        default=None, description="Filter entries created before this time (inclusive)"
    )
    expand_query: bool | None = Field(
        default=None,
        description=(
            "Apply the facility vocabulary expansion (shorthand/acronyms). "
            "None = the configured default"
        ),
    )


def format_jev_result(
    entry: EnhancedLogbookEntry,
    score: float,
    highlights: list[str] | None = None,
) -> dict[str, Any]:
    """Format one re-ranked result for agent consumption.

    Args:
        entry: The entry.
        score: Its blended score.
        highlights: Highlighted snippets from the lexical stage, if any.

    Returns:
        Formatted dict for the agent.
    """
    from osprey.services.ariel_search.models import _format_entry_base

    return {**_format_entry_base(entry), "score": score, "highlights": highlights or []}


def get_parameter_descriptors(config: ARIELConfig | None = None) -> list[ParameterDescriptor]:
    """Return tunable parameters for the capabilities API.

    Defaults are the deployment's, not the shipped ones, so a panel never
    invites an operator to "leave the default alone" and get something other
    than what the deployment searches with. Describing the module never fails
    on bad config: a refusal from the settings parser falls back to the shipped
    defaults, and startup validation is what names the offending key.

    ``rerank`` is declared because the search panel's two-phase path is driven
    by a parameter of that name: declaring it gets this module the paint-fast,
    replace-when-reranked behaviour with no frontend change. ``instant_search``
    is read the same way by the per-keystroke path.

    Args:
        config: The loaded ARIEL configuration, or None for shipped defaults.

    Returns:
        One descriptor per knob, in panel order.
    """
    try:
        settings = JevSearchSettings.from_ariel_config(config)
    except ValueError:
        settings = JevSearchSettings()

    return [
        ParameterDescriptor(
            name="rerank",
            label="Rerank with Jev",
            description=(
                "Reorder the keyword ranking with the Jev model. Off shows the keyword "
                "ranking alone, which is what the two orderings are compared against."
            ),
            param_type="bool",
            default=True,
            section="Retrieval",
        ),
        ParameterDescriptor(
            name="instant_search",
            label="Search as you type",
            description=(
                "Run the search on every keystroke instead of waiting for Enter. "
                "Each keystroke is one request; slower ones are superseded."
            ),
            param_type="bool",
            default=True,
            section="Retrieval",
        ),
        ParameterDescriptor(
            name="candidate_limit",
            label="Candidate Pool",
            description="Entries the keyword stage fetches for the model to reorder",
            param_type="int",
            default=settings.candidate_limit,
            min_value=1,
            max_value=MAX_CANDIDATE_LIMIT,
            step=1,
            section="Retrieval",
        ),
        ParameterDescriptor(
            name="min_relevance",
            label="Relevance Floor",
            description="Minimum model relevance (0-1) a candidate must reach to be shown",
            param_type="float",
            default=settings.min_relevance,
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            section="Retrieval",
        ),
    ]


def get_tool_descriptor() -> SearchToolDescriptor:
    """Return the descriptor for auto-discovery by the agent executor."""
    return SearchToolDescriptor(
        name="jev_search",
        description=(
            "Keyword retrieval reordered by the Jev decision model. Use when the "
            "operator's wording differs from the logbook's and the ranking matters "
            "more than the raw match — the model judges each candidate's relevance "
            "to the question actually asked."
        ),
        search_mode="jev",
        args_schema=JevSearchInput,
        execute=jev_search,
        format_result=format_jev_result,
        needs_embedder=False,
        accepts_expansion=True,
        query_parser=parse_keyword_query,
    )


__all__ = [
    "DEFAULT_CANDIDATE_LIMIT",
    "DEFAULT_MIN_RELEVANCE",
    "MAX_CANDIDATE_LIMIT",
    "RELEVANCE_LEVELS",
    "JevSearchInput",
    "JevSearchSettings",
    "blend_scores",
    "build_questions",
    "build_state",
    "format_jev_result",
    "get_parameter_descriptors",
    "get_tool_descriptor",
    "jev_search",
    "normalized_score",
    "recency_priors",
]
