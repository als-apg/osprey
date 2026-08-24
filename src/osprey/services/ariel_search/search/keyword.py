"""ARIEL keyword search module.

This module provides full-text search using PostgreSQL's built-in
text search capabilities with optional fuzzy matching fallback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from osprey.services.ariel_search.database.search_fts import (
    build_expanded_tsquery,
    keyword_fts_expression,
)
from osprey.services.ariel_search.exceptions import PatternError
from osprey.services.ariel_search.models import DiagnosticLevel, SearchDiagnostic
from osprey.services.ariel_search.search.base import (
    ExpansionGroup,
    ModuleOutput,
    ParameterDescriptor,
    ParsedKeywordQuery,
    PatternSpan,
    SearchToolDescriptor,
)
from osprey.services.ariel_search.search.patterns import (
    Accepted,
    accept_pattern,
    extract_pattern_spans,
    glob_to_are,
    is_glob,
    strip_trailing_question,
)
from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.database.repository import ARIELRepository
    from osprey.services.ariel_search.models import EnhancedLogbookEntry
    from osprey.services.ariel_search.search.base import QueryExpansion
    from osprey.services.ariel_search.search.patterns import RawSpan

logger = get_logger("ariel")

ALLOWED_OPERATORS = {"AND", "OR", "NOT"}
ALLOWED_FIELD_PREFIXES = {"author:", "date:"}

# Default fuzzy threshold (pg_trgm similarity)
DEFAULT_FUZZY_THRESHOLD = 0.3

MAX_QUERY_LENGTH = 1000

#: Whether the pattern machinery (``/regex/`` spans and ``*`` globs) is armed.
#: On by default: the forms are what an operator reaches for first, and a
#: deployment that wants ``*`` treated as ordinary text says so explicitly.
DEFAULT_PATTERNS_ENABLED = True

#: Statement-timeout envelope, in seconds, wrapped around a keyword search that
#: emitted at least one ``~*`` predicate. Ten seconds is generous for an
#: indexed pattern and short enough that a pathological one gives the operator
#: a timeout diagnostic instead of a hung control-room panel.
DEFAULT_PATTERN_TIMEOUT_SECONDS = 10.0

#: Config block the keyword knobs are read from.
_SETTINGS_PREFIX = "search_modules.keyword.settings"


@dataclass(frozen=True)
class KeywordSearchSettings:
    """Query knobs read from ``search_modules.keyword.settings``.

    Attributes:
        patterns_enabled: Whether ``/regex/`` spans and ``*`` globs are
            recognized. Defaults to :data:`DEFAULT_PATTERNS_ENABLED`; False
            makes both forms ordinary full-text tokens.
        pattern_timeout_seconds: Statement-timeout envelope applied to a
            pattern-bearing search. Defaults to
            :data:`DEFAULT_PATTERN_TIMEOUT_SECONDS`.
    """

    patterns_enabled: bool = DEFAULT_PATTERNS_ENABLED
    pattern_timeout_seconds: float = DEFAULT_PATTERN_TIMEOUT_SECONDS

    @classmethod
    def from_ariel_config(cls, config: ARIELConfig | None) -> KeywordSearchSettings:
        """Read the module's ``settings`` block, defaults filled in.

        An absent block is the normal case and yields the defaults. A *present*
        key of the wrong type is refused rather than defaulted, for the same
        reason the hybrid module refuses one: silently ignoring
        ``patterns_enabled: "yes"`` would leave the deployment on the behaviour
        it explicitly asked to leave, which is far harder to diagnose than a
        refusal. The timeout is likewise never clamped up into range — a
        configured ``0`` means the deployment believes it disabled the
        envelope, and answering that with a silent ten seconds is worse than
        naming the key.

        Args:
            config: The loaded ARIEL configuration, or ``None``.

        Returns:
            The resolved settings.

        Raises:
            ValueError: If ``patterns_enabled`` is present but not a boolean,
                or ``pattern_timeout_seconds`` is present but not a number
                ``>= 0.001``.
        """
        module = config.search_modules.get("keyword") if config is not None else None
        settings = module.settings if module is not None else None
        if not isinstance(settings, dict):
            return cls()

        patterns_enabled = settings.get("patterns_enabled", cls.patterns_enabled)
        if not isinstance(patterns_enabled, bool):
            raise ValueError(
                f"{_SETTINGS_PREFIX}.patterns_enabled must be a boolean, got {patterns_enabled!r}"
            )

        timeout = settings.get("pattern_timeout_seconds", cls.pattern_timeout_seconds)
        # ``bool`` is a subclass of ``int``, so ``True`` would otherwise pass as
        # the number 1 — a spelling that means nothing here and is refused.
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout < 0.001:
            raise ValueError(
                f"{_SETTINGS_PREFIX}.pattern_timeout_seconds must be a number >= 0.001, "
                f"got {timeout!r}"
            )

        return cls(patterns_enabled=patterns_enabled, pattern_timeout_seconds=float(timeout))


def _balance_quotes(query: str) -> str:
    """Balance unbalanced quotes in a query string.

    Handles queries with odd numbers of double quotes by escaping
    the unbalanced quote as literal text.

    Args:
        query: User search query that may have unbalanced quotes

    Returns:
        Query with balanced quotes
    """
    quote_count = query.count('"')
    if quote_count % 2 == 0:
        return query

    # Find the position of the last unbalanced quote and escape it
    # by removing it (treating it as literal text)
    logger.warning(
        f"Unbalanced quotes detected in query, auto-balancing: {query[:50]}..."
        if len(query) > 50
        else f"Unbalanced quotes detected in query, auto-balancing: {query}"
    )

    # Strategy: Find the last quote and remove it to balance
    # This preserves the most complete quoted phrases
    last_quote_idx = query.rfind('"')
    if last_quote_idx >= 0:
        query = query[:last_quote_idx] + query[last_quote_idx + 1 :]

    return query


def parse_query(query: str) -> tuple[str, dict[str, str], list[str]]:
    """Parse user query to extract field filters, phrases, and terms.

    Args:
        query: User search query

    Returns:
        Tuple of (search_text, field_filters, phrases)
        - search_text: Query text for FTS
        - field_filters: Dict of field:value filters
        - phrases: List of quoted phrases
    """
    field_filters: dict[str, str] = {}
    phrases: list[str] = []

    query = _balance_quotes(query)

    phrase_pattern = r'"([^"]+)"'
    phrase_matches = re.findall(phrase_pattern, query)
    phrases.extend(phrase_matches)

    remaining = re.sub(phrase_pattern, "", query)

    tokens = remaining.split()
    search_tokens: list[str] = []

    for token in tokens:
        lower_token = token.lower()

        if lower_token.startswith("author:"):
            field_filters["author"] = token[7:]
        elif lower_token.startswith("date:"):
            field_filters["date"] = token[5:]
        else:
            search_tokens.append(token)

    search_text = " ".join(search_tokens)
    return search_text, field_filters, phrases


def _pattern_diagnostic(message: str) -> SearchDiagnostic:
    """Build the INFO diagnostic used for every pattern-parsing notice."""
    return SearchDiagnostic(
        level=DiagnosticLevel.INFO,
        source="keyword",
        message=message,
        category="pattern",
    )


def parse_keyword_query(query: str, *, patterns_enabled: bool = True) -> ParsedKeywordQuery:
    r"""Parse a keyword-mode query into text, filters, phrases and patterns.

    This is the whole-query parser for keyword search. It wraps -- and never
    replaces -- :func:`parse_query`, which remains the only field/phrase parser:

    1. truncate to :data:`MAX_QUERY_LENGTH` (WARNING log + WARNING diagnostic)
    2. :func:`~osprey.services.ariel_search.search.patterns.strip_trailing_question`
       so ``/SR0[1-4]C___BPM\d+/?`` still parses as a pattern
    3. :func:`~osprey.services.ariel_search.search.patterns.extract_pattern_spans`
       to lift ``/.../`` spans out of the text
    4. :func:`parse_query` on the residual text for ``search_text`` /
       ``field_filters`` / ``phrases``
    5. glob detection over the returned ``search_text``

    A glob token or ``/.../`` body that cannot be index-accelerated is *not*
    dropped: the glob token stays where it was in ``search_text`` and a rejected
    span's BODY text is appended to ``search_text``, so ``/ab/`` still reaches
    full-text search as the word ``ab``. Both cases add an INFO diagnostic naming
    the original token. Only ``//`` loses its text.

    ``pattern_spans`` lists accepted glob tokens (in token order) followed by
    accepted ``/.../`` spans (in query order).

    Args:
        query: Raw user query.
        patterns_enabled: When False, skip the trailing-``?`` strip, span
            extraction and glob detection entirely -- ``*`` is ordinary text and
            a trailing ``?`` is left for :func:`parse_query` to handle as today.
            The trailing-``?`` strip exists only so ``/regex/?`` works, so it is
            switched off with the rest of the pattern machinery.

    Returns:
        A :class:`~osprey.services.ariel_search.search.base.ParsedKeywordQuery`.
    """
    diagnostics: list[SearchDiagnostic] = []

    if len(query) > MAX_QUERY_LENGTH:
        original_length = len(query)
        query = query[:MAX_QUERY_LENGTH]
        message = f"Query truncated from {original_length} to {MAX_QUERY_LENGTH} characters"
        logger.warning(message)
        diagnostics.append(
            SearchDiagnostic(
                level=DiagnosticLevel.WARNING,
                source="keyword",
                message=message,
                category="truncation",
            )
        )

    residual = query
    raw_spans: tuple[RawSpan, ...] = ()
    if patterns_enabled:
        extraction = extract_pattern_spans(strip_trailing_question(query))
        residual = extraction.residual_text
        raw_spans = extraction.spans
        diagnostics.extend(_pattern_diagnostic(notice) for notice in extraction.notices)

    search_text, field_filters, phrases = parse_query(residual)

    pattern_spans: list[PatternSpan] = []
    if patterns_enabled:
        kept_tokens: list[str] = []

        for token in search_text.split():
            if not is_glob(token):
                kept_tokens.append(token)
                continue
            decision = accept_pattern(glob_to_are(token))
            if isinstance(decision, Accepted):
                pattern_spans.append(PatternSpan(body=decision.are, source="glob", original=token))
            else:
                kept_tokens.append(token)
                diagnostics.append(_pattern_diagnostic(f'"{token}": {decision.reason}'))

        for span in raw_spans:
            decision = accept_pattern(span.body)
            if isinstance(decision, Accepted):
                pattern_spans.append(
                    PatternSpan(body=span.body, source="regex", original=span.original)
                )
            else:
                kept_tokens.extend(span.body.split())
                diagnostics.append(_pattern_diagnostic(f'"{span.original}": {decision.reason}'))

        search_text = " ".join(kept_tokens)

    return ParsedKeywordQuery(
        search_text=search_text,
        field_filters=field_filters,
        phrases=tuple(phrases),
        pattern_spans=tuple(pattern_spans),
        diagnostics=tuple(diagnostics),
    )


def build_tsquery(search_text: str, phrases: list[str]) -> str:
    """Build PostgreSQL tsquery from parsed query components.

    Args:
        search_text: Main search text with operators
        phrases: List of exact phrases to match

    Returns:
        tsquery expression string
    """
    tsquery_parts: list[str] = []

    # Process main search text
    if search_text.strip():
        # Replace operators with PostgreSQL equivalents
        ts_text = search_text
        ts_text = re.sub(r"\bAND\b", "&", ts_text, flags=re.IGNORECASE)
        ts_text = re.sub(r"\bOR\b", "|", ts_text, flags=re.IGNORECASE)
        ts_text = re.sub(r"\bNOT\b", "!", ts_text, flags=re.IGNORECASE)

        if any(op in ts_text for op in ("&", "|", "!")):
            tsquery_parts.append("websearch_to_tsquery('english', %s)")
        else:
            tsquery_parts.append("plainto_tsquery('english', %s)")

    # Add phrase matches
    for _phrase in phrases:
        tsquery_parts.append("phraseto_tsquery('english', %s)")

    if not tsquery_parts:
        return "plainto_tsquery('english', '')"

    return " && ".join(tsquery_parts)


_BOOLEAN_OPERATOR_RE = re.compile(r"\b(?:AND|OR|NOT)\b", re.IGNORECASE)

#: Message of the INFO diagnostic reporting a boolean query searched without
#: expansion. Shared with the service, which reports the same skip under its own
#: source when it declines the expansion before dispatch.
EXPANSION_SKIPPED_MESSAGE = "expansion skipped: boolean operators present"


def has_boolean_operators(search_text: str) -> bool:
    """Whether `search_text` would take the ``websearch_to_tsquery`` path.

    Mirrors :func:`build_tsquery`: the word operators are rewritten to ``&``,
    ``|`` and ``!``, and any of those three characters routes the query to
    ``websearch_to_tsquery``. Lives beside `build_tsquery` and is public because
    the search service asks the same question one layer up, when it decides
    whether to resolve an expansion at all -- the two must never disagree about
    which queries take that path.

    Args:
        search_text: The parsed keyword search text.

    Returns:
        True when the text carries an explicit boolean operator.
    """
    if _BOOLEAN_OPERATOR_RE.search(search_text):
        return True
    return any(op in search_text for op in ("&", "|", "!"))


def _expansion_skipped_diagnostic() -> SearchDiagnostic:
    """INFO diagnostic reporting a boolean query searched without expansion."""
    return SearchDiagnostic(
        level=DiagnosticLevel.INFO,
        source="keyword",
        message=EXPANSION_SKIPPED_MESSAGE,
        category="expansion",
    )


def _name_pattern(error: PatternError, spans: tuple[PatternSpan, ...]) -> PatternError:
    """Return `error` with `pattern` filled in from the spans this search sent.

    The repository knows only PostgreSQL's message, not which of the query's
    patterns produced it. This module sent the spans, so it can name the
    original query text: the sole span when there was only one, otherwise the
    first span whose bound body appears in the message.

    Args:
        error: The error the repository raised.
        spans: The pattern spans this search bound, in the order they were bound.

    Returns:
        A `PatternError` naming the offending pattern, or `error` unchanged when
        it already names one or the culprit cannot be identified.
    """
    if error.pattern is not None or not spans:
        return error
    if len(spans) == 1:
        culprit: PatternSpan | None = spans[0]
    else:
        culprit = next((span for span in spans if span.body in error.message), None)
    if culprit is None:
        return error
    return PatternError(
        error.message,
        pattern=culprit.original,
        technical_details=dict(error.technical_details),
    )


async def keyword_search(
    query: str,
    repository: ARIELRepository,
    config: ARIELConfig,
    *,
    max_results: int = 10,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    author: str | None = None,
    source_system: str | None = None,
    include_highlights: bool = True,
    fuzzy_fallback: bool = True,
    parsed: ParsedKeywordQuery | None = None,
    query_expansion: QueryExpansion | None = None,
    **kwargs: Any,
) -> list[tuple[EnhancedLogbookEntry, float, list[str]]] | ModuleOutput:
    """Execute keyword search against the logbook database.

    Uses PostgreSQL full-text search with optional fuzzy matching fallback,
    literal ``raw_text ~* %s`` pattern predicates, and vocabulary expansion.

    **Return shape.** A call that came through the ARIEL search service -- one
    that supplied `parsed` or `query_expansion` -- gets a `ModuleOutput`
    carrying diagnostics and the expansion the executed SQL actually contained.
    A direct call that supplied neither gets the bare list of result tuples it
    has always returned. Nothing about the search itself differs between the two.

    **Parsing.** The service parses once, above dispatch, and hands the result
    in as `parsed`; only a direct caller reaches :func:`parse_keyword_query`
    here. The service parses with the pattern machinery armed, so a deployment
    that set ``patterns_enabled: false`` gets its parse redone here without it
    -- ``*`` and ``/.../`` are then ordinary full-text tokens, as configured.

    Args:
        query: Search query with optional operators (AND, OR, NOT)
            and field prefixes (author:, date:)
        repository: ARIEL database repository
        config: ARIEL configuration
        max_results: Maximum entries to return (default: 10)
        start_date: Filter entries after this time
        end_date: Filter entries before this time
        author: Filter by author name (ILIKE match)
        source_system: Filter by source system (exact match)
        include_highlights: Include highlighted snippets (default: True)
        fuzzy_fallback: Fall back to fuzzy search if no exact matches
        parsed: The service's parse of `query`. ``None`` means a direct caller,
            and this module parses `query` itself.
        query_expansion: The vocabulary expansion resolved for this request, or
            ``None`` when none applies. A query carrying explicit boolean
            operators is searched unexpanded with an INFO diagnostic, because
            ``websearch_to_tsquery`` syntax cannot group alternatives.

    Returns:
        `ModuleOutput` when `parsed` or `query_expansion` was supplied, else a
        list of (entry, score, highlights) tuples sorted by relevance.

    Raises:
        SearchTimeoutError: If a pattern-bearing statement exceeded its
            timeout envelope; the caller reports it with the expansion the
            statement carried.
        PatternError: If PostgreSQL refused to compile one of the patterns,
            re-raised naming the query text it came from.
    """
    via_service = parsed is not None or query_expansion is not None

    if not query.strip():
        return ModuleOutput(entries=[]) if via_service else []

    logger.info(
        f"keyword_search: query={query!r}, max_results={max_results}, "
        f"start_date={start_date}, end_date={end_date}"
    )

    settings = KeywordSearchSettings.from_ariel_config(config)
    if parsed is None:
        parsed = parse_keyword_query(query, patterns_enabled=settings.patterns_enabled)
    elif not settings.patterns_enabled and parsed.pattern_spans:
        # The service parses with patterns armed; this deployment turned them
        # off. Re-parse rather than drop the spans, so their original text goes
        # back into the full-text query instead of vanishing.
        parsed = parse_keyword_query(query, patterns_enabled=False)

    diagnostics: list[SearchDiagnostic] = list(parsed.diagnostics)
    search_text = parsed.search_text
    phrases = list(parsed.phrases)
    field_filters = parsed.field_filters

    expansion_applied: tuple[ExpansionGroup, ...] = ()
    tsquery_sql: str | None = None
    tsquery_params: list[Any] | None = None

    params: list[Any] = []
    where_clauses: list[str] = []

    if search_text.strip() or phrases:
        fts_expression = keyword_fts_expression(config)
        expand = query_expansion is not None and bool(query_expansion.groups)

        if expand and has_boolean_operators(search_text):
            diagnostics.append(_expansion_skipped_diagnostic())
            expand = False

        if expand and query_expansion is not None:
            tsquery_sql, tsquery_params = build_expanded_tsquery(parsed, query_expansion)
            where_clauses.append(f"{fts_expression} @@ ({tsquery_sql})")
            params.extend(tsquery_params)
            expansion_applied = query_expansion.groups
        else:
            if search_text.strip():
                params.append(search_text)

            for phrase in phrases:
                params.append(phrase)

            where_clauses.append(f"{fts_expression} @@ ({build_tsquery(search_text, phrases)})")

    for span in parsed.pattern_spans:
        where_clauses.append("raw_text ~* %s")
        params.append(span.body)

    if "author" in field_filters:
        where_clauses.append("author ILIKE %s")
        params.append(f"%{field_filters['author']}%")

    if "date" in field_filters:
        date_val = field_filters["date"]
        if len(date_val) == 7:
            where_clauses.append("timestamp >= %s AND timestamp < %s")
            params.append(f"{date_val}-01")
            year, month = int(date_val[:4]), int(date_val[5:7])
            if month == 12:
                next_month = f"{year + 1}-01-01"
            else:
                next_month = f"{year}-{month + 1:02d}-01"
            params.append(next_month)
        else:
            where_clauses.append("DATE(timestamp) = %s")
            params.append(date_val)

    if start_date:
        where_clauses.append("timestamp >= %s")
        params.append(start_date)
    if end_date:
        where_clauses.append("timestamp <= %s")
        params.append(end_date)

    if author and "author" not in field_filters:
        where_clauses.append("author ILIKE %s")
        params.append(f"%{author}%")
    if source_system:
        where_clauses.append("source_system = %s")
        params.append(source_system)

    # Only a search that actually emitted a ``~*`` predicate asks for the
    # timeout envelope; without the keyword the repository runs exactly the
    # statement it runs today, outside any transaction.
    extra: dict[str, Any] = {}
    if tsquery_sql is not None:
        extra["tsquery_sql"] = tsquery_sql
        extra["tsquery_params"] = tsquery_params
    if parsed.pattern_spans:
        extra["pattern_timeout_seconds"] = settings.pattern_timeout_seconds

    probe_text = search_text or " ".join(phrases)

    try:
        results = await repository.keyword_search(
            where_clauses=where_clauses,
            params=params,
            search_text=probe_text,
            max_results=max_results,
            include_highlights=include_highlights,
            **extra,
        )
    except PatternError as e:
        raise _name_pattern(e, parsed.pattern_spans) from e

    # Fuzzy fallback probes the text the operator typed first, so expansion can
    # only ever add hits, never remove one. A pattern search never falls back: a
    # literal pattern that matched nothing means nothing matched, not that the
    # spelling was close.
    if not results and fuzzy_fallback and probe_text.strip() and not parsed.pattern_spans:
        results = await repository.fuzzy_search(
            search_text=probe_text,
            threshold=DEFAULT_FUZZY_THRESHOLD,
            max_results=max_results,
            start_date=start_date,
            end_date=end_date,
        )
        if not results and expansion_applied and query_expansion is not None:
            results = await repository.fuzzy_search(
                search_text=query_expansion.flattened_text,
                threshold=DEFAULT_FUZZY_THRESHOLD,
                max_results=max_results,
                start_date=start_date,
                end_date=end_date,
            )

    logger.info(f"keyword_search: returning {len(results)} results")
    if via_service:
        return ModuleOutput(
            entries=results,
            diagnostics=tuple(diagnostics),
            expansion=expansion_applied,
        )
    return results


class KeywordSearchInput(BaseModel):
    """Input schema for keyword search tool."""

    query: str = Field(
        description="Search terms. Supports phrases in quotes, AND/OR/NOT operators."
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
    expand_query: bool | None = Field(
        default=None,
        description=(
            "Apply the facility vocabulary expansion (shorthand/acronyms). "
            "None = the configured default (capabilities().vocabulary.expand_by_default)"
        ),
    )


def format_keyword_result(
    entry: EnhancedLogbookEntry,
    score: float,
    highlights: list[str],
) -> dict[str, Any]:
    """Format a keyword search result for agent consumption.

    Args:
        entry: EnhancedLogbookEntry
        score: Relevance score
        highlights: Highlighted snippets

    Returns:
        Formatted dict for agent
    """
    from osprey.services.ariel_search.models import _format_entry_base

    return {**_format_entry_base(entry), "score": score, "highlights": highlights}


def get_parameter_descriptors() -> list[ParameterDescriptor]:
    """Return tunable parameter descriptors for the capabilities API."""
    return [
        ParameterDescriptor(
            name="include_highlights",
            label="Include Highlights",
            description="Include highlighted snippets in search results",
            param_type="bool",
            default=True,
            section="Options",
        ),
        ParameterDescriptor(
            name="fuzzy_fallback",
            label="Fuzzy Fallback",
            description="Fall back to fuzzy matching when no exact matches are found",
            param_type="bool",
            default=True,
            section="Options",
        ),
    ]


def get_tool_descriptor() -> SearchToolDescriptor:
    """Return the descriptor for auto-discovery by the agent executor."""
    return SearchToolDescriptor(
        name="keyword_search",
        description=(
            "Fast text-based lookup using full-text search. "
            "Use for specific terms, equipment names, PV names, or phrases. "
            "Supports quoted phrases and AND/OR/NOT operators."
        ),
        search_mode="keyword",
        args_schema=KeywordSearchInput,
        execute=keyword_search,
        format_result=format_keyword_result,
        accepts_expansion=True,
        query_parser=parse_keyword_query,
    )
