"""Shared envelope pieces for the ARIEL search tools.

The three search tools take the same filter arguments, apply the same
over-fetch-then-exclude window, return the same flat success envelope and
report the same two statement-level faults, so all of that lives here rather
than three times over.

A fault is a search whose statement never ran to completion -- a pattern
PostgreSQL refused to compile, or a query the database cancelled on the pattern
timeout. The service reports both as an empty result carrying one ERROR
diagnostic, which on the wire is indistinguishable from "nothing matched". An
agent must not read either as "the corpus holds no answer", so the tools turn
them into error envelopes that name the offending pattern and still carry the
vocabulary expansion the failed statement contained.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NoReturn

from osprey.mcp_server.ariel.server import make_error, serialize_entry

if TYPE_CHECKING:
    from collections.abc import Iterable

    from osprey.services.ariel_search.exceptions import (
        PatternError,
        SearchTimeoutError,
        VocabularyError,
    )

#: Diagnostic category -> the ``error_type`` the tools report it under.
_FAULT_ERROR_TYPES = {
    "timeout": "search_timeout",
    "pattern": "invalid_pattern",
}

#: Operator/agent suggestions per fault, most actionable first.
_FAULT_SUGGESTIONS: dict[str, list[str]] = {
    "search_timeout": [
        "Narrow the search with a date range (start_date/end_date) or fewer terms.",
        "Pattern tokens (globs and /regex/) are the expensive part -- drop or tighten them.",
        "Raise ariel.search_modules.keyword.settings.pattern_timeout_seconds if the "
        "pattern is legitimate and the corpus is large.",
    ],
    "invalid_pattern": [
        "Fix the /regex/ token -- PostgreSQL rejected it, commonly an unclosed "
        "bracket, group or quantifier.",
        "Search the term as plain text, or use a * glob instead of an explicit /regex/.",
    ],
}


#: Characters of ``raw_text`` each entry carries into a search envelope.
_ENTRY_TEXT_LIMIT = 500


def advanced_params(
    *,
    author: str | None = None,
    source_system: str | None = None,
    similarity_threshold: float | None = None,
    expand_query: bool | None = None,
    rerank: bool | None = None,
) -> dict[str, Any]:
    """Build the ``advanced_params`` mapping a search request carries.

    Only arguments the caller actually supplied are put in: the service reads
    an absent key as "no preference" and a present one as an explicit choice,
    so passing ``None`` through would turn every default into an override.

    Args:
        author: Author filter, omitted when empty.
        source_system: Source-system filter, omitted when empty.
        similarity_threshold: Semantic threshold, omitted when not given.
        expand_query: Vocabulary-expansion preference, omitted when not given
            so the deployment's ``expand_by_default`` decides.
        rerank: Reranking preference, omitted when not given so the
            deployment's ``search_modules.hybrid.settings.rerank`` decides.

    Returns:
        The mapping to hand to :meth:`ARIELSearchService.search`.
    """
    params: dict[str, Any] = {}
    if author:
        params["author"] = author
    if source_system:
        params["source_system"] = source_system
    if similarity_threshold is not None:
        params["similarity_threshold"] = similarity_threshold
    if expand_query is not None:
        params["expand_query"] = expand_query
    if rerank is not None:
        params["rerank"] = rerank
    return params


@dataclass(frozen=True)
class ResultWindow:
    """The over-fetch-then-exclude window an iterative search runs in.

    Excluded entries are removed after ranking, not in the database, so the
    search has to ask for enough extra rows to refill what the exclusion takes
    away. That makes the fetch count and the post-filter two halves of one
    rule; keeping them in one object is what stops a tool from over-fetching
    against one exclusion set and filtering against another.

    Attributes:
        max_results: Entries the caller asked for.
        excluded: Entry IDs to drop from the ranking.
    """

    max_results: int
    excluded: frozenset[str]

    @classmethod
    def build(cls, max_results: int, exclude_entry_ids: list[str] | None) -> ResultWindow:
        """Create the window for one search call.

        Args:
            max_results: Entries the caller asked for.
            exclude_entry_ids: Entry IDs to exclude, or None.

        Returns:
            The window.
        """
        return cls(max_results=max_results, excluded=frozenset(exclude_entry_ids or ()))

    @property
    def fetch_count(self) -> int:
        """How many entries to ask the service for."""
        return self.max_results + len(self.excluded) if self.excluded else self.max_results

    def select(self, entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop the excluded entries, truncate, and serialize for the envelope.

        Args:
            entries: The service's ranked entries.

        Returns:
            At most ``max_results`` serialized entries, in ranking order.
        """
        kept = [e for e in entries if e["entry_id"] not in self.excluded][: self.max_results]
        return [serialize_entry(e, text_limit=_ENTRY_TEXT_LIMIT) for e in kept]


def success_envelope(
    query: str, mode: str, result: Any, entries: list[dict[str, Any]]
) -> dict[str, Any]:
    """Build the flat success envelope every search tool returns.

    ``diagnostics`` is not here because each tool attaches it after the fact:
    all three do so with ``diagnostics(result)``, but ``hybrid_search`` first
    promotes its own sidecar and configuration faults to an error envelope, so
    the diagnostics it attaches are always ones a successful search reported.

    Args:
        query: The query text as the caller typed it.
        mode: The search mode this tool ran.
        result: The service's search result.
        entries: The serialized entries, already windowed.

    Returns:
        The envelope, ready to ``json.dumps``.
    """
    return {
        "query": query,
        "mode": mode,
        "results_found": len(entries),
        "reasoning": result.reasoning,
        "sources": list(result.sources),
        "entries": entries,
        "expanded_terms": expanded_terms(result),
    }


def expanded_terms(result: object) -> list[dict[str, Any]]:
    """Serialize the vocabulary expansion a search actually applied.

    Args:
        result: The service's search result.

    Returns:
        One ``{"original": ..., "alternatives": [...]}`` mapping per expanded
        span, empty when nothing was expanded. Defensive against a result whose
        ``expanded_terms`` is absent or not a sequence, so the envelope stays
        JSON-serializable whatever the caller handed in.
    """
    terms = getattr(result, "expanded_terms", ()) or ()
    if not isinstance(terms, (list, tuple)):
        return []
    return [dict(term) for term in terms if isinstance(term, dict)]


def diagnostics(result: object) -> list[dict[str, Any]]:
    """Serialize a search's structured diagnostics for the envelope.

    Args:
        result: The service's search result.

    Returns:
        One mapping per diagnostic with the level as its string value, empty
        when the search reported none.
    """
    out: list[dict[str, Any]] = []
    for diagnostic in _iter_diagnostics(result):
        level = getattr(getattr(diagnostic, "level", None), "value", None)
        if level is None:
            continue
        out.append(
            {
                "level": level,
                "source": getattr(diagnostic, "source", ""),
                "message": getattr(diagnostic, "message", ""),
                "category": getattr(diagnostic, "category", None),
            }
        )
    return out


def statement_fault(result: object) -> tuple[str, str] | None:
    """Return the fault an ERROR diagnostic reports, or ``None``.

    Args:
        result: The service's search result.

    Returns:
        ``(error_type, message)`` for the first ERROR diagnostic whose category
        says the statement never completed, else ``None``. A module error of
        the generic ``search`` category is deliberately not a fault here --
        ``hybrid_search`` maps that one to its own sidecar advice.
    """
    from osprey.services.ariel_search.models import DiagnosticLevel

    for diagnostic in _iter_diagnostics(result):
        if getattr(diagnostic, "level", None) is not DiagnosticLevel.ERROR:
            continue
        error_type = _FAULT_ERROR_TYPES.get(getattr(diagnostic, "category", None) or "")
        if error_type is None:
            continue
        return error_type, str(getattr(diagnostic, "message", "") or "no detail reported")
    return None


def raise_on_statement_fault(result: object, mode: str) -> None:
    """Raise the standard error envelope when the statement never completed.

    Args:
        result: The service's search result.
        mode: The search mode, named in the error message.

    Raises:
        ToolError: Carrying the standard envelope, whose ``details`` repeat the
            expansion and the diagnostics so the caller can see what the failed
            statement contained.
    """
    fault = statement_fault(result)
    if fault is None:
        return
    error_type, message = fault
    make_error(
        error_type,
        f"ARIEL {mode} search failed: {message}",
        _FAULT_SUGGESTIONS[error_type],
        details={
            "expanded_terms": expanded_terms(result),
            "diagnostics": diagnostics(result),
        },
    )


def raise_for_fault_exception(exc: PatternError | SearchTimeoutError, mode: str) -> NoReturn:
    """Report a fault that reached the tool as an exception.

    The service converts both faults into diagnostics on an empty result, so
    this path runs only for a caller that bypassed it (a direct module call, a
    third-party module raising past the service). Keeping it means the agent
    gets the same envelope either way instead of a generic internal error.

    Args:
        exc: The pattern or timeout error.
        mode: The search mode, named in the error message.

    Raises:
        ToolError: Carrying the standard envelope. ``expanded_terms`` is empty
            here -- an exception does not carry the expansion the statement had.
    """
    from osprey.services.ariel_search.exceptions import PatternError

    if isinstance(exc, PatternError):
        error_type = "invalid_pattern"
        detail = f"invalid pattern {exc.pattern!r}: {exc}" if exc.pattern else str(exc)
    else:
        error_type = "search_timeout"
        detail = str(exc)
    make_error(
        error_type,
        f"ARIEL {mode} search failed: {detail}",
        _FAULT_SUGGESTIONS[error_type],
        details={"expanded_terms": [], "diagnostics": []},
    )


def raise_for_vocabulary_error(exc: VocabularyError, mode: str) -> NoReturn:
    """Report a failed facility-vocabulary load as the standard error envelope.

    The vocabulary loads once at config parse; when it failed, every search on
    the affected deployment would otherwise fall through to a generic
    "internal_error" with no clue that the fault is a fixable configuration
    problem. This turns it into a "service_unavailable" envelope naming the
    config key, the first loader error, and the remedy that clears it, so the
    three search tools agree with each other -- and with the web API's
    ``VocabularyError`` handler -- on how this failure reads.

    Args:
        exc: The vocabulary load failure.
        mode: The search mode, named in the error message.

    Raises:
        ToolError: Carrying error_type "service_unavailable", whose message
            names ``exc.config_key`` and the first loader error, and whose
            only suggestion is ``exc.remedy``.
    """
    first = exc.errors[0] if exc.errors else exc.message
    make_error(
        "service_unavailable",
        f"ARIEL {mode} search is unavailable: {exc.config_key}: {first}",
        [exc.remedy],
    )


def _iter_diagnostics(result: object) -> tuple[Any, ...]:
    """Return a result's diagnostics as a tuple, tolerating an absent field."""
    found = getattr(result, "diagnostics", ()) or ()
    if not isinstance(found, (list, tuple)):
        return ()
    return tuple(found)


__all__ = [
    "ResultWindow",
    "advanced_params",
    "diagnostics",
    "expanded_terms",
    "raise_for_fault_exception",
    "raise_for_vocabulary_error",
    "raise_on_statement_fault",
    "statement_fault",
    "success_envelope",
]
