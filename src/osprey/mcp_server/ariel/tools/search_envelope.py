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

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NoReturn

from osprey.mcp_server.ariel.server import make_error, serialize_entry
from osprey.services.ariel_search.attachments.compose import caption_model_id
from osprey.services.ariel_search.attachments.summaries import file_source_for

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

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


def advanced_params(
    *,
    author: str | None = None,
    source_system: str | None = None,
    similarity_threshold: float | None = None,
    expand_query: bool | None = None,
    rerank: bool | None = None,
    include_images: bool | None = None,
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
        include_images: Picture-search preference, omitted when not given so
            the service decides from ``image_embedding.enabled``.

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
    if include_images is not None:
        params["include_images"] = include_images
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

    @property
    def image_only_cap(self) -> int:
        """Most entries matched only through a picture that one page may hold."""
        return math.ceil(self.max_results / 3)

    def select(self, entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop the excluded entries and keep at most `max_results`.

        Image-only entries (``_matched_via == ["image"]``) beyond
        :attr:`image_only_cap` are dropped before the page is cut, so text hits
        fetched for the window refill the page the way they refill an
        exclusion.

        Args:
            entries: The service's ranked entries.

        Returns:
            At most ``max_results`` of the raw entry dicts, in ranking order;
            :func:`serialize_entries` turns them into the listing shape.
        """
        kept: list[dict[str, Any]] = []
        image_only = 0
        for entry in entries:
            if entry["entry_id"] in self.excluded:
                continue
            if entry.get("_matched_via") == ["image"]:
                if image_only >= self.image_only_cap:
                    continue
                image_only += 1
            kept.append(entry)
        return kept[: self.max_results]


async def serialize_entries(
    entries: Sequence[Mapping[str, Any]],
    *,
    text_limit: int,
    attachment_limit: int,
    repository: Any,
    model_id: str | None,
    file_source: bool,
    full_captions: bool = False,
    view_enabled: bool = True,
) -> list[dict[str, Any]]:
    """Serialize a page of entries with their attachment summaries.

    The attachment rows of the whole page come from one
    ``repository.get_attachment_rows`` call, made only when there is an entry
    and summaries are wanted. A ``DatabaseQueryError`` from that reader is
    treated as a store without copy state: every entry keeps its fallback
    summaries and the process logs the schema-gap warning once, so a failing
    reader never costs the caller a result.

    Args:
        entries: The raw entry dicts, in output order.
        text_limit: Characters of each entry's text to include.
        attachment_limit: Attachment summaries per entry; ``0`` omits them.
        repository: The ARIEL repository the rows are read from.
        model_id: The configured caption model id (``caption_model_id``).
        file_source: Whether the source resolves relative attachment paths
            (``file_source_for``).
        full_captions: Emit captions and visible text uncut.
        view_enabled: ``ariel.attachments.view.enabled``; false reads no rows
            and emits the entries without any attachment keys.

    Returns:
        One serialized dict per entry, in the given order.

    Raises:
        TypeError: If the reader returned something other than None or a dict.
    """
    from osprey.services.ariel_search.database.repository import read_attachment_rows

    if not entries:
        return []
    mapping: Mapping[str, Any] | None = {}
    if view_enabled and attachment_limit > 0:
        mapping = await read_attachment_rows(repository, [e["entry_id"] for e in entries])
    return [
        serialize_entry(
            entry,
            text_limit=text_limit,
            attachment_limit=attachment_limit,
            attachment_rows=None if mapping is None else mapping.get(entry["entry_id"], []),
            model_id=model_id,
            file_source=file_source,
            full_captions=full_captions,
            view_enabled=view_enabled,
        )
        for entry in entries
    ]


async def serialize_page(
    entries: Sequence[Mapping[str, Any]],
    config: Any,
    repository: Any,
    *,
    text_limit: int,
) -> list[dict[str, Any]]:
    """Serialize a page of entries the way every listing tool does.

    :func:`serialize_entries` with the attachment limit, caption model, file
    source and view switch all taken from the deployment's ARIEL config.

    Args:
        entries: The raw entry dicts, in output order.
        config: The ARIEL config.
        repository: The ARIEL repository the rows are read from.
        text_limit: Characters of each entry's text to include.

    Returns:
        One serialized dict per entry, in the given order.
    """
    return await serialize_entries(
        entries,
        text_limit=text_limit,
        attachment_limit=config.entry_text.listing_attachments,
        repository=repository,
        model_id=caption_model_id(config),
        file_source=file_source_for(config),
        view_enabled=config.attachments.view_enabled,
    )


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
    return _serialize_diagnostics(_iter_diagnostics(result))


async def envelope_diagnostics(result: object, repository: Any) -> list[dict[str, Any]]:
    """Serialize a search's diagnostics followed by the store's schema-behind ones.

    Args:
        result: The service's search result.
        repository: The ARIEL repository whose schema is checked.

    Returns:
        :func:`diagnostics` of ``result``, then one mapping per
        schema-behind diagnostic the store reports.
    """
    from osprey.services.ariel_search.database.repository import schema_behind_diagnostics

    schema_behind = await schema_behind_diagnostics(repository)
    return diagnostics(result) + _serialize_diagnostics(schema_behind)


def _serialize_diagnostics(found: Iterable[Any]) -> list[dict[str, Any]]:
    """Serialize diagnostics, dropping any without a level."""
    out: list[dict[str, Any]] = []
    for diagnostic in found:
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
    "envelope_diagnostics",
    "expanded_terms",
    "raise_for_fault_exception",
    "raise_for_vocabulary_error",
    "raise_on_statement_fault",
    "serialize_entries",
    "serialize_page",
    "statement_fault",
    "success_envelope",
]
