"""Base types for ARIEL search module auto-discovery.

Search modules export a `get_tool_descriptor()` function that returns
a `SearchToolDescriptor`. The agent executor uses these descriptors
to build LangChain tools automatically — no executor changes needed
when adding a new search module.

Modules may also export `get_parameter_descriptors()` to declare
tunable parameters for the frontend capabilities API.

This module is the public extension surface for third-party search modules:
`SearchToolDescriptor` declares what a module accepts, `ModuleOutput` is the
richer return shape a module may use instead of a bare list of entries, and
`ParsedKeywordQuery` / `QueryExpansion` are the argument types the service
hands to modules that opt in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypeVar

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from pydantic import BaseModel

    from osprey.services.ariel_search.models import SearchDiagnostic

_EntryT = TypeVar("_EntryT")


@dataclass(frozen=True)
class ParameterDescriptor:
    """Describes a tunable parameter for the frontend capabilities API.

    Attributes:
        name: Parameter key (e.g. "similarity_threshold")
        label: Human-readable label (e.g. "Similarity Threshold")
        description: Help text for the parameter
        param_type: One of "float", "int", "bool", "select", "date", "text",
            "dynamic_select"
        default: Default value
        min_value: Minimum value (float/int types)
        max_value: Maximum value (float/int types)
        step: Step increment (float/int types)
        options: Choices for select type, e.g. [{"value": "rrf", "label": "RRF"}]
        section: Grouping label in the advanced panel (e.g. "Retrieval")
        placeholder: Placeholder text for text inputs
        options_endpoint: API endpoint for dynamic_select to fetch options
    """

    name: str
    label: str
    description: str
    param_type: str  # "float", "int", "bool", "select", "date", "text", "dynamic_select"
    default: Any
    min_value: float | None = None
    max_value: float | None = None
    step: float | None = None
    options: list[dict[str, str]] | None = None
    section: str = "General"
    placeholder: str | None = None
    options_endpoint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict."""
        d: dict[str, Any] = {
            "name": self.name,
            "label": self.label,
            "description": self.description,
            "type": self.param_type,
            "default": self.default,
            "section": self.section,
        }
        if self.min_value is not None:
            d["min"] = self.min_value
        if self.max_value is not None:
            d["max"] = self.max_value
        if self.step is not None:
            d["step"] = self.step
        if self.options is not None:
            d["options"] = self.options
        if self.placeholder is not None:
            d["placeholder"] = self.placeholder
        if self.options_endpoint is not None:
            d["options_endpoint"] = self.options_endpoint
        return d


@dataclass(frozen=True)
class PatternSpan:
    """A literal-pattern predicate extracted from a search query.

    A pattern span is bound to the database as ``raw_text ~* %s`` with `body`
    as the single parameter, so `body` must already be a valid PostgreSQL
    advanced regular expression (ARE). Glob tokens are translated to an ARE
    before they reach this type; `/.../` spans carry their body verbatim.

    Attributes:
        body: PostgreSQL ARE to bind to a ``raw_text ~* %s`` predicate
        source: Where the span came from -- ``"glob"`` for a ``*`` token,
            ``"regex"`` for a ``/.../`` span
        original: The verbatim query text the span was extracted from
            (e.g. ``"SR01C___BPM*"``), used for diagnostics and for
            re-inserting a rejected pattern as plain text
    """

    body: str
    source: Literal["glob", "regex"]
    original: str


@dataclass(frozen=True)
class ParsedKeywordQuery:
    """The result of parsing a keyword-mode query into its components.

    Produced by a module's declared `SearchToolDescriptor.query_parser` and
    handed to the module as the `parsed=` keyword argument. Everything a
    keyword module needs to build its SQL is here -- the module never
    re-parses.

    Frozen only prevents attribute rebinding: `field_filters` is an ordinary
    dict and MUST be treated as read-only by every consumer. Copy it before
    mutating.

    Attributes:
        search_text: Query text left after field filters, quoted phrases and
            pattern spans were removed -- the text that drives full-text search
        field_filters: Field-scoped filters keyed by prefix without the colon
            (e.g. ``{"author": "jsmith"}``); read-only, do not mutate
        phrases: Quoted phrases in query order, without their quotes
        pattern_spans: Accepted literal patterns to bind as ``raw_text ~* %s``
        diagnostics: Parse-time diagnostics (e.g. a pattern rejected as too
            generic and searched as text instead)
    """

    search_text: str
    field_filters: dict[str, str] = field(default_factory=dict)
    phrases: tuple[str, ...] = ()
    pattern_spans: tuple[PatternSpan, ...] = ()
    diagnostics: tuple[SearchDiagnostic, ...] = ()


@dataclass(frozen=True)
class ExpansionGroup:
    """One expanded span of a query and the alternatives it expanded to.

    This is the transparency unit surfaced to callers as ``expanded_terms``:
    one group per matched span, carrying the text the user typed and the
    alternative forms searched alongside it.

    Attributes:
        original: The span of query text that matched a vocabulary concept
        alternatives: Alternative forms searched in addition to `original`,
            excluding `original` itself
    """

    original: str
    alternatives: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict.

        Returns:
            ``{"original": <str>, "alternatives": [<str>, ...]}``
        """
        return {"original": self.original, "alternatives": list(self.alternatives)}


@dataclass(frozen=True)
class QueryExpansion:
    """A resolved vocabulary expansion of one query.

    The service resolves this once per search and passes it to every module
    whose descriptor sets `SearchToolDescriptor.accepts_expansion`, as the
    `query_expansion=` keyword argument. Modules that match term-by-term use
    `groups`; modules that embed or rerank whole text use `flattened_text`.

    Attributes:
        groups: One group per matched span, in query order
        flattened_text: The query rewritten with every group's alternatives
            folded in -- the text an embedding or reranking module should use
    """

    groups: tuple[ExpansionGroup, ...]
    flattened_text: str


@dataclass(frozen=True)
class ModuleOutput:
    """Richer return shape for a search module.

    A search module may return either the bare list of results it returns
    today -- the existing contract, unchanged and still fully supported -- or a
    `ModuleOutput`, which carries diagnostics and expansion transparency
    alongside those same results.

    Attributes:
        entries: The module's results, exactly as a bare-list return would
            provide them (``(entry, score)`` or ``(entry, score, highlights)``
            tuples for the built-in modules)
        diagnostics: Diagnostics to surface to the caller alongside results
        expansion: The expansion groups the executed search actually contained,
            surfaced to callers as ``expanded_terms``. Empty when the module
            ran without expansion, even if one was offered.
    """

    entries: list[Any]
    diagnostics: tuple[SearchDiagnostic, ...] = ()
    expansion: tuple[ExpansionGroup, ...] = ()


def module_result(
    entries: list[_EntryT], query_expansion: QueryExpansion | None
) -> list[_EntryT] | ModuleOutput:
    """Return a module's entries in the shape its caller's contract asks for.

    The rule the built-in expansion-aware modules follow, in one place: a
    caller that passed no expansion gets the bare list of result tuples it has
    always received, and a caller that passed one gets a `ModuleOutput`
    carrying those same tuples plus the groups the executed search actually
    contained. A module that also reports diagnostics builds its `ModuleOutput`
    itself, so this is a convenience for the simple case rather than part of
    the documented extension surface (it is deliberately absent from
    ``__all__``).

    Args:
        entries: The result tuples the search produced.
        query_expansion: The expansion the caller passed, or None.

    Returns:
        `entries` unchanged, or a `ModuleOutput` wrapping it.
    """
    if query_expansion is None:
        return entries
    return ModuleOutput(entries=entries, expansion=query_expansion.groups)


@dataclass(frozen=True)
class SearchToolDescriptor:
    """Everything the agent executor needs to wrap a search module as a tool.

    Attributes:
        name: Tool name for LangChain (e.g. "keyword_search")
        description: Tool description shown to the LLM
        search_mode: Registry module name this tool dispatches to
            (e.g. "keyword"), as accepted in a search request's ``modes``
        args_schema: Pydantic model for tool input validation
        execute: Async function that performs the search
        format_result: Formats raw search results for the agent
        needs_embedder: Whether this tool requires an embedding provider
        accepts_expansion: Whether this module accepts a resolved vocabulary
            expansion. When true, the service passes `query_expansion=`, a
            `QueryExpansion`, but only when an expansion was actually resolved
            for the request -- the argument is never passed as ``None``.
            Modules that leave this false are called exactly as they are today.
        query_parser: Optional pure function turning the raw query string into
            a `ParsedKeywordQuery`. When set, the service calls it once per
            search and passes the result as `parsed=`; when unset, the module
            receives no `parsed=` argument and parses the query itself if it
            needs to. A module supplies its own parser or none -- it never
            receives another module's parse.
    """

    name: str
    description: str
    search_mode: str
    args_schema: type[BaseModel]
    execute: Callable[..., Awaitable[Any]]
    format_result: Callable[..., dict[str, Any]]
    needs_embedder: bool = False
    accepts_expansion: bool = False
    query_parser: Callable[[str], ParsedKeywordQuery] | None = None


__all__ = [
    "ExpansionGroup",
    "ModuleOutput",
    "ParameterDescriptor",
    "ParsedKeywordQuery",
    "PatternSpan",
    "QueryExpansion",
    "SearchToolDescriptor",
]
