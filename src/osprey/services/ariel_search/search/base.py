"""Base types for ARIEL search module auto-discovery.

Search modules export a `get_tool_descriptor()` function that returns
a `SearchToolDescriptor`. The agent executor uses these descriptors
to build LangChain tools automatically — no executor changes needed
when adding a new search module.

Modules may also export `get_parameter_descriptors()` to declare
tunable parameters for the frontend capabilities API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from pydantic import BaseModel

    from osprey.services.ariel_search.models import SearchMode


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
class SearchToolDescriptor:
    """Everything the agent executor needs to wrap a search module as a tool.

    Attributes:
        name: Tool name for LangChain (e.g. "keyword_search")
        description: Tool description shown to the LLM
        search_mode: Corresponding SearchMode enum value
        args_schema: Pydantic model for tool input validation
        execute: Async function that performs the search
        format_result: Formats raw search results for the agent
        needs_embedder: Whether this tool requires an embedding provider
    """

    name: str
    description: str
    search_mode: SearchMode
    args_schema: type[BaseModel]
    execute: Callable[..., Awaitable[Any]]
    format_result: Callable[..., dict[str, Any]]
    needs_embedder: bool = False


__all__ = ["ParameterDescriptor", "SearchToolDescriptor"]
