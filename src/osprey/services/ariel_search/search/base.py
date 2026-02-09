"""Base types for ARIEL search module auto-discovery.

Search modules export a `get_tool_descriptor()` function that returns
a `SearchToolDescriptor`. The agent executor uses these descriptors
to build LangChain tools automatically — no executor changes needed
when adding a new search module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from pydantic import BaseModel

    from osprey.services.ariel_search.models import SearchMode


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


__all__ = ["SearchToolDescriptor"]
