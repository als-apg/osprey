"""Protocol definitions for ARIEL RAP pipeline stages.

These protocols define the interfaces that pipeline components must implement.
Using Protocol (structural subtyping) for flexibility and duck typing support.

See 05_RAP_ABSTRACTION.md Sections 2.2-2.5 for specification.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from osprey.services.ariel_search.pipeline.types import (
        AssembledContext,
        AssemblyConfig,
        FormattedResponse,
        ProcessedResult,
        ProcessorConfig,
        RetrievalConfig,
        RetrievedItem,
    )


@runtime_checkable
class Retriever(Protocol):
    """Protocol for retriever implementations.

    Retrievers fetch relevant items from the database based on the query.

    Implementations:
        - KeywordRetriever: PostgreSQL full-text search
        - SemanticRetriever: Embedding similarity search
        - HybridRetriever: Combines multiple retrievers
    """

    @property
    def name(self) -> str:
        """Unique identifier for this retriever.

        Returns:
            Name string (e.g., "keyword", "semantic", "hybrid")
        """
        ...

    async def retrieve(
        self,
        query: str,
        config: RetrievalConfig,
    ) -> list[RetrievedItem]:
        """Retrieve items matching the query.

        Args:
            query: Search query string
            config: Retrieval configuration

        Returns:
            List of retrieved items sorted by score descending
        """
        ...


@runtime_checkable
class FusionStrategy(Protocol):
    """Protocol for result fusion strategies.

    Fusion strategies combine results from multiple retrievers.

    Implementations:
        - RRFFusion: Reciprocal Rank Fusion
        - WeightedFusion: Score-weighted combination
    """

    def fuse(
        self,
        results: list[list[RetrievedItem]],
    ) -> list[RetrievedItem]:
        """Fuse results from multiple retrievers.

        Args:
            results: List of result lists, one per retriever

        Returns:
            Combined and re-ranked list of retrieved items
        """
        ...


@runtime_checkable
class Assembler(Protocol):
    """Protocol for assembler implementations.

    Assemblers take retrieved items and prepare them for processing.

    Implementations:
        - TopKAssembler: Simple top-K selection
        - ContextWindowAssembler: Token-aware assembly for RAG
    """

    def assemble(
        self,
        items: list[RetrievedItem],
        config: AssemblyConfig,
    ) -> AssembledContext:
        """Assemble items into context for processing.

        Args:
            items: Retrieved items to assemble
            config: Assembly configuration

        Returns:
            Assembled context ready for processing
        """
        ...


@runtime_checkable
class Processor(Protocol):
    """Protocol for processor implementations.

    Processors transform assembled context into a result.

    The processor type indicates the reasoning complexity:
        - 'identity': Pass through unchanged
        - 'single_llm': One LLM call
        - 'agent': Multi-step ReAct loops

    Implementations:
        - IdentityProcessor: Returns context unchanged
        - SingleLLMProcessor: One LLM call for RAG
    """

    @property
    def processor_type(self) -> str:
        """Type of processor: 'identity', 'single_llm', or 'agent'.

        Returns:
            Processor type string
        """
        ...

    async def process(
        self,
        query: str,
        context: AssembledContext,
        config: ProcessorConfig,
    ) -> ProcessedResult:
        """Process the context to produce a result.

        Args:
            query: Original query string
            context: Assembled context from assembler
            config: Processing configuration

        Returns:
            Processed result with answer and metadata
        """
        ...


@runtime_checkable
class Formatter(Protocol):
    """Protocol for formatter implementations.

    Formatters transform processed results into the final response format.

    Implementations:
        - CitationFormatter: Text with [#id] citations
        - JSONFormatter: Structured JSON for API responses
    """

    def format(
        self,
        result: ProcessedResult,
        config: dict,
    ) -> FormattedResponse:
        """Format the result for output.

        Args:
            result: Processed result to format
            config: Format-specific configuration

        Returns:
            Formatted response ready for return
        """
        ...


__all__ = [
    "Assembler",
    "Formatter",
    "FusionStrategy",
    "Processor",
    "Retriever",
]
