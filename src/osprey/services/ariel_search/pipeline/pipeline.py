"""Pipeline composition for ARIEL RAP abstraction.

The Pipeline class composes retriever, assembler, processor, and formatter
stages into an executable pipeline.

See 05_RAP_ABSTRACTION.md Section 5.4 for specification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from osprey.services.ariel_search.pipeline.assemblers import TopKAssembler
from osprey.services.ariel_search.pipeline.formatters import JSONFormatter
from osprey.services.ariel_search.pipeline.processors import IdentityProcessor
from osprey.services.ariel_search.pipeline.types import (
    AssemblyConfig,
    FormattedResponse,
    ProcessorConfig,
    RetrievalConfig,
)

if TYPE_CHECKING:
    from osprey.services.ariel_search.pipeline.protocols import (
        Assembler,
        Formatter,
        Processor,
        Retriever,
    )

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution.

    Aggregates configuration for all pipeline stages.

    Attributes:
        retrieval: Configuration for retrieval stage
        assembly: Configuration for assembly stage
        processor: Configuration for processing stage
        formatter: Configuration for formatting stage
    """

    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    assembly: AssemblyConfig = field(default_factory=AssemblyConfig)
    processor: ProcessorConfig = field(default_factory=ProcessorConfig)
    formatter: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Result from pipeline execution.

    Contains the final response and intermediate results for debugging.

    Attributes:
        response: Final formatted response
        retrieval_count: Number of items retrieved
        assembly_count: Number of items assembled
        processor_type: Type of processor used
        truncated: Whether context was truncated
    """

    response: FormattedResponse
    retrieval_count: int
    assembly_count: int
    processor_type: str
    truncated: bool


class Pipeline:
    """Composable search pipeline.

    Composes retriever, assembler, processor, and formatter stages
    into an executable pipeline.

    Example:
        pipeline = Pipeline(
            retriever=KeywordRetriever(repository, config),
            assembler=TopKAssembler(),
            processor=IdentityProcessor(),
            formatter=JSONFormatter(),
        )
        result = await pipeline.execute("search query")
    """

    def __init__(
        self,
        retriever: Retriever,
        assembler: Assembler | None = None,
        processor: Processor | None = None,
        formatter: Formatter | None = None,
    ) -> None:
        """Initialize the pipeline.

        Args:
            retriever: Retriever implementation (required)
            assembler: Assembler implementation (default: TopKAssembler)
            processor: Processor implementation (default: IdentityProcessor)
            formatter: Formatter implementation (default: JSONFormatter)
        """
        self._retriever = retriever
        self._assembler = assembler or TopKAssembler()
        self._processor = processor or IdentityProcessor()
        self._formatter = formatter or JSONFormatter()

    @property
    def retriever_name(self) -> str:
        """Name of the retriever in use."""
        return getattr(self._retriever, "name", "unknown")

    @property
    def processor_type(self) -> str:
        """Type of processor in use."""
        return getattr(self._processor, "processor_type", "unknown")

    async def execute(
        self,
        query: str,
        config: PipelineConfig | None = None,
    ) -> PipelineResult:
        """Execute the pipeline.

        Runs retrieval -> assembly -> processing -> formatting.

        Args:
            query: Search query string
            config: Optional pipeline configuration

        Returns:
            PipelineResult with response and metadata
        """
        config = config or PipelineConfig()

        # Stage 1: Retrieval
        logger.debug(f"Pipeline: Retrieving with {self.retriever_name}")
        retrieved_items = await self._retriever.retrieve(query, config.retrieval)
        logger.debug(f"Pipeline: Retrieved {len(retrieved_items)} items")

        # Stage 2: Assembly
        logger.debug("Pipeline: Assembling context")
        context = self._assembler.assemble(retrieved_items, config.assembly)
        logger.debug(
            f"Pipeline: Assembled {len(context.items)} items, "
            f"{context.total_chars} chars, truncated={context.truncated}"
        )

        # Stage 3: Processing
        logger.debug(f"Pipeline: Processing with {self.processor_type}")
        result = await self._processor.process(query, context, config.processor)
        logger.debug(
            f"Pipeline: Processed, answer={'yes' if result.answer else 'no'}, "
            f"citations={len(result.citations)}"
        )

        # Stage 4: Formatting
        logger.debug("Pipeline: Formatting response")
        response = self._formatter.format(result, config.formatter)

        return PipelineResult(
            response=response,
            retrieval_count=len(retrieved_items),
            assembly_count=len(context.items),
            processor_type=self.processor_type,
            truncated=context.truncated,
        )


class PipelineBuilder:
    """Builder for constructing pipelines with fluent API.

    Example:
        pipeline = (
            PipelineBuilder()
            .with_retriever(keyword_retriever)
            .with_assembler(topk_assembler)
            .with_processor(identity_processor)
            .with_formatter(json_formatter)
            .build()
        )
    """

    def __init__(self) -> None:
        """Initialize the builder."""
        self._retriever: Retriever | None = None
        self._assembler: Assembler | None = None
        self._processor: Processor | None = None
        self._formatter: Formatter | None = None

    def with_retriever(self, retriever: Retriever) -> PipelineBuilder:
        """Set the retriever.

        Args:
            retriever: Retriever implementation

        Returns:
            Self for chaining
        """
        self._retriever = retriever
        return self

    def with_assembler(self, assembler: Assembler) -> PipelineBuilder:
        """Set the assembler.

        Args:
            assembler: Assembler implementation

        Returns:
            Self for chaining
        """
        self._assembler = assembler
        return self

    def with_processor(self, processor: Processor) -> PipelineBuilder:
        """Set the processor.

        Args:
            processor: Processor implementation

        Returns:
            Self for chaining
        """
        self._processor = processor
        return self

    def with_formatter(self, formatter: Formatter) -> PipelineBuilder:
        """Set the formatter.

        Args:
            formatter: Formatter implementation

        Returns:
            Self for chaining
        """
        self._formatter = formatter
        return self

    def build(self) -> Pipeline:
        """Build the pipeline.

        Returns:
            Configured Pipeline instance

        Raises:
            ValueError: If no retriever is set
        """
        if self._retriever is None:
            raise ValueError("Pipeline requires a retriever")

        return Pipeline(
            retriever=self._retriever,
            assembler=self._assembler,
            processor=self._processor,
            formatter=self._formatter,
        )


__all__ = [
    "Pipeline",
    "PipelineBuilder",
    "PipelineConfig",
    "PipelineResult",
]
