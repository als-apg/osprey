"""ARIEL RAP (Retrieval-Augmented Processing) Pipeline Abstraction.

This package provides the RAP abstraction layer for composable search pipelines.

Architecture:
    Query -> Retriever(s) -> Assembler -> Processor -> Formatter -> Response

See 05_RAP_ABSTRACTION.md for full specification.

Example:
    from osprey.services.ariel_search.pipeline import (
        Pipeline,
        PipelineConfig,
        RetrievalConfig,
    )
    from osprey.services.ariel_search.pipeline.retrievers import KeywordRetriever
    from osprey.services.ariel_search.pipeline.assemblers import TopKAssembler
    from osprey.services.ariel_search.pipeline.processors import IdentityProcessor
    from osprey.services.ariel_search.pipeline.formatters import JSONFormatter

    pipeline = Pipeline(
        retriever=KeywordRetriever(repository, config),
        assembler=TopKAssembler(),
        processor=IdentityProcessor(),
        formatter=JSONFormatter(),
    )
    result = await pipeline.execute("search query")
"""

from osprey.services.ariel_search.pipeline.pipeline import (
    Pipeline,
    PipelineBuilder,
    PipelineConfig,
    PipelineResult,
)
from osprey.services.ariel_search.pipeline.protocols import (
    Assembler,
    Formatter,
    FusionStrategy,
    Processor,
    Retriever,
)
from osprey.services.ariel_search.pipeline.types import (
    AssembledContext,
    AssemblyConfig,
    FormattedResponse,
    ProcessedResult,
    ProcessorConfig,
    RetrievalConfig,
    RetrievedItem,
)

__all__ = [
    # Pipeline
    "Pipeline",
    "PipelineBuilder",
    "PipelineConfig",
    "PipelineResult",
    # Protocols
    "Assembler",
    "Formatter",
    "FusionStrategy",
    "Processor",
    "Retriever",
    # Types
    "AssembledContext",
    "AssemblyConfig",
    "FormattedResponse",
    "ProcessedResult",
    "ProcessorConfig",
    "RetrievalConfig",
    "RetrievedItem",
]
