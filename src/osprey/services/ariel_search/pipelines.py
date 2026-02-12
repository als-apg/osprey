"""ARIEL pipeline descriptors.

Declares the RAG and Agent pipelines with their tunable parameters
for the frontend capabilities API.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from osprey.services.ariel_search.search.base import ParameterDescriptor


@dataclass(frozen=True)
class PipelineDescriptor:
    """Metadata for a pipeline execution strategy.

    Attributes:
        name: Pipeline key (e.g. "rag")
        label: Human-readable label (e.g. "RAG")
        description: What this pipeline does
        category: "llm" or "direct"
        parameters: Tunable parameters for this pipeline
    """

    name: str
    label: str
    description: str
    category: str  # "llm" or "direct"
    parameters: list[ParameterDescriptor] = field(default_factory=list)


RAG_PIPELINE = PipelineDescriptor(
    name="rag",
    label="RAG",
    description="AI-powered search with hybrid retrieval and answer generation",
    category="llm",
    parameters=[
        ParameterDescriptor(
            name="similarity_threshold",
            label="Similarity Threshold",
            description="Minimum cosine similarity for semantic retrieval",
            param_type="float",
            default=0.7,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            section="Retrieval",
        ),
        ParameterDescriptor(
            name="max_context_chars",
            label="Max Context Chars",
            description="Maximum characters in the LLM context window",
            param_type="int",
            default=12000,
            min_value=1000,
            max_value=50000,
            step=1000,
            section="Context Assembly",
        ),
        ParameterDescriptor(
            name="max_chars_per_entry",
            label="Max Chars/Entry",
            description="Maximum characters per entry in context",
            param_type="int",
            default=2000,
            min_value=100,
            max_value=10000,
            step=100,
            section="Context Assembly",
        ),
        ParameterDescriptor(
            name="temperature",
            label="Temperature",
            description="LLM temperature for answer generation",
            param_type="float",
            default=0.1,
            min_value=0.0,
            max_value=2.0,
            step=0.05,
            section="LLM Processing",
        ),
    ],
)

AGENT_PIPELINE = PipelineDescriptor(
    name="agent",
    label="Agent",
    description="Autonomous ReAct agent with multi-step reasoning",
    category="llm",
    parameters=[
        ParameterDescriptor(
            name="temperature",
            label="Temperature",
            description="LLM temperature for agent reasoning",
            param_type="float",
            default=0.1,
            min_value=0.0,
            max_value=2.0,
            step=0.05,
            section="LLM Processing",
        ),
    ],
)

_ALL_PIPELINES = [RAG_PIPELINE, AGENT_PIPELINE]
_PIPELINES_BY_NAME = {p.name: p for p in _ALL_PIPELINES}


def get_pipeline_descriptors() -> list[PipelineDescriptor]:
    """Return all pipeline descriptors."""
    return list(_ALL_PIPELINES)


def get_pipeline_descriptor(name: str) -> PipelineDescriptor:
    """Return a single pipeline descriptor by name.

    Used by the central Osprey registry to look up specific pipelines.

    Args:
        name: Pipeline name (e.g. "rag", "agent")

    Returns:
        The matching PipelineDescriptor

    Raises:
        KeyError: If no pipeline with the given name exists
    """
    return _PIPELINES_BY_NAME[name]


__all__ = [
    "AGENT_PIPELINE",
    "PipelineDescriptor",
    "RAG_PIPELINE",
    "get_pipeline_descriptor",
    "get_pipeline_descriptors",
]
