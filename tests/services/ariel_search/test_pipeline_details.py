"""Tests for pipeline details dataclasses.

Tests for RAGStageStats, AgentToolInvocation, AgentStep, and PipelineDetails.
"""

import pytest

from osprey.services.ariel_search.models import (
    AgentStep,
    AgentToolInvocation,
    PipelineDetails,
    RAGStageStats,
)


class TestRAGStageStats:
    """Tests for RAGStageStats dataclass."""

    def test_defaults(self):
        """All fields default to zero/False."""
        stats = RAGStageStats()
        assert stats.keyword_retrieved == 0
        assert stats.semantic_retrieved == 0
        assert stats.fused_count == 0
        assert stats.context_included == 0
        assert stats.context_truncated is False

    def test_full_construction(self):
        """All fields can be set."""
        stats = RAGStageStats(
            keyword_retrieved=5,
            semantic_retrieved=8,
            fused_count=10,
            context_included=7,
            context_truncated=True,
        )
        assert stats.keyword_retrieved == 5
        assert stats.semantic_retrieved == 8
        assert stats.fused_count == 10
        assert stats.context_included == 7
        assert stats.context_truncated is True

    def test_frozen(self):
        """RAGStageStats is immutable."""
        stats = RAGStageStats()
        with pytest.raises(AttributeError):
            stats.keyword_retrieved = 42  # type: ignore[misc]


class TestAgentToolInvocation:
    """Tests for AgentToolInvocation dataclass."""

    def test_defaults(self):
        """Defaults for optional fields."""
        inv = AgentToolInvocation(tool_name="keyword_search")
        assert inv.tool_name == "keyword_search"
        assert inv.tool_args == {}
        assert inv.result_summary == ""
        assert inv.order == 0

    def test_full_construction(self):
        """All fields can be set."""
        inv = AgentToolInvocation(
            tool_name="semantic_search",
            tool_args={"query": "beam loss"},
            result_summary="Found 3 entries",
            order=2,
        )
        assert inv.tool_name == "semantic_search"
        assert inv.tool_args == {"query": "beam loss"}
        assert inv.result_summary == "Found 3 entries"
        assert inv.order == 2

    def test_frozen(self):
        """AgentToolInvocation is immutable."""
        inv = AgentToolInvocation(tool_name="test")
        with pytest.raises(AttributeError):
            inv.tool_name = "changed"  # type: ignore[misc]


class TestAgentStep:
    """Tests for AgentStep dataclass."""

    def test_defaults(self):
        """Defaults for optional fields."""
        step = AgentStep(step_type="reasoning")
        assert step.step_type == "reasoning"
        assert step.content == ""
        assert step.tool_name is None
        assert step.order == 0

    def test_full_construction(self):
        """All fields can be set."""
        step = AgentStep(
            step_type="tool_call",
            content='{"query": "beam"}',
            tool_name="keyword_search",
            order=3,
        )
        assert step.step_type == "tool_call"
        assert step.content == '{"query": "beam"}'
        assert step.tool_name == "keyword_search"
        assert step.order == 3

    def test_frozen(self):
        """AgentStep is immutable."""
        step = AgentStep(step_type="reasoning")
        with pytest.raises(AttributeError):
            step.step_type = "changed"  # type: ignore[misc]


class TestPipelineDetails:
    """Tests for PipelineDetails dataclass."""

    def test_rag_pipeline(self):
        """PipelineDetails with RAG stats."""
        stats = RAGStageStats(keyword_retrieved=5, fused_count=5, context_included=3)
        pd = PipelineDetails(
            pipeline_type="rag",
            rag_stats=stats,
            step_summary="5 keyword, 5 fused, 3 in context",
        )
        assert pd.pipeline_type == "rag"
        assert pd.rag_stats is not None
        assert pd.rag_stats.keyword_retrieved == 5
        assert pd.agent_tool_invocations == ()
        assert pd.agent_steps == ()

    def test_agent_pipeline(self):
        """PipelineDetails with agent data."""
        inv = AgentToolInvocation(tool_name="keyword_search", order=0)
        step = AgentStep(step_type="tool_call", tool_name="keyword_search", order=1)
        pd = PipelineDetails(
            pipeline_type="agent",
            agent_tool_invocations=(inv,),
            agent_steps=(step,),
            step_summary="1 tool call(s): keyword_search",
        )
        assert pd.pipeline_type == "agent"
        assert pd.rag_stats is None
        assert len(pd.agent_tool_invocations) == 1
        assert len(pd.agent_steps) == 1
        assert pd.step_summary == "1 tool call(s): keyword_search"

    def test_defaults(self):
        """Default collections are empty tuples."""
        pd = PipelineDetails(pipeline_type="rag")
        assert pd.agent_tool_invocations == ()
        assert pd.agent_steps == ()
        assert pd.rag_stats is None
        assert pd.step_summary == ""

    def test_frozen(self):
        """PipelineDetails is immutable."""
        pd = PipelineDetails(pipeline_type="rag")
        with pytest.raises(AttributeError):
            pd.pipeline_type = "agent"  # type: ignore[misc]
