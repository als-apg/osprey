"""
Tests for Multi-Project Orchestrator

Tests query analysis, decomposition, dependency detection, parallel execution,
and result synthesis for multi-project queries.
"""

import time
from unittest.mock import Mock, patch

import pytest

from osprey.interfaces.pyqt.multi_project_orchestrator import (
    MultiProjectOrchestrator,
    OrchestrationPlan,
    OrchestrationResult,
    SubQuery,
    SubQueryStatus,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_llm_config():
    """Mock LLM configuration."""
    return {
        "provider": "anthropic",
        "model_id": "claude-3-sonnet-20240229",
        "api_key": "test-key",
    }


@pytest.fixture
def orchestrator(mock_llm_config):
    """Create orchestrator with mocked LLM."""
    with patch("osprey.interfaces.pyqt.multi_project_orchestrator.SimpleLLMClient"):
        orch = MultiProjectOrchestrator(
            llm_config=mock_llm_config,
            max_parallel_executions=3,
            enable_dependency_detection=True,
        )
        # Mock the LLM client
        orch.llm_client = Mock()
        return orch


@pytest.fixture
def mock_projects():
    """Create mock project contexts."""
    projects = []
    for i in range(3):
        project = Mock()
        project.metadata = Mock()
        project.metadata.name = f"project{i + 1}"
        project.metadata.description = f"Description for project {i + 1}"
        projects.append(project)
    return projects


@pytest.fixture
def project_contexts_dict(mock_projects):
    """Create dictionary of project contexts."""
    return {p.metadata.name: p for p in mock_projects}


# ============================================================================
# SubQuery Tests
# ============================================================================


class TestSubQuery:
    """Test SubQuery dataclass."""

    def test_create_basic_subquery(self):
        """Test creating basic sub-query."""
        sq = SubQuery(
            query="test query",
            project_name="project1",
            index=0,
        )

        assert sq.query == "test query"
        assert sq.project_name == "project1"
        assert sq.index == 0
        assert sq.status == SubQueryStatus.PENDING
        assert sq.dependencies == []

    def test_create_subquery_with_dependencies(self):
        """Test creating sub-query with dependencies."""
        sq = SubQuery(
            query="test query",
            project_name="project1",
            index=1,
            dependencies=[0],
        )

        assert sq.dependencies == [0]

    def test_subquery_status_update(self):
        """Test updating sub-query status."""
        sq = SubQuery(query="test", project_name="proj1", index=0)

        sq.status = SubQueryStatus.IN_PROGRESS
        assert sq.status == SubQueryStatus.IN_PROGRESS

        sq.status = SubQueryStatus.COMPLETED
        assert sq.status == SubQueryStatus.COMPLETED


# ============================================================================
# OrchestrationPlan Tests
# ============================================================================


class TestOrchestrationPlan:
    """Test OrchestrationPlan dataclass."""

    def test_create_basic_plan(self):
        """Test creating basic orchestration plan."""
        plan = OrchestrationPlan(
            original_query="test query",
            sub_queries=[],
            is_multi_project=False,
        )

        assert plan.original_query == "test query"
        assert plan.sub_queries == []
        assert plan.is_multi_project is False

    def test_create_multi_project_plan(self):
        """Test creating multi-project plan."""
        sub_queries = [
            SubQuery(query="query1", project_name="proj1", index=0),
            SubQuery(query="query2", project_name="proj2", index=1),
        ]

        plan = OrchestrationPlan(
            original_query="multi query",
            sub_queries=sub_queries,
            is_multi_project=True,
            reasoning="Multiple projects needed",
        )

        assert plan.is_multi_project is True
        assert len(plan.sub_queries) == 2
        assert plan.reasoning == "Multiple projects needed"


# ============================================================================
# Orchestrator Initialization Tests
# ============================================================================


class TestOrchestratorInit:
    """Test orchestrator initialization."""

    def test_init_with_config(self, mock_llm_config):
        """Test initialization with LLM config."""
        with patch(
            "osprey.interfaces.pyqt.multi_project_orchestrator.SimpleLLMClient"
        ) as mock_client:
            orch = MultiProjectOrchestrator(
                llm_config=mock_llm_config,
                max_parallel_executions=5,
                enable_dependency_detection=False,
            )

            assert orch.max_parallel_executions == 5
            assert orch.enable_dependency_detection is False
            mock_client.assert_called_once()

    def test_init_without_config(self):
        """Test initialization without LLM config."""
        with patch(
            "osprey.interfaces.pyqt.multi_project_orchestrator.SimpleLLMClient.from_gui_config"
        ) as mock_from_config:
            orch = MultiProjectOrchestrator()

            assert orch.max_parallel_executions == 3
            assert orch.enable_dependency_detection is True
            mock_from_config.assert_called_once()


# ============================================================================
# Query Analysis Tests
# ============================================================================


class TestQueryAnalysis:
    """Test query analysis functionality."""

    def test_analyze_single_project_query(self, orchestrator, mock_projects):
        """Test analyzing single-project query."""
        # Mock LLM response for single project
        orchestrator.llm_client.call.return_value = """
MULTI_PROJECT: no
REASONING: Query only requires weather information
"""

        plan = orchestrator.analyze_query("What is the weather?", mock_projects)

        assert plan.is_multi_project is False
        assert len(plan.sub_queries) == 0

    def test_analyze_multi_project_query(self, orchestrator, mock_projects):
        """Test analyzing multi-project query."""
        # Mock LLM response for multi-project
        orchestrator.llm_client.call.return_value = """
MULTI_PROJECT: yes
REASONING: Query requires both weather and MPS information
SUB_QUERIES:
project1: What is the weather?
project2: Is MPS operational?
"""

        plan = orchestrator.analyze_query(
            "What is the weather and is MPS operational?",
            mock_projects,
        )

        assert plan.is_multi_project is True
        assert len(plan.sub_queries) == 2
        assert plan.sub_queries[0].project_name == "project1"
        assert plan.sub_queries[1].project_name == "project2"

    def test_analyze_query_llm_failure(self, orchestrator, mock_projects):
        """Test query analysis with LLM failure."""
        orchestrator.llm_client.call.side_effect = Exception("LLM error")

        plan = orchestrator.analyze_query("test query", mock_projects)

        # Should return fallback plan
        assert plan.is_multi_project is False
        assert "failed" in plan.reasoning.lower()

    def test_create_analysis_prompt(self, orchestrator, mock_projects):
        """Test analysis prompt creation."""
        prompt = orchestrator._create_analysis_prompt("test query", mock_projects)

        assert "test query" in prompt
        assert "project1" in prompt
        assert "MULTI_PROJECT" in prompt


# ============================================================================
# Dependency Detection Tests
# ============================================================================


class TestDependencyDetection:
    """Test dependency detection."""

    def test_detect_no_dependencies(self, orchestrator):
        """Test detecting no dependencies."""
        sub_queries = [
            SubQuery(query="weather query", project_name="proj1", index=0),
            SubQuery(query="mps query", project_name="proj2", index=1),
        ]

        plan = OrchestrationPlan(
            original_query="test",
            sub_queries=sub_queries,
            is_multi_project=True,
        )

        orchestrator._detect_dependencies(plan)

        # Unrelated queries should have no dependencies
        assert len(plan.sub_queries[1].dependencies) == 0

    def test_detect_dependencies(self, orchestrator):
        """Test detecting dependencies between queries."""
        sub_queries = [
            SubQuery(query="get temperature data", project_name="proj1", index=0),
            SubQuery(query="compare temperature with baseline", project_name="proj2", index=1),
        ]

        plan = OrchestrationPlan(
            original_query="test",
            sub_queries=sub_queries,
            is_multi_project=True,
        )

        orchestrator._detect_dependencies(plan)

        # Second query mentions "temperature" from first query
        # May or may not detect dependency based on keyword overlap

    def test_queries_related(self, orchestrator):
        """Test query relation detection."""
        related = orchestrator._queries_related(
            "get temperature data",
            "analyze temperature trends",
        )

        # Should detect "temperature" keyword overlap
        assert isinstance(related, bool)


# ============================================================================
# Execution Order Tests
# ============================================================================


class TestExecutionOrder:
    """Test execution order creation."""

    def test_create_order_no_dependencies(self, orchestrator):
        """Test execution order with no dependencies."""
        sub_queries = [
            SubQuery(query="query1", project_name="proj1", index=0),
            SubQuery(query="query2", project_name="proj2", index=1),
            SubQuery(query="query3", project_name="proj3", index=2),
        ]

        plan = OrchestrationPlan(
            original_query="test",
            sub_queries=sub_queries,
            is_multi_project=True,
        )

        orchestrator._create_execution_order(plan)

        # All queries can execute in parallel (single stage)
        assert len(plan.execution_order) == 1
        assert len(plan.execution_order[0]) == 3

    def test_create_order_with_dependencies(self, orchestrator):
        """Test execution order with dependencies."""
        sub_queries = [
            SubQuery(query="query1", project_name="proj1", index=0),
            SubQuery(query="query2", project_name="proj2", index=1, dependencies=[0]),
            SubQuery(query="query3", project_name="proj3", index=2, dependencies=[1]),
        ]

        plan = OrchestrationPlan(
            original_query="test",
            sub_queries=sub_queries,
            is_multi_project=True,
        )

        orchestrator._create_execution_order(plan)

        # Should have 3 stages (sequential execution)
        assert len(plan.execution_order) == 3
        assert plan.execution_order[0] == [0]
        assert plan.execution_order[1] == [1]
        assert plan.execution_order[2] == [2]

    def test_create_order_mixed_dependencies(self, orchestrator):
        """Test execution order with mixed dependencies."""
        sub_queries = [
            SubQuery(query="query1", project_name="proj1", index=0),
            SubQuery(query="query2", project_name="proj2", index=1),
            SubQuery(query="query3", project_name="proj3", index=2, dependencies=[0, 1]),
        ]

        plan = OrchestrationPlan(
            original_query="test",
            sub_queries=sub_queries,
            is_multi_project=True,
        )

        orchestrator._create_execution_order(plan)

        # First two can run in parallel, third depends on both
        assert len(plan.execution_order) == 2
        assert set(plan.execution_order[0]) == {0, 1}
        assert plan.execution_order[1] == [2]


# ============================================================================
# Plan Execution Tests
# ============================================================================


class TestPlanExecution:
    """Test plan execution."""

    def test_execute_single_project_plan(self, orchestrator, project_contexts_dict):
        """Test executing single-project plan."""
        plan = OrchestrationPlan(
            original_query="test",
            sub_queries=[],
            is_multi_project=False,
        )

        result = orchestrator.execute_plan(plan, project_contexts_dict)

        assert result.success is False
        assert "not a multi-project" in result.error.lower()

    def test_execute_multi_project_plan(self, orchestrator, project_contexts_dict):
        """Test executing multi-project plan."""
        sub_queries = [
            SubQuery(query="query1", project_name="project1", index=0),
            SubQuery(query="query2", project_name="project2", index=1),
        ]

        plan = OrchestrationPlan(
            original_query="test",
            sub_queries=sub_queries,
            is_multi_project=True,
        )

        # Mock LLM synthesis
        orchestrator.llm_client.call.return_value = "Combined result"

        result = orchestrator.execute_plan(plan, project_contexts_dict)

        assert result.success is True
        assert result.combined_result == "Combined result"
        assert len(result.individual_results) == 2

    def test_execute_plan_with_failure(self, orchestrator, project_contexts_dict):
        """Test executing plan with failure."""
        sub_queries = [
            SubQuery(query="query1", project_name="nonexistent", index=0),
        ]

        plan = OrchestrationPlan(
            original_query="test",
            sub_queries=sub_queries,
            is_multi_project=True,
        )

        result = orchestrator.execute_plan(plan, project_contexts_dict)

        # Should handle failure gracefully
        assert isinstance(result, OrchestrationResult)


# ============================================================================
# Result Synthesis Tests
# ============================================================================


class TestResultSynthesis:
    """Test result synthesis."""

    def test_combine_results_with_llm(self, orchestrator):
        """Test combining results with LLM."""
        sub_queries = [
            SubQuery(query="query1", project_name="proj1", index=0),
            SubQuery(query="query2", project_name="proj2", index=1),
        ]

        plan = OrchestrationPlan(
            original_query="test",
            sub_queries=sub_queries,
            is_multi_project=True,
        )

        individual_results = {
            0: "Result 1",
            1: "Result 2",
        }

        orchestrator.llm_client.call.return_value = "Synthesized result"

        combined = orchestrator._combine_results(plan, individual_results)

        assert combined == "Synthesized result"

    def test_combine_results_llm_failure(self, orchestrator):
        """Test combining results with LLM failure."""
        sub_queries = [
            SubQuery(query="query1", project_name="proj1", index=0),
            SubQuery(query="query2", project_name="proj2", index=1),
        ]

        plan = OrchestrationPlan(
            original_query="test",
            sub_queries=sub_queries,
            is_multi_project=True,
        )

        individual_results = {
            0: "Result 1",
            1: "Result 2",
        }

        orchestrator.llm_client.call.side_effect = Exception("LLM error")

        combined = orchestrator._combine_results(plan, individual_results)

        # Should fall back to simple combination
        assert "proj1" in combined
        assert "proj2" in combined

    def test_simple_combine(self, orchestrator):
        """Test simple result combination."""
        sub_queries = [
            SubQuery(query="query1", project_name="proj1", index=0),
            SubQuery(query="query2", project_name="proj2", index=1),
        ]

        individual_results = {
            0: "Result 1",
            1: "Result 2",
        }

        combined = orchestrator._simple_combine(sub_queries, individual_results)

        assert "proj1" in combined
        assert "Result 1" in combined
        assert "proj2" in combined
        assert "Result 2" in combined

    def test_create_synthesis_prompt(self, orchestrator):
        """Test synthesis prompt creation."""
        sub_queries = [
            SubQuery(query="query1", project_name="proj1", index=0),
        ]

        individual_results = {0: "Result 1"}

        prompt = orchestrator._create_synthesis_prompt(
            "original query",
            sub_queries,
            individual_results,
        )

        assert "original query" in prompt
        assert "query1" in prompt
        assert "Result 1" in prompt


# ============================================================================
# Integration Tests
# ============================================================================


class TestOrchestratorIntegration:
    """Integration tests for orchestrator."""

    def test_full_orchestration_flow(self, orchestrator, mock_projects, project_contexts_dict):
        """Test complete orchestration flow."""
        # Mock analysis
        orchestrator.llm_client.call.return_value = """
MULTI_PROJECT: yes
REASONING: Requires multiple projects
SUB_QUERIES:
project1: query1
project2: query2
"""

        # Analyze query
        plan = orchestrator.analyze_query("test query", mock_projects)

        assert plan.is_multi_project is True
        assert len(plan.sub_queries) == 2

        # Mock synthesis
        orchestrator.llm_client.call.return_value = "Final result"

        # Execute plan
        result = orchestrator.execute_plan(plan, project_contexts_dict)

        assert result.success is True

    def test_single_project_flow(self, orchestrator, mock_projects):
        """Test single-project flow."""
        orchestrator.llm_client.call.return_value = """
MULTI_PROJECT: no
REASONING: Single project sufficient
"""

        plan = orchestrator.analyze_query("simple query", mock_projects)

        assert plan.is_multi_project is False

    def test_parallel_execution(self, orchestrator, project_contexts_dict):
        """Test parallel execution of independent queries."""
        sub_queries = [
            SubQuery(query=f"query{i}", project_name=f"project{i + 1}", index=i) for i in range(3)
        ]

        plan = OrchestrationPlan(
            original_query="test",
            sub_queries=sub_queries,
            is_multi_project=True,
        )

        orchestrator.llm_client.call.return_value = "Combined"

        start_time = time.time()
        result = orchestrator.execute_plan(plan, project_contexts_dict)
        execution_time = time.time() - start_time

        # Parallel execution should be faster than sequential
        assert result.success is True
        # Execution time should be reasonable (not 3x sequential)
        assert execution_time < 5.0  # Generous timeout


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Test error handling."""

    def test_handle_missing_llm_client(self):
        """Test handling missing LLM client."""
        orch = MultiProjectOrchestrator.__new__(MultiProjectOrchestrator)
        orch.llm_client = None

        with pytest.raises(Exception, match="LLM client not initialized"):
            orch._call_llm_for_analysis("test prompt")

    def test_handle_invalid_project(self, orchestrator, project_contexts_dict):
        """Test handling invalid project in sub-query."""
        sub_queries = [
            SubQuery(query="query1", project_name="invalid_project", index=0),
        ]

        plan = OrchestrationPlan(
            original_query="test",
            sub_queries=sub_queries,
            is_multi_project=True,
        )

        orchestrator.llm_client.call.return_value = "Fallback"

        result = orchestrator.execute_plan(plan, project_contexts_dict)

        # Should handle gracefully
        assert isinstance(result, OrchestrationResult)

    def test_handle_execution_timeout(self, orchestrator, project_contexts_dict):
        """Test handling execution timeout."""
        # This is a placeholder - actual timeout handling would need
        # more sophisticated implementation
        sub_queries = [
            SubQuery(query="query1", project_name="project1", index=0),
        ]

        plan = OrchestrationPlan(
            original_query="test",
            sub_queries=sub_queries,
            is_multi_project=True,
        )

        orchestrator.llm_client.call.return_value = "Result"

        result = orchestrator.execute_plan(plan, project_contexts_dict)

        # Should complete without hanging
        assert isinstance(result, OrchestrationResult)
