"""
Multi-Project Orchestrator for GUI Support

This module provides the MultiProjectOrchestrator class for handling queries
that span multiple projects by coordinating their execution and combining results.

Key Features:
- Query decomposition into sub-queries
- Dependency detection between sub-queries
- Parallel execution planning
- Result aggregation and synthesis
- Progress tracking and status updates
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from osprey.interfaces.pyqt.llm_client import SimpleLLMClient
from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from osprey.interfaces.pyqt.project_manager import ProjectContext

logger = get_logger("multi_project_orchestrator")


class SubQueryStatus(Enum):
    """Status of a sub-query execution."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class SubQuery:
    """Represents a decomposed sub-query."""

    query: str
    project_name: str
    index: int
    dependencies: list[int] = field(default_factory=list)
    status: SubQueryStatus = SubQueryStatus.PENDING
    result: str | None = None
    error: str | None = None
    execution_time: float = 0.0
    confidence: float = 0.0


@dataclass
class OrchestrationPlan:
    """Plan for executing a multi-project query."""

    original_query: str
    sub_queries: list[SubQuery]
    execution_order: list[list[int]] = field(default_factory=list)
    is_multi_project: bool = False
    reasoning: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class OrchestrationResult:
    """Result of orchestrated execution."""

    original_query: str
    plan: OrchestrationPlan
    combined_result: str
    individual_results: dict[int, str] = field(default_factory=dict)
    total_execution_time: float = 0.0
    success: bool = True
    error: str | None = None


class MultiProjectOrchestrator:
    """
    Orchestrates execution of queries spanning multiple projects.

    Handles:
    - Query analysis and decomposition
    - Dependency detection between sub-queries
    - Parallel execution planning
    - Result aggregation and synthesis
    """

    def __init__(
        self,
        llm_config: dict[str, Any] | None = None,
        max_parallel_executions: int = 3,
        enable_dependency_detection: bool = True,
    ):
        """Initialize orchestrator.

        Args:
            llm_config: Optional LLM configuration for orchestration.
            max_parallel_executions: Maximum parallel sub-query executions.
            enable_dependency_detection: Whether to detect dependencies between sub-queries.
        """
        self.logger = logger
        self.llm_config = llm_config or {}
        self.max_parallel_executions = max_parallel_executions
        self.enable_dependency_detection = enable_dependency_detection

        # Initialize SimpleLLMClient for orchestration (NO SINGLETON DEPENDENCY!)
        if llm_config:
            # Use provided LLM config
            self.llm_client = SimpleLLMClient(
                provider=llm_config.get("provider", "anthropic"),
                model_id=llm_config.get("model_id", "claude-3-sonnet-20240229"),
                api_key=llm_config.get("api_key"),
                base_url=llm_config.get("base_url"),
            )
            self.logger.info(
                f"Initialized orchestration LLM client from config: "
                f"{llm_config.get('provider')}/{llm_config.get('model_id')}"
            )
        else:
            # Use GUI config (reads user's configured 'classifier' model)
            try:
                self.llm_client = SimpleLLMClient.from_gui_config()
                self.logger.info(
                    "Initialized orchestration LLM client from gui_config.yml "
                    f"({self.llm_client.provider}/{self.llm_client.model_id})"
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to initialize LLM client from gui_config.yml: {e}. "
                    f"Orchestration will fail until LLM client is properly configured."
                )
                self.llm_client = None

        self.logger.info(
            f"Initialized MultiProjectOrchestrator "
            f"(max_parallel={max_parallel_executions}, "
            f"dependency_detection={enable_dependency_detection})"
        )

    def analyze_query(
        self, query: str, available_projects: list[ProjectContext]
    ) -> OrchestrationPlan:
        """Analyze query to determine if orchestration is needed.

        Args:
            query: User's query.
            available_projects: List of available projects.

        Returns:
            OrchestrationPlan with decomposition if needed.
        """
        try:
            # Create analysis prompt
            analysis_prompt = self._create_analysis_prompt(query, available_projects)

            # Call LLM for analysis
            analysis_result = self._call_llm_for_analysis(analysis_prompt)

            # Parse analysis result
            plan = self._parse_analysis_result(analysis_result, query, available_projects)

            if plan.is_multi_project:
                self.logger.info(
                    f"Multi-project query detected: {len(plan.sub_queries)} sub-queries"
                )
            else:
                self.logger.debug("Single-project query - no orchestration needed")

            return plan

        except Exception as e:
            self.logger.error(f"Query analysis failed: {e}")
            # Return single-project plan as fallback
            return OrchestrationPlan(
                original_query=query,
                sub_queries=[],
                is_multi_project=False,
                reasoning=f"Analysis failed: {e}",
            )

    def execute_plan(
        self,
        plan: OrchestrationPlan,
        project_contexts: dict[str, ProjectContext],
        progress_callback: Callable | None = None,
    ) -> OrchestrationResult:
        """Execute an orchestration plan.

        Args:
            plan: OrchestrationPlan to execute.
            project_contexts: Dictionary mapping project names to ProjectContext.
            progress_callback: Optional callback for progress updates.

        Returns:
            OrchestrationResult with combined results.
        """
        start_time = time.time()

        try:
            if not plan.is_multi_project or not plan.sub_queries:
                # Single project - no orchestration needed
                return OrchestrationResult(
                    original_query=plan.original_query,
                    plan=plan,
                    combined_result="",
                    success=False,
                    error="Not a multi-project query",
                )

            # Detect dependencies if enabled
            if self.enable_dependency_detection:
                self._detect_dependencies(plan)

            # Create execution order based on dependencies
            self._create_execution_order(plan)

            # Execute sub-queries in order
            individual_results = self._execute_sub_queries(
                plan, project_contexts, progress_callback
            )

            # Combine results
            combined_result = self._combine_results(plan, individual_results)

            execution_time = time.time() - start_time

            return OrchestrationResult(
                original_query=plan.original_query,
                plan=plan,
                combined_result=combined_result,
                individual_results=individual_results,
                total_execution_time=execution_time,
                success=True,
            )

        except Exception as e:
            self.logger.error(f"Plan execution failed: {e}")
            execution_time = time.time() - start_time

            return OrchestrationResult(
                original_query=plan.original_query,
                plan=plan,
                combined_result="",
                total_execution_time=execution_time,
                success=False,
                error=str(e),
            )

    # Private methods

    def _create_analysis_prompt(self, query: str, projects: list[ProjectContext]) -> str:
        """Create prompt for query analysis.

        Args:
            query: User's query.
            projects: Available projects.

        Returns:
            Formatted prompt for LLM.
        """
        project_descriptions = []
        for project in projects:
            project_descriptions.append(
                f"- {project.metadata.name}: {project.metadata.description}"
            )

        prompt = f"""You are a query analyzer for a multi-project system. Analyze the user's query to determine if it requires capabilities from multiple projects.

Available Projects:
{chr(10).join(project_descriptions)}

User Query: {query}

Analyze this query and respond in the following format:

MULTI_PROJECT: <yes/no>
REASONING: <brief explanation>
SUB_QUERIES: <if multi-project, list sub-queries, one per line, in format "PROJECT_NAME: query text">

Guidelines:
1. A query is multi-project if it explicitly asks about multiple domains or requires information from different projects
2. Look for connecting words like "and", "also", "both", "plus" that indicate multiple requests
3. Count the number of distinct questions - if there are 2+ separate questions, it's likely multi-project
4. Examples of multi-project queries:
   - "What's the weather in SF and is the MPS system operational?" → 2 questions, 2 projects
   - "What is the weather like in NY? Can you also tell me the last MPS fault and the storage ring beam current now?" → 3 questions, 3 projects
   - "Compare the temperature data with the beam current" → 2 data sources, likely 2 projects
   - "Show me both the channel finder results and the weather forecast" → 2 requests, 2 projects
5. Examples of single-project queries:
   - "What's the weather in San Francisco?" → 1 question, 1 project
   - "Is the MPS system operational?" → 1 question, 1 project
   - "Show me the channel finder results" → 1 request, 1 project
6. When decomposing multi-project queries:
   - Create ONE sub-query per distinct question/request
   - Each sub-query should be self-contained and answerable independently
   - Match each sub-query to the most appropriate project based on its domain
   - Preserve the specific details from the original query (locations, times, etc.)
7. IMPORTANT: If you identify multiple distinct questions, you MUST decompose them into separate sub-queries
   - Don't ask for clarification if the questions are clear
   - Each question should map to exactly one project

Respond now:"""

        return prompt

    def _call_llm_for_analysis(self, prompt: str) -> str:
        """Call LLM for query analysis - NO SINGLETON DEPENDENCY!

        Uses SimpleLLMClient which reads configuration from gui_config.yml
        or explicit config, avoiding the "Registry not initialized" error.

        Args:
            prompt: Analysis prompt.

        Returns:
            LLM response.

        Raises:
            Exception: If LLM call fails.
        """
        if not self.llm_client:
            raise Exception(
                "LLM client not initialized. Please ensure gui_config.yml has a "
                "'classifier' model configured, or provide llm_config during orchestrator initialization."
            )

        try:
            # Direct LLM call - no registry needed!
            response = self.llm_client.call(prompt=prompt, max_tokens=1000, temperature=0.0)

            return response

        except Exception as e:
            raise Exception(f"Failed to call LLM for analysis: {e}") from e

    def _parse_analysis_result(
        self, result: str, original_query: str, projects: list[ProjectContext]
    ) -> OrchestrationPlan:
        """Parse LLM analysis result.

        Args:
            result: LLM response.
            original_query: Original user query.
            projects: Available projects.

        Returns:
            OrchestrationPlan.
        """
        lines = result.strip().split("\n")
        data = {}

        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                data[key.strip()] = value.strip()

        is_multi_project = data.get("MULTI_PROJECT", "no").lower() == "yes"
        reasoning = data.get("REASONING", "")

        sub_queries = []

        if is_multi_project:
            # Parse sub-queries
            in_sub_queries = False
            index = 0

            for line in lines:
                if line.strip().startswith("SUB_QUERIES:"):
                    in_sub_queries = True
                    continue

                if (
                    in_sub_queries
                    and ":" in line
                    and not line.strip().startswith(("MULTI_PROJECT:", "REASONING:"))
                ):
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        project_name = parts[0].strip()
                        query_text = parts[1].strip()

                        # Validate project exists
                        project_names = [p.metadata.name for p in projects]
                        if project_name in project_names:
                            sub_queries.append(
                                SubQuery(query=query_text, project_name=project_name, index=index)
                            )
                            index += 1

        return OrchestrationPlan(
            original_query=original_query,
            sub_queries=sub_queries,
            is_multi_project=is_multi_project and len(sub_queries) > 1,
            reasoning=reasoning,
        )

    def _detect_dependencies(self, plan: OrchestrationPlan):
        """Detect dependencies between sub-queries.

        Args:
            plan: OrchestrationPlan to analyze.
        """
        # Simple dependency detection based on query content
        # More sophisticated detection could use LLM or semantic analysis

        for i, sub_query in enumerate(plan.sub_queries):
            for j, other_query in enumerate(plan.sub_queries):
                if i != j and i > j:
                    # Check if sub_query references concepts from other_query
                    # Simple heuristic: check for common keywords
                    if self._queries_related(sub_query.query, other_query.query):
                        sub_query.dependencies.append(j)
                        self.logger.debug(f"Detected dependency: sub-query {i} depends on {j}")

    def _queries_related(self, query1: str, query2: str) -> bool:
        """Check if two queries are related (simple heuristic).

        Args:
            query1: First query.
            query2: Second query.

        Returns:
            True if queries appear related.
        """
        # Simple keyword-based relation detection
        # Could be enhanced with semantic similarity

        keywords1 = set(query1.lower().split())
        keywords2 = set(query2.lower().split())

        # Remove common words
        common_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "is",
            "are",
            "was",
            "were",
        }
        keywords1 -= common_words
        keywords2 -= common_words

        # Check for overlap
        overlap = keywords1 & keywords2

        return len(overlap) >= 2

    def _create_execution_order(self, plan: OrchestrationPlan):
        """Create execution order based on dependencies.

        Args:
            plan: OrchestrationPlan to order.
        """
        # Topological sort to determine execution order
        # Group independent queries for parallel execution

        executed = set()
        execution_order = []

        while len(executed) < len(plan.sub_queries):
            # Find queries that can be executed (all dependencies met)
            ready = []

            for i, sub_query in enumerate(plan.sub_queries):
                if i not in executed:
                    deps_met = all(dep in executed for dep in sub_query.dependencies)
                    if deps_met:
                        ready.append(i)

            if not ready:
                # Circular dependency or error
                self.logger.warning("Circular dependency detected, executing remaining in order")
                ready = [i for i in range(len(plan.sub_queries)) if i not in executed]

            execution_order.append(ready)
            executed.update(ready)

        plan.execution_order = execution_order

        self.logger.info(
            f"Execution order created: {len(execution_order)} stages, "
            f"max parallel: {max(len(stage) for stage in execution_order)}"
        )

    def _execute_sub_queries(
        self,
        plan: OrchestrationPlan,
        project_contexts: dict[str, ProjectContext],
        progress_callback: Callable | None = None,
    ) -> dict[int, str]:
        """Execute sub-queries according to plan.

        Args:
            plan: OrchestrationPlan to execute.
            project_contexts: Available project contexts.
            progress_callback: Optional progress callback.

        Returns:
            Dictionary mapping sub-query index to result.
        """
        results = {}

        for stage_idx, stage in enumerate(plan.execution_order):
            self.logger.info(f"Executing stage {stage_idx + 1}/{len(plan.execution_order)}")

            # Execute queries in this stage in parallel
            stage_results = self._execute_stage(
                stage, plan.sub_queries, project_contexts, progress_callback
            )

            results.update(stage_results)

        return results

    def _execute_stage(
        self,
        stage_indices: list[int],
        sub_queries: list[SubQuery],
        project_contexts: dict[str, ProjectContext],
        progress_callback: Callable | None = None,
    ) -> dict[int, str]:
        """Execute a stage of sub-queries in parallel.

        Args:
            stage_indices: Indices of sub-queries to execute.
            sub_queries: All sub-queries.
            project_contexts: Available project contexts.
            progress_callback: Optional progress callback.

        Returns:
            Dictionary mapping sub-query index to result.
        """
        results = {}

        # Limit parallelism
        max_workers = min(self.max_parallel_executions, len(stage_indices))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all sub-queries in this stage
            futures = {}

            for idx in stage_indices:
                sub_query = sub_queries[idx]
                future = executor.submit(
                    self._execute_single_query, sub_query, project_contexts, progress_callback
                )
                futures[future] = idx

            # Collect results as they complete
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    results[idx] = result
                    sub_queries[idx].status = SubQueryStatus.COMPLETED
                    sub_queries[idx].result = result
                except Exception as e:
                    self.logger.error(f"Sub-query {idx} failed: {e}")
                    results[idx] = f"Error: {e}"
                    sub_queries[idx].status = SubQueryStatus.FAILED
                    sub_queries[idx].error = str(e)

        return results

    def _execute_single_query(
        self,
        sub_query: SubQuery,
        project_contexts: dict[str, ProjectContext],
        progress_callback: Callable | None = None,
    ) -> str:
        """Execute a single sub-query.

        Args:
            sub_query: SubQuery to execute.
            project_contexts: Available project contexts.
            progress_callback: Optional progress callback.

        Returns:
            Query result.
        """
        start_time = time.time()
        sub_query.status = SubQueryStatus.IN_PROGRESS

        if progress_callback:
            progress_callback(f"Executing: {sub_query.query[:50]}...", sub_query.project_name)

        try:
            # Get project context
            project = project_contexts.get(sub_query.project_name)
            if not project:
                raise Exception(f"Project not found: {sub_query.project_name}")

            # Execute query using project's gateway
            # This is a simplified version - actual implementation would use
            # the full gateway.process_message flow
            result = f"[Result from {sub_query.project_name}]"

            sub_query.execution_time = time.time() - start_time

            return result

        except Exception:
            sub_query.execution_time = time.time() - start_time
            raise

    def _combine_results(self, plan: OrchestrationPlan, individual_results: dict[int, str]) -> str:
        """Combine individual results into a coherent response.

        Args:
            plan: OrchestrationPlan.
            individual_results: Individual sub-query results.

        Returns:
            Combined result string.
        """
        try:
            # Create synthesis prompt
            synthesis_prompt = self._create_synthesis_prompt(
                plan.original_query, plan.sub_queries, individual_results
            )

            # Call LLM for synthesis
            combined = self._call_llm_for_synthesis(synthesis_prompt)

            return combined

        except Exception as e:
            self.logger.error(f"Result synthesis failed: {e}")
            # Fallback: simple concatenation
            return self._simple_combine(plan.sub_queries, individual_results)

    def _create_synthesis_prompt(
        self, original_query: str, sub_queries: list[SubQuery], results: dict[int, str]
    ) -> str:
        """Create prompt for result synthesis.

        Args:
            original_query: Original user query.
            sub_queries: List of sub-queries.
            results: Individual results.

        Returns:
            Synthesis prompt.
        """
        results_text = []
        for idx, sub_query in enumerate(sub_queries):
            result = results.get(idx, "No result")
            results_text.append(
                f"Sub-query {idx + 1} ({sub_query.project_name}): {sub_query.query}\n"
                f"Result: {result}\n"
            )

        prompt = f"""You are synthesizing results from multiple specialized systems to answer a user's question.

Original Question: {original_query}

Individual Results:
{chr(10).join(results_text)}

Synthesize these results into a single, coherent response that:
1. Directly answers the original question
2. Integrates information from all relevant results
3. Maintains context and relationships between different pieces of information
4. Is clear and concise
5. Acknowledges if any sub-queries failed

Provide your synthesized response:"""

        return prompt

    def _call_llm_for_synthesis(self, prompt: str) -> str:
        """Call LLM for result synthesis - NO SINGLETON DEPENDENCY!

        Uses SimpleLLMClient which reads configuration from gui_config.yml
        or explicit config, avoiding the "Registry not initialized" error.

        Args:
            prompt: Synthesis prompt.

        Returns:
            Synthesized result.

        Raises:
            Exception: If LLM call fails.
        """
        if not self.llm_client:
            raise Exception(
                "LLM client not initialized. Please ensure gui_config.yml has a "
                "'classifier' model configured, or provide llm_config during orchestrator initialization."
            )

        try:
            # Direct LLM call - no registry needed!
            response = self.llm_client.call(prompt=prompt, max_tokens=1500, temperature=0.0)

            return response

        except Exception as e:
            raise Exception(f"Failed to call LLM for synthesis: {e}") from e

    def _simple_combine(self, sub_queries: list[SubQuery], results: dict[int, str]) -> str:
        """Simple result combination (fallback).

        Args:
            sub_queries: List of sub-queries.
            results: Individual results.

        Returns:
            Combined result string.
        """
        combined_parts = []

        for idx, sub_query in enumerate(sub_queries):
            result = results.get(idx, "No result available")
            combined_parts.append(f"**{sub_query.project_name}**: {result}")

        return "\n\n".join(combined_parts)
