"""
Multi-Project Router for GUI Support

This module provides the MultiProjectRouter class for intelligently routing
user queries to the appropriate project/capability using LLM-based analysis.

Key Features:
- LLM-based query analysis and project selection
- Confidence scores for routing decisions
- Transparent routing explanations
- Fallback strategies for errors
- Support for both automatic and manual routing modes
- Routing decision caching for improved performance
- Conversation-aware routing with topic detection
"""

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List

from osprey.utils.logger import get_logger
from osprey.interfaces.pyqt.llm_client import SimpleLLMClient
from osprey.interfaces.pyqt.routing_cache import RoutingCache, CacheStatistics
from osprey.interfaces.pyqt.conversation_context import ConversationContext
from osprey.interfaces.pyqt.semantic_context_analyzer import SemanticContextAnalyzer
from osprey.interfaces.pyqt.multi_project_orchestrator import (
    MultiProjectOrchestrator,
    OrchestrationPlan,
    OrchestrationResult
)
from osprey.interfaces.pyqt.routing_analytics import RoutingAnalytics
from osprey.interfaces.pyqt.routing_feedback import RoutingFeedback

if TYPE_CHECKING:
    from osprey.interfaces.pyqt.capability_registry import CapabilityRegistry
    from osprey.interfaces.pyqt.project_manager import ProjectContext

logger = get_logger("multi_project_router")


@dataclass
class RoutingDecision:
    """Result of routing decision."""
    project_name: str
    capability_name: str | None = None
    confidence: float = 0.0
    reasoning: str = ""
    alternative_projects: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    from_cache: bool = False  # Indicates if decision came from cache
    routing_time_ms: float = 0.0  # Time taken for routing decision


class MultiProjectRouter:
    """
    Routes user queries to the appropriate project/capability.

    Uses LLM to analyze queries and select the best project based on:
    - Available capabilities in each project
    - Query content and intent
    - Capability descriptions and tags
    - Context from previous queries

    Supports two modes:
    - Automatic: LLM selects best project for each query
    - Manual: User-selected project for all queries
    """

    def __init__(
        self,
        capability_registry: 'CapabilityRegistry',
        llm_config: Dict[str, Any] = None,
        enable_cache: bool = True,
        cache_max_size: int = 100,
        cache_ttl_seconds: float = 3600.0,
        cache_similarity_threshold: float = 0.85,
        enable_advanced_invalidation: bool = True,
        enable_adaptive_ttl: bool = True,
        enable_probabilistic_expiration: bool = True,
        enable_event_driven_invalidation: bool = True,
        enable_conversation_context: bool = True,
        enable_semantic_context: bool = False,
        context_max_history: int = 10,
        context_confidence_boost: float = 0.2,
        semantic_similarity_threshold: float = 0.5,
        semantic_topic_threshold: float = 0.6,
        enable_orchestration: bool = True,
        orchestration_max_parallel: int = 3,
        enable_analytics: bool = True,
        analytics_max_history: int = 1000,
        enable_feedback: bool = True,
        feedback_max_history: int = 1000
    ):
        """Initialize router.

        Args:
            capability_registry: Global capability registry.
            llm_config: Optional LLM configuration for routing.
            enable_cache: Whether to enable routing cache (default: True).
            cache_max_size: Maximum number of cache entries (default: 100).
            cache_ttl_seconds: Cache entry TTL in seconds (default: 3600).
            cache_similarity_threshold: Similarity threshold for cache hits (default: 0.85).
            enable_advanced_invalidation: Enable advanced cache invalidation (default: True).
            enable_adaptive_ttl: Enable adaptive TTL for cache entries (default: True).
            enable_probabilistic_expiration: Enable probabilistic early expiration (default: True).
            enable_event_driven_invalidation: Enable event-driven invalidation (default: True).
            enable_conversation_context: Whether to enable conversation-aware routing (default: True).
            enable_semantic_context: Whether to use semantic context analyzer instead of simple context (default: False).
            context_max_history: Maximum conversation history to track (default: 10).
            context_confidence_boost: Confidence boost for topic continuity (default: 0.2).
            semantic_similarity_threshold: Similarity threshold for semantic context (default: 0.5).
            semantic_topic_threshold: Topic similarity threshold for semantic clustering (default: 0.6).
            enable_orchestration: Whether to enable multi-project orchestration (default: True).
            orchestration_max_parallel: Maximum parallel sub-query executions (default: 3).
            enable_analytics: Whether to enable routing analytics (default: True).
            analytics_max_history: Maximum analytics history to keep (default: 1000).
            enable_feedback: Whether to enable user feedback collection (default: True).
            feedback_max_history: Maximum feedback history to keep (default: 1000).
        """
        self.logger = logger
        self.capability_registry = capability_registry
        self.llm_config = llm_config or {}
        self._last_routing_explanation = ""
        self.routing_mode = "automatic"  # or "manual"
        self.manual_project = None

        # Initialize SimpleLLMClient for routing (NO SINGLETON DEPENDENCY!)
        if llm_config:
            # Use provided LLM config
            self.llm_client = SimpleLLMClient(
                provider=llm_config.get('provider', 'anthropic'),
                model_id=llm_config.get('model_id', 'claude-3-sonnet-20240229'),
                api_key=llm_config.get('api_key'),
                base_url=llm_config.get('base_url')
            )
            self.logger.info(
                f"Initialized routing LLM client from config: "
                f"{llm_config.get('provider')}/{llm_config.get('model_id')}"
            )
        else:
            # Use GUI config (reads user's configured 'classifier' model)
            try:
                self.llm_client = SimpleLLMClient.from_gui_config()
                self.logger.info(
                    "Initialized routing LLM client from gui_config.yml "
                    f"({self.llm_client.provider}/{self.llm_client.model_id})"
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to initialize LLM client from gui_config.yml: {e}. "
                    f"Routing will fail until LLM client is properly configured."
                )
                self.llm_client = None

        # Initialize cache
        self.cache_enabled = enable_cache
        if self.cache_enabled:
            self.cache = RoutingCache(
                max_size=cache_max_size,
                ttl_seconds=cache_ttl_seconds,
                similarity_threshold=cache_similarity_threshold
            )
            self.logger.info(
                f"Initialized MultiProjectRouter with caching enabled "
                f"(max_size={cache_max_size}, ttl={cache_ttl_seconds}s)"
            )
        else:
            self.cache = None
            self.logger.info("Initialized MultiProjectRouter with caching disabled")

        # Initialize conversation/semantic context
        self.context_enabled = enable_conversation_context
        self.semantic_enabled = enable_semantic_context

        if self.semantic_enabled:
            # Use semantic context analyzer for advanced routing
            self.conversation_context = SemanticContextAnalyzer(
                max_history=context_max_history,
                similarity_threshold=semantic_similarity_threshold,
                topic_similarity_threshold=semantic_topic_threshold,
                enable_intent_recognition=True
            )
            self.logger.info(
                f"Initialized semantic context analyzer "
                f"(max_history={context_max_history}, "
                f"similarity_threshold={semantic_similarity_threshold}, "
                f"topic_threshold={semantic_topic_threshold})"
            )
        elif self.context_enabled:
            # Use simple conversation context
            self.conversation_context = ConversationContext(
                max_history=context_max_history,
                confidence_boost=context_confidence_boost
            )
            self.logger.info(
                f"Initialized conversation-aware routing "
                f"(max_history={context_max_history}, boost={context_confidence_boost})"
            )
        else:
            self.conversation_context = None
            self.logger.info("Conversation-aware routing disabled")

        # Initialize orchestrator
        self.orchestration_enabled = enable_orchestration
        if self.orchestration_enabled:
            self.orchestrator = MultiProjectOrchestrator(
                llm_config=llm_config,
                max_parallel_executions=orchestration_max_parallel
            )
            self.logger.info(
                f"Initialized multi-project orchestration "
                f"(max_parallel={orchestration_max_parallel})"
            )
        else:
            self.orchestrator = None
            self.logger.info("Multi-project orchestration disabled")

        # Initialize analytics
        self.analytics_enabled = enable_analytics
        if self.analytics_enabled:
            self.analytics = RoutingAnalytics(
                max_history=analytics_max_history,
                enable_persistence=True
            )
            self.logger.info(
                f"Initialized routing analytics "
                f"(max_history={analytics_max_history})"
            )
        else:
            self.analytics = None
            self.logger.info("Routing analytics disabled")

        # Initialize feedback system
        self.feedback_enabled = enable_feedback
        if self.feedback_enabled:
            self.feedback = RoutingFeedback(
                max_history=feedback_max_history,
                enable_persistence=True
            )
            self.logger.info(
                f"Initialized routing feedback "
                f"(max_history={feedback_max_history})"
            )
        else:
            self.feedback = None
            self.logger.info("Routing feedback disabled")

    def route_query(
        self,
        query: str,
        available_projects: List['ProjectContext']
    ) -> RoutingDecision:
        """Route a user query to the best project.

        In AUTOMATIC mode (default):
        - Analyzes query against ALL available projects
        - Selects best match using LLM
        - Can switch between projects for each query

        In MANUAL mode:
        - Uses user-selected project
        - Bypasses LLM routing

        Args:
            query: User's query/question.
            available_projects: List of available ProjectContext objects.

        Returns:
            RoutingDecision with selected project and reasoning.

        Raises:
            RoutingError: If routing fails.
        """
        if not available_projects:
            raise RoutingError("No projects available for routing")

        # Manual mode: use selected project (no caching)
        if self.routing_mode == "manual" and self.manual_project:
            decision = RoutingDecision(
                project_name=self.manual_project,
                confidence=1.0,
                reasoning="Manual selection by user",
                alternative_projects=[],
                from_cache=False
            )
            self._last_routing_explanation = decision.reasoning
            return decision

        # If only one project, select it (no caching needed)
        if len(available_projects) == 1:
            project = available_projects[0]
            decision = RoutingDecision(
                project_name=project.metadata.name,
                confidence=1.0,
                reasoning="Only one project available",
                from_cache=False
            )
            self._last_routing_explanation = decision.reasoning
            return decision

        # Check cache first (automatic mode only)
        enabled_project_names = [p.metadata.name for p in available_projects]
        if self.cache_enabled and self.cache:
            cached_decision = self.cache.get(query, enabled_project_names)
            if cached_decision:
                # Convert cached decision to RoutingDecision
                decision = RoutingDecision(
                    project_name=cached_decision.project_name,
                    confidence=cached_decision.confidence,
                    reasoning=f"{cached_decision.reasoning} (from cache)",
                    alternative_projects=cached_decision.alternative_projects,
                    from_cache=True
                )
                self._last_routing_explanation = decision.reasoning
                self.logger.info(
                    f"Cache hit: Routed query to {decision.project_name} "
                    f"(confidence: {decision.confidence:.2f})"
                )
                return decision

        # Automatic mode: LLM-based routing
        try:
            # Generate capability descriptions
            capability_descriptions = self._generate_capability_descriptions(
                available_projects
            )

            # Create routing prompt
            routing_prompt = self._create_routing_prompt(
                query,
                capability_descriptions,
                available_projects
            )

            # Call LLM for routing
            routing_result = self._call_llm_for_routing(routing_prompt)
            decision = self._parse_routing_result(routing_result, available_projects)
            decision.from_cache = False

            # Apply feedback learning adjustments if enabled
            if self.feedback_enabled and self.feedback:
                adjusted_project, adjusted_confidence, feedback_reasoning = \
                    self.feedback.get_routing_adjustment(
                        query,
                        decision.project_name,
                        decision.confidence
                    )

                if adjusted_project != decision.project_name:
                    # Feedback suggests different project
                    self.logger.info(
                        f"Feedback adjustment: {decision.project_name} → {adjusted_project} "
                        f"(confidence: {decision.confidence:.2f} → {adjusted_confidence:.2f})"
                    )
                    decision.project_name = adjusted_project
                    decision.confidence = adjusted_confidence
                    decision.reasoning = f"{feedback_reasoning}; Original: {decision.reasoning}"
                elif adjusted_confidence != decision.confidence:
                    # Confidence adjustment only
                    self.logger.info(
                        f"Feedback confidence adjustment: "
                        f"{decision.confidence:.2f} → {adjusted_confidence:.2f}"
                    )
                    decision.confidence = adjusted_confidence
                    if feedback_reasoning:
                        decision.reasoning += f"; {feedback_reasoning}"

            # Apply conversation/semantic context boost if enabled
            if self.conversation_context:
                if self.semantic_enabled:
                    # Use semantic context boost
                    should_boost, boost_amount, boost_reason = \
                        self.conversation_context.should_boost_project(
                            query, decision.project_name
                        )

                    if should_boost:
                        original_confidence = decision.confidence
                        decision.confidence = min(1.0, decision.confidence + boost_amount)
                        decision.reasoning += f" ({boost_reason})"
                        self.logger.info(
                            f"Applied semantic context boost: "
                            f"{original_confidence:.2f} → {decision.confidence:.2f} "
                            f"({boost_reason})"
                        )
                else:
                    # Use simple conversation context boost
                    boost = self.conversation_context.get_confidence_boost(decision.project_name)
                    if boost > 0:
                        original_confidence = decision.confidence
                        decision.confidence = min(1.0, decision.confidence + boost)
                        decision.reasoning += f" (conversation context boost: +{boost:.0%})"
                        self.logger.info(
                            f"Applied conversation context boost: "
                            f"{original_confidence:.2f} → {decision.confidence:.2f}"
                        )

            self._last_routing_explanation = decision.reasoning

            # Store in cache
            if self.cache_enabled and self.cache:
                self.cache.put(
                    query=query,
                    enabled_projects=enabled_project_names,
                    project_name=decision.project_name,
                    confidence=decision.confidence,
                    reasoning=decision.reasoning,
                    alternative_projects=decision.alternative_projects
                )

            # Add to conversation/semantic context
            if self.conversation_context:
                if self.semantic_enabled:
                    # Semantic context uses 'project' parameter
                    self.conversation_context.add_query(
                        query=query,
                        project=decision.project_name,
                        confidence=decision.confidence
                    )
                else:
                    # Simple context uses 'project_name' parameter
                    self.conversation_context.add_query(
                        query=query,
                        project_name=decision.project_name,
                        confidence=decision.confidence,
                        reasoning=decision.reasoning
                    )

            # Record analytics
            if self.analytics_enabled and self.analytics:
                self.analytics.record_routing(
                    query=query,
                    project_selected=decision.project_name,
                    confidence=decision.confidence,
                    routing_time_ms=decision.routing_time_ms,
                    cache_hit=decision.from_cache,
                    mode=self.routing_mode,
                    reasoning=decision.reasoning,
                    alternative_projects=decision.alternative_projects,
                    success=True
                )

            self.logger.info(
                f"Routed query to {decision.project_name} "
                f"(confidence: {decision.confidence:.2f})"
            )
            return decision

        except Exception as e:
            self.logger.error(f"LLM routing failed: {e}")
            # Fallback to first project
            project = available_projects[0]
            decision = RoutingDecision(
                project_name=project.metadata.name,
                confidence=0.5,
                reasoning=f"LLM routing failed, using fallback: {str(e)}",
                from_cache=False
            )
            self._last_routing_explanation = decision.reasoning

            # Record failed routing in analytics
            if self.analytics_enabled and self.analytics:
                self.analytics.record_routing(
                    query=query,
                    project_selected=decision.project_name,
                    confidence=decision.confidence,
                    routing_time_ms=0.0,
                    cache_hit=False,
                    mode=self.routing_mode,
                    reasoning=decision.reasoning,
                    success=False,
                    error=str(e)
                )

            return decision

    def get_routing_explanation(self) -> str:
        """Get explanation of last routing decision."""
        return self._last_routing_explanation

    def set_automatic_mode(self):
        """Enable automatic routing (default)."""
        self.routing_mode = "automatic"
        self.manual_project = None
        self.logger.info("Switched to automatic routing mode")

    def set_manual_mode(self, project_name: str):
        """Enable manual mode with specific project.

        Args:
            project_name: Name of project to use for all queries.
        """
        self.routing_mode = "manual"
        self.manual_project = project_name
        self.logger.info(f"Switched to manual routing mode: {project_name}")

    def is_automatic_mode(self) -> bool:
        """Check if router is in automatic mode."""
        return self.routing_mode == "automatic"

    def get_cache_statistics(self) -> CacheStatistics | None:
        """Get cache statistics.

        Returns:
            CacheStatistics if caching is enabled, None otherwise.
        """
        if self.cache_enabled and self.cache:
            return self.cache.get_statistics()
        return None

    def clear_cache(self):
        """Clear routing cache."""
        if self.cache_enabled and self.cache:
            self.cache.clear()
            self.logger.info("Routing cache cleared")

    def enable_cache(self):
        """Enable routing cache."""
        if not self.cache_enabled:
            self.cache = RoutingCache()
            self.cache_enabled = True
            self.logger.info("Routing cache enabled")

    def disable_cache(self):
        """Disable routing cache."""
        if self.cache_enabled:
            self.cache_enabled = False
            self.cache = None
            self.logger.info("Routing cache disabled")

    def get_conversation_context_summary(self) -> str:
        """Get conversation context summary.

        Returns:
            Human-readable summary of conversation context.
        """
        if self.context_enabled and self.conversation_context:
            return self.conversation_context.get_context_summary()
        return "Conversation context disabled"

    def clear_conversation_context(self):
        """Clear conversation context history."""
        if self.context_enabled and self.conversation_context:
            self.conversation_context.clear()
            self.logger.info("Conversation context cleared")

    def enable_conversation_context(self):
        """Enable conversation-aware routing."""
        if not self.context_enabled:
            self.conversation_context = ConversationContext()
            self.context_enabled = True
            self.logger.info("Conversation-aware routing enabled")

    def disable_conversation_context(self):
        """Disable conversation-aware routing."""
        if self.context_enabled:
            self.context_enabled = False
            self.conversation_context = None
            self.logger.info("Conversation-aware routing disabled")

    def enable_orchestration(self):
        """Enable multi-project orchestration."""
        if not self.orchestration_enabled:
            self.orchestrator = MultiProjectOrchestrator(llm_config=self.llm_config)
            self.orchestration_enabled = True
            self.logger.info("Multi-project orchestration enabled")

    def disable_orchestration(self):
        """Disable multi-project orchestration."""
        if self.orchestration_enabled:
            self.orchestration_enabled = False
            self.orchestrator = None
            self.logger.info("Multi-project orchestration disabled")

    def analyze_for_orchestration(
        self,
        query: str,
        available_projects: List['ProjectContext']
    ) -> OrchestrationPlan:
        """Analyze query for multi-project orchestration needs.

        Args:
            query: User's query.
            available_projects: List of available projects.

        Returns:
            OrchestrationPlan indicating if orchestration is needed.
        """
        if not self.orchestration_enabled or not self.orchestrator:
            # Return empty plan if orchestration disabled
            from osprey.interfaces.pyqt.multi_project_orchestrator import OrchestrationPlan
            return OrchestrationPlan(
                original_query=query,
                sub_queries=[],
                is_multi_project=False,
                reasoning="Orchestration disabled"
            )

        return self.orchestrator.analyze_query(query, available_projects)

    def get_analytics(self) -> 'RoutingAnalytics' | None:
        """Get routing analytics instance.

        Returns:
            RoutingAnalytics if enabled, None otherwise.
        """
        return self.analytics if self.analytics_enabled else None

    def enable_analytics(self):
        """Enable routing analytics."""
        if not self.analytics_enabled:
            self.analytics = RoutingAnalytics()
            self.analytics_enabled = True
            self.logger.info("Routing analytics enabled")

    def disable_analytics(self):
        """Disable routing analytics."""
        if self.analytics_enabled:
            self.analytics_enabled = False
            self.analytics = None
            self.logger.info("Routing analytics disabled")

    def record_routing_feedback(
        self,
        query: str,
        selected_project: str,
        confidence: float,
        user_feedback: str,
        correct_project: str | None = None,
        reasoning: str = ""
    ):
        """Record user feedback on a routing decision.

        Args:
            query: User query.
            selected_project: Project that was selected.
            confidence: Routing confidence.
            user_feedback: "correct" or "incorrect".
            correct_project: If incorrect, the correct project.
            reasoning: Routing reasoning.
        """
        if not self.feedback_enabled or not self.feedback:
            self.logger.warning("Feedback system not enabled")
            return

        self.feedback.record_feedback(
            query=query,
            selected_project=selected_project,
            confidence=confidence,
            user_feedback=user_feedback,
            correct_project=correct_project,
            reasoning=reasoning
        )

        self.logger.info(
            f"Recorded {user_feedback} feedback for routing: "
            f"{query[:50]}... → {selected_project}"
        )

    def get_feedback_stats(self, project_name: str) -> Dict[str, Any]:
        """Get feedback statistics for a project.

        Args:
            project_name: Name of project.

        Returns:
            Dictionary with feedback statistics.
        """
        if not self.feedback_enabled or not self.feedback:
            return {}

        return self.feedback.get_project_feedback_stats(project_name)

    def enable_feedback(self):
        """Enable routing feedback collection."""
        if not self.feedback_enabled:
            self.feedback = RoutingFeedback()
            self.feedback_enabled = True
            self.logger.info("Routing feedback enabled")

    def disable_feedback(self):
        """Disable routing feedback collection."""
        if self.feedback_enabled:
            self.feedback_enabled = False
            self.feedback = None
            self.logger.info("Routing feedback disabled")

    # Private methods

    def _generate_capability_descriptions(
        self,
        projects: List['ProjectContext']
    ) -> str:
        """Generate descriptions of all capabilities for LLM.

        Args:
            projects: List of ProjectContext objects.

        Returns:
            Formatted string with capability descriptions.
        """
        descriptions = []

        for project in projects:
            descriptions.append(f"\n## Project: {project.metadata.name}")
            descriptions.append(f"Description: {project.metadata.description}")
            descriptions.append(f"Version: {project.metadata.version}")

            # Get capabilities for this project
            capabilities = self.capability_registry.get_capabilities_by_project(
                project.metadata.name
            )

            if not capabilities:
                descriptions.append("Capabilities: (No capabilities registered)")
            else:
                descriptions.append("Capabilities:")
                for cap_name in capabilities.keys():
                    cap_desc = self.capability_registry.get_capability_description(
                        cap_name,
                        project.metadata.name
                    )
                    descriptions.append(f"  - {cap_desc}")

        return "\n".join(descriptions)

    def _create_routing_prompt(
        self,
        query: str,
        capability_descriptions: str,
        projects: List['ProjectContext']
    ) -> str:
        """Create prompt for LLM routing decision.

        Args:
            query: User's query.
            capability_descriptions: Descriptions of available capabilities.
            projects: Available projects.

        Returns:
            Formatted prompt for LLM.
        """
        project_names = [p.metadata.name for p in projects]

        # Build base prompt
        prompt_parts = [
            "You are a routing system that directs user queries to the appropriate AI agent/project.",
            "",
            "Available Projects and Capabilities:",
            capability_descriptions,
            ""
        ]

        # Add conversation context if available
        if self.conversation_context and not self.semantic_enabled:
            # Only simple context has get_context_for_routing
            context_info = self.conversation_context.get_context_for_routing()

            if context_info.get("has_history"):
                prompt_parts.append("Conversation Context:")

                # Add recent queries
                if context_info.get("recent_queries"):
                    prompt_parts.append("Recent queries in this conversation:")
                    for i, rec in enumerate(context_info["recent_queries"], 1):
                        prompt_parts.append(
                            f"  {i}. \"{rec['query']}\" → {rec['project']} "
                            f"(confidence: {rec['confidence']:.0%})"
                        )
                    prompt_parts.append("")

                # Add active topic
                if context_info.get("active_topic"):
                    topic = context_info["active_topic"]
                    prompt_parts.append(
                        f"Active conversation topic: {topic['project']} "
                        f"({topic['query_count']} related queries, "
                        f"{topic['confidence']:.0%} confidence)"
                    )
                    prompt_parts.append(
                        "Consider topic continuity when routing - users often ask "
                        "follow-up questions about the same topic."
                    )
                    prompt_parts.append("")

        # Add user query
        prompt_parts.extend([
            f"User Query: {query}",
            "",
            "Based on the user's query and the available capabilities, determine which project should handle this query.",
            "",
            "Respond in the following format:",
            "PROJECT: <project_name>",
            "CONFIDENCE: <0.0-1.0>",
            "REASONING: <brief explanation of why this project was selected>",
            "ALTERNATIVES: <comma-separated list of alternative projects that could handle this>",
            "",
            "Consider:",
            "1. Which project's capabilities best match the query intent",
            "2. The description and purpose of each project",
            "3. The specific capabilities available in each project",
            "4. Any domain-specific knowledge required"
        ])

        # Add conversation context consideration
        if self.conversation_context and not self.semantic_enabled:
            # Only simple context has has_active_topic
            if self.conversation_context.has_active_topic():
                prompt_parts.append("5. Conversation context and topic continuity (if the query relates to the current topic)")

        prompt_parts.extend([
            "",
            "Make your decision based on the best match between the query and available capabilities.",
            "",
            f"Available project names: {', '.join(project_names)}"
        ])

        return "\n".join(prompt_parts)

    def _call_llm_for_routing(self, prompt: str) -> str:
        """Call LLM to make routing decision - NO SINGLETON DEPENDENCY!

        Uses SimpleLLMClient which reads configuration from gui_config.yml
        or explicit config, avoiding the "Registry not initialized" error.

        Args:
            prompt: Routing prompt for LLM.

        Returns:
            LLM response.

        Raises:
            RoutingError: If LLM call fails.
        """
        if not self.llm_client:
            raise RoutingError(
                "LLM client not initialized. Please ensure gui_config.yml has a "
                "'classifier' model configured, or provide llm_config during router initialization."
            )

        try:
            # Direct LLM call - no registry needed!
            response = self.llm_client.call(
                prompt=prompt,
                max_tokens=500,
                temperature=0.0
            )

            return response

        except Exception as e:
            raise RoutingError(f"Failed to call LLM for routing: {e}") from e

    def _parse_routing_result(
        self,
        result: str,
        available_projects: List['ProjectContext']
    ) -> RoutingDecision:
        """Parse LLM routing result.

        Args:
            result: LLM response.
            available_projects: Available projects.

        Returns:
            RoutingDecision.
        """
        lines = result.strip().split('\n')
        decision_data = {}

        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                decision_data[key.strip()] = value.strip()

        project_name = decision_data.get('PROJECT', '')
        confidence_str = decision_data.get('CONFIDENCE', '0.5')
        reasoning = decision_data.get('REASONING', '')
        alternatives_str = decision_data.get('ALTERNATIVES', '')

        # Parse confidence
        try:
            confidence = float(confidence_str)
            # Clamp to valid range
            confidence = max(0.0, min(1.0, confidence))
        except ValueError:
            self.logger.warning(f"Invalid confidence value: {confidence_str}")
            confidence = 0.5

        # Parse alternatives
        alternatives = [
            p.strip() for p in alternatives_str.split(',')
            if p.strip()
        ]

        # Validate project exists
        available_project_names = [p.metadata.name for p in available_projects]
        if project_name not in available_project_names:
            self.logger.warning(
                f"Selected project '{project_name}' not in available projects. "
                f"Using fallback."
            )
            # Use first available project as fallback
            project_name = available_project_names[0]
            reasoning = f"Selected project not found, using fallback: {project_name}"
            confidence = 0.3

        return RoutingDecision(
            project_name=project_name,
            confidence=confidence,
            reasoning=reasoning,
            alternative_projects=alternatives
        )


# Custom Exceptions

class RoutingError(Exception):
    """Raised when routing fails."""
    pass
