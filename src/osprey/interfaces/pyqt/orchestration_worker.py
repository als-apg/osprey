"""
Background worker thread for orchestrated multi-project queries

This module provides the OrchestrationWorker class which handles orchestrated
query execution in a background thread to keep the GUI responsive.
"""

from typing import Any, Dict
import time
from PyQt5.QtCore import pyqtSignal

from osprey.interfaces.pyqt.base_worker import BaseWorker
from osprey.utils.logger import get_logger

logger = get_logger("orchestration_worker")


class OrchestrationWorker(BaseWorker):
    """
    Background worker thread for orchestrated multi-project query execution.
    
    This worker executes orchestrated queries in a separate thread to prevent
    blocking the GUI. It emits signals to communicate progress and results
    back to the main GUI thread.
    
    Signals:
        sub_query_start: Emitted when a sub-query starts (index, project_name, query)
        sub_query_complete: Emitted when a sub-query completes (index, result)
        sub_query_error: Emitted when a sub-query fails (index, error_message)
        synthesis_start: Emitted when result synthesis begins
        final_result: Emitted with the final combined result
        error_occurred: Inherited from BaseWorker
        processing_complete: Inherited from BaseWorker
    """
    
    sub_query_start = pyqtSignal(int, str, str)  # (index, project_name, query)
    sub_query_complete = pyqtSignal(int, str)  # (index, result)
    sub_query_error = pyqtSignal(int, str)  # (index, error_message)
    synthesis_start = pyqtSignal()
    final_result = pyqtSignal(str)  # (combined_result)
    
    def __init__(
        self,
        plan,
        project_contexts: Dict[str, Any],
        base_config: Dict[str, Any],
        router
    ):
        """
        Initialize the orchestration worker.
        
        Args:
            plan: OrchestrationPlan with sub-queries
            project_contexts: Dictionary mapping project names to ProjectContext objects
            base_config: Base configuration dictionary
            router: MultiProjectRouter instance for result synthesis
        """
        super().__init__()
        self.plan = plan
        self.project_contexts = project_contexts
        self.base_config = base_config
        self.router = router
    
    def execute(self):
        """
        Execute orchestrated query processing in background thread.
        
        This method is called by BaseWorker.run() with an active event loop.
        """
        try:
            # Execute each sub-query
            results = {}
            for idx, sub_query in enumerate(self.plan.sub_queries):
                # Check if we should stop
                if self.should_stop():
                    logger.info("Worker stop requested during orchestration")
                    return
                
                try:
                    # Record start time for analytics
                    sub_query_start_time = time.time()
                    
                    # Emit start signal
                    self.sub_query_start.emit(idx, sub_query.project_name, sub_query.query)
                    
                    # Get project context
                    project = self.project_contexts.get(sub_query.project_name)
                    if not project:
                        error_msg = f"Project not found: {sub_query.project_name}"
                        self.sub_query_error.emit(idx, error_msg)
                        results[idx] = f"Error: {error_msg}"
                        
                        # Record failed routing in analytics
                        if self.router and self.router.analytics_enabled and self.router.analytics:
                            routing_time_ms = (time.time() - sub_query_start_time) * 1000
                            self.router.analytics.record_routing(
                                query=sub_query.query,
                                project_selected=sub_query.project_name,
                                confidence=0.0,
                                routing_time_ms=routing_time_ms,
                                cache_hit=False,
                                mode="orchestration",
                                reasoning=f"Sub-query {idx + 1} of multi-project orchestration",
                                alternative_projects=[],
                                success=False,
                                error=error_msg
                            )
                        continue
                    
                    # Initialize global registry for this project
                    project.initialize_global_registry()
                    
                    # Get project's graph
                    project_graph = project.graph
                    if not project_graph:
                        error_msg = f"Project {sub_query.project_name} has no graph loaded"
                        self.sub_query_error.emit(idx, error_msg)
                        results[idx] = f"Error: {error_msg}"
                        continue
                    
                    # Create project-specific config
                    project_config = {
                        "configurable": {
                            **self.base_config["configurable"],
                            "thread_id": f"{sub_query.project_name}_{idx}",
                            "session_id": f"{sub_query.project_name}_{idx}"
                        },
                        "recursion_limit": self.base_config.get("recursion_limit", 100)
                    }
                    
                    # Process the message through project's gateway
                    result = self.run_async(
                        project.gateway.process_message(
                            sub_query.query,
                            project_graph,
                            project_config
                        )
                    )
                    
                    # Execute graph if we have agent_state
                    if result.agent_state and not result.error:
                        final_state = self.run_async(
                            self._execute_graph(project_graph, result.agent_state, project_config)
                        )
                        
                        # Extract response from final state
                        response = self._extract_response(final_state)
                        self.sub_query_complete.emit(idx, response)
                        
                        # Record successful routing in analytics
                        if self.router and self.router.analytics_enabled and self.router.analytics:
                            routing_time_ms = (time.time() - sub_query_start_time) * 1000
                            self.router.analytics.record_routing(
                                query=sub_query.query,
                                project_selected=sub_query.project_name,
                                confidence=1.0,  # Sub-queries are pre-assigned to projects
                                routing_time_ms=routing_time_ms,
                                cache_hit=False,
                                mode="orchestration",
                                reasoning=f"Sub-query {idx + 1} of multi-project orchestration",
                                alternative_projects=[],
                                success=True
                            )
                    elif result.error:
                        response = f"Error: {result.error}"
                        self.sub_query_error.emit(idx, response)
                        
                        # Record failed routing in analytics
                        if self.router and self.router.analytics_enabled and self.router.analytics:
                            routing_time_ms = (time.time() - sub_query_start_time) * 1000
                            self.router.analytics.record_routing(
                                query=sub_query.query,
                                project_selected=sub_query.project_name,
                                confidence=1.0,
                                routing_time_ms=routing_time_ms,
                                cache_hit=False,
                                mode="orchestration",
                                reasoning=f"Sub-query {idx + 1} of multi-project orchestration",
                                alternative_projects=[],
                                success=False,
                                error=result.error
                            )
                    else:
                        response = "No response generated"
                        self.sub_query_error.emit(idx, response)
                        
                        # Record failed routing in analytics
                        if self.router and self.router.analytics_enabled and self.router.analytics:
                            routing_time_ms = (time.time() - sub_query_start_time) * 1000
                            self.router.analytics.record_routing(
                                query=sub_query.query,
                                project_selected=sub_query.project_name,
                                confidence=1.0,
                                routing_time_ms=routing_time_ms,
                                cache_hit=False,
                                mode="orchestration",
                                reasoning=f"Sub-query {idx + 1} of multi-project orchestration",
                                alternative_projects=[],
                                success=False,
                                error="No response generated"
                            )
                    
                    results[idx] = response
                    
                except Exception as e:
                    error_msg = f"Execution failed: {e}"
                    logger.error(f"Sub-query {idx} failed: {e}")
                    self.sub_query_error.emit(idx, error_msg)
                    results[idx] = f"Error: {error_msg}"
                    
                    # Record failed routing in analytics
                    if self.router and self.router.analytics_enabled and self.router.analytics:
                        routing_time_ms = (time.time() - sub_query_start_time) * 1000
                        self.router.analytics.record_routing(
                            query=sub_query.query,
                            project_selected=sub_query.project_name,
                            confidence=1.0,
                            routing_time_ms=routing_time_ms,
                            cache_hit=False,
                            mode="orchestration",
                            reasoning=f"Sub-query {idx + 1} of multi-project orchestration",
                            alternative_projects=[],
                            success=False,
                            error=str(e)
                        )
            
            # Check if stopped before synthesis
            if self.should_stop():
                return
            
            # Synthesize results
            self.synthesis_start.emit()
            combined_result = self.router.orchestrator._combine_results(self.plan, results)
            self.final_result.emit(combined_result)
            
        except Exception as e:
            self.handle_error(e, "Orchestration processing")
    
    async def _execute_graph(self, graph, agent_state, config):
        """
        Execute graph to completion.
        
        Args:
            graph: Project's graph instance
            agent_state: Initial agent state
            config: Configuration dictionary
            
        Returns:
            Final state from graph execution
        """
        async for chunk in graph.astream(
            agent_state,
            config=config,
            stream_mode="custom"
        ):
            # Check if we should stop
            if self.should_stop():
                logger.info("Worker stop requested during graph execution")
                break
        
        # Get final state
        final_state = graph.get_state(config=config)
        return final_state
    
    def _extract_response(self, final_state):
        """
        Extract response from final state.
        
        Args:
            final_state: Final state from graph execution
            
        Returns:
            str: Extracted response or error message
        """
        if final_state and final_state.values:
            messages = final_state.values.get("messages", [])
            if messages:
                # Get the last AI message
                for msg in reversed(messages):
                    if hasattr(msg, 'content') and msg.content:
                        if not hasattr(msg, 'type') or msg.type != 'human':
                            return msg.content
                return "No response generated"
            return "No response generated"
        return "No response generated"


