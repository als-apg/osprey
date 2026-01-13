"""Analysis Node for XOpt Optimizer Service.

This node analyzes XOpt results and decides whether to continue
with additional iterations or complete the optimization.
"""

from typing import Any

from osprey.utils.logger import get_logger

from ..models import XOptExecutionState

logger = get_logger("xopt_optimizer")


def create_analysis_node():
    """Create the analysis node for LangGraph integration.

    This factory function creates a node that analyzes XOpt results
    and decides whether to continue the optimization loop.

    Returns:
        Async function that takes XOptExecutionState and returns state updates
    """

    async def analysis_node(state: XOptExecutionState) -> dict[str, Any]:
        """Analyze XOpt results and decide whether to continue.

        Simple continuation logic based on iteration count.
        Future implementation may include:
        - Convergence detection
        - Improvement rate analysis
        - Domain-specific completion criteria
        """
        node_logger = get_logger("xopt_optimizer", state=state)
        node_logger.status("Analyzing optimization results...")

        run_artifact = state.get("run_artifact")
        iteration = state.get("iteration_count", 0) + 1
        max_iterations = state.get("max_iterations", 3)

        # Simple continuation logic (can be refined)
        # Future: Add convergence detection, improvement rate analysis, etc.
        should_continue = iteration < max_iterations

        # Generate analysis result
        # NOTE: This is a placeholder implementation for testing the workflow
        analysis_result = {
            "status": "PLACEHOLDER_TEST_SUCCESS",
            "message": "XOpt optimizer service workflow test completed successfully",
            "iteration": iteration,
            "max_iterations": max_iterations,
            "run_artifact": run_artifact,
            "should_continue": should_continue,
            "note": (
                "This is a placeholder implementation. All subsystems (state identification, "
                "decision, YAML generation, approval, execution, analysis) executed successfully "
                "with placeholder logic. Real optimization will be implemented when domain "
                "requirements are defined by facility operators."
            ),
        }

        # Generate recommendations (placeholder - clearly indicate test status)
        recommendations = []
        if should_continue:
            recommendations.append(f"[TEST] Continuing to iteration {iteration + 1}")
        else:
            recommendations.append(
                f"[TEST SUCCESS] XOpt optimizer workflow completed {iteration} iterations successfully"
            )
            recommendations.append(
                "[PLACEHOLDER] All subsystems executed with placeholder logic - "
                "ready for real implementation when domain requirements are defined"
            )
            recommendations.append(
                "[NEXT STEPS] Implement real machine state assessment, YAML generation, "
                "and XOpt execution based on facility-specific requirements"
            )

        node_logger.info(f"Iteration {iteration}/{max_iterations} complete")

        if should_continue:
            node_logger.info("Continuing to next iteration")
            return {
                "analysis_result": analysis_result,
                "recommendations": recommendations,
                "iteration_count": iteration,
                "should_continue": True,
                "current_stage": "state_id",
            }
        else:
            node_logger.info("Optimization complete")
            return {
                "analysis_result": analysis_result,
                "recommendations": recommendations,
                "iteration_count": iteration,
                "should_continue": False,
                "is_successful": True,
                "current_stage": "complete",
            }

    return analysis_node
