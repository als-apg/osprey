"""Analysis Node for XOpt Optimizer Service.

This node analyzes XOpt results and decides whether to continue
with additional iterations or complete the optimization.

When the run_artifact contains real data from the tuning_scripts API
(indicated by the presence of a ``job_id`` and ``data`` fields), the
node extracts the best point and generates meaningful recommendations.
Otherwise it falls back to the placeholder analysis path for testing.
"""

from typing import Any

from osprey.utils.logger import get_logger

from ..models import XOptExecutionState

logger = get_logger("xopt_optimizer")


def _analyze_real_data(run_artifact: dict[str, Any]) -> dict[str, Any]:
    """Analyze real optimization data from the tuning_scripts API.

    Args:
        run_artifact: Full result dict containing ``data``, ``objective_name``,
            ``variable_names``, etc.

    Returns:
        Analysis dict with ``best_point``, ``total_evaluations``, and ``summary``.
    """
    data = run_artifact.get("data") or []
    objective_name = run_artifact.get("objective_name")
    variable_names = run_artifact.get("variable_names") or []

    analysis: dict[str, Any] = {
        "total_evaluations": len(data),
        "objective_name": objective_name,
        "variable_names": variable_names,
        "job_id": run_artifact.get("job_id"),
    }

    if not data or not objective_name:
        analysis["summary"] = "No data or objective to analyze"
        analysis["best_point"] = None
        return analysis

    # Find best point (maximize by default — Xopt convention)
    best_record = max(
        (r for r in data if objective_name in r and r[objective_name] is not None),
        key=lambda r: r[objective_name],
        default=None,
    )

    if best_record is not None:
        best_value = best_record[objective_name]
        best_vars = {v: best_record.get(v) for v in variable_names if v in best_record}
        analysis["best_point"] = {"objective_value": best_value, "variables": best_vars}
        analysis["summary"] = (
            f"Best {objective_name} = {best_value:.6g} "
            f"achieved at {', '.join(f'{k}={v:.4g}' for k, v in best_vars.items() if v is not None)}"
        )
    else:
        analysis["best_point"] = None
        analysis["summary"] = f"No valid evaluations found for objective '{objective_name}'"

    return analysis


def _generate_real_recommendations(
    analysis: dict[str, Any], iteration: int, max_iterations: int, should_continue: bool
) -> list[str]:
    """Generate recommendations based on real optimization results."""
    recs: list[str] = []
    best = analysis.get("best_point")
    total_evals = analysis.get("total_evaluations", 0)

    if best:
        recs.append(f"Best result: {analysis['summary']}")
        recs.append(f"Total evaluations: {total_evals}")

    if should_continue:
        recs.append(f"Continuing to iteration {iteration + 1}/{max_iterations}")
    else:
        recs.append(f"Optimization complete after {iteration} iteration(s)")
        if best and best.get("variables"):
            recs.append(
                "Recommended setpoints: "
                + ", ".join(f"{k}={v:.4g}" for k, v in best["variables"].items() if v is not None)
            )

    return recs


def create_analysis_node():
    """Create the analysis node for LangGraph integration.

    This factory function creates a node that analyzes XOpt results
    and decides whether to continue the optimization loop.

    Returns:
        Async function that takes XOptExecutionState and returns state updates
    """

    async def analysis_node(state: XOptExecutionState) -> dict[str, Any]:
        """Analyze XOpt results and decide whether to continue.

        Routes real API data through ``_analyze_real_data`` when available,
        otherwise falls back to the placeholder path.
        """
        node_logger = get_logger("xopt_optimizer", state=state)
        node_logger.status("Analyzing optimization results...")

        run_artifact = state.get("run_artifact") or {}
        iteration = state.get("iteration_count", 0) + 1
        max_iterations = state.get("max_iterations", 3)

        should_continue = iteration < max_iterations

        # ----- Real data path -----
        has_real_data = run_artifact.get("job_id") and run_artifact.get("data")

        if has_real_data:
            node_logger.info(f"Analyzing real data from job {run_artifact['job_id']}")
            analysis_result = _analyze_real_data(run_artifact)
            analysis_result["iteration"] = iteration
            analysis_result["max_iterations"] = max_iterations
            analysis_result["should_continue"] = should_continue
            analysis_result["status"] = "success"

            recommendations = _generate_real_recommendations(
                analysis_result, iteration, max_iterations, should_continue
            )
        else:
            # ----- Placeholder path (backward compatible) -----
            analysis_result = {
                "status": "PLACEHOLDER_TEST_SUCCESS",
                "message": "XOpt optimizer service workflow test completed successfully",
                "iteration": iteration,
                "max_iterations": max_iterations,
                "run_artifact": run_artifact,
                "should_continue": should_continue,
                "note": (
                    "This is a placeholder implementation. All subsystems (state identification, "
                    "decision, config generation, approval, execution, analysis) executed successfully "
                    "with placeholder logic. Real optimization will be implemented when domain "
                    "requirements are defined by facility operators."
                ),
            }

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
