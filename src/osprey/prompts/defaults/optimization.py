"""
Optimization Capability Prompt Builder

Default prompts for XOpt optimization capability.
Provides baseline prompts that can be overridden by facility-specific implementations.
"""

import textwrap

from osprey.base import (
    ClassifierActions,
    ClassifierExample,
    OrchestratorExample,
    OrchestratorGuide,
    PlannedStep,
    TaskClassifierGuide,
)
from osprey.prompts.base import FrameworkPromptBuilder
from osprey.registry import get_registry


class DefaultOptimizationPromptBuilder(FrameworkPromptBuilder):
    """Default optimization capability prompt builder.

    Provides baseline prompts for XOpt optimization workflows. Facilities can
    override this builder to inject domain-specific instructions, machine
    state definitions, and optimization strategies.

    Override Points:
        - get_instructions(): Domain-specific optimization guidance
        - get_machine_state_definitions(): Facility-specific machine states
        - get_yaml_generation_guidance(): Historical patterns and templates
        - get_strategy_selection_guidance(): Strategy selection criteria
    """

    PROMPT_TYPE = "optimization"

    def get_role_definition(self) -> str:
        """Get the role definition for optimization.

        :return: Role definition string
        :rtype: str
        """
        return "You are an expert optimization assistant helping to configure and execute autonomous machine optimization using XOpt."

    def get_task_definition(self) -> str:
        """Get the task definition for optimization.

        :return: Task definition or None if task is provided externally
        :rtype: Optional[str]
        """
        return None  # Task is provided via request

    def get_instructions(self) -> str:
        """Get domain-specific optimization instructions.

        These instructions are domain-agnostic and apply to all optimization operations.
        Facilities should override to provide machine-specific guidance.

        :return: Instructions string for optimization workflows
        :rtype: str
        """
        # Placeholder - facilities override with domain-specific instructions
        return textwrap.dedent(
            """
            === OPTIMIZATION INSTRUCTIONS ===

            This is a placeholder for domain-specific optimization instructions.

            When implementing actual optimization:
            1. Assess machine readiness before proceeding
            2. Use appropriate optimization strategy (exploration vs optimization)
            3. Generate valid XOpt YAML configuration
            4. Request human approval before execution
            5. Analyze results and provide recommendations

            NOTE: Actual optimization parameters, channel addresses, and safety limits
            will be defined based on facility-specific requirements.
            """
        ).strip()

    def get_machine_state_definitions(self) -> dict[str, str]:
        """Get facility-specific machine state definitions.

        :return: Mapping of state names to descriptions
        :rtype: Dict[str, str]
        """
        # Placeholder - default states, facilities override
        return {
            "ready": "Machine is ready for optimization",
            "not_ready": "Machine cannot proceed with optimization",
            "unknown": "Machine state assessment inconclusive",
        }

    def get_yaml_generation_guidance(self) -> str:
        """Get guidance for XOpt YAML configuration generation.

        :return: Domain-specific YAML generation guidance
        :rtype: str
        """
        # Placeholder - facilities override with facility-specific templates
        return textwrap.dedent(
            """
            YAML Generation Guidance (Placeholder):

            When generating XOpt YAML configurations:
            - Use valid XOpt schema structure
            - Define appropriate variables, objectives, and constraints
            - Select suitable generator and evaluator

            NOTE: Actual YAML templates and parameter definitions will be
            provided based on facility-specific requirements and historical examples.
            """
        ).strip()

    def get_strategy_selection_guidance(self) -> str:
        """Get guidance for exploration vs optimization strategy selection.

        :return: Decision criteria for strategy selection
        :rtype: str
        """
        # Placeholder - facilities override
        return textwrap.dedent(
            """
            Strategy Selection Guidance (Placeholder):

            - EXPLORATION: Use when exploring unknown parameter space
            - OPTIMIZATION: Use when refining known good regions

            NOTE: Actual strategy selection criteria will be defined
            based on operational requirements and machine state.
            """
        ).strip()

    def get_orchestrator_guide(self) -> OrchestratorGuide | None:
        """Create orchestrator guide for optimization capability.

        :return: Orchestrator guide with examples and instructions
        :rtype: Optional[OrchestratorGuide]
        """
        registry = get_registry()

        # Define structured examples
        basic_optimization_example = OrchestratorExample(
            step=PlannedStep(
                context_key="optimization_results",
                capability="optimization",
                task_objective="Optimize the injection efficiency using XOpt",
                expected_output=registry.context_types.OPTIMIZATION_RESULT,
                success_criteria="Optimization completed with improved efficiency metrics",
                inputs=[],
            ),
            scenario_description="Autonomous optimization of machine parameters",
            notes=f"SINGLE STEP ONLY. The optimization service internally handles all machine investigation, channel discovery, and analysis. Output stored under {registry.context_types.OPTIMIZATION_RESULT}.",
        )

        tuning_example = OrchestratorExample(
            step=PlannedStep(
                context_key="tuning_results",
                capability="optimization",
                task_objective="Tune magnet settings for improved beam quality",
                expected_output=registry.context_types.OPTIMIZATION_RESULT,
                success_criteria="Magnet settings optimized with measurable improvement",
                inputs=[],
            ),
            scenario_description="Parameter tuning for specific performance goals",
            notes="SINGLE STEP ONLY. Do NOT pre-plan channel_finding or python steps - the optimization service handles this internally.",
        )

        return OrchestratorGuide(
            instructions=textwrap.dedent(
                f"""
                **CRITICAL: Optimization is a SELF-CONTAINED, AUTONOMOUS service.**

                The optimization capability is NOT a simple executor - it is an intelligent
                agent service that INTERNALLY handles:
                - Machine state investigation (finding channels, reading values)
                - Strategy selection (exploration vs optimization)
                - Configuration generation (creating XOpt YAML)
                - Execution and result analysis

                **DO NOT orchestrate pre-requisite steps before optimization.**

                WRONG approach (do NOT do this):
                  1. channel_finding -> find injection channels
                  2. python -> analyze current state
                  3. optimization -> run optimization

                CORRECT approach (do this):
                  1. optimization -> "Optimize injection efficiency" (single step)

                The optimization service will autonomously investigate the machine,
                find relevant channels, assess readiness, and handle everything.

                **When to plan "optimization" steps:**
                - User requests autonomous tuning or optimization of machine parameters
                - Need to maximize or minimize a performance metric
                - User wants to explore parameter space or find optimal settings
                - Multi-parameter search or exploration is required

                **Step Structure:**
                - context_key: Unique identifier for output (e.g., "optimization_results")
                - task_objective: Clear, high-level description of the optimization goal
                - inputs: Empty or minimal - the service investigates on its own

                **Output: {registry.context_types.OPTIMIZATION_RESULT}**
                - Contains: Run artifact, strategy used, iteration count, recommendations
                - Available to downstream steps via context system
                - Includes generated XOpt configuration and analysis

                **Important Notes:**
                - Human approval is ALWAYS required before execution
                - The service includes its own result analysis and recommendations
                - Just describe WHAT to optimize, not HOW to investigate the machine
                """
            ),
            examples=[basic_optimization_example, tuning_example],
            priority=50,
        )

    def get_classifier_guide(self) -> TaskClassifierGuide | None:
        """Create classifier guide for optimization capability.

        :return: Classifier guide with examples
        :rtype: Optional[TaskClassifierGuide]
        """
        return TaskClassifierGuide(
            instructions="Determine if the user query requires autonomous machine optimization, parameter tuning, or multi-parameter search.",
            examples=[
                ClassifierExample(
                    query="Optimize the injection efficiency",
                    result=True,
                    reason="This requires autonomous optimization of machine parameters.",
                ),
                ClassifierExample(
                    query="What is the current beam current?",
                    result=False,
                    reason="This is a read operation, not optimization.",
                ),
                ClassifierExample(
                    query="Tune the magnets for better beam quality",
                    result=True,
                    reason="This requires parameter tuning/optimization.",
                ),
                ClassifierExample(
                    query="Set the magnet to 5 amps",
                    result=False,
                    reason="This is a direct write operation, not autonomous optimization.",
                ),
                ClassifierExample(
                    query="Find the optimal settings for maximum intensity",
                    result=True,
                    reason="This requires optimization to find optimal parameters.",
                ),
                ClassifierExample(
                    query="Plot the beam current over time",
                    result=False,
                    reason="This is a visualization request, not optimization.",
                ),
                ClassifierExample(
                    query="Run an optimization campaign on the injector",
                    result=True,
                    reason="This explicitly requests an optimization campaign.",
                ),
                ClassifierExample(
                    query="Maximize the charge at the end of the linac",
                    result=True,
                    reason="This requires optimization to maximize a metric.",
                ),
            ],
            actions_if_true=ClassifierActions(),
        )
