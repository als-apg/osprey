"""
Python Capability Prompt Builder

Default prompts for Python code generation and execution capability.
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


class DefaultPythonPromptBuilder(FrameworkPromptBuilder):
    """Default Python capability prompt builder.

    **Customization Points:**

    +---------------------------------+----------------------------------------------+
    | I want to...                    | Override...                                  |
    +=================================+==============================================+
    | Change the agent identity       | ``get_role()``                    |
    +---------------------------------+----------------------------------------------+
    | Change code gen instructions    | ``get_instructions()``                       |
    +---------------------------------+----------------------------------------------+
    | Change orchestrator guidance    | ``get_orchestrator_guide()``                 |
    +---------------------------------+----------------------------------------------+
    | Change classifier guidance      | ``get_classifier_guide()``                   |
    +---------------------------------+----------------------------------------------+
    """

    PROMPT_TYPE = "python"

    def get_role(self) -> str:
        """Get the role definition for Python code generation."""
        return "You are an expert Python developer generating high-quality, executable code."

    def get_task(self) -> str:
        """Get the task definition for Python code generation."""
        return None  # Task is provided via capability_prompts

    def get_instructions(self) -> str:
        """Get the instructions for Python code generation.

        These instructions are domain-agnostic and apply to all Python code generation.
        Domain-specific guidance (like control system operations) should be added
        via custom prompt builders in application templates.
        """
        return textwrap.dedent(
            """
            === CODE GENERATION INSTRUCTIONS ===
            1. Generate complete, executable Python code
            2. Include all necessary imports at the top
            3. Use professional coding standards and clear variable names
            4. Add brief comments explaining complex logic
            5. STAY FOCUSED: Implement exactly what's requested - avoid over-engineering simple tasks
            6. Use provided context data when available (accessible via 'context' object)
            7. IMPORTANT: Store computed results in a dictionary variable named 'results' for automatic saving
            8. Generate ONLY the Python code, without markdown code blocks or additional explanation
            """
        ).strip()

    def get_orchestrator_guide(self) -> OrchestratorGuide | None:
        """Create orchestrator guide for Python capability."""
        registry = get_registry()

        # Define structured examples
        calculation_example = OrchestratorExample(
            step=PlannedStep(
                context_key="calculation_results",
                capability="python",
                task_objective="Calculate the area of a circle with radius 5 and display the result",
                expected_output=registry.context_types.PYTHON_RESULTS,
                success_criteria="Mathematical calculation completed with printed result",
                inputs=[],
            ),
            scenario_description="Simple mathematical calculations using Python",
            notes=f"Output stored under {registry.context_types.PYTHON_RESULTS} context type. Generated code, execution output, and any errors are captured.",
        )

        data_processing_example = OrchestratorExample(
            step=PlannedStep(
                context_key="processing_results",
                capability="python",
                task_objective="Process a list of numbers [1, 2, 3, 4, 5] and calculate mean, median, and standard deviation",
                expected_output=registry.context_types.PYTHON_RESULTS,
                success_criteria="Statistical analysis completed with all metrics calculated",
                inputs=[],
            ),
            scenario_description="Basic data processing and statistical calculations",
            notes=f"Output stored under {registry.context_types.PYTHON_RESULTS} context type. Demonstrates data manipulation and statistical functions.",
        )

        utility_example = OrchestratorExample(
            step=PlannedStep(
                context_key="utility_results",
                capability="python",
                task_objective="Generate 10 random numbers between 1 and 100 and find the maximum value",
                expected_output=registry.context_types.PYTHON_RESULTS,
                success_criteria="Random number generation completed with maximum value identified",
                inputs=[],
            ),
            scenario_description="Utility functions like random number generation and basic algorithms",
            notes=f"Output stored under {registry.context_types.PYTHON_RESULTS} context type. Shows how to handle randomization and list operations.",
        )

        return OrchestratorGuide(
            instructions=textwrap.dedent(
                f"""
                **When to plan "python" steps:**
                - User requests simple calculations or mathematical operations
                - Need to perform basic data processing or statistical analysis
                - User wants to execute Python code for computational tasks
                - Simple algorithms or utility functions are needed
                - Processing of lists, numbers, or basic data structures

                **Step Structure:**
                - context_key: Unique identifier for output (e.g., "calculation_results", "processing_output")
                - task_objective: Clear description of the computational task to perform
                - inputs: Optional, can work standalone for most simple tasks

                **Output: {registry.context_types.PYTHON_RESULTS}**
                - Contains: Generated Python code, execution output, and any errors
                - Available to downstream steps via context system
                - Includes code explanation and execution status

                **Use for:**
                - Mathematical calculations (area, volume, statistics)
                - Simple data processing (sorting, filtering, aggregations)
                - Basic computational tasks (random numbers, algorithms)
                - Code generation and mock execution
                - Utility functions and helper calculations

                **Dependencies and sequencing:**
                1. Often used as a standalone capability for simple tasks
                2. Can consume data from other capabilities if needed
                3. Results can feed into visualization or analysis steps
                4. Lightweight alternative to complex data analysis workflows

                ALWAYS prefer this capability for simple computational tasks that don't require
                sophisticated data analysis or complex external dependencies.
                """
            ),
            examples=[calculation_example, data_processing_example, utility_example],
            priority=40,
        )

    def get_classifier_guide(self) -> TaskClassifierGuide | None:
        """Create classifier guide for Python capability."""
        return TaskClassifierGuide(
            instructions="Determine if the user query requires Python code execution for simple computational tasks, mathematical calculations, or basic data processing.",
            examples=[
                ClassifierExample(
                    query="Calculate the area of a circle with radius 5",
                    result=True,
                    reason="This requires mathematical calculation using Python.",
                ),
                ClassifierExample(
                    query="What is your name?",
                    result=False,
                    reason="This is a conversational question, not a computational task.",
                ),
                ClassifierExample(
                    query="Process this list of numbers and find the average",
                    result=True,
                    reason="This requires data processing and statistical calculation.",
                ),
                ClassifierExample(
                    query="Show me the current time",
                    result=False,
                    reason="This is a simple information request, not requiring custom code generation.",
                ),
                ClassifierExample(
                    query="Generate a random number between 1 and 100",
                    result=True,
                    reason="This requires Python code to generate random numbers.",
                ),
                ClassifierExample(
                    query="Sort these numbers in ascending order: 5, 2, 8, 1, 9",
                    result=True,
                    reason="This requires Python code for data manipulation and sorting.",
                ),
                ClassifierExample(
                    query="What tools do you have available?",
                    result=False,
                    reason="This is a question about AI capabilities, not a computational task.",
                ),
                ClassifierExample(
                    query="Calculate the fibonacci sequence up to 10 numbers",
                    result=True,
                    reason="This requires Python code to implement an algorithm.",
                ),
                ClassifierExample(
                    query="How does machine learning work?",
                    result=False,
                    reason="This is an educational question, not a request for code execution.",
                ),
                ClassifierExample(
                    query="Convert temperature from 32°F to Celsius",
                    result=True,
                    reason="This requires mathematical calculation and unit conversion.",
                ),
            ],
            actions_if_true=ClassifierActions(),
        )
