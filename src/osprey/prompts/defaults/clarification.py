"""Default clarification prompts."""

import textwrap

from osprey.base import OrchestratorExample, OrchestratorGuide, PlannedStep, TaskClassifierGuide
from osprey.prompts.base import FrameworkPromptBuilder


class DefaultClarificationPromptBuilder(FrameworkPromptBuilder):
    """Default clarification prompt builder."""

    PROMPT_TYPE = "clarification"

    def get_role_definition(self) -> str:
        """Get the generic role definition."""
        return "You are helping to clarify ambiguous user queries for the assistant system."

    def get_task_definition(self) -> str:
        """Get the task definition."""
        return "Your task is to generate specific, targeted questions that will help clarify what the user needs."

    def get_instructions(self) -> str:
        """Get the generic clarification instructions."""
        return textwrap.dedent("""
            GUIDELINES:
            1. Ask about missing technical details (which system, time range, specific parameters)
            2. Clarify vague terms (what type of "data", "status", "analysis" etc.)
            3. Ask about output preferences (format, detail level, specific metrics, etc.)
            4. Be specific and actionable - avoid generic questions
            5. Limit to 2-3 most important questions
            6. Make questions easy to answer

            Generate targeted questions that will help get the specific information needed to provide accurate assistance.
            """).strip()

    def get_orchestrator_guide(self) -> OrchestratorGuide | None:
        """Create generic orchestrator guide for clarification capability."""

        ambiguous_system_example = OrchestratorExample(
            step=PlannedStep(
                context_key="data_clarification",
                capability="clarify",
                task_objective="Ask user for clarification when request 'show me some data' is too vague",
                expected_output=None,  # No context output - questions sent directly to user
                success_criteria="Specific questions about data type, system, and time range",
                inputs=[]
            ),
            scenario_description="Vague data request needing system and parameter clarification",
        )

        return OrchestratorGuide(
            instructions="""
                Plan "clarify" when user queries lack specific details needed for execution.
                Use instead of respond when information is insufficient.
                Replaces technical execution steps until user provides clarification.
                """,
            examples=[ambiguous_system_example],
            priority=99  # Should come near the end, but before respond
        )

    def get_classifier_guide(self) -> TaskClassifierGuide | None:
        """Clarify has no classifier guide - it's orchestrator-driven."""
        return None  # Always available, not detected from user intent



    def build_clarification_query(self, chat_history: str, task_objective: str) -> str:
        """Build clarification query for generating questions based on conversation context.

        Used by the clarification infrastructure to generate specific questions
        when information is missing from user requests.

        Args:
            chat_history: Formatted conversation history
            task_objective: Extracted task objective from user request

        Returns:
            Complete query for question generation with automatic debug printing
        """
        prompt = textwrap.dedent(f"""
            CONVERSATION HISTORY:
            {chat_history}

            EXTRACTED TASK OBJECTIVE: {task_objective}

            Based on the full conversation history and the extracted task, generate specific clarifying questions.
            Consider:
            - What has already been discussed in the conversation
            - What information is still missing to execute the task
            - Avoid asking for information already provided earlier in the conversation
            - Focus on the most critical missing information that would enable accurate assistance
            """).strip()

        # Automatic debug printing for framework helper prompts
        self.debug_print_prompt(prompt, "clarification_query")

        return prompt
