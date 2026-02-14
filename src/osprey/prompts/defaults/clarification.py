"""Default clarification prompts."""

import textwrap
from typing import TYPE_CHECKING

from osprey.base import OrchestratorExample, OrchestratorGuide, PlannedStep, TaskClassifierGuide
from osprey.prompts.base import FrameworkPromptBuilder

if TYPE_CHECKING:
    from osprey.state import AgentState


class DefaultClarificationPromptBuilder(FrameworkPromptBuilder):
    """Default clarification prompt builder.

    **Customization Points:**

    +---------------------------------+----------------------------------------------+
    | I want to...                    | Override...                                  |
    +=================================+==============================================+
    | Change the agent identity       | ``get_role()``                    |
    +---------------------------------+----------------------------------------------+
    | Change the task statement       | ``get_task()``                    |
    +---------------------------------+----------------------------------------------+
    | Change guidelines               | ``get_instructions()``                       |
    +---------------------------------+----------------------------------------------+
    | Change question generation      | ``build_clarification_query(...)``            |
    +---------------------------------+----------------------------------------------+
    """

    PROMPT_TYPE = "clarification"

    def get_role(self) -> str:
        """Get the generic role definition."""
        return "You are helping to clarify ambiguous user queries for the assistant system."

    def get_task(self) -> str:
        """Get the task definition."""
        return "Your task is to generate specific, targeted questions that will help clarify what the user needs."

    def get_instructions(self) -> str:
        """Get the generic clarification instructions."""
        return textwrap.dedent(
            """
            GUIDELINES:
            1. Ask about missing technical details (which system, time range, specific parameters)
            2. Clarify vague terms (what type of "data", "status", "analysis" etc.)
            3. Ask about output preferences (format, detail level, specific metrics, etc.)
            4. Be specific and actionable - avoid generic questions
            5. Limit to 2-3 most important questions
            6. Make questions easy to answer

            Generate targeted questions that will help get the specific information needed to provide accurate assistance.
            """
        ).strip()

    def get_orchestrator_guide(self) -> OrchestratorGuide | None:
        """Create generic orchestrator guide for clarification capability."""

        ambiguous_system_example = OrchestratorExample(
            step=PlannedStep(
                context_key="data_clarification",
                capability="clarify",
                task_objective="Ask user for clarification when request 'show me some data' is too vague",
                expected_output=None,  # No context output - questions sent directly to user
                success_criteria="Specific questions about data type, system, and time range",
                inputs=[],
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
            priority=99,  # Should come near the end, but before respond
        )

    def get_classifier_guide(self) -> TaskClassifierGuide | None:
        """Clarify has no classifier guide - it's orchestrator-driven."""
        return None  # Always available, not detected from user intent

    def build_prompt(self, state: "AgentState", task_objective: str) -> str:
        """Compose complete clarification prompt with runtime context.

        This override provides clarification-specific prompt composition where
        the orchestrator's specific instruction is the primary directive. The
        clarification_query already contains complete instructions with examples,
        so no additional generic guidelines are needed.

        Args:
            state: Current agent state containing messages and execution context
            task_objective: The orchestrator's clarification instruction
                          (e.g., "Ask user to specify location...")

        Returns:
            Complete prompt ready for LLM with automatic debug logging

        The composition structure:
        1. Role definition (who the AI is)
        2. Complete clarification query with:
           - Conversation history
           - User's original query
           - Orchestrator's specific instruction
           - Clear task description
           - Concrete example (good vs bad questions)
           - Context considerations
        3. Available context data (if present)
        """
        # Import here to avoid circular dependency
        from osprey.context.context_manager import ContextManager
        from osprey.state import ChatHistoryFormatter, StateManager

        # Extract runtime data from state
        messages = state.get("messages", [])
        chat_history_str = ChatHistoryFormatter.format_for_llm(messages)
        user_query = StateManager.get_user_query(state) or ""

        # Get relevant context using ContextManager
        context_manager = ContextManager(state)
        current_step = StateManager.get_current_step(state)
        relevant_context = context_manager.get_summaries(current_step)

        # Build the specific clarification query with all runtime context
        clarification_query = self.build_clarification_query(
            chat_history_str, user_query, task_objective
        )

        # Include available context if present
        context_info = ""
        if relevant_context:
            context_items = []
            for context_summary in relevant_context:
                context_type = context_summary.get("type", "Context")
                context_items.append(f"- {context_type}: {context_summary}")
            context_info = "\n\nAvailable context data:\n" + "\n".join(context_items)

        # Compose final prompt with role and the complete clarification query
        # The clarification_query already contains all necessary instructions and examples
        # No need for additional generic guidelines that would create redundancy
        role = self.get_role()

        final_prompt = f"""{role}

{clarification_query}{context_info}"""

        # Automatic debug logging (use standard name, not "full")
        self.debug_print_prompt(final_prompt)

        return final_prompt

    def build_clarification_query(
        self, chat_history: str, user_query: str, task_objective: str
    ) -> str:
        """Build clarification query for generating questions based on conversation context.

        Used by the clarification infrastructure to generate specific questions
        when information is missing from user requests.

        Args:
            chat_history: Formatted conversation history
            user_query: The user's original query/request
            task_objective: The orchestrator's instruction about what to clarify
                           (e.g., "Ask the user to specify which location...")

        Returns:
            Complete query for question generation with automatic debug printing
        """
        prompt = textwrap.dedent(
            f"""
            CONVERSATION HISTORY:
            {chat_history}

            USER'S ORIGINAL QUERY:
            "{user_query}"

            ORCHESTRATOR'S CLARIFICATION INSTRUCTION:
            {task_objective}

            Your task is to generate specific clarifying questions that address the missing information.

            IMPORTANT:
            - The user asked: "{user_query}"
            - The orchestrator identified what's missing: "{task_objective}"
            - Generate 1-3 specific, targeted questions that will get the missing information (simplicity is appreciated)
            - Make questions conversational and easy to answer
            - Be specific to what's missing - don't ask generic "what do you need?" questions
            - Reference the user's original intent when appropriate

            Example:
            - User query: "what's the weather?"
            - Missing info: "location not specified"
            - Good question: "Which location would you like weather information for?"
            - Bad question: "What do you need help with?"

            Consider:
            - What has already been discussed in the conversation history
            - Avoid asking for information already provided earlier
            - Focus on the specific missing information from the orchestrator's instruction
            """
        ).strip()

        # Automatic debug printing for framework helper prompts
        self.debug_print_prompt(prompt, "clarification_query")

        return prompt
