"""Default error analysis prompts."""

import textwrap

from osprey.prompts.base import FrameworkPromptBuilder


class DefaultErrorAnalysisPromptBuilder(FrameworkPromptBuilder):
    """Default error analysis prompt builder.

    **Customization Points:**

    +---------------------------------+----------------------------------------------+
    | I want to...                    | Override...                                  |
    +=================================+==============================================+
    | Change the agent identity       | ``get_role()``                    |
    +---------------------------------+----------------------------------------------+
    | Change analysis instructions    | ``get_instructions()``                       |
    +---------------------------------+----------------------------------------------+
    | Change dynamic context assembly | ``build_dynamic_context(...)``               |
    +---------------------------------+----------------------------------------------+
    """

    PROMPT_TYPE = "error_analysis"

    def get_role(self) -> str:
        """Get the generic role definition."""
        return "You are providing error analysis for the assistant system."

    def get_task(self) -> str | None:
        """Task definition is embedded in instructions."""
        return None

    def get_instructions(self) -> str:
        """Get the error analysis instructions."""
        return textwrap.dedent(
            """
            A structured error report has already been generated with the following information:
            - Error type and timestamp
            - Task description and failed operation
            - Error message and technical details
            - Execution statistics and summary
            - Capability-specific recovery options

            Your role is to provide a brief explanation that adds value beyond the structured data:

            Requirements:
            - Write 2-3 sentences explaining what likely went wrong
            - Focus on the "why" rather than repeating the "what"
            - Do NOT repeat the error message, recovery options, or execution details
            - Be specific to system operations when relevant
            - Consider the system capabilities context when suggesting alternatives
            - Keep it under 100 words
            - Use a professional, technical tone
            """
        ).strip()

    def build_dynamic_context(
        self, capabilities_overview: str = "", error_context=None, **kwargs
    ) -> str:
        """Build dynamic context with capabilities and error information."""
        sections = []

        # System capabilities
        if capabilities_overview:
            sections.append(f"SYSTEM CAPABILITIES:\n{capabilities_overview}")

        # Error context
        if error_context:
            error_info = textwrap.dedent(
                f"""
                ERROR CONTEXT:
                - Current task: {getattr(error_context, "current_task", "Unknown")}
                - Error type: {getattr(error_context, "error_type", {}).value if hasattr(getattr(error_context, "error_type", {}), "value") else "Unknown"}
                - Capability: {getattr(error_context, "capability_name", None) or "Unknown"}
                - Error message: {getattr(error_context, "error_message", "Unknown")}
                """
            ).strip()
            sections.append(error_info)

        return "\n\n".join(sections)
