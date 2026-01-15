"""Default orchestrator prompts."""

import textwrap

from osprey.base import BaseCapability, OrchestratorExample
from osprey.context import ContextManager
from osprey.prompts.base import FrameworkPromptBuilder


class DefaultOrchestratorPromptBuilder(FrameworkPromptBuilder):
    """Default orchestrator prompt builder."""

    PROMPT_TYPE = "orchestrator"

    def get_role_definition(self) -> str:
        """Get the generic role definition."""
        return "You are an expert execution planner for the assistant system."

    def get_task_definition(self) -> str:
        """Get the task definition."""
        return "TASK: Create a detailed execution plan that breaks down the user's request into specific, actionable steps."

    def get_instructions(self) -> str:
        """Get the generic planning instructions."""
        return textwrap.dedent(
            """
            Each step must follow the PlannedStep structure:
            - context_key: Unique identifier for this step's output (e.g., "data_sources", "historical_data")
            - capability: Type of execution node (determined based on available capabilities)
            - task_objective: Complete, self-sufficient description of what this step must accomplish
            - expected_output: Context type key (e.g., "HISTORICAL_DATA", "SYSTEM_STATUS")
            - success_criteria: Clear criteria for determining step success
            - inputs: List of input dictionaries mapping context types to context keys:
              [
                {"DATA_QUERY_RESULTS": "some_data_context"},
                {"ANALYSIS_RESULTS": "some_analysis_context"}
              ]
              **CRITICAL**: Include ALL required context sources! Complex operations often need multiple inputs.
            - parameters: Optional dict for step-specific configuration (e.g., {"precision_ms": 1000})

            Planning Guidelines:
            1. Dependencies between steps (ensure proper sequencing)
            2. Cost optimization (avoid unnecessary expensive operations)
            3. Clear success criteria for each step
            4. Proper input/output schema definitions
            5. Always reference available context using exact keys shown in context information
            6. **CRITICAL**: End plans with either "respond" or "clarify" step to ensure user gets feedback

            The execution plan should be an ExecutionPlan containing a list of PlannedStep json objects.

            Focus on being practical and efficient while ensuring robust execution.
            Be factual and realistic about what can be accomplished.
            Never plan for simulated or fictional data - only real system operations.
            """
        ).strip()

    def _get_dynamic_context(
        self, context_manager: ContextManager | None = None, **kwargs
    ) -> str | None:
        """Get dynamic context showing available context data."""
        if context_manager and context_manager.get_raw_data():
            return self._build_context_section(context_manager)
        return None

    def get_system_instructions(
        self,
        active_capabilities: list[BaseCapability] = None,
        context_manager: ContextManager = None,
        task_depends_on_chat_history: bool = False,
        task_depends_on_user_memory: bool = False,
        error_context: str | None = None,
        **kwargs,
    ) -> str:
        """
        Get system instructions for orchestrator agent configuration.

        Args:
            active_capabilities: List of active capabilities
            context_manager: Current context manager with available data
            task_depends_on_chat_history: Whether task builds on previous conversation context
            task_depends_on_user_memory: Whether task depends on user memory information
            error_context: Formatted error context from previous execution failure (for replanning)

        Returns:
            Complete orchestrator prompt text
        """
        if not active_capabilities:
            active_capabilities = []

        # Build the main prompt sections
        prompt_sections = []

        # 1. Add base orchestrator prompt (role, task, instructions)
        # Build directly without textwrap.dedent to avoid indentation issues
        base_prompt_parts = [
            self.get_role_definition(),
            self.get_task_definition(),
            self.get_instructions(),
        ]
        base_prompt = "\n\n".join(base_prompt_parts)
        prompt_sections.append(base_prompt)

        # 2. Add context reuse guidance if task builds on previous context
        context_guidance = self._build_context_reuse_guidance(
            task_depends_on_chat_history, task_depends_on_user_memory
        )
        if context_guidance:
            prompt_sections.append(context_guidance)

        # 3. Add error context for replanning if available
        if error_context:
            error_section = textwrap.dedent(
                f"""
                **REPLANNING CONTEXT:**

                The previous execution failed and needs replanning. Consider this error information when creating the new plan:

                {error_context}

                **Replanning Guidelines:**
                - Analyze the error context to understand why the previous approach failed
                - Consider alternative capabilities or different sequencing to avoid the same issue
                - If required context is missing, include clarification steps to gather needed information
                - Learn from the technical details and suggestions provided in the error context
                - Adapt the execution strategy based on the specific failure mode identified"""
            ).strip()

            prompt_sections.append(error_section)

        # 4. Add context information if available
        if context_manager and context_manager.get_raw_data():
            context_section = self._build_context_section(context_manager)
            if context_section:
                prompt_sections.append(context_section)

        # 5. Add capability-specific prompts with examples
        capability_sections = self._build_capability_sections(active_capabilities)
        prompt_sections.extend(capability_sections)

        # Combine all sections
        final_prompt = "\n\n".join(prompt_sections)

        # Debug: Print prompt if enabled (same as base class)
        self.debug_print_prompt(final_prompt)

        return final_prompt

    def _build_context_reuse_guidance(
        self, task_depends_on_chat_history: bool, task_depends_on_user_memory: bool
    ) -> str | None:
        """Build context reuse guidance section when task builds on previous context."""
        if not task_depends_on_chat_history and not task_depends_on_user_memory:
            return None

        guidance_parts = []

        if task_depends_on_chat_history:
            guidance_parts.append(
                "• **PRIORITIZE CONTEXT REUSE**: This task builds on previous conversation context. "
                "Look for existing context data that can be reused instead of recreating it."
            )

        if task_depends_on_user_memory:
            guidance_parts.append(
                "• **LEVERAGE USER MEMORY**: This task depends on user memory information. "
                "Check for existing memory context before planning new retrieval steps."
            )

        guidance_parts.append(
            "• **EFFICIENCY FIRST**: Avoid redundant context creation when suitable data already exists. "
            "Reference existing context keys in your step inputs."
        )

        guidance_text = "\n".join(guidance_parts)

        return textwrap.dedent(
            f"""
            **CONTEXT REUSE GUIDANCE:**
            {guidance_text}
            """
        ).strip()

    def _build_context_section(self, context_manager: ContextManager) -> str | None:
        """Build the context section of the prompt."""
        context_data = context_manager.get_raw_data()
        if not context_data:
            return None

        # Create a simple dictionary showing context_type -> [list of available keys]
        context_dict = {}
        for context_type, contexts in context_data.items():
            context_dict[context_type] = list(contexts.keys())

        # Format as a clean dictionary representation
        formatted_lines = ["["]
        for context_type, keys in context_dict.items():
            if len(keys) == 1:
                formatted_lines.append(f'    "{context_type}": "{keys[0]}",')
            else:
                keys_str = ", ".join(f'"{key}"' for key in keys)
                formatted_lines.append(f'    "{context_type}": [{keys_str}],')

        # Remove trailing comma from last line and close brace
        if len(formatted_lines) > 1:
            formatted_lines[-1] = formatted_lines[-1].rstrip(",")
        formatted_lines.append("]")

        formatted_context = "\n".join(formatted_lines)

        return textwrap.dedent(
            f"""
            **AVAILABLE CONTEXT (from previous queries):**
            {formatted_context}

            **CONTEXT REUSE PRINCIPLE:**
            - REUSE existing context only when user explicitly references previous results
              ("same time range", "that data", "the plot above", etc.)
            - CREATE NEW step if user specifies new values - compare against context keys above
            """
        ).strip()

    def _build_capability_sections(self, active_capabilities: list[BaseCapability]) -> list[str]:
        """Build capability-specific sections with examples."""
        sections = []

        # Group capabilities by order for proper sequencing
        capability_prompts = []
        for capability in active_capabilities:
            if capability.orchestrator_guide:
                capability_prompts.append((capability, capability.orchestrator_guide))

        # Sort by priority (lower priority = higher priority)
        sorted_prompts = sorted(capability_prompts, key=lambda p: p[1].priority)

        # Add header for capability sections
        if sorted_prompts:
            sections.append("# CAPABILITY PLANNING GUIDELINES")

        # Build each capability section with clear separators
        for _i, (capability, orchestrator_guide) in enumerate(sorted_prompts):
            if orchestrator_guide.instructions:  # Only include non-empty prompts
                # Add capability name header
                capability_name = capability.__class__.__name__.replace("Capability", "")
                section_text = f"## {capability_name}\n{orchestrator_guide.instructions}"

                # Add formatted examples if they exist
                if orchestrator_guide.examples:
                    examples_text = OrchestratorExample.join(
                        orchestrator_guide.examples, add_numbering=True
                    )
                    section_text += examples_text

                sections.append(section_text)

        return sections
