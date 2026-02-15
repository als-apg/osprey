"""Default orchestrator prompts."""

import textwrap
import warnings

from langchain_core.messages import BaseMessage

from osprey.base import BaseCapability, OrchestratorExample
from osprey.context import ContextManager
from osprey.prompts.base import FrameworkPromptBuilder
from osprey.state import ChatHistoryFormatter


class DefaultOrchestratorPromptBuilder(FrameworkPromptBuilder):
    """Default orchestrator prompt builder.

    Provides prompts for both plan-first and reactive orchestration modes.
    The builder decomposes prompts into shared components (role, step format,
    capabilities, context) and mode-specific strategies, allowing app
    developers to override individual pieces without affecting the other mode.

    Customization surface:

    ======================================  ================================
    I want to...                            Override...
    ======================================  ================================
    Change the agent's identity             ``get_role_definition()``
    Change step field definitions           ``get_step_format()``
    Change how capabilities are shown       ``build_capability_sections()``
    Change multi-step planning rules        ``get_planning_strategy()``
    Change single-step ReAct rules          ``get_reactive_strategy()``
    Fully customize plan-first prompt       ``get_planning_instructions()``
    Fully customize reactive prompt         ``get_reactive_instructions()``
    ======================================  ================================
    """

    PROMPT_TYPE = "orchestrator"

    def get_role_definition(self) -> str:
        """Get the generic role definition."""
        return "You are an expert execution planner for the assistant system."

    def get_task_definition(self) -> str:
        """Get the task definition."""
        return "TASK: Create a detailed execution plan that breaks down the user's request into specific, actionable steps."

    # ------------------------------------------------------------------
    # Shared components (used by both plan-first and reactive modes)
    # ------------------------------------------------------------------

    def get_step_format(self) -> str:
        """Get the PlannedStep field definitions shared by both orchestration modes.

        Override this to change the step schema description shown to the LLM.
        """
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
            """
        ).strip()

    # ------------------------------------------------------------------
    # Mode-specific strategy sections
    # ------------------------------------------------------------------

    def get_planning_strategy(self) -> str:
        """Get multi-step planning guidelines (plan-first mode only).

        Override this to change how the orchestrator plans multi-step executions.
        """
        return textwrap.dedent(
            """
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

    def get_reactive_strategy(self) -> str:
        """Get single-step ReAct constraints (reactive mode only).

        Override this to change how the orchestrator behaves in reactive mode.
        The orchestrator uses function calling (tools) to dispatch actions.
        """
        return textwrap.dedent(
            """
            You are operating in REACTIVE MODE with tool calling.

            **AVAILABLE TOOL TYPES:**
            1. **Lightweight tools** (read_context, list_available_context, get_context_summary,
               get_session_info, get_execution_status, list_system_capabilities):
               Use these to inspect accumulated context and agent state BEFORE deciding
               which capability to execute. They run inline — no graph cycle needed.

            2. **Capability tools** (the registered capabilities listed below):
               Call ONE capability tool to execute it. This exits the orchestrator
               and dispatches the capability for execution.

            **WORKFLOW:**
            - Use lightweight tools first to inspect context if needed
            - Then call exactly ONE capability tool to execute the next step
            - Call "respond" when all work is done
            - Call "clarify" when the request is unclear

            **RULES:**
            - Call at most ONE capability tool per response
            - Use exact capability names as tool names
            - Provide task_objective and context_key arguments for capability tools
            - Each context_key must be unique across the session
            """
        ).strip()

    def get_instructions(self) -> str:
        """Get combined step format and planning strategy.

        This satisfies the ``FrameworkPromptBuilder`` ABC contract.  App
        developers should override ``get_step_format()`` or
        ``get_planning_strategy()`` individually rather than this method.
        """
        return f"{self.get_step_format()}\n\n{self.get_planning_strategy()}"

    # ------------------------------------------------------------------
    # Plan-first composition
    # ------------------------------------------------------------------

    def get_planning_instructions(
        self,
        active_capabilities: list[BaseCapability] = None,
        context_manager: ContextManager = None,
        task_depends_on_chat_history: bool = False,
        task_depends_on_user_memory: bool = False,
        error_context: str | None = None,
        messages: list[BaseMessage] | None = None,
        **kwargs,
    ) -> str:
        """Get system instructions for plan-first orchestrator agent configuration.

        Args:
            active_capabilities: List of active capabilities
            context_manager: Current context manager with available data
            task_depends_on_chat_history: Whether task builds on previous conversation context
            task_depends_on_user_memory: Whether task depends on user memory information
            error_context: Formatted error context from previous execution failure (for replanning)
            messages: Chat history messages to include when task depends on conversation context

        Returns:
            Complete orchestrator prompt text
        """
        if not active_capabilities:
            active_capabilities = []

        # Build the main prompt sections
        prompt_sections = []

        # 1. Add base orchestrator prompt (role, task, step format, planning strategy)
        base_prompt_parts = [
            self.get_role_definition(),
            self.get_task_definition(),
            self.get_step_format(),
            self.get_planning_strategy(),
        ]
        base_prompt = "\n\n".join(base_prompt_parts)
        prompt_sections.append(base_prompt)

        # 2. Add chat history first (with visual separators) if task depends on conversation context
        if task_depends_on_chat_history and messages:
            chat_history_section = self.build_chat_history_section(messages)
            if chat_history_section:
                prompt_sections.append(chat_history_section)

        # 3. Add context reuse guidance if task builds on previous context
        context_guidance = self.build_context_reuse_guidance(
            task_depends_on_chat_history, task_depends_on_user_memory
        )
        if context_guidance:
            prompt_sections.append(context_guidance)

        # 4. Add error context for replanning if available
        if error_context:
            error_section = self.build_error_context_section(error_context)
            prompt_sections.append(error_section)

        # 5. Add context information if available
        if context_manager and context_manager.get_raw_data():
            context_section = self.build_context_section(context_manager)
            if context_section:
                prompt_sections.append(context_section)

        # 6. Add capability-specific prompts with examples
        capability_sections = self.build_capability_sections(active_capabilities)
        prompt_sections.extend(capability_sections)

        # Combine all sections
        final_prompt = "\n\n".join(prompt_sections)

        # Debug: Print prompt if enabled (same as base class)
        self.debug_print_prompt(final_prompt)

        return final_prompt

    def get_system_instructions(self, **kwargs) -> str:
        """Deprecated: use ``get_planning_instructions()`` instead.

        .. deprecated::
            ``get_system_instructions()`` on the orchestrator builder is
            deprecated and will be removed in a future release.  Use
            ``get_planning_instructions()`` for plan-first mode or
            ``get_reactive_instructions()`` for reactive mode.
        """
        warnings.warn(
            "DefaultOrchestratorPromptBuilder.get_system_instructions() is deprecated. "
            "Use get_planning_instructions() for plan-first mode or "
            "get_reactive_instructions() for reactive mode.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_planning_instructions(**kwargs)

    # ------------------------------------------------------------------
    # Reactive composition
    # ------------------------------------------------------------------

    def get_reactive_instructions(
        self,
        active_capabilities: list[BaseCapability] = None,
        context_manager: ContextManager = None,
        execution_history: str = "No steps executed yet",
        **kwargs,
    ) -> str:
        """Get system instructions for reactive (ReAct) orchestrator configuration.

        Composes a prompt from shared components (role, step format, capabilities,
        context) plus reactive-specific strategy and execution history.  Does NOT
        include plan-first-only sections (chat history, error context, context
        reuse guidance).

        Args:
            active_capabilities: List of active capabilities
            context_manager: Current context manager with available data
            execution_history: Formatted execution history from previous steps

        Returns:
            Complete reactive orchestrator prompt text
        """
        if not active_capabilities:
            active_capabilities = []

        prompt_sections = []

        # 1. Role definition
        prompt_sections.append(self.get_role_definition())

        # 2. Step format (shared)
        prompt_sections.append(self.get_step_format())

        # 3. Reactive strategy (mode-specific)
        prompt_sections.append(self.get_reactive_strategy())

        # 4. Context information if available
        if context_manager and context_manager.get_raw_data():
            context_section = self.build_context_section(context_manager)
            if context_section:
                prompt_sections.append(context_section)

        # 5. Capability-specific prompts with examples
        capability_sections = self.build_capability_sections(active_capabilities)
        prompt_sections.extend(capability_sections)

        # 6. Execution history
        prompt_sections.append(f"# EXECUTION HISTORY\n{execution_history}")

        # Combine all sections
        final_prompt = "\n\n".join(prompt_sections)

        # Debug: Print prompt if enabled
        self.debug_print_prompt(final_prompt, name="orchestrator_reactive")

        return final_prompt

    # ------------------------------------------------------------------
    # Reactive response context formatting
    # ------------------------------------------------------------------

    def format_reactive_response_context(self, react_messages: list[dict]) -> str:
        """Format the react_messages chain for the response-generation LLM.

        This is the customization point — app developers can override to
        format reactive context differently (e.g. include only observations,
        add domain-specific annotations, etc.).

        Args:
            react_messages: Accumulated assistant decisions and observations
                from the ReAct loop.

        Returns:
            Formatted text block to include in the response LLM's prompt.
        """
        if not react_messages:
            return "No reactive context available."

        lines: list[str] = []
        for msg in react_messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role == "assistant":
                lines.append(f"[Decision] {content}")
            elif role == "observation":
                lines.append(f"[Observation] {content}")
            else:
                lines.append(f"[{role}] {content}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Section builders (public API for overriding)
    # ------------------------------------------------------------------

    def build_context_reuse_guidance(
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

    def build_chat_history_section(self, messages: list[BaseMessage]) -> str | None:
        """Build the chat history section when task depends on conversation context.

        This provides the orchestrator with visibility into the actual conversation,
        enabling it to understand references like "the same time range" or
        "what did I just ask" that require knowledge of previous messages.

        Args:
            messages: List of conversation messages

        Returns:
            Formatted chat history section or None if no messages
        """
        if not messages:
            return None

        # Format messages using the standard formatter for consistency
        chat_formatted = ChatHistoryFormatter.format_for_llm(messages)

        # Use visual separators to clearly delineate chat history from other prompt sections
        return textwrap.dedent(
            f"""
            ════════════════════════════════════════════════════════════════════════════════
            **CONVERSATION HISTORY**
            The following is the conversation history that this task builds upon.
            Use this context to understand references to previous queries, results, or time ranges.
            ────────────────────────────────────────────────────────────────────────────────

{chat_formatted}

            ════════════════════════════════════════════════════════════════════════════════
            """
        ).strip()

    def build_error_context_section(self, error_context: str) -> str:
        """Build the error context section for replanning after execution failure."""
        return textwrap.dedent(
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

    def build_context_section(self, context_manager: ContextManager) -> str | None:
        """Build the context section of the prompt.

        Includes task_objective metadata when available to help the orchestrator
        understand what each context was created for, enabling intelligent reuse.
        """
        context_data = context_manager.get_raw_data()
        if not context_data:
            return None

        # Build context info with task_objective metadata when available
        # This helps the orchestrator understand what each context was created for
        formatted_lines = []

        for context_type, contexts in context_data.items():
            if context_type.startswith("_"):
                continue  # Skip internal keys like _execution_config

            for context_key, context_value in contexts.items():
                # Extract task_objective from metadata if available
                meta = context_value.get("_meta", {}) if isinstance(context_value, dict) else {}
                task_objective = meta.get("task_objective")

                # Format as {"TYPE": "key"} to match execution plan input format
                context_ref = f'{{"{context_type}": "{context_key}"}}'

                if task_objective:
                    # Include the task description for context reuse decisions
                    formatted_lines.append(f'  - {context_ref}: "{task_objective}"')
                else:
                    # Fallback to just showing the key
                    formatted_lines.append(f"  - {context_ref}")

        if not formatted_lines:
            return None

        formatted_context = "\n".join(formatted_lines)

        # Build the section without textwrap.dedent to avoid indentation issues
        return (
            "**AVAILABLE CONTEXT (from previous queries):**\n"
            "The following context data is already available from previous execution steps.\n"
            "Each entry shows the context type, key, and what it was created for:\n\n"
            f"{formatted_context}\n\n"
            "**CONTEXT REUSE PRINCIPLE:**\n"
            "- PREFER reusing existing context when the user's request involves the same data,\n"
            "  channels, or time ranges - even if not explicitly stated\n"
            "- The task descriptions above indicate what each context was created for;\n"
            "  use semantic similarity to decide on reuse\n"
            "- CREATE NEW retrieval step only when the user explicitly requests different\n"
            "  parameters (different channels, different time range, etc.)\n"
            '- Reference existing context in step inputs using: {"CONTEXT_TYPE": "context_key"}'
        )

    def build_capability_sections(self, active_capabilities: list[BaseCapability]) -> list[str]:
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
