"""Reactive Orchestrator Node - ReAct Agent Pattern.

Implements step-by-step reactive orchestration using the ReAct (Reasoning + Acting)
pattern. Instead of generating a full execution plan upfront, this orchestrator
decides the next action one step at a time, observing results between steps.

The reactive orchestrator produces individual ``PlannedStep`` objects that execute
through the same ``@capability_node`` infrastructure as the plan-first orchestrator.
Capabilities are unaware of which orchestration mode generated their step.

Key components:
    - ``ReactiveOrchestratorNode``: Infrastructure node implementing the ReAct loop
    - ``validate_single_step``: Reused from orchestration_node for step validation
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from langgraph.types import interrupt

from osprey.approval import (
    clear_approval_state,
    create_approval_type,
    create_step_approval_interrupt,
    get_approval_resume_data,
)
from osprey.base.decorators import infrastructure_node
from osprey.base.errors import (
    ErrorClassification,
    ErrorSeverity,
    InvalidContextKeyError,
    ReclassificationRequiredError,
)
from osprey.base.nodes import BaseInfrastructureNode
from osprey.base.planning import ExecutionPlan, PlannedStep
from osprey.context.context_manager import ContextManager
from osprey.infrastructure.orchestration_node import (
    _is_planning_mode_enabled,
    validate_single_step,
)
from osprey.models import get_chat_completion, set_api_call_context
from osprey.models.messages import ChatCompletionRequest, ChatMessage
from osprey.prompts.loader import get_framework_prompts
from osprey.registry import get_registry
from osprey.state import AgentState
from osprey.state.state import create_status_update
from osprey.utils.config import get_interface_context, get_model_config

_MAX_VALIDATION_RETRIES = 2  # Up to 3 total attempts (1 initial + 2 retries)
_MAX_LIGHTWEIGHT_CALLS = 5  # Max inline tool calls per orchestrator invocation
_MAX_CONSECUTIVE_REJECTIONS = 3  # Skip approval gate after this many consecutive rejections


# =============================================================================
# REACTIVE ORCHESTRATOR NODE
# =============================================================================


@infrastructure_node(quiet=True)
class ReactiveOrchestratorNode(BaseInfrastructureNode):
    """Reactive step-by-step orchestration using the ReAct pattern.

    This node generates one ``PlannedStep`` at a time instead of a full
    execution plan. After each capability executes, the router sends
    control back here so the LLM can observe the result and decide
    the next action.

    The node produces the same ``PlannedStep`` objects consumed by
    ``@capability_node``, so capabilities are unaware of which
    orchestration mode generated their step.
    """

    name = "reactive_orchestrator"
    description = "Reactive orchestration"

    @staticmethod
    def classify_error(exc: Exception, context: dict) -> ErrorClassification:
        """Error classification matching OrchestrationNode patterns."""
        if hasattr(exc, "__class__") and "timeout" in exc.__class__.__name__.lower():
            return ErrorClassification(
                severity=ErrorSeverity.RETRIABLE,
                user_message="LLM timeout during reactive orchestration, retrying...",
                metadata={"technical_details": str(exc)},
            )

        if isinstance(exc, (ConnectionError, TimeoutError)):
            return ErrorClassification(
                severity=ErrorSeverity.RETRIABLE,
                user_message="Network timeout during reactive orchestration, retrying...",
                metadata={"technical_details": str(exc)},
            )

        # Rate limit errors are retriable (with backoff handled by litellm)
        exc_str = str(exc).lower()
        exc_type = type(exc).__name__.lower()
        if "ratelimit" in exc_type or "rate_limit" in exc_type or "rate limit" in exc_str:
            return ErrorClassification(
                severity=ErrorSeverity.RETRIABLE,
                user_message="LLM rate limit hit during reactive orchestration, retrying...",
                metadata={"technical_details": str(exc)},
            )
        if isinstance(exc, ValueError) and any(
            indicator in exc_str
            for indicator in ["validation error", "field required", "pydantic", "json", "parsing"]
        ):
            return ErrorClassification(
                severity=ErrorSeverity.RETRIABLE,
                user_message="LLM failed to generate valid reactive decision, retrying...",
                metadata={"technical_details": str(exc)},
            )

        if isinstance(exc, ReclassificationRequiredError):
            return ErrorClassification(
                severity=ErrorSeverity.RECLASSIFICATION,
                user_message=f"Task needs reclassification: {str(exc)}",
                metadata={"technical_details": str(exc)},
            )

        if isinstance(exc, InvalidContextKeyError):
            return ErrorClassification(
                severity=ErrorSeverity.RETRIABLE,
                user_message="Invalid context key in reactive step, retrying...",
                metadata={"technical_details": str(exc)},
            )

        if isinstance(exc, (ValueError, TypeError)):
            return ErrorClassification(
                severity=ErrorSeverity.CRITICAL,
                user_message="Reactive orchestration configuration error",
                metadata={"technical_details": str(exc)},
            )

        return ErrorClassification(
            severity=ErrorSeverity.CRITICAL,
            user_message=f"Unknown reactive orchestration error: {str(exc)}",
            metadata={
                "technical_details": f"Error type: {type(exc).__name__}, Details: {str(exc)}"
            },
        )

    @staticmethod
    def get_retry_policy() -> dict[str, Any]:
        """Retry policy for LLM-based reactive orchestration."""
        return {
            "max_attempts": 4,
            "delay_seconds": 2.0,
            "backoff_factor": 2.0,
        }

    async def execute(self) -> dict[str, Any]:
        """Execute one iteration of the ReAct loop using tool calling.

        Two-tier tool architecture:

        - **Lightweight tools** (read_context, list_available_context, etc.):
          Execute inline within the LLM loop. Results are fed back as tool
          response messages for the next LLM turn.
        - **Capability tools** (channel_finding, respond, etc.):
          Trigger exit from the loop — the orchestrator builds a PlannedStep
          and returns control to the router for capability execution.

        The LLM may also return plain text (no tool call), which is parsed
        as an ExecutionPlan JSON for backward compatibility with models
        that don't support function calling.
        """
        from osprey.infrastructure.reactive_tools import (
            build_capability_tools,
            build_lightweight_tools,
            execute_lightweight_tool,
        )

        state = self._state
        logger = self.get_logger()

        # =================================================================
        # STEP 0: CHECK FOR APPROVAL RESUME
        # =================================================================
        self._needs_approval_cleanup = False
        _step_approval_type = create_approval_type("reactive_orchestrator", "step")

        has_approval_resume, approved_payload = get_approval_resume_data(state, _step_approval_type)

        if has_approval_resume and approved_payload:
            # APPROVED — return stored plan + react state without LLM call
            logger.info("Reactive step approved — returning stored plan")
            return {
                "planning_execution_plan": approved_payload["execution_plan"],
                "planning_current_step_index": 0,
                "react_messages": approved_payload["react_messages"],
                "react_step_count": approved_payload["react_step_count"],
                "react_rejection_count": 0,
                "control_has_error": False,
                "control_error_info": None,
                "control_last_error": None,
                "control_retry_count": 0,
                "control_current_step_retry_count": 0,
                **clear_approval_state(),
            }
        elif has_approval_resume:
            # REJECTED — flag for cleanup; fall through to LLM call
            logger.info("Reactive step rejected — re-evaluating with LLM")
            self._needs_approval_cleanup = True

        current_task = self.get_current_task()
        if not current_task:
            raise ValueError("No current task available for reactive orchestration")

        # Resolve active capabilities from state (set by classifier)
        active_capability_names = state.get("planning_active_capabilities", [])
        registry = get_registry()
        active_capabilities = [
            cap
            for name in active_capability_names
            if (cap := registry.get_capability(name)) is not None
        ]
        if not active_capabilities:
            raise ValueError("No valid capability instances for reactive orchestration")

        react_messages = list(state.get("react_messages", []))
        react_step_count = state.get("react_step_count", 0)

        # On rejection, inject a rejection observation so the LLM knows
        if self._needs_approval_cleanup:
            react_messages.append(
                {
                    "role": "observation",
                    "content": "User REJECTED the proposed step. Choose a different approach or respond.",
                }
            )

        # Build the system prompt using shared prompt infrastructure
        system_prompt = _build_system_prompt(state, logger, active_capabilities)

        # Build the conversation for the LLM
        observation = _format_observation(state)

        # Persist observation from previous step into react_messages
        if observation:
            react_messages.append({"role": "observation", "content": observation})

        logger.status(f"Deciding next action (step {react_step_count + 1})...")

        # Call LLM - fall back to orchestrator model if no reactive-specific config
        model_config = get_model_config("reactive_orchestrator")
        if not model_config:
            model_config = get_model_config("orchestrator")

        # Build tools: lightweight (inline) + capability (exit to router)
        lightweight_tool_defs, lightweight_tool_map = build_lightweight_tools(state)
        capability_tool_defs = build_capability_tools(active_capabilities)
        all_tool_defs = lightweight_tool_defs + capability_tool_defs

        capability_names = {getattr(c, "name", "") for c in active_capabilities}

        chat_request = _build_chat_request(system_prompt, current_task, react_messages)

        # --- Tool-calling loop ---
        # The LLM may call lightweight tools multiple times before selecting
        # a capability. Safety limit prevents infinite loops.
        lightweight_count = 0

        for _loop_iter in range(1 + _MAX_LIGHTWEIGHT_CALLS):
            logger.debug(
                "LLM prompt built",
                llm_prompt=chat_request.to_single_string(),
                stream=False,
            )

            set_api_call_context(
                function="execute",
                module="reactive_orchestrator_node",
                class_name="ReactiveOrchestratorNode",
            )

            plan_start_time = time.time()

            response = await asyncio.to_thread(
                get_chat_completion,
                chat_request=chat_request,
                model_config=model_config,
                tools=all_tool_defs,
            )

            execution_time = time.time() - plan_start_time

            # --- Handle tool call response ---
            if isinstance(response, list):
                logger.info(
                    f"Parsing response ({execution_time:.1f}s)",
                    llm_response=json.dumps(response, indent=2),
                    stream=False,
                )

                # Separate lightweight and capability tool calls
                lw_calls = []
                cap_call = None

                for tc in response:
                    fn_name = tc.get("function", {}).get("name", "")
                    if fn_name in lightweight_tool_map:
                        lw_calls.append(tc)
                    elif fn_name in capability_names:
                        if cap_call is None:
                            cap_call = tc
                        else:
                            logger.warning(
                                f"Multiple capability tool calls; ignoring {fn_name} "
                                f"(using {cap_call['function']['name']})"
                            )
                    else:
                        # Unknown tool — feed error back as tool result
                        chat_request.messages.append(
                            ChatMessage(
                                role="assistant",
                                tool_calls=[tc],
                            )
                        )
                        chat_request.messages.append(
                            ChatMessage(
                                role="tool",
                                tool_call_id=tc.get("id", ""),
                                name=fn_name,
                                content=f"Error: Unknown tool '{fn_name}'. "
                                f"Available capabilities: {', '.join(sorted(capability_names))}",
                            )
                        )
                        lightweight_count += 1

                # Execute lightweight tools inline
                if lw_calls:
                    # Append the assistant message with all tool calls
                    chat_request.messages.append(ChatMessage(role="assistant", tool_calls=lw_calls))
                    for tc in lw_calls:
                        fn_name = tc.get("function", {}).get("name", "")
                        fn_args = tc.get("function", {}).get("arguments", "{}")
                        logger.info(f"Executing tool: {fn_name}")
                        result = execute_lightweight_tool(fn_name, fn_args, lightweight_tool_map)
                        chat_request.messages.append(
                            ChatMessage(
                                role="tool",
                                tool_call_id=tc.get("id", ""),
                                name=fn_name,
                                content=result,
                            )
                        )
                        lightweight_count += 1

                # If a capability tool call was found, dispatch it
                if cap_call is not None:
                    return await self._dispatch_capability_tool_call(
                        cap_call, state, react_messages, react_step_count, current_task, logger
                    )

                # Only lightweight tools were called — continue the loop
                if lightweight_count > _MAX_LIGHTWEIGHT_CALLS:
                    raise ValueError(
                        f"Exceeded maximum lightweight tool calls ({_MAX_LIGHTWEIGHT_CALLS}). "
                        "The orchestrator must select a capability to execute."
                    )
                continue

            # --- Handle text response (fallback for non-tool-calling models) ---
            logger.info(
                f"Parsing response ({execution_time:.1f}s)",
                llm_response=str(response),
                stream=False,
            )
            return await self._handle_text_response(
                response, state, react_messages, react_step_count, current_task, logger
            )

        # If we exhaust the loop without dispatching, raise
        raise ValueError(
            f"Exceeded maximum lightweight tool calls ({_MAX_LIGHTWEIGHT_CALLS}). "
            "The orchestrator must select a capability to execute."
        )

    async def _dispatch_capability_tool_call(
        self,
        tool_call: dict,
        state: AgentState,
        react_messages: list[dict],
        react_step_count: int,
        current_task: str,
        logger,
    ) -> dict[str, Any]:
        """Build a PlannedStep from a capability tool call and return state updates."""
        fn = tool_call.get("function", {})
        capability = fn.get("name", "")
        args_str = fn.get("arguments", "{}")

        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
        except json.JSONDecodeError:
            args = {}

        task_objective = args.get("task_objective", current_task)

        # Intercept respond — generate response directly with full reactive context
        if capability == "respond":
            return await self._generate_direct_response(
                state,
                react_messages,
                react_step_count,
                current_task,
                task_objective,
                logger,
            )
        context_key = args.get("context_key", capability)
        expected_output = args.get("expected_output")
        success_criteria = args.get("success_criteria", "Step completed successfully")

        # Validate and resolve inputs
        inputs = _resolve_inputs(capability, state)

        # Pre-dispatch gate: check for unresolved requirements
        active_caps = _get_active_capabilities(state)
        missing = _check_unresolved_requirements(capability, inputs, active_caps)
        if missing:
            return _build_missing_requirements_response(
                capability, missing, react_messages, react_step_count, logger, state
            )

        step_with_inputs = PlannedStep(
            context_key=context_key,
            capability=capability,
            task_objective=task_objective,
            expected_output=expected_output,
            success_criteria=success_criteria,
            inputs=inputs,
        )
        validate_single_step(step_with_inputs, state, logger)

        return self._build_return_state(
            step_with_inputs, capability, react_messages, react_step_count, task_objective, logger
        )

    async def _handle_text_response(
        self,
        response: str,
        state: AgentState,
        react_messages: list[dict],
        react_step_count: int,
        current_task: str,
        logger,
    ) -> dict[str, Any]:
        """Parse text response as ExecutionPlan JSON (backward compatibility)."""
        try:
            execution_plan = json.loads(response) if isinstance(response, str) else response
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(
                f"LLM returned text that is not valid ExecutionPlan JSON: {str(response)[:200]}"
            ) from exc

        steps = execution_plan.get("steps", [])
        if not steps:
            raise ValueError("LLM returned an ExecutionPlan with no steps")

        if len(steps) > 1:
            logger.warning(
                f"Reactive mode expected 1 step but got {len(steps)}; using first step only"
            )

        step = steps[0]
        capability = step.get("capability", "")

        # Intercept respond — generate response directly with full reactive context
        if capability == "respond":
            return await self._generate_direct_response(
                state,
                react_messages,
                react_step_count,
                current_task,
                step.get("task_objective", current_task),
                logger,
            )
        inputs = _resolve_inputs(capability, state)

        # Pre-dispatch gate: check for unresolved requirements
        active_caps = _get_active_capabilities(state)
        missing = _check_unresolved_requirements(capability, inputs, active_caps)
        if missing:
            return _build_missing_requirements_response(
                capability, missing, react_messages, react_step_count, logger, state
            )

        step_with_inputs = PlannedStep(
            context_key=step.get("context_key", capability),
            capability=capability,
            task_objective=step.get("task_objective", current_task),
            expected_output=step.get("expected_output"),
            success_criteria=step.get("success_criteria", "Step completed successfully"),
            inputs=inputs,
        )
        validate_single_step(step_with_inputs, state, logger)

        return self._build_return_state(
            step_with_inputs,
            capability,
            react_messages,
            react_step_count,
            step.get("task_objective", current_task),
            logger,
        )

    async def _generate_direct_response(
        self,
        state: AgentState,
        react_messages: list[dict],
        react_step_count: int,
        current_task: str,
        task_objective: str,
        logger,
    ) -> dict[str, Any]:
        """Generate a response directly using the response prompt builder.

        Instead of delegating to the respond capability (which lacks access to
        react_messages), the reactive orchestrator generates the final response
        itself.  It uses the same response generation prompt builder for
        guidelines and formatting, but includes the full reactive context chain.
        """
        from datetime import datetime

        from langchain_core.messages import AIMessage

        from osprey.infrastructure.respond_node import ResponseContext
        from osprey.state import populate_legacy_fields_from_artifacts

        logger.status("Generating response directly from reactive context...")

        # 1. Get prompt builders
        prompt_provider = get_framework_prompts()
        response_builder = prompt_provider.get_response_generation_prompt_builder()
        orchestrator_builder = prompt_provider.get_orchestrator_prompt_builder()

        # 2. Gather context summaries (all types, not filtered by step inputs)
        context_manager = ContextManager(state)
        context_summaries = context_manager.get_summaries(None)

        # 3. Format reactive context via customisable method
        formatted_react_context = orchestrator_builder.format_reactive_response_context(
            react_messages
        )

        # 4. Build ResponseContext for the response builder's guidelines
        ui_artifacts = state.get("ui_artifacts", [])
        legacy_updates = {}
        if ui_artifacts:
            legacy_updates = populate_legacy_fields_from_artifacts(ui_artifacts)

        figures_available = len(legacy_updates.get("ui_captured_figures", []))
        commands_available = len(legacy_updates.get("ui_launchable_commands", []))
        notebooks_available = len(legacy_updates.get("ui_captured_notebooks", []))

        interface_context = get_interface_context()

        response_context = ResponseContext(
            current_task=current_task,
            execution_history=[],  # not used — reactive context replaces this
            relevant_context=context_summaries,
            is_killed=state.get("control_is_killed", False),
            kill_reason=state.get("control_kill_reason"),
            capabilities_overview=None,
            total_steps_executed=react_step_count,
            execution_start_time=state.get("execution_start_time"),
            reclassification_count=state.get("control_reclassification_count", 0),
            current_date=datetime.now().strftime("%Y-%m-%d"),
            figures_available=figures_available,
            commands_available=commands_available,
            notebooks_available=notebooks_available,
            interface_context=interface_context,
        )

        # 5. Build system prompt from response builder (includes guidelines)
        system_prompt = response_builder.get_system_instructions(
            current_task=current_task,
            info=response_context,
        )

        # 6. Build user message with task + objective + reactive context
        user_content = (
            f"TASK: {current_task}\n"
            f"OBJECTIVE: {task_objective}\n\n"
            f"REACTIVE EXECUTION CONTEXT:\n{formatted_react_context}"
        )

        chat_request = ChatCompletionRequest(
            messages=[
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=user_content),
            ]
        )

        logger.debug(
            "Response LLM prompt built",
            llm_prompt=chat_request.to_single_string(),
            stream=False,
        )

        # 7. Make LLM call with response model config
        set_api_call_context(
            function="_generate_direct_response",
            module="reactive_orchestrator_node",
            class_name="ReactiveOrchestratorNode",
        )

        response_model_config = get_model_config("response")
        resp_start_time = time.time()
        response = await asyncio.to_thread(
            get_chat_completion,
            chat_request=chat_request,
            model_config=response_model_config,
        )
        resp_time = time.time() - resp_start_time

        # Handle different response types
        if isinstance(response, str):
            response_text = response
        elif isinstance(response, list):
            text_parts = [str(block) for block in response if hasattr(block, "text")]
            response_text = "\n".join(text_parts) if text_parts else str(response)
        else:
            response_text = str(response) if response else "Unable to generate response."

        logger.info(
            f"Parsing response ({resp_time:.1f}s)", llm_response=response_text, stream=False
        )
        logger.key_info("Reactive orchestrator: response generated directly")

        # 8. Build return state
        react_messages.append(
            {
                "role": "assistant",
                "content": f"Action: respond\nObjective: {task_objective}",
            }
        )

        result: dict[str, Any] = {
            "messages": [AIMessage(content=response_text)],
            "react_messages": react_messages,
            "react_step_count": react_step_count + 1,
            "react_rejection_count": 0,
            "react_response_generated": True,
            "control_has_error": False,
            "control_error_info": None,
            "control_last_error": None,
            "control_retry_count": 0,
            "control_current_step_retry_count": 0,
            **legacy_updates,
            **create_status_update(
                message="Reactive orchestrator: response generated",
                progress=1.0,
                complete=True,
                node="reactive_orchestrator",
                capability="respond",
            ),
        }

        if self._needs_approval_cleanup:
            result.update(clear_approval_state())

        return result

    def _build_return_state(
        self,
        plan_step: PlannedStep,
        capability: str,
        react_messages: list[dict],
        react_step_count: int,
        task_objective: str,
        logger,
    ) -> dict[str, Any]:
        """Build the common return dict for any dispatch path."""
        is_terminal = capability in ("respond", "clarify", "error")

        react_messages.append(
            {
                "role": "assistant",
                "content": f"Action: {capability}\nObjective: {task_objective}",
                "step": dict(plan_step),
            }
        )

        logger.key_info(f"Reactive orchestrator: {capability}")

        single_plan = ExecutionPlan(steps=[plan_step])

        # ----- Per-step approval gate -----
        # Gate non-terminal steps when planning_mode_enabled is True.
        # After _MAX_CONSECUTIVE_REJECTIONS, skip the gate to prevent
        # infinite reject→propose→reject loops.
        rejection_count = self._state.get("react_rejection_count", 0)
        if (
            not is_terminal
            and _is_planning_mode_enabled(self._state)
            and rejection_count < _MAX_CONSECUTIVE_REJECTIONS
        ):
            interrupt_data = create_step_approval_interrupt(
                dict(plan_step), react_step_count + 1, single_plan
            )
            interrupt_data["resume_payload"]["react_messages"] = react_messages
            interrupt_data["resume_payload"]["react_step_count"] = react_step_count + 1
            interrupt(interrupt_data)  # raises GraphInterrupt, never returns

        # Build result — include approval cleanup if recovering from rejection
        result = {
            "planning_execution_plan": single_plan,
            "planning_current_step_index": 0,
            "react_messages": react_messages,
            "react_step_count": react_step_count + 1,
            "react_rejection_count": (
                (self._state.get("react_rejection_count", 0) + 1)
                if self._needs_approval_cleanup
                else 0
            ),
            "control_has_error": False,
            "control_error_info": None,
            "control_last_error": None,
            "control_retry_count": 0,
            "control_current_step_retry_count": 0,
            **create_status_update(
                message=f"Reactive orchestrator: {capability}"
                if is_terminal
                else f"Reactive step {react_step_count + 1}: {capability}",
                progress=1.0 if is_terminal else 0.5,
                complete=is_terminal,
                node="reactive_orchestrator",
                capability=capability,
            ),
        }

        if self._needs_approval_cleanup:
            result.update(clear_approval_state())

        return result


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _check_unresolved_requirements(
    capability_name: str,
    resolved_inputs: list[dict[str, str]],
    active_capabilities: list,
) -> list[tuple[str, str | None]]:
    """Check if a capability's requirements are satisfied by the resolved inputs.

    Compares the capability's ``requires`` list against the resolved inputs to
    identify missing context types.  For each missing type, looks up which
    registered capability ``provides`` it so the orchestrator can auto-expand
    the active set and tell the LLM what to call first.

    Searches **all** registered capabilities (not just active ones) so that
    provider capabilities omitted by the classifier can be discovered and
    activated automatically.

    :param capability_name: Name of the capability about to be dispatched
    :param resolved_inputs: Inputs returned by ``_resolve_inputs()``
    :param active_capabilities: Currently active capability instances (unused but
        kept for API consistency; the registry is searched instead)
    :return: List of ``(missing_context_type, provider_capability_name | None)`` tuples
    """
    registry = get_registry()
    cap_instance = registry.get_capability(capability_name)
    if not cap_instance:
        return []

    requires = getattr(cap_instance, "requires", []) or []
    if not requires:
        return []

    # Collect context types that were actually resolved
    resolved_types: set[str] = set()
    for inp in resolved_inputs:
        resolved_types.update(inp.keys())

    # Build a map: context_type -> providing capability name (from ALL registered)
    provider_map: dict[str, str] = {}
    for cap in registry.get_all_capabilities():
        for provided_type in getattr(cap, "provides", []) or []:
            provider_map[provided_type] = getattr(cap, "name", "unknown")

    missing: list[tuple[str, str | None]] = []
    for req in requires:
        context_type = req[0] if isinstance(req, tuple) else req
        if context_type not in resolved_types:
            missing.append((context_type, provider_map.get(context_type)))

    return missing


def _resolve_inputs(capability_name: str | None, state: AgentState) -> list[dict[str, str]]:
    """Auto-resolve inputs for a capability based on its requires and available context.

    Matches a capability's ``requires`` list against available keys in
    ``capability_context_data`` to build the inputs list automatically.

    :param capability_name: Name of the capability to resolve inputs for
    :param state: Current agent state
    :return: List of input reference dicts ``[{context_type: context_key}, ...]``
    """
    if not capability_name:
        return []

    registry = get_registry()
    cap_instance = registry.get_capability(capability_name)
    if not cap_instance:
        return []

    requires = getattr(cap_instance, "requires", []) or []
    if not requires:
        return []

    context_data = state.get("capability_context_data", {})
    inputs = []

    for req in requires:
        # Handle both string and tuple (context_type, cardinality) formats
        if isinstance(req, tuple):
            context_type = req[0]
        else:
            context_type = req

        # Find the most recent context key for this type
        type_contexts = context_data.get(context_type, {})
        if type_contexts and isinstance(type_contexts, dict):
            # Use the last key (most recently added)
            latest_key = list(type_contexts.keys())[-1]
            inputs.append({context_type: latest_key})

    return inputs


def _get_active_capabilities(state: AgentState) -> list:
    """Reconstruct the active capability instances from state.

    :param state: Current agent state
    :return: List of capability instances
    """
    registry = get_registry()
    return [
        cap
        for name in state.get("planning_active_capabilities", [])
        if (cap := registry.get_capability(name)) is not None
    ]


def _build_missing_requirements_response(
    capability: str,
    missing: list[tuple[str, str | None]],
    react_messages: list[dict],
    react_step_count: int,
    logger,
    state: AgentState,
) -> dict[str, Any]:
    """Build a state update that feeds a dependency error back to the LLM.

    Instead of dispatching a capability whose requirements aren't met, this
    returns an observation explaining what's missing so the LLM can
    course-correct (e.g., call ``channel_finding`` before ``channel_write``).

    As a safety net, any provider capabilities referenced in ``missing`` are
    auto-added to ``planning_active_capabilities`` so the LLM can actually
    call them on the next iteration.

    :param capability: Name of the capability that was attempted
    :param missing: List of ``(context_type, provider_name | None)`` tuples
    :param react_messages: Current react message history (mutated in-place)
    :param react_step_count: Current step count
    :param logger: Logger instance
    :param state: Current agent state (for reading active capabilities)
    :return: State update dict that loops back to the orchestrator
    """
    parts = []
    for context_type, provider in missing:
        if provider:
            parts.append(
                f"Cannot execute {capability}: requires {context_type} context "
                f"(produced by {provider}). Call {provider} first."
            )
        else:
            parts.append(
                f"Cannot execute {capability}: requires {context_type} context "
                f"but no active capability provides it."
            )
    observation = " ".join(parts)

    logger.warning(f"Pre-dispatch gate blocked {capability}: {observation}")

    react_messages.append(
        {"role": "assistant", "content": f"Action: {capability} (blocked — missing dependencies)"}
    )
    react_messages.append({"role": "observation", "content": observation})

    # Auto-expand active capabilities with providers for missing context types
    active_set = set(state.get("planning_active_capabilities", []))
    expanded = list(state.get("planning_active_capabilities", []))
    for _context_type, provider in missing:
        if provider and provider not in active_set:
            expanded.append(provider)
            active_set.add(provider)
            logger.info(f"Auto-expanded active capabilities: added '{provider}'")

    result: dict[str, Any] = {
        "react_messages": react_messages,
        "react_step_count": react_step_count + 1,
        "react_rejection_count": 0,
        "control_has_error": False,
        "control_error_info": None,
        "control_last_error": None,
        "control_retry_count": 0,
        "control_current_step_retry_count": 0,
        **create_status_update(
            message=f"Reactive step {react_step_count + 1}: {capability} blocked (missing deps)",
            progress=0.5,
            complete=False,
            node="reactive_orchestrator",
            capability=capability,
        ),
    }

    if len(expanded) > len(state.get("planning_active_capabilities", [])):
        result["planning_active_capabilities"] = expanded

    return result


def _format_execution_history(state: AgentState) -> str:
    """Format step results into a human-readable execution history string.

    :param state: Current agent state
    :return: Formatted execution history
    """
    step_results = state.get("execution_step_results", {})
    if not step_results:
        return "No steps executed yet"

    history_lines = []
    ordered = sorted(step_results.items(), key=lambda x: x[1].get("step_index", 0))
    for key, result in ordered:
        cap = result.get("capability", "unknown")
        success = result.get("success", False)
        objective = result.get("task_objective", "")
        status = "SUCCESS" if success else "FAILED"
        history_lines.append(f"- Step '{key}' ({cap}): {status} - {objective}")
    return "\n".join(history_lines)


def _build_system_prompt(
    state: AgentState,
    logger,
    active_capabilities: list,
) -> str:
    """Build the reactive orchestrator system prompt using shared prompt infrastructure.

    Delegates to ``get_reactive_instructions()`` on the orchestrator prompt builder,
    which composes the prompt from shared components (role, step format, capabilities,
    context) plus reactive-specific strategy and execution history.
    """
    prompt_provider = get_framework_prompts()
    builder = prompt_provider.get_orchestrator_prompt_builder()
    context_manager = ContextManager(state)
    execution_history = _format_execution_history(state)

    return builder.get_reactive_instructions(
        active_capabilities=active_capabilities,
        context_manager=context_manager,
        execution_history=execution_history,
    )


def _build_chat_request(
    system_prompt: str,
    current_task: str,
    react_messages: list[dict],
) -> ChatCompletionRequest:
    """Build a structured chat request from system prompt, task, and history.

    Observations are taken from ``react_messages`` (already appended by the
    caller), so each observation appears exactly once — fixing the
    duplication bug in the former ``_build_messages()`` function.

    :param system_prompt: Orchestrator system prompt (cached on Anthropic)
    :param current_task: The user's current task
    :param react_messages: Accumulated assistant/observation messages
    :return: A ``ChatCompletionRequest`` with proper multi-turn structure
    """
    msgs: list[ChatMessage] = [ChatMessage(role="system", content=system_prompt)]

    # Build user task message — append first-step instruction if no history
    task_content = f"USER TASK: {current_task}"
    if not react_messages:
        task_content += "\n\nThis is the first step. Analyze the task and decide what to do first."
    msgs.append(ChatMessage(role="user", content=task_content))

    # Map react_messages to proper chat roles
    for msg in react_messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if role == "assistant":
            msgs.append(ChatMessage(role="assistant", content=content))
        elif role == "observation":
            msgs.append(ChatMessage(role="user", content=f"OBSERVATION: {content}"))

    return ChatCompletionRequest(messages=msgs)


def _format_observation(state: AgentState) -> str | None:
    """Summarize the last step result or error for LLM context.

    :param state: Current agent state
    :return: Formatted observation string, or None if no observation available
    """
    # Check for errors first
    error_info = state.get("control_error_info")
    if error_info and isinstance(error_info, dict):
        classification = error_info.get("classification")
        if classification and hasattr(classification, "user_message"):
            cap_name = error_info.get("capability_name", "unknown")
            return (
                f"ERROR in {cap_name}: {classification.user_message}\n"
                f"Severity: {classification.severity.value if hasattr(classification.severity, 'value') else classification.severity}"
            )

    # Check for last result
    last_result = state.get("execution_last_result")
    if last_result:
        if hasattr(last_result, "success"):
            cap = getattr(last_result, "capability", "unknown")
            if last_result.success:
                return f"Step completed successfully (capability: {cap})"
            else:
                error = getattr(last_result, "error", "unknown error")
                return f"Step FAILED (capability: {cap}): {error}"
        elif isinstance(last_result, dict):
            cap = last_result.get("capability", "unknown")
            if last_result.get("success"):
                return f"Step completed successfully (capability: {cap})"
            else:
                error = last_result.get("error", "unknown error")
                return f"Step FAILED (capability: {cap}): {error}"

    # Check step results for any completed work
    step_results = state.get("execution_step_results", {})
    if step_results:
        last_key = list(step_results.keys())[-1]
        last = step_results[last_key]
        cap = last.get("capability", "unknown")
        success = last.get("success", False)
        if success:
            return f"Previous step '{last_key}' ({cap}) completed successfully"
        else:
            return f"Previous step '{last_key}' ({cap}) failed"

    return None
