Orchestrator Planning
=====================

.. currentmodule:: osprey.infrastructure.orchestration_node

.. dropdown:: 📚 What You'll Learn
   :color: primary
   :icon: book

   **Key Concepts:**

   - How the plan-first orchestrator creates execution plans from tasks and capabilities
   - How the reactive orchestrator decides one step at a time
   - Shared step validation (``validate_single_step``) across both modes
   - LLM-powered planning with capability integration
   - Approval workflow integration with plan validation
   - Plan structure and execution coordination

   **Prerequisites:** Understanding of :doc:`03_classification-and-routing` and :doc:`../05_production-systems/01_human-approval-workflows`

   **Time Investment:** 15 minutes for complete understanding

Core Concept
------------

Osprey supports two orchestration modes. This page focuses on the **plan-first** mode (default) which creates complete execution plans upfront. For the **reactive** mode (ReAct), see :ref:`reactive-orchestration-planning` below.

**Plan-First Approach:**
   ``User Query → Complete Plan Creation → Execute All Steps → Response``

**Reactive Approach:**
   ``User Query → Decide Step 1 → Execute → Observe → Decide Step 2 → Execute → ... → Respond``

**Plan-first benefits:** Single planning phase, full context utilization, human oversight integration, fewer LLM calls.

**Reactive benefits:** Dynamic adaptation, graceful error recovery, step-by-step observation.

Both modes share the same ``validate_single_step()`` function and ``PlannedStep`` data model.

Architecture
------------

.. code-block:: python

   @infrastructure_node
   class OrchestrationNode(BaseInfrastructureNode):
       name = "orchestrator"
       description = "Execution Planning and Orchestration"

       @staticmethod
       async def execute(state: AgentState, **kwargs):
           # Check for approved plan first (approval workflow)
           has_approval_resume, approved_payload = get_approval_resume_data(
               state, create_approval_type("orchestrator", "plan")
           )

           if has_approval_resume and approved_payload:
               approved_plan = approved_payload.get("execution_plan")
               return _create_state_updates(state, approved_plan, "approved_from_state")

           # Generate new execution plan
           current_task = StateManager.get_current_task(state)
           active_capability_names = state.get('planning_active_capabilities')

           # Get capability instances from registry
           active_capabilities = [
               registry.get_capability(name) for name in active_capability_names
           ]

           # Generate execution plan using LLM
           execution_plan = await _generate_plan_with_llm(
               current_task, active_capabilities, state
           )

           # Validate and fix execution plan
           execution_plan = _validate_and_fix_execution_plan(
               execution_plan, current_task, logger
           )

           # Handle planning mode (approval workflow)
           if _is_planning_mode_enabled(state):
               await _handle_planning_mode(execution_plan, current_task)

           return {
               "planning_execution_plan": execution_plan,
               "planning_current_step_index": 0
           }

Execution Plan Structure
------------------------

Plans use structured TypedDict format for LangGraph compatibility:

.. code-block:: python

   class PlannedStep(TypedDict):
       context_key: str          # Unique identifier for step output
       capability: str           # Capability name to execute
       task_objective: str       # What this step should accomplish
       expected_output: str      # Expected output description
       success_criteria: str     # How to determine success
       inputs: List[str]         # Input context keys from previous steps

   class ExecutionPlan(TypedDict):
       steps: List[PlannedStep]  # Ordered list of execution steps

**Example Plan:**

.. code-block:: python

   execution_plan = ExecutionPlan(
       steps=[
           PlannedStep(
               context_key="weather_data",
               capability="current_weather",
               task_objective="Retrieve current weather for San Francisco",
               success_criteria="Weather data retrieved successfully",
               expected_output="CURRENT_WEATHER",
               inputs=[]
           ),
           PlannedStep(
               context_key="user_response",
               capability="respond",
               task_objective="Present weather information to user",
               success_criteria="User receives formatted weather data",
               expected_output="user_response",
               inputs=["weather_data"]
           )
       ]
   )

LLM-Powered Plan Generation
---------------------------

Orchestrator generates plans using comprehensive prompts with capability context:

.. code-block:: python

   async def _generate_plan_with_llm(current_task, active_capabilities, state):
       # Create system prompt with capability guides
       context_manager = ContextManager(state)
       orchestrator_builder = prompt_provider.get_orchestrator_prompt_builder()

       system_instructions = orchestrator_builder.get_planning_instructions(
           active_capabilities=active_capabilities,
           context_manager=context_manager,
           task_depends_on_chat_history=state.get('task_depends_on_chat_history', False)
       )

       # Generate structured plan
       execution_plan = await asyncio.to_thread(
           get_chat_completion,
           message=f"{prompt}\n\nTASK TO PLAN: {current_task}",
           model_config=get_model_config("orchestrator"),
           output_model=ExecutionPlan
       )

       return execution_plan

**Key Features:**
- Structured output ensures consistent plan objects
- Capability guides provide orchestration examples
- Context integration includes conversation history
- Dependency management between steps

Shared Step Validation
----------------------

Both plan-first and reactive orchestrators use the shared ``validate_single_step()`` function to validate individual execution steps:

.. code-block:: python

   from osprey.infrastructure.orchestration_node import validate_single_step

   def validate_single_step(step, state, logger, *, available_keys=None, step_index=0):
       """Validate a single execution step for correctness.

       Used by both plan-first and reactive orchestrators.

       Checks:
       1. The step's capability exists in the registry
       2. Input context key references are valid

       Raises ValueError for hallucinated capabilities,
       InvalidContextKeyError for bad context references.
       """

- **Plan-first usage**: Called inside ``_validate_and_fix_execution_plan()`` for each step, with ``available_keys`` pre-built from the plan
- **Reactive usage**: Called directly on the single step, with ``available_keys`` built from existing state context

Plan Validation and Fixing
---------------------------

The plan-first orchestrator applies additional plan-level validation on top of ``validate_single_step()``:

.. code-block:: python

   def _validate_and_fix_execution_plan(execution_plan, current_task, logger):
       steps = execution_plan.get('steps', [])

       # Check all capabilities exist in registry
       hallucinated_capabilities = []
       for step in steps:
           capability_name = step.get('capability', '')
           if not registry.get_node(capability_name):
               hallucinated_capabilities.append(capability_name)

       # If hallucinated capabilities found, trigger re-planning
       if hallucinated_capabilities:
           available_capabilities = registry.get_stats()['capability_names']
           error_msg = (
               f"Orchestrator hallucinated non-existent capabilities: {hallucinated_capabilities}. "
               f"Available capabilities: {available_capabilities}"
           )
           raise ValueError(error_msg)

       # Ensure plan ends with respond or clarify
       last_step = steps[-1]
       if last_step.get('capability', '').lower() not in ['respond', 'clarify']:
           # Add respond step
           steps.append(_create_generic_respond_step(current_task))

       return {"steps": steps}

**Validation Benefits:**
- Prevents execution failures from non-existent capabilities
- Guarantees user response with respond/clarify steps
- Enables re-planning with specific error context
- Registry integration ensures accuracy

Approval Workflow Integration
-----------------------------

Orchestrator seamlessly integrates with LangGraph's interrupt system:

.. code-block:: python

   async def _handle_planning_mode(execution_plan, current_task):
       """Handle planning mode using structured approval system."""

       # Create structured plan approval interrupt
       interrupt_data = create_plan_approval_interrupt(
           execution_plan=execution_plan,
           step_objective=current_task
       )

       # LangGraph interrupt - execution stops here until user responds
       interrupt(interrupt_data)

**Approval Features:**
- Native LangGraph integration using interrupt system
- Structured interrupts include complete plan details
- Resume support extracts approved plans without re-planning
- State management with proper cleanup

**Approval Resume Handling:**

.. code-block:: python

   # Orchestrator checks for approved plans first
   has_approval_resume, approved_payload = get_approval_resume_data(
       state, create_approval_type("orchestrator", "plan")
   )

   if has_approval_resume and approved_payload:
       approved_plan = approved_payload.get("execution_plan")
       return _create_state_updates(state, approved_plan, "approved_from_state")

State Updates and Coordination
-------------------------------

Orchestrator creates comprehensive state updates for framework coordination:

.. code-block:: python

   def _create_state_updates(state, execution_plan, approach):
       return {
           "planning_execution_plan": execution_plan,
           "planning_current_step_index": 0,
           "control_plans_created_count": state.get('control_plans_created_count', 0) + 1,
           # Clear error state so router can execute new plan
           "control_has_error": False,
           "control_error_info": None,
           "control_retry_count": 0
       }

**State Update Benefits:**
- Router coordination through error state cleanup
- Progress tracking with plan creation counters
- Clean state for execution without interference

Error Handling
--------------

Orchestrator includes robust error handling for LLM operations:

.. code-block:: python

   @staticmethod
   def classify_error(exc: Exception, context: dict):
       # Retry on network/API timeouts
       if isinstance(exc, (ConnectionError, TimeoutError)):
           return ErrorClassification(
               severity=ErrorSeverity.RETRIABLE,
               user_message="Orchestration service timeout, retrying..."
           )

       # Don't retry on validation errors
       if isinstance(exc, (ValueError, TypeError)):
           return ErrorClassification(
               severity=ErrorSeverity.CRITICAL,
               user_message="Orchestration configuration error"
           )

**Retry Policy:** 4 attempts with 2.0x backoff for LLM service reliability.

**Error Scenarios:**
- **Capability Hallucination:** Triggers reclassification with specific error context
- **Empty Plans:** Automatically adds respond step
- **Approval Failures:** Handles rejection gracefully with state consistency

Integration Examples
--------------------

**Normal Planning Flow:**

.. code-block:: python

   # Input: Task + selected capabilities
   current_task = "What's the weather in San Francisco?"
   active_capabilities = ["current_weather", "respond"]

   # Output: Validated execution plan
   execution_plan = {
       "steps": [
           {
               "context_key": "sf_weather",
               "capability": "current_weather",
               "task_objective": "Retrieve weather for San Francisco",
               "inputs": []
           },
           {
               "context_key": "weather_response",
               "capability": "respond",
               "task_objective": "Present weather to user",
               "inputs": ["sf_weather"]
           }
       ]
   }

**Approval Mode Integration:**

.. code-block:: python

   # Enable planning mode
   state["agent_control"]["planning_mode_enabled"] = True

   # Orchestrator automatically creates approval interrupt
   # Execution pauses until user approves/rejects plan
   # Approved plans resume execution without re-planning

.. _reactive-orchestration-planning:

Reactive Orchestration
-----------------------

The reactive orchestrator (``ReactiveOrchestratorNode``) implements the **ReAct (Reasoning + Acting)** pattern, deciding one step at a time and observing results between steps.

.. code-block:: python

   from osprey.infrastructure.reactive_orchestrator_node import ReactiveOrchestratorNode

   @infrastructure_node
   class ReactiveOrchestratorNode(BaseInfrastructureNode):
       name = "reactive_orchestrator"
       description = "Reactive step-by-step orchestration using ReAct pattern"

       async def execute(self) -> dict[str, Any]:
           # 1. Resolve active capabilities from state
           # 2. Build system prompt using shared prompt infrastructure
           # 3. Build messages from react_messages + latest observation
           # 4. Call LLM for next decision (output: ExecutionPlan with 1 step)
           # 5. Validate step (internal retry loop for self-correction)
           # 6. Create single-step execution plan for any capability (including respond/clarify)

It uses the same prompt builder infrastructure but calls ``get_reactive_instructions()`` instead of ``get_planning_instructions()``:

.. code-block:: python

   # Reactive orchestrator prompt generation
   builder = prompt_provider.get_orchestrator_prompt_builder()
   system_prompt = builder.get_reactive_instructions(
       active_capabilities=active_capabilities,
       context_manager=context_manager,
       execution_history=execution_history,
   )

**Key differences from plan-first:**

- Produces a single-step ``ExecutionPlan`` per invocation (vs. a complete multi-step plan)
- Includes an internal validation-retry loop (up to 2 retries) that re-prompts the LLM with feedback on validation failures
- Auto-resolves capability inputs from available context data (vs. planned upfront)
- Accumulates reasoning history in ``react_messages`` state field

**Reactive execution plan example (single step):**

.. code-block:: python

   # Each reactive orchestrator invocation produces a plan with one step
   single_step_plan = ExecutionPlan(
       steps=[
           PlannedStep(
               context_key="beam_current_pvs",
               capability="pv_address_finding",
               task_objective="Find beam current PV addresses",
               expected_output="PV_ADDRESSES",
               inputs=[]  # auto-resolved from context
           )
       ]
   )

Auto-Input Resolution
~~~~~~~~~~~~~~~~~~~~~~

Unlike plan-first mode where input references are planned upfront, the reactive orchestrator automatically resolves inputs by matching a capability's ``requires`` list against available context data in state:

.. code-block:: python

   def _resolve_inputs(capability_name, state):
       """Match capability's requires list against available context data."""
       cap_instance = registry.get_capability(capability_name)
       requires = getattr(cap_instance, "requires", [])

       for req in requires:
           context_type = req[0] if isinstance(req, tuple) else req
           type_contexts = state["capability_context_data"].get(context_type, {})
           if type_contexts:
               latest_key = list(type_contexts.keys())[-1]
               inputs.append({context_type: latest_key})

Internal Validation-Retry Loop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before propagating errors to the infrastructure error system, the reactive orchestrator gives the LLM up to 2 additional attempts to self-correct validation failures (hallucinated capabilities, invalid context references, empty plans). Validation feedback is appended to the prompt so the LLM can fix its output.

Graceful Error Recovery
~~~~~~~~~~~~~~~~~~~~~~~~

When a capability fails, the router sends control back to the reactive orchestrator (instead of the error node). The orchestrator observes the error and can decide to retry a different approach, skip the step, or respond to the user. It clears the error state to enable this recovery.

Reactive State Fields
~~~~~~~~~~~~~~~~~~~~~~

The reactive orchestrator introduces two execution-scoped state fields:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Field
     - Type
     - Description
   * - ``react_messages``
     - ``list[dict]``
     - Accumulated LLM reasoning messages (decisions + observations) for the ReAct loop
   * - ``react_step_count``
     - ``int``
     - Safety counter tracking completed steps (max iterations guard)
These fields are execution-scoped (reset each invocation) and only used in reactive orchestration mode.

Reactive Routing
~~~~~~~~~~~~~~~~~

When ``orchestration_mode: react`` is configured, the router uses a separate ``_reactive_routing()`` function with the following priority:

1. **Direct chat mode** -- same behavior as plan-first
2. **Error handling** -- RETRIABLE errors retry the capability; all other severities route back to reactive orchestrator for re-evaluation
3. **Max iterations guard** -- routes to error node if ``react_step_count`` exceeds ``graph_recursion_limit`` (default: 100)
4. **Normal pipeline** -- ``task_extraction -> classifier -> reactive_orchestrator``
5. **Execution plan dispatch** -- routes to the capability in the current execution plan step (including respond/clarify)
6. **After capability execution** -- route back to reactive orchestrator for next decision

.. seealso::

   :doc:`../01_understanding-the-framework/04_orchestration-architecture`
       High-level overview of orchestration modes and when to use each

   :doc:`../../api_reference/02_infrastructure/04_orchestration`
      API reference for orchestration classes and functions

   :doc:`../05_production-systems/01_human-approval-workflows`
       LLM-powered planning with approval workflow integration

   :doc:`03_classification-and-routing`
       Capability selection and execution coordination patterns

   :doc:`05_message-generation`
       How execution results become user responses

   :doc:`06_error-handling-infrastructure`
       How orchestration errors are handled

Orchestrator Planning provides the strategic intelligence that converts capability selections into coherent, executable plans with comprehensive validation and approval integration.
