Orchestrator Planning
=====================

.. currentmodule:: osprey.infrastructure.orchestration_node

.. dropdown:: 📚 What You'll Learn
   :color: primary
   :icon: book

   **Key Concepts:**

   - How orchestrator creates execution plans from tasks and capabilities
   - LLM-powered planning with capability integration
   - Approval workflow integration with plan validation
   - Plan structure and execution coordination

   **Prerequisites:** Understanding of :doc:`03_classification-and-routing` and :doc:`../05_production-systems/01_human-approval-workflows`

   **Time Investment:** 15 minutes for complete understanding

Core Concept
------------

Orchestrator creates complete execution plans upfront, enabling validation, approval, and optimized execution without mid-stream planning decisions.

**Traditional Approach:**
   ``User Query → Tool Call 1 → Analyze → Tool Call 2 → Analyze → Tool Call 3 → Response``

**Orchestrator-First Approach:**
   ``User Query → Complete Plan Creation → Execute All Steps → Response``

**Benefits:** Single planning phase, full context utilization, human oversight integration, fewer LLM calls.

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

       system_instructions = orchestrator_builder.get_system_instructions(
           active_capabilities=active_capabilities,
           context_manager=context_manager,
           task_depends_on_chat_history=state.get('task_depends_on_chat_history', False)
       )

       # Generate structured plan
       execution_plan = await asyncio.to_thread(
           get_chat_completion,
           message=f"{system_instructions}\n\nTASK TO PLAN: {current_task}",
           model_config=get_model_config("orchestrator"),
           output_model=ExecutionPlan
       )

       return execution_plan

**Key Features:**
- Structured output ensures consistent plan objects
- Capability guides provide orchestration examples
- Context integration includes conversation history
- Dependency management between steps

Plan Validation and Fixing
---------------------------

Orchestrator includes comprehensive validation to prevent execution failures:

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

.. seealso::

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
