Orchestrator-First Architecture: Upfront Planning in Practice
=============================================================

.. dropdown:: 📚 What You'll Learn
   :color: primary
   :icon: book

   **Key Concepts:**

   - How the OrchestrationNode creates execution plans from task requirements
   - The role of plan validation in preventing capability hallucination
   - Real implementation patterns for upfront planning vs reactive execution
   - Integration with approval workflows and LangGraph interrupts

   **Prerequisites:** Understanding of :doc:`../03_core-framework-systems/01_state-management-architecture` and :doc:`../04_infrastructure-components/03_classification-and-routing`

   **Time Investment:** 10-15 minutes for complete understanding

The Orchestrator-First Approach
===============================

The Osprey Framework implements an **orchestrator-first architecture** where execution plans are created upfront before any capability execution begins. This contrasts with reactive agentic patterns that make decisions step-by-step during execution.

**Core Components:**

- **OrchestrationNode**: Creates execution plans using LLM-based planning
- **Plan Validation**: Prevents capability hallucination and ensures proper completion
- **Router Integration**: Executes plans step-by-step with deterministic routing
- **Approval Workflows**: Enables human oversight through LangGraph interrupts

OrchestrationNode Implementation
================================

Basic Structure
~~~~~~~~~~~~~~~

.. code-block:: python

   from osprey.infrastructure.orchestration_node import OrchestrationNode
   from osprey.base.planning import ExecutionPlan, PlannedStep

   @infrastructure_node
   class OrchestrationNode(BaseInfrastructureNode):
       name = "orchestrator"
       description = "Execution Planning and Orchestration"

       @staticmethod
       async def execute(state: AgentState, **kwargs) -> Dict[str, Any]:
           # 1. Extract current task and active capabilities
           current_task = StateManager.get_current_task(state)
           active_capability_names = state.get('planning_active_capabilities')

           # 2. Generate execution plan using LLM
           execution_plan = await _create_plan_with_llm(
               current_task, active_capabilities, state
           )

           # 3. Validate plan and fix common issues
           execution_plan = _validate_and_fix_execution_plan(
               execution_plan, current_task, logger
           )

           # 4. Return state updates with the plan
           return {
               "planning_execution_plan": execution_plan,
               "planning_current_step_index": 0
           }

LLM-Based Plan Generation
~~~~~~~~~~~~~~~~~~~~~~~~~

The orchestrator uses LLM calls to create structured execution plans:

.. code-block:: python

   # Real implementation from orchestration_node.py
   async def create_system_prompt() -> str:
       # Get osprey prompt builder
       prompt_provider = get_osprey_prompts()
       orchestrator_builder = prompt_provider.get_orchestrator_prompt_builder()

       # Create context-aware system instructions
       system_instructions = orchestrator_builder.get_system_instructions(
           active_capabilities=active_capabilities,
           context_manager=context_manager,
           task_depends_on_chat_history=state.get('task_depends_on_chat_history', False),
           task_depends_on_user_memory=state.get('task_depends_on_user_memory', False)
       )

       return system_instructions

   # Generate plan with single LLM call
   model_config = get_model_config("orchestrator")
   message = f"{system_prompt}\n\nTASK TO PLAN: {current_task}"

   execution_plan = await asyncio.to_thread(
       get_chat_completion,
       message=message,
       model_config=model_config,
       output_model=ExecutionPlan
   )

Plan Validation System
======================

Capability Hallucination Prevention
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The framework prevents LLMs from "inventing" non-existent capabilities:

.. code-block:: python

   def _validate_and_fix_execution_plan(execution_plan: ExecutionPlan, current_task: str, logger) -> ExecutionPlan:
       """Validate execution plan to ensure all capabilities exist and it ends properly."""

       steps = execution_plan.get('steps', [])
       hallucinated_capabilities = []

       # Check each capability exists in registry
       for i, step in enumerate(steps):
           capability_name = step.get('capability', '')
           if not registry.get_node(capability_name):
               hallucinated_capabilities.append(capability_name)
               logger.error(f"Step {i+1}: Capability '{capability_name}' not found in registry")

       # Trigger re-planning if hallucinated capabilities found
       if hallucinated_capabilities:
           error_msg = f"Orchestrator hallucinated non-existent capabilities: {hallucinated_capabilities}"
           raise ValueError(error_msg)

       # Ensure plan ends with respond or clarify
       last_step = steps[-1]
       if last_step.get('capability', '').lower() not in ['respond', 'clarify']:
           # Append respond step
           generic_response = PlannedStep(
               context_key="user_response",
               capability="respond",
               task_objective=f"Respond to user request: {current_task}",
               expected_output="user_response"
           )
           steps.append(generic_response)

       return {"steps": steps}

Error Handling and Re-planning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When validation fails, the system triggers re-classification:

.. code-block:: python

   try:
       execution_plan = _validate_and_fix_execution_plan(execution_plan, current_task, logger)
   except ValueError as e:
       # Orchestrator hallucinated capabilities - trigger re-planning
       logger.error(f"Execution plan validation failed: {e}")
       return {
           "control_needs_reclassification": True,
           "control_reclassification_reason": f"Orchestrator validation failed: {e}",
           "control_reclassification_severity": "re_planning"
       }

Router Integration
==================

Deterministic Plan Execution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The router executes plans step-by-step without runtime decisions:

.. code-block:: python

   def router_conditional_edge(state: AgentState) -> str:
       """Route to next planned step - deterministic execution."""

       # Get execution plan and current step
       execution_plan = StateManager.get_execution_plan(state)
       current_index = StateManager.get_current_step_index(state)

       if not execution_plan:
           return "orchestrator"  # Need to create plan

       plan_steps = execution_plan.get('steps', [])

       # Check if plan complete
       if current_index >= len(plan_steps):
           raise RuntimeError(
               f"CRITICAL BUG: current_step_index {current_index} >= plan_steps length {len(plan_steps)}. "
               f"Orchestrator validation failed - all plans must end with respond/clarify."
           )

       # Route to next capability in plan
       current_step = plan_steps[current_index]
       step_capability = current_step.get('capability', 'respond')

       # Validate capability exists
       if not registry.get_node(step_capability):
           logger.error(f"Capability '{step_capability}' not registered")
           return "error"

       return step_capability

Step Index Management
~~~~~~~~~~~~~~~~~~~~~

The framework tracks execution progress through state:

.. code-block:: python

   # StateManager utilities for plan execution
   @staticmethod
   def get_current_step_index(state: AgentState) -> int:
       """Get current step index with proper defaults."""
       return state.get('planning_current_step_index', 0)

   @staticmethod
   def get_current_step(state: AgentState) -> Optional[PlannedStep]:
       """Get current execution step from plan."""
       execution_plan = StateManager.get_execution_plan(state)
       if not execution_plan:
           return None

       current_index = StateManager.get_current_step_index(state)
       steps = execution_plan.get('steps', [])

       if current_index < len(steps):
           return steps[current_index]
       return None

Approval Workflow Integration
=============================

Planning Mode Support
~~~~~~~~~~~~~~~~~~~~~

The orchestrator integrates with LangGraph interrupts for human approval:

.. code-block:: python

   # Check for approved plan from previous interrupt
   has_approval_resume, approved_payload = get_approval_resume_data(
       state, create_approval_type("orchestrator", "plan")
   )

   if has_approval_resume and approved_payload:
       approved_plan = approved_payload.get("execution_plan")
       if approved_plan:
           logger.success("Using approved execution plan from agent state")
           return {
               **_create_state_updates(state, approved_plan, "approved_from_state"),
               **clear_approval_state()
           }

   # Handle planning mode with interrupts
   if _is_planning_mode_enabled(state):
       logger.info("PLANNING MODE DETECTED - entering approval workflow")
       await _handle_planning_mode(execution_plan, current_task, state, logger)

Planning Mode Detection
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def _is_planning_mode_enabled(state: AgentState) -> bool:
       """Check if planning mode is enabled via slash command or agent control."""
       agent_control = state.get('agent_control', {})
       return agent_control.get('planning_mode_enabled', False)

Production Advantages
=====================

**Predictable Execution**
   - Single LLM call for planning instead of iterative decisions
   - Deterministic routing follows predetermined plan
   - Complete validation before execution begins
   - Failed plans trigger re-classification rather than cascade failures

**Error Classification with Retry Policies**

.. code-block:: python

   @staticmethod
   def classify_error(exc: Exception, context: dict):
       """Built-in error classification for orchestration operations."""

       # Retry LLM timeouts (orchestration uses LLM heavily)
       if 'timeout' in exc.__class__.__name__.lower():
           return ErrorClassification(
               severity=ErrorSeverity.RETRIABLE,
               user_message="LLM timeout during execution planning, retrying...",
               metadata={"technical_details": str(exc)}
           )

       # Don't retry planning/validation errors (logic issues)
       if isinstance(exc, (ValueError, TypeError)):
           return ErrorClassification(
               severity=ErrorSeverity.CRITICAL,
               user_message="Execution planning configuration error"
           )

   @staticmethod
   def get_retry_policy() -> Dict[str, Any]:
       """Custom retry policy for LLM-based orchestration operations."""
       return {
           "max_attempts": 4,        # More attempts for LLM operations
           "delay_seconds": 2.0,     # Longer initial delay for LLM services
           "backoff_factor": 2.0     # Aggressive backoff for rate limiting
       }

Implementation Example
======================

Complete Orchestration Flow
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # 1. User request: "Find beam current PV addresses"

   # 2. Task extraction creates structured task
   task = "Find EPICS PV addresses for beam current monitoring in the ALS storage ring"

   # 3. Classification selects relevant capabilities
   active_capabilities = ["pv_address_finding", "respond"]

   # 4. Orchestrator creates execution plan
   execution_plan = {
       "steps": [
           {
               "context_key": "beam_current_pvs",
               "capability": "pv_address_finding",
               "task_objective": "Find EPICS PV addresses for beam current monitoring",
               "expected_output": "PV_ADDRESSES"
           },
           {
               "context_key": "user_response",
               "capability": "respond",
               "task_objective": "Present found PV addresses to user",
               "expected_output": "user_response",
               "inputs": [{"PV_ADDRESSES": "beam_current_pvs"}]
           }
       ]
   }

   # 5. Router executes plan deterministically
   # Step 1: router_conditional_edge(state) -> "pv_address_finding"
   # Step 2: router_conditional_edge(state) -> "respond"
   # Final: router_conditional_edge(state) -> "END"

.. seealso::

   :doc:`../02_quick-start-patterns/01_building-your-first-capability`
       Create capabilities that work with orchestrated plans

   :doc:`../03_core-framework-systems/01_state-management-architecture`
       Understand execution state handling

   :doc:`../05_production-systems/01_human-approval-workflows`
       Add human oversight to plans

   :doc:`../../example-applications/index`
       Orchestration in complex scenarios

   :doc:`../../api_reference/02_infrastructure/04_orchestration`
      Complete orchestration implementation and execution planning

   :doc:`../../api_reference/01_core_framework/02_state_and_context`
      Plan data structures and state management patterns

   :doc:`../../api_reference/02_infrastructure/05_execution-control`
       Plan execution routing and deterministic flow control

   :doc:`../../api_reference/01_core_framework/02_state_and_context`
       State utilities for orchestration and plan management
