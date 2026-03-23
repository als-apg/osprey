==========================
Message and Execution Flow
==========================

.. currentmodule:: osprey.infrastructure

The Osprey Framework implements a router-controlled message processing system that transforms user input into agent responses through coordinated infrastructure components.

.. dropdown:: 📚 What You'll Learn
   :color: primary
   :icon: book

   **Key Concepts:**

   - Router-controlled architecture and message processing flow
   - :class:`Gateway` preprocessing and state management
   - Infrastructure component coordination
   - Capability execution patterns
   - Error handling and approval workflow integration

   **Prerequisites:** Understanding of :doc:`01_state-management-architecture` (AgentState), :doc:`02_context-management-system` (ContextManager), and :doc:`03_registry-and-discovery` (Registry systems)

   **Time Investment:** 45-60 minutes for complete understanding

Architecture Overview
=====================

The framework uses a **router-controlled architecture** where a central RouterNode determines execution flow based on agent state:

**Core Components:**

- :class:`Gateway`: Single entry point for message preprocessing and state creation
- :class:`RouterNode`: Central routing authority that determines next execution steps
- :class:`TaskExtractionNode`: Converts conversations into actionable task descriptions
- :class:`ClassificationNode`: Selects appropriate capabilities based on task analysis
- :class:`OrchestrationNode`: Creates detailed execution plans for multi-step workflows (plan-first mode)
- :class:`ReactiveOrchestratorNode`: Decides one step at a time using the ReAct pattern (reactive mode)
- **Capabilities**: Registry-discovered components that execute business logic
- :class:`RespondCapability`: Final response generation
- :class:`ErrorNode`: Error handling and recovery
- :class:`ClarifyCapability`: Clarification requests for ambiguous tasks

**Execution Flow:**

1. Gateway processes user input and creates fresh state
2. Router determines next step based on current state
3. Infrastructure nodes (task extraction, classification) prepare execution
4. Orchestration depends on the configured mode:
   - **Plan-first** (default): ``OrchestrationNode`` creates a complete plan, router executes each step sequentially
   - **Reactive**: ``ReactiveOrchestratorNode`` decides one step at a time, router returns control after each capability execution
5. Router coordinates multi-step workflows
6. Response generation completes the cycle

Gateway: Single Entry Point
============================

The Gateway handles all message preprocessing, state management, and routing decisions.

.. code-block:: python

   from osprey.infrastructure.gateway import Gateway
   from osprey.graph import create_graph

   async def process_user_message(user_input: str) -> None:
       # Initialize Gateway and graph
       gateway = Gateway()
       graph = create_graph()
       config = {"configurable": {"thread_id": "session_id"}}

       # Gateway processes message and returns execution-ready result
       result = await gateway.process_message(user_input, graph, config)

       if result.error:
           print(f"Gateway error: {result.error}")
           return

       # Execute based on result type
       if result.resume_command:
           # Approval workflow resumption
           final_state = await graph.ainvoke(result.resume_command, config=config)
       elif result.agent_state:
           # Normal conversation flow
           final_state = await graph.ainvoke(result.agent_state, config=config)

**Gateway Result Types:**

.. code-block:: python

   @dataclass
   class GatewayResult:
       # For normal conversation flow
       agent_state: Optional[Dict[str, Any]] = None

       # For interrupt/approval flow
       resume_command: Optional[Command] = None

       # Processing metadata
       slash_commands_processed: List[str] = None
       approval_detected: bool = False
       is_interrupt_resume: bool = False

       # Error handling
       error: Optional[str] = None

**Gateway Processing Functions:**

.. code-block:: python

   class Gateway:
       async def process_message(self, user_input: str, compiled_graph, config) -> GatewayResult:
           # Check for pending interrupts (approval workflow)
           if self._has_pending_interrupts(compiled_graph, config):
               return await self._handle_interrupt_flow(user_input, compiled_graph, config)

           # Process as new conversation turn
           return await self._handle_new_message_flow(user_input, compiled_graph, config)

       async def _handle_new_message_flow(self, user_input: str, compiled_graph, config) -> GatewayResult:
           # Parse slash commands
           slash_commands, cleaned_message = self._parse_slash_commands(user_input)

           # Get current state to preserve context
           current_state = None
           if compiled_graph and config:
               graph_state = compiled_graph.get_state(config)
               current_state = graph_state.values if graph_state else None

           # Create fresh state with context preservation
           fresh_state = StateManager.create_fresh_state(
               user_input=cleaned_message.strip() if cleaned_message.strip() else user_input,
               current_state=current_state
           )

           # Apply slash commands
           if slash_commands:
               agent_control_updates = self._apply_slash_commands(slash_commands)
               fresh_state['agent_control'].update(agent_control_updates)

           return GatewayResult(
               agent_state=fresh_state,
               slash_commands_processed=list(slash_commands.keys()) if slash_commands else []
           )

Router-Controlled Execution Flow
=================================

The RouterNode serves as the central decision-making authority, determining execution flow based on agent state.

.. code-block:: python

   from osprey.infrastructure.router_node import RouterNode, router_conditional_edge

   @infrastructure_node(quiet=True)
   class RouterNode(BaseInfrastructureNode):
       name = "router"
       description = "Central routing decision authority"

       @staticmethod
       async def execute(state: AgentState, **kwargs) -> Dict[str, Any]:
           # Update routing metadata only
           return {
               "control_routing_timestamp": time.time(),
               "control_routing_count": state.get("control_routing_count", 0) + 1
           }

**Router Conditional Edge Logic:**

The router first checks the configured orchestration mode and delegates accordingly:

.. code-block:: python

   def router_conditional_edge(state: AgentState) -> str:
       """Central routing logic that determines next execution step."""

       # Reactive mode early exit
       orchestration_mode = get_config_value(
           "execution_control.agent_control.orchestration_mode", "plan_first"
       )
       if orchestration_mode == "react":
           return _reactive_routing(state, logger)

       # Plan-first routing:

       # Check for errors first
       if state.get('control_has_error'):
           return "error"

       # Check if task extraction needed
       if not state.get('task_current_task'):
           return "task_extraction"

       # Check if classification needed
       if not state.get('planning_active_capabilities'):
           return "classifier"

       # Check if orchestration needed
       if not state.get('planning_execution_plan'):
           return "orchestrator"

       # Route to next capability in execution plan
       current_step_index = state.get('planning_current_step_index', 0)
       execution_plan = state.get('planning_execution_plan', {})
       steps = execution_plan.get('steps', [])

       if current_step_index < len(steps):
           current_step = steps[current_step_index]
           return current_step.get('capability', 'respond')

       # Execution complete - generate response
       return "respond"

In reactive mode, ``_reactive_routing()`` follows a different priority: it uses execution plan dispatch for all capabilities (including respond/clarify), routes errors back to the orchestrator for re-evaluation, and enforces a max iterations guard. See :doc:`../04_infrastructure-components/03_classification-and-routing` for details.

Task Extraction
================

TaskExtractionNode converts conversation history into structured, actionable tasks.

.. code-block:: python

   from osprey.infrastructure.task_extraction_node import TaskExtractionNode
   from osprey.prompts.defaults.task_extraction import ExtractedTask

   @infrastructure_node
   class TaskExtractionNode(BaseInfrastructureNode):
       name = "task_extraction"
       description = "Task Extraction and Processing"

       @staticmethod
       async def execute(state: AgentState, **kwargs) -> Dict[str, Any]:
           # Get conversation messages
           messages = state["messages"]

           # Extract task using LLM
           extracted_task = await _extract_task(messages, retrieval_result, logger)

           # Update state with task information
           return {
               "task_current_task": extracted_task.task,
               "task_depends_on_chat_history": extracted_task.depends_on_chat_history,
               "task_depends_on_user_memory": extracted_task.depends_on_user_memory
           }

**ExtractedTask Structure:**

.. code-block:: python

   class ExtractedTask(BaseModel):
       task: str = Field(description="Clear, actionable task description")
       depends_on_chat_history: bool = Field(description="Whether task needs conversation context")
       depends_on_user_memory: bool = Field(description="Whether task needs persistent user context")

Classification
==============

ClassificationNode analyzes tasks and selects appropriate capabilities.

.. code-block:: python

   from osprey.infrastructure.classification_node import ClassificationNode, select_capabilities

   @infrastructure_node
   class ClassificationNode(BaseInfrastructureNode):
       name = "classifier"
       description = "Task Classification and Capability Selection"

       @staticmethod
       async def execute(state: AgentState, **kwargs) -> Dict[str, Any]:
           # Get current task
           current_task = state.get("task_current_task")

           # Get available capabilities from registry
           registry = get_registry()
           available_capabilities = registry.get_all_capabilities()

           # Select capabilities using LLM analysis
           active_capabilities = await select_capabilities(
               task=current_task,
               available_capabilities=available_capabilities,
               state=state,
               logger=logger
           )

           return {
               "planning_active_capabilities": active_capabilities,
               "planning_execution_plan": None,
               "planning_current_step_index": 0
           }

Orchestration
=============

OrchestrationNode creates detailed execution plans with LLM coordination.

.. code-block:: python

   from osprey.infrastructure.orchestration_node import OrchestrationNode
   from osprey.base.planning import ExecutionPlan, PlannedStep

   @infrastructure_node
   class OrchestrationNode(BaseInfrastructureNode):
       name = "orchestrator"
       description = "Execution Planning and Orchestration"

       @staticmethod
       async def execute(state: AgentState, **kwargs) -> Dict[str, Any]:
           # Get planning context
           current_task = state.get('task_current_task')
           active_capabilities = state.get('planning_active_capabilities', [])

           # Create execution plan using LLM
           execution_plan = await create_execution_plan(
               task=current_task,
               capabilities=active_capabilities,
               state=state
           )

           # Handle planning mode (approval workflow)
           if _is_planning_mode_enabled(state):
               await _handle_planning_mode(execution_plan, current_task, state, logger)
               # Execution pauses here until user approval

           return {
               "planning_execution_plan": execution_plan,
               "planning_current_step_index": 0
           }

**ExecutionPlan Structure:**

.. code-block:: python

   execution_plan = ExecutionPlan(
       steps=[
           PlannedStep(
               context_key="search_step",
               capability="pv_address_finding",
               task_objective="Find beam current PV addresses",
               success_criteria="PV addresses discovered",
               expected_output="PV_ADDRESSES",
               inputs=[]
           ),
           PlannedStep(
               context_key="analysis_step",
               capability="data_analysis",
               task_objective="Analyze beam current data",
               success_criteria="Analysis completed",
               expected_output="ANALYSIS_RESULTS",
               inputs=[{"PV_ADDRESSES": "search_step"}]
           )
       ]
   )

Capability Execution
====================

Capabilities execute business logic according to the orchestrated plan.

.. code-block:: python

   from osprey.base import BaseCapability
   from osprey.decorators import capability_node

   @capability_node
   class ExampleCapability(BaseCapability):
       name = "example_capability"
       description = "Example capability implementation"
       requires = ["INPUT_DATA"]
       provides = ["RESULTS"]

       async def execute(self) -> Dict[str, Any]:
           # Get required contexts (automatically extracted)
           input_data, = self.get_required_contexts()

           # Execute capability business logic
           result = await perform_business_logic(input_data)

           # Store results using helper method
           return self.store_output_context(result)

State Management
================

StateManager provides utilities for state creation and context storage.

.. code-block:: python

   from osprey.state import StateManager, AgentState

   class StateManager:
       @staticmethod
       def create_fresh_state(
           user_input: str,
           current_state: Optional[AgentState] = None
       ) -> AgentState:
           """Create fresh state preserving only capability context data."""
           # Implementation creates new state with preserved context

       @staticmethod
       def get_current_step(state: AgentState) -> PlannedStep:
           """Get current execution step from orchestration plan."""
           # Implementation extracts current step

       @staticmethod
       def store_context(
           state: AgentState,
           context_type: str,
           context_key: str,
           context_data: Any
       ) -> Dict[str, Any]:
           """Store capability results in context system."""
           # Implementation stores context data

Context Management
==================

ContextManager provides access to capability context data with Pydantic serialization.

.. code-block:: python

   from osprey.context.context_manager import ContextManager

   class ContextManager:
       def __init__(self, state: AgentState):
           self._data = state['capability_context_data']
           self._object_cache = {}

       def get_context(self, context_type: str, key: str) -> Optional[CapabilityContext]:
           """Retrieve context object with automatic Pydantic reconstruction."""
           # Implementation reconstructs context objects

       def set_context(self, context_type: str, key: str, value: CapabilityContext) -> None:
           """Store context object with automatic Pydantic serialization."""
           # Implementation stores context data

       def get_all_of_type(self, context_type: str) -> Dict[str, CapabilityContext]:
           """Get all context objects of specified type."""
           # Implementation returns all matching contexts

Error Handling
==============

ErrorNode handles error recovery and response generation.

.. code-block:: python

   from osprey.infrastructure.error_node import ErrorNode

   @infrastructure_node
   class ErrorNode(BaseInfrastructureNode):
       name = "error"
       description = "Error Response Generation"

       @staticmethod
       async def execute(state: AgentState, **kwargs) -> Dict[str, Any]:
           # Generate error response based on error context
           error_info = state.get('control_error_info', {})
           error_response = await generate_error_response(error_info)

           return {
               "messages": [AIMessage(content=error_response)]
           }

Complete Working Example
========================

.. code-block:: python

   from osprey.infrastructure.gateway import Gateway
   from osprey.graph import create_graph
   from osprey.registry import get_registry

   async def complete_message_processing():
       # Initialize framework
       registry = get_registry()
       gateway = Gateway()
       graph = create_graph(registry)
       config = {"configurable": {"thread_id": "demo"}}

       # Process user message
       user_message = "Find beam current PV addresses"

       # Gateway preprocessing
       result = await gateway.process_message(user_message, graph, config)

       if result.error:
           print(f"Error: {result.error}")
           return

       # Execute through router-controlled flow
       final_state = await graph.ainvoke(result.agent_state, config=config)

       # Access results
       messages = final_state.get("messages", [])
       final_response = messages[-1].content if messages else "No response generated"
       print(f"Response: {final_response}")

Graph Construction
==================

The framework uses LangGraph with router-controlled conditional edges.

.. code-block:: python

   from osprey.graph import create_graph
   from osprey.registry import get_registry

   def create_graph(registry: RegistryManager) -> StateGraph:
       # Get all nodes from registry
       all_nodes = registry.get_all_nodes().items()

       # Create StateGraph
       workflow = StateGraph(AgentState)

       # Add all nodes
       for name, node_callable in all_nodes:
           workflow.add_node(name, node_callable)

       # Set up router-controlled flow
       workflow.set_entry_point("router")
       workflow.add_conditional_edges("router", router_conditional_edge)

       # All nodes route back to router
       for name, _ in all_nodes:
           if name != "router":
               workflow.add_edge(name, "router")

       return workflow.compile()

.. seealso::
   :doc:`../04_infrastructure-components/01_gateway-architecture`
       Gateway implementation details
   :doc:`../04_infrastructure-components/02_task-extraction-system`
       Task extraction system documentation
   :doc:`../04_infrastructure-components/03_classification-and-routing`
       Classification and routing systems
   :doc:`../04_infrastructure-components/04_orchestrator-planning`
       Orchestration and planning system
