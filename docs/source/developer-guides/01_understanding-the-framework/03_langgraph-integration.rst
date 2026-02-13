LangGraph Integration: Native StateGraph and Workflow Execution
================================================================

.. dropdown:: 📚 What You'll Learn
   :color: primary
   :icon: book

   **Key Concepts:**

   - How the framework uses LangGraph's StateGraph for workflow execution
   - Native state management with MessagesState integration
   - Checkpoint-based persistence for conversation continuity
   - Interrupt system for human approval workflows
   - Real-time streaming implementation

   **Prerequisites:** Basic understanding of LangGraph concepts

   **Time Investment:** 15-20 minutes for complete understanding

Overview
========

The Osprey Framework leverages LangGraph's native features for production conversational agentic systems:

1. **StateGraph Workflow**: Registry-based node discovery with router-controlled flow
2. **MessagesState Foundation**: Native message handling with selective persistence
3. **Checkpoint System**: PostgreSQL and memory-based conversation persistence
4. **Native Interrupts**: Built-in human-in-the-loop for approval workflows
5. **Custom Streaming**: Real-time status updates through LangGraph's streaming API

Core LangGraph Integration
==========================

StateGraph Workflow Creation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The framework creates LangGraph workflows using registry-discovered components:

.. code-block:: python

   from langgraph.graph import StateGraph
   from osprey.state import AgentState
   from osprey.registry import get_registry
   from osprey.graph.graph_builder import create_graph

   # Framework automatically creates StateGraph from registry
   def create_graph(registry: RegistryManager) -> StateGraph:
       """Create LangGraph workflow with all framework components."""

       # Use framework's native state structure
       workflow = StateGraph(AgentState)

       # Add all nodes from registry (infrastructure + capabilities)
       all_nodes = registry.get_all_nodes().items()
       for name, node_callable in all_nodes:
           workflow.add_node(name, node_callable)

       # Router controls all flow via conditional edges
       workflow.add_conditional_edges("router", router_conditional_edge, {
           "task_extraction": "task_extraction",
           "classifier": "classifier",
           "orchestrator": "orchestrator",
           "pv_address_finding": "pv_address_finding",
           "respond": "respond",
           "END": END
       })

       # All nodes route back to router for next decision
       for name in node_names:
           if name not in ["router", "respond", "clarify", "error"]:
               workflow.add_edge(name, "router")

       return workflow.compile(checkpointer=checkpointer)

**Key integration points:**
- **AgentState**: Extends LangGraph's MessagesState for compatibility
- **Node Functions**: All decorators create LangGraph-compatible callables
- **Router Control**: Central routing using LangGraph's conditional edges
- **Automatic Compilation**: Framework handles checkpointer configuration

Native State Management
~~~~~~~~~~~~~~~~~~~~~~~

The framework's state extends LangGraph's MessagesState with selective persistence:

.. code-block:: python

   from langgraph.graph import MessagesState
   from typing import Annotated, Dict, Any

   class AgentState(MessagesState):
       """Framework state extending LangGraph's MessagesState."""

       # LangGraph automatically manages 'messages' field

       # PERSISTENT: Only this field survives conversation turns
       capability_context_data: Annotated[
           Dict[str, Dict[str, Dict[str, Any]]],
           merge_capability_context_data  # Custom reducer
       ]

       # EXECUTION-SCOPED: Reset automatically each turn
       task_current_task: Optional[str]
       planning_active_capabilities: List[str]
       planning_execution_plan: Optional[ExecutionPlan]
       planning_current_step_index: int

**State management features:**

- **Native Messages**: LangGraph handles message history automatically

- **Selective Persistence**: Only `capability_context_data` persists across conversations

- **Custom Reducer**: Framework provides specialized context merging

- **Type Safety**: Full TypedDict definitions with proper annotations

Checkpoint-Based Persistence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LangGraph's checkpointer system provides automatic state persistence:

.. admonition:: Checkpoint Implementation Options
   :class: note

   **Recommended Default**: The framework uses LangGraph's **in-memory checkpointer** (`MemorySaver`) as the recommended default for most use cases, providing fast performance and simple setup.

   **PostgreSQL Support**: PostgreSQL checkpointing is available through `use_postgres=True` for scenarios requiring long-term state persistence across system restarts and container lifecycles.

.. code-block:: python

   from langgraph.checkpoint.memory import MemorySaver
   from langgraph.checkpoint.postgres import PostgresSaver

   # Development: In-memory checkpointer
   def create_development_graph(registry):
       checkpointer = MemorySaver()
       return create_graph(registry, checkpointer=checkpointer)

   # Production: PostgreSQL checkpointer
   def create_production_graph(registry):
       checkpointer = create_async_postgres_checkpointer()
       return create_graph(registry, checkpointer=checkpointer)

   # Usage with conversation continuity
   config = {"configurable": {"thread_id": "user-123"}}

   # First conversation
   response1 = await graph.ainvoke(
       {"messages": [HumanMessage(content="Find beam current PVs")]},
       config=config
   )

   # Later conversation - automatically resumes with context
   response2 = await graph.ainvoke(
       {"messages": [HumanMessage(content="Show me the analysis")]},
       config=config  # Same thread_id = same conversation
   )

Human Approval Workflows
========================

The framework uses LangGraph's native interrupt system for human-in-the-loop operations:

.. code-block:: python

   from langgraph.types import interrupt
   from osprey.approval.approval_system import create_code_approval_interrupt

   # In Python executor service - request human approval
   @staticmethod
   async def analyze_code(state: PythonExecutionState) -> Dict[str, Any]:
       # Analyze code for safety
       safety_analysis = analyze_code_safety(generated_code)

       if safety_analysis.requires_approval:
           # Create structured approval interrupt
           approval_data = create_code_approval_interrupt(
               code=generated_code,
               safety_concerns=safety_analysis.concerns,
               execution_environment="container"
           )

           # LangGraph interrupt - execution stops here
           interrupt(approval_data)

       # Continue if no approval needed
       return {"analysis_complete": True}

**Interrupt workflow:**

.. code-block:: python

   # 1. Service generates code and requests approval
   result = await python_service.ainvoke(request, config=config)
   # Execution pauses - interrupt created

   # 2. Check for interrupts
   graph_state = graph.get_state(config)
   if graph_state.interrupts:
       interrupt_data = graph_state.interrupts[0]
       # Show approval UI with interrupt_data

   # 3. Human approves/rejects
   approval_response = "approved"  # or "rejected"

   # 4. Resume with approval
   resume_command = Command(resume={"approved": True})
   await graph.ainvoke(resume_command, config=config)

Service Integration with Interrupts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The framework handles service calls that may generate interrupts:

.. code-block:: python

   from osprey.approval.approval_system import handle_service_with_interrupts

   @capability_node
   class PythonExecutorCapability(BaseCapability):
       """Execute Python code with approval workflows."""

       async def execute(self) -> Dict[str, Any]:
           # Get Python executor service
           service = registry.get_service("python_executor")

           # Handle service with interrupt propagation
           try:
               result = await handle_service_with_interrupts(
                   service=service,
                   request={"code": python_code, "mode": "execute"},
                   config=config,
                   logger=logger,
                   capability_name="python_executor"
               )
               return {"execution_results": result}

           except RuntimeError as e:
               return {"error": f"Service execution failed: {e}"}

Real-Time Streaming
===================

The framework provides real-time status updates through LangGraph's streaming via the unified logger:

.. code-block:: python

   @capability_node
   class DataAnalysisCapability(BaseCapability):
       async def execute(self) -> Dict[str, Any]:
           # Get unified logger with automatic streaming
           logger = self.get_logger()

           # Provide real-time status updates
           logger.status("Loading data sources...")
           data = await load_data_sources()

           logger.status("Performing analysis...")
           analysis = await perform_analysis(data)

           logger.success("Analysis complete")

           return {"analysis_results": analysis}

   # Client receives real-time updates
   async for chunk in graph.astream(input_data, config, stream_mode="custom"):
       if chunk.get("event_type") == "status":
           print(f"Status: {chunk['message']}")

Configuration Options
=====================

PostgreSQL Checkpointer Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from osprey.graph.graph_builder import create_async_postgres_checkpointer

   def create_production_checkpointer():
       """Create PostgreSQL checkpointer for production."""
       # Uses environment POSTGRESQL_URI or defaults to local
       checkpointer = create_async_postgres_checkpointer()
       return checkpointer

   # Production graph with persistence
   production_graph = create_graph(
       registry=get_registry(),
       use_postgres=True  # Automatically uses PostgreSQL
   )

Checkpointer Options
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Recommended default: In-memory checkpointer (fast, simple setup)
   default_graph = create_graph(registry, use_postgres=False)

   # Long-term persistence: PostgreSQL (requires database setup)
   # Note: Available but not extensively production-tested
   persistent_graph = create_graph(registry, use_postgres=True)

   # Testing: No persistence for isolated tests
   test_graph = create_graph(registry, checkpointer=None)

Benefits of Native LangGraph Integration
========================================

**Zero-Configuration Persistence**
   Automatic conversation memory across restarts

**Built-in Human-in-the-Loop**
   Native approval workflows through interrupts

**Production-Ready Streaming**
   Real-time status updates from all framework components

**Fault Tolerance**
   Conversations survive system crashes and can resume from checkpoints

Common Integration Patterns
===========================

Interrupt Handling
~~~~~~~~~~~~~~~~~~
.. code-block:: python

   # Check for pending interrupts
   def check_interrupts(graph, config):
       state = graph.get_state(config)
       if state.interrupts:
           return {
               "has_interrupts": True,
               "interrupt_data": state.interrupts[0],
               "next": state.next
           }
       return {"has_interrupts": False}

Custom State Updates
~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

   # Framework provides StateManager for consistent state updates
   from osprey.state import StateManager

   # Store capability results (or use self.store_output_context in capabilities)
   return StateManager.store_context(
       state, "PV_ADDRESSES", context_key, pv_data
   )

Troubleshooting
===============

**Issue**: State not persisting between conversations
.. code-block:: python

   # Problem: No checkpointer configured
   graph = create_graph(registry)  # No persistence

   # Solution: Configure checkpointer
   graph = create_graph(registry, use_postgres=True)
   config = {"configurable": {"thread_id": "user-123"}}

**Issue**: Streaming not working

.. code-block:: python

   # Problem: Wrong stream mode
   async for chunk in graph.astream(input_data, config):
       # Only gets final results

   # Solution: Use custom stream mode
   async for chunk in graph.astream(input_data, config, stream_mode="custom"):
       # Gets real-time status updates

.. seealso::

   :doc:`../02_quick-start-patterns/01_building-your-first-capability`
       Create LangGraph-integrated capabilities

   :doc:`../03_core-framework-systems/01_state-management-architecture`
       Deep dive into state patterns

   :doc:`../05_production-systems/05_container-and-deployment`
       Deploy with PostgreSQL checkpointing

   :doc:`../05_production-systems/01_human-approval-workflows`
       Complex interrupt handling

   :doc:`../../api_reference/01_core_framework/02_state_and_context`
       Complete state structure and MessagesState integration

   :doc:`../../api_reference/02_infrastructure/01_gateway`
       LangGraph integration entry point for message processing

   :doc:`../../api_reference/03_production_systems/01_human-approval`
       Interrupt system API for human-in-the-loop workflows

   :doc:`../../api_reference/03_production_systems/05_container-management`
       Production deployment patterns with PostgreSQL checkpointing
