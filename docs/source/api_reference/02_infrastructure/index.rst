==============
Infrastructure
==============

.. toctree::
   :maxdepth: 1
   :hidden:

   01_gateway
   02_task-extraction
   03_classification
   04_orchestration
   05_execution-control
   06_message-generation

.. dropdown:: What You'll Find Here
   :color: primary
   :icon: book

   **Infrastructure pipeline components for agentic execution:**

   - **Gateway** - Single entry point with state lifecycle, slash commands, and approval flow management
   - **TaskExtractionNode** - LLM-powered conversation analysis with ExtractedTask structured output
   - **ClassificationNode** - Parallel capability selection using few-shot examples and CapabilityMatch models
   - **OrchestrationNode** - ExecutionPlan creation with LangGraph-native approval interrupts
   - **RouterNode & router_conditional_edge** - Intelligent flow control with state-based routing decisions
   - **ErrorNode** - LLM-powered error explanation with ErrorClassification and retry policies
   - **RespondCapability & ClarifyCapability** - Context-aware response generation with streaming support

   **Prerequisites:** Understanding of LangGraph state management and agentic system architecture

   **Target Audience:** Infrastructure developers, system architects, pipeline implementers

The infrastructure layer implements the **Orchestrator-First Architecture** that powers sophisticated agentic behavior with deterministic execution patterns. These components transform user conversations into validated execution plans with complete oversight and approval integration.

.. currentmodule:: osprey.infrastructure

Architecture Overview
=====================

The Osprey Framework infrastructure implements a **Orchestrator-First** pipeline that eliminates the unpredictability of traditional reactive agentic systems:

**Traditional Reactive Approach:**

.. code-block:: text

   User → LLM Tool Call → Analyze → Tool Call → Analyze → Tool Call → Response
   (Multiple LLM calls, limited context, unpredictable execution)

**Orchestrator-First Approach:**

.. code-block:: text

   User → Complete Plan Creation → Human Approval → Execute All Steps → Response
   (Single planning phase, full context, deterministic execution)

**Benefits:** fewer LLM calls, complete transparency, natural human oversight, scalable execution.

Core Pipeline Components
========================

.. grid:: 1 1 2 3
   :gutter: 3

   .. grid-item-card:: 🚪 Gateway
      :link: 01_gateway
      :link-type: doc
      :class-header: bg-primary text-white
      :class-body: text-center
      :shadow: md

      **Universal Entry Point**

      Single message processing interface managing state lifecycle, slash commands, and approval workflows.

   .. grid-item-card:: 🧠 Task Extraction
      :link: 02_task-extraction
      :link-type: doc
      :class-header: bg-success text-white
      :class-body: text-center
      :shadow: md

      **Context Compression**

      Converts chat history into focused, actionable tasks with resolved references and context.

   .. grid-item-card:: 🎯 Classification
      :link: 03_classification
      :link-type: doc
      :class-header: bg-info text-white
      :class-body: text-center
      :shadow: md

      **Capability Selection**

      LLM-powered analysis selecting appropriate capabilities for extracted tasks.

.. grid:: 1 1 2 3
   :gutter: 3

   .. grid-item-card:: 🎼 Orchestration
      :link: 04_orchestration
      :link-type: doc
      :class-header: bg-warning text-white
      :class-body: text-center
      :shadow: md

      **Execution Coordination**

      Creates validated execution plans with LangGraph-native approval integration.

   .. grid-item-card:: 🔧 Execution Control
      :link: 05_execution-control
      :link-type: doc
      :class-header: bg-danger text-white
      :class-body: text-center
      :shadow: md

      **Routing & Recovery**

      Manages flow control, error handling, and agentic decision-making with retry policies.

   .. grid-item-card:: 💬 Message Generation
      :link: 06_message-generation
      :link-type: doc
      :class-header: bg-secondary text-white
      :class-body: text-center
      :shadow: md

      **Response Generation**

      Context-aware response generation with clarification workflows and domain customization.

Pipeline Integration
====================

The infrastructure components work together in a deterministic processing flow:

.. tab-set::

   .. tab-item:: Message Processing Flow

      **Complete Pipeline Architecture:**

      .. code-block:: python

         # 1. Gateway - Single Entry Point
         gateway = Gateway()
         result = await gateway.process_message(user_input, compiled_graph, config)
         # Returns: GatewayResult with agent_state or resume_command

         # 2. Task Extraction - Context Compression
         task_updates = await TaskExtractionNode.execute(state)
         # Returns: {
         #     "task_current_task": "Find beam current PV addresses",
         #     "task_depends_on_chat_history": True,
         #     "task_depends_on_user_memory": False
         # }

         # 3. Classification - Capability Selection
         classification_updates = await ClassificationNode.execute(state)
         # Returns: {
         #     "planning_active_capabilities": ["pv_address_finding", "respond"],
         #     "planning_execution_plan": None,
         #     "planning_current_step_index": 0
         # }

         # 4. Orchestration - Plan Creation
         orchestration_updates = await OrchestrationNode.execute(state)
         # Returns: {
         #     "planning_execution_plan": {"steps": [...]},
         #     "planning_current_step_index": 0
         # }

         # 5. Execution Control - Flow Management
         routing_decision = router_conditional_edge(state)
         # Returns: str ("task_extraction", "classifier", "orchestrator", etc.)

         # 6. Message Generation - Response Creation
         response_updates = await RespondCapability.execute(state)
         # Returns: {"messages": [AIMessage(content="Here are the PV addresses...")]}

         # Error Handling
         error_updates = await ErrorNode.execute(state)
         # Returns: {"messages": [AIMessage(content="LLM-generated error explanation")]}

   .. tab-item:: State Management Pattern

      **Selective Persistence Strategy:**

      .. code-block:: python

         # Gateway manages state lifecycle
         if gateway_result.resume_command:
             # Approval flow - resume existing execution
             await compiled_graph.ainvoke(gateway_result.resume_command, config=config)
         elif gateway_result.agent_state:
             # New conversation - fresh state with context persistence
             await compiled_graph.ainvoke(gateway_result.agent_state, config=config)

         # Actual state structure from AgentState TypedDict
         state: AgentState = {
             # LangGraph native messages
             "messages": [HumanMessage(content="Find beam current PV addresses")],

             # Execution-scoped fields (reset each turn)
             "task_current_task": None,
             "planning_active_capabilities": [],
             "planning_execution_plan": None,
             "planning_current_step_index": 0,
             "control_needs_reclassification": False,
             "execution_step_results": {},

             # Persistent context data (accumulates across conversations)
             "capability_context_data": {
                 "PV_ADDRESSES": {
                     "beam_current": {"address": "SR:C02-BI:G02A:CURRENT_MONITOR"}
                 },
                 "ANALYSIS_RESULTS": {
                     "experiment_1": {"peak_current": 500.2, "timestamp": "2024-01-15"}
                 }
             }
         }

   .. tab-item:: Component Registration

      **Automatic Discovery Pattern:**

      .. code-block:: python

         # Infrastructure nodes auto-register with framework
         from osprey.base.decorators import infrastructure_node
         from osprey.base.nodes import BaseInfrastructureNode
         from osprey.base.errors import ErrorClassification, ErrorSeverity

         @infrastructure_node
         class CustomInfraNode(BaseInfrastructureNode):
             name = "custom_processor"
             description = "Custom processing logic"

             @staticmethod
             def classify_error(exc: Exception, context: dict) -> ErrorClassification:
                 if isinstance(exc, (ConnectionError, TimeoutError)):
                     return ErrorClassification(
                         severity=ErrorSeverity.RETRIABLE,
                         user_message="Network error, retrying...",
                         metadata={"technical_details": str(exc)}
                     )
                 return ErrorClassification(
                     severity=ErrorSeverity.CRITICAL,
                     user_message=f"Processing error: {exc}",
                     metadata={"technical_details": str(exc)}
                 )

             @staticmethod
             async def execute(state: AgentState, **kwargs) -> Dict[str, Any]:
                 # Return LangGraph-compatible state updates
                 return {
                     "control_routing_timestamp": time.time(),
                     "execution_step_results": {"custom_result": "processed"}
                 }

         # Capabilities register as infrastructure capabilities
         from osprey.base.decorators import capability_node
         from osprey.base.capability import BaseCapability
         from langchain_core.messages import AIMessage

        @capability_node
        class CustomCapability(BaseCapability):
            name = "custom_analysis"
            description = "Custom analysis capability"
            provides = ["CUSTOM_DATA"]
            requires = ["INPUT_DATA"]

            async def execute(self) -> Dict[str, Any]:
                # Get required contexts automatically
                input_data, = self.get_required_contexts()
                # Process and return LangGraph messages pattern
                return {
                    "messages": [AIMessage(content="Analysis complete")]
                }

.. dropdown:: Next Steps
   :color: primary
   :icon: arrow-up-right

   Master the infrastructure layer by exploring components in processing order:

   .. grid:: 1 1 2 2
      :gutter: 3

      .. grid-item-card:: 🚪 Start with Gateway
         :link: 01_gateway
         :link-type: doc
         :class-header: bg-primary text-white
         :class-body: text-center
         :shadow: md

         Universal entry point handling state management, slash commands, and approval workflows

      .. grid-item-card:: 🧠 Follow the Pipeline
         :link: 02_task-extraction
         :link-type: doc
         :class-header: bg-success text-white
         :class-body: text-center
         :shadow: md

         Task extraction → Classification → Orchestration - the three-pillar processing flow

   .. grid:: 1 1 2 2
      :gutter: 3

      .. grid-item-card:: 🔧 Master Control Flow
         :link: 05_execution-control
         :link-type: doc
         :class-header: bg-danger text-white
         :class-body: text-center
         :shadow: md

         Router and error handling for intelligent flow control and recovery

      .. grid-item-card:: 💬 Complete with Responses
         :link: 06_message-generation
         :link-type: doc
         :class-header: bg-secondary text-white
         :class-body: text-center
         :shadow: md

         Response generation and clarification capabilities for adaptive communication