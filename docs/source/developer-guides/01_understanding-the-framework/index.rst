Understanding Osprey
====================

.. toctree::
   :maxdepth: 2
   :caption: Understanding Osprey
   :hidden:

   01_infrastructure-architecture
   02_convention-over-configuration
   03_langgraph-integration
   04_orchestration-architecture

The Osprey Framework is a production-ready architecture for deploying agentic AI in large-scale, safety-critical control system environments. Built on LangGraph's StateGraph foundation, its distinctive architecture centers on **Classification and Orchestration** — capability selection followed by intelligent execution coordination (plan-first or reactive) — providing transparent, auditable multi-step operations with mandatory safety checks for hardware-interacting workflows.

.. image:: ../../_static/resources/workflow_overview.pdf
   :alt: Osprey Framework Workflow Overview
   :align: center
   :width: 100%

|  **Production Deployment Example**: This diagram illustrates the framework architecture using capabilities from the :doc:`ALS Accelerator Assistant <../../example-applications/als-assistant>` - our production deployment at Lawrence Berkeley National Laboratory's Advanced Light Source particle accelerator.


Processing Pipeline
-------------------

All user interactions—from CLI, web interfaces, or external integrations—flow through a unified :doc:`Gateway <../04_infrastructure-components/01_gateway-architecture>` that normalizes input and coordinates the processing pipeline. The framework converts natural-language requests into transparent, executable plans through four stages:

**1. Task Extraction**
   :doc:`Converts conversational context <../04_infrastructure-components/02_task-extraction-system>` into structured, actionable objectives. Transforms arbitrarily long chat history and external data sources into a well-defined current task with explicit requirements and context. Integrates facility-specific data from channel databases, archiver systems, operational memory, and knowledge bases to enrich task understanding.

**2. Classification**
   :doc:`Dynamically selects relevant capabilities <../04_infrastructure-components/03_classification-and-routing>` from your domain-specific toolkit. LLM-powered binary classification for each capability ensures only relevant prompts are used in downstream processes, preventing prompt explosion as facilities expand their capability inventories.

**3. Orchestration**
   :doc:`Coordinates capability execution <../04_infrastructure-components/04_orchestrator-planning>` via plan-first (complete upfront plans with explicit dependencies) or reactive (ReAct, step-by-step decisions). Both modes provide human oversight, operator review of proposed operations, and capability hallucination prevention.

**4. Execution**
   Executes capabilities with checkpointing, artifact management, and comprehensive safety enforcement. Pattern detection and static analysis identify hardware writes, PV boundary checking verifies setpoints against facility-defined limits, and approval workflows ensure operator oversight before any control system interaction. Plans execute step-by-step with LangGraph interrupts for human approval and containerized isolation for generated code.

Framework Functions
-------------------

.. tab-set::

   .. tab-item:: Task Processing

      Converts conversational input into structured, actionable tasks:

      .. code-block:: python

         # Chat history becomes focused task
         current_task = await _extract_task(
             messages=state["messages"],
             retrieval_result=data_manager.retrieve_all_context(request),
         )


      - **Context Compression**: Refines lengthy conversations into precise, actionable tasks
      - **Datasource Integration**: Enhances tasks with structured data from external sources
      - **Self-Contained Output**: Produces tasks that are executable without relying on prior conversation history

   .. tab-item:: Capability Selection

      Task classification determines which capabilities are relevant:

      .. code-block:: python

         # Each capability gets yes/no decision
         active_capabilities = await classify_task(
             task=state.current_task,
             available_capabilities=registry.get_all_capabilities()
         )

      - **Binary Decisions**: Yes/no for each capability
      - **Prompt Efficiency**: Only relevant capabilities loaded
      - **Parallel Processing**: Independent classification decisions

   .. tab-item:: Capability Orchestration

      Coordinates capability execution via two modes:

      .. code-block:: yaml

         # config.yml
         execution_control:
           agent_control:
             orchestration_mode: plan_first  # or: react

      - **Plan-First** (default): Complete plan created upfront, then executed step-by-step
      - **Reactive (ReAct)**: Decides one step at a time, observing results between steps
      - **Capability Validation**: Prevents hallucination in both modes

   .. tab-item:: State Management

      Manages conversation context and execution state:

      .. code-block:: python

         # Persistent context across conversations
         StateManager.store_context(
             state, "PV_ADDRESSES", context_key, pv_data
         )

      - **Conversation Persistence**: Context survives restarts
      - **Execution Tracking**: Current step and progress
      - **Context Isolation**: Capability-specific data storage

   .. tab-item:: Approval Workflows

      Human oversight through LangGraph interrupts:

      .. code-block:: python

         # Request human approval
         if requires_approval:
             interrupt(approval_data)
             # Execution pauses for human decision

      - **Planning Approval**: Review execution plans before running
      - **Code Approval**: Human review of generated code
      - **Native Integration**: Built on LangGraph interrupts


.. dropdown:: 🚀 Next Steps

   Now that you understand the framework's core concepts, explore the architectural principles that make it production-ready and scalable:

   .. grid:: 1 1 2 2
      :gutter: 3
      :class-container: guides-section-grid

      .. grid-item-card:: 🏗️ Infrastructure Architecture
         :link: 01_infrastructure-architecture
         :link-type: doc
         :class-header: bg-primary text-white
         :class-body: text-center
         :shadow: md

         Gateway-driven pipeline, component coordination, and the three-pillar processing architecture

      .. grid-item-card:: 🔧 Convention over Configuration
         :link: 02_convention-over-configuration
         :link-type: doc
         :class-header: bg-success text-white
         :class-body: text-center
         :shadow: md

         Configuration-driven component loading, decorator-based registration, and eliminating boilerplate

   .. grid:: 1 1 2 2
      :gutter: 3
      :class-container: guides-section-grid

      .. grid-item-card:: 🔗 LangGraph Integration
         :link: 03_langgraph-integration
         :link-type: doc
         :class-header: bg-info text-white
         :class-body: text-center
         :shadow: md

         StateGraph workflows, native checkpointing, interrupts, and streaming support

      .. grid-item-card:: 🎯 Orchestration Architecture
         :link: 04_orchestration-architecture
         :link-type: doc
         :class-header: bg-warning text-white
         :class-body: text-center
         :shadow: md

         Plan-first and reactive orchestration modes, routing, and approval workflows
