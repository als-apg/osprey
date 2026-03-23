Infrastructure Components
=========================

.. toctree::
   :maxdepth: 2
   :caption: Infrastructure Components
   :hidden:

   01_gateway-architecture
   02_task-extraction-system
   03_classification-and-routing
   04_orchestrator-planning
   05_message-generation
   06_error-handling-infrastructure

.. dropdown:: What You'll Learn
   :color: primary
   :icon: book

   **Gateway-First Processing Architecture:**

   - Gateway-driven message processing with universal entry point patterns
   - Task extraction system converting conversations into actionable requirements
   - LLM-powered classification and intelligent routing for capability selection
   - Plan-first and reactive (ReAct) orchestration modes
   - Adaptive response generation and sophisticated error handling with recovery

   **Prerequisites:** Understanding of Core Framework Systems and LangGraph StateGraph concepts

   **Target Audience:** Infrastructure developers and agentic system architects building reliable, controllable execution pipelines

The Infrastructure Components provide the intelligent processing core that makes sophisticated agentic behavior possible while maintaining the reliability and oversight required for production systems. Master these components, and you'll understand how to build agents that combine LLM intelligence with predictable, controllable execution.

Architecture Overview
=====================

The Osprey Framework implements a **Gateway-First, Three-Pillar Architecture** with structured orchestration that replaces ad-hoc tool calling with coordinated, auditable execution:

**Traditional Approach:**

.. code-block:: text

   User Query → Tool Call 1 → Analyze → Tool Call 2 → Analyze → Tool Call 3 → Response

**Plan-First Approach (default):**

.. code-block:: text

   User Query → Complete Plan Creation → Execute All Steps → Response

**Reactive Approach (ReAct):**

.. code-block:: text

   User Query → Decide Step → Execute → Observe → Decide Next Step → ... → Response

Both modes provide full context utilization, natural human oversight, and capability validation. Plan-first minimizes LLM calls; reactive adapts dynamically to intermediate results.

The Three Pillars
==================

.. grid:: 1 1 3 3
   :gutter: 3

   .. grid-item-card:: 🧠 Task Extraction
      :link: 02_task-extraction-system
      :link-type: doc
      :class-header: bg-primary text-white
      :class-body: text-center
      :shadow: md

      **Conversational Context Compression**

      Converts chat history into structured, actionable tasks with resolved references and context.

   .. grid-item-card:: 🎯 Classification & Routing
      :link: 03_classification-and-routing
      :link-type: doc
      :class-header: bg-success text-white
      :class-body: text-center
      :shadow: md

      **Intelligent Capability Selection**

      LLM-powered analysis with few-shot examples to select appropriate capabilities for tasks.

   .. grid-item-card:: 🎼 Orchestrator Planning
      :link: 04_orchestrator-planning
      :link-type: doc
      :class-header: bg-info text-white
      :class-body: text-center
      :shadow: md

      **Complete Execution Coordination**

      Plan-first and reactive orchestration with capability validation and approval integration.

Supporting Infrastructure
=========================

.. grid:: 1 1 3 3
   :gutter: 3

   .. grid-item-card:: 🚪 Gateway Architecture
      :link: 01_gateway-architecture
      :link-type: doc
      :class-header: bg-warning text-white
      :class-body: text-center
      :shadow: md

      **Universal Entry Point**

      Single message processing interface with state management and approval integration.

   .. grid-item-card:: 💬 Message Generation
      :link: 05_message-generation
      :link-type: doc
      :class-header: bg-secondary text-white
      :class-body: text-center
      :shadow: md

      **Adaptive Response System**

      Context-aware response generation with clarification workflows.

   .. grid-item-card:: 🔧 Error Handling
      :link: 06_error-handling-infrastructure
      :link-type: doc
      :class-header: bg-danger text-white
      :class-body: text-center
      :shadow: md

      **AI-Powered Recovery**

      Intelligent error classification with LLM-generated user explanations.

.. dropdown:: 🚀 Next Steps

   Now that you understand the infrastructure architecture, explore the processing pipeline:

   .. grid:: 1 1 2 2
      :gutter: 3
      :class-container: guides-section-grid

      .. grid-item-card:: 🚪 Start with Gateway
         :link: 01_gateway-architecture
         :link-type: doc
         :class-header: bg-primary text-white
         :class-body: text-center
         :shadow: md

         Universal entry point for all message processing with state management and approval integration

      .. grid-item-card:: 🧠 Follow the Pipeline
         :link: 02_task-extraction-system
         :link-type: doc
         :class-header: bg-success text-white
         :class-body: text-center
         :shadow: md

         Task extraction, classification, orchestration - the three-pillar processing flow

   .. grid:: 1 1 2 2
      :gutter: 3
      :class-container: guides-section-grid

      .. grid-item-card:: 🔧 Build Resilience
         :link: 06_error-handling-infrastructure
         :link-type: doc
         :class-header: bg-warning text-white
         :class-body: text-center
         :shadow: md

         AI-powered error recovery with intelligent retry policies and user communication

      .. grid-item-card:: 💬 Master Communication
         :link: 05_message-generation
         :link-type: doc
         :class-header: bg-info text-white
         :class-body: text-center
         :shadow: md

         Adaptive response generation with clarification workflows and domain customization
