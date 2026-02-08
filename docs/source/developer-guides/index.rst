================
Developer Guides
================

**Comprehensive learning-oriented guides** for deploying agentic AI in large-scale control system environments. These guides combine architectural understanding with practical implementation patterns, taking you from basic concepts to production control room deployment.

.. dropdown:: 🎯 Learning Paths & Quick Navigation

   .. tab-set::
      :class: natural-width

      .. tab-item:: New to the Framework
         :class-content: getting-started-tab

         **Essential Learning Path**

         1. :doc:`01_understanding-the-framework/index` - Core concepts and architecture
         2. :doc:`02_quick-start-patterns/01_building-your-first-capability` - Build your first capability
         3. :doc:`03_core-framework-systems/01_state-management-architecture` - Master state patterns
         4. :doc:`04_infrastructure-components/01_gateway-architecture` - Understand message processing
         5. :doc:`05_production-systems/01_human-approval-workflows` - Deploy with safety

         **Key Architecture Concepts:**

         * :doc:`01_understanding-the-framework/04_orchestrator-first-philosophy` - Why upfront planning works
         * :doc:`01_understanding-the-framework/02_convention-over-configuration` - Zero-boilerplate development
         * :doc:`01_understanding-the-framework/03_langgraph-integration` - StateGraph workflows

      .. tab-item:: Experienced Developers
         :class-content: advanced-tab

         **Fast Track to Productivity**

         * :doc:`02_quick-start-patterns/01_building-your-first-capability` - @capability_node patterns
         * :doc:`03_core-framework-systems/05_message-and-execution-flow` - Router-controlled architecture
         * :doc:`04_infrastructure-components/04_orchestrator-planning` - LLM-powered execution coordination
         * :doc:`05_production-systems/03_python-execution-service/index` - Pluggable code generation & execution

         **Advanced Integration Patterns:**

         * :doc:`03_core-framework-systems/02_context-management-system` - Pydantic context objects
         * :doc:`03_core-framework-systems/03_registry-and-discovery` - Convention-based loading
         * :doc:`05_production-systems/02_data-source-integration` - Provider framework patterns

      .. tab-item:: Development Workflows
         :class-content: workflow-tab

         **Development & Testing**

         * :doc:`02_quick-start-patterns/01_building-your-first-capability` - Complete capability implementation
         * :doc:`02_quick-start-patterns/02_state-and-context-essentials` - StateManager & ContextManager patterns
         * :doc:`02_quick-start-patterns/03_running-and-testing` - CLI interface & debugging workflows

         **State & Context Management**

         * :doc:`03_core-framework-systems/01_state-management-architecture` - Selective persistence strategy
         * :doc:`03_core-framework-systems/02_context-management-system` - Type-safe data exchange
         * :doc:`05_production-systems/04_memory-storage-service` - Cross-session context preservation

         **Error Handling & Recovery**

         * :doc:`04_infrastructure-components/06_error-handling-infrastructure` - LLM-powered recovery
         * :doc:`04_infrastructure-components/05_message-generation` - Clarification workflows

      .. tab-item:: Production Workflows
         :class-content: production-tab

         **Security & Approval Systems**

         * :doc:`05_production-systems/01_human-approval-workflows` - LangGraph-native interrupts
         * :doc:`05_production-systems/03_python-execution-service/index` - Code generation, analysis & secure execution
         * :doc:`04_infrastructure-components/01_gateway-architecture` - Universal entry point security

         **Deployment & Integration**

         * :doc:`05_production-systems/05_container-and-deployment` - Service orchestration & health monitoring
         * :doc:`05_production-systems/02_data-source-integration` - Provider framework & parallel retrieval
         * :doc:`05_production-systems/04_memory-storage-service` - File-based storage with framework integration

         **Processing Pipeline**

         * :doc:`04_infrastructure-components/02_task-extraction-system` - Conversational context compression
         * :doc:`04_infrastructure-components/03_classification-and-routing` - Intelligent capability selection
         * :doc:`04_infrastructure-components/04_orchestrator-planning` - Complete execution coordination

      .. tab-item:: Quick Solutions
         :class-content: solutions-tab

         **Implementation Shortcuts**

         **Build a new capability** → :doc:`02_quick-start-patterns/01_building-your-first-capability`

         **Add approval workflows** → :doc:`05_production-systems/01_human-approval-workflows`

         **Enable Python execution** → :doc:`05_production-systems/03_python-execution-service/index`

         **Store user context** → :doc:`03_core-framework-systems/02_context-management-system`

         **Deploy with containers** → :doc:`05_production-systems/05_container-and-deployment`

         **Handle complex data flows** → :doc:`03_core-framework-systems/05_message-and-execution-flow`

         **Integrate external data** → :doc:`05_production-systems/02_data-source-integration`

         **Register custom model provider** → :doc:`03_core-framework-systems/03_registry-and-discovery` (AI Provider Registration)

         **Understand the architecture** → :doc:`01_understanding-the-framework/01_infrastructure-architecture`

         **Debug execution flow** → :doc:`03_core-framework-systems/05_message-and-execution-flow`

         **Customize framework behavior** → :doc:`01_understanding-the-framework/02_convention-over-configuration`

      .. tab-item:: By System Component
         :class-content: component-tab

         **Gateway & Message Processing**

         * :doc:`04_infrastructure-components/01_gateway-architecture` - Universal entry point
         * :doc:`04_infrastructure-components/02_task-extraction-system` - Chat history compression
         * :doc:`04_infrastructure-components/05_message-generation` - Adaptive response system

         **Classification & Orchestration**

         * :doc:`04_infrastructure-components/03_classification-and-routing` - Capability selection logic
         * :doc:`04_infrastructure-components/04_orchestrator-planning` - Execution plan creation
         * :doc:`01_understanding-the-framework/04_orchestrator-first-philosophy` - Planning vs reactive patterns

         **State & Registry Systems**

         * :doc:`03_core-framework-systems/01_state-management-architecture` - AgentState lifecycle
         * :doc:`03_core-framework-systems/03_registry-and-discovery` - Component discovery & lazy loading
         * :doc:`01_understanding-the-framework/02_convention-over-configuration` - Decorator-based registration


Guide Categories
================

.. grid:: 1 1 2 2
   :gutter: 3
   :class-container: guides-section-grid

   .. grid-item-card:: 🏛️ Understanding the Framework
      :link: 01_understanding-the-framework/index
      :link-type: doc
      :class-header: bg-primary text-white
      :class-body: text-center
      :shadow: md

      Core concepts, design principles, and the orchestrator-first philosophy that makes the framework powerful and reliable.

   .. grid-item-card:: 🚀 Quick Start Patterns
      :link: 02_quick-start-patterns/index
      :link-type: doc
      :class-header: bg-success text-white
      :class-body: text-center
      :shadow: md

      Master capability development, state management, and testing workflows. Get productive immediately with convention-based patterns.


   .. grid-item-card:: ⚙️ Core Framework Systems
      :link: 03_core-framework-systems/index
      :link-type: doc
      :class-header: bg-info text-white
      :class-body: text-center
      :shadow: md

      State management, context systems, registry patterns, and execution flow. Master the internals for sophisticated applications.

   .. grid-item-card:: 🔧 Infrastructure Components
      :link: 04_infrastructure-components/index
      :link-type: doc
      :class-header: bg-warning text-white
      :class-body: text-center
      :shadow: md

      Gateway architecture, task extraction, classification, orchestration, and message generation for intelligent agent behavior.

.. grid:: 1 1 1 1
   :gutter: 3
   :class-container: guides-section-grid

   .. grid-item-card:: 🏭 Production Systems
      :link: 05_production-systems/index
      :link-type: doc
      :class-header: bg-danger text-white
      :class-body: text-center
      :shadow: md

      Human approval workflows, data integration, secure execution, memory storage, and container orchestration for production systems.

.. dropdown:: 📚 Complete Guide Structure

   **🏛️ Understanding the Framework**

   * :doc:`01_understanding-the-framework/01_infrastructure-architecture` - Gateway, three-pillar pipeline, and system integration
   * :doc:`01_understanding-the-framework/02_convention-over-configuration` - Configuration-driven loading, registry patterns, and reduced boilerplate
   * :doc:`01_understanding-the-framework/03_langgraph-integration` - StateGraph, interrupts, checkpoints, and native features
   * :doc:`01_understanding-the-framework/04_orchestrator-first-philosophy` - Upfront planning vs. iterative tool calling

   **🚀 Quick Start Patterns**

   * :doc:`02_quick-start-patterns/00_cli-reference` - Framework CLI commands for project creation and deployment
   * :doc:`02_quick-start-patterns/01_building-your-first-capability` - BaseCapability, decorators, and framework integration
   * :doc:`02_quick-start-patterns/02_state-and-context-essentials` - AgentState, ContextManager, and data sharing patterns
   * :doc:`02_quick-start-patterns/03_running-and-testing` - Gateway testing, CLI workflows, and debugging techniques
   * :doc:`02_quick-start-patterns/04_mcp-capability-generation` - Generate capabilities from MCP servers (prototype)

   **⚙️ Core Framework Systems**

   * :doc:`03_core-framework-systems/01_state-management-architecture` - LangGraph-native state with selective persistence
   * :doc:`03_core-framework-systems/02_context-management-system` - Type-safe data sharing between capabilities
   * :doc:`03_core-framework-systems/03_registry-and-discovery` - Component registration and convention-based loading mechanisms
   * :doc:`03_core-framework-systems/05_message-and-execution-flow` - Complete pipeline from input to response
   * :doc:`03_core-framework-systems/07_built-in-capabilities` - Reference for all framework-native capabilities

   **🔧 Infrastructure Components**

   * :doc:`04_infrastructure-components/01_gateway-architecture` - Single entry point with state and approval integration
   * :doc:`04_infrastructure-components/02_task-extraction-system` - Natural language to structured task conversion
   * :doc:`04_infrastructure-components/03_classification-and-routing` - Capability selection and intelligent flow control
   * :doc:`04_infrastructure-components/04_orchestrator-planning` - LLM-powered execution plan creation
   * :doc:`04_infrastructure-components/05_message-generation` - Adaptive response formatting and clarification
   * :doc:`04_infrastructure-components/06_error-handling-infrastructure` - AI-powered error recovery and communication

   **🏭 Production Systems**

   * :doc:`05_production-systems/01_human-approval-workflows` - LangGraph-native approval with rich context
   * :doc:`05_production-systems/02_data-source-integration` - Parallel retrieval and intelligent provider discovery
   * :doc:`05_production-systems/03_python-execution-service/index` - Pluggable code generation and secure execution
   * :doc:`05_production-systems/04_memory-storage-service` - Persistent user memory with framework integration
   * :doc:`05_production-systems/05_container-and-deployment` - Service orchestration and template-based deployment

.. toctree::
   :maxdepth: 2
   :hidden:

   01_understanding-the-framework/index
   02_quick-start-patterns/index
   03_core-framework-systems/index
   04_infrastructure-components/index
   05_production-systems/index