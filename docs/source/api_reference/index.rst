=============
API Reference
=============

**Complete technical reference** for all public APIs in the Osprey Framework. This is your authoritative source for classes, methods, functions, and their precise interfaces. See also the :doc:`../developer-guides/index` for learning-oriented content, then return here when you need specific implementation details.

.. dropdown:: 🎯 API Learning Paths & Quick Navigation
   :color: primary

   .. tab-set::

      .. tab-item:: Getting Started
         :class-content: getting-started-tab

         **Essential API Learning Path**

         1. :doc:`01_core_framework/01_base_components` - BaseCapability and decorators
         2. :doc:`01_core_framework/02_state_and_context` - AgentState and ContextManager
         3. :doc:`02_infrastructure/01_gateway` - Entry point architecture
         4. :doc:`03_production_systems/01_human-approval` - Safety and oversight
         5. :doc:`04_error_handling/01_classification_system` - Robust error handling

         **Core API Patterns:**

         * :class:`osprey.base.BaseCapability` with :func:`@capability_node <osprey.base.capability_node>` decorator
         * :class:`osprey.state.AgentState` for LangGraph-native state management
         * :class:`osprey.context.ContextManager` for type-safe data exchange
         * :class:`osprey.registry.RegistryManager` for component discovery

      .. tab-item:: Production Integration
         :class-content: advanced-tab

         **Advanced API Workflows**

         * :doc:`02_infrastructure/04_orchestration` - ExecutionPlan creation and coordination
         * :doc:`03_production_systems/03_python-execution` - Secure code execution APIs
         * :doc:`03_production_systems/02_data-management` - Unified data source integration
         * :doc:`03_production_systems/05_container-management` - Deployment and service management
         * :doc:`04_error_handling/03_recovery_coordination` - Recovery strategies and coordination

         **Production-Ready APIs:**

        * :class:`osprey.approval.ApprovalManager` - LangGraph-native approval workflows
        * :class:`osprey.data_management.DataSourceManager` - Provider discovery and retrieval
        * :mod:`osprey.deployment.container_manager` - Service orchestration and deployment
        * :func:`osprey.models.get_chat_completion` - Multi-provider LLM management
        * :doc:`01_core_framework/04_configuration_system` - Environment and settings management
        * :doc:`05_framework_utilities/index` - Logging, streaming, and observability

      .. tab-item:: Quick API Solutions
         :class-content: solutions-tab

         **Implementation Shortcuts**

         **Build a new capability** → :class:`~osprey.base.BaseCapability` + :func:`~osprey.base.decorators.capability_node`

         **Add approval workflows** → :class:`~osprey.approval.ApprovalManager` + :doc:`03_production_systems/01_human-approval`

         **Execute Python code safely** → :class:`~osprey.services.python_executor.PythonExecutorService`

         **Store user context** → :class:`~osprey.context.ContextManager` + :doc:`01_core_framework/02_state_and_context`

        **Deploy with containers** → :mod:`~osprey.deployment.container_manager` + :doc:`03_production_systems/05_container-management`

        **Handle complex data flows** → :class:`~osprey.data_management.DataSourceManager`

        **Integrate external data** → :class:`~osprey.data_management.DataSourceProvider` + provider patterns

        **Manage LLM models** → :func:`~osprey.models.get_chat_completion`

        **Configure logging and streaming** → :func:`~osprey.utils.logger.get_logger` + :class:`~osprey.utils.logger.ComponentLogger`

      .. tab-item:: By System Component
         :class-content: component-tab

         **Core Framework APIs**

         * :doc:`01_core_framework/01_base_components` - BaseCapability, BaseInfrastructureNode, decorators
         * :doc:`01_core_framework/02_state_and_context` - AgentState, StateManager, ContextManager
         * :doc:`01_core_framework/03_registry_system` - RegistryManager, component discovery
         * :doc:`01_core_framework/04_configuration_system` - Environment and settings management
         * :doc:`01_core_framework/05_prompt_management` - Framework prompt providers and builders

         **Infrastructure Pipeline APIs**

         * :doc:`02_infrastructure/01_gateway` - Gateway class, message processing, state lifecycle
         * :doc:`02_infrastructure/02_task-extraction` - TaskExtractionNode, conversation analysis
         * :doc:`02_infrastructure/03_classification` - ClassificationNode, capability selection
         * :doc:`02_infrastructure/04_orchestration` - OrchestrationNode, ExecutionPlan creation
         * :doc:`02_infrastructure/05_execution-control` - Router coordination and flow management
         * :doc:`02_infrastructure/06_message-generation` - Response generation and clarification

         **Production & Error Management**

         * :doc:`03_production_systems/01_human-approval` - ApprovalManager, approval workflows
         * :doc:`03_production_systems/02_data-management` - DataSourceManager, provider framework
         * :doc:`03_production_systems/03_python-execution` - PythonExecutorService, secure execution
         * :doc:`03_production_systems/04_memory-storage` - Memory management and persistence
         * :doc:`03_production_systems/05_container-management` - Container orchestration and deployment
         * :doc:`04_error_handling/01_classification_system` - Error classification and severity levels
         * :doc:`04_error_handling/02_exception_reference` - Exception hierarchy and handling
         * :doc:`04_error_handling/03_recovery_coordination` - Recovery strategies and coordination
         * :doc:`05_framework_utilities/index` - Logging, streaming, and developer tools


Core API Sections
==================

.. grid:: 1 2 2 3
   :gutter: 3
   :class-container: api-section-grid

   .. grid-item-card:: 🏗️ Core Framework
      :link: 01_core_framework/index
      :link-type: doc
      :class-header: bg-primary text-white
      :class-body: text-center
      :shadow: md

      **Essential daily-use APIs**

      State management, context handling, base components, registry system, and configuration APIs that developers use most frequently.

   .. grid-item-card:: ⚡ Infrastructure
      :link: 02_infrastructure/index
      :link-type: doc
      :class-header: bg-success text-white
      :class-body: text-center
      :shadow: md

      **Message processing pipeline**

      Gateway, task extraction, classification, orchestration, and execution control for building intelligent workflows.

   .. grid-item-card:: 🚀 Production Systems
      :link: 03_production_systems/index
      :link-type: doc
      :class-header: bg-info text-white
      :class-body: text-center
      :shadow: md

      **Production-ready services**

      Container management, data management, human approval systems, and deployment tools for building robust applications.

   .. grid-item-card:: 🛡️ Error Handling
      :link: 04_error_handling/index
      :link-type: doc
      :class-header: bg-danger text-white
      :class-body: text-center
      :shadow: md
      :columns: 6

      **Resilient error management**

      Exception hierarchy, recovery strategies, and error classification for building fault-tolerant systems.

   .. grid-item-card:: 🔧 Framework Utilities
      :link: 05_framework_utilities/index
      :link-type: doc
      :class-header: bg-warning text-white
      :class-body: text-center
      :shadow: md
      :columns: 6

      **Advanced utilities**

      Model factory, logging, streaming, and developer tools for extending and customizing framework behavior.





.. dropdown:: 📚 Complete API Documentation Tree

   **🏗️ Core Framework**

   * :doc:`01_core_framework/01_base_components` - Foundation classes, decorators, and planning
   * :doc:`01_core_framework/02_state_and_context` - LangGraph state management and tracking
   * :doc:`01_core_framework/03_registry_system` - Component discovery and registration
   * :doc:`01_core_framework/04_configuration_system` - Configuration management
   * :doc:`01_core_framework/05_prompt_management` - Framework prompt providers and builders

   **⚡ Infrastructure**

   * :doc:`02_infrastructure/01_gateway` - Main entry point and request routing
   * :doc:`02_infrastructure/02_task-extraction` - User input analysis and task identification
   * :doc:`02_infrastructure/03_classification` - Capability selection and routing logic
   * :doc:`02_infrastructure/04_orchestration` - Execution planning and coordination
   * :doc:`02_infrastructure/06_message-generation` - Response formatting and user communication
   * :doc:`02_infrastructure/05_execution-control` - Execution flow and state management

   **🚀 Production Systems**

   * :doc:`03_production_systems/01_human-approval` - LangGraph-native approval workflows
   * :doc:`03_production_systems/02_data-management` - Unified data source integration
   * :doc:`03_production_systems/03_python-execution` - Secure code generation and execution
   * :doc:`03_production_systems/04_memory-storage` - Persistent user memory management
   * :doc:`03_production_systems/05_container-management` - Service orchestration and deployment

   **🛡️ Error Handling**

   * :doc:`04_error_handling/01_classification_system` - Structured exception classification
   * :doc:`04_error_handling/02_exception_reference` - Exception hierarchy and reference
   * :doc:`04_error_handling/03_recovery_coordination` - Intelligent error recovery and retry

   **🔧 Framework Utilities**

   * :doc:`05_framework_utilities/index` - Model factory, logging and streaming utilities

.. toctree::
   :maxdepth: 2
   :hidden:

   01_core_framework/index
   02_infrastructure/index
   03_production_systems/index
   04_error_handling/index
   05_framework_utilities/index