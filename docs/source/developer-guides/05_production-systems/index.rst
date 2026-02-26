==================
Production Systems
==================

.. toctree::
   :maxdepth: 2
   :caption: Production Systems
   :hidden:

   01_human-approval-workflows
   02_data-source-integration
   03_python-execution-service/index
   04_memory-storage-service
   05_container-and-deployment
   06_control-system-integration
   07_logbook-search-service/index

.. dropdown:: What You'll Learn
   :color: primary
   :icon: book

   **Enterprise-Grade Production Architecture:**

   - LangGraph-native approval workflows with configurable security policies
   - Multi-source data integration through provider framework patterns
   - Container-isolated Python execution with security analysis and EPICS integration
   - Persistent memory storage with cross-session context preservation
   - Complete container management and service orchestration for scalable deployment

   **Prerequisites:** Solid understanding of Infrastructure Components and production deployment concepts

   **Target Audience:** DevOps engineers, system administrators, and architects deploying agentic systems in production environments

The Osprey Framework offers enterprise-grade infrastructure components designed for secure and scalable deployment of agentic systems. These production-ready systems ensure human oversight, data integration, secure execution, and orchestration capabilities essential for high-stakes environments. By implementing a Security-First, Approval-Centric Architecture, the framework delivers robust capabilities while maintaining the flexibility needed for diverse deployment scenarios.

Core Production Components
==========================

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: 🛡️ Human Approval Workflows
      :link: 01_human-approval-workflows
      :link-type: doc
      :class-header: bg-danger text-white
      :class-body: text-center
      :shadow: md

      LangGraph-native interrupts with configurable policies, rich context, and fail-secure defaults for production environments.

   .. grid-item-card:: 🔗 Data Source Integration
      :link: 02_data-source-integration
      :link-type: doc
      :class-header: bg-info text-white
      :class-body: text-center
      :shadow: md

      Data retrieval from multiple sources with provider framework and intelligent discovery mechanisms.

   .. grid-item-card:: 🐍 Python Execution Service
      :link: 03_python-execution-service/index
      :link-type: doc
      :class-header: bg-warning text-white
      :class-body: text-center
      :shadow: md

      Pluggable code generation (Basic LLM, Claude Code, Mock), security analysis, and flexible execution environments.

   .. grid-item-card:: 🧠 Memory Storage Service
      :link: 04_memory-storage-service
      :link-type: doc
      :class-header: bg-success text-white
      :class-body: text-center
      :shadow: md

      **Persistent User Memory**

      File-based storage with framework integration and cross-session context preservation.

   .. grid-item-card:: 🚀 Container & Deployment
      :link: 05_container-and-deployment
      :link-type: doc
      :class-header: bg-primary text-white
      :class-body: text-center
      :shadow: md

      Complete container management with template rendering and hierarchical service discovery.

   .. grid-item-card:: 🎛️ Control System Integration
      :link: 06_control-system-integration
      :link-type: doc
      :class-header: bg-secondary text-white
      :class-body: text-center
      :shadow: md

      Pluggable connectors for control systems (EPICS, LabVIEW, Tango, Mock) for development and production deployment.

   .. grid-item-card:: Logbook Search Service
      :link: 07_logbook-search-service/index
      :link-type: doc
      :class-header: bg-info text-white
      :class-body: text-center
      :shadow: md

      ARIEL search over electronic logbooks with keyword, semantic, and RAG-powered retrieval.

Production Integration Patterns
===============================

.. tab-set::

   .. tab-item:: Orchestrator Approval

      High-level execution plan approval with planning mode:

      .. code-block:: python

         # User enables planning mode
         user_input = "/planning Analyze beam performance and adjust parameters"

         # Agent processes input
         state_updates = await agent.ainvoke(
             {"user_query": user_input},
             config={"thread_id": "session_123"}
         )

         # Orchestrator automatically:
         # 1. Generates execution plan using LLM
         # 2. Validates capabilities exist
         # 3. Creates approval interrupt (planning mode enabled)
         # 4. Waits for user approval

         # After approval, orchestrator executes planned steps

      - **Automatic plan generation** using orchestrator LLM
      - **LangGraph-native interrupts** for plan approval
      - **Resumable workflow** after user approval/rejection
      - **File-based plan storage** for review and editing

   .. tab-item:: Python Execution

      Service-based Python execution with automatic approval handling:

      .. code-block:: python

         from osprey.registry import get_registry
         from osprey.services.python_executor import PythonExecutionRequest
         from osprey.approval import handle_service_with_interrupts

         # Get service from registry
         registry = get_registry()
         python_service = registry.get_service("python_executor")

         # Create execution request
         request = PythonExecutionRequest(
             user_query="Read EPICS beam current and create plot",
             task_objective="Analyze beam data",
             capability_prompts=["Use pyepics for PV access"],
             execution_folder_name="beam_analysis"
         )

         # Service handles generation, analysis, approval, execution
         result = await handle_service_with_interrupts(
             service=python_service,
             request=request,
             config=service_config,
             logger=logger,
             capability_name="BeamAnalysis"
         )

         # Access results
         data = result.execution_result.results

      - **Pluggable code generators** (Basic LLM, Claude Code, Mock)
      - **Automatic pattern detection** for control system reads/writes
      - **Configurable approval modes** (disabled, control_writes, all_code)
      - **Container or local execution** with seamless switching

   .. tab-item:: Data Integration

      Unified data access through the provider framework:

      .. code-block:: python

         # Parallel data retrieval pattern
         data_context = await data_manager.retrieve_all_context(
             DataSourceRequest(query=task.description)
         )

         # Available to all capabilities automatically
         user_memory = data_context.get("core_user_memory")
         domain_data = data_context.get("custom_provider")

      - **Automatic provider discovery** through registry system
      - **Parallel retrieval** with timeout management
      - **Type-safe integration** with capability context

   .. tab-item:: Service Orchestration

      Coordinated deployment and management of production services:

      .. code-block:: python

         # Container management using the function-based system
         from osprey.deployment.container_manager import find_service_config, setup_build_dir

         # Deploy services by configuring them in deployed_services list
         deployed_services = [
             "osprey.pipelines",
             "osprey.jupyter"
         ]

         # Services are deployed through container_manager.py script
         # python container_manager.py config.yml up -d

         # Service management through compose files
         for service_name in deployed_services:
             service_config, template_path = find_service_config(config, service_name)
             if service_config and template_path:
                 compose_file = setup_build_dir(template_path, config, service_config)

      - **Hierarchical service discovery** through osprey.* and applications.* naming
      - **Template-based configuration** for environment-specific deployments
      - **Podman Compose orchestration** with multi-file support

   .. tab-item:: Memory Integration

      Persistent user context with intelligent retrieval:

      .. code-block:: python

         # Memory-enhanced capability execution
         @capability_node
         class DataAnalysisCapability(BaseCapability):
             async def execute(state: AgentState, **kwargs):
                 # Retrieve user memory through data source integration
                 data_manager = get_data_source_manager()
                 requester = DataSourceRequester("capability", "data_analysis")
                 request = create_data_source_request(state, requester)
                 retrieval_result = await data_manager.retrieve_all_context(request)

                 # Access memory context from data sources
                 user_memory_context = retrieval_result.context_data.get("core_user_memory")
                 if user_memory_context:
                     user_memories = user_memory_context.data  # UserMemories object
                     # Use memory data to enhance analysis

      - **Data source integration** for automatic memory context injection
      - **Persistent memory storage** through UserMemoryProvider
      - **Framework-native memory operations** through MemoryOperationsCapability
