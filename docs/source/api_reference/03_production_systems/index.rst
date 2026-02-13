==================
Production Systems
==================

.. toctree::
   :maxdepth: 2
   :caption: Production Systems
   :hidden:

   01_human-approval
   02_data-management
   03_python-execution
   04_memory-storage
   05_container-management
   06_control-system-connectors
   07_ariel-search


.. dropdown:: What You'll Find Here
   :color: primary
   :icon: book

   **Production-ready services for secure, scalable agentic deployments:**

   - **Human Approval System** - LangGraph-native approval workflows with rich context, security analysis, and resumable execution
   - **Data Management Framework** - Unified data source integration with provider discovery, concurrent retrieval, and LLM-optimized formatting
   - **Python Execution Service** - Container-isolated code generation and execution with approval integration and flexible deployment modes
   - **Memory Storage System** - Persistent user memory with structured operations, data source integration, and approval workflows
   - **Container Management System** - Podman-based service orchestration with hierarchical discovery and Jinja2 template processing
   - **ARIEL Search Service** - Logbook search with pluggable search modules, execution pipelines, ingestion adapters, and enhancement modules

   **Prerequisites:** Understanding of production deployment patterns and security-first development

   **Target Audience:** DevOps engineers, security architects, production system implementers

Enterprise-grade services that transform research prototypes into production-ready agentic systems. These components provide the security, reliability, and scalability required for high-stakes scientific and industrial environments.

.. currentmodule:: osprey


The Seven Pillars
=================

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: 🛡️ Human Approval System
      :link: 01_human-approval
      :link-type: doc
      :class-header: bg-primary text-white
      :class-body: text-center
      :shadow: md

      **LangGraph-Native Oversight**

      Production-ready approval workflows with rich context, security analysis, and seamless resumption.

   .. grid-item-card:: 🔄 Data Management Framework
      :link: 02_data-management
      :link-type: doc
      :class-header: bg-success text-white
      :class-body: text-center
      :shadow: md

      **Unified Data Orchestration**

      Heterogeneous data source integration with provider discovery and concurrent retrieval.

   .. grid-item-card:: 🐍 Python Execution Service
      :link: 03_python-execution
      :link-type: doc
      :class-header: bg-info text-white
      :class-body: text-center
      :shadow: md

      **Secure Code Execution**

      Container-isolated Python execution with approval integration and flexible deployment.

   .. grid-item-card:: 🧠 Memory Storage System
      :link: 04_memory-storage
      :link-type: doc
      :class-header: bg-warning text-white
      :class-body: text-center
      :shadow: md

      **Persistent User Memory**

      Structured memory operations with data source integration and approval workflows.

   .. grid-item-card:: 🚢 Container Management
      :link: 05_container-management
      :link-type: doc
      :class-header: bg-secondary text-white
      :class-body: text-center
      :shadow: md

      **Service Orchestration**

      Podman-based deployment with hierarchical service discovery and template processing.

   .. grid-item-card:: 🎛️ Control System Connectors
      :link: 06_control-system-connectors
      :link-type: doc
      :class-header: bg-dark text-white
      :class-body: text-center
      :shadow: md

      **Hardware Abstraction Layer**

      Pluggable connectors for control systems and archivers with mock and production implementations.

   .. grid-item-card:: 🔍 ARIEL Search Service
      :link: 07_ariel-search
      :link-type: doc
      :class-header: bg-primary text-white
      :class-body: text-center
      :shadow: md

      **Logbook Search & Retrieval**

      Pluggable search modules, execution pipelines, ingestion adapters, and enhancement modules.

Production Integration
======================

These systems work together to provide comprehensive production capabilities:

.. tab-set::

   .. tab-item:: Security Flow

      How safety and oversight are maintained:

      .. code-block:: python

         # Approval system integration
         from osprey.approval import get_approval_manager
         approval_manager = get_approval_manager()

         # Secure execution with oversight
         from osprey.services.python_executor import PythonExecutorService, PythonExecutionRequest
         python_service = PythonExecutorService()
         request = PythonExecutionRequest(
             user_query="Analyze beam performance data",
             task_objective="Generate comprehensive performance report"
         )

         # Service automatically pauses for human review when requires_approval: true
         config = {"thread_id": "session_123"}
         result = await python_service.ainvoke(request, config)
         # Execution resumes after approval with audit trail

   .. tab-item:: Data Architecture

      Unified data access across systems:

      .. code-block:: python

         # Data source integration
         from osprey.data_management import (
             get_data_source_manager,
             create_data_source_request,
             DataSourceRequester
         )
         data_manager = get_data_source_manager()
         request = create_data_source_request(
             state,
             requester=DataSourceRequester(
                 capability_name="performance_analysis",
                 component_name="beam_analysis"
             ),
             query="beam current trends"
         )

         # Concurrent retrieval from all providers
         result = await data_manager.retrieve_all_context(request)

         # Memory integration
         from osprey.services.memory_storage import get_memory_storage_manager, MemoryContent
         from datetime import datetime
         memory_manager = get_memory_storage_manager()
         memory_entry = MemoryContent(
             timestamp=datetime.now(),
             content=f"Analysis results: {result.context_data}"
         )
         success = memory_manager.add_memory_entry(user_id, memory_entry)

   .. tab-item:: Deployment Strategy

      Container orchestration for scalability:

      .. code-block:: yaml

         # Container deployment configuration
         deployed_services:
           - jupyter              # Secure execution environment
           - pipelines            # Processing pipeline infrastructure
           - pv_finder            # Application data service (e.g., ALS Assistant)

         execution:
           execution_method: "container"   # Isolation by default
           modes:
             write_access:
               requires_approval: true    # Safety first
               allows_writes: true
               kernel_name: "python3-epics-write"

.. dropdown:: 🚀 Next Steps

   Now that you understand the production systems architecture, explore deployment and integration:

   .. grid:: 1 1 2 2
      :gutter: 3
      :class-container: guides-section-grid

      .. grid-item-card:: 🛡️ Start with Security
         :link: 01_human-approval
         :link-type: doc
         :class-header: bg-primary text-white
         :class-body: text-center
         :shadow: md

         Human approval workflows with LangGraph-native interrupts and rich approval context

      .. grid-item-card:: 🐍 Secure Execution
         :link: 03_python-execution
         :link-type: doc
         :class-header: bg-info text-white
         :class-body: text-center
         :shadow: md

         Container-isolated Python execution with approval integration and audit trails

   .. grid:: 1 1 2 2
      :gutter: 3
      :class-container: guides-section-grid

      .. grid-item-card:: 🔄 Data Integration
         :link: 02_data-management
         :link-type: doc
         :class-header: bg-success text-white
         :class-body: text-center
         :shadow: md

         Unified data source management with provider discovery and concurrent retrieval

      .. grid-item-card:: 🚢 Deploy at Scale
         :link: 05_container-management
         :link-type: doc
         :class-header: bg-secondary text-white
         :class-body: text-center
         :shadow: md

         Container orchestration with hierarchical service discovery and template processing

      .. grid-item-card:: 🔍 ARIEL Search
         :link: 07_ariel-search
         :link-type: doc
         :class-header: bg-primary text-white
         :class-body: text-center
         :shadow: md

         Logbook search service with pluggable modules, pipelines, and ingestion adapters
