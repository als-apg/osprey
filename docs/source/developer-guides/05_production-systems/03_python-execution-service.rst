================
Python Execution
================

**What you'll build:** Python execution system with LangGraph workflows, human approval integration, and flexible container/local deployment

.. dropdown:: 📚 What You'll Learn
   :color: primary
   :icon: book

   **Key Concepts:**

   - Using registry-based :class:`PythonExecutorService` access for code execution
   - Creating :class:`PythonExecutionRequest` with structured ``capability_prompts``
   - Implementing approval handling with ``handle_service_with_interrupts()``
   - Managing approval resume/clear state with ``get_approval_resume_data()`` and ``clear_approval_state()``
   - Proper service configuration with ``thread_id`` and ``checkpoint_ns``
   - Container vs local execution through ``config.yml`` settings

   **Prerequisites:** Understanding of :doc:`01_human-approval-workflows` and :doc:`../03_core-framework-systems/05_message-and-execution-flow`

Overview
========

The Python Execution Service provides a LangGraph-based system for Python code generation, static analysis, human approval, and secure execution. It supports both containerized and local execution environments with seamless switching through configuration.

**Key Features:**

- **Flexible Execution Environments**: Switch between container and local execution with configuration
- **Jupyter Notebook Generation**: Automatic creation of interactive notebooks for evaluation
- **Human-in-the-Loop Approval**: LangGraph-native interrupts with rich context and safety assessments
- **Exception-Based Flow Control**: Clean error handling with categorized errors for retry strategies
- **Multi-Stage Pipeline**: Code generation → analysis → approval → execution → result processing

**Execution Pipeline:**

1. **Code Generation**: LLM-based Python code generation with context awareness
2. **Static Analysis**: Security and policy analysis with configurable rules
3. **Approval Workflows**: Human oversight system with rich context and safety assessments
4. **Flexible Execution**: Container or local execution with unified result collection
5. **Notebook Generation**: Comprehensive Jupyter notebook creation for evaluation
6. **Result Processing**: Structured result handling with artifact management

Configuration
=============

Configure your Python execution system with environment settings and approval policies:

.. code-block:: yaml

   # config.yml - Python Execution Configuration
   osprey:
     # Python Executor Service Configuration
     python_executor:
       max_generation_retries: 3      # Maximum retries for code generation failures
       max_execution_retries: 3       # Maximum retries for execution failures
       execution_timeout_seconds: 600 # Execution timeout in seconds (default: 10 minutes)

     execution:
       execution_method: "container"  # or "local"
       modes:
         read_only:
           kernel_name: "python3-epics-readonly"
           allows_writes: false
           requires_approval: false
         write_access:
           kernel_name: "python3-epics-write"
           allows_writes: true
           requires_approval: true

       # Container execution settings
       container:
         jupyter_host: "localhost"
         jupyter_port: 8888

       # Local execution settings
       local:
         python_env_path: "${LOCAL_PYTHON_VENV}"

   # Approval configuration for Python execution
   approval:
     global_mode: "selective"
     capabilities:
       python_execution:
         enabled: true
         mode: "epics_writes"  # disabled, all_code, epics_writes

**Configuration Options:**

- **python_executor**: Service-level configuration for retry behavior and timeouts

  - **max_generation_retries**: Maximum attempts for code generation failures (default: 3)
  - **max_execution_retries**: Maximum attempts for code execution failures (default: 3)
  - **execution_timeout_seconds**: Maximum time allowed for code execution (default: 600 seconds)

- **execution_method**: "container" for secure isolation, "local" for direct host execution
- **modes**: Different execution environments with specific approval requirements
- **Container settings**: Jupyter endpoint configuration for containerized execution
- **Local settings**: Python environment path for direct execution

Control System Pattern Detection
================================

The Python Execution Service uses **pattern detection** to identify control system operations (reads/writes) in generated code. This enables the approval system to determine when human oversight is required.

.. dropdown:: How Pattern Detection Works
   :color: info

   The pattern detection system operates during the **static analysis** phase of code execution:

   1. **Code Generation**: LLM generates Python code
   2. **Pattern Detection**: Regex patterns scan code for control system operations
   3. **Approval Decision**: Based on detected operations, code may require approval
   4. **Execution**: Approved or non-risky code executes

   This provides defense-in-depth security for control system operations.

Configuration
-------------

Define regex patterns for your control system in ``config.yml``:

.. code-block:: yaml

   control_system:
     type: epics
     patterns:
       epics:
         write:
           - 'epics\.caput\('       # Matches: epics.caput(...)
           - '\.put\('              # Matches: pv.put(...)
         read:
           - 'epics\.caget\('       # Matches: epics.caget(...)
           - '\.get\('              # Matches: pv.get(...)

**Pattern Syntax:**

- Uses Python regex (``re`` module)
- Escape special characters: ``\.`` for literal dots, ``\(`` for literal parentheses
- Patterns match anywhere in the code
- Multiple patterns are OR'd together (any match triggers detection)

Approval Integration
--------------------

Pattern detection integrates with the approval system:

.. code-block:: yaml

   approval:
     global_mode: "selective"
     capabilities:
       python_execution:
         enabled: true
         mode: "epics_writes"    # Require approval for EPICS writes

**Approval Modes:**

- ``disabled``: No approval required (use with caution!)
- ``all_code``: Require approval for all Python execution
- ``epics_writes``: Require approval only for code with write operations (recommended)

The ``epics_writes`` mode uses pattern detection to identify write operations and only interrupts for human review when necessary.

Programmatic Usage
------------------

Use pattern detection directly in your code:

.. code-block:: python

   from osprey.services.python_executor.pattern_detection import detect_control_system_operations

   # Analyze generated code
   code = """
   current = epics.caget('BEAM:CURRENT')
   if current < 400:
       epics.caput('ALARM:STATUS', 1)
   """

   result = detect_control_system_operations(code)

   print(f"Has writes: {result['has_writes']}")         # True
   print(f"Has reads: {result['has_reads']}")           # True
   print(f"Control system: {result['control_system_type']}")  # 'epics'
   print(f"Write patterns matched: {result['detected_patterns']['writes']}")
   print(f"Read patterns matched: {result['detected_patterns']['reads']}")

**Use Cases:**

- Custom approval logic in capabilities
- Pre-execution safety checks
- Audit logging of control system operations
- Dynamic approval routing based on operation type

Custom Control Systems
----------------------

Define patterns for custom or non-EPICS control systems:

.. code-block:: yaml

   control_system:
     type: tango
     patterns:
       tango:
         write:
           - 'tango\.write_attribute\('
           - 'device_proxy\.write_attribute\('
         read:
           - 'tango\.read_attribute\('
           - 'device_proxy\.read_attribute\('

The pattern detection system is **control system agnostic** - it works with any control system by configuring appropriate patterns.

.. seealso::

   :doc:`06_control-system-integration`
       Complete guide to control system connectors and pattern detection

   :doc:`01_human-approval-workflows`
       How approval system uses pattern detection results

Integration Patterns
====================

Using Python Execution in Capabilities
---------------------------------------

Use the Python execution service directly in your capabilities with proper approval handling:

.. code-block:: python

   from osprey.base import BaseCapability, capability_node
   from osprey.state import AgentState, StateManager
   from osprey.registry import get_registry
   from osprey.services.python_executor import PythonExecutionRequest
   from osprey.approval import (
       create_approval_type,
       get_approval_resume_data,
       clear_approval_state,
       handle_service_with_interrupts
   )
   from osprey.utils.config import get_full_configuration
   from langgraph.types import Command

   @capability_node
   class DataAnalysisCapability(BaseCapability):
       """Data analysis capability using Python execution service."""

       async def execute(self) -> dict:
           # Get registry
           registry = get_registry()

           # Get Python executor service from registry
           python_service = registry.get_service("python_executor")
           if not python_service:
               raise RuntimeError("Python executor service not available")

           # Create service configuration
           main_configurable = get_full_configuration()
           service_config = {
               "configurable": {
                   **main_configurable,
                   "thread_id": f"data_analysis_{self._step.get('context_key', 'default')}",
                   "checkpoint_ns": "python_executor"
               }
           }

           # Check for approval resume first
           has_approval_resume, approved_payload = get_approval_resume_data(
               self._state, create_approval_type("data_analysis")
           )

           if has_approval_resume:
               # Handle approval resume
               if approved_payload:
                   resume_response = {"approved": True, **approved_payload}
               else:
                   resume_response = {"approved": False}

               service_result = await python_service.ainvoke(
                   Command(resume=resume_response), config=service_config
               )
               approval_cleanup = clear_approval_state()
           else:
               # Normal execution flow
               # Create structured prompts for Python generation
               capability_prompts = [
                   "**ANALYSIS REQUIREMENTS:**",
                   "- Generate statistical summary of the data",
                   "- Create visualizations to identify trends",
                   "- Identify patterns and anomalies",

                   "**EXPECTED OUTPUT:**",
                   "Create a results dictionary with:",
                   "- statistics: Statistical summary metrics",
                   "- trends: Identified trends and patterns",
                   "- visualizations: List of generated plots"
               ]

               # Create execution request
               execution_request = PythonExecutionRequest(
                   user_query=self._state.get("input_output", {}).get("user_query", ""),
                   task_objective=self.get_task_objective(),
                   capability_prompts=capability_prompts,
                   expected_results={
                       "statistics": "dict",
                       "trends": "list",
                       "visualizations": "list"
                   },
                   execution_folder_name="data_analysis",
                   capability_context_data=self._state.get('capability_context_data', {}),
                   retries=3
               )

               # Use centralized interrupt handler
               service_result = await handle_service_with_interrupts(
                   service=python_service,
                   request=execution_request,
                   config=service_config,
                   logger=logger,
                   capability_name="DataAnalysis"
               )
               approval_cleanup = None

           # Process results (both paths converge here)
           execution_result = service_result.execution_result

           # Store context using helper method
           context_updates = self.store_output_context(analysis_context)

           # Return with optional approval cleanup
           if approval_cleanup:
               return {**context_updates, **approval_cleanup}
           else:
               return context_updates

Execution Environment Management
================================

Container vs Local Execution
----------------------------

The Python execution service supports both container and local execution environments. The execution method is primarily configured through the config system, but you can also implement dynamic selection logic if needed.

**Configuration-Based Execution Method:**

The execution method is typically set in your ``config.yml``:

.. code-block:: yaml

   osprey:
     execution:
       execution_method: "container"  # or "local"
       container:
         jupyter_host: "localhost"
         jupyter_port: 8888
       local:
         python_env_path: "${LOCAL_PYTHON_VENV}"

**Example: Dynamic Environment Selection (Advanced Use Case):**

For advanced scenarios where you need to dynamically choose execution environments based on request characteristics, here's an example pattern:

.. code-block:: python

   class FlexiblePythonExecution:
       """Example: Dynamic execution environment selection.

       Note: This is an advanced pattern. Most use cases should rely on
       the standard config.yml execution_method setting.
       """

       def _select_execution_environment(self, code_request: dict) -> str:
           """Example: Select execution environment based on request characteristics.

           This would be used to override the default config.yml setting
           for specific requests that have special requirements.
           """

           requires_isolation = code_request.get("requires_isolation", False)
           has_dependencies = code_request.get("has_special_dependencies", False)
           is_long_running = code_request.get("estimated_time", 0) > 300
           security_level = code_request.get("security_level", "medium")

           # Example decision logic for environment selection
           if security_level == "high" or requires_isolation:
               return "container"
           elif has_dependencies or is_long_running:
               return "container"
           else:
               return "local"  # Faster for simple operations

       async def execute_with_dynamic_environment(self, state, request_data):
           """Example: Override execution method in service config."""

           # Get the dynamic execution method
           execution_method = self._select_execution_environment(request_data)

           # Override the config setting for this specific request
           main_configurable = get_full_configuration()
           service_config = {
               "configurable": {
                   **main_configurable,
                   "execution_method": execution_method,  # Override config.yml setting
                   "thread_id": f"dynamic_{execution_method}",
                   "checkpoint_ns": "python_executor"
               }
           }

           # Use the service with the dynamic configuration
           # ... rest of service call ...

Advanced Patterns
=================

Multi-Stage Analysis Pipeline
-----------------------------

Chain multiple Python executions for complex analysis workflows with proper approval handling:

.. code-block:: python

   async def multi_stage_analysis(self, state: AgentState, data_context: dict) -> dict:
       """Execute multi-stage analysis pipeline with approval handling."""

       registry = get_registry()
       python_service = registry.get_service("python_executor")
       main_configurable = get_full_configuration()
       logger = logging.getLogger(__name__)

       # Stage 1: Data preprocessing
       stage1_config = {
           "configurable": {
               **main_configurable,
               "thread_id": "stage1_preprocessing",
               "checkpoint_ns": "python_executor"
           }
       }

       preprocessing_prompts = [
           "**PREPROCESSING STAGE:**",
           "- Clean and validate the input data",
           "- Handle missing values and outliers",
           "- Prepare data for statistical analysis"
       ]

       preprocessing_request = PythonExecutionRequest(
           user_query="Data preprocessing stage",
           task_objective="Clean and prepare data for analysis",
           capability_prompts=preprocessing_prompts,
           expected_results={"cleaned_data": "pandas.DataFrame", "summary": "dict"},
           execution_folder_name="stage1_preprocessing",
           capability_context_data=data_context
       )

       stage1_result = await handle_service_with_interrupts(
           service=python_service,
           request=preprocessing_request,
           config=stage1_config,
           logger=logger,
           capability_name="PreprocessingStage"
       )

       # Stage 2: Statistical analysis (using results from stage 1)
       stage2_config = {
           "configurable": {
               **main_configurable,
               "thread_id": "stage2_analysis",
               "checkpoint_ns": "python_executor"
           }
       }

       analysis_prompts = [
           "**STATISTICAL ANALYSIS STAGE:**",
           "- Use the cleaned data from preprocessing stage",
           "- Perform comprehensive statistical analysis",
           "- Generate summary statistics and insights"
       ]

       # Combine original context with preprocessing results
       stage2_context = {
           **data_context,
           "preprocessing_results": stage1_result.execution_result.results
       }

       analysis_request = PythonExecutionRequest(
           user_query="Statistical analysis stage",
           task_objective="Perform statistical analysis on preprocessed data",
           capability_prompts=analysis_prompts,
           expected_results={"statistics": "dict", "insights": "list"},
           execution_folder_name="stage2_analysis",
           capability_context_data=stage2_context
       )

       stage2_result = await handle_service_with_interrupts(
           service=python_service,
           request=analysis_request,
           config=stage2_config,
           logger=logger,
           capability_name="AnalysisStage"
       )

       return {
           "pipeline_completed": True,
           "stages": {
               "preprocessing": stage1_result,
               "analysis": stage2_result
           }
       }

Troubleshooting
===============

**Common Issues:**

**Issue**: Python execution service not available
   - **Cause**: Service not registered in framework registry
   - **Solution**: Verify PythonExecutorService is registered in registry configuration using ``registry.get_service("python_executor")``

**Issue**: GraphInterrupt not being handled properly
   - **Cause**: Using direct service.ainvoke() instead of handle_service_with_interrupts()
   - **Solution**: Always use handle_service_with_interrupts() for service calls that may require approval

**Issue**: Approval resume not working
   - **Cause**: Missing approval resume check or incorrect Command usage
   - **Solution**: Check for approval resume with get_approval_resume_data() and use Command(resume=response) for resumption

**Issue**: Service configuration errors
   - **Cause**: Missing thread_id, checkpoint_ns, or incorrect configurable structure
   - **Solution**: Use get_full_configuration() and include proper thread_id and checkpoint_ns in service config

**Issue**: Container execution failing with connection errors
   - **Cause**: Jupyter container not accessible or misconfigured
   - **Solution**: Check container endpoints and ensure Jupyter is running

**Issue**: Generated notebooks not accessible
   - **Cause**: File path or URL generation issues
   - **Solution**: Check execution folder configuration and notebook link generation


.. seealso::

   :doc:`04_memory-storage-service`
       Integrate memory storage with Python execution

   :doc:`05_container-and-deployment`
       Advanced container orchestration

   :doc:`01_human-approval-workflows`
       Understanding the approval system integration

   :doc:`../../api_reference/03_production_systems/03_python-execution`
       Complete Python execution API