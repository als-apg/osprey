=======================
Python Service Overview
=======================

This guide covers the complete Python Execution Service architecture, integration patterns, and advanced usage.

.. contents:: Table of Contents
   :local:
   :depth: 2

Service Architecture
====================

The Python Execution Service orchestrates the complete lifecycle of Python code generation and execution through a LangGraph-based pipeline.

Execution Pipeline
------------------

The service follows this flow:

.. code-block:: text

   1. Code Generation
      ├─ Generator selection (Legacy/Claude/Mock/Custom)
      ├─ Request processing with context
      └─ Error-aware regeneration on retry

   2. Security Analysis
      ├─ Static code analysis
      ├─ Pattern detection (EPICS writes, etc.)
      └─ Security risk assessment

   3. Approval Decision
      ├─ Check approval configuration
      ├─ Pattern-based triggering (e.g., EPICS writes)
      └─ LangGraph interrupt if required

   4. Code Execution
      ├─ Environment selection (container/local)
      ├─ Isolated execution
      └─ Result capture

   5. Result Processing
      ├─ Success: Extract results, create notebook
      ├─ Failure: Error analysis, retry logic
      └─ Return structured response

Key Components
--------------

**Service Layer**
   LangGraph-based orchestration with state management, checkpointing, and human-in-the-loop interrupts

**Code Generators**
   Pluggable implementations for code generation (see :doc:`generator-basic`, :doc:`generator-claude`, :doc:`generator-mock`)

**Security Analyzer**
   Static analysis and pattern detection for risky operations

**Execution Engine**
   Container or local execution with consistent result handling

**Approval System**
   Integration with Osprey's approval framework for human oversight

Integration Patterns
=====================

Using Python Execution in Capabilities
---------------------------------------

Standard pattern for integrating Python execution into capabilities:

.. code-block:: python

   from osprey.base import BaseCapability, capability_node
   from osprey.registry import get_registry
   from osprey.services.python_executor import PythonExecutionRequest
   from osprey.approval import (
       create_approval_type,
       get_approval_resume_data,
       clear_approval_state,
       handle_service_with_interrupts
   )
   from osprey.utils.config import get_full_configuration

   @capability_node
   class DataAnalysisCapability(BaseCapability):
       """Capability using Python execution service."""

       async def execute(self) -> dict:
           registry = get_registry()
           python_service = registry.get_service("python_executor")

           # Service configuration
           main_configurable = get_full_configuration()
           service_config = {
               "configurable": {
                   **main_configurable,
                   "thread_id": f"analysis_{self._step.get('context_key', 'default')}",
                   "checkpoint_ns": "python_executor"
               }
           }

           # Check for approval resume
           has_approval_resume, approved_payload = get_approval_resume_data(
               self._state, create_approval_type("data_analysis")
           )

           if has_approval_resume:
               # Resume after approval
               resume_response = {"approved": True, **approved_payload} if approved_payload else {"approved": False}
               service_result = await python_service.ainvoke(
                   Command(resume=resume_response), config=service_config
               )
               approval_cleanup = clear_approval_state()
           else:
               # Normal execution
               capability_prompts = [
                   "**ANALYSIS REQUIREMENTS:**",
                   "- Generate statistical summary of the data",
                   "- Create visualizations to identify trends",
                   "**EXPECTED OUTPUT:**",
                   "- statistics: Statistical summary metrics",
                   "- visualizations: List of generated plots"
               ]

               execution_request = PythonExecutionRequest(
                   user_query=self._state.get("input_output", {}).get("user_query", ""),
                   task_objective=self.get_task_objective(),
                   capability_prompts=capability_prompts,
                   expected_results={
                       "statistics": "dict",
                       "visualizations": "list"
                   },
                   execution_folder_name="data_analysis",
                   capability_context_data=self._state.get('capability_context_data', {})
               )

               service_result = await handle_service_with_interrupts(
                   service=python_service,
                   request=execution_request,
                   config=service_config,
                   logger=logger,
                   capability_name="DataAnalysis"
               )
               approval_cleanup = None

           # Process results
           context_updates = self.store_output_context({
               "analysis": service_result.execution_result.results
           })

           return {**context_updates, **approval_cleanup} if approval_cleanup else context_updates

Multi-Stage Pipelines
----------------------

Chain multiple Python executions for complex workflows:

.. code-block:: python

   async def multi_stage_analysis(self, data_context: dict) -> dict:
       """Execute multi-stage analysis pipeline."""
       registry = get_registry()
       python_service = registry.get_service("python_executor")
       main_configurable = get_full_configuration()

       # Stage 1: Preprocessing
       stage1_config = {
           "configurable": {
               **main_configurable,
               "thread_id": "stage1_preprocessing",
               "checkpoint_ns": "python_executor"
           }
       }

       preprocessing_request = PythonExecutionRequest(
           user_query="Data preprocessing",
           task_objective="Clean and prepare data",
           capability_prompts=[
               "- Handle missing values and outliers",
               "- Prepare data for statistical analysis"
           ],
           expected_results={"cleaned_data": "pandas.DataFrame"},
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

       # Stage 2: Analysis (using results from stage 1)
       stage2_config = {
           "configurable": {
               **main_configurable,
               "thread_id": "stage2_analysis",
               "checkpoint_ns": "python_executor"
           }
       }

       stage2_context = {
           **data_context,
           "preprocessing_results": stage1_result.execution_result.results
       }

       analysis_request = PythonExecutionRequest(
           user_query="Statistical analysis",
           task_objective="Analyze preprocessed data",
           capability_prompts=["- Perform comprehensive statistical analysis"],
           expected_results={"statistics": "dict"},
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

Execution Environment Management
---------------------------------

The execution method is primarily configured in ``config.yml``:

.. code-block:: yaml

   osprey:
     execution:
       execution_method: "container"  # or "local"
       container:
         jupyter_host: "localhost"
         jupyter_port: 8888
       local:
         python_env_path: "${LOCAL_PYTHON_VENV}"

**Container Execution:**

- Secure isolation in Jupyter container
- Full dependency management
- Required for untrusted code

**Local Execution:**

- Direct execution on host
- Faster for simple operations
- Use only for trusted environments

For advanced scenarios requiring dynamic environment selection, you can override the configuration in the service config:

.. code-block:: python

   service_config = {
       "configurable": {
           **main_configurable,
           "execution_method": "container",  # Override config.yml
           "thread_id": "dynamic_execution",
           "checkpoint_ns": "python_executor"
       }
   }

Code Generator Architecture
============================

The service uses a **pluggable code generator architecture** based on Python's Protocol pattern.

CodeGenerator Protocol
----------------------

All code generators implement this simple interface:

.. code-block:: python

   from typing import Protocol, runtime_checkable
   from osprey.services.python_executor.models import PythonExecutionRequest

   @runtime_checkable
   class CodeGenerator(Protocol):
       """Protocol for code generators."""

       async def generate_code(
           self,
           request: PythonExecutionRequest,
           error_chain: list[str]
       ) -> str:
           """Generate Python code based on request and error feedback.

           Args:
               request: Execution request with task details, context, guidance
               error_chain: Previous errors from failed attempts (empty for first)

           Returns:
               Generated Python code as a string

           Raises:
               CodeGenerationError: If generation fails
           """
           ...

**Key Features:**

- **No inheritance required** - Just implement the method
- **Runtime type checking** - ``isinstance()`` works at runtime
- **Clean separation** - Generator independent from executor service
- **Error-aware** - ``error_chain`` enables iterative improvement

Available Generators
--------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 40

   * - Generator
     - Speed
     - Quality
     - Use Case
   * - **Basic LLM**
     - Fast
     - Good
     - Self-hosted, simple setups
   * - **Claude Code**
     - Slower
     - Excellent
     - Complex tasks, learning
   * - **Mock**
     - Instant
     - N/A (Fixed)
     - Testing only

See :doc:`generator-basic`, :doc:`generator-claude`, and :doc:`generator-mock` for detailed documentation.

Generator Selection
-------------------

Factory Pattern
^^^^^^^^^^^^^^^

Generators are created using a factory that handles discovery and instantiation:

.. code-block:: python

   from osprey.services.python_executor.generation import create_code_generator

   # Use configuration from config.yml
   generator = create_code_generator()

   # Or provide custom configuration
   config = {
       "execution": {
           "code_generator": "claude_code",
           "generators": {"claude_code": {"profile": "fast"}}
       }
   }
   generator = create_code_generator(config)

**Factory Features:**

- Automatic registry discovery for custom generators
- Graceful fallback when optional dependencies missing
- Configuration-driven selection
- Consistent initialization

Configuration-Driven Selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Configure in ``config.yml``:

.. code-block:: yaml

   osprey:
     execution:
       code_generator: "basic"  # or "claude_code", "mock", custom name

       generators:
         basic:
           model_config_name: "python_code_generator"
         claude_code:
           profile: "fast"
         mock:
           behavior: "success"

The executor service automatically uses the factory to create the configured generator.

Decision Guide
^^^^^^^^^^^^^^

**Choose Basic LLM when:**

- Self-hosted model support needed
- Minimal dependencies preferred (no Claude SDK)
- Fast response times important
- Simple, straightforward generation

**Choose Claude Code when:**

- Tasks require multi-step reasoning
- Have successful examples to learn from
- Code quality paramount
- Can accept longer generation times

**Choose Mock when:**

- Writing tests for Python executor
- Developing capabilities locally
- Running CI/CD pipelines
- Need deterministic, fast generation

Creating Custom Generators
===========================

Implementing the Protocol
--------------------------

Create a class with the ``generate_code()`` method:

.. code-block:: python

   from osprey.services.python_executor.models import PythonExecutionRequest
   from osprey.services.python_executor.exceptions import CodeGenerationError

   class DomainSpecificGenerator:
       """Custom generator for domain-specific code generation."""

       def __init__(self, model_config: dict | None = None):
           self.model_config = model_config or {}

       async def generate_code(
           self,
           request: PythonExecutionRequest,
           error_chain: list[str]
       ) -> str:
           """Generate code using domain-specific logic."""
           try:
               # Your generation logic
               task = request.task_objective
               context = request.capability_context_data

               code = await self._generate_with_domain_knowledge(task, context)

               # Incorporate error feedback if retrying
               if error_chain:
                   code = await self._fix_errors(code, error_chain)

               return code

           except Exception as e:
               raise CodeGenerationError(
                   f"Domain-specific generation failed: {str(e)}",
                   generation_attempt=len(error_chain) + 1,
                   error_chain=error_chain
               )

       async def _generate_with_domain_knowledge(self, task, context):
           # Implementation
           pass

       async def _fix_errors(self, code, errors):
           # Error-aware regeneration
           pass

**No inheritance required** - just implement the method!

Registering Custom Generators
------------------------------

Register through the Osprey registry:

.. code-block:: python

   # In your application's registry.py
   from osprey.registry.base import CodeGeneratorRegistration, RegistryConfig

   registry_config = RegistryConfig(
       code_generators=[
           CodeGeneratorRegistration(
               name="domain_specific",
               module_path="applications.myapp.generators.domain",
               class_name="DomainSpecificGenerator",
               description="Domain-specific code generator"
           )
       ]
   )

Then use in configuration:

.. code-block:: yaml

   osprey:
     execution:
       code_generator: "domain_specific"
       generators:
         domain_specific:
           domain: "physics"
           template_library: "scientific_computing"

The factory automatically discovers and instantiates your custom generator!

Testing Custom Generators
--------------------------

Test in isolation:

.. code-block:: python

   import pytest
   from osprey.services.python_executor.models import PythonExecutionRequest

   @pytest.mark.asyncio
   async def test_custom_generator():
       generator = DomainSpecificGenerator(model_config={"domain": "test"})

       request = PythonExecutionRequest(
           user_query="Calculate beam emittance",
           task_objective="Physics calculation",
           execution_folder_name="test_physics"
       )

       # Test generation
       code = await generator.generate_code(request, [])
       assert code
       assert "import" in code

       # Test error handling
       error_chain = ["NameError: undefined variable"]
       improved_code = await generator.generate_code(request, error_chain)
       assert improved_code != code

Best Practices
--------------

**For Custom Generators:**

1. **Handle errors gracefully** - Raise ``CodeGenerationError`` with descriptive messages
2. **Use error feedback** - Incorporate ``error_chain`` to improve code on retries
3. **Include imports** - Generate complete, executable Python code
4. **Document configuration** - Clearly document ``model_config`` options
5. **Test thoroughly** - Test both success and error paths

**Configuration:**

- Accept ``model_config`` parameter in ``__init__``
- Support both inline config and external config files
- Provide sensible defaults
- Document all options

Configuration Reference
=======================

Complete Service Configuration
-------------------------------

.. code-block:: yaml

   osprey:
     python_executor:
       max_generation_retries: 3
       max_execution_retries: 3
       execution_timeout_seconds: 600

     execution:
       # Generator selection
          code_generator: "basic"  # basic | claude_code | mock | custom

          # Execution environment
          execution_method: "container"  # or "local"

          # Generator-specific configuration
          generators:
            basic:
              # Option 1: Reference existing model config
              model_config_name: "python_code_generator"

              # Option 2: Inline configuration
              # provider: "openai"
              # model_id: "gpt-4"

            claude_code:
           profile: "fast"  # fast | balanced | robust
           claude_config_path: "claude_generator_config.yml"  # optional

         mock:
           behavior: "success"  # success | syntax_error | runtime_error | epics_write

       # Execution modes
       modes:
         read_only:
           kernel_name: "python3-epics-readonly"
           allows_writes: false
           requires_approval: false
         write_access:
           kernel_name: "python3-epics-write"
           allows_writes: true
           requires_approval: true

       # Container settings
       container:
         jupyter_host: "localhost"
         jupyter_port: 8888

       # Local execution settings
       local:
         python_env_path: "${LOCAL_PYTHON_VENV}"

Approval Configuration
----------------------

.. code-block:: yaml

   approval:
     global_mode: "selective"
     capabilities:
       python_execution:
         enabled: true
         mode: "epics_writes"  # disabled | all_code | epics_writes

Pattern Detection
-----------------

Configure control system operation detection:

.. code-block:: yaml

   control_system:
     type: epics
     patterns:
       epics:
         write:
           - 'epics\.caput\('
           - '\.put\('
         read:
           - 'epics\.caget\('
           - '\.get\('

**Programmatic Usage:**

.. code-block:: python

   from osprey.services.python_executor.analysis.pattern_detection import (
       detect_control_system_operations
   )

   result = detect_control_system_operations(code)
   if result['has_writes']:
       # Trigger approval workflow
       pass

Advanced Topics
===============

Conditional Generator Selection
--------------------------------

Select generators dynamically:

.. code-block:: python

   def select_generator_for_task(task_complexity: str) -> str:
       """Select generator based on task characteristics."""
       if task_complexity == "simple":
           return "legacy"  # Fast
       elif task_complexity == "complex":
           return "claude_code"  # High quality
       else:
           return "legacy"

   config = {
       "execution": {
           "code_generator": select_generator_for_task(complexity)
       }
   }
   generator = create_code_generator(config)

Hybrid Approaches
-----------------

Combine multiple strategies:

.. code-block:: python

   class HybridGenerator:
       """Use different strategies based on context."""

       def __init__(self, model_config: dict | None = None):
           self.fast_generator = BasicLLMCodeGenerator(model_config)
           self.quality_generator = ClaudeCodeGenerator(model_config)

       async def generate_code(
           self,
           request: PythonExecutionRequest,
           error_chain: list[str]
       ) -> str:
           # Fast generator first
           if not error_chain:
               return await self.fast_generator.generate_code(request, [])

           # Switch to quality generator on retry
           return await self.quality_generator.generate_code(request, error_chain)

Registry Integration
====================

The generator system integrates with Osprey's registry for automatic discovery:

**How It Works:**

1. Framework registers built-in generators (basic, claude_code, mock)
2. Applications register custom generators via ``RegistryConfig``
3. Factory queries registry for requested generator
4. Registry returns generator class and metadata
5. Factory instantiates with configuration

**Registry Storage:**

.. code-block:: python

   registry._registries['code_generators'] = {
       "basic": {
           "registration": CodeGeneratorRegistration(...),
           "class": BasicLLMCodeGenerator
       },
       "claude_code": {...},
       # Your custom generators appear here
   }

**Benefits:**

- Automatic discovery of application generators
- No hardcoded generator lists
- Consistent registration pattern
- Easy to extend without modifying framework

Troubleshooting
===============

Common Issues
-------------

**Service not available**
   Verify ``PythonExecutorService`` is registered: ``registry.get_service("python_executor")``

**GraphInterrupt not handled**
   Use ``handle_service_with_interrupts()`` for service calls that may require approval

**Approval resume not working**
   Check for resume with ``get_approval_resume_data()`` and use ``Command(resume=response)``

**Service configuration errors**
   Include ``thread_id`` and ``checkpoint_ns`` in service config. Use ``get_full_configuration()`` for base config

**Container execution failing**
   Check Jupyter container is accessible and ``jupyter_host``/``jupyter_port`` are correct

**Code generator not found**
   Verify ``code_generator`` setting in config and that the generator is registered

**Generated code quality issues**
   Consider using ``claude_code`` generator with ``fast`` (single-phase) or ``robust`` (multi-phase) profile

See Also
========

:doc:`generator-basic`
    Basic LLM generator for simple setups

:doc:`generator-claude`
    Advanced Claude Code generator with multi-phase workflows

:doc:`generator-mock`
    Mock generator for testing

:doc:`index`
    Python Execution Service landing page

:doc:`../01_human-approval-workflows`
    Understanding the approval system integration

:doc:`../../03_core-framework-systems/03_registry-and-discovery`
    Understanding the registry system

