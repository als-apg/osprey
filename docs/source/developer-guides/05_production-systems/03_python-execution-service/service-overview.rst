=======================
Python Service Overview
=======================

This guide covers the Python Execution Service architecture and how to extend it with custom code generators.

.. seealso::

   **Looking for integration patterns and configuration?** See :doc:`../03_python-execution-service/index` for complete integration examples, configuration reference, and usage patterns.

Service Architecture
====================

The Python Execution Service orchestrates code generation, security analysis, approval, and execution through a LangGraph-based pipeline.

**Pipeline Overview:**

.. code-block:: text

   Request → Generator Selection → Code Generation
                                         ↓
                                   Security Analysis
                                         ↓
                                   Approval (if needed)
                                         ↓
                                   Execution (container/local)
                                         ↓
                                   Result Processing → Response

**Key Components:**

1. **Service Layer** - LangGraph orchestration with checkpointing and interrupts
2. **Code Generators** - Pluggable implementations (Basic LLM, Claude Code, Mock, Custom)
3. **Security Analyzer** - Pattern detection for control system operations
4. **Execution Engine** - Container or local execution with consistent result handling
5. **Approval System** - Human-in-the-loop for high-stakes operations

For detailed integration patterns, see :doc:`../03_python-execution-service/index`.

.. admonition:: **Control System Operations in Generated Code**

   Generated Python code interacts with control systems using ``osprey.runtime`` utilities (``read_channel()``, ``write_channel()``), not direct connector imports. The execution wrapper automatically configures these utilities from the execution context, ensuring generated code works with any control system (EPICS, Mock, LabVIEW, etc.) and that notebooks remain reproducible. See :doc:`../../../getting-started/control-assistant-part3-production` for details.

Using the Service
=================

Quick Pattern
-------------

Minimal example of using the service in a capability:

.. code-block:: python

   from osprey.base import capability_node, BaseCapability
   from osprey.registry import get_registry
   from osprey.services.python_executor import PythonExecutionRequest
   from osprey.approval import handle_service_with_interrupts
   from osprey.utils.config import get_full_configuration

   @capability_node
   class MyCapability(BaseCapability):
       async def execute(self) -> dict:
           # Get service
           registry = get_registry()
           python_service = registry.get_service("python_executor")

           # Configure service
           service_config = {
               "configurable": {
                   **get_full_configuration(),
                   "thread_id": f"my_task_{self._step.get('context_key')}",
                   "checkpoint_ns": "python_executor"
               }
           }

           # Create request
           request = PythonExecutionRequest(
               user_query=self._state.get("input_output", {}).get("user_query", ""),
               task_objective=self.get_task_objective(),
               capability_prompts=[
                   "Generate analysis code that produces a results dictionary"
               ],
               expected_results={"statistics": "dict"},
               execution_folder_name="analysis",
               capability_context_data=self._state.get('capability_context_data', {})
           )

           # Execute with approval handling
           result = await handle_service_with_interrupts(
               service=python_service,
               request=request,
               config=service_config,
               logger=self.get_logger(),
               capability_name="MyCapability"
           )

           return self.store_output_context(result.execution_result.results)

.. seealso::

   - :doc:`../03_python-execution-service/index` for complete integration patterns including approval resume, multi-stage pipelines, and error handling
   - :doc:`../01_human-approval-workflows` for approval system integration

Code Generator System
=====================

The generator system uses Python's Protocol pattern for clean, extensible code generation.

Generator Protocol
------------------

All generators implement this simple interface:

.. code-block:: python

   from typing import Protocol
   from osprey.services.python_executor.models import PythonExecutionRequest

   @runtime_checkable
   class CodeGenerator(Protocol):
       async def generate_code(
           self,
           request: PythonExecutionRequest,
           error_chain: list[str]
       ) -> str:
           """Generate Python code based on request and error feedback."""
           ...

**Benefits:**

- No inheritance required - just implement the method
- Runtime type checking with ``isinstance()``
- Clean separation from executor service
- Error-aware iterative improvement via ``error_chain``

Available Generators
--------------------

The framework provides three built-in generators:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Generator
     - Best For
     - Documentation
   * - **Basic LLM**
     - Self-hosted models, simple setups
     - :doc:`generator-basic`
   * - **Claude Code**
     - Complex tasks, learning from codebase
     - :doc:`generator-claude`
   * - **Mock**
     - Testing, CI/CD
     - :doc:`generator-mock`

Generator Factory
-----------------

Generators are created via a factory that handles discovery and configuration:

.. code-block:: python

   from osprey.services.python_executor.generation import create_code_generator

   # Uses config.yml
   generator = create_code_generator()

   # Or provide custom config
   config = {
       "execution": {
           "code_generator": "claude_code",
           "generators": {"claude_code": {"profile": "fast"}}
       }
   }
   generator = create_code_generator(config)

**Factory Features:**

- Automatic registry discovery for custom generators
- Graceful fallback when dependencies missing
- Configuration-driven selection
- Consistent initialization

Creating Custom Generators
===========================

Implement the Protocol
-----------------------

Create a class with the ``generate_code()`` method:

.. code-block:: python

   from osprey.services.python_executor.models import PythonExecutionRequest
   from osprey.services.python_executor.exceptions import CodeGenerationError

   class DomainSpecificGenerator:
       """Custom generator for domain-specific code generation."""

       def __init__(self, model_config: dict | None = None):
           self.model_config = model_config or {}
           # Initialize your generation resources

       async def generate_code(
           self,
           request: PythonExecutionRequest,
           error_chain: list[str]
       ) -> str:
           """Generate code using domain-specific logic."""
           try:
               # Access request fields
               task = request.task_objective
               context = request.capability_context_data
               prompts = request.capability_prompts

               # Your generation logic
               code = await self._generate(task, context, prompts)

               # Use error feedback on retries
               if error_chain:
                   code = await self._improve_with_errors(code, error_chain)

               return code

           except Exception as e:
               raise CodeGenerationError(
                   f"Generation failed: {str(e)}",
                   generation_attempt=len(error_chain) + 1,
                   error_chain=error_chain
               )

       async def _generate(self, task, context, prompts):
           """Your generation implementation."""
           # Use templates, rules, LLMs, or any approach
           pass

       async def _improve_with_errors(self, code, errors):
           """Fix code based on error feedback."""
           # Learn from errors on retry
           pass

**No inheritance required** - just implement the method signature!

Register Your Generator
-----------------------

Register through the Osprey registry:

.. code-block:: python

   # In your application's registry.py
   from osprey.registry.base import CodeGeneratorRegistration, RegistryConfig

   registry_config = RegistryConfig(
       code_generators=[
           CodeGeneratorRegistration(
               name="domain_specific",
               module_path="myapp.generators.domain",
               class_name="DomainSpecificGenerator",
               description="Domain-specific code generator"
           )
       ]
   )

Configure and Use
-----------------

Use your generator via configuration:

.. code-block:: yaml

   # config.yml
   osprey:
     execution:
       code_generator: "domain_specific"
       generators:
         domain_specific:
           domain: "physics"
           template_library: "scientific_computing"

The factory automatically discovers and instantiates your generator!

Testing Custom Generators
--------------------------

Test generators in isolation:

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
       improved = await generator.generate_code(request, error_chain)
       assert improved != code

Best Practices
--------------

**For Custom Generators:**

1. **Raise CodeGenerationError** - Use framework exception for consistent error handling
2. **Use error_chain** - Incorporate previous errors to improve code on retries
3. **Generate complete code** - Include all imports and setup
4. **Document model_config** - Clearly document configuration options
5. **Test thoroughly** - Test success, retries, and error paths

**Configuration:**

- Accept ``model_config`` parameter in ``__init__``
- Provide sensible defaults
- Support both inline config and external files
- Document all options

Advanced Patterns
=================

Hybrid Generator
----------------

Combine multiple strategies:

.. code-block:: python

   class HybridGenerator:
       """Fast generator first, quality generator on retry."""

       def __init__(self, model_config: dict | None = None):
           self.fast_generator = BasicLLMCodeGenerator(model_config)
           self.quality_generator = ClaudeCodeGenerator(model_config)

       async def generate_code(
           self,
           request: PythonExecutionRequest,
           error_chain: list[str]
       ) -> str:
           # Fast generator for first attempt
           if not error_chain:
               return await self.fast_generator.generate_code(request, [])

           # Quality generator on retry
           return await self.quality_generator.generate_code(request, error_chain)

Conditional Selection
---------------------

Select generators dynamically based on task:

.. code-block:: python

   def select_generator_for_task(task_complexity: str) -> str:
       """Choose generator based on task characteristics."""
       if task_complexity == "simple":
           return "basic"  # Fast
       elif task_complexity == "complex":
           return "claude_code"  # High quality
       else:
           return "basic"

   # Use in capability
   config = {
       "execution": {
           "code_generator": select_generator_for_task(complexity)
       }
   }
   generator = create_code_generator(config)

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
       "your_generator": {...}
   }

**Benefits:**

- Automatic discovery of application generators
- No hardcoded generator lists
- Consistent registration pattern
- Easy extension without modifying framework

Troubleshooting
===============

**Generator not found**
   Verify ``code_generator`` setting in config.yml and that the generator is registered in your application's registry

**Code generation fails**
   Check that ``model_config`` is properly configured and the LLM is accessible

**Errors not improving on retry**
   Ensure your generator uses ``error_chain`` parameter to learn from previous failures

**Registry issues**
   Verify ``CodeGeneratorRegistration`` has correct ``module_path`` and ``class_name``

For service-level troubleshooting (approval, execution, configuration), see :doc:`../03_python-execution-service/index`.

See Also
========

:doc:`../03_python-execution-service/index`
    Complete integration guide with patterns, configuration, and examples

:doc:`generator-basic`
    Basic LLM generator for simple setups

:doc:`generator-claude`
    Advanced Claude Code generator with multi-phase workflows

:doc:`generator-mock`
    Mock generator for testing

:doc:`../01_human-approval-workflows`
    Understanding the approval system integration

:doc:`../../03_core-framework-systems/03_registry-and-discovery`
    Understanding the registry system
