==============
Mock Generator
==============

The Mock Code Generator provides instant, deterministic code generation for testing without API calls or external dependencies.

.. note::

   **For Testing Only** — The Mock Generator is designed exclusively for testing and development.

Overview
========

**Key Benefits:**

- Instant generation (no API latency)
- Works offline (no API keys needed)
- Deterministic output (same input → same output)
- Protocol-compliant (drop-in replacement)

**Use for:** unit tests, CI/CD pipelines, local development, capability testing.

Quick Start
===========

.. code-block:: python

   from osprey.services.python_executor.generation import MockCodeGenerator
   from osprey.services.python_executor import PythonExecutionRequest

   # Create generator with predefined behavior
   generator = MockCodeGenerator(behavior="success")

   request = PythonExecutionRequest(
       user_query="Calculate something",
       task_objective="Test the executor",
       execution_folder_name="test_folder"
   )

   # Generate code instantly
   code = await generator.generate_code(request, [])

**Configuration:**

.. code-block:: yaml

   # config.yml
   execution:
     code_generator: "mock"
     execution_method: "local"
     generators:
       mock:
         behavior: "success"

Predefined Behaviors
====================

The mock generator provides behaviors for common test scenarios:

.. code-block:: python

   # Valid code that executes successfully
   generator = MockCodeGenerator(behavior="success")

   # Code with syntax error (tests static analysis)
   generator = MockCodeGenerator(behavior="syntax_error")

   # Code with runtime error (tests execution error handling)
   generator = MockCodeGenerator(behavior="runtime_error")

   # Code missing results dictionary (tests validation)
   generator = MockCodeGenerator(behavior="missing_results")

   # Code with control system write operations (tests approval triggers)
   generator = MockCodeGenerator(behavior="channel_write")

   # Code with control system read operations (no approval needed)
   generator = MockCodeGenerator(behavior="channel_read")

   # Code with security concerns (tests security analysis)
   generator = MockCodeGenerator(behavior="security_risk")

Custom Code
===========

**Static Code** — Return the same code every time:

.. code-block:: python

   generator = MockCodeGenerator()
   generator.set_code("results = {'mean': 3.0}")
   code = await generator.generate_code(request, [])

**Sequence Mode** — Different code on successive calls (for retry testing):

.. code-block:: python

   generator = MockCodeGenerator()
   generator.set_code_sequence([
       "results = 1 / 0",           # First call fails
       "results = {'value': 42}"    # Second call succeeds
   ])

**Error-Aware** — Adapts to error feedback:

.. code-block:: python

   generator = MockCodeGenerator(behavior="error_aware")

   code1 = await generator.generate_code(request, [])
   code2 = await generator.generate_code(request, ["NameError: ..."])
   # Automatically adds missing imports or fixes detected errors

Call Tracking
=============

The generator tracks calls for test assertions:

.. code-block:: python

   generator = MockCodeGenerator(behavior="success")

   await generator.generate_code(request, [])

   assert generator.call_count == 1
   assert generator.last_request == request
   assert generator.last_error_chain == []

   generator.reset()  # Reset tracking, preserve configuration

Testing Patterns
================

**Service Integration Test:**

.. code-block:: python

   import pytest
   from osprey.services.python_executor import (
       PythonExecutorService,
       PythonExecutionRequest
   )

   @pytest.mark.asyncio
   async def test_executor_with_mock(test_config):
       test_config["execution"]["code_generator"] = "mock"
       test_config["execution"]["execution_method"] = "local"

       service = PythonExecutorService()
       request = PythonExecutionRequest(
           user_query="Calculate mean",
           task_objective="Test execution",
           execution_folder_name="test_mean"
       )

       result = await service.ainvoke(
           request,
           config={"configurable": test_config, "thread_id": "test"}
       )

       assert result.execution_result.success

**Retry Logic Test:**

.. code-block:: python

   generator = MockCodeGenerator()
   generator.set_code_sequence([
       "results = 1 / 0",           # First fails
       "results = {'value': 42}"    # Retry succeeds
   ])

   code1 = await generator.generate_code(request, [])
   code2 = await generator.generate_code(request, ["ZeroDivisionError"])
   assert generator.call_count == 2

**CI/CD Configuration:**

.. code-block:: python

   # conftest.py
   @pytest.fixture
   def test_config():
       return {
           "execution": {
               "code_generator": "mock",
               "execution_method": "local"
           },
           "approval": {"global_mode": "disabled"}
       }

Run in parallel (no rate limits):

.. code-block:: bash

   pytest -n auto

Key Points
==========

**Usage:**

- Use predefined behaviors for common scenarios
- Use custom code/sequences for specific test cases
- Track calls with ``call_count`` and ``last_request``
- Enable parallel test execution (no rate limits)

**Configuration:**

.. code-block:: python

   # Conditional selection for CI/CD
   if os.getenv("CI"):
       config["execution"]["code_generator"] = "mock"
   else:
       config["execution"]["code_generator"] = "claude_code"

**Limitations:**

- Testing only (not for production)
- Uses predefined patterns (not real AI generation)
- Returns code that looks valid, doesn't solve actual problems

See Also
========

:doc:`service-overview`
    Complete service documentation

:doc:`generator-basic`
    Basic LLM generator for simple setups

:doc:`generator-claude`
    Advanced code generation for production

:doc:`index`
    Python Execution Service documentation

