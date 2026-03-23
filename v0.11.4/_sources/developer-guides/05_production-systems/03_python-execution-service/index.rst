================
Python Execution
================

The Python Execution Service is Osprey's managed approach to Python code generation, security analysis, and execution. It provides a service layer that capabilities can invoke to safely generate and run Python code with human oversight.

Architecture Overview
=====================

**How It Works:**

The Python execution flow follows this pattern:

.. code-block:: text

   Capability (e.g., PythonCapability)
        ↓
   Python Executor Service
        ↓
   ┌─────────────────────────────┐
   │ 1. Code Generation          │ → Pluggable generators (Legacy/Claude/Mock)
   │ 2. Security Analysis        │ → Pattern detection (EPICS writes, etc.)
   │ 3. Approval (if needed)     │ → LangGraph interrupts for human review
   │ 4. Execution                │ → Container or local execution
   │ 5. Result Processing        │ → Structured results, notebooks, retry
   └─────────────────────────────┘
        ↓
   Results back to Capability

**Key Components:**

- **Service Layer**: Orchestrates the entire pipeline through a LangGraph-based workflow
- **Code Generators**: Pluggable implementations for code generation (the heart of the system)
- **Security Analysis**: Static analysis and pattern detection for risky operations
- **Approval System**: Human oversight for high-stakes operations (EPICS writes, etc.)
- **Execution Environment**: Container or local execution with consistent result handling

Core Concept: Pluggable Code Generators
========================================

The code generator is the core of the Python execution system. Osprey provides three built-in generators, each optimized for different use cases:

.. list-table:: Generator Comparison
   :header-rows: 1
   :widths: 20 20 20 40

   * - Generator
     - Speed
     - Quality
     - Best Use Case
   * - **Basic LLM**
     - Fast
     - Good
     - Self-hosted, simple setups, minimal dependencies
   * - **Claude Code**
     - Slower
     - Excellent
     - Complex tasks, learning from examples
   * - **Mock**
     - Instant
     - N/A (Fixed)
     - Testing, CI/CD, development

**Protocol-Based Architecture:**

All generators implement a simple protocol - no inheritance required:

.. code-block:: python

   class CodeGenerator(Protocol):
       async def generate_code(
           self,
           request: PythonExecutionRequest,
           error_chain: list[str]
       ) -> str:
           """Generate Python code based on request and error feedback."""
           ...

This enables:

- Easy creation of custom generators
- Clean separation from the executor service
- Runtime type checking
- Error-aware iterative improvement

**Learn More:**

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: 📚 Python Service Overview
      :link: service-overview
      :link-type: doc
      :class-header: bg-secondary text-white
      :shadow: md

      Complete service guide: architecture, integration, generators, and configuration

   .. grid-item-card:: 🤖 Basic LLM Generator
      :link: generator-basic
      :link-type: doc
      :class-header: bg-info text-white
      :shadow: md

      Simple single-pass LLM generation

   .. grid-item-card:: 🧠 Claude Code Generator
      :link: generator-claude
      :link-type: doc
      :class-header: bg-primary text-white
      :shadow: md

      Multi-phase agentic reasoning with codebase learning

   .. grid-item-card:: 🧪 Mock Generator
      :link: generator-mock
      :link-type: doc
      :class-header: bg-success text-white
      :shadow: md

      Fast, deterministic testing without API calls

Quick Start
===========

Minimal configuration to get started:

.. code-block:: yaml

   # config.yml
   osprey:
     execution:
       code_generator: "basic"  # or "claude_code", "mock"
       execution_method: "container"  # or "local"

       generators:
         basic:
           model_config_name: "python_code_generator"
         claude_code:
           profile: "fast"  # fast (DEFAULT, single-phase) | robust (multi-phase)
         mock:
           behavior: "success"

       container:
         jupyter_host: "localhost"
         jupyter_port: 8888

**Choose Your Generator:**

- **Basic LLM**: Simple single-pass generation - for self-hosted models and minimal setups
- **Claude Code**: Advanced multi-phase generation - learns from your codebase
- **Mock**: Testing only - instant, deterministic, no API calls

.. note::

   For complete service documentation including integration patterns, configuration reference, and creating custom generators, see :doc:`service-overview`.

.. toctree::
   :maxdepth: 2
   :hidden:

   service-overview
   generator-basic
   generator-claude
   generator-mock

See Also
========

:doc:`service-overview`
    Complete Python service guide with integration patterns and configuration

:doc:`../01_human-approval-workflows`
    Understanding the approval system integration

:doc:`../04_memory-storage-service`
    Integrate memory storage with Python execution

:doc:`../../api_reference/03_production_systems/03_python-execution`
    Complete Python execution API reference
