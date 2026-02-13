==================
Basic LLM Generator
==================

The Basic LLM Generator provides straightforward, single-pass code generation using any OpenAI-compatible LLM. It makes one API call with a well-structured prompt and returns clean, executable Python code.

**Recommended Model:** We recommend **Claude Haiku 4.5** for an excellent balance of speed, accuracy, and cost-effectiveness.

Key Features
============

- **Error Learning**: On retries, includes structured feedback from the last 2 failed attempts (code, traceback, error analysis) so the LLM can learn and fix issues
- **Results Convention**: Generated code automatically stores outputs in a ``results`` dictionary for automatic saving
- **Capability Integration**: Incorporates ``capability_prompts`` to guide generation with domain-specific context
- **Flexible Prompting**: Supports ``expected_results`` structure templates for precise output formatting

How It Works
============

1. Builds a comprehensive prompt from your request (task, context, capability prompts)
2. If retrying, adds structured error feedback from previous attempts (last 2)
3. Makes a single LLM API call via Osprey's ``get_chat_completion``
4. Cleans markdown formatting artifacts and returns executable Python code

Prompt Structure
================

The generator builds prompts with the following components (in order):

1. **System Instructions** - Professional coding standards, import requirements, focus on simplicity
2. **Task Details** - ``task_objective`` and ``user_query`` from the request
3. **Expected Results** - Optional structure template (from ``expected_results`` field)
4. **Capability Prompts** - Domain-specific guidance (from ``capability_prompts`` list)
5. **Error Feedback** - Structured error details from previous attempts (if retrying)

.. dropdown:: Detailed Prompt Generation
   :color: info
   :icon: code

   The ``_build_code_generation_prompt`` method constructs prompts by combining:

   **Base Instructions:**

   .. code-block:: text

      You are an expert Python developer generating high-quality, executable code.

      === CODE GENERATION INSTRUCTIONS ===
      1. Generate complete, executable Python code
      2. Include all necessary imports at the top
      3. Use professional coding standards and clear variable names
      4. Add brief comments explaining complex logic
      5. STAY FOCUSED: Implement exactly what's requested
      6. Use provided context data when available (accessible via 'context' object)
      7. IMPORTANT: Store computed results in a dictionary variable named 'results'
      8. Generate ONLY the Python code, without markdown code blocks

   **Leveraging capability_prompts:**

   Capabilities can inject sophisticated, domain-specific guidance through the ``capability_prompts`` parameter. Here's a real example from the Data Analysis capability:

   .. code-block:: python

      # From data_analysis.py - sophisticated capability-specific prompts
      prompts = [
          f"""
          **STRUCTURED EXECUTION PLAN:**
          Phase 1: Data Preprocessing
            • Load and validate input data
            • Handle missing values
            → Output: Clean dataset ready for analysis

          Phase 2: Statistical Analysis
            • Calculate descriptive statistics
            • Compute correlations
            → Output: Statistical metrics
          """,

          f"""
          **REQUIRED OUTPUT FORMAT:**
          Your code must create a results dictionary matching this structure:
          {{
              "metrics": {{
                  "mean_value": "<float>",
                  "std_deviation": "<float>"
              }},
              "findings": {{
                  "anomalies": "<list>",
                  "patterns": "<string>"
              }}
          }}

          Replace all placeholder values with actual computed values.
          """,

          f"""
          **AVAILABLE CONTEXT DATA:**
          - ARCHIVER_DATA.historical_beam_current: Access via context.ARCHIVER_DATA.historical_beam_current
          - PV_VALUES.current_readings: Access via context.PV_VALUES.current_readings
          """
      ]

      request = PythonExecutionRequest(
          task_objective="Analyze beam stability patterns",
          capability_prompts=prompts,  # Inject domain-specific guidance
          expected_results={"metrics": {...}, "findings": {...}}
      )

   **Error Feedback (on retries):**

   When code fails, the generator learns from previous attempts:

   .. code-block:: text

      === PREVIOUS ATTEMPT(S) FAILED - LEARN FROM THESE ERRORS ===
      Analyze what went wrong and fix the root cause, not just symptoms.

      ============================================================
      **Attempt 1 - EXECUTION FAILED**

      **Code that failed:**
      ```python
      result = data['missing_column'].mean()
      ```

      **Error Type:** KeyError
      **Error:** 'missing_column'

      **Traceback:**
      ```
      KeyError: 'missing_column'
        File "script.py", line 5, in <module>
      ```
      ============================================================
      Generate IMPROVED code that fixes these issues.

Configuration
=============

.. code-block:: yaml

   execution:
     code_generator: "basic"
     generators:
       basic:
         model_config_name: "python_code_generator"

   models:
     python_code_generator:
       provider: "anthropic"
       model_id: "claude-haiku-4-5-20251001"
       temperature: 0.7
       max_tokens: 4096

**Inline Configuration (Alternative):**

.. code-block:: yaml

   execution:
     code_generator: "basic"
     generators:
       basic:
         provider: "anthropic"
         model_id: "claude-haiku-4-5-20251001"
         temperature: 0.7

Troubleshooting
===============

**API Authentication**

Ensure your API key is set:

.. code-block:: bash

   export ANTHROPIC_API_KEY='your-api-key'
   # or for OpenAI
   export OPENAI_API_KEY='your-api-key'

**Poor Code Quality**

- Use more specific ``capability_prompts`` in your request
- Provide relevant ``capability_context_data``
- For complex multi-step tasks, consider :doc:`generator-claude` (Claude Code SDK)

See Also
========

:doc:`service-overview`
    Complete Python service documentation

:doc:`generator-claude`
    Advanced code generation with Claude Code SDK

:doc:`generator-mock`
    Testing with mock generator

:doc:`index`
    Python Execution Service overview
