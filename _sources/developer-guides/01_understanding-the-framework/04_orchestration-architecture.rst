Orchestration Architecture
==========================

.. dropdown:: 📚 What You'll Learn
   :color: primary
   :icon: book

   **Key Concepts:**

   - How Osprey supports two orchestration modes: plan-first and reactive (ReAct)
   - When to choose each mode and their respective tradeoffs
   - How each mode fits into the processing pipeline

   **Prerequisites:** Understanding of :doc:`../03_core-framework-systems/01_state-management-architecture` and :doc:`../04_infrastructure-components/03_classification-and-routing`

   **Time Investment:** 5-10 minutes

Orchestration Modes Overview
============================

Osprey supports two orchestration modes that share the same capability and context infrastructure but differ in *when* execution decisions are made:

.. list-table:: Orchestration Mode Comparison
   :header-rows: 1
   :widths: 25 35 35

   * - Aspect
     - Plan-First (default)
     - Reactive (ReAct)
   * - **Decision timing**
     - Full plan created upfront before any execution
     - One step decided at a time, after observing results
   * - **LLM calls for planning**
     - Single call produces complete plan
     - One call per step (more total calls)
   * - **Adaptability**
     - Plan is fixed once created
     - Adapts to intermediate results and errors
   * - **Predictability**
     - High -- full plan visible before execution
     - Lower -- steps emerge dynamically
   * - **Error recovery**
     - Triggers reclassification or error node
     - Orchestrator observes error and decides next action
   * - **Approval workflows**
     - Supports plan approval via LangGraph interrupts
     - Per-step approval via LangGraph interrupts (each step approved before execution)
   * - **Configuration**
     - ``orchestration_mode: plan_first``
     - ``orchestration_mode: react``
   * - **Orchestrator node**
     - ``OrchestrationNode``
     - ``ReactiveOrchestratorNode``
   * - **Best for**
     - Structured tasks with known steps, production deployments
     - Exploratory tasks, error-prone pipelines, dynamic workflows

**Configuration:**

.. code-block:: yaml

   # config.yml
   execution_control:
     agent_control:
       orchestration_mode: plan_first  # Options: plan_first | react

Both modes route through the same ``task_extraction → classifier`` pipeline before reaching their respective orchestrator nodes. Capabilities are unaware of which mode generated their step.

Plan-First Orchestration
========================

The plan-first approach creates a complete execution plan before any capability runs. This is the default mode.

.. code-block:: text

   User Query → Task Extraction → Classifier → Orchestrator (full plan) → Capability 1 → Capability 2 → ... → Respond

The ``OrchestrationNode`` generates a validated ``ExecutionPlan`` containing all steps, then the router executes them deterministically in order. Key characteristics:

- **Single LLM call** produces the complete plan with step dependencies
- **Plan validation** catches hallucinated capabilities and missing context references before execution begins
- **Approval integration** allows human review of the full plan via LangGraph interrupts before any capability runs
- **Deterministic routing** follows the predetermined step sequence

If the plan contains invalid capabilities, the orchestrator triggers reclassification with specific error context rather than attempting partial execution.

Reactive Orchestration (ReAct)
==============================

The reactive orchestrator implements the **ReAct (Reasoning + Acting)** pattern, deciding one step at a time and observing results between steps.

.. code-block:: text

   User Query → Task Extraction → Classifier → Reactive Orchestrator ←──┐
                                                       ↓                │
                                                   Capability → Router ─┘
                                                                  ↓ (step = respond)
                                                                 END

Each invocation of the ``ReactiveOrchestratorNode`` produces exactly one ``PlannedStep``. After the capability executes, the router sends control back to the reactive orchestrator so it can observe the result and decide the next action. Key characteristics:

- **Observation-driven decisions** -- each iteration appends both the LLM's decision and the capability result to a ``react_messages`` history, giving the LLM full execution context
- **Auto-input resolution** -- unlike plan-first where input references are planned upfront, the reactive orchestrator matches a capability's ``requires`` list against available context data automatically
- **Self-correcting validation** -- before propagating errors, the LLM gets up to 2 additional attempts to fix validation failures (hallucinated capabilities, invalid context references)
- **Graceful error recovery** -- when a capability fails, the orchestrator observes the error and can retry a different approach, skip the step, or respond to the user

When to Choose Each Mode
========================

**Choose plan-first when:**

- Tasks have well-known steps (e.g., "find PV addresses and report them")
- You need human approval of the execution plan before anything runs
- Minimizing LLM calls matters (single planning call vs. one per step)
- Predictability and auditability are priorities

**Choose reactive when:**

- Next steps depend on previous outcomes (e.g., "investigate this issue")
- Tasks involve error-prone operations where dynamic recovery is valuable
- The workflow is exploratory and the full set of steps isn't known upfront
- You want the orchestrator to adapt to intermediate results

.. seealso::

   :doc:`../04_infrastructure-components/04_orchestrator-planning`
       Implementation details: plan generation, validation, data structures, reactive state fields, and routing

   :doc:`../03_core-framework-systems/01_state-management-architecture`
       Execution state handling

   :doc:`../05_production-systems/01_human-approval-workflows`
       Add human oversight to plans

   :doc:`../../api_reference/02_infrastructure/04_orchestration`
      API reference for orchestration classes and functions

   :doc:`../../api_reference/02_infrastructure/05_execution-control`
       Plan execution routing and deterministic flow control
