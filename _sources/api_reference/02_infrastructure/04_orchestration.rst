Orchestration
=============

Infrastructure nodes that create and execute plans from active capabilities and task requirements. Osprey supports two orchestration modes: **plan-first** (default) and **reactive** (ReAct pattern).

Plan-First Orchestration
------------------------

.. currentmodule:: osprey.infrastructure.orchestration_node

OrchestrationNode
~~~~~~~~~~~~~~~~~

Creates complete execution plans upfront with native LangGraph interrupt support for human-in-the-loop approval.

.. autoclass:: OrchestrationNode
   :members:
   :inherited-members:
   :show-inheritance:
   :special-members: __init__

Shared Validation
~~~~~~~~~~~~~~~~~

.. autofunction:: validate_single_step

   Used by both ``OrchestrationNode`` and ``ReactiveOrchestratorNode`` to validate individual execution steps against the registry and available context keys.

Reactive Orchestration (ReAct)
------------------------------

.. currentmodule:: osprey.infrastructure.reactive_orchestrator_node

ReactiveOrchestratorNode
~~~~~~~~~~~~~~~~~~~~~~~~

Step-by-step orchestration using the ReAct (Reasoning + Acting) pattern. Generates one ``PlannedStep`` at a time, observing capability results between steps.

.. autoclass:: ReactiveOrchestratorNode
   :members:
   :inherited-members:
   :show-inheritance:
   :special-members: __init__

Core Models
-----------

Orchestration uses models defined in the core framework:

.. seealso::

   :class:`~osprey.base.ExecutionPlan`
       Structured execution plan model

   :class:`~osprey.base.PlannedStep`
       Individual step within execution plans

   :class:`~osprey.base.BaseInfrastructureNode`
       Base class for infrastructure components

Approval System Integration
---------------------------

.. currentmodule:: osprey.approval.approval_system

.. autofunction:: create_plan_approval_interrupt

.. autofunction:: clear_approval_state

.. autofunction:: create_approval_type

Registration
------------

**OrchestrationNode** is automatically registered as::

    NodeRegistration(
        name="orchestrator",
        module_path="osprey.infrastructure.orchestration_node",
        function_name="OrchestrationNode",
        description="Execution planning and orchestration"
    )

**ReactiveOrchestratorNode** is automatically registered as::

    NodeRegistration(
        name="reactive_orchestrator",
        module_path="osprey.infrastructure.reactive_orchestrator_node",
        function_name="ReactiveOrchestratorNode",
        description="Reactive step-by-step orchestration using ReAct pattern"
    )

.. seealso::

   :doc:`../01_core_framework/05_prompt_management`
       Prompt customization system

   :doc:`../03_production_systems/01_human-approval`
       Complete approval system architecture

   :doc:`../../developer-guides/01_understanding-the-framework/04_orchestration-architecture`
       Orchestration modes overview and architecture

   :doc:`../../developer-guides/04_infrastructure-components/04_orchestrator-planning`
       Implementation details and usage patterns
