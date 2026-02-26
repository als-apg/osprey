Execution Control
=================

Infrastructure components that control the flow of execution including routing decisions, error handling, and conditional logic.

Router Node
-----------

.. currentmodule:: osprey.infrastructure.router_node

.. autoclass:: RouterNode
   :members:
   :inherited-members:
   :show-inheritance:
   :special-members: __init__

.. autofunction:: router_conditional_edge

.. note::

   When ``orchestration_mode`` is set to ``react``, ``router_conditional_edge`` delegates to the
   internal ``_reactive_routing()`` function which implements reactive-specific routing logic
   (max iterations guard, execution plan dispatch, routing errors back to
   ``reactive_orchestrator``). See :doc:`../../developer-guides/04_infrastructure-components/03_classification-and-routing`
   for details on both routing modes.

Error Node
----------

.. currentmodule:: osprey.infrastructure.error_node

.. autoclass:: ErrorNode
   :members:
   :inherited-members:
   :show-inheritance:
   :special-members: __init__

.. autoclass:: ErrorType
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ErrorContext
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Core Models
-----------

Execution control uses models defined in the core framework:

.. seealso::

   :class:`~osprey.base.ErrorClassification`
       Error classification system

   :class:`~osprey.base.ErrorSeverity`
       Error severity levels

   :class:`~osprey.base.BaseInfrastructureNode`
       Base class for infrastructure components

Registration
------------

**RouterNode** is automatically registered as::

    NodeRegistration(
        name="router",
        module_path="osprey.infrastructure.router_node",
        function_name="RouterNode",
        description="Central routing decision authority"
    )

**ErrorNode** is automatically registered as::

    NodeRegistration(
        name="error",
        module_path="osprey.infrastructure.error_node",
        function_name="ErrorNode",
        description="Error response generation"
    )

.. seealso::

   :doc:`../01_core_framework/05_prompt_management`
       Prompt customization system

   :doc:`../../developer-guides/04_infrastructure-components/06_error-handling-infrastructure`
       Implementation details and usage patterns
