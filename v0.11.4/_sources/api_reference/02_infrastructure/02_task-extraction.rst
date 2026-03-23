Task Extraction
===============

.. currentmodule:: osprey.infrastructure.task_extraction_node

Infrastructure node that converts chat conversation history into focused, actionable tasks.

TaskExtractionNode
------------------

.. autoclass:: TaskExtractionNode
   :members:
   :inherited-members:
   :show-inheritance:
   :special-members: __init__

Supporting Functions
--------------------

.. autofunction:: build_task_extraction_prompt

Core Models
-----------

Task extraction uses models defined in the core framework:

.. seealso::

   :class:`~osprey.prompts.defaults.task_extraction.ExtractedTask`
       Structured output model for extracted tasks

   :class:`~osprey.base.BaseInfrastructureNode`
       Base class for infrastructure components

Registration
------------

Automatically registered as::

    NodeRegistration(
        name="task_extraction",
        module_path="osprey.infrastructure.task_extraction_node",
        function_name="TaskExtractionNode",
        description="Task extraction and processing"
    )

.. seealso::

   :doc:`../01_core_framework/05_prompt_management`
       Prompt customization system

   :doc:`../01_core_framework/03_registry_system`
       Component registration system

   :doc:`../../developer-guides/04_infrastructure-components/02_task-extraction-system`
       Implementation details and usage patterns
