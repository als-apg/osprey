============================
State and Context Management
============================

LangGraph-native state management with context persistence and sophisticated data management using Pydantic for automatic serialization, validation, and type safety.

.. currentmodule:: osprey.state

Core State System
=================

The state system provides a clean separation between persistent context data and execution-scoped fields that reset automatically between conversation turns, built on LangGraph's MessagesState foundation.

AgentState
----------

.. autoclass:: AgentState
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   .. rubric:: Key State Fields

   **Persistent Field:**

   .. attribute:: capability_context_data
      :type: Dict[str, Dict[str, Dict[str, Any]]]
      :no-index:

      Only field that persists across conversation turns. Uses three-level dictionary structure: {context_type: {context_key: {field: value}}}. Uses custom reducer :func:`merge_capability_context_data`.

   **Execution-Scoped Fields (reset each invocation):**

   .. attribute:: task_current_task
      :type: Optional[str]

      Current task being processed.

   .. attribute:: planning_execution_plan
      :type: Optional[ExecutionPlan]

      Current execution plan from orchestrator.

   .. attribute:: execution_step_results
      :type: Dict[str, Any]

      Results from completed execution steps.

   .. attribute:: control_has_error
      :type: bool

      Error state for manual retry handling.

   **Reactive Orchestration Fields (execution-scoped, react mode only):**

   .. attribute:: react_messages
      :type: list[dict]

      Accumulated LLM reasoning messages (decisions and observations) for the ReAct loop. Each entry has a ``role`` (``assistant`` or ``observation``) and ``content``.

   .. attribute:: react_step_count
      :type: int

      Safety counter tracking completed reactive steps. Used by the max iterations guard in ``_reactive_routing()``.

StateManager
------------

.. autoclass:: StateManager
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Key Methods

   .. automethod:: StateManager.create_fresh_state

   .. automethod:: StateManager.store_context

   .. automethod:: StateManager.get_current_step

   .. automethod:: StateManager.get_execution_plan

   .. automethod:: StateManager.register_figure

   .. automethod:: StateManager.register_notebook

Context Management System
=========================

.. currentmodule:: osprey.context

The context management system provides sophisticated functionality over dictionary data while maintaining LangGraph-compatible storage using Pydantic's built-in serialization capabilities.

ContextManager
--------------

.. autoclass:: ContextManager
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   .. rubric:: Key Methods

   **Context Storage and Retrieval:**

   .. automethod:: ContextManager.set_context

   .. automethod:: ContextManager.get_context

   .. automethod:: ContextManager.get_all_of_type

   .. automethod:: ContextManager.get_all

   **Context Extraction and Description:**

   .. automethod:: ContextManager.extract_from_step
      :no-index:

   .. automethod:: ContextManager.get_context_access_description

   .. automethod:: ContextManager.get_summaries

   **Data Access and Persistence:**

   .. automethod:: ContextManager.get_raw_data

   .. automethod:: ContextManager.save_context_to_file

CapabilityContext
-----------------

.. autoclass:: CapabilityContext
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   .. rubric:: Abstract Methods

   .. automethod:: CapabilityContext.get_access_details

   .. automethod:: CapabilityContext.get_summary

   .. rubric:: Properties

   .. autoproperty:: CapabilityContext.context_type

   .. rubric:: Class Constants

   .. attribute:: CONTEXT_TYPE
      :type: ClassVar[str]

      Context type identifier constant (must be overridden in subclasses).

   .. attribute:: CONTEXT_CATEGORY
      :type: ClassVar[str]

      Context category identifier constant (must be overridden in subclasses).

Unified Data Structure
======================

Both state and context systems use a three-level dictionary structure optimized for LangGraph serialization and efficient data management::

    {
        context_type: {
            context_key: {
                field: value,
                ...
            }
        }
    }

**Structure Components:**

- **context_type**: Defined by the CapabilityContext class's ``CONTEXT_TYPE`` constant (e.g., "PV_ADDRESSES", "ANALYSIS_RESULTS")
- **context_key**: Assigned by the orchestrator node during execution planning (e.g., "step_1", "beam_analysis_20240115")
- **field/value pairs**: User-defined data fields specific to the CapabilityContext subclass implementation

This structure enables efficient context storage, retrieval, and merging while maintaining compatibility with LangGraph's checkpointing system. The orchestrator coordinates context keys across execution steps, while capabilities define the context types and field structures.

Context Utilities
=================

.. currentmodule:: osprey.context

Context Loading
---------------

.. autofunction:: load_context

Namespace Access
----------------

.. autoclass:: ContextNamespace
   :members:
   :undoc-members:
   :show-inheritance:

   Provides dot notation access to context objects within a specific context type.

State Utilities
===============

.. currentmodule:: osprey.state

Custom Reducer Functions
------------------------

.. autofunction:: merge_capability_context_data

Event Creation Utilities
------------------------

.. autofunction:: create_status_update

.. autofunction:: create_progress_event

Execution Tracking
------------------

.. autofunction:: get_execution_steps_summary

Type Aliases
============

.. data:: StateUpdate
   :type: Dict[str, Any]

   Type alias for LangGraph state update dictionaries returned by capabilities and infrastructure nodes.

.. seealso::

   :class:`osprey.base.planning.ExecutionPlan`
       Execution planning structures used in state

   :class:`osprey.registry.RegistryManager`
       Component registry that manages context classes

   :doc:`03_registry_system`
       Registry system for component management

   :doc:`../../developer-guides/03_core-framework-systems/02_context-management-system`
       Complete guide to context management patterns

   :doc:`../../developer-guides/03_core-framework-systems/01_state-management-architecture`
       Complete guide to state management patterns
