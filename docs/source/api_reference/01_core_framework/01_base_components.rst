===============
Base Components
===============

Foundation classes for capabilities and infrastructure nodes with LangGraph integration, plus decorators for seamless framework integration.

.. currentmodule:: osprey.base

Base Classes
============

BaseCapability
--------------

.. autoclass:: BaseCapability
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   .. rubric:: Required Class Attributes

   .. attribute:: name
      :type: str

      Unique capability identifier for registration and routing (required).

   .. attribute:: description
      :type: str

      Human-readable description for documentation and logging (required).

   .. attribute:: provides
      :type: List[str]

      List of context types this capability generates (default: []).

   .. attribute:: requires
      :type: List[str | tuple[str, Literal["single", "multiple"]]]

      List of context types this capability depends on. Supports cardinality constraints:

      - **Simple format:** ``["CONTEXT_TYPE"]`` - No cardinality constraint
      - **Tuple format:** ``[("CONTEXT_TYPE", "single")]`` - Must be single object (not list)
      - **Tuple format:** ``[("CONTEXT_TYPE", "multiple")]`` - Must be a list

      The tuple format enables type-safe context extraction with automatic validation (default: []).

      Example::

          requires = [
              "OPTIONAL_DATA",              # No constraint
              ("TIME_RANGE", "single"),     # Must be single object
              ("CHANNELS", "multiple")      # Must be a list
          ]

   .. rubric:: Abstract Methods

   .. automethod:: BaseCapability.execute

   .. rubric:: Optional Methods

   .. automethod:: BaseCapability.classify_error

   .. automethod:: BaseCapability.get_retry_policy

   .. rubric:: Helper Methods

   These methods are available in instance-based ``execute()`` implementations and simplify common operations:

   .. automethod:: BaseCapability.get_required_contexts

   .. automethod:: BaseCapability.get_parameters

   .. automethod:: BaseCapability.get_task_objective

   .. automethod:: BaseCapability.store_output_context

   .. automethod:: BaseCapability.store_output_contexts

   .. automethod:: BaseCapability.process_extracted_contexts

   .. rubric:: Template Methods

   .. automethod:: BaseCapability._create_orchestrator_guide

   .. automethod:: BaseCapability._create_classifier_guide

   .. rubric:: Properties

   .. autoproperty:: BaseCapability.orchestrator_guide

   .. autoproperty:: BaseCapability.classifier_guide

BaseInfrastructureNode
----------------------

.. autoclass:: BaseInfrastructureNode
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   .. rubric:: Required Class Attributes

   .. attribute:: name
      :type: str
      :no-index:

      Infrastructure node identifier for routing and logging (required).

   .. attribute:: description
      :type: str
      :no-index:

      Human-readable description for documentation and monitoring (required).

   .. rubric:: Abstract Methods

   .. automethod:: BaseInfrastructureNode.execute
      :no-index:

   .. rubric:: Error Handling Methods

   .. automethod:: BaseInfrastructureNode.classify_error
      :no-index:

   .. automethod:: BaseInfrastructureNode.get_retry_policy
      :no-index:

LangGraph Integration Decorators
================================

capability_node
---------------

.. autofunction:: capability_node

   **Validation Requirements:**

   The decorator validates that the decorated class implements:

   - ``name`` (str): Unique capability identifier
   - ``description`` (str): Human-readable description
   - ``execute()`` (async method): Main business logic

     - **Recommended:** Instance method (enables helper methods like ``get_required_contexts()``, ``get_task_objective()``, etc.)
     - **Legacy:** Static method with ``(state: AgentState, **kwargs)`` signature (backward compatibility)

   - ``classify_error()`` (static method): Error classification (inherited or custom)
   - ``get_retry_policy()`` (static method): Retry configuration (inherited or custom)

   **Instance vs Static Methods:**

   Instance methods (recommended) provide access to:

   - ``self.get_required_contexts()`` - Extract contexts based on ``requires`` field
   - ``self.get_parameters()`` - Access step parameters
   - ``self.get_task_objective()`` - Get current task
   - ``self.store_output_context()`` - Simplified context storage with type inference
   - ``self._state`` - Injected state (advanced usage)
   - ``self._step`` - Injected step (advanced usage)

   Static methods (legacy) require manual state management via ``StateManager`` and ``ContextManager``.

   **Infrastructure Features:**

   - Error classification with domain-specific recovery strategies
   - Manual retry system via router (no LangGraph retries)
   - State management with automatic state updates and step progression
   - Streaming support through LangGraph's streaming system
   - Development mode with raw error re-raising for debugging
   - Comprehensive timing and performance monitoring

   **Usage Example:**

   .. code-block:: python

      @capability_node
      class WeatherCapability(BaseCapability):
          name = "weather_data"
          description = "Retrieve current weather conditions"
          provides = ["WEATHER_DATA"]
          requires = [("LOCATION", "single")]

          async def execute(self) -> Dict[str, Any]:
              # Use helper methods for simplified state access
              location, = self.get_required_contexts()

              # Business logic implementation
              weather_info = fetch_weather(location)
              context = WeatherDataContext(data=weather_info)

              # Simplified storage with type inference
              return self.store_output_context(context)

infrastructure_node
--------------------

.. autofunction:: infrastructure_node

   **Validation Requirements:**

   The decorator validates that the decorated class implements:

   - ``name`` (str): Infrastructure node identifier
   - ``description`` (str): Human-readable description
   - ``execute()`` (async static method): Orchestration/routing logic
   - ``classify_error()`` (static method): Error classification (inherited or custom)
   - ``get_retry_policy()`` (static method): Retry configuration (inherited or custom)

   **Infrastructure Features:**

   - Conservative error handling with fast failure detection
   - System monitoring with comprehensive timing and performance tracking
   - LangGraph native integration with streaming, configuration, and checkpoints
   - Development mode support with raw error re-raising
   - Optional quiet mode for high-frequency routing operations
   - Fatal error handling with immediate termination for system-level failures

   **Usage Examples:**

   .. code-block:: python

      @infrastructure_node
      class TaskExtractionNode(BaseInfrastructureNode):
          name = "task_extraction"
          description = "Extract and structure user tasks"

          @staticmethod
          async def execute(state: AgentState, **kwargs) -> Dict[str, Any]:
              # Task extraction logic
              return {"task_current_task": extracted_task}

      @infrastructure_node(quiet=True)
      class RouterNode(BaseInfrastructureNode):
          name = "router"
          description = "Dynamic routing based on agent state"

          @staticmethod
          async def execute(state: AgentState, **kwargs) -> Dict[str, Any]:
              # Routing logic without verbose logging
              return {"control_next_node": next_node}

Supporting Types
================

Error Classification
--------------------

.. autoclass:: ErrorSeverity
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ExecutionError
   :members:
   :undoc-members:
   :show-inheritance:

Result Types
------------

.. autoclass:: ExecutionResult
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ExecutionRecord
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: CapabilityMatch
   :members:
   :undoc-members:
   :show-inheritance:

Planning Types
--------------

.. autoclass:: PlannedStep
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ExecutionPlan
   :members:
   :undoc-members:
   :show-inheritance:

Example System
==============

.. autoclass:: BaseExample
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: OrchestratorExample
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: OrchestratorGuide
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ClassifierExample
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: TaskClassifierGuide
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ClassifierActions
   :members:
   :undoc-members:
   :show-inheritance:

.. seealso::

   :doc:`02_state_and_context`
       State and context management systems used by components

   :doc:`03_registry_system`
       Registry system for component management

   :doc:`04_configuration_system`
       Configuration system for component settings

   :doc:`../../developer-guides/03_core-framework-systems/03_registry-and-discovery`
       Complete guide to component registration patterns