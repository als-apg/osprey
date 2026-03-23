============================
State and Context Essentials
============================

.. currentmodule:: osprey.state

The Osprey Framework supports multi-turn conversations, preserving relevant context across conversation turns through selective persistence of capability data. Master the essential state and context management patterns for Osprey Framework development.

.. dropdown:: 📚 What You'll Learn
   :color: primary
   :icon: book

   **Key Concepts:**

   - Understanding AgentState structure and selective persistence
   - Using StateManager for state operations and context storage
   - Working with ContextManager for data access
   - Creating CapabilityContext classes for type-safe data containers
   - Multi-step workflow patterns with progressive context building

   **Prerequisites:** Basic capability development knowledge

   **Time Investment:** 15-20 minutes for essential patterns

Framework Approach
===================

The Osprey Framework uses **selective persistence**:

- **Only context data persists** across conversation turns
- **All execution fields reset** automatically
- **LangGraph-native patterns** for optimal performance

Core Components
===============

**AgentState**
  LangGraph-native state extending MessagesState

**StateManager**
  Static utilities for state creation and context storage

**ContextManager**
  Interface for accessing capability context data

**CapabilityContext**
  Pydantic base class for type-safe data containers

AgentState Structure
====================

The AgentState uses a flat structure with logical prefixes:

.. code-block:: python

   class AgentState(MessagesState):
       # ===== PERSISTENT FIELD =====
       capability_context_data: Dict[str, Dict[str, Dict[str, Any]]]

       # ===== EXECUTION-SCOPED FIELDS (Reset each turn) =====

       # Agent control
       agent_control: Dict[str, Any]

       # Task processing
       task_current_task: Optional[str]
       task_depends_on_chat_history: bool
       task_depends_on_user_memory: bool

       # Planning
       planning_active_capabilities: List[str]
       planning_execution_plan: Optional[ExecutionPlan]
       planning_current_step_index: int

       # Execution
       execution_step_results: Dict[str, Any]
       execution_last_result: Optional[ExecutionResult]

       # Control flow
       control_needs_reclassification: bool
       control_retry_count: int
       control_has_error: bool

**Key insight:** Only ``capability_context_data`` persists across conversation turns.

Context Data Structure
----------------------

The ``capability_context_data`` field is the **data layer** of your Osprey agent. This is where capabilities save their results and retrieve data from other capabilities. Think of it as a shared workspace where capabilities communicate by storing and reading structured data.

**Key Concept:** Each capability stores its results as context objects that other capabilities can access. For example, a channel-finding capability stores found channel addresses, which an archiver-retrieval capability later reads to know which channels to query.

.. code-block:: python

   capability_context_data = {
       "WEATHER_DATA": {           # Context type (what kind of data)
           "step_0": {             # Context key (specific instance)
               "location": "San Francisco",
               "temperature": 18.5,
               "conditions": "Sunny",
               "timestamp": "2024-01-01T12:00:00Z"
           }
       },
       "PV_ADDRESSES": {           # Different context type
           "beam_current": {       # Different instance
               "pvs": ["SR:C01:BI:Current", "SR:C02:BI:Current"],
               "description": "Beam current monitoring PVs",
               "count": 2
           }
       }
   }


Creating Context Classes
------------------------

Context classes are Pydantic models that define the structure of data stored in the context system. Each context class must inherit from ``CapabilityContext`` and define required methods:

.. code-block:: python

   from typing import ClassVar
   from pydantic import Field
   from osprey.context.base import CapabilityContext

   class WeatherDataContext(CapabilityContext):
       """Context for weather data with validation."""

       # Framework integration
       CONTEXT_TYPE: ClassVar[str] = "WEATHER_DATA"
       CONTEXT_CATEGORY: ClassVar[str] = "LIVE_DATA"

       # Data fields with validation
       location: str = Field(..., description="Location name")
       temperature: float = Field(..., description="Temperature in Celsius")
       conditions: str = Field(..., description="Weather conditions")

       def get_access_details(self, key: str) -> dict:
           """Provide access details for LLM consumption."""
           return {
               "summary": f"Weather in {self.location}: {self.temperature}°C",
               "conditions": self.conditions
           }

       def get_summary(self, key: str) -> dict:
           """Provide human-readable summary."""
           return {
               "title": "Weather Data",
               "content": f"{self.location}: {self.temperature}°C, {self.conditions}"
           }

.. seealso::

   **For comprehensive context class documentation**, including validation, serialization, best practices, and advanced patterns, see:

   - :doc:`../03_core-framework-systems/02_context-management-system` - Complete context management guide
   - :doc:`../../api_reference/01_core_framework/02_state_and_context` - API reference

.. dropdown:: **Legacy Pattern Reference:** StateManager Direct Usage
   :color: secondary
   :icon: archive

   For reference only - the automatic context management pattern below is recommended for new code.

   **Creating Fresh State:**

   .. code-block:: python

      from osprey.state import StateManager

      # Create fresh state for new conversation
      state = StateManager.create_fresh_state(
          user_input="What's the weather in San Francisco?",
          current_state=None  # No previous state
      )

      # Create fresh state preserving context
      new_state = StateManager.create_fresh_state(
          user_input="How about New York?",
          current_state=previous_state  # Preserves context data
      )

   **Storing Context Data (Manual Pattern):**

   .. code-block:: python

      from osprey.state import StateManager
      from my_app.context_classes import WeatherDataContext

      @capability_node
      class WeatherCapability(BaseCapability):
          name = "weather_data"
          description = "Retrieve current weather conditions"
          provides = ["WEATHER_DATA"]

          async def execute(self) -> Dict[str, Any]:
              # Your business logic here
              weather_data = await fetch_weather_data()

              # Create structured context
              context = WeatherDataContext(
                  location=weather_data.location,
                  temperature=weather_data.temperature,
                  conditions=weather_data.conditions,
                  timestamp=datetime.now().isoformat()
              )

              # Store and return state updates using helper method
              return self.store_output_context(context)


Context Management
==================

This section covers the most common context management pattern - the basic usage you'll use in 90% of your capabilities. For comprehensive context management documentation, including advanced patterns, cardinality constraints, tuple unpacking strategies, and the ``process_extracted_contexts()`` hook, see: :doc:`../03_core-framework-systems/02_context-management-system`


Storing Context
---------------

Use the helper method to store context data:

.. code-block:: python

   @capability_node
   class AnalysisCapability(BaseCapability):
       name = "analysis"
       provides = ["ANALYSIS_RESULTS"]

       async def execute(self) -> Dict[str, Any]:
           # Your business logic
           results = analyze_data()

           # Create and store context
           return self.store_output_context(
               AnalysisResults(data=results)
           )

Retrieving Context
------------------

Use the ``requires`` field and helper method to retrieve context data:

.. code-block:: python

   @capability_node
   class ReportingCapability(BaseCapability):
       name = "reporting"
       requires = ["ANALYSIS_RESULTS"]  # Declare what you need
       provides = ["REPORT"]

       async def execute(self) -> Dict[str, Any]:
           # Get required context (note trailing comma for single context)
           analysis, = self.get_required_contexts()

           # Use the context data
           report = generate_report(analysis.data)

           return self.store_output_context(Report(content=report))

Multiple Contexts
-----------------

Handle multiple required contexts:

.. code-block:: python

   @capability_node
   class ComparisonCapability(BaseCapability):
       name = "comparison"
       requires = ["ANALYSIS_RESULTS", "BASELINE_DATA"]
       provides = ["COMPARISON"]

       async def execute(self) -> Dict[str, Any]:
           # Tuple unpacking matches requires field order
           analysis, baseline = self.get_required_contexts()

           comparison = compare(analysis.data, baseline.data)
           return self.store_output_context(Comparison(result=comparison))

Quick Tips
==========

**Context Storage:**

- Always return the result from ``self.store_output_context()``
- Use descriptive context type names in your context classes
- Keep context classes simple with JSON-serializable fields only

**Context Retrieval:**

- Use tuple unpacking for cleaner code: ``analysis, = self.get_required_contexts()``
- Note the trailing comma when unpacking a single context
- Multiple contexts: order matches ``requires`` field order

Common Issues
=============

**Forgot to return context storage result:**

.. code-block:: python

   # ❌ Wrong - updates are lost
   self.store_output_context(context)
   return {}

   # ✅ Correct - return the state updates
   return self.store_output_context(context)

**Using non-JSON types in context:**

.. code-block:: python

   # ❌ Wrong - not serializable
   timestamp: datetime = Field(...)

   # ✅ Correct - JSON compatible
   timestamp: str = Field(..., description="ISO timestamp")

**Forgot trailing comma in tuple unpacking:**

.. code-block:: python

   # ❌ Wrong - missing comma
   analysis = self.get_required_contexts()  # Returns dict, not object!

   # ✅ Correct - trailing comma for single context
   analysis, = self.get_required_contexts()

Next Steps
==========

**Continue Learning:**

- :doc:`03_running-and-testing` - Learn to test your state-aware capabilities

**Deep Dive (Recommended):**

- :doc:`../03_core-framework-systems/02_context-management-system` - **Complete context management guide** with advanced patterns, cardinality constraints, tuple unpacking strategies, the ``process_extracted_contexts()`` hook, multi-step workflows, and best practices

**Advanced:**

- :doc:`../03_core-framework-systems/01_state-management-architecture` - Complete state lifecycle and persistence patterns

**API Reference:**

- :doc:`../../api_reference/01_core_framework/02_state_and_context` - StateManager and ContextManager API documentation
