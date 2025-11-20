=========================
Context Management System
=========================

.. currentmodule:: osprey.context

The Osprey Framework's context management system provides type-safe data sharing between capabilities using Pydantic for automatic serialization and LangGraph-native dictionary storage.

.. dropdown:: 📚 What You'll Learn
   :color: primary
   :icon: book

   **Key Concepts:**

   - :class:`ContextManager` and :class:`CapabilityContext` fundamentals
   - Creating and registering context classes
   - Storing and retrieving context data in capabilities
   - :doc:`03_registry-and-discovery` integration for automatic context discovery
   - Best practices for type-safe data sharing

   **Prerequisites:** Understanding of Pydantic models and :doc:`01_state-management-architecture` (AgentState structure)

   **Time Investment:** 25-30 minutes for complete understanding

Overview
========

**Core Components:**

- :class:`ContextManager`: Primary interface for context operations with caching
- :class:`CapabilityContext`: Pydantic base class for all context objects
- :class:`StateManager`: Utilities for storing context data in agent state
- **Registry Integration**: Automatic context class discovery and validation

**Key Benefits:**

- **Type Safety**: Pydantic validation and type checking
- **LangGraph Native**: Dictionary storage compatible with checkpointing
- **Performance**: Object caching and efficient serialization

**Data Structure:**

Context data is stored in a three-level dictionary within the agent state's `capability_context_data` field:

.. code-block:: python

   capability_context_data = {
       context_type: {           # e.g., "PV_ADDRESSES", "PV_VALUES"
           context_key: {        # e.g., "step_1", "beam_current_data"
               field: value,     # Serialized Pydantic model fields
               ...
           }
       }
   }

Creating Context Classes
========================

Context classes define the structure of data shared between capabilities:

.. code-block:: python

   from osprey.context import CapabilityContext
   from typing import List, Optional, ClassVar

   class PVAddresses(CapabilityContext):
       """Context class for storing EPICS PV address discovery results."""

       # Class constants for registration
       CONTEXT_TYPE: ClassVar[str] = "PV_ADDRESSES"
       CONTEXT_CATEGORY: ClassVar[str] = "METADATA"

       # Data fields
       pvs: List[str]  # List of found PV addresses
       description: str  # Description of the PVs

       def get_access_details(self, key: str) -> Dict[str, Any]:
           return {
               "pvs": self.pvs,
               "total_available": len(self.pvs),
               "comments": self.description,
               "data_structure": "List of PV address strings",
               "access_pattern": f"context.{self.CONTEXT_TYPE}.{key}.pvs",
           }

       def get_summary(self) -> Dict[str, Any]:
           return {
               "type": "PV Addresses",
               "total_pvs": len(self.pvs),
               "pv_list": self.pvs,
               "description": self.description,
           }

**Requirements:**

1. Inherit from :class:`CapabilityContext`
2. Define ``CONTEXT_TYPE`` and ``CONTEXT_CATEGORY`` class constants
3. Implement ``get_access_details()`` and ``get_summary()`` methods
4. Use JSON-serializable field types only

Storing Context Data
====================

Capabilities store context data using ``StateManager.store_context()``:

.. code-block:: python

   @capability_node
   class PVAddressFindingCapability(BaseCapability):
       @staticmethod
       async def execute(state: AgentState, **kwargs) -> Dict[str, Any]:
           step = StateManager.get_current_step(state)

           # Perform capability logic
           found_pvs = await search_for_pvs(step.get('task_objective'))

           # Create context object
           pv_context = PVAddresses(
               pvs=found_pvs,
               description="Found PV addresses for beam current monitoring"
           )

           # Store context using StateManager
           return StateManager.store_context(
               state,
               "PV_ADDRESSES",              # context_type
               step.get('context_key'),     # context_key from execution plan
               pv_context                   # CapabilityContext object
           )

Retrieving Context Data
=======================

Capabilities retrieve context data through ``ContextManager``:

.. code-block:: python

   @capability_node
   class PVValueRetrievalCapability(BaseCapability):
       @staticmethod
       async def execute(state: AgentState, **kwargs) -> Dict[str, Any]:
           step = StateManager.get_current_step(state)
           context_manager = ContextManager(state)

           # Extract required contexts with cardinality constraints
           contexts = context_manager.extract_from_step(
               step, state,
               constraints=[("PV_ADDRESSES", "single")],
               constraint_mode="hard"
           )
           pv_addresses_context = contexts["PV_ADDRESSES"]

           # The "single" cardinality constraint guarantees pv_addresses_context
           # is not a list, eliminating the need for isinstance checks

           # Use the context data
           pv_list = pv_addresses_context.pvs
           pv_values = await read_pv_values(pv_list)

           # Store result context
           result_context = PVValues(pv_values=pv_values)
           return StateManager.store_context(
               state, "PV_VALUES", step.get('context_key'), result_context
           )

**Alternative Access Patterns:**

.. code-block:: python

   context_manager = ContextManager(state)

   # Direct context access (when you know the exact key)
   pv_data = context_manager.get_context("PV_ADDRESSES", "step_1")

   # Get all contexts of a specific type
   all_pv_data = context_manager.get_all_of_type("PV_ADDRESSES")

Registry Integration
====================

Context classes are automatically discovered through the registry system:

.. code-block:: python

   # In applications/als_assistant/registry.py
   context_classes=[
       ContextClassRegistration(
           context_type="PV_ADDRESSES",
           module_path="applications.als_assistant.context_classes",
           class_name="PVAddresses"
       ),
       ContextClassRegistration(
           context_type="PV_VALUES",
           module_path="applications.als_assistant.context_classes",
           class_name="PVValues"
       )
   ]

Best Practices
==============

**Context Class Design:**
- Keep context classes simple with minimal fields
- Use clear, descriptive field names and types
- Implement meaningful ``get_access_details()`` and ``get_summary()`` methods
- Use JSON-serializable types only

**Storage Patterns:**
- Always use ``StateManager.store_context()`` for storing context data
- Use meaningful context keys that won't conflict
- Return the result directly from capability execute methods

**Retrieval Patterns:**
- Use ``ContextManager.extract_from_step()`` for dependency validation
- Handle missing context gracefully with appropriate error messages

Common Issues
=============

**Serialization Errors:**

.. code-block:: python

   # ❌ Incorrect - using non-serializable types
   class BadContext(CapabilityContext):
       complex_object: SomeComplexClass  # Not JSON serializable

   # ✅ Correct - using JSON-compatible types
   class GoodContext(CapabilityContext):
       data: Dict[str, Any]  # JSON serializable
       timestamp: datetime   # Handled by Pydantic JSON encoders

**Context Not Persisting:**

.. code-block:: python

   # ✅ Correct - using StateManager utilities
   context_obj = MyContext(data='value')
   return StateManager.store_context(state, 'NEW_TYPE', 'key', context_obj)

**Performance:**
- ContextManager implements intelligent caching for performance optimization
- First access reconstructs objects; subsequent access returns cached objects
- Context objects are cached per ContextManager instance

**Context Window Management:**
- Use ``recursively_summarize_data()`` utility function for large nested data structures
- Automatically truncates large lists and dictionaries to prevent LLM context window overflow
- Available from ``osprey.context.context_manager`` for use in ``get_summary()`` methods

.. code-block:: python

   from osprey.context.context_manager import recursively_summarize_data

   def get_summary(self) -> Dict[str, Any]:
       """Human-readable summary with automatic data truncation."""
       return {
           "analysis_results": recursively_summarize_data(self.results),
           "type": "Analysis Context"
       }

.. seealso::
   :doc:`01_state-management-architecture`
       State management and context integration patterns
   :doc:`03_registry-and-discovery`
       Component registration and convention-based loading mechanisms