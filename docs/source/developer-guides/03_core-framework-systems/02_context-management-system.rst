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

Context Management Patterns
===========================

The framework provides two approaches to context management:

1. **Automatic (Recommended):** Use BaseCapability helper methods
2. **Manual (Advanced):** Direct ContextManager usage

Automatic Context Management
----------------------------

**Recommended for most capabilities.** Uses the ``requires`` field for declarative dependency management.

The framework provides helper methods that eliminate boilerplate for context extraction and storage. This is the **recommended pattern** for all capabilities - it's simple, type-safe, and maintainable.

Basic Pattern
^^^^^^^^^^^^^

Two ways to access extracted contexts:

**Pattern 1: Tuple Unpacking (Recommended for most cases)**

.. code-block:: python

   @capability_node
   class AnalysisCapability(BaseCapability):
       name = "analysis"
       provides = ["ANALYSIS_RESULTS"]
       requires = ["INPUT_DATA"]  # Declarative dependencies

       async def execute(self) -> Dict[str, Any]:
           # Elegant tuple unpacking (note the trailing comma for single context)
           input_data, = self.get_required_contexts()

           # Your business logic
           results = analyze(input_data)

           # Automatic storage with type inference
           return self.store_output_context(
               AnalysisResults(data=results)
           )

**Pattern 2: Dict Access (Use for optional/soft contexts)**

.. code-block:: python

   @capability_node
   class AnalysisCapability(BaseCapability):
       name = "analysis"
       provides = ["ANALYSIS_RESULTS"]
       requires = ["PRIMARY_DATA", "OPTIONAL_METADATA"]

       async def execute(self) -> Dict[str, Any]:
           # Soft mode - at least one required, use dict access with .get()
           contexts = self.get_required_contexts(constraint_mode="soft")

           primary = contexts.get("PRIMARY_DATA")
           metadata = contexts.get("OPTIONAL_METADATA")  # May be None

           # Handle optional metadata
           results = analyze(primary, metadata=metadata if metadata else {})
           return self.store_output_context(AnalysisResults(data=results))

**When to use each:**

- **Tuple unpacking**: Default choice - cleaner and more Pythonic for production code
- **Dict access**: Use when you need ``.get()`` for optional contexts or soft mode

**Key Benefits:**

- No manual ``StateManager`` or ``ContextManager`` instantiation
- Automatic validation of required contexts
- Type-safe context extraction
- Cleaner, more readable code

Tuple Unpacking (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Tuple unpacking works for both single and multiple contexts (matches ``requires`` field order):

.. code-block:: python

   # Single context - note the trailing comma!
   requires = ["INPUT_DATA"]

   async def execute(self) -> Dict[str, Any]:
       input_data, = self.get_required_contexts()  # Trailing comma unpacks single item

       results = process(input_data)
       return self.store_output_context(ResultContext(data=results))

.. code-block:: python

   # Multiple contexts - cleaner than dict access
   requires = ["INPUT_DATA", "TIME_RANGE"]

   async def execute(self) -> Dict[str, Any]:
       input_data, time_range = self.get_required_contexts()  # Order matches requires

       results = process(input_data, time_range)
       return self.store_output_context(ResultContext(data=results))

Soft vs Hard Constraints
^^^^^^^^^^^^^^^^^^^^^^^^^

Control whether ALL contexts are required or AT LEAST ONE:

.. code-block:: python

   requires = ["OPTIONAL_DATA", "OTHER_OPTIONAL"]

   async def execute(self) -> Dict[str, Any]:
       # Hard mode (default): ALL contexts required - use tuple unpacking
       data1, data2 = self.get_required_contexts(constraint_mode="hard")

       # Soft mode: AT LEAST ONE context required - use dict access
       contexts = self.get_required_contexts(constraint_mode="soft")
       optional = contexts.get("OPTIONAL_DATA")  # May be None

**When to use each mode:**

- ``hard`` (default): Use when all contexts are mandatory → **use tuple unpacking**
- ``soft``: Use for optional dependencies → **use dict access with .get()**

.. note::

   **Tuple unpacking with soft mode** is unreliable because you don't know how many contexts will be returned. Always use dict access for soft mode.

Multiple Output Contexts
^^^^^^^^^^^^^^^^^^^^^^^^^

Store multiple contexts in a single operation:

.. code-block:: python

   async def execute(self) -> Dict[str, Any]:
       primary = PrimaryContext(data=main_results)
       metadata = MetadataContext(info=processing_info)
       stats = StatisticsContext(stats=statistics)

       return self.store_output_contexts(primary, metadata, stats)

Cardinality Constraints
^^^^^^^^^^^^^^^^^^^^^^^^

Control whether extracted contexts are single objects or lists using tuple format in the ``requires`` field.

**Why Cardinality Matters**

Without cardinality constraints, you must check types at runtime:

.. code-block:: python

   # ❌ Without cardinality - runtime uncertainty
   requires = ["TIME_RANGE"]

   async def execute(self) -> Dict[str, Any]:
       contexts = self.get_required_contexts()
       time_range = contexts["TIME_RANGE"]

       # Is it a list or single object? Must check!
       if isinstance(time_range, list):
           time_range = time_range[0]  # Take first?

       start_time = time_range.start  # Could fail if it's a list!

With cardinality constraints, types are guaranteed:

.. code-block:: python

   # ✅ With cardinality - compile-time guarantee
   requires = [("TIME_RANGE", "single")]

   async def execute(self) -> Dict[str, Any]:
       time_range, = self.get_required_contexts()
       start_time = time_range.start  # Guaranteed to work!

**Single Cardinality**

Use ``"single"`` when you need to **enforce** that exactly one instance is provided:

.. code-block:: python

   requires = [("TIME_RANGE", "single")]

   async def execute(self) -> Dict[str, Any]:
       time_range, = self.get_required_contexts()
       # time_range is guaranteed to be a single object (validation fails if list)

**When to use:** When you need to enforce that the orchestrator provides exactly one instance. The framework will **raise ValueError** if a list is provided.

**Important:** Use ``"single"`` only when you truly need single-instance enforcement. For contexts that could be either single or list, omit the cardinality constraint and handle both cases.

**Multiple Cardinality**

Use ``"multiple"`` when you need to **enforce** that multiple items are provided (e.g., for comparison):

.. code-block:: python

   # Example: Comparing two datasets requires both
   requires = [("DATASETS_TO_COMPARE", "multiple")]

   async def execute(self) -> Dict[str, Any]:
       datasets, = self.get_required_contexts()
       # datasets is guaranteed to be a list with 2+ items (validation fails otherwise)

       comparison_result = compare(datasets[0], datasets[1])
       return self.store_output_context(ComparisonContext(result=comparison_result))

**When to use:** Rarely! Only when you truly need to enforce multiple instances (e.g., comparing two things). The framework will **raise ValueError** if only a single instance is provided.

**Common Pattern - Flexible Handling Instead:**

Most of the time, you want to handle both single and multiple contexts flexibly using ``process_extracted_contexts()`` hook:

.. code-block:: python

   # NO cardinality constraint - handle both cases
   requires = ["CHANNEL_ADDRESSES"]

   async def execute(self) -> Dict[str, Any]:
       # Get contexts (could be single object or list)
       channels_to_retrieve, = self.get_required_contexts()
       # After process_extracted_contexts, this is a flat list of channel strings

       data = retrieve_data(channels_to_retrieve)
       return self.store_output_context(DataContext(data=data))

   def process_extracted_contexts(self, contexts):
       """Hook to normalize CHANNEL_ADDRESSES into flat list."""
       channels_raw = contexts["CHANNEL_ADDRESSES"]

       if isinstance(channels_raw, list):
           # Multiple contexts - flatten into single list
           channels_flat = []
           for ctx in channels_raw:
               channels_flat.extend(ctx.channels)
           contexts["CHANNEL_ADDRESSES"] = channels_flat
       else:
           # Single context - extract channel list
           contexts["CHANNEL_ADDRESSES"] = channels_raw.channels

       return contexts

**Best Practices:**

.. code-block:: python

   # ✅ Recommended: Flexible with normalization hook
   requires = [
       ("TIME_RANGE", "single"),      # Enforce exactly one
       "CHANNEL_ADDRESSES"            # Flexible - handle in process_extracted_contexts
   ]

   def process_extracted_contexts(self, contexts):
       # Normalize CHANNEL_ADDRESSES to flat list
       # (See example in Multiple Cardinality section above)
       ...

   # ✅ Also good: Simple single constraint when appropriate
   requires = [("TIME_RANGE", "single")]

   # ⚠️ Rare: Only for enforcing comparisons
   requires = [("DATASETS_TO_COMPARE", "multiple")]

   # ❌ Avoid: Too strict for most cases
   requires = [("CHANNEL_ADDRESSES", "multiple")]
   # Better to omit constraint and use process_extracted_contexts hook

**Guidelines:**

1. Use ``"single"`` when you need to enforce exactly one instance
2. Use NO constraint (default) for flexible contexts - normalize in ``process_extracted_contexts()``
3. Use ``"multiple"`` only when truly enforcing 2+ items (e.g., comparisons)
4. Document your normalization logic in ``process_extracted_contexts()`` docstring

Manual Context Management
--------------------------

Use direct ContextManager for advanced scenarios.

**When to Use Manual:**

- Complex conditional logic for context selection
- Dynamic context type resolution
- Cross-step context aggregation
- Legacy code migration

**Complex Conditional Logic Example:**

.. code-block:: python

   async def execute(self) -> Dict[str, Any]:
       from osprey.context import ContextManager

       context_manager = ContextManager(self._state)

       # Complex conditional extraction
       if some_condition:
           data = context_manager.get_context("TYPE_A", "key_1")
       else:
           data = context_manager.get_all_of_type("TYPE_B")

       # Custom logic
       result = complex_processing(data)

       # Can still use automatic storage
       return self.store_output_context(result)

**Direct Access Patterns:**

.. code-block:: python

   from osprey.context import ContextManager

   context_manager = ContextManager(state)

   # Get specific context by type and key
   pv_data = context_manager.get_context("PV_ADDRESSES", "step_1")

   # Get all contexts of a specific type
   all_pv_data = context_manager.get_all_of_type("PV_ADDRESSES")

**Automatic Extraction with Cardinality Constraints (Recommended):**

.. code-block:: python

   @capability_node
   class AdvancedCapability(BaseCapability):
       requires = [("PV_ADDRESSES", "single")]
       provides = ["RESULTS"]

       async def execute(self) -> Dict[str, Any]:
           # Automatic extraction with cardinality constraints
           pv_addresses, = self.get_required_contexts()
           # The "single" constraint guarantees pv_addresses is not a list

           # Process data
           result = process(pv_addresses)

           # Automatic storage with type inference
           return self.store_output_context(result)

.. note::

   For advanced use cases requiring manual control, you can still use ``ContextManager.extract_from_step()`` and ``StateManager.store_context()`` directly.

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

State Management
----------------

.. code-block:: python

   async def execute(self) -> Dict[str, Any]:
       # ✅ Use helper methods
       task = self.get_task_objective()

       # Process data and create context
       result_context = MyDataContext(
           processed_data=processed_results,
           timestamp=datetime.now().isoformat()
       )

       # ✅ Store context and return state updates
       return self.store_output_context(result_context)

Context Design
--------------

.. code-block:: python

   class MyDataContext(CapabilityContext):
       # ✅ Use descriptive context types
       CONTEXT_TYPE: ClassVar[str] = "PROCESSED_DATA"  # Not "DATA"
       CONTEXT_CATEGORY: ClassVar[str] = "ANALYSIS_RESULTS"

       # ✅ Include validation and metadata
       data: Dict[str, Any] = Field(..., description="Processed data")
       timestamp: str = Field(..., description="ISO timestamp")
       count: int = Field(..., ge=0, description="Number of records")

**Context Class Design:**
- Keep context classes simple with minimal fields
- Use clear, descriptive field names and types
- Implement meaningful ``get_access_details()`` and ``get_summary()`` methods
- Use JSON-serializable types only

**Storage Patterns:**
- **Recommended:** Use ``self.store_output_context()`` helper method for automatic type/key inference
- **Advanced:** Use ``StateManager.store_context()`` for manual control
- Always return the result directly from capability execute methods
- Use meaningful context keys when using manual storage

**Retrieval Patterns:**
- **Recommended:** Use ``self.get_required_contexts()`` with ``requires`` field for declarative dependencies
- **Advanced:** Use ``ContextManager.extract_from_step()`` for complex conditional logic
- Handle missing context gracefully with appropriate error messages

Multi-Step Workflows
--------------------

Handle progressive context building (uses manual pattern for complex conditional logic):

.. code-block:: python

   async def execute(self) -> Dict[str, Any]:
       context_manager = ContextManager(self._state)

       # Get existing partial results
       partial_results = context_manager.get_context("PARTIAL_ANALYSIS", "working")

       if partial_results:
           current_data = partial_results.accumulated_data
       else:
           current_data = []

       # Add new data and store updated results
       new_data = await process_current_step()
       current_data.extend(new_data)

       updated_context = PartialAnalysisContext(
           accumulated_data=current_data,
           steps_completed=len(current_data),
           last_updated=datetime.now().isoformat()
       )

       # Can use helper method even with manual ContextManager
       return self.store_output_context(updated_context)

Common Issues
=============

**"Context not found"**

Handle missing context gracefully:

.. code-block:: python

   context_manager = ContextManager(self._state)
   required_data = context_manager.get_context("REQUIRED_TYPE", "key")

   if not required_data:
       return {"error": "Required data not available"}

**"Context serialization failed"**

Use only JSON-compatible types:

.. code-block:: python

   # ✅ Correct - JSON compatible
   timestamp: str = Field(..., description="ISO timestamp")
   data: Dict[str, Any] = Field(default_factory=dict)

   # ❌ Wrong - not serializable
   # timestamp: datetime = Field(...)
   # custom_obj: MyClass = Field(...)

**Serialization Errors:**

.. code-block:: python

   # ❌ Incorrect - using non-serializable types
   class BadContext(CapabilityContext):
       complex_object: SomeComplexClass  # Not JSON serializable

   # ✅ Correct - using JSON-compatible types
   class GoodContext(CapabilityContext):
       data: Dict[str, Any]  # JSON serializable
       timestamp: datetime   # Handled by Pydantic JSON encoders

**"State updates not persisting"**

Always return the result from context storage:

.. code-block:: python

   # ✅ Correct - return the state updates
   return self.store_output_context(context)

   # ❌ Wrong - updates are lost
   # self.store_output_context(context)
   # return {}

**Context Not Persisting:**

.. code-block:: python

   # ✅ Correct - using helper method (recommended)
   context_obj = MyContext(data='value')
   return self.store_output_context(context_obj)

   # ✅ Also correct - manual storage (advanced)
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