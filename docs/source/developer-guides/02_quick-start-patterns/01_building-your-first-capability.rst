==============================
Building Your First Capability
==============================

.. currentmodule:: osprey.base

Create a production-ready capability using the Osprey Framework's core patterns.

.. dropdown:: 📚 What You'll Learn
   :color: primary
   :icon: book

   **Key Concepts:**

   - Implement BaseCapability with @capability_node decorator
   - Work with AgentState and context storage using StateManager utilities
   - Register capabilities for framework discovery
   - Handle errors and streaming properly
   - Create CapabilityContext classes for data exchange

   **Prerequisites:** Python environment with Osprey Framework installed

   **Time Investment:** 30-45 minutes for complete implementation

Core Concepts
=============

Capabilities are business logic components that:

- Extend BaseCapability and use @capability_node decorator
- Store results in Pydantic context objects
- Return state updates for LangGraph integration

Implementation Steps
====================

Step 1: Create Context Class
----------------------------

.. code-block:: python
   :caption: applications/my_app/context_classes.py

   from typing import ClassVar
   from pydantic import Field
   from osprey.context.base import CapabilityContext

   class ProcessedDataContext(CapabilityContext):
       """Context for processed user query data."""

       CONTEXT_TYPE: ClassVar[str] = "PROCESSED_DATA"
       CONTEXT_CATEGORY: ClassVar[str] = "ANALYSIS_RESULTS"

       query: str = Field(..., description="Original user query")
       word_count: int = Field(..., description="Number of words")
       contains_numbers: bool = Field(..., description="Whether query contains numbers")

       def get_access_details(self, key: str) -> dict:
           return {
               "summary": f"Processed data for query: '{self.query}'",
               "word_count": self.word_count,
               "key": key
           }

       def get_summary(self, key: str) -> dict:
           return {
               "title": "Query Analysis Results",
               "content": f"Analyzed '{self.query}' - {self.word_count} words"
           }

Step 2: Implement Capability
----------------------------

.. code-block:: python
   :caption: applications/my_app/capabilities/data_processor.py

   from typing import Dict, Any
   from osprey.base import BaseCapability, capability_node
   from osprey.utils.logger import get_logger
   from osprey.utils.streaming import get_streamer
   from applications.my_app.context_classes import ProcessedDataContext

   logger = get_logger("data_processor")

   @capability_node
   class DataProcessorCapability(BaseCapability):
       """Process and analyze user data requests."""

       name = "data_processor"
       description = "Process and analyze user data requests"
       provides = ["PROCESSED_DATA"]
       requires = []

       async def execute(self) -> Dict[str, Any]:
           """Execute the capability's core business logic."""
           current_task = self.get_task_objective()
           streamer = get_streamer("my_app", self._state)

           try:
               streamer.status("Processing your request...")

               # Process the user query
               user_query = current_task.lower() if current_task else ""
               processed_data = {
                   "word_count": len(user_query.split()) if user_query else 0,
                   "contains_numbers": any(char.isdigit() for char in user_query),
               }

               # Create structured context
               context = ProcessedDataContext(
                   query=current_task,
                   word_count=processed_data["word_count"],
                   contains_numbers=processed_data["contains_numbers"]
               )

               # Store context and return state updates
               streamer.status("Processing completed!")
               return self.store_output_context(context)

           except Exception as e:
               logger.error(f"Processing error: {e}")
               raise

Helper Methods Reference
------------------------

Instance-based capabilities provide access to helper methods that eliminate boilerplate code:

**get_task_objective()** - Get the current task

.. code-block:: python

   async def execute(self) -> Dict[str, Any]:
       task = self.get_task_objective()
       # With custom default
       task = self.get_task_objective(default="unknown task")

**get_parameters()** - Access step parameters

.. code-block:: python

   async def execute(self) -> Dict[str, Any]:
       params = self.get_parameters()
       timeout = params.get('timeout', 30)
       precision = params.get('precision_ms', 1000)

**get_required_contexts()** - Auto-extract contexts based on requires field

.. code-block:: python

   requires = ["INPUT_DATA", ("TIME_RANGE", "single")]

   async def execute(self) -> Dict[str, Any]:
       # Dict access
       contexts = self.get_required_contexts()
       input_data = contexts["INPUT_DATA"]

       # Tuple unpacking (matches requires order)
       input_data, time_range = self.get_required_contexts()

**store_output_context()** - Store single output context

.. code-block:: python

   async def execute(self) -> Dict[str, Any]:
       result = process_data()
       context = ResultContext(data=result)
       return self.store_output_context(context)  # Type and key inferred automatically

**store_output_contexts()** - Store multiple contexts

.. code-block:: python

   async def execute(self) -> Dict[str, Any]:
       primary = PrimaryContext(data=results)
       metadata = MetadataContext(info=info)
       return self.store_output_contexts(primary, metadata)

.. note::

   **Advanced State Access:** If you need direct state access, use ``self._state`` and ``self._step`` (automatically injected by the ``@capability_node`` decorator).

Step 3: Register Your Capability
--------------------------------

.. code-block:: python
   :caption: applications/my_app/registry.py

   from osprey.registry import (
       RegistryConfigProvider, RegistryConfig,
       CapabilityRegistration, ContextClassRegistration
   )

   class MyAppRegistryProvider(RegistryConfigProvider):
       def get_registry_config(self) -> RegistryConfig:
           return RegistryConfig(
               capabilities=[
                   CapabilityRegistration(
                       name="data_processor",
                       module_path="applications.my_app.capabilities.data_processor",
                       class_name="DataProcessorCapability",
                       description="Process and analyze user data requests",
                       provides=["PROCESSED_DATA"],
                       requires=[]
                   )
               ],
               context_classes=[
                   ContextClassRegistration(
                       context_type="PROCESSED_DATA",
                       module_path="applications.my_app.context_classes",
                       class_name="ProcessedDataContext",
                       description="Structured results from user query analysis"
                   )
               ]
           )

Step 4: Test Your Capability
----------------------------

.. code-block:: python
   :caption: test_my_capability.py

   import asyncio
   from osprey.state import StateManager
   from applications.my_app.capabilities.data_processor import DataProcessorCapability

   async def test_capability():
       # Create test state
       state = StateManager.create_fresh_state("Hello world, this has 123 numbers!")

       # Execute capability (manually inject state for testing)
       capability = DataProcessorCapability()
       capability._state = state
       capability._step = {
           'context_key': 'test_key',
           'task_objective': 'Hello world, this has 123 numbers!',
           'parameters': {}
       }
       result = await capability.execute()

       print("Capability result:", result)
       print("Context data keys:", list(result.get("capability_context_data", {}).keys()))

   if __name__ == "__main__":
        asyncio.run(test_capability())

Essential Patterns
==================

Error Handling
--------------

Add custom error classification for domain-specific retry logic:

.. code-block:: python

   @staticmethod
   def classify_error(exc: Exception, context: dict) -> ErrorClassification:
       from osprey.base.errors import ErrorClassification, ErrorSeverity

       if isinstance(exc, (ConnectionError, TimeoutError)):
           return ErrorClassification(
               severity=ErrorSeverity.RETRIABLE,
               user_message="Temporary service issue, retrying...",
               metadata={"technical_details": str(exc)}
           )

       return ErrorClassification(
           severity=ErrorSeverity.CRITICAL,
           user_message=f"Processing error: {exc}",
           metadata={"technical_details": str(exc)}
       )

Streaming Updates
-----------------

Provide progress feedback during operations:

.. code-block:: python

   async def execute(self) -> Dict[str, Any]:
       streamer = get_streamer("my_app", self._state)

       streamer.status("Phase 1: Extracting data...")
       raw_data = await extract_data()

       streamer.status("Phase 2: Processing data...")
       processed_data = await process_data(raw_data)

       streamer.status("Phase 3: Storing results...")
       context = create_context(processed_data)

       return self.store_output_context(context)

Common Issues
=============

**"Capability not found in registry"**

Ensure exact name matching in registration:

.. code-block:: python

   # In capability class:
   name = "data_processor"

   # In registry:
   CapabilityRegistration(name="data_processor", ...)  # Must match exactly

**"execute method not found"**

Ensure execute is an async method (instance method recommended):

.. code-block:: python

   async def execute(self) -> Dict[str, Any]:  # Instance method (recommended)
       pass

.. note::

   **Legacy Pattern:** Static methods with ``@staticmethod`` decorator are still supported for backward compatibility, but instance methods are now the recommended approach as they provide access to helpful methods like ``self.get_task_objective()`` and ``self.store_output_context()``.

**"Context serialization error"**

Use only JSON-compatible types:

.. code-block:: python

   # ✅ Correct
   timestamp: str  # Use ISO string
   data: Dict[str, Any]

   # ❌ Wrong
   timestamp: datetime  # Not JSON serializable

Next Steps
==========

**Essential:**
- :doc:`03_running-and-testing` - Learn debugging and deployment
- :doc:`02_state-and-context-essentials` - Master state management

**Advanced:**
- :doc:`../03_core-framework-systems/02_context-management-system` - Advanced context patterns
- :doc:`../04_infrastructure-components/06_error-handling-infrastructure` - Comprehensive error handling

**API Reference:**
- :doc:`../../api_reference/01_core_framework/01_base_components` - BaseCapability documentation
- :doc:`../../api_reference/01_core_framework/02_state_and_context` - StateManager utilities