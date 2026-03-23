===================
Running and Testing
===================

.. currentmodule:: osprey.infrastructure

Learn essential patterns for running, testing, and debugging capabilities within the Osprey Framework.

.. dropdown:: 📚 What You'll Learn
   :color: primary
   :icon: book

   **Key Concepts:**

   - Using the Gateway as single entry point for message processing
   - Setting up and running the CLI interface for interactive testing
   - Understanding the complete initialization and execution flow
   - Essential debugging patterns with logging and streaming
   - Programmatic testing with Gateway architecture

   **Prerequisites:** Basic capability development knowledge

   **Time Investment:** 15-20 minutes for testing workflow understanding

Gateway Architecture
====================

The Gateway serves as the single entry point for all message processing. It operates outside the compiled LangGraph to handle meta-operations like state creation, approval detection, and slash command processing before execution begins.

Basic Usage Pattern
-------------------

.. code-block:: python

   from osprey.infrastructure.gateway import Gateway, GatewayResult
   from osprey.graph import create_graph
   from osprey.registry import initialize_registry, get_registry
   from langgraph.checkpoint.memory import MemorySaver

   async def process_message_example():
       # Initialize framework components
       initialize_registry()
       registry = get_registry()
       checkpointer = MemorySaver()
       graph = create_graph(registry, checkpointer=checkpointer)
       gateway = Gateway()

       # Create session configuration
       config = {
           "configurable": {
               "thread_id": "test_session_123",
               "user_id": "test_user"
           },
           "recursion_limit": 50
       }

       # Process message through Gateway
       user_input = "Hello, can you help me analyze some data?"
       result: GatewayResult = await gateway.process_message(
           user_input, graph, config
       )

       # Handle Gateway result
       if result.error:
           print(f"Error: {result.error}")
           return

       if result.slash_commands_processed:
           print(f"Processed commands: {result.slash_commands_processed}")

       # Execute the result
       if result.resume_command:
           final_state = await graph.ainvoke(result.resume_command, config=config)
       elif result.agent_state:
           final_state = await graph.ainvoke(result.agent_state, config=config)
       else:
           print("No action required")
           return

       # Access results
       print(f"Response: {final_state.get('ui_final_response', 'No response')}")
       print(f"Context: {list(final_state.get('capability_context_data', {}).keys())}")

Gateway Result Types
--------------------

The Gateway returns structured results:

.. code-block:: python

   @dataclass
   class GatewayResult:
       # For normal conversation flow
       agent_state: Optional[Dict[str, Any]] = None

       # For interrupt/approval flow
       resume_command: Optional[Command] = None

       # Processing metadata
       slash_commands_processed: List[str] = None
       approval_detected: bool = False

       # Error handling
       error: Optional[str] = None

   def handle_gateway_result(result: GatewayResult, graph, config):
       if result.error:
           print(f"❌ Error: {result.error}")
           return None

       if result.resume_command:
           return graph.ainvoke(result.resume_command, config=config)
       elif result.agent_state:
           return graph.ainvoke(result.agent_state, config=config)
       else:
           return None

CLI Interface
=============

The CLI interface provides interactive testing with real-time streaming.

Running the CLI
---------------

.. code-block:: bash

   # From the osprey directory:
   osprey chat

The CLI automatically handles:

- Framework initialization and registry setup
- Graph creation with checkpointing
- Session management with unique thread IDs
- Real-time streaming and approval workflows

Example CLI Session
-------------------

.. code-block:: text

   ╔═════════════════════════════════════════════════════════════════╗
   ║         Command Line Interface for the Osprey Framework         ║
   ╚═════════════════════════════════════════════════════════════════╝
   💡 Type 'bye' or 'end' to exit

   🔄 Initializing framework...
   ✅ Framework initialized! Thread ID: cli_session_a1b2c3d4

   👤 You: Hello, can you help me analyze some data?
   🔄 Processing: Hello, can you help me analyze some data?
   🔄 Extracting tasks from conversation...
   🔄 Analyzing task complexity and requirements...
   🤖 I can help you analyze data. What type of data would you like to analyze?

   👤 You: /approval:off Execute python code
   ✅ Processed commands: ['approval']
   🔄 Extracting actionable task from conversation
   🔄 Classification completed with 1 capabilities
   🔄 Execution plan created
   🔄 Generating Python code...
   🔄 Executing Python code...
   🔄 Python execution complete - 1.42s, 0 figures
   🔄 Generating response...
   🤖 I've executed the Python code successfully. Here are the results...

   👤 You: bye
   👋 Goodbye!

Programmatic Testing
====================

Create simple test scripts using the Gateway pattern:

.. code-block:: python
   :caption: test_basic.py

   import asyncio
   import sys
   import os

   # Add osprey to path
   sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

   from osprey.registry import initialize_registry, get_registry
   from osprey.graph import create_graph
   from osprey.infrastructure.gateway import Gateway
   from langgraph.checkpoint.memory import MemorySaver
   from osprey.utils.config import get_full_configuration

   async def test_capability():
       # Initialize framework
       initialize_registry()
       registry = get_registry()
       checkpointer = MemorySaver()
       graph = create_graph(registry, checkpointer=checkpointer)
       gateway = Gateway()

       # Create test configuration
       config = {
           "configurable": get_full_configuration(),
           "recursion_limit": 50
       }
       config["configurable"].update({
           "thread_id": "test_thread",
           "user_id": "test_user"
       })

       # Test message processing
       user_input = "What's the current weather?"
       result = await gateway.process_message(user_input, graph, config)

       if result.error:
           print(f"❌ Error: {result.error}")
           return False

       # Execute the agent
       if result.agent_state:
           final_state = await graph.ainvoke(result.agent_state, config=config)
           print(f"✅ Response: {final_state.get('ui_final_response', 'No response')}")
           return True

       return False

   if __name__ == "__main__":
       success = asyncio.run(test_capability())
       print(f"Test {'PASSED' if success else 'FAILED'}")

Debugging Essentials
====================

Debug Logging
-------------

.. code-block:: python

   from osprey.utils.logger import get_logger

   # Enable debug logging for components
   framework_logger = get_logger("debug")
   capability_logger = get_logger("my_capability")

Capability Debugging
--------------------

.. code-block:: python

   @capability_node
   class DebuggingCapability(BaseCapability):
       name = "debug_capability"
       description = "Capability with debug instrumentation"

       async def execute(self) -> Dict[str, Any]:
           # Get unified logger with automatic streaming
           logger = self.get_logger()

           logger.debug("=== Capability Execution Started ===")
           logger.debug(f"State keys: {list(self._state.keys())}")

           try:
               logger.status("Debug: Starting execution...")
               logger.debug("Business logic starting")

               # Your business logic with debug logging
               result_data = {"debug": True, "timestamp": "2024-01-01T00:00:00Z"}
               logger.debug(f"Processed data: {result_data}")

               # Store context
               context = DebugContext(
                   debug_info=result_data,
                   execution_path="debug_capability"
               )

               logger.debug("=== Execution Completed ===")
               logger.success("Debug capability completed")
               return self.store_output_context(context)

           except Exception as e:
               logger.error(f"Execution failed: {e}", exc_info=True)
               raise

Effective Logging and Streaming
--------------------------------

.. code-block:: python

   async def execute(self) -> Dict[str, Any]:
       logger = self.get_logger()

       try:
           # High-level status updates stream automatically
           logger.status("Starting data processing...")
           data = await fetch_data()

           # Detailed info for CLI logs only
           logger.info(f"Retrieved {len(data)} records")

           # Stream specific progress updates
           logger.status("Processing batch data...")
           processed = await process_data(data)

           # Success streams automatically
           logger.success("Data processing complete")
           context = create_context(processed)

           return self.store_output_context(context)

       except Exception as e:
           # Errors automatically stream to web UI
           logger.error(f"Processing failed: {e}")
           raise

Common Issues
=============

Framework Initialization
------------------------

**"Registry contains no nodes"**

.. code-block:: python

   # ✅ Correct initialization order
   from osprey.registry import initialize_registry, get_registry

   initialize_registry()  # Initialize first
   registry = get_registry()  # Then get instance

   # Verify registration
   all_nodes = registry.get_all_nodes()
   if not all_nodes:
       print("Warning: No nodes registered")

**"Graph creation fails"**

.. code-block:: python

   # Check for required nodes
   node_names = list(registry.get_all_nodes().keys())
   required_nodes = ["task_extraction", "classification", "orchestration"]

   missing = [node for node in required_nodes if node not in node_names]
   if missing:
       print(f"Missing required nodes: {missing}")

Capability Execution
--------------------

**"Capability not found during execution"**

.. code-block:: python

   # Check capability registration
   capabilities = registry.get_all_capabilities()
   print(f"Registered capabilities: {list(capabilities.keys())}")

   # Ensure name matches exactly
   @capability_node
   class MyCapability(BaseCapability):
       name = "my_capability"  # Must match registry name exactly

**"Context data not persisting"**

.. code-block:: python

   # ✅ Correct pattern
   return self.store_output_context(context)  # Stores and returns updates!

Configuration Issues
--------------------

**"Configuration not loading"**

.. code-block:: python

   from osprey.utils.config import get_full_configuration
   import os

   # Check environment variables
   print(f"Current application: {os.getenv('CURRENT_APPLICATION', 'Not set')}")

   # Check configuration loading
   try:
       config = get_full_configuration()
       print(f"Configuration loaded: {list(config.keys())}")
   except Exception as e:
       print(f"Configuration error: {e}")

Next Steps
==========

**Advanced Development:**
- :doc:`../03_core-framework-systems/03_registry-and-discovery` - Component registration patterns
- :doc:`../03_core-framework-systems/01_state-management-architecture` - Advanced state management
- :doc:`../04_infrastructure-components/06_error-handling-infrastructure` - Error handling strategies

**Production Deployment:**
- :doc:`../../api_reference/03_production_systems/05_container-management` - Container deployment
- :doc:`../../api_reference/03_production_systems/01_human-approval` - Approval workflows

**API Reference:**
- :doc:`../../api_reference/02_infrastructure/01_gateway` - Gateway documentation
- :doc:`../../api_reference/01_core_framework/02_state_and_context` - State management utilities
