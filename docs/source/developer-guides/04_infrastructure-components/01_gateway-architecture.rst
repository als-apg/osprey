Gateway Architecture
====================

.. currentmodule:: osprey.infrastructure.gateway

.. dropdown:: 📚 What You'll Learn
   :color: primary
   :icon: book

   **Key Concepts:**

   - How Gateway centralizes all message processing logic
   - State lifecycle management and conversation handling
   - Slash command integration and approval workflow coordination
   - Interface implementation patterns

   **Prerequisites:** Understanding of :doc:`../03_core-framework-systems/01_state-management-architecture`

   **Time Investment:** 15 minutes for complete understanding

Core Concept
------------

Gateway serves as the **single entry point** for all message processing, eliminating interface duplication and ensuring consistent state management across CLI, web, and API interfaces. As an external coordinator, it operates outside the compiled LangGraph to handle preprocessing operations.

**Problem Solved:** Without centralized processing, each interface duplicates message logic, state creation, and approval handling.

**Solution:** All interfaces call ``Gateway.process_message()`` - Gateway handles preprocessing, interfaces handle presentation.

Architecture
------------

.. code-block:: python

   from osprey.infrastructure.gateway import Gateway

   # Universal pattern for all interfaces
   gateway = Gateway()
   result = await gateway.process_message(user_input, graph, config)

   # Execute based on result type
                   if result.resume_command:
       # Approval flow resumption
       response = await graph.ainvoke(result.resume_command, config=config)
                   elif result.agent_state:
       # Normal conversation turn
       response = await graph.ainvoke(result.agent_state, config=config)

Key Features
------------

**State Authority**
   Gateway is the only component that creates agent state, ensuring consistency.

**Slash Commands**
   Integrated with the centralized command system for ``/planning:on``, ``/approval:enabled``, ``/debug:on``, and performance bypass commands.

**Approval Integration**
   Automatic detection of approval/rejection responses with LangGraph interrupt handling.

**Interface Agnostic**
   Same processing logic for CLI, OpenWebUI, APIs, or custom interfaces.

Implementation Patterns
-----------------------

**Simple Interface Integration**

.. code-block:: python

   class MyInterface:
       def __init__(self):
           self.gateway = Gateway()
           self.graph = create_graph()

       async def handle_message(self, message: str) -> str:
           result = await self.gateway.process_message(message, self.graph, self.config)

           if result.error:
               return f"Error: {result.error}"

           # Execute and extract response
           execution_input = result.resume_command or result.agent_state
           response = await self.graph.ainvoke(execution_input, config=self.config)
           return self._extract_response(response)

**Streaming Interface Integration**

.. code-block:: python

   async def handle_streaming(self, message: str):
       result = await self.gateway.process_message(message, self.graph, self.config)

           if result.error:
           yield {"error": result.error}
               return

       # Stream execution
           execution_input = result.resume_command or result.agent_state
           async for chunk in self.graph.astream(execution_input, config=self.config):
           yield chunk

State Management
----------------

Gateway automatically handles:

- **Fresh state creation** for new conversation turns
- **Persistent field preservation** (execution history, user preferences)
- **Slash command application** before execution
- **Approval state injection** for interrupt resumption

**State Creation Pattern:**

.. code-block:: python

   # Gateway handles this automatically
   fresh_state = StateManager.create_fresh_state(
       user_input=cleaned_message,
       current_state=current_state  # Preserves persistent fields
   )

.. _slash-commands-section:
Slash Commands
--------------

Gateway parses and applies slash commands automatically:

**Planning and Execution Control:**

- ``/planning`` or ``/planning:on`` - Enable planning mode
- ``/planning:off`` - Disable planning mode (default)
- ``/approval:on`` - Enable approval workflows
- ``/approval:selective`` - Selective approval mode
- ``/debug:on`` - Enable debug logging

**Performance Optimization:**

- ``/task:off`` - :ref:`Bypass task extraction <bypass-task-extraction-section>` (use full chat history)
- ``/task:on`` - Enable task extraction (default)
- ``/caps:off`` - Bypass capability selection (activate all capabilities)
- ``/caps:on`` - Enable capability selection (default)

Commands are parsed from user input and applied to ``agent_control`` state before execution and can temporarily change bypass settings for the current conversation session, overriding system defaults.

.. dropdown:: 💡 Slash Command Examples
   :color: info
   :icon: terminal

   **Real CLI Sessions Demonstrating Different Modes**

   .. tab-set::

      .. tab-item:: Planning Mode
         :sync: planning

         **Human Approval Workflow Example**

         .. code-block:: text

            👤 You: /planning What's the weather in San Francisco?
            🔄 Processing: /planning What's the weather in San Francisco?
            ✅ Processed commands: ['planning']
            🔄 Extracting actionable task from conversation
            🔄 Analyzing task requirements...
            🔄 Generating execution plan...
            🔄 Requesting plan approval...

            ⚠️ **HUMAN APPROVAL REQUIRED** ⚠️

            **Planned Steps (2 total):**
            **Step 1:** Retrieve current weather conditions for San Francisco including temperature, weather conditions, and timestamp (current_weather)
            **Step 2:** Present the current weather information for San Francisco to the user in a clear and readable format (respond)

            **To proceed, respond with:**
            - **`yes`** to approve and execute the plan
            - **`no`** to cancel this operation

            👤 You: yes
            🔄 Processing: yes
            🔄 Resuming from interrupt...
            🔄 Using approved execution plan
            🔄 Executing current_weather... (10%)
            🔄 Weather retrieved: San Francisco - 18.0°C
            🔄 Executing respond... (10%)
            📊 Execution completed (execution_step_results: 2 records)

            🤖 Here is the current weather in San Francisco:
            As of today, the weather in San Francisco is **18.0°C and Partly Cloudy**.

         Planning mode provides transparent oversight of multi-step operations before execution begins.

      .. tab-item:: Performance Optimization
         :sync: performance

         **Bypass Mode Examples for Faster Response Times**

         .. code-block:: text


            # Example 1: Task extraction bypass for context-rich queries
            👤 You: /task:off Analyze the correlation we discussed earlier
            🔄 Processing: /task:off Analyze the correlation we discussed earlier
            ✅ Processed commands: ['task:off']
            🔄 Bypassing task extraction - using full conversation context
            🔄 Analyzing task requirements...
            🔄 Classification completed with 3 active capabilities
            🔄 Generating execution plan...

            # Example 2: Full bypass for quick status queries
            👤 You: /task:off /caps:off What's the current beam status?
            🔄 Processing: /task:off /caps:off What's the current beam status?
            ✅ Processed commands: ['task:off', 'caps:off']
            🔄 Bypassing task extraction - using full conversation context
            🔄 Bypassing capability selection - activating all capabilities
            🔄 Generating execution plan...



         **Performance Comparison:**

         - **Normal Mode**: Task Extraction → Classification → Orchestration → Execution (3 LLM calls)
         - **Task Bypass**: Classification → Orchestration → Execution (2 LLM call)
         - **Caps Bypass**: Task Extraction → Orchestration → Execution (2 LLM call)
         - **Full Bypass**: Orchestration → Execution (1 preprocessing LLM calls)

Approval Workflow Integration
-----------------------------

Gateway automatically detects approval responses using a two-tier detection system and creates resume commands:

**Explicit Yes/No Detection (Fast Path):**
   First checks for simple, explicit approval/rejection responses without LLM calls:

   - **Approval words**: ``yes``, ``y``, ``yep``, ``yeah``, ``ok``, ``okay`` (case-insensitive, with optional punctuation like ``.``, ``!``, ``?``)
   - **Rejection words**: ``no``, ``n``, ``nope``, ``nah``, ``cancel`` (case-insensitive, with optional punctuation)

   This provides instant, deterministic responses for common approval scenarios.

**LLM-Powered Approval Detection (Fallback):**
   For complex responses (e.g., "I think we should proceed"), uses the configured ``approval`` model from ``osprey.models`` to classify user responses.

**Approval Model Configuration:**
   Configured in ``src/osprey/config.yml`` under ``osprey.models.approval``. Only invoked for complex responses.

**Fail-Safe Behavior:**
   If LLM classification fails for any reason, the system defaults to "not approved" and logs a clear warning message.

**Resume Command Creation:**
   Gateway extracts interrupt payload and injects approval decision into agent state for processing.

Error Handling
--------------

Gateway provides graceful error handling:

.. code-block:: python

   # Gateway returns structured results
   @dataclass
   class GatewayResult:
       agent_state: Optional[Dict[str, Any]] = None
       resume_command: Optional[Command] = None
       error: Optional[str] = None
       slash_commands_processed: List[str] = None
       approval_detected: bool = False

Common error scenarios:
- Interrupt detection failures fall back to new message processing
- State access errors are handled gracefully
- Approval parsing failures provide clear guidance

Validation
----------

The Gateway pattern is validated by existing interfaces:

- **CLI Interface** (``interfaces/cli/direct_conversation.py``)
- **OpenWebUI Pipeline** (``interfaces/openwebui/main.py``)

Both follow the documented patterns exactly, providing real-world validation.

Best Practices
--------------

**Do:**
- Always use Gateway for message processing
- Handle both normal and approval flows
- Implement proper error handling for Gateway results

**Don't:**
- Create agent state manually in interfaces
- Duplicate approval detection logic
- Parse slash commands outside Gateway

Integration Example
-------------------

Complete CLI interface using Gateway:

.. code-block:: python

   class CLIInterface:
       def __init__(self):
           self.gateway = Gateway()
           self.graph = create_graph()
           self.config = {"configurable": {"thread_id": "cli_session"}}

       async def conversation_loop(self):
           while True:
               user_input = input("You: ").strip()
               if user_input.lower() in ['exit', 'quit']:
                   break

               # Process through Gateway
               result = await self.gateway.process_message(
                   user_input, self.graph, self.config
               )

               if result.error:
                   print(f"Error: {result.error}")
                   continue

               # Execute appropriate flow
               if result.resume_command:
                   response = await self.graph.ainvoke(result.resume_command, self.config)
               else:
                   response = await self.graph.ainvoke(result.agent_state, self.config)

               print(f"Agent: {self._extract_message(response)}")

Gateway Architecture provides the foundation for consistent, reliable message processing across all interfaces in the Osprey Framework.

.. seealso::

   :doc:`../../api_reference/02_infrastructure/01_gateway`
       API reference for Gateway classes and functions

   :doc:`../03_core-framework-systems/01_state-management-architecture`
       State lifecycle management and conversation handling

   :doc:`../05_production-systems/01_human-approval-workflows`
       Advanced approval workflow integration patterns