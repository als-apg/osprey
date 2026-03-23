Task Extraction
================

.. currentmodule:: osprey.infrastructure.task_extraction_node

.. dropdown:: 📚 What You'll Learn
   :color: primary
   :icon: book

   **Key Concepts:**

   - How conversations become structured, actionable tasks
   - Context compression and dependency detection
   - Data source integration during extraction
   - Task-specific context optimization

   **Prerequisites:** Understanding of :doc:`../03_core-framework-systems/05_message-and-execution-flow`

   **Time Investment:** 10 minutes for complete understanding

Core Problem
------------

**Challenge:** Agentic systems need conversational awareness without requiring every component to process entire chat histories.

**Traditional Approaches (Flawed):** Traditional methods often require processing the entire conversation history at every step, which can be slow and expensive. Some approaches use generic chat history summarization, but this often causes important task-relevant details to be lost. Other approaches provide no contextual information at all, leading to a loss of conversational awareness.

**Osprey Solution:** The framework implements a single-point context compression approach that extracts only task-relevant conversational context, enabling the Task Extraction system to efficiently convert human conversations into structured, actionable inputs for seamless processing.

Architecture
------------

Task extraction operates as the first pipeline step, converting raw conversations into structured tasks:

.. code-block:: python

   # Input: Full conversation history
   messages = [
       HumanMessage("Remember that data from yesterday?"),
       HumanMessage("Can you analyze the trends?")
   ]

   # Output: Structured task
   ExtractedTask(
       task="Analyze trends in the data from yesterday's conversation",
       depends_on_chat_history=True,
       depends_on_user_memory=False
   )

**Key Benefits:**
    - Downstream components receive compressed, actionable tasks
    - Conversational references are resolved ("that data" → specific context)
    - Dependencies are clearly identified for capability selection

Implementation
--------------

**TaskExtractionNode** processes conversations automatically:

.. code-block:: python

   @infrastructure_node
   class TaskExtractionNode(BaseInfrastructureNode):
       name = "task_extraction"
       description = "Task Extraction and Processing"

       @staticmethod
       async def execute(state: AgentState, **kwargs):
           # Get native LangGraph messages
           messages = state["messages"]

           # Retrieve external context if available
           retrieval_result = await data_manager.retrieve_all_context(state)

           # Extract task using LLM
           extracted_task = await _extract_task(messages, retrieval_result)

           return {
               "task_current_task": extracted_task.task,
               "task_depends_on_chat_history": extracted_task.depends_on_chat_history,
               "task_depends_on_user_memory": extracted_task.depends_on_user_memory
           }

.. _bypass-task-extraction-section:
.. dropdown:: Bypass LLM-based Task Extraction
   :color: secondary

   Task extraction can be bypassed either temporarily using the ``/task:off`` :ref:`slash command <slash-commands-section>`, or set as the default behavior via the :ref:`configuration system <performance-configuration-section>`. This allows you to skip LLM-based task extraction and use the full conversation history directly in downstream processing, based on your workflow needs.

   **Bypass Behavior:**
    - Skips LLM-based task extraction entirely
    - Passes full conversation history and retrieved datasource results as the "extracted task"
    - Sets dependency flags to True (assumes full context and chat history needed)
    - Maintains compatibility with downstream orchestration

   **When to Use Bypass Mode:**
    - Code R&D scenarios where full conversational context aids development
    - Short conversation histories where task extraction overhead exceeds benefits
    - Minimal external data scenarios where context compression isn't needed
    - High-throughput applications requiring reduced LLM call latency (trades orchestrator processing cost for extraction speed)

   **Advantages:**
    - Faster upstream pipeline (skips LLM-based task extraction)
    - No risk of losing conversational context or nuance

   **Disadvantages:**
    - Longer capability selection process (full conversation history included)
    - Longer orchestrator prompts (full conversation history included)
    - Slower plan generation (more tokens to process)
    - Potential for information overload in complex conversations



Structured Output
-----------------

Task extraction uses structured LLM generation for consistency:

.. code-block:: python

   class ExtractedTask(BaseModel):
       task: str = Field(description="Actionable task from conversation")
       depends_on_chat_history: bool = Field(description="Requires previous context")
       depends_on_user_memory: bool = Field(description="Requires stored user data")

**Task Compression Examples:**

.. code-block:: text

   User: "What's the weather like?"
   → Task: "Get current weather conditions"
   → Dependencies: history=False, memory=False

   User: "How does that compare to yesterday?"
   → Task: "Compare current weather to yesterday's weather data"
   → Dependencies: history=True, memory=False

   User: "Use my preferred location"
   → Task: "Get weather for the Bay Area"
   → Dependencies: history=False, memory=True

Data Source Integration
-----------------------

Task extraction automatically integrates available data sources:

.. code-block:: python

   # Automatic data retrieval during extraction
   try:
       data_manager = get_data_source_manager()
       retrieval_result = await data_manager.retrieve_all_context(request)
       logger.info(f"Retrieved data from {retrieval_result.total_sources_attempted} sources")
   except Exception as e:
       logger.warning(f"Data source retrieval failed, proceeding without external context: {e}")

**Graceful Degradation:** Task extraction continues without external data if sources are unavailable.

**Context Enrichment:** Available data sources can improve dependency detection and task clarity.

Error Handling
--------------

Task extraction includes retry policies optimized for LLM operations:

.. code-block:: python

   @staticmethod
   def classify_error(exc: Exception, context: dict):
       # Retry network/API timeouts
       if isinstance(exc, (ConnectionError, TimeoutError)):
           return ErrorClassification(
               severity=ErrorSeverity.RETRIABLE,
               user_message="Network timeout during task extraction, retrying..."
           )

       # Don't retry validation errors
       if isinstance(exc, (ValueError, TypeError)):
           return ErrorClassification(
               severity=ErrorSeverity.CRITICAL,
               user_message="Task extraction configuration error"
           )

**Retry Policy:** 3 attempts with exponential backoff for network issues.

Integration Patterns
--------------------

**Automatic Pipeline Integration:**
Task extraction runs automatically as the first infrastructure node in message processing.

**State Integration:**
Results are stored in agent state for downstream consumption:

.. code-block:: python

   # Downstream capabilities access extracted task
   current_task = state.get("task_current_task")
   needs_history = state.get("task_depends_on_chat_history", False)
   needs_memory = state.get("task_depends_on_user_memory", False)

**Prompt System Integration:**
Uses framework prompt builders for domain-specific extraction:

.. code-block:: python

   # Applications can customize with domain-specific builders
   class ALSTaskExtractionPromptBuilder(DefaultTaskExtractionPromptBuilder):
       def get_instructions(self) -> str:
           return "Extract tasks related to ALS accelerator operations..."

Configuration
-------------

Task extraction uses configuration:

.. code-block:: python

   # Osprey configuration
   task_extraction_config = get_model_config("task_extraction")

Troubleshooting
---------------

**Task Extraction Timeouts:**

- Built-in retry logic with exponential backoff
- Check network connectivity to LLM provider
- Verify model configuration in ``config.yml``

**Missing Dependencies:**

- Review prompt builder examples for dependency detection
- Check chat history formatting
- Consider application-specific prompt overrides

**Data Source Failures:**

- Task extraction gracefully degrades without external data
- Check data source provider registration
- Verify data source configuration

Performance Considerations
--------------------------

**Optimization Features:**

- Async processing with ``asyncio.to_thread()`` for non-blocking LLM calls
- Parallel data source retrieval
- Structured output for consistent parsing

**Memory Management:**

- Only essential task information persists in agent state
- Data retrieval results don't persist beyond extraction
- Native LangGraph message compatibility

.. seealso::

   :doc:`../../api_reference/02_infrastructure/02_task-extraction`
       API reference for task extraction classes and functions

   :doc:`../03_core-framework-systems/02_context-management-system`
       Context compression and dependency detection patterns

   :doc:`../04_orchestrator-planning`
       How tasks become execution plans

   :doc:`../03_core-framework-systems/05_message-and-execution-flow`
       Message processing pipeline architecture
