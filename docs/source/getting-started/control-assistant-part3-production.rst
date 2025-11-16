=============================================
Part 3: Integration & Deployment
=============================================


.. _step-5-observe-framework:

Step 5: Observe the Framework in Action
=======================================

Let's run a real multi-step query to see how the framework orchestrates complex tasks. We'll plot historical beam current data over 24 hours - a query that requires channel finding, time parsing, archiver data retrieval, and visualization.

This walkthrough uses mock services that simulate real control system hardware and archiver data. This lets you experience the complete framework workflow without requiring access to production systems. The mock services are explained in detail in :ref:`step-6-mock-services`, and switching from mock to production control systems is covered in :ref:`migrate-to-production`.

Start the Chat Interface
^^^^^^^^^^^^^^^^^^^^^^^^^

First, launch the interactive chat interface:

.. code-block:: bash

   # Direct command
   osprey chat

   # Or use the interactive menu
   osprey
   # Then select: [>] chat - Start CLI conversation

.. dropdown:: **Phase 0: Initialization and Component Loading** (happens during startup)
   :color: info

   When you start the chat interface (via ``osprey chat`` or through the interactive menu), the framework loads and prepares all system components before accepting any queries. The :doc:`registry system <../developer-guides/03_core-framework-systems/03_registry-and-discovery>` discovers and initializes capabilities, context types, and infrastructure components based on your application's registry configuration.

   **What's Loading:**

   Your ``my-control-assistant`` combines **framework-provided capabilities** (memory, Python execution, time parsing, response generation) with your **domain-specific capabilities** (channel finding, archiver retrieval, channel value reading). This modular approach means you only implement what's unique to your domain while leveraging battle-tested infrastructure.

   **Why This Matters:**

   The registry system enables :doc:`convention-over-configuration <../developer-guides/01_understanding-the-framework/02_convention-over-configuration>` patterns where the framework automatically discovers your capabilities without manual wiring. This makes the system extensible - adding new capabilities requires only implementing the class and registering it.

   **Terminal Output:**

   .. code-block:: text

      🔄 Initializing configuration...
      [11/11/25 11:20:35] INFO     Loading configuration from explicit path:
                                   <workspace>/my-control-assistant/config.yml
      🔄 Initializing framework...
      [11/11/25 11:20:36] INFO     Registry: Registry initialization complete!
                                   Components loaded:
                                      • 8 capabilities: memory, time_range_parsing, python, respond, clarify,
                                                        channel_finding, channel_value_retrieval, archiver_retrieval
                                      • 13 nodes (including 5 core infrastructure)
                                      • 6 context types: MEMORY_CONTEXT, TIME_RANGE, PYTHON_RESULTS,
                                                         CHANNEL_ADDRESSES, CHANNEL_VALUES, ARCHIVER_DATA
                                      • 1 data sources: core_user_memory
                                      • 1 services: python_executor
      ✅ Framework initialized! Thread ID: cli_session_dceb3a13

   **Further Reading:** :doc:`../developer-guides/03_core-framework-systems/03_registry-and-discovery`, :doc:`../developer-guides/01_understanding-the-framework/01_infrastructure-architecture`

Submit a Query
^^^^^^^^^^^^^^

Now that the framework is initialized, submit your query:

.. code-block:: text

   👤 You: plot the beam current over the last 24

Query Processing Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^

When you submit this query, the framework processes it through a sophisticated pipeline that transforms natural language into coordinated multi-step execution.

Phases 1-3: Task Analysis and Planning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Your query goes through three intelligent phases that transform natural language into a structured execution plan. Each phase builds on the previous one to ensure accurate task understanding and optimal execution strategy.

.. _phase1-task-extraction:

.. tab-set::

   .. tab-item:: Phase 1: Task Extraction

      The :doc:`task extraction system <../developer-guides/04_infrastructure-components/02_task-extraction-system>` analyzes your conversation to determine **context dependencies** - whether the current request needs information from previous messages or stored memories.

      **What's Happening:**

      For the request "plot the beam current over the last 24 hours", task extraction:

      1. Retrieves context from available :doc:`data sources <../developer-guides/05_production-systems/02_data-source-integration>` (currently ``core_user_memory`` for stored user memories, but applications can register additional sources like knowledge graphs or facility databases)
      2. Checks if the request references previous conversation (``depends_on_chat_history: False`` - it's standalone)
      3. Checks if it needs stored information (``depends_on_user_memory: False`` - no memory needed)
      4. Extracts the task: "Plot the beam current over the last 24 hours" (essentially unchanged since it's self-contained)

      **Why This Matters:**

      Task extraction serves three critical purposes:

      1. **Reference Resolution**: If you say "show me that channel again" or "what was it an hour ago?", task extraction resolves these references using chat history, creating a self-contained task like "Display the beam current channel (SR:C01-BI:G02D<IBPM1:signal1>-AM)" or "Retrieve CPU usage from approximately 13:23 (one hour before last check)". This is what allows conversational follow-ups to work correctly.

      2. **Context Reuse Optimization**: The dependency flags (``depends_on_chat_history`` and ``depends_on_user_memory``) tell the :doc:`orchestrator <../developer-guides/04_infrastructure-components/04_orchestrator-planning>` whether it should prioritize reusing context from previous executions. When ``depends_on_chat_history=True``, the orchestrator knows to check ``agent_context`` for existing channel addresses, time ranges, or other data from earlier in the conversation, avoiding redundant capability invocations.

      3. **Performance & Scalability**: Chat history grows rapidly in typical AI agent conversations. Without task extraction, you'd need to pass the entire conversation history (plus all integrated data sources) to every downstream component. This would significantly slow all language model operations. Task extraction compresses multi-turn conversations into focused task descriptions, keeping token counts manageable and response times fast.

      **Quality Control**: For task extraction to work properly with facility-specific terminology and conventions, you may need to customize the task extraction prompts to make the system aware of your facility's peculiarities. See :ref:`part4-framework-prompt-customization` in Part 4 for guidance on customizing task extraction and other framework prompts.

      **Performance Note:** The framework can operate in :ref:`bypass mode <bypass-task-extraction-section>` (controlled via ``/task:off`` slash command or config) which skips LLM-based extraction and passes formatted context directly to classification, trading some intelligence for speed on standalone queries.

      .. dropdown:: 🖥️ **View Terminal Output**

         .. code-block:: text

            👤 You: plot the beam current over the last 24
            🔄 Processing: plot the beam current over the last 24

            INFO Task_Extraction: Starting Task Extraction and Processing
            INFO Task_Extraction:  * Extracted: 'Plot the beam current over the last 24 hours'
            INFO Task_Extraction:  * Builds on previous context: False
            INFO Task_Extraction:  * Uses memory context: False
            INFO ✅ Task_Extraction: Completed Task Extraction and Processing in 1.23s

      **Further Reading:** :doc:`../developer-guides/04_infrastructure-components/02_task-extraction-system`

   .. tab-item:: Phase 2: Classification
      :name: phase2-classification

      The :doc:`classification system <../developer-guides/04_infrastructure-components/03_classification-and-routing>` determines which capabilities are needed to complete the extracted task. This uses LLM-based classification with the few-shot examples you provided in each capability's ``_create_classifier_guide()`` method.

      **What's Happening:**

      Your assistant has 6 capabilities available (see :doc:`registry system <../developer-guides/03_core-framework-systems/03_registry-and-discovery>` for component registration details):

      **Framework capabilities:**

      - ``time_range_parsing`` - Parse time expressions like "last 24 hours" into datetime objects
      - ``memory`` - Save and retrieve information from user memory files
      - ``python`` - Generate and execute Python code for calculations and plotting

      **Your application capabilities:**

      - ``channel_finding`` - Find control system channels using semantic search
      - ``channel_value_retrieval`` - Retrieve current values from control system channels
      - ``archiver_retrieval`` - Query historical time-series data from the archiver

      The framework evaluates all 6 capabilities independently using their classifier guides. For the request "Show me the beam current over the last 24 hours", it asks:

      - "Does this task require ``time_range_parsing``?" → **YES** (need to parse "last 24 hours")
      - "Does this task require ``memory``?" → **NO** (not storing/recalling information)
      - "Does this task require ``python``?" → **YES** (need to plot the data)
      - "Does this task require ``channel_finding``?" → **YES** (need to find the beam current channel)
      - "Does this task require ``channel_value_retrieval``?" → **NO** (need historical data, not current values)
      - "Does this task require ``archiver_retrieval``?" → **YES** (need to retrieve time-series data)

      The classification happens in parallel for efficiency, with each capability evaluated independently based on the examples and instructions in its ``_create_classifier_guide()`` method.

      **Why This Matters:**

      Accurate capability selection is critical because it limits the amount of context, examples, and prompts shown to the orchestrator in the next phase. By selecting only relevant capabilities (4 of 6 in this example), the orchestrator receives focused, targeted information rather than being overwhelmed with irrelevant examples. This improves both latency (fewer tokens to process) and accuracy (more relevant context for planning). When planning a plotting task, the orchestrator sees plotting and data retrieval examples, not memory storage or unrelated capability patterns, leading to cleaner execution plans.

      **Quality Control**: Your classifier examples directly determine selection accuracy. Good examples (clear positive/negative cases with reasoning) lead to accurate capability selection. Poor examples cause misclassification and failed executions. You can always refine the ``_create_classifier_guide()`` methods in your capabilities to improve accuracy.

      **Performance Note:** The framework can operate in :ref:`bypass mode <bypass-capability-selection-section>` (controlled via ``/caps:off`` slash command or config) which skips classification and activates all capabilities, useful for debugging when you're unsure which capabilities should be active.

      .. dropdown:: 🖥️ **View Terminal Output**

         .. code-block:: text

            INFO Classifier: Starting Task Classification and Capability Selection
            INFO Classifier: Classifying task: Plot the beam current measurements over the last 24 hours
            INFO Classifier: Classifying 6 capabilities with max 5 concurrent requests
            INFO Classifier:  >>> Capability 'time_range_parsing' >>> True
            INFO Classifier:  >>> Capability 'memory' >>> False
            INFO Classifier:  >>> Capability 'channel_value_retrieval' >>> False
            INFO Classifier:  >>> Capability 'channel_finding' >>> True
            INFO Classifier:  >>> Capability 'python' >>> True
            INFO Classifier:  >>> Capability 'archiver_retrieval' >>> True
            INFO Classifier: 6 capabilities required: ['respond', 'clarify', 'time_range_parsing',
                             'python', 'channel_finding', 'archiver_retrieval']
            INFO ✅ Classifier: Completed Task Classification and Capability Selection in 1.92s

      **Further Reading:** :doc:`../developer-guides/04_infrastructure-components/03_classification-and-routing`, :ref:`Classifier Guide example <hello-world-classifier-guide>`

   .. tab-item:: Phase 3: Planning (Orchestrator)

      With the active capabilities identified, the :doc:`orchestrator <../developer-guides/04_infrastructure-components/04_orchestrator-planning>` creates a complete execution plan upfront. This is the framework's :doc:`orchestrator-first philosophy <../developer-guides/01_understanding-the-framework/04_orchestrator-first-philosophy>` in action.

      **What's Happening:**

      Rather than making decisions step-by-step during execution, the orchestrator analyzes all available capabilities and creates a complete execution plan showing exactly how each capability will be used, what inputs each step requires, and how results flow between steps.

      For this query, the orchestrator creates a 5-step plan with clear dependency relationships:

      .. code-block:: text

         Step 1: channel_finding (no dependencies)
         Step 2: time_range_parsing (no dependencies)
         Step 3: archiver_retrieval (requires: Steps 1, 2)
         Step 4: python (requires: Step 3)
         Step 5: respond (requires: Step 4)

      The orchestrator uses the ``_create_orchestrator_guide()`` examples you provided in each capability to understand when and how to use each capability effectively.

      **Why This Matters:**

      1. **Transparency for High-Stakes Environments**: Complete visibility into planned operations before execution begins - which channels will be accessed, what data will be retrieved, and what operations will be performed. This is essential in scientific facilities and production environments where control system interactions require careful oversight.

      2. **Human-in-the-Loop Safety**: The explicit execution plan enables :doc:`human approval workflows <../developer-guides/05_production-systems/01_human-approval-workflows>` where operators can review and edit plans before hardware interaction. Unlike reactive approaches that only show what already happened, the orchestrator enables prevention-focused safety.

      3. **Dependency Analysis**: The orchestrator explicitly identifies which steps depend on others and which are independent. While the framework currently executes steps sequentially, the dependency structure positions Osprey for future optimizations like parallel execution of independent steps (see `GitHub issue #19 <https://github.com/als-apg/osprey/issues/19>`_ for planned improvements)

      **Quality Control**:

      .. dropdown:: 🖥️ **View Terminal Output**

         .. code-block:: text

            INFO Orchestrator: Starting Execution Planning and Orchestration
            INFO Orchestrator: Planning for task: Plot the beam current measurements over the last 24 hours
            INFO Orchestrator: Available capabilities: ['respond', 'clarify', 'time_range_parsing',
                               'python', 'channel_finding', 'archiver_retrieval']
            INFO Orchestrator: Creating execution plan with orchestrator LLM
            INFO Orchestrator: Orchestrator LLM execution time: 5.19 seconds
            INFO Orchestrator: ==================================================
            INFO Orchestrator:  << Step 1
            INFO Orchestrator:  << ├───── id: 'beam_current_channels'
            INFO Orchestrator:  << ├─── node: 'channel_finding'
            INFO Orchestrator:  << ├─── task: 'Find the channel addresses for beam current measurements in the system'
            INFO Orchestrator:  << └─ inputs: 'None'
            INFO Orchestrator:  << Step 2
            INFO Orchestrator:  << ├───── id: 'last_24h_timerange'
            INFO Orchestrator:  << ├─── node: 'time_range_parsing'
            INFO Orchestrator:  << ├─── task: 'Parse and convert the time range 'last 24 hours' to absolute
                                  datetime objects representing the start and end times'
            INFO Orchestrator:  << └─ inputs: 'None'
            INFO Orchestrator:  << Step 3
            INFO Orchestrator:  << ├───── id: 'beam_current_historical_data'
            INFO Orchestrator:  << ├─── node: 'archiver_retrieval'
            INFO Orchestrator:  << ├─── task: 'Retrieve historical beam current measurement data from the
                                  archiver for the last 24 hours'
            INFO Orchestrator:  << └─ inputs: '[{'CHANNEL_ADDRESSES': 'beam_current_channels'},
                                               {'TIME_RANGE': 'last_24h_timerange'}]'
            INFO Orchestrator:  << Step 4
            INFO Orchestrator:  << ├───── id: 'beam_current_plot'
            INFO Orchestrator:  << ├─── node: 'python'
            INFO Orchestrator:  << ├─── task: 'Create a time-series plot of beam current measurements over
                                  the last 24 hours using the retrieved archiver data'
            INFO Orchestrator:  << └─ inputs: '[{'ARCHIVER_DATA': 'beam_current_historical_data'}]'
            INFO Orchestrator:  << Step 5
            INFO Orchestrator:  << ├───── id: 'final_response'
            INFO Orchestrator:  << ├─── node: 'respond'
            INFO Orchestrator:  << ├─── task: 'Deliver the beam current measurement plot to the user with
                                  relevant context and interpretation'
            INFO Orchestrator:  << └─ inputs: '[{'PYTHON_RESULTS': 'beam_current_plot'}]'
            INFO Orchestrator: ==================================================
            INFO ✅ Orchestrator: Final execution plan ready with 5 steps

      **Further Reading:** :doc:`../developer-guides/04_infrastructure-components/04_orchestrator-planning`

.. _planning-mode-example:

.. dropdown:: 🔍 **Want to Review This Plan Before Execution? Use Planning Mode!**
   :color: primary

   Phases 1-3 always run automatically - the framework extracts the task, classifies capabilities, and creates the execution plan. But by default, **execution begins immediately** after the plan is created. What if you wanted to **review and approve** the plan before any execution starts?

   **Planning Mode: Pause Before Execution**

   Planning mode pauses after Phase 3 (plan creation) and before Phase 4 (execution), giving you full transparency into what the framework intends to do. This is especially valuable for control system operations where you want to verify the approach before any hardware interaction.

   **How to Enable Planning Mode:**

   .. code-block:: bash

      # Start the CLI chat interface
      osprey chat

      # Method 1: Use /planning slash command for a single query
      You: /planning plot the beam current over the last 24h

      # Method 2: Enable globally in config.yml (all queries require approval)
      # In config.yml:
      orchestration:
        planning_mode: true

      # Then normal queries will automatically enter planning mode:
      You: plot the beam current over the last 24h

   **What You'll See:**

   Instead of immediately executing, the framework will:

   1. **Generate the execution plan** (same 5-step plan shown above)
   2. **Save it** to ``_agent_data/execution_plans/pending_plans/pending_execution_plan.json``
   3. **Display the full plan** with all steps, dependencies, and expected outputs
   4. **Request your approval** before proceeding

   .. dropdown:: **Example: The Execution Plan JSON**
      :color: info

      Here's what the complete execution plan looks like in planning mode:

      .. literalinclude:: /_static/resources/execution_plans/accelerator_assistant_execution_plan.json
         :language: json
         :caption: Beam Current Plot - Execution Plan (JSON)
         :linenos:

   **Understanding the Plan Structure:**

   Each step in the plan includes:

   - **context_key**: Where results will be stored (e.g., "beam_current_channels")
   - **capability**: Which capability will execute (e.g., "channel_finding")
   - **task_objective**: What the step aims to accomplish
   - **expected_output**: The context type produced (e.g., "CHANNEL_ADDRESSES")
   - **inputs**: Which previous steps' outputs this step depends on (``[]`` = no dependencies)

   **What You'll See - Approval Workflow:**

   When planning mode is enabled, the framework interrupts execution and displays the plan:

   .. code-block:: text

      ⚠️ **HUMAN APPROVAL REQUIRED** ⚠️

      **Planned Steps (5 total):**
      **Step 1:** Find all channel addresses for beam current measurements
      in the system. (channel_finding)

      **Step 2:** Parse and convert 'last 24 hours' into absolute datetime
      range with start and end times. (time_range_parsing)

      **Step 3:** Retrieve historical beam current measurements from the
      archiver for the last 24 hours using identified channels.
      (archiver_retrieval)

      **Step 4:** Create a professional time-series plot of beam current
      measurements over the 24-hour period with appropriate labeling,
      legends, and formatting. (python)

      **Step 5:** Deliver the beam current plot and summary to the user.
      (respond)

      **Plan File:**
      `/path/to/_agent_data/execution_plans/pending_plans/pending_execution_plan.json`

      **To proceed, respond with:**
      - **`yes`** to approve and execute the plan
      - **`no`** to cancel this operation

      👤 You: _

   **How Approval Works:**

   The framework uses **LangGraph's interrupt system** to pause execution and wait for your response. Your response is analyzed semantically by an LLM, so you don't need to type exactly "yes" or "no" - natural language works:

   - **Approve**: "yes", "approve", "go ahead", "looks good", "execute it", etc.
   - **Reject**: "no", "cancel", "stop", "don't run this", "I don't approve", etc.

   After approval, the framework resumes from exactly where it paused and executes the plan.

   **Customizing Approval Messages (for Developers):**

   When building your own capabilities that require approval, you can customize the message shown to users. The framework uses LangGraph's interrupt system with custom interrupt data:

   .. code-block:: python

      # Example from machine_operations capability
      from osprey.approval import create_approval_type, handle_service_with_interrupts

      # In your capability's execute method:
      service_result = await handle_service_with_interrupts(
          service=python_service,
          request=execution_request,
          config=service_config,
          logger=logger,
          capability_name="YourCapability"  # Customizes the approval type
      )

   The approval system works through:

   1. **Interrupt Creation**: Service raises ``GraphInterrupt`` with custom message data
   2. **User Sees**: Formatted message explaining what will be executed
   3. **User Responds**: Natural language response (analyzed by LLM)
   4. **Resume Execution**: Service receives approval data via ``Command(resume=response)``

   You can customize:

   - The approval message format and content
   - What data is shown to the user (PV names, parameter values, etc.)
   - Warning levels for different operation types
   - Additional context or safety information

   **Now let's see what actually happened when this plan executed →**

Phase 4: Execution and Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With the execution plan complete, the framework executes each planned step in sequence, coordinated by the :doc:`router <../developer-guides/04_infrastructure-components/03_classification-and-routing>`. The router acts as the central decision authority, managing state updates, error handling, and dependency resolution automatically.

.. dropdown:: **How Execution Works**
   :color: info

   Each capability execution follows a consistent pattern managed by the ``@capability_node`` decorator:

   1. **Read from State**: Capability accesses required inputs from previous steps via :doc:`AgentState <../developer-guides/03_core-framework-systems/01_state-management-architecture>`
   2. **Execute Business Logic**: Your capability's ``execute()`` method performs its domain-specific task
   3. **Update State**: Results are stored as :doc:`context objects <../developer-guides/03_core-framework-systems/02_context-management-system>` for downstream capabilities
   4. **Return to Router**: Control returns to router to determine next step

   **Dependency Resolution:** Steps 1-2 have no dependencies and execute immediately, while Step 3 waits for both to complete. The router enforces these dependencies automatically based on the execution plan.

   **Further Reading:** :doc:`../developer-guides/03_core-framework-systems/05_message-and-execution-flow`, :doc:`../api_reference/01_core_framework/02_state_and_context`

.. tab-set::

   .. tab-item:: Step 1

      **Channel Finding**

      **📋 Execution Plan:**

      - **Task**: Find all channel addresses for beam current measurements in the system
      - **Output**: ``CHANNEL_ADDRESSES`` context (stored as "beam_current_channels")
      - **Dependencies**: None → Executes immediately
      - **Pipeline**: Hierarchical navigation through your channel database

      **✅ Execution Result:**

      The hierarchical pipeline successfully navigated: **system** (DIAG) → **family** (DCCT) → **device** (MAIN) → **field** (CURRENT) → **subfield** (RB)

      **Found 1 channel**: ``DIAG:DCCT:MAIN:CURRENT:RB``

      .. dropdown:: 🖥️  **View Terminal Output**

         .. code-block:: text

            INFO Router: Executing step 1/5 - capability: channel_finding
            INFO Channel_Finding: Channel finding query: "Find all channel addresses for beam current..."
            INFO Stage 1: Split into 1 atomic query
            INFO Stage 2: Navigating hierarchy...
            INFO   Level: system
            INFO     Available options: 4
            INFO     → Selected: ['DIAG']
            INFO     Level: family
            INFO       Available options: 4
            INFO       → Selected: ['DCCT']
            INFO       Level: device
            INFO         Available options: 1
            INFO         → Selected: ['MAIN']
            INFO         Level: field
            INFO           Available options: 2
            INFO           → Selected: ['CURRENT']
            INFO           Level: subfield
            INFO             Available options: 1
            INFO             → Selected: ['RB']
            INFO   → Found 1 channel(s)
            INFO Channel_Finding: Found 1 channel addresses

   .. tab-item:: Step 2

      **Time Range Parsing**

      **📋 Execution Plan:**

      - **Task**: Parse and convert 'last 24 hours' into absolute datetime range with start and end times
      - **Output**: ``TIME_RANGE`` context (stored as "last_24_hours_timerange")
      - **Dependencies**: None
      - **Processing**: Converts relative time expressions to absolute timestamps

      **✅ Execution Result:**

      Successfully parsed relative time "last 24 hours" into absolute datetime range:

      - **Start**: 2025-11-10 15:25:16
      - **End**: 2025-11-11 15:25:16

      .. dropdown:: 🖥️  **View Terminal Output**

         .. code-block:: text

            INFO Router: Executing step 2/5 - capability: time_range_parsing
            INFO Time_Range_Parsing: Starting time range parsing
            INFO Time_Range_Parsing: Query: "Parse and convert 'last 24 hours' into absolute datetime range..."
            INFO Time_Range_Parsing: Parsed time range:
            INFO Time_Range_Parsing:   Start: 2025-11-10 15:25:16
            INFO Time_Range_Parsing:   End:   2025-11-11 15:25:16

   .. tab-item:: Step 3

      **Archiver Data Retrieval**

      **📋 Execution Plan:**

      - **Task**: Retrieve historical beam current measurements from the archiver for the last 24 hours using identified channels
      - **Output**: ``ARCHIVER_DATA`` context (stored as "beam_current_archiver_data")
      - **Dependencies**: ⚠️  **BOTH** Step 1 (channels) **AND** Step 2 (time range) must complete first
      - **Inputs Required**:
        - ``CHANNEL_ADDRESSES`` from "beam_current_channels"
        - ``TIME_RANGE`` from "last_24_hours_timerange"

      This is a **converging dependency** - the framework waits for both parallel steps to complete before executing this step.

      **✅ Execution Result:**

      Successfully retrieved **10,000 data points** for **3 channels** covering the full 24-hour period (2025-11-10 15:25:16 to 2025-11-11 15:25:16).

      .. dropdown:: 🖥️  **View Terminal Output**

         .. code-block:: text

            INFO Router: Executing step 3/5 - capability: archiver_retrieval
            INFO Archiver_Retrieval: Starting archiver data retrieval: Retrieve historical beam current...
            INFO Archiver_Retrieval: Successfully extracted both required contexts: CHANNEL_ADDRESSES and TIME_RANGE
            INFO Archiver_Retrieval: Retrieved archiver data: 10000 points for 3 channels
                                     from 2025-11-10 15:25:16 to 2025-11-11 15:25:16

   .. tab-item:: Step 4

      **Python Visualization**

      **📋 Execution Plan:**

      - **Task**: Create a professional time-series plot of beam current measurements over the 24-hour period with appropriate labeling, legends, and formatting
      - **Output**: ``PYTHON_RESULTS`` context (stored as "beam_current_plot")
      - **Dependencies**: Step 3 (archiver data)
      - **Inputs**: ``ARCHIVER_DATA`` from "beam_current_archiver_data"
      - **Execution Mode**: Read-only (no hardware writes)
      - **Approval Required**: No (safe visualization operation)

      **✅ Execution Result:**

      - **Generated**: 3,027 characters of Python code (82 lines)
      - **Execution Time**: 2.52 seconds
      - **Mode**: Read-only (static analysis passed)
      - **Outputs**:
        - 2 figure files (14" × 7" dimensions)
        - Complete Jupyter notebook with execution code

      The Python capability automatically determined this was a safe read-only visualization task requiring no approval.

      .. dropdown:: 🖥️  **View Terminal Output**

         .. code-block:: text

            INFO Router: Executing step 4/5 - capability: python
            INFO Python: Python executor service using in-memory checkpointer
            INFO Python_Generator: Generating code with prompt length: 2668 characters
            INFO ✅ Python_Generator: Successfully generated 3027 characters of code
            INFO Python_Analyzer: Execution policy decision: ExecutionMode.READ_ONLY, approval: False
            INFO Osprey: Static analysis passed: 1 non-critical issues, execution mode: read_only
            INFO Python_Executor: LOCAL EXECUTION: Running code in read_only mode
            INFO Python_Executor: Code execution completed successfully

   .. tab-item:: Final Result

      **Complete Response to User**

      .. code-block:: text

         🤖 # Beam Current Measurements - 24 Hour Analysis

         ## Execution Summary

         ✅ **Task completed successfully** on 2025-11-11 at 07:25:24

         **Execution Details:**
         - **Status:** Success
         - **Execution Time:** 2.52 seconds
         - **Code Lines:** 82
         - **Python Environment:** `<workspace>/venv/bin/python3.11`

         ## Visualizations Generated

         **2 figures have been created and saved to the execution folder:**
         - These plots display the beam current time-series data over the 24-hour period
         - Professional styling applied with multi-channel visualization support
         - Figure size: 14" × 7" for detailed readability

         ## Data Processed

         The analysis retrieved beam current archiver data including:
         - **Multiple channels:** Available channels from the beam current archiver system
         - **Time-series measurements:** Complete 24-hour dataset with timestamps
         - **Data Source:** Archiver data accessed via context configuration

         📊 **Jupyter Notebook:**
         - A complete notebook with execution code and detailed results is available in the execution folder

         📊 Generated Figures:
         • /path/to/beam_current_timeseries.png
         • /path/to/figures/figure_01.png

      **What Just Happened:**

      The framework executed a 5-step plan that:

      1. Found 3 beam current channel addresses using hierarchical navigation
      2. Parsed "last 24 hours" into absolute timestamps (2025-11-10 15:25:16 → 2025-11-11 15:25:16)
      3. Retrieved 10,000 archiver data points across the 3 channels
      4. Generated 82 lines of Python code and created 2 visualization figures
      5. Delivered this comprehensive response with execution metadata


.. _step-6-mock-services:

Step 6: Mock Services for Development
=====================================

The Control Assistant template includes realistic mock services that let you develop and test your assistant without access to real control system hardware or archiver services.

**Framework Integration**

The mock services integrate seamlessly with the framework through the :doc:`connector abstraction layer <../developer-guides/05_production-systems/06_control-system-integration>`:

1. **Pluggable Architecture**: Capabilities use ``ConnectorFactory`` to create connectors based on your ``config.yml`` settings
2. **Zero Code Changes**: Switch from development to production by changing one config field (``type: mock`` → ``type: epics``) — see :ref:`migrate-to-production` for the complete migration guide
3. **Realistic Behavior**: Mock services simulate network latency, measurement noise, and control system patterns
4. **Universal Compatibility**: Accept any channel names—no need to predefine channel lists

The mock services are automatically used when you run the generated template. This allows you to:

- Complete the tutorial without hardware access
- Verify that the framework behavior is correct
- Understand how data flows through your capabilities
- Test your configuration before connecting to production systems



.. dropdown:: How Data is Generated
   :color: info

   .. tab-set::

      .. tab-item:: Mock Control System

         The mock control system simulates real-time channel value reads with realistic behavior.

         The mock connector attempts to generate reasonable values based on channel naming patterns:

         .. code-block:: python

            def _generate_initial_value(self, pv_name: str) -> float:
                  """Generate realistic values based on channel type."""
                  pv_lower = pv_name.lower()

                  if 'current' in pv_lower:
                     return 500.0 if 'beam' in pv_lower else 150.0
                  elif 'pressure' in pv_lower:
                     return 1e-9  # Vacuum pressure in Torr
                  elif 'voltage' in pv_lower:
                     return 5000.0
                  elif 'temp' in pv_lower:
                     return 25.0  # Temperature in °C
                  elif 'position' in pv_lower:
                     return 0.0
                  else:
                     return 100.0

         **Features:**

         - Accepts any channel name (no predefined channel list required)
         - Adds configurable measurement noise (default 1%)
         - Simulates network latency (default 10ms)
         - Maintains state between reads/writes
         - Infers units from channel names when possible

         **Configuration** (``config.yml``):

         .. code-block:: yaml

            control_system:
            type: mock  # Development mode (change to 'epics' for production)
            connector:
               timeout: 5.0
               # Mock uses sensible defaults - no additional config needed

      .. tab-item:: Mock Archiver

         The mock archiver generates synthetic historical data with realistic patterns and trends.

         The mock archiver creates time series with physics-inspired patterns:

         .. code-block:: python

            def _generate_time_series(self, pv_name: str, num_points: int):
                  """Generate synthetic time series with realistic patterns."""
                  t = np.linspace(0, 1, num_points)
                  pv_lower = pv_name.lower()

                  if ('beam' in pv_lower and 'current' in pv_lower) or 'dcct' in pv_lower:
                     # Beam current: decay with periodic refills (10 cycles, 5% loss per cycle)
                     base = 500.0
                     decay = base * (1 - 0.05 * (t % 0.10) / 0.10)
                     oscillation = 5 * np.sin(2 * np.pi * t * 5)
                     noise = np.random.normal(0, base * 0.01, num_points)
                     return decay + oscillation + noise

                  elif 'pressure' in pv_lower:
                     # Vacuum: slow drift with fast fluctuations
                     base = 1e-9
                     drift = base * (1 + 0.1 * t)
                     fluctuation = base * 0.05 * np.sin(2 * np.pi * t * 10)
                     return drift + fluctuation

         **Features:**

         - Accepts any channel names (no predefined channel list required)
         - Generates time series with trends, oscillations, and noise
         - Adjusts point density based on time range and precision
         - Returns pandas DataFrames matching production archiver format

         **Configuration** (``config.yml``):

         .. code-block:: yaml

            archiver:
            type: mock_archiver  # Development mode (change to 'epics_archiver' for production)
            # Mock uses sensible defaults - no additional config needed

.. _step-7-context-classes:

Step 7: Context Classes for Control System Data
===============================================

Context classes are the structured data containers that flow between capabilities in the Osprey framework. They serve as the "shared memory" that enables capabilities to work together—one capability produces context (e.g., channel addresses), and downstream capabilities consume it (e.g., to retrieve values from those channels).

**Why Context Classes Matter in Scientific Computing:**

In data-intensive scientific applications, context classes do more than just store data—they provide **LLM-optimized access patterns** that guide the agent in generating correct Python code. Unlike pure ReAct agents that pass tool outputs directly back to the LLM's context window (which fails immediately with archiver data containing thousands of datapoints), context classes keep large datasets in state memory while exposing only metadata and access instructions to the LLM. Scientific data often involves complex nested structures, domain-specific identifiers with special characters (e.g., ``SR:CURRENT:RB``, ``MAG:QF[QF03]:CURRENT:SP`` in control systems), and large time series datasets that require intelligent handling. Production-grade context classes explicitly describe how to access nested data structures, handle special characters, manage large datasets, and avoid common mistakes that LLMs make when generating code for scientific data access.

The control assistant template provides three production-validated context classes based on the deployed :doc:`ALS Accelerator Assistant <../example-applications/als-assistant>`. These patterns are broadly applicable to scientific computing applications beyond control systems:

- **ChannelAddressesContext**: Results from channel finding (list of found addresses)
- **ChannelValuesContext**: Live channel value reads (current measurements)
- **ArchiverDataContext**: Historical time series data with intelligent downsampling

**Demonstrating with Channel Values Context:**

Let's examine the ``ChannelValuesContext`` as a complete example of production-grade patterns. This context stores live channel value reads and demonstrates critical techniques for handling scientific data with special characters and nested structures.

.. admonition:: Requirements
   :class: note

   All context classes must inherit from ``CapabilityContext`` and implement two required methods: ``get_access_details()`` (for LLM code generation) and ``get_summary()`` (for human display).

**Class Structure:**

First, define a nested Pydantic model for individual channel values:

.. code-block:: python

   class ChannelValue(BaseModel):
       """Individual channel value data - simple nested structure for Pydantic."""
       value: str
       timestamp: datetime  # Pydantic handles datetime serialization automatically
       units: str

Then define the context class with its data fields:

.. code-block:: python

   class ChannelValuesContext(CapabilityContext):
       """
       Result from channel value retrieval operation and context for downstream capabilities.
       Based on ALS Assistant's PVValues pattern.
       """
       # Context type and category identifiers
       CONTEXT_TYPE: ClassVar[str] = "CHANNEL_VALUES"
       CONTEXT_CATEGORY: ClassVar[str] = "COMPUTATIONAL_DATA"

       # Data structure: dictionary mapping channel names to ChannelValue objects
       channel_values: Dict[str, ChannelValue]

       @property
       def channel_count(self) -> int:
           """Number of channels retrieved."""
           return len(self.channel_values)

**Required Method 1: get_access_details()**

This method provides rich, LLM-optimized documentation for code generation. Notice how it explicitly explains the bracket vs. dot notation pattern:

.. code-block:: python

       def get_access_details(self, key_name: Optional[str] = None) -> Dict[str, Any]:
           """Rich description for LLM consumption."""
           channels_preview = list(self.channel_values.keys())[:3]
           example_channel = channels_preview[0] if channels_preview else "SR:CURRENT:RB"

           # Get example value from the ChannelValue object
           try:
               example_value = self.channel_values[example_channel].value if example_channel in self.channel_values else '400.5'
           except:
               example_value = '400.5'

           key_ref = key_name if key_name else "key_name"
           return {
               "channel_count": self.channel_count,
               "channels": channels_preview,
               "data_structure": "Dict[channel_name -> ChannelValue] where ChannelValue has .value, .timestamp, .units fields - IMPORTANT: use bracket notation for channel names (due to special characters like colons), but dot notation for fields",
               "access_pattern": f"context.{self.CONTEXT_TYPE}.{key_ref}.channel_values['CHANNEL_NAME'].value (NOT ['value'])",
               "example_usage": f"context.{self.CONTEXT_TYPE}.{key_ref}.channel_values['{example_channel}'].value gives '{example_value}' (use .value not ['value'])",
               "available_fields": ["value", "timestamp", "units"],
           }

.. dropdown:: Critical Pattern: Understanding Context Object Access
   :color: warning

   The ``data_structure`` field in ``get_access_details()`` explicitly guides the LLM on the correct access patterns for your context data. This is critical because the framework uses **Pydantic models for type safety**, not plain dictionaries:

   **Context objects are Pydantic models**:
      - Access fields with **dot notation**: ``context.CHANNEL_VALUES.key_name.channel_values`` ✅
      - The ``channel_values`` field is a ``Dict[str, ChannelValue]`` (dictionary)

   **Dictionary fields use bracket notation**:
      - Access dictionary keys with **brackets**: ``channel_values['SR:CURRENT:RB']`` ✅
      - This works for any key (with or without special characters)

   **Nested Pydantic models use dot notation**:
      - ``ChannelValue`` is a Pydantic model with fields ``.value``, ``.timestamp``, ``.units``
      - Access these fields with **dot notation**: ``.value`` ✅
      - **NOT** bracket notation: ``['value']`` ❌ (this would fail)

   **Complete access pattern**:
      ``context.CHANNEL_VALUES.key_name.channel_values['SR:CURRENT:RB'].value``

      1. ``context.CHANNEL_VALUES.key_name`` → Pydantic object (dot notation)
      2. ``.channel_values`` → Dict field on Pydantic object (dot notation)
      3. ``['SR:CURRENT:RB']`` → Dictionary key access (bracket notation)
      4. ``.value`` → Field on ChannelValue Pydantic object (dot notation)

   Without this explicit guidance, LLMs frequently mix up dictionary access patterns with Pydantic field access, causing runtime errors. This pattern applies to any framework using Pydantic models for type-safe data structures.

**Required Method 2: get_summary()**

This method provides human-readable summaries for response generation, UI display, and debugging:

.. code-block:: python

       def get_summary(self, key_name: Optional[str] = None) -> Dict[str, Any]:
           """
           FOR HUMAN DISPLAY: Create readable summary for UI/debugging.
           Always customize for better user experience.
           """
           channel_data = {}
           for channel_name, channel_info in self.channel_values.items():
               channel_data[channel_name] = {
                   "value": channel_info.value,
                   "timestamp": channel_info.timestamp,
                   "units": channel_info.units
               }

           return {
               "type": "Channel Values",
               "channel_data": channel_data,
           }

.. admonition:: Key Pattern
   :class: tip

   ``get_access_details()`` provides **LLM-optimized documentation** for code generation, while ``get_summary()`` provides **human-readable** output for UIs and debugging. These serve different purposes: one teaches the LLM how to write correct code, the other presents data to users.

**Additional Production Context Classes:**

The control assistant template includes two more production-validated context classes, each demonstrating advanced patterns for specific scientific data scenarios.

.. dropdown:: Complete Channel Addresses Context Implementation

   Used for storing channel finding results. This is simpler than ChannelValuesContext but demonstrates the same core patterns.

   .. code-block:: python

      class ChannelAddressesContext(CapabilityContext):
          """
          Framework context for channel finding capability results.

          This is the rich context object used throughout the framework for channel address data.
          Based on ALS Assistant's PVAddresses pattern.
          """
          CONTEXT_TYPE: ClassVar[str] = "CHANNEL_ADDRESSES"
          CONTEXT_CATEGORY: ClassVar[str] = "METADATA"

          channels: List[str]  # List of found channel addresses
          description: str  # Description or additional information about the channels

          def get_access_details(self, key_name: Optional[str] = None) -> Dict[str, Any]:
              """Rich description for LLM consumption."""
              key_ref = key_name if key_name else "key_name"
              return {
                  "channels": self.channels,
                  "total_available": len(self.channels),
                  "comments": self.description,
                  "data_structure": "List of channel address strings",
                  "access_pattern": f"context.{self.CONTEXT_TYPE}.{key_ref}.channels",
                  "example_usage": f"context.{self.CONTEXT_TYPE}.{key_ref}.channels[0] gives '{self.channels[0] if self.channels else 'CHANNEL:NAME'}'",
              }

          def get_summary(self, key_name: Optional[str] = None) -> Dict[str, Any]:
              """
              FOR HUMAN DISPLAY: Create readable summary for UI/debugging.
              Always customize for better user experience.
              """
              return {
                  "type": "Channel Addresses",
                  "total_channels": len(self.channels),
                  "channel_list": self.channels,
                  "description": self.description,
              }

.. dropdown:: Complete Archiver Data Context Implementation - Critical Production Pattern

   The archiver data context demonstrates the **most critical production pattern**: automatic downsampling in ``get_summary()`` to prevent context window overflow while preserving full data access for analysis.

   .. code-block:: python

      class ArchiverDataContext(CapabilityContext):
          """
          Historical time series data from archiver.

          This stores archiver data with datetime objects for full datetime functionality and consistency.
          Based on ALS Assistant's ArchiverDataContext pattern with downsampling support.
          """
          CONTEXT_TYPE: ClassVar[str] = "ARCHIVER_DATA"
          CONTEXT_CATEGORY: ClassVar[str] = "COMPUTATIONAL_DATA"

          timestamps: List[datetime]  # List of datetime objects for full datetime functionality
          precision_ms: int  # Data precision in milliseconds
          time_series_data: Dict[str, List[float]]  # Channel name -> time series values (aligned with timestamps)
          available_channels: List[str]  # List of available channel names for intuitive filtering

          def get_access_details(self, key_name: Optional[str] = None) -> Dict[str, Any]:
              """Rich description of the archiver data structure."""
              total_points = len(self.timestamps)

              # Get example channel for demo purposes
              example_channel = self.available_channels[0] if self.available_channels else "SR:CURRENT:RB"
              example_value = self.time_series_data[example_channel][0] if self.available_channels and self.time_series_data.get(example_channel) else 100.5

              key_ref = key_name if key_name else "key_name"
              start_time = self.timestamps[0]
              end_time = self.timestamps[-1]
              duration = end_time - start_time

              return {
                  "total_points": total_points,
                  "precision_ms": self.precision_ms,
                  "channel_count": len(self.available_channels),
                  "available_channels": self.available_channels,
                  "time_info": f"Data spans from {start_time} to {end_time} (duration: {duration})",
                  "data_structure": "4 attributes: timestamps (list of datetime objects), precision_ms (int), time_series_data (dict of channel_name -> list of float values), available_channels (list of channel names)",
                  "CRITICAL_ACCESS_PATTERNS": {
                      "get_channel_names": f"channel_names = context.{self.CONTEXT_TYPE}.{key_ref}.available_channels",
                      "get_channel_data": f"data = context.{self.CONTEXT_TYPE}.{key_ref}.time_series_data['CHANNEL_NAME']",
                      "get_timestamps": f"timestamps = context.{self.CONTEXT_TYPE}.{key_ref}.timestamps",
                      "get_single_value": f"value = context.{self.CONTEXT_TYPE}.{key_ref}.time_series_data['CHANNEL_NAME'][index]",
                      "get_time_at_index": f"time = context.{self.CONTEXT_TYPE}.{key_ref}.timestamps[index]"
                  },
                  "example_usage": f"context.{self.CONTEXT_TYPE}.{key_ref}.time_series_data['{example_channel}'][0] gives {example_value}, context.{self.CONTEXT_TYPE}.{key_ref}.timestamps[0] gives datetime object",
                  "datetime_features": "Full datetime functionality: arithmetic, comparison, formatting with .strftime(), timezone operations"
              }

          def get_summary(self, key_name: Optional[str] = None) -> Dict[str, Any]:
              """
              FOR HUMAN DISPLAY: Format data for response generation.
              Downsamples large datasets to prevent context window overflow.

              🚨 CRITICAL PRODUCTION PATTERN 🚨

              This method demonstrates intelligent downsampling for large time series data.
              Without this, a 24-hour dataset at 1Hz (86,400 points) would consume massive
              context window space and make the agent unusable.

              The downsampling:
              - Keeps max 10 sample points (configurable)
              - Includes start, end, and evenly distributed middle points
              - Adds statistics (min, max, mean, first, last)
              - Warns LLM not to use downsampled data for final numerical answers
              - Directs LLM to use ANALYSIS_RESULTS context instead
              """
              max_samples = 10

              try:
                  total_points = len(self.timestamps)

                  # Create sample indices (start, middle, end)
                  if total_points <= max_samples:
                      sample_indices = list(range(total_points))
                  else:
                      # Include start, end, and evenly distributed middle points
                      step = max(1, total_points // (max_samples - 2))
                      sample_indices = [0] + list(range(step, total_points - 1, step))[:max_samples-2] + [total_points - 1]
                      sample_indices = sorted(list(set(sample_indices)))  # Remove duplicates and sort

                  # Sample timestamps
                  sample_timestamps = [self.timestamps[i] for i in sample_indices]

                  # Sample channel data
                  channel_summary = {}
                  for channel_name, values in self.time_series_data.items():
                      sample_values = [values[i] for i in sample_indices]

                      channel_summary[channel_name] = {
                          "sample_values": sample_values,
                          "sample_timestamps": sample_timestamps,
                          "statistics": {
                              "total_points": len(values),
                              "min_value": min(values),
                              "max_value": max(values),
                              "first_value": values[0],
                              "last_value": values[-1],
                              "mean_value": sum(values) / len(values)
                          }
                      }

                  return {
                      "WARNING": "🚨 THIS IS DOWNSAMPLED ARCHIVER DATA - DO NOT USE FOR FINAL NUMERICAL ANSWERS! 🚨",
                      "guidance": "For accurate analysis results, use ANALYSIS_RESULTS context instead of raw archiver data",
                      "data_info": {
                          "total_points": total_points,
                          "precision_ms": self.precision_ms,
                          "time_range": {
                              "start": self.timestamps[0] if self.timestamps else None,
                              "end": self.timestamps[-1] if self.timestamps else None
                          },
                          "downsampling_info": f"Showing {len(sample_indices)} sample points out of {total_points} total points"
                      },
                      "channel_data": channel_summary,
                      "IMPORTANT_NOTE": "Use this only for understanding data structure. For analysis results, request ANALYSIS_RESULTS context."
                  }

              except Exception as e:
                  import logging
                  logger = logging.getLogger(__name__)
                  logger.error(f"Error downsampling archiver data: {e}")
                  return {
                      "ERROR": f"Failed to downsample archiver data: {str(e)}",
                      "WARNING": "Could not process archiver data - use ANALYSIS_RESULTS instead"
                  }

   **Critical Production Patterns:**

   1. **Downsampling in get_summary()**: Prevents context window overflow by showing only 10 sample points + statistics instead of tens of thousands of data points
   2. **Warning Messages**: Explicitly tells LLM not to use downsampled data for final numerical answers
   3. **Statistics**: Provides min/max/mean/first/last values so LLM can understand data range without seeing all points
   4. **Error Handling**: Gracefully handles edge cases and provides fallback error messages
   5. **Full Data Access**: The LLM can still access the complete ``time_series_data`` dictionary directly when needed for analysis code generation



Step 8: Adapting for Your Facility
==================================

The control assistant template works immediately with mock services for development. This step shows how to adapt it for your actual facility by connecting to real hardware and customizing the channel database.

**What You'll Customize:**

.. grid:: 1 1 3 3

   .. grid-item-card:: 🗄️ Channel Database
      :link: #build-your-channel-database

      Your facility's channel list and naming patterns

   .. grid-item-card:: 🔌 Control System
      :link: #migrate-to-production-control-system

      Connect to EPICS, LabVIEW, Tango, or custom systems

   .. grid-item-card:: 📊 Archiver
      :link: #migrate-to-production-archiver

      Historical data from your archiver system

.. _build-your-channel-database:

8.1: Build Your Channel Database
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Building Your Production Database**

With the tools and concepts from :doc:`Part 2 <control-assistant-part2-channel-finder>` understood, you're ready to create your facility's channel database. This is often the most challenging aspect of deployment—the complexity depends not just on the number of channels, but on the consistency and organization of your control system's naming conventions and available documentation.

**Assessing Your Starting Point:**

The structure of your control system determines your approach:

- **Well-documented middle layers** (e.g., archiver appliance configurations, channel access gateway lists, IOC databases): These often contain structured metadata that can be leveraged for database creation. If your facility already has curated PV lists with groupings or hierarchies, this is your best starting point.

- **Flat PV lists without structure**: Many facilities have exported channel lists but lack semantic descriptions or hierarchical organization. This requires more manual curation but remains manageable with the provided tools.

- **Sparse or inconsistent documentation**: The most challenging scenario. Requires collaboration with domain experts to document channel purposes and relationships.

**Recommended Workflow:**

1. **Start small with in-context pipeline**: Begin with a CSV file containing your most critical channels (50-200 channels covering essential operations). This allows rapid iteration and validation before scaling up.

2. **Prioritize descriptions**: Rich, detailed descriptions are crucial for semantic matching. Invest time here—poorly described channels will produce unreliable query results regardless of pipeline choice.

3. **Validate early with benchmarks**: Gather test queries from operators and physicists immediately (see :ref:`Part 2 benchmarking tools <channel-finder-benchmarking>`). Run benchmarks frequently to catch description issues before they compound.

4. **Scale based on results**: Once your initial database performs well (>90% F1 score), decide whether to expand the in-context database or transition to hierarchical structure for larger systems.

5. **For large facilities**: If you're planning facility-wide deployment with thousands of channels, invest time upfront designing a hierarchical database that mirrors your control system's architecture. Leverage existing middle-layer solutions where available.

.. admonition:: Collaboration Welcome
   :class: outreach

   We are actively developing automated database builders for common middle-layer solutions (like Matlab Middle Layer). These will streamline the transition from existing infrastructure to semantic channel databases. We welcome collaborations on this effort! Please open an issue on GitHub if you're interested in contributing.

**Quick Reference: Essential Tools from Part 2**

.. tab-set::

   .. tab-item:: In-Context Pipeline

      **Best for:** few hundred channels, flat naming structures

      **Step 1:** Create CSV with your channels:

      .. code-block:: text

         address,description,family_name,instances,sub_channel
         BEAM_CURRENT_RB,Main beam current readback in mA,,,
         MAG:DIPOLE{instance:02d}:CURRENT:SP,Dipole magnet current setpoint,MAG:DIPOLE,16,CURRENT:SP
         MAG:DIPOLE{instance:02d}:CURRENT:RB,Dipole magnet current readback,MAG:DIPOLE,16,CURRENT:RB

      **Step 2:** Generate database with LLM naming:

      .. code-block:: bash

         cd my-control-assistant
         python src/my_control_assistant/data/tools/build_channel_database.py \
           --use-llm --config config.yml

      **Step 3:** Validate and preview:

      .. code-block:: bash

         python src/my_control_assistant/data/tools/validate_database.py
         python src/my_control_assistant/data/tools/preview_database.py

      **Complete Tutorial:** :doc:`control-assistant-part2-channel-finder` → In-Context Pipeline tab

   .. tab-item:: Hierarchical Pipeline

      **Best for:** > 1,000 channels, structured system hierarchy

      **Step 1:** Understand your hierarchy (e.g., System → Family → Device → Field → Subfield)

      **Step 2:** Create JSON with your structure:

      .. code-block:: json

         {
           "hierarchy_definition": ["system", "family", "device", "field", "subfield"],
           "naming_pattern": "{system}:{family}[{device}]:{field}:{subfield}",
           "tree": {
             "MAGNETS": {
               "_description": "Magnet System: Your facility description...",
               "DIPOLE": {
                 "_description": "Dipole Magnets: Bend beam trajectory...",
                 "devices": {"_type": "range", "_pattern": "D{:02d}", "_range": [1, 16]},
                 "fields": {
                   "CURRENT": {
                     "_description": "Magnet current in Amperes...",
                     "subfields": {
                       "SP": {"_description": "Setpoint (read-write)"},
                       "RB": {"_description": "Readback (read-only)"}
                     }
                   }
                 }
               }
             }
           }
         }

      **Step 3:** Validate and preview:

      .. code-block:: bash

         python src/my_control_assistant/data/tools/validate_database.py
         python src/my_control_assistant/data/tools/preview_database.py

      **Complete Tutorial:** :doc:`control-assistant-part2-channel-finder` → Hierarchical Pipeline tab


.. _migrate-to-production:

8.2: Migrate to Production Control System
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**The Power of Connector Abstraction:** Your capabilities already use ``ConnectorFactory`` to access control systems - no code changes needed! Simply update the configuration to switch from mock to production.

**For EPICS Control Systems (Native in Osprey):**

.. tab-set::

   .. tab-item:: Quick Start

      **Step 1:** Install dependencies:

      .. code-block:: bash

         pip install pyepics

      **Step 2:** Edit ``config.yml``:

      .. code-block:: yaml

         control_system:
           type: epics          # ← Changed from 'mock'
           connector:
             epics:
               gateways:
                 read_only:
                   address: cagw.your-facility.edu
                   port: 5064
                   use_name_server: false  # false = EPICS_CA_ADDR_LIST (direct gateway)
                                           # true = EPICS_CA_NAME_SERVERS (SSH tunnels, some setups)
               timeout: 5.0

      **Step 3:** Test connection:

      .. code-block:: bash

         osprey chat
         # Try: "What is the beam current?"

      **That's it!** Your capabilities work unchanged - ``ConnectorFactory`` automatically uses the EPICS connector based on configuration.

   .. tab-item:: Advanced Configuration

      **Gateway Configuration (Read-Only + Read-Write)**

      Configure separate gateways for monitoring and control:

      .. code-block:: yaml

         control_system:
           type: epics
           connector:
             epics:
               gateways:
                 read_only:
                   address: cagw.facility.edu
                   port: 5064
                   use_name_server: false  # false = EPICS_CA_ADDR_LIST (direct gateway)
                                           # true = EPICS_CA_NAME_SERVERS (SSH tunnels, some setups)
                 read_write:          # For control operations
                   address: cagw-rw.facility.edu
                   port: 5065
               timeout: 5.0
               retry_count: 3

      **Master Safety Switch** prevents writes even with configured gateway:

      .. code-block:: yaml

         execution_control:
           epics:
             writes_enabled: false  # Must be true to allow writes

      **Pattern Detection for Approval Workflows**

      Configure regex patterns for identifying control system operations in generated code:

      .. code-block:: yaml

         control_system:
           type: epics
           patterns:
             epics:
               write:
                 - 'epics\.caput\('
                 - '\.put\('
               read:
                 - 'epics\.caget\('
                 - '\.get\('

      This enables the approval system to require human review for write operations.

      **Development + Production Config Pattern**

      Use separate configs for development and production:

      .. code-block:: bash

         my-control-assistant/
         ├── config.yml              # Mock (development)
         └── config.production.yml   # EPICS (production)

      Switch at runtime:

      .. code-block:: bash

         # Development
         osprey chat

         # Production
         OSPREY_CONFIG=config.production.yml osprey chat

**For Other Control Systems (LabVIEW, Tango, Custom):**

Your facility uses something other than EPICS? The framework is designed to support custom connectors for any control system through a well-defined interface.

**How it Works:**

1. **Implement** the ``ControlSystemConnector`` interface for your system
2. **Register** your connector in ``registry.py``
3. **Configure** in ``config.yml``

Your capabilities work unchanged - ``ConnectorFactory`` automatically uses your custom connector. See :doc:`../developer-guides/05_production-systems/06_control-system-integration` for example implementations demonstrating the pattern for LabVIEW, Tango, and other custom systems.

.. admonition:: Collaboration Welcome
   :class: outreach

   We welcome collaboration in implementing and testing control system connectors for other platforms (LabVIEW, Tango, etc.). While the framework provides the architecture and example patterns, community contributions help validate these implementations in real production environments. If you're interested in adapting Osprey for your control system, please open an issue on GitHub - we're happy to support the development effort!


8.3: Migrate to Production Archiver
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Same Pattern, Different Service:** Archiver migration follows the same connector abstraction pattern - just update configuration.

**For EPICS Archiver Appliance:**

Osprey uses the `ALS Archiver Client <https://pypi.org/project/als-archiver-client/>`_ to interface with EPICS Archiver Appliance installations. This should work with any standard EPICS Archiver Appliance deployment.

**Step 1:** Install dependencies:

.. code-block:: bash

   pip install als-archiver-client

**Step 2:** Edit ``config.yml``:

.. code-block:: yaml

   archiver:
     type: epics_archiver     # ← Changed from 'mock_archiver'
     epics_archiver:
       url: https://archiver.your-facility.edu:8443
       timeout: 60

**Step 3:** Test archiver queries:

.. code-block:: bash

   osprey chat
   # Try: "Plot beam current over the last hour"

**That's it!** Your archiver capability works unchanged.

**For Custom Archiver Solutions:**

Need to integrate with a custom archiver setup or different archiver implementation? The framework supports custom archiver connectors following the same pattern as control system connectors. See :doc:`../developer-guides/05_production-systems/06_control-system-integration` for example archiver connector implementations.

.. admonition:: Collaboration Welcome
   :class: outreach

   We welcome collaboration in implementing and testing archiver connectors for different systems and setups. While the framework provides the architecture and example patterns, community contributions help validate these implementations across diverse archiver environments. If you're interested in adapting Osprey for your archiver system, please open an issue on GitHub - we're happy to support the development effort!

.. _deploy-containerized-services:

Step 9: Deploy Containerized Services
=====================================

At this stage of the tutorial, your control assistant is ready to run in a containerized environment. The default deployment includes three key services:

- **Pipelines** - The core agent runtime that executes your capabilities
- **Jupyter** - Python execution environment for running generated code and notebooks
- **OpenWebUI** - Web-based chat interface for interacting with your agent

Starting the Services
^^^^^^^^^^^^^^^^^^^^^

Use the framework CLI to start all services:

.. code-block:: bash

   # Start services in detached mode (background)
   osprey deploy up --detached

   # Check service status
   osprey deploy status

For detailed deployment options, troubleshooting, and production deployment strategies, see :doc:`../developer-guides/05_production-systems/05_container-and-deployment` and the :doc:`../developer-guides/02_quick-start-patterns/00_cli-reference`.

We recommend using **Docker Desktop** or **Podman Desktop** to inspect container logs for troubleshooting and monitoring:

- Docker Desktop: Open the GUI and navigate to the **Containers** tab to view live logs
- Podman Desktop: Similar interface with container management and log viewing
- Command line alternative: ``docker logs <container-name>`` or ``podman logs <container-name>``

After starting the services, configure OpenWebUI to connect to your agent:

1. **Access the Interface**: Navigate to http://localhost:8080
2. **Create Admin Account**: The first user becomes the admin
3. **Add External Model**:

   - Go to **Settings** → **Connections**
   - Add your agent's pipeline endpoint: ``http://pipelines:9099`` and add the API key from ``services/pipelines/docker-compose.yml.j2`` under ``PIPELINES_API_KEY`` (default ``0p3n-w3bu!``)
   - The agent will appear in the model dropdown

4. **Start Chatting**: Select your agent model and begin interacting

For detailed configuration options and troubleshooting, see the :ref:`OpenWebUI Configuration <openwebui-configuration>` section in the Installation Guide.


.. admonition:: Future: Automated OpenWebUI Setup
   :class: note

   We're working on automating the OpenWebUI configuration process to eliminate manual setup steps. Track progress and see our implementation plans in `GitHub Issue #17: Automate OpenWebUI Configuration <https://github.com/als-apg/osprey/issues/17>`_.

   Once implemented, most of these manual configuration steps will be automated through environment variables and initialization scripts.

Running Your Agent
^^^^^^^^^^^^^^^^^^

Once configured, your control assistant will appear as an "External" model in OpenWebUI's model selector (typically named after your project). Select it to start a conversation with your agent.

**How It Works:**

When you send a message, it flows through the backend services:

1. **Pipelines** container receives your request
2. Agent orchestrator analyzes it and creates an execution plan
3. Capabilities execute (with Python code running in the **Jupyter** container when using ``execution_method: container``)
4. Results are formatted by the pipeline script and sent back to **OpenWebUI**

The pipeline integration enables rich result presentation:

- Markdown responses are rendered as formatted text, tables, and code blocks
- Figures and plots are embedded directly in the conversation (via agent state registration)
- Generated code is packaged as Jupyter notebooks with clickable access links (see `Python Execution: Jupyter Container vs Local`_ below)
- Conversation context is preserved across messages, with past sessions organized in the sidebar

.. figure:: /_static/screenshots/openwebui_demo.png
   :alt: OpenWebUI interface showing a conversation with the control assistant
   :align: center
   :width: 90%

   Example conversation demonstrating the query "plot the beam current over the last 24 hours". The assistant responds with a comprehensive analysis including formatted Markdown tables, embedded visualizations, and links to the generated Jupyter notebook. Note the chat sidebar for session organization and the rich formatting of the response.


Python Execution: Jupyter Container vs Local
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When your assistant generates Python code for data analysis and visualization, the ``execution_method`` setting in ``config.yml`` controls where that code runs.

**Container Execution (Recommended for Production)**

With ``execution_method: "container"``, Python code executes in the dedicated Jupyter container. This provides:

- **Isolated execution environment** - Safer execution separate from the agent runtime
- **Persistent notebooks** - Generated notebooks remain accessible in Jupyter for review, rerun, or sharing
- **Embedded results in OpenWebUI** - Figures appear inline in chat responses with clickable links to notebooks

The template includes two Jupyter containers configured with different execution modes:

.. code-block:: yaml

   services:
     jupyter:
       containers:
         read:
           name: jupyter-read
           port_host: 8088
           execution_modes: ["read_only"]
         write:
           name: jupyter-write
           port_host: 8089
           execution_modes: ["write_access"]

   execution:
     execution_method: "container"
     modes:
       read_only:
         kernel_name: "python3-epics-readonly"
         allows_writes: false
       write_access:
         kernel_name: "python3-epics-write"
         allows_writes: true
         requires_approval: true

The framework automatically selects between containers using :doc:`pattern detection <../developer-guides/05_production-systems/03_python-execution-service>` - it analyzes generated code for control system write operations (like ``epics.caput()``) and routes to the appropriate container:

- **Read-only** container (port 8088) is used when no write patterns are detected
- **Write access** container (port 8089) is used when write patterns are detected (potentially triggering approval workflows)

Generated notebooks include clickable links like ``http://localhost:8088/notebooks/my_control_assistant/beam_current_analysis.ipynb``.

**Local Execution (Development)**

With ``execution_method: "local"``, Python code executes directly in your local environment using the Python interpreter specified in ``python_env_path``:

.. code-block:: yaml

   execution:
     execution_method: "local"
     python_env_path: /path/to/venv/bin/python

This is simpler for CLI-based development (``osprey chat``) but doesn't provide persistent notebooks or the safety benefits of containerized execution.

The framework automatically switches to container execution when you deploy with ``osprey deploy up``.


.. seealso::

   :doc:`installation`
      Complete installation and configuration reference

   :doc:`../developer-guides/05_production-systems/05_container-and-deployment`
      Production deployment strategies and best practices

   :doc:`../developer-guides/02_quick-start-patterns/00_cli-reference`
      Complete CLI command reference for deployment management

   `GitHub Issue #17 <https://github.com/als-apg/osprey/issues/17>`_
      Automated OpenWebUI configuration implementation plans

Troubleshooting
===============

.. dropdown:: Channel Finder Returns No Results
   :color: warning

   **Symptoms:** Queries return empty channel lists.

   **Solutions:**

   1. Check database path in ``config.yml``
   2. Validate database: ``python -m my_control_assistant.data.tools.validate_database``
   3. Preview database presentation: ``python -m my_control_assistant.data.tools.preview_database``
   4. Test with CLI: ``python -m my_control_assistant.services.channel_finder.cli``
   5. Enable debug mode: ``config.yml`` → ``development.prompts.print_all: true``
   6. Review saved prompts in ``_agent_data/prompts/``

.. dropdown:: EPICS Connection Issues (use_name_server Configuration)
   :color: warning

   **Symptoms:** Connection timeout or "Failed to connect to PV" errors, especially with localhost addresses, SSH tunnels, or certain gateway configurations.

   **What is use_name_server?**

   This parameter controls which EPICS Channel Access environment variable configuration method is used:

   - ``use_name_server: false`` (default): Uses ``EPICS_CA_ADDR_LIST`` - standard direct gateway access
   - ``use_name_server: true``: Uses ``EPICS_CA_NAME_SERVERS`` - alternative configuration method

   Neither is universally "better" - it depends on your specific EPICS gateway/network setup.

   **When you might need use_name_server: true:**

   - SSH tunnels to EPICS gateways (common case)
   - Some gateway configurations that require name server resolution
   - localhost/127.0.0.1 gateway addresses (often, but not always)
   - Your facility's EPICS setup specifically requires ``CA_NAME_SERVERS``

   **When to use the default (false):**

   - Standard direct gateway access (most common)
   - Your facility documentation specifies ``CA_ADDR_LIST``
   - Default works - don't change it!

   **How to configure:**

   .. code-block:: yaml

      control_system:
        type: epics
        connector:
          epics:
            gateways:
              read_only:
                address: localhost      # or your gateway address
                port: 5074              # your port
                use_name_server: true   # Try this if default fails

   **Example: SSH Tunnel Setup**

   .. code-block:: bash

      # Create SSH tunnel to EPICS gateway
      ssh -L 5074:cagw.facility.edu:5064 user@gateway.facility.edu

   Then try with ``use_name_server: true`` (but ``false`` might also work - depends on setup).

   **Troubleshooting approach:**

   1. Start with default (``use_name_server: false`` or omit the parameter)
   2. If connection fails, try ``use_name_server: true``
   3. Check with your facility's EPICS admin if both fail
   4. Verify gateway address/port are correct with ``caget YOUR:PV:NAME``

.. dropdown:: Debugging Agent Behavior with Development Features
   :color: info

   **Prompt Debugging**

   Enable detailed prompt logging in ``config.yml`` to see exactly what the LLM receives:

   .. code-block:: yaml

      development:
        prompts:
          print_all: true      # Save prompts to _agent_data/prompts/
          latest_only: true    # Overwrite vs. timestamped versions
          show_all: false      # Print to console (very verbose)

   Saved prompts include: ``orchestrator_latest.md`` (main prompt with conversation history),
   ``classification_latest.md``, ``error_analysis_latest.md``, and others.

   **Use this to:**

   - Understand why the agent made certain decisions
   - Debug capability selection or parameter extraction issues
   - Verify your database/context is properly included
   - Troubleshoot conversation history problems

   **Raw Error Raising**

   Get full stack traces instead of agent error handling:

   .. code-block:: yaml

      development:
        raise_raw_errors: true   # false (default) for production

   - ``true`` - Immediate exceptions with full traces (for developing capabilities or debugging framework issues)
   - ``false`` - Graceful error recovery by the agent (for production deployment)

   **Quick debugging workflow:**

   1. Set ``development.prompts.print_all: true`` in config
   2. Reproduce the issue
   3. Check ``_agent_data/prompts/orchestrator_latest.md``
   4. If needed, enable ``raise_raw_errors: true`` for detailed traces


Next Steps
==========

Your assistant is now running in production! Continue to Part 4 to customize and extend it for your facility.

Navigation
==========

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: ← Part 2: Channel Finder
      :link: control-assistant-part2-channel-finder
      :link-type: doc

      Return to channel finder guide

   .. grid-item-card:: Part 4: Customization →
      :link: control-assistant-part4-customization
      :link-type: doc

      Customize and extend your assistant
