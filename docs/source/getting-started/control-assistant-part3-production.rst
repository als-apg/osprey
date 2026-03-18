=============================================
Part 3: Integration & Deployment
=============================================

This part covers deploying your control assistant with production control systems,
container services, and safety patterns.

.. _step-5-observe-framework:

Step 5: Using the Chat Interface
================================

Launch the interactive chat interface to interact with your assistant:

.. code-block:: bash

   # Start Claude Code chat session
   osprey claude chat

   # Or use the interactive menu
   osprey
   # Then select: claude - Claude Code commands

Claude Code orchestrates multi-step queries by selecting and invoking MCP tools
(channel finding, archiver retrieval, Python execution, etc.) based on the user's
request. Tool selection, planning, and execution are handled natively by
Claude Code -- there is no separate classification or orchestration layer.

.. admonition:: PLACEHOLDER: CONCEPTUAL-MAPPING
   :class: warning

   **Old content (line 29):** "Phase 0: Initialization and Component Loading"
   **New equivalent:** Needs human judgment
   **Why this is fuzzy:** The old framework had explicit registry initialization with capability/node/context counts. Claude Code startup is different -- it loads MCP server configs and CLAUDE.md rules.
   **Action needed:** Document what actually happens during ``osprey claude chat`` startup (MCP server discovery, settings.json loading, hook registration).

.. admonition:: PLACEHOLDER: CONCEPTUAL-MAPPING
   :class: warning

   **Old content (line 79):** "Phases 1-3: Task Analysis and Planning" (task extraction, classification, orchestration)
   **New equivalent:** Needs human judgment
   **Why this is fuzzy:** Claude Code handles task understanding, tool selection, and multi-step planning internally. There are no discrete "phases" to document -- it is a single agentic loop.
   **Action needed:** Describe how Claude Code decides which MCP tools to call and in what order. Reference the system prompt / CLAUDE.md rules that guide tool selection.

.. admonition:: PLACEHOLDER: APPROVAL-WORKFLOW
   :class: warning

   **Old content (line 264):** "Planning Mode: Pause Before Execution" with LangGraph interrupt-based approval
   **New equivalent:** Claude Code's native permission prompt for tool calls
   **Why this is fuzzy:** The old system had plan-level approval (approve entire execution plan) and per-capability approval. Claude Code prompts per-tool-call. The safety granularity is different.
   **Action needed:** Document Claude Code's approval prompt behavior, how ``settings.json`` controls which tools require approval, and how per-tool approval policies work.


.. _step-6-mock-services:

Step 6: Mock Services for Development
======================================

The control assistant template includes mock services for development without
real hardware.

**Key Features:**

1. **Pluggable Architecture**: MCP tools use ``ConnectorFactory`` based on ``config.yml`` settings
2. **Zero Code Changes**: Switch ``type: mock`` to ``type: epics`` -- see :ref:`migrate-to-production`
3. **Realistic Behavior**: Simulates latency, measurement noise, and control system patterns
4. **Universal Compatibility**: Accepts any channel names

.. code-block:: yaml

   # config.yml -- development mode
   control_system:
     type: mock

   archiver:
     type: mock_archiver


Step 7: Adapting for Your Facility
==================================

.. _build-your-channel-database:

7.1: Build Your Channel Database
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With the tools from :doc:`Part 2 <control-assistant-part2-channel-finder>`, create
your facility's channel database.

**Recommended Workflow:**

1. **Start small**: 50--200 critical channels with the in-context pipeline
2. **Prioritize descriptions**: Rich descriptions are crucial for semantic matching
3. **Validate early**: Gather operator queries and run benchmarks (>90% F1 target)
4. **Scale based on results**: Expand in-context or transition to hierarchical pipeline

.. tab-set::

   .. tab-item:: In-Context Pipeline

      Best for few hundred channels with flat naming. Create a CSV, then:

      .. code-block:: bash

         osprey channel-finder build-database --use-llm
         osprey channel-finder validate
         osprey channel-finder preview

   .. tab-item:: Hierarchical Pipeline

      Best for 1,000+ channels with structured hierarchy. Create a JSON tree, then:

      .. code-block:: bash

         osprey channel-finder validate
         osprey channel-finder preview

See :doc:`Part 2 <control-assistant-part2-channel-finder>` for complete pipeline tutorials.

.. _migrate-to-production:

7.2: Migrate to Production Control System
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Your MCP tools already use ``ConnectorFactory`` -- no code changes needed.

**For EPICS Control Systems:**

.. tab-set::

   .. tab-item:: Interactive Configuration

      .. code-block:: bash

         osprey  # Launch interactive menu
         # Navigate: config -> set-control-system -> EPICS
         # Then: config -> set-epics-gateway -> select facility

      Supported facilities: **APS** (``pvgatemain1.aps4.anl.gov:5064``),
      **ALS** (``cagw-alsdmz.als.lbl.gov:5064``), or **Custom**.

   .. tab-item:: Manual Configuration

      .. code-block:: yaml

         control_system:
           type: epics
           writes_enabled: false  # Master safety switch
           connector:
             epics:
               gateways:
                 read_only:
                   address: cagw.facility.edu
                   port: 5064
                   use_name_server: false
                 read_write:
                   address: cagw-rw.facility.edu
                   port: 5065
               timeout: 5.0
               retry_count: 3

      Install EPICS dependencies:

      .. code-block:: bash

         uv sync  # includes pyepics

**Multi-config pattern:** Use ``config.yml`` for development (mock) and
``config.production.yml`` for EPICS. Switch at runtime with
``OSPREY_CONFIG=config.production.yml osprey claude chat``.

**Other control systems** (LabVIEW, Tango): implement ``ControlSystemConnector``
and configure in ``config.yml``.

7.3: Migrate to Production Archiver
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   archiver:
     type: epics_archiver
     epics_archiver:
       url: https://archiver.your-facility.edu:8443
       timeout: 60

Install: ``uv add als-archiver-client``

7.4: Connect Your Facility Logbook
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Configure ARIEL for production logbook search by setting ``ariel.database_uri``
and ``ariel.adapter`` in ``config.yml``, then run ``osprey ariel migrate`` and
``osprey ariel ingest``.


.. _deploy-containerized-services:

Step 8: Deploy Containerized Services
======================================

Default deployment includes:

- **Claude Code** -- The agentic runtime with MCP tools
- **Jupyter** -- Python execution for generated code and notebooks
- **Web Terminal** -- Browser-based terminal interface

Starting the Services
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   osprey deploy up --detached
   osprey deploy status

Python Execution: Jupyter Container vs Local
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``execution_method: "container"`` (production) to run generated code in
isolated Jupyter containers with separate read-only (port 8088) and write-access
(port 8089) environments. Use ``execution_method: "local"`` for CLI development.
The framework routes code to the appropriate container based on pattern detection
of control system write operations.

.. admonition:: PLACEHOLDER: CONFIG-ARCHITECTURE
   :class: warning

   **Old content (line 1615):** "Pipelines - The core agent runtime that executes your capabilities"
   **New equivalent:** Needs human judgment
   **Why this is fuzzy:** The old deployment used OpenWebUI + Pipelines as the web chat frontend. The new architecture uses Web Terminal (FastAPI + PTY) as the primary interface. The container topology has changed.
   **Action needed:** Document the current container topology: which services are deployed, how they communicate, and how the Web Terminal replaces OpenWebUI/Pipelines.


Troubleshooting
================

.. dropdown:: Channel Finder Returns No Results
   :color: warning

   1. Check database path in ``config.yml``
   2. Validate: ``osprey channel-finder validate``
   3. Preview: ``osprey channel-finder preview``
   4. Test with CLI: ``osprey channel-finder``

.. dropdown:: EPICS Connection Issues
   :color: warning

   **use_name_server configuration:**

   - ``false`` (default): Uses ``EPICS_CA_ADDR_LIST`` -- standard direct gateway
   - ``true``: Uses ``EPICS_CA_NAME_SERVERS`` -- for SSH tunnels, some gateways

   Troubleshooting: start with default, try ``true`` if connection fails,
   verify with ``caget YOUR:PV:NAME``.

.. dropdown:: Debugging Agent Behavior
   :color: info

   Enable prompt logging:

   .. code-block:: yaml

      development:
        prompts:
          print_all: true
          latest_only: true

   Check saved prompts in ``_agent_data/prompts/``.
   Enable ``development.raise_raw_errors: true`` for full stack traces.


Next Steps
==========

Continue to Part 4 to customize and extend your assistant.

Navigation
==========

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Part 2: Channel Finder
      :link: control-assistant-part2-channel-finder
      :link-type: doc

      Return to channel finder guide

   .. grid-item-card:: Part 4: Customization
      :link: control-assistant-part4-customization
      :link-type: doc

      Customize and extend your assistant
