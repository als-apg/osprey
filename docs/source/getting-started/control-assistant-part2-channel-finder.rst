:orphan:

=========================================
Part 2: Building Your Channel Finder
=========================================


Step 3: Semantic Channel Finding Pipelines
==========================================

.. admonition:: Academic Reference
   :class: seealso

   For a comprehensive theoretical framework and analysis of semantic channel finding in complex experimental infrastructure, see Hellert et al. (2025), "From Natural Language to Control Signals: A Conceptual Framework for Semantic Channel Finding in Complex Experimental Infrastructure," available on `arXiv:2512.18779 <https://arxiv.org/abs/2512.18779>`_.

**The Core Challenge: Bridging Human Language and Control System Addresses**

Control systems at scientific facilities present a fundamental communication gap: operators think in physical concepts ("beam current," "terminal voltage"), while control systems use technical addresses (``SR01C___DCCT1_AM00``, ``TMVST``). **Semantic channel finding** uses LLM-powered pipelines to translate natural language queries into specific control system addresses.

**Three Pipeline Implementations**

- **In-Context Search**: Direct semantic matching -- best for small to medium systems (few hundred channels)
- **Hierarchical Navigation**: Structured navigation through system hierarchy -- scales to large systems with strict naming patterns (thousands+ channels)
- **Middle Layer Exploration**: React agent with database query tools -- scales to large systems organized by function (thousands+ channels)

All three share the same interface, differing only in matching strategy. To switch pipelines:

.. code-block:: yaml

   channel_finder:
     pipeline_mode: in_context  # or "hierarchical" or "middle_layer"

.. admonition:: Customize prompts for your facility
   :class: tip

   After building your database, edit ``facility_description.py`` with your facility's systems and terminology to dramatically improve matching accuracy. See :ref:`part4-channel-finder-prompts` in Part 4.

.. _channel-finder-benchmarking:

.. tab-set::

   .. tab-item:: In-Context Pipeline

      **Concept:** Put the entire channel database in the LLM context and ask it to find matches.

      **Best for:** Small to medium systems (few hundred channels), rapid prototyping.

      **Pipeline Flow**: Query splitting -> Semantic matching against full database -> Validation/correction (iterative)

      **Try It Now:**

      .. code-block:: bash

         cd my-control-assistant
         osprey channel-finder

      **Database Format** -- flat list with optional template support. Each entry has ``channel`` (searchable name), ``address`` (PV address, hidden from LLM), and ``description``. Template entries define device families with instance ranges. The LLM sees only ``channel`` + ``description`` during matching.

      **Building Your Database:**

      .. code-block:: bash

         # Basic build (uses addresses as channel names)
         osprey channel-finder build-database

         # With LLM-generated descriptive names (recommended)
         osprey channel-finder build-database --use-llm

         # Validate and preview
         osprey channel-finder validate
         osprey channel-finder preview

      The ``--use-llm`` flag generates descriptive PascalCase names (e.g., ``Acc_I`` becomes ``AcceleratingTubeGradingResistorDiagnosticCurrent``) that reinforce the semantic signal from descriptions.

      **Configuration:**

      .. code-block:: yaml

         channel_finder:
           pipeline_mode: in_context
           pipelines:
             in_context:
               database:
                 type: template
                 path: src/my_control_assistant/data/channel_databases/in_context.json
                 presentation_mode: template
               processing:
                 chunk_dictionary: false
                 chunk_size: 50
                 max_correction_iterations: 2

   .. tab-item:: Hierarchical Pipeline

      **Concept:** Navigate through a structured hierarchy of systems -> families -> devices -> fields.

      **Best for:** Large systems (thousands+ channels) with well-defined hierarchical organization.

      **Pipeline Flow**: Query splitting -> Recursive hierarchical navigation (system -> family -> device -> field -> subfield) -> Channel assembly and validation

      **Example navigation** for "What's the beam current?":

      1. **system** -> [MAGNETS, VACUUM, RF, DIAGNOSTICS] -> Selected: DIAG
      2. **family** -> [DCCT, BPM, EMIT, TUNE] -> Selected: DCCT
      3. **device** -> [MAIN] -> Selected: MAIN
      4. **field** -> [CURRENT, STATUS] -> Selected: CURRENT
      5. **subfield** -> [SP, RB] -> Selected: RB

      Result: ``DIAG:DCCT[MAIN]:CURRENT:RB``

      **Try It Now:**

      .. code-block:: bash

         cd my-control-assistant
         # Edit config.yml: set pipeline_mode to "hierarchical"
         osprey channel-finder

      **Database Format** -- nested hierarchy with descriptions:

      .. code-block:: json

         {
           "hierarchy": {
             "levels": [
               {"name": "system", "type": "tree"},
               {"name": "family", "type": "tree"},
               {"name": "device", "type": "instances"},
               {"name": "field", "type": "tree"},
               {"name": "subfield", "type": "tree"}
             ],
             "naming_pattern": "{system}:{family}[{device}]:{field}:{subfield}"
           },
           "tree": {
             "MAG": {
               "_description": "Magnet System: Controls beam trajectory and focusing.",
               "QF": {
                 "_description": "Focusing Quadrupoles: Positive gradient magnets.",
                 "DEVICE": {
                   "_expansion": {"_type": "range", "_pattern": "QF{:02d}", "_range": [1, 16]},
                   "CURRENT": {
                     "_description": "Excitation Current (Amperes)",
                     "SP": {"_description": "Setpoint (read-write)"},
                     "RB": {"_description": "Readback (read-only)"}
                   }
                 }
               }
             }
           }
         }

      Level types: ``tree`` (navigate named categories) vs ``instances`` (expand numbered devices). Advanced features include navigation-only levels, friendly names (``_channel_part``), optional levels, and custom separators (``_separator``).

      **Validate and preview:**

      .. code-block:: bash

         osprey channel-finder validate
         osprey channel-finder preview --depth 4 --sections tree,stats

   .. tab-item:: Middle Layer Pipeline

      **Concept:** Agent explores database using query tools to find channels by function.

      **Best for:** Large systems organized by function (Monitor, Setpoint), facilities using MATLAB Middle Layer (MML) style organization.

      The agent uses five tools: ``list_systems()``, ``list_families(system)``, ``inspect_fields(system, family)``, ``list_channel_names(...)``, ``get_common_names(system, family)``.

      **Try It Now:**

      .. code-block:: bash

         cd my-control-assistant
         # Edit config.yml: set pipeline_mode to "middle_layer"
         osprey channel-finder

      **Database Format** -- functional hierarchy:

      .. code-block:: json

         {
           "SR": {
             "_description": "Storage Ring: Main synchrotron light source",
             "DCCT": {
               "_description": "DC Current Transformer: Measures beam current",
               "Monitor": {
                 "_description": "Beam current in milliamperes.",
                 "ChannelNames": ["SR:DCCT:Current"]
               }
             }
           }
         }

      **From MATLAB Middle Layer exports:**

      .. code-block:: bash

         python -m osprey.services.channel_finder.utils.mml_converter \
            --input path/to/mml_exports.py \
            --output src/my_control_assistant/data/channel_databases/middle_layer.json

      For facilities not using MML, create the JSON manually following the structure above.

Benchmarking (All Pipelines)
-----------------------------

All three pipelines share the same benchmarking tools. Run systematic benchmarks to measure accuracy:

.. code-block:: bash

   osprey channel-finder benchmark
   osprey channel-finder benchmark --queries 0,1
   osprey channel-finder benchmark --model anthropic/claude-sonnet

Benchmarks report precision, recall, F1 scores, success categorization, and consistency tracking.

.. dropdown:: Critical Best Practice: Domain Expert Involvement
   :color: warning

   **DO NOT** rely solely on developer-created test queries. Gather test queries from multiple domain experts and operators **before** finalizing your database. Developers unconsciously create queries matching their mental model -- 95%+ accuracy in development can drop to <60% in production with real users.

Configure benchmark datasets and execution settings (``runs_per_query``, ``max_concurrent_queries``, ``results_dir``) in ``config.yml`` under the ``benchmark`` key.


2.1: Framework Integration
^^^^^^^^^^^^^^^^^^^^^^^^^^

The channel finder integrates into Osprey as an MCP tool exposed via the ``control_system`` MCP server. The service layer (``osprey.services.channel_finder``) contains all channel finding logic, independently testable via CLI (``osprey channel-finder``) and benchmarks.

.. admonition:: PLACEHOLDER -- Capability-to-MCP migration details
   :class: warning

   The channel finder was previously integrated as a Python capability class (``BaseCapability`` subclass). It is now exposed as an MCP tool. Details of how Claude Code orchestrates channel finding via MCP tools will be documented here.

**Configuration:**

.. code-block:: yaml

   channel_finder:
     pipeline_mode: in_context  # or "hierarchical" or "middle_layer"


Step 4: The Service Layer Pattern
=================================

A key architectural pattern: **separate business logic from framework orchestration**.

- **Clean Boundaries**: Service layer contains pure business logic; the MCP tool layer handles orchestration
- **Independent Testing**: Test service logic directly without framework overhead
- **Reusability**: Same service powers CLI tools, MCP servers, and web APIs

.. code-block:: python

   # Service layer: osprey.services.channel_finder
   service = ChannelFinderService()
   result = await service.find_channels("beam current")
   assert "SR:CURRENT:RB" in result.channels

   # MCP tool layer
   @mcp.tool()
   async def find_channels(query: str):
       service = ChannelFinderService()
       return await service.find_channels(query)

.. tip::

   Need to customize a framework service beyond prompt overrides?
   Use ``osprey eject service channel_finder`` to copy the source to your project for modification.


Navigation
==========

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: <- Part 1: Setup
      :link: control-assistant-part1-setup
      :link-type: doc

      Return to project setup

   .. grid-item-card:: Part 3: Production ->
      :link: control-assistant-part3-production
      :link-type: doc

      Deploy and run your assistant
