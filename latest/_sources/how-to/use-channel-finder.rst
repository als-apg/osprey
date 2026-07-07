.. _how-to-channel-finder:

==============================
How to Use the Channel Finder
==============================

The Channel Finder translates natural language queries (e.g., "beam current,"
"terminal voltage") into control system addresses (e.g., ``SR:DCCT:Current``,
``TMVST``). It uses LLM-based pipelines, so a query can use everyday terms
rather than exact channel names.

.. seealso::

   Hellert et al. (2025), *From Natural Language to Control Signals*,
   `arXiv:2512.18779 <https://arxiv.org/abs/2512.18779>`_.


Choosing a Pipeline
===================

Set the active pipeline in ``config.yml``:

.. code-block:: yaml

   channel_finder:
     pipeline_mode: in_context  # or "hierarchical" or "middle_layer"

+---------------------------+----------------------------------------------+
| Pipeline                  | Best for                                     |
+===========================+==============================================+
| **In-Context**            | Small/medium systems (< few hundred channels)|
+---------------------------+----------------------------------------------+
| **Hierarchical**          | Large systems with strict naming patterns    |
+---------------------------+----------------------------------------------+
| **Middle Layer**          | Large systems organized by function (MML)    |
+---------------------------+----------------------------------------------+


In-Context Pipeline
===================

Loads the entire channel database into the LLM context for direct semantic
matching.

**How it works:** a single inner-LLM call — the complete channel database is
embedded in the system prompt and the model returns the most relevant channels
in one shot (no query-splitting or iterative-correction stage).

The database uses a flat JSON structure loaded by ``TemplateChannelDatabase``,
with standalone entries and template entries for device families:

.. code-block:: json

   {
     "channels": [
       {"template": false, "channel": "TerminalVoltageReadBack",
        "address": "TerminalVoltageReadBack",
        "description": "Actual value of the terminal potential"},
       {"template": true, "base_name": "BPM", "instances": [1, 10],
        "sub_channels": ["XPosition", "YPosition"],
        "address_pattern": "BPM{instance:02d}{suffix}",
        "description": "Beam Position Monitors"}
     ]
   }

Build a database from CSV, then validate and preview:

.. code-block:: bash

   osprey channel-finder build-database --use-llm
   osprey channel-finder validate
   osprey channel-finder preview


Hierarchical Pipeline
=====================

Navigates a nested hierarchy (system, family, device, field, subfield) using
recursive LLM-guided selection at each level.

The database defines levels and a naming pattern:

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
     "tree": { }
   }

Advanced features: navigation-only levels, friendly names via
``_channel_part``, optional levels with ``_is_leaf``, and custom separators
via ``_separator``.

Validate and preview:

.. code-block:: bash

   osprey channel-finder validate
   osprey channel-finder preview --depth 4 --sections tree,stats


Middle Layer Pipeline
=====================

A React agent explores the database using query tools
(``list_systems``, ``list_families``, ``inspect_fields``,
``list_channels``, ``get_common_names``).

The database follows MATLAB Middle Layer (MML) functional organization
(System -> Family -> Field -> ChannelNames). Convert from MML exports:

.. code-block:: bash

   python -m osprey.services.channel_finder.utils.mml_converter \
      --input path/to/mml_exports.py:MML_ao_SR \
      --output data/channel_databases/middle_layer.json


Web Interface
=============

Launch the browser-based channel explorer:

.. code-block:: bash

   osprey channel-finder web
   osprey channel-finder web --port 9000


Configuration Reference
=======================

Key ``config.yml`` settings:

.. code-block:: yaml

   channel_finder:
     pipeline_mode: in_context  # "in_context", "hierarchical", or "middle_layer"
     pipelines:
       in_context:
         database: {type: template, path: data/channel_databases/in_context.json}
       hierarchical:
         database: {type: hierarchical, path: data/channel_databases/hierarchical.json}
       middle_layer:
         database: {type: middle_layer, path: data/channel_databases/middle_layer.json}
     benchmark:
       dataset_path: data/benchmarks/queries.json
       # Concurrency and output dir are set per run via CLI flags
       # (osprey channel-finder benchmark --concurrency / --output-dir);
       # they are not read from config.yml.


.. _channel-finder-framework-integration:

Framework Integration
=====================

Each pipeline is exposed to the agent through a dedicated MCP server
(``channel_finder_in_context``, ``channel_finder_hierarchical``,
``channel_finder_middle_layer``). The active server is selected from
``channel_finder.pipeline_mode`` in ``config.yml`` and wired into the
agent's artifacts when you run ``osprey build`` (or ``osprey claude regen``
after editing the config). There is no public Python
``find_channels(...)`` entry point — drive the resolver from natural
language via the agent, or invoke the CLI directly:

.. code-block:: bash

   osprey channel-finder generate     # build database from template
   osprey channel-finder benchmark    # evaluate on a query dataset

.. tip::

   Use ``osprey eject service channel_finder`` to copy the channel finder
   service source into your project for custom modifications.
