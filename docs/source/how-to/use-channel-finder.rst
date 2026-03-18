.. _how-to-channel-finder:

==============================
How to Use the Channel Finder
==============================

The Channel Finder translates natural language queries (e.g., "beam current,"
"terminal voltage") into control system addresses (e.g., ``SR:DCCT:Current``,
``TMVST``). It uses LLM-powered pipelines so operators do not need to memorize
technical naming conventions.

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

**Pipeline stages:** query splitting, semantic matching against full database,
iterative validation/correction.

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

Advanced features (v0.9.4+): navigation-only levels, friendly names via
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
``list_channel_names``, ``get_common_names``).

The database follows MATLAB Middle Layer (MML) functional organization
(System -> Family -> Field -> ChannelNames). Convert from MML exports:

.. code-block:: bash

   python -m osprey.services.channel_finder.utils.mml_converter \
      --input path/to/mml_exports.py \
      --output data/channel_databases/middle_layer.json


Interactive Testing and Benchmarking
====================================

All pipelines share the same CLI for interactive testing:

.. code-block:: bash

   osprey channel-finder

Run benchmarks with domain-expert queries before production deployment:

.. code-block:: bash

   osprey channel-finder benchmark
   osprey channel-finder benchmark --queries 0,1 --model anthropic/claude-sonnet

Aim for >90 % F1 score. Gather test queries from multiple operators --
developer-only benchmarks often overestimate production accuracy.


Configuration Reference
=======================

Key ``config.yml`` settings:

.. code-block:: yaml

   channel_finder:
     pipeline_mode: in_context  # "in_context", "hierarchical", or "middle_layer"
     pipelines:
       in_context:
         database: {type: template, path: data/channel_databases/in_context.json}
         processing: {chunk_dictionary: false, chunk_size: 50}
       hierarchical:
         database: {type: hierarchical, path: data/channel_databases/hierarchical.json}
       middle_layer:
         database: {type: middle_layer, path: data/channel_databases/middle_layer.json}
     benchmark:
       dataset_path: data/benchmarks/datasets/benchmark.json
   benchmark:
     execution: {max_concurrent_queries: 5}
     output: {results_dir: data/benchmarks/results}


.. _channel-finder-framework-integration:

Framework Integration
=====================

.. admonition:: PLACEHOLDER: CAPABILITY-TO-MCP MIGRATION
   :class: warning

   **Old content (line 2219):** "The channel finder integrates into OSPREY as a capability... @capability_node class ChannelFindingCapability(BaseCapability)"
   **New equivalent:** Needs human judgment
   **Why this is fuzzy:** The old capability class pattern (BaseCapability, capability_node, CapabilityContext, provides/requires) has been replaced by MCP tools, but the exact MCP tool interface for channel finding is not specified in the rename map.
   **Action needed:** Document how the channel finder is exposed as an MCP tool and how downstream tools (e.g., channel read) consume its results.

The ``ChannelFinderService`` class remains the core business logic layer and
can be used independently of the framework:

.. code-block:: python

   service = ChannelFinderService()
   result = await service.find_channels("beam current")

.. tip::

   Use ``osprey eject service channel_finder`` to copy the channel finder
   service source into your project for custom modifications.
