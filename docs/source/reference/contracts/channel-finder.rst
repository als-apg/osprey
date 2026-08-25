.. _reference-channel-finder:

========================
Channel Finder Contracts
========================

The shapes the Channel Finder reads and is configured by: the JSON each
database pipeline expects, the ``config.yml`` keys that select a pipeline and
point it at its database, and how the active pipeline reaches the agent. For
the task of choosing and running one, see
:doc:`/how-to/use-channel-finder`.


Database Schema
===============

The in-context database is a flat JSON structure loaded by
``TemplateChannelDatabase``, with standalone entries and template entries for
device families:

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

The hierarchical database instead declares the levels the pipeline navigates
and the naming pattern the addresses follow, with the tree itself underneath:

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


Configuration Reference
=======================

Key ``config.yml`` settings:

.. code-block:: yaml

   channel_finder:
     pipeline_mode: in_context  # "in_context", "hierarchical", "middle_layer", or "graph"
     pipelines:
       in_context:
         database: {type: template, path: data/channel_databases/in_context.json}
       hierarchical:
         database: {type: hierarchical, path: data/channel_databases/hierarchical.json}
       middle_layer:
         database: {type: middle_layer, path: data/channel_databases/middle_layer.json}
       # graph has no entry: it reads the `services.graphdb` store, not a file.
     benchmark:
       dataset_path: data/benchmarks/queries.json
       # Concurrency and output dir are set per run via CLI flags
       # (osprey channel-finder benchmark --concurrency / --output-dir);
       # they are not read from config.yml.

Three of the four pipelines name the database they read, so a build renders the
one block its mode needs. Graph mode renders none: ``pipeline_mode: graph`` plus
the ``services.graphdb`` block is its whole configuration.


.. _channel-finder-framework-integration:

Framework Integration
=====================

Each pipeline ships as its own MCP server package
(``channel_finder_in_context``, ``channel_finder_hierarchical``,
``channel_finder_middle_layer``, ``channel_finder_graph``), and whichever one
``channel_finder.pipeline_mode`` names in ``config.yml`` is served to the agent
under the single name ``channel-finder`` — so the tools change with the mode
but the server the agent reaches does not. It is wired into the agent's
artifacts when you run ``osprey build`` (or ``osprey build``
after editing the config). There is no public Python
``find_channels(...)`` entry point — drive the resolver from natural
language via the agent, or invoke the CLI directly:

.. code-block:: bash

   osprey channel-finder generate     # build database from template
   osprey channel-finder benchmark    # evaluate on a query dataset

.. tip::

   Use ``osprey eject service channel_finder`` to copy the channel finder
   service source into your project for custom modifications.
