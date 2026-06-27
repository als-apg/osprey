.. _how-to-facility-knowledge:

==================
Facility Knowledge
==================

The Facility Knowledge system gives the OSPREY agent on-demand access to
structured narrative knowledge about your facility — subsystems, devices,
operational procedures, physics notes, and facility-specific references.

It uses a two-altitude model:

* ``facility.md`` — always loaded into the agent's context; contains only
  stable **identity** information (facility name, type, mission).
* **OKF bundle** — a directory of markdown concept documents; content is
  fetched on demand via ``list_concepts``, ``read_concept``, and ``search``
  tools exposed by the ``osprey_facility_knowledge`` MCP server.

This keeps the agent's context window lean for routine tasks while making
the full knowledge base available whenever the agent (or you) needs it.


How It Works
============

The ``osprey_facility_knowledge`` MCP server reads an OKF bundle at startup.
The bundle is a directory tree of markdown files with YAML frontmatter.
Each file is a *concept document* describing one topic (a subsystem,
a device class, an operational procedure, etc.).

.. code-block:: text

   data/facility_knowledge/
   ├── index.md                      ← bundle root index (auto-generated)
   ├── subsystems/
   │   ├── index.md                  ← sub-directory index (auto-generated)
   │   ├── power-supply-system.md
   │   └── timing-system.md
   ├── devices/
   │   ├── index.md
   │   └── magnet-power-supply.md
   └── procedures/
       ├── index.md
       └── beam-startup.md

The agent can:

* ``list_concepts`` — browse the index to discover what topics exist
* ``read_concept`` — retrieve a specific document by its bundle-relative path
* ``search`` — full-text search across all concept documents
* ``draft_concept`` — author a new concept document (requires human approval)


Configuring the Bundle Path
===========================

Point the server at your bundle by adding a ``facility_knowledge`` block to
``config.yml``:

.. code-block:: yaml

   facility_knowledge:
     bundle_path: data/facility_knowledge

Relative paths are resolved against the directory containing ``config.yml``
(the project root).  For the ``control_assistant`` preset the example bundle
is scaffolded into ``data/facility_knowledge/`` automatically.  Replace the
example content with your own facility's documents.

.. note::

   If ``facility_knowledge.bundle_path`` is missing, or points to a directory
   that does not exist, the ``osprey_facility_knowledge`` server still starts
   but logs a warning (missing key) or error (missing directory) and leaves the
   bundle uninitialised.  Its ``list_concepts``, ``read_concept``, ``search``,
   and ``draft_concept`` tools then return a ``server_not_initialised`` error on
   every call until the path is fixed.


Authoring Concept Documents
============================

Each concept document is a markdown file with YAML frontmatter at the top:

.. code-block:: markdown

   ---
   type: Subsystem
   title: Power Supply System
   description: DC power supplies for all storage-ring magnets.
   tags: [magnets, power-supplies, PSS]
   ---

   # Power Supply System

   ...document body...

Required frontmatter fields (authoring level):

+------------------+-------------------------------------------------------+
| Field            | Description                                           |
+==================+=======================================================+
| ``type``         | Document type — see conventional types below          |
+------------------+-------------------------------------------------------+
| ``title``        | Human-readable title                                  |
+------------------+-------------------------------------------------------+
| ``description``  | One-sentence summary used in indexes and search       |
+------------------+-------------------------------------------------------+

Conventional ``type`` values (free-form, but the following are recommended
for consistent index grouping):

* ``Subsystem`` — major facility subsystem (power supply, timing, vacuum…)
* ``Device`` — a device class or family of hardware components
* ``Procedure`` — operational procedure or checklist
* ``PhysicsNote`` — accelerator/beam physics explanation
* ``Reference`` — external document pointer, standard, or glossary entry

Optional fields: ``tags`` (list of strings), ``related`` (list of
bundle-relative paths to related docs).

Cross-links between documents use bundle-relative paths:

.. code-block:: markdown

   See [Timing System](/subsystems/timing-system.md) for synchronisation details.


.. _knowledge-cli:

``osprey knowledge`` CLI
========================

The ``osprey knowledge`` command group provides three operations for managing
an OKF bundle from the terminal.

.. code-block:: console

   $ osprey knowledge --help
   Usage: osprey knowledge [OPTIONS] COMMAND [ARGS]...

     Manage OKF facility knowledge bundles.

   Commands:
     regen-index    Regenerate index.md files throughout an OKF bundle.
     validate       Validate all OKF documents in a bundle.
     seed-from-ttl  Seed OKF stub documents from a NARAD/als-ontology TTL file.


regen-index
-----------

Regenerates ``index.md`` files throughout the bundle.  Run this after adding
or removing concept documents:

.. code-block:: console

   $ osprey knowledge regen-index data/facility_knowledge

When called without an argument the bundle path is read from
``facility_knowledge.bundle_path`` in ``config.yml``.  The command is
idempotent — a second run produces bit-identical output.


validate
--------

Validates every ``*.md`` file in the bundle and reports all failures in one
pass (collect-all, never aborts on the first error):

.. code-block:: console

   $ osprey knowledge validate data/facility_knowledge
   All files in data/facility_knowledge are valid.

Concept documents are checked at the *authoring* level (``type``, ``title``,
and ``description`` must be present and non-empty).  Index files are checked
for frontmatter compliance (OKF §6/§11).  The command exits 0 on a clean
bundle, 1 if any file fails.


seed-from-ttl
-------------

Seeds one OKF stub document per device node in a NARAD/als-ontology
Turtle file.  Requires the ``knowledge`` extra:

.. code-block:: console

   $ pip install "osprey-framework[knowledge]"
   $ osprey knowledge seed-from-ttl devices.ttl data/facility_knowledge

Idempotency rules applied per stub:

* **File absent** — write and report ``written``.
* **File present, same body** — skip and report ``unchanged``.
* **File present, body differs, no** ``--force`` — skip and report
  ``differs, use --force``.
* **File present, body differs,** ``--force`` set — overwrite and report
  ``overwritten``.

Use ``--force`` only when you want to reset hand-edited stubs to the
auto-generated content.

.. warning::

   ``--force`` overwrites existing stubs.  Omit it to protect hand-edited
   documents.


.. _regenerating-indexes:

Regenerating Indexes
====================

After adding or removing concept documents, regenerate the index files:

.. code-block:: console

   $ osprey knowledge regen-index data/facility_knowledge

Or via the Python API:

.. code-block:: python

   from osprey.services.facility_knowledge.okf.index import regenerate_indexes
   from pathlib import Path

   regenerate_indexes(Path("data/facility_knowledge"))

The root ``index.md`` includes an ``okf_version`` frontmatter key; sub-directory
indexes contain no frontmatter (they are consumed by the server, not the agent
directly).

.. note::

   ``regen-index`` is idempotent — running it twice produces the same
   output.  It never touches concept documents, only index files.


.. _drafting-concepts:

Drafting Concepts from an Agent Session
========================================

The ``draft_concept`` tool lets you author new knowledge documents directly
from a conversation with the OSPREY agent.  Because it writes to your project's
knowledge bundle, it requires **human approval** before executing.

Example agent prompt:

.. code-block:: text

   Draft a concept document for the vacuum system interlock procedure —
   how to respond to a sudden pressure spike in sector 4.

The agent will call ``draft_concept`` with a proposed path, frontmatter, and
body.  You review and approve in the standard approval dialog; the document
is written into the bundle.  ``draft_concept`` does not rebuild the index, so
run ``osprey knowledge regen-index`` afterwards for the new document to appear
in ``list_concepts`` and ``search``.


The ``facility-knowledge`` Subagent
=====================================

When the ``facility-knowledge`` subagent is enabled in a project's config, the
main OSPREY agent delegates facility knowledge questions to a dedicated
specialist rather than handling them inline.  The subagent has access only to
the three read-only MCP tools (``list_concepts``, ``read_concept``, ``search``)
and follows a fixed retrieval strategy: list first to orient, search to narrow,
read to synthesize, then submit the result.

The subagent is enabled by default in the ``control_assistant`` preset.  To
enable or disable it in a generated project, set the agent's ``enabled`` flag
under ``claude_code.agents`` in ``config.yml``:

.. code-block:: yaml

   claude_code:
     agents:
       facility-knowledge:
         enabled: false   # disable the subagent (omit to keep it enabled)

.. note::

   The subagent never writes to the bundle.  Authoring new concept documents
   is done via the ``draft_concept`` MCP tool in the main agent session
   (see :ref:`drafting-concepts`) or directly via ``osprey knowledge
   seed-from-ttl`` (see :ref:`knowledge-cli`).


Replacing the Example Bundle
==============================

The ``control_assistant`` preset ships an Example Research Facility (ERF)
bundle under ``data/facility_knowledge/``.  To replace it with your own
facility's knowledge:

1. Delete the example concept documents (keep the directory structure as a
   guide, or start fresh).
2. Author concept documents for your facility's subsystems, devices,
   procedures, physics notes, and references.
3. Run ``osprey knowledge regen-index data/facility_knowledge``
   (see :ref:`regenerating-indexes`) to rebuild the index files.
4. Update ``facility.md`` with your facility's name, type, and mission
   (the identity section at the top).

There is no required structure beyond the top-level directory.  Organise
sub-directories however makes sense for your facility.

.. seealso::

   :doc:`build-profiles`
      How to assemble facility-specific OSPREY projects with custom data bundles.

   :doc:`/architecture/mcp-servers`
      MCP servers provided by the framework, including ``osprey_facility_knowledge``.
