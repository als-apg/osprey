.. _reference-facility-json:

==============================
The ``facility.json`` Contract
==============================

``build/facility.json`` is the description's published product: every record
of the facility in one versioned document. This page is the contract between
the build that writes it and every program that reads it -- the panel engine, a
fork of it, the framework's own readers, or a reader in another language. What
the records mean is in :ref:`architecture-facility-file`; this page covers the
version rules only.

Parts of the contract are not enforced yet. Each rule below says whether it
holds today or is a target, and which phase of the package split makes it hold.
Every reader checks the version today: the panel engine and each framework
reader in `The readers`_.

Producer
========

The description's build is the only producer. ``osprey build`` assembles the
document and writes it to ``<render root>/facility.json``. No other code writes
the file, and no reader edits it: a reader that needs a different shape asks
the build for it.

The header
==========

The document's first key is its header:

.. code-block:: json

   {"schema": "osprey.facility.facility/4", "identity": {"...": "..."}}

The header is ``osprey.facility.facility/N``, where ``N`` is a positive integer
with no leading zero. The current version is ``4``; the build takes it from
``osprey_facility.validate.FACILITY_HEADER``.

Tables a reader needs besides the records
=========================================

The document carries what a reader would otherwise derive from the
description's vocabulary, so a reader needs the document alone:

- ``signals`` is the effective signal-role table, one row per role: the
  vocabulary's roles and the ones the facility adds, sorted by name. A row
  keeps the fields its source states: ``name``, ``aliases``, ``description``,
  ``nameplate`` (the signal names the unit, not its state) and ``title`` (the
  word a screen names it by).
- ``relation_types`` is the vocabulary's relation types, sorted by name: each
  type's ``name``, the record kinds it may join, whether it is symmetric, and
  what it is single per.
- every group has a ``kind``. When no source states one, the build writes
  ``family`` and records it in the group's ``provenance.defaults``.

Version ``4`` added ``relation_types``, widened ``signals`` from the roles
the facility adds (all that version ``3`` carried) to the effective table and
made it always present, and made a group's ``kind`` always present.

Breaking changes and the bump rule
==================================

A change is **breaking** when it removes a key, renames a key, changes a key's
type, or changes what a key's value means. A breaking change bumps ``N`` by one
in the same change.

A change is **additive** when it only adds a key. An added key does not bump
``N``.

Readers ignore keys they do not know. That is what makes an added key safe: a
reader written against version ``N`` reads every later document at version
``N``, whatever keys were added since.

Two consequences for a reader:

- Never refuse a document because it holds a key you did not expect.
- Treat a key added after the build you wrote against as optional: an older
  build at the same ``N`` does not have it.

The reader rule
===============

Every reader declares the versions it reads, as a set of complete header
strings, and compares the document's ``schema`` value against that set exactly.
A document whose header is missing, malformed or not in the set is one the
reader cannot read. The framework's readers share this check in
``osprey_facility.reading``, and its message names the rebuild:

.. code-block:: text

   build/facility.json is osprey.facility.facility/3; this reader reads
   osprey.facility.facility/4 -- rebuild with osprey build

What a reader does next is its channel, one of three:

refuse
   The reader raises with that message and does nothing with the document. A
   write, a deploy or a tool call stops there; the user rebuilds and runs it
   again. Most readers refuse.
report corrupt
   The reader answers with a corrupt-source report whose detail is that
   message, so the operator sees why the source gave nothing and that a
   rebuild clears it. The channel roster reports this way.
degrade as absent
   The reader carries on as if no document were there, with a warning or a
   marked-unresolved result in place of what the document would have given;
   a rebuild restores it. Only a reader whose output stays correct without
   the document degrades: an assistant context, a screen guide, a search
   index's facility links. A write never does.

A reader does not try to read a version it did not declare, older or newer, and
does not guess from the keys present. A fork that tracks the document declares
the versions it has been tested against -- today that is ``/4`` -- and adds a
new ``N`` only after it has read that version's breaking change.

The published schema
====================

The document's shape is published as a JSON Schema generated from the
description's schema model, in ``osprey_facility/schema/_generated/`` of the
``osprey-facility`` package:

- ``facility.schema.json`` is the current shape. It is open: it does not
  refuse a key it does not name, so a document with an added key validates.
  Its ``$schema`` names the JSON Schema 2020-12 dialect and its ``$id`` the
  URL it is served under, ending in the header's ``N``. Within one ``N`` the
  current shape only grows additively, so the ``$id`` names the newest
  additive form of ``N``.
- ``facility-<N>.schema.json`` is the floor for version ``N``, written once
  when ``N`` first appears and never rewritten. It carries no ``$schema`` or
  ``$id``. ``facility-4.schema.json`` is today's floor.
- ``facility-<N>.final.schema.json`` is written at a bump of ``N``: the last
  current of the old ``N``, with its ``$id``, so the old URL keeps serving.

The docs serve these files, copied from the package at build time:

- the current schema and each final at ``schemas/facility/<N>.schema.json``;
  today that is ``https://als-apg.github.io/osprey/latest/schemas/facility/4.schema.json``;
- each floor at ``schemas/facility/<N>.floor.schema.json``.

``uv run python -m osprey_facility.schema._generate`` regenerates them, and
its ``--check`` fails when the current file is stale or a floor is missing. A
test compares the current schema with the floor of the current ``N``: a change
that is not additive -- a removed, renamed or retyped key, a newly required
key, a removed enum value -- fails until its author bumps ``N``. The test
sees shape only; a key that keeps its shape and changes its meaning is held by
review.

The schema files ship inside the ``osprey-facility`` wheel, at
``osprey_facility/schema/_generated/``, so forks and readers outside Python can
validate a document without OSPREY's source tree or the framework.
``osprey_facility.schema.published_schema()`` returns the current file;
``published_schema(N)`` returns the floor for version ``N`` and raises
``LookupError`` for a version the package ships no floor for:

.. code-block:: python

   import json

   from osprey_facility.schema import published_schema

   schema = json.loads(published_schema().read_text(encoding="utf-8"))

What holds today
================

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Rule
     - Status
     - Where it holds, or the phase that makes it hold
   * - The build writes the header ``osprey.facility.facility/N``
     - Enforced
     - ``tests/facility/test_facility_headers.py``: every facility output
       names its header, and ``facility.json``'s is ``FACILITY_HEADER``
   * - The build is the only producer
     - Holds, not checked
     - Only the build's render writes the file; no test guards it
   * - ``N`` bumps on a change of shape, not on an added key
     - Enforced
     - ``packages/osprey-facility/tests/osprey_facility_tests/test_facility_json_schema.py``:
       the current schema must be additive over the floor of the current ``N``
   * - ``N`` bumps on a change of meaning that keeps the shape
     - Holds, not checked
     - Review; no test can see a meaning
   * - The generated JSON Schema is pinned by a test
     - Enforced
     - ``packages/osprey-facility/tests/osprey_facility_tests/test_facility_json_schema.py``:
       the committed current schema equals the generator's output, a floor
       exists for the current ``N``, and the schema's header is
       ``FACILITY_HEADER``. A floor is not compared with the generator: it is
       written once and parts from the current schema at the first additive
       change
   * - The schema ships in the description's package
     - Enforced
     - ``packages/osprey-facility/tests/osprey_facility_tests/test_wheel_contents.py``:
       the built ``osprey-facility`` wheel holds both schema files where
       ``published_schema`` resolves them, and the compat module
   * - The served schema's ``$id`` is the URL it is served under
     - Enforced
     - ``tests/docs/test_published_schemas.py``: the docs build hook copies the
       package's schema files unchanged, and every served ``$id`` equals its
       served URL
   * - Readers ignore unknown keys
     - Enforced for the panel engine
     - ``packages/osprey-panels/tests/osprey_panels_tests/test_unknown_keys.py``; the framework's readers
       look up the keys they use and pass over the rest, not checked
   * - Readers check ``N`` and act on others through their channel
     - Enforced
     - ``tests/facility/test_reader_coverage.py``: every function that reads
       the document reaches the header check and is on the reader roster
       with its channel and a test that hands it a ``/3`` document; the
       panel engine in
       ``packages/osprey-panels/tests/osprey_panels_tests/test_document.py``

The readers
===========

The contract binds every reader of ``facility.json``, and every reader checks
the header today. The panel engine (``osprey_panels.document``)
refuses: it reads ``/4`` and imports no facility code, because it is the
reader a fork replaces.

Each framework reader, and the ``osprey-panels`` command's, is on the reader
roster with its channel, below. The
roster is ``tests/facility/_reader_roster.py``;
``tests/facility/test_reader_coverage.py`` holds it complete, and a docs test
holds this table equal to it.

.. list-table::
   :header-rows: 1
   :widths: 75 25

   * - Reader
     - Channel
   * - ``osprey_facility.reading.load_facility``
     - refuse
   * - ``osprey_facility.reading.check_header``
     - refuse
   * - ``osprey.runtime.guarded_run._facility_commands``
     - refuse
   * - ``osprey.simulation.model_request._corrector_delta``
     - refuse
   * - ``osprey.mcp_server.python_executor.tools.pyaml_measure._facility``
     - refuse
   * - ``osprey.mcp_server.python_executor.tools._pyaml_measure_result.measure_labels``
     - refuse
   * - ``osprey.apps.sdk.testing._deployment.RenderedTree.read``
     - refuse
   * - ``osprey_facility.records.FacilityRecords.from_file``
     - refuse
   * - ``osprey.apps.sdk._model.model_members``
     - refuse
   * - ``osprey.services.ariel_search.ingestion.facility_references.load_facility_records``
     - degrade-absent
   * - ``osprey.interfaces.okf_panel.app._load_facility_records``
     - degrade-absent
   * - ``osprey.facility.views.facts.parsed_facility_file``
     - refuse
   * - ``osprey.channel_roster.sources.read_facility_roster``
     - report-corrupt
   * - ``osprey.integrations.canopy.screen_guide._guide_facility``
     - degrade-absent
   * - ``osprey.cli.templates.claude_code._read_facility``
     - degrade-absent
   * - ``osprey.utils.facility._recorded_identity``
     - refuse
   * - ``osprey.utils.facility.facility_name``
     - refuse
   * - ``osprey.runtime._facility_records``
     - refuse
   * - ``osprey.mcp_server.channel_finder_hierarchical.server_context.ChannelFinderHierContext.initialize``
     - refuse
   * - ``osprey.mcp_server.channel_finder_middle_layer.server_context.ChannelFinderMLContext.initialize``
     - refuse
   * - ``osprey.mcp_server.channel_finder_in_context.server_context.ChannelFinderICContext.initialize``
     - refuse
   * - ``osprey.interfaces.channel_finder.app._create_lifespan``
     - refuse
   * - ``osprey.deployment.web_terminals.artifacts.resolve_render_inputs``
     - refuse
   * - ``osprey.facility.views.facts.read_facts``
     - refuse
   * - ``osprey.mcp_server.python_executor.server.pyaml_view_present``
     - refuse
   * - ``osprey.mcp_server.python_executor.tools.pyaml_measure.advertised``
     - refuse
   * - ``osprey.mcp_server.python_executor.tools._pyaml_measure_args._facts``
     - refuse
   * - ``osprey.cli.templates.claude_code.config_derived_context``
     - refuse
   * - ``osprey.apps.sdk._model.closed_orbit_models``
     - refuse
   * - ``osprey.apps.sdk._facts.read_facts``
     - refuse
   * - ``osprey.cli.build_injectors._inject_dispatch``
     - refuse
   * - ``osprey_panels.cli._graph``
     - refuse

The write guard (``osprey.runtime.guarded_run``) refuses, because a write must
never pass on a document the guard cannot vouch for.
