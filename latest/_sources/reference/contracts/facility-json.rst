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
The panel engine checks the version today; the framework's readers do not yet.

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
``osprey.facility.validate.FACILITY_HEADER``.

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

Version ``4`` added these tables and made a group's ``kind`` always present.

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
A document whose header is missing, malformed or not in the set is refused, and
the refusal tells the user to rebuild:

.. code-block:: text

   build/facility.json is osprey.facility.facility/3; this reader reads
   osprey.facility.facility/4 -- rebuild with osprey build

A reader does not try to read a version it did not declare, older or newer, and
does not guess from the keys present. A fork that tracks the document declares
the versions it has been tested against -- today that is ``/4`` -- and adds a
new ``N`` only after it has read that version's breaking change.

The published schema
====================

The document's shape is published as a JSON Schema generated from the
description's schema model, in ``src/osprey/facility/schema/_generated/``:

- ``facility.schema.json`` is the current shape. It is open: it does not
  refuse a key it does not name, so a document with an added key validates.
- ``facility-<N>.schema.json`` is the floor for version ``N``, written once
  when ``N`` first appears and never rewritten. ``facility-4.schema.json`` is
  today's floor.

``uv run python scripts/facility_schema/json_schema.py`` regenerates them, and
its ``--check`` fails when the current file is stale or a floor is missing. A
test compares the current schema with the floor of the current ``N``: a change
that is not additive -- a removed, renamed or retyped key, a newly required
key, a removed enum value -- fails until its author bumps ``N``. The test
sees shape only; a key that keeps its shape and changes its meaning is held by
review.

The schema files ship inside the ``osprey`` wheel today, so forks and readers
outside Python can validate a document without OSPREY's source tree. In
phase 4 they ship in the description's own package.

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
     - ``tests/facility/test_facility_json_schema.py``: the current schema
       must be additive over the floor of the current ``N``
   * - ``N`` bumps on a change of meaning that keeps the shape
     - Holds, not checked
     - Review; no test can see a meaning
   * - The generated JSON Schema is pinned by a test
     - Enforced
     - ``tests/facility/test_facility_json_schema.py``: the committed schema
       and the floors equal the generator's output, and the schema's header is
       ``FACILITY_HEADER``
   * - The schema ships in the description's package
     - Target
     - Phase 4, when the description becomes its own package; today it ships
       in the ``osprey`` wheel
   * - Readers ignore unknown keys
     - Enforced for the panel engine
     - ``tests/facility/panels/test_unknown_keys.py``; the framework's readers
       look up the keys they use and pass over the rest, not checked
   * - Readers check ``N`` and refuse others
     - Enforced for the panel engine
     - ``tests/facility/panels/test_document.py`` and
       ``tests/cli/test_facility_panels.py``; the framework's readers in
       phase 3, see the readers below

The readers
===========

The contract binds every reader of ``facility.json``. The panel engine checks
the header today; each framework reader starts checking it in the phase shown,
and until then reads whatever ``N`` it finds.

.. list-table::
   :header-rows: 1
   :widths: 45 40 15

   * - Reader
     - What it reads
     - Checks ``N`` from
   * - The panel engine (``osprey.facility.panels.document``)
     - Every record and the ``signals`` and ``relation_types`` tables, to lay
       out the facility's screens
     - Enforced: reads ``/4``
   * - The write guard (``osprey.runtime.guarded_run``)
     - Each channel's ``command``, before a write is let through
     - Phase 3, first among the framework readers
   * - The channel roster (``osprey.channel_roster.sources``)
     - The channel list the roster serves
     - Phase 3
   * - The facility identity (``osprey.utils.facility``)
     - The recorded ``identity``
     - Phase 3
   * - The app SDK model (``osprey.apps.sdk``)
     - The records an app reads through ``FacilityRecords``
     - Phase 3
   * - Every other framework reader (the simulation model request, the
       Canopy screen guide, the channel finder's database, the CLI)
     - The records each one needs
     - Phase 3

The panel engine checks first because it is the reader a fork replaces: it
imports no facility code and reads the document alone. The write
guard is the first framework reader to check, because a write must never pass
on a document the guard cannot vouch for.
