=====================
Keep the Agent Record
=====================

What the agent was asked, what it answered and what it did is kept in several
places, and most of them delete on a clock of their own. The ``archive`` service
copies all of them, once a day, into an append-only tree on the deployment
host. This page lists every store, what deletes it, and what the archive keeps.

For the telemetry itself — what the agent emits and how to query it — see
:doc:`monitor-agent` and :doc:`read-the-record`.

Where the record lives
======================

.. list-table::
   :header-rows: 1
   :widths: 22 26 30 22

   * - What it holds
     - Where
     - Deleted by
     - In the archive
   * - Web-terminal transcripts
     - each user's ``<user>-claude-config`` volume, ``projects/``
     - Claude Code at startup, after
       ``claude_code.transcripts.retention_days`` (default 30)
     - yes
   * - Web-terminal artifact stores
     - each user's ``<user>-agent-data`` volume, ``artifacts/`` (files and
       ``artifacts.json``)
     - deleting an artifact in the gallery, and removing the user's volumes
     - yes, files and index
   * - Dispatch transcripts
     - the worker's agent-data volume, ``claude-config/projects/``
     - Claude Code, after the same key
     - yes
   * - Dispatch run records
     - the worker's agent-data volume, ``dispatch/*.json``; at most 200 tool
       calls, 16 KB per result
     - ``RETENTION_DAYS`` and the dashboard's clear-history
     - yes, from the next pass on — a record cleared before it is gone
   * - Dispatch artifact store
     - the worker's agent-data volume, ``artifacts/`` (files and
       ``artifacts.json``)
     - ``RETENTION_DAYS``, and deleting an artifact in the gallery
     - yes, files and index
   * - Plan-queue history
     - each lane's Redis volume
     - removing the volume
     - yes, as Redis persistence files
   * - Audit ledger
     - ``var/audit/<identity>/``
     - ``osprey reset --purge-audit``
     - yes; dot-named ``.json`` and ``.tmp`` files there are a service's working state and are
       not copied
   * - Telemetry store: the agent's logs, traces and metrics, and the audit
       ledger as rows
     - the ``telemetry_store`` volume, tables ``telemetry.*``
     - ``services.telemetry_store.retention_days`` (default 14)
     - yes, one Parquet file per table per day
   * - The archive
     - ``var/archive/<YYYY-MM-DD>/``, with ``MANIFEST.jsonl``
     - nothing in OSPREY, ``osprey reset`` included
     - —

``claude_code.transcripts.retention_days`` also sets how long Claude Code keeps
its file history, shell snapshots, plan files and debug logs. The
control-assistant preset sets it to 3650; budget disk for that.

Turn it on and off
==================

The control-assistant preset deploys the archive. In any other profile, add it
under ``services:``:

.. code-block:: yaml

   services:
     archive:
       template: osprey.archive

Delete the block to turn it off. ``osprey build`` and ``osprey up`` carry either
change into the deployment.

``services.archive.interval_seconds`` sets the time between passes (default
``86400``, one day). ``osprey health`` warns when the last pass is older than
twice that, when it recorded errors, and, with the telemetry store deployed,
when the newest archived telemetry day is older than the day before yesterday
(UTC).

Permissions
===========

The archive holds verbatim copies. ``osprey build`` creates ``var/archive/`` at
mode ``0700``; an existing directory keeps the mode it has. Every directory the
archive writes gets the root's mode, and every file that mode without execute
bits, so widening the root to ``0750`` for a group is a deliberate, per-facility
decision.

Under rootless podman the service's root is your own user, so every copy is
owned by you. Under rootful docker the service hands every path it creates to
the owner of ``var/archive/``.

Read it
=======

Each day directory holds the files copied that day, under
``<kind>/<name>/<path>``, and one ``MANIFEST.jsonl``. It is appended to and never
rewritten, with three kinds of line:

- ``file`` — one copied file: its ``source``, its ``path`` in the archive, its
  ``size`` and ``sha256``; a file exported from the telemetry store also carries
  its ``table``, ``day`` and ``rows``;
- ``telemetry_day`` — one exported day of the telemetry store, with the rows
  written per table;
- ``pass`` — the last line of every pass: when it ran, how many files and bytes
  it copied, and any errors.

A file is copied again only when its content changed. A transcript that grew is
copied whole; a second copy on the same day gets the pass time before its
suffix, for example ``abc.T142233Z.jsonl``.

.. _agent-record-telemetry-export:

The telemetry store's days
--------------------------

A UTC day is exported once, by the first pass at or after ten past midnight of
the next day, as ``<pass-day>/telemetry/<day>/<table>.parquet``: one file for
each table of the store and one for the store's own ``query_log``. A table is
read by **arrival day**, the day a row reached the store (its ``InsertedAt``),
so rows that arrive late after a store outage are in the export of the day they
arrived. ``query_log`` is read by its own ``event_date``. A day whose export
fails part-way is not recorded and is exported again by the next pass; a pass
walks back at most ``services.telemetry_store.retention_days`` days, the
oldest day the store still holds.
The archive reads the store with its own read-only account.

.. _agent-record-dedup:

Duplicate rows
--------------

Each stored row is exported once, but the store can hold a row twice. The
gateway's ledger feed is at least once: a gateway stopped between a stored batch
and its acknowledgement stores that batch again, and a gateway whose state
volume is lost reads every ledger file from the start, storing the retention
window once more. A duplicate in the store is a duplicate in the archive. Drop
them when you read, with ``LIMIT 1 BY`` on the natural key:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Table
     - Key
   * - ``audit_ledger``
     - ``User``, ``Surface``, ``ToolUseId``, ``Timestamp``, ``cityHash64(Body)``
   * - ``otel_logs``
     - ``User``, ``SessionId``, ``Timestamp``, ``cityHash64(Body)``

The queries the history page and :doc:`reconstruct-session` run use the same
keys.

To restore a lane's plan-queue history, point ``redis-server --appendonly yes``
at a copied ``appendonlydir``. A copy taken mid-write can end on a torn command;
``redis-check-aof --fix`` repairs it.

Removing a user
===============

The archive reads every user's volumes, so ``osprey users remove`` and
``osprey users prune`` with ``--archive`` or ``--purge`` run a last pass and remove
the archive container before removing the volumes. Run ``osprey build``, then
``osprey up``, to recreate it without them.

What is not archived
====================

- The Tiled catalog of a plan lane: it is beamline data, with its own
  retention and backup, not a record of what the agent was asked and did.
- Sessions of ``osprey chat`` or ``osprey web`` run on the host, which live
  under your own ``~/.claude``.
- Anything deleted between two passes: a dispatch run cleared or expired, or a
  transcript Claude Code pruned before the first pass after
  ``claude_code.transcripts.retention_days`` was set.
- Anything a user's volumes held when they are purged while the archive service
  is not running.
