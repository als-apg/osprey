=====================
Keep the Agent Record
=====================

What the agent was asked, what it answered and what it did is one record: the
``agent-record`` service keeps every entry for ``agent_record.retention_days``,
and exports each closed day into an append-only archive on the deployment host.
The files the agent produced and the plan-queue history are kept beside it: the
``archive`` service copies them into the same archive once a day. This page
says where each part lives, what deletes it, how to check on it, and what the
archive keeps.

For who may read the record, see :doc:`read-the-record`.

Where the record lives
======================

.. list-table::
   :header-rows: 1
   :widths: 22 26 30 22

   * - What it holds
     - Where
     - Deleted by
     - In the archive
   * - The agent record's entries: prompts, answers, tool calls, transcripts,
       safety decisions, dispatch runs and bridge turns
     - ``var/agent_record/`` (``backend: embedded``), or the
       ``agent_record_clickhouse_data`` volume (``backend: clickhouse``)
     - ``agent_record.retention_days`` (default 14), only for days the archive
       holds; ``osprey reset --purge-audit``
     - yes, one Parquet file per day
   * - The spool every writer appends to before the record ingests it
     - ``var/spool/<identity>/``
     - the record, once a segment is ingested; ``osprey reset --purge-audit``
     - no; what it carried is in the record
   * - Web-terminal transcripts
     - each user's ``<user>-claude-config`` volume, ``projects/``
     - Claude Code at startup, after the larger of
       ``agent_record.retention_days`` and 30 days
     - no; the record ingests them as they are written
   * - Dispatch transcripts
     - the worker's agent-data volume, ``claude-config/projects/``
     - Claude Code, after the same key
     - no; the record ingests them as they are written
   * - Host-session transcripts (``osprey chat``, ``osprey web`` on the host)
     - your own ``~/.claude/projects/``, the folder for the deployment's build
     - Claude Code, after its own setting
     - no; the record ingests them as they are written
   * - Web-terminal artifact stores
     - each user's ``<user>-agent-data`` volume, ``artifacts/`` (files and
       ``artifacts.json``)
     - deleting an artifact in the gallery, and removing the user's volumes
     - yes, files and index
   * - Dispatch artifact store
     - the worker's agent-data volume, ``artifacts/`` (files and
       ``artifacts.json``)
     - ``RETENTION_DAYS``, and deleting an artifact in the gallery
     - yes, files and index
   * - Plan-queue history
     - each lane's Redis volume
     - removing the volume
     - yes, as Redis persistence files
   * - The archive
     - ``var/archive/<YYYY-MM-DD>/``, with ``MANIFEST.jsonl``
     - nothing but ``osprey reset --purge-audit``, which removes the record's
       exports only
     - —

The larger of ``agent_record.retention_days`` and 30 days also sets how long
Claude Code keeps its file history, shell snapshots, plan files and debug logs
on these volumes. Host settings are never written.

How long the record keeps an entry
==================================

The record counts days by **arrival day**: the UTC day an entry reached the
record. ``agent_record.retention_days`` (default 14, at least 1) is the window:
an arrival day earlier than that many days before today is past retention.

Once a day, right after the export, the record removes the days past retention,
oldest first, and only the days the archive holds. It stops at the first day
the archive does not hold, so nothing is ever removed that has no lasting copy:
an export that is down for longer than the window removes nothing, and the
record grows until the export catches up. Whole days are removed, never single
entries.

A removed day stays out. A transcript Claude Code rewrites still carries the
lines of removed days, and the record skips every line older than its oldest
remaining day, so those lines are not taken in a second time. The export never
goes back to a day before the record's oldest day either.

Lowering ``agent_record.retention_days`` takes effect at the next daily run;
raising it keeps the days still held, not the ones already removed. Those are
in the archive.

Turn the archive on and off
===========================

The control-assistant preset deploys the archive service. In any other profile,
add it under ``services:``:

.. code-block:: yaml

   services:
     archive:
       template: osprey.archive

Delete the block to turn it off. ``osprey build`` and ``osprey up`` carry either
change into the deployment. Without it the record still exports its own days
into ``var/archive/``; only the artifact and plan-queue copies stop.

``services.archive.interval_seconds`` sets the time between passes (default
``86400``, one day). ``osprey health`` warns when the last pass is older than
twice that, or when it recorded errors.

Check it
========

``osprey health`` asks the record for its own account of itself and shows one
``agent_record`` group of rows. Every problem is a warning: the record keeps
working in part, and the row says where.

- ``agent_record_backend`` — the store the record writes. A warning names why
  it is not ok, or that it refused a spool pass.
- ``agent_record_spool`` — only when the spool set aside paths it would not
  read.
- ``agent_record_source.<identity>`` — one per spool directory, that is per
  writer identity. It reads **receiving** while entries arrive, and **idle**
  when the source has no live session and nothing waiting; both are fine. It
  warns when the source is **stopped** — a live session, or lines waiting,
  and nothing from it reached the record for 5 minutes — when its **backlog
  grows** (lines not yet in the record that are older than 2 minutes), when
  the record **refused** its lines in the last 24 hours, or when its writers
  logged errors or abandoned segments.
- ``agent_record_transcripts.<identity>`` — one per transcript folder, the
  same way: receiving, idle, stopped, or a folder or file that cannot be read.
- ``agent_record_export`` — how far the export to the archive is behind. One
  closed day behind is normal, because yesterday is exported by the next daily
  run; it warns at more than one closed day, when the last export failed, or
  when the archive cannot be read.

A plain ``osprey health`` on the host, with nothing in that shell pointing at
a record, shows one skipped row instead. Two host commands ask the record
directly, starting the host-local one for the call when no stack or host
session runs it:

- ``osprey agent-record health`` prints every source, transcript folder, the
  backend and the export lag, and names each problem.
- ``osprey agent-record export`` runs the catch-up export now and prints the
  days it wrote.

See :doc:`/reference/cli` for both.

Permissions
===========

The archive is the lasting half of the audit trail, and it has two writers:
agent-record exports the record's closed days into it, and the archive service
copies artifacts and Bluesky history into it. Each runs in its own container
under its own user, so ``var/archive/`` is shared the way
``var/agent_record/`` is. Every ``osprey build`` sets it to mode ``2770``:
setgid, read and write for the owner and the group, nothing for anyone else.
Its group is the one ``var/agent_record/`` has. Both writers give every
directory they create the root's mode, and every file that mode without the
setgid and execute bits, so every copy is readable by that group, artifact and
Bluesky copies included. Only agent-record and the archive service mount the
directory, and the agent's own code may not touch it.

You own the directory and belong to its group, so the commands you run on the
host, ``osprey health``, an export and ``osprey reset --purge-audit`` among
them, read and delete what either container wrote. Under rootless podman a
container's root is your own user, so its copies are owned by you. Under
rootful docker the archive service hands every path it creates to the owner and
group of ``var/archive/``.

Read it
=======

Each day directory holds the files copied that day, under
``<kind>/<name>/<path>``, the record's export of that day, under
``agent_record/entries.parquet``, and one ``MANIFEST.jsonl``. It is appended to
and never rewritten, with these kinds of line:

- ``file`` — one copied or exported file: its ``source``, its ``path`` in the
  archive, its ``size`` and ``sha256``; the record's export file also carries
  its ``day`` and ``rows``;
- ``agent_record_day`` — the record's export of that day is whole: its ``day``
  and ``rows``. It is the last line of the export, and a day with no entries
  gets it with ``rows: 0`` and no file. A day without it is not exported, and
  the next export writes that day again, over any file it left;
- ``agent_record_purged`` — the record's export of that day was purged; it
  clears the day's earlier ``agent_record_day`` lines;
- ``pass`` — the last line of every archive-service pass: when it ran, how many
  files and bytes it copied, and any errors.

Agent-record exports each UTC day once, after the day has closed, by arrival
day. It runs when the service starts and daily, and catches up every closed day
a stopped service missed, oldest first.

The archive service copies a file again only when its content changed. A file
that changed is copied whole; a second copy on the same day gets the pass time
before its suffix, for example ``artifacts.T142233Z.json``.

To restore a lane's plan-queue history, point ``redis-server --appendonly yes``
at a copied ``appendonlydir``. A copy taken mid-write can end on a torn command;
``redis-check-aof --fix`` repairs it.

Audit files from before the record
==================================

A deployment that wrote safety decisions before it had the agent record keeps
those files in ``var/audit/``. They stay on disk as they were: the record does
not read them, they are not archived, and the agent's own code may not write
them. ``osprey reset`` keeps them, and only ``osprey reset --purge-audit``
deletes them. Read them on the host with ``grep`` or ``jq``, as before.

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
- The audit files from before the record, in ``var/audit/``.
- An artifact deleted between two archive passes, or a dispatch run's
  artifacts cleared or expired before the first pass after they were written.
- Anything a user's volumes held when they are purged while the archive service
  is not running.
