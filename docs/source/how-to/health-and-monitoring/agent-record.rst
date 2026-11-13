=====================
Keep the Agent Record
=====================

Use this page to turn the archive on or off, set how long the record keeps
entries, and check that the record and the archive are healthy. It also lists
where each part lives, what deletes it, and what the archive keeps.

The **agent record** holds what the agent was asked, what it answered and what
it did. The ``agent-record`` service keeps every entry for
``agent_record.retention_days``. It also exports each closed day into an
append-only **archive** on the deployment host. The ``archive`` service copies
the files the agent produced and the plan-queue history into the same archive
once a day.

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

Set ``agent_record.retention_days`` (default 14, at least 1) to the number of
days the record keeps. The record counts days by **arrival day**: the UTC day
an entry reached the record. An arrival day earlier than that many days before
today is past retention.

Once a day, right after the export, the record removes the days past retention,
oldest first. It removes only the days the archive holds, and stops at the
first day the archive does not hold. Nothing is ever removed that has no
lasting copy. So an export that is down for longer than the window removes
nothing, and the record grows until the export catches up. The record removes
whole days, never single entries.

A removed day stays out. A transcript that Claude Code rewrites still carries
the lines of removed days. The record skips every line older than its oldest
remaining day, so it does not take those lines in a second time. The export
never goes back to a day before the record's oldest day either.

A lower ``agent_record.retention_days`` takes effect at the next daily run. A
higher value keeps the days the record still holds, not the ones it already
removed. Those are in the archive.

Turn the archive on and off
===========================

The demo preset deploys the archive service. To turn it on in any other
profile, add it under ``services:``:

.. code-block:: yaml

   services:
     archive:
       template: osprey.archive

To turn it off, delete the block. Either way, run ``osprey build`` and
``osprey up`` to carry the change into the deployment.

Without the archive service, the record still exports its own days into
``var/archive/``. Only the artifact and plan-queue copies stop.

``services.archive.interval_seconds`` sets the time between passes (default
``86400``, one day). ``osprey health`` warns when the last pass is older than
twice that, or when it recorded errors.

Check it
========

Run ``osprey health`` to see one ``agent_record`` group of rows: the record's
own account of itself. Every problem is a warning, because the record keeps
working in part; the row says where.

- ``agent_record_backend`` — the store the record writes. A warning names why
  it is not ok, or that it refused a spool pass.
- ``agent_record_spool`` — only when the spool set aside paths it would not
  read.
- ``agent_record_source.<identity>`` — one per spool directory, that is, per
  writer identity. It reads **receiving** while entries arrive. It reads
  **idle** when the source has no live session and nothing waiting. Both are
  fine. It warns when:

  - the source is **stopped**: a live session, or lines waiting, and nothing
    from it reached the record for 5 minutes;
  - its **backlog grows**: lines not yet in the record are older than 2
    minutes;
  - the record **refused** its lines in the last 24 hours;
  - its writers logged errors or abandoned segments.

- ``agent_record_transcripts.<identity>`` — one per transcript folder, the
  same way: receiving, idle, stopped, or a folder or file that cannot be read.
- ``agent_record_export`` — how far the export to the archive is behind. One
  closed day behind is normal, because the next daily run exports yesterday.
  It warns at more than one closed day, when the last export failed, or when
  the archive cannot be read.

On the host, a plain ``osprey health`` shows one skipped row instead when
nothing in that shell points at a record. Two host commands ask the record
directly. When no stack or host session runs the record, each command starts
the host-local one for the call:

- ``osprey agent-record health`` prints every source, transcript folder, the
  backend and the export lag, and names each problem.
- ``osprey agent-record export`` runs the catch-up export now and prints the
  days it wrote.

See :doc:`/reference/cli` for both.

.. _agent-record-read-a-session:

Read a session
==============

The Activity page shows what the agent did in one session and what was decided
about each call. It reads the record, so a past session reads the same as the
live one. In the web terminal, **☰ More** › **Activity log** opens it on the
session in the terminal; its sidebar lists your sessions. It has four views:

- **Timeline** — prompts, answers and tool calls in order, a subagent's calls
  under the call that started it. Open a call for its arguments, its result
  and the safety decisions on it.
- **Tool calls** — every call as a table, built-in tools included. Sort it, or
  narrow it to **Control system and services** or to **Built in**.
- **Safety** — how many calls were allowed without asking and by what, how
  many asked the operator, and how many were refused without asking; each
  approval with what was asked and how long the answer took; and the calls a
  server gate filed an entry for.
- **Artifacts** — what the session saved, each with the call that saved it.

.. figure:: /_static/screenshots/activity_safety_light.png
   :class: only-light
   :figclass: only-light
   :alt: The Safety view of one session: 33 calls allowed, one approved by the operator after 31.1 s, none denied, and the 18 calls a server gate checked.
   :width: 100%

   The **Safety** view of a Bluesky grid scan on the virtual accelerator. Of 34
   calls, one asked the operator. Captured with OSPREY |captured_activity_safety|.

.. figure:: /_static/screenshots/activity_safety_dark.png
   :class: only-dark
   :figclass: only-dark
   :alt: The Safety view of one session: 33 calls allowed, one approved by the operator after 31.1 s, none denied, and the 18 calls a server gate checked.
   :width: 100%

   The **Safety** view of a Bluesky grid scan on the virtual accelerator. Of 34
   calls, one asked the operator. Captured with OSPREY |captured_activity_safety|.

The chip in the toolbar says how much of this the record vouches for. Click it
for the reason:

- **Record complete** — the record returned the whole session.
- **Record incomplete** — the read did not return the whole session. What is
  shown is correct, but entries may be missing.
- **Record unavailable** — the session could not be read. The page keeps the
  last reading it has of that session and says so, offers **Try again**, and
  tries again by itself every 10 seconds.

The session in the terminal is read again every 10 seconds. A session picked
from the sidebar is read again only while it is live.

Permissions
===========

``var/archive/`` is shared by two writers through its group, the same way
``var/agent_record/`` is. The archive is the lasting half of the audit trail.
Its two writers each run in their own container under their own user:

- agent-record exports the record's closed days into it;
- the archive service copies artifacts and Bluesky history into it.

The modes are:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Path
     - Mode
   * - ``var/archive/``
     - ``2770``, set by every ``osprey build``: setgid, read and write for the
       owner and the group, nothing for anyone else. Its group is the one
       ``var/agent_record/`` has.
   * - Each directory either writer creates
     - The root's mode.
   * - Each file either writer creates
     - The root's mode without the setgid and execute bits.

So that group can read every copy, artifact and Bluesky copies included. Only
agent-record and the archive service mount the directory, and the agent's own
code may not touch it.

You own the directory and belong to its group. So the commands you run on the
host can read and delete what either container wrote. These include
``osprey health``, an export and ``osprey reset --purge-audit``.

- Under rootless podman, a container's root is your own user, so you own its
  copies.
- Under rootful docker, the archive service hands every path it creates to the
  owner and group of ``var/archive/``.

Read it
=======

Each day directory holds:

- the files copied that day, under ``<kind>/<name>/<path>``;
- the record's export of that day, under ``agent_record/entries.parquet``;
- one ``MANIFEST.jsonl``.

The manifest is appended to and never rewritten. It has these kinds of line:

- ``file`` — one copied or exported file: its ``source``, its ``path`` in the
  archive, its ``size`` and ``sha256``; the record's export file also carries
  its ``day`` and ``rows``;
- ``agent_record_day`` — the record's export of that day is whole: its ``day``
  and ``rows``. It is the last line of the export. A day with no entries gets
  it with ``rows: 0`` and no file. A day without it is not exported, and the
  next export writes that day again, over any file it left;
- ``agent_record_purged`` — the record's export of that day was purged; it
  clears the day's earlier ``agent_record_day`` lines;
- ``pass`` — the last line of every archive-service pass: when it ran, how many
  files and bytes it copied, and any errors.

Agent-record exports each UTC day once, by arrival day, after the day has
closed. It runs when the service starts and then daily. It catches up every
closed day a stopped service missed, oldest first.

The archive service copies a file again only when its content changed. It
copies a changed file whole. A second copy on the same day gets the pass time
before its suffix, for example ``artifacts.T142233Z.json``.

To restore a lane's plan-queue history, point ``redis-server --appendonly yes``
at a copied ``appendonlydir``. A copy taken mid-write can end on a torn command;
``redis-check-aof --fix`` repairs it.

Audit files from before the record
==================================

A deployment that wrote safety decisions before it had the agent record keeps
those files in ``var/audit/``. They stay on disk as they were:

- the record does not read them;
- they are not archived;
- the agent's own code may not write them.

``osprey reset`` keeps them, and only ``osprey reset --purge-audit`` deletes
them. To read them, use ``grep`` or ``jq`` on the host, as before.

Removing a user
===============

``osprey users remove`` and ``osprey users prune`` with ``--archive`` or
``--purge`` run a last archive pass first, because the archive reads every
user's volumes. They then remove the archive container before they remove the
volumes. To recreate the archive container without those volumes, run
``osprey build``, then ``osprey up``.

What is not archived
====================

- The Tiled catalog of a plan lane: it is beamline data, with its own
  retention and backup, not a record of what the agent was asked and did.
- The audit files from before the record, in ``var/audit/``.
- An artifact deleted between two archive passes, or a dispatch run's
  artifacts cleared or expired before the first pass after they were written.
- Anything a user's volumes held when they are purged while the archive service
  is not running.
