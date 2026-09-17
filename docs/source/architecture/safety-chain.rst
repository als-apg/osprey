:html_theme.sidebar_secondary.remove: true

.. _architecture-safety-chain:

Safety Chain
============

Every tool call that could write to a machine passes a chain of checks --- the **PreToolUse
hooks** --- before it reaches the server that would carry it out; a tool that only reads
passes without them. OSPREY's servers are started by the
harness itself and expose no network port of their own, so nothing reaches one without
passing the checks first. That guarantee belongs to the harness: a process on the host that
starts a server by hand sits outside it, which is why access to the host is part of the trust
boundary.

The checks are the first gate, not the only one. There are four ways a value reaches a
machine --- a single channel write, Python the agent wrote, a Bluesky plan, and a notebook
cell --- and each has gates of its own below the checks. Two of them can also be driven by a
person, from the BLUESKY panel or a Jupyter cell, and a person's action goes through none of
the checks. All four end at the same connector and leave the same audit record.

.. raw:: html
   :file: ../_diagrams/safety-chain.html

The checks
==========

They run in this order; a channel write meets all three, a Python run and a plan meet the
first and the last:

1. **Are writes switched on?** (``osprey_writes_check``) --- the kill switch. It refuses any
   write when writes are switched off for the machine the deployment points at ---
   ``control_system.connector.<type>.writes_enabled`` in ``config.yml``, or the
   ``control_system.writes_enabled`` a type without that key inherits --- and when the
   operator has narrowed that machine to read-only from the web terminal. It applies to channel
   writes, read-write Python runs, and queueing or starting a plan. Where no machine may be
   written at all, channel writes and plan queueing are switched off outright rather than
   refused call by call, and a Python run is always asked. Stopping a plan is never switched
   off.

2. **Is the value within limits?** (``osprey_limits``) --- checks a channel write against the
   limits database: that the channel is in it, that it may be written, and the allowed range.
   The step size is left to the connector, the only layer that can read the channel first.
   Channel writes only.

3. **Does a human approve?** (``osprey_approval``) --- the operator answers a prompt at the
   terminal. How often it asks is set per tool: ``always``, ``selective``, or ``skip``. Under
   ``selective`` the check reads the request itself and asks only when a Python run is
   read-write or its code contains a write; a channel write is always asked. When a Python run
   needs approval, the exact code is saved as a **Pre-Execution Review** notebook in the
   artifact gallery and linked from the prompt, so the operator can read what is about to run.
   The approval is the terminal prompt; the gallery only shows the code.

Every check reads the deployment's settings: which machine the deployment points at and
whether the operator has narrowed it come from the one record the deployment keeps, read
fresh on every call, so a switch made anywhere reaches the agent's next tool call; whether
writes are switched on for that machine is rendered from ``config.yml`` at build time and
takes a rebuild to change. See :ref:`web-terminal-session-posture` for the narrowing and
:doc:`/how-to/control-systems/switch-control-target` for the machine.

Four paths below the checks
===========================

One channel write
-----------------

One call, one or more channels, each with its value; one approval covers the call. The three
checks are its first gates. The control-system server then checks the limits once more and
that the machine is still the one the operator approved, and hands the values to the
connector.

Python the agent wrote
----------------------

A Python run carries a whole program, so the checks above can only judge what they can read
in it. The Python executor takes over from there. Before the code runs, it is read: dangerous
patterns are refused, code that writes into the deployment's own files is refused, and
read-only code may neither import a control-system client library nor spell a write. Then
the code runs in its own process with guards in place: read-only code cannot write even if it
tries, writes through the client libraries are checked against the limits, it cannot write
into the deployment's own files, and it is stopped after a time limit. When the code finishes, its result is saved to a record and the process ends at once,
so nothing a library does on the way out can hold it open; the executor reports the run from
that record. All nine layers are described on :doc:`python-executor`.

A Bluesky plan
--------------

The agent queues or starts a plan through its Bluesky tools, which pass the kill switch and
the approval prompt. A person reaches the same queue from the BLUESKY panel's **Start** button,
with none of the checks in the way. Both then meet the gates the queue holds:

- **A launch token.** Sending work toward hardware --- starting the queue, or adding to a queue
  that is already running --- needs a credential the deployment holds per lane, and grants to a
  role only where that role may write to the machine the lane drives.
- **Every channel the plan declares must answer first.** Before a run starts, each channel the
  plan declares it will read or write is asked to respond; if any does not, the run is refused
  before anything is written.
- **Each write is checked as it happens.** Are writes on for the machine the lane drives, and
  is the value within limits.

Stopping a plan never needs a token, anywhere. See :doc:`/how-to/bluesky/queue` for the token
and :doc:`/how-to/bluesky/write-plans` for the pre-run check.

A Jupyter cell
--------------

A cell is run by a person, so no tool call and none of the checks are involved. What guards it
are the deployment's settings, which the notebook kernel carries. The kernel starts with writes
off, and before every cell it re-reads which machine the deployment points at and whether it
may write; a switch made anywhere reaches the next cell, and narrowing the machine reaches a
cell already running, because the connector checks the narrowing on every write. A refused
write is shown in the cell, with a line saying what to do where something in the notebook
would change it. The agent may edit notebooks only in the ``notebooks`` and ``artifacts``
folders under the agent-data root, held there by a file-write guard that protects the host's
files, not the machine. See :doc:`/how-to/web-terminal/notebooks`.

What every path shares
======================

**The connector.** Every path ends at the connector, the last step before the machine. It
checks the operator's narrowing on every write, so narrowing a machine refuses at once
whichever path the write took. Where limits checking is on, it checks the value against the
limits --- the allowed range, the step size, and whether the channel may be written at all. It
sends the value and, where that channel's policy asks for it, waits for the machine to confirm,
reads the channel back and compares what it now holds with what was sent. See
:doc:`/reference/contracts/connectors`.

**The audit trail.** Every refusal and every approval prompt on every path, and every tool
call that was allowed, is written down as one line under ``var/audit/`` in the deployment
repository, by the check, the server or the Python executor that made the decision; the
connector writes none itself, the path that called it does. Each line records the write
posture in force and who acted, or which kernel. See :ref:`reference-audit-trail`.
