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

A write also carries a name --- whose write it is. Narrowing a machine to read-only is
that person's decision about their own writes, so the name decides which narrowing a write
is judged against. :ref:`Whose write is it <architecture-safety-chain-owner>`, at the foot of
this page, says where the name comes from and what it does not cover.

.. raw:: html
   :file: ../_diagrams/safety-chain.html

The checks
==========

They run in this order; a channel write meets all three, a Python run and a plan meet the
first and the last:

1. **Are writes switched on?** (``osprey_writes_check``) --- the kill switch. It refuses any
   write when writes are switched off for the machine the deployment points at ---
   ``control_system.connector.<type>.writes_enabled`` in ``config.yml``, or the
   ``control_system.writes_enabled`` a type without that key inherits --- and when the person
   who asked has narrowed that machine to read-only from their own terminal. It applies to channel
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

Every check reads the deployment's settings: which machine the deployment points at, and
whether the person who asked has narrowed it, come from the records the deployment keeps ---
one per person --- read fresh on every call, so a switch made anywhere reaches the agent's
next tool call; whether writes are switched on for that machine is rendered from
``config.yml`` at build time and takes a rebuild to change. See
:ref:`web-terminal-session-posture` for the narrowing and
:doc:`/how-to/control-systems/switch-control-target` for the machine.

.. note::

   After upgrading a deployment, choose the machine again. Records an older version kept in
   a single file, at ``var/agent_data/control_target/control_context.json``, are not read ---
   including a narrowing that record held.

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

A Python run carries a whole program, so the per-write checks can only judge what they can read
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
- **Each write is checked as it happens.** Are writes on for the machine the lane drives, is
  the value within limits, and has the person the plan belongs to narrowed that machine.

Stopping a plan never needs a token, anywhere. See :doc:`/how-to/bluesky/queue` for the token
and :doc:`/how-to/bluesky/write-plans` for the pre-run check.

**A plan carries the name of whoever queued it.** The name rides along on the queued item, and
the lane's own container reads it back when the plan starts: the plan runs inside a task that
holds that name, so each of its writes is judged against that person's narrowing --- nobody
else's, and a narrowing made after the plan started reaches its next write. A queue row OSPREY
relays shows the name only when the item carries one.

The queue has other doors, and a plan that came through one of those carries whatever name it
claims, or none at all. OSPREY did not put that name there and cannot vouch for it. A claimed
name still binds: the plan's writes are judged against that person's narrowing, and the row
shows the name. A plan carrying no name runs at whatever the lane itself allows --- **the
lane's ceiling**.

**A start is bound to the queue the approver looked at.** The approval prompt records which
queue was shown, and the start carries that queue's identity with it; if the queue changed in
between --- a plan added, moved or removed --- the start is refused and the view refreshes
rather than running something the approver never saw. That holds for the agent's tool, the
BLUESKY panel's **Start** button and the terminal's queue bar alike.

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

**The connector.** Every path ends at the connector, the last step before the machine. On
every write it asks the deployment's records one question --- may this person write to this
machine right now --- and gets back one of three answers:

``permitted``
   nothing is in the way; the write goes on to the limits check below.

``narrowing``
   the person this write belongs to has narrowed this machine to read-only. Narrowing refuses
   at once, whichever path the write took, including a plan already running.

``control_context_unavailable``
   the records could not be read --- no record where one was expected, or a directory the
   container cannot see into. The write is refused rather than allowed, and the refusal names
   what to check.

Where the records are what refused a write, the message ends in the word they answered with,
``narrowing`` or ``control_context_unavailable``, so the reason survives being passed along ---
through a plan that failed, through a run record, into the audit line. Refusals decided
elsewhere read differently and say so: a Python run that was pinned read-only when it launched
is told to re-run the script, because a write state set since applies to the next run, not to
one already in flight.

Where limits checking is on, the connector then checks the value against the limits --- the
allowed range, the step size, and whether the channel may be written at all. It sends the value
and, where that channel's policy asks for it, waits for the machine to confirm, reads the
channel back and compares what it now holds with what was sent. See
:doc:`/reference/contracts/connectors`.

**The audit trail.** Every refusal and every approval prompt on every path, and every tool
call that was allowed, is written down as one line under ``var/audit/`` in the deployment
repository, by the check, the server or the Python executor that made the decision; the
connector writes none itself, the path that called it does. Each line records the write
posture in force and who acted, or which kernel. See :ref:`reference-audit-trail`.

.. _architecture-safety-chain-owner:

Whose write is it
=================

Where the name comes from
-------------------------

A request carries a name only because one of two doors put it there, and both put it there the
same way --- in the ``X-Osprey-Owner`` header, from the credential that got the request in,
never from what the request claimed.

- **The web terminal's proxy** strips any name a request arrives with and sets the account
  whose terminal it is.
- **The sidecar's own login gate** overwrites the name on every request it admits, from the
  credential that matched.

That header is the only place a request's name is read from: a name written anywhere else in a
request --- a field in its body, an argument to a tool --- names nobody. The account name is
not treated as a secret: it travels with proxied requests and appears in the deployment's own
logs. What the two doors protect is that nobody can choose to be someone else.

The panel's own address is a service door, not a terminal: someone who logs in there directly
is not named, and the plans they queue run at the lane's ceiling. In the single-user
containerised shape, where one sidecar is reached with the deployment-wide secret, requests
name nobody by design --- that secret says a request is allowed, not who sent it. See
:doc:`/how-to/web-terminal/operate`.

A job the OSPREY agent fires goes out through the terminal's own panel proxy. The agent never
holds the dispatcher's key; the proxy holds it, adds it on the way out and stamps the same name
the rest of that terminal's traffic carries. The worker that runs the job hands that name on to
the job's own agent session, so the job runs as the person who asked and the writes it makes
meet their narrowing. A backend behind the EVENTS panel is handed that key and that name
only when the deployment's config declares it **and** it answers on the loopback address ---
the same rule the operator secret follows. See :doc:`/how-to/agent-interfaces/event-dispatch`.

What the boundary does not cover
--------------------------------

The queue belongs to the facility, and OSPREY gates its own doors to it, not every door. Three
gaps follow from that, and a deployment should know them:

- **A plan added from inside a container.** Something already running there --- a Python run,
  or a persona whose profile allows a shell --- can reach the queue's own port and add a plan
  under any name it chooses, or none. The gap is that such a name is unvouched: OSPREY did not
  mint it, and it binds all the same, so the plan is judged against whoever it names. With no
  name it runs at the lane's ceiling.
- **A shell reaching the panel's dispatcher address.** The panel token sits in the agent's own
  environment, so the same shell can fire a job without the approval prompt. Building a persona
  that allows a shell warns about exactly this, and a sandboxed run cannot do it --- neither the
  token nor the port is there. The job still runs as the real person, under their narrowing.
- **A direct login at the panel's own address.** Unnamed, as above, and at the lane's ceiling.

**The lane's ceiling** is what a lane allows when nobody is named, and it describes the lanes
this deployment deploys. An external-worker lane (``bluesky.external:``) is a facility's own
RunEngine and devices: no OSPREY check runs inside it, so neither a person's narrowing nor the
lane's ceiling reaches its writes. The name OSPREY puts on such an item is attribution --- it
says who queued the plan, not what gated it.

Where the narrowing records live
--------------------------------

Each person's record sits in a directory of its own under the deployment's agent-data root, and
the lane and dispatch containers read that tree without being able to write to it. The tree is
shared by group, so a narrowing is readable by the accounts that share it; it is state, not a
secret. The build refuses to set the tree up on a planted symlink, and the readers do not
follow links, so nothing on the path can be swapped for a record from somewhere else. That
guarantee rests on one thing the operator owns: the agent-data root itself must not be writable
by the shared group. A deployment that moves the agent-data root onto shared storage has to
keep the root owned by the account that deploys. The arrangement uses POSIX file ownership and
runs on Linux and macOS hosts.
