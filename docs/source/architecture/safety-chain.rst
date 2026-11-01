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

The checks are the first gate, not the only one. There are five ways a value reaches a
machine through the connector --- a single channel write, Python the agent wrote, a pyAML tool
the executor runs, a Bluesky plan, and a notebook cell --- and each has gates of its own below
the checks. Two of them can also be driven by a person, from the BLUESKY panel or a Jupyter
cell, and a person's action goes through none of the checks. All five end at the same
connector and leave the same audit record. A deployment that enables the ``phoebus`` server
and sets ``phoebus.agent_access: read_write`` has one more: ``phoebus_drive`` passes the checks and then
the Phoebus product writes the value through its own connections, not through the connector
--- see :ref:`A Phoebus drive <architecture-safety-chain-phoebus-drive>`.

A write also carries a name --- whose write it is. A machine's write switch is one login's,
so the name decides whose tier and whose switch a write is judged against. :ref:`Whose write is it <architecture-safety-chain-owner>`, at the foot of
this page, says where the name comes from and what it does not cover.

.. raw:: html
   :file: ../_diagrams/safety-chain.html

The checks
==========

They run in this order; a channel write meets all three; a Python run, a plan and a Phoebus
drive meet the first and the last:

1. **May this login write to this machine?** (``osprey_writes_check``) --- the write
   verdict: the login's tier, then the owner, then that login's switch for the machine the
   deployment points at (:ref:`write-safety-config`). It applies to channel writes, Phoebus
   drives, read-write Python runs, and queueing or starting a plan. On a ``never`` tier the
   build denies those tools outright rather than refusing them call by call, and a Python run
   is always asked. Stopping a plan is never refused.

2. **Is the value within limits?** (``osprey_limits``) --- checks a channel write against the
   limits database: that the channel is in it, that it may be written, and the allowed range.
   The step size is left to the connector, the only layer that can read the channel first.
   Channel writes only.

3. **Does a human approve?** (``osprey_approval``) --- for a governed write the switch's
   position decides: at *asked* the operator answers a prompt at the terminal, at *allowed*
   the write runs unprompted on a record naming the switch. Every other tool keeps its own
   policy: ``always``, ``selective``, or ``skip``. When a Python run
   needs approval, the exact code is saved as an **Approval preview** notebook in the
   artifact gallery and linked from the prompt, so the operator can read what is about to run.
   The approval is the terminal prompt; the preview only shows the code.

   **What a prompt guarantees (stage 1).** A governed write, and a Phoebus drive where its
   policy asks, go ahead only on the record the hook left for that call; a call the hook
   never saw is refused by the server that
   would carry it out. That is all it claims: the record is checked on the agent's own host,
   by processes of the agent's own user, so it binds every tool path and does not contain code
   already running there (see :ref:`architecture-safety-chain-residual`). Python code that names the store holding the machine's switch is asked
   in every position and refused where nothing can ask, as in a dispatch run. The executor
   checks it again before the code runs, read-only runs included.

Which machine the deployment points at, and the login's switch position for it, come
from the records the deployment keeps --- one per login --- read fresh on every call, so a
flip reaches the agent's next tool call. The tier is rendered at build time and takes a
rebuild to change. See :ref:`web-terminal-session-posture` for the switch and
:doc:`/how-to/control-systems/switch-control-target` for the machine.

.. _architecture-safety-chain-order:

Enforcement order
-----------------

A write meets its gates in this order, and the first that refuses ends it:

1. **The render.** A ``never`` tier denies the write tools before the agent can call them.
2. **The hooks**, above: the verdict, the limits, the prompt.
3. **The server** carrying the call checks that the prompt recorded it, and that the machine
   is still the one approved.
4. **The connector** asks the verdict again on every put, then the limits, then confirms.
5. **The network.** Where a machine has a read-only gateway, a connector whose switch is off
   is pointed at it, so the gateway refuses a put that got past everything else. A machine
   with only a write gateway has the software layers alone, and the build says so.

The hooks and the connector share one verdict and one order, so a write refused at either
names the same reason.

Where a watchdog has flagged the session, the hook's verdict (gate 2), the server (gate 3) and
the connector (gate 4) each refuse the write with ``session_held``. The approval prompt does
not answer a flag, so a write approved while the flag is open is refused at the server. See
:ref:`Safety watchdogs <architecture-safety-watchdogs>`.

.. _architecture-safety-watchdogs:

Safety watchdogs
================

.. raw:: html
   :file: ../_diagrams/safety-watchdogs.html

A **watchdog** is a check a facility writes, in plain language, that judges each operator
prompt in the background: is this session heading somewhere it should not? A deployment may
enable any number, off by default, and each uses one of two engines:

- **The decision engine** asks a decision model (TypeSafe Jev) to pick among options the
  facility wrote. It sees the watchdog's instructions and the session's recent history, and
  nothing else, and usually answers within seconds.
- **The agent engine** runs a read-only agent, which also sees the watchdog's references and
  facility knowledge and can read the machine. It is slower and costs more, and each run is
  capped in time and cost. Every write or edit it attempts is refused and recorded.

A watchdog answers *clear*, *flagged* or *unavailable*. Nothing it says means an action is
safe, so no record or screen says so: a clear verdict is logged and not shown. An unavailable
watchdog raises no dialog and no hold, and the terminal shows it as unavailable.

**The hold.** A flag holds the writes of the agent in that session, by every route the agent
has: write tools, executed scripts, plans queued through its tools, ``phoebus_drive``. Each
flag stays open until an operator answers it, whether or not anyone is attached; a reload or
a reattach brings the dialog back. The hold is one more reason to refuse, and it loosens
nothing: every other gate on this page still applies once it is gone. Stopping a plan, and
switching the control target, are never held.

**A watchdog is advisory, not a gate on the first write.** The prompt goes to the agent at
once, because a watchdog must never delay it, so a verdict can land after a write has already
run. The hold covers the time from the flag to the answer. A fooled or failed watchdog fails
to warn, and that is all it can do.

**The dialog** names each watchdog, the prompt it judged and its reason, and flags raised
together share one dialog. Only a click answers it. **Stop** interrupts the turn and waits;
when the turn has ended and nothing the agent started still runs, the flags are released. If
not, they stay open, the dialog says the turn did not stop, and it lists what still runs.
**Continue** releases the flags and records who chose to proceed. The next write then meets
the checks above, including the approval prompt.

What watchdogs do not cover
---------------------------

- **Only the prompt.** A watchdog judges when a prompt arrives, not after each tool call or
  at the end of a turn. The decision engine sees only about the last 30,000 tokens of a
  session, and it does not treat what a tool returned as hostile.
- **Only the agent's writes.** A write the operator makes directly, from a notebook cell or
  the BLUESKY panel, is not held.
- **Not motion already running.** A plan the agent started keeps running; stop it with the
  existing tools.
- **Only the web terminal holds.** ``osprey chat`` or a bare ``claude`` in a built project
  records its flags and is never held, because no dialog could answer them.
- **Only an operator's prompts.** Dispatch jobs, webhooks and ``osprey query`` trigger no
  watchdog.
- **Not code set on releasing its own hold.** The hold record sits with the deployment's other
  state records, and the agent's file tools and shell are denied its directory. Code running
  as the agent's user that deliberately rewrites the record can still release its own hold
  with no audited Continue, the same boundary as :ref:`the switch's record
  <architecture-safety-chain-residual>`.

Setting one up
--------------

A watchdog is an entry in the profile's ``watchdogs:`` block plus an instructions file at
``watchdogs/<name>/instructions.md``, and ``enabled: true`` and a rebuild turn it on. Run
``osprey watchdog replay`` over labelled prompts before enabling one: the thresholds are not
calibrated for any facility. A decision watchdog sends session history to TypeSafe (hosted in
the US) or OpenRouter; an agent watchdog uses the deployment's own LLM provider. Each adds its
cost to every prompt.

:doc:`/how-to/safety-watchdogs` has the steps, and :ref:`reference-watchdog` has the exact
behaviour.

Five paths below the checks
===========================

One channel write
-----------------

One call, one or more channels, each with its value; one approval covers the call. The three
checks are its first gates. The control-system server then checks the limits once more and
that the machine is still the one the operator approved, and hands the values to the
connector.

.. raw:: html
   :file: ../_diagrams/write-sequence.html

Python the agent wrote
----------------------

A Python run carries a whole program, so the per-write checks can only judge what they can read
in it. The Python executor takes over from there. Before the code runs, it is read: dangerous
patterns are refused, code that writes into the deployment's own files is refused, and
read-only code may neither import a control-system client library nor spell a write;
read-write code that plainly spells a client library's own put is refused by the approval hook
before anyone is asked to approve it. Before a read-write run starts, the executor
checks that the approval prompt recorded this call and that what the call carries matches
what it recorded; otherwise nothing runs. This check runs beside the script on one host, not
out of the script's reach. A deployment can also turn on code review: a second AI agent then
reads the code, following instructions the deployment writes itself, and says whether it is
safe to run. Depending on how the deployment set it up, unsafe code is either stopped or
allowed through with a warning (:ref:`architecture-safety-chain-code-review`). Then the code
runs in its own process with guards in place: read-only code cannot write even if it tries, read-write code writes through the
connector only because a client library's own put is refused (a PVAccess put, which the
connector cannot carry yet, is limits-checked instead), it cannot write into the
deployment's own files, and it is stopped after a time limit. When the code finishes, its result is saved to a record and the process ends at once,
so nothing a library does on the way out can hold it open; the executor reports the run from
that record. All ten layers are described on :doc:`python-executor`.

.. _architecture-safety-chain-pyaml-tool:

A pyAML tool
------------

``pyaml_measure`` runs one of pyAML's tuning tools --- an orbit response, a dispersion, a tune
or a chromaticity routine; pyAML, the Python accelerator middle layer, calls them measurements
--- on the control target. It is not how the agent measures: the agent reads channels, runs
its own Python and queues plans for that. It is the one path on which the executor runs code
OSPREY ships rather than code the agent wrote: the pyAML view the build rendered from the
facility description, with the tool's settings from ``data/facility/measurement/``. The call
passes the write verdict and the approval prompt. The prompt lists the model, the kind, the
digest of the view the tool runs on and every setpoint the call steps; a call that would step
more setpoints than a prompt can list is refused without asking. The tool then runs in the
same sandbox as the agent's Python, under the same guards, and each setpoint it moves is an
ordinary write through the connector --- under the target's run lock; the result lists each
setpoint the call moved with its value before the call
(:ref:`Guarded Runs <architecture-safety-chain-guarded-run>`). A pyAML write anywhere else,
``execute`` included, is refused. See :doc:`/how-to/measure-with-pyaml`.

A Bluesky plan
--------------

The agent queues or starts a plan through its Bluesky tools, which pass the write verdict and
the approval prompt. A person reaches the same queue from the BLUESKY panel's **Start** button,
with none of the checks in the way. Both then meet the gates the queue holds:

- **A launch token.** Sending work toward hardware --- starting the queue, or adding to a queue
  that is already running --- needs a credential the deployment holds per lane, and grants to a
  role only where that role may write to the machine the lane drives.
- **Every channel the plan declares must answer first.** Before a run starts, each channel the
  plan declares it will read or write is asked to respond; if any does not, the run is refused
  before anything is written.
- **Each write is checked as it happens.** The verdict for the plan's owner on the machine
  the lane drives, then the limits.

Stopping a plan never needs a token, anywhere. See :doc:`/how-to/bluesky/queue` for the token
and :doc:`/how-to/bluesky/write-plans` for the pre-run check.

**A plan carries the name of whoever queued it.** The name rides along on the queued item, and
the lane's own container reads it back when the plan starts: the plan runs inside a task that
holds that name, so each of its writes is judged against that person's tier and switch ---
nobody else's, and a flip made after the plan started reaches its next write. A queue row OSPREY
relays shows the name only when the item carries one.

The queue has other doors, and a plan that came through one of those carries whatever name it
claims, or none at all. OSPREY did not put that name there and cannot vouch for it. A claimed
name still binds: the plan's writes are judged against that person's tier and switch, and the
row shows the name. A plan carrying no name writes nothing: it is refused ``owner_unknown``.

**A start is bound to the queue the approver looked at.** The approval prompt records which
queue was shown, and the start carries that queue's identity with it; if the queue changed in
between --- a plan added, moved or removed --- the start is refused and the view refreshes
rather than running something the approver never saw. That holds for the agent's tool, the
BLUESKY panel's **Start** button and the terminal's queue bar alike.

A Jupyter cell
--------------

A cell is run by a person, so no tool call and none of the checks are involved. What guards it
are the deployment's settings, which the notebook kernel carries. Before every cell the kernel
re-reads which machine the deployment points at, and the connector asks the verdict on every
write, so moving the switch to off reaches a cell already running. A cell reaches
the machine through ``osprey.runtime``: importing a control-system client library, or loading
one through ``ctypes`` or ``cffi``, is refused whatever the chip says, for reads as for writes,
so a cell written the ordinary way reads and writes through the connector. The block is
cooperative, and its gaps are listed with the others on this page. A refused
write is shown in the cell, with a line saying what to do where something in the notebook
would change it. The agent may edit notebooks only in the ``notebooks`` and ``artifacts``
folders under the agent-data root, held there by a file-write guard that protects the host's
files, not the machine. With code review turned on, the same reviewing agent reads a code cell
the OSPREY agent writes into a notebook before it is saved, and can stop it or warn about it
the same way; cells a person writes are never reviewed
(:ref:`architecture-safety-chain-code-review`). See :doc:`/how-to/web-terminal/notebooks`.

.. _architecture-safety-chain-guarded-run:

Guarded Runs
============

None of the five paths remembers what it moved. Each write is checked on its own, and a
script, a plan or a cell that stops half way leaves the machine where it stopped. A *guarded
run* adds a lock around a run's writes, and only the two paths that run code in the
executor's sandbox take part in it: Python the agent wrote and a pyAML tool. A channel write,
a Bluesky plan and a notebook cell do not hold the lock, and nothing puts back what any path
moved.

**One run per machine.** A read-write Python run and a pyAML tool run each hold a lock on
their control target for as long as they run, from whichever terminal, job or container of
the deployment they were launched. A second run on that target, of either kind, is refused
before it starts, naming the holder. The lock is between these two kinds of run only; it keeps
no channel write, plan or cell off the machine.

**What a stopped tool leaves.** Nothing is written back. A pyAML tool that stops, fails or is
interrupted leaves the machine where it stopped, and its result lists each setpoint it moved
with its value before the call; a run killed outright, at its time limit or otherwise, lists
only the channels it attempted. :ref:`reference-guarded-run` has the lock and what a run
reports.

.. _architecture-safety-chain-shared:

What every path shares
======================

**The connector.** Each of the five paths ends at the connector, the last step before the
machine. On every write it asks the verdict --- may this login write to this machine right now
--- whichever path the write took, a plan already running included. A refusal ends in a
``[osprey-refusal: <reason>]`` trailer, so the reason survives being passed along --- through
a plan that failed, through a run record, into the audit line. The reasons are listed in
:ref:`write-safety-config`. A Python run started with writes held off is refused
``launch_pinned`` and told to run again: a flip applies to the next run, not to one in
flight.

Where limits checking is on, the connector then checks the value against the limits --- the
allowed range, the step size, and whether the channel may be written at all. It sends the value
and, where that channel's policy asks for it, waits for the machine to confirm, reads the
channel back and compares what it now holds with what was sent. See
:doc:`/reference/contracts/connectors`.

**The audit trail.** Every refusal and every approval prompt on every path, and every tool
call that was allowed, is written down as one ``audit`` entry in the agent record, by the
check, the server or the Python executor that made the decision; the
connector writes none itself, the path that called it does. Each line records the write
posture in force and who acted, or which kernel. See :ref:`reference-audit-trail`.

.. _architecture-safety-chain-code-review:

Code review
===========

Everything above checks code by its spelling: a client library's own put, the name of the
store that holds the switch, a pattern on a list. None of it can tell whether a write makes
sense --- whether it belongs to the task the agent was given, whether the step is reasonable
for the magnet it moves, whether a loop will ever end. Code review closes that gap: a second
AI agent reads the code and judges what it would do to the machine. (The approval preview is
something else: there the operator reads the code at the prompt; here a machine does the
reading.)

A deployment turns it on in its profile (:doc:`/how-to/control-systems/code-review`). From
then on the reviewer reads two things: a Python run the OSPREY agent sends to the executor,
before the operator is asked to approve it, and a code cell the agent writes into a
notebook, before it is saved. Nothing else is reviewed: channel writes, plans and
pyAML tools carry no code of the agent's, and cells a person writes are the person's own.

The reviewer follows instructions the deployment writes itself, in a Markdown file in the
project (``data/code_review/reviewer.md``). OSPREY fills in the general part --- find every
write, however it is spelled; check which channels it reaches, what value it sends, how much
changes at once, and what the machine is left with if the code stops halfway. The facility
adds its own rules: which channels must never be written, when writes need extra care, which
procedures a write has to follow. The reviewer has no tools and takes one turn, and anything
the code says to it is treated as evidence, never as an instruction.

Its answer is *pass*, *concern* or *block*, with the reasons and, where a small change would
fix it, a corrected version. All of that goes back to the OSPREY agent, which revises its code
and tries again. A run described as "read the orbit" that also moves a corrector is a block; a
sweep whose step size comes from a script argument is at least a concern. The deployment
chooses what a block does. In blocking mode (``level: block``, the default) the code is
stopped. In advisory mode (``level: flag``) the code runs and the agent is told what the
reviewer found. A concern never stops code; it only comes with an explanation.

Three things are worth knowing. First, the review happens *before* the operator's approval.
In blocking mode, code the reviewer stops never reaches the operator; otherwise the answer and
the reasons stand on the approval prompt and in the approval preview, so the operator decides
knowing what the reviewer found. A run no prompt sees --- a policy that asks nobody, a
read-only run under a selective policy --- is reviewed by the executor just before it starts.
Second, the operator sees a *code review* line in the activity log whenever the reviewer found
something --- what was reviewed, the answer, and whether the code ran; the reasons go to the
agent, and on a run they are on the prompt too; the audit trail keeps a record that a review
happened and how it ended (:ref:`reference-audit-trail`). Third, if the reviewer cannot
answer --- the model is unreachable, or the time limit runs out --- blocking mode stops the
code and the agent is told to report the problem rather than retry, while advisory mode lets
the code run unreviewed and says so. The reviewer is never a safety boundary: every gate on
this page holds whatever it answers.

.. raw:: html
   :file: ../_diagrams/code-review.html

.. _architecture-safety-chain-phoebus-drive:

A Phoebus drive
===============

With the ``phoebus`` server enabled and ``phoebus.agent_access: read_write`` set (both off
unless a deployment switches them on), ``phoebus_drive`` clicks a control or types a value into
a widget of a display open in a running Phoebus product. Under the default, ``read``, the agent
is not offered the tool and the server refuses a call to it. The call passes the write verdict
and the approval prompt, like the other tool paths, and is refused while the control target is
switched away from the deployment's baseline, because the Phoebus product stays connected to
the baseline machine. The switch is judged once, when the call passes the verdict.
``phoebus.agent_access`` is all or nothing, because a panel runs whatever its widgets are wired
to; finer control belongs to the EPICS gateway or access security the Phoebus product connects
through.

Below the checks, OSPREY does not write the value. The Phoebus agent bridge performs the drive
inside the Phoebus process, and the product writes through its own channel connections in both
modes: synthetic, the default, fires the widget's GUI event so the display's own confirm
dialogs, enable rules and scripts run; semantic writes a typed value straight to the widget's
channel. So none of :ref:`what every path shares <architecture-safety-chain-shared>` reaches
the value: the limits check does not see it, neither at the ``osprey_limits`` hook nor at a
connector; no connector asks the verdict at write time; and nothing reads the channel back.

The audit trail records the call as it records any tool call: one entry naming
``phoebus_drive``, who called it, the posture in force and the decision. The tool call's
own entry in the agent record carries the widget, verb, mode and value the call was given
and the bridge's answer. Neither names the channel written, because the bridge decides
which channel a widget writes. See
:doc:`/how-to/control-systems/phoebus-bridge` for the configuration and for the displays a
drive is fit for.

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

The panel's own address is neither of those doors: a login there mints no name at all. See
:ref:`service door, not a terminal <web-terminal-service-door>` for what that costs the person
using it. In the single-user containerised shape, where one sidecar is reached with the
deployment-wide secret, requests name nobody by design --- that secret says a request is
allowed, not who sent it.

A job the OSPREY agent fires goes out through the terminal's own panel proxy. The agent never
holds the dispatcher's key; the proxy holds it, adds it on the way out and stamps the same name
the rest of that terminal's traffic carries. The worker that runs the job hands that name on to
the job's own agent session, so the job runs as the person who asked and its writes follow
that person's tier and switch. A scheduled job runs for nobody and writes nothing. A backend behind the EVENTS panel is handed that key and that name
only when the deployment's config declares it **and** it answers on the loopback address ---
the same rule the operator secret follows. See :doc:`/how-to/agent-interfaces/event-dispatch`.

What the boundary does not cover
--------------------------------

The queue belongs to the facility, and OSPREY gates its own doors to it, not every door. Three
gaps follow from that, and a deployment should know them:

- **A plan added from inside a container.** Something already running there --- a Python run,
  or a tier whose profile allows a shell --- can reach the queue's own port and add a plan
  under any name it chooses, or none. The gap is that such a name is unvouched: OSPREY did not
  mint it, and it binds all the same, so the plan is judged against whoever it names. With no
  name it writes nothing.
- **A shell reaching the panel's dispatcher address.** The panel token sits in the agent's own
  environment, so the same shell can fire a job without the approval prompt. Building a tier
  that allows a shell warns about exactly this, and a sandboxed run cannot do it --- neither the
  token nor the port is there. The job still runs as the real person, under their switch.
- **A direct login at the panel's own address.** Unnamed, as above, so it writes nothing.

An external-worker lane (``bluesky.external:``) is a facility's own RunEngine and devices: no
OSPREY check runs inside it, so no tier or switch reaches its writes. The name OSPREY puts on
such an item is attribution --- it says who queued the plan, not what gated it.

The raw-put block that sends Python runs through the connector has gaps of the same kind. It
refuses a client library's own put, and it tells a connector's put apart by the connector's
write door, which records who is making the put rather than guarding it. Code that goes around
the client library altogether is not stopped: a ``libca`` handle loaded through ``ctypes``, a
raw socket, a user-installed pyepics' callback under ``PREEMPTIVE_CALLBACK=False`` run inside a
connector's waiting put, or a connector subclass written in the run itself.

The store check matches the store's path as the code spells it, case-sensitively: code that
assembles the path at run time is not caught.

A notebook kernel's block refuses the client library itself: client imports and Python's two
loaders (``ctypes`` and ``cffi``) are refused. What remains is the OS loader itself (an extension
module that links a client library, loaded from a copied package), a fresh interpreter with no
block at all (a shell escape such as ``!python -c …``, or ``subprocess``), an edited
``sys.meta_path``, and a raw socket.

These are **cooperative bypasses**: the block keeps honest code on the checked path, and it is
not a sandbox. See :ref:`python-executor-armed-block`, :ref:`connector-write-door` and
:ref:`what a cell may load <notebooks-write-channel>`. The next layer is **network posture**:
only connector-host children should reach a gateway. The web tier runs ``network_mode: host``
today, so a kernel, and anything a cell starts, can reach any gateway a connector-host child
can.

**The approval prompt attributes a write to a person; it does not contain code already running
in the container.** The prompt is answered at the terminal, and the terminal's server runs in
the same container, as the same user, as the agent and everything the agent runs. What keeps
that code from answering in the operator's name is that the server's credentials are out of
its reach: they leave the environment before any agent is spawned, and the server closes its
own memory to other processes of its user, the measure an SSH agent takes for the keys it holds;
the dispatch worker does the same. What the prompt cannot do is stand in for the checks above
against code that has its own way to a write --- the cooperative bypasses, or the container's
network reach to the control system. Those are the connector's checks and the deployment's
network to close. The mechanical boundaries of record are the connector's mediation and the
containers themselves; the prompt is the record that a person intended the write.

.. _architecture-safety-chain-residual:

What the switch does not close
------------------------------

**The same-uid residual.** The switch's record is written by the terminal's server, and the
agent, its Python runs and its notebook kernels run as the same Unix user. OSPREY closes what
it can reach: the agent's file tools are denied the record's store, a shell is denied on every
``asked`` or ``allowed`` tier (lifting it fails the build), Python code that names the store
needs approval and is refused where nothing can ask, and every writer other than the person's own route
leaves the switch as it found it. Code in the executor running as that user, assembling the
path at run time, can still reach it. A writer the agent cannot reach --- the record in its
own user or container --- is the structural close, and is left to the sandboxing work (SBX).

**Routes left to later work.** These writes are refused today and have no route back yet:

- **An ownerless write** --- a queue item or request no login can be named for. The gate on
  the external lane (BYP) is where it gets a route back.
- **A scheduled dispatch job.** It runs for no login, so no switch covers it. The route back is
  a schedule that names the login it follows; that is a dispatch feature.
- **A stop across logins.** *Turn all writes off* stops the pressing login's writes only. A
  deployment-wide stop needs one writer that can reach every login's record (SBX).

The name the control system sees
--------------------------------

The name above is OSPREY's. The control system reads a different one: a Channel Access or PV
Access client tells the server the account name of the process making the put, and a gateway's
put-log and an IOC's access-security rules see that name. Inside a container every process runs
as uid 1000, so by default every write arrives as ``osprey``.

A personal card behind a login wall can carry a ``control_identity`` in the roster, and its
container then gives uid 1000 that name at start (:ref:`multi-user-control-identity`). The
shared writers carry fixed service names instead: dispatch worker *i* writes as
``osprey-dispatch-<i>``, and each Bluesky lane OSPREY renders writes as ``osprey-bluesky`` or
``osprey-bluesky-<lane>``. An external-worker lane writes under whatever account the facility
runs it as. The audit trail records the same name beside the person who asked, so a put-log
line joins back to them; see :ref:`audit-trail-attribution`.

**The name is attribution, not authentication.** The client asserts it and the server takes it
on trust. OSPREY authenticates the person at the login wall, and the name repeats that answer
in a place the control system can read. It is process-wide, so it covers every write route in
the card --- the connector, the Python executor, notebook kernels and raw client libraries ---
but it is not a sandbox: a process running as uid 1000 can still reach the control system
around the blocks above --- through ``ctypes`` in an executor run, the OS loader, a fresh
interpreter or a raw socket --- and what gates a write is the chain above, not the name.

.. note::

   **A control identity inherits the grants of the account it names.** Where a gateway or IOC
   access-security file (ACF) grants access by user name, a card whose identity is ``alice``
   gets what ``alice`` gets when writing from a host those rules admit. That is the intent: a
   person writes from their card with the same standing as from their own login. Review the
   ACF rules before assigning identities. The service names also change what a site's rules see
   from the shared writers: dispatch workers write as ``osprey-dispatch-<i>`` rather than
   ``osprey``, and rendered Bluesky lanes as ``osprey-bluesky*`` rather than ``root``. An ACF
   or put-log rule keyed on ``osprey`` or ``root`` needs updating to match. Roll the images and
   the rendered project together; ``osprey health`` reports a card or worker writing under a
   different name than the render expects.

Where the switch records live
-----------------------------

Each login's record sits in a directory of its own under the deployment's agent-data root, and
the lane and dispatch containers read that tree without being able to write to it. The tree is
shared by group, so a switch is readable by the accounts that share it; it is state, not a
secret. The build refuses to set the tree up on a planted symlink, and the readers do not
follow links, so nothing on the path can be swapped for a record from somewhere else. That
guarantee rests on one thing the operator owns: the agent-data root itself must not be writable
by the shared group. A deployment that moves the agent-data root onto shared storage has to
keep the root owned by the account that deploys. The arrangement uses POSIX file ownership and
runs on Linux and macOS hosts.
