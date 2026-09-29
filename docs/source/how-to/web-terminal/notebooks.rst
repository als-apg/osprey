Notebooks
=========

The **JUPYTER** tab is a JupyterLab served from inside the Web Terminal. Its
kernels import ``osprey.runtime``, so a cell reads and writes the control
system through the same connector the agent's own Python uses, under the same
write gates and with the same refusal text. Notebooks are ordinary files on a
durable path, so they outlive the container, and the OSPREY agent can edit
them alongside you.

Turning the panel on
--------------------

The panel is a built-in with the id ``jupyter``. Every persona built from the
``control-assistant`` preset family already selects it, so a standard build
has the tab. The ``ariel-standalone``, ``channel-finder-standalone`` and
``hello-world`` presets do not select it.

Name it under ``web_panels`` to add it to a build profile of your own:

.. code-block:: yaml

   web_panels:
     - jupyter         # JUPYTER tab

In a hand-written ``config.yml`` it is enabled the way its peers are:

.. code-block:: yaml

   web:
     panels:
       jupyter: true

.. note::

   JupyterLab, ``jupyter-server`` and ``ipykernel`` are core OSPREY
   dependencies, so an image grows by roughly 50 MB whether or not the panel
   is selected. Selecting it costs nothing beyond that. Leaving it out starts
   no process and opens no port.

Where notebooks live
--------------------

Notebooks live in ``notebooks/`` under the deployment's agent-data root —
``var/agent_data/notebooks/`` in a container build. That directory is on the
durable volume, so notebooks survive ``osprey down && osprey up``. Kernels do
not: stopping the deployment stops every kernel.

JupyterLab opens on that directory and cannot see above it. A path that
resolves outside it is refused with a ``404``, symlinks included — the check
is against the resolved path, not the text of it. *Download* and *Copy
Download Link* in the file browser obey the same rule: they serve files from
``notebooks/`` and nothing else.

*Delete* in the file browser removes the notebook from the volume. There is no
trash to bring it back from.

When the terminal starts the sidecar and ``notebooks/`` holds no notebook at
all, it writes ``getting-started.ipynb``, a two-cell notebook holding one note
and one code cell. The note reads *Cells read and write through the
deployment's current control target and write posture.* Any existing ``.ipynb``
suppresses it, and it is never rewritten once written, so it is yours to edit
or delete.

The code cell reads a channel where the deployment already names one it can
serve, so running it returns a value rather than only importing two names.
The channel comes from the deployment's own configuration, in this order: the
``archiver_freshness`` health check's channel, which a facility declares is
still moving, then a control target's ``probe_channel``. A deployment that
names neither — the ``hello-world`` preset runs a mock connector and declares
no channel — gets the import line alone, because a cell naming a channel
nothing serves would fail on its first run.

Which machine a kernel writes to
--------------------------------

A kernel has no control-system identity of its own, and it follows no chat
session. Cells read and write through the deployment's current control target
and write posture — the same one the chip in the header shows and the agent
uses.

The kernel re-reads that before **every cell**, so a switch made anywhere, by
anyone, reaches the next cell you run. A cell can read back what it was routed
with:

.. code-block:: python

   import os

   os.environ["OSPREY_CONTROL_TARGET"]             # the machine this cell writes to
   os.environ["OSPREY_CONTROL_TARGET_GENERATION"]  # which switch it belongs to
   os.environ["OSPREY_LAUNCH_POSTURE"]             # what this cell may write

**Nothing restarts.** Switch the control target from the chip, or turn writes
on or off, and the next cell you run is routed by the new state.

Inside the web terminal the header chip is the only place to do that; the
JUPYTER tab carries no chip of its own. A notebook opened in its own window —
popped out of the terminal, or opened at its own address — has no header above
it, so that page shows the same chip in its top-right corner. It is the same
switch either way: a change made from either place lands deployment-wide.

A cell already running is not re-routed --- but taking writes away still
reaches it. The two directions are not symmetric, and the difference is worth
knowing:

- **Turning writes off lands at once.** The connector re-reads the write
  posture on every write, so the running cell's very next write is refused.
- **Turning writes on waits for the next cell.** A cell can never write more
  than it was allowed when it started, so widening cannot reach it.
- **Switching the machine waits for the next cell** as well, so a switch never
  lands halfway through your work.

Re-run the cell to pick up either of the last two, and the refusal says which
one you are in:

.. code-block:: text

   The control target changed while this cell ran. Re-run the cell.
   Writes are off for this cell. Turn writes on from the chip, then re-run the cell.

A switch takes a moment to reach the machines, and a cell run in that window is
routed nowhere rather than to a guess:

.. code-block:: text

   switch_in_progress:5150. A control-target switch is in progress on pid 5150; re-run the cell when the chip settles.

It works the other way too. From a cell's first control-system call until that
cell ends, a switch asked for anywhere in the deployment is refused and names
the kernel holding the target, so whoever asked knows to interrupt it rather
than wait on nothing. A cell that touches no channel holds nothing.

The record a cell reads is the one belonging to whoever the kernel's terminal
is acting as, and a kernel finds it without being told. The build points each
terminal's container at its own record directory through
``OSPREY_CONTROL_CONTEXT_DIR``, and a kernel reads that; where nothing names
one, it falls back to the agent-data root and the name it is acting as. Where
its own audit records are filed is a separate question, answered by that name
rather than by a directory: a kernel starts with most of the environment around
it stripped away, and ``OSPREY_AUDIT_IDENTITY`` --- the name the deployment
gives the container it runs in --- is what survives the stripping. In a
multi-user deployment both answers are the user whose terminal the kernel
belongs to, so a cell reads that user's own chip settings and nobody else's.

Each channel a cell writes leaves one ``allowed`` record in
``notebook_kernel.jsonl``, filed after the put. Its reason says how the write
ended: ``write_landed`` when the write was verified, ``write_unconfirmed`` when
the value was sent but not verified. The record's ``detail`` names the channel
and the account and host the control system saw the write come from, which is
what joins it to a gateway's put-log. A refused write reaches no channel and
leaves only its refusal record. See :ref:`audit-trail-attribution`.

A cell carries no target at all in two cases: the deployment's control-context
record is missing or unreadable, or it names a machine this deployment cannot
build --- a target whose connector block was never rendered, or was removed
under it. Reads then answer from the deployment's baseline target and every
write is refused. The first case clears itself as soon as the web terminal has
written the record; the second is a configuration gap and stays until somebody
fixes it.

.. _notebooks-write-channel:

A cell writes through ``osprey.runtime``. A client library's own put ---
``epics.caput``, ``PV.put``, a caproto ``PV.write``, a Tango
``write_attribute``, an ophyd-async signal set directly --- goes around the
connector, and with it around the write posture, the limits check and the
audit record, so the kernel refuses it whatever the chip says. It raises
``ChannelWriteBlockedError`` with reason ``RAW_CLIENT_WRITE``, nothing is
sent, and the cell prints one line above the traceback:

.. code-block:: text

   Direct client-library writes are refused. Write through osprey.runtime.write_channel(address, value) instead.

Turning writes on does not change that answer. Replace the put with the
runtime's call, which takes the same address and value:

.. code-block:: python

   # Refused:
   #   from epics import caput
   #   caput("DEMO:CORR1:SP", 1.5)

   from osprey.runtime import write_channel, write_channels

   write_channel("DEMO:CORR1:SP", 1.5)
   write_channels({"DEMO:CORR1:SP": 1.5, "DEMO:CORR2:SP": -0.4})

Reads through a client library are unchanged. So are p4p's ``rpc`` and Tango
commands, which carry no channel value and stay allowed in a kernel, and the
PVAccess puts (a p4p ``Context.put``, a pvaPy ``Channel.put``): the connector
does not write PVAccess yet, so ``write_channel`` has no route to a PVAccess
channel and a raw put is how one is written. A kernel does not limits-check
it. A
``RunEngine`` driving ophyd or ophyd-async devices inside a cell is refused
the same way, because its devices end in a raw put; submit the plan to a
Bluesky lane queue instead, where it runs unchanged. The refusal is filed in
``notebook_kernel.jsonl`` with reason ``raw_client_write``. The full list of
refused entry points is under :ref:`python-executor-armed-block`.

A refusal for any other reason — a write ceiling, a limits violation — carries
no extra line, because nothing you do in the notebook would change it.

A page reload keeps the kernel running, but the notebook comes back as it was
last saved. Output from cells that ran since that save is gone.

Notebooks the agent also edits
------------------------------

The OSPREY agent may edit a notebook under ``notebooks/`` (and, as before,
under ``artifacts/``); an edit anywhere else is refused by the memory guard.
When it edits one, the rail's JUPYTER entry badges and the history row reads
*agent edited <notebook>*.

What you see next depends on what the notebook was doing at the time:

- **Not open** — it opens with the agent's change already in it.
- **Open with no unsaved edits of yours** — pick *File → Reload Notebook from
  Disk* to take the change.
- **Open with unsaved edits of yours** — JupyterLab notices the file moved
  under it and offers *Overwrite* or *Revert* when you save. Your work is
  never silently replaced.

Notebooks in a multi-user deployment
------------------------------------

The multi-user stack gives every user their own container and their own
volumes, which a redeploy never touches (:doc:`multi-user/index`), so
notebooks are per user. There is no shared notebook folder in this release.
Sharing a notebook means handing over the file.

.. _notebooks-theming:

How the tab is themed
---------------------

JupyterLab starts in the deployment's pinned theme. A ``web.theme`` that names
a dark or a light look (:doc:`theming`) starts the tab in JupyterLab's matching
built-in theme; a family name on its own pins nothing.

Unlike the other panels, this one does not follow the terminal's Appearance
toggle. Switching the terminal between light and dark leaves JupyterLab as it
is. Pick a theme inside the tab from *Settings → Theme* instead. That pick is
stored on the durable volume, so it comes back after a sidecar restart.

When the tab fails to start
---------------------------

A dimmed JUPYTER entry whose tooltip reads *JUPYTER failed to start:
<reason>* means the sidecar did not start. On a host where the first start is
slow, the terminal gives up after 60 s. Set ``web.sidecar_ready_timeout_s``
(:ref:`config-web`) to give the sidecar longer. The reason is one line: what
went wrong, and the sidecar's last error line when it has one. The terminal log
has the full error output. The rest of the terminal is unaffected — only that
one tab is unavailable.

Click the entry to start the sidecar again. The terminal runs one attempt at a
time and waits as long as it does at startup; the tooltip reads *JUPYTER is
starting* until the attempt settles. A start that succeeds opens the tab; one
that fails again shows its new reason.

A sidecar that dies after it started turns the entry the same way within about
ten seconds. JupyterLab reads *Disconnected* first and saves fail, and there is
nowhere else to save to, so copy any unsaved cells out of the browser before you
click the entry to start it again.

``osprey health`` never fetches the panel. It shows the same *failed to start*
sentence as a warning row when the terminal recorded a failure, and a skip row
otherwise, so it never claims a panel is healthy on evidence it does not have.

Not in this release
-------------------

- The agent cannot run cells. It edits notebook files; you run them.
- No real-time collaborative editing. Two people in one notebook fall back to
  the save-time dialog above.
- No health probe of the sidecar; the row reports what the terminal recorded.
- No live theme following. The tab starts in the pinned theme and stays there
  until you pick another one inside it.

.. seealso::

   :doc:`panels`
      The other tabs, and how the panel proxy treats credentials.

   :doc:`operate`
      Running the terminal that hosts them.
