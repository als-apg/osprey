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

A cell carries no target at all in two cases: the deployment's control-context
record is missing or unreadable, or it names a machine this deployment cannot
build --- a target whose connector block was never rendered, or was removed
under it. Reads then answer from the deployment's baseline target and every
write is refused. The first case clears itself as soon as the web terminal has
written the record; the second is a configuration gap and stays until somebody
fixes it.

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

When the tab is grey
--------------------

A grey JUPYTER entry means the sidecar did not start. The terminal log says
why, with the last lines of the sidecar's own error output. The rest of the
terminal is unaffected — only that one tab is unavailable, and it stays grey
until the terminal is restarted.

A sidecar that dies after it started shows itself differently. JupyterLab
reads *Disconnected* and saves fail, and there is nowhere else to save to, so
copy any unsaved cells out of the browser before you restart the terminal. The
terminal log carries one ``notebook sidecar exited`` line with the sidecar's
last error lines. The tab greys on the next page load.

``osprey health`` does not probe the panel. It reports one row per enabled
sidecar reading *not probed — served inside the web terminal*, so it never
claims a panel is healthy on evidence it does not have.

Not in this release
-------------------

- The agent cannot run cells. It edits notebook files; you run them.
- No real-time collaborative editing. Two people in one notebook fall back to
  the save-time dialog above.
- No health probe of the sidecar, only the skip row.
- No live theme following. The tab starts in the pinned theme and stays there
  until you pick another one inside it.
- No per-panel restart. Restart the terminal.

.. seealso::

   :doc:`panels`
      The other tabs, and how the panel proxy treats credentials.

   :doc:`operate`
      Running the terminal that hosts them.
