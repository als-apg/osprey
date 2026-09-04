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
and one import line. Any existing ``.ipynb`` suppresses it, and it is never
rewritten once written, so it is yours to edit or delete.

Which session a kernel follows
------------------------------

A kernel has no control-system identity of its own. It follows a terminal
session, and the rule is **the session most recently attached** in the
browser. Open a chat session first, then start the kernel.

At start the kernel stamps itself with that session's control target, the
target's generation, and the write posture in force. A cell can read those
back:

.. code-block:: python

   import os

   os.environ["OSPREY_CONTROL_TARGET"]    # the target this kernel is pinned to
   os.environ["OSPREY_LAUNCH_POSTURE"]    # what it was allowed to write at launch

A running kernel keeps the target it started with and can never write more
than it could at launch. Turning writes off takes effect on its next write.
Switch the control target from the chip, or turn writes on, and the kernel
does not follow — restart it. (The chip's own "nothing restarts" wording
covers the agent's ``execute()`` runs, which are a fresh child process every
time. A kernel is not.) The first write after the change says which case you
are in:

.. code-block:: text

   The session's control target changed. Restart the kernel to follow it.
   This kernel started with writes off. Turn writes on from the chip, then restart the kernel.

Closing a session tile detaches the tile. It does not end the session, so the
kernel keeps following that session — and can still write whatever it could
write at launch — until the session itself ends. *+ New* in the tile ends the
current session and starts another, and the running kernel's next write is
refused:

.. code-block:: text

   The chat session this kernel followed has ended. Restart the kernel.

A kernel started with no chat session open carries no target at all. Reads
answer from the deployment's baseline target, and every write is refused:

.. code-block:: text

   No chat session was open when this kernel started. Open one, then restart the kernel.

You meet that state in two places: before a terminal's first session exists,
and after the session a kernel followed has ended.

A refusal for any other reason — a write ceiling, a limits violation — carries
no extra line, because no restart would change it.

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
