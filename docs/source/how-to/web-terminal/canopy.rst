.. _how-to-web-terminal-canopy:

CANOPY
======

CANOPY is the operator GUI: machine screens, Trace and the correlation plot.
In an OSPREY deployment it is a tab of the web terminal, and every terminal
runs its own. This page says what that means for the people who use it and
for the person who runs the deployment.

.. dropdown:: What You'll Learn
   :color: primary
   :icon: book

   - What "one CANOPY per user" means, and what a user sees of it
   - How the tab follows the control-target chip, and where history comes from
   - How writes from a screen are judged, confirmed and audited
   - Where screens live, and how one gets into the shared library
   - What the agent can see and change in the panel, and how to undo it
   - What ``spec_unavailable`` means in this release
   - What the panel costs in memory and image size
   - How to upgrade CANOPY, and how to go back

   **Prerequisites:** a deployment whose profile shows the CANOPY panel, and a
   CANOPY payload for the build (see `The payload`_). Authoring and reviewing
   library screens has its own page, :doc:`../canopy-screens`.

One CANOPY per user
===================

The CANOPY tab is a built-in panel. The web terminal starts it, as two
processes that live and stop with the terminal: a **front** process, which
serves the viewer, Trace and the agent's screen tools, and a **data** process,
which reads and writes PVs. Neither listens anywhere but the loopback
interface, and the only way to them is the terminal's own panel proxy, behind
the terminal's login.

The data process has no EPICS client of its own. It reads through the
deployment's control session, on the connector the terminal's agent and
notebooks use. So a screen and the agent of the same terminal see the same
machine, under the same tier, with the same limits.

In a multi-user deployment "per user" means per login card. Every card on the
roster has a terminal container, and every container whose render shows the
panel runs one CANOPY, from the moment the terminal starts. A
:ref:`shared card <multi-user-shared-card>` is one instance: the people who
open it share its chip, its screens and its saved views, and a write made
through it is audited under the card's name. A wall display in a control room
is best served by one shared card, since ten users on one screen are ten sets
of monitors on the control system.

Who gets the panel and the agent's tools follows the render:

- a profile that lists ``canopy`` in ``web_panels``, and the tiers that inherit
  it, show the panel and give the agent its tools;
- the knowledge, logbook and channel-finder apps inherit nothing from the
  profile, so they carry neither;
- a dispatched job has no terminal, so it has no CANOPY and no CANOPY tools.

The machine badge
-----------------

The viewer shows a badge naming the machine its values come from. It is the
name of the control target the data process was started for. Three other
states can show:

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Badge
     - Meaning
   * - the target's name
     - Values come from that machine.
   * - ``fallback — no control target record``
     - The terminal has no readable record of a chosen target. CANOPY serves
       the deployment's default target, read-only.
   * - ``cannot be served``
     - The chosen target is one CANOPY cannot reach. Every channel shows
       disconnected and every write is refused. No other machine is shown in
       its place.
   * - ``Disconnected``
     - The browser lost its connection to the data process. The viewer
       reconnects by itself.

When the chip changes
---------------------

The :ref:`control-target chip <web-terminal-session-posture>` decides which
machine a terminal is on, and CANOPY follows it. One data process serves one
target for its whole life. When the user picks another target, the data
process finishes any write already sent, exits, and the terminal starts a new
one for the new target. The screens go blank for about 1.5 s, plus up to
0.25 s until the change is noticed, and then show the new machine's values.
Nothing from the old machine stays on screen.

A watchdog hold, or moving the chip's write switch between *asked* and
*allowed*, restarts nothing: the open screens show it at the next heartbeat.
Moving the switch to or from *off* on a machine with separate read and write
gateways replaces the data process, as a target change does: it picks its
gateway when it connects.

A write click made on a screen that still shows the old machine cannot land
on the new one: every write names the machine its screen showed, and the data
process refuses one that is not its own (``target_changed``).

The data process also leaves, and is replaced at once, when the deployment's
EPICS transport goes stale after an outage of the control system. Values come
back without a restart of the terminal.

History follows the machine
---------------------------

Trace and the correlation page read history from the archive of the machine
the user is on. When the deployment's archive serves another target, or
none, the panel says ``No archiver``. Live values and history never come
from two different machines. A user on the simulator sees its store when the
deployment archives the simulator; a user on the live machine or on its
stand-in sees the facility's archiver appliance when the deployment names
one; everyone else sees ``No archiver``.

Trace and the correlation page complete PV names from the channel catalog.
``osprey build`` stages the deployment's channel snapshot once, at
``build/web_terminals/channels.json``, and every web container mounts it
read-only at ``/etc/osprey/channels.json``. A deployment with no channels to
suggest has no such file, and the two pages offer no completion.

Saved views
-----------

Saved views in Trace and the correlation page are kept in the user's own
agent data (``canopy/views``). They are per user, and they start empty: no
view saved in another user's CANOPY, or in a shared ``canopy`` service of an
earlier release, is carried over.

Configure the panel
===================

The panel is declared like any built-in panel, and its settings sit in the
same block. In a profile they are written as flat keys under ``config:``:

.. code-block:: yaml

   config:
     web.panels.canopy.trace_default_query: pv=SR:DIAG:BPM:03:POSITION:X&range=-1h
     web.panels.canopy.correlation_default_query: x=SR:RF:CAVITY:01:TEMPERATURE:RB&y=SR:RF:CAVITY:01:POWER:REV&dt=1
     web.panels.canopy.waveform_default_query: pv=SR:MAG:SEPTUM:01:CURRENT:WAVEFORM
     web.panels.canopy.home_screen: facility/overview
     web.panels.canopy.vole_bytes_max: 268435456
     web.panels.canopy.theme_file: data/canopy/theme/colors.json

The channel names in the queries are the example facility's, the one the
``control-assistant`` preset ships; a deployment uses its own.

``home_screen`` names the screen CANOPY opens first, by its path in the
library without ``.screen.json``. ``osprey build`` and ``osprey validate``
refuse an id the library does not hold and list the screens it does. Without
the key CANOPY opens its first screen.

The panel has no ``url``, ``label`` or ``health_endpoint`` to set: the
terminal starts the server itself. Every key is described in
:ref:`the configuration reference <config-web-panels-canopy>`, together with
what the panel declares to the terminal (its tools for the agent, its entry
pages, its stop grace).

To leave the panel out of a tier, exclude it and its skill by name:

.. code-block:: yaml

   exclude:
     skills:
       - canopy-panel-builder
     web_panels:
       - canopy

The payload
-----------

OSPREY does not build CANOPY. The canopy repository's
``scripts/build-payload.sh`` writes a **payload** directory: the CANOPY and
PyDMConverter wheels, the viewer (whose ``ir-map.json`` is the widget
catalog), Trace, a ``STAGED.json`` naming the commits it was built from, and
a ``spec/`` directory when the build was given one. Name that directory in
the deployment's ``.env``:

.. code-block:: bash

   OSPREY_CANOPY_PAYLOAD=/srv/canopy/payload

A pre-build step in the profile checks the directory and copies it into the
deployment as ``canopy_payload/`` on every ``osprey build``. A missing or
half-built payload stops the build with one line that says what to do.

.. code-block:: yaml

   lifecycle:
     pre_build:
       - name: canopy-payload
         timeout: 30
         run: '"{python}" -m osprey.integrations.canopy.stage'

With ``OSPREY_CANOPY_PAYLOAD`` unset, the build leaves the CANOPY panel and its
skills out of the deployment and says so in one line. Name a payload and
rebuild to add them.

The payload must be built at the CANOPY and PyDMConverter commits this OSPREY
is proven against: its CANOPY tests ran against exactly that pair. A payload
whose ``STAGED.json`` names any other commit stops the build, and nothing is
staged. Read the two commits from the installed OSPREY:

.. code-block:: bash

   python -m osprey.integrations.canopy --pins

Writes from a screen
====================

A write from a CANOPY screen is the user's own OSPREY write. The tier's
ceiling, the chip's posture, a read-only run, the channel limits and the audit
are the ones every other write in the terminal goes through. A refusal is one
sentence in the toast: a limits violation shows OSPREY's reason word for word,
and a write-posture refusal names what refused it — the target is kept
read-only for this login, or its writes are off from the control-target chip,
or its recorded write state could not be read — with OSPREY's full explanation
in the terminal's log. A screen full of write buttons is inert for a read-tier
user.

CANOPY has no writes setting of its own. Each click is judged when it is
made, by the same verdict as every write: tier, session hold, record, switch.

A click then ends in a toast with one of seven outcomes: ``confirmed``,
``mismatch`` (the readback differs), ``unconfirmed`` (no readback to check),
``unknown`` (the reply was lost: the write may have happened), ``refused``,
``target_changed`` or ``switch_in_progress``.

The confirmation dialog
-----------------------

At *asked* on the chip, every run of writes shows a confirmation dialog: the
click and its confirm are the approval, and a write without it is refused
``approval_missing``. At *allowed* there is no dialog. While a watchdog has
flagged any live session of the terminal, the panel says "writes held" with
the watchdog's reason and refuses every write ``session_held``: the agent
opens and arranges CANOPY screens and drafts, so a held agent could stage a
click.

- A slider or a spinner writes nothing while it is dragged and asks once on
  release, for the final value.
- A declined dialog sends no request and leaves no audit record.
- A chip change while the dialog is open cancels it, and nothing is written.
- The dialog governs clicks. The agent never writes a PV through CANOPY's
  tools: its writes stay on OSPREY's own write tool and its approval.
- At *allowed*, a write button on a screen the agent opened sends
  without a dialog, to the target the agent chose. See
  `A screen the agent opened`_.

Every write and every refusal is recorded as an ``audit`` entry in the agent
record, under the user's name, with the surface ``canopy``, the value, and the
screen and its revision. For a shared card the record names the card.

A terminal that is stopped gives CANOPY 34 s to finish a write already sent
before the container is killed.

Screens: the personal store and the library
===========================================

A user's CANOPY reads screens from two places.

The **library** is the directory ``data/canopy/library`` of the deployment
repository. A build copies it into every image, and every CANOPY serves it
read-only, to administrators too, as ``library/<id>``. It changes only by a
commit to the repository and a build.

The **personal store** is a directory in the user's own agent data
(``canopy/drafts``). It is kept across rebuilds, only that user's CANOPY
mounts it, and only that user sees what is in it. A shared card has one
personal store, shared by the people who use the card.

The panel builder
-----------------

The agent in a terminal reaches that terminal's CANOPY and nobody else's. A
build writes a ``canopy`` entry into the project's ``.mcp.json`` and the
``canopy-panel-builder`` and ``canopy-operator`` skills into
``.claude/skills/``. Ten read tools run without a prompt (``check``,
``read_pv``, ``search_pvs``, ``screenshot``, ``get_screen``, ``list_drafts``,
``list_widgets``, ``get_widget``, ``get_schema``, ``list_collections``), and
so do the two workspace tools, ``get_workspace`` and ``set_workspace`` (see
`What the agent sees and changes`_). Seven authoring tools ask on every call (``new_screen``,
``patch_node``, ``add_node``, ``remove_node``, ``convert_legacy``,
``delete_draft``, ``request_push``). The authoring tools write into the
personal store only.

No review applies inside a personal store: what a screen there can do is
bounded by its author's own tier.

Push to source
--------------

A screen leaves a personal store one way. The user asks the agent to share
it, the agent calls ``request_push``, and that files a bundle in the user's
own agent data (``canopy/push``). Nothing is published by it. An
administrator reviews the request on the host with
``python -m osprey.integrations.canopy.push`` (``list``, ``show``, ``accept``,
``reject``), reading every write binding the screen carries, and accepts it by
the digest the review printed. The accepted files land in
``data/canopy/library``, the administrator commits them, and the next build
ships them to everyone. :doc:`../canopy-screens` is the walkthrough, with the
recovery from an interrupted accept and the way to roll a promotion back.

Three things to know before the first push:

- **Keep the deployment under version control.** ``osprey init`` makes the
  deployment a git repository, and the review relies on it: git holds the
  history of the library, and the push tool keeps replaced files for 14 days
  only.
- **Commit the library before you re-initialize.** ``osprey init --force``
  replaces ``data/``, the library included. Accepted screens that were not
  committed are destroyed by it.
- **The review runs with docker.** Podman is refused in this release.

``spec_unavailable``
--------------------

CANOPY checks a screen in three ways: against the schema of the screen
format, against the property schema of each widget, and for what it can do
(what it writes, runs and opens). The two schemas are not part of CANOPY or of
OSPREY. They belong to the CANOPY specification, and they reach a deployment
as the ``spec/`` directory of the payload.

**The payload of this release has no** ``spec/`` **directory**, because the
specification repository is not yet available to the payload build. This is
an interim state, and it is reported rather than hidden:

- ``check`` returns exactly one warning, ``spec_unavailable``, naming what is
  missing, and runs neither the schema check nor the property check. Every
  other check runs unchanged, including the one that lists what a screen
  writes to the machine.
- ``list_widgets``, ``get_widget`` and ``get_schema`` answer, and carry a
  ``spec_unavailable`` note where a schema would be.
- ``push show`` and ``push accept`` print one ``spec_unavailable:`` line, and
  the log entry of an accepted push records it.

A screen is therefore never reported as passing a check that did not run, and
the review of a push is still a person reading every write binding. No copy
of the schemas is bundled anywhere as a stand-in.

The state ends without any change to OSPREY or to the deployment's profile:
build a payload that carries ``spec/``, point ``OSPREY_CANOPY_PAYLOAD`` at
it, and run ``osprey build`` and ``osprey up``. The build notices the changed
payload and stages it again. For the checks to cover the screens of this
release, the specification has to carry version 1.1 of the screen format and
the property schemas for colour roles, for a button's image and transparency,
and for the ``xy-plot`` and ``pv-array-plot`` widgets.

Tiles and tabs in the panel
===========================

.. code-block:: text

   one tile: its tabs in the strip       two or more: a bar per tile
   +-[ RF 1 ][ RF 2 ][ Trace ]---- = +   +-[ RF 1 ][ RF 2 ]--+-[ Trace ]---- = +
   |                                 |   |                   |                 |
   |                                 |   |                   |                 |
   +---------------------------------+   +-------------------+-----------------+

The CANOPY panel docks its screens and tools by the rules of the terminal's
own dock (:ref:`web-terminal-tiles`): a tile holds tabs, and a tab, a whole
tile or a sidebar entry dropped on a tile's centre, tab bar, edge, a border or
the outer band lands the same way, down to the 40 px floor. With one tile its
tabs sit in the strip; with two or more each tile has its own bar and the
strip goes. The window-wide actions (reset layout, save open tabs as a
collection, recently closed, close all, the tool's docs and bug links) are in
the **☰** menu at the right end of the top-right tile's bar, and
:kbd:`Alt+M` (:kbd:`⌥M` on macOS) opens it.

A sidebar click, a link or the agent opening a screen puts it as a new tab of
the focused tile; nothing resizes, and a screen already open is focused where
it is. Closing a tile's last tab removes the tile, and **Recently closed** or
the closed-tabs toast's Undo brings tabs back.

A collection saved with tabs per tile opens as saved. One saved without a
layout opens side by side, one tile each; when they do not all fit in a row
they open as a grid, and when even that does not fit, as tabs of one tile with
a short notice. A collection saved with the layout of an earlier release is
refused with its reason, and nothing changes.

What the agent sees and changes
===============================

The agent can read and rearrange what is open inside the user's own CANOPY
panel: the tabs, the tiles, the zoom of each screen, and the Trace,
Correlation and Waveform tools with their queries. ``get_workspace`` reads it,
every tile with its tabs and the tab in front; ``set_workspace`` sends a batch
of commands to the window the user last focused. The agent makes every drop a
person can (``drop``), opens, focuses, closes (``close_tab``, ``close_tile``),
reopens, zooms or steps back a screen, opens several screens and tools at once
as tabs, side by side or in a layout it gives (``open_set``), opens a
collection the same ways, or opens a tool and changes its query. Neither asks
first, because
neither touches the machine: no command writes a PV, presses a widget, takes a
url or a file, or changes what the user may launch. The
``canopy-operator`` skill tells the agent how to use them.

The ``canopy-screens`` skill tells it which screen to open. The build writes
it from the project's screen library: every screen's ``ref``, what it is for,
the screens it opens, and the devices it shows by label ("SR: ION-PUMP
37–42", "SR: EPU7"), with the tools' default views and the collections. Ask
"show me ION-PUMP 38" and the agent opens the sector that holds it. The skill
follows the library, so a rebuild after a screen change updates it.

The user's own actions win. A change the user makes while a batch is on its
way makes the agent's next command fail, and the agent reads the panel again
before it tries anything else.

Seeing what the agent did
-------------------------

- **Highlight.** The tab or tile a command touched highlights.
- **Activity log.** Each batch is a row in the terminal's activity log
  (**⋯ More**), beside the agent's ``open_panel`` rows. The row lists the
  commands and the tab, screen or tool each one named, and the batch status
  as the tool returned it: ``done``, ``pending`` or ``refused`` with its
  reason. A pending row stays pending: it is not updated when the batch
  finishes later. A call that came back with an error reads ``error``, and
  one whose answer the terminal never saw reads ``unanswered``.

Undo
----

Everything the agent closes can be brought back. Each command that can lose
something (a close, a reset of the tiles, a collection or set opened with a
layout) goes on the same closed-tabs stack the user's own closes use, and
shows the closed-tabs toast with Undo. One toast shows at a time, and the newest wins. Undo puts the
tabs back where they were and keeps whatever the user changed since.

When the toast is gone, **Recently closed** in the **☰** menu lists the
closed groups, each labelled as closed by the agent or by the user, so the
agent's group can be reopened after the user closed something else.
:kbd:`Cmd+Shift+T` reopens the newest group, and the agent can reopen a group
it closed.

The input guard
---------------

A change by the agent can move a write button under the user's pointer. After
any agent command that changed the panel, every write button in that window
refuses for 750 ms, with a visible notice. The time runs from when the change
is drawn, and for a newly opened screen from when that screen has finished
rendering. A write that had already started is not cut off.

While a write dialog or a run of several writes is open, an agent batch waits
for it to close. If it is still open when the batch's wait ends, the batch is
rejected ``operator_busy`` and nothing changes.

A screen the agent opened
-------------------------

The agent chooses which screen to open, with which macros, and may open one
of its own drafts. The write buttons on that screen address what the agent
chose. At *asked*, the dialog shows each write before it is sent.
**At allowed** (see `The confirmation dialog`_), there is no dialog:
a press on a write button of an agent-opened screen sends at once, without
showing its PV, to a target the agent chose. The ``canopy-operator`` skill
has the agent say which screen it opened and with which macros. At
*allowed*, read what an agent-opened screen
addresses before pressing a write button on it.

``open_panel`` and CANOPY urls
------------------------------

The agent's ``open_panel`` puts the CANOPY panel on screen when it is called
without a url. Called with a url for CANOPY, whether ``open_panel('canopy',
url)``, an address under ``/panel/canopy/`` in any spelling, or the address
of the panel's own process, it is refused with ``agent_hop_panel_url`` and
opens nothing; the refusal points the agent at ``set_workspace``. What the
panel shows is changed only through ``set_workspace``. Links a person follows
into CANOPY work as before.

What it costs
=============

Memory
------

One CANOPY takes about 270 MiB (data process about 140, front process about
130), and one runs for every roster user whose render shows the panel,
whether or not that user has the tab open. The live-value buffers of one
instance are capped by ``web.panels.canopy.vole_bytes_max``.

The agent's ``screenshot`` tool renders a screen in a headless browser. The
browser starts on the first screenshot and stops when it is idle, so an
instance that never takes one pays nothing at run time. While it runs it adds
at most 450 MiB on top of the instance.

Image size
----------

The browser adds about 1 GB to an image. Every image carries it, whether or
not its profile shows the panel: the artifact converter needs the same
headless shell, so it is part of the base image layout rather than a
per-panel layer.

The screenshot browser
----------------------

Inside a web container the browser runs unsandboxed
(``CANOPY_BROWSER_SANDBOX=0``), under the container runtime's default seccomp
profile. It is locked to the instance's own address and refuses every request
that is not a read, so it can never send a write. One residual remains and
crosses the user boundary: once a push is accepted, every user's browser
decodes the pushed images, and a pushed image is checked by its leading bytes
only. SVG is refused in pushes and in the library for that reason. Review the
images of a push as carefully as its write bindings.

What the terminal tells CANOPY
------------------------------

The terminal renders the environment of both processes at every start. Four
of the settings explain behaviour described above:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Setting
     - Value under the panel
   * - ``CANOPY_LIBRARY_DIR``
     - The deployment's library, as a second, read-only root. Set only when
       ``data/canopy/library`` exists when the terminal starts.
   * - ``CANOPY_HOME_SCREEN``
     - The screen named by ``web.panels.canopy.home_screen``. Set only when
       the library holds it; otherwise the terminal logs a warning and CANOPY
       opens its first screen.
   * - ``CANOPY_PUSH_DIR``
     - ``canopy/push`` in the user's agent data, where ``request_push`` files
       its bundles.
   * - ``CANOPY_BROWSER_SANDBOX``
     - ``0``: see `The screenshot browser`_.

They are not configuration keys, and a value set in the terminal's
environment or in a ``.env`` does not reach CANOPY: the terminal starts both
processes with an environment of its own making.

.. _canopy-upgrading:

Upgrading CANOPY
================

A CANOPY upgrade is an OSPREY upgrade, a new payload and a rebuild:

1. Upgrade OSPREY to a release proven against the new CANOPY. Its
   ``PROVEN_COMMITS`` names the commits to build (see `The payload`_).
2. Build the new payload in the canopy repository at those commits and point
   ``OSPREY_CANOPY_PAYLOAD`` at it.
3. Regenerate the library's report with the **new** wheels and commit it, even
   when no screen changed
   (``python -m canopy_mcp.report <library> --served-at library --write``; see
   :ref:`the report after an upgrade <canopy-screens-after-upgrade>`). The
   image build compares the committed report with what the CANOPY inside the
   image computes.
4. Run ``osprey build``, then ``osprey up``.

The rebuild replaces the render images, so it restarts every terminal and
ends open agent sessions. Announce it. Personal stores, saved views and
pending push requests are in the users' agent data and are kept.

Going back to the previous payload
----------------------------------

Keep the previous payload directory until the new one has run for a while.
To go back, install the OSPREY release that payload was staged with, point
``OSPREY_CANOPY_PAYLOAD`` at it again, restore the report that was committed
for it (``git revert`` of the report commit of step 3), and
run ``osprey build`` and ``osprey up``. A library screen that was accepted
under the newer CANOPY stays in the library; roll it back separately if the
older CANOPY refuses it (see :doc:`../canopy-screens`).

.. seealso::

   :doc:`../canopy-screens`
      Authoring library screens, reviewing a push request, rolling one back.

   :doc:`panels`
      How panels are declared, and how to add your own.

   :doc:`multi-user/index`
      Rosters, tiers and shared cards.

   :ref:`The configuration reference <config-web-panels-canopy>`
      Every ``web.panels.canopy`` key, and what the panel declares.
