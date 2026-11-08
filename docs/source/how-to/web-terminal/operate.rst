Run the Web Terminal
====================

Start the Web Terminal from any OSPREY project directory:

.. code-block:: bash

   osprey web

It boots a local server on ``http://127.0.0.1:10100`` and prints a login URL of
the form ``http://127.0.0.1:10100/?token=…``. Opening that URL (your browser
opens it for you) signs you in and then redirects to the clean
``http://127.0.0.1:10100`` address; every later request rides the cookie it
sets. That cookie is good for 12 hours by default and outlives closing the
browser, so on a console other people sit at, shorten it — set
``modules.web_terminals.auth.session_lifetime`` in the deployment's config
(see :doc:`multi-user/login`). The URL is printed once, but the token in it is
the server's own secret and keeps working for as long as that server runs —
treat it like a password rather than a one-shot code.

If ``OSPREY_TERMINAL_SECRET`` is already set in the environment you launch
from, ``osprey web`` uses that value rather than minting one, and says so
instead of printing a URL — the URL was printed wherever that secret came from.
Unset it and start again to get a freshly minted one.

Override the defaults, or point it at another project, with ``--host``,
``--port``, and ``--repo``.

To keep it running after you close the terminal, start it in the background:

.. code-block:: bash

   osprey web --detach     # start in the background
   osprey web stop         # stop it again

In background mode the process id and logs are written to
``var/osprey-web.pid`` and ``var/osprey-web.log`` in the project directory. The
login token lives only in the running process's memory and is never written to
disk, so if you lose the printed URL there is no way to recover it: stop the
server and start it again (``osprey web stop`` then ``osprey web --detach``) to
mint a fresh one. Browsers already signed in stay signed in across that
restart — their sessions live in a store on disk; ``osprey web sessions clear``
(with the server stopped) forgets them.

What you get
------------

The window has a sidebar on the left and two panes:

- **Sidebar** (left) — the list of everything you can open:
  the :ref:`control-target block <web-terminal-session-posture>` (which
  machine this deployment writes to, and whether it may), search, the
  **Sessions** row, the **Apps** and **System** sections, and a footer with **Docs**,
  **Feedback** (see :doc:`send-feedback`), **Sign out** (multi-user
  deployments only) and **☰ More**. See
  :ref:`web-terminal-sidebar`.
- **Agent** (right) — the operator chat, or a real terminal running the OSPREY
  agent when you open one from the pane's session switcher (the session title
  at the left of the Agent bar). It survives reconnects, and you can keep a
  few background conversations alive and hop between them.
  Close the pane with its **×**; the **Sessions** row in the sidebar opens it again.
- **Work pane** (middle) — the artifacts and panels you open from the sidebar,
  in tiles you arrange by dragging (`Tiles`_). New artifacts,
  plots, and data files appear under their session in the sidebar
  (`Sessions and artifacts`_) as the agent creates them, with no refresh.
  On a fresh demo deployment the tree is not empty: its first
  deploy lays down three past sessions from the ``demo-history`` scenario,
  with their transcripts and artifacts, which you can open and resume (see
  :ref:`how-to-run-scenarios`). Your
  control-system tools (Channel Finder, the Logbook, the knowledge panel, and so
  on) open here too. See :doc:`panels`.

The settings drawer (**☰ More** › **Settings**) lets you read and edit the
project's ``config.yml`` — and
the agent's own setup and memory files — from the browser, so you rarely need
to drop back to an editor. Changes prompt you to restart the terminal so the
agent picks them up.

Copying text works the way it does in a desktop terminal: drag over the
agent's output and the selection is already on your clipboard — no key to
press. To grab raw screen text instead (say, while the agent is busy), hold
Option (macOS) or Shift while dragging, then copy with Cmd+C or Ctrl+Shift+C;
plain Ctrl+C always interrupts the agent. Serve the terminal over HTTPS for
this to work in every browser — on a plain ``http://`` page copying falls
back to an older browser mechanism that Safari may refuse.

.. _web-terminal-tiles:

Tiles
-----

The work pane is tiles, and a tile shows one panel. Its header shows the
panel's name, up to three of the panel's icons, **⋯**, **Open in new window**
and **×**.

**⋯**, or a right-click on the header, opens the tile's menu. **This panel**
lists the panel's own actions and any icons a narrow tile has no room for.
**Tile** lists the tile's actions, among them **Reload** and, when the panel
has a documentation page, **Help**.

A panel you open — from its sidebar row, from a link, from the **+** menu,
or one the agent opens for you — takes the service tile you used last and
replaces the panel it showed. Nothing resizes,
and the panel it replaced stays in the sidebar, one click away. A panel
already open is brought forward where it is. To open a panel in a tile of
its own, pick **Open in a new tile** from its row's menu, or drag the row
(below).

Drag a whole tile (by its header) or a sidebar row, and drop it on a tile's
edge or on the work pane's edge:

- **Edge**, the outer band of a tile on each side: the tile splits in two, the
  dropped item on that side. No other tile changes size.
- **Dock edge**, the outermost strip of the work pane: a new tile along that
  whole side.

The highlight during the drag is where the item lands. A drop on a tile's
header or centre does nothing. The agent pane is always its own tile: a drop
onto it splits it, and a panel never replaces it.

**×** closes the tile; its space goes to the tile it was split from. The panel
stays in the sidebar, and its row opens it again. The arrangement — tiles, the
panel each shows, sizes — is remembered by the browser you set it in. A layout
saved by an earlier release with several panels in one tile comes back
showing the panel that was in front; the others are in the sidebar.

.. _web-terminal-one-session:

The chat and the terminal
-------------------------

The Agent pane is the operator chat. To reach the agent's terminal, click the
session title at the left of the Agent bar and pick **Open terminal**, the last
row; **Back to chat** returns.
The choice belongs to the browser tab, and a new tab opens on the chat. The
chat and the terminal are two windows onto one conversation: they share one
session, one write state and one transcript, and only the one you are looking
at runs the agent. A deployment can leave the terminal out; see
:ref:`config-web-terminal`.

Handing the agent over waits for the work in flight. Switch while the agent is
mid-turn and the side you arrive in reads *Finishing the running turn* with
the time you have been waiting, and offers **Stop and switch now**. Let it
finish and the conversation reappears where it left off, with everything said
so far already in the log. Nothing is ended on a clock: only the turn
finishing, closing the page, or that button ends it. The button interrupts the
agent, and stops it outright if it has not yielded within five seconds.

Switch while the other side's agent is idle and there is nothing to wait for:
the side you arrive in reads *Restarting the agent…*, with no
clock and no stop button. A restart that runs past about four seconds turns
into the wait above.

A refusal can meet you instead of the wait. *This session is in use in
another tab.* means another browser tab holds it, so close that tab or
work there instead. *The previous agent is still shutting down.* is the one
worth retrying, and the card offers **Retry**.

.. note::

   **Deployments built before this release.** The wait needs the
   ``turn-state`` hook, which tells the web terminal when a turn starts and
   ends, and a deployment scaffolded earlier does not have it: add
   ``turn-state`` to the profile's ``hooks:`` list and run ``osprey build``.
   Without it, switching away from a terminal that might be working never
   waits. It says *The agent may still be working.* and offers only
   **Stop and switch now**, and a terminal that was idle all along can still
   take the full five seconds to hand over.

The chat and the terminal run the same Claude Code version, the one
``claude_code.cli_version`` names. If the chat is refused because that version
is not installed, the server log says which one to install; see
:doc:`/getting-started/installation`.

While a turn runs in the chat, **Send** becomes **Stop**. A message entered
meanwhile waits above the composer and is sent when the turn ends; **Interrupt
and ask now** ends the turn and sends it at once.

Clearing a conversation
~~~~~~~~~~~~~~~~~~~~~~~

Run ``/clear`` in the terminal to start a new conversation under the same
session. The write state and the machine you are standing on carry over, and
the chat shows the cleared conversation too.

Beneath the composer, a context meter shows how full the context window is.
From 80 percent it suggests ``/compact`` or a new conversation.

Attaching a file
~~~~~~~~~~~~~~~~

Attach a file to a message with the paperclip button. You can also paste an
image, or drop a file on the composer. The file is uploaded when you attach
it. The chip shows the file name and its size. To remove a chip, select its
**x** button. Type your question, then send. The chat does not send a file by
itself. The agent opens the file with its Read tool.

The chat accepts images (PNG, JPEG, GIF, WebP), PDF files, and plain-text files
(TXT, LOG, CSV, JSON, YAML, Markdown, Python). A file can be 20 MB at most. A
message can carry 10 files. One chat can hold 200 MB of files. A file of
another type, or a file that is too large, is refused. The chip shows the
reason.

The agent reads an attached file from the uploads directory under the agent
data directory. The deployment allows the agent to read that directory, so the
read raises no approval card. A file outside it can still raise one. A file
that you attach but never send is deleted after 24 hours.

.. _web-terminal-session-posture:

The control-target chip
-----------------------

The sidebar carries a block that answers, at a glance, the question every write
depends on: *if the agent writes now, which machine does it land on, and may
it?* It is one row: an icon with the machine's dot, the machine's name, the
write state, a dot for the health of the deployment's services, and an arrow.
It reads like this::

   ⌖ Rehearsal  writes asked  •  ⇕

Put the pointer on the block to see the same facts as one sentence, for
example *Control system: Rehearsal, writes asked*. Put the pointer on the small
health dot to see the health of the services.

The first part names the machine this deployment stands on, by what it **is**:

- **Live** --- the facility's own. Writes move hardware.
- **Rehearsal** --- a copy of the real machine's controls, same channel names,
  no hardware behind it. Nothing moves.
- **Simulator** --- the virtual accelerator: a physics model with beam in it,
  served from its container or in process. Nothing moves.

A deployment can put its own names on its machines (*Example Research
Facility storage ring* rather than *Live*); what the machine is stays
behind the small ⓘ beside its name either way, and that tooltip also keeps the
controls server's own technical label.

The second part is **your login's** write switch for that machine, and what
the next write would get:

- **writes asked** --- every machine write asks you first, in the approval card.
- **writes allowed** --- the same writes run without a prompt.
- **writes off** --- writes are refused. Reads are untouched: the agent keeps
  its full view of the control system and of the project, and can still run
  analysis, plots and read-only Python.
- **writes locked** --- your switch is armed, but something the chip cannot lift
  refuses the write. The popover shows the phrase for it and who lifts it.
- **· program** after any of these --- for example **writes off · program**:
  a program's write authorization you approved is in force on this machine
  (see :ref:`web-terminal-approve-program`). The word before it is still your
  own switch; the program may write the channels you approved, inside their
  ranges, until it ends, whatever your switch says. Its dot is a ring around a
  centre rather than a solid dot. The machine's *off* control revokes it, and
  the program's next write is refused.

A switch is one login's. Two logins on the same machine each have their own,
and one login's position never changes the other's writes.

Click the chip and a popover opens next to the sidebar on **every** control
target the deployment configures, not only the one you are standing on. A
click outside the popover, Escape, or a change of the window or sidebar width
closes it. When a confirm dialog is open over the popover, the first Escape
closes only the dialog.

The machine you are on, and the others
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The machine the agent stands on is a card at the top: its name and its writes
switch. Every other machine is a row below the card with the same shape and
the actions. What writing to a machine means (*Writes move hardware*, or one
of the *nothing moves* lines) sits with its endpoint and the server's own
label behind the small ⓘ beside each name, on hover or keyboard focus:

- **The writes switch** --- per machine, for every session of this login:
  *off*, *asked* or *allowed*. *Off* is always available. A position your tier
  or a lock forbids is disabled, with the reason on hover.
- **Switch to** --- moves the deployment onto that machine. Where a switch is
  not available, the button's place is taken by a short phrase for the reason
  --- ``not set up``, ``needs gateway ack`` --- with the server's full
  sentence on the tooltip. On a fresh deployment the real machine reading
  ``not set up`` is the normal state, not a fault: authoring its gateways is
  the go-live edit.

A machine that stops answering says so --- ``not answering``, in red, next to
its name. A machine that answers says nothing about it. Endpoints, gateway
roles and the age of the last probe stay on the tooltips and in the
confirmation dialogs, where the decision is actually made.

The foot has **Turn all writes off**: every switch of your login goes off in
one click, a machine no longer configured included, and every program
authorization you approved is revoked. It stops your login's writes only;
other logins keep theirs. It is enabled while any of your switches is armed or
a program authorization of yours is in force.

.. _web-terminal-approve-program:

Approve a program's writes
~~~~~~~~~~~~~~~~~~~~~~~~~~

A program that writes with nobody at a prompt, such as an optimizer run, asks
for a **write authorization**: these channels, inside these ranges, this many
writes, until this time. A request that names your login appears in the
popover under **Programs**, with **Review**. The card shows:

- every channel with the range asked for, the limits-database bound it sits
  inside, and its current value;
- the number of writes (steps), what the run does at the end (nothing, apply
  the best point found, or restore the values it started from), the expiry, the
  lease the program renews while it runs, the program (holder) and the run's
  label;
- that a write the limits refuse still costs its step.

While the current values are read the request line says *reading values…*; a
machine that does not answer within a few seconds leaves them *not read*.

**Approve** issues it. It is refused, with the reason on the card, when your
tier is ``never`` or cannot be read, this terminal runs read-only, a range
is outside the limits database, a channel has no numeric range there or a
current value that is not a number, or another program already holds one of
the channels on this machine. Approving is the confirmation for every write
the run makes inside it, even when your tier is *asked*: the program's writes
are not prompted one by one. Once approved, the machine's word reads
**· program** after it, and the authorization is listed under Programs with
its steps used of its total, when it ends (the time left; hover for the exact
time) and **Revoke**.

A program that issued its own authorization while your switch stood at
*allowed* is listed there too, marked self-issued, and an approval refuses to
overlap it. It has no **Revoke**: moving the machine's switch off *allowed*
ends it.

An authorization ends when its steps are used, at its expiry, when the
program stops renewing its lease, or at a stop. To stop it, use **Revoke**, the
machine's *off* control, or **Turn all writes off**: the program's next write
is refused. **Revoke** on one that had already ended says *Already stopped*.
One whose steps are used or whose lease lapsed leaves the chip at once.
Nothing renews an ended authorization; the program asks again. The same approval
is available without a web terminal through
:ref:`osprey write-authorization <cli-osprey-write-authorization>`.

Turn writes off, or on
~~~~~~~~~~~~~~~~~~~~~~

The switch is **per machine**. Turn the real machine off and work on the
simulator carries on, which is the point: you put the machine you are worried
about out of reach without giving up the one you are working on.

**Narrowing** (to *asked*, or to *off*) applies as you click, and nothing
refuses it --- no lock, no read-only run, no second terminal. If a stop cannot
be recorded you are told plainly that it did not take effect; the chip never
shows an off that was not recorded.

**Widening** asks first. The confirmation names the machine and says whether
writes will still prompt. Tick *Don't ask again on any machine* to skip it for
every move to *asked* or *allowed*, the live machine included; Shift-click or
*Ask again* in the popover brings it back. If a stop lands while you are
confirming, the move does not take effect: the chip re-reads and you click again.

**Nothing is restarted.** Every gate reads the switch at the moment of the
write. A narrowing reaches every next write, runs already going included: the
agent's next tool call, a running notebook cell's or Python run's next write.
A widening reaches the agent's tool writes at their next call; a notebook cell
or Python run already going keeps the position it started with, and the next
one follows (:doc:`notebooks`). The confirmation says so.

**What the chip cannot lift.** A write is refused for one reason, the first
that holds, and locks no click lifts come before the switch. The chip shows
that reason's phrase on the row, and the agent is refused in the same words.
Each phrase ends by naming the machine --- *(Concerns the rehearsal
machine.)*:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Phrase
     - Who lifts it
   * - *the deployment is running read-only*
     - The deployment admin. The chip cannot.
   * - *this login's write tier does not allow them*, *this login's write tier
       is not known*
     - The deployment admin.
   * - *this scheduled or triggered job runs for no login*
     - The deployment admin. No switch covers it.
   * - *this machine's writes belong to another login*
     - Only that login.
   * - *the machine this connector writes to is not known*, *the write switch
       cannot be read*, *the write gateway has not yet followed the switch*
     - Retry, then the deployment admin.
   * - *this run was started with writes held off*
     - Run it again without the pin.
   * - *Writes are off*, *a stop came after the turn-on began*
     - You: move the switch to asked or allowed.

A position past your login's tier (:doc:`multi-user/tiers`) acts as the tier
until the tier is restored.

Switch the deployment to another machine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Switch to** on a row moves the whole deployment onto that machine, after a
confirmation that says where every control read and write goes next and
your switch's position there --- the switch is per machine, and it does not
travel with you. This dialog too can be skipped with its own *Don't ask
again on any machine*. The page that holds the control context runs the same
checks the agent's own switch runs --- that the move is allowed, and that
nothing is mid-flight on the machine you are leaving --- and records the
outcome; the deployment's control-system servers then move themselves onto it
and report when they have arrived. The row reads ``switching…`` until every
one of them has, and then ``✓ switched``, or ``✗`` with the phrase for the
refusal --- the same refusal, for the same reason, the agent is given. A
server that could not make the move says so by name, so you know which one to
look at. If nothing answers within 30 seconds the row reads
``request_expired``: nothing that could carry out the switch was alive to pick
it up. The outcome line leaves the row after about a minute --- an outcome is
news for as long as someone is watching for it.

Ask for a second switch while one is still landing and it is refused rather
than queued, naming the servers still working. Wait for the chip to settle and
ask again.

**Switch to** is the one control a second web terminal of the same login does
not hold: only the process holding the control context moves the deployment,
and elsewhere the button's place names the holder. The writes switches work
from either.

What the switch itself is gated on --- the approval prompt, the limits
posture, the archive --- is :doc:`../control-systems/switch-control-target`.

Where the control target and the write state live
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both belong to **the login the terminal acts as**, not to any one session,
and nothing is written to ``config.yml``. Every session, window and notebook
kernel of that login sees the same machine and the same switches: turning
writes off on a machine takes them away from all of them, and a switch made in
one window applies in the other. That is the point --- a switch you can
only see from the page you happen to be on is not a safety control.

Both are recorded together on disk, written as soon as you click and read back
when a server starts, so restarting the container never quietly widens a
switch or moves the deployment off the machine somebody put it on. Coming back
to the deployment baseline is a switch like any other.

That record sits in a directory named for whoever the terminal is acting as:
``var/agent_data/control_target/<name>/control_context.json``. On a multi-user
deployment every user has their own container and their own volumes
(:doc:`multi-user/index`), so "the deployment" here means the stack that user
is working in, and ``<name>`` is that user --- alice's settings are in
``control_target/alice/`` on the host, and no user's chip decides another
user's writes.

What refuses a write, and how firmly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each write route is refused by the layer that owns it. One of them is
best-effort rather than enforced:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - What the agent tries
     - What refuses it
     - What that means
   * - ``channel_write`` --- any control-system write
     - The connector, before the control system is asked
     - **Enforced.** The writes-check hook denies it first as well, but the
       connector is the layer that holds if the hook does not run.
   * - ``execute`` with ``execution_mode="readwrite"``
     - The Python executor's own gate
     - **Enforced.** The run is refused outright; a ``readonly`` ``execute``
       is unaffected and runs normally.
   * - Any other write tool the writes-check hook covers --- for example the
       Bluesky queue's arming tools, where that server is enabled
     - The writes-check hook
     - **Best-effort.** It is the first hook in the chain and the only
       switch-aware layer those tools have; a hook that fails to run does not
       refuse.

Every layer speaks the phrase from the table above, so a refusal never sends
you to the wrong control. A switch you turned off sounds like *"Writes are
off; turn it on from the chip in the web terminal. (Concerns the real
machine.)"* --- and the chip is where you move it. A read-only run says the
chip cannot lift it, and a refusal nobody set --- a switch that cannot be read
--- says to retry, then ask the deployment admin.

.. note::

   **The other surfaces.** The operator chat and websocket run
   their agent through the Agent SDK rather than a terminal, and notebook
   kernels run no agent at all. All of them read your login's switches at
   write time, so a switch you set and a move you make reach every one of
   them --- because what the chip changes is your login's record rather than
   anything belonging to the window you are looking at.

   A notebook kernel is the one surface where *when* is worth knowing. Off
   refuses the running cell's next write; a widening, or a move to another
   machine, reaches the next cell you run. See :doc:`notebooks`.

   .. _web-terminal-service-door:

   Where a deployment also serves a panel at an address of its own, that
   address is a **service door, not a terminal**. Log in there directly and the
   deployment has no name to put on what you do, so no switch covers it: a
   plan queued from that page writes nothing (``owner_unknown``). Reaching the
   same panel from its place in the terminal is the named path, and the one the
   chip governs.

.. _web-terminal-sidebar:

The sidebar
-----------

The sidebar is on the left. From top to bottom, it shows:

- The OSPREY mark and the label toggle.
- The control-target block. It is one row that shows the machine that this
  deployment writes to, if the agent can write to it, and the health of the
  services. Click the block to open the
  :ref:`control-target popover <web-terminal-session-posture>`.
- **Search**. It opens the command palette. The shortcut is ⌘K on macOS and
  Ctrl+K on other systems.
- The groups of panel rows. At first there are three:

  - **Sessions**, the first group, has no heading. It starts with this row:

    - **Sessions** opens the agent pane and your session tree below it
      (`Sessions and artifacts`_); with the pane open, it folds or unfolds
      the tree. Its **+** starts a new session. When the pane is closed, the
      row is dimmed. When the agent waits for your answer, the row shows the
      number of open questions. It is always the first row of the sidebar.

  - **Apps** starts with all other panels that this deployment enables.
  - **System** starts with **Events** (only when the deployment enables the
    events panel) and **Health**.

  In each group, the rows are in your order. **+ Add panel** is always the
  last row of **Apps**. See :doc:`panels`.
- The footer: **Docs**, **Feedback**, **Sign out** and **☰ More**. **Sign
  out** shows only on a multi-user deployment. See
  :ref:`web-terminal-sign-out`.

When the window is short, only the group that holds the session tree
scrolls. The other groups and the footer always stay on the screen.

.. _web-terminal-sessions:

Sessions and artifacts
~~~~~~~~~~~~~~~~~~~~~~

The **Sessions** row lists your sessions, newest first. A session is named
from your first prompt ("naming…" until the name arrives); ``/clear`` starts a
new one. Under a session, its artifacts are grouped by day; **Group by Kind**
sorts them into **Runs**, **Plots**, **Trends**, **Tables**, **Documents**,
**Dashboards**, **Images**, **Data**, plus any kind the deployment registers.
A script run is one row with a dot for **succeeded**, **failed** or **not
run** (none while it runs); its approval preview, code and outputs are its
parts. A run that was never run
opens on its proposed code. **More sessions** loads older ones.

- Click a session name to switch to it: its layout comes back as you left it
  (panels, tiles, sizes, the Agent pane, and each panel's page where the hub
  can read it), with the artifact last open there (its newest if none). A new
  or older session keeps the layout on screen. The sidebar stays as it is.
  Click the current session's name to fold or unfold it. Only the current
  session shows its artifacts; the others stay folded until you switch to them.
  ⌥-click (Alt+click) a session or **Sessions** to expand every session, or
  collapse them all if all are open. The session the agent works in has a
  breathing accent bar.
- Hover a folded session to see its artifacts; click one to show it.
- Click an artifact to show it. This never changes the session.
- The title in the agent tile's header has its own ▾ switcher: your sessions
  with state, age and message count, and **+ New session**.
- While the agent is answering in chat, a switch from any of these asks
  first: **Cancel**, or **Stop & switch**.

Right-click a session to pin it, rename it, export its transcript or group
its artifacts by day or by kind, or delete a past session with its artifacts;
a run to open its result, approval preview or code. Runs and artifacts also
have **Delete…**.

Pinned sessions (the pin button on a row) also show in the **Pinned** group
above **Sessions**; right-click its heading to rename or hide it. **Show pinned
sessions**, on the **Sessions** menu or on empty sidebar space, brings it back.

A reload, a new tab or another device resumes your last session and its
layout. Its last-open artifact is shown in the Artifacts panel if that panel
is docked, otherwise revealed in the tree. The tree highlights only the
artifact the Artifacts panel shows. The command
palette finds your sessions and artifacts (by name, kind, or session name);
opening a result enters its session. The Artifacts panel has no sidebar row;
open it from the palette.

.. _web-terminal-sidebar-groups:

Arrange the groups
~~~~~~~~~~~~~~~~~~

You can rename the groups, add your own groups, and move rows between them.

To change a group, right-click its heading (or focus its name and press
Shift+F10). The menu has these items:

- **Rename**. Type the new name and press Enter. Press Escape to keep the old
  name. A name can have up to 40 characters. An empty name keeps the old one.
- **New group below**. This adds an empty group below this group and opens
  its name for editing.
- **Move up** and **Move down**. These move the group in the sidebar.
- **Delete group**. This item shows only for a group that you added. The
  rows of the group go back to **Apps**. You cannot delete **Apps** or
  **System**, but you can rename them.

Double-click a group's name to fold it down to its heading, and double-click
again to unfold it. **Sessions**, the first group, has no
heading, so it has no menu and always stays first. To make it taller, drag the
faint line below it; double-click the line to go back to automatic height.

To move a row to a different group, do one of these:

- Right-click the row and pick **Move to** and the name of the group. Pick
  **Move to new group…** to put the row in a new group below its current
  group. Then type the name of the new group.
- Drag the row into the other group. Release it where it must go.

You can move each row except **Sessions**. A group without rows shows "No
panels — move one here from a row's menu".

Your groups are kept in this browser, for your user only. They do not change
the sidebar of other users.

Hide the labels
~~~~~~~~~~~~~~~

To hide the labels, click the toggle at the top of the sidebar, or press ⌘\\
(Ctrl+\\ on other systems). The sidebar then shows only icons. Put the pointer
on an icon to see its name. In this state, a click on **Sessions** opens the
session tree next to the sidebar. To show the labels again, click the toggle
or press the same keys.

.. note::

   When a terminal has the focus, Ctrl+\\ goes to the terminal, not
   to the sidebar. In the terminal, Ctrl+\\ sends SIGQUIT to the running
   program. To hide or show the labels in this state, click the toggle.

You can also press Tab, also after a click inside a panel. Tab does not hide
or show the labels when the cursor is in a text field, in the message box, in
the terminal, or in an open menu or dialog. In those places, Tab does its
usual work. Shift+Tab moves the focus back, as usual. To turn the Tab key off,
open **More** and clear **Tab shows or hides the sidebar**.

.. _web-terminal-pop-out:

Open a pane in a new window
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To move a pane into a window of its own, click **Open in new window** in its
tile's header, or pick it from the **Tile** section of its **⋯** menu or from
a right-click on its sidebar row. The command palette has
**Open <panel> in new window**. This works for the Agent pane and for each
service panel. The tile leaves the workspace.

The window's bar is the tile's own bar, with **Reattach** at the top right.
The pane's sidebar row shows the reattach icon: a click on the row brings the
window to the front, a click on the icon reattaches the pane. Reattach puts
the pane back where it was. With no room for it, the pane stays in its window
and the terminal shows *No room to reattach <panel>.*

Closing the window closes the pane. Reloading the window (Cmd+R) keeps the
pane in it. After you reload the terminal, the windows are closed and their
panes are back in place. The terminal's keyboard shortcuts do not work in the
window.

.. note::

   If the browser blocks pop-ups for this site, the terminal shows
   *Pop-ups are blocked for this site.* and the tile stays. Allow pop-ups for
   the site in the browser, then try again.

   A panel that runs at an opaque origin (a page the agent wrote) cannot open
   in a new window. Its **Open in new window** button and row are disabled.

The More menu
~~~~~~~~~~~~~

**☰ More** shows your user name and role at the top. It holds the theme
(see :doc:`theming`), **Settings** and the tour.

.. _web-terminal-sign-out:

Sign out
~~~~~~~~

On a multi-user deployment, the sidebar footer has a **Sign out** row above
**☰ More**. To sign out:

1. Click **Sign out**. You can also click the OSPREY mark at the top of the
   sidebar, or pick **Log out** in the command palette.
2. Read the dialog. It asks *Sign out of OSPREY?* and tells you that your
   terminal and agent sessions end, and that the agent stops what it does.
3. Click **Sign out** to sign out. Click **Cancel** or press Escape to stay.

When you sign out, the browser goes to the landing page of the deployment. A
deployment for one user has no **Sign out** row, and the OSPREY mark goes to
the start page.

Where your preferences live
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The browser remembers the theme, the sidebar
labels, the open lists in the sidebar, the sidebar groups, and the panel
layout. A different browser or a different machine starts from the defaults
of the deployment.

Earlier releases had a header and a status bar that you could arrange. The
terminal does not show them, and it does not read the bar arrangements that
those releases saved.

.. _personal-login:

A user's own Claude subscription
--------------------------------

By default every terminal authenticates with the deployment's provider
(``claude_code.provider``), one gateway and one key. Some sites let a
colleague use their own Claude subscription inside the terminal while
forbidding the deployment to hold that person's credential. The route that
satisfies both is the interactive ``/login`` inside the user's own terminal,
whose credential stays in that terminal's config volume.

.. code-block:: yaml

   config:
     claude_code.allow_personal_login: true   # default false

With the key on, a terminal that holds a completed ``/login`` is launched
with the deployment provider withheld: no auth token, base URL or model
variables and no translation proxy, so Claude Code's own credential
precedence falls through to the subscription. The decision is made per
terminal container at launch, by the presence of the credential file under
that user's ``CLAUDE_CONFIG_DIR``; the file is never read, and a login
survives a container restart because the config volume does. A terminal
without a login routes to the deployment provider exactly as before and
shows a dismissable card saying ``/login`` is on offer. With the key off,
nothing changes.

Whether the ``/login`` flow completes inside a terminal container has not
been tried yet: the first site that enables the key does that trial. In
a headless container the CLI should fall back to its paste-a-code path,
which the terminal can carry.

Two consequences follow. A login-routed session bypasses the gateway, so it
leaves no row in the deployment's spend ledger; that is the point. And
``osprey health`` can no longer treat a failed probe of the agent provider
as the whole deployment down: with the key on it stays a warning, naming the
dispatch worker and the terminals without a login as the ones affected.
The dispatch worker and every other headless path keep the deployment
provider unconditionally.

Documentation and feedback settings
-----------------------------------

Five ``web`` keys aim the sidebar's **Documentation** link and **Feedback** button
— including the issue trackers the dialog offers — and bound the feedback
store. The table, the shipped defaults, and what a blank value means are in
:ref:`config-web`.

.. dropdown:: Under the hood
   :icon: gear

   .. tab-set::

      .. tab-item:: Settings

         Two sections in ``config.yml`` are easy to confuse. ``web_terminal:``
         is the terminal **process** — which shell to launch, which directory to
         watch for live files, how many background conversations to keep alive —
         and command-line flags override those for a single run. ``web:`` is the
         browser **UI** the process renders, including the deployment name and
         the bounds on the operator-chat pool; those keys are catalogued in
         :ref:`config-web`.

         One key sits outside both, in the multi-user ``modules.web_terminals``
         block: ``modules.web_terminals.auth.session_lifetime`` sets how long a
         login cookie stays valid, in whole seconds, and defaults to ``43200``
         (12 hours). It is the only key ``osprey web`` reads from that block.

      .. tab-item:: Companion servers

         The panels are powered by small companion servers OSPREY launches for
         you — an artifact gallery always, and a domain server for each enabled
         panel. You normally never touch them.

      .. tab-item:: For developers

         Every feature above is backed by a REST and WebSocket API. The endpoints
         are discoverable directly in the source
         (``src/osprey/interfaces/web_terminal/``); a coding agent working in the
         codebase can wire against them without a hand-maintained list here.

.. seealso::

   :doc:`theming`
      Choose or design the theme every OSPREY interface uses.

   :doc:`panels`
      Add your own tools as side panels.

   :doc:`send-feedback`
      The feedback dialog these settings configure, and the ``osprey feedback``
      verbs that read the results back.

   :ref:`config-web`
      Every ``web`` key, with its default and what a blank value means.
