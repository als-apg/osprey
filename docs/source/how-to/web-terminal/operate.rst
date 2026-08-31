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

The window has three working areas plus a header:

- **Terminal** (right) — a real terminal running the Osprey agent. It survives
  reconnects, and you can keep a few background conversations alive and hop
  between them.
- **Workspace** (left) — a live view of your project files. New artifacts,
  plots, and data files appear as the agent creates them, with no refresh.
- **Side panels** — your control-system tools (Channel Finder, ARIEL, the
  lattice dashboard, and so on), opened from the icon rail and arranged as
  dockable tiles. See :doc:`panels`.
- **Utility controls** — pinned to the far end of the same rail, a
  **Documentation** link and a **Feedback** button that lets whoever is at the
  terminal report a problem without leaving it. See
  :doc:`send-feedback`.
- **Header** — the :ref:`control-target chip <web-terminal-session-posture>`
  (which machine this session writes to, and whether it may), the display menu
  (a small dot holding the light/dark, Expert/Simple, and theme controls — see
  :doc:`theming`), a settings drawer, and an optional name badge to tell one
  deployment from another.

The settings drawer lets you read and edit the project's ``config.yml`` — and
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

.. _web-terminal-session-posture:

The control-target chip
-----------------------

The header carries a chip that answers, at a glance, the question every write
depends on: *if the agent writes now, which machine does it land on, and may
it?* It reads like this::

   ● Rehearsal · writes on ▾

The first part names the machine this session stands on, by what it **is**:

- **Real machine** --- the facility's own. Writes move hardware.
- **Rehearsal** --- a copy of the real machine's controls, same channel names,
  no hardware behind it. Nothing moves.
- **Simulator** --- the virtual accelerator: a physics model with beam in it.
  Nothing moves.
- **Demo** --- mock data. Nothing moves.

A deployment can put its own names on its machines
(``control_system.target_display_names`` in ``config.yml`` --- *ALS storage
ring* rather than *Real machine*); what the machine is stays on the popover's
descriptor line and the tooltips either way, and the tooltip also keeps the
controls server's own technical label.

The second part is the write state **on that machine**, for **your session**:

- **writes on** --- the agent may write there, under whatever write gates the
  deployment configures.
- **writes off** --- *you* turned writes off for this session. Reads are
  untouched: the agent keeps its full view of the control system and of the
  project, and can still run analysis, plots and read-only Python. One click
  turns writes back on.
- **writes locked** --- the deployment does not arm writes on that target, or
  the whole deployment is running read-only. Nothing in the browser lifts
  this.

Click the chip and a popover opens on **every** control target the deployment
configures, not only the one you are standing on.

The machine you are on, and the others
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The machine the agent stands on is a card at the top: its name, what writing
there means (*Writes move hardware*, or one of the *nothing moves* lines), the
write state, and the one button that changes it. Every other machine is a row
below the card, carrying the same two facts, its write-state pill, and the
actions:

- **Turn writes off / Turn writes on** --- per machine, for your session.
- **Switch to** --- moves this session onto that machine. Where a switch is
  not available, the button's place is taken by a short phrase for the reason
  --- ``not set up``, ``needs gateway ack`` --- with the server's full
  sentence on the tooltip. On a fresh deployment the real machine reading
  ``not set up`` is the normal state, not a fault: authoring its gateways is
  the go-live edit.

A machine that stops answering says so --- ``not answering``, in red, next to
its name. A machine that answers says nothing about it. Endpoints, gateway
roles and the age of the last probe stay on the tooltips and in the
confirmation dialogs, where the decision is actually made.

The foot has **Turn all writes off**, which takes writes away from every
machine it can in one click, and the popover's scope, said once: *your
session only*.

Take writes away, or give them back
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The write state is **per machine**. Turn writes off on the real machine and
the session keeps working on the simulator, which is the point: you put the
machine you are worried about out of reach without giving up the one you are
working on.

Taking writes away applies as you click --- it needs no ceremony. Turning
writes back on asks you to confirm first, every time and with nothing
remembered between clicks, and the confirmation names the machine and the
endpoint the agent would then be able to write to.

**Nothing is restarted.** Every gate reads the write state at the moment of
the write, so the change lands on the conversation that is already running:
the agent obeys it on its very next write, and the turn in flight is not
interrupted. One lag is worth knowing about: turn writes off on the machine
the session is *on* and the change reaches the agent when the connector is
rebuilt, which waits for a running execution to finish --- and the card says
so rather than leaving a button that appears to have done nothing.

**The chip only takes writes away.** What you set here tightens what the
deployment permits; it can never hand out writes the deployment did not arm.
Where the state is locked the button stays on screen, disabled, and the
reason is on its tooltip:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Reason (on the tooltip)
     - What it means
   * - *kept read-only by the deployment*
     - This deployment does not arm writes on that machine at all. Only a
       rebuild changes that, not a click.
   * - *the whole deployment is running read-only*
     - ``OSPREY_EXECUTION_MODE=readonly`` is set, which sits above any one
       session.
   * - *changes here would not reach the agent*
     - This page has no way to deliver a change to this session's
       control-system server, so a state set here would be read by nobody.
       The roster still renders --- it is worth reading --- but the buttons
       govern nothing, and a banner across the top of the popover says so.
   * - *changes cannot be recorded right now*
     - The folder where write states are recorded is missing or unreadable,
       so there is nowhere to keep a setting the agent would read back.
       Nothing was changed.
   * - *no read-only endpoint configured*
     - Read-only on this machine would route it through a gateway the
       deployment has not configured, leaving the machine unusable. You are
       told before you act, not after.

One more refusal can meet the click itself: you cannot turn writes back on
while the agent is still running something. The run keeps the write state it
started with, so wait for it to finish, or stop it, and try again.

Switch this session to another machine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Switch to** on a row moves the session onto that machine, after a
confirmation that says where every control read and write goes next and
whether writes are on or off for you there --- the write state is per
machine, and it does not travel with you. The browser does not perform the
switch: it records the request, and the part of the deployment that owns the
connection to the machines picks it up, re-checks at that moment that the
move is allowed and the machine answers, and reports the outcome back. The
row then reads ``✓ switched``, or ``✗`` with the phrase for the refusal ---
the same refusal, for the same reason, the agent is given. While a request is
out the chip reads ``switching…``, and one request is outstanding at a time.
If nothing answers within 30 seconds the row reads ``request_expired``:
nothing that could carry out the switch was alive to pick it up.

What the switch itself is gated on --- the approval prompt, the limits
posture, the archive --- is :doc:`../control-systems/switch-control-target`.

Where the write state lives
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The write state belongs to **one session**, not to the deployment. Nothing is
written to ``config.yml``. Two people working in two sessions of the same
deployment hold their own settings, and one of them turning writes off on a
machine does not touch the other.

Your settings are recorded in
``var/agent_data/control_target/session-postures.json``, written as soon as
you click and read back when the server starts, so restarting the container
never quietly turns a session's writes back on.

What refuses a write, and how firmly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Turning writes off is not a single choke point --- each write route is
refused by the layer that owns it. The difference between those layers is
worth knowing, because one of them is best-effort rather than enforced:

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
       state-aware layer those tools have; a hook that fails to run does not
       refuse.

Each layer says which gate refused, in its own words, so the message never
sends you to the wrong control:

- The hook --- *"WRITES OFF --- this session refuses control-system writes
  to the <target> target. Turn writes back on from the control-target chip in
  the header; config.yml is not the gate here."*
- The connector --- *"Write to '<channel>' blocked: writes are off for the
  '<target>' control target in this session --- turned off from the
  control-target chip in the header, and in force for this session only. Turn
  writes back on for '<target>' from the chip if the write is intended;
  config.yml is not the gate here."*
- The executor --- *"Writes are off for the '<target>' control target in this
  session --- turned off from the control-target chip in the header, and in
  force for this session only."* --- offering a re-run as ``readonly``, and
  saying to turn writes back on from the chip if the write is intended.

Those three are what writes you turned off from the chip sound like, and the
chip is where you turn them back on. A **deployment-wide read-only run** is a
different story and says so, because no click lifts that one:

- The connector --- *"Write to '<channel>' blocked: this deployment is running
  in readonly execution mode (OSPREY_EXECUTION_MODE=readonly), which refuses
  control-system writes for every session. The control-target chip in the
  header cannot lift it."*
- The executor --- *"This deployment is running in readonly execution mode,
  which refuses control-system writes regardless of what the run asks for."*
  --- offering the same re-run as ``readonly``, and saying that writes need
  the deployment started without the variable.

The chip shows the same thing: every button locked, with *the whole
deployment is running read-only* as the reason.

No writes-off refusal mentions the deployment's ``writes_enabled`` keys,
deliberately: changing one would not lift it, and a message that pointed at
one would send an operator to rebuild a deployment when a single click was
the remedy. The reverse holds too --- a write refused because this target is
not armed says so in its own words, names the key that would arm it, and
says nothing about the chip.

.. note::

   **The other two surfaces.** Simple mode's chat and the operator websocket run
   their agent through the Agent SDK rather than a terminal. Both read the same
   record at write time, so a write state you set reaches them exactly as it reaches
   a terminal session. Where they differ is in what the chip can do for them.

   A **chat session's write controls work** --- its writes meet the same ceiling
   and the same locks a terminal's do --- but it is offered no **Switch to**: a
   chat has no machine connection of its own to move. Know one thing before
   relying on a chat's write state: the chat page starts a fresh chat every
   time it loads, so a state you set for it lasts as long as that page does
   rather than following the conversation.

   An **operator websocket session's write controls govern nothing**: the chip
   has no way to hand a write state to that kind of session, so what you set
   here never reaches it.

.. dropdown:: Why the websocket session is out of the chip's reach
   :icon: gear

   The websocket session's identifier is created when the connection is
   accepted and identifies nothing once the connection ends, so no write state
   can be recorded against it and there is nothing to restore across a restart.
   That stays true until an operator client exists to define its reconnect
   protocol. Every audit record such a session emits is labelled
   ``posture_source=spawn`` --- the trail's way of saying the write state was
   fixed when the session started rather than read from a live setting; see the
   record fields in :ref:`the audit trail contract <audit-trail-record>`.

Documentation and feedback settings
-----------------------------------

Four ``web`` keys aim the rail's **Documentation** link and **Feedback** button
and bound the feedback store. The table, the shipped defaults, and what a blank
value means are in :ref:`config-web`.

.. dropdown:: Under the hood
   :icon: gear

   .. tab-set::

      .. tab-item:: Settings

         Two sections in ``config.yml`` are easy to confuse. ``web_terminal:``
         is the terminal **process** — which shell to launch, which directory to
         watch for live files, how many background conversations to keep alive —
         and command-line flags override those for a single run. ``web:`` is the
         browser **UI** the process renders, including the header name badge and
         the bounds on the Simple-mode chat pool; those keys are catalogued in
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
