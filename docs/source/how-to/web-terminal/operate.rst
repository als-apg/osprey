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
depends on: *if the agent writes now, which machine does it land on, and will
it be refused?* It reads like this::

   ● STAND-IN · writes ▾

The first word is the machine this session stands on --- ``LIVE`` for the
facility's own machine, ``STAND-IN`` for a rehearsal machine that is operated
like one, ``VIRTUAL`` for the virtual accelerator, ``SIMULATED`` for anything
else simulated. The second is the write state **on that machine**:

- **writes** --- the agent may write there, under the deployment's usual
  approval and write-safety rules. This is the baseline.
- **read-only** --- the deployment does not arm writes on that target, or the
  whole deployment is running read-only. Nothing in the browser lifts this.
- **sandbox** --- *you* narrowed that target for this session. Reads are
  untouched: the agent keeps its full view of the control system and of the
  project, and can still run analysis, plots and readonly Python.

Click the chip and a popover opens on **every** control target the deployment
configures, not only the one you are standing on.

One row per machine
~~~~~~~~~~~~~~~~~~~

Each row names a machine and carries the two things you can do to it:

- **What it is** --- the name the deployment gave it, which already says what
  kind of machine it names (*LIVE MACHINE*, *LIVE MACHINE (stand-in)*,
  *virtual accelerator (simulation)*). That label comes from what the
  connector actually is, so a stand-in never renders as the facility's own
  machine. The row you are on is tagged ``current``.
- **Whether it is answering** --- ``connected``, ``unreachable`` or ``unknown``,
  with how long ago that was measured. ``unknown`` means nothing has vouched for
  it yet, not that it is down.
- **writes | read-only** --- the posture *this session* holds on that target, as
  a two-segment toggle. It is the readout as well as the control: where it
  cannot move it still shows which state holds, with the reason beside it.
- **Switch** --- moves this session onto that machine. Where a switch is not
  available the button is replaced by a short phrase for the reason ---
  ``not configured``, ``needs gateway ack`` --- so the gap is explained rather
  than merely empty, and the server's full sentence sits on the tooltip. On a
  fresh deployment the live machine reading ``not configured`` is the normal
  state, not a fault: authoring its gateways is the go-live edit.

The foot of the popover has **Sandbox everything**, which narrows every target
that can be narrowed in one gesture, and the sentence that bounds the whole
surface: nothing here changes the deployment's config.

Narrow a target, or widen it
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The posture is **per target**. Narrowing the live machine leaves a session
working on the virtual accelerator alone, which is the point: you can put the
machine you are worried about out of reach without giving up the one you are
working on.

Narrowing applies as you click it --- taking reach away needs no ceremony.
Widening a target back to writes asks you to confirm first, every time and with
nothing remembered between clicks, and the confirmation names the machine and
the endpoint the agent would then be able to write to.

**Nothing is restarted.** Every gate reads the posture at the moment of the
write, so a narrowing lands on the conversation that is already running: the
agent obeys it on its very next write, and the turn in flight is not
interrupted. The one lag worth knowing about is a narrowing on the target the
session is *on* --- it reaches the agent when the connector is rebuilt, which
waits for a running execution to finish, and the row says so rather than
leaving a toggle that appears to have done nothing.

**It narrows, and never widens.** Only narrowings are recorded, so the chip can
tighten what the deployment permits and never loosen it. Where the toggle
cannot move it is locked and carries the reason:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Reason
     - What it means
   * - ``persona ceiling``
     - This deployment does not arm writes on that target at all. Only a
       rebuild changes that, not a click.
   * - ``readonly run``
     - The whole deployment is running read-only
       (``OSPREY_EXECUTION_MODE=readonly``), which sits above any one session.
   * - ``not enforceable``
     - This session's control-system server is not one this web server can
       address, so a narrowing recorded here would be read by nobody. The
       roster still renders --- it is worth reading --- but the toggles govern
       nothing.
   * - ``store unavailable``
     - The deployment's agent-data root does not resolve, so there is nowhere to
       record a posture the agent would read back. Nothing was changed.
   * - ``no read-only endpoint``
     - Narrowing this target would select a gateway role the deployment has not
       configured, leaving the target unusable. You are told before you act,
       not after.

One more refusal arrives on the gesture rather than on the toggle: you cannot
turn writes back on while the agent is still running something. The run keeps
the posture it started with, so wait for it to finish, or stop it, and try
again.

Switch this session to another machine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Switch** on a row moves the session onto that machine, after a confirmation
that names it and says which write posture the session arrives in there ---
posture is per target, and it does not travel with you. The web server does
not perform the switch: it records the request, and the control-system server
that owns the connector picks it up, re-checks eligibility and reachability at
that moment, and publishes the outcome back. The row then reads ``✓ switched``,
or ``✗`` with the phrase for the refusal --- the same refusal, by the same
code, the agent is given. While a
request is out the chip reads ``switching…``, and one request is outstanding at
a time. If nothing answers within 30 seconds the row reads ``request_expired``:
the request was written, and nothing that could act on it was alive.

What the switch itself is gated on --- the approval prompt, the limits posture,
the archive --- is :doc:`../control-systems/switch-control-target`.

Simple and Expert
~~~~~~~~~~~~~~~~~

The popover follows the session's density setting (the Simple/Expert control in
the display menu; see :doc:`theming`). Simple mode shows the machine's name,
whether it is answering, the toggle and **Switch**. Expert mode adds
the endpoint, the gateway role, the age of the last probe, and the notes under a
locked toggle. The controls and the confirmations are the same in both: the
density changes what is on screen, never what a click does.

Where the posture lives
~~~~~~~~~~~~~~~~~~~~~~~

The posture belongs to **one session**, not to the deployment. Nothing is
written to ``config.yml``. Two people working in two sessions of the same
deployment hold their own postures, and one of them narrowing theirs does not
touch the other.

Narrowings are recorded in
``var/agent_data/control_target/session-postures.json``, written as soon as you
click and read back when the server starts, so restarting the container never
quietly returns a narrowed session to writes.

What refuses a write, and how firmly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Nothing about the posture is a single choke point --- each write route is
refused by the layer that owns it. The difference between those layers is worth
knowing, because one of them is a belt rather than the buckle:

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
     - **Enforced.** The run is refused outright; a ``readonly`` ``execute`` is
       unaffected and runs normally.
   * - Any other write tool the writes-check hook covers --- for example the
       Bluesky queue's arming tools, where that server is enabled
     - The writes-check hook
     - **Best-effort.** It is the first hook in the chain and the only
       posture-aware layer those tools have; a hook that fails to run does not
       refuse.

Each layer says which gate refused, in its own words, so the message never
sends you to the wrong control:

- The hook --- *"SANDBOX POSTURE --- this session refuses control-system writes
  to the <target> target. Switch it back to writes on the control-target chip in
  the header; config.yml is not the gate here."*
- The connector --- *"Write to '<channel>' blocked: this session's posture for
  the '<target>' control target is read-only --- set from the control-target
  chip in the header, and in force for this session only. Set '<target>' back to
  writes from the chip if the write is intended; config.yml is not the gate
  here."*
- The executor --- *"This session's write posture for the '<target>' control
  target is read-only --- set from the control-target chip in the header, and in
  force for this session only."* --- offering a re-run as ``readonly``, and
  saying to set that target back to writes from the chip if the write is
  intended.

Those three are what a narrowing you made from the chip sounds like, and the
chip is where you lift it. A **deployment-wide read-only run** is a different
story and says so, because no click lifts that one:

- The connector --- *"Write to '<channel>' blocked: this deployment is running
  in readonly execution mode (OSPREY_EXECUTION_MODE=readonly), which refuses
  control-system writes for every session. The control-target chip in the
  header cannot lift it."*
- The executor --- *"This deployment is running in readonly execution mode,
  which refuses control-system writes regardless of what the run asks for."* ---
  offering the same re-run as ``readonly``, and saying that writes need the
  deployment started without the variable.

The chip shows the same thing: every toggle locked, with ``readonly run`` as the
reason.

No posture refusal mentions the deployment's ``writes_enabled`` keys,
deliberately: changing one would not lift a posture refusal, and a message that
pointed at one would send an operator to rebuild a deployment when a single
click was the remedy. The reverse holds too --- a write refused because this
target is not armed says so in its own words, names the key that would arm it,
and says nothing about postures.

.. note::

   **The other two surfaces.** Simple mode's chat and the operator websocket run
   their agent through the Agent SDK rather than a terminal. Both read the same
   record at write time, so a narrowing reaches them exactly as it reaches a
   terminal session. Where they differ is in what the chip can do for them.

   A **chat session's toggles work** --- its writes meet the same ceiling and the
   same narrowing a terminal's do --- but it is offered no **Switch**: a chat has
   no control-system server of its own for a switch request to be addressed to,
   and every row says ``chat_session`` where the button would be. One thing is
   worth knowing before relying on a chat's posture: the chat page mints a new
   chat id every time it loads, so a chat's posture lasts as long as that page
   does rather than following the conversation.

   An **operator websocket session cannot be addressed from the chip.** Its id is
   minted when the connection is accepted and names nothing afterwards, so no
   narrowing can ever be recorded against it and there is nothing to restore
   across a restart. That stays true until an operator client exists to define
   its reconnect protocol. Every audit record such a session emits is labelled
   ``posture_source=spawn``, which is the trail's way of saying the posture was
   fixed when the session started rather than read from a live setting --- see
   the record fields in :ref:`the audit trail contract <audit-trail-record>`.

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
