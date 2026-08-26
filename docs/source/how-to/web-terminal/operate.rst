Run the Web Terminal
====================

Start the Web Terminal from any OSPREY project directory:

.. code-block:: bash

   osprey web

It boots a local server on ``http://127.0.0.1:8087`` and prints a login URL of
the form ``http://127.0.0.1:8087/?token=…``. Opening that URL (your browser
opens it for you) sets a session cookie and then redirects to the clean
``http://127.0.0.1:8087`` address; every later request rides that cookie. The
URL is printed once, but the token in it is the server's own secret and keeps
working for as long as that server runs — treat it like a password rather than
a one-shot code.

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
mint a fresh one.

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
  :doc:`/how-to/send-feedback`.
- **Header** — the display menu (a small dot holding the light/dark,
  Expert/Simple, and theme controls — see :doc:`theming`), a settings drawer,
  and an optional name badge to tell one deployment from another.

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

Sandbox one session
-------------------

Every terminal session carries a **posture**, shown as a badge in the terminal
card's header:

- **Writes** --- the session runs whatever the deployment permits, under its
  usual approval and write-verification rules. This is the baseline.
- **Sandbox** --- the session refuses every control-system write, for as long as
  it stays sandboxed. Reads are untouched: the agent keeps its full view of the
  control system and of the project, and can still run analysis, plots and
  readonly Python.

Click the badge to switch. Both directions ask you to confirm, every time and
with nothing remembered between switches, because switching restarts the agent
for that session: the turn it is running right now stops immediately. Your
conversation history is preserved --- the terminal reconnects to the same
session and you pick up where you left off.

The posture belongs to **one session**, not to the deployment. Nothing is
written to ``config.yml``; the session's agent is respawned with the sandbox
marker in its environment, and every process it launches inherits it. Two
people working in two sessions of the same deployment can hold different
postures, and one of them sandboxing theirs does not touch the other.

It narrows, and never widens. On a deployment rendered with
``control_system.writes_enabled`` off, no session can leave the sandbox: the
request is refused with *"This deployment is rendered with
control_system.writes_enabled off; no session can step out of the sandbox"*,
and the badge on a sandboxed session there is shown disabled rather than
offering a switch that is certain to fail.

A terminal session has to have started before it can be given a posture --- it
only exists on disk once it has been sent a prompt. Until then the request is
refused with *"This session has not started yet --- send one prompt first, then
set its posture."* (Chat sessions answer to a different rule; see the note at
the end of this section.)

Postures survive a restart. They are stored in
``var/agent_data/session-postures.json``, written as soon as you switch and
read back when the server starts, so restarting the container never quietly
returns a sandboxed session to writes.

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

- The hook --- *"SANDBOX POSTURE --- this terminal session refuses
  control-system writes. Switch the session to writes posture from the terminal
  card; config.yml is not the gate here."*
- The connector --- *"Write to '<channel>' blocked: this terminal session is in
  the sandbox posture --- readonly execution mode is in force for the whole
  session, not for a single script. Switch the session to the writes posture
  from the terminal card if the write is intended; config.yml is not the gate
  here."*
- The executor --- *"This terminal session is in the sandbox posture, which
  refuses control-system writes regardless of what the run asks for."*

None of them mentions ``control_system.writes_enabled``, deliberately: changing
that key would not lift a posture refusal, and a message that pointed at it
would send an operator to rebuild a deployment when a single click was the
remedy. The reverse holds too --- a write refused because the deployment has
writes off says so in its own words and says nothing about postures.

.. note::

   **The other two surfaces.** Simple mode's chat and the operator websocket
   run their agent through the Agent SDK rather than a terminal, and both apply
   the posture at the moment they start that agent, exactly as the terminal
   does. Where they differ is in whether you can switch one while it runs.

   A **chat session can be switched**, and the flip restarts its agent the same
   way. The badge in the card header addresses terminal sessions; a chat is
   addressed by its chat id, through the same ``POST /api/terminal/posture``
   route. The rule about having sent a prompt first does not apply to it ---
   a chat is addressable from the moment its first prompt starts its agent,
   including while that agent is still starting up, and it stays addressable
   for as long as it is running or has a posture on record, so a chat you
   sandboxed can always be brought back out. One thing is worth knowing before
   relying on this: the chat page mints a new chat id every time it loads, so
   a chat's posture lasts as long as that page does rather than following the
   conversation.

   An **operator websocket session keeps the posture it was started with.**
   There is no way to flip one while it runs, and its posture is not restored
   across a restart --- its id is minted when the connection is accepted and
   names nothing afterwards, so there would be nothing to restore it to. That
   stays true until an operator client exists to define its reconnect protocol.
   Every audit record such a session emits is labelled
   ``posture_source=spawn``, which is the trail's way of saying the posture was
   fixed when the session started rather than read from a live setting --- see
   the record fields on the :ref:`protected set <how-to-protected-set>` page.

Documentation and feedback settings
-----------------------------------

Four ``web`` keys aim the rail's two utility controls and bound the feedback
store:

.. list-table::
   :header-rows: 1
   :widths: 30 34 36

   * - Key
     - Default
     - What it does
   * - ``web.docs_url``
     - ``https://als-apg.github.io/osprey``
     - Where the **Documentation** control — in the rail and in the status bar
       — points. Set it to your own hosted copy of the docs.
   * - ``web.feedback.github_repo``
     - ``als-apg/osprey``
     - ``owner/repo`` the feedback dialog's GitHub channel opens a prefilled
       new issue against.
   * - ``web.feedback.email``
     - ``thellert@lbl.gov``
     - Recipient of the prefilled mail draft the dialog's Email channel opens.
   * - ``web.feedback.max_store_bytes``
     - ``268435456`` (256 MB)
     - Ceiling on the on-disk feedback store. Over it, the oldest saved session
       contexts are dropped; the submissions themselves are always kept.

.. code-block:: yaml

   web:
     docs_url: https://docs.example-facility.org/osprey
     feedback:
       github_repo: example-facility/controls
       email: controls-support@example.org
       max_store_bytes: 268435456

Three ways of writing one of the three string keys mean three different things:

- **Leave the key out** and the deployment uses the shipped default above.
- **Set it to an explicitly blank value** (``docs_url: ""``) and the deployment
  declares it has no such target: the Documentation link is not rendered at all,
  or the matching feedback channel is refused with an explanation rather than
  aimed at the upstream maintainers. This is the air-gapped posture — blanking
  ``web.docs_url`` is how you avoid shipping a link that opens a dead tab.
- **Write the key with no value at all** (``docs_url:`` and nothing after it)
  and it reads as *absent*, not blank — "I have not decided yet" rather than
  "there is none" — so you get the default. Write ``""`` when you mean none.

``max_store_bytes`` takes a positive byte count; anything else is reported in
the log and the default is used. A build profile overrides all four keys from
its ``config:`` block in the dotted form, e.g.
``web.feedback.max_store_bytes: 536870912``.

.. dropdown:: Under the hood
   :icon: gear

   .. tab-set::

      .. tab-item:: Settings

         A few options live under the ``web_terminal`` key in ``config.yml`` —
         which shell to launch, which directory to watch for live files, and how
         many background conversations to keep alive — and command-line flags
         override them for a single run. Give a deployment a name badge in the
         header with ``web.app_name`` (or the ``OSPREY_WEB_APP_NAME`` environment
         variable, handy when several containers share one config image).

         Three ``web`` keys bound the Simple-mode operator-chat pool:

         .. code-block:: yaml

            web:
              chat_turn_timeout_s: 600    # max seconds for one chat turn
              chat_idle_timeout_s: 1800   # idle sessions reaped after this
              chat_max_sessions: 5        # concurrent chat sessions cap

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

   :doc:`/how-to/send-feedback`
      The feedback dialog these keys configure, and the ``osprey feedback``
      verbs that read the results back.
