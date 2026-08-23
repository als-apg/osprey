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
