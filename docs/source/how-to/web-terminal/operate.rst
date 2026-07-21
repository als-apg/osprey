Run the Web Terminal
====================

Start the Web Terminal from any OSPREY project directory:

.. code-block:: bash

   osprey web

It boots a local server on ``http://127.0.0.1:8087`` and opens your browser.
Override the defaults, or point it at another project, with ``--host``,
``--port``, and ``--project``.

To keep it running after you close the terminal, start it in the background:

.. code-block:: bash

   osprey web --detach     # start in the background
   osprey web stop         # stop it again

In background mode the process id and logs are written to ``.osprey-web.pid``
and ``.osprey-web.log`` in the project directory.

What you get
------------

The window has three working areas plus a header:

- **Terminal** (right) — a real terminal running the Osprey agent. It survives
  reconnects, and you can keep a few background conversations alive and hop
  between them.
- **Workspace** (left) — a live view of your project files. New artifacts,
  plots, and data files appear as the agent creates them, with no refresh.
- **Side panels** — your control-system tools (Channel Finder, ARIEL, the
  lattice dashboard, and so on) as tabs beside the chat. See :doc:`panels`.
- **Header** — the theme switcher (:doc:`theming`), a settings drawer, and an
  optional name badge to tell one deployment from another.

The settings drawer lets you read and edit the project's ``config.yml`` — and
the agent's own setup and memory files — from the browser, so you rarely need
to drop back to an editor. Changes prompt you to restart the terminal so the
agent picks them up.

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
