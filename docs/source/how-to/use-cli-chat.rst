Use the CLI Chat Interface
==========================

The CLI chat interface launches Claude Code in your native terminal while
running OSPREY's companion services in the background. This gives you the full
Claude Code TUI experience — keyboard shortcuts, slash commands, native
scrollback — with access to companion services (artifact gallery, session
analytics, etc.) via their URLs in a browser.

Launching
---------

From any OSPREY project directory:

.. code-block:: bash

   osprey claude chat

This command:

1. Regenerates Claude Code integration files (``.mcp.json``, ``CLAUDE.md``,
   ``.claude/settings.json``) from ``config.yml``.
2. Resolves the configured LLM provider and injects authentication.
3. Starts the translation proxy if the provider needs it (see
   :doc:`configure-providers`).
4. Launches companion web servers in the background.
5. Opens the Claude Code TUI in your terminal.

Options
^^^^^^^

.. code-block:: bash

   osprey claude chat --project /path/to/project   # explicit project dir
   osprey claude chat --resume SESSION_ID           # resume a previous session
   osprey claude chat --print                       # non-interactive (pipe-friendly)
   osprey claude chat --effort high                 # set effort level

When ``--project`` is omitted, the current directory is used.

Companion Services
------------------

On startup, ``osprey claude chat`` launches the same companion servers as
``osprey web``. Each server's URL is printed before Claude Code starts:

.. code-block:: text

   Companion servers:
     * Artifact gallery  http://127.0.0.1:8086
     * ARIEL server      http://127.0.0.1:8085

Open any of these URLs in a browser to access the service while Claude Code
runs in your terminal. Which servers start depends on your ``config.yml`` —
each server respects its own ``auto_launch`` setting.

The servers run as background threads and stop automatically when you exit
Claude Code.

When to Use CLI vs. Web Terminal
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - CLI chat (``osprey claude chat``)
     - Web terminal (``osprey web``)
   * - Native terminal experience
     - Browser-based split-pane UI
   * - Full Claude Code TUI with keyboard shortcuts
     - Embedded terminal emulator
   * - Companion services in separate browser tabs
     - Companion services as side panels
   * - No additional port for the terminal itself
     - Terminal served on port 8087
   * - Ideal for SSH or remote sessions
     - Ideal for local development with visual tools

Both modes launch the same MCP servers, companion services, and translation
proxy. The agent capabilities are identical.
