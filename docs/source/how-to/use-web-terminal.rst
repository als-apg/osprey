Use the Web Terminal
====================

The OSPREY Web Terminal is a browser-based interface that wraps Claude Code in a
split-pane layout: a real terminal (PTY) on the left and a live workspace file
viewer on the right. It is built with FastAPI and communicates with the terminal
session over WebSocket, so every keystroke and resize event is forwarded in real
time.

Launching
---------

Start the web terminal from any OSPREY project directory:

.. code-block:: bash

   osprey web

The command boots a Uvicorn server on ``http://127.0.0.1:8087`` and
automatically opens your default browser. You can override the defaults:

.. code-block:: bash

   osprey web --host 0.0.0.0 --port 9000

The server reads ``config.yml`` from the project directory (or the path given by
``--config``). Use ``--shell`` to replace the default ``claude`` command.

**Background mode** — run the server as a detached process:

.. code-block:: bash

   osprey web --detach          # start in background
   osprey web stop              # stop the background server

When detached, the PID is written to ``.osprey-web.pid`` and logs go to
``.osprey-web.log`` in the project directory.

Companion Servers
^^^^^^^^^^^^^^^^^

On startup the web terminal auto-launches companion web servers that power the
side panels:

- **Artifact gallery** (port 8086) — workspace artifact browser
- **Karma analytics** (port 8741) — session transcript viewer

Domain-specific servers are launched when their config section is present:

- **ARIEL** (port 8085) — logbook search
- **Tuning panel** (port 8090) — parameter tuning UI
- **Channel Finder** (port 8092) — channel lookup
- **Lattice dashboard** (port 8097) — accelerator visualization

Each server's host, port, and ``auto_launch`` flag can be overridden in
``config.yml`` under its own section (e.g., ``artifact_server.port``).


Key Features
------------

Terminal PTY
^^^^^^^^^^^^

The left pane is a full terminal emulator backed by a server-side PTY. Input and
output travel over the ``/ws/terminal`` WebSocket. The terminal supports:

- **Resize**: the client sends ``{"type": "resize", "cols": N, "rows": N}``
  and the PTY window is adjusted immediately.
- **Session resume**: pass ``?session_id=UUID&mode=resume`` to reconnect to an
  existing Claude Code conversation.
- **Session switching**: send ``{"type": "switch_session", "session_id": UUID}``
  to hop between background sessions without losing state.
- **Background pool**: up to ``max_background_sessions`` (default 5) detached
  sessions are kept alive in a pool for quick switching.

Operator Mode
^^^^^^^^^^^^^

An alternative WebSocket endpoint (``/ws/operator``) drives Claude through the
Agent SDK instead of a PTY. The client sends structured JSON prompts and
receives typed events (text, thinking, tool use, errors).

Live Workspace Viewer
^^^^^^^^^^^^^^^^^^^^^

The right pane shows the workspace directory tree in real time. A file watcher
broadcasts change events over Server-Sent Events (``/api/files/events``), so new
artifacts, plots, and data files appear without a manual refresh.

- ``GET /api/files/tree`` -- directory tree as JSON
- ``GET /api/files/content/{path}`` -- file preview (up to 1 MB, text only)

The viewer can be scoped to a specific session by appending
``?session_id=UUID`` to the URL.

Settings Drawer
^^^^^^^^^^^^^^^

The settings drawer exposes the project ``config.yml`` through a REST API:

- ``GET /api/config`` -- read agent-relevant config sections as structured JSON
  plus the raw YAML.
- ``PUT /api/config`` -- validate and overwrite the entire file (a ``.yml.bak``
  backup is created automatically).
- ``PATCH /api/config`` -- apply dot-notation field updates (e.g.
  ``"control_system.writes_enabled": true``) while preserving comments and
  formatting.

After saving, the UI prompts you to restart the terminal so the agent picks up
the new settings.

Claude Code Setup Editor
^^^^^^^^^^^^^^^^^^^^^^^^

Read and edit the Claude Code integration files (``.claude/`` directory) without
leaving the browser:

- ``GET /api/claude-setup`` -- list all Claude Code files in the project.
- ``PUT /api/claude-setup`` -- save changes to an existing file.
- ``POST /api/claude-setup`` -- create a new file in an allowed subdirectory.

Session Diagnostics
^^^^^^^^^^^^^^^^^^^

A dedicated session panel lets you inspect the running conversation:

- **Session list** (``GET /api/sessions``) -- all Claude Code sessions
  registered for the project.
- **Agent hierarchy** (``GET /api/session-agents``) -- subagent tree with tool
  call breakdowns.
- **Agent timeline** (``GET /api/session-agent-timeline``) -- full internal
  timeline of a subagent's execution.
- **Tool call log** (``GET /api/session-log``) -- filterable by agent, tool
  name, error status, and time range.
- **Chat history** (``GET /api/session-chat``) -- human-readable conversation
  turns.
- **Artifact summary** (``GET /api/session-summary``) -- inventory of workspace
  artifacts with channel extraction.

Prompt Gallery
^^^^^^^^^^^^^^

Browse, edit, and override the prompt artifacts that shape the agent:

- ``GET /api/prompts`` -- list all prompt artifacts with framework/user-owned
  status.
- ``POST /api/prompts/{name}/scaffold`` -- scaffold an override from the
  framework template.
- ``PUT /api/prompts/{name}/override`` -- save a custom override.
- ``DELETE /api/prompts/{name}/override`` -- revert to the framework version.
- ``GET /api/prompts/{name}/diff`` -- unified diff between framework and
  override.

Memory Gallery
^^^^^^^^^^^^^^

Manage Claude memory files (Markdown notes persisted across sessions):

- ``GET /api/claude-memory`` -- list memory files.
- ``POST /api/claude-memory`` -- create a new memory file.
- ``PUT /api/claude-memory/{filename}`` -- update an existing file.
- ``DELETE /api/claude-memory/{filename}`` -- delete a file.

Side Panels
^^^^^^^^^^^

The web terminal can embed companion services as iframe panels. **Universal
panels** (artifacts, karma analytics) are always enabled. **Domain panels** are
activated in ``config.yml``:

.. code-block:: yaml

   web:
     panels:
       ariel: true
       tuning: true
       channel-finder: true

Custom panels pointing to external URLs are also supported:

.. code-block:: yaml

   web:
     panels:
       my-dashboard:
         label: My Dashboard
         url: http://localhost:3000
         health_endpoint: /health

The ``/api/panel-focus`` endpoint lets MCP tools programmatically switch the
active panel.


Configuration Reference
-----------------------

All options live under the ``web_terminal`` key in ``config.yml``:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Key
     - Default
     - Description
   * - ``shell``
     - ``claude``
     - Shell command spawned in the PTY.
   * - ``watch_dir``
     - ``./osprey-workspace``
     - Directory watched for live file events.
   * - ``max_background_sessions``
     - ``5``
     - Maximum detached PTY sessions kept alive in the pool.

The server itself accepts ``--host`` (default ``127.0.0.1``), ``--port``
(default ``8087``), ``--config``, ``--shell``, and ``--detach`` on the command
line. CLI flags take precedence over ``config.yml`` values.

Hook Debugging
^^^^^^^^^^^^^^

When ``hooks.debug`` is enabled in ``config.yml``, the web terminal exposes:

- ``GET /api/hooks/debug-status`` -- whether hook debug mode is active.
- ``GET /api/hooks/debug-log`` -- recent entries from ``hook_debug.jsonl``.

Health Check
^^^^^^^^^^^^

``GET /health`` returns the service status, version, and server session ID.
