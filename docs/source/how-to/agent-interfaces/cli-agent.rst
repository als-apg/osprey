.. _how-to-cli-agent:

===================================
Run the Agent from the Command Line
===================================

Two commands run the Osprey agent without the web terminal. ``osprey chat``
opens an interactive session in your native terminal; ``osprey query`` runs a
single headless prompt and exits with a meaningful code, for CI pipelines and
automated workflows.

.. dropdown:: What You'll Learn
   :color: primary
   :icon: book

   - Launching interactive terminal sessions with ``osprey chat``, and how they
     compare to the web terminal
   - The companion web services both modes share
   - Headless one-shot runs with ``osprey query``: exit codes, ``--json``
     output, and the read-only guarantee
   - CI patterns, and how ``osprey query`` differs from ``osprey health``

   **Prerequisites:** A project built with ``osprey build``.

Interactive Sessions (``osprey chat``)
======================================

The CLI chat interface launches the Osprey agent in your native terminal while
running OSPREY's companion services in the background. This runs the agent
in the full terminal TUI — keyboard shortcuts, slash commands, native
scrollback — with access to companion services (artifact gallery, session
analytics, etc.) via their URLs in a browser.

Launching
---------

From anywhere inside a deployment repository:

.. code-block:: bash

   osprey chat

This command:

1. Finds the deployment by walking up to the nearest ``profile.yml``, and starts
   the agent in that repository's ``build/`` — nothing is re-rendered, since
   ``osprey build`` owns that. A profile that has changed since the last build
   is reported as a warning and the session starts anyway.
2. Resolves the configured LLM provider and injects authentication.
3. Starts the translation proxy if the provider needs it (see
   :doc:`../llm-providers/configure-providers`).
4. Launches companion web servers in the background.
5. Opens the Osprey agent TUI in your terminal.

Options
^^^^^^^

.. code-block:: bash

   osprey chat --repo /path/to/deployment    # explicit deployment repo
   osprey chat --resume SESSION_ID           # resume a previous session
   osprey chat --print                       # non-interactive (pipe-friendly)
   osprey chat --effort high                 # set effort level
   osprey chat --no-pin                      # ignore the pinned CLI version

When ``--repo`` is omitted, the deployment enclosing the current directory is used.

If ``claude_code.cli_version`` is set in ``config.yml``, chat launches that
exact agent CLI version instead of whatever is installed globally, so every
launch of the project behaves the same. ``--no-pin`` opts out and uses the
global installation.

Companion Services
------------------

On startup, ``osprey chat`` launches the same companion servers as
``osprey web``. Each server's URL is printed before the Osprey agent starts:

.. code-block:: text

   Companion servers
     Artifact gallery   http://127.0.0.1:8086
     ARIEL server       http://127.0.0.1:8085

Open any of these URLs in a browser to access the service while the Osprey agent
runs in your terminal. Which servers start depends on your ``config.yml`` —
each server respects its own ``auto_launch`` setting.

The servers run as background threads and stop automatically when you exit
the Osprey agent.

When to Use CLI vs. Web Terminal
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - CLI chat (``osprey chat``)
     - Web terminal (``osprey web``)
   * - Native terminal experience
     - Browser-based split-pane UI
   * - Full Osprey agent TUI with keyboard shortcuts
     - Embedded terminal emulator
   * - Companion services in separate browser tabs
     - Companion services as side panels
   * - No additional port for the terminal itself
     - Terminal served on port 8087
   * - Ideal for SSH or remote sessions
     - Ideal for local development with visual tools

Both modes launch the same MCP servers, companion services, and translation
proxy. The agent capabilities are identical.

Headless Queries (``osprey query``)
===================================

``osprey query "<prompt>"`` boots the full OSPREY agent — MCP servers,
tools, provider authentication — passes your prompt, waits for a response,
and exits with a code that summarises the outcome. Write, execute, and
destructive tools are blocked at the SDK level (see `Read-Only Guarantee`_
below): the command will not write to hardware, run shell commands or code,
or delete stored data, making it safe for CI pipelines and automated
workflows. (To deliver an answer it may still create workspace artifacts such
as plots — see the guarantee's exact scope below.)

Use ``osprey query`` when you need to verify that the *entire agent stack*
works end-to-end against your real control system, not just that the model
is reachable.

Deployment Resolution
---------------------

``osprey query`` asks the deployment enclosing the current directory: it walks
up to the nearest ``profile.yml`` and answers from that repository's ``build/``
as it was last rendered. ``--repo DIRECTORY`` names another one.

.. code-block:: bash

   # The deployment you are standing in
   cd ~/deployments/als-assistant
   osprey query "List vacuum sections"

   # A deployment somewhere else
   osprey query --repo ~/deployments/als-assistant "List vacuum sections"

When ``profile.yml`` or a file it points at has changed since that build, a
warning on stderr says so and the query runs anyway. The command exits with
code ``2`` if there is no build to answer from.

Exit Codes
----------

.. list-table::
   :header-rows: 1
   :widths: 10 25 65

   * - Code
     - Meaning
     - When it occurs
   * - ``0``
     - Pass
     - The agent completed without error and all expected MCP servers
       connected.
   * - ``1``
     - Verdict fail
     - The agent ran but the outcome was unsatisfactory: a required MCP
       server was missing from the session, the agent returned an error
       result, or the run hit its cost budget (USD) or turn limit.
   * - ``2``
     - Infra / usage error
     - The run never started: no project found at the resolved path, the
       agent SDK is not installed, or the provider is not configured.

Machine-Readable Output (``--json``)
------------------------------------

Pass ``--json`` to receive a structured JSON object instead of plain text:

.. code-block:: bash

   osprey query --json "Summarise recent alarms"

Output format:

.. code-block:: json

   {
     "final_text": "...",
     "tool_traces": [
       {"name": "tool_name", "input": {}, "result": "...", "is_error": false}
     ],
     "mcp_servers": {"controls": "connected", "ariel": "connected"},
     "exit_code": 0
   }

``mcp_servers`` maps each expected server name to its connection-status
string (e.g. ``"connected"``) from the MCP readiness snapshot taken before
the prompt is sent. ``exit_code`` repeats the shell exit code so the whole
verdict is self-contained in the JSON object.

Read-Only Guarantee
-------------------

``osprey query`` enforces read-only execution at the SDK level by passing a
comprehensive ``disallowed_tools`` list directly to the runner. The agent
cannot call any of these tools even if it tries. The list is the union of:

- **Built-in write/exec/network tools** — ``Bash`` (arbitrary shell, which
  could itself write hardware via ``caput``), ``Write``, ``Edit``,
  ``MultiEdit``, ``NotebookEdit``, ``WebFetch``, ``WebSearch``.
- **Control-system write tools** — ``mcp__controls__channel_write`` and
  ``mcp__python__execute`` (plus any facility-specific tools in the project's
  ``hook_config.json`` ``write_tools`` block). These are always included even
  if a project's ``hook_config.json`` omits them.
- **Approval-required MCP actions** — every tool a framework MCP server marks
  as side-effecting (e.g. ``mcp__ariel__entry_create`` logbook writes,
  ``mcp__python__execute_file``, ``mcp__osprey_workspace__setup_patch``),
  derived from the server registry so the blocked set tracks the framework
  rather than a hand-maintained copy.
- **Destructive workspace tools** — ``artifact_delete`` and
  ``artifact_delete_all``, which permanently remove stored artifacts.
- **Facility-custom write tools** — any tool a custom MCP server in
  ``claude_code.servers`` marks under ``permissions.ask`` (or a custom server
  that writes-check-gates all its tools) is blocked too. Declare custom write
  tools under ``permissions.ask`` so the read-only path recognises them.

Audited *read* tools (e.g. ``mcp__controls__channel_read``) remain available.
The agent may still **create** workspace artifacts (plots, documents) and write
session logs to deliver its answer — these are additive and regenerable. The
guarantee is specifically that the run performs no hardware write, no code or
shell execution, and no deletion of stored data.

There is no ``--allow-writes`` flag. If you need an agent run that can write,
use ``osprey chat`` or the event-dispatch pipeline instead.

.. note::

   This guarantee is independent of the ``claude_code.permissions`` settings
   in ``config.yml``: the headless runner uses ``bypassPermissions`` (under
   which the interactive allow/deny/approval guards are inert), so the
   read-only guarantee rests *entirely* on the SDK-level ``disallowed_tools``
   block. That is precisely why the block must cover the built-in and
   approval-required tools above, not just the ``hook_config.json`` write list.

CI Loop Pattern
---------------

.. important::

   OSPREY ships no canned queries. A query that references a channel address,
   PV name, or subsystem label only makes sense for your facility. Write
   queries that reflect your real control system — a generic query cannot
   validate a facility-specific agent.

The recommended CI pattern is a direct ``|| exit 1`` guard:

.. code-block:: bash

   osprey query "What PVs are used for beam current?" || exit 1

For pipelines that parse results, use ``--json`` and ``jq``:

.. code-block:: bash

   result=$(osprey query --json "Summarise recent alarms")
   echo "$result" | jq '.final_text'
   echo "$result" | jq '.exit_code'

For a deployment not in the current directory:

.. code-block:: bash

   osprey query \
     --repo /opt/osprey/als-assistant \
     "List all vacuum sections" \
     || exit 1

``osprey query`` vs. ``osprey health``
--------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - ``osprey health``
     - ``osprey query``
   * - Raw model ping — "can I reach the model?"
     - Full agent run — "does the agent boot its MCP servers and answer
       against my control system?"
   * - Checks configuration, connectivity, and optionally sends a minimal
       test prompt
     - Sends your real prompt through the full MCP + tools stack
   * - Fast; optional API cost
     - Slower; costs one normal agent run
   * - Good for diagnosing a broken install
     - Good for verifying end-to-end agent capability in CI

Run ``osprey health`` first when something is wrong with the install.
Run ``osprey query`` to confirm the agent answers control-system questions
correctly once the install is healthy.

.. seealso::

   :doc:`../web-terminal/index`
       The browser cockpit — the interactive alternative to ``osprey chat``.

   :doc:`event-dispatch`
       Turn external events into headless agent runs via webhooks and cron.

   :doc:`/reference/cli`
       Full ``osprey chat``, ``osprey query``, and ``osprey health`` command
       reference.
