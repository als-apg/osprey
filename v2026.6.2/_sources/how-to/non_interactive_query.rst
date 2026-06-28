Non-Interactive Agent Queries
==============================

How to use ``osprey query`` to run the OSPREY agent headlessly — in CI
pipelines, health checks, and automated workflows.

.. dropdown:: What You'll Learn
   :color: primary
   :icon: book

   - What ``osprey query`` does and when to reach for it
   - How project resolution works (``--project`` → ``OSPREY_PROJECT`` → cwd)
   - The ``--json`` flag for machine-readable output
   - Exit codes and what each means
   - The read-only guarantee and how it is enforced
   - How ``osprey query`` differs from ``osprey health``

   **Prerequisites:** A project built with ``osprey build``.

Overview
========

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

Project Resolution
==================

``osprey query`` resolves the project directory in this order:

1. ``--project DIR`` flag (explicit)
2. ``OSPREY_PROJECT`` environment variable
3. Current working directory (cwd)

.. code-block:: bash

   # Explicit project
   osprey query --project ~/projects/als-assistant "List vacuum sections"

   # Via environment variable
   export OSPREY_PROJECT=~/projects/als-assistant
   osprey query "List vacuum sections"

   # Current directory (cwd)
   cd ~/projects/als-assistant
   osprey query "List vacuum sections"

The command exits with code ``2`` if no valid project is found at the
resolved path.

Exit Codes
==========

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
=====================================

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
===================

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
- **Destructive workspace tools** — ``data_delete``, ``artifact_delete``, and
  ``artifact_delete_all``, which permanently remove stored data/artifacts.
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
use ``osprey claude chat`` or the event-dispatch pipeline instead.

.. note::

   This guarantee is independent of the ``claude_code.permissions`` settings
   in ``config.yml``: the headless runner uses ``bypassPermissions`` (under
   which the interactive allow/deny/approval guards are inert), so the
   read-only guarantee rests *entirely* on the SDK-level ``disallowed_tools``
   block. That is precisely why the block must cover the built-in and
   approval-required tools above, not just the ``hook_config.json`` write list.

CI Loop Pattern
===============

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

For a project not in the current directory:

.. code-block:: bash

   osprey query \
     --project /opt/osprey/als-assistant \
     "List all vacuum sections" \
     || exit 1

``osprey query`` vs. ``osprey health``
=======================================

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

   :doc:`use-cli-chat`
       Interactive agent sessions in your native terminal.

   :doc:`event-dispatch`
       Turn external events into headless agent runs via webhooks and cron.

   :doc:`/cli-reference/index`
       Full ``osprey query`` and ``osprey health`` command reference.
