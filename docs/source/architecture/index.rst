Architecture Overview
=====================

OSPREY deploys agentic AI in safety-critical control system environments — particle accelerators,
fusion experiments, and beamlines. It uses **Claude Code** as the orchestrator, **MCP servers** as
the tool interface, and **pluggable connectors** for protocol-agnostic hardware access.

.. mermaid::

   flowchart LR
       User["Operator"] --> WebTerm["Web Terminal"]
       WebTerm --> Claude["Claude Code"]
       Claude --> Hooks["Safety Hooks"]
       Hooks --> MCP["MCP Servers"]
       MCP --> Conn["Connectors"]
       Conn --> HW["Control System<br/>(EPICS / Mock)"]

       style User fill:#f9f,stroke:#333
       style Claude fill:#4a90e2,stroke:#333,color:#fff
       style Hooks fill:#e74c3c,stroke:#333,color:#fff
       style MCP fill:#50c878,stroke:#333,color:#fff
       style HW fill:#ff9800,stroke:#333,color:#fff


Safety Chain
------------

Every tool invocation passes through a configurable chain of **PreToolUse hooks** before reaching
the MCP server. The chain for ``channel_write`` — the most safety-critical tool — has three stages:

.. mermaid::

   flowchart LR
       CC["Claude Code<br/>calls channel_write"] --> WC["writes_check<br/><i>kill switch</i>"]
       WC -->|enabled| LIM["limits<br/><i>min / max / step</i>"]
       LIM -->|valid| APR["approval<br/><i>human gate</i>"]
       APR -->|approved| MCP["MCP Server<br/>executes write"]
       WC -->|disabled| BLOCK["❌ Blocked"]
       LIM -->|invalid| BLOCK
       APR -->|rejected| BLOCK

       style WC fill:#e74c3c,stroke:#333,color:#fff
       style LIM fill:#e67e22,stroke:#333,color:#fff
       style APR fill:#f39c12,stroke:#333,color:#fff
       style BLOCK fill:#95a5a6,stroke:#333,color:#fff

1. **osprey_writes_check** — Kill switch. Blocks all writes when ``control_system.writes_enabled``
   is ``false`` in ``config.yml``. Applies to both ``channel_write`` and ``execute``.

2. **osprey_limits** — Validates the setpoint against the channel limits database
   (min, max, step size, writable flag). Only applies to ``channel_write``.

3. **osprey_approval** — Human approval gate. Per-tool policy dispatch: ``always`` (require
   approval every time), ``selective`` (ask Claude to decide), or ``skip``.


Build & Deploy
--------------

OSPREY projects are assembled from **build profiles** — YAML files that declare which MCP servers,
connectors, hooks, agents, rules, and skills to include.

.. code-block:: bash

   # Assemble a project from a build profile
   osprey build my-project profiles/als.yml

   # Start Docker-compose services (web terminal, supporting services)
   osprey deploy up

``osprey build`` produces a self-contained project directory with:

- ``.mcp.json`` — MCP server configuration for Claude Code
- ``config.yml`` — facility-specific settings (connector type, channel limits, approval policies)
- ``claude/hooks/`` — safety hook chain, wired per the registry
- ``claude/agents/``, ``rules/``, ``skills/`` — Claude Code customization
- A Python virtual environment with all dependencies

Build profiles can inject additional MCP servers, override default hooks, and customize approval
policies per facility.


Layers
------

OSPREY is organized into seven layers:

.. list-table::
   :header-rows: 1
   :widths: 18 30 52

   * - Layer
     - Location
     - Purpose
   * - **CLI**
     - ``src/osprey/cli/``
     - 14 Click commands (``init``, ``build``, ``deploy``, ``config``, ``health``, etc.). Lazy-loaded.
   * - **MCP Servers**
     - ``src/osprey/mcp_server/``
     - 8 in-tree FastMCP servers. Build profiles can add more.
   * - **Connectors**
     - ``src/osprey/connectors/``
     - EPICS + Mock for both control system and archiver, via ``ConnectorFactory``.
   * - **Services**
     - ``src/osprey/services/``
     - 6 internal packages: ariel_search, channel_finder, machine_state, migration,
       prompts, python_executor.
   * - **Interfaces**
     - ``src/osprey/interfaces/``
     - Web UIs: Web Terminal (FastAPI + PTY), ARIEL, Artifacts gallery, Channel Finder,
       Lattice Dashboard.
   * - **Runtime API**
     - ``src/osprey/runtime/``
     - ``write_channel`` / ``read_channel`` for generated Python scripts.
   * - **Templates**
     - ``src/osprey/templates/claude_code/``
     - 8 hooks, 5 agents, 10 rules, 6 skills — assembled by ``osprey build``.


Data Flow
---------

A typical control system write follows this path. The three safety hooks fire between Claude Code
and the MCP server:

.. mermaid::

   sequenceDiagram
       participant Op as Operator
       participant WT as Web Terminal
       participant CC as Claude Code
       participant H as Safety Hooks
       participant CS as Control System MCP
       participant Conn as EPICS/Mock Connector

       Op->>WT: "Set beam current to 500 mA"
       WT->>CC: Forward message
       CC->>CS: channel_read("BEAM:CURRENT")
       CS->>Conn: read_channel()
       Conn-->>CS: ChannelValue(450.0)
       CS-->>CC: "Current value: 450.0 mA"

       Note over CC,H: channel_write triggers hook chain
       CC->>H: PreToolUse: channel_write("BEAM:CURRENT", 500.0)
       H->>H: 1. writes_check → enabled
       H->>H: 2. limits → 500.0 within range
       H->>Op: 3. approval → "Approve write to 500?"
       Op->>H: Approve
       H->>CS: Proceed
       CS->>Conn: write_channel() [limits validated]
       Conn-->>CS: ChannelWriteResult(success=True)
       CS-->>CC: "Write confirmed, verified at 500.0 mA"
       CC->>Op: "Done. Beam current set to 500 mA."


Runtime API
-----------

For generated Python scripts that need to interact with hardware, OSPREY provides a synchronous
runtime API that works like EPICS ``caput``/``caget``:

.. code-block:: python

   from osprey.runtime import write_channel, read_channel

   # Read current value (like caget)
   current = read_channel("BEAM:CURRENT")

   # Write new value (like caput) — limits validated automatically
   write_channel("BEAM:CURRENT", 500.0)

The runtime auto-configures from ``config.yml``, manages connector lifecycle, and validates writes
against safety limits. Limits are enforced at two levels: the runtime ``LimitsValidator`` checks
before dispatching, and the connector may enforce its own constraints.

.. seealso::

   :doc:`mcp-servers`
      Complete list of MCP servers and their tools.

   :doc:`/how-to/add-connector`
      How to add a custom control system connector.

   :doc:`/how-to/deploy-project`
      How to create and deploy an OSPREY project.

.. toctree::
   :hidden:

   mcp-servers
