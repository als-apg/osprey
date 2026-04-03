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


Data Flow
---------

A typical control system write follows this path. The three safety hooks fire between Claude Code
and the MCP server:

.. mermaid::

   sequenceDiagram
       participant Op as Operator
       participant WT as Web Terminal
       participant CC as Claude Code
       participant CF as Channel Finder Sub-Agent
       participant H as Safety Hooks
       participant CS as Control System MCP
       participant Conn as EPICS/Mock Connector

       Op->>WT: "Bump the horizontal corrector at sector 3 by 0.5 A"
       WT->>CC: Forward message

       Note over CC,CF: Channel address lookup via sub-agent
       activate CF
       CC->>CF: "horizontal corrector setpoint, sector 3"
       CF->>CF: Query Channel Finder DB
       CF-->>CC: SR03C:COR:H1:CURRENT
       deactivate CF

       Note over CC,CS: Read current value to compute target
       CC->>CS: channel_read("SR03C:COR:H1:CURRENT")
       CS->>Conn: read_channel()
       Conn-->>CS: ChannelValue(1.2)
       CS-->>CC: "Current value: 1.2 A"

       Note over CC,H: channel_write triggers hook chain
       CC->>H: PreToolUse: channel_write("SR03C:COR:H1:CURRENT", 1.7)
       H->>H: 1. writes_check → enabled
       H->>H: 2. limits → 1.7 within range
       H->>Op: 3. approval → "Write 1.7 A to SR03C:COR:H1:CURRENT?"
       Op->>H: Approve
       H->>CS: Proceed
       CS->>Conn: write_channel() [limits validated]
       Conn-->>CS: ChannelWriteResult(success=True)
       CS-->>CC: "Write confirmed, verified at 1.7 A"
       CC->>Op: "Done. Corrector SR03C:COR:H1 set to 1.7 A (+0.5 A)."


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
