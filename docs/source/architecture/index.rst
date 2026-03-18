Architecture Overview
=====================

OSPREY deploys agentic AI in safety-critical control system environments — particle accelerators,
fusion experiments, and beamlines. It uses **Claude Code** as the orchestrator, **MCP servers** as
the tool interface, and **pluggable connectors** for protocol-agnostic hardware access.

.. mermaid::

   flowchart LR
       User["Operator"] --> WebTerm["Web Terminal"]
       WebTerm --> Claude["Claude Code"]
       Claude --> MCP["MCP Servers"]
       MCP --> Conn["Connectors"]
       Conn --> HW["Control System<br/>(EPICS / Mock)"]

       Claude -->|"approval<br/>prompt"| User

       style User fill:#f9f,stroke:#333
       style Claude fill:#4a90e2,stroke:#333,color:#fff
       style MCP fill:#50c878,stroke:#333,color:#fff
       style HW fill:#ff9800,stroke:#333,color:#fff


Key Concepts
------------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Claude Code as Orchestrator
      :class-header: sd-bg-primary sd-text-white

      Claude Code replaces the previous LangGraph-based orchestration. It reads user intent,
      selects MCP tools, and manages multi-step workflows. All orchestration logic lives in
      Claude Code's conversation context — not in framework code.

   .. grid-item-card:: MCP Servers as Tool Interface
      :class-header: sd-bg-success sd-text-white

      Each MCP server is a `FastMCP <https://github.com/jlowin/fastmcp>`_ process that exposes
      domain-specific tools via the Model Context Protocol. Claude Code discovers and calls
      these tools directly.

      :doc:`See all MCP servers → <mcp-servers>`

   .. grid-item-card:: Connectors for Protocol Abstraction
      :class-header: sd-bg-info sd-text-white

      Connectors abstract control system protocols. Switch between EPICS (production) and
      Mock (development) via ``config.yml`` — no code changes needed.

      Supported: **EPICS**, **Mock**, or any custom connector via dotted import path.

   .. grid-item-card:: Human-in-the-Loop Safety
      :class-header: sd-bg-warning sd-text-white

      Every hardware write requires explicit operator approval through Claude Code's
      approval prompt. This is enforced at the tool level — ``channel_write`` always
      triggers a confirmation before executing.


Data Flow
---------

A typical control system interaction follows this path:

.. mermaid::

   sequenceDiagram
       participant Op as Operator
       participant WT as Web Terminal
       participant CC as Claude Code
       participant CS as Control System MCP
       participant Conn as EPICS/Mock Connector

       Op->>WT: "Set beam current to 500 mA"
       WT->>CC: Forward message
       CC->>CS: channel_read("BEAM:CURRENT")
       CS->>Conn: read_channel()
       Conn-->>CS: ChannelValue(450.0)
       CS-->>CC: "Current value: 450.0 mA"
       CC->>Op: "Current is 450 mA. Approve write to 500?"
       Op->>CC: Approve
       CC->>CS: channel_write("BEAM:CURRENT", 500.0)
       CS->>Conn: write_channel() [limits validated]
       Conn-->>CS: ChannelWriteResult(success=True)
       CS-->>CC: "Write confirmed, verified at 500.0 mA"
       CC->>Op: "Done. Beam current set to 500 mA."


Layers
------

OSPREY is organized into five layers, each with a clear responsibility:

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - Layer
     - Location
     - Purpose
   * - **CLI**
     - ``src/osprey/cli/``
     - 13 Click commands (``init``, ``deploy``, ``claude``, ``web``, etc.). Lazy-loaded.
   * - **MCP Servers**
     - ``src/osprey/mcp_server/``
     - FastMCP servers exposing tools for control systems, workspace, channel finding,
       ARIEL search, Python execution, MATLAB, and accelerator papers.
   * - **Connectors**
     - ``src/osprey/connectors/``
     - Protocol adapters (EPICS, Mock) created via ``ConnectorFactory``. Handles reads,
       writes, and archiver queries.
   * - **Services**
     - ``src/osprey/services/``
     - Backend logic: Channel Finder pipelines, ARIEL search, Python executor, DePlot.
   * - **Interfaces**
     - ``src/osprey/interfaces/``
     - Web UIs: Web Terminal (FastAPI + PTY), ARIEL search, Artifacts gallery,
       Channel Finder, lattice dashboard.


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

The runtime auto-configures from ``config.yml``, manages connector lifecycle, and validates
writes against safety limits before they reach the control system.

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
