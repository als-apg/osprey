===============================
Hello World Tutorial
===============================

Build your first OSPREY agent. One MCP server, one mock control system, zero
complexity. By the end of this tutorial, you'll have an agent that reads
control system channels using Claude Code.

.. dropdown:: **Prerequisites**
   :color: info
   :icon: list-unordered

   **Required:**

   - Python 3.11+
   - `Claude Code <https://docs.anthropic.com/en/docs/claude-code/overview>`_ CLI installed
   - Osprey framework installed (``uv sync``)
   - ``ANTHROPIC_API_KEY`` set in your environment

   If you haven't installed the framework yet, follow the :doc:`installation guide <installation>`.

Step 1: Create the Project
---------------------------

.. tab-set::

   .. tab-item:: Interactive Mode (Recommended)

      Launch the interactive wizard:

      .. code-block:: bash

         osprey init

      Select ``hello_world`` when prompted for a template, then follow the
      prompts to choose your AI provider and model.

   .. tab-item:: Direct CLI Command

      .. code-block:: bash

         osprey init my-first-agent --template hello_world
         cd my-first-agent

Both methods create identical project structures. The wizard auto-detects
API keys from your environment.

Step 2: Understand What Was Generated
--------------------------------------

Your project has three key files that control everything:

**config.yml** --- Project Configuration

.. code-block:: yaml

   control_system:
     type: mock    # Change to 'epics' for real hardware
     writes_enabled: false   # Read-only by default

   claude_code:
     servers:
       controls: {enabled: true}   # The one MCP server

This tells OSPREY to use the **mock connector**, which accepts any channel
name and returns synthetic data. In production, you change ``type: mock`` to
``type: epics`` and point it at real hardware --- your agent code stays the same.

**.mcp.json** --- MCP Server Discovery

.. code-block:: json

   {
     "mcpServers": {
       "controls": {
         "command": "python",
         "args": ["-m", "osprey.mcp_server.control_system"],
         "env": { "OSPREY_CONFIG": "config.yml" }
       }
     }
   }

This is how Claude Code discovers the control system MCP server. When Claude
starts, it launches this server and gains access to tools like
``channel_read`` and ``channel_write``.

**CLAUDE.md** --- Agent Behavior Instructions

This file contains instructions that Claude reads on startup. It defines how
the agent should behave, what safety rules to follow, and how to format
responses. Edit this file to customize your agent's personality and behavior.

Step 3: Start the Agent
------------------------

From your project directory:

.. code-block:: bash

   claude

.. note::

   On first run, Claude Code will ask you to trust the MCP servers in this
   project. Accept to allow Claude to use the control system tools.

   Claude connects to the ``control_system`` MCP server and is ready to accept
   queries.

Step 4: Try It Out
-------------------

Try these queries to see the mock control system in action:

**Read a single channel:**

.. code-block:: text

   You: Read channel SR:CURRENT:RB

The mock connector returns synthetic beam current data. It generates
realistic values based on the channel name.

**Ask what's available:**

.. code-block:: text

   You: What channels are available?

The agent explains it can read any channel name --- the mock connector accepts
all PV names without needing a real control system.

**Read multiple channels:**

.. code-block:: text

   You: Read SR:MAG:QF:01:CURRENT:RB and SR:MAG:QD:01:CURRENT:RB

Claude calls the ``channel_read`` tool for each channel and presents the
results together.

Step 5: Understand the Flow
----------------------------

Here's what happens when you ask "Read channel SR:CURRENT:RB":

.. code-block:: text

   You ──→ Claude Code ──→ channel_read MCP tool
                                    │
                              Mock Connector
                                    │
                           Synthetic data ──→ Claude formats response ──→ You

1. You type a natural language query
2. Claude decides to call the ``channel_read`` MCP tool
3. The MCP server receives the call and routes it to the **mock connector**
4. The mock connector generates synthetic data (realistic noise, naming-based values)
5. Claude formats the response and presents it to you

**This is the same flow in production** --- just with ``type: epics`` instead of
``type: mock``. The connector handles the difference; your agent and queries
stay the same.

Step 6: Customize
------------------

**Change agent behavior** by editing ``CLAUDE.md``:

Add a line like "Always report values with 4 decimal places" or "When reading
magnet channels, also explain what the magnet does." Restart Claude to see
the effect.

**Enable writes** by editing ``config.yml``:

.. code-block:: yaml

   control_system:
     type: mock
     writes_enabled: true   # ← changed from false

Now try:

.. code-block:: text

   You: Write 150.0 to SR:MAG:QF:01:CURRENT:SP

Claude will ask for your **explicit approval** before executing the write ---
this is OSPREY's human-in-the-loop safety mechanism. In production with real
hardware, this approval step ensures no accidental writes to the control system.

Next Steps
==========

You've built a working agent with one MCP server and a mock control system.
Here's where to go from here:

- **Production deployment**: The :doc:`control-assistant` template adds channel
  finder, electronic logbook search, archiver access, and a web terminal
- **Architecture deep dive**: The :doc:`conceptual-tutorial` explains the
  MCP server architecture, connector system, and safety mechanisms
- **CLI reference**: See :doc:`../cli-reference/index` for all ``osprey`` commands
