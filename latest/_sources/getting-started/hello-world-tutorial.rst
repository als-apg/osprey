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

Create a ready-to-run project in one command:

.. code-block:: bash

   osprey build my-first-agent --preset hello-world
   cd my-first-agent

.. note::

   First run may take 1--2 minutes to create a virtual environment and install
   dependencies. Subsequent builds are near-instant.

Step 2: Understand What Was Generated
--------------------------------------

Your project has three key files that control everything:

**config.yml** --- Project Configuration

.. code-block:: yaml

   control_system:
     type: mock    # Change to 'epics' for real hardware
     writes_enabled: false   # Read-only by default

     limits_checking:
       enabled: true
       database_path: "data/channel_limits.json"
       allow_unlisted_channels: true

   claude_code:
     servers:
       controls: {enabled: true}   # The one MCP server

This tells OSPREY to use the **mock connector**, which accepts any channel
name and returns synthetic data. In production, you change ``type: mock`` to
``type: epics`` and point it at real hardware --- your agent code stays the same.

The ``limits_checking`` section enables the safety limits system. The limits
database defines allowed ranges and read-only channels. With
``allow_unlisted_channels: true``, channels not in the database can still be
read freely.

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

.. note::

   The ``hello_world`` template also installs Claude Code **hooks** for
   safety enforcement. These hooks run automatically before each tool call to
   check limits and require human approval for writes. You don't need to
   configure them manually.

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

   You: Read channel SR:BEAM:CURRENT

Expected output:

.. code-block:: text

   Channel: SR:BEAM:CURRENT
   Value:   250.3 mA
   Status:  OK

The mock connector returns synthetic beam current data. It generates
realistic values based on the channel name.

**Ask what's available:**

.. code-block:: text

   You: What channels are available?

The agent explains it can read any channel name --- the mock connector accepts
all PV names without needing a real control system.

**Read multiple channels:**

.. code-block:: text

   You: Read SR:VAC:PRESSURE:01 and SR:MAG:QF:01:CURRENT:RB

Claude calls the ``channel_read`` tool for each channel and presents the
results together --- vacuum pressure and magnet readback values.

Step 5: Safety & Limits
------------------------

OSPREY enforces safety at two levels: **limits checking** (automatic) and
**human approval** (interactive). This step walks through both.

**1. Enable writes**

Edit ``config.yml`` to allow write operations:

.. code-block:: yaml

   control_system:
     type: mock
     writes_enabled: true   # Changed from false

**2. Write within limits** (succeeds with approval)

.. code-block:: text

   You: Write 150.0 to SR:MAG:QF:01:CURRENT:SP

The value 150.0 is within the allowed range for this channel (0--300 A in the
limits database). The approval hook prompts you before executing:

.. code-block:: text

   ⚠ Write requested: SR:MAG:QF:01:CURRENT:SP = 150.0
   Approve? [y/N]

Type ``y`` to approve. The mock connector accepts the write and confirms.

**3. Write outside limits** (blocked)

.. code-block:: text

   You: Write 500.0 to SR:MAG:QF:01:CURRENT:SP

The value 500.0 exceeds the maximum of 300.0 defined in
``data/channel_limits.json``. The limits hook blocks the write before it
reaches the connector --- no approval prompt appears.

.. code-block:: text

   ✗ Write blocked: 500.0 exceeds max=300.0 for SR:MAG:QF:01:CURRENT:SP

**4. Write to a read-only channel** (blocked)

.. code-block:: text

   You: Write 1.0 to SR:BEAM:CURRENT

``SR:BEAM:CURRENT`` is marked as read-only in ``data/channel_limits.json``.
The limits hook blocks the write regardless of the value:

.. code-block:: text

   ✗ Write blocked: SR:BEAM:CURRENT is read-only

Step 6: Understand the Flow
----------------------------

Here's what happens when you ask "Read channel SR:BEAM:CURRENT":

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

Step 7: Customize
------------------

**Change agent behavior** by editing ``CLAUDE.md``:

Add a line like "Always report values with 4 decimal places" or "When reading
magnet channels, also explain what the magnet does." Restart Claude to see
the effect.

**Add new channels to the limits database** by editing
``data/channel_limits.json``:

.. code-block:: json

   {
     "SR:MAG:CORR:01:CURRENT:SP": {
       "min_value": -5.0,
       "max_value": 5.0,
       "writable": true
     }
   }

This lets you define safe operating ranges for additional channels. Any write
to a channel listed here is checked against its ``min_value``/``max_value``
range before the approval prompt.

Next Steps
==========

You've built a working agent with one MCP server and a mock control system.
Here's where to go from here:

- **Production deployment**: The :doc:`control-assistant` template adds channel
  finder, electronic logbook search, archiver access, and a web terminal
- **Architecture deep dive**: The :doc:`conceptual-tutorial` explains the
  MCP server architecture, connector system, and safety mechanisms
- **CLI reference**: See :doc:`../cli-reference/index` for all ``osprey`` commands
