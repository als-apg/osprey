.. _phoebus-bridge:

=====================================
Connect the Agent to Phoebus Displays
=====================================

How to let the agent open, read and drive the displays of a running Phoebus
product through the ``phoebus`` MCP server, which is off in every preset until
a deployment switches it on.

.. dropdown:: What You'll Learn
   :color: primary
   :icon: book

   - What the ``phoebus`` server gives the agent, and which of its tools are
     asked before they run
   - How to point the server at the Phoebus agent bridge and switch it on
   - How to register the displays the agent may open by name
   - How to keep several terminals on one Phoebus from racing for focus
   - Where Data Browser plots and snapshots go, and which archiver a plot uses
   - How to add a second Phoebus instance
   - What guards a drive, and what does not

   **Prerequisites:** a Phoebus product built with the agent bridge, and a
   deployment you can rebuild with ``osprey build``.

What the server gives the agent
===============================

The server has eight tools, one per bridge operation:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Tool
     - What it does
   * - ``phoebus_list_displays``
     - Lists every display open in Phoebus, with its name, whether it is ready
       and whether it has focus.
   * - ``phoebus_perceive``
     - Walks one display's widget tree and reports each widget's type, name,
       bounds, visibility and live PV state.
   * - ``phoebus_perceive_region``
     - The same report, for only the widgets inside a screen rectangle.
   * - ``phoebus_snapshot``
     - Captures a PNG of one widget, saves it and registers it as an artifact.
   * - ``phoebus_open_panel``
     - Opens a display by its registered name and returns a handle that
       addresses that display in later calls.
   * - ``phoebus_panel_lookup``
     - Answers the reverse question: which registered name, if any, opens a
       given display file on this terminal.
   * - ``phoebus_open_databrowser``
     - Writes a Data Browser ``.plt`` for a list of channels over a time span,
       and opens it live in Phoebus.
   * - ``phoebus_drive``
     - Clicks an action control or types a value into a text control on a
       display.

The seven tools other than ``phoebus_drive`` read displays or open them and
run without a prompt. ``phoebus_drive`` is offered only under
``phoebus.agent_access: read_write``, and then it is always asked.

Before you start
================

The server talks to the **agent bridge**, an HTTP server that runs inside a
Phoebus product built with it. By default the bridge listens on
``http://127.0.0.1:7979``. It has to be reachable from wherever the
``phoebus`` server runs; this is the check the server's own error message
suggests:

.. code-block:: bash

   curl http://127.0.0.1:7979/displays

A bridge that answers returns a JSON list of the open displays. The bridge's
port can be changed with the ``org.phoebus.applications.bridge.web/port``
preference in a Phoebus settings file. When it is, ``phoebus.port`` must be set
to the same number.

Enable the server
=================

Switch the server on and name the bridge in the profile's ``config:`` block:

.. code-block:: yaml

   config:
     claude_code.servers.phoebus.enabled: true
     phoebus.host: 127.0.0.1
     phoebus.port: 7979

Then run ``osprey build`` and restart the stack. The build writes the
server's ``PHOEBUS_BRIDGE_URL`` into its entry as
``${PHOEBUS_BRIDGE_URL:-http://<phoebus.host>:<phoebus.port>}``. A
``PHOEBUS_BRIDGE_URL`` in the agent's environment therefore wins over both
keys; without one, the keys decide.

Register your panels
====================

``phoebus_open_panel`` opens displays by name, and the names come from
``phoebus.panels``, one dotted line per display:

.. code-block:: yaml

   config:
     phoebus.panels.overview: /opt/displays/overview.bob

A value may be an absolute path, a ``file:`` URL, or a path relative to the
directory of the rendered ``config.yml`` --- the project's ``build/``
directory, not the profile's. The path is handed to the Phoebus process, so it
must name a file that process can read. A name that is not registered is
refused, and the refusal lists the names that are. ``phoebus_panel_lookup``
answers the other way round: given a display path, it returns the name that
opens it.

The names belong to the deployment. OSPREY ships none.

Several terminals, one Phoebus
==============================

Every tool that takes a ``display`` argument defaults to ``"active"``, the
display that has focus in Phoebus. With several web terminals on one bridge,
that focus is shared, so two terminals working at once can resolve
``"active"`` to each other's display.

.. code-block:: yaml

   config:
     phoebus.require_handle: true

With this key set, ``"active"`` is refused. Callers pass the handle
``phoebus_open_panel`` returned (``"handle:d-3"``) or an explicit display name
from ``phoebus_list_displays``. ``PHOEBUS_REQUIRE_HANDLE`` (``1``/``true``/``yes``/``on``
or ``0``/``false``/``no``/``off``) outranks the key either way.

A multi-user web-terminal deployment stamps ``PHOEBUS_REQUIRE_HANDLE=1`` on
every terminal whose project runs a Phoebus server. Elsewhere the switch is off
by default and is turned on with ``phoebus.require_handle: true`` or
``PHOEBUS_REQUIRE_HANDLE=1``. ``phoebus.require_handle: false`` keeps
``"active"`` in a multi-user deployment too.

Data Browser plots and snapshots
================================

``phoebus_open_databrowser`` writes a ``.plt`` and opens it live. The archiver
bound into it comes from ``PHOEBUS_ARCHIVER_URL``, else
``phoebus.archiver_url``. With neither set, the plot has no archiver binding
and shows live values only.

.. code-block:: yaml

   config:
     phoebus.archiver_url: http://archiver.example.org:17668/retrieval

``phoebus.plot_dir`` and ``phoebus.snapshot_dir`` move the ``.plt`` files and
the PNG snapshots. By default they go to ``plots/`` and ``screenshots/`` under
``agent_data.base_dir``. A relative value is anchored on the project root.

A second Phoebus
================

A second instance of the server is declared with ``extends`` and given its own
bridge URL in its ``env:``. ``phoebus2`` is only an example name:

.. code-block:: yaml

   config:
     claude_code.servers.phoebus2.extends: phoebus
     claude_code.servers.phoebus2.env.PHOEBUS_BRIDGE_URL: "${PHOEBUS2_BRIDGE_URL:-http://127.0.0.1:7980}"

The clone reads the same ``phoebus.*`` keys as the first instance, so only the
URL tells the two apart. Its ``phoebus_drive`` is offered and asked like the original's.

What guards a drive
===================

The agent is offered ``phoebus_drive`` only where the build profile sets
``phoebus.agent_access: read_write``. Under the default, ``read``, the tool is
left out of the agent's tool list and permissions, and the server refuses a
call to it by the key's name. Under ``read_write`` every drive passes the
writes kill switch (``control_system.writes_enabled``) and then the approval
prompt, is refused while the control target is switched away from the
deployment's baseline (:doc:`switch-control-target`), and is audited as a tool
call. The key is all or nothing, because a panel runs whatever its widgets are
wired to, so finer control --- per panel, widget, channel or mode --- belongs to
the EPICS gateway or access security the Phoebus product connects through.

A drive does not pass through a connector: the bridge performs it inside the Phoebus process,
through that product's own PV connections. Synthetic mode, the default, runs
the display's confirm dialogs and enable rules; semantic mode bypasses them. So
OSPREY's channel-limits check does not apply to a drive, in either mode. Keep
``phoebus_drive`` for displays whose controls may be driven as they stand.
