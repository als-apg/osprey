.. _phoebus-bridge:

=====================================
Connect the Agent to Phoebus Displays
=====================================

This page shows you how to let the agent open, read and drive the displays of
a running Phoebus product. The agent does this through the ``phoebus`` MCP
server. The server is off in every preset until a deployment switches it on.

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

The seven tools other than ``phoebus_drive`` read or open displays, and they
run without a prompt. The agent is offered ``phoebus_drive`` only under
``phoebus.agent_access: read_write``, and then it is always asked.

Before you start
================

The server talks to the **agent bridge**, an HTTP server that runs inside a
Phoebus product built with it. By default the bridge listens on
``http://127.0.0.1:7979``.

Check that the bridge answers from the host where the ``phoebus`` server runs.
This is the check the server's own error message suggests:

.. code-block:: bash

   curl http://127.0.0.1:7979/displays

A bridge that answers returns a JSON list of the open displays.

To move the bridge to another port, set the
``org.phoebus.applications.bridge.web/port`` preference in a Phoebus settings
file. Then set ``phoebus.port`` to the same number.

Enable the server
=================

#. Switch the server on and name the bridge in the profile's ``config:``
   block:

   .. code-block:: yaml

      config:
        claude_code.servers.phoebus.enabled: true
        phoebus.host: 127.0.0.1
        phoebus.port: 7979

#. Run ``osprey build``. The build writes the server's ``PHOEBUS_BRIDGE_URL``
   into its entry as
   ``${PHOEBUS_BRIDGE_URL:-http://<phoebus.host>:<phoebus.port>}``.

#. Restart the stack.

#. Ask the agent to list the open Phoebus displays. It calls
   ``phoebus_list_displays``, which lists the same displays as the ``curl``
   check.

If the agent's environment sets ``PHOEBUS_BRIDGE_URL``, that value wins over
both keys. Otherwise the keys decide.

Register your panels
====================

``phoebus_open_panel`` opens displays by name, and the names come from
``phoebus.panels``, one dotted line per display:

.. code-block:: yaml

   config:
     phoebus.panels.overview: /opt/displays/overview.bob

A value can be an absolute path, a ``file:`` URL, or a relative path. A
relative path is relative to the directory of the rendered ``config.yml``.
That is the project's ``build/`` directory, not the profile's. The path is
handed to the Phoebus process, so it must name a file that process can read.

The server refuses a name that is not registered, and the refusal lists the
names that are. ``phoebus_panel_lookup`` answers the other way round: given a
display path, it returns the name that opens it.

The names belong to the deployment. OSPREY ships none.

Several terminals, one Phoebus
==============================

Every tool that takes a ``display`` argument defaults to ``"active"``, the
display that has focus in Phoebus. Several web terminals on one bridge share
that focus. Two terminals working at once can then resolve ``"active"`` to
each other's display. To prevent this, require a handle:

.. code-block:: yaml

   config:
     phoebus.require_handle: true

With this key set, the server refuses ``"active"``. Callers pass the handle
that ``phoebus_open_panel`` returned (``"handle:d-3"``), or an explicit display
name from ``phoebus_list_displays``. The ``PHOEBUS_REQUIRE_HANDLE`` environment
variable outranks the key either way. It accepts
``1``/``true``/``yes``/``on`` or ``0``/``false``/``no``/``off``.

A multi-user web-terminal deployment sets ``PHOEBUS_REQUIRE_HANDLE=1`` on every
terminal whose project runs a Phoebus server. Elsewhere the switch is off by
default. Turn it on with ``phoebus.require_handle: true`` or
``PHOEBUS_REQUIRE_HANDLE=1``. To keep ``"active"`` in a multi-user deployment
too, set ``phoebus.require_handle: false``.

Data Browser plots and snapshots
================================

``phoebus_open_databrowser`` writes a ``.plt`` and opens it live. The plot's
archiver comes from ``PHOEBUS_ARCHIVER_URL`` if it is set, otherwise from
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

To add a second instance of the server, declare it with ``extends`` and give
it its own bridge URL in its ``env:``. ``phoebus2`` is only an example name:

.. code-block:: yaml

   config:
     claude_code.servers.phoebus2.extends: phoebus
     claude_code.servers.phoebus2.env.PHOEBUS_BRIDGE_URL: "${PHOEBUS2_BRIDGE_URL:-http://127.0.0.1:7980}"

The clone reads the same ``phoebus.*`` keys as the first instance, so only the
URL tells the two apart. Its ``phoebus_drive`` is offered and asked like the
original's.

What guards a drive
===================

The agent is offered ``phoebus_drive`` only where the build profile sets
``phoebus.agent_access: read_write``. Under the default, ``read``, the tool is
left out of the agent's tool list and permissions. The server also refuses a
call to it, and the refusal names the key.

Under ``read_write``, every drive:

- passes the write verdict (:ref:`write-safety-config`) and then the approval
  prompt;
- is refused while the control target is switched away from the deployment's
  baseline (:doc:`switch-control-target`);
- is audited as a tool call.

The key is all or nothing, because a panel runs whatever its widgets are wired
to. Finer control (per panel, widget, channel or mode) belongs to the EPICS
gateway or access security that the Phoebus product connects through.

A drive does not pass through a connector. The bridge performs it inside the
Phoebus process, through that product's own PV connections. Synthetic mode,
the default, runs the display's confirm dialogs and enable rules. Semantic mode
bypasses them. In either mode, OSPREY's channel-limits check does not apply to
a drive. Keep ``phoebus_drive`` for displays whose controls may be driven as
they stand.
