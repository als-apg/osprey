.. _how-to-pv-directory:

=======================
How to Look Up Any PV
=======================

**All PVs** is a view of the Channels panel. It lists every PV the control
system announces, not only the curated channels the facility tree declares.
Each PV and IOC has its own page.

All PVs is not curated. Names, descriptions and properties are what the
control system's own directory records, as it records them. All PVs is also
read-only: neither the view, its pages nor its agent tools can write.

.. seealso::

   :doc:`use-channel-finder`
      The curated channels, and the panel All PVs is part of.

   :doc:`/reference/connectors-api/pv-directory`
      The record shapes and the per-target source table.


Turn it on
==========

All PVs exists only when ``config.yml`` has a ``pv_directory`` section. The
``demo`` preset ships one. Without the section, the panel, the agent and the
rendered files are unchanged. The keys are in :ref:`config-pv-directory`.

- **Simulator only.** A section with no ``type`` lists the virtual
  accelerator's channels and IOC health PVs. This is what ``demo`` ships.
- **Facility services.** ``type`` names the directory service, such as
  ``channelfinder``. The ``alarm_state``, ``alarm_log`` and ``put_log``
  sub-blocks each add their service.
- **Off.** ``osprey set config.pv_directory=null``, or ``pv_directory: null``
  under ``config:`` in a profile that extends ``demo``, removes the section.

On a multi-user host, the ``channel-finder`` shared login carries the
``pv-directory`` server. The ``logbook`` and ``knowledge`` logins switch the
server off (:doc:`web-terminal/multi-user/tiers`).


Search All PVs
==============

1. Open **Channels** from the **Apps** section of the sidebar.
2. Click **All PVs** in the view switch at the toolbar's left. The switch reads
   **Curated**, **All PVs** and **Feedback**.
3. Type words into the toolbar's search box and press Enter.

The search words carry across views in both directions. Under curated
results, a ``Not here? Search All PVs`` line leads here. A curated search with
no hits offers ``Search all N PVs for "<words>" →`` instead.

A search works the way ChannelFinder searches:

- Words match descriptions first, and names only when no description matched.
  Every word must match. On the *Simulator* they match in any order; on
  ChannelFinder, in the order given.
- A word holding ``*`` or ``?`` matches names as a glob. A leading ``=`` asks
  for the exact name.

The line above the rows says in plain words what was searched, with the hit
count. **show request** reveals the literal request that ran, with a copy
button. The sidebar filters the whole result by **Origin**, **IOC** and
**Record type**.

Each row carries one of two marks:

- ◆ **Curated**: the facility tree declares the PV.
- ◇ **Control system only**: the control system announces it, and the facility
  tree does not declare it.

A row also shows the description, current value, record type, IOC and status.
An inactive PV is dimmed. A value reads ``no answer`` when the PV did not
answer.

Tick rows to act on them; the selection survives new words, filters and
paging.

- **Export CSV** downloads the search result as CSV.
- **Plot** opens the selection in CANOPY's trace view, at most 12 PVs. It is
  shown only when CANOPY is enabled and serves its trace view.
- **Send to assistant** puts the names into the prompt. It is offered only
  inside the web terminal.


PV and IOC pages
================

Click a name to open its PV page. Its header shows the origin mark, the
Active or Inactive chip, and a link to the IOC. Each section below appears only
when its service is configured:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Section
     - What it shows, and where it comes from
   * - Directory
     - The PV's properties as the directory records them.
   * - Alarm state
     - The current alarm, latched or acknowledged, from the alarm server's feed.
       A feed that stopped updating marks the page ``alarm feed stale``.
   * - Alarm history
     - Up to 20 alarm events, from the alarm logger.
   * - Alarm configuration
     - Whether the PV is in the alarm tree, with its path, guidance and
       displays.
   * - Put history
     - Up to 20 recorded writes, from the put log.
   * - Logbook
     - Logbook entries that mention the PV, from ARIEL.
   * - Current value
     - The value, time and alarm, read from the control target.

A PV the directory does not list reads ``<name> is not listed in All PVs``,
with a button that searches for it.

Click an IOC to open its IOC page: host, ports, PV counts, engineer and
health PVs with their values. The **Alive** chip reads the ``HEARTBEAT`` PV
twice, 2 s apart; a value that advanced means alive.


The control target decides the source
======================================

Every section answers only from the machine the session is on, as the
control-target chip shows it. No section shows another machine's data under
this machine's PV.

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - Section
     - *Simulator*
     - *Live* and *Rehearsal*
   * - Directory
     - the build's simulator view
     - the ``type`` service, such as ChannelFinder
   * - Alarm state and configuration
     - the replay file
     - the alarm server's feed, or the alarm logger's last logged configuration
   * - Alarm history, put history, logbook
     - none
     - the configured service

*Rehearsal* reads the facility's services, because the stand-in rehearses
that machine. A section with no source for the current target reads ``Not
available for this target``.

Current values go through a separate read-only process on the control
target's read gateway. A connector block with only a write gateway has nothing
to read through, so every value reads unavailable.


The demo
========

The ``demo`` preset runs All PVs on the virtual accelerator:

- The directory is the simulator view: every channel, plus the read-only
  ``<code>:SIM:<model>:HEARTBEAT``, ``UPTIME`` and ``ENGINEER`` health PVs of
  each simulated IOC.
- The alarm feed is a packaged replay, ``demo:example-alarms``. It holds one
  latched, one acknowledged and one cleared alarm, each with its alarm
  configuration.

``SR:RF:CAVITY:01:TEMPERATURE:RB`` and ``SR:VAC:GAUGE:SR03:PRESSURE:RB`` carry
replayed alarms.


Agent tools
===========

The ``pv-directory`` MCP server gives the agent the same facts. Every tool is
a read and runs without an approval prompt:

- ``directory_search``, ``directory_properties``, ``pv_record`` and
  ``ioc_record`` are always present.
- ``alarm_state``, ``alarm_config``, ``alarm_history`` and ``put_history``
  are present only when their source is configured.

Each PV or IOC result carries ``links.panel``, its page in this panel. The
agent passes it to ``open_panel``. By default this opens the page in a new
split tile, or focuses it when it is already on screen.

The prompts tell the agent to check the curated channels first and to say
when a name is the control system's own. That is guidance, not a setting.
:doc:`/architecture/mcp-servers` lists the tools.
