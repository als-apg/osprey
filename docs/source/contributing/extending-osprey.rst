.. _contributing-extending-osprey:

================
Extending Osprey
================

A facility can add its own control system, its own logbook, its own health
checks and its own panels without forking Osprey. Each of those is a **seam**:
a base class or a registry entry the framework already knows how to load, plus
a config key that names your implementation. Deployers configure; developers
extend.

This page is the map, not the tutorial. For each seam it names the class or
registry to work against, one test that pins that seam's behaviour, and the
how-to a deployer uses to switch your implementation on. The base classes and
the tests beside them are written to be read, so the fastest route to working
code is to hand both to a coding agent and let it draft against them.

Everything on this page assumes a development checkout --- see
:doc:`development-setup`.

.. _extending-connector:

Connector
---------

A connector is Osprey's single interface to one control system. Subclass
``ControlSystemConnector`` from ``osprey.connectors.control_system.base``
--- the source lives in
``packages/osprey-connectors/src/osprey_connectors/control_system/`` --- and
return the ``ChannelValue``, ``ChannelMetadata``, ``ChannelWriteResult`` and
``WriteVerification`` models it defines. Register it with
``ConnectorFactory.register_control_system``, or ship it as an
``osprey.registry.base.ConnectorRegistration`` passed to
``osprey.registry.helpers.extend_framework_registry``. Pinning test:
``tests/connectors/test_connector_factory.py``. What every connector owes its
callers is :doc:`/reference/contracts/connectors`; deployers point a project
at yours in :doc:`/how-to/control-systems/use-connectors`.

.. _extending-archiver:

Archiver
--------

An archiver connector answers questions about the past instead of the present.
Subclass ``ArchiverConnector`` from ``osprey.connectors.archiver.base``
(alongside the control-system connectors in
``packages/osprey-connectors/src/osprey_connectors/archiver/``) and register it
the same two ways, with ``ConnectorFactory.register_archiver`` or a
``ConnectorRegistration`` whose ``connector_type`` is ``archiver``. The return
shape is a contract of its own --- one row per sample, historical enum values
as strings rather than indices --- and it is written down under "Archiver
Connectors" in :doc:`/reference/contracts/connectors`. Pinning test:
``tests/connectors/test_archiver_mock_default.py``, which holds the
``archiver.type`` key to a fail-closed default so a missing archiver never
looks like a working one.

.. _extending-mcp-server:

MCP server
----------

An MCP server is how new tools reach the agent. Adding an *external* server
needs no Python at all --- it is a ``config.yml`` block, covered in
:doc:`/how-to/agent-interfaces/add-mcp-server`. A *framework* server is the
developer seam: a package under ``src/osprey/mcp_server/`` holding a
module-level ``FastMCP`` instance and a ``create_server()`` factory in
``server.py``, one tool per module under ``tools/``, and an
``osprey.registry.mcp.ServerDefinition`` added to ``FRAMEWORK_SERVERS`` in
``src/osprey/registry/mcp.py``. The controls server
(``osprey.mcp_server.control_system``) is the canonical example to copy.
Pinning test: ``tests/registry/test_mcp.py``, which resolves those definitions
into the generated agent configuration --- permissions, hooks and environment
included.

.. _extending-chat-bridge:

Chat bridge
-----------

Nextcloud Talk and Google Chat ship with Osprey; Slack, Mattermost or plain
email would each be a new bridge. Almost none of a bridge is per-service ---
deduplication, history, retries, honest give-up and crash recovery live in the
engine under ``src/osprey/bridges/core/``. What you write is an arrival loop
and one class, ``osprey.bridges.core.ports.ChannelOps``. Read that module's
docstring first: it carries the failure contract member by member, and getting
it wrong is the one mistake that leaves a bridge looking healthy while losing
answers. Copy ``src/osprey/bridges/google_chat/`` or
``src/osprey/bridges/nextcloud_talk/``, and expect to touch the build-profile
and injector modules in ``src/osprey/cli/`` so a profile can switch the new
service on. Pinning test: ``tests/bridges/test_ports.py``. Deployers start
from :doc:`/how-to/agent-interfaces/chat-bridges/index`.

.. _extending-ariel:

ARIEL
-----

ARIEL is the interface to logbook data, and it has three seams: a **facility
adapter** (``osprey.services.ariel_search.ingestion.base.FacilityAdapter``)
that reads your logbook and normalises it, an **enhancement module**
(``osprey.services.ariel_search.enhancement.base.BaseEnhancementModule``) that
enriches stored entries, and a **search module** that exposes a new way to
query them. All three are registered through
``osprey.registry.helpers.extend_framework_registry`` with the matching
``osprey.registry.base.ArielIngestionAdapterRegistration``,
``ArielEnhancementModuleRegistration`` or ``ArielSearchModuleRegistration``,
then named from ``config.yml``. Reasoning *over* ARIEL's results is a skill in
your build profile, not an ARIEL extension. Pinning test:
``tests/registry/test_ariel_module_registrations.py``; deployer view:
:doc:`/how-to/ariel/data-ingestion`.

.. _extending-health-plugin:

Health plugin
-------------

Most health checks are declarative YAML. A plugin is for the ones that need
real Python --- querying a facility service, computing a derived state. Write
a module that exposes ``get_health_categories()``, returning a mapping of
category name to a no-argument callable (sync or async) that produces a list of
``osprey.health.models.CheckResult``, then name it under ``health.plugins`` —
either as a dotted path to an importable module (``my_package.health_checks``)
or as a path to a ``.py`` file (``./health/facility_checks.py``). A relative
file path is resolved against the project root, the same anchor ``data/`` and
``plans/`` use, so a deployment can keep its checks beside its profile without
packaging them or setting ``PYTHONPATH``. A file loaded that way is given a
synthetic module name derived from its path and is **re-executed** every time
the health config is reloaded, so it must not keep state at module level.
``osprey.health.plugins.load_plugin_categories``
loads it, and does so fail-safe: a plugin that will not import, returns the
wrong type, or collides with an existing category name becomes one ``error``
row rather than a crashed suite. Pinning test:
``tests/health/test_plugins.py``. Config side:
:doc:`/how-to/health-and-monitoring/configure-health-checks`.

.. _extending-panel:

Panel
-----

A panel is a tab in the Web Terminal. The supported route is the guided skill
--- ``osprey skills install creating-an-osprey-panel``, whose source is
``src/osprey/templates/skills/creating-an-osprey-panel/`` --- which walks a
coding agent through a bundle that already meets every rule: a directory under
the project's ``panels/`` holding a ``manifest.json`` and an entry HTML file.
Discovery is off until a deployer sets ``web.allow_runtime_panels``, and it is
fail-closed, so a malformed bundle is skipped rather than served. Pinning test:
``tests/interfaces/web_terminal/test_panel_discovery.py``. What a panel may
and may not do at runtime --- notably the proxy's header stripping, which rules
out backends that authenticate their own callers --- is on
:doc:`/how-to/web-terminal/panels`.

.. _extending-lume-model:

LUME model
----------

The virtual accelerator serves whatever physics you hand it, as long as that
physics is a ``lume.model.LUMEModel``. The floor is
``osprey.services.virtual_accelerator.serving.model_stub.NullModel``, which
serves a channel list and no physics at all; at the other end,
``src/osprey/services/virtual_accelerator/serving/write_path.py:182`` shows a
shipped model wrapped so that setpoint writes carry a calibration and push
recomputed readings back onto their channels. The seam is guarded in both
directions: ``tests/va/test_facility_seam.py`` pins that a facility without a
lattice boots with no accelerator-physics imports on the path at all, and
``tests/va/test_pyat_ring_model.py`` covers the shipped ring model. Deployers
configure the result in
:doc:`/how-to/control-systems/use-virtual-accelerator`.
