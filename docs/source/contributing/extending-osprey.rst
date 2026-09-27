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

Each seam's registration lives in your deployment's own registry module, and
the framework finds that file through the ``registry_path`` key in its
``config.yml`` --- set it in your profile's ``config:`` block, as
``registry_path: project/registry.py``. ``application.registry_path`` is
accepted as an alias, and the ``REGISTRY_PATH`` environment variable outranks
both, as :doc:`/reference/configuration/environment-variables` records. Without
one of them the framework loads only its own components, and nothing you
register is reachable.

.. _extending-connector:

Connector
---------

A connector is Osprey's single interface to one control system. Subclass
``ControlSystemConnector`` from ``osprey.connectors.control_system.base``
--- the source lives in
``packages/osprey-connectors/src/osprey_connectors/control_system/`` --- and
return the ``ChannelValue``, ``ChannelMetadata`` and ``ChannelWriteResult``
models it defines. A write's result carries the ``outcome`` your connector
determined --- use the shared ``values_match`` helper from the same module to
compare what you read back against what was sent, so your connector confirms a
write exactly as every other one does. Register it with
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
``ConnectorRegistration`` whose ``connector_type`` is ``archiver``. A deployment
can also select one without registering it, by setting ``archiver.type`` to its
dotted module path. Either way it is configured from ``archiver.settings``, which
the factory hands to ``connect()``. The return
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

Nextcloud Talk, Google Chat and Microsoft Teams ship with Osprey; Slack,
Mattermost or plain email would each be a new bridge. Almost none of a bridge is per-service ---
deduplication, history, retries, honest give-up and crash recovery live in the
engine under ``src/osprey/bridges/core/``. What you write is an arrival loop
and one class, ``osprey.bridges.core.ports.ChannelOps``. Read that module's
docstring first: it carries the failure contract member by member, and getting
it wrong is the one mistake that leaves a bridge looking healthy while losing
answers. A bridge that can list who is in a conversation also implements the
optional ``osprey.bridges.core.ports.RoomRoster``; the engine then tells the
agent who is in the room, and the bridge renders the agent's ``<@ID>`` mentions
in its own chat system's syntax. Copy ``src/osprey/bridges/google_chat/``,
``src/osprey/bridges/nextcloud_talk/`` or ``src/osprey/bridges/teams/``, and
expect to touch the build-profile
and injector modules in ``src/osprey/cli/`` so a profile can switch the new
service on. Pinning test: ``tests/bridges/test_ports.py``. Deployers start
from :doc:`/how-to/agent-interfaces/chat-bridges/index`.

.. _extending-trigger-source:

Trigger source
--------------

A trigger source is what wakes the event dispatcher, as a webhook call, a
fixed interval or an EPICS channel crossing does; a message queue or a file
drop would each be a new one. Write a class matching
``osprey.dispatch.sources.base.TriggerSource``: a ``source_type`` class
attribute, ``register_routes``, which runs before the dispatcher's app is built
and does something only for a source that serves HTTP routes, and ``start`` and
``stop``, which run with the event loop going. ``start`` receives the triggers
that name this source and a fire callback. The source reads each trigger's own
``source_config`` and awaits the callback with the trigger and a payload
mapping for every event it detects. The callback returns the queued run's id,
returns ``None`` when the trigger is disabled, and raises
``osprey.dispatch.pool.QueueFullError`` when the queue is full; what happens
then is the source's decision, and every shipped source logs the event and
drops it. A fire from a source is attributed to nobody.

This seam is the exception to the registry-module paragraph that opens this
page: sources are not registered in a registry module. They are found through a Python package
entry point, in the group OSPREY declares its own three under ---
``[project.entry-points."osprey.trigger_sources"]`` in ``pyproject.toml`` ---
and the entry point's name is the ``source:`` value a trigger writes.
``osprey.dispatch.source_registry.SourceRegistry`` loads the group when the
dispatcher starts, and the dispatcher does not start when an entry point's
class lacks the three methods, or when one name is claimed by two different
classes. The package has to be installed wherever the dispatcher runs: a
dispatcher started with ``python -m osprey.dispatch`` finds it in that
environment, while the image OSPREY builds for the containerized dispatcher
installs OSPREY alone, so a container deployment names an image that carries
the package through ``services.event_dispatcher.image``
(:ref:`deployment-image-overrides`). Copy
``src/osprey/dispatch/sources/cron.py`` for a source with no routes, or
``src/osprey/dispatch/sources/webhook.py`` for one that serves them. Pinning
test: ``tests/dispatch/test_source_registry.py``; deployer view:
:ref:`event-dispatch-trigger-sources`.

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
real Python --- querying a facility service, computing a derived state. Write a
module that exposes ``get_health_categories()``, returning a mapping of
category name to a callable (sync or async) that produces a list of
``osprey.health.models.CheckResult``, then name it under ``health.plugins`` —
either as a dotted path to an importable module (``my_package.health_checks``)
or as a path to a ``.py`` file (``./health/facility_checks.py``). A relative
file path is resolved against the project root, the same anchor ``data/`` and
``plans/`` use, so a deployment can keep its checks beside its profile without
packaging them or setting ``PYTHONPATH``. A file loaded that way is given a
synthetic module name derived from its path and is **re-executed** every time
the health config is reloaded, so it must not keep state at module level.
``osprey.health.plugins.load_plugin_categories`` loads it, and does so
fail-safe: a plugin that will not import, returns the wrong type, or collides
with an existing category name becomes one ``error`` row rather than a crashed
suite. A category callable normally takes no arguments. An ``async def`` one
may instead declare a ``runtime`` parameter and receive the suite's shared
``osprey.health.runtime.HealthRuntime``, so ``await runtime.get_connector()``
hands back the one control-system connector the whole run shares instead of
opening a second Channel Access context. The runtime is valid only for that
call. A sync callable cannot take it — it runs on a worker thread with no event
loop — and one that declares ``runtime`` is refused with an ``error`` row
saying to make it ``async def``. Pinning test:
``tests/health/test_plugins.py``. Config side:
:doc:`/how-to/health-and-monitoring/configure-health-checks`.

.. _extending-panel:

Panel
-----

A panel is a tab in the Web Terminal. The supported route is the guided skill
--- ``/osprey:panel``, whose source is ``plugins/osprey/skills/panel/`` ---
which walks a coding agent through a bundle that already meets every rule: a
directory under the project's ``panels/`` holding a ``manifest.json`` and an
entry HTML file. Discovery is off until a deployer sets
``web.allow_runtime_panels``, and it is fail-closed, so a malformed bundle is
skipped rather than served. Pinning test:
``tests/interfaces/web_terminal/test_panel_discovery.py``. What a panel may
and may not do at runtime --- notably the proxy's header stripping, which
rules out backends that authenticate their own callers --- is on
:doc:`/how-to/web-terminal/panels`.

.. _extending-lume-model:

LUME model
----------

The virtual accelerator serves one ``lume.model.LUMEModel``, chosen by the
module the container runs. The shipped
``osprey.services.virtual_accelerator.entrypoint`` builds
``osprey.services.virtual_accelerator.model.pyat.PyATRingModel`` over the
served tree's lattice, or the floor,
``osprey.services.virtual_accelerator.serving.model_stub.NullModel``, which
serves a channel list and no physics. A different pyAT deck needs no code (see
"Bringing your own model" on :doc:`/architecture/virtual-accelerator`). A
different backend is a replacement entrypoint module named in
``VA_ENTRYPOINT_MODULE``. For a shipped model wrapped so that setpoint writes
carry a calibration and push recomputed readings back onto their channels, see
``osprey.services.virtual_accelerator.serving.write_path.SetpointRoutedModel``.

A replacement entrypoint honours the same contract as the shipped one, which
is its reference implementation:

- **Runnable and importable.** The container runs it as
  ``python -u -m <module>``, so it is importable in the image and has a
  ``__main__`` guard, or is a package with a ``__main__.py``. Its image is one
  the facility builds on top of OSPREY's, because the stock image installs
  OSPREY and nothing else.
- **The same inputs.** The data directory is ``VA_DATA_DIR``, default
  ``/data/simulation``. The channel manifest ``VA_CHANNELS_FILE`` is required
  and resolved against that directory, and the manifest is the served
  namespace. ``VA_LATTICE`` and ``VA_STATE_DIR`` are set too. Channel Access is
  served on ``EPICS_CAS_SERVER_PORT``, which the image's command exports from
  ``EPICS_CA_SERVER_PORT``, and pvAccess on ``EPICS_PVAS_SERVER_PORT``. The
  deployment's health check is a TCP connect to the Channel Access port and
  nothing more.
- **Clamp writes to the drive bands.** The shipped entrypoint reads
  ``channel_limits.json`` from the data directory and hands the bands to the
  runner, which clamps every written value into its band before anything else
  happens to it (the clamp is
  ``osprey.services.virtual_accelerator.serving.write_path.clamp_into``). A
  replacement that builds its own server without them serves setpoints with no
  band on the IOC side. The connector's own ``limits_checking`` is separate and
  unchanged.
- **Announce readiness.** Once every boot value is on the wire, print one line
  starting with ``osprey.services.virtual_accelerator.entrypoint.READY_MARKER``
  (``virtual accelerator IOC serving PVs``), followed by ``: <N> channels``.
  The image boot check and the container test fixtures wait on that line and
  read the count out of it.
- **Leave on SIGTERM.** The image's command ``exec``\ s Python, so the module
  is the container's first process and receives ``docker stop``'s SIGTERM
  itself. A first process with no handler for it ignores it and is killed when
  the stop timeout runs out. The shipped entrypoint installs handlers for
  SIGINT and SIGTERM once its servers are up, and they leave through the
  runner's own exit.
- **One module for every instance.** With a live stand-in, both containers
  run the same image and the same ``VA_ENTRYPOINT_MODULE``.

The seam is guarded in both directions: ``tests/va/test_facility_seam.py``
pins that a facility without a lattice boots with no accelerator-physics
imports on the path at all, and ``tests/va/test_pyat_ring_model.py`` covers
the shipped ring model. How to deploy a replacement is in
:ref:`va-serving-your-own-model`.
