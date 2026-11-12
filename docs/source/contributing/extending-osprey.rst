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
the tests beside them are written to be read. So the fastest route to working
code is to hand both to a coding agent and let it draft against them.

Everything on this page assumes a development checkout --- see
:doc:`development-setup`.

Each seam's registration lives in your deployment's own registry module. The
framework finds that file through the ``registry_path`` key in its
``config.yml``. Set it in your profile's ``config:`` block, as
``registry_path: project/registry.py``. ``application.registry_path`` is
accepted as an alias, and the ``REGISTRY_PATH`` environment variable outranks
both, as :doc:`/reference/configuration/environment-variables` records. Without
one of them the framework loads only its own components, and nothing you
register is reachable.

.. _extending-connector:

Connector
---------

A connector is Osprey's single interface to one control system. Subclass
``ControlSystemConnector`` from ``osprey_connectors.control_system``
--- the source lives in
``packages/osprey-connectors/src/osprey_connectors/control_system/`` --- and
return the ``ChannelValue``, ``ChannelMetadata`` and ``ChannelWriteResult``
models it defines. A write's result carries the ``outcome`` your connector
determined. Use the shared ``values_match`` helper from the same module to
compare what you read back against what was sent. That way your connector
confirms a write exactly as every other one does. Register it with
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
Subclass ``ArchiverConnector`` from ``osprey_connectors.archiver``
(alongside the control-system connectors in
``packages/osprey-connectors/src/osprey_connectors/archiver/``) and register it
the same two ways, with ``ConnectorFactory.register_archiver`` or a
``ConnectorRegistration`` whose ``connector_type`` is ``archiver``. A deployment
can also select one without registering it, by setting ``archiver.type`` to its
dotted module path. Either way it is configured from ``archiver.settings``, which
the factory hands to ``connect()``. The return
shape is a contract of its own: one row per sample, historical enum values
as strings rather than indices. It is written down under "Archiver
Connectors" in :doc:`/reference/contracts/connectors`. Pinning test:
``tests/connectors/test_archiver_mock_default.py``, which holds the
``archiver.type`` key to a fail-closed default so a missing archiver never
looks like a working one.

.. _extending-mcp-server:

MCP server
----------

An MCP server is how new tools reach the agent. Adding an *external* server
needs no Python at all --- it is an entry in the ``mcp_servers:`` block of
``profile.yml``, covered in
:doc:`/how-to/agent-interfaces/add-mcp-server`. A *framework* server is the
developer seam: a package under ``src/osprey/mcp_server/`` holding a
module-level ``FastMCP`` instance and a ``create_server()`` factory in
``server.py``, and one tool per module under ``tools/``. The server also needs
an ``osprey.registry.mcp.ServerDefinition`` added to ``FRAMEWORK_SERVERS`` in
``src/osprey/registry/mcp.py``. The controls server
(``osprey.mcp_server.control_system``) is the canonical example to copy.
Pinning test: ``tests/registry/test_mcp.py``, which resolves those definitions
into the generated agent configuration --- permissions, hooks and environment
included.

.. _extending-chat-bridge:

Chat bridge
-----------

Nextcloud Talk, Google Chat, Microsoft Teams and Slack ship with Osprey;
Mattermost or plain email would each be a new bridge. Almost none of a bridge is per-service ---
deduplication, history, retries, honest give-up and crash recovery live in the
engine under ``src/osprey/bridges/core/``. What you write is an arrival loop
and one class, ``osprey.bridges.core.ports.ChannelOps``. Read that module's
docstring first: it carries the failure contract member by member, and getting
it wrong is the one mistake that leaves a bridge looking healthy while losing
answers. A bridge that can list who is in a conversation also implements the
optional ``osprey.bridges.core.ports.RoomRoster``. The engine then tells the
agent who is in the room. The bridge renders the agent's ``<@ID>`` mentions
in its own chat system's syntax. Copy ``src/osprey/bridges/google_chat/``,
``src/osprey/bridges/nextcloud_talk/``, ``src/osprey/bridges/teams/`` or
``src/osprey/bridges/slack/``, and
expect to touch the build-profile
and injector modules in ``src/osprey/cli/`` so a profile can switch the new
service on. Pinning test: ``tests/bridges/test_ports.py``. Deployers start
from :doc:`/how-to/agent-interfaces/chat-bridges/index`.

.. _extending-trigger-source:

Trigger source
--------------

A trigger source is what wakes the event dispatcher, as a webhook call, a
fixed interval or an EPICS channel crossing does. A message queue or a file
drop would each be a new one. To write a trigger source:

#. Write a class matching ``osprey.dispatch.sources.base.TriggerSource``.
#. Give it a ``source_type`` class attribute.
#. Implement ``register_routes``. It runs before the dispatcher's app is built,
   and does something only for a source that serves HTTP routes.
#. Implement ``start`` and ``stop``, which run with the event loop going.
   ``start`` receives the triggers that name this source and a fire callback.
#. Read each trigger's own ``source_config``. For every event the source
   detects, await the callback with the trigger and a payload mapping.
#. Stamp the event's instant as a timezone-aware ``datetime``. The dispatcher
   stores it as given and shows it to the agent in the facility zone. A time
   carried as text reaches the agent unchanged.
#. Handle the callback's result. It returns the queued run's id, or ``None``
   when the trigger is disabled. It raises
   ``osprey.dispatch.pool.QueueFullError`` when the queue is full. What happens
   then is the source's decision; every shipped source logs the event and
   drops it.
#. Declare the class as an entry point (below) and install the package where
   the dispatcher runs.

A fire from a source is attributed to nobody.

This seam is the exception to the registry-module paragraph that opens this
page: sources are not registered in a registry module. They are found through
a Python package entry point in
``[project.entry-points."osprey.trigger_sources"]``, the group under which
OSPREY's own ``pyproject.toml`` declares its three sources. The entry point's name is
the ``source:`` value a trigger writes.
``osprey.dispatch.source_registry.SourceRegistry`` loads the group when the
dispatcher starts. The dispatcher does not start when an entry point's
class lacks the three methods, or when one name is claimed by two different
classes.

The package has to be installed wherever the dispatcher runs. A
dispatcher started with ``python -m osprey.dispatch`` finds it in that
environment. The image OSPREY builds for the containerized dispatcher
installs OSPREY alone. So a container deployment names an image that carries
the package, through ``services.event_dispatcher.image``
(:ref:`deployment-image-overrides`). Copy
``src/osprey/dispatch/sources/cron.py`` for a source with no routes, or
``src/osprey/dispatch/sources/webhook.py`` for one that serves them. Pinning
test: ``tests/dispatch/test_source_registry.py``; deployer view:
:ref:`event-dispatch-trigger-sources`.

.. _extending-ariel:

ARIEL
-----

ARIEL is the interface to logbook data, and it has three seams:

- a **facility adapter**
  (``osprey.services.ariel_search.ingestion.base.FacilityAdapter``) that reads
  your logbook and normalises it
- an **enhancement module**
  (``osprey.services.ariel_search.enhancement.base.BaseEnhancementModule``)
  that enriches stored entries
- a **search module** that exposes a new way to query them

All three are registered through
``osprey.registry.helpers.extend_framework_registry`` with the matching
``osprey.registry.base.ArielIngestionAdapterRegistration``,
``ArielEnhancementModuleRegistration`` or ``ArielSearchModuleRegistration``,
then named from ``config.yml``. Reasoning *over* ARIEL's results is a skill in
your build profile, not an ARIEL extension. Pinning test:
``tests/registry/test_ariel_module_registrations.py``; deployer view:
:doc:`/how-to/ariel/data-ingestion`.

A facility adapter may also declare the fields an author fills in when writing
an entry. It does so through two optional hooks on the same class:
``get_entry_field_descriptors`` returns ``ParameterDescriptor`` declarations
and ``get_entry_field_options`` lists a ``dynamic_select`` field's choices.
Both default to declaring nothing, which keeps the built-in entry form. The
declaration checks, the value coercion and the reserved names
(``osprey.services.ariel_search.entry_fields.RESERVED_FIELD_NAMES``) live in
``osprey.services.ariel_search.entry_fields``. Pinning tests:
``tests/services/ariel_search/test_entry_fields.py`` and
``tests/integration/test_ariel_entry_fields_roundtrip.py``; author view: the
Entry Fields section of :doc:`/how-to/ariel/data-ingestion`.

.. _extending-health-plugin:

Health plugin
-------------

Most health checks are declarative YAML. A plugin is for the ones that need
real Python --- querying a facility service, computing a derived state. To
write a health plugin:

#. Write a module that exposes ``get_health_categories()``. It returns a
   mapping of category name to a callable (sync or async) that produces a list
   of ``osprey.health.models.CheckResult``.
#. Name the module under ``health.plugins``, either as a dotted path to an
   importable module (``my_package.health_checks``) or as a path to a ``.py``
   file (``./health/facility_checks.py``).

A relative file path is resolved against the project root, the same anchor
``data/`` and ``plans/`` use. So a deployment can keep its checks beside its
profile without packaging them or setting ``PYTHONPATH``. A file loaded that
way is given a synthetic module name derived from its path. It is
**re-executed** every time the health config is reloaded, so it must not keep
state at module level.

``osprey.health.plugins.load_plugin_categories`` loads the plugin, and does so
fail-safe. A plugin that will not import, returns the wrong type, or collides
with an existing category name becomes one ``error`` row rather than a crashed
suite.

A category callable normally takes no arguments. An ``async def`` one
may instead declare a ``runtime`` parameter and receive the suite's shared
``osprey.health.runtime.HealthRuntime``. Then ``await runtime.get_connector()``
hands back the one control-system connector the whole run shares, instead of
opening a second Channel Access context. The runtime is valid only for that
call. A sync callable cannot take it: it runs on a worker thread with no event
loop. A sync callable that declares ``runtime`` is refused with an ``error``
row saying to make it ``async def``. Pinning test:
``tests/health/test_plugins.py``. Config side:
:doc:`/how-to/health-and-monitoring/configure-health-checks`.

.. _extending-panel:

Panel
-----

A panel is a tab in the Web Terminal. The supported route is the guided skill,
``/osprey:panel``, whose source is ``plugins/osprey/skills/panel/``. The skill
walks a coding agent through a bundle that already meets every rule: a
directory under the project's ``panels/`` holding a ``manifest.json`` and an
entry HTML file. Discovery is off until a deployer sets
``web.allow_runtime_panels``, and it is fail-closed, so a malformed bundle is
skipped rather than served. Pinning test:
``tests/interfaces/web_terminal/test_panel_discovery.py``. What a panel may
and may not do at runtime --- notably the proxy's header stripping, which
rules out backends that authenticate their own callers --- is on
:doc:`/how-to/web-terminal/panels`.

.. _extending-login-service:

Login service
-------------

The login service is a closed seam, not a plugin point. It serves exactly the
methods in ``osprey.services.auth_sidecar.methods.SUPPORTED_METHODS``, and a
facility registers no other. A site meets it at two places. One is an OIDC
issuer: a sign-in that is not OIDC is brokered to one. The other is the answer
nginx asks of it, ``GET /verify`` with the four headers named in
``osprey.services.auth_sidecar.identity_headers``, which every terminal decodes
through ``osprey.interfaces.common_middleware.forwarded_identity``. A new login
method is a framework change: it joins ``SUPPORTED_METHODS`` and the render
postures in ``osprey.deployment.web_terminals.render.SUPPORTED_AUTH_METHODS``
together, with an audit category of its own. Pinning tests:
``tests/services/auth_sidecar/test_methods_parity.py`` for the method set,
``tests/services/auth_sidecar/test_verify.py`` for the answer, and
``tests/deployment/web_terminals/test_nginx_auth_surface.py`` for nginx's
side. Deployer view: :doc:`/how-to/web-terminal/multi-user/login`.

.. _extending-lume-model:

LUME model
----------

The virtual accelerator serves one composite, and each physics model under it
is a ``lume.model.LUMEModel`` built by an *engine plug-in*. The container
always runs the shipped ``osprey.services.virtual_accelerator.entrypoint``; a
different physics backend is a different plug-in, never a different
entrypoint. A different pyAT deck needs no code at all (see "Bringing your own
model" on :doc:`/architecture/virtual-accelerator`). The shipped pyat plug-in,
``osprey.simulation.engines.pyat``, is the reference implementation; the
shipped injection plug-in, ``osprey.simulation.engines.injection``, is the
reference for one that reads no deck and is time-driven.

What a plug-in is given and asked for:

- **Registration.** A plug-in is a module registered under the
  ``osprey.simulation.engines`` entry-point group in its package's own
  ``pyproject.toml``, under the name a model record's ``engine`` gives. The
  composite looks it up with ``metadata.entry_points(group="osprey.simulation.engines",
  name=...)``; an engine no package registers fails that model.
- **Build-time contract.** ``osprey build`` calls four pure functions of the
  module, none of which builds a model:

  - ``locate`` gives an element's place along the deck.
  - ``prepare`` checks the deck against the model's settings.
  - ``start_values`` derives each wired channel's operating point from the
    deck. A plug-in setting ``deckless_start_values = True`` is called with
    ``deck`` ``None`` for a model that names no deck.
  - ``describe(record)`` says what one wiring record is.

  ``describe`` returns ``{engine_role, plane, refresh}``:

  - ``engine_role`` is ``setpoint``, ``readback``, ``monitor`` or ``output``.
  - ``plane`` is ``x`` or ``y`` for a record that steers or reads one
    transverse plane, else ``None``.
  - ``refresh`` is ``pass`` for a value each write's solve updates,
    ``periodic`` for one only the periodic solve publishes, ``fast`` for one
    read at the fast period, or ``event`` for one that changes with each
    event the engine raises.

  It may add ``marks``, a list: ``command`` marks a write record whose write
  is a command. The words are the facility schema's, and a word it does not
  state stops the build. The build writes the description onto each wiring
  record, so no reader re-derives it from engine attributes.
- **Build.** The composite calls the module's
  ``build(model, wiring, deck, settings, active=...)`` once per served model.
  It returns the ``LUMEModel``. The arguments:

  - ``wiring`` is the model's wiring records from the simulator view's
    ``variables.json``.
  - ``deck`` is the path of the model's deck, ``decks/<model>.<its suffix>``,
    or ``None`` for a model that names none.
  - ``settings`` is the model record's settings.
  - ``active`` holds the active scenarios' setpoint values and fault seeds.

- **Set and get.** The composite reaches the model only through its ``set()``
  and ``get()``. Each variable is named for the channel address its wiring
  record claims, so a plug-in parses no channel names.
- **Report a failed build.** A ``build`` that raises leaves the model failed.
  Its ``<code>:SIM:<model>:STATUS`` channel carries the text the module's
  optional ``error_text(exc)`` gives, else the exception's own text. The
  rest of the namespace is still served.
- **The contract in code.** ``osprey.simulation.engines.EnginePlugin`` names
  those five functions, and ``isinstance(module, EnginePlugin)`` checks a
  module states them. ``osprey.simulation.engines.OPTIONAL_NAMES`` lists
  every optional name, and the protocol's docstring says what each gives and
  what a model whose engine lacks it gets.
  ``fault_variables`` is called as ``fault_variables(wiring, settings)``,
  with the model's ``settings`` second.
- **The time contract.** A model that changes between solves, as an
  injection chain does during a shot, is *time-driven*: its engine states the
  four optional functions ``osprey.simulation.engines.TIME_CONTRACT`` names.
  ``advance(model, t_s)`` applies every step due by epoch second ``t_s``;
  ``next_deadline(model)`` gives the next due step's epoch second, or
  ``None``; ``emitted(model)`` gives the ``event`` addresses fired since the
  last call, cleared; ``client_write(model, values, t_s)`` tells the engine a
  client wrote these setpoints. The model has no thread or clock of its own,
  and the same calls with the same clock readings always yield the same
  values. Its ``set`` only stores values, because restores and scenario
  writes reach it too; what a client write triggers lives behind
  ``client_write`` alone. ``build`` receives no clock, so the composite calls
  ``advance`` right after a build, a rebuild or a reset. Only the served
  venue drives the contract; in process the model serves its start values.
- **Model inputs.** An engine whose models read other models' records states
  two optional functions. ``describe_input(input_record)`` is called at
  facility build with one input's ``{address, engine}`` record; raising
  ``ValueError`` refuses it, and the build stops ``engine-invalid`` with its
  text, naming the input. ``take_inputs(model, inputs, t_s, *, baseline)``
  gets one ``(input_record, value)`` pair per declared input whose producer
  has not failed, the value the producer's current plain value. It only
  stores, like ``set``; what a new value starts happens in the ``advance``
  that follows. ``baseline=True`` marks a starting point: compare nothing
  against what was held. The ``engine`` block travelling with each pair is
  how a model tells its inputs apart; ``build`` is unchanged. A model that
  declares inputs over an engine without ``take_inputs`` is refused at build.
  ``osprey.simulation.engines.pyat`` states both, and the time contract, for
  a turn-by-turn ring (:ref:`va-turn-by-turn`).
- **The response kind.** An optional ``response_matrix`` is called with the
  keyword ``kind``, ``"orbit"`` or ``"trajectory"``, and names each corrector
  without a column in ``unsolved`` with its text. One whose signature takes
  no ``kind`` is called without it for ``"orbit"``; any other kind is refused
  as a request, naming the engine, without calling it. Its answer may state
  ``row_positions`` and ``column_positions``, one lattice element index per
  row and per column, counted for ``"trajectory"`` from the injection
  element. An answer without them is served with every ``position`` ``null``;
  positions that do not line up with the rows or columns fail the solve.

What the container keeps the same for every engine:

- **The environment.** ``VA_INSTANCE`` is required; ``VA_DATA_DIR`` (default
  ``/data``) is the data root the simulator view sits under; ``VA_STATE_DIR``,
  ``VA_POLL_INTERVAL_S``, ``VA_FAST_INTERVAL_S`` and
  ``VA_MODEL_WRITE_TOKEN`` are read as the
  entrypoint describes them. Channel Access is served on
  ``EPICS_CAS_SERVER_PORT``, which the image's command exports from
  ``EPICS_CA_SERVER_PORT``.
- **Clamping.** The runner clamps every written setpoint into the
  ``value_range`` the view's limits records give it before anything else
  happens to the value. The connector's own ``limits_checking`` is separate
  and unchanged.
- **The ready line.** Once the first publishing pass has published, the
  entrypoint prints one line starting with
  ``osprey.services.virtual_accelerator.entrypoint.READY_MARKER``
  (``virtual accelerator IOC serving PVs``), followed by ``: <N> channels``.
  The image boot check and the container test fixtures wait on that line and
  read the count out of it. A first pass that fails exits non-zero without
  it.
- **SIGTERM.** The image's command ``exec``\ s Python, so the entrypoint is
  the container's first process. It receives ``docker stop``'s SIGTERM
  itself. It handles SIGTERM as it handles SIGINT and leaves through the
  runner's own exit.
- **One image for every instance.** With a live stand-in, both containers run
  the same image and the same entrypoint; ``VA_INSTANCE`` tells them apart.

``tests/va/test_entrypoint.py`` pins what the shipped entrypoint reads and
refuses. How to deploy a plug-in is in :ref:`va-serving-your-own-model`.

.. _extending-agent-harness:

Agent harness
-------------

The agent harness is not a seam. Osprey ships one harness, Claude Code, and
there is no configuration key or base class for choosing another. What the
codebase keeps instead is a boundary: the code specific to Claude Code lives in
one package, ``src/osprey/agent_runner/``, and the rest of the framework reaches
the agent through it. That package owns:

- the Agent SDK calls. ``osprey.agent_runner.run_query`` and
  ``osprey.agent_runner.stream_query`` run one prompt,
  ``osprey.agent_runner.agent_session`` holds a conversation of many turns, and
  each hands its caller the plain event records of
  ``osprey.agent_runner.events`` rather than SDK types;
- the command line that starts the CLI: its program name, its flags and the
  resolution of its bare name (``src/osprey/agent_runner/launcher.py``);
- the provider and model environment the agent is launched with
  (``src/osprey/agent_runner/provider_env.py``);
- the first-run state the CLI expects in ``.claude.json``, and the suffix that
  names each user's state volume (``src/osprey/agent_runner/claude_state.py``);
- the catalog of files a build writes for the CLI
  (``src/osprey/agent_runner/build_artifacts/``);
- the built-in tool names every deny list is made of
  (``src/osprey/agent_runner/tool_names.py``), and the permission hook and
  callback of a dispatched run (``src/osprey/agent_runner/tool_policy.py``).

The templates a build renders for the CLI --- settings, hooks, rules, agents and
skills --- are Claude Code's own file formats and stay with the other templates,
in ``src/osprey/templates/claude_code/``. The deny list in the rendered settings
comes from ``src/osprey/agent_runner/tool_names.py``.

A lint holds the boundary. Ruff's banned-api rule (``TID251``, configured in
``pyproject.toml``) refuses an import of ``claude_agent_sdk`` anywhere in
``src/`` or ``packages/`` outside the adapter. The unit suite refuses the
same import, including the dynamic imports ruff cannot see. No module is exempt
by name. Tests and developer scripts sit outside the shipped trees and drive the
SDK directly. New code that needs the agent calls the adapter; a Claude-specific
need the adapter does not meet yet is added to the adapter, not written inline
where it is needed. Pinning test: ``tests/test_harness_fence.py``.
