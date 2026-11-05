MCP Servers
===========

OSPREY exposes control system operations, data retrieval, and workspace
management as tools through `FastMCP <https://github.com/jlowin/fastmcp>`_
servers. The OSPREY agent discovers servers from ``.mcp.json`` at startup and calls
tools via stdio JSON-RPC. This page lists every server that ships in-tree,
with every tool each one can register. The **10 core servers** are the ones a
deployment normally renders: ``control_system``, the four channel-finder
variants, ``workspace``, ``python_executor``, ``ariel``,
``osprey_facility_knowledge`` and ``graph``. Four more ship in-tree but are off
by default: ``phoebus``, ``optimizer``, ``bluesky`` and ``health``. A profile
enables one with ``claude_code.servers.<name>.enabled: true``. The
``control-assistant`` preset enables ``bluesky`` and ``health``; ``phoebus`` and
``optimizer`` stay off in every preset. Build profiles can inject additional servers beyond
these.

.. raw:: html
   :file: ../_diagrams/mcp-server-map.html

The four channel-finder variants count as four of the ten, but a deployment
serves exactly one of them — whichever ``channel_finder.pipeline_mode`` names —
under the single ``channel-finder`` name, which is why the map shows seven
running processes.


Control System
--------------

``control_system``
~~~~~~~~~~~~~~~~~~

Package: ``osprey.mcp_server.control_system``

Reads and writes control system channels and queries archiver history.
This is the primary server for live hardware interaction and includes
safety-limits enforcement on all write operations.

**Tools:**

- ``channel_read`` -- Read current values from one or more control system channels.
- ``channel_write`` -- Write values to one or more control system channels. Every write
  goes through the writes check and the limits check. The write switch's position then
  decides: at *off* the write is refused, at *asked* the operator approves it at a prompt,
  and at *allowed* it runs unprompted (see :ref:`architecture-safety-chain-order`).
- ``archiver_read`` -- Retrieve historical archived data for one or more channels over a
  time range. ``processing`` selects the per-bin aggregation (``raw``, ``mean``, ``min``,
  ``max``, ``median``, ``std``, ``count``) and ``bin_size`` sets the bin width in seconds;
  ``bin_size=0`` returns full resolution and is valid only with ``processing="raw"``.
  When ``bin_size`` is omitted the bin is derived from the time span so a continuously
  archived channel returns about ``archiver.auto_bin_points`` (default 10 000) points —
  1 s for anything under ~2.8 hours, ~53 minutes for a year — and the summary reports the
  bin that was used and whether it was requested or chosen automatically.
- ``channel_limits`` -- Query the channel write-safety limits database: which channels are
  write-gated and under what constraints (lookup, pattern match, summary). Not a channel
  roster -- an unlisted channel may still exist; use channel-finder to discover what channels
  exist.
- ``control_target`` -- Report which control system the deployment points at (``live``,
  ``va`` or ``standin``), each target's endpoints and reachability, and whether the session
  may move to it now. Read-only: it connects to nothing.
- ``control_target_set`` -- Point the deployment's control-system tools at another target.
  The target belongs to the deployment, so every session on it moves too (requires human
  approval).


Channel Finding
---------------

OSPREY provides four channel finder variants, each suited to different
facility data models and search strategies. A deployment picks one with
``channel_finder.pipeline_mode``, and whichever it picks is served under the one
``channel-finder`` name.

``channel_finder_hierarchical``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Package: ``osprey.mcp_server.channel_finder_hierarchical``

Finds channel addresses using a hierarchical drill-down through an
in-memory JSON database. Users navigate levels (system, subsystem,
device, signal) to narrow results.

**Tools:**

- ``get_options`` -- Get available options at a specific hierarchy level, filtered by prior selections.
- ``build_channels`` -- Build channel addresses from a set of hierarchy selections.
- ``view_examples`` -- View operator-verified search examples from prior sessions.

``channel_finder_in_context``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Package: ``osprey.mcp_server.channel_finder_in_context``

Answers natural-language questions about channels by handing the full
channel database to an inner LLM as context. Designed for databases small
enough to fit in a single context window.

**Tools:**

- ``ask_channels`` -- Answer a natural-language question about channels; an inner LLM call sees the full channel database as context and returns the answer text together with tokenizer-estimated input/output token counts (used by the benchmark harness for cost accounting).

``channel_finder_middle_layer``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Package: ``osprey.mcp_server.channel_finder_middle_layer``

Exposes the MATLAB Middle Layer (MML) channel database as MCP tools.
Organized by system, family, and field with support for common-name
lookups and validation.

**Tools:**

- ``list_systems`` -- List all systems in the channel database with descriptions.
- ``list_families`` -- List device families in a system with descriptions.
- ``list_channels`` -- Get channel names for a system/family/field path.
- ``get_common_names`` -- Get common (friendly) names for devices in a family.
- ``inspect_fields`` -- Inspect the field structure of a device family.
- ``validate`` -- Validate that channel names exist in the database.
- ``statistics`` -- Get database statistics (total channels, systems, families).
- ``run_sql`` -- Run a read-only SQL query directly against the channel finder DuckDB database (``channels``, ``systems``, ``families`` tables). Answers are bounded by ``channel_finder.query_max_rows``; a truncated reply says so and names the key, so a partial list is never presented as complete.

``channel_finder_graph``
~~~~~~~~~~~~~~~~~~~~~~~~

Package: ``osprey.mcp_server.channel_finder_graph``

Finds channels by searching the facility knowledge graph. Instead of drilling
through a database file, the agent looks a phrase up in the search index the
build derives from the corpus, or writes read-only Cypher against the seeded
``graphdb`` store, so a question can start from a description, an alternate name
or a system just as easily as from an address. Selected with
``channel_finder.pipeline_mode: graph``, and needs a ``services.graphdb`` store
to search -- deployed with the rest of the stack, or one the facility already
runs.

``read_cypher`` and ``get_schema`` are the ``graph`` server's own tools, served
here under the ``channel-finder`` name: the same read transaction, the same
refusal of extension procedures and ``LOAD CSV``, the same row and time bounds.
The examples catalogue is this package's own, written around the questions
operators ask about channels rather than around general graph exploration. See
:doc:`/how-to/facility-knowledge/use-facility-graph`.

**Tools:**

- ``search_channels`` -- Look addresses up by keyword and facet in the search
  index built from the corpus, a page of rows at a time.
- ``read_cypher`` -- Run one read-only Cypher query and return the matching rows,
  bounded by ``services.graphdb.query_max_rows`` and
  ``services.graphdb.query_timeout_s``.
- ``get_schema`` -- Report the node labels, relationship types, sampled per-label
  property names and namespace prefixes this graph holds.
- ``example_queries`` -- Return runnable Cypher examples for the common channel
  questions, each with per-corpus parameter values.
- ``capabilities`` -- Report the server description, tool list and operating notes.

Workspace
---------

``workspace``
~~~~~~~~~~~~~

Package: ``osprey.mcp_server.workspace``

Manages artifacts, data context, screen capture, visualizations,
documents, and session state. This is the largest server, grouping
tools into several functional areas.

**Artifacts:**

- ``artifact_register`` -- Register a file on disk, or literal text, as a gallery artifact.
- ``artifact_get`` -- Look up an artifact by ID to get its file path and metadata.
- ``artifact_list`` -- List stored artifacts, optionally narrowed by category, tool or agent.
- ``artifact_read`` -- Read a stored artifact's full content (small artifacts only).
- ``artifact_delete`` -- Delete an artifact from the gallery.
- ``artifact_delete_all`` -- Delete a whole scope of artifacts (one category, or everything)
  in a single call. The ``scope`` argument is required.
- ``artifact_export`` -- Export an artifact to a different format (e.g., PNG, SVG, PDF).
- ``artifact_focus`` -- Select an artifact in the gallery so the user sees it.
- ``artifact_pin`` -- Pin or unpin an artifact for quick-access filtering.

**Visualization:**

- ``create_static_plot`` -- Execute Python/Matplotlib code to produce a static plot image.
- ``create_interactive_plot`` -- Execute Plotly code to produce an interactive HTML plot.
- ``create_dashboard`` -- Execute Panel/Bokeh code to produce a live dashboard app.
- ``create_document`` -- Compile LaTeX source to PDF (Beamer slides or article reports).

**Stored data:**

Datasets are not a separate namespace -- an archiver read or a run result is an
artifact with a ``category``, so the artifact tools above list, read and delete
them (``artifact_list(category="archiver_data")``).

- ``archiver_downsample`` -- Downsample an archiver artifact to a target point count.

**Lattice Dashboard:**

- ``lattice_state`` -- Get current lattice state (summary, families, figures, baseline).
- ``lattice_set_param`` -- Set a magnet family parameter override.
- ``lattice_refresh`` -- Recompute figures: a figure name refreshes that one figure,
  ``verify`` runs ``da`` and ``lma``, and no argument refreshes the selected model's fast figures.
- ``lattice_set_baseline`` -- Snapshot the current state as the comparison baseline.
- ``lattice_clear_baseline`` -- Discard the saved comparison baseline.
- ``lattice_get_figure`` -- Return a named figure with its status and key when it is current for
  the inputs on screen, else an error naming its status (stale, computing, failed, not yet
  computed) or the reason the model cannot draw it.
- ``lattice_get_data`` -- Return the numeric data behind a named figure with its status and key
  when it is current for the inputs on screen, else the same errors as ``lattice_get_figure``.
- ``lattice_get_settings`` -- Get current dashboard settings (display options, baselines).
- ``lattice_update_settings`` -- Update dashboard settings.

**Panels:**

- ``list_panels`` -- List the panels available in the Web Terminal (built-in and
  custom), the dock's layout (every tile with its tabs and showing tab, the
  terminal included) and its ``rev``.
- ``open_panel`` -- Put a panel on screen: a new tab of the service tile used
  last, or focus it where it already is.
- ``open_panel_set`` -- Open several panels in one call, as tabs of one tile or
  side by side.
- ``arrange_workspace`` -- Lay out the service tiles from a configured preset or
  an exact layout; the terminal keeps its place.
- ``drop_panel`` -- Make any drop a person can make (tile centre, tab position,
  edge, a whole side of the dock), plus a drop between two tiles, with a new
  panel, an open tab or a whole tile.
- ``close_panel_tab`` -- Close a panel's tab, leaving it on the rail.
- ``close_panel_tile`` -- Close a whole tile with its tabs.
- ``add_panel_to_rail`` -- Make a panel launchable from the rail in one click
  (requires human approval).
- ``remove_panel_from_rail`` -- Take a panel off the rail (and off screen with it)
  (requires human approval).
- ``register_panel`` -- Add a rail entry at runtime for an upstream URL the Web
  Terminal proxies (requires human approval).

The three rail tools ask because rail membership is their whole effect:
removing an entry costs the operator the ability to launch that panel back, and
adding or registering one puts a new entry in front of them. The layout tools
say what is on screen and are auto-allowed, even where a layout implies a rail
change: a panel an agent opens joins the rail once it is on screen, and
``arrange_workspace`` with a ``preset`` also drops non-members -- the same
pruning the operator gets by clicking that layout.

Every layout tool is one agent dock operation sent through the op helper
(``panel_op``): the web terminal relays it to the browser holding the dock and
answers with the resulting layout and ``rev``, or with a named refusal
(``stale_base``, ``no_dock_client``, ``terminal_not_agent_movable``, ...) that
changed nothing. The tools that name tile ids (``drop_panel``,
``close_panel_tile``) take the ``rev`` of the agent's last read as
``base_rev``. The terminal is never an agent drop source or close target, and a
layout never names it.

**Screen Capture:**

- ``screenshot_capture`` -- Capture a screenshot (full screen, window, or region).
- ``list_windows`` -- List visible windows with optional app filter.
- ``manage_window`` -- Manage window state (raise, minimize, resize).

**Session:**

- ``session_log`` -- Retrieve the structured session activity log.
- ``session_summary`` -- Return a compact inventory of all data and artifacts in the session.
- ``submit_response`` -- Submit a formatted response to the web terminal.
- ``facility_description`` -- Get the hand-written facility description and the build's generated facts page (places with device counts, device classes with their groups, models, and the signals and roles the channels use).
- ``prior_answer_read`` -- Read the full text of an earlier answer a chat bridge replayed shortened (dispatched runs only).
- ``provenance_locator`` -- Return the telemetry coordinates (session id, service name, time
  window) that locate this session in the agent record, for a filed issue to carry.

**Setup / Diagnostics:**

- ``setup_inspect`` -- Inspect OSPREY agent configuration (config files, env vars, MCP state).
- ``setup_patch`` -- Modify an OSPREY configuration file (config.yml or .mcp.json).


Python Executor
---------------

``python_executor``
~~~~~~~~~~~~~~~~~~~

Package: ``osprey.mcp_server.python_executor``

Runs agent-authored Python as a separate process on the host --- a process
boundary with limits enforcement and timeout protection, not a sandbox; see
:doc:`/architecture/python-executor`.

**Tools:**

- ``execute`` -- Execute Python code with safety checks, process isolation, and timeout.
- ``execute_file`` -- Execute a Python file from the workspace with the same safety envelope as ``execute``.

Two more tools are registered only when the render's facility facts record a pyAML view
(see :doc:`/how-to/measure-with-pyaml`):

- ``pyaml_measure`` -- Measure a response matrix or the chromaticity with pyAML on the
  control target. It steps setpoints, so it is a write: see :ref:`A pyAML tool
  <architecture-safety-chain-pyaml-tool>`.
- ``model_response_matrix`` -- Solve a served model's orbit response to its correctors on
  the model alone. Read-only.


ARIEL
-----

``ariel``
~~~~~~~~~

Package: ``osprey.mcp_server.ariel``

Searches facility logbook entries and operational records. Supports
keyword search, semantic (embedding-based) search, direct SQL queries,
and entry creation with attachments.

**Tools:**

- ``browse`` -- Browse recent logbook entries with optional date and field filtering.
- ``filter_options`` -- Get distinct values for a filterable field (authors, systems, etc.).
- ``keyword_search`` -- Search the logbook using PostgreSQL full-text keyword search.
- ``semantic_search`` -- Search the logbook using embedding-based semantic similarity.
- ``hybrid_search`` -- Search the logbook with keyword and semantic ranking combined.
- ``sql_query`` -- Execute a read-only SQL query against the logbook database.
- ``entry_get`` -- Get a single logbook entry by its ID.
- ``entries_by_ids`` -- Get multiple logbook entries by their IDs in a single call.
- ``attachment_view`` -- Look at one viewable picture attached to an entry (hidden when ``ariel.attachments.view.enabled`` is off).
- ``entry_open`` -- Show an entry, and optionally one of its pictures enlarged, in the ARIEL web panel.
- ``attachment_to_artifact`` -- Copy one viewable picture into the artifact gallery and select it there (hidden when ``ariel.attachments.view.enabled`` is off).
- ``entry_create`` -- Create a new logbook entry, optionally with file attachments.
- ``entry_publish`` -- Publish an existing ARIEL entry to the facility logbook.
- ``capabilities`` -- Report available ARIEL search capabilities.
- ``status`` -- Get ARIEL service health, database connectivity, and statistics.


Facility Knowledge
------------------

``osprey_facility_knowledge``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Package: ``osprey.mcp_server.facility_knowledge``

Serves a facility's curated knowledge bundle (OKF §2 concept documents) so the
agent can look up operational know-how and draft new concepts for human review.

**Tools:**

- ``capabilities`` -- Report facility knowledge bundle capabilities.
- ``list_concepts`` -- List all concepts in the facility knowledge bundle.
- ``read_concept`` -- Read a concept document by its OKF §2 concept ID.
- ``search`` -- Search the facility knowledge bundle for a query string.
- ``draft_concept`` -- Draft a new concept document for human review and approval.


Facility Graph
--------------

``graph``
~~~~~~~~~

Package: ``osprey.mcp_server.graph``

Read-only Cypher search over the facility knowledge graph -- the
NARAD-convention RDF corpus held by the ``graphdb`` store. Rendered only where
``services.graphdb`` is configured, and called by the
**facility-knowledge-graph subagent**: the main agent's route to the graph is
delegation, mirroring the channel finder. Every query runs in a read
transaction, and extension procedures, extension functions and ``LOAD CSV`` are
refused before the store is dialed. See :doc:`/how-to/facility-knowledge/use-facility-graph`.

**Tools:**

- ``read_cypher`` -- Run one read-only Cypher query and return the matching rows,
  bounded by ``services.graphdb.query_max_rows`` and
  ``services.graphdb.query_timeout_s``.
- ``get_schema`` -- Report the node labels, relationship types, sampled per-label
  property names and namespace prefixes this graph holds.
- ``example_queries`` -- Return curated, runnable Cypher examples for the common
  question shapes, each with per-corpus parameter values.
- ``capabilities`` -- Report the server description, tool list and operating notes.


Phoebus
-------

``phoebus``
~~~~~~~~~~~

Package: ``osprey.mcp_server.phoebus``

Drives a running Phoebus product through its agent bridge, an HTTP server inside
the Phoebus process: it opens displays, reads their widgets and live values,
snapshots widgets, drives controls and opens Data Browser plots. It is off in
every preset and switched on with ``claude_code.servers.phoebus.enabled: true``;
a second Phoebus product is reached by a second instance declared with
``claude_code.servers.<name>.extends: phoebus``. A drive is offered to the agent
only under ``phoebus.agent_access: read_write``, and it writes through the
Phoebus product, not a connector: see :ref:`A Phoebus drive
<architecture-safety-chain-phoebus-drive>`. Configuration:
:doc:`/how-to/control-systems/phoebus-bridge`.

**Tools:**

- ``phoebus_list_displays`` -- List the displays open in the Phoebus product, with
  whether each is ready and which is focused.
- ``phoebus_open_panel`` -- Open a display by its name in ``phoebus.panels`` and
  return a handle later calls can address it by.
- ``phoebus_panel_lookup`` -- Answer whether a display file is registered in this
  terminal's ``phoebus.panels``, and under which name.
- ``phoebus_perceive`` -- Walk a display's widget tree and report each widget with
  its bounds and its channel's live value, severity and writability.
- ``phoebus_perceive_region`` -- The same, for the widgets inside one on-screen
  rectangle.
- ``phoebus_snapshot`` -- Save a PNG of one widget as an artifact.
- ``phoebus_drive`` -- Click a control or type a value into a widget; offered only
  under ``phoebus.agent_access: read_write`` and asked for approval.
- ``phoebus_open_databrowser`` -- Open a Data Browser plot of a channel list over a
  time range, bound to the archiver when ``phoebus.archiver_url`` is set.


Optimizer
---------

``optimizer``
~~~~~~~~~~~~~

Package: ``osprey.mcp_server.optimizer``

Reads the optimizer service a profile deploys with its ``optimizer:`` section
(:doc:`/reference/configuration/profile`). It is off in every preset and
switched on with ``claude_code.servers.optimizer.enabled: true``. The build
refuses that switch on a render with no optimizer to dial: a deployment that
does not run one, or an attached render whose host publishes no
``services.optimizer.port``. The server dials ``services.optimizer.port`` on
loopback, or ``OPTIMIZER_URL`` when that is set.

The server holds no launch token and cannot start a run: a person starts
every run. A tool that reads a machine (``environment_status``,
``read_observables``, ``initial_bounds``) asks about the session's current
control target unless it is given a ``target``; the optimizer serves only the
targets its profile lists (``va`` and ``standin`` today), and refuses any other.

**Tools:**

- ``list_environments`` -- List the optimization environments the service offers.
- ``get_environment`` -- Read one environment's variable and observable definitions.
- ``environment_status`` -- Check whether an environment is reachable on a machine.
- ``read_observables`` -- Read the current values of an environment's observables.
- ``list_presets`` -- List the run presets an environment offers.
- ``initial_bounds`` -- Resolve each variable's starting bounds from its current setpoint.
- ``list_jobs`` -- List the jobs that have not finished.
- ``get_job`` -- Read one job's status, optionally with its result summary or full data.
- ``current_job`` -- Read the current-job pointer, the job a start launches.
- ``get_job_config`` -- Read an optimization job's stored config and its revision.
- ``list_results`` -- List the stored runs.
- ``get_result`` -- Read one stored run: config, best point and evaluated records.
- ``list_models`` -- List the saved surrogate model files.


Bluesky
-------

``bluesky``
~~~~~~~~~~~

Package: ``osprey.mcp_server.bluesky``

A thin HTTP client of the facility-side Bluesky bridge. It lists plans and
devices, and stages a shared plan draft. It queues and starts that draft, stops
a run and reads run data. It also writes and validates session plans. It is off
by default and enabled with ``claude_code.servers.bluesky.enabled: true``; the
``control-assistant`` preset enables it. ``queue_add`` and
``queue_start`` can set hardware in motion, so they are governed writes, like
``channel_write``: the writes check runs first, then the write switch's position
decides whether the operator is asked. See :doc:`/how-to/bluesky/queue`.

**Plans and devices:**

- ``list_plans`` -- List the plans registered on the bridge, with each plan's metadata,
  provenance and parameter schema.
- ``get_plan_source`` -- Read one plan's source text as it sits on disk.
- ``list_devices`` -- List the device names this deployment's plans accept, one page at a
  time.

**Draft:**

- ``get_draft`` -- Read the shared plan draft, its revision and its lane.
- ``set_draft`` -- Create or edit the shared plan draft; every open Bluesky panel shows
  the edit live.
- ``clear_draft`` -- Clear the shared plan draft.

**Authoring:**

- ``write_plan`` -- Write a session-tier plan file on the bridge; the bridge never runs
  it (requires human approval).
- ``validate_plan`` -- Validate a written session plan with the static checks and a
  mock-device dry run (requires human approval).

**Queue:**

- ``queue_status`` -- Report whether this deployment can run plans at all or is
  browse-only.
- ``queue_list`` -- Read the queue: pending items, the running item and the manager's
  state.
- ``queue_add`` -- Add the shared draft, at a pinned revision, to the queue. Starts
  nothing on a stopped queue.
- ``queue_start`` -- Start draining the queue: every pending item runs, in order.
- ``queue_stop`` -- Stop the queue after the running item finishes, or withdraw a
  pending stop (requires human approval).
- ``queue_remove`` -- Remove one pending item from the queue (requires human approval).

**Runs:**

- ``list_runs`` -- List the runs OSPREY queued, newest first.
- ``get_run`` -- Read one run's lifecycle status.
- ``get_run_data`` -- Read a bounded window of a run's data.
- ``get_run_figure`` -- Read a run's figure, bounded to a fixed point budget.
- ``stop_run`` -- Abort the running plan now. Data collected so far is kept and the
  hardware stays where the plan left it (requires human approval).


Health
------

``health``
~~~~~~~~~~

Package: ``osprey.mcp_server.health``

Runs the facility health suite for the agent. It is off by default and enabled
with ``claude_code.servers.health.enabled: true``; the ``control-assistant``
preset enables it. Its tools take only a
category filter, so every probe they run is declared in config and read-only.
See :doc:`/how-to/health-and-monitoring/configure-health-checks`.

**Tools:**

- ``health_check`` -- Run the cached poll tier: config load, connector reachability,
  provider canaries, archiver freshness and other fast read-only checks.
- ``health_check_full`` -- Run the full tier, which adds the costly ``on_demand`` checks
  the poll tier reports as skipped (requires human approval).
