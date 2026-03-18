MCP Servers
===========

OSPREY exposes control system operations, data retrieval, and workspace
management as tools through `FastMCP <https://github.com/jlowin/fastmcp>`_
servers. Each server is a standalone process that registers tools via the
``@mcp.tool()`` decorator and communicates with Claude Code over the MCP
(Model Context Protocol) transport.

.. admonition:: PLACEHOLDER: CONCEPTUAL-MAPPING
   :class: warning

   **Old content:** N/A (new page)

   **New equivalent:** Needs human judgment

   **Why this is fuzzy:** How Claude Code discovers and selects MCP tools at runtime depends on deployment configuration decisions not yet documented.

   **Action needed:** Describe how tool selection works in Claude Code's MCP architecture, or link to Claude Code docs.


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
- ``channel_write`` -- Write values to one or more control system channels (requires human approval).
- ``archiver_read`` -- Retrieve historical archived data for one or more channels over a time range.
- ``channel_limits`` -- Query the channel safety limits database (lookup, pattern match, summary).


Channel Finding
---------------

OSPREY provides four channel finder variants, each suited to different
facility data models and search strategies. Deployments typically enable
one or two variants depending on available metadata.

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

Provides synchronous access to channel databases stored in flat or
template format. Designed for smaller databases that fit in context.

**Tools:**

- ``get_channels`` -- Get channels from the database, optionally paginated in chunks.
- ``resolve_addresses`` -- Resolve channel names to their control system PV addresses.
- ``statistics`` -- Get database statistics (total channels, format, chunk info).

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

``direct_channel_finder``
~~~~~~~~~~~~~~~~~~~~~~~~~

Package: ``osprey.mcp_server.direct_channel_finder``

Queries live PV metadata backends and uses curated channel databases as
reference context for naming patterns. Best for facilities with an
online channel finder service.

**Tools:**

- ``search_pvs`` -- Search for PVs matching a glob pattern with optional filters.
- ``get_pv_metadata`` -- Get detailed metadata for specific PV names.
- ``get_naming_patterns`` -- Get a summary of PV naming conventions from the curated database.


Workspace
---------

``workspace``
~~~~~~~~~~~~~

Package: ``osprey.mcp_server.workspace``

Manages artifacts, data context, screen capture, visualizations,
documents, and session state. This is the largest server, grouping
tools into several functional areas.

**Artifacts:**

- ``artifact_save`` -- Save a file or generated content as a gallery artifact.
- ``artifact_get`` -- Look up an artifact by ID to get its file path and metadata.
- ``artifact_delete`` -- Delete an artifact from the gallery.
- ``artifact_export`` -- Export an artifact to a different format (e.g., PNG, SVG, PDF).
- ``artifact_focus`` -- Select an artifact in the gallery so the user sees it.
- ``artifact_pin`` -- Pin or unpin an artifact for quick-access filtering.

**Visualization:**

- ``create_static_plot`` -- Execute Python/Matplotlib code to produce a static plot image.
- ``create_interactive_plot`` -- Execute Plotly code to produce an interactive HTML plot.
- ``create_dashboard`` -- Execute Panel/Bokeh code to produce a live dashboard app.
- ``create_document`` -- Compile LaTeX source to PDF (Beamer slides or article reports).

**Data Context:**

- ``data_list`` -- List all data entries in the workspace with optional filters.
- ``data_read`` -- Read the full data content of a data entry.
- ``data_delete`` -- Delete a data entry from the workspace.
- ``archiver_downsample`` -- Downsample an archiver data entry to a target point count.

**Graph Analysis:**

- ``graph_extract`` -- Extract numerical data from a chart image using the DePlot service.
- ``graph_compare`` -- Compare two datasets using mathematical distance metrics.
- ``graph_save_reference`` -- Save a dataset as a named reference for future comparisons.

**Lattice Dashboard:**

- ``lattice_init`` -- Load a lattice file into the dashboard and compute optics.
- ``lattice_state`` -- Get current lattice state (summary, families, figures, baseline).
- ``lattice_set_param`` -- Set a magnet family parameter override.
- ``lattice_refresh`` -- Trigger recomputation of lattice figures.
- ``lattice_set_baseline`` -- Snapshot the current state as the comparison baseline.

**Screen Capture:**

- ``screenshot_capture`` -- Capture a screenshot (full screen, window, or region).
- ``list_windows`` -- List visible windows with optional app filter.
- ``manage_window`` -- Manage window state (raise, minimize, resize).

**Session:**

- ``session_log`` -- Retrieve the structured session activity log.
- ``session_summary`` -- Return a compact inventory of all data and artifacts in the session.
- ``submit_response`` -- Submit a formatted response to the web terminal.
- ``facility_description`` -- Get facility description and context.

**Setup / Diagnostics:**

- ``setup_inspect`` -- Inspect OSPREY agent configuration (config files, env vars, MCP state).
- ``setup_patch`` -- Modify an OSPREY configuration file (config.yml or .mcp.json).


Python Executor
---------------

``python_executor``
~~~~~~~~~~~~~~~~~~~

Package: ``osprey.mcp_server.python_executor``

Executes Python code in a sandboxed environment with process isolation,
limits enforcement, and timeout protection.

**Tools:**

- ``execute`` -- Execute Python code with safety checks, process isolation, and timeout.


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
- ``sql_query`` -- Execute a read-only SQL query against the logbook database.
- ``entry_get`` -- Get a single logbook entry by its ID.
- ``entries_by_ids`` -- Get multiple logbook entries by their IDs in a single call.
- ``entry_create`` -- Create a new logbook entry, optionally with file attachments.
- ``capabilities`` -- Report available ARIEL search capabilities.
- ``status`` -- Get ARIEL service health, database connectivity, and statistics.


AccelPapers
------------

``accelpapers``
~~~~~~~~~~~~~~~

Package: ``osprey.mcp_server.accelpapers``

Hybrid BM25 + vector search over approximately 63,000 INSPIRE
accelerator physics papers stored in a local Typesense collection.

**Tools:**

- ``papers_search`` -- Search papers using hybrid BM25 + vector search.
- ``papers_search_author`` -- Search for papers by a specific author.
- ``papers_browse`` -- Browse papers by conference, year, or document type.
- ``papers_get`` -- Get full details for a specific paper by its texkey identifier.
- ``papers_list_conferences`` -- List conferences with paper counts.
- ``papers_stats`` -- Get summary statistics for the paper database.


MATLAB
------

``matlab``
~~~~~~~~~~

Package: ``osprey.mcp_server.matlab``

Search and browse approximately 3,000 MATLAB Middle Layer functions
stored in a local SQLite FTS5 database. Includes call-graph traversal
for dependency analysis.

**Tools:**

- ``mml_search`` -- Search MML functions using full-text BM25-ranked search.
- ``mml_browse`` -- Browse functions by group or type with sorting.
- ``mml_get`` -- Get full details for a specific MML function by name.
- ``mml_list_groups`` -- List all groups in the MML codebase with function counts.
- ``mml_dependencies`` -- Get the dependency graph for a function (recursive call traversal).
- ``mml_path`` -- Find the shortest call path between two functions (BFS).
- ``mml_stats`` -- Get summary statistics for the MML function database.
