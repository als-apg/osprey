.. _reference-ariel:

===============
ARIEL Contracts
===============

ARIEL is integrated into Osprey as a dedicated MCP server
(``osprey.mcp_server.ariel``) that exposes the logbook search service to the
agent layer through a set of specialized tools. When a user asks a question
like "What happened with the RF cavity last week?", the Osprey agent selects
the appropriate ARIEL MCP tool based on the query type, which invokes the
``ARIELSearchService`` and returns structured results that the agent uses to
produce a cited answer. This page is the reference for the contracts that
integration rests on: the MCP tools, the search result structure, the service
factory, the capabilities endpoint the web interface reads, and the database
schema every entry is stored in.

Integration Architecture
========================

.. raw:: html
   :file: ../../_diagrams/ariel-integration.html

The flow begins when the Osprey agent determines that a user query involves
historical logbook data. It selects from ARIEL's specialized MCP tools based
on the query type --- for example, ``keyword_search`` for exact-match lookups,
``semantic_search`` for conceptual queries, or ``browse`` for exploring recent
entries. Each tool builds the appropriate request and routes it through the
``ARIELSearchService``. Results are returned directly to the agent, which
uses them to generate a cited response.


ARIEL MCP Tools
===============

ARIEL exposes the following tools through its dedicated MCP server. The
Osprey agent selects the appropriate tool based on the user's query.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Tool
     - Purpose
   * - ``keyword_search``
     - Full-text keyword search across logbook entries
   * - ``semantic_search``
     - Vector similarity search using embeddings
   * - ``sql_query``
     - Direct SQL queries against the logbook database
   * - ``browse``
     - Browse entries with pagination and filters
   * - ``filter_options``
     - Get distinct values for a filterable field (authors or source systems)
   * - ``status``
     - Check ARIEL service health and configuration
   * - ``entry_publish``
     - Publish an existing ARIEL entry to the facility logbook
   * - ``capabilities``
     - List available search capabilities and their status
   * - ``entry_get``
     - Retrieve a single entry by ID
   * - ``entries_by_ids``
     - Batch retrieve multiple entries by their IDs
   * - ``entry_create``
     - Create a new logbook entry

**Source:** :file:`src/osprey/mcp_server/ariel/tools/`


Search Result Structure
=======================

Internally, each search produces an ``ARIELSearchResult`` with the fields
below; the MCP tools serialize this into their own JSON envelope for the
agent:

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Field
     - Type
     - Description
   * - ``entries``
     - ``tuple[dict, ...]``
     - Matching entries, ranked by relevance
   * - ``answer``
     - ``str | None``
     - Synthesized answer text (set by callers)
   * - ``sources``
     - ``tuple[str, ...]``
     - Entry IDs used as sources
   * - ``search_modes_used``
     - ``tuple[str, ...]``
     - Modes executed (e.g., ``keyword``, ``semantic``)
   * - ``reasoning``
     - ``str``
     - Explanation of results
   * - ``diagnostics``
     - ``tuple[SearchDiagnostic, ...]``
     - Structured diagnostics (level, source, message) from search execution


Service Factory
===============

The ``get_ariel_search_service()`` function provides a singleton
``ARIELSearchService`` instance. The service is lazily initialized from
``config.yml`` on first access and reused for subsequent calls.

.. code-block:: python

   from osprey.services.ariel_search.capability import get_ariel_search_service

   service = await get_ariel_search_service()
   async with service:
       result = await service.search(query="RF cavity fault")

**Lifecycle:** The singleton is created once per process. In the web
interface, the ``create_app()`` factory manages its own service instance
through the FastAPI lifespan. For cleanup in tests, use
``close_ariel_service()`` (closes the connection pool) or
``reset_ariel_service()`` (resets without closing).

**Source:** :file:`src/osprey/services/ariel_search/capability.py`


Capabilities API
================

The web interface discovers its search modes and tunable parameters dynamically at startup by calling ``GET /api/capabilities``.

**The "add a module, get a UI knob for free" pattern:** When you register a custom search module with ``get_parameter_descriptors()``, its parameters automatically appear in the web interface's advanced options panel. The ``ParameterDescriptor`` dataclass supports types ``int``, ``float``, ``bool``, ``text``, ``date``, ``select``, and ``dynamic_select`` (which fetches options from an API endpoint). Parameters are grouped by ``section`` for visual organization.

.. admonition:: Security & Resilience
   :class: note

   - **XSS-safe highlights:** Search result highlights from PostgreSQL ``ts_headline`` are sanitized by ``sanitizeHighlight()`` in ``components.js`` --- only ``<b>`` and ``</b>`` tags are preserved; all other HTML is escaped.
   - **No cross-origin surface:** The interface app registers no CORS middleware. The browser and the ``/api`` endpoints it calls are served from the same origin, so there is no cross-origin request path to allow or restrict.
   - **Frontend fallback:** If ``/api/capabilities`` is unavailable at startup, the frontend falls back to a default mode list so the interface remains usable.

.. dropdown:: Technical Reference

   .. tab-set::

      .. tab-item:: REST API

         All endpoints are mounted under the ``/api`` prefix.

         .. list-table::
            :header-rows: 1
            :widths: 10 30 60

            * - Method
              - Endpoint
              - Description
            * - GET
              - ``/api/capabilities``
              - Discover available search modes and parameters
            * - GET
              - ``/api/filter-options/{field}``
              - Get distinct values for a filterable field (``authors``, ``source_systems``)
            * - POST
              - ``/api/search``
              - Execute a search query (body: :class:`~osprey.interfaces.ariel.api.schemas.SearchRequest`)
            * - GET
              - ``/api/entries``
              - List entries with pagination and filtering
            * - GET
              - ``/api/entries/{entry_id}``
              - Get a single entry by ID
            * - POST
              - ``/api/entries``
              - Create a new logbook entry (body: :class:`~osprey.interfaces.ariel.api.schemas.EntryCreateRequest`)
            * - POST
              - ``/api/entries/upload``
              - Create a new entry from multipart form data with attachments
            * - GET
              - ``/api/attachments/{attachment_id}``
              - Download an attachment binary
            * - GET
              - ``/api/status``
              - Service health, module status, and statistics
            * - GET
              - ``/api/config``
              - Read the current ARIEL configuration block
            * - PUT
              - ``/api/config``
              - Update the ARIEL configuration block
            * - GET
              - ``/api/publish-info``
              - Describe the configured logbook's write capability (the create
                form adapts its credential prompt to it)
            * - POST
              - ``/api/drafts``
              - Create a draft entry (pre-fill data for the web form)
            * - GET
              - ``/api/drafts/{draft_id}``
              - Read a draft entry
            * - GET
              - ``/api/drafts/{draft_id}/attachments/{filename}``
              - Download a draft's attachment

         Additionally, a ``GET /health`` endpoint at the root level returns a simple health check response.

         **SearchResponse:**

         .. code-block:: json

            {
              "entries": [
                {
                  "entry_id": "12345",
                  "source_system": "ALS eLog",
                  "timestamp": "2025-01-15T08:30:00Z",
                  "author": "J. Smith",
                  "raw_text": "RF cavity trip at 08:15...",
                  "summary": "RF cavity fault requiring manual reset",
                  "keywords": ["RF", "cavity", "trip"],
                  "score": 0.92,
                  "highlights": ["<b>RF cavity</b> trip at 08:15"]
                }
              ],
              "search_modes_used": ["keyword", "semantic"],
              "total_results": 1,
              "execution_time_ms": 340
            }

         **StatusResponse:**

         .. code-block:: json

            {
              "healthy": true,
              "database_connected": true,
              "database_uri": "postgresql://localhost:5432/ariel",
              "entry_count": 15230,
              "embedding_tables": [
                {"table_name": "text_embeddings_nomic_embed_text", "entry_count": 15230, "dimension": 768, "is_active": true}
              ],
              "active_embedding_model": "nomic-embed-text",
              "enabled_search_modules": ["keyword", "semantic"],
              "enabled_enhancement_modules": ["text_embedding", "semantic_processor"],
              "last_ingestion": "2025-01-15T06:00:00Z",
              "errors": []
            }

         See :mod:`osprey.interfaces.ariel.api.schemas` for the full Pydantic model definitions.

      .. tab-item:: Capabilities

         The ``/api/capabilities`` endpoint returns a JSON structure that groups enabled search modes under category objects (currently a single ``direct`` category), along with shared parameters:

         .. code-block:: json

            {
              "categories": {
                "direct": {
                  "label": "Direct",
                  "modes": [
                    {
                      "name": "keyword",
                      "label": "Keyword",
                      "description": "Full-text PostgreSQL search...",
                      "parameters": [
                        {
                          "name": "fuzzy_fallback",
                          "label": "Fuzzy Fallback",
                          "type": "bool",
                          "default": true,
                          "section": "Options"
                        }
                      ]
                    },
                    {
                      "name": "semantic",
                      "label": "Semantic",
                      "description": "Embedding-based similarity search...",
                      "parameters": []
                    }
                  ]
                }
              },
              "shared_parameters": [
                {"name": "max_results", "type": "int", "default": 10},
                {"name": "start_date", "type": "date"},
                {"name": "author", "type": "text"},
                {"name": "source_system", "type": "dynamic_select",
                 "options_endpoint": "/api/filter-options/source_systems"}
              ]
            }

         **How it works:** The ``get_capabilities()`` function in :mod:`osprey.services.ariel_search.capabilities` iterates over enabled search modules from the registry. Each module provides a ``get_tool_descriptor()`` (for its description) and optionally ``get_parameter_descriptors()`` (for its tunable parameters).

      .. tab-item:: App Internals

         **App factory:** The ``create_app()`` function in :mod:`osprey.interfaces.ariel.app` is a standard FastAPI app factory. It accepts an optional ``config_path`` argument and returns a fully configured FastAPI application with API routes and static file serving. It registers no CORS middleware --- the app and its ``/api`` endpoints share one origin, so there is no cross-origin surface.

         **Lifespan management:** The app uses FastAPI's ``lifespan`` context manager to initialize the ``ARIELSearchService`` on startup and clean it up on shutdown. During initialization:

         1. **Registry bootstrap** --- pre-creates the framework registry singleton (without an application registry path) so that ARIEL's search module discovery works even when running outside a full Osprey application.

         2. **Config loading** --- searches for ``config.yml`` in four locations: the provided ``config_path``, ``/app/config.yml`` (Docker mount), the ``CONFIG_FILE`` environment variable, and the current directory. Applies the ``ARIEL_DATABASE_HOST`` environment variable override for Docker networking.

         3. **Service creation** --- creates the ``ARIELSearchService`` from the loaded config and stores it in ``app.state.ariel_service``.

         4. **Health check** --- validates the database connection and logs the result.

         **Docker environment overrides:**

         .. list-table::
            :header-rows: 1
            :widths: 35 65

            * - Variable
              - Description
            * - ``CONFIG_FILE``
              - Path to config.yml (alternative to default search)
            * - ``ARIEL_DATABASE_HOST``
              - Override database hostname in URI (e.g., ``postgresql`` for Docker compose networking)

      .. tab-item:: Frontend

         The frontend is a vanilla JavaScript SPA --- no build tools, no framework, no transpilation. All files are served as static assets from :file:`src/osprey/interfaces/ariel/static/`.

         **JavaScript modules:**

         .. list-table::
            :header-rows: 1
            :widths: 25 75

            * - Module
              - Responsibility
            * - ``app.js``
              - Application initialization and hash-based routing
            * - ``api.js``
              - REST client wrapping ``fetch()`` for all API endpoints
            * - ``search.js``
              - Search form, query submission, results rendering
            * - ``advanced-options.js``
              - Capabilities-driven advanced options panel (dynamic parameter controls)
            * - ``entries.js``, ``entries-detail.js``, ``entries-form.js``, ``entries-helpers.js``
              - Browse view with pagination, entry detail view, and the new
                entry form (split across the four modules)
            * - ``dashboard.js``
              - Status dashboard rendering and periodic health refresh
            * - ``components.js``
              - Shared UI components (entry cards, loading states, error messages)
            * - ``utils.js``
              - Small shared helpers
            * - ``settings.js``
              - Settings UI: read/write ARIEL config block via ``/api/config``

         Side-drawer behavior (filters, advanced options, settings panels) comes
         from the shared ``<osprey-drawer>`` design-system component rather than
         a page-local module. The header's display-preferences popover is the
         shared ``<osprey-display-menu>`` component in the same way: ARIEL
         mounts the element and passes it the settings drawer's id, and the
         component supplies the trigger, the card, the Appearance/View/Theme
         rows, and the hiding-when-embedded rule.

         Theming role: standalone, ARIEL is the *hub* --- the display menu is
         the operator's control and its pick is the one that gets remembered.
         Embedded, it is a *follower*: the web terminal owns the pick and
         broadcasts it, and ARIEL applies what it is sent.

         **CSS architecture:**

         .. list-table::
            :header-rows: 1
            :widths: 25 75

            * - File
              - Scope
            * - ``base.css``
              - Reset, typography, form elements
            * - ``components.css``
              - Cards, buttons, badges, modals, search results
            * - ``layout.css``
              - Header, navigation, main content, responsive grid
            * - ``drawer.css``
              - Page-level styling around the shared drawer component
            * - ``settings.css``
              - Settings drawer styling (config form, YAML editor, save bar)

         Design tokens (colors, spacing, typography, transitions) are not
         defined per page — they come from the shared design system's
         ``/design-system/css/tokens.css``.

         **Routing:** The app uses ``window.location.hash`` for navigation. The ``app.js`` module listens for ``hashchange`` events and shows/hides view sections (``#search``, ``#browse``, ``#create``, ``#status``). No page reloads occur during navigation.


.. _`database`:

Database Schema
===============

All ingested and enhanced data lives in PostgreSQL. The core ``enhanced_entries`` table stores one row per logbook entry with the normalized fields that every adapter produces --- entry ID, timestamp, author, raw text, and a JSONB metadata column for facility-specific extras. Enhancement modules write their results either into columns on this same table (keywords, summaries) or into dedicated per-model tables (vector embeddings). The pgvector extension provides the ``vector`` column type and cosine-distance operators that power semantic search. Core tables and an embedding table for each configured model are created automatically by ``osprey ariel migrate``, which reads ``enhancement_modules.text_embedding.models`` to determine which embedding tables to create. ``osprey ariel reembed`` (re)creates and backfills a single model's table on demand.

.. admonition:: pgvector requirement
   :class: important

   The **pgvector** extension is required for semantic search. It is automatically installed in the Osprey-managed PostgreSQL container (``osprey up``). For external databases, install it manually: ``CREATE EXTENSION IF NOT EXISTS vector;``

Core Tables
-----------

**enhanced_entries** --- Primary storage for logbook entries:

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Column
     - Type
     - Description
   * - ``entry_id``
     - ``TEXT PRIMARY KEY``
     - Unique entry identifier
   * - ``source_system``
     - ``TEXT``
     - Origin system name (e.g., "ALS eLog")
   * - ``timestamp``
     - ``TIMESTAMPTZ``
     - Entry creation time
   * - ``author``
     - ``TEXT``
     - Entry author
   * - ``raw_text``
     - ``TEXT``
     - Full entry text (subject + details)
   * - ``summary``
     - ``TEXT``
     - LLM-generated summary (from semantic processor)
   * - ``keywords``
     - ``TEXT[]``
     - LLM-extracted keywords (from semantic processor)
   * - ``metadata``
     - ``JSONB``
     - Additional structured data (title, tags, attachments)

**Per-model embedding tables** (e.g., ``text_embeddings_nomic_embed_text``):

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Column
     - Type
     - Description
   * - ``id``
     - ``SERIAL PRIMARY KEY``
     - Auto-incrementing primary key
   * - ``entry_id``
     - ``TEXT UNIQUE``
     - Foreign key to enhanced_entries
   * - ``embedding``
     - ``vector(<dim>)``
     - pgvector embedding column

**ingestion_runs** --- Tracks ingestion history:

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Column
     - Type
     - Description
   * - ``id``
     - ``SERIAL``
     - Auto-incrementing ID
   * - ``started_at``
     - ``TIMESTAMPTZ``
     - Ingestion start time
   * - ``completed_at``
     - ``TIMESTAMPTZ``
     - Ingestion completion time
   * - ``entries_added``
     - ``INTEGER``
     - Number of new entries added
   * - ``entries_updated``
     - ``INTEGER``
     - Number of existing entries updated
   * - ``entries_failed``
     - ``INTEGER``
     - Number of entries that failed
   * - ``source_system``
     - ``TEXT``
     - Source adapter name


Migration System
----------------

Migrations are run via ``osprey ariel migrate`` and managed by the ``run_migrations()`` function in ``database/migrations.py``. When ``text_embedding`` is enabled, ``migrate`` creates an embedding table for each model listed in ``enhancement_modules.text_embedding.models`` (falling back to ``nomic-embed-text``, 768, if none is configured); ``osprey ariel reembed`` (re)creates and backfills a single model's table on demand.

.. admonition:: Schema Evolution
   :class: outreach

   The current schema was designed around three facility logbook formats (ALS, JLab, ORNL) and may not capture every field your facility needs. The ``metadata`` JSONB column provides flexibility for facility-specific extras, but if your logbook requires a fundamentally different table structure, please open a pull request or contact us --- the ingestion and storage layers are designed to accommodate new schemas without disrupting existing ones.
