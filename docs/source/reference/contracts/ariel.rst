.. _reference-ariel:

===============
ARIEL Contracts
===============

ARIEL is integrated into Osprey as a dedicated MCP server
(``osprey.mcp_server.ariel``) that exposes the logbook search service to the
agent layer through a set of specialized tools. When a user asks a question
like "What happened with the RF cavity last week?", the OSPREY agent selects
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

The flow begins when the OSPREY agent determines that a user query involves
historical logbook data. It selects from ARIEL's specialized MCP tools based
on the query type --- for example, ``keyword_search`` for exact-match lookups,
``semantic_search`` for conceptual queries, or ``browse`` for exploring recent
entries. Each tool builds the appropriate request and routes it through the
``ARIELSearchService``. Results are returned directly to the agent, which
uses them to generate a cited response.


ARIEL MCP Tools
===============

ARIEL exposes the following tools through its dedicated MCP server. The
OSPREY agent selects the appropriate tool based on the user's query.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Tool
     - Purpose
   * - ``keyword_search``
     - Full-text keyword search across logbook entries
   * - ``semantic_search``
     - Vector similarity search using embeddings
   * - ``hybrid_search``
     - Keyword and semantic ranking combined, with optional query expansion
       and reranking (offered when the hybrid module is enabled)
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
   * - ``attachment_view``
     - Return one viewable picture attached to an entry, with its summary
   * - ``entry_open``
     - Show an entry, and optionally one of its viewable pictures enlarged, in
       the ARIEL web panel
   * - ``attachment_to_artifact``
     - Copy one viewable picture into the artifact gallery and select it there
   * - ``entry_create``
     - Create a new logbook entry

Search and ``browse`` results carry the first ``ariel.entry_text.listing_chars``
characters of each entry's ``raw_text``, and ``entries_by_ids`` the first
``ariel.entry_text.read_chars``. An entry cut there also carries
``raw_text_truncated: true`` and its full ``raw_text_length``. ``entry_get``
returns the whole entry. See :ref:`config-ariel-entry-text`.

``entries_by_ids`` names each requested ID the logbook does not hold under
``missing``, mapped to ``null``; when it holds none of them the call fails as
``nothing_answered``. See :doc:`partial-answers`.

Attachment summaries
--------------------

Every tool that returns entries describes their attachments with one summary
shape, listed under ``attachments``:

.. code-block:: text

   {attachment_id?, filename, mime_type, viewable, copy_status, skip_reason?,
    caption?, caption_source?, visible_text?, url?}

Listings (search results, ``browse`` and ``entries_by_ids``) cut ``caption``
and ``visible_text`` to 200 characters and mark a cut field with
``caption_truncated`` or ``visible_text_truncated``; they carry only the first
few summaries and count them all in ``attachment_count``. A search hit that
matched through an attachment names it in ``matched_attachment_ids``.
``entry_get`` returns every summary with its caption in full.

An attachment is ``viewable`` when its copy finished (``copy_status`` is
``copied``), it was not skipped (no ``skip_reason``), its type is an image,
and its display rendition is stored. ``attachment_view`` takes the
``attachment_id`` of a viewable attachment and returns a JSON block (the
summary plus ``entry_id``, ``source_url``, ``size_bytes``, ``rendition_size``
and ``rendition_sha256``) followed by the picture. An unknown id is
``not_found``; an attachment that is not viewable is ``no_results``.

``ariel.attachments.view.enabled`` (default ``true``) switches picture viewing
for the agent. With it off:

- ``attachment_view`` and ``attachment_to_artifact`` are absent from
  ``tools/list``, and a call to either returns ``not_supported``;
- listings carry no ``attachments``, ``attachment_count`` or
  ``matched_attachment_ids``;
- ``entry_get`` carries the entry's stored ``attachments`` items unchanged,
  without summaries.

The web panel's attachment display does not depend on this setting.

``entry_open`` takes an ``entry_id`` and an optional ``attachment_id`` and asks
the web terminal to bring the ARIEL panel forward at
``#sel=<entry_id>``, with ``&att=<attachment_id>`` when that attachment is
one of the entry's and is viewable. It returns ``entry_id``,
``attachment_id``, ``opened`` (``entry`` or ``entry_and_picture``), the panel
``url`` and a ``message``; without a web terminal the ``url`` is the link to
follow. An unknown entry, or an attachment that is not the entry's, is
``not_found``. The panel route also works in a browser on the standalone ARIEL
web page. The web terminal relays each ARIEL tool call to the panel's iframe
and pop-out as ``osprey-agent-activity`` ``{tool, detail, ts}``; on
``entry_create`` and ``entry_publish`` the list refreshes or counts the new row.

``attachment_to_artifact`` takes the ``attachment_id`` of a viewable attachment
and saves its display rendition, never the original upload, to the artifact
gallery as an ``image`` artifact in the ``visualization`` category, titled with
the filename and entry id. The entry id, attachment id, filename, and caption
with its ``caption_source`` are recorded in the artifact's description and in
``metadata.logbook_picture``. The artifact is keyed by the rendition's sha256,
so saving the same picture again returns the same artifact. The result is the
gallery's usual artifact response (``artifact_id``, ``title``,
``artifact_type``, ``category``, ``gallery_url`` ...) plus ``entry_id``,
``attachment_id``, ``created`` and ``focused``. Errors match
``attachment_view``'s.

**Source:** :file:`src/osprey/mcp_server/ariel/tools/`

Captions and picture search availability
----------------------------------------

The presets turn ``image_caption``, ``image_embedding`` and
``ariel.attachments.view.enabled`` on. Each picture module runs when its server
and model answer and is skipped otherwise, like semantic search without
Ollama; nothing else stops working. See :doc:`/how-to/ariel/picture-search`.

``capabilities.attachments`` carries:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Meaning
   * - ``captions``
     - ``image_caption`` is enabled
   * - ``picture_search``
     - ``image_embedding`` and the ``hybrid`` search module are both enabled
   * - ``picture_search_unavailable``
     - ``null``, or why picture search cannot answer: ``unreachable`` (the
       embedding server does not answer), ``model`` (it does not serve the
       configured model), ``auth`` (it refused the API key) or ``config``
       (the module's configuration or the database schema is incomplete).
       ``null`` when picture search is off, has not failed, or its last
       attempt succeeded.

A deployment **without llama-server** (or with ``image_embedding`` not
reachable for another reason) sees ``osprey ariel status`` name the skipped
``image_embedding`` module and why, ``hybrid_search`` answer on text only with
the diagnostic "Picture search unavailable --- results are matched on text
only, so entries known only by their pictures are missing.", and
``picture_search_unavailable`` set to the reason.

A deployment **without the caption model** sees ``osprey ariel status`` name
the skipped ``image_caption`` module and why; entries are searched on their
text without captions.

Each module is turned off with its own key:
``ariel.enhancement_modules.image_caption.enabled: false``,
``ariel.enhancement_modules.image_embedding.enabled: false``, and
``ariel.attachments.view.enabled: false`` for the view tool.


Entry fields
------------

A facility adapter may declare extra fields an author fills in when an entry
is written (see the Entry Fields section of
:doc:`/how-to/ariel/data-ingestion`). ``capabilities`` lists them under
``entry_fields``, in form order, and ``entry_create`` and ``entry_publish``
take their values as a ``fields`` object keyed by field name.

``capabilities.entry_fields`` is a list with one item per declared field:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Meaning
   * - ``name``
     - The key the value is passed under in ``fields`` and stored under in
       the entry's ``metadata``
   * - ``label``, ``description``, ``section``
     - How the field is shown and grouped in the entry form
   * - ``type``
     - ``text``, ``int``, ``float``, ``bool``, ``date``, ``select`` or
       ``dynamic_select``
   * - ``default``
     - The form's pre-filled value; never applied to a value left out
   * - ``options``
     - The allowed ``{value, label}`` choices of a ``select``
   * - ``min``, ``max``, ``step``, ``placeholder``
     - Present when declared
   * - ``required``
     - Present (``true``) when the field must be filled before the entry is
       written to the logbook
   * - ``depends_on``
     - Present when non-empty: the static fields whose values decide a
       ``dynamic_select``'s choices
   * - ``facet``
     - Present (``true``) when the field is a filter dimension

The list is empty when no ingestion adapter is configured or the adapter
declares no fields. When the declarations are broken, ``entry_fields`` is
empty and ``entry_fields_error`` says what is wrong; the rest of
``capabilities`` still answers.

``entry_create`` checks ``fields`` against the declarations before anything is
written. A draft (the default) may leave ``required`` fields out; a direct
create may not. Its check covers each value's type, length, range and
``select`` options, but does not ask the adapter for a ``dynamic_select``'s
current choices. A draft stores the coerced values under ``fields`` in the
draft JSON. ``entry_publish`` merges its ``fields`` over the values stored on
the entry and checks the result in full, including the adapter's current
``dynamic_select`` choices; a ``logbook`` argument wins over a stored or
passed ``logbook`` field. Both tools refuse a name that is not declared.

A refused value is a ``validation_error`` whose ``details.field`` names the
field; for a ``select`` the details also carry ``allowed``, the first 50
allowed values. When the adapter cannot list a ``dynamic_select``'s choices,
``entry_publish`` returns ``internal_error`` with ``details.field``. Broken
declarations are an ``internal_error`` from either tool.

The checks cover shape only. Whether a value is right for the facility is
decided by the adapter and the logbook.


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
   * - ``total``, ``total_exact``, ``has_more``
     - ``int | None``, ``bool``, ``bool``
     - Matches across all pages (capped at 1000), whether that count is
       exact, and whether a later page exists


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
through the FastAPI lifespan. The web lifespan, the MCP server's shutdown and
``close_ariel_service()`` all end the service with ``ARIELSearchService.aclose()``,
which ends its search-call threads and closes its connection pools.
``reset_ariel_service()`` drops the singleton without closing it.

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
              - Execute a search query (body: :class:`~osprey.interfaces.ariel.api.schemas.SearchRequest`;
                ``filters``, ``offset`` and ``view`` as below)
            * - GET
              - ``/api/entries``
              - List entries newest first: ``filters``, ``view``, and ``cursor``
                (or the legacy ``page``) with ``page_size``
            * - GET
              - ``/api/facets``
              - Value counts per filter dimension and a day histogram for
                ``q``, ``mode``, ``filters`` and ``advanced_params``
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
              - Service health, module status, statistics and ``enhancement_coverage``
            * - GET
              - ``/api/config``
              - Read the current ARIEL configuration block
            * - PUT
              - ``/api/config``
              - Update the ARIEL configuration block
            * - GET
              - ``/api/publish-info``
              - Describe the configured logbook's write capability and its
                declared entry fields (the create form adapts its credential
                prompt and its field sections to it)
            * - GET
              - ``/api/entry-fields/{name}/options``
              - List the current choices of a declared ``dynamic_select``
                entry field
            * - POST
              - ``/api/drafts``
              - Create a draft entry (pre-fill data for the web form)
            * - GET
              - ``/api/drafts``
              - List unexpired drafts, newest first
            * - GET
              - ``/api/drafts/{draft_id}``
              - Read a draft entry (422 when its file is not a readable draft)
            * - DELETE
              - ``/api/drafts/{draft_id}``
              - Discard a draft (204); its staged attachments are kept
            * - GET
              - ``/api/drafts/{draft_id}/attachments/{filename}``
              - Download a draft's attachment
            * - GET
              - ``/api/entries/{entry_id}/thread``
              - ``{parent_id, parent, children, truncated}``: the parent and the
                newest 200 follow-ups as rows, oldest first; ``parent`` null when
                ARIEL does not hold it; ``truncated`` true, else omitted, when
                older follow-ups were left out
            * - GET
              - ``/api/entries/{entry_id}/enhancements``
              - ``{entry_id, modules}``: one card per enabled module in execution
                order, with its ``state``, stored status fields, ``detail`` and,
                when failed or stale, the re-run ``command`` (404 for an unknown entry)
            * - GET
              - ``/api/entries/{entry_id}/related``
              - ``{state, entries}``: up to ``limit`` (default 5, at most 10)
                nearest rows by text embedding, never the entry itself;
                ``state`` is ``ok``, ``no_embedding`` or ``disabled``

         **Entry fields.** ``GET /api/publish-info`` returns
         ``{supports_write, requires_auth, source_system, entry_fields}``.
         ``entry_fields`` is always a list, empty when no adapter is configured
         or none is declared; its items carry the same keys as the
         ``capabilities.entry_fields`` items of the MCP server, and a
         ``dynamic_select`` item also carries ``options_endpoint``
         (``/entry-fields/{name}/options``, relative to the ``/api`` prefix).
         ``GET /api/entry-fields/{name}/options`` returns
         ``{"field": name, "options": [{"value": ..., "label": ...}]}``. It reads
         only the field's ``depends_on`` values from the query string, each
         checked against its own declaration, and is 404 unless ``name`` is a
         declared ``dynamic_select``. ``POST /api/entries`` and
         ``POST /api/entries/upload`` check the declared values in full,
         including the adapter's current ``dynamic_select`` choices, before
         anything is written; the adapter receives the declared values only,
         and undeclared metadata stays in ARIEL's copy of the entry. A draft
         read from ``GET /api/drafts/{draft_id}`` carries the values an agent
         stored as ``fields``; ``POST /api/drafts`` does not accept them. A
         draft id that is not ``draft-<12 hex>`` is 422.

         **Follow-ups.** ``metadata.follows_up`` names the entry this one
         follows up, on ``POST /api/entries`` and ``/upload`` and in the draft
         that ``entry_publish`` publishes. It must be an entry ARIEL holds and not the
         entry itself; otherwise the refusal names ``follows_up`` (REST:
         ``invalid_entry_field``; MCP: ``validation_error``). ``follows_up`` is
         reserved: an adapter cannot declare it. Both create endpoints return
         ``attachments_published``: null without attachments, ``false`` when
         the logbook did not take them and they stayed in ARIEL only.

         An entry-field failure on any of these endpoints returns one of three
         envelopes:

         .. code-block:: json

            {"detail": "...", "code": "invalid_entry_field", "field": "book"}

         is 422: a submitted value (or an options query's parent value) is
         invalid and the author can correct it.

         .. code-block:: json

            {"detail": "...", "code": "entry_field_options_unavailable", "field": "scan"}

         is 502: the adapter failed or took longer than 10 seconds to list a
         ``dynamic_select``'s choices. The detail is a generic message, not the
         adapter's error text.

         .. code-block:: json

            {"detail": "...", "code": "entry_fields_misdeclared"}

         is 500: the adapter declares its fields wrongly, and only a change to
         the adapter fixes it.

         **Filters and paging.** ``filters`` is a JSON object
         ``{"<dimension>": [values], "time": {"start": ..., "end": ...}}``: OR
         within a dimension, AND across, exact match ignoring case. The
         dimensions are ``author``, ``source_system``, ``tags``, each entry field
         declared ``facet: true``, and ``enhancement`` (enabled enhancement
         modules); ``capabilities.filter_dimensions`` lists them. On
         ``/api/search`` the filter is ANDed with the legacy author, source and
         date fields. ``/api/entries`` pages by an opaque ``cursor``: each page
         returns ``next_cursor``, null on the last; ``page`` keeps offset paging
         and returns no cursor. ``/api/search`` pages by ``offset`` with
         ``max_results`` as the page size and returns ``total``, ``total_exact``
         and ``has_more`` (``total_results`` equals ``total``). A refused
         filter or cursor, or ``page`` with ``cursor``, is 422
         ``{"detail": {"field": ..., "reason": ...}}``. ``/api/facets`` counts
         over the filter alone when ``q`` is blank, else over the mode's
         matches; it returns ``{available, dimensions, histogram, exact, tier}``
         and ``available: false`` when it cannot count.

         **Entry shapes.** An entry carries ``subject``, ``entry_url`` (null for
         an entry that exists only in ARIEL, or with no
         ``ariel.entry_url_template``) and ``enhancement_status``, where each
         enabled module also gets a ``state`` of ``pending``, ``stale``,
         ``failed``, ``complete`` or ``n/a``. ``view=row`` returns list rows
         instead (:class:`~osprey.interfaces.ariel.api.schemas.RowResponse`):
         no ``raw_text``, a ``snippet``, the entry's ``facets`` values and its
         ``enhancements`` states.

         ``GET /health`` at the root level is the one route the sign-in gate leaves open. It
         reports ``status`` (``healthy`` or ``degraded``), a fixed ``message``, ``config_status``,
         and a ``service`` object carrying ``entry_count``, ``last_ingestion``,
         ``enabled_search_modules`` and ``enabled_enhancement_modules``. ``service`` is ``null``
         when the panel runs without its search service or the store does not answer. Everything
         that names the store (its address, ``database_uri``, the embedding tables and the
         driver's ``errors``) is only on the signed-in ``/api/status``.

         **HealthResponse:**

         .. code-block:: json

            {
              "status": "healthy",
              "message": "ARIEL service healthy",
              "config_status": "ok",
              "service": {
                "entry_count": 48291,
                "last_ingestion": "2026-09-30T08:00:00Z",
                "enabled_search_modules": ["keyword"],
                "enabled_enhancement_modules": ["text_embedding"]
              }
            }

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
              "total": 1,
              "total_exact": true,
              "has_more": false,
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
              "errors": [],
              "enhancement_coverage": {
                "text_embedding": {"label": "Text embedding", "complete": 15102, "pending": 120, "stale": 0, "failed": 8, "n/a": 0}
              }
            }

         ``enhancement_coverage`` counts, per enabled module, the entries in each
         state; ``osprey ariel status --json`` reports it as ``coverage``. A
         failed or stale card's ``command`` is
         ``osprey ariel enhance --module <name> --entry <entry_id>``, or the
         ``osprey ariel reembed`` line for a stale ``text_embedding``.

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
              ],
              "filter_dimensions": [
                {"name": "author", "label": "Author", "kind": "builtin"}
              ],
              "enhancement_modules": [
                {"name": "image_caption", "label": "Image caption"}
              ]
            }

         **How it works:** The ``get_capabilities()`` function in :mod:`osprey.services.ariel_search.capabilities` iterates over enabled search modules from the registry. Each module provides a ``get_tool_descriptor()`` (for its description) and optionally ``get_parameter_descriptors()`` (for its tunable parameters).

      .. tab-item:: App Internals

         **App factory:** The ``create_app()`` function in :mod:`osprey.interfaces.ariel.app` is a standard FastAPI app factory. It accepts an optional ``config_path`` argument and returns a fully configured FastAPI application with API routes and static file serving. It registers no CORS middleware --- the app and its ``/api`` endpoints share one origin, so there is no cross-origin surface.

         **Lifespan management:** The app uses FastAPI's ``lifespan`` context manager to initialize the ``ARIELSearchService`` on startup and clean it up on shutdown. During initialization:

         1. **Registry bootstrap** --- pre-creates the framework registry singleton (without an application registry path) so that ARIEL's search module discovery works even when running outside a full Osprey application.

         2. **Config loading** --- searches for ``config.yml`` in four locations: the provided ``config_path``, the ``CONFIG_FILE`` environment variable, ``/app/config.yml`` (a hand-authored container mount), and the current directory. The DSN is then resolved by ``resolve_ariel_dsn``, which applies the ``ARIEL_DATABASE_HOST`` and ``ARIEL_DATABASE_PORT`` overrides for Docker networking.

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
              - Host the derived DSN dials (e.g., ``ariel-postgres``, the store's compose network alias). Default: ``localhost``
            * - ``ARIEL_DATABASE_PORT``
              - Port the derived DSN dials (e.g., ``5432``, the in-network port, where ``services.postgresql.port_host`` is the published one). Default: ``services.postgresql.port_host``

         Both overrides apply only where the DSN is *derived* from ``services.postgresql``. A DSN written out in ``ariel.database.uri``, or in the legacy ``connection_string``, is used verbatim. The ARIEL panel resolves its DSN through the same function, so it does not redirect an authored one either. A blank value counts as unset. A non-integer ``ARIEL_DATABASE_PORT`` is refused rather than ignored.

      .. tab-item:: Frontend

         The frontend is a vanilla JavaScript SPA --- no build tools, no framework, no transpilation. All files are served as static assets from :file:`src/osprey/interfaces/ariel/static/`.

         **JavaScript modules:**

         .. list-table::
            :header-rows: 1
            :widths: 25 75

            * - Module
              - Responsibility
            * - ``app.js``
              - Boot: capabilities, theme, then every pane; the router starts last
            * - ``store.js``, ``router.js``
              - One state object, and its two-way mapping to the address hash
            * - ``layout.js``
              - Rail and list splitters, stacked mode, drawers, gear menu
            * - ``rail.js``, ``list.js``, ``reader.js``
              - The three panes: mailboxes and saved searches, the entry list
                with search box and mode chip, the entry and picture viewer
            * - ``facets.js``, ``paging.js``
              - The rail's facet counts and day histogram; Load more
            * - ``filter-state.js``, ``query-ops.js``
              - The filter object, and ``dim:value`` operators in the search box
            * - ``saved-searches.js``, ``display-prefs.js``
              - Saved searches in browser storage; the badge-strip preference
            * - ``keyboard.js``, ``header.js``
              - Key bindings; the embedded tile-bar contribution
            * - ``api.js``
              - REST client wrapping ``fetch()`` for all API endpoints
            * - ``search.js``
              - Search driver: runs a (two-phase) search and reports results
            * - ``advanced-options.js``
              - Capabilities-driven filters panel (dynamic parameter controls)
            * - ``entries.js``, ``entries-form.js``, ``entries-helpers.js``
              - The compose sheet and its draft loading and submission
            * - ``compose-autosave.js``, ``compose-lifecycle.js``, ``drafts-list.js``
              - Browser-local compose autosave; the Drafts mailbox
            * - ``entry-fields.js``
              - The new entry form's facility-declared field sections: renders
                them from ``/api/publish-info``, refetches a ``dynamic_select``'s
                choices when a field it depends on changes, and marks the input
                a 422 response names
            * - ``dashboard.js``
              - Status drawer rendering and periodic health refresh
            * - ``components.js``
              - Shared UI components (list rows, banners, loading and error states)
            * - ``utils.js``
              - Small shared helpers
            * - ``settings.js``
              - Settings UI: read/write ARIEL config block via ``/api/config``

         Side-drawer behavior (the Status and Settings drawers) comes
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
              - Cards, buttons, badges, chips, search results
            * - ``layout.css``
              - Header, rail | list | reader panes, stacked mode
            * - ``drawer.css``
              - Page-level styling around the shared drawer component
            * - ``settings.css``
              - Settings drawer styling (config form, YAML editor, save bar)

         Design tokens (colors, spacing, typography, transitions) are not
         defined per page — they come from the shared design system's
         ``/design-system/css/tokens.css``.

         **Routing:** The hash is the screen's state, ``#q=&mode=&box=&f=&sel=&att=&compose=``
         (empty keys omitted; ``box`` is ``all``, ``drafts``, ``mine`` or
         ``saved:<name>``; ``f`` is the filter as JSON; ``compose`` is
         ``draft:<id>`` for an open draft only), kept in step with ``history.replaceState``. Legacy
         aliases are applied and then rewritten canonical: ``#entry?id=&attachment=``
         selects the entry and opens the picture, ``#create[?draft=]`` opens
         the compose sheet, ``#status`` opens the Status drawer (embedded: the list), and
         ``#search``, ``#browse`` or an unknown hash show the list.

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
   * - ``subject``
     - ``TEXT``
     - Subject line derived from the entry (``''`` when it has none)
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

Migrations are run via ``osprey ariel migrate`` and managed by the ``run_migrations()`` function in ``database/migrations.py``. When ``text_embedding`` is enabled, ``migrate`` creates an embedding table for each model listed in ``enhancement_modules.text_embedding.models`` (falling back to ``nomic-embed-text``, 768, if none is configured); ``osprey ariel reembed`` (re)creates and backfills a single model's table on demand. ``migrate`` also adds the ``subject`` column and fills it for existing entries.

.. admonition:: Schema Evolution
   :class: outreach

   The current schema was generalised from the logbook formats the shipped ingestion adapters read, and may not capture every field your facility needs. The ``metadata`` JSONB column provides flexibility for facility-specific extras, but if your logbook requires a fundamentally different table structure, please open a pull request or contact us --- the ingestion and storage layers are designed to accommodate new schemas without disrupting existing ones.
