===================
Osprey Integration
===================

ARIEL is integrated into Osprey as an MCP tool exposed through the
``control_system`` MCP server. When a user asks a question like "What happened
with the RF cavity last week?", Claude Code routes the request to the
``ariel_search`` tool, which invokes the ARIELSearchService and returns
structured results that Claude uses to produce a cited answer. This page
documents how that integration works: the MCP tool, the service factory,
and the error classification that drives recovery strategies.

Integration Architecture
========================

.. code-block:: text

   User Query
       ↓
   Claude Code (via osprey claude chat)
       ↓  selects ariel_search MCP tool
   MCP Server → ARIELSearchService.search()
       ↓
   Structured search results (entries + RAG answer)
       ↓
   Claude Code → User Response

The flow begins when Claude determines that a user query involves historical
logbook data. Claude selects the ``ariel_search`` MCP tool, which builds an
``ARIELSearchRequest`` with ``mode=RAG`` and an optional ``time_range`` tuple.
The service routes through the RAG pipeline by default: retrieve → fuse →
assemble → generate. Results are returned directly to Claude, which uses
them to generate a cited response.

.. admonition:: PLACEHOLDER: CONCEPTUAL-MAPPING
   :class: warning

   **Old content (line 14):** "Task Classifier (uses ClassifierGuide) → Orchestrator (uses OrchestratorGuide) → LogbookSearchCapability.execute()"
   **New equivalent:** Needs human judgment
   **Why this is fuzzy:** The old orchestrator/classifier/capability pipeline is replaced by Claude Code's native tool selection, but the routing intelligence now lives in Claude's system prompt and MCP tool descriptions rather than explicit guide objects.
   **Action needed:** Confirm whether any classifier/orchestrator guidance is baked into the MCP tool description or system prompt, and document that instead.


The ``ariel_search`` MCP Tool
=============================

The ``ariel_search`` tool is registered in the ``control_system`` MCP server
and provides logbook search functionality to Claude Code sessions.

**Execution flow:**

1. **Parse parameters** --- extracts the user's query and optional time range
   from the tool call arguments.

2. **Get service** --- calls ``get_ariel_search_service()`` to obtain the
   singleton ``ARIELSearchService`` instance (lazily initialized from
   ``config.yml``).

3. **Execute search** --- invokes ``service.search()`` with the request
   parameters. The service routes through the RAG pipeline by default:
   retrieve → fuse → assemble → generate.

4. **Return results** --- returns structured results (entries, RAG answer,
   sources, metadata) to Claude Code.


Search Result Structure
=======================

The MCP tool returns a structured result containing:

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Field
     - Type
     - Description
   * - ``entries``
     - ``list[dict]``
     - Matching entries, ranked by relevance
   * - ``answer``
     - ``str | None``
     - RAG-generated answer (if RAG was used)
   * - ``sources``
     - ``list[str]``
     - Entry IDs cited in the answer
   * - ``search_modes_used``
     - ``list[str]``
     - Search modes invoked (e.g., ``["keyword", "rag"]``)
   * - ``query``
     - ``str``
     - Original query text
   * - ``time_range_applied``
     - ``bool``
     - Whether a time filter was used


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


Error Classification
====================

ARIEL exceptions are mapped to structured error responses that drive recovery.
Each classification includes a severity level and an actionable ``user_message``
that helps users resolve common setup issues.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Exception
     - Severity
     - User Guidance
   * - ``DatabaseConnectionError``
     - CRITICAL
     - "Run ``osprey deploy up`` to start the database, then ``osprey ariel migrate`` and ``osprey ariel ingest``."
   * - ``DatabaseQueryError`` (missing tables)
     - CRITICAL
     - "Run ``osprey ariel migrate`` to create tables, then ``osprey ariel ingest`` to populate data."
   * - ``DatabaseQueryError`` (other)
     - RETRIABLE
     - "Database query error, retrying..."
   * - ``EmbeddingGenerationError``
     - CRITICAL
     - "Embedding model unavailable. Disable semantic search if not needed."
   * - ``ConfigurationError``
     - CRITICAL
     - Includes the specific ``exc.message`` from the configuration validator.
   * - Other exceptions
     - CRITICAL
     - Includes ``str(exc)`` for debugging.

The severity level determines how the error is handled: ``CRITICAL`` errors
are surfaced to the user immediately with the guidance message, while
``RETRIABLE`` errors trigger automatic retry with backoff.

.. dropdown:: Exception Hierarchy

   All ARIEL exceptions inherit from
   :class:`~osprey.services.ariel_search.exceptions.ARIELException`, which
   carries a ``message``, an ``ErrorCategory``, and optional
   ``technical_details``. The ``is_retriable`` property returns ``True`` for
   ``DATABASE`` and ``EMBEDDING`` categories. Categories: ``DATABASE``
   (connection/query errors), ``EMBEDDING`` (vector generation),
   ``SEARCH`` (execution errors), ``INGESTION`` (data loading, missing
   adapters), ``CONFIGURATION`` (invalid config, disabled modules),
   ``TIMEOUT`` (search exceeded limit).

   **Source:** :file:`src/osprey/services/ariel_search/exceptions.py`


See Also
========

:doc:`data-ingestion`
    How data gets into the system --- facility adapters, enhancement modules, and database schema

:doc:`search-modes`
    Search module and pipeline architecture

:doc:`web-interface`
    Web interface architecture and REST API
