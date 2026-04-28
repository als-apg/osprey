===================
Osprey Integration
===================

ARIEL is integrated into Osprey as a dedicated MCP server
(``osprey.mcp_server.ariel``) with 11 specialized tools. When a user asks a
question like "What happened with the RF cavity last week?", Claude Code
selects the appropriate ARIEL MCP tool based on the query type, which invokes
the ``ARIELSearchService`` and returns structured results that Claude uses to
produce a cited answer. This page documents how that integration works: the
MCP tools, the service factory, and the search result structure.

Integration Architecture
========================

.. code-block:: text

   User Query
       ↓
   Claude Code (via osprey claude chat)
       ↓  selects from ARIEL MCP tools
   ARIEL MCP Server → ARIELSearchService
       ↓
   Structured search results (entries + RAG answer)
       ↓
   Claude Code → User Response

The flow begins when Claude determines that a user query involves historical
logbook data. Claude selects from ARIEL's specialized MCP tools based on the
query type --- for example, ``keyword_search`` for exact-match lookups,
``semantic_search`` for conceptual queries, or ``browse`` for exploring recent
entries. Each tool builds the appropriate request and routes it through the
``ARIELSearchService``. Results are returned directly to Claude, which uses
them to generate a cited response.


ARIEL MCP Tools
===============

ARIEL exposes 11 tools through its dedicated MCP server. Claude selects the
appropriate tool based on the user's query.

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
     - Get available filter values (tags, authors, date ranges)
   * - ``status``
     - Check ARIEL service health and configuration
   * - ``entry_publish``
     - Publish a new logbook entry
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


See Also
========

:doc:`data-ingestion`
    How data gets into the system --- facility adapters, enhancement modules, and database schema

:doc:`search-modes`
    Search module and pipeline architecture

:doc:`web-interface`
    Web interface architecture and REST API
