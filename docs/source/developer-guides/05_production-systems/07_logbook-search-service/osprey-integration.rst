===================
Osprey Integration
===================

ARIEL is not a standalone search engine --- it is a capability within Osprey's agent orchestration system. When a user asks a question like "What happened with the RF cavity last week?", Osprey's task classifier routes the request to the ``logbook_search`` capability, which invokes the ARIELSearchService and returns structured results that the response generator uses to produce a cited answer. This page documents how that integration works: the capability class, the context it produces, the prompt builder that customizes agent behavior, and the error classification that drives recovery strategies.

Integration Architecture
========================

.. code-block:: text

   User Query
       ↓
   Task Classifier (uses ClassifierGuide)
       ↓  routes to logbook_search
   Orchestrator (uses OrchestratorGuide)
       ↓  plans: [time_range_parsing → logbook_search → respond]
   LogbookSearchCapability.execute()
       ├── get TIME_RANGE context (soft constraint)
       ├── build ARIELSearchRequest
       └── call ARIELSearchService.search()
           ↓
   LogbookSearchResultsContext
       ↓
   RespondCapability → User Response

The flow begins when the task classifier determines that a user query involves historical logbook data. The classifier uses a ``TaskClassifierGuide`` --- a set of labeled examples that teach the LLM which queries should activate this capability. Once classified, the orchestrator creates an execution plan. If the query contains time references ("last week", "in January"), the plan includes a ``time_range_parsing`` step before the logbook search; otherwise, the search runs without a time filter. The ``LogbookSearchCapability`` executes the search and stores the results as a ``LogbookSearchResultsContext``, which the ``RespondCapability`` uses to generate the final user-facing response.


The ``logbook_search`` Capability
=================================

The :class:`~osprey.capabilities.logbook_search.LogbookSearchCapability` bridges Osprey's orchestration layer with the ARIELSearchService. It is registered as a built-in capability and auto-discovered by the registry.

.. list-table::
   :widths: 25 75

   * - **Name**
     - ``logbook_search``
   * - **Provides**
     - ``LOGBOOK_SEARCH_RESULTS``
   * - **Requires**
     - ``[]`` (empty --- ``TIME_RANGE`` is consumed via soft constraint)
   * - **Description**
     - Search and query historical logbook entries. Use when the user asks about past events, equipment history, operational incidents, or wants to find specific logbook entries by keyword or time period.

**Execution flow:**

1. **Get time range** --- calls ``get_required_contexts(constraint_mode="soft")`` to retrieve any ``TIME_RANGE`` context produced by an earlier planning step. If no time range was parsed, the search runs unfiltered.

2. **Build request** --- extracts the user's query from the task objective and constructs an ``ARIELSearchRequest`` with ``mode=RAG`` and an optional ``time_range`` tuple.

3. **Get service** --- calls ``get_ariel_search_service()`` to obtain the singleton ``ARIELSearchService`` instance (lazily initialized from ``config.yml``).

4. **Execute search** --- invokes ``service.search()`` with the request parameters. The service routes through the RAG pipeline by default: retrieve → fuse → assemble → generate.

5. **Store context** --- wraps the ``ARIELSearchResult`` in a ``LogbookSearchResultsContext`` and stores it for downstream capabilities.

**Source:** :file:`src/osprey/capabilities/logbook_search.py`


``LogbookSearchResultsContext``
===============================

The capability produces a :class:`~osprey.capabilities.logbook_search.LogbookSearchResultsContext` --- a frozen dataclass that extends :class:`~osprey.context.base.CapabilityContext`.

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
     - RAG-generated answer (if RAG was used)
   * - ``sources``
     - ``tuple[str, ...]``
     - Entry IDs cited in the answer
   * - ``search_modes_used``
     - ``tuple[str, ...]``
     - Search modes invoked (e.g., ``("keyword", "rag")``)
   * - ``query``
     - ``str``
     - Original query text
   * - ``time_range_applied``
     - ``bool``
     - Whether a time filter was used

.. list-table::
   :widths: 25 75

   * - ``CONTEXT_TYPE``
     - ``"LOGBOOK_SEARCH_RESULTS"``
   * - ``CONTEXT_CATEGORY``
     - ``"DATA"``

**``get_summary()``** returns both metadata (entry count, search modes, time filter) and content (the RAG answer, source IDs, and entry previews truncated to 200 characters). This follows the same pattern as ``PythonResultContext`` --- including actual content in the summary ensures the ``RespondCapability`` has enough material to generate a meaningful user response without needing to access the full context object.

**``get_access_details(key)``** provides instructions for other capabilities to access the results programmatically, following the standard ``context.LOGBOOK_SEARCH_RESULTS.<key>`` access pattern.


Orchestrator & Classifier Guides
=================================

The ``logbook_search`` capability provides two guidance objects that teach Osprey's LLM-based planner and classifier how to use it. Both are loaded from the prompt builder system, which allows facility-specific customization.

.. tab-set::

   .. tab-item:: Orchestrator Guide

      The ``OrchestratorGuide`` teaches the planning LLM *when* and *how* to include logbook search in execution plans.

      **Default instructions** (from ``DefaultLogbookSearchPromptBuilder``):

      - Plan ``logbook_search`` when users ask about facility history, past incidents, equipment behavior, or operational procedures
      - Use descriptive context keys: ``ls_<topic>`` (e.g., ``ls_injector_failures``, ``ls_rf_trips``)
      - Plan ``time_range_parsing`` *before* ``logbook_search`` when the query contains time references
      - The internal ReAct agent automatically selects the best search strategy

      **Examples** included in the guide:

      1. Semantic: "Search logbook for historical injector failure incidents" (with optional TIME_RANGE input)
      2. Keyword: "Find logbook entries containing 'BTS chicane alignment'" (no dependencies)
      3. RAG: "Answer: How do we typically handle RF trips?" (generates cited answer)

   .. tab-item:: Classifier Guide

      The ``TaskClassifierGuide`` teaches the routing LLM which queries should activate this capability.

      **Examples:**

      .. list-table::
         :header-rows: 1
         :widths: 50 10 40

         * - Query
           - Result
           - Reason
         * - "What happened last time the injector failed?"
           - True
           - Request for historical incident information
         * - "Find entries about BTS chicane alignment"
           - True
           - Explicit request to search logbook entries
         * - "What is the current beam energy?"
           - False
           - Request for live data, not historical search
         * - "How do we typically handle RF trips?"
           - True
           - Operational knowledge documented in logs
         * - "Set the quadrupole current to 5A"
           - False
           - Control action, not logbook search
         * - "Show me beam loss events from January"
           - True
           - Historical entries with time filter
         * - "What did the night shift report about the vacuum issue?"
           - True
           - Operational shift report from logbook
         * - "How does the accelerator work?"
           - False
           - General physics question, not logbook search

**Customizing guides:**

To provide facility-specific guidance, create a custom prompt builder that extends ``DefaultLogbookSearchPromptBuilder`` and override ``get_orchestrator_guide()`` or ``get_classifier_guide()``. Register it through your application's prompt configuration. See :doc:`/developer-guides/03_core-framework-systems/04_prompt-customization` for details.

.. admonition:: Collaboration Welcome
   :class: outreach

   If you write a custom prompt builder with classifier examples or orchestrator instructions tuned to your facility, consider opening a pull request so other sites can benefit from your improvements. Additional classifier examples improve routing accuracy for all users.


.. dropdown:: Service Factory & Prompt Builder Details

   Prompt Builder
   --------------

   The :class:`~osprey.prompts.defaults.logbook_search.DefaultLogbookSearchPromptBuilder` provides the default orchestrator and classifier guidance for the ``logbook_search`` capability. It extends :class:`~osprey.prompts.base.FrameworkPromptBuilder` and is loaded automatically through the prompt builder system.

   .. list-table::
      :widths: 30 70

      * - ``PROMPT_TYPE``
        - ``"logbook_search"``
      * - ``get_role_definition()``
        - "You are a facility logbook search expert that finds relevant historical entries."
      * - ``get_instructions()``
        - Instructions for search intent analysis, strategy selection, and citation requirements.
      * - ``get_orchestrator_guide()``
        - Returns the ``OrchestratorGuide`` described above.
      * - ``get_classifier_guide()``
        - Returns the ``TaskClassifierGuide`` described above.

   **Fallback behavior:** If the prompt builder system is unavailable (e.g., in a minimal deployment without a registry), the ``LogbookSearchCapability`` falls back to built-in default guides defined in ``_default_orchestrator_guide()`` and ``_default_classifier_guide()``. These provide the same core guidance without the extended examples.

   **Source:** :file:`src/osprey/prompts/defaults/logbook_search.py`

   Service Factory
   ---------------

   The ``get_ariel_search_service()`` function in :mod:`osprey.services.ariel_search.capability` provides a singleton ``ARIELSearchService`` instance. The service is lazily initialized from ``config.yml`` on first access and reused for subsequent calls.

   .. code-block:: python

      from osprey.services.ariel_search.capability import get_ariel_search_service

      service = await get_ariel_search_service()
      async with service:
          result = await service.search(query="RF cavity fault")

   **Lifecycle:** The singleton is created once per process. In the web interface, the ``create_app()`` factory manages its own service instance through the FastAPI lifespan. For cleanup in tests, use ``close_ariel_service()`` (closes the connection pool) or ``reset_ariel_service()`` (resets without closing).

   **Source:** :file:`src/osprey/services/ariel_search/capability.py`


Error Classification
====================

The ``classify_error()`` static method maps ARIEL exceptions to structured ``ErrorClassification`` objects that drive Osprey's recovery system. Each classification includes a severity level and an actionable ``user_message`` that helps users resolve common setup issues.

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

The severity level determines how Osprey handles the error: ``CRITICAL`` errors are surfaced to the user immediately with the guidance message, while ``RETRIABLE`` errors trigger automatic retry with backoff.

.. dropdown:: Exception Hierarchy

   All ARIEL exceptions inherit from :class:`~osprey.services.ariel_search.exceptions.ARIELException`, which carries a ``message``, an ``ErrorCategory``, and optional ``technical_details``. The ``is_retriable`` property returns ``True`` for ``DATABASE`` and ``EMBEDDING`` categories.

   .. list-table::
      :header-rows: 1
      :widths: 30 20 50

      * - Exception
        - Category
        - Description
      * - ``DatabaseConnectionError``
        - DATABASE
        - Unable to connect to the ARIEL PostgreSQL database
      * - ``DatabaseQueryError``
        - DATABASE
        - A database query failed during execution
      * - ``EmbeddingGenerationError``
        - EMBEDDING
        - Embedding model failed to generate vectors
      * - ``SearchExecutionError``
        - SEARCH
        - A search operation failed during execution
      * - ``IngestionError``
        - INGESTION
        - Data ingestion failed during processing
      * - ``AdapterNotFoundError``
        - INGESTION
        - Requested ingestion adapter is not registered
      * - ``ConfigurationError``
        - CONFIGURATION
        - ARIEL configuration is invalid
      * - ``ModuleNotEnabledError``
        - CONFIGURATION
        - Attempted to use a module that is not enabled
      * - ``SearchTimeoutError``
        - TIMEOUT
        - Search execution exceeded the configured timeout

   **Source:** :file:`src/osprey/services/ariel_search/exceptions.py` · :doc:`API Reference </api_reference/03_production_systems/07_ariel-search>`


See Also
========

:doc:`data-ingestion`
    How data gets into the system --- facility adapters, enhancement modules, and database schema

:doc:`search-modes`
    Search module and pipeline architecture

:doc:`web-interface`
    Web interface architecture and REST API

:doc:`/developer-guides/03_core-framework-systems/07_built-in-capabilities`
    Built-in capabilities reference

:doc:`/developer-guides/03_core-framework-systems/04_prompt-customization`
    Prompt builder customization

:doc:`/api_reference/03_production_systems/07_ariel-search`
    Full API reference for ARIEL classes
