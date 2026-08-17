============
Search Modes
============

ARIEL's search system is built around **search modules** --- leaf-level functions that each implement a single retrieval strategy over the logbook. The framework ships three: keyword full-text search, embedding-based semantic similarity, and ``qmd``, a hybrid of the two answered by a separate search sidecar container. All three produce a common ``ARIELSearchResult``. Higher-level reasoning over results --- multi-step retrieval, answer synthesis, custom prompting --- lives in the Osprey agent layer, which calls these search modules through ARIEL's MCP tools.

**Dispatch is registry-driven.** A search request names a mode as a plain string --- ``"keyword"``, ``"semantic"``, ``"qmd"``. The ``ARIELSearchService`` looks that name up in Osprey's central registry and calls the module's own ``execute``; it carries no per-mode branch of its own. The registry is the only source of routable modes, so the service, the web interface's capabilities API and the agent's MCP tools cannot disagree about which modes exist, and adding a module needs no change to the service.

.. note::

   ``sql_query`` is a **tool, not a search mode.** It runs read-only SQL against the same database, which is precision filtering --- exact matches, exhaustive date and author ranges, counts --- not ranked retrieval. It has no relevance score to merge with the others, so it is exposed only as its own MCP tool and never appears as a ``--mode`` value.

Search Architecture
-------------------

.. code-block:: text

   User Query
       ↓
   ARIELSearchService.search(mode="keyword" | "semantic" | "qmd" | ...)
       ↓
   registry lookup  →  that module's execute()
       ↓
   ARIELSearchResult (entries, search_modes_used)

The service refuses a mode that is not registered, or is registered but disabled in configuration, rather than quietly falling back to another one.

**CLI usage:**

.. code-block:: bash

   osprey ariel search "RF cavity fault"                  # default: keyword
   osprey ariel search "RF cavity fault" --mode keyword
   osprey ariel search "RF cavity fault" --mode semantic
   osprey ariel search "RF cavity fault" --mode qmd

The ``--mode`` choices are read from the registry when the command runs, so a facility that registers its own search module gets it as a choice --- and in ``--help`` --- without any code change.


Search Modules
==============

Search modules are leaf-level functions that execute a single search strategy against the database. Each module exports a ``get_tool_descriptor()`` function that describes its capabilities, input schema, and execution function. The web interface discovers modules through this descriptor via ARIEL's capabilities API; each built-in module is exposed to the Osprey agent through its own ARIEL MCP tool (``keyword_search``, ``semantic_search``, ``qmd_search``). The framework ships with the following built-in search modules:

.. tab-set::

   .. tab-item:: Keyword Search

      **Module:** ``search/keyword.py``

      PostgreSQL full-text search with optional fuzzy matching fallback. Best for specific terms, equipment names, PV names, and exact phrases.

      **Query syntax:**

      .. code-block:: text

         # Simple terms (implicit AND)
         RF cavity fault

         # Boolean operators
         RF AND cavity
         vacuum OR pressure
         beam NOT injection

         # Quoted phrases
         "RF cavity trip"

         # Field prefixes
         author:smith
         date:2024-06

         # Combined
         author:jones "beam loss" date:2024-01

      **How it works:**

      1. Validates and preprocesses the query --- empty queries return immediately, queries longer than 1,000 characters are truncated, and unbalanced quotes are auto-balanced by removing the last unmatched quote
      2. Parses the query to extract field filters (``author:``, ``date:``), quoted phrases, and remaining search terms
      3. Builds a PostgreSQL ``tsquery`` using the function appropriate for the query shape:

         - ``plainto_tsquery`` --- for simple terms (implicit AND)
         - ``websearch_to_tsquery`` --- for queries with Boolean operators (AND, OR, NOT)
         - ``phraseto_tsquery`` --- for quoted phrases

         When multiple components are present (e.g. terms *and* phrases), they are combined with ``&&`` (tsquery AND).

      4. Executes full-text search against the ``raw_text`` column with ``ts_rank`` scoring, applying any field filters (``author ILIKE``, date range) and time range constraints
      5. If no results and fuzzy fallback is enabled, falls back to ``pg_trgm`` trigram similarity (default threshold: 0.3)
      6. Returns results as ``(entry, score, highlights)`` tuples --- highlights are generated via ``ts_headline``

      **Configuration:**

      .. code-block:: yaml

         ariel:
           search_modules:
             keyword:
               enabled: true

   .. tab-item:: Semantic Search

      **Module:** ``search/semantic.py``

      Embedding-based similarity search using pgvector. Best for conceptual queries where exact keywords may not appear in the text.

      **How it works:**

      1. Resolves the similarity threshold using a 3-tier priority:

         a. Per-query ``similarity_threshold`` parameter (highest)
         b. Config value (``search_modules.semantic.settings.similarity_threshold``)
         c. Hardcoded default: 0.5 (lowest)

      2. Determines the embedding model from config (``search_modules.semantic.model``) and resolves provider credentials via Osprey's centralized ``api.providers`` configuration
      3. Generates a query embedding using the configured provider, with a dimension-mismatch warning if the returned embedding size does not match the configured ``embedding_dimension``
      4. Searches the per-model embedding table using cosine distance (``<=>`` operator)
      5. Filters results by similarity threshold and optional time range
      6. Returns results as ``(entry, similarity_score)`` tuples

      **Configuration:**

      .. code-block:: yaml

         ariel:
           search_modules:
             semantic:
               enabled: true
               provider: ollama
               model: nomic-embed-text
               settings:
                 similarity_threshold: 0.5
                 embedding_dimension: 768

      **Requirements:** Ollama (or another embedding provider) running with the configured model, embedding table populated via the ``text_embedding`` :ref:`enhancement module <Enhancement Pipeline>`, and the pgvector extension installed in PostgreSQL.

   .. tab-item:: qmd (Hybrid)

      **Module:** ``search/qmd.py``

      Hybrid keyword-plus-semantic search, answered by the **qmd search sidecar** --- a separate container that indexes a markdown mirror of the logbook and returns one merged ranking. Best when a question mixes specific terms with a described situation, or when keyword search returned too little.

      Unlike the other two modes, ``qmd`` does not search PostgreSQL. It needs two things running together:

      1. the ``services.qmd`` sidecar (see :ref:`qmd-search-sidecar`), and
      2. the ``qmd_export`` :ref:`enhancement module <Enhancement Pipeline>`, which writes the markdown mirror the sidecar indexes.

      Either one alone is useless: an export with no sidecar indexes nothing, and a sidecar with no export searches an empty corpus. Both are off by default.

      ``qmd`` also does not degrade the way semantic search does. A query against a sidecar that is not there is reported as *search is down*, deliberately, so that the agent cannot read an outage as "nothing matched".

      **Configuration:**

      .. code-block:: yaml

         ariel:
           search_modules:
             qmd:
               enabled: true
               settings:
                 rerank: true          # default
                 candidate_limit: 40   # default

      .. warning::

         The knobs **must** stay under ``settings:``. ARIEL's search-config loader keeps only ``enabled``, ``provider``, ``model`` and ``settings``, and drops every other key without a word. A knob written as a sibling of ``enabled`` is inert forever --- no error, no warning, just the defaults.

      **The rerank decision.** ``rerank`` turns on qmd's LLM reranker, which reorders candidates for quality. It is the single most important knob here, because it is the dominant latency term. Measured against a 134,996-entry logbook:

      .. list-table::
         :header-rows: 1
         :widths: 40 30 30

         * - Corpus
           - p95, ``rerank: false``
           - p95, ``rerank: true``
         * - 135,000 entries
           - 811 ms
           - 3927 ms
         * - 2,000 entries
           - --
           - 1587 ms

      Reranking costs roughly **4x** the query budget, and its cost barely grows with corpus size --- so no logbook is small enough to outrun it. ``qmd_search`` is an agent tool with no interactive budget to protect, so it ships with the quality path on. Set ``rerank: false`` for the fast path. (The OKF bundle, which backs an interactive panel, defaults the other way; see :doc:`../okf-bundle`.)

      ``candidate_limit`` is how many candidates the reranker considers. Lowering it trades recall for latency.

      **Filtering is best-effort.** ``qmd_search`` ranks the corpus first and applies the date, author and source filters *afterwards*, to the top of that ranking --- not inside the database. A selective filter can therefore return fewer entries than you asked for even when more matching entries exist. Read a short result set as "the ranked window ran out", not as "there is nothing else". When a filter has to be exhaustive, use ``keyword_search`` or ``sql_query``, which filter in the database.

      .. admonition:: Known limitation --- the first reranked query times out
         :class: warning

         The first query with ``rerank: true`` loads a 610 MB model on CPU, and that load exceeds the client's default timeout. A deployment that enables reranking gets a hard failure on its **first** query; subsequent queries are fine. Run one throwaway query after starting the sidecar, or start with ``rerank: false``.

      .. admonition:: Known limitation --- entry IDs that are not numeric
         :class: note

         qmd normalises the document paths it reports: ``_`` and ``%`` both become ``-``, runs collapse, and a leading one is dropped. ARIEL rehydrates each hit from the document's title rather than the reported path, so the entries you get back are correct. But two entry IDs that differ *only* in characters qmd collapses --- ``beam_current_setpoint`` and ``beam-current-setpoint``, say --- index as one document, and one of them becomes unreachable through this mode.

         Over the real 134,996-entry ALS logbook the measured collision rate is **0.0000%**: every ALS entry ID is a 4-6 digit decimal string, so no real pair can collide. This matters only for a facility whose entry IDs are not numeric.

**Registering a custom search module:**

To add your own search module, create a Python module that exports ``get_tool_descriptor()`` (and optionally ``get_parameter_descriptors()``), then register it through your application's registry configuration:

.. code-block:: python

   from osprey.registry.helpers import extend_framework_registry
   from osprey.registry.base import ArielSearchModuleRegistration

   app_config = extend_framework_registry(
       ariel_search_modules=[
           ArielSearchModuleRegistration(
               name="my_search",
               module_path="my_app.search.my_module",
               description="Custom search module for my facility",
           ),
       ],
   )

Once registered and enabled in ``config.yml`` (``search_modules.my_search.enabled: true``), the module is routable by name --- through ``osprey ariel search --mode my_search`` and through the web interface's capabilities API --- with no change to ``ARIELSearchService``. Making it callable by the Osprey agent additionally requires a matching ARIEL MCP tool (contributions welcome). The ``get_tool_descriptor()`` function must return a ``SearchToolDescriptor``, whose ``search_mode`` field is simply the registered module name:

:class:`~osprey.services.ariel_search.search.base.SearchToolDescriptor` — a frozen dataclass whose key fields are ``execute`` (the async search function), ``format_result`` (formats results for agent consumption), and ``args_schema`` (a Pydantic model for input validation). See the class definition in the source for the full field list.

Modules may also export ``get_parameter_descriptors()`` to declare tunable parameters for the frontend capabilities API. Each :class:`~osprey.services.ariel_search.search.base.ParameterDescriptor` describes a single knob --- its name, type, default, range, and UI grouping --- so the web interface can render controls dynamically.

.. admonition:: Collaboration Welcome
   :class: outreach

   If you implement a search module that could benefit other facilities --- for example, a structured-metadata search, a time-series correlation search, or a cross-entry linking search --- we encourage you to open a pull request so it becomes natively available in Osprey.


.. _semantic-mode-retirement:

The Semantic Mode's Planned Retirement
======================================

``qmd`` covers what ``semantic`` covers and more: it is hybrid rather than
vector-only, it needs no embedding provider and no pgvector extension, and it
keeps its index in its own container instead of in the logbook database. The
plan is therefore to retire the semantic leg once ``qmd`` has been through a
burn-in comparison against it in production.

Nothing has been removed and no date has been set. ``semantic`` is fully
supported in this release, and the burn-in is the gate --- if it does not show
``qmd`` matching or beating ``semantic`` on real queries, the semantic leg
stays. Treat this as a direction to plan for, not a deprecation to act on: keep
``semantic`` configured if you use it, and read the release notes before
assuming otherwise.


Need behavior beyond these search modules --- multi-step reasoning, answer
synthesis, custom prompting? That lives in the Osprey agent layer; see
:doc:`osprey-integration` under "Extending the integration."


See Also
========

:doc:`data-ingestion`
    How data gets into the system --- facility adapters, enhancement modules, and database schema

:doc:`osprey-integration`
    MCP tools, service factory, and search result structure

:doc:`web-interface`
    Web interface architecture and capabilities API
