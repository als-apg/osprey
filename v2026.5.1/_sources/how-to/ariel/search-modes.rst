============
Search Modes
============

ARIEL's search system is built around **search modules** --- leaf-level functions that each implement a single retrieval strategy against the logbook database. The framework ships two modules out of the box: keyword full-text search and embedding-based semantic similarity. At query time, the ``ARIELSearchService`` routes each request to a specific module or runs both and merges the results in ``auto`` mode. All modes share the same underlying :ref:`database <database>` and produce a common ``ARIELSearchResult``. Higher-level reasoning over results --- multi-step retrieval, answer synthesis, custom prompting --- lives in the Osprey agent layer, which calls these search modules through ARIEL's MCP tools.

Search modules are discovered through Osprey's central registry, so you can add your own without modifying any framework code. A custom search module only needs to export a ``get_tool_descriptor()`` function. Once registered, it is automatically available to the Osprey agent through the ARIEL MCP server and in the web interface.

Search Architecture
-------------------

.. code-block:: text

   User Query
       ↓
   ARIELSearchService.search(mode=...)
       ├── KEYWORD  → keyword_search()       → ranked entries
       ├── SEMANTIC → semantic_search()      → ranked entries
       └── AUTO     → run enabled modules in parallel and merge
       ↓
   ARIELSearchResult (entries, search_modes_used)

The service validates that the requested mode is enabled in configuration before routing. Keyword and semantic are direct function calls; ``auto`` runs the enabled modules in parallel and returns a merged list. All modes return an ``ARIELSearchResult`` with entries and the list of search modes that were invoked.

**CLI usage:**

.. code-block:: bash

   osprey ariel search "RF cavity fault" --mode auto    # default
   osprey ariel search "RF cavity fault" --mode keyword
   osprey ariel search "RF cavity fault" --mode semantic

The ``--mode`` option accepts ``keyword``, ``semantic``, or ``auto`` (the
default). In ``auto`` mode, the service runs all enabled modules and merges
the results.


Search Modules
==============

Search modules are leaf-level functions that execute a single search strategy against the database. Each module exports a ``get_tool_descriptor()`` function that describes its capabilities, input schema, and execution function so the rest of the system --- the ARIEL MCP server and the web interface --- can discover and use it automatically. The framework ships with the following built-in search modules:

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

         search_modules:
           semantic:
             enabled: true
             provider: ollama
             model: nomic-embed-text
             settings:
               similarity_threshold: 0.5
               embedding_dimension: 768

      **Requirements:** Ollama (or another embedding provider) running with the configured model, embedding table populated via the ``text_embedding`` :ref:`enhancement module <Enhancement Pipeline>`, and the pgvector extension installed in PostgreSQL.

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

Once registered and enabled in ``config.yml`` (``search_modules.my_search.enabled: true``), the module is automatically available as an ARIEL MCP tool that the Osprey agent can call, and as a search option in the web interface. The ``get_tool_descriptor()`` function must return a ``SearchToolDescriptor``:

:class:`~osprey.services.ariel_search.search.base.SearchToolDescriptor` — a frozen dataclass whose key fields are ``execute`` (the async search function), ``format_result`` (formats results for agent consumption), and ``args_schema`` (a Pydantic model for input validation). See the class definition in the source for the full field list.

Modules may also export ``get_parameter_descriptors()`` to declare tunable parameters for the frontend capabilities API. Each :class:`~osprey.services.ariel_search.search.base.ParameterDescriptor` describes a single knob --- its name, type, default, range, and UI grouping --- so the web interface can render controls dynamically.

.. admonition:: Collaboration Welcome
   :class: outreach

   If you implement a search module that could benefit other facilities --- for example, a structured-metadata search, a time-series correlation search, or a cross-entry linking search --- we encourage you to open a pull request so it becomes natively available in Osprey.


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
