============
Search Modes
============

ARIEL's search system is organized into two layers: **search modules** and **pipelines**. Search modules are leaf-level functions that each implement a single retrieval strategy --- keyword full-text search or embedding-based semantic similarity. Pipelines are higher-level execution strategies that compose one or more search modules with LLM reasoning to produce richer answers. At query time, the ``ARIELSearchService`` routes each request to one of four modes: the two search modules can be called directly for fast, focused lookups; the RAG pipeline retrieves, fuses, and generates a cited answer deterministically; and the agent mode lets a ReAct agent autonomously decide which tools to call and how many times. All four modes share the same underlying :ref:`database <database>` and produce a common ``ARIELSearchResult``.

Both layers are designed for extensibility. Search modules and pipelines are discovered through Osprey's :doc:`central registry </developer-guides/03_core-framework-systems/03_registry-and-discovery>`, so you can add your own without modifying any framework code. A custom search module only needs to export a ``get_tool_descriptor()`` function; a custom pipeline only needs to export a ``get_pipeline_descriptor()`` function. Once registered, they are automatically available in the agent executor, the RAG pipeline's retrieval list, and the web interface.

Search Architecture
-------------------

.. code-block:: text

   User Query
       ↓
   ARIELSearchService.search(mode=...)
       ├── KEYWORD  → keyword_search()      → ranked entries
       ├── SEMANTIC → semantic_search()      → ranked entries
       ├── RAG      → RAGPipeline.execute()
       │                ├── Retrieve (keyword ∥ semantic)
       │                ├── Fuse (Reciprocal Rank Fusion)
       │                ├── Assemble (token-aware context)
       │                └── Generate (LLM + citations)
       └── AGENT    → AgentExecutor.execute()
                        └── ReAct loop (search tools → synthesize)
       ↓
   ARIELSearchResult (entries, answer, sources, reasoning)

The service validates that the requested mode is enabled in configuration before routing. Keyword and semantic are direct function calls; RAG and agent instantiate their respective classes with the repository, config, and an embedder loader. All four modes return an ``ARIELSearchResult`` with entries, an optional LLM-generated answer, source entry IDs, and a reasoning string.


Search Modules
==============

Search modules are leaf-level functions that execute a single search strategy against the database. They are the building blocks that pipelines compose. Each module exports a ``get_tool_descriptor()`` function that describes its capabilities, input schema, and execution function so the rest of the system --- the agent executor, the RAG pipeline, and the web interface --- can discover and use it automatically. The framework ships with the following built-in search modules:

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
         c. Hardcoded default: 0.7 (lowest)

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
               similarity_threshold: 0.7
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

Once registered and enabled in ``config.yml`` (``search_modules.my_search.enabled: true``), the module is automatically available as a tool in the agent executor, as a retrieval source in any pipeline that lists it in ``retrieval_modules``, and as a search option in the web interface. The ``get_tool_descriptor()`` function must return a ``SearchToolDescriptor``:

:class:`~osprey.services.ariel_search.search.base.SearchToolDescriptor` — a frozen dataclass whose key fields are ``execute`` (the async search function), ``format_result`` (formats results for agent consumption), and ``args_schema`` (a Pydantic model for input validation). See the :doc:`ARIEL API reference </api_reference/03_production_systems/07_ariel-search>` for the full field list.

Modules may also export ``get_parameter_descriptors()`` to declare tunable parameters for the frontend capabilities API. Each :class:`~osprey.services.ariel_search.search.base.ParameterDescriptor` describes a single knob --- its name, type, default, range, and UI grouping --- so the web interface can render controls dynamically. See the :doc:`ARIEL API reference </api_reference/03_production_systems/07_ariel-search>` for the full field list.

.. admonition:: Collaboration Welcome
   :class: outreach

   If you implement a search module that could benefit other facilities --- for example, a structured-metadata search, a time-series correlation search, or a cross-entry linking search --- we encourage you to open a pull request so it becomes natively available in Osprey.


Pipelines
=========

Pipelines compose search modules into higher-level execution strategies. Each pipeline specifies which ``retrieval_modules`` it uses in configuration and is declared through a ``PipelineDescriptor`` that provides metadata and tunable parameters for the web interface. Like search modules, pipelines are discovered through the Osprey :doc:`registry </developer-guides/03_core-framework-systems/03_registry-and-discovery>` --- you can register your own execution strategy without modifying any framework code. The framework ships with the following built-in pipelines:

.. tab-set::

   .. tab-item:: RAG Pipeline

      **Module:** ``rag.py``

      Deterministic 4-stage pipeline for direct question answering. Produces auditable, reproducible results.

      **Stages:**

      1. **Retrieve** --- runs the pipeline's configured ``retrieval_modules`` (keyword and/or semantic) in parallel using ``asyncio.gather``. If a retrieval module fails, it logs a warning and continues with whichever modules succeeded.

      2. **Fuse** --- combines results using Reciprocal Rank Fusion (RRF). Each entry's score from a single source is ``1 / (k + rank + 1)`` where *k* is the fusion parameter (default: 60) and *rank* is the 0-based position in that source's result list. When an entry appears in both keyword and semantic results, its RRF scores are summed. When only one source returns results, they pass through directly with their single-source RRF scores.

      3. **Assemble** --- builds a token-aware context window from fused entries. Each entry is formatted as:

         .. code-block:: text

            ENTRY #<id> | <timestamp> | Author: <name> | <title>
            <content truncated to max_chars_per_entry>

         Entries are joined with ``---`` separators. Total context is limited to ``max_context_chars`` (default: 12,000). If the limit is reached mid-entry and at least 100 characters remain, the entry is truncated with ``...``; otherwise it is excluded entirely. The pipeline tracks whether truncation occurred.

      4. **Generate** --- sends the assembled context and query to the configured LLM using a prompt that instructs the model to cite entries using ``[#id]`` notation. The prompt, temperature, provider, and model are all configurable.

      **Citation extraction:** After generation, the pipeline extracts ``[#id]`` patterns from the answer to produce a list of unique source entry IDs in order of appearance. If no citations are found in the text, the pipeline falls back to citing all entries that were included in the context.

      **Configuration:**

      .. code-block:: yaml

         pipelines:
           rag:
             enabled: true
             retrieval_modules: [keyword, semantic]  # Which search modules to use

         reasoning:
           provider: cborg
           model_id: anthropic/claude-haiku
           temperature: 0.1

   .. tab-item:: Agent Mode

      **Module:** ``agent/executor.py``

      ReAct agent via LangGraph's ``create_react_agent``. The agent autonomously decides which search tools to call and how to synthesize an answer from multiple invocations. Unlike the RAG pipeline, the agent can refine its searches iteratively --- for example, broadening a query that returned no results, or issuing follow-up searches to corroborate initial findings.

      **How it works:**

      1. **Tool discovery** --- loads ``SearchToolDescriptor`` instances from the registry, filtered to the pipeline's configured ``retrieval_modules`` and further filtered to only those that are enabled. Each descriptor is wrapped into a LangChain ``StructuredTool`` with its Pydantic ``args_schema`` for input validation.

      2. **Time range resolution** --- each tool call uses a 3-tier priority for date filtering:

         a. Tool call parameter (highest) --- agent explicitly passes ``start_date``/``end_date``
         b. Request context --- from the capability's ``time_range``
         c. No filter (lowest) --- search all entries

      3. **Agent creation** --- creates a ReAct agent with a system prompt that instructs it to search the logbook, cite entry IDs, and synthesize findings. The LLM is lazy-loaded from Osprey's centralized provider configuration.

      4. **Execution** --- runs the agent with a configurable recursion limit (``max_iterations * 2 + 1``) and a total timeout enforced via ``asyncio.wait_for``. Timeout produces a graceful ``SearchTimeoutError`` rather than crashing.

      5. **Result parsing** --- extracts the final answer from the last AI message, citation IDs from ``[#id]`` patterns, and which search modes were actually invoked by inspecting the tool call history.

      **Configuration:**

      .. code-block:: yaml

         pipelines:
           agent:
             enabled: true
             retrieval_modules: [keyword, semantic]  # Which search tools the agent can use

         reasoning:
           provider: cborg
           model_id: anthropic/claude-haiku
           max_iterations: 5              # Maximum ReAct cycles
           temperature: 0.1
           total_timeout_seconds: 120     # Total agent timeout

.. admonition:: RAG vs. Agent
   :class: tip

   The RAG pipeline and agent mode are peers, not layers --- they are two independent execution strategies. **RAG** is deterministic and auditable: the same query always follows the same retrieve-fuse-assemble-generate path, making it ideal for operational use where reproducibility matters. **Agent** is exploratory and non-deterministic: the LLM decides what to search and may iterate, making it better for complex questions that require multi-step reasoning or query refinement.

**Registering a custom pipeline:**

To add your own execution strategy, create a Python module that exports ``get_pipeline_descriptor()``, then register it through your application's registry configuration:

.. code-block:: python

   from osprey.registry.helpers import extend_framework_registry
   from osprey.registry.base import ArielPipelineRegistration

   app_config = extend_framework_registry(
       ariel_pipelines=[
           ArielPipelineRegistration(
               name="my_pipeline",
               module_path="my_app.pipelines.my_pipeline",
               description="Custom execution strategy",
               category="llm",  # "llm" for LLM-powered, "direct" for deterministic
           ),
       ],
   )

The ``get_pipeline_descriptor()`` function must return a :class:`~osprey.services.ariel_search.pipelines.PipelineDescriptor` — a frozen dataclass with ``name``, ``label``, ``description``, ``category`` (``"llm"`` or ``"direct"``), and ``parameters`` (a list of :class:`~osprey.services.ariel_search.search.base.ParameterDescriptor`). See the :doc:`ARIEL API reference </api_reference/03_production_systems/07_ariel-search>` for the full field list.

Once registered, enable the pipeline in ``config.yml`` (``pipelines.my_pipeline.enabled: true``) and configure its ``retrieval_modules`` to control which search modules it uses. The pipeline will appear in the web interface alongside the built-in options.

.. admonition:: Collaboration Welcome
   :class: outreach

   The built-in pipelines cover the most common retrieval patterns, but there is plenty of room for new strategies --- for example, a multi-hop pipeline that chains searches based on intermediate results, a map-reduce pipeline that summarizes large result sets, or a hybrid pipeline that combines RAG with structured database queries. If you build a useful pipeline, we encourage you to open a pull request so it becomes natively available to all Osprey users.


See Also
========

:doc:`data-ingestion`
    How data gets into the system --- facility adapters, enhancement modules, and database schema

:doc:`osprey-integration`
    Capability, context flow, and error classification

:doc:`web-interface`
    Web interface architecture and capabilities API

:doc:`/api_reference/03_production_systems/07_ariel-search`
    Full API reference and YAML configuration

:doc:`/developer-guides/03_core-framework-systems/03_registry-and-discovery`
    Registry system used for search module and pipeline discovery

:doc:`/developer-guides/03_core-framework-systems/07_built-in-capabilities`
    ``logbook_search`` capability reference
