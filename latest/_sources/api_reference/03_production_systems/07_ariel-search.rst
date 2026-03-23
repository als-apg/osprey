====================
ARIEL Search Service
====================

ARIEL (Agentic Retrieval Interface for Electronic Logbooks) is a logbook search service
built on Osprey's registry-based plugin architecture. It provides multiple search strategies
--- keyword, semantic, RAG, and agentic --- through pluggable search modules that are
discovered and composed at runtime. A capabilities-driven ingestion pipeline with
registry-discovered adapters and enhancement modules keeps the search index up to date,
while a deterministic execution pipeline coordinates retrieval, fusion, and answer generation
for each query.

.. seealso::

   :doc:`/developer-guides/05_production-systems/07_logbook-search-service/index`
      Architecture overview, design rationale, and getting started

   :doc:`/developer-guides/05_production-systems/07_logbook-search-service/search-modes`
      Search module implementation, pipeline registration, and mode selection

   :doc:`/developer-guides/05_production-systems/07_logbook-search-service/data-ingestion`
      Ingestion adapters, enhancement modules, scheduling, and database schema

   :doc:`/developer-guides/05_production-systems/07_logbook-search-service/osprey-integration`
      Capability, context flow, prompt builder, and error classification

   :doc:`/developer-guides/05_production-systems/07_logbook-search-service/web-interface`
      Web interface architecture, REST API, and capabilities API


Data Models
===========

.. currentmodule:: osprey.services.ariel_search.models

.. autoclass:: EnhancedLogbookEntry
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: ARIELSearchRequest
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: ARIELSearchResult
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: SearchMode
   :members:
   :show-inheritance:
   :no-index:


Search Module Interface
=======================

.. currentmodule:: osprey.services.ariel_search.search.base

.. autoclass:: SearchToolDescriptor
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: ParameterDescriptor
   :members:
   :show-inheritance:
   :no-index:


Keyword Search Module
=====================

.. currentmodule:: osprey.services.ariel_search.search.keyword

.. autofunction:: get_tool_descriptor

.. autofunction:: get_parameter_descriptors

.. autofunction:: keyword_search

.. autofunction:: format_keyword_result

.. autoclass:: KeywordSearchInput
   :members:
   :show-inheritance:
   :no-index:


Semantic Search Module
======================

.. currentmodule:: osprey.services.ariel_search.search.semantic

.. autofunction:: get_tool_descriptor

.. autofunction:: get_parameter_descriptors

.. autofunction:: semantic_search

.. autofunction:: format_semantic_result

.. autoclass:: SemanticSearchInput
   :members:
   :show-inheritance:
   :no-index:


Pipeline Interface
==================

.. currentmodule:: osprey.services.ariel_search.pipelines

.. autoclass:: PipelineDescriptor
   :members:
   :show-inheritance:
   :no-index:

.. autofunction:: get_pipeline_descriptor

.. autofunction:: get_pipeline_descriptors


Ingestion Interface
===================

.. currentmodule:: osprey.services.ariel_search.ingestion.base

.. autoclass:: BaseAdapter
   :members:
   :show-inheritance:
   :no-index:


Enhancement Interface
=====================

.. currentmodule:: osprey.services.ariel_search.enhancement.base

.. autoclass:: BaseEnhancementModule
   :members:
   :show-inheritance:
   :no-index:


Configuration
=============

YAML Reference
--------------

ARIEL is configured under the ``ariel:`` key in ``config.yml``. The configuration is parsed into the ``ARIELConfig`` dataclass hierarchy.

.. code-block:: yaml

   ariel:
     # --- Database (required) ---
     database:
       uri: postgresql://ariel:ariel@localhost:5432/ariel

     default_max_results: 10     # Default results per search (default: 10)
     cache_embeddings: true      # Cache embeddings in memory (default: true)

     # --- Ingestion ---
     ingestion:
       adapter: generic_json     # als_logbook | jlab_logbook | ornl_logbook | generic_json
       source_url: path/to/logbook.json
       poll_interval_seconds: 3600
       proxy_url: null           # SOCKS proxy (or ARIEL_SOCKS_PROXY env var)
       verify_ssl: false
       chunk_days: 365           # Days per API request window
       request_timeout_seconds: 60
       max_retries: 3
       retry_delay_seconds: 5

     # --- Search Modules (leaf-level search functions) ---
     # provider: references api.providers for credentials
     search_modules:
       keyword:
         enabled: true
       semantic:
         enabled: false          # Requires embedding model
         provider: ollama        # References api.providers.ollama
         model: nomic-embed-text # Which embedding model's table to query
         settings:
           similarity_threshold: 0.7
           embedding_dimension: 768

     # --- Pipelines (compose search modules) ---
     # retrieval_modules: which search modules each pipeline uses
     pipelines:
       rag:
         enabled: true
         retrieval_modules: [keyword]       # Add semantic when ready
       agent:
         enabled: true
         retrieval_modules: [keyword]       # Add semantic when ready

     # --- Enhancement Modules ---
     # Run during ingestion to enrich entries
     enhancement_modules:
       semantic_processor:
         enabled: false
         provider: cborg
         model:
           provider: cborg
           model_id: anthropic/claude-haiku
           max_tokens: 256
       text_embedding:
         enabled: false
         provider: ollama
         models:
           - name: nomic-embed-text
             dimension: 768

     # --- Embedding Provider (fallback) ---
     embedding:
       provider: ollama          # Default provider for modules without explicit provider

     # --- Reasoning (ReAct agent LLM) ---
     # provider: references api.providers for credentials
     reasoning:
       provider: cborg
       model_id: anthropic/claude-haiku
       max_iterations: 5         # Maximum ReAct cycles
       temperature: 0.1
       tool_timeout_seconds: 30  # Per-tool call timeout
       total_timeout_seconds: 120 # Total agent timeout


Config Dataclass Hierarchy
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Dataclass
     - Description
   * - ``ARIELConfig``
     - Root configuration; contains all sub-configs
   * - ``DatabaseConfig``
     - PostgreSQL connection URI
   * - ``SearchModuleConfig``
     - Per-module: enabled, provider, model, settings
   * - ``PipelineModuleConfig``
     - Per-pipeline: enabled, retrieval_modules, settings
   * - ``EnhancementModuleConfig``
     - Per-module: enabled, provider, models list, settings
   * - ``IngestionConfig``
     - Adapter type, source URL, polling, proxy, retry settings
   * - ``ReasoningConfig``
     - LLM provider, model, iteration limits, timeouts
   * - ``EmbeddingConfig``
     - Default embedding provider fallback
   * - ``ModelConfig``
     - Individual model name, dimension, max_input_tokens


Dataclass API
-------------

.. currentmodule:: osprey.services.ariel_search.config

.. autoclass:: ARIELConfig
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: SearchModuleConfig
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: PipelineModuleConfig
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: EnhancementModuleConfig
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: IngestionConfig
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: ReasoningConfig
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: EmbeddingConfig
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: DatabaseConfig
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: ModelConfig
   :members:
   :show-inheritance:
   :no-index:


.. seealso::

   :doc:`/developer-guides/05_production-systems/07_logbook-search-service/search-modes`
       Search modules, pipelines, and registration guide

   :doc:`/developer-guides/05_production-systems/07_logbook-search-service/data-ingestion`
       Ingestion adapters, enhancement modules, and database schema
