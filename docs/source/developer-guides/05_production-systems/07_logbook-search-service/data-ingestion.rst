==============
Data Ingestion
==============

ARIEL's ingestion system converts facility-specific logbook data into a common schema and optionally enriches it through a pipeline of enhancement modules. Before ARIEL can search anything, logbook data must be ingested into its PostgreSQL database. Every facility stores its logbook entries differently --- different APIs, file formats, field names, and conventions. ARIEL's ingestion layer abstracts over these differences through pluggable `facility adapters`_ that normalize entries into a common schema and store them in the `database`_. After ingestion, optional `enhancement modules <Enhancement Pipeline_>`_ can enrich the stored entries with additional computed fields --- vector embeddings for semantic search, LLM-extracted keywords and summaries, or any other derived metadata. Enhancement is a separate step from ingestion: you can ingest first and enhance later, re-enhance with different models, or skip enhancement entirely if you only need keyword search.

Ingestion Architecture
----------------------

.. code-block:: text

   Source System (HTTP API / JSONL file)
           ↓
   Facility Adapter (BaseAdapter)
           ↓
   EnhancedLogbookEntry (TypedDict)
           ↓
   ARIELRepository.upsert_entry()
           ↓
   PostgreSQL (enhanced_entries table)
           ↓
   Enhancement Modules (optional)
       ├── TextEmbeddingModule → per-model embedding tables
       └── SemanticProcessorModule → keywords + summary fields

The ingestion pipeline follows a linear flow. A `facility adapter <Facility Adapters_>`_ connects to the source system --- whether that is a live HTTP API, a JSONL dump, or any other data source --- and yields entries one at a time as ``EnhancedLogbookEntry`` TypedDicts. Each entry carries a unique ID, timestamp, author, raw text, and a metadata dict for facility-specific fields. The ``ARIELRepository`` upserts these entries into the ``enhanced_entries`` table in PostgreSQL, deduplicating by entry ID so that re-running ingestion is safe and idempotent. Once the base entries are stored, optional `enhancement modules <Enhancement Pipeline_>`_ can be run as a separate step to compute additional derived fields --- embeddings, keywords, summaries, or any other enrichment --- and write them back to the `database`_.

.. admonition:: Batch and Live Ingestion
   :class: note

   ARIEL supports both **batch** and **live** ingestion. Use ``osprey ariel ingest``
   for one-time bulk imports and ``osprey ariel watch`` for continuous polling.
   See `Live Ingestion`_ below for watch-mode details.


.. _`facility adapters`:

Facility Adapters
=================

Every logbook system has its own API, data format, and naming conventions. Facility adapters encapsulate these differences behind a uniform interface so that the rest of ARIEL --- storage, enhancement, search --- never needs to know where the data came from. Each adapter connects to one source system, fetches entries within an optional time range, and yields them as ``EnhancedLogbookEntry`` TypedDicts that the repository can store directly. All adapters inherit from ``BaseAdapter`` and implement two required members:

.. code-block:: python

   class BaseAdapter(ABC):
       @property
       @abstractmethod
       def source_system_name(self) -> str:
           """Return the source system identifier."""

       @abstractmethod
       def fetch_entries(
           self,
           since: datetime | None = None,
           until: datetime | None = None,
           limit: int | None = None,
       ) -> AsyncIterator[EnhancedLogbookEntry]:
           """Yield entries from the source system."""

Adapters are discovered through Osprey's :doc:`central registry </developer-guides/03_core-framework-systems/03_registry-and-discovery>`. The framework ships with the following built-in adapters:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Adapter
     - Registry Name
     - Description
   * - **ALS eLog**
     - ``als_logbook``
     - Production adapter for the Advanced Light Source electronic logbook. Supports JSONL file and HTTP API modes with SOCKS proxy, time-windowed chunked requests, retry with backoff, and entry deduplication.
   * - **JLab Logbook**
     - ``jlab_logbook``
     - Schema-ready prototype for Jefferson Lab. Parses JLab JSON format into the common schema but does not yet implement the facility's native API protocol.
   * - **ORNL Logbook**
     - ``ornl_logbook``
     - Schema-ready prototype for Oak Ridge National Laboratory. Parses ORNL JSON format into the common schema but does not yet implement the facility's native API protocol.
   * - **Generic JSON**
     - ``generic_json``
     - Reads from a JSON or JSONL file with flexible field mapping. Useful for demos, testing, and facilities without a custom API.

**Registering a custom adapter:**

To add your own facility adapter, subclass ``BaseAdapter``, implement ``source_system_name`` and ``fetch_entries()``, and register it through your application's registry configuration:

.. code-block:: python

   from osprey.registry.helpers import extend_framework_registry
   from osprey.registry.base import ArielIngestionAdapterRegistration

   app_config = extend_framework_registry(
       ariel_ingestion_adapters=[
           ArielIngestionAdapterRegistration(
               name="my_facility",
               module_path="my_app.adapters.my_facility",
               class_name="MyFacilityAdapter",
               description="Adapter for My Facility's logbook system",
           ),
       ],
   )

Once registered, you can use your adapter by setting ``ariel.ingestion.adapter: my_facility`` in ``config.yml``. See the :doc:`ARIEL API reference </api_reference/03_production_systems/07_ariel-search>` for the full :class:`~osprey.services.ariel_search.ingestion.base.BaseAdapter` interface (including the optional ``count_entries()`` method) and the :class:`~osprey.services.ariel_search.models.EnhancedLogbookEntry` field reference.

.. admonition:: Collaboration Welcome
   :class: outreach

   The adapters above reflect the logbook schemas we have had access to so far. If you implement an adapter for your facility and test it successfully, we encourage you to open a pull request to make it natively available in Osprey --- this makes it easier for other sites running similar logbook systems to get started.


.. _`Enhancement Pipeline`:

Enhancement Pipeline
====================

Enhancement modules run after ingestion to add computed fields to stored entries. While the base ingestion captures the raw logbook text and metadata, enhancement modules derive additional structure from that text --- generating vector embeddings that enable semantic similarity search, using an LLM to extract keywords and summaries that improve search recall and RAG context quality, or performing any other analysis that produces useful derived data. Each module inherits from ``BaseEnhancementModule`` and is discovered through the Osprey :doc:`registry </developer-guides/03_core-framework-systems/03_registry-and-discovery>`. Because enhancement is decoupled from ingestion, you can ingest a large dataset first and enhance it later, swap out models without re-ingesting, or run only the modules you need. Run them with ``osprey ariel enhance``.

The built-in enhancement modules:

.. tab-set::

   .. tab-item:: Text Embedding

      **Module:** ``enhancement/text_embedding.py``

      Generates vector embeddings for each entry using a configurable embedding model. Embeddings are stored in dedicated per-model tables (e.g., ``embeddings_nomic_embed_text``), allowing multiple models to coexist.

      **Configuration:**

      .. code-block:: yaml

         enhancement_modules:
           text_embedding:
             enabled: true
             provider: ollama
             models:
               - name: nomic-embed-text
                 dimension: 768

      **Requirements:** Ollama (or another embedding provider) running with the specified model.

   .. tab-item:: Semantic Processor

      **Module:** ``enhancement/semantic_processor.py``

      Uses an LLM to extract keywords and generate summaries for each entry. These fields improve keyword search recall and RAG context quality.

      **Configuration:**

      .. code-block:: yaml

         enhancement_modules:
           semantic_processor:
             enabled: true
             provider: cborg
             model:
               provider: cborg
               model_id: anthropic/claude-haiku
               max_tokens: 256

**Registering a custom enhancement module:**

To add your own module, subclass ``BaseEnhancementModule``, implement the ``name`` property and ``enhance()`` method, and register it through your application's registry configuration:

.. code-block:: python

   from osprey.registry.helpers import extend_framework_registry
   from osprey.registry.base import ArielEnhancementModuleRegistration

   app_config = extend_framework_registry(
       ariel_enhancement_modules=[
           ArielEnhancementModuleRegistration(
               name="my_enhancer",
               module_path="my_app.enhancement.my_enhancer",
               class_name="MyEnhancerModule",
               description="Custom enhancement module",
               execution_order=30,  # Runs after built-in modules (10, 20)
           ),
       ],
   )

The ``execution_order`` field controls the order in which modules run during enhancement. Built-in modules use orders 10 (semantic processor) and 20 (text embedding). See the :doc:`ARIEL API reference </api_reference/03_production_systems/07_ariel-search>` for the full :class:`~osprey.services.ariel_search.enhancement.base.BaseEnhancementModule` interface, including ``configure()``, ``health_check()``, and the ``migration`` property.

.. admonition:: Collaboration Welcome
   :class: outreach

   The enhancement modules above are a starting point --- there is plenty of room for new modules (e.g., named-entity extraction, automatic tagging, cross-entry linking). If you build a useful enhancement module, we encourage you to open a pull request so it becomes natively available to all Osprey users.


.. _`live ingestion`:

Live Ingestion
==============

The ``osprey ariel watch`` command runs the same adapter and enhancement pipeline as batch ingestion, but continuously. It polls the configured source at a regular interval, using the ``ingestion_runs`` table to determine the since-timestamp automatically --- only entries newer than the last successful run are fetched. This makes live ingestion fully incremental and idempotent.

CLI Usage
~~~~~~~~~

.. code-block:: bash

   # Daemon mode --- poll using configured interval
   osprey ariel watch

   # Preview one cycle without storing anything
   osprey ariel watch --once --dry-run

   # Override poll interval to 5 minutes
   osprey ariel watch --interval 300

   # Override source URL
   osprey ariel watch -s https://api.example.com/logbook

All ``--source`` / ``-s`` and ``--adapter`` / ``-a`` options from ``osprey ariel ingest`` are also available to override configuration at the command line.

Configuration
~~~~~~~~~~~~~

Watch-mode settings live under the ``ingestion.watch`` key in your ARIEL config block:

.. code-block:: yaml

   ariel:
     ingestion:
       adapter: als_logbook
       source_url: https://api.example.com/logbook
       poll_interval_seconds: 3600  # Base poll interval (seconds)
       watch:
         require_initial_ingest: true
         max_consecutive_failures: 10
         backoff_multiplier: 2.0
         max_interval_seconds: 3600

.. list-table::
   :header-rows: 1
   :widths: 30 15 15 40

   * - Field
     - Type
     - Default
     - Description
   * - ``require_initial_ingest``
     - ``bool``
     - ``true``
     - Require at least one prior ``osprey ariel ingest`` run before watching
   * - ``max_consecutive_failures``
     - ``int``
     - ``10``
     - Stop the scheduler after this many consecutive poll failures
   * - ``backoff_multiplier``
     - ``float``
     - ``2.0``
     - Multiply the poll interval by this factor on each consecutive failure
   * - ``max_interval_seconds``
     - ``int``
     - ``3600``
     - Maximum poll interval after backoff (seconds)

The base poll interval is set by the parent ``poll_interval_seconds`` key (default ``3600``).

Backoff Behavior
~~~~~~~~~~~~~~~~

On consecutive failures the scheduler increases the poll interval exponentially:

::

   interval = poll_interval_seconds × backoff_multiplier ^ consecutive_failures

The computed interval is capped at ``max_interval_seconds``. After a successful poll the interval resets to the base ``poll_interval_seconds``. If the number of consecutive failures reaches ``max_consecutive_failures``, the scheduler logs an error and exits.

.. admonition:: Initial Ingest Required
   :class: tip

   By default, ``osprey ariel watch`` requires at least one prior ``osprey ariel ingest``
   run so that it has a since-timestamp to poll from. If no previous run is found the
   scheduler will log a message and skip the cycle. Set ``require_initial_ingest: false``
   in the ``watch`` config block to start polling from the beginning of time instead.


.. _`database`:

Database Schema
===============

All ingested and enhanced data lives in PostgreSQL. The core ``enhanced_entries`` table stores one row per logbook entry with the normalized fields that every adapter produces --- entry ID, timestamp, author, raw text, and a JSONB metadata column for facility-specific extras. Enhancement modules write their results either into columns on this same table (keywords, summaries) or into dedicated per-model tables (vector embeddings). The pgvector extension provides the ``vector`` column type and cosine-distance operators that power semantic search. All tables are created and updated automatically by ``osprey ariel migrate``, which reads the current configuration to determine which embedding tables need to exist.

.. admonition:: pgvector requirement
   :class: important

   The **pgvector** extension is required for semantic search. It is automatically installed in the Osprey-managed PostgreSQL container (``osprey deploy up``). For external databases, install it manually: ``CREATE EXTENSION IF NOT EXISTS vector;``

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

**Per-model embedding tables** (e.g., ``embeddings_nomic_embed_text``):

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Column
     - Type
     - Description
   * - ``entry_id``
     - ``TEXT PRIMARY KEY``
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
   * - ``entries_processed``
     - ``INTEGER``
     - Number of entries ingested
   * - ``source_system``
     - ``TEXT``
     - Source adapter name


Migration System
----------------

Migrations are run via ``osprey ariel migrate`` and managed by the ``run_migrations()`` function in ``database/migrate.py``. The migration system automatically creates embedding tables based on the ``enhancement_modules.text_embedding.models`` configuration.

.. admonition:: Schema Evolution
   :class: outreach

   The current schema was designed around three facility logbook formats (ALS, JLab, ORNL) and may not capture every field your facility needs. The ``metadata`` JSONB column provides flexibility for facility-specific extras, but if your logbook requires a fundamentally different table structure, please open a pull request or contact us --- the ingestion and storage layers are designed to accommodate new schemas without disrupting existing ones.


See Also
========

:doc:`search-modes`
    How search uses the ingested and enhanced data

:doc:`osprey-integration`
    Capability, context flow, and error classification

:doc:`/api_reference/03_production_systems/07_ariel-search`
    Full API reference and YAML configuration

:doc:`/developer-guides/02_quick-start-patterns/00_cli-reference`
    CLI reference for ``osprey ariel ingest``, ``osprey ariel enhance``, and other commands

:doc:`/developer-guides/03_core-framework-systems/07_built-in-capabilities`
    ``logbook_search`` capability reference
