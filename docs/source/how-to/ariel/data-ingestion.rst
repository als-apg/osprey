==============
Data Ingestion
==============

ARIEL's ingestion system converts facility-specific logbook data into a common schema and optionally enriches it through a pipeline of enhancement modules. Before ARIEL can search anything, logbook data must be ingested into its PostgreSQL database. Logbook systems differ in their APIs, file formats, field names, and conventions; ARIEL's ingestion layer abstracts over these differences through pluggable `facility adapters`_ that normalize entries into a common schema and store them in the :doc:`database </reference/contracts/ariel>`. After ingestion, optional `enhancement modules <Enhancement Pipeline_>`_ can enrich the stored entries with additional computed fields --- vector embeddings for semantic search, LLM-extracted keywords and summaries, or any other derived metadata. Enhancement is a separate step from ingestion: you can ingest first and enhance later, re-enhance with different models, or skip enhancement entirely if you only need keyword search.

Ingestion Architecture
----------------------

.. raw:: html
   :file: ../../_diagrams/ariel-ingestion.html

The ingestion pipeline follows a linear flow. A `facility adapter <Facility Adapters_>`_ connects to the source system --- whether that is a live HTTP API, a JSONL dump, or any other data source --- and yields entries one at a time as ``EnhancedLogbookEntry`` TypedDicts. Each entry carries a unique ID, timestamp, author, raw text, and a metadata dict for facility-specific fields. The ``ARIELRepository`` upserts these entries into the ``enhanced_entries`` table in PostgreSQL, deduplicating by entry ID so that re-running ingestion is safe and idempotent. Once the base entries are stored, optional `enhancement modules <Enhancement Pipeline_>`_ can be run as a separate step to compute additional derived fields --- embeddings, keywords, summaries, or any other enrichment --- and write them back to the :doc:`database </reference/contracts/ariel>`.

.. admonition:: Batch and Live Ingestion
   :class: note

   ARIEL supports both **batch** and **live** ingestion. Use ``osprey ariel ingest``
   for one-time bulk imports and ``osprey ariel watch`` for continuous polling.
   See `Live Ingestion`_ below for watch-mode details.


.. _`facility adapters`:

Facility Adapters
=================

Every logbook system has its own API, data format, and naming conventions. Facility adapters encapsulate these differences behind a uniform interface so that the rest of ARIEL --- storage, enhancement, search --- never needs to know where the data came from. Each adapter connects to one source system, fetches entries within an optional time range, and yields them as ``EnhancedLogbookEntry`` TypedDicts that the repository can store directly. All adapters inherit from ``FacilityAdapter`` and implement two required members --- a source-system name and an entry generator. Writing one for a logbook Osprey does not ship is a developer task: the base class, the registration and the test that pins them are the ARIEL seam in :doc:`/contributing/extending-osprey`.

Adapters are discovered through Osprey's central registry. The framework ships with the following built-in adapters:

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
     - Reads from a JSON file with flexible field mapping. Useful for demos, testing, and facilities without a custom API.

**Using a custom adapter:**

An adapter written and registered as described in :doc:`/contributing/extending-osprey` is selected the same way as a built-in one: set ``ariel.ingestion.adapter`` to its registered name in ``config.yml``.

.. admonition:: Collaboration Welcome
   :class: outreach

   The adapters above reflect the logbook schemas we have had access to so far. If you implement an adapter for your facility and test it successfully, we encourage you to open a pull request to make it natively available in Osprey --- this makes it easier for other sites running similar logbook systems to get started.


.. _`Enhancement Pipeline`:

Enhancement Pipeline
====================

Enhancement modules run after ingestion to add computed fields to stored entries. While the base ingestion captures the raw logbook text and metadata, enhancement modules derive additional structure from that text --- generating vector embeddings that enable semantic similarity search, using an LLM to extract keywords and summaries that improve search recall and the quality of the context the agent layer surfaces, or performing any other analysis that produces useful derived data. Each module inherits from ``BaseEnhancementModule`` and is discovered through the Osprey registry. Because enhancement is decoupled from ingestion, you can ingest a large dataset first and enhance it later, swap out models without re-ingesting, or run only the modules you need. Run them with ``osprey ariel enhance``.

The built-in enhancement modules:

.. tab-set::

   .. tab-item:: Text Embedding

      **Module:** ``enhancement/text_embedding/`` (entry point: ``embedder.py``)

      Generates vector embeddings for each entry using a configurable embedding model. Embeddings are stored in dedicated per-model tables (e.g., ``text_embeddings_nomic_embed_text``), allowing multiple models to coexist.

      **Configuration:**

      .. code-block:: yaml

         ariel:
           enhancement_modules:
             text_embedding:
               enabled: true
               provider: ollama
               models:
                 - name: nomic-embed-text
                   dimension: 768

      **Requirements:** Ollama (or another embedding provider) running with the specified model.

   .. tab-item:: Semantic Processor

      **Module:** ``enhancement/semantic_processor/`` (entry point: ``processor.py``)

      Uses an LLM to extract keywords and generate summaries for each entry. These fields improve keyword search recall and the quality of context the agent layer surfaces over results.

      **Configuration:**

      .. code-block:: yaml

         ariel:
           enhancement_modules:
             semantic_processor:
               enabled: true
               provider: cborg
               model:
                 model_id: anthropic/claude-haiku
                 max_tokens: 256

   .. tab-item:: qmd Export

      **Module:** ``enhancement/qmd_export/`` (entry point: ``exporter.py``)

      Writes one markdown file per entry into a **mirror tree** --- the corpus the qmd search sidecar indexes. It is what makes the ``hybrid`` :doc:`search mode <search-modes>` able to answer anything; the shipped templates enable both halves together, and they have to stay that way --- either one alone is useless. Entries created through the ARIEL panel or the agent's ``entry_create`` tool are mirrored inline at creation time (best-effort), so they become hybrid-searchable without waiting for the next enhancement run.

      **Configuration:**

      .. code-block:: yaml

         ariel:
           enhancement_modules:
             qmd_export:
               enabled: true
               settings:
                 mirror_path: var/ariel_mirror

      ``mirror_path`` is resolved against the project root --- the deployment repo, not the ``build/`` render inside it, which is exactly where the qmd sidecar bind-mounts the same path from. Keep it under ``var/``: the mirror is machine-written from PostgreSQL, it is as large as the logbook, and ``var/`` is the directory git ignores --- a path under ``data/`` would commit a generated corpus. An enabled export with no ``mirror_path`` is refused at startup rather than skipped, because a mirror nobody writes looks exactly like "search returns nothing".

      **Requirements:** the ``services.qmd`` sidecar, which bind-mounts this same directory read-only. See :ref:`qmd-search-sidecar`.

**Using a custom enhancement module:**

A module of your own runs alongside the built-in ones once it is registered --- see :doc:`/contributing/extending-osprey`. Its registration carries an ``execution_order`` that decides where in the run it lands; the built-ins use 10 (semantic processor), 20 (text embedding) and 30 (qmd export), so a value above 30 runs last.

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

The PostgreSQL schema this data lands in --- the ``enhanced_entries`` table, the
per-model embedding tables, and the migrations that create them --- is documented
in :doc:`/reference/contracts/ariel`.


Deployed Ingestion
~~~~~~~~~~~~~~~~~~

``osprey ariel watch`` keeps the mirror fresh only while you leave the command
running. A deployment needs the same loop with no terminal attached, so Osprey
ships a service that runs it in a container. Declare it in the build profile:

.. code-block:: yaml

   # profile.yml
   services:
     ariel_sync:
       template: osprey.ariel_sync

The build copies the bundled compose template into the project and adds
``ariel_sync`` to ``deployed_services``. ``osprey up`` then starts a container
that runs ``osprey ariel sync --watch``. That command syncs once on startup ---
schema migration, an incremental ingest, and an enhancement pass over entries
that are still missing derived fields --- and then enters the polling loop. Every later poll runs the same enhancement pass after it stores
what it fetched, so a larger backlog is worked down over successive polls. A
failure in that pass is logged and does not count toward
``max_consecutive_failures``.

The deployed loop does not wait for a prior ingest. It overrides
``require_initial_ingest``, so a container started against an empty database
does the full first ingest itself.

The container runs the project image that ``osprey up`` builds, the same image
the web terminals run by default, and the container runtime restarts it unless
you stop it. That is what keeps the mirror moving between deployments instead of
freezing at the last ``osprey up``. When the failure cap is reached the process
exits with a non-zero status, and the restart policy starts it again. A
repeatedly failing source therefore shows up as a restarting container, not as a
stopped one. The two readings that tell the difference are the container's
restart count and the ``ariel_last_ingestion`` health row described below.

The service publishes no port and declares no container health check. It only
makes outbound calls, to the logbook and to the database, so a container probe
would have nothing to ask it. A container that is up but no longer ingesting is
exactly the failure such a probe would report as healthy, so staleness is
reported by ``osprey health`` instead --- see below.

**Reaching the database.** When the deployment also runs the ARIEL database, the
rendered compose file sets ``ARIEL_DATABASE_HOST`` and ``ARIEL_DATABASE_PORT`` in
the container's environment. On the default bridge network they name the database
container's network alias ``ariel-postgres`` and its container port ``5432``.
When the service is declared with ``network: host`` under its ``config:`` key,
they name ``localhost`` and the port the database publishes on the host. The two
variables are written into the compose file only, never into the project's
``.env``, so commands you run on the host are unaffected. They apply only where
the connection string is derived from the ``services.postgresql`` block;
:doc:`/reference/contracts/ariel` documents the full precedence order.

When the deployment does not run the database, neither variable is written and
the derived address stays ``localhost``. Inside a bridge-networked container
that is the container itself. Point an external store at an address the
container can resolve with ``ariel.database.uri``. Declaring the service with
``network: host`` helps only when the store listens on the deployment host
itself.

.. admonition:: An authored URI is used exactly as written
   :class: warning

   If ``ariel.database.uri`` is set, that value is used verbatim and the two
   variables above are ignored. A URI naming ``localhost`` is correct from the
   host, but inside a bridge-networked container ``localhost`` is the container
   itself, so the sync cannot reach the database. Either write a URI the
   container can resolve, or remove the key and let the address be derived.

**The mirror directory.** When ``ariel.enhancement_modules.qmd_export`` is
enabled with a ``mirror_path``, the host directory it names is bind-mounted into
the container read-write. The exporter runs inside this container, so without
that mount its files would land in the container's own writable layer and be
discarded on the next recreate. The web terminals mount the same directory, so
every process that enhances an entry writes into one mirror.

**A source nobody ingests.** If ``ariel.ingestion.source_url`` is an HTTP or
HTTPS URL and the deployment declares no ``ariel_sync`` service, ``osprey build``
prints one warning line:

::

   ⚠ ariel.ingestion.source_url is https://api.example.com/logbook, but no
   service in this deployment ingests it — add a services: entry with
   `template: osprey.ariel_sync`.

The advisory never fails the build. A ``source_url`` naming a local file is
silent, because a file needs no polling service.

**Watching for a stalled mirror.** ``osprey health`` reports an
``ariel_last_ingestion`` row. When the config carries an ``ariel.ingestion``
block, that row becomes a warning once the newest ingestion is older than
``poll_interval_seconds`` plus ``watch.max_interval_seconds`` --- two hours with
the defaults above. The warning reads ``Last ingestion is <age> old, ingestion
interval is <threshold>``. Without an ``ariel.ingestion`` block the expected
cadence is unknown, and the row stays ``ok`` as long as some ingestion has been
recorded. A store that has never been ingested warns either way.


See Also
========

:doc:`search-modes`
    How search uses the ingested and enhanced data

:doc:`/reference/contracts/ariel`
    MCP tools, the capabilities API, and the database schema

:doc:`/reference/cli`
    CLI reference for ``osprey ariel ingest``, ``osprey ariel enhance``, and other commands
