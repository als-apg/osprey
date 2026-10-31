================================
Logbook Search Service (ARIEL)
================================

ARIEL (Agentic Retrieval Interface for Electronic Logbooks) provides intelligent
search over facility electronic logbooks. It is built around a **modular
architecture** with three main layers: a :doc:`data ingestion pipeline
<data-ingestion>` that normalizes logbook entries from any facility into a common
PostgreSQL schema, composable :doc:`search modules <search-modes>` such as
keyword and semantic similarity matching (plus a raw read-only ``sql_query``
MCP tool for structural queries the agent can run directly against the
database), and a :doc:`web interface
<web-interface>` for interactive exploration. These components connect to the
rest of Osprey through the :doc:`integration layer </reference/contracts/ariel>`, where
the agent invokes ARIEL's MCP tools as part of broader workflows.

Every layer is designed to be **facility-agnostic and extensible**. Ingestion
adapters, search modules, and enhancement stages are all registerable --- you can
implement your own and plug them into the full pipeline without modifying ARIEL's
source code. Out of the box, adapters are included for the reference logbook
formats listed on :doc:`data-ingestion`, and search strategies range from fast
keyword lookup to embedding-based semantic similarity, with multi-step
reasoning over results delegated to the OSPREY agent layer.

.. figure:: /_static/resources/ariel_overview.svg
   :alt: ARIEL Logbook Search Architecture
   :align: center
   :width: 100%

   ARIEL data flow: facility logbooks are normalized through pluggable adapters
   into a shared PostgreSQL database, enhanced by modular processing stages, and
   queried through composable search modules.

.. dropdown:: Prerequisites
   :color: info
   :icon: checklist

   ARIEL requires a working Osprey installation. Make sure you have the following
   ready before proceeding.

   - **Python 3.11+** with a virtual environment
   - **Osprey installed:** ``uv sync``
   - **Container runtime:** `Docker Desktop 4.0+ <https://docs.docker.com/get-docker/>`_ or `Podman 4.0+ <https://podman.io/getting-started/installation>`_ (for PostgreSQL)
   - **LLM API access:** An API key for your configured provider (e.g., ``ANTHROPIC_API_KEY``)
   - **(Recommended) Ollama** --- for local text embeddings powering semantic search:

     .. tab-set::

        .. tab-item:: macOS

           .. code-block:: bash

              brew install ollama
              ollama serve &            # Start Ollama in the background
              ollama pull nomic-embed-text

        .. tab-item:: Linux

           .. code-block:: bash

              curl -fsSL https://ollama.com/install.sh | sh
              ollama serve &            # Start Ollama in the background
              ollama pull nomic-embed-text

     Ollama is optional. Without it, semantic queries return a diagnostic
     and keyword search keeps answering. Semantic search also needs its
     embedding table, which ``osprey ariel quickstart`` creates (it needs the
     pgvector extension, which the deployed PostgreSQL image includes). A
     semantic search configured with no table is refused, and the refusal
     names this command.

   - **(Optional) Picture captions** --- a local vision model, also served by
     Ollama:

     .. code-block:: bash

        ollama pull qwen3-vl:4b

   - **(Optional) Picture search** --- a site-run ``llama-server`` with a
     multimodal embedding model; see :doc:`picture-search`.

     Both are optional. Like semantic search without Ollama, ARIEL degrades
     gracefully: without the caption model or without ``llama-server`` the
     module is skipped, ``osprey ariel status`` says why, and search keeps
     working on text.

.. dropdown:: Quick Start

   .. tab-set::

      .. tab-item:: 1. Configure

         The easiest way to get started is to create a new project from the
         ``control-assistant`` template, which includes ARIEL pre-configured
         and ships the logbook app on a shared card:

         .. code-block:: bash

            osprey init my-project --preset control-assistant
            cd my-project

         Run ``osprey build`` in it and you get a ready-to-use ``config.yml``
         with PostgreSQL, the ARIEL web interface, and all search modules
         enabled --- skip to Step 2. The repository's ``profile.yml`` is what
         the build reads, and its ``.env`` is where your provider keys belong.

      .. tab-item:: 2. Deploy

         Two commands bring everything up. The first run pulls container images,
         so it may take a few minutes depending on your internet connection.

         Generate Docker Compose files from your ``config.yml``, pull the
         container images, and start them in the background:

         .. code-block:: bash

            osprey up -d

         ``osprey up`` creates the schema and seeds the demo logbook. Then
         generate its embeddings (and picture captions, where a vision model
         answers):

         .. code-block:: bash

            osprey ariel quickstart

      .. tab-item:: 3. Search

         Three ways to query the logbook:

         .. tab-set::

            .. tab-item:: Web Interface
               :selected:

               Start the ARIEL web UI in a separate terminal, then open it in
               your browser.

               .. code-block:: bash

                  osprey ariel web

               Then open ``http://localhost:10300`` in your browser.

            .. tab-item:: CLI Search

               Query the logbook service directly from the command line.

               .. code-block:: bash

                  osprey ariel search "What happened with the RF cavity?"

            .. tab-item:: Agent Chat

               Ask the OSPREY agent. The logbook-search MCP tools connect the
               agent to the ARIEL search service, so it can combine logbook
               results with other context.

               .. code-block:: bash

                  osprey chat
                  >>> What does the logbook say about the last RF cavity trip?

Learn More
==========

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Data Ingestion
      :link: data-ingestion
      :link-type: doc
      :class-header: bg-success text-white
      :shadow: md

      Facility adapters, enhancement pipeline, and database schema.

   .. grid-item-card:: Search Modes
      :link: search-modes
      :link-type: doc
      :class-header: bg-info text-white
      :shadow: md

      Keyword and semantic search modules and how to add your own.

   .. grid-item-card:: Osprey Integration
      :link: /reference/contracts/ariel
      :link-type: doc
      :class-header: bg-primary text-white
      :shadow: md

      MCP tools, service factory, and search result structure.

   .. grid-item-card:: Web Interface
      :link: web-interface
      :link-type: doc
      :class-header: bg-secondary text-white
      :shadow: md

      FastAPI app, frontend architecture, capabilities API, and REST endpoints.

   .. grid-item-card:: The Search Sidecar (``qmd``)
      :link: search-sidecar
      :link-type: doc
      :class-header: bg-dark text-white
      :shadow: md

      The container behind the ``hybrid`` search mode: configuration, corpus
      mounts, disk footprint, and where it listens.

   .. grid-item-card:: Picture Captions and Picture Search
      :link: picture-search
      :link-type: doc
      :class-header: bg-info text-white
      :shadow: md

      Captions and picture search over attachments: the site's llama-server,
      measured values, and upgrade notes.



CLI Commands
============

All ARIEL functionality is available through the ``osprey ariel`` command group:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Command
     - Description
   * - ``quickstart``
     - Quick setup: migrate, load the demo logbook, and enable search
   * - ``status``
     - Show ARIEL service status
   * - ``search``
     - Search the logbook (``--mode keyword|semantic``)
   * - ``ingest``
     - Ingest logbook entries from a source file or URL
   * - ``migrate``
     - Run ARIEL database migrations
   * - ``sync``
     - Sync database: migrate, incremental ingest, and enhance
   * - ``watch``
     - Watch a source for new logbook entries
   * - ``enhance``
     - Run enhancement modules on entries
   * - ``models``
     - List embedding models and their tables
   * - ``reembed``
     - Re-embed entries with a new or existing model
   * - ``web``
     - Launch the ARIEL web interface
   * - ``purge``
     - Purge all ARIEL data from the database


.. toctree::
   :maxdepth: 2
   :hidden:

   data-ingestion
   search-modes
   web-interface
   search-sidecar
   picture-search
