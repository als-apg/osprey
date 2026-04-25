================================
Logbook Search Service (ARIEL)
================================

ARIEL (Agentic Retrieval Interface for Electronic Logbooks) provides intelligent
search over facility electronic logbooks. It is built around a **modular
architecture** with three main layers: a :doc:`data ingestion pipeline
<data-ingestion>` that normalizes logbook entries from any facility into a common
PostgreSQL schema, composable :doc:`search modules <search-modes>` such as keyword
matching, semantic similarity, and RAG-powered question answering, and a
:doc:`web interface <web-interface>` for interactive exploration. These components
connect to the rest of Osprey through the :doc:`integration layer
<osprey-integration>`.

Every layer is designed to be **facility-agnostic and extensible**. Ingestion
adapters, search modules, and enhancement stages are all registerable --- you can
implement your own and plug them into the full pipeline without modifying ARIEL's
source code. Out of the box, adapters are included for facilities such as ALS,
JLab, and ORNL, and search strategies range from keyword lookup to a multi-step
ReAct agent that chains searches autonomously.

.. figure:: /_static/resources/ariel_overview.svg
   :alt: ARIEL Logbook Search Architecture
   :align: center
   :width: 100%

   ARIEL data flow: facility logbooks are normalized through pluggable adapters
   into a shared PostgreSQL database, enhanced by modular processing stages, and
   queried through composable search modules and pipelines.

.. dropdown:: Prerequisites
   :color: info
   :icon: checklist

   ARIEL requires a working Osprey installation. Make sure you have the following
   ready before proceeding.

   - **Python 3.11+** with a virtual environment
   - **Osprey installed:** ``uv sync``
   - **Container runtime:** `Docker Desktop 4.0+ <https://docs.docker.com/get-docker/>`_ or `Podman 4.0+ <https://podman.io/getting-started/installation>`_ (for PostgreSQL and the web interface)
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

     Ollama is optional. ARIEL degrades gracefully to keyword-only search
     if Ollama or pgvector is unavailable. You can install them later and
     re-run ``osprey ariel quickstart`` to enable semantic search.

.. dropdown:: Quick Start

   .. tab-set::

      .. tab-item:: 1. Configure

         The easiest way to get started is to create a new project from the
         ``control_assistant`` template, which includes ARIEL pre-configured:

         .. code-block:: bash

            osprey build my-project --preset control-assistant
            cd my-project

         This generates a ready-to-use ``config.yml`` with PostgreSQL, the ARIEL
         web interface, and all search modules enabled --- skip to Step 2.

      .. tab-item:: 2. Deploy

         Two commands bring everything up. The first run pulls container images,
         so it may take a few minutes depending on your internet connection.

         Generate Docker Compose files from your ``config.yml``, pull the
         container images, and start them in the background:

         .. code-block:: bash

            osprey deploy up

         Once the containers are running, connect to PostgreSQL, run database
         migrations, then ingest the demo logbook data and generate embeddings:

         .. code-block:: bash

            osprey ariel quickstart

      .. tab-item:: 3. Search

         Three ways to query the logbook:

         .. tab-set::

            .. tab-item:: Web Interface
               :selected:

               Open the ARIEL web UI. Already running from Step 2.

               .. code-block:: text

                  Open http://localhost:8085 in your browser

            .. tab-item:: CLI Search

               Query the logbook service directly from the command line.

               .. code-block:: bash

                  osprey ariel search "What happened with the RF cavity?"

            .. tab-item:: Claude Chat

               Ask the Osprey agent. The logbook search MCP tool connects
               Claude Code with the ARIEL search service, so it can combine
               logbook results with other context.

               .. code-block:: bash

                  osprey claude chat
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

      Keyword, semantic, RAG pipeline, and agent execution strategies.

   .. grid-item-card:: Osprey Integration
      :link: osprey-integration
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


CLI Commands
============

All ARIEL functionality is available through the ``osprey ariel`` command group:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Command
     - Description
   * - ``quickstart``
     - Quick setup: migrate, ingest demo data, and enable search
   * - ``status``
     - Show ARIEL service status
   * - ``search``
     - Search the logbook (``--mode keyword|semantic|rag|auto``)
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
   osprey-integration


See Also
========

