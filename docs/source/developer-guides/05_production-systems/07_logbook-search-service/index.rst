================================
Logbook Search Service (ARIEL)
================================

ARIEL (Agentic Retrieval Interface for Electronic Logbooks) provides intelligent search over facility electronic logbooks. It is built around a **modular architecture** with three main layers: a :doc:`data ingestion pipeline <data-ingestion>` that normalizes logbook entries from any facility into a common PostgreSQL schema, composable :doc:`search modules <search-modes>` such as keyword matching, semantic similarity, and RAG-powered question answering, and a :doc:`web interface <web-interface>` for interactive exploration. These components connect to the rest of Osprey through the ``logbook_search`` :doc:`capability integration <osprey-integration>`.

Every layer is designed to be **facility-agnostic and extensible**. Ingestion adapters, search modules, and enhancement stages are all registerable --- you can implement your own and plug them into the full pipeline without modifying ARIEL's source code. Out of the box, adapters are included for facilities such as ALS, JLab, and ORNL, and search strategies range from keyword lookup to a multi-step ReAct agent that chains searches autonomously.

.. figure:: /_static/resources/ariel_overview.pdf
   :alt: ARIEL Logbook Search Architecture
   :align: center
   :width: 100%

   ARIEL data flow: facility logbooks are normalized through pluggable adapters into a shared PostgreSQL database, enhanced by modular processing stages, and queried through composable search modules and pipelines.

.. dropdown:: Prerequisites
   :color: info
   :icon: checklist

   ARIEL requires a working Osprey installation. Make sure you have the following
   ready before proceeding. See the :doc:`/getting-started/installation` guide for
   detailed setup instructions.

   - **Python 3.11+** with a virtual environment
   - **Osprey installed:** ``pip install osprey-framework``
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

            osprey init my-project --template control_assistant
            cd my-project

         This generates a ready-to-use ``config.yml`` with PostgreSQL, the ARIEL
         web interface, and all search modules enabled --- skip to Step 2.

         .. dropdown:: Manual configuration (existing projects)
            :color: light
            :icon: pencil

            If you already have an Osprey project, add the following to your
            ``config.yml``:

            .. code-block:: yaml

               # -- Deployed Services (add to existing list) --
               deployed_services:
                 - jupyter
                 - postgresql          # Required for ARIEL
                 - ariel_web           # ARIEL web interface

               # -- Container Configuration --
               services:
                 postgresql:
                   database: ariel
                   username: ariel
                   password: ariel
                   port_host: 5432

                 ariel_web:
                   path: ./services/ariel-web
                   port_host: 8085

               # -- ARIEL Search Configuration --
               ariel:
                 database:
                   uri: postgresql://ariel:ariel@localhost:5432/ariel
                 search_modules:
                   keyword:
                     enabled: true
                   semantic:
                     enabled: true        # Degrades gracefully if unavailable
                     provider: ollama
                     model: nomic-embed-text
                 pipelines:
                   rag:
                     enabled: true
                     retrieval_modules: [keyword, semantic]
                   agent:
                     enabled: true
                     retrieval_modules: [keyword, semantic]
                 enhancement_modules:
                   text_embedding:
                     enabled: true        # Degrades gracefully if unavailable
                     provider: ollama
                     models:
                       - name: nomic-embed-text
                         dimension: 768
                 reasoning:
                   provider: cborg
                   model_id: anthropic/claude-haiku

      .. tab-item:: 2. Deploy

         Two commands bring everything up. The first run pulls container images,
         so it may take a few minutes depending on your internet connection and
         how many services you have configured.

         Generate Docker Compose files from your ``config.yml``, pull the
         container images (PostgreSQL, ARIEL web UI, and any other deployed
         services), and start them in the background with networking and volume
         mounts:

         .. code-block:: bash

            osprey deploy up

         Once the containers are running, connect to PostgreSQL, run database
         migrations (creates tables, indexes, and the ``pgvector`` extension if
         available), then ingest the demo logbook data and generate embeddings:

         .. code-block:: bash

            osprey ariel quickstart

      .. tab-item:: 3. Search

         Three ways to query the logbook. The web interface is the easiest way to
         explore during development; the CLI and agent are better for scripting and
         interactive sessions.

         .. tab-set::

            .. tab-item:: Web Interface
               :selected:

               Open the ARIEL web UI, which provides a direct interface to the
               database and the full search service. Already running from Step 2.

               .. code-block:: text

                  Open http://localhost:8085 in your browser

            .. tab-item:: CLI Search

               Query the logbook service directly from the command line.

               .. code-block:: bash

                  osprey ariel search "What happened with the RF cavity?"

            .. tab-item:: Osprey Chat

               Ask the Osprey agent. The ``logbook_search`` capability connects
               the main agent with the ARIEL search service, so it can combine
               logbook results with other context.

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

      Keyword, semantic, RAG pipeline, and agent execution strategies.

   .. grid-item-card:: Osprey Integration
      :link: osprey-integration
      :link-type: doc
      :class-header: bg-primary text-white
      :shadow: md

      Capability, context flow, prompt builder, and error classification.

   .. grid-item-card:: Web Interface
      :link: web-interface
      :link-type: doc
      :class-header: bg-secondary text-white
      :shadow: md

      FastAPI app, frontend architecture, capabilities API, and REST endpoints.


.. toctree::
   :maxdepth: 2
   :hidden:

   data-ingestion
   search-modes
   osprey-integration
   web-interface


See Also
========

:doc:`/developer-guides/03_core-framework-systems/07_built-in-capabilities`
    Built-in capabilities reference including ``logbook_search``

:doc:`/developer-guides/03_core-framework-systems/04_prompt-customization`
    Logbook search prompt builder for facility-specific customization

:doc:`/developer-guides/02_quick-start-patterns/00_cli-reference`
    CLI reference for all ``osprey ariel`` commands
