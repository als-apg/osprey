================================
Logbook Search Service (ARIEL)
================================

ARIEL (Agentic Retrieval Interface for Electronic Logbooks) provides intelligent search over facility electronic logbooks. It combines keyword full-text search, semantic embedding similarity, RAG-powered question answering, and an agentic ReAct executor into a single service that integrates with Osprey's capability system.

ARIEL is designed to be **facility-agnostic**. Pluggable ingestion adapters normalize logbook data from any source (ALS, JLab, ORNL, or custom JSON) into a common PostgreSQL schema. At query time, Osprey's ``logbook_search`` capability forwards the user's natural language question to the ARIEL service, which routes it through the configured search strategy --- keyword lookup for specific terms, semantic similarity for conceptual queries, or a full RAG pipeline that retrieves, fuses, and generates a cited answer. An optional ReAct agent mode can chain multiple searches autonomously to answer complex questions.

.. figure:: /_static/resources/ariel_overview.pdf
   :alt: ARIEL Logbook Search Architecture
   :align: center
   :width: 100%

.. dropdown:: Quick Start

   **1. Configure** --- add the ARIEL section to your ``config.yml`` and include the ``postgresql`` and ``ariel_web`` deployed services:

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
        pipelines:
          rag:
            enabled: true
            retrieval_modules: [keyword]
        reasoning:
          provider: cborg
          model_id: anthropic/claude-haiku

   **2. Deploy and ingest** --- start services, create tables, and load logbook data:

   .. code-block:: bash

      osprey deploy up            # Start PostgreSQL + ARIEL web
      osprey ariel quickstart     # Migrate + ingest demo data

   **3. Search** --- three ways to query the logbook:

   .. code-block:: bash

      # Direct CLI search
      osprey ariel search "What happened with the RF cavity?"

   .. code-block:: bash

      # Through the Osprey agent (logbook_search capability is auto-registered)
      osprey chat
      >>> What does the logbook say about the last RF cavity trip?

   .. code-block:: text

      # Web interface (already running from step 2)
      Open http://localhost:8085 in your browser

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
