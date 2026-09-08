=====================
Standalone Deployment
=====================

The ``ariel-standalone`` preset packages ARIEL together with a **sandboxed
Osprey agent** scoped to ARIEL's MCP surface. There is no
control-system runtime, no shell, no filesystem writes, no Python executor.
Skills and plugins extend what the agent can *reason about* over your
logbook corpus; they cannot extend what it can *touch* on your host.

Use this preset when a facility wants ARIEL for logbook research but does
not run an OSPREY control-assistant — for example, when hardware control
is handled by a different system, or when a team wants to evaluate ARIEL
on their corpus before committing to the full deployment.

.. seealso::

   Already running the ``control-assistant`` multi-user stack? The same
   research terminal ships there as its ``ariel`` persona — a card on the
   landing page beside the operator logins, sharing that deployment's
   PostgreSQL and logbook. This page is the route when ARIEL should stand
   alone; :ref:`how-to-multi-user` covers the persona.

.. image:: /_static/resources/ariel_standalone.png
   :alt: ARIEL-standalone bundle: PostgreSQL+pgvector, the ARIEL service, and a sandboxed Osprey agent reaching ARIEL only through the MCP boundary
   :align: center
   :width: 100%

.. dropdown:: Prerequisites
   :color: info
   :icon: checklist

   The standalone preset requires a working Osprey installation. Make sure
   you have the following ready before proceeding.

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

         Create a new project from the ``ariel-standalone`` preset. This
         skeleton ships with Postgres+pgvector services, the ARIEL config,
         and the sandboxed Osprey agent persona — no channel finder, no
         archiver, no Python executor:

         .. code-block:: bash

            osprey init my-logbook --preset ariel-standalone
            cd my-logbook
            osprey build

         The generated ``config.yml`` enables keyword and semantic search
         out of the box. Higher-level reasoning over results — multi-step
         retrieval, answer synthesis, custom prompting — runs in the
         sandboxed agent on top of these search modes (see
         :doc:`search-modes`).

      .. tab-item:: 2. Deploy

         Two commands bring everything up. The first run pulls container
         images, so it may take a few minutes depending on your internet
         connection.

         Generate Docker Compose files from your ``config.yml``, pull the
         PostgreSQL+pgvector container image, and start it in the
         background:

         .. code-block:: bash

            osprey up -d

         Once the container is running, run database migrations and
         ingest the bundled demo logbook with embeddings:

         .. code-block:: bash

            osprey ariel quickstart

      .. tab-item:: 3. Search

         Three ways to query the logbook:

         .. tab-set::

            .. tab-item:: Web Terminal
               :selected:

               Launch the web terminal. It hosts the ARIEL search panel
               *and* an Osprey agent chat surface scoped to the ARIEL MCP
               tools and the ``logbook-deep-research`` skill.

               .. code-block:: bash

                  osprey web

            .. tab-item:: CLI Search

               Query the logbook service directly from the command line.

               .. code-block:: bash

                  osprey ariel search "What happened with the RF cavity?"

            .. tab-item:: Claude Chat

               Ask the sandboxed agent from a terminal REPL. The same
               ARIEL MCP tools and ``logbook-deep-research`` skill are
               available.

               .. code-block:: bash

                  osprey chat
                  >>> What does the logbook say about the last RF cavity trip?

Customizing for your facility
=============================

There are two customization paths, depending on how durable your changes
need to be.

Quick edits (one-off tweaks)
----------------------------

For small in-place adjustments to a project you just built, edit files
directly:

1. **Edit** ``.claude/rules/facility.md``. The default ships with a thin
   placeholder for the "Example Research Facility (ERF)" --- a facility-identity
   stub (name, type, mission) plus a pointer to the ``facility_knowledge`` tools
   (``list_concepts``/``read_concept``/``search``) for deeper content. Replace
   this with your facility's real terminology, system names, and naming
   conventions so the agent uses the right vocabulary when interpreting user
   questions.

   .. note::

      ``.claude/rules/facility.md`` is auto-registered as user-owned
      during ``osprey build``. ``osprey build`` will preserve your
      edits.

2. **Set provider credentials in the repository's ``.env``** (e.g.
   ``ANTHROPIC_API_KEY`` or ``CBORG_API_KEY``). The default provider is
   ``anthropic``. That file is the deployment's one secret store, and a build
   never rewrites it, so a value set there survives every rebuild.

3. **Replace the demo logbook seed.** The bundled
   ``data/logbook_seed/demo_logbook.json`` is 28 entries of fictional
   accelerator events. Either:

   - Replace the file with a dump from your real logbook (preserve the
     ``generic_json`` schema), or
   - Edit ``ariel.ingestion.adapter`` / ``source_url`` in ``config.yml``
     to point at your facility's logbook system (see
     :doc:`data-ingestion`).

Durable customization (a profile you own)
-----------------------------------------

For changes you want to keep across rebuilds — adding a custom skill,
overriding a rule, wiring up a real logbook — create a deployment repository
from the ``ariel-standalone`` preset:

.. code-block:: bash

   osprey init my-ariel --preset ariel-standalone
   cd my-ariel

This writes a ``my-ariel/`` git repository whose ``profile.yml`` is a standalone
manifest — the preset's full configuration written out explicitly, with no
``extends:`` back to the preset — plus the preset's ``data/`` tree and an
``.env.example`` listing every variable the agent reads. Directories for your
own artifacts are not created up front: make a ``rules/``, ``skills/`` or
``agents/`` directory when you have something to put in it, and the build
carries its contents through. Edit ``profile.yml``, the data files, and those
directories, then rebuild whenever you change something:

.. code-block:: bash

   osprey build

The render lands in the repository's ``build/`` directory, which is kept out of
git. The profile is your facility's source of truth — commit it. ``build/`` is a
regenerable artifact. See :doc:`../build-profiles` for the full schema and the
preset → profile → build model.
