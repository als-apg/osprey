====================
Container Deployment
====================

How to deploy and manage containerized services using the ``osprey deploy`` CLI.

.. dropdown:: What You'll Learn
   :color: primary
   :icon: book

   - Using ``osprey deploy`` for service deployment and orchestration
   - Configuring services in ``config.yml``
   - Managing Jinja2 Docker Compose templates
   - Development vs production deployment patterns

   **Prerequisites:** Docker/Podman basics, familiarity with project configuration.

   **Time:** 30--45 minutes

Overview
========

The deployment system handles service discovery, Docker Compose template rendering,
and container orchestration using Docker or Podman.

- **Runtime Flexibility**: Automatic Docker/Podman detection
- **Template Rendering**: Jinja2 processing with full configuration context
- **Build Management**: Automated build directories with source and config copying
- **Localhost by Default**: Services bind to ``127.0.0.1`` unless ``--expose`` is used

Service Configuration
=====================

Services are defined in the ``services:`` section of ``config.yml``:

.. code-block:: yaml

   services:
     jupyter:
       path: ./services/jupyter
       containers:
         read:
           name: jupyter-read
           hostname: jupyter-read
           port_host: 8088
           port_container: 8088
           execution_modes: ["read_only"]
         write:
           name: jupyter-write
           hostname: jupyter-write
           port_host: 8089
           port_container: 8088
           execution_modes: ["write_access"]
       copy_src: true
       render_kernel_templates: true

     open_webui:
       path: ./services/open-webui
       hostname: localhost
       port_host: 8080
       port_container: 8080

     pipelines:
       path: ./services/pipelines
       port_host: 9099
       port_container: 9099
       copy_src: true

   deployed_services:
     - jupyter
     - open_webui
     - pipelines

**Key options:** ``path`` (service directory), ``port_host``/``port_container``,
``copy_src`` (copy ``src/`` into build), ``additional_dirs``, ``containers``
(multi-container services like Jupyter read/write).

CLI Commands
============

.. code-block:: bash

   osprey deploy up                  # Start (foreground)
   osprey deploy up --detached       # Start (background)
   osprey deploy down                # Stop
   osprey deploy restart             # Restart
   osprey deploy status              # Show status table
   osprey deploy clean               # Remove containers and volumes
   osprey deploy rebuild             # Clean + rebuild from scratch

   # Flags
   osprey deploy up --dev            # Use local osprey source
   osprey deploy up --expose         # Bind to 0.0.0.0 (use with caution)
   osprey deploy up --config alt.yml # Custom config file

Deployment Workflow
===================

When ``osprey deploy up`` runs:

1. Load and merge configuration
2. Discover services from ``deployed_services``
3. Create clean build directories
4. Render Jinja2 templates with configuration context
5. Copy service files, source code, additional directories
6. Generate flattened ``config.yml`` per service
7. Run Docker/Podman Compose

Docker Compose Templates
========================

Templates live at ``{service_path}/docker-compose.yml.j2`` and have access to the
full configuration:

.. code-block:: yaml

   # services/jupyter/docker-compose.yml.j2
   services:
     jupyter-read:
       container_name: {{services.jupyter.containers.read.name}}
       ports:
         - "{{services.jupyter.containers.read.port_host}}:{{services.jupyter.containers.read.port_container}}"
       environment:
         - TZ={{system.timezone}}
       networks:
         - osprey-network

Access patterns: ``{{services.<name>.<key>}}``, ``{{file_paths.<key>}}``,
``{{system.<key>}}``, ``{{project_root}}``.

Network Binding and Security
============================

.. versionchanged:: 0.10.7
   Services now bind to ``127.0.0.1`` by default (previously ``0.0.0.0``).

Use ``--expose`` only when you have authentication and firewalling in place.
You can also set ``deployment.bind_address`` in ``config.yml``.

Container networking uses service names as hostnames (e.g.,
``http://pipelines:9099``). For host access from containers, use
``host.docker.internal`` (Docker) or ``host.containers.internal`` (Podman).

Development Mode
================

The ``--dev`` flag deploys with your locally installed Osprey source instead of
the PyPI version:

.. code-block:: bash

   osprey deploy up --dev

The system copies the framework source to ``build/services/<svc>/osprey_override/``
and sets ``DEV_MODE=true`` in the container environment. If the local source cannot
be found, containers fall back to PyPI.

.. admonition:: PLACEHOLDER: INSTALL COMMAND
   :class: warning

   **Old content (line 886):** "Verify osprey is installed in your active virtual environment"
   **New equivalent:** Needs human judgment
   **Why this is fuzzy:** The old text assumed ``pip install osprey-framework``; the rename map says use ``uv sync``, but dev-mode deploy copies source rather than installing via uv.
   **Action needed:** Confirm whether ``uv sync`` is relevant in a dev-mode deployment context, or if this tip should simply say "Ensure osprey is importable in your environment."

Troubleshooting
===============

**Services fail to start:** Check logs (``podman logs <name>``), verify
``config.yml`` syntax, ensure ``.env`` variables are set, confirm service paths
contain ``docker-compose.yml.j2``.

**Port conflicts:** ``lsof -i :<port>`` to find the culprit; update port mappings.

**Template errors:** Verify Jinja2 syntax (``{{var}}`` not ``{var}``); inspect
rendered files in ``build/``.

**Dev mode issues:** Confirm ``osprey_override/`` exists in build directory;
check ``DEV_MODE`` env var inside container.

.. seealso::

   :doc:`../cli-reference/index`
       CLI command reference

