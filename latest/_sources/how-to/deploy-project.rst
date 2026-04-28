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

.. note::

   The ``execution`` section in ``config.yml`` is optional. If omitted, the system
   defaults to local Python execution. You may see a warning about this on every
   command — it is safe to ignore.

CLI Commands
============

.. code-block:: bash

   osprey deploy up                  # Start (foreground)
   osprey deploy up --detached       # Start (background)
   osprey deploy down                # Stop
   osprey deploy restart             # Restart
   osprey deploy status              # Show status table
   osprey deploy build               # Prepare compose files without starting
   osprey deploy clean               # Remove containers and volumes
   osprey deploy rebuild             # Clean + rebuild from scratch

   # Flags
   osprey deploy up --dev            # Use local osprey source
   osprey deploy up --expose         # Bind to 0.0.0.0 (use with caution)
   osprey deploy up --config alt.yml # Custom config file
   osprey deploy up --project /path  # Specify project directory

   # Environment variable alternative to --project
   export OSPREY_PROJECT=~/my-project
   osprey deploy up

.. tip::

   The project directory is resolved in this order: ``--project`` flag >
   ``OSPREY_PROJECT`` environment variable > current working directory.

.. tip::

   There is no ``osprey deploy logs`` subcommand. Access container logs directly
   with ``docker logs <container-name>`` or ``podman logs <container-name>``.

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

Each service needs a ``docker-compose.yml.j2`` template in its service directory.
In addition, a **root-level** ``services/docker-compose.yml.j2`` is required to
define the shared network (``osprey-network``). Without it, ``deploy build`` and
``deploy up`` will fail.

.. code-block:: text

   services/
   ├── docker-compose.yml.j2          # Required: shared network definition
   ├── jupyter/
   │   └── docker-compose.yml.j2      # Per-service template
   └── postgresql/
       └── docker-compose.yml.j2      # Per-service template

Per-service templates have access to the full configuration:

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
``{{system.<key>}}``, ``{{project_root}}``, ``{{deployment.<key>}}``
(e.g., ``deployment.bind_address``), ``{{osprey_labels.<key>}}``
(``project_name``, ``project_root``, ``deployed_at``).

Network Binding and Security
============================

.. versionchanged:: 0.10.7
   Services now bind to ``127.0.0.1`` by default (previously ``0.0.0.0``).

Use ``--expose`` only when you have authentication and firewalling in place.
You can also set ``deployment.bind_address`` in ``config.yml``.

Container networking uses service names as hostnames (e.g.,
``http://pipelines:9099``). For host access from containers, use
``host.docker.internal`` (Docker) or ``host.containers.internal`` (Podman).

Environment Variables (``.env``)
=================================

The deploy system loads a ``.env`` file from the project root using python-dotenv
and passes it to Docker Compose. This file typically contains API keys, timezone
settings, and proxy configuration.

To set up:

.. code-block:: bash

   cp .env.example .env
   # Edit .env with your actual values

If no ``.env`` file is found, services start with default/empty environment
variables and a warning is logged.

Development Mode
================

The ``--dev`` flag deploys with your locally installed Osprey source instead of
the PyPI version:

.. code-block:: bash

   osprey deploy up --dev

The system builds a wheel package from your local Osprey source and copies it into
each service's build directory. It also sets ``DEV_MODE=true`` in the container
environment. If the local source cannot be found (e.g., Osprey was installed from
PyPI rather than in editable/development mode), containers fall back to the PyPI
version.

Ensure Osprey is importable in your environment before using ``--dev`` (e.g., via
``uv sync`` or an editable install). The ``--dev`` flag also requires the Python
``build`` package to create the wheel:

.. code-block:: bash

   uv pip install build   # or: pip install build

Troubleshooting
===============

**Services fail to start:** Check logs (``podman logs <name>``), verify
``config.yml`` syntax, ensure ``.env`` variables are set, confirm service paths
contain ``docker-compose.yml.j2``.

**Port conflicts:** ``lsof -i :<port>`` to find the culprit; update port mappings.

**Template errors:** Verify Jinja2 syntax (``{{var}}`` not ``{var}``); inspect
rendered files in ``build/``.

**Dev mode issues:** Confirm the Osprey wheel (``.whl``) exists in the service
build directory; check ``DEV_MODE`` env var inside the container.

.. seealso::

   :doc:`../cli-reference/index`
       CLI command reference
