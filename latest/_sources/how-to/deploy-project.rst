====================
Container Deployment
====================

How to run an Osprey project's containerized services with ``osprey deploy``.

.. dropdown:: What You'll Learn
   :color: primary
   :icon: book

   - What ``osprey deploy`` does and when you need it
   - Configuring services in ``config.yml`` (minimal example)
   - Authoring ``docker-compose.yml.j2`` templates
   - Network binding, ``.env`` loading, and the ``--dev`` workflow

   **Prerequisites:** Docker or Podman installed locally.

.. tip::

   This page is the operator/service-author reference for ``osprey deploy``.
   For the end-to-end build → ship workflow (CI/CD, release operations), use
   the ``osprey-build-deploy`` skill that the build interview installs into
   your profile repo. For the full ``services:`` schema as authored inside a
   build profile, see :ref:`profile-services`.

Overview
========

``osprey deploy`` renders each service's Jinja2 Docker Compose template,
copies source and configuration into a per-service build directory, and
hands the result to Docker or Podman Compose. A new project built from the
``control-assistant`` preset deploys exactly one service out of the box
(``postgresql``); the ``hello-world`` preset deploys none. You only need
this page when you add or customize a containerized service.

Service Configuration
=====================

Services are declared under ``services:`` in ``config.yml`` and selected for
deployment via ``deployed_services:``. A minimal example (this is what the
``control-assistant`` preset ships with):

.. code-block:: yaml

   services:
     postgresql:
       path: ./services/postgresql
       database_name: ariel
       username: ariel
       password: ariel
       port_host: 5432

   deployed_services:
     - postgresql

Each service entry must point ``path:`` at a directory containing a
``docker-compose.yml.j2`` template. Everything else under the service key is
project-specific configuration exposed to the template as
``{{services.<name>.<key>}}``. For the canonical schema (including
``copy_src``, ``additional_dirs``, ``render_kernel_templates``, and
``containers`` for multi-container services), see :ref:`profile-services`.

Service lookup namespaces
-------------------------

``find_service_config`` resolves a name from ``deployed_services`` in three
places, in order: ``osprey.<name>``, ``applications.<app>.<name>``, and
top-level ``services.<name>``. The flat form shown above is the legacy
pattern; the namespaced forms are preferred for build profiles that ship
multiple applications.

CLI Commands
============

.. code-block:: bash

   osprey deploy up [-d|--detached]   # Start services
   osprey deploy down                 # Stop services
   osprey deploy status               # Show status table
   osprey deploy build                # Render compose files without starting

Full command and flag reference: :doc:`../cli-reference/index`. Note there
is no ``osprey deploy logs`` subcommand — use ``docker logs <name>`` or
``podman logs <name>`` directly.

The project directory is resolved as: ``--project`` flag, then
``OSPREY_PROJECT`` environment variable, then current working directory.

Container Runtime Selection
===========================

The runtime is auto-detected: if Docker's daemon is reachable it is
preferred, otherwise Podman is used. Force a specific runtime with the
``CONTAINER_RUNTIME`` environment variable or by setting
``container_runtime: docker|podman|auto`` at the root of ``config.yml``.

Deployment Workflow
===================

When ``osprey deploy up`` runs:

1. Resolve the project directory and load ``config.yml`` via ``ConfigBuilder``.
2. Set ``deployment.bind_address`` (``127.0.0.1`` by default, ``0.0.0.0`` with ``--expose``).
3. Render the root ``services/docker-compose.yml.j2`` (shared ``osprey-network``).
4. For each entry in ``deployed_services``: clean and create the build dir, render the service compose template, copy service files.
5. If ``copy_src: true``, copy ``src/`` into the build as ``repo_src/``, plus ``requirements.txt`` and ``pyproject.toml`` (renamed ``pyproject_user.toml``).
6. With ``--dev``, build a wheel from the local Osprey checkout and drop it into the build dir.
7. Copy any ``additional_dirs`` into the build.
8. Auto-create ``_agent_data/`` subdirectories declared under ``file_paths``.
9. Write a flattened ``config.yml`` per service. ``${VAR}`` placeholders are preserved (secrets stay out of the rendered output and are resolved at container start).
10. Shell out to ``docker compose`` / ``podman compose``.

Docker Compose Templates
========================

Each service needs a ``docker-compose.yml.j2`` template in its service
directory. In addition, a **root-level** ``services/docker-compose.yml.j2``
is required to define the shared network (``osprey-network``). Without it,
``deploy build`` and ``deploy up`` will fail.

.. code-block:: text

   services/
   ├── docker-compose.yml.j2          # Required: shared network definition
   └── postgresql/
       └── docker-compose.yml.j2      # Per-service template

Per-service templates have access to the full configuration plus a few
engine-injected values:

.. code-block:: yaml

   # services/postgresql/docker-compose.yml.j2
   services:
     postgresql:
       container_name: {{services.postgresql.container_name | default('osprey-postgres')}}
       labels:
         osprey.project.name: "{{osprey_labels.project_name}}"
         osprey.project.root: "{{osprey_labels.project_root}}"
         osprey.deployed.at: "{{osprey_labels.deployed_at}}"
       ports:
         - "{{deployment.bind_address}}:{{services.postgresql.port_host}}:5432"
       environment:
         TZ: {{system.timezone}}
       networks:
         - osprey-network

Common access patterns: ``{{services.<name>.<key>}}``,
``{{file_paths.<key>}}``, ``{{system.<key>}}``, ``{{project_root}}``,
``{{deployment.bind_address}}``, and ``{{osprey_labels.project_name}}`` /
``project_root`` / ``deployed_at`` (injected by the deploy engine).

Network Binding and Security
============================

Services bind to ``127.0.0.1`` by default. Use ``--expose`` only when you
have authentication and firewalling in place — ``--expose`` overrides any
``deployment.bind_address`` you set in ``config.yml``.

Container networking uses service names as hostnames (e.g.,
``postgresql:5432``). For host access from inside containers, use
``host.docker.internal`` (Docker) or ``host.containers.internal`` (Podman).

Environment Variables (``.env``)
=================================

The deploy system passes a ``.env`` file from the project root to Docker /
Podman Compose via ``--env-file``. Variables defined there are available to
Compose substitution and to running containers.

.. code-block:: bash

   cp .env.example .env
   # Edit .env with your actual values

If no ``.env`` file is found, services start with default/empty environment
variables and a warning is logged.

Development Mode
================

The ``--dev`` flag deploys with your locally installed Osprey source
instead of the PyPI version:

.. code-block:: bash

   osprey deploy up --dev

The system builds a wheel from your local Osprey source and copies it into
each service's build directory, then sets ``DEV_MODE=true`` in the
container environment. If the local source cannot be found (e.g., Osprey
was installed from PyPI rather than editable mode), containers fall back to
the PyPI version.

``--dev`` requires the Python ``build`` package:

.. code-block:: bash

   uv pip install build   # or: pip install build

Troubleshooting
===============

**Services fail to start:** Check logs (``docker logs <name>`` or
``podman logs <name>``), verify ``config.yml`` syntax, ensure ``.env``
variables are set, confirm service paths contain ``docker-compose.yml.j2``.

**Port conflicts:** ``lsof -i :<port>`` to find the culprit; update
``port_host``.

**Template errors:** Verify Jinja2 syntax (``{{var}}`` not ``{var}``);
inspect rendered files under ``build/services/<name>/``.

**Daemon not running:** Both Docker and Podman print platform-specific
hints; on macOS, start Docker Desktop or run ``podman machine start``.

**``--dev`` issues:** Confirm the Osprey wheel (``.whl``) exists in the
service build directory; check ``DEV_MODE`` env var inside the container.

.. seealso::

   :doc:`../cli-reference/index`
       Full ``osprey deploy`` command and flag reference.

   :ref:`profile-services`
       Authoritative ``services:`` schema for build profiles.
