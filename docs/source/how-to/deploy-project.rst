====================
Container Deployment
====================

How to run a deployment's containerized services.

.. dropdown:: What You'll Learn
   :color: primary
   :icon: book

   - What ``osprey build`` and ``osprey up`` do, and when you need them
   - Configuring services in ``config.yml`` (minimal example)
   - Authoring ``docker-compose.yml.j2`` templates
   - Network binding, ``.env`` loading, and the ``--dev`` workflow

   **Prerequisites:** Docker or Podman installed locally.

.. tip::

   This page is the operator/service-author reference for the container side of
   a deployment. For the end-to-end walkthrough — deployment repository, CI
   pipeline, stack up — follow :doc:`deploy-a-facility`; the judgment that goes
   with running it day to day lives in the ``osprey-deploy-ops`` skill. For the
   full ``services:`` schema as authored inside a build profile, see
   :ref:`profile-services`.

Overview
========

``osprey build`` renders each service's Jinja2 Docker Compose template and
copies source and configuration into a per-service build directory; ``osprey
up`` hands the result to Docker or Podman Compose. A deployment created from the
``control-assistant`` preset deploys a full stack out of the box:
``postgresql``, ``openobserve``, ``event_dispatcher`` and ``dispatch_worker``,
``bluesky`` (with its co-deployed Tiled data server), ``virtual_accelerator``,
``bluesky_panels``, and the multi-user web-terminal stack. Even the minimal
``hello-world`` preset deploys one service (``openobserve``, for telemetry).
You only need this page when you add or customize a containerized service.

Service Configuration
=====================

Services are declared under ``services:`` in ``config.yml`` and selected for
deployment via ``deployed_services:``. A minimal example (one of the services
the ``control-assistant`` preset ships with):

.. code-block:: yaml

   services:
     postgresql:
       path: ./services/postgresql
       database_name: ariel
       username: ariel
       port_host: 5432

   deployed_services:
     - postgresql

Each service entry must point ``path:`` at a directory containing a
``docker-compose.yml.j2`` template. Everything else under the service key is
project-specific configuration exposed to the template as
``{{services.<name>.<key>}}``. Beyond ``path``, a service entry may also
declare ``copy_src``, ``additional_dirs``, and ``render_kernel_templates``
(a multi-container service is expressed by the ``docker-compose.yml.j2``
template defining more than one compose service). For how facility services
are declared inside a build profile, see :ref:`profile-services`.

Service lookup namespaces
-------------------------

A name in ``deployed_services`` is looked up by its literal spelling — there
is no search order. A plain name like ``postgresql`` resolves to top-level
``services.postgresql``. A dotted name picks its namespace explicitly:
``osprey.<name>`` reads ``osprey.services.<name>``, and
``applications.<app>.<name>`` reads ``applications.<app>.services.<name>``.
The flat form shown above is the common case; the namespaced forms exist for
build profiles that ship multiple applications.

CLI Commands
============

.. code-block:: bash

   osprey build                  # Render build/, compose files included
   osprey up [-d|--detached]     # Start services, as built
   osprey up --build             # Re-render first, then start
   osprey down                   # Stop services, keeping all data
   osprey restart                # Stop then start services
   osprey status                 # Show status table
   osprey logs [SERVICE] [-f]    # Show container logs
   osprey reset                  # Wipe back to a fresh state (destructive)
   osprey users seed [USER]      # (Re)seed multi-user web-terminal workspaces
   osprey users remove USER      # Remove one user's workspace (--archive | --purge)
   osprey users prune            # Remove workspaces of users no longer on the roster
                                 #   (--archive | --purge, --dry-run)

Full command and flag reference: :doc:`../cli-reference/index`.

Every verb acts on the deployment repository enclosing the working directory —
the nearest ``profile.yml`` at or above it. ``--repo DIRECTORY`` names another
one explicitly.

Container Runtime Selection
===========================

The runtime is auto-detected: if Docker's daemon is reachable it is
preferred, otherwise Podman is used. Force a specific runtime with the
``CONTAINER_RUNTIME`` environment variable or by setting
``container_runtime: docker|podman|auto`` at the root of ``config.yml``.

Deployment Workflow
===================

``osprey build`` renders the compose files; ``osprey up`` starts what it
rendered. Steps 1–9 happen at build time, step 10 at start:

1. Resolve the deployment repository and load ``config.yml`` via ``ConfigBuilder``.
2. Apply ``deployment.bind_address`` (``127.0.0.1`` by default; change it with ``osprey set deployment.bind_address=0.0.0.0`` and rebuild).
3. Render the root ``services/docker-compose.yml.j2`` (shared ``osprey-network``).
4. For each entry in ``deployed_services``: clean and create the build dir, render the service compose template, copy service files.
5. If ``copy_src: true``, copy ``src/`` into the build as ``repo_src/``, plus ``requirements.txt`` and ``pyproject.toml`` (renamed ``pyproject_user.toml``).
6. With ``--dev``, build a wheel from the local Osprey checkout and drop it into the build dir.
7. Copy any ``additional_dirs`` into the build.
8. Auto-create the ``_agent_data/`` subdirectories the deploy step sweeps (currently ``registry_exports_dir``). Others declared under ``file_paths`` — ``api_calls_dir`` — are created on demand by the code that writes to them.
9. Write a flattened ``config.yml`` per service. ``${VAR}`` placeholders are preserved (secrets stay out of the rendered output and are resolved at container start).
10. Shell out to ``docker compose`` / ``podman compose``.

Keeping a Rendered Deployment Up to Date
========================================

``build/`` is a *rendered artifact*: ``osprey build`` writes its ``config.yml``
and service scaffolding from ``profile.yml``, and ``osprey up`` starts exactly
what that render describes. Nothing is re-derived at start time, so a change to
the profile only reaches the containers once you rebuild.

To pick up a profile edit::

   osprey build
   osprey up -d

or in one step::

   osprey up --build -d

Every build wipes and re-renders ``build/`` and preserves what you own: ``.env``
(your provider keys, plus the service tokens and passwords your existing
container volumes were initialized with), the agent's memory under ``var/``, and
the repository's ``.git`` history. ``data/`` in the build zone is
re-materialized from the profile; the source zone is never touched by a build.

Two guards make render drift visible:

* **Drift refusal** — ``osprey up`` recomputes a fingerprint over the resolved
  profile (stamped into ``.osprey-manifest.json`` at build time) and compares it
  with the stamp. If the profile has moved on, ``up`` refuses and says what
  changed, because starting would deploy something other than what the profile
  now describes. Rebuild with ``--build``, or start the old render knowingly
  with ``--as-built``. The fingerprint covers the profile's data tree and
  convention directories as well as ``profile.yml``, so regenerating a channel
  database or adding a rule trips it too. ``osprey status`` reports the same
  comparison without acting on it.
* **Endpoint summary** — every ``osprey up`` ends with a summary of
  the published service endpoints, including an explicit ``web terminal
  (not configured in this project)`` line when the config declares no web
  tier, so a missing service is a stated fact rather than a silent absence.

Docker Compose Templates
========================

Each service needs a ``docker-compose.yml.j2`` template in its service
directory. In addition, a **root-level** ``services/docker-compose.yml.j2``
is required to define the shared network (``osprey-network``). Without it,
``osprey build`` and ``osprey up`` will fail.

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

Service Template Ownership
==========================

The service templates under ``<project>/services/`` are framework-managed:
every ``osprey build`` refreshes them from the installed OSPREY version, so
compose fixes reach your project automatically. Do not edit them in place —
your changes would be overwritten on the next build.

To customize a service template, claim it — which **moves** it into the build
profile the project was built from, where edits survive:

.. code-block:: bash

   osprey scaffold claim services/postgresql   # move it into the profile
   osprey scaffold diff services/postgresql    # compare yours against the framework
   osprey scaffold unclaim services/postgresql # restore framework management

Edit the moved copy under ``<profile>/services/postgresql/``, then rebuild with
``--force`` to deploy it. Every build copies it back and marks it yours, so
later re-renders leave it alone. ``osprey scaffold list`` shows what is
framework-managed and what is yours; the same mechanism covers the agent
artifacts (rules, agents, skills, hooks). See :ref:`profile-claim` for the full
workflow and the artifacts a claim refuses.

Before reaching for a claim, check whether a config key or environment
variable already covers your need — most service knobs (ports, images,
credentials, retention) are configurable without forking the template.

Overriding Service Images
=========================

Every service image resolves through the same three-layer chain — an
environment variable wins, then a ``config.yml`` key, then the packaged
default:

.. list-table::
   :header-rows: 1

   * - Service
     - Environment variable
     - Config key
   * - postgresql
     - ``OSPREY_POSTGRES_IMAGE``
     - ``services.postgresql.image``
   * - openobserve
     - ``OSPREY_OPENOBSERVE_IMAGE``
     - ``services.openobserve.image``
   * - event_dispatcher
     - ``OSPREY_DISPATCH_IMAGE``
     - ``services.event_dispatcher.image``
   * - dispatch_worker
     - ``OSPREY_WORKER_IMAGE``
     - ``services.dispatch_worker.image``
   * - nextcloud_bridge
     - ``OSPREY_NEXTCLOUD_BRIDGE_IMAGE``
     - ``services.nextcloud_bridge.image``
   * - gchat_bridge
     - ``OSPREY_GCHAT_BRIDGE_IMAGE``
     - ``services.gchat_bridge.image``
   * - bluesky
     - ``OSPREY_BLUESKY_BRIDGE_IMAGE``
     - ``services.bluesky.image``
   * - bluesky (Tiled sidecar)
     - ``OSPREY_TILED_IMAGE``
     - ``services.bluesky.tiled_image``
   * - bluesky_panels
     - ``OSPREY_BLUESKY_PANELS_IMAGE``
     - ``services.bluesky_panels.image``
   * - virtual_accelerator
     - ``OSPREY_VA_IMAGE``
     - ``services.virtual_accelerator.image``

Point either layer at an internal registry mirror or a pinned digest when
your deployment host cannot (or should not) pull public images.

Network Binding and Security
============================

Services bind to ``127.0.0.1`` by default. Reaching them from off-host is a
property of the build, not of a start-time flag: the bind address is rendered
into every published port. Change it with ``osprey set
deployment.bind_address=0.0.0.0`` and rebuild, and only when you have
authentication and firewalling in place.

Container networking uses service names as hostnames (e.g.,
``postgresql:5432``). For host access from inside containers, use
``host.docker.internal`` (Docker) or ``host.containers.internal`` (Podman).

Environment Variables (``.env``)
=================================

The deploy system passes the repository's ``.env`` to Docker / Podman Compose
via ``--env-file``. Compose uses these values to fill in ``${VAR}`` placeholders
in the rendered compose files; a variable reaches a running container only where
a template maps it in.

That ``.env`` is the deployment's one secret store, and a build never rewrites
what is in it, so set a value there once:

.. code-block:: bash

   cp .env.example .env
   # Edit .env with your actual values

See :ref:`profile-secrets`.

``osprey up`` also *writes* to these files. On first deploy it mints any
missing service tokens and passwords (for example ``EVENT_DISPATCHER_TOKEN``,
``ZO_ROOT_USER_PASSWORD``, or ``ARIEL_DB_PASSWORD``) so services never start
with blank or publicly-known credentials, restricts the file to owner-only
permissions, and then writes those values **back into the profile's** ``.env``
under a "Minted by deploy" heading. That is what makes the stack reproducible: a
rebuild from the same profile comes up on the same credentials instead of
minting a second set the running containers do not trust.

The write-back never overwrites. A value already in the profile wins — it is
pinned by the docker volume that was initialized with it — and a deploy whose
own value disagrees says so by variable name (never by value) and keeps using
its own, leaving you to reconcile the two.

If the profile cannot be reached — it has moved or been deleted, or the project
names none — the deploy still succeeds. The secrets stay in the project
``.env``, a warning names the path that failed, and the project records that its
``.env`` is the only copy; a later ``osprey build`` repeats that warning before
it touches the directory. Back that file up.

Keep both ``.env`` files out of version control (the profile's ``.gitignore``
does this for you).

.. note::

   Postgres reads ``ARIEL_DB_PASSWORD`` (as ``POSTGRES_PASSWORD``) only when
   initializing a **fresh** data volume. A volume created before the password
   was minted keeps its original password; the ``${ARIEL_DB_PASSWORD:-ariel}``
   fallback — applied by the compose template and by the DSN the agent derives
   from ``services.postgresql`` — keeps such deployments working. To adopt
   the minted password, remove the ``ariel_postgres_data`` volume and redeploy
   (this deletes the stored logbook data — re-ingest afterwards).

If no ``.env`` file is found, services start with default/empty environment
variables and a warning is logged.

Development Mode
================

The ``--dev`` flag runs the deployment on your locally installed Osprey
source instead of the PyPI version. Dev-ness is a property of the *build*:
``osprey build --dev`` stages a wheel from your local source into each
service's build context and marks the render as a dev build. ``osprey up
--dev`` then starts that render with freshly rebuilt images:

.. code-block:: bash

   osprey build --dev
   osprey up --dev

   # or in one step
   osprey up --build --dev

Your dev source is baked into the images at build time; nothing changes
inside an already-running container. ``osprey up --dev`` on a build that was
rendered without ``--dev`` refuses rather than silently starting the
published release, and a plain ``osprey up`` of a dev build warns that the
images carry your local checkout.

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
service build directory, and that the image was rebuilt after your source
change — rerun ``osprey up --build --dev`` to re-render and rebuild it.

.. seealso::

   :doc:`../cli-reference/index`
       Full lifecycle command and flag reference.

   :ref:`profile-services`
       Authoritative ``services:`` schema for build profiles.

   :doc:`containerize-project`
       The *project image* (assistant + web terminal in one container) built
       from the generated ``Dockerfile`` — distinct from the service
       containers this page covers.
