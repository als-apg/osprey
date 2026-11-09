====================
Container Deployment
====================

Use this page when you add or customize a containerized service in a
deployment. It shows how to configure services, how ``osprey build`` and
``osprey up`` turn them into running containers, and how to keep a deployment
up to date.

.. dropdown:: What You'll Learn
   :color: primary
   :icon: book

   - What ``osprey build`` and ``osprey up`` do, and when you need them
   - Configuring services in ``config.yml`` (minimal example)
   - Keeping a rendered deployment up to date, and the ``--dev`` workflow
   - Which compose providers are supported, and what changes between them
   - Overriding service images, and mirroring every image into one registry
   - Where the deeper reference lives: compose templates, networking, and the ``.env`` chain each have their own subpage

   **Prerequisites:** Docker or Podman installed locally.

.. tip::

   This page is the operator and service-author reference for the container
   side of a deployment. For related tasks, go elsewhere:

   - To follow the end-to-end walkthrough (deployment repository, CI pipeline,
     stack up), and for the commands you run day to day once the stack is up,
     see :doc:`/how-to/deploy-a-facility`.
   - To look up the full ``services:`` schema as authored inside a build
     profile, see :ref:`profile-services`.

Overview
========

``osprey build`` *renders* the deployment into ``build/``. It fills in each
service's Jinja2 Docker Compose template, and copies source and configuration
into a per-service build directory. ``osprey up`` hands the result to Docker or
Podman Compose.

A deployment created from the ``demo`` preset deploys a full stack out of the
box:

- ``postgresql``;
- ``event_dispatcher`` and ``dispatch_worker``;
- ``bluesky`` (with its co-deployed Tiled data server);
- ``virtual_accelerator``;
- ``bluesky_web``;
- the multi-user web-terminal stack.

Every deployment, the minimal ``hello-world`` one included, also runs
``agent-record``, which keeps the agent record
(:doc:`/how-to/health-and-monitoring/agent-record`).

Service Configuration
=====================

Declare services under ``services:`` in ``config.yml``, and select them for
deployment with ``deployed_services:``. Here is a minimal example, one of the
services the ``demo`` preset ships with:

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
project-specific configuration. The template sees it as
``{{services.<name>.<key>}}``.

Beyond ``path``, a service entry may also declare ``copy_src`` and
``additional_dirs``. To make a multi-container service, define more than one
compose service in its ``docker-compose.yml.j2`` template. For how facility
services are declared inside a build profile, see :ref:`profile-services`.

Service lookup namespaces
-------------------------

A name in ``deployed_services`` is looked up by its literal spelling; there is
no search order.

- A plain name like ``postgresql`` resolves to top-level
  ``services.postgresql``.
- ``osprey.<name>`` reads ``osprey.services.<name>``.
- ``applications.<app>.<name>`` reads ``applications.<app>.services.<name>``.

The flat form shown above is the common case. The dotted, namespaced forms
exist for build profiles that ship multiple applications.

The graph store (``graphdb``)
-----------------------------

Alongside ``postgresql``, the ``demo`` preset deploys a ``graphdb`` service: a
Neo4j store holding the facility's knowledge graph. Its ``services.graphdb``
block names:

- the image;
- the bolt port the seeder and the health checks dial (``port_host``);
- the HTTP port of the Neo4j Browser an operator opens (``http_port_host``);
- the corpus the store seeds (``ttl_path``);
- the JVM memory the container runs with.

``ttl_path`` resolves against the ``config.yml`` directory. When the block
does not spell one, it is filled with the graph view the build writes from the
facility file, ``data/graph/facility.ttl``.

Two more keys bound what one *query* may cost, rather than what the container
may use:

- ``query_timeout_s`` (15 seconds by default) is the transaction timeout the
  store enforces on a single query. So a runaway traversal is cancelled
  server-side, rather than left open by a client that has given up.
- ``query_max_rows`` (200) is how many rows come back before the answer is
  truncated.

The OSPREY agent's graph search reads both, and tells you when a result was cut
short. Raising them spends the agent's context window rather than the store's
memory. A few thousand rows crowd out the conversation long before they trouble
Neo4j. For what the agent does with the store once it is up, including the
query tools and the read-only posture, see
:doc:`/how-to/facility-knowledge/use-facility-graph`.

The block deliberately carries **no password**, the same convention
``postgresql`` follows. ``osprey up`` mints ``GRAPHDB_PASSWORD`` into the
project ``.env`` when it is unset, and the container reads it from there.
Nothing reads a password written into ``config.yml``.

Every deploy starts the store ahead of the rest of the stack and bootstraps it.
It then compares the store's seed marker with the digest of ``ttl_path``:

- A store that already holds that corpus is left alone.
- Any other store, an empty one included, is wiped and the corpus imported. So
  a changed facility file reaches the store on the next
  ``osprey build && osprey up``, with no seeding step.
- A store whose graph configuration is not OSPREY's is replaced the same way.

If bootstrapping or seeding fails, the deploy warns and carries on. It names
``osprey build && osprey up``, which finishes the job once the cause is fixed.

To query a graph store this deployment does *not* run, point the block at it
instead of deploying one. See
:ref:`Pointing at a store the facility runs <graph-external-store>`.

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

Full command and flag reference: :doc:`/reference/cli`.

Every verb acts on the deployment repository enclosing the working directory:
the nearest ``profile.yml`` at or above it. To name another one explicitly,
pass ``--repo DIRECTORY``.

Container Runtime Selection
===========================

``osprey build`` chooses the container runtime once, and records it in
``build/config.yml`` as ``container_runtime: docker`` or ``podman``.

- With the default ``container_runtime: auto``, the build takes Docker when its
  daemon answers, and Podman otherwise. A host where no runtime answers
  records ``auto``.
- An explicit ``docker`` or ``podman`` in the profile's ``config:`` block is
  taken as written. A tier or app file may not state one.

Every render of the build (each tier, app and image copy) records the same
answer. Every later verb reads the recorded runtime and talks only to it. That
covers ``up``, ``down``, ``restart``, ``reset`` and the port preflight. If that
runtime says it is not running, or does not answer within two minutes, the verb
refuses and says which. It never switches to the other runtime. A stack started
under Docker is invisible from Podman, so a ``down`` served there would report
it stopped while it keeps running.

To move a deployment to the other runtime, stop it where it runs, then build
again on the new runtime. ``CONTAINER_RUNTIME=docker|podman`` in the
environment overrides the recorded value for one invocation. Use it only for a
stack you know runs there. When a host has both runtimes installed and ``auto``
skips one because its probe failed, the build logs a warning naming the probe.

On a host with the ``podman-docker`` package (standard on RHEL), ``docker`` is
podman's Docker-CLI emulation. It answers every probe because podman does.
``auto`` recognises it and records ``podman``, which is what such a host runs
on.

- A ``docker`` set explicitly on such a host is refused rather than quietly
  re-pointed. The message names the setting to change.
- A ``docker`` that an earlier build recorded from ``auto`` is fixed by running
  ``osprey build`` again.

.. _compose-provider-compatibility:

Supported compose providers
---------------------------

OSPREY supports two compose providers. Before it starts anything,
``osprey up`` asks the runtime which compose implementation stands behind it
(``docker compose version`` / ``podman compose version``). It reads the banner
that comes back. The supported providers are:

* **Docker Compose v2** — any 2.x release.
* **podman-compose 1.0.6 or newer.** 1.0.6 is what EPEL 8 ships as standard
  (EPEL 9 ships 1.5.0), so a site running distribution packages needs nothing
  built from source.

.. raw:: html
   :file: ../../_diagrams/compose-provider-fork.html

The answer comes from the banner, never from the command you typed.
``podman compose`` is a dispatcher that hands the work to whichever provider
the host has configured. So on a host that delegates to Docker Compose v2, a
``podman`` command reports the Docker banner and is treated as Docker Compose
v2. What matters is the provider that parses the command, not the binary that
forwards it.

Anything else is refused before a single container is touched: an unrecognized
banner, Compose v1, a podman-compose below 1.0.6, or a probe that could not run
at all. The refusal names what was wrong and prints what the provider reported,
so you can see what the host actually has.

Refusing rather than guessing is the point. A provider handed the wrong command
shape does not report an unsupported deployment. It starts a stack whose file
paths resolve against the wrong directory.

What the provider changes
-------------------------

One deployment, two command shapes. Nothing you author picks between them;
OSPREY shapes the command from what the probe found.

Docker Compose v2 is handed the rendered files where they sit:

.. code-block:: text

   docker compose --project-directory <repo>
       -f <repo>/build/services/<service>/docker-compose.yml   (one -f per rendered file)
       -f <repo>/build/osprey-labels.override.yml
       --env-file <repo>/.env.shared --env-file <repo>/.env

The render writes ``build/osprey-labels.override.yml`` with the project,
checkout, project-path and config-digest labels for every service it rendered.
It comes last, so its values win. A service template therefore does not have to
spell those labels.

podman-compose is handed one document and one env file:

.. code-block:: text

   podman compose
       -f <repo>/.osprey-compose.yml
       --env-file <repo>/build/.env.merged
   # plus COMPOSE_PROJECT_DIR=<repo> in the command's environment

``.osprey-compose.yml`` is every rendered compose file and the labels override,
merged into a single document at the repository root. ``build/.env.merged`` is
the env chain merged the same way (see :ref:`deployment-env-chain`). Both are
machine artifacts:

- every command that needs them rewrites them from scratch;
- they are kept out of version control and out of every container build
  context;
- ``osprey reset`` removes them.

Neither holds a resolved secret. ``${VAR}`` references are copied through
exactly as the rendered files spell them, so filling them in stays the
runtime's job.

The merge exists because of where podman-compose looks for relative paths:

- Versions 1.0.6 – 1.3.0 change directory twice while assembling a project.
  They end up in the directory of the *first* ``-f`` file, overriding
  ``COMPOSE_PROJECT_DIR``.
- Versions 1.4.1 and newer change directory once, to ``COMPOSE_PROJECT_DIR``.

So a ``-f`` list pointing into ``build/`` resolves every relative path in the
rendered files differently by version. ``env_file:`` entries and bind mounts
resolve against ``build/`` on one version and against the repository root on
the next. That happens from identical inputs, and without an error on either.
One document at the repository root makes both readings land in the same
place.

No compose profiles
-------------------

If you write your own service template, leave ``profiles:`` out of it.
OSPREY renders no ``profiles:`` key and passes no ``--profile``, anywhere.

``deployed_services`` settles which services a deployment runs when it is
built. A service you did not deploy is absent from the render, rather than
present and switched off. podman-compose's handling of profiles varies between
versions. The failure it can produce is the worst kind: a service quietly left
out of the project, with a deploy that reports success. Every deployed service
is in the ``-f`` list, and everything in the ``-f`` list starts.

Where ``${VAR}`` values come from
---------------------------------

Compose fills in ``${VAR}`` placeholders from the env file(s) the command
passes and from the shell you typed it in. The two supported providers disagree
about which wins. For the precedence table, and what ``osprey up`` does about
the disagreement, see :ref:`compose-interpolation-precedence`.

.. _podman-network-backend:

Podman's network backend (required by the Bluesky and optimizer stacks)
-----------------------------------------------------------------------

On Podman, the Bluesky and optimizer stacks need the **netavark** network
backend. The ``bluesky`` service puts its bridge, RE Manager and Redis on an
internal network. It *dual-homes* the queueserver: it attaches it to two
networks. The ``optimizer`` service does the same with the optimizer and its
Redis job store. So these containers must resolve each other by name. On
Podman, that depends on which networking backend the host runs:

* **netavark** (Podman 4.0+ default) — ships aardvark-dns, which serves
  container-name DNS on every network a container is attached to. **Required.**
* **cni** (the legacy backend, still configured on some RHEL 8 hosts) — has no
  aardvark-dns. A dual-homed container receives only its *first* network's
  resolver, so ``bluesky-queueserver`` can never resolve ``bluesky-redis``, nor
  ``optimizer`` its ``optimizer-redis``.

On ``cni`` that container never becomes healthy and ``osprey up`` aborts. The
whole deployment is down, on a DNS fact nothing in the deploy output points at.
So ``osprey up`` checks the backend up front. It refuses, before touching a
container, when ``bluesky`` or ``optimizer`` is deployed on a ``cni`` host.

Only those two stacks need this. Every other service here runs fine on either
backend, and a Docker host is unaffected.

To switch a host over, install ``aardvark-dns`` and set the backend in
``containers.conf``. That file is ``/etc/containers/containers.conf``, or
``~/.config/containers/containers.conf`` for a rootless deployment:

.. code-block:: ini

   [network]
   network_backend = "netavark"

Afterwards, recreate the existing Podman networks, because the old backend
created them. Use ``podman network rm`` for the project's own networks, or
``podman system reset`` on a host with nothing else to lose. Check the result
with ``podman info --format '{{.Host.NetworkBackend}}'``.

What osprey build and osprey up do
==================================

``osprey build`` renders the compose files, and ``osprey up`` starts what it
rendered. This list is what the two commands do, in order; you don't run these
steps yourself. ``osprey build`` does steps 1–9, and ``osprey up`` does step
10:

1. Resolve the deployment repository and load ``config.yml`` via ``ConfigBuilder``.
2. Apply ``deployment.bind_address`` (``127.0.0.1`` by default; change it with ``osprey set config.deployment.bind_address=0.0.0.0`` and rebuild).
3. Render the root ``services/docker-compose.yml.j2`` (shared ``osprey-network``).
4. For each entry in ``deployed_services``: clean and create the build dir, render the service compose template, copy service files.
5. If ``copy_src: true``, copy ``src/`` into the build as ``repo_src/``, plus ``requirements.txt`` and ``pyproject.toml`` (renamed ``pyproject_user.toml``).
6. With ``--dev``, build a wheel from the local Osprey checkout and drop it into the build dir.
7. Copy any ``additional_dirs`` into the build.
8. Auto-create the ``_agent_data/`` subdirectories the deploy step sweeps (``registry_exports_dir``, the one subdirectory declared under ``file_paths``).
9. Write a flattened ``config.yml`` per service. ``${VAR}`` placeholders are preserved (secrets stay out of the rendered output and are resolved at container start).
10. Shell out to ``docker compose`` / ``podman compose``.

.. _deployment-start-waits:

How ``osprey up`` waits
-----------------------

A start is finished when the stack answers, not when compose returns.

**Each compose start is watched and bounded.** Every 30 seconds, a
``compose up`` that is still running prints a progress report. The report
names the project's containers that are not up yet (``created``, ``starting``,
``unhealthy``, ``exited``), and the last line compose wrote.

podman-compose waits on a dependency's health with no limit of its own. A
dependency that is never created (a full disk) or never becomes healthy (a
service that fails at startup) would otherwise hold the start forever. Past the
time limit, the start is stopped and fails. The failure names what it was
waiting on, with the last log lines of every container that exited or is
unhealthy. Ctrl-C prints the same account.

**Web terminals are waited for.** With ``modules.web_terminals`` enabled,
``up`` runs each web-stack service's own healthcheck until it passes. It
reports each one as it does (``· terminal alice ready (21s)``). A terminal that
stops, or does not answer within the window, fails the start. Nothing is torn
down. The message names the terminals that are serving beside the ones that
are not, with the log lines of each one that is not. While a terminal's app is
still starting, nginx answers for it with a page that reloads itself, instead
of a bare ``502 Bad Gateway``.

Both limits are settings, and each takes a number of seconds. The environment
variable wins over the top-level config key.

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - Config key
     - Environment variable
     - Default
   * - ``compose_up_timeout``
     - ``OSPREY_COMPOSE_UP_TIMEOUT``
     - ``1800`` (30 minutes)
   * - ``web_terminal_ready_timeout``
     - ``OSPREY_WEB_TERMINAL_READY_TIMEOUT``
     - ``300`` (5 minutes)

Keeping a Rendered Deployment Up to Date
========================================

A change to the profile reaches the containers only once you rebuild.
``build/`` is a *rendered artifact*: ``osprey build`` writes its ``config.yml``
and service scaffolding from ``profile.yml``. ``osprey up`` starts exactly what
that render describes, and re-derives nothing at start time.

To pick up a profile edit::

   osprey build
   osprey up -d

or in one step::

   osprey up --build -d

Every build wipes and re-renders ``build/``, and preserves what you own:

- the env chain (``.env.shared`` and ``.env``): your provider keys, plus the
  service tokens and passwords your existing container volumes were
  initialized with;
- the agent's memory under ``var/``;
- the repository's ``.git`` history.

``data/`` in the *build zone* (``build/``) is re-materialized from the profile.
A build never touches the *source zone*: the files you author, such as
``profile.yml`` and ``data/`` at the repository root.

Three guards keep ``osprey up`` from starting something you did not mean to:

* **Ownership refusal** — every container records which checkout created it.
  Before ``osprey up`` touches a container, it checks the ones already running
  under this deployment's ``project_name``. If they came from another checkout
  of the repository, it refuses and names that checkout. Stop the deployment
  there, or give this copy its own ``project_name`` in a variant overlay
  (:ref:`deploy-two-checkouts-one-host`). A *variant overlay* is a small
  per-host file beside the profile (:ref:`profile-host-variants`).
* **Drift refusal** — *drift* means the profile changed after the last build.
  At build time, OSPREY stamps a fingerprint of the resolved profile into
  ``.osprey-manifest.json``. ``osprey up`` recomputes the fingerprint and
  compares it with the stamp. If the profile has moved on, ``up`` refuses and
  says what changed, because starting would deploy something other than what
  the profile now describes. Rebuild with ``--build``, or start the old render
  knowingly with ``--as-built``.

  The fingerprint covers ``profile.yml``, the profile's data tree and its
  convention directories (see :doc:`/how-to/build-profiles`). So regenerating
  a channel database or adding a rule trips it too. ``osprey status`` reports
  the same comparison without acting on it.
* **Endpoint summary** — every ``osprey up`` ends with a summary of the
  published service endpoints. When the config declares no web tier, it
  includes an explicit ``web terminal (not configured in this project)`` line.
  So a missing service is a stated fact rather than a silent absence.

Going Deeper
============

Use these subpages when you author or customize a service, expose the stack
beyond one machine, or manage its configuration values:

.. grid:: 1 1 3 3
   :gutter: 3

   .. grid-item-card:: Compose Templates & Images
      :link: compose-templates
      :link-type: doc

      Authoring ``docker-compose.yml.j2`` templates, claiming
      framework-managed templates, and pointing services at mirrored or
      pinned images.

   .. grid-item-card:: Networking
      :link: networking
      :link-type: doc

      ``bridge`` vs ``host`` attachment, bind addresses, and running more
      than one OSPREY project on one host.

   .. grid-item-card:: The .env Chain
      :link: env-chain
      :link-type: doc

      ``.env.shared`` and ``.env``, what a deploy writes back, and the
      warnings and refusals that keep the chain honest.

   .. grid-item-card:: The Project Image
      :link: project-image
      :link-type: doc

      The agent's own container image: what ``osprey build`` generates, who
      owns the ``Dockerfile``, and how to customize it.

   .. grid-item-card:: Search Sidecar
      :link: ../ariel/search-sidecar
      :link-type: doc

      The optional search service deployed beside the stack, and where it
      listens.

.. toctree::
   :hidden:

   compose-templates
   networking
   env-chain
   project-image

.. _development-mode:

Development Mode
================

To run the deployment on your locally installed OSPREY source instead of the
PyPI version, use the ``--dev`` flag. Dev mode is a property of the *build*.
``osprey build --dev`` stages a wheel from your local source into each
service's build context, and marks the render as a dev build. ``osprey up
--dev`` then starts that render with freshly rebuilt images:

.. code-block:: bash

   osprey build --dev
   osprey up --dev

   # or in one step
   osprey up --build --dev

Your dev source is baked into the images at build time. Nothing changes inside
an already-running container.

- ``osprey up --dev`` on a build rendered without ``--dev`` refuses, rather
  than silently starting the published release.
- A plain ``osprey up`` of a dev build warns that the images carry your local
  checkout.

``--dev`` requires the Python ``build`` package:

.. code-block:: bash

   uv pip install build   # or: pip install build

It also needs the checkout's release tags. Each wheel takes its version from
git tags: the framework from ``v*``, and ``osprey-connectors`` from its own
``osprey-connectors-v*``. A checkout with no connectors tag reachable builds
that package as ``0.0.postN``. The framework's requirement on it refuses that
version, so ``--dev`` stops before staging and says so.

To fix it, run ``git fetch --tags``. A fork or an internal mirror must sync the
``osprey-connectors-v*`` tags as well as the ``v*`` ones.

A dev build installs third-party packages at the versions the checkout's
``uv.lock`` holds, not the newest the index has that day. So the images run
your code against the dependency set the repo's own tests ran with.

- Each build context receives ``osprey-lock-constraints.txt``, the lock
  exported with ``uv export --locked``.
- Every install layer in the image is held to it. Nothing is installed as
  ``osprey-framework`` from the index.
- The same applies to the project venv ``osprey build`` creates when it
  installs an editable checkout.

``--dev`` therefore needs ``uv`` on the host and a current lock. With no
``uv.lock``, or one out of date with ``pyproject.toml``, it stops and tells you
to run ``uv lock``. A release build stages no such file, and resolves from the
index as before.

The optimizer image takes torch's CPU build from the PyTorch index
(``OPTIMIZER_TORCH_INDEX_URL``). The lock pins exactly that build
(``torch==<version>+cpu``). A site that empties the variable and serves torch
from its own mirror must mirror the ``+cpu`` build. A mirror carrying only
PyPI's plain torch has no candidate the lock admits, so the dev build of that
image fails.

Troubleshooting
===============

**Services fail to start:**

#. Check the logs (``docker logs <name>`` or ``podman logs <name>``).
#. Verify the ``config.yml`` syntax.
#. Make sure the ``.env`` variables are set.
#. Confirm the service paths contain ``docker-compose.yml.j2``.

**Port conflicts:** run ``lsof -i :<port>`` to find the culprit, then update
``port_host``.

**Template errors:** Verify Jinja2 syntax (``{{var}}`` not ``{var}``);
inspect rendered files under ``build/services/<name>/``.

**Daemon not running:** Both Docker and Podman print platform-specific
hints; on macOS, start Docker Desktop or run ``podman machine start``.

**Runtime did not answer:** each verb waits for ``docker ps`` (or ``podman
ps``) to answer. It says it is still waiting after 5 seconds, and stops after 2
minutes. Run the same command yourself. If it hangs there too, restart the
runtime.

**A start stopped at its time limit, or a terminal was not ready:** the
failure lists what was still pending. It also shows the last log lines of each
container that exited or is unhealthy; that log is usually the cause. If the
start is only slow (images pulled over a slow link, a loaded host), raise the
limit (:ref:`deployment-start-waits`).

**"Unsupported compose provider":** OSPREY cannot drive the host's compose
implementation correctly, so it stopped before starting anything. The message
prints the version banner it got. Compare it against
:ref:`compose-provider-compatibility`. Then either upgrade the provider, or
point ``CONTAINER_RUNTIME`` at the other runtime.

**``--dev`` issues:** Confirm the OSPREY wheel (``.whl``) exists in the
service build directory. Confirm the image was rebuilt after your source
change. To re-render and rebuild it, rerun ``osprey up --build --dev``.

.. seealso::

   :doc:`/reference/cli`
       Full lifecycle command and flag reference.

   :ref:`profile-services`
       Authoritative ``services:`` schema for build profiles.

   :doc:`project-image`
       The *project image* (assistant + web terminal in one container) built
       from the generated ``Dockerfile`` — distinct from the service
       containers this page covers.

   :doc:`/how-to/ariel/search-sidecar`
       The qmd search sidecar — a shared search service some presets
       co-deploy, documented with the ARIEL logbook search it serves.
