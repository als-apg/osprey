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
   - Which compose providers are supported, and what changes between them
   - Network binding and attachment, the ``.env`` chain, and the ``--dev`` workflow

   **Prerequisites:** Docker or Podman installed locally.

.. tip::

   This page is the operator/service-author reference for the container side of
   a deployment. For the end-to-end walkthrough — deployment repository, CI
   pipeline, stack up — follow :doc:`deploy-a-facility`, which also covers the
   commands you run day to day once the stack is up. For the
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
``bluesky_web``, and the multi-user web-terminal stack. Even the minimal
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

The graph store (``graphdb``)
-----------------------------

Alongside ``postgresql`` and ``openobserve``, the ``control-assistant`` preset
deploys a ``graphdb`` service: a Neo4j store holding the facility's knowledge
graph. Its ``services.graphdb`` block names the image, the bolt port the seeder
and the health checks dial (``port_host``), the HTTP port of the Neo4j Browser an
operator opens (``http_port_host``), the Turtle corpus to load (``ttl_path``,
resolved against the ``config.yml`` directory), and the JVM memory the container
runs with.

The block carries **no password**, deliberately — the same convention
``postgresql`` follows. ``osprey up`` mints ``GRAPHDB_PASSWORD`` into the
project ``.env`` when it is unset, and the container reads it from there; a
password written into ``config.yml`` would be read by nobody.

On a first bring-up the deploy starts the store ahead of the rest of the stack,
bootstraps it, and imports ``ttl_path`` — a store that came up empty would answer
every query with zero rows, which reads as wrong data rather than as no data.
Later deploys find the corpus already there and leave it alone. If bootstrapping
or seeding fails the deploy warns and carries on, naming ``osprey knowledge
seed-graph`` (see :doc:`okf-bundle`), the verb that finishes the job by hand.

To query a graph store this deployment does *not* run, give the block an
explicit ``uri:`` and leave ``graphdb`` out of ``deployed_services``. Nothing is
minted or seeded on that path — the graph is somebody else's, so set
``GRAPHDB_PASSWORD`` in the project ``.env`` yourself.

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

.. _compose-provider-compatibility:

Supported compose providers
---------------------------

Choosing the runtime is only half the answer. Before it starts anything,
``osprey up`` asks that runtime which compose implementation stands behind it
(``docker compose version`` / ``podman compose version``) and reads the banner
that comes back. Two are supported:

* **Docker Compose v2** — any 2.x release.
* **podman-compose 1.0.6 or newer.** 1.0.6 is what EPEL 8 ships as standard
  (EPEL 9 ships 1.5.0), so a site running distribution packages needs nothing
  built from source.

The answer comes from the banner, never from the command you typed.
``podman compose`` is a dispatcher that hands the work to whichever provider
the host has configured, so on a host that delegates to Docker Compose v2 a
``podman`` command reports the Docker banner and is treated as Docker Compose
v2. What matters is the provider that parses the command, not the binary that
forwards it.

Anything else is refused before a single container is touched: an unrecognized
banner, Compose v1, a podman-compose below 1.0.6, or a probe that could not run
at all. The refusal names what was wrong and prints what the provider reported,
so you can see what the host actually has. Refusing rather than guessing is the
point — a provider handed the wrong command shape does not report an
unsupported deployment, it starts a stack whose file paths resolve against the
wrong directory.

What the provider changes
-------------------------

One deployment, two command shapes. Nothing you author picks between them;
OSPREY shapes the command from what the probe found.

Docker Compose v2 is handed the rendered files where they sit:

.. code-block:: text

   docker compose --project-directory <repo>
       -f <repo>/build/services/<service>/docker-compose.yml   (one -f per rendered file)
       --env-file <repo>/.env.shared --env-file <repo>/.env

podman-compose is handed one document and one env file:

.. code-block:: text

   podman compose
       -f <repo>/.osprey-compose.yml
       --env-file <repo>/build/.env.merged
   # plus COMPOSE_PROJECT_DIR=<repo> in the command's environment

``.osprey-compose.yml`` is every rendered compose file merged into a single
document at the repository root, and ``build/.env.merged`` is the env chain
merged the same way (see :ref:`deployment-env-chain`). Both are machine
artifacts: rewritten from scratch by every command that needs them, kept out of
version control and out of every container build context, and removed by
``osprey reset``. Neither holds a resolved secret — ``${VAR}`` references are
copied through exactly as the rendered files spell them, so filling them in
stays the runtime's job.

The merge exists because of where podman-compose looks for relative paths.
Versions 1.0.6 – 1.3.0 change directory twice while assembling a project and
end up in the directory of the *first* ``-f`` file, overriding
``COMPOSE_PROJECT_DIR``; 1.4.1 and newer change directory once, to
``COMPOSE_PROJECT_DIR``. A ``-f`` list pointing into ``build/`` therefore
resolves every relative path in the rendered files — ``env_file:`` entries,
bind mounts — against ``build/`` on one version and against the repository root
on the next, from identical inputs and without an error on either. One document
at the repository root makes both readings land in the same place.

No compose profiles
-------------------

OSPREY renders no ``profiles:`` key and passes no ``--profile``, anywhere.
Which services a deployment runs is settled when it is built, by
``deployed_services``: a service you did not deploy is absent from the render
rather than present and switched off. podman-compose's handling of profiles
varies between versions, and the failure it can produce is the worst kind — a
service quietly left out of the project, with a deploy that reports success.
Every deployed service is in the ``-f`` list, and everything in the ``-f`` list
starts.

If you write your own service template, leave ``profiles:`` out of it.

.. _compose-interpolation-precedence:

Where ``${VAR}`` values come from
---------------------------------

Compose fills in ``${VAR}`` placeholders in the rendered compose files from two
sources: the env file(s) the command passes, and the environment of the shell
you typed the command in. **The two supported providers disagree about which of
those wins**, and the disagreement is a straight inversion:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Provider
     - What gets substituted when both sources set a variable
   * - Docker Compose v2
     - The **exported shell value**. ``--env-file`` is its lower-precedence
       source, so a variable exported in your shell overrides the env chain.
   * - podman-compose
     - The **env-file value**. It resolves from the env file before the calling
       shell, so the export reaches nothing.

You do not have to remember this. ``osprey up`` compares the two sources on
every start, and when an exported variable disagrees with the value the env
chain resolves to it warns by name — never by value — and states which value
the provider it just probed will actually use, plus what to do about it. The
reliable habit is to put the value in the env chain and leave your shell out of
it: that is the one gesture that means the same thing on both providers.

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

Every build wipes and re-renders ``build/`` and preserves what you own: the env
chain (``.env.shared`` and ``.env`` — your provider keys, plus the service
tokens and passwords your existing container volumes were initialized with),
the agent's memory under ``var/``, and the repository's ``.git`` history.
``data/`` in the build zone is re-materialized from the profile; the source
zone is never touched by a build.

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
       ports:
         - "{{deployment.bind_address}}:{{services.postgresql.port_host}}:5432"
       environment:
         TZ: {{system.timezone}}
       networks:
         - osprey-network

Common access patterns: ``{{services.<name>.<key>}}``,
``{{file_paths.<key>}}``, ``{{system.<key>}}``, ``{{project_root}}``,
``{{deployment.bind_address}}``, and ``{{osprey_labels.project_name}}`` /
``project_root`` (injected by the deploy engine).

The engine deliberately injects no deploy timestamp. Everything a template
renders comes from the project's configuration, so building the same project
twice produces the same files — which is what lets you diff a rebuilt
``build/`` directory and see only your own changes. A timestamp would also
make every container look changed to the container runtime on each ``osprey
up``, restarting the whole stack for nothing. If you need to know when a
container was started, ask the runtime: ``docker inspect`` reports it as
``.Created``. Earlier versions of these templates carried an
``osprey.deployed.at`` label; a template you have customized that still sets it
keeps building, and the label just renders empty, but you should drop the line.

What that timestamp was doing by accident, two labels now do on purpose::

      osprey.env.digest: "${OSPREY_ENV_DIGEST:-}"
      osprey.config.digest: "${OSPREY_CONFIG_DIGEST:-}"

Your service reads its settings from files — the env chain, and the
``config.yml`` mounted into the container — and the container runtime decides
whether to restart a container by comparing the compose document, which names
neither file's *contents*. So editing ``.env`` or running ``osprey set`` would
leave the running container on the values it started with. Each label carries a
hash of one of those files, which turns such an edit into a document change and
restarts exactly the containers that read it. ``osprey up`` sets both variables
for you; they interpolate to empty if you run ``docker compose`` by hand. **A
service template you wrote yourself should carry both lines** — without them
your service keeps serving its old settings after a change, with nothing to say
so.

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

Edit the moved copy under ``<profile>/services/postgresql/``, then run
``osprey build`` again to deploy it. Every build copies it back and marks it
yours, so later re-renders leave it alone. ``osprey scaffold list`` shows what is
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
   * - bluesky_web
     - ``OSPREY_BLUESKY_WEB_IMAGE``
     - ``services.bluesky_web.image``
   * - virtual_accelerator
     - ``OSPREY_VA_IMAGE``
     - ``services.virtual_accelerator.image``
   * - qmd
     - ``OSPREY_QMD_IMAGE``
     - ``services.qmd.image``

Point either layer at an internal registry mirror or a pinned digest when
your deployment host cannot (or should not) pull public images.

Deploying Prebuilt Images
=========================

Some hosts cannot build images at all — no build tooling, no registry in
reach — and run instead on images built elsewhere and loaded from a tarball.
There, the image build that a :ref:`dev-mode <development-mode>` deploy
normally runs is not merely slow but impossible. The top-level
``prebuilt_images`` key skips it, so ``osprey up --dev`` starts the containers
from the tags already on the host:

.. code-block:: yaml

   # config.yml
   prebuilt_images: true

.. code-block:: bash

   # or, for one shell
   OSPREY_PREBUILT_IMAGES=1 osprey up --dev

``1``, ``true``, ``yes`` and ``on`` turn the switch on; ``0``, ``false``,
``no`` and ``off`` turn it off — case does not matter. The variable wins over
the config key in both directions, so ``OSPREY_PREBUILT_IMAGES=0`` forces a
build for one shell even on a host whose ``config.yml`` pins the key. With
neither set, deploys build as they always have.

The deploy reports ``skipped image build (prebuilt images)`` where it would
otherwise have built. Nothing checks up front that the tags are really
present — a missing one surfaces as compose's own ``No such image`` error,
which names the image to load.

.. _qmd-search-sidecar:

The Search Sidecar (``qmd``)
============================

``qmd`` is a service that indexes the deployment's **markdown corpora** and
answers hybrid keyword-plus-semantic queries over HTTP. Two parts of OSPREY
use it:

* the :doc:`facility-knowledge (OKF) bundle <okf-bundle>` --- its panel and its
  MCP ``search`` tool;
* ARIEL's logbook mirror --- the ``hybrid`` :doc:`search mode
  <ariel/search-modes>` and the ``hybrid_search`` MCP tool.

It is entirely self-contained --- its language models are baked into the image,
so unlike the ``semantic`` search mode it needs no Ollama on the host --- and
the ``ariel-standalone`` and ``control-assistant`` templates deploy it **by
default**, together with its two ARIEL consumers (the ``qmd_export``
enhancement module and the ``hybrid`` search mode). The OKF bundle needs only
``facility_knowledge.bundle_path``, which it already has.

The image is built locally, never pulled. ``osprey build`` renders
``./services/qmd``; ``osprey up`` builds the image on first run and tags it
``<project>-qmd:local``, project-prefixed so two OSPREY projects on one host
cannot race for one tag. The baked-in models make that first build about
2.1 GB of download; later runs reuse the local tag.

Configuration
-------------

.. code-block:: yaml

   services:
     qmd:
       path: ./services/qmd
       port: 8180      # host port clients talk to
       interval: 30    # fallback corpus-sweep period, seconds

   deployed_services:
     - qmd

Those three keys are the whole schema. Notably **there is no
``bind_address`` here** --- see `Where the sidecar listens`_ below.

Neither consumer strictly needs the sidecar --- OKF search falls back to
substring matching and hybrid logbook search reports an outage --- so a
deployment that does not want a second index can switch it off. That is three
edits, not one, because any subset leaves either a container nobody queries or
a search mode with nothing to search: comment out the ``qmd:`` entry under
``services:`` and the ``- qmd`` line under ``deployed_services:``, and disable
both ``ariel.search_modules.hybrid`` and
``ariel.enhancement_modules.qmd_export``.

``interval`` is a ceiling on staleness, not the usual lag: a corpus writer
touches a ``.qmd-touch`` marker file and the sidecar re-indexes within one poll.
The interval only catches writers that forgot to touch it. Raise it on a large
corpus --- a sweep that finds nothing changed still costs about 12.5 seconds at
135,000 documents, so the 30-second default leaves the loop busy roughly 42% of
the time discovering nothing.

What gets mounted
-----------------

Each corpus is bind-mounted **read-only** into the sidecar at
``/corpus/<collection>``, and the same list generates the sidecar's collection
config --- so a corpus can never end up mounted without a collection, or
declared without a mount. Read-only is deliberate: the sidecar indexes these
trees, and everything that *writes* them lives outside the container.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Collection
     - Source
     - Present when
   * - ``okf``
     - ``facility_knowledge.bundle_path``
     - the bundle path is set
   * - ``ariel``
     - ``ariel.enhancement_modules.qmd_export`` → ``mirror_path``
     - the export is enabled and names a path

The index itself lives in a named volume rather than a bind mount. It is
derived data the sidecar owns end to end, it is large, and rebuilding it costs
about **41 minutes** at ALS scale --- which is precisely why it must survive a
container recreate. The service's health check allows a one-hour start period
for the same reason: the sidecar refuses to open its port until the index is
built and provably non-empty, and a container that is working correctly should
not be reported unhealthy for most of its first hour.

Sharing the knowledge bundle
----------------------------

The OKF bundle is different from the other corpora: web terminals write to it.
So the *deployment's* bundle is bind-mounted **read-write** into every
``web-<user>`` service whose persona enables facility knowledge, at a target
computed from that persona's own project directory inside its container. There
is one directory, not a copy per user --- and it deliberately **shadows the
copy baked into each persona image**, because the bundle is operational
knowledge that changes far more often than images are rebuilt.

Sharing it works through a Unix group, and ``osprey up`` sets both halves up:

* The shared corpus directory is made **setgid and group-writable** (mode
  ``2770`` --- note that the ``other`` triad is deliberately left unset). An
  operator's pre-existing directory only ever *gains* bits here; it never
  loses any.
* Each entitled ``web-<user>`` service is rendered with
  ``group_add: ["<gid>"]``.

Both halves are needed, and this is the part that is easy to get wrong: setgid
makes **new files inherit the directory's group**. It does *not* make any
container process a member of that group. Without ``group_add`` the container
is not in the group at all, and the group-write bit grants it nothing.

One limit follows from how Unix works rather than from OSPREY: setgid fixes who
*owns* a new file, not its permission bits, which come from the writing
process's umask --- normally ``rw-r--r--``. So the supported cross-container
operation is **read and index**, not overwrite. See
:ref:`One bundle, many terminals <shared-bundle-multi-user>` for the operator's
view of the same mechanism.

Disk footprint
--------------

Measured against a real 134,996-entry ALS logbook:

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - Component
     - Size
   * - Markdown mirror (logical)
     - 41 MB
   * - Markdown mirror (allocated on disk)
     - 553 MB
   * - qmd index
     - 695 MB
   * - **Total per 135,000 entries**
     - **1.25 GB** (~925 MB per 100,000)

The gap between logical and allocated size is the point to budget for: the
mirror is one small file per entry, so it is dominated by filesystem block
overhead rather than by content. Chunking is close to 1:1 for logbook
micro-documents (2,000 documents produced 2,001 chunks), so the index grows with
entry count rather than with entry length.

.. _qmd-where-the-sidecar-listens:

Where the sidecar listens
-------------------------

The sidecar publishes port **8180**. qmd's own daemon runs on **8181** on the
container's internal loopback and is fronted by a small forwarder. That split is
not cosmetic: qmd hardcodes a loopback-only bind with no option to change it,
which makes it unreachable from any other container. Only the forwarder owns a
routable port.

.. warning::

   **The sidecar has no authentication.** No token, no TLS, no per-caller
   identity --- it answers any request that reaches it, over the whole indexed
   corpus. That is safe exactly as long as only this host can reach it.

   This is why ``bind_address`` is **not** a ``services.qmd`` key. Like every
   other service, the sidecar publishes on the project-wide
   ``deployment.bind_address`` (default ``127.0.0.1``), so a deployment cannot
   put an unauthenticated search endpoint on an interface the rest of the stack
   is not already on. Moving that one key off loopback moves this service too.

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

.. _deployment-network-attachment:

Network attachment: ``bridge`` or ``host``
------------------------------------------

By default every service joins the compose-managed project network
(``osprey-network``) and publishes the ports it wants reachable. That is
``bridge``, and it is what a deployment gets when it says nothing.

Some services cannot work that way. A service that has to see broadcast traffic
— control-system protocols, device discovery — or that has to reach ports other
software already publishes on the machine needs the host's own network
namespace instead. That is ``host``:

.. code-block:: yaml

   # in profile.yml — the event dispatcher and its workers
   dispatch:
     network: host

   # a facility-owned service
   services:
     my-service:
       template: services/my-service
       config:
         network: host

``dispatch.network`` is deliberately **one knob for two services**. The event
dispatcher and its workers talk to each other over addresses the build writes,
so a dispatcher on the compose network and workers on the host's could not
reach each other at all. Writing ``network:`` on ``services.event_dispatcher``
or ``services.dispatch_worker`` individually is rejected by ``osprey build``,
which tells you to set ``dispatch.network`` instead.

Under ``network: host`` the render changes in four ways:

* No ``ports:`` block. There is nothing to publish — the container's listening
  socket *is* a host socket, on the port the service was configured with.
* ``network_mode: host`` replaces the service's ``osprey-network`` membership.
* Services bind **loopback**, ``127.0.0.1``, rather than every interface. On
  the compose network, binding every interface is what makes a service
  reachable by name and the network itself is the boundary; on the host network
  there is no such boundary, so the default is the private one. Reaching the
  event dispatcher from off-host is then a deliberate act:
  ``services.event_dispatcher.bind``.
* Addresses OSPREY writes between services become ``localhost:<port>`` instead
  of compose service names — the dispatcher's target for its workers, and the
  Google Chat and Nextcloud bridges' URLs for the dispatch pair.

Services that talk to each other have to be on the same side of that boundary,
and ``osprey build`` refuses to render a deployment where they are not: a
co-deployed bridge on the compose network with a host-mode dispatch pair, the
reverse of that, or any address naming a service across the boundary. The build
also refuses a service that declares ``network: host`` whose rendered compose
file does not carry it, since the setting would otherwise be quietly inert.
Every one of those failures names the service and the key to change.

Running more than one project on one host
-----------------------------------------

Two OSPREY projects on the same machine compete for the same host ports, and
compose's own report of that is a bare "address already in use" partway through
starting. ``osprey up`` therefore checks every host port this deployment needs
before it touches a container — ports two of its own services would both
publish, and ports something else is already listening on — and stops if any is
taken. Every conflict is listed with the config key that moves it
(``services.postgresql.port_host``, ``dispatch.worker_port_base``, and so on).
A listener that belongs to this project's own containers is not a conflict, so
restarting a running stack stays quiet.

Host-mode services are part of that check even though they publish no ports:
their host bindings are worked out from the rendered configuration instead of
read out of a ``ports:`` block. That covers the case most likely to catch you
out — two projects whose dispatch pairs are both on the host network take the
same default ports (``8020`` for the dispatcher, ``9190`` upward for the
workers), and no ``ports:`` line anywhere would have shown it. Give the second
project its own ``dispatch.dispatcher_port`` and ``dispatch.worker_port_base``.
A facility service you place on the host network is covered the same way,
read from its ``services.<name>.port`` key; one without that key cannot be
checked, and ``osprey up`` says so rather than skipping it silently.

.. _deployment-env-chain:

Environment Variables (the ``.env`` chain)
==========================================

A deployment reads its environment from two files at the repository root:

.. list-table::
   :header-rows: 1
   :widths: 20 55 25

   * - File
     - What belongs in it
     - Tracked in git?
   * - ``.env.shared``
     - What the whole site shares: a proxy, a facility hostname, a port
       everyone uses. Never a secret — this file is committed.
     - yes
   * - ``.env``
     - This host's own values, and every secret: API keys, service tokens,
       passwords.
     - no

**The local file wins.** A variable set in both takes its value from ``.env``,
on every path that reads them — the deploy, the CLI, the containers. That is
the whole rule: same syntax, same variables, ``.env.shared`` simply sits lower.
Setting a key in ``.env`` is how one host departs from a shared default, and
there is nothing else to do about it.

Both files stay on the host. Neither ever enters a container image: they are
read at run time and handed to the container runtime, which uses them to fill
in the ``${VAR}`` placeholders in the rendered compose files. A variable reaches
a running container only where a template maps it in. *How* the two files are
handed over differs by compose provider (see
:ref:`compose-provider-compatibility`); what they resolve to does not.

``.env`` has to exist. Rather than start a stack whose every ``${VAR}``
substitutes to nothing, ``osprey up`` refuses when the file is missing. On an
interactive terminal it first offers to seed one, but only when your shell has
the key to seed it with — this deployment's own provider auth variable,
exported. Otherwise start from the example:

.. code-block:: bash

   cp .env.example .env
   # Edit .env with your actual values

See :ref:`profile-secrets`.

The ``.env*`` family
--------------------

Two files are yours to edit and one is documentation. The rest are written for
you — most of them derived, rewritten whenever a command needs them, and not
worth editing because the next command overwrites them:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - File
     - Role
   * - ``.env.shared``
     - edit — shared defaults, the same on every host
   * - ``.env``
     - edit — this host's values and every secret
   * - ``.env.example``
     - docs — every variable this deployment reads, with no values
   * - ``.env.users``
     - machine — the env file every per-user web-terminal container runs with,
       derived from the chain (multi-user deployments only)
   * - ``.env.auth``
     - both — the web terminals' password hashes and cookie-signing secrets,
       minted by the deploy, but also where you put an OIDC client id and
       secret by hand (multi-user deployments only; see :doc:`multi-user`)
   * - ``build/.env.merged``
     - machine — the chain collapsed into one file, for compose providers that
       accept only one
   * - ``build/.env.chain-state.json``
     - machine — fingerprints of the shared values as of the last deploy, so a
       stale local pin can be spotted. It stores digests, never values.

The generated ``.gitignore`` keeps all of them out of version control except
``.env.shared`` and ``.env.example``, which carry nothing a host may not share.

.. note:: Upgrading a multi-user deployment

   The web terminals' env file changed name to ``.env.users``. ``osprey up``
   does the rename for you the next time you deploy, and if both names are
   present it keeps the new one and removes the leftover, naming both paths.
   Only ``osprey up`` does this, so a stack you stop before you next deploy
   still carries the old name — and because the web stack's compose file names
   ``.env.users``, ``osprey down`` fails with an env-file-not-found error until
   the rename has happened. Do it yourself in that case:

   .. code-block:: bash

      mv .env.production .env.users

What a deploy writes back
-------------------------

``osprey up`` also *writes* to ``.env``. On first deploy it mints any missing
service tokens and passwords (for example ``EVENT_DISPATCHER_TOKEN``,
``ZO_ROOT_USER_PASSWORD``, or ``ARIEL_DB_PASSWORD``) so no service ever starts
on a blank or publicly-known credential, appends them under a "Minted by
deploy" heading, and restricts the file to owner-only permissions.
``osprey build`` appends the pointers it derives from what it just rendered,
under a "Derived by build" heading.

Both writers are append-only, and a value already on file always wins. That is
what makes the stack reproducible: a later start comes up on the same
credentials the running containers were initialized with, instead of minting a
second set they do not trust. There is no second copy anywhere — the ``.env``
beside ``profile.yml`` is the deployment's whole secret store — so back it up.

Minted values only ever land in ``.env``, never in ``.env.shared``. A minted
credential belongs to this host, and ``.env.shared`` is committed.

What a deploy tells you about the chain
---------------------------------------

Two layered files make one new mistake possible, so every ``osprey up`` reports
on the chain before it starts anything. Variable names are printed; values never
are.

* **Overrides** — the keys ``.env`` overrides in ``.env.shared`` are listed
  once, by name, as information. That is the chain working as intended.
* **Stale pins** — a warning, and it is reserved for one exact case: ``.env``
  still holds the value ``.env.shared`` carried *before* the shared file
  changed. That is the signature of a value copied from the default of the day
  and then forgotten rather than one this host chose, and the stack starts on
  the superseded value looking perfectly healthy. To adopt the shared value,
  remove the key from ``.env``; to keep a local value deliberately, set it to
  the value this host actually wants. The warning repeats until you do one or
  the other.
* **Chain drift** — a refusal. Which env files the stack reads is decided when
  the project is rendered, not when it starts: adding ``.env.shared`` to a
  project built without one puts none of its values into the containers, and
  removing one leaves the render pointing at a file that is gone. ``osprey up``
  refuses and names ``osprey build``. Re-render, then start.
* **Shell exports** — when a variable exported in your shell disagrees with the
  value the chain resolves to, ``osprey up`` names it and says which of the two
  the compose provider it just probed will actually substitute (see
  :ref:`compose-interpolation-precedence`).

.. note::

   Postgres reads ``ARIEL_DB_PASSWORD`` (as ``POSTGRES_PASSWORD``) only when
   initializing a **fresh** data volume. A volume created before the password
   was minted keeps its original password; the ``${ARIEL_DB_PASSWORD:-ariel}``
   fallback — applied by the compose template and by the DSN the agent derives
   from ``services.postgresql`` — keeps such deployments working. To adopt
   the minted password, remove the ``ariel_postgres_data`` volume and redeploy
   (this deletes the stored logbook data — re-ingest afterwards).

.. _development-mode:

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

**"Unsupported compose provider":** the host's compose implementation is not
one OSPREY can drive correctly, and it stopped before starting anything. The
message prints the version banner it got; compare it against
:ref:`compose-provider-compatibility` and either upgrade the provider or point
``CONTAINER_RUNTIME`` at the other runtime.

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
