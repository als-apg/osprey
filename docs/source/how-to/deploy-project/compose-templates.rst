============================
Compose Templates and Images
============================

How each service container is defined and where its image comes from —
authoring ``docker-compose.yml.j2`` templates, taking ownership of a
framework-managed template, and pointing services at mirrored or pinned
images. Declaring a service in the first place is covered on the parent
page: :doc:`index`.

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

