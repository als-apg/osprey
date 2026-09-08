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

The service templates under ``<project>/services/`` are framework-managed, so
an edit made to one in place is overwritten on the next build; to keep an edit,
claim the template into the build profile with ``osprey scaffold claim
services/<name>`` (:ref:`profile-claim`).

Overriding Service Images
=========================

Every service image resolves through the same three-layer chain — an
environment variable wins, then a ``config.yml`` key, then the packaged
default — and the two stack-wide axes assemble the default name of an
OSPREY-built image. The thirteen images, the two axes and their precedence are
catalogued in :ref:`config-deployment`.

The web tier names its registry separately
------------------------------------------

The web tier — the landing page and the one containerized terminal per
operator, described in :doc:`/how-to/web-terminal/multi-user/index` — carries
its own, older spelling of the registry and tag axes, and the two vocabularies
coexist rather than merging:

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * -
     - Service images
     - Web tier
   * - Registry
     - ``images.registry`` / ``OSPREY_IMAGE_REGISTRY``
     - ``registry.url``
   * - Tag
     - ``images.tag`` / ``OSPREY_IMAGE_TAG``
     - ``modules.web_terminals.image_tag``

Neither pair reaches the other's images. ``images.registry`` does not move the
web images, and ``registry.url`` does not move the service images. A
deployment that mirrors everything sets both pairs, and sets them to the same
place — that is the one case where the divergence costs you a line of config
rather than nothing.

Whether the web tier pulls those images or builds them on the deploy host is a
third, separate setting: ``modules.web_terminals.image_source``
(``registry``, the default, or ``local``). It is unrelated to the axes, and it
is also the setting that governs the persona and auth-sidecar builds — see
below.

.. _deployment-mirror-channel:

Mirroring every image into one registry
---------------------------------------

A host behind a strict firewall, or with no route to the public internet at
all, needs every image it starts to come from a registry it can reach. There
are four channels to point at that mirror, and a deployment that misses one
fails at ``up`` on the image it forgot:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Images
     - How to point them at your mirror
   * - The eight OSPREY-built images
     - Set the registry axis once — ``images.registry``, or
       ``OSPREY_IMAGE_REGISTRY`` for a single build.
   * - The five upstream pins
     - One row at a time: ``services.<name>.image`` (or the row's
       ``OSPREY_..._IMAGE`` variable) naming your mirrored copy.
   * - The web tier's images
     - ``registry.url`` plus ``modules.web_terminals.image_tag``.
   * - nginx and the auth sidecar
     - ``modules.web_terminals.nginx_image`` and
       ``modules.web_terminals.auth.image``. **These two carry no environment
       variable**, so a mirror reaches them through ``config.yml`` only —
       there is no one-shell equivalent.

.. code-block:: yaml

   # config.yml — all four channels, one mirror
   images:
     registry: registry.example.org/accelerator
     tag: "2026.08.1"

   registry:
     url: registry.example.org/accelerator

   services:
     postgresql:
       image: registry.example.org/mirror/pgvector/pgvector:pg16
     mongodb:
       image: registry.example.org/mirror/mongo:7
     openobserve:
       image: registry.example.org/mirror/openobserve:v0.14.4
     bluesky:
       tiled_image: registry.example.org/mirror/tiled:0.2.12
       redis_image: registry.example.org/mirror/redis:7.4-alpine

   modules:
     web_terminals:
       image_tag: "2026.08.1"
       nginx_image: registry.example.org/mirror/nginx:1.27-alpine
       auth:
         image: registry.example.org/accelerator/demo-assistant-auth:2026.08.1

Copy only the rows for services you actually deploy — the upstream pins for a
service that is not in ``deployed_services`` are never rendered.

Naming the mirror is half the job: the host also has to stop *building*. That
is the switch in :ref:`Deploying Prebuilt Images <deployment-prebuilt-images>`. If your route is a self-built image rather
than a mirror, :doc:`project-image` covers the air-gapped build trio —
``OSPREY_PIP_SPEC`` for an internal package mirror, ``PIP_NO_PROXY`` to exempt
it from the proxy, and ``OSPREY_OFFLINE=1`` to vendor the web assets into the
image.

.. _deployment-prebuilt-images:

Deploying Prebuilt Images
=========================

Some hosts cannot build images at all — no build tooling, no registry in
reach — and run instead on images pulled from a mirror or loaded from a
tarball. There, an image build is not merely slow but impossible. The
top-level ``prebuilt_images`` key turns building off, so the deploy starts the
containers from the tags already on the host:

.. code-block:: yaml

   # config.yml
   prebuilt_images: true

.. code-block:: bash

   # or, for one shell
   OSPREY_PREBUILT_IMAGES=1 osprey up

``1``, ``true``, ``yes`` and ``on`` turn the switch on; ``0``, ``false``,
``no`` and ``off`` turn it off — case does not matter. The variable wins over
the config key in both directions, so ``OSPREY_PREBUILT_IMAGES=0`` forces a
build for one shell even on a host whose ``config.yml`` pins the key. With
neither set, deploys build as they always have.

The switch covers both ways a build can start:

* In :ref:`dev mode <development-mode>` it skips the wheel-and-image build
  step, and the deploy reports ``skipped image build (prebuilt images)`` where
  it would otherwise have built.
* In ordinary (non-dev) mode there is no build step to skip — but compose
  would still build any service whose compose document carries a ``build:``
  block the first time it brings it up. The switch passes ``--no-build``, so
  compose starts what is there instead. Nothing is reported as skipped,
  because nothing was scheduled.

**What the switch does not reach.** It governs compose's implicit builds only.
The persona images and the auth sidecar are built explicitly, by a different
mechanism, and stay governed by ``modules.web_terminals.image_source``: a host
that cannot build at all needs ``image_source: registry`` *as well as*
``prebuilt_images``. Setting only one of the two is the usual way a genuinely
build-less host still ends up trying to build something.

Nothing checks up front that the tags are really present. A missing one
surfaces as compose's own ``No such image`` error, which names the image to
load or pull.

Together with the mirror settings above, the pull-only shape of a restricted
deployment is: point all four image channels at the mirror, set
``prebuilt_images: true``, and set ``modules.web_terminals.image_source:
registry``.
