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

.. _deployment-image-overrides:

Overriding Service Images
=========================

Every service image resolves through the same three-layer chain — an
environment variable wins, then a ``config.yml`` key, then the packaged
default. Thirteen images, one row each:

.. list-table::
   :header-rows: 1
   :widths: 20 30 28 22

   * - Service
     - Environment variable
     - Config key
     - Packaged default
   * - postgresql
     - ``OSPREY_POSTGRES_IMAGE``
     - ``services.postgresql.image``
     - upstream pin
   * - openobserve
     - ``OSPREY_OPENOBSERVE_IMAGE``
     - ``services.openobserve.image``
     - upstream pin
   * - mongodb
     - ``OSPREY_MONGODB_IMAGE``
     - ``services.mongodb.image``
     - upstream pin
   * - event_dispatcher
     - ``OSPREY_DISPATCH_IMAGE``
     - ``services.event_dispatcher.image``
     - ``<project>-dispatch``
   * - dispatch_worker
     - ``OSPREY_WORKER_IMAGE``
     - ``services.dispatch_worker.image``
     - ``<project>``
   * - nextcloud_bridge
     - ``OSPREY_NEXTCLOUD_BRIDGE_IMAGE``
     - ``services.nextcloud_bridge.image``
     - ``<project>-nextcloud-bridge``
   * - gchat_bridge
     - ``OSPREY_GCHAT_BRIDGE_IMAGE``
     - ``services.gchat_bridge.image``
     - ``<project>-gchat-bridge``
   * - bluesky
     - ``OSPREY_BLUESKY_BRIDGE_IMAGE``
     - ``services.bluesky.image``
     - ``<project>-bluesky-bridge``
   * - bluesky (Tiled sidecar)
     - ``OSPREY_TILED_IMAGE``
     - ``services.bluesky.tiled_image``
     - upstream pin
   * - bluesky (Redis sidecar)
     - ``OSPREY_BLUESKY_REDIS_IMAGE``
     - ``services.bluesky.redis_image``
     - upstream pin
   * - bluesky_web
     - ``OSPREY_BLUESKY_WEB_IMAGE``
     - ``services.bluesky_web.image``
     - ``<project>-bluesky-web``
   * - virtual_accelerator
     - ``OSPREY_VA_IMAGE``
     - ``services.virtual_accelerator.image``
     - ``<project>-va``
   * - qmd
     - ``OSPREY_QMD_IMAGE``
     - ``services.qmd.image``
     - ``<project>-qmd``

Point either of the first two layers at an internal registry mirror or a
pinned digest when your deployment host cannot (or should not) pull public
images.

Five of the thirteen are **upstream pins** — images somebody else publishes,
named exactly as they publish them. The other eight are **built by OSPREY**
from your project, and their default reference is assembled rather than
fixed: a project name, a per-service suffix, and the two axes below.

.. _deployment-image-axes:

The two image axes
------------------

An OSPREY-built default is always spelled the same way::

   <registry>/<project><service suffix>:<tag>

Two stack-wide settings supply the ends of that name, so an entire deployment
can be moved to a registry — or to a different tag — without touching any of
the thirteen rows above:

.. list-table::
   :header-rows: 1
   :widths: 16 30 24 30

   * - Axis
     - Environment variable
     - Config key
     - When neither is set
   * - Registry
     - ``OSPREY_IMAGE_REGISTRY``
     - ``images.registry``
     - no prefix at all
   * - Tag
     - ``OSPREY_IMAGE_TAG``
     - ``images.tag``
     - ``local``

.. code-block:: yaml

   # config.yml — every OSPREY-built image comes from the mirror,
   # at the tag the pipeline pushed
   images:
     registry: registry.example.org/accelerator
     tag: "2026.08.1"

.. code-block:: bash

   # or, for one build
   OSPREY_IMAGE_REGISTRY=registry.example.org/accelerator \
     OSPREY_IMAGE_TAG=2026.08.1 osprey build

For each axis the environment variable wins, then the config key, then the
packaged default. A blank value counts as unset on both layers — an
exported-but-empty variable is how a shell spells "I did not set this", so a
stray ``OSPREY_IMAGE_TAG=`` cannot render an image reference with no tag. A
trailing slash on the registry is optional: one is added if you leave it out,
and never doubled if you put it in.

Two things about *when* and *where* this applies are worth having straight:

* **The axes are the innermost layer, not an override.** They decide what the
  packaged default of an OSPREY-built image is. A ``services.<name>.image``
  pin still beats them for that one service, and an ``OSPREY_<SVC>_IMAGE``
  variable still beats both. Setting an axis moves everything you have not
  pinned individually.
* **The axes are read when the compose files are rendered**, not when the
  containers start. Export them for the ``osprey build`` that produces the
  deployment; the per-image ``OSPREY_<SVC>_IMAGE`` variables, by contrast, are
  filled in by compose at ``osprey up`` time. With neither axis set the render
  is what it always was — ``<project>:local`` and its siblings — so a
  deployment that never heard of them is unaffected.

The axes never touch the five upstream pins. Prefixing ``mongo:7`` with your
registry would name an image that exists in no registry; mirror those through
their own row instead.

The web tier names its registry separately
------------------------------------------

The web tier — the landing page and the one containerized terminal per
operator, described in :doc:`/how-to/multi-user` — carries its own, older spelling of
the same two ideas, and the two vocabularies coexist rather than merging:

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
is the switch in the next section. If your route is a self-built image rather
than a mirror, :doc:`/how-to/containerize-project` covers the air-gapped build trio —
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
