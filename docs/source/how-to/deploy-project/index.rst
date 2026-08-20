====================
Container Deployment
====================

How to run a deployment's containerized services.

.. dropdown:: What You'll Learn
   :color: primary
   :icon: book

   - What ``osprey build`` and ``osprey up`` do, and when you need them
   - Configuring services in ``config.yml`` (minimal example)
   - Keeping a rendered deployment up to date, and the ``--dev`` workflow
   - Which compose providers are supported, and what changes between them
   - Where the deeper reference lives: compose templates, networking, and the ``.env`` chain each have their own subpage

   **Prerequisites:** Docker or Podman installed locally.

.. tip::

   This page is the operator/service-author reference for the container side of
   a deployment. For the end-to-end walkthrough — deployment repository, CI
   pipeline, stack up — follow :doc:`/how-to/deploy-a-facility`, which also covers the
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

Full command and flag reference: :doc:`/cli-reference/index`.

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

.. raw:: html
   :file: ../../_diagrams/compose-provider-fork.html

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

Going Deeper
============

The rest of the container story lives on three focused subpages — the
reference material you need when you author or customize a service, expose
the stack beyond one machine, or manage its configuration values:

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

.. toctree::
   :hidden:

   compose-templates
   networking
   env-chain

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

   :doc:`/cli-reference/index`
       Full lifecycle command and flag reference.

   :ref:`profile-services`
       Authoritative ``services:`` schema for build profiles.

   :doc:`/how-to/containerize-project`
       The *project image* (assistant + web terminal in one container) built
       from the generated ``Dockerfile`` — distinct from the service
       containers this page covers.

   :doc:`/how-to/search-sidecar`
       The qmd search sidecar — a shared search service some presets
       co-deploy, documented with the facility-knowledge system it serves.
