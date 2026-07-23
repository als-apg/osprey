=======================
Containerize a Project
=======================

How to build and run the container image that ``osprey build`` generates for
every project.

.. dropdown:: What You'll Learn
   :color: primary
   :icon: book

   - What the generated ``Dockerfile`` / ``.dockerignore`` are and who owns them
   - Building and running the image (ports, secrets, volumes)
   - The three build-arg extension points for site-specific installs
   - Path relocation with ``osprey claude regen --runtime-root``
   - Air-gapped images, the non-root requirement, and Kubernetes notes

   **Prerequisites:** Docker (or Podman) installed; a project built with
   ``osprey build``.

Overview
========

Every project built by ``osprey build`` includes a reference container
recipe at the project root:

- ``Dockerfile`` — an image definition that installs Claude Code and OSPREY,
  copies the project in, relocates its recorded paths, and serves the web
  terminal.
- ``.dockerignore`` — keeps secrets (``.env``) and host-specific state
  (``.venv``, ``.git``, ``_agent_data/``) out of the image.

Both files are **generated once and then yours**: edit them freely, but keep
``.dockerignore`` in place — the build depends on it, and it is what keeps
your ``.env`` secrets out of the image. ``osprey claude regen`` never touches
either file. To get a fresh copy, rebuild the project with
``osprey build --force`` — note that this overwrites any edits you made to
the Dockerfile and ``.dockerignore`` (only ``.env``, ``_agent_data/``, and
``.git`` survive a force rebuild).

.. note::

   This page covers the **project image** — one container that runs the
   assistant and its web terminal. ``osprey deploy`` manages the project's
   *service* containers (databases, MCP servers) — see :doc:`deploy-project`
   — but the two meet in one place: a deploy that includes the dispatch
   worker builds this same project image (tagged ``<project>:local``) for
   the worker to run.

Quickstart
==========

.. code-block:: bash

   cd my-project          # the directory osprey build created
   docker build -t my-project .
   docker run --rm -p 8087:8087 --env-file .env my-project

Then open http://localhost:8087. Secrets are passed at runtime via
``--env-file`` — the ``.dockerignore`` guarantees ``.env`` itself never
enters the image.

Build Arguments
===============

The image exposes these knobs for site-specific builds:

.. list-table::
   :header-rows: 1
   :widths: 22 22 56

   * - ARG
     - Default
     - Purpose
   * - ``OSPREY_PIP_SPEC``
     - ``osprey-framework``
     - pip requirement for OSPREY. Override with a ``git+https`` URL to pin
       an unreleased build or an internal mirror.
   * - ``PIP_NO_PROXY``
     - ``""``
     - Hosts exempted from any proxy during ``pip install`` (e.g. an
       internal GitLab serving the OSPREY package).
   * - ``OSPREY_OFFLINE``
     - ``"0"``
     - ``"1"`` vendors web assets (JS/CSS/fonts) into the image via
       ``osprey vendor fetch`` so the web UI works without internet access.
   * - ``CLAUDE_CLI_VERSION``
     - pinned at build time
     - Version of the agent CLI installed into the image. The default pin
       matches the framework version that generated the Dockerfile; override
       to test a newer CLI without regenerating the project.

(A fifth ARG, ``OSPREY_DEV``, is used internally by ``osprey deploy up
--dev`` to install a locally built wheel; you normally never set it by hand.)

Example — install OSPREY from an internal mirror behind a proxy, with
vendored assets for an air-gapped host:

.. code-block:: bash

   docker build -t my-project \
     --build-arg OSPREY_PIP_SPEC="git+https://git.example.gov/tools/osprey.git@main" \
     --build-arg PIP_NO_PROXY="git.example.gov" \
     --build-arg OSPREY_OFFLINE=1 .

.. warning::

   Build-arg **values persist in the image history** (``docker history``).
   Never put credentials in ``OSPREY_PIP_SPEC`` URLs for images you
   distribute — prefer `Docker build secrets
   <https://docs.docker.com/build/building/secrets/>`_ or a credential-free
   internal mirror.

Path Relocation
===============

A project built on a host records host paths in ``config.yml``
(``project_root``, ``execution.python_env_path``). The generated Dockerfile
fixes both during the image build:

.. code-block:: docker

   RUN osprey claude regen --project /app/my-project --runtime-root /app/my-project

``--runtime-root`` rewrites ``project_root`` in ``config.yml``
(comment-preserving), replaces a recorded ``python_env_path`` that doesn't
exist in the container with the image's interpreter, and re-renders the
Claude Code artifacts (``.mcp.json``, ``CLAUDE.md``, ``.claude/``) against
the new root. This works for projects built with or without
``osprey build --runtime-root``.

Why Non-Root
============

The image creates and switches to an unprivileged ``osprey`` user because
**Claude Code refuses to run in bypassPermissions mode as root**. The agent
CLI itself is installed as a pinned global npm package, so it is runnable
by any user — keep the non-root user if you customize the recipe.

Runtime State and Volumes
=========================

Two kinds of state are worth persisting across container restarts:

.. code-block:: bash

   docker run --rm -p 8087:8087 --env-file .env \
     -v my-project-agent-data:/app/my-project/_agent_data \
     -v my-project-home:/home/osprey \
     my-project

- ``_agent_data/`` — executed scripts, user memory, API call logs.
- ``/home/osprey`` — Claude Code's per-user state (sessions, credentials);
  set ``CLAUDE_CONFIG_DIR`` if you want it somewhere more explicit.

Kubernetes notes
----------------

- Give each user/instance a PVC for ``/home/osprey`` (or
  ``CLAUDE_CONFIG_DIR``) and one for ``_agent_data/`` — session state does
  not survive pod rescheduling otherwise.
- The container already runs as a non-root user, so a restricted
  ``securityContext`` (``runAsNonRoot: true``) works out of the box.
- Expose port ``8087`` (or override the ``CMD`` with ``--port``).

Troubleshooting
===============

**pip fails building** ``accelerator-toolbox`` **on Apple Silicon** — parts
of OSPREY's dependency chain ship prebuilt wheels for ``linux/amd64`` only,
so an arm64 build must compile them from source, which is where it breaks.
Build (and run) the amd64 image under Docker Desktop's emulation instead; it
matches the usual amd64 deployment target:

.. code-block:: bash

   docker build --platform linux/amd64 -t my-project .
   docker run --platform linux/amd64 --rm -p 8087:8087 --env-file .env my-project

Customizing
===========

The file is yours — common edits:

- **Layer a site image on top**: build the generated image as a base, then
  ``FROM`` it in a small site Dockerfile that adds credentials helpers,
  enterprise ``managed-settings.json``, or extra processes.
- **Change the entrypoint**: the default ``CMD`` runs
  ``osprey web --host 0.0.0.0 --port 8087 --project /app/<project>``;
  override it to run a process supervisor if you add sidecars.
- **Template-level override**: a build profile's app bundle can ship its own
  ``apps/<bundle>/Dockerfile.j2``, which takes precedence over the framework
  template at build time — use this when every project built from a bundle
  needs the same customization.

.. seealso::

   :doc:`deploy-project`
       Service containers (databases, MCP servers) via ``osprey deploy`` —
       the complement to the project image on this page.

   :doc:`../cli-reference/index`
       ``osprey claude regen --runtime-root`` and ``osprey vendor`` reference.
