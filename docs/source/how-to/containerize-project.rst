=======================
Containerize a Project
=======================

How to build and run the container image that ``osprey build`` generates for
every project.

.. dropdown:: What You'll Learn
   :color: primary
   :icon: book

   - What the generated ``Dockerfile`` / ``.dockerignore`` are and who owns them
   - How to keep a customized Dockerfile across rebuilds
   - Building and running the image (ports, secrets, volumes)
   - The three build-arg extension points for site-specific installs
   - Building behind a proxy, and staging model files for the search sidecar
   - Path relocation with ``osprey build --runtime-root``
   - Air-gapped images, the non-root requirement, and Kubernetes notes

   **Prerequisites:** Docker (or Podman) installed; a project built with
   ``osprey build``.

Overview
========

Every project built by ``osprey build`` includes a reference container
recipe at the project root:

- ``Dockerfile`` — an image definition that installs the agent CLI and OSPREY,
  copies the project in, relocates its recorded paths, and serves the web
  terminal.
- ``.dockerignore`` — keeps secrets (``.env``) and host-specific state
  (``.venv``, ``.git``, ``var/``) out of the image.

Both files are **generated, then yours to edit in place**: change them freely,
but keep ``.dockerignore`` — the build depends on it, and it is what keeps your
``.env`` secrets out of the image. ``osprey build`` never touches either
file.

An edit made in ``build/`` lasts only until the next ``osprey build``, which
re-renders both from the framework. To make a customization durable, put your
version in the source zone's ``project/`` mirror instead:

.. code-block:: text

   my-facility/
     project/
       Dockerfile          # copied verbatim onto the render, every build

The mirror is applied after the framework render, so your copy wins each time.
See :doc:`build-profiles` for the profile's convention directories.

.. note::

   This page covers the **project image** — one container that runs the
   assistant and its web terminal. The lifecycle verbs manage the deployment's
   *service* containers (databases, MCP servers) — see :doc:`deploy-project/index`
   — but the two meet in one place: a deploy that includes the dispatch
   worker builds this same project image (tagged ``<project>:local``) for
   the worker to run.

Quickstart
==========

.. code-block:: bash

   osprey build                            # renders the container repo too
   cd build/.image/my-project              # the context the build rendered
   docker build -t my-project -f build/Dockerfile .
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

(A fifth ARG, ``OSPREY_DEV``, is used internally by ``osprey up
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

Building Behind a Proxy
-----------------------

Two things have to line up for a build on a proxied network: the proxy values
have to reach the build, and the tools inside it have to find them under the
names they actually read.

**Getting the values in.** Docker takes them as build arguments —
``docker build --build-arg HTTP_PROXY=... --build-arg HTTPS_PROXY=...``. The
uppercase spellings are predeclared by the builder, so no ``ARG`` line is needed
for them. Podman does this for you: it forwards the host's proxy environment
into every build unless you pass ``--http-proxy=false``.

**Finding them inside a build step.** ``apt`` and ``pip`` read only the
lowercase ``http_proxy`` / ``https_proxy`` / ``no_proxy``, and an uppercase
build argument does not answer to a lowercase name. So every ``RUN`` that
reaches the network in a generated Dockerfile opens with a bridge line:

.. code-block:: dockerfile

   RUN export http_proxy="${http_proxy:-${HTTP_PROXY:-}}" \
              https_proxy="${https_proxy:-${HTTPS_PROXY:-}}" \
              no_proxy="${no_proxy:-${NO_PROXY:-}}"; \
       apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates

Each expansion prefers a lowercase value that is already set, falls back to the
uppercase one, and otherwise defaults to empty — so a build with no proxy
exports three empty variables and behaves exactly as it did before.

The two mechanisms answer different questions and work together: the build
arguments (or Podman's forwarding) deliver the values, and the bridge closes the
upper/lowercase gap inside the step. ``PIP_NO_PROXY`` remains the knob for the
other half — which hosts to *skip* the proxy for. In the project image's
dependency layer the bridge runs first and the existing
``NO_PROXY="$PIP_NO_PROXY"`` export follows it, so a site that needs an
exemption there sets ``PIP_NO_PROXY`` exactly as before while ``http_proxy`` and
``https_proxy`` are bridged normally.

Prefetched Models for the Search Sidecar
========================================

A deployment that includes the qmd search sidecar builds a second image, and
that one downloads three model files from ``huggingface.co`` during its build.
On a network with no route there, stage the files on the host once and name that
directory of prefetched models in the project's ``config.yml``:

.. code-block:: yaml

   services:
     qmd:
       models_dir: /srv/osprey/qmd-models   # absolute host path

That one key does both halves of the job: the build skips the downloads, and the
directory is bind-mounted read-only into the container at the location the image
looks for models. The container-side path is fixed by the image and is not
configurable.

Staging the files
-----------------

The three files must sit in that directory under exactly these names:

.. code-block:: text

   hf_ggml-org_embeddinggemma-300M-Q8_0.gguf
   hf_ggml-org_qwen3-reranker-0.6b-q8_0.gguf
   hf_tobil_qmd-query-expansion-1.7B-q4_k_m.gguf

These are cache names, not the names the files download under. The model loader
recognizes a staged file only in this form; under any other name it reads as
**absent**, and the sidecar tries to fetch it again — into a read-only mount, on
a network that cannot reach the model host.

Copy the files from a machine that can reach ``huggingface.co`` (or from the
model cache of one that has already built the sidecar image), rename them, and
check them before setting the key:

.. code-block:: bash

   sha256sum /srv/osprey/qmd-models/*.gguf

Compare the three digests against the pins in the qmd service's Dockerfile
(``services/qmd/Dockerfile`` in the rendered deployment), which carries both the
source URL and the expected SHA256 of each file.

What gets checked
-----------------

- **Before anything starts.** If ``services.qmd.models_dir`` is set but the
  directory is missing, or a model is absent, misnamed, or zero bytes, the
  deploy refuses up front and prints the expected filenames along with the
  staging steps. Nothing has been started at that point.
- **At container start.** The entrypoint verifies all three SHA256 digests
  against the pins the image was built with, every time, and fails with a clear
  message rather than letting the sidecar crash-loop unnoticed. A stamp file
  records what it already verified, so ordinary restarts stay fast.

Leaving ``models_dir`` unset keeps the default: the models are baked into the
image at build time, which needs a build host that can reach ``huggingface.co``.
Those downloads use ``curl``, which reads the uppercase ``HTTPS_PROXY``
directly, so an online build behind a proxy needs nothing beyond the build
arguments described above.

Path Relocation
===============

A render made on a host records that host's path in ``config.yml`` as
``project_root``, which would be wrong inside an image. Nothing in the
Dockerfile fixes that: ``osprey build`` renders a second copy of the deployment
specifically for the container, against its ``/app`` path rather than the
building host's, and that copy is what the image build uses as its context. The
recorded ``project_root``, the agent artifacts (``.mcp.json``, ``CLAUDE.md``,
``.claude/``) and every path they name are already the container's before the
first ``docker build`` layer runs.

That is why the by-hand build below runs from ``build/.image/<name>/`` rather
than from the repository root. ``osprey build --runtime-root PATH`` is the same
mechanism exposed directly, for a render whose output will run somewhere other
than where it was made.

Why Non-Root
============

The image creates and switches to an unprivileged ``osprey`` user because
**the agent CLI refuses to run in bypassPermissions mode as root**. The CLI
itself is installed as a pinned global npm package, so it is runnable
by any user — keep the non-root user if you customize the recipe.

Runtime State and Volumes
=========================

Two kinds of state are worth persisting across container restarts:

.. code-block:: bash

   docker run --rm -p 8087:8087 --env-file .env \
     -v my-project-agent-data:/app/my-project/var/agent_data \
     -v my-project-home:/home/osprey \
     my-project

- ``var/agent_data/`` — API call logs and generated data artifacts.
- ``/home/osprey`` — the agent CLI's per-user state (sessions, credentials);
  set ``CLAUDE_CONFIG_DIR`` if you want it somewhere more explicit.

Kubernetes notes
----------------

- Give each user/instance a PVC for ``/home/osprey`` (or
  ``CLAUDE_CONFIG_DIR``) and one for ``var/agent_data/`` — session state does
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
  enterprise settings, or extra processes.
- **Change the entrypoint**: the default ``CMD`` runs
  ``osprey web --host 0.0.0.0 --port 8087 --project /app/<project>``;
  override it to run a process supervisor if you add sidecars.
- **Carry the edit in the profile**: put your Dockerfile in the profile's
  ``project/`` mirror (above) so every rebuild lands it again.
- **Template-level override**: an app bundle can ship its own
  ``apps/<bundle>/Dockerfile.j2``, which takes precedence over the framework
  template at build time — use this when every project built from a bundle
  needs the same customization.

.. seealso::

   :doc:`deploy-project/index`
       Service containers (databases, MCP servers) via ``osprey up`` —
       the complement to the project image on this page.

   :doc:`../cli-reference/index`
       ``osprey build --runtime-root`` and ``osprey vendor`` reference.
