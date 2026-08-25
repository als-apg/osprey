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
   - The build-arg extension points for site-specific installs
   - Building behind a proxy — including one that re-signs TLS with a site CA
     — and staging model files for the search sidecar
   - Path relocation with ``osprey build --runtime-root``
   - The root-to-``osprey`` privilege split, air-gapped images, and
     Kubernetes notes

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

Then open http://localhost:8087. The container's ``osprey web`` prints a login
URL (``…/?token=…``) to its logs at startup; open that URL to set your session
cookie, after which it redirects to the clean address. Anything that can read
those logs can read the token, so treat both like a password. If your ``.env``
sets ``OSPREY_TERMINAL_SECRET`` itself, the container uses that value and
explains why it is printing no URL — browse to
``http://localhost:8087/?token=<that value>`` instead. Secrets are
passed at runtime via ``--env-file`` — the ``.dockerignore`` guarantees
``.env`` itself never enters the image.

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
   * - ``OSPREY_SITE_CA``
     - ``""``
     - Name of a site CA file staged in the build context, for networks
       where a proxy re-signs TLS with its own CA — see
       `TLS-Intercepting Proxies`_ below. Unset, the CA step does nothing.

(A sixth ARG, ``OSPREY_DEV``, is used internally by ``osprey up
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

TLS-Intercepting Proxies
------------------------

Some proxies do more than relay: they terminate every HTTPS connection and
re-sign it with the site's own CA (Squid is a common example). Delivering the
proxy values as above is then not enough — every fetch inside the build (apt,
npm, pip, the optional ``osprey vendor fetch``) fails cert verification,
because nothing in the image trusts that CA yet.

The ``OSPREY_SITE_CA`` build argument closes the gap. A Dockerfile ``COPY``
cannot reach outside the build context, so first stage a copy of the CA bundle
beside the Dockerfile, then name it:

.. code-block:: bash

   cd build/.image/my-project
   cp /etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem site-ca.pem
   docker build -t my-project --build-arg OSPREY_SITE_CA=site-ca.pem \
     -f build/Dockerfile .

That first path is where a RHEL-family build host keeps its system bundle
(individual CA files also live under ``/etc/pki/ca-trust/source/anchors/``).
The path is a host fact, not an image fact: the image is Debian, and inside it
the staged CA is installed with ``update-ca-certificates`` — before the first
fetch — into the merged Debian bundle at ``/etc/ssl/certs/ca-certificates.crt``.

Installing the CA into the system store only covers the tools that read it.
Each tool family finds its trust through a different variable, so the image
also points all four at that merged bundle, for build steps and the running
container alike:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Variable
     - Who reads it
   * - ``NODE_EXTRA_CA_CERTS``
     - Node (and so npm) — Node ships its own CA list and consults no
       system store without it.
   * - ``PIP_CERT``
     - pip — which otherwise trusts only its bundled ``certifi`` store.
   * - ``SSL_CERT_FILE``
     - Python's ``ssl`` module (any default SSL context).
   * - ``REQUESTS_CA_BUNDLE``
     - The ``requests`` library.

With no CA staged, the whole mechanism is a no-op: the variables restate each
tool's default and the build behaves exactly as before.

.. note::

   The same interception can fail ``osprey build`` **on the host**, before any
   container is involved — and the symptom is confusing, because plain
   connectivity checks all pass. While preparing the project environment,
   pip's build-isolation subprocess dies with ``SSL: CERTIFICATE_VERIFY_FAILED``
   fetching a build backend such as ``hatchling``, even though ``curl`` and the
   system Python verify the same endpoints fine. The difference is the trust
   store: those tools read the system bundle, which knows the proxy's CA, while
   the venv's pip trusts only its bundled ``certifi`` store, which cannot build
   the proxied chain. Point the certifi-backed clients at the system bundle
   before building:

   .. code-block:: bash

      export SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt \
             REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt \
             PIP_CERT=/etc/ssl/certs/ca-bundle.crt

   (``/etc/ssl/certs/ca-bundle.crt`` is the RHEL-family spelling of the system
   bundle; Debian-family hosts use ``/etc/ssl/certs/ca-certificates.crt``.)
   These are the same three variables the image sets for its own inside —
   exported here, they cover the host-side build, which reads them from the
   shell rather than from the deployment's env files.

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

.. _containerize-privilege-split:

The Privilege Split
===================

The agent never runs as root: **the agent CLI refuses to run in
bypassPermissions mode as root**, and running it there would in any case put
the whole project within its own reach. The image satisfies that in a way that
still lets the container fix itself at startup — it *starts* as root, does the
few things only root can do, and drops to an unprivileged ``osprey`` user
before the command that serves requests exists.

Two areas of the project, two owners:

.. list-table::
   :header-rows: 1
   :widths: 26 16 58

   * - Area
     - Owned by
     - What lives there
   * - The render — ``build/``, including ``config.yml``, ``.mcp.json`` and
       the ``.claude/`` artifacts
     - ``root``
     - Everything that decides what the agent may do: the permission lists,
       the hook wiring, the approval policy, the limits table. An agent that
       could rewrite these could rewrite its own limits, and no deny list or
       approval prompt would outlive that
   * - The state zone — ``var/``
     - ``osprey``
     - What the running deployment legitimately writes: agent data and
       API-call logs (``var/agent_data/``), the config backups every config
       write takes first (``var/agent_data/config-backups/``), and the
       protected-write audit ledger (``var/audit/``). The facility-knowledge
       bundle under ``build/data/`` joins them when the deployment has one,
       because the agent drafts into it

The ``osprey`` user's uid and gid are pinned to ``1000:1000`` rather than left
to whatever the base image hands out, and the image states the pair in
``OSPREY_RUNTIME_UID``. Anything outside the container that has to agree with
it — the multi-user stack's per-user volume seeding, for one — reads it out of
the image instead of guessing.

What the entrypoint does
------------------------

There is no ``USER`` instruction. ``ENTRYPOINT`` is the ``build/entrypoint.sh``
that ``osprey build`` renders beside ``config.yml``, and it runs as root:

#. **Re-renders drifted agent artifacts**, and only those that actually
   drifted, so a container whose ``config.yml`` has not changed rewrites
   nothing.
#. **Restores your own artifact bodies** from the durable store, so a
   recreated container comes back with the artifacts you saved rather than the
   ones the image shipped. In a container that store is always the per-user
   volume — the profile baked into the image is never treated as an editable
   source, whatever its permissions look like — so this step really does put
   your claimed bodies back into a render nothing else can write. A store
   record naming a protected path is skipped and recorded instead, under the
   surface name ``scaffold_restore``; see :doc:`protected-set`.
#. **Hands the state zone back** to the ``osprey`` user — only the paths root
   actually left behind — because both steps above wrote into ``var/`` as
   root, including the protected-write audit ledger the running server has to
   keep appending to.
#. **Drops privileges** with ``exec gosu osprey "$@"``. Because it execs the
   arguments it was handed, overriding the command on ``docker run`` still
   goes through the drop.

Both maintenance steps fail open and say so: a container running slightly
stale artifacts and reporting it in its logs beats one that will not boot. The
privilege drop is the opposite — a missing ``gosu``, or an image with no
``osprey`` user, is fatal, because continuing would run the agent as root.

Two environment variables carry the consequences:

- ``OSPREY_RENDER_ZONE_READONLY=1`` tells the server the render is not
  writable by this process, so it skips the startup regen and restore the
  entrypoint has already done, and logs what a regen *would* have changed
  instead of failing on a tree it cannot write.
- ``OSPREY_RUNTIME_UID=1000:1000`` states the user the entrypoint drops to.

.. note::

   **Run the image with** ``--user`` **and both maintenance steps are
   skipped.** They cannot write a root-owned render from an unprivileged
   process, and ``gosu`` cannot drop to a user it is not root to become — so
   the entrypoint says so loudly in the log and runs the command directly. The
   agent artifacts are then whatever the image was built with. That is a valid
   way to run it; it is not the way to run it if you expect a configuration
   change to be picked up at start.

One tier gets one more file
---------------------------

A deployment can grant a single persona the ability to edit its own
configuration — the agent's ``setup_patch`` tool and the browser's Config
panel. The image reads that grant off the render's own permissions: where the
setup tool is left out of the deny list, and only there, ``build/config.yml``
is handed to the ``osprey`` user, which is also what lets the Config panel's
write land. Every other render leaves the file root-owned, which is what makes
the boundary a fact of the filesystem rather than a permission list the agent
is asked to respect. Grant the two surfaces together, the way the bundled
tiers do — a persona handed the panel alone would find the file it has to
write still owned by root. See :doc:`multi-user/tiers` for the tiers this exists
for.

On a bare host it is ownership you set up
-----------------------------------------

Run the same project with ``osprey web`` on a host instead of in a container
and the split is **not** there. There is no second user: the server runs as
you, and it keeps the self-healing startup behaviour the container moved into
its entrypoint — it re-renders drifted artifacts and restores your saved
artifact bodies in-process, because there is no root step ahead of it to have
done so.

The parts that are code rather than file ownership do carry over. The restore
refuses a protected path on a bare host exactly as it does under the
entrypoint, because both call the same function — a private copy for the
container would be a second gate to keep in step, and the one running as root
is the worst place to discover a drift.

The rest is worth saying plainly: on a bare host the render is only as
protected as the file ownership you give it. If that matters for your
deployment, run the container image, or own the render as a user other than
the one the server runs as.

Runtime State and Volumes
=========================

Two kinds of state are worth persisting across container restarts:

.. code-block:: bash

   docker run --rm -p 8087:8087 --env-file .env \
     -v my-project-agent-data:/app/my-project/var/agent_data \
     -v my-project-home:/home/osprey \
     my-project

- ``var/agent_data/`` — API call logs, generated data artifacts, and the
  backup every config write takes before it writes
  (``config-backups/<name>.bak``). The backup anchors on the project root, so
  it lands in the same place whether the project is flat, a deployment
  repository, or this image — unless the deployment relocates
  ``agent_data.base_dir``, which moves the backups with it.
- ``/home/osprey`` — the agent CLI's per-user state (sessions, credentials);
  set ``CLAUDE_CONFIG_DIR`` if you want it somewhere more explicit.

Kubernetes notes
----------------

- Give each user/instance a PVC for ``/home/osprey`` (or
  ``CLAUDE_CONFIG_DIR``) and one for ``var/agent_data/`` — session state does
  not survive pod rescheduling otherwise.
- The image has no ``USER``: it starts as root and drops to uid 1000 itself
  (see `The Privilege Split`_). A ``securityContext`` with
  ``runAsNonRoot: true`` therefore refuses the pod, and one pinning
  ``runAsUser: 1000`` starts it but makes the entrypoint skip the startup
  regen and restore, exactly as ``--user`` does. Let the entrypoint do the
  drop, or accept that the agent artifacts are whatever the image was built
  with.
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
  ``osprey web --host 0.0.0.0 --port 8087`` (the deployment is discovered
  from the working directory); override it to run a process supervisor if you
  add sidecars.
- **Carry the edit in the profile**: put your Dockerfile in the profile's
  ``project/`` mirror (above) so every rebuild lands it again.
- **Template-level override**: an app bundle can ship its own
  ``apps/<bundle>/Dockerfile.j2``, which takes precedence over the framework
  template at build time — use this when every project built from a bundle
  needs the same customization.

.. warning::

   **A Dockerfile you forked keeps the recipe you forked.** A copy in your
   deployment's ``project/`` mirror is laid over the render on every build,
   and no regeneration ever rewrites it — that is what the mirror is for, and
   it is also why your copy never picks up anything the framework's recipe
   gained afterwards. `The Privilege Split`_ is exactly such a gain: a forked
   Dockerfile still creates the ``osprey`` user, hands it the *whole* project
   with a blanket ``chown``, and switches to it with ``USER`` — so the render
   it is meant to protect stays writable by the agent.

   Seven changes port the split into a forked recipe. Or delete your fork,
   rebuild, and re-apply your customization on top of the current one, which is
   usually the smaller job.

   .. list-table::
      :header-rows: 1
      :widths: 36 64

      * - Change
        - What it looks like
      * - Install ``gosu``
        - Add it to the ``apt-get install`` line in the layer that already
          installs ``curl``, ``git`` and Node — not in a layer of its own near
          the end, or every render change re-runs ``apt``
      * - Pin the runtime uid and gid
        - ``groupadd --gid 1000 osprey && useradd --uid 1000 --gid 1000 …``,
          so the pair ``OSPREY_RUNTIME_UID`` declares is the pair the
          container really runs as. The multi-user stack's per-user volume
          seeding chowns each volume to the declared value, so a recipe that
          lets ``useradd`` pick the next free id hands the volume to the wrong
          user
      * - Narrow the ownership change
        - ``chown -R osprey:osprey`` on ``var/`` only, plus
          ``build/data/facility_knowledge`` when the deployment renders one.
          Not the project root — that blanket chown is what hands the render
          away
      * - Hand over ``config.yml`` for one tier only
        - ``chown osprey:osprey .../build/config.yml``, and only in the image
          for a persona allowed to edit the deployment. Every other image
          leaves it root-owned
      * - Remove ``USER osprey``
        - The container has to start as root for the entrypoint's two writes;
          leaving ``USER`` in place makes the entrypoint skip both — it warns
          and runs the command anyway, so the image's own artifacts are what
          you get
      * - Point ``ENTRYPOINT`` at the rendered script
        - ``ENTRYPOINT ["/app/<project>/build/entrypoint.sh"]``. It ships
          inside the render, so name it at that path rather than copying it
          somewhere else. Keep the existing ``CMD``: the entrypoint execs it
      * - Add the two ``ENV`` lines
        - ``ENV OSPREY_RUNTIME_UID=1000:1000`` and
          ``ENV OSPREY_RENDER_ZONE_READONLY=1``. Without the second, the
          server retries the writes the entrypoint already made and logs
          failures it cannot act on

.. seealso::

   :doc:`deploy-project/index`
       Service containers (databases, MCP servers) via ``osprey up`` —
       the complement to the project image on this page.

   :doc:`../cli-reference/index`
       ``osprey build --runtime-root`` and ``osprey vendor`` reference.
