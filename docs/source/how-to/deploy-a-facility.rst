.. _how-to-deploy-a-facility:

=================
Deploy a Facility
=================

This page builds one facility deployment from an empty directory to running
containers, using only ``osprey`` commands. Follow it in order and you end up
with a git repository you can commit, review, and hand to a colleague.

The facility is called **Demo Facility**. It runs three services, serves two web
terminals — one read-only, one write-capable — and ships one container the
facility writes itself.

.. dropdown:: What You'll Learn
   :color: primary
   :icon: book

   - Creating a deployment repository with ``osprey init``
   - Editing the profile down to the services your facility actually runs
   - Adding a container the facility owns, with its own image-build job
   - Emitting the CI pipeline and the health check with ``osprey scaffold ci``
   - Rendering the build and bringing the stack up

   **Prerequisites:** A working OSPREY installation, ``git``, and Docker or
   Podman running locally.

   **Time:** 30--45 minutes.

.. tip::

   Read :doc:`build-profiles` first if the words *preset*, *profile* and
   *build* are new. This page assumes you know that the profile is the source
   of truth and ``build/`` is a rendered artifact.


What you are building
=====================

One repository, holding the facility's editable source and the deployment
scaffolding that goes with it:

.. code-block:: text

   demo-facility/
   ├── profile.yml                       the manifest you own — edit this
   ├── data/                             channel databases, knowledge, lattice
   ├── personas/                         one delta per web-terminal persona
   ├── services/facility-mcp/            the facility's own container
   ├── scripts/verify.sh                 the post-deploy health check
   ├── .env                              the deployment's one secret store
   ├── build/                            render target, kept out of git
   ├── var/                              agent memory and audit log, kept out of git
   ├── .gitlab-ci.yml                    emitted from the profile's deploy: block
   ├── ci-extra.yml                      your own CI jobs; never regenerated
   └── .gitignore

Three services run:

.. list-table::
   :header-rows: 1
   :widths: 26 18 56

   * - Service
     - Comes from
     - What it does
   * - ``openobserve``
     - packaged
     - Telemetry store for the agent's logs and metrics.
   * - ``virtual_accelerator``
     - packaged
     - A PyAT soft-IOC serving EPICS Channel Access on port 5064, standing in
       for the real machine.
   * - ``facility-mcp``
     - this profile
     - The facility's own MCP server on port 8200, built from a Dockerfile that
       lives in the profile.

The first two are packaged with OSPREY and come from their own upstreams. The
third is the interesting one: it is the part no framework can ship for you, and
everything about how a facility-owned container is declared, built, and reached
is in :ref:`deploy-a-facility-own-service`.


Step 1 — Create the facility repository
=======================================

.. code-block:: bash

   osprey init demo-facility --preset control-assistant
   cd demo-facility

The command writes the whole repository, runs ``git init`` at its root, and
commits nothing. It prints the handful of entries you edit: the profile, the
data directory, the two personas it rendered, the two env files, and the
README.

It says nothing about CI, which is expected: the profile ships with its
``deploy:`` block commented out, so there are no coordinates to render a
pipeline from. Step 5 fills the block in and Step 7 renders the pipeline from
it.


Step 2 — Trim the profile to this facility's stack
==================================================

Open ``profile.yml``. The ``control-assistant`` preset ships a fuller
stack than Demo Facility runs, so the first edit is subtraction. Delete:

* the top-level ``bluesky:``, ``bluesky_panels:`` and ``dispatch:`` blocks. Each
  of these is a trigger: leaving one in place adds its service to the
  deployment, whatever else you write below.
* from ``skills:`` — ``writing-bluesky-plans`` and ``operating-bluesky-scans``.
* from ``agents:`` — ``logbook-search`` and ``logbook-deep-research``. Both
  query a logbook database at runtime, and this facility does not deploy one.
* from ``web_panels:`` — ``ariel``, ``events`` and ``bluesky``.
* from ``config:`` — the ``claude_code.servers.bluesky.enabled`` line and every
  ``web.panels.events.*`` and ``web.panels.bluesky.*`` override. The panels they
  configure no longer exist.

Keep the ``virtual_accelerator:`` block. It is the trigger for the soft-IOC this
facility drives.


Step 3 — Name the facility and pin its services
===============================================

Still in ``profile.yml``, under ``config:``, set these six values. Some
are already present with a different value; some ship commented out.

.. code-block:: yaml

   config:
     control_system.type: virtual_accelerator
     claude_code.servers.health.enabled: true
     system.timezone: America/Los_Angeles
     facility.name: Demo Facility
     facility.prefix: demo
     deployed_services:
       - openobserve

``control_system.type: virtual_accelerator`` points the agent at the deployed
soft-IOC, so correctors move and BPMs read through exactly the approval and
limit layers a live machine would use. The preset ships ``mock``, which touches
nothing.

``deployed_services`` looks too short, and is not. The ``virtual_accelerator:``
block and the ``services:`` entry you add in Step 4 each append their own
service to this list at build time. Naming ``openobserve`` explicitly is what
keeps the packaged skeleton's ``postgresql`` *out* — declared is not deployed,
and this list is what ``osprey up`` reads.

``facility.prefix`` becomes the container-name prefix for the web tier
(``demo-nginx``, ``demo-web-alice``), so keep it short and distinct from the
project name.

Then rename the two personas, further down the same ``config:`` block under
``modules.web_terminals:``:

.. code-block:: yaml

     personas:
       readonly:
         project: demo-facility-readonly
         project_path: build/demo-facility-readonly
         build_profile: personas/readonly.yml
       readwrite:
         project: demo-facility-readwrite
         project_path: build/demo-facility-readwrite
         build_profile: personas/readwrite.yml

.. important::

   Each persona's ``project`` must equal the basename of its ``project_path``.
   ``osprey build`` derives a persona render's name the same way, which is how
   it lands exactly where the web tier mounts it.


.. _deploy-a-facility-own-service:

Step 4 — Add the facility's own container
=========================================

Declare the service, and the MCP server it serves. Replace the ``services: {}``
line and the commented ``mcp_servers:`` example with:

.. code-block:: yaml

   services:
     facility-mcp:
       template: services/facility-mcp
       config:
         port: 8200

   mcp_servers:
     facility:
       port: 8200
       transport: http
       permissions:
         allow: [machine_status]

``template:`` is profile-relative, so the next thing to do is write that
directory. The port appears twice because it is the same fact told to two
parties: the container publishes it, and the agent dials it.

Now create ``services/facility-mcp/`` with four files.

``requirements.txt``:

.. code-block:: text

   mcp==1.9.4

``server.py`` — a read-only tool over streamable HTTP. Nothing here imports
OSPREY: the agent reaches this container by URL, so the framework never has to
be installed alongside it.

.. code-block:: python

   """Demo Facility's own MCP server: read-only machine-status lookups."""

   from __future__ import annotations

   import json
   import os
   from pathlib import Path

   from mcp.server.fastmcp import FastMCP

   STATUS_FILE = Path(os.environ.get("FACILITY_STATUS_FILE", "/data/machine-status.json"))

   mcp = FastMCP("demo-facility", host="0.0.0.0", port=int(os.environ.get("PORT", "8200")))


   @mcp.tool()
   def machine_status() -> str:
       """Report the control room's current machine state and operating mode."""
       if not STATUS_FILE.is_file():
           return f"No machine status available ({STATUS_FILE} is not present)."
       return json.dumps(json.loads(STATUS_FILE.read_text()), indent=2)


   if __name__ == "__main__":
       mcp.run(transport="streamable-http")

.. note::

   Keep a facility tool like this read-only. Anything that writes to the machine
   belongs behind the control-system connector, which is the single interface
   every write goes through.

``Dockerfile`` — a service directory that carries one earns its own image-build
job in the pipeline:

.. code-block:: dockerfile

   FROM python:3.11-slim

   WORKDIR /app

   COPY requirements.txt ./
   RUN pip install --no-cache-dir -r requirements.txt

   COPY server.py ./

   # Unbuffered so container logs appear as the server writes them rather than
   # when the process exits.
   ENV PYTHONUNBUFFERED=1
   EXPOSE 8200

   CMD ["python", "server.py"]

``docker-compose.yml.j2`` — the same channel every packaged service uses: one
compose template per service directory, rendered by the build.

.. code-block:: jinja

   services:
     facility-mcp:
       image: ${OSPREY_FACILITY_MCP_IMAGE:-{{ (services['facility-mcp'] | default({})).image | default(osprey_labels.project_name ~ '-facility-mcp:local') }}}
       build:
         # With multiple `-f` compose files every relative path resolves against
         # the FIRST file's directory — the compose project dir, not this file's
         # own subdir — so the context is the service directory by name.
         context: ./facility-mcp
         dockerfile: Dockerfile
       # container_name is a HOST-GLOBAL identifier: namespace it per-project so
       # two OSPREY projects can run this service on one host.
       container_name: {{ osprey_labels.project_name }}-facility-mcp
       labels:
         osprey.project.name: "{{ osprey_labels.project_name }}"
         osprey.project.root: "{{ osprey_labels.project_root }}"
         osprey.deployed.at: "{{ osprey_labels.deployed_at }}"
       restart: unless-stopped
       ports:
         - "{{ deployment.bind_address | default('127.0.0.1') }}:{{ (services['facility-mcp'] | default({})).port | default(8200) }}:8200/tcp"
       environment:
         PORT: "8200"
         FACILITY_STATUS_FILE: /data/machine-status.json
         TZ: {{ system.timezone }}
       volumes:
         - facility_mcp_data:/data
       networks:
         - osprey-network
       healthcheck:
         test: ["CMD-SHELL", "python -c \"import socket,sys; s=socket.socket(); s.settimeout(3); sys.exit(0 if s.connect_ex(('localhost', 8200)) == 0 else 1)\""]
         interval: 10s
         timeout: 5s
         retries: 5
         start_period: 10s

   volumes:
     facility_mcp_data:

   networks:
     osprey-network:

The image reference follows the framework convention — environment variable,
then config key, then a local tag. A laptop deploy builds the image from the
sources beside the template; the deploy host selects the pipeline's pushed image
by setting ``OSPREY_FACILITY_MCP_IMAGE``. :doc:`deploy-project` covers the
template variables in full.


Step 5 — Fill in the deployment coordinates
===========================================

Two edits, and they belong together.

First, ``env:`` and the ``deploy:`` block. Replace the preset's ``env.required``
list, and uncomment and fill in the ``deploy:`` block near the bottom of the
file:

.. code-block:: yaml

   env:
     required:
       - DEMO_REGISTRY_TOKEN

   deploy:
     ci: gitlab
     registry:
       url: registry.example.org/accelerator/demo-facility
       token_env_var: DEMO_REGISTRY_TOKEN
     host:
       name: demo-deploy
       fqdn: demo-deploy.example.org
       user: osprey
       project_path: /opt/demo-facility
     image_source: local

Substitute your own registry and host. ``registry.url`` carries no scheme — it
appears verbatim inside an image name. ``host.name`` must be ssh-resolvable for
whoever presses the deploy button, and ``host.project_path`` is where this
repository is checked out on that server. Credentials are *named* here, never
written here.

``image_source: local`` says the deploy host builds the web-terminal images
itself from the rendered persona projects, rather than pulling them from a
registry.

.. important::

   In the same edit, delete the ``image_source: local`` line from the
   ``modules.web_terminals:`` block under ``config:``. The ``deploy:`` block is
   that fact's only home, and the build writes the value into the rendered
   config for you. Saying it in both places is refused by name — the two are
   free to disagree, and the disagreement decides whether the host builds images
   or pulls them.


Step 6 — Add the secrets
========================

The profile owns this deployment's secrets, so that is where they go.

Step 1 may already have written ``.env`` for you, seeded from a matching
key your shell exports — it says so in its output. If that file does not exist
yet, start from the documented list:

.. code-block:: bash

   cp .env.example .env

Either way, edit ``.env`` and set ``ANTHROPIC_API_KEY`` (or the key for
whichever provider your profile names) and ``DEMO_REGISTRY_TOKEN``. The
repository's ``.gitignore`` keeps ``.env`` out of git.

Service credentials such as ``ZO_ROOT_USER_PASSWORD`` stay blank. ``osprey up``
mints a strong value for each unset one on first deploy and writes it back into
``.env``, so a later rebuild comes up on the same credentials the running
volumes were initialized with.

``.env.example`` is written by ``osprey init``. After you change
``env.required``, update the example alongside it — it is the documented list
that whoever sets this up next will read.


Step 7 — Validate, then emit the deployment files
=================================================

.. code-block:: bash

   osprey validate

A valid profile reports its name and its deploy target — ``Deploy: gitlab CI →
osprey@demo-deploy``. If anything is wrong, every problem is reported at once
rather than one per run.

Now render the deployment files from the ``deploy:`` block:

.. code-block:: bash

   osprey scaffold ci

Two files appear:

``.gitlab-ci.yml``
   The pipeline, at the repository root. Its header lists the CI/CD variables
   the pipeline reads, which you set in your GitLab project settings, masked and
   protected: ``DEMO_REGISTRY_TOKEN`` (the registry credential named by your
   ``deploy:`` block) and ``OSPREY_DEPLOY_SSH_KEY``, a *File*-type variable
   holding the private key for the deploy-host account. The SSH key is CI-only —
   it authenticates the deploy job and is never part of the deployment's own
   environment, which is why it is not in ``env.required``.

``scripts/verify.sh``
   The post-deploy health check, at the repository root.

Re-run ``osprey scaffold ci`` whenever the ``deploy:`` block changes. It is
safe to re-run: a file whose content already matches is left untouched, and a
file the scaffolder did not write is reported and left alone unless you pass
``--force``.

``ci-extra.yml``, which ``osprey init`` created in Step 1, is the
facility's own include point. The pipeline includes it after everything the
scaffolder emits, so a job you add there can also override a scaffolded job by
redefining it under the same name. Nothing ever regenerates that file.


Step 8 — Build the project
==========================

.. code-block:: bash

   osprey build

``osprey build`` walks up to the repository's ``profile.yml`` and renders
``build/`` from it, from whichever directory inside the repository you run it.
Watch for these lines in the output:

* ``Injected 1 profile service(s) for deploy`` — ``facility-mcp`` was picked up.
* ``Injected Virtual Accelerator soft-IOC (CA port 5064)``.
* ``Persisted 1 MCP server(s) to config.yml`` — the agent can reach
  ``facility-mcp``.

Confirm the service list is exactly the three you expect:

.. code-block:: bash

   osprey config --rendered | grep -A3 '^deployed_services:'

It should read ``openobserve``, ``facility-mcp``, ``virtual_accelerator``.


Step 9 — Deploy and check
=========================

.. code-block:: bash

   osprey up -d

.. note::

   Running OSPREY from a **source checkout** rather than a released install?
   Add ``--dev`` — ``osprey up`` otherwise refuses, because a container built
   from PyPI would run different code than your checkout, and ``--dev`` builds
   the image from the checkout instead; see :doc:`deploy-project` for that
   workflow.

The first run is slow: the virtual accelerator and the facility's own image are
both built locally. When the containers are up, ``osprey up`` runs
``scripts/verify.sh`` itself and prints a summary of the published endpoints.

.. code-block:: bash

   osprey status

Read the lines *above* the status table first. A drift report there means
``build/`` was rendered from an older profile than the one on disk now, and the
fix is ``osprey build``, not a restart.

You can run the health check by hand at any time, and it is worth doing once
before you trust it:

.. code-block:: bash

   ./scripts/verify.sh              # every probe
   ./scripts/verify.sh services     # one group

It always exits 0 — the output is the report, and the exit code says nothing.
Probes are advisory: a failed probe tells you where to look, and must never be
the reason a deploy is called a failure.


What the pipeline does
======================

The walkthrough above is the local path. The pipeline in ``.gitlab-ci.yml``
takes the same profile through three stages:

**validate** runs on every commit and needs no credentials. It runs
``osprey validate``, then ``osprey build`` with ``--skip-lifecycle
--skip-deps`` — CI has no container runtime for post-build hooks, and nothing
there runs the agent, so its virtual environment would be dead weight. The
render is published as an artifact so a reviewer can see exactly what the
commit produces.

**images** builds one image per facility-owned service that carries a
Dockerfile — here, just ``facility-mcp``. The build context is the service
directory in the *source zone*, not the rendered copy. Every build pushes a
commit-SHA tag; only the default branch moves ``:latest``, so a feature branch
can never reach the deploy host by accident.

**deploy** is manual and default-branch only, serialized by a resource group so
two operators cannot interleave on the host. It re-renders on the host from the
same commit rather than unpacking the artifact, which is what makes the running
stack reproducible from git alone:

.. code-block:: bash

   osprey build
   osprey users env --output .env.users
   osprey up -d

``osprey users env`` writes the env file every per-user web-terminal
container runs with, from the deploy host's own env chain — ``.env.shared``
then ``.env``, with ``.env`` winning on any key both set. Passing
``--output`` is not optional: without it the command writes the assembled
secrets to stdout, which in a pipeline is the job log. ``--output`` also creates
the file at mode ``0600`` from its first byte, which a shell redirect would not.


Changing something later
========================

The loop is always the same: edit the profile, rebuild, redeploy. Run it
anywhere inside the repository:

.. code-block:: bash

   osprey set connector=epics          # or edit profile.yml by hand
   osprey build
   osprey up -d

Or in one step, ``osprey up --build -d``. Every build re-renders everything the
framework owns and preserves what you own: ``.env``, ``var/``, and the
repository's git history. The source zone itself is never touched by a build.

If you changed the ``deploy:`` block, run ``osprey scaffold ci`` again first
so the pipeline and health check match the new coordinates.


Operating it
============

Everything past "the stack is up" runs from anywhere inside the repository:

.. code-block:: bash

   osprey status                  # what is running, where it answers
   osprey logs event_dispatcher   # one service's output; -f to follow
   osprey health                  # config, environment, providers, telemetry
   osprey restart                 # stop and start again
   osprey down                    # stop it, keeping the volumes

``osprey status`` reads and reports — it starts nothing and renders nothing —
so it is safe against a live stack and is the right first command when
something looks wrong. It also says whether ``build/`` still matches
``profile.yml``, which is the same check ``osprey up`` refuses on.

When the answer is not obvious from those, run them inside an OSPREY agent
session in the repository: the agent reads the same output you do, and it has
the deployment's configuration and rendered files at hand.

.. seealso::

   :doc:`build-profiles`
       What lives in a profile, the convention directories, and taking ownership
       of a framework artifact.

   :doc:`deploy-project`
       The container-deployment reference: service configuration, compose template
       variables, image overrides, and the ``--dev`` workflow.

   :doc:`multi-user`
       The web tier this facility deploys — the landing page and one
       containerized terminal per operator.

   :doc:`use-virtual-accelerator`
       The PyAT soft-IOC, and driving it from the agent.

   :doc:`../cli-reference/index`
       Every ``osprey`` command and flag.
