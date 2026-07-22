.. _how-to-multi-user:

=====================
Multi-User Deployment
=====================

The multi-user Web Terminal turns one OSPREY project into a small shared
product: a landing page where each member of your team picks their name, and a
private, containerized Web Terminal behind each card — all served from a single
host, brought up with a single ``osprey deploy up``.

.. dropdown:: What You'll Learn
   :color: primary
   :icon: book

   - Why the single-user ``osprey web`` workflow is unchanged, and when each
     mode is the right tool
   - The three ideas behind the multi-user stack: one container per user,
     personas as capability tiers, and one nginx front door
   - The ``modules.web_terminals`` config block that switches it on
   - Day-to-day operations: adding, reseeding, and removing users

   **Prerequisites:** The concepts need none. To stand the stack up you'll
   want Docker (or Podman) and a generated project — the
   ``multi-user-demo`` preset ships everything pre-wired, and
   :doc:`multi-user-demo` walks through it end to end.

Single-user is still the front door
===================================

Nothing about multi-user changes the everyday workflow. From any project
directory,

.. code-block:: bash

   osprey web

still launches the single-user Web Terminal as one local process — no
containers, no proxy, ready in seconds at ``http://127.0.0.1:8087``. It remains
the fastest way to try OSPREY and the right tool whenever one person sits in
front of one machine; :doc:`operate` covers it.

The multi-user stack is strictly opt-in. It lives in a
``modules.web_terminals`` block in the project's ``config.yml``, read only by
``osprey deploy`` (and validated by ``osprey scaffold web-terminals lint``).
``osprey web`` never looks at it — so a project that carries the block (the ``multi-user-demo`` preset
ships one) still runs single-user exactly as before. Reach for the multi-user
stack when several people need their own terminal on a shared machine, and
stay with ``osprey web`` for everything else.

How it works
============

.. mermaid::

   flowchart LR
       B[Browser] -->|:9080| N[nginx landing page]
       N -->|/u/alice/| A[alice's terminal container]
       N -->|/u/bob/| Bo[bob's terminal container]
       A --- S[(shared services:<br/>databases · telemetry)]
       Bo --- S

Three ideas carry the whole design:

**One container per user.** Every name on the roster gets its own Web Terminal
container, plus two named volumes that belong to the *user*, not the container:
a workspace volume (the files the agent reads and writes) and an
agent-configuration volume. Upgrading or rebuilding an image replaces the
container but never touches those volumes, so a user's files and settings
survive every redeploy. On first start, ``osprey deploy up`` seeds each user's
configuration volume automatically — no per-user setup steps.

**A persona is a capability tier — and a whole project.** Users map to
*personas*, and each persona is its own rendered OSPREY project with its own
``config.yml``, permissions, skills, and tool servers. Because permissions are
a property of a project, the tiers are genuinely different agents — not one
agent with a UI toggle. The demo ships two: a *read-only* tier and a
*read-write* tier — the same agent and tool surface, differing on exactly one
config key (``control_system.writes_enabled``). ``osprey deploy up``
auto-renders any persona project that doesn't exist yet and builds its
container image locally, so no registry or CI is involved.

**One front door.** An nginx reverse proxy serves the landing page and proxies
``/u/<name>/`` to that user's container. The per-user containers are pinned to
the loopback interface, so nginx is the only network path in. The landing
page's user cards are a convenience for choosing an identity on a trusted
machine — not authentication (see the **Authentication and TLS** tab below).

Running the stack
=================

.. tab-set::

   .. tab-item:: Switching it on

      The whole feature is one config block. This is what the
      ``multi-user-demo`` preset renders into ``config.yml``:

      .. code-block:: yaml

         modules:
           web_terminals:
             enabled: true
             image_source: local       # deploy up builds persona images itself
             nginx_port: 9080          # the landing page
             web_base_port: 9091       # per-user ports: base + user index
             artifact_base_port: 9291
             ariel_base_port: 9391
             lattice_base_port: 9491
             channel_finder_base_port: 9591
             default_persona: readonly
             users:
             - alice                   # bare name → default_persona
             - name: bob
               index: 1
               persona: readwrite
             personas:
               readonly:
                 project: multi-user-demo-readonly
                 project_path: ../multi-user-demo-readonly
                 build_profile: multi-user-demo-readonly
               readwrite:
                 project: multi-user-demo-readwrite
                 project_path: ../multi-user-demo-readwrite
                 build_profile: multi-user-demo-readwrite

      The ``users`` list is the roster — the single source of truth for who
      exists. A bare name resolves to ``default_persona``; an entry with an
      explicit ``persona`` picks its tier. Each user's host ports are
      ``base + index`` in every port family — one family per companion panel
      (artifact gallery, ARIEL, channel finder, lattice dashboard, …) plus the
      terminal itself — so alice (index 0) serves her terminal on ``9091`` and
      bob (index 1) on ``9092``. A panel whose ``*_base_port`` you don't set
      falls back to its built-in default, so the block above lists them only
      to make the layout visible.

      .. tip::

         Give every roster entry an explicit ``index`` before you ever
         *remove* one. Once indices are pinned, deleting an earlier user can
         no longer shift a later user's ports out from under a running
         deployment.

   .. tab-item:: Day-to-day operations

      The roster drives everything: edit it, then let ``osprey deploy``
      reconcile reality against it.

      .. list-table::
         :header-rows: 1
         :widths: 34 66

         * - Task
           - Command
         * - **Add a user**
           - Add a roster entry with the next free ``index``, then
             ``osprey deploy up``. The new container comes up with freshly
             allocated ports and a seeded workspace; existing users are
             untouched.
         * - **Reseed workspaces**
           - ``osprey deploy seed [USER]`` re-applies the seeded configuration
             for one user, or for everyone when ``USER`` is omitted.
         * - **Remove one user**
           - ``osprey deploy decommission USER`` stops and removes the user's
             container. Their volumes are **retained** by default; add
             ``--archive`` to tarball them into ``web_terminal_archives/``
             first, or ``--purge`` to delete them outright.
         * - **Clean up leftovers**
           - ``osprey deploy prune`` removes workspaces of users no longer on
             the roster. ``--dry-run`` shows what it would do first, and the
             same ``--archive`` / ``--purge`` policy applies.
         * - **Tear it all down**
           - ``osprey deploy nuke`` removes the entire multi-user stack —
             containers, volumes, and images — after a typed confirmation.

      ``osprey deploy status`` and ``osprey deploy down`` work exactly as they
      do for any other OSPREY service stack.

   .. tab-item:: Authentication and TLS

      The multi-user stack ships with **no authentication**: anyone who can
      reach the nginx port can open any user's terminal. The landing page's
      user cards are a convenience for choosing an identity on a shared
      machine, not an access-control boundary — and the persona split is a
      *capability* boundary enforced per project, not an identity one.
      Traffic is plain HTTP; there is no TLS. That makes this stack right for
      a **single trusted host** — a workstation or control-room machine you
      already trust — and wrong for an untrusted network.
      :doc:`multi-user-demo` spells out the full posture.

      Both are recognized seams rather than oversights.
      ``modules.web_terminals.auth.method`` and ``modules.web_terminals.tls``
      are part of the config schema, and the nginx config renders against
      them. What's missing is a backend behind the auth seam; wiring one
      (oauth2-proxy, for instance) is the remaining work.

      .. warning::

         Don't set ``auth.method`` expecting a login prompt. Until a real
         backend sits behind the seam, its ``auth_request`` target is a
         placeholder that returns ``403`` — fail-closed by design, so
         enabling it locks *every* user out rather than silently authorizing
         them.

      If authentication is what stands between you and deploying this,
      please `open an issue <https://github.com/als-apg/osprey/issues>`_ or
      get in touch. The seam is already in place, and knowing someone needs
      it is what moves it up the queue.

Try it
======

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Run the Multi-User Demo
      :link: multi-user-demo
      :link-type: doc

      Stand up the two-persona demo — a read-only and a write-capable tier,
      one container each — from a fresh checkout with one build and one
      deploy.

.. toctree::
   :hidden:

   multi-user-demo
