.. _how-to-multi-user:

==================
Multi-User Support
==================

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
   - Standing the two-persona stack up from the ``control-assistant`` preset,
     and watching the write boundary refuse — and approve — a real write
   - Day-to-day operations: adding, reseeding, and removing users

   **Prerequisites:** The concepts need none. To stand the stack up you'll
   want Docker (or Podman) and your model-provider credentials — the
   ``control-assistant`` preset ships the whole block pre-wired.

Single-user is still the front door
===================================

Nothing about multi-user changes the everyday workflow. From any project
directory,

.. code-block:: bash

   osprey web

still launches the single-user Web Terminal as one local process — no
containers, no proxy, ready in seconds at ``http://127.0.0.1:8087``. It remains
the fastest way to try OSPREY and the right tool whenever one person sits in
front of one machine; :doc:`web-terminal/operate` covers it.

The multi-user stack is strictly opt-in. It lives in a
``modules.web_terminals`` block in the project's ``config.yml``, read only by
``osprey deploy`` (and validated by ``osprey scaffold web-terminals lint``).
``osprey web`` never looks at it — so a project that carries the block (the
``control-assistant`` preset ships one) still runs single-user exactly as
before. Reach for the multi-user stack when several people need their own
terminal on a shared machine, and stay with ``osprey web`` for everything else.

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
agent with a UI toggle. The ``control-assistant`` preset ships two: a
*read-only* tier and a *read-write* tier — the same agent and tool surface,
differing on exactly one config key (``control_system.writes_enabled``).
``osprey deploy up`` auto-renders any persona project that doesn't exist yet
and builds its container image locally, so no registry or CI is involved.

**One front door.** An nginx reverse proxy serves the landing page and proxies
``/u/<name>/`` to that user's container. The per-user containers are pinned to
the loopback interface, so nginx is the only network path in. The landing
page's user cards are a convenience for choosing an identity on a trusted
machine — not authentication (see :ref:`multi-user-require-a-login`).

The config block
================

.. tab-set::

   .. tab-item:: Switching it on

      The whole feature is one config block. This is what the
      ``control-assistant`` preset renders into ``config.yml``:

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
                 project: control-assistant-readonly
                 project_path: ../control-assistant-readonly
                 build_profile: control-assistant-readonly
               readwrite:
                 project: control-assistant-readwrite
                 project_path: ../control-assistant-readwrite
                 build_profile: control-assistant-readwrite

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

Run the two-persona stack
=========================

From a fresh checkout, build a project from the bundled preset, then bring the
stack up from inside it:

.. code-block:: bash

   # 1. Render the project from the control-assistant preset
   osprey build control-assistant --preset control-assistant

   # 2. From inside the project, bring the whole stack up
   cd control-assistant
   osprey deploy up

That is the entire setup: the preset ships the ``modules.web_terminals`` block
above, so no extra flags or configuration are needed. Alongside the web tier,
``deploy up`` brings up everything else the control-assistant tutorial deploys
— the virtual accelerator, the scan services, and the supporting
PostgreSQL/OpenObserve containers — so the two personas open onto a working
control room, not an empty shell.

.. note::

   The personas' agent needs your provider credentials at run time. Add them
   to the project's ``.env`` before ``osprey deploy up`` (the preset defaults
   to Anthropic — set ``ANTHROPIC_API_KEY``). If you chose a different
   provider at build time (``--set provider=...``), that choice is recorded in
   the project and forwarded automatically to every auto-rendered persona
   project.

.. note::

   Running from a **source checkout** of the OSPREY repository rather than a
   released install? Add ``--dev`` to ``osprey deploy up``. The images install
   the framework from PyPI by default, and a source tree's version isn't
   published there; ``--dev`` bakes your local checkout into the images
   instead.

What ``osprey deploy up`` does for the web tier
-----------------------------------------------

#. **Auto-renders the two persona projects.** For any persona whose project
   directory does not yet exist, ``deploy up`` renders it from the persona's
   ``build_profile`` — landing it as a sibling of the main project
   (``../control-assistant-readonly`` and ``../control-assistant-readwrite``)
   and forwarding the choices you made at build time (provider, model). An
   already-rendered project is user-owned and never overwritten; a
   half-written one errors with a remediation hint rather than being rebuilt
   over.

#. **Builds each persona's image.** In the preset's local mode
   (``image_source: local``), ``deploy up`` builds each persona's image
   (tagged ``<project>-<persona>:local``) itself from that rendered project —
   no registry, no CI.

#. **Brings up the web tier.** An nginx reverse proxy (container ``ca-nginx``)
   serves the landing page on ``http://127.0.0.1:9080``, and one Web Terminal
   container comes up per user — ``ca-web-alice`` on host port ``9091`` and
   ``ca-web-bob`` on ``9092`` — each reached through the landing page. (The
   ``ca-`` prefix is the preset's ``facility.prefix``; change it for your
   site.)

Stop the stack again with ``osprey deploy down``; check on it with
``osprey deploy status``.

.. note::

   The web stack runs with host networking. On Linux,
   ``http://127.0.0.1:9080`` is reachable as-is. On **macOS**, a container's
   "host" is Docker Desktop's Linux VM — enable *host networking* in Docker
   Desktop (Settings → Resources → Network) so the stack's ports reach your
   browser.

   If another OSPREY deployment already occupies a service port on this host,
   change it in the project's ``config.yml`` (e.g.
   ``services.postgresql.port_host``) before ``osprey deploy up``.

The landing page
----------------

Open ``http://127.0.0.1:9080``. The landing page groups the users into cards,
each labelled with the persona it resolves to:

.. figure:: /_static/resources/multi_user_landing.png
   :alt: The multi-user landing page — two user cards under a Terminals heading,
         each badged with the persona its session resolves to
   :align: center
   :width: 100%

   The grouped landing page: alice resolves to the readonly persona, bob to
   readwrite. Click a card to open that user's session.

alice is a bare roster entry, so she resolves to the preset's
``default_persona`` (readonly). bob names his persona (readwrite) explicitly.
Clicking a card opens that user's terminal at ``/u/<name>/``, proxied by nginx
to the user's own container.

Two sessions, two write postures
--------------------------------

Each persona is a self-contained OSPREY project with its **own** permissions,
because permissions are a property of a project's ``config.yml`` — the two
tiers are genuinely different agents, not one agent with a UI toggle. They
differ on exactly **one** config key, the reference monitor's master write
switch:

.. list-table::
   :header-rows: 1
   :widths: 14 30 56

   * - User
     - ``control_system.writes_enabled``
     - What that means in the session
   * - **alice**
     - ``false``
     - Read-only. Channel reads, the channel finder, the archiver, and logbook
       search all work — but every write surface refuses: channel writes,
       read-write Python execution, all of it, from the single switch.
   * - **bob**
     - ``true``
     - Write-capable — and supervised, not unguarded. A channel write still
       passes the writes-check hook, per-channel min/max limits, and a human
       approval prompt before the connector executes it.

The posture is a property of the **session**, not a statement about the
person: which teammates get a write-capable tier is your roster's call, and
the point is that the framework provisions genuinely different postures from
one deployment.

The boundary is enforced, not asserted — so you can watch it act. Open each
user's terminal and ask both agents to do the same two things:

**Read.** Ask either agent about a channel — a corrector setpoint, a BPM
reading. Both sessions answer identically: reads are ungated on both tiers.

**Write.** Ask each agent to change a setpoint. In bob's session the write
goes to a human approval prompt, then executes. In alice's session the same
request is **refused**: the write tool is denied in her project's rendered
permissions, and the refusal states plainly that writes are disabled in her
configuration.

Both agents carry the *same* tool surface — the readonly tier is not a
stripped-down agent that never heard of writing. It is the same agent whose
write path is switched off in its own project, which is exactly what you want
to demonstrate to a control room: the boundary holds at the enforcement layer,
not at the menu.

Logging out and switching users
-------------------------------

Every session's header carries a chip naming the signed-in user; clicking it
opens a small menu with **Log out**. That POSTs to the terminal's logout
route, clears the local session pointer, and returns you to the landing page.
From there, pick another card to open a different user. Logging out ends the
session for real — the terminal drops its running processes, so the next login
starts **fresh**. Simply navigating away (without logout) keeps the session
warm, and returning to the same user reconnects to it.

.. _multi-user-require-a-login:

Require a login
===============

Today the stack ships with **no authentication and no TLS**: anyone who can
reach the nginx port can open any user's terminal, over plain HTTP. The user
cards are a convenience for choosing an identity on a shared trusted machine,
not an access-control boundary — and the persona split is a *capability*
boundary enforced per project, not an identity one. That posture is right for
a **single trusted host** — a workstation or control-room machine you already
trust — and wrong for an untrusted network.

.. note::

   Per-user login (passwords OSPREY manages for you, or your facility's
   single sign-on) and TLS are being added and will document themselves in
   this section. Until then, do not set
   ``modules.web_terminals.auth.method``: the seam is fail-closed by design,
   so enabling it locks *every* user out rather than silently authorizing
   them.

Related pages
=============

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Web Terminal
      :link: web-terminal/index
      :link-type: doc

      The terminal itself — running it single-user, operating sessions,
      companion panels, and theming.

   .. grid-item-card:: Deploy a Project
      :link: deploy-project
      :link-type: doc

      The ``osprey deploy`` lifecycle the multi-user stack rides on:
      up, status, down, and the service containers.
