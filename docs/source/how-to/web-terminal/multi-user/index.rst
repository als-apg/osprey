.. _how-to-multi-user:

==================
Multi-User Support
==================

The multi-user Web Terminal turns one OSPREY project into a small shared
product: a landing page where each member of your team picks their name, and a
private, containerized Web Terminal behind each card — all served from a single
host, brought up with a single ``osprey up``.

.. dropdown:: What You'll Learn
   :color: primary
   :icon: book

   - How it sits beside the single-user ``osprey web`` workflow, and when each
     mode is the right tool
   - The three ideas behind the multi-user stack: one container per user,
     personas as capability tiers, and one nginx front door
   - The ``modules.web_terminals`` config block that switches it on
   - Standing the preset's full stack up — its read-only, read-write and admin
     logins plus a standalone ARIEL terminal
   - Day-to-day operations: adding, reseeding, and removing users

   Two subjects have pages of their own: :doc:`what each tier may do, and what
   holds it <tiers>`, and :doc:`how to require a login <login>` — passwords
   OSPREY manages, or your facility's single sign-on.

   **Prerequisites:** The concepts need none. To stand the stack up you'll
   want Docker (or Podman) and your model-provider credentials — the
   ``control-assistant`` preset ships the whole block pre-wired.

Single-user is the front door
=============================

The everyday workflow stands on its own. From any project directory,

.. code-block:: bash

   osprey web

launches the single-user Web Terminal as one local process — no containers, no
proxy, ready in seconds at ``http://127.0.0.1:8087``. It prints a login URL
(``…/?token=…``, once, at startup) that sets a session cookie before
redirecting to the clean address. The token is that server's operator secret
rather than a one-shot code, so treat the URL like a password. It is the
fastest way to try OSPREY and the right tool whenever one person sits in front
of one machine; :doc:`../operate` covers it.

The multi-user stack is strictly opt-in. It lives in a
``modules.web_terminals`` block in the deployment's built ``config.yml``, read
only by the lifecycle verbs (and validated by
``osprey scaffold web-terminals lint``).
``osprey web`` never looks at it — so a project that carries the block (the
``control-assistant`` preset ships one) runs single-user exactly like one that
does not. Reach for the multi-user stack when several people need their own
terminal on a shared machine, and stay with ``osprey web`` for everything else.

How it works
============

.. raw:: html
   :file: ../../../_diagrams/multi-user.html

Three ideas carry the whole design:

**One container per user.** Every name on the roster gets its own Web Terminal
container, plus two named volumes that belong to the *user*, not the container:
a workspace volume (the files the agent reads and writes) and an
agent-configuration volume. Upgrading or rebuilding an image replaces the
container but never touches those volumes, so a user's files and settings
survive every redeploy. On first start, ``osprey up`` seeds each user's
configuration volume automatically — no per-user setup steps.

**A persona is a capability tier — and a whole project.** Users map to
*personas*, and each persona is its own rendered OSPREY project with its own
``config.yml``, permissions, skills, and tool servers. Because permissions are
a property of a project, the tiers are genuinely different agents — not one
agent with a UI toggle. The ``control-assistant`` preset ships three of them:
a *read-only* and a *read-write* tier, separated by whether the session may
move hardware, and an *admin* tier, separated by whether it may change the
deployment it runs in. :doc:`Privilege Tiers <tiers>` lays out what each one
carries and what makes the boundary hold.

It also ships a fourth persona that is not a tier of that agent at all.
``ariel`` is the standalone logbook research assistant: no control-system tool
servers, no Python sandbox, no plan queue — a different product, reached from
its own card. Nothing special makes that possible. A persona is already a
whole project, so it can differ in what it *is* as easily as in what it may
write. It shares this deployment's PostgreSQL and logbook, so the operators
and the research terminal read one logbook together.

``osprey build`` renders one persona project per delta in ``personas/``, and
``osprey up`` builds each one's container image locally, so no registry or CI is
involved.

**One front door.** An nginx reverse proxy serves the landing page and proxies
``/u/<name>/`` to that user's container. The per-user containers are pinned to
the loopback interface, so nginx is the only network path in. The landing page
lists the roster, and its cards are how someone picks an identity; whether
clicking a card *lets them in* depends on whether you have turned login on —
see :doc:`login`.

The config block
================

.. tab-set::

   .. tab-item:: Switching it on

      The whole feature is one config block. This is what a project built from
      the ``control-assistant`` preset carries in its ``config.yml``:

      .. code-block:: yaml

         modules:
           web_terminals:
             enabled: true
             image_source: local       # osprey up builds persona images itself
             nginx_port: 9080          # the landing page
             web_base_port: 9091       # per-user ports: base + user index
             artifact_base_port: 9291
             ariel_base_port: 9391
             lattice_base_port: 9491
             channel_finder_base_port: 9591
             default_persona: readonly
             landing:
               groups:
               - type: users
                 label: Users
             users:
             - name: alice
               index: 0
               persona: readwrite
               display_name: "Control Room (Alice)"
             - name: bob
               index: 1
               persona: readonly
               display_name: "Read-Only View (Bob)"
             - name: ariel
               index: 2
               persona: ariel
               display_name: "ARIEL Logbook Research"
             - name: carol
               index: 3
               persona: admin
               display_name: "Deployment Admin (Carol)"
             personas:
               readonly:
                 project: control-assistant-readonly
                 project_path: build/control-assistant-readonly
                 build_profile: personas/readonly.yml
               readwrite:
                 project: control-assistant-readwrite
                 project_path: build/control-assistant-readwrite
                 build_profile: personas/readwrite.yml
               admin:
                 project: control-assistant-admin
                 project_path: build/control-assistant-admin
                 build_profile: personas/admin.yml
               ariel:
                 project: control-assistant-ariel
                 project_path: build/control-assistant-ariel
                 build_profile: personas/ariel.yml
                 landing_group: Standalone deployments

      Each ``build_profile`` names that persona's **delta** in the deployment
      repository — the file ``osprey init`` writes under ``personas/`` and
      points the catalog at. The delta merges over ``profile.yml``, so every
      persona shares one data
      tree, one set of secrets, and one set of your own artifacts. A bundled
      preset name, an absolute path, or a path outside ``personas/`` is
      rejected by both ``osprey scaffold web-terminals lint`` and
      ``osprey up``.

      The ``users`` list is the roster — the single source of truth for who
      exists. A name becomes a URL path segment and an environment-variable
      suffix, so it has to match ``[a-z0-9][a-z0-9_-]*``; that is checked in
      every auth mode, not only behind a login wall. A bare name (``- dave``)
      resolves to ``default_persona`` —
      read-only, so a hastily added user lands on the safe side; an entry with
      an explicit ``persona`` picks its tier, and an optional ``display_name``
      becomes that user's browser tab title. Each user's host ports are
      ``base + index`` in every port family — one family per companion panel
      (artifact gallery, ARIEL, channel finder, lattice dashboard, …) plus the
      terminal itself — so alice (index 0) serves her terminal on ``9091``,
      bob (index 1) on ``9092``, ariel (index 2) on ``9093`` and carol
      (index 3) on ``9094``. A panel
      whose ``*_base_port`` you don't set falls back to its built-in default,
      so the block above lists them only to make the layout visible.

      One optional key belongs beside these rather than in the roster:
      ``external_origin``, the address browsers actually reach this deployment
      on. Leave it out and that address is derived from ``deploy.fqdn`` and
      ``nginx_port``, which is right whenever a browser talks to this nginx
      directly. Set it when something else stands in front — a facility load
      balancer terminating TLS, a reverse proxy, a DNS alias — because the
      terminals check it before allowing any action, and nothing here can guess
      what that front door answers on. See :ref:`multi-user-https`.

      A persona may also name a landing-page section with ``landing_group``.
      Its users are lifted out of the roster's default section into one of
      that name, drawn as a panel — which is how the page shows a standalone
      service as something other than another login. The ``users`` group takes
      a ``label`` for the same reason: so both halves can be named. Neither
      changes anything about a container.

      .. tip::

         Give every roster entry an explicit ``index`` before you ever
         *remove* one. Once indices are pinned, deleting an earlier user can
         no longer shift a later user's ports out from under a running
         deployment.

   .. tab-item:: Day-to-day operations

      The roster drives everything: edit it, then let the lifecycle verbs
      reconcile reality against it.

      Edit it in the **source profile** — the ``modules.web_terminals.users``
      entry under ``config:`` in ``profile.yml`` — and rebuild. A roster change
      made directly in the built ``build/config.yml`` deploys, but the next
      build overwrites it. Rebuilding also seeds an empty
      ``web-terminal-context/<user>/`` slot for each new operator, which is
      where their per-user context goes.

      Beside those slots sits ``web-terminal-context/base.md`` — the shared
      baseline every seeded user's ``CLAUDE.md`` starts from. ``osprey init``
      materializes it from the preset so the text is visible and editable in
      your repo; edit it there and rebuild, and every terminal picks up the
      change. A profile without one falls back to a generic framework
      baseline.

      .. list-table::
         :header-rows: 1
         :widths: 34 66

         * - Task
           - Command
         * - **Add a user**
           - Add a roster entry with the next free ``index``, then
             ``osprey up``. The new container comes up with freshly
             allocated ports and a seeded workspace; existing users are
             untouched.
         * - **Reseed workspaces**
           - ``osprey users seed [USER]`` re-applies the seeded configuration
             for one user, or for everyone when ``USER`` is omitted.
         * - **Remove one user**
           - ``osprey users remove USER`` stops and removes the user's
             container. Their volumes are **retained** by default; add
             ``--archive`` to tarball them into ``web_terminal_archives/``
             first, or ``--purge`` to delete them outright.
         * - **Clean up leftovers**
           - ``osprey users prune`` removes workspaces of users no longer on
             the roster. ``--dry-run`` shows what it would do first, and the
             same ``--archive`` / ``--purge`` policy applies.
         * - **Tear it all down**
           - ``osprey reset`` wipes the whole deployment back to a fresh state —
             containers, volumes, and images, every user's workspace included —
             after a typed confirmation. It prints the removal plan first;
             ``--dry-run`` stops there.

      ``osprey status`` and ``osprey down`` work exactly as they
      do for any other OSPREY service stack.

Run the multi-user stack
========================

From a fresh checkout, create a deployment repository from the bundled preset,
then build and bring the stack up from inside it:

.. code-block:: bash

   # 1. Create the deployment repo from the control-assistant preset
   osprey init control-assistant --preset control-assistant

   # 2. From inside the repo, render it and bring the whole stack up
   cd control-assistant
   osprey build
   osprey up

That is the entire setup: the preset ships the ``modules.web_terminals`` block
above, so no extra flags or configuration are needed. Alongside the web tier,
``osprey up`` brings up everything else the control-assistant tutorial deploys
— the virtual accelerator, the bluesky services, and the supporting
PostgreSQL/OpenObserve containers — so the control-room terminals open onto a
working machine, and the ARIEL terminal onto a live logbook, not an empty
shell.

.. note::

   The personas' agent needs your provider credentials at run time. Add them
   to the repository's ``.env`` before ``osprey up`` (the preset defaults
   to Anthropic — set ``ANTHROPIC_API_KEY``). If you chose a different provider
   (``osprey set provider=...``), that choice is recorded in ``profile.yml`` and
   carried into every persona project the build renders.

.. note::

   Running from a **source checkout** of the OSPREY repository rather than a
   released install? Add ``--dev`` to ``osprey up``. The images install
   the framework from PyPI by default, and a source tree's version isn't
   published there; ``--dev`` bakes your local checkout into the images
   instead.

What ``osprey build`` and ``osprey up`` do for the web tier
-----------------------------------------------------------

#. **The build renders the persona projects.** ``osprey build`` renders one
   project per **delta** in ``personas/``, into the build zone beside the main
   render (``build/control-assistant-readonly``,
   ``build/control-assistant-readwrite``, ``build/control-assistant-admin`` and
   ``build/control-assistant-ariel``).
   Because each delta merges over
   ``profile.yml``, every persona shares its data tree, secrets and artifacts,
   and inherits the choices recorded there (provider, model): edit the profile
   once and every terminal picks the change up from the file rather than from a
   replayed command line. A start renders none of this — ``build/`` is the
   whole account of what will run — so a persona project missing at start time
   is reported as a stale or partial build, with a rebuild as the remedy.

#. **The start builds each persona's image.** In the preset's local mode
   (``image_source: local``), ``osprey up`` builds each persona's image
   (tagged ``<project>-<persona>:local``) from that rendered project —
   no registry, no CI.

#. **Brings up the web tier.** An nginx reverse proxy (container ``ca-nginx``)
   serves the landing page on ``http://127.0.0.1:9080``, and one Web Terminal
   container comes up per user — ``ca-web-alice`` on host port ``9091``,
   ``ca-web-bob`` on ``9092``, ``ca-web-ariel`` on ``9093`` and
   ``ca-web-carol`` on ``9094`` — each reached
   through the landing page. (The
   ``ca-`` prefix is the preset's ``facility.prefix``; change it for your
   site.)

Stop the stack again with ``osprey down``; check on it with
``osprey status``.

.. note::

   The web stack runs with host networking. On Linux,
   ``http://127.0.0.1:9080`` is reachable as-is. On **macOS**, a container's
   "host" is Docker Desktop's Linux VM — enable *host networking* in Docker
   Desktop (Settings → Resources → Network) so the stack's ports reach your
   browser.

   If another OSPREY deployment already occupies a service port on this host,
   change it in the profile and rebuild — for example
   ``osprey set config.services.postgresql.port_host=5433 && osprey build`` —
   before ``osprey up``.

The landing page
----------------

Open ``http://127.0.0.1:9080``. The landing page groups the users into cards,
each labelled with the persona it resolves to:

.. figure:: /_static/resources/multi_user_landing.png
   :alt: The multi-user landing page — alice's and bob's cards under a Users
         heading, and the ARIEL terminal in a Standalone deployments panel
         beneath them
   :align: center
   :width: 100%

   The grouped landing page: alice resolves to the readwrite persona, bob to
   readonly, and the ariel card opens the standalone logbook terminal. Click
   a card to open that session.

Each operator card names its persona explicitly — alice the readwrite tier,
bob the readonly one; the preset's roster adds carol on the admin tier, last
so the operator cards keep their ports. (A bare roster entry would fall back to the
preset's ``default_persona``, readonly, so an implicit user always lands on
the safe side.) The ariel card sits apart, in the accent-edged panel its
persona's ``landing_group`` names — and carries no persona badge, because the
badge answers *which tier is this user on?* and here the card and its persona
are the same word. Clicking any card opens that session at ``/u/<name>/``,
proxied by nginx to its own container.

What your operators read first
------------------------------

At the bottom of the landing page sit collapsible **notices** — the things your
facility wants people to read before they open a terminal. Each notice is one
markdown file, listed in config:

.. code-block:: yaml

   modules.web_terminals:
     landing:
       notices:
       - data/landing/working-safely.md
       - data/landing/local-procedures.md
       footer: "ALS control room. Questions: ext. 5555."

The file's first heading (``# Working safely with the agent``) becomes the
section label, and everything after it becomes the panel. So adding a section
means writing a file and listing it — there is no schema to learn, and nothing
about the text lives in ``config.yml``.

``osprey init`` writes a starter ``data/landing/working-safely.md`` into your
project. It is yours: rewrite it for your facility, or drop it and list your own
files instead. Sections appear in the order you list them, and each one gets an
id from its filename, so you can point someone at
``http://…:9080/#local-procedures`` rather than at the page.

Two edge cases are worth knowing:

* **Leave ``notices`` out entirely** and you get OSPREY's built-in safety
  notice. A config that says nothing still ships something.
* **Set ``notices: []``** for no notices at all. That is the explicit way to
  turn them off.

A file you list that does not exist is skipped and reported by ``osprey build``
as a warning. It is *not* replaced by the built-in notice — showing OSPREY's
safety text where your own procedures should have been would be worse than
showing a gap.

.. note::

   Notice files are rendered to HTML at build time, so they are trusted input
   at the same level as ``config.yml`` itself — anyone who can edit a notice can
   already edit your deployment's configuration. The ``footer`` is a plain
   string and is always escaped.

Logging out and switching users
-------------------------------

Every session's header carries a chip in the top-left naming the terminal — the
display name where the roster sets one, the username otherwise. Clicking it
opens a small menu naming the signed-in user, with **Log out**. That POSTs to the terminal's logout
route, clears the local session pointer, and returns you to the landing page.
From there, pick another card to open a different user. Logging out ends the
session for real — the terminal drops its running processes, so the next login
starts **fresh**. Simply navigating away (without logout) keeps the session
warm, and returning to the same user reconnects to it.


Going further
=============

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Privilege Tiers
      :link: tiers
      :link-type: doc
      :shadow: md

      What each login may do — read-only, read-write, admin — where a tier
      comes from, and what stops a session doing more.

   .. grid-item-card:: Require a Login
      :link: login
      :link-type: doc
      :shadow: md

      Passwords OSPREY manages or your facility's single sign-on, HTTPS, and
      what happens when someone leaves.

Related pages
=============

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Web Terminal
      :link: ../index
      :link-type: doc

      The terminal itself — running it single-user, operating sessions,
      companion panels, and theming.

   .. grid-item-card:: Deploy a Project
      :link: ../../deploy-project/index
      :link-type: doc

      The lifecycle the multi-user stack rides on: build, up, status, down,
      and the service containers.

.. toctree::
   :hidden:

   tiers
   login
