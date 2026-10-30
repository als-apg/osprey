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
     tiers and apps, and one nginx front door
   - The ``modules.web_terminals`` config block that switches it on
   - Standing the preset's full stack up — its read-only, read-write and admin
     logins plus the logbook, knowledge and channel-finder apps
   - Day-to-day operations: adding, reseeding, and removing users
   - What each terminal records, and who can read it

   **Prerequisites:** The concepts need none. To stand the stack up you'll
   want Docker (or Podman) and your model-provider credentials — the
   ``control-assistant`` preset ships the whole block pre-wired.

Two subjects have pages of their own:

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

      Token, open, passwords OSPREY manages or your facility's single sign-on,
      HTTPS, and what happens when someone leaves.

Single-user is the front door
=============================

The everyday workflow stands on its own. From any project directory,

.. code-block:: bash

   osprey web

launches the single-user Web Terminal as one local process — no containers, no
proxy, ready in seconds at ``http://127.0.0.1:10100``. It prints a login URL
(``…/?token=…``, once, at startup) that signs you in for 12 hours by default —
the cookie outlives closing the browser — before redirecting to the clean
address. The token is that server's operator secret rather than a one-shot
code, so treat the URL like a password. It is the fastest way to try OSPREY and
the right tool whenever one person sits in front of one machine;
:doc:`../operate` covers it.

The multi-user stack is strictly opt-in. It lives in a
``modules.web_terminals`` block in the deployment's built ``config.yml``, read
only by the lifecycle verbs (and validated by
``osprey scaffold web-terminals lint``).
``osprey web`` reads exactly one key from it — ``auth.session_lifetime``, which
sets how long its own login cookie lasts — and nothing else, so a project that
carries the block (the ``control-assistant`` preset ships one) still runs
single-user like one that does not. Reach for the multi-user stack when several
people need their own terminal on a shared machine, and stay with ``osprey web``
for everything else.

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

**A tier is an access level of the deployment's own agent.** Each user is
assigned a tier, and each tier is its own rendered OSPREY project with its own
``config.yml``, permissions, skills and tool servers, built from one small file
in ``tiers/`` laid over ``profile.yml``. A tier only adds; it never removes
anything from the profile. The ``control-assistant`` preset ships three: a
*read-only* and a *read-write* tier, separated by whether the session may move
the simulator, and an *admin* tier, separated by whether it may change the
deployment it runs in. :doc:`Privilege Tiers <tiers>` lays out what each one
carries and what makes the boundary hold.

**An app is a second agent on the same services.** Each file in ``apps/`` is a
complete agent definition, opened from a shared card by everyone on the roster
and read-only on the machine. The preset lists three under ``shared:``:
``logbook`` searches and synthesises the logbook from the same PostgreSQL the
operators write to; ``knowledge`` answers how the machine is built from the
facility knowledge graph, the channel finder and the knowledge bundle;
``channel-finder`` returns channel addresses from the hierarchy. See
:ref:`multi-user-shared-card`.

``osprey build`` renders one project per file in ``tiers/`` and ``apps/``. In
the preset's local mode, ``osprey up`` builds each one's container image
locally, so no registry or CI is involved. A deployment can pull them from a
registry instead; see :ref:`multi-user-registry-images`.

**One front door.** An nginx reverse proxy serves the landing page and proxies
``/u/<name>/`` to that user's container. The per-user containers are pinned to
the loopback interface, so nginx is the only network path in. The landing page
lists the roster, and its cards are how someone picks an identity; whether
clicking a card *lets them in* depends on whether you have turned login on —
see :doc:`login`.

The config block
================

.. dropdown:: Show the config block
   :color: light

   .. tab-set::

      .. tab-item:: Switching it on

         The whole feature is one config block. This is what a project built from
         the ``control-assistant`` preset carries in its ``config.yml``:

         .. code-block:: yaml

            modules:
              web_terminals:
                enabled: true
                image_source: local       # osprey up builds the images itself
                default_tier: readonly
                landing:
                  groups:
                  - type: users
                    label: Users
                  - type: apps
                    label: Apps
                users:
                - name: alice
                  index: 0
                  tier: readwrite
                  display_name: "Control Room (Alice)"
                - name: bob
                  index: 1
                  tier: readonly
                  display_name: "Read-Only View (Bob)"
                - name: carol
                  index: 2
                  tier: admin
                  display_name: "Deployment Admin (Carol)"
                shared:
                - logbook
                - knowledge
                - channel-finder

         Each ``tier`` names a file in the repository's ``tiers/`` and each
         ``shared:`` entry a file in ``apps/``, both written by ``osprey init``.
         Every render shares one data tree, one set of secrets and one set of
         your own artifacts. The build renders each into
         ``build/<project_name>-<tier|app>`` and records it in
         ``build/config.yml``; nothing here names a path. An unknown tier or
         app is rejected by both ``osprey scaffold web-terminals lint`` and
         ``osprey up``.

         The ``users`` list is the roster — the single source of truth for who
         exists. A name becomes a URL path segment and an environment-variable
         suffix, so it has to match ``[a-z0-9][a-z0-9_-]*``; that is checked in
         every auth mode, not only behind a login wall. A bare name (``- dave``)
         resolves to ``default_tier`` —
         read-only, so a hastily added user lands on the safe side; an entry with
         an explicit ``tier`` picks it, and an optional ``display_name``
         becomes that user's browser tab title. An entry may name a **role**
         instead of a tier — one mapping written once rather than a tier
         pinned per user, and the half a single sign-on provider's groups can
         decide; see :ref:`multi-user-role-from-sso`. Behind a login wall an
         entry may also set ``access:``, the set of principals allowed to open
         it as a shared card: ``any`` for every roster login, or a list naming
         single sign-on identities or a whole email domain — see
         :ref:`multi-user-shared-card`. The roster sets
         no ports.
         Each user's host ports come from the deployment's port layout — one
         hundred-port family per companion panel (artifact gallery, ARIEL,
         channel finder, lattice dashboard, …) plus the terminal itself, and
         each user takes the family's first port plus their index. At the
         default ``deployment.port_base`` that puts alice (index 0) on
         ``10100``, bob (index 1) on ``10101`` and carol (index 2) on
         ``10102``, with their artifact galleries on ``10200``–``10202``. Apps
         take slots counted down from the top of the block, in ``shared:``
         order: logbook on ``10199``, knowledge on ``10198``, channel-finder
         on ``10197``. :ref:`reference-ports-panels` lists every family;
         move the whole deployment with ``deployment.port_base`` rather than a
         family at a time. A facility that must pin one family on its own can
         still set that family's ``modules.web_terminals.<family>_base_port``
         here — see :ref:`reference-ports-keys`.

         One optional key belongs beside these rather than in the roster:
         ``external_origin``, the address browsers actually reach this deployment
         on. Leave it out and that address is derived from ``deploy.fqdn`` and
         ``nginx_port``, which is right whenever a browser talks to this nginx
         directly. Set it when something else stands in front — a facility load
         balancer terminating TLS, a reverse proxy — because the terminals check
         it before allowing any action, and nothing here can guess what that
         front door answers on. nginx answers only on that address. A request
         under any other name, whether another DNS name, the bare IP or
         ``localhost``, is redirected to the same path there, so put the name
         browsers should open in ``deploy.fqdn`` or ``external_origin``. See
         :ref:`multi-user-https`.

         ``landing.groups`` lists the page's sections in order: ``users`` (the
         people), ``apps`` (one card per ``shared:`` app, drawn as a panel and
         titled by the app's ``name:``) and ``links``. Left out, the page shows
         people, then apps. Each group takes a ``label``; the apps panel is
         headed "Apps" by default. Neither changes anything about a container.

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
            * - **Rename or drop a user**
              - Edit the roster entry, and the user's entry in
                ``agent_record.readers`` if it has one, then
                ``osprey up``. A container whose
                user is no longer on the roster is stopped and removed, so it
                cannot keep serving that index's ports; its volumes are
                retained until ``osprey users prune``.
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

That is the entire setup for the preset's roster: the preset ships the
``modules.web_terminals`` block above, so no extra flags or configuration are
needed. Alongside the web tier,
``osprey up`` brings up everything else the control-assistant tutorial deploys
— the virtual accelerator, the bluesky services, and the supporting
PostgreSQL and agent-record containers — so the control-room terminals open onto a
working machine, the logbook app onto a live logbook and the knowledge
app onto a seeded facility graph, not an empty shell.

**Trim the roster** to the users you want, in ``profile.yml`` before
``osprey build``:

- Delete the ``users`` entries you do not want.
- Delete ``tiers/<name>.yml`` for every tier nobody uses.
- If ``readonly`` is gone, point ``default_tier`` at a tier that is left.
- Drop removed users from ``agent_record.readers``, or
  ``osprey up`` still mints a reader key for them.

.. note::

   Every terminal's agent needs your provider credentials at run time. Add them
   to the repository's ``.env`` before ``osprey up`` (the preset defaults
   to Anthropic — set ``ANTHROPIC_API_KEY``). If you chose a different provider
   (``osprey set provider=...``), that choice is recorded in ``profile.yml`` and
   carried into every tier and app the build renders.

.. note::

   Running from a **source checkout** of the OSPREY repository rather than a
   released install? Add ``--dev`` to ``osprey up``. The images install
   the framework from PyPI by default, and a source tree's version isn't
   published there; ``--dev`` bakes your local checkout into the images
   instead.

What ``osprey build`` and ``osprey up`` do for the web tier
-----------------------------------------------------------

#. **The build renders every tier and app.** ``osprey build`` renders one
   project per file in ``tiers/`` and ``apps/``, into the build zone beside the
   main render (``build/control-assistant-readonly``,
   ``build/control-assistant-readwrite``, ``build/control-assistant-admin``,
   ``build/control-assistant-logbook``, ``build/control-assistant-knowledge``
   and ``build/control-assistant-channel-finder``). Every render shares the
   profile's data tree, secrets and artifacts, and inherits the choices
   recorded there (provider, model): edit the profile once and every terminal
   picks the change up. A start renders none of this — ``build/`` is the
   whole account of what will run — so a render missing at start time is
   reported as a stale or partial build, with a rebuild as the remedy.

#. **The start builds each image.** In the preset's local mode
   (``image_source: local``), ``osprey up`` builds one image per render
   (tagged ``<project>:local``, e.g. ``control-assistant-readwrite:local``)
   from that rendered project — no registry, no CI. Registry mode pulls them
   instead; see :ref:`multi-user-registry-images`.

#. **Brings up the web tier.** An nginx reverse proxy (container
   ``control-assistant-nginx``) serves the landing page on
   ``http://127.0.0.1:10000``, and one Web Terminal container comes up per
   user — ``control-assistant-web-alice`` on host port ``10100``,
   ``control-assistant-web-bob`` on ``10101`` and ``control-assistant-web-carol``
   on ``10102``, and one per app — ``control-assistant-web-logbook`` on
   ``10199``, ``control-assistant-web-knowledge`` on ``10198`` and
   ``control-assistant-web-channel-finder`` on ``10197`` — each reached through the
   landing page. With login on, the auth sidecar runs as
   ``control-assistant-auth``, from a locally built
   ``control-assistant-auth:local`` image. Every one of these names is
   ``<project_name>-…``, after the profile's ``project_name:``, which
   ``osprey init`` took from the folder name.

   .. note:: Upgrading a multi-user deployment

      Before this naming, the web containers and the auth image took their
      prefix from ``facility.prefix`` (``ca-nginx``, ``ca-web-alice``,
      ``ca-assistant-auth:local``). The first ``osprey up`` after upgrading
      replaces them with the ``<project_name>-…`` containers and builds the auth
      image under its new tag; no old-named container is left running. The
      volumes keep their names, so every user's workspace carries over.

Stop the stack again with ``osprey down``; check on it with
``osprey status``.

.. note::

   The web stack runs with host networking. On Linux,
   ``http://127.0.0.1:10000`` is reachable as-is. That address is the
   preset's ``deploy.fqdn``, and ``http://localhost:10000`` redirects to it.
   On **macOS**, a container's "host" is the Linux VM. With Docker Desktop,
   enable *host networking* (Settings → Resources → Network) so the stack's
   ports reach your browser; Colima needs no setting. Host networking is one of the limits
   listed in :ref:`perimeter-limits`.

   If another OSPREY deployment already occupies these ports on this host, give
   this one its own block rather than moving services one by one — for example
   ``osprey set config.deployment.port_base=20000 && osprey build`` — before
   ``osprey up``. See :ref:`reference-ports`.

.. _multi-user-registry-images:

Pull the terminal images from a registry
----------------------------------------

With ``image_source: registry`` (the value whenever the key is unset),
``osprey up`` builds no terminal image. It pulls one image per render (each
tier the roster uses and each shared app), under names it derives itself.

.. list-table::
   :header-rows: 1

   * - Who pulls it
     - Image
   * - The ``default_tier``
     - ``<registry.url>/web-terminal:<tag>``
   * - Every other tier, and every shared app
     - ``<registry.url>/web-terminal-<render>:<tag>``
   * - The auth sidecar, when login is on
     - ``<registry.url>/<project_name>-auth:<tag>``

``<render>`` is the tier's or app's file name (``tiers/<name>.yml``,
``apps/<name>.yml``), not its ``project``. The build derives the ``project`` as
``<project_name>-<render>`` from the profile's ``project_name:``.
A tier or app whose image is built elsewhere states its ``project``, the
project the image was built from, in its own file (and may state its
``image``); its terminal runs in ``/app/<project>``.

``<tag>`` is ``modules.web_terminals.image_tag``, ``latest`` when unset.
A ``${VAR}`` in it is expanded from the environment of ``osprey up``, and a
variable that is unset there expands to nothing.
``osprey scaffold web-terminals lint`` warns when the whole tag comes out empty
(code ``web_terminals.empty_image_tag``).

``<registry.url>`` is the top-level ``registry.url`` of ``config.yml``, with no
scheme.
It is the only namespace: two deployments that share a registry keep their
images apart by giving ``registry.url`` different paths.

When the profile has a ``deploy:`` block in registry mode, the build writes
``deploy.registry.url`` into this ``registry.url``, so the registry is named once.
Setting ``registry.url`` in the profile's ``config:`` block
(``osprey set config.registry.url=<path>``) points the terminal images somewhere
else, and that value wins.
A profile with no ``deploy:`` block sets it there.

The scaffolded pipeline builds only the facility's own service images, not
these, so the facility's pipeline must push them.
Put that job in ``ci-extra.yml`` (see :doc:`../../deploy-a-facility`).
Build each image from its rendered project, ``build/<project_name>-<render>``:
the same render ``image_source: local`` builds as
``<project_name>-<render>:local``.

The pipeline pushes the auth sidecar under its name too, unless
``modules.web_terminals.auth.image`` pins a published one; see :doc:`login`.

``osprey scaffold web-terminals lint``, ``osprey profile validate`` and
``osprey build`` refuse a registry-mode config with no ``registry.url`` (code
``web_terminals.registry_mode_missing_url``).
``osprey up`` refuses to render it.
They also refuse a config with web terminals enabled and no tier files (so no
render catalog) or no ``default_tier``, in either image source (code
``web_terminals.requires_catalog``), and a tier or app whose image is built
elsewhere and states no ``project`` (code
``web_terminals.render_missing_project``); ``osprey up`` refuses both before it
mints anything.

For example, with ``tiers/readonly.yml`` and ``tiers/readwrite.yml`` in the repo
and this config:

.. code-block:: yaml

   registry:
     url: registry.example.org/accelerator/demo

   modules:
     web_terminals:
       image_tag: "2026.09.1"
       default_tier: readonly
       users:
       - name: alice
         index: 0
         tier: readwrite
       - name: bob
         index: 1

``osprey up`` pulls:

.. code-block:: text

   alice  registry.example.org/accelerator/demo/web-terminal-readwrite:2026.09.1
   bob    registry.example.org/accelerator/demo/web-terminal:2026.09.1

``bob`` names no tier, so ``bob`` gets the ``default_tier`` and the unsuffixed
image.

The landing page
----------------

Open ``http://127.0.0.1:10000``. The landing page groups the users into cards,
each labelled with the tier it resolves to:

.. figure:: /_static/resources/multi_user_landing_light.png
   :class: only-light
   :figclass: only-light
   :alt: The multi-user landing page — alice's, bob's and carol's cards under
         a Users heading, and the Logbook Research, Facility Knowledge and
         Channel Finder cards in a panel headed Apps beneath them
   :width: 100%

   The grouped landing page: alice resolves to the readwrite tier, bob to
   readonly, carol to admin, and the app cards open the logbook, knowledge and
   channel-finder apps. Click a card to open that session.

.. figure:: /_static/resources/multi_user_landing_dark.png
   :class: only-dark
   :figclass: only-dark
   :alt: The multi-user landing page — alice's, bob's and carol's cards under
         a Users heading, and the Logbook Research, Facility Knowledge and
         Channel Finder cards in a panel headed Apps beneath them
   :width: 100%

   The grouped landing page: alice resolves to the readwrite tier, bob to
   readonly, carol to admin, and the app cards open the logbook, knowledge and
   channel-finder apps. Click a card to open that session.

Each operator card names its tier explicitly — alice the readwrite tier,
bob the readonly one; carol is on the admin tier. (A bare roster entry would fall back to the
preset's ``default_tier``, readonly, so an implicit user always lands on
the safe side.) The app cards sit apart, in the
accent-edged Apps panel, each titled by its app's ``name:`` — and carry no
tier badge, because an app is not a person on a tier. An app card can never
write to the machine or change the deployment, whatever the host sets. Every app
card is shared: the
login page asks for a roster name and that person's password, so nobody needs
a credential for the card itself. Clicking any card opens that session at
``/u/<name>/``, proxied by nginx to its own container.

A deployment that would rather not publish its roster can replace the name
cards with one sign-in button; see :ref:`multi-user-hide-names`.

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
       footer: "Example Research Facility control room. Questions: ext. 5555."

The file's first heading (``# Working safely with the agent``) becomes the
section label, and everything after it becomes the panel. So adding a section
means writing a file and listing it — there is no schema to learn, and nothing
about the text lives in ``config.yml``.

``osprey init`` writes a starter ``data/landing/working-safely.md`` into your
project. It is yours: rewrite it for your facility, or drop it and list your own
files instead. Sections appear in the order you list them, and each one gets an
id from its filename, so you can point someone at
``http://…:10000/#local-procedures`` rather than at the page.

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

The **☰ More** menu at the bottom of the sidebar names the signed-in user —
the display name where the roster sets one, the username otherwise — and
their role, and has **Sign out**. That POSTs to the terminal's logout
route, clears the local session pointer, and returns you to the landing page.
From there, pick another card to open a different user. Logging out ends the
session for real — the terminal drops its running processes, so the next login
starts **fresh**. Simply navigating away (without logout) keeps the session
warm, and returning to the same user reconnects to it.


What the stack records
----------------------

Each terminal writes its own audit trail — refused writes, tool calls, hook
decisions, config edits — as entries in the agent record, through a spool
directory of its own under ``var/spool/<user>/`` in the project on the deploy
host. A container is handed that one directory and nothing else under
``var/spool/``, and the record files every entry under the identity of the
directory it came from, so no user's terminal can rewrite another's trail or
write in another's name. The authentication service records logins through its
own ``sidecar`` spool directory, which no terminal mounts at all.

That makes the agent record the place a question spanning several people is
answered, by a reader whose record login covers those identities
(:doc:`/how-to/health-and-monitoring/read-the-record`). :ref:`The audit trail
<audit-trail-record>` has the record shape, and what append-only does and does
not buy you.

.. _multi-user-history-page:

The history page
================

Where the web terminals sign people on (``auth.method`` of ``password`` or
``oidc``), ``osprey up`` also runs a history page: every session the agent
record holds, each with its conversation, tool calls, approvals and preflight
checks, read in a browser. It is served at ``/history/`` on the **history
page's origin**, the web tier's second server on its own port
(``modules.web_terminals.history_port``, or
``modules.web_terminals.history_origin`` behind a load balancer), so a page an
agent wrote on the terminals' origin cannot read it. A visit without a valid
sign-in goes to the terminals' sign-in and comes back. Without sign-on there is
no history page; each terminal still shows its own past sessions.

Who sees what is decided per person, from the roster, every time a page loads:

.. code-block:: yaml

   modules:
     web_terminals:
       authorization:
         roles:
           operator: {tier: readwrite, history: all}
       users:
         - name: alice
           index: 0
           tier: readwrite             # history: own, the default
         - name: carol
           index: 2
           tier: admin
           history: all

   agent_record:
     readers:
       - name: carol                   # a reader is a reader everywhere

- **Own** (the default): the sessions recorded under the person's own roster
  name.
- **All**: ``history: all`` on the person's roster entry, or on a role in
  ``authorization.roles`` the person holds, shows every user's sessions.
- **A reader** — a roster user named under ``agent_record.readers``, or a
  reader recognised by its ``oidc_subject`` — sees every user's sessions,
  bounded by that reader's ``kinds`` and ``days_back`` (:doc:`/how-to/health-and-monitoring/read-the-record`).
  The reader entry takes precedence over ``history:``.

The rules at the edges:

- A shared card's sessions are recorded under the card's name, so they are
  listed only for ``all`` viewers and readers. Opening a shared card never
  widens the opener's own scope. Signing in *as* a shared card lists nothing
  unless the card's own roster entry (or its role) says ``history: all``.
- A person admitted to a card by an access rule, with no roster entry of their
  own, is refused: the page has no person to scope by.
- A dispatched job is recorded under its worker, ``dispatch-worker-<i>``, so it
  is visible to ``all`` viewers and readers; the dispatch dashboard links each
  run to it.
- A preflight check filed by the queueserver appears on a person's own page
  only when it matches that person's own queue-submit call within 60 seconds;
  agent-authored session plans run inside the queueserver container, so an
  unmatched row is never attributed to anyone in an own view. ``all`` viewers
  and readers see every row as stored.
- A payload the record stored outside itself is shown as a marked reference.
- Where several people are signed in in one browser, the page shows the last
  sign-in and offers a switch among the others, each in their own scope.
- A change to the roster, the roles or the readers applies on the next page
  load after ``osprey up``; a person taken out of ``readers`` is back in their
  own scope then, even with the page still open.

The page names the viewer to the agent record, and the record applies the
scope above itself, from this deployment's config, so the scope is a
permission of the record, not a filter of the page.

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
