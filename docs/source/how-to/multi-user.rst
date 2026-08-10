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
   - How to require a login — passwords OSPREY manages, or your facility's
     single sign-on

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
the loopback interface, so nginx is the only network path in. The landing page
lists the roster, and its cards are how someone picks an identity; whether
clicking a card *lets them in* depends on whether you have turned login on —
see :ref:`multi-user-require-a-login`.

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
                 build_profile: personas/readonly.yml
               readwrite:
                 project: control-assistant-readwrite
                 project_path: ../control-assistant-readwrite
                 build_profile: personas/readwrite.yml

      Each ``build_profile`` names that persona's **delta** inside the build
      profile this project was rendered from — the file
      ``osprey profile new`` writes under ``personas/`` and points the catalog
      at. The delta merges over that profile, so every persona shares one data
      tree, one set of secrets, and one set of your own artifacts. A bundled
      preset name, an absolute path, or a path outside ``personas/`` is
      rejected by both ``osprey scaffold web-terminals lint`` and
      ``osprey deploy up``.

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

      Edit it in the **build profile** — the ``modules.web_terminals.users``
      entry under ``config:`` in ``profile.yml`` — and rebuild the project with
      ``--force``. A roster change made directly in the project's ``config.yml``
      deploys, but the next build overwrites it. Rebuilding also seeds an empty
      ``web-terminal-context/<user>/`` slot in the profile for each new
      operator, which is where their per-user context goes.

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
   directory does not yet exist, ``deploy up`` renders it from that persona's
   **delta** in the build profile beside your project — landing it as a sibling
   of the main project (``../control-assistant-readonly`` and
   ``../control-assistant-readwrite``). Because each delta merges over that
   profile's ``profile.yml``, both personas share its data tree, secrets and
   artifacts, and inherit the choices recorded there (provider, model): edit
   the profile once and both terminals pick the change up from the file rather
   than from a replayed command line. An already-rendered project is
   user-owned and never overwritten; a half-written one errors with a
   remediation hint rather than being rebuilt over.

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

Out of the box the stack asks for no credentials and speaks plain HTTP:
clicking a card on the landing page opens that terminal, and anyone who can
reach the nginx port can click any card. That posture suits a **single trusted
host** — a workstation or control-room machine you already trust — and nothing
beyond it. It is the walkthrough's choice, not a limit of the stack: it keeps
one ``osprey deploy up`` on a laptop free of certificates and identity
providers. No preset ships with login enabled, so this is something you turn on
deliberately.

Set ``auth.method`` and every request under ``/u/<name>/`` — pages, APIs and
the terminal's live connection alike — is refused unless the browser holds a
valid session for *that* user. The check happens at the front door: a small
authentication service joins the stack in its own container, and nginx asks it
about each request before proxying anything. Nothing depends on the per-user
containers policing themselves.

Note that the persona split is a *capability* boundary, enforced per project —
it decides what a session may do, never who may open it. Those are separate
questions, and login answers only the second.

Choose a method
---------------

**Passwords**, managed by OSPREY. Nothing extra to run or operate:

.. code-block:: yaml

   modules:
     web_terminals:
       tls:
         enabled: true
         host_cert_dir: /etc/ssl/facility     # host side; mounted for you
         cert: /etc/osprey/tls/facility.crt   # container side
         key: /etc/osprey/tls/facility.key
       auth:
         method: password

**OIDC**, against the single sign-on your facility already runs. Each roster
entry names the identity that maps to it, so a valid login as somebody else
cannot open this user's terminal:

.. code-block:: yaml

   modules:
     web_terminals:
       tls:
         enabled: true
         host_cert_dir: /etc/ssl/facility     # host side; mounted for you
         cert: /etc/osprey/tls/facility.crt   # container side
         key: /etc/osprey/tls/facility.key
       auth:
         method: oidc
         oidc:
           issuer: https://sso.example.org/realms/accelerator
           client_id_env: OSPREY_AUTH_OIDC_CLIENT_ID
           client_secret_env: OSPREY_AUTH_OIDC_CLIENT_SECRET
           claim: sub                       # ID-token claim to match on
       users:
         - name: alice
           index: 0
           oidc_subject: "8f4c1e02-..."     # alice's value of that claim
         - name: bob
           index: 1
           oidc_subject: "b7d9a340-..."

The ``*_env`` keys hold environment-variable **names**, not credentials: put the
client id and secret in the project's ``.env.auth`` under those names — that is
the only file the authentication service reads credentials from. The names
shown are the ones OSPREY reads when you omit the keys, and ``claim`` falls back
to ``sub`` in the authentication service itself. ``oidc_subject`` is not a
secret — it is the identifier your provider already publishes for that person.

.. warning::

   **No secret may contain a dollar sign** — not in ``.env.auth``, and not in
   the ``.env`` and ``.env.production`` that carry your provider API key and
   facility passwords. Depending on which container stack reads these files,
   ``$`` sequences inside the *values* are substituted on the way through —
   with Docker Compose, ``secret$abc`` arrives as ``secret`` and ``P@$$w0rd``
   arrives as ``P@$w0rd``; other stacks mangle a different set. Either way the
   file on disk still reads correctly, so the only symptom is a login or a
   token exchange that refuses for no visible reason.

   This bites hardest with a client secret your identity provider generated for
   you, since you did not choose those characters. If yours contains a ``$``,
   issue a new one rather than trying to escape it — escaping is not portable
   between container runtimes, so there is no spelling that works everywhere.

   ``osprey deploy`` refuses to start a stack whose secrets would be corrupted
   this way and names the offending variables, so you find out before the
   deployment is running rather than after someone cannot log in.

   The same rule extends to each user's ``oidc_subject``, which travels a
   different route (the rendered compose file rather than an env file) but is
   rewritten the same way: lint refuses a subject containing ``$`` and names
   the user. If your provider genuinely issues one, map a different claim via
   ``auth.oidc.claim``.

Three more keys are optional. ``auth.port`` is the port the authentication
service listens on (default ``9070``); ``auth.session_lifetime`` is how long a
session stays valid, in **whole seconds** (default ``43200``, twelve hours); and
``auth.image`` names the service's image, which is **required** when
``image_source: registry`` — your CI publishes that image the same way it
publishes the terminal images. In ``image_source: local`` mode
``osprey deploy up`` builds it for you and ``auth.image`` is not needed.

.. warning::

   ``auth.port`` and ``auth.session_lifetime`` must be plain positive integers.
   A duration string like ``"12h"``, a decimal, zero or a negative number is
   **silently replaced by the default** — nothing warns you — so a deployment
   that meant eight-hour sessions would quietly keep twelve-hour ones.

The service listens on ``127.0.0.1`` on the deploy host itself (the web stack
uses host networking), so nginx reaches it and nothing off-host does. It is not
published as a container port, and anyone with a shell on the deploy host can
reach it — the same as every per-user terminal.

Serve it over HTTPS
-------------------

A session cookie sent over plain HTTP is readable by anything on the path, so a
deployment with ``auth.method`` other than ``none`` and ``tls.enabled: false``
**refuses to render at all** rather than hand out cookies in the clear. You
therefore have to pick one of two ways to get the connection encrypted.

**Let this nginx terminate TLS.** Set ``tls.enabled: true`` with a certificate
and key, and nginx serves HTTPS on 443, redirects the plain port to it, and
marks session cookies so browsers only ever send them over HTTPS. Bringing the
certificate is still your job, but getting it *into* the container is not:

.. code-block:: yaml

   tls:
     enabled: true
     host_cert_dir: /etc/ssl/facility          # on the deploy host
     cert: /etc/osprey/tls/facility.crt        # inside the container
     key: /etc/osprey/tls/facility.key

``host_cert_dir`` is the only key here that names a path on the **deploy host**;
``cert`` and ``key`` are paths **inside the nginx container**. Setting
``host_cert_dir`` bind-mounts that directory, read-only, at the directory
``cert`` sits in — so the certificate is where nginx looks without you writing
any compose of your own. Renewals need nothing extra: the mount is a directory,
so a replaced file is picked up on the next nginx reload.

Because one mount has to deliver both files, ``cert`` and ``key`` must sit in
the same directory, and ``host_cert_dir`` must be absolute. A deployment that
breaks either rule is refused at render time, naming the reason — rather than
starting an nginx that immediately dies looking for a file nobody mounted.

.. note::

   ``host_cert_dir`` is optional. Leave it out and nothing is mounted: the
   compose overlay renders exactly as it does without TLS, and supplying the
   certificate is yours to arrange — a bind mount from a small compose file of
   your own, listed after the web overlay in ``runtime.compose_files``, or
   whatever your facility's certificate management already does. That is the
   route to take when a plain directory bind cannot express how certificates
   reach this host.

**Or terminate TLS in front of this nginx.** If a facility load balancer or
ingress proxy already presents the certificate and forwards to this host, set
``auth.allow_insecure_http: true`` and leave ``tls.enabled`` off. This is a
normal deployment, not a workaround: the browser's connection is encrypted by
the thing in front, and the hop it forwards over is yours to keep private.

What ``allow_insecure_http`` is *not* is a way to postpone certificates on a
reachable host. With it set and nothing terminating TLS, anyone who can watch
the traffic can copy a session cookie and become that user. An isolated network
where you accept that risk is the only other case for it.

Passwords, and where they live
------------------------------

In password mode ``osprey deploy up`` makes sure every user on the roster has a
password hash before it starts anything, and aborts before a single container
starts if it cannot — an unwritable file is caught here rather than becoming a
stack nobody can log in to. The same check covers the keys used to sign session
cookies, so an OIDC deployment can abort the same way even though it provisions
no passwords at all. The usual cause either way is permissions on ``.env.auth``
or on the project directory.

The hashes and signing keys live in ``.env.auth`` in the project root — mode
``0600``, listed in the generated ``.gitignore`` next to ``.env.production``,
and handed to the authentication service alone. No terminal container ever sees
it.

For each user, in order:

#. An existing hash in ``.env.auth`` is kept. Deploying again never resets
   anyone's password.
#. Otherwise, a plaintext ``OSPREY_AUTH_PW_<USER>`` in the project's ``.env`` is
   hashed into ``.env.auth`` — the way to set a password you already chose. The
   plaintext stays on the deploy host; only the hash reaches a container.
#. Otherwise a password is generated, hashed, and **printed once**, on that
   deploy's output. Nothing can recover it afterwards, so capture it and hand it
   to the person.

``<USER>`` is the username uppercased with ``-`` turned into ``_``. Because that
mapping is what keeps one user's password out of another user's terminal, an
authenticated deployment refuses to render when two roster names collide under
it (``alice-b`` and ``alice_b``), or when a name falls outside
``[a-z0-9][a-z0-9_-]*``.

To change a password later:

.. code-block:: bash

   osprey deploy passwd alice

It prompts (never echoing), rewrites that one hash, and restarts the
authentication service. Alice's existing sessions stop working immediately, and
nobody else's are touched.

Sessions, logging out, and rolling back
---------------------------------------

The landing page stays public — it lists the roster so people can find their own
card, and a card is a prompt, not a door. One browser may hold several unlocked
users at once, which is what a shared control-room machine needs; logging out
ends that one user's session and leaves the others alone.

.. note::

   The list of logged-out sessions is held in the authentication service's
   memory, so restarting that container forgets it. A cookie captured before a
   logout could be replayed until it expires on its own — within
   ``auth.session_lifetime``.

Removing someone needs care, because a credential can outlive the person's
account in three different ways:

**Use** ``osprey deploy decommission alice`` **, not a hand-edit.** Deleting a
roster entry and running ``osprey deploy up`` removes alice's container, and the
authentication service stops answering for a name that is no longer on the
roster — but her hash stays in ``.env.auth``. Add the name back months later and
her old password works again. ``decommission`` (or ``prune``, for names already
edited out) is what actually retires the credential.

**A plaintext password in** ``.env`` **survives decommission.** If you seeded
alice's password by putting ``OSPREY_AUTH_PW_ALICE`` in the project's ``.env``,
decommissioning her clears the hash but leaves that line — and the next
``osprey deploy up`` for a new alice hashes it straight back in, handing the new
person the departed one's password. The decommission warns you, but the warning
scrolls past in a deploy log weeks before anyone reuses the name. **Delete the**
``.env`` **line by hand when the person leaves.**

**In OIDC mode,** ``decommission`` **is the verb that ends a session.**
``prune`` cleans up users already off the roster, but it only restarts the
authentication service when it actually removed a password entry — and an OIDC
user has none. Their container is gone either way, so the stale route just
fails; but if what you need is that person's *session* closed now, run
``decommission``.

To turn login back off, set ``auth.method: none`` and run ``osprey deploy up``.
That re-renders nginx and the compose file, drops the authentication service,
and returns the stack to the open posture described at the top of this section.
``.env.auth`` is left in place, so turning login on again keeps everyone's
existing password.

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
