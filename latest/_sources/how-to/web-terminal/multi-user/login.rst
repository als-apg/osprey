.. _multi-user-require-a-login:

===============
Require a Login
===============

The landing page lists the roster; ``modules.web_terminals.auth.method``
decides what stands between a card and the terminal behind it.

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - ``method``
     - What a person meets
   * - ``token``
     - **The default.** No login page; each terminal is opened once from that
       user's own login URL.
   * - ``none``
     - **Open.** A card opens its terminal. Only for a room where everyone who
       can reach the page is trusted.
   * - ``password``
     - A login page, against passwords OSPREY manages.
   * - ``oidc``
     - A login page, against the single sign-on your facility already runs.

.. raw:: html
   :file: ../../../_diagrams/auth-postures.html

Two things to read off the drawing. The front door is nginx, and
``auth.method`` decides what it does there — nothing, vouch for the caller, or
ask the authentication service. The terminal behind it always checks a
credential of its own; the postures differ only in who supplies it.

Choose a method
===============

**Token** needs no stanza. ``osprey up`` mints an operator secret for every
roster user into the deployment's ``.env``; a terminal refuses you until you
have opened that user's login URL once:

.. code-block:: bash

   osprey users login-url alice

Send each person only their own URL, as you would a password. To rotate one,
delete that user's ``OSPREY_TERMINAL_SECRET_*`` line from ``.env`` and run
``osprey up`` again. Token mode cannot tell one person from another — whoever
holds a URL is that user — so it suits a single trusted host and nothing
beyond it.

.. _multi-user-open-mode:

**Open** — no credential anywhere, for a console behind a locked door:

.. code-block:: yaml

   modules:
     web_terminals:
       auth:
         method: none

nginx stamps each user's operator secret onto every request it proxies, so a
card opens its terminal and no login URL exists. That is only safe if nothing
inside the deployment can reach nginx back:

.. raw:: html
   :file: ../../../_diagrams/open-mode-egress.html

``osprey up`` refuses to start an open deployment unless every persona's
``.claude/settings.json`` denies ``Bash``, ``WebFetch``, ``WebSearch`` and
``mcp__plugin_playwright_playwright__*`` (``osprey scaffold web-terminals
lint`` reports the same, as ``web_terminals.open_mode_egress``). All four are
in OSPREY's deny defaults, so a refusal means a persona lifted one — put it
back in that persona's ``config:`` block and rebuild. The python executor's
own guard against executed code reaching the web ports is defence in depth,
not a boundary: ``none`` is for rooms where the agents are trusted too.

**Passwords**, managed by OSPREY:

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

**OIDC**, against your facility's single sign-on. Each roster entry names the
identity that maps to it, so a valid login as somebody else cannot open this
user's terminal:

.. code-block:: yaml

   modules:
     web_terminals:
       tls:
         enabled: true
         host_cert_dir: /etc/ssl/facility
         cert: /etc/osprey/tls/facility.crt
         key: /etc/osprey/tls/facility.key
       auth:
         method: oidc
         oidc:
           issuer: https://sso.example.org/realms/accelerator
           client_id_env: OSPREY_AUTH_OIDC_CLIENT_ID      # names in .env.auth
           client_secret_env: OSPREY_AUTH_OIDC_CLIENT_SECRET
           claim: sub
       users:
         - name: alice
           index: 0
           oidc_subject: "8f4c1e02-..."     # alice's value of that claim

Under ``password`` or ``oidc`` a small authentication service joins the stack
and nginx asks it about every request under ``/u/<name>/`` before proxying
anything. Optional keys: ``auth.port`` (the port layout's ``10001`` unless you
set it — see :ref:`reference-ports`),
``auth.session_lifetime`` in whole seconds (default ``43200``), and
``auth.image``, required with ``image_source: registry``.

.. dropdown:: Where ``auth.session_lifetime`` applies
   :icon: gear

   The key reaches past this page: it sets how long a terminal session cookie
   lasts wherever that cookie is used — ``osprey web``, ``auth.method:
   token``, and ``login: false`` roster entries. For everyone who does go
   through the login page, nginx rather than that cookie is what lets them
   through, so here the key sets the login page's cookie.

``tls.port`` is optional in the same way: nginx serves HTTPS on 443 unless you
name another port, for a host that cannot bind 443 or already carries another
deployment's HTTPS. It is HTTPS's own default rather than a port-layout slot,
so :ref:`reference-ports` does not list it, and a non-default value changes the
address browsers reach — see :ref:`multi-user-https`. A ``tls.port`` or
``auth.port`` that is not a whole number between 1 and 65535 falls back to that
key's default, which ``osprey scaffold web-terminals lint`` reports as
``web_terminals.invalid_listener_port``.

.. warning::

   No secret may contain a ``$`` — not in ``.env.auth``, ``.env`` or
   ``.env.users``, and not in an ``oidc_subject``. Container stacks substitute
   ``$`` sequences on the way through, and the only symptom is a login that
   refuses for no visible reason. ``osprey up`` refuses such a stack and names
   the variable; if a provider issued the secret, issue a new one.

.. _multi-user-role-from-sso:

Let single sign-on pick the tier
================================

Instead of pinning a persona on every roster entry, name roles once and let
the provider's groups choose:

.. code-block:: yaml

   modules:
     web_terminals:
       authorization:
         roles:
           operator: {persona: readwrite}
           viewer: {persona: readonly}
         claims:
           claim: groups          # the ID-token claim holding group membership
           map:
             ca-operators: operator
             ca-viewers: viewer
       users:
         - name: alice
           index: 0
           role: operator         # in place of `persona: readwrite`
           oidc_subject: "8f4c1e02-..."

A roster entry carries ``role:`` or ``persona:``, never both. The rules:

- Every value of the claim is matched, in any order. Exactly one distinct role
  must result: none → refused (``unmapped_role_claim``), more than one →
  refused (``ambiguous_role_claim``).
- The role the token grants must be the role the roster named for the card
  that was clicked; otherwise the login is refused (``role_mismatch``). Fix
  whichever of roster or provider has drifted.
- A role is resolved at login and travels inside the session — together with
  its origin, the roster entry or the provider's claim — so a change at the
  provider or in the roster reaches the *next* login. To withdraw a role now,
  end the session: ``osprey users decommission <name>``.

Every login and refusal is recorded in ``var/audit/sidecar/auth_sidecar.jsonl``
on the deploy host. A ``claims`` stanza under ``password`` resolves nothing;
``osprey up`` warns rather than fails.

.. note::

   Microsoft Entra ID leaves ``groups`` out of the token for accounts in many
   groups (*group overage*), which lands on a missing-claim refusal. Either
   emit only the groups assigned to this application, or define app roles and
   point ``claim`` at ``roles``.

Leave one entry public
======================

A roster entry that fronts a read-only service — the preset's ARIEL logbook —
can opt out of the login wall:

.. code-block:: yaml

   users:
     - name: ariel
       index: 2
       persona: ariel
       login: false

Only the literal ``false`` opts out; anything else means "login required". The
entry is still gated the way every terminal is under ``token``, by its own
login URL. A ``login: false`` entry whose persona can edit the deployment
(``setup_patch`` or the Config panel) is refused at build and at ``osprey up``.

.. _multi-user-https:

Serve it over HTTPS
===================

A login page hands out session cookies, so ``password`` and ``oidc`` refuse to
render with ``tls.enabled: false`` unless something else encrypts the
connection. Two shapes:

**This nginx terminates TLS.** Set ``tls.enabled: true`` with a certificate
and key; nginx serves HTTPS on 443 — or on ``tls.port`` when you set one — and
redirects the plain port to it. ``host_cert_dir`` is the only key that names a
path on the deploy host — it is bind-mounted, read-only, where ``cert`` and
``key`` (paths inside the container) sit, so both must be in that one directory
and the path must be absolute. Leave ``host_cert_dir`` out to mount the
certificate your own way.

A non-default ``tls.port`` also becomes part of the address browsers reach.
The deployment's origin is then ``https://<fqdn>:<port>``, built from
``deploy.fqdn`` unless ``external_origin`` names the address itself, and
everything derived from it carries the port: the landing
link, the address each terminal checks a state-changing request came from, and
under ``oidc`` the callback the authentication service sends to your provider.
Register ``https://<fqdn>:<port>/auth/oidc/callback`` with the identity
provider, or change an existing registration to match — a provider refuses a
callback that is not character-for-character the registered one. On 443 the
port stays out of the origin and the callback is
``https://<fqdn>/auth/oidc/callback``.

**Something in front terminates TLS** — a facility load balancer or ingress:

.. code-block:: yaml

   modules:
     web_terminals:
       external_origin: https://terminals.example.org   # what the browser reaches
       auth:
         method: password
         allow_insecure_http: true

``external_origin`` is required here: every terminal refuses a state-changing
request unless the browser says it came from that address, and nothing else
in the configuration can work out what the thing in front answers on. Write
it as a bare origin — scheme, host, port if non-default, no path.
``allow_insecure_http`` is not a way to postpone certificates on a reachable
host; with nothing terminating TLS, anyone watching the traffic can become
that user.

Passwords, and where they live
==============================

Password hashes and cookie-signing keys live in ``.env.auth`` in the project
root — mode ``0600``, gitignored, mounted into the authentication service
only. On every ``osprey up``, for each user in order:

#. An existing hash in ``.env.auth`` is kept; deploying never resets a
   password.
#. Otherwise a plaintext ``OSPREY_AUTH_PW_<USER>`` in ``.env`` is hashed in —
   the way to set a password you chose. ``<USER>`` is the name uppercased with
   ``-`` turned into ``_``.
#. Otherwise a password is generated, hashed, and printed once. Capture it.

To change one later, ``osprey users passwd alice`` prompts, rewrites that hash
and ends alice's sessions; nobody else is touched. Password login is
rate-limited per user but never locks anyone out — a control-room operator
must not be shut out of the terminals.

Removing someone, and turning it off
====================================

A credential can outlive an account, so:

- **Use** ``osprey users remove alice``, not a hand-edit of the roster —
  removing the entry alone leaves her hash in ``.env.auth``, and adding the
  name back months later revives her password. ``decommission`` (or
  ``prune``, for names already edited out) retires the credential and, under
  OIDC, ends the session.
- **A plaintext** ``OSPREY_AUTH_PW_ALICE`` **in** ``.env`` **survives
  decommission** and would be hashed straight back in for the next alice.
  Delete the line by hand when the person leaves.
- **Logging out ends a terminal session** on the server: the cookie it was
  carrying is refused from that moment on. The login page's cookie is the
  other case — that logout is remembered in the authentication service's
  memory only, so a copy captured beforehand can be replayed until it
  expires, within ``auth.session_lifetime``.
- **Terminal sessions are kept on disk** — the ones behind ``login: false``
  entries here, and behind ``auth.method: token`` and ``osprey web``
  elsewhere — so they outlive a restart of the web terminals and a change of
  the operator secret. A password change or a decommission ends the
  login-page session, not this one.
- **A shortened** ``auth.session_lifetime`` **reaches sessions already
  running** at the next restart of the web terminals, when their deadlines
  are clamped to the new value.
- **A terminal already on screen outlives its deadline** until that page is
  closed, reloaded, or logged out from — the deadline is checked when a page
  connects, not on a timer, so a logout elsewhere does not cut a terminal that
  is already open.

To turn the login page off, set ``auth.method: token`` and run ``osprey up``;
``.env.auth`` is kept, so turning it back on keeps everyone's password.
``auth.method: none`` goes one step further and drops the login URLs too —
read :ref:`the open posture <multi-user-open-mode>` first.
