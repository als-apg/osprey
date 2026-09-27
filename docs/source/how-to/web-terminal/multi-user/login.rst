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
     - A login page, against your facility's single sign-on, spoken as OIDC
       (:ref:`multi-user-site-sign-in`).

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
           scopes: [openid, profile, email]
       users:
         - name: alice
           index: 0
           oidc_subject: "8f4c1e02-..."     # alice's value of that claim

``scopes`` is what is asked for at the authorization endpoint, and it decides
which claims the provider releases: a claim that was never requested is simply
absent, and an absent claim is a denied login. The three above are the default,
so a deployment whose provider publishes its identity claim under ``profile``
or ``email`` can leave the line out. ``openid`` may not be dropped — without it
the provider issues no ID token at all, and the sidecar refuses to serve rather
than run a login it cannot check.

A scope is a request for a claim, not for *where* it is delivered. OIDC lets a
provider answer a scope by putting the claim in the ID token or by holding it
at its UserInfo endpoint, and a provider that follows the specification
strictly (Connect2id is one) does the latter under the code flow. The sidecar
reads the identity only from the signed ID token and never calls UserInfo —
that endpoint would be a second, weaker trust path — so against such a
provider every login fails with ``the ID token carries no usable 'email'
claim`` although ``email`` was in ``scopes`` and the provider released it.
That symptom has one fix:

.. code-block:: yaml

         oidc:
           issuer: https://sso.example.org/realms/accelerator
           claim: email
           claims_in_id_token: true

With it the login route sends the OIDC ``claims`` request parameter — the
specification's own way to ask for a claim in the ID token — naming the claims
the sidecar actually reads: the identity claim as essential, and
``email_verified`` and any role-binding claim (:ref:`multi-user-role-from-sso`)
as voluntary. Nothing is spelled by hand; the sidecar derives the parameter
from its own configuration. The key is off by default because the parameter is
optional in the specification, many providers ignore it, and one may refuse an
authorization request that carries it. A provider advertises support as
``claims_parameter_supported`` in its discovery document; one that ignores the
parameter still delivers nothing in the token, and the provider's own client
configuration is then where the claim has to be released.

A login matches when the asserted claim equals the card's ``oidc_subject``.
The comparison is exact for every claim except ``email``, which is compared
case-insensitively: an address is the same mailbox in any case, and an
identity provider is free to release the directory's spelling
(``Alice@example.org``) where the roster says ``alice@example.org``. ``sub`` is
an opaque, case-sensitive identifier by specification and stays exact.

Under ``password`` or ``oidc`` a small authentication service joins the stack
and nginx asks it about every request under ``/u/<name>/`` before proxying
anything. Optional keys: ``auth.port`` (the port layout's ``10001`` unless you
set it — see :ref:`reference-ports`),
``auth.session_lifetime`` in whole seconds (default ``43200``), and
``auth.image``, required with ``image_source: registry`` — a published build of
OSPREY's login service; :ref:`multi-user-login-service-contract` says what it
answers.

.. dropdown:: Where ``auth.session_lifetime`` applies
   :icon: gear

   The key reaches past this page: it sets how long a terminal session cookie
   lasts wherever that cookie is used — ``osprey web`` and ``auth.method:
   token``. For everyone who goes through the login page, nginx rather than
   that cookie is what lets them through, so here the key sets the login
   page's cookie.

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

.. _multi-user-site-sign-in:

When your site's sign-in is not OIDC
====================================

The login service serves two methods, ``password`` and ``oidc``, and no
others. A site whose single sign-on is SAML, Kerberos, CAS, or a header set by
a proxy after its own login reaches OSPREY through an **OIDC broker**: an
identity provider that signs people in against the site's system and issues
OIDC ID tokens to OSPREY. Keycloak and Dex are two widely used ones. Many site
identity providers also publish an OIDC endpoint beside their SAML one, so ask
the site's identity team before running a second service.

Point ``auth.oidc.issuer`` at the broker and set ``claim`` to the attribute it
releases for each person. The rest of this page applies unchanged:
``oidc_subject``, ``scopes``, role binding, shared cards.

OSPREY takes no identity from a request header. The four ``X-Osprey-Auth-*``
headers the terminals read are written by nginx from the login service's
answer and cleared on every other route
(:ref:`multi-user-login-service-contract`), so a proxy in front of OSPREY
cannot say who a user is, and whatever it sends is overwritten. A login that trusted such a header would
let anyone who reaches nginx without passing that proxy name themselves. A
broker keeps the proof in a signed token that the login service checks itself.

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

- Every value of the claim is matched, in any order. No mapped role → refused
  (``unmapped_role_claim``).
- On a card whose entry names a ``role:``, the login carries that role, and it
  must be one of the roles the token maps to; otherwise it is refused
  (``role_mismatch``). A person in several mapped groups opens the card built
  for any one of them, and the login record lists the mapped roles in
  ``detail`` (``mapped_roles=…``). Fix whichever of roster or provider has
  drifted.
- On a card whose entry names a ``persona:`` instead, the token's role is
  carried, so it must map to exactly one role; several → refused
  (``ambiguous_role_claim``).
- A role is resolved at login and travels inside the session — together with
  its origin, the roster entry or the provider's claim — so a change at the
  provider or in the roster reaches the *next* login. Short of the next login,
  the one way to end a role now is to end the person's access:
  ``osprey users remove <name>``.

Every login and refusal is recorded in ``var/audit/sidecar/auth_sidecar.jsonl``
on the deploy host. A ``claims`` stanza under ``password`` resolves nothing;
``osprey up`` warns rather than fails.

.. note::

   Microsoft Entra ID leaves ``groups`` out of the token for accounts in many
   groups (*group overage*), which lands on a missing-claim refusal. Either
   emit only the groups assigned to this application, or define app roles and
   point ``claim`` at ``roles``.

Behind a proxy that re-signs TLS with a site certificate authority, point
``images.site_ca`` at the bundle on the deploy host and rebuild: the CA is
installed into the login service's image, which is where its trust store lives.
The site-CA block in ``.env.shared`` does not do this — a CA path in the chain
names a file that image does not carry. See :ref:`deployment-env-chain` for
what the chain delivers to which container.

This applies to the image OSPREY builds. In registry mode
(``modules.web_terminals.auth.image``) the login service runs a published image
the facility built itself, so its trust store is that build's business.

.. _multi-user-login-service-contract:

What the login service answers
==============================

In both image modes the stack starts the login service with a fixed command,
``uvicorn osprey.services.auth_sidecar.app:create_app --factory``, bound to the
loopback address on ``auth.port``, and probes it with the image's own
``python``. An ``auth.image`` is therefore a build of OSPREY's login service
from the same release as the terminals, published by the facility's CI. It is
not a place for a login service of the site's own. What follows is what nginx
and the terminals rely on, and what a build has to keep answering across an
upgrade.

``GET /verify?user=<card>`` is asked once for every request under
``/u/<card>/``, from an internal location whose card name is fixed when the
stack is rendered, so nothing in the request picks the card. It is answered
over loopback within 2 s to connect and 5 s to read. An empty 200 admits the
request and a bare 401 refuses it, with no body and no redirect; any other
status fails the request. A refused browser page load is sent to
``/auth/login?user=<card>``, and a program gets the bare 401. ``/verify`` is not
under ``/auth/`` and cannot be reached from outside.

A 200 carries up to four headers:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Header
     - Present
     - What it names
   * - ``X-Osprey-Auth-Account``
     - On every 200.
     - The roster card the request is on.
   * - ``X-Osprey-Auth-Subject``
     - When the session holds who proved the login.
     - The provider's asserted identity, or under ``password`` the roster name
       of whoever typed the password. It equals the account on an own card and
       differs on a shared one.
   * - ``X-Osprey-Auth-Role``
     - Only when the session holds a role.
     - The role. Absent means no privileges, never a default.
   * - ``X-Osprey-Auth-Role-Source``
     - Only beside a role.
     - ``roster`` or ``claim``, for display only.

Values are printable ASCII with no leading or trailing space. A value that
cannot travel refuses the request rather than leaving its header off.

``/auth/`` is the public login surface: the login page ``/auth/login``,
``/auth/oidc/login``, the provider callback ``/auth/oidc/callback`` and
``/auth/logout``. nginx proxies the whole prefix without a gate, because it is
where a session comes from.

``/health`` answers 200 with ``status``, ``service``, ``method`` and
``configured``, even when the service is misconfigured. In that state it
reports ``configured: false`` and every other path answers 503. The container
healthcheck polls it on the service's own port, and nginx never proxies it.

nginx writes all four header names in every location that proxies. A gated
``/u/<card>/`` sets them from the ``/verify`` answer; the ungated branch, the
internal ``/verify`` target and ``/auth/`` clear them. A client's own copy of
these headers therefore never reaches a terminal or the login service, and the
answer to ``/verify`` is the only way an identity reaches a terminal. How the
terminals record them: :ref:`audit-trail-identity-keys`.

.. _multi-user-shared-card:

Share a card with more than one person
======================================

A roster entry admits one person. ``access:`` is the key that changes that.
Its value is a **principal set** — the principals allowed to open the card,
and nobody else. The two words you already know name one-principal sets:
``own``, the default, is ``[self]``, the entry's own login and nobody else's;
``any`` is ``[roster]``, everyone this deployment can authenticate. Every
other set is written as a list.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Principal
     - Who it admits
   * - ``self``
     - The entry's own login. ``access: own`` is this set, spelled shorter.
   * - ``roster``
     - Everyone this deployment can authenticate. ``access: any`` is this
       set, spelled shorter.
   * - ``user:<id>``
     - One identity, written exactly as your provider asserts it under
       ``auth.oidc.claim``. Needs ``method: oidc``.
   * - ``domain:<domain>``
     - Everyone whose asserted email address is in that domain. Needs
       ``method: oidc`` and ``claim: email``.

A card whose set is anything but ``[self]`` is a **shared card**: one
terminal, one persona, one audit directory, opened by more than one person.
Here is one shared with the whole roster:

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
           client_id_env: OSPREY_AUTH_OIDC_CLIENT_ID
           client_secret_env: OSPREY_AUTH_OIDC_CLIENT_SECRET
           claim: sub
       users:
         - name: alice
           index: 0
           persona: readwrite
           oidc_subject: "8f4c1e02-..."
         - name: bob
           index: 1
           persona: readonly
           oidc_subject: "41ab97d0-..."
         - name: ops-desk
           index: 2
           persona: readonly
           access: any                # any roster login opens this card

And here are two cards that nobody needs a roster entry to open — one shared
with a whole single sign-on domain, one with two named people:

.. code-block:: yaml

   modules:
     web_terminals:
       auth:
         method: oidc
         oidc:
           issuer: https://sso.example.org/realms/accelerator
           client_id_env: OSPREY_AUTH_OIDC_CLIENT_ID
           client_secret_env: OSPREY_AUTH_OIDC_CLIENT_SECRET
           claim: email             # domain: reads the email address
       users:
         - name: ops-desk
           index: 2
           persona: readonly
           access: [domain:example.org]
         - name: controls
           index: 3
           persona: readwrite
           access: [user:ada@example.org, user:bo@example.org]

The ``control-assistant`` preset ships its two standalone cards this way —
``logbook`` and ``knowledge``, each ``access: any`` under ``password`` — so
the logbook research and facility knowledge terminals are opened with any
roster login's own password and carry no credential of their own.

Who can open it is answered one principal at a time. ``roster`` admits
anyone this deployment can authenticate: under ``password`` that is every roster entry
with a provisioned password; under ``oidc``, every entry that carries an
``oidc_subject:``, the shared card itself included when it carries one.
``user:`` admits the one identity it names, and ``domain:`` admits every
asserted email address in that domain — in both cases whether or not that
person has a roster entry of their own. The set only ever grants: a card
admits someone when at least one of its principals covers them, nothing on
the card takes admission back again, and a person no principal covers is
refused.

Write ``self`` into the list where the card's own login has to keep working
beside the rest. ``[self, domain:example.org]`` admits the card's owner by
their own credential and everyone in the domain; ``[domain:example.org]``
alone does not admit the owner, unless the identity they sign in with is
itself in that domain. ``roster`` covers the owner too, as one roster member
among the others.

``domain:`` compares the domain exactly. ``domain:example.org`` admits
``ada@example.org`` and refuses ``ada@labs.example.org``; name the subdomain
as a principal of its own where you want it too. Case in the domain is
ignored, because domain names ignore it, and there is a domain to read only
where the identity claim is the email address — so ``domain:`` is refused
before the deployment starts unless ``claim: email`` is set. Two further
checks run once, at login, on claims the login service never stores: a token
whose ``hd`` claim disagrees with the domain of its email address is refused,
and so is one carrying ``email_verified: false``. A provider that never
verifies addresses — a Keycloak realm with *Verify email* switched off, say —
emits that claim for everyone, so its ``user:`` principals refuse everybody
until the address is verified, which is the provider behaving to spec rather
than a fault.

Under ``password`` only ``self`` and ``roster`` are evaluated: nothing there
asserts an identity or an email domain, so a card listing only ``user:`` or
``domain:`` principals would admit nobody at all. The lint refuses that card
before it can be deployed. A card whose set names ``roster`` gains a username
field on its login form:
the person types their *own* roster name beside the password, and it is that
name's stored password that is checked — and that name the rate limit counts
against. A card that never had a password of its own has no credential to
offer here; one flipped from ``own`` still does, until its hash is retired —
see :ref:`Removing someone <multi-user-shared-card-removal>` below.

A session opened by someone the roster names carries who opened it — the
*opener* — and re-checks that person against the roster on every request.
Rotating the opener's password or removing them with ``osprey users remove``
ends every shared session they opened at the next request, and under ``oidc``
so does editing or removing their ``oidc_subject:`` — that per-person
revocation is how a shared card is taken away from one user without touching
the rest.

A session admitted by a ``user:`` or ``domain:`` principal has no roster
entry behind it. It carries the identity the provider asserted, and every
request re-checks that identity against the set the card carries *now*.
Revocation is the same move made in a different place: drop the principal
that covers someone, and their next request on that card is refused. Editing
the ``access`` key settles the same way throughout — a card widened from
``own`` refuses the sessions minted while it was its owner's, a card returned
to ``own`` refuses the shared ones, and a narrowed list refuses whoever it no
longer covers.

The ledger records who opened a session beside the card
(:ref:`audit-trail-identity-keys`), so a shared terminal's records still say
who did what.

The card's role rides the card. The claims binding
(:ref:`multi-user-role-from-sso`) is not consulted on a shared card — the
card's terminal is only ever built as its own tier — so a person the binding
would refuse for their own card can still open a shared one. If the binding
is your membership gate, do not share a card.

These rosters are refused by ``osprey scaffold web-terminals lint`` and the
build verbs that run it:

- **A deployment-editing card cannot be shared**
  (``web_terminals.shared_card_privileged``). A persona holding the setup
  tool or the Config panel was lifted for named people behind their own
  cards; any set wider than ``[self]`` would hand it to people it was never
  lifted for.
- **A principal OSPREY does not recognise is an error, never a wider card**
  (``web_terminals.invalid_user_access``). An empty list, a value that is not
  a list, an unknown prefix and a misspelled one are all refused where they
  are written, rather than quietly leaving the card owner-only. ``group:``
  gets its own wording: the prefix is reserved and not supported yet.
- **Both new principal kinds need** ``method: oidc``
  (``web_terminals.access_principal_without_idp``). Under ``password``
  nothing asserts an identity or an email domain, so a card written that way
  could not mean what it says.
- **A** ``domain:`` **principal needs** ``claim: email``
  (``web_terminals.access_domain_without_email_claim``). Every other claim is
  refused, because there is no email address to read a domain from and the
  card would admit nobody.
- **Two** ``user:`` **principals on one card must not name the same
  identity** (``web_terminals.duplicate_access_principal``). Identical values
  are refused, and so are values differing only in case under
  ``claim: email``, because the login service has to resolve an identity to a
  single grant.
- **With a shared card on an** ``oidc`` **roster, two entries must not carry
  the same** ``oidc_subject`` (``web_terminals.shared_card_duplicate_subject``).
  One person could always hold two cards — the same ``oidc_subject:`` on
  both, every login arriving through the card that was clicked. A shared card
  changes the question: the login service must resolve each identity to a
  single roster entry to know who opened it, so once any card is shared, such
  a person keeps ``oidc_subject:`` on one card only — a login by that
  identity would otherwise be ambiguous, and is refused. Under
  ``claim: email`` two subjects that differ only in case count as the same.

The three principal refusals — ``web_terminals.access_principal_without_idp``,
``web_terminals.access_domain_without_email_claim`` and
``web_terminals.duplicate_access_principal`` — say nothing where no login wall
stands. Under ``token`` or ``none`` there is no door for a principal to open,
and a profile's passive base is allowed to carry the key for the variant that
arms it. Once the deployment is running, a card whose
``OSPREY_AUTH_ROSTER_ACCESS_*`` value the login service cannot read admits
nobody, its owner included, and the service logs a warning naming the
variable until the deployment is rendered again from a corrected profile.

.. _multi-user-roster-contract:

Who the roster names
====================

The roster is the list of people and shared cards, written in the profile.
Adding a person, removing one or changing their tier is a roster edit followed
by ``osprey up`` (the day-to-day table on :ref:`how-to-multi-user`).
``osprey users remove`` is the removal that also retires the credential.
Nothing reads the roster from a directory at run time.

Under ``oidc`` the provider proves who someone is. The roster decides whether
that person has a terminal, and which tier. A person the provider knows but
whom no roster entry or principal covers is refused.

A provider's groups can already pick the tier of a person the roster names
(:ref:`multi-user-role-from-sso`). They add nobody to the roster and admit
nobody to a card. ``user:`` and ``domain:`` principals are how people without
an entry of their own reach a shared card.

``group:`` is reserved for a principal naming a provider group. It will be
built with the first site whose sign-in releases a groups claim to OSPREY.
Groups are defined in the site's identity provider, not in OSPREY, and some
providers release none. Until then the lint refuses it
(``web_terminals.invalid_user_access``).


.. _multi-user-https:

Serve it over HTTPS
===================

A login page hands out session cookies, so ``password`` and ``oidc`` refuse to
render with ``tls.enabled: false`` unless something else encrypts the
connection. Two shapes:

**This nginx terminates TLS.** Set ``tls.enabled: true`` with a certificate
and key; nginx serves HTTPS on 443 — or on ``tls.port`` when you set one — and
redirects the plain port to the deployment's origin. ``host_cert_dir`` is the only key that names a
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

The deployment answers on that one origin, and every other name is redirected
there. A front proxy must forward the browser's own ``Host`` (for nginx in
front: ``proxy_set_header Host $host;``). A proxy that sends its upstream's name
gets every request redirected back to itself. A tunnel opened on ``localhost``
is redirected to the origin too; it could not act before either, because the
terminals accept actions only from the origin.
``allow_insecure_http`` is not a way to postpone certificates on a reachable
host; with nothing terminating TLS, anyone watching the traffic can become
that user.
Lint does not warn about plain HTTP in this shape because the browser's
origin is ``https``; the hop from the terminator to this nginx is still plain
HTTP, so keep it on a network you trust.

A single-user ``osprey web`` behind the same kind of TLS terminator sets
``OSPREY_TERMINAL_EXTERNAL_ORIGIN`` to that address instead. It is the same
origin every terminal checks a state-changing request against, and the login
URL ``osprey web`` prints is built from it — without it the printed URL names
the bind address, and a browser arriving from it has every write refused.

Passwords, and where they live
==============================

Password hashes and cookie-signing keys live in ``.env.auth`` in the project
root — mode ``0600``, gitignored, mounted into the authentication service
only. On every ``osprey up``, for each user in order:

#. An existing hash in ``.env.auth`` is kept; deploying never resets a
   password.
   A hash the authentication service cannot read, such as a truncated paste
   or another tool's format, is kept as well, and ``osprey up``,
   ``osprey scaffold web-terminals lint`` and the service's startup log each
   name the user. ``osprey users passwd <user>`` replaces it.
#. Otherwise a plaintext ``OSPREY_AUTH_PW_<USER>`` in ``.env`` is hashed in —
   the way to set a password you chose. ``<USER>`` is the name uppercased with
   ``-`` turned into ``_``.
#. Otherwise a password is generated, hashed, and printed once. Capture it.

A password ``profile.yml`` publishes under ``env.defaults`` (the preset's
demo logins are one example) is refused by ``osprey up`` once browsers reach
the deployment anywhere but this machine: ``external_origin``, or
``deploy.fqdn`` when that is unset, names a host other than ``127.0.0.1`` or
``localhost``. Run ``osprey users passwd <user>`` for each login it
names; for a shared card, delete its ``OSPREY_AUTH_PW_HASH_<CARD>`` line from
``.env.auth`` and run ``osprey up``. HTTPS does not lift the refusal.

To change one later, ``osprey users passwd alice`` prompts, rewrites that hash
and ends alice's sessions — her own card's, and every
:ref:`shared-card <multi-user-shared-card>` session she opened, since those
are held open by this same credential. Sessions held open by other people's
passwords stay up. Password login is rate-limited per user but never locks
anyone out — a control-room operator must not be shut out of the terminals.

.. _multi-user-shared-card-removal:

Removing someone, and turning it off
====================================

A credential can outlive an account, so:

- **An OSPREY login is its own session.** Under ``oidc`` the provider's proof
  is checked once, at sign-in. The session that follows lasts until logout or
  ``auth.session_lifetime``, whatever happens at the provider afterwards.
  Signing out at the provider does not end it, and neither does disabling the
  account there. ``osprey users remove <name>`` ends it now, and for a shared
  card so does editing that person's ``oidc_subject:``. The reverse also
  holds: logging out of OSPREY does not sign the browser out of the provider,
  so on a shared machine sign out there too.
- **Use** ``osprey users remove alice``, not a hand-edit of the roster —
  removing the entry alone leaves her hash in ``.env.auth``, and adding the
  name back months later revives her password. ``remove`` (or ``prune``, for
  names already edited out) retires the credential and ends the person's
  login-page session.
- **A shared card is revoked per person, through their own credential.**
  ``osprey users passwd alice`` ends every shared-card session alice opened
  along with her own (see above); under ``oidc``, editing or removing her
  ``oidc_subject:`` ends her shared-card sessions at the next request — her
  *own* card's session is different, lapsing at expiry or logout as it always
  has, unless ``osprey users remove alice`` ends it now.

- **Sharing a card does not retire the card's own password** — the hash stays
  in ``.env.auth`` and still works: anyone who knows it can open the shared
  card by typing the card's own name into the username field. When you share
  a card that used to have its own password, delete its
  ``OSPREY_AUTH_PW_HASH_<CARD>`` line from ``.env.auth`` and run
  ``osprey up``; returning the card to ``own`` later mints it a fresh
  password. A hash left in place is revived by that return. ``<CARD>`` is the
  name uppercased with ``-`` turned into ``_``.
  ``osprey users passwd <card>`` is refused while the card is shared —
  there is no password of its own to change.
- **A plaintext** ``OSPREY_AUTH_PW_ALICE`` **in** ``.env`` **survives**
  ``osprey users remove`` and would be hashed straight back in for the next
  alice.
  Delete the line by hand when the person leaves.
- **Logging out ends a terminal session** on the server: the cookie it was
  carrying is refused from that moment on. The login page's cookie is refused
  too, and stays refused when the authentication service restarts or is
  recreated (``osprey up``, ``osprey users passwd``): the service keeps each
  logged-out session as a one-way digest in ``var/audit/sidecar/``, until that
  session would have expired anyway. If that directory cannot be written, the
  service logs a warning and a logout lasts until the service restarts.
- **Terminal sessions are kept on disk** — behind ``auth.method: token`` and
  ``osprey web``, not behind the login page here — so there they outlive a
  restart of the web terminals and a change of the operator secret. A
  password change or a removal ends the login-page session, not those.
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
