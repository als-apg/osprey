.. _multi-user-require-a-login:

===============
Require a Login
===============

The landing page lists the roster; a login decides who may open a card. This
page covers the two ways to require one — passwords OSPREY manages, or the
single sign-on your facility already runs — the HTTPS that either needs, where
passwords live, and what to do when someone leaves.


With no ``auth`` stanza nginx asks for no credentials and speaks plain HTTP —
but the terminals behind it are not open. Each one authenticates every request
against its own operator secret, which ``osprey up`` mints into the
deployment's ``.env`` for every roster user whether or not authentication is
on. Clicking a card on the landing page therefore reaches a terminal that
refuses you until you have opened that user's login URL once:

.. code-block:: bash

   osprey users login-url alice

The URL carries alice's secret and trades it for a session cookie. Send each
person only their own, the way you would send a password; it stays valid until
you rotate it, which means deleting that user's ``OSPREY_TERMINAL_SECRET_*``
line from ``.env`` and running ``osprey up`` again. (``osprey up`` names the
verb in its summary but never prints the URLs.)

What this posture does *not* do is tell one person from another at the front
door: whoever holds a URL is that user. It suits a **single trusted host** —
a workstation or control-room machine you already trust — and nothing beyond
it.

The ``control-assistant`` preset ships with password login switched on, in its
demo posture: each roster user's password is seeded into the repository's
``.env`` by ``osprey init`` (``alice``/``alice``, ``bob``/``bob``,
``carol``/``carol`` — change them there, or rotate with
``osprey users passwd``), the ARIEL entry stays public
via ``login: false`` (see below), and ``allow_insecure_http: true`` keeps the
demo on plain HTTP. Those passwords authenticate a demo, not a facility: for
any reachable host, set real passwords and serve TLS as described here.

Set ``auth.method`` and every request under ``/u/<name>/`` — pages, APIs and
the terminal's live connection alike — is refused unless the browser holds a
valid session for *that* user. The check happens at the front door: a small
authentication service joins the stack in its own container, and nginx asks it
about each request before proxying anything. Nothing depends on the per-user
containers policing themselves.

Note that the persona split is a *capability* boundary, enforced per project —
it decides what a session may do, never who may open it. Login answers the
separate question of who may open a session. In this multi-user stack that login
is the per-user auth described above; note that even single-user ``osprey web``
gates every request on a session cookie, handed out by the login URL it prints
at startup, so "no login" is never the single-user default either.

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
   the ``.env`` and ``.env.users`` that carry your provider API key and
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

   ``osprey up`` refuses to start a stack whose secrets would be corrupted
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
``osprey up`` builds it for you and ``auth.image`` is not needed.

.. warning::

   ``auth.port`` and ``auth.session_lifetime`` must be plain positive integers.
   A duration string like ``"12h"``, a decimal, zero or a negative number is
   **silently replaced by the default** — nothing warns you — so a deployment
   that meant eight-hour sessions would quietly keep twelve-hour ones.

The service listens on ``127.0.0.1`` on the deploy host itself (the web stack
uses host networking), so nginx reaches it and nothing off-host does. It is not
published as a container port, and anyone with a shell on the deploy host can
reach it — the same as every per-user terminal.

Leave one entry public
----------------------

Not every card on the landing page is a person's terminal. A roster entry that
fronts a read-only service — the preset's ARIEL logbook assistant, say — can
opt out of the login wall:

.. code-block:: yaml

   users:
     - name: ariel
       index: 2
       persona: ariel
       login: false

With authentication on, that entry sits outside the login wall: nginx never
asks the authentication service about it, and no password is provisioned for it
(``osprey users passwd`` refuses the name and says why). Outside the wall is not
the same as open — the entry is gated exactly as the whole deployment is with
authentication off, by its own operator secret, so a browser still has to open
``osprey users login-url ariel`` once. Cookies from the login wall never reach
its container.

Only the literal ``false`` opts an entry out. Absence, ``true``, and any typo
all mean "login required" — a misspelling can lock an entry down, never open it
up — and lint reports a non-boolean value. The key is inert while
``auth.method`` is ``none``, which lint points out as well.

Opting out is for entries whose *content* is public by design. Anything that
can reach a control system, write anywhere, or spend provider tokens belongs
behind the wall.

For the capability it would be worst to leave open, that is a check rather
than advice: a ``login: false`` entry resolving to a persona holding either
deployment-editing surface — the agent's ``setup_patch`` tool or the web
Config panel — fails ``osprey profile validate`` and ``osprey build`` with the
user named, and ``osprey up`` refuses to start a stack whose render still
carries one. It holds whether or not your profile floors those surfaces for
its other tiers; see :doc:`tiers` for what the
check reads and what it tells you to do about it. The
preset's own admin card sits behind the wall for exactly that reason, and is
last in the roster so the operator cards keep their ports.

.. _multi-user-https:

Serve it over HTTPS
-------------------

A session cookie sent over plain HTTP is readable by anything on the path, so a
deployment with ``auth.method`` other than ``none`` and ``tls.enabled: false``
**refuses to render at all** rather than hand out cookies in the clear. You
therefore have to pick one of two ways to get the connection encrypted.

**Let this nginx terminate TLS.** Set ``tls.enabled: true`` with a certificate
and key, and nginx serves HTTPS on 443, redirects the plain port to it, and marks
session cookies — the login wall's and each terminal's own — so browsers only
ever send them over HTTPS. Bringing the
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

This shape needs one key more, and it is **required**, not optional:

.. code-block:: yaml

   modules:
     web_terminals:
       external_origin: https://terminals.example.org   # what the browser reaches
       auth:
         method: password
         allow_insecure_http: true

``external_origin`` is the address **browsers** open, which here is the load
balancer's, not this host's. Every terminal refuses a request that would change
something — a chat message, an approval, a file write — unless the browser says
it came from that address, and nothing in the rest of this configuration can
work out what the thing in front answers on. Leave it unset and the
deployment looks entirely healthy: the containers are up, the landing page
renders, each terminal opens — and every action taken in one is refused.

Write it as a bare origin: a scheme, a host, and a port if it is not the
scheme's default. No path, no trailing slash. Anything else is refused when you
build, which is the point — the alternative is finding out from a browser.

Set it in any deployment where the address people type is not this nginx's own,
including a plain reverse proxy or a DNS alias in front of it. When browsers
reach this nginx directly — every other shape on this page — leave it out and
the address is derived from ``deploy.fqdn`` and the published port.

What ``allow_insecure_http`` is *not* is a way to postpone certificates on a
reachable host. With it set and nothing terminating TLS, anyone who can watch
the traffic can copy a session cookie and become that user. An isolated network
where you accept that risk is the only other case for it.

Passwords, and where they live
------------------------------

In password mode ``osprey up`` makes sure every user on the roster has a
password hash before it starts anything, and aborts before a single container
starts if it cannot — an unwritable file is caught here rather than becoming a
stack nobody can log in to. The same check covers the keys used to sign session
cookies, so an OIDC deployment can abort the same way even though it provisions
no passwords at all. The usual cause either way is permissions on ``.env.auth``
or on the project directory.

The hashes and signing keys live in ``.env.auth`` in the project root — mode
``0600``, listed in the generated ``.gitignore`` next to ``.env.users``,
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

``<USER>`` is the username uppercased with ``-`` turned into ``_``. That mapping
is what keeps one user's credentials out of another user's terminal, and it keys
each terminal's operator secret as well as its password — so *every* deployment,
authenticated or not, refuses to render when two roster names collide under it
(``alice-b`` and ``alice_b``), or when a name falls outside
``[a-z0-9][a-z0-9_-]*``.

To change a password later:

.. code-block:: bash

   osprey users passwd alice

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

**Use** ``osprey users remove alice`` **, not a hand-edit.** Deleting a
roster entry and running ``osprey up`` removes alice's container, and the
authentication service stops answering for a name that is no longer on the
roster — but her hash stays in ``.env.auth``. Add the name back months later and
her old password works again. ``decommission`` (or ``prune``, for names already
edited out) is what actually retires the credential.

**A plaintext password in** ``.env`` **survives decommission.** If you seeded
alice's password by putting ``OSPREY_AUTH_PW_ALICE`` in the project's ``.env``,
decommissioning her clears the hash but leaves that line — and the next
``osprey up`` for a new alice hashes it straight back in, handing the new
person the departed one's password. The decommission warns you, but the warning
scrolls past in a deploy log weeks before anyone reuses the name. **Delete the**
``.env`` **line by hand when the person leaves.**

**In OIDC mode,** ``decommission`` **is the verb that ends a session.**
``prune`` cleans up users already off the roster, but it only restarts the
authentication service when it actually removed a password entry — and an OIDC
user has none. Their container is gone either way, so the stale route just
fails; but if what you need is that person's *session* closed now, run
``decommission``.

To turn login back off, set ``auth.method: none`` and run ``osprey up``.
That re-renders nginx and the compose file, drops the authentication service,
and returns the stack to the open posture described at the top of this section.
``.env.auth`` is left in place, so turning login on again keeps everyone's
existing password.
