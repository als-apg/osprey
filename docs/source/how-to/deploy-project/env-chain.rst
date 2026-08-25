.. _deployment-env-chain:

==========================================
Environment Variables (the ``.env`` chain)
==========================================

A deployment reads its environment from two files at the repository root:

.. list-table::
   :header-rows: 1
   :widths: 20 55 25

   * - File
     - What belongs in it
     - Tracked in git?
   * - ``.env.shared``
     - What the whole site shares: a proxy, a facility hostname, a port
       everyone uses. Never a secret — this file is committed.
     - yes
   * - ``.env``
     - This host's own values, and every secret: API keys, service tokens,
       passwords.
     - no

**The local file wins.** A variable set in both takes its value from ``.env``,
on every path that reads them — the deploy, the CLI, the containers. That is
the whole rule: same syntax, same variables, ``.env.shared`` simply sits lower.
Setting a key in ``.env`` is how one host departs from a shared default, and
there is nothing else to do about it — unless the profile has declared that
variable the shared file's to decide (see :ref:`deployment-pinned-env`).

Both files stay on the host. Neither ever enters a container image: they are
read at run time and handed to the container runtime, which uses them to fill
in the ``${VAR}`` placeholders in the rendered compose files. A variable reaches
a running container only where a template maps it in. *How* the two files are
handed over differs by compose provider (see
:ref:`compose-provider-compatibility`); what they resolve to does not.

``.env`` has to exist. Rather than start a stack whose every ``${VAR}``
substitutes to nothing, ``osprey up`` refuses when the file is missing. On an
interactive terminal it first offers to seed one, but only when your shell has
the key to seed it with — this deployment's own provider auth variable,
exported. Otherwise start from the example:

.. code-block:: bash

   cp .env.example .env
   # Edit .env with your actual values

See :ref:`profile-secrets`.

The ``.env*`` family
--------------------

Two files are yours to edit and one is documentation. The rest are written for
you — most of them derived, rewritten whenever a command needs them, and not
worth editing because the next command overwrites them:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - File
     - Role
   * - ``.env.shared``
     - edit — shared defaults, the same on every host
   * - ``.env``
     - edit — this host's values and every secret
   * - ``.env.example``
     - docs — every variable this deployment reads, with no values
   * - ``.env.users``
     - machine — the env file every per-user web-terminal container runs with,
       derived from the chain (multi-user deployments only)
   * - ``.env.auth``
     - both — the web terminals' password hashes and cookie-signing secrets,
       minted by the deploy, but also where you put an OIDC client id and
       secret by hand (multi-user deployments only; see :doc:`/how-to/web-terminal/multi-user/index`)
   * - ``build/.env.merged``
     - machine — the chain collapsed into one file, for compose providers that
       accept only one
   * - ``build/.env.chain-state.json``
     - machine — fingerprints of the shared values as of the last deploy, so a
       stale local value can be spotted. It stores digests, never values.

The generated ``.gitignore`` keeps all of them out of version control except
``.env.shared`` and ``.env.example``, which carry nothing a host may not share.

.. note:: Upgrading a multi-user deployment

   The web terminals' env file changed name to ``.env.users``. ``osprey up``
   does the rename for you the next time you deploy, and if both names are
   present it keeps the new one and removes the leftover, naming both paths.
   Only ``osprey up`` does this, so a stack you stop before you next deploy
   still carries the old name — and because the web stack's compose file names
   ``.env.users``, ``osprey down`` fails with an env-file-not-found error until
   the rename has happened. Do it yourself in that case:

   .. code-block:: bash

      mv .env.production .env.users

What a deploy writes back
-------------------------

``osprey up`` also *writes* to ``.env``. On first deploy it mints any missing
service tokens and passwords (for example ``EVENT_DISPATCHER_TOKEN``,
``ZO_ROOT_USER_PASSWORD``, or ``ARIEL_DB_PASSWORD``) so no service ever starts
on a blank or publicly-known credential, appends them under a "Minted by
deploy" heading, and restricts the file to owner-only permissions.
``osprey build`` appends the pointers it derives from what it just rendered,
under a "Derived by build" heading.

Both writers are append-only, and a value already on file always wins. That is
what makes the stack reproducible: a later start comes up on the same
credentials the running containers were initialized with, instead of minting a
second set they do not trust. There is no second copy anywhere — the ``.env``
beside ``profile.yml`` is the deployment's whole secret store — so back it up.

Minted values only ever land in ``.env``, never in ``.env.shared``. A minted
credential belongs to this host, and ``.env.shared`` is committed.

What a deploy tells you about the chain
---------------------------------------

Two layered files make one new mistake possible, so every ``osprey up`` reports
on the chain before it starts anything. Variable names are printed; values never
are.

* **Overrides** — the keys ``.env`` overrides in ``.env.shared`` are listed
  once, by name, as information. That is the chain working as intended.
* **Stale pins** — a warning, and it is reserved for one exact case: ``.env``
  still holds the value ``.env.shared`` carried *before* the shared file
  changed. That is the signature of a value copied from the default of the day
  and then forgotten rather than one this host chose, and the stack starts on
  the superseded value looking perfectly healthy. To adopt the shared value,
  remove the key from ``.env``; to keep a local value deliberately, set it to
  the value this host actually wants. The warning repeats until you do one or
  the other.
* **Pinned variables** — a refusal. A profile can declare that some variables
  are the shared file's to decide and no one else's; ``osprey up`` then refuses
  to start when anything contradicts that. See :ref:`deployment-pinned-env`.
* **Chain drift** — a refusal. Which env files the stack reads is decided when
  the project is rendered, not when it starts: adding ``.env.shared`` to a
  project built without one puts none of its values into the containers, and
  removing one leaves the render pointing at a file that is gone. ``osprey up``
  refuses and names ``osprey build``. Re-render, then start.
* **Shell exports** — when a variable exported in your shell disagrees with the
  value the chain resolves to, ``osprey up`` names it and says which of the two
  the compose provider it just probed will actually substitute (see
  :ref:`compose-interpolation-precedence`). A warning, so the export stays
  available as an escape hatch — except for a pinned name, where it is a
  refusal.

.. note::

   Postgres reads ``ARIEL_DB_PASSWORD`` (as ``POSTGRES_PASSWORD``) only when
   initializing a **fresh** data volume. A volume created before the password
   was minted keeps its original password; the ``${ARIEL_DB_PASSWORD:-ariel}``
   fallback — applied by the compose template and by the DSN the agent derives
   from ``services.postgresql`` — keeps such deployments working. To adopt
   the minted password, remove the ``ariel_postgres_data`` volume and redeploy
   (this deletes the stored logbook data — re-ingest afterwards).

.. _deployment-pinned-env:

Pinning a variable to the chain
-------------------------------

``.env`` winning is the right default, and for a few variables it is the wrong
one: a proxy every host at the site has to go through, a hostname the whole
facility shares. A profile can say so, in ``profile.yml``:

.. code-block:: yaml

   env:
     pinned:
       - HTTPS_PROXY
       - FACILITY_ARCHIVER_HOST

Pinning does not change how a value resolves. It changes what ``osprey up``
does when something contradicts the declaration: it refuses to start, rather
than let the stack run on a value from a source the deployment said it would
not come from. Three checks, one for each way that can happen:

* **The declaration itself.** Pinning a name the deploy writes for you is
  refused — a service token or password it mints, a service default it records,
  a Bluesky substrate variable, a credential ``--reuse-stores`` restores from a
  surviving volume. A pin says the chain is the variable's only source, and
  ``osprey up`` writing that same name into ``.env`` contradicts it by
  construction. The message names each offender and its remedy:
  ``unpin <NAME>; it is machine-minted.`` The check covers every name any writer
  can produce, not only the services this deployment currently enables, so
  turning a service on later cannot make a profile that was fine start
  refusing.
* **A local override.** ``.env`` setting a pinned name is refused. Without the
  pin that is just this host's business; with it, the stack would start on a
  value contradicting what the deployment declares it runs on — and nothing
  else would notice, because the override is well-formed and the stack comes up
  healthy.
* **A shell export.** An export disagreeing with the chain is a warning for any
  unpinned variable, deliberately: exporting over the store is a legitimate
  one-off. For a pinned name it is a refusal, under either compose provider.
  Docker Compose would substitute the exported value; podman-compose would
  ignore it. Neither outcome is acceptable for a name the deployment declared
  its own, and a repo must not be startable on one host and refused on another
  for the same shell state, so the answer is the same on both — only the
  explanation differs.

All three run before any secret is minted and before any image is built, so a
deploy that is going to stop here stops having provisioned nothing. All three
print names and never values.

Either remedy is one edit. To start on the declared value, remove the name from
``.env`` or unset the export. To let this host decide it after all, drop the
name from ``env.pinned`` and re-run ``osprey build``.

**A pin covers the chain and the shell, not the files derived from them.**
Those two are where a value can reach a container while every reading of
``.env.shared`` goes on describing something else. ``.env.users`` carries a
copy of what the chain already resolved to, and ``.env.auth`` holds credentials
the deploy mints; neither is a place a pinned variable's value gets decided, so
there is nothing there for the declaration to contradict.

The declaration is read from ``profile.yml`` at the repository root, as
``osprey build`` emitted it — a flat document. A hand-written profile that
inherits its ``env`` block through ``extends:`` leaves the parent's pins unread
and nothing enforced; spell them out in the repository's own profile.

.. note::

   The warning ``osprey up`` prints as **Stale pin** is a different thing: a
   value pinned by hand in ``.env`` that has fallen behind the shared default.
   It is unrelated to ``env.pinned``, and it stays a warning.
