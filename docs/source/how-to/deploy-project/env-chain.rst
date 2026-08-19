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
there is nothing else to do about it.

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
       secret by hand (multi-user deployments only; see :doc:`/how-to/multi-user`)
   * - ``build/.env.merged``
     - machine — the chain collapsed into one file, for compose providers that
       accept only one
   * - ``build/.env.chain-state.json``
     - machine — fingerprints of the shared values as of the last deploy, so a
       stale local pin can be spotted. It stores digests, never values.

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
* **Chain drift** — a refusal. Which env files the stack reads is decided when
  the project is rendered, not when it starts: adding ``.env.shared`` to a
  project built without one puts none of its values into the containers, and
  removing one leaves the render pointing at a file that is gone. ``osprey up``
  refuses and names ``osprey build``. Re-render, then start.
* **Shell exports** — when a variable exported in your shell disagrees with the
  value the chain resolves to, ``osprey up`` names it and says which of the two
  the compose provider it just probed will actually substitute (see
  :ref:`compose-interpolation-precedence`).

.. note::

   Postgres reads ``ARIEL_DB_PASSWORD`` (as ``POSTGRES_PASSWORD``) only when
   initializing a **fresh** data volume. A volume created before the password
   was minted keeps its original password; the ``${ARIEL_DB_PASSWORD:-ariel}``
   fallback — applied by the compose template and by the DSN the agent derives
   from ``services.postgresql`` — keeps such deployments working. To adopt
   the minted password, remove the ``ariel_postgres_data`` volume and redeploy
   (this deletes the stored logbook data — re-ingest afterwards).
