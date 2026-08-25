.. _how-to-build-profiles:

==============
Build Profiles
==============

A **build profile** is the directory your facility owns: one ``profile.yml``, your
data tree, your secrets, and any rules, skills or scripts you write. ``osprey build``
reads that directory and renders a project from it.

The profile is the source of truth. The project is a derived artifact — regenerable,
and safe to delete and rebuild at any time.

.. dropdown:: What You'll Learn
   :color: primary
   :icon: book

   - Creating a deployment repository and rendering its ``build/`` zone
   - What lives in a profile: the convention directories, ``data/``, secrets, personas
   - Moving an artifact you want to own out of ``build/`` and into the profile
   - Shipping and wiring your own hook scripts
   - Keeping a profile and its build in step

   **Prerequisites:** A working OSPREY installation (``uv sync``).

   **Time:** 15--30 minutes for a basic profile.


Preset → Profile → Build
========================

.. raw:: html
   :file: ../_diagrams/build-profiles.html

- **Preset** — a bundled starting point, shipped inside OSPREY
  (``src/osprey/profiles/presets/``). Examples: ``hello-world``,
  ``control-assistant``, ``ariel-standalone``, ``channel-finder-standalone``.
  Run ``osprey profile presets`` to list them.
- **Profile** — the ``profile.yml`` at the root of your deployment repository,
  with the material it names beside it. Created once from a preset, then edited
  and kept in version control. Everything the preset configured is written out
  here explicitly: nothing is inherited at build time.
- **Build** — the ``build/`` zone ``osprey build`` renders. Never edit it in
  place; the next build wipes and re-renders the whole thing.

Because nothing is inherited, a later OSPREY release that improves a preset does
**not** change your profile. To see what moved, create a fresh deployment in a
scratch directory and diff it:

.. code-block:: bash

   osprey init /tmp/fresh --preset control-assistant
   diff -u /tmp/fresh/profile.yml my-facility/profile.yml


Creating a deployment
=====================

One command creates a deployment repository from a preset:

.. code-block:: bash

   osprey init my-facility --preset control-assistant

That writes the repository and stops. Look at the profile, edit it, then render
and start it from inside:

.. code-block:: bash

   cd my-facility
   osprey validate
   osprey build
   osprey up -d

.. admonition:: Every build reads the repository's own profile
   :class: important

   There is no build that renders straight out of a bundled preset. The preset
   is applied once, at ``osprey init``, and written out in full — after that,
   ``profile.yml`` is the only input. A later OSPREY release that changes the
   preset does not reach an existing deployment.

What ``osprey init`` writes
---------------------------

.. code-block:: text

   my-facility/
     profile.yml     the full configuration — edit freely
     data/           facility content: channel databases, knowledge, lattice
     .env.example    every variable the agent reads, documented, no values
     .env.shared     shared, committed defaults — no secrets
     .env            your values and secrets (only when your shell had keys to seed)
     README.md       explains the layout, for whoever opens the repository next
     triggers.yml    the events the agent runs on (dispatch profiles only)
     personas/       one delta per web-terminal persona (persona presets only)
     web-terminal-context/  the shared base.md baseline, plus one seeded
                     directory per operator on the roster
     ci-extra.yml    the facility's own CI jobs; never regenerated
     .gitignore      keeps build/, var/ and .env out of version control
     build/          rendered by `osprey build`; disposable
     var/            agent memory and audit log; durable

``triggers.yml``, ``personas/`` and ``web-terminal-context/`` appear only when
the preset calls for them — a ``hello-world`` deployment has none of them.

``git init`` and an initial commit run at the end. There is no CI pipeline yet:
the profile ships its ``deploy:`` block commented out, so there are no
coordinates to render one from. Fill the block in and ``osprey scaffold ci``
writes the pipeline — see :doc:`deploy-a-facility`.

Directories for your own artifacts (``rules/``, ``skills/``, and the rest) are
**not** created up front. Create the ones you need; a directory you never create
simply means the profile contributes nothing of that kind.


Convention directories
======================

Put a file in the directory that matches what it is, and the build carries it
into the project. There is nothing to declare in ``profile.yml``: the directory
name *is* the declaration, and where each one lands is fixed.

.. list-table::
   :header-rows: 1
   :widths: 26 30 44

   * - Put it here
     - It lands here
     - One entry is
   * - ``rules/``
     - ``.claude/rules/``
     - a ``.md`` file
   * - ``skills/``
     - ``.claude/skills/``
     - a directory with a ``SKILL.md``
   * - ``agents/``
     - ``.claude/agents/``
     - a ``.md`` file
   * - ``commands/``
     - ``.claude/commands/``
     - a ``.md`` file
   * - ``output-styles/``
     - ``.claude/output-styles/``
     - a ``.md`` file
   * - ``hooks/``
     - ``.claude/hooks/``
     - a script, usually ``.py``
   * - ``web-terminal-context/``
     - ``docker/web-terminal-context/``
     - a directory named for one operator, plus one shared ``base.md``
   * - ``mcp_servers/``
     - ``_mcp_servers/``
     - a directory per server
   * - ``services/``
     - ``services/``
     - a directory per compose service
   * - ``project/``
     - the project root
     - any file, mirrored verbatim

Nested paths inside a markdown directory are preserved, so
``commands/orbit/correct.md`` stays namespaced. Skills, MCP servers, services
and per-user context copy as whole directories — the directory *is* the entry,
and a build replaces it wholesale.

A file you ship this way is registered as **yours** in the project: later
re-renders never overwrite it, and cleanup never removes it. Name a file after
something the framework also renders (``rules/safety.md``) and yours wins.

Ownership is *derived* from what the build actually copied, after exclusions
are applied — there is no list to maintain. An artifact a persona delta
excludes is not copied and therefore not owned, so the framework's own version
renders in its place.

.. admonition:: A misspelled directory is silent
   :class: important

   ``rule/`` is not ``rules/``, and nothing reads it. The build warns about
   unrecognized top-level entries in a profile for exactly this reason — read
   that warning rather than wondering why an artifact never arrived.

.. _profile-reserved-paths:

Paths the profile may not write
-------------------------------

``project/`` is the escape hatch for anything without a home in the table above.
It cannot write paths the build already owns, because each of those has its own
channel:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Path
     - Written by
   * - ``config.yml``
     - the profile's ``config:`` block
   * - ``.claude/settings.json``
     - ``config:`` keys — ``claude_code.permissions``, ``claude_code.hooks``
   * - ``.claude/hooks/hook_config.json``
     - the build, from your ``mcp_servers:`` and ``control_system.write_tools``
   * - ``.mcp.json``
     - the profile's ``mcp_servers:`` block
   * - ``CLAUDE.md``
     - the profile's ``claude_md_template:`` key
   * - ``.env`` / ``.env.example``
     - the profile's own ``.env`` file and ``env:`` keys
   * - ``.osprey-manifest.json``
     - the build itself
   * - ``data/simulation/channel_manifest.json``, ``channel_limits.json``
     - the profile's ``data/`` directory

A profile that targets one of these is rejected at build time, with the owning
channel named. The same refusal applies to a claim (below).

This table is about *build channels* — which part of the profile is allowed to
produce a file. A separate question is which files the running agent may not
rewrite, whatever channel produced them; that is
:doc:`the protected set <control-systems/protected-set>`, and it is neither a subset nor a
superset of this one.

``hook_config.json`` is the one worth understanding: the write-safety hook reads
it to decide what counts as a hardware write. A hand-written copy would be
treated as yours and never regenerate, quietly freezing that decision.

.. _profile-context-baseline:

The shared web-terminal baseline
--------------------------------

``web-terminal-context/`` is the one convention directory that accepts a loose
file: ``base.md``, the text every operator's terminal starts from. It sits
beside the per-operator directories and rides the same channel, landing at
``docker/web-terminal-context/base.md``.

Three layers stack, in this order:

- **the framework's fallback.** Every build installs a generic ``base.md`` into
  the rendered project, so a deployment always has one.
- **the profile's own copy.** A ``web-terminal-context/base.md`` in the profile
  is copied over that fallback. ``osprey init`` materializes one — from the app
  template's text where it ships its own, otherwise the framework's — so the
  baseline is in your repository from the first minute, where you can read and
  edit it.
- **each operator's ``extra.md``.** At seed time the baseline and that user's
  ``web-terminal-context/<user>/extra.md`` are concatenated, and the result
  becomes their ``CLAUDE.md``.

Unlike the per-operator directories, the baseline is **not** roster-derived. It
is copied on every build, including a persona render that has no roster at all,
and ``osprey init`` writes it whenever the profile stands up web terminals
(``modules.web_terminals.enabled``) — even when the roster is still empty, so a
profile that names no operators yet still starts from text it can see. A
profile with no web-terminal module gets no ``web-terminal-context/``.

A persona drops it with the ordinary convention vocabulary, qualified like any
other profile-shipped file:

.. code-block:: yaml

   exclude:
     web-terminal-context:
       - web-terminal-context/base.md   # the framework fallback renders instead

The baseline is the only part of this tree a persona has to exclude. A persona
delta switches the web-terminal module off, so its render resolves an empty
roster and copies no operator directories in the first place — naming one here
matches nothing and is silently ignored.

To keep the baseline in the deployment but leave it out of one persona's
terminals, set ``seed_base: false`` on that persona's catalog entry
(``modules.web_terminals.personas.<name>``): its users are seeded from their
own ``extra.md`` alone.

.. admonition:: ``base.md`` is the only loose file
   :class: important

   Any other file directly in ``web-terminal-context/`` is refused at build
   time, and the message names both routes: move it into the directory of the
   operator it belongs to, or rename it to ``base.md`` if it is the text
   everyone starts from. Directory names here are matched against the resolved
   roster, so a directory invented to hold a stray file is skipped as an
   operator who has left — and nothing in it would ever be read.


.. _profile-self-contained:

A profile is self-contained
===========================

Everything a profile contributes lives inside the profile directory. That is
what keeps a build reproducible: clone the deployment repository anywhere, and
the same project renders.

The build holds the convention directories to it. An entry that is a symlink
resolving outside the profile is refused, with the remedy — copy the target in
instead. ``osprey scaffold claim`` refuses the same shape before it moves
anything, because a claim that carried such a link into the profile would
report success and then break every later build.

.. note::

   The check catches the mistake, not an adversary: copies dereference
   symlinks, and the scan does not descend into a symlinked directory. Read it
   as the rule the build holds you to, not as a boundary it enforces.

The ``data:`` tree is the exception
-----------------------------------

``data:`` is a path the profile *names*, not material the build finds by
convention, and it is checked differently. The path resolves against the
profile directory, and only its existence and shape are checked — that it is
there, and that it is a directory. Containment is never checked, so a tree may
sit above the profile directory (``data: ../shared-data``): a tree more than
one profile builds from can live beside them rather than inside either.

Staleness is unaffected. The resolved tree is folded into the profile hash
wherever it sits, so editing it still reports the built project as out of date.
The exception is about where the tree may live, not about the build losing
sight of it.


.. _profile-claim:

Taking ownership of a framework artifact
========================================

Everything the build renders is framework-managed: every ``osprey build``
refreshes it from the installed OSPREY version, so framework fixes reach your
project on their own — and an edit made in place is overwritten the next time.
Claiming is how an edit survives.

To customize something OSPREY generates — a rule, an agent, a service template
— move it into the profile:

.. code-block:: bash

   cd my-facility
   osprey scaffold claim rules/safety
   osprey scaffold claim agents/channel-finder
   osprey scaffold claim services/postgresql

The artifact is **moved** out of ``build/`` and into the matching convention
slot of the repository's profile — ``rules/safety`` lands at
``<profile>/rules/safety.md``, ``services/postgresql`` at
``<profile>/services/postgresql/``. Edit it there, then rebuild:

.. code-block:: bash

   osprey build

The next build copies it back and registers it as yours. There is no YAML to
edit — ownership is derived from what the build copied, not declared.

.. code-block:: bash

   osprey scaffold list                     # what is framework-managed, what is yours
   osprey scaffold diff rules/safety        # how far your copy has drifted
   osprey scaffold unclaim rules/safety     # give it back to the framework

``unclaim`` holds only until the next build: while the profile still supplies the
file, the build copies it in and registers it again. To give an artifact up for
good, delete it from the profile.

A claim is refused, with the reason, when:

- the project names no profile to claim into (nothing would keep the edit);
- the artifact is **generated**, not authored — ``hook_config.json``,
  ``settings.json``, ``.mcp.json``, ``CLAUDE.md``. The message names the config
  key that *does* control it;
- the artifact is a symlink, or holds one pointing outside itself — a profile
  must be :ref:`self-contained <profile-self-contained>`;
- the profile slot is already occupied. A claim never overwrites profile
  material.

The web terminal offers the same move from the browser: its scaffold gallery
overrides a framework-generated artifact by claiming it for you (see
:doc:`web-terminal/operate`).

Before reaching for a claim, check whether a config key or an environment
variable already covers your need — most service knobs (ports, images,
credentials, retention) are configurable without owning the template.


Custom hooks
============

A hook is a script the agent runs at a defined moment — before a tool call, at
session start. Ship yours through ``hooks/``:

.. code-block:: text

   my-facility/
     hooks/
       facility_guard.py

That copies the script to ``.claude/hooks/facility_guard.py``. **It does not make
the script run.** Shipping and wiring are two steps; the second is a ``config:``
key naming the event it fires on:

.. code-block:: yaml

   config:
     claude_code.hooks.PreToolUse:
       - hook: facility_guard.py
         matcher: "mcp__controls__.*"   # optional — defaults to every tool call
         timeout: 10                    # optional — seconds, defaults to 60
     claude_code.hooks.SessionStart:
       - facility_banner.py             # shorthand when there is nothing to qualify

Valid events: ``PreToolUse``, ``PostToolUse``, ``UserPromptSubmit``,
``SessionStart``, ``SessionEnd``, ``Stop``, ``SubagentStop``, ``Notification``,
``PreCompact``.

.. admonition:: An undeclared hook never runs
   :class: important

   Without a declaration the script still lands in ``.claude/hooks/`` and
   survives every rebuild — doing nothing. This matters most for a safety check,
   where "present" looks like "enforcing."

Declared wiring is **added** to the framework's, never put in its place. Your
declaration cannot remove, alter, or displace anything the generated settings
already wire — the write-safety gate and the rest render unchanged. Every hook
whose matcher fits runs, so a declared hook is one *more* check on top of the
framework's, never a substitute for one.

A declaration is refused at build time, with the reason, if it names a hook the
resolved profile does not ship, a built-in hook whose wiring the framework
already owns, or anything outside ``hooks/``.

.. _profile-unwire-hook:

Unwiring a hook in a persona
----------------------------

The wiring is a ``config:`` key, so a persona delta overrides it — but you have
to use the right spelling:

.. code-block:: yaml

   config:
     claude_code.hooks.PreToolUse: null   # this event now wires nothing
     claude_code.hooks: {}                # or: unwire every event at once

Either form leaves the script itself in place: unwired, not unshipped.

.. admonition:: An empty list does nothing
   :class: important

   Persona lists merge **additively** with the profile's, so
   ``claude_code.hooks.PreToolUse: []`` adds no entries and leaves the hook
   wired — silently. ``null`` is the spelling that works.

   This matters because a persona that ``exclude:``\ s a shipped hook **must**
   unwire it in the same delta: the build refuses a declaration pointing at a
   hook the persona dropped. That refusal prints the exact ``null`` line to
   paste, so if you reach for ``[]`` on an excluded hook the build hands you the
   correction rather than letting it pass.

Replacing a built-in hook
-------------------------

Shipping and declaring behave differently for the framework's own hooks.

**Shipping** a file named for a built-in *replaces* it: ``hooks/`` is keyed by
filename, because that is what the generated settings run. The built-in
``writes-check`` hook is ``osprey_writes_check.py``, so a profile file of that
name is the one the agent runs wherever the framework already wired that name.

**Declaring** a built-in hook is refused. The framework wires its own hooks from
the profile's ``hooks:`` selection, so a declaration naming one would invoke it
twice. Select or unselect a built-in through ``hooks:``; never through
``claude_code.hooks``.


Personas
========

Some presets give each operator their own web terminal, and each terminal runs
with a persona — usually a capability posture, such as read-only versus
write-capable, though a persona can just as well be a different product
sharing the deployment. For those presets (``control-assistant``),
``osprey init`` writes one file per persona:

.. code-block:: text

   my-facility/
     profile.yml
     data/
     personas/
       readonly.yml       # a read-only terminal
       readwrite.yml      # a write-capable terminal
       admin.yml          # the one terminal that may edit the deployment
       ariel.yml          # the standalone ARIEL logbook terminal

Each file holds only that persona's **differences** — for the read-only persona,
chiefly ``control_system.writes_enabled: false``; for the admin one, the
privileges the profile below it deliberately withholds. Sitting in ``personas/`` beside
``profile.yml`` is what makes it a persona: the build merges it over that profile
automatically. There is no ``extends:`` line to maintain, and no second data tree
or set of convention directories — everything else comes from the profile above
and stays in one place.

Edit a delta to change what that persona's terminal can do, and see the merged
result with:

.. code-block:: bash

   osprey validate personas/readonly.yml

``profile.yml`` points at these files by path — its web-terminal catalog carries
``build_profile: personas/<name>.yml`` for each one — so keep the names in step
if you rename one. ``osprey up`` reads the same catalog but renders nothing: a
persona project missing from ``build/`` stops the start and points at
``osprey build`` as the remedy. A bundled preset
name in that field is rejected, because a persona built from a preset of its own
would not share this profile's data tree, secrets or artifacts.


.. _profile-exclude:

Removing something a profile brings
-----------------------------------

``exclude:`` subtracts entries, and the spelling decides what it removes:

.. code-block:: yaml

   exclude:
     skills:
       - writing-bluesky-plans        # bare: stop selecting the built-in skill
     agents:
       - agents/channel-finder        # qualified: drop the profile's own file

A **bare** name unselects a built-in artifact, so it is no longer installed. A
**qualified** name (``<directory>/<name>``) omits the profile's own file for that
name — which is how a persona that wants the stock version back gets it: drop
your shadowing copy, and the still-selected built-in renders again.

Bare exclusion accepts ``skills``, ``rules``, ``hooks``, ``agents``,
``output_styles``, ``web_panels`` and ``dependencies``; qualified exclusion
accepts any convention directory. Excluding something that is not there is a
silent no-op.

Excluding a **declared hook** takes one more line: the same delta has to unwire
it with ``claude_code.hooks.<Event>: null``, or the build refuses the wiring
that now points at a file the persona dropped. See :ref:`profile-unwire-hook`.

.. admonition:: The mistake worth knowing about
   :class: important

   Use the **bare** spelling on a name your profile also ships a file for, and
   the exclusion does nothing visible: the built-in is unselected, but your file
   still renders, so the project comes out byte-identical and the build
   succeeds. The build warns when it sees this and names the qualified spelling
   that would actually drop the file. Read that warning rather than trusting a
   green build as proof the exclusion took.

.. note::

   ``exclude:`` carves a tier by *removing* capability. When the boundary you
   want is "may not write," prefer flipping the enforcement switch instead —
   the bundled ``control-assistant-readonly`` preset differs from its
   write-capable sibling only on ``control_system.writes_enabled``, leaving
   the tool surface identical (see :doc:`web-terminal/multi-user/tiers`).

To keep the bluesky server **on** while hiding an individual plan, set
``bluesky.excluded_plans`` instead:

.. code-block:: yaml

   bluesky:
     excluded_plans: [orm]

The named plan is then invisible to the agent and non-runnable. The same
block's ``plan_dir`` key does the opposite — it installs a directory of your
facility's own plans; see :doc:`bluesky/write-plans`.


.. _profile-host-variants:

One repository, several hosts
=============================

A test stand and a control-room machine can run the same deployment and still
need different settings — another hostname, other ports, a different theme.
Keeping two copies of the repository in step is the thing to avoid. Keep one
profile instead, plus a small overlay per host:

.. code-block:: text

   my-facility/
     profile.yml            # what every host shares
     profiles/
       teststand.yml        # what the test stand changes
       control-room.yml     # what the control room changes
     .env.variant           # which of the two THIS host builds

An overlay holds only the differences, in the same spelling ``profile.yml``
uses:

.. code-block:: yaml

   # profiles/teststand.yml
   config:
     deploy.fqdn: teststand.example.org
     web.theme: dark

Each host names the one it wants in ``.env.variant`` at the repository root:

.. code-block:: bash

   echo OSPREY_PROFILE_VARIANT=teststand > .env.variant

``osprey build`` merges that overlay over ``profile.yml`` before it renders
anything, and says which variant it used in its output. Personas are rendered
for the same host, so a stack cannot come up half-configured for another one.

The overlays are committed: which hosts a deployment has is part of what the
deployment is. The choice between them is not — ``.env.variant`` is covered by
the generated ``.gitignore`` along with the rest of the ``.env*`` family, so
each host keeps its own answer, and a build warns if the file ever does get
committed. There is no command-line flag for this today; the file is how a
host chooses. If one is added later, a variant named on the command line will
win over the file.

Three cases worth knowing about:

- **No setting, or an empty one.** The build renders ``profile.yml`` as
  tracked and the overlays sit unused. This is also what a host that has never
  heard of variants does, so adding ``profiles/`` to a repository changes
  nothing until a host selects something.
- **A name the repository has no file for.** The build stops and lists the
  names that would work. The previous ``build/`` is left as it was, as with
  every other refusal.
- **Switching variants.** Rebuild. ``build/`` is the render of one host's
  profile, and editing the setting does not re-render anything by itself —
  but switching ``.env.variant`` (or editing the selected overlay) marks the
  existing ``build/`` out of date, so ``osprey up`` names the stale render
  instead of starting it silently.

An overlay adds and replaces; it does not subtract. Lists are merged rather
than swapped, so use :ref:`exclude: <profile-exclude>` to take something away
on one host — that works in an overlay like anywhere else.


.. _profile-secrets:

Secrets
=======

API keys and service credentials live in one file: the ``.env`` at the root of
the deployment repository, read together with the committed ``.env.shared``
beside it. :ref:`deployment-env-chain` is that pair in full — which of the two
wins, what a deploy writes back into them, and the machine-written ``.env*``
files that go with them.

Seeding, once
-------------

``osprey init`` seeds the new repository's ``.env`` from your shell, and only
the keys of providers this profile actually references. Keys you exported for
other providers are named in the summary rather than copied in, so you can tell
"seen and not needed" from "lost". If your shell exported nothing usable, no
``.env`` is written at all (an empty secrets file reads as a configured one);
start it yourself:

.. code-block:: bash

   cp .env.example .env

.. admonition:: The only moment a shell export reaches the repository unasked
   :class: important

   It happens **once**, at ``osprey init``, and what it took is written under a
   "Seeded by ``osprey init`` from your shell" heading — so the file itself
   records where each value came from. No build reads your environment for
   secrets, and a later build never re-reads your shell. (``osprey up`` will
   seed a *missing* ``.env`` with this deployment's provider key, but only on an
   interactive terminal and only if you say yes at the prompt — see
   :ref:`deployment-env-chain`.)

   The practical consequence: once the repository has a ``.env``, exporting a
   key does not get it in — no writer ever overwrites a value already on file.
   Put it in ``.env`` yourself.

Who else writes to ``.env``
---------------------------

``osprey up`` and ``osprey build`` both append to the file — minted credentials
and derived pointers respectively — and both leave a value already on file
alone; :ref:`deployment-env-chain` has what each writes and why.


Profile YAML reference
======================

Every key ``profile.yml`` accepts — the field table, ``config:`` overrides, MCP
servers, tool permissions, ``services``, ``va_archiver``, lifecycle commands,
``env``, dependencies and the execution environment — is in
:doc:`/reference/configuration/profile`.


Regenerating a channel database
===============================

``osprey channel-finder build-database`` writes the generated database **into the
profile**, not into the project — beside the CSV inputs it came from, where it
survives a rebuild. The sequence is meant to run to completion:

.. code-block:: bash

   osprey channel-finder build-database
   # the deployment now reports its build as out of date
   osprey build
   # the report clears

The drift report in between is the reminder that the new database has not been
deployed yet — not a problem to fix. Use ``--output`` to write somewhere else.


Building
========

.. code-block:: text

   osprey build [OPTIONS]

Run it with no arguments, anywhere inside the deployment repository. It walks up
to ``profile.yml`` and renders the whole ``build/`` zone from it.

**Options**

.. list-table::
   :widths: 30 70

   * - ``-s, --stream``
     - Stream lifecycle step output in real time.
   * - ``--skip-lifecycle``
     - Skip ``pre_build``, ``post_build``, and ``validate``.
   * - ``--skip-deps``
     - Skip venv creation and dependency installation (CI mode).
   * - ``--runtime-root PATH``
     - Override ``project_root`` in the rendered config, for a build whose
       output runs somewhere other than where it was made.
   * - ``--repo DIRECTORY``
     - Deployment repository to act on (default: the nearest ``profile.yml``
       at or above the working directory).

.. admonition:: Settings are changed before the build, not during it
   :class: important

   ``osprey build`` takes no configuration overrides. Change a setting with
   ``osprey set``, which writes it into ``profile.yml`` — comments and
   formatting intact — and then build. The profile always describes what the
   build will produce, so there is no layer that vanishes afterwards.

Every build wipes and re-renders ``build/`` and preserves what you own: the env
chain (``.env.shared`` and ``.env``), ``var/``, and the repository's ``.git``.
It never touches the source zone — only ``osprey init --force`` replaces that.

**Examples**

.. code-block:: bash

   # See what presets ship
   osprey init --list-presets

   # Create the deployment, then render it
   osprey init my-assistant --preset control-assistant
   cd my-assistant
   osprey build

   # Change a setting, then carry it through to build/
   osprey set model=claude-sonnet-4-6
   osprey build

   # Render another repository's build/ without cd-ing to it
   osprey build --repo ~/deployments/als-test

Checking a profile without building
-----------------------------------

.. code-block:: bash

   osprey validate
   osprey validate personas/readonly.yml

Resolves the profile and runs the full consistency check — convention
directories, the data tree, service templates, lifecycle steps, environment
variables — reporting every problem found, not just the first.


What the build does
===================

1. Settle the profile (materialize from a preset on first use, or read the one
   you named), writing any ``--set`` / ``-O`` / ``--tier`` into it.
2. Resolve and validate the profile, including any persona delta merged over it.
3. Check ``requires_osprey_version``; abort if unsatisfied.
4. Clear the previous render. ``build/`` is wiped whole and re-made; nothing
   durable lives inside it to step around.
5. Run ``pre_build`` commands.
6. Create the project venv and install OSPREY plus the profile's dependencies.
7. Render the base template and the profile's ``data/`` tree. No env file is
   carried in: the repository's own ``.env.shared`` and ``.env`` stay where
   they are, and the containers are handed them at start time.
8. Apply the ``config:`` overrides.
9. Copy service templates and inject the profile's own services.
10. Apply the convention directories, and register what was copied as yours.
11. Persist ``mcp_servers:`` into ``config.yml``.
12. Stamp the manifest (``.osprey-manifest.json``), including the fingerprint
    ``osprey up`` compares the profile against before it starts anything.
13. Re-render the agent artifacts against the complete config, and validate that
    every tool an agent declares is backed by a permission.
14. Initialize git, then run ``post_build`` and ``validate`` commands.

The venv is created before rendering so templates can reference the resolved
Python path. The generated project runs standalone — nothing reaches back to the
profile at runtime.


What gets generated
===================

.. code-block:: text

   my-project/
   ├── .claude/
   │   ├── agents/           # built-ins, plus anything from the profile's agents/
   │   ├── rules/            # built-ins, plus the profile's rules/
   │   ├── hooks/            # hook scripts, plus the generated hook_config.json
   │   ├── skills/
   │   ├── output-styles/
   │   └── settings.json     # permissions, hook wiring, model config
   ├── .mcp.json             # MCP server configurations
   ├── CLAUDE.md             # generated system prompt
   ├── config.yml            # config with the profile's overrides applied
   ├── data/                 # the profile's data tree, materialized
   ├── _mcp_servers/         # facility server code from the profile
   └── .env.example          # every variable this deployment reads, no values

Which built-in agents, rules, hooks and skills are installed comes from the
``agents:``, ``rules:``, ``hooks:`` and ``skills:`` lists in ``profile.yml``.
Your own files come from the convention directories and are marked as yours.


Troubleshooting
===============

**"Either a profile path or --preset is required"** — every build reads a
profile. Name one, or ``--preset`` to have one materialized.

**"was materialized from preset X, but this build asks for Y"** — the profile
directory beside this project came from a different preset. Build ``Y`` under a
different project name so it gets a profile of its own.

**"Profile convention directories are invalid"** — a convention directory has
the wrong shape: a ``.md`` directory holding something else, a skill that is a
file rather than a directory, or a symlink pointing outside the profile. Every
problem is listed at once. The symlink case is the self-containment rule; copy
the target in (see :ref:`profile-self-contained`).

**"web-terminal-context/<name> is a file"** — that directory holds one
directory per operator, plus the shared ``base.md``. Move the file into the
directory of the operator it belongs to, or rename it to ``base.md`` if it is
the text everyone starts from (see :ref:`profile-context-baseline`).

**"project/ mirror writes N build-owned path(s)"** — the mirror targets a path
another channel owns. The message names the channel, and the exact move where
one exists.

**"Profile has N unrecognized top-level entry/entries"** — a warning, not an
error: a directory in the profile that nothing copies. Usually a typo of a
convention directory name.

**"Unknown profile key(s): 'overlay'"** — a profile has no ``overlay``
section. Move the files into the convention directory that matches what they
are (see the table above), or into ``project/`` for anything without one.

**"is already an OSPREY deployment repo"** — ``osprey init`` will not lay a
source zone over one that is already there. To re-render the project, run
``osprey build`` in the repo: it wipes and re-renders ``build/`` in place and
leaves the source zone alone. To replace the source zone itself from the
preset, re-run ``osprey init --force`` — which rewrites ``profile.yml``,
``data/``, ``personas/``, ``triggers.yml``, ``web-terminal-context/`` and
``.env.example``, losing any edit to them.

**"already exists, is not empty, and is not an OSPREY deployment repo"** — a
deployment repo is one directory that holds nothing else, so ``osprey init``
will not write into a directory that is already someone's. Choose an empty or
new path; ``--force`` does not apply here.

**"OSPREY X does not satisfy requires_osprey_version"** — upgrade OSPREY, or
relax the constraint in the profile.


.. seealso::

   :doc:`/reference/cli`
       Complete CLI command reference

   :doc:`agent-interfaces/add-mcp-server`
       How to build custom MCP servers for OSPREY

   :doc:`deploy-project/index`
       Container deployment after building
