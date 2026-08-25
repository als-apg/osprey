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

To customize something OSPREY generates — a rule, an agent, a service template
— move it into the profile:

.. code-block:: bash

   cd my-facility
   osprey scaffold claim rules/safety
   osprey scaffold claim agents/channel-finder
   osprey scaffold claim services/postgresql

The artifact is **moved** out of ``build/`` and into the matching convention
slot of the repository's profile. Edit it there, then rebuild:

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

Before reaching for a claim, check whether a config key already covers your need
— most service knobs (ports, images, credentials, retention) are configurable
without owning the template.


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
       ariel.yml          # the standalone ARIEL logbook terminal

Each file holds only that persona's **differences** — for the read-only persona,
chiefly ``control_system.writes_enabled: false``. Sitting in ``personas/`` beside
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
   the tool surface identical (see :doc:`multi-user`).

To keep the bluesky server **on** while hiding an individual plan, set
``bluesky.excluded_plans`` instead:

.. code-block:: yaml

   bluesky:
     excluded_plans: [orm]

The named plan is then invisible to the agent and non-runnable. The same
block's ``plan_dir`` key does the opposite — it installs a directory of your
facility's own plans; see :doc:`bluesky/write-plans`.

``bluesky.devices_file`` names the third piece: the file listing the devices
those plans may drive or record.

.. code-block:: yaml

   bluesky:
     devices_file: data/bluesky_devices.yml

That default puts the file inside the project, so it is built and shipped with
the deployment. An absolute path is yours instead — the build reads it where it
is and never rewrites or relocates it. A malformed entry fails the build, and a
deployment running the Virtual Accelerator with no file yet at a project path
gets one written for it from the project's own channel-limits database. The
file's format and the three cases are in :doc:`bluesky/write-plans`.


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
the deployment repository. That file is the deployment's single secret store.
A build never copies secrets into it or out of it, so a value you set once
survives every rebuild, and wiping ``build/`` takes no secret with it.

Three files at the repository root, and the difference matters:

- ``.env.example`` lists every variable the agent reads, with no values. It is
  safe to commit, and it is the file to read when you want to know what can be
  set.
- ``.env.shared`` holds the settings the whole site shares — a proxy, a
  facility hostname, a port everyone uses. It **is** committed, so nothing
  secret belongs in it.
- ``.env`` holds this host's own values and every secret. The generated
  ``.gitignore`` keeps it out of git.

``.env.shared`` and ``.env`` are read together, lowest first: a variable set in
both takes its value from ``.env``. Setting a key locally is how one host
departs from a shared default. :ref:`deployment-env-chain` covers the rest —
what a deploy reports about the pair, and the machine-written ``.env*`` files
that go with them.

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

.. admonition:: This is the only moment a shell export reaches the repository
   :class: important

   It happens **once**, at ``osprey init``, and what it took is written under a
   "Seeded by ``osprey init`` from your shell" heading — so the file itself
   records where each value came from. Nothing else in the pipeline reads your
   environment for secrets, and a later build never re-reads your shell.

   The practical consequence: exporting a key *after* the repository exists does
   not get it in. Put it in ``.env`` yourself.

Who else writes to ``.env``
---------------------------

Two writers append to the file, and both follow the same rule: **a value
already on file always wins.** Nothing overwrites what you put there.

- ``osprey up`` mints the credentials only a deploy can produce — database
  passwords, service tokens — and appends them under a "Minted by deploy"
  heading. Because a minted value is then on file, a later start comes up on the
  *same* secrets instead of minting a second set the running containers do not
  trust.
- ``osprey build`` appends the pointers it derives from what it just rendered —
  currently the virtual accelerator's channel manifest — under a "Derived by
  build" heading.

Both write to this one file, and to ``.env`` rather than ``.env.shared``: a
minted credential belongs to this host, and the shared file is committed. There
is no second copy anywhere — ``build/`` holds no secrets, and every service
reads them from here — so this is the file to back up.

The write-back is **append-only**. A key already in the profile keeps its value —
it is pinned by the docker volume that was initialized with it, and overwriting it
would leave the stack authenticating with something its own volumes reject — and a
value that disagrees is reported by name (never by value) for you to resolve by hand.

If the profile cannot be reached — it has moved or been deleted, or the project
names none — the deploy still works. The secrets stay in the project ``.env``, a
warning names the path that failed, and the project records that its ``.env`` is
the only copy. A later ``osprey build`` repeats that warning before touching the
directory.


Profile YAML reference
======================

.. list-table::
   :header-rows: 1
   :widths: 22 12 14 52

   * - Field
     - Type
     - Default
     - Description
   * - ``name``
     - string
     - *required*
     - Human-readable profile name.
   * - ``app_template``
     - string
     - ``control_assistant``
     - App template (data bundle) to render. Valid: ``control_assistant``,
       ``hello_world``, ``ariel_standalone``, ``channel_finder_standalone``.
   * - ``data``
     - string
     - ``None``
     - Facility data tree, relative to the profile directory (``data`` in a
       materialized profile). Replaces the bundled tree wholesale. May resolve
       above the profile directory; only existence and shape are checked
       (see :ref:`profile-self-contained`).
   * - ``provider``
     - string
     - *required*
     - LLM provider. Built-ins: ``anthropic``, ``cborg``, ``als-apg``; any
       provider declared under ``api.providers`` also works. The build aborts
       if none is set.
   * - ``model``
     - string
     - ``None``
     - Default model: a tier name (``haiku``, ``sonnet``, ``opus``) or a full
       provider model ID.
   * - ``channel_finder_mode``
     - string
     - ``None``
     - Channel finder pipeline (``hierarchical``, ``middle_layer``,
       ``in_context``, ``graph``). ``graph`` searches the deployment's graph
       store instead of a channel database, so it needs a ``services.graphdb``
       block (see :ref:`profile-graph-mode`).
   * - ``tier``
     - int
     - derived
     - Channel-database tier (1 or 3). Defaults from the channel finder mode;
       tier 1 is ``in_context``-only. ``graph`` has no tiered artifacts at all —
       leave ``tier`` unset there.
   * - ``connector``
     - string
     - *from preset*
     - Control-system connector (``mock``, ``virtual_accelerator``, ``epics``,
       …). Shorthand for ``config: {control_system.type: ...}``, so it can be
       set from the command line as ``--set connector=epics``. Setting both
       spellings on one command line is an error rather than a silent
       last-one-wins; a custom connector is still addressed by its dotted
       module path under ``config``.
   * - ``config``
     - mapping
     - ``{}``
     - Dot-notation overrides for the generated ``config.yml``.
   * - ``exclude``
     - mapping
     - ``{}``
     - Entries to subtract from what this profile would otherwise bring
       (see :ref:`profile-exclude`).
   * - ``hooks`` / ``rules`` / ``skills`` / ``agents`` / ``output_styles``
     - list
     - ``[]``
     - Built-in artifacts to install. Your own files go in the matching
       convention directory instead.
   * - ``mcp_servers``
     - mapping
     - ``{}``
     - MCP server definitions to inject.
   * - ``services``
     - mapping
     - ``{}``
     - Container services the deployment runs (see :ref:`profile-services`).
   * - ``va_archiver``
     - mapping
     - absent
     - Declares a stored archive for a simulated machine: a MongoDB store and a
       recorder the deploy stands up, seeds and records into
       (see :ref:`profile-va-archiver`).
   * - ``lifecycle``
     - mapping
     - ``{}``
     - Commands to run at build phases (``pre_build``, ``post_build``,
       ``validate``).
   * - ``env``
     - mapping
     - ``{}``
     - Variables the deployment needs: ``required``, ``defaults``, ``file``,
       ``pinned``. ``env.pinned`` lists variables ``.env.shared`` decides and a
       host may not override; a deploy refuses rather than start on a
       contradicting value (see :ref:`deployment-pinned-env`). Same name pattern
       as ``required``, and a machine-minted name cannot be pinned.
   * - ``dependencies``
     - list
     - ``[]``
     - Python packages to install into the project venv.
   * - ``environment``
     - mapping
     - ``{}``
     - Base interpreter the project environment is built from
       (see :ref:`profile-environment`).
   * - ``requires_osprey_version``
     - string
     - ``None``
     - PEP 440 specifier (e.g. ``>=2026.5.0``). The build aborts if unsatisfied.
   * - ``osprey_install``
     - string
     - ``local``
     - How to install OSPREY in the project venv: ``local``, ``pip``, or a
       PEP 508 spec.
   * - ``python_env``
     - string
     - ``project``
     - Python used by MCP servers: ``project``, ``build``, or an absolute path.
   * - ``provenance``
     - mapping
     - *written*
     - Which preset this profile was materialized from, and that preset's hash.
       Written by the materialization; do not edit it.


Configuration overrides
=======================

The ``config:`` section uses **dot notation** to override any key in the
generated ``config.yml``. The base keys are in
``src/osprey/templates/project/config.yml.j2``; app data bundles add further
sections in their own ``config.yml.j2``.

.. warning::

   Always write overrides as **dotted keys**, one per line — never as nested
   YAML. A nested block counts as *one* override whose value replaces the entire
   subtree. ``config: {claude_code: {model: opus}}`` wipes out everything else
   under ``claude_code`` (servers, permissions, …), silently. The dotted form
   ``claude_code.model: opus`` changes just that setting.

.. code-block:: yaml

   config:
     # Control system
     control_system.type: epics
     control_system.writes_enabled: true
     control_system.limits_checking.enabled: true

     # Archiver
     archiver.type: epics_archiver
     archiver.epics_archiver.url: https://archiver.facility.org

     # Set your real facility zone: it governs how the agent reads operator
     # times (parsed as facility-local) and renders every timestamp — not
     # just a display label.
     system.timezone: America/Los_Angeles

     # Channel finder
     channel_finder.pipeline_mode: middle_layer

     # Approval policy
     approval.default_policy: always


MCP server injection
====================

Custom MCP servers are recorded in the project's ``config.yml`` (under
``claude_code.servers``) and rendered from there into ``.mcp.json`` (server
configuration) and ``.claude/settings.json`` (tool permissions) — so a later
``osprey build`` re-renders them instead of losing them.

.. code-block:: yaml

   mcp_servers:
     my_server:
       command: python
       args: ["-m", "my_server"]
       env:
         CONFIG: "{project_root}/config.yml"
         API_KEY: "${MY_API_KEY}"
       permissions:
         allow: ["safe_tool"]
         ask: ["write_tool"]

Remote servers declare a ``url`` instead of a ``command``, plus an optional
``transport`` — ``http`` (streamable-HTTP, the default) or ``sse`` (legacy
Server-Sent Events):

.. code-block:: yaml

   mcp_servers:
     matlab:
       transport: http
       url: "http://localhost:8008/mcp"
       permissions:
         allow: ["mml_search"]

``command`` and ``url`` are mutually exclusive, and stdio servers must not set
``transport`` (launching via ``command`` *is* the transport).

**Placeholders:** ``{project_root}`` resolves at build time to the absolute
project path; ``${ENV_VAR}`` is preserved for the container or shell to resolve
at runtime.

**Permission wiring:** for a server named ``my_server`` with
``allow: ["safe_tool"]``, the build adds ``mcp__my_server__safe_tool`` to the
allow list.

Shipping the server's code
--------------------------

Put the package in the profile's ``mcp_servers/`` directory — one directory per
server. The build copies it to ``_mcp_servers/`` in the project, so the launch
command finds it:

.. code-block:: text

   my-facility/
     mcp_servers/
       phoebus/
         __init__.py
         __main__.py
         server.py

.. code-block:: yaml

   mcp_servers:
     phoebus:
       command: python
       args: ["-m", "phoebus"]
       env:
         OSPREY_CONFIG: "{project_root}/config.yml"
         PYTHONPATH: "{project_root}/_mcp_servers"
       permissions:
         allow: ["phoebus_launch"]

The directory name and the ``mcp_servers:`` key are independent: the directory
delivers the code, the key launches it.


.. _profile-tool-permissions:

Tool permissions
================

By default OSPREY blocks a handful of general-purpose tools — ``Bash``,
``Edit``, ``WebFetch``, ``WebSearch``, and the Playwright/Context7 plugins — so a
stock control-operator agent cannot shell out or browse the web. These defaults
are overridable per facility from ``config:``, using dotted keys:

.. code-block:: yaml

   config:
     claude_code.permissions.remove_deny: ["WebSearch", "WebFetch"]  # drop from the deny list
     claude_code.permissions.allow: ["WebSearch"]                    # then allow outright
     claude_code.permissions.ask: ["WebFetch"]                       # or route to human approval

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Key
     - Effect
   * - ``remove_deny``
     - Remove entries from the built-in deny defaults. Removing a write-capable
       tool that nothing else gates fails the build — see below
   * - ``deny``
     - Add facility-specific deny entries
   * - ``allow``
     - Add allow entries (no approval prompt)
   * - ``ask``
     - Add entries that route through human approval
   * - ``remove_ask``
     - Remove entries from the ask list

.. admonition:: Deny wins, and it wins at runtime too
   :class: important

   Permissions resolve as **deny > ask > allow**, and a static ``deny`` entry
   cannot be overridden during a session — an in-session "allow once" will not
   unblock it. Use ``ask`` for tools you want gated but still reachable. For the
   same reason, listing a tool under ``ask`` or ``allow`` does nothing until you
   also remove it from the deny list.

.. admonition:: You cannot un-gate a tool that can write
   :class: warning

   ``Bash``, ``Edit``, ``Write``, ``MultiEdit`` and ``NotebookEdit`` can write
   files or shell out, so ``osprey build`` refuses a profile in which one of
   them is neither in ``permissions.deny`` nor matched by a ``PreToolUse`` hook
   matcher. A ``remove_deny: ["Bash"]`` with nothing put in its place is
   therefore a build failure, not a silent widening. (The shipped presets gate
   the three file-writing tools with the ``memory-guard`` hook rather than
   denying them, so ordinary memory and notebook writes still work.)

   The build checks that a covering rule *exists*, not that the hook behind it
   refuses anything: a ``PreToolUse`` hook that exits 0 without a
   ``permissionDecision`` allows the call. A tool whose only cover is a matcher
   your own profile declares therefore builds with a warning — check that hook
   really denies. Matchers are read the way the agent runtime reads them: the
   bare tool name, a ``|`` alternation, any regex that matches the name
   (unanchored), or ``*``/``.*``/empty for every tool.


.. _profile-services:

Services
========

The ``services`` section defines facility containers the deployment runs
alongside OSPREY's built-in ones.

.. code-block:: yaml

   services:
     typesense:
       template: services/typesense     # relative to the profile directory
       config:
         port: 8108
         api_key: "${TYPESENSE_API_KEY}"

The ``template`` directory must contain at least ``docker-compose.yml.j2``. It is
copied into the project's ``services/`` tree, and the service is registered in
``config.yml``. Optional ``config`` values land under ``services.<name>``.

A service directory placed in the profile's ``services/`` convention directory is
carried across the same way and marked as yours — that is what
``osprey scaffold claim services/<name>`` produces.

One ``config`` key is read by the build itself: ``network``, which is either
``bridge`` (the default — the service joins the compose network and publishes
the ports it wants reachable) or ``host`` (it shares the host's network
namespace, which is what a service needs to see broadcast traffic or reach
ports other software publishes on the machine). Your template has to render the
setting for it to mean anything, and ``osprey build`` refuses a service that
declares ``network: host`` whose render does not carry it. See
:ref:`deployment-network-attachment` for what host mode changes and for
``dispatch.network``, the single knob that covers the event dispatcher and its
workers.

.. _profile-graph-mode:

Graph-mode channel finding
==========================

``channel_finder_mode: graph`` points the channel finder at the deployment's
graph store: the agent searches the facility knowledge graph for channels
instead of reading a channel database. The store *is* the database, so the
profile ships no channel-database inputs and pins no ``tier`` — what it does
need is a ``services.graphdb`` block, and the paradigm works with either shape
that block comes in.

A deployment that runs its own store already has one. The ``control_assistant``
app template renders ``graphdb`` into ``services`` and ``deployed_services``,
and ``osprey up`` starts, bootstraps and seeds it — see :doc:`deploy-project/index`:

.. code-block:: yaml

   name: control-room
   app_template: control_assistant
   provider: anthropic
   channel_finder_mode: graph     # no `tier` — graph has no tiered artifacts

To search a store the facility already runs, name it instead of deploying one.
That takes three things: an explicit ``uri`` on the block, the store's password
in the project ``.env`` as ``GRAPHDB_PASSWORD`` (nothing is minted for a store
this deployment does not own), and ``graphdb`` off the ``deployed_services``
list — otherwise the build still stands up a local Neo4j nobody queries.
``deployed_services`` is a list, so the override replaces it whole; write out
the services you *do* want rather than the one you are removing.

.. code-block:: yaml

   channel_finder_mode: graph
   config:
     services.graphdb.uri: bolt://graph.facility.org:7687
     services.graphdb.username: neo4j    # default `neo4j`
     # Whole-list replacement: the template's default is
     # [postgresql, openobserve, qmd, graphdb].
     deployed_services: [postgresql, openobserve, qmd]

An external store has to already hold a corpus for the mode to answer anything:
generate one with ``osprey knowledge build-ttl`` and load it with ``osprey
knowledge seed-graph`` (see :doc:`use-facility-graph`). A store this deployment
runs gets that during ``osprey up``; a store it only connects to does not.

A profile whose app template carries no ``services.graphdb`` block at all — the
channel-finder app template carries none — is refused at build time, naming the
missing block, rather than rendering a channel finder with nothing to read. An
attached project (``deploy_services: false``) is refused the same way unless it
names an external store, because it renders ``services: {}`` whatever its app
template says. For what the mode changes about the agent's answers, see
:doc:`use-channel-finder`; for the corpus behind them,
:doc:`use-facility-graph`.

.. _profile-va-archiver:

The ``va_archiver`` block
=========================

A deployment that serves simulated channels still needs somewhere to keep what
those channels did. Declaring ``va_archiver:`` is what gives it one: the build
adds a MongoDB store and a recorder to the service stack, ``osprey up``
seeds the store with history and then records the running machine into it, and
the ``mongodb_archiver`` connector reads it back.

.. code-block:: yaml

   va_archiver:
     host: localhost
     retention_days: 30
     hot_span_hours: 48
     hot_cadence_sec: 10
     tail_cadence_sec: 60
     freshness_channel: SR:DIAG:DCCT:01:CURRENT:RB

Every key is optional and the defaults describe a working archive; the block's
presence is the decision, not its contents.

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Key
     - Default
     - Meaning
   * - ``retention_days``
     - ``30``
     - How far back the archive reaches — both what a fresh deployment holds
       and what a running one keeps.
   * - ``hot_span_hours``
     - ``48``
     - How much of the recent end is kept at the dense cadence. May not exceed
       ``retention_days``.
   * - ``hot_cadence_sec``
     - ``10``
     - Seconds between samples inside the hot span.
   * - ``tail_cadence_sec``
     - ``60``
     - Seconds between samples outside it. Must be a whole multiple of
       ``hot_cadence_sec`` — the sparse tier is a subset of the dense grid, so a
       cadence that does not divide would put the two on timestamps that never
       coincide.
   * - ``recorder_cadence_sec``
     - ``10``
     - How often the recorder samples the live machine.
   * - ``recorder_tail_cadence_sec``
     - ``60``
     - How often one of those samples is additionally kept for the full
       retention span, so recorded history survives as the dense copy ages out.
       Same whole-multiple rule.
   * - ``recorder_poll_sec``
     - ``30``
     - How often the recorder re-reads the deployment's config to decide whether
       to record at all. It records only for a ``virtual_accelerator`` control
       system, and this is what lets that flip take effect without a restart.
   * - ``freshness_channel``
     - unset
     - Canary channel for a derived ``archiver_freshness`` health check. Unset
       derives no check (see :doc:`configure-health-checks`).
   * - ``host``
     - ``localhost``
     - Where the store is. **Required** when ``deploy_services`` is false: an
       attached project deploys no store of its own, so it has to name the host
       whose archive it reads.
   * - ``port_host``
     - ``27017``
     - Host port the store publishes on — or, for an attached project, the port
       the other host published.
   * - ``database`` / ``collection``
     - ``osprey_archiver`` / ``pv_history``
     - Where the samples live inside the store.
   * - ``compression``
     - ``zstd``
     - Block compressor for the collection: ``zstd``, ``snappy``, ``zlib`` or
       ``none``.
   * - ``username`` / ``auth_database``
     - ``osprey`` / ``admin``
     - The database user the deployment creates and the agent connects as, and
       the database it authenticates against.
   * - ``password_env``
     - ``MONGO_ROOT_PASSWORD``
     - **Name** of the variable holding that password. The value is minted into
       the deployment's ``.env``; it is never a profile field.
   * - ``timeout_sec``
     - ``5``
     - How long the connector waits to reach the store.

One fact, one home
------------------

The block is where the archive is described, and the build writes the rest from
it. Do **not** also spell these in ``config:`` — a profile that does is refused,
by name, rather than silently having one copy win:

- the connector's eight connection keys —
  ``archiver.mongodb_archiver.host``, ``.port``, ``.name``, ``.collection``,
  ``.auth``, ``.username``, ``.password_env``, ``.timeout`` — all derived from
  the keys above;
- the shape knobs, written to ``va_archiver.*`` in the rendered ``config.yml``
  for the seeder and the recorder to read;
- ``health.categories.archiver``, when ``freshness_channel`` is set.

Two homes for one fact are free to disagree, and the disagreement is the
dangerous case: a stale ``collection`` or ``host`` in ``config:`` points the
agent at an archive nothing is writing, which reads as empty rather than as
broken.

What the block does *not* do is select the archiver. Declaring where an archive
lives and choosing it as the deployment's archiver are separate decisions, so
the block never flips ``archiver.type`` out from under you — set
``config: {archiver.type: mongodb_archiver}`` yourself, or the project deploys a
store and then reads something else beside it.

.. warning::

   ``osprey build`` **refuses** a profile that pairs a ``virtual_accelerator``
   control system with the mock archiver, or with no ``archiver.type`` at all
   (which resolves to the mock): a simulated machine whose history is
   synthesized at read time reports a past that never happened, and nothing can
   catch it. The error names the fix — declare this block and select
   ``mongodb_archiver``, point the archiver at a store you run yourself, or set
   the control system to ``mock`` for an honestly storeless project. See
   :doc:`use-virtual-accelerator`.


Lifecycle commands
==================

Lifecycle commands run shell commands at three phases of the build:

- **pre_build** — before rendering (cwd: profile directory)
- **post_build** — after git init (cwd: project directory)
- **validate** — advisory checks that warn but don't abort (cwd: project directory)

.. code-block:: yaml

   lifecycle:
     pre_build:
       - name: "Check dependencies"
         run: "pip check"
     post_build:
       - name: "Build search index"
         run: "python scripts/build_index.py"
         cwd: "data"
         timeout: 300
         stream: true

Each step requires ``name`` and ``run``. Optional: ``cwd`` (relative to the phase
default), ``timeout`` (seconds, default 120), and ``stream`` (print output live;
also available for all steps via ``--stream``).

``{project_root}`` is replaced with the built project's absolute path. The
project venv's ``bin/`` is prepended to ``PATH``, so ``python`` and ``pytest``
resolve to the project's own Python.


Environment variables
=====================

The ``env`` section declares what the deployment needs. ``required`` documents
variables the operator must supply; secrets live in the repository's ``.env``
and never in the profile (see :ref:`profile-secrets`).

.. code-block:: yaml

   env:
     required:
       - API_KEY
       - DB_HOST
     defaults:
       LOG_LEVEL: info

Both lists are rendered into the repository's ``.env.example``, so an operator
opening that file sees them alongside every other variable. Required names must
match ``^[A-Z_][A-Z0-9_]*$``.

``defaults`` values are additionally seeded into the repository's ``.env`` by
``osprey init``, under their own section banner, so a deployment created from
the profile starts with them in force. Seeding is append-only: a value already
in the file — set by the operator, or minted by a deploy — always wins, and
later ``init`` runs never rewrite one. Declare a default only for a value the
profile's author can honestly choose for every deployment (the
``control-assistant`` preset's demo login passwords, say); values a *site*
should share across hosts belong in ``.env.shared``, which is committed with
the repository and read by every host (see :ref:`deployment-env-chain`).


Dependencies
============

``dependencies`` adds Python package specifiers to the built project. They are
installed into the project venv and recorded in its generated ``pyproject.toml``:

.. code-block:: yaml

   dependencies:
     - numpy>=1.24
     - pandas
     - scipy~=1.11

.. code-block:: bash

   cd my-project
   uv run osprey web     # uses my-project/.venv
   uv sync               # rebuilds it from pyproject.toml

Builds run with ``--skip-deps`` create no environment and no ``pyproject.toml``;
install dependencies yourself in that mode.

.. _profile-environment:

The execution environment
-------------------------

``dependencies`` says what *else* to install. The ``environment:`` block says
what the project environment is built *on top of* — which interpreter it starts
from, and, when that interpreter belongs to a virtual environment your facility
already maintains, which of its packages to carry over.

.. code-block:: yaml

   environment:
     python: /opt/facility/analysis-env/bin/python   # base interpreter
     packages:                                       # installed on top
       - lmfit>=1.3
     inherit_exclude:                                # left out of the freeze
       - facility-inhouse-tools

All three keys are optional; the block as a whole can be omitted.

``python``
   The base interpreter, as an absolute path. It may be a plain interpreter
   (``/usr/bin/python3.12``) or the interpreter inside a virtual environment —
   the syntax is the same. The build aborts if the path does not exist or is not
   executable.

``packages``
   Extra requirements installed into the project environment. Resolved in the
   same install as ``dependencies``, so the two cannot disagree; where both name
   the same distribution, a pinned version wins over a bare name, and between two
   pins ``packages`` wins.

``inherit_exclude``
   Distribution names to leave out of the freeze described below. Only meaningful
   with a virtual environment base; declaring it otherwise is rejected at
   validation time rather than silently ignored.

**Carrying a virtual environment's packages over.** Basing a project on a virtual
environment's *interpreter* does not inherit its *packages*. What carries them
over is a **freeze**: when ``environment.python`` names a virtual environment's
interpreter, the build records that environment's installed distributions as
exact ``name==version`` requirements in the project's ``pyproject.toml``. The
project venv — and any container image built from it — installs that same set.

A pin in ``dependencies`` or ``packages`` overrides the version the base
happened to carry.

The freeze runs **only when a base interpreter is declared**. Without
``environment.python`` the base is whatever interpreter OSPREY itself was
installed into — an accident, not a curated environment — and its packages are
deliberately not carried over.

**The build stops if a package cannot be reproduced.** Two cases are refused: a
distribution with no package-index coordinate (installed from a local path, a
VCS checkout, or a bare archive URL), and a version outside OSPREY's own
requirement for that package. Every offending package is named in a single
message, along with the ``inherit_exclude`` block that clears all of them.


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

   :doc:`../cli-reference/index`
       Complete CLI command reference

   :doc:`add-mcp-server`
       How to build custom MCP servers for OSPREY

   :doc:`deploy-project/index`
       Container deployment after building
