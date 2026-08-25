=============
CLI Reference
=============

Complete reference for all Osprey Framework CLI commands.

**Prerequisites:** Framework installed (``uv sync``)

Overview
========

All commands are accessed through the ``osprey`` command.

A deployment is a git repository: ``profile.yml`` at its root is the source you
edit, ``build/`` is what a build renders from it, and ``var/`` is durable state.
The lifecycle verbs find that repository by walking up from wherever you are
standing, so none of them has to be told where it is; ``--repo DIRECTORY``
names another one explicitly.

.. code-block:: bash

   osprey --version          # Show framework version
   osprey init               # Create a deployment repo from a preset
   osprey set KEY=VALUE      # Write a setting into profile.yml
   osprey validate           # Check profile.yml without building
   osprey build              # Render build/ from profile.yml
   osprey up                 # Start the deployment, as built
   osprey down               # Stop it, keeping all data
   osprey restart            # Stop and start it again
   osprey status             # Show what it is doing
   osprey logs               # Show its container logs
   osprey reset              # Wipe it back to a fresh state
   osprey config             # Show the deployment configuration
   osprey chat               # Talk to this deployment's agent
   osprey users              # Manage web-terminal users
   osprey feedback           # Read feedback sent from the web terminals
   osprey profile            # Validate and inspect build profiles
   osprey health             # Check system health
   osprey channel-finder     # Channel finder CLI
   osprey knowledge          # Facility knowledge bundles and graph corpus
   osprey eject              # Copy framework components for customization
   osprey ariel              # ARIEL logbook search service
   osprey artifacts          # Artifact gallery
   osprey web                # Launch web terminal
   osprey theme-lab          # Design and preview a theme in the browser
   osprey scaffold           # CI files, boot unit, build artifact overrides
   osprey audit              # Audit project or profile safety
   osprey skills             # Manage bundled Osprey skills
   osprey vendor             # Manage locally bundled vendor assets

Global Options
==============

``--version``
   Show framework version and exit.

``-v, --verbose``
   Show debug output, including every container command run. The transcript
   takes the place of the progress report rather than joining it, so under
   ``-v`` no phase or step lines are printed at all. Without it, a run prints
   the progress report and the command's own output only; log records below
   WARNING never reach the screen, though they still reach every sink the
   deployment configures.

``--help``
   Show help for any command (e.g., ``osprey build --help``).

osprey init
===========

Create a deployment repository from a bundled preset.

.. code-block:: bash

   osprey init [DIRECTORY] --preset NAME [OPTIONS]

``DIRECTORY`` is the repository the deployment lives in, and its name *is* the
deployment's name. Omit it to initialize the current directory in place, which
is how a repository cloned empty from a forge is filled in.

The repository holds four zones — source you edit, secrets, disposable build
output, and durable state:

.. code-block:: text

   DIRECTORY/
     profile.yml                    the manifest; everything the preset configures
     data/ personas/ triggers.yml   the material it names — yours to edit
     .env                           provider keys, seeded from your shell
     build/                         rendered by `osprey build`; gitignored
     var/                           agent memory and audit log; gitignored

``git init`` and an initial commit run at the end, unless a git repository
already encloses the target or ``--no-git`` is given.

``--preset NAME`` — Bundled preset to materialize.

``--list-presets`` — List bundled preset names and exit.

``-O, --override FILE`` — Layer a YAML file on top of the preset before writing
(repeatable, in order).

``--set KEY.PATH=VALUE`` — Inline scalar/list override baked into the emitted
profile (repeatable). RHS is parsed as YAML. Top-level shorthands: ``provider``,
``model``, ``channel_finder_mode``, ``connector``.

``--force`` — Re-materialize the source zone of an existing deployment
repository, discarding edits to ``profile.yml``, ``data/``, ``personas/``,
``triggers.yml``, ``web-terminal-context/``, and ``.env.example``. Never touches
``.env``, ``.git``, ``var/``, ``build/``, ``.gitignore``, ``README.md``,
``ci-extra.yml``, the CI file, or ``scripts/verify.sh``.

``--no-git`` — Skip ``git init`` and the initial commit.

``--up`` / ``-d, --detach`` / ``--dev`` — Build and start the deployment right
away, optionally in the background or in development mode.

.. code-block:: bash

   osprey init --list-presets
   osprey init als-assistant --preset control-assistant
   osprey init demo --preset control-assistant --up -d --dev

osprey set
==========

Write settings into the deployment profile.

.. code-block:: bash

   osprey set KEY=VALUE... [--repo DIRECTORY]

Each ``KEY=VALUE`` is written into this repository's ``profile.yml`` in place,
comments intact. That file is the source of truth, so this is the only command
that edits configuration for you — the rendered ``build/config.yml`` is
generated from it and is never hand-edited. Run ``osprey build`` to carry a
setting through to ``build/``, then ``osprey up`` to deploy it.

``KEY`` is a top-level profile key (``provider``, ``model``, ``tier``,
``channel_finder_mode``, ``connector``) or a dotted path. Keys under ``config.``
address the rendered config: ``config.control_system.type=epics`` writes that
literal dotted entry into the profile's ``config:`` block. ``VALUE`` is read as
YAML, so ``true``/``false`` become booleans and bare numbers become numbers.

``channel_finder_mode`` names the channel-finder paradigm the build renders:
``in_context``, ``hierarchical``, ``middle_layer`` or ``graph``. The first three
search a channel database file; ``graph`` searches the graph store named by
``services.graphdb`` -- deployed with the stack or facility-hosted -- instead of
a database file.

Two shorthands stand in for longer key paths: ``connector=`` writes
``config.control_system.type``, and ``epics_gateway=`` writes a known facility's
EPICS gateway addresses. (Control systems beyond the bundled ones are reachable
through custom connector packages — see :doc:`/how-to/add-connector`.)

.. code-block:: bash

   osprey set model=sonnet
   osprey set connector=epics
   osprey set tier=1 channel_finder_mode=in_context
   osprey set config.facility.name='ALS Storage Ring'
   osprey set epics_gateway=als
   osprey set --repo ~/als-assistant config.control_system.writes_enabled=true

osprey validate
===============

Check the deployment profile without building.

.. code-block:: bash

   osprey validate [TARGET] [--repo DIRECTORY]

With no argument, validates the deployment repository enclosing the working
directory. ``TARGET`` names a different profile to check instead — a persona
delta file under ``personas/``, or a directory holding a ``profile.yml``.

Resolves ``extends:`` chains and runs the full consistency check — convention
directories, the ``data:`` tree, service templates, lifecycle steps, env vars —
then lints the declared web stack against the config a build would render. Every
problem found is reported, not just the first. Exits 0 when the profile is
valid, 2 with the accumulated errors when it is not, so a CI job can gate on it.

.. code-block:: bash

   osprey validate
   osprey validate personas/reader.yml
   osprey validate --repo ~/als-assistant

osprey config
=============

Show the deployment configuration.

.. code-block:: bash

   osprey config [--rendered | --defaults] [--repo DIRECTORY]

With no flag, prints the source ``profile.yml`` — the tracked, hand-edited
manifest — exactly as it is on disk, comments included. Output is piped through
unchanged when stdout is not a terminal. This command only reads; to change a
setting, use ``osprey set``.

``--rendered`` — Show the built ``config.yml`` the deployment actually runs on.

``--defaults`` — Show the framework's default template, with every key the
framework understands and its default. Needs no deployment repository.

.. code-block:: bash

   osprey config
   osprey config --rendered
   osprey config --defaults > defaults.yml

osprey profile
==============

Validate and inspect build profiles. The profile is the durable, facility-owned
source a deployment is built from — see :doc:`/how-to/build-profiles`.

.. code-block:: bash

   osprey profile validate TARGET
   osprey profile presets
   osprey profile artifacts

``osprey profile validate TARGET``
   Check a profile without building anything. ``TARGET`` is a directory holding
   a ``profile.yml`` or a path to a profile file. Resolves ``extends:`` chains
   and reports every problem found — convention directories, the ``data:``
   tree, service templates, lifecycle steps, env vars. Exits 0 when valid, 2
   with the accumulated errors when not. Inside a deployment repository, plain
   ``osprey validate`` checks that repository without naming a target.

``osprey profile presets``
   List bundled preset names, one per line. Every name printed is usable as
   ``--preset NAME`` for ``osprey init``.

``osprey profile artifacts``
   List every artifact the six profile lists can name — hooks, rules, skills,
   agents, output styles, and web panels — with a one-line description of
   each. The same menu appears as commented entries in an emitted profile.

.. code-block:: bash

   osprey profile presets
   osprey init my-facility --preset control-assistant --set model=opus
   cd my-facility
   osprey validate
   osprey build

osprey build
============

Render this deployment repository's ``build/`` from its profile.

.. code-block:: bash

   osprey build [OPTIONS]

Run it with no arguments, anywhere inside a deployment repository. It walks up
to the repository's ``profile.yml`` and renders the whole output zone from it:
``config.yml``, the Osprey agent artifacts, the data tree, the service templates
and the compose files that deploy them.

``build/`` is derived in full and holds nothing durable — your keys are in
``.env``, the agent's memory is in ``var/`` — so every build wipes and
re-renders it. The render lands in ``build/.tmp`` and replaces ``build/`` only
once it has succeeded, so a build that fails, or one you interrupt, leaves the
previous build exactly as it was and still able to stop the stack it started.

It renders files, never containers: rebuild while the stack is up and the change
takes effect at the next ``osprey up`` or ``osprey restart``.

``-s, --stream`` — Stream lifecycle step output in real time.

``--skip-lifecycle`` — Skip the profile's ``pre_build``, ``post_build``, and
``validate`` steps.

``--skip-deps`` — Skip venv creation and dependency installation (CI mode).

``--runtime-root PATH`` — Override ``project_root`` in the rendered config, for
a build whose output runs somewhere other than where it was made.

``--repo DIRECTORY`` — Deployment repository to act on (default: the nearest
``profile.yml`` at or above the working directory).

.. code-block:: bash

   osprey build
   osprey build --repo ~/deployments/als-assistant
   osprey build --skip-lifecycle --skip-deps          # CI: no venv, no hooks

Lifecycle verbs
===============

``osprey up``, ``down``, ``restart``, ``status``, ``logs`` and ``reset`` run the
deployment's container stack. Each one takes ``--repo DIRECTORY`` and otherwise
needs no arguments: run it anywhere inside the deployment repository.

osprey up
---------

Start this deployment from ``build/``, as built.

It starts what the last ``osprey build`` rendered and re-renders nothing from
``profile.yml``, so the services that come up are always the ones you can read
on disk. It reads ``profile.yml`` for one thing: a fingerprint. If the profile
has changed since the build, ``up`` refuses and says what moved, because
starting would deploy something other than what the profile now describes.

One exception, by design: a deployment with web terminals re-renders that stack
at every start — its compose file, nginx config, landing page, and any persona
whose project is missing. Those follow the user roster rather than the build, so
a roster edit takes effect on the next start.

Whether the deployment is reachable off-host is a property of the build, not of
this command: the bind address is rendered into every published port. Change it
with ``osprey set deployment.bind_address=0.0.0.0``, then rebuild.

``-d, --detached`` — Run services in the background.

``--dev`` — Bake the local osprey checkout into the images instead of the
published release.

``--build`` — Re-render ``build/`` from ``profile.yml`` first, then start it.

``--as-built`` — Start ``build/`` as it was rendered, even though ``profile.yml``
has moved on.

``--keep-archiver-base`` — Keep the existing archiver history even when the
profile's retention/cadence knobs no longer match it.

osprey down
-----------

Stop this deployment, keeping all data. It renders nothing; destroying data is
``osprey reset``, which asks first.

If ``build/`` is gone or was never rendered, ``down`` does not re-derive the
compose files from ``profile.yml`` — those describe what would be started now,
not what is running. Instead it stops the containers this repository labelled as
its own, which is the recovery path for a ``build/`` deleted while the stack was
up. Containers are labelled when they are *created*, so a stack started before
this labelling existed cannot be found that way; run ``osprey build`` to restore
``build/`` and ``down`` works normally again.

osprey restart
--------------

Stop and start this deployment again. Takes the same options as ``up``:
``-d/--detached``, ``--dev``, ``--build``, ``--as-built``,
``--keep-archiver-base``.

osprey status
-------------

Show what this deployment is doing. It reads and reports — it starts nothing,
stops nothing and renders nothing — so it is safe to run against a live stack at
any time.

Four sections. **Build** says whether ``build/`` still matches ``profile.yml``,
the same check ``osprey up`` refuses on, and which version of osprey rendered it.
**Containers** is what the container runtime reports, not what compose thinks
should exist. **Endpoints** is where the services are declared to answer.
**Agent** is the provider, whether its credential can be found, and whether the
rendered agent files still match the config.

``--agents`` — Also show the per-subagent model assignments.

osprey logs
-----------

Show this deployment's container logs, and step out of the way: ``-f`` streams,
Ctrl-C stops it, piping into other commands works, and the exit code is the
runtime's own. Both halves of a deployment are covered — the services and, when
there is one, the web-terminal stack.

``-f, --follow`` — Keep streaming new output until interrupted.

``--tail INTEGER`` — Show only the last N lines per container.

.. code-block:: bash

   osprey logs
   osprey logs event-dispatcher -f
   osprey logs --tail 50

osprey reset
------------

Wipe this deployment back to a fresh state. It stops the stack, removes the
containers, volumes and images carrying this checkout's identity, destroys the
agent's memory under ``var/agent_data/``, strips the tokens ``osprey up`` minted
out of ``.env``, and deletes ``build/``.

Two things survive, and the plan says so before you confirm: ``var/audit/``, the
safety audit log, and everything in ``.env`` that a deploy did not mint — your
provider keys. The source zone is never touched: ``profile.yml``, ``data/``,
``personas/``, ``triggers.yml`` and ``.git`` are exactly what they were.

Reset removes a container or volume only when it also carries this repository's
identity label, and refuses outright — removing nothing — when it finds
same-named resources created from a different path.

``--dry-run`` — Show the removal plan and stop. Nothing is stopped, removed, or
written.

``-y, --yes`` — Skip the typed confirmation. The plan is still printed.

``--purge-audit`` — Destroy ``var/audit/`` as well. It is kept by default.

Exit status: ``0`` the reset did what it said (or there was nothing to do),
``1`` you declined or it refused, ``3`` it ran but the deployment is only partly
reset, with each survivor named above the exit.

.. code-block:: bash

   osprey up -d
   osprey status
   osprey up --build -d
   osprey restart --dev
   osprey logs -f
   osprey down
   osprey reset --dry-run

osprey users
============

Manage this repository's web-terminal users. A multi-user deployment gives each
person on the roster their own web terminal, container and workspace volumes;
these verbs act on that roster, which lives in the profile. Every verb takes
``--repo DIRECTORY``.

.. list-table::
   :header-rows: 1
   :widths: 22 46 32

   * - Verb
     - What it does
     - Also accepts
   * - ``remove USER``
     - Retire one person: remove their web-terminal workspace.
     - ``--archive``, ``--purge``, ``-y/--yes``
   * - ``prune``
     - Remove workspaces for people no longer on the roster.
     - ``--archive``, ``--purge``, ``-y/--yes``, ``--dry-run``
   * - ``seed [USER]``
     - (Re)seed workspaces from the roster; ``USER`` targets one person, omit
       to reseed all.
     - —
   * - ``passwd USER``
     - Change one user's login password (password authentication only).
       Prompts without echoing, and ends that user's sessions.
     - —
   * - ``login-url USER``
     - Print the URL that opens that user's terminal. Only the URL goes to
       stdout, so it can be piped or copied. Refuses for a user who signs in
       through the login page.
     - —
   * - ``env``
     - Render ``.env.users``, the env file every per-user container runs
       with.
     - ``--env-file``, ``-o/--output``

``--archive`` -- Archive a user's workspace before removing it (mutually exclusive with ``--purge``).

``--purge`` -- Permanently delete a user's workspace without archiving it (mutually exclusive with ``--archive``).

``-y, --yes`` -- Assume yes to confirmation prompts.

``--dry-run`` -- Show what would happen without making changes.

``osprey users login-url`` builds the URL from that user's operator secret in
the repository's ``.env``, which ``osprey up`` mints for every roster user in
every auth mode. Opening it once trades the token for a session cookie. It is
how someone gets in when nginx authenticates nobody — ``auth.method: none``, or
a roster entry with ``login: false`` — and it is a password: send each person
only their own. Rotate one by deleting that user's ``OSPREY_TERMINAL_SECRET_*``
line from ``.env`` and running ``osprey up`` again.

For a user who *is* behind the login wall the command refuses and names their
login page instead. The URL would not work for them — the authentication
service turns the request away before the terminal can read the token — and
printing it would put a live credential in someone's inbox for nothing.

``osprey users env`` renders the same subset a deploy would generate, from the
same two inputs — the rendered deploy config and the repository root's env
chain (``.env.shared`` then ``.env``, so ``.env`` wins on a key both set) — so
a file rendered here and one generated by ``osprey up`` cannot disagree. Values
come only from those files, never from the surrounding environment.
``--env-file PATH`` renders from that one file instead of the chain.
``-o/--output PATH`` writes the result at mode ``0600`` instead of to stdout;
unlike a deploy, which never overwrites an existing ``.env.users``, an
explicit ``--output`` is taken as an instruction and replaces what is there. In
CI, pass it: without ``--output`` the assembled secrets go to the job log.

.. code-block:: bash

   osprey users remove alice --archive
   osprey users prune --dry-run
   osprey users seed alice
   osprey users passwd alice
   osprey users login-url alice
   osprey users env --output .env.users

See :doc:`/how-to/deploy-a-facility` for the walkthrough that uses these
verbs end to end.

osprey feedback
===============

Read the feedback this deployment's web terminals collected. People using a web
terminal can send a report from the terminal itself; each submission is stored
with the session it was sent from, in the sender's own workspace, and these
verbs read it back. Every verb takes ``--repo DIRECTORY``.

A multi-user deployment keeps one store per person, and reading it starts a
short-lived read-only container on each person's workspace — so the
deployment's container runtime has to be up. A single-user deployment keeps one
store on this machine, read straight off disk.

.. list-table::
   :header-rows: 1
   :widths: 22 46 32

   * - Verb
     - What it does
     - Also accepts
   * - ``list``
     - One line per submission, newest first: id, who sent it, when (UTC),
       which channel it went out on, and the start of the message. Headers
       only.
     - —
   * - ``export``
     - Every submission's whole record plus the session captured with it, as
       one JSON array. Written a record at a time, so a large store never has
       to fit in memory.
     - ``-o/--output``

``-o, --output PATH`` -- Write the export to this file instead of stdout. Naming
a directory, or a file inside a directory that does not exist, is a usage error
at parse time, before any workspace is read.

With no ``--output``, ``export`` puts the JSON document on stdout and nothing
else, so it can be redirected straight into a file; errors and the closing
summary go to standard error. With ``--output``, the human output stays on the
terminal and the file holds the export. An empty store exports ``[]``.

A workspace that cannot be read is reported and the rest of the deployment is
still listed or exported; the export's closing summary distinguishes a workspace
that contributed nothing from one that was only partly readable. If nothing at
all could be read, both verbs stop rather than reporting a deployment with no
feedback in it. A submission whose session was dropped to keep the store inside
``web.feedback.max_store_bytes`` carries ``context_pruned``; one missing its
session for any other reason carries ``context_missing``. The submission is
exported either way.

Exit status: ``0`` on success, including a deployment nobody has sent feedback
from. Non-zero when the repository has no build, when every workspace failed to
read, and when an export stopped part-way — in which case the file still parses
and holds what was read.

``agent_data.base_dir`` is checked on the **single-user** path only, and a
``~``-relative value there is non-zero with the explanation (the data lands
where the verb cannot reach it). The multi-user path never consults the key: it
mounts each person's volume and reads a fixed path inside it, so a rostered
deployment with a ``~``-relative ``agent_data.base_dir`` writes its feedback to
a volume these verbs do not mount and simply reads as having none. Nothing
detects that, so do not read an empty multi-user listing as proof. See
:doc:`/how-to/send-feedback`.

.. code-block:: bash

   osprey feedback list
   osprey feedback list --repo ../other-deployment
   osprey feedback export --output feedback.json
   osprey feedback export > feedback.json

See :doc:`/how-to/send-feedback` for the dialog these records come from and what
each submission carries.

osprey health
=============

Run comprehensive system health check.

.. code-block:: bash

   osprey health [OPTIONS]

``--project DIRECTORY`` -- Deployment repository or rendered project directory (default: the repository enclosing the current directory).

``-v, --verbose`` -- Show per-warning and per-error details in the summary.

``--json`` -- Emit the report as a single JSON document on stdout.

``--category NAME`` -- Run only the named category (repeatable).

``--full`` -- Also run on-demand categories (live model chat, pinned CLI download).

Exit codes: ``0`` healthy, ``1`` warnings only, ``2`` one or more errors,
``3`` the command itself failed, ``130`` interrupted. See
:doc:`/how-to/health-json-contract` for the ``--json`` document's shape and the
``jq`` patterns for consuming it, and :doc:`/how-to/configure-health-checks` for
the ``health:`` config block.

osprey chat
===========

Talk to this deployment's agent. See :doc:`/how-to/cli-agent`.

.. code-block:: bash

   osprey chat [PROMPT] [OPTIONS]

Starts the agent in the deployment's ``build/`` directory, wired to the
provider, control system and facility knowledge that build was rendered with.
``PROMPT``, when given, is the opening message — with ``--print`` it is answered
and the command exits, which is the shape a script wants.

Nothing is re-rendered: ``osprey build`` owns that. When the profile has changed
since the last build, a warning says so and the session starts anyway against
the build as it stands.

``--resume SESSION_ID`` — Resume a previous agent session by ID.

``--print`` — Print the answer and exit.

``--effort [low|medium|high|max]`` — Reasoning effort (default:
``claude_code.effort`` from the build).

``--no-pin`` — Ignore ``claude_code.cli_version`` and use the installed agent CLI.

``--repo DIRECTORY`` — Deployment repository to act on.

.. code-block:: bash

   osprey chat
   osprey chat --print "what is the stored beam current?"
   osprey chat --resume abc123
   osprey chat --repo ~/als-assistant

osprey eject
============

Copy framework services to your project for customization.

``osprey eject list``
   List all ejectable framework capabilities and services.

``osprey eject service NAME [--output PATH] [--include-tests]``
   Copy a framework service directory locally.

.. code-block:: bash

   osprey eject list
   osprey eject service channel_finder --include-tests

osprey channel-finder
=====================

Tools for building, validating, previewing, and serving control system
channel databases.

Options: ``--project PATH``, ``-v, --verbose``

``osprey channel-finder build-database``
   Build a channel database from a CSV file.

``osprey channel-finder validate [--database PATH] [--pipeline hierarchical|in_context|middle_layer] [-v]``
   Validate a channel database JSON file. The paradigm is auto-detected from the
   project's config; ``--pipeline`` overrides that.

``osprey channel-finder preview``
   Preview a channel database with flexible display options.

``osprey channel-finder generate [--output-dir DIR] [--source PATH] [--format in_context|hierarchical|middle_layer|all] [--tier 1|3|none] [--validate]``
   Generate channel databases from a hierarchical template. Produces one
   or more pipeline formats (default: all three) with optional tier filtering.

``osprey channel-finder benchmark --model PROVIDER/WIRE_ID [--queries SPEC] [--runs-per-query N] [--concurrency N] [--output-dir DIR] [--queries-path PATH] [-v]``
   Run the benchmark harness against a channel-finder pipeline using a
   LiteLLM-form model id (e.g. ``anthropic/claude-haiku-4-5``). Saves per-run
   JSON results for accuracy/cost analysis.

``osprey channel-finder web``
   Launch the Channel Finder web interface.

A graph-mode project has no channel database file: ``validate`` and ``preview``
say so and point at ``osprey knowledge seed-graph`` and the ``read_cypher`` tool,
and ``--pipeline`` does not offer ``graph`` for the same reason.

.. code-block:: bash

   osprey channel-finder build-database
   osprey channel-finder validate
   osprey channel-finder preview
   osprey channel-finder generate --format hierarchical
   osprey channel-finder benchmark --model anthropic/claude-haiku-4-5
   osprey channel-finder web

osprey knowledge
================

Build and load the facility's knowledge material: the OKF bundle of concept
documents, and the NARAD-convention corpus the graph store holds. An omitted
``BUNDLE`` comes from ``facility_knowledge.bundle_path`` and an omitted ``TTL``
from ``services.graphdb.ttl_path``. See :doc:`/how-to/okf-bundle` and
:doc:`/how-to/use-facility-graph`.

``osprey knowledge regen-index [BUNDLE]``
   Regenerate the ``index.md`` files throughout an OKF bundle. Run it twice and
   the second run changes nothing.

``osprey knowledge validate [BUNDLE]``
   Check every document in a bundle against the OKF format.

``osprey knowledge seed-from-ttl TTL BUNDLE [--force]``
   Write stub concept documents into a bundle from a TTL corpus, one per class,
   for a person to fill in.

``osprey knowledge compile-ontology SOURCE OUTPUT [--check]``
   Compile an authored LinkML schema into the ontology table ``build-ttl``
   reads. ``SOURCE`` is the schema, and ``OUTPUT`` the JSON table to write; an
   existing file is overwritten. The output is deterministic, so compiling an
   unchanged schema twice leaves ``git diff`` silent.

   ``--check`` writes nothing. It compares ``OUTPUT`` against a fresh compile of
   ``SOURCE`` and exits non-zero when the two disagree, naming the classes,
   synonyms and families that differ --- which is what a CI job or a pre-commit
   hook runs to prove a committed table still matches its schema.

``osprey knowledge build-ttl OUTPUT [--channel-db PATH] [--descriptions PATH] [--limits PATH] [--ontology PATH]``
   Derive a NARAD-convention TTL corpus from the channel database: one device
   node per device, one binding per address, a read or write direction on every
   signal, and the descriptions that let the corpus be searched by meaning
   rather than by address alone.

   ``--channel-db`` is the hierarchical database, and defaults to the one
   ``channel_finder.pipelines.hierarchical.database.path`` names.
   ``--descriptions`` is the in-context database of the same machine, whose
   per-channel text becomes each binding's description; unnamed, it is looked
   for as ``in_context.json`` beside the file ``--channel-db`` named, which is
   how the OSPREY source tree keeps the two. A rendered project holds only the
   paradigm it runs, so name both there. ``--limits`` decides which signals are
   written and ``--ontology`` maps device families to classes; both fall back to
   the config and the shipped table. :doc:`/how-to/use-facility-graph` has the
   detail.

   A graph-mode project ships no channel database by design, so the command
   refuses there and says to name the sources with ``--channel-db`` and
   ``--descriptions``.

``osprey knowledge seed-graph [TTL] [--force]``
   Load a TTL corpus into the deployed graph store. ``--force`` wipes the store
   and imports from scratch, which is what a changed store configuration needs.

.. code-block:: bash

   osprey knowledge build-ttl data/demo_machine.ttl
   osprey knowledge seed-graph data/demo_machine.ttl
   osprey knowledge validate

osprey ariel
============

Manage the ARIEL logbook search service.

``quickstart [--source PATH]`` -- Full setup: migrate and ingest demo data.

``status [--json]`` -- Show service status.

``migrate`` -- Create or update database tables.

``sync [--limit N]`` -- Idempotent migrate + incremental ingest + enhance.
Safe to run on every build; on a fresh database, runs a full ingest.

``ingest --source PATH [--adapter TYPE] [--since DATE] [--limit N] [--dry-run]``
   Ingest logbook entries from file or URL.

``watch [--source] [--once] [--interval N] [--dry-run]`` -- Poll for new entries.

``enhance [--module NAME] [--force] [--limit N]`` -- Run enhancement modules.

``qmd-resync [--rebuild]``
   Re-export entries the qmd markdown mirror never saw --- entries created in
   the web interface, attachment uploads, and entries written through the
   logbook write service. ``--rebuild`` wipes the mirror and re-exports
   everything, which is what to run after ``purge``.
   ``ingest`` and ``watch`` already do this pass on their own.

``models`` -- List embedding models and tables.

``search QUERY [--mode keyword|semantic|hybrid] [--limit N] [--json]``
   Execute a search query. Without ``--mode``, the deployment's
   ``ariel.default_search_mode`` decides.

``vocab-check [PATH] [--json]``
   Validate a facility vocabulary file --- the shorthand-to-prose mapping that
   query expansion uses. Checks ``PATH``, or the file named by
   ``ariel.vocabulary.path`` when no path is given. Needs no database. Exits 1
   and lists every error when the file is broken; warnings never fail it.

``reembed --model NAME --dimension N [--batch-size N] [--force]``
   Re-embed entries with a different model.

``web [--port N] [--host ADDR] [--reload]`` -- Launch web interface.

``purge [--yes] [--embeddings-only]`` -- Delete all ARIEL data.

.. code-block:: bash

   osprey ariel quickstart
   osprey ariel search "RF cavity fault"
   osprey ariel vocab-check data/ariel/vocabulary.yml
   osprey ariel web --port 8080

osprey artifacts
================

Manage the OSPREY Artifact Gallery -- a local web gallery that displays
interactive plots, tables, and other outputs produced by the Osprey agent during
analysis sessions. Artifacts are written by the Osprey agent via ``save_artifact()`` in
``osprey execute`` or the ``artifact_register`` MCP tool.

``osprey artifacts web [OPTIONS]``
   Launch the Artifact Gallery web interface. Starts a FastAPI server on
   ``http://127.0.0.1:8086`` by default.

   ``-p, --port INTEGER`` — Port (default: from ``config.yml`` or ``8086``).

   ``-h, --host TEXT`` — Host to bind to (default: from ``config.yml`` or
   ``127.0.0.1``).

   ``--reload`` — Enable auto-reload for development.

.. code-block:: bash

   osprey artifacts web                    # Start on localhost:8086
   osprey artifacts web --port 9000        # Custom port
   osprey artifacts web --host 0.0.0.0     # Bind to all interfaces
   osprey artifacts web --reload           # Development mode

osprey web
==========

Launch the Web Terminal interface. See :doc:`/how-to/web-terminal/operate`.

``osprey web [OPTIONS]``
   Start the web terminal server (default: ``http://127.0.0.1:8087``). On
   startup it prints a login URL (``…/?token=…``) that sets a session cookie
   and then redirects to the clean address. The URL is printed once, but its
   token is the server's own secret and stays valid for the life of the
   process — treat it like a password.

   ``-p, --port INTEGER`` — Port (default: from config or 8087).

   ``--host TEXT`` — Host to bind to (default: ``127.0.0.1``).

   ``--shell TEXT`` — Shell command to run (default: ``claude``).

   ``--repo PATH`` — Deployment repo to act on (default: nearest ``profile.yml`` at or above cwd).

   ``--detach`` — Run in background (PID written to ``var/osprey-web.pid``). The
   login token is held only in memory; if the printed URL is lost, stop and
   restart the server to mint a new one.

   ``--reload`` — Auto-reload for development.

``osprey web stop``
   Stop a background web terminal server.

.. code-block:: bash

   osprey web
   osprey web --port 9000 --host 0.0.0.0
   osprey web --detach
   osprey web stop

osprey theme-lab
================

Design a theme in the browser. Starts a local server for OSPREY's design
system and opens the Theme Lab, where you pick an accent color and see it
previewed live on dark and light mock-ups of the web terminal, with contrast
badges that update as you go. Copying the export block gives you a
ready-to-paste description of the theme to request; the lab itself does not
write theme files. See :doc:`/how-to/web-terminal/theming`.

``osprey theme-lab [OPTIONS]``
   Serve the Theme Lab and open it. The URL is printed as well, so the page can
   be opened by hand if no browser appears. It is a login URL at the server root
   (``…/?token=…``): opening it sets a session cookie and then redirects to the
   lab itself.

   ``-p, --port INTEGER`` — Port to serve on (default: an unused port chosen
   automatically).

   ``--no-browser`` — Do not open a browser window; print the URL only.

.. code-block:: bash

   osprey theme-lab
   osprey theme-lab --port 9000
   osprey theme-lab --no-browser

osprey audit
============

Audit a build profile or project directory for safety risks. Uses an AI
reviewer to analyze permissions, hooks, MCP server configs, convention
directories, and lifecycle scripts.

.. code-block:: bash

   osprey audit TARGET [OPTIONS]

``--build`` — Build a profile in a temp directory, then audit the result.

``--model TEXT`` — Model for the reviewer agent.

``--budget FLOAT`` — Maximum budget in USD.

``-v, --verbose`` — Show verbose output.

``--json`` — Output as JSON.

.. code-block:: bash

   osprey audit my-project/
   osprey audit profile.yml --build
   osprey audit project/ --json

osprey scaffold
===============

Emit the repository's CI files and boot unit, and manage build artifact
ownership.
Framework-managed build artifacts (agents, rules, etc.) can be claimed
per-facility for in-place editing. A claim moves the artifact out of the build
zone and into the profile beside it; the next build copies it back and registers
it as user-owned, so a rebuild leaves your version alone.

All subcommands accept a common flag:

``--repo DIRECTORY`` — Deployment repository to act on (default: the nearest
``profile.yml`` at or above the working directory).

``osprey scaffold ci [--force]``
   Emit this repository's CI pipeline and health check from the profile's
   ``deploy:`` block: the pipeline at the repository root and the post-deploy
   health check at ``scripts/verify.sh``. Run it again whenever the block
   changes. Re-running is safe — a file whose content already matches is left
   untouched, stamp included, so an OSPREY upgrade alone produces no diff. A
   file the scaffolder did not write is reported and left alone unless
   ``--force`` is given. ``ci-extra.yml`` is never touched: it is yours, and the
   pipeline includes it.

``osprey scaffold systemd [--force]``
   Emit a systemd user unit that starts this deployment at boot: ``osprey.service``
   at the repository root, plus the commands to install it. The unit runs
   ``osprey up -d`` from this repository and ``osprey down`` on stop, with both
   the repository and the ``osprey`` program written in as full paths — systemd
   starts a unit with no working directory and a short ``PATH``. Run it on the
   machine that will run the deployment, and again after the repository moves or
   OSPREY is reinstalled elsewhere. Re-running is safe on the same terms as
   ``ci``: a matching file is left untouched, and a file the scaffolder did not
   write needs ``--force``. Refuses when no ``osprey`` program can be found to
   name. Starting at boot also needs ``loginctl enable-linger $USER`` once —
   see :doc:`../how-to/deploy-a-facility`.

``osprey scaffold list``
   List all build artifacts and their ownership status (framework vs.
   user-owned).

``osprey scaffold claim NAME``
   Move an artifact out of ``build/`` and into the repository's profile, into
   the convention directory for its kind (``rules/safety.md``,
   ``skills/orbit-check/``, ``services/postgresql/``, ``hooks/my-guard``). A
   file moves as a file; skills and services move as whole directories. The
   build copy is *moved*, not copied — it lives in one place until the next
   ``osprey build`` renders it again.

   Refused, with the reason: a **generated** artifact rather than an authored one —
   ``CLAUDE.md``, ``.claude/settings.json``, ``.mcp.json``,
   ``hook_config.json`` — where the message names the config key that does
   control it; and a profile slot that is already occupied. See
   :ref:`profile-claim`.

``osprey scaffold diff NAME``
   Show a unified diff between the current framework template (re-rendered)
   and your file at the canonical output path. For a claimed service
   directory, diffs every file in the directory against the packaged
   template.

``osprey scaffold unclaim NAME``
   Release ownership and restore framework management. The next build overwrites
   the file with the framework template. Ownership a build derived from the
   profile is re-registered by the next build, so this holds only until then —
   give the artifact up for good by deleting it from the profile's convention
   directory.

``osprey scaffold web-terminals lint [--repo PATH]``
   Validate the deployment's ``modules.web_terminals`` stanza (port-family
   allocation, reserved service names, duplicate users, persona references).
   Exits non-zero on error-severity findings; warnings do not fail the check,
   so it is safe to wire into a CI gate.

``osprey scaffold web-terminals render [--repo PATH] -o DIRECTORY``
   Render the multi-user deployment artifacts (docker-compose overlay, nginx
   routing fragment, static landing page) into ``-o/--output``. Lints first by
   default and aborts on errors; ``--no-lint`` skips the pre-check.

   Both verbs read the stanza from the repository's built ``config.yml``.

.. code-block:: bash

   osprey scaffold ci                             # Re-emit the CI files
   osprey scaffold systemd                        # Write the boot unit
   osprey scaffold list                           # Show all artifacts
   osprey scaffold claim agents/channel-finder    # Claim for editing
   osprey scaffold claim services/postgresql      # Freeze a service template
   osprey scaffold diff agents/channel-finder     # Compare yours vs framework
   osprey scaffold unclaim rules/safety           # Restore framework management
   osprey scaffold web-terminals lint             # lint this project's stanza
   osprey scaffold web-terminals render -o deploy/

osprey skills
=============

Manage bundled Osprey skills — agent skills shipped with OSPREY that
can be installed either globally or into a specific project's
``.claude/skills/`` directory.

``osprey skills install NAME [--target PATH]``
   Install a bundled skill into ``<target>/<name>/`` (defaults to
   ``~/.claude/skills/<name>/``). If the target already exists and is
   non-empty, the prior content is renamed to
   ``<name>.bak.<YYYYMMDD-HHMMSS>/`` before the new copy is written, so a
   previous version is never lost.

   ``--target PATH`` — directory to install into. Tilde is expanded. Use a
   project-local ``.claude/skills/`` path to scope the skill to one repo
   (e.g., a facility repository's ``.claude/skills/``). Omit for the global
   install.

   Currently supported skills:

   * ``osprey-build-interview`` — set up or migrate an OSPREY deployment
     through a guided conversation anchored on the live deployment repo (see
     :doc:`/getting-started/osprey-build-interview`). Typically installed
     globally so it is available in any Osprey agent session.
   * ``osprey-contribute`` — walks a contributor through the GitHub Flow
     journey from a working-tree change to a merged PR on ``main`` (branching,
     atomic commits, push, PR, rebase, merge).
   * ``osprey-pre-commit`` — runs the quick / ci / premerge check scripts at
     the right gate before committing, pushing, or opening a PR.
   * ``osprey-release`` — cuts a CalVer release: opens the version-bump PR,
     tags the merge commit, and verifies the automated PyPI publish.
   * ``osprey-design-philosophy`` — OSPREY's design and architecture principles
     for designing, adding, or reviewing a feature. Useful for framework
     contributors; install globally to have it available when working on
     ``src/osprey`` in any session.
   * ``creating-an-osprey-panel`` — author a themed, token-only web-terminal
     panel.

.. code-block:: bash

   osprey skills install osprey-build-interview
   osprey skills install creating-an-osprey-panel --target .claude/skills/

osprey vendor
=============

Manage locally bundled vendor assets (JS/CSS/fonts) for firewalled
deployments. By default OSPREY interfaces load third-party libraries directly
from CDN; set ``OSPREY_OFFLINE=1`` (or ``offline: true`` in ``config.yml``) to
switch the interfaces over to local bundles.

``osprey vendor fetch [OPTIONS]``
   Download all vendor assets declared in the manifest into
   ``static/vendor/``. Run once on firewalled deployments before starting
   ``osprey web`` with ``OSPREY_OFFLINE=1``. In default CDN mode this command
   is optional.

   ``-q, --quiet`` — Suppress per-file output.

   Behind a proxy that re-signs TLS with a site CA (e.g. Squid), the fetch
   fails cert verification until it is pointed at the site's CA bundle.
   That is the first remedy to reach for — verification stays on:

   .. code-block:: bash

      OSPREY_CA_BUNDLE=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem \
        osprey vendor fetch

   ``SSL_CERT_FILE`` works too (Python's default SSL context honors it);
   ``OSPREY_CA_BUNDLE`` wins when both are set. The path above is the
   RHEL-family system bundle — on Debian-family hosts it is
   ``/etc/ssl/certs/ca-certificates.crt``.

   ``-k, --insecure`` — The fallback when no CA bundle is at hand: skip TLS
   cert verification. Every asset is still checked against its manifest
   SHA256, so this is safe behind corporate proxies that intercept TLS.
   Also enabled via ``OSPREY_VENDOR_INSECURE=1``.

``osprey vendor verify``
   Verify all vendor assets exist on disk with correct SHA256 checksums.

.. code-block:: bash

   osprey vendor fetch                    # Download all assets
   OSPREY_CA_BUNDLE=/path/to/ca.pem osprey vendor fetch   # TLS-intercepting proxy, verified
   osprey vendor fetch --insecure         # Fallback: no CA bundle at hand
   osprey vendor verify                   # Check checksums

Environment Variables
=====================

.. code-block:: bash

   ANTHROPIC_API_KEY=sk-...          # Or OPENAI_API_KEY, GOOGLE_API_KEY, etc.

Provider keys live in the deployment repository's ``.env`` and are read from
there. No environment variable selects which deployment a command acts on:
every lifecycle verb finds the repository by walking up from the working
directory, and ``--repo DIRECTORY`` names another one.
