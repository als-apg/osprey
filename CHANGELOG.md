# Changelog

All notable changes to the Osprey Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Calendar Versioning](https://calver.org/).

Versions follow `YYYY.MM.MICRO`. Year and month identify the release window;
the micro segment increments for hotfixes and same-month follow-up releases.
Compatibility is documented in release notes, not encoded in the version string.

Unreleased changes are written as fragments in `changelog.d/` and folded into
the `[Unreleased]` section below when a release is cut.

## [Unreleased]

## [YYYY.M.P] - YYYY-MM-DD

<!--
TODO(release-notes PR, remove before the tag): the version and date above are
placeholders the owner fills at tag time.

Final fold on main d0ea04503d (2026-11-10): every fragment on main is folded
into the sections below and changelog.d/ is empty. Still open before the tag:
- The condensation pass (release skill, Step 4) has not run: the sections
  below the migration guide are the raw fold of every fragment since
  v2026.9.0b3.
- The Highlights and Breaking changes above them cover the release-bound list
  only. The apps contract train (#2109) is folded below without headline
  treatment (owner ruling 2026-11-06), and so are the other merges since
  2026-11-03 that are not on that list. Their **Breaking change:** entries
  are in the sections below and not yet in the migration guide: the agent's
  artifact tools (#2470), the LATTICE panel retired (#2456) and CANOPY's
  `formula-engine` rebase (#2350).
- #2397 (deny path for a waiting write-authorization request) is still open
  and needs another connectors release. If it lands, drop its Known gaps line
  and add that connectors release.
-->

First stable release after the 2026.9.0b3 public beta, and the first to run
on a facility machine. Install with

    uv tool install osprey-framework

The framework now requires `osprey-connectors>=2026.11.8`, which is released
on its own version stream (see the migration guide). Read **Breaking changes**
before upgrading a deployment: most of them stop `osprey build` with a message
naming the fix, and `osprey init --revalidate` rewrites a repo for many of them.

### Highlights

- **A three-position write switch.** Each login and machine has its own
  switch: `off`, `asked` (every machine write prompts) or `allowed` (writes run
  unprompted). The tier sets the furthest position a login may pick. The chip
  is coloured by position, one "Don't ask again" covers every machine, and the
  facility's machine is called "Live" everywhere. (#2110, #2192, #2193, #2194,
  #2214)
- **One write verdict for every surface.** The agent's tools, Python runs,
  notebooks, Bluesky lanes, the dispatch worker and facility write tools all
  ask the same switch, and every write carries the person it belongs to.
- **Safety watchdogs.** Facility-written checks judge each operator prompt in
  the background. In the web terminal a flag holds the session, and an
  unanswered hold pauses that session's writes in the connector. An admin can
  turn the watchdog off for their own sessions from Settings > Safety without a
  restart. (#2228, #2250, #2279)
- **Code review before code runs.** A second agent can read the Python the
  agent writes, against rules the deployment writes, and stop it or report it
  before the operator is asked to approve. (#2269)
- **A labelled sidebar and one web terminal.** The sidebar replaces the panel
  rail, header and status bar; the Expert and Simple views are gone, and the
  terminal is an opt-in tab beside the Agent pane. (#2185, #2249)
- **Tiers and apps.** A deployment's web terminals are access tiers
  (`readonly`, `readwrite`, `admin`) and apps that share its services. A shared
  app can say who may open it, and `osprey users list` shows who can open which
  card. (#2224, #2246)
- **One facility description drives the build.** `data/facility/` is the
  single source for channel limits, the channel-finder indexes, the knowledge
  graph, the simulator view and the agent's facility facts.
  `osprey facility import mml` and `osprey facility import list` bring existing
  machine data in, and `osprey facility validate` checks it.
- **One simulator.** The `mock` type is retired. The virtual accelerator serves
  the facility's models from its container or inside the asking process.
- **The agent record** keeps prompts, answers, tool calls, safety decisions,
  dispatch runs and bridge turns in one place, and replaces OpenObserve.
- **The optimizer** ships as its own package, `osprey-optimizer`, released
  with the framework at the same version. An Xopt-driven run writes only under
  one bounded write authorization, approved from the write chip or
  `osprey write-authorization approve`, and an `optimizer:` profile block
  deploys its API and job store; a read-only `optimizer` MCP server lets the
  agent read it. (#2353, #2398, #2399, #2403)
- **A Slack bridge** joins the Nextcloud Talk, Google Chat and Microsoft Teams
  ones: message the app directly or mention it in a channel, and the answer,
  plots and files come back in the thread. It connects out over Slack's Socket
  Mode, so it opens no port. (#2501)
- **Facility follow-ups** from the first facility deployment: process stops,
  the deployment record, connector failures and test ports. (#2182)
- **Hosts behind a registry mirror or without internet.**
  `images.base_registry` names the registry that stands in for docker.io for
  every image recipe, pulled service image and pipeline job; `offline: true`
  vendors the Bluesky panel's assets too, and a web terminal whose assets did
  not load says so instead of staying empty. (#2615)

### Breaking changes

Write switch and write posture:

- **Breaking change:** the write switch has three positions per login and
  machine, `off`, `asked` and `allowed`, and the tier key
  `control_system.writes: never | asked | allowed` is the furthest position a
  login may pick. The shipped `readwrite` tier is `asked`, `admin` is
  `allowed`. A stored `on` reads as `asked`. (#2192, #2193, #2194)
  Migration: a tier that spells `allowed` now permits unprompted writes;
    spell `asked` where prompts are wanted. The old `writes_enabled` keys are
    rewritten at build with one warning each until 2026.12.0, which refuses
    them, and `osprey init --revalidate` rewrites the files. A render built before this release reads as `never`
    until it is rebuilt. Upgrade in this order: install the connectors
    package, rebuild the Bluesky lane and dispatch containers, then rebuild the
    project. An old connectors container that meets an `asked` login-tier map
    refuses every login.
- **Breaking change:** recorded per-machine "Don't ask again" waivers no
  longer apply. One waiver now covers every move of the switch to `asked` or
  `allowed`, the live machine included, and a second covers target switches.
  (#2214)
  Migration: nothing to change in the repo; tell operators they will be asked
    once more and opt in afresh.
- **Breaking change:** the facility's machine is called "Live" instead of
  "Real machine", including the banner `Live — writes move hardware` and the
  refusal phrase `the live machine`. (#2214)
  Migration: update any monitoring, log filter or test that matches the old
    wording.
- **Breaking change:** the switch, not `approval.tools.<tool>`, decides for the
  governed write tools (`channel_write`, write-mode `execute` and
  `execute_file`, `pyaml_measure`, `queue_add`, `queue_start`, hooked facility
  write tools). `skip` now prompts at `asked`, and `approval.enabled: false` no
  longer silences them.
  Migration: `osprey init --revalidate` removes such a policy. It warns in this
    release and is refused from 2026.12.0.
- **Breaking change:** connector writes start off on every machine until that
  machine's switch is turned on, and a write tier now arms every machine (the
  shipped `readwrite` and `admin` tiers armed only the virtual accelerator
  before). An unknown owner, target or record refuses the write.
- **Breaking change:** a write with no owner is refused. Scheduled and
  triggered dispatch jobs run for no login, so their writes are refused
  (`owner_unscheduled`), with no route back in this release. Bluesky items with
  no owner no longer write either. Shared containers read a build-rendered
  login-to-tier map, `control_system.login_tiers`, and refuse a login missing
  from it as `owner_unknown`.
- **Breaking change:** shared writers reach the control system under fixed
  service names: dispatch worker *i* writes as `osprey-dispatch-<i>` instead of
  `osprey`, and each Bluesky lane OSPREY renders writes as `osprey-bluesky` or
  `osprey-bluesky-<lane>` instead of `root`.
  Migration: update any gateway or IOC access-security (ACF) or put-log rule
    keyed on `osprey` or `root`, and roll the images and the rendered project
    together; `osprey health` reports a worker writing under the wrong name.
- **Breaking change:** raw client puts (`caput`, `PV.put`, caproto, DOOCS,
  PyTango, ophyd-async signals, a RunEngine in a notebook) from the Python
  executor and notebook kernels are refused in readwrite runs too, with
  `ChannelWriteBlockedError` reason `RAW_CLIENT_WRITE`.
  Migration: replace raw puts with `osprey.runtime.write_channel` or
    `write_channels`, and submit plans to a Bluesky lane queue.
- **Breaking change:** every write surface decides by the one write verdict. A
  host process refuses a write for an owner other than its login, a live lane
  writes only when its live connector block is configured, and a facility
  server guarded whole by `writes_check` refuses its reads too while `live` is
  off. On a render whose tier is `allowed`, a
  `claude_code.permissions.remove_deny` that lifts `Bash` or `Monitor` is a
  build refusal.
- **Breaking change:** `control_system.write_tools` is deprecated in favour of
  `mcp_servers.<server>.writes`, and a facility server whose `writes:` list
  leaves out a tool it asks for, or one whose name looks destructive, gets a
  build warning. Both warnings become refusals in 2026.12.0.
  Migration: move the tools into `mcp_servers.<server>.writes`.
- **Breaking change:** `target_eligibility` loses `effective_writes_for_target`
  and `narrowing_refusal`; use `target_write_verdict(section, target).allowed`.
  `osprey.services.python_executor` no longer exports `ExecutionControlConfig`
  or `get_execution_control_config`.

osprey-connectors:

- **Breaking change:** `osprey-connectors` is released on its own
  `osprey-connectors-vYYYY.M.P` tags, separately from the framework. Thirteen
  releases since 2026.9.0b3: 2026.10.0 (the first on its own stream, with its
  public API declared, #2144), 2026.10.1 (three-position switch, #2199),
  2026.10.2 (Live wording, #2220), 2026.11.0 (session hold, #2257), 2026.11.1
  (stale-transport restart, channel keys, connection callbacks and top-off;
  tagged at #2216, notes folded in #2318), 2026.11.2 (four more
  framework-only names leave the public API, and the env chain skips launch
  variables; #2318), 2026.11.3 (a program writes under one bounded write
  authorization, #2332, released in #2351), 2026.11.4 (archiver events on a
  shared channel compose within a scenario set, #2465), 2026.11.5 (the
  simulator view names its record roles and the `PERIODIC` refresh, #2489),
  2026.11.6 (`burst` and `loss` archiver event shapes, #2510), 2026.11.7 (the
  archive seed accepts a model with no deck, #2525), 2026.11.8 (`engine_role`
  in the simulator view, now schema `osprey.facility.simulator/3`,
  `ipc.declared_gateways`, and write-authorization ceilings; notes in #2618)
  and 2026.11.9 (engines with a time contract, and public names nothing used
  made private or removed; notes in #2645). 2026.10.0 and later do not support
  framework 2026.9.0b1 to b3. This release requires
  `osprey-connectors>=2026.11.8` (raised in #2638), because the virtual
  accelerator, target eligibility and the shared-directory writers read names
  that release is the first to carry; `pyaml-cs-osprey` states the same floor,
  and `osprey-optimizer` requires 2026.11.6 or later.
  Migration: a deployment that pins versions pins two streams now, the
    framework and `osprey-connectors`, and upgrades them together. A checkout
    that builds the package needs the `osprey-connectors-v*` tags fetched; a
    fork or mirror must sync them.
- **Breaking change:** `osprey-connectors` declares its public API. The
  `control_system`, `archiver` and `ipc` subpackages hide their implementation
  modules behind an underscore and export public names from the subpackage;
  old module paths are not kept as aliases. The old `osprey.*` import paths for
  modules that moved to the package are gone too. (#1810)
  Migration: import from `osprey_connectors.control_system`, `.archiver` and
    `.ipc` (the import map is in `packages/osprey-connectors/README.md`). A
    connector subclass that overrides `_current_value_reader` must rename it
    `current_value_reader`, or every write to a channel with a `max_step`
    limit is refused. A refused connector write result must name an
    `osprey_connectors.errors.RefusalReason`. Re-run `osprey build`: a project
    rendered earlier has a limits hook that imports a removed path.
- **Breaking change:** the EPICS connector uses aioca, not pyepics, and
  pyepics is no longer a dependency.
  Migration: code that imported `epics` through OSPREY's environment installs
    pyepics itself.

Web terminal:

- **Breaking change:** a labelled sidebar replaces the panel rail, the header
  and the status bar. The customizable header and status bars, Customize, the
  `/api/bar-items` routes and saved bar layouts are removed, and a saved layout
  is not migrated. `web.rail_position` is no longer read. (#2185)
  Migration: remove `web.rail_position` and `web.bar_items` from configs.
- **Breaking change:** the Expert and Simple views are gone; the Agent pane is
  the operator chat for everyone. (#2249)
  Migration: rename `web.expert_view` to `web.terminal` (a build refuses the
    old key) and drop `web.ui_mode` (a build warns and drops it).
- **Breaking change:** web terminals need a `modules.web_terminals.renders`
  catalog with a `default_tier` in registry mode too.
  Migration: a registry deployment adds one whose default tier names the
    project its image was built from, for example `default_tier: main` with
    `renders: {main: {project: <that project>}}`, which keeps saved sessions.
- **Breaking change:** the web tier is named after `project_name`:
  `<project>-nginx`, `<project>-auth` and `<project>-web-<user>`.
  Migration: update scripts and monitoring that address containers by name.
- **Breaking change:** the roster's `persona` spellings become `tier`
  (`users[].persona`, `default_persona`, a `personas/` folder). They load with
  a warning until 2026.12.0, which refuses them.
  Migration: `osprey init --revalidate` rewrites them. A roster that names the
    removed `control-assistant-va-readwrite` preset names
    `control-assistant-readwrite`.
- **Breaking change:** notebook outputs no longer run script (Bokeh's
  in-notebook renderer and Vega/Altair stop rendering), and content an agent
  writes is served in an opaque origin that cannot call OSPREY's APIs.

Facility description and build:

- **Breaking change:** channel limits are authored in
  `data/facility/limits.yaml` and rendered by the build.
  `control_system.limits_checking.mode: exclusive | optional` replaces
  `allow_unlisted_channels` (`false` is `exclusive`). A profile that states
  `limits_checking.database_path` or ships its own `channel_limits.json` stops
  the build, and the limits reader refuses a `defaults` block.
  Migration: write the old table's records into `data/facility/limits.yaml`,
    delete the key and the old file, then `osprey build`.
- **Breaking change:** the `mock` control-system type is retired.
  Migration: `osprey set connector=virtual_accelerator
    config.control_system.connector.virtual_accelerator.serving=in_process`,
    then `osprey build`.
- **Breaking change:** the build writes the channel-finder indexes, the
  knowledge graph and the simulator view from the facility file. The
  hierarchical index's levels change, graph IRIs are minted from the facility
  code, and the commands and keys that built them by hand are removed
  (`osprey mml`, `osprey channel-finder build-database` and `generate`,
  `osprey knowledge build-ttl`, `seed-graph`, `build-index` and
  `compile-ontology`, `channel_finder.pipelines.*.database.type`).
  Migration: import machine data with `osprey facility import mml` or
    `osprey facility import list`, check it with `osprey facility validate`,
    and rebuild. Saved paths and queries on the old levels or `demo_` IRIs no
    longer resolve.
- **Breaking change:** a deployment's own skills, agents, rules and output
  styles render only where a list in `profile.yml`, a tier or an app names
  their path.
  Migration: `osprey init --revalidate` names every existing one, so renders
    stay as they are.
- **Breaking change:** profile and config keys removed or renamed, each
  refused or warned with its replacement: the profile field `tier`,
  `osprey profile expand` (use `osprey init --revalidate`), `facility.prefix`,
  `facility.name`, `facility.timezone`, `facility.ontology`,
  `dispatch.facility_name`, `bluesky.devices_file`, `services.graphdb.index_path`,
  `web.feedback.max_store_bytes`, the `openobserve` service and the
  `claude_code.telemetry` block.
- **Breaking change:** connector and archiver bounds are `timeout_s`
  (`control_system.connector.<type>.timeout_s`, `archiver.settings.timeout_s`),
  and the MongoDB archiver's login sits under `auth:`. A seed's or scenario's
  `noise` is `{absolute: x}` or `{relative: f}`, and a scenario's `stuck` fault
  is `channel_faults: {<address>: stuck}`.
- **Breaking change:** each qmd corpus has its own sidecar and index.
  Migration: the first deploy rebuilds each index; the old `qmd_index` volume
    can be removed.
- **Breaking change:** models are named by the id the gateway serves.
  `claude_code.models` is `claude_code.aliases`, and
  `logbook.composition.default_tier` is `logbook.composition.model`. A
  deployment that names no model now runs Sonnet 5, and the als-apg entry no
  longer lists Opus 5.
- **Breaking change:** a wiring record's engine word is `engine_role`, apart
  from a channel's own `role`: the facility file's wiring records, the
  simulator view (now schema `osprey.facility.simulator/3`), an engine's
  `describe()` and a scenario's `still` selection all spell it that way.
  (#2333, #2638)
  Migration: rename `role:` to `engine_role:` in a scenario's `still`, have a
    custom engine's `describe()` return `engine_role`, and rebuild; a view
    from an older build is refused with a rebuild message.
- **Breaking change:** the deploy block's `ci_image_prefix` is gone, and a
  profile that sets it is refused. (#2615)
  Migration: set `images.base_registry` under `config:` instead. It keeps
    Docker Hub's paths, so the pipeline pulls
    `<mirror>/library/python:3.12-slim` where it pulled
    `<prefix>python:3.12-slim`.

Optimizer:

- **Breaking change:** the optimizer moves out of the framework into the
  `osprey-optimizer` distribution (`import osprey_optimizer`); the framework's
  `optimizer` extra installs its base, and `osprey-optimizer[service]` the
  engine and HTTP service. `POST /jobs/{id}/start` requires
  `?config_revision=` and answers `409` when the draft changed since it was
  shown. A deployed optimizer drives `va` and `standin` only; `live` is
  refused at build. (#2353, #2399)
  Migration: imports of the optimizer come from `osprey_optimizer`. A client
    that starts runs reads the draft with `GET /jobs/{id}/config` and sends
    its `X-Config-Revision`. Rebuild and `osprey up`: the deploy provisions
    the write ledger `var/write_ledger/` and mints `OPTIMIZER_LAUNCH_TOKEN`.

Logbook (ARIEL):

- **Breaking change:** the ARIEL web interface is one Logbook screen with
  facets and paging, and stored entries saved from it change their source
  name: an always-on migration renames `source_system: "ARIEL Web"` to
  `"OSPREY Logbook Web"`, and new local saves store the new name. The MCP
  tools return a null `entry_url` for every entry that lives only in ARIEL.
  (#2391, #2375)
  Migration: run `osprey ariel migrate` after upgrading to backfill entry
    subjects, and update any query, export or filter that matches
    `"ARIEL Web"`. A custom search module that declares neither
    `accepts_entry_filter` nor `accepts_offset` is now called with a wider
    `max_results`.

### Known gaps

- The write switch does not yet govern PVAccess puts (p4p `Context.put`, pvaPy
  `Channel.put`) from the Python executor and notebooks: they keep their
  approval ask and limits check, as before. Fixed in the next release.
- External-worker Bluesky lanes, and plans that reach a queue other than
  through OSPREY, are not governed by the write switch: they run at the lane's
  ceiling. Fixed in the next release.
- Scheduled and triggered dispatch jobs cannot write (see Breaking changes).
- A program's write-authorization request can be approved but not denied:
  closing the review card changes nothing, so an operator who does not want it
  cancels the run, or the run waits out its timeout (#2397).

### Added

- Optimizer service (engine): the Xopt-driven tuning engine ships as its own
  distribution, `osprey-optimizer` (import `osprey_optimizer`), released with the
  framework and carrying the same version. Its base install is the models and
  the environment-file format (the framework's `optimizer` extra pulls in just
  that); the engine and its HTTP service are the `osprey-optimizer[service]`
  extra.
  An optimization environment is a YAML file of setpoint/readback channels and
  observables (a single channel, or the `rms`, `mean` or `max` of several), and
  every write it makes goes through the control-system connector, so the write
  switch, channel limits and `max_step` apply to each one; a search window wider
  than a channel's `max_step` is refused when the run is created. A run names its
  control target (`va`, `standin` or `live`) and starts from the setpoints as it
  finds them, writing nothing before its search begins. A cancelled or failed run
  leaves the machine where it is; its starting setpoints are recorded, and
  writing them back is the operator's call, through the results restore
  endpoint. A cancel marks the job `cancelling` until the worker has stopped,
  and no other run can start before then. An environment can
  offer presets, named run setups (variables, search windows, algorithm, length,
  objective) that a draft names to fill whatever it leaves unset. Driving it
  from the agent comes in a later release. (#1811)
- A declared qmd corpus with `embed: false` builds its keyword index only and skips qmd's
  embedding pass. (#2383)
- `images.base_registry` (env `OSPREY_BASE_REGISTRY`) names the registry that stands in for docker.io: every shipped image recipe's base, the pulled service images and the scaffolded pipeline's job images come from it. (#2385)
- The chat answers `/mcp` itself: it lists each MCP server with its tool count
  and OK or the error that kept it from listing, without a turn. (#2494)
- ⌥-click (Alt+click) a session or the **Agent** row to expand every session in the tree, or collapse them all when all are open.
- `osprey app check` reports what `osprey build` reports of a deployment's applications, plus the
  import boundary and each application's panel lint and JS tests, without building anything.
- Applications are checked against the rules that hold for every render — container, panel, tool-name and vendor-path rules, names OSPREY keeps for itself — and an application package that would move a version OSPREY pins is refused naming both versions; every violation is reported at once.
- An application can be described by an `apps/<name>/app.yml` manifest: its containers, panel, tool server and vendored files. OSPREY validates the file strictly, and a compose key such as `volumes` or `network` is refused with a message naming where it belongs.
- `osprey app new NAME` copies the shipped example application into
  `apps/NAME/` under the new name, lists it in `profile.yml`, and keeps fetched
  panel files out of git.
- A login arms an application it shows when its `control_system.writes` tier is
  `asked` or `allowed`; an armed application with write tools needs a container
  that sets `drives_target: true`, and a shared login that arms one is refused,
  naming its cards and the application.
- Application containers that run on the project image (`runtime: osprey`) render through a bundled `app_runtime` compose template, published on loopback only. `osprey build` builds each application package under `apps/<name>/` and each local-path dependency to a wheel and installs it in the build venv and in every project image under the framework's pins, which it never moves. Each is bound to its wheel, so no index can serve its name; a wheel that does not fit the image or holds a `.env`, and a distribution named twice, stop the build.
- An application's `vendor:` entries are fetched into `apps/<name>/<into>`, each
  checked against its SHA256; a target outside the application, or a file name
  that does not carry the version it fetches, is refused, and an offline or failed
  fetch reuses a verified copy or fails naming the application and package.
  `osprey vendor fetch --target DIR` fetches one manifest target, which may name
  `"base": "package"`, and an application panel's `/vendor/` mount is cached as
  immutable like an interface's static vendor tree; proxied panel traffic and cap
  paths keep their own headers, and a route's `no-store` is never overridden.
- The application SDK `osprey.apps.sdk` carries its public surface: `serve_tools`, the approval and write-chain names, the run lock, facts readers and the helpers an application imports.
- The agent's `entry_create` takes `follows_up`, the id of the entry the new
  one continues: drafts and direct writes thread under that parent the way the
  Logbook's own Follow up does, and the compose sheet shows the parent of an
  agent's follow-up draft. An unknown parent is refused.
- ARIEL: the web interface composes in a sheet over the list (``c``; **Expand**
  spans the panel), autosaved to the browser only, with a line saying where
  Submit goes. **Follow up** (``r``) links a new entry to the selected one via
  ``metadata.follows_up``, and the reader shows the thread from the new
  ``/api/entries/{entry_id}/thread``. The **Drafts** mailbox lists agent-staged
  drafts (``GET /api/drafts``) and **Discard** deletes one
  (``DELETE /api/drafts/{draft_id}``). Entry creation returns
  ``attachments_published``; ``entry_publish`` carries ``follows_up``. Run
  ``osprey ariel migrate`` to index follow-ups.
- ARIEL entries carry the channel addresses and device ids they quote as facility references; the entry view and the agent's logbook tools show them with the records they name, and the stored text is unchanged.
- ARIEL: the reader's **Enhancements** tab (``e``) shows each module's state and
  output for the entry, with a copyable re-run command for a failed or stale
  one, and **Related** lists up to five entries nearest by embedding. Embedded,
  **Ask agent** (``a``) puts a citation of the entry into the assistant prompt
  without sending it, and entries the agent creates flash in the list or count
  in an **n new** pill. New: ``osprey ariel enhance --module NAME --entry ID``,
  ``GET /api/entries/{entry_id}/enhancements`` and
  ``GET /api/entries/{entry_id}/related``.
- `/osprey:build-app` skill: guides a coding agent through building an app
  (panel, backend, agent tools) and first checks itself against the installed
  app contract, printing that contract from source.
- `osprey build` checks a deployment's applications before writing anything: manifest and
  per-login rules, pinned versions, projected ports and vendor files each refuse by name. It then
  projects each application the profile's `apps:` block lists into every render, writes each
  service's `services/<key>/app-config.yml`, reports an `apps/<name>/` the block does not list
  (neither built nor installed; a package that is no application is a `dependencies` local path, and
  a requirement naming a package only that directory holds is refused), and the deploy summary gains
  an `apps.<name>` block.
- A message entered in the chat while the agent is replying now waits above the composer and is sent when the turn ends. Its bubble offers Interrupt and ask now, or takes the text back into the input; Stop also returns it to the input.
- `osprey.runtime.guarded_run.command_write` fires one value at a batch of
  commands under the run lock. Command-ness is the rendered facility's, and the
  batch is refused whole unless the run's owner, target, limits posture and
  limits validator all admit it; every call files a `command_write` audit
  record naming the commands that fired, or the whole batch as attempted when
  the batch call raised.
- Chat: while the conversation is being compacted, the status line shows a stack
  of lines folding into one beside its label, and returns to "Working…" once the
  summary lands.
- An application container's declared `state` directory is provisioned inside its state zone at build and mounted at its declared path, through the per-service `state_mounts` fact.
- A Channel Access target whose block spells `gateways: []` declares that it has
  nothing to probe: a return to it as the deployment baseline skips the
  reachability check and says "target declares no endpoints". A missing
  `gateways` table is still refused as `gateways_missing`, so a forgotten table
  is never read as a direct connection, and a table with gateways is probed as
  before.
- Right-click a past session in the session tree and pick **Delete…** to delete
  it with its artifacts and its transcript. The agent record keeps its rows.
- An application container that declares `drives_target` renders with its staged config, the guarded-run mount and, on an armed render, the control tree and limits mounts, plus a stop grace period from the connector's call bound and Channel Access addressing on the compose bridge. A bridge container whose Channel Access target has no derivable name server refuses the build.
- An engine's `describe()` may list `marks` for a wiring record (the facility schema states `command`, for a write record alone); the build writes them onto the record beside its role, plane and refresh, and stops on any description word the schema does not state.
- An engine plug-in may state `scenario_stops(wiring, faults)`; a build stops on each refusal it returns, naming the scenario fault and the plug-in's remedy.
- The error guidance hook classifies application error types from `hook_config.json`'s `app_error_types` and adds their guidance text to the injected context.
- Application `image:` fragments render from a stated, strict context and may not mount the guarded-run, limits or identity paths; a projected container publishes its ports on loopback only.
- Bluesky: `get_run` can wait for a run to finish or change before it answers,
  and a run record now says when the run started, how long it has run, and how
  fast it is going.
- The approval hook decides an application's declared write tools by the write
  switch's position: at `asked` it prompts, bound to what the prompt showed; at
  `allowed` the call runs bound and stamped the same way; a call it cannot show is
  refused in every position.
- Web terminal: the hub lays a deploy's seeded scenario sessions (transcripts, ledger rows, titles, artifacts) into its own stores at startup, filling only what is absent, and deletes the seed file once every session is complete.
- A deployment repo's own `.dockerignore` now keeps the files it names out of the deployment's images.
- `config: {images.platform: linux/amd64}` in `profile.yml` names the platform
  every image OSPREY builds is built for: `osprey build` judges deployment wheels
  against it and names it on the project Dockerfile's `FROM`, and `osprey up`
  builds every image and compose-built service for it. Unset, images are built
  for this host, as before.
- The application import boundary, under which application code imports OSPREY only through `osprey.apps.sdk`, `osprey.facility.kinds` and public `osprey_connectors`, is now checked by `osprey.apps.boundary`, and `osprey.apps.loader` is reachable only from the build.
- `osprey init` copies each application its preset lists into `apps/<name>/`, with a `.provenance` sidecar recording the preset and the files as copied; `--force` replaces the copy, and `osprey validate` and `osprey init --revalidate` report an edited application as drift. The deployment `.gitignore` ignores `/apps/*/panel/vendor/`, where `osprey build` fetches vendor files.
- A machine can live in its own repository: it builds and serves its virtual
  accelerator alone, and a deployment mounts it with the root profile key
  `machine:`, keeping only `knowledge/`, `ariel/` and `benchmarks/` itself.
  Scenarios marked `start: true` form the start set.
- A build profile's `optimizer:` block (`environments_dir`, `port`)
  deploys the optimizer API and its Redis job store. The API publishes at the
  layout's `optimizer` slot on the deployment's bind address (loopback by
  default), the job store publishes no port, `osprey up` mints the
  `OPTIMIZER_LAUNCH_TOKEN` that arms the routes that start runs, write to the
  machine or load a model, and the environment files are mounted read-only.
  The login-tier map is staged into its config, so each run is judged at its
  owner's rung.
- The optimizer's results restore, `PUT /environments/{name}/variables` and
  `POST /environments/{name}/evaluate` write for the login `X-Osprey-Owner`
  names, each under a one-step write authorization the service issues for
  exactly its values, uses once and revokes. Below that login's `allowed` switch
  the call answers `403` with the reason and writes nothing.
- The `optimizer` MCP server can now draft runs and stop them: `create_draft`,
  `update_draft`, `pause_job`, `cancel_job` and `preview_restore`. A pause or a
  cancel asks for the operator's approval. A draft is always for the session's
  control target: a config naming another is refused. It still cannot start, resume or
  restore a run; a person does. A new `operating-optimizer` skill renders when
  the server is on.
- An `optimizer` MCP server reads the optimizer service: environments, observable
  readbacks, presets and initial bounds, jobs and their configs, stored results and
  saved models. It is off until `claude_code.servers.optimizer.enabled: true`.
- A new OPTIMIZER tab in the web terminal, where a person starts the optimization
  run an agent drafted. Turn it on with an `optimizer_web:` block beside the
  profile's `optimizer:` block; the build deploys an `optimizer-web` sidecar that
  serves the panel and relays to the optimizer service. Start runs exactly the
  draft shown: a draft changed since the panel displayed it is refused, and a
  replaced draft holds Start until the operator has looked at it. The panel warns
  before Start when the session names no operator account, since the service may
  then refuse the run's writes (`owner_unknown`). Cancelling stops the run where it
  is: the machine keeps the last point written, and restoring the starting
  setpoints stays a manual step. The build also raises
  `web.panels.optimizer.read_timeout_s` above the panel proxy's 30 s default, so a
  slow Start is not cut off; an explicit value wins.
- An optimizer run writes to the machine only under one bounded write
  authorization: the setpoints it writes with the run's window each, its
  resolved evaluations plus one step for `apply_best` or an explicitly set
  `restore` (none for `hold`), an expiry and a lease the run renews. While the
  owner's switch stands at `allowed` the run issues its own; otherwise it asks,
  its job shows `write_authorization.state: awaiting`, and a person approves it
  from the chip or `osprey write-authorization approve`. Each evaluation is one
  write call; a run whose authorization is spent, stopped, out of range or
  lapsed ends with that reason and writes nothing more, and a cancel or failure
  still writes nothing.
- A deployed optimizer binds the deployment's write ledger, `var/write_ledger`,
  read-write as `OSPREY_WRITE_LEDGER_DIR` whenever it binds the control-context
  tree, and warns at start when it cannot write there.
- The bundled example app is now orbit-bump, a closed local orbit bump computed
  from the lattice model and applied under approval, replacing `knob-example`;
  `osprey app new` copies a separate tiny scaffold instead. The app SDK gains
  closed-orbit models, model members with s-positions, the model's orbit response
  matrix and the connector's pre-write verdict, and apps can ship agent skills
  with their panels framed in the web terminal. `osprey app check` fails a panel
  that does not load the design-system stylesheets or sets a class with no CSS rule.
  The SDK also adds `owner_target`, the machine a request's owner is on, and
  `serve_tools` takes a `register(server)` function and builds the server itself.
- Web terminal: an application panel's declared `web.panels.<id>.agent_routes`
  are admitted on the panel credential, exact match only. On a login whose
  `control_system.writes` tier is not `allowed`, the panel proxy refuses write
  methods and websocket upgrades to the application, naming the login and the
  application; a panel block spelling `arm_writes` is refused at load.
- Web terminal: an application panel is armed for writes by the login's
  `control_system.writes` tier, `asked` or `allowed`, and the owner's switch position
  then decides each write; a render that cannot be read leaves it unarmed.
- The design system has the panel-chrome primitives: a panel toolbar, a text
  segmented control, a search field, status chips, a resizable panel sidebar, a
  centred state, an info popover, a content strip and 15 new icons. A disabled
  button stays hoverable, so its tooltip can say why. `DESIGN.md` section 12
  states the panel-side rules for new panels.
- An application panel's launch token is held by the terminal and injected on that panel's proxy hop, never sent to the browser; an inbound `X-Launch-Token` is dropped on every panel hop. A panel naming any variable but its application's own launch token refuses the deploy.
- Web terminal panels take `web.panels.<id>.read_timeout_s`: a number of
  seconds that bounds that panel's proxied requests, or `null` for unbounded. A
  panel that omits it keeps the shared 30 s default.
- An application panel declaring `roster_grant` gets the entitled users' terminal secrets on its own service, through the same grant the Bluesky panel uses; with `single_user_alias` on a deployment without a roster, it accepts the deployment secret under the build account's name. Every service template now reads its own `service_key`, `service` block and `panel_grant`.
- In plan mode, every OSPREY tool that acts is refused with one message saying to leave plan
  mode to run it. Stopping a queue or a run, removing a queued item, validating a plan and the
  full health check stay available.
- The profile `apps:` block lists the applications a deployment runs, with their `config` and per-container `network`/`env`; a login selects from them with `apps.<name>.shown` and `lists: own | add`.
- `osprey up` builds the project image for projected `runtime: osprey` application containers, and mints a projected panel's launch token beside OSPREY's own credentials.
- An application's tool server is projected onto every login that shows it, as a facility server whose approval tools are its declared writes: an armed render (`asked` or `allowed`) hands them to the write switch and a `never` render denies them. The render also carries each tool's approval entry, the application's error types and the list of tools denied in jobs.
- The build can project an application's panel onto each login that shows it, as a `web.panels.<id>` block on the application's loopback port; its write routes and launch token are projected only where the login arms the application.
- The build can project each application the profile's `apps:` block names into `services.<key>` entries: a `runtime: osprey` container renders through the bundled application runtime, an `image:` container through its own compose fragment, each on a loopback port from the facility band unless pinned.
- The Claude Code hooks gain a stdlib-only prompt renderer that fills an approval prompt from `input.*` and `state.*` values with escaping and caps, shipped as the `hooks/prompt-render` row beside the approval hook. Approval-prompt labels now escape bidi overrides, zero-width marks and C1 controls to a visible codepoint token.
- The facility schema states two more refresh words for a wiring record, `fast` and `event`, matching the connector view vocabulary.
- Guarded runs: the record a run takes the lock with now says whether limits
  checking is off for its target, taken from the target's limits posture; a
  record that claims otherwise is refused before the run starts.
- Facility: a scenario may state a `sessions` block, past conversations with the
  agent. The build refuses a malformed block, a channel the facility lacks, a
  logbook id that is neither `nominal`'s nor the scenario's own, and an unknown
  OSPREY server; `scenarios.json` carries the block.
- A scenario's `still` can select readings by model or place, narrowed by group, signal, `engine_role` and plane, instead of listing every address.
- The application SDK gains the approved-call check an application tool server runs before it acts.
- Simulation: a deploy that activates a default set holding scenario `sessions` writes their history (transcripts plus artifacts, anchored on the scenario clock) for the first readwrite roster user and the host identity; `osprey sim apply` and archive reseeds write none.
- Web terminal: drag a sidebar group's heading to move the whole group, rows and all, among the other groups; the order is kept per user like the rest of the groups.
- Web terminal: double-click a sidebar group's heading to fold it, and drag the faint line below the Workspace group to make it taller (double-click the line to reset); both are kept per user.
- Web terminal: pin sessions in the sidebar. A pinned session also shows in a **Pinned** group above **Sessions**, in pin order; the pin is kept on the session record (`PUT /api/sessions/{id}/pin`).
- Sidebar: Tab shows or hides the labels, as Ctrl+\ does, except where Tab
  already does something (a text field, the message box, the terminal, an open
  menu or dialog). **More › Tab shows or hides the sidebar** turns it off.
- A Slack bridge joins the Nextcloud Talk, Google Chat and Microsoft Teams ones:
  message the app directly or mention it in a channel, and the answer, plots and
  files come back in the thread. `slack_bridge:` in a build profile deploys it;
  it connects out over Slack's Socket Mode with `SLACK_BOT_TOKEN` and
  `SLACK_APP_TOKEN`, so it opens no port.
- The provider catalog ships a keyless `vllm` entry for a self-hosted vLLM server
  (default `http://localhost:8000/v1`, or `VLLM_BASE_URL`), and the `als-apg`
  entry lists `claude-sonnet-5-5`. On `als-apg` the `sonnet` alias now resolves
  to `claude-sonnet-5-5` in every deployment, as `haiku` and `opus` already
  resolve to their 5.5 models; the default model stays `claude-sonnet-5`.
- The deploy provisions the holders' write ledger, `var/write_ledger/` with its
  `requests/` directory. Every web terminal binds `var/write_ledger/requests`
  read-only (`OSPREY_WRITE_REQUESTS_DIR`) and the whole `var/write_ledger`
  read-only for the programs' self-issued grants and their ledgers
  (`OSPREY_WRITE_LEDGER_VIEW_DIR`); no terminal sees another login's grants. The
  entrypoint joins both directories' group.
- The web terminal shows and approves programs' bounded write authorizations.
  `GET /api/terminal/write-authorizations` lists the requests waiting on the
  signed-in login (read from the holders' request directory,
  `OSPREY_WRITE_REQUESTS_DIR`) and the login's authorizations, each with its
  steps used of its total; `GET /api/terminal/write-authorizations/requests/{id}`
  is the approve card (channels, ranges, the limits bound of each range, current
  values read together within about 3 s (a slow machine's pool closes in the background), steps, expiry, lease, holder, label, completion,
  and that a write the limits refuse still costs its step);
  `POST /api/terminal/write-authorizations` `{request_id}` approves, refusing a
  target the render does not configure, a login whose ceiling is `never` or
  `unknown`, a read-only run, a range outside the limits database, a
  non-numeric channel and a channel another program of this login holds
  (`auth_overlap`); `POST /api/terminal/write-authorizations/{id}/revoke` stops
  one. An approval is the confirmation for every write of the run, also for a
  login whose ceiling is `asked`. The popover gains a Programs section, which
  also lists a program's self-issued grant at `allowed` (`self_issued`, not
  `revocable`: moving the switch off `allowed` ends it), shows expiries as the
  time left, says *reading values…* while a card loads and *Approving…* while an
  approval is sent, names in the Limits column a channel the limits database
  does not list or marks not writable, names every channel a refusal is about,
  and says *Already stopped.* for a revoke of one that had ended. `GET /api/terminal/posture` rows
  gain `program_hold` and `programs` (`used` included); a grant whose lease
  lapsed or whose steps are spent is not one. The chip keeps the switch word for
  such a machine and adds `· program` (for example `writes off · program`) with
  a ring dot, `any_armed` counts it, and the row's off control revokes it.
- `osprey write-authorization approve <request> | list | revoke <id>` approves,
  lists and revokes programs' bounded write authorizations without a web
  terminal: inside a deployment repo against the control-context tree (any
  owner), outside one against this account's own state directory. An approval
  checks the request's target against the configured control targets, the
  owner's tier (judged as the program's own writes will be: the staged login map
  for a program in a shared container, the rendered config for one on the host)
  and the limits database as the web terminal's card does, and
  records `approver.identity` as the acting identity with `surface: "cli"`;
  `list` prunes grants that ended more than an hour ago, shows each one's steps
  used of its total (one whose steps are spent or whose lease lapsed is not in
  force) and the programs' self-issued grants from `var/write_ledger`, which
  `revoke` also reaches. `osprey status` adds a table of the programs'
  authorizations in force under Writes: owner, holder (program), target, steps
  used of total, time to expiry and lease. Every surface also finds the request and the ledger of
  a program run on the host with no ledger bound, which keeps them in its
  owner's own `authorizations` directory.
- ARIEL: a facility adapter can declare the entry fields it needs (for example a
  log category or a severity), and the create form and the agent's entry tools
  fill them in while the server validates each value, including choices an
  adapter lists at request time. The upload route now forwards only declared
  fields to the adapter; undeclared keys in the submitted `metadata` are no
  longer passed through. (#1022)
- `mya_archiver`: read channel history from MYA over the myquery HTTP service, the
  archiver every Jefferson Lab machine shares. Every setting is optional — the
  `jlab-archiver-client` library carries its own server and protocol — so a
  deployment inside the facility's network needs only `archiver.type: mya_archiver`
  plus `jlab-archiver-client>=4.0.1` in its profile's `dependencies:`. Aggregates
  are computed by myquery's `mystats` endpoint; `median` alone falls back to
  client-side binning, which MYA cannot compute. Because MYA records changes
  rather than samples, reads ask for the prior point, so a channel whose last
  change predates the window still reports that change at its recorded time. (#1050)
- An archiver's settings have one home, `archiver.settings`, for a built-in and
  for a connector selected by its dotted module path alike, so a deployment-local
  archiver can now be configured from a profile. A built-in's own-name block
  (`archiver.epics_archiver`) still reads.
  Any other block under `archiver:` that no archiver reads now stops `osprey build`
  and is named. (#1052)
- The Simple view's message box suggests the project's skills and commands when a message starts with `/`. Arrow keys move through the list, and Tab or Enter fills the chosen command in; Enter again sends it. (#1077)
- The approval prompt of a read-write `execute` or `execute_file` call binds it to
  its control target and asks: the approved call carries `approved_target`, set
  over any value the agent supplied. A read-only call keeps its prompt and carries
  no binding. A run on a held target fails before it starts, naming the holder.
- Deployment: `osprey build` provisions a shared `var/guarded_run/<target>/`
  directory for each configured control target before compose runs, setgid and
  group-writable, and every web terminal and dispatch worker mounts
  `var/guarded_run` read-write under its project directory. A session run on the
  host creates the directory under the repo root when it is missing.
  `osprey.runtime.guarded_run.guarded_run_dir(target)` resolves it, keyed by the
  target name.
- `osprey.runtime.guarded_run` gains `lock(target)`: one read-write run per
  control target across every process and container of the deployment. A second
  run on the target is refused with `OspreyRunBusy` naming the holder's pid and
  start time, a read-only run is refused before the lock is taken, and a process
  forked at any moment never holds the lock. `osprey.runtime` re-exports
  `guarded_run_dir`, `GUARDED_RUN_DIR` and `LOCK_FILE_NAME`.
- `osprey.runtime` gains `read_channels`, which reads several channels in one
  connector call, and `channel_limits`, which returns a channel's limits record. A
  batched read that leaves any channel without a value raises
  `ChannelReadFailedError` naming every failed channel and the error each one
  raised. A cancelled `execute` or `execute_file` run gets `SIGINT` and the time
  left to its deadline to finish before it is killed, and both tools run one
  shared gate sequence. A read-write run is reported as write activity even when
  no write pattern is detected. The EPICS connector re-types a channel whose
  request type was fixed before it connected. A channel limits record must state
  `writable`.
- A new read-only tool on the `python` server, `model_response_matrix(model, correctors)`,
  solves a served model's orbit response to its corrector setpoints on the model's own
  deck. Each corrector is stepped by the measurement file's `corrector_delta`, converted
  to the corrector's hardware unit, and every row and column names its address and
  device. The solve runs in a sandbox child stopped at the executor's timeout.

  A new tool on the `python` server, `pyaml_measure(kind, model, correctors, modulation,
  by)`, measures an orbit, dispersion, tune or chromaticity response, or the chromaticity,
  with pyAML on the control target. Every write holds the control target's run lock; a
  stopped or failed run leaves its setpoints where it stopped and lists each with its
  value before the call, each step is sized so no setpoint's largest single write
  exceeds half its limits.yaml `max_step`, and the run starts only on the pyAML view the
  build recorded and the approval prompt showed. A group stepped as one knob is stepped
  unipolar and moves every member by the same integrated strength, sized so the tune or
  chromaticity moves no more than under the strongest single-magnet step; its column is
  per unit integrated strength, the sum of its members' columns. Both tools register when
  the render carries a pyAML view.

  `pyaml_measure` asks for approval on every call and, with writes off, is denied like
  every write tool, whether or not the render carries a pyAML view; `model_response_matrix`
  is read-only and runs without a prompt. The prompt lists every setpoint the measurement
  writes, the RF frequency setpoint included, and the modulation the call resolves to.

  A render with a pyAML view installs the `pyaml-tools` skill and a measurements section in
  `CLAUDE.md`: every measurement goes through `pyaml_measure`, pyAML reads stay in
  `execute`, and a single-pass model, which has no design simulator, is read live only. The
  control-room presets list the skill; the knowledge and logbook apps, which run no
  Python sandbox, drop it.
- The web terminal can open the Session Activity Log — agents, tool calls, artifacts and conversation of the session you are on — from an Activity button on the panel rail or the command palette, in both views. It opens in a new tab and shows the session the terminal was on when you clicked. (#1315)
- A panel tile whose backend has stopped answering now shows a warning strip with the time it stopped and greys the page below it, on the same poll that dims its rail entry. The page underneath is left as it was and stays readable and usable, and the notice clears on the backend's next answer. (#1319)
- `services.qmd.corpora` declares further corpora, each served by its own sidecar
  at `services.qmd.port` + 2 onwards. A corpus is `managed` (the sidecar indexes
  its `source`) or `prebuilt` (the sidecar serves an index built elsewhere, such
  as on a GPU host, from `index_dir`). A new `qmd` health category shows each
  sidecar's documents and how many still have no vectors. (#1566)
- Viewable stored logbook pictures offer **Use in Workspace**, which keeps the
  prepared rendition with logbook provenance in the local artifact gallery for
  agent-assisted image reasoning. (#1822)
- A facility adapter can publish an entry's files with the entry: one that
  declares `supports_attachments` receives them in `create_entry`, and the
  facility's copies replace ARIEL's once read back. A file the logbook refuses
  fails the submit with a warning that the entry may already exist without it.
  The `generic_json` adapter keeps files beside its JSON file. (#1952)
- pyat engine: element errors are scenario faults. Every element a setpoint binds
  carries `<element>/dx`, `/dy` and `/roll`, and one bound on a polynomial also
  `/dB<n>` and `/dA<n>` (index 0 to 9), keyed by the served deck element's name so
  the magnets on one series supply are faulted one by one. A scenario writes them
  under `faults.<model>`, as `{QF_01_1/dx: 1.0e-4}` or `{QF_01_1: {dx: 1.0e-4}}`;
  they are seeded once at build, a multipole delta is kept through every later
  write of the supply, and a seed with no closed orbit fails the model with a
  solve error naming the elements, its status naming the scenario. (#2188)
- MML import: devices are placed from the export's `DeviceList`. A model's
  `places` answer in the mml mapping (`sectors: {count, token}`, `model_wide`,
  `sector_override`) makes the import write the model node, every listed sector
  and `imported/mml/places.yaml`, and place each device by its own rows, so the
  magnets on one series supply land in their own sectors although their supply's
  one address is in sector 1. A row sector beyond the list, or a device placed
  by its rows while its only address names another sector, stops the import
  unless answered. `place_from` gains `imported`: a place only an importing layer
  states yields to a span that is that place or inside it, stands under a span
  that holds it, and stops `place-conflict` against any other span. (#2189)
- `osprey users list [USER]` shows who is on the roster and who can open which card: every card with its render, write posture and port, a grid of logins against shared cards, unused tiers, and with `--live` container state and last sign-in. `--json` prints the same document. (#2223)
- A `shared:` login can say who may open it: `{name: <login>, access: [...]}` takes the same principals as a roster entry, and a new `tier:<tier>` principal admits every roster login of that tier. A bare name still opens to the whole roster. (#2245)
- The virtual accelerator gains a model runner that serves a simulator view's composite on Channel Access and PVA beside the existing runner: writes are clamped into each setpoint's limits band, a refused write raises an alarm and is never echoed, chromaticity readings publish on the next periodic pass after a write, and the model RPC answers over the view's address set. A float channel in the facility file may state a display `precision`, which the model runner serves; the build stops when another value type states one.
- `osprey build` stops on a scenario archiver event whose `at` or `until` lies outside 0 to 1, whose `width` is not greater than 0, whose `at_time` is not an `HH:MM[:SS]` time of day, whose `at_when` is not `{days_ago: <int>, time: HH:MM[:SS]}`, or whose `to` the channel's value type, options or shape refuses.
- `web.panels.<id>.rewrite_prefixes` names further root-absolute prefixes the
  panel proxy rewrites in one panel's responses — a bundle's router base, a
  second mount point — on top of the panel's own `path`. Each list applies to
  the panel that declares it and to no other. (#877)
- A custom MCP server can hand back a display path and let the framework judge it:
  `phoebus_panel_lookup` answers whether the terminal may open the display at that
  path, and returns the panel name `phoebus_open_panel` opens it by. A path reached
  through a symlinked display mount, spelled relative to `config.yml`, or given as a
  `file:` URL matches the registration it names. (#881)
- The health-check how-to now shows how to grade a deploy host's CPU load and
  free memory from the statistics IOC a facility already runs on that host: one
  unbanded row for reachability, `channel_read` bands on the readings, and the
  readings held behind the reachability row so an unreachable IOC reports once. (#887)
- A link to a panel in an agent's answer in the web terminal's Simple view now
  opens that panel in the workspace, so a cited facility-knowledge concept
  opens the KNOWLEDGE panel at that concept. Other links in an answer open in
  a new tab instead of replacing the terminal page. (#920)
- The facility-knowledge tools now name a link for every concept.
  `list_concepts`, `read_concept` and `search` each carry a `url` that opens the
  concept in the KNOWLEDGE panel, and the knowledge specialist cites it as a
  markdown link, so an operator can open a document straight from the answer
  instead of carrying a concept ID over to the panel by hand. (#920)
- The artifact gallery can delete several artifacts at once: Shift-click picks a
  run of rows, Cmd/Ctrl-click adds or removes one, and Delete in the bar above the
  list removes them after one confirm. (#924)
- A web terminal can run on its user's own Claude subscription. With
  `claude_code.allow_personal_login: true` (off by default), a terminal whose
  `CLAUDE_CONFIG_DIR` holds a completed `/login` is launched with the deployment
  provider withheld -- no auth, base URL or model variables and no translation
  proxy -- so Claude Code's credential precedence falls through to the
  subscription; the credential never leaves the user's config volume and is
  never read. Without a login the terminal routes to the deployment provider
  exactly as before, and a dismissable card says `/login` is on offer. `osprey
  health` keeps a failed probe of the agent provider at warning on such a
  deployment, naming who is affected. The dispatch worker always keeps the
  deployment provider. (#925)
- The agent record keeps what the agent was asked, what it answered and what it
  did in one place: prompts, answers, thinking, tool calls with their arguments
  and results, transcripts, safety decisions, dispatch runs, bridge turns and
  feedback, each as one entry. Every deployment runs it as the `agent-record`
  service, and `osprey chat`, `osprey web` and `osprey query` on the host run a
  local one. It is configured in the top-level `agent_record:` block: entries
  are kept for `agent_record.retention_days` (default 14), and each closed day is
  exported as Parquet into `var/archive/<day>/` before that day is deleted, so a
  day leaves the record only once the archive holds it. `agent_record.backend`
  chooses where entries are kept; the default, `embedded`, needs no extra
  service. Claude Code keeps its transcripts on the deployment's volumes for the
  larger of `agent_record.retention_days` and 30 days. `osprey health` reports
  when each writer was last seen and how far the export lags.
- A deployment can turn on code review: a second AI agent reads the Python the
  OSPREY agent writes, on Python runs and on notebook cells, following
  rules the deployment writes in `data/code_review/reviewer.md`, which ships with
  one worked rule for the demo facility on its new `SR:BEAM:USER_MODE` channel, and
  either stops the code or only reports it (`code_review.level`). On a Python
  run the review comes before the approval prompt: code the reviewer blocks never
  reaches the operator, and otherwise the verdict and reasons are on the prompt
  and in the approval preview. The agent gets the reasons; the operator gets a
  line in the activity log.
- The als-apg provider entry lists Haiku 5.5 (`claude-haiku-5-5`).
- An approval prompt's answer is now in the audit trail: a `hook_approval` entry
  records `approved` when the call ran and `denied` when the turn ended without
  it, on the same `tool_use_id` as the prompt, and a control-target switch
  records the target it left and the one it asked for.
- `osprey archive` copies the web terminals' and the dispatch worker's artifact
  stores (files and index) and the plan-queue history into an append-only tree
  with a sha256 manifest, beside the agent record's own daily export. It never
  deletes or rewrites a copy.
- The bundled `archive` service (`template: osprey.archive`) runs `osprey archive
  --watch` once a day into the host directory `var/archive/`, and `osprey health`
  warns when its last pass is stale. `var/archive/` is set to mode 2770 and
  shares the agent record's group, and only the archive service and the agent
  record mount it. The control-assistant preset deploys it.
- The EPICS Archiver Appliance connector can reach an appliance behind a login
  or a site CA: `archiver.settings.auth` names the environment variable that
  holds a bearer token (`token_env`) or a user and the variable that holds the
  password (`username`, `password_env`), and `archiver.settings.tls.ca_bundle`
  names the CA file the appliance's certificate is checked against. The login is
  sent only to the configured host.
- `archiver_read` returns the label of each bool and enum sample beside its
  value, taken from the options the channel roster gives that channel.
- `osprey ariel enhance`, and the enhancement step of every `osprey ariel sync`
  and watch cycle, end with one line counting the entries processed and the
  enhancements that succeeded, failed, and were set aside.
- ARIEL can expand logbook entries through the facility vocabulary before they
  are embedded, so the entry side carries the same canonical phrases the query
  side already adds. Set `ariel.vocabulary.expand_entries: true` to turn it on;
  the default is off and embeds the same bytes as before. The qmd mirror renders
  the same expanded text. `osprey ariel status` counts the stored vectors that
  were built under a different switch, expansion rule or vocabulary than the
  running configuration, and prints the command that rebuilds them.
- `ariel.entry_text.listing_chars` (default 500) sets how much of each entry's
  text search and `browse` results carry, and `ariel.entry_text.read_chars`
  (default 1000) how much `entries_by_ids` carries. `browse` results now carry
  the same 500 characters as search results, up from 300.
- The two logbook presets carry a commented `ariel.entry_url_template`, and the
  configuration reference and the ARIEL ingestion how-to explain it: set your
  logbook's URL for one entry and the agent cites entries as links instead of
  bare IDs.
- ARIEL's `/api/capabilities` names the facility time zone as
  `facility_timezone`, the zone its entry timestamps are rendered in, including
  when the configuration is invalid.
- `ariel.search_modules.keyword.settings.fuzzy_threshold` sets the trigram
  similarity, 0 to 1, a logbook entry needs to be returned when a keyword search
  falls back to fuzzy matching. It defaults to 0.3, the value used before, and a
  value outside that range is refused by name.
- ARIEL: hybrid search takes an opt-in `ariel.search_modules.hybrid.settings.strategy:
  rrf`, which fuses qmd's BM25 ranking with ARIEL's configured semantic provider by
  reciprocal-rank fusion and never calls qmd's reranker. `qmd` stays the default.
- The ARIEL web panel's picture lightbox links the stored original at full
  resolution ("Open original") beside the 1024 px view, and a file card for a
  picture that was not copied or is not shown says why.
- ARIEL can now search logbook entries by their pictures and show a stored picture
  to the agent. With `ariel.attachments.copy_on_ingest: images`, ingest copies each
  entry's pictures into the database; the `image_caption` enhancement module asks a
  vision model to describe each copied picture and list its visible text, and folds
  that caption into the entry's searchable text, so keyword, semantic and hybrid
  search find an entry by what its plots show. The `image_embedding` module embeds
  each copied picture with a multimodal model served by a site-run `llama-server`
  (the new `llama-cpp` provider), and `hybrid` search also ranks pictures against
  the query text. The agent's `attachment_view` tool returns one stored picture,
  and `entry_open` shows an entry, or one of its pictures enlarged, to the operator
  in the ARIEL panel; the panel opens the same view from a
  `#entry?id=<entry_id>&attachment=<attachment_id>` link. `attachment_to_artifact`
  keeps a stored picture's display rendition in the artifact gallery, with the entry,
  attachment, filename and caption it came from; the same picture saved twice in one
  session is one artifact.
  The `control-assistant` preset turns all three on; each
  picture module runs when its server and model answer and is otherwise skipped,
  with `osprey ariel status` naming the skipped module and why.
  A simulation scenario's logbook entries can carry pictures (`attachments`), and
  three of the control-assistant demo entries now do.

  Upgrade notes: an existing `profile.yml` does not gain the new keys, so the
  picture modules stay off until they are added (`osprey validate` lists them as
  drift; `osprey init --revalidate` writes them). For generic file sources, sidecar
  metadata is now fetched only from relative paths under the entry's
  `source_url` directory, and an ingest without an adapter fetches no sidecar.
  The upgrade builds new full-text and trigram indexes and folds captions already
  present upstream into entries' searchable text, which marks those entries' text
  embeddings as owed again; see the picture-search guide for counting and draining
  that backlog.
- ARIEL vocabulary files accept a third concept kind, `synonym`, for a cluster of
  phrasings that all mean the same thing. It is bidirectional and ungated: every
  member of a synonym concept reaches every other member, independent of the
  `canonical_to_acronym` and `canonical_to_shorthand` direction gates that govern
  the other two kinds.
- `GET /api/artifacts` on the artifact gallery takes `run=true` to list only
  the records of script runs (the approval review, the executor's notebook and
  result record), and `run=false` to list all other artifacts. The `total` and
  the page cursor follow the filter. Each artifact that the gallery returns or
  sends on its event stream has a `run_record` field with this verdict.
- `osprey_connectors.dotenv.atomic_replace(path, data, *, mode)` replaces a file
  whole through an unpredictable temporary name; the `.env` writer uses it.
- `osprey scaffold web-terminals lint` and `osprey up` name a roster user whose
  stored password hash in `.env.auth` the login service cannot evaluate, so the
  broken entry is found before anyone tries to log in.
- A Bluesky queue item enqueued by the agent carries the conversation id and
  `tool_use_id` of the call that queued it, and each lane's queueserver records
  every pre-flight reachability verdict as a `preflight` audit entry.
- `osprey build` writes each render's `data/bluesky_devices.yml` from the
  facility file when the render runs a Bluesky lane: one settable per setpoint
  channel, one readable per readback channel, each named by its address, under a
  `schema: osprey.facility.bluesky_devices/1` first line. The Bluesky staging
  copies that file unchanged into the worker's build context, and a mock control
  system stages no copy.
- A channel bridge now announces a delayed answer. When a question was parked in
  the retry queue during an outage and its retry finally completes, the answer
  opens with a "Resuming your request queued at <time> (<elapsed> ago) — service
  is restored, answer follows" line in the same thread, so an answer that lands
  hours after the "queued" notice says which question it belongs to. Every
  bundled channel (Google Chat, Microsoft Teams, Nextcloud Talk) posts it; a
  custom `ChannelOps` adds the new `post_resumed` member.
- The agent in a web terminal reaches that user's own CANOPY. Every project that
  shows the CANOPY panel renders a `canopy` entry in `.mcp.json` that goes
  through the terminal's own panel proxy (`/panel/canopy/mcp`). The read tools
  (`check`, `read_pv`, `search_pvs`, `screenshot`, `get_screen`, `list_drafts`,
  `list_widgets`, `get_widget`, `get_schema`, `list_collections`) and the
  workspace tools (`get_workspace`, `set_workspace`) run without a prompt; the
  authoring tools (`new_screen`, `patch_node`, `add_node`, `remove_node`,
  `convert_legacy`, `delete_draft`, `request_push`) ask on every call and write
  only the user's personal store. No workspace command writes a PV or presses a
  widget. Each `set_workspace` batch is a row in the activity log with the
  status the tool returned. A CANOPY url, whether given to `open_panel` or posted
  to the terminal's panel ops or panel registration, is refused with
  `agent_hop_panel_url`. Such a project carries the `canopy-panel-builder` and
  `canopy-operator` skills, CANOPY's own text bundled byte for byte.
- The design system publishes `canopy-host.css`, which maps OSPREY's design
  tokens and font faces onto the names of CANOPY's host contract (version 1),
  so a CANOPY page that links it after `tokens.css` follows the active OSPREY
  theme. The contract's bar, tab, dropdown and stacking names read the `chrome`
  tokens, so a CANOPY tile bar matches the dock's.
- CANOPY screens can write through the deployment's control session. Each click
  is judged by the execution verdict every write shares: tier, session hold,
  record and switch, with no CANOPY setting. At `asked` the panel asks in place:
  every run shows a confirm, and a write without it is refused
  `approval_missing`; at `allowed` there is no dialog. The panel's writes belong
  to the terminal's sessions: while a watchdog has flagged any of them, the panel
  says "writes held" with the watchdog's reason and refuses `session_held`. A
  switch or a hold reaches the open panel at its next heartbeat; a control target
  move, or a switch moved to or from `off` where the machine has a separate write
  gateway, replaces the data process, which lets admitted writes finish within
  their bound first. The
  panel's 34 s stop grace lets a terminal that is shutting down finish a write
  already sent.
- Each web-terminal user gets their own CANOPY, the operator GUI with machine
  screens, Trace and the correlation and waveform pages, as the builtin `canopy`
  sidecar panel (`web.panels.canopy`). The terminal runs CANOPY's panel server
  and its data process for that user on loopback ports it holds, so a request
  made while the data process is replaced waits for the next one. The data
  process reads every PV through the deployment's control session rather than a
  second EPICS client, says on every socket which machine it serves (the
  target's name, a fallback or unservable notice, the binding id), and keeps a
  channel's kind while it is down. It is replaced at once when the control
  target moves or its transport goes stale, and the panel is
  retracted after five exits in a minute or a refused start. Its not-answering
  statement and write state are the ones the control-target chip shows.

  The panel's settings are `home_screen` (the library screen opened first;
  `osprey build` and `osprey validate` refuse an id the library does not hold
  and list the ones it does), `trace_default_query`,
  `correlation_default_query` and `waveform_default_query` (the view a bare page
  opens with), `theme_file` (relative to the project root, or absolute) and
  `vole_bytes_max`. CANOPY takes its look from three stylesheets the terminal
  hands it: the design tokens, the font faces and `canopy-host.css`. Each user
  has a personal screen store and collection store under their agent data, and
  the deployment's `data/canopy/library` and `data/canopy/collections` are
  served read-only beside them. Trace's and the correlation page's autocomplete
  offer the deployment's channels from one snapshot the build stages at
  `build/web_terminals/channels.json` and mounts read-only into every web
  container. The header command palette lists matching CANOPY screens and
  collections and opens one in the panel, navigating an open viewer in place.
  Only CANOPY's own application pages (`/`, `/s/*`, `/embed/*`, `/trace/*`) are
  exempt from the panel sandbox; drafts stay sandboxed.
- The CANOPY panel ships as a prebuilt payload: the `canopy` and
  `pydmconverter` wheels, the viewer (whose `ir-map.json` is the widget
  catalog), Trace and, when it has one, the staged spec of screen-format and
  widget property schemas, made by the canopy repo's
  `scripts/build-payload.sh`. The `canopy-payload` pre-build step
  (`python -m osprey.integrations.canopy.stage`) copies the payload
  `OSPREY_CANOPY_PAYLOAD` names into the deployment repository, and refuses one
  whose `STAGED.json` and `spec/` disagree or whose commits are not the CANOPY
  and PyDMConverter commits this OSPREY is proven against.
  `python -m osprey.integrations.canopy` reports whether the installed CANOPY is
  at those commits (only a git install at the pin counts), and `--pins` prints
  them. `OSPREY_CANOPY_PAYLOAD_CHECK=0` skips staging and the pin check, for
  development builds only. The image build checks a deployment's CANOPY screen
  library and facility collections with
  `python -I -m osprey.integrations.canopy.image_check library|collections`; an
  image without CANOPY passes both checks and says so in the build log.
- A deployment's CANOPY screen library lives at `data/canopy/library`: screens
  under `screens/` and, beside them, CANOPY's write-binding report
  (`LIBRARY-REPORT.json`, schema `canopy-library-report/1`, and `LIBRARY-REPORT.md`),
  written by `python -m canopy_mcp.report <library> --served-at library
  --write`. `osprey build` refuses a library its report no longer covers, even
  with `OSPREY_CANOPY_PAYLOAD_CHECK=0`: files changed since the report was
  written (the paths are printed), a file that does not belong in it, any
  `.svg`, or a file the container's runtime user cannot read or could write; the
  refusal prints the command that repairs it. A user's agent files a screen for
  the library with `request_push`, and the host reviews it with
  `python -m osprey.integrations.canopy.push list|show|accept|reject`: `show`
  re-checks the bundle in a hardened throwaway container with the image's own
  CANOPY and prints every pushed file, report row and string with a digest that
  `accept` must repeat; `accept` takes a reviewer who may edit the deployment,
  writes the library in an order that an interruption leaves refused rather than
  half-written, logs it, keeps replaced files for 14 days and prints the git
  commands for the admin to run. The build turns the library into the
  `canopy-screens` skill, which tells the agent which screen shows which device.
  The how-to "Author and Review Screens in the CANOPY Library" covers both
  avenues.
- Each machine caption now records the prompt it was made with. After a change to
  `ariel.enhancement_modules.image_caption.prompt_template`, `osprey ariel status`
  reports how many captions were made with an older prompt, and `osprey ariel
  enhance --module image_caption --refresh-stale` captions those pictures again;
  nothing is re-captioned without that command. A picture keeps its old caption
  in search until the new one replaces it, also when the model refuses the new
  prompt. Captions stored before this release record no prompt and are left as
  they are.
- Channel Finder: the graph-mode Explore view sorts on the server by any column, and its columns move, resize, fit and hide, remembered per browser. Rows, facets, chips, column headers and the device card have right-click menus, and the table is driven from the keyboard.
- The Simple view accepts files. Attach a file with the paperclip button, paste
  an image, or drop a file on the composer. The file is stored under the
  agent-data root when you attach it, and the agent opens it by its path when
  you send the message: the render allows the agent to read exactly that uploads
  directory, so no approval card is raised for it. A file of an unsupported
  type, a file over 20 MB, or a file past the store's limit is refused with the
  reason on its chip. A tab closed with files still attached takes them back.
  A file that was sent with a message is kept. A file that was never sent is
  deleted after 24 hours.
- The Simple view copies a finished reply as Markdown from a control under the
  reply. The control shows on hover and on keyboard focus. The session bar has
  an Export button that downloads the whole conversation as a Markdown file,
  with each tool call on one line with its decision and outcome.
- A chat follow-up no longer re-sends a long earlier answer in full: an answer
  over `HISTORY_ANSWER_LIMIT` characters (default 3000) is sent as its opening
  and a note naming its run, and the agent reads the whole answer with the new
  `prior_answer_read` tool when it needs it. Add
  `mcp__osprey_workspace__prior_answer_read` to a chat trigger's `allowed_tools`
  to let its agent read answers back.
- Chat bridges now tell the agent who asked each question, and every remembered
  exchange records who asked it. In Google Chat, Microsoft Teams and Nextcloud
  Talk the agent also sees who is in the conversation and can @mention a member
  when someone asks it to pass something on; set `mentions: false` in the
  bridge's build-profile block (`gchat_bridge`, `teams_bridge`,
  `nextcloud_bridge`) to post mentions as plain text instead.
- The sidebar's control-system block shows the facility clock: the time in
  `system.timezone`, with the facility's city beside it (for example
  `15:11 Los Angeles`).
- The Settings drawer has a Review tab: whether code review is on, its time limit, and the code reviewer's rules (`code_review.prompt`), read-only. An admin-tier terminal can turn code review off there until it restarts, like the safety watchdog; each move is audited, and each skipped review is audited and shown in the session. It says what reviews do while the rules file cannot be used, and an empty rules file now counts as missing. Each review's audit entry also carries a short digest of the reviewed code and the review's time in milliseconds. The rules no longer appear under Safety.
- Every render writes `build/osprey-labels.override.yml`, which gives each rendered service the
  `osprey.project.name`, `com.osprey.repo-id`, `osprey.project.root` and `osprey.config.digest`
  labels, and every compose command OSPREY runs passes it last. A facility service whose template
  leaves these labels out is now listed by `osprey status`, found by `osprey down` and
  `osprey reset`, and recreated on a config change. Bluesky's per-lane Redis and Tiled containers
  gain the same labels, so they are also recreated on a config change.
- `osprey_connectors.ipc.ConnectorHostPool` lets one process drive several
  control-system targets at once, including targets that serve the same channel
  names. It runs one connector-host child per `(target, execution_mode)`, so
  `live`, a read-only `live` and `standin` can sit side by side. Each child is
  started on first use. Before any call reaches it, the pool checks the child's
  endpoint and write posture against what the config derives.

  The pool refuses some targets before it spawns anything
  (`ConnectorHostStartError`, stage `"config"`):
  - a Channel Access block (`epics`, `virtual_accelerator`, `live_standin`) with
    no gateway for the role the run selects, whose child would search by
    broadcast;
  - a `standin` or `va` target whose selected gateway has the same address and
    port as one of the `live` target's gateways.

  Failed calls are never retried:
  - A child that dies with a call in flight fails that call with
    `ConnectorHostLostError`, and the next call gets a fresh child.
  - A child that misses a call's deadline is pinged first. One that still
    answers is left running, and the call raises `TimeoutError`.
  - A child that doesn't answer the ping is killed, and the call raises
    `ConnectorHostUnresponsiveError`, which is also a `TimeoutError`.
  - With `kill_on_write_timeout=True`, a write that misses its deadline kills
    its child even if it answers the ping, and raises
    `ConnectorHostWriteTimeoutError` (a `TimeoutError`). The write may or may
    not have landed, but the child can no longer send it after a newer write.
    With the flag off (the default), a timed-out write may still land later.

  The pool never loads a Channel Access or pvAccess client itself.

  The connector-host proxy and child now also serve `validate_channel`, and the
  proxy offers `write_channel_checked`. The endpoint derivation and post-connect
  verification behind the target switch are exported from
  `osprey_connectors.ipc`, so the switch and the pool share one
  check.
- The `control-assistant` preset shows the CANOPY panel and opens on it at
  `facility/overview`, selects the `canopy-panel-builder`, `canopy-operator` and
  `canopy-screens` skills, and stages the payload in a `canopy-payload` pre-build
  step; a build that names no payload leaves the panel out and says so. Screen
  writes follow each login's write switch, as the agent's do: at `asked` the
  panel asks before each write, at `allowed` it writes, and a readonly login's
  panel is inert. Its default collection
  opens the overview, orbit feedback and three tools each on a view of its own:
  Orbit in Trace, RF cavity 01's temperature against its reflected power on the
  correlation page, and the septum pulse on the waveform page with the last 10
  shots overlaid. A repo made before this release keeps its profile: add
  `canopy` to `web_panels`, the three skills and the `canopy-payload` step by
  hand.
- The control-assistant template ships a CANOPY screen library for the example
  facility, with its generated library report: a facility overview whose
  plan opens GTL (`gtl/GTL`, gun through linac and the LTB dogleg drawn as one
  line), the booster (`br/overview`), the BTS (`bts/BTS`) and each storage-ring
  sector; twelve sector screens (`sr/sector-01` to `sr/sector-12`) with the
  lattice, the orbit at each BPM as X and Y bars, the sector's peripherals in
  beam order and its vacuum and absorber sub-screens; and storage-ring orbit
  feedback. Embedded screens and images the screens share live under
  `common/`.
- A personal card behind a login wall can set `control_identity` in the roster.
  Its container then gives uid 1000 that account name at start, so every write
  the card makes (through the connector, the Python executor, a notebook kernel
  or a raw client library) reaches the control system under that name rather
  than `osprey`. The audit trail records `ca_user`, `ca_host` and, for shared
  writers, `owner` on each write route so a gateway put-log line can be joined
  back to the person, notebook kernels file one `allowed` record per channel
  written, and `osprey health` checks the name every card and dispatch worker
  writes under. Lint and the build refuse an unusable name, a name on a shared
  card, and two people on one name.
- `osprey.runtime.session.ControlSession` gives a long-lived process, such as a
  panel backend, one binding to the deployment's control target. It reads the
  control-context record once and builds one connector for the target it names,
  reports an absent or unservable control system as a state rather than an
  error, names its target by the display name every other surface uses, and
  tells each subscription its channel's connection whenever it changes. The
  process exits with a dedicated status so its supervisor starts a fresh one:
  `EXIT_BINDING_MOVED` (75) when the record or the bound target's switch
  position moves, `EXIT_TRANSPORT_STALE` (76) when the connector reports its
  transport stale, and `EXIT_START_REFUSED` (78), never respawned, for an
  environment refused at start.

  `ControlSession.write()` writes through the bound connector in one fixed
  order: an in-flight marker a target switch names when it refuses, the target
  pin, the binding state, the execution verdict, the switch gate, an `attempt`
  audit record, the checked write and one outcome record. Whether this login
  may write is the execution verdict shared with every other write path, the
  session's hold included; a refusal names the verdict's reason in the
  verdict's phrase, and a hold on the session pauses its writes in place
  without a restart. `writes` states the verdict's own terms (`allowed`,
  `position`, `reason`, `kind`). A session write's audit records carry the
  value written as `value_len=` and `value=` detail tokens, the one exception
  to the envelope's rule; `value=` is a JSON string, so a value cannot forge a
  token.
- `web.control_target_picker: false` takes the control-target picker out of the web terminal's title bar. It is on by default.
- The demo facility now covers the whole injector: the gun-to-linac line `GTL`,
  the linac `LN`, the linac-to-booster line `LTB` and the booster-to-storage line
  `BTS` gain their magnets, monitors, screens, gun and RF stand-ins, and the
  booster gains its injection kicker, extraction septa, screens and energy-ramp
  setpoints. One readback gate chain runs in beam order from the gun to storage
  ring injection: while the gun's high voltage or heater is off, or any upstream
  screen paddle is in, every BPM sum and charge reading downstream drops to a
  hundredth of its value and every BPM `STATUS:VALID` reads `FALSE`; a screen's
  lamp follows its own paddle, and the "all paddles out" count follows every
  paddle. `scripts/facility_demo/panel_rename.py` re-maps the GTL and BTS CANOPY
  panels from legacy names to the demo channels: `check --facility` confirms
  every mapped channel exists, and `apply --library` rewrites the panels'
  bindings in place.
- The control-assistant demo gains `LINE`, a synthetic transfer line solved
  single-pass: 20 devices and 40 `LINE:` channels beside the periodic model `SR`.
  The simulator now serves `[LINE, SR]`, so it publishes one status channel per
  model (`<code>:SIM:LINE:STATUS` and `<code>:SIM:SR:STATUS`), `osprey sim status`
  prints a line for each. The hierarchical and middle-layer channel-finder indexes gain the 40
  `LINE:` addresses; the in-context index and `data/facility/limits.yaml` are
  unchanged, so a `LINE` setpoint is written with no limits record. The benchmark
  query "Show me the transfer line corrector setpoints for both planes" now names
  the booster-to-storage transfer line; the ambiguous-category query "Show me the
  second focusing quad setpoint in the transfer line" is kept as is.
- Facility seeds gain `pulse`, which gives a waveform channel a trace scaled by a
  driver channel and switched by an enable channel, and `gate`, which holds a
  channel at its `gated` value while any input reads its blocking label; the
  build checks both and refuses a gate cycle. The Virtual Accelerator's archive
  stores waveform traces in a sibling `<collection>_waveforms` collection once a
  minute, the MongoDB archiver connector reads them raw as `list[float]`, and
  `archiver_read` returns them unbinned within a 32,000-point budget per read.
- The control-assistant template ships its scenarios under `data/facility/scenarios/`,
  one `<name>.yaml` per scenario: a `description`, `overrides`, `faults` per model and
  address, `archiver`, `logbook`, and the shared-driver blocks `drivers`, `couple` and
  `noise`. A per-element monitor error becomes the same fault on the element's x and y
  readings, and a corrector gain becomes its setpoint's `cal_factor`. The build now stops
  on a scenario override out of its limits band, on a locked setpoint, or on a channel a
  model computes; on a fault value out of its range; on a fault field the address does
  not carry, including `roll` on a y-axis reading; and on a fault key that names neither
  a channel nor a variable of the model's engine.
- `osprey deploy` checks once, before building, that Docker has BuildKit (the
  buildx plugin) and stops with how to install it; podman needs nothing.
- Design system: a shared right-click menu (`context-menu.js`) and movable, resizable table columns (`table-columns.js`) for any panel.
- The design system has semantic surface tokens (`--surface-canvas`,
  `--surface-card`, `--surface-well` and the hover, press, selected and focus
  washes) in all eight themes, and shared shell primitives: `primitives.css`
  with sidebar rows, list rows, popovers, a tooltip layer, buttons, status dots
  and the pane header, plus the `icons.js`, `popover.js`, `tooltip.js`,
  `collapsible.js` and `sidebar-boot.js` modules. The new contributor page
  "Design Language" gives the rules.
- A `source: cron` trigger can fire at clock times: `at: ["07:45"]` with an optional `days: [mon, tue, wed, thu, fri]`, read in the facility zone (`system.timezone`). A time that daylight saving skips fires at the first minute after the jump, a repeated time fires once, and a time the dispatcher was not running for is not made up. The dispatch dashboard shows each clock trigger's next fire.
- The event dispatcher records who fired each run. A fire or re-fire made
  from the web terminal stores the person's name as `owner` on the trigger's
  history entry and as `requester` on the run's entry in the agent record; a
  webhook, cron or chat-bridge fire is stored without one. The dashboard's run
  detail shows it in Expert mode.
  Cancelling a run and enabling or disabling a trigger are logged with the name
  of whoever asked.
- The configuration reference documents `screen_capture.output_dir`, the
  directory screenshot captures are written to, and `osprey config --defaults`
  now lists it.
  The build profile reference now says how a lifecycle step gets its tests
  printed as a results table: write JUnit XML to
  `{project_root}/check_results.xml`.
- The controls server's gateway probe stamps `down_since`, the moment a gateway's
  current run of unreachable sweeps began, beside each reachability row.
- `control_system.connector.<type>.fresh_reads` is accepted by EPICS-family
  connector blocks. Every Channel Access read already asks the IOC rather than a
  monitor cache, so an IOC that computes its readbacks on get is read fresh
  whatever the key says; the key is kept, and validated, so a configuration that
  sets it per block (a stand-in simulator but not the live machine) keeps loading.
- A new `event_dispatcher` MCP server fires a dispatch job from a web terminal:
  `manual_fire` asks for approval, the job carries whoever fired it and a
  cron-fired job nobody. A trigger may not name a dispatcher tool, so a job
  cannot fire jobs.
- The example facility's storage-ring sectors carry their peripherals: 72 ion
  pumps, straight and arc gauges with trip and reset, two gate valves per
  sector, 14 insertion devices named by kind and number (U1, EPU7, W4, IVU5),
  front-end shutters, absorber thermocouples, scrapers, the tunnel environment,
  harmonic cavities and the septa's interlock words. It adds the beam-lifetime
  readback and an operation-mode channel that reads `Machine studies`, matching
  `SR:BEAM:USER_MODE`, runs the stored current as top-off, and starts with
  magnet and vacuum fault flags clear and gate valves open. The `rf-thermal`
  beam dump ends, so the stored current recovers to the top-off sawtooth.
- `osprey build` builds the facility file once from the profile's
  `data/facility/`, before the project environment and every render, and writes
  it as `facility.json` at the root of every render: the deployment's own, each
  tier's and app's, and each container image's. A facility error stops the build with its
  one line before anything is installed or rendered.
- `osprey facility import mml EXPORT...` writes MML exports as the mml layer's
  sources under `data/facility/imported/mml/` and seeds each authored file that
  does not exist yet. The mapping is checked against the exports first: a mapping
  with a problem prints one `<key>: <message>` line per problem and
  `<n> problems in <path>; fix each and check again.`, writes nothing and exits 1.
  The verb takes every export the mapping names: a system the mapping names and
  no given export carries is one such problem, `models.<system>: <system> is no
  exported system`. A mapping with the wrong structure, or a profile that does
  not resolve, prints a one-line failure with its cause and exits 1.
  While `data/facility/` holds an authored record source that would merge
  against the layer — a file of `records/` or `decks/`, `models.yaml`, or a
  `seeds.yaml`, `limits.yaml`, `identity.yaml` or `measurement/` file that does
  not open with the layer's header line — the verb prints
  `import mml: authored-present: <n> files` (`1 file` for one) and one
  `rm <path>` line per file and exits 1; `fixes.yaml`, `classes.yaml`,
  `scenarios/` and `knowledge/` are never named.
  `--print-exporter` prints the MATLAB exporter the layer ships and needs neither
  a repo nor an export. A wired setpoint whose export `Range` lacks a finite edge
  is seeded into `limits.yaml` as `writable: false` beside whichever edge is
  stated, and a channel a write field retypes as a setpoint takes that field's
  unit and description.
  After the import it prints, as `rm` lines, each scenario file that names a
  channel or model the facility no longer has, and deletes none.
- `osprey facility validate` runs every check `osprey build` makes of
  `data/facility/` and renders the facility file against the repo's main profile
  in a temporary directory, writing nothing into the repo. A clean tree exits 0
  and prints nothing; otherwise every error of the first failing stage prints on
  stderr, one line each and sorted, and the command exits 1. A stale fix's line
  carries the block to paste in its place.
- A profile whose `project/` mirror ships `data/facility_facts.json` or
  `data/facility_facts.md` stops `osprey build` and `osprey facility validate`
  with one `facility: profile-invalid:` line naming `data/facility/` as the tree
  the facility is authored in. Agent-side writes to those two paths are refused.
- `osprey build` writes each render's `data/facility_facts.json` and
  `data/facility_facts.md`: the facility's identity, place levels, device classes
  with their aliases and families, models and channel count. The agent context
  takes the facility name and these facts from that file, and a render without
  one is read as a facility with no sources, named after the project.
- The channel-finder servers and the channel-finder web app take the facility
  name from the render's facility file (`facility.json`), and fall back to the
  project name when the render holds none.
- `osprey facility import mml` holds each export's deck to the lattice fingerprint the export states. A deck with another element count, family-name digest or parameter-element indices stops the import with `import mml: export-invalid`, naming the fact, both values and the remedy; an energy that differs alone is reported and the import goes on.
- MML exports now enter the facility description through
  `osprey facility import mml`; the how-to "Import a Middle Layer export" walks
  the import.
  `osprey facility validate` prints the response
  check of each kept response export.

  The response check's entry-by-entry bar is exercised on the nsls2 fixture's
  model-derived matrix: a measured matrix is judged by its median size ratio and
  sign agreement, not entry by entry, so a single wrong entry is caught only in a
  block the export computed from a model.

  The spear3, nsls2 and synthetic MML fixtures are re-exported with
  `mml_export.m` 2.1.0 and each gains its `<stem>.model.json`.
- A facility description takes its records from two importers beside the authored files: `osprey facility import mml` writes a MATLAB Middle Layer export under `data/facility/imported/mml/`, and `osprey facility import list` writes a CSV channel list under `data/facility/imported/list/`. `osprey build` merges every layer with the authored files field by field, and two layers stating different values for one field stop the build with `layer-conflict`. No output of an existing deployment changes.
- An imported MML model wires its tune readback to the engine's tunes: the
  mapping's `tune` block names one address per plane, each read as a scalar, or
  one address read as a waveform of every plane, typed `value_type: waveform`.
  The pyat engine serves a waveform readback wired to `tune` or `chromaticity`
  as the whole output, read-only; a non-float setpoint, or a non-float readback
  wired to no array output, still stops the build. The seeded measurement file
  reads the tune on that readback and allows the tune response.

  The import wires the energy knob to the deck energy through the export's
  energy table, its readback following the setpoint read-only; wires every
  member's readback of a shared supply; wires a family with no stated hardware
  nominal from the deck where neither a turning conversion nor a series needs
  the nominal; carries a device's polarity on its slice where the deck holds the
  other sign; and seeds `scenarios/readout.yaml` once from the export's monitor
  readout, naming each readout it cannot carry. A monitor's offset that the
  export's conversion holds is served through the wiring, so the Middle Layer's
  correction, Gain x (Raw - Offset), gives back the model's position; an offset
  the conversion is not seen to hold, and a gain other than 1, is not carried.
  The draft mapping proposes the energy knob, a correcting dipole string and the
  cavity, and a signal role for each magnet current and monitor position field;
  the import writes that role on each channel and keys its group's sentence by
  it. A readback several setpoints share pairs none of them.

  A view stop on a record only `fixes.yaml` produced names `fixes.yaml`.
- A profile whose `project/` mirror carries `facility.json` or anything under
  `data/facility/` stops `osprey build` and `osprey facility validate` with one
  `facility: profile-invalid:` line naming `data/facility/` as the tree the
  facility is authored in. `osprey facility validate` writes nothing outside its
  scratch render.
  A tier or app file that names `data:` stops with a `facility: profile-invalid:`
  line: every tier and app shares the facility tree the root `profile.yml` names.
- `osprey facility validate` holds each response export an mml import kept
  (`imported/mml/<model>.response.json`) against the model the facility file
  describes and prints one `response check <model>:` line per model to stderr,
  naming the judged block nearest its bar. A block the export computed from a
  model needs 0.99 of its entries inside the 5 % band; a measured block needs a
  median size ratio in 0.8 to 1.25 and sign agreement of 0.95. A failing check
  exits 1; nothing is written.
- A logbook entry in a facility scenario file (`data/facility/scenarios/<name>.yaml`)
  can carry pictures through an `attachments` list, each `{path: <picture>}` or
  `{plot: <plot spec .json>}` relative to `scenarios/<name>/`. `osprey build`
  copies the attached files into the render's simulator view, and the deploy-time
  logbook seed hands them over with their entries. The control-assistant demo's
  facility scenarios carry the same three pictures as its simulation bundles.
  An attachment that names a missing file, leaves its scenario directory, does not
  hold the picture its suffix names or is not a `.json` plot spec stops the build
  and `osprey facility validate` on one line, and `osprey facility import mml`
  lists a stale scenario's folder of attached files beside its file.
- A setpoint whose limits band excludes 0 and that has no seed stops the build
  with one `facility: seed-missing:` line naming the channel and the nominal to
  add for it in `data/facility/seeds.yaml`.
- `osprey facility show [--json] [ID]` builds the facility in memory, as
  `osprey facility validate` does, and prints its identity, the record count per
  kind and the wiring count per model, each model's engine, served flag and solve
  setting, and each view of the main render with its path and whether it is
  written, naming the config key or model fact that leaves a view out. With an
  ID it prints every record of that id, each under its kind, with its provenance
  and the fixes applied to it; an id that names no record exits 1 with one line
  on stderr. `--json` prints one JSON document on stdout — under an ID, a
  `records` list — and every other line on stderr, and leaves stdout empty on an
  error.
- The config key `simulation.models` selects which models of the facility file
  the simulator serves. `null` or absent serves every model; `[texture]` and `[]`
  serve no physics. A name the facility file does not hold is a `profile-invalid`
  stop, and so is a tier or app file that sets the key while its baseline
  target is a VA instance, whose container serves the deployment's list. Every
  shipped preset sets `simulation.models: null`. No reader consumes the list yet.
- A profile whose `project/` mirror ships a file under `data/simulator/` stops
  the build with one `facility: profile-invalid:` line naming `data/facility/` as
  the tree the facility is authored in. Agent-side writes under `data/simulator/`
  are refused.
- The build writes variables.json, seeds.json and scenarios.json into data/simulator/.
- Every render of `osprey build` now writes the simulator view into
  `data/simulator/`: `served_models.json`, `addresses.json` and a byte copy of
  each deck-bearing model's deck under `decks/<model>.json`. No reader consumes
  them yet.
- The simulator view exposes each model's wiring entries with direction, unit, default and value_range.
- The control-assistant, ARIEL and channel-finder presets and hello-world now ship
  their facility description as committed sources under `data/facility/`, and
  `osprey build` writes the facility file built from them as `facility.json` at
  every render root. `osprey facility validate` checks that tree and renders the
  facility file without writing anything into the repo.

  The control-assistant demo serves four addresses beyond its previous set: the readbacks
  `SR:DIAG:TUNE:X`, `SR:DIAG:TUNE:Y`, `SR:DIAG:CHROM:X` and `SR:DIAG:CHROM:Y`,
  computed from the deck's optics. The first cavity's existing
  `SR:RF:CAVITY:01:FREQUENCY:SP` and `SR:RF:CAVITY:01:FREQUENCY:RB` are now wired
  to the deck's cavity frequency (MHz on the channel, Hz in the deck); the second
  cavity stays unwired.

  The demo's `data/facility/limits.yaml` holds exactly three teaching records and
  no defaults block: `SR:MAG:HCM:01:CURRENT:SP` bounded to [-12.0, 12.0];
  `SR:RF:CAVITY:01:FREQUENCY:SP` bounded to [500.0, 500.8] MHz with a largest
  single step of 0.01 MHz, a band re-based around the deck's 500.417 MHz where
  the packaged `channel_limits.json` carries [499.0, 500.3]; and
  `SR:VAC:ION-PUMP:01:VOLTAGE:SP` not writable. Every other channel follows the
  deployment's limits setting. The build still ships the packaged
  `channel_limits.json` until the limits view renders `limits.yaml`, so the
  re-based cavity band, and any record a deployment adds to `limits.yaml`,
  reaches a render only from that view.

  The channel-finder standalone preset serves more channels. New SR families
  `SR:MAG:QFA`, `SR:MAG:SHD` and `SR:MAG:SHF` (devices 01-24, each with
  `CURRENT:SP/RB/GOLDEN` and `STATUS:ON/READY/FAULT`) and the `SR:DIAG:TUNE:X/Y`
  and `SR:DIAG:CHROM:X/Y` readbacks join it; `SR:DIAG:BPM`, `SR:MAG:HCM` and
  `SR:MAG:VCM` grow from 20 to 72 devices, `SR:MAG:DIPOLE` from 24 to 36,
  `SR:MAG:QD` and `SR:MAG:QF` from 16 to 24, and `SR:MAG:SD` and `SR:MAG:SF` from
  12 to 24. Every address the preset served before is still served.
- The web terminal, its pop-out activity page and the artifact gallery carry the
  facility time zone from `system.timezone` on `<html data-facility-timezone>`,
  the same zone the agent is told.
- Facility: the build writes the knowledge graph view `data/graph/facility.ttl` from the facility file, its first line `# osprey:facility-sha256 <hex>` naming the facility file's hash. Every device carries its place path, section code, position and length in metres and its ordinals within its place and model; every channel is a binding joined to its signal by its role.
- `health.disk.min_free_gb` and `health.disk.max_used_percent` set when the
  `disk_space` health row warns (defaults 1.0 GB free and 90 % full), so a large
  shared volume that runs full by design no longer leaves `osprey health`
  degraded. An invalid value is refused at load time and names the key.
- `osprey build` writes `data/channel_finder/hierarchical.json` from the facility
  file when a render selects the hierarchical pipeline, under
  `"schema": "osprey.facility.channel_finder/1"`. Its levels are the facility's
  place level words, then class, device and leaf; every channel sits at the same
  depth, and an absent place, class or device is the node `-`. Each leaf spells
  its channel's full address and carries the sentence its device's family group
  keeps for that kind of signal, else the channel's own description. The
  hierarchical pipeline loads that file, and
  `channel_finder.pipelines.hierarchical.database.path` renders to it. The keys
  `channel_finder.pipelines.hierarchical.database.type` and
  `channel_finder.pipelines.in_context.database.presentation_mode` are gone. A
  render that selects another pipeline writes no hierarchical index and prints no
  line about it.
- Under web-terminal sign-on, the history page at `/history/` shows each person
  the agent record of their own sessions, or every user's with `history: all` on
  their roster entry or role. It is served on the history page's origin
  (`modules.web_terminals.history_port`, or `modules.web_terminals.history_origin`
  behind a load balancer), so pages on the terminals' origin cannot read from it.
  People named under `agent_record.readers` read there too, each limited to the
  entry kinds and days their entry allows.
- The facility vocabulary gains `gap_setpoint`, `gap_readback`, `offset_setpoint`
  and `offset_readback`, so an insertion device's gap and row-offset setpoints
  pair with their readbacks like every other quantity.
- `osprey facility import mml` prints one line per family whose sampled calibrations it cut to the stretch holding the operating point (`import mml: curves trimmed: <model>: family <family> (<n>)`) and one per family whose supplies feed several devices in series (`import mml: supplies folded: …`), with the count. The files it writes are unchanged.
- `osprey build` writes `data/channel_finder/in_context.json` from the facility
  file when a render selects the in_context pipeline: one row per channel tagged
  `in_context`, sorted by address, named by the channel's first name (else its
  address), with its address and description, under
  `"schema": "osprey.facility.channel_finder/1"`. The in_context pipeline loads
  that file with the flat loader, and `channel_finder.pipelines.in_context.database.path`
  renders to it. The `channel_finder.pipelines.in_context.database.type` key is
  gone. A render that selects in_context while no channel is tagged stops with
  `facility: view-unsupported`.
  A render that selects another pipeline writes no in_context index and prints no
  line about it.
- The setup skill asks which model runs each agent. In BUILD, a MODELS card
  lists every agent the deployment selects on the main model, and the user pins
  any of them to an id the provider serves. DISCOVER lists a deployment's
  existing pins, and MAP ports each one the provider still serves.
- An `openrouter` provider reaches models from many vendors with `OPENROUTER_API_KEY`, and `osprey.models.jev` adds a client for the Jev structured-decision API.
- The watchdog decision engine can run on a self-hosted model: `api.jev.route: openai` sends each question to an OpenAI-compatible chat server (a stock vLLM) as one constrained one-token completion and reads the option probabilities from its logprobs. Set `api.jev.routes.openai.base_url` and `model`; `api_key` may be omitted on a loopback or private address.
- Knowledge pages link facility channels and devices with
  `[label](facility:ADDRESS)`. The build warns on a link the facility file does
  not hold, and the KNOWLEDGE panel shows the linked record.
- `osprey build` warns, once per page, when a knowledge page under `data/facility/knowledge` names a `device_id` in its frontmatter that is no device id in the facility file. The build goes on; a page without the key is not linked.
- A `type: users` section of the web-terminal landing page takes `names: hidden`,
  which replaces its name cards with one "Log in to your terminal" button, so the
  page no longer lists who works there. Service trays and link sections are
  unchanged, and `names: shown` stays the default. `osprey build` refuses
  `names: hidden` unless `auth.method` is `password` or `oidc`, since under
  `token` or `none` the name card is the only way in.
- The documentation landing page plays a short demo video of one agent session
  on the simulated control-assistant demo, in light and dark, with speed
  controls and a clock showing the real elapsed time. `make -C docs demo-video`
  records it again and `make -C docs upload-demo-video` publishes it per release.
- **Breaking change:** `control_system.limits_checking.allow_unlisted_channels` is
  replaced by `control_system.limits_checking.mode: exclusive | optional` (`false`
  is `exclusive`, `true` is `optional`), and a config that still states the old
  key fails the build. The `data/channel_limits.json` that `osprey build` renders
  holds only the records in `data/facility/limits.yaml`; a channel with no record
  follows the mode. The limits, the agent facts and the facility's display name
  all come from the build's facility sources, and the profile's
  `dispatch.facility_name` and the config's `facility.name`, `facility.timezone`
  and `facility.ontology` are gone. Re-baselined on purpose: the
  `control-assistant` render's limits database carries the three records its
  `limits.yaml` holds (`SR:MAG:HCM:01:CURRENT:SP`,
  `SR:RF:CAVITY:01:FREQUENCY:SP`, `SR:VAC:ION-PUMP:01:VOLTAGE:SP`) in place of
  one entry per served address, and the `hello-world` render's carries its three
  records with no `defaults` block.
- `osprey facility import list FILE` imports a CSV channel list as the list layer's sources under
  `data/facility/imported/list/`: a header row names `address` and any of `role`, `pair`, `device`,
  `place`, `unit`, `description`, `tags`, `s` and `model`, and a file without a header holds one
  address per line. A row's `s` and `model` are its device's position, which `osprey build` places
  by the span that contains it. A `model` stated for a device no row
  states `s` for is ignored with a warning naming the file, row, device and model.
- Multi-user web terminals: `modules.web_terminals.auth.throttle` sets how failed
  password logins are slowed — `initial_delay_s`, `multiplier`, `max_delay_s` and
  `forget_after_s`, defaulting to the previous fixed 1 s doubling to 30 s,
  forgotten after 300 s quiet. Logins are slowed, never locked out; a key left out
  or written with no value takes its default, and `osprey build` refuses a value
  the throttle cannot use.
- `osprey build` writes `data/channel_finder/middle_layer.json` from the facility
  file when a render selects the middle-layer pipeline, under
  `"schema": "osprey.facility.channel_finder/1"`, and writes the DuckDB copy
  `run_sql` queries beside it as `data/channel_finder/middle_layer.duckdb`. A
  System is a top place and a Family is a group, named by its id less a leading
  `<System>/`; each Field lists one channel per member in
  `CommonNames` order, so `ChannelNames` lines up with `DeviceList`, and carries
  the family's sentence for that field. A channel whose field some member lacks
  or holds twice is its own Field keyed by its address; a channel of no family is
  left out, and the build names both counts in one note. A facility with no group
  stops the build with `view-unsupported`, as does a host where DuckDB cannot load
  its full-text-search extension.
  `channel_finder.pipelines.middle_layer.database.path` and `duckdb_path` render
  to the two files, and `channel_finder.pipelines.middle_layer.database.type` is
  gone. A hierarchical selection now stops with `view-unsupported` when a tree
  key begins with `_`.
  The middle-layer terminology table names each family as the index files and
  names it, with its System where a class spans several.
- A measurement file's group role may name a list of groups, taken together as one pyAML array named after the role. The MML import lists the families the export tags as tune and chromaticity correctors and writes every measurement kind its groups and instruments allow, chromaticity response and chromaticity monitor included. The example measures both quadrupole and both sextupole families, so its pyAML view carries the tune and chromaticity correction tools. A group knob in a measurement result is labelled with its array name and the list of its groups.
- The MongoDB archiver trusts a site certificate authority named by
  `tls.ca_bundle`. Without it, a `tls=true` connection string checks the store's
  certificate against the container's trust store.
- The MongoDB archiver can name its store by `url`, a MongoDB connection string,
  so it reaches TLS, replica-set and x509 stores. The url wins over `host` and
  `port`, may not carry a password, and is only ever read from: the recorder and
  the archive rewrite write only to the store the deployment runs.
- Single sign-on can ask the identity provider to put the claims the login
  reads into the ID token. New key
  `modules.web_terminals.auth.oidc.claims_in_id_token: true` makes the login
  route send the OIDC `claims` request parameter naming the identity claim
  (essential), `email_verified` and any role-binding claim (voluntary). It is
  the fix for a provider that follows OIDC Core §5.4 strictly and serves
  scope-requested claims from UserInfo, which the sidecar never calls: every
  login failed with "the ID token carries no usable claim" although the scope
  was requested. Off by default; with it off the authorization request is
  unchanged.
- Removing something in the web terminal with one click can be undone for a few
  seconds.
- The web terminal has an OSPREY logo: a pixel osprey drawn on a 16 px grid. It
  is the browser tab icon, on a light tile so that it shows on light and dark
  tab bars. It also sits to the left of the OSPREY name in the header. In the
  Simple view the osprey beats its wings on the turn's live status line while
  the agent works. When the agent stops thinking, the osprey there beats its
  wings twice and does a full roll. History that the view loads again does not
  play the roll, and the reduced-motion setting stops all of it.
- A person can log in to a multi-user deployment at `/auth/enter` without choosing a card on the landing page. They are sent to their own terminal, or shown each terminal they may open when there is more than one.
- A plan queued through OSPREY carries the person who queued it, and obeys
  their control-target chip at every write in the lane's queueserver and the
  dispatch worker. A plan that reached the queue any other way, and every
  external-worker lane, runs at the lane's ceiling.
- A setpoint states how close its readback must come before a move counts as done: `tolerance: {absolute: <x>}` in the channel's unit, or `tolerance: {relative: <f>}` of the demand, next to its `pair`. The build stops when a seeded readback moves wider than its setpoint's tolerance and warns when a scenario does. The Bluesky device file carries the tolerance, and the worker settles within it on every lane; a setpoint without one settles within its readback's motion band (`motion_band`) on a lane that serves the simulator, and within the profile's floor on a live or stand-in lane, with one build warning per such lane naming the count. The Middle Layer import fills `tolerance` from each setpoint's `Setpoint.Tolerance`, and the example facility's setpoints carry per-class tolerances.
- The command palette finds the Settings drawer's files: agents, skills, rules, CLAUDE.md, hooks and the code reviewer's rules. Picking one opens it in Settings.
- A sidecar panel that opens its MCP transport to the agent gets its `.mcp.json`
  entry from that declaration alone: `registry.mcp.derive_panel_servers()`
  builds a URL server named after the panel id that dials `/panel/<id><mcp_path>`
  on the terminal's own proxy with the panel token and the calling session, takes
  `mcp_allow` / `mcp_ask` as its permission lists, and gates each ask tool with
  the approval prompt — behind the writes check for the tools the panel names in
  `mcp_writes`. The entry is rendered only for a project that declares the panel,
  and `claude_code.servers.<id>.enabled: true` cannot switch it on without it.

  Alongside it: a `url` or `command` set on a framework server's name in
  `claude_code.servers` is reported as ignored instead of dropped silently;
  headless runs outside a web terminal no longer wait for MCP servers that
  authenticate with the panel token; and the warning for a profile-declared
  `Bash` or `Monitor` gate names every panel transport the panel token reaches.
- An embedded panel can hand keyboard focus back to the web terminal by posting
  `{type: 'osprey-focus-exit', direction: 'forward' | 'backward'}` to its parent;
  the terminal moves focus to the element before or after the panel's tile.
- A sidecar panel may name `image_checks`, shell commands the image build runs from the deployment directory for every profile that selects the panel; a failing check fails the build.
- Panel pages the web terminal relays carry the facility time zone from
  `system.timezone` on `<html data-facility-timezone>`, the zone the agent is
  told, including panels whose own server reads no configuration.
- Web terminal panels pop out into their own browser window. The **Pop out**
  button in a tile's header, the **Pop out** row in a sidebar row's menu and the
  **Pop out <panel>** palette action open the panel as a popup the size of its
  tile; the tile leaves the workspace, the sidebar row shows a window glyph, and
  the window follows the terminal's theme, view and session. Closing the window,
  or **Bring back** in the row menu, puts the tile back beside the active one. A
  focus from the agent brings the window to the front, a reload of the terminal
  reconnects the open windows, and a blocked pop-up leaves the tile where it is
  with a notice. The one-way **Open in a new window** action is replaced.
- A how-to page covers connecting the agent to live Phoebus displays: enabling the `phoebus`
  server, registering panels, and what guards `phoebus_drive`. The `phoebus.*` keys and the
  `PHOEBUS_*` variables it reads now have reference rows.
- The provider table in the LLM provider guide gains a Hosting column that says
  whether a provider is a cloud service or a server your site runs itself. Each
  provider adapter declares it as `self_hosted`; `ollama`, `vllm`, `ds4` and
  `llama-cpp` are self-hosted.
- The translation proxy for OpenAI-protocol providers sends images to routes that
  take them, including images a tool returns. Content a route cannot take is
  replaced by a note the model sees, and each kind is logged once per
  conversation. Local model servers take no images unless their `providers.yml`
  entry says `supports_images: true`.
- Web terminal: a rail entry whose panel stops answering its health check for
  about 20 seconds dims until the panel answers again. It stays clickable, and its
  tooltip says "not answering since HH:MM".
- Web terminal: the terminal tile's right-click menu has a "Reset layout" row that
  puts the current view's default arrangement back.
- The web terminal's command palette offers **Reset layout**, which restores the
  current view's default tile arrangement.
- A pyAML write outside a `run_tool` call, `execute` included, is refused as
  `OspreyWriteRefused` ("pyAML writes run only inside pyaml_measure") before
  anything is read or written.

  `pyaml_cs_osprey.run_tool.run_tool` calls a pyAML tool method under the
  target's run lock, joining the lock the process already holds: a second run on
  the target is refused busy before the tool starts, each setpoint's value is
  read before its first write, and the call returns a `ToolRun` whose `moved`
  lists the setpoints the runtime sent or landed, each with its value before the
  call; a refused write lists nothing, and a write whose outcome is unknown (the
  call raised for another reason, or was interrupted, before its outcome came
  back) is listed. An escaping exception carries the same
  list as `moved_setpoints`. Nothing is written back.
  A Ctrl-C stops every tool of the run that takes a callback, in any thread,
  through pyAML's own abort path. A run that reaches the execution timeout is
  killed like any other, with no early stop.

  `pyaml_cs_osprey.measure.StepMeasurement` measures what pyAML's own tools step
  only bipolar or one magnet at a time: a unipolar orbit response with a step per
  corrector, a unipolar dispersion, and tune and chromaticity responses that step
  each group of magnets as one knob, every write inside its `measure()` call.

  The `pyaml-cs-osprey` package is built, installed and released with the
  framework: images, `--dev` wheel staging and a source checkout's venv install
  every workspace member from the same checkout, and a release uploads the member
  wheels before the framework.
- pyAML (`accelerator-middle-layer-apg` 0.3.1.post1, upstream 0.3.1 with its
  zero-length magnet fix) ships with OSPREY as the `pyaml-cs-osprey`
  control-system backend: every pyAML read and write goes through the OSPREY
  connector, so channel limits, write posture, control-target pinning and audit
  apply to it. A pyAML write runs only inside a pyaml_measure run (a run_tool
  call); anywhere else it is refused and nothing is written.

  `osprey build` writes a pyAML view for every served model that names a deck
  and has a `measurement/<model>.yaml`: `data/pyaml/<model>/configuration.yaml`,
  with the design simulator's lattice beside it (a `single_pass` model has no
  design simulator), one array per measurement group, one tool per measurement
  kind and, for a periodic model, the tune and chromaticity response matrices of
  its design optics. The lattice is the deck with each corrector the view drives
  carrying its kick as the dipole polynomials pyAML reads, so a corrector step
  in pyAML's design mode moves the orbit the served deck's kick moves; a
  zero-length corrector stays a zero-length thin multipole carrying its kick. An
  engine that cannot carry the kicks as polynomials (no `polynomial_kicks()`)
  stops the build of a periodic model whose view drives correctors
  (`engine-invalid`). A magnet sliced across zero-length and thick elements has
  no pyAML magnet model and is left out with the other setpoints pyAML cannot
  convert; a magnet whose every slice is zero-length is driven at unit length,
  its polynomials being its integrated strength. Magnets are named after their
  setpoint addresses, BPMs and arrays after their device and group ids; one
  group named as both `hcor` and `vcor` is two arrays, its name followed by `_h`
  and `_v`. A served model without a view is named in a note on stderr.
  `data/pyaml/` is written by the build only; a profile's `project/` mirror may
  not carry it.

  A measurement file is held to the kinds it allows: `orm` needs groups `bpm`,
  `hcor` and `vcor`; `dispersion` adds instrument `rf`; `trm` needs group `quad`
  and instrument `tune`; `crm` needs group `sext` and instruments `tune` and `rf`;
  `chromaticity_monitor` needs instruments `tune` and `rf`. A member the file does
  not name or the model does not wire, or any kind but `orm` on a `single_pass`
  model, stops the build (`reference-missing`); a step or settle key the kind's
  tool takes, missing, stops it (`value-invalid`). Group `hcor` stands for the
  setpoints its model steers horizontally and `vcor` for those it steers
  vertically, so one group named for both planes is split into two plane arrays.

  `data/facility_facts.json` records each pyAML view the render wrote under
  `measurement_models` (the configuration's path and digest, and the digest of
  each file beside it), and the rendered `.claude/hooks/hook_config.json` gains
  `measurement`: per model, the kinds the measurement file allows with the arrays
  each steps or reads, the addresses each group stands for, and the view's digest.
- Queue rows, history entries, stream frames and `GET /runs` records carry an
  `owner`, present only when the item named somebody. The deployment mints that
  name from the credential the request arrived with, in `X-Osprey-Owner`;
  nothing else names an owner.
- The BLUESKY panel's History now lists what was taken off the queue: a row for
  every plan removed, whether one at a time or by clearing the queue, and one for
  every plan aborted, each with the time and the name of whoever asked.
  `GET /queue/removals` serves the same list. It records who asked; it decides
  nothing about who may ask.
- `POST /queue/start` takes an optional `expected_plan_queue_uid`, the queue
  token the approver saw, and refuses a queue that moved since with
  `409 queue_changed_since_approval`. The `queue_start` tool, the panel's Start
  button and the terminal's queue bar all send it.
- Drag a rail entry along the rail to reorder it; the order is shared by every open page and kept across reloads.
- `osprey.runtime.record_writes()` is a context manager that yields the
  `(address, phase)` records of the writes the process makes while it is open:
  `attempt` once a write passed the target pin and the limits net, then `sent` or
  `landed` once its value reached the control system. A write that reached no
  control system records no second phase. When the connector itself raises
  partway through a multi-channel write, the channels it handled before raising
  also carry only their `attempt`: their outcome is unknown, and the value may
  have gone out.
- An admin-tier web terminal can turn the safety watchdog off for its own sessions
  without a restart, in Settings > Safety (or `POST /api/safety-switches/watchdog`).
  The move is audited and names who made it; while it is off, prompts there are not
  judged and no new hold opens; a restart turns it back on.
- `osprey.runtime.settle()` waits after a write until each setpoint's readback is
  within the tolerance the facility states for it, and fails when it is not.
- Safety watchdogs: facility-written checks, defined in the profile's `watchdogs:` block and off by default, judge each operator prompt in the background with a decision model or a read-only agent. In the web terminal a flag holds the session's agent writes (`session_held`) and asks the operator to stop or continue; a plain CLI session records flags only. A watchdog is advisory, not a gate on the first write, and a decision watchdog sends session history to TypeSafe or OpenRouter. `osprey watchdog replay` tries one on labelled prompts before it is enabled, and the `control-assistant` preset ships `guard-circumvention`, disabled. This needs an `osprey-connectors` release with the session hold, so the connectors floor must be raised before the next stable framework release.
- The facility build now stops on a malformed scenario archiver event: an unknown shape, a missing key, other than exactly one position key, a non-numeric value where a number is read, or a `ramp` or `spike` on a channel that is not `float`. The line names the scenario file and the channel, so the event is fixed before a deploy or `osprey sim apply` reads it.
- A scenario archiver `step` on a float channel may end with `until`,
  `until_offset` or `until_when`, after which the channel returns to its
  baseline.
- A service on the host network can now say what it binds, under `services.<name>.config`. `listens: false` means it opens no socket. `bind_env: NAME` names the environment variable its compose file renders the bind address into. `osprey up` reads that address to decide whether the deployment is reachable from other machines, and skips a service that listens on nothing in its host-port check.
- The Bluesky bridge lists every authored session plan at `GET /plans/session`, validated or not, so a cleanup can retire a plan whose validation is still in flight.
- The web terminal's sidebar groups are now yours to arrange. Rename
  **Workspace** and **Tools**, add groups of your own and move them up or down
  from the ⋯ button on a group's heading, and move any row except **Agent** to
  another group with its right-click **Move to** rows or by dragging it there.
  **Script runs** moves with **Artifacts**, and the groups are kept per user in
  the browser.
- The web terminal's sidebar has a resize handle on its right edge: drag it (or
  use the arrow keys) to set the labelled width between 180 and 400 px, drag it
  past the edge to show icons only, double-click it to reset. The width is kept
  per user across reloads.
- A sidecar panel that declares an MCP transport (`mcp_path`) can opt in to an
  agent hop with `agent_hop=True`. The terminal then admits exactly
  `GET|POST|DELETE /panel/<id><mcp_path>` on the panel token
  (`web_auth.panel_tier_routes()`), and the panel proxy relays that request
  unrewritten, without the caller's `Authorization`, `Cookie` or
  `X-Osprey-Terminal-Secret` and without re-issuing the operator secret. A call
  to one of the panel's ask tools is refused while the calling session is held,
  as a job fired on the dispatcher hop is. A browser session or the operator
  secret gets 403 on that path, every other spelling of it stays refused, and a
  panel that declares `mcp_path` alone is closed to every caller.
- A web-terminal panel whose sidecar failed to start, or stopped later, starts
  again when its tab is opened, one attempt at a time. The rail and `osprey
  health` name the panel and the reason it failed.
- Web terminal: `web.sidecar_ready_timeout_s` sets how long the terminal waits for a panel that runs its own server, such as the JUPYTER tab, to answer before it greys the tab (default 60 s). A value that is not a positive number is refused with a warning naming the key, and the default applies.
- A scenario archiver event can be placed with `at_when`, the logbook's own
  `{days_ago, time}`, so an event and the logbook entry narrating it land at one
  instant whatever time of day the scenario is applied. The control-assistant
  `rf-thermal` scenario uses it: its three CAVITY01 excursions now fall on the
  days and clock times its logbook entries describe, ending with the trip four
  days back, and nothing happens after the repair two days back.
- Simulation scenarios can give channels a common cause: a `drivers` block declares
  named slow signals (the `wander` texture, keyed by the driver's name and evaluated
  on epoch time), `couple` adds `gain * driver(t)` to chosen channels before their
  own noise — optionally with a `gain_wander` envelope so the coupling strength
  drifts — and `noise` replaces a channel's noise sigmas while the scenario is
  active. Live reads and synthesized history carry the same coupled term. The
  control-assistant template ships `rf-thermal-live`, which couples cavity 01
  temperature, reflected and forward power and tuner position for live
  correlation plots.
- The Simple view answers the agent's approval requests in the chat. A tool
  call that the safety hooks send for approval, a plan, and a question from the
  agent each show as a card in the turn. A call that the limits or the writes
  kill switch refuses never reaches a card, so Allow cannot override the
  refusal. "Allow for this session" is never offered for a tool that can change
  the machine or its run queue. Where the approval binds a call to a control
  target, the card names that target, such as Simulator. While the card has
  focus, Enter allows and Escape denies; Enter on a focused Deny or Details
  button presses that button. A new card does not take focus from the message
  box while you type.
- In the Simple view, an `artifact_id: <id>` in an agent answer is a link. Click
  it to show that artifact in the workspace: the gallery selects the artifact and
  the artifacts panel opens. If the gallery does not have the artifact, the link
  is struck through and nothing opens.
- The Simple view shows when the agent waits on the operator. The tab title
  carries a count of pending approvals and questions while they wait. An opt-in
  bell in the console bar sends a desktop notification when a request arrives
  while the tab is hidden, and another when a turn finishes while the tab is
  hidden. Replayed history never triggers either signal.
- The slash-command suggestions in the Simple view include the agent CLI's own
  `/clear` and `/compact`, beside the project's commands. A `/compact` keeps the
  conversation and shows a notice in the log.
- An approval, question or plan card that nobody answers within 30 minutes is
  declined by the chat. The tool does not run, a plan stays in planning, and
  the agent is told there was no answer. The card shows that it expired, and
  the turn goes on or ends as it would after a decline.
- The Simple view shows how full the conversation's context window is, beneath
  the composer. From 80 percent it warns and suggests `/compact` or a new
  conversation. The reading comes from the agent CLI when each response ends,
  and the meter never shows a cost.
- In the Simple view, a question from the agent stays in the chat after you answer
  it. The card folds to one line, for example "Answered · Lattice". Click the line
  to see each question the agent asked and the answer you gave. A skipped or
  expired question folds the same way. When the view loads the conversation
  again, it shows the same line with the same answers.
- The Simple view speaks to screen readers. A finished reply is announced once,
  with its opening words. An approval request, a question from the agent and an
  error are each announced once, when they happen. Replayed history is never
  announced. The message log is labelled as a conversation, and every card is a
  labelled group. Keyboard focus is drawn on every control.
- In the Simple view, a skill has a row of its own, with a book icon and the
  skill's name. The row says "Used skill" when the agent chose the skill, and
  "Ran skill" when the operator ran it by its slash command. The status line
  says "Using skill <name>…" while the skill loads. A reloaded conversation
  shows the same rows. Before, a typed skill showed as the CLI's raw
  `<command-message>` text. The slash list marks each entry as a skill, a
  command or a built-in, and marks a skill that only the operator runs as
  "manual".
- In the Simple view, the agent can draw a small picture in its reply: a fenced
  `svg` block is shown as the drawing, in a box with its source one click away.
  The agent uses it for a quick look at a result, such as the top three values
  or a few key numbers, and links the artifact that holds the full result. It
  draws bars only for values in the same unit, with each bar's length set by its
  value, and shows values in different units as labelled numbers. The drawing
  uses only the theme's colours, so it is easy to read in light and dark themes:
  any other colour is removed, and a part with no colour uses the text colour.
  It cannot run code, embed a web page, animate, or load anything from outside
  the page. A block that is not one complete drawing stays code, so a drawing
  that is still streaming shows as code until it is complete.
- In the Simple view, the agent's task list sits in a bar above the message box.
  The folded bar shows how many tasks are done, the task in hand or the next one,
  and a progress bar. Click the bar to see the whole checklist. The bar reads the
  `TaskCreate` and `TaskUpdate` calls that the current Claude Code makes, and also
  `TodoWrite`. Before, the task card inside the turn read only `TodoWrite`, so it
  never appeared. The task calls no longer show as rows in the turn. The list
  stays for the whole conversation and comes back when the page reloads. When
  every task is done, the list stays until the next prompt. The workflow rule now
  tells the agent to mark each task in progress when it starts it, and completed
  when it finishes it.
- A reference page documents the simulation bundle a facility writes —
  `machine.json`, each scenario's `scenario.json` and `logbook.json` — with the
  event scripts, how scenarios compose and every refusal the loader and
  `osprey sim apply` raise.
- `osprey_connectors.simulation.composite` serves a set of model children (physics first, texture last) from the simulator view, marks a child whose engine raises as failed with the engine's text as its status, and appends model records to `var/simulator/<model>.log`.
  An active scenario's shared drivers and noise replacements move the channels it couples.
- `osprey_connectors.simulation.model_status(connector, model)` returns a served physics model's status, `ok` or the engine's error text, from the composite the connector serves in process, and refuses an unknown model by naming the served models.
- `simulation.tick_s` sets the simulator's tick period in seconds, default 1.0. A value at or below 0 stops the build with `profile-invalid`.
- A setpoint and its readback must agree on units: when both state a `unit` and the two differ, the build stops with `pair-invalid` naming both channels.
- A scenario can state `still`; every facility has a `still` scenario that serves every reading without drift, couplings or noise.
- A sidecar panel can now ask for time to stop. `SidecarPanel.stop_grace_s`
  (default `0`) is the number of seconds its stop waits between the stop signal
  and the kill, and the one number the deployment sizes the compose service's
  `stop_grace_period` from: the slowest panel a terminal shows plus the
  terminal's own stop, rendered whenever that exceeds compose's default. The
  JupyterLab panel declares the five seconds its stop already waited.
- `osprey build` and `osprey validate` print one reminder when `system.timezone` is UTC or unset, naming the key to set in `profile.yml`. UTC still builds.
- `control_system.target_switch.probe_timeout_s` (default 5) sets how long a
  target switch waits for the new connection to read the target's
  `probe_channel` before it refuses the switch and keeps the current target. A
  value that is not a positive number is logged and the default is used.
- The Microsoft Teams bridge delivers documents a run produces — tables, PDFs
  and other files — and images too large to post inline. Set
  `TEAMS_FILES_DRIVE_ID` (and optionally `TEAMS_FILES_FOLDER`) to a SharePoint
  library an administrator grants the bot, and each file is uploaded there and
  shared with the members of the conversation it answers; without a library,
  every file the bridge cannot deliver is named in a closing note.
- The `osprey` plugin adds the `/osprey:test-audit` agent skill: a gate that new
  and changed tests pass before they land, and an audit workflow for tests that
  re-assert source, duplicate stronger proof, or keep test-only seams alive.
- `claude_code.thinking.display` sets whether the agent shows a summary of its
  thinking. It defaults to `summarized`, so the Simple-view chat and the agent
  timeline now show thinking from sessions that previously recorded none,
  including headless runs and `osprey chat --print`. Setting it to `omitted`
  turns this off.
- Facility seeds gain `topoff`, a stored current that decays with a lifetime
  and refills on a period, and `topoff_lifetime`, a readback of that lifetime.
- The virtual accelerator's pvAccess host port is set with
  `virtual_accelerator.pva_port` in the build profile. Two deployments that each
  run a virtual accelerator can share one host; before, both published port 5075.
- The health check `osprey scaffold ci` emits at `scripts/verify.sh` now lists every container of the deployment and flags any that is not running or reports an unhealthy healthcheck, beside its endpoint probes. It still exits 0; `./scripts/verify.sh --strict` exits 1 when anything is flagged, for a job that should fail on it.
- The facility vocabulary gains seven signal roles: `temperature_limit_readback`,
  `blade_position_setpoint`, `blade_position_readback`, `coolant_flow_readback`,
  `valve_position_readback`, `beam_lifetime_readback` and
  `operation_mode_status`.
- `osprey web --port 0` serves on a free port the system picks and prints it,
  with `--detach` too. `--reload` needs a fixed port and refuses `--port 0`.
- `osprey.runtime.budget` computes one write time budget from the connector
  blocks: the call bound T and step-read bound S (maxima over every block,
  clamped so a write's bound `S + 2T` stays within 24 s), the stop grace and the
  relay timeout, plus `bound_budget()` for the block a session is bound to. A
  non-finite bound (NaN or ±infinity) in a block's `step_read_timeout_s`, or in
  the budget's `T` or `S`, is refused with a `ValueError` that names it.
- `osprey.deployment.write_gates.WRITE_GATED_TOOLS` is the one list of
  write-gated agent tools. The headless agent runner and the MCP audit
  middleware read their degraded floor from it instead of carrying a copy; the
  hooks' literal stays, pinned against it. Whether a person is asked before a
  write is the execution verdict's `position`, read nowhere else.
- The header chip switches the control target and turns writes on or off. A
  notebook opened in its own window shows the same chip in its top-right corner.
- Shared containers (the Bluesky queue, the dispatch worker) read a
  build-rendered login→tier map, `control_system.login_tiers`, and write only
  when the container's tier, the owner's tier and the owner's switch all allow
  it. A login missing from the map is refused as `owner_unknown`.
- `osprey status` gains a Writes section: per login, its tier, the control
  target its record points at, and whether writes are on.

### Changed

- The framework requires `osprey-connectors>=2026.11.8`; `pyaml-cs-osprey`
  states the same floor, and `osprey-optimizer` requires 2026.11.6 or later.
  The floor rose with each connectors release the framework came to read from:
  the `Live` banner and refusal wording (2026.10.2), the pause of a held
  session's writes (2026.11.0), archive attribution to a machine, connection
  state and display-only reports (2026.11.1), the env chain without the
  shell-only launch variables (2026.11.2), programs' bounded write
  authorization (2026.11.3), scenarios that script archiver events on a shared
  channel together (2026.11.4), the simulator's `PERIODIC` refresh and
  `SETPOINT` role names (2026.11.5), and in 2026.11.8 the simulator view's
  `engine_role` (`Binding.engine_role`, `ENGINE_ROLES`),
  `ipc.declared_gateways`, through which target eligibility and the switch
  gate read `gateways: []`, and `safety_file.ensure_shared_dir`. The simulator
  view no longer carries the transitional `role` copy of `engine_role` that
  connectors before 2026.11.8 read. (#2353, #2462, #2571, #2592, #2638)
- The `ariel` qmd sidecar skips the embedding pass when ARIEL hybrid search does not query its
  vectors: under `strategy: rrf`, or with hybrid search off. (#2383)
- The web-terminal entrypoint's access-probe warning names the SELinux label check when SELinux confines the container. (#2384)
- **Breaking change:** `artifact_list` lists the current session's artifacts
  (`scope="all"` for every session), drops its `source_agent` and `artifact_type`
  filters, and returns 20 rows unless `last_n` (1–200) says otherwise.
  `artifact_delete_all("everything")` deletes only the current session's
  artifacts. `artifact_read` returns a JSON artifact's content parsed.
- Web terminal: the agent's `open_panel` opens a panel in a new tile split off the
  service tile used last instead of replacing the panel there. `place="replace"`
  replaces it, and `tile`/`edge` choose the target and side; the result reports
  how the panel was placed and notes when no room for a new tile forced a
  replace. Without `place`, an open splits only when both tiles keep at least
  300 px along the split; an explicit `place="split"` goes down to the dock's
  minimum. Panels the Logbook and Phoebus tools open follow the same default.
  Opens from the sidebar and from links are unchanged.
- The Agent row in the web terminal sidebar no longer has a chevron: click the row to fold the session list, and the right-click menu's Expand all now opens every session, like ⌥-click.
- An application container that runs the project image now resolves its config, guarded-run directory and limits database under its own image root, and no longer bind-mounts the limits file. A bridge container's Channel Access name server is now the gateway the render's write tier selects: read-only unless the tier arms writes.
- An application's write tool is placed on the session's control target, not the
  real machine, so it can write on a simulator-only deployment.
- The approval record reads an application tool's requirements, including exactly-one-of groups, from the rendered config.
- The Logbook names who wrote a draft or entry in words operators use: "staged
  by the agent" and "Source: Agent" instead of the MCP server name `ariel-mcp`.
  Stored values are unchanged.
- An enhancement that never ran on an entry now reads "not run" (state
  `not_run`) in the Logbook's badges, the Enhancements tab, the Status drawer's
  coverage and `osprey ariel status`, instead of "pending"; "pending" is kept
  for a recorded run that has not completed.
- ARIEL: the web interface filters by facet counts in the rail, a day histogram
  and ``author:``/``date:`` search operators, and pages with **Load more**; entry
  fields declared ``facet: true`` become filter dimensions. ``/api/entries``,
  ``/api/search`` and the new ``/api/facets`` take a ``filters`` object, page by
  cursor or offset and report ``total``/``total_exact``/``has_more``. Run
  ``osprey ariel migrate`` to backfill entry subjects. The MCP tools now return a
  null ``entry_url`` for every ARIEL-only entry, including local ARIEL Web saves.
  **Search modules:** descriptors gain ``accepts_entry_filter`` and
  ``accepts_offset``, ``ModuleOutput`` gains totals, and a module declaring
  neither is called with a wider ``max_results``.
- ARIEL: the **Status** drawer, ``osprey ariel status`` and its ``--json``
  ``coverage`` block count each enhancement module's entries as complete,
  pending, stale, failed or n/a. Text modules go stale when their config
  changes and catch-up re-runs them. An agent's ``entry_open`` selects the
  entry in place, keeping the search and scroll.
- The Logbook's entry list separates every entry with a hairline, keeps the day
  heading pinned while its entries scroll, and puts the enhancement badges on the
  author line. The selected facet now shows as selected, the reader stacks its
  metadata under the text when the pane is narrow, form fields use the panel's
  font, and an empty reader says how to pick an entry.
- Logbook rows under a day heading show only the time (the full date is the
  tooltip), enhancements that do not apply no longer draw a badge, the list
  column starts at a third of the panel, and search issues are a dot on the
  search-mode button (text in its tooltip and popover) instead of a banner.
- The web terminal calls the ARIEL panel **Logbook**: sidebar, tab, palette,
  page title and status text. Typing "ariel" in the palette still finds it.
- A new always-on ARIEL migration, `logbook_web_source_system`, renames stored
  entries whose `source_system` is exactly `"ARIEL Web"` to
  `"OSPREY Logbook Web"`, so the Source System facet shows one name. Renamed
  entries are re-exported to the hybrid-search mirror on its next resync.
- An entry saved locally from the Logbook's compose sheet now stores
  `source_system: "OSPREY Logbook Web"` instead of `"ARIEL Web"`. Existing
  entries keep their stored value; the agent's direct writes keep `"ARIEL MCP"`.
- ARIEL: the web interface is one screen — a mailbox rail, the entry list and a
  reader — with keyboard navigation, and old ``#search``/``#browse``/``#create``/
  ``#status``/``#entry`` links still open it. Embedded ARIEL now always offers
  search in the tile bar.
- A deployment with its own logbook database can name a read-only login with `ariel.database.readonly_uri`. Without one, the agent's SQL tool is not offered. A store whose read-only login is refused no longer falls back to the owner's: the SQL tool reports itself unavailable with the reason.
- `artifact_read` returns a PNG, JPEG, GIF or WebP artifact of up to 512 KB as
  image content beside its metadata, so the agent sees the figure without the
  Read tool. Other binary files still report `content_omitted`.
- Artifacts belong to the session that made them. The gallery's session view lists
  one session's artifacts and its search stays within that session; the session
  page and the feedback bundle list what the session owns, the bundle naming
  unowned artifacts from the same time window in a separate section.
  `[Gallery Focus]` is kept per session, and pins set before this change reach no
  agent prompt. In `osprey chat`, pins and focus set in the companion gallery no
  longer reach the agent; the agent's own pins and focus still do.
- Web terminal: the **Pop out** button in a tile's header now moves the tile's
  content into a window that stays attached to the terminal, for the agent pane
  too. The window opens at the terminal's address with a bar naming the pane and
  a **Reattach** button; the tile keeps its place and shows **Reattach** too.
  Closing the window or reloading the terminal also puts the content back. The
  standalone page is now **Open in new window** on the panel's menus and in the
  command palette. The agent pane's Export button moved to its tile menu as
  **Export conversation**.
- A profile that still sets `web.bar_items` builds, with one warning: the web
  terminal's sidebar replaced the header and the status bar that key arranged,
  so the key is dropped at the parse point and ignored rather than refused.
  The warning names the sidebar and says to drop the key; `osprey init
  --revalidate` rewrites the profile.
- The channel-finder benchmark query file is a facility view, written with the other views; the build no longer copies and then prunes the query sources. The control-assistant preset no longer states `channel_finder.benchmark.dataset_path`, `ariel.vocabulary.enabled` or `ariel.vocabulary.path`: the build states them when the facility ships benchmark query sources or `data/facility/ariel/vocabulary.yml`.
- The open-model benchmark board runs the agentic tests only, scores each on its
  first attempt, and groups them by tier: Tier 1 (basic tasks, compared on time
  and tokens per task) and Tier 2 (harder tasks, shown test by test). Each test
  row names what the operator asks and what passes.
- Every file `osprey init` copies from a bundled facility says so on its first line; `osprey facility import`, in every format, replaces a bundled facility in one step and refuses only hand-written records of the kinds that format writes, naming them.
- **Breaking change:** OSPREY now runs CANOPY rebased onto SLAC's current main,
  where the formula module is `formula-engine` (was `fox`), served at
  `/formula-engine/...`. Build the CANOPY payload from the rebased commits
  `proven.py` names; a payload built at the old commits is refused.
  The pinned PyDMConverter refuses unsafe `.bob` input: rule conditions and formula
  PVs outside the Formula Engine subset or holding a macro, web links other than
  http, https or relative, and file references that are not paths.
- Web terminal: the agent chat's messages fade out over the last 20 px as they
  scroll down to the message box, in place of a hard cut.
- Figure axes name the columns they plot, and the bridge states those columns' declared units; a plan render that sets a unit itself is served as `render_failed`.
- `check_config_keys` now renders a deployment with applications, and the config-key manifest describes everything the build projects from one as families (`services.<app-container>.*`, `web.panels.<app-panel>.*`, `claude_code.servers.<server>.*`, `apps.<app>.*`) plus `dispatch.denied_app_tools`.
- Web terminal: a right-click opens the menu of the place clicked — tile, sidebar
  row or heading, chat, terminal, empty dock, Channel Finder results — with
  submenus, shortcut hints and keyboard navigation; Shift+right-click shows the
  browser's own menu.
- A link out of a profile convention directory is refused once, by the repository-boundary line "… resolves to …, outside the repository. Move it into the repository."; the second "symlink pointing outside the profile" refusal is gone.
- The demo's sector 7 beam loss now looks like one. The SR07 pressure jumps to about 1.5e-07 Torr within a sample and pumps down over seconds with a slower tail, and the stored current drops about 18 mA in one step and stays down until the next top-off cycle refills it in about 25 s. Before, both curves were symmetric bumps that started moving two minutes before the burst. The plot is now titled "Stored beam current and Sector 7 pressure", the logbook entry, session and draft note describe the new shapes, and the daily `vacuum-burst` scenario uses the same shapes. Scenario files can use the new `burst` and `loss` event shapes.
- The `demo` preset no longer turns on the orbit bump app. It still ships with OSPREY; `osprey init --preset demo --set 'apps.orbit-bump={}'` turns it on.
- The `control-assistant` preset is now `demo`: `osprey init osprey-demo --preset
  demo`. It is the full OSPREY demo on the simulated example facility, and still a
  starting point to trim down (`hello-world` is the one to build up from). The old
  name keeps resolving, so existing repos, `extends: control-assistant` and
  `osprey scaffold pull control-assistant:…` work unchanged.
- `osprey build --dev` / `osprey up --dev` images, and the project venv built
  from an editable checkout, now install every third-party package at the
  version the checkout's `uv.lock` holds instead of resolving it fresh from the
  index, and no longer prime a released `osprey-framework` first. `--dev` needs
  `uv` and a current lock; a missing or stale `uv.lock` stops the build with a
  pointer to `uv lock`. Release builds are unchanged.
- Dispatch jobs deny the application tools the render's config lists under
  `dispatch.denied_app_tools`, and deny every MCP tool outside the framework's own
  servers when that list cannot be read.
- `scripts/check.py` re-syncs a stale editable install before it runs, and a
  machine that cannot judge (too little free disk, Docker storage that fails)
  stops with `check: environment: <probe>: <fact>` and exit status 3.
- The example custom connector that `osprey build`'s limits-block hint and the
  docs spell is now `mypkg.SiteConnector`, a name that belongs to no real
  control-system stack.
- The example facility carries its benchmark queries and ARIEL vocabulary under `data/facility/`; `rules/facility.md` states no facility fact and points at `data/facility_facts.md`, the one identity, rebuilt every build. The derived archiver freshness check watches the first readback by address, the probe's channel.
- The example facility's storage-ring BPM positions are now in mm, matching its
  BTS and BR BPMs; every channel, wiring gain, seed, screen and scenario that
  uses them moves at equal physical value, and the SR GOLDEN seeds become 0.0 mm.
- `osprey build` refuses a stated `ariel.vocabulary.path` that is not a file while the vocabulary is on, naming the path.
- The build, the KNOWLEDGE panel, ARIEL and `osprey facility show` look facility records up one way.
- Every facility view reads the facility file's records one way; the views it writes are unchanged.
- The control-assistant preset now starts in `[rf-thermal, demo-history]`, so a fresh deploy opens with example sessions in the sidebar in place of the retired example plot.
- An application `image:` fragment may carry only the stated compose keys; any other key, a volume driver or driver option, or host networking the service's network axis did not ask for is refused by name.
- An application `image:` fragment may no longer carry top-level `configs` or
  `secrets` (bind the file on the service instead), and a long-form volume's
  `type` must be `bind`, `volume` or `tmpfs`, spelled literally.
- Web terminal sidebar: a group's menu opens from a right-click anywhere on its heading (or
  Shift+F10 on its name); the ⋯ button is gone. The Agent row's group has no heading: it always
  leads the sidebar, and row menus call it Sessions.
- The System Health panel is now a stack of layers — operator interface,
  agent, safety chain, MCP servers, data services, connections out,
  configuration and build, host — with one tile per component and its details
  opening in place. New checks cover the write hooks, write switches, limits
  file, watchdogs and exec sandbox, render drift, versions, dispatch workers
  and the services behind panels; a provider no role uses is no longer
  probed or warned about.
- A container receives the staged login→tier map and the control-identity module because its service declares `drives_target` / `control_identity`, not because its name is on a list; a projected service's audit identity comes from its `services.<key>.audit_identity`.
- The Agent pane's bar is the session title ▾, + and the tile's ⋯, Open in new window and ×, the same buttons as every other pane: Export conversation and Desktop notifications are ⋯ rows, and the status dot is gone; Jupyter's own rows sit under This panel in its ⋯.
- A login file (`tiers/<name>.yml`) now refuses the deployment facts (`connector`, `provider`, `port_base`, the service blocks, `artifact_server`, `requires_osprey_version`, `osprey_install`, `python_env`), naming `profile.yml` as their home; a login still picks its `model`. `osprey init --revalidate` lists a tier that spells `provider` as a manual step.
- Tile headers show the panel's name and controls only: the small grey subtitle
  beside the name (CANOPY's screen path and title, EVENTS' trigger and run counts)
  is gone. The terminal ignores a panel's `text` header item. In the EVENTS tile,
  an unreachable worker or a full queue shows as a banner above the runs.
- The knowledge panel's reading pane now states which concept it shows: `#reader-content` carries `data-concept-id` once a concept has rendered, and drops it while loading or showing anything else.
- Every login lives in `tiers/`: a narrow one says `lists: own`, and one named by
  `shared:` is sealed read-only. `osprey init --revalidate` moves an
  `apps/<name>.yml` login to `tiers/` for one release; `osprey scaffold apps`
  refuses, naming `osprey scaffold tiers` and `osprey app new`; landing
  `type: apps` reads as `type: shared` ("Shared logins") for one release; and the
  build fingerprint covers each `apps/<name>/` application directory.
- The web terminal's work pane holds one panel per tile again, and the tile
  header's tab strip is gone. Opening a panel replaces the panel in the service
  tile you used last, which stays in the sidebar one click away; Open in a new
  tile and drops on a tile's edge or the work pane's edge still split. A layout
  saved with several panels in one tile comes back showing the panel that was in
  front, the others in the sidebar. For the agent, `close_panel_tab` and
  `close_panel_tile` become `close_panel`, `open_panel_set` opens panels side by
  side, `drop_panel` takes edge, border and outer targets, and a layout, preset
  or report with several panels in one tile is refused `one_panel_per_tile`.
- The optimizer takes no `targets` list and its API no `OPTIMIZER_TARGETS`
  allowlist: the session's control-target switch and the machine a write
  authorization names decide what a run drives. The `optimizer` MCP server's
  machine reads take no `target` either: they read the session's control target.
- An optimizer evaluation whose objective or constraint reads back unreadable
  waits the environment's settle dwell and re-reads the observables without
  writing the setpoints again, so a re-measurement spends no extra step of the
  run's write authorization.
- **Breaking change:** `POST /jobs/{id}/start` now requires `?config_revision=`.
  Read the draft with `GET /jobs/{id}/config`, take its `X-Config-Revision`
  header, and send that value; when the draft changed since it was shown, the
  start answers `409`. The draft routes (`POST /optimizations`,
  `PUT /jobs/{id}/config`, `PUT /jobs/current`) no longer need the launch token.
- Any pane, the Agent included, opens in a new window and leaves the workspace. The window carries the tile's bar with Reattach, the sidebar row's icon reattaches, closing the window closes the pane, and after a hub reload detached panes are back in place.
- Panels open on the standard toolbar, and their other verbs are tile ⋯ rows.

  - Artifacts: the artifact alone, a content strip for plot, table and note tools, Info and Download.
  - Channel Finder: sidebar toggle, Explore | Feedback, result count and search; the facets are a shared sidebar.
  - Knowledge: a breadcrumb, a bundle-health chip and search; the contents list is a shared sidebar.
  - Bluesky: the views, lane picker, queue status, Start and the halts; the plan filter is in a shared sidebar.
  - System health: categories, status filters with counts and scan now, with a categories sidebar.
  - Events: Activity | Triggers and search, segmented filters, and one centred state when it cannot read.
- Application directories and local-path `dependencies` packages are profile inputs: editing one reports the build as stale, and a package outside the repository is refused like every other input.
- Sidebar: Health uses a stethoscope and Bluesky the Bluesky project's own cloud-and-b mark; the Workspace divider's `aria-valuenow` stays within its limits and follows the drag.
- An application manifest whose `require.one_of` group names a digest or `approved_target` is refused (`one_of_readbacks`): a `one_of` group holds readback (`bind.state_fields`) names only.
- Web terminal: a script run's row in chat has the run's menu (Open result,
  approval preview or code, Copy run ID, Delete…), the same as its sidebar row.
  An artifact's menu reads the same in the sidebar and on the Artifacts tile, and
  Delete… asks the same question from both.
- The agent's `get_run_figure` now reports a run whose rows could not be read as an error that says so, instead of an empty figure. The BLUESKY panel is unchanged.
- The build derives the simulator's probe channel, the archive's freshness check and the start set from the facility file; a stated value is checked against the file at build; a repo with no facility tree builds.
- The build's service check now knows the bluesky second lanes by name, and its "Bundled services are" list includes them.
- Each web terminal session keeps its own layout on the hub: switching back to a session brings back its panels, tiles, sizes, the Agent pane and each panel's page where the hub can read it. A new or older session keeps the layout on screen.
- A session row in the web terminal sidebar no longer shows a title and hint tooltip on hover; only the artifacts preview remains.
- Web terminal: in the Workspace tree only the active session is bold; the session menu drops "Open in Agent" (a click opens it) and offers "Use automatic name" only on a hand-named session.
- Web terminal: the sidebar's session tree shows only the current session's
  artifacts. Other sessions stay folded, with no ▸ and no **Browse**, until you
  switch to them; click or double-click the current session to fold it.
- With the Agent pane open, a click on the sidebar's **Agent** row folds or
  unfolds the session list and leaves the pane open; with the pane closed it
  opens the pane and the list. A click on the current session's name folds or
  unfolds its artifacts. Artifacts under a session are grouped by day unless the
  operator chose **Group by Kind**.
- The sidebar's Tools section is now called Apps; a saved layout keeps its arrangement.
- Sidebar: the Workspace group grows with its sessions up to 480 px, then scrolls; session rows no longer squash when it runs out of room. A dragged height is kept as a share of the sidebar, so it holds when the window is resized; the drag snaps silently to the automatic height and resets there.
- Sidebar: the width handle sits on the gap between the sidebar and the tiles,
  so the pointer finds it where the two meet.
- Web terminal: the sidebar's Agent row is now **Sessions**, with a new-session button. The session the agent works in has a breathing accent bar, a clicked session takes the accent at once, and hovering a folded session shows its artifacts, which open on click.
- The agent now writes a plan when a measurement would otherwise take several runs of an existing one, and sizes a step itself when a channel has no limits record instead of stopping.
- `osprey build --skip-deps` still builds each deployment package's wheel, in the scrubbed environment, so its render's image installs. It skips only the venv and its install.
- The web terminal sidebar has a third built-in group, **System**, after Tools. By default it holds the Events and Health rows instead of Events sitting in Workspace. A layout already stored in a browser keeps its groups and placements and gains System at the end.
- Every tile has one ⋯ menu, opened from its bar or by a right-click on the bar: the panel's own rows under "This panel", then the tile verbs, with Reload and, where the panel has a docs page, Help. A panel shows at most three icons in its bar; the rest fold into ⋯.
- The session tree in the Workspace sidebar group no longer has a 280 px height cap: it fills all the room down to the divider and scrolls inside when its content is longer, and in automatic sizing it takes what the rail leaves.
- The virtual accelerator image now installs the framework in the same layer as
  the other service images and adds its `virtual-accelerator` extra on top, so a
  deploy that builds the VA and the bluesky bridge installs the framework once
  instead of twice: one fewer multi-minute install, and about 2.7 GB less image
  data. On arm64 the image no longer compiles its own PyAT wheel, which the
  shared layer already installs.
- The facility file's wiring records, the simulator view and a scenario's `still` selection name what a model's engine says a record is `engine_role`, apart from a channel's own `role`; an engine's `describe()` returns `engine_role`. To upgrade, rename `role:` to `engine_role:` in a scenario's `still`.
- Ending a terminal session — a restart, logout, server shutdown, the session pool making room,
  or a switch to the Simple view — now also ends the processes its agent started, including
  shell commands that run in a process group of their own, and logs what it ended. (#1077)
- Ending a Simple-view conversation's agent — a switch to the Expert view, a restart, logout,
  server shutdown, or the chat pool making room for a new conversation — now also ends the
  commands and scripts that agent started and still has running, and logs what it ended. (#1077)
- Switching to the Expert view, or *Stop and switch now* there, now says which still-running
  commands the Simple view's agent started will end with it: *Stop both* ends the agent and those
  commands, *Cancel* returns to the Simple view with both still running. With nothing running the
  switch happens at once, as before. (#1077)
- A Simple-view conversation whose agent still has a command it started running is no longer
  reaped as idle. The idle timeout (`web.chat_idle_timeout_s`, 1800 s by default) counts from when
  the last such command ends, to within one reaper sweep. A reaped conversation keeps its
  transcript: the next message resumes it. (#1077)
- *Stop and switch now*, and a switch to the Simple view that finds the terminal agent idle, now
  say which still-running commands the agent started will end with it: *Stop both* ends the
  agent and those commands, *Cancel* returns to the Expert view with both still running. With
  nothing running the switch happens at once, as before. (#1077)
- Bin edges from the mock, MongoDB, DOOCS and MYA archivers now start at the
  query window, so a window opening off the old midnight-anchored lattice shifts
  its bins (1-hour bins from 00:17 are labelled 00:17, 01:17 rather than 00:00,
  01:00); EPICS keeps the Archiver Appliance's own epoch-anchored grid. (#1078)
- A profile whose approval policy asks for `execute`, `execute_file` or
  `pyaml_measure` and does not select the `approval` hook stops at build
  (`profile-invalid` on `approval.enabled`). Add `approval` to `hooks:`, or set
  `config: approval.enabled: false` (or `skip` for the guarded tools). An
  unstated `approval.enabled` counts as asking.
- A read-write `execute` or `execute_file` run now holds its control target's guarded-run lock
  from before its first line of code until its result is saved, so it never runs at the same time
  as a pyAML tool run or another read-write run on the same machine. A run that finds the lock
  held does not start: it fails with `failure_kind: guarded_run_busy` and names the holder's
  process id and start time. A read-only run takes no lock. A process the run forks does not
  keep the lock once the run ends.
- The `pyaml_measure` approval prompt lists every setpoint the call steps, no
  longer the first twenty and a count of the rest. A call that would step more
  than 500 setpoints, the most a prompt lists, is refused without asking.

  The pyAML view writes `trm.json` and `crm.json` whenever every member of the
  `quad` or `sext` group has a modelled column, so `pyaml_measure` steps such a
  group as one knob, `crm` by group included, even when the group is a single
  supply. The tune or chromaticity correction tool that loads the matrix is still
  added only when the matrix has rank two. The example's `SR` view now carries
  `crm.json`.
- An agent that points a panel at a new page while the panel's backend is not answering no longer replaces the last good page with an error page. The new page loads on the backend's next answer. (#1319)
- The web terminal's workspace file routes now answer `400 invalid_session_id` to
  a `session_id` that is not a canonical session UUID, as the posture routes do,
  instead of quietly serving the unscoped workspace tree. (#1322)
- **Breaking change:** each corpus now has its own qmd sidecar (`qmd-okf`,
  `qmd-ariel`, …) with its own index and its own port in the qmd band
  (`services.qmd.port` + 0 for the OKF bundle, + 1 for the ARIEL mirror), so a
  search restricted to a small corpus is no longer crowded out by a large one
  sharing its index. The first deploy builds each corpus's index from scratch;
  the old shared `qmd_index` volume is no longer used and can be removed. (#1566)
- `osprey up` and `osprey restart` refuse to start while containers of the same
  project name, created from another copy of the repo, are on the host. Nothing
  is stopped. The refusal names that copy's path, the `osprey down --repo`
  command that stops it, and the `profiles/<variant>.yml` overlay that gives this
  copy a project name of its own. Volumes no longer carry the
  `com.osprey.repo-id` label; they belong to the project by name. The first
  `osprey up` after upgrading keeps every existing volume, and `osprey reset`
  now removes all of the project's volumes once no other copy's container holds
  the name. (#1647)
- A tier or app the build does not render — an entry in
  `modules.web_terminals.renders` with no `build_profile`, whose `project_path`
  points at a project supplied ready-made — now proves its tier by that render's
  own `config.yml` at build time, the same file `osprey up` reads at start. Such
  a render may be the `default_tier` or sit on a shared card like any
  delta-rendered one; one whose `project_path` holds no `config.yml` is still
  refused there, naming the path the build looked at. (#1647)
- **Breaking change:** the web tier is named after the deployment's
  `project_name`: the reverse proxy is `<project>-nginx`, the auth sidecar
  `<project>-auth` (its locally built image `<project>-auth:local`) and a user's
  terminal `<project>-web-<user>`. Lint, provisioning, `osprey users remove`,
  orphan reconciliation and teardown all resolve the same name. The first
  `osprey up` after upgrading replaces the old-named containers and leaves the
  old `<prefix>-assistant-auth:local` image for you to remove. Anything that
  addresses the containers by name (scripts, monitoring, `docker exec`) needs
  the new names. (#1647)
- **Breaking change:** `osprey-connectors` declares its public API. Every public
  module lists its names in `__all__`. The `control_system`, `archiver` and `ipc`
  subpackages hide their implementation modules behind an underscore
  (`control_system/epics_connector.py` is now `control_system/_epics_connector.py`)
  and export their public names from the subpackage itself. The old module paths
  are not kept as aliases. The `simulation` package and its modules stay public,
  each with its own `__all__`. (#1810)

  - Import connector-contract, archiver and connector-host names from
    `osprey_connectors.control_system`, `.archiver` and `.ipc`. Build a concrete
    connector through `ConnectorFactory` from its type name; a registry that
    needs the built-ins by module path takes them from
    `osprey_connectors.factory.BUILTIN_CONNECTORS`. `type:` values in
    `config.yml` are unaffected.
  - Names that move or change: the per-call timeout keys are in the new
    `osprey_connectors.call_timeout`; `DEFAULT_VA_PORT` is in
    `osprey_connectors.types`; the MongoDB store's `address_overrides`,
    `mongo_client_kwargs`, `CA_FILE_ERRORS` and override variable names are
    exported from `osprey_connectors.archiver`; `ConnectorHostPool` and the rest
    of the connector-host plumbing are exported from `osprey_connectors.ipc`.
    `types.SIMULATED_TYPES`, `config.get_configurable`,
    `honesty.invents_history`, `honesty.target_section`,
    `posture_store.NoOwner` and
    `ControlSystemConnector.current_value_reader()` lose their underscore;
    `identity._usable` is `identity.usable_identity`, and the verification
    module's `_section` and `_sub` are `ipc.config_section` and `ipc.subsection`.
  - A connector subclass that overrides `_current_value_reader` must rename the
    override to `current_value_reader`. The old name is no longer called, so an
    unrenamed override is ignored and every write to a channel with a `max_step`
    limit is refused.
  - A framework from an earlier release imports the old module paths and fails
    to start against this `osprey-connectors`. Upgrade the framework together
    with the connectors, or pin `osprey-connectors` below this release.
  - The rendered limits hook imports `LimitsValidator` from
    `osprey_connectors.control_system`, and falls back to the module path an
    earlier release used, so a hook rendered by this release still checks limits
    against an installed framework that predates the namespaces.
  - The package ships a `py.typed` marker, and the lint job now fails if the
    framework imports past these namespaces or takes an underscore name from the
    package, or if the package imports `osprey`.
- Web terminals in registry mode: the auth sidecar's image no longer has to be
  named by hand. With no `modules.web_terminals.auth.image` it is
  `<registry.url>/<project_name>-auth:<image_tag>`, on the same tag as the
  terminal images, so one `OSPREY_IMAGE_TAG` names every image a pipeline pushes.
  A set `auth.image` still wins. (#1938)
- **Breaking change:** a deployment's own skills, agents, rules and output
  styles render only where a list names their path, like library artifacts are
  named: `skills/<name>`, `agents/<name>`, `rules/<name>` or
  `output-styles/<name>` in the matching list of `profile.yml` (and so every
  tier), a tier, or an app. Before, each one landed in every render. A bare entry
  always means OSPREY's artifact; a file named like one replaces it only where
  its path is listed, and listing both is refused. A facility hook is wired, and
  copied, by its `claude_code.hooks` declaration alone; a `hooks/<file>` list
  entry is accepted only for a file replacing an OSPREY hook. Guards are a floor:
  every `PreToolUse` hook in force on the deployment's render, OSPREY's or the
  facility's, reaches every tier and app render; an app runs a host hook on any
  other event only when its own `config:` declares it. The build warns about a
  file nothing selects, and `osprey init --revalidate` names every one in
  `profile.yml` and each app, and declares the host's non-guard hooks in each
  app, so existing renders stay as they are. A facility description left only in
  `build/` by an older repo is moved into the profile by `osprey init
  --revalidate`, no longer by the build. (#2002)
- The write hooks' `is_write_call` reads the `execution_mode` of any mapping, not
  only a dict, as `osprey_connectors.write_target` does. A test now holds the
  hooks' stdlib copies of the write-tool helpers and the launch-pin term to their
  connector originals, answer for answer. (#2113)
- The Bluesky queue tools' launch-token refusals, and the Bluesky plans skill,
  no longer name config files. They say a token reaches the agent only from the
  operator and send the action to the BLUESKY queue panel. (#2116)
- An app file (`apps/<name>.yml`) may now set a top-level `model:` to run a model other than the
  deployment's, for example a cheaper one on a shared research card. The provider is still
  `profile.yml`'s; an id its `models` list does not carry is used as written and warned about, as
  for `profile.yml`, and an app that names no model runs the deployment's as before. A
  `config: claude_code.default_model` line in a tier or app file is now refused naming that file
  instead of `profile.yml`, and `osprey init --revalidate` names a retired card's `model:` in the
  step it leaves for carrying an edited card into its app. (#2124)
- The live stand-in no longer carries BPM readout offsets, so the build no longer refuses a stand-in whose environment serves no lattice. The virtual accelerator containers append their model logs under `var/simulator/` (the stand-in under `var/simulator/standin/`).
- The archive recorder records the `channels` of the simulator view's `addresses.json`, mounted read-only from `build/data/simulator`; its compose service no longer reads `VA_CHANNELS_FILE`. A build that configures the recorder stops with `view-unsupported` on a channel address the archive cannot store as a field name (one containing `.`, starting with `$`, or holding a NUL byte).
- The virtual accelerator's tick comes from `simulation.tick_s` (default 1.0 s); a `VA_POLL_INTERVAL_S` in the project `.env` is not read.
  The virtual-accelerator compose blocks no longer read `VA_BPM_ERRORS`, `VA_STANDIN_BPM_ERRORS`, `VA_CORR_GAIN`, `VA_STUCK_SETPOINTS`, `VA_CHANNELS_FILE` or `VA_LATTICE`.
  `osprey build` no longer writes the virtual accelerator's channel manifest or the derived `VA_CHANNELS_FILE`/`VA_LATTICE` keys into `.env`.
  `var/simulator/` and `var/simulator/standin/` are provisioned setgid and group-writable before compose runs; each virtual-accelerator instance mounts its own, and web terminals and dispatch workers mount `var/simulator/` whenever a simulated target is configured.
- The shipped `diagnose` skill reads each simulator model's status channel (`<code>:SIM:<model>:STATUS`, through `channel_read`) and its model log, states a model's failure reason only from the status value, and names `osprey sim status` as the operator's terminal equivalent. (#8)
- A `control-assistant` deployment opens its sessions on the sandbox simulator
  instead of the live stand-in. The stand-in is still deployed, so reaching it is
  a deliberate target switch. (#892)
- The `control-assistant-readwrite` and `control-assistant-admin` tiers and the
  control-assistant root set `control_system.writes: allowed`; the `readonly`
  tier, every app and the hello-world root set `never`. A tier that allows
  writes arms no machine by itself: each stays off until its own switch is on. (#892)
- The artifact gallery loads the newest artifacts straight away and fetches more
  as the list is scrolled, so a workspace with thousands of them opens as fast as
  a small one; `artifact_server.page_size` sets how many arrive at a time
  (default 20). The gallery's filter matches a title, file name, description or
  type, within the session in view or across every session. (#924)
- `osprey set config.claude_code.agent_models.<agent>=<id>` makes the checks
  `osprey set model=` makes: it refuses a bare `haiku`, `sonnet` or `opus` and
  notes an id the provider does not list. It also refuses, before anything is
  written, the pins `osprey build` refuses.
- Web terminal panes are now rounded cards on a darker canvas, with a 40 px
  header row. The terminal pane, formerly labelled SESSION, is now the Agent
  pane. In the Simple view, its
  one header row shows the status dot, `Agent`, New conversation,
  notifications, Export, a ⋯ menu of earlier conversations, and a close
  button. The Simple view no longer shows `Connected` or `+ New`. To open a
  closed Agent pane again, use the Agent row in the sidebar.
- An agent's `Bash` and `Monitor` commands start without the panel token. OSPREY
  points `CLAUDE_ENV_FILE` at a script that unsets it, so only the agent's MCP
  client and hooks hold it, and a shell command can no longer fire a dispatcher
  job or call a panel's MCP tools around their approval prompts. The build's
  warning for a profile-declared shell gate no longer names those routes.
- The EPICS connector speaks Channel Access through aioca instead of pyepics,
  and pyepics is no longer a dependency: aioca is the one Channel Access client,
  on one long-lived event loop per process, so a process that also uses aioca
  (ophyd-async devices, for example) shares one CA context instead of two. Every
  read now asks the IOC, so `fresh_reads` no longer changes anything. A write to
  a channel that never connects reports that nothing was sent. A `connect()`
  that points Channel Access at a different gateway gives the process a fresh CA
  context, as pyepics did, closing the old one's channels; the connector's
  monitors are opened again on the new context and keep delivering. `epics_ca`
  dispatch triggers watch through the connector, so they stay armed across such
  a repoint too. The archiver recorder samples through the connector as well,
  with the `EPICS_CA_*` its container is started with, so its channels live on
  the process's one CA loop; one unreachable channel still costs only itself,
  and every sample is still bounded by the cadence. Text written to a numeric
  channel is read by that channel's own field type before it is sent, so `"1e3"`
  is refused for an integer field instead of landing as 1; text written to a
  char waveform is stored as a string; and a put the IOC answers without taking
  (a disabled record) is reported as failed.
- The `als-apg` provider now ships the gateway endpoint it fronts,
  `https://llm.als.lbl.gov`, so a deployment that names none reaches it without
  configuration. `ALS_APG_BASE_URL` and `api.providers.als-apg.base_url` remain
  overrides for a site whose gateway is elsewhere, and the generated
  `.env.example` no longer lists the endpoint as a value you have to supply.
- After the operator approves a call at the approval prompt, the agent is told so,
  whether the call then succeeds or fails. The agent can no longer report an
  approved call as auto-approved. Calls that a safety hook refuses, and calls
  that the operator declines, get no such note.
- The notebook the approval prompt links for a Python run is titled *Approval
  preview* (was *Pre-Execution Review*): it shows the code the operator is about
  to approve and reviews nothing.
- The logbook and knowledge apps no longer carry the JUPYTER panel, and they ship `web.control_target_picker: false`: nothing in them reaches a control target, and the title bar names none.
- The archive's seeded history, and every window `osprey sim apply` rewrites, is computed by the same composite the simulator serves, readout faults and clamps included; existing stores reseed once on the next `osprey up`. `osprey sim apply` activates the scenarios the simulator view lists and seeds the logbook from their entries.
- **Breaking change:** history read on a control target comes only from that
  machine's own archive. The facility's archive (the EPICS appliance, MYA,
  DOOCS), when the `archiver:` block names one, serves the live machine and the
  stand-in that rehearses it; the deployment's own store serves the virtual
  accelerator, and the recorder samples only while the `va` target lands on
  this deployment's simulator. The stand-in no longer owns a store, the live
  target is no longer refused on the archive (`archive_belongs_to_standin` is no
  longer read), and `archiver_read` answers `no_archive` on a target the archive
  does not serve, naming the machine whose past it holds.
- `osprey build` and `osprey facility validate` stop with one `value-invalid` line on a scenario `archiver` event whose `at_when.days_ago` is negative, or whose `at_time` or `at_when.time` carries a timezone offset. Archiver events now read `{days_ago, time}` by the same rules as logbook entries.
- **Breaking change:** the archiver recorder's
  `RECORDING_CONTROL_SYSTEM` is renamed `RECORDED_TARGET`. It names the `va`
  target, the only one the recorder samples.
- **Breaking change:** the EPICS, MYA and DOOCS archivers spell their request
  bound `archiver.settings.timeout_s`; `timeout` is refused with a message naming
  the new key. MYA and DOOCS refuse an `auth:` or `tls:` block, because their
  client libraries cannot send a login or a per-connection CA.
- With `ariel.vocabulary.expand_modes` unset, semantic and hybrid search no longer
  expand the query while `ariel.vocabulary.expand_entries` is off: expanding the
  query alone makes dense retrieval worse than not expanding (JLAB-TN-26-024).
  Keyword search still expands; turning entry expansion on brings the dense modes
  back, and an explicit `expand_modes` list still names them directly.
- ARIEL in a web terminal tile no longer shows a second title in its body: the
  "Search the Logbook", "Browse Entries" and "New Entry" headings are hidden
  when embedded, because the tile bar already names the panel and marks the
  active view. The standalone page keeps them.
- An ARIEL entry whose text a search, `browse` or `entries_by_ids` result cuts
  short now says so: it carries `raw_text_truncated: true` and its full length
  in `raw_text_length`, and `entry_get` returns the whole entry.
- ARIEL's open `/health` page now reports the entry count, the last ingestion time and the enabled search and enhancement modules, under `service`. It never shows the database address or a driver's error text; those stay on the signed-in `/api/status`.
- The qmd mirror of an ARIEL entry now states the entry time in the facility zone with its UTC
  offset, the same string the ARIEL panel and the agent show. The next export or
  `osprey ariel qmd resync` rewrites every mirrored file once, so qmd re-indexes the whole
  logbook one time.
- The artifacts viewer in the web terminal no longer opens its own event stream.
  The hub's sidebar already follows that stream, and now tells the viewer when
  the artifact it shows is updated or deleted. This frees one of the browser's
  six connections to the hub, so tiles do not wait for a free connection.
- `osprey up` without `-d` now builds the service images in a step of its own
  before it hands the terminal to compose, the way `osprey up --dev` already
  did. That build's output is spooled to `var/logs/` instead of streaming;
  pass `--verbose` to watch it live.
- Audit records now carry millisecond timestamps, and every record a tool call
  produced names that call's `tool_use_id`, so a refusal, an approval prompt and
  the tool's own record join on one key.
- Safety decisions are now `audit` entries in the agent record instead of lines
  in per-surface files under `var/audit/<identity>/`. Each entry names the
  surface that decided (`hook_approval`, `scaffold_gallery`, `preflight`,
  `auth_sidecar` and the rest) in its `surface` field. The record keeps them for
  `agent_record.retention_days`, and the archive keeps each closed day in
  `var/archive/<day>/agent_record/entries.parquet`. `osprey reset` keeps the
  record and the archive; `osprey reset --purge-audit` deletes the record and its
  exports.
- The web-terminal login service no longer loads the deployment and command-line
  modules when it starts, so its process holds only the code it serves with.
- Every channel-finder index `osprey build` writes (`data/channel_finder/in_context.json`,
  `hierarchical.json`, `middle_layer.json`) states its row count as `count` beside `schema`: the
  rows of an in_context index, the leaves of a hierarchical tree, the distinct addresses a
  middle-layer index lists. `osprey channel-finder benchmark` reads that key for the run's
  `channel_count` instead of re-parsing the index, and takes the count before it sends a query: an
  index that is missing or states no `count` stops the run with an error naming the file, where it
  used to record 0. A top place whose id is `count` stops a middle-layer build with
  `view-unsupported`, as `schema` does.
- The shipped benchmark query files are `in_context_queries.json` and
  `tree_queries.json` under `data/benchmarks/cross_paradigm/queries/`. The build
  copies the one matching `channel_finder_mode` to `data/benchmarks/queries.json`:
  `in_context` takes `in_context_queries.json`, every other mode
  `tree_queries.json`.
- The `queue_add` and `queue_start` approval prompts now have 30 seconds rather
  than 5, enough to read the queue and render the preview they show.
- In the web terminal, the BLUESKY panel shows its Plans, Queue and Results tabs
  in the tile bar in Simple mode as well as Expert, and no longer draws a second
  set of tabs in its body. The panel's type sizes, weights, line heights, corner
  radii and transition times now come from the shared design tokens.
- The BLUESKY Plans view now shows a sentence in the plan detail pane when no plan is selected ("Select a plan to see its parameters."). Before, the pane was an empty bordered box. The sentence goes as soon as a plan is selected.
- A Bluesky move that times out names the bound its readback had to reach:
  `readback <address> did not settle to <demand> ± <bound> within <t>s`.
- A channel bridge reads a conversation's earlier turns back from the agent
  record instead of keeping its own copy, so how far back a follow-up can refer
  is `agent_record.retention_days` rather than a fixed 180 days. When the record
  cannot answer that read in full, the bridge sends no history at all and the
  payload's `conversation_record` key says why (`unavailable` or `incomplete`).
- `osprey build` no longer prints "No channel search index" for a project with no `services.graphdb` block: a project without a graph store has no index to miss.
- `osprey build` and `osprey facility validate` read and write the facility's YAML through libyaml when PyYAML has it, cutting the YAML share of a large import's build several-fold; outputs are unchanged.
- The bundled demo facilities live in `src/osprey/templates/facilities/` (`example/`, `hello_world/`), owned by no app template. A preset names the one it shows with the preset-only key `facility:`; `osprey init` copies it into the profile's `data/facility/` beside the app template's `data/`, and `osprey scaffold pull` offers it at the same paths. The example facility's `identity.yaml` now states its display name, `Example Research Facility`. A profile spelling `facility:` is refused.
- A deployment that names no CANOPY payload (`OSPREY_CANOPY_PAYLOAD` unset) now builds without the CANOPY panel and its skills, with one line saying so, instead of stopping the build. A named payload that is missing, half-built or built at other commits still stops it.
- The CANOPY panel follows the CANOPY that went upstream: the screen report is `LIBRARY-REPORT.*` (schema `canopy-library-report/1`), the collection check is `canopy_mcp.collections_cli`, the sidecar sets `CANOPY_URL_PREFIX`, `CANOPY_PROXY_MARK_HEADERS` and `CANOPY_LIBRARY_IGNORED_NAMES`, and the mount stamps `data-canopy-storage-scope`.
- The web terminal's panel messages take the names of CANOPY's host contract:
  the terminal sends `canopy-host-header-action`, `canopy-host-navigate` and
  `canopy-host-navigate-probe`, and a panel sends
  `canopy-host-header-contribution`, `canopy-host-navigate-ready` and
  `canopy-host-focus-exit`. For one release the terminal still reads the old
  `osprey-header-contribution`, `osprey-navigate-ready` and `osprey-focus-exit`.
- Channel Finder: the graph-mode Explore view follows the fleet layout. The device card opens beside the results, the corpus provenance sits in the strip above them, facets fold and clear one at a time, and inside the terminal the search box and ⋯ menu move into the panel's tile bar.
- Every failure about one channel is a `ChannelError` that names the channel the same way in process and across the connector host.
- **Breaking change:** `stuck` under a scenario's `faults` stops the build; write `channel_faults: {<address>: stuck}`. Channel faults `frozen` and `disconnected` are new, and any simulated channel, wired or not, can carry one.
- The channel finder keeps every Field a channel is listed under: a channel
  listed under several Fields or Subfields of one Family is one membership, and
  one `channels` row, per listing, instead of keeping only the last.
- `osprey channel-finder validate` and `osprey channel-finder preview` read an
  in_context database as the index the build writes: an object with
  `"schema": "osprey.facility.channel_finder/1"` and one `channel`, `address` and
  `description` per row. A bare list or a template database is refused, and the
  preview has no presentation mode.
- The channel-finder index views under `data/channel_finder/` are build output: a
  `project/` mirror file there stops the build with a `profile-invalid` line, and
  no agent-side writer may write one.
- **Breaking change:** `osprey build` writes each channel-finder index from the
  facility description, and its hierarchical index differs from the hand-written
  databases it replaces.

  - The hierarchical index's levels are the facility's place level words, then
    `class`, `device` and `leaf`. Every level is present for every channel, and
    the node `-` names an absent place, class or device. The `field` and
    `subfield` levels are dropped: their descriptions are kept in the groups'
    `signals` sentences and shown on leaves and middle_layer Fields. A query or
    a saved path that walks the old levels no longer resolves.

  The three `channel_finder.pipelines.<pipeline>.database.type` keys are gone:
  each pipeline reads the one index the build writes, so there is no loader to
  choose.
- A channel's meaning is its `signal` and its `description`: a group no longer carries per-signal sentences, and a group stating `signals` is refused. The build describes a channel that states no description but names a signal from its device, its signal and its unit. The middle-layer channel-finder index keys Fields by signal and files the devices of a class no single-class group holds as a Family named by the class. The MML import writes one group per physical family and describes each channel by its device and its field sentence.
- The channel roster reads the facility file a build writes at the root of every
  render (`facility.json`), whatever channel-finder mode a project configures and
  whether or not it configures one. The plan-device file, the channel-suggestions
  snapshot, the channel finder's membership routes and the virtual-accelerator
  channel set are all taken from its channel records: a record's `role` states
  the direction, a setpoint's `pair` states its readback, and the addresses a
  simulator serves for its own models' status are never listed. A render no build
  has written the file into reports `The facility file facility.json is not
  built` and names `osprey build`. The build's virtual-accelerator fact names the
  source as `this project's facility file (facility.json)`. The channel finder's
  membership and enumeration routes answer 503 naming `osprey build` as the
  remedy when the render holds no facility file. A facility file that declares no
  channels reports `The facility file facility.json declares no channels: the
  project's data/facility tree holds no channel records.`; the plan-device file
  is not staged, and the channel finder's 503 names `data/facility` as the remedy.
  A project that stages no channel database gets its virtual-accelerator channel
  set from the facility file in every channel-finder mode, read from the render
  being built. Such a manifest carries `_metadata.source_paradigms: []` — no
  channel database fed it, and `source_corpus` names `facility.json` — where it
  carried `["graph"]`; a manifest built from staged channel databases is
  unchanged. When the facility file declares no channels, a build deploying a
  virtual accelerator stops with that sentence in place of `no channel database
  is staged at tier N`.
- The pinned Claude Code CLI moves to 2.1.300, so the generated projects, the
  dispatch worker image and the hosts that run them all use one Claude Code
  version. The dispatch worker picks it up when its image is rebuilt.
- The web terminal clock always names its zone. It shows the facility's city,
  or the viewer's own short zone name (for example `PST`) when the browser cannot
  resolve `system.timezone`. A deployment still on the default
  `system.timezone: UTC` therefore shows `UTC` beside the clock.
- The control-assistant mock serves every channel the lattice physics model computes from the lattice deck, and every other channel from its seeds. With no scenario active it reads the same value and the same noise as before on every channel, except these, re-baselined on purpose: each `SR` beam position monitor's horizontal position (`SR:DIAG:BPM:<nn>:POSITION:X`, 72 channels) reads the deck's closed orbit, at most 20 micrometres, in place of 0; the wired RF cavity's `SR:RF:CAVITY:01:FREQUENCY:SP` and `SR:RF:CAVITY:01:FREQUENCY:RB` read the deck's 500.417 MHz in place of 499.654 MHz; the six `SR:VAC:ION-PUMP:<nn>:VOLTAGE:SP` setpoints read what they hold, with no noise; and every bool channel reads its label (`TRUE` or `FALSE`) with no noise, in place of a noisy number. A float channel's noise is the `noise` of its seed: a read adds that sigma times a draw keyed on the channel and the instant, on top of its drift. The `hello-world` project's mock reads a stored beam current near 250 mA on `SR:BEAM:CURRENT`, in place of 0.
- The simulator's composite now reads each waveform channel as an array of the shape it declares, and the model RPC's `get` returns a waveform in that shape.
- One dead channel no longer restarts the connector host. A connector now raises
  `ChannelUnreachableError` for a channel it cannot reach, which the controls
  tools report as `channel_unreachable` and recover nothing for, and
  `ConnectorUnavailableError` when its own transport is gone, which its
  supervisor replaces once and names under `details.recovery`.
- A connector host that does not answer within its start bound, or exits before answering, now says how far it had got — interpreter up, init frame read, connector modules imported, connector built — and when, so a slow start names the phase that was slow.
- Connector-mediated settables implement Bluesky's `Locatable`, so a plan can ask `bps.locate(device)` where a device was told to go and where it is. The shipped plans now read their working points that way.
- **Breaking change:** the EPICS, TANGO and virtual-accelerator connectors read their call bound from
  `control_system.connector.<type>.timeout_s`, as DOOCS does; a config whose connector block still
  carries `timeout` is refused when it is read, naming its new spelling. Every control-system connector
  refuses a `timeout_s` that is not a positive number of seconds when it connects.
- `osprey-connectors` now versions from its own `osprey-connectors-vYYYY.M.P`
  calendar tags instead of the framework's `v*` tags, so it releases when the
  package changes rather than with every framework release. Each release stays
  backward-compatible with every supported framework release, and the framework
  keeps declaring only a floor on it. A checkout building the package needs those
  tags fetched (a fork or mirror must sync the `osprey-connectors-v*` tags):
  without one reachable it builds as `0.0.postN`, and `osprey up --dev` now
  refuses such a build before staging it, naming the tag glob.
- `osprey-connectors` now releases on its own: an `osprey-connectors-vYYYY.M.P`
  tag publishes it from `release-connectors.yml`, gated on a green CI run of the
  tagged tree, with release notes from the package's own
  `packages/osprey-connectors/CHANGELOG.md`. A framework `v*` tag no longer builds
  or publishes `osprey-connectors`; the framework's `osprey-connectors>=` floor
  must already be on PyPI, which the release now checks.
- The `contribute` skill runs the changelog fragment check on the working tree
  before the commit, so a missing fragment is caught while it can still join it.
- **Breaking change:** shared writers reach the control system under fixed
  service names. Dispatch worker *i* writes as `osprey-dispatch-<i>` instead of
  `osprey`, and each Bluesky lane OSPREY renders writes as `osprey-bluesky` or
  `osprey-bluesky-<lane>` instead of `root`; external-worker lanes are
  unchanged. A container or lane that cannot apply its service name (a
  read-only root filesystem, a non-root start) logs a warning and starts under
  its old name. Update any gateway or IOC access-security (ACF) or put-log rule
  keyed on `osprey` or `root`, and roll the images and the rendered project
  together: `osprey health` reports a worker writing under a different name
  than the render expects.
- `ControlSession` judges a change of its target's write switch per write
  instead of ending the process: only a move of the control record, or a switch
  change that asks for the other gateway, exits 75. At
  the `asked` position a write needs `confirmed=True` and is refused
  `approval_missing` without it. `open(session_keys=...)` names the sessions a
  process writes for; it is held while any of them is, and refuses every write
  when the list cannot be read. A write's audit detail names its `position`,
  `confirmed` and `sessions`.
- Web terminal: the sidebar's control-target block is one row — machine, write
  state, service health and an arrow — without the "Control system" caption, the
  clock or the second line. Its popover now opens and closes like every other
  sidebar popover, on the design system's popover rows, switches and buttons,
  and keeps focus on the same control when it refreshes.
- The control-target switch refuses a `va` or `standin` target whose selected
  gateway has the same address and port as one of the `live` target's gateways.
  The roster reports it as ineligible with reason `reaches_live_machine` ("points
  at live machine"). A deployment that pointed its virtual accelerator or stand-in
  at a live gateway can no longer switch to it.
- Control-target records, reports, switch requests and in-flight markers now
  live per user under `var/agent_data/control_target/<identity>/`, bound from
  the host in multi-user deployments. Pick the control target again after
  upgrading: the old single-file record is not read.
- The example facility's simulated readback noise and drift now come from a per-signal rule table:
  magnet, RF frequency, tuner, cavity voltage and high-voltage readbacks read with instrument-sized
  noise, every corrector readback carries the same slow drift, and stored reference readbacks no
  longer move.
- The dispatch worker's tool denylist names the background-command tool as the
  CLI now calls it, `TaskStop`, so a trigger that lists it is refused with `403`.
  Neither unattended floor lists `TaskOutput` or `MultiEdit`, which the pinned CLI
  no longer has.

  Both unattended floors, the dispatch denylist and the `osprey query` floor, also
  deny `Monitor`, `Workflow`, `CronCreate`, `ScheduleWakeup`, `SendMessage` and
  `EnterWorktree`.
- `osprey build` now writes a registry-mode profile's `deploy.registry.url` into the
  rendered `registry.url` when the profile's `config:` block names none, so the web
  terminals' registry is stated once. A `registry.url` in `config:` still wins.
- `osprey up` records the runtime and compose project it started in `var/deployment.json`, and `scripts/verify.sh` checks that deployment instead of guessing it; `osprey reset` plans against the recorded project.
- A built image installs the profile's own `dependencies:` under the versions the framework resolved, frozen to `/opt/osprey/constraints.txt`: a profile can add packages but never move one the framework needs, and a build where it would fails instead of shipping a mismatched image. The deps layer closes with `pip check` (skipped under `OSPREY_DEV=1`); a dropped wheel installs under the same file minus its own lines, so it may change its own version and nothing else.
- The OSPREY design principles gain an eighth: where two lines of work meet,
  design from the end product. The `/osprey:design-philosophy` skill and the
  contributor guide apply it to every merge, conflict resolution and plan
  written against an older tree.
- The dock layout engine is CANOPY's: `packages/dock-engine` leaves OSPREY, and
  the web terminal serves the screen runtime's built `dock-engine.mjs` from
  `/static/vendor/` as a tracked byte copy pinned to one runtime commit, with the
  server's layout rules in `osprey.interfaces.web_terminal.dock_layout` replaying
  the engine's own cases. The retired `slot` and key-leaf layout shapes are now
  refused as `not_layout`.
- A web terminal tile holds tabs. Its header shows the tab in front — the tab
  strip once it holds more than one — with that tab's controls, Pop out and ×.
  Drops come from the dock engine: a tab or a tile dropped on a tile's header or
  centre joins it as a tab, on its edge splits it, and on the work pane's edge
  makes a new tile along that side; a sidebar row makes a new tile and never
  tabs in; the agent pane is always its own tile. The highlight is where the
  item lands.
- Engine plug-ins have a written contract, `osprey.simulation.engines.EnginePlugin`, naming the functions a plug-in states and the optional ones. The model response-matrix tool and the response check of `osprey facility validate` reach a model's engine only through its plug-in and read a request with no solve from what that plug-in declares, so they import no engine library.
- `osprey ariel enhance` run while another pass holds a picture module now says
  `<module>: skipped, another pass is running it (osprey ariel watch runs the
  picture modules on every poll)` instead of `running in another process`.
- The plan preview on a `queue_add` prompt now spends that prompt's own time
  budget rather than a fixed 15 seconds, so a slow bridge costs the trajectory
  and says so, never the prompt itself.
- `osprey-connectors` now requires p4p 4.3 or newer, which brings pvxslibs 1.5.3
  and epicscorelibs 7.0.10.99.0.2. On Python 3.14 these install from prebuilt
  wheels on linux aarch64 too, so an arm64 install there no longer needs a C
  compiler; on Python 3.11 to 3.13, arm64 still builds them from source.
- The CLI help, module docstrings and the CLI reference use `my-assistant` as the
  example deployment name, so no example in OSPREY's own text names a facility.
- Configuration names models by the id the gateway serves. The haiku / sonnet /
  opus tier words are gone from profiles, presets and the provider catalog; each
  catalog entry lists the models its gateway serves and names its default.
  Claude Code's own three alias names are filled at build from that list (or
  from a gateway's optional `claude_code_aliases`). `claude_code.models` is now
  `claude_code.aliases`; `logbook.composition.default_tier` is now
  `logbook.composition.model`. Model names shown to people drop the vendor
  prefix: Sonnet 5, Haiku 4.5, Fable 5.1.
  A deployment on anthropic that names no model now runs Sonnet 5 where it ran
  Haiku 4.5. A deployment on als-apg that names no model now runs Sonnet 5 too,
  and its health check still probes Haiku 4.5. The als-apg entry no longer lists
  Opus 5 (`claude-opus-5`); name Opus 5.5 (`claude-opus-5-5`) instead.
  The als-apg entry also lists the GPT-6 family (`gpt-6-sol`, `gpt-6-luna`,
  `gpt-6-astra`).
- The `facility_description` tool returns the build's generated facts page
  (`data/facility_facts.md`) under `generated` alongside the hand-written
  description, and `source` names both paths. The not-found envelope is raised
  only when both files are absent, so an agent sees the generated facts where no
  hand-written page exists.
- The web-terminal landing title, the sign-in page's facility name, the event
  dispatcher's default name and the agent prompts take the facility's name from
  the build's facility identity (`data/facility/identity.yaml`), else the project
  name; the config's `facility.name` and top-level `facility_name` no longer name
  anything. A facility file that is present but unreadable, not JSON, or missing
  its identity code is refused with an error that names the file.
- `osprey build` stops on a float channel whose `precision` is above 17 (`value-invalid`, "`precision` is not an int from 0 to 17", naming the channel and the file that states it); a double carries no more than 17 significant digits.
- The `control-assistant` preset no longer ships an empty `data/facility/classes.yaml`, matching
  `hello-world`. A tree without the file builds with no added classes, and
  `osprey facility import mml` on a preset now seeds `classes.yaml` with the
  classes its mapping adds.
- `osprey facility validate` prints one more stderr line per model when the response check left rows out: `response check <model>: left out <n> rows (<k> unwired, <j> no width, <u> unsolved, <t> table calibration)`. A row the export marks `Status` 0 is not counted, and a monitor whose calibration is a table is left out of the comparison. A model that leaves nothing out prints only its verdict line, which is unchanged.
- A facility build computes its model wiring, engine lookup, graph Turtle and
  parsed facility file once and reuses them for every render, instead of
  recomputing them per render.
- The facts the agent reads about a facility now carry its places with the devices
  at or under each, the signals and roles its channels use, and list a device
  class's groups as groups. The channel-finder prompts and the graph tools
  describe the generated database in OSPREY's words and identify a device by its
  id.
- Every render with a `services.graphdb` block seeds its store from the graph
  view the build writes, `./data/graph/facility.ttl`, and ships the search index
  `data/channel_databases/graph.duckdb` derived from it, whatever the
  channel-finder mode. The index's bindings carry each device's place path,
  s position in metres and ordinal in its place (index schema version 2), and
  the deploy refuses a rendered graph store block that names no corpus.
- Graph readers: the graph tools' example queries, the channel finder's device card, the device stubs and the graph agents' prompts read the predicates the graph view writes. A device's ordinal is `ordinalInPlace`; a binding's prose is its `description`, with no field or subfield description; bindings carry no `protocol` or `confidence`; a device's descriptions are `familyDescription` and `systemDescription`. Example queries that mean devices skip the places that carry channels of their own, and a device stub's Channel column names each channel by its address, with no Protocol column.
- Graph store: `osprey up` compares the store's seed marker with the digest of the configured knowledge-graph TTL and, on a mismatch, wipes the store in bounded transactions and imports the TTL, so a changed corpus reaches the store with no manual seeding step. A store already holding the corpus is left as it is.
  The seed marker and the `graphdb_seed` health row carry the digest only; the direction source is no longer recorded on the marker or stated in the graph agents' prompt snapshot.
- `osprey up` wipes and reseeds a graph store whose neosemantics configuration is not OSPREY's, as it does a store holding another corpus, instead of leaving it as it was.
- **Breaking change:** the knowledge graph is the graph view `osprey build` writes from the facility file, and three things differ from the hand-written corpus it replaces.

  - Every node IRI is minted from the facility identity's `code` rather than the word `demo`: a device's local name is `<code>_device_<escaped id>` under `https://narad.example.org/device/`, and places, channels and groups follow the same `<code>_<kind>_` form. A query or a saved link that spells an old `demo_` IRI no longer matches; `deviceId` and `bindingId` carry the raw ids.
  - A device's ordinals are `ordinalInPlace` (its position among the devices of its class in its place) and `ordinalInModel` (among the devices of its class in its model); `ordinalInFacility` and `ordinalInSection` are gone.
  - `sPositionM` and `lengthM` are the device's position and length in metres along the lattice, written only when the facility file carries them, where the hand corpus carried placeholder positions.
- The graph view under `data/graph/` and the graph index at
  `data/channel_databases/graph.duckdb` are build output: a `project/` mirror file
  at either path stops the build with a `profile-invalid` line, and no agent-side
  writer may write either path.
- Graph store: health, deploy and agent-facing remedies for an empty, stale or unbootstrapped graph now name `osprey build && osprey up`.
- Guarded runs: a run refused because another holds the target's lock is now
  told whose run it is, its target and the name it writes under, besides its
  process id and start time; the lock file records the same fields.
- The documentation no longer describes the agent harness or the code-execution
  backend as swappable. Claude Code is the harness OSPREY ships, and the
  contributor guide names `src/osprey/agent_runner/` as the package that holds
  its code.
- The control-assistant preset runs every helper agent on the deployment's main
  model. On anthropic and als-apg, logbook deep research moves from Opus 5.5 to
  the main model. On cborg, the channel finder, the knowledge-graph agent and
  deep research move to Haiku 4.5, the provider's default. On every other
  provider those three agents now work instead of sending a model id the
  provider does not serve. A deployment made from the preset earlier keeps the
  three `claude_code.agent_models` lines in its own `profile.yml`, and
  `osprey validate` now reports them as a difference from the preset: remove
  them, or mark them `# DEVIATION:` to keep the pins.
- A profile names only the hooks it wants in `hooks:`; the helper files those hooks need ship with them, and naming a helper is refused. `osprey init --revalidate` removes the helpers an existing profile names.
- Code that calls ARIEL's `hybrid_search` directly passes the search service's
  search-call pool as `search_calls=`, as `semantic_search` already requires. The
  ARIEL service passes it itself, so searches through the service, the MCP tools
  and the web interface are unchanged.
- Image builds keep their pip and apt downloads between builds. Every pip and
  apt install in the project, tier, app and service images fetches through a
  BuildKit cache mount, so a layer that has to rebuild (a new framework pin, a
  changed dependency manifest) downloads only what changed, and images building
  in the same deploy share one copy.
- Image builds install the framework with uv. The deps layer of the project,
  tier, app and service images resolves and downloads with uv instead of pip,
  which is much faster on a slow package index; uv is installed from the same
  index the site configures, honours the same constraints and index settings,
  and is removed again before the layer is committed.
- `osprey build` stops with `view-unsupported` when two channels take one key of the hierarchical or middle-layer channel-finder index, which happens when one channel's address is another's signal; before, the later channel silently replaced the earlier one.
- The retired `--config` refusal and the `build_profile` errors of a tier or app from `osprey scaffold web-terminals lint` and deploy link the guided setup's documentation page, instead of naming a slash command for one coding-agent client.
- A refusal that sends an operator to the guided setup now ends with the same sentence everywhere: the guided setup's link, then what the setup does for them.
- The JUPYTER tab follows the light or dark side of the terminal's theme: after
  JupyterLab starts, it shows *JupyterLab Dark* or *JupyterLab Light*.
  JupyterLab stores each change, so a JupyterLab page that you open at its own
  address starts in the theme that the tab used last.
- A notebook cannot import a control-system client library (epics, p4p, caproto,
  aioca, pvaccess, tango, doocs4py, epicscorelibs) or load one through ctypes or
  cffi, for reads as for writes; ophyd v1 and `ophyd_async.epics` are refused
  with them, so PVAccess puts, rpc, Tango commands and serving a PV from a cell
  are no longer possible in a kernel. Every kernel, on a single-target or
  in-process simulator deployment too, reads and writes through a connector-host
  child it starts on its first control-system call.
- An agent-side writer may write under `data/facility/knowledge/`; every other path under `data/facility/` stays refused, and a `project/` mirror file anywhere under `data/facility/` still stops the build.
- The `control-assistant` preset's knowledge pages live under `data/facility/knowledge`, and its `facility_knowledge.bundle_path` names that directory; they were under `data/facility_knowledge`. `osprey scaffold pull control-assistant:data/facility/knowledge` reads the new directory, and the reference image hands the agent's user the directory `facility_knowledge.bundle_path` names. A deployment materialized earlier moves its pages with `git mv data/facility_knowledge data/facility/knowledge` and sets `facility_knowledge.bundle_path: data/facility/knowledge`.
- `osprey knowledge seed-from-ttl` takes the bundle as its one argument and reads the build's graph view, `build/data/graph/facility.ttl` in the deployment repo you stand in or `--repo` names, unless `--ttl` names another Turtle file: every stub's `device_id` is the facility file's device id, so the build links the page to its device.
- The agent prompts, tool descriptions and dashboards name the accelerator model
  a lattice and its file a deck, and the hierarchical channel-finder tools show
  their examples with the levels a built index has (`machine`, `sector`,
  `class`, `device`, `leaf`). No CLI verb, config key or tool name changes.
- `api_protocol` in a `providers.yml` entry now overrides the provider's own
  protocol in both directions: `api_protocol: openai` on `anthropic`, `cborg` or
  `als-apg` routes the launch through the translation proxy instead of being
  ignored. A provider class registered through `ProviderRegistration` that
  declares `api_protocol = "anthropic"` launches without the proxy.
- A Bluesky arm refused for a missing or wrong launch token now answers
  `launch_token_invalid`; `launch_token_required` means the deployment has no
  launch token configured at all. One shared check, `osprey.utils.launch_token`,
  gives the same coded answer on every path.
- The agent now reads a launched run's data back before analysing it.
- `import osprey` no longer resolves `osprey.__version__`: it is resolved on first read, so a
  short-lived process that imports osprey without reading the version, such as a Claude Code hook,
  does not run `git describe` in a source checkout. `from osprey import __version__` and
  `osprey.__version__` read as before.
- A channel-limits database is required at build time only when limits checking
  is enabled for a target that arms writes. With checking off, an armed
  deployment builds and runs without one, as the connector already allowed, and
  the compose files it renders mount no limits database.
- **Breaking change:** `osprey build`, `osprey validate` and `osprey facility validate` stop with one `profile-invalid` line when a profile, a tier or app file, a host variant or a `--set` edit states `control_system.limits_checking.database_path` (or a per-type `…limits_checking.database_path`). The limits database is always the one the build renders from `data/facility/limits.yaml`. To upgrade, write the old table's records into `data/facility/limits.yaml`, delete the key from the profile and delete the old table file.
- **Breaking change:** `osprey build` and `osprey facility validate` stop with one `profile-invalid` line when a profile ships its own `data/channel_limits.json`, `data/simulation/channel_limits.json` or `project/data/channel_limits.json`; limits are authored in `data/facility/limits.yaml`, which the build renders into the limits database. `osprey facility import mml` seeds the export's bands into `limits.yaml`, and the control-assistant preset no longer ships a hand-written limits file.
- **Breaking change:** `control_system.limits_checking.mode: exclusive | optional`
  replaces `allow_unlisted_channels` (`false` is `exclusive`, `true` is
  `optional`), deployment-wide and per connector type. Under `exclusive` only
  channels in the limits file can be written; under `optional` channels in the
  file are held to their limits and every other channel is written with no
  limits. `osprey build` and `osprey validate` refuse any other `mode` and any
  leaf a limits block does not define. The `control-assistant` and `hello-world`
  presets ship `mode: optional`, the `control-assistant` preset writes no
  per-type limits block, the `channel_limits` tool reports `mode` and `mode_key`,
  and switching a session onto `live` or `standin` no longer depends on the
  limits settings.
- **Breaking change:** the channel-limits database reader no longer accepts a `defaults` block. Every entry states its own `writable`; an entry without it, or a top-level `defaults` key, fails the load, and limits checking then blocks every write. `confirm` is read from the channel's own entry, else confirmation is on. The `channel_limits` MCP tool no longer reports a `defaults` field. Databases rendered by `osprey build` already hold one entry per record with `writable` stated.
- **Breaking change:** `osprey build` writes each render's
  `data/channel_limits.json` from `data/facility/limits.yaml`: `_version: "4.0"`,
  one entry per limits record with `writable` and `confirm` stated, and no
  `defaults` block. A channel with no record has no entry and follows
  `control_system.limits_checking.mode`. The build derives
  `control_system.limits_checking.database_path`, so a profile no longer states
  it, and the `hello-world` preset no longer ships a hand-written
  `data/channel_limits.json`.
- The logbook app pins `control_system.writes: never`, as the knowledge app does, so a profile whose tier allows writes cannot arm it for the logbook.
- The controls server's target switch now launches and retires its connector-host child through
  the same handshake and teardown as the connector pool. A connector-host child now exits when its
  parent process changes, including when it is handed to a subreaper rather than to init. A
  restarting controls server no longer signals PIDs recorded by a dead predecessor.
- The approval prompt for a `manual_fire` names the trigger the fire would
  start, and the payload where the call carries one, so the name to check is on
  the prompt rather than in the agent's message above it.
- The deny defaults name the whole plugin and claude.ai connector tool namespaces,
  `mcp__plugin_*` and `mcp__claude_ai_*`, in place of the Playwright and Context7
  entries, on the interactive floor, the headless read-only floor and the event
  dispatch denylist. An open deployment now requires every tier and app to deny
  `mcp__plugin_*`.
- `osprey build` refuses an MCP server whose name puts its tools under
  `mcp__plugin_` or `mcp__claude_ai_`, since every launch denies those
  namespaces; rename the server in the profile's `mcp_servers:`.
- The middle-layer index and the agent facts name every facility group as a family; a channel belongs to every family it is in, so the channel finder's `channels` table holds one row per channel and family; a middle-layer render of a facility with no group stops the build.
- In the middle-layer channel-finder index, a device whose place has no sibling
  places — one at the machine itself, in an only-child place, or with no place —
  now has the `DeviceList` row `[0, k]`, where it had `[n, 1]` by its position in
  the family; no two rows of a family are equal. Each family's `setup` gains
  `PlaceList`, every device's place beside its row. `list_channels` takes `place`
  (a place id or path prefix) where it took `sectors`, the `device_map` table has
  `place` and `place_index` where it had `sector`, and the explorer names each
  group and filter chip by its place.
- `mml_export.m` 2.1.0 writes a sixth file, `<stem>.model.json`: what the Middle
  Layer's model answers for the exported lattice (tune, chromaticity, dispersion
  and their responses, in physics and hardware units). Each section reloads the
  saved lattice first, and a section that fails is recorded as refused rather than
  stopping the export. The exporter ships with the mml layer as well as in the
  control-assistant data template, and the template README describes re-running
  the export on a MATLAB host.
- `osprey facility import mml` into a tree that already has a `classes.yaml` now appends the classes the import derives and the file lacks, instead of leaving them out and stopping the build on an unknown class. Rows already in the file are kept as written.
- The mock connector serves the addresses of the built facility file from the simulator view and refuses any other address with `<address> is not a channel of this facility`; it needs `osprey build` before it connects.
  The mock connector refuses a write to a channel that is not a writable setpoint before anything is put, and keeps session writes in a journal under the simulation state directory, which every mock connector on the same build replays and which a reset or a change of the active scenarios empties; a journal that is not replayed is logged in the process log and in every physics model's log.
  The mock archiver serves the history of the built simulator view through the archive composite and refuses any other address; it no longer reads a machine file.
- `model_response_matrix` names each corrector with no solve under `missing`,
  with the engine's text. When no corrector solves, the call fails as
  `nothing_answered` with the same map under `details.missing`.
- When no corrector solves, the refusal of `model_response_matrix` now carries the unsolved correctors as a map in `details.unsolved`, the same map a partial result returns as `unsolved`. An engine that refuses the request is now refused as a `validation_error` with the engine's own text and no corrector advice.
- `model_response_matrix` reports a request the engine refuses with the engine's text, a broken facility file with the remedy to rebuild, and any other engine error as a failure with its stderr.
- **Breaking change:** the MongoDB archiver block names its login under `auth:`
  (`auth.source`, `auth.username`, `auth.password_env`) and its wait as
  `timeout_s`. The flat `auth`, `username`, `password_env` and `timeout` keys are
  no longer read. A project that deploys its own store gets the new block from
  `osprey build`.
- A container that cannot join a bind mount's group now warns that the mount
  will be unreachable rather than unwritable. The same step joins the read-only
  control-state tree, which nothing in the container writes, so the old wording
  sent operators looking for a writer that does not exist.
- **Breaking change:** a channel's `noise` is one record in `seeds.yaml` and in a scenario's `noise` block alike: `{absolute: <sigma>}`, the sigma in the channel's unit, or `{relative: <fraction>}`, the sigma as a fraction of the reading, stating exactly one of the two. A bare-float seed `noise` and a scenario's `{noise, noise_abs}` pair now stop the build; write `noise: 0.1` as `noise: {absolute: 0.1}` and `{noise: 0.0, noise_abs: 0.1}` as `{absolute: 0.1}`. A relative term is taken of the seed's `nominal`, so a seed without one states `absolute`. The facility file, `seeds.json` and `scenarios.json` carry version 2 of their headers, and a render from an earlier OSPREY is rebuilt with `osprey build`.
- A `nothing_answered`, `connection_error` or `timeout_error` envelope from the controls tools now says under `details.recovery` when it dropped the connector or replaced the connector-host child.
- Under an OIDC claims binding, a person whose groups map to several roles now signs
  in to a card built for any one of them, and the login record lists the mapped
  roles. A card whose roster entry names no role still refuses such a login as
  `ambiguous_role_claim`.
- A view a render does not carry is named as a note on stderr — indented and dim,
  with no warning glyph — rather than as a warning: `view <name> not written:
  <reason>`.
- The `orm` plan's response matrix is now per unit of commanded corrector current (its setpoint), the convention of MML and pySC; a `regressor` parameter chooses the readback instead, and the run records which one it used.
- A deployment whose chat bridges, logbook sync or record archive run on the host network is no longer reported as reachable from other machines on their account, because they open no listening socket. A host-network service that declares neither `listens: false` nor `bind_env:` is still counted as reachable.
- A control-context record too large to read is refused with its own remedy —
  the size it exceeded, and to re-select the control target — rather than with
  the unreadable record's advice to check a mode and group that are correct.
- A process OSPREY starts and stops is now stopped through one call that reaps it and answers gone, the processes still running, or that the process table could not be read; nothing else reaps it while it is stopped.
- `osprey build` now pairs a setpoint with the one readback on the same device whose signal names
  the same quantity, unless a source states `pair`; the Middle Layer import no longer writes pairs.
  The vocabulary names the plane of a coil's current (`current_x_setpoint`, `current_y_readback`
  and their siblings) for a device that holds one coil per plane. A source may state `s` (and
  `model`, `length`) for a device the deck does not place, and the Middle Layer import writes the
  export's `Position`. The simulator view serves each deck under its own suffix, the pyat-specialist loads it by
  that file, and a model
  without a deck gets `direction`, `unit` and `value_range` on its wiring.
- A panel opened in the web terminal — from its sidebar row, a link, the "+"
  menu or the agent's `open_panel` — lands as a new tab in front of the service
  tile used last, so nothing resizes and the covered tab is one click away.
  Before, a row click replaced the tile's panel and an agent open halved a tile.
  **Open in a new tile** in the row's menu is the one open that still splits.
- A panel preset (`panel_presets` in a profile, `web.presets` in config) is the
  dock layout it opens: a tile `{tabs: [<panel id>, ...], showing?, size?}` or a
  split `{split: row|column, items: [...]}`, never naming the Agent pane. The
  "Layouts" click and `arrange_workspace(preset=...)` apply it exactly. The
  `label: [ids]` list form fails the build with the layout to write instead.
- The agent changes the dock through one op path. Every layout tool is relayed
  to the browser that holds the dock and answers with the resulting layout and
  its `rev`, or with a named refusal that changed nothing; `list_panels` reports
  the layout, the panels in pop-out windows and the `rev`. `close_panel` is
  replaced by `close_panel_tab` and `close_panel_tile`; `drop_panel` and
  `open_panel_set` are new; `arrange_workspace` takes a preset or an exact
  layout. `POST /api/panel-close`, `POST /api/panel-arrange` and the agent form
  of `POST /api/panel-focus` are gone; the panel tier gets `/api/panel-op/*`. The
  terminal is never an agent drop source or close target, and a panel in its own
  window is brought forward, never moved or closed.
- `channel_read` and `entries_by_ids` name what they did not answer under `missing`, a map from
  each requested item to the source's own text (`{}` when everything answered); this replaces
  `channels_failed` and the `requested`/`found` counts. A request nothing answered is refused with
  the same map under `details.missing`: as the one error every item failed with, or as the new
  `nothing_answered` error, which the agent's error-handling rule names with the class
  "Unanswered". The contract is documented on the new Partial Answers reference page.
- `deploy.fqdn` and `modules.web_terminals.external_origin` must name a DNS host
  name or IPv4 address, and a value nginx could not use as a server name is
  refused at render.

  nginx now serves a multi-user deployment only on its external origin's host and
  answers any other name (another DNS name, the bare IP, `localhost`) with a
  `301` to the same path on the origin. Before, such a page loaded and every
  action on it was refused.

  With TLS on, the plain port redirects to the external origin, and an `http://`
  `external_origin` is refused while `tls.enabled` is true.
- The agent's bridge to Phoebus is read-only unless the build profile sets `phoebus.agent_access: read_write`. With the default `read`, `phoebus_drive` is left out of the agent's tools and permissions and the phoebus server refuses a drive; `read_write` offers it behind the approval prompt, the login's write tier (`control_system.writes: allowed`) and the machine's switch. Any other value is refused by name.
- The `place-wrapped` warning's fix asks for an `s` within the deck's length, or for the stated `s` to be dropped.
- A device whose stated `s` lies outside its periodic model's deck still builds, placed modulo the deck's length, and `osprey build` and `osprey facility validate` now print one `place-wrapped` warning per such device naming the stated s, the deck length and the point it was placed at.
- `validate_plan` refuses a plan that moves a channel no event of its run
  records, naming the channel and the fix. Such a run completes with readings
  but no record of the setting each was taken at, so its analysis and default
  figure have no x axis.
- Files the previous release wrote are left where they are when you upgrade: the
  audit files under `var/audit/`, dispatch run records, bridge conversation
  history and saved feedback. The agent record does not read them and the
  archive does not copy them, so their contents are not in the record, on the
  history page or in `var/archive/`. `osprey reset` keeps `var/audit/`; only
  `osprey reset --purge-audit` deletes it. Read the old files on the host with
  `grep` or `jq` as before.
- Connector-host children and job workers are put down by one shared helper in `osprey_connectors.process`, and `osprey_connectors.ipc.launch.terminate_host` is now `osprey_connectors.process.terminate`, which returns the reaped exit code.
- The web terminal reads the process table in-process, so a busy host no longer makes it
  report its commands as unknown.
- A provider class registered through `ProviderRegistration` names the variable
  its API key arrives in with `api_key_env_var`. In a process that has loaded
  the application registry, the launch, `.env.users`, `.env.example` and the
  build's key summary read that variable instead of `<NAME>_API_KEY`. The
  built-in providers and config-only `providers.yml` entries keep the variables
  they use today.
- A provider's request cap is now `requests_per_minute` on its `providers.yml` entry,
  and the in-context channel finder and the benchmark ReAct loop pace their model
  calls to it; a provider without the key is not paced. The shipped `cborg` entry
  sets 18. A repository whose `providers.yml` was copied by an earlier `osprey init`
  runs `cborg` unpaced until `osprey init --revalidate` refreshes it.
- The proxy's note in place of an image, on a provider route that takes no
  images, now has the same wording as its note for a document or any other
  content that route leaves out.
- The pyAML adapter's `set_and_wait` and OSPREY's stepped measurements now wait
  for each setpoint's readback to come within the tolerance the facility states
  for it, and fail when it does not.
- The approval prompt of `pyaml_measure` now only describes a call; a call for a kind the model's measurement file does not list goes straight to the tool, which refuses it with one sentence. The build stops when a measurement file names a tune reading of one plane that no reading of the other plane pairs with.
- Every `pyaml_measure` result carries `missing`: `{}` for a completed run; a
  stopped run maps what it did not measure to the first line of what stopped it,
  or `null` when nothing raised. The top-level `remainder` and `error` keys are
  gone.
- The pyAML view build names a magnet whose slices mix zero-length and thick elements in a note of its own.
- Every image that overlays local wheels now asserts that exactly one
  distribution owns the `pyaml` package (`python -m osprey.deployment.pyaml_owner`).
- The pyAT specialist's recipe now says how `get_optics` rows map to elements:
  one row per `refpts` entry, in that order, so a value at a named element is
  read by the element's position within the `refpts` passed, which equals its
  ring index only when every element was requested.
- The pyat-specialist agent loads each served model's deck from `data/simulator/decks/` and reads channel wiring from `data/simulator/variables.json`. A render that serves no model with a deck has no pyat-specialist agent.
- The qmd sidecar image carries three patches on the pinned qmd 2.5.3 release,
  each proposed upstream: a natural-language question now matches on its content
  words instead of requiring every word, and vector search scans an in-memory
  copy of the vectors. Searches on a large index drop from seconds to about a
  quarter of a second; the cost is about 3 KiB of RAM per indexed chunk per
  sidecar, and `QMD_VEC_SCAN=vec0` turns the in-memory scan off.
- Without a measurement queue, the agent no longer refuses a scan: it runs a
  multi-setting measurement as separate `channel_write` calls or as one
  `osprey.runtime` script the operator approves as a whole, whichever fits.
- An enqueue whose plan arguments are not a mapping of names to values is
  refused with `invalid_item` and `HTTP 400`. Nothing is queued: the arguments
  are no longer dropped to make room for the name of whoever queued the plan.
- The web terminal's rail keeps Docs and Feedback and moves the tour and the
  activity log into a new **⋯ More** menu. SYSTEM is a row under Tools, or it can
  be added from **+ Add panel**. It opens without taking a rail entry and can be
  pinned from its tile.
- **Breaking change:** raw client puts from the Python executor and from
  notebook kernels are now refused in readwrite runs too, raising
  `ChannelWriteBlockedError` with reason `RAW_CLIENT_WRITE`. This covers
  `caput`, `PV.put`, caproto, DOOCS and PyTango writes, ophyd-async signals written
  directly in a kernel, and `PV(..., monitor_delta=…)`, which writes `.MDEL`.
  A RunEngine driving ophyd or ophyd-async devices inside a notebook or execute
  script is refused the same way; submit the plan to a Bluesky lane queue
  instead, where plans run unchanged. Replace raw puts with
  `osprey.runtime.write_channel` or `write_channels`, as described in
  `docs/source/architecture/python-executor.rst` and
  `docs/source/how-to/web-terminal/notebooks.rst`. Reads are unchanged.
  PVAccess puts (p4p `Context.put`, pvaPy `Channel.put`) are the exception: the
  connector does not write PVAccess yet, so they keep their approval ask and,
  with limits checking on, their limits check, as before.
- The approval hook now refuses an `execute` call in a readwrite run whose
  code spells a raw client put — `caput(`, `caput_many(`, `epics.caput(`,
  `PV(...).put(`, `aioca.caput(`, `write_door` or `open_door` —
  before any human is asked to approve it. The hook answers with a deny that
  names `osprey.runtime.write_channel`, and files a refused audit record with
  reason `raw_client_write`. Comments and string literals do not count.
  `write_channel`, `write_channels`, a bare `.put(`, the PVAccess puts (p4p
  `ctxt.put(`, pvaPy `Channel.put`) and `execute_file` are asked about as
  before.
- A ``readonly`` Python execution no longer imports Bluesky and ophyd-async
  before the script runs. Their write entry points are refused the moment the
  script imports one of them, so a run that reads and computes starts sooner.
  Control-system client libraries are unchanged: their writes are still refused
  before any line of the script runs.
- The middle-layer channel-finder index derives each family's device list from where its devices are placed, so its place filters agree with the facility's places for every facility. The in_context index lists every channel unless the facility tags a subset `in_context`. A record's `names` are a set, and a new `label` slot gives a device, channel or group one display name. The MML import records each export row's device in its own table, `imported/mml/rows.json`, instead of on the device.
- **Breaking change:** the web terminal's Expert and Simple views are gone. The
  Agent pane is the operator chat for everyone, and the terminal is an opt-in per
  tab from the Agent bar's ⋯ menu. `web.ui_mode` is no longer read: a build warns
  and drops it. `web.expert_view` becomes `web.terminal`, and a build refuses the
  old key.
- A facility build writes each facility view once per distinct set of config
  keys it reads and copies it into every other render, and the approval hook's
  measurement block is computed once per distinct input instead of per render.
- The response check of `osprey facility validate` sweeps every corrector of a model in one call of the pyat engine's `response_matrix`, which returns the columns that solved and names each corrector whose step has no solve instead of raising on the first; a model's lattice is copied once instead of once per corrector. `model_response_matrix` returns the solved columns with `unsolved` naming the others, and is refused with every unsolved text only when no corrector solves.
- The retired roster spellings, which `tier`, `default_tier` and `tiers/`
  replaced, and the retired write-posture keys (`writes_enabled`,
  `control_system.write_tools`, a facility write tool missing from its server's
  `writes:` list) keep building with a warning one release longer: they are
  refused from OSPREY 2026.12.0, no longer from 2026.11.0. A governed tool's
  `approval.tools.<tool>` policy is refused from the same release, as announced.
- `osprey init --revalidate` treats `providers.yml` the way it treats
  `profile.yml`: it keeps every value the file states, an edited entry OSPREY
  ships included, and adds only the entries and keys OSPREY ships that the file
  lacks. `osprey validate` compares each shipped entry key by key as preset
  drift — a changed value is a note, a key or model one side lacks is refused
  unless a deviation comment claims it — in place of the old whole-file note.
  `osprey init --force` still resets the shipped entries.
- The agent's control-system safety rule now follows the wire the connector
  speaks, so a deployment on the served simulator or the live stand-in is told
  EPICS Channel Access; the OPC-UA and LabVIEW sections are gone, and a build
  whose render does not know the wire stops instead of writing lighter rules.
  The test-IOC rule also renders on a live stand-in deployment.
- pyAML measurement steps and `set_and_wait` wait for each setpoint's readback
  through `osprey.runtime.settle()`: within the facility's stated tolerance, else
  on the simulator within the readback's motion band.
- Every file OSPREY rewrites in place, the hooks' files included, is replaced
  whole through a hidden `.<name>.*.tmp` file that is flushed to disk and removed
  on failure. Files whose permissions came from the process umask now get fixed
  ones whatever the umask: 0644, in-flight markers and approval stamps 0640, and
  the agent's `.claude.json` 0600. Profiles add the `safety-file` helper to
  their hook list. A container that sets a control identity mounts the helper
  beside the identity module.
- `osprey build` and `osprey facility validate` stop with `profile-invalid` when `control_system.connector.virtual_accelerator.probe_channel` (or the stand-in's) names a channel the facility file does not hold. The stop names the key, the address and one readback of the facility file to put there.
- Shipped service templates no longer write the `osprey.project.name`, `com.osprey.repo-id`, `osprey.project.root` and `osprey.config.digest` container labels; the build's generated `build/osprey-labels.override.yml` gives them to every service. A facility template need not write them either, and one that still does keeps working because the values agree.
- The plugin skill that sets up a facility deployment is `/osprey:setup` (`$setup`
  in Codex). `/osprey:install` no longer exists; update the plugin and type the
  new name.
- Service images now share one framework install. Every Python service recipe
  is identical through its deps layer, so `osprey up` installs the framework once
  per deploy rather than once per image, and each image adds only what it needs
  (an extra, the queueserver pin, Node) in a layer after it.
- The web terminal's artifact list is now in the sidebar. The **Artifacts** row
  lists the artifacts of the session, grouped by type or by day. It has a title
  filter (always open above eight artifacts) and a ⋯ menu with **All sessions**
  and **Refresh**. The **Script runs** row lists the records of executed
  scripts: the approval review, the notebook and the result record. The
  artifacts pane shows one artifact under one header row: `Artifacts › <title>`,
  ‹ › with `N of M` to go through the list, **Open full size** and **Save**.
  Print, Compose logbook entry and Delete are in the pane's ⋯ menu. The panel
  label is now `Artifacts` (it was `WORKSPACE`). The gallery has no Simple layout
  now: in its own window, it shows the full gallery in both views.
- Web-terminal sidecar panels are described in one place: a registry entry now
  carries the panel's label, sidebar icon and tooltip sentence, health route and
  whether the panel proxy forwards the operator secret to it. The browser builds
  one sidebar row per registered sidecar with no edit to its bundle, and the
  JupyterLab panel keeps its routes and label.
- Web terminal: **Sign out** is now a row of the sidebar footer, above ☰ More,
  instead of a row inside ☰ More. It asks first ("Sign out of OSPREY?"), and so
  do the OSPREY mark and the command palette's **Log out**, which now open the
  same dialog; the OSPREY mark stays a home link where there is nothing to sign
  out of.
- `osprey sim apply` refuses a scenario set two of whose scenarios write one target, judged on the build's simulator view, and names that target before anything is written.
- `osprey sim apply` no longer writes a scenario's physics faults into the deployment's `.env`, and stops with `run 'osprey build' first` on a build without a simulator view. `osprey sim status` reports a virtual-accelerator or stand-in target's model status, read from each model's status channel. The control-assistant preset ships its scenarios only as facility scenario files under `data/facility/scenarios/`; `data/simulation/scenarios/` and the simulation bundle reference page are gone.
- `osprey sim list` and `osprey sim status` read the simulator view the build writes under `build/data/simulator/`, and `osprey sim apply` judges a requested scenario set on that view. `osprey sim status` prints one `<model>: <status>` line per served physics model and one `log: <absolute path>` line per model; `--target <name>` selects the control target to report on, and defaults to the deployment's own.
- A fresh control-assistant deployment now starts in the `rf-thermal` scenario:
  `osprey up` activates the scenarios the new profile key
  `simulation.default_scenarios` names (a list of scenario names; absent or `[]`
  starts in `nominal` alone) when the deployment has never chosen a set, and seeds
  that set's archived history and logbook narrative. The control-assistant preset
  states `[rf-thermal]`; a value that is not a list of names stops the build
  with `profile-invalid`. `osprey sim apply` still means exactly the set it names,
  and a deployment that already chose a set (including `nominal` alone) keeps it.
- `osprey sim list` and `osprey sim status` read the simulator view the build writes under `build/data/simulator/`. `osprey sim status [--target <name>]` prints one `<model>: <status>` line per served physics model (`ok`, or the model's error text), then one `log: <absolute path>` line per model naming the log the simulator appends to, followed by any scenario-overlap records from that log as `<model> (log, instance <i>, pid <p>): scenario-overlap on <target>`. A target served by anything but the mock connector exits 1 with `sim status: <target type> targets do not report model status yet`.
- In the Simple view, each action of an agent turn — a thought, a tool call, a
  subagent — is one row of the same shape: an icon in a left column, a short
  phrase, the detail dimmed, and the result at the right edge. A hairline joins
  the rows, so a turn reads as a timeline. The icon shows the action's state: a
  colour of its own kind while it runs (pink for a thought, blue for a tool call,
  gold for a subagent), with a soft pulse on the current action only, grey once it
  settles, and the error colour when it failed. The reduced-motion setting stops
  the pulse. A settled thought says how long it took, also in a replayed
  conversation. The osprey no longer sits on each reasoning row; it
  stays on the turn's status line and in the header. Consecutive tool calls of
  one kind fold into one row that counts them and opens to the calls, with a
  failure inside shown on the folded row. A call that needs the operator's
  approval, and a control-system write, each stay on a row of their own, so the
  decision stays in view. A refused write's reason shows under its row, or under
  the folded row or subagent row that holds it, without opening anything. The
  line that closes a turn now states its duration, then how many actions and
  approvals the turn had, and leaves out what did not happen.
- In the Simple view, an opened tool call or subagent shows each part of it in
  its own rounded box with a title: Input and Result for a tool, Activity and
  Result for a subagent. The boxes look like the task bar and the approval cards.
  A failed result has a red border, and the "shortened" note is in the Result
  title. An opened thought shows its reasoning in the same kind of box.
- In the Simple view, a drawing that the agent is still writing shows as a box
  that says "Drawing…", and the status line also says "Drawing…". Before, the
  drawing's code showed in the reply until the drawing was complete. The drawing
  replaces the box when it is complete.
- The Simple view's chat is redesigned. Operator turns are soft bubbles, and
  agent replies read as plain text. Thinking, tool calls and the agent's plan
  render as inline blocks in the turn, and a turn shows a live status line with
  its elapsed time. Chat text uses the interface's own font. Code, tool inputs
  and numbers stay monospace. Math in agent replies (`$…$`, `$$…$$`, `\(…\)`,
  `\[…\]`) is typeset with KaTeX. A subagent shows as one row in the turn, with
  a live note of what it is doing. Its own tool calls and messages open under
  it. A turn that hands work to a background agent stays open until the agent
  reports, and Stop ends the background agent too.
- In the Simple view, the status line with the OSPREY mark stays at the end of
  the turn from the moment the agent starts until it finishes. Before, it went
  away while the agent wrote text, and while it wrote a tool call. It now says
  "Writing…" while the reply streams, and "Waiting for your answer" while an
  approval or question card waits. When the agent starts a tool call, the line
  names the call at once, before the call is complete.
- The Simple view's chat text is larger. Messages, replies and the message box
  are now 16px (before, 14px). Action rows and cards are 14.5px (before, 13px),
  and the smaller labels are 1-1.5px larger too. The rest of the web terminal is
  unchanged.
- In the Simple view, a tool call's input and its result show as key and value
  rows, with nested objects indented under their key. Code and other multi-line
  text keep their line breaks, with no JSON escapes. A long result keeps its
  structure. Its long values are shortened inside it, and a shortened result
  carries a quiet note. A reloaded chat shows the same result preview as the
  live chat did.
- The archiver seeder module is now `osprey_connectors.simulation.archive`.
  `archive.build(view, active_set)` builds a simulator view's history at one active scenario set's start state: archiver events, drift, keyed noise and clamp per timestamp, with `bool` and `enum` channels as option indices.
- `osprey_connectors.simulation` no longer re-exports the engine, machine and expression names; import them from `osprey_connectors.simulation.engine`, `.machine` and `.expressions`.
  The package root now imports neither numpy nor lume and holds the tick period (`resolve_tick_s`), `decode_char_waveform` and the model status lookups; `coerce` is imported from `osprey_connectors.simulation.values` and the active-scenario state helpers from `osprey_connectors.simulation.state`.
- The simulator view says each binding's engine role and plane; a view built by an older OSPREY is refused with a rebuild message.
- The build's simulator view, ``variables.json``, is now schema ``osprey.facility.simulator/3``: each
  wiring record carries the ``engine_role``, ``plane`` and ``refresh`` its engine's ``describe()`` states, and
  each channel carries the node it is ``on``. ``osprey_connectors.simulation.view`` is the reader of
  the view; it refuses a render from an older OSPREY and asks for a rebuild with ``osprey build``.
- When the commands an agent started cannot be told apart by name — two runs
  of the same script, or command lines too long to name in a sentence — the
  question asked before ending the agent counts them in its title ("This also
  ends 2 commands the agent started.") and lists each one whole in its body. A
  command line that begins with its own name is listed once rather than twice.
- The built summary card suggests the logbook seed command when the build's scenarios list a logbook narrative.
- `osprey build` and `osprey validate` refuse a `system.timezone` that is not a zone name in the IANA time zone database, spelled exactly, case included, and suggest the closest one. Such a value used to build, reach the agent's rules and every container's `TZ`, and run as UTC. A value that references an environment variable is not judged at build.
- A CSV or TSV file that a dispatched run creates is delivered as the file
  itself instead of as a PNG picture of the table, so it can be opened in a
  spreadsheet. Google Chat posts it as a document link and Nextcloud Talk
  shares it into the room; the Teams bridge, which delivers images only, no
  longer posts it. Logbook attachments still render tables.
- The channel-finder agent's terminology table lists the device classes in the
  build's `data/facility_facts.json`, with each class's aliases as authored and
  its families; a build with no device class prints one line saying so. The table
  no longer reads `facility.ontology`, and its prose says "channel address" where
  it named a protocol.
- **Breaking change:** the write switch has three positions per login and
  machine: `off`, `asked` (every machine write prompts) and `allowed` (it runs
  unprompted). The tier key `control_system.writes: never | asked | allowed` is
  the furthest position a login may pick; the shipped readwrite tier is `asked`,
  admin `allowed`.

  - A stored `on` reads as `asked`. A tier spelling `allowed` now permits
    unprompted writes; spell `asked` where prompts are wanted.
  - The switch, not `approval.tools.<tool>`, decides for the governed write tools
    (`channel_write`, write-mode `execute`/`execute_file`, `pyaml_measure`,
    `queue_add`, `queue_start`, hooked facility write tools); `skip` now prompts
    at `asked` and `approval.enabled: false` no longer silences them. Such a
    policy warns now and is refused from 2026.12.0; `osprey init --revalidate`
    removes it.
  - Upgrade order: install the connectors package, rebuild the lane and dispatch
    containers, then rebuild the project. An old connectors container that meets
    an `asked` login-tier map refuses every login, not just the asked ones; an
    unrebuilt project reads `asked` as `never`.
- A deployment's web terminals are now tiers and apps. A tier is an access level
  (`readonly`, `readwrite`, `admin`) in `tiers/<name>.yml`, which users and roles
  pick with `tier:` and `default_tier:`. An app is a second agent that shares the
  deployment's services and data (logbook, knowledge, channel finder) in
  `apps/<name>.yml`, listed under `modules.web_terminals.shared:` and opened from
  a shared card on the landing page. `osprey init` writes both folders out in
  full, and `osprey scaffold tiers` and `osprey scaffold apps` add them to an
  existing repo. `dispatch.tier` names the tier the event dispatcher's workers
  run. `osprey profile card` lists the `graph` and `event_dispatcher` servers
  when a render ships them.
- The optional Tiled server image moves to `ghcr.io/bluesky/tiled:0.2.18`, so the
  shipped server is no longer older than the `tiled` client OSPREY installs.
- Tooltips in the web terminal now explain what a control does. They show in
  the expanded sidebar as well as the collapsed one, and on header buttons and
  other titled controls. A tip shows a label and a one-line explanation 450 ms
  after the pointer arrives (or at once when it follows another tip), stays
  inside the window, and goes when you press, leave or hit Escape. In the
  expanded sidebar a row shows only its explanation.
- The web terminal tour's spotlight now follows the shape of what it shows: on
  the control-target step it outlines the chip and its open popover as one
  L-shaped cutout, so the sidebar rows under the chip stay dimmed.
- `osprey facility validate` and `osprey facility show` print the settle-tolerance warnings `osprey build` prints: a scenario that moves a readback beyond its setpoint's tolerance, and the count of setpoints with no `tolerance` on a lane that does not serve the simulator. The example facility's cavity tuner setpoints settle within 0.11 cm, which covers the tuner motion of every shipped scenario, so a stock build prints neither.
- The response check of `osprey facility validate` no longer counts sign agreement in a block whose monitors read the plane its correctors do not drive; the VA report says `not counted` there.
- When a project's config file exists but cannot be read, the Web Terminal's
  Config panel, its scaffold gallery writes and ARIEL's settings editor now
  refuse with the same sentence, naming the file and telling the operator to
  fix it and restart the server.
- The virtual accelerator boots one composite over the simulator view `osprey build` renders from the facility definition, and its served channels are that view's `addresses.json`; no channel manifest is read, so a project `.env` carrying `VA_CHANNELS_FILE` changes nothing.
  A different physics engine is registered as an engine plug-in under the `osprey.simulation.engines` entry-point group; the container no longer runs a replacement entrypoint module, and `VA_ENTRYPOINT_MODULE` has no effect.
  The cavity frequency, `SR:RF:CAVITY:01:FREQUENCY:SP` and `SR:RF:CAVITY:01:FREQUENCY:RB`, now serves the value the deck states.
  The twelve ion-pump voltages, `SR:VAC:ION-PUMP:01:VOLTAGE:SP` through `SR:VAC:ION-PUMP:06:VOLTAGE:SP` and their `:RB` readbacks, now serve the seed's nominal instead of zero.
  The virtual accelerator now serves five more addresses: the model's optics outputs `SR:DIAG:TUNE:X`, `SR:DIAG:TUNE:Y`, `SR:DIAG:CHROM:X` and `SR:DIAG:CHROM:Y`, and the physics model's status channel `ca:SIM:SR:STATUS`.
- The virtual accelerator serves the simulator view through the composite and the model runner, on Channel Access and PVAccess.
  It reads the view from `/data/simulator/` (`VA_DATA_DIR` names a different data root) and requires `VA_INSTANCE` (`virtual_accelerator` or `live_standin`); a missing or unknown value refuses the boot, and the model RPC's `status` reply reports the value.
  The image runs `osprey.services.virtual_accelerator.entrypoint` by name; `VA_ENTRYPOINT_MODULE` is not read.
- The virtual accelerator prints its ready line only after its first publishing pass has published every served channel, and exits non-zero without it when that pass fails. The model RPC's `status` reply carries `last_failed_pass`: `null`, or the `error` and `uptime_s` of the latest publishing pass that failed.
- The virtual accelerator's compose healthcheck now fails when its publishing
  passes keep failing. The runner keeps a health record -- `serving`,
  `degraded` or `failed`, with the latest pass and pass counters -- rewrites it
  to `/run/osprey-va/health.json` after every pass, and the healthcheck reads
  that file in place of a TCP connect to the Channel Access port. More than
  three failed passes in a row mark the container unhealthy; a pass that
  succeeds marks it healthy again. The model RPC's `status` reply carries the
  same record as `health`.
- The hello-world deployment's control-target chip reads "Simulator", and its session starts on `va`.
  A control-context record nobody has switched now follows the deployment's baseline when a server claims it, so a deployment that used to be baselined on `live` moves to `va` on its first start; a switched record is kept.
  A subscription on the in-process simulator now fires for a channel an active scenario couples or makes noisy.
  A switch-capable deployment that states no control-system type now starts its `va` child in process.
- **Breaking change:** the `mock` control-system type is retired. The simulator is one
  connector type, `virtual_accelerator`, served either from its container
  (`control_system.connector.virtual_accelerator.serving: served`, the default) or inside
  the process that asks (`serving: in_process`, no network and no containers). A profile,
  `osprey set` or a connector build that states `control_system.type: mock`, or keeps a
  `control_system.connector.mock` block, is refused with the fix: `osprey set
  connector=virtual_accelerator config.control_system.connector.virtual_accelerator.serving=in_process`,
  then `osprey build`.

  - A deployment that states no control-system type is now the simulator in process, with
    baseline target `va`.
  - A deployment serving the simulator in process that also configures a live or stand-in
    block can switch between them.
  - The in-process simulator's writes journal lives under `<state>/inprocess/`; a journal
    left under `<state>/mock/` is ignored.
  - The Bluesky bridge's browse-only remedy is `osprey set
    config.control_system.connector.virtual_accelerator.serving=served`.
- The model RPC's `status` reply carries `instance`, `endpoint`, `last_cycle_ms`, `queue_depth`, `uptime_s` and `last_refused_write`, and its `info` reply carries `variables` alone; neither reports a backend, a lattice source or an update rate.
  `info` names the side of each variable as `served` or `model`, and over a simulator view it lists each physics model's own variables as `<model>/<name>` after the served addresses.
  `reset` over a simulator view writes each drifted model variable back to the value its model held when it was built; no setpoint, held value or session write is touched.
- The virtual accelerator image builds natively for linux/amd64 and linux/arm64:
  on arm64 it compiles the Channel Access server its serving stack needs, so
  Apple Silicon no longer runs the simulator under emulation. The
  `OSPREY_VA_PLATFORM` override and the `platform:` pin are gone.
- A virtual-accelerator write now posts only the channels it changed, plus the
  written setpoints, instead of all of them, so the server's posting step costs
  about a third of what it did and monitor clients see one event per real
  change. Every channel still refreshes on the periodic pass.
- A setpoint that an active scenario marks `stuck` now reads the value last
  written to it, on every read and in `held()`, while its readback keeps showing
  the model where it was. A write to it confirms on the virtual accelerator and
  on the mock connector alike, with no dependence on when the next publishing
  pass runs. Clearing the fault or a reset drops the written value along with
  every other session write.
- The mock connector, the mock archiver and the archiver seed open the simulator view through
  `SimulatorView`, and refuse a view from an older build with a message to rebuild with `osprey build`.
- The pyAT engine's `describe()` also states the `kind` of device a wiring record drives or reads:
  `kick`, `strength`, `rf`, `energy` or `monitor`, and none for an optics output. The simulator view
  stamps only the record's `engine_role`, `plane` and `refresh`.
- Every web-terminal service renders `init: true`, so the processes a headless
  browser in the image leaves behind are reaped.
- `osprey status`, container seeding, `osprey reset` and `osprey feedback`
  address web-terminal containers by the project name (`project_name`, else the
  project directory's name): the status table looks up `<project>-web-<user>`,
  and seeding writes each user's context into that container.
- **Breaking change:** web terminals need a `modules.web_terminals.renders`
  catalog with a `default_tier` in registry mode too, as they already did with
  `image_source: local`; `osprey up`, the render and
  `osprey scaffold web-terminals lint` (code `web_terminals.requires_catalog`)
  refuse a config without one. A terminal now always runs in `/app/<project>`,
  where `<project>` is its tier's or app's project, and one whose image is built
  elsewhere must state that `project` (code `web_terminals.render_missing_project`).
  A registry deployment that has no catalog adds one whose default tier names
  the project its image was built from — for example `default_tier: main` with
  `renders: {main: {project: <that project>}}`. That keeps each user's saved
  Claude Code sessions, which are stored under the terminal's working directory.
- The built facility file states what each model variable is — its engine role, plane and refresh — and
  the simulator view reads it there instead of asking the engine; the wiring stage describes each
  record from the engine load it already does, and stops on a record its engine cannot describe.
- The "working safely" landing notice no longer promises an approval before
  every hardware write. It now says your control system team sets which writes
  ask, which stays true when a deployment turns approval off or skips a tool.
- **Breaking change:** a write with no owner is refused, since no switch covers
  it. Scheduled and triggered dispatch jobs run for no login, so their writes are
  refused (`owner_unscheduled`), with no route back in this release; Bluesky
  items with no owner no longer write either.
- Chat shows a refused write in plain words naming the reason, the machine and
  the switch that would change it, instead of the raw refusal code.
- A rehearsal on a virtual accelerator now writes under its own identity, so its
  records and refusals name the rehearsal rather than the machine it imitates.
- **Breaking change:** on a render whose tier is `allowed`, a
  `claude_code.permissions.remove_deny` that lifts `Bash` or `Monitor` is now a
  build refusal; on a `never` render it stays a warning.
- **Breaking change:** every write surface now decides by the one write verdict.
  A host process refuses a write for an owner other than its login. A live lane
  writes only when its live connector block is configured. A facility server
  guarded whole by `writes_check` refuses its reads too while `live` is off.
- **Breaking change:** whether a login may write is now one key,
  `control_system.writes: allowed | never`, set in a tier file, the deployment
  root or an app (apps always `never`); a tier overrides the root and silence is
  `never`. The old `writes_enabled` keys are rewritten at build with one warning
  each (a literal `true` reads as `allowed`, anything else as `never`); a file
  spelling both is refused. `osprey init --revalidate` rewrites the files. A
  render built before this release reads as `never` until rebuilt.
- **Breaking change:** connector writes now follow one verdict and start off on
  every machine until that machine's switch is turned on. An unknown owner,
  target or record refuses the write. A write tier now arms every machine; the
  shipped `readwrite` and `admin` tiers armed only the virtual accelerator
  before. Each machine stays off until its own switch is on.
- A connector write result that is refused must name a known refusal reason; one
  that does not is refused at construction. A result also cannot carry a reason
  unless it is refused, or a message unless it is refused, failed or unconfirmed.
  The reasons are `osprey_connectors.errors.RefusalReason`, each naming who
  refuses and what to do next.
- A `channel_write` call where no value reached any channel now carries its rows under
  `details.results`. A batch in which any write was sent and not taken is refused as
  `nothing_answered` instead of `internal_error`.
- The write switch's colour now means the position on every machine: off green, asked yellow, allowed red. The machine shows as an outline and icon, and only allowed on the live machine fills the block; the hardware line stays red on the live machine. Every machine's dialog now carries a banner.
- The facility's machine is now called "Live" instead of "Real machine" everywhere an operator or the agent reads it, including the banner `Live — writes move hardware` and the refusal word `the live machine`.
- One "Don't ask again on any machine" now covers every move of the writes switch to asked or allowed, the live machine included, and a second one covers target switches. Per-machine waivers already recorded no longer apply, so each operator opts in afresh; "Ask again" in the popover clears both.
- A built project's `memory-guard` `PreToolUse` matcher is `Write|NotebookEdit`, and the build's write-tool check lists `Bash`, `Edit`, `Write` and `NotebookEdit`: the Claude Code CLI no longer has a `MultiEdit` tool.

### Deprecated

- The roster's old spellings still load until OSPREY 2026.12.0, each with a warning
  that names its successor; `osprey init --revalidate` rewrites a repo to the new
  spellings in place:

  - `users[].persona` and `authorization.roles.<role>.persona`, read as `tier` <!-- retired-spelling -->
  - `modules.web_terminals.default_persona`, read as `default_tier` <!-- retired-spelling -->
  - a `personas/` folder beside `profile.yml`, read as `tiers/` <!-- retired-spelling -->
  - the `modules.web_terminals.personas` catalog, ignored: each tier or app is its own file in `tiers/` or `apps/` <!-- retired-spelling -->
- The retired `writes_enabled` keys still build with a warning naming
  `control_system.writes` and `osprey init --revalidate`; they are refused from
  OSPREY 2026.12.0.
- A facility server with a `writes:` list that leaves out a tool it lists under
  `permissions.ask`, or whose name looks destructive, now gets a build warning
  naming the tool to add; the tool is gated as a write anyway. The warning
  becomes a build refusal from OSPREY 2026.12.0.
- `control_system.write_tools` is deprecated: use `mcp_servers.<server>.writes`.
  Its tools are still gated as writes, now following the real machine's switch,
  and the build warns until the key is refused from OSPREY 2026.12.0.

### Removed

- **Breaking change:** the deploy block's `ci_image_prefix` is gone and a profile that sets it is refused. Set `images.base_registry` under `config:` instead. It keeps Docker Hub's paths, so the pipeline pulls `<mirror>/library/python:3.12-slim` where it pulled `<prefix>python:3.12-slim`. (#2385)
- **Breaking change:** the agent's `artifact_get` and `session_summary` tools are
  gone; `artifact_read` returns what `artifact_get` did, and `artifact_list`
  replaces `session_summary`.
- The Agent row in the sidebar no longer shows a green status dot beside its name.
- Artifact categories are retired: the agent's save tools no longer take a
  category or data type, and `osprey build` refuses `artifact_server.categories`.
  Migrate each category to `artifact_server.extension_types.<key>: {label,
  base_form}`, optionally with a `viewer` module; the sidebar now lists artifacts
  under their agent session, grouped by kind, in place of the Artifacts and
  Script runs rows.
- Artifact pins are removed: the `artifact_pin` tool, `POST /api/artifacts/{id}/pin`, the `?pinned=` filter and the gallery's Pin button and Pinned section. The artifact menu no longer has **Pin**.
- OSPREY no longer deletes channel readings; keep or delete them in the gallery. The `control_system.channel_read_artifact_retention` key is gone.
- `osprey.utils.config_writer` no longer has `update_yaml_file`,
  `get_control_system_type` or `set_control_system_type`, which nothing in OSPREY
  called. Switch the control system with `osprey set connector=…` and
  `osprey build`, and edit config fields with `config_update_fields`.
- `aiofiles` is no longer a dependency of `osprey-framework`, and the unused
  `serialize_results_to_file_async` helper is gone; results are written by
  `serialize_results_to_file`, as they already were. Install `aiofiles` yourself
  if your own code imports it.
- `env.file` is removed: it was never read. A profile that still sets it is
  refused as an unknown `env` key.
- The session tree no longer has an **Earlier** row: every row it shows is an agent session's, and `session-list`/`session-summary` drop `earlier`. Artifacts no session owns stay in the command palette and the Artifacts panel.
- `osprey.registry` no longer exports `get_framework_defaults()` or
  `generate_explicit_registry_code()`; nothing in OSPREY called them. Use
  `extend_framework_registry()` to build on the framework registry.
- Provider adapters no longer declare `requires_model_id`, and a provider class
  is no longer refused for leaving it out. Nothing read the flag: every chat
  completion is refused without a model id, whatever the provider. A custom
  provider that still sets it only carries an unused class attribute.
- `osprey eject` is removed. It copied the Channel finder package into a deployment, where the copy drifted from the installed one; the `channel_finder` service registration that fed it goes with it.
- The gallery no longer seeds a shipped example artifact, and `artifact_server.example_artifact` is removed. A profile that still states it builds with a warning naming the line, and the key is dropped.
- **Breaking change:** the LATTICE panel is retired: the lattice dashboard
  server, its `lattice_*` workspace tools and its port family (offset 400,
  `OSPREY_LATTICE_DASHBOARD_PORT`) are gone. A profile that still lists `lattice`
  in `web_panels:`, names it as `default_panel`, or states `web.panels.lattice` or
  a `lattice_dashboard` section builds without it and logs one warning per line.
  Optics questions go to the pyat-specialist agent.
- Panel pages have no standalone form: a direct visit to a panel's address opens the web terminal on that panel, ARIEL hash links included. The `?popout=true` page and the display menu are gone.
- `osprey ariel web`, `osprey artifacts web` and `osprey channel-finder web` are retired: each exits with a pointer to `osprey web`, where the panels run.
- The procedural demo lattice package `osprey.simulation.lattice`, the channel-database generator `osprey.services.channel_finder.tools.generate_from_spec` and its naming vocabulary, and the control-assistant preset's `data/lattice/` and `data/simulation/` files (`machine.json`, `va_bindings.json`, `lattice.json`) are removed; the example facility's committed `decks/SR.json` is the demo model. The virtual-accelerator channel manifest builds from a data tree that carries no `machine.json`.
- The simulator's model-file and noise keys are removed: `control_system.connector.mock.simulation_file`, `control_system.connector.virtual_accelerator.simulation_file`, `control_system.connector.mock.noise_level` and `control_system.connector.virtual_accelerator.noise_level`, together with the Virtual Accelerator compose file's `VA_NOISE_LEVEL` variable. Every connector serves the simulator view the build writes from the facility file. The build no longer reserves `data/simulation/channel_manifest.json` or `data/simulation/channel_limits.json` as profile paths; a hand-written limits file there is still refused.
- The machine-file simulation engine is removed: `osprey_connectors.simulation.engine`, `.machine`, `.procedural` and `.expressions`, and the `osprey.simulation.engine`, `.machine`, `.procedural`, `.expressions`, `.series` and `.archiver_seed` import paths. `osprey.simulation` exports nothing; the series primitives import from `osprey_connectors.simulation.series`, the scenario-state directory from `osprey_connectors.workspace.resolve_simulation_state_dir`, and `DEFAULT_SCENARIO` from `osprey_connectors.simulation.state`. The channel schema and facility spec modules (`osprey.simulation.channel_schema`, `osprey.simulation.facility_spec`) and the channel taxonomy (`osprey_connectors.channel_taxonomy`, `osprey.connectors.channel_taxonomy`) are removed with it.
- The old virtual-accelerator model is removed: the `osprey.services.virtual_accelerator.model`, `.lattice`, `.bindings` and `.manifest` import paths, and the template channel database (`osprey.services.channel_finder.databases.template`, `TemplateChannelDatabase`). The pyat engine's `osprey.simulation.engines.pyat_model.PyATLatticeModel` and `osprey.simulation.engines.pyat_variables` are the virtual accelerator's model. The control-assistant preset no longer ships `data/channel_databases/tiers/`, `data/channel_databases/TEMPLATE_EXAMPLE.json` or `data/machine_state_channels.json`; the channel-finder indexes are views the build writes from `data/facility/`.
- `osprey knowledge compile-ontology`, the ontology compiler (`osprey.services.facility_knowledge.ontology_compiler`) and the `knowledge` install extra are removed. The graph view types devices from the facility file's classes, so no authored ontology schema or compiled class table is needed.
- The TTL generator package (`osprey.services.facility_knowledge.ttl_generator`) and its demo ontology tables are removed. The graph view `osprey build` writes from the facility file is the only Turtle writer, and the knowledge seeder reads it.
- The `control-assistant-va-readwrite` preset. Its posture is what
  `control-assistant-readwrite` now ships, so a roster that names the removed
  preset must name `control-assistant-readwrite` instead. (#892)
- The hub's footer activity strip went with the status bar. The agent's activity stays in the Activity log (☰ More) and on the sidebar rows' badges.
- **Breaking change:** The web terminal's customizable header and status bars are
  removed, together with Customize, the `GET`/`PUT`/`DELETE /api/bar-items`
  route, the per-operator saved bar layouts under `bar_items/` in the agent-data
  directory, and the `web.bar_items` configuration key. The sidebar takes over
  the bars' controls; the Bluesky queue chip and the stopwatch are dropped, and a
  saved bar layout is not migrated. The command palette no longer offers
  `Customize bars` or `Move panel rail to top/left`.
- `osprey.services.channel_finder.benchmarks.generator` (`expand_hierarchy`,
  `collect_middle_layer_pvs`) is removed; the benchmark reads the count the index states.
- The `bluesky.devices_file` build-profile key and the `devices_file` key on
  every Bluesky lane block. The queue server's device file is the build's view of
  the facility file, staged unchanged, so a profile that names
  `bluesky.devices_file` is refused.
- The control-assistant preset no longer ships `data/channel_databases/examples/`: the hierarchical pipeline reads the channel-finder view the build writes, and the channel-finder contract page states its format.
- `osprey channel-finder build-database` and `osprey channel-finder generate`, with
  no replacement: the build writes the channel-finder indexes from the facility
  description. The three `channel_finder.channel_name_generation.llm_model.*` keys
  (`provider`, `model_id`, `max_tokens`) went with the CSV builder that read them.
- The Google Sheets channel database and the `sheets` extra are removed. The
  control-assistant template no longer ships `data/raw/`.
- The Channel Finder no longer edits channel databases. The web API's write
  routes (adding, editing and deleting channels, tree nodes, families and
  expansions), its two impact-preview routes and `GET /api/tree/expansion` are
  gone, and a request to one answers 404 or 405. The flat, template,
  hierarchical and middle-layer databases lose their write methods with them.
  Browsing, search, validation and the pipeline switch answer as before.
- **Breaking change:** the old `osprey.*` import paths for modules that moved to
  `osprey-connectors` are gone. Import `osprey.connectors.*`, `osprey.errors`,
  and `osprey.utils.{config,dotenv,identity,log_filter,logger,relative_time,workspace}`
  from `osprey_connectors` instead; the import map in
  `packages/osprey-connectors/README.md` lists each replacement.
  The `resolve_workspace_root`, `BaseAdapter`, `LegacyChannelDatabase` and named
  `ensure_*_server` aliases are also removed; use `resolve_agent_data_root`,
  `FacilityAdapter`, `FlatChannelDatabase` and `ensure_web_server("<key>")`.
  Re-run `osprey build` after upgrading: a project rendered before this release
  has a limits hook that imports the removed `osprey.connectors` path, so it no
  longer refuses an out-of-limits write before the tool call (the connector still
  refuses it at write time).
- Rendered projects no longer carry `data/demo_machine.ttl`; the graph store seeds from `data/graph/facility.ttl`, the graph view the build writes.
- `osprey up` no longer checks `.env` for the virtual accelerator's build-derived
  channels-file and lattice pointers, and `osprey reset` no longer strips or
  counts a build-derived `.env` section: reset strips only the minted blocks
  OSPREY wrote, and a line under no OSPREY banner is left alone.
- The build no longer refuses a Bluesky lane that arms writes with limits
  checking off when it derives that lane's plan devices from the facility
  roster. The connector's per-write limits check remains the enforcement point.
- The dispatch dashboard's Clear history button. Run history is kept in the agent
  record and leaves it after `agent_record.retention_days`; the dashboard links
  each run to the history page. The worker's `RETENTION_DAYS` now ages out
  artifacts only.
- The `facility.prefix` config key is removed, with the lint that reported it
  empty (`web_terminals.empty_facility_prefix`). Container names and tier and
  app projects come from the project name, and the knowledge graph's identifiers
  from the facility file's identity code; a profile that still sets the key
  loads unchanged.
- The control-assistant preset no longer ships `data/mml/`: the MATLAB exporter comes from `osprey facility import mml --print-exporter`, whose help text carries what the directory's README said.
- `web.feedback.max_store_bytes`. Feedback sent from the web terminal is an
  agent-record entry, kept for `agent_record.retention_days`, and a profile that
  still sets the key stops the build with a message saying to remove it.
- `services.graphdb.index_path` is removed: the search index always sits at
  `data/channel_databases/graph.duckdb` under the render. The roster's graph
  reader and the index's `channels` table are deleted with it; the channel
  roster reads the facility file only.
- `osprey knowledge build-ttl`, `osprey knowledge seed-graph` and `osprey knowledge build-index` are removed. `osprey build` writes the graph view `data/graph/facility.ttl` and the channel search index from the facility file, and `osprey up` seeds the graph store from that view: after a change, run `osprey build && osprey up`.
- The `mml_converter` channel-finder script is gone; build middle-layer channel databases with `osprey facility import mml` and `osprey build` instead.
- The `osprey mml` command group (`import`, `map`, `emit`, `verify`) and the `osprey.services.mml` package are removed. `osprey facility import mml` reads a MATLAB Middle Layer export into `data/facility/imported/mml/`, and `osprey build` writes the channel database, the graph and the simulator view from the facility description.
- The `openobserve` service and its `services.openobserve` settings (`path`,
  `port`, `retention_days`, `image`). The agent record takes its place: what
  OpenObserve stored for the agent's telemetry is now kept as agent-record
  entries for `agent_record.retention_days`. Remove `openobserve` from
  `deployed_services`; the build refuses it as an unknown service.
- The Phoebus tools no longer read `PHOEBUS_BRIDGE_PORT`. The bridge URL comes
  from `PHOEBUS_BRIDGE_URL` when set, otherwise from `phoebus.host` and
  `phoebus.port`.
- `osprey profile expand` is removed. `osprey init --revalidate` now also fills in the config keys a profile leaves to its preset and refreshes the provider entries OSPREY ships; `--preset NAME` picks the preset.
- The profile field `tier` is gone. A profile that spells it stops with
  `facility: profile-invalid: path tier`, and the line names the `in_context` tag
  on a channel as what selects the in_context subset. `osprey set` no longer
  accepts `tier=`, and an emitted `profile.yml` no longer carries a commented
  `tier:` block. The build picks the benchmark query set from
  `channel_finder_mode`: `in_context` takes `in_context_queries.json`, every
  other mode `tree_queries.json`, copied to `data/benchmarks/queries.json`. A render carries
  no `data/channel_databases/tiers/`, `data/benchmarks/cross_paradigm/` or
  `data/raw/`, and no `data/channel_databases/<paradigm>.json` flattened from a
  tier; each channel-finder index is the view the build writes under
  `data/channel_finder/`. The build's profile line reads
  `profile <name> (bundle <bundle>)`.
- A measurement file no longer takes `singular_values`: no measurement tool reads
  it. The mml import no longer seeds it.

  A measurement file no longer names `instruments.chromaticity`: pyAML measures
  chromaticity from the tunes and an RF step, so `crm` uses `instruments.tune` and
  `instruments.rf`.

  The mml import no longer zeroes `PolynomA` and `PolynomB` on the elements its
  corrector families own: an imported deck's correctors are served as the deck
  states them.
- The Channel Finder web explorer no longer edits channels; it is read-only.
  Corrections to a channel go into `data/facility/fixes.yaml` and take effect
  with `osprey build`. An editor for the fixes file is a planned follow-up
  feature.
- A MATLAB Middle Layer export reaches a deployment through one path: `osprey facility import mml` reads the export and the reviewed `data/facility/imported/mml/mapping.yaml` into the facility description, `osprey facility validate` checks it, including each kept response export, and `osprey build` writes every channel-finder index, the graph view and the simulator view from it. The verbs that imported, mapped, emitted and verified an export as their own command group are gone, and so are the files they emitted beside the facility description.
- The old simulation is removed: the machine-file engine and its import paths, the procedural demo lattice and channel-database generator, the control-assistant preset's `data/lattice/` and `data/simulation/` model files, the `scripts/va/` launcher and model demo, and the simulator's model-file and noise keys with the `VA_NOISE_LEVEL` compose variable. Every connector serves the simulator view `osprey build` writes from the facility file, and the example facility's `decks/SR.json` is the demo model. No served nominal value is re-baselined: the mock nominal golden drops its per-channel noise widths and the frozen goldens' headers drop the deleted generator's command, with every pinned value unchanged.
- **Breaking change:** A labelled sidebar replaces the web terminal's panel rail
  and header in both views: **Workspace** (Agent, Artifacts, Script runs,
  Events), **Tools** and a footer with Docs, Feedback and **☰ More**, whose
  labels ⌘\ (Ctrl+\ elsewhere) hides. The rail's top position goes with it:

  - `web.rail_position` is no longer read, and the retro theme uses the sidebar.
  - `GET /api/panels` no longer returns `rail_position`,
    `rail_position_configured` or `family_rail_defaults`.
  - The onboarding tour walks the sidebar, and invites every user once more.
- The `sim-scenarios` skill is no longer shipped into deployments. Simulation
  scenarios are the fixtures a deployment's agent is exercised against, not
  something it lists, switches or writes: applying one purges and reseeds the
  logbook and rewrites the stored archive. A person switches scenarios with
  `osprey sim apply`, and the simulation bundle reference holds the authoring
  rules the skill used to carry. No shipped preset listed the skill; a profile
  that added it to `skills:` drops the line.
- The simulation engine no longer reads a single-name `active_scenario` state
  file. The active set is read from `active_scenarios` only, which
  `osprey sim apply` has always written.
- The `claude_code.telemetry` block: `enabled`, `backend`, `protocol`,
  `openobserve` (`user`, `password`, `org`), `log_user_prompts`,
  `log_assistant_responses`, `log_tool_details` and `log_raw_api_bodies`. Claude
  Code's OpenTelemetry logs now always go to the agent record, and no key selects
  another destination or capture level. A profile that still sets the block stops
  the build with a message saying to remove it.
- The `ariel-standalone` and `channel-finder-standalone` presets, and the
  per-card `control-assistant-<name>` presets: the logbook and the channel finder
  run as apps inside a deployment, and the bundled tiers and apps replace the
  per-card presets. The `ariel.demo_narrative` key goes with them; `osprey up`
  still seeds an empty logbook with the active scenarios' entries.
- The virtual accelerator's old co-hosted runner is removed, together with its serving database, write path, null model and `ioc` package. The virtual accelerator serves through `ModelRunner`.
- **Breaking change:** `osprey.mcp_server.control_system.target_eligibility` no
  longer has `effective_writes_for_target` or `narrowing_refusal`. Use
  `target_write_verdict(section, target).allowed`.
- **Breaking change:** `osprey.services.python_executor` no longer exports
  `ExecutionControlConfig`, and `get_execution_control_config` is gone with it.
  The executor asks the connector's write verdict instead.

### Fixed

- A terminal that reconnects to a session now gets a fresh agent process when the server's launch environment for it has changed, instead of the old one. (#1484)
- Simulator: a physics model that fails to build or solve is now visible outside its
  log. `osprey sim status` prints `FAILED:` with the model's error and exits 1, and
  `osprey health` has a `simulator` category that reads every model's status channel
  on the `va` and `standin` targets and reports a failed model as an `error` row,
  while the container keeps serving the other models. A model that names no deck and
  wires no channel is no longer reported as a `TypeError`: it is not built, by
  design, and reports `no deck`. (#2235)
- A roster `control_identity` may contain a dot after the first character, so `first.last` accounts can be named. `osprey.*` names are reserved like `osprey-*`. (#2373)
- In registry mode, a `modules.web_terminals.image_tag` whose `${VAR}` expands to empty is now refused by lint, `osprey build` and `osprey up`, naming the variable. Before, the render wrote a tagless image ref that compose rejected with a YAML error. (#2377)
- On a host with `images.base_registry` set, the scaffolded pipeline's docker jobs keep their daemon, and an empty `agent_record.clickhouse.image` names the mirrored default. (#2385)
- The web terminal pane no longer stays empty when xterm.js fails to load. It says the assets did not load from the CDN and that a host without internet needs images built with `offline: true`. On an offline host it says they did not load from this server. (#2386)
- `offline: true` now vendors the bluesky-web panel assets into its image; the plans panel no longer requests files the image never fetched. (#2388)
- ARIEL hybrid search under `strategy: rrf` no longer loses its answer, or hides the loss, when one text lane is down: a semantic lane that fails for any reason (unreachable embedding provider, model error, timeout) returns the keyword ranking with a warning naming the cause, a qmd sidecar that is down or fails the query returns the semantic ranking with a warning naming the sidecar, and only both lanes down is an error, naming both. Semantic search itself now reports a failed query embedding as a search error naming the provider and cause, instead of an empty result that read as "no matches". (#2453)
- A channel that does not answer now reports whether the control target itself
  answers. The `channel_unreachable` error measures the active target's gateway
  when the read fails and states the result, instead of leaving the agent to
  infer it from further reads. (#2483)
- Opening an artifact the sidebar has not loaded yet (a subagent run's "Open
  code") now shows that artifact, not the session's first one.
  When two artifacts are opened in quick succession, the later one stays shown. (#2497)
- The sidebar's **+ Add panel** row is the last row of Tools, not of the last group, so it no longer sits under System.
- The agent can read its own artifacts and execution data without an approval
  card again: the Read rules over the agent-data root now name its `artifacts/`
  and `data/` directories by absolute path. The rest of the root still asks.
  The Edit rules over its `artifacts/` and `notebooks/` trees are named the same
  way, so its notebook edits there no longer ask either.
- Clicking the current session in the sidebar reopens a closed Agent pane and focuses its input, and the pane tile's × now offers "Agent pane closed · Undo" like the Delete key does.
- A host session whose agent-record port is held by another project's record now
  says so at once (it refused the host login), instead of waiting out the bound
  and, on a loaded host, reporting a `ReadTimeout`. A later probe that times out
  no longer replaces the answer an earlier probe had.
- A click on the **Agent** row with the Agent pane open folds and unfolds the session tree again.
- An application panel on a deployment with per-user web terminals switched off accepts the deployment's terminal secret, instead of refusing every request.
- The scaffold's and orbit bump's `build_connector` drop the configured
  gateways inside a container only for the deployment's baseline target, whose
  name server the container is given; any other target is reached through its
  own gateways instead of silently reading the baseline's machine. A second
  Channel Access target in one backend process is refused, since the process's
  one Channel Access context would move the first target's reads and writes.
- A login can name the panel of an application it shows as its
  `default_panel` (and in `panel_presets`), so a card can open on the
  application's panel; the panel of an application the login hides is still
  refused.
- A shared login's card that shows an application's panel is now in the
  application container's roster grant, so its terminal's proxied panel requests
  and agent tool calls are no longer refused `401` by the application's gate.
  The grant names the card's requests by the card's name.
- The refusal of a `panel_presets` member now says, as `default_panel`'s
  does, that the panel of an application the login shows is accepted. A
  roster of shared cards with no people is a roster for a panel with
  `single_user_alias`: each card is granted under its own secret, where the
  grant went to the build account and the card's requests were refused.
- `osprey app new` refuses a name whose package would be a keyword or shadow a module,
  and an unreadable profile before `--force` removes anything; `--force` replaces a file or symlink;
  `osprey app check` and `osprey build` refuse an application whose tool module's package is so named.
- A watchdog's hold now refuses an application panel's write at the terminal,
  the agent's and the operator's alike, including an agent write sent after its
  tool's own check passed.
- An application's tool server now reads the rendered config, as framework servers do,
  instead of refusing every tool when started outside the project directory.
- The check for a requirement only an unlisted `apps/<dir>/` declares now reads
  `environment.packages` and each built package's optional dependencies and
  build requirements too, not only `dependencies`.
- An application container that drives the target now reads the machine each
  login's session is on when the deployment's root tier is `never`. The
  control-context tree is bound read-only to every such container instead of
  only on an armed render, so `owner_target` no longer answers the baseline for
  every login; writes are still refused on a `never` render.
- An application tool's write is now refused while a watchdog holds the session, with the flag's source in the refusal, as every other write tool already is. An MCP server's `env:` can no longer set `OSPREY_SESSION_ID` or `OSPREY_SESSION_HOLD_DIR`; a pinned value is dropped with a warning.
- The agent's `entry_open` reports whether the Logbook panel actually showed the
  entry: an open the web terminal did not confirm answers `shown: false`,
  `status: "not_shown"` and a `reason`, with the link for the operator, instead
  of claiming the entry is on screen. Its messages now say "Logbook panel".
- ARIEL's embedding migrations now check the pgvector version, not only its presence: on pgvector older than 0.5, which has no HNSW index, they skip with a message naming the version found and the 0.5.0 floor (and `ALTER EXTENSION vector UPDATE` when the extension is installed), instead of failing with a raw `CREATE INDEX ... USING hnsw` error.
- Hybrid search results in the Logbook preview the entry's own text instead of
  qmd's excerpt markup (`1: @@ -1,3 @@ …`, the `# Entry` heading and the
  "Logged by" line). The agent's search results carry the same clean highlight.
- ARIEL: ``entry_url`` is left off entries that exist only in ARIEL, a corrupt
  draft answers 422 naming the problem rather than a server error, and compose
  returns focus to whatever opened it on close.
- The Logbook rail's facet counts update when an entry is submitted or the
  agent creates one, instead of waiting for the next filter change.
- Closing the ARIEL search service also stops its search threads, and the ARIEL MCP server closes its service on exit.
- An entry submitted from the Logbook's compose sheet appears in the list right
  away, selected and flashed, with the count updated, the same way an entry the
  agent creates does; while the list is scrolled it counts on the "n new" pill.
- Submitting a Logbook entry on a deployment with no `ariel.ingestion.adapter`
  saves it locally, as the compose sheet says, instead of failing with HTTP 500.
- An ARIEL entry with no page in the facility logbook no longer carries a facility link (`entry_url`) that leads nowhere. This covers a Logbook web entry saved before October 2026, a simulation seed entry, and an entry read from a Generic JSON file. Every ARIEL write now records whether the entry exists only in ARIEL; a write that leaves this out keeps the status already stored; and a one-time migration stamps the entries stored before this change.
- An Artifacts tile opened in a new window keeps the artifact it showed instead of
  opening on "No artifact selected", and a later pick in the sidebar reaches the
  window too.
- Every bridge refusal carries `retryable`, and a refusal code the bridge does not declare fails where it is made.
- An `artifact_delete_all` in a process that never registered the artifact-activity listeners no longer parks its summary frame for a later registration to deliver to the Web Terminal as fresh activity.
- A button with no font rule of its own now draws in the page's typeface and size instead of the browser's system font at 13.33px.
- A failed chat turn on the buffered `/api/chat` answer now reports the error's own text, or its type when the text is empty, instead of the placeholder "Unknown error"; with neither, the answer says only that it failed.
- On a deployment whose write tier is `never`, a chat tool call that is not a
  write (Read, Glob, a read-only MCP tool) now raises the approval card instead
  of being refused with "This deployment does not arm writes".
- Web terminal: the chat's starter suggestions go away once used. Clicking one
  removes the row (the sentence above it stays until the first message is sent),
  and a reloaded session that already has messages no longer shows the
  first-contact block above its conversation.
- The chat client's wire no longer carries the agent's conversation id: `system` frames are reduced, like `result` and `context`, to the fields the chat draws, declared once per frame type and checked against what `chat-system.js` reads.
- Deployment: the generated labels override quotes a value that YAML 1.2 reads
  as a number, such as the repo-id digest `096780950e46`, so compose labels the
  containers with the string and `osprey reset` still finds them.
- A profile whose `config:` block pairs a scalar with a dotted key under it (say
  `control_system: epics` beside `control_system.type: mock`) no longer refuses
  readers that ask about an unrelated section such as `modules.web_terminals`;
  the conflict is still reported to the reader whose section it decides.
- `CONTRIBUTING.md` now runs its test steps through `uv run` and
  `uv run python scripts/check.py quick`, instead of a bare `pytest tests/` that
  also collects the CANOPY tests and so fails on a fresh checkout.
- Web interfaces: the design system's scripts and stylesheets are no longer kept
  in the browser cache across deploys, so a page no longer stops loading on a
  stale copy after an update.
- The web terminal now stores a dock change that you make just before a reload
  or before you close the tab. Before, the web terminal lost a change made in
  the last 150 ms before the page went away, and the reload showed the older
  layout.
- A dock layout report from the web terminal that does not reach the server is
  now sent again. A report with no answer, a network error or a server error
  goes out again after a wait that doubles up to 30 s. A report that the server
  refuses goes to the browser console with the reason, and goes out again only
  after the layout changes. Before, a lost report went out again only at the
  next layout change, and until then the agent read an old layout.
- When the browser applies an agent's dock change, the agent now always gets
  the layout on screen as the answer. Before, a failure in a later step, such
  as showing the panel or opening its URL, stopped the answer. After 5 s, the
  agent got the refusal that no browser showed the workspace, but the browser
  made the change.
- When the browser does not answer an agent's dock change within five seconds,
  the agent now gets `no_dock_answer`. The browser can still make the change,
  so the agent reads the panels again before it tries again. Before, the agent
  got `no_dock_client`, which says that no browser shows the workspace and that
  nothing changed. The web terminal now records a late answer from the browser
  as the layout on screen. Before, it refused that answer.
- A reload right after a dock change now keeps that change. The dock saves its
  layout a moment after each change, and a reload or a closed tab inside that
  moment brought the previous arrangement back, for instance equal tile widths
  in place of the sizes just set. A pending save is now written when the page
  goes away.
- The web terminal now brings back the tile sizes of a stored dock layout on a
  reload. Before, a reload could apply the stored layout before the dock had
  measured the page, so the tiles came back at other sizes, often all the same
  width, and those sizes replaced the stored ones.
- The audit-trail contract page now lists the `canopy` surface, on which every
  CANOPY panel write and refused panel write is recorded, and says that every
  record of a panel write carries the value sent in `detail`
  (`value_len=<n> value=<text>`, at most 256 characters of it); the page used to
  say that only `executor` records a payload. It also documents the record fields
  `on_behalf_of` and `role`, and the executor-only `source` and
  `source_truncated`, which the field table left out, and the `mcp` surface a
  server records on when it was never told its name.
- The build-profiles how-to now matches today's `osprey init` and `osprey build`:
  its troubleshooting quotes the messages the build prints, "What the build does"
  drops `--set`, the preset and `git init`, and the init tree, hook events and
  build options are complete and checked by a new docs test.
- The CLI reference now documents every option the commands accept. Added:
  `osprey init --reset` (which destroys the previous deployment's containers,
  data volumes and images without asking), `osprey build --dev`,
  `osprey up`/`osprey restart --reuse-stores`, `osprey web --skip-preflight`,
  the `osprey channel-finder preview` display options, `osprey ariel enhance
  --refresh-stale` and `--entry`, the retired `osprey scaffold web-terminals
  --config`, and the short forms of several options. The overview now lists `osprey query` and
  `osprey sim`, the web terminal's default port is given relative to
  `port_base` instead of as a fixed number, and `init --force` names
  the files it really replaces and keeps. A new docs test fails when a visible
  option, or a top-level command in the overview, goes undocumented.
- The ARIEL contract page's `/api/capabilities` example is the whole response
  the service returns: it now shows each mode's own parameters (the keyword
  mode's `include_highlights`, the semantic mode's `similarity_threshold`), every
  shared parameter with its labels, and the `default_mode`, `vocabulary` and
  `attachments` blocks. The search response example carries every field the
  API sends, including the required `created_at` and `updated_at`, and names one
  search mode, as a search reports. The entry-field error envelopes show the
  detail text the service writes.
- The contract reference pages' examples now match the code, and a test loads
  each one with the code it describes. The ARIEL service-factory example no
  longer wraps the shared service in `async with`, which closed it for every
  later caller in the process; it calls `close_ariel_service()` at shutdown
  instead. The health report example's counts now agree with its rows, and the
  page's `jq` recipes run over it. The hierarchical Channel Finder example has a
  tree that `osprey channel-finder validate` accepts. The Python executor
  parameter tables list the hook-filled `approved_target` and
  `approved_view_sha256`.
- Contributor docs: the "Verify Installation" step now runs
  `uv run python scripts/check.py quick`, the merge gate's quick tier, instead of a
  bare `pytest` that also collects the CANOPY tests and so fails on a fresh checkout;
  the pre-commit step runs through `uv run`; the CI page describes the daily
  revalidation of `main` (`main-full-ci.yml`) and the `last-green` helper, and
  counts ten plugin skills. The MCP server page now
  lists every tool each in-tree server can register, including `control_target`,
  `control_target_set`, `provenance_locator`, the pyAML tools and the `bluesky` and
  `health` servers, and a docs test holds it to the code. The Python executor's
  config snippets use the dotted `config:` form of `profile.yml`.
- The installation pages name `osprey up` instead of the retired `osprey deploy`, and `osprey init`
  instead of `osprey build` for creating a project. The Claude Code pin example is a dotted
  `claude_code.cli_version` line, and an MCP connection failure now points to `osprey chat` or
  `osprey web`. The provider guide no longer names an `osprey registry` command.
- Docs: the chat-bridge pages now describe `GITLAB_URL`, `GITLAB_PROJECT` and
  `GITLAB_ISSUES_TOKEN` as what they are, an optional alert host: with the URL and
  token set, the bridge holds queued questions while the project has an open
  `osprey-alert` issue or GitLab cannot be reached. The pages no longer say the
  bridge files an issue. The Nextcloud Talk page now lists the three variables,
  which the bridge reads like the other two bridges. The Microsoft Teams page
  says `TEAMS_CLOUD=commercial` is the explicit form of the default, and shows the
  drive-id lookup and its two `.env` lines as separate blocks. The event-dispatch
  trigger examples and the three bridge profile fragments are now files the pages
  include and tests load, and the pages' worker denylist and environment
  variables are checked against the code.
- The facility-graph how-to now states the view sizes a reader sees after
  `osprey build` on the `demo` preset: 4731 channels (1016 devices) in the
  graph view; 4731 in the hierarchical and graph channel finders;
  4725 in the middle-layer channel finder; 571 in the in-context index; and every
  channel plus one status channel per physics model in the virtual accelerator.
  The page used to give the pre-expansion 2908 / 569 / 1036 figures and called
  the simulator a subset. The figures are pinned in the demo census, and new
  tests hold the page's n10s initialisation Cypher to the graph seeder, the
  `graph` server's tool list on both facility-graph pages to the registered
  tools, and the facility-rules page's rule and scaffold names to the artifact
  catalog. The external-store setup is now a numbered procedure.
- The "Describe your facility" page's examples now work on the demo
  preset it describes. The fixes example and the seed examples named channels
  the preset does not have and stopped the build; they now use preset channels.
  The channel-list CSV now gives its device a position, so the import writes the
  device the channels name instead of leaving a `reference-missing` build. The
  page no longer says an MML import never rewrites an existing `classes.yaml`
  (it appends the classes it lacks), and it lists every top-level mapping block
  and family key, including `branches`. The limits and measurement examples are
  the preset's own files, included rather than copied. The MML import page marks
  its two YAML blocks as fragments of `mapping.yaml` and documents `branches` and
  `models.<name>.tune`. A docs test pastes each example into a copy of the preset
  and runs the facility validator on it.
- The getting-started tutorials now match what the presets ship. The
  demo tutorial names its real default channel-finder mode,
  `graph`, and describes that pipeline's tools instead of the hierarchical ones;
  its first read uses a channel the example facility serves
  (`SR:DIAG:DCCT:01:CURRENT:RB`) rather than `SR:BEAM:CURRENT`, which it does
  not. The hello-world tutorial runs its DuckDB query with
  `build/.venv/bin/python`, where DuckDB is installed; shows the `.env` line
  `osprey init` prints; counts 15 commented blocks and 11 hooks; and shows
  the shipped `limits.yaml` whole. The guided-setup page loses its stray
  `# skip-ci` lines and no longer tells you to `cd` into a directory you are
  already in. Tests now check the tutorials' addresses, profile excerpts, counts
  and init summary against the presets.
- Several how-to pages now state what the code does. The code-review page puts
  the `code_review` section under `config:` (a top-level `code_review:` is refused
  as an unknown profile key); the virtual accelerator page names the archive
  store's host port as 10801 at the default port base, not the in-container
  27017; the add-MCP-server note says an `mcp_servers:` entry takes `hooks:` and
  `writes:`, and the read-only query page says a server's `writes:` list is
  blocked too; the Bluesky queue route table lists the move and export routes;
  the channel-finder page counts four pipelines, spells ReAct, and names preset
  channel addresses. A new test pins these and other quoted facts (the retired
  `mock` refusal, the mya client floor, the Phoebus `curl` hint and the
  `osprey query --json` keys) to the source that defines them.
- How-to pages for ARIEL, facility knowledge, deployment and health checks no
  longer give wrong instructions. The ARIEL quick start now runs `osprey build`
  before `osprey up`, and its command table lists every `osprey ariel` verb and
  the `hybrid` search mode. The picture-search page names `osprey ariel sync
  --watch`, which the `ariel-sync` service runs. Search-module and adapter keys
  go under `config:` in `profile.yml`, not in `config.yml`. The facility-knowledge
  page says `osprey knowledge` has three verbs and shows its real help. The
  compose-template excerpt matches the shipped PostgreSQL template. The
  project-image page says `osprey build` re-renders the Dockerfile every build.
  New docs tests check the ARIEL command table, the knowledge help screen and
  the template excerpt against the code.
- The Send and Retrieve Feedback page now lists everything the deployment-metadata checkbox attaches: the preset and its hash, the channel-finder mode and the escalation link as well as the version, application name and browser. The page used to say the version, application name and browser were all it added. The same change makes the Web Terminal landing page, the notebooks, send-feedback and privilege-tier pages, and the ARIEL search-sidecar page easier to read, with no other change of meaning.
- The Run the Web Terminal page quotes the switch-off refusal with the machine words the code appends ("the live machine", not "the real machine"), says a switch nobody answered within 30 seconds reads "✗ no answer" rather than `request_expired`, and places *reading values…* after **Review** is clicked, where the panel shows it.
- The build-profile reference now lists every top-level key `profile.yml`
  accepts, including the seventeen it had left out (`extends`, `machine`,
  `port_base`, `deploy`, `dispatch`, `optimizer`, the four chat bridges,
  `watchdogs` and others), and has a key table for an `mcp_servers` entry that documents
  `hooks` and `writes`. The ports page's examples moved the port base with a
  nested `config: deployment:` block, which replaces the whole `deployment`
  subtree; they now use the dotted key or the top-level `port_base:` shorthand,
  and its `dispatch:` example gains the `triggers` key the loader requires. The
  environment-variable page adds `OSPREY_PROFILE_VARIANT` and the three
  `OSPREY_ARCHIVE_*` variables. New tests check each table against the loader's
  key sets and load each ports example through the profile loader.
- The web-terminal guides now match the code in several places a reader acts on. Running the multi-user stack from a source checkout needs `--dev` on both `osprey build` and `osprey up` (or `osprey up --build --dev`); `osprey up --dev` alone refuses a render built without it. Sign out is the sidebar-footer row, not an item in the More menu. Panels discovered from disk run at an opaque origin, not the terminal's own. The multi-user config block is marked as an excerpt of the preset and shows its `auth:` lines. The roster and login snippets are written in the `profile.yml` form, `config:` with one dotted `modules.web_terminals:` key, because a nested `modules:` block there replaces the whole rendered `modules` subtree. The CANOPY pages' sections now nest under their titles. A new docs test holds the quoted refusal lines, environment names, message types, stylesheet paths, theme names, lint codes and preset excerpts to their definitions in the source.
- Web terminal: opening the events panel, or any panel declared under
  `web.panels`, no longer logs a 404 for `/api/panels/<id>/cap` in the browser
  console. The hub now answers that a panel is first-party instead of returning
  404, and treats a 404 as an opaque panel with no cap.
- The example facility's first six BTS horizontal and vertical correctors now read
  back a negative setpoint. Their readbacks were clamped at zero, although the
  supplies are bipolar and their limits allow down to -10 A.
- A run export on a deployment with no Tiled catalog now answers that no catalog is configured (409 `tiled_not_configured`), instead of asking the caller to retry.
- `osprey facility import` names files seeded under the pre-#2369 mml header, with the line-1 fix, instead of calling them hand-written.
- Enter now submits the web terminal sign-in pages even when a password manager
  handles the keystroke, and the landing and sign-in pages show the OSPREY tab
  icon. A folded sidebar group shows a chevron that unfolds it.
- `osprey build` records the render's file checksums after re-rendering
  `.claude/` and `.mcp.json`, so a fresh build no longer compares as drifted
  against its own manifest.
- Health: the versions row no longer warns that a CANOPY wheel installed at the
  proven commit is off the pin. A wheel records no commit, so the row now reads
  the commit from the version's `+g<hex>` label before calling it a mismatch.
- Web terminal: a hidden **Pinned** group stays hidden when another session is pinned, and across a reload, until **Show pinned sessions** brings it back. The first pin still shows the group.
- Hub tooling runs (the automatic session namer, the agentic code reviewer) no
  longer appear in the session list as phantom sessions that then name one another
  in a chain: the reviewer writes no transcript for the record to pick up.
- Generated project images build again. Once Playwright 1.65 was released the
  image build failed at the Chromium step, because Playwright's bundled Node
  needs `libatomic1`, which the image no longer carried; the image now installs it.
- An enabled watchdog's directory is now a build input: editing it marks a build out of date, images carry it, and the repository-boundary check covers it.
- A link inside a `data:`, convention or service-template tree that points
  outside the repository is now refused at load, like a top-level path.
- A link inside a profile input that does not resolve (missing target, a link to itself, two links to each other) is now refused before anything is installed, instead of failing the image copy after the install.
- `osprey facility import` now indexes a repo's own knowledge pages even when the tree held no bundled file, so `osprey knowledge validate` passes right after every import.
- The bluesky MCP server no longer loads FastAPI when it starts; the launch-token check imports it only when it refuses a caller.
- ``post_build`` and ``validate`` steps run on the project venv, as documented; ``pre_build`` steps and ``--skip-deps`` builds see the shell's ``PATH``. Lifecycle steps read the repository's env chain (``.env`` over ``.env.shared``), not the staged tree's.
- A session whose artifacts all fall on one day no longer shows a lone "Today" (or other) day heading in the Activity tree; headings appear once a second day does.
- Closing any context menu with the keyboard gives the focus back to the rail
  entry, tile tab, heading or tree row it was opened from even when that surface
  redrew while the menu was open, where it used to fall to the page.
- Run figures carry the units the facility declares for each channel, instead of a fixed "BPM units / A"; `get_run_data` states each column as `{name, unit}`, in the order a row's values follow.
  The `orm` plan's largest-kick parameter is renamed `span` (was `span_a`): the kick is in the corrector's own unit.
- Typing a panel's name in the command palette (its label, its id such as
  `ariel`, or a domain word such as `elog`) lists that panel first instead of
  under every setting whose key contains the word.
- An agent's second panel step in a row is no longer refused as `stale_base`
  when it reaches the page before the page has taken in the first step's
  answer. The page now runs agent panel operations one at a time, each after
  the previous one's answer is acknowledged (waiting at most the server's own
  5-second answer timeout), so the revision an agent was handed is always the
  one the page checks against. A step with nothing ahead of it still runs at
  once.
- Web terminal: the panel rail keeps the same order across restarts. `/api/panels`
  lists `enabled` and `visible` in the built-in catalog's order instead of a
  per-process order that changed on every redeploy.
- A web-terminal user, or an application, whose name takes a deployed application's audit identity is refused at render, by lint and by the application rules.
- Exporting a Plotly artifact whose plot did not draw within the render budget no longer writes a blank image; `artifact_export` answers with an error naming the file and the budget, and no image artifact is saved.
- Web terminal: a tile's **Open in new window** button hovered before its panel
  had loaded no longer stays greyed out once the panel has loaded. Before, it
  ignored every later hover and click until the page was reloaded.
- Approval pre-flight names a missing queue server as not configured, never as unreachable, and names no cause for a failure it cannot tell.
- A repository whose profile reads a file or data tree outside it is refused by `osprey validate` and `osprey build` before anything is installed, instead of building images without that input.
- A `.dockerignore` that would drop a profile's `extends:` base from the image now fails the build, naming the file.
- A build run with `--skip-deps` now renders the same dependency list into the
  generated Dockerfile as a full build: `environment.packages` and a declared
  base interpreter's frozen pins join `dependencies` in the image's install line.
  `osprey facility validate` renders that list too.
- A CANOPY `set_workspace` call relayed through the panel proxy now leaves its `unanswered` history row when the relay fails with any error, not only an upstream connection error.
- The qmd sidecar no longer leaves a stray `sleep` behind, or stalls for two
  minutes, when a long stage finishes the moment it starts: stopping the
  progress heartbeat now takes its sleep along wherever the stop signal lands.
- A qmd corpus whose embedding is turned off no longer keeps the vectors of earlier passes.
- The plans panel and the agent's queue tools tell you to retry only when the bridge
  says retrying can help.
- A bridge with no queue server refuses with `manager_not_configured` (409, not
  retryable) on every queue route; every queue refusal says whether retrying can
  help.
- `osprey init --revalidate` no longer names a repo's own skills, agents, rules,
  output styles and hooks in every login when its profile already names one by
  path: that repo is under the by-name rule already, so what it leaves unnamed
  stays unnamed.
- `osprey init --revalidate` no longer adds a provider entry or key OSPREY ships
  that a deviation marker in `providers.yml` claims absent
  (`# DEVIATION: providers.openrouter: …`), so `osprey validate` stops warning
  that the marker marks no difference.
- The Results table, the default figure, the orbit-bump monitor panel and every grid-scan panel state the units the run declares.
- One build asks the container runtime once and every render records the same answer; a tier or app file that states `container_runtime` is refused; `osprey facility validate` no longer asks the container runtime.
- `osprey scaffold --help` no longer lists the retired `osprey scaffold apps` among
  its examples.
- A service `template: osprey.<name>` is a bundled template only when `<name>` is a plain name; every other spelling is a profile-relative directory and meets the repository-boundary check.
- A profile service template outside the repository is refused by `osprey validate` and `osprey build` before anything is installed, and editing one inside it marks the build out of date.
- A profile naming a bundled service template OSPREY does not ship is refused at
  validation, naming the service and the value, and no longer builds without it.
- Session menu rename, export and delete, and artifact delete, now carry the hub URL prefix.
- Automatic session names in the web terminal hub now come from one plain chat
  completion on the provider's fast model instead of a Claude Code run, so a
  title arrives in seconds rather than after a CLI cold start.
- Web terminal: moving the pointer from one folded session in the sidebar onto
  the next now opens the next session's peek. The first peek's delayed close
  used to cancel the second one's pending open.
- Web terminal: session titles in the sidebar start 13 px further left. A closed
  session's status dot draws nothing there, and it no longer takes up room.
- A session's title now comes from the operator's own first words, or the slash
  command it started with, never from a subagent's completion notice or hand-back.
- Dragging the Sessions divider down on a short window no longer scrolls the
  sidebar: the Sessions group now stops at the room the groups below it leave,
  so the line stays under the pointer and the session list stays below the
  Search box. A height saved on a taller window is clamped the same way on load
  and on resize.
- The session ledger, the in-flight markers of an execution, a notebook cell and a
  session write, and nested facility-knowledge concepts no longer create their
  directory at the process umask: under umask `077` that left a `0640` file in a
  `0700` directory another uid could not enter, and when the ledger or a marker got
  to an identity's state directory first the control-context record found it existing
  and set no mode either. Every writer of a group-shared location now creates it
  through one helper that sets each level it creates to `2770`. The helper is
  `safety_file.ensure_shared_dir` with its named group-shared modes.
- The sidebar's highlights follow what is on screen: a tool row is highlighted only while its panel holds the workspace tile in the current layout, and an artifact only while the Artifacts panel shows it, so switching sessions or closing a panel leaves no stale highlight. Both use a quieter selected style.
- The sidebar's resize edge is easier to grab (a 12 px zone instead of 6) and shows an accent line while hovered, focused or dragged; the column-resize cursor stays for the whole drag.
- A `--skip-deps` build and `osprey facility validate` no longer refuse an `osprey_install` they would only need for a project venv they do not create.
- A `--skip-deps` build and `osprey facility validate` no longer refuse an unreleased, non-editable osprey (a deployment that pins osprey through committed wheels): the osprey spec is resolved only for a build that creates the project venv.
- Concepts drafted into the shared facility-knowledge bundle and the session
  ledger are now written group-readable (`0640`), so another web terminal or a
  dispatch worker running under its own uid can read them through the
  directory's group. Every file that inherited `0600` from `mkstemp` now states
  its mode beside the writer, with the reader it is chosen for.
- Sidebar: a layout saved by an earlier build now shows Events and Health under System, whichever group it held them in. The framework health panel is labelled Health, and the control-assistant preset shows it in System from the first load.
- An unreadable config no longer lets the web terminal's control-context owner claim at a guessed baseline (it claims nothing and refuses switch requests naming the file), the owner now claims from the terminal's own config file (an empty render when none resolved), and `${VAR}` in that file means the same on the posture chip as in the privilege gates.
- The terminal no longer loses its connected glow while it is connected. When the
  terminal is asked to resume a session that has no transcript yet, it drops that
  socket and starts a new session; if the new socket opened before the old one
  finished closing, the old close turned the glow off, and it stayed off. Only
  the live connection's close turns it off now.
- Three multi-user web-terminal snippets now pass `osprey profile validate` as written. The `control_identity` roster gives each entry its `index`. The snippet for TLS terminated in front sets `history_origin` beside `external_origin`, which the build requires once a sign-in deploys the history page. The history-scope roster now has an entry that holds the `operator` role it defines. Both multi-user pages also say that a list in a snippet, such as `users:`, stands as the whole list.
- Folding the last sidebar group no longer makes its heading jump, and the fold chevrons on group headings and the caret on the session picker are gone; double-click still folds.
- Closing an artifact or run row's menu in the sidebar tree with Escape gives the
  focus back to that row even when the tree redrew while the menu was open, where
  it used to fall to the page.
- The refusal of an invalid `va_archiver.password_env` now states every rule the
  name must follow, including that it may not start with a digit.
- Validating a session plan now refuses sample arguments that name a device the
  queue worker does not have, with the same `unknown_device` answer an enqueue
  gives, instead of passing a plan the queue later refuses.
- When git is slow, missing or fails inside a repository, the build now says it
  could not check whether `.env.variant` is tracked, with git's reason, instead of
  saying nothing. When git answers, or no repository encloses the deployment, nothing
  changes.
- A web terminal run on a host now marks its render trusted for Claude Code at
  startup, so the chat applies the deployment's allow list instead of asking to
  approve every read-only tool.
- An optimizer run that finished, failed or crashed under a person-approved
  write authorization no longer blocks approving another program on its channels
  for the rest of the eight-hour expiry; it stops blocking about two minutes
  after its lease is last renewed, when the approver can see the run's ledger
  (with none in sight it still blocks until it expires). A paused run still
  renews, so it still blocks. `OPTIMIZER_AUTHORIZATION_EXPIRES_S` is held just
  under 24 hours.
- Write authorization follow-ups. An optimizer run cancelled while waiting for
  approval no longer reports `awaiting`; a manual write with the switch at `asked`
  is refused `switch_not_allowed` rather than `auth_revoked`; and a run's request
  is labelled `<environment> optimization (<job id>)`. The open chip popover keeps
  its programs current, the machine name is no longer cut off by the program
  badge, and the confirmation for `allowed` says programs then authorize
  themselves. `osprey write-authorization approve` on a request that is no longer
  pending says whether it was already approved. `.env.example` leaves
  `ANTHROPIC_API_KEY` commented out, so copying it to `.env` no longer overrides a
  Claude login with a placeholder key.
- A failed ARIEL migration no longer leaves the database half-changed. Each
  migration now runs inside one transaction that also carries its
  `ariel_migrations` row, so a migration that drops an index before creating its
  replacement can no longer destroy the old one when the create fails. Two
  shipped migrations do exactly that -- the HNSW embedding-index reconcile and
  the semantic FTS index -- and in both the create is an index build, the
  statement most likely to fail on a large table. Committing the bookkeeping row
  in the same transaction also means a migration is never recorded as applied
  unless its work landed, so it cannot run a second time against the state it
  already made. (#1025)
- The bundled PostgreSQL service now asks for 1 GiB of `/dev/shm` instead of
  taking the container runtime's 64 MiB default, so parallel index builds and
  parallel queries stop failing with `could not resize shared memory segment ...
  No space left on device` on a host with plenty of free disk. PostgreSQL places
  every parallel worker's dynamic shared memory in that mount, and an index build
  sizes its segment from `maintenance_work_mem`, which defaults to the same
  64 MiB the mount was capped at. The new `services.postgresql.shm_size` key
  overrides it; raising `maintenance_work_mem` past the mount re-creates the
  failure, so the two move together. (#1026)
- Semantic search in ARIEL deployments created before the HNSW switch ran on a
  vector index trained over an empty table, which silently gave it low recall.
  The `text_embedding_hnsw_index` migration rebuilds the index on upgrade;
  treat semantic-search results from before that migration as unreliable and
  re-run anything you concluded from them. (#1028)
- Two channels in one archiver request no longer answer the same window on
  different bin grids when their data starts on different days: client-side
  binning now anchors on the query window instead of on each channel's first
  sample, which also puts the MYA connector's `median` on the same grid as its
  `mystats` modes. (#1078)
- The isolated picture renderer's per-loop guard task is dropped without an
  asyncio `Task was destroyed but it is pending!` log record when its event loop
  is closed, or stops for good, before the task can finish.
- A completed `chromaticity_monitor` measurement no longer fails while it is
  saved: `pyaml_measure` writes each measurement as JSON itself, every array as a
  list, where pyAML's own save could not write the monitor's arrays.
- A measurement file in which two group roles name the same group stops the build
  as `value-invalid`, naming both roles, unless the two roles are `hcor` and
  `vcor`: those may share a group, which is split by plane. The pyAML name mapping
  refuses the same shape.

  A profile whose `approval.enabled` is stated as anything but `true` or `false` (an empty
  line, a quoted string, a number) stops the build as `profile-invalid` on
  `approval.enabled`: read by truthiness, such a value turned every approval prompt off
  without a word. An unstated switch still reads as asking.
- `use_name_server: "false"` (for example from `${USE_NS:-false}`) now selects
  address-list routing instead of name servers; a value that is neither true nor
  false is refused by name. (#1252)
- The Session Activity Log page remembers a theme picked on it, follows the system light/dark setting while on auto, and opens in the deployment's configured theme on a first visit, like the other pages that run on their own. (#1315)
- The agents and tool-log views of the Session Activity Log show the conversation a session is on after `/clear`, matching its conversation view, instead of the one it had before. (#1315)
- Replacing a pooled web terminal whose launch environment changed no longer
  blocks the web server while the old process shuts down; the kill runs in a
  worker thread, as eviction's already does. (#1320)
- The web terminal's session picker no longer lists transcript files that hold no
  readable record, which it showed as empty "(no user message)" conversations that
  could not be resumed. A session's message count now counts only readable records. (#1321)
- After a `claude_code.provider` switch, `osprey up` now re-renders the
  `.env.users` it wrote, so web terminals start with the new provider's key
  instead of asking for a claude.ai login. An `.env.users` that is missing the
  key the terminals sign in with now refuses the deploy (when you wrote or
  edited the file yourself) and shows up as an error in `osprey health`. (#1446)
- `osprey up` no longer fails at the CA layer when it rebuilds a service image
  for a deployment with `images.site_ca` set. Every start now copies the site CA
  into each compose-built service's build context just before the build and
  removes it afterwards, even when the start fails; `osprey build` no longer
  leaves a copy of the CA in `build/services/`. (#1447)
- ARIEL's `sql_query` tool no longer refuses a read-only query because a quoted
  string in it contains a word such as `vacuum`, `set` or `update`, or a `;`.
  Forbidden keywords and statement separators are now matched only outside
  string literals. (#1450)
- `/clear` sent as a message in the chat no longer leaves the session key on the conversation it just left: the chat child reports the transcript it is on with every turn, the chat route records that id in the transcript map, and a replay, a hand-off to the terminal and the next child under the key all continue the conversation the operator is in. The stream marks the boundary with a `session_reset`, drawn as a divider. (#1483)
- The Tiled service no longer crash-loops on a catalog volume written by an
  older Tiled: it runs `tiled catalog upgrade-database` before serving, and a
  fresh volume still initialises. If the upgrade fails, the container logs the
  reason once and reports `unhealthy`, so the problem shows in `osprey health`
  instead of restarting forever. (#1516)
- The qmd sidecar no longer serves a partly embedded index as if it were
  complete. `qmd embed` stops after a 30-minute session and still reports
  success, so the sidecar now repeats it until no document is left without
  vectors, or until a pass embeds nothing, and logs how many remain. (#1568)
- The qmd sidecar no longer dies at startup with a healthy daemon when its
  container carries a proxy environment whose `no_proxy` lacks `::1`. The
  entrypoint's health probes and the compose healthcheck now bypass any proxy,
  since both only ever talk to the container's own loopback. (#1576)
- A deployment now has a tracked name: `project_name:` in `profile.yml` names its
  compose project, volumes, images and tier and app renders, so a clone under any
  folder name deploys as the same thing. `osprey init` proposes it from the
  folder name. An older profile stops `osprey build` once with the line to add,
  and once more if its render catalog still spells `project`/`project_path`,
  which the build now derives. (#1647)
- Container commands that OSPREY runs with captured output no longer inherit the
  terminal's input, so a prompt reads end-of-input instead of waiting unseen; a
  site `scripts/verify.sh` that prompts now reads EOF. When compose keeps an
  existing volume that no longer matches the compose file, a warning names the
  volume and the log file, and `scripts/verify.sh` takes the compose project
  from `project_name` in `build/config.yml`. (#1652)
- A provider class that leaves out `requires_api_key`, `requires_base_url`,
  `requires_model_id` or `supports_proxy` is now refused when it is defined, with
  a `TypeError` naming the provider and the missing flags. Such a class used to
  inherit `NotImplemented`, which counted as true on Python 3.11 to 3.13 and
  raised `TypeError: NotImplemented should not be used in a boolean context` on
  Python 3.14 wherever the flag was read. The unit tests now also run on Python
  3.14. (#1778)
- `osprey up` no longer fails on a host where `docker` is podman's Docker-CLI
  emulation (the `podman-docker` package, standard on RHEL). `container_runtime:
  auto` now recognises the emulation and drives the host as podman, so compose is
  no longer handed a `--progress` flag podman-compose rejects, and the
  podman-specific steps run. A `docker` set explicitly on such a host is refused
  with the setting that matches it, instead of being quietly re-pointed. (#1826)
- `osprey up` reports success only once the web terminals answer. It waits for
  each web-stack service's own healthcheck (`web_terminal_ready_timeout`, default
  5 minutes) and reports each terminal as it becomes ready; a terminal that stops
  or does not answer fails the start with its log, and the ones that are serving
  are named and left running. While a terminal is still starting, nginx answers
  with a page that reloads itself instead of a bare `502 Bad Gateway`, and the
  ARIEL launcher no longer logs a failed health check for a server that is only
  slow to answer. (#1827)
- A `compose up` that waits forever on a dependency no longer leaves `osprey up`
  showing only a running timer. Every 30 seconds it names the containers it is
  still waiting on and the last line compose wrote; after a time limit
  (`compose_up_timeout`, default 30 minutes) it stops and fails with that account
  plus the last log lines of every container that exited or is unhealthy, and
  Ctrl-C prints the same account. A failed run's replayed output prints a line
  compose repeated once, with its count. (#1827)
- `osprey chat` in a deployment with no logbook, such as one made from the
  `hello-world` preset, no longer fills the terminal with ARIEL connection
  warnings. The ARIEL companion server now starts only when the build rendered
  an `ariel:` section, which it does when the ARIEL server is on; without one
  the panel had nothing to reach but a PostgreSQL the deployment never declared,
  so every first session opened on pool errors and a degraded-mode banner. (#1947)
- `entry_create` with `draft=True` now refuses an artifact it cannot attach, as direct mode already did, instead of writing a draft without it. Container images ship the headless Chromium the converter needs, so HTML, Markdown and notebook artifacts (such as Plotly plots) attach as PNGs. (#1951)
- `osprey ariel` commands find the deployment's `build/config.yml` when run from the repo root or any directory inside it, like the other commands; no `CONFIG_FILE` export is needed. (#1954)
- The ARIEL `als_logbook` adapter now writes correctly to the ALS olog's 3.4.032
  RPC API. A refused username or password reopens the credential prompt instead
  of failing as "Publish failed" (older olog releases, which answered a refused
  login with 200, used to have the error stored as the new entry's ID). The entry
  XML is sent as the `xml` form field, so a `+` or `%xx` in the text is no longer
  URL-decoded by the server. A logbook the olog does not list is refused before
  anything is posted, instead of the entry landing in Operations. A success
  response without an entry ID is now an error that says to check the logbook
  before resubmitting. (#1982)
- A follow-up question in a chat thread whose earlier answer carried a plot over
  5 MiB no longer fails at once. The bridge re-attached earlier plots within an
  8 MiB total but never checked one plot against the worker's 5 MiB per-image cap,
  so the worker refused every later question in that thread. An earlier plot over
  the cap is now skipped and noted as unavailable, and the question runs. (#2051)
- A long chat answer no longer trips Google Chat's write quota and gets re-posted
  from the top. The bridge posted an answer's chunks back to back, and a
  "Quota limit exceeded" part-way through made the next retry post the whole
  answer again. Message creates are now spaced a second apart, and a rate-limit
  refusal is waited out and retried before it counts as a failed delivery. (#2053)
- The CBorg and AmSC i2 providers now serve text embeddings, so ARIEL's
  `text_embedding` module can use them. CBorg defaults to `lbl/nomic-embed-text`. (#2092)
- `osprey deploy` now reads `control_system.limits_checking.database_path`
  the way the connector does. A path with surrounding whitespace, which the
  connector opened, was refused at deploy as a missing file; it is now mounted. (#2114)
- A host variant that adds a `services:` entry no longer stops `osprey build` on
  a repository with tiers or apps. The variant overlays the deployment's
  `profile.yml`, so each tier and app render now sees it through that root and is
  then attached as before; it was merged into each tier and app file instead, and
  every render was refused for a `services:` key its file never spelled.
  `osprey validate` on a tier or app file now judges it over the varied root too,
  and a render's drift fingerprint folds the selected variant. (#2128)
- A web terminal no longer comes up read-only following a thread of itself as the control-context owner, including in the target-state hook. (#2172)
- Signing out of the web terminal no longer lands on the login page when a
  channel failed in the moment before the page left. (#2185)
- The web terminal's sidebar no longer calls a panel that is not served: the
  artifact lists and the health dot run exactly while the panel has a backend. (#2185)
- The virtual accelerator and its live stand-in serve every float channel with a display precision: the facility description's `precision` when the channel states one, else 4 decimals. Before, a channel stating none was served Channel Access's default of 0, so a display that takes the PV's precision showed a 0.05 mrad corrector kick or a 0.219 tune as `0`. (#2244)
- A panel backed by a service hosted under a sub-path of its own now loads. The
  panel proxy rewrites that sub-path in the pages and bundles the service serves,
  so an asset reference written root-absolute resolves inside the tab instead of
  at the terminal's own address. (#877)
- The artifacts gallery no longer takes minutes to show a long list. Every row
  reveals with a short fade whose delay grew with the row's position, so after
  each page that arrived or each artifact the server announced, the list was
  rebuilt and a row a thousand places down stayed invisible for half a minute;
  the delay is now staggered over the first dozen rows only. The row handlers
  are installed once on the list instead of three per row on every rebuild.
  Measured with 1,500 artifacts held: the deepest row visible 0.6 s after the
  last page instead of 45 s. (#924)
- A notebook cell run just after a control-target switch is no longer refused
  with `switch_in_progress` once the header chip has settled. A controls server
  that holds no connector and is launching its first one refuses only its own
  session; a server swapping a connector it already holds still refuses the
  whole deployment until the swap lands. (#935)
- A relative `registry_path` now anchors on the deployment's `project_root` rather than the working directory, so a project's own registry — and the ARIEL adapters, search modules and enhancement modules it registers — loads from anywhere a deployment verb runs, not only from the repo root. (#965)
- ARIEL's module and watcher locks are free the moment a pass ends, so a pass started right after another one no longer finds the lock held and skips its work.
- `osprey build` refuses a `claude_code.agent_models` pin that names no agent,
  and lists the agents there are. It also refuses a pin on a claimed or profile
  agent whose own file runs a different model, instead of ignoring the pin.
  `osprey status --agents` lists the agents the build ships, each with the model
  its agent file names.
- A process that used aioca before the EPICS connector connected through a
  gateway no longer segfaults at exit, and aioca calls on that thread no longer
  time out after `connect()`. The connector speaks Channel Access through aioca
  itself now, so the process has one CA context, and the connector's CA loop is
  stopped before aioca tears that context down at exit.
- `osprey build` prints one warning when Claude Code's haiku, sonnet or opus
  alias runs the main model because the provider serves no model of that family,
  and one naming every configured model id the provider's served list does not
  carry. `osprey status` says the alias substitution once instead of twice.
- A request through the als-apg gateway no longer carries a temperature: its Claude 5 models (Haiku 5.5, Sonnet 5, Opus 5.5, Fable 5.1) refused every value but 1, which failed any completion that asked for 0.
- The `als_logbook` adapter stores entry text as plain text. HTML entities are decoded and the logbook's markup becomes line breaks, paragraphs and list items, so ARIEL Browse, keyword search, embeddings and agent tools no longer show `&lt;br /&gt;`. The backslash escaping the logbook adds is undone as well, so `it didn\'t` reads `it didn't` and `Y:\\opstat` reads `Y:\opstat`. Entries stored earlier are rewritten once by `osprey ariel migrate` (which `osprey ariel sync` runs first) and are enhanced again on the next enhance pass.
- A login's render binds each facility guard once. A host's `claude_code.hooks.PreToolUse`
  guard spelled as a mapping (`hook:` with a `matcher:`) was added again by the
  guard floor, so the login's settings ran it twice.
- A `config.yml` whose `approval.enabled` is edited after the build to anything but `true` or `false` (or whose `approval` section is left bare) no longer switches approval off: the approval hook denies every call it is wired to and a read-write Python run does not start, both naming the key and the fix.
- The pre-execution review notebook now shows how the call ended. When an
  approved call runs, the notebook says it ran. When the call fails, the
  notebook says it failed. When the operator declines the call, the notebook
  says the call was not run. A call that is interrupted while it runs is marked
  not run when the conversation next ends a turn.

  The gallery line for the review changes to match. It no longer says the code
  is pending approval.
- The mock archiver and the archive seed remove the temporary directory of each archive composite they build when they are done with it, instead of leaving it to the garbage collector.
- A web terminal now receives every credential variable its archiver block names
  under `auth:` (`auth.token_env` or `auth.password_env`), so an archiver behind a
  login answers inside a terminal as it does on the host.
- A deployment with the archive recorder now builds and records channels whose
  address carries a field name (`<record>.RBV`) or any other character the store
  treats as syntax. Such channels are stored and read back under their own
  address, and archives written before keep reading unchanged.
- `osprey up` now refuses an archiver `auth.token_env` or `auth.password_env` with
  surrounding whitespace, naming the key. Such a name used to deploy and then fail
  at the first archiver read inside the web terminal.
- A channel whose address equals one of the archive's own document fields
  (`date`, `_id`, `expireAt`, the densify marker `osprey_densified`, or a seed
  manifest field such as `fingerprint`) is now recorded, seeded and read back
  under its own address instead of overwriting or matching that field. Its first
  character is stored escaped, so a channel named `date` lives under `%64ate`.
- The ARIEL panel reads `web.config_panel.enabled` through the loaded
  configuration, as the web terminal already did. A deployment that writes the
  switch as an environment-variable reference — `enabled: "${PANEL:-false}"` —
  now closes the Config panel on both surfaces; before, it closed only on the
  terminal and ARIEL left the panel reachable.
- A logbook entry longer than its embedding model's input window is cut so its
  start is embedded, instead of failing on every enhancement pass. Each model
  under `ariel.enhancement_modules.text_embedding.models` states its window as
  `max_input_tokens`; the presets set 2048 for `nomic-embed-text`, and a model
  that states none is cut to 512 tokens. `osprey ariel reembed` applies the same
  cut.
- An ARIEL search or listing whose `end_date` is a bare date (`2025-10-06` or
  `20251006`) now includes the entries written on that day; it used to stop at
  midnight at its start. An end date with a time of day is still taken exactly
  as written, and a number is still read as epoch seconds.
- An entry whose enhancement fails three times is left out of later enhancement
  passes instead of being retried every cycle; the count is kept in the entry's
  enhancement status, and a success clears it. Each module now runs only on the
  entries it has not finished.
- The ARIEL panel shows entry times in the facility time zone and in the
  browser's own date format, matching the times ARIEL's agent tools quote. They
  no longer shift to the viewer's zone or force US month/day order.
- Vocabulary expansion of a long ARIEL logbook entry no longer slows down with
  the square of its length (an 87,000-token entry: about 0.5 s instead of 8 s).
  The expanded text is unchanged, so stored embeddings stay valid.
- `osprey health` reports ARIEL's entry count, last ingestion and enabled modules again on a gated panel. The probe reads the panel's open `/health` page instead of the signed-in `/api/status`, which refused it.
- `osprey ariel qmd-resync --help` no longer cites source line numbers.
- `ariel.ingestion.poll_interval_seconds` and `ariel.ingestion.watch.max_interval_seconds` are refused at config load, naming the key, unless they are a positive, finite number of seconds; an infinite or NaN value no longer stalls `osprey ariel watch` or stops its event loop. `osprey health` judges ingestion staleness from the same parsed values and warns with the key when ARIEL refuses them.
- ARIEL keyword search no longer comes back empty when no entry contains every
  word of a plain-word query. It then returns the entries holding at least half
  of the words, ranked by how many they hold, each naming its `matched_terms`
  and `missing_terms`, with an INFO diagnostic listing the words no hit
  contains. A vocabulary alternative counts as the one word it stands for, and
  hybrid search decides that a picture caption matched by the same word count.
  Operator queries, quoted phrases and patterns are matched as before.
- An ARIEL keyword query whose vocabulary term carries edge punctuation
  (`BPM?`, `the RF.`) keeps its expansion instead of silently dropping it.
- ARIEL now reads a logbook time written without a UTC offset as facility-local time
  (`system.timezone`) and stores the instant it names. Such times were stored as if they were
  UTC, and `osprey ariel watch` skipped every one of them. Entries already stored keep their old
  time until they are ingested again; re-running `osprey ariel ingest` over the same source
  corrects them. `osprey ariel ingest --since` is read in the facility zone too.
- ARIEL's New Entry form: Cancel now discards the form and returns to the view it was opened from. Before, it only cleared the text fields and stayed on the form, so on an empty form it seemed to do nothing, and it left the tags behind. Cancel and a successful Submit now also discard an agent draft's staged attachments, metadata and banner; before, the next entry attached the same files again. After Submit, the new entry opens over Browse instead of over an empty form.
- Asked to open "the entry" after a logbook subagent read the answer off one of
  its pictures, the agent now opens it with that picture enlarged. The logbook
  subagents hand back the entry id and `attachment_id` of every picture they
  viewed; `entry_open` takes `attachment_id` as a required field that may be
  null, and lists the entry's viewable pictures when opened without one. The
  logbook guidance no longer offers an entry id as a value the entry records, and
  views an uncaptioned picture before reporting an answer missing.
- ariel search: search calls run on at most four worker threads per process. When a hung embedding provider holds all of them, the next search is refused at once with a message saying so. Semantic search reports an embedding timeout as a timeout instead of returning no matches; hybrid search with the `rrf` strategy keeps its keyword matches and warns that semantic ranking was unavailable.
- ARIEL search-time provider calls (query embedding and the reachability check
  behind it) work in a process forked after the parent has made one: each
  process now admits these calls through its own bound, so a child never waits on
  a worker that exists only in its parent. A call the search gave up on after its
  timeout keeps running on a daemon thread, so it no longer keeps the interpreter
  alive at exit while the provider has not answered.
- Stopping the ARIEL service (the MCP server, the web interface, or `close_ariel_service()`) now also ends its search-call worker threads, so a process that ran one search no longer holds them open until it exits.
- `osprey ariel status` reported `pending: 0` for an enhancement module that had
  never run, so a freshly seeded logbook looked fully enhanced until the first pass
  wrote a status key. A module the store has never seen now reads every entry as
  pending, the same count a module that has processed one entry reports as the
  rest.
- `osprey ariel sync` and the other ARIEL commands no longer open the agent's read-only SQL pool or log `SQL tool is running on the ingestion role`. Only the ARIEL MCP server, which hosts the SQL tool, opens that pool.
- Logbook entries that ARIEL skips because it cannot read them now count as failed in
  `osprey ariel watch` and `osprey ariel sync`, and `osprey ariel ingest` says how many it
  skipped. They used to appear only in the log.
- An ARIEL logbook entry whose time is missing or cannot be read is now skipped and named in the
  ingest log. The built-in adapters used to store such an entry with the time of the ingest run,
  or with 1 January 1970, and nothing marked it.
- `osprey artifacts web --port 0` binds the port it is given instead of the
  configured one, and with both `--host` and `--port` given the command no
  longer reads the config for its address, as `osprey ariel web` and
  `osprey channel-finder web` already behave.
- `osprey artifacts web` no longer fails at startup with a `TypeError`. The
  gallery serves the deployment's shared agent-data directory when no root is passed.
- `osprey up` without `-d` no longer leaves a copy of the site CA bundle under
  `build/services/<name>/` once the images that needed it are built. Both start
  shapes now clear it, as the project, tier, app and auth-sidecar builds
  already do.
- A repeated `osprey up --dev` no longer fails building the auth sidecar image: wheels staged by an earlier dev deploy are cleared from its build context first.
- A web-terminal login whose stored password hash the login service cannot read
  is now recorded as `credential_unevaluable` instead of `bad_credential`, and the
  service names the user at startup. The browser still sees the ordinary refusal;
  `osprey users passwd <user>` replaces the hash.
- A webhook trigger or a dispatch-worker call whose `Authorization` header carries
  a character outside ASCII now answers 401 rather than 500. Both gates weigh the
  bearer as bytes, as the dispatcher's own routes already did, so a wrong
  credential is refused on every surface.
- The channel-finder benchmark's coverage judge runs on a provider the project configures under
  `api.providers` (the benchmarked model's own, or another named with `judge_provider=`) instead of whichever
  vendor key happens to be exported. A judge the project cannot run is refused when `BenchmarkRunner` is
  built, and a judge call that fails fails its query instead of being rescored by substring match.
- The channel-finder benchmark refuses a project `config.yml` that does not parse
  or is not a mapping, naming the file. `--backend auto` no longer falls back to
  the default backend on such a file, and the coverage judge and the ReAct
  backend no longer fail on it with an unrelated `AttributeError`.
- The channel-finder benchmark's `sdk` backend records each tool call's result,
  waits for the channel-finder server to connect before the first turn, and
  routes OpenAI-protocol providers through the translation proxy. Scores are
  still computed from the agent's answer text alone.
- The BLUESKY panel's History no longer says "No completed runs yet." next to
  rows for removed plans. The line appears only when History lists nothing.
- `list_devices` answers a Bluesky bridge reply that is not a device page with a
  `bluesky_bridge_error` refusal instead of failing with an unhandled `TypeError`.
- The `orm` and `orbit_bump_sweep` plans run again under bluesky 1.16, where
  they failed before taking a point with `Plan failed: 'setpoint'`. bluesky 1.16
  changed `bps.locate` to answer by device name; the plans now read each
  working point with `bps.loc`. OSPREY now requires `bluesky>=1.16.0`, and the
  `writing-bluesky-plans` skill teaches `bps.loc`.
- The Bluesky bridge, queueserver and sidecar containers now load the project
  config through OSPREY's own loader. `bluesky.plan_module`, `bluesky.plan_dirs`,
  `bluesky.excluded_plans` and `python_executor.child_env_passthrough` set in
  the config now take effect there, the sidecar lists every plan lane a two-lane
  deployment renders, and pre-flight and sidecar audit records reach the
  project's agent record instead of staying inside the container.
- A deployment that declares a `bluesky_standin` plan lane now hands the
  Bluesky panel sidecar that lane's bridge URL and launch token, starts the
  sidecar after that lane's bridge is healthy, and hands the dispatch worker
  the lane's bridge URL, so the panel and the worker's agent can reach the
  stand-in lane like any other.
- A queued question whose destination the platform permanently refuses no longer
  stays in the retry queue failing the same post once per drain pass. Google
  Chat's "This Chat app is not a member of this space" (the app was removed from
  the space) and a 404 for a space that no longer exists now raise the new
  `UndeliverableError`, and the bridge settles the entry terminal with the
  refusal recorded in `give_up_reason` instead of retrying it for a week.
  Transient failures keep today's behaviour and stay queued.
- A build refused because its base environment cannot be reproduced, or because its OSPREY install spec is invalid, now leaves the project environment untouched.
- When the dependency install of `osprey build` cannot reach the package index, the build error now ends with the same advice as the install timeout, after the installer's own output: re-running resumes from the local cache, and `UV_INDEX_URL` (or `PIP_INDEX_URL` without uv) points the build at a closer mirror.
- `osprey build` no longer removes `build/nginx/` and `build/docker-compose.web.yml`
  from under a running web-terminal stack, so the landing page keeps being served
  while a rebuild runs.
- The `seed-invalid` stop for a nominal outside its limits band prints the nominal and the bound in full, so widening the limits record to the printed nominal clears the stop.
- The BuildKit pre-build check reads `DOCKER_BUILDKIT` the way Docker does (`false`, `true`, `F` …) and refuses, by name, a value Docker would refuse.
- The CA file a connection block names under `tls.ca_bundle` is now mounted
  read-only at the same path into every web terminal, dispatch worker and archive
  recorder that reads the block, so the key works in a container as it does on
  the host. The path must be absolute: a `~` path is refused, because each process
  would expand it against its own home.
- Raising `ariel.enhancement_modules.image_caption.max_images_per_entry` now
  reaches pictures already recorded as over the cap: `osprey ariel enhance
  --module image_caption --retry-failed` captions the ones now within it, and
  lowering the cap keeps existing captions. `osprey ariel status` reports how many
  pictures are past the cap and names the command.
- The middle-layer channel-finder import now loads its DuckDB tables in bulk
  instead of one row at a time, so building a channel-finder index for a large
  middle layer takes seconds rather than minutes. The imported rows and the
  reported counts are unchanged.
- The channel-finder MCP servers resolve `${VAR}` and `${VAR:-default}` in
  `config.yml`, so a database path or model set per host through the environment
  reaches them. A config file that cannot be parsed is reported by name, and the
  server starts unconfigured instead of failing.
- The channel-finder MCP servers find `config.yml` the way every other OSPREY
  process does: `OSPREY_CONFIG`, then `build/config.yml`, then `config.yml` in the
  working directory. A server started by hand from a repo root now reads the render
  and resolves its database paths against it.
- In a narrow Channel Finder tile, an opened device's card now shows at once, docked under the results. Before, the card was drawn below the whole device table and the operator had to scroll the tile to find it. The facets, the results and the card each scroll inside themselves, and the row the card was opened from stays in view.
- The channel-finder web API answers `503` with `Database not available for pipeline '<paradigm>'` on every route when the active paradigm has no database; the tree, structure, in-context edit and device-info routes answered `500`.
- The middle-layer channel index stores `channels.updated_at` as a timestamp with
  time zone, so it reads back as the import's instant instead of local wall time
  labelled as UTC.
- `osprey channel-finder validate --pipeline <paradigm>` without `--database`
  now validates the file at `channel_finder.pipelines.<paradigm>.database.path`
  instead of the auto-detected paradigm's database, and refuses, naming that
  key, when it is unset.
- An unreachable channel no longer tells the agent the rest of the control system still answers; it says how to tell a dead channel from a dead machine.
- `osprey chat` no longer paints its companion servers' log records over the
  agent's screen. Once the agent is launched, what the companion threads log --
  a health check that passes late, a database pool that times out on a slow
  host -- goes to `var/logs/chat-<timestamp>.log` in the deployment repo, named
  on the line before the launch; errors still reach the terminal. Nothing is
  dropped: every record is emitted as before and reaches the file.
- The Simple view shows the session picker again, so an operator can return to
  an earlier conversation. Picking a session moves the chat to that conversation
  and replays it, and does not start a terminal. The picker list takes keyboard
  focus when it opens, and Escape returns focus to the picker button.
- A claimed service template (`services/<name>`) now reports framework drift when a
  newer OSPREY changes the packaged template, as a claimed Claude Code artifact
  already did. The drift check used to look for every template under the Claude
  Code tree and skipped services without a word.
- A deployment whose profile sets `claude_md_template:` now treats that template as its
  `CLAUDE.md` everywhere: a `CLAUDE.md` listed as user-owned under the template's name
  survives `osprey build`, the web terminal's scaffold gallery names and renders that
  template, and an unknown `claude_md_template:` fails the build by name.
- The lint warning about logins over plain HTTP now reads `external_origin`, the address browsers
  use, instead of `deploy.fqdn` alone. It appears for a loopback `deploy.fqdn` behind a real
  `http://` origin, and it no longer appears behind an `https://` terminator.
- `get_chat_completion` refuses a call with no model id for every provider
  instead of handing the provider an empty one, and a `model_config` that names
  no provider is refused as such rather than as `Unknown provider: None`.
- A re-rendering deploy of a project with a live stand-in no longer has its
  host-port preflight report the virtual accelerator's ports as conflicting
  with themselves. The compose render now lists each compose file once,
  however many deployed services share its template.
- A site build-arg value carrying a quote character no longer renders a compose
  fragment that fails to parse. The value reaches the build exactly as it was
  declared, so an index or proxy URL with credentials in it is usable.
- A write a served physics model cannot solve is refused by the mock connector with the model's error text, and the model keeps the inputs it had.
- An empty section in `config.yml` (for example a bare `services:`) now reads as no settings instead of failing on first use. A section that is not a mapping is refused at load with its key and file named.
- The Web Terminal settings form shows the keys the build owns read-only, with a note to change them in the profile's `config:` block and re-run `osprey build`, instead of offering a control whose Apply is refused.
- A connection block's `auth.token_env` or `auth.password_env` that ends in a newline is refused
  as not naming an environment variable. It was accepted with the newline kept, so the connector
  looked the secret up under a name no environment holds.
- A connector-host child no longer loads the project `.env` from its working
  directory when it reads its config file, so an `EPICS_CA_*` or `EPICS_PVA_*` line
  there cannot undo the scrub the child was started with.
- A connector-host child that exits on its own before answering its init frame
  is now reported with the exit code it exited with. Terminating a child that
  had already exited but was not yet reaped stole its exit status from the event
  loop, so the start error could say "exit code 255" for a child that exited 3.
- Putting a connector-host child down sends no signal to a child that has
  already exited or already been reaped. A pid that asyncio's child watcher
  thread had reaped could otherwise still be signalled, and by then it may
  belong to another process.
- A TANGO or DOOCS read that cannot reach its device now reports a connection
  error naming the channel, as an EPICS read already did. The control-system
  tools answer it the way they answer an unreachable EPICS channel — the
  operator is told the control system could not be reached, and the connector
  is rebuilt on the next call — instead of reporting it as an unexpected
  internal error.
- A container runtime that is slow to answer is no longer reported as not
  running. The deploy verbs wait for `docker ps` / `podman ps` and the compose
  version probe to answer, say they are still waiting after 5 seconds, and stop
  only after 2 minutes, saying the runtime did not answer. A refusal names only
  the runtimes it probed, each with the probe that failed.
- `python -m osprey.deployment.runtime_helper` prints the container runtime
  `osprey up` would drive, and the virtual-accelerator developer scripts take
  their runtime from it, set with `CONTAINER_RUNTIME`.
  The virtual-accelerator build check reads the status vocabulary from the model
  surface instead of restating it.
- Write-refusal messages, the control-target popover and the docs now say a switch applies to every session of this login, not deployment-wide: the switch record is kept per login identity.
- A machine switch that lands says "Switched to <machine>" under the sidebar's
  control-target block, so the popover's re-render is not the only sign that the
  deployment moved.
- A cron trigger whose `interval_sec` is infinite, NaN or too large for a float is now skipped with a warning like any other bad interval, instead of never firing, dying on its first wait, or stopping the dispatcher from starting. A quoted number such as `"90"` is read as that many seconds.
- In the web terminal's Customize mode, an item's options popover takes the
  keyboard focus when it opens and gives it back to the item when it closes.
  After Move left or Move right it stays open on the moved item, so pressing the
  button again moves the item further.
- The Customize bars sheet keeps the keyboard focus on the tile it was on when
  an edit is saved. Adding an item with Enter no longer drops the focus to the
  page.
- The `container` health check grades only this deployment's containers. A
  container labelled for another OSPREY or compose project on the same host is
  never reported as this deployment's, a service is matched by whole name
  segments or its compose service name, and a service that runs several
  containers reports each one that is not running.
  `osprey status` claims containers by the same rule, so a container labelled
  for another project is never listed as this deployment's.
- `osprey down`, `status`, `logs`, `reset` and the container health checks act on the project `osprey up` started, with the runtime it started it with, even from a fresh shell or after a rebuild renamed it. With no deployment record they say nothing has been deployed from this repo yet.
- `osprey web --detach --port 0` reads the port only from a whole startup line in
  the child's log, so a line still being written can no longer be read as the
  wrong port or crash the launch.
- A trigger's `action.allowed_tools` that is not a list of tool names is refused
  when the triggers file loads, naming the trigger. Write one tool as `[get_pv]`;
  the bare `get_pv` form is gone. A blank `allowed_tools:` means no tools. Before,
  such a trigger loaded and every fire was refused by the worker.
- The dispatch dashboard shows a trigger with an empty `allowed_tools` as having no tools (`none`); it read "all allowed", though such a run may use no tool from its trigger.
- A tool call whose result has no content is recorded in a dispatch run as an
  empty result, instead of the text `None`.
- The event dispatcher's dashboard now shows the deployment's `facility.name` when the profile's `dispatch.facility_name` is unset, instead of an empty name. `dispatch.facility_name` still overrides it.
- A dispatched agent now reads an event's time, and the times `trigger_history`,
  `list_triggers` and `trigger_status` report, in the facility zone set by
  `system.timezone`, the zone of its own clock. The dispatcher's stored history
  stays in UTC, and a time a webhook body carries reaches the agent as sent.
- An unattended dispatch run can keep a task list: the dispatch permission hook lets the CLI's task-list tools (`TaskCreate`, `TaskGet`, `TaskList`, `TaskUpdate`) through without a trigger allow rule. It no longer names `TodoWrite`, which the CLI build dispatch runs does not enable, or `WaitForMcpServers`, which that build never offers once the worker has waited for its MCP servers before sending the prompt.
- A dispatch run's record and its live `result` event carry the cost in USD
  that the agent reports for the run, instead of an empty `cost_usd`.
- The dispatcher's run list shows how long ago a finished run was fired and
  lists finished runs newest first. A finished run's record keeps the time it
  was accepted and, for a failed or cancelled run, the prompt it was given.
- A dispatched run no longer loses a subagent's report: the dispatch tool policy lets a declared subagent call `SubagentHandback`, which the permission hook used to refuse as outside its `tools:` list.
- A trigger's `action.surface_tools` that is not a list of tool names is refused
  when the triggers file loads, as `action.surface_prompt` already is. Before, it
  loaded, and every fire of that trigger was refused by the worker.
- The dispatch worker's session transcripts are written to its agent-data volume, so recreating the worker no longer deletes them.
- A trigger in `triggers.yml` whose entry, `action`, `on_error` or
  `source_config` is not a mapping is refused when the file loads, with an
  error naming the trigger and the field. Before, the dispatcher failed to
  start with an `AttributeError`, or `osprey build` stopped with a traceback. A
  blank `source_config:` means an empty one.
- Editing the trigger file a profile's `dispatch.triggers` names now marks the build out of date, wherever beside the profile that file lives. The drift check used to fold only a file literally named `triggers.yml` at the profile root, so a profile naming `triggers/site.yml` could start a dispatcher on superseded triggers while reporting itself in sync. A bundled trigger file is never folded, so upgrading OSPREY does not mark a build out of date, and a `triggers.yml` that no `dispatch:` block names no longer does either.
- A failed dispatch run's recorded stderr no longer opens with the agent CLI's
  notice that it ignored the project's allow rules. Unattended runs keep those
  rules off on purpose; the trigger's allowed tools and the policy hook decide
  what a run may do.
- In the `control-assistant` preset a dispatched job could not reach the control
  system or the plan queue: its worker ran on the container network, where the
  `localhost` addresses it was given name the container itself. The worker now
  runs on the host network, like the web terminals. A worker kept on the
  container network is given the Bluesky bridge's in-network address.
- The web terminal opens on the panel a profile pins with `default_panel:` even
  when that panel is one the terminal starts and health-polls, such as
  JupyterLab. Before, another panel took the slot while the default's first
  health probe was still in flight.
- A web-terminal layout stored for a user comes back whole on the next login:
  the tile in front is filled by its own panel, and the profile's default panel
  takes the workspace only when the stored layout holds no service tile. Before,
  the default claimed the slot as soon as it was healthy, adding itself to the
  restored layout.
- The DOOCS connector gives every ENS lookup, property read and property set a
  bound, `control_system.connector.doocs.timeout_s` (default 5 seconds). A read
  that runs out raises a timeout error, and a set that runs out is reported
  unconfirmed, because the value may still arrive.
- A DOOCS reading served through the connector host no longer fails to encode:
  its metadata names the value's type as text rather than carrying the Python
  type itself.
- A DOOCS deployment's control-system safety rule names the DOOCS bypass
  routes — `doocs4py.get`, `doocs4py.set` and `doocs4py.subscribe` — instead
  of the generic "no direct hardware library calls" text.
- Importing an OSPREY model provider no longer stalls on an air-gapped host.
  LiteLLM fetched its model map from GitHub at import, so every process that
  loaded a provider waited for that fetch to time out; it now uses the map the
  installed LiteLLM ships, and an operator who set `LITELLM_LOCAL_MODEL_COST_MAP`
  keeps their value. LiteLLM also no longer loads a `.env` found by searching
  upward from the process. OpenAI model ids are routed with an explicit `openai`
  prefix, so a model newer than the installed LiteLLM is no longer refused. The
  web interfaces serve their static assets with the same content types on every
  host instead of reading the host's mime table, which on RHEL served `.js` as
  `application/javascript`.
- A `control_system.target_switch.drain_timeout_s` that is an integer beyond
  float range, NaN, infinity, `true` or negative now falls back to the 5 s
  default with a warning, instead of raising `OverflowError`, waiting forever
  or tearing in-flight work down at once. `0` still tears the old target down
  without waiting.
- `osprey build` and `osprey status` each print one warning naming a
  `claude_code.aliases` or provider `claude_code_aliases` key that is not haiku,
  sonnet or opus, and so is ignored. Before, the build showed nothing and status
  printed raw log lines.
- The channel finder's full-text index no longer looks for a bundled DuckDB
  extension file nobody ships; it downloads the extension (through
  `http_proxy`/`HTTP_PROXY` when set), and a failed download says the host needs
  network access or a proxy.
- The DuckDB copy of a middle-layer channel database fills `units` with the
  unit a field is served in (`HWUnits`, or `PhysicsUnits` when `Units` is
  `Physics`) instead of the MML `Units` mode word; a plain unit under `Units`
  still lands as before.
- Asked to show "the entry" after discussing one of its pictures, the agent now
  opens the entry with that picture enlarged. The agent guidance and the
  `entry_open` description pass the picture under discussion along with the entry
  id, whatever noun the operator used.
- The EPICS connector no longer lets a write to an integer channel change the
  value on its way out. It refuses (`VALIDATION_ERROR`, nothing sent) a fraction,
  a NaN, a non-numeric string or an out-of-range number bound for a `longout`, a
  short, a char, an enum or their arrays; before, a soft IOC stored 1.5 as 1 and
  2**31 as -2147483648, and with confirmation off nothing said so. A whole float
  bound for an integer field is sent as the integer it is. A confirmed text write
  to a char array is `CONFIRMED` when the array holds the text, rather than a
  mismatch against its bytes.
- A `fresh_reads` value that spells neither on nor off now refuses to connect
  instead of silently reading as off.
- An `epics_ca` trigger whose `threshold` or `cool_down_sec` is not a number, is
  not finite, or (for `cool_down_sec`) is negative is skipped with a warning
  naming it, and the other triggers arm. Before, a non-numeric value stopped the
  dispatcher from starting, and `true` or `.nan` was accepted silently.
- The example facility's vacuum gauges and ion-pump pressures read Torr, not Pa;
  ion-pump current reads uA and pump voltage kV; beam current reads mA, RF cavity
  voltage MV, and area dose rates sub-mrem/hr. Each channel's seed values now
  match its unit.
- The Python executor's `execute` and `execute_file` tools reply with the run's
  JSON summary as their text result. A successful run was sent as a serialized
  MCP envelope nested inside that text, which the tools' own declared string
  schema refused, so a standard MCP client rejected every successful run.
- Python execution and visualization sandbox folders are named for their start time
  in the facility zone rather than the host clock.
- A null or otherwise unbounded `python_executor.execution_timeout_seconds` now refuses every run instead of running without a time limit.
- The BLUESKY panel lists withdrawn queue work with times on the facility clock
  from `system.timezone`, and "today" is the facility's calendar day.
- Channel finder feedback times read on the facility clock from
  `system.timezone`. When the browser's own clock reads differently, the Last
  Activity column names the zone.
- The event dispatcher dashboard's timeline marks show their time on the
  facility clock from `system.timezone`, and name the zone when the browser's
  own clock reads differently.
- The artifact gallery reads its card, timeline, preview and print-header times,
  its Today/Yesterday groups and its time-series table and chart axis on the
  facility clock from `system.timezone`. The chart axis and the print header
  always name the zone; the axis previously showed the archiver's UTC digits
  unlabelled.
- The web terminal's session activity, hook log, rail "agent touched" tooltip,
  session picker dates and system-health card read times on the facility clock
  from `system.timezone`, the one the agent quotes. When the browser's own clock
  reads differently, the time or its column names the zone.
- The channel-finder feedback capture now records search answers too large for
  Claude Code's tool-output limit. The hook reads the answer from the file Claude
  Code saved it to, and follows only a file in the session's own `tool-results`
  directory; before, such answers were dropped from the review queue silently.
- The artifact gallery's path chip, copy-path button and drag-to-terminal text now name the directory the deployment's artifact store writes to. A project with a non-default `agent_data.base_dir` was handed a `var/agent_data/artifacts/…` path that did not exist.
- A deployment whose `requires_osprey_version` names a pre-release now gets a
  GitLab pipeline that installs it with `--pre`. pip admits a pre-release only
  for the requirement that names one, and OSPREY's floor on `osprey-connectors`
  names none, so without the flag the validation job could pair a pre-release
  framework with an older stable connectors than the pre-release it was tested
  with. A stable floor renders exactly as before.
- The go-live steps in the facility deployment and control-assistant guides,
  and the matching comment in the `control-assistant` preset, name your own
  control system and its archiver rather than EPICS alone and link the
  connector guide. The deployment guide no longer says the preset ships
  `mock`, and the DOOCS archiver's `avg_window` is documented in seconds, the
  unit it is read in.
- Device stubs seeded from a facility graph are titled `<device id> (<class>)`, and the graph server's example queries q1c, q2 and q6 return each device's `deviceId`, the id q3 takes.
- A pyAML tool run from a helper thread or an asyncio task inside a read-write run
  now runs under that run's lock, where it was refused busy against its own
  process. The lock belongs to the process, and a forked child holds no lock.
- Switching between the chat and the terminal now asks before ending the
  commands the agent started when the process table cannot be read, instead of
  ending commands it could not see without a question. The question says it could
  not check what the agent started; Stop all ends them with the agent, Cancel
  ends nothing.
- `osprey health` reports `system.timezone: UTC` as information, so a deployment on the shipped default no longer exits 1. A zone name no reader can open, or one naming an unset variable, is now an error. Such a value used to show as OK.
- A hierarchical channel database whose `naming_pattern` is placeholders alone
  loads each channel exactly as its parts spell it: a run of underscores inside an
  address is no longer collapsed.
  Building channels from selections resolves each selected tree key to the
  `_channel_part` its node names, so the names `get_options` offers build the
  channel the database holds.
- Inside a deployed container, the hooks looked for agent state and wrote their
  audit records under the build host's project path instead of the path the
  deployment runs at. A dispatched job's control-system write was refused with
  "write state unknown", and hook audit records landed outside the audit
  directory. The hooks now anchor on the repo they run in.
- Hooks now follow a relocated `agent_data.base_dir`: the prompt's gallery focus block is read from and
  cleaned under the configured agent-data root instead of the default one.
  Channel-finder captures land under the configured agent-data root, and the review app reads its
  pending reviews from the same place.
  When no agent-data root is stamped into the session, the write-posture reader looks for the
  control-target record under the configured root too.
- `osprey.utils.identity` is now the same module object as
  `osprey_connectors.identity`, as every other path that moved into the
  connectors package already is, so patching a name through either path reaches
  the identity ladder.
- An `OSPREY_<SERVICE>_IMAGE` override that names another image now keeps that
  service out of every build a deploy makes. The deploy reports
  `<service> runs <image> (<variable>); not built` and starts the image you named,
  instead of rebuilding OSPREY's recipe and tagging it with that name.
- A deployment repository's own `CLAUDE.md`, `CLAUDE.local.md`, `.claude/` and `.mcp.json` no longer reach its container images. The agent in an image runs one directory below the repository root and loaded that `CLAUDE.md` into every session; the render's own files under `build/` are unchanged.
- `mml_export.m` samples each stepped magnet's conversion at its nominal
  and one `DeltaRespMat` above it, so a model imported with
  `osprey facility import mml` changes by the Middle Layer's own amount
  per ampere over a response step and starts each setpoint at the Middle
  Layer's nominal; re-run the exporter and re-import to pick it up.
- `osprey channel-finder preview` prints each in_context row as written, so an address such as `SR:DIAG:BPM:01:POSITION:X` is no longer rendered with an emoji in place of `:X:`, and a bracket in a description is no longer read as markup.
- A write journal the in-process simulator refuses on replay is now logged under
  the address the simulator refused, or under the journal file when the refusal
  names no single channel, instead of under the journal's first address. The
  in-process connector also reads its write journal only when the file changed,
  so reads in a long session no longer re-parse the whole journal each time.
- `osprey init` now seeds the gateway endpoint variable beside the API key when
  the shell exports it. A provider that ships no default endpoint no longer needs
  a hand edit of `.env` before the first `osprey up`.
- A connector bug inside the connector-host child is reported as an internal
  error; it no longer restarts the connector or reads as an unreachable control
  system. A reply the connector-host child sends that cannot be rebuilt fails only
  that call, the same way, and the connection stays up.
- The notebook panel starts only the OSPREY kernel. A request for any other
  kernelspec, including the interpreter's own `python3`, is refused, and a
  notebook that names no kernel gets the OSPREY one, so every kernel runs with
  the write gates and files its records under its own kernel name.
- After a control-target switch, a notebook cell now reads from and writes to the
  new target. Before, the kernel's next cell rebuilt its connector in the same
  process, and Channel Access kept the server it had reached first: the connector
  reported the new target while its reads (and any write) went to the old one.
  On a deployment that can switch, the notebook kernel now takes its connector
  from a connector-host child of its own for each target, so the kernel no longer
  has to be restarted after a switch. (#1518)
- `osprey facility validate` and `osprey facility show` print a knowledge page linked to a missing device once, as a `link-dangling` facility warning, like `osprey build`.
- The KNOWLEDGE rail entry shows its icon again instead of an empty box.
- `osprey knowledge validate` fails when a bundle's `index.md` no longer matches
  the pages beside it, or when a directory with pages has none, and names
  `osprey knowledge regen-index` as the fix.
  The example knowledge bundle's root index lists each directory's titles in the
  same order as that directory's own index.
- `osprey web` stops before launch when a gateway that speaks the Anthropic API
  natively has no API key (`cborg`, `als-apg`, or a `providers.yml` entry with
  `api_protocol: anthropic` and no provider class), naming the variable. Before,
  it warned about a subscription login these gateways do not offer and launched
  into an authentication error. Direct `anthropic` without `ANTHROPIC_API_KEY`
  still launches with a warning, since it offers an interactive login.
- Closing a web terminal or the notebook sidecar now stops its whole process group and names any process that survived, or says the process table could not be read.
- A lifecycle step's Integration Test Results table now prints only when that
  step wrote `check_results.xml`. A later step that writes no results no longer
  repeats an earlier step's table, and the file stays in place for later steps.
- Limits checking in `optional` mode with no `database_path` no longer needs a
  limits database: every channel is written as unlisted, and build and deploy
  no longer refuse the configuration. A configured path must still exist.
- The live Channel Access gate (`scripts/va/live_ca/gate.py`) now gives each
  module a wall-clock limit (300 s by default; `--module-timeout=SECONDS` or
  `OSPREY_LIVE_CA_MODULE_TIMEOUT` overrides it). A module that hangs is killed
  together with every process it started and reported as a failure that names
  it, instead of stalling the gate until the CI job times out. Output is read
  from files, so a server process left running can no longer hold the gate open
  either.
- The multi-user login guide names `osprey users remove` where it named a
  `decommission` command that does not exist, and the seeded-password refusal
  now tells a shared card to delete its stored hash from `.env.auth` and run
  `osprey up` instead of naming that command. The guide now also documents the
  login service's answer to nginx, how a site sign-in that is not OIDC is
  brokered to it, what the roster is, and that logging out at the identity
  provider does not end an OSPREY session.
- A Claude Code managed policy that sets a provider variable to the value the
  deployment already launches with no longer stops `osprey chat`, the Web
  Terminal or the dispatch worker from starting. A policy value that differs,
  or that sets a variable the deployment leaves unset, still refuses the launch.
  The refusal names the policy's value and the deployment's for each variable,
  never prints a credential, and says which side to change.
- A child profile that switches off an inherited MCP server with
  `claude_code.servers.<name>.enabled: false` no longer renders that server:
  the build keeps the switch when it persists the profile's `mcp_servers`.
- The middle-layer channel-finder index stops the build with `view-unsupported`
  on a System or Family key its loader would drop: a key beginning with `_`, or a
  top place whose id is `schema`.
- ARIEL no longer switches semantic search off by itself when the database has no embedding table for the configured model. Every search refuses that configuration and names the command that creates the table; the MCP server, the web interface and `osprey ariel status` report it when they open the service. A semantic search model that the deployment's own `text_embedding.models` does not list is refused by validation.
- A terminal whose session transcript is missing starts a new session on its own, instead of waiting for Enter.
- Middle Layer lattice decks always load at periodicity 1; a transport line whose bend angles cancel no longer gets a periodicity guessed from a rounding residue on Python 3.11.
- The shipped `mml_export.m` reads a monitor whose lattice type reads turn by
  turn (`xTurns`, `yTurns`, ...) for one pass on a transport line, one reading
  per device, where it asked 50 turns and refused the family. Its model file's
  orbit reads take one pass the same way.
- `osprey facility import mml` stops with `mapping-invalid` when a `same_as` family is matched by `[sector, device]` to a family whose common names give one row two different devices; the slot no longer takes the first name silently. The drafted mapping proposes `address`, not `same_as`, for the other axis of such a family.
- `facility import mml` no longer writes a wired channel under two device ids, which
  `facility validate` refused. Where one device listed under several families would
  resolve to a different id in each, the import stops with `mapping-invalid`,
  naming the channel, the families, both ids and the fix. `devices:` takes
  `{coordinates: <stem>}`, one device per `[sector, device]` row. `same_as` matches
  by `[sector, device]` and may name a family with more devices, or one identified
  by address or coordinates. The draft proposes one identity for families that list
  the same devices.
- `osprey facility import mml` places each wired family on its deck from the
  Accelerator Objects' `AT.ATIndex` (the wired field's AT block, else the
  family's). A family whose nominal reading the export could not sample is now
  wired.
- The mock archiver connects when only `osprey-connectors` is installed. Its
  connect no longer fails with `No module named 'osprey'`: the simulation
  file it derives from the control-system config is now resolved inside the
  connectors package.
- A mock writes journal that is rejected on replay is logged once and emptied, so the writes that follow are journalled and carry across mock connectors again.
- The model-RPC handler is its own module, free of the Channel Access server, and
  its exactly-once and timeout behaviour is tested on every host. A reply the
  server could not deliver is followed by the timeout answer instead of silence.
- The deploy guard on Bluesky launch tokens counts a render's shell as closed only when its settings deny both `Bash` and `Monitor`. `Monitor` runs shell commands too, so a render that lifted only the `Monitor` deny could read the token and arm a queue with no approval.
- The notebook sidecar's early-exit error carries the process's last stderr lines.
- A knowledge-bundle index summary lists its entries in the same
  case-insensitive order as the index it describes.
- Headless agent runs (dispatch worker, chat view, `osprey query`) now execute
  the Claude Code version the project pins — `claude_code.cli_version`, else the
  installed `claude` — instead of the copy bundled inside the Agent SDK, so the
  terminal, the chat and the dispatcher run one build. **Behaviour change on a
  developer machine:** a headless run uses only an installed `claude` that
  reports the pinned version. When none does, the run is refused before anything
  starts, with one message naming the version asked for, what was found, and the
  two ways out: `npm install -g @anthropic-ai/claude-code@<version>`, or change
  `claude_code.cli_version`. A dispatch worker in that state refuses to start
  rather than accept jobs it would refuse. **`claude_code.cli_version` must now
  be one exact version** such as `2.1.300`: a range, a tag (`latest`, `stable`),
  an alias, a URL, a git address, a file or a path is refused, and surrounding
  whitespace is no longer trimmed. The same check applies to the terminal, to
  headless runs, to the health check and to the image builds. Sessions on newer
  models and on a gateway's own model names also get the task list again: OSPREY
  turns Claude Code's task-list tools on for every process it starts, so all
  three surfaces are offered the same tools whatever model they run.
- The OpenAI provider sends its output-token limit as `max_completion_tokens`,
  the parameter OpenAI's API takes on every chat model, and sends no
  temperature, which OpenAI's reasoning models refuse. Completions and health
  checks on a GPT model that LiteLLM does not yet recognise, such as
  `gpt-6-sol`, no longer fail with HTTP 400 `Use max_completion_tokens instead`,
  and completions on `gpt-5.6-*` no longer fail on temperature 0. Claude Code
  sessions on the `openai` provider send the same request shape through the
  translation proxy.
- The `openai` provider sends a caller's temperature to OpenAI's chat models
  (the `gpt-4o`, `gpt-4.1`, `gpt-4.5`, `gpt-4` and `gpt-3.5` families) and leaves
  it out only for the reasoning models, on direct calls and through the
  translation proxy alike. A deployment naming `gpt-4o` no longer samples at the
  model default whatever temperature was asked for.
- A streamed lifecycle step that overruns its timeout is now silenced before it is killed, so a process it leaves behind can no longer print into the build's output after the step has been reported.
- A panel whose config request fails or does not get an answer at page load no
  longer stays disabled on the rail until the next reload. The web terminal now
  stops each config request after 5 seconds and tries again with backoff, up to
  every 15 seconds, until it gets an answer. A request can be queued behind the
  browser's limit of six connections to one host, so before this change one slow
  load could disable every rail entry at once. A panel page that has not loaded
  after 15 seconds now shows the "Backend not answering" notice on its tile
  instead of a blank rectangle.
- Without dockview, the web terminal's fallback panel host now shows a tile bar
  with the active panel's title and its session items, and a panel iframe may
  hand the operator a file download.
- A `web.panels` id outside `[A-Za-z0-9._-]`, or one that does not start with a
  letter or digit, no longer answers 500 on every request for that panel. The
  terminal refuses to start and names the block, `osprey validate` refuses the
  same id in a profile, and a runtime registration under such an id answers 422.
- `osprey build` now refuses a `web.panels` id with a dot in it when the web
  terminal would refuse the same id at startup, such as `a.b c` written as a
  key under `web.panels:`. The build used to check only the part before the
  first dot, so the deployment built and then failed to start.
- A custom panel whose id has a dot in it, such as `beam.viewer` written as a
  key under `web.panels:`, can now be selected in `web_panels`, and the build
  switches it on or off in its own block. It used to write the switch under
  `beam` instead, so an unselected dotted panel stayed on. A `panels.<id>` key
  written inside a `web:` mapping is now refused, because it renders as a
  literal key under `web` and never as a panel.
- The web terminal serves its shared fonts (`/static/fonts/fonts.css`) to every embedded panel, so a URL-backed panel that links them renders in the hub's typeface.
- A panel link an agent spells without the web terminal's per-user prefix
  (`/panel/ariel/#entry…`, as the ARIEL tools do) now opens that panel in the
  workspace under a multi-user deployment, instead of a new tab at a path the
  deployment does not serve.
- In a multi-user deployment the agent's artifact-focus call no longer fails with
  `HTTP 401`. Saving an artifact used to start the artifact gallery from inside
  the agent's own MCP server process, which consumed that process's panel token
  and left every later panel call unauthenticated; the gallery is now started only
  by a process that can authenticate it, and a refused credential population
  leaves the process the panel token it was handed.
- The facility deployment guide has a new *What the perimeter needs* section
  stating the four limits of the multi-user web tier: its own hostname or
  host:port, one origin, the host network, and at most 100 users. The refusals
  and warnings that enforce them name the limit and link that section, and a
  roster past 100 users no longer suggests moving a port family's base port,
  which never lifted the limit.
- The safety-chain page names `phoebus_drive` as a write path that does not reach the connector: the Phoebus product writes the value through its own connections, so connector limits, the connector's write verdict and read-back do not apply to a drive. The architecture server list now includes the `phoebus` server.
- Multi-user web terminals whose project runs a Phoebus server now refuse the
  implicit `"active"` display and address a handle or a named display instead,
  so two users of one Phoebus product no longer act on each other's focused
  display. Set `phoebus.require_handle: false` to keep `"active"`.
- A build refused because pip cannot locate the workspace members from its OSPREY install spec now
  leaves the project environment untouched; before, the refused build had already replaced it.
- The `orm` and `orbit_bump_sweep` plans now restore each corrector to the setpoint it held before the scan, not to a readback sample.
- Logbook and export images of interactive Plotly plots now show the plot
  instead of a blank page, including images exported through `artifact_export`.
  The artifacts gallery also no longer loads Plotly a second time into a plot
  that already bundles its own copy.
- Phoebus Data Browser files write an explicit time range and annotation times as
  wall-clock in the facility zone, so a timezone-aware time no longer lands hours
  off. The file name is stamped in the facility zone rather than the host clock.
- A connector-host child that exits on its own now reports its own exit code, not 255, when the connector-host pool or the controls server puts it down.
- A `control_system.target_switch.probe_timeout_s` set to an integer beyond
  float range now falls back to the default with a warning, as
  `web.sidecar_ready_timeout_s` already did, instead of raising `OverflowError`
  during a target switch.
- A `control_system.target_switch.probe_interval_s` that is an integer beyond
  float range, infinity or `true` now falls back to the 30 s default, instead
  of failing the controls server's start, never refreshing the reachability
  roster, or probing every second.
- A deployment step stopped at its time limit or by Ctrl-C now waits for every
  process of its group, including one owned by another user, and logs by pid any
  process still running after SIGKILL.
- An idle Simple-view chat is no longer kept open indefinitely while the server cannot read the
  host's process table: it is reaped once idle for twice the idle timeout, and its stop still ends
  only what it can see. Teardown waits about five seconds, not six, for a table that does not answer.
- Ending a web-terminal or chat session no longer signals started processes it could not re-identify when the process table is unreadable, and names them as possibly still running instead of as survivors.
- A build profile refuses an environment-variable name that ends in a newline — in `env.required`,
  `env.pinned`, `services.<name>.env`, `dispatch.env`, `bind_env`, `deploy.registry.token_env_var`,
  `va_archiver.password_env` and the `bluesky.external` key variables — as every other config
  surface already did. The `deploy` block's refusal no longer says a name must be uppercase.
- A profile's `config:` block that states one key both dotted and nested is now checked against the value the build renders (the one written last), so a `simulation.tick_s` or `simulation.default_scenarios` the render carries can no longer slip past its check.
- A key `osprey init --set` adds to a block of the profile now renders inside
  that block's section, above the comment that introduces the next block.
- The project-image how-to and the rendered `Dockerfile` header now run every by-hand line from the deployment repo root, naming the rendered container repo as the build context. The build-args and Apple Silicon examples, which named no Dockerfile, now build.
- `osprey up` builds the project image whenever a deployed service runs it — the dispatch worker, the ARIEL sync service or the record archive — instead of only when the dispatch worker is deployed, so a deployment without the worker no longer tries to pull `<project>:local` from a registry.
- A process started without a config file now anchors agent data and the audit
  ledger on the deployment repo it stands in, found by walking up to
  `profile.yml` as the hooks already do, instead of on its working directory.
- Creating a built project's virtual environment on a heavily loaded host is no
  longer abandoned after one minute: the step waits up to five minutes, a bound
  only a hung uv or interpreter reaches.
- providers: a registered provider class that does not subclass BaseProvider is read as declaring no capability flags, so it needs neither a key nor an endpoint, instead of failing every flag check.
- `exclude_providers` in an application registry now removes the provider from
  the registry that model calls resolve through, so a config that names an
  excluded provider fails with `Unknown provider` instead of reaching it, and
  `osprey registry` shows what the runtime can use. An exclusion that names no
  built-in provider is reported with a warning.
- The translation proxy for OpenAI-protocol providers forwards the headers set in `ANTHROPIC_CUSTOM_HEADERS`, such as a corporate trace header, as the Anthropic route already did. It forwards no other client header.
- Web terminals and the login service now get each proxy setting that has a value as both
  `HTTPS_PROXY` and `https_proxy` (and likewise for `HTTP_PROXY` and `NO_PROXY`), so tools that
  read only the lowercase name, such as curl, use the site proxy too. A setting with no value is
  no longer written as an empty variable.
- A panel served through the web terminal at a path that starts with `/panel/`
  no longer has its asset and API links rewritten twice into
  `/panel/<id>/panel/<id>/...`.
- The translation proxy now asks OpenAI-protocol servers for token usage on streamed answers and passes it on, so agents running on open models report real input and output token counts instead of zero.
- The control-target popover now reads "✗ no answer" for a switch request nothing answered before its deadline, instead of the raw code `request_expired`.
- A failed write in the web terminal's Memory Gallery and prompt galleries names
  the action once (`Save failed (HTTP 500)`) instead of repeating it
  (`Save failed: Save failed (HTTP 500)`).
- The scaffold gallery refuses writes from an app that never resolved
  `web.scaffold_gallery.write_enabled`, refuses to delete an owned artifact as an
  orphan (release it with `delete_file` instead), deletes the file an ownership
  record actually names when releasing with `delete_file`, and says so when a
  release cannot restore the framework copy.
- A chat prompt sent the moment the previous answer finishes no longer fails
  with `409 turn_in_progress`: the server frees the turn before it sends the
  turn's final event.
- The web terminal's control roster names the identity the terminal claimed the
  control context as, so its owner row and the record agree on the port even
  when `OSPREY_WEB_PORT` changes after startup.
- The Memory Gallery lists only the memory files it can open, so a dot-prefixed
  file no longer appears as an entry that cannot be opened or deleted, and a
  filename holding consecutive dots (`notes..v2.md`) is accepted. The Config
  panel's routes refuse when the app never resolved `web.config_panel.enabled`,
  instead of serving, and both the panels payload and ARIEL's capabilities
  report the panel as disabled for the same unresolved flag.
- A pyAML observable without a plane, or with an unknown plane, no longer ends a `pyaml_measure` run in a traceback: the result is refused by the observable's name, and the measurement file the run saved is named.
- `pyaml_measure`'s `moved` now lists every setpoint whose write may have gone out: a refusal raised partway through a batch, an interrupt while the record is filed, or an exception that refuses attributes no longer drops a channel that was written.
- Switching the pyat-specialist agent on by hand no longer renders it when no served model has a deck; the build says it needs one.
- The `pyat-specialist` agent names each lattice deck and `variables.json` by the
  path `mcp__python__execute` resolves, relative to the deployment root, so it can
  load its deck. The artifacts rule leaves `artifact_focus` to the session that
  answers the user; a subagent hands back the artifact id instead.
- The ARIEL sync daemon and the dispatch workers now reach the qmd search
  sidecars by their compose service names on the compose network, rather than at
  the host's published port, which from inside those containers is their own
  loopback. The dispatch workers are also given the ARIEL store's address, as the
  sync daemon already was, so their agent's ARIEL tools reach the store.
- The `queue_list` tool's description now names every key its `status`
  carries, including `runs_removed` and `queue_removals`, which OSPREY adds
  beside the queue manager's own.
- The post-deploy check that nginx is reachable from the host no longer follows
  nginx's redirect. On a TLS deployment, or one whose origin is not the loopback
  address, it no longer reports a healthy stack as unreachable.
- A batched channel read that is cancelled part-way now raises the cancellation instead of returning it as one channel's reading. Reads that fail with an ordinary error are still left out of the result.
- A plan move now settles once its readback is within the motion the facility
  declares for that readback: the drift amplitude plus six times the noise of the
  readback's `simulation` seed. The build writes that band into each settable's
  `settle_tolerance` in `bluesky_devices.yml`, and `bluesky.settle_tolerance`
  stays the floor for every device.

  Seeding the archive no longer rescans every model variable once per sample:
  the pyat engine reads a model's monitor map once and reuses it for that model.
- A readonly run imports the simulation stack on Python 3.14, and a refused
  shared-library load names the library instead of a refused write.
- An idle Simple-view conversation is no longer reaped when the process table
  cannot be read: the reaper keeps it open, with a warning in the log, instead of
  ending commands the agent started that it could not see.
- A refused control-system write ends with one `[osprey-refusal: <reason>]`
  trailer naming why it was refused, so a script or hook can tell the reasons
  apart without parsing the words.
- A registry-mode web-terminal deployment with no `registry.url` is refused by `osprey scaffold web-terminals lint`, `osprey profile validate`, `osprey build` and `osprey up` whether or not it has a render catalog, instead of rendering the unpullable image `/web-terminal:<tag>`. The refusal names `deploy.registry.url` and `config: registry.url`.
- After a page reload, an agent action the operator had not seen keeps its rail
  badge even when the page lands on that panel. The badge clears only when the
  operator or the agent surfaces the panel, so the result no longer depends on
  whether the landing or the badge restore runs first.
- `osprey ariel status` names why the render worker is unavailable (exited at start, no ready line within the bound, invalid ready line) instead of a bare `unavailable`.
- A picture backfill run from the command line no longer ends with
  `RuntimeError: Event loop is closed` on Python 3.11 and 3.12. The picture render
  worker is now closed and reaped before its event loop shuts down, including when
  a render is cancelled.
- `osprey facility validate`'s response check counts a corrector row whose wiring record names no deck element (such as an energy knob) as left out, unwired, instead of stopping the check for every model.
- `osprey init --revalidate` no longer writes the comments that followed the
  retired render catalog inside a one-line flow list before it: a last roster
  entry ending in `access: [roster]` keeps it on one line, and the comments stay
  above the key they introduce.
- The rf-thermal simulation scenario now dumps the stored beam from the CAVITY01 trip at 03:15 until the 04:30 refill, so the beam-current archive agrees with its logbook; it can no longer be active together with vacuum-burst, which also scripts the beam current.
- The build's warning about unrecognized repo-root entries no longer says nothing copies them: it says they stay out of the rendered project and still travel in every container image. The repository's own `CLAUDE.md` and `CLAUDE.local.md` no longer trigger it.
- The Bluesky `get_run_data` tool now bounds an answer by size as well as by
  rows. A row is as wide as the run, so a few hundred rows of a facility-scale
  response matrix came back as megabytes of JSON, more than an agent session
  accepts in one message. A window that would exceed 64 kB now returns the
  rows that fit, with `truncated: true` and a note naming how many rows were
  left out and how to read the rest.
- `osprey scaffold pull` no longer lists or copies a `__pycache__` directory that importing a template module left in the installed template.
- `osprey sim apply` and scenario activation no longer refuse with "the archive
  is configured but MONGO_ROOT_PASSWORD is not in .env" in a project whose
  `archiver.type` is not `mongodb_archiver`: that project reads no stored
  archive, so there is nothing to rewrite and no password to ask for.
- The `seed-invalid` stops for a scenario value outside its limits band or its fault range print the value and the bounds in full, as the nominal stop does.
- Headless agent runs (`osprey audit`, the benchmark runner, the e2e suite) start the Claude Code CLI with `TZ` set to the project's facility timezone, so the agent's calendar matches the deployment's data; a project with no `system.timezone` runs on UTC, as its containers do.
- Headless and SDK-driven agent runs on a LiteLLM gateway keep the headers set in `ANTHROPIC_CUSTOM_HEADERS`. Spend attribution is now added beside them instead of replacing them.
- Executed code can no longer open a secret `.env` file by spelling it in another
  case. On a case-insensitive filesystem (macOS's default) `.ENV`, `.Env.shared`
  or a recased project directory opened the secret file, because the filesystem
  guard compared the name and the secret roots case-sensitively; both are now
  compared casefolded.
- The translation proxy, `osprey web` and `osprey theme-lab` bind their
  listening socket before they name a port, and serve on that socket, so the
  port they report can no longer be taken by another process in between.
- A `network:` or `http:` written for a service in the profile's `config:` block now reaches the build. Before, the build dropped it for any service it writes itself, including the chat bridges, so `services.teams_bridge.network: host` left the bridge on the compose network and the build refused the deployment.
- Naming your own image in `services.<name>.image` now runs that image as named
  for the event dispatcher, the qmd sidecar, the Bluesky bridge and web panel, and
  the Google Chat, Nextcloud and Teams bridges. The service renders without a
  `build:` block, so a deploy no longer rebuilds OSPREY's recipe and tags it with
  your image's name.
- The sidebar machine chip no longer reads "held" for a session that has not started yet.
- The `session_log` tool now compares its `since` and `before` bounds with event
  times as instants, so a bound written with a UTC offset selects exactly the
  events at or after (or at or before) that moment. A bound without an offset is
  read as UTC.
- The web terminal's session picker no longer closes when you scroll its list or
  when new output scrolls the chat, so older sessions can be reached mid-turn.
- The Expert view no longer points a fresh tab at a session key that is not the
  terminal's. The key both views share is written once per page load, by the view
  that owns the session: the id the server confirms in the Expert view, and the
  operator console's own key in the Simple view. A panel opened while the
  terminal is still connecting is told the confirmed id instead of a placeholder
  that is replaced a moment later.
- A settle whose readback cannot be read, or answers no value, raises `SettleReadError` naming the readback; a pyAML settle reports the readback, not the setpoint, as the failed read. `osprey.runtime.settle` with no deployment config gives the config loader's one message.
- `osprey sim apply` left the ARIEL text and image embedding tables missing until a
  manual `osprey ariel migrate`: the purge that clears the previous narrative drops
  them, and a running ingest watcher never recreates them. The seed step now migrates
  again after the purge, so vector and picture search work on the reseeded logbook.
- A logbook entry whose `when.time` is not a valid `HH:MM:SS` time is now refused
  under `'when.time'`, the key the entry has, rather than under the archiver
  event key `'at_time'`.
- A simulation `machine.json` that is not valid JSON is now refused as
  `Machine file <path> is not valid JSON: …` with the decoder's line and column,
  instead of a bare `JSONDecodeError` that named no file.
- `osprey sim status` on the live stand-in target now prints the stand-in's own model log, `var/simulator/standin/<model>.log`, and reports the overlap records from it.
- The feedback report describes the conversation a session is on after `/clear`,
  matching its conversation view, instead of the one it had before.
- In the Simple view, a link in an agent answer to the loopback address of a
  panel server now opens that panel in the workspace. An example is the gallery
  address `http://127.0.0.1:<port>#focus` that the artifact tools give to the
  agent. Before, the link opened a new tab. That tab did not connect when the
  operator used the hub from a different computer, for example through an SSH
  tunnel. A loopback link to a different port still opens a new tab. The
  `/api/panels` response now gives the port of each panel server that uses a
  loopback address (`loopback_ports`). It gives no other data about the server.

  The Simple view no longer removes the files in the attachment tray when it
  reads the same conversation again. This occurs after a feed reconnect, at the
  end of a turn from a different tab, and after a hand-off. Before, the tray
  became empty and the server deleted the files that the operator attached but
  did not send. A new conversation and a change to a different conversation
  still empty the tray and delete those files.
- In the Simple view, the reasoning row now shows while the agent thinks. Before,
  a model that keeps its reasoning hidden sent no visible text while it thought,
  so the row appeared only when the thought ended and closed at once. The chat
  now opens the row when the agent starts to think.
- In the Simple view, a tool result no longer shows the notes that Claude Code
  adds for the agent (`<system-reminder>…`). The agent still receives them. The
  agent is also told to decide on a drawing before it writes its reply. Before,
  it sometimes wrote the reply, and then made tool calls that did nothing (such
  as a read of `/dev/null` or a subagent with nothing to do) so that it could
  add the drawing in a second reply.
- In plan mode the agent can write its plan file again. The memory guard allows
  a Markdown file directly in the plans directory, and only while the session
  is planning, so the approval card shows the plan the operator is asked to
  approve. The chat refuses an exit from plan mode when no plan is on file; the
  operator is not shown a card for it, and the agent writes the plan first.
- The chat no longer shows a dollar sign inside code as math, also while a code
  block is still being written. A wide table in an answer now scrolls sideways
  inside the message. The status line goes back to "Working…" when a tool result
  or a background agent report arrives.
- The Simple view chat now refuses a slash command that it does not list, such
  as `/login`, `/model` or `/config`. The server refuses the prompt before the
  turn starts, and the chat shows that the command is not available. The chat
  lists the project's skills and commands, `/clear` and `/compact`. A prompt
  that starts with a path, such as `/home/x is full`, still goes to the agent.

  The operator's skill row no longer shows the list of attached files as part
  of the skill's arguments.

  When the agent process stops between turns, the chat now shows that the agent
  process ended. The next prompt starts a new agent process that continues the
  same conversation. Before, every prompt failed until the operator started a
  new conversation.
- A tool named `constructor` or `__proto__` no longer drops its row from the
  Simple chat. A run of skills that the operator typed now reads "Ran 2 skills"
  and not "Used 2 skills". A long skill name no longer hides the elapsed time in
  the status line, and a long argument hint no longer pushes the kind label out
  of the slash menu. The artifact link now shows a clear focus ring. When an
  artifact link fails, its tooltip gives the reason: not in the gallery, session
  expired, refused by the server, or gallery not reachable.
- The Simple view no longer stays in the working state after its live feed
  reconnects. A tab that follows a turn from another tab now ends that turn when
  the server says that no turn runs or that no agent runs. The tab then reads
  the conversation again, so the turn shows complete. When the feed misses part
  of a running turn, the tab reads the conversation again and continues the
  turn. A new turn from another tab now shows as a new turn with its prompt.

  A reload during a turn no longer shows that turn as finished. The turn stays
  open, and the live stream continues it in the same reply. A turn that waits
  for its background agents also stays open until they report.

  A reloaded turn that failed with a model API error now shows the error and a
  failed result. Before, it showed as a success.

  When a prompt does not reach the agent, the chat now removes its message and
  puts the text back in the message box. This applies to a refused prompt, a
  server error and a network error. Before, the text was lost.
- In the Simple view, a subagent keeps its activity when the view loads the
  conversation again. Before, a reload or a switch from the Expert view showed
  "Nothing yet." under Activity. Now it shows the tool calls and text that the
  subagent showed live.
- After a page reload, the Simple view shows each tool call's decision again. A
  call you allowed shows "Allowed by you". A call you declined, in either view,
  shows "Denied by you". A call allowed for the whole session also shows
  "Allowed by you", because the transcript does not record that difference. A
  call that a safety check refused shows no decision.
- In the Simple view, the panels now get the session that the chat is on. Before
  this change, after a switch from the Expert view, the panels kept the session
  id that the terminal had before the switch. When the chat then started a new
  conversation, the workspace gallery filtered on the old session and showed no
  artifacts from the new one. The activity log and the feedback form also used
  the old session.
- A single-pass line with an enabled RF cavity now solves: its optics are
  propagated from `twiss_in` through a 4D copy of the line, while the tracked
  pass still sees the cavity.
- The question asked before ending an agent whose started commands are still
  running now names each command by what the agent launched: the command line
  it gave its shell, with the shell's own wrapping removed. A shell loop no
  longer reads as `sleep` or `date` depending on the moment, two different
  loops no longer read alike, and a command reads the same every time it is
  listed. The log line naming what was ended keeps naming each command by its
  file or program name.
- The question asked before ending an agent whose started commands are still
  running lists those commands in one fixed order: earliest start first, then
  process group id. Commands started within the same second could appear in
  either order from one look to the next; the log line naming what was ended
  follows the same order.
- A deployment step stopped at its time limit now says when it could not read the process table, instead of naming the group's id as a process still running.
- Stopping the Channel Access loop or the runtime's connector loop now closes the loop once its thread has ended, and the runtime's connector loop refuses use from a forked child instead of waiting forever.
- On a multi-user deployment each person's panel layouts are now their own: the
  WORKSPACE gallery's split and orientation, the LATTICE sidebar, tab, layout and
  figure order, the KNOWLEDGE sidebar and the BLUESKY plan list width. They were
  one setting per browser, so the last person to change one set it for everyone.
  Layouts saved before this change are not carried over; each person sets theirs
  once.
- A model imported with `osprey facility import mml` serves each magnet of a series string through its own exported conversion, from its own nominal current, by the supply's change from its start, so every magnet changes by the Middle Layer's own amount per ampere; a wiring slice may state its own `curve`. Re-import to pick it up.
- A structured reply that carries a raw control character, such as a newline,
  inside a JSON string now parses on every provider instead of failing the call.
  A reply that still does not parse is asked for once more, with the same
  request, before the error is raised.
- A structured request through a provider without native structured output (the
  prompt fallback, Ollama, AskSage) no longer comes back silently empty when the
  model answers with the schema's shape instead of an instance of it. The request
  now asks for an instance by field name, a reply with a top-level `properties`
  or `type`/`title` the model does not define is asked again, and a second such
  reply raises `StructuredReplySchemaEchoError`.
- A supervised child's argv and environment are rendered outside the supervisor's lock, so a renderer that reads the supervisor or stops it no longer deadlocks the terminal.
- The controls server's target switch now holds a new connector-host child to the
  same checks the connector-host pool applies. A child that reports another
  connector type, or a write posture or readonly run other than the one the switch
  derived, is refused and the previous target stays active; before, a child with
  no gateways, or with a read-only gateway alone, passed whatever posture it came
  up with. A target whose connector block still carries an unresolved `${VAR}`
  placeholder is refused as `target_unresolvable`, including on a return to the
  deployment baseline, instead of passing verification because the placeholder
  matched itself. The switch's child now has a call deadline, so a call that names
  no timeout ends; a child that then answers no ping is replaced on the same
  target, while one that still answers is kept. A virtual-accelerator target whose
  gateway port is unset is derived from the config file the child is handed, not
  from whatever `CONFIG_FILE` the controls server's environment names, so the two
  no longer disagree on the port.

  The connector-host pool's refusal for a child that exits before answering its
  init frame now names the child's real exit code; on Linux it could report 255
  when the child was signalled after it had already exited.
- A TANGO read or write made without a timeout of its own is now bounded by the
  connector's `control_system.connector.tango.timeout_s`, as the device proxy
  beneath it already was; a write that runs out is reported unconfirmed.
- The web terminal's control-target roster, posture toggle and switch gate fill an unset virtual-accelerator gateway port from the config file they read, not from the process's default config.
- A web terminal whose `OSPREY_TERMINAL_USER` falls outside `[A-Za-z0-9._-]` no
  longer answers 500 on every panel request. That name is both the container's
  URL mount and a header on each proxied panel hop, and neither carries a
  character outside that class — nor does the nginx front door route one — so the
  container now refuses to start and names the variable instead.
- A CI container snapshot stops at its first timed-out engine call, runs outside the
  failing test's timeout and says `capture incomplete` when it could not finish; the
  end-of-run thread report leaves idle executor workers out of its count; and a test
  session fails when a test writes into the packaged template tree.
- The render-worker cancellation tests and the video-probe browser tests wait for
  the state they assert instead of a fixed sleep, and the render tests default to a
  ready wait a loaded host cannot trip, so they no longer fail on a busy CI runner.
- A sandboxed artifact page or panel follows the hub's theme change again.
- A tier file that sets `extra_mounts`, `seed_base` or `image` at its top level,
  as `osprey init --revalidate` writes them when it retires the render catalog,
  builds instead of being refused with "Unknown profile key(s)".
- An address the in-process simulator or the mock archiver does not serve is reported as `<address> is not a channel of this facility` (error type `not_found`; write refusal reason `CHANNEL_NOT_FOUND`) instead of an internal error or a control-system access refusal, also when the simulator runs in a connector-host child.
- `osprey users remove` no longer says a user's volumes were removed when it
  cannot update `profile.yml`. The message now names each volume the runtime
  kept, with its reason and the `osprey users prune` command that removes it,
  says the volumes were kept when no `--archive` or `--purge` was given, and
  claims no removal on a re-run that finishes an earlier one.
- Naming your own image in `services.virtual_accelerator.image` (or
  `services.live_standin.image`) now runs that image as named. The service renders
  without a `build:` block, so a deploy no longer rebuilds OSPREY's recipe and
  tags it with your image's name.
- A deployment whose `.env` pins `VA_LATTICE` at a lattice the built tree does
  not carry no longer starts the image builds and then fails to boot: `osprey up`
  refuses up front, naming the file it looked for, `osprey build` repoints the
  retired `builtin` spelling at the lattice the project's own tree serves, and
  `osprey reset` discards the build-derived pointers along with the `build/` tree
  they address.
- `osprey build` reads `data/machine_state_channels.json` as JSON, so every address in it is reconciled against the virtual-accelerator channel manifest whatever order its entries' members are written in, and a list that is not a readable JSON object stops the build naming the file instead of reporting 0 channels checked.
- A virtual-accelerator build that cannot read `simulation/machine.json` or
  `machine_state_channels.json` now only asks for the file to be repaired. It no
  longer suggests removing it, which leaves a tree the build refuses as
  incomplete. The warning for a tree whose every staged channel database is
  unreadable drops the same suggestion.
- The virtual accelerator boots a model whose model-only writable variables
  include a type with no `default_value` field, such as a particle-group
  variable. Such a variable has no boot seed, and `reset` leaves it alone.
- Virtual-accelerator BPM readings now carry the slow drift and noise their
  `machine.json` entry declares, re-served on every telemetry tick, and the
  deploy-time archive seed centres those readings on the orbit the served lattice
  solves, so recorded archive data continues the seeded history without a step.
- A host-port collision on the virtual accelerator's pvAccess port now names
  `services.virtual_accelerator.pva_port`, the key that moves it. It used to name
  the Channel Access port's key, which left the collision in place.
- The virtual accelerator serves a float waveform readback (for example an imported machine's tune vector) on Channel Access and PVAccess; such a channel used to fail every runner pass, so the server published no value at all.
- A virtual-accelerator write of a physics setting now reads back only the
  setting, its readback and what its model computes from its settings, instead
  of the whole served roster, so an apply over every corrector no longer runs
  past `action_timeout_s`.
- `validate_plan` no longer reports the Bluesky bridge unreachable while the
  bridge is still running the dry run it asked for. The request now waits for
  the dry run, its grace and the upload of a pass, so a plan whose dry run takes
  longer than fifteen seconds is reported as it was judged instead of failing
  on the client while the bridge passes and uploads it. A `dry_run_timeout`
  past 300 seconds is refused, by the tool and by the bridge. The plan-authoring
  skill now says where a plan's channels come from: the queue worker's devices,
  not every address the channel finder knows.
- A vendor asset is cached as immutable only on a 200 response; an error or
  redirect on a vendor path is no longer cached. The design-system validator
  skips a panel's `vendor/` directory in its hex colour scan.
- When creating the project venv runs past its five-minute bound, `osprey build` now stops with a build error that names the step and says to check the base interpreter, instead of printing `Unexpected error`.
- An OSPREY source checkout imported from inside another repository's git hook now reports its own version, not the version of the repository running the hook.
- A record file that cannot be opened no longer reads as already open by another process when its path happens to contain the word "lock".
- Reading OSPREY's version in a source checkout no longer writes git's index, so a version probe stopped at its deadline leaves no `index.lock` behind.
- An OSPREY source checkout on a heavily loaded host reports the commit it runs
  rather than the version stamped at its last rebuild: the `git describe` probe
  behind the running version waits up to 30 seconds, a bound only a hung git
  reaches.
- Screenshot, Phoebus snapshot and image channel-read results tell the agent to open the saved PNG with its file-reading tool, instead of naming one coding-agent client's tool.
- `osprey users remove` and `osprey users prune` with `--archive` or `--purge`
  no longer report a volume as removed when the container runtime refused to
  remove it. They finish every other step, name each volume that was kept with
  the runtime's reason, and exit non-zero.
- `osprey facility validate` and `osprey facility show` print a warning from the render — a model the provider does not list, a knowledge corpus that could not become a channel search index — once, as `osprey build` does.
- The SYSTEM panel and the status bar in a containerized web terminal no longer
  report every MCP server declared with a `port` as unreachable. A build whose
  deployment serves web terminals writes `health.auto.mcp.url_key: host_url`
  into each render, the address the agent itself dials from the host's network;
  a profile that sets the key to anything else there is refused.
- Web terminals, their auth sidecar and nginx now run in `system.timezone`, like
  every other deployed service; they ran in UTC. On a deployment whose zone is
  not UTC, the next `osprey up` re-renders `.env.users` and recreates those
  containers.
- Stopping a chat or terminal session now ends the commands its agent started
  even when the process table cannot be read on the first try. The read is
  retried while the agent is alive, and a warning names the session when it
  still cannot look.
- `osprey ariel web` and `osprey channel-finder web` bind the host set in
  `ariel.web.host` / `channel_finder.web.host` when `--host` is not given, as
  `osprey artifacts web` does. `osprey channel-finder --project DIR web` takes
  its host and port from that project's config, not the working directory's.
- A workspace section you collapse now stays collapsed when the agent adds a new
  artifact to it.
- ``setup_inspect`` masks a secret written into a nested block. A key that names
  a secret --- key, token, secret or password --- now masks every literal beneath
  it rather than only a string directly under it, so a credential typed into a
  nested part of ``config.yml`` or ``.mcp.json`` no longer reaches the
  transcript, and an unquoted number under ``password`` is masked too. A
  ``${VAR}`` placeholder still shows which variable the value comes from, and a
  key that merely contains one of those words inside a longer one (``keyword``,
  ``keystore``) is no longer treated as a secret.
- A `--set` edit now wins over a nested mapping above its key, and can no longer arm an app render or open its edit doors.
- A write batch that every channel refused now names each refuser. An unknown
  address in such a batch no longer hides a limits or policy refusal on another
  row, and an address that is not a channel is never described as refused by the
  reference monitor. A refusal that carries no reason names no refuser instead of
  being reported as writes disabled.
- The Claude Code status line no longer crashes when its input is JSON but not an object; it renders the placeholder line, as it does for unreadable input. A config section of the wrong shape (an `ariel:` that is not a mapping, a non-string `control_system.type`) is now read as absent instead of failing later on a missing attribute.
- A comment-preserving config write that failed part-way no longer breaks every later write in the same process with "I/O operation on closed file".

### Security

- `osprey app check` runs an application's ESLint and Vitest without the shell's
  secrets or proxies; only the toolchain install sees the full environment.
- `osprey app check` also keeps `UV_*` and `PIP_*` settings, which can carry index
  credentials and publish tokens, from an application's ESLint and Vitest.
- The artifacts gallery now sanitizes agent-written markdown before rendering it.
  Raw HTML in that markdown used to run as script on the console's origin; math
  still renders.
- A `.env` or `.env.shared` line can no longer change `PATH`, `HOME`, `SHELL` or
  the Python, Node, shell start-up and dynamic-loader variables of any OSPREY
  process. Such lines are ignored with a warning naming them. The same holds for
  the git, npm, uv, pip, compiler, Docker, Compose, Claude Code, Perl, Ruby, Java,
  OpenSSL, glibc, pager, editor, browser and askpass variables, `XDG_CONFIG_*`, `XDG_CACHE_HOME`, `XDG_DATA_HOME`,
  `PS4`, `IFS` and exported bash functions, in any letter case: a `.env` that set
  one of these (e.g. `DOCKER_HOST`) is now ignored with a warning; export it in
  the shell instead.
- The repository-boundary check now follows directory links the way the copy does, so a second link that leaves the repository is refused, and so is a link that leads back into its own tree.
- An optimizer config's `model_file` must lie under the results directory (a run's
  saved model or an upload); any other path is refused `422` when the draft is
  saved and again at start, so a launch-token holder cannot have the service
  unpickle an arbitrary file.
- `osprey up` refuses a login that holds an application's launch token or a write agent route while its shipped settings leave a shell open, and `dangerously_allow_bash` does not waive it.
- A terminal on its user's own Claude login applies the env chain through the same rule as every other launch: a `.env` line never sets a shell-only launch variable.
- Agent-run Python now receives only the environment variables it needs: the
  execution sandboxes pass a fixed allowlist, plus names listed in
  `python_executor.child_env_passthrough`, instead of removing a short list of
  osprey's own tokens. Executed code is also refused when it opens the
  project's `.env` files or a process's `/proc` environment or command line. (#1147)
- The python executor decides whether a read-write `execute`, `execute_file` or
  `pyaml_measure` call was approved before it starts anything. When the tool's
  approval policy asks, the approval hook's record of the call must exist and its
  `approved_target` (and, for `pyaml_measure`, its `approved_view_sha256`) must
  equal the call's; a call approved for another target than the run's is refused
  too. Otherwise nothing runs and the result carries
  `failure_kind: approval_missing`, which reaches the agent as an
  `execution_error` whose `details.kind` is `approval_missing`. The approval hook
  writes those fields into its record only after the prompt is sent, and nothing
  of the record reaches the sandbox. A malformed saved-artifact manifest no
  longer fails the run: entries that name no file are skipped. This check runs
  beside the script on one host.
- `pyaml_measure` refuses a measurement before any setpoint moves when a magnet's
  strength, a hardware conversion, an RF write or a `max_step` it sizes a step
  from is not a finite number, or when a `max_step` is not above zero. Such a
  value used to pass every comparison and hand the run the full requested step.

  `pyaml_measure` saves to the artifact store only the measurement file its run
  left in the run's own execution folder under the script's fixed name. A run that
  names any other path saves nothing; the answer carries `data_file: null` and a
  `data_file_refused` line saying what was named.
- Files an agent's Python run leaves for the artifact gallery are read back as
  untrusted input. A saved artifact is collected only when its manifest entry
  names a plain file sitting directly in the run's `artifacts/` folder, and a
  figure only when it is a plain file of the run's folder or its `figures/`
  folder: a name that climbs out of the folder, an absolute path, a symbolic
  link, a hard link to another file, a file over 512 MiB, and anything past
  1000 files or 2 GiB in one run are skipped with a warning, and the run still
  succeeds with its other artifacts. The bytes are read from the same open file
  the checks were made on, so a file swapped after the check is not read.
- The web terminal now accepts a session id only in canonical lowercase UUID form
  on every surface that takes one. A `mode=resume` connect or a `switch_session`
  message carrying any other id is refused before a terminal is spawned or pooled
  for it. A trailing newline or a non-string id is refused too, and a non-string
  `switch_session` id no longer ends the connection. (#1322)
- The web terminal's vendored DOMPurify sanitizer moves from 3.4.12 to 3.4.16,
  past the `IN_PLACE` advisories GHSA-55q2-fjhq-7xh7 and GHSA-6688-9rhm-gjv2.
  OSPREY never calls DOMPurify with `IN_PLACE` or hooks, so neither path was
  reachable. (#1539)
- The web terminal's server and the dispatch worker now keep the operator
  secret, the event-dispatcher bearer and the worker bearer where code running
  in the same container as the same user cannot read them. Each process takes
  its credentials out of its environment into memory before it serves or spawns
  anything, and on Linux marks itself non-dumpable, so its exec-time environment
  and memory are closed to other processes of its uid. (#1783)
- The web terminal, nginx, login and dispatch-worker containers run with
  `no-new-privileges`, so no process in them can gain a privilege through a
  setuid binary. The deployment guide names the facility's EPICS access security
  as the write boundary that does not depend on OSPREY's own code. (#1784)
- A login's render built with `--set` now keeps every guard its host deployment
  runs, whatever its lists say. Before, `--set hooks=…` dropped the host's library
  guards (approval, limits, the memory guard), and a `--set` of
  `config.claude_code.hooks` dropped the host's facility guards. (#2117)
- The artifact gallery serves every content page (artifact files, rendered
  notebooks and markdown) with `Content-Security-Policy: sandbox allow-scripts
  allow-downloads`, so agent-authored HTML runs in an opaque origin without the
  operator's session; only the gallery shell is exempt. The pages it builds are
  self-contained (offline, the vendor libraries are inlined), rendered markdown
  goes through DOMPurify, preview iframes lose `allow-same-origin`, HTML and
  notebook cards show a summary instead of a live page, and printing such an
  artifact opens it in a new tab for the browser's own print.
- Logging out of the login page now survives a restart or redeploy of the login service. Logged-out sessions are kept as one-way digests under `var/state/sidecar/`, so a cookie copied before the logout stays refused until it would have expired.
- A guarded run no longer follows a symbolic link planted as `run.lock` in the
  shared `var/guarded_run/<target>/` directory. The lock opens that name without
  following a link; a link there raises `GuardedRunDirError` naming the path, the
  run does not start, and the link's target is neither read, written nor created.
- A guarded run opens `run.lock` relative to the opened `var/guarded_run/<target>/`
  directory, without following a link at `guarded_run`, at `<target>` or at the
  file, and only as a regular file with one name. A hard link, a FIFO, a directory
  or any other non-regular file at that name, and a link at either directory, is
  refused with the path named: the run does not start, and nothing is read,
  written, truncated or created through it. A FIFO there no longer blocks the
  open.
- Every built project's `permissions.deny` now also denies `Monitor`, which runs
  shell commands in the background, and `EnterWorktree`, which creates a git
  worktree on disk, in every session whatever the write posture. `osprey build`
  refuses a profile that lifts either with `remove_deny` and gates it with
  nothing else, as it does for `Bash` and `Edit`. `osprey up` refuses an open
  deployment (`auth.method: none`) whose tier or app lifts `Monitor`, as it does for
  `Bash`.
- The executor's readonly import check and the new kernel-side block, which
  refuses a control-system client at import and a Channel Access or PVAccess
  client library at load, read one list of client packages. A test fails when a
  connector imports a client the list does not name, or when a client package
  ships a shared library the block's library list does not match.
- A host web terminal (`osprey web`) now loads the deployment's launch tokens,
  and an MCP server spawned by the agent no longer reads any `*_LAUNCH_TOKEN` or
  other credential back out of the project `.env` chain.
- A channel with numeric limits now checks a string write as the number the EPICS
  client parses it to (C rules: `"0x10"` is 16), instead of letting any string
  `float()` rejects past the range check. A string the client would read
  differently by channel type (`"010"`: 8 or 10), a non-numeral, `NaN`, `±inf`
  and lists are refused. A `max_step` check whose current reading is not a finite
  number now fails closed.
- The web terminal refuses to navigate a panel anywhere but within that panel: a
  path under its own mount, or an `http(s)://` URL on its own server, which opens
  through the mount. An agent's `open` URL and a runtime panel registration URL
  are read the way a browser's URL parser reads them (encoded `..`, `//` and `\`
  included), and another path on this origin, a foreign host, or a `javascript:`,
  `data:`, `vbscript:`, `blob:` or scheme-relative `//host` URL is answered 422;
  the agent's `open_panel` returns the refusal as an error, and the browser
  refuses the same URL again before an iframe moves. A registered panel's URL is stored
  normalised; an empty or bare relative URL is now refused.
- A dock op's `panel_op` frame goes only to the dock client's event stream, so
  another page on the same session cannot answer the op in its place. A stalled
  event stream is dropped after 1024 queued frames instead of growing without
  bound.
- The web terminal serves every panel document an agent could have written into
  an opaque origin: proxied panels and discovered `/panel-static/` bundles answer
  with their own `Content-Security-Policy: sandbox allow-scripts allow-downloads`
  on every response that is not JSON, a raster image, CSS, JavaScript or a font,
  and an upstream policy cannot loosen it. Only a panel's declared entry pages
  are exempt: the new `SidecarPanel.entry_pages` (JupyterLab's application
  pages), each companion server's root page, and a config-declared panel's own
  `path`. The knowledge (OKF) panel renders its markdown through DOMPurify. A
  discovered panel that loads its own assets from the server, or a config-declared
  panel whose pages lie outside its `path`, no longer loads them on a gated
  deployment.
- The web terminal's session picker now escapes each session's id where it
  writes it into the dropdown, as it already did for the preview text, so an id
  containing markup can no longer inject HTML into the page.
- The web terminal's scaffold Preview now sanitises the markdown it renders.
  Agent-writable `.claude/` files (rules, agents, hook docstrings) could carry
  markup such as `<img onerror>` that ran in the terminal page when an operator
  opened the file; the Preview now shares the chat log's DOMPurify path and shows
  plain text when the sanitiser is unavailable.
- A readwrite run now refuses a pvaPy put on a channel opened with
  `pvaccess.CA` with `RAW_CLIENT_WRITE`, as it refuses `caput`: it is the same
  raw Channel Access write. A channel opened on pvAccess keeps its approval and
  limits check. pvaPy `MultiChannel` writes and the in-process `CaIoc` (whose
  records can write real channels through their links) are refused in readwrite
  runs too, and `RpcClient.invoke` is refused wherever p4p's `rpc` is: in every
  readonly run and in an executor run with limits checking on. A readonly run
  also refuses pvaPy's server updates. The approval hook and the pre-execution
  checks now detect pvaPy reads, monitors, typed setters and rpc — the writes
  also through an import assembled at runtime — and the EPICS safety rule names
  pvaPy's prohibited calls.
- A readonly run now also refuses a raw Channel Access put made through the
  `libca` handle that `epicscorelibs.ca.cadef` already holds, through any other
  `ctypes` handle that resolves `ca_array_put`, `ca_array_put_callback` or
  `ca_sg_array_put`, and through every spelling of the `ctypes` loader class.
  Reads through aioca are unaffected.
- A readonly Python run refuses every shared-library load at the loader underneath
  every `ctypes` spelling and at cffi's `FFI.dlopen`, closing `_ctypes.dlopen`,
  cffi and `ctypes.CDLL` references bound before the guard was installed.
- A readonly run's shared-library loaders are refused outright and hold no reference to the real loader.
- The framework release workflow no longer grants write scopes to every job:
  only the PyPI upload mints an OIDC token and only the GitHub-release job
  writes contents, as in the connectors release workflow. Both release
  workflows refuse a tag that `main` cannot reach, and the framework jobs are
  guarded on a tag ref rather than a ref name a branch could carry.
- `osprey up` refuses to start a password-login deployment whose browser address is not this
  machine while any login still accepts a password published in `profile.yml`, such as the
  preset's demo logins. Run `osprey users passwd <user>` for each user it names.
  `osprey scaffold web-terminals lint` reports the same finding as
  `web_terminals.auth_seeded_password`.
- An agent reply can no longer make the browser load a picture or media file
  from another site when the reply is drawn. Pictures, video, audio and tracks
  load only from the page's own origin. A reply can no longer set inline styles
  or a background, and can no longer add a form, input, button, select or
  text area. SVG drawings in a reply now lose any attribute that holds a
  backslash or a `url()` that does not point to a part of the drawing.
- The web terminal's session diagnostics no longer read a JSON Lines file outside
  the agent's transcript directory. A session or subagent id that is not a plain
  file name in that directory, such as one with `../` or an absolute path, now
  reads as a transcript that does not exist.
- The Config panel and the scaffold gallery's write routes stay closed when the deployment's config file exists but cannot be read or parsed, instead of opening at their shipped defaults; the refusal and the settings drawer name the file. A deployment with no config file keeps the shipped defaults.
- The agent loads only the MCP servers the build renders into `.mcp.json`, on the
  web terminal, `osprey chat`, `osprey query`, event dispatch and the operator
  chat. A Claude Code plugin's server or a claude.ai connector no longer reaches
  it; to give the agent a server, declare it under the profile's `mcp_servers:`.
- Connector writes fail closed on unknowns, and record writes are serialized
  across processes. The residual: the serialization does not hold across a macOS
  VM file share.
- **Breaking change:** JupyterLab in the console no longer runs script in
  notebook outputs: `IPython.display.Javascript`, `<script>` in HTML outputs and
  bokeh's in-notebook renderer stop working, and
  `@jupyterlab/javascript-extension` is disabled. Vega and Vega-Lite outputs,
  and so Altair charts, no longer render: `@jupyterlab/vega5-extension` is
  disabled. matplotlib, Plotly, ipywidgets, tables, markdown, math and images
  still render. The policy keeps `'unsafe-eval'`, as the eval check did not
  clear it.
- **Breaking change:** content an agent can write, such as artifacts, gallery
  files and channel-finder output, is no longer served on the console's origin.
  It renders as before in an opaque origin, but can no longer call OSPREY's own
  APIs, so it cannot switch writes on.
- The approval prompt now also guards `channel_write`, `queue_start`, an armed
  `queue_add` and `phoebus_drive`; it proves the prompt ran and was recorded, not
  that a person said yes. The switch record's lock no longer follows symlinks.
  Executor code that names the switch's store needs approval and is refused where
  nothing asks. The residual: that check matches spellings only.

## [2026.9.0b3] - 2026-09-15

Third public beta: fixes and hardening after b2. A plain `uv tool install
osprey-framework` resolves to the last stable release and skips betas; install
this one with

    uv tool install --prerelease allow "osprey-framework==2026.9.0b3"

(the flag is what admits the paired `osprey-connectors` beta; with pip,
`pip install --pre osprey-framework` or the exact pin is enough).

### Added

- A Microsoft Teams bridge joins the Google Chat and Nextcloud Talk ones:
  mention the bot in a team channel or a group chat, or write to it in a 1:1
  chat, and the OSPREY agent answers in the same conversation, with its PNG
  plots inline. `teams_bridge:` in a build profile deploys it, and the adapter
  installs from a new `teams` extra. A Teams bot receives messages only by
  HTTPS POST, so the build also writes an Azure Function relay that checks
  each request's Bot Framework token and puts the activity on a Service Bus
  queue; the bridge pulls that queue, so it opens no port of its own and a
  question asked while it is down waits on the queue until it comes back.
  `TEAMS_CLOUD` picks the endpoints: `commercial` by default, `gcchigh` for a
  GCC High tenant.
- Every image OSPREY builds accepts the site's network settings. A top-level
  `offline` key and an `images.site_ca` / `images.pip_no_proxy` /
  `images.pip_index_url` / `images.pip_extra_index_url` block are passed as
  build arguments to the project image, each web-terminal persona image, the
  web-terminal login service image and the managed service images, so a
  deployment behind a TLS-intercepting proxy, on an internal package index, or
  on an air-gapped host is configured once in `config.yml` instead of in a
  hand-run `docker build`. `osprey up` installs the certificate named by
  `images.site_ca` into the login image, which could previously start with a
  green health check and fail every login at the identity provider's discovery
  fetch. A profile's `deploy:` block also takes a `ci_image_prefix:`, which
  the scaffolded GitLab pipeline puts in front of the base images its own jobs
  run in, for a runner that cannot reach Docker Hub. A deployment that sets
  none of them builds exactly what it built before.
- The Virtual Accelerator answers a model RPC on the pvAccess channel
  `model_rpc`: you can list the model's variables, read the ones it keeps off
  the channel namespace, and compare what it holds against what the control
  system serves. Those model-only variables — for the bundled demo lattice,
  the per-BPM and per-magnet imperfections — are writable through the same RPC
  when the container is started with `VA_MODEL_WRITE_TOKEN` set; a write that
  presents no token, or the wrong one, is refused. The `virtual_accelerator`
  instance publishes its pvAccess port alongside its Channel Access port so
  clients outside the container can reach the RPC.
- The Virtual Accelerator's telemetry cadence and noise level are set from
  configuration. The noise level is
  `control_system.connector.virtual_accelerator.noise_level`, falling through
  to `control_system.connector.mock.noise_level` when unset; it was reachable
  only as the `VA_NOISE_LEVEL` container variable, which an operator editing
  the deployment's configuration never saw, and that variable still works and
  still wins where a deployment exports it. The cadence is
  `VA_POLL_INTERVAL_S` (default 1.0 seconds) beside `VA_NOISE_LEVEL` (default
  0.01) in the deployment's `.env`. The noise level previously could not be
  changed at all — the IOC never read the one it was given. Both are refused
  at boot if they are not a number or are out of range, rather than being
  clamped.
- Caps and timeouts that were fixed in the code are now configuration keys,
  each defaulting to the value it replaced:

  - `ariel.attachments.max_file_mb` (10) — the largest file a logbook entry
    may attach. The refusal names the limit in force. Attachments live in the
    same Postgres as the logbook, so raising it spends database storage.
  - `ariel.enhancement_modules.semantic_processor.max_input_chars` (8000) —
    how much of an entry the semantic processor sends for keywords and a
    summary. An entry longer than the budget is still cut, but the cut is now
    logged naming the entry, so a summary that covers only an opening says so.
  - `artifact_server.max_timeseries_file_mb` (200) — the largest timeseries
    file the artifact gallery will chart or tabulate. Over the cap those two
    views still refuse with a 413 and the file stays downloadable; raising it
    costs resident memory on the machine serving the gallery.
  - `channel_finder.query_max_rows` (500) — the rows the middle-layer
    `run_sql` tool hands back. A truncated answer says which key cut it and at
    what number, so the agent narrows the query rather than presenting a
    partial list as complete.
  - `bluesky.live_max_runs` (50) and `bluesky.live_max_rows_per_run` (10000) —
    how many finished runs stay readable and how many rows of one run are
    stored. Rows past the cap are still counted, so a long run reports its
    true length over a truncated buffer.
  - `bluesky.settle_timeout_s` (5.0 seconds) and `bluesky.settle_tolerance`
    (`1e-9`, an absolute difference) — how long a plan waits for a device to
    settle after a write. The old pair suited only devices whose readback
    echoes the setpoint exactly; a magnet or an insertion-device gap can now
    be given the room it needs. Running out of budget still fails the plan,
    and a value the settle loop cannot use fails the worker's start rather
    than the first plan that writes.
  - `dispatch.max_turns` (25) — how many agentic turns one dispatched run may
    take, beside the two clock budgets. A trigger can name its own
    `max_turns:` in its `action:` block, a whole number of turns, refused when
    the triggers file loads if it is not.
  - `services.qmd.first_index_grace` (3600 seconds) — how long the qmd
    sidecar's health check waits out its first full index build. The sidecar
    does not open its port until the index exists and is non-empty, so a
    corpus that takes longer to index would otherwise report a working
    container as failed.
  - `control_system.connector.<type>.step_read_timeout_s` (2.0 seconds) — how
    long a `max_step` check waits for a channel's present value before a
    write, beside the connector's own `timeout`. Raise it for a slow gateway:
    running out of budget still refuses the write, so it buys room and never a
    weaker check. The Python executor's sandbox applies the same number.

  Two keys that already existed, `python_executor.execution_timeout_seconds`
  (the wall-clock ceiling on one agent Python run) and `web.tour` (who is
  offered the onboarding tour, and how often — `once`, `always`, `never`), are
  now in the configuration reference, printed by `osprey config --defaults`,
  and carried as commented stanzas in every shipped preset. The unused second
  copy of the executor setting is gone; the sandbox reads the value its server
  process started with, so a change takes effect on the next restart.
- The event dispatcher and its workers honour the `env:` passthrough every
  other service already did, declared once as `dispatch.env` and written into
  both halves. A deployment whose channels sit behind a gateway can name
  `EPICS_CA_ADDR_LIST` and `EPICS_CA_NAME_SERVERS` there and have the pair
  reach it; before, the two containers were the only ones the axis could not
  reach.
- A service your deployment brings with it can say that it answers HTTP, with
  `services.<name>.config.http: true`. The deploy summary then prints its
  address as a link you can open instead of a bare `host:port`. Default off,
  so a service speaking a binary protocol is never shown as a link that cannot
  open.
- An external graph store whose corpus lives in a database other than `neo4j`
  can be named with `services.graphdb.database`. The seeder's writes and the
  agent's reads open every session against it, and an attached persona render
  keeps the name its host sets beside the store's address and account, so the
  two halves cannot end up on different databases. The store OSPREY runs for
  itself serves one database and is unaffected. (#910)
- The EPICS Archiver Appliance connector takes an optional
  `archiver.epics_archiver.retrieval_path`, the prefix under which the
  appliance's retrieval servlet is reached. A bare appliance serves it at
  `/retrieval`, which stays the default; a facility whose appliance sits
  behind a reverse proxy that publishes the servlet under another name can
  point the connector at the proxy instead of needing a route to the
  appliance itself. (#931)
- Single sign-on deployments can name the scopes requested at the provider's
  authorization endpoint, with `modules.web_terminals.auth.oidc.scopes`. The
  previous fixed list (`openid profile email`) is still the default, so a
  provider that publishes its identity claim under some other scope no longer
  needs a code change. A list that drops `openid` is refused — without it
  there is no ID token to check — and so is one carrying a blank entry, with
  the offending index named; a blank used to be dropped and the remaining
  scopes shipped as if they were what you had written.
- The sign-in page wears the deployment's theme and names the facility. It
  reads the same `web.theme` the landing page and the terminals behind it use,
  and shows `facility.name` above the OSPREY wordmark, instead of always
  rendering in the framework palette under a bare wordmark. A deployment that
  sets neither is unchanged.
- `web_terminal.shell` accepts a full argv, so a facility whose harness needs
  arguments no longer has to hide them in a wrapper script. Write it as a
  string (`harness --profile ops`, quoting honoured) or as a YAML list
  (`["harness", "--profile", "ops"]`); only the first element is resolved to
  an absolute path and the rest are passed through. The `--shell` flag takes
  the same spellings, a single-command value behaves exactly as before, and
  the PTY still appends its own session and effort flags to whatever is set.
- The five OFL-licensed font families OSPREY bundles ship with their licence
  text: `OFL.txt` sits beside the binaries in every wheel and image, and
  `NOTICE` names the families and points at it. An institutional review of a
  mirrored wheel no longer finds font binaries with no licence anywhere.
- The hello-world preset's example MCP server prints a startup-timing line to
  stderr when it is ready to serve, matching the framework's own servers, so a
  launcher can tell a slow start from a stuck one.

### Changed

- **Breaking change:** the event-dispatch profile key
  `dispatch.pv_strip_prefix` is now `dispatch.channel_strip_prefix` (container
  env `CHANNEL_STRIP_PREFIX`). `dispatch:` is a closed block, so a profile
  that still sets the old name fails the build with a message naming the key
  that replaced it. The whole `dispatch:` block is now documented in the
  profile reference.
- **Breaking change:** the `als-apg` provider no longer ships a built-in
  endpoint. Its gateway is a site's own host, so the URL now comes from
  `api.providers.als-apg.base_url` in `providers.yml` (shipped as
  `${ALS_APG_BASE_URL}`) or from that variable directly; naming a provider
  with no endpoint anywhere is refused instead of falling back on someone
  else's host. Existing deployments that relied on the old default must export
  `ALS_APG_BASE_URL`. The generated `.env.example` now lists the endpoint
  variables that have no default beside the provider API keys, so a deployment
  sees both halves of what `als-apg` needs before its first launch rather than
  after it.
- A release tag is refused unless a full CI run, including the lanes that drive
  a real agent, passed on the tagged tree. The release workflow checks this
  before it builds anything, and `/osprey:release` says how to satisfy it. The
  Teams bridge end-to-end module skips itself when the `teams` extra is absent
  instead of ending collection for the whole run.
- ARIEL's embedding index is now an HNSW index over the pgvector column. An
  existing deployment is moved across by `osprey ariel migrate`, which drops
  the old index and builds the new one; on a large corpus that build is the
  cost of the move.
- `osprey ariel status` reports one enhancement-module table covering every
  registered enhancement and search module — including `qmd_export` and any a
  deployment registered itself — with whether each is enabled and how many
  entries it has completed, failed and left pending, instead of a fixed list
  of five. The counts behind it are aggregated by module key rather than by
  name, and store keys no registered module claims are reported separately,
  under `orphaned_enhancement_modules` in the `--json` payload and as one line
  in the human output, so leftover rows are visible rather than counted into a
  module that no longer exists. `osprey ariel`'s `--adapter`, `--module` and
  `--mode` options accept every name the registry carries, so a module a
  deployment registered itself can be named on the command line. `osprey ariel
  ingest` no longer defaults `--adapter` to `generic_json`: with no flag it
  uses the `ariel.ingestion.adapter` from `config.yml`, the same rule `watch`
  follows, and with neither the run stops and says that key is required.
- An adapter's declared default endpoint and its keyless posture now follow
  from what the provider itself declares, instead of each adapter opting in by
  hand. A provider added to a deployment cannot carry a default endpoint that
  is silently ignored, and a provider that declares it needs no API key sends
  the placeholder the endpoint expects without its adapter having to repeat
  it.
- `osprey status --agents` lists every agent the deployment can run, including
  `facility-knowledge` and `facility-knowledge-graph`, which were missing from
  the table. Both also gain an explicit default model tier instead of falling
  through to `sonnet` unstated.
- Shipped prose describes your deployment rather than someone else's.
  Docstrings, comments and how-to pages that named one laboratory now use the
  placeholder cast — *Example Research Facility* and friends — and a guard
  keeps the tree that way; magnitudes measured on a real corpus stay, because
  they are facts about the software. The bundled demo channel database, the
  demo knowledge graph seeded beside it and the generator that grows both
  describe their quadrupole and sextupole families as belonging to a storage
  ring rather than to one named accelerator, so the CHANNELS panel, the graph
  search and the model no longer read a different facility's machine name out
  of the demo corpus. The bundled pyAT specialist agent's prompt no longer
  names a ring, its lattice-analysis tool's path example is generic, and the
  lattice agent describes itself as computing against the deployment's
  configured lattice model in both its subagent file and the rendered roster.
- OSPREY says out loud that a deployment describes one real machine. The
  refusal for a `live` target that cannot be derived, the unknown-target
  refusal, and the how-tos for deploying a facility and switching the control
  target all name the limit, and a config that carries a second non-simulated
  connector block — one no control target can reach — is now warned about
  instead of passed over in silence. A complex with two real machines runs one
  deployment per machine. A deployment whose control system is not reached
  over Channel Access is also told so by name: the control-target roster says
  switching is not supported by that connector type, and the plan lane says
  the connector type does not execute plans, instead of reporting the machine
  as "not set up". Such a target is still treated as the deployment's real
  machine, with the posture that goes with one.
- `osprey knowledge build-ttl` now orders the corpus by the section order the
  channel database's own tree lists, instead of by a built-in `SR, BR, BTS`
  ring order that only the demo machine has. `--section-order` names the order
  explicitly for a database whose key order carries no meaning; a section
  neither source names sorts after the ones they do, alphabetically. The
  facility token every IRI and identifier embeds now defaults to the project's
  own `facility.prefix` rather than to `demo`, so a project that already names
  its facility does not name it twice. Each run reports the token it minted
  with and the ontology table it emitted against, and warns when a corpus
  built from a database that is not the packaged demo one still carries
  `demo`. **Behaviour change:** a project with `facility.prefix` set that
  regenerated its corpus on the old default mints different IRIs on the next
  `build-ttl` — pass `--facility demo` to keep the old ones.
- Two silent degradations are now refusals. `osprey channel-finder
  build-database` fails, naming the family, when a device family cannot be
  turned into a template; such a family was previously degraded to plain rows
  and the run still reported success, so the database silently lost the
  family's device navigation. The archiver recorder refuses to start when the
  channel manifest lists an address the archive cannot store as a field name —
  one containing `.`, one starting with `$`, or one containing a NUL byte —
  and names the address, where it previously started and warned once per tick
  while archiving nothing for that channel.
- A dispatcher run by hand from a `triggers.yml` that omits
  `max_concurrent_runs` and `max_queue_depth` now starts at 2 concurrent runs
  and a queue of 50 — the pair the documentation describes and the pair a
  built deployment already writes. A deployment built by `osprey build` is
  unaffected; it writes both keys from the profile.
- The web terminal ships no default recipient for feedback mail. A prefilled
  draft can carry a session's scrollback, so the mailbox that receives it is
  now named by the deployment (`web.feedback.email`, or
  `web.feedback.owner.email`) rather than inherited; until one is set the
  dialog offers its issue tracker only, and `osprey feedback list` / `export`
  still records every submission locally. The shipped presets no longer render
  `web.docs_url`, `web.feedback.email` and `web.feedback.github_repo` as live
  keys, so `osprey init` stops writing the OSPREY project's own documentation
  site, mailbox and tracker into your deployment's `profile.yml` as though you
  had chosen them; all three are documented beside the keys as commented
  examples, and the docs and tracker defaults are unchanged. An existing
  `profile.yml` that carries any of those lines is now a difference from its
  preset, so `osprey validate` reports it — delete the line to take the
  default, keep it and mark it `# DEVIATION: <why>`, or run with
  `--drift=warn`. The channel-finder Feedback dialog no longer asks operators
  to type a facility name: the entry is filed against the name the deployment
  already resolved, so one machine cannot end up with two spellings in the
  same store.
- A notebook rendered for the artifact gallery now keeps its remote assets —
  inline LaTeX, interactive widgets and Mermaid diagrams all render — unless
  the deployment sets `offline: true`, which restores the self-contained
  render that needs no network at view time.
- The scaffolded GitLab pipeline and the `ci-extra.yml` header no longer say
  that a job in the facility's own include can override a scaffolded job of
  the same name. The including file is merged last, so the scaffolded
  pipeline's keys win on every key it sets; a facility job needs its own name
  to run as written.
- The `control-assistant` preset's comments now name every connector and
  archiver type OSPREY ships, with commented starting stanzas for `doocs` and
  `tango` beside the existing `mock`, `virtual_accelerator` and `epics`
  blocks, and for `mongodb_archiver` and `doocs_archiver` beside
  `mock_archiver` and `epics_archiver`; each is a commented example, so no
  rendered configuration changes. The `tango` stanza spells the device
  database's conventional port rather than a placeholder, so the line can be
  copied as written. `hello-world` points at `osprey config
  --defaults`, which now names those types too, beside `control_system.type`
  and `archiver.type`, and prints the accepted values beside every key the
  build refuses to guess at — including the five that carried no note:
  `approval.enabled`, `approval.default_policy`, `hooks.debug`,
  `claude_code.telemetry.enabled` and `facility_knowledge.bundle_path`.

### Removed

- **Breaking change:** `osprey set epics_gateway=<facility>` is gone, along
  with the built-in table of named facilities' EPICS gateway hostnames it
  expanded. A gateway address is site infrastructure, so write your own:
  `osprey set
  config.control_system.connector.epics.gateways.read_only.address=gw.example.org`.
  The command now refuses the old spelling with that line rather than writing
  a key the next build rejects.
- A `data/textbooks` directory beside a deployment no longer grants the agent
  a `Read` permission by its name alone. The convention was undeclared and
  could not be revoked from a profile; grant extra read roots through
  `claude_code.permissions.allow`, which is auditable in the profile.
- The ARIEL entries API no longer advertises `sort_order`; ordering was and is
  newest-first.

### Fixed

- `osprey channel-finder benchmark` hands the provider's gateway URL to the
  channel-finder server it spawns, so an `als-apg` project no longer fails
  every in-context query with `Base URL required for als-apg`.
- `osprey up` from a pre-release install no longer fails every image build
  with `No matching distribution found for osprey-connectors`. A beta
  framework exists only beside a beta `osprey-connectors`, which plain `pip`
  never picks for a requirement that names none; every image build now passes
  `OSPREY_PIP_PRE=1` for a pre-release pin and the recipes resolve with
  `--pre`, the same admission `osprey build` already gave its `uv` resolve.
  The `pip` fallback of that venv install does the same.
- Python runs on a pyepics-backed deployment no longer hang after the script
  finishes and get reported as a timeout with no output. The sandbox now
  leaves without running interpreter shutdown hooks, the EPICS connector
  switches pyepics' shutdown hook off, and a sandbox that is killed after its
  script completed is reported from the script's own record, with a note that
  the sandbox did not exit, rather than as a timeout. Readonly runs can read
  through `osprey.runtime` again on such a deployment: the readonly guard
  refused every shared-library load, including the one pyepics makes to reach
  Channel Access; that load is now permitted, and the library's put entry
  points are refused on the handle it gets.
- The deploy env chain reaches every per-user web terminal. A terminal now
  receives `HTTP_PROXY`, `HTTPS_PROXY` and `NO_PROXY` as the login service
  already did — on a proxied site the agent in a terminal could not reach the
  model provider, and nothing said so, because the stack started and the
  health check was green — and the variable naming a provider's gateway
  endpoint crosses into `.env.users` with the provider's key, so terminals
  behind a gateway with no default endpoint stop restarting forever. A deploy
  whose chain sets neither is refused by name even when `.env.users` already
  exists, instead of starting containers that restart forever; `osprey health`
  reports the same gap, and the `.env` seed `osprey up` offers harvests that
  endpoint from your shell alongside the provider's key. The
  lowercase-spelling warning `osprey up` prints now covers any deployment with
  web terminals, not only one using OIDC logins.
- `osprey build` no longer recovers from a failed project-venv install by
  linking the project at the build host's installed packages. An install that
  fails now fails the build and reports the resolver's own output, so a built
  project runs against the dependency set it recorded and nothing else.
- A deployment's service build contexts no longer keep a copy of the site CA
  bundle under `build/services/<name>/` once the images that needed it are
  built, matching what the project, persona and auth-sidecar builds already
  do.
- Provider facts now come from one producer each, so a deployment that
  configures its own endpoints is reported and run as it is configured.

  - A provider that needs an endpoint and has no source for one now says so
    whichever way it is called. Previously only calls through OSPREY's
    completion entry point were refused; a direct call to the provider adapter
    ran on with no endpoint and failed inside the model client with an
    unrelated authentication error. This affects `als-apg`, `amsc-i2` and
    `cborg`, whose endpoint comes from `api.providers.<name>.base_url` or the
    provider's environment variable.
  - A `base_url` left as an unresolved `${VAR}` — the variable is not exported
    — is read as "no endpoint configured" instead of being sent to the HTTP
    client as a hostname, so the failure names the provider and the missing
    value. The channel-finder benchmark's ReAct backend expands the reference,
    from the shell or the deployment's `.env`, rather than dialling it.
  - `osprey health` reports a provider it has no adapter class to probe as
    skipped, including the agent's own, instead of failing the deployment, and
    reports a missing endpoint by name instead of probing another vendor's API
    with the gateway's key. A keyless adapter passes on its own
    `requires_api_key` declaration rather than by being named in a list.
  - `osprey audit` runs its reviewer on the audited deployment's provider and
    default model, refuses a bare profile with no build to resolve one from,
    and on a project whose `config.yml` names no provider says so and names
    the fix instead of exiting on a traceback from inside the reviewer's agent
    loop.
  - A provider registered in an application registry under a built-in name
    replaces that built-in instead of being dropped without a word, a
    `gateway:` key in a provider entry overrides the built-in default for
    spend-attribution headers, and a misspelled `api_protocol` in
    `providers.yml` is refused at load.
  - A build for the `asksage` provider reports a missing credential by name
    instead of failing inside the build: the provider table names
    `ASKSAGE_API_KEY` as the variable the key arrives in, and `.env.example`
    lists it.
  - The `osprey status` remedy and the no-provider error name the keys that
    actually set a provider, and `registry_path` is documented.
- The web terminal's file panel and the artifact gallery keep up with the
  filesystem when the operating system's change notifications go quiet or
  arrive coarse. Each watcher now re-reads the directories it tracks on an
  interval and announces what the notifications did not, within a couple of
  seconds; set the interval with `web.file_watch_reconcile_interval_s`. A
  directory created while notifications were quiet is read in its own right,
  so the panel learns what is inside it rather than only that it appeared. A
  change the operating system reports as a single event about a directory is
  resolved by re-listing that directory one level deep and announcing what
  actually differs, where such a frame used to be forwarded empty or dropped
  outright — on macOS a burst of writes could leave both listings stale until
  the next reload. A directory renamed inside the workspace or the artifacts
  store drops its listing and everything under it, so a long-running
  terminal's memory stays bounded by the tree it is watching; a directory
  removed rather than renamed drops that same subtree, however finely the
  platform reports the removal. And a path the operating system reports as
  bytes no longer stops the watcher with a `TypeError`.
- Writes to a virtual accelerator reach the lattice on a facility that does
  not spell addresses the way the demo does. Which channels are setpoints and
  which are their readbacks is read from each channel's `subfield` in the
  manifest, as the manifest already declared it, instead of from the address
  text, so drive limits, value ranges and physics routing apply however a
  facility spells an address; and the magnets the VA holds a nominal current
  for come from the manifest's lattice-backed partition rather than from an
  address ending in `:CURRENT:SP`, so a facility that spells magnet current
  setpoints any other way gets its baseline. A build whose hierarchical
  channel database is levelled some other way now says so and serves that
  project's channels, rather than refusing with "could not be built … Repair
  the file" against a valid file, or coming up as a silently static machine.
- The address grammar `osprey knowledge build-ttl` reads is now written in one
  place and quoted from there by every message that names it, so a refusal
  cannot describe a shape the parser does not actually read. A channel
  database written on another facility's level grammar is refused once, naming
  the grammar it reads, instead of once per address or with a message about
  the database's naming pattern. When no channel limits file is available and
  the grammar decides directions instead, a run that finds nothing writable
  now says so — a corpus asserting that no channel on the machine can be
  written is usually the wrong setpoint token, not the machine — and the
  agent-facing note about that fallback quotes the same token the generator
  applied.
- `osprey channel-finder preview --db-path` now recognises a middle-layer
  database by its shape — nested groups ending in a `ChannelNames` list —
  instead of by a fixed list of the demo machine's system codes, so a database
  whose top-level groups are named for your own machine previews as one rather
  than being rendered as an in-context file. `osprey channel-finder
  build-database` now honours the CSV's `address` column as the family's
  address pattern whenever it holds a placeholder, so a machine whose
  addresses are not `<family><NN><suffix>` keeps its own shape instead of
  having one synthesised from the family name; rows of one family that give
  different addresses, and patterns naming a placeholder the expander cannot
  fill in, stop the build and name the family. The `instances` column also
  accepts an explicit range (`4-11`) for device numbering that does not start
  at one.
- A first deploy no longer skips seeding the facility knowledge graph. The
  wait for the graph store to accept a connection was shorter than the store's
  own healthcheck grace period, so on a first start the deploy gave up while
  the container was still coming up — and the deployment came up with an empty
  graph and nothing to say why.
- A session can always come home to the deployment's own baseline. Returning
  no longer fails the eligibility checks when that target's connector block
  carries no gateways, no selectable gateway role or no probe channel — the
  shape a baseline on another protocol, or a block still half authored, may
  never fill in. A deployment baselined on the live machine whose block leaves
  `probe_channel` commented out could otherwise rehearse on the simulator once
  and stay there until the controls server restarted. Coming home still
  requires the target's connector block, still refuses a stand-in block
  pointed anywhere but this deployment's own stand-in, and is still proven
  with the baseline's probe channel wherever the block names one. A baseline
  whose block derives no endpoint at all is still refused as not probed while
  a controls server is live.
- Browser-facing links now follow the deployment's own origin — the
  dispatcher's telemetry link, the login URL `osprey web` prints, the CI
  environment URL — and are omitted rather than guessed where no origin is
  derivable. Assumptions about the machine are checked instead of asserted:
  the container runtime, container paths, the home directory, Docker Desktop,
  and the browser and OS the terminal is open in.
- Operator-facing timestamps now carry the facility timezone. The header of
  the notebook written for an executed script shows the facility offset
  instead of a UTC literal, and a Phoebus Data Browser plot opened without an
  explicit time range spans the last 24 hours of facility-local wall clock
  rather than the container's. Execution metadata written by an agent Python
  run carries a UTC offset on its `start_time` and `end_time`, matching every
  other timestamp OSPREY reports and making the records orderable against
  them.
- Logbook entries name the operator the audit ledger names. The session
  metadata attached to an entry resolved the operator from the process
  account, which in a per-user terminal container is a service account rather
  than the person at the keyboard; it now climbs the same identity ladder
  every audit record uses, and falls back to `unknown` rather than to no name
  at all.
- The ARIEL `browse` tool filters by author and source system in the database
  query instead of over the page it just fetched, so a filter on a logbook
  larger than one page no longer silently drops matching entries or reports a
  total that counts the whole table.
- ARIEL reaches the adapters and modules a deployment registered itself,
  whichever way it is entered. The ARIEL panel reads its deployment's
  registry, so a logbook adapter and that deployment's enhancement modules
  stay available to it and publishing or creating an entry through such an
  adapter works from the panel. A process that only searches — the ARIEL MCP
  server, or any caller that builds the search service itself — finds the
  registered search modules and can route to a configured mode, and the
  capabilities report offers the same modes.
- Operator- and agent-facing copy no longer assumes EPICS, one facility's
  device vocabulary, or a particular logbook product. The visualization tools
  describe the packages their sandbox can actually import, the network guard
  and the archiver's empty-window notice name the control system rather than a
  protocol, the ARIEL entry form takes free-text shift and logbook names, and
  `generic_json` ingestion keeps unrecognised top-level fields under
  `metadata` instead of dropping them. The
  `ariel.enhancement_modules.semantic_processor.prompt_template` key is now
  documented as where a facility puts its own extraction vocabulary.
- Web-terminal surfaces read their server or registry authority instead of a
  hand-kept copy. The panel rail draws its own glyph for the Jupyter and
  System Health panels instead of the generic placeholder both fell back to;
  the lattice dashboard renders in the fleet typeface rather than its own; the
  settings drawer offers the policy dropdown for every `approval.tools.*` key
  a deployment states, not just six of them, and no longer shows a
  `screen_capture.enabled` toggle nothing reads; a health check whose name
  contains a dot keeps its full name in the health bar item, with only the
  prefix spelling its category dropped, matching the System Health dashboard;
  and the theme lab's exported proposal names the families that actually ship,
  states per-document what to do about `$extensions.inherits`, and includes
  the optional `family_label` field. The queue bar item shows the queue as
  busy, and leaves Start alone, when the run engine reports a manager state
  the web terminal does not recognise — it used to read any unrecognised state
  as "at rest" and offer Start against a manager that was doing something. The
  BLUESKY panel does the same: Start is not offered, the unfamiliar state is
  named in the button's tooltip, and an item added in that condition says on
  the Plans tab that it waits its turn rather than claiming it runs now. Stop
  stays sendable in every condition and names the state too.
- The artifact gallery explains an artifact it cannot place. A save under a
  category the gallery does not know now names `artifact_server.categories`,
  the config key that registers a facility's own categories — documented in
  the configuration reference and carried as a commented stanza in every
  shipped preset — instead of a source file inside the installed package. An
  artifact saved from raw bytes shows a "Binary" badge, where its type was
  missing from the type registry and the badge rendered with the raw key and
  no colour.
- A `config.yml` render that reaches the web-workspace block without the panel
  registry now stops and says so, instead of writing `panels: {}` and silently
  dropping every tab the profile selected.
- Two pieces of safety copy point where they should. The notice a deployment
  gets when it lists no notices of its own now carries the same advice as the
  copy `osprey init` writes into a facility profile; two sections had been
  added to the scaffolded copy only, so the deployment that made no choice
  read the shorter warning. The writes-disabled refusal now sends you to the
  build profile — "Arm them under `config:` in profile.yml ... then run
  `osprey build`" — instead of to `config.yml`, which every `osprey build`
  regenerates, so the change an operator made there was gone by the time it
  would have taken effect.
- The Nextcloud Talk bridge no longer tells a room that "nothing was changed"
  when a run fails. A run that was approved a write and then timed out reaches
  the same notice, so the claim was a safety assurance the bridge could not
  check.
- The Bluesky bridge's live-document forwarder is now stopped by a control
  message rather than by closing its sockets from another thread, so
  restarting the document plane no longer risks taking the process down, and
  each plane releases its authenticator thread when it stops. A plan dry-run
  neutralises pvAccess addressing as well as Channel Access, and drops every
  inherited `EPICS_CA_*` / `EPICS_PVA_*` variable before setting its own inert
  values, so a validation run can no longer broadcast on the local network
  looking for pvAccess servers.
- The EPICS connector's channel-existence probe now uses the `timeout` the
  deployment configured for it, like its reads and writes already did, instead
  of a fixed 2 seconds. A facility on a slow or distant gateway can raise the
  one key and have every probe follow.
- The MCP readiness budget is now `OSPREY_MCP_READY_TIMEOUT`, and it is
  documented. It gates every run — `osprey query` exits 1 when a declared
  server has not registered in time — not only the end-to-end tests its old
  `OSPREY_E2E_MCP_READY_TIMEOUT` name implied. The old name is still read and
  will be dropped after one release.
- `osprey validate` no longer reports preset drift for a persona's `project`
  and `project_path`. Those two name the render this deployment builds and
  mounts, so a repo checked out under any directory name validates the same
  way.
- `.env.example` and the generated README no longer claim to list every
  variable the deployment reads. The file carries the variables the deployment
  itself supplies — provider keys, whatever the profile declares, and the
  tokens `osprey up` mints — and points at the Environment Variables
  reference, which now gathers the host-level knobs (`CONTAINER_RUNTIME`,
  `OSPREY_OFFLINE`, `OSPREY_CA_BUNDLE`, `REGISTRY_PATH`,
  `OSPREY_TERMINAL_BIND_HOST` and the two image axes) in one place.
- Two teardown races are gone. The approval hook no longer starts an artifact
  gallery server from its own short-lived process when it saves the
  pre-execution review notebook; the notebook still lands in the shared store,
  where a running gallery shows it, and the hook process exits cleanly instead
  of racing the server thread's teardown. And a health check that timed out no
  longer prints an `Event loop is closed` traceback when its abandoned worker
  thread finishes after the loop it started on has gone.

### Security

- The agent's SQL tool now queries the logbook through a SELECT-only Postgres
  role (`<username>_ro`) rather than the role ingestion writes with, so a
  statement naming a server-side function such as `pg_read_file()` is refused
  by the database itself. The role is created when the ARIEL database volume
  is first initialized, and `osprey up` mints its password as
  `ARIEL_DB_READONLY_PASSWORD`. A database created before this release has no
  such role and keeps working on the previous connection — the agent logs one
  warning at start-up, and "Read-only role for the SQL tool" in the standalone
  deployment how-to has a one-shot command for adopting it.

## [2026.9.0b2] - 2026-09-10

Second public beta. A plain `uv tool install osprey-framework` resolves to
the last stable release and skips betas; install this one with

    uv tool install --prerelease allow "osprey-framework==2026.9.0b2"

(the flag is what admits the paired `osprey-connectors` beta; with pip,
`pip install --pre osprey-framework` or the exact pin is enough).

### Added

- The control-context hook records why each run ended in the hook debug log
  when `hooks.debug` is on, so a hook that stayed quiet and a hook that never
  ran no longer look the same from outside.
- A provider entry in `providers.yml` may carry an `extra_body` mapping, sent
  in the request body of every completion. This is how a LiteLLM gateway with
  client-side auth receives a per-user upstream key alongside the gateway key
  in `api_key`. (#930)

### Changed

- The CBORG provider's `sonnet` and `opus` tiers resolve to `claude-sonnet-5`
  and `claude-opus-5`; `haiku` is unchanged.
- CI's eight model-spending lanes run nightly on `main` and on pull requests
  only under the `full-ci` label; the roll-up gate names what skipped and
  prints the label command.
- The DOOCS and Tango connectors run limits validation and the write itself in
  one background thread, so a `max_step` pre-read no longer stalls everything
  else the assistant is doing. Outcomes are unchanged: a write refused by
  validation sends nothing, and a value the device did not take is a failure.
- Switching between the Simple and Expert views no longer shows "Finishing in
  the other view" with a running clock while the other view's agent is idle.
  The clock and "Stop and switch now" appear only when the other view is
  mid-turn or a restart takes longer than expected.
- The web-terminal front proxy image moves from `nginx:1.27-alpine` (end of
  life) to `nginx:1.31-alpine`; `modules.web_terminals.nginx_image` still
  overrides it.
- The release skill's notes-PR step includes the changelog condensation pass
  after the final fold, and its phantom-section and connectors-floor
  instructions match what the changelog gate and the resolvers accept.
- `add_panel_to_rail`, `remove_panel_from_rail` and `register_panel` now ask
  for approval first; `open_panel`, `close_panel` and `arrange_workspace` still
  run without a prompt. `approval.tools.<tool>: skip` in the profile's `config:`
  block silences a prompt, and `skip` is the only policy that does. Six
  workspace tools that carried no policy are auto-allowed: `artifact_pin` and
  the lattice dashboard's `lattice_get_data`, `lattice_get_figure`,
  `lattice_get_settings`, `lattice_update_settings` and
  `lattice_clear_baseline`. The three rail tools and `lattice_clear_baseline`
  are refused outright in a headless read-only run.

### Fixed

- The session-start config drift check no longer goes quiet on a deployment
  that lists extra `control_system.write_tools`. Those tools are approved per
  call and never enter the deny list, which the check read as a hand-edited
  file.
- The facility description lives in the deployment repo at `rules/facility.md`,
  written by `osprey init` and copied into each render, so `rm -rf build` no
  longer discards it. A repo that has it only under `build/` gets it moved into
  place on the next build.
- Hook wiring in `settings.json` follows the profile's `hooks:` list, so a
  dropped hook no longer leaves an entry pointing at a script the build did not
  install. A build that arms writes without the approval, writes-check or
  limits hook is refused, naming the hook and the server it guarded.
- The JUPYTER tab no longer shows a second control-target chip inside the web
  terminal; the header chip is the one switch. A notebook popped out into its
  own window keeps the chip.
- A mediated channel write names the limits check it applies instead of
  picking one at run time; the older-validator fallback could never be taken
  and hid which checks a refusal had made.
- A `readwrite` run checks channel limits on Tango's asynchronous, write-read
  and `AttributeProxy` attribute writes and on caproto's `read_write_read`,
  `Batch` and asyncio `PV` writes, which reached the machine unbounded before.
  The `max_step` pre-read carries an explicit timeout on every caproto client.
- Generated `.claude/settings.json` files no longer make the OSPREY agent print
  two `NotebookEdit(...)` permission-rule warnings at every start; the notebook
  and artifact trees are allowed with `Edit(<path>/**)` rules. Rebuild a
  deployment to pick up the change.
- Per-user web terminals of a graph-mode deployment carry the channel search
  index. Keys that name a file in the render's own data tree survive into
  attached renders, so the CHANNELS tab, the roster and `search_channels` no
  longer answer "No corpus is configured".
- Runtime remedies name `osprey init <dir> --preset <NAME>`; no shipped header,
  skill or message points at a `src/osprey/` path; both safety hooks list every
  tool they gate; setup-mode and demo-ui name the profile as the place to change
  protected keys; the hello-world README and tutorial carry the right hook
  counts; the backup recipe names `.env.auth`.
- The installation page says how to install a pre-release: `uv tool install
  osprey-framework` skips betas, and a version pin alone fails on the matching
  `osprey-connectors` pre-release; `--prerelease allow` is the flag both need.
- The scaffold panel loads a file once when it opens the editor for it, instead
  of racing two requests for the same pane.
- A control target switch made before an agent session's first channel read no
  longer sits on `switching…` for 30 s and then reports `request_expired`. A
  controls server that has not launched its connector yet brings it up on the
  new target, and the header chip stops waiting on servers that hold no
  connector at all.
- `osprey build` from a pre-release install no longer stops in "Preparing the
  project environment". The build pins `osprey-framework` to the running
  version, and uv admitted that beta but not the `osprey-connectors` beta it
  ships with; a pre-release pin now makes the whole resolve admit pre-releases,
  and the recorded `pyproject.toml` carries `[tool.uv] prerelease = "allow"` so
  a later `uv sync` in the built directory resolves the same way.

### Security

- ARIEL logbook ingestion honors the `ariel.ingestion` transport settings on
  every adapter; the generic-JSON, JLab and ORNL fetches went direct and skipped
  a site CA before. An `ariel.ingestion.verify_ssl:` key left without a value
  no longer turns verification off.

## [2026.9.0b1] - 2026-09-08

First public beta.

### Highlights

- A JupyterLab panel in the web terminal. Notebook kernels read and write the
  control system through `osprey.runtime`, under the same write gates as the
  agent's own Python.
- Bluesky plans run from the web terminal: up to two plan lanes, queue
  autostart, and a probe of the channels a plan declares before it runs.
- The simulator's physics is swappable; the shipped ring uses pyAT, and a
  surrogate or another tracking code can replace it while the channel names
  stay the same.
- One control target per deployment, switched from a header chip that labels
  the target (real machine, rehearsal, simulator, demo) and shows its write
  state. Write posture is set per target.
- Channel search answers from an index built over the facility corpus at build
  time, with an Explore view for browsing and an `ask_channels` tool for the
  agent.
- Event dispatch runs agents from triggers on machine events, with an Activity
  and Triggers panel, and bridges reach the deployment from Google Chat and
  Nextcloud Talk.
- First-session onboarding: Simple and Expert views onto one session, an
  optional tour, and a seeded example workspace.
- The header and status bars can be rearranged, feedback goes to the
  deployment owner from inside the terminal, and roster cards can be shared.
- `/osprey:install` builds a facility deployment interactively, confirming
  each step.
- Authentication, per-user roles, origin checks and an audit trail; see
  Security below.

### Breaking changes

- **Breaking change:** a write reports `refused`, `failed`, `confirmed`,
  `mismatch`, `unconfirmed` or `unrequested`, and `channel_limits.json`
  replaces `verification_level` with `confirm: true|false`. (#465, #768)
  Migration: replace `verification_level` and its tolerance fields with
    `confirm` in `channel_limits.json` and re-run `osprey build`, or the check
    that a write's approval matches its target is skipped; a write whose value
    does not land raises on the Python path.
- **Breaking change:** write posture is per control target and deployment
  instance, not per session: `control_system.connector.<type>.writes_enabled`
  falls back to `control_system.writes_enabled`. (#713, #737)
  Migration: the record schema and the hooks changed; run `osprey build` to
    regenerate the deployment before starting it.
- **Breaking change:** derived bluesky plan-device files come from the
  facility's channel roster instead of the channel-limits database, and the
  shipped plans name their read side `readbacks` rather than `detectors`.
  Migration: rebuild, and rename `detectors` to `readbacks` in out-of-tree
    plans.
- **Breaking change:** `ArchiverConnector.get_data` returns long format, a
  `pandas.DataFrame` of `timestamp`, `channel` and `value` with no shared
  index or forward fill, and takes a `processing` keyword.
  Migration: an `ArchiverConnector` subclass returns the long shape and takes
    `processing`; callers of the old wide frame pivot or select by `channel`.
- **Breaking change:** published host ports derive from `deployment.port_base`
  (default `10000`) plus a fixed offset, so a deployment occupies `port_base`
  to `port_base + 999` and the shipped defaults move.
  Migration: `osprey init --set port_base=42000` or `osprey set
    config.deployment.port_base=42000`, then `osprey build`; an explicit port
    wins, and a deployment built earlier keeps its ports.
- **Breaking change:** the facility-config `gitlab:` block is replaced by `ci:
  {provider: "gitlab", ...}`, and a config still carrying `gitlab:` fails with
  a `ConfigurationError` naming the replacement.
- **Breaking change:** the agent's workspace tools are `artifact_*`
  throughout: `artifact_register` replaces `artifact_save`, the `data_*`
  family is gone, and the channel-finder tools are `ask_channels` and
  `run_sql`.
- **Breaking change:** `osprey skills install` is gone. The six bundled agent
  skills ship as the `osprey` plugin from a marketplace at the repository
  root.
  Migration: install the plugin; see 'Agent Skills' under Contributing in the
    docs.
- **Breaking change:** `login: false` on a `modules.web_terminals.users` entry
  is gone; such an entry deploys behind the login wall, and `osprey users
  login-url` applies to `auth.method: token` deployments.
  Migration: replace the key with `access: any` where the card was meant to be
    shared, and rebuild.
- **Breaking change:** the `app_template:` key and its templates are gone,
  presets carry those config keys; a profile's `config:` block may not set
  `api.providers.*`, and an empty `services.graphdb` block is refused.
  Migration: run `osprey profile expand` on a profile that still names a
    template, and delete the block's keys to drop a graph store.
- **Breaking change:** `osprey build --emit-profile` is gone with no alias, so
  a script still passing the flag fails with an unknown-option error.
  Migration: `osprey profile new DIR --preset X` writes what the flag wrote
    plus the preset's `data/` tree, and takes the same `-O` / `--set` layers.

### Added

- ARIEL gains a `hybrid` search mode and `hybrid_search` tool over a markdown
  mirror of the logbook, regenerated by `osprey ariel qmd-resync`;
  `ariel.default_search_mode` picks the mode when a caller names none.
- `ariel.vocabulary.enabled` and `ariel.vocabulary.path` teach ARIEL search
  the control room's acronyms and shorthand, and keyword search takes glob and
  regular-expression tokens under `keyword.settings.patterns_enabled`.
- `template: osprey.ariel_sync` deploys a container that ingests the logbook
  and then polls the source; `osprey ariel sync --watch` runs the same loop by
  hand and `osprey ariel status` reports the last ingestion. (#786)
- Bluesky plans run in a queue server rather than inside the bridge process:
  `queue_add` composes the queue and `queue_start` runs it, so a queue can be
  reviewed before anything moves and survives a bridge restart.
- `bluesky.external:` attaches the plan lane to a facility-run bluesky-
  queueserver (`zmq_control_addr`) and Tiled (`tiled_uri`) instead of
  deploying its own; `parameter_schemas` types external plan parameters.
  (#816, #817)
- A third shipped plan, `orbit_bump_sweep`, drives a closed local orbit bump
  stated in orbit space, solving for the kicks of the correctors you name with
  no lattice model to supply, and restores them on exit.
- The workspace shows what the agent did to it: a tile it focuses or
  rearranges glows, an activity strip names each MCP tool call that changed a
  panel, and a panel changed while you were away keeps a badge.
- Simple view is chat-first, showing the chat alone until the agent shares
  something; `web.ui_mode` picks the view rendered first, with a `?mode=`
  override.
- `agent_session(...)` holds one agent conversation open across several turns,
  so a caller can decide each message from the previous reply, with per-turn
  and cumulative cost tracking and a session-wide budget.
- `connector:` is a top-level build-profile key and the short spelling of
  `config: control_system.type`, so `osprey set connector=doocs` picks one;
  DOOCS also names `archiver.type: doocs_archiver`.
- `podman-compose` works as a container provider beside Docker Compose v2:
  OSPREY detects which is present, records it in `build/config.yml` as
  `container_runtime`, and `down`, `restart` and `reset` act on that runtime.
- The `hello-world` preset ships a working example MCP server: `osprey init`
  seeds it into `mcp_servers/example_server/` and the profile launches it, so
  a first session has a facility tool, `example_status`, to call.
- Chat bridges answer questions in a chat room and return plots and files
  there: add a `nextcloud_bridge:` or `gchat_bridge:` block to a build
  profile, set the bot credentials in `.env`, and `osprey up` starts it.
- A deployment can run a facility knowledge graph: add `graphdb` to
  `deployed_services` with a `services.graphdb` block naming image, ports and
  a Turtle corpus that `osprey knowledge build-ttl` generates.
- `channel_finder.pipeline_mode: graph` answers channel questions from the
  `services.graphdb` store over `index_path`, built by `osprey knowledge
  build-index`; a `facility-knowledge-graph` subagent runs `read_cypher`.
- A `qmd` sidecar indexes the deployment's markdown corpora and answers hybrid
  keyword-and-semantic queries from models in its image; its endpoint has no
  authentication and binds to `deployment.bind_address`.
- A **KNOWLEDGE** panel browses a facility-knowledge bundle read-only from
  `facility_knowledge.bundle_path`; add the panel id `okf` to `web_panels`.
  With the `qmd` sidecar its search ranks hits by relevance.
- A `channel-finder-standalone` preset packages the channel finder and its
  CHANNELS panel as a read-only deployment with a demo database, and the
  control-assistant stack adds standalone ARIEL and `knowledge` cards.
- The browser interfaces theme from one shared token system with light, dark
  and `auto` modes; `web.theme` picks a family (`main`, `high-contrast`,
  `retro`) and `osprey theme-lab` designs a new one.
- A `pyat-specialist` subagent computes lattice and optics quantities over the
  simulated ring. The ring is reached through a pluggable `LUMEModel`, and the
  ALS-U Accumulator Ring ships as `osprey.simulation.lattice`.
- The virtual accelerator's channel manifest is generated from the build's
  data tree, or the graph corpus in graph mode, into `VA_CHANNELS_FILE` and
  `VA_LATTICE`; a build with no readable channel database refuses.
- A virtual accelerator can be deployed with a MongoDB archive and an
  archiver-recorder service that records channels as they move, so questions
  about earlier behaviour are answered from recorded samples.
- A multi-user deployment can require a login:
  `modules.web_terminals.auth.method` takes `password` or `oidc`, and the
  default is `none`; rendering stops without `tls.enabled` unless
  `auth.allow_insecure_http` is set.
- The `control-assistant` preset ships the multi-user web tier: `osprey up`
  stands up nginx with a landing page on `:9080` and two-user terminals whose
  demo passwords in `.env` rotate with `osprey users passwd`.
- `modules.web_terminals.personas.<name>` gives web-terminal users their own
  container image and rendered project, so per-tool permissions come from that
  project's `config.yml`; a roster entry picks one with `persona:`.
- A **Feedback** button in the panel rail sends a report, optionally attaching
  deployment metadata and the session log; `web.feedback.owner` names who
  receives it, read back with `osprey feedback list`. (#828)
- The Expert workspace docks one panel per tile: the rail selects panels,
  dragging one splits the workspace, and `web.presets` layouts apply from the
  Layouts menu. **Customize bars** rearranges both bars (`web.bar_items`).
- `control_target_set` points a running session at the facility gateways
  (`live`), a deployed stand-in (`standin`) or the virtual accelerator (`va`);
  `live` also needs `target_switch.live_gateway_acknowledged`.
- A **JUPYTER** panel serves JupyterLab in a tab, with kernels that read and
  write the control system through `osprey.runtime` under the agent's own
  write gates; notebooks live in `notebooks/` under the agent-data root.
- Expert and Simple views are two windows onto one session: switching hands
  the conversation over with the same transcript, write state and control
  target. An earlier deployment needs the `turn-state` hook and a rebuild.
- A TANGO Controls connector ships in-tree: `control_system.type: tango` reads
  and writes device attributes through PyTango, addressed as
  `domain/family/member/attribute`, with quality as the alarm state.
- The EPICS connector reads pvAccess channels beside Channel Access, routed by
  a `pva_channels` glob, and refuses PVA writes and RPC; oversized reads such
  as camera images come back as gallery artifacts.
- `bluesky.queue_autostart` arms the plan queue so an added plan runs at once,
  and `queue_remove` drops a pending item.
- Before a plan moves anything the bluesky worker probes the channels it
  declares, so a plan naming an unreachable address is refused before its
  first move; `stop_run` and `POST /queue/abort` halt a running scan.
- Multi-user web terminals are rendered from a `users[]` roster by `osprey
  scaffold web-terminals render`; traffic routes through one nginx origin at
  `/u/<user>/`, and `landing_group` sets a persona's landing section.
- The `osprey` plugin adds two agent skills, `/osprey:housekeeping` and
  `/osprey:doc-sync`.
- ARIEL read tools emit an `entry_url` for facility logbook entries, rendered
  from `ariel.entry_url_template`, so the agent links an entry instead of
  composing a plausible dead URL; it stays off until a template is set.
- The `qmd` sidecar can take its model files from a host directory named by
  `services.qmd.models_dir`, mounted read-only, for build hosts with no route
  to the model host.
- The control-assistant workspace gallery is seeded with one example artifact,
  an interactive plot of synthetic data; `artifact_server.example_artifact`
  switches it, and deleting it persists across redeploys.
- A `bluesky-plans` skill tells the OSPREY agent which plans a deployment
  ships and whether each moves the machine, from a build-time snapshot, and
  `get_plan_source` reads a plan's source.
- Channel fields in the BLUESKY panel's plan forms offer typeahead suggestions
  from the project's Channel Finder catalog, tuned by
  `web.channel_suggestions.enabled` and `max_channels`.
- Finished runs can be removed from History one at a time or with a two-step
  Clear, an authored session-tier plan can be retired without a container
  restart, and `bluesky.excluded_plans` hides a plan from the agent.
- `osprey web sessions clear` removes the stored sessions of a stopped single-
  user server, and `osprey profile artifacts` prints the deployment's artifact
  menu grouped by kind.
- `/osprey:install` records places where OSPREY does not express what a
  facility needs in `INTERVIEW.md`, verifies them against the installed
  framework, and offers each as a GitHub issue or a mail, sent on approval.
- `arrange_workspace` sets a whole layout in one call, or applies one named in
  the deployment's config, and `list_panels` reports the open tiles in order
  and how long ago the arrangement changed.
- A health plugin category can share the run's control-system connector: an
  `async def` callable declaring a `runtime` parameter awaits
  `runtime.get_connector()`; `health.plugins` also takes `.py` paths. (#735)
- A generated deployment repo explains itself: `profile.yml` maps the repo's
  zones and carries each hook, rule, skill, agent and panel you could opt into
  as a commented entry, and `osprey init` prints a composition card.
- Build profiles take an `environment:` block declaring the Python environment
  agent code runs in, with `python`, `packages` and `inherit_exclude`; a named
  venv's distributions are frozen into the project.
- A `health:` block in `config.yml` extends `osprey health` with facility
  checks, beside new `reach`, `ariel` and `channel_finder` categories and an
  `archiver_freshness` probe; a `SYSTEM` panel shows them in the browser.
- Deploys run preflight checks before touching a container: a published port
  held by another stack, a missing `modules.shared_disk` path or Docker
  Desktop host networking left off is named before compose runs.
- `osprey init --reset` destroys the containers, volumes and images left by an
  earlier deployment of that name and re-materializes the source zone; `osprey
  up --reuse-stores` adopts volumes a fresh `.env` would break.
- `osprey up` and `osprey status` warn when a project's render is stale, from
  a different osprey version or changed preset content stamped in `.osprey-
  manifest.json`, and name the `osprey build --force` that re-renders it.
- `osprey scaffold ci` writes the deployment's CI pipeline, `scaffold claim`
  freezes a service template for local editing, `scaffold diff` reports drift,
  and `scaffold pull PRESET[:PATH]` copies part of a preset in. (#738)
- A `network:` key in the build profile attaches the dispatch pair and the
  chat bridges to the host network, for facilities whose control system
  answers only there, and `dispatch.worker_port_stride` spaces those workers.
- A trigger's `action.surface_prompt` appends a static awareness fragment to
  the dispatched agent's system prompt, and an optional per-surface tool scope
  narrows its `allowed_tools`.
- `images.registry` and `images.tag` set where OSPREY-built service images
  come from, with a per-service `image` key or `OSPREY_<SVC>_IMAGE` taking
  precedence; `prebuilt_images: true` uses tags already on the host.
- Mock and virtual-accelerator reads put an absolute floor under relative
  noise, and machine files accept `noise_abs`, an absolute sigma, and
  `texture`, slow baseline motion, so channels sitting at zero still move.
- `channel_limits` takes a `name_contains` parameter for literal substring
  search, so channel names holding regex metacharacters are found without
  escaping; it excludes the regex `pattern` parameter.
- An ORM result adds **Response by BPM** and **Singular values** views, and
  the response matrix is drawn diverging about zero against a symmetric
  maximum, so two runs stay comparable and unread cells are hatched.
- The repository ships `CITATION.cff`, a `SECURITY.md` documenting private
  vulnerability reporting, and a `NOTICE` carrying the Berkeley Lab clauses
  that sat in `LICENSE.txt`; licensing terms are unchanged.
- A roster entry's `access:` decides who may open its card: `own` by default,
  `any` for any roster login, or a list of `user:` and `domain:` principals; a
  login not on the list is refused. (#807, #872)
- `modules.web_terminals.tls.port` names the port nginx serves TLS on, so the
  listener need not be 443, and an `OSPREY_SITE_CA` build argument keeps
  builds on verified TLS behind an intercepting proxy. (#794, #803)
- A panel is a directory bundling a themed, token-only HTML entry point and a
  `manifest.json`; one dropped under `<project>/panels/` is served at `/panel-
  static/<id>/` when `web.allow_runtime_panels: true`. (#784)
- A `claude_code.telemetry` block in `config.yml`, off by default, emits the
  agent's operational logs and metrics over OTLP, with a local OpenObserve
  backend as a `deployed_services` add-on.
- Requests to a LiteLLM-fronted provider carry the acting identity for spend
  attribution: `x-litellm-end-user-id` names the roster user, dispatch worker
  or local account and `x-litellm-tags` names the surface.
- Profiles carry artifacts into a build through convention directories
  (`rules/`, `skills/`, `agents/`, `hooks/`), wire their own hooks with
  `config: claude_code.hooks.<Event>`, and subtract inherited ones with
  `exclude:`.
- `osprey validate` refuses each difference between a profile and its preset
  that no `# DEVIATION: <why>` comment claims (`--drift=warn` demotes it), and
  `osprey init --force` re-materializes the source zone. (#864)
- The virtual accelerator serves its model over pvAccess as well as Channel
  Access, and a write on either transport moves both views; in graph mode
  `osprey build` pairs setpoint and readback channels so a write echoes.
- `bluesky.second_lane: true` renders a second plan lane, one per control
  target, and plans route to the lane serving the session's target; the
  BLUESKY panel's lane picker moves plans, queue and results together. (#736)
- `osprey scaffold systemd` writes an `osprey.service` user unit that brings
  the deployment back after a reboot, plus a boot hook for homes on NFS or
  autofs; a `systemd_unit` health category reports on it. (#738)
- Containers open their first terminal session on the control-room prompt: the
  entrypoint seeds first-run state, onboarding, workspace trust and API-key
  approval, into the session volume merge-only.
- An onboarding tour of spotlighted steps drives the live shell and states the
  deployment's own configuration, with invite policy in `web.tour`, a per-user
  `tour:` field, and a retake from the rail or command palette.
- Two optional MCP servers, both off by default: `phoebus` perceives and
  drives a running Phoebus panel with driving approval-gated, and `health`
  serves the health suite through `health_check` and `health_check_full`.

### Changed

- A deployment is a git repo and lifecycle verbs are top-level (`osprey init`,
  `set`, `build`, `up`, `down`, `web`), replacing the `osprey deploy ...`
  forms; each walks up for the repo, or takes `--repo`. (#313)
- A profile directory is the source of truth for a deployment, managed with
  `osprey profile new|expand|validate|presets`; `osprey validate` checks it
  against its preset without building.
- CLI output is a progress report: each verb prints the phase it is in with
  elapsed time and ends on a summary card, container output spools to
  `var/logs/`, and `osprey -v` brings the full transcript back.
- Any LLM gateway can be a provider: name it as `provider:` in `profile.yml`,
  describe it under `api.providers.<name>` in `config.yml`, and put
  `<NAME>_API_KEY` in `.env`; `base_url` overrides a built-in provider. (#725)
- First-party Anthropic, OpenAI and Google provider defaults move to current
  model ids (`claude-haiku-4-5`, `claude-sonnet-5`, `claude-opus-5`,
  `gpt-5.6-sol`, `gemini-3.8-flash`), in the shipped config templates too.
- `model:` accepts any model ID the provider serves rather than only its tier
  map, and `claude_code.default_model` takes a tier name or a declared model
  ID, with anything else an error naming the valid values.
- `archiver_read` derives its bin from the requested range to target
  `archiver.auto_bin_points` (default 10 000) per channel, accepts `now` and
  `2h ago` alongside ISO-8601, and reports coverage for an empty result.
  (#117)
- The `control-assistant` preset defaults to the `virtual_accelerator`
  connector instead of `mock`, so its scans drive the soft-IOC the same stack
  deploys; `osprey set connector=mock` switches back.
- `pymongo` is a core dependency and the `archiver-mongodb` extra is gone;
  drop it from install commands, since pip warns about an unknown extra rather
  than failing.
- The Control Assistant template ships no facility values: the `epics`
  gateways, `probe_channel` and the `epics_archiver` `url` are commented
  placeholders, and the archiver refuses to start without a `url`.
- Importing an `osprey` module no longer loads `.env`; it is read at the CLI,
  at MCP server startup and on the agent launch paths, so library callers use
  `osprey.utils.config.load_project_dotenv()`.
- The PLAN and RESULTS tabs are one **BLUESKY** panel with Plans, Queue and
  Results views sharing the queue state and its two halts; Results leads with
  the run figure and **Export CSV** writes the whole run.
- A Bluesky deployment declares the devices its plans may drive in
  `data/bluesky_devices.yml` (or `bluesky.devices_file`), which `osprey build`
  validates; `list_devices` and `GET /devices` page that namespace. (#766)
- The event dispatcher panel is rebuilt around Activity and Triggers tabs with
  one trigger list and one operator view, and **Clear history** removes
  finished runs from the persisted record and the in-memory list.
- The channel explorer's Explore view is a finder over the graph pipeline:
  search by address, description, device, signal or class, narrow by facet,
  and copy the addresses you select or send them to the prompt.
- Pairing a virtual accelerator with the mock archiver is refused at build, at
  deploy and at MCP server startup, since the mock archiver invents history
  for channels the VA moves; move to the MongoDB archiver.
- The Web Terminal and Artifacts interfaces moved to a flat card design over a
  neutral canvas, the panel tab strip became a vertical icon rail, and the
  gallery header collapsed into one toolbar with a JSON viewer.
- Logging is configured explicitly and goes to stderr: importing Osprey
  installs no handler, so entry points and library callers call
  `osprey.configure_logging()`, leaving stdout for `--json` payloads and MCP
  stdio.
- **Upgrade note:** `modules.web_terminals.auth.method: none` means an open
  deployment with no login page; the posture that value used to select is
  spelled `token`, which is also what an absent `auth` stanza means.
- Custom artifact-gallery categories move from the top-level `categories` key
  into `artifact_server.categories`, which also takes `host`, `port` and
  `auto_launch`; the WORKSPACE tab appears only while that server runs.
- `osprey build` refuses an MCP server name containing `__` or ending in `_`,
  a key at the wrong nesting depth, a provider tier that maps no model, and an
  `api_protocol` other than `anthropic` or `openai`.
- The deployment-build skill `/osprey:build-interview` is `/osprey:install`:
  it inventories what exists, then builds from the `hello-world` preset in
  confirmed phases, keeping decisions in a resumable `INTERVIEW.md`.
- `ariel.database.uri` is optional and derives from the project's
  `services.postgresql` block when unset; an explicit URI is used verbatim and
  no longer rewritten to a container loopback address. (#786)
- ARIEL requires the choices it used to guess: `logbook.composition.provider`
  (falling back to `claude_code.provider`) and `ariel.ingestion.adapter` are
  configuration errors when unset.
- ARIEL search modes are plain strings (`search(mode="keyword")`, the
  `SearchMode` enum is gone), the panel follows the deployment's configuration
  for untouched knobs, and a reranked search shows fast results first.
- Osprey calls a bluesky generator a plan, not a scan, across the panels, tool
  descriptions and docs, and the browser sidecar is the `bluesky-web` service
  (was `bluesky-panels`); rebuild to pick up both renames.
- `osprey build` refuses unknown keys in a profile's `bluesky:` block and a
  superseded `channel_limits.json`, and an unhealthy container's failure
  report carries its last 30 log lines. (#465, #815)
- The status bar starts with a **System health** dot and the clock; any bar
  item can sit in either bar, and the panel tab strip gained show and close
  controls plus a URL field under `web.allow_runtime_panels`. (#847)
- Connector-level names no longer assume EPICS: the write failure code is
  `WRITE_FAILED`, `pv` is `channel`, and a write the control system itself
  denies reports `CONTROL_SYSTEM_REFUSED` instead of an internal error.
- A config that does not name `control_system.type` resolves to the mock
  connector with a warning instead of EPICS, and a missing or blank
  `archiver.type` resolves to the mock archiver.
- Dev mode is a property of the build: `osprey build --dev` bakes the local
  checkout into the images and `osprey up --dev` starts that render; locally
  built images are named `<project>-<service>:local`. (#818)
- `osprey init --set` replaces the value at a key and the `-O/--override` flag
  is gone; `exclude:` gained a `config:` block. (#888)
- `osprey build --force` re-renders a project in place, keeping `.env`,
  `_agent_data/` and `.git`, and records dependencies in a generated
  `pyproject.toml` instead of `requirements.txt`. (#841)
- Config keys settled on one spelling: `execution.execution_method` names the
  backend (`subprocess`; `local` and `container` are deprecated),
  `facility.name` the facility, `agent_data.base_dir` the agent-data
  directory.
- `osprey health` splits cheap `poll` checks from costly `on_demand` ones, so
  a bare run skips live model chats and the pinned-CLI download, which need
  `--full`; a host with no container runtime reports a skip.
- Each compose invocation pins `COMPOSE_PROJECT_NAME` to the resolved project
  name, so a deployment on a shared host owns its own compose project and
  volume namespace; images build in their own step before `up --no-build`.
- The facility ontology is authored as a LinkML schema and compiled with
  `osprey knowledge compile-ontology`; the channel-finder agents read their
  vocabulary from the store and from `facility.ontology`. (#739, #742)
- `osprey channel-finder generate` requires `--source PATH` or `--demo` and
  refuses to overwrite an existing database without `--force`; generated
  `in_context.json` files record their source in `_metadata`.
- The `control-assistant` preset runs the graph channel finder
  (`channel_finder_mode: graph`) and shares its standalone cards (`access:
  any`), and the ARIEL card is renamed `logbook`.
- A docked panel contributes its toolbar controls into the tile header over a
  postMessage contract and hides its own top bar, so panels show one header;
  the terminal and standalone panels share one display menu.
- Config-relative paths (`mirror_path`, `facility_knowledge.bundle_path`,
  `ariel.vocabulary.path`) resolve against the project root with `~` expanded;
  regenerate a legacy logbook mirror with `osprey ariel qmd-resync`.
- The documentation site root serves the latest release; development docs
  moved to `/latest/` and carry a development banner.
- Config-write backups land in `var/agent_data/config-backups/` and `osprey
  users remove --archive` writes to `<repo>/var/web_terminal_archives`; files
  written before this release stay where they are.
- Bridge conversation replay is bounded by a 100k-character budget with a
  100-turn backstop instead of a turn count, and a turn stays eligible for
  replay for 180 days rather than 90.
- `claude_code.telemetry.protocol: grpc` combined with an auto-derived
  OpenObserve endpoint fails the build, since OpenObserve serves HTTP only and
  that pairing dropped metrics and logs silently.
- Virtual Accelerator gateways that declare no `port` follow
  `services.virtual_accelerator.port` instead of a hardcoded `5064`, so moving
  the deployed soft-IOC's port moves the connector with it.
- Subagents that file their answer as an artifact hand the orchestrator a
  pointer — the artifact id, a headline and the identifiers asked for — and
  the orchestrator focuses that artifact in the gallery.
- The web terminal's header carries a search box, a display menu holding
  appearance, view and theme pickers, and a chip whose session menu names the
  signed-in user, the resolved role and its source. (#746)
- The agent's panel tools are `add_panel_to_rail` / `remove_panel_from_rail`
  and `open_panel` / `close_panel`, replacing `show_panel`, `hide_panel` and
  `switch_panel`; opening a panel adds a tile beside the current one.
- Deployments with TLS enabled serve the web terminal over HTTP/2, and the
  session cookie honours `modules.web_terminals.auth.session_lifetime` with a
  matching `Max-Age`, so a session survives closing the browser.
- Plotly, HTML and table artifacts draw in the active theme's chart palette
  and typeface, while a figure that sets its own trace colors keeps them; the
  `high-contrast` family is monochrome, separating by brightness.
- `osprey users env` renders `.env.users` (was `osprey users env-production`
  writing `.env.production`), `osprey up` re-renders one it generated itself,
  and its credential gate runs before any image builds.
- Generated container images bake no `USER`: a root entrypoint re-renders
  drifted agent artifacts, then drops to `osprey` (uid 1000) via `gosu`.
  Kubernetes `runAsNonRoot: true` rejects them; set `runAsUser: 1000`.
- Local model servers that need no API key — Ollama, vLLM and DS4 — deploy and
  launch a web terminal without an auth secret, where a missing secret is a
  note rather than a refusal.
- The framework version derives from the git tag, so a build between releases
  reports its distance from the last one (`2026.8.0.post12+g83fda5e60`) in
  `osprey --version`, the status line and the health payload.
- The event-dispatch worker runs the full project image, so dispatched agents
  see the same facility overlays and `data/` files as the web-terminal agent;
  this takes a dispatch worker image rebuild.
- The `hello-world` preset is the onboarding path: its `profile.yml` leaves
  most keys unset so each arrives as a commented block, and its tutorial runs
  `osprey init`, `osprey build`, then `osprey web`.
- Only a literal boolean `true` arms writes: `writes_enabled: 'true'` and
  `writes_enabled: 1` are refused at both the deployment-wide and per-
  connector keys, and the key is a cold change needing a restart. (#713)
- The control-target chip and popover name machines by what they are — Real
  machine, Rehearsal, Simulator, Demo, or
  `control_system.target_display_names` — and approval prompts name the target
  they would act on.
- Limits checking can be stated per connector type:
  `control_system.connector.<type>.limits_checking` carries `enabled` and
  `allow_unlisted_channels` for that connector alone, and a block states both.
  (#779)
- The `orm` scan plan kicks each corrector either side of where it found it
  and puts it back, `span_a` is the size of that kick bounded by
  `channel_limits.json`, and `orbit_bump_sweep` gained a `leakage_tolerance`
  band.
- The generated safety rules name the tool for each request — `channel_read`,
  `channel_write`, `execute`, Bluesky for a queued scan — and the agent
  answers verify-first, leading with tool-sourced data. (#465)
- The profile's `.env` is where a project's secrets live: `osprey build`
  derives the project `.env` from it alone, environment resolves from
  `.env.shared` then `.env`, and `osprey reset` removes `.env.auth`.
- The `control-assistant` preset ships `readonly`, `readwrite` and `admin`
  tiers beside the `ariel` persona, written as deltas under `personas/`; only
  `admin` keeps the Config panel and the gallery editors.
- The Bluesky tool surface is draft-first:
  `get_draft`/`set_draft`/`clear_draft`, `get_run`, `get_run_data`, and
  `launch_run` taking a `draft_revision`; the arming token is
  `BLUESKY_LAUNCH_TOKEN` with no fallback.

### Removed

- Direct execution of Bluesky plans inside the bridge is removed: `POST
  /runs`, `POST /runs/{id}/launch`, `POST /draft/run` and `POST
  /runs/{id}/stop` answer `410 Gone`.
  Migration: enqueue with `POST /queue/items`, start with `POST /queue/start`,
    halt with `POST /queue/stop` or `POST /queue/abort`.
- The `osprey deploy` and `osprey claude` command groups, the `osprey config`
  subcommands, `osprey profile new` and `profile try`, several `osprey build`
  options, and the interactive menu bare `osprey` launched are gone; `osprey
  profile presets` and `osprey profile validate` are unchanged.
  Migration: `osprey deploy up|down|restart|status|build` are the same verbs
    on `osprey`; `deploy clean|rebuild|nuke` are `osprey reset` or `osprey up
    --build`; `deploy decommission|prune|seed|passwd|render-env-production`
    are `osprey users remove|prune|seed|passwd|env`; `deploy scaffold` is
    `osprey scaffold ci`; `claude regen` is `osprey build`; `claude
    status|chat` are `osprey status|chat`; `config show|export` are `osprey
    config --rendered|--defaults`; `config set-control-system TYPE` is `osprey
    set connector=TYPE`; `config set-epics-gateway --facility NAME` is `osprey
    set epics_gateway=NAME`; `build --tier N` and `build --set K=V` are
    `osprey set tier=N` and `osprey set K=V`; `build PROJECT --preset P` and
    `profile new DIR --preset P` are `osprey init ... --preset P` then `osprey
    build`; `profile try` is `osprey init --preset P --up`.
- The Tuning optimization panel and its companion web server are retired;
  tuning is not a built-in panel.
  Migration: remove `tuning` from `web_panels:` in project configs.
- The `bluesky.demo_runner` build-profile knob and the in-bridge runner it
  switched on are removed; a profile that still sets it fails the build with
  the list of valid `bluesky:` keys.
- The Bluesky stack's HEALTH panel and the sidecar's `/health/full` rollup are
  retired; the SYSTEM tab shows service status for the deployment.
  Migration: remove `web_panels: [health]` entries from project configs.
- The DOOCS connector's `max_points` history-decimation path is removed.
- The `osprey-build-deploy` skill is removed; `osprey scaffold ci` and the
  deploy verbs cover the CI pipeline, deployment files and post-deploy health
  check it used to scaffold by hand.
- The `overlay:` profile key and the `overlays/` seed directory are removed;
  put a file in the convention directory that matches what it is.
- The built project's `.env.template` is replaced by `.env.example`, which
  lists the variables the agent reads and derives its provider API-key list
  from the provider registry.
  Copying `.env.example` to `.env` is safe: it ships no uncommented proxy
    placeholders, which used to export an unparseable `HTTP_PROXY`.
- `osprey build` no longer harvests provider API keys out of the environment
  it runs in, so a build does not depend on the shell that ran it. Exporting a
  key still works for a host-local run.
  Migration: a key reaches a built project through the profile's `.env`; put
    it there and rebuild if a project is missing one.
- Scaffolded projects no longer create `_agent_data/user_memory/`, and `.env`
  no longer carries a `TZ` line detected from the host.
  Migration: the facility timezone is `system.timezone` in `config.yml`.
- The facility-knowledge `capabilities` tool no longer reports
  `write_enabled`; a tool that fails to reach the bundle reports which case
  applies and names its own fix.
- The ALS GTB example corpus (`als_gtb.ttl`) is removed; the generated demo-
  machine corpus is the facility graph OSPREY ships, and both presets seed it.
- python-executor drops the deprecated `epics_writes_enabled` field;
  `control_system_writes_enabled` is the write-gating flag.
  Migration: rename the key in any project config that still sets it.
- The scripted target-switch demo `scripts/demo_target_switch.py` is removed;
  a live stand-in deployed from the build profile replaces it.
- The configuration the loader hands to the runtime drops the OpenWebUI-era
  identity fields (`user_id`, `chat_id`, `session_id`, `thread_id`,
  `session_url`) and the `applications` scoping.
- The tier-2 channel databases and their benchmark query set are retired, and
  tier 2 is not selectable in build profiles.
- `facility-config.yml` is gone: the `modules.web_terminals` stanza lives in
  the repo's built `config.yml`, emitted from the profile's `config:` block,
  and deployment files come from its `deploy:` block.
  Migration: `osprey scaffold web-terminals --config` is an error; pass
    `--repo`, or run from inside the repo.
- The `multi-user-demo`, `multi-user-demo-readonly` and `multi-user-demo-
  readwrite` presets are gone; `control-assistant` ships the same two-persona
  multi-user web tier.
  Migration: build with `--preset control-assistant`; the walkthrough is at
    How-To then Multi-User Support.
- The `render_kernel_templates` service-entry key and the Jupyter kernel
  template rendering behind it are gone; they belonged to a container
  execution backend that did not ship. (#769)
- The `apex` theme family is gone.
- The unused `basePath` iframe query parameter is gone from the web terminal.
- The session-wide posture store that predated per-target postures
  (`var/agent_data/session-postures.json`) is no longer read. (#791)
  Migration: a session narrowing recorded only there does not carry over; set
    the posture again from the terminal after upgrading.
- Configuration keys nothing read are retired from templates, presets and the
  loader:
  `control_system.write_verification.{enabled,fail_on_mismatch,timeout}`,
  `approval.tools.channel_limits`, `control_system.connector.timeout`,
  `connector.mock.simulate_delays`, the `machine_state:` block,
  `channel_finder.explicit_validation_mode`, the channel-finder `benchmark`,
  `processing` and `tree_preview` sub-blocks, `file_paths.{agent_data_dir,user
  _memory_dir,execution_plans_dir,prompts_dir}`, `workspace.base_dir`,
  `api.providers.ollama.{host,port}`,
  `ariel.{reasoning,default_max_results,cache_embeddings}`, the
  `applications:` block, and `system.facility_name`. A `config.yml` that still
  carries them loads unchanged; they have no effect.
  Exceptions: `write_verification.fail_on_mismatch: true` warns once at the
    first write; `file_paths.agent_data_dir` is no longer read, so move a non-
    default value to `agent_data.base_dir`.

### Fixed

- The `osprey` CLI reports a failure as a `✗` line with a cause and a remedy
  instead of a Python traceback: its console script routes through the error
  handler across the CLI's verbs.
- `archiver_read` applies its `processing` parameter (`mean`, `min`, `max`) to
  the query instead of echoing it back; EPICS aggregates server-side and
  MongoDB, DOOCS and mock client-side.
- The EPICS Archiver Appliance connector converts query windows to UTC, so an
  `archiver_read` at a non-UTC `system.timezone` covers the window asked for,
  and it rejects a `bin_size` the appliance has no representation for.
- A headless dispatch run waits up to 90 s for its declared MCP servers to
  connect before the first turn, refuses the run as an `infrastructure` error
  when one fails, and records an `mcp_servers` readiness snapshot.
- `osprey build` copies each service fact — ports, sidecar URLs, per-user
  secrets — from the deployment's render into its personas, so the BLUESKY and
  EVENTS tabs and hybrid logbook search work in a multi-user deployment.
- Image builds get through constrained networks and arm64 hosts: apt uses
  HTTPS mirrors with retries and the proxy settings it was given, Node comes
  from the base image's Debian release, and the login sidecar is pinned.
- A web terminal knows its session id when it opens, so an idle tab resumes
  and reconnects on reload, and `osprey up` repairs the `/data/claude-config`
  volume that left hooks failing with `EACCES`. (#785, #827)
- `max_step` reads a channel's present value through the connector doing the
  write — the simulator's store, DOOCS, TANGO or the configured EPICS client —
  so it applies on control systems other than EPICS Channel Access.
- Enum channel reads carry their state labels in `enum_label` and
  `enum_labels`, and enum plots keep each channel's rungs to itself and break
  the line on a `null` sample instead of drawing a state the channel was not
  in.
- PVA reads through `pva_gateway` with no `port` use the client's UDP default
  rather than TCP 5075, `address` may list several hosts, and the build warns
  when `pva_channels` is set with no `pva_gateway`. (#862)
- The MongoDB archiver returns documents carrying any requested PV and honours
  `precision_ms`, DOOCS `get_data` takes a 60 s default `timeout` and a
  centered `avg_window`, and the mock archiver is seeded reproducibly.
- `artifact_read`'s over-cap preview of an archiver artifact reports channel
  names, point counts and each channel's first and last sample, and plotting
  an artifact whose channels have no samples in the window works.
- HTML-to-image export installs Chromium when a launch reports the binary
  missing, at most once per process, instead of running `playwright install
  chromium` on each conversion.
- Web-terminal panels no longer freeze after a pane resize or a lost drag,
  mount once per view flip, and the panel event stream reconnects and re-syncs
  rail membership after a proxy or backend hiccup. (#638, #817)
- A relative `control_system.limits_checking.database_path` resolves against
  the loaded config's directory, and a load failure refuses with "limits
  database unavailable" rather than as missing channels. (#636)
- Artifacts gallery: `format=table` returns the column list its rows were
  built from, `format=chart` totals are computed server-side, paging builds
  only the requested page, and Simple view renders the types Expert does.
- `artifact_focus` and `artifact_pin` report gallery failures, the panel token
  reaches the terminal agent so panel tools and session hooks work instead of
  returning 401, and artifact stores share one data root.
- A stale `CLAUDE_CODE_USE_BEDROCK` or other backend selector left in the
  operator's shell is scrubbed before launch instead of rerouting the agent
  away from the configured provider. (#356)
- ARIEL keyword search keeps the raw and semantic full-text indexes in sync,
  ingestion keeps an entry whose payload omits its `id`, and `osprey ariel
  purge` clears the migration record so migrate reruns.
- Deploys that include the ARIEL logbook store work on podman-compose hosts,
  and persona containers mount the hosting deployment's logbook mirror, so an
  entry created from a persona is indexed by the sidecar.
- Lattice dashboard summary stats (energy, tunes, chromaticity) recompute
  after a magnet change instead of holding their load-time values.
- Environment-variable names in a build profile (`services.<name>.env`,
  `env.required`, `env.pinned`) may be lowercase, so a deployment can spell
  both cases of `http_proxy`, `https_proxy` and `no_proxy`. (#783)
- Per-lane Bluesky CURVE certificates live under `data/.runtime/`, which the
  build fingerprint skips, so a started deployment does not register as
  drifted; existing key material is relocated, not re-minted. (#716)
- Deployed containers are named `<project>-<service>` and `osprey
  decommission`, `prune` and `nuke` match by compose-project label, so two
  projects can share a host; `osprey up` removes a dropped user's terminal.
  (#850)
- The graphdb service runs `neo4j:5.26-community`, the newest line the
  neosemantics plugin covers, and its healthcheck probes with `wget`, so
  `osprey knowledge seed-graph` works and the container does not sit
  `unhealthy`.
- Host-port preflight covers host-network facility services and each per-user
  panel port, companion panels take per-user port families from the web-server
  registry, and a stale listener no longer blocks a launch. (#327)
- `osprey health --project` and `osprey web --project` resolve the target
  repo's whole env chain, `.env.shared` included, so `${VAR}` placeholders
  expand and the project's own `web_terminal` settings apply.
- Rendered MCP server definitions follow configuration: `phoebus` uses
  `phoebus.host`/`phoebus.port`, and a facility server's `command` resolves
  the `{project_root}` and `{current_python_env}` placeholders. (#829, #748)
- `osprey up` removes the project's own stale containers, including one
  holding a published host port, and recreates nginx rather than hot-reloading
  it, so a redeploy does not serve the previous deploy's configuration.
- A `$` in an env-file value no longer reaches containers truncated: `osprey
  up` refuses such a stack and names the variables across `.env`,
  `.env.shared` and `.env.auth`, and `osprey users passwd` checks first.
- The agent CLI is pinned by the deployment and its background auto-updater is
  switched off, so the terminal shows no red `Auto-update failed` line;
  upgrading the CLI is an image rebuild.
- `osprey build` stops, naming the key, when a render is entitled to a shared
  host directory that is not there, and web-terminal containers under rootless
  podman or docker can write their audit directory. (#770, #774)
- The settings Config tab's form view lists the sections `config.yml` has,
  including `execution`, `archiver`, `logbook` and `facility_knowledge`, in
  place of a `python_execution` section config does not have.
- A target-switch readiness-probe failure names the gateway role, host and
  port it probed, in the message and in structured `details.gateway`. (#718)
- Providers reach their endpoints: ALS-APG and Stanford supply their own
  `base_url`, `argo` points at `apps.inside.anl.gov` and reads
  `ARGO_BASE_URL`, and AskSage falls back to its static model list.
- Python submitted to the executor is refused on an actual call of `eval`,
  `exec`, `__import__` or `compile`, not on the letters `eval(` appearing in
  `retrieval(...)`, `df.eval("a + b")` or `model.eval()`.
- Control-system failure envelopes name the active target: a connect failure,
  timeout, limits refusal or denied write carries its name, label and endpoint
  in `details.active_target` and in the message.
- The virtual-accelerator compose service declares `platform: linux/amd64`,
  overridable with `OSPREY_VA_PLATFORM`, and deploys drop orphan-container and
  platform-mismatch warnings that did not indicate a problem.
- Per-user web terminals on one origin scope their browser-stored UI state —
  mode, theme, dock layout, terminal session, palette history — to each user;
  state stored before this release is not carried over.
- Logins behind a site proxy work: the login service receives the proxy
  variables, an OIDC `email` claim matches `oidc_subject:` case-insensitively,
  and audit records name the account. (#792, #793, #800, #857)
- A deploy on podman served by Docker Compose v2 is reported before the build,
  with the two resolutions: pin podman-compose in `containers.conf`, or deploy
  on docker.
- `osprey build` fails with an actionable error when
  `claude_code.default_model` names a model the provider does not serve, and
  an Anthropic-native provider with a `/v1` base URL resolves to a single
  `/v1`. (#312, #352)
- `web.theme`, `web.app_name` and `web.bar_items` take effect as configured:
  the theme family is not overridden on first load, and Customize Bars stays
  editable on a deployment without `system-health`. (#863)
- Selecting text in the web terminal copies it: Option or Shift drag with
  Cmd+C or Ctrl+Shift+C grabs raw screen text.
- A profile or persona key with no entries under it, such as `web_panels:` or
  `skills:`, reads as the empty selection, and a channel database in the wrong
  shape is refused instead of loading as zero channels.
- The channel-limits and config-drift hooks match the tool names a deployment
  renders, `entry_publish` asks for approval, and the python executor rejects
  an `execution_mode` other than `readonly` or `readwrite`.
- Headless dispatch runs enforce the trigger's `allowed_tools` through a
  PreToolUse hook, so project `settings.json` allow-rules and the approval
  hook's explicit allows do not widen a run's tool surface.

### Security

- The Web Terminal, ARIEL, Channel Finder, artifact gallery and Theme Lab
  authenticate HTTP and WebSocket requests, and a launcher prints a login URL
  whose token is the server's secret for the life of the process.
- A readonly Python run refuses `write_channel` through the connectors and
  keeps the EPICS connector on the read-only gateway. Refused at import and at
  the call: `epics`, `p4p`, `caproto`, `pvaccess`, `tango`, `doocs4py`,
  `aioca`, dynamically resolved entry points, a shelled-out `caput`, `ctypes`
  library loads, ophyd-async signal puts and a Bluesky `RunEngine`. These
  checks run at runtime, beside the write-pattern scan; work needing a
  subprocess or shared library is resubmitted as readwrite.
- Agent processes started by the Web Terminal and by `osprey chat` do not
  carry the interface-API `*_LAUNCH_TOKEN` credentials, which disarms the
  Bluesky queue-launch tool there; panel-click launches are unaffected.
- New `modules.web_terminals.authorization` block names a role once
  (`operator: {persona: readwrite}`) and maps single-sign-on ID-token group
  claims onto roles; a claim matching nothing or two roles is refused with a
  403.
- A custom web panel may not point at a URL that resolves to the deployment
  host's own address, since the panel proxy fetches server-side; host the
  dashboard elsewhere or reach it from the landing page.
- ARIEL logbook composition sends tool names and result snippets to the
  configured LLM provider, not tool arguments.
- `osprey up` mints a per-user terminal secret for each roster entry in any
  auth mode, so a deployment with authentication off is gated by that secret;
  hand out addresses with `osprey users login-url <user>`.
- `modules.web_terminals.external_origin` names the address browsers reach the
  deployment at; set it when TLS terminates in front of the stack's nginx, or
  the terminals refuse writes from that address as cross-origin.
- Mutating requests and WebSocket upgrades are checked against the
  deployment's external origin on the cookie and token paths, so a multi-user
  stack behind nginx configures that origin; the generated compose files set
  it.
- Logging out of the web terminal revokes the session on the server, so a
  cookie value that has already left the process is refused afterwards.
- Browsing to a gated page without a session returns a readable "not signed
  in" page pointing back at the login link; API calls get a JSON 401.
- The signed-in user and their role reach each terminal as headers set by the
  login service, and each proxied location clears the browser's own copies
  first in any auth mode; logins and refusals go to `var/audit/sidecar/`.
- The nginx access log records neither query strings nor `Referer`,
  `Authorization` is cleared on proxied requests, cookies are scoped to each
  user's path, and roster names are held to `^[a-z0-9][a-z0-9_-]*$`.
- `osprey validate` and `osprey build` refuse a `default_persona` that
  resolves to a persona able to edit the deployment; with authentication off
  deployment-wide the check warns instead.
- The panel proxy strips `Set-Cookie`, `Clear-Site-Data`, `Refresh`, `WWW-
  Authenticate` and `Access-Control-*` from responses and relays a backend
  redirect instead of following it, refusing one to a third site with a 502.
- One audit trail records a deployment's safety decisions (refused writes,
  refused protected keys, MCP tool calls, hook denials, approval prompts,
  logins) under `var/audit/<identity>/<surface>.jsonl`, one line each; records
  carry identifiers and config keys, not values or messages, except that a
  refused Python run stores the code it refused. Each container is handed only
  its own `var/audit/<user>/` and serves `GET /api/audit/recent` behind the
  Config panel's switch, and the Bluesky panel service records under
  `var/audit/bluesky-web/`. `osprey reset --purge-audit` clears it.
- A protected set names the files and `config.yml` keys the agent may not
  rewrite: the rendered config and settings, the hook and MCP wiring, `.env`,
  the `osprey_*` hooks, `.claude/skills/` and `.claude/rules/`.
- `web.config_panel.enabled` closes the Config panel's `/api/config` and
  `/api/claude-setup` routes server-side; ARIEL's settings editor returns 403,
  and `web.scaffold_gallery.write_enabled` closes the gallery's writes.
- The `setup_inspect` agent tool reports `config.yml` with its `${VAR}`
  placeholders intact, and masks literal values under keys named `*KEY*`,
  `*TOKEN*`, `*SECRET*` or `*PASSWORD*` in the config and in `.mcp.json`.
- Agent-supplied artifact metadata is HTML-escaped at the Artifacts gallery's
  render sinks and artifact ids are percent-encoded in URL paths, closing a
  stored cross-site-scripting hole in the gallery sidebar.
- `osprey claude chat`, the web terminal and the dispatch worker start the
  agent with `--setting-sources project`, so a personal
  `~/.claude/settings.json` or `.claude/settings.local.json` does not override
  the configured provider, and they refuse to launch when enterprise managed-
  policy settings set a provider variable OSPREY manages.
- The sandbox posture is enforced by the tool servers as well as the agent-
  side hook: a sandboxed session's write tools are refused before the tool
  body runs. Re-render with `osprey build` to pick this up.
- Readwrite Python executions require human approval under the `selective`
  policy, whether or not the write-pattern scanner recognises the code's
  spelling.
- The DOOCS, TANGO and simulator connectors refuse the write when limits
  validation does not complete, including a missing database or a validator
  that could not run, and report it as a refused write.
- ARIEL's `sql_query` tool refuses a query that reads no allowlisted table,
  such as `SELECT pg_read_file(...)`.
- `osprey up`, the render step and the new `web_terminals.open_mode_egress`
  lint rule refuse an open deployment whose personas can reach the host
  network; the python executor refuses connections to its web ports.
- The Bash and launch-token guard at `osprey up` covers roster entries that
  run no persona, that is the deploy project itself; a `Bash`-capable persona
  gets the launch token only with `dangerously_allow_bash: true`.
- `remove_deny` does not lift a deny the writes kill switch imposed, and a
  profile that writes a permission list as a bare string, or spells
  `claude_code` both dotted and nested, is refused at build time.
- `osprey build` refuses a profile in which `Bash`, `Edit`, `Write`,
  `MultiEdit` or `NotebookEdit` is neither in `permissions.deny` nor covered
  by a `PreToolUse` hook matcher.
- The operator secret is cleared from the server process's environment as each
  interface app is built; on Linux a process that already received it keeps a
  copy in `/proc/<pid>/environ`, readable by the same user.
- `DISPATCH_WORKER_TOKEN` is stripped from sandboxed executions alongside the
  other credentials.
- ARIEL logbook ingestion verifies TLS certificates by default; a site CA is
  named with the new `ariel.ingestion.ca_bundle` key, or verification is
  turned off with `ariel.ingestion.verify_ssl: false`.
- `phoebus_drive` checks the control-system writes kill switch
  (`control_system.writes_enabled`, or a per-connector `writes_enabled`)
  before the approval gate; a panel drive was gated on operator approval
  alone. (#882)
- A `readwrite` run checks channel limits on the `aioca`, pvaPy `Channel` and
  p4p raw-client routes, and on structure and JSON values. Writes with no
  number to bound, such as p4p `rpc` and Tango group writes, are refused.
- ARIEL's `sql_query` refuses a `FROM`/`JOIN` shape its allowlist does not
  resolve: comma-separated FROM lists, parenthesised joins, quoted
  identifiers, comments and dollar quoting. Multi-CTE `WITH` queries are
  accepted.

## [2026.6.2] - 2026-06-28

### Added

- **Runtime panel control for the web terminal** — the agent can show/hide configured panels and register ad-hoc URL panels at runtime (`show_panel`, `hide_panel`, `register_panel`; `list_panels` reports per-panel visibility + active tab). Panels can launch hidden (`web.panels.<id>.hidden`). Runtime URL registration is off by default (`web.allow_runtime_panels`) with an optional SSRF-validated `web.runtime_panel_allowlist`.
- `osprey query "<prompt>"` — headless, read-only agent run for CI: boots the full MCP + tools stack, exits 0 / 1 / 2 (pass / verdict fail / infra error), `--json` for machine-readable output.
- `osprey skills install osprey-design-philosophy` — bundle OSPREY's design principles as an installable skill for contributors.
- `scripts/benchmark/` — model-capability benchmark: runs the model-driving e2e subset across a model × provider matrix declared in one config, and renders a per-test pass-rate dashboard. Adding a model or provider (e.g. local DeepSeek `ds4`, Ollama, vLLM) is a config edit (#259).
- How-to guide for running the agent on open-weight / self-hosted models and reproducing the benchmark (`docs/how-to/run-open-models`).
- Documented the `claude_code.permissions` profile block (`remove_deny`/`deny`/`allow`/`ask`/`remove_ask`) for overriding the default tool deny list (#292).

### Changed

- Removed the non-functional `on_violation` knob from `control_system.limits_checking` and the `config.yml` templates; limit enforcement is unconditional and fail-closed (an `on_violation` key is now ignored).
- ARIEL `capabilities` MCP tool now documents it is *not* a health check (config-only, never touches the DB); use the `status` tool / `osprey ariel status` for live connectivity.
- All agent-facing timestamps now share one configurable facility timezone (`system.timezone`) — archiver queries, live reads, simulated events, ARIEL logbook, and executed-script run times — rendered with an explicit UTC offset; operator-provided times ("today", "14:32") are read as facility-local. Shipped presets pin `system.timezone: UTC`; set your real zone in production (#286).
- OSPREY logs a one-time startup warning when the container `$TZ` disagrees with `system.timezone` (which stays authoritative).
- ARIEL web upload entries (with attachments) now publish through the facility adapter with the same credential and publish-failure handling as text-only entries (previously always saved local-only). Adapters declare whether publishing needs credentials via a fail-closed `requires_write_auth` property, and the web create form adapts its credential prompt via a new `GET /api/publish-info` endpoint. Attachments are stored in ARIEL only (the OLOG write API cannot accept file uploads).
- `claude-agent-sdk` upgraded to 0.2.106 (bundles CLI 2.1.185); `uv.lock` regenerated (#278).

### Fixed

- The Web Terminal's panel nav bar (WORKSPACE, ARIEL, CHANNELS, LATTICE, …) now scrolls horizontally when the tabs outgrow the header instead of letting the trailing tabs get clipped or slide under the header action buttons on a narrow viewport; the scroll has no visible scrollbar and the action buttons stay pinned on the right.
- ARIEL `semantic_search` now degrades gracefully when semantic search is unavailable (no pgvector table / Ollama) — returns an empty result steering the agent to `keyword_search` instead of a tool failure (#276).
- Workspace gallery now shows new artifacts without a manual refresh: the store index is written atomically (temp file + `os.replace`) and the watcher detects the rename (`on_moved`), so a cross-process reader no longer reads a half-written file (#289).
- Channel Finder in-context Explore filter now searches the whole database and re-chunks results, instead of filtering only the displayed page (a match on a later page previously read as "No channels match the filter") (#299).
- ARIEL web Browse pagination now works: `/api/entries` derives an `offset` from `page`, and the declared `author` / `source_system` filters (previously silently ignored) and `total_pages` are now honored.
- `osprey ariel migrate` now creates an embedding table for every model in `enhancement_modules.text_embedding.models`, not just the hardcoded default; previously configured non-default models were dropped and `osprey ariel enhance` produced no embeddings.
- The `control-assistant` **EVENTS** tab now renders in `osprey web` — `events` is now a custom URL-backed panel (it was registered as built-in, so its URL was discarded and the tab never appeared), health-gated against the dispatcher's `/health`.
- The EPICS connector now routes writes through the configured `gateways.write_access` gateway when `control_system.writes_enabled` is true (falling back to `gateways.read_only` otherwise); previously it always used `read_only`, so writes failed on a genuine read-only gateway.
- The channel-limits `defaults` block is now actually inherited — a channel omitting a field (`writable`, `min_value`/`max_value`, `max_step`, `verification`) picks it up from the top-level `defaults` (so `defaults: {writable: false}` takes effect); previously it was validated but dropped.
- Dispatched agent runs now fail fast when the model provider never responds (invalid/expired credential or unreachable base URL): a per-message inactivity watchdog (`DISPATCH_INACTIVITY_SEC`, default 120s) aborts a hung run instead of stalling to the 300s timeout.
- Creating an ARIEL logbook entry no longer silently saves local-only when publishing needs credentials or fails: a logbook requiring credentials returns HTTP 401 and the web form prompts (keeping the entry populated); a real publish failure surfaces an error. Only a genuinely read-only adapter falls back to local-only.
- Web-uploaded (ARIEL-only) attachments on a published entry are no longer erased by a later re-ingestion poll — an upsert carrying no attachments preserves the existing ones (a non-empty list still replaces them) (#291).
- `provider=ds4` (local DeepSeek V4) now resolves in the `control_assistant` and `hello_world` presets (the `api.providers` stanza was missing from the app templates).
- Benchmark matrix runs now provision an isolated ARIEL database per model×seed cell, so concurrent cells no longer drop each other's embedding tables mid-test (#259).
- Simulation `at_time` events now fire at the facility wall-clock time-of-day regardless of the deploy host's `$TZ` (previously placed in the box's local zone).
- Seeded and ingested logbook entries now resolve their relative time-of-day in the facility timezone, so on a non-UTC facility the narrative lands at the same wall-clock time as the telemetry (previously anchored in UTC).
- The ARIEL web **API** now returns entry timestamps as facility-local ISO with an explicit offset, matching the MCP path; web and MCP share one rendering helper and the single-entry `get` is localized too.
- Translation proxy (open-model path) now hoists a `role: "system"` message inside `messages` into the OpenAI system message instead of dropping it, so mid-conversation system instructions survive translation.
- `osprey query` / SDK runner / dispatch worker now expand `${VAR}` in a custom provider's `base_url` against the project `.env`, and start the translation proxy on these paths for non-native (OpenAI-protocol) providers; previously the literal `${VAR}` reached the proxy and the request failed to parse as a URL (#307).

## [2026.6.1] - 2026-06-17

### Added

- **Event dispatch (opt-in).** New `osprey.dispatch` FastMCP server + `osprey.mcp_server.dispatch_worker` service turn external events into headless agent runs, with a live dashboard, an in-memory FIFO pool with backpressure, per-trigger tool allowlists, and a server-side tool denylist (shell + web/browser tools blocked regardless of a trigger's allowlist). All dispatcher and worker HTTP endpoints that carry agent output or accept writes are bearer-token gated (the in-terminal EVENTS tab injects the token server-side, so the browser never holds it); `osprey deploy up` auto-generates the tokens into the project `.env` so a fresh deploy is secure with zero editing. Enable per profile via a `dispatch:` block; the `control-assistant` preset ships four control-system-free tutorial triggers (fire `hello-dispatch` with a single `curl`). Trigger sources are pluggable via the `osprey.trigger_sources` entry-point group (built-in: `webhook`, `cron`). Worker containers mount the project `.env` read-only so dispatched agent runs can authenticate to the LLM provider. The pipeline is exercised end-to-end by real-token e2e tests — a subprocess sweep over the shipped triggers and a full Docker-stack deploy. `osprey deploy up` builds a shared local image for the dispatcher + worker from a bundled Dockerfile (no published image required); use `--dev` to bake in a local osprey checkout, or set `OSPREY_DISPATCH_IMAGE`/`OSPREY_WORKER_IMAGE` to use a prebuilt image.
- The `control-assistant` preset now surfaces the event-dispatcher dashboard as an in-terminal **EVENTS** tab in `osprey web` (health-gated; repoint via `EVENT_DISPATCHER_URL`).
- **Facility Knowledge (OKF).** Structured markdown bundle (`osprey_facility_knowledge` MCP server, `list_concepts` / `read_concept` / `search` tools) for on-demand retrieval of subsystem descriptions, device details, operational procedures, and facility-specific references. `facility.md` is thinned to facility identity only; deep content is fetched via the agent on demand. The `control_assistant` preset ships an Example Research Facility bundle. Includes `draft_concept` write tool (approval-gated) for authoring new concept docs directly from an agent session. See :doc:`/how-to/use-facility-knowledge`.
- `osprey knowledge` CLI: `regen-index` (regenerate bundle indexes, idempotent), `validate` (collect-all frontmatter + index validation, exits 1 on any failure), `seed-from-ttl` (seed device stubs from a NARAD/als-ontology TTL; requires `knowledge` extra; `--force` to overwrite hand-edited stubs).
- `facility-knowledge` subagent: specialist agent scoped to `list_concepts` / `read_concept` / `search`; enabled by default in `control_assistant`; delegates facility knowledge lookups out of the main agent's context.
- Simulation engine: `at_time: "HH:MM:SS"` event anchor — daily local-time recurrence for archiver events (step/spike; width in seconds), complementing window-fraction `at` and activation-relative `at_offset`.
- Simulation engine: optional per-channel `min`/`max` physical bounds clamp live reads and synthesized history on the way out (e.g. forward RF power floored at 0 saturates instead of going negative during a trip); overrides and writes are stored verbatim.
- **Composable, self-contained simulation scenarios.** Scenarios are bundles under `data/simulation/scenarios/<name>/` — each owns its telemetry overlay (`scenario.json`) and, optionally, its logbook narrative (`logbook.json`). Several can be active at once as long as they touch disjoint channel sets (overlapping channels are a hard error). The `control_assistant` preset ships `vacuum-burst` (telemetry-only) and `rf-thermal` (with its DEMO-026/027/028 incident arc).
- **`osprey sim` CLI** — `osprey sim list | status | apply NAME...` composes and applies one or more scenarios: it writes the simulator state (with a shared apply-time anchor) and purges + reseeds the ARIEL logbook from the active bundles, so the narrative the agent searches matches the telemetry it reads, against one clock. `apply` confirms before purging (`--yes` to skip, `--no-seed` for telemetry only).
- **Relative timestamps for demo/seed logbooks** (`osprey.utils.relative_time`) — demo and seed entries express timestamps as `{days_ago, time}` and resolve to concrete datetimes when data enters the system (`osprey sim apply` for scenario bundles; `osprey ariel ingest`/`quickstart` for the generic JSON adapter, which now accepts a relative `when` alongside absolute `timestamp`). The data lands at a recent, deterministic position with no build-time file mutation.
- Deterministic statistical-contract tests (`tests/simulation/test_control_assistant_scenarios.py`, `test_scenario_composition.py`) pin the scenarios' signatures (anti-correlation, excursion positions, derived-channel consistency) and the disjoint-composition contract, so the LLM-judge e2e tests run against a known-good data substrate.

### Changed

- Hello-world tutorial now launches via `osprey claude chat` instead of bare `claude`, so the provider configured under `claude_code` is actually used (bare `claude` silently falls back to the shell's provider); links the CLI-chat how-to (#261).
- The control_assistant scenario e2e tests (`test_vacuum_burst_scenario` and `test_rf_cavity_correlation_scenario`) drive their archiver ground truth from the simulation engine via `activate_scenarios`, which also seeds the matching ARIEL logbook deterministically at setup (replacing the manual `purge && ingest` pre-seed and its stale-DB footgun). They build at tier 3 so every simulated channel is discoverable through the channel finder.

### Fixed

- Drop the unused `ANTHROPIC_API_KEY` forwarding from the `python` executor MCP server env; nothing in the executor stack reads it (#264).
- `osprey deploy up --dev` now passes `--build` so the freshly-rendered local wheel is actually baked into the image; previously compose reused the cached image and ran stale code after the first build.
- `osprey deploy up --dev` now builds the local wheel with the active interpreter (`sys.executable`) instead of bare `python3`. In a non-activated virtualenv, PATH `python3` is the system/pyenv interpreter, which lacks the `build` package — so the wheel build silently failed and containers fell back to the released PyPI version (missing any unreleased local code).
- `osprey build` no longer aborts when a profile removes a framework agent (e.g. dropping the logbook agents) while also disabling the MCP server those agents depended on. OSPREY's two-pass `.claude/` render left the dropped agents' `.md` files orphaned on disk with their original `mcp__*` tool frontmatter, which the agent tool/permission drift check then flagged as a false positive and aborted the build. Files outside the active manifest selection are now deleted on re-render instead of left behind. (#266)

### Removed

- The build-time logbook timestamp rebase (`rebase_logbook_timestamps`) is gone: demo/seed logbooks now carry declarative relative timestamps resolved at ingest/apply time instead of being mutated in place during `osprey build` (which no longer requires a running Postgres for logbook setup).
- **BREAKING:** the mock archiver no longer emits the built-in Sector-7 vacuum-burst and RF cavity-C1 thermal-excursion demo events from hard-coded source. That physics is now data-driven (ship it as a `machine.json` scenario, e.g. the `control_assistant` preset's `vacuum-burst`/`rf-thermal`). Without a `simulation_file`, the mock archiver synthesizes only generic per-PV-type waveforms.

## [2026.6.0] - 2026-06-12

### Added

- **`ds4` provider** — local DwarfStar/DeepSeek-V4 inference server (OpenAI-compatible, keyless, default `http://127.0.0.1:8000/v1`). Introduces a per-provider `supports_native_structured_output` capability flag (`True`/`False`/`None`=auto-detect via LiteLLM) replacing the hardcoded structured-output whitelist; ds4 declares `False` because the server accepts but ignores `response_format: json_schema`, so OSPREY's prompt-based JSON fallback is used. Also fixes URL mangling in the vLLM/ds4 health checks (`rstrip('/v1')` stripped characters, breaking ports ending in `1`).
- **Data-driven simulation engine** (`osprey.simulation`) backing the mock control-system and archiver connectors: a `machine.json` defines channels (baseline values or derived expressions), scenarios (override sets), and archiver event scripts (step/ramp/spike, window-fraction or wall-clock anchored), so corrective writes propagate through physics couplings and archived history correlates with live values. Ships with a generic `sim-scenarios` skill for listing/switching scenarios.
- **Generated reference Dockerfile.** `osprey build` now renders a self-documenting `Dockerfile` + `.dockerignore` into every project root — install Claude Code + OSPREY, copy the project, relocate paths, serve the web terminal on 8087 as a non-root user. Generated once and user-owned (`regen` never touches them); site extension via exactly three build ARGs (`OSPREY_PIP_SPEC`, `PIP_NO_PROXY`, `OSPREY_OFFLINE`). Profile pip `dependencies:` are baked into the install line. New how-to: `docs/source/how-to/containerize-project.rst`. Guarded by unit content tests, a CLI cross-check (every `osprey` invocation in the rendered Dockerfile must resolve against the real click tree), and a docker-build e2e (`dockerfile-e2e` CI job, advisory).
- **`osprey claude regen --runtime-root PATH`.** Rewrites `project_root` in `config.yml` (comment-preserving) and re-renders Claude Code artifacts against the new root; a recorded `execution.python_env_path` that doesn't exist on the current filesystem is replaced with the current interpreter. Supersedes the manual "clear python_env_path + regen" container fix; used by the generated Dockerfile.
- New e2e scenario `tests/e2e/test_corrector_limit_honest_refusal_scenario.py` — asserts the agent's *behavior under refusal* (no channel-shopping, no intent-splitting, no false success, clear safety-attribution, operator looped in), complementing the existing mechanism-level safety tests. Two-layer grading (deterministic + LLM judge).

### Changed

- **`claude-agent-sdk` upgraded to 0.2.93** (bundles CLI 2.1.167); als-apg routing re-verified (hello-world canonical flow + approval-hook e2e).
- `data-visualizer` subagent now defaults to `create_interactive_plot` when the caller does not explicitly request a static figure. Fixes the case where vague requests (e.g. "3D waterfall plot") produced an unreadable fixed-viewpoint matplotlib image instead of a rotatable Plotly view.

### Fixed

- The Claude Code translation proxy (used to route OpenAI-protocol providers through `osprey claude` and the web terminal) now pools a single upstream `httpx` client for the app's lifetime and closes it via a FastAPI lifespan handler, instead of opening and tearing down a connection per request. Under sustained load the per-request client exhausted the host's ephemeral-port pool via accumulated `TIME_WAIT` sockets; the pooled client (20 keepalive / 100 max connections, 30s expiry) fixes the exhaustion.
- E2E SDK harness now waits for MCP servers to finish their async handshake before sending the prompt. The OSPREY `controls` stdio server cold-starts ~1.5s after launch (slower than `python`/`osprey_workspace`); under load an agent's first turn could beat it, see no controls tools, and give up — a flaky failure misattributed to model capability. Both `run_sdk_query` and `run_sdk_query_with_hooks` now poll `ClaudeSDKClient.get_mcp_status()` until the project's declared servers report `connected`, and capture that snapshot on the result (`mcp_server_status`/`registered_tools`/`tool_was_registered`, optional per-test sidecar via `OSPREY_E2E_INIT_SIDECAR`) so a missing tool is provably infrastructure rather than a silent model give-up.
- `ariel search` CLI now renders keyword-search entries when no composed answer is available, instead of printing an empty result.
- `osprey build` re-anchors profile-overlaid logbook seeds to the current date (overlays land after the in-template timestamp rebase).
- Editing `config.yml` no longer leaves the agent running stale settings (#244). Config changes via the web settings panel and `osprey config set-*` now auto-regenerate the Claude Code artifacts (so e.g. flipping `control_system.writes_enabled` actually takes effect), the web server re-syncs them on launch, and a SessionStart guard warns when a hand-edited `config.yml` has drifted from the generated `.claude/` artifacts. The hello-world tutorial documents the `osprey claude regen` + relaunch step.
- Regen drift detection (`osprey claude status` and the auto-regen gate above) no longer reports phantom drift for user-owned artifacts (e.g. the create-only `facility.md`), which would have re-rendered and backed up artifacts on every web launch.
- `lttb_downsample()` no longer crashes (`TypeError`) on archiver `None` gap values; gaps are treated as `0.0` for downsample selection and preserved as `null` in the returned data so charts render true gaps (#247).
- `rules/data-visualization.md` is now gated on the data-visualizer subagent being disabled. When the subagent is enabled (the default), CLAUDE.md forbids the main agent from calling `create_static_plot` / `create_interactive_plot` / `create_dashboard` / `python_execute` / `Write`, so shipping a rule that teaches those tools was contradictory context. The file is now a `.md.j2` template that renders empty (and is auto-unlinked) when the subagent is enabled.
- **Framework safety hooks were silently not launching on hosts without a `python` on `PATH`** (stock macOS, many Linux ship only `python3`). The approval, writes-kill-switch, limits, error-guidance, and feedback PreToolUse/PostToolUse hook commands were rendered as bare `python "..."`, so the whole hook layer failed to start and safety tests passed vacuously. Hooks now render via the project's resolved venv interpreter (`current_python_env`); a new regen test asserts every framework hook command launches via an absolute interpreter path.
- Writes-disabled kill switch now covers the Python executor. When `control_system.writes_enabled` is false, `mcp__python__execute` is pulled out of `permissions.ask` (it can't be denied wholesale — it has a legitimate read-only path), so the `osprey_writes_check` hook's deny on a read-write execute stands alone. Previously a static `ask` entry drove the call to the Claude Code approval prompt even when the hook denied in parallel (claude-agent-sdk 0.2.93), letting a read-write execute reach the operator instead of being blocked.
- Channel-finder benchmark tier-1 database now includes the vacuum gauge system (`VAC/GAUGE`, 205 → 217 channels) consistently across all three paradigms, regenerated from the `TierSpec` source of truth (previously hand-edited into only the `hierarchical` paradigm, leaving `in_context`/`middle_layer` and the spec out of sync).

### Removed

- Unused `caproto` dependency and stale `osprey generate soft-ioc` hints (command removed in the LangGraph-era cleanup).

## [2026.5.2] - 2026-05-27

### Added

- Pin Claude Code CLI version per project. New `claude_code.cli_version` config field; when set, OSPREY launches via `npx -y @anthropic-ai/claude-code@<version>` instead of bare `claude`. Use to insulate projects from upstream CC breakage (#218). Verified by `osprey health`; bypass with `osprey claude chat --no-pin`.
- **MongoDB archiver connector.** New `mongodb_archiver` type for querying MongoDB time-series collections; install via `pip install "osprey-framework[archiver-mongodb]"` and configure with `archiver.type: mongodb_archiver` plus host/database/collection/auth settings. Ports PR #84 (RemiLehe) onto the current registry; documents are expected to have shape `{date: ISODate, PV1: value, PV2: value, ...}`.

### Changed

- Documentation cleanup pass over `docs/source/`: resync getting-started, how-to, architecture, and reference pages with the current codebase; fix stale APIs, config keys, and cross-references accumulated since the native-capabilities migration.
- **`pyepics` promoted to base dependencies.** EPICS connector users no longer need `pip install "osprey-framework[dev]"` to get a working EPICS install; pyepics is used in three production paths (control_system connector, limits_validator, python_executor wrapper).
- **`claude-agent-sdk` upgraded to 0.2.87** (lockfile + venv now match `pyproject.toml`; PR #231 bumped the pin without regenerating the lock). Bundles CLI 2.1.150; CBORG/als-apg/anthropic-direct routing re-verified — the old "0.1.27+ breaks CBORG" warning was stale.

### Fixed

- `BaseStore`: in-process lock fixes a lost-update race between the gallery index-watcher thread and same-process saves.
- `quick_check.sh` / `ci_check.sh` prune stale `__pycache__` + empty dirs so a deleted package can't resurface as a namespace package locally.
- e2e SDK trace collector now reads sub-agent transcripts via `list_subagents`/`get_subagent_messages`. CLI ≥2.1.x stopped streaming sub-agent messages through `query()`, blinding delegation/viz/search e2e tests; they observe sub-agent tool calls again.
- `test_channel_read_archiver_plot_with_audit`: dropped the chronological tool-ordering assertions. Harvested sub-agent traces are appended after the main stream, so `tool_names` isn't time-ordered once the agent delegates (the norm on Haiku); the ordering check failed spuriously even when the workflow was correct and the PNG persisted.
- `test_claude_code_build_integration` archiver+plot tests: dropped the belt-and-suspenders scan of the agent's closing `--print` message for plot/BPM vocabulary. That message is free-form and model-dependent (Haiku sometimes ends with "what next?"), so it flaked while the PNG/data-file artifact assertions — the authoritative workflow check — still passed.
- `hello_world` preset triggered "OSPREY APPROVAL REQUIRED" on `channel_read`. Its `config.yml.j2` shipped without an `approval:` block, so after the `global_mode` removal (May 5) every MCP tool fell through to `default_policy: "always"`. Added the standard per-tool policies (`channel_read: skip`, `channel_write: always`, ...) matching the other presets.
- channel-finder `in_context`/`middle_layer` MCP servers crashed on startup (missing required `workspace_root` arg after RF-001); both now resolve it like the other servers.
- Multi-step agentic-pipeline e2e tests now `@pytest.mark.flaky(reruns=2)`: the discover→fetch→plot/persist-artifact tests in `test_sdk_workflows.py` (×3), `test_audit_observability.py`, and `test_data_visualizer.py` (×2). The Haiku orchestrator occasionally (~5%/run) stops before completing the pipeline or persisting the plot. Reruns absorb that stochastic miss while every deterministic assertion stays strict — a real regression fails all attempts. Deterministic safety/approval/delegation e2e tests are intentionally left strict. Adds `pytest-rerunfailures` dev dep.

### Removed

- ARIEL's duplicate Claude Setup tab (`/api/claude-setup` endpoints, `claude-setup.js`, related HTML/CSS). The canonical agent-file editor remains in the web terminal at `web_terminal/routes/config.py`; ARIEL's settings drawer now only edits `config.yml`.

## [2026.5.1] - 2026-05-21

### Added
- **Paradigm-agnostic channel finder.** `in_context` / `hierarchical` / `middle_layer` share one tier-resolved query set; `control_assistant` ships all 9 tier DBs so you switch tier per run instead of re-initialising.
- **ARIEL standalone preset.** Logbook deployment without the control-system stack (no channel finder, no archiver, no Python executor). See `docs/source/how-to/ariel/standalone-deployment.rst`.
- **Virtual-accelerator scenarios.** Mock archiver emits seeded correlated events (Sector-7 vacuum burst, RF-cavity-C1 thermal runaway) for operator-style investigation tests. *(Superseded — these are now data-driven simulation scenarios; see the Unreleased "Removed" entry.)*
- **Profile-mode builds.** `osprey build --emit-profile DIR --preset X` scaffolds an editable profile; `extends:` accepts bundled preset names.

### Changed
- **GitHub Flow.** Trunk-based on `main`; `next` retired.
- **BREAKING.** `prompts` → `scaffold` (CLI, web, Python, config). No shim.
- **BREAKING.** ARIEL internal RAG / Agent pipelines removed; drop `pipelines.rag` / `pipelines.agent` from configs.
- **BREAKING.** Channel-finder hierarchical schema: bare-numeric device IDs (`SR:DIAG:BPM:01:…` instead of `SR:DIAG:BPM:BPM01:…`), realistic facility sizes.
- **BREAKING.** `build-interview` skill renamed to `osprey-build-interview`.
- **BREAKING.** `channel_finder_mode="all"` removed.
- `query_channels` MCP tool replaces the `list/get/search_channels` triplet.

### Fixed
- Cleanup batch that should have been in v2026.5.0: `osprey build` after `uv tool install` (#216), `osprey deploy up` on presets with no services, `suffix_map` honoured in channel addresses, ARGO base URL (#214), E2E suite green again on CI, and various polish.

### Removed
- Legacy hierarchical "container" schema, vestigial benchmark CLI flags (`--tier`, `--judge-provider`), `start_typesense.sh`, `tests/cassettes/`, `pull_request_template.md`.

## [2026.5.0] - 2026-04-29

This release retires the LangGraph-based orchestration that powered Osprey
through the 0.x series and rebuilds the framework on top of Claude Code
plus a small fleet of MCP servers. Roughly 475 commits touched almost
every subsystem; the entries below summarize the result rather than each
step. Versioning also changes: Osprey adopts CalVer (`YYYY.MM.MICRO`).
The release machinery is scheme-blind, but downstream parsers that
assume `MAJOR.MINOR.PATCH` semantics need updating.

### Highlights
- **Claude Code + MCP orchestration.** The LangGraph agent graph,
  pipelines, and supporting plumbing are gone. Agents now run under
  Claude Code, consuming five FastMCP servers (`control_system`,
  `python_executor`, `workspace`, `accelpapers`, `matlab`) and a registry
  of native tools. Hardware writes still require human approval through
  the Claude Code prompt surface.
- **Unified `osprey build`.** `osprey init`, `osprey migrate`, the init
  wizard, `osprey config set-models`, and `osprey tasks` are retired.
  A single `osprey build <name> --preset <preset>` (or
  `osprey build <name> <profile.yml>`) scaffolds projects, with `-O FILE`
  overlays and `--set KEY.PATH=VALUE` overrides composing the final
  profile. Profiles support `extends:` inheritance and a `lifecycle:`
  section for build-time commands.
- **First-class skills.** `build-interview` and `osprey-build-deploy`
  ship as installable Claude Code skills via `osprey skills install`. New
  `lattice-evaluation` skill orchestrates nonlinear-dynamics analysis
  (resonance diagram, dynamic aperture, frequency map).
- **Agent-agnostic framing.** Documentation, the landing page, and the
  architecture diagrams are rewritten around the Claude-Code-plus-MCP
  story. ALS-specific agents and servers moved out to facility profiles.

### Breaking changes
- LangGraph removed entirely — source, dependencies, CI hooks, templates,
  and tests. There is no compatibility shim; downstream projects pinning
  LangGraph nodes will not import.
- `osprey init`, `osprey migrate`, `osprey tasks`, `osprey config
  set-models`, and the interactive init wizard are removed. Use
  `osprey build` for all scaffolding.
- `lattice_design` template removed. Build profiles strictly accept
  `{hello_world, control_assistant}`.
- `migrate-legacy` skill removed; the legacy-project migration flow is
  now a path inside `build-interview`.
- Generators / code-generator concept, Karma analytics, the Grafana/OTEL
  monitoring stack, the DePlot service, the graph-analyst agent, and the
  direct-channel-finder MCP server are retired or archived.
- `archivertools` dependency dropped in favor of direct HTTP to the
  EPICS Archiver Appliance.
- Manifest schema → 1.2.0. `.osprey-manifest.json` records `build_args`
  (renamed from `init_args`) with a `source: "preset"|"profile"`
  discriminator; `creation.registry_style` is removed. No reader shim.
- Versioning: SemVer → CalVer (`YYYY.MM.MICRO`).
- Default LLM provider: `cborg` (LBNL-only) → `anthropic`.
- Workspace and agent-data directories unified under `_agent_data`
  (`agent_data.base_dir`); the previous `osprey-workspace/` layout is
  gone.

### Migration guide
- **Existing 0.x project →** run `build-interview` in Claude Code to
  reverse-engineer a build profile from the project, then
  `osprey build <name> <profile.yml>`. The build-interview migration
  phase walks per-service decomposition (jupyter, open-webui, pipelines,
  postgres, ariel) and ends with a smoke-import + `osprey audit` check.
- **Custom LangGraph nodes →** rewrite as MCP tools. The
  `python_executor`, `workspace`, and `control_system` servers are the
  intended extension surfaces.
- **Custom connectors →** override `write_multiple_channels()` if you
  need atomic batch semantics; the default is sequential.
- **Pinning models →** prefer tier names (`haiku` / `sonnet` / `opus`)
  in config; `resolve_model_id()` maps them to provider-specific IDs at
  runtime.

### Added
- **Build & deploy.** `osprey build` with preset overlays and profile
  inheritance, relocatable builds (`--runtime-root`), background
  deployment (`--detach` / `osprey web stop`), CI-friendly flags
  (`--skip-deps`, `--stream`), per-step lifecycle timeouts,
  `requires_osprey_version` profile field, `.env.template` generation,
  auto-install of profile dependencies, and `osprey audit` for project
  safety auditing.
- **Connectors.** `write_multiple_channels()` batch-write hook;
  defense-in-depth `writes_enabled` enforcement via `__init_subclass__`;
  protocol-aware safety rules covering 15 protocols; centralized
  connector type strings; EPICS Archiver Appliance via direct HTTP.
- **Channel Finder.** Web UI with tree preview, multi-select sector
  chips, inline description editing, delete-impact dialog, feedback
  management with pending-review workflow, address resolution, DuckDB
  and Google Sheets backends, and a device-info endpoint.
- **Lattice design.** Interactive lattice dashboard with live optics
  visualization, drag-and-drop panel layout, theme-aware Plotly
  rendering, configurable settings sidebar; bisection-based LMA with
  12-sector auto-detection; ALS-U accumulator ring lattice; nonlinear
  dynamics skills (`resonance-diagram`, `dynamic-aperture`,
  `frequency-map`) and the `lattice-evaluation` orchestration skill.
- **Web Terminal.** PTY session pool for fast switching, session
  persistence and resume, theme system, memory panel and prompt gallery,
  activity diagnostics panel (agent / log / summary / chat), safety
  guidelines page, agent timeline, REST `/api/chat` endpoint, welcome
  modal with version display, and effort-level control via config and
  CLI.
- **Custom panels.** Template-driven panel configuration; WebSocket and
  reverse-proxy support for containerized companion servers; iframe
  embedding contract via `basePath` and `X-Forwarded-Prefix`;
  `list_panels` / `switch_panel` MCP tools.
- **ARIEL.** Bidirectional facility-adapter writes (atomic local JSON
  append, olog RPC POST); per-user OLOG credentials via web form;
  multi-artifact compose with model tiers; `logbook-search` sub-agent;
  artifact-to-attachment converter registry; sync command and publish
  tool.
- **MCP servers.** MATLAB Middle Layer server; `python_executor`
  `execute_file` tool; `workspace` `data_context_read` and
  `artifact_get`; HTTP / SSE / streamable-http transport support; native
  textbook file access replacing the prior textbooks MCP server.
- **Skills.** `build-interview` (with friction-log capture, migration
  path, and Phase 8 install verification), `osprey-build-deploy`,
  `lattice-evaluation`, `analyze-working-point`, `demo-gallery`, and a
  `setup-mode` diagnostic.
- **Providers.** AMSC; per-project Claude Code provider isolation;
  OpenAI-compatible-only providers; tier-name resolution
  (`haiku` / `sonnet` / `opus`); provider-registry overhaul.
- **Vendor & offline.** Manifest-driven vendor assets via
  `osprey vendor fetch`; CDN by default with `OSPREY_OFFLINE=1` opt-in
  for firewalled deployments; `--insecure` for corporate proxies
  (verified against per-asset SHA256); bundled Google Fonts.
- **Safety & permissions.** Per-tool approval policies; structured
  limits errors; facility-configurable `settings.json` permissions with
  drag-and-drop editor; memory-guard hook; configurable artifact
  categories per build profile; `OSPREY_HOOK_DEBUG` activity logging.
- **CI.** Doc-executability gate that builds a wheel from the current
  SHA, installs it on a clean Ubuntu runner, and runs the bash blocks
  extracted from `installation.rst` / `build-interview.rst`. Tiered MCP
  boot-smoke and preset-agentic test pyramid.

### Changed
- Documentation rewritten around the MCP architecture (installation,
  tutorial, architecture, deploy, MCP servers, build profiles); landing
  page and data-flow diagram refreshed.
- AccelPapers search migrated from SQLite FTS5 to Typesense (hybrid
  BM25 + vector); embedding URL / model / key configurable via env.
- Stores moved to `osprey.stores/`; MCP servers consolidated under
  `mcp_server/`; large modules (`web_terminal/routes.py`,
  `registry/manager.py`, `deployment/container_manager.py`,
  `cli/templates.py`, `cli/interactive_menu.py`) split into focused
  packages.
- Build-profile validation: unknown top-level keys warn (will hard-fail
  in a future release); unknown `web_panels` hard-fail today.
- CBORG model IDs pinned to versioned aliases so Claude Code negotiates
  the correct API schema.

### Fixed
- Datetime normalization, NaN sanitization in archiver and artifact
  responses, and async-safe PV reads (carried over from the 0.11.5 line).
- Numerous web-terminal panel-proxy edge cases for iframe embedding,
  WebSocket forwarding, and corporate-proxy bypass.
- `osprey build` now exits with status 2 (usage error) for unknown
  presets, matching the documented contract.

### Changed (Benchmarks)
- **Benchmarks**: `format_in_context()` now returns `{_metadata, channels}` envelope with auto-generated aliases (`channel`) and PV addresses (`address`) instead of a plain list — **breaking change** for code that directly parses the in-context JSON as a list
- **Benchmarks**: `format_middle_layer()` now generates `_setup` blocks (`CommonNames`, `DeviceList`, `ElementList`) for every family, enabling sector filtering
- **Benchmarks**: `validate_queries()` supports per-tier validation via `tier_queries` parameter; backward-compatible with single-file mode
- **Benchmarks**: `BenchmarkRun.paradigm` / `tier` now `Optional` (defaults `None`) — single-backend consumers no longer need synthetic labels
- **Benchmarks**: `BenchmarkSuite` slimmed to bundle role; `AggregatedCell` and matrix-aggregation helpers (`aggregate_by_cell`, `to_table`, `MODEL_PRICES_PER_MTOK`, `infer_cost_usd`) moved to companion paper repo
- **Benchmarks**: `evaluate_response()` and `BenchmarkRunner` gain `use_llm_judge` flag (default `False`) — Stage 2 LLM precision judge is now true opt-in

### Removed (Benchmarks paper-split)
- **Benchmarks**: Cross-paradigm orchestration scripts, tier datasets, and archived result artifacts moved to companion paper repo (`~/LBL/ML/osprey-cf-paper`)
- **Benchmarks**: 9 dev-residue benchmark/debug scripts (`bench_test*`, `bench_matrix`, `bench_warmth`, `debug_benchmark_scaffold`, `trace_single_query`, `migrate_benchmark_filenames`, `analyze_test_coverage`, `backfill_in_context_tokens`)
- **Tooling**: `scripts/capture_ariel_screenshots.py` → `docs/tooling/` (documentation utility, not a runtime script)

## [0.11.5] - 2026-03-13

### Fixed
- **Timezone**: Normalize all datetimes to local timezone with human-readable `timezone_name` field throughout the pipeline — fixes UTC/local confusion in responses, plots, and time range queries (#189, #187)
- **Open WebUI**: Fix artifacts not showing up in Open WebUI interface (#179)
- **Benchmarks**: Fix channel finder benchmark rate limit handling — make OpenAI retry budget dynamic with `retry_budget_override()` context manager, add agent-level retry in middle layer pipeline, and query-level retry with backoff in benchmark runner
- **Config**: `.env` file now overrides existing environment variables (`load_dotenv(override=True)`) so `.env` is the source of truth for API keys

## [0.11.4] - 2026-02-23

### Added
- **ARIEL**: Bidirectional facility adapter write support (#174)
  - Rename `BaseAdapter` → `FacilityAdapter` with backwards-compatible alias; add `supports_write` and `create_entry()` to adapter interface
  - Implement write path for `GenericJSONAdapter` (atomic local JSON append) and `ALSLogbookAdapter` (olog RPC XML POST with retry)
  - `ARIELSearchService.create_entry()` orchestrates writes to facility logbook first, then optimistic local upsert with re-ingestion sync
  - New models: `FacilityEntryCreateRequest`, `FacilityEntryCreateResult`, `SyncStatus`, `WriteConfig`
- **Machine State**: Add `MachineStateReader` service for bulk channel snapshots (#173)
  - Reads channel snapshots from the control system connector with structured models for channel definitions, results, and snapshots
  - Pipeline-aware Jinja2 template (`machine_state_channels.json.j2`) selects demo channels matching the active channel finder pipeline
- **Channel Finder**: Add Google Sheets channel database backend (#171)
  - `GoogleSheetsChannelDatabase` reads/writes channel data from a Google Sheets spreadsheet via `gspread`
  - Integrates with the `in_context` pipeline via `source: google_sheets` config option
  - Optional dependency: `pip install osprey[sheets]`
- **Config**: `resolve_model_id()` utility to resolve tier names (haiku/sonnet/opus) to provider-specific model IDs
- **Channel Finder**: Tree preview, hierarchy selections paths tracking, feedback store for successful runs, hint injection, and device info endpoint
- **Hooks**: Shared `osprey_hook_log.py` utility with `OSPREY_HOOK_DEBUG` gated activity logging
- **Transcript Reader**: Agent start/stop lifecycle events and string content handling
- **Web Terminal**: Session diagnostics panel (Activity tab) with agent, log, summary, and chat views; local safety guidelines page

### Fixed
- **AccelPapers**: Fix double-path 404 in Typesense auto-embedding — URL was `http://localhost:11434/v1/embeddings` but Typesense appends `/v1/embeddings` itself
- **Hooks**: Deterministic project dir resolution via `hook_input["cwd"]` (replaces inconsistent env var fallback chains)
- **Hooks**: Dynamic `write_tools` for writes kill switch — custom MCP server write tools now blocked when `writes_enabled: false` (GP-003)

### Changed
- **Config**: Default provider changed from `cborg` (LBNL-only) to `anthropic` (universally accessible) (GP-006)
- **Stores**: Move `ArtifactStore`, `BaseStore`, `type_registry`, `notebook_renderer` from `mcp_server/` to new `osprey.stores/` package (RF-005)
- **Textbooks**: Replace textbooks MCP server with native file access for faster lookups
- **Init**: Config now uses tier names (haiku/sonnet/opus) instead of provider-specific model IDs; default tier changed to haiku
- **Services**: Services now resolve tier names to provider-specific model IDs via `resolve_model_id`
- **AccelPapers**: Migrated search backend from SQLite FTS5 to Typesense for hybrid BM25 + vector search
- **MCP Servers**: Consolidate all 9 MCP servers under `mcp_server/` — moved ARIEL MCP from `interfaces/ariel/mcp/` and 3 channel finder MCPs from `services/channel_finder/mcp/` (RF-005)
- **AccelPapers**: Embedding URL, model, and API key now configurable via `ACCELPAPERS_OLLAMA_URL`, `ACCELPAPERS_EMBEDDING_MODEL`, `ACCELPAPERS_EMBEDDING_API_KEY` env vars; `--data-dir` falls back to `ACCELPAPERS_DATA_DIR`; added `--ollama-url` CLI flag

### Fixed
- **CI**: Make E2E test failures non-blocking in gate job — E2E tests are LLM-dependent and fail due to API rate limits, not code issues
- **Tests**: Mark flaky `test_first_order_backend_multiple_setpoint_changes` as `xfail` on macOS CI runners due to intermittent caproto server timeouts

## [0.11.3] - 2026-02-22

### Added
- **Providers**: Add American Science Cloud (AMSC) as LLM provider (#170)

### Fixed
- **Safety**: Block retry on channel limits violation — LLM could previously retry around a safety-blocked write, now sets `is_failed=True` to prevent workaround attempts
- **State**: Slash commands `/task:off`, `/caps:off`, `/approval:off` silently dropped state changes due to missing fields in `AgentControlState` (#169)

### Changed
- **Build**: Migrate from pip/setuptools to uv/hatchling (#166)
  - Switch build backend to hatchling with dynamic versioning via `hatch-vcs`
  - Replace pip with uv across all CI workflows, container scripts, and developer tooling
  - Delete legacy `src/setup.py` stub

## [0.11.2] - 2026-02-15

### Added
- **Infrastructure**: Add reactive orchestrator with ReAct-style tool loop (#162)
  - `ReactiveOrchestratorNode`: autonomous Reason+Act loop that replaces rigid plan-then-execute with iterative, LLM-driven decision-making
  - Reactive tool system with tool registry, argument parsing, and result formatting for native tool calling
  - `ChatRequest`/`ChatResponse` models for structured LLM interactions with `tool_calls` and `tool_results` message support
  - Router extensions for classifying reactive vs. planning mode
  - Approval system hooks for gating tool execution
  - Classifier-level dependency expansion: capabilities with unsatisfied `requires` automatically pull in their providers (e.g., selecting `channel_write` adds `channel_finding`) with transitive resolution
  - Pre-dispatch dependency validation to prevent premature capability execution
- **Events**: Add unified typed event system replacing dict-based logging
  - 18 typed dataclass events across 7 categories: status, phase lifecycle, data output, capability, LLM, tool/code, and control flow
  - `EventEmitter` with LangGraph-first streaming and fallback handler support
  - `parse_event()` for reconstructing typed events from serialized dicts
  - `consume_stream()` multi-mode helper combining typed events with LLM token streaming
  - `LLMRequestEvent`/`LLMResponseEvent` with token counts, cost, and duration metadata
  - `ApprovalRequiredEvent`/`ApprovalReceivedEvent` for hardware write gating
  - Completely eliminates raw Python logger usage across osprey core
- **Interfaces**: Add web debug interface for real-time event visualization
  - FastAPI server with WebSocket event streaming at `/ws/events`
  - Dark-themed minimalist browser UI with component filtering and search
  - Tooltips, event previews, and level-specific styling (warning/error/success/key_info)
  - Color endpoint returning component color mappings with hex palette for terminal color matching
  - LLM streaming groups with tabbed viewer
- **Interfaces**: Add LLM token streaming across all interfaces
  - CLI: streaming for respond node and code generator with Rich table-based output
  - TUI: `StreamingChatMessage` with Textual `MarkdownStream` for buffered token rendering
  - TUI: `CollapsibleCodeMessage` with auto-collapse, attempt tracking, and syntax highlighting
  - Open WebUI: streaming support with event parser adapted to unified typed event system
  - Multi-mode streaming architecture combining `custom`, `messages`, and `updates` stream modes
  - Subgraph streaming (`subgraphs=True`) for nested service graphs (Python executor, etc.)
- **TUI**: Improve terminal user interface
  - Info bar with local/SSH environment awareness
  - Consistent keyboard shortcut formatting
  - Notebook preview in artifacts viewer with navigation shortcuts
  - Debounced auto-scroll behavior; todo list accessible at any time
  - Debug block widget showing `[component] STATUS | phase | message` with clear button
  - Log viewer refinements: fix last-line cutoff, content height capping, log belonging
- **Models**: Add `chat_request()` method to LiteLLM adapter for native message-based completions alongside existing text completion API
- **Prompts**: Add ReAct-specific system, planning, and tool prompt templates for orchestrator
- **State**: Add `reactive_mode` flag and tool execution tracking to conversation state
- **Config**: Add `orchestration_mode` setting to project and app config templates
- **Channel Finder**: Add `--delimiter` option to `build-database` for CSV files (#161, @RemiLehe)
- **Build**: Add `uv.lock` for reproducible dependency resolution; consolidate `pytest.ini` into `pyproject.toml`

### Fixed
- **Capabilities**: Remove redundant `found` field from `WriteOperationsOutput` schema — CBORG Haiku omits it from structured output, causing Pydantic validation failures
- **Capabilities**: Clear approval state after approved `channel_write` completes to prevent reactive orchestrator from misinterpreting stale approval flags
- **Reactive**: Classify rate-limit errors and fix E2E assertions
- **E2E**: Rebuild `execution_trace` from graph state instead of Python logging
- **Dependencies**: Move ARIEL dependencies from optional groups to core
- **Open WebUI**: Fix 1-minute waiting issue, side-quest routing through osprey core, multiple code streams combined into one block
- **TUI**: Fix duplicated steps, todo list rendered twice, auto-scroll to middle of todo list, final response style
- **CLI**: Fix component message alignment, filter counter mismatch
- **Streaming**: Fix missing starting tokens in TUI, unpack issue from subgraph streaming, sync call on Python script execution in async function
- **Logging**: Suppress pydantic warnings on LiteLLM model calls

### Changed
- **Prompts**: Refactor all LLM prompts into composable `FrameworkPromptBuilder` subclasses (#163)
  - Rename `get_role_definition()` / `get_task_definition()` → `get_role()` / `get_task()` with deprecation bridges
  - Add `build_dynamic_context(**kwargs)` for runtime context injection (current datetime, user queries, channel mappings)
  - Move `channel_write` and `time_range_parsing` hardcoded runtime prompts (~300 lines) into builder `get_instructions()` + `build_dynamic_context()` methods
  - Capabilities retain runtime context assembly (registry lookups, state access) and delegate prompt composition to builders
  - Extract 4 new capability guide builders: `channel_read`, `channel_write`, `channel_finding_orchestration`, `archiver_retrieval`
  - All framework infrastructure nodes updated to use new builder API
  - Applications can now customize any LLM prompt via subclass overrides without forking capability code
- **Logging**: Replace all Python logger calls with component logger calls using unified `get_logger` system
  - Add explanatory comments to all remaining bare `except: pass` blocks
  - Add debug logging to previously empty except blocks
- **Registry**: Change `REGISTRY` constant to lowercase for naming consistency
- **CI**: Disable auto Claude Code review in PR workflow

## [0.11.1] - 2026-02-13

### Fixed
- **Capabilities**: Replace hardcoded year constraint with training-anchor prompt in time range parsing (#158)
- **Dependencies**: Move `psycopg[binary,pool]` from dev extra to core dependencies so ARIEL logbook search works without `pip install osprey[dev]`; fix macOS CI by bundling libpq via `[binary]`
- **Dependencies**: Pin `claude-agent-sdk==0.1.26` — versions 0.1.27+ bundle a Claude CLI that breaks CBORG API proxy
- **CI**: Fix broken pre-commit config (#154) — update ruff hook to v0.14.3, pre-commit-hooks to v6.0.0, remove invalid `--safe` flag from check-yaml, remove mypy hook (anti-pattern; stays in CI), apply formatting fixes across 180 files

## [0.11.0] - 2026-02-12

### Added
- **Capabilities**: Migrate control capabilities to native Python modules
  - `channel_finding`, `channel_read`, `channel_write`, `archiver_retrieval` moved from Jinja2 templates to `src/osprey/capabilities/`
  - Context classes inlined into capability files (no separate `context_classes.py.j2`)
  - `FrameworkRegistryProvider` registers native capabilities and context classes automatically
- **Services**: Migrate Channel Finder service to native package
  - 48 service files moved from templates to `src/osprey/services/channel_finder/`
  - Default prompt builders added at `src/osprey/prompts/defaults/channel_finder/`
  - Facility-specific prompt overrides via framework prompts
- **CLI**: Add `osprey eject` command for customization escape hatch
  - Copy framework capabilities or services into a project for modification
  - Subcommands: `eject list`, `eject capability`, `eject service` with `--output` and `--include-tests` options
- **CLI**: Add `osprey channel-finder` command with interactive REPL, query, and benchmark modes
- **Registry**: Add shadow warning system for backward compatibility
  - Detects when generated apps override native capabilities without explicit `override_capabilities` config
  - Warns at registration time to guide users toward `osprey eject` workflow
- **CLI**: Add `build-database`, `validate`, and `preview` subcommands to `osprey channel-finder`
  - Database tools migrated from Jinja2 templates to native `osprey.services.channel_finder.tools`
  - Replaces generated `data/tools/` scripts with first-class CLI commands
  - LLM channel namer available as library via `osprey.services.channel_finder.tools.llm_channel_namer`
- **ARIEL**: Add electronic logbook search capability
  - Full-text and semantic search over facility logbooks (OLOG, custom sources)
  - Web interface with dashboard, search, and entry browsing (`osprey ariel web`)
  - CLI commands: `osprey ariel ingest`, `osprey ariel search`, `osprey ariel purge`
  - Deployment support: PostgreSQL and web service templates for `osprey deploy up`
  - Pluggable search modules and enhancement pipeline with registry-based discovery

### Changed
- **Templates**: Simplify `control_assistant` template (~130 → ~40 files)
  - `registry.py.j2` now uses `extend_framework_registry()` with prompt providers only
  - Capabilities, services, and database tools no longer generated from templates

## [0.10.9] - 2026-02-08

### Fixed
- **Registry**: Config-driven provider loading skips unused provider imports (#138)
  - Eliminates ~30s startup delay on air-gapped machines caused by timeout on provider network calls
  - Removes module-level `get_available_models(force_refresh=True)` from `argo.py` and `asksage.py`
- **Argo**: Add structured output handler for Argo provider
  - Argo API does not support the `response_format` parameter; structured output now uses direct httpx calls with JSON schema prompting
  - Includes `_clean_json_response()` to strip markdown fences and fix Python-style booleans
- **Tests**: Fix e2e LLM provider tests broken by config-driven provider filtering
  - Test config's `models` section only listed `openai`, causing all other providers to be skipped
  - Test fixtures now add `models` entries for all available providers
- **Tests**: Remove flaky `gpt-4o` from e2e test matrix (80% pass rate on react_agent due to extra fields in structured output)

### Changed
- **Docs**: Update citation to published APL Machine Learning paper (doi:10.1063/5.0306302)

### Added
- **CLI**: Add `--channel-finder-mode` and `--code-generator` options to `osprey init`
  - Options are included in manifest's `reproducible_command` for full project recreation
- **Capabilities**: Add capability-specific slash commands
  - Unregistered slash commands (e.g., `/beam:diagnostic`, `/verbose`) are forwarded to capabilities
  - `slash_command()` helper and `BaseCapability.slash_command()` method for reading commands
  - Commands are execution-scoped (reset each conversation turn)

## [0.10.8] - 2026-02-02

### Added
- **Skills**: Improve release workflow skill with full step-by-step guidance and CHANGELOG sanitization
- **Generators**: Add pluggable simulation backends for soft IOCs
  - Runtime backend loading from `config.yml` - change behavior without regenerating IOC code
  - Built-in backends: `passthrough` (no-op) and `mock_style` (archiver-like behavior)
  - `ChainedBackend` for composing multiple backends (base + overrides)
  - `SimulationBackend` protocol for custom physics implementations
  - Documentation guide for custom backend development

### Fixed
- **Templates**: Fix `pyproject.toml` template using wrong package search path
  - Template creates `src/<package_name>/` layout but configured `where = ["."]`
  - Changed to `where = ["src"]` so editable installs can find the package
- **Generators**: Fix `config_updater` functions returning wrong type
  - `set_control_system_type()`, `set_epics_gateway_config()`, `update_all_models()`, and `add_capability_react_to_config()` now return `(updated_content, preview)` tuple as expected by CLI callers
- **Channel Finder**: Fix string ChannelNames causing character-by-character iteration
  - MATLAB Middle Layer exports may produce bare strings (e.g., `"SR:DCCT"`) instead of single-element arrays
  - Without the fix, iterating over string produces `['S', 'R', ':', 'D', 'C', 'C', 'T']` instead of `['SR:DCCT']`
  - Normalizes strings to lists in `_extract_channels_from_field()` and `list_channel_names()`
- **Skills**: Fix release workflow skill name to follow `osprey-` naming convention

## [0.10.7] - 2026-01-31

### Added
- **CLI**: Add `osprey migrate` command for project version migration
  - `migrate init` creates manifest for existing projects (retroactive)
  - `migrate check` compares project version against installed OSPREY
  - `migrate run` performs three-way diff analysis and generates merge guidance
  - Classifies files as AUTO_COPY, PRESERVE, MERGE, NEW, or DATA
  - Generates `_migration/` directory with detailed merge prompts for AI-assisted merging
  - Supports exact version recreation via temporary virtualenv
- **Templates**: Add manifest generation during `osprey init`
  - `.osprey-manifest.json` records OSPREY version, template, registry style, and all init options
  - Includes SHA256 checksums for all trackable project files
  - Stores reproducible command string for exact project recreation
- **Assist**: Add `migrate-project` task for AI-assisted migrations
  - Instructions for Claude Code integration with merge workflow
  - Step-by-step guide for handling three-way conflicts
- **Dependencies**: Add `caproto` to core dependencies for soft IOC generation
- **CLI**: Add `osprey generate soft-ioc` command for generating Python soft IOCs
  - Generates caproto-based EPICS soft IOCs from channel databases
  - Supports all 4 channel database types (flat, template, hierarchical, middle_layer)
  - Auto-detects database type, infers PV types and access modes from naming conventions
  - Two simulation backends: `passthrough` (no-op) and `mock_style` (archiver-like behavior)
  - Optional SP/RB pairings file for setpoint-readback tracking with noise
  - Dry-run mode for previewing generation without writing files
  - `--init` flag for interactive simulation config setup (uses channel database from `channel_finder` config)
  - Auto-offers interactive setup when `simulation:` section is missing from config.yml
- **Models**: Add AskSage provider for LLM access (#122)
  - OpenAI-compatible adapter with custom request parameters
  - Supports dynamic model discovery via API
- **Connectors**: Add unit tests for `EPICSArchiverConnector`
  - 26 tests covering connect/disconnect, get_data, error handling, metadata, and factory integration
  - Mock fixtures matching real `archivertools` library format (secs/nanos columns)
- **Config**: Add "Local Simulation" preset to EPICS gateway configuration
  - Select from interactive menu to connect to local soft IOC on localhost:5064
  - Warns if no IOC is detected on the port with instructions to generate/run one
  - Use with `osprey generate soft-ioc` for offline development and testing
- **Tests**: Add unit tests for interactive menu simulation port check
  - 5 tests covering port open/closed detection, timeout handling, and error cases

### Fixed
- **Dependencies**: Pin `claude-agent-sdk>=0.1.26` to fix CBORG proxy beta header incompatibility
- **Security**: Bind docker/podman services to localhost by default (#126)
  - Prevents unintended network exposure when generating server configurations with `osprey deploy up`
  - Use `--expose` option to bind to public interfaces, if firewalling/authentification is set up properly
- **CLI**: Auto-prompt to switch control system mode when configuring EPICS gateway
  - After setting a production gateway (ALS, APS, custom), prompts user to switch from 'mock' to 'epics' mode
  - Handles edge cases: missing config key, other control system types (tango, labview)
- **Connectors**: Fix `EPICSArchiverConnector` timestamp handling for real `archivertools` library
  - Real library returns DataFrame with `secs`/`nanos` columns and RangeIndex
  - Connector now properly converts secs/nanos to DatetimeIndex and removes those columns
  - Fallback preserves backward compatibility for other DataFrame formats
- **Deployment**: Fix `--dev` mode error message showing broken install instructions (#119)
  - Rich markup was stripping `[dev]` from the message due to bracket interpretation
  - Error now correctly shows: `pip install build or pip install -e ".[dev]"`
- **Deployment**: Fix `osprey deploy build` exposing API keys in build config files (#118)
  - `osprey deploy build` was expanding `${VAR}` placeholders to actual values in `build/services/pipelines/config.yml`
  - Now preserves `${VAR}` placeholders; secrets are resolved at container runtime from environment variables
- **Execution**: Fix channel limits database path resolution in subprocess execution
  - Relative paths in `control_system.limits_checking.database_path` now resolve against `project_root`
  - Fixes "Channel limits database not found" error when running Python code locally
- **Connectors**: Fix EPICS connector PV cache to prevent soft IOC crashes
  - Reuse PV objects instead of creating new ones per read
  - Prevents subscription flood that causes caproto race condition (`deque mutated during iteration`)
  - Adds thread-safe locking for PV cache access
- **Config**: Fix control system type update regex to handle comment lines
  - Config files with comments between `control_system:` and `type:` now update correctly

## [0.10.6] - 2026-01-18

### Added
- **CLI**: Add Claude Code skill for release workflow (`osprey claude install release-workflow`)
  - Custom SKILL.md wrapper with quick reference for version files and commands
  - Version consistency check command, pre-release testing steps, tag creation
- **Orchestration**: Context key validation in execution plans
  - Validates that all input key references match actual context keys (existing or from earlier steps)
  - Detects ordering errors where a step references a key created by a later step
  - Triggers replanning (not reclassification) with helpful error context listing available keys
  - New `InvalidContextKeyError` exception for distinguishing from capability hallucination
- **Context**: Store task_objective metadata alongside capability context data (#108)
  - ContextManager now accepts optional `task_objective` parameter in `set_context()`
  - Metadata stored in `_meta` field, stripped before Pydantic validation
  - New helper methods: `get_context_metadata()`, `get_all_context_metadata()`
  - Orchestrator prompt displays task_objective for each available context
  - Enables intelligent context reuse by showing what each context was created for

### Fixed
- **Graph**: Propagate chat history to orchestrator and respond nodes (#111)
  - Orchestrator now receives full conversation context when `task_depends_on_chat_history=True`
  - Enables follow-up queries like "use the same time range" to resolve correctly
  - Chat history formatted with visual separators for clear delineation in prompts
- **Deployment**: Fix Claude Code config path resolution in pipelines container
  - Pipelines container has working directory `/app/` but files are mounted at `/pipelines/`
  - Config file was copied but relative path `claude_generator_config.yml` couldn't be found
  - Now reads `claude_config_path` from config, copies the file, and updates path to absolute `/pipelines/` for pipelines service

## [0.10.5] - 2026-01-16

### Added
- **Testing**: E2E test for LLM channel naming workflow (#103)

### Changed
- **Docs**: Update ALS Assistant reference to published paper (Phys. Rev. Res. **8**, L012017)
- **Models**: Decouple LiteLLM adapter from hardcoded provider checks
  - Providers now declare LiteLLM routing via class attributes (`litellm_prefix`, `is_openai_compatible`)
  - Structured output detection now uses LiteLLM's `supports_response_schema()` function
  - Custom providers can integrate without modifying the adapter layer
  - Maintains backward compatibility with fallback for existing code

### Fixed
- **CI**: Fix deploy-e2e test to actually test PR code by using `--dev` mode
  - Container was installing osprey from PyPI instead of the PR branch
  - Now builds and installs local wheel so the test validates actual changes
- **Channel Finder**: Fix `load_config` not defined error in LLM channel namer (#103)
  - Added `get_config_builder()` and `load_config()` as public API in `osprey.utils.config`
  - Exposed `load_config` in channel finder config utilities
  - Updated channel finder components to use public API instead of internal `_get_config`
- **Deployment**: Fix `--dev` mode failing when osprey is installed from PyPI (#86)
  - Detect site-packages installation and show clear warning about editable mode requirement
  - Add helpful error message when `build` package is missing
  - Add `build` to dev dependencies for wheel building support
- **Models**: Handle Python-style booleans in LLM JSON responses (#102)
  - Some LLM providers (including Argo) return `True`/`False` instead of `true`/`false`
  - `_clean_json_response()` now converts Python-style booleans to JSON-style
- **CLI**: Display full absolute paths for plot files in artifact output (#96)
  - Figure and notebook paths now resolved to absolute before artifact registration
  - Ensures users can directly access generated files from CLI output
- **Packaging**: Include TUI styles.tcss in package data (#97)
  - Textual CSS file was missing from PyPI releases since TUI was introduced in 0.10.0
  - Issue went unnoticed because editable installs (`pip install -e .`) symlink to source

## [0.10.4] - 2026-01-15

### Fixed
- **Dependencies**: Pin aiohttp>=3.10 for litellm compatibility (#87)
  - Fixes `AttributeError: module aiohttp has no attribute ConnectionTimeoutError`
  - `aiohttp.ConnectionTimeoutError` was added in aiohttp 3.10; litellm requires it but doesn't pin the version

## [0.10.3] - 2026-01-14

### Changed
- **CI**: Add E2E tests to GitHub Actions workflow
  - Runs on PRs only (not pushes) to control API costs
  - Skips fork PRs where secrets are unavailable
- **Dependencies**: Move TUI (textual) from optional to base dependencies
  - Removes `[tui]` extras group since textual is now always installed

## [0.10.2] - 2026-01-14

### Added
- **State**: Unified artifact system with `ArtifactType` enum and `register_artifact()` API
  - Single source of truth (`ui_artifacts`) for all artifact types: IMAGE, NOTEBOOK, COMMAND, HTML, FILE
  - Legacy methods (`register_figure`, `register_notebook`, `register_command`) delegate to new API
  - `populate_legacy_fields_from_artifacts()` helper for backward compatibility at finalization
- **TUI**: Artifact gallery and viewer widgets for interactive artifact browsing
  - ArtifactGallery with keyboard navigation (Ctrl+a focus, j/k navigate, Enter view, o open external)
  - ArtifactViewer modal with type-specific details and actions (copy path, open in system app)
  - Native image rendering via textual-image (Sixel for iTerm2/WezTerm, Kitty Graphics Protocol)
  - New/seen tracking with [NEW] badges for artifacts from current turn

### Changed
- **Tooling**: Consolidated formatting/linting to Ruff, removed Black and Isort (#80)
  - Ruff now handles both linting and formatting as a single tool
  - Updated scripts, docs, and templates to reference only Ruff
- **Capabilities**: Python capability uses unified `register_artifact()` API directly
  - Clean single-accumulation pattern for figures and notebooks
  - Legacy fields populated at finalization rather than registration
- **CLI**: Modernized artifact display to use unified `ui_artifacts` registry
  - Single `_extract_artifacts_for_cli()` replaces three legacy extraction methods
  - Supports all artifact types: IMAGE, NOTEBOOK, COMMAND, HTML, FILE
  - Grouped display with type-specific formatting and icons

### Fixed
- **Gateway**: `/chat` without arguments no longer triggers graph execution
  - Displays available capabilities table correctly, then returns immediately
  - New check for locally-handled commands with no remaining message
  - CLI handles state-only updates with no agent_state gracefully
- **Orchestrator**: Use descriptive context keys to prevent incorrect time range reuse (#90)
  - Similar time ranges (e.g., 12/5-12/10 vs 12/5-12/8) no longer incorrectly reuse old context
  - Context keys now encode actual dates (tr_MMDD_MMDD format) for proper comparison
- **Approval**: Fix KeyError when optional approval config keys are omitted (#79)
  - Logger now uses initialized config object instead of raw dict keys
- **Templates**: Include deployment infrastructure config for all templates (#85)
  - Fixes `osprey deploy up` failures for hello_world_weather template
  - Jupyter kernel templates now render correctly with execution.modes section
- **CLI**: Restrict `load_dotenv()` search to current directory only (#95)
  - Prevents python-dotenv from parsing shell config files in parent directories
  - Fixes warnings when users have `~/.env` as a Korn shell configuration file

## [0.10.1] - 2026-01-09

### Added
- **State**: Session state persistence for user preferences and mode tracking
  - New `session_state` field in AgentState with custom merge reducer
  - Enables direct chat mode and other session-level settings to persist across conversation turns
- **Infrastructure**: Direct chat mode routing and message handling
  - Router detects direct chat mode and routes directly to capability
  - Gateway preserves message history in direct chat mode
  - Validates capability supports direct_chat_enabled before routing
- **Capabilities**: Context management tools for ReAct agents
  - read_context, list_available_context, save_result_to_context
  - remove_context, clear_context_type, get_context_summary
  - Enables agents to manage accumulated context during direct chat
- **Capabilities**: StateManager capability for interactive state management
  - Natural language interface for context and agent settings
  - State inspection tools: session info, execution status, capability list, settings
  - State modification tools: clear session, modify agent settings
  - Registered as framework-level capability (/chat:state_manager)
- **CLI**: Direct chat mode for conversational interaction with capabilities
  - `/chat:<capability>` enters direct chat mode
  - `/chat` lists available direct-chat capabilities
  - `/exit` returns to normal mode (adds transition marker for context)
  - Dynamic prompt shows current mode (normal vs capability name)
  - Quieter logging during direct chat for cleaner experience
- **Generators**: Direct chat mode support in MCP capability generator
  - Generated capabilities have direct_chat_enabled=True by default
  - Adds context management tools when in direct chat mode
  - Handles both orchestrated and direct chat execution modes
  - Updated docstrings with direct chat usage examples
- **Models**: LangChain model factory for full LangGraph ReAct agent support
  - `get_langchain_model()` creates BaseChatModel instances from osprey config
  - Supports all 8 providers: anthropic, openai, google, ollama, cborg, vllm, stanford, argo
  - Native integration with `create_react_agent` and other LangGraph workflows
  - Automatic configuration loading from osprey's config system
- **Models**: New vLLM provider adapter for high-throughput local inference
  - Uses LiteLLM's OpenAI-compatible interface
  - Auto-detects served models via `/models` endpoint
  - Supports structured outputs with json_schema
- **Models**: Direct Ollama API for thinking models (bypasses LiteLLM bug #15463)
  - gpt-oss and other thinking models now work correctly
  - Automatic minimum token allocation (100) for thinking phase
- **Tests**: Consolidated E2E test suite for LLM providers (`tests/e2e/test_llm_providers.py`)
  - Provider × model × task matrix approach (anthropic, openai, google, cborg, ollama, vllm)
  - Tests basic completion, structured output (Pydantic), and ReAct agent workflows
  - Auto-skips unavailable providers/models based on environment
  - Graceful handling of API quota/rate limit errors (skips with warning instead of failing)
- **Documentation**: Direct chat mode user and developer documentation
  - CLI Reference: `/chat` and `/exit` commands, Direct Chat Mode section with examples
  - Gateway Architecture: Direct chat mode handling, message history preservation, GatewayResult fields
  - Classification and Routing: Router priority with direct chat bypass
  - Building First Capability: `direct_chat_enabled` attribute and tip box

### Changed
- **Capabilities**: Support direct chat execution mode in capability decorator
  - Creates synthetic execution step when no execution plan exists
  - Skips step progression in direct chat mode
  - Changed classifier missing log from warning to debug (expected for direct-chat-only capabilities)
- **Logging**: Reduced verbose third-party logging for cleaner CLI output
  - Added quiet_logging() context manager for temporary log suppression
  - Suppressed LiteLLM debug messages
- **Models**: Migrated all LLM provider implementations to LiteLLM unified interface (#23)
  - Replaced ~2,200 lines of custom provider code with ~700 lines using LiteLLM adapter
  - All 8 providers (anthropic, google, openai, ollama, cborg, stanford, argo, vllm) now use LiteLLM
  - Preserved extended thinking, structured outputs, and health check functionality
  - Access to 100+ providers through LiteLLM

### Removed
- **Models**: Removed unused `get_model()` function and `factory.py` module
  - The function was dead code (never called anywhere in the codebase)
  - All model access now goes through `get_chat_completion()`

### Fixed
- **Code Generation**: Fix `${VAR}` environment variable expansion in `claude_code_generator`
- **CBORG Provider**: Add missing `temperature` parameter to API calls
  - Fixes non-deterministic code generation behavior causing intermittent test failures
  - Both regular text completion and structured output paths now respect temperature setting
- **Code Generation**: Add simplicity guidance to prevent over-engineered solutions
  - LLM now prefers direct context usage over building complex systems to fetch data
- **Documentation**: Fixed workflow file references to use correct `@src/osprey/workflows/` path for copy-paste into Claude Code and Cursor
- **Gateway**: Mode switch handling for direct chat entry/exit
  - Use `update_state()` for mode switches instead of `ainvoke()` to avoid full graph execution
  - Correct field names (`planning_execution_plan`, `planning_current_step_index`)
  - New `is_state_only_update` flag signals callers to use proper update method
  - New `exit_interface` flag for `/exit` outside direct chat mode
- **Commands**: New `gateway_handled` flag ensures state-affecting commands route through gateway
  - /exit, /planning, /approval, /task, /caps, /chat marked as gateway_handled
  - Ensures consistent behavior across all interfaces (CLI, OpenWebUI, API)
- **CLI**: Proper routing for gateway_handled vs local commands
  - Local commands (/help, /clear) handled directly for instant response
  - State commands route through gateway for consistent state management
- **Router**: Suppress routing logs during state-only evaluations
  - Mode switches no longer produce confusing "routing to task extraction" logs
  - Uses `execution_start_time` to detect active vs state-only execution
- **Capabilities**: Context tool changes now persist to LangGraph state
  - State manager and MCP capabilities return `capability_context_data` in state updates
  - Fixes context save/remove operations having no effect in direct chat mode

## [0.10.0] - 2026-01-08

### Added
- **TUI**: New Textual-based Terminal User Interface (`osprey chat --tui`)
  - Full-screen terminal experience with real-time streaming of agent responses
  - Step-by-step visualization: Task Extraction → Classification → Orchestration → Execution
  - Welcome screen with ASCII banner and quick-start guidance
  - Theme support with 15+ built-in themes and interactive theme picker (Ctrl+T)
  - Command palette for quick access to all actions (Ctrl+P)
  - Slash commands support (`/exit`, `/caps:on`, `/caps:off`, etc.)
  - Query history navigation with up/down arrows
  - Content viewer for prompts and responses with multi-tab support and markdown rendering
  - Log viewer with live updates for debugging
  - Todo list visualization showing agent planning progress
  - Keyboard shortcuts for navigation (scroll, focus input, toggle help)
  - Double Ctrl+C to quit for safety
  - ~5,500 lines of new code across 17 files in `src/osprey/interfaces/tui/`
- **Logging**: Enhanced logging system with TUI data extraction support
  - New `_build_extra()` method embeds streaming event data into Python logs
  - Enables TUI to receive all data through a single logging source
  - Added `QueueLogHandler` for async log processing in TUI
- **CLI**: New `osprey tasks` command for browsing AI assistant tasks
  - `osprey tasks` - Interactive task browser (default)
  - `osprey tasks list` - List all available tasks
  - `osprey tasks show <task>` - Print task instructions to stdout
  - `osprey tasks copy <task>` - Copy task to project's `.ai-tasks/` directory
  - `osprey tasks path <task>` - Print path to task's instructions file
- **CLI**: New `osprey claude` command for Claude Code skill management
  - `osprey claude install <task>` - Install a task as a Claude Code skill
  - `osprey claude list` - List installed and available skills
- **Assist System**: General-purpose architecture for AI coding assistant integrations
  - Tool-agnostic task instructions in `src/osprey/assist/tasks/`
  - Tool-specific wrappers in `src/osprey/assist/integrations/`
  - Pre-commit task for validating code before commits
  - Migration task for upgrading downstream OSPREY projects
- **Tests**: Comprehensive tests for `tasks_cmd.py` and `claude_cmd.py`

### Changed
- **CLI**: `osprey chat` now supports `--tui` flag to launch the TUI interface
  - Default behavior unchanged (CLI interface)
  - TUI requires textual package: `pip install osprey-framework[tui]`
- **CLI**: Deprecated `osprey workflows` command (use `osprey tasks` instead)
  - Command still works for backward compatibility but shows deprecation warning
- **Code Generation**: Enhanced `claude_code_generator` with environment variable support
  - Config template now supports custom environment variables via `claude_generator_config.yml`
  - Added ARGO endpoint configuration to template
  - Fixed default URL to use correct localhost link
- **Documentation**: Updated workflow references to use new command structure
  - `osprey tasks list` for browsing tasks
  - `osprey claude install <task>` for installing Claude Code skills
- **Documentation**: Updated release-workflow instructions with accurate test counts
  - Unit tests: ~1850 tests (~1-2 min) instead of outdated ~370-380 tests (~5s)
  - E2E tests: ~32 tests (~10-12 min) instead of outdated ~5 tests (~2-3 min)

### Removed
- **Workflows**: Removed duplicate workflow files from `src/osprey/workflows/`
  - Content consolidated into `src/osprey/assist/tasks/{name}/instructions.md`
  - Only `README.md` deprecation notice remains in workflows directory

## [0.9.10] - 2025-01-03

### Fixed
- **Channel Finder**: Initialize `query_splitting` attribute in HierarchicalPipeline
  - Fixes `AttributeError: 'HierarchicalPipeline' object has no attribute 'query_splitting'`

### Added
- **Channel Finder**: Optional `query_splitting` parameter for hierarchical and middle_layer pipelines
  - Disable query splitting for facility-specific terminology that shouldn't be split
  - Enabled by default for backward compatibility

### Changed
- **Channel Finder Prompts**: Modularized prompt structure across all pipelines
  - Split `system.py` into `facility_description.py` (REQUIRED) and `matching_rules.py` (OPTIONAL)
  - Users now edit `facility_description.py` for facility-specific content
  - `system.py` auto-combines modules (no manual editing needed)
  - Query splitter prompts now accept `facility_name` parameter
- **Benchmark Dataset**: Renamed `in_context_main.json` to `in_context_benchmark.json` for consistency
- **Documentation**: Updated control assistant tutorials for modular prompt structure
  - Part 1: Updated directory structure with new prompt file layout
  - Part 2: Added cross-references to prompt customization section
  - Part 4: Expanded channel finder prompt customization with step-by-step guidance

### Added
- **Channel Finder**: Added explicit detection functionality to channel finder service
  - New `explicit_detection.py` prompt module for detecting explicit channel names, PV names, and IOC names
  - Updated `BasePipeline` with `build_result()` helper method for constructing pipeline results
  - Enhanced all pipeline implementations (hierarchical, in-context, middle layer) to use explicit detection
  - Added unit tests for explicit detection prompt and `build_result()` method
  - Updated e2e tests to verify explicit detection behavior
  - Configuration updates to include explicit detection in pipeline workflows
- **Tests**: `test_memory_capability.py`: 32 tests for memory operations, context, exceptions, and helper functions (37.7% → 62.4% coverage)
- **Tests**: `test_logging.py`: 27 tests for API call logging, caller info extraction, and file creation (29.1% → 55.7% coverage)
- **Tests**: `test_models.py` (generators): 21 tests for capability generation Pydantic models (0% → 100% coverage)
- **Tests**: `test_models_utilities.py` (python_executor): 39 tests for execution error handling, notebook tracking, and utility functions
- **Tests**: `test_models.py` (memory_storage): 13 tests for memory content formatting and validation (0% → 100% coverage)
- **Tests**: `test_storage_manager.py`: 22 tests for memory persistence, file operations, and entry management (24.1% → 72.4% coverage)
- **Tests**: `test_memory_provider.py`: 23 tests for memory data source integration and prompt formatting (32.2% → 94.9% coverage)
- **Tests**: `test_providers_argo.py`: 27 tests for ARGO provider adapter (18.6% → 54.8% coverage)
- **Tests**: `test_providers_ollama.py`: 31 tests for Ollama provider with fallback logic (24.2% → 96.0% coverage)
- **Tests**: `test_providers_anthropic.py`: 27 tests for Anthropic provider metadata, model creation, and health checks (23.5% → 50.0% coverage)
- **Tests**: `test_completion.py`: 28 tests for TypedDict conversion and proxy validation (30.9% → 58.0% coverage)
- **Tests**: `test_logging.py`: 19 tests for API call context and result sanitization (13.3% → 29.1% coverage)
- **Tests**: `test_respond_node.py`: 26 tests for response generation, context gathering, and mode determination (37.7% → 72.1% coverage, infrastructure module 54.7% → 58.4%)
- **Tests**: `test_task_extraction_node.py`: 25 tests for task extraction, data source integration, and error classification (33.0% → 62.1% coverage, infrastructure module 52.1% → 54.7%)
- **Tests**: `test_error_node.py`: 29 tests for error response generation and context handling (33.6% → 91.8% coverage, infrastructure module 45.2% → 52.1%)
- **Tests**: Expanded infrastructure and models tests - 40 new tests for error classification, retry policies, and helper functions (infrastructure module 37.2% → 45.2%, overall 45.8% → 46.4%)
- **Tests**: Added comprehensive tests for CLI and deployment modules (coverage expansion)
  - `test_preview_styles.py`: 23 tests for theme preview and color display functionality (0% → 88.1% coverage)
  - `test_main.py`: 23 tests for CLI entry point and lazy command loading (28.6% → 95.2% coverage)
  - `test_health_cmd.py`: 38 tests for health checks and environment diagnostics (0% → 69.6% coverage)
  - `test_loader.py`: 55 tests for YAML loading, imports, and parameter management (0% → 86.6% coverage)
  - `test_chat_cmd.py`: 15 tests for command execution and output formatting
  - `test_export_config_cmd.py`: 16 tests for deprecation warnings and format options
  - `test_deploy_cmd.py`: 23 tests for deployment actions (up/down/restart/status/build/clean/rebuild)
  - `test_registry_cmd.py`: 22 tests for registry display functions
  - `test_config_cmd.py`: 23 tests for config subcommands (show/export/set-control-system/set-epics-gateway/set-models)
  - `test_remove_cmd.py`: 16 tests for capability removal and backups
  - `test_generate_cmd.py`: 37 tests for code generation commands (capability/mcp-server/claude-config)
  - `test_orchestration_node.py`: 12 tests for execution planning validation and error handling
  - `test_classification_node.py`: 13 tests for capability classification structure and error handling
  - Fixed missing `Dict` import in `scripts/analyze_test_coverage.py`
  - Renamed `analyze_coverage.py` → `analyze_test_coverage.py` for clarity

### Fixed
- **CLI**: Fixed broken imports in `config_cmd.py`
  - Changed `update_control_system_type` → `set_control_system_type` (correct function name)
  - Changed `update_epics_gateway` → `set_epics_gateway_config` (correct function name)
  - Updated function calls to handle return values correctly (both functions return tuple of new_content, preview)

### Changed
- **Control Assistant**: Write access now enabled by default in control assistant template (`writes_enabled: true` for mock connector)
  - Simplifies tutorial experience - users can test write operations immediately with mock connector
  - Production deployments should carefully review hardware implications before enabling writes
- **License**: Added explicit "BSD 3-Clause License" header to LICENSE.txt for clarity

### Documentation
- Updated Hello World tutorial to reflect current weather capability implementation with natural language location handling
- Fixed version picker showing non-existent versioned directories causing 404 errors
  - Updated docs workflow to only list actually deployed versions (stable and latest/development)
  - Removed all individual version tag entries from versions.json until versioned directories are implemented
- Fixed double slash typos in image paths causing 404 errors on GitHub Pages for in-context and hierarchical channel finder CLI screenshots
- Added "Viewing Exported Workflows" section to AI-assisted development guide showing example output of exported workflow files
- Removed v0.9.2+ migration guide (no longer needed as framework has fully transitioned to instance method pattern)
  - Cleaned up all cross-references to migration guide across documentation
  - Streamlined architecture overview sections in main index and developer guides
  - Updated main index diagram from workflow to architecture overview
- Added academic reference (Hellert et al. 2025, arXiv:2512.18779) for semantic channel finding theoretical framework

## [0.9.9] - 2025-12-22

### Fixed
- **Testing**: Fixed middle layer benchmark test assertion to use `queries_evaluated` instead of `total_queries` field from benchmark results

### Changed
- **Workflows**: Moved AI workflow files from `docs/workflows/` to `src/osprey/workflows/` for package bundling
  - Workflows now distributed with installed package
  - Enables version-locked workflow documentation
- **Documentation**: Updated workflow references to use `@osprey-workflows/` path
  - Added workflow export instructions to AI-assisted development guide
  - Updated all @-mention examples across documentation

### Added
- **CLI**: New `osprey workflows` command to export AI workflow files
  - `osprey workflows export` - Export workflows to local directory (default: ./osprey-workflows/)
  - `osprey workflows list` - List all available workflow files
  - Interactive menu integration for easy access
- **Documentation - AI Workflows**: Channel Finder workflow guides for AI-assisted development
  - New workflow files: pipeline selection guide and database builder guide with AI prompts and code references
  - Workflow cards in AI-assisted development guide linking to pipeline selection and database building workflows
  - AI-assisted workflow dropdowns in tutorial "Build Your Database" sections for all three pipelines (in-context, hierarchical, middle layer)
  - AI-assisted pipeline selection dropdown before pipeline tab-set in tutorial
  - Enhanced workflows with guidance for AI assistants to read database format code and examples before giving advice
  - Code reference sections showing AI how to use source files for evidence-based recommendations
- **Documentation**: Comprehensive middle layer pipeline guide in Sphinx docs
  - Complete tutorial with architecture comparison and usage examples
  - CLI screenshots and integration examples
  - End-to-end benchmark tests validating complete integration
- **Channel Finder - Sample Data**: Middle layer database and benchmarks
  - 2,033-channel sample database covering 3 systems (SR, BR, BTS)
  - 20 device families with full metadata
  - 35-query benchmark dataset (20% coverage ratio - best of all pipelines)
  - Realistic accelerator physics context
- **Channel Finder - Tools**: Middle layer support across all CLI tools
  - Database preview tool with tree visualization for functional hierarchy
  - CLI query interface with middle_layer pipeline support
  - Benchmark runner with middle_layer dataset support
- **Templates - Channel Finder**: Middle layer configuration support
  - Conditional config generation for middle_layer pipeline
  - Dynamic AVAILABLE_PIPELINES list based on enabled pipelines
  - Database and benchmark paths auto-configured
- **Channel Finder - Middle Layer Testing**: Comprehensive tool and utility tests
  - 480 lines of tests covering all database query tools
  - Tests for prompt loader with middle_layer support
  - Tests for MML converter utility enhancements
- **Channel Finder - Middle Layer**: React agent prompts for functional navigation
  - Query splitter prompt for decomposing complex queries
  - System prompt with database exploration tools
- **Registry Manager**: Silent initialization mode for clean CLI output
  - Suppress INFO/DEBUG logging during initialization when `silent=True`
  - Useful for CLI tools that need clean output without verbose registry logs
- **Channel Finder: Middle Layer Pipeline**: Complete React agent-based channel finder pipeline for MATLAB Middle Layer (MML) databases with System→Family→Field hierarchy; includes MiddleLayerDatabase with O(1) validation and device/sector filtering, MiddleLayerPipeline with 5 database query tools (list_systems, list_families, inspect_fields, list_channel_names, get_common_names), MMLConverter utility for converting Python MML exports to JSON, optional _description fields at all levels for enhanced LLM guidance, comprehensive test suite (14 tests), sample database, and complete documentation

### Changed
- **CLI - Project Initialization**: Enhanced channel finder selection
  - Added middle_layer option to interactive menu
  - Changed default from "both" to "all" (now includes all three pipelines)
  - Updated descriptions for clarity: in_context (<200 channels), hierarchical (pattern-based), middle_layer (functional)
- **Channel Finder - Middle Layer Pipeline**: Migrated from Pydantic-AI to LangGraph
  - Now uses LangGraph's create_react_agent for improved agent behavior
  - Converted tools from Pydantic-AI format to LangChain StructuredTool
  - Enhanced structured output with ChannelSearchResult model
  - Better error handling and agent state management

### Fixed
- **Build Scripts**: Removed trailing whitespace from configuration and script files
- **Testing: Channel Finder test path correction**: Fixed incorrect database path in `test_multiple_direct_signals_fix.py` to point to correct example database location
- **Channel Finder: Multiple direct signal selection**: Fixed leaf node detection to properly handle multiple direct signals (e.g., "status and heartbeat") selected together at optional levels
- **Channel Finder: Optional levels LLM awareness**: Enhanced database descriptions and prompts to better distinguish direct signals from subdevice-specific signals
- **Channel Finder: Separator overrides**: Fixed `build_channels_from_selections()` to respect `_separator` metadata from tree nodes via new `_collect_separator_overrides()` method
- **Channel Finder: Separator overrides with expanded instances**: Fixed `_collect_separator_overrides()` navigation through expanded instance names (e.g., `CH-1`) by checking `_expansion` definitions to find container nodes
- **Channel Finder: Navigation through expanded instances**: Fixed `_navigate_to_node()` and `_extract_tree_options()` to properly handle expanded instances at optional levels - base containers with `_expansion` no longer appear as selectable options, and navigation through expanded instance names works correctly

### Removed
- **Documentation**: Obsolete markdown tutorials for middle layer
  - Content migrated to Sphinx documentation (control-assistant-part2-channel-finder.rst)

## [0.9.8] - 2025-12-19

### Added
- **Testing: Hello World Weather template coverage**: Added comprehensive unit test suite for hello_world_weather template including mock weather API validation, response formatting, and error handling scenarios
- **Hello World Weather: LLM-based location extraction**: Added structured output parser using LLM to extract locations from natural language queries, replacing simple string matching with intelligent parsing that handles nicknames, abbreviations, and defaults to "local" when no location is specified
- **Documentation Version Switcher**: PyData Sphinx Theme version switcher for GitHub Pages with multi-version documentation support; workflow dynamically generates `versions.json` from git tags and preserves historical versions in separate directories (e.g., `/v0.9.7/`, `/latest/`)
- **Developer Workflows System**: New `docs/workflows/` directory with 10 comprehensive workflow guides (pre-merge cleanup, commit organization, release process, testing strategy, AI code review, docstrings, comments, documentation updates) featuring YAML frontmatter metadata and AI assistant integration prompts
- **Custom Sphinx Extension**: `workflow_autodoc.py` extension with `.. workflow-summary::` and `.. workflow-list::` directives for auto-documenting workflow files from markdown with YAML frontmatter, including custom CSS styling
- **Testing: Workflow autodoc extension**: Comprehensive test suite for custom Sphinx extension including frontmatter parsing, directive rendering, and integration tests with actual workflow files
- **Contributing Guide**: Professional `CONTRIBUTING.md` with quick start guide, branch naming conventions, code standards summary, and links to comprehensive documentation
- **CI/CD Infrastructure**: Comprehensive GitHub Actions CI pipeline with parallel jobs for testing (Python 3.11 & 3.12, Ubuntu & macOS), linting (Ruff), type checking (mypy), documentation builds, and package validation
- **Pre-commit Hooks**: `.pre-commit-config.yaml` with Ruff linting/formatting, file quality checks (trailing whitespace, merge conflicts, large files), and optional mypy type checking
- **Dependabot Configuration**: Automated weekly dependency updates for Python packages and GitHub Actions with intelligent grouping (development, Sphinx, LangChain dependencies)
- **Release Automation**: `.github/workflows/release.yml` for automated PyPI publishing using trusted publishing (OIDC), version verification, and optional TestPyPI deployment
- **Pre-merge Check Script**: `scripts/premerge_check.sh` automated scanning for debug code, commented code, hardcoded secrets, missing CHANGELOG entries, incomplete docstrings, and unlinked TODOs
- **Code Coverage Reporting**: Codecov integration in CI pipeline with coverage reports uploaded for Python 3.11 Ubuntu runs
- **Status Badges**: README.md badges for CI status, documentation, code coverage, PyPI version, Python version support, and license

### Changed
- **Code Quality: Comprehensive Linting Cleanup**: Fixed multiple code quality issues across 47 files - B904 exception chaining (30 instances), E722 bare except clauses (5 instances), B007 unused loop variables (4 instances), formatting issues; removed B904 from ruff ignore list and added intentional per-file ignores for test files and example scripts; all changes verified with full test suite (968 unit + 15 e2e tests passing)
- **Code Formatting**: Applied automated Ruff formatting across codebase - modernized type hints to Python 3.10+ style (`Optional[T]` → `T | None`, `List[T]` → `list[T]`), normalized quotes, cleaned whitespace, and removed unused imports; no functional changes
- **Documentation Workflows**: Migrated workflow files from `docs/resources/other/` to `docs/workflows/` with updated references throughout; workflows now feature consistent YAML frontmatter for machine parsing and AI integration
- **Documentation Structure**: Reorganized contributing documentation from placeholder to comprehensive guide with 6 dedicated sections (Getting Started, Git & GitHub, Code Standards, Developer Workflows, AI-Assisted Development, Community Guidelines) using sphinx-design cards and grids
- **Contributing Guide**: Restructured `docs/source/contributing/index.rst` from placeholder to comprehensive 400+ line guide with learning paths, AI integration examples, workflow categories, and automation tools documentation
- **CI Pipeline**: Enhanced documentation job to create preview artifacts for pull requests with 7-day retention; added clear separation between CI checks (`.github/workflows/ci.yml`) and deployment (`.github/workflows/docs.yml`)
- **Development Dependencies**: Added `pytest-cov` to `[dev]` optional dependencies in `pyproject.toml` for code coverage reporting in CI pipeline
- **Hello World Weather: Mock API simplification**: Refactored mock weather API to accept any location string and generate random weather data, removing hardcoded city list and enabling flexible location support for tutorial demonstrations
- **Documentation: Citation update**: Updated paper citation to reflect new title "Osprey: Production-Ready Agentic AI for Safety-Critical Control Systems"
- **Documentation: Framework name cleanup**: Replaced all remaining references to "Alpha Berkeley Framework" with "Osprey Framework" across README, templates, documentation, and test files
- **Testing: E2E hello_world_weather tutorial test**: Enhanced test to exercise both weather AND Python capabilities with a multi-step query that validates configuration defaults, context passing, and code generation/execution workflows
- **Hello World Weather Template**: Enhanced mock weather API with improved error handling and response formatting; updated tutorial documentation for better clarity

### Fixed
- **Configuration: Execution defaults for Python code generation**: Added missing code generator configuration defaults to `ConfigBuilder._get_execution_defaults()`. Now includes `code_generator: "basic"` and corresponding generators configuration, preventing "Unknown provider: None" errors when using Python capabilities in projects with minimal configuration
- **Hello World Weather Template**: Fixed template conditional to include execution infrastructure configuration while excluding only EPICS-specific settings, ensuring Python code generation works out-of-the-box
- **Testing: CI workflow autodoc test collection**: Fixed `ModuleNotFoundError: No module named 'sphinx'` in CI by adding `pytest.importorskip` to `tests/documentation/test_workflow_autodoc.py`; Sphinx is only required for documentation builds and is not part of `[dev]` dependencies, so workflow autodoc tests now gracefully skip when Sphinx is unavailable

### Removed
- **Documentation: Local server launcher**: Removed `docs/launch_docs.py` script; users should use standard Sphinx commands (`make html` and `python -m http.server`) for local documentation builds and serving

## [0.9.7] - 2025-12-14

### Added
- **CLI: Model Configuration Command**: New `osprey config set-models` command to update all model configurations at once with interactive or direct mode
- **Channel Finder: API call context tracking**: Added context tracking to channel finder pipeline for better API call logging and debugging

### Changed
- **Documentation: Python version requirement consistency**: Updated all documentation and templates to consistently specify "Python 3.11+" instead of "Python 3.11", matching the pyproject.toml requirement of `>=3.11`
- **Channel Finder Service**: Improved configuration validation with clearer error messages when channel_finder model is not configured
- **Control Assistant Template: Use Osprey's completion module**: Removed duplicate `completion.py` implementation from channel finder service; now uses `osprey.models.completion` for consistency and maintainability

### Fixed
- **Channel Finder: Optional levels navigation**: Fixed bug where direct signals incorrectly appeared as subdevice options in optional hierarchy levels. The system now correctly distinguishes between container nodes (which belong at the current optional level) and leaf/terminal nodes (which belong to the next level). Also fixed `build_channels_from_selections()` to handle missing optional levels and apply automatic separator cleanup (removes `::` and trailing separators).
- **Hello World Weather Template**: Added service configuration (container runtime, deployed services) to prevent `'services/docker-compose.yml.j2' not found` error when following installation guide
- **Channel Write Capability**: Removed `verification_levels` field from approval `analysis_details` that incorrectly called `_get_verification_config()` method before connector initialization
- **Testing**: Added integration test for channel_write approval workflow to catch capability-approval interaction bugs
- **Testing: Channel Finder registration tests**: Updated test mocks to include `channel_finder` model configuration in the mocked `configurable` dict, fixing tests broken by stricter validation introduced in commit 5834de3
- **Testing: E2E workflow test**: Updated `test_hello_world_template_generates_correctly` to expect services directory and deployment configuration, matching current template structure
- **Testing: E2E benchmark tests**: Fixed registry initialization in `test_channel_finder_benchmarks.py` by calling `initialize_registry()` before creating `BenchmarkRunner` to prevent "Registry not initialized" errors
- **Code Quality**: Pre-merge cleanup - removed unused imports, applied black formatting to 13 files, and documented DEBUG and CONFIG_FILE environment variables in env.example

## [0.9.6] - 2025-12-06

### Added
- **Control Assistant Template: Custom Task Extraction Prompt**: Added control-system-specific task extraction prompt builder that replaces framework defaults with domain-specific examples
  - 14 control system examples covering channel references, temporal context, write operations, and visualization requests
  - Unit test suite verifying custom prompt usage without LLM invocation
  - Documentation in Part 4 tutorial explaining single-point-of-failure importance
- **Channel Finder: Enhanced Database Preview Tool**: Flexible display options for better hierarchy visibility
  - `--depth N` parameter to control tree depth display (default: 3, -1 for unlimited)
  - `--max-items N` parameter to limit items shown per level (default: 10, -1 for unlimited)
  - `--sections` parameter with modular output sections: tree, stats, breakdown, samples, all
  - `--path PATH` parameter to preview any database file directly without modifying config
  - `--focus PATH` parameter to zoom into specific hierarchy branches
  - New `stats` section showing unique value counts at each hierarchy level
  - New `breakdown` section showing channel count breakdown by path
  - New `samples` section showing random sample channel names
  - Backwards compatible `--full` flag support
  - Comprehensive unit tests covering all preview features and edge cases

### Changed
- **Channel Finder: Preview Tool Default Depth**: Default tree display depth increased from 2 to 3 levels for better visibility

### Fixed
- **MCP Server Template: Dynamic timestamps instead of hardcoded dates**: Fixed MCP server generation template to use current UTC timestamps instead of hardcoded November 15, 2025 dates. Prevents e2e test failures due to stale mock data and ensures demo servers return realistic "current" weather data.
- **Tests: Channel Finder unit test updates**: Updated channel finder test files for compatibility with hierarchical database changes (optional levels, custom separators)
- **Tests: Registry mock cleanup and fixture name collisions**: Fixed 7 registry isolation test failures caused by session-level registry mock pollution from capability tests, renamed conflicting test fixtures to prevent pytest naming collisions
- **Python Executor: Context File Creation for Pre-Approval Notebooks**: Fixed timing issue where `context.json` was not created until execution, causing warnings and test failures when approval was required. Context is now saved immediately when creating pre-approval, syntax error, and static analysis failure notebooks.
- **Code Quality: Pre-merge cleanup**: Removed unused imports and applied code formatting standards (black + isort) across entire codebase for consistency
- **Documentation: Fixed RST docstring formatting**: Corrected docstring syntax in `BaseInfrastructureNode.get_current_task()` to use proper RST code block notation (eliminates Sphinx warnings)

### Added
- **Hierarchical Channel Finder: Custom Separator Overrides**: Per-node control of channel name separators
  - New `_separator` metadata field overrides default separators from naming pattern
  - Solves EPICS naming conventions with mixed delimiters (e.g., `:` for subdevices, `_` for suffixes, `.` for legacy subsystems)
  - Backward compatible: nodes without `_separator` use pattern defaults
  - Documentation: New "Custom Separators" tab in Advanced Hierarchy Patterns section
- **Hierarchical Channel Finder: Automatic Leaf Detection**: Eliminates verbose `_is_leaf` markers for childless nodes
  - Nodes without children are automatically detected as leaves (no explicit marker needed)
  - `_is_leaf` now only required for nodes that have children but are also complete channels
  - Reduces verbosity in database definitions (e.g., RB/SP readback/setpoint nodes)
  - Backward compatible: explicit `_is_leaf` markers still work (take precedence)
  - Updated all examples and documentation to reflect cleaner syntax
  - Test coverage: 2 new tests for automatic leaf detection functionality
- **Channel Finder: Pluggable Pipeline and Database System**: Registration pattern for custom implementations
  - `register_pipeline()` and `register_database()` methods for extending channel finder
  - Discovery API: `list_available_pipelines()` and `list_available_databases()`
  - Config-driven selection without modifying framework code
  - Examples for RAG pipeline and PostgreSQL database implementations
- **Hierarchical Channel Finder: Flexible Naming Configuration**: Navigation-only levels and decoupled naming
  - Naming pattern can reference subset of hierarchy levels (not all required in pattern)
  - New `_channel_part` field decouples tree keys from naming components
  - Enables semantic tree organization with PV names at leaf (JLab CEBAF pattern)
  - Enables friendly navigation with technical naming ("Magnets" → "MAG")
  - Backward compatible: existing databases work unchanged
  - Example database: `hierarchical_jlab_style.json` demonstrating both features
  - Test coverage: 18 new tests for flexible naming functionality

#### Configuration Management
- **EPICS Gateway Presets**: Built-in configurations for APS and ALS facilities
  - APS: pvgatemain1.aps4.anl.gov:5064 (read-only and write-access)
  - ALS: cagw-alsdmz.als.lbl.gov:5064 (read-only), :5084 (write-access)
  - Custom facility support with interactive configuration
- **Configuration Management API**: Programmatic control system and EPICS gateway configuration
  - `get_control_system_type()`, `set_control_system_type()` for runtime connector switching
  - `get_epics_gateway_config()`, `set_epics_gateway_config()` for gateway management
  - `validate_facility_config()` for preset validation
  - Comprehensive test coverage for all configuration operations
- **Unified Configuration Command**: `osprey config` command group following industry standards
  - `osprey config show` - Display current project configuration
  - `osprey config export` - Export framework default configuration
  - `osprey config set-control-system` - Switch between Mock/EPICS connectors
  - `osprey config set-epics-gateway` - Configure EPICS gateway (APS, ALS, custom)
  - Interactive menu integration for guided configuration workflows

#### Control System Operations
- **Runtime Utilities for Control System Operations**: Control-system-agnostic utilities for generated Python code
  - New `osprey.runtime` module with synchronous API (write_channel, read_channel, write_channels)
  - Automatic configuration from execution context for reproducible notebooks
  - Async operations handled internally for simple generated code
  - Works with any control system (EPICS, Mock, etc.) without code changes
  - Complete unit and integration test coverage
  - API reference documentation with usage examples
- **Connector Auto-Verification**: Connectors automatically determine verification level and tolerance from configuration
  - Per-channel verification config from limits database (highest priority)
  - Global verification config from config.yml (fallback)
  - Hardcoded safe defaults if no config available (test environments)
  - New `LimitsValidator.get_verification_config()` method for per-channel lookup
  - Automatic limits validation on all connector writes (no application-level checks needed)
  - Comprehensive test coverage including mock and EPICS connectors
- **Control System Prompt Builders**: Custom prompt builders teaching LLMs to use runtime utilities
  - New ControlSystemPythonPromptBuilder with osprey.runtime documentation
  - Automatic injection of domain-specific instructions into capability prompts
  - Enhanced classifier examples for control system operations
  - Graceful fallback if custom prompts unavailable
  - Comprehensive test coverage for prompt builder integration
  - Complete tutorial on framework prompt customization

#### Testing Infrastructure
- **E2E Test Infrastructure**: Improved test isolation and added warnings to prevent common test failures
  - Added pytest hook to warn users when running `pytest -m e2e` instead of `pytest tests/e2e/`
  - Enhanced registry cleanup in test fixtures to prevent state pollution between tests
  - Added module cleanup in channel finder benchmarks to prevent stale imports
  - Updated README.md, TESTING_GUIDE.md, and tests/e2e/README.md with correct test commands
  - Added critical warnings in pytest.ini about proper e2e test execution
- **Runtime Utilities E2E Tests**: Comprehensive end-to-end test suite validating complete workflows
  - LLM learning osprey.runtime API from prompts
  - Context snapshot preservation and configuration
  - Channel limits safety integration (validates runtime respects boundaries)
  - Positive and negative test cases for write operations
  - Calculation + write workflows (e.g., "set voltage to sqrt(4150)")
- **E2E Test Infrastructure**: Warnings and cleanup mechanisms to prevent state pollution from incorrect test invocation
- **Unit Tests**: Registry isolation and channel finder registration test coverage

#### Documentation
- **EPICS Integration and Configuration Guides**: Comprehensive documentation for production deployment
  - Getting Started: Mock-first workflow with clear migration path to EPICS
  - CLI Reference: Complete `osprey config` command documentation
  - Production Guide: EPICS gateway configuration with facility presets
  - Architecture Guide: Pattern detection security model and design principles
  - API Reference: Framework-standard pattern detection reference
- **Documentation Positioning**: Updated README and tutorials to emphasize production-ready control system focus
  - Highlight plan-first orchestration and control system safety
  - Emphasize protocol-agnostic integration (EPICS, LabVIEW, Tango)
  - Note production deployment at major facilities (LBNL Advanced Light Source)
  - Updated feature list for control system use cases
  - Added comprehensive tutorial section on how generated code interacts with control systems using osprey.runtime
- **Developer Documentation**: Commit organization workflow guide for managing complex Git changes

### Changed

- **Code Quality**: Pre-merge cleanup improvements across codebase
  - Code formatting: Applied Black and isort to all changed files for consistent style
  - Linting fixes: Resolved ruff warnings (unused imports, bare except, unused variables)
  - Logging improvements: Replaced debug print() statements with proper logger.debug() calls in runtime module
  - Type hints: Added return type hints to 6 public functions for better IDE support
  - Import cleanup: Removed duplicate import in memory capability

#### Configuration and Architecture
- **CLI Organization**: Deprecated `osprey export-config` in favor of `osprey config export`
  - Backward compatibility maintained with deprecation notice
  - All configuration operations now unified under `osprey config` namespace
- **Pattern Detection Architecture**: Refactored to framework-standard patterns with security enhancements
  - Control-system-agnostic patterns work across all connector types
  - Comprehensive security coverage detects circumvention attempts (epics.caput, tango.DeviceProxy, etc.)
  - Framework provides sensible defaults; users can override in config.yml
  - Separated approved API patterns (write_channel, read_channel) from direct library call detection
  - `control_system.type` config now only affects runtime connector, not pattern detection
- **Project Templates**: Simplified pattern detection configuration with framework defaults
  - Removed verbose per-control-system pattern definitions
  - Framework automatically provides comprehensive security patterns
  - Clear guidance on when to override patterns (advanced/custom workflows only)
  - Updated README with EPICS gateway configuration instructions
  - Mock-first approach: Projects start in Mock mode, switch to EPICS when ready
- **Dependencies**: Promoted Claude Agent SDK from optional to core dependency
  - Advanced code generation now available in all installations
  - No longer requires separate installation with [claude-agent] extra
  - Minimum framework version 0.9.6+ for Claude Code generator support

#### Control System Connectors and Safety
- **Control System Connector API**: Unified channel naming and comprehensive write verification
  - Method rename: read_pv → read_channel, write_pv → write_channel (deprecated methods emit DeprecationWarning)
  - Class rename: PVValue → ChannelValue, PVMetadata → ChannelMetadata (deprecated classes emit DeprecationWarning)
  - Three-tier write verification: none/callback/readback with configurable tolerance
  - Rich result objects: ChannelWriteResult and WriteVerification with detailed status
  - Mock connector verification simulation for development testing
  - All deprecated APIs will be removed in v0.10
- **Runtime Channel Limits Validation**: Comprehensive safety system for validating writes against configured boundaries
  - Synchronous validation engine with min/max/step/writable constraints
  - Failsafe design blocks all unlisted channels by default
  - Optional max_step checking with I/O overhead warnings
  - Configurable policy modes: strict (error) vs resilient (skip)
  - JSON-based limits database with embedded defaults support
  - New exception: ChannelLimitsViolationError with detailed violation context
- **Python Executor Limits Checking Integration**: Automatic runtime validation of all epics.caput() calls
  - Transparent monkeypatching of epics.caput() and PV.put() methods
  - Embedded validator configuration in wrapper for container isolation
  - Graceful degradation if pyepics unavailable
  - Clear operator feedback with safety status messages

#### Python Execution and Code Generation
- **Python Execution Infrastructure**: Integrated runtime utilities with execution wrapper and notebooks
  - Execution wrapper automatically configures runtime from context snapshots
  - Context manager preserves control system config for reproducible execution
  - Notebooks include runtime configuration cell for standalone execution
  - Proper cleanup in finally block ensures resource release
  - E2E test artifacts now include generated Python code files
  - Developer guide documentation with integration details
- **Channel Write Capability Template**: Simplified by removing limits config loading (now automatic in connector)
  - Capabilities focus on orchestration (parsing, approval)
  - Connectors handle safety (limits, verification)
  - Cleaner separation of concerns

#### Template Configuration and Capabilities
- **Template Configuration**: Updated minimal template and project config for control system safety features
  - Added control_system section with writes_enabled, limits_checking, write_verification
  - Updated integration guides for new connector API
  - Framework capabilities updated for connector method rename
  - Pattern detection updated with new read_channel/write_channel patterns
  - Registry and utility updates for new context types
- **Channel Value Retrieval Renamed to Channel Read**: Renamed `channel_value_retrieval` capability to `channel_read` throughout the entire codebase for consistency and clarity
  - **Capability Name**: `channel_value_retrieval` → `channel_read`
  - **Class Name**: `ChannelValueRetrievalCapability` → `ChannelReadCapability`
  - **File Name**: `channel_value_retrieval.py.j2` → `channel_read.py.j2`
  - **Description**: Updated from "Retrieve current values" to "Read current values"
  - **Documentation**: Updated all references in .rst files, README, and examples
  - **Symmetric Naming**: Now matches `channel_read` (read) / `channel_write` (write) pattern
  - **Registry**: Updated capability registration and context type references
  - **Config**: Updated logging colors and capability lists
- **Channel Write Approval Workflow**: Human-in-the-loop approval for direct control system writes
  - Structured interrupt with operation summary and safety concerns
  - Integration with existing approval_manager and evaluator system
  - Clear approval prompts with channel addresses and target values
  - Resume payload includes complete operation context
- **BaseCapability Helper Method**: get_step_inputs() for accessing orchestrator-provided input contexts
  - Simplifies access to step inputs list from within execute()
  - Handles None values gracefully with configurable defaults
  - Comprehensive tests for various edge cases

#### UI/UX and Documentation Structure
- **CLI Approval Display**: Enhanced approval message presentation with heavy-bordered panel, bold title, and helpful subtitle for improved visibility and user experience
- **Gateway Approval Detection**: Enhanced approval response detection with two-tier system - instant pattern matching for simple yes/no responses, with LLM-powered fallback for complex natural language
- **Documentation Structure**: Refactored Python execution service documentation for improved organization
  - Removed obsolete standalone 03_python-execution-service.rst file
  - Streamlined service-overview.rst (793 → 452 lines, 40% reduction)
  - Focused content on generator extensibility for developers
  - Updated all cross-references to use directory structure
  - Improved navigation and reduced redundancy
- **OpenWebUI**: Enhanced configuration for improved out-of-box experience
  - Auto-configure Ollama and Pipeline connections in docker-compose
  - Disable authentication for local development (WEBUI_AUTH=false)
  - Documentation: automatic vs manual configuration guidance
  - Documentation: Docker vs Podman container networking (host.docker.internal vs host.containers.internal)

#### Miscellaneous
- **Error Node**: Removed deprecated manual streaming code and progress tracking in favor of unified logger system with automatic streaming

### Fixed

- **Runtime Utilities**: Context file now created during pre-approval stage, ensuring configuration access for osprey.runtime

#### Test Infrastructure and Stability
- **E2E Test**: Fixed `test_runtime_utilities_basic_write` by ensuring `context.json` is created during pre-approval stage
  - Context file now created before pre-approval notebook generation in `_create_pre_approval_notebook()`
  - Executor node reuses existing context file instead of recreating it
  - Test properly disables approval workflows for automated e2e execution
- **Test Configuration Pattern Detection**: Removed pattern overrides from test fixtures to use framework defaults
  - Test configs now use complete default patterns from `pattern_detection.py`
  - Fixes approval workflow tests to correctly detect `write_channel`/`read_channel` operations
  - Ensures tests validate actual framework behavior rather than incomplete test-specific patterns
  - Fixed 3 failing tests in `TestApprovalWorkflow` integration test suite
- **E2E Test Stability**: Improved test isolation and removed flaky test
  - Added approval manager singleton cleanup to prevent state pollution between tests
  - Removed redundant `test_runtime_utilities_calculation_with_write` (flaky due to ambiguous LLM prompt)
  - Fixed runtime utilities tests to disable limits checking when testing LLM code generation
  - Corrected config field name from `limits_file` to `database_path`
  - Fixed `_disable_capabilities` helper to properly comment out multi-line capability registrations

#### Validation and Logging
- **Limits validator**: Properly exclude metadata fields (description, source) from unknown field warnings
- **Error Node Logging**: Removed duplicate start/completion logging that occurred when combining decorator's automatic logging with manual status messages

## [0.9.5] - 2025-12-01

### Added
- **CLI Commands**: New `osprey generate claude-config` command to generate Claude Code generator configuration files with sensible defaults and auto-detection of provider settings
- **Interactive Menu**: Added 'generate' command to project selection submenu, centralized menu choice management with `get_project_menu_choices()`, improved consistency between main and project selection flows
- **E2E Test Suites**: Added comprehensive end-to-end test coverage
  - **Claude Config Generation Tests** (`test_claude_config_generation.py`): Validates `osprey generate claude-config` command, tests configuration file structure, provider auto-detection, and profile customization
  - **Code Generator Workflow Tests** (`test_code_generator_workflows.py`): Tests complete code generation pipeline with basic and Claude Code generators. Validates example script guidance following, instruction adherence, and deterministic assertions for generated code content
  - **MCP Capability Generation Tests** (`test_mcp_capability_generation.py`): End-to-end MCP integration testing including server generation/launch, capability generation from live MCP server, registry integration, and query execution with LLM judge verification

### Changed
- **API Call Logging**: Enhanced with caller context tracking across all LLM-calling components. Logging metadata now includes capability/module/operation details for better debugging. Improved JSON serialization with Pydantic model support (mode='json') and better error visibility (warnings instead of silent failures)
- **Claude Code Generator Configuration**: Major simplification - profiles now directly specify phases to run instead of using planning_modes abstraction. Default profile changed from 'balanced' to 'fast'. Unified prompt building into single data-driven `_build_phase_prompt()` method. Reduced codebase by 564 lines through elimination of duplicate prompt builders and dead code
- **Registry Display**: Filtered infrastructure nodes table to exclude capability nodes (avoid duplication with Capabilities table), moved context classes to verbose-only mode, improved handling of tuple types in provides/requires fields
- **MCP Generator Error Handling**: Added pre-flight connectivity checks using httpx, much clearer error messages when server is not running, and actionable instructions in error messages
- **Test Infrastructure**: Added auto-reset registry fixtures in both unit and E2E test conftest files to ensure complete test isolation. Fixtures now reset registry, clear config caches, and clear CONFIG_FILE env var before/after each test to prevent state leakage. Removed manual registry reset calls from individual tests

### Removed
- **Claude Code Generator Profiles**: Removed 'balanced' profile (consolidated to 'fast' and 'robust' only)
- **Claude Code Generator Configuration**: Removed 'workflow_mode' setting (use direct 'phases' list specification), removed 'planning_modes' abstraction (profiles specify phases directly), removed dead code (_generate_direct, _generate_phased, _build_phase_options, 7 duplicate prompt builders)

### Fixed
- **Registry Import Timing**: Fixed module-level `get_registry()` calls that could cause initialization order issues. Moved registry access to runtime (function/method level) in python capability, time_range_parsing capability, generate_from_prompt, and hello_world_weather template
- **Python Executor Logging**: Replaced deprecated `get_streamer` with unified `get_logger` API in code generator node for consistent streaming support
- **MCP Generator Configuration**: Added proper model configuration validation with clear error messages when provider is not configured. Improved error handling with unused variable cleanup and better logging integration
- **Time Range Parsing Tests**: Added mock for `store_output_context` to bypass registry validation, allowing tests to run independently of registry state. Removed obsolete decorator integration tests that were duplicating coverage
- **Tutorial E2E Tests**: Relaxed over-strict plot count assertion (1+ PNG files instead of 2+) to accommodate both single-figure and multi-figure plotting approaches
- **Claude Code Generator Tests**: Refactored to skip low-level prompt building tests (implementation details now covered by E2E tests). Improved test maintainability by focusing on behavior rather than internal methods
- **E2E Test Documentation**: Complete rewrite of tests/e2e/README.md with clearer structure, better isolation guidance, and comprehensive examples. Added warnings about running E2E tests separately from unit tests
- **Documentation**: Updated all Claude Code generator documentation to reflect simplified configuration model. Restructured generator-claude.rst with improved UX using collapsible dropdowns and tabbed sections. Updated all examples to use 'fast' as default profile
- **Tests**: Updated Claude Code generator tests to check 'profile_phases' instead of removed 'workflow_mode', removed tests for removed features, added tests for new phase-based configuration model

### Added
- **Python Executor Service - Complete Modular Refactoring**
  - **Modular Subdirectory Structure**: Reorganized python_executor service into focused subdirectories
    - `analysis/` - Code analysis, pattern detection, and policy enforcement
    - `approval/` - Human approval workflows
    - `execution/` - Container management and code execution
    - `generation/` - Pluggable code generator system
    - Each subdirectory has proper `__init__.py` and dedicated README documentation

  - **Pluggable Code Generator System**: New extensible architecture for code generation
    - **Abstract Interface**: `CodeGenerator` protocol defining standard generator contract
    - **Generator Factory**: Dynamic registration and instantiation with `GeneratorFactory`
    - **Multiple Implementations**:
      * `BasicGenerator` - Simple template-based generation for straightforward tasks
      * `ClaudeCodeGenerator` - Advanced AI-powered generation with:
        - Full conversation history management
        - Result validation and error recovery
        - Streaming support with callbacks
        - Tool use integration
        - Configurable via `execution.code_generator` and `execution.generators` settings
      * `MockGenerator` - Deterministic generator for testing
    - **Registry Integration**: Generator lifecycle managed through framework registry system
    - **State Model Extensions**: `PythonExecutorState` enhanced to support generator configuration

  - **Generator Configuration**: Explicit, flexible configuration structure
    - New `execution.code_generator` setting specifies active generator
    - Generator-specific config in `execution.generators` with model references or inline config
    - Deprecation warnings for old `models.python_code_generator` approach (backward compatible)
    - Updated project templates with examples for all generator types

  - **Integration Enhancements**: Connected generator system to framework
    - Python capability updated to support generator configuration
    - Analysis node enhanced with generator-aware validation
    - Execution pipeline improved for generator output handling
    - Container engine with better error reporting

  - **Comprehensive Test Suite**: Extensive test coverage for new system
    - Unit tests for all generator implementations (BasicGenerator, ClaudeCodeGenerator, MockGenerator)
    - Integration tests for generator-service interaction
    - Pattern detection integration tests
    - Result validation test suites
    - State reducer tests
    - Shared test fixtures and utilities in `tests/services/python_executor/`

  - **CLI and Template Improvements**: Enhanced user experience
    - Generator selection and configuration in interactive menu
    - Template system with generator-specific configurations
    - Claude generator config template (`claude_generator_config.yml.j2`)
    - Example plotting scripts for common use cases (time series, multi-subplot, publication-quality)
    - Improved README templates with generator setup instructions

## [0.9.4] - 2025-11-28

### Added
- **Channel Finder E2E Benchmarks**
  - New benchmark test suite for hierarchical channel finder pipeline
  - Tests query processing across all hierarchy complexity levels
  - Performance metrics: navigation depth, branching factor, channel count
  - Validates correct channel finding across diverse hierarchy patterns
  - Example queries testing system understanding and multi-level navigation
- **Flexible Hierarchical Database Schema**
  - Clean, flexible schema for defining arbitrary control system hierarchies
  - Single `hierarchy` section combines level definitions and naming pattern with built-in validation
  - Support arbitrary mixing of tree navigation (semantic categories) and instance expansion (numbered/patterned devices) at any level
  - Enable multiple consecutive instance levels (e.g., SECTOR→DEVICE, FLOOR→ROOM), instance-first hierarchies, or any tree/instance pattern
  - Automatic validation ensures level names and naming patterns stay in sync (catches errors at load time, not runtime)
  - Each level specifies `name` and `type` (`tree` for semantic categories, `instances` for numbered expansions)
  - Removed redundant/confusing fields from schema (eliminated `_structure` documentation field, consolidated three separate config fields into one)
  - Comprehensive test suite with 33 unit tests including 6 new naming pattern validation tests (all passing)
  - Example databases demonstrating real-world use cases:
    - `hierarchical.json`: Accelerator control (1,048 channels) - SYSTEM[tree]→FAMILY[tree]→DEVICE[instances]→FIELD[tree]→SUBFIELD[tree]
    - `mixed_hierarchy.json`: Building management (1,720 channels) - SECTOR[instances]→BUILDING[tree]→FLOOR[instances]→ROOM[instances]→EQUIPMENT[tree]
    - `instance_first.json`: Manufacturing (85 channels) - LINE[instances]→STATION[tree]→PARAMETER[tree]
    - `consecutive_instances.json`: Accelerator naming (4,996 channels) - SYSTEM[tree]→FAMILY[tree]→SECTOR[instances]→DEVICE[instances]→PROPERTY[tree]
  - Backward compatibility: Legacy databases with implicit configuration automatically converted with deprecation warnings
  - Support hierarchies from 1 to 15+ levels with any combination of types
  - Updated documentation with clean schema examples and comprehensive guides
- **Hello World Weather E2E Test**
  - New end-to-end test validating complete Hello World tutorial workflow
  - Tests weather capability execution, mock API integration, and registry initialization
  - LLM judge evaluation ensures beginner-friendly experience
  - Validates template generation and framework setup for new users

### Changed
- **Test Infrastructure**
  - Fixed test isolation between unit tests and e2e tests using `reset_registry()`
  - Updated all e2e tests to use Claude Haiku (faster, more cost-effective)
  - Separated unit test and e2e test execution to prevent registry mock contamination
  - Updated channel finder tests to use new unified database schema (`"type"` instead of `"structure"`)
  - Documentation: Updated `RELEASE_WORKFLOW.md` with clear instructions for running unit tests (`pytest tests/ --ignore=tests/e2e`) and e2e tests (`pytest tests/e2e/`) separately

## [0.9.3] - 2025-11-27

### Added
- **LLM API Call Logging** - Comprehensive logging of all LLM API interactions for debugging and transparency
  - New `development.api_calls` configuration section with `save_all`, `latest_only`, and `include_stack_trace` options
  - Automatic capture of complete input/output pairs with rich metadata (caller function, module, class, line number, model config)
  - Context variable propagation through async/thread boundaries using Python's `contextvars` for accurate caller detection
  - Intelligent caller detection that skips thread pool and asyncio internals to find actual business logic
  - Integration with classifier and orchestrator nodes via `set_api_call_context()` helper function
  - Capability-aware logging: classifier logs include capability name in filename for parallel classification tasks
  - Files saved to `_agent_data/api_calls/` with descriptive naming: `{module}_{class}_{function}_{capability}_latest.txt`
  - Documentation added to prompt customization guide and configuration reference
  - Complements existing prompt debugging (`development.prompts`) for complete LLM interaction transparency
- **End-to-End Test Infrastructure** - Complete LLM-based testing system for workflow validation
  - New `tests/e2e/` directory with comprehensive e2e test framework
  - **LLM Judge System** (`judge.py`) - AI-powered test evaluation with structured scoring
    - Evaluates workflows against plain-text expectations for flexible validation
    - Provides confidence scores (0.0-1.0) and detailed reasoning
    - Identifies warnings and concerns even in passing tests
  - **E2E Project Factory** (`conftest.py`) - Automated test project creation and execution
    - Creates isolated test projects from templates in temporary directories
    - Full framework initialization with registry, graph, and gateway setup
    - Query execution with complete state management and artifact collection
    - Working directory management for correct `_agent_data/` placement
    - Root logger capture for comprehensive execution trace logging
  - **Tutorial Tests** (`test_tutorials.py`) - Validates complete user workflows
    - `test_bpm_timeseries_and_correlation_tutorial` - Full control assistant workflow (channel finding, archiver retrieval, plotting)
    - `test_simple_query_smoke_test` - Quick infrastructure validation
  - **CLI Options** - Flexible test execution and debugging
    - `--e2e-verbose` - Real-time progress updates during test execution
    - `--judge-verbose` - Detailed LLM judge reasoning and evaluation
    - `--judge-provider` and `--judge-model` - Configurable judge AI model
  - **Comprehensive Documentation** (`tests/e2e/README.md`) - Complete testing guide with examples
  - **Belt and Suspenders Validation** - LLM judge + hard assertions for reliable testing
- **CLI Provider/Model Configuration** - Added `--provider` and `--model` flags to `osprey init` command for configuring AI provider during project creation
- **Unified Logging with Automatic Streaming**
  - Added `BaseCapability.get_logger()` method providing single API for logging and streaming
  - Enhanced `ComponentLogger` with automatic LangGraph streaming support
  - New `status()` method for high-level progress updates
  - Streaming behavior configurable per method with `stream` parameter
  - Smart defaults: `status()`, `error()`, `success()`, `warning()` stream automatically to web UI
  - Detailed logging methods (`info()`, `debug()`) remain CLI-only by default
  - Lazy stream writer initialization with graceful degradation when LangGraph unavailable
  - Custom metadata support via `**kwargs` on all logging methods
  - Automatic step tracking integrated with existing TASK_PREPARATION_STEPS
  - All infrastructure nodes, capabilities, service nodes, and templates migrated to unified pattern
  - Comprehensive test coverage with 26 test cases in `tests/utils/test_logger.py`
  - Backward compatible: existing `get_logger()` and `get_streamer()` patterns continue to work

### Changed
- **Capability Base Class** - Moved exception handling for classifier/orchestrator guide creation to base class properties with warning logs
- **Capability Templates** - Cleaned up unused imports and logger usage in all capability templates (control_assistant, minimal)

## [0.9.2] - 2025-11-25

### 🎉 Major Features

- **Complete Documentation**: Comprehensive docs for new architecture
  - Main python_executor service documentation with architecture overview
  - Per-subdirectory READMEs (analysis, approval, execution, generation)
  - Detailed generator implementation guides:
    * BasicGenerator usage and customization
    * ClaudeCodeGenerator configuration and features
    * MockGenerator for testing
  - Updated developer guides with new modular architecture
  - API reference documentation updates

**Benefits of New Architecture**:
- **Extensibility**: Easy to add new code generators (e.g., Claude Code SDK, GPT-4, custom generators)
- **Testability**: MockGenerator enables deterministic testing without API calls
- **Maintainability**: Clear separation of concerns with modular subdirectories
- **Flexibility**: Swap generators without modifying core service logic
- **Zero Breaking Changes**: Existing configurations continue to work with deprecation warnings

### Fixed
- **Interactive Menu Registry Contamination** ([#29](https://github.com/als-apg/osprey/issues/29))
  - Fixed bug where creating multiple projects in the same interactive menu session caused capability contamination
  - Global registry singleton now properly reset when switching between projects
  - Added `reset_registry()` calls in `handle_chat_action()` before launching chat
  - Prevents second project from inheriting capabilities from first project
  - Added comprehensive test suite to verify registry isolation

#### Argo AI Provider (ANL Institutional Service)
- **New provider adapter** for Argonne National Laboratory's Argo proxy service
- **8 models supported**: Claude (Haiku 4.5, Sonnet 4.5, Sonnet 3.7, Opus 4.1), Gemini (2.5 Flash, 2.5 Pro), GPT-5, GPT-5 Mini
- **OpenAI-compatible interface** with automatic structured output support
- Uses `$USER` environment variable for ANL authentication
- File: `src/osprey/models/providers/argo.py`
- Added `ARGO_API_KEY` to all project templates

#### Infrastructure Node Instance Method Migration
- **All 7 infrastructure nodes** migrated from static method pattern to instance method pattern
- Aligns infrastructure nodes with capability node implementation
- **Decorator Enhancements**:
  - Automatic detection of static vs instance methods (backward compatible)
  - Runtime injection of `_state` for all infrastructure nodes
  - Selective `_step` injection only for in-execution nodes (clarify, respond)
  - Defensive None checks for step injection with warning logs
  - Validation for invalid method types (classmethod, property)
- **Migrated Nodes**:
  - Router: Minimal state usage, routing metadata
  - Task Extraction: Data source integration, state refs updated
  - Classification: Extensive state usage (100+ refs), bypass mode
  - Clarify: First `_step` injection, task_objective extraction
  - Respond: `_step` injection, response generation
  - Error: NO `_step` injection (uses `StateManager.get_current_step_index()`)
  - Orchestration: 200+ lines, nested functions via closure
- **Testing**: Added 15 unit tests for infrastructure pattern
  - Tests validate decorator injection logic (_state, _step)
  - Tests verify backward compatibility with static methods
  - All tests passing

#### Capability Instance Method Pattern Testing
- Added 12 comprehensive tests for migrated capabilities
- New test directory: `tests/capabilities/` with fixtures and integration tests
- Memory Capability Tests (4 tests): signature validation, state/step injection, decorator integration
- Python Capability Tests (3 tests): instance method pattern validation
- TimeRangeParsing Capability Tests (5 tests): full end-to-end integration
- All tests formatted with black and linted

#### Instance Method Pattern for Capabilities
- **New Recommended Pattern**: Capabilities can now use instance methods instead of static methods
  - Helper methods available via `self`: `get_required_contexts()`, `get_task_objective()`, `get_parameters()`, `store_output_context()`
  - Eliminates ~60% of boilerplate code in capability implementations
  - More intuitive and Pythonic API design
  - Full backward compatibility maintained - static methods still work
- **Automatic Context Extraction**: `get_required_contexts()` method with tuple unpacking support
  - Matches order of `requires` field for elegant unpacking: `data, time = self.get_required_contexts()`
  - Falls back to dict access when preferred: `contexts = self.get_required_contexts(); data = contexts["DATA"]`
  - Automatic extraction with cardinality validation
- **New Helper Methods** in `BaseCapability`:
  - `get_required_contexts()` - Extract required contexts with automatic validation
  - `get_task_objective()` - Get current task description
  - `get_parameters()` - Get step parameters
  - `store_output_context()` - Store single output context
  - `store_output_contexts()` - Store multiple output contexts
- **Runtime State Injection**: `@capability_node` decorator injects `_state` and `_step` at runtime
  - Available within `execute()` method context
  - Clean separation between class definition and runtime state
- **Migration Guide**: Comprehensive documentation for upgrading from static to instance pattern
  - Side-by-side code comparisons
  - Migration checklist
  - Common issues and solutions
  - Gradual migration strategy
  - Located at: `docs/source/developer-guides/migration-guide-instance-methods.rst`

### Added
- **Comprehensive Test Suite**: Added 15 tests for capability helper methods
  - Tests for `get_required_contexts()`, `get_task_objective()`, `get_parameters()`
  - Tests for `store_output_context()` and `store_output_contexts()`
  - Error case validation and edge condition handling
  - Located at: `tests/base/test_capability_helpers.py`
- **Prompt-Based Capability Generator**: Natural language capability generation
  - `--from-prompt` CLI option for natural language capability descriptions
  - LLM-powered capability implementation generation
  - Automatic domain inference and classification
- **Test Infrastructure**: Global test utilities for all Osprey tests
  - `create_test_state()` factory for AgentState objects with sensible defaults
  - `PromptTestHelpers` for structural prompt testing
  - Reusable pytest fixtures in `tests/conftest.py`
  - Reduces test boilerplate by 140+ lines per test file
- **Comprehensive Clarification Tests**: 21 tests for clarification prompt generation
  - 10 core functionality tests (prompt structure, content extraction)
  - 9 error handling tests (edge cases, malformed inputs, unicode)
  - 2 integration tests (full workflow validation)
- **Interactive Menu Enhancements**: Version number display in interactive menu banner
- **Stanford AI Playground Provider**: Added Stanford AI playground as a built-in API provider
- **Cardinality Constraints**: New optional cardinality validation in `requires` field
  - Capabilities declare requirements with cardinality: `requires = [("CONTEXT_TYPE", "single")]`
  - Framework automatically validates and raises clear errors if violated
  - Eliminates repetitive `isinstance(context, list)` checks in capability code
  - Options: `"single"` (exactly one), `"multiple"` (must be list), or plain string (any cardinality)
  - Works seamlessly with `get_required_contexts()` helper method
  - Added 9 comprehensive test cases for cardinality validation
  - Updated all framework capability templates to use new pattern

### Changed
- **Generator Architecture**: Refactored monolithic generator into modular design
  - Split MCP capability generator into `BaseGenerator`, `MCPCapabilityGenerator`, and `PromptCapabilityGenerator`
  - Added generator models for type safety and validation
  - Improved CLI with lazy imports for better performance
  - Better separation of concerns and extensibility
- **Clarification System**: Improved prompt structure and context extraction
  - Enhanced clarification prompt builder with better orchestrator integration
  - Improved `get_system_instructions()` method for cleaner prompt composition
  - Better task_objective prioritization in clarification queries
- **Mock Archiver Connector**: Improved BPM position data generation
  - BPM positions now use realistic ±100 µm equilibrium offsets with ±10 µm oscillations
  - Each BPM has unique, reproducible random characteristics based on PV name
  - Slow drift patterns simulate realistic beam position variations
  - Adjusted default noise level from 0.01 to 0.1 for more realistic data
- **Template Updates**: All capability templates now use instance method pattern
  - `hello_world_weather` template updated with new pattern and helper methods
  - `control_assistant` templates (archiver_retrieval, channel_finding, channel_value_retrieval) updated
  - `minimal` template updated to show recommended pattern
  - All templates include proper `requires` field with cardinality constraints

### Fixed
- **Interactive Menu Registry Contamination** ([#29](https://github.com/als-apg/osprey/issues/29))
  - Fixed bug where creating multiple projects in the same interactive menu session caused capability contamination
  - Global registry singleton now properly reset when switching between projects
  - Added `reset_registry()` calls in `handle_chat_action()` before launching chat
  - Prevents second project from inheriting capabilities from first project
  - Added comprehensive test suite to verify registry isolation
- **Stanford API Key Detection**: Added missing STANFORD_API_KEY to environment variable detection (Reported by Marty)
- **Weather Template**: Fixed context extraction example in hello world weather template (PR #26)
- **CRITICAL BUG FIX**: `ContextManager.extract_from_step()` now correctly handles multiple contexts of the same type
  - Previously, when multiple contexts of the same type were requested (e.g., two `CURRENT_WEATHER` contexts), only the last one was returned, causing silent data loss
  - Now returns a list when multiple contexts of the same type exist: `{"CURRENT_WEATHER": [ctx1, ctx2]}`
  - Single contexts still returned as objects for backward compatibility: `{"CURRENT_WEATHER": ctx_obj}`
  - Capabilities can check `isinstance(context, list)` to detect and handle multiple contexts
  - Added 17 comprehensive test cases covering all scenarios
- **Interactive Menu Registry Contamination** ([#29](https://github.com/als-apg/osprey/issues/29))
  - Fixed bug where creating multiple projects in the same interactive menu session caused capability contamination
  - Global registry singleton now properly reset when switching between projects
  - Added `reset_registry()` calls in `handle_chat_action()` before launching chat
  - Prevents second project from inheriting capabilities from first project
  - Added comprehensive test suite to verify registry isolation

### Breaking Changes
- **BREAKING CHANGE**: `BaseCapabilityContext.get_access_details()` signature simplified
  - **Old:** `get_access_details(self, key_name: Optional[str] = None)` with defensive fallback
  - **New:** `get_access_details(self, key: str)` - key parameter is required
  - **Reason:** Framework always provides the key; optional parameter was unnecessary defensive programming
  - **Impact:** Custom context classes must update signature
  - **Migration:** Remove `Optional[str] = None` and fallback logic; use `key` parameter directly
- **BREAKING CHANGE**: `BaseCapabilityContext.get_summary()` signature simplified
  - **Old:** `get_summary(self, key: str)` - required the storage key
  - **New:** `get_summary(self)` - no parameters needed
  - **Reason:** Summaries describe the context data, not storage details
  - **Impact:** Custom context classes must remove `key` parameter
  - **Migration:** Remove `key` parameter from method signature
- **BREAKING CHANGE**: `ContextManager.get_summaries()` now returns `list[dict]` instead of `dict[str, Any]`
  - Simplifies the API by eliminating flattened key format (e.g., `"CONTEXT_TYPE.key"`)
  - Each summary dict already contains a `"type"` field for identification
  - More natural format for UI/LLM consumption
  - Updated 4 consumer files: `respond_node.py`, `clarify_node.py`, `memory.py`, `response_generation.py`

### Documentation
- **Complete Documentation Overhaul**: Updated 20+ documentation files for new patterns
  - API Reference: Updated `BaseCapability` and `BaseCapabilityContext` documentation
  - Developer Guides: Updated all capability and context management guides
  - Quick Start: Updated building-your-first-capability guide with instance pattern
  - Tutorials: Updated hello-world-tutorial with new recommended pattern
  - Example Applications: Updated ALS Assistant and control assistant examples
- **New Migration Guide**: Comprehensive migration documentation
  - Side-by-side pattern comparisons (static vs instance)
  - Step-by-step migration checklist
  - Common migration issues with solutions
  - Gradual migration strategy
  - Testing patterns for migrated capabilities
- **Context Management**: Enhanced context management system documentation
  - Updated with list-handling examples for multiple contexts
  - Added "Handling Multiple Contexts" pattern to integration guide
  - Documented cardinality constraint usage patterns
  - Explained two-phase extraction algorithm

### Migration Notes
- **Early Access Phase**: This is an acceptable breaking change as the framework is in early access (0.9.x)
- **For Capability Developers (Cardinality)**: Replace `isinstance(context, list)` checks with cardinality constraints
  - Old: `constraints=["PV_ADDRESSES"]` + manual `isinstance` check
  - New: `constraints=[("PV_ADDRESSES", "single")]` - framework handles validation
- **For Capability Developers (Multi-Context)**: Add `isinstance(context, list)` validation after extracting contexts to ensure your capability behavior matches expectations (only if not using cardinality constraints)
- **For get_summaries() Consumers**: Update code to iterate over list instead of dict.items()
  - Old: `for key, summary in summaries.items()`
  - New: `for summary in summaries: context_type = summary.get('type')`

## [0.9.1] - 2025-11-16

### Added
- **MCP Capability Generator (Prototype)**: Auto-generate Osprey capabilities from MCP servers
  - `osprey generate capability` command for creating capabilities from MCP servers
  - `osprey generate mcp-server` command for creating demo MCP servers for testing
  - Automatic ReAct agent integration with LangGraph
  - LLM-powered classifier and orchestrator guide generation with examples
  - Interactive registry and config integration with user confirmation
  - Support for FastMCP server generation with weather demo preset
  - Complete end-to-end MCP integration tutorial
  - Dependencies: `langchain-mcp-adapters`, `langgraph`, provider-specific LangChain packages
- **Capability Removal Command**: Clean removal of generated capabilities
  - `osprey remove capability` command for safe capability cleanup
  - Removes registry entries, config models, and capability files
  - Automatic backup creation before modifications
  - Interactive confirmation with preview of changes

### Changed
- **Core Dependencies**: Added `matplotlib>=3.10.3` to core dependencies
  - Python capability visualization now works out of the box without requiring `[scientific]` extras
  - Ensures tutorial examples (plotting beam current, etc.) work immediately after installation
  - Moved from optional `scientific` extras to required dependencies for improved user experience

## [0.9.0] - 2025-11-16

### Added
- **Prompt Customization System**: Flexible inheritance for domain-specific prompt builders
  - Added `include_default_examples` parameter to `DefaultTaskExtractionPromptBuilder`
  - Applications can now choose to extend or replace framework examples
  - Exported `TaskExtractionExample` and `ExtractedTask` from `osprey.prompts.defaults` for custom builders
  - Weather template includes 8 domain-specific examples for conversational context handling
  - New `framework_prompts.py.j2` template demonstrating prompt customization patterns
- **Domain Adaptation Tutorial**: Comprehensive Step 5 in hello-world tutorial
  - Explains why domain-specific examples improve conversational AI
  - 8 weather-specific task extraction examples covering location carry-forward, temporal references, etc.
  - Shows complete implementation with code examples and explanations
  - Demonstrates multi-turn conversation context synthesis
- **Conceptual Tutorial**: New comprehensive tutorial introducing Osprey's core concepts and design patterns
  - Explains Osprey's foundation on LangGraph with link to upstream framework
  - Compares ReAct vs Planning agents with clear advantages/disadvantages
  - Introduces capabilities and contexts with architectural motivation (addressing context window limitations)
  - Walks through designing a weather assistant as practical example
  - Visual grid cards for capability design with color-coded headers
  - Extracted design pattern summary for general application
  - Step-by-step orchestration examples showing how capabilities chain together
  - Location: `docs/source/getting-started/conceptual-tutorial.rst`
- **Control System Connectors**: Two-layer pluggable abstraction for control systems and archivers
  - **MockConnector**: Development/R&D mode - works with any PV names, no hardware required
  - **EPICSConnector**: Production EPICS Channel Access with gateway support (requires `pyepics`)
  - **MockArchiverConnector**: Generates synthetic historical time series data
  - **EPICSArchiverConnector**: EPICS Archiver Appliance integration (requires `archivertools`)
  - **ConnectorFactory**: Centralized creation with automatic registration via registry system
  - **Pattern Detection**: Config-based regex patterns for detecting control system operations in generated code
  - **Plugin Architecture**: Custom connectors (LabVIEW, Tango, etc.) via `ConnectorRegistration`
  - Seamless switching between mock and production via config.yml `type` field
  - Comprehensive API reference and developer guide with LabVIEW example
- **Control Assistant Template**: Production-ready template for accelerator control applications
  - Complete multi-capability system with PV value retrieval, archiver integration, and Channel Finder
  - Dual-mode support (mock for R&D, production for control room)
  - 4-part tutorial series (setup, Channel Finder integration, production deployment, customization)
  - Python execution service with read/write container separation and approval workflows
  - Full documentation with screenshots and step-by-step guides
- **Pattern Detection Service**: Static code analysis for control system operations
  - Configurable regex patterns per control system type
  - Used by approval system to identify read vs write operations
  - Location: `osprey.services.python_executor.pattern_detection`
- **Registry System Enhancements**: Added `ConnectorRegistration` dataclass for connector management
  - Automatic connector registration during framework initialization
  - Lazy loading with unified component management
  - Support for control_system and archiver connector types
- **CLI Template Support**: Added control_assistant template to CLI initialization system
  - New template option in `osprey init` command
  - Interactive menu displays control assistant with description
  - Template validation and configuration support

### Changed
- **FrameworkPromptProviderRegistration API**: Simplified registration interface
  - Removed `application_name` parameter (no longer used by framework)
  - Removed `description` parameter (no longer used by framework)
  - Framework now uses `module_path` as the provider key
  - **Backward Compatible**: Old parameters still accepted with deprecation warnings until v0.10
  - Updated all documentation examples to reflect new simplified API

### Deprecated
- **FrameworkPromptProviderRegistration fields**: `application_name` and `description` parameters
  - Will be removed in v0.10
  - Deprecation warnings emitted when used
  - Migration: Simply remove these parameters from your `FrameworkPromptProviderRegistration` calls

### Removed
- **Migration Guides**: Removed version-specific migration documentation
  - Removed `docs/source/getting-started/migration-guide.rst` (v0.6→v0.8 and v0.7→v0.8 guides)
  - Removed `docs/resources/MIGRATION_GUIDE_v0.6_to_v0.8.md`
  - Removed `docs/resources/MIGRATION_GUIDE_v0.7_to_v0.8.md`
  - Superseded by conceptual tutorial which provides better onboarding for current version
  - Historical migration information still available in git history if needed
- **Wind Turbine Template**: Removed deprecated wind turbine application template
  - Replaced by Control Assistant template with better real-world applicability
  - Removed `src/osprey/templates/apps/wind_turbine/` directory and all associated files
  - Removed `docs/source/getting-started/build-your-first-agent.rst` (superseded by control assistant tutorials)

### Changed
- **Channel Finder Presentation Mode**: Renamed `presentation_mode` value from "compact" to "template"
  - Updated all config files, documentation, and database implementations
  - Method `_format_compact()` renamed to `_format_template()`
- **Hello World Tutorial**: Simplified and improved tutorial UX
  - Removed unnecessary container deployment steps (tutorial only needs `osprey chat`)
  - Added "Ready to Dive In?" admonition for users who want to run first, learn later
  - Added comprehensive API key dropdown matching Control Assistant tutorial format
  - Improved messaging to welcome institutional providers (CBorg, Stanford AI Playground) while recommending Claude Haiku 4.5
  - Simplified prerequisites to focus on essentials (Python, framework, API key)
  - Updated Step 7 from "Deploy and Test" to "Run Your Agent" with streamlined setup
- **Hello World Weather Template**: Simplified template to match minimal tutorial scope
  - Removed container runtime configuration (no containers needed for basic tutorial)
  - Removed safety controls (approval, execution_control) - not relevant for simple weather queries
  - Removed execution infrastructure (EPICS, Jupyter modes, python_executor) - production features only
  - Template system now conditionally generates config sections based on template type
  - Services directory no longer created for hello_world_weather template
  - Generated config.yml now contains only essential sections: project identity, models, API providers, logging
  - Updated template README with streamlined setup instructions and accurate time estimate
  - Test coverage ensures hello_world_weather stays minimal (no production features)
- **Environment Template**: Updated `env.example` with clearer API key guidance
  - Fixed typo: `ANTHROPIC_API_KEY_o` → `ANTHROPIC_API_KEY`
  - Reordered to prioritize Anthropic (recommended) while showing institutional alternatives
  - Added helpful comments about provider flexibility
- **Configuration System**: Enhanced to handle missing configuration sections gracefully
  - Added `_get_approval_config()` with sensible defaults for tutorial environments
  - Added `_get_execution_config()` with local Python execution defaults
  - Removed strict validation in approval_manager that required all sections to be present
  - Enables minimal templates (like hello_world_weather) to work without production-only config sections
  - Provides helpful warnings when using framework defaults instead of explicit configuration
- **Control Assistant Part 3 Documentation**: Improved classification phase explanation
  - Added concrete list of all 6 capabilities (3 framework + 3 application) with file locations
  - Clarified why classification matters: reduces orchestrator context for better latency and accuracy
  - Provided specific YES/NO classification examples for each capability
- **Documentation Build Instructions**: Updated installation.rst to use modern `pip install -e ".[docs]"` workflow
  - Replaced deprecated `pip install -r docs/requirements.txt` approach
  - Uses optional dependencies from pyproject.toml for cleaner package management
- **Dependencies**: Moved `pandas` and `numpy` from optional `scientific` dependencies to base requirements
  - Required by archiver connectors which return pandas DataFrames for time-series data
  - Needed for MongoDB connector support
  - Fixes initialization error when running tutorials from scratch without manual pandas installation
  - `scientific` extra now includes only scipy, matplotlib, seaborn, scikit-learn, and ipywidgets
- **Provider API Key Metadata**: Established providers as single source of truth for API key acquisition information
  - Added `api_key_url`, `api_key_instructions`, and `api_key_note` fields to `BaseProvider`
  - Updated all provider implementations (Anthropic, OpenAI, Google, CBorg, Ollama) with verified metadata
  - Refactored CLI interactive menu to dynamically read API key help from provider metadata
  - Eliminates hardcoded API key instructions in CLI code
  - New providers automatically inherit help system support
  - Follows consistent metadata pattern across framework
- **Configuration API Simplification**: Streamlined `get_model_config()` function signature
  - Removed unused `service` and `model_type` parameters
  - Function now accepts only `(model_name, config_path)` for cleaner API
  - Removed 50+ lines of legacy nested config format support
  - All internal framework calls updated to use new signature
  - Updated documentation examples across 6 files

### Fixed
- **Anthropic Provider Structured Outputs**: Fixed task extraction failures when using Claude Haiku
  - Added structured output support for all Anthropic models (Haiku, Sonnet, Opus)
  - Uses native `response_format` API for Sonnet 4.5 and Opus 4.1 models
  - Falls back to prompt-based JSON extraction for Haiku and older models
  - Resolves `'str' object has no attribute 'task'` error in task extraction
- **Google Provider Structured Outputs**: Added structured output support for Google Gemini models
  - Implements prompt-based JSON extraction with schema validation
  - Fixed health check to use adequate token budget (100 tokens) for models with thinking capabilities
  - Added proper error handling when model uses all tokens for thinking with no output
  - Updated available models list to only include working Gemini 2.5 models (pro, flash, flash-lite)
- **Test Suite**: Updated tests to reflect runtime helper daemon verification improvements (commit 6bf0a1d)
  - Fixed 13 runtime helper tests to expect both compose version and ps daemon checks
  - Fixed 6 connector factory tests with proper registration fixture
  - Removed 4 deprecated wind_turbine template tests
  - All 189 tests now passing
  - Removed Gemini 1.5 models that are not available in current API version
  - Ensures consistent behavior across all LLM providers
- **Jinja2 Template Syntax**: Fixed invalid `.get('KEY')` method calls in Jinja2 templates
  - Replaced `env.get('CBORG_API_KEY')` with `env.CBORG_API_KEY` in conditionals
  - Fixed `env.get('TZ', 'default')` to use proper Jinja2 filter syntax: `env.TZ | default('default')`
  - Affects `project/README.md.j2` and `project/env.j2` templates
  - Resolves "expected token 'end of print statement', got ':'" error during project creation
- **Hello World Tutorial**: Fixed project naming inconsistencies (`weather-demo` → `weather-agent` to match template output)
- **Container Path Resolution**: Fixed database and file paths in containerized deployments
  - Deployment system now automatically adjusts `src/` paths to `repo_src/` (or `/pipelines/repo_src/` for pipelines service) in container configs
  - Fixes channel finder database loading and other file-based resources in containers
  - Simplifies configuration by removing `PROJECT_ROOT` environment variable requirement for basic usage
  - `project_root` now hardcoded in `config.yml` during `framework init` for simpler tutorial experience
  - `PROJECT_ROOT` environment variable remains available for advanced multi-environment deployments
- **Dev Mode Pipeline Container**: Fixed namespace collision by switching from editable source install to wheel-based installation
  - Prevents osprey's `utils` module from shadowing OpenWebUI base image's `/app/utils/pipelines`
- **Container Runtime Detection**: Fixed auto-detection to verify daemon is running, enabling proper fallback from Docker to Podman when Docker Desktop isn't running
- **OrchestratorExample Formatting**: Fixed PlannedStep fields not appearing in orchestrator prompt examples
  - Changed from `getattr()` to `.get()` for TypedDict field access in `OrchestratorExample.format_for_prompt()`
  - Previously resulted in empty `PlannedStep()` blocks, now correctly displays all fields
- **Approval Detection**: Increased max_tokens for approval detection from 10 to 50
  - Critical fix for models that require more tokens to generate complete JSON structures
  - Previously caused "yes" responses to be rejected due to incomplete structured output
  - Ensures reliable approval parsing across all supported models
- **Control Assistant Tutorial Documentation**: Fixed project structure tree and file path inconsistencies
  - Part 1: Removed non-existent `mock_control_system/` and `mock_archiver/` directories (they're in framework, not project)
  - Part 1: Added missing files that are actually generated: `address_list.csv`, benchmark datasets, `llm_channel_namer.py`, `data/README.md`
  - Part 2: Fixed incorrect database output path (`data/processed/` → `data/channel_databases/`)
  - Part 2: Added `CSV_EXAMPLE.csv` reference and clarified distinction between format reference and real UCSB FEL data
  - Documentation now accurately reflects actual generated project structure
- **Environment Variable Substitution**: Added support for bash-style default value syntax `${VAR:-default}`
  - Previously only supported simple `${VAR}` and `$VAR` forms
  - Now properly resolves environment variables with fallback defaults
  - Enables flexible configuration for both local and remote deployments
- **Configuration Override**: Fixed `set_as_default` parameter to properly override existing default config
  - Previously ignored when a default config was already set
  - Now honors explicit caller intent when `set_as_default=True`
  - Fixes issues with CONFIG_FILE environment variable initialization
- **Documentation Build**: Fixed v0.8.5 documentation build failures
  - Resolved compatibility issues with Sphinx build system

### Breaking Changes
- **`get_model_config()` signature changed**: `(model_name, service, model_type, config_path)` → `(model_name, config_path)`
  - **Impact**: Low - Framework model calls are internal and already updated
  - **User Action Required**: Only if you have custom application code using application-specific models
  - **Migration**:
    ```python
    # Old (if you have this in your application code):
    get_model_config('my_app', 'custom_model')

    # New:
    get_model_config('custom_model')
    ```
  - **Note**: Most users will not need to change anything - framework models (orchestrator, classifier, response, etc.) are handled internally

## [0.8.5] - 2025-11-10

### Fixed
- **Python Executor Configuration**: Removed deprecated 'framework' config nesting from python_executor components
- **Subprocess Execution**: Added `CONFIG_FILE` environment variable support for proper registry/context loading in subprocesses
  - Critical fix for execution scenarios where CWD ≠ project root
  - Updated `execution_wrapper.py` to pass config_path to registry initialization
  - Fixed `LocalCodeExecutor` to correctly access python_env_path from flat config structure
- **Exception Handling**: Improved exception chaining with `from e` for better error traceability across multiple modules
- **Configuration Access**: Updated `utils/config.py` to remove legacy nested format references

### Changed
- **Code Quality**: Removed all trailing whitespace (W291, W293) across codebase
- **Formatting**: Applied automatic ruff formatting fixes for consistency
- **Logging**: Improved logging with reduced verbosity and structured formatting
- **CLI**: Extracted duplicate streaming logic into reusable helper method

## [0.8.4] - 2025-11-09

### Added
- **Registry Modes**: Introduced Standalone and Extend modes for application registries
  - **Extend Mode** (recommended): Applications extend framework defaults via `ExtendedRegistryConfig`
    - Framework components loaded automatically (memory, Python, time parsing, etc.)
    - Applications can add, exclude, or override framework components
    - Returned by `extend_framework_registry()` helper function
    - Reduces boilerplate and simplifies upgrades
  - **Standalone Mode** (advanced): Applications provide complete registry including all framework components
    - Framework registry is NOT loaded
    - Full control over all components
    - Used when `RegistryConfig` is returned directly (not via helper)
  - Mode detection is automatic based on registry type (`isinstance(config, ExtendedRegistryConfig)`)
- **New Class**: `ExtendedRegistryConfig` marker class for signaling Extend mode
  - Subclass of `RegistryConfig` with identical fields
  - Type-based detection enables automatic framework merging
  - Added to `__all__` exports in `osprey.registry`
- **New Helper Function**: `generate_explicit_registry_code()` for template generation
  - Generates complete registry Python code combining framework + app components
  - Used by CLI template system for creating explicit registries
  - Useful for applications that want full visibility of all components
  - Takes app metadata and component lists, returns formatted Python source code
- Comprehensive test suite for registry modes (500+ lines across 4 new test files)
  - `test_registry_modes.py`: Tests for Extend vs Standalone mode detection
  - `test_registry_loading.py`: Tests for registry loading mechanisms
  - `test_registry_helpers.py`: Tests for helper functions
  - `test_registry_validation.py`: Tests for registry validation

### Changed
- **Registry Helper**: `extend_framework_registry()` now returns `ExtendedRegistryConfig` instead of `RegistryConfig`
  - Backward compatible (ExtendedRegistryConfig is a subclass of RegistryConfig)
  - Type signature change enables automatic mode detection
  - Applications using type hints should update return type annotation
- Enhanced registry documentation with comprehensive coverage of both modes
  - Developer guide updated with mode selection guidance
  - API reference documentation expanded with ExtendedRegistryConfig details
  - Code examples updated to show ExtendedRegistryConfig return type

### Breaking Changes
- **RegistryManager Constructor**: Parameter changed from `registry_paths: List[str]` to `registry_path: Optional[str]`
  - **Impact**: Low - most applications use `initialize_registry()` which reads from config
  - **Migration**: Change `RegistryManager([path1, path2])` to `RegistryManager(path)` for single registry
  - **Rationale**: Simplified to single-application model matching actual usage patterns
  - Framework now supports one application registry per instance (loaded from `config.yml`)
- **Type Signature**: `extend_framework_registry()` return type changed to `ExtendedRegistryConfig`
  - **Impact**: Very low - backward compatible at runtime (subclass relationship)
  - **Migration**: Update type hints from `-> RegistryConfig` to `-> ExtendedRegistryConfig`
  - Only affects code using explicit type checking

### Removed
- Test file `test_path_based_discovery.py` (replaced with mode-specific tests)

### Developer Notes
- Registry system now uses type-based mode detection for cleaner separation of concerns
- Standalone mode enables minimal deployments and custom framework variations
- Extend mode remains the recommended default for >95% of applications
- See developer guide "Registry and Discovery" for complete mode selection guidance

## [0.8.3] - 2025-11-09

### Added
- **Docker Runtime Support**: Framework now supports both Docker and Podman container runtimes
  - New `runtime_helper.py` module for automatic runtime detection
  - Configuration setting `container_runtime` in `config.yml` (options: `auto`, `docker`, `podman`)
  - Environment variable `CONTAINER_RUNTIME` for per-command runtime override
  - Auto-detection prefers Docker first, falls back to Podman
  - Requires Docker Desktop 4.0+ or Podman 4.0+ (native compose support)
  - **User-friendly error messages**: Platform-specific guidance when Docker/Podman not running
    - macOS: "Open Docker Desktop from Applications" with menu bar icon hints
    - Linux: systemctl commands and docker group permissions
    - Windows: Start menu and system tray instructions
  - Comprehensive test suite for runtime detection and selection (33 tests)
- **Custom AI Provider Registration**: Applications can now register custom AI model providers through the registry system
  - Added `providers` parameter to `extend_framework_registry()` helper function
  - Added `exclude_providers` parameter to exclude framework providers
  - Added `override_providers` parameter to replace framework providers with custom implementations
  - Provider merging support in `RegistryManager._merge_application_with_override()`
  - Comprehensive test suite (16 tests) covering all provider registration scenarios
  - Support for institutional AI services (Azure, Stanford AI, national lab endpoints) and commercial providers

### Changed
- **Container Management**: All deployment commands now use runtime abstraction layer
  - Updated `container_manager.py`: 6 functions now use runtime helper
  - Updated `health_cmd.py`: Container health checks are runtime-agnostic
  - Updated `interactive_menu.py`: Mount checking uses configured runtime
  - All compose operations work seamlessly with both Docker and Podman
  - **Fixed JSON parsing**: `osprey deploy status` now handles both Docker (NDJSON) and Podman (JSON array) output formats
- **Dependencies**: Removed Python `podman` and `podman-compose` packages
  - Container runtimes must be installed via system package managers
  - Framework uses CLI tools (`docker`/`podman` commands), not Python SDKs
  - Added installation documentation for both runtimes
- Enhanced registry helper functions to support provider registration parameters
- Updated developer guide documentation with provider registration examples

### Breaking Changes
- **Installation**: Users must install Docker Desktop 4.0+ or Podman 4.0+ separately
  - Python packages no longer provide container runtime functionality
  - See installation guide for platform-specific instructions
- **Note**: Existing Podman users are unaffected - auto-detection will find Podman if Docker not installed

## [0.8.2] - 2025-11-05

### Added
- Registry display command (`osprey registry`) with themed output
- Rebuild and clean deployment actions in interactive menu with safety confirmations
- Strategic test suite: 22 tests covering logging filters and container status logic
- OSPREY_QUIET environment variable for subprocess noise reduction
- Helper functions for status table creation (reduced duplication)

### Changed
- **Complete CLI style migration**: All commands now use centralized Styles constants
- **Container manager logging**: Converted 53 print statements to ComponentLogger system
- **Status command rewrite**: Now uses direct `podman ps` for more reliable state checking
- Improved service name matching with underscore/hyphen variation handling
- Enhanced log suppression using quiet_logger for cleaner CLI output
- Theme-aware command completer using active theme colors
- Condensed verbose comments for better code readability

### Fixed
- Container status display now works independently of compose files
- Smart container-to-project matching with backward compatibility
- Proper separation of project vs non-project containers in status display
- CONFIG logger properly suppressed in interactive menu operations

### Improved
- Net change: -641 lines through cleanup and consolidation
- Better error handling in status command with timeout protection
- Enhanced maintainability through style consistency
- Clearer deployment operation confirmations for destructive actions

## [0.8.1] - 2024-11-04

### Fixed
- Post-release fixes and improvements from initial 0.8.0 testing
- Package distribution and metadata updates

### Changed
- Final production release of Osprey Framework rebrand
- Improved documentation and migration guides
- Enhanced CLI theme system consistency

## [0.8.0] - 2025-11-02

### 🦅 Major Changes - Rebranding to Osprey Framework

**BREAKING CHANGES:**
This release represents a complete rebranding of the project from "Alpha Berkeley Framework" to "Osprey Framework".

**Package & Installation Changes:**
- **Package name:** `alpha-berkeley-framework` → `osprey-framework`
  - Install with: `pip install osprey-framework` (note: hyphen in package name)
  - PyPI URL: https://pypi.org/project/osprey-framework/
- **Import paths:** `from framework.*` → `from osprey.*`
  - All Python imports updated throughout codebase
  - Example: `from osprey.state import AgentState`
- **CLI command:** `framework` → `osprey`
  - New primary command: `osprey init`, `osprey chat`, `osprey deploy`, etc.
  - Legacy `alpha-berkeley` commands maintained for backward compatibility
- **Repository:** `thellert/alpha_berkeley` → `als-apg/osprey`
  - New GitHub repository: https://github.com/als-apg/osprey
  - Old URLs automatically redirect via GitHub
- **Documentation:** https://als-apg.github.io/osprey

**Migration Guide:**

For existing users upgrading from Alpha Berkeley Framework:

1. **Uninstall old package:**
   ```bash
   pip uninstall alpha-berkeley-framework
   ```

2. **Install new package:**
   ```bash
   pip install osprey-framework
   ```

3. **Update imports in your code:**
   - Find and replace: `from framework.` → `from osprey.`
   - Find and replace: `import framework` → `import osprey`

4. **Update CLI commands:**
   - Replace `framework` with `osprey` in scripts and documentation
   - Example: `framework init` → `osprey init`

5. **Update project dependencies:**
   - In `requirements.txt`: `alpha-berkeley-framework` → `osprey-framework`
   - In `pyproject.toml`: `alpha-berkeley-framework` → `osprey-framework`

**Note:** GitHub automatically redirects old repository URLs. However, we recommend updating your git remotes for long-term stability:
```bash
git remote set-url origin https://github.com/als-apg/osprey.git
```

**Technical Details:**
- 134 Python files updated with new imports
- 67 documentation files updated with new branding
- All templates updated to generate osprey-based projects
- Package structure: `src/osprey/` (was `src/framework/`)
- Distribution files: `osprey_framework-0.8.0.whl` (underscore is automatic)

### Includes All Features from v0.7.7 and v0.7.8
- Interactive TUI menu system
- Multi-project support
- Enhanced documentation
- All bug fixes from previous releases

---

## [0.7.8] - 2025-11-01

### Fixed
- Fixed config system test failure by correcting global variable references (`_default_config` and `_default_configurable`)
- Enhanced `get_config_value()` function to fall back to raw config when path not found in processed configurable dict
- Updated template documentation to clarify "Example categories" instead of "Valid categories"

## [0.7.7] - 2025-11-01

### Added
- **Interactive Terminal UI (TUI)** - Comprehensive menu system for guided workflows
  - New `interactive_menu.py` (1,771 lines) - Main TUI implementation with context-aware menus
  - Context detection: Automatically adapts interface based on whether user is in a project directory
  - Interactive project initialization with template, provider, and model selection
  - Automatic API key detection from shell environment
  - Secure password-style input for API keys not found in environment
  - Beautiful Rich-formatted interface with colors and styled panels
  - Smart defaults based on detected environment variables
  - Seamless integration with existing Click commands
- **Multi-Project Support** - Work seamlessly across multiple framework projects
  - New `project_utils.py` (90 lines) - Unified project path resolution utilities
  - `--project` flag added to all CLI commands (init, chat, deploy, health, export-config)
  - `FRAMEWORK_PROJECT` environment variable support for persistent project selection
  - Three ways to specify project: current directory, --project flag, or env var
  - Explicit `config_path` parameter throughout configuration system
  - Per-path config caching for efficient multi-project workflows
  - Registry path resolution relative to config file location
  - Project isolation with no cross-project configuration contamination
- **Provider Descriptions** - User-friendly provider identification
  - Added `description` field to `BaseProvider` abstract class
  - All provider adapters updated with descriptions for TUI menus:
    - anthropic: "Anthropic (Claude models)"
    - openai: "OpenAI (GPT models)"
    - google: "Google (Gemini models)"
    - ollama: "Ollama (local models)"
    - cborg: "LBNL CBorg proxy (supports multiple models)"
- **Environment Variable Auto-Detection** - Intelligent project initialization
  - `_detect_environment_variables()` method in TemplateManager
  - Automatically detects API keys from system environment
  - Updates `.env.example` template with detected values
  - Displays detected environment variables during init command
  - Falls back to placeholder values if vars not found
- **Enhanced Container Status Display** - Professional formatted output
  - Rich table formatting for `framework deploy status` command
  - Colored status indicators with emoji (● Running / ● Stopped)
  - Health status display (healthy/unhealthy/starting) when available
  - Clear port mapping display (host→container format)
  - JSON-based parsing for structured container information
  - Helpful guidance when no services are running

### Changed
- **Configuration System** - Enhanced for multi-project scenarios
  - All config utility functions now accept optional `config_path` parameter:
    - `get_model_config()`, `get_provider_config()`, `get_framework_service_config()`
    - `get_config_value()`, `get_full_configuration()`
  - Implemented per-path config caching for performance
  - Added `set_as_default` parameter for explicit path handling
  - Maintains backward compatibility with singleton pattern
- **Registry Manager** - Enhanced path resolution
  - Added `config_path` parameter to `get_registry()` and `initialize_registry()`
  - Resolve relative registry paths against config file location
  - Pass config_path when initializing registry components
  - Better base path resolution for registry files
- **Data Source Manager** - Improved logging and status tracking
  - Enhanced logging to distinguish empty vs. failed data sources
  - Track sources that succeed but return no data
  - Better summary format: "Data sources checked: 3 (1 with data, 1 empty, 1 failed)"
  - Clearer UX for understanding data availability and debugging
- **Default Model Selection** - Better out-of-box experience
  - Changed default model from `gemini-2.0-flash-exp` to `claude-3-5-haiku-latest`
  - Better performance and reliability for common tasks
  - Lower latency and more consistent responses
- **Docker Compose Templates** - Cleaner initial configuration
  - Optional settings now commented out by default in generated templates
  - Simpler initial setup for new projects
  - Easy to uncomment and enable advanced features when needed
- **Template Manager** - Enhanced for TUI integration
  - Exported key functions for reuse by interactive menu
  - Better separation of concerns between CLI and TUI
  - Improved error handling and validation

### Enhanced
- **CLI Commands** - Unified project path resolution
  - All commands updated with `--project` flag support
  - Consistent path resolution using new `resolve_project_path()` utility
  - Better error messages when project path invalid
  - Project-aware command execution throughout
- **User Experience** - Professional terminal interface
  - Questionary library integration for interactive prompts
  - Custom styling matching framework theme
  - Rich console output with formatted panels and tables
  - Helpful guidance and next-step suggestions
  - Lower barrier to entry for new users
  - Faster workflows for common tasks

### Technical Details
- **TUI Architecture** - ~1,900 lines of new code
  - Context-aware menu system with adaptive interface
  - Integration with TemplateManager for project scaffolding
  - Integration with registry system for provider metadata
  - Direct function calls (not Click commands) for efficiency
  - Optional dependency on questionary (graceful fallback)
- **Multi-Project Infrastructure** - ~300 lines of enhanced code
  - Configuration system: per-path caching and explicit path support
  - Registry manager: config-aware initialization and path resolution
  - Data management: config path propagation
  - CLI commands: unified path resolution utility
- **Zero Breaking Changes** - Complete backward compatibility
  - All existing CLI commands work unchanged
  - TUI only activates when no arguments provided
  - Direct commands remain primary interface for power users
  - Configuration system maintains singleton pattern when no path specified

## [0.7.6] - 2025-10-30

### Added
- **Provider Registry System** - Centralized AI provider management integrated into framework registry
  - New `ProviderRegistration` dataclass for minimal provider metadata (module_path, class_name only)
  - Provider metadata introspected from class attributes (single source of truth)
  - Registry methods: `get_provider()`, `list_providers()`, `get_provider_registration()`
  - Providers added to component initialization order (early loading for use by capabilities)
- **Provider Adapter Architecture** - New base class and five framework provider implementations
  - `BaseProvider` abstract class defining provider interface (create_model, execute_completion, check_health)
  - `AnthropicProviderAdapter` with extended thinking support
  - `OpenAIProviderAdapter` with structured outputs and token parameter handling
  - `GoogleProviderAdapter` with extended thinking support
  - `OllamaProviderAdapter` with automatic localhost ↔ host.containers.internal fallback
  - `CBorgProviderAdapter` for LBNL institutional AI service
- **Log Filtering Utilities** - Dynamic log suppression system
  - New `framework.utils.log_filter` module with `LoggerFilter` class
  - Context managers: `suppress_logger()`, `suppress_logger_level()`, `quiet_logger()`
  - Filter by logger name, level, message patterns, or combinations
  - Thread-safe with pre-compiled regex patterns for performance
- **Testing Infrastructure** - pytest configuration and VCR support
  - New `pytest.ini` with test markers (unit, integration, requires_api, vcr, etc.)
  - `tests/cassettes/` directory with comprehensive README for VCR usage
  - Test markers for provider-specific tests (requires_openai, requires_anthropic, etc.)
  - Added pytest-vcr and vcrpy to dev dependencies
- **Custom Provider Registration** - Applications can register institutional/commercial providers
  - Full documentation with Azure OpenAI example in registry guide
  - Support for institutional AI services (Stanford AI Playground, national lab endpoints)
  - Support for commercial providers (Cohere, Mistral AI, Together AI, etc.)

### Changed
- **Model Factory Refactoring** - Simplified to use provider registry (~280 lines removed)
  - Replaced hardcoded provider requirements dict with registry lookups
  - Use `provider.create_model()` for all provider types
  - Removed `_create_openai_compatible_model()` helper function
  - Removed `_get_ollama_fallback_urls()` and `_test_ollama_connection()` (moved to OllamaProviderAdapter)
  - Validation uses provider class metadata instead of hardcoded dict
- **Completion Module Refactoring** - Streamlined to use provider registry (~290 lines removed)
  - Replaced all provider-specific if/elif blocks with `provider.execute_completion()`
  - Removed provider requirements validation dict
  - Removed `_get_ollama_fallback_urls()` helper
  - Added `temperature` parameter to completion function
- **Health Check Refactoring** - Updated to use provider registry (~290 lines removed)
  - Initialize registry before provider checks with loading spinner
  - Use `provider.check_health()` instead of provider-specific logic
  - Removed all provider-specific if/elif blocks
  - Removed `_test_provider_connectivity()` method
  - Added `quiet_logger` usage to suppress verbose registry initialization logs
  - Preserved charge-avoiding health checks for Anthropic and Google
- **Memory Storage Logging** - Improved logging consistency
  - Switched from root logger to framework logger (`get_logger("memory_storage")`)
  - Changed initialization message from INFO to DEBUG level
  - Reduced log verbosity for non-critical operations
- **Module Exports** - Cleaned up framework.models exports
  - Removed `_create_openai_compatible_model` from public API

### Fixed
- **Test Import Paths** - Updated config imports for relocated modules
  - Changed from `configs.config` to `framework.utils.config`
  - Added minimal config.yml in tests to prevent loading errors
  - Updated integration tests for new configuration module location

### Removed
- **Deprecated Code Cleanup**
  - Deleted `src/framework/interfaces/openwebui/` (deprecated interface implementation)
  - Deleted `docs/resources/other/EXECUTION_POLICY_SYSTEM.md` (outdated design document)

### Documentation
- **Provider Registry Documentation** - Comprehensive documentation for new system
  - Added `ProviderRegistration` to registry API reference
  - Custom provider registration guide with complete Azure OpenAI example
  - Updated component initialization order in all docs
  - Added provider access methods to RegistryManager documentation
  - Updated configuration docs with custom provider extensibility information
- **README and Examples** - Updated with custom provider examples
  - Common use cases: Azure OpenAI, institutional services, commercial providers
  - Integration with `get_model()` and `get_chat_completion()`
  - Health check system integration

### Technical Details
- **Code Reduction**: ~860 lines removed from factory.py, completion.py, health_cmd.py
- **New Code**: ~1,090 lines of well-structured provider adapter implementations
- **Net Result**: More maintainable, extensible architecture with single source of truth
- **Zero Breaking Changes**: All existing APIs remain unchanged

## [0.7.5] - 2025-10-28

### Added
- **Parallel Capability Classification** - Multiple capabilities now classified simultaneously using `asyncio.gather()`
  - New `CapabilityClassifier` class for individual capability processing with proper resource management
  - Semaphore-controlled concurrency to prevent API flooding while maintaining performance
  - Configurable `max_concurrent_classifications` setting (default: 5) in `execution_control.limits`
  - Enhanced error handling for individual classification failures
- **Improved Reclassification Logic** - New `_detect_reclassification_scenario()` function
  - Better detection of reclassification scenarios from error state
  - Cleaner error state cleanup during reclassification
  - Enhanced logging for reclassification process
- **New Configuration Function** - `get_classification_config()` for accessing classification settings
- **Documentation Build System** - Added `docs/config.yml` for documentation build compatibility

### Changed
- **Classification Architecture** - Refactored from sequential to parallel processing
  - `select_capabilities()` now uses parallel task execution with semaphore control
  - Removed old `_classify_capability()` function in favor of `CapabilityClassifier` class
  - Improved error handling and logging throughout classification process
- **Router Logic** - Simplified reclassification handling
  - Removed manual state setting in router, moved responsibility to classifier
  - Cleaner separation of concerns between router and classifier
- **State Management** - Enhanced agent control state with new classification limits
  - Added `max_concurrent_classifications` to `AgentControlState`
  - Updated state manager defaults and configuration builder

### Fixed
- **Documentation Build System** - Updated for pip-installable framework structure
  - Changed from `requirements.txt` to `pip install -e ".[docs]"` in Makefile and GitHub Actions
  - Added mock imports for documentation build compatibility
  - Fixed dropdown syntax and removed unused CSS rules
- **Installation Guide** - Added docs extras install option and fixed formatting
- **Command Help Text** - Fixed escaped newlines in command help strings

## [0.7.4] - 2025-10-27

### Fixed
- **Template Registry Class Names** - Fixed duplicate "RegistryProvider" suffix in generated registry class names
  - Class name generation now produces correct names like `WeatherTutorialRegistryProvider` instead of `WeatherTutorialRegistryProviderRegistryProvider`
  - Updated `_generate_class_name()` method to return PascalCase prefix only
  - Templates correctly append "RegistryProvider" suffix
  - Affects all three app templates: hello_world_weather, wind_turbine, minimal
- **Template Import Paths** - Updated documentation examples to use v0.7.0 import patterns
  - Changed from `applications.hello_world_weather.*` to `hello_world_weather.*`
  - Updated mock_weather_api.py documentation examples
  - Updated capabilities/__init__.py documentation and Sphinx references
  - Ensures generated projects follow correct v0.7.0 decoupled architecture
- **Requirements Template Rendering** - Fixed framework version substitution in generated requirements.txt
  - Moved requirements.txt from static files to rendered templates
  - Now properly replaces `{{ framework_version }}` placeholder with actual version
  - Ensures generated projects pin correct framework version in requirements.txt

## [0.7.3] - 2025-10-26

### Added
- **Development Mode Support** - New `--dev` flag for deploy CLI command
  - Local framework override capability for seamless development testing
  - Smart dependency installation in containers with dev mode detection
  - Automatic local framework installation when DEV_MODE is enabled

### Changed
- **Container Deployment** - Enhanced service templates and deployment workflow
  - Project templates now use PyPI framework distribution by default
  - Removed hardcoded framework paths from configuration templates
  - Improved container startup scripts with better logging and error handling
  - Changed container restart policy to 'no' for better development experience
- **Project Templates** - Automatic framework dependency management
  - Added framework dependency to generated `pyproject.toml` and `requirements.txt`
  - Created proper agent data directory structure for container mounts
  - Enhanced fallback mechanisms for missing requirements files

### Fixed
- **Container Manager** - Improved registry path resolution for different service types
- **Environment Handling** - Graceful .env file handling with fallback warnings
- **Mount Points** - Ensure container mount directories exist before deployment

## [0.7.2] - 2025-10-26

### Changed
- **Simplified Installation** - PostgreSQL dependencies moved to optional `[postgres]` extra
  - Basic framework now installs without PostgreSQL requirements
  - Uses in-memory checkpointing by default (perfect for development/testing)
  - Production users can install `alpha-berkeley-framework[postgres]` for persistent state
  - Resolves installation issues on systems without PostgreSQL packages

## [0.7.1] - 2025-10-26

### Added
- **Centralized Slash Command System** - Unified command registry for CLI and web interfaces
  - Command categorization (CLI, agent control, service commands)
  - Autocompletion and help system
  - Context-aware command execution

### Changed
- Enhanced CLI health command with command system integration
- Updated gateway architecture for command processing
- Improved state management for command execution context

## [0.7.0] - 2025-10-25

### 🎉 Major Architecture Release - Framework Decoupling

This is a **major breaking release** that fundamentally changes how applications are built and deployed. The framework is now pip-installable, enabling independent application development in separate repositories.

### Added

#### Unified CLI System
- **`framework` command** - Main CLI entry point with lazy loading for fast startup
- **`framework init`** - Create new projects from templates with project scaffolding
  - Templates: minimal, hello_world_weather, wind_turbine
  - Options: `--template`, `--registry-style`, `--output-dir`, `--force`
- **`framework deploy`** - Manage Docker services (up/down/restart/status/rebuild/clean)
  - Intelligent service management with validation
  - Service health checking
- **`framework chat`** - Interactive CLI conversation interface
- **`framework health`** - Comprehensive system diagnostics
  - Validates Python version, dependencies, configuration, registry files, containers
  - ~968 lines of diagnostic code
- **`framework export-config`** - View framework default configuration template
  - Supports YAML and JSON output
  - Helps understand configuration options

#### Template System
- **3 Production-Ready Templates** - Instant project generation
  - `minimal` - Bare-bones starter with TODO placeholders
  - `hello_world_weather` - Simple weather query example
  - `wind_turbine` - Complex multi-capability monitoring system
- **Project Scaffolding** - Complete self-contained projects
  - Application code (capabilities, registry, context classes)
  - Service configurations (Jupyter, OpenWebUI, Pipelines)
  - Self-contained configuration (~320 lines)
  - Environment template (.env.example)
  - Dependencies file (pyproject.toml)
  - Getting started documentation

#### Registry Helper Functions
- **`extend_framework_registry()`** - Simplify application registries by ~70%
  - Compact style: 5-10 lines instead of 80+ lines of boilerplate
  - Automatic framework component inclusion
  - Clean exclusion syntax: `exclude_capabilities=["python"]`
  - Optional override support for advanced customization
- **`get_framework_defaults()`** - Inspect framework components
- **Progressive disclosure** - Start simple, go explicit when needed

#### Path-Based Discovery
- **Explicit registry file paths** in `config.yml`
- **`registry_path`** configuration (top-level or nested)
- **`importlib.util` based loading** - Robust module loading
- **Temporary sys.path manipulation** - Like Django, Sphinx, Airflow
- **Strict validation** - Exactly one `RegistryConfigProvider` per file
- **Rich error messages** - Comprehensive resolution hints

#### Self-Contained Configuration
- **One `config.yml` per application** - Complete transparency
- **~320 lines** - All framework settings visible and editable
- **Framework defaults included** at project creation
- **`.env` file support** - Automatic loading with python-dotenv
- **Well-organized** - Clear section comments for easy navigation

#### Documentation
- **Migration Guide** - Comprehensive upgrade documentation (~730 lines)
  - Breaking changes overview
  - Step-by-step migration instructions (10 steps)
  - Production and tutorial migration paths
  - Common issues and solutions
  - Migration progress checklist
- **Updated Getting Started** - Fresh installation and migration paths
- **CLI Reference** - Complete command documentation
- **Registry Helper Documentation** - Helper function usage and examples

### Changed

#### Breaking Changes - Repository Structure
- **Framework** → Pip-installable package (`alpha-berkeley-framework`)
- **Applications** → Separate repositories (production) or templates (tutorials)
- **`interfaces/`** → `src/framework/interfaces/` (pip-installed)
- **`deployment/`** → `src/framework/deployment/` (pip-installed)
- **`src/configs/`** → `src/framework/utils/` (merged)

#### Breaking Changes - Import Paths
```python
# OLD ❌
from applications.my_app.capabilities import MyCapability

# NEW ✅
from my_app.capabilities import MyCapability
```

All `applications.*` imports must be updated to package names.

#### Breaking Changes - CLI Commands
```bash
# OLD ❌
python -m interfaces.CLI.direct_conversation
python -m deployment.container_manager deploy_up

# NEW ✅
framework chat
framework deploy up
```

#### Breaking Changes - Configuration
- **Per-application config** - Each app has own `config.yml`
- **No global framework config** - Self-contained configuration
- **`registry_path` required** - Explicit registry file location
- **All settings visible** - Complete transparency (~320 lines)

#### Breaking Changes - Discovery
- **Explicit path-based discovery** - No automatic `applications/` scanning
- **Registry must be importable** - Proper Python package structure required
- **Exactly one provider per file** - Strict enforcement

### Enhanced

#### Performance
- **Lazy Loading CLI** - Heavy dependencies loaded only when needed
- **Fast Help Display** - `framework --help` loads instantly
- **Immediate Code Changes** - No reinstall/rebuild required

#### Developer Experience
- **Template-Based Generation** - New projects in seconds
- **Registry Helpers** - 70% less boilerplate code
- **Health Diagnostics** - Comprehensive validation with one command
- **Self-Contained Config** - All settings in one place
- **Natural Imports** - Module paths match package structure

#### Backward Compatibility
- **Legacy entry points maintained** - `alpha-berkeley`, `alpha-berkeley-deploy` still work
- **Registry interface preserved** - `RegistryConfigProvider` unchanged
- **Core functionality maintained** - All framework features work as before

### Migration Guide

#### For Production Applications
1. Install framework: `pip install alpha-berkeley-framework`
2. Create new repository structure
3. Copy application code to new structure
4. Update import paths (find-and-replace `applications.` → ``)
5. Simplify registry with `extend_framework_registry()`
6. Create self-contained `config.yml`
7. Setup `.env` file with API keys
8. Validate with `framework health`
9. Test functionality with `framework chat`
10. Initialize git repository and push

#### For Tutorial Applications
Regenerate from templates:
```bash
framework init my-weather --template hello_world_weather
framework init my-turbine --template wind_turbine
```

#### Complete Instructions
See comprehensive migration guide:
https://als-apg.github.io/osprey/getting-started/migration-guide

### Implementation Stats
- **100+ tasks completed** across 6 implementation phases
- **CLI infrastructure** - 5 commands with lazy loading (~2000 lines)
- **Template system** - 3 app templates + project + services
- **Registry helpers** - `extend_framework_registry()` (~200 lines)
- **Migration guide** - Comprehensive documentation (~730 lines)
- **Health diagnostics** - System validation (~968 lines)

### Related Issues
- Implements [#8 - Decouple Applications from Framework Repository](https://github.com/thellert/alpha_berkeley/issues/8)

## [0.6.0] - 2025-10-14

### Added
- **Performance Optimization System**: Configurable bypass modes for task extraction and capability selection
- **Task Extraction Bypass**: Skip LLM-based task extraction and use full conversation context for downstream processing
- **Capability Selection Bypass**: Skip LLM-based classification and activate all registered capabilities
- **Runtime Slash Commands**: Added `/task:off`, `/task:on`, `/caps:off`, `/caps:on` for dynamic performance control
- **Configuration Support**: New `agent_control` section in config.yml with bypass settings and system-wide defaults
- **Comprehensive Documentation**: Added bypass mode documentation with use cases, tradeoffs, and real CLI examples

### Enhanced
- **Gateway**: Parse and apply new performance bypass slash commands with readable command formatting
- **Task Extraction Node**: Implement bypass logic that formats full chat history and data sources without LLM processing
- **Classification Node**: Implement bypass logic that activates all capabilities without LLM analysis
- **State Manager**: Add bypass configuration defaults to agent_control state
- **Documentation**: Cross-referenced gateway, task extraction, and classification docs with performance configuration section

### Fixed
- **Data Source Request Creation**: Fixed user_id extraction to properly use session info instead of non-existent state field

### Performance Benefits
- Reduced LLM call overhead in preprocessing pipeline (1-2 fewer LLM calls per request)
- Flexible performance tuning for R&D, debugging, and high-throughput scenarios
- Trade orchestration complexity for extraction/classification speed based on use case
- Configurable via both system defaults and runtime slash commands

## [0.5.1] - 2025-10-13

### Fixed
- **Task Extraction Data Integration**: Enhanced task extraction to properly format retrieved data content from external sources
- **LLM Context Quality**: Improved the quality of context provided to task extraction for better results
- **Data Source Formatting**: Added robust fallback handling for data source content formatting

## [0.5.0] - 2025-09-26

### Added
- **ALS Assistant Application**: Complete domain-specific application for Advanced Light Source operations
- **PV Finder Service**: Intelligent EPICS process variable discovery with MCP integration
- **Application Launcher Service**: Desktop integration with MCP protocol support
- **Comprehensive Knowledge Base**: ALS accelerator objects database, PV naming structures, and MATLAB codebase analysis
- **Observability Integration**: Langfuse support with Docker containerization
- **Data Analysis Capabilities**: 7 new capability modules for accelerator physics operations
- **Infrastructure Services**: MongoDB database service, container orchestration for specialized services

### Enhanced
- **Container Execution**: Improved WebSocket connectivity, proxy handling, and error recovery
- **UI State Management**: Renamed `ui_notebook_links` to `ui_captured_notebooks` for clarity
- **Documentation**: Complete RST documentation with architectural diagrams and setup guides
- **Benchmarking Suite**: Performance analysis tools and model comparison frameworks

### Technical Details
- Added 144 new files with 430,647 lines of code
- Integrated MCP (Model Context Protocol) for external service communication
- Enhanced Docker compose templates with Langfuse environment variables
- Added comprehensive test coverage for core ALS services
- Implemented specialized databases for accelerator operations (11k+ PVs, AO structures)
- Enhanced framework capabilities with domain-specific prompt engineering

This release represents the framework's first complete domain-specific application, demonstrating the capability-based architecture's effectiveness for specialized scientific computing environments.

## [0.4.5] - 2025-09-23

### Added
- **Centralized Launchable Commands System**: New infrastructure for registering and displaying executable commands (web apps, desktop tools) through both CLI and OpenWebUI interfaces
- **Enhanced UI Result Display**: Comprehensive display system for figures, commands, and notebooks with rich formatting and metadata
- **MCP Protocol Support**: Added `fastmcp` dependency for Model Context Protocol integrations

### Enhanced
- **CLI Interface**: Added comprehensive result display methods with formatted output for figures, commands, and notebooks
- **OpenWebUI Interface**: Refactored result extraction with improved command and notebook handling
- **Configuration Management**: Enhanced path resolution with host/container awareness and application-specific file paths
- **State Management**: New `ui_launchable_commands` registry and `StateManager.register_command()` method
- **Response Generation**: Updated prompts to handle command display with interface-aware formatting

### Improved
- **Documentation**: Reorganized static resources following Sphinx best practices
- **Service Configuration**: Streamlined deployed services configuration with better maintainability
- **Error Handling**: Enhanced logging and fallback mechanisms throughout UI components

### Technical Details
- Added `ui_launchable_commands` field to AgentState for centralized command registry
- Implemented command registration system for capability-agnostic command handling
- Enhanced `get_agent_dir()` with `host_path` parameter for container/host path control
- Updated response context with `commands_available` field for UI awareness
- Improved container environment detection and path resolution

## [0.4.4] - 2025-09-17

### Refactored
- **Example Formatting System**: Consolidated example formatting with unified `BaseExample.join()` static method
- **Code Deduplication**: Removed duplicate `format_examples_for_prompt()` methods from `OrchestratorExample` and `ClassifierExample` subclasses
- **Flexible Formatting Options**: Added configurable formatting with support for separators, numbering, randomization, and example limits
- **Bias Prevention**: Maintained randomization for classifier examples to prevent positional bias in few-shot learning
- **API Consistency**: Unified formatting interface reduces maintenance burden for future example types

### Technical Details
- Added `BaseExample.join()` with parameters: `separator`, `max_examples`, `randomize`, `add_numbering`
- Updated `classification_node.py` to use `join()` with randomization for bias prevention
- Updated prompt builders (`memory_extraction.py`, `orchestrator.py`) to use `join()` with numbering
- Maintains all existing formatting behavior while reducing code duplication by 23 lines

## [0.4.3] - 2025-09-13

### Enhanced
- **OpenWebUI Interface**: Added notebook link display functionality with comprehensive response integration
- **Response Generation**: Enhanced prompts with notebook awareness and interface-specific guidance for better user experience
- **Context Loading**: Improved logging and registry initialization for better debugging and error handling

### Improved
- **Wind Turbine Application**: Refactored response generation guidelines with streamlined structure and cleaner code organization
- **User Experience**: Better integration of text responses, figures, and clickable notebook links in OpenWebUI
- **Debugging**: Replaced print statements with proper logging throughout context loading system

### Technical Details
- Added notebook link extraction and display in OpenWebUI response pipeline
- Enhanced response prompts with conversational guidelines and notebook availability context
- Improved context loader with registry initialization for proper context reconstruction

## [0.4.2] - 2025-09-13

### Enhanced
- **Python Execution Integration**: Python capability now registers notebooks using centralized StateManager.register_notebook() with rich metadata
- **Notebook Link Generation**: Improved notebook URL generation in both local and container execution modes with FileManager integration
- **Notebook Structure**: Enhanced notebook cell organization with separate markdown headers and executable code blocks

### Technical Details
- Added notebook registration with execution time, context key, and code metrics to Python capability
- Standardized notebook naming to 'notebook.ipynb' across execution modes
- Improved notebook generation with cleaner separation of results documentation and executable code

## [0.4.1] - 2025-09-13

### Enhanced
- **Centralized Notebook Registry**: Added structured notebook registry system replacing simple link list with rich metadata support
- **StateManager Enhancements**: Added `register_notebook()` method for capability-agnostic notebook registration with timestamps and metadata
- **Response Context Tracking**: Enhanced ResponseContext to track notebook availability for improved user guidance

### Technical Details
- Replaced `ui_notebook_links` with structured `ui_captured_notebooks` registry in agent state
- Added notebook registration method supporting display names, metadata, and automatic timestamp generation
- Updated state reset logic to use new registry format for better notebook management

## [0.4.0] - 2025-09-12

### Major Features
- **Context Memory Optimization**: Added recursive data summarization with `recursively_summarize_data()` utility to prevent context window overflow
- **Configurable Python Executor**: Complete Python executor configuration system with `PythonExecutorConfig` class for centralized settings
- **Enhanced Figure Registration**: Added batch figure registration support with accumulation for improved performance
- **OpenWebUI Performance Optimizations**: Response chunking for large outputs (>50KB) and static URL serving for figures

### Fixed
- **Critical Infinite Loop Bug**: Fixed infinite reclassification loop when orchestrator hallucinated non-existent capabilities
- **Reclassification Limit Enforcement**: Router now properly enforces `max_reclassifications` limit for all reclassification paths
- **Dependency Issues**: Fixed OpenTelemetry version constraints to resolve compatibility issues
- **Error Handling**: Enhanced retry logic and error classification in infrastructure nodes

### Changed
- **Unified Error Handling**: Consolidated reclassification system to use single error-based path instead of dual state/error approaches
- **Context Method Naming**: Renamed `get_human_summary()` to `get_summary()` across all context classes with backwards compatibility
- **Infrastructure Node Improvements**: Infrastructure nodes now raise `ReclassificationRequiredError` exceptions instead of directly manipulating state
- **State Cleanup**: Removed obsolete `control_needs_reclassification` field from agent state

### Enhanced
- **Python Executor Improvements**: Configurable execution timeouts, better error handling, and improved figure collection in both local and container modes
- **Context Window Management**: Automatic truncation of large execution results and code outputs to manage LLM context limits
- **Deployment Configuration**: Updated for static file serving with proper environment variable support
- **Error Classification**: Better distinction between retriable LLM failures and configuration errors

### Technical Details
- Added `ReclassificationRequiredError` exception to framework error system
- Enhanced router error handling to enforce limits consistently across all reclassification triggers
- Updated orchestrator and classifier to use proper exception-based error handling
- Improved architecture with cleaner separation between error handling and state management
- Added python_executor configuration section with sensible defaults
- Implemented graceful fallback for legacy method names with deprecation warnings

## [0.3.1] - 2025-09-10

### Enhancements
- **Documentation Workflow Improvements**: Added manual trigger capability to GitHub Actions documentation workflow
- **Tag-based Documentation Rebuilds**: Documentation now automatically rebuilds when version tags are created or moved
- **Enhanced Build Controls**: Documentation workflow now supports both automatic (tag/push) and manual triggering

### Bug Fixes
- **Documentation Version Sync**: Fixed issue where moving git tags didn't trigger documentation rebuilds, ensuring docs always reflect current version
- **Gitignore Cleanup**: Added `.nfs*` pattern to gitignore and fixed malformed entries

### Technical Details
- Added `workflow_dispatch` trigger to `.github/workflows/docs.yml` for manual execution
- Added `tags: ['v*']` trigger for automatic rebuilds on version tag changes
- Updated deployment conditions to support manual and tag-based triggers
- Improved build artifact and deployment logic for consistent documentation updates

## [0.3.0] - 2025-09-09

### Features
- **Interface Context System**: Added runtime interface detection for multi-interface support (CLI, OpenWebUI)
- **Centralized Figure Registry**: Implemented capability-agnostic figure registration system with rich metadata
- **Enhanced Figure Display**: Added automatic base64 figure conversion for OpenWebUI with interface-aware rendering
- **Real-time Log Viewer**: Added `/logs` command to OpenWebUI for in-memory log viewing and debugging
- **Robust JSON Serialization**: Comprehensive serialization utilities for scientific objects (matplotlib, numpy, pandas)

### Framework Enhancements
- **Interface-Aware Response Generation**: Context-sensitive prompts and responses based on interface capabilities
- **Python Executor Improvements**: Enhanced error handling and metadata serialization with fallback mechanisms
- **State Management Updates**: Centralized figure registry with capability source tracking and timestamps
- **Configuration System**: Added `get_interface_context()` for runtime interface detection

### Technical Improvements
- **Serialization Utilities**: Added `make_json_serializable()` and `serialize_results_to_file()` for robust data handling
- **Path Resolution**: Capability-agnostic figure path resolution for different execution environments
- **Error Handling**: Enhanced Python executor with detailed error reporting and serialization failure recovery
- **UI Integration**: Seamless figure display with metadata and creation timestamps

## [0.2.2] - 2025-08-16

### Major Features
- **New RECLASSIFICATION Error Severity**: Added `RECLASSIFICATION` severity level to ErrorSeverity enum for improved task-capability matching
- **Enhanced Error Classification Workflow**: Capabilities can now request reclassification when receiving inappropriate tasks
- **Reclassification Routing Logic**: Router node now properly handles reclassification errors with configurable attempt limits

### Breaking Changes
- **ErrorClassification Metadata Migration**: Replaced custom error fields with unified metadata field in ErrorClassification
  - `format_for_llm()` now generically processes all metadata keys
  - Enhanced error context richness for better LLM understanding
  - All infrastructure nodes and capabilities updated to use metadata field
  - Maintains backward compatibility through systematic migration

### Framework Enhancements
- **Enhanced Classification Node**: Improved reclassification workflow with proper failure context handling
- **Router Node Improvements**: Added reclassification attempt tracking and routing logic
- **Execution Limits Configuration**: Added support for configurable reclassification limits
- **Error Node Enhancements**: Comprehensive error handling improvements with better metadata processing

### Documentation & Examples
- **Major Documentation Cleanup**: Removed outdated markdown files and enhanced RST documentation structure
- **Enhanced Hello World Weather Example**: Added comprehensive classifier examples and improved context access details
- **Error Handling Documentation**: Complete documentation updates for new reclassification workflow
- **API Reference Updates**: Enhanced error handling API documentation with examples and usage patterns
- **Developer Guide Improvements**: Updated infrastructure components documentation

### Infrastructure Improvements
- **Framework-wide Capability Updates**: All capabilities updated to use new ErrorClassification metadata approach
- **Enhanced Time Range Parsing**: Improved time range parsing capability with better error handling
- **Configuration System Updates**: Enhanced config system to support execution limits and reclassification controls

### Technical Details
- Enhanced error classification system enables better task-capability matching
- Unified metadata approach provides richer context for error analysis and recovery
- Reclassification workflow prevents infinite loops with configurable attempt limits
- Complete migration maintains backward compatibility across the entire framework

## [0.2.1] - 2025-08-11

### Critical Fixes
- **Containerized Python Execution**: Fixed critical bug where execution metadata wasn't being created in mounted volumes
- **Container Build Failures**: Removed obsolete python3-epics-simulation kernel mounts that caused build failures
- **Path Mapping**: Fixed hardcoded path patterns in container execution using config-driven approach
- **Timezone Consistency**: Standardized timezone across all services with centralized configuration

### Security & Stability
- **Repository Security**: Updated .gitignore to exclude development services and sensitive configurations
- **Network Security**: Renamed container network from als-agents-network to alpha-berkeley-network for consistency
- **Service Cleanup**: Removed mem0 service references and cleaned up leftover container code

### Developer Experience Improvements
- **Configuration System Refactoring**: Renamed `unified_config` module to `config` for improved developer experience
- **Professional Naming**: Replaced `UnifiedConfigBuilder` with `ConfigBuilder` to eliminate confusing terminology
- **Automatic Environment Detection**: Added container-aware Python environment detection for convenience
- **Graceful Ollama Fallback**: Implemented automatic URL fallback for development workflows
- **Documentation**: Updated all references across 43+ files to use consistent naming conventions

### Infrastructure Enhancements
- **Git-based Versioning**: Added automatic version detection from git tags in documentation
- **Path Resolution**: Replaced hardcoded paths with configuration-driven approach using `get_agent_dir()`
- **Container Integration**: Improved container execution reliability and error handling
- **Documentation Cleanup**: Enhanced error handling documentation and API references

### Technical Details
- Fixed 'Failed to read execution metadata from container' error through proper volume mounting
- Eliminated manual reconfiguration when switching between local and containerized execution
- Complete refactoring eliminates confusing "unified" terminology from LangGraph migration era
- Added proper timezone data (tzdata) package in Jupyter containers for accurate timestamps
- Maintains backward compatibility through systematic import updates across entire codebase

## [0.2.0] - 2025-01-31

### Added
- Enhanced execution plan editor with file-based persistence
- Comprehensive approval system with human-in-the-loop workflows
- Complete advanced wind turbine tutorial application
- Improved documentation with execution plan viewer
- Execution plan viewer JavaScript support for interactive documentation

### Changed
- Modernized docker-compose configurations
- Enhanced framework robustness and capabilities
- Improved documentation build system and content

### Fixed
- Repository hygiene improvements with better .gitignore
- Removed deprecated version fields from docker-compose files
- Cleaned up PID files from repository

## [0.1.1] - 2025-08-08

### Fixed
- Remove invalid retry_count parameter from ErrorClassification calls in infrastructure nodes
- Fix runtime error: `ErrorClassification.__init__() got an unexpected keyword argument 'retry_count'`
- Update documentation examples to reflect correct ErrorClassification API usage
- Complete migration from dual retry tracking to state-only retry tracking

## [0.1.0] - 2024-12-XX

### Added
- Core capability-based agent architecture
- LangGraph integration for structured orchestration
- Complete hello world weather agent tutorial
- Framework installation and setup documentation
- API reference documentation (actively being developed)
- Developer guides covering infrastructure components
- Container-based deployment system
- Basic CLI interface for direct conversation
- Memory storage and context management systems
- Human approval workflow integration
- Error handling and recovery infrastructure

### Documentation
- Getting started guide with installation instructions
- Complete hello world tutorial with working weather agent
- Early access documentation warnings across all sections
- API reference for core framework components
- Developer guides for infrastructure understanding

### Known Limitations
- Documentation is under active development
- Some advanced tutorials not yet included
- APIs may evolve before 1.0.0 release

---

*This is an early access release. We welcome feedback and contributions!*
