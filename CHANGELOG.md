# Changelog

All notable changes to the Osprey Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Calendar Versioning](https://calver.org/).

Versions follow `YYYY.MM.MICRO`. Year and month identify the release window;
the micro segment increments for hotfixes and same-month follow-up releases.
Compatibility is documented in release notes, not encoded in the version string.

## [Unreleased]

### Fixed

- The build-drift gate no longer counts material `osprey up` mints itself: the
  per-lane Bluesky CURVE certificates now live under `data/.runtime/`, a
  reserved runtime-output subpath the fingerprint never hashes and the build
  never stages, so a started deployment stays CLEAN and the scaffolded systemd
  unit's bare `osprey up -d` comes back after a reboot (#716). Existing key
  material at the old `data/<lane>_curve/` path is relocated — not re-minted —
  on the next `osprey up`.
- Target switch: a configured-but-unserved `write_access` gateway no longer strands a
  write-armed session off the baseline. When the write-role readiness probe fails, the
  switch retries through the `read_only` gateway and lands with a warning; writes on
  such a session still go through the unchanged write path and are refused at the
  read-only gateway. (#718)
- Target switch: a readiness-probe failure now names the gateway role, host and port
  it probed — in the error detail, the structured `details.gateway`, and an
  actionable suggestion — instead of only the probe channel, which misread as "the
  control system is down" when a single gateway beside a healthy one was unserved.
  (#718)

### Added

- Build interview: an upstream fit watch — places where OSPREY cannot express what a
  facility needs are recorded as candidates in `INTERVIEW.md`, verified against the
  installed framework by a scout, and offered to the user as a GitHub issue or email
  to the maintainers. Nothing is sent without the user approving the full text.

### Security

- Every OSPREY interface — the Web Terminal, ARIEL, Channel Finder, the artifact
  gallery, the Theme Lab and the rest — now authenticates every HTTP and
  WebSocket request, and every launcher that serves one prints a login URL
  (`http://<host>:<port>/?token=…`) that trades the token for a session cookie
  and redirects to the clean address. The URL is printed once, but its token is
  the server's own secret and stays valid for the life of the process: treat it
  like a password. A backgrounded (`--detach`) server keeps it only in memory,
  so if you lose the URL, stop and restart to mint a new one.
- Processes the agent spawns no longer carry that credential, so the agent
  cannot call the interface APIs that change safety-relevant configuration, and
  the checks themselves run outside its reasoning. As a consequence, the Bluesky MCP
  server's queue-launch tool no longer works from the Web Terminal agent;
  launching a queue from a panel is unaffected.
- The operator secret no longer lingers in the server process's environment:
  each interface app clears it as it is built, so an agent the server spawns
  cannot read it back — on the ordinary foreground launch and inside a per-user
  container alike. One residual on Linux: a process that already received the
  secret through `exec` keeps a copy in `/proc/<pid>/environ`, readable by the
  same user.
- `osprey chat` starts its agent without `*_LAUNCH_TOKEN` credentials too, so the
  Bluesky queue-launch tool is disarmed in a chat session as well; panel-click
  launches from the Web Terminal are unaffected.
- The terminal's agent can arrange panels again. The low-privilege panel token
  now reaches the terminal's agent and the chat agent both, so panel tools and
  session hooks no longer fail silently with a 401.
- Browsing to a gated page without a session now returns a readable "not signed
  in" page pointing back at the login link; API calls still get a JSON 401.
- Mutating requests and WebSocket upgrades are checked against the deployment's
  external origin on every credential path. A multi-user stack behind nginx must
  have that origin configured — the generated compose files set it.
- The Web Terminal's panel proxy now strips `Set-Cookie`, `Clear-Site-Data`,
  `Refresh`, `WWW-Authenticate` and `Access-Control-*` from panel responses, and
  relays a backend redirect instead of following it so the browser's re-request
  is authenticated again. A panel that redirects to a third site is refused with
  a 502 rather than forwarding the operator, so a panel URL can no longer serve
  as an open redirect on the terminal's own address. The event-dispatch token is
  injected only toward a loopback events panel declared in your config.
- Multi-user: `osprey up` mints a per-user terminal secret for every roster entry
  in every auth mode, so a deployment with authentication off — or a single
  `login: false` entry — is gated by its own secret rather than open to anyone
  who can reach the port. Hand each person their URL with the new `osprey users
  login-url <user>`, and rotate one by deleting its `OSPREY_TERMINAL_SECRET_*`
  line from `.env` and re-running `osprey up`. The verb refuses for a user
  behind the login wall, who signs in through the login page instead.
- New `modules.web_terminals.external_origin` key: the address browsers actually
  reach the deployment at. Required when something else terminates TLS in front
  of the stack's nginx — without it the terminals refuse every write from that
  address as cross-origin. `osprey up`'s closing card now names this address,
  not the loopback one.
- Multi-user hardening: the nginx access log no longer records query strings or
  `Referer`; a per-user container never receives another user's cookies and
  `Authorization` is cleared on every proxied terminal request; roster names are
  held to `[a-z0-9][a-z0-9_-]*` in every auth mode, not only behind a login wall.
- `osprey build` now refuses a profile in which `Bash`, `Edit`, `Write`,
  `MultiEdit` or `NotebookEdit` is neither in `permissions.deny` nor covered by a
  `PreToolUse` hook matcher; a tool covered only by a matcher the profile itself
  declares builds with a warning instead. The `memory-guard` hook's matcher
  widened from `Write` to `Write|MultiEdit|NotebookEdit`.
- `DISPATCH_WORKER_TOKEN` is now stripped from sandboxed executions alongside the
  other credentials.
- Readonly Python executions are now enforced at runtime, not just by
  pattern scanning: the sandbox refuses every direct control-system write
  entry point (however the call is spelled), the connectors refuse
  `write_channel`, and the EPICS connector stays on the read-only gateway.
  Readonly runs also can no longer import control-system client libraries
  (`epics`, `p4p`, `caproto`, `pvaccess`, `tango`) — reads go through
  `read_channel()`.
- The runtime refusal now covers the routes that reach a control system
  without importing a client: caproto, pvaPy and Tango entry points resolved
  dynamically, starting a process (a shelled-out `caput`), and loading a
  shared library through `ctypes`. A readonly script therefore cannot shell
  out or open a shared library at all; resubmit such work as readwrite.
- A refused write is now visible, not silent. The operator is alerted that a
  write was attempted and blocked, and the attempt is recorded — with the
  offending source — in `var/audit/readonly-refusals.jsonl`, which survives
  builds and `osprey reset`.
- Readwrite Python executions now always require human approval under the
  `selective` policy, independent of whether the write-pattern scanner
  recognises the code's spelling.
- One protected set now names the files and `config.yml` keys the agent may
  not rewrite: the rendered config and settings, the hook and MCP wiring,
  `.env`, the `osprey_*` hooks, and everything under `.claude/skills/` and
  `.claude/rules/`. Every framework writer consults it — the scaffold
  galleries, the Claude-setup file API, the Config panel, ARIEL's config
  editor, the `setup_patch` tool and the restore that runs at session start —
  and a refusal names the channel that does own the file. Each blocked attempt
  is shown in the terminal's activity feed and recorded in
  `var/audit/protected-writes.jsonl`.
  Executed Python is held to the same line by zone: it cannot write into
  `build/`, the profile sources, or the audit ledger. See the "Protected Set"
  how-to.
- A terminal session can now be sandboxed without changing the deployment. A
  **Writes** / **Sandbox** badge in the card header switches the session's
  posture after a confirmation, restarting its agent and keeping the
  conversation. Sandbox refuses every control-system write and is enforced at
  the hook, the connector, the Python executor and every process the agent
  spawns. Postures are per session, persist across restarts in
  `var/agent_data/session-postures.json`, and a deployment rendered with
  writes off cannot leave the sandbox. See "Sandbox one session" in the Web
  Terminal how-to.
- Two new keys, `web.config_panel.enabled` and
  `web.scaffold_gallery.write_enabled` (both default `true`), close the Config
  panel with its `/api/config` and `/api/claude-setup` routes, and the scaffold
  gallery's write and delete routes, server-side; the browser withdraws the
  controls. Gallery reads stay open.
- `osprey profile validate` and `osprey build` refuse a `default_persona`, or
  a `login: false` roster entry, that resolves to a persona able to edit the
  deployment. With authentication off deployment-wide the check warns instead.
- `remove_deny` can no longer lift a deny the writes kill switch imposed. A
  profile that writes a permission list as a bare string, or spells
  `claude_code` both dotted and nested, is refused at build time instead of
  silently changing which tools end up denied.

### Fixed

- The graphdb container's healthcheck now probes with `wget`, which the Neo4j
  community image ships, instead of `curl`, which it does not — the container
  no longer sits at `unhealthy` forever while the store is serving.
- The documented removal spelling for the graph store works: a bare
  `services.graphdb:` in a profile's `config:` block now renders a project with
  no graphdb container, no `graphdb` in `deployed_services`, and no `graph` MCP
  server, instead of crashing the build. The ambiguous `services.graphdb: {}`
  is refused at validation with a message naming both working spellings.
- `osprey knowledge seed-graph` can succeed out of the box again: the graphdb
  service's default image moved to `neo4j:5.26-community`, the newest server
  line the neosemantics (n10s) plugin manifest covers. A store whose plugin
  never installed is now reported as "no compatible n10s plugin installed for
  Neo4j <version>" with the remedy, instead of a missing stored procedure.
- Dependabot pull requests no longer fail CI: the Deploy E2E lane now skips on
  Dependabot runs (which get no Actions secrets) like every other secret-gated
  lane, and the run summary names it among the lanes to revalidate. Dependency
  updates switched to the `uv` ecosystem — one grouped weekly PR bumping
  `uv.lock` (what CI actually tests) instead of ten PRs bumping the
  `pyproject.toml` floors, which stay as hand-maintained compatibility
  promises. A new lint step (`uv lock --check`) fails CI when the two files
  drift apart.
- Control-system failure envelopes now name the machine they happened on: a
  connect failure, a timeout, a limits refusal or a denied write carries the
  active target's name, label and endpoint (`details.active_target`, plus an
  `active target: ...` clause in the message). On a deployment with a runtime
  live/VA switch, "the write timed out" no longer leaves the operator to
  reconstruct *which machine* from session memory.
- Enum channel reads (EPICS `mbbi`/`bi`/`bo` and PVA `NTEnum`) now carry
  their state labels: the value stays the integer index on every protocol,
  and the reading's metadata adds `enum_label` (the state the channel is in)
  and `enum_labels` (every state in index order). An operator asking "what
  mode is the device in?" is told `ACQUIRING`, not `2`.
- A write the control system itself denies — EPICS access security answering
  a put with "Write access denied" — is now reported as a refusal naming the
  control system, instead of an `internal_error` that named nothing. The new
  `CONTROL_SYSTEM_REFUSED` reason states that the write was sent, that the
  control system turned it down, and that no value was written, so the
  operator is not sent to check OSPREY's own write settings for access the
  control system grants.
- Sub-agents that file their answer as an artifact (channel finder, knowledge
  graph, logbook search, facility knowledge, pyAT) now hand the orchestrator a
  pointer — the artifact id, a short headline and the identifiers asked for —
  instead of repeating the artifact, and the orchestrator focuses that artifact
  in the gallery rather than re-typing its tables into the chat.
- Readonly Python runs can import `at` (accelerator-toolbox) again: h5py's
  import reads the processor name, which CPython resolves by running `uname -p`,
  and the readonly guard refused that as a subprocess call — so the pyAT
  specialist could not load its own simulation library. The guard now resolves
  the value before it locks subprocess down; nothing user code can call was
  loosened.
- Container deployments now get the graph schema snapshot baked into the agent
  prompts. `osprey up` patched only the host renders, so every container shipped
  the "no snapshot yet" placeholder and each delegation paid a schema prelude.
  The channel finder's graph paradigm carries the same snapshot now, with its
  own example catalogue.
- Interactive plot artifacts now follow the active theme everywhere: opened in
  their own tab ("Open in new tab"), on first visit with a deployment-pinned
  `web.theme`, and under every theme family — not just light/dark inside the
  gallery. Previously a standalone plot page always rendered dark, and any
  theme outside the main family fell back to the dark palette. HTML and table
  artifacts without styling of their own now take the theme's colors too, and
  the print view is always light.
- Web-terminal personas now reach every service the hosting deployment runs.
  `osprey build` copies each client-facing fact — the qmd sidecar's port, the
  graph store's bolt port, the Postgres the logbook lives in, the telemetry
  store's port, the Bluesky bridge, the EVENTS and BLUESKY tab URLs — from the
  deployment's own render into every persona built beside it, so a service
  moved on the hosting profile moves every persona with it and the persona
  presets pin nothing. A persona whose `config:` contradicts one of these
  facts is refused, naming both values; one built alone is told what its app
  template deploys. Fixes hybrid logbook search failing in every persona with "no qmd
  sidecar is configured", and `osprey build` now refuses any consumer switched
  on with nothing to dial instead of rendering it. **Upgrading:** persona files
  written by an earlier `osprey init` (`personas/*.yml`) pin these facts
  themselves (`services.graphdb.port_host`, `web.panels.events.*`,
  `web.panels.bluesky.*`); `osprey build` refuses each such line by name —
  delete them, the build supplies the values.
- The qmd sidecar indexes the logbook mirror again: `qmd_export` wrote the
  mirror under `build/var/ariel_mirror` while the sidecar mounted
  `var/ariel_mirror`, so a live deployment searched an empty corpus.
  Config-relative paths (`mirror_path`, `facility_knowledge.bundle_path`,
  `ariel.vocabulary.path`) now resolve against the project root, the one
  anchor the compose mounts already use; `services.graphdb.ttl_path` alone
  stays relative to the render, because the corpus it names is read from the
  `data/` tree the build assembles for the project. A deployment
  with files under the legacy `build/var/ariel_mirror` regenerates the mirror
  with `osprey ariel qmd-resync`.
- Agent telemetry from per-user web-terminal containers is delivered again:
  the exporter posted to `http://openobserve:5080`, a name a host-networked
  container cannot resolve, and to the store's listen port rather than the
  port the deployment publishes. Per-user containers now post to the loopback
  address on `services.openobserve.port`.
- `osprey health`'s ARIEL rows probe the port the interface actually listens
  on (`OSPREY_ARIEL_PORT` in multi-user deployments), as the panel rows already
  did.
- Persona containers mount the hosting deployment's logbook mirror, so an
  entry created from a persona is indexed by the sidecar rather than written
  into the container's writable layer; each user's Python-executor refusal
  audit is kept on the host under `var/audit/<user>/` instead of being lost on
  recreate.
- The Bluesky MCP server and the virtual-accelerator connector follow a
  bridge or CA port moved on the hosting profile (`services.bluesky.port`,
  `services.virtual_accelerator.port`) instead of dialing compiled-in defaults.
- The BLUESKY tab works in web-terminal personas: the bluesky-web sidecar now
  accepts each user's own operator secret (the one their terminal proxies
  with) beside the deployment-wide one, for every user whose persona shows the
  tab — no user's container is handed the deployment secret, and no other
  process (the host's own `osprey web` included) accepts roster secrets.
- The EVENTS tab's label and health endpoint are written by the build where
  its URL is derived, so the tab health-gates itself on the host and in every
  persona; a web-terminal persona is told exactly the tabs it selects, and is
  refused rather than rendered tabless when the deployment it was told about
  runs no such sidecar.
- The graph channel finder follows the graph store like the graph MCP server
  does: a persona that switches that server off but keeps the channel finder
  is still told the store's address.
- `osprey up` refuses a persona that runs a logbook mirror export the
  deployment does not write (or writes elsewhere), instead of letting that
  persona's mirror land in its container's writable layer.
- The graph channel finder no longer mistakes its own row limit for the whole
  answer: `read_cypher` now warns when a result exactly fills the query's own
  `LIMIT`, the example catalogue gained a census query (count before listing
  "all of X") and a composed hardware-plus-signal filter that keeps sibling
  signals out of the match, and the agent's report now ends with a definitive
  **Channels found** list separated from any related addresses mentioned in
  prose.
- The cross-paradigm benchmark's "third vertical corrector" query now names
  the storage ring: the machine also has a third vertical corrector in the
  transfer line, so the old wording was ambiguous and graded a guess.
- The pyat-specialist grounding e2e no longer misreads a saved metadata key
  ending in a plane letter (an `index` next to the betas) as the horizontal
  value — a numerically correct results artifact could fail the check.
- The kill-switch e2e tests no longer fail when the agent makes a
  preparatory read before the denied write: the hook assertion is scoped to
  the write tool instead of requiring zero approval events overall.
- `osprey up` now refuses a Bluesky deployment on a Podman host still using
  the legacy `cni` network backend, naming netavark as the requirement and
  the `containers.conf` setting to change. Without aardvark-dns the
  dual-homed `bluesky-queueserver` receives only its first network's
  resolver, so it can never resolve `bluesky-redis`: it goes unhealthy and
  the deploy aborts before the web slice renders — the whole deployment down
  on a DNS fact nothing in the output pointed at. The check runs before any
  container is touched, is skipped unless `bluesky` is deployed, and never
  blocks a deploy on a host it could not read.
- The package inventory the agent sees no longer names compiled `__mypyc`
  shims as importable packages. The filter dropped them only when the shim's
  hash happened to start with a digit (making it a non-identifier); a
  letter-leading hash such as `ada92cb5d92a588d1b93__mypyc` passed straight
  through and was advertised to the agent as a top-level import.
- The session event log — and everything built on it: the agent-activity
  view, feedback bundles, logbook context, and the agent's own `session_log`
  tool — now records tool calls from every MCP server in the session instead
  of a fixed five. Calls from `osprey_facility_knowledge`, `phoebus`,
  `bluesky`, `health`, and any facility-declared custom server were silently
  dropped, so a sub-agent's work could vanish from a feedback report while
  the drill-down timeline still showed it.
- Tool names from servers with underscores in their name (`osprey_workspace`,
  `osprey_facility_knowledge`, or a facility's own) now display cleanly in
  the web terminal. The prefix strip assumed no underscores, so the activity
  strip showed "Mcp Osprey Workspace Submit Response" instead of
  "Submit Response", and the facility-knowledge tools never resolved their
  operator phrases ("browsing facility knowledge").
- The activity view's server badges are colored again for every framework
  server. The color map still said `workspace` after the server was renamed
  `osprey_workspace` (so those badges rendered grey), and the facility
  knowledge, Phoebus, Bluesky, and health servers never had a color at all.
  A facility's own server takes the neutral badge.
- A sub-agent that made no MCP tool calls no longer sorts to the top of the
  event log with an empty timestamp — its start/stop now carry the dispatch
  time of the Task that launched it.
- A custom MCP server whose configured name contains `__` (or ends in `_`)
  is now rejected at build time with a warning, like an `extends` clone
  name. Tool names are `mcp__<server>__<tool>`, so such a name would make
  the event log and the display misattribute every call the server makes.
- Logbook composition no longer sends tool arguments to the configured LLM
  provider — the session-activity section now carries tool names and result
  snippets only, matching the feedback report's privacy policy.
- Feedback sent by GitHub or Email no longer arrives without its session
  context because nobody knew about the paste step. The context cannot fit a
  prefilled URL, so with the context box ticked the flow is now one honest
  paste: the whole report (text, metadata, context) is copied to the
  clipboard, the draft opens with a single paste-here line instead of a
  partial body that a paste would duplicate, and the dialog states both steps
  before the button is pressed. Without context the draft opens complete and
  the clipboard is left alone. The browser and the full session id now ride
  in the composed report itself, and "Copy session context" moved out of the
  action row onto the checkbox's own row, enabled whenever a session exists.
- The prefilled issue title / mail subject is no longer the first line of the
  report text. It is now a stable "OSPREY feedback" title, suffixed with a
  short session id when context is attached.
- The rail's Documentation and Feedback controls now draw SVG marks (open
  book, speech bubble) instead of font glyphs. The old `ⓘ` read as "About"
  and rendered oversized; the old `✉` rendered small, painted full-colour
  emoji on Windows/Android, and promised email for a dialog that also files
  locally or to GitHub.
- The rail's Documentation and Feedback controls are now labelled DOCS and
  FEEDBACK, like every other cell in the rail. The marks alone left the
  operator hovering over them to find out what they open.
- The web terminal now knows its own session id from the moment it opens.
  It previously waited for the session's transcript file to appear on disk,
  which only happens once the session has content — so a terminal left idle
  for a few seconds never picked its id up at all, and never did later. That
  tab then behaved as if no session existed: feedback could not attach
  session context ("no session yet"), the session was not stored for resume,
  and every page reload started another one instead of reconnecting.
- Closing a web terminal no longer risks killing a session another tab is
  using. A tab whose agent had exited could, on close, terminate whatever
  session had since taken its place.
- A web terminal that cannot resume its stored session now falls back to a
  fresh one instead of stopping at "[Process exited]".
- A subagent's written summary no longer lands in the gallery category that
  holds its computed results. `submit_response` filed its Markdown under a
  category derived from the agent's name — so a pyAT summary and the JSON
  numbers it describes both appeared as "Lattice Analysis", and neither the
  gallery nor the agent could tell them apart. The summary is now an "Agent
  Response", and a `data_type` you pass is no longer overridden.
- The pyAT specialist no longer sometimes answers without filing its numbers.
  Its computed quantities used to be saved by a separate step inside the
  computing code that the agent could skip — and did, intermittently, leaving
  a prose answer with no "Lattice Analysis" JSON behind it. The numbers now
  travel with the answer: `submit_response` takes a `data` dict and files it as
  the JSON results artifact in the category the agent's definition declares
  (`results_category:` in its frontmatter), and refuses a hand-in that owes
  data and carries none. Prose and numbers arrive in one call, so there is no
  step to skip.
- Timeseries previews in the workspace gallery render again in CDN-mode
  (non-offline) deployments: the lazy Plotly loader now uses the same
  CDN-or-vendored URL resolution as the gallery's other vendor assets
  instead of hardcoding the offline-only path.
- `save_artifact()` inside `execute` now stores a Bokeh layout as a dashboard
  rather than as its text representation, and a `category` passed from
  visualization code stays on the artifact.
- The pyAT specialist and data visualizer no longer carry the
  file-registration tool; the specialist's results artifact is saved as JSON
  from the code that computed it.
- ARIEL keyword search now keeps raw and semantic full-text indexes in sync;
  semantic summaries and keywords reach keyword and hybrid retrieval immediately,
  and existing QMD mirrors backfill automatically after the renderer changes.
- The pyAT specialist now saves and confirms a structured lattice-analysis
  JSON artifact before returning computed optics results.
- Web terminal: selecting text now copies it, as in a desktop terminal. The
  agent's copy requests were being dropped by the browser terminal; Option/
  Shift-drag plus Cmd+C or Ctrl+Shift+C grabs raw screen text.
- Web terminal: session history, memory and transcripts are found inside the
  per-user container again (`CLAUDE_CONFIG_DIR` is honoured; readers no longer
  look one `.claude` too deep).
- `osprey up`: a persona whose config reads an authenticated archiver now
  receives the store's password variable in its terminal container.
- `archiver_read` no longer defaults to 1-second bins regardless of span
  (#117): without an explicit `bin_size` the bin is derived from the time
  range to target `archiver.auto_bin_points` (default 10 000) per channel, and
  the summary reports the bin used. Reads under ~2.8 hours are unchanged.

### Added

- `osprey health` gained a `reach` category, also shown in the system-health
  tab inside each user's container: one row per shared service a client is
  switched on for, knocked on at the address the client itself resolves. A
  closed port or a client with nothing to dial warns and names the config key.
- `osprey init` now prints a composition card under its report: who can sign
  in (each user's persona, rights, login method and port), what the agent runs
  on (model, MCP servers, bundled toolkit), what machine it talks to
  (connector, archiver, channel database) and what else runs beside it. The
  card is derived from the resolved profile, so it shows exactly the
  deployment about to be built — useful reading while a chained `--up` pulls
  its images.
- ARIEL search now speaks your control room's shorthand. Drop a
  `vocabulary.yml` in your project, set `ariel.vocabulary.enabled: true` and
  point `ariel.vocabulary.path` at it, and a search for `t/s bpm` also finds
  "troubleshoot" and "beam position monitor" —
  in keyword, semantic and hybrid mode. Every rewrite is reported back with the
  results, and `expand_query` turns it off for a single search. Gate the
  directions with `ariel.vocabulary.canonical_to_acronym` and
  `canonical_to_shorthand`, limit which modes expand with `expand_modes`, and
  validate every edit with `osprey ariel vocab-check` (no database needed).
  Keyword search also takes pattern tokens now — `SR01C___BPM*` globs and
  `/SR0[1-4]C___BPM\d+/` regular expressions, with
  `ariel.search_modules.keyword.settings.patterns_enabled` and
  `pattern_timeout_seconds`. The `control-assistant` template ships a
  twenty-concept example file; see the new "Vocabulary Expansion and Pattern
  Search" section of the Search Modes how-to.

- Web Terminal users can now send feedback from the terminal itself. A
  **Feedback** button in the panel rail (beside a new **Documentation** link)
  opens a dialog with three channels: **Local** records the report on this
  deployment, **GitHub** also opens a prefilled new issue, **Email** also opens
  a prefilled mail draft. Every channel records the report locally, and nothing
  leaves the browser until an action button is clicked. Two checkboxes decide
  what is attached: deployment metadata (on by default) and the session's event
  log, chat history, scrollback and artifact titles (off by default). Read the
  results back with `osprey feedback list` and `osprey feedback export`, which
  cover per-user workspaces and single-user deployments alike. Point the
  controls at your facility with `web.docs_url`,
  `web.feedback.github_repo`, `web.feedback.email` and
  `web.feedback.max_store_bytes`; blanking one of the first three retires that
  affordance for an air-gapped deployment. See the new "Send and Retrieve
  Feedback" how-to.

- A fresh `profile.yml` now shows its whole menu: every hook, rule, skill,
  agent, output style, and web panel you could opt into appears as a commented
  entry with a one-line description, right in the list it belongs to.
- `osprey profile artifacts` prints that same artifact menu from the command
  line, grouped by kind.

- Sites behind a TLS-intercepting proxy can now stay on verified TLS at every
  layer: the generated Dockerfile takes an `OSPREY_SITE_CA` build argument
  naming a site CA staged in the build context (installed before the first
  fetch, with `NODE_EXTRA_CA_CERTS`/`PIP_CERT`/`SSL_CERT_FILE`/
  `REQUESTS_CA_BUNDLE` pointed at the merged bundle), `osprey vendor fetch`
  honors a new `OSPREY_CA_BUNDLE` variable, and the `.env.shared` starter
  carries a commented site-CA block beside the proxy one.

- The OSPREY agent now knows which Bluesky plans a deployment ships with —
  what each one does and whether it moves the machine — so it can reuse an
  existing plan instead of writing a new one. New `bluesky-plans` skill,
  enabled in the `control-assistant` preset. The list is a build-time
  snapshot; `list_plans` still gives the live one.

- New read-only `get_plan_source` tool: the agent can read a plan's actual
  source instead of working from its one-line description.

- The Bluesky plan panel's Source tab now shows Python with standard syntax
  highlighting instead of one flat grey, using the same highlight.js and
  atom-one palette the Web Terminal and Artifacts gallery already use, so code
  reads the same way across the fleet and follows the light/dark theme mode.
  Long lines scroll rather than wrapping mid-identifier, and the view stays
  read-only. Offline deployments pick the highlighter up on their next `osprey
  vendor fetch`; until then the tab shows plain, uncoloured source.
- The multi-user landing page's text is now yours. Files listed in
  `modules.web_terminals.landing.notices` render as collapsible sections at the
  bottom of the page — safety guidance, local procedures, contacts, whatever
  operators should read first. Each file is markdown; its first heading becomes
  the section label. `osprey init` writes a starter
  `data/landing/working-safely.md` you can rewrite. Omit the key for the
  built-in safety notice, or set it to `[]` for none. `landing.footer` sets the
  line underneath.

- `osprey scaffold systemd` writes a systemd user unit that brings the
  deployment back after a reboot. It emits `osprey.service` at the repository
  root and prints how to install it; the unit runs `osprey up -d` and `osprey
  down` with the repository and the `osprey` program named as full paths. Boot
  start also needs `loginctl enable-linger $USER` once — see the deploy how-to.

- New `ariel.default_search_mode` setting names the search module that answers
  when a caller asks for no mode — the web interface's opening tab, `osprey
  ariel search` without `--mode`, and the service API. The shipped templates
  set it to `hybrid`. A name that no enabled module matches is refused at
  startup instead of being answered by a different mode; left unset, it
  resolves to `hybrid` where that module is enabled and `keyword` elsewhere.

- The `orbit_bump_sweep` plan can now assert bump closure at its monitor BPMs:
  an optional `leakage_tolerance` band judges every `readbacks` BPM against the
  reference orbit at each settled step, failing the run (or recording the miss
  under `best_effort`) when the bump leaks beyond it. Unset, monitors stay
  recorded-only as before.
- Web-terminal roster entries can opt out of the login wall with `login:
  false` — the entry is served without authentication while every other
  terminal stays gated, and no password is provisioned for it. Meant for
  entries that front a public read-only service, like the ARIEL card.
- Profile `env.defaults` values are now seeded into the repository's `.env` by
  `osprey init` (append-only; a value already set wins), so a preset can ship
  working starting values.
- The `control-assistant` preset now deploys its web terminals with password
  login on: Alice and Bob log in with demo passwords seeded into `.env`
  (`alice`/`alice`, `bob`/`bob` — edit there, or rotate with `osprey users
  passwd`), while the ARIEL terminal stays open via `login: false`.

- Build profiles can now author the shared web-terminal context baseline at
  `web-terminal-context/base.md`, overriding the framework's copy. `osprey
  init` materializes it from the preset (the control assistant ships its own
  text), so the context every seeded user starts from is visible and editable
  in the deployment repo instead of hidden in the installed package.
- The EPICS connector now reads pvAccess (PVA) channels alongside Channel
  Access, routed by a `pva_channels` glob so one deployment can mix both
  protocols; PVA writes are refused with a typed reason rather than
  attempted. Oversized reads — camera images and long waveforms, from either
  protocol — now come back as gallery artifacts (an interactive chart for
  1-D data, an image preview plus the raw array for 2-D and up) with an
  inline summary instead of flooding the context or being stringified, which
  also fixes oversized Channel Access waveform reads. p4p write patterns are
  now checked by the same pattern-detection, limits, and safety-rule layers
  as pyepics writes, and PVA RPC calls are refused outright.

- Channel fields in the BLUESKY panel's plan forms offer typeahead
  suggestions drawn from the project's Channel Finder catalog, snapshotted at
  build time. On by default when a channel database is configured;
  `web.channel_suggestions.enabled` and `web.channel_suggestions.max_channels`
  tune or disable it.

- A host that cannot build images can now run without building, with
  `prebuilt_images: true` in `config.yml` or `OSPREY_PREBUILT_IMAGES=1` for one
  shell. `osprey up` then starts the containers from the image tags already on
  the host, in dev mode and outside it. The persona and auth-sidecar images are
  a separate mechanism and still follow
  `modules.web_terminals.image_source`.
- Two new keys name the registry and tag that every OSPREY-built service image
  defaults to: `images.registry` and `images.tag` (or `OSPREY_IMAGE_REGISTRY`
  and `OSPREY_IMAGE_TAG` for one build), so a whole stack can be moved to a
  mirror or to a pipeline's tag in one place. A per-service `image` pin or
  `OSPREY_<SVC>_IMAGE` still wins; unset, images are named as before.
- The qmd search sidecar can take its three model files from a host directory
  instead of downloading them during the image build. Point
  `services.qmd.models_dir` at a directory holding the staged files: the build
  skips the downloads and the directory is mounted read-only into the
  container, with the same SHA256 check moved to container start. For build
  hosts with no route to the model host.
- `osprey up` warns, by name, when a service declares a host variable under
  `env:` that nothing on this host sets. The container would otherwise start
  with the variable set-and-empty. Advisory only; `env.required` is where a
  deployment says a variable is mandatory.
- The `control-assistant` preset now ships tiers named `readonly`, `readwrite`
  and `admin` beside the `ariel` research persona. Bob (`readonly`) has
  control-system writes off; Alice (`readwrite`) has them on, supervised;
  Carol (`admin`, demo password `carol`) has them on too and is the only
  terminal that may edit the deployment — the Config panel, the gallery editors
  and the approval-gated `setup_patch` tool. Each tier is its own profile
  (`control-assistant-readonly`, `-readwrite`, `-admin`, `-ariel`).

- A session can now be pointed at a different control system while it runs.
  `control_target_set` moves it between the real machine (`live`) and the
  virtual accelerator (`va`), and `control_target` reports where the session is
  and what else it could reach — so a script can be rehearsed on the simulator
  and then run on the machine without a rebuild, a redeploy or a restart. The
  destination is connected and proved reachable, by reading its
  `probe_channel`, before the current one is retired, so a switch that fails
  leaves the session where it was. Switching *to* the live machine also
  requires strict limits checking and
  `control_system.target_switch.live_gateway_acknowledged`, set to the live
  gateway's hostname; returning to a deployment's own baseline requires
  neither. No target choice outlives the session — every server start returns
  to the deployment baseline. `osprey build` now renders both targets'
  connector blocks, so a project from the standard template has a
  `virtual_accelerator:` block beside its `epics:` one. See the new "Switch the
  Control Target at Run Time" how-to.

- Approval prompts now name the control target they would act on — writes, plan
  starts, patches and executions alike — and archive reads carry the target and
  the archiver that served them.

- A deployment can render a second Bluesky plan lane, one per control target,
  with `bluesky.second_lane: true` in the build profile. Plans are then routed
  to the lane serving the session's target, and `queue_start` names the lane
  `queue_add` bound the plan to. Without it a deployment keeps its single lane
  and refuses to queue or start plans while the session is pointed at a target
  that lane does not serve.

- The Virtual Accelerator serves its physics model's own variables over
  pvAccess as well as Channel Access. A write arriving on either transport
  moves both views and a refused write moves neither, with one difference:
  a PVA put comes back carrying the model's own reason for refusing, where
  Channel Access can only signal an alarm and the absence of movement.
  Channel Access remains the authoritative view of the facility's whole
  channel namespace, and the only transport the container publishes; PVA is
  reachable only from inside the container.
- The Virtual Accelerator's physics is now pluggable. The ring is reached
  through a `LUMEModel`, so a surrogate or a different tracking code can be
  supplied without touching the layer that serves the channels, and the
  generic pyAT tracking behind the shipped ring model now comes from an
  upstream, facility-agnostic package, `lume-pyat`, rather than living
  inside OSPREY.
- The facility ontology is now authored as a LinkML schema (`demo_ontology.yaml`)
  and compiled to the JSON table format with `osprey knowledge compile-ontology`;
  `--check` fails if the compiled output would differ from what's checked in,
  guarding against drift. The shipped `demo_ontology.json` is now a generated
  artifact. `linkml-runtime` joins the `knowledge` extra.

### Changed

- Local-mode persona images are now tagged by their rendered project alone
  (`my-assistant-readwrite:local` instead of
  `my-assistant-readwrite-readwrite:local`). Lint newly rejects two personas
  sharing a `project` across different renders, and a persona `project` equal
  to the deployment's own name. Images under the old doubled tags are not
  removed automatically — prune them by hand after the next `osprey up`.
- The ARIEL search panel now follows the deployment's configuration for every
  knob you have not touched — Reset returns it to that state — and shows the
  fast results of a reranked search before the reranking finishes, replacing
  them when it does. When the reranker is unavailable the fast ranking stays on
  screen with a note instead of the search failing. The `hybrid_search` tool
  takes a per-query `rerank` argument; panel and tool share the one
  `ariel.search_modules.hybrid.settings.rerank` key.

- A malformed ARIEL `hybrid` or `semantic` search setting is now named at
  startup validation instead of passing silently, and a malformed per-query
  `rerank` or `candidate_limit` sent over HTTP is refused with a 400 rather
  than coerced.

- CI: the unit lane's step summary now tabulates the slowest tests of the run
  and the uploaded diagnostics artifact carries the full pytest log.
- `osprey-connectors` now versions with the framework's calendar stream —
  both wheels are built from one checkout and carry one number, so the
  independent `0.x` line (and the question of what counts as a minor) is
  gone.
- Docs site: the root now always shows the latest release; development docs
  moved to `/latest/` with a development banner.

- The workspace tool that registers a file on disk, or literal text, as a
  gallery artifact is now `artifact_register` (was `artifact_save`), and
  inline `content` must name its `content_type`. `save_artifact()` inside
  `execute` is unchanged and remains the way to save a live Python object.

- CI measures coverage on the Python 3.12 cell with PEP 669 tracing
  (`COVERAGE_CORE=sysmon`) rather than on the 3.11 cell, where coverage.py
  falls back to the C tracer. Same coverage numbers, roughly half the unit
  lane's runtime — it had grown to ~36 minutes against its own 40-minute cap.

- `osprey web` now says when it is using an `OSPREY_TERMINAL_SECRET` that was
  already set in its environment: it prints no login URL (the value is never
  echoed) and points at where that secret was minted. A multi-user deployment
  that supplied no secret now stops with a readable message naming the fix
  instead of a traceback.

- The `hello-world` preset now wires the `memory-guard` hook, eleven hooks in
  all. Its profile hash moved with it, so an already-deployed hello-world project
  reports the staleness advisory once; rebuild to clear it.

- The interface apps no longer register CORS middleware. Nothing OSPREY serves is
  cross-origin, so there was no request path left to allow or restrict.

- The agent's telemetry no longer authenticates to the local OpenObserve store
  as root. `osprey up` starts the store, creates a dedicated ingest service
  account, and saves the token the store issues into the project `.env`
  (`ZO_INGEST_USER_EMAIL` / `ZO_INGEST_SA_TOKEN`) — nothing to set by hand, and
  the root password now stays on the deploy host. A token that has stopped
  working, after the account was rotated or the `openobserve_data` volume was
  recreated, is replaced on the next `osprey up`. The ingest account cannot
  create or delete users, but it can read all stored telemetry and the user
  roster: OpenObserve has no ingest-only role in any edition. Existing projects
  pick this up on a rebuild; a `.env.users` written earlier needs
  `ZO_INGEST_USER_EMAIL` appended to it, and the deploy says so.

- Authored plans can now be retired. `DELETE /plans/session/{name}` on the
  Bluesky bridge removes a session-tier plan file, which until now stayed in
  the catalog until the container restarted. The plan leaves `GET /plans`
  immediately; anything already queued or running is unaffected.

- Any LLM gateway can now be a provider. Name it as `provider:` in
  `profile.yml`, describe it under `config:` as `api.providers.<name>`, and
  put its key in the project's `.env` as `<NAME>_API_KEY`. The preset
  profiles ship a worked example.

- `model:` now accepts any model ID the provider serves, not only an ID
  listed in the provider's tier map. A newly released model or a
  gateway-only alias works immediately; a misspelt ID fails at the provider,
  which names it, rather than at build time.

- Local model servers that need no API key — Ollama, vLLM, DS4 — no longer
  require an auth secret to deploy or to launch a web terminal. A missing
  secret for those providers is now a note rather than a refusal. Providers
  OSPREY does not recognise keep the strict behaviour.
- Generated container images bake no `USER`: a root entrypoint re-renders
  drifted agent artifacts and restores gallery-claimed bodies, then drops to
  the `osprey` user (uid 1000) via `gosu`. The render (`build/`, including
  `.claude/` and `config.yml`) is root-owned at runtime and `var/` belongs to
  the runtime user; `config.yml` stays writable only for a persona that may
  edit the deployment. Starting the container as another user (`docker run
  --user`, Kubernetes `runAsUser`) skips both startup steps, and
  `runAsNonRoot: true` now refuses the pod because the image names no user.
  Bare-host deployments have no second user and keep their self-healing
  startup.
- Config-write backups now land in `var/agent_data/config-backups/` instead of
  beside the file as `config.yml.bak`.
- On rebuild, a `control-assistant` deployment's `readonly` and `readwrite`
  terminals lose the Config panel and the gallery editors; only the `admin`
  tier keeps them. Put a roster entry on `admin` to keep an editing terminal.

- The documentation is reorganized around what you are doing: the how-to
  guides are grouped into task sections (deploy, LLM providers, web
  terminal, agent interfaces, health and monitoring, control systems,
  facility knowledge), and everything that is a lookup rather than a task
  moved into a new Reference section — the CLI (`reference/cli`),
  configuration, and the connector, ARIEL, Channel Finder, facility-graph
  and health-JSON contracts. The contributing guide splits into setup,
  workflow, docs and a new map of the framework's extension seams. Every
  old URL redirects, so existing links and bookmarks keep working.

### Changed

- The `osprey-build-interview` skill now creates your deployment repo up front
  and refines it with you in place, keeps its decisions in an `INTERVIEW.md`
  you can resume from, and reads every menu and default from the repo itself.

- The bundled Claude Code CLI moved from 2.1.191 to 2.1.228, via
  `claude-agent-sdk` 0.2.136.

- The multi-user landing page drops the drawn ASCII wordmark; the page
  now shows the text OSPREY at every window width.

- Logging out now starts where operators look for it: the chip in the web
  terminal's top-left corner naming the terminal. Clicking it opens a small
  session menu with the signed-in user and **Log out**. Log out is also in the
  display menu next to Settings, so it is reachable from either place; the
  identity line itself lives only in the chip.

- The control assistant writes for the control room. Its answers now follow a
  plain-language style: one idea per sentence, active voice naming the actor,
  abbreviations expanded on first use, and no filler. Facility terms like
  "abort" and "interlock" are kept exactly as the facility says them. The
  rules are in `.claude/output-styles/control-operator.md`, so a facility can
  claim that file and set its own house style.

- Error messages and documentation that told you to edit `config.yml` now
  name the build profile instead. `config.yml` is regenerated by every
  `osprey build`, so an edit there is silently lost; the remediation text now
  says to set the value under `config:` in `profile.yml` and rebuild.

- A provider tier with no model mapped no longer refuses the build. The tier
  falls back to the default model and the build warns, naming every
  substitution, so a partial tier map is usable while still being visible.
  A provider with no models *and* no default model at all is still refused.

- Generated `.env`, `.env.users` and `.env.auth` files now open with a short
  header saying which process reads the file, what belongs in it, and how it
  relates to the others. Existing files are left alone — the header is
  written only when OSPREY creates the file.

- The generated safety rules now name the tool for each kind of request:
  reading a channel goes to `channel_read`, writing to `channel_write`,
  analysis to `execute`, Python that itself touches the control system to
  `osprey.runtime`, and — where the queue is enabled — a scan to Bluesky. A deployment with no queue is told to
  say so rather than imitate a scan with a loop of writes. The rules describe
  how a write's verification level is resolved instead of prescribing one.

- The executor tool descriptions now match what the executor does: it runs
  Python in a subprocess. They no longer claim a container backend.

- Every `channel_write` result now carries one `write_state` word per channel
  in `summary.results[]` — `blocked`, `write_failed`,
  `verification_not_reported`, `verification_not_requested`,
  `verified_with_alarm`, `verified`, `readback_failed`, `unverified_alarm` or
  `verification_failed` — so a caller can tell a refused write from one that
  went out unverified. `summary.verification_failed` counts the executed
  writes that asked for callback or readback verification and did not get it;
  refused and failed writes are not counted.

- `channel_write.verification_level` is now optional. Omit it and the
  deployment resolves the level, and any readback tolerance, per write;
  `access_details.verification_level` is then null. A level you do pass is
  honored as written — the validator no longer overrides it, and it no longer
  costs you the tolerance held in the limits database.

- Batch writes and `osprey.runtime.write_channels` now resolve verification
  per channel the same way single writes do, instead of being pinned to
  callback. A project whose defaults ask for readback verification will see
  batch writes verified by readback, and the Python path raise when a readback
  does not match.

- A readback-verified write now reports the readback's alarm state by name:
  `readback_alarm_status` and `readback_alarm_severity`. Channel Access reads
  and subscriptions report alarm names too; the raw numeric code stays in
  `raw_metadata["status"]`.

- A write whose readback read itself fails is now reported as such —
  `failure_kind="readback_failed"` on EPICS, DOOCS and Mock — and DOOCS no
  longer labels a failed `set()` as verification level `none`.

- `WriteVerification` gained three optional fields for connector authors.
  Connectors that do not set them keep working; see the connector how-to.

- This partially addresses #465. Still open: carrying the post-write readback
  value at every verification level, flipping `success` when the readback read
  fails, and labeling an infrastructure failure differently from a
  control-system failure.

- A deployment can now run a facility knowledge graph as a service. Add
  `graphdb` to `deployed_services` alongside a `services.graphdb` block naming
  the image, host ports and a Turtle corpus (`ttl_path`), and `osprey up`
  starts the store, mints its `GRAPHDB_PASSWORD` into the project `.env`, and
  loads the corpus. `osprey knowledge seed-graph` re-seeds or re-checks it
  later, and `osprey health` reports bolt connectivity and how many resources
  the graph holds, under a new `graphdb` category. The `control-assistant` and
  `ariel-standalone` presets ship the block already filled in.
- Agents can query that graph through a new `graph` MCP server. `read_cypher`
  runs Cypher in a read-only transaction — writes, extension procedures and
  `LOAD CSV` are refused — `get_schema` reports the labels, relationship types
  and properties actually present, and `example_queries` serves a curated,
  parameterized set covering device rollups, channel bindings and section
  walks. `services.graphdb.query_timeout_s` (default 15 s) and
  `services.graphdb.query_max_rows` (default 200) bound a single query. The
  server is rendered wherever `services.graphdb` is configured, and its tools
  belong to a new `facility-knowledge-graph` subagent: the main agent answers
  structural questions ("what sits in section 7, in beam order?") by
  delegating to it, exactly as address lookups delegate to the channel finder.
  Safety, output-style, setup-mode and diagnose prompts name the agent where
  they enumerate its siblings.
- The store's schema and the curated examples are baked into that agent's
  rendered prompt at seed time: whichever verb seeds or re-verifies the store
  (`osprey up`'s staging step, `osprey knowledge seed-graph`) captures them
  from the live store — property lists complete, example parameters narrowed
  to the seeded corpus — and rewrites the prompt's snapshot section, stamped
  with the seed marker's checksum. The agent starts oriented instead of
  spending its first turn on `example_queries`/`get_schema`; both tools stay
  registered as the recovery path for a store re-seeded out of band.
- `osprey knowledge build-ttl` generates a Turtle corpus from a project's
  hierarchical channel database (`--channel-db`) and the matching in-context
  database (`--descriptions`), deriving devices from the channel-name grammar,
  each binding's read/write direction from the limits database, and the prose
  describing them from both. The `control-assistant` preset seeds its graph from
  that corpus, so its graph and channel finder describe the same machine.
- The `control-assistant-readonly` and `control-assistant-readwrite` operator
  terminals get the facility-knowledge-graph agent too, reading the hosting
  deployment's store over `services.graphdb.port_host`. Move that port on the
  deployment and move the same number in both preset files.

- The graph is available as a fourth channel-finder paradigm:
  `channel_finder.pipeline_mode: graph` (`osprey init --set
  channel_finder_mode=graph`) answers channel questions from the deployment's
  graph store instead of a database file. The channel-finder subagent gets four
  tools under the usual `channel-finder` server name — `read_cypher` for
  read-only Cypher, `get_schema`, `capabilities` for the corpus conventions a
  query has to follow, and `example_queries` for a catalogue of worked lookups.
  A `services.graphdb` block is the whole configuration, whether it deploys the
  store with the project or names an external one (`uri`, `username` and
  `GRAPHDB_PASSWORD`); there is no `pipelines.graph` block to fill in.
- The corpus the `control-assistant` preset ships describes its machine in
  words, so a graph-mode search can match an operator's phrasing: every channel
  binding carries a description plus its field and subfield text, every device
  carries family, system and ring descriptions and a `system` token, and
  synonyms are stored as lists.
- `osprey health` reports a graph-mode channel finder as store-backed — bolt
  reachability and the store's resource count, with no database file to look
  for — and the channel-finder web app serves a pane naming the graph tools,
  answering the file-backed routes (statistics, validation, pipeline switching)
  by pointing at `read_cypher`.
- The channel-finder benchmark runs a fourth lane, scoring the graph paradigm
  against the same tier-3 query set as the other three.

### Changed

- A missing or unknown `channel_finder.pipeline_mode` now fails with an error
  naming the modes that exist, everywhere the mode is read: build, agent
  render, `osprey health`, the channel-finder web app and the benchmark runner.
- A benchmark lane fails outright when any of its queries fails to run, so a
  score is only ever reported over the whole query set.

### Fixed

- The contribute and release skills' CI watch loops no longer treat an empty
  `gh` answer as success: no rows from `gh pr checks` or no matching run from
  `gh run watch` now reads as unknown, to be re-queried with an explicit
  PR/run selector before anything counts as green.
- Container builds no longer fail when one package download is cut short. apt
  now fetches one request per connection: with pipelining left at its default,
  a connection reset partway through a batch could fail the whole image even
  though retries were configured.
- `osprey up` no longer aborts on arm64 hosts, such as Apple Silicon, while
  building the web-terminal login sidecar. That image was the one recipe still
  missing the build pin the others already carry, so the EPICS packages it has
  to compile from source there failed and took the deployment down with them.
- The web terminal's agent-activity strip no longer appears to swallow keyboard
  focus while it is empty. On the pages where an empty strip is invisible it now
  reveals itself when focused, instead of being a tab stop with no visible
  outline. It stays reachable either way: the strip opens the recent-activity
  history, which lives on the server and is there after a reload even when the
  strip itself is empty.
- Channel suggestions in the Bluesky plan form are visible again. The popup
  opened correctly but was drawn inside containers that clip their overflow for
  rounded corners, so in a channel list it was cut to nothing and in a table
  cell to a two-pixel sliver — most channel fields appeared to offer no
  suggestions at all. It now floats above the form, and flips above the field
  when there is no room below.
- The arrow-key cursor in the channel suggestion popup is visible in the
  `light` theme, where it previously took the same colour as the popup behind
  it. Hover and the armed row are now told apart as well.
- Channel suggestions no longer go missing from the plan form the Bluesky panel
  opens with. That form reads the channel catalog as it is built and is not
  revisited afterwards, so whenever the catalog arrived a moment too late the
  form offered no suggestions for as long as it stayed open. It now waits for
  the catalog before rendering, and renders without suggestions rather than
  stalling if the catalog cannot be fetched.
- A service's `env:` passthrough list is no longer dropped for bluesky,
  bluesky_web, gchat_bridge, nextcloud_bridge, virtual_accelerator, mongodb and
  archiver_recorder. All ten services now honour it, in both the nested and the
  dotted spelling.
- Seeding a simulated logbook (`osprey sim apply`, and the deploy's own
  first-bring-up seed) now writes the markdown mirror the qmd sidecar indexes.
  Seeding skips the enhancement passes, so `hybrid` search previously searched
  an index nothing had built and returned no hits — which reads as "the logbook
  has nothing on that" rather than as a missing index.
- Container builds now hand apt the proxy settings they were given. A facility
  proxy arrives as `HTTP_PROXY`/`HTTPS_PROXY`, which apt does not read, so on a
  network with no direct egress every image build stalled in its first package
  install.
- The project and dispatch images install Node and npm from the base image's
  own Debian release instead of a third-party apt repository. That repository's
  setup step had stopped configuring anything on current base images, leaving
  them without npm and without a working agent launch path.
- Model files staged inside the project tree no longer end up in the built
  wheel, which they had been inflating to several gigabytes.
- The qmd sidecar now finds its own daemon on hosts where `localhost` resolves
  to IPv4. It previously probed only the IPv6 loopback, and where that was the
  wrong one the container crash-looped while the deploy still reported success.
- Agents now route a measurement that needs more than one setting through the
  Bluesky queue instead of stepping a setpoint with repeated `channel_write`
  calls. The `operating-bluesky-plans` skill also triggers on requests phrased
  as physics ("step a corrector across a few settings and record the beam")
  rather than only on the words *plan*, *run*, *queue* and *start*, and the
  control-system safety rule states the routing directly. Hand-stepping cost
  the operator one approval per write instead of one per measurement and left
  no run behind.
- A panel closed by an agent workspace arrange no longer pops back open when a
  browser's tab-switch report, sent before the arrange, arrives after it. The
  server drops the stale report instead of re-adding the panel to every
  client's rail and stealing the active tab.
- A relative `control_system.limits_checking.database_path` now anchors on the
  directory of the config actually loaded, so the limits gate finds the render's
  database regardless of how the process was launched. Previously a Claude Code
  hook running without `CONFIG_FILE` resolved it against the repo root and the
  empty-database failsafe denied every write (#636).
- The limits failsafe now refuses with "limits database unavailable" instead of
  reporting every channel as "not in limits database", so a load failure is no
  longer mistaken for a data problem (#636).
- Resizing a web-terminal pane no longer freezes the other panels until a
  browser refresh: the adapter's sash shield poisoned dockview's own
  pointer-events snapshot and is removed — dockview shields iframes during
  sash drags itself (#638).
- `osprey up` now refuses as a precondition, rather than failing with a generic
  "Deployment failed", when a project deploys the archiver store and pymongo is
  missing. The refusal names the interpreter it is missing from — OSPREY seeds
  the store from the process running the CLI, not from the project's
  `build/.venv` — so a `dependencies:` entry in the build profile is visibly
  the wrong lever.
- The `osprey` command no longer prints a Python traceback when something goes
  wrong. Its console script was wired straight to the Click group, past the
  handler that turns an error into a `✗` line with a cause and a remedy, so
  every failure no verb caught reached the terminal as a stack trace ending in
  installed-package paths. Affects every verb.
- `osprey init --reset` and `osprey init --up` now check for a running
  container runtime before they create anything. Both need one — `--reset` to
  read what the previous deployment owns, `--up` to start the new one — but the
  check ran after the repo had been written, git-initialized and committed, so
  a stopped Docker left a repo behind that nobody asked for.
- `osprey init --provider cborg` (and `--model`, `--connector`,
  `--channel-finder-mode`) now say which spelling works — `--set
  provider=cborg` — instead of suggesting the unrelated `--override`.
- Artifacts built from a dev checkout report the right version again. A release
  tag cut for the workspace sibling (`osprey-connectors-v0.1.0`) matched the
  build backend's default tag glob, so every wheel, editable install and
  container built off `main` since 2026-08-15 was stamped `0.1.0.postN` while
  `osprey --version` said `2026.6.2.postN`. The build now describes against the
  same `v[0-9]*` tags the runtime always has, the commit hash is pinned to one
  width instead of varying with clone size, and a test holds the two
  derivations byte-identical.
- `osprey init --reset` no longer crashes with a Python traceback when the
  containers it would remove belong to another copy of this repo. `osprey
  reset` has always caught that refusal and rendered it; this path never did,
  so the same deliberate guard looked like a bug in OSPREY depending on which
  verb you typed. It is a refusal now, and it says what is on disk afterwards.
- The same-name-different-checkout refusal leads with its conclusion. It used
  to open with the count and the identity hashes, print one line per resource,
  and only then explain that a worktree or a second clone shares its parent
  directory's name. On a real deployment that put the explanation and the way
  out about thirty lines below the top, where nobody reads them. Now the
  finding, the other copy's path and the remedy come first, each path is listed
  once instead of once per resource, and the per-resource evidence prints under
  `--verbose`. No claim changed: it still says only what the labels prove.
- `osprey init --reset` also offers the way out that destroys nothing. `reset`
  can only suggest going and wiping the other deployment; whoever ran `init`
  asked to create something, so deploying this copy under its own name is
  named too.
- A deploy whose web terminals are unreachable now says so on the terminal. The
  warning naming the Docker Desktop remedy was emitted with `logger.warning`,
  which the altitude gate drops while a lifecycle verb owns the terminal, and
  the root logger carries no other handler. So on the one path that mattered,
  a self-heal bounce that did not help, the run printed "bounced the web
  stack ..." and then went straight to the endpoint table, which reads as
  success. It is promoted through `warn_fact` now.

### Added

- `scripts/ci/flake_report.py` ranks flaky CI tests from GitHub Actions re-run
  history. A test counts as flaky only when it failed and then passed on the
  identical commit; failures that never went green are listed separately so a
  branch bug is never filed as a flake.
- `osprey up` and `osprey restart` warn in Preflight when Docker Desktop's
  "Enable host networking" is off and the deployment has web terminals. The
  post-up probe already caught this, but only after every image was built and
  every container was up, which on a first deploy is a quarter of an hour after
  the operator could have fixed it. Read from Docker Desktop's own settings
  (its backend API, falling back to the persisted settings store), so the
  warning names the cause instead of listing suspects. A setting that cannot be
  read stays silent and leaves the post-up probe to speak.
- The post-up reachability warning now separates a forwarder that is switched
  off from a port registration the running forwarder missed. A definite "off"
  skips the self-heal restart, which cannot help, and states the cause; an
  unreadable setting keeps bouncing first and then names the setting as
  something to check.

### Changed

- A finished `osprey up -d` now ends by saying what to do next: one line naming
  the landing page, and, when the roster's passwords are still the ones the
  profile declared, the logins to sign in with. URLs the CLI prints are
  clickable in terminals that support it.
- The "This deploy wrote" block is grouped by file instead of repeating the
  filename on every row, and its long values wrap under their own bullet rather
  than running off the terminal. It now prints before the closing summary card,
  so a run ends on where to go rather than on what it wrote.
- The web terminal no longer opens with a full-screen welcome banner. The
  OSPREY wordmark now heads the landing page, and the safety guidelines live at
  the bottom of that page instead of behind a link in a banner nobody sees
  twice.

- `osprey up` now writes the OpenObserve account name `ZO_ROOT_USER_EMAIL` into
  `.env` alongside the minted password, so both halves of the telemetry login
  are in one findable place. Previously only the password was written and the
  email existed solely as a default inside the templates. The value is
  unchanged (`root@example.com`), and a value you already set is never
  overwritten.
- The minted `ZO_ROOT_USER_PASSWORD` is now 12 characters instead of 48, drawn
  from an alphabet without the easily-misread `l I 1 O 0` — you read this one
  off a terminal and type it into a browser login. Existing projects keep the
  password already in their `.env`.

- `pymongo` is now a core dependency instead of the `archiver-mongodb` extra.
  The `control-assistant` preset deploys a MongoDB archive, so a plain `pip
  install osprey-framework` has to be able to run it. The extra is gone —
  drop it from any install command, since pip only warns about an unknown
  extra rather than failing. The preset no longer lists `pymongo` under
  `dependencies:`, which moves its profile hash: rebuilt projects will report
  staleness once, then match.
- The browser-facing bluesky sidecar is now the `bluesky-web` service (was
  `bluesky-panels`): it is named for its role — the web half of the bluesky
  stack, beside `bluesky-bridge` — rather than for the one panel it serves.
  Everything moves with it: the `bluesky_web:` build-profile block, the
  `services.bluesky_web.*` config keys, the compose service and image names,
  and the `OSPREY_BLUESKY_WEB_IMAGE` / `BLUESKY_WEB_URL` variables. Rebuild
  and redeploy to pick up the new names; nothing keeps the old spellings
  alive.

- Osprey now calls a bluesky plan a plan, not a scan. A plan is any bluesky
  generator — a scan is only one kind — so the word is gone from the operator
  panels, the live activity labels, the agent's tool descriptions and the docs.
  The `operating-bluesky-scans` skill is now `operating-bluesky-plans`: rebuild
  your project to pick up the new name, or the old skill file lingers in
  `.claude/skills/`. The "Run your first scan" how-to is now "Run your first
  plan" at a new URL.

- The bluesky bridge's plan-device env vars are now named for what a control
  room calls them: `BLUESKY_EPICS_MOTORS` is `BLUESKY_EPICS_SETPOINTS` and
  `BLUESKY_EPICS_DETECTORS` is `BLUESKY_EPICS_READBACKS`. Their values and format
  are unchanged. `osprey up` writes the new names; a project whose `.env` still
  holds the old ones will find them ignored, so remove those two lines (or run
  `osprey reset` then `osprey up`, which rewrites the block) to get plan devices
  back.

- The two shipped plans now name their read side `readbacks` instead of
  `detectors`, in the plan form, the queue summary, the approval prompt and the
  validation errors. A saved draft or a plan written against the old field name
  needs that one key renamed. Facility-authored plans are unaffected: a plan of
  your own may still call its read side `detectors`, `dets`, or `readables`.

- The `hello-world` preset is now the onboarding path. Its `profile.yml` leaves
  most keys unset on purpose, so each one arrives in your copy as a commented
  block you can turn on later, and its tutorial runs the agent in the web
  terminal (`osprey init` → `osprey build` → `osprey web`) instead of a terminal
  chat session.

- Lifecycle output now carries the CLI's theme. Phase openers anchor in the
  theme's primary color with a blank line before each phase, finished phases
  and durations dim so the open phase stays prominent, promoted facts and
  remedy arrows use the accent, and the closing "This deploy wrote" block
  styles its heading and file paths like the rest of the CLI. Long phases
  group their steps under quiet per-service headers (`archiver`, `ariel`,
  `personas`, `services`, …) so a busy `osprey up` reads as sections instead
  of one flat column. Piped output is unchanged apart from the new blank
  lines and group headers — color never reaches a pipe.
- Deploy output keeps one shape from start to finish. Facts a deploy used to
  print as paragraphs between phases (minted tokens, generated certificates, a
  renamed secrets file) are now one short line each in the step column, with
  the details collected into a "This deploy wrote" block after the closing
  summary. Warnings an operator must see print in the same indented `⚠` shape
  instead of as timestamped log blocks, and raw log warnings stay in the
  transcript (`-v`, file sinks) while a lifecycle command is drawing its
  progress. `osprey init --reset` now reports what the reset removed and kept
  as steps of its own phase; the full destruction plan remains what standalone
  `osprey reset` shows before asking for confirmation.

- What the CLI prints is now a progress report rather than a log. A verb prints
  the phase it is in, the steps under it, and a summary at the end; the
  timestamped `INFO` records that used to scroll past no longer reach the
  screen, though every one of them still reaches the sinks the deployment
  configures. `osprey -v <verb>` brings the whole transcript back. Warnings and
  failures read the same way in every verb now (a one-line summary, the cause
  under it, and what to do about it), and every one of them goes to stderr, so a
  script reading a command's output no longer has to filter trouble out of it.
  Under `--json`, stdout carries the JSON document and nothing else.

- ARIEL search modes are plain strings dispatched through the registry:
  `search(mode="keyword")` rather than `SearchMode.KEYWORD`. The `SearchMode`
  enum is gone, so code that imports it needs updating — the mode names
  themselves are unchanged. `osprey ariel search --mode` now takes its choices
  from the registry, so a facility that registers its own search module gets it
  as a mode without a framework change.

- Lifecycle commands no longer go quiet while they work. `osprey init`,
  `build`, `up`, `restart`, `down` and `reset` keep a spinner and a running
  elapsed under the phase they are in, and while images are building there is
  a row per service naming the step it is on, what that step is doing right
  now, and how long it has been at it; a service drops off the list as its
  image lands. Phases that have finished keep the lines they always printed.
  Where there is no terminal to repaint — piped to a file, or a CI log — each
  service still building appends a line roughly every thirty seconds instead,
  so a build that takes a quarter of an hour is never silent either way. No
  percent bars, no estimated finish times, and nothing is kept about how long
  earlier builds took.

- The agent's panel tools now say which of the two things they do. A panel can be
  on the launcher rail (reachable in one click) and it can be on screen; the old
  `show_panel` moved rail membership despite its name, `hide_panel` moved both,
  and `switch_panel` was the only verb that put anything on screen. They are now
  `add_panel_to_rail` / `remove_panel_from_rail` and `open_panel` /
  `close_panel`, each pair reversible by its partner.

- The agent's workspace tools use one word per thing. Stored records are
  `artifact_*` throughout (the parallel `data_*` family is gone, and `data_delete`
  with it), the record id is always `artifact_id`, and `artifact_delete_all` now
  requires the category to delete — it previously defaulted to deleting
  everything. The channel-finder tools are `ask_channels` (natural language) and
  `run_sql` (a query you wrote).

- Connector-level names no longer assume EPICS. The write failure code is
  `WRITE_FAILED` rather than `CAPUT_FAILED`, and `pv` is `channel` across the
  archiver, simulation and taxonomy surfaces. `ChannelMetadata` exposes
  `display_low`/`display_high`, and `channel_read` no longer advertises limits it
  discarded.

- `osprey validate` and `osprey profile validate` are one implementation, so the
  two commands can no longer disagree about whether a profile is valid.

- `IngestionScheduler.start` is now `run_forever`, which is what it does — it
  blocks until cancelled rather than starting a background task.

- Onboarding output is now written for the people who run accelerators rather
  than for the people who wrote OSPREY. `osprey init` prints the five entries
  you edit instead of a forty-line tour; the generated `profile.yml`,
  `README.md` and `.env.example` explain what each setting does and what
  changes if you alter it, rather than why it is designed that way; and
  `.env.example` now puts your provider's API key first and comments out the
  rest. The advice about setting up a CI pipeline moved from `init`'s output to
  the generated README, where it is relevant. No settings or defaults changed.

- Generated deployment repos, bundled skills, agent instructions and the
  documentation now describe the system as it is, rather than as a set of
  differences from an earlier arrangement. Guidance for moving an existing
  deployment onto the profile format is unchanged.

- Web terminal tile headers were redesigned: the bar now spans its whole tile
  (close always at the tile's right edge), sits on one seated surface whose
  hairline turns accent on the active tile, and renders one unified 24px
  control language with SVG icons. Panel names lead the bar with contributed
  text as a subtitle; on narrow tiles a contributed search collapses to its
  magnifier and remaining controls fold into a ⋯ menu instead of vanishing.
  The six-dot drag grip is gone — the bar itself remains the drag handle.

- Lifecycle commands now report what they are doing instead of scrolling their
  transcript past. `osprey init`, `build`, `up`, `restart` and `down` print one
  line per phase as they work; `init`, `build`, `down` and a detached
  `up -d`/`restart -d` finish with a summary card saying where the deployment
  stands and what to run next (an attached start ends inside the live log
  stream, so it gets none), and `osprey reset` ends with a one-line summary.
  The container build and compose output they used to stream is spooled to
  `var/logs/` — and a step that fails replays its own spool in full before the
  error, so the reason is still on the screen. `osprey -v` streams everything
  to the terminal as before.

- The BLUESKY panel's Results view leads with the run's figure. The raw data
  table sits below it behind a disclosure that names the run's row count and
  ships closed — these tables run to thousands of rows. What the table shows is
  a bounded preview, labelled with how much of the run it is withholding.
  **Export CSV**, on the same row, writes the whole run to a file: the browser's
  save dialog where there is one, the Downloads folder otherwise, and the note
  afterwards says which happened and how many rows landed. An export the bridge
  can only partly serve reports both counts rather than passing itself off as
  complete.

- `osprey users env-production` is now `osprey users env`, and the file it
  writes is `.env.users` instead of `.env.production` — the old name read as an
  environment name in a tool that has no environments. `osprey up` does the
  rename for you on the next deploy. A stack you stop before deploying again
  still carries the old name, and `osprey down` fails until it is renamed; the
  deploy guide gives the one-line fix.

- The control-assistant preset's two-user roster now ships alice as the
  write-capable operator and bob as the read-only viewer. The tiers differ
  visibly, not just in enforcement: the write-armed terminal keeps the full
  expert workspace with the EVENTS and BLUESKY panels, the read-only one gets
  a chat-first simple layout without them, both default to the light theme,
  and each browser tab is titled after its role.

- Dependency floors raised — `accelerator-toolbox`, `aiohttp`, `authlib`,
  `certifi`, `google-auth`, `openai`, `plotly`, `ruff`, `testcontainers`;
  `uv.lock` regenerated to match. `openai` is now capped below 3.x: the 3.0
  major is a client rewrite, so adopting it should be a deliberate change
  rather than something a lock refresh picks up on its own.

- Controls no longer slide out from under the pointer when a neighbouring
  control appears, disappears, or changes its label. Fixed across the web
  interfaces: the bluesky panel's Plans / Queue / Results switcher (the plan
  filter now opens to its left), the operator chat's Send button (Stop now
  opens inboard of it, so Send never lands where Stop was), the ARIEL entry
  pager (Previous and Next stay put and grey out at the ends instead of
  vanishing), the channel finder's feedback toolbar (Clear All now opens
  left of Add/Export instead of shoving them), and the confirm step on the
  bluesky emergency abort and the lattice dashboard's Baseline button, whose
  armed labels no longer widen the button and shove the control beside it.

### Added

- A third shipped plan, `orbit_bump_sweep`, drives a closed local orbit
  bump. The bump is stated in orbit space — the BPMs the beam should move at
  and by how much, plus the BPMs it must not move at all — and the plan solves
  for the kicks of the three or four correctors you name, so there is no
  lattice model to supply. It records a reference orbit and per-BPM noise,
  probes each corrector's response, then walks the amplitude up and back down,
  trimming each step inside the tolerance band before moving on. The last step
  commands the correctors back to their recorded working points and verifies,
  rather than trims, that the machine came back. A step that will not come
  inside tolerance stops the sweep unless `best_effort` is set. An optional
  beam-current guard re-reads the current before every write batch — each
  probe, each step, each trim pass — and stops the run when it falls below a
  minimum you set. Every corrector is returned to its pre-scan value on any
  exit. The run brings its own figure: the orbit shift across the BPMs at each
  step, the residual against the tolerance band, the corrector offsets, and
  the response of any extra monitor channels the run was asked to record.

- The `control-assistant` preset now stands up a third web terminal beside
  Alice and Bob: a standalone ARIEL logbook assistant, on its own card at
  `/u/ariel/`. It shares the deployment's Postgres and logbook, and runs no
  control-system tools at all — no channel access, no Python sandbox, no scan
  queue. Existing deployments are unaffected until they adopt the new preset.

- A persona can name the landing-page section its terminals appear under, with
  `landing_group` in the `modules.web_terminals.personas` catalog. The roster
  then splits: people stay in the default section, and each declared group gets
  its own below, drawn as a panel — which is how the landing page shows a
  standalone service as something other than another login. The `users` landing
  group also takes a `label` now, so both halves can be named. Nothing else
  about a terminal changes; a deployment that sets neither renders as before.

- A `qmd` search sidecar indexes the deployment's markdown corpora — the
  facility-knowledge bundle, and a markdown mirror of the ARIEL logbook — and
  answers hybrid keyword-plus-semantic queries. It is self-contained: its
  language models are baked into the image (built locally on the first
  `osprey up`, about 2.1 GB), so it needs no Ollama on the host. The
  `control-assistant` and `ariel-standalone` templates deploy it by default;
  comment out `services.qmd` and its `deployed_services` entry to opt out. The
  endpoint carries no authentication and publishes on the project-wide
  `deployment.bind_address`, which defaults to loopback and should stay there.
  Budget about 1.25 GB of disk per 135,000 logbook entries, and expect the
  first index build to take around 40 minutes at that size.

- Facility-knowledge search is ranked when that sidecar is configured. The
  KNOWLEDGE panel and the facility-knowledge `search` tool return hits in
  relevance order with a `score`, so a question phrased in an operator's own
  words can find a document that never uses those words. Without a sidecar both
  fall back to substring matching and `score` is `null`. `rerank` under
  `facility_knowledge.search` is off by default: it costs roughly four times
  the query latency, and these surfaces are interactive.

- ARIEL gains a `hybrid` search mode and a matching `hybrid_search` tool for
  the OSPREY agent, answering over a markdown mirror of the logbook written by
  the new `qmd_export` enhancement module. The templates enable both by
  default, and both are needed — the mirror with no search mode is never
  queried, and the search mode with no mirror has nothing to read. Configure
  them under `ariel.search_modules.hybrid` and
  `ariel.enhancement_modules.qmd_export`; the search knobs must sit under
  `settings:`, as keys written beside `enabled` are ignored. `rerank` is on
  here, where ranking quality is worth the latency. Entries created through
  the ARIEL web interface or the agent's `entry_create` tool are mirrored
  inline at creation time, so they become hybrid-searchable without waiting
  for the next enhancement run.

  Filtering in this mode is best-effort: results are ranked first and the date,
  author and source filters applied afterwards, so a selective filter can
  return fewer entries than asked for even when more exist. Use `keyword_search`
  or `sql_query` when a filter has to be exhaustive.

- `osprey ariel qmd-resync` re-exports logbook entries the markdown mirror
  never saw — those written by paths that bypass the enhancement modules and
  the inline mirror write. `ingest` and `watch` run this pass themselves, so a
  routine deployment never needs it by hand. `--rebuild` clears the mirror and
  re-exports everything, which is what to reach for after `osprey ariel
  purge`.

- `close_panel` takes a panel's tile off the operator's screen and leaves it on
  the rail, so it is one click from coming back. There was previously no way to
  clear a single panel without also making it unlaunchable.

- A red CI lane now leaves evidence behind. Every Docker lane captures its
  container logs, exit codes and `OOMKilled` flags — plus runner disk and
  memory state — before teardown removes the containers, and uploads them as a
  `ci-diag-<lane>` artifact. The unit-test lane records, per parallel worker
  and flushed as it goes, which test was in flight, alongside a stack snapshot
  of every thread taken every five minutes — including after the last test, so
  a hang during shutdown is visible too. A lane that is killed rather than failing
  (job timeout, runner stall) now names the test each worker stopped on in the
  run summary, instead of ending as a silent `cancelled`. Lanes that declare a
  time budget now cap their test step below it, so a hang fails that step and
  the capture still runs, rather than the whole job being cancelled mid-teardown.

- The OSPREY agent can start a Bluesky queue on a single approval: approving
  its `queue_start` call in chat arms and runs the queue, with no separate
  confirmation step elsewhere. The token that arms a start is granted only to a
  persona configured for control-system writes that also runs the bluesky MCP
  server, so a read-only persona still cannot start anything. The BLUESKY
  panel's own **Start queue** control is unchanged.

  A deploy refuses to grant that token to a persona that is also permitted to
  run a shell: the approval gates the `queue_start` tool, not a shell, so such
  an agent could read the token out of its own environment and arm a queue with
  no approval at all. Restore `Bash` to that persona's `permissions.deny` and
  rebuild its image — or republish and re-pull it, if you deploy from a
  registry — or move the persona off the bluesky server or off writes.

- Environment now resolves from a two-tier `.env` chain. `.env.shared` carries
  the values that are the same on every host and is tracked in git; the `.env`
  beside it carries the secrets and any per-host override, and wins on any key
  both set. `osprey init` writes both. Previously a single `.env` had to hold
  both, so a shared default could not be committed without committing the file
  that held the provider keys.

- A `network:` key in the build profile attaches the dispatch pair and the
  bridges to the host network instead of the compose bridge, for facilities
  whose control system answers only on the host. `osprey build` re-reads what
  it rendered and refuses a shape that cannot start.
- `podman-compose` now works as a container provider alongside Docker Compose
  v2. OSPREY detects which one is present and shapes the deploy to match; the
  two differ in how they resolve relative paths and how they order `--env-file`
  precedence.
- An archiver read that comes back empty now says why: the response carries a
  coverage verdict — the window predates or postdates the archive, the channel
  was never recorded, or the window holds a genuine gap — with the archive's
  real bounds, so an empty answer is never a silent one.

- A virtual accelerator can now be deployed with a real archive behind it: a
  MongoDB store plus an archiver-recorder service that records the machine's
  channels as they move. Scenario history is seeded into the store when the
  stack comes up, so questions about what a channel did earlier are answered
  from recorded samples rather than synthesized at read time.
- Simulation scenarios are generated against absolute timestamps, so a
  scenario's history lands at the wall-clock times it describes instead of
  being anchored to when it was run.
- The OSPREY agent can see what is actually on screen. `list_panels` now
  reports the open service tiles in left-to-right order plus how long ago that
  arrangement last changed, and tells "nothing open" apart from "no browser
  reporting" — so the agent can work from your view instead of guessing at it.
  It is also told what changed in the workspace between turns, and told nothing
  when nothing changed.
- `arrange_workspace` sets a whole layout in one call: exactly these tiles, in
  this order, optionally focusing one — or a named layout from the deployment's
  config, the same arrangement the **Layouts** menu applies. Clicking
  **Layouts** yourself behaves as it did before.
- Docked panels now show **one** header bar instead of two. A panel embedded
  in the web terminal contributes its real toolbar controls (view switcher,
  action buttons, live text) into the tile's header bar over a new
  postMessage contract; its own top bar disappears. ARIEL, Channel Finder,
  the lattice dashboard, and System Health were curated accordingly — the
  lattice summary stats moved into the panel body, Channel Finder's pipeline
  switcher and corpus stats into a bottom strip, and the lattice Baseline
  button now asks for confirmation before overwriting.
- The rest of the docked panels moved their toolbars into that one header bar
  too. **WORKSPACE** contributes its filter, its Types/Activity switch and its
  ⋯ menu; **KNOWLEDGE** its search; **EVENTS** its Activity/Triggers tabs. A
  panel's search box now renders with the same magnifier as the terminal's own
  search, so the two read alike. In Simple view the search stays in the panel
  body, where that view puts it front and centre.
- The **PLAN** and **BLUESKY** tabs are now one **BLUESKY** panel with three
  views — Plans, Queue, Results. The queue's state and its two halts (**Stop
  after current item**, **Abort running plan**) stay on screen across all
  three, and picking a run in Queue opens it under Results. Drop `plan` from
  your profile's `web_panels` and remove any `web.panels.plan.*` override.
- `osprey -v` (`--verbose`) shows debug output, including every container
  command a deploy runs. Normal runs no longer echo those commands, so a
  deploy reads as a report — ending in the endpoint summary — rather than a
  transcript.
- Bluesky scan agents can discover the worker's device namespace: a
  `list_devices` MCP tool and a `GET /devices` bridge endpoint. Substrate
  devices are named by their control-system channel address, and `queue_add`
  now checks a plan's device names against the worker's list at add time,
  refusing unknown names with a clear error instead of failing later in the
  worker.
- CI runs the whole Bluesky scan-stack e2e family: a new agent-driven scan
  lane (ORM and grid scans executed end-to-end and graded by a structural
  floor plus an LLM judge), a queue-stack lane, and the grid-scan roundtrip
  adopted into the ORM lane — all wired into the merge gate.
- Profiles carry artifacts into a build through **convention directories** —
  `rules/`, `skills/`, `agents/`, `commands/`, `output-styles/`, `hooks/`,
  `web-terminal-context/`, `mcp_servers/`, `services/`, and `project/` for
  anything without a home. The directory name is the declaration; there is
  nothing to list in `profile.yml`. A build warns about an unrecognized
  top-level entry, so a misspelled `rule/` no longer fails silently. `hooks/`
  is new — it installs a script into the project's `.claude/hooks/`.
- A profile can wire its own hooks into `.claude/settings.json` with
  `config: claude_code.hooks.<Event>`, naming a script the profile ships plus
  an optional `matcher` and `timeout`. Wiring is **additive**: it cannot
  remove, alter, or displace anything the generated settings already wire, so
  a declared hook is one more check on top of the framework's, never a
  substitute. A declaration is refused at build time when it names a hook the
  profile does not ship, a built-in whose wiring the framework owns, or a path
  outside `hooks/`. A persona unwires an event with
  `claude_code.hooks.<Event>: null` — an empty list merges additively and
  leaves the hook wired.
- `exclude:` now distinguishes a bare name (stop selecting the built-in) from a
  qualified `<directory>/<name>` (drop the profile's own file), so a persona
  can hand a shadowed artifact back to the framework. A bare name used where
  the profile also ships a file for it is warned about, with the qualified
  spelling that would take effect.
- `osprey init --force` re-materializes an existing repo's source zone from the
  preset — `profile.yml`, `data/`, `personas/`, `triggers.yml`,
  `web-terminal-context/`, `.env.example` — losing any edit to them. It never
  touches `.env`, `.git`, `var/`, `build/`, `.gitignore`, `README.md`,
  `ci-extra.yml`, `.gitlab-ci.yml` or `scripts/verify.sh`.
- The emitted `profile.yml` header now opens with a map of the repo's four
  zones — source, secrets, build output, durable state — and the
  edit → `osprey build` → `osprey up` loop that connects them.
- A profile can carry a `deploy:` block: CI platform, deploy host, and the
  container registry when the host pulls its images. Credentials are named
  there, never written there.
- `osprey scaffold ci` emits the repo's CI pipeline and post-deploy health
  check from that block. Re-running is safe — a file whose content already
  matches is left untouched, and a file the scaffolder did not write is
  reported rather than overwritten unless `--force` is given. `ci-extra.yml`
  is never touched; the pipeline includes it.
- `osprey users env-production` renders `.env.production`, the env file each
  per-user web-terminal container runs with, from the deploy config and one
  secrets file. `--output` writes it at mode `0600` instead of to stdout.
- `archiver_read` gained `bin_size=0` for full resolution — every real
  archived sample in the requested range, with no per-bin decimation. Only
  valid with `processing="raw"` (an aggregate has no bin to aggregate
  over); a non-raw `processing` with `bin_size=0`, or a negative
  `bin_size`, is a validation error.
- Bluesky scans now run in a queue server instead of inside the bridge
  process. Execution is two steps — add the composed plan to the queue, then
  start the queue — so a queue can be assembled and reviewed before anything
  moves. Adding to an idle queue needs no launch token; starting requires one,
  and with `control_system.writes_enabled` off the agent's `queue_add` and
  `queue_start` are denied outright. A queue survives a bridge restart, and a
  deployment that cannot execute plans refuses to hold items rather than
  accepting work it could never run. New guide: How-To → Run Scans Through
  the Queue.
- Emergency abort for a scan already moving hardware: **Abort running plan** in
  the BLUESKY panel, `stop_run` for the agent, `POST /queue/abort` for
  integrations. It is ungated on every surface — no launch token, no writes
  switch — so a halt stays available on a stack with writes disabled. It is
  distinct from stopping the queue, which lets the running scan finish first.
  An abort leaves hardware wherever the scan had moved it, and says plainly
  when it did not manage to stop the plan.
- An interrupted plan — aborted, halted, or failed — stays in the queue, reports
  as `stopped` (`error` for a failure) rather than as pending work, and blocks
  the next queue start until it is removed. Removing it is what unblocks the
  queue; to run it again, remove it first and then stage and add it afresh.
- `connector:` is a new top-level build-profile key and the short spelling of
  `config: {control_system.type: ...}`, so a connector can be chosen from the
  command line with `--set connector=epics`. Giving both spellings on one
  command line is an error rather than a silent last-one-wins.
- You can see what the agent did to your workspace. A tile the agent focuses
  or rearranges glows briefly, and its rail tab flashes with it, so a layout
  that changes under you is never unattributed — your own clicks stay quiet.
  An activity strip names each action in plain words ("agent opened
  WORKSPACE"), and its history popover holds the recent ones for when you
  looked away. Panels that changed while you were elsewhere keep a badge
  across a reload until you visit them. Every agent tool that changes
  something — queue and plan authoring, logbook entries, Phoebus drives,
  python execution, lattice and window management — reports itself there.

### Changed

- **A deployment is a git repo, and every lifecycle verb is top-level.**
  `osprey init` creates the repo, `osprey set` edits its `profile.yml`,
  `osprey validate` checks that profile without building, and `osprey build`
  renders `build/` from it. `osprey up`, `down`, `restart`, `status`, `logs`
  and `reset` operate the deployment; `osprey chat` talks to it; `osprey users`
  manages the web-terminal roster; `osprey scaffold ci` emits the CI pipeline.
  Each verb finds the repo by walking up from the working directory, so none of
  them is given a project or config path — `--repo` overrides the starting
  point. Running `osprey` with no arguments prints the command list.
- Dev mode is a property of the build: `osprey build --dev` bakes the local
  osprey checkout into the service images, and `osprey up --dev` starts that
  render — refusing a render built without `--dev` instead of silently starting
  the published release (`osprey up --build --dev` chains both). A plain
  `osprey up` of a dev build warns that the images carry the local checkout.
- Deployed agents can no longer reconfigure their own harness: the Claude Code
  CLI's bundled harness-configuration skills (`update-config`,
  `keybindings-help`, `fewer-permission-prompts`) are switched off in every
  rendered project, and the `setup-mode` skill (which can patch config.yml)
  left the operator preset's default roster — it stays in the artifact catalog
  for admin profiles to opt into. Rebuilt control-assistant projects will
  report preset staleness once; that is the intended signal.
- **Log out** moved into the web terminal's display menu, alongside
  **Settings** — the two now sit side by side under a line naming the signed-in
  user. The separate user chip in the header is gone, leaving search and the
  display menu there. Single-user terminals are unchanged apart from
  **System Settings** being relabelled **Settings**.
- Pairing a virtual accelerator with the mock archiver is refused — at build,
  at deploy, and at MCP server startup — because the VA moves channels for
  modelled reasons while the mock archiver invents history at read time, and
  the pair reports a past that never happened. The mock archiver remains the
  default for mock control systems; a project that pairs the two must move to
  the MongoDB archiver or to a mock control system.
- Asking the agent to open a panel no longer replaces what you were looking at.
  `switch_panel` opens the tile *beside* your current one — focusing it instead
  if it is already open — so no tile you had open is evicted. The Simple web
  UI's single workspace slot is unchanged.
- Raised minimum versions for `psycopg`, `psycopg-pool`, `uvicorn`, `rich`,
  `fastapi`, `charset-normalizer`, `unique-namer`, and `pymongo`.
- Web-terminal archives written by `osprey users remove --archive` now land in
  `<repo>/var/web_terminal_archives`, not `<project>/web_terminal_archives`.
  Archives written before this release are left where they are; move them
  yourself if you want them all in one place.
- `osprey scaffold claim` moves an artifact out of the build zone and into the
  matching convention directory of the repo's source zone, instead of marking it
  user-owned where it sits. The next build copies it back and registers it, so
  ownership is derived from what the build actually copied — there is no list
  to maintain, and an artifact a persona excludes is not owned, letting the
  framework's version render in its place.
- The profile's `.env` is where a project's secrets live. `osprey build`
  derives the project's `.env` from it and from nothing else, and a later build
  never re-reads your shell. A shell export reaches a profile only once, at
  materialization, and only for providers the profile actually references —
  keys exported for other providers are named in the summary rather than copied
  in. `osprey up` writes the credentials it mints back into the profile's
  `.env`, append-only, so a rebuild comes up on the same secrets
  instead of minting a second set the running containers do not trust.
- The web terminal's System Settings drawer explains itself. Each tab opens
  with a standing one-line subtitle, and the category help tooltips now
  describe how each kind of file is *loaded* — when it enters the session,
  what runs it, whether it advises or enforces — instead of summarizing what
  the shipped files happen to say, which went stale as soon as an operator
  edited one. Ownership prompts likewise state what taking or releasing
  ownership actually does.
- Each settings gallery now opens on artifacts rather than controls. Search and
  the category chips moved behind a `Filter` disclosure on a single muted
  summary line, and every category except the pinned ones starts collapsed.
  An active filter stays named on that line, with a one-click clear, so a
  narrowed list can never be mistaken for a short one.
- The Behavior tab labels `CLAUDE.md` "project instructions" rather than
  "system prompt": it is delivered as a message after the system prompt, while
  the output style is what actually modifies it.
- Bridge conversation history keeps more context. Replay is now bounded by the
  character budget (raised to 100k chars) rather than by turn count, which
  becomes a runaway backstop (100 turns), and a turn stays eligible for replay
  for 180 days instead of 90. Long-lived direct-message threads no longer drop
  older turns while sitting far under their size budget.
- `osprey init` now writes persona profiles as small deltas under `personas/`
  instead of full standalone copies. A file there merges over the repo's
  `profile.yml` implicitly, so edit the host profile once and every persona
  inherits the change, while each persona file keeps its own capability posture
  (e.g. `control_system.writes_enabled: false`) pinned explicitly.
  Model-selection choices baked at materialization time — and `tier` — now
  reach personas through inheritance.
- The shipped web-terminal rosters spell out `name`/`index`/`persona` on every
  user entry instead of bare-string shorthand for the first user. Behavior is
  unchanged; already-deployed projects will see a one-time profile-staleness
  advisory from the preset content change.
- The EPICS connector now rejects a `bin_size` the appliance cannot express
  rather than quietly serving a different resolution. Sub-second and
  non-whole-second widths used to be floored to the nearest second (500 ms
  *and* 1500 ms both became 1 s).
- `archiver_read`'s `access_details` payload no longer repeats the same access
  rule once per channel: 3548 → 1075 bytes for a twenty-channel read, and now
  flat in channel count rather than growing with it.
- **Breaking change: `ArchiverConnector.get_data` now returns long-format
  data instead of a shared-index wide frame.** Every archiver connector
  correctness bug fixed below traced back to forcing every requested
  channel onto one shared index, which required forward-filling gaps and
  resampling data into existence just to keep a rectangular shape.
  - **Before:** a wide `pandas.DataFrame` indexed by timestamp, one column
    per channel, reindexed/forward-filled onto a shared grid so every
    column had a value at every row.
  - **After:** a long `pandas.DataFrame` with exactly three columns —
    `timestamp` (`datetime64[ns, UTC]`), `channel` (`str`), and `value` —
    sorted by `channel` then `timestamp`. Each channel contributes only its
    own real samples (or, under a non-`raw` `processing` mode, only its own
    real per-bin aggregates); nothing is forward-filled, reindexed onto a
    shared grid, or otherwise manufactured, and a channel with no data in
    range contributes no rows. An empty result is an empty frame with the
    same three columns.
  - `value` is not dtype-constrained: `float64` when every requested
    channel's samples are numeric, or pandas' natural mixed dtype once any
    channel is non-numeric — enum/status channels (EPICS `mbbi` / DOOCS
    `DBR_STRING`: machine mode, interlock state, RF state) are archived as
    strings and round-trip as strings, never coerced.
  - `raw` processing now decimates each bin to its last **real** sample,
    keeping that sample's own true timestamp rather than a relabeled bin
    edge — matching EPICS's long-standing `lastSample_N` semantics on every
    backend.
  - The `archiver_read` artifact payload changed from a split-orient wide
    frame to `{"query": ..., "series": {"<channel>": {"timestamps": [...],
    "values": [...]}}}`. Artifacts already saved in the old layout still
    render — `extract_channel_series` normalizes all three historical
    layouts.
  - **Any out-of-tree `ArchiverConnector` subclass must be updated** to
    return the new long-format shape; downstream code no longer accepts
    the old wide, shared-index format.
- `ArchiverConnector.get_data` gained a trailing `processing: str = "raw"`
  keyword (one of `raw`, `mean`, `min`, `max`, `median`, `std`, `count`).
  It's appended last with a default, so existing positional callers are
  unaffected; an out-of-tree connector that overrides `get_data` must
  accept the new keyword (even just to ignore it) to remain
  call-compatible.
- The chat bridge how-to is now a section, `how-to/chat-bridges/`, with an
  overview page and a page per chat system. Adds a guide to connecting a
  service that does not ship with Osprey, such as Slack or email. The old
  `how-to/deploy-chat-bridge` page is gone; its content moved into the new
  Nextcloud Talk and Google Chat pages.
- Ruff moved to 0.16, pinned to one minor in the `dev` extra so the
  pre-commit hook and CI agree on formatting. The formatter skips Markdown,
  leaving documentation snippets as written.
- The `control-assistant` preset now defaults to the `virtual_accelerator`
  connector instead of `mock`, so its scans drive the soft-IOC the same stack
  already deploys and run end to end out of the box. `mock` remains the
  fallback for environments with no containers to depend on, where scans are
  browse-only — plans compose and validate, but the queue will not hold them.
  Switch with `osprey set connector=mock`.
- The Bluesky **RESULTS** panel is now **BLUESKY**, and holds the scan queue as
  well as the selected run's results. Move your own `web.panels.results.*`
  entries to `web.panels.bluesky.*`. The preset rename changes its resolved
  content, so an already-deployed project reports staleness on its next
  `osprey up`. That is the correct signal rather than noise — the tab a
  user sees is renamed — and rebuilding picks it up.
- Unknown keys in a build profile's `bluesky:` block now fail the build, naming
  the valid keys (`excluded_plans`, `plan_dir`, `port`, `tiled_enabled`,
  `tiled_port`). They used to be dropped in silence, so a typo — or a key a
  later release removed — took effect as "unset" with no warning anywhere.

### Removed

- The ALS GTB example corpus (`als_gtb.ttl`). The generated demo-machine corpus
  is now the only facility graph OSPREY ships; both presets seed it, the
  `ariel_standalone` preset carries its own copy, and each curated Cypher
  example ships one parameter set whose values exist in the demo machine.

- Registry and ARIEL exports that nothing called, including a second connector
  registry that shadowed the real one.

- The `osprey deploy` and `osprey claude` command groups, the `osprey config`
  subcommands, `osprey profile new` and `profile try`, several `osprey build`
  options, and the interactive menu that bare `osprey` used to launch. What to
  run instead:

  | Removed | Use instead |
  | --- | --- |
  | `osprey deploy up` / `down` / `restart` / `status` / `build` | `osprey up` / `down` / `restart` / `status` / `build` |
  | `osprey deploy clean` / `rebuild` / `nuke` | `osprey reset`, or `osprey up --build` to re-render and start |
  | `osprey deploy decommission` / `prune` / `seed` / `passwd` / `render-env-production` | `osprey users remove` / `prune` / `seed` / `passwd` / `env-production` |
  | `osprey deploy scaffold` | `osprey scaffold ci` |
  | `osprey claude regen` | `osprey build` |
  | `osprey claude status` / `chat` | `osprey status` / `osprey chat` |
  | `osprey config show` / `export` | `osprey config --rendered` / `--defaults` |
  | `osprey config set-control-system TYPE` | `osprey set connector=TYPE` |
  | `osprey config set-epics-gateway --facility NAME` | `osprey set epics_gateway=NAME` |
  | `osprey build --tier N` / `--set K=V` | `osprey set tier=N` / `osprey set K=V` |
  | `osprey build PROJECT --preset P` | `osprey init PROJECT --preset P`, then `osprey build` |
  | `osprey profile new DIR --preset P` | `osprey init DIR --preset P` |
  | `osprey profile try` | `osprey init --preset P --up` |

  `osprey profile presets` and `osprey profile validate` are unchanged.
- The `osprey-build-deploy` skill. What it used to scaffold by hand — the CI
  pipeline, the deployment files, the post-deploy health check — is now
  `osprey scaffold ci` and the deploy verbs themselves.
- `facility-config.yml`. The `modules.web_terminals` stanza lives in the repo's
  built `config.yml`, emitted from the profile's `config:` block, and the
  deployment files come from the profile's `deploy:` block. Passing `--config`
  to `osprey scaffold web-terminals` is now an error naming both replacements;
  use `--repo`, or run from inside the repo.
- The `overlay:` profile key and the `overlays/` seed directory. Put a file in
  the convention directory that matches what it is; there is nothing left to
  declare.
- The built project's `.env.template`. `.env.example` replaces it and lists
  every variable the agent reads, not just the ones the profile declared, and
  the profile ships an identical copy so the two can never disagree.
- `osprey build` no longer harvests provider API keys out of the environment it
  happened to run in. A key now reaches a project only by way of the profile's
  `.env`, so what a build produces does not depend on the shell that ran it.
  Exporting a key still works for a host-local run and still seeds a profile at
  materialization; it no longer leaks into a built project unrecorded.
- Removed the `multi-user-demo`, `multi-user-demo-readonly`, and
  `multi-user-demo-readwrite` presets. The `control-assistant` preset ships
  the same two-persona multi-user web tier, so the demo family was a lighter
  clone of it; build from `--preset control-assistant` instead. The multi-user
  walkthrough now lives at How-To → Multi-User Support.
- Removed the DOOCS connector's `max_points` history-decimation path. It built
  a fixed `np.linspace` grid and forward-filled onto it with a zero-order hold,
  which the "nothing is manufactured" contract forbids, and no production
  caller could reach it — the connector always passed `None`.

- Removed direct execution of Bluesky plans inside the bridge process. `POST
  /runs`, `POST /runs/{id}/launch`, `POST /draft/run` and `POST
  /runs/{id}/stop` now answer `410 Gone` with a refusal naming their queue
  replacement. The queue is the only path to hardware: enqueue with `POST
  /queue/items`, start with `POST /queue/start`, and halt with `POST
  /queue/stop` or `POST /queue/abort`.

- Removed the `bluesky.demo_runner` build-profile knob and the in-bridge runner
  it switched on. A profile that still sets it now fails the build with the
  list of valid `bluesky:` keys, rather than dropping the key silently.

### Fixed

- A web-terminal persona that drops a skill by name now really builds without
  it. Persona builds decided which skill files to write from the host
  deployment's artifact list rather than the persona's own, so a persona could
  add a skill but never remove one, and the terminal shipped skills whose tool
  servers it does not run. Hooks, rules and subagents were already correct.

- The `orm` scan plan now kicks each corrector either side of where it found it
  and puts it back there. It previously drove absolute currents either side of
  zero and ended every corrector's sweep at 0 A, which is only correct on a
  machine whose correctors idle at zero — on a ring holding a corrected orbit it
  would have measured about a point the machine was not at and then dropped the
  correction. `span_a` is now the size of the kick away from a corrector's
  working point, and its 10 A ceiling is gone: what a corrector will take is the
  deployment's own `channel_limits.json`, which is checked on every write. A
  corrector whose read-back is not a number is refused before anything is
  written to it.

- The source-tree sweep behind the lifecycle criteria no longer fails when a
  file disappears while it is reading. It walks the live tree, so a temporary
  file another test had staged under `src/` could be listed and then deleted
  before its turn came, failing a criterion that was never evaluated.

- Deploys with the ARIEL logbook store no longer break on podman-compose hosts:
  the store's own `up` was the one compose invocation still built in the docker
  shape, so it aborted the deploy (or ran against the wrong project directory
  with `.env.shared` dropped) in the middle of an otherwise podman-shaped start.

- `osprey users nuke` now completes on podman-compose hosts. Its container
  teardown is built like every other compose command — rendered files, provider
  shape, pinned project directory — instead of a bare `docker compose -p down`
  only Docker can parse.

- A `$` in an `.env.shared` value now stops a deploy on every compose provider,
  not only podman-compose. Docker Compose interpolates env-file values, so such
  a secret reached containers truncated or spliced with host values, silently.

- The host-port preflight and the `osprey up` closing summary now cover any
  facility service placed on the host network, read from its
  `services.<name>.port` key. A host-mode service without that key is named in
  a warning instead of silently escaping the check.

- `osprey health --project <repo>` run from another directory now resolves the
  target repo's whole env chain — `.env.shared` included — the way build, chat
  and compose do, and the long-lived health surfaces notice `.env.shared`
  edits instead of answering from a stale environment forever.

- The release pipeline again verifies that the framework wheel's dependencies
  resolve from PyPI before anything is published; the check had been quietly
  lost when the install-docs lane switched to local wheels for PR runs.

- A control-system write whose read-back could not be verified now fails instead
  of reporting success. `write_channel` logged `Wrote <channel>` whenever the
  write itself returned, so an operator could be told a setpoint had moved when
  nothing had confirmed it.

- Panels you switch off with `auto_launch: false` no longer appear as working
  tabs. Five of the six companion panels published their URL before checking the
  setting, so the panel was offered in the rail and its iframe returned a 502.
  The same five now also survive an empty `OSPREY_<PANEL>_PORT` in a compose
  file, which used to kill the launch outright and leave a dead tab.

- `osprey health` runs from a subdirectory of your project, like every other
  `osprey` verb, and its panel probe honours `OSPREY_<PANEL>_PORT` — so on a
  multi-user deployment it checks each user's own panel rather than reporting
  everyone's as down.

- The `vllm`, `deepseek`, `ollama` and `argo` providers now ship a base URL that
  can actually be reached.

- New projects reach Argo again. The scaffolded `config.yml` pointed at
  `https://argo-bridge.cels.anl.gov`, which no longer serves; it now uses
  `https://apps.inside.anl.gov/argoapi/v1` and reads `ARGO_BASE_URL`, so you can
  redirect a deployed project at another gateway without editing the file.

- The CLI reference described a `--project` option on `osprey web` that the
  command does not have. It takes `--repo`.

- Logbook watch results keep `entries_updated`, which was dropped between the
  service and the CLI, so an update reported as zero changes.

- A `host`, `port` or `auto_launch` key written at the wrong nesting depth is now
  refused with the correct path, instead of being silently ignored — which used
  to start a panel you had switched off, or bind a port you had not asked for.

- Per-user web terminals now receive ARIEL's database password. Without it, the
  ARIEL tab reported the database as unavailable and the agent's logbook tools
  failed, on a deployment whose database was running and healthy: the password
  `osprey up` mints never reached the containers, so both authenticated with the
  shipped default instead. Each user whose persona configures ARIEL is now given
  it, and a persona that configures none is not.
- A start now creates ARIEL's schema for a database it brought up itself, and
  writes the active scenarios' logbook entries when the logbook is empty. Until
  now nothing on the deploy path created the tables — that lived only behind
  `osprey ariel migrate`/`quickstart` — so a first start left a database with no
  tables behind a panel that could not say so usefully. An existing logbook is
  left untouched, and a database that cannot be reached warns and names the
  command that finishes the job rather than failing the deploy.
- A start that would generate store credentials a surviving data volume cannot
  accept now stops before it starts anything. Deleting a deployment directory
  leaves its volumes behind, so the next `osprey up` minted fresh passwords for
  postgresql, openobserve and mongodb — which each read their password only
  when initializing an empty volume, and go on rejecting the new one. The
  previous warning could only guess that this might be happening; the start now
  asks the container runtime, names every affected store at once, and says
  whether each one's original credential can still be recovered. It reports
  this before the image build instead of at a health probe minutes later —
  which matters, because starting the stack recreates the store containers, and
  a container holds the only copy of the credential its volume was created
  with. `osprey restart` runs the same check before it stops anything, for the
  same reason.
- New: `osprey up --reuse-stores` and `osprey restart --reuse-stores` adopt
  those volumes instead of discarding them, restoring each store's original
  credential to `.env`. They refuse if any affected volume can no longer be
  reopened, rather than starting a stack that is part-adopted and part-doomed.
- New: `osprey init --reset` starts over on a name that has been used: it
  destroys the containers, volumes and images left by a previous deployment of
  that name, and re-materializes the source zone as `--force` does when the
  repo directory still exists — so re-creating a deployment is one command,
  with no `rm -rf` first. Provider keys in `.env`, git history and the audit
  log survive, exactly as they do under `--force`. If anything of that name
  survives the sweep, `init` stops and says so instead of starting: `reset`
  removes only what carries the checkout's own label, so a deployment created
  before that label existed has to be cleared by hand once.
- Dev deploys now report each service image as it finishes building, instead
  of one summary line after the whole `compose build` — the longest step of a
  deploy, and previously silent for its entire duration.
- The EVENTS tab now works in a multi-user deployment. Its dashboard is
  bearer-gated, but per-user terminal containers never received the dispatcher's
  token, so the tab loaded and then reported "No triggers are registered." — for
  a dispatcher that had every trigger loaded. Users whose persona declares the
  EVENTS panel now get the token in their own container; personas without the
  panel, such as a read-only tier, deliberately still do not.
- The event dashboard no longer claims a dispatcher has no triggers when it was
  simply not authorised to read it. Both trigger views now say they were
  refused, and a panel opened inside the terminal is told its terminal has no
  token rather than to open the tab it is already in.
- `osprey health` now answers from either stance. It looks for the config where
  a build writes it (`build/config.yml`) and reads credentials from the repo's
  `.env`, so running it at the repo root no longer reports the config missing,
  and pointing it at the render no longer runs the provider canaries and the
  environment scan with no credentials loaded.
- Container detection now recognizes Podman's `/run/.containerenv`, not only
  Docker's `/.dockerenv`. Inside a Podman deployment the derived MCP health
  probes were aimed at host URLs.
- `osprey set` now says so when a key is not one the profile recognizes. The
  key is still written, but the profile schema is closed, so the next
  `osprey build` refuses the whole profile — which used to be the first hint,
  reported against `profile.yml` rather than against the command that made the
  edit. Keys addressing the rendered config (`config.…`) are unaffected.
- A dispatch worker whose `agent_data.base_dir` is an absolute path now mounts
  its workspace volume where the worker actually writes. The mount target was
  re-anchored under the project directory, so the volume landed on a path
  nothing used while the records went to the container's writable layer and
  were lost on every recreate.
- Several messages and rendered comments still named commands the redesign
  removed — among them the refusal `osprey up` raises when `.env.production`
  is missing, which pointed at `osprey deploy render-env-production` instead of
  `osprey users env-production`.
- A scan-stack deployment no longer intermittently refuses every scan with
  "not in the list of allowed plans". The bridge opens the Run Engine worker
  once at startup, and abandoned it whenever it won the boot race against the
  queue server — leaving the list of runnable plans empty until someone started
  the queue by hand. It now waits for the queue server to answer first.
- One operator's tab switches no longer rearrange every other window of the
  same workspace: a human panel focus is now mirrored to the server silently
  (the agent can still read where the operator is looking) instead of being
  broadcast back, whose delayed echo could evict tiles the operator had open —
  in the gesturing window and in every other one. Closing a tile no longer
  reports its side-effect focus change either.
- Web terminal panels no longer freeze permanently — rendering but ignoring
  every click — when a drag from the panel rail loses its end event (for
  example the dragged entry was removed mid-drag by the agent or another
  client). Drag cleanup now has document-level failsafes.
- The web terminal's panel event stream reconnects after a proxy or backend
  hiccup and re-syncs rail membership on every reconnect, so a browser that
  missed events while disconnected converges instead of silently drifting.
  Event-handling errors are now logged instead of swallowed.
- Workspace gallery: the "Draft created" confirmation no longer sticks as a
  permanent full-panel overlay after a successful logbook submit.
- Workspace gallery: deleting the artifact being viewed fullscreen (locally
  or agent-side) exits fullscreen instead of stranding a pane with no
  controls.
- The web terminal welcome screen wires its dismiss controls before any
  fallible boot step, and a terminal-library load failure degrades the
  terminal card instead of aborting the whole page boot.
- Lattice dashboard summary stats (energy, tunes, chromaticity) no longer
  freeze at load time — they recompute with the fast figures after a magnet
  change. Also removed dead panel chrome the audit surfaced: ARIEL's unwired
  "Connected" indicator and the System Health panel's no-op manual refresh
  and misleading fetch-time timestamp.
- The **EVENTS** panel drew two header bars when docked — its own, plus the
  tile's. It now hides its own, like every other panel.
- On Docker Desktop (macOS/Windows), `osprey up` now repairs a web stack that
  is fully healthy yet unreachable from the browser. Docker
  Desktop forwards a host-network port only if it watched the container open
  it, so a container that restarted while Docker Desktop itself was starting
  stays invisible from the host — and re-running `osprey up` could never fix
  it, because nothing in the container's definition changed. The post-deploy
  reachability probe now restarts the web stack once and re-checks before
  pointing at the host-networking setting.
- A deploy is quieter about things that were never wrong: no more
  orphan-container warnings for its own sibling stack (two compose
  invocations share one project by design), no more garbled progress
  rendering on long service names (docker runs use `--progress plain`), and
  no more platform-mismatch warning for the virtual accelerator on Apple
  Silicon — that image is amd64 by design (its Channel Access server has no
  arm64 wheels), and the compose service now declares it via
  `platform: linux/amd64`, overridable with `OSPREY_VA_PLATFORM`.
- Importing the control-system connectors no longer drags in pandas. The
  connector factory imported the archiver base eagerly for a type annotation,
  so anything that wanted only `osprey.connectors.control_system` had to
  install the archiver's dataframe stack to import it at all.
- A secret containing `$` no longer reaches a container truncated. Compose
  substitutes `$` sequences inside env-file values, so `secret$abc` arrived as
  `secret` and `P@$$w0rd` as `P@$w0rd` — while the file on disk still read
  correctly, leaving a login that refused for no visible reason. `osprey up`
  now refuses such a stack and names the offending variables (never their
  values). All three files a deploy reads secrets from are checked —
  `.env`, `.env.production` and `.env.auth` — including ones OSPREY did not
  write itself, so a CI-built `.env.production` and a hand-added OIDC client
  secret are covered. `osprey users passwd` checks before storing a new
  password.
- The OIDC section of the multi-user guide named `.env` as the file to put
  client credentials in. It is `.env.auth` — credentials placed as documented
  never reached the login service.
- Editing `.env.auth` by hand (the documented way to add OIDC client
  credentials) now takes effect on the next `osprey up`. On podman the
  login service previously kept running with the old file's contents —
  healthy-looking but rejecting every login — until it was recreated manually.
- Lint now refuses a roster `oidc_subject` containing `$`. The subject travels
  through the rendered compose file, where `$` sequences are rewritten, so
  that user could never log in and nothing said why.
- The settings drawer's `CLAUDE.md` section now has a help tooltip. Its help
  text was filed under a category name no gallery ever displays, so the button
  silently never rendered — on the one artifact that matters most.
- The settings Config tab's form view no longer hides most of `config.yml`. It
  listed a `python_execution` section that does not exist (the section is
  `execution`), so execution settings were reachable only through Raw YAML;
  `archiver`, `logbook`, and `facility_knowledge` are now editable there too.
- Enum/status channels no longer render a state the channel was never in. A
  gap in an enum channel (a `null` sample) was being turned into the literal
  category rung `"<channel>: null"` on the chart's shared category axis, so a
  disconnect drew as a real state. Gaps now break the line as they always did
  for numeric channels. The same axis also had its tick labels clipped by a
  fixed right margin and drew an off-theme gridline per rung; both fixed.
- `raw` and the aggregate processing modes now place bin boundaries on the same
  lattice. `raw` decimation anchored its bins to the Unix epoch while the
  aggregates anchored theirs to the start of the day, so for any `bin_size`
  that does not divide a day evenly (7 s, say) the two disagreed — an operator
  comparing `raw` against `mean` over one window got bins that did not line up.
  Whole-second widths that divide a day, which is every width the framework had
  been exercised with, are unaffected.
- The archiver freshness health probe stopped silently discarding sub-microsecond
  precision. It converted each sample's timestamp through `to_pydatetime()`,
  which emits `Discarding nonzero nanoseconds` on every EPICS probe run.
- The timeseries table's header came from a different HTTP request than its
  cells — the header from `format=chart`, the rows from `format=table`. For an
  artifact being written while it is viewed the two can disagree, showing values
  under the wrong PV name. `format=table` now returns the very column list its
  rows were built from, and the client uses it.
- The artifact gallery's info-bar totals are now computed server-side and
  reported under a new `summary` object on `format=chart`. The client had been
  summing each channel's own point count, a number that cannot be reconciled
  with the table's unioned row count — only the server sees both axes.
- The EPICS Archiver Appliance connector formatted query-window bounds
  with a literal UTC `Z` suffix without actually converting to UTC first,
  so at any facility whose `system.timezone` is not UTC every
  `archiver_read` window against EPICS landed hours away from the one an
  operator actually asked for — e.g. "the last hour" could silently pull
  data from several hours in the past or future, depending on the
  facility's UTC offset. EPICS now converts to UTC before the query
  reaches the wire. MongoDB and DOOCS did not share this regression
  through the actual `archiver_read` path — the tool always hands
  `get_data()` a timezone-aware datetime, and pymongo's BSON encoding and
  `datetime.timestamp()` are each already correct for an aware datetime
  regardless of its zone — but both connectors (and the mock/simulation
  archiver) now call the same `to_utc()` helper as EPICS, hardening the
  connector-level contract for a caller that bypasses the tool: a naive
  (timezone-less) datetime passed directly to `get_data()` is now read as
  facility-local rather than depending on the caller's own zone or
  assuming UTC, consistent with every other connector.
- `archiver_read`'s `processing` parameter (e.g. `mean`, `min`, `max`) was
  accepted and echoed back in the response, but never actually applied to
  the query — a request for a 60-second mean silently returned the same
  last-sample-per-60-second data as `processing="raw"`, with no error or
  warning. `processing` is now honored end-to-end: the EPICS connector
  pushes the aggregation to the Archiver Appliance server-side, and
  MongoDB/DOOCS/mock apply it client-side, so the values returned actually
  match the requested aggregation.
- The MongoDB archiver connector ANDed a `$exists` condition for every
  requested PV onto the same query, so a request spanning channels
  archived in separate documents — a common ingestion pattern — matched no
  documents at all and silently returned an empty frame with no error, as
  if none of the channels had ever been archived. It also ignored
  `precision_ms` entirely (returning every raw document at ingestion
  cadence regardless of the requested bin width) and treated `timeout=0`
  as "no timeout given," silently substituting the connector's default.
  The query now matches any document carrying at least one requested PV
  (`$or` instead of ANDed per-PV `$exists` checks), `precision_ms` is
  honored via per-channel resampling, and an explicit `timeout=0` is
  respected rather than overridden.
- `DOOCSArchiverConnector.check_availability` built its "everything
  unavailable" result when the connector was disconnected, but never
  returned it, so execution fell through into querying the live DOOCS ENS
  anyway — a disconnected connector could still report channels as
  available instead of cleanly reporting all of them unavailable. The
  disconnected guard now returns immediately.
- Plotting an archiver artifact that mixes channels with and without data in
  the requested window no longer fails. A channel with no samples produced an
  untyped (object-dtype) column, and Plotly Express refuses a wide frame
  "with columns of different type" — so `px.line(data)` over "beam current
  plus a channel that was down" raised instead of drawing the channel that
  did have data. Empty channels now carry a `float64` column.
- `data_read` on an over-cap archiver artifact now previews the long-format
  payload it actually receives. It recognized only the old wide/split-orient
  shape, so a `{query, series}` file — the common reason for exceeding the
  100 KB inline cap — fell through to a bare `json_object` preview listing
  `["query", "series"]` and nothing else. The preview now reports channel
  names, total and per-channel point counts, and each channel's first and
  last sample, including the zero-sample channels that explain a missing
  trace.
- The session-report skill's reference file still taught the pre-long-format
  Chart.js recipe (`data: { labels: timestamps, datasets: [{ data: values }] }`).
  `archiver_downsample` gives every channel its own timestamps, so a
  multi-channel report built from that recipe drew each series against the
  first channel's x-values — a plausible-looking chart that was silently
  wrong. The recipe now maps per-dataset `{x, y}` points, skips enum/status
  channels that cannot share the numeric axis, and needs no date-adapter
  script.
- The DOOCS connector no longer fails when its configured `avg_window` is wider
  than the queried time span. The moving average used `np.convolve(mode="same")`,
  which returns `max(len(data), window)` points rather than `len(data)`, so the
  returned `data` array outgrew its `time` array and `get_data` died with a
  length mismatch. The average is now a pandas centered rolling mean, which
  returns one value per input point and shrinks the window at the edges instead
  of zero-padding — removing the separate edge-renormalization pass as well.
- Plotting two or more enum/status channels together no longer draws them on
  interleaved rungs. They share one categorical y-axis, and Plotly builds that
  axis's rungs from the union of its traces in first-appearance order, so each
  channel's step line crossed rungs belonging to the other — rendering as a
  state the channel was never in. Each channel's rungs are now namespaced to it;
  hover still shows the real, unprefixed value.
- The artifact gallery's `format=table` view no longer rebuilds the entire
  timeseries on every page click. It pivoted every (timestamp, channel) cell in
  the file and then sliced to the 50-row page, against a 200 MB file cap; only
  the requested page's rows are built now. The shared timestamp axis is still
  unioned in full — that is what the row count and page offset are measured
  against — but that is one entry per timestamp rather than one per timestamp
  and channel.
- The EPICS connector now honors two contract rules it documented but did not
  enforce, both reachable only through the connector API directly (not through
  `archiver_read`):
  - Aggregating a non-numeric channel with anything but `processing="raw"` now
    raises `ValueError` naming the channel, as `base.py` and the add-a-connector
    guide both require. EPICS pushes aggregation to the appliance and so never
    reached the client-side helper where the other three backends enforce this —
    `mean` over a string-valued PV came back as its raw `CW`/`STANDBY` values
    labelled as means.
  - A `precision_ms` that is not a whole number of seconds is now rejected
    instead of floored. The appliance's operator syntax takes seconds, so a
    500 ms request was silently served at 1 s (and 1500 ms likewise), while every
    other backend binned at exactly the width asked for.
- A DOOCS connector configured with `avg_window` no longer manufactures its
  timestamps. Because the moving average was a fixed-width convolution kernel,
  it needed a constant `dt`, so setting `avg_window` forced the samples onto a
  `numpy.linspace` grid via a zero-order hold — every returned timestamp was a
  grid position rather than an archived one, and every value was forward-filled
  onto it. `get_data` deliberately bypasses that grid, but `avg_window` brought
  it back through a config key, so the one connector with a smoothing knob was
  also the one still violating the no-manufacturing contract. The average is now
  a real time-duration window applied to the archived samples at their own
  irregular timestamps. An explicit `max_points` still returns a uniform grid —
  that is what the caller asked for.
- The DOOCS connector no longer waits forever when `get_data` is called without
  an explicit `timeout`. It passed the argument straight to `asyncio.wait_for`,
  where `None` blocks indefinitely, and had no configured default to fall back
  on — so an unresponsive ENS hung the caller with no recovery. `connect` now
  accepts a `timeout` config key (default 60 seconds), matching EPICS and
  MongoDB. An explicit `timeout=0` is still honored as a real request.
- The mock archiver is now genuinely reproducible, as its docstring always
  claimed. Noise for every non-BPM channel was drawn from the unseeded global
  `numpy.random` instead of the per-PV generator, so two identical queries in
  one process returned different data; and the per-PV generator was seeded from
  `hash(pv_name)`, which CPython salts per process, so even the seeded path
  differed between runs. Seeding now uses a stable checksum and all noise comes
  from the per-PV generator.
- `data_read`'s over-cap preview now also unwraps the legacy `_osprey_metadata`
  envelope, matching the plot tools' reader. Older archiver artifacts on disk
  carry that wrapper and were previewed as an opaque JSON object.

- Built projects' `config.yml` no longer misplaces section comments: entries
  added at build time (service blocks, `deployed_services` names, web panels,
  config overrides) rendered *after* the next section's comment banner —
  splitting the `deployed_services` list around the `SAFETY CONTROLS` header.
  Appended entries now render inside their own section, with the banner kept
  at the section boundary.

- The test suite no longer inherits a `TZ` supplied by a `.env` file, which made
  `tests/connectors/test_archiver_timezone.py` error on any machine whose system
  timezone differs from the one in `.env`. CI has no `.env`, so it never saw it.
- Importing an `osprey` module no longer loads `.env` into the environment.
  Previously any `import osprey.…` rewrote `os.environ` from whatever `.env`
  sat in the working directory — or, through LiteLLM, in any parent directory —
  overriding values the caller had set. `.env` now loads only where an
  application asks for it: the `osprey` CLI, MCP server startup, and the Claude
  Code launch paths. Every key is still passed through, unchanged, at those
  points. Code that imports OSPREY as a library and relied on the side effect
  must call `osprey.utils.config.load_project_dotenv()` itself.
- The `nextcloud_bridge` block in a generated `profile.yml` described itself as
  turning "a Nextcloud folder" into a trigger source. It answers questions from
  a Talk room; the comment now says so.
- Deleting an artifact from the gallery, and the dispatch worker's retention
  sweep, no longer show up as agent actions in the web terminal's activity
  strip. Only mutations the agent actually performed are reported.
- The python executor tools now reject `execution_mode` values other than
  `readonly` and `readwrite`. An unrecognized spelling used to slip past both
  write gates and run write-pattern code even with
  `control_system.writes_enabled=false`. The deployment-level kill switch also
  now covers `execute_file`, which previously had no such check.

- BLUESKY panel views taller than the window can be scrolled again. The panel's
  layout column grew to fit its content instead of staying at the viewport, so
  the scroll container had nothing left to scroll and everything past the fold
  was unreachable — worst in an embedded tile, where the overflow was clipped
  outright. Embedded panels also no longer lose the last 28px of every view to
  padding the frame already supplies.

- The axes table in a scan's parameter form no longer truncates device names —
  its columns are sized to the names they hold.

- The web terminal header names the product once. The page title repeating it
  alongside is gone.

- An embedded status strip now keeps its own inset. The queue badge sat against
  the tile frame and the halt buttons touched its corner, while the content
  below them was comfortably inset.

- The session picker's dropdown opens in front of the terminal rather than
  behind it. The tab strip it is adopted into clipped the menu and pinned it
  under the content pane whatever its stacking order; it now mounts at body
  level and closes on scroll, resize, Escape, or a click outside.

- The ORM figure reads as a signed response matrix. A corrector kicks the beam
  and a BPM downstream reads positive or negative by phase advance, but the
  matrix was painted on a single-hue ramp anchored to the raw extent: zero
  landed mid-ramp, so an unresponsive BPM looked like a real response, and the
  strongest negative response painted faintest. It is now diverging about zero
  and scaled against a symmetric max|value|, so equal and opposite responses
  read as equally strong and two runs stay comparable. Cells with no reading
  are hatched rather than left blank, and the legend became a colorbar carrying
  the numbers.

- New: an ORM result carries the two views that conventionally accompany a
  response matrix. **Response by BPM** reads the matrix column-wise, where the
  sign oscillation is legible and a dead corrector shows as a flat line against
  a dead BPM's shared spike; you pick which correctors are drawn without
  refetching. **Singular values** plots the spectrum on a log axis, with modes
  at or below the numerical-rank tolerance drawn as gaps rather than as the
  ~1e-16 residue a rank-deficient matrix leaves behind. The raw per-corrector
  sweeps move below the fit into a collapsed section that shows one corrector
  at a time; when the fit is skipped they stay inline.

### Added

- The web terminal's command palette keeps a Recent section listing the
  commands you last ran.
- The command palette can open a panel in a new window or in a new tile — the
  same two verbs the right-click menu offers. The new-window rows include the
  panel you are looking at, which is the one most often wanted in its own tab.
  The new-tile rows are absent in simple mode, whose layout holds one service
  tile.

### Changed

- Panel actions in the web terminal now live in a right-click menu, on the
  panel's rail entry and on its tile header alike: focus the panel, open it in
  a new tile, open it in a new window, remove it from the rail. It replaces the
  ↗ and ⊞ corner glyphs, which appeared only on hover and only on the rail. The
  menu also opens from the keyboard — the menu key or Shift+F10 on a focused
  rail entry.

### Fixed

- The command palette takes typing straight away. The search box was focused
  before the overlay was made visible, which does nothing, so the first
  keystrokes after Cmd/Ctrl+K were dropped and you had to click the box first.
- Palette search answers to the words operators actually reach for: a panel's
  domain vocabulary ("logbook" finds ARIEL, "pv" finds Channel Finder) and the
  verbs' synonyms ("window", "popout", "tile", "split").

## [2026.8.0]

### Removed

- **Breaking:** `osprey build --emit-profile` is gone, with no alias — the
  build command builds projects, and profile authoring now has its own verb.
  Materialize a profile directory with:

  ```
  osprey profile new DIR --preset X
  ```

  It writes everything the flag wrote, plus the preset's `data/` tree, and
  accepts the same `-O` / `--set` layers. Scripts still passing the old flag
  fail with an unknown-option error rather than silently doing something else.

### Changed

- A profile directory is now the durable source of truth for a deployment.
  `osprey profile new` writes a fully explicit, standalone `profile.yml` — the
  preset's resolved configuration (including any `extends` chain) materialized
  with its comments preserved, nothing inherited at build time — alongside the
  preset's `data/` tree copied verbatim and an `overlays/` seed. `--set` and
  `-O/--override` values are baked in place, so a validated build one-liner
  carries straight into an editable facility profile. Edit the profile
  directory and rebuild; the project stays a regenerable artifact.
  `osprey profile validate DIR` checks a profile without building, and
  `osprey profile presets` lists the bundled presets.

- The framework version is now derived from the git tag instead of a literal in
  `src/osprey/__init__.py`. A build between releases reports its distance from
  the last one (`2026.8.0.post12+g83fda5e60`) rather than claiming to be that
  release, in `osprey --version`, the status line, the web terminal health
  payload, and generated project READMEs. Cutting a release is now just tagging
  — there is no version to bump.

- `osprey deploy up` and `osprey build` now refuse to pin
  `osprey-framework==<version>` when running from a development checkout, since
  no published release corresponds to that code. Use `--dev` to build and stage
  a local wheel, or set `OSPREY_PIP_SPEC` to pin explicitly. Previously this
  emitted a pin for a version that was never published and failed later, inside
  the image build, with an opaque pip error.

- Web terminal header: the username badge and the logout button are now one
  identity chip on the right, whose menu holds the deployment name and Log out.
  The deployment name (`web.app_name`) moved to the left, beside "Web Terminal".
- Custom artifact-gallery categories moved from the top-level `categories`
  key into the `artifact_server:` block (`artifact_server.categories`), in
  both build profiles and rendered config.yml — the bare name was ambiguous
  next to unrelated notions like `health.categories`. No alias: the old key
  is no longer read. The profile-side block also accepts `host`/`port`/
  `auto_launch` overrides for the gallery server. Materialized profiles now
  include commented guidance for adding facility `mcp_servers:` and
  `artifact_server.categories`.
- Building the `control-assistant` preset now generates the virtual
  accelerator's channel manifest from the data tree the build sources, and
  writes `VA_CHANNELS_FILE` and `VA_LATTICE` into the project `.env`. The
  simulated machine therefore serves the channels in your profile's databases
  rather than the container's packaged fallback set. Both keys are rewritten
  on every rebuild, so an edited channel database reaches the running IOC.

### Fixed

- Providers that ship a default endpoint (ALS-APG, Stanford) now work without a
  `base_url` in config. The requirement check ran before the provider could
  supply its own default, so a config that omitted `base_url` failed with
  "Base URL required" instead of using the endpoint the provider already knew.
  An explicit `base_url`, and the environment override, still take precedence in
  that order.
- A run whose steps each wrote a file with the same name (two `plot.png`, say)
  now delivers all of them to a chat bridge. Previously the second overwrote the
  first in the shared upload directory and both were reported as delivered, so a
  plot went missing with nothing in the logs to say so.
- `web.theme` set to a concrete theme id (e.g. `desy-light`) now actually pins
  light or dark as the deployment default. It painted correctly and was then
  overridden by the viewer's OS preference a moment later. A bare family
  (`desy`) still follows the OS, and a user's own pick still wins over both.
- The theme picker now labels the DESY family `DESY` rather than `Desy`.
- Newly scaffolded projects set `web.theme: osprey`, a family that no longer
  exists; the terminal warned and fell back on every start. Now `main`.
- HTML-to-image export no longer runs `playwright install chromium` on every
  conversion. The browser availability check used Playwright's sync API, which
  fails inside a running event loop; the failure was misread as "browser
  missing", so an async caller (e.g. the dispatch artifact byte route) paid a
  subprocess and a network round-trip per conversion — and on hosts that cannot
  reach the browser CDN, every conversion failed even with Chromium installed.
  The browser launch is now itself the availability check: Chromium is installed
  only when a launch reports the binary is absent, at most once per process, and
  a launch that fails for any other reason surfaces unchanged instead of being
  reported as a missing browser.
- `osprey build` now fails with an actionable error when
  `claude_code.default_model` (e.g. `--set model=...`) names a model the
  selected provider does not serve. Previously the build only warned and the
  deployed web terminals crash-looped behind the reverse proxy (502).
- Chat bridges no longer drop an artifact whose conversion failed. A run's
  artifact descriptors predict `image/png` for everything the worker intends to
  render, but a conversion that fails at fetch time makes the byte route serve
  the original file instead — and a delivery path routing on the prediction
  rejected those bytes as "not a PNG" and discarded them with no error anywhere.
  Delivery now routes on what actually arrived (the bytes and the served
  Content-Type), so a failed render is delivered as a document rather than lost,
  and its filename takes the extension of what was served. The same prediction
  drove prior-image re-injection, which could hand the agent a text file
  labelled as an image on a follow-up question; a prior artifact is now
  re-injected under the type it was really served as, or not at all.

### Added

- Web terminal workspace: every panel tile now has the same header bar — six-dot
  drag grip, panel name, and a close button that closes just that tile (the
  panel stays on the rail; one click reopens it). Panels can be opened side by
  side as first-class gestures: drag a rail icon into the workspace to split
  exactly where you drop it, or use the ⊞ "open in a new tile" corner on a rail
  entry's hover. Opening an already-open panel beside moves its tile instead of
  duplicating it.
- A multi-user web-terminal deployment can now require a real login. Set
  `modules.web_terminals.auth.method` to `password` (passwords OSPREY manages
  and hashes for you) or `oidc` (your facility's single sign-on, mapped onto
  roster entries by an `oidc_subject` field), and every request to a user's
  terminal — pages, APIs and the live connection — is refused unless the
  browser holds a valid session for that user. `osprey deploy up` provisions
  each user's password hash into a `0600`, gitignored `.env.auth` that only the
  login service can read, printing any password it has to generate exactly once;
  `osprey deploy passwd <user>` rotates one later and ends that user's sessions.
  It fails closed throughout: without `tls.enabled` the deployment refuses to
  render rather than send session cookies in the clear (override with
  `auth.allow_insecure_http` only behind a TLS-terminating proxy), and a deploy
  aborts before starting anything if a password hash cannot be established. The
  default stays `auth.method: none`, and no preset turns it on — see the
  multi-user how-to for the full setup and how to roll it back.
- Each user in a multi-user deployment can have their own default theme: set
  `theme:` on a roster entry in `modules.web_terminals.users` (a family such as
  `desy`, or a concrete id such as `desy-light` to also pin light/dark). It
  overrides the image's `web.theme` for that user only, and the user's own pick
  in the display menu still overrides it.
- The multi-user landing page now uses the deployment's configured theme
  instead of a fixed palette, so it matches the terminals it links to.
- DOOCS facilities can now select their connectors by name: `control_system.type:
  doocs` and `archiver.type: doocs_archiver`, in `config.yml`, through `osprey
  config set-control-system doocs`, or from the interactive config menu.
  Previously the connectors shipped but were reachable only by spelling out their
  dotted class paths. Both still require `doocs4py` from the DOOCS environment.

- New `osprey.bridges.core` package: a channel-agnostic engine for connecting a
  chat or email channel to the OSPREY dispatcher/worker pair as its own process.
  It owns the parts that are the same for every channel — a crash-safe dedup
  claim taken before dispatch, conversation history replayed with each question,
  a retry queue drained in the background once the pair is healthy again, startup
  recovery for messages that were in flight when the process stopped, and worker
  artifact handling. A channel contributes only its wire format and platform I/O,
  through the `ChannelOps` seam.
- Your team can now ask the agent questions from a Nextcloud Talk chat room and
  get answers — including plots and files — back in the same room. Add a
  `nextcloud_bridge:` block to a build profile alongside a `dispatch:` block, set
  the bot account and room list in the project `.env`, and `osprey deploy up`
  brings up the bridge with the rest of the stack. In a group room only messages
  that mention the bot are answered; files are shared with the room's members
  rather than published as public links. Messages posted while the bridge is down
  are picked up on restart. See the "Deploy a Chat Bridge" how-to.
- Your team can now ask the agent questions from Google Chat — in a space or a
  direct message — and get answers, including plots and files, back in the same
  thread. Add a `gchat_bridge:` block to a build profile alongside a `dispatch:`
  block, set the Google service account key, subscription, and app identity in
  the project `.env`, and `osprey deploy up` brings up the bridge with the rest
  of the stack. In a space only messages that @mention the app are answered; in
  a direct message every message counts. Plots and files come back as public
  links anyone who has the link can open, because that is the only way Chat can
  show them; leave the bucket unset to answer text only. Run one bridge per
  subscription — Google hands each message to a single reader. See the "Deploy a
  Chat Bridge" how-to.
- Every service image is now overridable through the same `env → config →
  default` chain: new `OSPREY_POSTGRES_IMAGE` env var plus
  `services.postgresql.image`, `services.openobserve.image`, and
  `services.bluesky.tiled_image` config keys (the built service images already
  supported both layers). Useful for internal registry mirrors and pinned
  digests.
- Service compose templates are now claimable build artifacts: `osprey
  scaffold claim services/<name>` freezes a template for local editing (with
  `scaffold diff` drift reporting), and `osprey build` skips claimed services
  when refreshing templates from the framework.
- `osprey deploy up` now mints a strong per-deploy `ARIEL_DB_PASSWORD` into
  `.env` when the postgresql service is deployed; the container and the ariel
  DSN read the same value (previously both used the fixed `ariel` password).
  Existing Postgres volumes keep their original password — see the deploy
  how-to for the migration note.
- Simulation machine files accept two optional per-channel keys: `noise_abs`, an
  absolute Gaussian sigma in the channel's own units, and `texture`, slow
  baseline motion declared as `{"kind": "wander", "amplitude": …, "period_s": …}`.
  The existing `noise` key stays relative (a fraction of the value), so channels
  that sit at zero can now be given movement. Machine files using neither key
  parse and behave exactly as before.
- Build profiles take a new `environment:` block declaring the Python
  environment agent code runs in: `python` (the base interpreter — either a
  bare interpreter or a venv's), `packages` (extra requirements, additive to
  `dependencies`), and `inherit_exclude`. Where `python` names a venv, that
  venv's installed distributions are frozen into the built project's dependency
  record; basing on a venv interpreter does not otherwise carry its packages
  over. The build fails, naming every offender at once, on packages it cannot
  reproduce — ones installed from no package index, and ones whose version
  conflicts with osprey's own requirements. `inherit_exclude` is how you drop
  them.
- A CI-enforced guard keeps `config.yml` honest: every key the shipped templates
  render must have a reader in the framework, or a recorded reason it has none,
  and a key that was retired cannot come back in a template, a preset override,
  or the loader's defaults. Contributors can run it from a checkout with
  `uv run python scripts/check_config_keys.py`.
- `osprey theme-lab` opens a browser workbench for designing a theme: pick its
  two accent colors — the main one and the second used for highlights and
  warnings — see them previewed live in dark and light with contrast badges,
  then copy an export block describing the theme to request it. One set of
  controls edits whichever accent is selected. The second accent carries a
  contrast badge of its own, because the build holds it to the stricter
  body-text standard the main accent is not held to.

### Changed

- The event dispatcher panel is rebuilt around two tabs — Activity and
  Triggers — instead of five surfaces competing for the same screen. There is
  one place to fire a trigger, one trigger list, and one operator (Simple)
  view. Three long-standing faults go with it: timeline marks now sit at their
  actual times (every mark previously rendered at the left edge, so a quiet
  trigger looked the same as a busy one), an open transcript survives the
  three-second refresh instead of collapsing what you had expanded, and write
  actions no longer pop a token prompt inside the embedded panel. A run now
  also links to the trigger that started it and, where a telemetry store is
  deployed, to that run's own records — each link appearing only when there is
  something real to open.
- Raised minimum versions for `aiofiles`, `click`, `fastapi`, `httpx`,
  `matplotlib`, `mss`, `playwright`, `requests`, and `typing_extensions`, and
  regenerated `uv.lock` to match.
- `osprey web` now resolves the project it serves once, up front (`--project`,
  then `OSPREY_CONFIG`, then the current directory) and refuses to launch when
  no `config.yml` is resolvable, instead of silently serving a terminal with
  only the universal panels. The launch banner names the resolved project, the
  resolved config is published to child processes (including the `--reload`
  worker), and a detached server's command line always carries `--project` so
  a copied restart cannot lose the project identity.
- Every draggable divider in the web terminal now looks and behaves the same.
  Panes sit flush against each other with the grip attached to the pane edge
  (previously the workspace gallery and the plan panel floated their panes in a
  gutter), every divider can be moved with Arrow keys as well as the pointer,
  and double-clicking one collapses that pane and restores it to the width or
  height you had. The lattice dashboard's control sidebar, which could only be
  collapsed, can now be resized too.
- `osprey deploy --dev` now fails with a clear error when the local osprey wheel
  cannot be built, instead of warning and deploying the pinned PyPI release.
  Previously a missing `build` package (or a broken local checkout) produced one
  warning among many info lines and an exit code of 0, so the containers came up
  running released osprey and the deployment silently tested something other than
  the local code. The preconditions — editable install, source checkout, `build`
  package — are now checked before any deploy work begins, and `build` moved from
  the `dev` extra to a base dependency so an editable install always has it.
- `osprey build` now prints a provider-credentials summary that reports the API
  keys it *found*, not just the ones it didn't. It leads with the provider the
  project was built for, names where that key came from (project `.env`, the
  build directory's `.env`, or the shell), and warns if the selected provider's
  key is missing. Previously the build logged one line per *unresolved*
  placeholder — twice — so a successful key was silent, and a missing key for
  the selected provider looked identical to the irrelevant misses for providers
  the project never uses. The per-placeholder resolver line moved to `DEBUG`.
- Scaffolded `.env` / `.env.example` files derive their provider API-key list
  from the provider registry instead of a hand-maintained list (which had
  drifted: `ALS_APG_API_KEY` was missing, a stale Langfuse block remained, and
  a detected `ARGO_API_KEY` value was discarded in favor of a `$${USER}`
  placeholder).
- Shipped preset configs now document `deployment.bind_address` and point the
  Virtual Accelerator instructions at `osprey deploy up` instead of a
  repo-internal container path.
- Deploying the Bluesky bridge with `control_system.writes_enabled: true`
  no longer leaves the launch path permanently unarmed — `BLUESKY_LAUNCH_TOKEN`
  is now minted for every deployed bridge. The enforced boundary is unchanged:
  the connector re-reads `writes_enabled` and applies limits on every setpoint.
- `execution.execution_method` now names the backend that actually runs:
  `subprocess`. Generated configs write it, `local` is accepted silently as a
  synonym, and `container` still loads but runs on the subprocess backend and
  logs a one-time deprecation warning naming the config file it came from. Both
  legacy values stop being accepted in 2027.1.
- Generated `config.yml` files no longer record `execution.python_env_path`, an
  absolute host interpreter path that went stale as soon as the project moved.
  Agent Python runs in the project's own `.venv` when it has one, resolved at
  run time. Configs that still carry the key load unchanged; it is ignored.
- Other Jupyter-era execution keys nothing reads are gone the same way:
  generated configs no longer carry `execution.modes`,
  `python_executor.max_generation_retries` / `max_execution_retries`, or
  `file_paths.executed_python_scripts_dir`, and an `execution.modes` block in
  an already-deployed config is ignored on load. A config without an
  `execution:` section no longer logs a warning — subprocess execution is the
  default, not an anomaly.
- The unreachable Jupyter-container execution machinery is deleted: the
  container engine, the wrapper's container mode, the notebook/file managers
  (and the `http://localhost:8088` notebook links they minted), their models
  and exception hierarchy, and the artifacts API's interactive-notebook
  endpoint. `osprey.services.python_executor` now exports only the analysis,
  limits-validation, and serialization utilities the subprocess backend uses.
- The Python-execution and visualization tool descriptions now name the
  packages actually importable where each one runs code, enumerated once at
  server start, instead of a fixed `numpy, pandas, scipy, at, matplotlib,
  plotly` list. The visualization tools report the sandbox's installed set
  intersected with its import allowlist. If the environment cannot be
  enumerated, the description names no packages rather than guessing.

- The model-benchmark matrix now scores two lanes separately: `agentic_benchmark`
  marks genuine model-capability e2e tests (the headline pass rate) and
  `harness_benchmark` marks model-independent safety/plumbing assertions, so
  harness passes no longer pad a model's capability score. Every in-scope e2e
  test must declare its lane (gated per matrix cell and in CI); 19 non-LLM e2e
  files moved to the matrix exclusion list. The `e2e_benchmark` marker was
  renamed to `channel_finder_benchmark` to say what it actually covers.
- Web terminal header: the Expert/Simple toggle and theme controls are collapsed
  into a single display-menu dot that opens a popover with appearance
  (light/dark), view, and theme-family pickers. The header's search box and the
  display menu — System Settings included — now look and behave the same in both
  Expert and Simple, so nothing in the top-right corner moves when you switch
  view; the standalone "?" button is gone (the safety guide is still one search
  away). The popover also stays open while you switch appearance, theme, or
  view, so you can flip back and forth without reopening it.
- The default theme family is now named **main** (it was `osprey`): use
  `web.theme: main`.
- Workspace gallery browser: the three stacked header rows (title/count bar,
  type filter chips, controls row) are collapsed into a single toolbar —
  filter input, Types/Activity toggle, and a `⋯` menu holding the rare
  actions (all-sessions scope, refresh, layout). The all-sessions scope shows
  as a dismissible pill above the list while active, and pinned artifacts are
  promoted into a "Pinned" section at the top of the type tree.
- The **high-contrast** family is now fully monochrome — pure black and white,
  with status, diffs, chart series and terminal colors separating by brightness
  instead of hue. It was previously a high-contrast variant of the pre-redesign
  palette, and still meets the same WCAG AAA gates.

- A config that does not say which control system it talks to now gets the mock
  connector instead of EPICS, with a warning naming `control_system.type`. The
  same rule applies to the archiver: a missing or blank `archiver.type` resolves
  to the mock archiver — previously a missing one selected the EPICS archiver and
  a blank one crashed. The `hello_world` and `project` templates now ship a
  minimal `archiver:` block so the choice is visible. Configs that name their
  connector and archiver are unaffected.
- `claude_code.default_model` is resolved in three ways and never silently
  substituted: unset uses the provider's default tier, a tier name
  (`haiku`/`sonnet`/`opus`) selects that tier, and a model ID the provider's
  tier map declares is used verbatim. Anything else is now an error that
  names the valid tiers and the provider's model IDs. Previously an unrecognized
  value — a stale model ID, or one belonging to a different provider — was
  quietly replaced by the provider's default tier, so a project asking for Opus
  could run Haiku with nothing in the log. *Migration:* if a build now fails on
  this key, set it to a tier name (which stays valid when the provider changes)
  or to one of the model IDs the error lists. The shipped presets now set the
  tier `haiku`.
- A provider that maps no model for a tier is refused at build time, with the
  `api.providers.<name>.models` block to fill in. Unmapped tiers were previously
  filled with Anthropic's own model IDs, so a proxy or gateway shipping no map
  launched the agent asking for a model it does not serve.
- `api.providers.<name>.base_url` now overrides the built-in endpoint for
  built-in providers too, matching how the model map already worked — a facility
  fronting a shipped provider with its own gateway gets the agent pointed at the
  endpoint `osprey health` probes.
- `ariel.database.uri` is optional. With it unset, the DSN is derived from the
  project's `services.postgresql` block (username, database name, host port, and
  the `ARIEL_DB_PASSWORD` the deploy mints), so moving the database port no
  longer means editing a second copy of the same facts; the templates no longer
  render a hardcoded `uri:`. An explicit `uri` still wins verbatim, as does the
  older `connection_string` spelling (honored, with a warning naming its
  replacement), and `osprey health` now cross-checks an explicit loopback DSN
  against `services.postgresql.port_host`.
- Virtual Accelerator gateways that declare no `port` now follow
  `services.virtual_accelerator.port` instead of a hardcoded `5064`, so moving
  the deployed soft-IOC's port moves the connector with it. An explicit gateway
  port still wins; the templates no longer render `port: 5064`.
- The mock archiver derives `simulation_file` from the control system's own
  simulation file when its own key is unset, so archived history and live reads
  come from one machine model. An explicit archiver-side value still wins, and a
  disagreement between the two is warned about.
- `osprey health` reports configuration more honestly: an empty
  `deployed_services` list is a skip rather than a warning (attached and
  service-free projects ship it empty), the timezone remediation names
  `system.timezone` in `config.yml` instead of a `TZ` variable that no longer
  clears it, the container checks query the runtime `container_runtime` selects
  rather than whatever auto-detection finds first, and the agent-data check reads
  `agent_data.base_dir`.
- `facility.name` is the canonical facility identity, read the same way by the
  build path and by every interface that labels its UI; presets now set it in
  the `facility:` block. A top-level `facility_name` still works as a fallback.
- `agent_data.base_dir` is the single key naming the agent-data directory. The
  runtime, the health check, and the compose mounts all resolve the same
  directory from it, and generated configs declare it explicitly.
- Logbook composition uses the project's configured provider —
  `logbook.composition.provider`, falling back to `claude_code.provider` — and
  fails with a clear error when none is configured, instead of always calling
  Anthropic. The model ID comes from that provider's tier map;
  `logbook.composition.model_id` is no longer written into generated configs but
  is still honored to pin a literal ID.
- `ariel.enhancement_modules.semantic_processor.provider` is required when that
  enhancement is enabled, and an unset one is an actionable error rather than a
  silent fall-through to `ariel.embedding.provider` (which defaults to `ollama`,
  an embedding endpoint, not a completion one). The duplicate nested
  `model.provider` key is gone; the module-level `provider` is the only one.
- `claude_code.telemetry.protocol: grpc` combined with an auto-derived
  OpenObserve endpoint now fails the build. OpenObserve serves HTTP only, so
  that pairing produced an exporter that dropped every metric and log silently.
- The artifact gallery tab appears only when its server is actually running:
  with `artifact_server.auto_launch: false`, or after a failed launch, the
  WORKSPACE tab is unavailable instead of an enabled tab whose iframe returned a
  bare 502. Every companion panel's host and port now come from one resolver, so
  the URL published to the terminal and the port the server binds cannot
  disagree, and the `OSPREY_*_PORT` overrides apply on both sides.
- `facility_knowledge.bundle_path` resolves identically for its three readers
  (the MCP server, `osprey knowledge`, and the KNOWLEDGE panel): `~` is expanded,
  and a relative path is resolved against the directory holding `config.yml`.
- The control-system wizard disables the MongoDB archiver choice when the
  `archiver-mongodb` extra is not installed, naming the install command, instead
  of writing an `archiver.type` the environment cannot construct.
- The agent's setup-mode config patcher reports `control_system.writes_enabled`
  as a cold change requiring `osprey claude regen` and a restart. It was
  advertised as taking effect immediately, so an operator who flipped it
  in-session was told writes were live while the connector and the enforced deny
  list still blocked them.
- A project that still sets
  `control_system.write_verification.fail_on_mismatch: true` gets a one-time
  warning at its first write. Nothing ever read that key: a failed verification
  does not block or roll back a write. `write_channel_checked()` is the path
  that enforces verification, and scan plans write through it.
- The web terminal's settings drawer edits the write-verification level
  (`control_system.write_verification.default_level`) as a dropdown of `none` /
  `callback` / `readback`. Its enum was attached to a key shape that does not
  exist, so the live setting was previously edited as free text.
- Generated configs now document keys that were previously discoverable only in
  the source: the panel-port block for every web panel a project ships, the
  `web_terminal:` and `hooks:` blocks, `bluesky.plan_dirs` trust tiers, the three
  channel-finder pipeline modes, `development.api_calls`, and `web.theme`.
  Comments that described behavior the code does not have were corrected —
  including the safety surface: how far `approval.default_policy` actually
  reaches (only tools whose matcher runs the approval hook; everything else is
  gated by the rendered `settings.json` permissions), all three effects of
  `hooks.debug` and its unrotated JSONL, the warning that
  `control_system.patterns` overrides rather than extends the built-in patterns,
  and what `control_system.write_tools` covers.


### Removed

- The `apex` theme family.

- Configuration keys that nothing read are retired — from the shipped templates
  and presets, and from the framework's own config classes and loader:
  `control_system.write_verification.{enabled,fail_on_mismatch,timeout}`,
  `approval.tools.channel_limits`, `control_system.connector.timeout`,
  `connector.mock.simulate_delays` (the mock's real knobs are `response_delay_ms`
  and `noise_level`), the `machine_state:` block and its unused reader,
  `channel_finder.explicit_validation_mode`, the channel-finder `benchmark`,
  `processing` and `tree_preview` sub-blocks,
  `file_paths.{agent_data_dir,user_memory_dir,execution_plans_dir,prompts_dir}`,
  `workspace.base_dir`, `api.providers.ollama.{host,port}`,
  `ariel.{reasoning,default_max_results,cache_embeddings}`, the `applications:`
  block, and `system.facility_name`. An existing `config.yml` that still carries
  any of them keeps loading, silently and unchanged — retired keys are tolerated,
  not fatal; they simply have no effect. Two exceptions:
  `write_verification.fail_on_mismatch: true` warns once at the first write, and
  `file_paths.agent_data_dir` is no longer read at all, so a non-default value
  there now resolves under `./_agent_data` — move it to `agent_data.base_dir`.
- The configuration the loader hands to the runtime no longer fabricates
  OpenWebUI-era identity fields (`user_id`, `chat_id`, `session_id`,
  `thread_id`, `session_url`) or the `applications` / `current_application`
  scoping that went with them. Nothing read the identity fields; the
  `applications` scoping was read only to resolve per-application `file_paths`
  overrides, which no shipped template ever declared.
- Scaffolded projects no longer create `_agent_data/user_memory/`, and `.env`
  no longer carries a `TZ` line detected from the host — the facility timezone
  is `system.timezone` in `config.yml`.


### Fixed

- Dragging the horizontal dividers in the events panel no longer lags behind the
  pointer. The timeline pane animated the same height the drag was setting, so it
  eased toward a target the cursor had already left and trailed by up to 85
  pixels for the whole gesture.
- `osprey build` no longer aborts partway through creating a project's virtual
  environment on a slow connection. Installing osprey's dependencies was capped
  at five minutes, which a first-time download can exceed, and the build stopped
  with an unexplained "Unexpected error". The limit is now generous enough for a
  full download, and if it is ever reached the message names the install as the
  step that ran long and suggests what to try.
- Dispatched runs that delegate to a subagent now wait for the delegated work
  and return the full answer. Previously the reply could stop at "the agent is
  searching, I'll notify you when it completes" and nothing further arrived.
- `osprey web --project <dir>` launched from outside the project now behaves the
  same as running `osprey web` inside it. Previously the flag only set the
  terminal's working directory, so the project's `.env` was never loaded
  (leaving `${VAR}` placeholders such as a provider `api_key` unexpanded), the
  project's `web_terminal` and `claude_code` settings were replaced by built-in
  defaults, and `_agent_data/` was created next to wherever the command was run.
- A built project's container image now installs the same package set as its
  host environment — both are rendered from the project's own recorded
  dependencies. Previously the image was built from a separate list, so a
  package the agent could import on the host could be missing from the
  deployed image.
- Agent Python execution works in a freshly built project. Any
  `execution_method` other than the literal `local` fell through to a
  Jupyter-container backend that OSPREY does not ship, so execution failed;
  the subprocess backend is now the only path.
- A dispatched agent run no longer starts before its MCP servers finish
  registering. The servers connect asynchronously, so a run whose first turn
  fired during that window saw none of the project's tools and answered "I
  don't have that tool" — indistinguishable, after the fact, from the model
  declining to use them. The worker now waits for the project's declared
  servers to report connected before sending the prompt, as interactive runs
  already did. A server that never registers is logged and the run proceeds.
- Turning off a telemetry content gate (e.g.
  `claude_code.telemetry.log_assistant_responses: false`) now writes an
  explicit `OTEL_LOG_*=0` into the deployed environment. Previously the
  variable was simply omitted, and Claude Code's own fallback chain could
  re-enable capture the config had turned off.
- ARIEL logbook ingestion no longer skips an otherwise-valid entry when the source
  payload omits its `id`: the ALS and generic adapters now fall back to an empty
  entry id (matching the JLab/ORNL adapters) instead of raising a `KeyError` the
  fetch loop caught and dropped the entry on.
- All artifact stores are now rooted at the shared data root, so artifacts saved
  from session-scoped writers (e.g. resumed web-terminal sessions) stay visible to
  the gallery.
- `artifact_focus`/`artifact_pin` now report gallery failures honestly instead of
  always claiming success.
- `web.app_name` in `config.yml` now actually labels the web terminal header: the
  runtime read the key from a nested section nothing generates, so only the
  `OSPREY_WEB_APP_NAME` env override worked. It now reads top-level `web.app_name`,
  matching `web.theme` and `web.presets` (env override still wins).
- A server-configured `web.theme` family now survives a visitor's first page load:
  the in-browser theme runtime adopted the default family on first visit instead of
  the configured one. Light/dark still follows the OS until the visitor picks a mode.
- How-to documentation refreshed against the current code: provider model IDs,
  deploy/build semantics (`--force` preservation, `--dev` image builds, full
  subcommand list), telemetry now documented as on-by-default, MCP/executor error
  contracts, and the ARIEL web-interface module tables.
- Presets that render Claude Code artifacts now ship the `osprey_focus_validate.py`
  and `osprey_panels_context.py` hooks their `settings.json` already referenced;
  existing rendered projects will be flagged stale and pick up the two hooks on
  regeneration.
- Simulated channels sitting at a `0.0` baseline no longer read back as dead-flat
  constants. Relative `noise` is multiplicative, so it vanishes at zero and BPM
  positions and corrector current readbacks declared noisy were perfectly still —
  in live reads and in synthesized history alike. Mock and Virtual Accelerator
  reads now put an absolute per-kind floor under the noise (a `noise_level` of
  exactly `0.0` still means deterministic), machine files can declare `noise_abs`
  and `texture`, and loading a machine file that declares relative noise on a zero
  baseline now warns and names the affected channels.
- Synthesized archiver history is pointwise deterministic: each sample's noise is
  keyed to its channel and timestamp instead of drawn from a running stream, so
  repeated, overlapping and time-shifted queries agree at shared timestamps.
  Timestamps are keyed at millisecond resolution; windows whose timestamps are not
  convertible to epoch seconds keep per-window determinism only.
- The shipped control-assistant simulation data now produces organic BPM and
  corrector-readback signals instead of flat lines, and corrector channels gained
  the symmetric upper current limit their lower limit implied.
- Workspace gallery: the Simple view's result card now shows every artifact type
  the Expert preview does. Markdown, JSON, plain text, PDFs and archiver
  timeseries previously appeared there as a type icon or a raw summary dump —
  which covered channel-finder results and the agent's own written answers, since
  those are saved as markdown or JSON. Both views now render through one shared
  renderer, so no type can display in one view and not the other.

### Added

- Web terminal: the panel rail can now sit along the top (`web.rail_position: top` or the panel "+" menu).
- Web terminal: new `retro` theme family restoring the pre-redesign look (`web.theme: retro`) — the navy/teal palette, the CRT treatment, and the horizontal tab bar. Setting `web.rail_position` explicitly still pins the rail in every theme.
- A `demo-ui` skill runs short scripted demonstrations of the agent driving the web workspace: a panel tour, an artifact hand-off, and a layout switch, individually or back to back. It reads the live panel inventory rather than assuming a fixed tab set, and restores the starting layout when it finishes.
- The web terminal's Simple mode now starts as a clean chat-first experience: with an empty agent workspace the page shows only the chat, and the WORKSPACE panel appears the moment the agent shares its first artifact (`show_panel`); a workspace that already holds artifacts opens as before. The OSPREY agent is told at session start which surface it serves — Simple sessions are instructed to bring up the WORKSPACE panel whenever they produce something the operator should see.
- A `channel-finder-standalone` preset packages OSPREY's natural-language channel/PV address finder — the channel-finder pipeline plus its interactive CHANNELS web panel — as a standalone, read-only deployment with no control-system stack, archiver, logbook, or Python executor. It ships a bundled demo hierarchical database so it runs out of the box; `channel_finder_mode` selects the `in_context`, `hierarchical`, or `middle_layer` pipeline.
- The control-assistant preset now ships the KNOWLEDGE panel, a browser for the project's facility knowledge bundle. Existing projects gain the tab by adding `okf` to `web_panels` and rebuilding.
- Agent actions are now highlighted live in the web terminal: the plan panel follows the OSPREY agent's drafts (with a banner instead of a switch when you have unsaved edits), panels the agent touches glow and carry an attention badge on the panel rail until you open them, and backend actions — channel writes, run launches — appear briefly in the status-bar activity strip.
- Explicit `--set provider=` / `--set model=` / `--set channel_finder_mode=` build overrides now propagate to the persona projects that multi-user deploys auto-render: the manifest records which of these keys were explicitly passed, and `osprey deploy up` forwards them to each persona's `osprey build` — so one override at build time retints the whole stack. Preset defaults are never forwarded, keeping per-persona provider customization intact.
- Broad unit-test coverage for previously untested modules across services (migration engine, channel-finder data layer and tools, python-executor sandbox plumbing), interfaces (lattice-dashboard physics workers, web-terminal file/chat/scaffold routes incl. the path-traversal guard), CLI menus, MCP servers, registry loader/export, deployment, template hooks, and utilities.
- A bluesky scan plan can now be hidden from the agent without turning off the whole scan server. Set `bluesky.excluded_plans` on the profile of the project that deploys the bridge; the deploy render carries it into the bridge as the `BLUESKY_EXCLUDED_PLANS` environment variable. An excluded plan is both absent from the agent's plan list and non-runnable — it cannot be staged or launched by name. The bare config key is a local/development convenience; the environment variable is the production channel.

- `osprey deploy up` and `osprey deploy status` now warn when the project's render is stale — i.e. it was rendered by a different osprey version, or the preset's content has changed since the render (a content hash of the resolved preset is stamped into `.osprey-manifest.json` at build time). The warning names the exact `osprey build ... --force` command to re-render; it never blocks a deploy, and legacy projects without the stamp are unaffected.
- Every `osprey deploy up` now ends with a service-endpoint summary (published host ports from the rendered compose files, plus the web-terminal landing URL — or an explicit "web terminal (not configured in this project)" line when the config declares no web tier).

### Changed

- Logging is now configured explicitly and writes to stderr. Importing Osprey no longer installs a log handler as a side effect — entry points call `osprey.configure_logging()` once at startup, and code that embeds Osprey as a library (notebooks, scripts, preset repos) should do the same to see log output. Log lines that previously appeared on stdout now appear on stderr, so stdout carries only program output: `--json` payloads stay machine-readable and MCP stdio traffic stays clean. `configure_logging()` adds to whatever logging a host application has already set up and never removes handlers it did not install.
- The ARIEL panel no longer shows the logbook Search tab when embedded in the web terminal — search there goes through the agent, so the embedded panel offers Browse, New Entry, and Status and opens on Browse. Standalone ARIEL keeps Search as the default view.
- `osprey build` now records a project's dependencies in a generated `pyproject.toml` instead of `requirements.txt`. This makes `uv run osprey web` (and any other command) resolve the project's own `.venv` rather than walking up to an ancestor project's environment, and makes `uv sync` rebuild the environment instead of pruning it empty. Existing projects can delete their now-unused `requirements.txt` on the next `osprey build --force`.
- `osprey deploy up` now runs the web-terminal preflight (persona auto-render and the fail-closed `.env.production` credential gate) *before* building any image, so a deploy doomed to abort on a missing provider secret says so in seconds instead of after the full image build. When the missing variable is exported in the caller's shell but absent from `.env`, the error now says so and names the exact copy-in command (`.env` remains the only secret source the generator reads).
- The `osprey-build-interview` skill now asks the installed framework what it offers instead of carrying its own catalog: presets, build artifacts, providers, and config keys are all read from the live installation at interview time, so a newly shipped capability is offered without anyone editing the skill. It generates the profile with `osprey build --emit-profile` rather than a hand-written YAML template, and builds that profile itself before handing it over — what you receive is known to build. The interview now adapts its questions to the person rather than following a fixed script, and takes about five minutes. Legacy-project migration, the feedback prompts, and the web-panel design step were removed; panel authoring belongs to the `creating-an-osprey-panel` skill.
- New `osprey.build` package holds the build-time kernel shared across layers (Claude Code model/provider resolution, telemetry env block, channel-finder tier defaults, manifest primitives); agent-runtime helpers (clean child-env, SDK system-prompt, artifact-path resolution, Claude Code project-path encoding) moved to `osprey.agent_runner`. This removes the `services`/`mcp_server` → `cli`/`interfaces` layering inversions; internal import paths changed with no compatibility shims.
- Removed the legacy facility-config `gitlab:` block: a config that still carries it now fails closed with a `ConfigurationError` naming the `ci: {provider: "gitlab", ...}` replacement, instead of being silently aliased.
- Bluesky panels app moved from services to interfaces (import path `osprey.interfaces.bluesky_panels`).
- python-executor: removed the deprecated `epics_writes_enabled` field; `control_system_writes_enabled` is now the single write-gating flag.
- The seven LiteLLM-thin provider adapters (anthropic, als-apg, amsc-i2, cborg, google, openai, stanford) now share a single data-driven delegating base; behavior is unchanged (Stanford keeps its base-URL fallback).
- The three channel-finder MCP servers now share one bootstrap module (config load, path resolution, `python -m` entry point, startup sequence); behavior and entry points are unchanged.
- `osprey build --force` now re-renders an existing project *in place* instead of deleting the directory: `.env` (existing values win over freshly detected ones, and keys only it carries are kept), `_agent_data/`, and the project's `.git` are preserved; everything framework-rendered, including `data/`, is rebuilt. A profile-provided `env.file` is likewise merged into an existing `.env` rather than overwriting it.

### Fixed

- `osprey ariel purge` now clears the text-embedding migration record along with the dropped embedding tables, so a subsequent `osprey ariel migrate` actually recreates them instead of silently no-opping.
- Containerized Python execution no longer misclassifies an infrastructure failure during result collection as a code error: pre-classified executor errors keep their retry category, so an infrastructure fault re-executes the same code instead of triggering code regeneration.
- The Stanford provider's health-check model id had a typo (`gpt-4.omini` → `gpt-4o-mini`), also fixed in its available-models list.
- The AskSage provider now falls back to its static default model list when a `/models` fetch fails or credentials are missing, instead of returning a malformed value that could reach a UI caller. The fetched list is also cached across adapter instances, so an AskSage completion no longer pays a repeat `/models` round-trip on every request.
- Every companion web panel now gets its own per-user port family in multi-user deployments. The family set is derived from the web-server registry — previously it was a hand-maintained list that missed the channel-finder and OKF panels, so a second user's container collided with the first on the panel's fixed port (crash-looping the container once the `osprey web` preflight landed). Families omitted from config fall back to registry defaults (`channel_finder_base_port` 9591, `okf_base_port` 9691), so existing configs deploy unchanged.
- Bluesky PLAN/RESULTS/HEALTH panels now resolve their API endpoints correctly under the multi-user `/u/<user>/` mount: the shared `panelApiPrefix()` helper accepts an outer proxy prefix, the health panel no longer relies on the proxy's content rewrite (which double-prefixes once the runtime prefix is correct — this also fixes the proxied single-user health panel), and a guard test keeps panel bundles free of literals colliding with the proxy rewrite list.
- Panel tabs without a configured health endpoint (e.g. PLAN and RESULTS) now show a green LED instead of a permanently red one — the tab-state painter runs for panels that skip health polling.
- The multi-user landing page is served only at `/`; any other path outside a `/u/<user>/` mount now returns 404 instead of silently answering with landing-page HTML.
- `osprey deploy up` hot-reloads nginx after reconciling the web-terminal stack, so re-rendered `nginx.conf` routing changes take effect without a manual container restart.

- `osprey deploy up` is now idempotent from any prior state: it first removes the project's own stale non-running containers (a container left in `created` state by an aborted deploy holds its published host ports on Docker Desktop, blocking the next `up` with "address already in use"), and the plain services path reconciles away containers of services removed from the config. Running containers, volumes, and sibling deployments on the same host are untouched.
- `osprey deploy rebuild` on a web-terminals project now brings the web-terminal stack (nginx, per-user containers) back up after the clean; previously it restarted only the backend services. Per-user volumes survive a rebuild.
- Local-mode `.env.production` generation now includes the auth secret for every `claude_code.provider` in play — the deploy config's own and each referenced persona project's. A persona whose secret is missing from `.env` aborts the deploy naming the exact variable, and an existing `.env.production` that contains none of the configured LLM credentials draws a warning; previously the file could silently omit the credential entirely, producing healthy-looking web terminals that fail authentication on their first prompt.
- `osprey deploy up` probes the web-terminal landing page from the host after bringing the stack up and warns when it is unreachable. On Docker Desktop (macOS/Windows), `network_mode: host` binds inside the Docker VM unless the opt-in host-networking setting is enabled — previously this state was reported as a fully successful deploy; the warning now names that setting.

### Added

- **Shared ALS-U Accumulator Ring lattice** — the real ALS-U AR design (2.0 GeV, 182.12 m, harmonic 304) ported to pyAT as a plain-venv-importable `osprey.simulation.lattice` module (`build_ring()`), alongside a declarative `FacilitySpec` (`osprey.simulation.facility_spec`) that is the single source of truth for device families, per-family counts, and the `AR:{sup}:{fam}:{id}` naming scheme. A canonical `.mat` build artifact ships with the package; regenerate it with `python -m osprey.simulation.lattice.build`.
- `channel_limits` gains a `name_contains` parameter for literal substring search, so channel names containing regex metacharacters (`[]`, `()`, `.`, `^`) can be looked up without manual escaping. The existing regex `pattern` behavior is unchanged; the two are mutually exclusive.
- The `control-assistant` preset now ships the multi-user web-terminal stack built in: `osprey deploy up` stands up nginx (landing page at `:9080`) and per-user terminals for a two-user roster mapped to two new persona presets — `control-assistant-readonly` (the default) and `control-assistant-readwrite` (write-capable through the ordinary safety chain). Single-user onboarding is unchanged (`osprey web` never reads the block); set `modules.web_terminals.enabled: false` in the rendered config to deploy backend services without the web tier.
- **Canonical logbook entry URLs from ARIEL** — ARIEL read tools (search, browse, entry lookup, SQL query, publish) now emit an `entry_url` for facility logbook entries, rendered from a configurable `ariel.entry_url_template`, so the agent links entries verbatim instead of fabricating a plausible-but-dead URL. Off by default: deployments with no template configured, and ARIEL-native entries not yet in the facility logbook, emit no URL. The read path is fail-safe — a malformed template degrades to no URL rather than crashing.
- **Health checks on the agent surface** — an opt-in `health` MCP server (`claude_code.servers.health.enabled: true`) lets the OSPREY agent read the health suite through two tiered tools: `health_check` (auto-approved, served from a per-session cache) and `health_check_full` (approval-gated, always runs the `on_demand` checks fresh). The `mcp_servers` category is now auto-derived from the wired `claude_code.servers` blocks — one reachability check per server, with expected tools taken from each server's declared permissions — so no `health.categories.mcp_servers` need be hand-authored (`health.auto.mcp.{enabled,url_key}` tunes it). See the "Configure Health Checks" how-to.
- **Built-in service health tiles and an archiver data probe** — `osprey health` (and the SYSTEM dashboard) gains two presence-gated categories that appear only when the corresponding service is configured: `ariel` (interface reachability, logbook entry count, last ingestion, registered search/enhancement modules) and `channel_finder` (active pipeline, channel-database presence and age, middle-layer channel count). A new `archiver_freshness` probe type verifies the archiver is actually accumulating data — the newest archived sample of a canary channel must be younger than `max_age_s`. The health-checks how-to gains a control-system smoke-test recipe (canary `channel_read` + `archiver_freshness`).
- **Multi-user web terminals behind a single origin** — the multi-user web-terminal stack now binds every per-user service to loopback and routes all browser traffic through one nginx origin at `/u/<user>/`, instead of exposing each per-user app on its own host port; the Web Terminal SPA serves identically under that per-user prefix. Optional, off-by-default auth and TLS seams are wired into the deploy config (`modules.web_terminals.auth` / `.tls`): they render the corresponding nginx blocks with no frontend change and fail closed (an enabled `auth.method` with no backend returns 403) — so this remains a trust-the-network deployment until a real auth backend and TLS certificates are configured. Logging out now ends the warm terminal session and returns to the landing page.
- **Panel presets ("Layouts") for the Web Terminal** — a deployer can declare named sets of panels under a new `web.presets` block in `config.yml`, and an operator applies one in a single click from a "Layouts" section at the top of the panel "+" menu. Applying a preset shows exactly its panels and closes the rest. It reuses the existing panel-visibility path (no new endpoints or state), and when no presets are configured the "+" menu looks exactly as before. See the "Panels" how-to.
- **Add and remove Web Terminal panels from the browser** — the panel tab strip now has a browser-style "+" button to show a hidden panel, and each tab has a hover "×" to close it. When `web.allow_runtime_panels` is enabled, the "+" menu also offers a "new panel from URL" field (subject to the same URL validation and allowlist as agent-driven registration). The human and the agent share one panel set, so a panel added or removed either way updates for everyone.
- **Agent telemetry over OpenTelemetry (opt-in)** — a `claude_code.telemetry` block in `config.yml` (off by default) makes the OSPREY agent emit its operational logs and metrics over OTLP to any OpenTelemetry-compatible backend, from every launch path (CLI chat, Web Terminal, dispatch worker, SDK). An optional local **OpenObserve** backend ships as an opt-in `osprey deploy` add-on (add `openobserve` to `deployed_services`) — a single public container that ingests OTLP and serves a browser UI, with ingest auth bootstrapped from the same `.env` credentials it uses for its admin login. Full-content capture (prompts, responses, tool calls, raw provider bodies) defaults on for the local air-gapped store and is per-key configurable. See the "Monitor Your OSPREY Agent" how-to.
- **Per-surface dispatch differentiation** — a trigger's `action.surface_prompt` appends a static awareness fragment to the dispatched agent's system prompt, and an optional per-surface tool scope narrows (never widens) the trigger's `allowed_tools`. Both are no-ops when unset, so existing triggers behave exactly as before.
- `pyat-specialist` subagent: specialist agent scoped to lattice/optics computation over the simulated ALS-U AR ring, writing and executing pyAT code via the python execution service; enabled by default in `control_assistant`; delegates lattice/optics computation out of the main agent's context and returns provenance-carrying answers (`lattice_analysis` artifacts, labeled as simulation-derived).
- Every OSPREY browser interface — Web Terminal, Artifacts, ARIEL, Channel Finder, the Lattice dashboard, the event dispatch dashboard, the KNOWLEDGE (facility-knowledge) panel, and the session activity/safety pages — now themes itself from one shared design-token system with dark, light, and `auto` (follows your OS color-scheme preference) modes. See the "Theming" how-to for adding a new theme or wiring a new interface into it.
- **Design token scales** — type, font weight, line-height, spacing, radius, z-index, and duration are now generated CSS variables (`--text-*`, `--weight-*`, `--leading-*`, `--space-*`, `--radius-*`, `--z-*`, `--duration-*`) alongside the existing color and font tokens, with a hygiene check enforcing zero bare scale literals in migrated interfaces. The Web Terminal and design-system CSS are fully migrated onto them. A live, runtime-enumerated token reference page is served at `/design-system/reference.html` in every interface; see `src/osprey/interfaces/design_system/DESIGN.md` for the designer-facing contract.
- Themes are now grouped into **families** — a family is a `{light, dark}` pair. Alongside the existing `osprey` family, a new WCAG-AAA `high-contrast` family ships out of the box. The theme switcher now picks a family, and toggling light/dark stays within the active family. A new `web.theme` key under `config.yml`'s `web:` section sets the default family (or a specific theme) the Web Terminal server-renders on first paint, independent of the CLI's own `cli.theme`. See the "Theming" how-to for authoring a new theme or family.
- A new **`apex`** theme family — a warm, gold-forward skin with softer slate dark surfaces and an Instrument Serif / IBM Plex Sans typographic pairing — ships alongside `osprey` and `high-contrast`, selectable from the theme switcher. The product default theme is now pinned explicitly via `$extensions.default`, so adding a theme whose filename sorts ahead of the default can no longer change which theme the interfaces boot into.
- **Web Terminal UI modes** — a new `web.ui_mode` key under `config.yml`'s `web:` section chooses the interface density the terminal server-renders on first paint: `expert` (default) shows the full operator shell; `simple` shows a pared-down shell for lighter-weight use. An operator can override per session with a `?mode=expert|simple` URL parameter or the in-app header toggle (remembered across reloads); an unknown value falls back to `expert`. Every panel follows the mode live — Workspace, ARIEL, Channels, Lattice, Knowledge, the Events dashboard, and the Bluesky scan panels each ship a simple variant (one primary surface, plain-language cards, expert-only chrome hidden) alongside their unchanged expert view.
- **Simple-mode operator chat** — in Simple UI mode the Web Terminal's terminal card becomes a minimal chat: you type a prompt and the OSPREY agent's reply streams back, with a one-line activity indicator while it works. Conversations are multi-turn for the life of the page (a reload starts a fresh one; chat history is not persisted); Expert mode keeps the full interactive terminal. Three `web` keys bound the chat pool — `chat_turn_timeout_s` (600), `chat_idle_timeout_s` (1800), and `chat_max_sessions` (5). See the "Operate" how-to.
- **Rearrangeable Web Terminal workspace** — in Expert mode the fixed panel/terminal split is now a docking workspace of tiles, one panel per tile. The icon rail is the workspace's tab system: clicking a panel switches the focused tile to it (the replaced panel dims on the rail, one click from coming back), clicking a panel that is already open jumps to its tile, and the "+" menu opens a panel in a new tile beside the active one. Drag any tile (or the terminal card) into side-by-side splits; drops that would stack panels as tabs inside one tile are rejected. Your arrangement is saved per project and restored on reload, and "Reset layout" returns to the default. Simple mode stays a fixed, locked layout with a single panel tile, and agent-driven panel changes still apply in either mode.
- **Panel authoring standard** — a panel is a directory bundling a themed, token-only HTML entry point plus a `manifest.json`. Author one from the reference panel, then check it against the panel validator (`assert_valid_panel`), which verifies the manifest schema, the pre-paint theme boot and token stylesheet, and that no raw hex colors bypass the design tokens. The new `creating-an-osprey-panel` skill (`osprey skills install creating-an-osprey-panel`) is the guided path, and the "Panels" how-to documents the contract.
- **Local panel discovery** — drop a compliant panel bundle under `<project>/panels/` and, with `web.allow_runtime_panels: true` (off by default), the Web Terminal discovers it on startup and serves it same-origin at `/panel-static/<id>/`. Discovery is fail-closed: a malformed or non-compliant bundle is skipped and logged, never served, and never affects the other panels. See the "Panels" how-to. Note: the Web Terminal has no application-level authentication yet — enabling this trusts the panels made available to the terminal; first-class auth is a tracked follow-up.
- Dev/CI-only front-end JavaScript toolchain — `npm run typecheck` (`tsc --noEmit`) and `npm run test:js` (Vitest), enforced by a CI job; JS files opt into type-checking with a `// @ts-check` comment. Not needed to install or run OSPREY.
- Dev/CI-only Python-Playwright browser-test foundation under `tests/interfaces/` — a shared server/browser conftest plus an `assert_page_loads_clean` helper and a per-interface "loads clean in a real browser" smoke over all six web interfaces (`-m browser`), wired into the existing theming CI job. Skips cleanly when Chromium is absent; not needed to install or run OSPREY.
- Dev-only contact-sheet renderer (`python -m docs.screenshots.contact_sheet --out DIR`) — boots the real Web Terminal in every theme × UI-mode variant against a pre-seeded demo workspace (no live agent, provider, or hardware), then every supported panel standalone in the same 2×2 matrix, and composes one self-contained `contact-sheet.html` for reviewing a redesign at a glance; `--accents` renders each hub variant under both accent candidates for an A/B decision. A capture/review tool only — nothing it produces is committed or CI-gated. See the contributing guide.
- **Native Phoebus control panels** — an optional `phoebus` MCP server lets the agent perceive a running [Phoebus](https://control-system-studio.readthedocs.io/) panel's widget tree, snapshot widgets, and drive controls (driving is approval-gated, like any hardware write). Off by default; enable with `claude_code.servers.phoebus.enabled: true` and configure the bridge and named panels via the `phoebus.*` config keys (see the build-deploy config schema). The Phoebus agent bridge itself is a facility build, not part of OSPREY.
- **KNOWLEDGE web panel** — a read-only browser panel over a facility-knowledge (OKF) bundle: concept tree, markdown reader, substring search, and a bundle-health summary, served as the `KNOWLEDGE` tab in the Web Terminal (the `okf` builtin panel). Reads the bundle configured at `facility_knowledge.bundle_path`.
- **Multi-turn agent sessions** — `agent_session(...)` holds one agent conversation open across several turns so a caller can decide each message from the agent's previous reply, with per-turn and cumulative cost tracking and a session-wide budget; `run_turns(...)` is a convenience for a fixed prompt sequence. The single-turn `osprey query` path (`run_query`) is unchanged and now shares the same provider-routing and stream-parsing code.
- `osprey web` runs a fast, offline pre-flight check before binding and aborts with a consolidated error if a companion panel port is already in use, a proxy provider's auth secret is missing from the environment or `.env`, or `config.yml`/`.claude/settings.json` fails to parse — catching launch-time misconfiguration up front instead of as a silent wrong-state or a runtime error. Pass `--skip-preflight` to bypass.
- **Configurable health checks** — a `health:` block in `config.yml` lets a facility extend `osprey health` beyond the built-in checks. Declare probe-based checks — HTTP endpoints, MCP servers, deployed containers, control-system channel reads, and model-provider canaries — grouped into named categories with per-check timeouts and `requires:` dependencies, register facility health plugins in code, and tune the suite's timeouts. Checks are classed `poll` (cheap, run by default) or `on_demand` (costly, run only with `--full`). See the "Configure Health Checks" how-to.
- **`SYSTEM` health dashboard panel** — panel-shipping Web Terminal builds gain a read-only `SYSTEM` tab showing the health suite's poll-class results (status ring, per-category cards, per-check LEDs) in the browser, refreshed on the `interval_s` cadence. The dashboard never runs `on_demand` checks — those appear as informational cards with a copyable `osprey health --full --category <name>` hint. Hosting is configured under `health.web.{host,port,auto_launch}` and the heading under `health.title`; enable the tab by adding `system-health` to a build's `web_panels`. The tab's LED indicates sidecar liveness (up/down), not aggregate check status. Config and `.env` edits are picked up on the next refresh; a `control_system` change after the first channel read is surfaced as a restart notice rather than applied live. See the "Configure Health Checks" how-to.
- **Multi-user web terminals from a declarative `users[]`** — the multi-user web-terminal stack (per-user containers, an auto-populated grouped landing page, and nginx routing) is now generated deterministically from a single `users[]` list in `facility-config.yml` by `osprey scaffold web-terminals render`, with `osprey scaffold web-terminals lint` as a consistency and port-overlap gate for CI. This generator is the sole path that produces those artifacts, so editing `users[]` and re-rendering keeps the compose overlay, routing, and landing page in sync with no hand-editing. Each per-user terminal surfaces its user in the header and, when a landing origin is configured, offers a logout control that returns to the landing page and reconnects the user's still-warm session on return (single-user `osprey web` is unchanged — the control is absent). A per-user context overlay — `CLAUDE.md` (from a shared base plus a per-user `extra.md`) and a project-scope `skills/` tree — is seeded into each user's volume at deploy time, tracked so deploy never clobbers skills a user installed themselves. Not authenticated and not TLS-terminated: intended for perimeter-trusted networks only, with first-class auth, TLS, and single-origin routing as tracked follow-ups.
- **Web-terminal personas** — `modules.web_terminals.personas.<name>` lets a facility give different web-terminal users their own container image and their own rendered OSPREY project (and, since permissions live in that project's own `config.yml`, their own real per-tool permissions for free); `users[]` entries gain an optional `persona:` key, with `default_persona` as the fallback. Omitting `personas:` entirely is unchanged, zero-migration behavior.
- **Registry-optional web-terminal deploys** — `modules.web_terminals.image_source: local` builds each referenced persona's image directly from a locally rendered project directory instead of pulling from a registry, for facilities without a CI/registry pipeline.
- `modules.web_terminals.mcp.topology` schema key for the framework MCP server tier; `per_container_stdio` (today's behavior) is the only accepted value, and `shared_http` is rejected fail-closed (lint error and render error) pending future wiring.
- **Two-persona multi-user demo** — a new `multi-user-demo` preset family ships the multi-user web-terminal stack out of the box: a read-only and a read-write persona (`multi-user-demo-readonly` / `multi-user-demo-readwrite`) differing on exactly one config key, `control_system.writes_enabled`, plus an alice/bob roster. The demo is deliberately scan-free (no Bluesky bridge, no Virtual Accelerator) so it demonstrates multi-user provisioning and the write boundary, nothing else; the full scan stack stays with `control-assistant`. `osprey deploy up` (local mode) auto-renders any referenced persona project that hasn't been built yet, so one `osprey build` + `osprey deploy up` brings up the whole stack with no per-persona builds. A new multi-user demo walkthrough documents the flow.
- `exclude:` build-profile key — a profile that `extends:` another can subtract inherited entries from the string-list fields (`skills`, `rules`, `hooks`, `agents`, `output_styles`, `web_panels`, `dependencies`). A deeper extends layer re-adding an entry wins; override files and `--set` merge before exclusion and cannot re-add. Documented in the build-profiles how-to.
- `deploy_services: false` build-profile key marks an **attached project** — one that connects to services deployed by another OSPREY project on the same host instead of scaffolding its own (service sections still parse and validate, but the built project gets `deployed_services: []` and no `services/` tree; `osprey deploy up` in it is a clean no-op). The persona presets (`multi-user-demo-readonly`/`-readwrite`) build attached, consuming the demo stack's shared services instead of carrying deployable copies whose host ports collide with it.
- `osprey deploy up` now runs a host-port preflight before touching any container: every port the deploy would publish is probed, and one already held by another stack or process aborts the deploy naming the holder and the config key to change — instead of failing mid-`up` with a bare `address already in use`. Ports held by the project's own containers are exempt, so redeploys are unaffected.
- Web-terminals lint: a persona `project_path` that doesn't exist yet but is auto-renderable is now an info instead of an error; new errors catch a persona `project` not matching `basename(project_path)` and an empty `facility.prefix` on user-serving configs (both previously surfaced only as `deploy up` failures).
- The grouped landing page shows each user's persona as a badge on their card, and a pixel-diff visual test now covers the multi-user landing.
- `facility-config.yml`'s `gitlab:` block is renamed to a provider-tagged `ci: {provider: "gitlab", ...}` block, mirroring the `llm.provider` pattern; the old `gitlab:` shape still works via a one-time deprecation warning, and `registry.token_env_var` can now be set independently of the CI token instead of always riding the same PAT.
- The multi-user web-terminal compose overlay now declares each container's published port via `OSPREY_TERMINAL_WEB_PORT`, authoritative over `--port`/config, mirroring the existing `OSPREY_TERMINAL_BIND_HOST` declaration.
- `osprey deploy decommission`/`prune`/`nuke` now discover a deployment's containers, volumes, and (for local-mode personas) images by compose-project label instead of name-prefix matching, so two OSPREY deployments on the same host no longer risk cross-matching each other's resources.
- `osprey deploy up` now runs `scripts/verify.sh` automatically after bringing the web-terminal stack up (streamed, advisory — its exit code doesn't fail the deploy); a missing script is a silent skip.
- `modules.shared_disk`'s host path is now checked before compose runs, so a missing mount path aborts the deploy with an actionable error instead of failing inside a container later.

- `CITATION.cff`, enabling GitHub's "Cite this repository" button.
- `SECURITY.md` documenting private vulnerability reporting, plus a bug-report issue template.
- `NOTICE`, carrying the Berkeley Lab endorsement clause, Enhancements grant, and U.S. Government rights notice that previously sat inside `LICENSE.txt`. The licensing terms are unchanged; `LICENSE.txt` is now the unmodified BSD 3-Clause text, so automated tooling identifies the license correctly.
- The channel-finder benchmark suite gains near-miss *discrimination* queries — pairs that separate the correct channel from a plausible-but-wrong neighbour — and a stratified end-to-end evaluation slice. Development/CI only; not needed to install or run OSPREY.

### Changed

- `osprey deploy --dev` now fails with a clear error when the local osprey wheel
  cannot be built, instead of warning and deploying the pinned PyPI release.
  Previously a missing `build` package (or a broken local checkout) produced one
  warning among many info lines and an exit code of 0, so the containers came up
  running released osprey and the deployment silently tested something other than
  the local code. The preconditions — editable install, source checkout, `build`
  package — are now checked before any deploy work begins, and `build` moved from
  the `dev` extra to a base dependency so an editable install always has it.
- `osprey build` now prints a provider-credentials summary that reports the API
  keys it *found*, not just the ones it didn't. It leads with the provider the
  project was built for, names where that key came from (project `.env`, the
  build directory's `.env`, or the shell), and warns if the selected provider's
  key is missing. Previously the build logged one line per *unresolved*
  placeholder — twice — so a successful key was silent, and a missing key for
  the selected provider looked identical to the irrelevant misses for providers
  the project never uses. The per-placeholder resolver line moved to `DEBUG`.
- Scaffolded `.env` / `.env.example` files derive their provider API-key list
  from the provider registry instead of a hand-maintained list (which had
  drifted: `ALS_APG_API_KEY` was missing, a stale Langfuse block remained, and
  a detected `ARGO_API_KEY` value was discarded in favor of a `$${USER}`
  placeholder).
- Shipped preset configs now document `deployment.bind_address` and point the
  Virtual Accelerator instructions at `osprey deploy up` instead of a
  repo-internal container path.

- The model-benchmark matrix now scores two lanes separately: `agentic_benchmark`
  marks genuine model-capability e2e tests (the headline pass rate) and
  `harness_benchmark` marks model-independent safety/plumbing assertions, so
  harness passes no longer pad a model's capability score. Every in-scope e2e
  test must declare its lane (gated per matrix cell and in CI); 19 non-LLM e2e
  files moved to the matrix exclusion list. The `e2e_benchmark` marker was
  renamed to `channel_finder_benchmark` to say what it actually covers.

- The ARIEL panel no longer shows the logbook Search tab when embedded in the web terminal — search there goes through the agent, so the embedded panel offers Browse, New Entry, and Status and opens on Browse. Standalone ARIEL keeps Search as the default view.
- `osprey build` now records a project's dependencies in a generated `pyproject.toml` instead of `requirements.txt`. This makes `uv run osprey web` (and any other command) resolve the project's own `.venv` rather than walking up to an ancestor project's environment, and makes `uv sync` rebuild the environment instead of pruning it empty. Existing projects can delete their now-unused `requirements.txt` on the next `osprey build --force`.
- `osprey deploy up` now runs the web-terminal preflight (persona auto-render and the fail-closed `.env.production` credential gate) *before* building any image, so a deploy doomed to abort on a missing provider secret says so in seconds instead of after the full image build. When the missing variable is exported in the caller's shell but absent from `.env`, the error now says so and names the exact copy-in command (`.env` remains the only secret source the generator reads).
- The `osprey-build-interview` skill now asks the installed framework what it offers instead of carrying its own catalog: presets, build artifacts, providers, and config keys are all read from the live installation at interview time, so a newly shipped capability is offered without anyone editing the skill. It generates the profile with `osprey build --emit-profile` rather than a hand-written YAML template, and builds that profile itself before handing it over — what you receive is known to build. The interview now adapts its questions to the person rather than following a fixed script, and takes about five minutes. Legacy-project migration, the feedback prompts, and the web-panel design step were removed; panel authoring belongs to the `creating-an-osprey-panel` skill.
- New `osprey.build` package holds the build-time kernel shared across layers (Claude Code model/provider resolution, telemetry env block, channel-finder tier defaults, manifest primitives); agent-runtime helpers (clean child-env, SDK system-prompt, artifact-path resolution, Claude Code project-path encoding) moved to `osprey.agent_runner`. This removes the `services`/`mcp_server` → `cli`/`interfaces` layering inversions; internal import paths changed with no compatibility shims.
- Removed the legacy facility-config `gitlab:` block: a config that still carries it now fails closed with a `ConfigurationError` naming the `ci: {provider: "gitlab", ...}` replacement, instead of being silently aliased.
- Bluesky panels app moved from services to interfaces (import path `osprey.interfaces.bluesky_panels`).
- python-executor: removed the deprecated `epics_writes_enabled` field; `control_system_writes_enabled` is now the single write-gating flag.
- The seven LiteLLM-thin provider adapters (anthropic, als-apg, amsc-i2, cborg, google, openai, stanford) now share a single data-driven delegating base; behavior is unchanged (Stanford keeps its base-URL fallback).
- The three channel-finder MCP servers now share one bootstrap module (config load, path resolution, `python -m` entry point, startup sequence); behavior and entry points are unchanged.
- `osprey build --force` now re-renders an existing project *in place* instead of deleting the directory: `.env` (existing values win over freshly detected ones, and keys only it carries are kept), `_agent_data/`, and the project's `.git` are preserved; everything framework-rendered, including `data/`, is rebuilt. A profile-provided `env.file` is likewise merged into an existing `.env` rather than overwriting it.

- **`osprey health` now separates cheap poll-class checks from costly on_demand checks**, run by default and only with `--full` respectively. Three behaviors change as a result:
  - Bare `osprey health` no longer performs live model-chat completions — the `model_chat` category is now `on_demand` and runs only under `--full`.
  - Pinned-CLI verification (the `npx @anthropic-ai/claude-code@<pin>` download) moved to the `on_demand` `claude_cli_pinned` category and likewise requires `--full`; a bare run keeps only the cheap `claude --version` availability check.
  - A host with no container runtime installed now reports container checks as `skip` (exit 0) instead of `warning` (exit 1) — an absent runtime is no longer graded as a problem.
- **Bluesky scan tool surface renamed to a draft-first vocabulary.** The agent's plan-draft and run tools were renamed — `get_plan_draft`/`set_plan_draft`/`clear_plan_draft` → `get_draft`/`set_draft`/`clear_draft`, `run_status` → `get_run`, `read_run_data` → `get_run_data` — and `launch_run` now takes a required `draft_revision` (its `run_id` argument is gone); `create_run_intent` is removed, since the agent now stages one complete draft and launches that pinned revision. The scan arming token was renamed with no fallback to the old names: `BLUESKY_PROMOTE_TOKEN` → `BLUESKY_LAUNCH_TOKEN`, the `X-Promote-Token` header → `X-Launch-Token`, and the `bluesky.promote_token` config key → `bluesky.launch_token`. **Existing deployments must re-render and redeploy after upgrading** — the renamed tools appear in rendered `.claude/settings.json` allowlists, and the old token names are no longer honored.
- The shipped agent is now **instructed to answer verify-first** — to lead with real, tool-sourced data (naming the source) and flag anything not tool-backed plainly and up front, rather than a confident pretrained lead trailed by an optional "…I can verify if you want." Substantive multi-tool answers close with an explicit provenance summary (sources + a confidence/scope note); trivial reads stay terse. The doctrine ships in the `control-operator` output-style and the generated `CLAUDE.md` personas; a deployment that has `osprey scaffold claim`ed either must unclaim + regen (or merge by hand) to adopt it. See the "Facility Rules" how-to.
- `osprey build` now bundles the compose template for every service *declared* under `services:`, not only those in `deployed_services`. An opt-in deploy add-on can therefore be enabled later by adding it to `deployed_services` and running `osprey deploy up`, with no rebuild; a bundled-but-not-deployed template is inert until deployed.
- Every `osprey deploy` compose invocation (`up`, `down`, `restart`, `rebuild`, `clean`) pins `COMPOSE_PROJECT_NAME` to the resolved project name, so each deployment on a shared host owns its own compose project and volume namespace and never adopts, recreates, or removes another deployment's containers or volumes.
- `osprey deploy up --dev` and `osprey deploy rebuild` build images in a dedicated `compose build` step before `up --no-build`, so a build and its container-create never share one invocation — which can fail with `No such image` under Docker's containerd image store. A non-dev `up` still builds a build-only service implicitly.
- The Virtual Accelerator image now builds for the host's native architecture instead of pinning `linux/amd64`. On Apple Silicon it compiles `accelerator-toolbox`/`softioc` from source at build time (a slower first build) and then runs natively with no x86 emulation; on x86_64 it installs the prebuilt wheels as before. A single-arch `amd64` published image on an arm64 host must supply its own `platform` override.
- **The event-dispatch worker now runs the full project image** instead of a lean image that rebuilt its `.claude` artifacts from `config.yml` at startup. Dispatched agents now see the same facility overlays (custom skills, agents, and rules) and `data/` files as the Web Terminal agent, by construction — previously overlays and `data/` were silently absent from dispatched runs. `osprey deploy up` builds the project image (`<project>:local`; `--dev` installs the locally built wheel) and the worker references it via `OSPREY_WORKER_IMAGE`. **Requires rebuilding the dispatch worker image on redeploy.**
- **Dev image rebuilds are now incremental, and locally built service images are project-prefixed.** Each service Dockerfile now splits third-party dependencies (a cached layer) from the locally built OSPREY wheel (a fast layer), so a code-only change rebuilds in seconds and an unchanged deploy rebuilds nothing. Locally built images are now named per project — `<project>-dispatch:local`, `<project>-va:local`, `<project>-bluesky-bridge:local`, `<project>-bluesky-panels:local` — so multiple OSPREY projects on one host no longer overwrite or delete each other's images; every one carries a `com.osprey.project` label, and `osprey deploy clean` now targets the correct compose project. The old host-global images can be removed with `docker rmi osprey-dispatch:local osprey-va:local osprey-bluesky-bridge:local osprey-bluesky-panels:local`. **Migration:** an already-rendered project needs `build/` added to its `.dockerignore` — add it by hand to keep any custom Dockerfile edits, or re-render with `osprey build --force` (which overwrites hand edits). A project name that previously ended in `_` or `-` is now normalized without the trailing separator, changing its compose project and image names; run `osprey deploy down` on the old version before upgrading such a deployment.
- Refreshed the architecture diagram shown on the documentation landing page and the Architecture Overview page to match the current system design.
- `claude-agent-sdk` upgraded to 0.2.110 (bundles CLI 2.1.191); `uv.lock` regenerated (#311).
- `fastmcp` floor raised to `>=3.4.4` (brings FastAPI 0.139 / Starlette 1.x); `uv.lock` regenerated. Route-registration checks now read the app's OpenAPI schema rather than `router.routes`, which Starlette 1.0 no longer flattens for included routers.
- Dependency floors raised — `bokeh`, `gspread`, `watchdog`, `questionary`, `pillow`, `openai`, `scipy`, `bluesky`, `sphinx`, `docker`, `duckdb`, `idna`, `nltk`, `ollama`, `testcontainers`, `tiled`, `urllib3`, `uvicorn`; `uv.lock` regenerated to match.
- The Claude Code launch environment now builds its model-tier variables from a single declaration shared by the launch, e2e-override, and scrub paths, so adding a model tier can no longer leave those lists out of sync (#357). The full project-`.env` passthrough into the agent environment — which feeds `.mcp.json` `${VAR}` references such as `EPICS_CA_ADDR_LIST` — is now explicit and test-covered, and proxy providers carry their raw API-key variable through the launch path as well.
- README rewritten: corrected the connector claim (EPICS and Mock ship in-tree; other stacks use the connector interface), fixed the `osprey skills install` quickstart command, and removed stale release and conference notices. The PyPI package description now matches the documentation.
- **The Web Terminal and Artifacts interfaces have been visually redesigned.** A flat card idiom replaces the prior CRT/terminal look, over a new neutral-gray canvas with an azure accent, and the horizontal panel tab strip is now a vertical icon rail (its show/close and add-panel affordances move onto the rail). The panel content now sits directly beside the rail that selects it, with the terminal in the right-hand column (the divider still resizes the split), and the header's documentation shortcut is now a `Docs` link in the status bar. Panel behavior and APIs are unchanged.
- The Web Terminal and ARIEL settings drawers now share one accessible `<osprey-drawer>` component (focus trap and restore, `Escape`/backdrop close, screen-reader dialog semantics, inert background); each interface keeps its own look, and the Web Terminal drawer's tabs, resizing, and unsaved-changes guard behave as before.
- The Web Terminal's first-run theme default changed from forced-dark to `auto`; use the in-app theme toggle if you want a fixed theme regardless of OS preference.
- All web interface factories now share one app-setup helper for CORS, middleware, and static mounts; the Lattice dashboard picks up the standardized CORS policy and two request middlewares it was previously missing.
- The ES-module browser interfaces now import shared front-end helpers (`el`, `escapeHtml`, `debounce`) from a single `dom.js` module instead of per-file copies.
- The Artifacts gallery, the Web Terminal's Scaffold gallery, and the Lattice dashboard are now ES modules broken into focused files instead of single monolithic scripts; behavior is unchanged.
- The Lattice dashboard's accent color changed from a cyan-blue hue to OSPREY's canonical teal accent, matching every other interface.
- The event dispatcher dashboard now themes itself from the shared OSPREY design system instead of a hardcoded dark palette, and follows the web-terminal hub's theme when embedded as the EVENTS panel. **Requires a dispatcher container redeploy** to pick up the new `/design-system/*` asset route.
- Every interface's light/dark toggle is now the shared `<osprey-theme-switcher>` custom element (previously per-interface markup); it's consistently hidden when a panel is embedded in the Web Terminal hub (the hub owns theme chrome there) and visible when opened standalone.
- JSON artifacts in the Artifacts gallery now render through a syntax-highlighted, collapsible inline viewer instead of a plain read-only iframe of the raw file.
- Artifacts gallery: timeseries table values now use magnitude-adaptive precision (≤5 significant figures, scientific notation for extremes) and a compact month/day + HH:MM:SS index format.
- The channel-finder tier databases are now generated for all three paradigms (`in_context`, `hierarchical`, `middle_layer`) from a single `FacilitySpec` by one generator, and drift-gate tests fail if a committed database no longer matches that spec. Tier 1 now ships the `in_context` paradigm only, and the valid build tiers are `1` and `3`.
- ARIEL seed and scenario logbooks now name devices by their flat control-system identifiers (for example `QF08`, `CAVITY01`, `VALVE05`), matching the channel namespace, with a naming guard that keeps them from drifting back to the older hierarchical forms.

### Removed

- Retired the Bluesky scan stack's HEALTH panel and the sidecar's `/health/full` rollup. Service status for the whole deployment is shown by the SYSTEM tab; `web_panels: [health]` entries should be removed from project configs.
- Retired the Tuning optimization panel and its companion web server. It is no longer a built-in panel, and `web_panels: [tuning]` entries should be removed from project configs.
- Dropped the unused `basePath` iframe query parameter from the Web Terminal.
- Retired the tier-2 channel databases and their benchmark query set; build profiles can no longer select tier 2.
### Fixed

- `osprey web --project <dir>` launched from outside the project now behaves the
  same as running `osprey web` inside it. Previously the flag only set the
  terminal's working directory, so the project's `.env` was never loaded
  (leaving `${VAR}` placeholders such as a provider `api_key` unexpanded), the
  project's `web_terminal` and `claude_code` settings were replaced by built-in
  defaults, and `_agent_data/` was created next to wherever the command was run.
- ARIEL logbook ingestion no longer skips an otherwise-valid entry when the source
  payload omits its `id`: the ALS and generic adapters now fall back to an empty
  entry id (matching the JLab/ORNL adapters) instead of raising a `KeyError` the
  fetch loop caught and dropped the entry on.
- All artifact stores are now rooted at the shared data root, so artifacts saved
  from session-scoped writers (e.g. resumed web-terminal sessions) stay visible to
  the gallery.
- `artifact_focus`/`artifact_pin` now report gallery failures honestly instead of
  always claiming success.
- `web.app_name` in `config.yml` now actually labels the web terminal header: the
  runtime read the key from a nested section nothing generates, so only the
  `OSPREY_WEB_APP_NAME` env override worked. It now reads top-level `web.app_name`,
  matching `web.theme` and `web.presets` (env override still wins).
- A server-configured `web.theme` family now survives a visitor's first page load:
  the in-browser theme runtime adopted the default family on first visit instead of
  the configured one. Light/dark still follows the OS until the visitor picks a mode.
- How-to documentation refreshed against the current code: provider model IDs,
  deploy/build semantics (`--force` preservation, `--dev` image builds, full
  subcommand list), telemetry now documented as on-by-default, MCP/executor error
  contracts, and the ARIEL web-interface module tables.

- `osprey ariel purge` now clears the text-embedding migration record along with the dropped embedding tables, so a subsequent `osprey ariel migrate` actually recreates them instead of silently no-opping.
- Containerized Python execution no longer misclassifies an infrastructure failure during result collection as a code error: pre-classified executor errors keep their retry category, so an infrastructure fault re-executes the same code instead of triggering code regeneration.
- The Stanford provider's health-check model id had a typo (`gpt-4.omini` → `gpt-4o-mini`), also fixed in its available-models list.
- The AskSage provider now falls back to its static default model list when a `/models` fetch fails or credentials are missing, instead of returning a malformed value that could reach a UI caller. The fetched list is also cached across adapter instances, so an AskSage completion no longer pays a repeat `/models` round-trip on every request.
- Every companion web panel now gets its own per-user port family in multi-user deployments. The family set is derived from the web-server registry — previously it was a hand-maintained list that missed the channel-finder and OKF panels, so a second user's container collided with the first on the panel's fixed port (crash-looping the container once the `osprey web` preflight landed). Families omitted from config fall back to registry defaults (`channel_finder_base_port` 9591, `okf_base_port` 9691), so existing configs deploy unchanged.
- Bluesky PLAN/RESULTS/HEALTH panels now resolve their API endpoints correctly under the multi-user `/u/<user>/` mount: the shared `panelApiPrefix()` helper accepts an outer proxy prefix, the health panel no longer relies on the proxy's content rewrite (which double-prefixes once the runtime prefix is correct — this also fixes the proxied single-user health panel), and a guard test keeps panel bundles free of literals colliding with the proxy rewrite list.
- Panel tabs without a configured health endpoint (e.g. PLAN and RESULTS) now show a green LED instead of a permanently red one — the tab-state painter runs for panels that skip health polling.
- The multi-user landing page is served only at `/`; any other path outside a `/u/<user>/` mount now returns 404 instead of silently answering with landing-page HTML.
- `osprey deploy up` hot-reloads nginx after reconciling the web-terminal stack, so re-rendered `nginx.conf` routing changes take effect without a manual container restart.

- `osprey deploy up` is now idempotent from any prior state: it first removes the project's own stale non-running containers (a container left in `created` state by an aborted deploy holds its published host ports on Docker Desktop, blocking the next `up` with "address already in use"), and the plain services path reconciles away containers of services removed from the config. Running containers, volumes, and sibling deployments on the same host are untouched.
- `osprey deploy rebuild` on a web-terminals project now brings the web-terminal stack (nginx, per-user containers) back up after the clean; previously it restarted only the backend services. Per-user volumes survive a rebuild.
- Local-mode `.env.production` generation now includes the auth secret for every `claude_code.provider` in play — the deploy config's own and each referenced persona project's. A persona whose secret is missing from `.env` aborts the deploy naming the exact variable, and an existing `.env.production` that contains none of the configured LLM credentials draws a warning; previously the file could silently omit the credential entirely, producing healthy-looking web terminals that fail authentication on their first prompt.
- `osprey deploy up` probes the web-terminal landing page from the host after bringing the stack up and warns when it is unreachable. On Docker Desktop (macOS/Windows), `network_mode: host` binds inside the Docker VM unless the opt-in host-networking setting is enabled — previously this state was reported as a fully successful deploy; the warning now names that setting.

- Generated Dockerfiles (project/persona, virtual accelerator, Bluesky bridge, event dispatcher) switch Debian apt mirrors to HTTPS and set bounded apt retries, so image builds survive networks that throttle or drop plain-HTTP bulk transfers.
- Web-terminal seeding now chowns each user's `CLAUDE.md` and skills to the container's actual runtime user (queried per container) instead of a hardcoded `dispatch` user, which the persona images don't define.
- The `control-assistant` preset now ships `docker/web-terminal-context/base.md` and sets `deploy.fqdn`, so `osprey deploy up` completes without hand-added config (both were hard requirements that aborted the deploy).
- Persona auto-rendering (and the interactive deploy menu) now re-enter the CLI via the running interpreter (`python -m osprey`, newly supported) instead of whatever `osprey` is first on `PATH`, which could resolve to a different install with different presets.
- The scaffolded GitLab CI template no longer writes CI/registry/sidecar tokens into `.env.production`, aligning the CI path with the local-mode environment allowlist.
- `resolve_project_name` now normalizes to a valid docker-compose project name (lowercase, invalid characters replaced, valid leading character) at its single source, so `COMPOSE_PROJECT_NAME`, labels, image tags, and volume names all agree for mixed-case or spaced project roots.
- Every persona container now mounts its agent data at `/app/<persona project>` — the default persona was previously pinned to a facility-derived path that could diverge from its image's `WORKDIR`.
- Web Terminal logout: the button announces its in-flight state to assistive technology and carries a persona-aware label; a misconfigured landing URL leaves the button usable.
- A Web Terminal panel closed with its "×" no longer reappears on its own. A panel's backend coming back online was treated as a reason to bring the panel to the front, so a hidden panel could surface itself — most visibly when the Workspace panel was unavailable and a closed panel was the only healthy one left. Health checks now only ever affect a panel's enabled state; showing a panel remains something only you or the agent can ask for.
- A stale `CLAUDE_CODE_USE_BEDROCK` (or other backend/model selector) in the operator's shell no longer reroutes the agent away from the configured provider — these are now scrubbed before launch (#356).
- `env.example` no longer ships uncommented placeholder proxy values. Copying it to `.env` exported an unparseable `HTTP_PROXY`, which broke Claude Code launches and OSPREY's own HTTP clients.
- A proxy env var that Claude Code cannot parse (e.g. `HTTP_PROXY=http-proxy` left in `.env` or the shell) is now reported with a clear warning before every agent launch — CLI, Web Terminal, and dispatch worker — instead of surfacing only as an opaque startup crash (#352).
- The `InContextBackend` benchmark test (the sole real-LLM test outside `tests/e2e/`) moved into `tests/e2e/`, so the fast lane (`pytest tests/ --ignore=tests/e2e`) stays hermetic even when a provider key is exported — previously it made a live LLM call and failed with an unrelated gateway auth error on a blocked/expired key. A placement guard prevents recurrence for the whole credentialed `requires_*` marker family.
- Web companion servers (Artifacts, ARIEL, Channel Finder, Lattice, KNOWLEDGE) no longer skip their launch when a stale or foreign process briefly answers `/health` on their port during a restart, which previously left the panel unbacked and permanently 502ing. The launcher now decides ownership by whether a live TCP listener holds the port, waits out a shutting-down predecessor, and logs distinctly when it defers to a legitimate external server versus when the port is held by an unresponsive one (#327).
- The session activity page ("Activity" / session log viewer) no longer applies its `?theme=` query parameter directly to the page; an invalid or unexpected value now falls back to the resolved default instead of being written straight into the page's theme attribute.
- Web Terminal's and Artifacts' syntax-highlighting theme stylesheet no longer 404s in the default CDN vendor mode (it was previously served from a hardcoded local vendor path regardless of the configured vendor mode).
- ARIEL now themes itself from the shared OSPREY design system instead of a hardcoded dark palette, and follows the web-terminal hub's theme when embedded; also fixes a phantom `--amber` CSS variable that was never defined (draft banner and image-lightbox link color).
- Anthropic-native providers configured with a `/v1` base URL (e.g. Argo via `api_protocol: anthropic`) no longer resolve to a doubled `…/v1/v1/messages`: the Claude-Code-facing `ANTHROPIC_BASE_URL` is stripped of a trailing `/v1` (Claude Code appends `/v1/messages` itself), while the translation-proxy upstream keeps its `/v1`. All four launch paths (CLI, web terminal, SDK runner, dispatch worker) now start the proxy from the resolved `upstream_base_url` field rather than the stripped env var, so OpenAI-compatible providers still forward to `…/v1/chat/completions` (#312).
- Headless dispatch runs now enforce the trigger's `allowed_tools` as the single authority via a PreToolUse hook: project `settings.json` allow-rules and the approval hook's explicit allows can no longer widen a run's tool surface, and declared subagents (`.claude/agents/*.md`) work with exactly their declared tools — no trigger changes needed.
- `osprey web --project X` launched from another directory now spawns the interactive terminal's OSPREY agent with `cwd = X`, so it reads `X/.mcp.json` and starts the project's MCP servers (the PTY path previously ignored `--project` and inherited the launch directory) (#313).
- Channel Finder's embedded mode no longer hides its whole header: only the logo is hidden now, so the pipeline switcher and navigation stay visible and usable when embedded in the Web Terminal hub (previously the entire header — including those controls — was hidden, with no compensating layout change).
- Toggling the theme (in the Web Terminal hub or any standalone interface) no longer leaves a stale `?theme=` in the URL; reloading after a toggle now falls back to your saved preference (or the OS setting in `auto` mode) instead of being pinned to whatever value was in the URL at toggle time.
- Creating a new artifact from the Web Terminal's Scaffold gallery no longer fails with an HTTP 405 — the create-artifact request now uses `POST` directly instead of being routed through a GET-only fetch helper.
- Artifacts gallery: filter chips no longer accumulate click listeners across live-refresh cycles, and a failed Plotly script load is retried on the next chart render instead of failing for the rest of the page's lifetime.
- Deployed service containers are now named `<project>-<service>` (e.g. `<project>-ariel-postgres`, `<project>-virtual-accelerator`, `<project>-bluesky-bridge`) instead of host-global names, so two OSPREY projects can deploy the same services on one host concurrently without colliding on a container name. In-network service discovery is unaffected (it uses the compose service key, not the container name); the `ariel-postgres` DSN hostname is preserved via a network alias.

### Security

- Provider isolation now holds against a personal `~/.claude/settings.json`. Claude Code loads settings-file `env` blocks on top of the process environment, so a model or base-URL value in an operator's global (or gitignored local) settings would silently override the project's configured provider. The `osprey claude chat` and Web Terminal launch paths now start Claude Code with `--setting-sources project`, matching the SDK launch paths, so only the project's own `.claude/settings.json` applies. Additionally, `osprey claude chat`, the Web Terminal, and the dispatch worker now refuse to launch when enterprise **managed-policy** settings set a provider variable OSPREY manages — the one scope that outranks the project — naming the variable and file rather than starting against an unconfigured backend. **Behavior change for interactive users:** because the user and local settings scopes are no longer loaded when launching the agent, personal `~/.claude/settings.json` customizations (status line, output style, hooks) and persisted "always allow" permission grants in `.claude/settings.local.json` no longer apply inside `osprey claude chat` / the Web Terminal; put project-wide settings in the project's `.claude/settings.json` instead.
- Artifacts gallery: agent-supplied artifact metadata (`category`, `artifact_type`, type-registry labels) is now HTML-escaped at every render sink, the shared `escapeHtml` escapes quotes for attribute contexts, and artifact ids are percent-encoded in URL paths — closing a stored-XSS in the gallery sidebar.

## [2026.6.3] - 2026-06-29

### Fixed

- The `events` web panel now builds and renders when its URL is derived from a `dispatch` block. Two regressions from the v2026.6.2 move of `events` out of `BUILTIN_PANELS` are addressed: (1) the build-profile validator now accepts a dispatch-backed `events` panel that has no manual `web.panels.events.url` (the URL is derived post-build, after validation, from `dispatch.dispatcher_port`); (2) the derivation emits a bare-host `url` plus a `/dashboard` `path` instead of baking `/dashboard` into `url`, matching the custom-panel proxy convention (`url.rstrip('/') + '/' + path`) so sub-routes are not double-prefixed. A facility-pinned `web.panels.events.path` is preserved, and an explicit `web.panels.events.url` override still wins.

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
