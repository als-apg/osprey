"""The gold-standard three-zone deployment repo, materialized for tests.

This module is the hand-authored reference deployment (``als-exemplar/``) that
the lifecycle verbs are developed against, and the shape ``osprey init`` must
emit. Exemplar-first: the content below was authored by hand, derived from the
``control-assistant`` preset emission and rewritten onto the new command
surface; ``init`` is then made to reproduce it, never the reverse.

The layout is the three-zone repo — one directory, four kinds of content::

    als-exemplar/
    │  ═ SOURCE — tracked, user-edited ═══════════════
    ├── profile.yml  triggers.yml  README.md
    ├── data/  personas/  web-terminal-context/
    ├── .gitignore  .env.example  ci-extra.yml
    ├── .gitlab-ci.yml  scripts/verify.sh   (with_ci=True only)
    │  ═ SECRETS — ignored, durable ══════════════════
    ├── .env                       (only with ``seed_env=True``)
    │  ═ OUTPUT — ignored, disposable ════════════════
    ├── build/                     (never materialized here: a build makes it)
    │  ═ STATE — ignored, durable ════════════════════
    └── var/agent_data/  var/audit/

``build/`` is deliberately absent. It is 100% derived, so the source repo the
exemplar models is the repo *before* any build has run — which is also the
fresh-clone state (SC-10). A test that needs a rendered build stubs one itself.

Two variants, and which one a caller wants is not a matter of taste:

``with_ci=False`` (the default) is the **init-reproducible** shape. A bare
``osprey init --preset control-assistant`` leaves the ``deploy:`` block a
commented stub, and with no deployment coordinates there is nothing to render a
CI pipeline from — so the repo carries neither ``.gitlab-ci.yml`` nor
``scripts/verify.sh``. This is the shape Task 2.1 compares ``init``'s emission
against, byte for byte.

``with_ci=True`` fills the coordinates in and adds the pipeline pair — the
shape ``osprey scaffold ci`` produces, and what Tasks 2.5 and 2.8 compare
against. ``image_source`` moves with the block: with deployment coordinates it
lives in the ``deploy:`` block and the ``config:`` block must not repeat it,
which the profile parser enforces.

Everything the profile says apart from the redesign's own concerns is the
bundled preset's content, written out verbatim from the standalone emission —
the full artifact lists, the Bluesky/virtual-accelerator stack, every
``config:`` key. The exemplar exists to pin the *form* ``init`` emits; the
content belongs to the preset, and a byte-for-byte gate is only satisfiable
that way. What the redesign does change: the three-zone header, comments moved
onto the new verb surface, the persona catalog pointed at ``build/`` and at
``personas/*.yml``, and the deploy block above.

Two values cannot be frozen into the text: the installed OSPREY version and the
bundled presets' content hashes, both of which the real emission stamps into the
provenance header. They are written as ``@OSPREY_VERSION@`` and
``@PRESET_HASH:<preset>@`` sentinels and expanded at materialization, so a
byte-comparison against a live ``osprey init`` stays meaningful as the repo
moves. Sentinels rather than ``str.format``/``%`` because the YAML carries
literal ``${VAR:-default}`` shell expansions.

Usage::

    def test_something(lifecycle_repo):          # als-exemplar/ in tmp_path
        assert (lifecycle_repo / "profile.yml").is_file()

    def test_two_checkouts(lifecycle_repo_factory, tmp_path):
        a = lifecycle_repo_factory(tmp_path / "a")
        b = lifecycle_repo_factory(tmp_path / "b", seed_env=True)
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

#: Directory name of the exemplar deployment. One repo is one deployment and the
#: directory name is the deployment name, so this is also the compose project
#: name a test should expect.
EXEMPLAR_DIRNAME = "als-exemplar"

#: The preset the exemplar was materialized from, and the two persona presets
#: whose deltas sit in ``personas/``.
EXEMPLAR_PRESET = "control-assistant"
PERSONA_PRESETS: Mapping[str, str] = {
    "readonly": "control-assistant-readonly",
    "readwrite": "control-assistant-readwrite",
}

#: Durable state zone, created empty. A build recreates these when absent
#: (FR-2), so a fresh clone and a reset repo look identical here.
STATE_DIRS: tuple[str, ...] = ("var/agent_data", "var/audit")

#: Paths the shipped ``.gitignore`` must keep out of git, anchored to the repo
#: root. Unanchored spellings are the foot-gun this list exists to pin: they
#: also swallow a same-named path anywhere deeper in the tree, silently.
IGNORED_ZONE_PATTERNS: tuple[str, ...] = ("/build/", "/var/", "/.env*")

#: Source files that must be executable once written.
EXECUTABLE_FILES: frozenset[str] = frozenset({"scripts/verify.sh"})


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE zone — profile.yml
# ─────────────────────────────────────────────────────────────────────────────

PROFILE_YML = """\
# Als Exemplar — OSPREY deployment repo
#
# The repository is the deployment. One directory, four zones:
#
#   SOURCE   tracked, yours to edit — profile.yml, data/, personas/,
#            triggers.yml, web-terminal-context/, scripts/, the CI files
#   SECRETS  .env — git-ignored, durable: provider keys you set, plus the
#            service tokens `osprey up` mints
#   OUTPUT   build/ — git-ignored, 100% disposable. Every `osprey build`
#            wipes and re-renders it; `rm -rf build/` loses nothing, ever
#   STATE    var/ — git-ignored, durable: var/agent_data holds the agent's
#            memory and sessions, var/audit the audit log. No build touches it
#
#   +--------------+  osprey   +--------------+  osprey  +----------------+
#   |    SOURCE    |  build    |    build/    |    up    |   DEPLOYMENT   |
#   | profile.yml  +---------->| config.yml   +--------->| agent CLI/web  |
#   | data/  ...   |           | .mcp.json  … |          | + containers   |
#   +--------------+           +--------------+          +----------------+
#          ^                                                     |
#          +---- edit -> osprey build -> osprey up --------------+
#
# `osprey up` starts strictly from build/ as it was built — it never renders
# from this file. Edit profile.yml and `up` refuses until you re-run
# `osprey build`, so a half-finished edit can never reach a running stack.
#
# Emitted from the bundled `control-assistant` preset as a fully explicit,
# standalone profile: everything the preset configures is written out below and
# is yours to edit. Nothing is inherited at build time.
#
# Provenance — what this profile was materialized from:
#   source preset: control-assistant
#   preset content hash: @PRESET_HASH:control-assistant@
#   emitted by OSPREY @OSPREY_VERSION@

name: Als Exemplar

# Which packaged project skeleton `osprey build` renders (config.yml, README,
# Dockerfile). "control_assistant" is the full skeleton. Its data files
# (channel databases, benchmarks, facility knowledge, logbook seeds) are copied
# into this repo's data/ directory, and that copy — not the packaged one — is
# what the build uses from then on.
app_template: control_assistant

# Default LLM provider and model. Override here or with `osprey set
# provider=...` / `osprey set model=...`, which write this file in place,
# comments intact. The persona terminals build from deltas over that same
# profile, so they pick the change up from the file rather than from a replayed
# command line.
provider: anthropic
model: haiku   # tier (haiku/sonnet/opus), or a model ID the provider serves

# Channel-finder pipeline. Override with
# `osprey set channel_finder_mode=in_context|middle_layer`.
channel_finder_mode: hierarchical

# Extra packages the built project's environment needs. pymongo is what the
# archiver connector, the deploy-time seeder and the recorder all read the store
# through; the tutorial declares a `va_archiver:` block below, so without it the
# staged bring-up aborts at its pymongo preflight before any image is built.
# Matches the pin the `archiver-mongodb` extra carries in pyproject.toml.
dependencies:
  - pymongo>=4.0

# ── Artifact selection ───────────────────────────────────────────────────────
# Each list entry is a short name resolved from the osprey artifact library.
# Remove entries you do not need; add facility-specific artifacts via overlay.

hooks:
  - hook-log          # Append every tool call to a structured JSONL audit log
  - hook-config       # Inject project config.yml path into every tool call env
  - approval          # Gate hardware-write tool calls on human approval prompt
  - writes-check      # Pre-write safety check: confirm channel is writable
  - limits            # Enforce per-channel min/max limits before writes
  - error-guidance    # Post-error hook that surfaces remediation hints
  - memory-guard      # Warn when context window approaches threshold
  - notebook-update   # Sync CLAUDE.md notebook after each session
  - cf-feedback-capture  # Capture channel-finder accuracy feedback for tuning
  - config-drift      # Warn at session start when the build drifted from source
  - focus-validate    # Strip stale artifact IDs from focus_state.txt on each prompt
  - panels-context    # Inject the web terminal panel inventory into agent context
  - workspace-delta   # Report web workspace changes since the agent's last turn

rules:
  - safety              # Core safety rules: never write without approval
  - error-handling      # Standard error handling and retry guidance
  - artifacts           # Rules for saving diagnostic and tuning artifacts
  - facility            # Facility-level conventions (naming, units, logbook)
  - workflows           # Approved workflow patterns (scan, ramp, restore)
  - timezone            # Always localise timestamps to facility timezone
  - python-execution    # Rules governing Python executor sandbox usage
  - data-visualization  # Rules for producing control-room-ready plots
  - control-system-safety  # EPICS PV safety: alarm limits, soft-IOC guards
  - test-ioc-safety     # Test-IOC port isolation (renders only for EPICS-family control systems)

skills:
  - diagnose        # Run a structured fault-diagnosis workflow
  # setup-mode (config diagnostics + setup_patch) is deliberately NOT part of
  # the operator tier: it can patch config.yml/.mcp.json, which is admin work.
  # Add it to an admin-facing profile's skills list to opt in — it remains in
  # the artifact catalog.
  - session-report  # Summarise session actions and outcomes to the logbook
  - demo-gallery    # Launch guided capability demonstrations
  - demo-ui         # Run a scripted demo of the agent driving the web workspace
  - writing-bluesky-plans  # Author, validate, and queue a session-tier plan (requires the Bluesky MCP server)
  - operating-bluesky-scans  # Stage, queue, and watch a registered scan through the shared draft (requires the Bluesky MCP server)

agents:
  - channel-finder          # Semantic search over channel databases (hierarchical)
  - data-visualizer         # Produce strip charts and correlation plots
  - logbook-search          # Search facility logbook for historical entries
  - logbook-deep-research   # Multi-hop logbook research with synthesis
  - facility-knowledge      # Look up facility documentation, procedures, and device specs
  - pyat-specialist         # Lattice/optics computation sub-agent (pyAT)

output_styles:
  - control-operator  # Terse, actionable output style for control-room operators

web_panels:
  - ariel           # ARIEL search interface (past experiments, papers)
  - channel-finder  # Interactive channel-finder web UI
  - okf             # KNOWLEDGE tab — browse the facility knowledge bundle
  - system-health   # SYSTEM tab — framework health dashboard (sidecar-backed)
  # events + bluesky (the write-oriented panels) are declared by the readwrite
  # persona, beside the web.panels.<id>.url overrides that give them meaning —
  # a panel id and its URL declaration travel together (see the note in
  # config: below).

# ── Bluesky stack (turn-key, VA-backed) ─────────────────────────────────────────
# The preset ships the FULL Bluesky-mediated scan stack out of the box: a
# Bluesky bridge (+ co-deployed Tiled catalog), a PyAT Virtual Accelerator
# soft-IOC standing in for real EPICS hardware, and the bluesky-panels sidecar
# that serves the BLUESKY web_panel above. Each top-level block below is an
# injector trigger — it renders the corresponding `services.<name>` compose
# service, appends to `deployed_services`, and (for bluesky_panels) registers
# the `web.panels.<id>` URLs. Remove this section (and the bluesky panel id
# above) to ship a Bluesky-free deployment.
bluesky:
  port: 8090
  tiled_enabled: true      # co-deploys the Tiled catalog sidecar alongside the bridge
  tiled_port: 8091

virtual_accelerator:
  # Channel Access port the soft-IOC serves on. The connector's gateway port
  # follows this value, so changing it here moves both the container and the
  # connector that talks to it.
  port: 5064

bluesky_panels:
  port: 8095

# ── Stored archive (MongoDB) ─────────────────────────────────────────────────
# Declaring this block is what gives the deployment a REAL archive: `osprey up`
# stands up a MongoDB store and an archiver-recorder beside the VA, seeds the
# store with history on the first deploy, and records the machine from then on.
# Without it the agent's history is synthesized at read time — plausible numbers
# for questions nobody recorded the answer to.
#
# The block is the single home for the archive's coordinates: the build derives
# the connector's eight `archiver.mongodb_archiver.*` keys AND the shape knobs
# below from it, and REFUSES a profile that also spells them in `config:` (one
# fact, two homes, free to disagree). Selecting the archiver is a separate
# decision and lives in `config:` as `archiver.type` — see the note there.
#
# Every key is optional; these are the shipped defaults, stated so the tutorial
# documents the shape it deploys rather than hiding it in a dataclass.
va_archiver:
  # Where the agent reaches the store. Stated rather than left to the deploy to
  # infer, because a profile resolved WITHOUT a service stack — a persona delta,
  # an attached project, any of the resolution paths that carry no
  # deploy_services — is required to name the host whose archive it reads, and
  # would otherwise fail to resolve at all. `localhost` is the host side of the
  # published `port_host`, which is where this project's own store answers;
  # point it at the real host when attaching to someone else's.
  host: localhost
  retention_days: 30       # how far back the archive reaches
  hot_span_hours: 48       # how much of it is kept at the dense cadence
  hot_cadence_sec: 10      # seconds between samples inside the hot span
  tail_cadence_sec: 60     # and outside it (must be a whole multiple of the above)
  # How often the recorder samples the running machine. Stated because the
  # freshness threshold below is derived from it: leaving it to the dataclass
  # default would hide the one number that decides when this deployment calls
  # its own archive stale.
  recorder_cadence_sec: 10
  # The channel `osprey health` reads to tell a REACHABLE archive from one still
  # being WRITTEN — a wedged recorder leaves the store answering queries while
  # history quietly stops, and only the age of the newest sample separates the
  # two. Naming it here derives the whole `archiver_freshness` check, threshold
  # included: that follows `recorder_cadence_sec` above, so slowing the recorder
  # cannot leave a stale threshold behind.
  #
  # The stored-beam DCCT current is this simulated facility's canary, for the
  # reason a real control room would pick it: it is the first thing to stop
  # moving when the machine does. Point it at your own equivalent when you take
  # this preset to hardware; drop the key and no check is derived.
  freshness_channel: SR:DIAG:DCCT:01:CURRENT:RB

# ── Config overrides ─────────────────────────────────────────────────────────
# Dotted keys ONLY: each entry is a literal `key.path` written into the
# rendered config.yml after the template renders. A nested mapping here would
# replace the whole rendered subtree and silently drop its siblings.
#
# This block is the source of truth for configuration. build/config.yml is
# generated from it and is never hand-edited — `osprey set` writes here.
config:
  # The VA soft-IOC ships and is deployed as part of the turn-key Bluesky stack
  # above, so control_system.type defaults to "virtual_accelerator": the
  # preset's scan plans drive it end to end out of the box (correctors move,
  # BPMs read, a settle-verified run COMPLETEs). Use "epics" for live hardware.
  # "mock" is the documented fallback for environments with no containers to
  # depend on — its readbacks are a non-tracking simulation, so scans are
  # browse-only (a settle-verified run never COMPLETEs on mock) — flip back
  # with `osprey set connector=mock`, the shorthand that writes this key.
  control_system.type: virtual_accelerator
  # Selects the archive declared by the `va_archiver:` block above as the
  # deployment's archiver. This is a SEPARATE decision from declaring where the
  # archive lives, and deliberately so: the block never flips `archiver.type`
  # out from under a facility that set it. A project that declared the block but
  # left this alone would deploy a store and then read the mock beside it.
  #
  # Dotted, like every key here. A nested `archiver:` mapping would prefix
  # -collapse; and spelling any `archiver.mongodb_archiver.*` key here is
  # refused outright while the block is present, because the build already
  # derives those from it.
  archiver.type: mongodb_archiver
  # The Bluesky MCP server is off-by-default in the framework registry
  # (default_enabled=False); opt in here so the agent can author and launch
  # scan plans out of the box.
  claude_code.servers.bluesky.enabled: true
  # The system-health MCP server is likewise off-by-default in the framework
  # registry (default_enabled=False); opt in here so the agent can run
  # read-only framework/stack health checks out of the box.
  claude_code.servers.health.enabled: true
  # system.timezone: America/Los_Angeles
  # Facility display name, woven into the agent prompts (CLAUDE.md, the
  # channel-finder/logbook agents) and the web-terminal landing page. Defaults
  # to the deployment name when unset. Sits in the same `facility:` block as
  # `facility.prefix` below.
  # facility.name: My Facility
  # Default web theme for every terminal: the `main` family pinned to light
  # mode (`light` is main's concrete light id — other families spell theirs
  # `desy-light` etc.). A default, not a lock: the in-browser display menu,
  # ?theme= and localStorage override it per browser.
  web.theme: light
  # EVENTS + BLUESKY panel declarations live in the readwrite persona delta,
  # NOT here. Deliberate: a persona delta can only ADD config keys (`config:`
  # is not excludable), so anything declared here reaches every persona — and
  # the read-only persona must be built without these two write-oriented
  # panels. Declaring them only in the readwrite delta is the one mechanism
  # that makes them genuinely absent from the readonly build (`enabled: false`
  # is inert for URL panels — only builtin ids honor it). A full deployment
  # render still gets both panels: the dispatch and bluesky-panels injectors
  # fill in defaults when the profile doesn't declare them.
  # ── Multi-user web-terminal stack (built-in) ───────────────────────────────
  # This deployment is natively multi-user: `osprey up` stands up nginx, the
  # landing page, and one web-terminal container per roster user, alongside the
  # scan stack above. Single-user onboarding is unchanged — `osprey web` never
  # reads this block (only `osprey up` does), so a plain host-run terminal
  # works at any time. To deploy backend services without the web tier, set
  # `modules.web_terminals.enabled: false` here.
  #
  # Container-name prefix for the web stack: names are `<prefix>-nginx` /
  # `<prefix>-web-<user>`, so this MUST be a non-empty valid Docker name start
  # ([a-zA-Z0-9]). Keep it short and distinct from the deployment name (used for
  # image tags). Set it to your own facility abbreviation.
  facility.prefix: ca
  # Landing-URL origin for the multi-user web stack: render refuses to build
  # OSPREY_TERMINAL_LANDING_URL without deploy.fqdn once users are configured.
  # 127.0.0.1 matches the single-trusted-host posture; set your
  # browser-reachable hostname when deploying anywhere else.
  deploy.fqdn: 127.0.0.1
  # This MUST stay ONE literal dotted key — `modules.web_terminals` addressing
  # the whole module subtree with a nested value. A nested `modules:` mapping
  # under `config:` would wholesale-replace the rendered `modules` subtree,
  # silently dropping any sibling module: config_writer applies each dotted key
  # verbatim, setting only the addressed leaf (see utils/config_writer.py).
  #
  # `image_source: local` means `osprey up` builds each persona's image itself
  # from `project_path`, and `osprey build` is what put a project there:
  # one render per delta in `personas/`. So `osprey build && osprey up` stands
  # the whole stack up with no registry. Each persona's `project` equals its
  # `project_path` basename — both are derived from the repo's own directory
  # name, which is how the render and the mount land on the same path.
  #
  # The `build_profile` values below are PRESET names, and are consumed only
  # here, at materialization: `osprey init` renders each one into a delta at
  # `personas/<name>.yml` beside the emitted profile and rewrites this catalog
  # to point at that file. The build reads the deltas themselves and accepts
  # nothing else — a persona is built from a delta over this profile, never
  # from a preset of its own.
  modules.web_terminals:
    enabled: true
@WEB_TERMINALS_IMAGE_SOURCE@
    nginx_port: 9080            # public-facing reverse proxy / landing page
    # Per-user port families: user i gets base + i in each family. Every
    # companion panel gets its own family (the containers share the host
    # network namespace, so per-user offsets are what prevent collisions);
    # families omitted here fall back to registry defaults (registry/web.py).
    # All families sit above this deployment's own service ports (5064, 8020,
    # 8090/8091/8095) so the two stacks never collide on one host.
    web_base_port: 9091           # first per-user web-terminal port
    artifact_base_port: 9291      # first per-user artifact-gallery port
    ariel_base_port: 9391         # first per-user ARIEL search port
    lattice_base_port: 9491       # first per-user lattice-dashboard port
    channel_finder_base_port: 9591  # first per-user channel-finder panel port
    default_persona: readonly   # roster entries with no persona resolve here
    users:
      # One web terminal per entry. `index` pins the user's port offsets
      # (each `*_base_port` above + index), so removing an entry never shifts
      # another user's ports; `persona` names a catalog entry below. A bare
      # string (`- alice`) is also accepted: it takes its list position as
      # index and falls back to `default_persona`.
      # `display_name` becomes the browser window/tab title (OSPREY_WEB_APP_NAME)
      # — with both terminals on the same light theme it is the visible marker
      # of which one is write-armed.
      - name: alice
        index: 0
        persona: readwrite
        display_name: "Control Room (Alice)"
      - name: bob
        index: 1
        persona: readonly
        display_name: "Read-Only View (Bob)"
    personas:
      # A persona render is build output like everything else under build/.
      # `osprey build` renders one project per delta in `personas/`, and it
      # replaces build/ whole every time — nothing under there is edited in
      # place or kept across a build. `osprey up` renders nothing: if a project
      # is missing it refuses and names `osprey build`.
      readonly:
        project: als-exemplar-readonly
        project_path: build/als-exemplar-readonly
        build_profile: personas/readonly.yml
      readwrite:
        project: als-exemplar-readwrite
        project_path: build/als-exemplar-readwrite
        build_profile: personas/readwrite.yml

# ── Event dispatch (optional) ────────────────────────────────────────────────
# Turns external events (webhooks) into headless agent runs. This ships a set of
# control-system-free triggers so you can exercise the pipeline with a single
# `curl` after `osprey up`. Remove this block to disable dispatch.
dispatch:
  triggers: triggers.yml            # repo-relative path or bundled name
  worker_count: 1
  workspace_mode: isolated
  max_concurrent_runs: 2
  max_queue_depth: 50

# ── Environment variables ────────────────────────────────────────────────────
# Bearer tokens guarding the dispatcher's inbound webhook/dashboard auth
# (EVENT_DISPATCHER_TOKEN) and the dispatcher→worker calls (DISPATCH_WORKER_TOKEN).
# No defaults on purpose: the dispatch services fail closed on an unset token.
# `osprey up` mints a strong random value for each unset token into this repo's
# .env (and logs where), so a fresh deployment is secure by default with zero
# editing. Set your own values in .env to override.
env:
  required:
    - EVENT_DISPATCHER_TOKEN
    - DISPATCH_WORKER_TOKEN
data: data
# Minimum OSPREY release that understands this profile's keys. Builds below it abort.
requires_osprey_version: '>=2026.9.0'
# What this profile was materialized from. Emitted, not hand-written: a
# build compares it against the installed preset and mentions it when the
# preset has moved on. Advisory only — this profile is the source of truth.
provenance:
  preset: control-assistant
  preset_hash: @PRESET_HASH:control-assistant@
# true builds its own services stack; false attaches to another project's.
deploy_services: true
# Services this profile declares. Injected ones are added at build time.
services: {}
# Named web-terminal layouts, as label -> list of panel ids.
panel_presets: {}

# --- Channel-database tier ---------------------------------------------------
# Build-time only (1 or 3), selecting which bundled tier DB is materialized.
# Left unset the build picks a paradigm-aware default, which is why it stays
# commented: pinning it here would override that default on every rebuild.
# Tier 1 ships the in_context paradigm only.
#
# tier: 3

# --- Default web-terminal panel ----------------------------------------------
# Panel id opened when the web terminal loads. Must be a built-in, an entry in
# the web_panels list above, or a custom panel backed by a web.panels.<id>.url
# config override.
#
# default_panel: artifacts
@DEPLOY_BLOCK@
# --- Facility MCP servers ----------------------------------------------------
# Your own MCP servers, injected into the build's .mcp.json next to the
# framework ones. An entry is either stdio (command/args/env) or remote (url,
# or just port to derive http://localhost:<port>/mcp). transport defaults to
# "http"; "sse" is the legacy event-stream wire and needs an explicit url.
# Tool names under permissions are bare — `allow` runs them unprompted, `ask`
# prompts the operator on every call.
#
# mcp_servers:
#   matlab:
#     command: /opt/matlab/bin/mcp-matlab
#     args: [--workspace, /opt/matlab/scripts]
#     env:
#       MATLAB_LICENSE: "${MATLAB_LICENSE}"
#     permissions:
#       allow: [run_script]
#   lattice:
#     url: http://lattice.example.org:8400/mcp
#     port: 8400
#     transport: http
#     permissions:
#       allow: [get_twiss]

# --- Artifacts gallery: custom categories ------------------------------------
# Extra buckets in the artifacts gallery. Each key is the id a facility MCP
# tool passes as category="<key>" when it saves an artifact; label and color
# (#RRGGBB) decide how the gallery renders that bucket. The artifact_server
# block also accepts host/port/auto_launch overrides for the gallery server.
#
# artifact_server:
#   categories:
#     optics:
#       label: Optics
#       color: "#4C9AFF"

# --- Nextcloud bridge --------------------------------------------------------
# Answers questions asked from a Nextcloud Talk room. The trigger name must
# match one declared in the dispatch triggers file.
#
# nextcloud_bridge:
#   trigger: nextcloud-question

# --- Google Chat bridge ------------------------------------------------------
# Answers questions asked from a Google Chat space or direct message. The
# trigger name must match one declared in the dispatch triggers file.
#
# The Google credentials and destinations are runtime env, not profile keys:
# declare GCHAT_SA_KEY, GCHAT_SUBSCRIPTION and GCHAT_APP_ID under `env.required`
# (plus GCS_BUCKET / GCS_PROJECT to deliver plots and files as links).
#
# gchat_bridge:
#   trigger: gchat-question
"""

#: Deployment coordinates, filled in. Where this repo runs once it leaves the
#: laptop; the CI pipeline is rendered from it.
#:
#: ``image_source`` lives here and nowhere else. The build propagates it into
#: ``modules.web_terminals.image_source``, so a second copy under ``config:``
#: would be one fact with two homes, free to disagree about whether the deploy
#: host builds its images or pulls them — and the profile is rejected for it.
DEPLOY_BLOCK_ACTIVE = """
# --- Deployment coordinates --------------------------------------------------
# Where this deployment is built and run once it leaves the laptop;
# `osprey scaffold ci` renders the pipeline from it.
#
# Credentials are named here, never written here: declare each variable under
# `env.required` and put its value in the deploy host's .env.
deploy:
  ci: gitlab
  image_source: local   # the deploy host builds its own images; no registry
  host:
    name: appsdev2
    fqdn: appsdev2.example.org
    user: operator
    project_path: /home/operator/deployments/als-exemplar

"""

#: The same block as the bundled preset leaves it — commented out, no
#: coordinates, no CI pipeline to render from. Without a deploy block nothing
#: propagates ``image_source``, so the ``config:`` block states it instead.
DEPLOY_BLOCK_COMMENTED = """
# --- Deployment coordinates --------------------------------------------------
# Where this deployment is built, pushed, and run. Needed only once it leaves
# the laptop; `osprey scaffold ci` renders the pipeline from it.
#
# Credentials are named here, never written here: declare each variable under
# `env.required` and put its value in the deploy host's .env.
#
# deploy:
#   ci: gitlab
#   registry:
#     url: git.example.org:5050/physics/production/facility-profiles
#     token_env_var: FACILITY_REGISTRY_TOKEN
#   host:
#     name: appsdev2
#     fqdn: appsdev2.example.org
#     user: operator
#     project_path: /home/operator/projects/facility-profiles

"""

#: The ``config:`` line the commented-out deploy block leaves the profile
#: needing, and the filled-in one forbids. Substituted into the profile text at
#: the ``@WEB_TERMINALS_IMAGE_SOURCE@`` marker.
WEB_TERMINALS_IMAGE_SOURCE_LINE = (
    "    # No deploy block yet, so this is image_source's only home: build the\n"
    "    # per-user terminal images here rather than pulling them.\n"
    "    image_source: local"
)


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE zone — persona deltas
# ─────────────────────────────────────────────────────────────────────────────

PERSONA_READONLY_YML = """\
# Als Exemplar (readonly) — persona profile, a delta over ../profile.yml
#
# Sitting in personas/ beside profile.yml is the inheritance: the build merges
# this file over that profile — including any edit you make there — so the keys
# below are this persona's only differences and there is no `extends:` to
# write. See the resolved whole with:
#   osprey validate personas/readonly.yml
#
# Provenance — what this persona was materialized from:
#   source preset: control-assistant-readonly
#   preset content hash: @PRESET_HASH:control-assistant-readonly@
#   emitted by OSPREY @OSPREY_VERSION@

name: Als Exemplar (readonly)

# Attached render: this persona builds per-user terminal images only and
# connects to the shared web tier the hosting deployment runs on the same host.
# No services are scaffolded — the injector blocks inherited from the base are
# all gated on this flag and skip cleanly.
deploy_services: false

# ── Config overrides ─────────────────────────────────────────────────────────
# Dotted keys ONLY — see the base profile's block.
config:
  # The single axis this persona hard-pins: this key is the tier boundary, so
  # it must not drift if the base's default ever changes — it is what makes
  # the read-only terminal read-only.
  control_system.writes_enabled: false
  # Pared-down operator layout: chat only, workspace hidden until the agent
  # puts something in it. Pinned on both sides of the tier boundary (readwrite
  # pins `expert`), same rationale as writes_enabled.
  #
  # This persona also has no EVENTS/BLUESKY panels — not by any key here, but
  # because their declarations live in the readwrite persona delta and never
  # reach this build (see the note in the base's config: block).
  web.ui_mode: simple
  # The hosting deployment owns the web-terminal tier (nginx, landing,
  # per-user containers). Without this override the inherited roster would
  # make this render try to host a second web tier on the same host ports.
  modules.web_terminals.enabled: false
"""

PERSONA_READWRITE_YML = """\
# Als Exemplar (readwrite) — persona profile, a delta over ../profile.yml
#
# Sitting in personas/ beside profile.yml is the inheritance: the build merges
# this file over that profile — including any edit you make there — so the keys
# below are this persona's only differences and there is no `extends:` to
# write. See the resolved whole with:
#   osprey validate personas/readwrite.yml
#
# Provenance — what this persona was materialized from:
#   source preset: control-assistant-readwrite
#   preset content hash: @PRESET_HASH:control-assistant-readwrite@
#   emitted by OSPREY @OSPREY_VERSION@

name: Als Exemplar (readwrite)

# Attached render: this persona builds per-user terminal images only and
# connects to the shared web tier the hosting deployment runs on the same host.
deploy_services: false

# The write-oriented panels, listed beside their web.panels.<id>.url overrides
# below (a panel id and its URL declaration travel together). Persona lists
# UNION over the base, so these are added to the inherited builtin set.
web_panels:
  - events          # EVENTS dashboard tab (event dispatcher)
  - bluesky         # Plan authoring, the scan queue, and the run's live results

# ── Config overrides ─────────────────────────────────────────────────────────
# Dotted keys ONLY — see the base profile's block.
config:
  # The single axis this persona hard-pins: this key is the tier boundary, so
  # it must not drift silently if the base's default ever changes.
  control_system.writes_enabled: true
  # Full split-pane terminal + workspace layout for the write-armed operator.
  # Pinned on both sides of the tier boundary (readonly pins `simple`) rather
  # than left to the server default, for the same reason writes_enabled is.
  web.ui_mode: expert
  # The hosting deployment owns the web-terminal tier (nginx, landing,
  # per-user containers). Without this override the inherited roster would
  # make this render try to host a second web tier on the same host ports.
  modules.web_terminals.enabled: false
  # EVENTS + BLUESKY: the write-oriented panels, declared HERE and not in the
  # base so the readonly persona is built without them (a persona can only add
  # config keys, never subtract inherited ones — see the note in the base's
  # config: block).
  # EVENTS — the event-dispatcher dashboard as an in-terminal tab. URL defaults
  # to the host-run dispatcher; override EVENT_DISPATCHER_URL for
  # containerized/remote web terminals.
  web.panels.events.label: EVENTS
  web.panels.events.url: "${EVENT_DISPATCHER_URL:-http://localhost:8020}"
  web.panels.events.path: /dashboard
  web.panels.events.health_endpoint: /health
  # BLUESKY — operator UI for the mediated Bluesky stack, served by the
  # bluesky-panels sidecar. Override BLUESKY_PANELS_URL for containerized/
  # remote web terminals (same pattern as EVENTS above).
  web.panels.bluesky.label: BLUESKY
  web.panels.bluesky.url: "${BLUESKY_PANELS_URL:-http://localhost:8095}"
  web.panels.bluesky.path: /bluesky/
"""


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE zone — dispatch triggers
# ─────────────────────────────────────────────────────────────────────────────

TRIGGERS_YML = """\
# triggers.yml
#
# Four control-system-free demonstration triggers shipped with the
# control-assistant preset. Each illustrates one event-dispatch concept so a
# new user can exercise the pipeline end-to-end without any facility hardware:
#
#   1. hello-dispatch    — anatomy of a trigger + first successful round-trip
#   2. triage-event      — a webhook payload becomes the agent's context
#   3. save-report       — tool use, a short multi-turn loop, and persistence
#   4. denied-tool-demo  — the worker's server-side tool denylist (safety)
#
# Fire one with (`osprey up` mints EVENT_DISPATCHER_TOKEN into this repo's
# .env; load it first: export $(grep -E '^EVENT_DISPATCHER_TOKEN=' .env | xargs)):
#   curl -X POST http://localhost:8020/webhook/hello-dispatch \\
#     -H "Authorization: Bearer $EVENT_DISPATCHER_TOKEN" \\
#     -H "Content-Type: application/json" -d '{}'
#
# Watch progress stream in the dashboard at http://localhost:8020/dashboard
#
# (Retries fire on *dispatch failure* — i.e. when the dispatcher cannot reach
# the worker — via the per-trigger `on_error: retry` policy. That path is not
# exercised by a curl against a healthy stack; see the docs and the unit test
# tests/unit/dispatch/test_server_routes.py for the retry/backoff behaviour.)

dispatcher:
  # The dispatcher forwards each fired trigger to this worker. The compose
  # template names the single worker "dispatch-worker-1" on port 9190.
  # (Multi-worker load distribution is not yet implemented — see docs.)
  dispatch_target: http://dispatch-worker-1:9190
  max_concurrent_runs: 2
  max_queue_depth: 50

triggers:
  # 1. Anatomy + minimal end-to-end check: webhook in, one sentence out, no tools.
  - name: hello-dispatch
    source: webhook
    action:
      prompt: >-
        Reply with a single friendly sentence confirming the event-dispatch
        pipeline is working end to end. Do not use any tools.
      allowed_tools: []

  # 2. The webhook JSON body arrives as the agent's context. Zero tools keeps
  #    this cheap and focused on the payload lesson. Try it with a realistic
  #    event body, e.g.:
  #      curl -X POST http://localhost:8020/webhook/triage-event \\
  #        -H "Authorization: Bearer $EVENT_DISPATCHER_TOKEN" \\
  #        -H "Content-Type: application/json" \\
  #        -d '{"signal":"demo:vacuum:pressure","value":4.2,"threshold":3.0}'
  - name: triage-event
    source: webhook
    action:
      prompt: >-
        An automated monitor fired this event and handed you its JSON payload as
        context. In plain language: summarize what the event reports, say whether
        it looks normal or concerning given any threshold in the payload, and
        outline what you would investigate first. Do not use any tools — reason
        only from the payload.
      allowed_tools: []

  # 3. Tool use + a short multi-turn loop + persistence via the workspace MCP
  #    artifact tool. Artifacts land in the worker's mounted workspace volume,
  #    so they survive the run. This is the sanctioned persistence channel: the
  #    preset's memory guard intentionally blocks arbitrary file writes, so the
  #    agent persists through the artifact tool.
  - name: save-report
    source: webhook
    action:
      prompt: >-
        Investigate this event and save a short status report. First take a
        quick look at the working directory (Glob/Read) to ground yourself, then
        use the workspace artifact tool to save a concise markdown report
        summarizing the event payload and what you would do next. Confirm the
        artifact you created.
      allowed_tools:
        - Glob
        - Read
        - mcp__osprey_workspace__artifact_save
        - mcp__osprey_workspace__create_document

  # 4. Requests a tool the worker blocks server-side; teaches the denylist.
  - name: denied-tool-demo
    source: webhook
    action:
      prompt: >-
        Attempt to fetch https://example.com with WebFetch and report what
        happens. WebFetch is on the worker's server-side denylist, so the run is
        rejected regardless of the tools this trigger requests — demonstrating
        that the denylist is enforced independently of the trigger config.
      allowed_tools: [WebFetch]
"""


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE zone — git, secrets, README
# ─────────────────────────────────────────────────────────────────────────────

GITIGNORE = """\
# This repo is the deployment: the source zone is tracked, and the three
# generated or secret zones below never are. A fresh deployment has a clean
# `git status` from birth.

# OUTPUT — rendered by `osprey build` from the source zone. Regenerable in
# full, so it is never committed.
/build/

# STATE — the agent's memory, sessions, and audit log. Durable, host-local,
# and nobody else's business.
/var/

# The source zone `osprey init --force` is replacing, while the new one
# renders. A successful run removes it; one that is killed outright leaves it,
# and the next `osprey init` puts its contents back. Never committed either
# way — for the seconds it exists it is a second copy of files already tracked.
/.osprey-replaced-source-zone/

# SECRETS — provider keys you set plus the tokens `osprey up` mints, and the
# lock file the write-back path creates beside them. .env.example carries no
# values and is the single exception.
#
# Every zone entry above is anchored to the repo root with a leading slash. An
# unanchored `build/` or `.env*` would also swallow a same-named path anywhere
# deeper in the tree — including files moved there later — and it would do it
# silently.
/.env*
!/.env.example

# OS / editor noise. Deliberately unanchored: these are junk at any depth.
.DS_Store
*.swp
*.swo
"""

ENV_EXAMPLE = """\
# Als Exemplar Environment Configuration
#
# The single documented list of every variable this agent reads. Copy it to
# `.env` beside this file and fill in the values you need. That one file is the
# deployment's whole secret store: `osprey build` never reads or rewrites it,
# and the containers mount it from the repo root, so a value here survives
# every rebuild and every `rm -rf build/`.
#
# This file carries no secrets and is safe to commit.

# API Keys — set the key(s) for the provider(s) your profile uses.
# (Provider list derived from the OSPREY provider registry.)
ANTHROPIC_API_KEY=your-anthropic-api-key-here
OPENAI_API_KEY=your-openai-api-key-here
GOOGLE_API_KEY=your-google-api-key-here
CBORG_API_KEY=your-cborg-api-key-here
AMSC_I2_API_KEY=your-amsc-i2-api-key-here
ARGO_API_KEY=your-argo-api-key-here
STANFORD_API_KEY=your-stanford-api-key-here
ALS_APG_API_KEY=your-als-apg-api-key-here

# Required by this profile — the profile's `env.required` names these, and
# the agent cannot run without them.
EVENT_DISPATCHER_TOKEN=
DISPATCH_WORKER_TOKEN=

# Runtime overrides (optional - for advanced use cases)
#LOCAL_PYTHON_VENV=/path/to/your/venv/bin/python

# Optional: Proxy settings (uncomment if behind corporate firewall)
# NO_PROXY=localhost,127.0.0.1
# HTTP_PROXY=http://proxy.example.com:8080
# HTTPS_PROXY=http://proxy.example.com:8080

# Service credentials — minted automatically by `osprey up` when the matching
# service is deployed, and appended to this repo's .env. Set one by hand only
# to pin a value the deployment must not replace (an existing database volume's
# password, say); an unset variable is minted, never guessed.
# EVENT_DISPATCHER_TOKEN=  # event_dispatcher, dispatch_worker — authenticates callers to the event-dispatcher API
# DISPATCH_WORKER_TOKEN=  # event_dispatcher, dispatch_worker — authenticates the dispatch worker back to the dispatcher
# BLUESKY_LAUNCH_TOKEN=  # bluesky — arms the Bluesky bridge's scan-launch endpoint
# BLUESKY_TILED_API_KEY=  # bluesky — the key the bridge presents to the co-deployed Tiled catalog
# ZO_ROOT_USER_PASSWORD=  # openobserve — OpenObserve root/ingest credential
# ARIEL_DB_PASSWORD=  # postgresql — ARIEL Postgres password (also fills the agent's derived DSN)
# MONGO_ROOT_PASSWORD=  # mongodb — archiver store root password (the seeder, recorder and agent all authenticate with it)
"""

#: Written only when the factory is asked for a seeded repo. Values are the
#: obviously-fake shape a test wants: present, well-formed, never a real key.
ENV_SEEDED = """\
ANTHROPIC_API_KEY=sk-ant-exemplar-not-a-real-key
EVENT_DISPATCHER_TOKEN=exemplar-dispatcher-token
DISPATCH_WORKER_TOKEN=exemplar-worker-token
"""

README_MD = """\
# Als Exemplar

This repository is an OSPREY deployment. Everything the assistant is made of
lives here, and the directory name is the deployment's name.

## The four zones

| Zone | Path | Tracked? | Survives? |
| --- | --- | --- | --- |
| Source | `profile.yml`, `data/`, `personas/`, `triggers.yml`, `web-terminal-context/`, `.env.example`, `.gitignore`, `README.md`, `ci-extra.yml`, `.gitlab-ci.yml`, `scripts/verify.sh` | yes | it *is* the record |
| Secrets | `.env` | no | yes — durable |
| Output | `build/` | no | no — 100% disposable |
| State | `var/agent_data/`, `var/audit/` | no | yes — durable |

`build/` is derived in full from the source zone. `rm -rf build/` loses
nothing, ever: no configuration, no keys, no agent memory. Nothing durable is
allowed to live there.

## Daily use

```bash
osprey build          # render build/ from profile.yml
osprey up -d          # start the deployment from build/, as built
osprey status         # containers, endpoints, drift, versions
osprey logs           # follow the stack's logs
osprey down           # stop it
```

Every command walks up from wherever you are to this directory, so they work
from any subdirectory with no flags. `--repo PATH` overrides that.

## Changing something

Edit `profile.yml` (or `osprey set model=sonnet` for a single key), then:

```bash
osprey build && osprey up -d
```

`osprey up` starts strictly from `build/` as it was built — it never renders
from `profile.yml`. If the source zone has moved on, `up` refuses and names
what changed, so a half-finished edit can never reach a running stack. Use
`osprey up --build` to chain the render, or `--as-built` to start the previous
build knowingly.

## Starting over

```bash
osprey reset          # containers, volumes, agent data, build/ — all gone
```

`reset` keeps `var/audit/` and your provider keys. `osprey reset --purge-audit`
destroys the audit log too; that plus `rm -rf` on this directory is a complete
uninstall.

## Backup and restore

Git covers the source zone. `var/` and `.env` are the entire durable state, so
a backup is a tarball of those two, and a restore is:

```bash
git clone <this repo> && tar xf state.tar.gz && osprey build && osprey up -d
```
"""


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE zone — CI
# ─────────────────────────────────────────────────────────────────────────────

CI_EXTRA_YML = """\
# Als Exemplar's own pipeline jobs.
#
# .gitlab-ci.yml is emitted by `osprey scaffold ci` and will be overwritten the
# next time it runs. This file never is — put anything facility-specific here:
# extra tests, an IOC smoke check, a notification hook. It is included after
# the scaffolded pipeline, so it can also override a job by redefining it under
# the same name.
#
# Example:
#
#   ioc-smoke-test:
#     stage: validate
#     image: python:3.11-slim
#     script:
#       - ./ci/ioc_smoke_test.sh

# Placeholder so the include always parses. Delete it when you add a job.
.facility-jobs-go-here: {}
"""

GITLAB_CI_YML = """\
# =============================================================================
# Als Exemplar — deployment pipeline
# =============================================================================
# osprey-scaffold: deploy/gitlab-ci
# osprey-version: @OSPREY_VERSION@
#
# Emitted by `osprey scaffold ci` from the `deploy:` block in profile.yml.
# Re-run that command after editing the block; the marker line above is what
# makes re-emission safe, so a file without it is treated as hand-written and
# left alone unless you pass --force.
#
# Facility-specific jobs go in ci-extra.yml, which this pipeline includes. That
# file is yours — the scaffolder creates it once and never writes it again.
#
# Project-level CI/CD variables this pipeline reads (Settings -> CI/CD ->
# Variables; mask and protect it):
#
#   DEPLOY_SSH_KEY            Private key (File type) for the deploy-host
#                             account in `deploy.host`. CI-only: it
#                             authenticates the deploy job and is never part
#                             of the deployment's own environment.
#
# The deploy host builds its own images (`deploy.image_source: local`), so this
# pipeline needs no registry credential at all.
#
# No secret is ever read into an artifact. The deploy host keeps its own .env —
# the facility's secrets stay where the repo says they live.
# =============================================================================

include:
  # Facility-owned jobs, layered on top of everything below. Guarded by
  # `exists` so a repo that deleted the file still has a valid pipeline.
  - local: ci-extra.yml
    rules:
      - exists:
          - ci-extra.yml

stages:
  - validate
  - deploy

variables:
  DEPLOY_HOST: appsdev2.example.org
  DEPLOY_USER: operator
  DEPLOY_PATH: /home/operator/deployments/als-exemplar

# -----------------------------------------------------------------------------
# Stage 1 — the source zone still renders.
#
# Runs on every commit and needs no credentials: it proves profile.yml is
# well-formed and that build/ can be rendered from it, which is the failure a
# facility most wants to hear about before the deploy window, not during it.
# -----------------------------------------------------------------------------
render-build:
  stage: validate
  image: python:3.11-slim
  before_script:
    # The floor the profile itself declares (requires_osprey_version), so the
    # pipeline can never run an OSPREY that does not understand it.
    - pip install --no-cache-dir "osprey-framework>=2026.9.0"
  script:
    - osprey validate
    - osprey build --skip-lifecycle --skip-deps
  artifacts:
    paths:
      - build/
    expire_in: 1 week

# -----------------------------------------------------------------------------
# Stage 2 — deploy.
#
# Manual and default-branch only: this is the single gate between a green
# pipeline and a running control-room service. `resource_group` serializes it
# so two operators pressing the button cannot interleave on the host.
#
# The host re-renders from the same commit rather than unpacking the artifact
# above, so what runs is reproducible from git alone. `git reset --hard` moves
# only what git tracks — the deployment's own .env and var/ are git-ignored by
# the repo's .gitignore, so the host's secrets and the agent's memory survive
# every deploy by construction. That is why nothing here copies files onto the
# host: an rsync of the working tree would have to be trusted not to delete
# them.
#
# `osprey users env-production` turns the host's .env into the runtime secrets
# file the web-terminal containers read, before any container starts. `--output`
# is what makes that safe, and it is not optional: without it the command writes
# the assembled secrets to stdout, which here is the job log. It also creates
# the file at mode 0600 from its first byte, which a shell redirect would not —
# on a shared deploy host the difference is every other account being able to
# read the deployment's credentials. The file lands at the repo root, where
# compose reads it, and .gitignore keeps it out of git — so the next deploy's
# `git reset --hard` leaves it alone.
#
# `osprey up` runs scripts/verify.sh afterwards on its own — the deploy's
# health report needs no job of its own.
# -----------------------------------------------------------------------------
deploy:
  stage: deploy
  image: alpine:3.20
  needs:
    - render-build
  before_script:
    - apk add --no-cache openssh-client git
    - mkdir -p ~/.ssh && chmod 700 ~/.ssh
    - cp "$DEPLOY_SSH_KEY" ~/.ssh/id_ed25519 && chmod 600 ~/.ssh/id_ed25519
    - ssh-keyscan -H "$DEPLOY_HOST" >> ~/.ssh/known_hosts
  script:
    - |
      ssh "$DEPLOY_USER@$DEPLOY_HOST" bash -euo pipefail <<REMOTE
      cd $DEPLOY_PATH
      git fetch --prune origin
      git reset --hard $CI_COMMIT_SHA
      osprey build
      osprey users env-production --output .env.production
      osprey up -d
      REMOTE
  environment:
    name: production
    url: https://$DEPLOY_HOST
  resource_group: production
  rules:
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH
      when: manual
"""

VERIFY_SH = """\
#!/usr/bin/env bash
# =============================================================================
# Als Exemplar — post-deploy health check
# =============================================================================
# osprey-scaffold: deploy/verify
# osprey-version: @OSPREY_VERSION@
#
# Emitted by `osprey scaffold ci` into the repo's scripts/ directory. `osprey
# up` runs it automatically once the containers are up; you can also run it by
# hand from anywhere in the repo:
#
#   ./scripts/verify.sh                    # every probe
#   ./scripts/verify.sh services           # one group
#
# ALWAYS exits 0. Verification is advisory: a failed probe tells an operator
# where to look, and must never be the reason a deploy is reported as failed.
#
# No `set -e`: one probe timing out must not skip the ones after it.
# =============================================================================
set -uo pipefail

GREEN=$'\\033[32m'; RED=$'\\033[31m'; DIM=$'\\033[90m'; BOLD=$'\\033[1m'; RESET=$'\\033[0m'

# Probe groups, selectable as arguments. Default is all of them. Not named
# GROUPS: bash owns that name, and assigning to it silently does nothing.
PROBE_GROUPS="${*:-services web dispatch}"

# An HTTP endpoint that answers. Used for anything speaking HTTP.
probe_http() {
  local label="$1" url="$2"
  if curl -sf --max-time 5 -o /dev/null "$url"; then
    printf '  %s✓%s %s\\n' "$GREEN" "$RESET" "$label"
  else
    printf '  %s✗%s %s — no response from %s\\n' "$RED" "$RESET" "$label" "$url"
  fi
}

# A TCP listener. The virtual accelerator serves EPICS Channel Access, not
# HTTP, so a connect is as far as a probe can go without an EPICS client.
probe_tcp() {
  local label="$1" host="$2" port="$3"
  if python3 -c "import socket,sys; s=socket.socket(); s.settimeout(3); \\
sys.exit(s.connect_ex(('$host', $port)))" 2>/dev/null; then
    printf '  %s✓%s %s\\n' "$GREEN" "$RESET" "$label"
  else
    printf '  %s✗%s %s — nothing listening on %s:%s\\n' "$RED" "$RESET" "$label" "$host" "$port"
  fi
}

wants() { case " $PROBE_GROUPS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

# ── Deployed services ────────────────────────────────────────────────────────
if wants services; then
  printf '\\n%s── Services ──%s\\n\\n' "$BOLD" "$RESET"
  probe_tcp  'virtual-accelerator: Channel Access on 5064' localhost 5064
fi

# ── Web tier ─────────────────────────────────────────────────────────────────
if wants web; then
  printf '\\n%s── Web terminal ──%s\\n\\n' "$BOLD" "$RESET"
  probe_http 'landing page'      http://localhost:9080/
  probe_http 'terminal (alice)'  http://localhost:9091/
  probe_http 'terminal (bob)'    http://localhost:9092/
fi

# ── Event dispatch ───────────────────────────────────────────────────────────
if wants dispatch; then
  printf '\\n%s── Event dispatch ──%s\\n\\n' "$BOLD" "$RESET"
  probe_http 'dispatcher health' http://localhost:8020/health
fi

printf '\\n%sProbes are advisory — a failure here does not mean the deploy failed.%s\\n\\n' \\
  "$DIM" "$RESET"
exit 0
"""


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE zone — data tree
# ─────────────────────────────────────────────────────────────────────────────
# Hand-authored and deliberately small. The packaged bundle a real `init`
# copies here is ~2 MB across ~60 files; a per-test tmp_path materialization
# wants the *shape* — every path the profile and the build reference — not the
# volume. Each file below is valid content of its real kind.

DATA_README_MD = """\
# Data

Everything the agent reads from disk lives here: channel databases, benchmark
query sets, facility knowledge, and simulation scenarios. These are your files.
They are tracked, and `osprey build` only ever reads them.

```
data/
├── raw/                                  # CSV address data (in_context path)
├── channel_databases/
│   ├── tiers/tier{1,3}/<paradigm>.json  # staged, one per paradigm
│   └── TEMPLATE_EXAMPLE.json            # database format example
├── benchmarks/cross_paradigm/queries/    # staged query sets, one per tier
├── channel_limits.json                   # per-channel write limits
├── machine_state_channels.json           # channels in the machine-state view
├── facility_knowledge/                   # markdown knowledge bundle
└── simulation/                           # mock-connector scenarios
```

The build collapses the staged sets down to the ones `channel_finder_mode` and
`tier` select, writing the result under `build/`. This directory is never
rewritten by a build.
"""

CHANNEL_DB_HIERARCHICAL_JSON = """\
{
  "_comment": "Hierarchical channel database. Unified 6-level naming: RING:SYSTEM:FAMILY:DEVICE:FIELD:SUBFIELD.",
  "hierarchy": {
    "levels": [
      { "name": "ring", "type": "tree" },
      { "name": "system", "type": "tree" },
      { "name": "family", "type": "tree" },
      { "name": "device", "type": "instances" },
      { "name": "field", "type": "tree" },
      { "name": "subfield", "type": "tree" }
    ],
    "naming_pattern": "{ring}:{system}:{family}:{device}:{field}:{subfield}"
  },
  "tree": {
    "SR": {
      "DIAG": {
        "BPM": {
          "DEVICE": {
            "_expansion": { "_type": "list", "_instances": ["01", "02"] },
            "POSITION": {
              "X": { "description": "Horizontal beam position", "units": "mm" },
              "Y": { "description": "Vertical beam position", "units": "mm" }
            }
          }
        },
        "DCCT": {
          "DEVICE": {
            "_expansion": { "_type": "list", "_instances": ["01"] },
            "CURRENT": {
              "RB": { "description": "Total stored beam current", "units": "mA" }
            }
          }
        }
      },
      "MAG": {
        "HCM": {
          "DEVICE": {
            "_expansion": { "_type": "list", "_instances": ["01", "02"] },
            "CURRENT": {
              "RB": { "description": "Horizontal corrector current readback", "units": "A" },
              "SP": { "description": "Horizontal corrector current setpoint", "units": "A" }
            }
          }
        }
      }
    }
  }
}
"""

CHANNEL_DB_IN_CONTEXT_JSON = """\
{
  "_comment": "Flat in-context channel database. One entry per address.",
  "channels": {
    "SR:DIAG:DCCT:01:CURRENT:RB": {
      "description": "Total stored beam current",
      "units": "mA"
    },
    "SR:DIAG:BPM:01:POSITION:X": {
      "description": "BPM 1 horizontal beam position",
      "units": "mm"
    },
    "SR:MAG:HCM:01:CURRENT:SP": {
      "description": "Horizontal corrector 1 current setpoint",
      "units": "A"
    }
  }
}
"""

CHANNEL_DB_TEMPLATE_EXAMPLE_JSON = """\
{
  "_comment": "Database format example. Copy this shape when hand-authoring a channel database.",
  "channels": {
    "FACILITY:SYSTEM:FAMILY:01:FIELD:RB": {
      "description": "What this channel reports, in one sentence",
      "units": "mm"
    }
  }
}
"""

BENCHMARK_QUERIES_JSON = """\
[
  {
    "user_query": "What is the stored beam current?",
    "targeted_pv": ["SR:DIAG:DCCT:01:CURRENT:RB"]
  },
  {
    "user_query": "Show me the horizontal position of the first two BPMs",
    "targeted_pv": ["SR:DIAG:BPM:01:POSITION:X", "SR:DIAG:BPM:02:POSITION:X"]
  }
]
"""

CHANNEL_LIMITS_JSON = """\
{
  "_comment": "Write limits, enforced by the limits hook before any write reaches the control system. A channel is writable if and only if it is a setpoint (:SP); every other address is read-only, whatever this file says.",
  "SR:MAG:HCM:01:CURRENT:SP": { "min": -5.0, "max": 5.0, "units": "A" },
  "SR:MAG:HCM:02:CURRENT:SP": { "min": -5.0, "max": 5.0, "units": "A" }
}
"""

MACHINE_STATE_CHANNELS_JSON = """\
{
  "_comment": "Channels shown in the machine-state view. One canonical list regardless of channel-finder mode.",
  "_version": "2.0",

  "SR:DIAG:DCCT:01:CURRENT:RB": { "label": "Beam current (DCCT)", "group": "beam" },
  "SR:DIAG:BPM:01:POSITION:X": { "label": "BPM 1 horizontal position", "group": "orbit" },
  "SR:DIAG:BPM:01:POSITION:Y": { "label": "BPM 1 vertical position", "group": "orbit" },
  "SR:MAG:HCM:01:CURRENT:RB": { "label": "Corrector 1 current", "group": "magnets" }
}
"""

RAW_ADDRESS_LIST_CSV = """\
address,description,family_name,instances,sub_channel
# === STANDALONE CHANNELS (no templating) ===
SR:DIAG:DCCT:01:CURRENT:RB,Total stored beam current in milliamps,,,
# === DEVICE FAMILIES (one row expands to one channel per instance) ===
SR:DIAG:BPM:{i}:POSITION:X,Horizontal beam position,BPM,01;02,X
SR:DIAG:BPM:{i}:POSITION:Y,Vertical beam position,BPM,01;02,Y
"""

FK_INDEX_MD = """\
---
okf_version: "0.1"
---

# Subdirectories

* [subsystems](/subsystems/) - Contains 1 entry: Vacuum System (VAC).
* [procedures](/procedures/) - Contains 1 entry: Vacuum Recovery.
"""

FK_SUBSYSTEMS_INDEX_MD = """\
---
okf_version: "0.1"
---

# Subsystems

* [vacuum](vacuum.md) - Vacuum System (VAC)
"""

FK_VACUUM_MD = """\
---
okf_version: "0.1"
title: Vacuum System (VAC)
abbreviation: VAC
---

# Vacuum System (VAC)

The vacuum system holds the storage ring at ultra-high vacuum so the stored
beam is not scattered out by residual gas.

## What it is made of

Ion pumps distributed around the ring do the continuous pumping; cold-cathode
gauges report pressure. Both are exposed as channels under `SR:VAC:`.

## Normal readings

Ring pressure sits near 1e-9 mbar with beam stored. A gauge above 1e-8 mbar is
worth investigating; above 1e-7 mbar the interlock trips the beam.
"""

FK_PROCEDURES_INDEX_MD = """\
---
okf_version: "0.1"
---

# Procedures

* [vacuum-recovery](vacuum-recovery.md) - Vacuum Recovery
"""

FK_VACUUM_RECOVERY_MD = """\
---
okf_version: "0.1"
title: Vacuum Recovery
---

# Vacuum Recovery

Restores ring pressure after a vent or a pressure excursion.

## Steps

1. Confirm the affected sector from the gauge readings under `SR:VAC:GAUGE:`.
2. Verify the sector valves either side of it are closed.
3. Watch the sector's pressure fall; it should drop an order of magnitude an
   hour once the pumps are running.
4. Open the valves only once the sector is within one order of magnitude of
   its neighbours.

## Safety

Never open a sector valve against a pressure differential. The interlock will
refuse, and forcing it risks the whole ring's vacuum.
"""

#: Each channel is a mapping carrying exactly one of ``value`` or ``expr`` --
#: the schema ``osprey.simulation.machine.parse_machine`` enforces, and the one
#: the shipped presets are written in. A bare number here would look like a
#: reasonable shorthand and is not: the parser rejects it, so the exemplar would
#: name a simulation model that no engine can load.
SIMULATION_MACHINE_JSON = """\
{
  "name": "Als Exemplar demo machine",
  "description": "Nominal machine values the mock connector serves as readbacks.",
  "channels": {
    "SR:DIAG:DCCT:01:CURRENT:RB": {
      "value": 500.0,
      "units": "mA",
      "description": "Stored beam current"
    },
    "SR:DIAG:BPM:01:POSITION:X": {
      "value": 0.02,
      "units": "mm",
      "description": "Beam position monitor 1, horizontal"
    },
    "SR:DIAG:BPM:01:POSITION:Y": {
      "value": -0.01,
      "units": "mm",
      "description": "Beam position monitor 1, vertical"
    },
    "SR:MAG:HCM:01:CURRENT:RB": {
      "value": 0.0,
      "units": "A",
      "description": "Horizontal corrector 1 current, readback"
    },
    "SR:MAG:HCM:01:CURRENT:SP": {
      "value": 0.0,
      "units": "A",
      "description": "Horizontal corrector 1 current, setpoint"
    }
  }
}
"""

SIMULATION_NOMINAL_SCENARIO_JSON = """\
{
  "description": "All systems nominal."
}
"""

SIMULATION_VACUUM_BURST_SCENARIO_JSON = """\
{
  "description": "A vacuum excursion in sector 1 costs beam lifetime; stored current falls.",
  "overrides": {
    "SR:DIAG:DCCT:01:CURRENT:RB": 380.0
  }
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# The exemplar, assembled
# ─────────────────────────────────────────────────────────────────────────────

#: Source-zone files present in every exemplar, as repo-relative posix path ->
#: text. The CI pipeline pair is not here — it is conditional on the profile
#: carrying deploy coordinates; see :data:`CI_PIPELINE_FILES`.
BASE_SOURCE_FILES: Mapping[str, str] = {
    ".gitignore": GITIGNORE,
    ".env.example": ENV_EXAMPLE,
    "README.md": README_MD,
    "ci-extra.yml": CI_EXTRA_YML,
    "triggers.yml": TRIGGERS_YML,
    "personas/readonly.yml": PERSONA_READONLY_YML,
    "personas/readwrite.yml": PERSONA_READWRITE_YML,
    "web-terminal-context/alice/.gitkeep": "",
    "web-terminal-context/bob/.gitkeep": "",
    "data/README.md": DATA_README_MD,
    "data/channel_databases/TEMPLATE_EXAMPLE.json": CHANNEL_DB_TEMPLATE_EXAMPLE_JSON,
    "data/channel_databases/tiers/tier1/in_context.json": CHANNEL_DB_IN_CONTEXT_JSON,
    "data/channel_databases/tiers/tier3/hierarchical.json": CHANNEL_DB_HIERARCHICAL_JSON,
    "data/benchmarks/cross_paradigm/queries/tier3_queries.json": BENCHMARK_QUERIES_JSON,
    "data/channel_limits.json": CHANNEL_LIMITS_JSON,
    "data/machine_state_channels.json": MACHINE_STATE_CHANNELS_JSON,
    "data/raw/address_list.csv": RAW_ADDRESS_LIST_CSV,
    "data/facility_knowledge/index.md": FK_INDEX_MD,
    "data/facility_knowledge/subsystems/index.md": FK_SUBSYSTEMS_INDEX_MD,
    "data/facility_knowledge/subsystems/vacuum.md": FK_VACUUM_MD,
    "data/facility_knowledge/procedures/index.md": FK_PROCEDURES_INDEX_MD,
    "data/facility_knowledge/procedures/vacuum-recovery.md": FK_VACUUM_RECOVERY_MD,
    "data/simulation/machine.json": SIMULATION_MACHINE_JSON,
    "data/simulation/scenarios/nominal/scenario.json": SIMULATION_NOMINAL_SCENARIO_JSON,
    "data/simulation/scenarios/vacuum-burst/scenario.json": SIMULATION_VACUUM_BURST_SCENARIO_JSON,
}

#: The scaffolded CI pipeline. Emitted only where the profile names deploy
#: coordinates — there is nothing to render a pipeline from otherwise.
CI_PIPELINE_FILES: Mapping[str, str] = {
    ".gitlab-ci.yml": GITLAB_CI_YML,
    "scripts/verify.sh": VERIFY_SH,
}

_PRESET_HASH_SENTINEL = re.compile(r"@PRESET_HASH:([a-z0-9-]+)@")
_VERSION_SENTINEL = "@OSPREY_VERSION@"
_IMAGE_SOURCE_MARKER = "@WEB_TERMINALS_IMAGE_SOURCE@"
_DEPLOY_BLOCK_MARKER = "@DEPLOY_BLOCK@"


def _osprey_version() -> str:
    """The installed OSPREY version, as the emitters stamp it."""
    from osprey import __version__

    return __version__


def _preset_hash(preset_name: str) -> str:
    """Content hash of a bundled preset, or the emitters' unavailable marker."""
    from osprey.cli.build_profile_merge import compute_preset_hash

    return compute_preset_hash(preset_name) or "(unavailable)"


def expand_sentinels(text: str) -> str:
    """Resolve ``@OSPREY_VERSION@`` and ``@PRESET_HASH:<preset>@`` in ``text``.

    The two values the real emission stamps at materialization time. Resolving
    them here rather than freezing them keeps a byte-comparison against a live
    ``osprey init`` honest across version bumps and preset edits.
    """
    text = text.replace(_VERSION_SENTINEL, _osprey_version())
    return _PRESET_HASH_SENTINEL.sub(lambda m: _preset_hash(m.group(1)), text)


def exemplar_source_files(*, with_ci: bool = False) -> dict[str, str]:
    """The exemplar's source zone as repo-relative posix path -> final text.

    Sentinels are expanded, so this is byte-for-byte what
    :func:`build_exemplar_repo` writes — the mapping a byte-comparison against
    ``osprey init`` reads from.

    Args:
        with_ci: Fill in the deploy coordinates and emit the CI pipeline they
            are rendered from. The default False is the init-reproducible
            shape — a bare ``osprey init --preset control-assistant`` has no
            coordinates to render a pipeline from, so it emits neither.
    """
    if with_ci:
        deploy_block = DEPLOY_BLOCK_ACTIVE
        # image_source has exactly one home. With a deploy block that home is
        # the deploy block, and a second copy under `config:` is rejected.
        profile = PROFILE_YML.replace(_IMAGE_SOURCE_MARKER + "\n", "")
    else:
        deploy_block = DEPLOY_BLOCK_COMMENTED
        profile = PROFILE_YML.replace(_IMAGE_SOURCE_MARKER, WEB_TERMINALS_IMAGE_SOURCE_LINE)

    files = dict(BASE_SOURCE_FILES)
    files["profile.yml"] = profile.replace(_DEPLOY_BLOCK_MARKER + "\n", deploy_block)
    if with_ci:
        files.update(CI_PIPELINE_FILES)
    return {path: expand_sentinels(text) for path, text in files.items()}


def build_exemplar_repo(
    dest: Path,
    *,
    with_ci: bool = False,
    seed_env: bool = False,
    git: bool = False,
) -> Path:
    """Materialize the gold-standard three-zone deployment repo at ``dest``.

    ``dest`` is created if absent; it is the repo root, and its name is the
    deployment name. The exemplar's own identity (``Als Exemplar``) is written
    verbatim whatever the directory is called — two checkouts of one deployment
    at two paths is a real situation the lifecycle verbs have to tell apart,
    and this is how a test stages it.

    One consequence to know before materializing under a different name: the
    persona catalog's ``project``/``project_path`` values are derived from the
    deployment's directory name at emission (``als-exemplar-readonly``), so
    they are the one part of this text that a rename would make stale. They are
    frozen rather than templated because the byte comparison against a live
    ``osprey init`` is what this fixture exists for, and that comparison runs
    at :data:`EXEMPLAR_DIRNAME`.

    Args:
        dest: Directory to materialize into.
        with_ci: See :func:`exemplar_source_files`.
        seed_env: Also write the SECRETS zone — a ``.env`` with fake but
            well-formed values. Off by default: a freshly emitted repo has no
            ``.env`` until an operator seeds one.
        git: Run ``git init`` and commit the source zone, as ``osprey init``
            does. Off by default because most tests do not need it and it
            costs a subprocess.

    Returns:
        The repo root (``dest``, resolved).
    """
    root = Path(dest)
    root.mkdir(parents=True, exist_ok=True)

    for rel, text in exemplar_source_files(with_ci=with_ci).items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        if rel in EXECUTABLE_FILES:
            path.chmod(0o755)

    # STATE zone: present and empty. Git-ignored, so no .gitkeep — a marker
    # file there would be ignored too, and would be the one thing a `reset`
    # wipe has to work around.
    for rel in STATE_DIRS:
        (root / rel).mkdir(parents=True, exist_ok=True)

    if seed_env:
        env_path = root / ".env"
        env_path.write_text(ENV_SEEDED, encoding="utf-8")
        env_path.chmod(0o600)

    if git:
        _git_init(root)

    return root.resolve()


def _git_init(root: Path) -> None:
    """``git init`` plus one commit of the source zone, as ``init`` does.

    Hermetic on purpose: the developer's global and system git config are
    routed to /dev/null and identity comes from the environment, so a machine
    with commit signing, a template directory, or a global ``core.excludesFile``
    configured cannot change what this fixture produces. Signing in particular
    would either prompt or fail the commit outright in CI.
    """
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_AUTHOR_NAME": "OSPREY",
        "GIT_AUTHOR_EMAIL": "osprey@example.org",
        "GIT_COMMITTER_NAME": "OSPREY",
        "GIT_COMMITTER_EMAIL": "osprey@example.org",
    }

    def run(*args: str) -> None:
        subprocess.run(
            ["git", "-c", "commit.gpgsign=false", *args],
            cwd=root,
            env=env,
            check=True,
            capture_output=True,
        )

    run("init", "--quiet", "--initial-branch", "main")
    run("add", "--all")
    run("commit", "--quiet", "-m", "Initial deployment")


@pytest.fixture
def lifecycle_repo_factory(tmp_path: Path) -> Callable[..., Path]:
    """Materialize exemplar repos on demand, anywhere under ``tmp_path``.

    Called with no argument it makes ``tmp_path/als-exemplar``; pass a path for
    a second checkout, a nested repo, or a differently-named deployment.
    """

    def factory(dest: Path | str | None = None, **kwargs: object) -> Path:
        target = Path(dest) if dest is not None else tmp_path / EXEMPLAR_DIRNAME
        return build_exemplar_repo(target, **kwargs)  # type: ignore[arg-type]

    return factory


@pytest.fixture
def lifecycle_repo(lifecycle_repo_factory: Callable[..., Path]) -> Path:
    """The exemplar deployment repo at ``tmp_path/als-exemplar``, no ``.env``."""
    return lifecycle_repo_factory()
