"""The gold-standard four-zone deployment repo, materialized for tests.

This module is the hand-authored reference deployment (``als-exemplar/``) that
the lifecycle verbs are developed against, and the shape ``osprey init`` must
emit. Exemplar-first: the content below was authored by hand, derived from the
``control-assistant`` preset emission and rewritten onto the new command
surface; ``init`` is then made to reproduce it, never the reverse.

The layout is the four-zone repo — one directory, four kinds of content::

    als-exemplar/
    │  ═ SOURCE — tracked, user-edited ═══════════════
    ├── profile.yml  triggers.yml  README.md
    ├── data/  personas/  web-terminal-context/
    ├── .gitignore  .env.example  .env.shared  ci-extra.yml
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
that way. What the redesign does change: the four-zone header, comments moved
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

#: The preset the exemplar was materialized from, and the persona presets whose
#: deltas sit in ``personas/`` — two operator tiers plus the standalone ARIEL
#: logbook terminal the stack runs beside them.
EXEMPLAR_PRESET = "control-assistant"
PERSONA_PRESETS: Mapping[str, str] = {
    "admin": "control-assistant-admin",
    "ariel": "control-assistant-ariel",
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
# This file is your assistant's settings. Edit it, then run `osprey build`.
#
#   +--------------+  osprey   +--------------+  osprey  +----------------+
#   |    SOURCE    |  build    |    build/    |    up    |   DEPLOYMENT   |
#   | profile.yml  +---------->| config.yml   +--------->| agent CLI/web  |
#   | data/  ...   |           | .mcp.json  … |          | + containers   |
#   +--------------+           +--------------+          +----------------+
#          ^                                                     |
#          +---- edit -> osprey build -> osprey up --------------+
#
#   SOURCE   yours to edit, kept in git: this file, data/, personas/
#   SECRETS  .env, your API keys. Not in git. Rebuilds never touch it
#   OUTPUT   build/, generated. Never edit it; deleting it is safe
#   STATE    var/, the agent's memory and audit log. Not in git. Kept
#
# Made from the bundled `control-assistant` preset. Everything it sets is written
# out below and is yours to edit. Nothing is hidden or inherited.
#
#   emitted by OSPREY @OSPREY_VERSION@
#   preset content hash: @PRESET_HASH:control-assistant@

name: Als Exemplar

# Which packaged app this builds. "control_assistant" is the full one. Its data
# files are copied into data/, and that copy is what the build uses from then on.
app_template: control_assistant

# Which model answers. `osprey set provider=...` / `osprey set model=...` edit
# these in place, keeping your comments.
provider: anthropic
model: haiku   # tier (haiku/sonnet/opus), or any model ID the provider serves

# Any custom gateway works too: name it as the provider, describe it under
# `config:` below, and put its key in this repo's .env — the variable name
# derives from the provider name, <NAME>_API_KEY. Worked example:
#
# provider: my-gateway
# config:
#   api.providers.my-gateway.api_key: ${MY_GATEWAY_API_KEY}
#   api.providers.my-gateway.base_url: https://my-gateway.example.com/v1
#   # Optional — the gateway speaks Anthropic natively (e.g. a LiteLLM proxy
#   # in Anthropic mode), so the local translation proxy is skipped:
#   api.providers.my-gateway.api_protocol: anthropic
#   # Optional tier map, model IDs as the gateway names them. Unmapped tiers
#   # fall back to `model:` above, with a build-time warning:
#   api.providers.my-gateway.models:
#     haiku: claude-haiku-4-5
#     sonnet: claude-sonnet-4-6
#     opus: claude-opus-4-6

# How the agent searches for channels. `osprey set channel_finder_mode=...`
# also accepts in_context, middle_layer, or graph (the knowledge graph store).
channel_finder_mode: hierarchical

# ── What the agent can do ────────────────────────────────────────────────────
# Each entry is a name from the OSPREY artifact library. Delete what you do not
# need; add your own through an overlay.

hooks:
  - hook-log          # Append every tool call to a structured JSONL audit log
  - target-state      # Stdlib reader for the control-target state file (helper, not wired to an event)
  - hook-config       # Inject project config.yml path into every tool call env
  - approval          # Gate hardware-write tool calls on human approval prompt
  - writes-check      # Kill switch: refuse every write while writes_enabled is false
  - limits            # Enforce per-channel min/max limits before writes
  - error-guidance    # Post-error hook that surfaces remediation hints
  - memory-guard      # Gate Write/MultiEdit to memory files, NotebookEdit to agent-data artifacts
  - notebook-update   # Sync CLAUDE.md notebook after each session
  - cf-feedback-capture  # Capture channel-finder accuracy feedback for tuning
  - config-drift      # Warn at session start when the build is out of date
  - focus-validate    # Strip stale artifact IDs from focus_state.txt on each prompt
  - panels-context    # Tell the agent which web terminal panels exist
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
  - test-ioc-safety     # Test-IOC port isolation (EPICS-family control systems only)

skills:
  - diagnose        # Run a structured fault-diagnosis workflow
  # setup-mode is left out on purpose: it can edit config.yml and .mcp.json,
  # which is admin work. The `control-assistant-admin` persona adds it back,
  # which is where an admin-facing profile of your own should start too.
  - session-report  # Summarise session actions and outcomes to the logbook
  - demo-gallery    # Launch guided capability demonstrations
  - demo-ui         # Run a scripted demo of the agent driving the web workspace
  - writing-bluesky-plans  # Write, check and queue a plan (needs the Bluesky server)
  - operating-bluesky-plans  # Stage, queue and watch a plan (needs the Bluesky server)
  - bluesky-plans  # Browse which plans this deployment can run
  # Available — uncomment to enable:
  # - logbook-deep-research  # Multi-phase logbook investigation skill
  # - sim-scenarios  # List and switch simulated machine scenarios

agents:
  - channel-finder          # Semantic search over channel databases (hierarchical)
  - data-visualizer         # Produce strip charts and correlation plots
  - logbook-search          # Search facility logbook for historical entries
  - logbook-deep-research   # Multi-hop logbook research with synthesis
  - facility-knowledge      # Look up facility documentation, procedures, and device specs
  - facility-knowledge-graph  # Structural machine queries against the facility knowledge graph
  - pyat-specialist         # Lattice/optics computation sub-agent (pyAT)

output_styles:
  - control-operator  # Terse, actionable output style for control-room operators

web_panels:
  - ariel           # ARIEL search interface (past experiments, papers)
  - channel-finder  # Interactive channel-finder web UI
  - okf             # KNOWLEDGE tab, for browsing the facility knowledge bundle
  - system-health   # SYSTEM tab, a framework health dashboard
  # The events and bluesky panels are declared in personas/readwrite.yml
  # instead, so the read-only login is built without them.
  # Available — uncomment to enable:
  # - lattice  # Lattice dashboard

# ── Scanning and simulated hardware ──────────────────────────────────────────
# These three blocks give you a working plan setup with no real hardware: a
# Bluesky bridge with a Tiled data catalog, a simulated accelerator that speaks
# EPICS, and the web panels for both. Delete this section (and the bluesky
# panel above) if you do not want it.
bluesky:
  port: 8090
  tiled_enabled: true      # also runs the Tiled data catalog
  tiled_port: 8091

virtual_accelerator:
  # EPICS port the simulator serves on. The agent follows this value, so
  # changing it moves both.
  port: 5064

bluesky_web:
  port: 8095

# ── Stored archive (MongoDB) ─────────────────────────────────────────────────
# With this block, `osprey up` runs a real archive: a MongoDB store that is
# seeded with history on the first deploy and records the machine from then on.
# Without it, the agent invents plausible history when asked about the past.
#
# Every key is optional. These are the defaults, written out so you can see the
# shape. Do not also set `archiver.mongodb_archiver.*` under `config:` below;
# the build derives those keys from here and refuses a profile that has both.
va_archiver:
  # Where the agent reaches the store. `localhost` is this deployment's own;
  # point it at another host to read someone else's archive.
  host: localhost
  retention_days: 30       # how far back the archive reaches
  hot_span_hours: 48       # how much of it is kept at the dense sample rate
  hot_cadence_sec: 10      # seconds between samples inside the hot span
  tail_cadence_sec: 60     # and outside it (must be a whole multiple of the above)
  recorder_cadence_sec: 10  # how often the recorder samples the machine
  # A channel `osprey health` watches to check the archive is still recording:
  # if this stops moving, history has quietly stopped. Point it at something on
  # your machine that always changes; delete the key for no check at all.
  freshness_channel: SR:DIAG:DCCT:01:CURRENT:RB

# ── Everything else ──────────────────────────────────────────────────────────
# One dotted `key.path` per line. Never write a nested block here: it would
# replace the whole subtree and silently drop the keys beside it.
#
# This block is where configuration lives. build/config.yml is generated from
# it and should never be hand-edited. `osprey set` writes here.
config:
  # Which control system to talk to. "virtual_accelerator" is the built-in
  # simulator and works out of the box; "epics" is real hardware; "mock" needs
  # no containers but cannot complete a plan. `osprey set connector=epics`.
  control_system.type: virtual_accelerator
  # Use the archive declared by `va_archiver:` above. Declaring the block does
  # not turn it on; without this line you would deploy a store and not read it.
  archiver.type: mongodb_archiver
  # Both servers are off by default in OSPREY. Turn them on so the agent can
  # write and launch plans, and run read-only health checks.
  claude_code.servers.bluesky.enabled: true
  claude_code.servers.health.enabled: true
  # ── Tier floor ─────────────────────────────────────────────────────────────
  # The privileges every tier built from this preset starts WITHOUT. Each key
  # here is off at the bottom and lifted back on by exactly the tier that is
  # meant to have it, so a new persona inherits the restricted posture by
  # default and a privilege is something a profile has to ask for by name.
  #
  # The agent's own deployment-editing tool. It rewrites this repo's profile
  # and config, which is administration, not control-room work — so the base
  # takes it away and the admin tier lifts it back with
  # `claude_code.permissions.remove_deny`.
  #
  # Deliberately a `deny` and NOT a base-level `remove_ask`. String lists UNION
  # across `extends`: a child can add to an inherited list but never subtract
  # from it, so a `remove_ask` written here would leak into EVERY tier —
  # including admin — and strip the approval prompt the admin tier relies on to
  # keep the tool supervised. `deny` inverts that: it is subtractable per tier
  # via `remove_deny`, which is the direction a floor has to work in.
  claude_code.permissions.deny:
    - mcp__osprey_workspace__setup_patch
  # The web Config panel edits the running deployment's configuration from the
  # browser. Same reasoning as the tool above: administration, so it is off
  # here and turned back on by the admin tier alone.
  web.config_panel.enabled: false
  # The scaffold gallery stays READABLE at every tier — browsing the prompt and
  # skill library is ordinary work. What this turns off is writing to it: the
  # gallery's edit, create and delete surfaces are shared deployment state.
  web.scaffold_gallery.write_enabled: false
  # system.timezone: America/Los_Angeles
  # Your facility's name, used in the agent's prompts and on the web landing
  # page. Defaults to the deployment name.
  # facility.name: My Facility
  # Starting theme for every web terminal. Each browser can override it from
  # the display menu.
  web.theme: light
  # ── Web terminals ──────────────────────────────────────────────────────────
  # `osprey up` runs a landing page and one terminal per user listed below.
  # `osprey web` ignores all of this, so a single terminal on your own machine
  # works at any time. Set `modules.web_terminals.enabled: false` for backend
  # services only.
  #
  # Short prefix for the web container names (`<prefix>-nginx`, `<prefix>-web-
  # <user>`). Must start with a letter or digit. Use your facility's initials.
  facility.prefix: ca
  # The hostname people open in a browser. 127.0.0.1 is your own machine; set
  # your real hostname to reach it from anywhere else.
  deploy.fqdn: 127.0.0.1
  # Keep this as ONE dotted key. A nested `modules:` block here would replace
  # the whole modules subtree and silently drop the others.
  modules.web_terminals:
    enabled: true
@WEB_TERMINALS_IMAGE_SOURCE@
    nginx_port: 9080            # the landing page everyone opens first
    # Browsers reach this nginx directly here, so the address they open is
    # derived from deploy.fqdn and the port above — and that is the address
    # every terminal checks an action against. Put something in front of this
    # nginx (a load balancer terminating TLS, a reverse proxy, a DNS alias) and
    # add `external_origin: https://<what browsers open>` beside this line, or
    # every action inside a terminal is refused while every page still loads.
    # User number i gets base + i in each family below, so removing a user
    # never shifts anyone else's ports. These all sit above this deployment's
    # own service ports (5064, 8020, 8090/8091/8095) to avoid collisions.
    web_base_port: 9091           # first per-user web-terminal port
    artifact_base_port: 9291      # first per-user artifact-gallery port
    ariel_base_port: 9391         # first per-user ARIEL search port
    lattice_base_port: 9491       # first per-user lattice-dashboard port
    channel_finder_base_port: 9591  # first per-user channel-finder panel port
    default_persona: readonly   # used for any user below with no persona
    # Every terminal below asks for a login (user alice: password alice, and
    # so on — set in this repo's .env; rotate with `osprey users passwd`).
    auth:
      method: password
      # Accepts login over plain HTTP, which fits 127.0.0.1 and nothing else.
      # For any reachable host, delete this line and configure tls instead.
      allow_insecure_http: true
    # Which tier a user lands on is pinned per entry below. Single sign-on can
    # pick it instead by mapping provider groups onto declared roles — see
    # "Let single sign-on pick the tier" in the multi-user login guide.
    #   authorization:
    #     roles: {operator: {persona: readwrite}, viewer: {persona: readonly}}
    #     claims: {claim: groups, map: {ca-operators: operator, ca-viewers: viewer}}
    # How the landing page is laid out. Omit this whole block and you get one
    # section holding every entry below, headed "Terminals".
    landing:
      # Each file below becomes one collapsible section at the bottom of the
      # landing page, in this order. The file's `# H1` is the section label, so
      # adding a section means adding a markdown file and listing it here.
      # `data/landing/working-safely.md` ships with this preset and is yours to
      # rewrite; add your own for local procedures, contacts or shift handover.
      # Drop this key entirely and you get OSPREY's built-in safety notice
      # instead; set it to [] for no notices at all.
      notices:
        - data/landing/working-safely.md
      # The line under everything. Set to "" for no footer.
      footer: OSPREY multi-user web terminal stack. Experimental system. Proceed with caution.
      groups:
        # `users` renders the roster. It also SPLITS it: any entry whose
        # persona declares a `landing_group` (see `ariel` below) moves into a
        # section of its own, drawn as an accent-edged panel underneath it.
        # So the page reads people first, services after.
        - type: users
          label: Users
    users:
      # One web terminal per entry. `index` pins that user's ports, `persona`
      # picks their permissions from the list below, and `display_name` becomes
      # the browser tab title, which is how you tell the terminals apart.
      - name: alice
        index: 0
        persona: readwrite
        display_name: "Control Room (Alice)"
      - name: bob
        index: 1
        persona: readonly
        display_name: "Read-Only View (Bob)"
      # Not a person: a second product running beside them. Same machinery as
      # any other entry — its own container, ports and volumes — but what is
      # behind the card is the ARIEL logbook assistant, not a control terminal.
      - name: ariel
        index: 2
        persona: ariel
        display_name: "ARIEL Logbook Research"
        # Read-only research service, open without a login on purpose.
        login: false
      # The one login that can change this deployment's configuration — the web
      # Config panel, the scaffold gallery's editors, and the agent's own setup
      # tool. Behind the login wall like every other person on this page: an
      # admin card without one would hand deployment edits to anyone who opens
      # it. Last in the roster so the operator cards stay where they are.
      - name: carol
        index: 3
        persona: admin
        display_name: "Deployment Admin (Carol)"
    personas:
      # `osprey build` builds one of these per file in personas/, into build/.
      # `osprey up` builds nothing: if one is missing it stops and says so.
      readonly:
        project: als-exemplar-readonly
        project_path: build/als-exemplar-readonly
        build_profile: personas/readonly.yml
      readwrite:
        project: als-exemplar-readwrite
        project_path: build/als-exemplar-readwrite
        build_profile: personas/readwrite.yml
      admin:
        project: als-exemplar-admin
        project_path: build/als-exemplar-admin
        build_profile: personas/admin.yml
      ariel:
        project: als-exemplar-ariel
        project_path: build/als-exemplar-ariel
        build_profile: personas/ariel.yml
        # Puts this persona's users under their own landing-page heading
        # instead of in with the people. Presentation only — it changes
        # nothing about the container, its ports, or what it can do.
        landing_group: Standalone deployments

# ── Answering webhooks (optional) ────────────────────────────────────────────
# Lets an outside system ask the agent a question over HTTP. The triggers that
# ship need no control system, so a single `curl` after `osprey up` exercises
# it. Delete this block to turn it off.
dispatch:
  triggers: triggers.yml            # a path in this repo, or a bundled name
  worker_count: 1
  workspace_mode: isolated
  max_concurrent_runs: 2
  max_queue_depth: 50

# ── Environment variables ────────────────────────────────────────────────────
# Passwords for the webhook service above. They have no defaults on purpose:
# the service refuses to start without them. `osprey up` generates a strong
# random value for each one into this repo's .env, so a new deployment is
# secure with no editing. Put your own values in .env to override.
env:
  required:
    - EVENT_DISPATCHER_TOKEN
    - DISPATCH_WORKER_TOKEN
  # Demo login passwords for the terminals above, written into this repo's
  # .env by `osprey init`. Edit them there — these are not secrets.
  defaults:
    OSPREY_AUTH_PW_ALICE: alice
    OSPREY_AUTH_PW_BOB: bob
    OSPREY_AUTH_PW_CAROL: carol
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
    "    # No deploy block yet, so this is image_source's only home: build each\n"
    "    # terminal image here rather than pulling it from a registry.\n"
    "    image_source: local"
)


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE zone — persona deltas
# ─────────────────────────────────────────────────────────────────────────────

PERSONA_ARIEL_YML = """\
# Als Exemplar (ariel) — settings for one web login
#
# Only the differences from profile.yml belong here. The build merges this
# file over that one, picking up any edit you make there. To see the combined
# result:
#   osprey validate personas/ariel.yml
#
# Made from the bundled `control-assistant-ariel` preset.
#
#   emitted by OSPREY @OSPREY_VERSION@
#   preset content hash: @PRESET_HASH:control-assistant-ariel@

name: Als Exemplar (ariel)

# Attached render: this persona builds a per-user terminal image only and
# connects to the shared web tier the hosting deployment runs on the same host.
# No services are scaffolded — the ARIEL Postgres it reads is the hosting
# deployment's, already declared there.
deploy_services: false

# The logbook-research persona instead of the control-room operator one. This is
# the single biggest reason this tier feels like a different product: the agent's
# whole brief is written for logbook work.
claude_md_template: CLAUDE.ariel.md.j2

# Open on the ARIEL tab rather than the always-on Workspace tab. The workspace
# tab stays present — the deep-research skill hands artifacts off through it —
# it is just not the one you land on.
default_panel: ariel

# ── What this tier drops ─────────────────────────────────────────────────────
# A persona can only add to an inherited list, never subtract from it, so
# everything the control-room agent carries and a logbook agent has no use for
# is removed here by name. What is left is the `ariel-standalone` selection.
exclude:
  hooks:
    - writes-check        # No hardware writes to pre-check
    - limits              # No per-channel limits to enforce
    - cf-feedback-capture  # No channel finder to tune
  rules:
    - python-execution    # The Python sandbox is off (see config: below)
    - data-visualization  # Plotting needs the sandbox this tier does not run
    - control-system-safety  # EPICS PV rules, with no EPICS behind them
    - test-ioc-safety     # Test-IOC port isolation, likewise
  skills:
    - diagnose            # Fault diagnosis is control-room work
    - demo-gallery
    - demo-ui
    - writing-bluesky-plans    # Plan authoring needs the Bluesky server
    - operating-bluesky-plans  # and so does running one
    - bluesky-plans            # and so does listing them
  agents:
    - channel-finder
    - data-visualizer
    - facility-knowledge
    - pyat-specialist
    # Logbook search and deep research are re-added below as SKILLS, which is
    # how the standalone ARIEL agent invokes them: from the main agent, with the
    # whole session's context, rather than through a subagent boundary that a
    # single-purpose agent gains nothing from crossing.
    - logbook-search
    - logbook-deep-research
  web_panels:
    - channel-finder
    - okf
    - system-health
# The `safety` rule is deliberately NOT excluded, and this is the one place this
# tier differs from the standalone preset. Its tools are gone, so the rule
# governs nothing today and costs a few lines of prompt. It stays because this
# agent runs inside a deployment that does move hardware: if anyone ever turns a
# control server back on here, the rule should already be in place rather than
# be the thing someone remembered to add.

skills:
  - logbook-deep-research   # Multi-phase logbook investigation

# ── Config overrides ─────────────────────────────────────────────────────────
# Dotted keys ONLY — see the base profile's block.
config:
  # The axis this tier is defined by: no control-system surface at all. Each
  # line switches off a tool server the base turns on. Together they are what
  # makes this a logbook terminal rather than a control-room one with the
  # panels hidden — the tools are absent, not merely unused.
  claude_code.servers.controls.enabled: false
  claude_code.servers.python.enabled: false
  claude_code.servers.channel-finder.enabled: false
  claude_code.servers.bluesky.enabled: false
  claude_code.servers.health.enabled: false
  claude_code.servers.osprey_facility_knowledge.enabled: false
  # The graph store is a control-room surface too, and this tier's exclusion
  # of it has to be said: the build tells every attached render where the
  # hosting deployment's services are (the Reach Contract), and a
  # `services.graphdb` block is what makes the graph server render. Only a
  # server switched off is told nothing about the store.
  claude_code.servers.graph.enabled: false
  # Pinned even though the server that would honour it is gone, for the same
  # reason the other two tiers pin it: this key is the write boundary, and it
  # must not drift if the base's default ever changes.
  control_system.writes_enabled: false
  # Creating a logbook entry is this agent's only write of any kind, and it is
  # approval-gated like every other write in OSPREY.
  approval.tools.entry_create: always
  # Full split-pane layout: the ARIEL search panel is the point of this tier, so
  # it needs the panel area. Pinned rather than left to the server default for
  # the same reason the other two tiers pin theirs.
  web.ui_mode: expert
  # The hosting deployment owns the web-terminal tier (nginx, landing, per-user
  # containers). Without this override the inherited roster would make this
  # render try to host a second web tier on the same host ports.
  modules.web_terminals.enabled: false
  # Nothing here says where the hosting deployment's services are — the qmd
  # sidecar hybrid logbook search dials (the point of this tier), the Postgres
  # the logbook lives in, the telemetry store. This persona is an attached
  # render (`services: {}` of its own) built beside that deployment, and the
  # build copies every such fact from the deployment's own render into it
  # (the Reach Contract, `osprey.deployment.reach`). Per-user web-terminal
  # containers run `network_mode: host`, so container `localhost` IS the
  # deployment host and the copied ports are dialed there. Built alone, with
  # no hosting deployment in the repo, the build copies what the app template
  # deploys at its defaults instead — and a host that differs is named here.
"""

PERSONA_READONLY_YML = """\
# Als Exemplar (readonly) — settings for one web login
#
# Only the differences from profile.yml belong here. The build merges this
# file over that one, picking up any edit you make there. To see the combined
# result:
#   osprey validate personas/readonly.yml
#
# Made from the bundled `control-assistant-readonly` preset.
#
#   emitted by OSPREY @OSPREY_VERSION@
#   preset content hash: @PRESET_HASH:control-assistant-readonly@

name: Als Exemplar (readonly)

# Attached render: this persona builds per-user terminal images only and
# connects to the shared web tier the hosting deployment runs on the same host.
# No services are scaffolded — the injector blocks inherited from the base are
# all gated on this flag and skip cleanly.
deploy_services: false

# ── Config overrides ─────────────────────────────────────────────────────────
# Dotted keys ONLY — see the base profile's block.
config:
  # The axis this persona hard-pins, and it takes three keys rather than one.
  # The flat key is the posture a connector type inherits when its own block
  # says nothing, so on its own it is not a floor: a per-type `true` anywhere
  # in the chain — the base, an overlay, a facility's own edit — would arm that
  # type over it. So the read-only tier pins every block off as well as the key
  # they inherit from, and a type that arms writes here has to be added by name.
  control_system.writes_enabled: false
  control_system.connector.epics.writes_enabled: false
  control_system.connector.virtual_accelerator.writes_enabled: false
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
  # Nothing here says where the hosting deployment's services are — the graph
  # store's bolt port, the qmd sidecar's port, the Postgres the logbook lives
  # in, the telemetry store, the bluesky bridge. This persona is an attached
  # render (`services: {}` of its own) built beside that deployment, and the
  # build copies every such fact from the deployment's own render into it
  # (the Reach Contract, `osprey.deployment.reach`). Per-user web-terminal
  # containers run `network_mode: host`, so container `localhost` IS the
  # deployment host and the copied ports are dialed there. Move a port on the
  # hosting profile and every persona follows; spell a different one here and
  # the build refuses the contradiction. Built alone, with no hosting
  # deployment in the repo, the build copies what the app template deploys
  # at its defaults instead — and a host that differs IS named here.
"""

PERSONA_READWRITE_YML = """\
# Als Exemplar (readwrite) — settings for one web login
#
# Only the differences from profile.yml belong here. The build merges this
# file over that one, picking up any edit you make there. To see the combined
# result:
#   osprey validate personas/readwrite.yml
#
# Made from the bundled `control-assistant-readwrite` preset.
#
#   emitted by OSPREY @OSPREY_VERSION@
#   preset content hash: @PRESET_HASH:control-assistant-readwrite@

name: Als Exemplar (readwrite)

# Attached render: this persona builds per-user terminal images only and
# connects to the shared web tier the hosting deployment runs on the same host.
deploy_services: false

# The write-oriented panels. Persona lists UNION over the base, so these are
# added to the inherited builtin set. Each tab's URL, path and label are not
# spelled here: the hosting deployment's build derives them when it injects
# the event dispatcher and the bluesky-web sidecar, and this attached render
# is told them from that render (the Reach Contract, `osprey.deployment.reach`)
# — so a sidecar moved on the hosting profile moves the tab with it.
web_panels:
  - events          # EVENTS dashboard tab (event dispatcher)
  - bluesky         # Plan authoring, the plan queue, and the run's live results

# ── Config overrides ─────────────────────────────────────────────────────────
# Dotted keys ONLY — see the base profile's block.
config:
  # The single axis this persona hard-pins. It is the inherited posture rather
  # than a verdict: a `control_system.connector.<type>.writes_enabled` key
  # anywhere in the chain answers for that type instead, which is how the
  # simulator-only tier is built. With none written, every type reads this key,
  # so it must not drift silently if the base's default ever changes.
  control_system.writes_enabled: true
  # Full split-pane terminal + workspace layout for the write-armed operator.
  # Pinned on both sides of the tier boundary (readonly pins `simple`) rather
  # than left to the server default, for the same reason writes_enabled is.
  web.ui_mode: expert
  # The hosting deployment owns the web-terminal tier (nginx, landing,
  # per-user containers). Without this override the inherited roster would
  # make this render try to host a second web tier on the same host ports.
  modules.web_terminals.enabled: false
  # Nothing here says where the hosting deployment's services are — the graph
  # store's bolt port, the qmd sidecar's port, the Postgres the logbook lives
  # in, the telemetry store, the bluesky bridge, the EVENTS and BLUESKY tabs'
  # URLs. This persona is an attached render (`services: {}` of its own) built
  # beside that deployment, and the build copies every such fact from the
  # deployment's own render into it (the Reach Contract,
  # `osprey.deployment.reach`). Per-user web-terminal containers run
  # `network_mode: host`, so container `localhost` IS the deployment host and
  # the copied ports are dialed there. Move a port on the hosting profile and
  # every persona follows; spell a different one here and the build refuses
  # the contradiction. Built alone, with no hosting deployment in the repo,
  # the build copies what the app template deploys at its defaults instead
  # — and a host that differs IS named here.
"""

PERSONA_ADMIN_YML = """\
# Als Exemplar (admin) — settings for one web login
#
# Only the differences from profile.yml belong here. The build merges this
# file over that one, picking up any edit you make there. To see the combined
# result:
#   osprey validate personas/admin.yml
#
# Made from the bundled `control-assistant-admin` preset.
#
#   emitted by OSPREY @OSPREY_VERSION@
#   preset content hash: @PRESET_HASH:control-assistant-admin@

name: Als Exemplar (admin)

# Attached render: this persona builds per-user terminal images only and
# connects to the shared web tier the hosting deployment runs on the same host.
# No services are scaffolded — the injector blocks inherited from the base are
# all gated on this flag and skip cleanly.
deploy_services: false

# The guided configuration workflow, left out of the base on purpose because it
# edits config.yml and .mcp.json. Skill lists UNION over the base, so this is
# added to the inherited selection rather than replacing it.
skills:
  - setup-mode      # Inspect the deployment's configuration and patch it

# ── Config overrides ─────────────────────────────────────────────────────────
# Dotted keys ONLY — see the base profile's block.
config:
  # The admin tier sits above readwrite: it keeps the write-armed control
  # posture and adds deployment editing on top. This is the posture every
  # connector type inherits when its own
  # `control_system.connector.<type>.writes_enabled` block says nothing, and
  # none is written here, so it is the answer for every machine the session can
  # be pointed at. Pinned like the tiers beneath it pin their own side, so the
  # boundary cannot drift if the base's default ever changes.
  control_system.writes_enabled: true
  # The axis this tier is defined by: the three privileges the base floors, all
  # lifted here and nowhere else.
  #
  # The deployment-editing tool. `remove_deny` subtracts the base's deny for
  # this render alone — the direction a floor has to be lifted in, since string
  # lists only ever union across `extends`. What is left is the `ask` entry the
  # workspace server declares, which routes the call through the approval hook.
  claude_code.permissions.remove_deny:
    - mcp__osprey_workspace__setup_patch
  # The same capability from the browser: the web Config panel edits the
  # running deployment's configuration.
  web.config_panel.enabled: true
  # The gallery is readable at every tier; this turns its edit, create and
  # delete surfaces back on. Its contents are shared deployment state, which is
  # exactly what this tier is for.
  web.scaffold_gallery.write_enabled: true
  # Full split-pane terminal + workspace layout: the Config panel and the
  # gallery editors need the panel area. Pinned rather than left to the server
  # default, for the same reason the other tiers pin theirs.
  web.ui_mode: expert
  # The hosting deployment owns the web-terminal tier (nginx, landing, per-user
  # containers). Without this override the inherited roster would make this
  # render try to host a second web tier on the same host ports.
  modules.web_terminals.enabled: false
  # Nothing here says where the hosting deployment's services are — the graph
  # store's bolt port, the qmd sidecar's port, the Postgres the logbook lives
  # in, the telemetry store, the bluesky bridge. This persona is an attached
  # render (`services: {}` of its own) built beside that deployment, and the
  # build copies every such fact from the deployment's own render into it
  # (the Reach Contract, `osprey.deployment.reach`). Per-user web-terminal
  # containers run `network_mode: host`, so container `localhost` IS the
  # deployment host and the copied ports are dialed there. Move a port on the
  # hosting profile and every persona follows; spell a different one here and
  # the build refuses the contradiction. Built alone, with no hosting
  # deployment in the repo, the build copies what the app template deploys
  # at its defaults instead — and a host that differs IS named here.
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
        (content_type markdown) summarizing the event payload and what you would
        do next. Confirm the artifact you created.
      allowed_tools:
        - Glob
        - Read
        - mcp__osprey_workspace__artifact_register
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
# This repo is the deployment: the source zone is tracked, and the
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
# lock file the write-back path creates beside them. Two exceptions carry no
# values a host may not share: .env.example, the documented variable list, and
# .env.shared, this deployment's committed defaults.
#
# Every zone entry above is anchored to the repo root with a leading slash. An
# unanchored `build/` or `.env*` would also swallow a same-named path anywhere
# deeper in the tree — including files moved there later — and it would do it
# silently.
#
# The same pattern covers `.env.variant`, which is not a secret but is
# host-local for the same reason: it holds `OSPREY_PROFILE_VARIANT=<name>`, naming
# which `profiles/<name>.yml` overlay THIS host builds. Committing it would
# hand this host's choice to every other one.
/.env*
!/.env.example
!/.env.shared

# The compose document a deploy merges here when the container runtime needs a
# single file. Machine-written, rewritten by every `osprey up`, removed by
# `osprey reset` — anchored for the same reason the zones above are.
/.osprey-compose.yml

# OS / editor noise. Deliberately unanchored: these are junk at any depth.
.DS_Store
*.swp
*.swo
"""

ENV_EXAMPLE = """\
# Als Exemplar Environment Configuration
#
# Every variable this agent reads, listed in one place. Copy this file to `.env`
# beside it and fill in what you need. That one file holds all your secrets, and
# a value in it survives every rebuild.
#
# This file has no secrets in it and is safe to commit.
#
# The `.env` files, and which one to edit:
#
#   .env.shared    the settings that are the same on every host; committed
#   .env           this host's own values and every key; wins over
#                  .env.shared when both set the same one
#   .env.example   this file: documentation, no values, never read at run time
#
# Anything else starting with `.env` is generated by a deploy — kept out of
# git, and never edited by hand.

# API key for the provider this assistant uses. Fill this in.
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Other providers OSPREY supports. Uncomment one if you switch to it with
# `osprey set provider=...`.
# OPENAI_API_KEY=your-openai-api-key-here
# GOOGLE_API_KEY=your-google-api-key-here
# CBORG_API_KEY=your-cborg-api-key-here
# AMSC_I2_API_KEY=your-amsc-i2-api-key-here
# ARGO_API_KEY=your-argo-api-key-here
# STANFORD_API_KEY=your-stanford-api-key-here
# ALS_APG_API_KEY=your-als-apg-api-key-here

# Required by this profile. Its `env.required` names these, and the agent
# cannot run without them.
EVENT_DISPATCHER_TOKEN=
DISPATCH_WORKER_TOKEN=

# Declared by this profile with a default (`env.defaults`). Override only if
# your facility needs a different value.
OSPREY_AUTH_PW_ALICE=alice
OSPREY_AUTH_PW_BOB=bob
OSPREY_AUTH_PW_CAROL=carol

# Runtime overrides (optional - for advanced use cases)
#LOCAL_PYTHON_VENV=/path/to/your/venv/bin/python

# Proxy settings (NO_PROXY, HTTP_PROXY, HTTPS_PROXY) live in `.env.shared`:
# a site behind a corporate firewall is behind it on every host, so they are a
# shared default rather than something each operator sets again.

# Service passwords. `osprey up` generates a strong random value for each of
# these when it deploys the matching service, and writes it into this repo's
# .env. Set one by hand only to keep a value the deployment must not change,
# such as the password on a database volume you already have.
# EVENT_DISPATCHER_TOKEN=  # event_dispatcher, dispatch_worker — authenticates callers to the event-dispatcher API
# DISPATCH_WORKER_TOKEN=  # event_dispatcher, dispatch_worker — authenticates the dispatch worker back to the dispatcher
# BLUESKY_LAUNCH_TOKEN=  # bluesky — arms the Bluesky bridge's plan-launch endpoint
# BLUESKY_TILED_API_KEY=  # bluesky — the key the bridge presents to the co-deployed Tiled catalog
# BLUESKY_VA_LAUNCH_TOKEN=  # bluesky_va — arms the plan-launch endpoint of the second Bluesky lane, the one serving the virtual accelerator (only on a deployment with `bluesky.second_lane`)
# BLUESKY_LIVE_LAUNCH_TOKEN=  # bluesky_live — arms the plan-launch endpoint of the second Bluesky lane, the one serving the live machine (only on a deployment with `bluesky.second_lane`)
# OSPREY_TERMINAL_SECRET=  # bluesky_web — the operator login secret for the bluesky-web panel's web gate
# ZO_ROOT_USER_PASSWORD=  # openobserve — OpenObserve root/ingest credential
# ARIEL_DB_PASSWORD=  # postgresql — ARIEL Postgres password (also fills the agent's derived DSN)
# MONGO_ROOT_PASSWORD=  # mongodb — archiver store root password (the seeder, recorder and agent all authenticate with it)
# GRAPHDB_PASSWORD=  # graphdb — graph store password (the seeder, health check and deploy staging all authenticate with it)
"""

#: The committed half of the env chain, as `osprey init` authors it: every line
#: a comment, because a deployment needs no shared defaults to run. `.env`
#: beside it carries this host's values and wins on any key both files set.
ENV_SHARED = """\
# Als Exemplar — shared, committed defaults.
#
# The non-secret half of this deployment's environment, and the one env file
# that IS tracked in git. Every host that clones this repo starts from the
# values here, so a setting the whole site needs — a proxy, a facility
# hostname, a shared port — belongs in this file rather than in each
# operator's own `.env`.
#
# Precedence, lowest first:
#
#   .env.shared   these defaults, committed, the same on every host
#   .env          this host's own values and every secret — LOCAL WINS
#
# A key set in both files takes its value from `.env`. There is nothing more to
# it than that: same syntax, same variables, lower precedence.
#
# One exception, and only if profile.yml asks for it: a variable listed under
# `env.pinned` is this file's to decide. `osprey up` refuses to start when
# `.env` or a shell export sets one.
#
# Never put a secret here — this file is committed. An API key, a token or a
# password goes in `.env`, which git ignores and which never leaves the host.
# Neither file ever enters a container image; both are read at run time.

# Proxy settings — uncomment if this site sits behind a corporate firewall.
# NO_PROXY=localhost,127.0.0.1
# HTTP_PROXY=http://proxy.example.com:8080
# HTTPS_PROXY=http://proxy.example.com:8080

# Site CA bundle — uncomment if a proxy re-signs TLS with a site CA.
# On RHEL-family hosts the system bundle lives here:
# SSL_CERT_FILE=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem
# REQUESTS_CA_BUNDLE=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem
# NODE_EXTRA_CA_CERTS=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem
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

This folder is your OSPREY assistant. Everything it is made of lives here, and
the folder name is the assistant's name.

## What is in here

| What | Where | In git? | Kept? |
| --- | --- | --- | --- |
| Your settings | `profile.yml`, `data/`, `personas/` | yes | yes |
| Your API keys | `.env` | no | yes |
| Generated files | `build/` | no | no, safe to delete |
| The agent's memory and audit log | `var/agent_data/`, `var/audit/` | no | yes |

In full, the first row is: `profile.yml`, `data/`, `personas/`, `triggers.yml`, `web-terminal-context/`, `.env.example`, `.gitignore`, `.env.shared`, `README.md`, `ci-extra.yml`, `.gitlab-ci.yml`, `scripts/verify.sh`.

`build/` is generated from your settings every time you run `osprey build`.
Deleting it is always safe: no settings, no keys and no agent memory live there.

## The `.env` files

Two of these are yours to edit, one is documentation, and anything else
starting with `.env` is generated by a deploy — kept out of git, and never
edited by hand.

| File | What it is for | In git? |
| --- | --- | --- |
| `.env.shared` | edit — the settings that are the same on every host | yes |
| `.env` | edit — this host's own values, and every key | no |
| `.env.example` | documentation — every variable this deployment reads | yes |
| `.env.merged` | generated — the settings a deploy hands the containers | no |

`.env.shared` and `.env` are read together, `.env` last: if the same
setting appears in both, the one in `.env` wins. That is how a single host
changes a shared default without affecting anyone else. None of these files go
into a container image — they are all read when the deployment starts.

If `profile.yml` lists a variable under `env.pinned`, that one is the exception:
`.env.shared` decides it, and `osprey up` refuses to start when `.env`
or a shell export disagrees.

`.osprey-compose.yml` at the root is generated the same way, so a deploy
can hand the container runtime one file instead of several. It is kept out of
git, holds no keys, is rewritten by every `osprey up`, and `osprey reset`
removes it.

## Everyday commands

```bash
osprey build          # turn your settings into something runnable
osprey up -d          # start it in the background
osprey status         # what is running, and is it up to date
osprey logs           # watch the logs
osprey down           # stop it
```

Run these from anywhere inside this folder. They find their way to the top on
their own, so they need no arguments. `--repo PATH` points them somewhere else.

## Changing something

Edit `profile.yml` (or run `osprey set model=sonnet` to change one setting),
then:

```bash
osprey build && osprey up -d
```

`osprey up` starts what `osprey build` last produced. If you change your
settings without rebuilding, `up` stops and tells you what changed, so a
half-finished edit cannot reach a running system. `osprey up --build` does both
steps; `osprey up --as-built` starts the previous build anyway.

## Running it on a server

To run this somewhere other than your own machine, fill in the `deploy:` section
at the end of `profile.yml` (which server, which CI system), then run:

```bash
osprey scaffold ci
```

That writes the pipeline files. Your own extra CI jobs go in `ci-extra.yml`,
which nothing ever overwrites.

## Starting over

```bash
osprey reset          # stops everything, then deletes containers, agent data and build/
```

`reset` keeps `var/audit/` and your API keys. `osprey reset --purge-audit`
deletes the audit log as well; that plus deleting this folder removes it all.

## Backups

Git covers your settings. `var/` and `.env` are everything else, so a backup
is a copy of those two, and a restore is:

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
# `osprey users env` turns the host's .env into the runtime secrets
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
      osprey users env --output .env.users
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
# The landing page is nginx's own file, served before anything asks
# the caller for a credential. A terminal is the application, which
# answers an uncredentialed GET / with a 401 — so it is probed at
# /health, the route its auth gate lets through.
if wants web; then
  printf '\\n%s── Web terminal ──%s\\n\\n' "$BOLD" "$RESET"
  probe_http 'landing page'      http://localhost:9080/
  probe_http 'terminal (alice)'  http://localhost:9091/health
  probe_http 'terminal (bob)'    http://localhost:9092/health
  probe_http 'terminal (ariel)'  http://localhost:9093/health
  probe_http 'terminal (carol)'  http://localhost:9094/health
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
    ".env.shared": ENV_SHARED,
    "README.md": README_MD,
    "ci-extra.yml": CI_EXTRA_YML,
    "triggers.yml": TRIGGERS_YML,
    "personas/ariel.yml": PERSONA_ARIEL_YML,
    "personas/readonly.yml": PERSONA_READONLY_YML,
    "personas/readwrite.yml": PERSONA_READWRITE_YML,
    "personas/admin.yml": PERSONA_ADMIN_YML,
    "web-terminal-context/alice/.gitkeep": "",
    "web-terminal-context/ariel/.gitkeep": "",
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
    """Materialize the gold-standard four-zone deployment repo at ``dest``.

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
