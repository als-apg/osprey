"""The gold-standard four-zone deployment repo, materialized for tests.

This module is the hand-authored reference deployment (``als-exemplar/``) that
the lifecycle verbs are developed against, and the shape ``osprey init`` must
emit. Exemplar-first: the content below was authored by hand, derived from the
``control-assistant`` preset emission and rewritten onto the new command
surface; ``init`` is then made to reproduce it, never the reverse.

The layout is the four-zone repo — one directory, four kinds of content::

    als-exemplar/
    │  ═ SOURCE — tracked, user-edited ═══════════════
    ├── profile.yml  providers.yml  triggers.yml  README.md
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

Three values cannot be frozen into the text: the installed OSPREY version, the
bundled presets' content hashes, and the provider catalog's content hash, all of
which the real emission stamps into the provenance header. They are written as
``@OSPREY_VERSION@``, ``@PRESET_HASH:<preset>@`` and ``@PROVIDERS_HASH@``
sentinels and expanded at materialization, so a byte-comparison against a live
``osprey init`` stays meaningful as the repo moves. Sentinels rather than
``str.format``/``%`` because the YAML carries literal ``${VAR:-default}`` shell
expansions.

``providers.yml`` is the fourth thing not frozen. Init copies the packaged
catalog beside the profile verbatim, so the fixture reads that file rather than
holding a second copy of it; see :func:`packaged_providers_yml`.

Usage::

    def test_something(lifecycle_repo):          # als-exemplar/ in tmp_path
        assert (lifecycle_repo / "profile.yml").is_file()

    def test_two_checkouts(lifecycle_repo_factory, tmp_path):
        a = lifecycle_repo_factory(tmp_path / "a")
        b = lifecycle_repo_factory(tmp_path / "b", seed_env=True)
"""

from __future__ import annotations

import contextlib
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
#: deltas sit in ``personas/`` — the operator tiers plus the two standalone
#: terminals the stack runs beside them, logbook research and facility knowledge.
EXEMPLAR_PRESET = "control-assistant"
PERSONA_PRESETS: Mapping[str, str] = {
    "admin": "control-assistant-admin",
    "knowledge": "control-assistant-knowledge",
    "logbook": "control-assistant-logbook",
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

PROFILE_YML = r"""# Als Exemplar — OSPREY deployment repo
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

# Which model answers. `osprey set provider=...` / `osprey set model=...` edit
# these in place, keeping your comments.
provider: anthropic
model: haiku   # tier (haiku/sonnet/opus), or any model ID the provider serves

# `provider:` names an entry in providers.yml, the provider catalog beside this
# file. To use a gateway of your own, add its entry there (api_key, base_url and
# a models tier map) and name it here; the key goes in this repo's .env under the
# variable the entry's `api_key` references. A `config: api.providers.*` key is
# refused: the catalog is the one home for provider endpoints.

# How the agent searches for channels: `graph` answers from the facility
# knowledge graph this deployment runs (`services.graphdb`), the same store
# the facility-knowledge-graph agent reads, so channels, devices and their
# addresses have one source. `osprey set channel_finder_mode=...` also
# accepts hierarchical, in_context or middle_layer, each reading a channel
# database file from data/ instead; every persona inherits the mode.
channel_finder_mode: graph

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
  - memory-guard      # Gate Write/MultiEdit to memory files, NotebookEdit to agent-data artifacts and notebooks
  - notebook-update   # Sync CLAUDE.md notebook after each session
  - cf-feedback-capture  # Capture channel-finder accuracy feedback for tuning
  - config-drift      # Warn at session start when the build is out of date
  - focus-validate    # Strip stale artifact IDs from focus_state.txt on each prompt
  - panels-context    # Tell the agent which web terminal panels exist
  - workspace-delta   # Report web workspace changes since the agent's last turn
  - control-context   # Tell the agent the active control target and each target's write state
  - turn-state        # Tell the web terminal when a turn starts and ends

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
  - channel-finder          # Finds channel addresses (the mode above decides how)
  - data-visualizer         # Produce strip charts and correlation plots
  - logbook-search          # Search facility logbook for historical entries
  - logbook-deep-research   # Multi-hop logbook research with synthesis
  - facility-knowledge      # Look up facility documentation, procedures, and device specs
  - facility-knowledge-graph  # Structural machine queries against the facility knowledge graph
  - pyat-specialist         # Lattice/optics computation sub-agent (pyAT)

output_styles:
  - control-operator  # Terse, actionable output style for control-room operators

# Tabs the web workspace offers beside the terminal. To turn on a panel
# the framework already ships, uncomment one listed under this key.
# A panel of your own is a list entry plus its address under `config:`:
#
#   web_panels:
#     - elog
#
# and, under `config:`, web.panels.elog.url alongside web.panels.elog.label,
# web.panels.elog.path and — optional — web.panels.elog.health_endpoint.
web_panels:
  - ariel           # ARIEL search interface (past experiments, papers)
  - channel-finder  # Interactive channel-finder web UI
  - okf             # KNOWLEDGE tab, for browsing the facility knowledge bundle
  - system-health   # SYSTEM tab, a framework health dashboard
  - jupyter         # JUPYTER tab, JupyterLab with kernels that follow the terminal session
  # The events and bluesky panels are declared by the write-armed personas
  # (readwrite and admin) instead, so the read-only login is built without them.
  # Available — uncomment to enable:
  # - lattice  # Lattice dashboard

# ── Scanning and simulated hardware ──────────────────────────────────────────
# These three blocks give you a working plan setup with no real hardware: a
# Bluesky bridge with a Tiled data catalog, a simulated accelerator that speaks
# EPICS, and the web panels for both. To drop it, delete this section, the
# bluesky panel above, AND the `claude_code.servers.bluesky.enabled: true`
# line under `config:` below — the block deploys the bridge, the line switches
# on the server that dials it, and the build refuses the line left on alone.
bluesky:
  # The bridge and its Tiled catalog publish inside this deployment's port
  # block, so neither needs a number here. Add `port:` or `tiled_port:` only to
  # pin one somewhere outside the block.
  tiled_enabled: true      # also runs the Tiled data catalog

virtual_accelerator:
  # EPICS port the simulator serves on. The agent follows this value, so
  # changing it moves both.
  port: 5064
  # A second copy of the simulator with a small fixed offset on its readouts,
  # stood up as this deployment's own third control target: `standin`. From
  # this key alone the build derives the target's connector block,
  # `control_system.connector.live_standin` — seven leaves, nothing else.
  # `control_target_set standin` points a session at it, and what an operator
  # meets there is a real machine's behaviour — approval prompts, strict-limit
  # refusals, the LIVE MACHINE (stand-in) banner — on something that cannot
  # move a magnet.
  # Delete the line on a laptop: the simulator image is amd64-only, so a
  # second emulated container on Apple Silicon doubles what QEMU has to run —
  # one more emulated soft-IOC for the life of the deployment.
  # `live` still means the machine YOU author under `epics:`, on a deployment
  # running a stand-in exactly as on one that is not — the build writes no key
  # in that block — so pointing this deployment at your facility is that one
  # edit and nothing here.
  # `osprey sim apply` moves both machines — a scenario changes the world, not
  # one lane. The archiver records the stand-in, and its seeded history carries
  # the same offsets: the archive belongs to the machine.
  # `true` serves the stand-in on this deployment's own stand-in port, so two
  # deployments on one host never collide over it. Write a Channel Access port
  # number instead only to pin it somewhere specific.
  live_standin: true

# The block's presence is the switch, and its only key — `port:` — comes from
# this deployment's port block, so there is nothing to write inside it.
bluesky_web: {}

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
#
# Not written here, because the build derives them: the project's paths, the
# provider catalog (providers.yml beside this file), the agent's, the
# semantic processor's and the logbook composer's provider and model (from
# `provider:` / `model:` above), the channel-finder pipeline block (from
# `channel_finder_mode:`), the panel selection (from `web_panels:`), every
# host port, and everything the `bluesky:`, `virtual_accelerator:`,
# `va_archiver:`, `dispatch:` and `mcp_servers:` sections above stand for.
config:
  # ── Facility ───────────────────────────────────────────────────────────────
  # Your facility's name, used in the agent's prompts and on the web landing
  # page. Defaults to the deployment name.
  # facility.name: My Facility
  # This facility's compiled ontology table, the JSON `osprey knowledge
  # compile-ontology` writes, relative to the project root. It is the one
  # source for the device vocabulary the channel-finder subagent's terminology
  # table renders. Drop the key and the subagent is told no vocabulary was
  # declared; point it at a missing file and the build stops and says so.
  facility.ontology: data/facility_ontology.json
  # Your facility's own registry module — the file that registers its
  # connectors, providers and ARIEL adapters with the framework (see
  # "Extending Osprey" in the docs). Relative to the project root; unset means
  # no application registry, and the framework-only one is used. `REGISTRY_PATH`
  # in the environment outranks this key, for a container pointed at a registry
  # mounted somewhere the config could not have named.
  # registry_path: project/registry.py

  # ── Control system ─────────────────────────────────────────────────────────
  # Which machine a session starts on. "live_standin" is the stand-in declared
  # above, so this deployment's baseline is a facility-shaped soft IOC that
  # behaves like hardware and moves nothing. "virtual_accelerator" is the
  # sandbox simulator, "epics" your own control system, "mock" needs no
  # containers but cannot complete a plan. "doocs" and "tango" reach those
  # control systems in place of Channel Access. Those five are every type
  # `osprey init` will materialize; `osprey config --defaults` lists them too.
  # `control_target_set live` moves a session onto the machine authored under
  # `epics:`. The template ships that block unconfigured — author its
  # `gateways` and `probe_channel` first — then the switch probes that target,
  # requires the strict limits pair below, then your own
  # `control_system.target_switch.live_gateway_acknowledged` — the operator
  # saying those gateways really are this facility's — and it still refuses
  # while this deployment records its own archive from the stand-in, because
  # that store's history is the stand-in's (see `va_archiver:` above).
  # `osprey set connector=epics` makes your facility's machine the session
  # baseline again, in place of the stand-in — together with
  # `osprey set config.archiver.type=epics_archiver` and
  # `osprey set va_archiver=null`, because the recorded archive goes with it.
  control_system.type: live_standin
  # Master write switch, the FIRST guard in the write-safety chain: while
  # false, every hardware write is refused before the limits check or the
  # approval prompt is consulted. On here because the baseline is the stand-in,
  # which cannot move a magnet; the read-only persona pins it off. Write
  # posture is per connector type: a `control_system.connector.<type>.
  # writes_enabled` overrides this for that type alone, and only a literal
  # `true` arms writes at either level.
  control_system.writes_enabled: true
  # Extra tools the kill switch refuses while writes are off. Framework write
  # tools are covered automatically; list your own servers' write tools here.
  # control_system.write_tools: [mcp__my_server__dangerous_write]
  # Operator-facing names for the control targets on the web terminal's
  # control-target chip. Defaults: "Real machine", "Rehearsal", "Simulator".
  # control_system.target_display_names.live: Real machine
  # control_system.target_display_names.standin: Rehearsal
  # control_system.target_display_names.va: Simulator
  # Every write checked against data/channel_limits.json, and a channel that
  # file does not list refused rather than waved through. Both hardware-shaped
  # targets — `standin` and `live` — require this pair before a session may
  # switch onto them, so a rehearsal runs the posture the real machine gets.
  control_system.limits_checking.enabled: true
  control_system.limits_checking.allow_unlisted_channels: false
  # The limits file, relative to the build directory. It is a build copy: edit
  # the one in data/ beside this file and rebuild.
  control_system.limits_checking.database_path: data/channel_limits.json
  # The sandbox simulator is the exception, and it states the exception as a
  # whole block: a per-type posture REPLACES the pair above for that connector
  # type rather than merging with it, so both leaves are written out here.
  # Writes to the simulator are still checked against the same file; what
  # changes is that a channel the file does not list is allowed through
  # instead of refused, because on a scratch machine an unlisted channel is a
  # gap in the file rather than a hazard. `live_standin` deliberately gets NO
  # block of its own: it is hardware-shaped, so it keeps the strict pair the
  # real machine gets, and a permissive block here would make
  # `control_target_set standin` refuse the very switch this preset exists to
  # rehearse.
  control_system.connector.virtual_accelerator.limits_checking.enabled: true
  control_system.connector.virtual_accelerator.limits_checking.allow_unlisted_channels: true
  # Largest array channel_read returns inline, in elements. Anything bigger is
  # saved as an artifact and reported as a summary plus a handle. One call
  # inlines at most 4x this many elements across all the channels it read;
  # past that budget a channel takes the artifact path and says so.
  control_system.read_inline_max_elements: 2000
  # Newest N UNPINNED read artifacts kept per channel; older ones are pruned
  # on save. Pinned entries are never pruned. 0 keeps everything.
  control_system.channel_read_artifact_retention: 20
  # Pattern detection: every control-system operation in generated code is
  # caught for approval, direct library calls included (epics.caput, .put()).
  # Extend the framework's patterns for a custom library; `override` REPLACES
  # them and drops that circumvention coverage.
  # control_system.patterns.mode: extend
  # control_system.patterns.write: ['my_custom_cs_lib\.write\(']
  # control_system.patterns.read: ['my_custom_cs_lib\.read\(']
  #
  # Mock connector: driven by the simulation machine model below. Switch
  # scenarios with `osprey sim apply NAME...` (see the sim-scenarios skill).
  control_system.connector.mock.simulation_file: data/simulation/machine.json
  # Virtual-accelerator connector: a containerized PyAT-backed soft IOC
  # reached over real EPICS Channel Access, with the same gateway shape as the
  # `epics` block. Deployed by the `virtual_accelerator:` section above.
  #
  # Channel Access timeout in seconds.
  control_system.connector.virtual_accelerator.timeout: 5.0
  # Same machine model as the mock connector, so `osprey sim apply` stays
  # consistent whichever connector is active.
  control_system.connector.virtual_accelerator.simulation_file: data/simulation/machine.json
  # Fractional noise on the synthesised readings. Unset falls through to
  # `control_system.connector.mock.noise_level`, and then to the simulator's
  # own 0.01; `0` serves the channels flat, which is what a comparison of two
  # reads wants. It reaches the container as VA_NOISE_LEVEL.
  # control_system.connector.virtual_accelerator.noise_level: 0.01
  # Write posture for the simulator alone. Uncomment to arm writes here while
  # the master switch keeps the live machine read-only; the shipped
  # `control-assistant-va-readwrite` persona is exactly this key.
  # control_system.connector.virtual_accelerator.writes_enabled: true
  # Channel the target switch reads to prove this target is reachable before
  # making it active. Served by the simulation machine model.
  control_system.connector.virtual_accelerator.probe_channel: SR:VAC:GAUGE:SR01:PRESSURE:RB
  # Gateways in CA name-server (TCP) mode against localhost, the one
  # host-to-container configuration that works across container runtimes. No
  # port is written: the connector follows `services.virtual_accelerator.port`,
  # so moving the deployed soft IOC is a one-place edit. Set a port on a
  # gateway only to reach a VA this deployment does not run.
  control_system.connector.virtual_accelerator.gateways.read_only.address: localhost
  control_system.connector.virtual_accelerator.gateways.read_only.use_name_server: true
  # The write lane, same host and mode as the read lane.
  control_system.connector.virtual_accelerator.gateways.write_access.address: localhost
  control_system.connector.virtual_accelerator.gateways.write_access.use_name_server: true
  # EPICS connector for the live machine: ships unconfigured on purpose. A
  # facility's gateways cannot be guessed, and shipping someone else's would
  # make the `live` target look ready while pointing at hardware you never
  # configured. Authoring the gateways and probe channel below is the go-live
  # edit; until then the live target shows as not configured.
  #
  # Channel Access timeout in seconds.
  control_system.connector.epics.timeout: 5.0
  # How long the pre-write `max_step` check waits for a channel's present
  # value, in seconds. Running out of budget refuses the write, so raise this
  # for a slow gateway — it buys room, never a weaker check.
  # control_system.connector.epics.step_read_timeout_s: 2.0
  # Write posture for the live machine. Stating it pins it: a type with its
  # own posture never falls back to the master switch.
  # control_system.connector.epics.writes_enabled: false
  # Limits posture for the live machine alone, both leaves required.
  # control_system.connector.epics.limits_checking.enabled: true
  # control_system.connector.epics.limits_checking.allow_unlisted_channels: false
  # Channel the target switch reads to prove the live machine is reachable.
  # While unset this target is never switched to.
  # control_system.connector.epics.probe_channel: SR:BEAM:CURRENT
  # Your facility's Channel Access gateways. use_name_server: true for SSH
  # tunnels (EPICS_CA_NAME_SERVERS), false for a direct gateway
  # (EPICS_CA_ADDR_LIST).
  # control_system.connector.epics.gateways.read_only.address: your-ca-gateway.example.com
  # control_system.connector.epics.gateways.read_only.port: 5064
  # control_system.connector.epics.gateways.read_only.use_name_server: false
  # control_system.connector.epics.gateways.write_access.address: your-ca-gateway.example.com
  # control_system.connector.epics.gateways.write_access.port: 5084
  # control_system.connector.epics.gateways.write_access.use_name_server: false
  # PVAccess read routing: addresses matching these globs are read through
  # p4p (the transport camera frames arrive on); everything else stays on
  # Channel Access. Read-only, and `pva_gateway` is the ONLY route: EPICS_PVA_*
  # variables never reach the connector. `address` maps to
  # EPICS_PVA_ADDR_LIST (UDP search, default port 5076), or to
  # EPICS_PVA_NAME_SERVERS when use_name_server is true (TCP, default 5075).
  # control_system.connector.epics.pva_channels: ["*:IMAGE*", "*:ARRAY*"]
  # control_system.connector.epics.pva_gateway.address: your-pva-gateway.example.com
  # control_system.connector.epics.pva_gateway.use_name_server: false
  # DOOCS connector: no coordinates of its own. doocs4py reaches the ENS the
  # facility's own DOOCS environment already names, so the empty coordinate set
  # is the point of this block rather than an omission — what is left is the
  # four leaves every connector type answers.
  # Write posture for the DOOCS machine. Same tri-state as the epics leaf
  # above: stating it pins it, and only a literal true arms writes.
  # control_system.connector.doocs.writes_enabled: false
  # Limits posture for the DOOCS machine alone, both leaves required.
  # control_system.connector.doocs.limits_checking.enabled: true
  # control_system.connector.doocs.limits_checking.allow_unlisted_channels: false
  # Property the target switch reads to prove this machine is reachable.
  # control_system.connector.doocs.probe_channel: FACILITY/DEVICE/LOCATION/PROPERTY
  # TANGO connector: one coordinate of its own, the device database, spelled
  # `host:port`. Leave `tango_host` unset and PyTango reads the TANGO_HOST the
  # environment already carries; write it to name a database that environment
  # does not. The port is the database's own, not one of this deployment's.
  # control_system.connector.tango.tango_host: your-tango-db.example.com:<db-port>
  # Seconds a device call waits before it is given up on.
  # control_system.connector.tango.timeout: 5.0
  # The same four leaves as every other connector type, for the TANGO machine.
  # control_system.connector.tango.writes_enabled: false
  # control_system.connector.tango.limits_checking.enabled: true
  # control_system.connector.tango.limits_checking.allow_unlisted_channels: false
  # control_system.connector.tango.probe_channel: sys/tg_test/1/ampli
  # Target switch: how a running session moves between the connectors above.
  #
  # Seconds in-flight operations get to finish on the old target before it is
  # torn down regardless.
  control_system.target_switch.drain_timeout_s: 5
  # Seconds between background reachability probes of every target's gateways.
  control_system.target_switch.probe_interval_s: 30
  # Operator acknowledgment for the live machine: set it to your own live
  # gateway's hostname to confirm the `epics` gateways above really are your
  # facility's. While unset a session may not switch TO the live target.
  # control_system.target_switch.live_gateway_acknowledged: your-ca-gateway.example.com

  # ── Archiver ───────────────────────────────────────────────────────────────
  # Use the archive declared by `va_archiver:` above. Declaring the block does
  # not turn it on; without this line you would deploy a store and not read it.
  # The store's coordinates (`archiver.mongodb_archiver.*`) are derived from
  # that block, so they are not written here. The alternatives are
  # "mock_archiver" (synthesized history), "epics_archiver" (an Archiver
  # Appliance, configured below) and "doocs_archiver" (DOOCS local history).
  archiver.type: mongodb_archiver
  # When a read names no bin size, the bin is chosen so a continuously archived
  # channel returns about this many points. The agent is told which bin it got.
  archiver.auto_bin_points: 10000
  # Mock archiver: synthesizes history from the same simulation machine model
  # as the control-system connector, derived from
  # `control_system.connector.<type>.simulation_file`. Set only to override.
  # archiver.mock_archiver.simulation_file: data/simulation/machine.json
  # EPICS Archiver Appliance: ships unconfigured on purpose, for the same
  # reason as the `epics` gateways. Authoring it travels with the flip to
  # `archiver.type: epics_archiver`.
  # archiver.epics_archiver.url: https://your-archiver.example.com:8443
  # archiver.epics_archiver.timeout: 60
  # MongoDB archiver pointed at a store this deployment does NOT run. The
  # coordinates above are derived from `va_archiver:`; spell them here instead
  # to read an archive someone else keeps, and drop the `va_archiver:` block so
  # this deployment does not record a second history beside it.
  # archiver.mongodb_archiver.host: your-mongo.example.com
  # archiver.mongodb_archiver.port: 27017
  # archiver.mongodb_archiver.name: your-archive-database
  # archiver.mongodb_archiver.collection: your-archive-collection
  # archiver.mongodb_archiver.auth: your-auth-database
  # archiver.mongodb_archiver.username: your-readonly-user
  # archiver.mongodb_archiver.password_env: OSPREY_ARCHIVER_PASSWORD
  # archiver.mongodb_archiver.timeout: 60
  # DOOCS local history: like the DOOCS connector it takes no coordinates and
  # reaches the ENS the environment names. Both knobs are optional — a centered
  # moving average over this many seconds, and the read budget.
  # archiver.doocs_archiver.avg_window: 20
  # archiver.doocs_archiver.timeout: 60

  # ── Scan plans (Bluesky) ───────────────────────────────────────────────────
  # Both servers are off by default in OSPREY. Turn them on so the agent can
  # write and launch plans, and run read-only health checks.
  claude_code.servers.bluesky.enabled: true
  claude_code.servers.health.enabled: true
  # The plan queue runs by default: a plan added to it runs as soon as the
  # queue reaches it, with no separate Start. Stop and Abort disarm the queue
  # until the next Start. Delete the line (or set it false) for a queue that
  # comes up stopped and drains only after a Start.
  bluesky.queue_autostart: true
  # Extra plan directories, as IN-CONTAINER paths (the bridge mounts this
  # config at /app/project/config.yml). Plans found here load at the `preset`
  # trust tier and are exec'd on discovery with no review: a directory listed
  # here is code you choose to run. To publish a host directory instead, use
  # the `bluesky.plan_dir` field above.
  # bluesky.plan_dirs: [/app/project/extra_plans]

  # ── Channel finder ─────────────────────────────────────────────────────────
  # The pipeline itself is derived from `channel_finder_mode:` above. When that
  # field selects the graph paradigm it answers from the `services.graphdb.*`
  # store below.
  #
  # The scored benchmark query set every paradigm reads, materialized by the
  # build; `osprey channel-finder benchmark` runs it (`--queries-path FILE`
  # for another).
  channel_finder.benchmark.dataset_path: data/benchmarks/queries.json
  # Rows one `run_sql` answer carries before it is cut. The cap is on the
  # AGENT's context, not on DuckDB: a truncated answer says so and names this
  # key, so the agent narrows the query rather than presenting a partial list.
  channel_finder.query_max_rows: 500
  # osprey:panel-port channel_finder
  # The CHANNELS tab's own web server. It launches when `channel-finder` is in
  # `web_panels:` above, on this deployment's channel-finder slot;
  # OSPREY_CHANNEL_FINDER_PORT or the port key below override it. `host` has
  # no env override. Uncomment to move or disable it.
  # channel_finder.web.host: 127.0.0.1
  # channel_finder.web.port: <a port outside this deployment's block>
  # channel_finder.web.auto_launch: true
  # Descriptive names for channels that belong to no device family, generated
  # offline by `osprey channel-finder build-database --use-llm`. That flag
  # needs `provider` set (no fallback to the agent's provider); `model_id` is a
  # tier or a model ID the provider serves. Build-time only.
  # channel_finder.channel_name_generation.llm_model.provider: anthropic
  # channel_finder.channel_name_generation.llm_model.model_id: haiku
  # channel_finder.channel_name_generation.llm_model.max_tokens: 1000
  # channel_finder.channel_name_generation.llm_batch_size: 10

  # ── Human-in-the-loop approval ─────────────────────────────────────────────
  # The THIRD guard: a write that passed the master switch and the limits check
  # still pauses for a yes/no prompt. Applied by the approval hook, so it
  # reaches only hook-wired tools; the health check, channel-finder queries and
  # most workspace tools are gated by the rendered settings.json permissions.
  approval.enabled: true
  # Policy for any hook-wired tool not listed below. "always" is fail-closed:
  # reads need their own "skip" entry to run unprompted.
  approval.default_policy: always
  # Per-tool policies: always (prompt every time), skip (no prompt), or
  # selective (content-aware, for `execute` only; other tools read it as
  # always).
  approval.tools.channel_write: always
  approval.tools.channel_read: skip
  approval.tools.archiver_read: skip
  # Python execution: content-aware, so a script that only reads runs unprompted.
  approval.tools.execute: selective
  # The agent's own config-editing tool: a config change needs approval.
  approval.tools.setup_patch: always
  # Creating a logbook entry always asks first.
  approval.tools.entry_create: always
  # Publishing it through to the facility's logbook does too.
  approval.tools.entry_publish: always
  # The panel rail is the operator's own view. Adding or removing a panel, and
  # registering a new one, changes what the next person sees, so each asks.
  approval.tools.add_panel_to_rail: always
  approval.tools.remove_panel_from_rail: always
  approval.tools.register_panel: always

  # ── Hook observability ─────────────────────────────────────────────────────
  # On here. Every hook call logs one line to stderr and appends to
  # .claude/hooks/hook_debug.jsonl (never rotated: prune it yourself), which
  # is what the web terminal's Safety panel hook feed reads. OSPREY_HOOK_DEBUG
  # in the environment forces it on regardless of this key.
  hooks.debug: true

  # ── ARIEL logbook search ───────────────────────────────────────────────────
  # No `ariel.database.uri`: the DSN is derived from `services.postgresql.*`
  # below, so moving the database stays a one-place edit. Set it only to point
  # ARIEL at a Postgres this deployment does not run; an explicit uri wins.
  # ariel.database.uri: postgresql://ariel:${ARIEL_DB_PASSWORD}@logbook-db.example.org:5432/ariel
  # No `ariel.ingestion` block: the logbook is seeded from the simulation
  # scenario bundles by `osprey sim apply NAME...`. For production, add
  # `ariel.ingestion.adapter` and `ariel.ingestion.source_url` for your
  # logbook system and use `osprey ariel ingest`. That fetch verifies the
  # logbook's certificate by default: name your site CA with
  # `ariel.ingestion.ca_bundle` when it is not in the image trust store, and
  # keep `ariel.ingestion.verify_ssl: false` for a certificate that cannot be
  # verified at all.
  # osprey:panel-port ariel
  # The ARIEL tab's own web server. It launches when `ariel` is in
  # `web_panels:` above, on this deployment's ARIEL slot; OSPREY_ARIEL_PORT or
  # the port key below override it. `host` has no env override.
  # ariel.web.host: 127.0.0.1
  # ariel.web.port: <a port outside this deployment's block>
  # ariel.web.auto_launch: true
  # Which module answers a search that names no mode: the web interface's
  # opening tab, `osprey ariel search` without `--mode`, and the service API.
  # Naming a module that is off below is refused at startup.
  ariel.default_search_mode: hybrid
  # Largest file one entry may attach, in MB. Attachments are stored as rows in
  # the same Postgres the logbook lives in, so this number is a storage
  # decision in both directions.
  # ariel.attachments.max_file_mb: 10
  # Facility vocabulary: control-room shorthand ("t/s the bpm offset") mapped
  # to the words the logbook prose contains, so a search typed in shorthand
  # finds the entries about it. Plain dictionary matching, every rewrite
  # reported back as `expanded_terms`. The file is read once at startup; a
  # broken one fails loudly there (panel in CONFIGURATION INVALID mode, search
  # 503, MCP server refuses to start). Check edits with
  # `osprey ariel vocab-check data/ariel/vocabulary.yml`.
  ariel.vocabulary.enabled: true
  # A twenty-concept EXAMPLE covering what most storage-ring facilities share
  # (a linac or FEL should delete the orbit-and-ring group). A starting point,
  # not your vocabulary: edit it. Relative to the project root.
  ariel.vocabulary.path: data/ariel/vocabulary.yml
  # Whether a search that expresses no preference gets expansion. The
  # per-request `expand_query` argument overrides it either way.
  ariel.vocabulary.expand_by_default: true
  # Reverse expansion gates. Matching a form and adding its canonical is
  # always on (`bpm` finds "beam position monitor"); these decide whether a
  # spelled-out canonical also searches its short form. Free recall for an
  # acronym, noise for an ordinary word ("calibration" → "cal").
  ariel.vocabulary.canonical_to_acronym: true
  ariel.vocabulary.canonical_to_shorthand: false
  # No `ariel.vocabulary.expand_modes`: unset, every enabled search module
  # expands. Set `[keyword, semantic]` to drop `hybrid` alone if the reranked
  # ordering degrades under expansion.
  #
  # Search modules. Knobs MUST sit under `settings`: the loader keeps only
  # `enabled`, `provider`, `model` and `settings` and drops any other key
  # silently. Entries seeded by `osprey sim apply` carry no embeddings: run
  # `osprey ariel migrate` then `osprey ariel enhance` to recover semantic
  # search; hybrid search answers immediately from the exported mirror.
  #
  # Keyword search over Postgres.
  ariel.search_modules.keyword.enabled: true
  # Pattern tokens in a keyword query: `*` globs and explicit `/regex/`, run
  # as case-insensitive matches (a glob is also anchored to word boundaries).
  # Both need 3 consecutive literal characters to use the index. Set false to
  # make `*` and `/…/` plain words again.
  ariel.search_modules.keyword.settings.patterns_enabled: true
  # Wall-clock envelope for a pattern search, in seconds. A pattern that cannot
  # use the index scans the whole logbook; this returns a timeout diagnostic
  # instead of holding the panel open.
  ariel.search_modules.keyword.settings.pattern_timeout_seconds: 10.0
  # Semantic search over pgvector embeddings. Degrades to keyword-only when
  # Ollama or pgvector is unavailable, and says so.
  ariel.search_modules.semantic.enabled: true
  # Embedding provider (an entry in providers.yml) and model.
  ariel.search_modules.semantic.provider: ollama
  ariel.search_modules.semantic.model: nomic-embed-text
  # Hybrid keyword+semantic search answered by the qmd sidecar: the
  # best-ranked retrieval this preset ships, and the only semantic-quality
  # mode that needs nothing on the host. It needs `services.qmd.*` below and
  # the `qmd_export` enhancement; switch all three off together or not at
  # all. Unlike semantic search it does not degrade: a query against a
  # missing sidecar is reported as "search is down".
  ariel.search_modules.hybrid.enabled: true
  # qmd's LLM reranker: an LLM reviews every candidate, so it dominates query
  # latency. One key for both surfaces, the agent's hybrid_search tool and the
  # ARIEL panel; each can override it per call or per session.
  ariel.search_modules.hybrid.settings.rerank: true
  # Candidates the reranker considers. Lowering it trades recall for latency.
  ariel.search_modules.hybrid.settings.candidate_limit: 40
  # Enhancement modules, run during ingestion.
  #
  # Semantic processor: LLM keyword extraction and summarisation. Off because
  # it costs LLM calls; the provider and model follow `provider:` / `model:`.
  ariel.enhancement_modules.semantic_processor.enabled: false
  # Token budget for each of its LLM calls.
  ariel.enhancement_modules.semantic_processor.model.max_tokens: 256
  # How much of an entry is sent for keywords and summary. Longer entries are
  # cut here — the cut is logged, naming the entry — so this is what a facility
  # with long entries and a roomy context window raises.
  # ariel.enhancement_modules.semantic_processor.max_input_chars: 8000
  # The extraction prompt. Unset means the module's own default, whose examples
  # are categories rather than devices. Set it to put this facility's vocabulary
  # in front of the model; the replacement must keep the {text} placeholder and
  # the JSON schema the module parses.
  # ariel.enhancement_modules.semantic_processor.prompt_template: |
  # Text embedding for semantic search. Degrades gracefully when Ollama or
  # pgvector is unavailable.
  ariel.enhancement_modules.text_embedding.enabled: true
  ariel.enhancement_modules.text_embedding.provider: ollama
  # Embedding models and their vector dimension.
  ariel.enhancement_modules.text_embedding.models:
    - name: nomic-embed-text
      dimension: 768
  # IVFFlat `lists` for the vector index, chosen when the index is created.
  # Rule of thumb: rows/1000 for corpora up to a million entries. It cannot be
  # derived — `osprey ariel migrate` runs against an empty table — and changing
  # it later needs the index dropped and recreated.
  # ariel.enhancement_modules.text_embedding.index_lists: 224
  # qmd export: one markdown file per entry into the mirror tree the sidecar
  # indexes. On for the same reason `hybrid` above is; an enabled export with
  # no mirror_path is refused at startup.
  ariel.enhancement_modules.qmd_export.enabled: true
  # The mirror tree, relative to the project root. Under var/ (the durable
  # STATE zone, kept out of git): it is machine-written and as large as the
  # logbook. The compose generator binds this same path into the sidecar.
  ariel.enhancement_modules.qmd_export.settings.mirror_path: var/ariel_mirror
  # Default provider for EMBEDDING modules that name none of their own. Not a
  # general fallback: the semantic processor's provider is derived separately.
  ariel.embedding.provider: ollama

  # ── Logbook composition ────────────────────────────────────────────────────
  # The compose panel in the artifact gallery. Its provider follows
  # `provider:` above; this is the tier used when the operator picks none
  # (haiku | sonnet | opus), mapped to a model ID through providers.yml.
  logbook.composition.default_tier: haiku

  # ── Facility knowledge ─────────────────────────────────────────────────────
  # OKF bundle (subsystems, devices, procedures, physics notes) behind the
  # facility_knowledge server, `osprey knowledge` and the KNOWLEDGE tab.
  # Relative to the project root. Replace with your own bundle once you have
  # customised the example content.
  facility_knowledge.bundle_path: data/facility_knowledge
  # osprey:panel-port okf
  # The KNOWLEDGE tab's own web server. It launches when `okf` is in
  # `web_panels:` above, on this deployment's knowledge slot. Uncomment to
  # move or disable it.
  # facility_knowledge.host: 127.0.0.1
  # facility_knowledge.port: <a port outside this deployment's block>
  # facility_knowledge.auto_launch: true

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
  # Override model IDs per tier, or the tier one agent runs at.
  # claude_code.models.haiku: anthropic/claude-haiku-alt
  # claude_code.agent_models.logbook-search: haiku
  # claude_code.agent_models.logbook-deep-research: sonnet
  # Switch a framework server or subagent off, or add an MCP server of your
  # own (the `mcp_servers:` field above is the usual home for one).
  # claude_code.servers.python.enabled: false
  # claude_code.agents.logbook-search.enabled: false

  # ── Telemetry ──────────────────────────────────────────────────────────────
  # The agent emits OTLP logs and metrics to the OpenObserve store this
  # deployment runs (`services.openobserve.*` below). On by default: the
  # harness already records every prompt and API body to disk, so this adds no
  # exposure, only a queryable local store you own.
  claude_code.telemetry.enabled: true
  # openobserve | generic
  claude_code.telemetry.backend: openobserve
  # http/protobuf | grpc. grpc needs an explicit `claude_code.telemetry.endpoint`
  # and is refused against the auto-derived openobserve endpoint (HTTP only).
  claude_code.telemetry.protocol: http/protobuf
  # No endpoint key: with backend openobserve it is derived per network
  # context (the host's OpenObserve slot, or the store's own listen port
  # inside the deploy network), so the in-container dispatch worker does not
  # emit to its own loopback.
  # The store's INGEST account: `osprey up` creates a service account named by
  # ZO_INGEST_USER_EMAIL and writes the token it issues to this repo's .env as
  # ZO_INGEST_SA_TOKEN. No default for the token on purpose: a literal default
  # would be a published credential.
  claude_code.telemetry.openobserve.user: ${ZO_INGEST_USER_EMAIL:-ingest@example.com}
  claude_code.telemetry.openobserve.password: ${ZO_INGEST_SA_TOKEN}
  # OpenObserve organisation the records land in.
  claude_code.telemetry.openobserve.org: default
  # Content gates, all on. The store captures full transcripts behind the
  # ZO_ROOT_USER_PASSWORD `osprey up` writes into .env; anyone with that
  # password and a route to the host can read everything. Set any gate to
  # false to keep that category out of the emitted telemetry.
  claude_code.telemetry.log_user_prompts: true
  claude_code.telemetry.log_assistant_responses: true
  claude_code.telemetry.log_tool_details: true
  # Raw provider request and response bodies.
  claude_code.telemetry.log_raw_api_bodies: true

  # ── Services ───────────────────────────────────────────────────────────────
  # Containerized companion services. Declare one as `services.<name>.*` and
  # add its name to `deployed_services` below to launch it with `osprey up`.
  # The `bluesky:`, `virtual_accelerator:`, `va_archiver:` and `dispatch:`
  # sections above add their own services at build time. No host ports are
  # written here: each service publishes on its fixed slot above
  # `deployment.port_base` (set that key to move the whole block). Add
  # `services.<name>.port` (`port_host` for the stores) only to pin one
  # outside it.
  #
  # PostgreSQL backs ARIEL logbook search. Compose directory, under build/.
  services.postgresql.path: ./services/postgresql
  # Database and role ARIEL connects as. No password key: it lives in this
  # repo's .env as ARIEL_DB_PASSWORD (minted by `osprey up`), read by both the
  # container and the agent's ARIEL DSN, which is derived from this block.
  services.postgresql.database_name: ariel
  services.postgresql.username: ariel
  # OpenObserve, the local telemetry store the agent emits to (see
  # `claude_code.telemetry.*` above). `osprey up` mints a strong
  # ZO_ROOT_USER_PASSWORD into .env. Compose directory, under build/.
  services.openobserve.path: ./services/openobserve
  # Growth bound: drop telemetry older than N days (min 3). A named volume has
  # no size cap, so age is the size knob; `osprey health` warns as the backing
  # disk fills.
  services.openobserve.retention_days: 14
  # qmd, the semantic-search sidecar. It indexes the facility-knowledge bundle
  # and the ARIEL markdown mirror that `ariel.enhancement_modules.qmd_export`
  # writes, and answers ranked KNOWLEDGE search and the `hybrid` logbook mode
  # over HTTP, with its language models baked into the image (no Ollama
  # needed). The image is built locally on the first `osprey up`; the ~2.1 GB
  # of models make that first build long. It publishes on
  # `deployment.bind_address` with no token and no TLS, so leave that on
  # loopback. To run without it, remove this key, `services.qmd.interval`, the
  # `qmd` entry below and the two ARIEL consumers (`qmd_export`, `hybrid`).
  services.qmd.path: ./services/qmd
  # Fallback corpus-sweep period in seconds. The bundle's `.qmd-touch` marker
  # is the primary re-index trigger, so this is the ceiling on staleness.
  # Raise it on a large corpus, where a no-op sweep is not free.
  services.qmd.interval: 30
  # How long the container's healthcheck holds off, in seconds, while the first
  # full index is built — the sidecar does not open its port until the index
  # exists and is non-empty, so until then it is legitimately unhealthy. Scale
  # it with the corpus: too short reports a working container as failed.
  # services.qmd.first_index_grace: 3600
  # Neo4j graph store holding a DISPOSABLE mirror of an RDF/Turtle corpus. The
  # TTL on disk stays the source of truth and `osprey knowledge seed-graph`
  # rebuilds the graph from it. It answers the multi-hop questions keyword and
  # semantic search cannot, and it is what the channel finder reads when
  # `channel_finder_mode:` selects the graph paradigm.
  # No password key: the container reads GRAPHDB_PASSWORD from this repo's
  # .env, minted by `osprey up`.
  services.graphdb.path: ./services/graphdb
  # Pinned to the 5.26 LTS line: neosemantics (n10s), the plugin that imports
  # the RDF, has no manifest entry for anything newer. Repoint it at a mirror
  # or a pre-baked image on an air-gapped host.
  services.graphdb.image: neo4j:5.26-community
  # Corpus to seed the store from, relative to the build directory. This is
  # the demo machine: the same devices and channels the channel database in
  # data/ describes, rebuilt as a graph, so graph answers and channel search
  # agree. Regenerate it after editing the channel database with
  # `osprey knowledge build-ttl data/demo_machine.ttl`. Point it at your own
  # TTL, or remove the key to bring the store up bootstrapped but empty.
  services.graphdb.ttl_path: ./data/demo_machine.ttl
  # Search index derived from the corpus above at build time, and the default —
  # uncomment only to move it. The channel explorer's search, the channel roster
  # and the agent's keyword tool read this file rather than querying the store,
  # so it answers in milliseconds at any corpus size. Resolved against the
  # render like `ttl_path`; rebuild it by hand after regenerating the TTL with
  # `osprey knowledge build-index`.
  # services.graphdb.index_path: ./data/channel_databases/graph.duckdb
  # JVM memory. Neo4j sizes nothing automatically inside a container, so all
  # three are spelled out. Budget roughly heap_max_size + pagecache_size +
  # ~0.5G overhead; a substantially larger graph wants more.
  services.graphdb.heap_initial_size: 512m
  services.graphdb.heap_max_size: 1G
  # Off-heap cache for graph data and indexes, separate from the heap.
  services.graphdb.pagecache_size: 512m
  # Bounds on ONE agent query through the `read_cypher` tool: the server-side
  # transaction timeout in seconds, and the rows returned before truncation
  # (the tool says when it cut). Raising them spends the agent's context.
  services.graphdb.query_timeout_s: 15
  services.graphdb.query_max_rows: 200
  # To use a graph store this deployment does not run, name it here and take
  # `graphdb` out of `deployed_services`. Nothing is minted on that path: set
  # GRAPHDB_PASSWORD in this repo's .env yourself.
  # services.graphdb.uri: bolt://graph.example.org:7687
  # services.graphdb.username: neo4j
  # Which database on that store holds the corpus. The store this deployment
  # runs serves exactly one, called `neo4j`; a cluster of your own may not.
  # services.graphdb.database: neo4j
  # Which declared services `osprey up` launches. qmd and graphdb each go
  # together with their `services.<name>.*` keys above: remove both or neither.
  deployed_services:
    - postgresql
    - openobserve
    - qmd
    - graphdb
  # Host interface the services publish on. 127.0.0.1 keeps every port
  # loopback-only, the safe state; 0.0.0.0 exposes them to the network.
  # deployment.bind_address: 127.0.0.1

  # ── Web terminal ───────────────────────────────────────────────────────────
  # Every panel tab is served by its own small web server, configured under its
  # own section: WORKSPACE `artifact_server`, ARIEL `ariel.web`, CHANNELS
  # `channel_finder.web`, LATTICE `lattice_dashboard`, KNOWLEDGE
  # `facility_knowledge`, SYSTEM `health.web`. Each defaults to its slot above
  # `deployment.port_base`; the panel's env var wins, then its `port` key.
  # Multi-user deployments export the env var per user. The reference guide's
  # ports page prints the table.
  #
  # osprey:panel-port artifact
  # The WORKSPACE tab's artifact server. Under `osprey web` it launches at
  # startup whatever the panel list says; `auto_launch: false` is the only
  # thing that stops it. `host` has no env override.
  artifact_server.host: 127.0.0.1
  artifact_server.auto_launch: true
  # Largest timeseries data file the gallery will chart or tabulate, in MB.
  # The handler loads the whole file to build the view, so raising this spends
  # memory on the machine serving the gallery. Over the cap the browser views
  # refuse with a 413; the file itself stays downloadable either way.
  # artifact_server.max_timeseries_file_mb: 200
  # Extra artifact categories on top of the ones the gallery ships, so a badge
  # reads in this facility's own vocabulary. One dotted line per category, each
  # value a `label` and a `#RRGGBB` `color`. An artifact handed in under a
  # category nobody declared is still stored — it just keeps the default badge,
  # and the save logs a warning naming this key.
  # artifact_server.categories.beam_diagnostics: {label: Beam Diagnostics, color: "#f59e0b"}
  # Seed one shipped example (an interactive plot, synthetic data) into an
  # empty WORKSPACE on the gallery's first start. Deleting it there is permanent.
  artifact_server.example_artifact: true
  # osprey:panel-port lattice_dashboard
  # The LATTICE tab never auto-launches here: it needs this section and
  # `lattice` in `web_panels:` above.
  # lattice_dashboard.host: 127.0.0.1
  # lattice_dashboard.port: <a port outside this deployment's block>
  # lattice_dashboard.auto_launch: true
  # osprey:panel-port system_health
  # The SYSTEM tab needs no section: with `system-health` in `web_panels:` it
  # binds its slot. Uncomment only to move it or switch it off.
  # health.web.host: 127.0.0.1
  # health.web.port: <a port outside this deployment's block>
  # health.web.auto_launch: true
  # The terminal process itself (`osprey web`), every key at its default. The
  # multi-user compose sets OSPREY_TERMINAL_BIND_HOST on every container, which
  # outranks `host`; `shell` REPLACES the launcher and defeats the CLI pin.
  # `shell` is argv, written either way — a string (quoting honoured) or a
  # list; only the first word is resolved to an absolute path and the rest are
  # passed through:
  #   web_terminal.shell: /opt/harness/run --profile ops
  #   web_terminal.shell: ["/opt/harness/run", "--profile", "ops"]
  # The PTY spawn still appends `--session-id`/`--resume` and `--effort` to
  # whatever is set here, so a harness that does not take those flags needs a
  # wrapper that drops them.
  # web_terminal.host: 127.0.0.1
  # web_terminal.port: <a port outside this deployment's block>
  # web_terminal.max_background_sessions: 5
  # web_terminal.watch_dir: var/agent_data
  # Starting theme for every web terminal. A family ("main", "desy",
  # "high-contrast", "retro") leaves light/dark to the viewer's OS; a concrete
  # id ("desy-light") pins it. Each browser can override it from the display
  # menu, and a roster entry's `theme:` overrides it per user.
  web.theme: light
  # Who gets offered the onboarding tour, and how often. `once` (the default)
  # invites until a browser dismisses it or finishes the tour; `always` invites
  # on every load and offers no permanent dismissal, which is what a shared
  # read-only screen wants; `never` offers nothing and leaves the tour on the
  # rail's Tour control and the command palette. A roster entry's `tour:` field
  # overrides it per user.
  # web.tour: once
  # Target of the Documentation button. Point it at a locally hosted copy of
  # the docs when the control room has no route to the public site. Commented
  # rather than shipped live: a rendered value would put the OSPREY project's
  # own documentation site into this deployment's profile.yml as though the
  # facility had chosen it. Unset, the button points at the published site.
  # web.docs_url: https://docs.example.org/osprey
  # The Feedback dialog's outbound channels. Nothing is posted for the user:
  # the browser opens a prefilled issue form or mail draft. Every submission
  # is also recorded here (`osprey feedback list` / `export`).
  #
  # Reports go to ONE destination: whoever owns this deployment. A user is
  # not asked whether a bug is OSPREY's or this facility's configuration.
  # Left unconfigured, the owner is the OSPREY project itself. A facility
  # that names an owner becomes the destination, and the reports it
  # receives carry a one-click link for forwarding a framework bug
  # upstream. One block, because the address and the tracker have to move
  # together: redirecting the mail but leaving the tracker upstream sends
  # half the reports to strangers.
  # web.feedback.owner:
  #   name: Example Controls
  #   email: controls@example.org
  #   tracker:
  #     kind: gitlab            # gitlab | github
  #     target: https://git.example.org/controls/osprey
  #                             # gitlab: the project URL; github: owner/name
  #     label: Controls GitLab  # optional caption for the radio
  # owner/repo whose new-issue form the GitHub channel prefills. Predates
  # `owner` above and still wins over it wherever it is spelled, so an
  # existing deployment keeps meaning what it meant; "" offers no GitHub
  # channel. Commented rather than shipped live: a rendered value would put
  # the OSPREY project's own tracker into this deployment's profile.yml as
  # though the facility had chosen it.
  # web.feedback.github_repo: my-org/controls
  # Further trackers, one channel each: a `gitlab` entry takes the project's
  # base URL, a `github` entry owner/repo; `label` captions it.
  # web.feedback.trackers:
  #   - kind: gitlab
  #     url: https://git.example.org/controls/osprey
  #     label: Facility GitLab
  # Recipient of the prefilled mailto: draft the Email channel opens. Wins
  # over `owner.email` above, on the same grounds as `github_repo`, and
  # commented for the same reason.
  # web.feedback.email: controls@example.org
  # Ceiling in bytes on the on-disk feedback store (256 MB). Above it the
  # oldest saved session contexts are deleted; submission headers are kept.
  web.feedback.max_store_bytes: 268435456
  # Channel-name typeahead in the web panels: the build snapshots the names in
  # this deployment's channel-finder database next to the generated config, so
  # a form field can complete what an operator types. No control-system
  # traffic and nothing to sync at run time.
  web.channel_suggestions.enabled: true
  # Guards the browser, not the build: every panel load fetches the whole
  # snapshot, so a database with more channels than this is skipped (the build
  # says so) and the fields simply offer no suggestions.
  web.channel_suggestions.max_channels: 50000
  # Custom panels are `web.panels.<id>.*` keys here; the built-in tabs are
  # switched by `web_panels:` above. `rewrite_json_paths` opts a backend's
  # JSON bootstrap endpoints into the reverse proxy's path rewrite.
  # web.panels.my-grafana.label: GRAFANA
  # web.panels.my-grafana.url: http://grafana.local:3000
  # web.panels.my-grafana.health_endpoint: /api/health
  # web.panels.my-grafana.path: /
  # web.panels.my-grafana.hidden: true
  # web.panels.my-grafana.rewrite_json_paths: ["/config.json"]
  # Runtime panel control by the agent, off by default. Named layouts a human
  # applies from the "+" popover are the `panel_presets:` field, not a key.
  # web.allow_runtime_panels: true
  # web.runtime_panel_allowlist: ["grafana.local:3000"]

  # ── Multi-user web terminals ───────────────────────────────────────────────
  # `osprey up` runs a landing page and one terminal per user listed below.
  # `osprey web` honours only `auth.session_lifetime` from this block, so a
  # single terminal on your own machine works at any time. Set
  # `modules.web_terminals.enabled: false` to have `osprey up` deploy backend
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
    # No port keys here on purpose. Every host port this deployment publishes
    # is `deployment.port_base` (set that key to move them all) plus a fixed offset:
    # the landing page at the base itself, the shared services just above it,
    # one hundred ports per per-user family from base + 100 up, and the stores
    # at base + 800. User number i gets its family's first port + i, so
    # removing a user never shifts anyone else's ports. Give a second
    # deployment its own `port_base` and its whole block moves with it; the
    # virtual accelerator's Channel Access port (5064) is the one exception.
    # The reference guide's ports page prints the table.
    # Browsers reach the landing page's nginx directly here, so the address
    # they open is deploy.fqdn plus that port — and that is the address every
    # terminal checks an action against. Put something in front of this nginx
    # (a load balancer terminating TLS, a reverse proxy, a DNS alias) and add
    # `external_origin: https://<what browsers open>` here, or every action
    # inside a terminal is refused while every page still loads.
    # To override one port rather than move the block, name it here:
    #   nginx_port: 18000         # the landing page everyone opens first
    #   web_base_port: 18100      # first per-user web-terminal port
    default_persona: readonly   # used for any user below with no persona
    # Every terminal below asks for a login (user alice: password alice, and
    # so on — set in this repo's .env; rotate with `osprey users passwd`).
    auth:
      method: password
      # How long a browser stays signed in, in whole seconds. Applies to every
      # terminal here and to `osprey web`; 43200 is twelve hours.
      session_lifetime: 43200
      # Accepts login over plain HTTP, which fits 127.0.0.1 and nothing else.
      # For any reachable host, delete this line and configure tls instead.
      allow_insecure_http: true
      # Single sign-on instead of passwords: set `method: oidc` above and give
      # your provider's details here. `scopes` is what is asked for at the
      # authorization endpoint — the default below suits a provider that
      # publishes the identity claim under `profile` or `email`; add whatever
      # scope yours publishes it under. `openid` cannot be dropped: without it
      # the provider issues no ID token and the sidecar refuses every login.
      #   oidc:
      #     issuer: https://idp.example.org
      #     claim: preferred_username
      #     scopes: [openid, profile, email]
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
        # persona declares a `landing_group` (see `logbook` and `knowledge` below) moves into a
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
      - name: logbook
        index: 2
        persona: logbook
        display_name: "Logbook Research"
        # A shared card: everyone on this roster opens it with their own
        # password. It has no password of its own, so none is set in .env.
        access: any
      # The second product beside them: the facility knowledge layer — the
      # knowledge graph, the graph channel finder and the knowledge bundle —
      # read-only, with no control system behind it. Shared like the card
      # above. Index 4 rather than 3 so carol's ports below stay where they
      # were; the page follows roster order, so it sits with its sibling.
      - name: knowledge
        index: 4
        persona: knowledge
        display_name: "Facility Knowledge"
        access: any
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
      logbook:
        project: als-exemplar-logbook
        project_path: build/als-exemplar-logbook
        build_profile: personas/logbook.yml
        # Puts this persona's users under their own landing-page heading
        # instead of in with the people. Presentation only — it changes
        # nothing about the container, its ports, or what it can do.
        landing_group: Standalone deployments
      knowledge:
        project: als-exemplar-knowledge
        project_path: build/als-exemplar-knowledge
        build_profile: personas/knowledge.yml
        landing_group: Standalone deployments

  # ── Runtime ────────────────────────────────────────────────────────────────
  # Agent Python runs as a host subprocess.
  execution.execution_method: subprocess
  # Wall-clock ceiling on one agent Python run, in seconds. A run that reaches
  # it is killed and reported as a timeout, so raise it for a facility whose
  # analyses legitimately run long and lower it to keep a runaway script from
  # holding the sandbox.
  # python_executor.execution_timeout_seconds: 600
  # Console colour theme for the CLI: default | custom. With custom, set the
  # colours (`cli.custom_theme.primary` and friends) and optionally a banner.
  cli.theme: default
  # cli.custom_theme.primary: "#C75F71"
  # cli.banner: |
  #   Your custom ASCII art here
  # Facility timezone: how operator times are read and every timestamp is
  # rendered. Pinned to UTC for reproducibility; set your real zone in
  # production (e.g. America/Los_Angeles). Avoid ${TZ:-...}: inheriting the
  # host $TZ makes archiver queries and simulated events non-deterministic.
  system.timezone: UTC
  # Container runtime `osprey up` uses: auto (Docker first, then Podman),
  # docker, or podman. CONTAINER_RUNTIME in the environment overrides it.
  container_runtime: auto

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
# Environment variables the deployment needs. Replace `env: {}` with any
# of `required` (the variable must be set somewhere), `pinned` (this
# repo's own env chain owns it outright, and nowhere else), `defaults`
# (name to value) and `file` (a profile-relative path copied in as .env):
#
#   env:
#     required: [DISPATCH_WORKER_TOKEN]
#     pinned: [ARIEL_DB_PASSWORD]
#     defaults:
#       OSPREY_FACILITY_NAME: "Example Facility"
#     file: env/facility.env
#
# If `env:` already has children, add yours under it.
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
# preset has moved on; `osprey validate` refuses every difference from the
# preset that no `# DEVIATION: <why>` comment above the line claims (tag set
# by `deviation_marker:`). This profile is the source of truth either way.
# `providers_hash` is the same record for providers.yml beside this file;
# `osprey profile expand --providers` refreshes the packaged entries in it.
provenance:
  preset: control-assistant
  preset_hash: @PRESET_HASH:control-assistant@
  providers_hash: @PROVIDERS_HASH@
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
# Tier 1 is the flat whole-database view, so it serves one paradigm only.
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
# A Python server's package lives at mcp_servers/<package>/ beside this file;
# the build copies it to build/_mcp_servers/<package>/ for `-m <package>`.
#
# mcp_servers:
#   my_server:
#     command: "{current_python_env}"
#     args: [-m, my_server]
#     env:
#       OSPREY_CONFIG: "{project_root}/build/config.yml"
#       PYTHONPATH: "{project_root}/build/_mcp_servers"
#     permissions:
#       allow: [my_tool]
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

PERSONA_LOGBOOK_YML = """\
# Als Exemplar (logbook) — settings for one web login
#
# Only the differences from profile.yml belong here. The build merges this
# file over that one, picking up any edit you make there. To see the combined
# result:
#   osprey validate personas/logbook.yml
#
# Made from the bundled `control-assistant-logbook` preset.
#
#   emitted by OSPREY @OSPREY_VERSION@
#   preset content hash: @PRESET_HASH:control-assistant-logbook@

name: Als Exemplar (logbook)

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
  # approval-gated like every other write in OSPREY. Publishing it is the half
  # that actually reaches the facility's logbook, so it is gated too.
  approval.tools.entry_create: always
  approval.tools.entry_publish: always
  # Full split-pane layout: the ARIEL search panel is the point of this tier, so
  # it needs the panel area. Pinned rather than left to the server default for
  # the same reason the other two tiers pin theirs.
  web.ui_mode: expert
  # No shipped example plot in this WORKSPACE: the example invites the reader
  # to ask the agent for a live plot, which this persona cannot produce.
  artifact_server.example_artifact: false
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
  # deployment host and the copied ports are dialed there. The build renders
  # no services for this persona and writes `deployed_services: []` into its
  # config — every `services.*` key inherited from the base profile is dropped
  # from this render, except the ones that name a file in the render's own
  # data tree (the graph corpus and its search index), which stay as this
  # render's own — and a host that differs is named here.
"""

PERSONA_KNOWLEDGE_YML = """\
# Als Exemplar (knowledge) — settings for one web login
#
# Only the differences from profile.yml belong here. The build merges this
# file over that one, picking up any edit you make there. To see the combined
# result:
#   osprey validate personas/knowledge.yml
#
# Made from the bundled `control-assistant-knowledge` preset.
#
#   emitted by OSPREY @OSPREY_VERSION@
#   preset content hash: @PRESET_HASH:control-assistant-knowledge@

name: Als Exemplar (knowledge)

# Attached render: this persona builds a per-user terminal image only and
# connects to the shared web tier the hosting deployment runs on the same host.
# No services are scaffolded — the graph store it reads is the hosting
# deployment's, already declared there.
deploy_services: false

# The knowledge persona instead of the control-room operator one: the agent's
# brief is written for answering "how is the machine built" from three
# read-only sources and citing which one answered.
claude_md_template: CLAUDE.knowledge.md.j2

# Open on the KNOWLEDGE tab rather than the always-on Workspace tab. The
# CHANNELS tab is the other one this persona keeps.
default_panel: okf

# ── What this persona drops ──────────────────────────────────────────────────
# A persona can only add to an inherited list, never subtract from it, so
# everything the control-room agent carries and a knowledge terminal has no
# use for is removed here by name. What stays: the channel-finder,
# facility-knowledge and facility-knowledge-graph agents, the KNOWLEDGE and
# CHANNELS panels, and the channel-finder feedback hook (it captures
# channel-finder results and needs no control system).
exclude:
  hooks:
    - writes-check        # No hardware writes to pre-check
    - limits              # No per-channel limits to enforce
  rules:
    - workflows           # Scan, ramp and restore patterns: operator work
    - python-execution    # The Python sandbox is off (see config: below)
    - data-visualization  # Plotting needs the sandbox this persona does not run
    - control-system-safety  # EPICS PV rules, with no EPICS behind them
    - test-ioc-safety     # Test-IOC port isolation, likewise
  skills:
    - diagnose            # Fault diagnosis is control-room work
    - session-report      # Writes to the logbook, which this persona cannot reach
    - demo-gallery
    - demo-ui
    - writing-bluesky-plans    # Plan authoring needs the Bluesky server
    - operating-bluesky-plans  # and so does running one
    - bluesky-plans            # and so does listing them
  agents:
    - data-visualizer
    - logbook-search          # The logbook is the logbook persona's
    - logbook-deep-research
    - pyat-specialist         # Lattice computation needs the Python sandbox
  web_panels:
    - ariel
    - system-health
# The `safety` rule is deliberately NOT excluded. Its tools are gone, so the
# rule governs nothing today and costs a few lines of prompt. It stays because
# this agent runs inside a deployment that does move hardware: if anyone ever
# turns a control server back on here, the rule should already be in place.

# ── Config overrides ─────────────────────────────────────────────────────────
# Dotted keys ONLY — see the base profile's block.
config:
  # The axis this persona is defined by: knowledge tools and nothing else.
  # Each line switches off a tool server the base turns on. The graph server,
  # the channel finder (graph mode, see `channel_finder_mode` in the base) and
  # the facility knowledge server are the three left on.
  claude_code.servers.controls.enabled: false
  claude_code.servers.python.enabled: false
  claude_code.servers.bluesky.enabled: false
  claude_code.servers.health.enabled: false
  claude_code.servers.ariel.enabled: false
  # Read-only, stated at the one tool that would write: drafting a knowledge
  # concept edits the shared bundle, which is curation work for a login that
  # can edit the deployment. Deny lists UNION across `extends`, so this adds to
  # the base's floor.
  claude_code.permissions.deny:
    - mcp__osprey_facility_knowledge__draft_concept
  # Pinned even though every server that would honour it is gone, for the
  # same reason the read-only tier pins it: this key is the write boundary,
  # and it must not drift if the base's default ever changes. Three keys, not
  # one — see the read-only preset for why.
  control_system.writes_enabled: false
  control_system.connector.epics.writes_enabled: false
  control_system.connector.virtual_accelerator.writes_enabled: false
  # Full split-pane layout: the KNOWLEDGE panel is the point of this persona,
  # so it needs the panel area. Pinned rather than left to the server default,
  # like the other tiers pin theirs.
  web.ui_mode: expert
  # No shipped example plot in this WORKSPACE: the example invites the reader
  # to ask the agent for a live plot, which this persona cannot produce.
  artifact_server.example_artifact: false
  # The hosting deployment owns the web-terminal tier (nginx, landing, per-user
  # containers). Without this override the inherited roster would make this
  # render try to host a second web tier on the same host ports.
  modules.web_terminals.enabled: false
  # Nothing here says where the hosting deployment's services are — the graph
  # store's bolt port above all. This persona is an attached render
  # (`services: {}` of its own) built beside that deployment, and the build
  # copies every such fact from the deployment's own render into it (the
  # Reach Contract, `osprey.deployment.reach`). Per-user web-terminal
  # containers run `network_mode: host`, so container `localhost` IS the
  # deployment host and the copied ports are dialed there. The build renders
  # no services for this persona and writes `deployed_services: []` into its
  # config — every `services.*` key inherited from the base profile is dropped
  # from this render, except the ones that name a file in the render's own
  # data tree (the graph corpus and its search index), which stay as this
  # render's own — and a host that differs is named here.
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
  # the build refuses the contradiction. The build renders no services for
  # this persona and writes `deployed_services: []` into its config — every
  # `services.*` key inherited from the base profile is dropped from this
  # render, except the ones that name a file in the render's own data tree
  # (the graph corpus and its search index), which stay as this render's own
  # — and a host that differs IS named here.
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
  # the contradiction. The build renders no services for this persona and
  # writes `deployed_services: []` into its config — every `services.*` key
  # inherited from the base profile is dropped from this render, except the
  # ones that name a file in the render's own data tree (the graph corpus and
  # its search index), which stay as this render's own — and a host that
  # differs IS named here.
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

# The write-oriented panels, exactly as the readwrite tier declares them: the
# admin tier is a full operator desk plus deployment editing, never an operator
# desk minus. Persona lists UNION over the base, so these are added to the
# inherited builtin set. Each tab's URL, path and label are not spelled here:
# the hosting deployment's build derives them when it injects the event
# dispatcher and the bluesky-web sidecar, and this attached render is told
# them from that render (the Reach Contract, `osprey.deployment.reach`).
web_panels:
  - events          # EVENTS dashboard tab (event dispatcher)
  - bluesky         # Plan authoring, the plan queue, and the run's live results

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
  # the build refuses the contradiction. The build renders no services for
  # this persona and writes `deployed_services: []` into its config — every
  # `services.*` key inherited from the base profile is dropped from this
  # render, except the ones that name a file in the render's own data tree
  # (the graph corpus and its search index), which stay as this render's own
  # — and a host that differs IS named here.
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
# The dispatcher answers on this deployment's dispatcher port:
# `deployment.port_base` + 10, which is 10010 unless the deployment moved its
# port block. Fire one with (`osprey up` mints EVENT_DISPATCHER_TOKEN into this
# repo's .env; load it first:
# export $(grep -E '^EVENT_DISPATCHER_TOKEN=' .env | xargs)):
#   curl -X POST http://localhost:10010/webhook/hello-dispatch \\
#     -H "Authorization: Bearer $EVENT_DISPATCHER_TOKEN" \\
#     -H "Content-Type: application/json" -d '{}'
#
# Watch progress stream in the dashboard at http://localhost:10010/dashboard
#
# (Retries fire on *dispatch failure* — i.e. when the dispatcher cannot reach
# the worker — via the per-trigger `on_error: retry` policy. That path is not
# exercised by a curl against a healthy stack; see the docs and the unit test
# tests/unit/dispatch/test_server_routes.py for the retry/backoff behaviour.)

dispatcher:
  # The dispatcher forwards each fired trigger to this worker. The compose
  # template names the single worker "dispatch-worker-1", one port above the
  # dispatcher itself — `deployment.port_base` + 11, so 10011 at the default
  # base. Moving the block moves both. Under `dispatch.network: host` the build
  # rewrites this line to the worker's host address instead.
  # (Multi-worker load distribution is not yet implemented — see docs.)
  dispatch_target: http://dispatch-worker-1:10011
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
  #      curl -X POST http://localhost:10010/webhook/triage-event \\
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

# OS / editor noise, and the bytecode Python leaves beside any server package
# run in place. Deliberately unanchored: these are junk at any depth.
.DS_Store
*.swp
*.swo
__pycache__/
*.py[co]
"""

ENV_EXAMPLE = """\
# Als Exemplar Environment Configuration
#
# Every variable this deployment supplies: the provider keys, whatever its
# profile declares, and the tokens `osprey up` mints. Copy this file to `.env`
# at the repository root, beside `profile.yml`, and fill in what you need. That
# one file holds all your secrets, and a value in it survives every rebuild.
#
# Not listed here: the host-level knobs a command reads from its own environment
# (CONTAINER_RUNTIME, OSPREY_OFFLINE, ...), and the names the build stamps into
# the containers. The Environment Variables reference covers both.
#
# This file has no secrets in it and is safe to commit.
#
# The `.env` files at the repository root, and which one to edit:
#
#   .env.shared    the settings that are the same on every host; committed
#   .env           this host's own values and every key; wins over
#                  .env.shared when both set the same one
#   .env.example   this file: documentation, no values, never read at run time
#
# Anything else starting with `.env` is written by OSPREY itself (`osprey up`
# keeps `.env.auth` there, for one) — kept out of git, and not edited by hand.

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

# Gateway endpoints. These providers front a gateway that is your own host, so
# OSPREY ships no default: switch to one of them and it will not start until
# its endpoint is set here.
# als-apg
# ALS_APG_BASE_URL=

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
# BLUESKY_STANDIN_LAUNCH_TOKEN=  # bluesky_standin — arms the plan-launch endpoint of the Bluesky lane serving the live stand-in soft IOC (only on a deployment with `bluesky.second_lane`)
# OSPREY_TERMINAL_SECRET=  # bluesky_web — the operator login secret for the bluesky-web panel's web gate
# ZO_ROOT_USER_PASSWORD=  # openobserve — OpenObserve root/ingest credential
# ARIEL_DB_PASSWORD=  # postgresql — ARIEL Postgres password (also fills the agent's derived DSN)
# ARIEL_DB_READONLY_PASSWORD=  # postgresql — password of the SELECT-only Postgres role the agent's SQL tool queries through
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
# The login service does not receive it (nothing mounts a CA into that image),
# so an identity-provider fetch behind such a proxy still fails there.
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

In full, the first row is: `profile.yml`, `providers.yml`, `data/`, `personas/`, `triggers.yml`, `web-terminal-context/`, `.env.example`, `rules/`, `.gitignore`, `.env.shared`, `README.md`, `ci-extra.yml`, `.gitlab-ci.yml`, `scripts/verify.sh`.

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

Git covers your settings. `var/` and the root's `.env` files are everything
else — `.env`, and on a deployment with web terminals the deploy-written `.env.auth`,
which holds the password hashes and cannot be regenerated: without it `osprey up`
mints a new password for every user. A backup is a copy of those, and a restore is:

```bash
git clone <this repo> && tar xf state.tar.gz && osprey build && osprey up -d
```

## Common questions

### Adding an MCP server of your own

Put the server's Python package at `mcp_servers/<name>/`. `osprey build` copies
it to `build/_mcp_servers/<name>/`, and an entry under `mcp_servers:`
in `profile.yml` says how to start it:

```yaml
mcp_servers:
  my_server:
    command: "{current_python_env}"
    args: [-m, my_server]
    env:
      PYTHONPATH: "{project_root}/build/_mcp_servers"
```

The hello-world preset ships `example_server` as a worked copy to read. Which
buckets the artifact gallery sorts into is a separate, top-level key, one entry
per bucket:

```yaml
artifact_server:
  categories:
    beam_studies: {label: Beam studies, color: "#4C6EF5"}
```

### Adding a panel of your own

A panel is two halves, and one without the other is a tab that opens nothing.
The id goes in the `web_panels:` list, and its address goes under `config:`, as
the comment above `web_panels:` in `profile.yml` shows:

```yaml
web_panels:
  - elog

config:
  web.panels.elog.url: http://elog.example.org
  web.panels.elog.label: Elog
  web.panels.elog.path: /elog
```

### Mounting an extra directory into a web terminal

Deployments that give people their own web terminals mount per persona, under
`config:`. Each entry is a container volume string of two or three non-empty
parts, `source:target` or `source:target:mode`:

```yaml
config:
  modules.web_terminals.personas.operator.extra_mounts:
    - /opt/facility/data:/data:ro
```

Every key named here is written up in full at https://als-apg.github.io/osprey/.
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
#     image: python:3.12-slim
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
  image: python:3.12-slim
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
  image: alpine:3.23
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
    # The address browsers open this deployment at, derived from the same
    # origin the landing page carries and every terminal checks a write
    # against — not the SSH host the job connects to.
    url: http://127.0.0.1:10000
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
  probe_tcp  'virtual-accelerator: Channel Access on 5064'  localhost 5064
  probe_http 'openobserve: telemetry store on 10050'        http://localhost:10050/healthz
fi

# ── Web tier ─────────────────────────────────────────────────────────────────
# The landing page is nginx's own file, served before anything asks
# the caller for a credential. A terminal is the application, which
# answers an uncredentialed GET / with a 401 — so it is probed at
# /health, the route its auth gate lets through.
if wants web; then
  printf '\\n%s── Web terminal ──%s\\n\\n' "$BOLD" "$RESET"
  probe_http 'landing page'          http://localhost:10000/
  probe_http 'terminal (alice)'      http://localhost:10100/health
  probe_http 'terminal (bob)'        http://localhost:10101/health
  probe_http 'terminal (logbook)'    http://localhost:10102/health
  probe_http 'terminal (carol)'      http://localhost:10103/health
  probe_http 'terminal (knowledge)'  http://localhost:10104/health
fi

# ── Event dispatch ───────────────────────────────────────────────────────────
if wants dispatch; then
  printf '\\n%s── Event dispatch ──%s\\n\\n' "$BOLD" "$RESET"
  probe_http 'dispatcher health' http://localhost:10010/health
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
├── facility_ontology.json                # device vocabulary (facility.ontology)
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

#: The exemplar's own compiled ontology — the table ``facility.ontology`` names.
#:
#: A profile that carries a ``data:`` tree REPLACES the bundle's, so the copy
#: control-assistant ships never reaches this repo: an exemplar facility
#: declares its own vocabulary or it declares none, and a declared table that is
#: not on disk stops the build by design. Written against this repo's own three
#: families (``BPM``, ``DCCT``, ``HCM``) rather than copied from the demo
#: machine, because that is what a real facility's table looks like and what the
#: rendered terminology tables should show.
FACILITY_ONTOLOGY_JSON = """\
{
  "_generated": "Generated from facility_ontology.yaml by `osprey knowledge compile-ontology`. Do not edit.",
  "root": "AcceleratorDevice",
  "family_to_class": {
    "BPM": "BeamPositionMonitor",
    "DCCT": "BeamCurrentMonitor",
    "HCM": "HCorrector"
  },
  "classes": {
    "AcceleratorDevice": { "altLabels": [], "parent": null },
    "BeamCurrentMonitor": {
      "altLabels": ["beam current monitor", "current monitor", "dcct"],
      "parent": "Instrumentation"
    },
    "BeamPositionMonitor": {
      "altLabels": ["beam position monitor", "bpm", "position monitor"],
      "parent": "Instrumentation"
    },
    "Corrector": {
      "altLabels": ["corrector", "orbit corrector", "steering magnet"],
      "parent": "Magnet"
    },
    "HCorrector": {
      "altLabels": ["hcor", "horizontal corrector", "horizontal steering magnet"],
      "parent": "Corrector"
    },
    "Instrumentation": { "altLabels": ["diagnostics", "instrumentation"], "parent": "AcceleratorDevice" },
    "Magnet": { "altLabels": ["electromagnet", "magnet"], "parent": "AcceleratorDevice" }
  }
}
"""

CHANNEL_LIMITS_JSON = """\
{
  "_comment": "Write limits, enforced by the limits hook before any write reaches the control system. A channel is writable if and only if it is a setpoint (:SP); every other address is read-only, whatever this file says.",
  "SR:MAG:HCM:01:CURRENT:SP": { "min_value": -5.0, "max_value": 5.0, "writable": true },
  "SR:MAG:HCM:02:CURRENT:SP": { "min_value": -5.0, "max_value": 5.0, "writable": true }
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
DEMO_MACHINE_TTL = """\
# The knowledge-graph corpus `services.graphdb` seeds and the graph channel
# finder answers from. One device per channel family of machine.json, each
# binding stating its address and its direction.
@prefix narad_p: <https://narad.example.org/property/> .
@prefix narad_sem: <https://narad.example.org/schema/shared_semantics/> .

<https://narad.example.org/device/demo_SR_DCCT01> a narad_sem:BeamCurrentMonitor ;
    narad_p:deviceId "narad:device:demo:SR:DCCT01" ;
    narad_p:facility "demo" ;
    narad_p:hasBinding <https://narad.example.org/binding/demo_SR_DCCT01_CURRENT_RB> ;
    narad_p:ordinalInFacility 1 ;
    narad_p:ordinalInSection 1 ;
    narad_p:rawType "DCCT" ;
    narad_p:sPositionM 0.0 ;
    narad_p:sectionCode "SR" ;
    narad_p:sourceName "DCCT01" ;
    narad_p:system "DIAG" .

<https://narad.example.org/binding/demo_SR_DCCT01_CURRENT_RB> a narad_sem:ChannelBinding ;
    narad_p:bindingId "narad:binding:demo:SR:DCCT01:CURRENT_RB" ;
    narad_p:description "Stored beam current" ;
    narad_p:fullPv "SR:DIAG:DCCT:01:CURRENT:RB" ;
    narad_p:protocol "ca" ;
    narad_p:readsSignal narad_sem:beam_current .

<https://narad.example.org/device/demo_SR_BPM01> a narad_sem:BeamPositionMonitor ;
    narad_p:deviceId "narad:device:demo:SR:BPM01" ;
    narad_p:facility "demo" ;
    narad_p:hasBinding <https://narad.example.org/binding/demo_SR_BPM01_POSITION_X>,
        <https://narad.example.org/binding/demo_SR_BPM01_POSITION_Y> ;
    narad_p:ordinalInFacility 2 ;
    narad_p:ordinalInSection 2 ;
    narad_p:rawType "BPM" ;
    narad_p:sPositionM 1.0 ;
    narad_p:sectionCode "SR" ;
    narad_p:sourceName "BPM01" ;
    narad_p:system "DIAG" .

<https://narad.example.org/binding/demo_SR_BPM01_POSITION_X> a narad_sem:ChannelBinding ;
    narad_p:bindingId "narad:binding:demo:SR:BPM01:POSITION_X" ;
    narad_p:description "Beam position monitor 1, horizontal" ;
    narad_p:fullPv "SR:DIAG:BPM:01:POSITION:X" ;
    narad_p:protocol "ca" ;
    narad_p:readsSignal narad_sem:bpm_position_x .

<https://narad.example.org/binding/demo_SR_BPM01_POSITION_Y> a narad_sem:ChannelBinding ;
    narad_p:bindingId "narad:binding:demo:SR:BPM01:POSITION_Y" ;
    narad_p:description "Beam position monitor 1, vertical" ;
    narad_p:fullPv "SR:DIAG:BPM:01:POSITION:Y" ;
    narad_p:protocol "ca" ;
    narad_p:readsSignal narad_sem:bpm_position_y .

<https://narad.example.org/device/demo_SR_HCM01> a narad_sem:HorizontalCorrector ;
    narad_p:deviceId "narad:device:demo:SR:HCM01" ;
    narad_p:facility "demo" ;
    narad_p:hasBinding <https://narad.example.org/binding/demo_SR_HCM01_CURRENT_RB>,
        <https://narad.example.org/binding/demo_SR_HCM01_CURRENT_SP> ;
    narad_p:ordinalInFacility 3 ;
    narad_p:ordinalInSection 3 ;
    narad_p:rawType "HCM" ;
    narad_p:sPositionM 2.0 ;
    narad_p:sectionCode "SR" ;
    narad_p:sourceName "HCM01" ;
    narad_p:system "MAG" .

<https://narad.example.org/binding/demo_SR_HCM01_CURRENT_RB> a narad_sem:ChannelBinding ;
    narad_p:bindingId "narad:binding:demo:SR:HCM01:CURRENT_RB" ;
    narad_p:description "Horizontal corrector 1 current, readback" ;
    narad_p:fullPv "SR:MAG:HCM:01:CURRENT:RB" ;
    narad_p:protocol "ca" ;
    narad_p:readsSignal narad_sem:hcm_current .

<https://narad.example.org/binding/demo_SR_HCM01_CURRENT_SP> a narad_sem:ChannelBinding ;
    narad_p:bindingId "narad:binding:demo:SR:HCM01:CURRENT_SP" ;
    narad_p:description "Horizontal corrector 1 current, setpoint" ;
    narad_p:fullPv "SR:MAG:HCM:01:CURRENT:SP" ;
    narad_p:protocol "ca" ;
    narad_p:writesSignal narad_sem:hcm_current .
"""

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
    "personas/logbook.yml": PERSONA_LOGBOOK_YML,
    "personas/knowledge.yml": PERSONA_KNOWLEDGE_YML,
    "personas/readonly.yml": PERSONA_READONLY_YML,
    "personas/readwrite.yml": PERSONA_READWRITE_YML,
    "personas/admin.yml": PERSONA_ADMIN_YML,
    "web-terminal-context/alice/.gitkeep": "",
    "web-terminal-context/logbook/.gitkeep": "",
    "web-terminal-context/bob/.gitkeep": "",
    "web-terminal-context/knowledge/.gitkeep": "",
    "data/README.md": DATA_README_MD,
    "data/channel_databases/TEMPLATE_EXAMPLE.json": CHANNEL_DB_TEMPLATE_EXAMPLE_JSON,
    "data/channel_databases/tiers/tier1/in_context.json": CHANNEL_DB_IN_CONTEXT_JSON,
    "data/channel_databases/tiers/tier3/hierarchical.json": CHANNEL_DB_HIERARCHICAL_JSON,
    "data/benchmarks/cross_paradigm/queries/tier3_queries.json": BENCHMARK_QUERIES_JSON,
    "data/channel_limits.json": CHANNEL_LIMITS_JSON,
    "data/facility_ontology.json": FACILITY_ONTOLOGY_JSON,
    "data/machine_state_channels.json": MACHINE_STATE_CHANNELS_JSON,
    "data/raw/address_list.csv": RAW_ADDRESS_LIST_CSV,
    "data/facility_knowledge/index.md": FK_INDEX_MD,
    "data/facility_knowledge/subsystems/index.md": FK_SUBSYSTEMS_INDEX_MD,
    "data/facility_knowledge/subsystems/vacuum.md": FK_VACUUM_MD,
    "data/facility_knowledge/procedures/index.md": FK_PROCEDURES_INDEX_MD,
    "data/facility_knowledge/procedures/vacuum-recovery.md": FK_VACUUM_RECOVERY_MD,
    "data/demo_machine.ttl": DEMO_MACHINE_TTL,
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
_PROVIDERS_HASH_SENTINEL = "@PROVIDERS_HASH@"
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


def _providers_hash() -> str:
    """Content hash of the packaged provider catalog, as init stamps it."""
    from osprey.profiles.providers import compute_providers_hash, packaged_catalog_path

    return compute_providers_hash(packaged_catalog_path())


def packaged_providers_yml() -> str:
    """The packaged ``providers.yml``, which ``osprey init`` copies verbatim.

    Read rather than frozen, for the same reason the hashes are sentinels: the
    package owns this file's content, so a frozen copy would prove only that
    someone remembered to update two places. What the byte comparison is for is
    that init copies the catalog through unchanged, and that is what reading it
    here asserts.
    """
    from osprey.profiles.providers import packaged_catalog_path

    return packaged_catalog_path().read_text(encoding="utf-8")


def expand_sentinels(text: str) -> str:
    """Resolve the version, preset-hash and providers-hash sentinels in ``text``.

    The values the real emission stamps at materialization time. Resolving them
    here rather than freezing them keeps a byte-comparison against a live
    ``osprey init`` honest across version bumps, preset edits and additions to
    the provider catalog.
    """
    text = text.replace(_VERSION_SENTINEL, _osprey_version())
    text = text.replace(_PROVIDERS_HASH_SENTINEL, _providers_hash())
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
    # The provider catalog init writes beside the profile. Not in
    # BASE_SOURCE_FILES because it is read from the package rather than frozen.
    files["providers.yml"] = packaged_providers_yml()
    if with_ci:
        files.update(CI_PIPELINE_FILES)
    return {path: expand_sentinels(text) for path, text in files.items()}


@contextlib.contextmanager
def preserved_environ():
    """Confine a repo's ``.env`` to the code run inside this block.

    Loading a project config exports its ``.env`` into ``os.environ``
    (``ConfigBuilder`` → ``load_dotenv(override=True)``) — correct for the real
    CLI, where the process exits afterwards, but an in-process build in a test
    shares its process with every test after it. A leaked seeded token then
    changes later tests' behavior: the service-token mint treats a var already
    present in the process env as operator-provided and writes no ``.env`` at
    all. Wrap every in-process ``osprey build`` (or config load) of a repo that
    carries a ``.env`` in this guard.
    """
    snapshot = os.environ.copy()
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(snapshot)


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
