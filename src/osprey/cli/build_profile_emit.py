"""Standalone profile emission for ``osprey init``.

Materializes a bundled preset into a fully explicit, self-contained
``profile.yml`` — no ``extends:`` — so a facility profile shows every knob
it configures and never depends on upstream preset content at build time.

Content and comments are produced by two cooperating passes:

1. **Content** comes from the exact pipeline the render path uses
   (:func:`~osprey.cli.build_profile_presets._load_preset_raw` +
   :func:`~osprey.cli.build_profile_resolve.merge_cli_overrides` +
   :func:`~osprey.cli.build_profile_merge._resolve_extends`), so the emitted
   profile resolves to byte-identical semantics as building from the preset
   directly (with the same ``-O`` / ``--set`` layers).
2. **Comments** come from ruamel.yaml round-trip documents of the preset
   file(s) — the whole ``extends`` chain, root first, overlaid child-over-base
   so each layer contributes the comments for the keys it introduces. The
   commented document is then synced to the resolved content: keys absent from
   the resolved dict (``extends``, ``exclude``, excluded entries) are dropped,
   and differing values are replaced in place, which keeps the preset's
   comment attached to the key whose value changed.
"""

from __future__ import annotations

import copy
import io
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML, CommentedMap
from ruamel.yaml.error import CommentMark
from ruamel.yaml.tokens import CommentToken

from osprey import __version__
from osprey.errors import BuildProfileError

from .build_profile_load import _PROFILE_SCHEMA_MIN_OSPREY
from .build_profile_merge import _deep_merge, _resolve_extends, compute_preset_hash
from .build_profile_presets import _load_preset_raw
from .build_profile_resolve import merge_cli_overrides

# YAML-surface spellings that differ from the canonical resolved-content field
# name (D2: the rename is confined to the profile-YAML surface, so the
# resolved dict still keys off the Python identifier). `_sync_to_resolved`
# diffs `resolved` against the raw preset doc, which is genuinely YAML-spelled
# — re-spell the resolved side to match before diffing, or the doc's key
# looks "removed" and its ca entry (holding the NEXT key's comment block, per
# ruamel's trailing-comment packing) gets deleted along with it.
_FIELD_TO_YAML: dict[str, str] = {"data_bundle": "app_template"}


# Every BuildProfile field belongs to exactly one of the three sets below, keyed
# by FIELD name (the YAML spelling is applied later via _FIELD_TO_YAML). The
# sets differ in *synthesis* policy, not in whether a carried value survives —
# a value the resolved profile carries is always emitted, whatever its set.
#
# Classifying a new field is a necessary condition plus a judgement, not a
# lookup:
#   - a `None` loader default FORCES COMMENTED — writing the key would emit
#     `null`, and "commenting beats null noise";
#   - a concrete default (bool/list/dict/str) PERMITS EXPLICIT but does not
#     require it. The opt-in facility blocks (mcp_servers,
#     artifact_server) default to `{}` and still stay COMMENTED, because an
#     empty block teaches nothing while a commented example teaches the shape;
#   - build mechanics are never synthesized at all, whatever their default.

# Written in every emitted profile. Members absent from the resolved dict are
# synthesized from the loader default (see _EXPLICIT_DEFAULTS) under a rationale
# comment, so a facility never has to guess whether an unset knob is off or
# merely undocumented.
_EXPLICIT_KEYS: frozenset[str] = frozenset(
    {
        "name",  # identifies the deployment; the emitter always sets it
        "data_bundle",  # app template selector — emitted as `app_template`
        "deploy_services",  # self-contained vs attached project
        "config",  # config.yml overrides: the facility's main dial
        "services",  # container services the profile declares
        "env",  # environment variables the deployment requires
        "panel_presets",  # named web-terminal layouts
        "hooks",  # the six artifact-selection lists below are the agent's
        "rules",  # capability surface — silence about them would hide what
        "skills",  # the deployment can actually do
        "agents",
        "output_styles",
        "web_panels",
        "requires_osprey_version",  # schema stamp; see _PROFILE_SCHEMA_MIN_OSPREY
        "provenance",  # source preset + its hash; see _PROVENANCE_COMMENT
    }
)

# Opt-in blocks: active when the resolved profile carries a value, a commented
# documented template when it does not — never absent. Each member needs an
# entry in _COMMENTED_TEMPLATES (guard-tested).
_COMMENTED_TEMPLATE_KEYS: frozenset[str] = frozenset(
    {
        "data",  # profile-carried data tree; active once `osprey init` writes one
        "provider",  # every bundled preset sets these two, so they emit active
        "model",  # today; the template only covers a preset that omits them
        "channel_finder_mode",
        "tier",  # pinning it would break mode-edit parity — see PROPOSAL D-notes
        "default_panel",
        "deploy",  # CI/registry/host coordinates, filled in per facility
        "mcp_servers",  # facility tool servers
        "artifact_server",  # gallery categories + host/port
        "dispatch",  # the seven optional feature blocks
        "bluesky",
        "virtual_accelerator",
        "va_archiver",
        "bluesky_web",
        "nextcloud_bridge",
        "gchat_bridge",
    }
)

# How the build runs, not what gets deployed. FR4 calls these keys "absent",
# which means absent from what the emitter ADDS: never synthesized (a facility
# should not inherit build plumbing it did not ask for) and never offered as a
# commented template. It does NOT mean stripped when carried — `claude_md_template`
# is a preset-author primitive that two bundled presets rely on, and dropping it
# would silently swap their CLAUDE.md persona and break emitted-vs-preset parity.
_BUILD_MECHANICS_KEYS: frozenset[str] = frozenset(
    {
        "osprey_install",
        "python_env",
        "claude_md_template",
        "environment",  # interpreter + packages: the built Python env
        "dependencies",
        "lifecycle",  # pre/post-build steps
    }
)

# Loader defaults for EXPLICIT members, used when the resolved profile lacks
# one. `name`, `requires_osprey_version` and `provenance` are always set by the
# emitter, so they need no fallback.
_EXPLICIT_DEFAULTS: dict[str, Any] = {
    "data_bundle": "control_assistant",
    "deploy_services": True,
    "config": {},
    "services": {},
    "env": {},
    "panel_presets": {},
    "hooks": [],
    "rules": [],
    "skills": [],
    "agents": [],
    "output_styles": [],
    "web_panels": [],
}

# Why a synthesized key is present at all — written above it so the emitted
# profile explains its own completeness rather than looking padded.
_SYNTHESIS_RATIONALE: dict[str, str] = {
    "data_bundle": "App template this deployment renders from.",
    "deploy_services": "true builds its own services stack; false attaches to another project's.",
    "config": "config.yml overrides, as dotted keys. Empty means template defaults stand.",
    "services": "Services this profile declares. Injected ones are added at build time.",
    "env": "Environment variables the deployment needs (required:/defaults:/file:).",
    "panel_presets": "Named web-terminal layouts, as label -> list of panel ids.",
    "hooks": "Safety and lifecycle hooks installed into .claude/.",
    "rules": "Facility rules installed into .claude/rules/.",
    "skills": "Skills installed into .claude/skills/.",
    "agents": "Subagents installed into .claude/agents/.",
    "output_styles": "Output styles installed into .claude/output-styles/.",
    "web_panels": "Panels offered by the web terminal.",
}

# The stamp is always written, so it carries its explanation rather than a
# synthesis rationale: it is a floor for readers, not a preset choice.
_REQUIRES_VERSION_COMMENT = (
    "# Minimum OSPREY release that understands this profile's keys. Builds below it abort."
)

# Same rationale as the version stamp: always written, so it explains itself.
# Emitted as a key rather than left to the header comment because a later build
# reads it — the header says the same thing for people.
_PROVENANCE_COMMENT = [
    "# What this profile was materialized from. Emitted, not hand-written: a",
    "# build compares it against the installed preset and mentions it when the",
    "# preset has moved on. Advisory only — this profile is the source of truth.",
]

# Round-trip mode preserves comments, key order, and quoting style (same
# conventions as osprey.utils.config_writer).
_yaml = YAML(typ="rt")
_yaml.preserve_quotes = True
_yaml.width = 4096
# Match the bundled presets' list style (`  - item` under the key).
_yaml.indent(mapping=2, sequence=4, offset=2)


# Appended when the resolved profile has no ``mcp_servers:`` section — the
# facility's own tool servers, which bundled presets never carry.
_MCP_SERVERS_APPENDIX = """
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
"""

# Appended when the resolved profile has no ``artifact_server:`` block.
_CATEGORIES_APPENDIX = """
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
"""


# Commented templates for the remaining COMMENTED_TEMPLATE_KEYS. Keyed by FIELD
# name; appended (in _COMMENTED_TEMPLATE_ORDER) whenever the resolved profile
# does not carry the key, so every opt-in knob is visible and documented in
# every emitted profile rather than being discoverable only from the docs.
_COMMENTED_TEMPLATES: dict[str, str] = {
    "data": """
# --- Facility data tree ------------------------------------------------------
# Path (relative to this profile) of the data tree the build copies in place of
# the bundled apps/<app_template>/data/. `osprey init` materializes one
# and sets this key; a persona delta under personas/ inherits it and resolves it
# against this profile's directory. Full replacement, not a fallback.
#
# data: data
""",
    "provider": """
# --- LLM provider ------------------------------------------------------------
# Provider the deployment's agents talk to. Must be configured in the project's
# config.yml api.providers block.
#
# provider: anthropic
""",
    "model": """
# --- Default model -----------------------------------------------------------
# A tier (haiku/sonnet/opus) or a model ID the provider serves.
#
# model: sonnet
""",
    "channel_finder_mode": """
# --- Channel-finder paradigm -------------------------------------------------
# How the agent looks up channels: in_context (whole DB in the prompt),
# hierarchical (system/device drill-down), or middle_layer. Drives which
# channel database the build materializes.
#
# channel_finder_mode: hierarchical
""",
    "tier": """
# --- Channel-database tier ---------------------------------------------------
# Build-time only (1 or 3), selecting which bundled tier DB is materialized.
# Left unset the build picks a paradigm-aware default, which is why it stays
# commented: pinning it here would override that default on every rebuild.
# Tier 1 ships the in_context paradigm only.
#
# tier: 3
""",
    "default_panel": """
# --- Default web-terminal panel ----------------------------------------------
# Panel id opened when the web terminal loads. Must be a built-in, an entry in
# the web_panels list above, or a custom panel backed by a web.panels.<id>.url
# config override.
#
# default_panel: artifacts
""",
    "dispatch": """
# --- Event dispatch ----------------------------------------------------------
# Runs the agent unattended against a trigger file (facility events in, agent
# runs out). worker_count sets parallelism; workspace_mode isolated gives each
# run its own copy of the project.
#
# dispatch:
#   triggers: triggers/my-facility.yml
#   worker_count: 1
#   workspace_mode: isolated
#   facility_name: ALS
""",
    "bluesky": """
# --- Bluesky bridge -----------------------------------------------------
# Exposes Bluesky plans to the agent. tiled_enabled adds the data-access
# service; excluded_plans removes shipped plans from the catalog.
#
# bluesky:
#   port: 8090
#   tiled_enabled: false
#   excluded_plans: []
""",
    "virtual_accelerator": """
# --- Virtual accelerator -----------------------------------------------------
# Containerized soft-IOC serving simulated channels, so the deployment can be
# exercised without touching the machine.
#
# virtual_accelerator:
#   port: 5064
""",
    "va_archiver": """
# --- Stored archive ----------------------------------------------------------
# Where a simulated deployment keeps its history: a MongoDB store the build
# seeds with a deterministic base series, a recorder samples the running
# machine into, and the mongodb_archiver connector reads back. Declaring it
# makes history stored data rather than values synthesized at read time.
#
# Every knob below travels into the rendered config.yml, so retention and
# cadence are profile edits rather than code changes. The connector's own
# connection keys are DERIVED from this block — do not also write
# archiver.mongodb_archiver.* under `config:`, which would be the same fact
# twice. Selecting the archiver stays a separate decision: set
# `archiver.type: mongodb_archiver` in `config:` to read from this store.
#
# The password is never written here: `osprey up` mints it into the
# deployment's .env under the name password_env gives.
#
# An attached project (deploy_services: false) deploys no store of its own and
# must set `host` to the machine whose archive it reads.
#
# va_archiver:
#   retention_days: 30          # how far back history reaches
#   hot_span_hours: 48          # recent window kept at the dense cadence
#   hot_cadence_sec: 10         # seconds between samples inside it
#   tail_cadence_sec: 60        # ...and outside it (a multiple of the above)
#   recorder_cadence_sec: 10    # how often the live machine is sampled
#   recorder_tail_cadence_sec: 60
#   recorder_poll_sec: 30       # how often the recorder re-reads config.yml
#   port_host: 27017
#   database: osprey_archiver
#   collection: pv_history
#   compression: zstd           # zstd | snappy | zlib | none
#   username: osprey
#   auth_database: admin
#   password_env: MONGO_ROOT_PASSWORD
#   timeout_sec: 5
#   host: archive.example.org   # only for an attached project
""",
    "bluesky_web": """
# --- Bluesky web panels ------------------------------------------------------
# Scan-monitoring panels for the web terminal (requires the bluesky block).
#
# bluesky_web:
#   port: 8095
""",
    "nextcloud_bridge": """
# --- Nextcloud bridge --------------------------------------------------------
# Answers questions asked from a Nextcloud Talk room. The trigger name must
# match one declared in the dispatch triggers file.
#
# nextcloud_bridge:
#   trigger: nextcloud-question
""",
    "gchat_bridge": """
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
""",
    "deploy": """
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
""",
    # The two long-form blocks above this module's helpers, registered here so
    # the partition has a single template table to guard.
    "mcp_servers": _MCP_SERVERS_APPENDIX,
    "artifact_server": _CATEGORIES_APPENDIX,
}

# Append order for the commented templates, so two emissions of the same preset
# are byte-identical and diffs between presets stay readable. Deployment-shaping
# keys first, facility extension points next, optional services last.
_COMMENTED_TEMPLATE_ORDER: tuple[str, ...] = (
    "data",
    "provider",
    "model",
    "channel_finder_mode",
    "tier",
    "default_panel",
    "deploy",
    "mcp_servers",
    "artifact_server",
    "dispatch",
    "bluesky",
    "virtual_accelerator",
    "va_archiver",
    "bluesky_web",
    "nextcloud_bridge",
    "gchat_bridge",
)


def _config_segments(config: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    """Split each dotted ``config`` key into its path segments.

    Non-string keys (a YAML ``on:`` parsed as a bool, say) address no path and
    are left out, so they can never be read as anyone's prefix.
    """
    return {key: tuple(key.split(".")) for key in config if isinstance(key, str)}


def _collapse_config_prefixes(config: dict[str, Any]) -> dict[str, Any]:
    """Fold every ``config`` key that addresses a subtree of another one into it.

    ``config_update_fields`` applies dotted keys in iteration order, each one
    setting its addressed path verbatim. Two keys where one path-prefixes the
    other therefore have an order-dependent result: reordering
    ``modules.web_terminals:`` (a whole subtree, ``enabled: true``) and
    ``modules.web_terminals.enabled: false`` flips the deployment's web tier
    back on. An emitted profile is meant to be edited, so it must not carry a
    pair whose meaning depends on where the user leaves the lines.

    Prefixing is by SEGMENT, not by string: ``modules.web`` does not address
    anything inside ``modules.web_terminals``. The deeper key wins — its value
    is set at its path inside the shallower key's mapping — which is what
    applying the two in file order does today, and stays true however the
    emitted file is later reordered.

    Raises:
        BuildProfileError: if the shallower key's value (or an intermediate
            node under it) is not a mapping, so it cannot carry the deeper
            key's nested override. Applying such a pair is the ``TypeError``
            ``config_update_fields`` raises mid-build, reported here instead.
    """
    segments = _config_segments(config)
    # Shallowest ancestor per key: a chain a < b < c collapses wholly into a,
    # so c's override is placed relative to a rather than to the b that is
    # itself about to disappear.
    roots: dict[str, str] = {}
    for key, path in segments.items():
        chain = [
            other
            for other, segs in segments.items()
            if len(segs) < len(path) and path[: len(segs)] == segs
        ]
        if chain:
            roots[key] = min(chain, key=lambda other: len(segments[other]))
    if not roots:
        return dict(config)

    result = {key: copy.deepcopy(value) for key, value in config.items() if key not in roots}
    # Shallowest first, so a deeper key's override lands after the shallower
    # ones it refines however the file happened to order them.
    for key in sorted(roots, key=lambda k: len(segments[k])):
        root = roots[key]
        node = result[root]
        # The path the override has to travel through, named as it goes so a
        # non-mapping on the way can be reported as the key a reader sees.
        blocking = root
        for part in segments[key][len(segments[root]) : -1]:
            if not isinstance(node, dict):
                break
            node.setdefault(part, {})
            blocking = f"{blocking}.{part}"
            node = node[part]
        if not isinstance(node, dict):
            raise BuildProfileError(
                f"config: {key!r} sets a value inside {blocking!r}, but {blocking!r} holds a "
                f"{type(node).__name__}, not a mapping — a scalar cannot carry a nested "
                f"override. Drop one of the two keys, or give the shallower one a mapping value."
            )
        node[segments[key][-1]] = copy.deepcopy(config[key])
    return result


# The ``config:`` subtree the multi-user web-terminal stack lives under. A
# profile that turns this module on with a non-empty persona catalog owns its
# personas too, so ``osprey init`` materializes a sibling profile per
# catalog entry (D7a) instead of leaving the catalog pointed at bundled preset
# names that would ignore the facility's own data tree.
_WEB_TERMINALS_PATH: tuple[str, ...] = ("modules", "web_terminals")


def _merge_subtree(into: dict[str, Any], value: Mapping[str, Any]) -> None:
    """Deep-merge ``value`` into ``into`` in place; ``value`` wins on conflict."""
    for key, item in value.items():
        current = into.get(key)
        if isinstance(current, dict) and isinstance(item, Mapping):
            _merge_subtree(current, item)
        else:
            into[key] = copy.deepcopy(item)


def effective_web_terminals(config: Mapping[str, Any]) -> dict[str, Any]:
    """The single ``modules.web_terminals`` subtree a ``config:`` block sets.

    ORDER IS THE WHOLE POINT, and it is fixed here so no caller can get it
    wrong. A ``config:`` block is a flat bag of dotted keys applied verbatim in
    iteration order, and the bundled persona presets inherit their parent's
    whole ``modules.web_terminals:`` subtree (``enabled: true``, catalog and
    all) while switching the module off with a separate
    ``modules.web_terminals.enabled: false``. Reading either key alone answers
    "enabled". So:

    1. :func:`_collapse_config_prefixes` runs FIRST, folding every
       prefix-related pair deeper-key-wins;
    2. then every key that still addresses this path — the path itself, a key
       under it, or an ancestor carrying it nested inside its value — is folded
       into one subtree, shallowest first so the deeper key again wins.

    Returns a plain dict (empty when the config addresses nothing here); the
    caller owns it and may mutate it freely.
    """
    collapsed = _collapse_config_prefixes(dict(config))
    segments = _config_segments(collapsed)
    depth = len(_WEB_TERMINALS_PATH)
    subtree: dict[str, Any] = {}
    for key in sorted(segments, key=lambda k: len(segments[k])):
        path = segments[key]
        value = collapsed[key]
        if path == _WEB_TERMINALS_PATH:
            if isinstance(value, Mapping):
                _merge_subtree(subtree, value)
        elif path[:depth] == _WEB_TERMINALS_PATH:
            # A key addressing INSIDE the subtree, e.g.
            # `modules.web_terminals.personas.ops.build_profile`.
            node = subtree
            for part in path[depth:-1]:
                child = node.get(part)
                if not isinstance(child, dict):
                    child = {}
                    node[part] = child
                node = child
            node[path[-1]] = copy.deepcopy(value)
        elif _WEB_TERMINALS_PATH[: len(path)] == path:
            # An ancestor key (`modules:`) carrying the subtree nested inside
            # its own value. Bundled presets never spell it this way — they
            # warn against it, since a nested `modules:` wholesale-replaces the
            # rendered subtree — but a facility's own profile may, and a
            # predicate that missed it would silently emit no personas.
            probe: Any = value
            for part in _WEB_TERMINALS_PATH[len(path) :]:
                probe = probe.get(part) if isinstance(probe, Mapping) else None
            if isinstance(probe, Mapping):
                _merge_subtree(subtree, probe)
    return subtree


def emits_persona_profiles(config: Mapping[str, Any]) -> bool:
    """Whether ``config`` stands up a persona stack of its own (D7a trigger).

    True exactly when the effective ``modules.web_terminals`` subtree (see
    :func:`effective_web_terminals`) is enabled AND carries a non-empty
    ``personas`` mapping — a profile that deploys per-persona terminals. False
    for the persona presets themselves, which inherit the catalog but disable
    the module: emitting personas from one of those would produce
    personas-of-a-persona.
    """
    web_terminals = effective_web_terminals(config)
    personas = web_terminals.get("personas")
    return bool(web_terminals.get("enabled")) and isinstance(personas, Mapping) and bool(personas)


def persona_catalog(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The persona entries a sibling profile is emitted for — ``{}`` when the
    profile does not trigger.

    Shares :func:`emits_persona_profiles`'s verdict rather than re-deriving it,
    so the trigger and the thing triggered can never disagree. Entries that are
    not name→mapping pairs are dropped: catalog well-formedness is deploy lint's
    job, and emission has nothing to write for them.
    """
    if not emits_persona_profiles(config):
        return {}
    personas = effective_web_terminals(config)["personas"]
    return {
        name: dict(entry)
        for name, entry in personas.items()
        if isinstance(name, str) and isinstance(entry, Mapping)
    }


def _to_plain(value: Any) -> Any:
    """Recursively convert ruamel container types to plain dict/list.

    Scalars are left as-is: ruamel scalar wrappers subclass their plain
    counterparts, so equality against plain values holds.
    """
    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    return value


def _comment_block(text: str) -> list[str]:
    """Split raw comment text into lines, dropping the blank ones at each edge.

    ruamel's comment tokens carry the newlines that separated the block from
    its neighbors; those blanks are layout, not content, and re-emitting them
    would compound a blank line per round trip.
    """
    lines = text.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def _tokens_to_lines(tokens: list[Any] | None) -> list[str]:
    """Flatten comment tokens to raw text lines, trimming edge blank lines."""
    if not tokens:
        return []
    return _comment_block("".join(tok.value for tok in tokens if tok is not None))


def _entry_texts(entry: list[Any]) -> tuple[str | None, list[str]]:
    """Split a ruamel ca entry into (eol comment, trailing block lines).

    ruamel packs the end-of-line comment on a key's own line *and* every
    comment line up to the next key into one token (index 2 of the entry).
    The first line is the key's eol comment; the rest visually belongs to the
    NEXT key in the file.
    """
    tok = entry[2]
    if tok is None:
        return None, []
    first, _, rest = tok.value.partition("\n")
    eol = first.strip() if first.strip().startswith("#") else None
    return eol, _comment_block(rest)


def _extract_visual(
    cm: CommentedMap, parent_entry: list[Any] | None = None
) -> tuple[dict[Any, list[str]], dict[Any, str]]:
    """Map each key of ``cm`` to the comments that visually belong to it.

    Returns ``(pre, eol)``: ``pre[key]`` is the comment block ABOVE the key's
    line (stored by ruamel on the predecessor key — or, for a nested map's
    first key, on the parent's ca entry), ``eol[key]`` the comment on the
    key's own line.
    """
    pre: dict[Any, list[str]] = {}
    eol: dict[Any, str] = {}
    keys = list(cm.keys())
    if keys and parent_entry is not None and parent_entry[3]:
        lines = _tokens_to_lines(parent_entry[3])
        if lines:
            pre[keys[0]] = lines
    for idx, key in enumerate(keys):
        entry = cm.ca.items.get(key)
        if entry is None:
            continue
        own_pre = _tokens_to_lines(entry[1])
        if own_pre:
            pre[key] = pre.get(key, []) + own_pre
        eol_text, trailing = _entry_texts(entry)
        if eol_text:
            eol[key] = eol_text
        if trailing and idx + 1 < len(keys):
            pre[keys[idx + 1]] = trailing
    return pre, eol


def _set_pre_comment(cm: CommentedMap, key: Any, lines: list[str], indent: int) -> None:
    """Attach comment ``lines`` above ``key``, indented to the key's depth."""
    tokens = [
        CommentToken((line.strip() if line.strip() else "") + "\n", CommentMark(indent), None)
        for line in lines
    ]
    entry = cm.ca.items.setdefault(key, [None, None, None, None])
    entry[1] = (entry[1] or []) + tokens


def _deepest_leaf(cm: CommentedMap, key: Any) -> tuple[CommentedMap, Any]:
    """Return the map and key whose ca entry holds ``key``'s trailing block.

    ruamel stores a trailing comment on the last key actually rendered under
    ``key``, which is the deepest last leaf when ``key``'s value is a mapping.
    """
    container: CommentedMap = cm
    leaf_key = key
    while isinstance(container.get(leaf_key), CommentedMap) and len(container[leaf_key]) > 0:
        container = container[leaf_key]
        leaf_key = list(container.keys())[-1]
    return container, leaf_key


def _take_trailing(cm: CommentedMap, key: Any) -> str | None:
    """Detach and return the comment block ``key`` holds for its successor.

    ruamel packs a key's own end-of-line comment and every comment line up to
    the next key into one token, so the block that visually introduces the NEXT
    key lives on this one. Returns ``None`` when there is no such block.
    """
    container, leaf_key = _deepest_leaf(cm, key)
    entry = container.ca.items.get(leaf_key)
    if not entry or entry[2] is None:
        return None
    # `entry` is ruamel's untyped ca list, so pin the token text to `str`
    # before splitting it — the three parts are what this returns.
    first, sep, rest = str(entry[2].value).partition("\n")
    if not sep or not rest.strip():
        return None
    entry[2].value = first + "\n"
    return rest


def _append_trailing(cm: CommentedMap, key: Any, block: str) -> None:
    """Render ``block`` after ``key``, keeping any block ``key`` already holds."""
    entry = cm.ca.items.setdefault(key, [None, None, None, None])
    if entry[2] is None:
        entry[2] = CommentToken("\n" + block, CommentMark(0), None)
    else:
        entry[2].value += block


def _relocate_trailing(cm: CommentedMap, old_last_key: Any, new_last_key: Any) -> None:
    """Move ``old_last_key``'s trailing comment block behind ``new_last_key``.

    A section-header comment for the NEXT sibling section is stored by ruamel
    as the trailing block of a map's last key. Appending new keys to the map
    would render them AFTER that header, visually placing them in the wrong
    section — so the trailing block moves to the appended final key.
    """
    block = _take_trailing(cm, old_last_key)
    if block is not None:
        _append_trailing(cm, new_last_key, block)


def _merge_commented(
    base: CommentedMap,
    child: CommentedMap,
    child_parent_entry: list[Any] | None = None,
    indent: int = 0,
) -> None:
    """Overlay ``child`` onto ``base`` in place, carrying comments.

    Keys already in ``base`` keep the base file's comments (their values are
    replaced); keys new in ``child`` bring along the comments that visually
    belong to them in the child file (block above + eol), re-anchored to the
    key itself so later deletions of neighboring keys cannot orphan them.
    List semantics deliberately do not mirror the render path's union-dedup —
    :func:`_sync_to_resolved` enforces content afterwards; this pass only
    decides which file's comments win.
    """
    pre, eol = _extract_visual(child, child_parent_entry)
    base_keys = list(base.keys())
    old_last_key = base_keys[-1] if base_keys else None
    appended = False
    for key, child_val in child.items():
        base_val = base.get(key)
        if isinstance(base_val, CommentedMap) and isinstance(child_val, CommentedMap):
            _merge_commented(base_val, child_val, child.ca.items.get(key), indent + 2)
            continue
        is_new = key not in base
        base[key] = child_val
        if is_new:
            appended = True
            if key in pre:
                _set_pre_comment(base, key, pre[key], indent)
            if key in eol:
                base.yaml_add_eol_comment(eol[key], key)
    if appended and old_last_key is not None:
        new_last_key = list(base.keys())[-1]
        if new_last_key != old_last_key:
            _relocate_trailing(base, old_last_key, new_last_key)


def _sync_to_resolved(doc: CommentedMap, resolved: dict[str, Any]) -> None:
    """Make ``doc``'s content exactly match ``resolved``, keeping comments.

    Values that already compare equal are left untouched so their item-level
    comments survive; differing values are replaced (the key-level comment
    stays attached); keys absent from ``resolved`` are deleted together with
    their comments.
    """
    for key in [k for k in doc.keys() if k not in resolved]:
        # The block a key holds introduces the key AFTER it (a collapsed
        # `config` entry is typically the last one in its section, holding the
        # next section's header), so it survives on the preceding key rather
        # than dying with this one.
        block = _take_trailing(doc, key)
        keys = list(doc.keys())
        index = keys.index(key)
        if block is not None and index > 0:
            _append_trailing(doc, keys[index - 1], block)
        del doc[key]
        doc.ca.items.pop(key, None)
    for key, value in resolved.items():
        current = doc.get(key)
        if isinstance(current, CommentedMap) and isinstance(value, dict):
            _sync_to_resolved(current, value)
        elif key not in doc or _to_plain(current) != _to_plain(value):
            doc[key] = value


def _replace_header(text: str, header: str) -> str:
    """Swap the leading comment block of ``text`` for ``header``.

    The preset file's own header addresses preset readers ("copy this file
    ..."); the emitted profile gets a header describing what it is and how to
    rebuild from it.
    """
    lines = text.splitlines()
    body_start = 0
    while body_start < len(lines) and (
        lines[body_start].startswith("#") or not lines[body_start].strip()
    ):
        body_start += 1
    return header.rstrip("\n") + "\n\n" + "\n".join(lines[body_start:]).rstrip("\n") + "\n"


def emit_persona_delta_yaml(
    preset_name: str,
    profile_name: str,
    profile_filename: str,
    host_filename: str = "profile.yml",
) -> str:
    """Render a persona sibling as a pure DELTA over the materialized host profile.

    The bundled persona presets are already deltas — a handful of keys over
    their base preset, each carrying its own explanatory comment. This emission
    keeps exactly that shape: the preset's raw text is reused comments-and-all,
    with the ``extends`` line REMOVED and the display ``name`` retitled.
    Nothing else is added or subtracted, so the emitted file is the preset's own
    layer and nothing more.

    There is no ``extends:`` because a file under ``personas/`` beside a
    ``profile.yml`` is merged over that profile implicitly (FR-10), and the
    merge anchors every profile-relative path — ``data:``, the trigger config,
    the convention dirs — at the host profile's directory. A delta that
    re-anchored them itself would fight that.

    Callers must only pass a ``preset_name`` whose ``extends`` names the host
    profile's source preset: the emitted delta drops every layer between the
    two, so a preset that sits anywhere else would resolve to something the
    caller never asked for. ``osprey init`` rejects such catalog entries
    outright rather than emitting an approximation.

    The ``extends`` removal is line surgery on the preset's raw text — that is
    what preserves its comments — so it assumes the key is written on ONE line
    with its value beside the colon (``extends: control-assistant``). A preset
    spelling it across lines is rejected rather than cut in half, since dropping
    only the ``extends:`` line would leave its value behind as a stray document
    fragment.

    Args:
        preset_name: Bundled persona preset whose raw text is the delta body.
        profile_name: Display name for the emitted persona profile.
        profile_filename: Display path (``my-profile/personas/readonly.yml``)
            for the header's validate hint.
        host_filename: The host profile's file name one directory up, named in
            the generated header.

    Returns:
        Complete persona delta text: the preset's own commented overrides,
        retitled, with no ``extends:``.

    Raises:
        BuildProfileError: If the preset is unknown, declares no ``extends``
            line — it is then not a delta over anything, and emitting its raw
            layer would silently drop the base it was written against — or
            writes ``extends`` across more than one line.
    """
    raw, preset_path = _load_preset_raw(preset_name)
    if not isinstance(raw.get("extends"), str) or not raw["extends"]:
        raise BuildProfileError(
            f"Preset {preset_name!r} declares no extends chain — it cannot be "
            f"emitted as a delta over a host profile."
        )
    body: list[str] = []
    titled = False
    dropped_extends = False
    for line in preset_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("extends:"):
            # The inheritance is positional now, so the key goes away entirely.
            # Only a one-line spelling can go away by dropping this line: with
            # the value on a following line, cutting here would leave that value
            # behind as a stray fragment, and the emitted delta would be neither
            # the preset's layer nor valid.
            if not line.split(":", 1)[1].strip():
                raise BuildProfileError(
                    f"Preset {preset_name!r} writes 'extends' across more than one line. "
                    f"A persona delta is emitted by removing that line from the preset's "
                    f"own text, which only works when the value sits beside the colon "
                    f"(extends: {raw['extends']}) — rewrite it that way."
                )
            dropped_extends = True
            continue
        if dropped_extends and not line.strip():
            # Swallow only the blank line the dropped key left behind, so the
            # spacing around the neighbouring blocks is the preset's own.
            dropped_extends = False
            continue
        dropped_extends = False
        if line.startswith("name:"):
            body.append(f"name: {profile_name}")
            titled = True
        else:
            body.append(line)
    text = "\n".join(body).rstrip("\n") + "\n"

    preset_hash = compute_preset_hash(preset_name) or "(unavailable)"
    header = (
        f"# {profile_name} — settings for one web login\n"
        f"#\n"
        f"# Only the differences from {host_filename} belong here. The build merges this\n"
        f"# file over that one, picking up any edit you make there. To see the combined\n"
        f"# result:\n"
        f"#   osprey validate {profile_filename}\n"
        f"#\n"
        f"# Made from the bundled `{preset_name}` preset.\n"
        f"#\n"
        f"#   emitted by OSPREY {__version__}\n"
        f"#   preset content hash: {preset_hash}"
    )
    text = _replace_header(text, header)

    if titled:
        return text
    return text.rstrip("\n") + "\n\n" + f"name: {profile_name}\n"


# The zone map at the top of a materialized profile: what the repo holds, which
# parts of it are the operator's and which the build owns, and the loop between
# them. No box-drawing glyphs and <= 80 columns, so it survives any editor a
# facility opens config files in. Emitted only into the main profile.yml —
# persona siblings share the mental model and repeating the picture three times
# per repo would just be noise.
_ZONE_MAP = """\
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
#   STATE    var/, the agent's memory and audit log. Not in git. Kept"""


def emit_standalone_profile_yaml(
    preset_name: str,
    overrides: tuple[Path, ...],
    set_pairs: tuple[str, ...],
    profile_name: str,
    extra_layers: tuple[Mapping[str, Any], ...] = (),
    include_flow_diagram: bool = False,
) -> str:
    """Render the standalone ``profile.yml`` text for ``osprey init``.

    Args:
        preset_name: Bundled preset to materialize (any CLI spelling).
        overrides: ``-O`` override files, layered in declaration order.
        set_pairs: ``--set KEY=VALUE`` pairs, layered on top.
        profile_name: Display name written to the profile's ``name:`` key.
        extra_layers: Raw profile fragments merged after the user's layers,
            through the same :func:`_deep_merge` channel a trailing ``-O`` file
            would use — so they win over the user's, and so nothing here is a
            second path into the resolved content. ``osprey init`` uses
            this to repoint the emitted persona catalog at the sibling profiles
            it is about to write: those values are derived from the resolved
            catalog, so they cannot come from a file the caller passes.
        include_flow_diagram: Embed the four-zone map and the edit → build → up
            loop in the generated header. Set for the main ``profile.yml``
            only; persona siblings keep the compact header.

    Returns:
        Complete ``profile.yml`` content: fully explicit (no ``extends:``),
        preset comments preserved, overrides applied in place.
    """
    # Content authority — identical layering to the render path: preset raw,
    # -O files, --set pairs, then extends resolution (which also applies any
    # exclude: subtractions and consumes the extends/exclude keys).
    raw, base_anchor = _load_preset_raw(preset_name)
    raw = merge_cli_overrides(raw, overrides, set_pairs)
    for layer in extra_layers:
        raw = _deep_merge(raw, dict(layer))
    chain: list[Path] = []
    resolved = _resolve_extends(raw, base_anchor, chain)
    resolved["name"] = profile_name

    # The schema floor a *reader* of this profile needs — pinned, never the
    # running version, which would let the emitting release satisfy its own gate
    # while ignoring the keys it just wrote.
    resolved["requires_osprey_version"] = f">={_PROFILE_SCHEMA_MIN_OSPREY}"

    # Where this profile came from, as data rather than as a comment a later
    # build would have to parse (FR-6). The generated header says the same thing
    # in prose for people; both are written here from the same two values, so
    # they cannot disagree.
    normalized = base_anchor.stem
    preset_hash = compute_preset_hash(preset_name) or "(unavailable)"
    resolved["provenance"] = {"preset": normalized, "preset_hash": preset_hash}

    # Defaults-synthesis: every EXPLICIT member the preset left unset is written
    # out at its loader default, so the emitted profile shows the whole
    # deployment-shaping surface rather than only the keys the preset happened
    # to mention.
    synthesized = [field for field in _EXPLICIT_DEFAULTS if field not in resolved]
    for field in synthesized:
        resolved[field] = _EXPLICIT_DEFAULTS[field]

    # A preset layer that refines an inherited subtree leaves the emitted file
    # with both keys; collapsing them is what makes the emitted config safe to
    # reorder (see _collapse_config_prefixes).
    if isinstance(resolved.get("config"), dict):
        resolved["config"] = _collapse_config_prefixes(resolved["config"])

    # Comment source — the extends chain's files, root first, so the richly
    # commented base contributes most comments and each child layer brings the
    # comments for the keys it introduces.
    doc: CommentedMap | None = None
    for layer_path in reversed(chain):
        with open(layer_path, encoding="utf-8") as f:
            layer_doc = _yaml.load(f)
        if not isinstance(layer_doc, CommentedMap):  # pragma: no cover - presets are mappings
            continue
        if doc is None:
            doc = layer_doc
        else:
            _merge_commented(doc, layer_doc)
    if doc is None:  # pragma: no cover - _load_preset_raw already validated
        doc = CommentedMap()

    for field, yaml_key in _FIELD_TO_YAML.items():
        if field in resolved:
            resolved[yaml_key] = resolved.pop(field)

    _sync_to_resolved(doc, resolved)

    # Rationale above each synthesized key, so a reader can tell a deliberate
    # default from a value the preset chose. Applied after the sync, which is
    # what appends resolved-only keys to the document.
    for field in synthesized:
        rationale = _SYNTHESIS_RATIONALE.get(field)
        key = _FIELD_TO_YAML.get(field, field)
        if rationale and key in doc:
            _set_pre_comment(doc, key, [f"# {rationale}"], 0)
    if "requires_osprey_version" in doc:
        _set_pre_comment(doc, "requires_osprey_version", [_REQUIRES_VERSION_COMMENT], 0)
    if "provenance" in doc:
        _set_pre_comment(doc, "provenance", _PROVENANCE_COMMENT, 0)

    buffer = io.StringIO()
    _yaml.dump(doc, buffer)
    text = buffer.getvalue()

    zone_block = f"{_ZONE_MAP}\n#\n" if include_flow_diagram else ""
    header = (
        f"# {profile_name} — OSPREY deployment repo\n"
        f"#\n"
        f"{zone_block}"
        f"# Made from the bundled `{normalized}` preset. Everything it sets is written\n"
        f"# out below and is yours to edit. Nothing is hidden or inherited.\n"
        f"#\n"
        f"#   emitted by OSPREY {__version__}\n"
        f"#   preset content hash: {preset_hash}"
    )
    text = _replace_header(text, header)

    # Opt-in knobs the resolved profile does not carry appear as documented
    # commented templates, so no configurable surface is invisible. Look the key
    # up under its YAML spelling: the re-spell above rewrote _FIELD_TO_YAML
    # members in place, so a field-name probe would miss a carried value and
    # append a template beside the active key.
    for field in _COMMENTED_TEMPLATE_ORDER:
        if _FIELD_TO_YAML.get(field, field) not in resolved:
            text += _COMMENTED_TEMPLATES[field]
    return text
