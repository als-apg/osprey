# Config key parity classification

Input to the resurrection-guard manifest. Every dotted key the config-honesty
ledger touched is classified as **all-presets** (must appear in all four shipped
presets' resolved `config:`) or **per-preset** (deliberately present in some and
absent from others, with the reason).

## The two sources

A rendered `build/config.yml` has two authors, and the guard's union is their
sum. Nothing may be in both — that is the whole point of the split.

| source | id | what it writes |
|---|---|---|
| `src/osprey/templates/project/config.yml.j2` | `framework` | only what the BUILD derives: the project layout, the artifact port, `api.providers` from `providers.yml`, and the keys a profile FIELD (`provider:`, `model:`, `channel_finder_mode:`, `default_panel:`, `panel_presets:`, `web_panels:`) decides — the list in `osprey.cli.derived_keys.DERIVED_KEYS` |
| `src/osprey/profiles/presets/*.yml` | the preset's stem | everything else, as dotted `config:` keys with a comment on the line that sets each one |

The four presets, which are what `osprey init --preset <name>` copies out as the
operator's own `profile.yml`:

| id | path |
|----|------|
| `hello-world` | `src/osprey/profiles/presets/hello-world.yml` |
| `ariel-standalone` | `src/osprey/profiles/presets/ariel-standalone.yml` |
| `channel-finder-standalone` | `src/osprey/profiles/presets/channel-finder-standalone.yml` |
| `control-assistant` | `src/osprey/profiles/presets/control-assistant.yml` |

The persona presets (`control-assistant-*.yml`) are deltas that `extends:` the
root and are deliberately out of scope: every key they carry is either inherited
from `control-assistant` — already counted — or a persona-scoped override of
one, so including them would make parity a claim about deltas rather than about
the four documents an operator starts from.

Every claim below was produced by rendering the framework template with
`ChainableUndefined` over the guard's matrix, resolving each preset through
`resolve_build_profile` + `_expand_dotted` with `layout_port_fill` applied, and
— for MCP/approval claims — feeding the resulting `claude_code` section to
`osprey.registry.mcp.resolve_servers`.

## Scope caveat — read before deriving guard rules

This document covers the keys the **config-honesty ledger touched**. It is *not*
an exhaustive enumeration of every key in the four presets, and the
"all-presets" table is *not* automatically the complete set of keys that happen
to appear in all four.

Three rules follow for anything consuming this list:

1. **Do not auto-derive "present in all four ⇒ values must match."** Presence
   and value-equality are separate properties here. `deployed_services` is
   present in all four and its values deliberately differ; `container_runtime`
   is present in all four and its value must match. The tables below state which
   applies — infer neither from the other.
2. **Absence from this document is not evidence of anything.** A key not listed
   was outside the ledger's scope, not judged parity-exempt. Classify it on its
   own evidence before adding it to a guard.
3. **A derived key can never be parity-marked.** No preset spells one — a
   `config:` entry for a member of `DERIVED_KEYS` is refused at profile
   validation — so marking one would demand a second home for a fact the build
   already writes. The mark is a claim about the operator's document only.

## All-presets (parity-required)

Present in all four presets' resolved `config:`. A guard fires if one goes
missing. Values are stated where they must match and flagged where they must
not.

| dotted key | value | notes |
|---|---|---|
| `container_runtime` | `auto` everywhere | absent behaves identically to `auto` (`deployment/runtime_helper.py`), so this is a documentation guarantee, not a behavioral one |
| `system.timezone` | `UTC` everywhere | pinned for reproducibility |
| `approval.enabled` | `true` everywhere | |
| `approval.default_policy` | `always` everywhere | fail-closed for hook-wired tools not listed |
| `claude_code.telemetry.enabled` | diverges | `false` in `channel-finder-standalone`, `true` in the other three; that preset deploys no OpenObserve, so telemetry has nowhere local to land. Presence is required because it is a posture-floor key |
| `hooks.debug` | diverges | `false` in `hello-world`, `true` in the other three. Also a posture-floor key, which is what moved it from a per-template divergence to a parity requirement: every preset must state it either way |
| `artifact_server.host` | `127.0.0.1` everywhere | |
| `artifact_server.auto_launch` | `true` everywhere | |
| `execution.execution_method` | `subprocess` everywhere | |
| `deployed_services` | diverges | `[]` (`channel-finder-standalone`), `[openobserve]` (`hello-world`), `[postgresql, openobserve, qmd, graphdb]` (`ariel-standalone`, `control-assistant`). Presence is universal, the value is capability-scoped. A guard must check presence only |

Their parent blocks — `approval`, `artifact_server`, `claude_code`,
`claude_code.servers`, `claude_code.telemetry`, `execution`, `hooks`, `system` —
carry the mark too, because a path is in the union at every level and the mark
is presence-only. They are not separate claims.

## Per-preset (deliberate divergence)

A guard must **not** require these everywhere. Rationale is per key.

### Capability-scoped sections

Present only where the deployment has the capability.

| dotted key | present in | rationale |
|---|---|---|
| `control_system.*` | `hello-world`, `control-assistant` | the two standalones disable the `controls` server; no hardware surface |
| `archiver.*` | `hello-world`, `control-assistant` | rides with `control_system` |
| `ariel.*` | `ariel-standalone`, `control-assistant` | logbook deployments only |
| `logbook.*` | `ariel-standalone`, `control-assistant` | rides with `ariel` |
| `channel_finder.*` | `control-assistant` | the channel-finder standalone configures its pipeline entirely through the `channel_finder_mode:` FIELD, so it needs no `config:` block; only control-assistant adds the benchmark corpus and the panel's web server |
| `facility_knowledge.*` | `control-assistant` | the only preset shipping an OKF corpus |
| `services.*` | `hello-world`, `ariel-standalone`, `control-assistant` | the channel finder is file-backed and deploys no standing service |
| `services.postgresql`, `services.qmd`, `services.graphdb` | `ariel-standalone`, `control-assistant` | only ARIEL needs the logbook stores |
| `cli.*` | `control-assistant` | console theming; the minimal presets omit it |
| `modules.web_terminals.*`, `deploy.fqdn` | `control-assistant` | the multi-user roster. The other three are single-terminal starting points |
| `bluesky.*`, `claude_code.servers.bluesky.enabled` | `control-assistant` | the plan queue |
| `claude_code.permissions.deny` | `control-assistant` | the tier floor every persona inherits |
| `web.*` | `ariel-standalone`, `channel-finder-standalone`, `control-assistant` | `hello-world` selects no panel and states no web key, so it renders no `web:` block at all |

### Deliberately divergent defaults

| dotted key | values | rationale |
|---|---|---|
| `control_system.writes_enabled` | `true` in `control-assistant`; `false` in `hello-world` | control-assistant is the reference facility demonstrating the approval flow; hello-world is a read-only-by-default starting point |
| `web.theme` | live `light` in `control-assistant`; commented in `hello-world`; absent from the two standalones | the default is `"osprey"` either way (`web_terminal/app.py`), so nothing behavioral is at stake |
| `facility.name` | live `Example Research Facility` in the two standalones; commented in `hello-world`; absent from `control-assistant` | the standalones are demo deployments with a name to show; the other two fall back to the project name |
| `facility.prefix` | commented in the two standalones, live `ca` in `control-assistant` | only the multi-user web-terminal stack reads it, so only the preset that ships one sets it |
| `channel_finder.benchmark.dataset_path` | `control-assistant` only | see below |
| `deployment.bind_address` (commented) | `hello-world`, `ariel-standalone`, `control-assistant` | absent ⇒ `127.0.0.1`, the safe state |

`control_system.connector.<type>.writes_enabled` is a per-connector-type
override of the global `control_system.writes_enabled` row above, and is
deliberately **not** parity-required. It is written commented in the presets
that carry the matching connector block, so it is never in the resolved
`config:` and never enters the union a parity guard would compare. Its semantics
are tri-state: absent inherits the global key, literally `true` arms writes for
that connector type, and any other value leaves them unarmed. Arming writes is a
per-facility decision, so no preset may ship it live.

`control_system.connector.<type>.limits_checking` is the same story for the
limits posture, and is likewise **not** parity-required. It overrides the
deployment-wide `control_system.limits_checking` block whole -- a per-type
block states both `enabled` and `allow_unlisted_channels` and then answers
alone. Only `virtual_accelerator`, `epics` and `live_standin` carry entries;
`mock` and `doocs` write no block. `database_path` has no per-type spelling: a
deployment mounts one limits database, so that leaf stays deployment-wide, and a
per-type block omitting it is complete rather than half-written.

The `epics` and `live_standin` blocks are written commented, so they are not in
the union. The `virtual_accelerator` block is different, and the difference is
the whole reason the per-type shape exists: `control-assistant` ships both of
its leaves **live** and permissive, because the relaxation is about the
simulator and must not reach the machine beside it. Parity is still not required
-- the other three presets carry no virtual-accelerator block at all -- but
"no preset ships either leaf live" is false, and the manifest entries carry no
`rendered: false` flag on that account.

### Keys nothing ships (`rendered: false`)

Seventy-five manifest keys have real readers, real spellings and no writer among
the framework template or the four presets. They are marked `rendered: false` so
the phantom-key check does not read them as manifest rot.

The flag is checked in **both** directions. A key carrying it that IS in the
union fails as a phantom-key contradiction naming the source that renders it,
because otherwise the flag is a one-way escape: a key that starts being shipped
keeps a marking saying nothing ships it, and the prose beside it goes on
describing a commented example. Four entries had rotted exactly that way under
the preset conversion -- `facility.prefix` and the three virtual-accelerator
`limits_checking` paths, all live in `control-assistant`.

The preset conversion added twenty of the seventy-five, in three groups:

| group | why nothing ships it |
|---|---|
| `control_system.connector.epics.gateways.*` (9) | a facility's Channel Access gateways cannot be guessed, and shipping someone else's would point a control system at a stranger's machine. `control-assistant` documents the whole block commented and the operator authors it at go-live |
| `archiver.mongodb_archiver.*` (9) | the build's archiver injector derives the connection block from the deployment's own `services.postgresql` / `va_archiver:` declaration. The preset states only `archiver.type: mongodb_archiver` |
| `control_system.connector.mock.{response_delay_ms,noise_level}` (2) | the presets run the mock connector at the connector's own defaults; a slower or noisier machine is authored per deployment |

### Deleted keys documented as commented examples

Two keys on the `deleted` list are still documented, commented, in a preset, and
the manifest's `deleted_commented_examples` section records why each is not a
resurrection: both left what OSPREY *ships* while staying live in their readers.
`ariel.database.uri` still wins over the derived DSN when set, and
`archiver.mock_archiver.simulation_file` still overrides the derived simulation
file. Under the app templates that documentation sat in a template, which the
resurrection check never read for comments; converting the templates into
presets moved the same lines onto the surface it does read. The exemption is
narrow — a LIVE spelling of either key in a preset is still a resurrection, as
is either one reappearing in the rendered union or in the loader's synthesized
defaults — and the exemption itself is checked, so one that names a key that was
never deleted, carries no reason, or outlives its preset line fails the guard.

### `api.providers` membership

Not a fixed set, and no longer a per-template question: `providers.yml` is the
single catalog, the framework template renders it verbatim, and a
`config: api.providers.*` key is refused. All ten providers ship. The guard
checks **shape** (every listed provider carries `base_url` + a complete tier
map), never membership.

### `facility` identity

`facility.name` is canonical; top-level `facility_name` is the retired spelling,
still honored as a fallback (`utils/facility.py`). No preset ships
`facility_name` any more.

**Guard note:** top-level `facility_name` is a resurrection candidate — it must
not reappear in any preset, though the *reader* fallback stays.

### `channel_finder.benchmark` — resolved

`channel-finder-standalone` ships no `benchmark:` block, and adding one would
name a phantom path: only the control-assistant bundle ships the query corpus,
and `materialize_tier_artifacts` returns silently for bundles with no
`data/channel_databases/tiers/` subtree (`cli/templates/scaffolding.py`), which
the channel-finder bundle does not have.

So benchmark **is** deliberately control-assistant-only. The command still works
in the channel-finder deployment via `--queries-path`, which bypasses the config
read (`cli/channel_finder_cmd.py`). The bare `KeyError` on the config subscript
was replaced with an error naming both remedies
(`services/channel_finder/benchmarks/runner.py`), and the preset documents the
omission.

## Open items for the manifest (not fixed here)

1. **Test fixtures are out of guard scope.** `tests/utils/fixtures/legacy_config_all_deleted_keys.yml`
   deliberately holds every retired key. If the guard scans `tests/`, it must
   allowlist that fixture.
2. **`approval.tools.entry_create` is inert in `hello-world`** — that preset
   disables the `ariel` server, so the tool never exists. Harmless (the policy
   is `always`, and the block is fail-closed anyway), but it is a policy for a
   tool that cannot run.
3. **`draft_concept` is approval-governed but listed nowhere.** `resolve_servers`
   shows `osprey_facility_knowledge` enabled in `hello-world` and
   `control-assistant`, and its `draft_concept` tool carries the approval hook.
   It is absent from every `approval.tools` block, so it falls to
   `default_policy: always` — fail-closed and correctly described by the
   shipped comment. No change needed; recorded so it is not mistaken for drift.
4. **`facility.timezone` is read but shipped by no preset**
   (`deployment/web_terminals/env_production.py`,
   `deployment/web_terminals/render.py`), distinct from `system.timezone`.
   Hidden-key stanza candidate.
5. **`facility.prefix` has a stated convention and no validator — by design.**
   The 2-6-character lowercase-alnum-plus-hyphens rule this entry was opened
   against came from a schema document that no longer exists (it went with the
   `facility-config.yml` surface). The convention survives in prose only, and
   the only validation anywhere is a non-emptiness lint
   (`deployment/web_terminals/lint.py`, `_check_empty_facility_prefix`).

   The absence is real, not an artifact of an incomplete search. The same lint
   module defines `_USERNAME_CHARSET_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")`
   and enforces it at two sites — usernames (`_check_username_charset`) and
   persona names (`_check_persona_charset`). So this codebase does write charset
   checks where it wants them, and has none for `facility.prefix`.

   **The rule is not merely unenforced — it is inert.** Tracing `facility.prefix`
   to its sinks: two container-name interpolations plus the personas path, and
   nothing else. It never becomes an nginx location key or a URL segment, which
   is the specific reason usernames and persona names *do* get the charset
   regex. There is no sink at which violating the 2-6/lowercase rule breaks
   anything.

   **Resolved: it stays a convention — do NOT add a validator.** A new charset
   check would reject configurations that work correctly today, turning a
   cosmetic inconsistency into a breaking change. The presets state the enforced
   rule (non-emptiness) alongside the Docker constraint, which is the right
   two-altitude framing.

## Where the web-terminal lint runs

`deployment/web_terminals/lint.py` validates `modules.web_terminals` at two
altitudes, and a guard touching either surface should know which one it is on:

| entry point | reads | run by |
|---|---|---|
| `lint_web_terminals(config)` | a rendered project `config.yml` | `osprey scaffold web-terminals lint`, and the pre-render gate inside `... render` |
| `lint_profile_config(config)` | a build profile's `config:` block (dotted keys, nested internally) | profile validation, before anything is built |

The profile-altitude pass skips the two checks that need a rendered project —
persona `project_path` existence and the `build_profile` delta shape, both
inside `_check_persona_project_paths` — because `osprey init` only rewrites
catalog entries into `personas/<name>.yml` deltas at materialization.
Every shipped preset is pinned clean at that altitude by
`tests/deployment/web_terminals/test_lint.py`.

Port-overlap coverage follows the same split: the collision set is the per-user
port families, `nginx_port`, the TLS listener when enabled, and every host port
a `services.<name>` entry publishes (`port`, `port_host`, `*_port`).
Container-internal listeners are deliberately excluded — the dispatch worker's
`worker_port_base` binds nothing on the host — so a guard must not treat every
port-shaped key as contended.

Since presets state no host ports at all, those keys reach the lint through
`layout_port_fill`, which the build and the guard both apply: a preset that
deploys `services.graphdb` and spells no port gets the layout's two graphdb
ports at the deployment's own base.
