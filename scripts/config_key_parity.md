# Config key parity classification

Input to the resurrection-guard manifest. Every dotted key the config-honesty
ledger touched is classified as **all-presets** (must appear in every shipped
root preset's resolved `config:`) or **per-preset** (deliberately present in
some and absent from others, with the reason). Value equality is a third,
separate classification, carried by the manifest's `same-value` mark and
described below. Parity is a claim about the root presets only; the bundled
apps are union sources, never parity subjects.

## The sources

A rendered `build/config.yml` has two authors, and an app render is composed
from both. The guard's union is the sum of the sources below. The framework
template and a root preset may never both write a key — that is the whole point
of the split.

| source | id | what it writes |
|---|---|---|
| `src/osprey/templates/project/config.yml.j2` | `framework` | only what the BUILD derives: the project layout, the artifact port, `api.providers` from `providers.yml`, and the keys a profile FIELD (`provider:`, `model:`, `channel_finder_mode:`, `default_panel:`, `panel_presets:`, `web_panels:`) decides — the list in `osprey.cli.derived_keys.DERIVED_KEYS` |
| `src/osprey/profiles/presets/*.yml` (`render_contexts.presets`) | the preset's stem | everything else, as dotted `config:` keys with a comment on the line that sets each one |
| `src/osprey/profiles/presets/apps/*.yml` (`render_contexts.apps`) | the app's stem | each app's `config:` composed over `render_contexts.host` through the resolver's app branch: the host's facts, the app's keys over them, every write switch pinned off |

The root presets, which are what `osprey init --preset <name>` copies out as
the operator's own `profile.yml`:

| id | path |
|----|------|
| `hello-world` | `src/osprey/profiles/presets/hello-world.yml` |
| `control-assistant` | `src/osprey/profiles/presets/control-assistant.yml` |

The bundled apps, each composed over `control-assistant` (the host):

| id | path |
|----|------|
| `channel-finder` | `src/osprey/profiles/presets/apps/channel-finder.yml` |
| `knowledge` | `src/osprey/profiles/presets/apps/knowledge.yml` |
| `logbook` | `src/osprey/profiles/presets/apps/logbook.yml` |

An app is a complete agent definition, not a delta, so a key only an app states
has its writer in the apps context: the `python` and
`osprey_facility_knowledge` server switches, `web.control_target_picker`,
`web.ui_mode`, the `graph` server switch and the per-type write pins. The
governed-set claims are measured from the `logbook` and `channel-finder` app
renders.

The bundled tiers (`tiers/*.yml`) are deltas over their root and deliberately
out of scope: every key they carry is either inherited from the root — already
counted — or a tier-scoped override of one, so including them would make parity
a claim about deltas rather than about the two documents an operator starts
from.

Every claim below was produced by rendering the framework template with
`ChainableUndefined` over the guard's matrix, resolving each preset through
`resolve_build_profile` + `_expand_dotted` with `layout_port_fill` applied,
composing each app the same way from a scratch deployment that extends the host,
and — for MCP/approval claims — feeding the resulting `claude_code` section to
`osprey.registry.mcp.resolve_servers`.

## Scope caveat — read before deriving guard rules

This document covers the keys the **config-honesty ledger touched**. It is *not*
an exhaustive enumeration of every key in the presets, and the
"all-presets" table is *not* automatically the complete set of keys that happen
to appear in both.

Three rules follow for anything consuming this list:

1. **Do not auto-derive "present in both ⇒ values must match."** Presence
   and value-equality are separate properties here. `deployed_services` is
   present in both and its values deliberately differ; `container_runtime`
   is present in both and its value must match. The tables below state which
   applies — infer neither from the other. The machine-checked form of "its
   value must match" is `same-value: true` in the manifest, and failure mode 6
   of `check_config_keys.py` compares the presets' stated values for every key
   that carries it.
2. **Absence from this document is not evidence of anything.** A key not listed
   was outside the ledger's scope, not judged parity-exempt. Classify it on its
   own evidence before adding it to a guard.
3. **A derived key can carry neither mark.** No preset spells one — a
   `config:` entry for a member of `DERIVED_KEYS` is refused at profile
   validation — so marking one would demand a second home for a fact the build
   already writes. The mark is a claim about the operator's document only. The
   guard refuses a `same-value` mark no two presets state, which is how a
   derived key shows up.

## All-presets (parity-required)

Present in every root preset's resolved `config:`. A guard fires if one goes
missing. A row reading "`X` everywhere" carries `same-value: true` beside
`all-templates: true` in the manifest, and the guard compares the presets'
stated values for it. A row reading "diverges" carries `all-templates` only. The
value column is prose for the reader, because the presets are the one place a
value is written, and
`tests/scripts/test_config_key_guard.py::test_the_parity_table_names_exactly_the_value_marked_keys`
holds this table and the marks to one set of keys.

| dotted key | value | notes |
|---|---|---|
| `container_runtime` | `auto` everywhere | absent behaves identically to `auto` (`deployment/runtime_helper.py`), so this is a documentation guarantee, not a behavioral one |
| `system.timezone` | `UTC` everywhere | pinned for reproducibility |
| `approval.enabled` | `true` everywhere | |
| `approval.default_policy` | `always` everywhere | fail-closed for hook-wired tools not listed |
| `hooks.debug` | diverges | `false` in `hello-world`, `true` in `control-assistant`. Also a posture-floor key, which is what moved it from a per-template divergence to a parity requirement: every preset must state it either way |
| `artifact_server.host` | `127.0.0.1` everywhere | |
| `artifact_server.auto_launch` | `true` everywhere | |
| `execution.execution_method` | `subprocess` everywhere | |
| `deployed_services` | diverges | `[]` (`hello-world`), `[postgresql, qmd, graphdb]` (`control-assistant`). Presence is universal, the value is capability-scoped. A guard must check presence only |

Their parent blocks — `approval`, `artifact_server`, `claude_code`,
`claude_code.servers`, `execution`, `hooks`, `system` —
carry `all-templates` too, because a path is in the union at every level and
that mark is presence-only. They are not separate claims. No block carries
`same-value`, which marks leaves.

## Same value wherever stated

A `same-value` mark without `all-templates` leaves presence free and pins only
agreement among the presets that state the key.

- `approval.tools.<tool>` (eight rows): a tool's policy follows what the tool
  does, not which deployment serves it. Presence is capability-scoped, since a
  preset lists only tools its servers gate
  (`tests/profiles/test_approval_tools_parity.py`), and an unlisted tool falls to
  `approval.default_policy`. `entry_create` and `entry_publish` carry no mark:
  `control-assistant` is the only root preset that states them (the `logbook`
  app states them too, and apps are not parity subjects), so the mark would
  compare nothing.

### Considered and left unmarked

| keys | reason |
|---|---|
| `services.<name>.*` sizes, intervals, images, paths | capacity or per-deployment layout; host ports come from the layout fill and no preset states them |
| `ariel.*` tuning | per-deployment, and `ariel.vocabulary.enabled` already diverges |
| `control_system.*` | `type` and `writes` already diverge |
| `facility.*`, `channel_finder.query_max_rows`, `claude_code.servers.<name>.enabled` | capability-scoped |

## Per-preset (deliberate divergence)

A guard must **not** require these everywhere. Rationale is per key.

### Capability-scoped sections

Present only where the deployment has the capability.

| dotted key | present in | rationale |
|---|---|---|
| `control_system.*` | `hello-world`, `control-assistant` | every shipped preset reaches a control system; a deployment that disables the `controls` server has no hardware surface and states none |
| `archiver.*` | `hello-world`, `control-assistant` | rides with `control_system` |
| `ariel.*` | `control-assistant` | logbook deployments only |
| `logbook.*` | `control-assistant` | rides with `ariel` |
| `channel_finder.*` | `control-assistant` | `hello-world` serves no channel finder; control-assistant adds the benchmark corpus and the panel's web server |
| `facility_knowledge.*` | `control-assistant` | the only preset shipping an OKF corpus |
| `services.*` | `hello-world`, `control-assistant` | a deployment whose services are all file-backed deploys no standing service |
| `services.postgresql`, `services.qmd`, `services.graphdb` | `control-assistant` | only ARIEL needs the logbook stores |
| `cli.*` | `control-assistant` | console theming; `hello-world` omits it |
| `modules.web_terminals.*`, `deploy.fqdn` | `control-assistant` | the multi-user roster. `hello-world` is a single-terminal starting point |
| `bluesky.*`, `claude_code.servers.bluesky.enabled` | `control-assistant` | the plan queue |
| `claude_code.permissions.deny` | `control-assistant` | the floor every tier and app inherits |
| `web.*` | `control-assistant` | `hello-world` selects no panel and states no web key, so it renders no `web:` block at all |
| `simulation.default_scenarios` | `control-assistant` | the only preset whose facility tree ships scenarios to start in |

### Deliberately divergent defaults

| dotted key | values | rationale |
|---|---|---|
| `control_system.writes` | `allowed` in `control-assistant`; `never` in `hello-world` | control-assistant is the reference facility demonstrating the approval flow; hello-world is a read-only-by-default starting point |
| `web.theme` | live `light` in `control-assistant`; commented in `hello-world` | the default is `"osprey"` either way (`web_terminal/app.py`), so nothing behavioral is at stake |
| `channel_finder.benchmark.dataset_path` | `control-assistant` only | see below |
| `deployment.bind_address` (commented) | `hello-world`, `control-assistant` | absent ⇒ `127.0.0.1`, the safe state |

`control_system.writes` is the tier's write ceiling, `allowed` or `never`, and
is target-blind: which machine a session writes is the session's switch, not
the tier's. The tiers state it (`allowed` in `readwrite` and `admin`, `never` in
`readonly`), and an app render is pinned `never` by the composer.

`control_system.writes_enabled` and every
`control_system.connector.<type>.writes_enabled` are the retired spellings it
replaced. Nothing ships them — no preset, tier or app — so each carries
`rendered: false`. A profile that still states one is read once, at the single
parse point in `build_profile_document`, which answers `control_system.writes`
from them (`allowed` only when one is a literal `true`) and drops them.

`control_system.connector.<type>.limits_checking` is the same story for the
limits posture, and is likewise **not** parity-required. It overrides the
deployment-wide `control_system.limits_checking` block whole -- a per-type
block states both `enabled` and `mode` and then answers
alone. `database_path` has no per-type spelling: a
deployment mounts one limits database, so that leaf stays deployment-wide, and a
per-type block omitting it is complete rather than half-written. A profile that
states `database_path`, at either scope, stops the build: the build derives it,
writing `data/channel_limits.json` from `data/facility/limits.yaml` and naming
that file in every render that states a limits block.

No preset writes a per-type limits block, so none is in the union and every
per-type entry carries `rendered: false`.

### Keys nothing ships (`rendered: false`)

The manifest keys marked `rendered: false` have real readers, real spellings and
no writer among the framework template, the root presets or the apps. They are marked `rendered: false` so
the phantom-key check does not read them as manifest rot.

The flag is checked in **both** directions. A key carrying it that IS in the
union fails as a phantom-key contradiction naming the source that renders it,
because otherwise the flag is a one-way escape: a key that starts being shipped
keeps a marking saying nothing ships it, and the prose beside it goes on
describing a commented example.

The preset conversion added twenty of the seventy-five, in three groups:

| group | why nothing ships it |
|---|---|
| `control_system.connector.epics.gateways.*` (9) | a facility's Channel Access gateways cannot be guessed, and shipping someone else's would point a control system at a stranger's machine. `control-assistant` documents the whole block commented and the operator authors it at go-live |
| `archiver.mongodb_archiver.*` (9) | the build's archiver injector derives the connection block from the deployment's own `services.postgresql` / `va_archiver:` declaration. The preset states only `archiver.type: mongodb_archiver` |
| `control_system.connector.virtual_accelerator.response_delay_ms` (1) | the presets serve the simulator in process at the connector's own delay; a slower machine is authored per deployment |

### `api.providers` membership

Not a fixed set, and no longer a per-template question: `providers.yml` is the
single catalog, the framework template renders it verbatim, and a
`config: api.providers.*` key is refused. All ten providers ship. The guard
checks **shape** (every listed provider carries `base_url` + a complete tier
map), never membership.

### `facility` identity

The facility's display name is the build's facility identity. `facility.name`
and the top-level `facility_name` are both retired spellings, and no preset
ships either.

**Guard note:** both are resurrection candidates — neither may reappear in any
preset.

### `channel_finder.benchmark` — resolved

No preset but control-assistant ships a `benchmark:` block, and adding one
elsewhere would name a phantom path: only the control-assistant bundle ships the query corpus,
and the build copies the mode's benchmark query file to
`data/benchmarks/queries.json` (`cli/templates/scaffolding.py`); a bundle that
ships none has no `queries.json`.

So benchmark **is** deliberately control-assistant-only. The command still works
in a deployment without the corpus via `--queries-path`, which bypasses the config
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

## Where the web-terminal lint runs

`deployment/web_terminals/lint.py` validates `modules.web_terminals` at two
altitudes, and a guard touching either surface should know which one it is on:

| entry point | reads | run by |
|---|---|---|
| `lint_web_terminals(config)` | a rendered project `config.yml` | `osprey scaffold web-terminals lint`, and the pre-render gate inside `... render` |
| `lint_profile_config(config)` | a build profile's `config:` block (dotted keys, nested internally) | profile validation, before anything is built |

The profile-altitude pass skips the two checks that need a rendered project —
render `project_path` existence and the `build_profile` delta shape — because
`osprey init` only rewrites catalog entries into `tiers/<name>.yml` deltas at
materialization.
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
