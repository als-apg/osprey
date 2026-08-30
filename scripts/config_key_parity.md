# Config key parity classification

Input to the resurrection-guard manifest. Every dotted key the config-honesty
ledger touched is classified as **all-templates** (must appear in all five
shipped `config.yml.j2` templates, with the same default) or **per-template**
(deliberately present in some and absent from others, with the reason).

The five templates:

| id | path |
|----|------|
| `project` | `src/osprey/templates/project/config.yml.j2` |
| `control_assistant` | `src/osprey/templates/apps/control_assistant/config.yml.j2` |
| `hello_world` | `src/osprey/templates/apps/hello_world/config.yml.j2` |
| `ariel_standalone` | `src/osprey/templates/apps/ariel_standalone/config.yml.j2` |
| `channel_finder_standalone` | `src/osprey/templates/apps/channel_finder_standalone/config.yml.j2` |

Every claim below was produced by rendering each template with
`ChainableUndefined`, `yaml.safe_load`-ing the result, and — for MCP/approval
claims — feeding the parsed `claude_code` section to
`osprey.registry.mcp.resolve_servers`.

## Scope caveat — read before deriving guard rules

This document covers the keys the **config-honesty ledger touched**. It is *not*
an exhaustive enumeration of every key in the five templates, and the
"all-templates" table is *not* the complete set of keys that happen to appear in
all five.

Two rules follow for anything consuming this list:

1. **Do not auto-derive "present in all five ⇒ values must match."** Presence
   and value-equality are separate properties here. `deployed_services` is
   present in all five and its values deliberately differ; `container_runtime`
   is present in all five and its value must match. The tables below state which
   applies — infer neither from the other.
2. **Absence from this document is not evidence of anything.** A key not listed
   was outside the ledger's scope, not judged parity-exempt. Classify it on its
   own evidence before adding it to a guard.

## All-templates (parity-required)

These must be present in all five with the stated value. A guard should fire if
one goes missing or drifts.

| dotted key | value | notes |
|---|---|---|
| `project_name` | `{{ project_name }}` | rendered from build context |
| `build_dir` | `./build` | |
| `project_root` | `{{ project_root }}` | |
| `container_runtime` | `auto` | absent behaves identically to `auto` (`deployment/runtime_helper.py:27-43`), so this is a documentation guarantee, not a behavioral one |
| `approval.enabled` | `true` | |
| `approval.default_policy` | `always` | fail-closed for hook-wired tools not listed |
| `system.timezone` | `UTC` | pinned for reproducibility |
| `agent_data.base_dir` | `var/agent_data` | the only key naming this dir (`utils/workspace.py:41`, read at `:124-142`) |
| `file_paths.api_calls_dir` | `api_calls` | absent → dir named after the key itself (`utils/config.py:689-694`) |
| `file_paths.registry_exports_dir` | `registry_exports` | same fallback; also gates deploy-time pre-creation (`deployment/compose_generator.py:448-458`) |
| `claude_code.provider` | `{{ default_provider }}` | |
| `claude_code.default_model` | `{{ default_model \| default("haiku") }}` | |
| `api.providers.<name>.models.{haiku,sonnet,opus}` | complete tier map | every provider stanza in every template maps all three tiers; an unmapped provider now hard-errors (`build/claude_code_resolver.py`) |
| `api.providers.<name>.base_url` | present | |
| `api.providers.ollama.base_url` | `${OLLAMA_HOST:-http://localhost:11434}` | `${VAR:-default}` uses bash `:-` semantics (`utils/config.py:69-80`) |

## Per-template (deliberate divergence)

A guard must **not** require these everywhere. Rationale is per key.

### Capability-scoped sections

Present only where the app has the capability. The standalone bundles disable
the corresponding MCP servers outright, verified via `resolve_servers`.

| dotted key | present in | rationale |
|---|---|---|
| `control_system.*` | `project`, `control_assistant`, `hello_world` | the two standalone apps disable the `controls` server; no hardware surface |
| `archiver.*` | `project`, `control_assistant`, `hello_world` | rides with `control_system` |
| `execution.*` | `project`, `control_assistant`, `hello_world` | the two standalone apps disable the `python` server |
| `ariel.*` | `control_assistant`, `ariel_standalone` | logbook apps only |
| `logbook.composition.*` | `control_assistant`, `ariel_standalone` | rides with `ariel` |
| `channel_finder.*` | `control_assistant`, `channel_finder_standalone` | channel-finding apps only |
| `facility_knowledge.bundle_path` | `control_assistant` | the only bundle shipping an OKF corpus |
| `services.postgresql` | `control_assistant`, `ariel_standalone` | only ARIEL needs Postgres |

### Deliberately divergent defaults

| dotted key | values | rationale |
|---|---|---|
| `control_system.writes_enabled` | `true` in `control_assistant`; `false` in `project`, `hello_world` | `control_assistant` is the reference facility demonstrating the approval flow; the other two are read-only-by-default starting points |
| `claude_code.telemetry.enabled` | `true` in four; `false` in `channel_finder_standalone` | that bundle deploys no OpenObserve service, so telemetry has nowhere local to land |
| `hooks.debug` | live `true` in `control_assistant`, `ariel_standalone`, `channel_finder_standalone`; commented (→ `false`) in `project`, `hello_world` | absent reads as `false` (`osprey_hook_log.py:106`, `web_terminal/routes/config.py:175`). The app bundles ship the Safety-panel hook feed working out of the box; the two skeleton templates leave it off. Both comment texts match code. |
| `web.theme` | live `osprey` in `project`; commented in `control_assistant`; absent elsewhere | default is `"osprey"` either way (`web_terminal/app.py:607`), so nothing behavioral is at stake. `project` is the canonical-defaults template and states it live; `control_assistant` documents it commented; the minimal bundles omit it. |
| `artifact_server.*` | live in the four app templates; commented in `project` | `project` documents the defaults rather than pinning them |
| `cli.*` | `project`, `control_assistant` | console theming; the minimal bundles omit it |
| `deployment.bind_address` (commented) | `control_assistant`, `hello_world`, `ariel_standalone` | absent ⇒ `127.0.0.1`, the safe state |
| `channel_finder.benchmark.dataset_path` | `control_assistant` only | see below |
| `deployed_services` | present in **all five**, values deliberately diverge: `[]` (`channel_finder_standalone`), `[openobserve]` (`project`, `hello_world`), `[postgresql, openobserve]` (`control_assistant`, `ariel_standalone`) | presence is universal, the value is capability-scoped — the channel finder is file-backed and needs no standing service. A guard must check presence only, never value equality. |
| `api.providers.*` explanatory paragraph (comment) | `project`, `control_assistant` headers only | the 5-line `base_url` precedence + `default_model_id` paragraph is deliberate: those two are the documentation-carrying templates. `hello_world`, `ariel_standalone` and `channel_finder_standalone` keep one-line headers by design — the minimal bundles do not document key-by-key. Comment-only, so it never affects a rendered value. |

`control_system.connector.<type>.writes_enabled` is a per-connector-type
override of the global `control_system.writes_enabled` row above, and is
deliberately **not** parity-required. It is written commented in the templates
that carry the matching connector block, so it is never rendered and never
enters the union a parity guard would compare. Its semantics are tri-state:
absent inherits the global key, literally `true` arms writes for that connector
type, and any other value leaves them unarmed. Arming writes is a per-facility
decision, so no template may ship it live in any of the five.

`control_system.connector.<type>.limits_checking` is the same story for the
limits posture, and is likewise **not** parity-required. It overrides the
deployment-wide `control_system.limits_checking` block whole -- a per-type
block states both `enabled` and `allow_unlisted_channels` and then answers
alone -- and is written commented in the templates that carry the matching
connector block, so it is never rendered and never enters the union a parity
guard would compare. Only `virtual_accelerator`, `epics` and `live_standin`
carry entries; `mock` and `doocs` write no block. `database_path` has no
per-type spelling: a deployment mounts one limits database, so that leaf stays
deployment-wide, and a per-type block omitting it is complete rather than
half-written. How relaxed a machine's limits checking may be is a per-facility
decision, so no template may ship either leaf live in any of the five.

### `api.providers` membership

Not a fixed set. `project`, `control_assistant` and `hello_world` ship the full
10-provider roster; `ariel_standalone` and `channel_finder_standalone` ship a
6-provider subset (`cborg`, `openai`, `anthropic`, `google`, `ollama`,
`als-apg`), omitting `amsc-i2`, `stanford`, `argo`, `ds4`. The guard should
check **shape** (every listed provider carries `base_url` + a complete tier
map), not membership.

### `facility` identity

`facility.name` is canonical; top-level `facility_name` is the retired spelling,
still honored as a fallback (`utils/facility.py:35-38`). No template ships
`facility_name` any more. `ariel_standalone` and `channel_finder_standalone`
ship a live `facility.name` (`"Example Research Facility"`); the other three
ship the whole block commented and fall back to the project name. `facility.prefix`
is commented everywhere — only the multi-user web-terminal stack reads it.

**Guard note:** top-level `facility_name` is a resurrection candidate — it must
not reappear in any template, though the *reader* fallback stays.

### `channel_finder.benchmark` — resolved

`channel_finder_standalone` ships no `benchmark:` block, and adding one would
name a phantom path: only the `control_assistant` bundle ships the query corpus
(`templates/apps/control_assistant/data/benchmarks/cross_paradigm/queries/tier{1,3}_queries.json`),
and `materialize_tier_artifacts` returns silently for bundles with no
`data/channel_databases/tiers/` subtree (`cli/templates/scaffolding.py:405-407`),
which `channel_finder_standalone` does not have.

So benchmark **is** deliberately control-assistant-only. The command still works
in the channel-finder app via `--queries-path`, which bypasses the config read
(`cli/channel_finder_cmd.py:608`). The bare `KeyError` on the config subscript
was replaced with an error naming both remedies
(`services/channel_finder/benchmarks/runner.py`), and the template documents the
omission.

## Open items for the manifest (not fixed here)

1. **Test fixtures are out of guard scope.** `tests/utils/fixtures/legacy_config_all_deleted_keys.yml`
   deliberately holds every retired key. If the guard scans `tests/`, it must
   allowlist that fixture.
2. **`approval.tools.entry_create` is inert in `hello_world`** — that template
   disables the `ariel` server, so the tool never exists. Harmless (the policy
   is `always`, and the block is fail-closed anyway), but it is a policy for a
   tool that cannot run. Left alone: the block is task-1.6-owned.
3. **`draft_concept` is approval-governed but listed nowhere.** `resolve_servers`
   shows `osprey_facility_knowledge` enabled in `project`, `control_assistant`
   and `hello_world`, and its `draft_concept` tool carries the approval hook. It
   is absent from every `approval.tools` block, so it falls to
   `default_policy: always` — fail-closed and correctly described by the
   shipped comment. No change needed; recorded so it is not mistaken for drift.
4. **`facility.timezone` is read but shipped by no template**
   (`deployment/web_terminals/env_production.py:225`,
   `deployment/web_terminals/render.py:142`), distinct from `system.timezone`.
   Hidden-key stanza candidate — task 6.1 territory, not fixed here.
5. **`facility.prefix` has a stated convention and no validator — by design.**
   The 2-6-character lowercase-alnum-plus-hyphens rule this entry was opened
   against came from a schema document that no longer exists (it went with the
   `facility-config.yml` surface). The convention survives in prose only — the
   two presets and the `control_assistant` template state it — and the only
   validation anywhere is a non-emptiness lint
   (`deployment/web_terminals/lint.py:648-679`, `_check_empty_facility_prefix`).

   The absence is real, not an artifact of an incomplete search. The same lint
   module defines `_USERNAME_CHARSET_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")`
   (`lint.py:48`) and enforces it at two sites — usernames
   (`lint.py:250`, `_check_username_charset`) and persona names
   (`lint.py:550`, `_check_persona_charset`). So this codebase does write
   charset checks where it wants them, and has none for `facility.prefix`.

   **The rule is not merely unenforced — it is inert.** Tracing `facility.prefix`
   to its sinks: two container-name interpolations plus the personas path, and
   nothing else. It never becomes an nginx location key or a URL segment, which
   is the specific reason usernames and persona names *do* get the charset
   regex. There is no sink at which violating the 2-6/lowercase rule breaks
   anything.

   **Resolved: it stays a convention — do NOT add a validator.** A new charset
   check would reject configurations that work correctly today, turning a
   cosmetic inconsistency into a breaking change. The templates and both presets
   state the enforced rule (non-emptiness) alongside the Docker constraint,
   which is the right two-altitude framing; the one document that presented the
   convention as a hard requirement is gone, so nothing contradicts the code any
   more. Recorded here so the gap is not mistaken for an oversight and
   "fixed" with a validator later.

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
