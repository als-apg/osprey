# `osprey build` output inventory

Traced against `/Users/thellert/code/osprey` at `a12268aff`, verified against the rendered tree in `/Users/thellert/code/osprey/my-control-assistant/build/`.

Scope: everything `osprey build` writes into `build/`, including per-persona subdirectories and `build/.image/*`. For each output: the source template, the governing `profile.yml` key(s) or IMPLICIT, what its content is computed from, and the override mechanism that exists today.

---

## How the render is assembled

`_build_repo` (`src/osprey/cli/build_cmd.py:3065`) renders everything into `build/.tmp/<repo>/` and atomically swaps it onto `build/` (`_render_zones` at `build_cmd.py:268`, `_STAGE_DIRNAME` at `build_cmd.py:156`, `_swap_in_render` at `build_cmd.py:416`).

One repo-level pass plus **six** calls to `_render_project` (`build_cmd.py:1823`): the deployment, one per `personas/*.yml` delta, and one image build context per each of those. Every pass runs the identical steps with the identical inputs bar the profile.

Inside one pass the order is:

1. base template render — `manager.create_project` (`src/osprey/cli/templates/manager.py:158`)
2. config overrides — `_apply_config_overrides` (`build_cmd.py:2085`)
3. Reach Contract projection for attached renders (`build_cmd.py:2101-2148`)
4. service injection — `_inject_services` (`build_cmd.py:2153`, def at `build_cmd.py:3958`)
5. panel selection projection — `panel_selection_overrides` (`build_cmd.py:2163`)
6. **profile convention copies** — `_apply_conventions` (`build_cmd.py:2204`, def at `src/osprey/cli/build_persistence.py:200`)
7. graph index + VA channel manifest (`build_cmd.py:2237-2266`)
8. manifest — `manager.generate_manifest` (`build_cmd.py:2298`) then `_stamp_repo_manifest` (`build_cmd.py:2306`)
9. **Claude Code re-render** — `manager.regenerate_claude_code` (`build_cmd.py:2308`, def at `manager.py:646`)

The Claude Code artifacts are therefore rendered twice — once inside `create_project` (`manager.py:450-452`) and again at step 9 — and the second render is the one that lands.

### The gate that decides which `.claude/` files exist

`resolve_manifest_outputs` (`src/osprey/cli/templates/manifest.py:189`) seeds the allowed set with four paths unconditionally (`manifest.py:205`):

```
CLAUDE.md   .mcp.json   .claude/settings.json   .claude/statusline.py
```

then adds one output path per name in the profile's `hooks:` / `rules:` / `skills:` / `agents:` / `output_styles:` lists, resolved through `BuildArtifactCatalog` (`src/osprey/services/build_artifacts/catalog.py:491`). Category-key → canonical-prefix mapping is `_MANIFEST_CATEGORY_PREFIX` (`manifest.py:39`).

Anything rendered outside that set is **deleted** from disk (`src/osprey/cli/templates/claude_code.py:1873-1881`), and any template that renders to whitespace is deleted too, with its parent directory if empty (`claude_code.py:1885-1891`).

The selection is persisted into `.osprey-manifest.json` under `artifacts` (`manifest.py:498`), and regen reads it back from there in preference to the preset (`claude_code.py:2049-2070`).

---

## 1. Repo-root files in `build/`

| output path | source template | governing profile key(s) | content computed from | override mechanism today |
|---|---|---|---|---|
| `build/config.yml` | `src/osprey/templates/project/config.yml.j2` (`scaffolding.py:215`) | IMPLICIT existence; content from `provider:`, `model:`, `channel_finder_mode:`, `default_panel:`, `panel_presets:`, `web_panels:`, `deployment.port_base`, `environment.*` | `project_name`, `project_root`, `osprey_ports.artifact`, `default_provider`, `default_model`, `default_pipeline`, `enable_in_context`/`enable_hierarchical`/`enable_middle_layer`, `ariel_server_on`, `builtin_panels`, `selected_web_panels`, `provider_catalog` (rendered verbatim from `providers.yml`), `preset`. Only keys in `osprey.cli.derived_keys.DERIVED_KEYS` plus `api.providers` are rendered here | the profile's `config:` block, applied over the render (`build_cmd.py:2085`). **Reserved** against the `project/` mirror (`profile_conventions.py:320`) |
| `build/CLAUDE.md` | `templates/claude_code/CLAUDE.md.j2` by default; `CLAUDE.ariel.md.j2`, `CLAUDE.knowledge.md.j2`, `CLAUDE.channel-finder.md.j2` are the alternates (`claude_code.py:1813-1826`) | `claude_md_template:` picks which file; **existence is IMPLICIT** | only `agents` (`.name`, `.enabled`, `.description`) and `agent_data_root`. Every prose section — the "Verify first" instruction, the delegation mandates, the tool-confinement language — is hardcoded in the template. The agent roster block is gated per enabled agent | `claude_md_template:` selects among the 4 shipped templates; `osprey scaffold claim claude-md`. **Reserved** against the `project/` mirror (`profile_conventions.py:338`) |
| `build/.mcp.json` | `templates/claude_code/mcp.json.j2` (`claude_code.py:1805`) | IMPLICIT existence; membership from `mcp_servers:` and `config.claude_code.servers.<name>.enabled` | `servers` resolved by `resolve_servers` from `FRAMEWORK_SERVERS` in `src/osprey/registry/mcp.py`. Nine definitions: `controls` (`:132`), `phoebus` (`:173`, **default off**), `python` (`:220`), `osprey_workspace` (`:249`), `ariel` (`:306`), `osprey_facility_knowledge` (`:347`), `bluesky` (`:363`, **default off**), `health` (`:474`, **default off**), `channel-finder` (`:496`, `condition="channel_finder_pipeline"` at `:503`), `graph` (`:515`, `condition="graphdb_configured"` at `:527`). Each block is `command` = the derived venv interpreter, `args` = `-m <module>`, `env` from the definition | `mcp_servers:` adds facility servers; `config.claude_code.servers.*.enabled` flips framework ones; `claude_code.servers.<name>.extends: phoebus` clones one. **Reserved** against the `project/` mirror (`profile_conventions.py:335`) |
| `build/README.md` | `templates/project/README.md.j2` (`scaffolding.py:217`) | IMPLICIT | `project_name`, `project_description` only | `project/` mirror |
| `build/Dockerfile` | `templates/project/Dockerfile.j2` (`scaffolding.py:219`) | IMPLICIT | `project_name`, `osprey_ports.web`, `claude_code_cli_version`, `pip_dependency_args`, `is_setup_patch_capable` (derived by `_profile_setup_patch_capable`, `build_cmd.py:3572`, from whether the profile denies the setup-patch tool) | `project/` mirror |
| `build/entrypoint.sh` | `templates/project/entrypoint.sh` — **static copy**, no `.j2` (`scaffolding.py:233`, `:242-245`) | IMPLICIT | nothing; no Jinja in the file at all | `project/` mirror |
| `build/.gitignore` | `templates/project/gitignore`, static, renamed (`scaffolding.py:247-250`) | IMPLICIT | hardcoded | `project/` mirror |
| `build/.dockerignore` | `templates/project/dockerignore`, static, renamed (`scaffolding.py:252-256`) | IMPLICIT | hardcoded | `project/` mirror |
| `build/.env.example` | `templates/project/env.example.j2` (`scaffolding.py:216`) | IMPLICIT | `provider_api_keys`, `active_provider_vars`, `service_token_vars`, `env_required`, `env_defaults`, `project_name` — derived from `provider:`, `env:`, and the set of deployed services | `env:` keys. **Reserved** against the `project/` mirror (`profile_conventions.py:337`) |
| `build/.osprey-manifest.json` | no template — `generate_manifest` (`templates/manifest.py:498`), then `_stamp_repo_manifest` (`build_cmd.py:556`) | IMPLICIT | `schema_version`; `creation.osprey_version` (release lineage, not running version), `creation.timestamp`, `creation.template` (the recorded preset or `null`), `creation.preset_hash` (content hash of the resolved profile), optional `creation.claude_md_template`; `build_args` (`default_provider`, `default_model`, `profile_path_abs`, optional `channel_finder_mode`); `reproducible_command`; `file_checksums` of the whole render; `artifacts` (the profile's own five lists); `user_owned`. `_stamp_repo_manifest` adds `creation.<key_digests>` — per-top-level-profile-key digests, **deployment pass only** (`build_cmd.py:598-599`) | none. **Reserved** against the `project/` mirror (`profile_conventions.py:321`) |
| `build/env-chain.json` | none — `record_env_chain_membership` (`src/osprey/deployment/compose_generator.py:304`, single call site `:3599`) | **IMPLICIT, unconditional** — written even for a deployment with zero services | `{"env_chain": [".env.shared", ".env"]}` — bare filenames in chain order, lowest precedence first, missing members skipped. Reflects which files exist at the **repo root**, not anything in `profile.yml`. Constant at `compose_generator.py:108` | none |
| `build/pyproject.toml` | none — `_write_project_pyproject` (`src/osprey/cli/build_environment.py:370`, called `:986`) | `dependencies:`, `environment.packages`, `environment.python`, `environment.inherit_exclude`, `osprey_install:` | provenance header + `[project]` with `name`/`version`/`requires-python`/`dependencies`. Deliberately **no** `[build-system]` (`build_environment.py:430-436`). Created at `build/pyproject.toml` and copied into the stage (`build_cmd.py:3311-3313`) so record and environment ship together | `--skip-deps` suppresses it entirely |
| `build/.venv/` | none — `_create_project_venv` (`build_environment.py:737`, called `build_cmd.py:3310`) | same keys as above | uv `venv --clear` if available, else stdlib `venv`. Base interpreter from `environment.python` else `sys.executable`. Install list = `osprey_install:` spec + merged `dependencies:` / `environment.packages` | `--skip-deps`; `UV` env var picks the uv binary. Written at its final path and renamed *into* the stage at swap (`build_cmd.py:448-450`) — it is the one artifact that cannot be rendered elsewhere and moved |
| `build/triggers.yml` | `<profile_dir>/<dispatch.triggers>`, else the bundled `src/osprey/profiles/triggers/<name>.yml` (`src/osprey/cli/build_injectors.py:527-534`) | **`dispatch:`** (specifically `dispatch.triggers`) | copied with `copy2`, then **patched in place**: `dispatcher.max_concurrent_runs` and `max_queue_depth` from `dispatch.*`, and `dispatcher.dispatch_target` rewritten — `http://localhost:<worker port>` on `dispatch.network: host`, else port-rebased against `dispatch.worker_port_base` (`build_injectors.py:542-581`) | author your own file and point `dispatch.triggers` at it |
| `build/.env.chain-state.json` | — | **not written by `build`** | written by `osprey up`: `ENV_CHAIN_STATE_RELPATH` at `src/osprey/deployment/container_lifecycle.py:2551`, `_write_chain_state` at `:2619`, reached only from `_start_stack` (`:5964`). Content is `{"version": 1, "recorded_at": ..., "shared_value_digests": {KEY: sha256}}` — digests only, never values | n/a |
| `build/.env.merged` | — | **not written by `build`** | `up`-only, podman branch only (`container_lifecycle.py:2387`, `:2487`) | n/a |

**No `.env` or `.env.shared` is ever written into the render**, by design — `create_project_structure`'s docstring states it (`scaffolding.py:187-201`). The repo-root `.env` is the deployment's whole secret store.

**The one write `build` makes outside `build/`**: `_wire_build_derived_env` (`build_cmd.py:2947`, called `:3450` after the swap) appends `VA_CHANNELS_FILE` and `VA_LATTICE` to the repo-root `.env`. Append-only through `append_profile_env`; a value already on file always wins and a disagreement is warned about, never resolved.

Other side effects outside `build/`: `_ensure_state_zone` creates `var/agent_data` and `var/audit` (`build_cmd.py:280`); `_backup_outgoing_claude_artifacts` snapshots the outgoing `.claude`/`.mcp.json`/`CLAUDE.md` into `var/agent_data/backup/claude-code-<UTC stamp>/` (`build_cmd.py:509`).

---

## 2. The `.claude/` tree

Rendered by `create_claude_code_integration` (`claude_code.py:1749`). It walks `src/osprey/templates/claude_code/claude/` recursively, maps `claude/` → `.claude/`, strips the `.j2` suffix, and **skips any file under a `_`-prefixed directory** (`claude_code.py:1855-1857`) — that is what keeps `_shared/` and `_terminology/` off disk. Hook `.py` files are chmod'd executable afterwards (`claude_code.py:1908-1913`).

### 2a. Always present, not nameable in the profile

| output path | source template | governing key | content computed from | override mechanism today |
|---|---|---|---|---|
| `.claude/settings.json` | `templates/claude_code/claude/settings.json.j2` | **IMPLICIT** — always in the allowed set (`manifest.py:205`) | see the breakdown below | `config.claude_code.permissions.{allow,deny,ask,remove_deny,remove_ask}`; `config.claude_code.hooks` appends declared hook wiring; `osprey scaffold claim settings-json`. **Reserved** against the `project/` mirror (`profile_conventions.py:326`) |
| `.claude/statusline.py` | `templates/claude_code/claude/statusline.py` — **static Python**, 111 lines | **IMPLICIT** | nothing; verbatim copy. Reads Claude Code state JSON from stdin, emits one colored line, looks up the framework version via `import osprey` at run time | `osprey scaffold claim statusline` only. The `project/` mirror is **refused**: `reserved_path_channel` falls through to `_framework_rendered_outputs()` (`profile_conventions.py:637`), which is derived from the catalog |

#### `settings.json` content, line by line

Read in full from the template. Three categories:

**Hardcoded — no profile key reaches them:**

- `"showThinkingSummaries": true`
- `"outputStyle": ".claude/output-styles/control-operator.md"` — a **literal**. It does not follow the `output_styles:` list; a profile that selects a different output style still gets this path written.
- `"enableAllProjectMcpServers": true`
- `"skillOverrides"` switching `update-config`, `keybindings-help` and `fewer-permission-prompts` to `"off"` — the harness-config skill family is disabled so a deployed operator agent cannot reconfigure its own harness.
- The `_PROMPT_PROVIDER_NOTE` string.

**Computed:**

- `theme` — from `web.theme`, and only when that value pins a concrete mode (`_terminal_theme`, `claude_code.py:463`). A theme *family* pins nothing and the key is absent.
- `statusLine.command` — `"<derived interpreter>" "$CLAUDE_PROJECT_DIR/.claude/statusline.py"`.
- `permissions.allow` — `Read(<agent_data_root>/**)` and four subpaths keyed off `agent_data.base_dir`; `Read(<textbooks_root>/**)` when set; per enabled server, each `permissions_allow` tool and each `fixed_allow` entry; `NotebookEdit` on the artifacts and notebooks trees; **one `Task(<name>)` per enabled agent**; then `facility_permissions.allow`.
- `permissions.deny` — three parts in order: `deny_defaults` (the `DENY_DEFAULTS` constant at `claude_code.py:52`) minus `remove_deny`; the profile's own `deny` minus `remove_deny`; then `killswitch_deny` appended **last and never filtered**. That last part is the whole reason the kill switch renders through its own context key: `remove_deny` is profile-authored, so anything it can reach is liftable, and a writes-off deny must not be.
- `permissions.ask` — per-server `permissions_ask` and `fixed_ask`, plus `facility_permissions.ask`, minus `remove_ask`.

**Hook event wiring — entirely hardcoded in the template, not derived from the `hooks:` list:**

- `SessionStart` always wires `osprey_config_drift.py` (timeout 5), `osprey_panels_context.py` (5), `osprey_control_context.py` (3), plus a second entry with matcher `"startup|resume|clear"` running `osprey_turn_state.py` (3). The matcher is load-bearing: it keeps `compact` out, because auto-compaction happens inside a running turn.
- `UserPromptSubmit` always wires `osprey_focus_validate.py` (2), `osprey_workspace_delta.py` (3), `osprey_control_context.py` (3), `osprey_turn_state.py` (3).
- `Stop` and `StopFailure` always wire `osprey_turn_state.py` (3).
- `PreToolUse` / `PostToolUse` are the only data-driven events: rules come from `framework_pre_hooks`/`framework_post_hooks` (built by `_build_framework_hook_rules` from selected hooks' frontmatter, `claude_code.py:1094`) **plus** each enabled server's `hooks_pre`/`hooks_post` from `registry/mcp.py`. Declared profile hooks are appended last and are purely additive.
- Events the framework wires nothing on exist as keys only when `config.claude_code.hooks` declares them (`declared_extra_events`).

**Gap worth flagging.** A profile can drop `panels-context` from `hooks:` and the file will not be rendered, but `settings.json` will still carry a `SessionStart` command pointing at `.claude/hooks/osprey_panels_context.py`. The `2>/dev/null || true` suffix on every framework hook command is what keeps that from failing loudly. The same applies to `config-drift`, `control-context`, `focus-validate`, `workspace-delta` and `turn-state`.

### 2b. Selected by the profile's artifact lists

All gated by name. `validate_artifacts` (`src/osprey/cli/templates/artifact_library.py:207`) hard-fails the build with a did-you-mean on an unknown name, so dead names are impossible in these five categories.

| output path | source template | governing key | content computed from | override mechanism today |
|---|---|---|---|---|
| `.claude/hooks/osprey_*.py` (14 files available) | `templates/claude_code/claude/hooks/osprey_*.py` — **static Python** | `hooks:` list entry | nothing; verbatim copies. Each carries YAML frontmatter in its module docstring (`name`, `description`, `event`, `tools`, `safety_layer`, `wiring`, `timeout`) that `_build_framework_hook_rules` reads to build `PreToolUse`/`PostToolUse` rules | `hooks/` convention dir overlays them (`profile_conventions.py:143`, shape `FILE`); `osprey scaffold claim hooks/<filename>` |
| `.claude/hooks/hook_config.json` | `templates/claude_code/claude/hooks/hook_config.json.j2` | `hooks: - hook-config` | `server_prefixes` and `approval_prefixes` built by walking every enabled server's `hooks_pre` rules for `osprey_approval.py`; `write_tools` built the same way for `osprey_writes_check.py`, unioned with `config.control_system.write_tools`; plus `mixed_read_write_tools` and `lane_addressed_tools` (from `QUEUE_CONTROL_TOOLS`). Every key emitted unconditionally, never under the kill-switch condition | **Reserved** against the `project/` mirror (`profile_conventions.py:329`). The exact reservation deliberately beats the `hooks/` convention prefix (`profile_conventions.py:620-628`) so a hand-written copy cannot freeze the write-safety layer's own configuration |
| `.claude/agents/<name>.md` (7 available) | `templates/claude_code/claude/agents/*.md.j2` | `agents:` list | `claude_code_model_spec.agent_tier(<name>)` for the `model:` frontmatter, `maxTurns`, explicit `tools`/`disallowedTools` lists, `facility_name`, `enabled_agents`. For `channel-finder`: `channel_finder_tools`, `channel_finder_mode`, `channel_finder_hierarchy`, `facility_vocabulary`. Whole file wrapped in `{% if "<name>" in enabled_agents %}` | `agents/` convention dir (`profile_conventions.py:122`); persona delta `exclude:`; `scaffold claim agents/<name>` |
| `.claude/rules/<name>.md` (10 available) | `templates/claude_code/claude/rules/` — 4 static (`artifacts.md`, `error-handling.md`, `python-execution.md`, `workflows.md`), 6 Jinja | `rules:` list | `control_system_type`, `enabled_servers`, `target_switch_enabled` (control-system-safety); `enabled_agents` (data-visualization, safety, facility); `system_timezone` (timezone); `control_system_type` (test-ioc-safety) | `rules/` convention dir (`profile_conventions.py:108`) |
| `.claude/skills/<name>/SKILL.md` (11 available) | `templates/claude_code/claude/skills/*/` — 8 static, 3 Jinja | `skills:` list | `facility_name`, `enabled_servers`, `bluesky_plan_index` (a build-time plan index), `preset`, `channel_finder_pipeline`, `project_name`, `agent_data_root`. `session-report` is multi-file: `SKILL.md` plus `reference.md`, matched by the catalog's prefix rule (`manifest.py:212-220`) | `skills/` convention dir (`profile_conventions.py:115`, shape `DIRECTORY` — whole-directory replace) |
| `.claude/output-styles/control-operator.md` | `templates/claude_code/claude/output-styles/control-operator.md.j2` | `output_styles:` list | `enabled_agents` gates one section. Frontmatter `keep-coding-instructions: false` | `output-styles/` convention dir (`profile_conventions.py:136`) |
| `.claude/rules/facility.md` | `templates/claude_code/claude/rules/facility.md.j2` | `rules: - facility` | rendered **create-only** and auto-registered user-owned (`claude_code.py:1831-1843`, `auto_register_user_owned` at `claude_code.py:1073`). Never rewritten by a later build | edit in place — it is yours after the first build |
| `.claude/commands/` | **no framework template exists** | only the profile's `commands/` convention dir (`profile_conventions.py:129`) | n/a | the convention dir is the only writer. Nested subdirectories are preserved, since Claude Code namespaces commands that way |

#### Conditionally dead names

Two entries in the example profile's `rules:` list produce no file:

- `test-ioc-safety` — the whole template is gated on `control_system_type in ('epics','virtual_accelerator')`; this deployment is `live_standin`.
- `data-visualization` — gated on `"data-visualizer" not in enabled_agents`; that agent **is** enabled.

Both render to whitespace and are unlinked with no message (`claude_code.py:1885-1891`). Compare: `build/.claude/rules/` holds 8 files for a 10-name list. This is the one way a valid profile name silently yields nothing.

#### Listable but inert

`hooks: target-state` and `hooks: hook-log` copy `osprey_target_state.py` (1092 lines) and `osprey_hook_log.py` (714 lines). Neither carries a frontmatter `event:`; both are stdlib-only shared libraries the other hooks import (`load_hook_config`, `get_repo_root`, the control-target reader). They are wired to no event. The example profile's own comment says so: *"helper, not wired to an event"*.

---

## 3. Per-persona subdirectories — `build/<repo-name>-<persona>/`

`_persona_deltas` (`build_cmd.py:2344`) enumerates the direct children of `personas/` that are files and are not dot-prefixed, sorted by name (`:2375-2382`). No suffix filter — the directory holds deltas and nothing else. Each is containment-checked with `resolve_profile_root` (`:2394-2408`); a symlinked delta raises.

`_render_persona_projects` (`build_cmd.py:2412`) resolves each delta merged over the root profile (with any `.env.variant` overlay applied too, `:2458`) and calls `_render_project` with `project_name=f"{repo_root.name}-{delta.stem}"`, `output_dir=zones.stage`, `deployment=False`.

Each persona directory is a **complete render**: its own `config.yml`, `CLAUDE.md`, `.mcp.json`, `.claude/`, `data/`, `docker/web-terminal-context/`, `.osprey-manifest.json`, `Dockerfile`, `entrypoint.sh`, `README.md`, `.env.example`, `.gitignore`, `.dockerignore`.

Personas are rendered **before** the compose files (`build_cmd.py:3384-3392`) because the `bluesky_web` sidecar's per-persona roster grants read each persona's rendered `config.yml`.

A persona does **not** get — confirmed against code, not just the docstring at `build_cmd.py:1846-1856`:

| excluded | confirmed by |
|---|---|
| `.venv/` | `_create_project_venv` called once, from `_build_repo` (`build_cmd.py:3310`) |
| `pyproject.toml` | same call |
| `services/` and the compose files | `_render_compose_files` called once (`build_cmd.py:3392`) |
| lifecycle phases (`pre_build`/`post_build`/`validate`) | `_run_lifecycle_phase` only in `_build_repo` (`:3290`, `:3415`, `:3423`) |
| provider-credential report | gated `if deployment:` (`build_cmd.py:2283`) |
| manifest per-key digests | `key_digests=deployment` (`build_cmd.py:2306`) |

A persona also passes `extra_known=already_warned` so it does not repeat the deployment's unknown-root-entry warning, and inherits `resolved.excluded_artifacts` into `_apply_conventions` (`build_cmd.py:2212`).

**Observed deltas in the example deployment** (5 personas):

| persona | `.claude/` difference from the deployment render |
|---|---|
| `admin` | adds `skills/setup-mode/SKILL.md` |
| `readonly` | identical file set; differs in `settings.json` content (writes-off kill switch) |
| `readwrite` | identical file set |
| `knowledge` | drops 4 agents, `hooks/osprey_limits.py`, `hooks/osprey_writes_check.py`, 3 rules, all 8 skills |
| `logbook` | drops all 7 agents, `cf-feedback-capture`, `limits`, `writes-check`, 2 rules, 5 skills; keeps `logbook-deep-research` and `session-report` skills |

Personas are **also** given an image build context under `build/.image/<repo>-<persona>/`, in addition to the flat render, not instead of it (`build_cmd.py:2875-2882`).

---

## 4. `build/.image/<name>/` — container build contexts

`_render_container_projects` (`build_cmd.py:2850`), one per image: the deployment plus one per persona. **Skipped entirely under `--runtime-root`** (`build_cmd.py:3401-3407`) — that build is already aimed at a runtime elsewhere.

`_render_container_project` (`build_cmd.py:2762`), five steps, all IMPLICIT — no profile key names any of it:

1. A full `_render_project` into `image_root/.render/<name>`, renamed to `image_root/build/` (`:2823-2835`), with `runtime_root = "/app/<project_name>"` and `runtime_interpreter = "/usr/local/bin/python"` (`_CONTAINER_INTERPRETER`, `build_cmd.py:183`) substituted into every MCP server command and every hook command. Without that substitution every server and hook in a container render would name the build host's venv.
2. `_stage_source_zone` (`build_cmd.py:2470`) copies the **entire repo root by exclusion**: `repo_root.iterdir()` minus `_NON_SOURCE_ROOT_ENTRIES` = `{build, var, .git, <merged compose filename>}` (`build_cmd.py:165-167`), minus anything whose name starts with `.env` at the root **and at any depth** inside copied directories (the `_ignore_env_files` hook, `:2504-2505`). Exclusion-based deliberately, so it cannot drift from the drift fingerprint's fold.
3. `_strip_secrets` (`build_cmd.py:2517`) rglobs `.env*` and unlinks every match except `.env.example` — the third independent guard after copy-time exclusion and the `.dockerignore`.
4. `_write_image_context_dockerignore` (`build_cmd.py:2547`) **derives** the context-root `.dockerignore` from `image_root/build/.dockerignore` — the rendered copy of `templates/project/dockerignore`. Each non-comment line is re-spelled `**/`-anchored (`:2599-2600`) because the context root is now one level above the render. Two deliberate deviations: `Dockerfile` is dropped from the exclusions (`:2597`) so the image ships `build/Dockerfile`, and there is no `build/` entry (`:2580-2583`). `_prune_ignored_entries` (`:2649`) then physically **deletes** everything those patterns name, honoring `!`-negations, so the tree the build fingerprints is byte-for-byte the tree the image receives.
5. `_relocate_container_manifest` (`build_cmd.py:2694`) rewrites `image_root/build/.osprey-manifest.json`: `build_args.profile_path` → `/app/<name>/<relpath>` only if already present; `build_args.profile_path_abs` → `/app/<name>/profile.yml` unconditionally; `creation.preset_hash` re-stamped from the image root's own `profile.yml`. Best-effort — an unreadable manifest is skipped at DEBUG.

`_prune_runtime_state_from_stage` is applied to each image's `build/` too (`build_cmd.py:3439-3446`), so runtime-state directories are not baked into the image where the agent-data volume mounts over them.

`--dev` stages a locally built wheel into each service build context that has a `Dockerfile` (`compose_generator.py:1586`).

---

## 5. `build/services/`

`_inject_services` (`build_cmd.py:3958`) runs inside **every** render pass, but only the deployment pass's injected list is reported (`build_cmd.py:3339`). It returns `[]` immediately when `deploy_services: false` (`:3981-3986`). Injector order is load-bearing (`:3971-3975`).

| service directory | injector | governing profile key |
|---|---|---|
| `services/` + `docker-compose.yml.j2` + `_env_axis.j2` + `_network_axis.j2` | `_copy_service_templates` (`build_injectors.py:300`) | **IMPLICIT / unconditional** — the root compose template is always copied so `osprey up` works even with no deployed services |
| one dir per entry | `_inject_profile_services` (`build_injectors.py:407`) | `services:` |
| `event_dispatcher/`, `dispatch_worker/` | `_inject_dispatch` (`build_injectors.py:486`) | `dispatch:` |
| `nextcloud_bridge/` | `_inject_nextcloud_bridge` (`build_injectors.py:1630`) | `nextcloud_bridge:` |
| `gchat_bridge/` | `_inject_gchat_bridge` (`build_injectors.py:1730`) | `gchat_bridge:` |
| `bluesky/` (+ `bluesky_devices.yml`, `user_group_permissions.yaml`) | `_inject_bluesky` (`build_injectors.py:1030`) | `bluesky:` (reads `virtual_accelerator:` and the port base) |
| `bluesky_web/` (+ `channels.json`) | `_inject_bluesky_web` (`build_injectors.py:1524`) | `bluesky_web:` |
| `virtual_accelerator/` | `_inject_va` (`build_injectors.py:1359`) | `virtual_accelerator:`; `live_standin` registers a second instance sharing the same directory (`:1435-1440`) |
| `mongodb/`, `archiver_recorder/`, and `build/data/simulation/` | `_inject_va_archiver` (`build_injectors.py:1827`, dir at `:1880`) | `va_archiver:` |
| `graphdb/`, `openobserve/`, `postgresql/`, `qmd/` | via `services:` / `config.services.*` blocks | `config.services.<name>.*` |

Per-service outputs, written by `compose_generator.setup_build_dir`:

| output | writer | key |
|---|---|---|
| `services/docker-compose.yml` (the network declaration) | `compose_generator.py:3631-3636` | only when `deployed_services` is non-empty |
| `services/<svc>/docker-compose.yml` | `compose_generator.py:2971` | one per deployed service; template found via `services.<name>.path` |
| `services/<svc>/config.yml` (flattened, `${VAR}` unexpanded) | `compose_generator.py:3037-3078` | IMPLICIT per service |
| `services/<svc>/repo_src/`, `requirements.txt`, `pyproject_user.toml` | `compose_generator.py:2981-2998` | `services.<name>.copy_src: true` |
| `services/<svc>/<additional dirs>` | `compose_generator.py:3001-3027` | `services.<name>.additional_dirs` — this is how `services/event_dispatcher/triggers.yml` appears (`build_injectors.py:620-623`) |
| `services/<svc>/*.whl` | `_stage_dev_wheel_for_context` (`compose_generator.py:1586`) | `--dev` CLI flag |
| `services/qmd/index.yml` | `render_service_templates` (`compose_generator.py:2053`) | the qmd service block |

`_refresh_service_dir` does `rmtree` + `copytree` — **unless** the name is in `config.scaffold.user_owned` (`_user_owned_services`, `build_injectors.py:134-153`). That is the `osprey scaffold claim services/<name>` override. Every injector preserves the one author-owned key `env:` via `_carry_authored_keys` (`build_injectors.py:242`).

**Not written by `build`.** `docker-compose.web.yml`, `nginx.conf` and `landing.html` come from `src/osprey/templates/modules/web_terminals/` and are rendered by `osprey up` (`src/osprey/deployment/web_terminals/provision.py:1280`). The `--runtime-root` path also skips compose rendering and `env-chain.json` entirely (`build_cmd.py:1061-1066`).

---

## 6. `build/data/` and `build/docker/`

| output | source | governing key | override |
|---|---|---|---|
| `build/data/**` | verbatim `copytree` of the profile's resolved data root (`scaffolding.copy_template_data` at `scaffolding.py:314`, called `manager.py:272`) | **`data:`** — required of every profile, no packaged fallback. `data/.runtime/` is excluded at the top level | edit the profile's `data/` tree |
| `build/data/channel_databases/<paradigm>.json` | flattened from `data/channel_databases/tiers/tier{1,3}/` by `materialize_tier_artifacts` (`manager.py:325`) | `channel_finder_mode:` plus the resolved tier (`BuildProfile.resolved_tier()`; `tier_mode_conflict` at `build_cmd.py:2003-2008`) | `tier:` |
| `build/data/simulation/channel_manifest.json`, `channel_limits.json` | `write_project_manifest` (`build_cmd.py:2261`) | derived from the `data:` tree and tier; deferred until after the render for `channel_finder_mode: graph` (`build_cmd.py:1980`, `:2242`) | **Reserved** against the `project/` mirror (`profile_conventions.py:340-341`) |
| `build/data/channel_databases/graph.duckdb` | `_build_graph_index` (`build_cmd.py:1739`, called `:2237`) | `channel_finder_mode: graph`; corpus from `services.graphdb.ttl_path`, output from `services.graphdb.index_path` | failure is a warning, not a build failure (`build_cmd.py:1805-1807`) |
| `build/machine_data/` | `manager.py:283-287` | only when the data bundle ships one | — |
| `build/docker/web-terminal-context/base.md` | `templates/claude_code/web-terminal-context/base.md` — **static**, copied unconditionally (`manager.py:289-298`) | **IMPLICIT framework fallback** | the profile's own `web-terminal-context/base.md` overrides it, copied by `_apply_conventions` afterwards. **Reserved** against the `project/` mirror (`profile_conventions.py:342-346`) |
| `build/docker/web-terminal-context/<user>/` | `.gitkeep` seeded per roster user (`build_persistence.py:288-293`) | `config.modules.web_terminals` roster, read back from the **rendered** config (`_resolve_context_roster`, `build_persistence.py:85`) | a persona that switches the module off gets no per-user context at all. Missing roster users are seeded *in the profile*; departed users are warned about and skipped, never deleted |

---

## 7. Implicit-only items, ranked by how much they shape agent behaviour

Prompt and instruction files first, plumbing last.

1. **`.claude/settings.json`** — the permission floor, the deny list, the ask list, and every hook event wiring. Nothing in `profile.yml` names the file. `config.claude_code.permissions` and `config.claude_code.hooks` can only *add* to what the template emits unconditionally. The hardcoded `skillOverrides` block and the literal `outputStyle` path are invisible from the profile entirely.
2. **`CLAUDE.md`** — the whole system prompt. The profile picks one of four shipped templates via `claude_md_template:` and nothing more. Every sentence of behavioural instruction — verify-first, the per-agent delegation mandates naming specific forbidden tools, the "never fabricated" rule — is hardcoded in the `.j2` and unreachable from `profile.yml`.
3. **`.claude/agents/_shared/handback.md.j2`** — the "Handing Back" contract (`submit_response` files the artifact, the reply to the parent is a pointer, never paste the body), `{% include %}`d into every submitting subagent's prompt. Not in the artifact catalog, skipped by the `_`-prefix rule so it never appears on disk, and not nameable, excludable or claimable.
4. **`.claude/agents/_shared/graph_snapshot.md.j2`** — the "Graph at Hand" block included by `channel-finder` and `facility-knowledge-graph`. Its managed marker region is filled **after** the build, by `bake_snapshot` (`src/osprey/services/facility_knowledge/seeder/prompt_snapshot.py:599`), called from `osprey up` (`container_lifecycle.py:5313`) and `osprey knowledge` (`knowledge_cmd.py:466`).
5. **`.claude/agents/_terminology/{graph,hierarchical,in_context,middle_layer}.md.j2`** — the entire channel-database vocabulary the channel-finder subagent reasons in, one partial per paradigm. Selected indirectly by `channel_finder_mode:`, never named. Rows come from `facility_vocabulary`, derived from `facility.ontology`.
6. **`docker/web-terminal-context/base.md`** — the seed context every web-terminal session starts from (approval-required writes, no improvising missing capabilities, disclose a mock backend). Shipped as a framework fallback; overridable only through the `web-terminal-context/` convention dir.
7. **`.mcp.json`** — which tools exist at all. Server membership comes from `FRAMEWORK_SERVERS` defaults and `condition=` expressions in `registry/mcp.py`; `mcp_servers:` and `claude_code.servers.*` can only flip or clone what the registry already defines.
8. **`.claude/hooks/hook_config.json`** — the runtime contract the write kill switch and the MCP audit middleware read as data. Nameable as `hook-config`, but its *contents* are derived wholly from the server registry and `config.control_system.write_tools`.
9. **`.claude/statusline.py`** — cosmetic. Static copy, no override but `scaffold claim`.
10. **`config.yml` derived keys**, `.env.example`, `Dockerfile`, `entrypoint.sh`, `README.md`, `.gitignore`, `.dockerignore` — plumbing.
11. **`env-chain.json`, `.osprey-manifest.json`, `pyproject.toml`, `.venv/`, `.image/**`, `services/docker-compose.yml.j2`, `services/_env_axis.j2`, `services/_network_axis.j2`** — build bookkeeping.

---

## 8. Dead names and non-listable templates

### Profile names mapping to nothing

**None are possible** in `hooks:` / `rules:` / `skills:` / `agents:` / `output_styles:`. `validate_artifacts` (`artifact_library.py:207`) raises `ValueError` with a did-you-mean suggestion before the render begins, called from `_collect_profile_artifacts` (`build_cmd.py:3562`).

I cross-checked all 66 catalog entries against the template tree: **zero** catalog `template_path` entries lack a file on disk.

The one silent case is the conditionally-empty render described in §2b (`test-ioc-safety`, `data-visualization`) — a valid name that yields no file, with no message.

`web_panels:` is the looser key: it is validated warn-only at manifest load time and deliberately bypasses the library check because it is not file-backed (`build_cmd.py:3546-3549`). A `web_panels:` entry the host projects no address for is caught separately by `selected_panel_errors` (`build_cmd.py:2145-2149`).

### Template files not listable in the profile — the complete set

Cross-checked programmatically against `BuildArtifactCatalog.default().all_artifacts()`:

```
src/osprey/templates/claude_code/claude/agents/_shared/graph_snapshot.md.j2
src/osprey/templates/claude_code/claude/agents/_shared/handback.md.j2
src/osprey/templates/claude_code/claude/agents/_terminology/graph.md.j2
src/osprey/templates/claude_code/claude/agents/_terminology/hierarchical.md.j2
src/osprey/templates/claude_code/claude/agents/_terminology/in_context.md.j2
src/osprey/templates/claude_code/claude/agents/_terminology/middle_layer.md.j2
src/osprey/templates/claude_code/web-terminal-context/base.md
```

Plus, outside `claude_code/`, everything under `templates/project/` (7 files), `templates/deploy/` (4), `templates/modules/web_terminals/` (6), and the shared service partials `templates/services/_env_axis.j2` and `_network_axis.j2`.

Plus the four catalog entries whose canonical names exist but have **no** `_MANIFEST_CATEGORY_PREFIX` category, so no profile list can reach them:

```
claude-md   (and claude-md-ariel / -knowledge / -channel-finder)
mcp-json
settings-json
statusline
```

These four are exactly the always-included set in `resolve_manifest_outputs` (`manifest.py:205`).

### Catalog entries not reachable through the five list keys, but reachable elsewhere

The 14 `services/*` catalog entries (`services/bluesky`, `services/graphdb`, …) have canonical names and are claimable via `osprey scaffold claim services/<name>`, but they are not selectable through any of the five artifact-list keys — the injectors and `services:` blocks decide which are deployed.

---

## 9. Override mechanisms that exist today

| mechanism | what it reaches | notes |
|---|---|---|
| profile convention directories — `rules/ skills/ agents/ commands/ output-styles/ hooks/ services/ mcp_servers/ web-terminal-context/` | the matching `.claude/` subtree, `services/`, `_mcp_servers/`, `docker/web-terminal-context/` | table at `profile_conventions.py:105-177`. Applied **after** the base render (`build_cmd.py:2204`). Directory-shaped copies do `rmtree` + `copytree`, so nothing from a prior build survives inside one. Copies dereference symlinks and preserve mode and mtime |
| `project/` verbatim mirror | any project-root path **not** reserved | shape `MIRROR`, destination `""` (`profile_conventions.py:172`). Refused for the 12 exact reservations (`RESERVED_PROJECT_PATHS`, `profile_conventions.py:319`), for every `.claude/` subtree that has a convention channel, and for every framework-rendered output (`reserved_path_channel`, `profile_conventions.py:600`). The error message names the correct channel and, where derivable, the exact move |
| `osprey scaffold claim <canonical-name>` | anything in the catalog, including `settings-json`, `claude-md`, `statusline`, `services/<name>` | writes `config.scaffold.user_owned` plus a `user_owned` entry in the manifest; regen then skips the file (`is_user_owned`, `claude_code.py:1035`). `osprey scaffold release` reverses it (`scaffold_cmd.py:1575`) |
| persona delta `exclude:` | subtracts convention artifacts per persona | `resolved.excluded_artifacts` passed into `_apply_conventions` (`build_cmd.py:2212`) |
| the profile's `config:` block | every non-derived `config.yml` key | derived keys the profile also spells are **refused** at validation, not silently overwritten. `layout` port fill is the one fill-if-absent exception (`build_cmd.py:2062-2072`) |
| `.env.variant` + `profiles/<name>.yml` | a whole host-specific profile layer | merged over `profile.yml` by the same deep merge a persona delta uses, ahead of `extends:` resolution, for the deployment and every persona alike (`build_cmd.py:3155-3162`) |
| `osprey eject service <name>` | copies framework service source into the repo | `src/osprey/cli/eject_cmd.py:106`. Source ejection, not a render override |
| create-only semantics | `.claude/rules/facility.md` alone | rendered once, auto-claimed as user-owned, never rewritten |
| `--skip-deps`, `--dev`, `--runtime-root` | suppress the venv / add local wheels / skip compose, `env-chain.json` and all image contexts | CLI flags, not profile keys |

**No override channel exists** for the `_shared`/`_terminology` agent partials, the hardcoded prose in the four `CLAUDE.md*.j2` templates, or the hardcoded hook event wiring in `settings.json.j2`. Changing any of those requires an upstream framework change.
