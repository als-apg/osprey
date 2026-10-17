# OSPREY deployment profile: artifact list semantics

Repo: `/Users/thellert/code/osprey`. All paths below are relative to that root
unless absolute. Read-only research; nothing was modified.

---

## 1. How the five lists are consumed

**They are pure filters over the framework catalog. A deployment-local file can
never legally appear in them.**

### Schema location

The five lists are plain `list[str]` fields on the `BuildProfile` dataclass.
They are **not** in `build_profile_schema.py` — that module holds only the
nested block shapes (`mcp_servers`, `lifecycle`, `env`, `services`, `dispatch`,
`bluesky`, `virtual_accelerator`, …), per its own docstring at
`src/osprey/cli/build_profile_schema.py:1-12`.

- `src/osprey/cli/build_profile_model.py:317-321` — `hooks`, `rules`, `skills`,
  `agents`, `output_styles` (and `web_panels` at `:322`), each
  `field(default_factory=list)`.
- `src/osprey/cli/build_profile_load.py:1160-1165` — parsed straight from the
  raw YAML with `raw.get("<key>", [])`. No per-entry validation happens here.
- `src/osprey/cli/build_profile_load.py:152-200` — `_KNOWN_PROFILE_KEYS`, the
  closed top-level key set. An unrecognized top-level key is refused with a
  "did you mean" suggestion and the full valid set
  (`build_profile_load.py:432-436`).
- `src/osprey/cli/build_profile_document.py:37-46` — `_EMPTY_COLLECTION_KEYS`.
  A present-but-empty `skills:` normalizes to `[]` **before** any `extends` or
  persona merge, so a persona delta cannot subtract an inherited list by
  blanking it. The docstring at `:49-63` is explicit that this is what keeps
  `exclude:` the only subtraction mechanism.

### The catalog

`src/osprey/cli/templates/artifact_library.py:21-27` maps each profile key to a
subdirectory under `src/osprey/templates/claude_code/claude/`:

| profile key | directory | one entry is |
|---|---|---|
| `hooks` | `hooks/` | a `.py` file, or `hook_config.json.j2` |
| `rules` | `rules/` | a `.md` or `.md.j2` file |
| `skills` | `skills/` | a subdirectory |
| `agents` | `agents/` | a `.md` / `.md.j2` file |
| `output_styles` | `output-styles/` | a `.md` / `.md.j2` file |

Short names are **derived, not declared**:

- Hooks strip the `osprey_` prefix and hyphenate underscores —
  `osprey_writes_check.py` → `writes-check` (`artifact_library.py:72-87`);
  `hook_config.json.j2` → `hook-config` (`:172-175`).
- Everything else drops the `.md` / `.md.j2` suffix (`:176-182`).
- Files and directories whose name starts with `_`, and `__pycache__`, are
  excluded from enumeration (`:155-156`). That is why `_shared/` and
  `_terminology/` partials are not selectable.

### Unknown name → hard error, not a warning or a silent skip

`validate_artifacts` accumulates every offender and raises `ValueError`
(`artifact_library.py:207-235`):

```
Invalid artifact(s) in profile:
  rules/orbit: not found. did you mean 'workflows'? Available rules: artifacts, ...
```

The "did you mean" is Levenshtein-based with a `max(2, len(name) // 2)`
threshold (`:275-300`).

Call chain:

- `_collect_profile_artifacts` — `src/osprey/cli/build_cmd.py:3542-3568`
  (gathers the five lists off the profile, then calls `validate_artifacts` at
  `:3560-3562`).
- Invoked from `_render_project` at `build_cmd.py:1917`, i.e. before anything
  is written.
- The `ValueError` is caught by the build's `except ValueError` arm at
  `build_cmd.py:3475-3477` → `logger.error("✗ Error: %s", e)` → `click.Abort`.

### `web_panels` is the exception

It bypasses the library check entirely and is added by the caller after
validation (`build_cmd.py:3544-3547` docstring: "`web_panels` is validated at
manifest load time (warn-only) and is not file-backed, so it bypasses the
library check"; the add is at `:1918-1919`).

### What selection actually does downstream

Selected names become the `artifacts` block of `.osprey-manifest.json`, which
resolves to a set of allowed output paths
(`src/osprey/cli/templates/manifest.py:189-223`, prefix-matched against
`BuildArtifactCatalog` so multi-file artifacts like `skills/session-report`
pull in their `reference.md`). The render loop then copies/renders
`claude/**` → `.claude/**` and, for any path **not** in that set, deletes a
stale copy rather than orphaning it:

- `.j2` branch — `src/osprey/cli/templates/claude_code.py:1876-1885`
- plain-file branch — `claude_code.py:1907-1912`

So omitting a name from a list removes the file from the build. Notably
`hooks: [hook-config]` is what renders `.claude/hooks/hook_config.json` at all
— it is a catalog entry like any other
(`src/osprey/services/build_artifacts/catalog.py:246-248`).

---

## 2. Deployment-local override / extension mechanisms

**Yes — "convention directories" beside `profile.yml`. This is the documented
and only channel for deployment-authored Claude Code artifacts.**

### The convention table

`src/osprey/cli/profile_conventions.py:104-172` (`CONVENTION_DIRS`):

| profile dir | build destination | shape (`EntryShape`) |
|---|---|---|
| `rules/` | `.claude/rules/` | MARKDOWN (file per rule) |
| `skills/` | `.claude/skills/` | DIRECTORY (one per skill) |
| `agents/` | `.claude/agents/` | MARKDOWN |
| `commands/` | `.claude/commands/` | MARKDOWN |
| `output-styles/` | `.claude/output-styles/` | MARKDOWN |
| `hooks/` | `.claude/hooks/` | FILE (any extension) |
| `web-terminal-context/` | `docker/web-terminal-context/` | DIRECTORY, per-user |
| `mcp_servers/` | `_mcp_servers/` | DIRECTORY |
| `services/` | `services/` | DIRECTORY |
| `project/` | build root (`""`) | MIRROR (verbatim escape hatch) |

Shapes are defined at `profile_conventions.py:57-78`. **No declaration is
needed** — the directory name is the declaration.

The two namespaces are disjoint by construction.
`src/osprey/cli/build_profile_merge.py:117-131` states it outright:
`validate_artifacts` hard-rejects any name the built-in library does not carry,
"so a profile's own `agents/orbit-writer.md` can never legally appear in
`agents:`".

### Shadowing

Convention copies run **after** the framework render and overwrite what is
there — `src/osprey/cli/build_persistence.py:298-336`. The shadow branch and
its debug line are at `:317-330`:

```
"  ↷ profile overrides framework %s '%s'"
```

So a profile that ships `rules/safety.md` wins over the framework's rendered
`safety.md`. Copied entries are registered into `config.yml`'s
`scaffold.user_owned` (`build_persistence.py:331-333`), and every later render
skips a user-owned path:

- `is_user_owned` — `claude_code.py:1035-1071`
- applied at `claude_code.py:1807` (`.mcp.json`), `:1818` (`CLAUDE.md`),
  `:1868-1871` (`.j2` artifacts), `:1900-1903` (plain files)

Excluded copies are skipped entirely and never enter `owned_artifacts`, so what
they would have shadowed keeps the framework render
(`build_persistence.py:299-305`).

The build refuses to read convention material from inside the installed osprey
package (`build_persistence.py:269-277`) and refuses a destination that escapes
the project root via a symlink (`:315-323`).

### Taking over a framework artifact: `osprey scaffold claim`

`src/osprey/cli/scaffold_cmd.py:621-703` (`claim_into_profile`), CLI at
`:1393-1420`. It **moves** `build/.claude/rules/safety.md` to
`<profile>/rules/safety.md`; after a claim the artifact exists in one place
only until the next build renders it again (`:1403-1405`).

Refusals, all raised before anything is touched (`:632-639`):

- no profile reachable from the project (`_no_profile_message`, `:443`)
- slot already occupied, symlinks included (`:657-661`)
- source missing (`:663-664`)
- wrong shape — a file where a directory belongs or vice versa (`:666-675`)
- symlinks escaping the profile (`_escaping_symlinks`, `:552-582`)
- **generated** paths — `hook_config.json`, `settings.json`, `.mcp.json`,
  `CLAUDE.md` — via `_refuse_generated` / `_generated_path_message`
  (`:225-262`)

Companions: `scaffold list` (`:1347`), `diff` (`:1444`), `unclaim` (`:1572`).
Registration is deliberately **not** written by claim: the next build's
convention scan derives `scaffold.user_owned` from what it copied
(`:626-630`).

### Subtraction: `exclude:`

`build_profile_merge.py:242-360` (`_apply_exclude`). The **spelling** of each
entry decides what it removes:

- **bare** (`channel-finder`) — subtracts from the string list, so a built-in
  an `extends` base selected is no longer selected. Excludable fields:
  `_EXCLUDABLE_FIELDS` at `:81-91` (`skills`, `rules`, `hooks`, `agents`,
  `output_styles`, `web_panels`, `dependencies`).
- **qualified** (`agents/channel-finder`) — records a file omission, so the
  profile's own shadowing file is not copied and the still-selected built-in
  renders again (`:262-270`).
- `exclude.config` — a mapping of dotted list-valued config key → items to
  remove (`_apply_config_exclude`, `:205-241`). This is how a host-variant
  overlay deploys *fewer* services than the profile it sits on, since the
  inheritance merge unions lists.

Malformed entries are refused with a reason (`_split_exclude_entry`,
`:159-205`): a non-string entry, a prefix naming nothing, or a qualified entry
under the wrong key. Excluding something absent is a silent no-op.

### Custom hooks need a second step

Shipping a script into `hooks/` copies it into `.claude/hooks/`; it does not
run it. Wiring is a **config key**, `claude_code.hooks.<Event>`, resolved by
`_build_declared_hook_rules` (`claude_code.py:1237-1260+`):

- accepted events — `CLAUDE_CODE_HOOK_EVENTS`, `claude_code.py:1148-1160`
- declared wiring is strictly **additive**; nothing there can remove or alter a
  framework entry (`:1244-1252`)
- declaring a **built-in** hook filename is refused —
  `_framework_hook_filenames`, `:1189-1194`
- absolute paths, `..` traversals, backslashes and other channels' prefixes are
  refused rather than normalized — `_declared_hook_filename`, `:1196-1235`
- default timeout for a declaration is 60 s, Claude Code's own
  (`_DECLARED_HOOK_TIMEOUT`, `:1166-1170`)

### Mechanisms that are NOT an override channel

- **`osprey eject`** — `src/osprey/cli/eject_cmd.py`, read in full (182 lines).
  It copies a framework **Python service package** (`osprey.services.<name>`)
  into `./src/<package>/services/<name>/` (`:97-182`; default output at
  `:103`, `:153`). `eject list` enumerates registry services (`:82-94`).
  It has nothing to do with `.claude/` artifacts, profiles, or convention
  directories, and it assumes a `src/<package>/` Python layout that a
  deployment repo does not have (`_get_project_package_dir`, `:25-40`). Its
  "Next steps" tell the user to "Update imports in your capabilities to use the
  local service" (`:179-182`).
- **Personas** — `personas/*.yml` beside `profile.yml`
  (`src/osprey/cli/scaffold_personas.py:1-32`). Pure profile **deltas** merged
  over `profile.yml`; they can change list selections, `config:` and use
  `exclude:`, but carry no second set of convention directories. The example
  deployment holds five: `admin.yml`, `knowledge.yml`, `logbook.yml`,
  `readonly.yml`, `readwrite.yml`.
- **Host variants** — `profiles/<host>.yml` selected by `.env.variant`; a
  profile-YAML overlay folded into the render hash
  (`build_profile_merge.py:1027-1045`). It adds and replaces; it does not
  subtract except through `exclude:`.
- **`_shared/` and `_terminology/` partials** — include-only fragments, skipped
  by the render loop (`claude_code.py:1856-1858`, "Skip files in `_`-prefixed
  directories") and unlisted by the catalog (`artifact_library.py:155`). There
  is no channel to override one; only the agent template that includes it can
  be shadowed.
- **`web-terminal-context/`** — a convention directory, but per-user: entries
  are resolved against the build's web-terminal roster, missing users are
  seeded with an empty dir plus `.gitkeep`, departed users are skipped with a
  warning (`build_persistence.py:282-296`, `:336-347`).
- **No `extra_*` / `custom_*` / `overrides` / `local_rules` / `user_rules`
  key exists.** The top-level key set is closed (`_KNOWN_PROFILE_KEYS`,
  `build_profile_load.py:152-200`), and the only non-list escape hatch is
  `project/`, itself blocked from writing build-owned paths
  (`profile_conventions.py` `reserved_path_channel`, documented at
  `docs/source/how-to/build-profiles.rst:193-215`).

---

## 3. Which templates are Jinja, and what they consume

Root: `src/osprey/templates/claude_code/`. No `claude/commands/` directory
exists in the framework tree.

### Jinja templates and their top-level context names

| Template | Kind | Context names used |
|---|---|---|
| `CLAUDE.md.j2` | j2 | `agents`, `agent_data_root` |
| `CLAUDE.ariel.md.j2` | j2 | `agents`, `facility_name` |
| `CLAUDE.channel-finder.md.j2` | j2 | `agents`, `facility_name`, `agent_data_root` |
| `CLAUDE.knowledge.md.j2` | j2 | `agents`, `facility_name` |
| `mcp.json.j2` | j2 | `servers` (uses `.name/.command/.args/.url/.transport/.env`) |
| `claude/settings.json.j2` | j2 | `agents`, `servers`, `facility_permissions`, `deny_defaults`, `killswitch_deny`, `declared_hooks`, `declared_extra_events`, `framework_pre_hooks`, `framework_post_hooks`, `current_python_env`, `terminal_theme`, `textbooks_root`, `textbooks_root_tilde`, `agent_data_root` |
| `claude/hooks/hook_config.json.j2` | j2 | `servers`, `control_system_write_tools`, `lane_addressed_tools`, `mixed_read_write_tools` |
| `claude/agents/channel-finder.md.j2` | j2 | `enabled_agents`, `channel_finder_mode`, `channel_finder_hierarchy`, `channel_finder_tools`, `claude_code_model_spec`, `facility_name`; sets `handback_label`; includes 6 partials |
| `claude/agents/data-visualizer.md.j2` | j2 | `enabled_agents`, `claude_code_model_spec`, `facility_name` |
| `claude/agents/facility-knowledge.md.j2` | j2 | `enabled_agents`, `claude_code_model_spec`, `facility_name`; `_shared/handback` |
| `claude/agents/facility-knowledge-graph.md.j2` | j2 | `enabled_agents`, `claude_code_model_spec`; sets `graph_terminology_structural`; `_shared/graph_snapshot`, `_shared/handback`, `_terminology/graph` |
| `claude/agents/logbook-search.md.j2` | j2 | `enabled_agents`, `claude_code_model_spec`, `facility_name`; `_shared/handback` |
| `claude/agents/logbook-deep-research.md.j2` | j2 | same as above |
| `claude/agents/pyat-specialist.md.j2` | j2 | same as above |
| `claude/agents/_shared/handback.md.j2` | j2 partial | `handback_label` (defaults to `"Results"`) |
| `claude/agents/_shared/graph_snapshot.md.j2` | j2 partial | none — no Jinja expressions |
| `claude/agents/_terminology/graph.md.j2` | j2 partial | `graph_terminology_structural` |
| `claude/agents/_terminology/hierarchical.md.j2` | j2 partial | `facility_vocabulary` |
| `claude/agents/_terminology/in_context.md.j2` | j2 partial | `facility_vocabulary` |
| `claude/agents/_terminology/middle_layer.md.j2` | j2 partial | `facility_vocabulary` |
| `claude/output-styles/control-operator.md.j2` | j2 | `enabled_agents` |
| `claude/rules/control-system-safety.md.j2` | j2 | `control_system_type`, `enabled_servers`, `target_switch_enabled` |
| `claude/rules/data-visualization.md.j2` | j2 | `enabled_agents` |
| `claude/rules/facility.md.j2` | j2 | `enabled_agents` |
| `claude/rules/safety.md.j2` | j2 | `enabled_agents` |
| `claude/rules/test-ioc-safety.md.j2` | j2 | `control_system_type` |
| `claude/rules/timezone.md.j2` | j2 | `system_timezone` |
| `claude/skills/bluesky-plans/SKILL.md.j2` | j2 | `enabled_servers`, `bluesky_plan_index` (`.rows/.overflow/.unreadable_dirs`) |
| `claude/skills/logbook-deep-research/SKILL.md.j2` | j2 | `facility_name` |
| `claude/skills/setup-mode/SKILL.md.j2` | j2 | `project_name`, `facility_name`, `agent_data_root`, `preset`, `enabled_servers`, `channel_finder_pipeline` |

### Partials and their consumers

Pulled in by absolute loader path
(`{% include "claude_code/claude/agents/..." %}`), never by macro import:

- `_shared/handback.md.j2` — included by `channel-finder`,
  `facility-knowledge`, `facility-knowledge-graph`, `logbook-search`,
  `logbook-deep-research`, `pyat-specialist`. Callers set `handback_label`
  first.
- `_shared/graph_snapshot.md.j2` — included by `channel-finder.md.j2` (graph
  branch) and `facility-knowledge-graph.md.j2`.
- `_terminology/{graph,hierarchical,in_context,middle_layer}.md.j2` — included
  by `channel-finder.md.j2`, one per `channel_finder_mode` branch (an unknown
  mode calls `fail()`); `_terminology/graph.md.j2` is also included by
  `facility-knowledge-graph.md.j2` with `graph_terminology_structural = true`.

### Copied verbatim (no rendering context needed)

- Hook scripts (all 15): `claude/hooks/osprey_approval.py`,
  `osprey_cf_feedback_capture.py`, `osprey_config_drift.py`,
  `osprey_control_context.py`, `osprey_error_guidance.py`,
  `osprey_focus_validate.py`, `osprey_hook_log.py`, `osprey_limits.py`,
  `osprey_memory_guard.py`, `osprey_notebook_update.py`,
  `osprey_panels_context.py`, `osprey_target_state.py`,
  `osprey_turn_state.py`, `osprey_workspace_delta.py`,
  `osprey_writes_check.py`
- `claude/statusline.py`
- Rules: `artifacts.md`, `error-handling.md`, `python-execution.md`,
  `workflows.md`
- Skills: `demo-gallery/SKILL.md`, `demo-ui/SKILL.md`, `diagnose/SKILL.md`,
  `operating-bluesky-plans/SKILL.md`, `session-report/SKILL.md`,
  `session-report/reference.md`, `sim-scenarios/SKILL.md`,
  `writing-bluesky-plans/SKILL.md`
- `web-terminal-context/base.md` — copied as a directory tree, not rendered
  (`src/osprey/cli/templates/manager.py:295-298`)

### Where the render context is assembled

Jinja env: `src/osprey/cli/templates/manager.py:80-81`
(`FileSystemLoader(self.template_root)`, which is why includes are spelled
`claude_code/claude/agents/...`). Render call:
`src/osprey/cli/templates/_rendering.py:25`.

Two paths build the context and share `config_derived_context`:

**A. `build_claude_code_context` — `claude_code.py:550`**

- `:619-648` base dict: `project_name`, `package_name`, `project_root`,
  `current_python_env`, `preset`, `claude_md_template`, `facility_name`,
  `system_timezone`, `selected_hooks`
- `:659` `ctx.update(config_derived_context(...))`
- `:684-693` `channel_finder_pipeline`, `channel_finder_mode`,
  `default_pipeline`, `channel_finder_tools`
- `:700`, `:702` `channel_finder_hierarchy`
- `:706` `facility_permissions`
- `:710-714` `servers`, `agents`, `enabled_servers`, `enabled_agents`
- `:725` `mixed_read_write_tools`; `:734` `bluesky_plan_index`
- `:763` `apply_textbooks_root` → `textbooks_root`, `textbooks_root_tilde`
- `:785` `claude_code_model_spec`
- `:944-945` writes-off kill switch overwrites `facility_permissions`, sets
  `killswitch_deny`
- `:1793-1794` (inside `create_claude_code_integration`, `:1749`)
  `framework_pre_hooks`, `framework_post_hooks`

**B. `config_derived_context` — `claude_code.py:360`, dict at `:383-460`**

Keys: `user_owned`, `declared_hooks`, `declared_extra_events`,
`agent_data_root`, `control_system_write_tools`, `lane_addressed_tools`,
`control_system_type`, `target_switch_enabled`, `graphdb_configured`,
`phoebus_bridge_default`, `facility_vocabulary`, `deny_defaults`,
`killswitch_deny`, `mixed_read_write_tools`, `terminal_theme`.

**C. `TemplateManager._project_context` / `create_project` — `manager.py`**

- `:550-603` base dict (`project_name`, `package_name`, `app_display_name`,
  `app_class_name`, `registry_class_name`, `project_description`,
  `framework_version`, `project_root`, `venv_path`, `current_python_env`,
  `preset`, `selected_hooks`, `selected_web_panels`, `builtin_panels`,
  `provider_api_keys`, `active_provider_vars`, `service_token_vars`,
  `env_required`, `env_defaults`, `port_base`, then `**context`)
- `:342` `facility_permissions`; `:356` `claude_code_model_spec`;
  `:360` `system_timezone`; `:365`/`:379` `facility_name`;
  `:374` `channel_finder_hierarchy`
- `:388` merges `config_derived_context`; `:404` `deny_defaults`;
  `:406` `apply_textbooks_root`
- `:411-414` `servers`, `agents`, `enabled_servers`, `enabled_agents`;
  `:423` `mixed_read_write_tools`; `:609` `osprey_ports`

The Jinja environment is **not** strict (`claude_code.py:360-377`), so a
context key omitted by one of the two paths renders as empty rather than
erroring.

---

## 4. How hooks get wired to Claude Code events

**There are three separate wiring mechanisms, and only one of them reads the
profile's `hooks:` list.**

### (a) Hardcoded by filename in `settings.json.j2`

`src/osprey/templates/claude_code/claude/settings.json.j2`:

- `SessionStart` (`:144-162`) — `osprey_config_drift.py` (t/o 5),
  `osprey_panels_context.py` (5), `osprey_control_context.py` (3); plus
  `osprey_turn_state.py` under a `"startup|resume|clear"` matcher (`:171-180`)
- `UserPromptSubmit` (`:182-207`) — `osprey_focus_validate.py` (2),
  `osprey_workspace_delta.py` (3), `osprey_control_context.py` (3),
  `osprey_turn_state.py` (3)
- `Stop` (`:272-282`) and `StopFailure` (`:283-293`) — `osprey_turn_state.py`

These entries are emitted **unconditionally** — they are not gated on the
profile's `hooks:` list. Dropping such a name from the list removes the file
from `.claude/hooks/` but leaves the wiring; every command is suffixed
`2>/dev/null || true`, so the failure is silent.

### (b) Per-MCP-server, from the server registry

`src/osprey/registry/mcp.py:93-111` defines the `HookEntry` constants
`_APPROVAL` (`osprey_approval.py`, t/o 5), `_WRITES_CHECK`
(`osprey_writes_check.py`, 5), `_LIMITS` (`osprey_limits.py`, 10),
`_ERROR_GUIDANCE` (`osprey_error_guidance.py`, 5) and `_CF_FEEDBACK`
(`osprey_cf_feedback_capture.py`, 10). They are attached to each framework
server's `hooks_pre` / `hooks_post` with a tool matcher, and the template loops
over enabled servers (`settings.json.j2:213-217` for pre, `:241-245` for post).
Gating is by **server enablement**, not by the `hooks:` list.

### (c) Frontmatter-driven, from the profile's `hooks:` list

`_build_framework_hook_rules` (`claude_code.py:1094-1146`) resolves each
selected hook name, parses its docstring frontmatter, and emits a rule sorted
ascending by `safety_layer` (lower = outermost gate, `:1143-1146`). The
`python3` token is rewritten to the project's resolved interpreter by the
template (`settings.json.j2:231`, `:259`).

`parse_hook_frontmatter` returns `None` unless the block carries **both**
`event:` and `wiring: standalone` (`artifact_library.py:136-142`); a
`wiring: standalone` block missing `tools:` logs a warning and is skipped
(`:140-142`).

Surveying every hook's frontmatter, only **two** declare `wiring: standalone`:

- `osprey_memory_guard.py` — PreToolUse, `Write|MultiEdit|NotebookEdit`,
  safety_layer 0
- `osprey_notebook_update.py` — PostToolUse, `NotebookEdit`, safety_layer 99

Every other hook's `event:` line is documentation only; those hooks reach
`settings.json` through mechanism (a) or (b).

### Is `target-state` really "not wired to an event"? Yes

`osprey_target_state.py` has **no** frontmatter block. Its docstring opens:
"Not a hook: this is the frontmatter-less library that the hooks import to
answer two questions — *which control-system target is this deployment pointed
at?* and *may a write reach it?*". `osprey_hook_log.py` is likewise a shared
library ("Shared hook logging utility. No external dependencies.").

`artifact_library.py:31-36` states the rule explicitly: selection is what
copies a file into `.claude/hooks/` while docstring frontmatter is what wires
it to an event — "A helper is selected everywhere and wired nowhere." The
profile comment is accurate.

### Custom (deployment-shipped) hooks

Wired only via `config: claude_code.hooks.<Event>`, and appended **after** the
framework's entries in every block: `settings.json.j2:218-223` (PreToolUse),
`:249-251` (PostToolUse), and the inline `declared.get(...)` loops at `:180`
(SessionStart), `:206` (UserPromptSubmit), `:281` (Stop), `:292`
(StopFailure). Events the framework wires nothing on become new keys via
`declared_extra_events` (`:301-305`), with `Stop`/`StopFailure` filtered out
there so a duplicate JSON key cannot silently shadow the framework's copy
(`:294-300`).

`FRAMEWORK_WIRED_EVENTS` (`claude_code.py:1162-1166`) names the six the
framework already wires; `CLAUDE_CODE_HOOK_EVENTS` (`:1148-1160`) is the full
accepted set, adding `SessionEnd`, `SubagentStop`, `Notification`,
`PreCompact`.

### Do `settings.json` permissions depend on profile keys? Yes, heavily

`settings.json.j2:49-142`:

- **`allow`** — `Read(<agent_data_root>/**)` and three subtrees keyed off the
  project's own `agent_data.base_dir` (`:57-60`); optional `textbooks_root`
  and its tilde form (`:61-66`); per-enabled-server `permissions_allow` and
  `fixed_allow` (`:68-75`); `NotebookEdit(<agent_data_root>/artifacts|notebooks/**)`
  (`:76-77`); one `Task(<name>)` per enabled agent (`:79-81`); then
  `facility_permissions.allow` (`:82-84`).
- **`deny`** — three parts in order (`:99-119`): `deny_defaults` minus
  `remove_deny`; profile-authored `deny` minus `remove_deny`; then
  `killswitch_deny` appended **last and never filtered**, so a writes-off deny
  cannot be lifted by a profile's `remove_deny`. `deny_defaults` is the
  `DENY_DEFAULTS` constant in `claude_code.py`, deliberately not restated in
  the template (`:87-98`).
- **`ask`** — per-server `permissions_ask` / `fixed_ask` plus
  `facility_permissions.ask`, filtered by `remove_ask` (`:123-141`).
- Also `theme` from the deployment's `web.theme` when it pins a concrete id
  (`:24-29`), `statusLine` from `current_python_env` (`:31-34`),
  `enableAllProjectMcpServers: true` (`:36`), and a fixed `skillOverrides`
  block switching off `update-config`, `keybindings-help` and
  `fewer-permission-prompts` (`:44-48`).

`facility_permissions` comes from `claude_code.permissions` in the resolved
config; the writes-off kill switch computes `killswitch_deny` / `remove_ask` at
`claude_code.py:860-945`.

---

## 5. What the docs say about customizing rules / skills / agents / hooks

### Canonical page: `docs/source/how-to/build-profiles.rst`

- `:117-160` — the convention-directory table (same ten rows as section 2).
  "Put a file in the directory that matches what it is, and the build carries
  it into the project. There is nothing to declare in ``profile.yml``: the
  directory name *is* the declaration, and where each one lands is fixed."
- `:113-115` — "Directories for your own artifacts (``rules/``, ``skills/``,
  and the rest) are **not** created up front. Create the ones you need."
- `:167-170` — "A file you ship this way is registered as **yours** in the
  project: later re-renders never overwrite it, and cleanup never removes it.
  Name a file after something the framework also renders (``rules/safety.md``)
  and yours wins."
- `:172-175` — ownership is *derived* from what the build copied, after
  exclusions; there is no list to maintain.
- `:162-166` — nested paths preserved in markdown dirs
  (`commands/orbit/correct.md`); skills, MCP servers, services and per-user
  context copy as whole directories, replaced wholesale.
- `:177-183` — "A misspelled directory is silent. ``rule/`` is not ``rules/``,
  and nothing reads it."
- `:188`, `:193-215` — `project/` is the escape hatch; the reserved-path table
  lists `config.yml`, `.claude/settings.json`, `.claude/hooks/hook_config.json`,
  `.mcp.json`, `CLAUDE.md`, `.env`/`.env.example`, `.osprey-manifest.json`,
  `data/simulation/channel_manifest.json`. "A profile that targets one of these
  is rejected at build time, with the owning channel named."
- `:387-436` — custom hooks, the one two-step case: "That copies the script to
  ``.claude/hooks/facility_guard.py``. **It does not make the script run.**
  Shipping and wiring are two steps; the second is a ``config:`` key naming the
  event it fires on". Valid events at `:413-415`; "An undeclared hook never
  runs" (`:417-422`); "Declared wiring is **added** to the framework's, never
  put in its place" (`:424`).
- `:471-478` — shipping a file named for a built-in *replaces* it (`hooks/` is
  keyed by filename; the built-in `writes-check` hook is
  `osprey_writes_check.py`); **declaring** a built-in hook is refused.
- `:325-385` (anchor `_profile-claim:`) — "To customize something OSPREY
  generates — a rule, an agent, a service template — move it into the profile":
  `osprey scaffold claim rules/safety`, `agents/channel-finder`,
  `services/postgresql`. Companions `list`, `diff`, `unclaim` at `:356-360`;
  `unclaim` "holds only until the next build" (`:362`); refusals at `:368-377`.
- `:527-573` (anchor `_profile-exclude:`) — bare vs qualified exclusion, the
  accepted fields, and the rule that excluding a *declared* hook also needs
  `claude_code.hooks.<Event>: null` in the same delta (`:571-573`, `:437-467`).
- `:504-506` — personas are deltas merged over `profile.yml`, with "no second
  data tree or set of convention directories".
- `:628-697` — host overlays in `profiles/<host>.yml` selected by
  `.env.variant`; "An overlay adds and replaces; it does not subtract."
- `:290-300` — self-containment: "Everything a profile contributes lives inside
  the profile directory… An entry that is a symlink resolving outside the
  profile is refused."

### `docs/source/reference/configuration/profile.rst`

- `:99-103` — "``hooks`` / ``rules`` / ``skills`` / ``agents`` /
  ``output_styles`` — list — ``[]`` — Built-in artifacts to install. **Your own
  files go in the matching convention directory instead.**"
- `:85-170` — the full field table (`exclude`, `mcp_servers`, `services`,
  `config`, `lifecycle`, `env`, `dependencies`, …).

### `osprey eject` in the docs

Only two places:

- `docs/source/reference/cli.rst:787-801` — "**osprey eject** — Copy framework
  services to your project for customization. ``osprey eject list`` … ``osprey
  eject service NAME [--output PATH] [--include-tests]``". One-line index entry
  at `cli.rst:43`.
- `docs/source/reference/contracts/channel-finder.rst:181-184` — a tip: "Use
  ``osprey eject service channel_finder`` to copy the channel finder service
  source into your project for custom modifications."

No page connects `eject` to the profile, convention directories, or the claim
workflow; `build-profiles.rst` never mentions it. `CHANGELOG.md:1380-1382`
names an `eject capability` subcommand that no longer exists in
`eject_cmd.py` and is absent from `cli.rst`.

### Hand-editing `build/`

Documented repeatedly as generated and overwritten:

- `build-profiles.rst:44-45` — "**Build** — the ``build/`` zone ``osprey
  build`` renders. Never edit it in place; the next build wipes and re-renders
  the whole thing."
- `:9-11` — "The profile is the source of truth. The project is a derived
  artifact — regenerable, and safe to delete and rebuild at any time."
- `:329-333` — "Everything the build renders is framework-managed… an edit made
  in place is overwritten the next time. Claiming is how an edit survives."
- `:822-825`, `:883-885` — every build wipes and re-renders `build/`,
  preserving the env chain, `var/` and `.git`.
- `cli.rst:368-372` — the render lands in `build/.tmp` and replaces `build/`
  only once it has succeeded.
- `my-control-assistant/README.md:9-19` — the deployment README's zone table
  marks `build/` "Tracked? no / Survives? no — 100% disposable".
- Exceptions: user-owned files survive (`build-profiles.rst:167-169`), and
  `facility-knowledge/facility-rules.rst:102-106` — "``facility.md`` is yours
  to edit — it is user-owned, and ``osprey build`` never overwrites it. The
  framework-generated rules *are* re-rendered by ``osprey build``; to keep an
  edit to one of those, claim it into the profile first with ``osprey scaffold
  claim rules/safety``." Same note at
  `how-to/ariel/standalone-deployment.rst:164-168`.
- `how-to/deploy-project/compose-templates.rst:79-84` — service templates under
  `<project>/services/` "are framework-managed, so an edit made to one in place
  is overwritten on the next build."
- `how-to/deploy-a-facility.rst:775-796` — the zone list plus a seealso
  pointing at build-profiles for "the convention directories, and taking
  ownership of a framework artifact."

### Canonical pages to cite

`docs/source/how-to/build-profiles.rst` (anchors `_profile-claim:` `:325`,
`_profile-exclude:` `:527`, `_profile-self-contained:` `:288`,
`_profile-reserved-paths:` `:186`, `_profile-unwire-hook:` `:437`,
`_profile-host-variants:` `:628`) ·
`docs/source/reference/configuration/profile.rst` ·
`docs/source/reference/cli.rst` (`eject` `:787-801`, `scaffold` `:1174-1271`) ·
`docs/source/how-to/facility-knowledge/facility-rules.rst:96-140` ·
`docs/source/how-to/deploy-project/compose-templates.rst:79-84` ·
`docs/source/how-to/deploy-a-facility.rst:775-796`.

**Documented gap:** no page walks through authoring a
`skills/<name>/SKILL.md`, an `agents/*.md`, a `commands/*.md` or an
`output-styles/*.md` for a deployment. The convention table plus the
`exclude:`/claim sections are the whole documented story for those four kinds;
only `hooks/` gets a dedicated section and only `rules/` a dedicated how-to
page. (`docs/source/contributing/agent-skills.rst` is about installing the
`osprey` contributor plugin, not a deployment's `.claude/skills/`.)

---

## 6. Build drift and a hand-edited `build/.claude/rules/safety.md`

**On the next `osprey build` the edit is discarded, with no user-visible
warning.**

`build/` is wiped and re-rendered whole; the render lands in `build/.tmp` and
replaces `build/` atomically:

- `src/osprey/cli/build_cmd.py:7` — module docstring: "Nothing durable lives
  there, so it is wiped and re-rendered whole every…"
- `:470-473` — "``build/`` is output: 100% derived, wiped and re-rendered by
  every build. Nothing durable may live there, and ``rm -rf build/`` losing
  nothing is the property the whole zone layout rests on."
- `_swap_in_render(zones)` at `:3443`

Before the swap, `_backup_outgoing_claude_artifacts` (`:509-520`) snapshots the
outgoing `.claude/`, `.mcp.json` and `CLAUDE.md` — **only files that actually
differ** — into the durable zone `var/agent_data/backup/` (`_BACKUP_RELDIR`,
`:223-230`; `_CLAUDE_ARTIFACT_ROOTS`, `:229-230`). Its docstring: "``build/``
is disposable and not a place to edit anything — ``osprey scaffold claim`` is
how a file becomes source — but an operator can still edit a rendered hook or
agent by mistake, and a snapshot that lets them get it back costs nothing."

The notice is **debug level only**:
`build_cmd.py:3505-3506` — `logger.debug("  ✓ Previous Claude Code artifacts
saved to %s", backup_dir)`. Nothing at info/warning level tells the operator an
edit was overwritten.

If the file has been claimed (so `rules/safety` is in `scaffold.user_owned`),
the render skips it (`claude_code.py:1868-1871`) and the profile's copy is
re-landed by the convention pass (`build_persistence.py:317-330`).

On the **regen** path (`regenerate_claude_code`, `claude_code.py:1990-2196`)
no backup is taken — the comment at `:2148-2155` explains that a snapshot
written inside the tree being regenerated would be discarded by the next build,
so the durable one in `build_cmd` is the one that counts. Regen does compute
before/after SHA-256 checksums of tracked files and returns a `changed` /
`unchanged` split (`:2076-2085`, `:2157-2172`), and `--dry-run` renders into a
temp dir and diffs (`:2100-2140`).

### The three drift mechanisms, and what each actually watches

- **`src/osprey/cli/build_profile_drift.py`** — *preset* drift. Compares
  `profile.yml` against the document `osprey init` would write from the
  installed preset today, keyed on the `provenance.preset_hash` stamp
  (`:1-25`). `REFUSAL_KINDS` (`:64-70`) — `missing`, `missing_member`, `extra`,
  `extra_member`, `exclude` — are what `osprey validate --drift=error` refuses;
  a differing *value* is `NOTE_KINDS` (`:72-77`), reported and never refused.
  A `# DEVIATION: why` comment within three lines above the line silences a
  finding (`MARKER_REACH`, `:79-80`); the tag is
  `provenance.deviation_marker`. It reads only what the repo tracks, never a
  host-variant overlay (`:207-208`).
- **`src/osprey/deployment/staleness.py`** — *build vs source* drift. One
  fingerprint over the resolved profile document, the file material it names
  (the `data:` tree, the convention directories, `triggers.yml`, the
  `personas/` deltas — `MATERIAL_KEY`, `:76-84`), the host-variant overlay
  `.env.variant` selects, and the `creation.osprey_version` stamp, compared
  against `.osprey-manifest.json`. Two readings (`:20-35`): the legacy
  `staleness_reasons` / `warn_if_project_stale` advisory that warns and fails
  open, and `check_drift`, the verdict `up`/`restart` **refuse** on, `status`
  prints, and `chat`/`query` warn on. A profile that no longer parses, or a
  build carrying no fingerprint, is `UNRESOLVABLE` and a start verb refuses
  rather than guessing; `--as-built` is the documented escape from both
  refusals.
- **`osprey_config_drift.py`** (SessionStart hook, selected as `config-drift`)
  — warns when `config.yml`'s mtime is newer than `.claude/settings.json`'s
  (check a), or when a single `control_system.writes_enabled` disagrees with
  this render's write tools in `permissions.deny` (check b). The tool list
  comes from the render's own `.claude/hooks/hook_config.json`, not a literal.
  It abstains on a config carrying per-connector-type posture blocks
  (`control_system.connector.<type>.writes_enabled`), since it cannot resolve
  which target the session will land on. Warn-only, never mutates or blocks,
  stdlib-only, fails open and silent on any IO/parse error.

### One further warning, on the regen path only

`check_user_owned_drift` (`claude_code.py:1927-1987`) compares each claimed
artifact's stored `framework_hash` — recorded at claim time by
`build_user_owned_manifest` (`manifest.py:449-495`) via
`framework_template_hash` (`manifest.py:62-108`, which renders a `.j2` first so
a comment change in the template is not drift) — against a fresh render of the
framework template, and prints:

```
⚠ Framework updated <name> since you claimed it.
  Run `osprey scaffold diff <name>` to review changes.
```

That detects the **framework** changing under a claim. It does not detect an
operator editing the build.
