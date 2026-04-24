# Scaffolding — Generating Deploy Infrastructure From `facility-config.yml`

After the deploy interview produces `facility-config.yml`, the next action is **scaffolding**: rendering the templates in `templates/` into real files at the repo root. The output is everything the deploy pipeline needs:

- `docker-compose.yml` (and overrides per `runtime.compose_files`)
- `.gitlab-ci.yml`
- `scripts/deploy.sh`
- `scripts/verify.sh`
- `.env.template`
- `README.md` (deploy-section appended to existing)

Plus per-module additions when modules are enabled (e.g., `nginx/` for web terminals, `triggers.yml` for event dispatcher).

This file describes how rendering works. Read it before generating any file.

---

## Approach: render from prose templates, not literal copy

The templates in `templates/core/` and `templates/modules/` are **skeleton text with placeholder syntax** — not Jinja2, not handlebars, not any specific engine. The skill renders them by reading the template + `facility-config.yml`, doing substitutions and conditional expansion, and writing the result.

This sounds informal but is deliberate: the substitutions involve cross-references and conditional logic that's awkward in templating engines and natural for an LLM. The benefit is templates that are readable as plain text examples, not noisy with `{% if %}` tags.

### Placeholder syntax used in templates

When you see `${config.X.Y}` in a template, substitute the value from the loaded `facility-config.yml`. Examples:

| Placeholder | Source | Example resolved value |
|-------------|--------|------------------------|
| `${config.facility.prefix}` | `facility.prefix` | `als` |
| `${config.gitlab.host}` | `gitlab.host` | `git.als.lbl.gov` |
| `${config.deploy.host}` | `deploy.host` | `appsdev2` |
| `${config.registry.url}` | `registry.url` | `git.als.lbl.gov:5050/physics/production/als-profiles` |
| `${config.ports.matlab}` | `ports.matlab` | `8001` |

When you see `${env.VAR_NAME}` in a template, that's a shell env var reference that should remain literal in the output (becomes `$VAR_NAME` in the rendered file). Used in deploy.sh and compose files.

### Conditional sections

Templates use comment-marked conditional blocks. Two flavors:

```
# IF MODULE event_dispatcher.enabled
... block to include only if modules.event_dispatcher.enabled is true ...
# END IF
```

```
# IF network.http_proxy
... block to include only if the referenced config value is truthy ...
# END IF

# IF registry.external_projects
... block to include only if the list is non-empty ...
# END IF
```

**Distinction:** use `# IF MODULE <name>.enabled` when gating on a `modules.<name>` block — this form is the advertised interface for module-gated blocks and is the only form that propagates into the SKILL.md module list. Use `# IF <dotted.path>` for any other configuration gate (proxy set, external_projects list non-empty, a feature flag that isn't a module). Mixing the two produces false "module missing" errors in the cross-reference audit.

Nested conditionals are supported:

```
# IF MODULE ariel.enabled
# IF modules.ariel.deployment == container
... only when ARIEL is enabled AND deployed in-compose ...
# END IF
# END IF
```

An `# ELSE` branch is supported inside any `# IF` block:

```
# IF modules.event_dispatcher.epics_ca.enabled
network_mode: host
# ELSE
networks: [ ${config.facility.prefix}-net ]
# END IF
```

When rendering: include the matching branch verbatim, omit the other. The marker comments are not written to the output.

### List expansion

Some templates expand a list from config — e.g., one compose service per custom MCP server, or one container per web terminal user:

```
# FOR each in modules.web_terminals.users
  ${config.facility.prefix}-web-${each}:
    image: ${config.registry.url}/web-terminal:latest
    ports:
      - "${config.modules.web_terminals.base_port + each.index}:8000"
# END FOR
```

When rendering: instantiate the block once per list element, with `${each}` and `${each.index}` substituted appropriately. The marker comments are not written.

**Numeric range form** — used for fixed-count expansions like sidecar containers:

```
# FOR i in 0..${config.modules.event_dispatcher.sidecar_count - 1}
  dispatch-sidecar-${i}:
    ports: [ "${config.modules.event_dispatcher.sidecar_port_base + i}:9100" ]
# END FOR
```

The renderer evaluates the endpoint expression once, then iterates `i` over the inclusive integer range.

**Bare-name resolution inside `FOR each`** — when a list element refers to another service by name (e.g., `modules.custom_mcp_servers.servers[].depends_on`, `modules.shared_disk.services_to_mount`), the scaffolder resolves the bare name to the fully-qualified compose service key at render time. For a custom MCP server whose `name: matlab`, a bare `"matlab"` in another module's list resolves to `${config.facility.prefix}-mcp-matlab` (the actual service key) before being written. This keeps module configs readable while the compose output stays consistent.

### Arithmetic inside placeholders

`${config.X + N}` and `${config.X - N}` (simple integer +/-) are supported inside placeholders and `FOR` range endpoints:

- `${config.modules.web_terminals.base_port + each.index}` → `9091`, `9092`, …
- `${config.modules.event_dispatcher.sidecar_port_base + i}` → `9190`, `9191`, …
- `0..${config.modules.event_dispatcher.sidecar_count - 1}` → range endpoint

Nothing more complex than integer +/- is supported — if you need multiplication, division, or string manipulation, do it in config or in a helper shell function inside the template, not inline.

---

## Rendering order

Render in dependency order so cross-references resolve cleanly:

1. **`.env.template`** first — every other file references env vars; templates reference the var *names* (which match `*_env_var` fields in config), and the template lists every var with a placeholder value and a comment.
2. **`docker-compose.yml`** — base services + module additions. List custom MCP servers from `modules.custom_mcp_servers.servers`. Add module-specific services (postgres if ARIEL, ollama if Ollama, event-dispatcher + sidecars if event_dispatcher, web-terminal containers + nginx if web_terminals).
3. **Compose overlays** (e.g., `docker-compose.host.yml`) — host-network or other override files, one per entry in `runtime.compose_files` after the first.
4. **`.gitlab-ci.yml`** — stages, jobs, registry references. One `build-${name}` job per custom MCP server.
5. **`scripts/deploy.sh`** — uses values from runtime, registry, deploy. Includes module-specific pull steps (e.g., per-external-project pulls).
6. **`scripts/verify.sh`** — health checks for every enabled service. Always advisory.
7. **`README.md`** — deploy-section appended to any existing README. Lists what was generated, where to look for what, common operations.

---

## Idempotence and safe overwrites

**Generated files must be safe to regenerate.** The user will re-run scaffolding whenever they:
- Add or remove a module via the interview.
- Change a port allocation.
- Add a custom MCP server.
- Update `facility-config.yml` for any reason.

**Files the skill owns and may overwrite without asking:**
- `.env.template` — always regenerated; user-edited values live in `.env` (gitignored), not `.env.template`.
- `docker-compose.yml` — regenerated; users with custom additions should keep them in `docker-compose.local.yml` (gitignored, added to `runtime.compose_files` after the base file).
- `scripts/deploy.sh`, `scripts/verify.sh` — regenerated; users should not edit these directly. If they need a custom step, add it to the templates and contribute back, or use a `scripts/deploy.local.sh` wrapper.
- `.gitlab-ci.yml` — regenerated; users should not edit directly.

**Files the skill never overwrites:**
- `.env` (user-managed secrets).
- `facility-config.yml` (only updated through the interview).
- Any user-added files outside the explicit "owned" list above.

**Before overwriting any owned file, diff against the existing version and show the user.** If the diff is large or the user has clearly hand-edited a generated file, ask before proceeding.

---

## Module-conditional generation

Each enabled module triggers additions to one or more rendered files. The mapping:

| Module | Reference file | Affects |
|--------|----------------|---------|
| `event_dispatcher` | `modules/event-dispatcher.md` | adds `event-dispatcher` + N `dispatch-sidecar-${i}` services to compose (both host-networked when `epics_ca.enabled`); adds `triggers.yml` if missing; adds dispatcher + sidecar tokens to `.env.template`; adds `build-event-dispatcher` AND `build-dispatch-sidecar` CI jobs and their release entries |
| `web_terminals` | `modules/web-terminals.md` | adds N web-terminal services + nginx to compose; renders `nginx/nginx.conf` and `nginx/landing.html`; adds `build-web-terminal` CI build job; adds per-user CLAUDE.md seed step to deploy.sh |
| `olog` | `modules/olog.md` | adds OLOG creds to `.env.template`; adds OLOG check to verify.sh; affects integration_tests config |
| `ariel` | `modules/ariel-database.md` | if `deployment: container`, adds `ariel-postgres` service to compose with named volume `ariel_postgres_data`; adds `ariel-sync` one-shot service; always adds `ARIEL_DSN` to `.env.template` and the CI `.env.production` assembly; adds ARIEL check to verify.sh |
| `ollama` | `modules/ollama-embeddings.md` | does NOT add a container (Ollama lives elsewhere); adds `OLLAMA_URL` + `OLLAMA_EMBEDDING_MODEL` to env and verify check |
| `wiki_search` | `modules/wiki-search.md` | adds wiki creds to `.env.template`; adds wiki check to verify.sh |
| `shared_disk` | `modules/shared-disk.md` | adds bind-mount to specified compose services; adds a host_path pre-flight check to deploy.sh; adds mount-presence probe to verify.sh |
| `custom_mcp_servers` | `modules/custom-mcp-servers.md` | one compose service (`${prefix}-mcp-${name}`) + one `build-${name}` CI build job per server, both guarded by `IF MODULE custom_mcp_servers.enabled`; bare names in `depends_on` and `shared_disk.services_to_mount` resolve to full service keys |
| `benchmarks` | `modules/e2e-benchmarks.md` | adds the `modules.benchmarks.suite_path` directory to the web-terminal image's build context (COPYed into `/app/.../data/benchmarks/`); documents `podman exec` invocation in deploy README. No standalone compose service. |
| `test_ioc` | `modules/test-ioc-safety.md` | NO compose changes (test IOC runs as a host process on exotic CAS ports, not a container); scaffolds `${config.modules.test_ioc.db_path}` skeleton if absent and writes a startup-script template to `${config.modules.test_ioc.startup_script_path}`. Gated on `control_system.type == "epics"`. |

When rendering, walk through `modules.*` and for each `enabled: true` entry, apply that module's contributions to the relevant files. Order matters — e.g., add the `ariel-postgres` service before any service that declares `depends_on: [ariel-postgres]`.

### Module reference file naming

Module schema names and reference filenames do not match mechanically. The mapping is fixed — don't invent new filenames without updating the table above:

| Schema key (`modules.X`) | Reference file |
|--------------------------|----------------|
| `ariel` | `references/modules/ariel-database.md` |
| `ollama` | `references/modules/ollama-embeddings.md` |
| `shared_disk` | `references/modules/shared-disk.md` |
| `benchmarks` | `references/modules/e2e-benchmarks.md` |
| `test_ioc` | `references/modules/test-ioc-safety.md` |
| `event_dispatcher`, `web_terminals`, `olog`, `wiki_search`, `custom_mcp_servers` | `references/modules/<same-name-with-hyphens>.md` |

When adding a new module, add a row to both tables and to `SKILL.md`'s "Gated on optional modules" section — the cross-reference audit treats any unlisted module as an integration break.

---

## Validation after generation

Before declaring scaffolding done:

1. **Compose files parse** — `${runtime.compose_command} -f docker-compose.yml -f <overlays> config` should succeed and produce the merged config without errors.
2. **`.gitlab-ci.yml` parses** — at minimum, valid YAML and uses no undefined CI variables.
3. **`scripts/deploy.sh` is syntactically valid** — `bash -n scripts/deploy.sh`.
4. **No leftover placeholders** — grep for `${config.` in generated files; should return nothing. If any survived, that's a rendering bug — surface it.
5. **`.env.template` covers every secret referenced** in compose, deploy.sh, .gitlab-ci.yml. Every `${ENV_VAR}` reference must have a matching entry.

---

## What to tell the user after scaffolding

```
Scaffolding complete. Generated:

  .env.template        — copy to .env and fill in secrets
  docker-compose.yml   — N services (M custom MCP servers + module additions)
  docker-compose.host.yml  — host-network overlay
  .gitlab-ci.yml       — N CI jobs
  scripts/deploy.sh    — pull + start + verify
  scripts/verify.sh    — post-deploy health checks
  README.md            — deploy section appended

Next steps:
  1. cp .env.template .env  AND fill in your secrets
  2. Review the generated files (they're plain text, you can read them)
  3. git add .  &&  git commit  &&  git push to start CI
  4. Watch the pipeline; trigger the manual `release` job when CI lands at it
  5. ssh into your deploy server and run scripts/deploy.sh
```

If any modules are enabled, mention what additional setup the user needs:
- web_terminals: "Each user will need a `docker/web-terminal-context/<user>.md` file with their per-user Claude Code memory; create empty stubs to start."
- event_dispatcher: "Edit `triggers.yml` to define your webhook and EPICS-CA triggers; the file is generated empty."
- test_ioc: "Author `${config.modules.test_ioc.db_path}` with your test PV definitions; see `references/modules/test-ioc-safety.md`."

---

## When templates can't express what you need

Same principle as the SKILL.md "Fixing OSPREY" section: if a real facility need can't be expressed by adding a placeholder, conditional, or list expansion to a template, don't paper over it with a one-off shell command. Either:

1. Extend the template + add a corresponding question to the interview + add a field to the schema.
2. Or, if the gap is in OSPREY itself (e.g., the build command can't do what you need), fix OSPREY.

Templates evolve with the skill. Future facilities benefit from every gap closed.
