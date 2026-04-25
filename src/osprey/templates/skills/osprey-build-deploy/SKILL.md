---
name: osprey-build-deploy
description: >
  Deployment control plane for an OSPREY-based facility profile repository. Owns the
  deploy lifecycle end-to-end: scaffolds the deploy infrastructure (docker-compose,
  .gitlab-ci.yml, scripts/deploy.sh, .env.template) from facility-config.yml, drives
  the GitLab CI/CD container pipeline (push → CI → manual release tag → server deploy),
  brings containers up on the deploy server with `scripts/deploy.sh`, and runs
  post-deploy health checks. Modular opt-in support for OLOG/logbook integration,
  multi-user web terminals, Ollama embeddings, shared-disk mounts, EPICS-driven event
  dispatcher, ARIEL database, wiki search, custom MCP servers, e2e benchmarks, and
  EPICS test IOC management. Use this skill whenever the user is asking about
  deploying, releasing, pushing to GitLab, building containers, modifying
  docker-compose files, scripts/deploy.sh, .gitlab-ci.yml, the CI pipeline, the
  container registry, the deploy server, the .env, the facility-config.yml, webhook
  triggers, the event dispatcher, web terminals, OLOG, Ollama, ARIEL, integration
  tests, test IOC, OSPREY:TEST PVs, agent benchmarks, or anything CI/CD or
  on-server-deploy related — even if they don't explicitly say "deploy". Mandatory
  entry point for any work touching the project's deploy pipeline; if the user says
  "let's set this up", "ship this", "release", "promote to prod", "rebuild on the
  server", or anything similar, invoke this skill before doing anything else. On
  first invocation, runs an interactive deploy interview to capture facility-specific
  values (GitLab host, deploy server, container runtime, proxy, ports, optional
  modules) and writes them to `facility-config.yml` at the repo root; subsequent
  invocations read that config. Does NOT author build profile YAMLs — that is the
  job of the separate `build-interview` skill (`/build-interview`). If the user wants
  to create or edit a profile, hand off to `/build-interview` and stop.
---

# OSPREY Build & Deploy

Deployment control plane for a facility profile repository.

This skill is project-local — it lives at `<profile-repo>/.claude/skills/osprey-build-deploy/` and is installed by `/build-interview` at the end of Phase 8 (when the profile repo is generated). To refresh or re-install it later (e.g., after upgrading OSPREY), run from the profile repo root:

```bash
osprey skills install osprey-build-deploy --target .claude/skills/
```

The previous copy is automatically backed up to `.claude/skills/osprey-build-deploy.bak.<timestamp>/`.

## Scope (read this — separation matters)

This skill owns **deployment**, not profile authoring. The split is:

| Concern | Skill | Produces / Operates on |
|---------|-------|-----------------------|
| Author the OSPREY build profile YAML for one assistant (signals, channels, write safety, AI provider, channel finder, archiver, dashboard) | **`build-interview`** (separate, invoked as `/build-interview`) | `build-profile/profile.yml`, channel databases, channel limits |
| Stand up CI/CD, deploy infra, on-server runtime; ongoing release operations | **this skill** | `docker-compose.yml`, `.gitlab-ci.yml`, `scripts/deploy.sh`, `.env.template`, container deploys, health checks |

**If the user wants to create or modify a profile YAML, hand off to `/build-interview` and stop.** Do not edit profile YAMLs in this skill — that's not its job and the two skills must not overlap.

This skill **gates on a deploy interview**. The interview is the only way the skill learns site-specific values (GitLab host, deploy server, ports, which optional modules the facility wants). Once the interview has run, those values live in `facility-config.yml` at the repo root and every subsequent action reads from there. Treat that file as the authoritative source of truth — never hardcode hostnames or ports inside generated files; always derive from config.

The skill is **control-system-agnostic at its core** — EPICS, DOOCS, TANGO, mock all valid. Control-system-specific operations (e.g., starting an EPICS test IOC) live in opt-in modules and are only offered when the corresponding `control_system.type` matches.

---

## STEP 0 — Configuration Gate (do this every time)

**Before doing anything else**, check whether `facility-config.yml` exists at the project root:

```bash
test -f facility-config.yml && echo present || echo missing
```

- **Missing** → the project hasn't been set up yet. Read `references/setup-interview.md` and walk the user through it. Do not proceed to any other action until the interview is complete and `facility-config.yml` is written. This is non-negotiable: every other section of this skill depends on values from that file.
- **Present** → load it (Read tool), parse the YAML mentally, and use it as the substitution source for every templated value below. When you see `${config.X.Y}` in this skill or any reference file, substitute the value from the loaded config.

If the user explicitly asks to **re-run** or **update** the interview, jump to `references/setup-interview.md` regardless — it knows how to merge new answers into existing config without losing untouched values.

### What the config contains (overview)

`facility-config.yml` has a small **core** block that every project needs (facility identity, control system, GitLab, deploy server, container runtime, proxy, ports) plus a **modules** block where each opt-in module is either enabled with its own sub-config or absent. See `references/facility-config-schema.md` for the full schema with comments.

The skill never asks the user the same setup question twice — answers are durable. If the user changes a value (e.g., the facility migrates to a new deploy server), update `facility-config.yml` and the next action picks up the new value automatically.

---

## Action Routing — Read This First

If the user's intent is clear, match it to one action below and go directly. If ambiguous, present the relevant subset using `AskUserQuestion`. Some actions are gated on optional modules — only offer them when the corresponding module is enabled in `facility-config.yml`. Some actions are gated on a specific control system type (e.g., test IOC management requires EPICS).

### Always available (core)

- **Scaffold deploy infrastructure** — generate `docker-compose.yml`, `.gitlab-ci.yml`, `scripts/deploy.sh`, `scripts/verify.sh`, `.env.template`, and a deploy `README.md` from templates, all parameterized by `facility-config.yml`. → `references/scaffolding.md`.
- **Deploy via CI/CD** — push to GitLab, watch CI build the container images, trigger the manual release job, run `scripts/deploy.sh` on the deploy server. → `references/gitlab-ci-pipeline.md` and `references/deploy-server.md`.
- **Build local client** — produce a Claude Code project on the developer's machine that connects to the remote MCP services running on the deploy server (no local containers). → `references/local-client.md`.
- **Diagnose deployment** — SSH into the deploy server, inspect container status, logs, networking, regenerated MCP config. → `references/post-deploy-diagnosis.md`.
- **Run integration health checks** — execute the `integration_tests` MCP server's `/checks` endpoint (or pytest equivalents) to verify infrastructure. → `references/integration-tests.md`.
- **Re-run / update the deploy interview** — modify `facility-config.yml` interactively. → `references/setup-interview.md`.

### Hand off to other skills

- **Author or edit a profile YAML, channel database, or channel limits** → use `/build-interview` instead. Do not handle this in this skill. Tell the user: *"Profile authoring is the job of the separate `build-interview` skill — try `/build-interview` and it will walk you through creating or updating the profile."*

### Gated on optional modules

Show only when the corresponding `modules.X.enabled: true` in `facility-config.yml`:

- **Manage the event dispatcher / webhook triggers / EPICS-driven agents** — `modules.event_dispatcher` → `references/modules/event-dispatcher.md`.
- **Manage web terminals (per-user containers + nginx)** — `modules.web_terminals` → `references/modules/web-terminals.md`.
- **Run e2e agent benchmarks** — `modules.benchmarks` → `references/modules/e2e-benchmarks.md`.
- **OLOG / logbook actions** — `modules.olog` → `references/modules/olog.md`.
- **ARIEL database (Postgres + embeddings)** — `modules.ariel` → `references/modules/ariel-database.md`.
- **Ollama / local embedding service** — `modules.ollama` → `references/modules/ollama-embeddings.md`.
- **Wiki search (Confluence or other)** — `modules.wiki_search` → `references/modules/wiki-search.md`.
- **Shared-disk mount (NFS, host bind-mount)** — `modules.shared_disk` → `references/modules/shared-disk.md`.
- **Custom facility MCP servers** — `modules.custom_mcp_servers` → `references/modules/custom-mcp-servers.md`.

### Gated on control system

- **EPICS test IOC management (start, stop, configure exotic CAS ports for full Channel Access isolation from production EPICS)** — requires `control_system.type == "epics"` AND `modules.test_ioc.enabled == true`. **Read `references/modules/test-ioc-safety.md` first** — it contains mandatory port-isolation rules that must be obeyed for any EPICS test IOC near a real accelerator.

When `control_system.type` is something else (DOOCS, TANGO, mock, custom), do not offer the EPICS test IOC action even if asked — explain that the module is EPICS-specific and their facility's equivalent test infrastructure is out of scope for this skill.

### When the action isn't clear

Use `AskUserQuestion` with options drawn from the subset above that's actually enabled. Do **not** offer disabled modules — if the user wants one, they can re-run the interview to enable it.

---

## Deploy Pipeline (Core Flow)

This is the canonical deploy path. Every facility's pipeline has the same shape; only values differ.

```
                     manual                   ssh
[git push] ──→ [CI builds N images] ──→ [release job re-tags as :latest] ──→ [deploy server: deploy.sh]
                                                                                       │
                                                                                       ▼
                                                                               podman/docker login
                                                                               compose pull
                                                                               compose up -d
                                                                               verify.sh (advisory)
```

### Quick reference (substitute from config)

```bash
# 1. Push (triggers CI)
git push ${config.gitlab.remote_name} ${config.gitlab.default_branch}

# 2. Watch the pipeline until status: "manual"
curl -s --header "PRIVATE-TOKEN: $${config.gitlab.token_env_var}" \
  "https://${config.gitlab.host}/api/v4/projects/${config.gitlab.project_id}/pipelines?per_page=1" \
  | python3 -m json.tool

# 3. Trigger the manual `release` job (GitLab UI, or POST to /jobs/<id>/play)

# 4. Deploy on the server (single command pulls code + images + starts containers + verifies)
ssh ${config.deploy.host} "cd ${config.deploy.project_path} && ./scripts/deploy.sh"

# 4b. Clean restart (compose down first)
ssh ${config.deploy.host} "cd ${config.deploy.project_path} && ./scripts/deploy.sh --clean"

# 4c. From-scratch rebuild (also wipes images, volumes, networks)
ssh ${config.deploy.host} "cd ${config.deploy.project_path} && ./scripts/deploy.sh --nuke"
```

| Flag | What `deploy.sh` does |
|------|------------------------|
| (none) | git pull → registry login → pull images → restart changed containers → verify |
| `--clean` | git pull → **compose down** → registry login → pull images → start all → verify |
| `--nuke` | git pull → **compose down + remove images/volumes/networks** → registry login → pull images → start all → verify |

`verify.sh` runs after every deploy and is **advisory** — it never fails the deploy, just reports health. If it surfaces a real problem, the operator decides what to do.

### What NOT to do before deploying

These are anti-patterns the operations team learned to avoid:

- SSH into the deploy server to "check what's there" — the deploy is idempotent, just run it.
- Compare git logs between local and remote — if you pushed, CI has it; the server pulls on deploy.
- Read `.env.production` or `config.yml` on the remote — they're generated/templated, inspect locally.
- `git diff` to "summarize what changed" — irrelevant; the deploy is what runs.
- Make a "what changed" table — the commit message and CI logs are the source of truth.

**What to do instead:** sync the code, run the deploy, read its output. If something fails, investigate that specific failure with `references/post-deploy-diagnosis.md`.

### Pre-deploy checklist (substitute from config)

1. Working tree clean: `git status` shows nothing.
2. Pushed: `git push ${config.gitlab.remote_name} ${config.gitlab.default_branch}`.
3. CI passes through `docker-build` and lands at `release` (manual gate).
4. Trigger the release job (re-tags all images as `:latest`).
5. Run `deploy.sh` on the server.

---

## Local Client Build (developer workflow)

A "client" build produces a Claude Code project on a developer's machine that talks to the **remotely deployed MCP services** instead of running its own containers. Useful for developers who need the full assistant without standing up the whole stack locally.

This is a thin wrapper around `osprey build` against a `*-client.yml` profile. The skill does **not** author the profile (use `/build-interview` for that); it documents the build + connection workflow once the profile already exists.

Prerequisites:
- `pip install osprey-framework`
- `.env.local` exists in the project repo (needs the LLM provider key, `${config.llm.api_key_env_var}`)
- Network access to the deploy server (typically requires being on the facility's control network)

Build command (substitute from config):

```bash
osprey build ${config.facility.prefix}-client \
  $(pwd)/${config.facility.prefix}-client.yml \
  -o ~/projects --force
```

Then: `cd ~/projects/${config.facility.prefix}-client && claude`.

The Claude Code instance connects to `http://${config.deploy.host}:<port>/mcp` for each remote MCP server (port from `${config.ports.<server_name>}`). See `references/local-client.md` for full details and troubleshooting.

---

## Eliminating Manual Steps

Every manual step in deployment is a bug — it means the templates or `facility-config.yml` are missing something. The ideal deploy is: sync code, run one command. If you find yourself running more than two SSH commands during a deploy, stop and ask why.

| Need | Automate via | Not manual commands |
|------|-------------|---------------------|
| Start/stop containers | `lifecycle.post_build:` + compose | Standalone SSH + podman/docker |
| Install Python deps | `dependencies:` list in profile | `pip install` or `uv pip install` |
| Copy files to project | `overlay:` section in profile | Manual `cp` into project tree |
| Set env vars | `env:` section + `.env.production` | Editing `.env` by hand |
| Index search data | `lifecycle.post_build:` steps | Standalone index commands |
| Validate deployment | `lifecycle.validate:` | Ad-hoc SSH + test scripts |
| Configure services | `services:` section | Manual `config.yml` edits |
| Pull images on the server | `deploy.sh` | `podman pull <image>` one-by-one |

If a real workflow can't be expressed through these primitives, fix OSPREY (see next section) or extend `facility-config.yml` and the templates rather than papering over with shell commands.

---

## Fixing OSPREY When the Templates Can't Express What You Need

OSPREY is actively developed. When something can't be expressed through the current templates or build system, change OSPREY directly. A 10-line change that eliminates a manual step for all facilities is always worth making.

| What | Where (relative to OSPREY repo root) |
|------|--------------------------------------|
| Build command (presets, overrides, --set) | `src/osprey/cli/build_cmd.py` |
| Bundled preset YAML profiles | `src/osprey/profiles/presets/` |
| Profile schema/dataclasses | `src/osprey/cli/build_profile.py` |
| Template manager | `src/osprey/cli/templates/manager.py` |
| Scaffolding (file copy logic) | `src/osprey/cli/templates/scaffolding.py` |
| App templates | `src/osprey/templates/apps/` |
| Project template (single-profile build) | `src/osprey/templates/project/` |
| Bundled skills (this skill + `build-interview`) | `src/osprey/templates/skills/` |
| Built-assistant Claude templates | `src/osprey/templates/claude_code/` |

Workflow:
1. Identify the gap (missing config key, schema limitation, template feature, build pipeline gap).
2. Make the minimal, generic change that benefits any facility — not a one-off hack for the facility you're working on.
3. Add tests in `tests/cli/`.
4. Push to OSPREY's repo, follow OSPREY's own release flow.
5. Bump the OSPREY pin in the facility's CI base image (`pyproject.toml` or `requirements.txt`).

---

## Key Mechanics (OSPREY internals operators rely on)

- **Overlay paths**: source relative to the profile YAML's directory, destination relative to project root. No `..` traversal allowed.
- **`{project_root}`**: resolved at build time. **`${ENV_VAR}`**: preserved for runtime substitution.
- **Config overrides use dot notation**: `control_system.type: epics` → nested `config.yml` key.
- **Manifest-driven**: only artifacts in the template's `manifest.yml` get generated.
- **`.env` auto-injection**: OSPREY parses `.env` and passes variables to lifecycle subprocesses via `env=`. Lifecycle commands don't need `set -a; . .env; set +a` preambles.
- **PYTHONPATH auto-injection**: OSPREY prepends `_mcp_servers` to `PYTHONPATH` for all lifecycle commands. Wrap in `sh -c` only if the command uses `${ENV_VAR}` expansion.
- **Dependencies**: appended to `requirements.txt`, installed in the project `.venv`. Lifecycle commands and MCP servers use this venv automatically.
- **`osprey build --force` is destructive**: wipes the entire output project directory. Back up `.env` and manual customizations first.
- **The built assistant is its own git repo**, sibling to the facility profile repo — not nested inside it.

---

## Consistency Rule

When modifying any file in the facility profile repo, verify the change reaches all the places that depend on it:

- **Both profiles** (prod + client) — does the change apply to both?
- **Overlay entries** — source path exists, destination correct?
- **MCP server entries** — `PYTHONPATH` set, permissions listed?
- **MCP tool sync** — if you add, remove, or rename an `@mcp.tool()`, three lists must agree:
  1. `@mcp.tool()` decorators in source
  2. `permissions.allow` in the profile YAML
  3. `EXPECTED_TOOLS` in the integration tests check (if `modules.integration_tests` is enabled)
- **Lifecycle hooks** — OSPREY auto-injects `_mcp_servers` into PYTHONPATH; only wrap in `sh -c` for `${ENV_VAR}` expansion.
- **`.env.template`** — if you add a required env var anywhere, it must appear in `.env.template` so operators know to set it.
- **`facility-config.yml`** — if a new piece of facility-specific data is now needed, extend the schema and ask the user via the interview rather than hardcoding.
- **After deploy** — run integration health checks to verify.

---

## When to Update This Skill

If the deploy mechanics change, update the skill. Stale skills cause failed deploys.

- New scaffolding template → add to `templates/core/` or `templates/modules/<name>/`.
- New optional module → add a section under "Gated on optional modules" above, write `references/modules/<name>.md`, and extend the interview in `references/setup-interview.md`.
- New config field → update `references/facility-config-schema.md` AND `references/setup-interview.md` to ask the question.
- New OSPREY CLI flag or build phase that affects deploy → update the relevant section above.
- New deploy mode (beyond `--clean` and `--nuke`) → document under "Deploy Pipeline".
- New control-system type with its own deploy quirks → add a control-system-gated section to "Action Routing" and a module reference for the system-specific operations.

The skill and its references are the single source of truth that future sessions use to operate this project. If the skill is stale, future deploys fail or use outdated commands.

---

## Reference index

| Reference | When to read |
|-----------|--------------|
| `references/setup-interview.md` | First-time setup, or when re-running the deploy interview |
| `references/facility-config-schema.md` | Editing `facility-config.yml` by hand, understanding what each field means |
| `references/scaffolding.md` | Generating deploy-infra files from templates after an interview |
| `references/gitlab-ci-pipeline.md` | All CI/CD pipeline questions (stages, jobs, registry, tagging) |
| `references/deploy-server.md` | On-server setup, `.env.production`, container runtime install |
| `references/local-client.md` | Local Mac/laptop builds connecting to remote MCP |
| `references/integration-tests.md` | Health check architecture (always available, not module-gated) |
| `references/post-deploy-diagnosis.md` | Container is sick after deploy — what to inspect |
| `references/modules/<name>.md` | Module-specific operations (only when that module is enabled) |
| `references/modules/test-ioc-safety.md` | **Read before any EPICS test IOC operation** — mandatory port-isolation rules |
