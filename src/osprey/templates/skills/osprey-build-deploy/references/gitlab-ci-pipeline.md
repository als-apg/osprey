# GitLab CI/CD Pipeline Reference

CI-native deployment for an OSPREY facility profile repo. Every container image (core MCP servers, custom MCP servers, web terminal, event dispatcher) is built and published through a five-stage GitLab CI/CD pipeline. The deploy server only ever pulls and runs — it never builds. OSPREY renders the Claude Code project at CI time so the web-terminal image can bake it in.

This file documents the pipeline shape, jobs, variables, tagging, and the operations on top of it. Every concrete value (host, project ID, ports, token names) comes from `facility-config.yml`.

---

## Architecture

```
GitLab CI/CD pipeline (5 stages):
  build-ci-image → osprey-build → checks → docker-build → release
                        ↓
       config.yml + module data artifacts
       (everything written to artifacts/)
                        ↓
                 docker-build (N parallel)
                        ↓
       ${config.registry.url}        (one image per service)
                        ↓
${config.deploy.host}: deploy.sh → registry login → compose pull → compose up -d
```

Every stage runs in a runner with internet access only via `${config.network.http_proxy}` (if set), so `HTTP_PROXY`/`HTTPS_PROXY`/`NO_PROXY` must be passed through to every job that talks outside the GitLab host.

---

## GitLab projects

Read these from `facility-config.yml`:

| Field | Source | Used for |
|-------|--------|----------|
| GitLab host | `${config.gitlab.host}` | Registry hostname, CI URLs, OSPREY mirror reachability |
| Project path | `${config.gitlab.project_path}` | Image namespace, deploy.sh login target |
| Project ID | `${config.gitlab.project_id}` | API URLs (`/api/v4/projects/${id}/...`) |
| Default branch | `${config.gitlab.default_branch}` | Branch the release stage gates on |
| PAT env var | `${config.gitlab.token_env_var}` | All authenticated git/registry/API calls |

The OSPREY framework itself is installed into the CI base image from a GitLab-reachable mirror (most facility GitLab instances cannot reach GitHub directly). That mirror is referenced inside `docker/Dockerfile.ci` — its location is not part of `facility-config.yml`, but the token used to clone it is `${config.gitlab.token_env_var}`.

`${config.registry.external_projects[*]}` lists sibling projects whose images this deploy also pulls (e.g., a separate team's service with its own pipeline). Each external project needs:
- Its own deploy token created in that project, scope `read_registry`.
- An entry in `.env.production` for the token, name from `<external>.token_env_var`.
- A `podman pull --creds` (or `docker pull` after explicit login) step in `scripts/deploy.sh` before the main `compose pull`.

---

## CI variables (project-level, masked)

These must be added at GitLab project Settings → CI/CD → Variables, all masked, all protected to the default branch. The skill never writes them — they're set once per project by an operator.

Required for every facility:

| Variable name | Purpose |
|---------------|---------|
| `${config.llm.api_key_env_var}` | LLM provider key for the assistant (baked into web-terminal image's `.env.production` at build time) |
| `${config.gitlab.token_env_var}` | Used by every CI job for git clone, registry login, API calls. Must have `api`, `read_registry`, and `write_registry` scopes |

Module-conditional variables (only required when the module is enabled in `facility-config.yml`):

| Module | Variable name(s) |
|--------|------------------|
| `modules.olog.enabled` | `${config.modules.olog.username_env_var}`, `${config.modules.olog.password_env_var}` |
| `modules.wiki_search.enabled` | `${config.modules.wiki_search.token_env_var}` |
| `modules.event_dispatcher.enabled` | `${config.modules.event_dispatcher.token_env_var}`, `${config.modules.event_dispatcher.sidecar_token_env_var}` |
| `modules.custom_mcp_servers.servers[*].build_args` | Whatever build-time secrets each custom server needs (e.g., a search-index API key) |
| `modules.registry.external_projects[*].token_env_var` | One per external project |

If a required variable is missing, CI fails at the first job that references it (usually `osprey-build` or `docker-build-web-terminal`). Always check Settings → CI/CD → Variables before debugging deeper.

---

## Image tagging convention

Every image is tagged twice:

1. **`${config.registry.url}/<service>:${env.CI_COMMIT_SHORT_SHA}`** — written during `docker-build`. Immutable, traceable to a specific commit. The deploy server never pulls these directly, but they're useful for rollback (`podman pull <image>:<sha>` then re-tag locally).
2. **`${config.registry.url}/<service>:latest`** — re-tagged during the manual `release` stage on the default branch. This is what `compose pull` fetches.

Why two tags: separating "built" from "released" creates a manual gate. CI is allowed to build broken images; only an operator pressing the release button promotes them to `:latest`. This is the only protection between a bad commit and a production deploy.

The release stage is restricted to `${config.gitlab.default_branch}` and configured `when: manual` (see Stage 5 below).

---

## Pipeline stages — detail

### Stage 1: `build-ci-image`

Builds `${config.registry.url}/ci-base:latest` — the Python image with all profile dependencies + OSPREY pre-installed.

**Triggers** (any of):
- Changes to `pyproject.toml`
- Changes to `docker/Dockerfile.ci`
- Changes to `.gitlab-ci.yml`
- Manual run via GitLab UI

**Why these triggers**: anything that affects the *contents* of the base image must rebuild it. The CI YAML is in the trigger list because tweaking job definitions sometimes reveals base-image gaps (e.g., a missing apt package); a manual rebuild is then needed to test the change.

**What it does**:
1. `docker login ${config.registry.url%%/*}` using `${config.gitlab.token_env_var}`.
2. `docker build --no-cache -t ${config.registry.url}/ci-base:latest -f docker/Dockerfile.ci .`
3. `docker push ${config.registry.url}/ci-base:latest`

`--no-cache` is mandatory: pip layers cache aggressively and silently use stale OSPREY versions when the upstream pin advances. The trigger frequency is low enough that the ~2-minute rebuild is acceptable.

**Env vars needed**: `${config.gitlab.token_env_var}`, plus proxy vars from `${config.network.*}` if the runner needs them to reach OSPREY's mirror.

### Stage 2: `osprey-build`

Runs in the freshly-built `ci-base:latest`. Produces all data artifacts that downstream `docker-build` jobs `COPY` into their images.

**Triggers**: every pipeline (no path filter).

**What it does**:
1. Generates `.env.production` from CI masked variables (every required env var listed in the prod profile YAML must be available as a CI variable; the job concatenates them into the file).
2. Sets `PYTHONPATH=mcp_servers` so `python -m <server>` invocations work for any custom MCP server lifecycle steps.
3. Runs `osprey build ${config.facility.prefix}-assistant ${config.facility.prefix}-prod.yml -o build-output --skip-lifecycle --skip-deps --runtime-root /app/${config.facility.prefix}-assistant`.
   - `--skip-lifecycle`: there's no container runtime in CI, so any pre/post-build lifecycle hooks (which often start containers) must not run here.
   - `--skip-deps`: the CI base image already pip-installed everything; the per-project `.venv` would be redundant.
   - `--runtime-root`: tells OSPREY what path the project will occupy *inside the web-terminal container*, not on the CI runner. This is what makes generated MCP commands like `/app/${config.facility.prefix}-assistant/.venv/bin/python` resolvable at container runtime.
4. Runs any module-specific data builds. Each enabled module that needs build-time data emits its artifact under `artifacts/`. Examples:
   - If `modules.ariel.enabled`, a DuckDB file may be assembled from upstream JSON (one-shot indexing per build).
   - If a custom MCP server lists `artifacts:` in `modules.custom_mcp_servers.servers[*]`, the source files are produced by a job step here and placed at the named paths.
5. Copies `build-output/${config.facility.prefix}-assistant/config.yml` → `artifacts/config.yml` (so any container that needs the rendered project config can `COPY` it without re-running OSPREY).

**Artifacts produced**: everything under `artifacts/` is uploaded as a CI artifact and made available to subsequent stages via `dependencies:`. Typical contents:
- `artifacts/config.yml` (the rendered OSPREY config for the prod profile)
- `artifacts/${config.facility.prefix}-assistant/` (the entire rendered Claude Code project — used only by the web-terminal image)
- `artifacts/.env.production` (gitignored, used by the web-terminal image)
- Any module-specific artifacts (databases, indexes, etc.)

**Env vars needed**: every required env var listed in the prod profile (CI fails fast if any are missing) plus the ones referenced by each enabled module.

### Stage 3: `checks` (parallel)

Lightweight gates that fail the pipeline early.

**Jobs**:
- `lint`: ruff (or whatever linter is configured) on `mcp_servers/` and `tests/`.
- `test`: `pytest` integration tests, **excluding** marker categories that need running services. The exclusion list depends on which modules are enabled — anything that requires a live archiver, database, or external API gets a marker and is skipped here. Live-service integration tests run post-deploy via `references/integration-tests.md`, not in CI.
- Module-conditional: if `modules.benchmarks.enabled`, a smoke benchmark may run here (single-prompt fast check, not the full suite). The full suite runs post-deploy.

**Triggers**: every pipeline.

**Env vars needed**: minimal — these jobs run fully offline against fixtures.

### Stage 4: `docker-build` (parallel)

One job per buildable image. Every job:
1. Logs into `${config.registry.url%%/*}` using `${config.gitlab.token_env_var}`.
2. Pulls the artifacts from `osprey-build`.
3. Builds the image with `${env.CI_COMMIT_SHORT_SHA}` tag.
4. Pushes the SHA-tagged image.

**Always-built images**:

| Image | Dockerfile | Artifacts COPYed | Notes |
|-------|------------|------------------|-------|
| `web-terminal` | `docker/Dockerfile.web-terminal` | `artifacts/${config.facility.prefix}-assistant/`, `artifacts/.env.production` | Bakes Claude CLI, OSPREY, the rendered project, and the dispatch sidecar. Runs supervisord. |

**Module-conditional images** — one per enabled module that owns a container:

| Module | Image | Notes |
|--------|-------|-------|
| `modules.event_dispatcher` | `event-dispatcher` | COPYs `${config.modules.event_dispatcher.triggers_file}` and any dashboard HTML |
| `modules.integration_tests` (always built when integration_tests is in core) | `integration-tests` | COPYs `artifacts/config.yml` plus any module data needed for checks |

**Per-custom-MCP-server images** — one job per entry in `modules.custom_mcp_servers.servers`:

```
# FOR each in modules.custom_mcp_servers.servers
docker-build-${each.name}:
  stage: docker-build
  script:
    - docker login ${config.registry.url%%/*} -u gitlab-ci-token -p ${env.CI_JOB_TOKEN}
    - docker build -t ${config.registry.url}/mcp-${each.name}:${env.CI_COMMIT_SHORT_SHA} \
                   -f ${each.dockerfile} ${each.build_context}
    - docker push ${config.registry.url}/mcp-${each.name}:${env.CI_COMMIT_SHORT_SHA}
  needs: [osprey-build]
# END FOR
```

Each job pulls only the artifacts it needs (declared via `dependencies:`). A custom MCP server that lists `artifacts: ["artifacts/foo.db"]` in its config gets exactly that one file in its build context, not the full artifact set.

**Env vars needed**: `${config.gitlab.token_env_var}` (or the per-job `${env.CI_JOB_TOKEN}`, which works for the same project's registry).

### Stage 5: `release` (manual, default-branch only)

Re-tags every SHA-tagged image as `:latest`.

**Restrictions**:
- `only: [${config.gitlab.default_branch}]` — never runs on feature branches.
- `when: manual` — only fires when an operator clicks "play" in the GitLab UI.

**What it does**, for each built image:
```
docker pull ${config.registry.url}/<image>:${env.CI_COMMIT_SHORT_SHA}
docker tag  ${config.registry.url}/<image>:${env.CI_COMMIT_SHORT_SHA} ${config.registry.url}/<image>:latest
docker push ${config.registry.url}/<image>:latest
```

**Triggering manually**:
- **GitLab UI**: navigate to the pipeline, find the `release` job in the rightmost column, click play.
- **API**:
  ```bash
  # Find the job ID for the release stage on the latest pipeline
  curl -s -H "PRIVATE-TOKEN: ${env.${config.gitlab.token_env_var}}" \
    "https://${config.gitlab.host}/api/v4/projects/${config.gitlab.project_id}/pipelines?per_page=1" \
    | python3 -c "import json,sys;p=json.load(sys.stdin)[0];print(p['id'])"
  # then list jobs and POST /play to the release one:
  curl -X POST -H "PRIVATE-TOKEN: ${env.${config.gitlab.token_env_var}}" \
    "https://${config.gitlab.host}/api/v4/projects/${config.gitlab.project_id}/jobs/<job_id>/play"
  ```

After the release job succeeds, the deploy server can `compose pull` and pick up the new images.

---

## CI base image: `pyproject.toml` + `Dockerfile.ci`

The base image is the only place where Python/system dependencies are declared. Every other image inherits from it. Two layers contribute:

**`pyproject.toml`** — Python deps. Every package any MCP server, lifecycle hook, or test needs. Update this when:
- A new MCP server adds a dependency.
- A test file imports something new.
- An OSPREY release requires a newer transitive dep.

**`docker/Dockerfile.ci`** — system layer + OSPREY install. Update this when:
- A new system package is needed (apt-get install).
- The OSPREY pin changes (`pip install osprey-framework@<branch-or-tag>`).
- Proxy or PyPI mirror config changes.

**Rebuilding the base image** is automatic when either file changes (see Stage 1 triggers). When OSPREY itself advances on its mirror without a `pyproject.toml` bump, you must trigger `build-ci-image` manually — runtime cache-bust is not enough because pip layers will reuse the old OSPREY commit.

---

## docker-compose orchestration

`scripts/deploy.sh` runs:
```
${config.runtime.compose_command} -f ${config.runtime.compose_files[0]} -f ${config.runtime.compose_files[1]} ... up -d
```

The compose files are scaffolded from `facility-config.yml` — see `references/scaffolding.md`. Image references in the generated compose use:
```
image: ${config.registry.url}/<service>:latest
```

So the link between CI and runtime is just: CI tags `:latest`, compose pulls `:latest`. Nothing more.

---

## Deploy script (`scripts/deploy.sh`) — CI/runtime contract

`deploy.sh` lives at `scripts/deploy.sh` and is generated by scaffolding. Its contract with CI is minimal:

```bash
#!/usr/bin/env bash
set -euo pipefail
set -a; . .env.production; set +a

# Login to the main registry
echo "${env.${config.gitlab.token_env_var}}" \
  | ${config.runtime.engine} login --username deploy --password-stdin "${config.registry.url%%/*}"

# IF MODULE registry.external_projects (any entries)
# For each external project, pull explicitly using its dedicated token
# FOR each in registry.external_projects
${config.runtime.engine} pull --creds=deploy:${env.${each.token_env_var}} ${each.url}/${each.image}
# END FOR
# END IF

# Pull all main-registry images and bring up
${config.runtime.compose_command} pull
${config.runtime.compose_command} ${runtime.compose_files[*] -f flags} up -d
```

CI's job is to make sure `${config.registry.url}/<service>:latest` always exists and is functional. The deploy server never sees CI internals.

---

## Common CI failures & fixes

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `docker login` fails with `denied: access forbidden` | `${config.gitlab.token_env_var}` lacks `read_registry`+`write_registry` scopes, or it's not protected/masked correctly | Recreate the PAT with `api`, `read_registry`, `write_registry`. Re-add to project CI/CD vars, mask + protect. |
| `osprey-build` fails with `command not found: osprey` | CI base image is stale — OSPREY moved on its mirror but base image still has old install | Manually trigger `build-ci-image`, or push a no-op change to `pyproject.toml`/`Dockerfile.ci` |
| `osprey-build` fails with `KeyError: '<env-var>'` | Required env var listed in prod profile not in CI variables | Add the variable at Settings → CI/CD → Variables, masked + protected |
| Custom MCP server build fails with `COPY failed: artifacts/foo.db not found` | The `osprey-build` stage didn't produce that artifact, or the custom server's `artifacts:` list doesn't match the actual build output paths | Check the `osprey-build` job logs; reconcile the paths in `modules.custom_mcp_servers.servers[*].artifacts` with what the build emits |
| Pipeline passes but `:latest` is unchanged after release | Release job ran on a non-default branch, or the job is waiting on manual trigger | Confirm pipeline ran on `${config.gitlab.default_branch}`; click play on the release job |
| `release` job fails with `denied` on push | PAT scope issue (same as login failure) | See first row |
| Web-terminal container has stale Claude project after deploy | The build pulled cached layers for `COPY artifacts/${config.facility.prefix}-assistant/`. Often happens when artifacts changed but Dockerfile didn't | Rebuild with `--no-cache` (add a CI knob, or trigger build-ci-image to invalidate) |
| External project image fails to pull on deploy server | External project's deploy token expired or was rotated | Regenerate the token in the external project, update the value in `.env.production` on the deploy server |
| Pipeline runs but no jobs appear | `.gitlab-ci.yml` parse error | Run `yamllint .gitlab-ci.yml` locally; check for tab/space mixing, missing `:` |
| Lint/test job fails on import the prod build doesn't | The CI base image is missing a dev dependency. Add to `pyproject.toml` under `[project.optional-dependencies] dev` and update Dockerfile.ci to install with `[dev]` extra |

When you see a CI failure not in this table, read the failing job's full log before changing anything. The pipeline is mostly idempotent — re-running a single job is cheap and often confirms whether the failure was transient.

---

## Updating the OSPREY pin

When OSPREY ships a new release with a feature this facility needs:

1. Push the new OSPREY version to the GitLab mirror (if your facility uses a mirror).
2. Bump the OSPREY pin in `pyproject.toml` (the version constraint or the git ref in the install URL).
3. Bump or annotate `docker/Dockerfile.ci` if the install line itself changed (e.g., new branch, new extras).
4. Push. Stage 1 (`build-ci-image`) auto-triggers and rebuilds the base.
5. Watch all subsequent stages — sometimes a new OSPREY changes generated config keys, causing downstream failures that surface in `osprey-build` or `docker-build-web-terminal`.

If runtime cache-bust isn't enough (no file changed but OSPREY advanced on the mirror), trigger `build-ci-image` manually.

---

## When CI templates can't express what you need

The `.gitlab-ci.yml` is fully scaffolded from `facility-config.yml`. If you need a new job shape (e.g., a custom test stage, a release notification, a security scan), the right move is:

1. Extend the template at `templates/core/gitlab-ci.yml` (and possibly `templates/modules/<name>/gitlab-ci.yml.fragment`).
2. Add the corresponding question to the deploy interview if it's user-configurable.
3. Add the corresponding field to `references/facility-config-schema.md`.

Hand-editing the generated `.gitlab-ci.yml` works once but loses the change next time scaffolding runs. Always feed structural changes back through templates.
