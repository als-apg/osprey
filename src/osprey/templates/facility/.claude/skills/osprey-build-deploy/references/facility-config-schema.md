# `facility-config.yml` — Schema Reference

`facility-config.yml` is the durable contract between this skill and the facility profile repo. It captures every site-specific value once, so generated files (`docker-compose.yml`, `.gitlab-ci.yml`, `scripts/deploy.sh`, `.env.template`) can derive everything from it without hardcoding.

**Treat this file like a Terraform state file:** version-controlled, source of truth, never lost. Re-running the deploy interview merges new answers into the existing file rather than overwriting it.

**Secrets do NOT live here.** API keys, deploy tokens, OLOG passwords go in `.env` (gitignored). This file references env var *names*, never values.

---

## Top-level structure

```yaml
schema_version: 1               # bump only when the schema changes incompatibly

facility: { ... }               # who you are
control_system: { ... }         # what control system the facility runs
gitlab: { ... }                 # CI/CD source
registry: { ... }               # container image destination
deploy: { ... }                 # the server everything runs on
runtime: { ... }                # container engine + compose flavor
network: { ... }                # proxy / no_proxy
llm: { ... }                    # which LLM provider feeds the assistant
ports: { ... }                  # MCP server + service port allocations
modules: { ... }                # opt-in features (event_dispatcher, web_terminals, olog, ...)
```

---

## `facility` — facility identity

```yaml
facility:
  name: "Advanced Light Source"          # full human-readable name
  prefix: "als"                           # short slug; used in profile filenames (als-prod.yml, als-client.yml)
                                          # and container names (als-mcp-matlab, als-web-thellert)
  timezone: "America/Los_Angeles"         # IANA timezone for log timestamps and schedules
```

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `name` | string | yes | Free text, shown in dashboards and Claude context |
| `prefix` | string (lowercase, alnum + hyphens, 2–6 chars) | yes | Drives generated filenames and container names; choose carefully — changing later requires renaming many files |
| `timezone` | IANA TZ | no | Default: `UTC` if omitted |

---

## `control_system` — control system type

```yaml
control_system:
  type: "epics"                           # epics | doocs | tango | mock | custom
  ca_addr_list: "10.0.0.1 10.0.0.2"       # EPICS only: broadcast addresses for CA discovery
  archiver_url: "http://arch.example.org:17668"  # optional, control-system-specific archiver REST URL
```

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `type` | enum | yes | OSPREY ships connectors for `epics` and `mock` today. `doocs`, `tango`, and `custom` are **roadmap values only** — selecting one writes the value into config but NO connector is built, so the resulting assistant has no live control-system access. Use `mock` for development on non-EPICS facilities until a real connector lands. Any value other than `epics` disables the EPICS test IOC module. |
| `ca_addr_list` | string | EPICS only | Used in compose files that need EPICS broadcast |
| `archiver_url` | URL | no | Used by integration tests and analytics agents |

When `type != "epics"`, the EPICS test IOC module is automatically unavailable regardless of `modules.test_ioc.enabled`.

---

## `gitlab` — GitLab project where source lives and CI runs

```yaml
gitlab:
  host: "git.als.lbl.gov"                 # GitLab server hostname (no scheme)
  remote_name: "gitlab"                   # `git remote` name for the GitLab origin
  default_branch: "main"                  # branch CI watches
  project_id: 951                         # numeric GitLab project ID (Settings → General)
  project_path: "physics/production/als-profiles"  # group/subgroup/project path
  token_env_var: "ALS_GITLAB_TOKEN"       # name of env var holding the PAT (NOT the value)
```

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `host` | hostname | yes | No `https://`, no path |
| `remote_name` | string | yes | The name of the git remote pointing to GitLab — typically `origin` or `gitlab` |
| `default_branch` | string | yes | Usually `main`; CI release job is restricted to this branch |
| `project_id` | int | yes | Find at GitLab project Settings → General → Project ID |
| `project_path` | string | yes | The URL path after the host: `<group>/<subgroup>/<project>` |
| `token_env_var` | string | yes | Name of the env var that holds the PAT/deploy token; the value lives in `.env` and is loaded by deploy.sh |

The token must have at minimum: `api` and `read_registry` scopes (`write_registry` if CI pushes images). Document this in the `.env.template`.

---

## `registry` — container image destination

```yaml
registry:
  url: "git.als.lbl.gov:5050/physics/production/als-profiles"   # full registry URL incl. port + path
  external_projects:                       # optional — separate registries with their own deploy tokens
    - name: "beam-viewer"
      url: "git.als.lbl.gov:5050/physics/production/beam-viewer"
      image: "beam-viewer:latest"
      token_env_var: "BEAM_VIEWER_DEPLOY_TOKEN"
```

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `url` | string | yes | Where CI pushes built images and deploy.sh pulls from. For GitLab projects, this is `<gitlab-host>:5050/<project_path>` |
| `external_projects` | list | no | Other GitLab projects whose images this deploy also pulls (e.g., a sibling team's service); each needs its own deploy token |

---

## `deploy` — the server everything runs on

```yaml
deploy:
  host: "appsdev2"                         # SSH-resolvable hostname
  fqdn: "appsdev2.als.lbl.gov"             # used by client-mode profiles to reach MCP services
  user: "thellert"                         # SSH user for deploys
  project_path: "/home/thellert/projects/als-profiles"  # absolute path on server
```

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `host` | string | yes | Must be in operator's `~/.ssh/config` so `ssh ${host}` works |
| `fqdn` | hostname | yes | Reachable from developers' laptops; used in client-mode profiles |
| `user` | string | yes | Owns the project dir; runs containers (rootless podman or docker group) |
| `project_path` | absolute path | yes | Where the facility profile repo is cloned on the server |

---

## `runtime` — container engine + compose flavor

```yaml
runtime:
  engine: "podman"                         # podman | docker
  compose_command: "podman-compose"        # podman-compose | docker compose | docker-compose
  compose_files:                           # ordered list passed to compose -f
    - "docker-compose.yml"
    - "docker-compose.host.yml"
```

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `engine` | enum | yes | Affects login/pull command syntax in deploy.sh |
| `compose_command` | string | yes | The actual command name on the deploy server |
| `compose_files` | list | yes | Order matters — later files override earlier ones |

---

## `network` — proxy + no_proxy

```yaml
network:
  http_proxy: "http://squid-ctrl.als.lbl.gov:3128"     # empty/null if no proxy needed
  https_proxy: "http://squid-ctrl.als.lbl.gov:3128"
  no_proxy:
    - "localhost"
    - "127.0.0.1"
    - "host.containers.internal"
    - "host.docker.internal"
    - "*.als.lbl.gov"
    # add internal services that must bypass the proxy
```

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `http_proxy` / `https_proxy` | URL or null | no | If null, no proxy lines are written into `.env.template` |
| `no_proxy` | list of strings | no | Hosts/patterns that bypass the proxy; both `NO_PROXY` and `no_proxy` env vars get set (different tools respect different cases) |

---

## `llm` — assistant's LLM provider

```yaml
llm:
  provider: "cborg"                        # cborg | anthropic | openai | google | ollama | asksage | vllm | argo | als-apg | stanford | amsc | other
  api_key_env_var: "CBORG_API_KEY"         # name of env var holding the key (NOT the value)
  model: "anthropic/claude-sonnet-4-20250514"  # default model id; profile YAMLs may override per agent
  base_url: null                           # optional override (e.g., for self-hosted ollama)
```

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `provider` | enum | yes | Must match a provider OSPREY supports (see `osprey/models/providers/`) |
| `api_key_env_var` | string | yes | Name only; value lives in `.env` |
| `model` | string | yes | Default model id; can be overridden per-agent in profile YAMLs |
| `base_url` | URL or null | no | Override for self-hosted endpoints (Ollama, vLLM, etc.) |

---

## `ports` — service port allocations

```yaml
ports:
  # Core MCP servers (only those that actually exist for the facility)
  matlab: 8001
  accelpapers: 8002
  phoebus: 8003
  integration_tests: 8004
  direct_channel_finder: 8005
  # Optional services — only present if the corresponding module is enabled
  event_dispatcher: 8010
  beam_viewer: 8007
  # Web terminals (port range start; one port per user)
  web_terminal_base: 9091
  web_terminal_nginx: 9080
```

Allocate ports the facility actually controls. Avoid the EPICS Channel Access defaults (5064/5065) and any ports occupied by other services on the deploy server. The interview asks the user to pick a base and adds offsets for multi-user services.

---

## `modules` — opt-in features

Each module is **off by default** (absent from `modules:` block, or `enabled: false`). When enabled, it has its own sub-config with module-specific values.

### `modules.event_dispatcher` — webhook + EPICS-CA → headless agent dispatch

```yaml
modules:
  event_dispatcher:
    enabled: true
    port: 8010                             # also referenced in ports.event_dispatcher
    token_env_var: "EVENT_DISPATCHER_TOKEN"
    sidecar_count: 5                       # one sidecar per web-terminal user (or per concurrent dispatch)
    sidecar_port_base: 9190                # sidecars on 9190, 9191, ... 9190+sidecar_count-1
    sidecar_token_env_var: "DISPATCH_SIDECAR_TOKEN"
    triggers_file: "triggers.yml"
    epics_ca:                              # only if control_system.type == epics
      enabled: true
      ca_addr_list: "10.0.0.1 10.0.0.2"
```

### `modules.web_terminals` — multi-user web terminal stack

```yaml
modules:
  web_terminals:
    enabled: true
    nginx_port: 9080                       # public-facing reverse proxy
    base_port: 9091                        # first per-user terminal port
    users:                                 # one container per user, named ${facility.prefix}-web-${user}
      - thellert
      - gmartino
      - scleemann
    landing_page_template: "default"       # default | custom (custom requires nginx/landing.html)
```

### `modules.olog` — electronic logbook integration

```yaml
modules:
  olog:
    enabled: true
    api_url: "https://controls.als.lbl.gov/olog/"
    test_url: "https://controls.als.lbl.gov/olog_test/rpc.php"  # optional
    auth_method: "basic"                   # basic | bearer | api_key
    username_env_var: "OLOG_USERNAME"
    password_env_var: "OLOG_PASSWORD"
    write_test_enabled: false              # set true to allow writes from integration tests
```

### `modules.ariel` — ARIEL DB (Postgres + embeddings)

```yaml
modules:
  ariel:
    enabled: true
    deployment: "container"                # container | external
    dsn: "postgresql://ariel:ariel@ariel-postgres:5432/ariel"  # container mode default
    sync_source: "olog"                    # olog | logbook | custom — must be a valid ARIEL adapter
    embeddings_provider: "ollama"          # references modules.ollama if set, or 'openai', 'cborg' etc.
```

In container mode, the DSN host MUST be the compose service key (`ariel-postgres` by default). Docker/podman DNS resolves that to the container IP inside the project network; any other hostname will fail to resolve.

### `modules.ollama` — local embedding / inference server

```yaml
modules:
  ollama:
    enabled: true
    url: "http://doudna.als.lbl.gov:11434"
    embedding_model: "nomic-embed-text"
    chat_model: null                       # optional — for local LLM calls
```

### `modules.wiki_search` — facility wiki (Confluence-flavored)

```yaml
modules:
  wiki_search:
    enabled: true
    type: "confluence"                     # confluence | mediawiki | custom
    base_url: "https://commons.lbl.gov"
    api_path: "/rest/api/"
    auth_method: "bearer"                  # bearer | basic
    token_env_var: "CONFLUENCE_ACCESS_TOKEN"
    spaces:                                # restrict search to specific spaces
      - "ALSAUFCONTROLS"
```

### `modules.shared_disk` — NFS or bind-mount for shared data

```yaml
modules:
  shared_disk:
    enabled: true
    host_path: "/home/als/physbase"        # path on the deploy server
    container_path: "/physbase"            # path inside containers that mount it
    mount_mode: "ro"                       # ro | rw
    services_to_mount: ["matlab", "integration_tests"]  # which compose services get the bind
```

### `modules.custom_mcp_servers` — facility-specific MCP servers

```yaml
modules:
  custom_mcp_servers:
    enabled: true
    servers:
      - name: "matlab"
        port: 8001                         # must match ports.matlab
        dockerfile: "docker/Dockerfile.matlab"
        build_context: "."
        artifacts:                         # build-time artifacts copied into image
          - "artifacts/mml.db"
        depends_on: []
      - name: "accelpapers"
        port: 8002
        dockerfile: "docker/Dockerfile.accelpapers"
        build_context: "."
        artifacts: []
        depends_on: ["typesense"]
      # ...
```

The skill renders compose entries and CI build jobs from this list. Each server gets its own Dockerfile path that the user owns.

### `modules.benchmarks` — e2e agent benchmark suite

```yaml
modules:
  benchmarks:
    enabled: true
    suite_path: "data/benchmarks/e2e_workflow_benchmarks.json"
    runs_in_container: "${facility.prefix}-web-${first_user}"   # which web-terminal container to exec into
    project_dir: "/app/${facility.prefix}-assistant/"
    judge_model: null                      # null = use llm.model
```

Requires `modules.web_terminals.enabled == true` (the suite runs inside a web terminal container).

### `modules.test_ioc` — EPICS test IOC management

```yaml
modules:
  test_ioc:
    enabled: true                          # only honored if control_system.type == "epics"
    cas_server_port: 59064                 # exotic non-standard port to isolate from production CA
    cas_beacon_port: 59065
    pv_prefix: "OSPREY:TEST:"              # all test PVs use this prefix
    db_path: "ioc/test.db"
    startup_script_path: "/tmp/start-test-ioc.sh"
```

**See `references/modules/test-ioc-safety.md`** for mandatory port-isolation rules. The test IOC will refuse to start if `cas_server_port` falls in the EPICS default range (5064–5065 or 5066–5076 commonly used by IOCs).

---

## Validation rules

When the interview writes or updates this file, validate:

1. **Required core blocks present**: `facility`, `control_system`, `gitlab`, `registry`, `deploy`, `runtime`, `llm`, `ports`.
2. **`facility.prefix` is lowercase, 2–6 chars, alphanumeric + hyphens**. No underscores, no uppercase (container name compatibility).
3. **`gitlab.host` does not include `https://` or trailing path**.
4. **`registry.url` includes the port** (typically `:5050` for GitLab).
5. **`deploy.host` is reachable** by `ssh -o BatchMode=yes ${host} true` (warn, don't fail — operator may not have keys yet).
6. **`network.no_proxy` includes `localhost` and `127.0.0.1`** (warn if missing).
7. **`ports.*` values are unique** (no two services on the same port).
8. **`modules.test_ioc.cas_server_port` is outside 5064–5076** if test_ioc is enabled.
9. **`modules.benchmarks` requires `modules.web_terminals.enabled`**.
10. **`modules.event_dispatcher.epics_ca.enabled` requires `control_system.type == "epics"`**.
11. **Port range overlap** — with `web_terminals` base `B_w` and N users binding `[B_w, B_w+N-1]`, and `event_dispatcher` base `B_d` and M sidecars binding `[B_d, B_d+M-1]`, the two ranges MUST NOT overlap. Reject configs where either range contains a value in the other.
12. **Custom MCP server names must not collide with reserved service keys** — `ariel-postgres`, `typesense`, `event-dispatcher`, `integration-tests`, `dispatch-sidecar-*`, `ariel-sync`, `nginx`. Custom names are rendered as `${prefix}-mcp-${name}` so they don't collide at the compose level, but bare references to reserved names in `depends_on` or `services_to_mount` are reserved for the built-ins.

If validation fails, do not silently overwrite — surface the error and ask the user to confirm the fix.

---

## Migration (schema_version bumps)

When this skill ships a new schema version with breaking changes, the interview includes a migration step:

1. Read the existing `facility-config.yml`.
2. Apply field renames / restructures based on the version delta.
3. Ask the user to confirm the migration result.
4. Write the migrated file with the new `schema_version`.

Never silently mutate `facility-config.yml` — always show the diff and get confirmation.
