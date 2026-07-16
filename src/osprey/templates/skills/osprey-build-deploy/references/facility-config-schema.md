# `facility-config.yml` — Schema Reference

`facility-config.yml` is the durable contract between this skill and the facility profile repo. It captures every site-specific value once, so generated files (`docker-compose.yml`, `.gitlab-ci.yml`, `.env.template`) and `osprey deploy` itself (which reads `config.yml`, derived from this file) can derive everything from it without hardcoding.

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
  timezone: "America/Los_Angeles"         # facility timezone — drives container TZ and the agent's system.timezone
```

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `name` | string | yes | Free text, shown in dashboards and Claude context |
| `prefix` | string (lowercase, alnum + hyphens, 2–6 chars) | yes | Drives generated filenames and container names; choose carefully — changing later requires renaming many files |
| `timezone` | IANA TZ | no | The facility timezone. Drives container `TZ`; mirror it into the profile's `system.timezone`. Default: `UTC` if omitted |

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
| `token_env_var` | string | yes | Name of the env var that holds the PAT/deploy token; the value lives in `.env`, read via `osprey deploy`'s compose `--env-file` and by the operator's manual registry login before it |

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
| `url` | string | yes | Where CI pushes built images and `osprey deploy up` pulls from. For GitLab projects, this is `<gitlab-host>:5050/<project_path>` |
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
| `engine` | enum | yes | Affects login/pull command syntax for the operator's registry login and for `osprey deploy` |
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
  provider: "cborg"                        # cborg | anthropic | openai | google | ollama | asksage | vllm | argo | als-apg | stanford | amsc-i2 | other
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
  # Web terminals — nginx is a single port and is mirrored here. The four
  # per-user port RANGES are not single values, so they are not mirrored here;
  # they live authoritatively in modules.web_terminals.*_base_port (see below).
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

Each user gets **four** independent per-user ports — one per service family — so a
single user's terminal, artifact gallery, ARIEL search, and lattice dashboard never
collide with each other or with any other user's. For a user at position `i`
(0-indexed, per `users[]` order), the per-family host port is `<family>_base_port +
i`. This arithmetic is implemented once in `deployment/web_terminals/ports.py`
(`allocate_ports`) and consumed by the renderer and the lint — do not reimplement it.

```yaml
modules:
  web_terminals:
    enabled: true
    nginx_port: 9080                       # public-facing reverse proxy / landing page
    web_base_port: 9091                    # first per-user web-terminal port      → OSPREY_WEB_PORT
    artifact_base_port: 9291               # first per-user artifact-gallery port  → OSPREY_ARTIFACT_SERVER_PORT
    ariel_base_port: 9391                  # first per-user ARIEL search port      → OSPREY_ARIEL_PORT
    lattice_base_port: 9491                # first per-user lattice-dashboard port → OSPREY_LATTICE_DASHBOARD_PORT
    users:                                 # one container per user, named ${facility.prefix}-web-${user}
      - thellert
      - gmartino
      - scleemann
    landing:                               # grouped landing page served at nginx_port
      groups:                              # ordered list; rendered top to bottom
        - type: "users"                    # auto-populated: one card per entry in `users` above
        - type: "links"                    # a static group of operator-supplied links
          label: "Facility Tools"
          links:
            - label: "Elog"
              url: "https://elog.example.org"
            - label: "Status Page"
              url: "https://status.example.org"
    auth:                                 # OPTIONAL — forward-looking, config-gated seam; INERT (see note below)
      method: "none"                      # none (default); no other value is exercised in this schema revision
    tls:                                  # OPTIONAL — forward-looking, config-gated seam; INERT (see note below)
      enabled: false                      # default; v1 ships plain HTTP only
      cert: "/etc/osprey/tls/facility.crt"  # only read when enabled: true
      key: "/etc/osprey/tls/facility.key"   # only read when enabled: true
```

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `enabled` | bool | yes | Off by default |
| `nginx_port` | int | yes | Reverse proxy + landing page port; must be unique across `ports.*` |
| `web_base_port` | int | yes | First per-user web-terminal port; binds `OSPREY_WEB_PORT` in the per-user container |
| `artifact_base_port` | int | yes | First per-user artifact-gallery port; binds `OSPREY_ARTIFACT_SERVER_PORT` |
| `ariel_base_port` | int | yes | First per-user ARIEL search port; binds `OSPREY_ARIEL_PORT` |
| `lattice_base_port` | int | yes | First per-user lattice-dashboard port; binds `OSPREY_LATTICE_DASHBOARD_PORT` |
| `users` | list of strings | yes | May be empty when `enabled: true` (see validation rule below) |
| `landing.groups` | list of group objects | no | Defaults to a single `type: "users"` group if omitted. `type: "users"` groups take no other fields and auto-populate from `users[]`; `type: "links"` groups require `label` and a `links` list of `{label, url}` objects |
| `auth.method` | string | no | Defaults to `"none"` (no authentication). Forward-looking seam only — see note below |
| `tls.enabled` | bool | no | Defaults to `false` (plain HTTP). Forward-looking seam only — see note below |
| `tls.cert` | string | required if `tls.enabled: true` | Path to the TLS certificate file, mounted into the nginx container |
| `tls.key` | string | required if `tls.enabled: true` | Path to the TLS private key file, mounted into the nginx container |

Every port-valued field in the table above must be free of collisions with every
other port allocation in the config: `nginx_port` against `ports.*` (its mirror is
`ports.web_terminal_nginx`), and each `*_base_port` against every other port
allocation for its whole per-family range `[base, base+len(users)-1]`, not just the
base value itself. The four `*_base_port` fields are deliberately **not** mirrored
into `ports.*` (see the `ports` section above) — they are ranges, not single ports.
See validation rule 11 for the full closed set of checks.

The field names above (`web_base_port`, `artifact_base_port`, `ariel_base_port`,
`lattice_base_port`) are backend-neutral — they name the four service families, not
any specific container runtime, so a future non-container backend can reuse them
unchanged.

The `web` family's internal container port — the port the `osprey web` process
itself listens on inside the per-user container, before any per-user remapping —
is a **fixed constant, 8087** (the `osprey web` default; see `cli/web_cmd.py`) and
is **not configurable** via this schema; only the externally-published,
per-user `web_base_port + i` host port varies. Templates and docs must never
reintroduce the old `9087` value.

> **`auth`/`tls` are forward-looking, config-gated seams — not an implemented
> security feature.** With the defaults shown (`auth.method: none`,
> `tls.enabled: false`), the rendered stack is functionally equivalent to Phase 1
> (minus the port→path relocation): perimeter-trust, plain HTTP, open to anyone who
> reaches the deploy host. OSPREY does not ship an auth backend or provision
> certificates for you — these stanzas only exist so that a future phase can wire
> a real `auth_request` backend and TLS termination as a config flip instead of a
> template rewrite. The multi-user-support epic's two CRITICAL findings, **C1 (no
> authentication)** and **C2 (all traffic is cleartext HTTP)**, remain **OPEN and
> explicitly deferred** — defining this schema does not close them, and no
> deployment should be treated as authenticated or encrypted on the strength of
> these fields existing.

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

> **Agent-facing MCP declaration (`claude_code.servers`)** — the block above builds/serves the containers; what the *agent* sees is declared in the built project's `config.yml` under `claude_code.servers.<name>` (rendered into `.mcp.json` / `settings.json` at build/regen). A custom server needs `command`+`args` or `url`, plus optional `permissions.allow/ask` and `hooks.pre_tool_use` presets. A **second instance of a framework server** is declared with `extends` instead:
>
> ```yaml
> claude_code:
>   servers:
>     phoebus2:
>       extends: phoebus     # clone the framework server under a new name
>       env:
>         PHOEBUS_BRIDGE_URL: "${PHOEBUS2_BRIDGE_URL:-http://127.0.0.1:7980}"
> ```
>
> The clone inherits the template's permissions and per-tool hooks with `mcp__<template>__` matchers rewritten to `mcp__<name>__`; spec `env` keys override the template's; `permissions.allow/ask` overrides may add but never remove the template's approval-gated (`ask`) tools. `extends` is only expressible via `config:` dotted overrides in a build profile (e.g. `claude_code.servers.phoebus2.extends: phoebus`) or directly in `config.yml` — the build-profile `mcp_servers:` block requires `command` or `url` and cannot express it. Note: approval policies key on the bare tool name, so they apply to every instance of a template (no per-instance gating).

> **The `phoebus` framework server** — native interaction with a running [Phoebus](https://control-system-studio.readthedocs.io/) control panel (perceive the widget tree, snapshot widgets, drive controls via the in-JVM agent bridge). Off by default; a facility enables and configures it entirely through `config.yml` — no code changes:
>
> ```yaml
> claude_code:
>   servers:
>     phoebus: { enabled: true }         # expose mcp__phoebus__* (drive is approval-gated)
> phoebus:
>   host: "127.0.0.1"                    # agent-bridge host (or set PHOEBUS_BRIDGE_URL)
>   port: 7979                           # agent-bridge port
>   require_handle: true                 # on a shared backend, reject the implicit "active" display
>   archiver_url: "pbraw://arch.example.org/retrieve"  # optional; backs phoebus_open_databrowser
>   panels:
>     site_overview: /path/to/site.bob   # register a logical panel name → .bob file
> ```
>
> The bridge itself lives in the Phoebus product (a facility build), not in OSPREY. Every value resolves from config or env with `127.0.0.1` defaults — nothing facility-specific is baked into the framework.

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
11. **Port range overlap** — a closed rule over four named sets; reject a config if any value appears in more than one of them, or if any two ranges in `S1 ∪ S2` overlap:
    - **S1 (web_terminals families)** — for N = `len(users)`, one range per family: `[base_f, base_f+N-1]` for each `base_f` in `{web_base_port, artifact_base_port, ariel_base_port, lattice_base_port}`.
    - **S2 (event_dispatcher sidecars)** — `[sidecar_port_base, sidecar_port_base+sidecar_count-1]`.
    - **S3 (`ports.*` literals)** — every value in the top-level `ports.*` map. `modules.web_terminals.nginx_port` and `modules.event_dispatcher.port` are each required to equal their `ports.*` mirror (`ports.web_terminal_nginx` / `ports.event_dispatcher`), so checking `ports.*` once already covers both — do not check the module field a second time, or every valid config would (correctly) show the same value twice and falsely look like a collision.
    - **S4 (module ports with no `ports.*` mirror)** — `modules.test_ioc.cas_server_port` and `modules.test_ioc.cas_beacon_port`. (`modules.custom_mcp_servers.servers[].port` values are required to already equal an existing `ports.*` entry — see the per-server comment convention — so they're already covered by S3 and need no separate check. The four `web_terminals` family base ports are likewise NOT mirrored into `ports.*` — see the `ports` section above — precisely so they don't need de-duplicating against S1.)

    Concretely: take `S1 ∪ S2 ∪ S3 ∪ S4` as a value multiset (ranges expanded to their member ports) and flag any value with multiplicity > 1. This set is closed and enumerable — these four sets are the complete input, nothing further needs discovering from the config.
12. **Reserved service names** — no entry in `modules.web_terminals.users`, and no `custom_mcp_servers` server name, may collide with a reserved service key: `nginx`, `ariel-postgres`, `typesense`, `event-dispatcher`, `integration-tests`, `dispatch-sidecar-*`, `ariel-sync`. A username becomes part of a compose service name (`${facility.prefix}-web-${user}`), and the web-terminals reverse proxy is always named `nginx` — a user literally named `nginx` (or any other reserved key) would collide with a built-in service. Custom MCP server names are rendered as `${prefix}-mcp-${name}` so they don't collide at the compose level, but bare references to reserved names in `depends_on` or `services_to_mount` are reserved for the built-ins.
13. **`modules.web_terminals` with an empty `users[]`** — if `enabled: true` and `users` is `[]`:
    - If `modules.benchmarks.enabled` is `false` (or `modules.benchmarks` is absent), this is a **warning, not a failure**: the module still renders nginx + the landing page (with the `users` landing group rendering empty), just with zero per-user terminal services. Warn and continue; do not reject the config.
    - If `modules.benchmarks.enabled` is `true`, this **is a failure**: rule 9 requires `web_terminals.enabled` for benchmarks, but `modules.benchmarks.runs_in_container` resolves to a specific user's container (e.g. `${facility.prefix}-web-${first_user}`) and there is no user to resolve `first_user` to. Reject the config in this combination.

If validation fails, do not silently overwrite — surface the error and ask the user to confirm the fix.

---

## Migration (schema_version bumps)

When this skill ships a new schema version with breaking changes, the interview includes a migration step:

1. Read the existing `facility-config.yml`.
2. Apply field renames / restructures based on the version delta.
3. Ask the user to confirm the migration result.
4. Write the migrated file with the new `schema_version`.

Never silently mutate `facility-config.yml` — always show the diff and get confirmation.
